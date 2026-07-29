#!/usr/bin/env python3
"""Full pySCF FDBM job bench: setup + SCF + GPU ρ_proj (isolated process per row).

Compares post-setup-optimization profiles vs the 2026-07-28 SPAMMM snapshot
(setup 7–10 s, ρ_proj ~18 s). See:
  /home/prokop/git/pyscf/doc/reports/2026-07-SCF_setup_breakdown.md

Usage:
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \\
    python tests/SPM/run_bench_pyscf_full_job.py

  # one config only (used by the driver as a subprocess):
  python tests/SPM/run_bench_pyscf_full_job.py --worker \\
    --mol pentacene --profile production_radial_screened_splitk --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
PYSCF = os.environ.get('SPAMMM_PYSCF_ROOT', '/home/prokop/git/pyscf')
OUTDIR_DEFAULT = os.path.join(ROOT, 'debug', 'pyscf_full_job_bench')

# Profiles to time (spammm default + fast DF variants from setup breakdown)
PROFILES = [
    'production_radial_screened_splitk',  # spammm DEFAULT_GPU_PROFILE
    'fast_full_gpu',                      # OTF + full GPU DF
    'fast_full_gpu_compact',              # OTF + compact GPU DF (A2-class)
]
MOLS = ['pentacene', 'PTCDA']


def _worker(mol_name, profile, out_json, *,
            step_A=0.1, margin_A=5.0, z_extra_A=6.0, n_threads=8,
            max_cycle=40, skip_rho=False):
    """One isolated process: setup → SCF → GTO ρ_proj. Write JSON."""
    if PYSCF not in sys.path:
        sys.path.insert(0, PYSCF)
    os.environ.setdefault('OMP_NUM_THREADS', str(n_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    import numpy as np
    from pyscf import gto, dft, lib
    from pyscf.data import nist
    from pyscf.OpenCL.gpu_profiles import apply_gpu_profile, get_profile
    from pyscf.OpenCL.grid_gto import GTOGridProjector, make_grid_spec

    lib.num_threads(int(n_threads))
    ang = float(nist.BOHR)
    xyz = os.path.join(PYSCF, 'data', 'xyz', f'{mol_name}.xyz')
    if not os.path.isfile(xyz):
        xyz = os.path.join(ROOT, 'data', 'xyz', f'{mol_name}.xyz')

    def read_xyz(path):
        lines = open(path).read().splitlines()
        atoms = []
        for line in lines[2:]:
            p = line.split()
            if len(p) >= 4 and p[0][0].isalpha():
                atoms.append(f'{p[0]} {float(p[1]):.8f} {float(p[2]):.8f} {float(p[3]):.8f}')
        return '\n'.join(atoms)

    t_all0 = time.perf_counter()
    mol = gto.M(atom=read_xyz(xyz), basis='def2-SVP', unit='Angstrom', verbose=0)
    mol.max_memory = 16000
    mf = dft.RKS(mol, xc='PBE').density_fit()
    mf.max_memory = 16000
    mf.with_df.max_memory = 16000
    mf.grids.level = 3
    mf.max_cycle = int(max_cycle)

    # --- setup (Clock A): apply_gpu_profile = grids + XC plan + DF ---
    # Stage split mirrors 2026-07-SCF_setup_breakdown.md
    from pyscf.OpenCL import init_device
    init_device(quiet=True)
    prof = get_profile(profile)

    t0 = time.perf_counter()
    # replicate apply_gpu_profile stages for timers
    mf.backend = prof['mf_backend']
    from pyscf.OpenCL.gpu_profiles import apply_scf_kw, prepare_df_for_scf, _ensure_splitk_tile_config
    apply_scf_kw(mf, prof.get('scf_kw', {}))
    df_backend = prof.get('df_backend')
    mode = prof.get('df_build_mode')
    if mode is not None:
        from pyscf.OpenCL.df_jk import resolve_df_build_mode
        mode = resolve_df_build_mode(mode)
    if mode == 'compact_gpu':
        df_backend = 2 if df_backend is None else (int(df_backend) | 2)
    if df_backend is not None and mf.with_df is not None:
        mf.with_df.backend = df_backend
    mf.with_df.storage = 'incore'

    t_g0 = time.perf_counter()
    mf.initialize_grids(mol)
    t_grids = time.perf_counter() - t_g0

    t_x0 = time.perf_counter()
    xc_path = prof.get('xc_path')
    setup_kw = dict(prof.get('setup_kw', {}))
    gpu_xc = setup_kw.pop('gpu_xc', 'auto')
    _ensure_splitk_tile_config(setup_kw)
    if xc_path == 'precomputed':
        from pyscf.OpenCL.xc_grid import setup_precomputed_gto
        mf._xc_gpu_plan = setup_precomputed_gto(
            mol, mf.grids, mf.xc, gpu_only=True, gpu_xc=gpu_xc, **setup_kw)
        mf._gpu_xc_path = 'precomputed'
    elif xc_path == 'onthefly':
        from pyscf.OpenCL.xc_grid import setup_xc_grid_gpu
        mf._xc_gpu_plan = setup_xc_grid_gpu(
            mol, mf.grids, mf.xc, gpu_xc=gpu_xc, **setup_kw)
        mf._gpu_xc_path = 'onthefly'
    else:
        raise ValueError(xc_path)
    t_xc = time.perf_counter() - t_x0

    t_d0 = time.perf_counter()
    prepare_df_for_scf(mf, storage='incore', require_incore=True, build_mode=mode)
    t_df = time.perf_counter() - t_d0
    mf._gpu_profile_name = profile
    t_setup = time.perf_counter() - t0

    # --- SCF (Clock B total) ---
    t0 = time.perf_counter()
    e = mf.kernel()
    t_scf = time.perf_counter() - t0
    n_cyc = int(getattr(mf, 'cycles', 0) or 0)
    dm = mf.make_rdm1()
    conv = bool(mf.converged)

    # --- ρ_proj on FDBM box (GPU GTO) ---
    t_rho = None
    q_rho = None
    npts = None
    if not skip_rho:
        coords_A = mol.atom_coords() * ang
        lo = coords_A.min(0) - margin_A
        hi = coords_A.max(0) + margin_A
        hi[2] += z_extra_A
        step_B = step_A / ang
        origin_B = lo / ang
        ngrid = np.ceil((hi - lo) / step_A).astype(np.int32)
        ngrid = ((ngrid + 7) // 8) * 8
        grid_spec = make_grid_spec(origin_B, step_B, ngrid)
        npts = int(np.prod(ngrid))
        proj = GTOGridProjector(mol, verbosity=0)
        proj.project_multizeta(dm, grid_spec)  # warmup (compile amortize)
        t0 = time.perf_counter()
        rho, info = proj.project_multizeta(dm, grid_spec)
        t_rho = time.perf_counter() - t0
        q_rho = float(rho.sum() * (step_B ** 3))

    t_all = time.perf_counter() - t_all0
    row = {
        'mol': mol_name,
        'profile': profile,
        'nao': int(mol.nao_nr()),
        'natm': int(mol.natm),
        'Ne': int(mol.nelectron),
        'n_threads': int(n_threads),
        'setup_s': t_setup,
        'grids_s': t_grids,
        'xc_setup_s': t_xc,
        'df_s': t_df,
        'scf_s': t_scf,
        'cycles': n_cyc,
        'scf_per_cyc_s': (t_scf / n_cyc) if n_cyc else None,
        'converged': conv,
        'E_Ha': float(e),
        'rho_proj_s': t_rho,
        'rho_q': q_rho,
        'rho_npts': npts,
        'total_s': t_all,
        'df_build_mode': getattr(mf, '_df_build_mode', mode),
        'df_kind': getattr(mf, '_df_storage_kind', None),
    }
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(row, f, indent=2)
    # also print one-liner for live log
    print(json.dumps(row), flush=True)
    return 0


def _run_isolated(mol, profile, outdir, n_threads=8):
    out_json = os.path.join(outdir, f'{mol}__{profile}.json')
    cmd = [
        sys.executable, os.path.abspath(__file__), '--worker',
        '--mol', mol, '--profile', profile, '--json', out_json,
        '--threads', str(n_threads),
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = PYSCF + os.pathsep + env.get('PYTHONPATH', '')
    env['OMP_NUM_THREADS'] = str(n_threads)
    env['OPENBLAS_NUM_THREADS'] = '1'
    print(f'\n=== ISOLATED {mol} / {profile} ===', flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    log_path = os.path.join(outdir, f'{mol}__{profile}.log')
    open(log_path, 'w').write(proc.stdout)
    print(proc.stdout, end='' if proc.stdout.endswith('\n') else '\n', flush=True)
    if proc.returncode != 0:
        print(f'FAILED rc={proc.returncode}  REVIEW: {log_path}', flush=True)
        return None
    row = json.load(open(out_json))
    row['subprocess_wall_s'] = wall
    print(f'  setup={row["setup_s"]:.2f}s (grids={row["grids_s"]:.2f} xc={row["xc_setup_s"]:.2f} df={row["df_s"]:.2f})  '
          f'SCF={row["scf_s"]:.2f}s/{row["cycles"]}c  ρ={row["rho_proj_s"]:.3f}s  '
          f'total={row["total_s"]:.2f}s  conv={row["converged"]}  E={row["E_Ha"]:.6f}', flush=True)
    print(f'REVIEW: {out_json}', flush=True)
    return row


def _dftb_row(mol_name, outdir, step_A=0.1, margin_A=5.0, z_extra_A=6.0):
    """Quick DFTB 3ob reference for product comparison (same FDBM box intent)."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import numpy as np
    import spammm.atomicUtils as au
    from spammm.config_utils import get_dftb_basis_path
    from spammm.SPM import AFM_utils as U

    xyz = os.path.join(ROOT, 'data', 'xyz', f'{mol_name}.xyz')
    pos, _, names, _, _ = au.load_xyz(xyz)
    atomPos = np.asarray(pos, dtype=np.float64)
    ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in names], dtype=np.int32)
    grid_spec, origin, ngrid, step = U._make_grid_spec(atomPos, step_A, margin_A, z_extra_A)
    # pad to 8 for fairness with GTO path size (DFTB doesn't need it)
    ngrid = tuple(int(x) for x in ngrid)
    basis_hsd = get_dftb_basis_path('3ob-3-1')
    work = os.path.join(outdir, f'dftb_{mol_name}')
    t0 = time.perf_counter()
    res = U.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, work, grid_spec=grid_spec, step=step, verbosity=0)
    wall = time.perf_counter() - t0
    dV = step ** 3
    q = float(np.asarray(res['rho_scf']).sum() * dV)
    row = {
        'mol': mol_name, 'profile': 'DFTB_3ob',
        'setup_s': None, 'scf_s': wall, 'cycles': None, 'rho_proj_s': None,
        'total_s': wall, 'converged': True, 'E_Ha': None, 'rho_q': q,
        'note': 'DFTB SCF+ρ folded into total (no separate clocks)',
    }
    path = os.path.join(outdir, f'{mol_name}__DFTB_3ob.json')
    json.dump(row, open(path, 'w'), indent=2)
    print(f'DFTB {mol_name}: total={wall:.3f}s  q={q:.2f}  REVIEW: {path}', flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--worker', action='store_true', help='single-config worker (do not call directly)')
    ap.add_argument('--mol', default='pentacene')
    ap.add_argument('--profile', default='production_radial_screened_splitk')
    ap.add_argument('--json', default=None)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--outdir', default=OUTDIR_DEFAULT)
    ap.add_argument('--molecule', nargs='*', default=None)
    ap.add_argument('--profiles', nargs='*', default=None)
    ap.add_argument('--skip-dftb', action='store_true')
    args = ap.parse_args(argv)

    if args.worker:
        return _worker(args.mol, args.profile, args.json, n_threads=args.threads)

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    mols = args.molecule or MOLS
    profiles = args.profiles or PROFILES
    rows = []

    for mol in mols:
        if not args.skip_dftb:
            try:
                rows.append(_dftb_row(mol, outdir))
            except Exception as e:
                print(f'DFTB {mol} failed: {e}', flush=True)
        for prof in profiles:
            r = _run_isolated(mol, prof, outdir, n_threads=args.threads)
            if r:
                rows.append(r)

    # SUMMARY table
    summary = os.path.join(outdir, 'SUMMARY.out')
    lines = [
        'Full pySCF FDBM job bench (isolated process per GPU row)',
        f'Host: RTX 3090  OMP={args.threads}  basis=def2-SVP  xc=PBE  DF=incore',
        f'Date: {time.strftime("%Y-%m-%d")}',
        '',
        'Baseline 2026-07-28 (spammm, production_radial_screened_splitk ≈):',
        '  pentacene GPU: setup=6.91  SCF=1.73/12  ρ=17.52  total≈26',
        '  PTCDA     GPU: setup=10.11 SCF=3.88/19  ρ=24.38  total≈38',
        '',
        f'{"mol":<10} {"profile":<36} {"setup":>6} {"grids":>5} {"xc":>5} {"df":>5} '
        f'{"SCF":>6} {"cyc":>3} {"/cyc":>5} {"ρ":>6} {"tot":>6} {"conv":>5} {"E":>14}',
    ]
    for r in rows:
        if r.get('profile') == 'DFTB_3ob':
            lines.append(
                f'{r["mol"]:<10} {"DFTB_3ob":<36} {"—":>6} {"—":>5} {"—":>5} {"—":>5} '
                f'{r["total_s"]:>6.2f} {"—":>3} {"—":>5} {"—":>6} {r["total_s"]:>6.2f} {"True":>5} {"—":>14}')
            continue
        lines.append(
            f'{r["mol"]:<10} {r["profile"]:<36} {r["setup_s"]:>6.2f} {r["grids_s"]:>5.2f} '
            f'{r["xc_setup_s"]:>5.2f} {r["df_s"]:>5.2f} {r["scf_s"]:>6.2f} {r["cycles"]:>3} '
            f'{(r["scf_per_cyc_s"] or 0):>5.2f} {(r["rho_proj_s"] or 0):>6.3f} {r["total_s"]:>6.2f} '
            f'{str(r["converged"]):>5} {r["E_Ha"]:>14.6f}')
    lines += [
        '',
        'Baseline ratios (setup): new / 2026-07-28',
    ]
    base = {'pentacene': 6.91, 'PTCDA': 10.11}
    for r in rows:
        if r.get('profile') == 'production_radial_screened_splitk' and r['mol'] in base:
            lines.append(f'  {r["mol"]} setup {r["setup_s"]:.2f} / {base[r["mol"]]:.2f} = '
                         f'{base[r["mol"]]/r["setup_s"]:.2f}× faster')
    text = '\n'.join(lines) + '\n'
    open(summary, 'w').write(text)
    print('\n' + text)
    print(f'REVIEW: {summary}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
