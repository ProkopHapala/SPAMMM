#!/usr/bin/env python3
"""Verify fork GTOGridProjector: timing/parity already in pyscf test; here AFM images.

Pipeline (pentacene / PTCDA):
  1. Load Fukui PBE/def2-SVP cube + geometry
  2. GPU RKS SCF on cube atoms → DM
  3. Project ρ with GTOGridProjector (multizeta) onto FDBM AFM grid
  4. Sample parity vs numint; compare ∫ρ / MAE vs remeshed cube
  5. FDBM AFM strip: cube | gto_gpu | prolonged  (same Pauli A,β; gto uses cube ES)

Output: debug/gto_grid_afm_verify/<mol>/

Usage:
  python tests/SPM/run_verify_gto_grid_afm.py --molecule pentacene
  python tests/SPM/run_verify_gto_grid_afm.py --molecule pentacene PTCDA
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUTDIR_DEFAULT = os.path.join(ROOT, 'debug', 'gto_grid_afm_verify')


def _import_pyscf_utils():
    import importlib.util
    path = os.path.join(ROOT, 'spammm', 'quantum', 'pySCF_utils-new.py')
    spec = importlib.util.spec_from_file_location('pySCF_utils_new', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _afm_grid_to_bohr(origin_A, step_A, ngrid):
    """AFM Å lattice → Bohr grid_spec for GTOGridProjector (nist.BOHR = Å/a0)."""
    from pyscf.data import nist
    from pyscf.OpenCL.grid_gto import make_grid_spec
    ang = float(nist.BOHR)  # Å per Bohr ≈ 0.529
    ngrid = np.asarray(ngrid, dtype=np.int32).reshape(3)
    ngrid8 = ((ngrid + 7) // 8) * 8
    origin_B = np.asarray(origin_A, dtype=np.float64) / ang
    step_B = float(step_A) / ang
    return make_grid_spec(origin_B, step_B, ngrid8), ngrid8, ang


def run_one(mol_name, outdir_root, *, step=0.1, margin=4.0, z_vac=6.0,
            h_min=3.7, h_max=4.7, h_step=0.1, amp=1.0, parity_npts=2000):
    import spammm.atomicUtils as au
    import spammm.SPM.AFM as afm
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import (
        parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list,
    )
    from spammm.SPM import AFM_utils as U

    pyscf_u = _import_pyscf_utils()
    pyscf_u.ensure_local_pyscf()
    from pyscf.OpenCL.grid_gto import GTOGridProjector, grid_coords, rho_numint_ref

    outdir = os.path.join(outdir_root, mol_name)
    os.makedirs(outdir, exist_ok=True)
    lines = [f'GTO grid AFM verify: {mol_name}', '']

    cube_dir = U.fukui_cube_dir(mol_name)
    rho_cube_path = os.path.join(cube_dir, 'rho_N.cube')
    if not os.path.isfile(rho_cube_path):
        raise FileNotFoundError(rho_cube_path)

    # Prefer XYZ from FUKUI_PANEL if listed; else cube atoms only
    xyz_rel = dict(U.FUKUI_PANEL).get(mol_name)
    d_cube = U.get_density_from_cube(rho_cube_path, use_esp_cube=False, verbosity=0)
    atomPos = np.asarray(d_cube['atomPos'], dtype=np.float64)
    atomZ = np.asarray(d_cube['atomZ'], dtype=np.float64)
    atomTypes = np.array([int(round(z)) for z in atomZ], dtype=np.int32)
    inv_z = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
    enames = [inv_z.get(int(z), 'C') for z in atomTypes]

    grid_spec_A, origin, ngrid, step = U.make_fdbm_grid_com_zsym(
        atomPos, step, margin, z_vac=float(z_vac))
    # Ensure 8-aligned for GTO projector (may grow nz slightly)
    grid_bohr, ngrid8, ang = _afm_grid_to_bohr(origin, step, ngrid)
    if tuple(ngrid8) != tuple(int(x) for x in ngrid):
        ngrid = tuple(int(x) for x in ngrid8)
        grid_spec_A = {
            'origin': np.asarray(origin, dtype=np.float64),
            'dA': np.array([step, 0.0, 0.0]),
            'dB': np.array([0.0, step, 0.0]),
            'dC': np.array([0.0, 0.0, step]),
            'ngrid': np.array(ngrid, dtype=np.int32),
        }
        print(f'  padded ngrid → {ngrid} for 8³ tiles')
    nx, ny, nz = ngrid
    dV = float(step) ** 3
    lines.append(f'cube={cube_dir}')
    lines.append(f'grid={nx}x{ny}x{nz} step={step} origin={origin}')
    print(f'\n######## {mol_name}  grid={nx}x{ny}x{nz} ########')

    # --- cube remesh ---
    cube_prep = U.allelectron_cube_to_fdbm_grid(
        d_cube['rho_scf'], d_cube['origin'], d_cube['step'], atomPos, atomZ,
        origin, step, ngrid, rc_na=0.6, R_sphere=0.6, verbosity=0)
    rho_cube = cube_prep['rho_scf']
    rho_diff_cube = cube_prep['rho_diff']
    q_cube = float(rho_cube.sum() * dV)
    lines.append(f'cube remesh q_scf={q_cube:.3f} q_diff={cube_prep["q_diff"]:.3e}')

    # --- GPU SCF ---
    backend = pyscf_u.resolve_backend('gpu')
    print(f'[{mol_name}] SCF PBE/def2-SVP backend={backend} …')
    t0 = time.perf_counter()
    scf_out = pyscf_u.run_scf_geometry(
        enames, atomPos, basis='def2-SVP', xc='PBE', backend=backend,
        n_threads=4, release=False)
    t_scf = time.perf_counter() - t0
    mol = scf_out['mol']
    dm = scf_out['dm']
    print(f'  SCF E={scf_out["E_Ha"]:.6f} Ha  cycles={scf_out["cycles"]}  wall={t_scf:.2f}s')
    lines.append(f'SCF E={scf_out["E_Ha"]:.6f} Ha cycles={scf_out["cycles"]} wall={t_scf:.2f}s backend={backend}')

    # --- GPU GTO project ---
    print(f'[{mol_name}] GTOGridProjector.project_multizeta …')
    proj = GTOGridProjector(mol, verbosity=0)
    proj.project_multizeta(dm, grid_bohr)  # warmup
    t0 = time.perf_counter()
    rho_gto, info = proj.project_multizeta(dm, grid_bohr)
    t_proj = time.perf_counter() - t0
    # ρ projector = e/a0³; FDBM cube path: ρ_Å = ρ_B / (Å/a0)³  (see get_density_from_cube)
    rho_gto_A = (rho_gto / (ang ** 3)).astype(np.float32)
    q_gto = float(rho_gto_A.sum() * dV)
    q_bohr = float(rho_gto.sum() * (float(step) / ang) ** 3)
    print(f'  ∫ρ Bohr-vol={q_bohr:.3f}  Å-vol={q_gto:.3f}')
    ker_ms = float(info.get('kernel_s', 0.0)) * 1e3
    print(f'  ρ_proj wall={t_proj:.3f}s ker={ker_ms:.1f}ms  ∫ρ={q_gto:.3f}  Ne={mol.nelectron}')
    lines.append(f'GTO multizeta wall={t_proj:.3f}s ker={ker_ms:.1f}ms q_scf={q_gto:.3f} Ne={mol.nelectron}')

    # --- sample parity vs numint ---
    coords = grid_coords(grid_bohr)
    npts = coords.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(npts, size=min(parity_npts, npts), replace=False)
    t0 = time.perf_counter()
    rho_ref_s = rho_numint_ref(mol, dm, coords[idx])  # e/Bohr³
    t_samp = time.perf_counter() - t0
    rho_gto_s = rho_gto.ravel()[idx].astype(np.float64)
    max_d = float(np.max(np.abs(rho_gto_s - rho_ref_s)))
    mae = float(np.mean(np.abs(rho_gto_s - rho_ref_s)))
    print(f'  sample parity n={len(idx)}: max|Δ|={max_d:.3e} MAE={mae:.3e}  (numint sample {t_samp*1e3:.1f} ms)')
    lines.append(f'sample vs numint max|Δ|={max_d:.3e} MAE={mae:.3e}')

    # --- vs cube (different SCF → expect moderate Δ) ---
    d_cube_gto = rho_gto_A.astype(np.float64) - rho_cube.astype(np.float64)
    mae_c = float(np.mean(np.abs(d_cube_gto)))
    max_c = float(np.max(np.abs(d_cube_gto)))
    print(f'  vs remeshed cube: Δq={q_gto-q_cube:.3f}  MAE={mae_c:.3e}  max|Δ|={max_c:.3e}')
    lines.append(f'vs cube remesh Δq={q_gto-q_cube:.3f} MAE={mae_c:.3e} max|Δ|={max_c:.3e}')

    # slice plot ρ
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # molecular plane: z near atom mean
        z_mol = float(atomPos[:, 2].mean())
        iz = int(np.clip(round((z_mol - origin[2]) / step), 0, nz - 1))
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, data, title in (
            (axs[0], rho_cube[:, :, iz], f'cube remesh z={origin[2]+iz*step:.2f}'),
            (axs[1], rho_gto_A[:, :, iz], 'GTO GPU (SCF)'),
            (axs[2], d_cube_gto[:, :, iz], 'GTO − cube'),
        ):
            im = ax.imshow(np.asarray(data, dtype=float).T, origin='lower', aspect='equal')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f'{mol_name} ρ XY  q_cube={q_cube:.1f} q_gto={q_gto:.1f}')
        fig.tight_layout()
        p_rho = os.path.join(outdir, 'rho_xy_compare.png')
        fig.savefig(p_rho, dpi=120)
        plt.close(fig)
        print(f'REVIEW: {p_rho}')
        lines.append(f'REVIEW: {p_rho}')
    except Exception as e:
        print(f'  rho plot skipped: {e}')

    # --- DFTB prolonged (Pauli) + stock ES already from cube for gto ---
    basis = '3ob-3-1'
    basis_hsd = get_dftb_basis_path(basis)
    print(f'[{mol_name}] DFTB prolonged …')
    res_stock = U.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, os.path.join(outdir, 'dftb_work_stock'),
        grid_spec=grid_spec_A, step=step, verbosity=0)
    rho_diff_stock = res_stock['rho_diff']
    basis_data = parse_wfc_hsd(basis_hsd)
    basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
    proj_prol = make_slater_tail_species_list(basis_ang)
    res_prol = U.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, os.path.join(outdir, 'dftb_work_prolonged'),
        grid_spec=grid_spec_A, step=step, verbosity=0, projection_basis_ang=proj_prol)
    rho_prol = res_prol['rho_scf']

    pa = dict(afm.PAULI_FITTED_DEFAULTS['3ob-3-1'])
    A, beta = float(pa['A']), float(pa['beta'])
    lines.append(f'Pauli A={A:.3f} β={beta:.4f}')

    # gto ES: cube Δρ (same ES as cube row → Pauli channel isolates density quality)
    variants = {}
    specs = [
        ('cube', rho_cube, rho_diff_cube),
        ('gto_gpu', rho_gto_A, rho_diff_cube),
        ('prolonged', rho_prol, rho_diff_stock),
    ]
    for key, rho, rho_d in specs:
        r = U.run_fdbm_pp_from_density(
            key, rho, atomPos, atomTypes, origin, step, ngrid, A, beta, 'co', outdir,
            rho_diff=rho_d, basis=basis, margin=margin,
            h_min=h_min, h_max=h_max, h_step=h_step, amp=amp, amp_align=True,
            use_fast_s3=True,
        )
        variants[key] = r
        lines.append(f'[{key}] df=[{r["df"].min():.3e},{r["df"].max():.3e}]')

    heights = variants['cube']['heights']
    row_specs = [
        ('df', 'cube', f'df  DFT cube\nA={A:.1f} β={beta:.2f}', 'gray'),
        ('df', 'gto_gpu', f'df  GTO GPU ρ\n(same ES as cube)', 'gray'),
        ('df', 'prolonged', f'df  prolonged\nA={A:.1f} β={beta:.2f}', 'gray'),
        ('Fz', 'cube', f'Fz  cube\n@h−{amp:.1f}Å', 'seismic'),
        ('Fz', 'gto_gpu', f'Fz  GTO GPU\n@h−{amp:.1f}Å', 'seismic'),
        ('Fz', 'prolonged', f'Fz  prolonged\n@h−{amp:.1f}Å', 'seismic'),
    ]
    title = (f'{mol_name} FDBM CO | cube vs GTO GPU projector vs prolonged\n'
             f'GTO: PBE/def2-SVP SCF + tiled Hermite; ES(gto)=cube Δρ')
    cmp = os.path.join(outdir, 'compare_cube_gto_prolonged.png')
    U.plot_afm_variant_height_strip(
        variants, row_specs, heights, cmp, scale='per_image', title=title,
        dpi=140, amp=amp, amp_align=True, long_axis_vertical=True, tight=True)
    print(f'REVIEW: {cmp}')
    lines.append(f'REVIEW: {cmp}')

    # quantitative df discrepancy cube vs gto
    df_c = variants['cube']['df']
    df_g = variants['gto_gpu']['df']
    df_d = df_g.astype(np.float64) - df_c.astype(np.float64)
    rms = float(np.sqrt(np.mean(df_d ** 2)))
    mx = float(np.max(np.abs(df_d)))
    rel = rms / (float(np.std(df_c)) + 1e-30)
    print(f'  df GTO−cube: RMS={rms:.3e}  max|Δ|={mx:.3e}  RMS/σ_cube={rel:.3f}')
    lines.append(f'df GTO−cube RMS={rms:.3e} max|Δ|={mx:.3e} RMS/σ={rel:.3f}')

    np.savez(os.path.join(outdir, 'scan_cube_gto_prolonged.npz'),
             heights=heights,
             df_cube=df_c, Fz_cube=variants['cube']['Fz'],
             df_gto_gpu=df_g, Fz_gto_gpu=variants['gto_gpu']['Fz'],
             df_prolonged=variants['prolonged']['df'], Fz_prolonged=variants['prolonged']['Fz'],
             A=A, beta=beta, q_cube=q_cube, q_gto=q_gto,
             t_proj_s=t_proj, sample_max_d=max_d)

    pyscf_u.release_scf(scf_out['mf'])
    summary = os.path.join(outdir, 'SUMMARY.out')
    open(summary, 'w').write('\n'.join(lines) + '\n')
    print(f'REVIEW: {summary}')
    return {
        't_proj': t_proj, 'max_d': max_d, 'q_gto': q_gto, 'q_cube': q_cube,
        'df_rms': rms, 'df_rel': rel, 'cmp': cmp,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--molecule', nargs='*', default=['pentacene'])
    p.add_argument('--outdir', default=OUTDIR_DEFAULT)
    p.add_argument('--step', type=float, default=0.1)
    p.add_argument('--margin', type=float, default=4.0)
    p.add_argument('--z-extra', type=float, default=6.0, dest='z_extra')
    args = p.parse_args(argv)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    results = []
    for mol in args.molecule:
        results.append((mol, run_one(
            mol, outdir, step=args.step, margin=args.margin, z_vac=args.z_extra)))
    master = os.path.join(outdir, 'SUMMARY.out')
    with open(master, 'w') as f:
        f.write('GTO grid AFM verify\n\n')
        for mol, r in results:
            f.write(f'{mol}: ρ_proj={r["t_proj"]:.3f}s  sample_max|Δ|={r["max_d"]:.3e}  '
                    f'q_gto={r["q_gto"]:.2f} q_cube={r["q_cube"]:.2f}  '
                    f'df_RMS/σ={r["df_rel"]:.3f}\n')
            f.write(f'  REVIEW: {r["cmp"]}\n')
    print(f'REVIEW: {master}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
