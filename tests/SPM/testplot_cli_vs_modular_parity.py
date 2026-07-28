#!/usr/bin/env python3
"""CLI legacy FDBM path vs ModularPipeline FAST_S3 GPU — step parity + timing.

Goal (USER 2026-07-28): before replacing CLI ``_run_from_density`` with ModularPipeline,
prove (1) numerical parity on shared inputs and (2) GPU speed advantage.

Shared inputs (SSOT):
  - geometry, DFTB SCF + stock Δρ→ES + prolonged Pauli ρ (dual-basis)
  - tip (CO), Pauli A,β from ``PAULI_FITTED_DEFAULTS['3ob-3-1']``
  - scan lattice + amp heights (CLI SSOT 3.7–4.7 / amp=1)
  - PP: ``scan_fdbm(..., use_fire=True)`` default FIRE pars (CLI), not soft compose pars

Paths compared (Stage 3–4 only; SCF shared):
  LEGACY = CLI ``_run_from_density`` style: NumPy FFT Pauli/ES + host sum + GPU grad + scan
  FAST   = ``stage3_fdbm_fields_fast`` (FAST_S3) + same ``scan_fdbm``

Usage:
  python tests/SPM/testplot_cli_vs_modular_parity.py
  python tests/SPM/testplot_cli_vs_modular_parity.py --mol pentacene PTCDA

Outputs: ``debug/cli_vs_modular_parity/<mol>/`` — SUMMARY.out, parity table, PNG diffs.
Do NOT mark Done without USER review of REVIEW: paths.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault('PYOPENCL_CTX', '0')
os.environ.setdefault('SPAMMM_VERBOSITY', '0')
os.environ.setdefault('AFM_DEBUG_PLOT_LEVEL', '0')
# Parity run needs host E_* from fast path
os.environ['SPAMMM_AFM_DIAG_DOWNLOAD'] = '1'
os.environ.setdefault('SPAMMM_AFM_BENCH_NO_IO', '1')

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MOL_XYZ = {
    'pentacene': os.path.join(ROOT, 'data', 'xyz', 'pentacene.xyz'),
    'PTCDA': os.path.join(ROOT, 'data', 'xyz', 'PTCDA.xyz'),
}


def _set_cpu_fft(on: bool):
    """Toggle NumPy vs gpyFFT for host FFT helpers; rebind module flags."""
    import spammm.SPM.AFM as afm
    if on:
        os.environ['SPAMMM_AFM_CPU_FFT'] = '1'
        afm.AFM_CPU_FFT = 1
    else:
        os.environ.pop('SPAMMM_AFM_CPU_FFT', None)
        afm.AFM_CPU_FFT = 0


def _metrics(a, b, name):
    from tests.helpers.parity import rmse, max_err, correlation
    a = __import__('numpy').asarray(a, dtype=float).ravel()
    b = __import__('numpy').asarray(b, dtype=float).ravel()
    # Allow NaN-free only
    m = __import__('numpy').isfinite(a) & __import__('numpy').isfinite(b)
    a, b = a[m], b[m]
    if a.size == 0:
        return {'name': name, 'rmse': float('nan'), 'max': float('nan'), 'corr': float('nan'), 'n': 0}
    return {
        'name': name,
        'rmse': rmse(a, b),
        'max': max_err(a, b),
        'corr': correlation(a, b) if a.size > 2 else float('nan'),
        'n': int(a.size),
        'ref_rms': float(__import__('numpy').sqrt(__import__('numpy').mean(a * a))),
        'rel_rmse': float('nan'),
    }


def _finalize_rel(m):
    if m['ref_rms'] > 1e-30:
        m['rel_rmse'] = m['rmse'] / m['ref_rms']
    else:
        m['rel_rmse'] = float('nan')
    return m


def _plot_triple(legacy, fast, title, path, cmap='seismic'):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spammm.SPM.AFM_utils import imshow_afm

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    diff = np.asarray(fast, float) - np.asarray(legacy, float)
    imshow_afm(axes[0], legacy, cmap=cmap, title='LEGACY (CLI-style)')
    imshow_afm(axes[1], fast, cmap=cmap, title='FAST (Modular FAST_S3)')
    imshow_afm(axes[2], diff, cmap='bwr', title='FAST − LEGACY')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {path}')


def _prepare_shared(xyz, outdir, *, basis='3ob-3-1', step=0.1, margin=4.0,
                    scan_margin=2.0, h_min=3.7, h_max=4.7, h_step=0.1, amp=1.0,
                    projection='prolonged'):
    """SCF + density + tip + scan lattice — identical for both Stage3/4 paths."""
    import numpy as np
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.SPM.ModularPipeline import ModularAFMPipeline
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import (
        parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list,
    )
    from spammm.atomicUtils import load_xyz
    from spammm.forcefields.FFController import orient_long_axis_x

    pos, _, names, _, _ = load_xyz(xyz)
    atomPos = np.asarray(pos, dtype=np.float64).copy()
    orient_long_axis_x(atomPos)
    enames = list(names)

    work = os.path.join(outdir, 'shared_work')
    os.makedirs(work, exist_ok=True)
    # GPU FFT for density projection / Modular S2
    _set_cpu_fft(False)

    h_df, h_Fz, h_scan = afm_utils.afm_df_height_stacks(h_min, h_max, h_step, amp=amp, amp_align=True)
    pipe = ModularAFMPipeline(
        xyz_file=None, output_dir=work, atomPos=atomPos, enames=enames,
        basis=basis, tip_mode='co', step=step, margin=margin, z_extra=6.0,
        scan_range=float(scan_margin), scan_step=float(step),
        height_range=(float(h_scan[0]), float(h_scan[-1]) + 0.5 * float(h_step)),
        height_step=float(h_step), backend='dftb',
        work_dir=os.path.join(work, 'dftb'),
    )
    pipe.heights = np.asarray(h_scan, dtype=np.float64)
    # CLI-like lateral scan: arange with scan_margin (not linspace scan_range)
    scan_xs = np.arange(float(atomPos[:, 0].min() - scan_margin),
                        float(atomPos[:, 0].max() + scan_margin), step, dtype=np.float32)
    scan_ys = np.arange(float(atomPos[:, 1].min() - scan_margin),
                        float(atomPos[:, 1].max() + scan_margin), step, dtype=np.float32)
    pipe.scan_xs, pipe.scan_ys = scan_xs, scan_ys

    t0 = time.perf_counter()
    dm, eigvecs, eigvals = pipe.stage1_scf(force_recompute=True)
    t_scf = time.perf_counter() - t0
    t0 = time.perf_counter()
    rho_scf_stock, rho_na, rho_diff = pipe.stage2_project(dm, force_recompute=True)
    t_s2 = time.perf_counter() - t0

    if projection == 'prolonged':
        basis_hsd = get_dftb_basis_path(basis)
        basis_ang = convert_wfc_to_species_list_ang(parse_wfc_hsd(basis_hsd), resolution_bohr=0.04)
        prol = make_slater_tail_species_list(basis_ang)
        t0 = time.perf_counter()
        rho_scf_pauli = pipe.project_pauli_rho(dm, projection='prolonged', rho_scf_stock=rho_scf_stock)
        t_prol = time.perf_counter() - t0
    else:
        rho_scf_pauli = rho_scf_stock
        t_prol = 0.0

    tip_tot, tip_del = afm_utils.get_tip_densities(
        tip_mode='co', target_shape=tuple(int(x) for x in pipe.ngrid),
        step=pipe.step, margin=margin, output_dir=work, backend='dftb',
        pad_mode='cpu',  # full-grid rolled tip — both LEGACY and FAST share identical arrays
    )
    pa = dict(afm.PAULI_FITTED_DEFAULTS[basis])
    return {
        'pipe': pipe, 'atomPos': atomPos, 'enames': enames,
        'rho_scf_pauli': rho_scf_pauli, 'rho_na': rho_na, 'rho_diff': rho_diff,
        'tip_tot': tip_tot, 'tip_del': tip_del,
        'scan_xs': scan_xs, 'scan_ys': scan_ys,
        'h_df': h_df, 'h_Fz': h_Fz, 'h_scan': h_scan, 'amp': amp,
        'A': pa['A'], 'beta': pa['beta'],
        't_scf': t_scf, 't_s2': t_s2, 't_prol': t_prol,
        'mol_z': float(atomPos[:, 2].max()),
        'origin': pipe.origin, 'step': pipe.step, 'ngrid': pipe.ngrid,
    }


def _run_legacy_s3s4(shared, K_LAT_Nm=0.5, K_RAD=20.0, bond_length=3.0):
    """CLI ``_run_from_density`` Stage3–4 (CPU FFT fields + GPU grad + FIRE scan)."""
    import numpy as np
    from spammm.SPM import AFM as afm

    _set_cpu_fft(True)  # match run_spm.py afm default
    origin, step, ngrid = shared['origin'], shared['step'], shared['ngrid']
    atomPos = shared['atomPos']
    A, beta = shared['A'], shared['beta']
    rho_scf, rho_diff = shared['rho_scf_pauli'], shared['rho_diff']
    tip_tot, tip_del = shared['tip_tot'], shared['tip_del']
    atomTypes = shared['pipe'].atomTypes

    t0 = time.perf_counter()
    V_ES = afm.fft_poisson(rho_diff, step)
    overlap = afm.compute_pauli_overlap(rho_scf, tip_tot, step, tip_rolled=True)
    E_pauli = afm.scale_pauli_field(overlap, step, A, beta, return_grads=False)
    E_ES = afm.compute_es_conv_field(V_ES, tip_del, step, tip_rolled=True, return_grads=False)
    afmulator = afm.AFMulator(use_morse=False, nloc=32)  # use_fire=True default
    E_vdw = afm.compute_dispersion_grid(
        atomPos, atomTypes, origin, step, ngrid, C6_CO=30.0, return_grads=False, afmulator=afmulator)
    E_total = E_pauli + E_ES + E_vdw
    F_total = afmulator.compute_gradient_cl(E_total, step, bAlloc=True)
    afmulator.queue.finish()
    t_s3 = time.perf_counter() - t0

    K_LAT = afm.stiffness_Nm_to_eVA2(K_LAT_Nm)
    t0 = time.perf_counter()
    afmulator.setup_fdbm_grid(F_total, origin, step)
    FEs, tip_disp = afmulator.scan_fdbm(
        shared['scan_xs'], shared['scan_ys'], shared['h_scan'], mol_z=shared['mol_z'],
        K_LAT=K_LAT, K_RAD=K_RAD, bond_length=bond_length, ppm_mode=True, use_fire=True,
    )
    afmulator.queue.finish()
    t_s4 = time.perf_counter() - t0

    Fz_full = FEs[:, :, :, 2]
    dz = float(shared['h_scan'][1] - shared['h_scan'][0])
    df_full = afm.compute_df_amp(Fz_full, dz, amp=float(shared['amp']))
    idx_df = [int(np.argmin(np.abs(shared['h_scan'] - h))) for h in shared['h_df']]
    return {
        'V_ES': V_ES, 'E_pauli': E_pauli, 'E_ES': E_ES, 'E_vdw': E_vdw, 'F_total': F_total,
        'FEs': FEs, 'tip_disp': tip_disp, 'df_full': df_full, 'df': df_full[:, :, idx_df],
        'Fz': Fz_full[:, :, idx_df],  # amp-aligned display uses h_Fz; here same idx as df for slice plot
        't_s3': t_s3, 't_s4': t_s4, 'path': 'LEGACY_CPU_FFT',
    }


def _run_fast_s3s4(shared, K_LAT_Nm=0.5, K_RAD=20.0, bond_length=3.0):
    """Modular FAST_S3 Stage3 + same FIRE scan as CLI."""
    import numpy as np
    from spammm.SPM import AFM as afm

    _set_cpu_fft(False)
    os.environ['SPAMMM_AFM_FAST_S3'] = '1'
    afm.AFM_FAST_S3 = 1

    origin, step, ngrid = shared['origin'], shared['step'], shared['ngrid']
    atomPos = shared['atomPos']
    A, beta = shared['A'], shared['beta']
    atomTypes = shared['pipe'].atomTypes

    afmulator = afm.AFMulator(use_morse=False, nloc=32)
    t0 = time.perf_counter()
    V_ES, E_pauli, E_ES, E_vdw, F_total = afm.stage3_fdbm_fields_fast(
        afmulator, shared['rho_scf_pauli'], shared['rho_diff'],
        shared['tip_tot'], shared['tip_del'],
        origin, step, ngrid, atomPos, atomTypes, A, beta, C6_CO=30.0,
        tip_already_rolled=True, download_fields=True,
    )
    afmulator.queue.finish()
    t_s3 = time.perf_counter() - t0

    K_LAT = afm.stiffness_Nm_to_eVA2(K_LAT_Nm)
    t0 = time.perf_counter()
    # grid already set by stage3_fdbm_fields_fast → setup_fdbm_grid_from_img
    FEs, tip_disp = afmulator.scan_fdbm(
        shared['scan_xs'], shared['scan_ys'], shared['h_scan'], mol_z=shared['mol_z'],
        K_LAT=K_LAT, K_RAD=K_RAD, bond_length=bond_length, ppm_mode=True, use_fire=True,
    )
    afmulator.queue.finish()
    t_s4 = time.perf_counter() - t0

    Fz_full = FEs[:, :, :, 2]
    dz = float(shared['h_scan'][1] - shared['h_scan'][0])
    df_full = afm.compute_df_amp(Fz_full, dz, amp=float(shared['amp']))
    idx_df = [int(np.argmin(np.abs(shared['h_scan'] - h))) for h in shared['h_df']]
    # Optional V_ES for parity (diag download already requested)
    if V_ES is None:
        V_ES = afm.fft_poisson(shared['rho_diff'], step)
    return {
        'V_ES': V_ES, 'E_pauli': E_pauli, 'E_ES': E_ES, 'E_vdw': E_vdw, 'F_total': F_total,
        'FEs': FEs, 'tip_disp': tip_disp, 'df_full': df_full, 'df': df_full[:, :, idx_df],
        'Fz': Fz_full[:, :, idx_df],
        't_s3': t_s3, 't_s4': t_s4, 'path': 'FAST_S3_GPU',
    }


def _z_slice_idx(shared, z_want=4.2):
    import numpy as np
    h = shared['h_df']
    return int(np.argmin(np.abs(h - z_want)))


def run_molecule(name, xyz, out_root, **kw):
    import numpy as np

    outdir = os.path.join(out_root, name)
    os.makedirs(outdir, exist_ok=True)
    print(f'\n======== {name}  xyz={xyz} ========')
    shared = _prepare_shared(xyz, outdir, **kw)
    print(f"  grid={shared['ngrid']} step={shared['step']}  "
          f"A={shared['A']:.2f} β={shared['beta']:.4f}  "
          f"scan=({len(shared['scan_xs'])},{len(shared['scan_ys'])},{len(shared['h_scan'])})")
    print(f"  shared: SCF={shared['t_scf']:.3f}s  S2={shared['t_s2']:.3f}s  prol={shared['t_prol']:.3f}s")

    # Warm GPU once (ignore timing)
    _ = _run_fast_s3s4(shared)

    leg = _run_legacy_s3s4(shared)
    fas = _run_fast_s3s4(shared)

    rows = []
    for key, lab in (
        ('E_pauli', 'E_pauli'),
        ('E_ES', 'E_ES'),
        ('E_vdw', 'E_vdw'),
        ('F_total', 'F_total'),
    ):
        m = _finalize_rel(_metrics(leg[key], fas[key], lab))
        rows.append(m)
    m = _finalize_rel(_metrics(leg['FEs'][..., 2], fas['FEs'][..., 2], 'Fz_relax'))
    rows.append(m)
    m = _finalize_rel(_metrics(leg['df_full'], fas['df_full'], 'df_amp'))
    rows.append(m)
    dxy_l = np.hypot(leg['tip_disp']['dx'], leg['tip_disp']['dy'])
    dxy_f = np.hypot(fas['tip_disp']['dx'], fas['tip_disp']['dy'])
    rows.append(_finalize_rel(_metrics(dxy_l, dxy_f, '|dxy|')))

    iz = _z_slice_idx(shared, 4.2)
    _plot_triple(leg['E_pauli'][:, :, leg['E_pauli'].shape[2] // 2],
                 fas['E_pauli'][:, :, fas['E_pauli'].shape[2] // 2],
                 f'{name} E_pauli mid-z', os.path.join(outdir, '01_E_pauli_midz.png'))
    _plot_triple(leg['E_ES'][:, :, leg['E_ES'].shape[2] // 2],
                 fas['E_ES'][:, :, fas['E_ES'].shape[2] // 2],
                 f'{name} E_ES mid-z', os.path.join(outdir, '02_E_ES_midz.png'))
    _plot_triple(leg['df'][:, :, iz], fas['df'][:, :, iz],
                 f'{name} df amp @ h≈{float(shared["h_df"][iz]):.2f}Å',
                 os.path.join(outdir, '03_df_h42.png'), cmap='gray')
    _plot_triple(dxy_l[:, :, iz], dxy_f[:, :, iz],
                 f'{name} |dxy| @ h≈{float(shared["h_df"][iz]):.2f}Å',
                 os.path.join(outdir, '04_dxy_h42.png'), cmap='magma')

    # Speedup (S3+S4 only; SCF shared)
    t_leg = leg['t_s3'] + leg['t_s4']
    t_fas = fas['t_s3'] + fas['t_s4']
    speed = t_leg / max(t_fas, 1e-9)

    lines = []
    lines.append(f'# {name} CLI-legacy vs Modular FAST_S3 parity')
    lines.append(f'xyz={xyz}')
    lines.append(f'grid={shared["ngrid"]} step={shared["step"]} A={shared["A"]:.4f} beta={shared["beta"]:.4f}')
    lines.append(f'h_df=[{float(shared["h_df"][0]):.2f},{float(shared["h_df"][-1]):.2f}] amp={shared["amp"]}')
    lines.append('')
    lines.append('## Timing (warm; S3+S4 only)')
    lines.append(f'LEGACY  S3={leg["t_s3"]:.4f}s  S4={leg["t_s4"]:.4f}s  sum={t_leg:.4f}s  [{leg["path"]}]')
    lines.append(f'FAST    S3={fas["t_s3"]:.4f}s  S4={fas["t_s4"]:.4f}s  sum={t_fas:.4f}s  [{fas["path"]}]')
    lines.append(f'SPEEDUP S3+S4 = {speed:.2f}×  (LEGACY/FAST)')
    lines.append(f'shared  SCF={shared["t_scf"]:.3f}s S2={shared["t_s2"]:.3f}s prol={shared["t_prol"]:.3f}s')
    lines.append('')
    lines.append('## Parity (FAST vs LEGACY)')
    lines.append(f'{"qty":12s}  {"corr":>8s}  {"rmse":>10s}  {"rel_rmse":>10s}  {"max":>10s}')
    for m in rows:
        lines.append(
            f'{m["name"]:12s}  {m["corr"]:8.5f}  {m["rmse"]:10.3e}  {m["rel_rmse"]:10.3e}  {m["max"]:10.3e}')
    # Soft gates for SUMMARY (not pytest assert — USER reviews)
    ok_fields = all(m['corr'] > 0.99 for m in rows if m['name'] in ('E_pauli', 'E_ES', 'E_vdw'))
    ok_df = next(m for m in rows if m['name'] == 'df_amp')['corr'] > 0.98
    lines.append('')
    lines.append(f'PASS_fields_corr>0.99: {ok_fields}')
    lines.append(f'PASS_df_corr>0.98: {ok_df}')
    lines.append(f'PASS_speedup>1.5: {speed > 1.5}')
    lines.append('')
    lines.append(f'REVIEW: {outdir}/')
    text = '\n'.join(lines) + '\n'
    summary = os.path.join(outdir, 'SUMMARY.out')
    with open(summary, 'w') as f:
        f.write(text)
    print(text)
    print(f'REVIEW: {summary}')
    return {
        'name': name, 'rows': rows, 't_leg': t_leg, 't_fas': t_fas, 'speed': speed,
        'ok_fields': ok_fields, 'ok_df': ok_df, 'outdir': outdir,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mol', nargs='*', default=['pentacene', 'PTCDA'])
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'debug', 'cli_vs_modular_parity'))
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--scan-margin', type=float, default=2.0)
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    results = []
    for name in args.mol:
        if name not in MOL_XYZ:
            raise SystemExit(f'Unknown mol {name!r}; choose from {sorted(MOL_XYZ)}')
        xyz = MOL_XYZ[name]
        if not os.path.isfile(xyz):
            raise SystemExit(f'Missing {xyz}')
        results.append(run_molecule(
            name, xyz, args.outdir, step=args.step, scan_margin=args.scan_margin))

    master = os.path.join(args.outdir, 'SUMMARY.out')
    with open(master, 'w') as f:
        f.write('# CLI legacy vs Modular FAST_S3 — master summary\n')
        for r in results:
            f.write(f'{r["name"]}: speedup={r["speed"]:.2f}×  '
                    f'fields_ok={r["ok_fields"]} df_ok={r["ok_df"]}  '
                    f't_leg={r["t_leg"]:.3f}s t_fas={r["t_fas"]:.3f}s\n')
            f.write(f'  REVIEW: {r["outdir"]}/\n')
        f.write(f'REVIEW: {master}\n')
    print(f'REVIEW: {master}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
