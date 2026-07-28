#!/usr/bin/env python3
"""FDBM stage-by-stage diagnostic + tip_mode comparison (gaussian vs co).

Fixes / checks:
  - CO tip along +z with O rolled to array index (0,0,0)
  - tip_mode=gaussian | tip_mode=co (switchable)
  - Per-stage XY/XZ plots + tip symmetry metrics

Usage:
  PYOPENCL_CTX=0 python tests/SPM/testplot_fdbm_relax.py \\
      --xyz data/xyz/benzene.xyz --tip-mode both --outdir debug/afm_fdbm_diag

Outputs under outdir/:
  tip_*.png, stage_*_{gaussian,co}.png, df_*_{gaussian,co}.png, compare_*.png, SUMMARY.out
"""
import os, sys, argparse, numpy as np

os.environ.setdefault('PYOPENCL_CTX', '0')
os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

def _mirror_asym(sl, axis=0):
    """Mirror asymmetry about the peak of the slice (crop), not array center."""
    a = np.asarray(sl, float)
    if a.ndim != 2:
        return float('nan')
    peak_ij = np.unravel_index(int(np.argmax(np.abs(a))), a.shape)
    r = int(min(peak_ij[0], peak_ij[1], a.shape[0] - 1 - peak_ij[0], a.shape[1] - 1 - peak_ij[1], 20))
    if r < 2:
        return 1.0
    c = a[peak_ij[0] - r:peak_ij[0] + r + 1, peak_ij[1] - r:peak_ij[1] + r + 1]
    peak = np.max(np.abs(c)) + 1e-30
    if axis == 0:
        return float(np.max(np.abs(c - c[::-1, :])) / peak)
    return float(np.max(np.abs(c - c[:, ::-1])) / peak)

def _grid_caption(step, shape_or_ngrid, origin=None, extra=''):
    """Compact param string for figure captions (dstep, ngrid, L, origin)."""
    n = np.asarray(shape_or_ngrid, dtype=int).ravel()[:3]
    nx, ny, nz = int(n[0]), int(n[1]), int(n[2])
    s = f'dstep={step:.3f}Å  ngrid={nx}×{ny}×{nz}  L=({nx*step:.2f},{ny*step:.2f},{nz*step:.2f})Å'
    if origin is not None:
        o = np.asarray(origin, float).ravel()
        s += f'  origin=({o[0]:.2f},{o[1]:.2f},{o[2]:.2f})'
    if extra:
        s += f'  {extra}'
    return s

def _plot_tip(rho, rho_d, outdir, tag, step):
    """Delegates to AFM_utils.plot_afm_tip_debug."""
    from spammm.SPM import AFM_utils as afm_utils
    return afm_utils.plot_afm_tip_debug(rho, rho_d, outdir, tag, step)

def _plot_stages(fields, origin, step, atomPos, outdir, tag, z_above=2.5):
    """Delegates to AFM_utils.plot_afm_fdbm_stages."""
    from spammm.SPM import AFM_utils as afm_utils
    return afm_utils.plot_afm_fdbm_stages(fields, origin, step, atomPos, outdir, tag, z_above=z_above)

def _run_one(xyz, basis, step, margin, tip_mode, outdir, args):
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    import spammm.atomicUtils as au

    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17}
    pos, _, names, _, _ = au.load_xyz(xyz)
    atomPos = np.array(pos, dtype=np.float64)
    enames = list(names)
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in enames], dtype=np.int32)
    mol_z = float(atomPos[:, 2].max())
    tag = tip_mode
    sub = os.path.join(outdir, tip_mode)
    os.makedirs(sub, exist_ok=True)

    grid_spec, origin, ngrid = afm.setup_density_grid(atomPos, step=step, margin=margin, z_extra=6.0)
    nx, ny, nz = int(ngrid[0]), int(ngrid[1]), int(ngrid[2])
    print(f"\n=== tip_mode={tip_mode}  grid={nx}x{ny}x{nz}  mol={os.path.basename(xyz)} ===")

    basis_hsd = get_dftb_basis_path(basis)
    work = os.path.join(sub, 'dftb_work')
    result = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, work,
        grid_spec=grid_spec, step=step, margin=margin, z_extra=6.0, verbosity=0
    )
    rho_scf, rho_diff, V_ES = result['rho_scf'], result['rho_diff'], result['V_ES']

    rho_tip, rho_tip_d = afm_utils.get_tip_densities(
        tip_mode=tip_mode, target_shape=(nx, ny, nz), step=step, margin=margin,
        basis=basis, output_dir=sub, backend='dftb', force_recompute=(tip_mode == 'co'),
    )
    tip_info = _plot_tip(rho_tip, rho_tip_d, outdir, tag, step)

    pauli_params = afm.PAULI_FITTED_DEFAULTS.get(basis, {'A': 787.22, 'beta': 1.2371})
    overlap = afm.compute_pauli_overlap(rho_scf, rho_tip, step, tip_rolled=True)
    E_pauli = afm.scale_pauli_field(overlap, step, pauli_params['A'], pauli_params['beta'], return_grads=False)
    E_ES = afm.compute_es_conv_field(V_ES, rho_tip_d, step, tip_rolled=True, return_grads=False)
    E_vdw = afm.compute_dispersion_grid(atomPos, atomTypes, origin, step, ngrid, C6_CO=30.0, return_grads=False)
    E_total = E_pauli + E_ES + E_vdw
    afmulator = afm.AFMulator(use_morse=False, nloc=32)
    F_total = afmulator.compute_gradient_cl(E_total, step, bAlloc=True)

    fields = {
        'rho_scf': rho_scf, 'E_pauli': E_pauli, 'E_ES': E_ES, 'E_vdw': E_vdw,
        'E_total': E_total, 'Fz': -F_total[..., 2],
    }
    stage_path = _plot_stages(fields, origin, step, atomPos, outdir, tag, z_above=args.height)

    scan_margin = args.scan_margin
    scan_xs = np.arange(float(atomPos[:, 0].min() - scan_margin), float(atomPos[:, 0].max() + scan_margin), step, dtype=np.float32)
    scan_ys = np.arange(float(atomPos[:, 1].min() - scan_margin), float(atomPos[:, 1].max() + scan_margin), step, dtype=np.float32)
    heights = np.arange(args.h_min, args.h_max + 0.5 * args.h_step, args.h_step, dtype=np.float32)
    K_LAT_eV = afm.stiffness_Nm_to_eVA2(args.K_LAT)  # CLI: N/m → internal eV/Å²
    print(f"  K_LAT={args.K_LAT:.3f} N/m → {K_LAT_eV:.4f} eV/Å²  K_RAD={args.K_RAD} eV/Å²  L={args.bond_length} Å")
    afmulator.setup_fdbm_grid(F_total, origin, step)
    FEs, tip_disp = afmulator.scan_fdbm(
        scan_xs, scan_ys, heights, mol_z=mol_z,
        K_LAT=K_LAT_eV, K_RAD=args.K_RAD, bond_length=args.bond_length,
        ppm_mode=True, use_fire=True,
    )
    Fz = FEs[:, :, :, 2]
    df = afm.compute_df_amp(Fz, float(heights[1] - heights[0]), amp=args.amp)
    x_ext = [float(scan_xs[0]), float(scan_xs[-1])]
    y_ext = [float(scan_ys[0]), float(scan_ys[-1])]
    afm_utils.plot_grid_Fz(df, heights, f'df tip={tag}  {_grid_caption(step, (len(scan_xs), len(scan_ys), len(heights)), origin)}', f'df_{tag}.png',
                           x_ext=x_ext, y_ext=y_ext, save_dir=outdir, cmap=args.df_cmap)
    afm_utils.plot_grid_Fz(Fz, heights, f'Fz tip={tag}  {_grid_caption(step, (len(scan_xs), len(scan_ys), len(heights)), origin)}', f'Fz_{tag}.png',
                           x_ext=x_ext, y_ext=y_ext, save_dir=outdir, cmap=args.cmap)

    # Mid-height asymmetry of df
    ih = len(heights) // 2
    df_asym = _mirror_asym(df[:, :, ih], 0)
    print(f"  df mirrorX @h={heights[ih]:.2f}: {df_asym:.3e}  range=[{df.min():.3e},{df.max():.3e}]")

    return {
        'tip': tip_info, 'stage_path': stage_path, 'df': df, 'Fz': Fz, 'heights': heights,
        'scan_xs': scan_xs, 'scan_ys': scan_ys, 'df_asym': df_asym, 'fields': fields,
        'atomPos': atomPos, 'origin': origin, 'step': step,
    }

# FITTING ONLY (PTCDA Ez calibration 2026-07-20) — used by run_ptcda_stock_vs_sa.
# NEVER import into panel-fukui / AFM_CLI_FDBM / default evaluation (transferable
# defaults live in AFM.PAULI_FITTED_DEFAULTS).
PTCDA_PAULI_FIT = {
    'stock': {'A': 12.817, 'beta': 0.6514, 'label': 'stock 3ob'},
    'sa':    {'A': 11.762, 'beta': 0.8519, 'label': 'SA-prolonged'},
}


def _grid_spec_from_meta(meta):
    origin = np.asarray(meta['origin'], dtype=float)
    step = float(meta['step'])
    ngrid = tuple(int(x) for x in meta['ngrid'])
    return {
        'origin': origin, 'ngrid': ngrid,
        'dA': [step, 0.0, 0.0], 'dB': [0.0, step, 0.0], 'dC': [0.0, 0.0, step],
    }, origin, step, ngrid


def _run_from_density(tag, rho_scf, V_ES, atomPos, atomTypes, origin, step, ngrid, A, beta, tip_mode, outdir, args,
                      rho_diff=None):
    """Thin wrapper → ``AFM_utils.run_fdbm_pp_from_density`` (FAST_S3 by default).

    Kept so visual demos / Fukui helpers in this script stay short. Prefer calling
    AFM_utils directly from product CLI (``run_spm.py afm`` already does).
    Pass ``rho_diff`` for dual-basis ES (stock Δρ); required for FAST_S3.
    """
    from spammm.SPM import AFM_utils as afm_utils

    plots = getattr(args, '_plots', None)
    if plots is None:
        raw = getattr(args, 'plots', None)
        if raw is None:
            plots = {'compare', 'stage', 'tip', 'df', 'fz'}
        else:
            parts = [p.strip().lower() for p in str(raw).replace(';', ',').split(',') if p.strip()]
            if 'all' in parts or 'debug' in parts:
                plots = {'compare', 'stage', 'tip', 'df', 'fz', 'per_image'}
            elif 'none' in parts or 'off' in parts:
                plots = set()
            else:
                plots = set(parts)
    use_fast = not bool(getattr(args, 'cpu_fft', False))
    if os.environ.get('SPAMMM_AFM_CPU_FFT', '0') == '1':
        use_fast = False
    if rho_diff is None:
        rho_diff = getattr(args, '_rho_diff', None)
    return afm_utils.run_fdbm_pp_from_density(
        tag, rho_scf, atomPos, atomTypes, origin, step, ngrid, A, beta, tip_mode, outdir,
        rho_diff=rho_diff, V_ES=V_ES,
        basis=getattr(args, 'basis', '3ob-3-1'), margin=getattr(args, 'margin', 4.0),
        h_min=getattr(args, 'h_min', 3.7), h_max=getattr(args, 'h_max', 4.7),
        h_step=getattr(args, 'h_step', 0.1), amp=getattr(args, 'amp', 1.0),
        amp_align=not bool(getattr(args, 'no_amp_align', False)),
        K_LAT_Nm=getattr(args, 'K_LAT', 0.5), K_RAD=getattr(args, 'K_RAD', 20.0),
        bond_length=getattr(args, 'bond_length', 3.0),
        scan_margin=getattr(args, 'scan_margin', 2.0),
        plots=plots & {'tip', 'stage', 'df', 'fz'},
        df_cmap=getattr(args, 'df_cmap', 'gray'), cmap=getattr(args, 'cmap', 'seismic'),
        stage_height=getattr(args, 'height', 4.2), use_fast_s3=use_fast,
    )


def run_ptcda_stock_vs_sa(args):
    """PTCDA AFM: stock 3ob vs SA-prolonged projection, with Ez-fitted Pauli A,β."""
    import json
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    import spammm.atomicUtils as au

    os.environ['SPAMMM_AFM_CPU_FFT'] = '1'  # PTCDA ny=176 has prime factor 11
    _ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    dens_dir = os.path.join(_ROOT, 'debug', 'densities')
    params_path = args.sa_params if os.path.isabs(args.sa_params) else os.path.join(_ROOT, args.sa_params)
    with open(params_path) as f:
        sa_js = json.load(f)

    xyz = args.xyz if os.path.isabs(args.xyz) else os.path.join(_ROOT, args.xyz)
    ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
    pos, _, names, _, _ = au.load_xyz(xyz)
    atomPos = np.array(pos, dtype=np.float64)
    enames = list(names)
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in enames], dtype=np.int32)

    meta_sa = np.load(os.path.join(dens_dir, 'rho_PTCDA_dftb_3ob_sa.meta.npz'))
    grid_spec, origin, step, ngrid = _grid_spec_from_meta(meta_sa)
    # Prefer meta atom positions (same grid as density fit)
    if 'atom_pos' in meta_sa.files:
        atomPos = np.asarray(meta_sa['atom_pos'], dtype=np.float64)
        enames = list(meta_sa['atom_names'])
        atomTypes = np.array([ELEM_Z.get(e, 6) for e in enames], dtype=np.int32)

    basis_hsd = get_dftb_basis_path(args.basis)
    # SA Slater list only needed if regenerating ρ; here we load cached SA ρ
    _ = sa_js  # kept for SUMMARY provenance

    os.makedirs(args.outdir, exist_ok=True)
    lines = [
        'PTCDA FDBM: stock 3ob vs SA-prolonged (CO tip)',
        f'Pauli A,β from Ez fit (pySCF GPU PBE/def2-SVP, CO tip, z∈[1.7,2.5])',
        f'sa_params={params_path}',
        f'grid={ngrid} step={step}',
        '',
    ]

    variants = {}
    # Stock: full density + V_ES. SA: Pauli from SA ρ, ES from stock (ρ_NA imbalance otherwise).
    # DUAL BASIS (SSOT): prolonged/SA ρ is for Pauli ONLY — do NOT charge-normalize it;
    # ES always uses stock Δρ. See make_slater_tail_species_list / doc/DFTB_basis_fit.md.
    work_stock = os.path.join(args.outdir, 'dftb_work_stock')
    print("\nDensity stock (ρ + V_ES)...")
    res_stock = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, work_stock, grid_spec=grid_spec,
        step=step, verbosity=0)
    V_ES = res_stock['V_ES']
    rho_stock = res_stock['rho_scf']
    npy_stock = os.path.join(dens_dir, 'rho_PTCDA_dftb_3ob.npy')
    if os.path.isfile(npy_stock) and np.load(npy_stock, mmap_mode='r').shape == rho_stock.shape:
        rho_stock = np.load(npy_stock)
        print(f"  Using cached stock ρ  q={rho_stock.sum()*step**3:.2f}")
    npy_sa = os.path.join(dens_dir, 'rho_PTCDA_dftb_3ob_sa.npy')
    if not os.path.isfile(npy_sa):
        raise FileNotFoundError(f"Need {npy_sa} — run optimize_basis.py --project-density first")
    rho_sa = np.load(npy_sa)
    print(f"  Loaded SA ρ  q={rho_sa.sum()*step**3:.2f}  (V_ES shared from stock)")

    for key, rho in [('stock', rho_stock), ('sa', rho_sa)]:
        pa = PTCDA_PAULI_FIT[key]
        r = _run_from_density(
            key, rho, V_ES, atomPos, atomTypes, origin, step, ngrid,
            pa['A'], pa['beta'], 'co', args.outdir, args, rho_diff=res_stock['rho_diff'])
        variants[key] = r
        lines.append(f"[{key}] A={pa['A']:.3f} β={pa['beta']:.4f}  "
                     f"df=[{r['df'].min():.3e},{r['df'].max():.3e}]  REVIEW: df_{key}.png")
    lines.append('NOTE: both use stock V_ES; SA-prolonged changes Pauli ρ + A,β only')
    lines.append('SA = Simulated Annealing fit of single-exponent Slater tails (N,ζ) vs pySCF ρ')
    lines.append('stock = multi-zeta 3ob-3-1 projection (NOT mio)')
    lines.append('DUAL BASIS: prolonged ρ is NOT charge-normalized; never use it for ES')

    # One figure: columns = heights, rows = df/Fz × stock 3ob / SA-prolonged
    heights = variants['stock']['heights']
    n_h = len(heights)
    pa_s, pa_a = PTCDA_PAULI_FIT['stock'], PTCDA_PAULI_FIT['sa']
    row_specs = [
        ('df', 'stock', f"df  stock 3ob\nA={pa_s['A']:.2f} β={pa_s['beta']:.3f}", args.df_cmap),
        ('df', 'sa',    f"df  SA-prolonged\nA={pa_a['A']:.2f} β={pa_a['beta']:.3f}", args.df_cmap),
        ('Fz', 'stock', f"Fz  stock 3ob", args.cmap),
        ('Fz', 'sa',    f"Fz  SA-prolonged", args.cmap),
    ]

    fig, axes = plt.subplots(4, n_h, figsize=(1.55 * n_h + 1.2, 9.5),
                             squeeze=False)
    for ih, h in enumerate(heights):
        # Per-height scale so far tips stay visible (not crushed by h=2.5)
        vmax_df_h = max(
            np.percentile(np.abs(variants['stock']['df'][:, :, ih]), 99),
            np.percentile(np.abs(variants['sa']['df'][:, :, ih]), 99),
            1e-30)
        vmax_Fz_h = max(
            np.percentile(np.abs(variants['stock']['Fz'][:, :, ih]), 99),
            np.percentile(np.abs(variants['sa']['Fz'][:, :, ih]), 99),
            1e-30)
        for ir, (qty, key, ylab, cmap) in enumerate(row_specs):
            ax = axes[ir, ih]
            vmax = vmax_df_h if qty == 'df' else vmax_Fz_h
            im = ax.imshow(variants[key][qty][:, :, ih].T, origin='lower', cmap=cmap,
                           vmin=-vmax, vmax=vmax, aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if ir == 0:
                ax.set_title(f'h={h:.2f}Å', fontsize=9)
            if ih == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if ih == n_h - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        'PTCDA FDBM  CO tip  |  stock = multi-ζ 3ob-3-1  |  '
        'SA-prolonged = Simulated-Annealing Slater-tail projection (fit vs pySCF ρ)\n'
        'Pauli A,β refit vs pySCF Ez (CO tip); shared stock V_ES  |  color scale per column (stock↔SA comparable)',
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(args.outdir, 'compare_stock3ob_vs_SAprolonged_heights.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    lines.append(f'REVIEW: {p}')
    print(f'REVIEW: {p}')

    # Persist for replot
    np.savez(os.path.join(args.outdir, 'scan_stock_vs_sa.npz'),
             heights=heights,
             df_stock=variants['stock']['df'], Fz_stock=variants['stock']['Fz'],
             df_sa=variants['sa']['df'], Fz_sa=variants['sa']['Fz'],
             A_stock=pa_s['A'], beta_stock=pa_s['beta'],
             A_sa=pa_a['A'], beta_sa=pa_a['beta'])

    summary = os.path.join(args.outdir, 'SUMMARY.out')
    lines += ['', f'REVIEW: {summary}', f'Total variants: {len(variants)}', f'heights={list(map(float, heights))}']
    open(summary, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'REVIEW: {summary}')
    return variants


# Fukui panel — SSOT lives in AFM_utils (CLI uses that; this script is a thin demo wrapper)
from spammm.SPM.AFM_utils import (  # noqa: E402
    FUKUI_CUBE_ROOTS, FUKUI_CUBE_ROOT, FUKUI_PANEL,
    fukui_cube_dir as _fukui_cube_dir,
    run_fukui_one as _run_fukui_one_mod,
    run_fukui_panel as _run_fukui_panel_mod,
    replot_fukui_per_image,
)


def _fft_friendly_grid_spec(atomPos, step, margin, z_extra):
    """Deprecated name — delegates to ``make_fdbm_grid_com_zsym``."""
    from spammm.SPM import AFM_utils as afm_utils
    z_vac = float(z_extra) if z_extra is not None else 7.0
    return afm_utils.make_fdbm_grid_com_zsym(atomPos, step, margin, z_vac=z_vac)


def run_fukui_one(mol, xyz_rel, args):
    """Thin wrapper → ``AFM_utils.run_fukui_one`` (FAST_S3 unless ``args.cpu_fft``)."""
    use_fast = not bool(getattr(args, 'cpu_fft', False))
    if os.environ.get('SPAMMM_AFM_CPU_FFT', '0') == '1':
        use_fast = False
    return _run_fukui_one_mod(
        mol, xyz_rel, args.outdir,
        step=getattr(args, 'step', 0.15), margin=getattr(args, 'margin', 4.0),
        basis=getattr(args, 'basis', None) or '3ob-3-1', tip_mode='co',
        h_min=getattr(args, 'h_min', 3.7), h_max=getattr(args, 'h_max', 4.7),
        h_step=getattr(args, 'h_step', 0.1), amp=getattr(args, 'amp', 1.0),
        amp_align=not bool(getattr(args, 'no_amp_align', False)),
        K_LAT=getattr(args, 'K_LAT', 0.5), K_RAD=getattr(args, 'K_RAD', 20.0),
        bond_length=getattr(args, 'bond_length', 3.0),
        scan_margin=getattr(args, 'scan_margin', 2.0), height=getattr(args, 'height', 4.2),
        cmap=getattr(args, 'cmap', 'seismic'), df_cmap=getattr(args, 'df_cmap', 'gray'),
        use_fast_s3=use_fast,
    )


def run_fukui_panel(args):
    """Thin wrapper → ``AFM_utils.run_fukui_panel``."""
    use_fast = not bool(getattr(args, 'cpu_fft', False))
    if os.environ.get('SPAMMM_AFM_CPU_FFT', '0') == '1':
        use_fast = False
    return _run_fukui_panel_mod(
        args.outdir, molecules=getattr(args, 'molecule', None), use_fast_s3=use_fast,
        step=getattr(args, 'step', 0.15), margin=getattr(args, 'margin', 4.0),
        basis=getattr(args, 'basis', None) or '3ob-3-1', tip_mode='co',
        h_min=getattr(args, 'h_min', 3.7), h_max=getattr(args, 'h_max', 4.7),
        h_step=getattr(args, 'h_step', 0.1), amp=getattr(args, 'amp', 1.0),
        amp_align=not bool(getattr(args, 'no_amp_align', False)),
        K_LAT=getattr(args, 'K_LAT', 0.5), K_RAD=getattr(args, 'K_RAD', 20.0),
        bond_length=getattr(args, 'bond_length', 3.0),
        scan_margin=getattr(args, 'scan_margin', 2.0), height=getattr(args, 'height', 4.2),
        cmap=getattr(args, 'cmap', 'seismic'), df_cmap=getattr(args, 'df_cmap', 'gray'),
    )


def run_fukui_es_diag_one(mol, xyz_rel, args):
    """Cube electrostatic chain diagnostics (ρ, Δρ, V_ES, E_ES, tip) + mirror metrics.

    Compares legacy Gaussian+scipy-sample vs pyridine SSOT (clamp→compact + GridsOCL project).
    Writes under ``<outdir>/<mol>/es_diag/``.
    """
    import matplotlib; matplotlib.use('Agg')
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    from spammm.utils.GridsOCL import grid_moments

    os.environ['SPAMMM_AFM_CPU_FFT'] = '1'
    _ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Prefer cube frame for ES diag too
    cube_dir = _fukui_cube_dir(mol)
    outdir = os.path.join(args.outdir, mol, 'es_diag')
    os.makedirs(outdir, exist_ok=True)
    z_above = tuple(getattr(args, 'z_above', None) or (1.0, 5.0))

    print(f'\n######## ES diag {mol} ########')
    rho_cube = os.path.join(cube_dir, 'rho_N.cube')
    if not os.path.isfile(rho_cube):
        raise FileNotFoundError(rho_cube)
    d_cube = afm_utils.get_density_from_cube(rho_cube, use_esp_cube=False, verbosity=1)
    atomPos = np.asarray(d_cube['atomPos'], dtype=np.float64)
    atomTypes = np.array([int(round(z)) for z in d_cube['atomZ']], dtype=np.int32)

    # ── Native: Gaussian Δρ (legacy get_density_from_cube) vs clamp→compact ──
    V_gauss = np.asarray(d_cube['V_ES'], dtype=np.float32)
    lines_nat, png_nat = afm_utils.plot_cube_es_chain_diag(
        d_cube['rho_scf'], d_cube['rho_diff'], V_gauss, None, None, None,
        d_cube['origin'], d_cube['step'], atomPos,
        os.path.join(outdir, 'es_chain_native_gaussNA.png'),
        z_above=z_above, title=f'{mol} NATIVE + Gaussian NA (LEGACY — wrong)',
        rho_na=d_cube.get('rho_na'))

    clamp = afm_utils.delta_rho_clamp_compact_na(
        d_cube['rho_scf'], d_cube['origin'], d_cube['step'],
        atomPos, d_cube['atomZ'], rc_na=0.6, R_sphere=0.6)
    V_clamp_nat = afm.fft_poisson_cpu(clamp['rho_diff'], d_cube['step'])
    lines_cl, png_cl = afm_utils.plot_cube_es_chain_diag(
        clamp['rho_scf_clamped'], clamp['rho_diff'], V_clamp_nat, None, None, None,
        d_cube['origin'], d_cube['step'], atomPos,
        os.path.join(outdir, 'es_chain_native_clamp_compact.png'),
        z_above=z_above, title=f'{mol} NATIVE clamp→element-invariant compact NA',
        rho_na=clamp['rho_na'])

    # ── Canonical NA-origin bisect (preserve dipole_origin_bisect.png) ──
    lines_orig, png_orig = [], None
    rho_na_path = os.path.join(cube_dir, 'rho_NA.cube')
    if os.path.isfile(rho_na_path):
        from spammm.quantum.DFTB.DFTBplusParser import read_cube
        rho_na_b, _, _, _, _ = read_cube(rho_na_path)
        rho_na_cube = (rho_na_b / (afm_utils.BOHR_TO_ANG ** 3)).astype(np.float64)
        if rho_na_cube.shape != np.asarray(d_cube['rho_scf']).shape:
            raise ValueError(
                f'{mol}: rho_NA.cube shape {rho_na_cube.shape} != '
                f'rho_N {np.asarray(d_cube["rho_scf"]).shape}')
        lines_orig, png_orig = afm_utils.plot_cube_delta_rho_na_origin_diag(
            d_cube['rho_scf'], d_cube['origin'], d_cube['step'],
            atomPos, d_cube['atomZ'],
            os.path.join(outdir, 'dipole_origin_bisect.png'),
            rho_na_cube=rho_na_cube, z_above=float(z_above[0]),
            title=f'{mol} NATIVE: where does Δρ dipole come from? (NO manual dipole strip)')
        open(os.path.join(outdir, 'DIPOLE_ORIGIN.out'), 'w').write('\n'.join(lines_orig) + '\n')
        print(f'REVIEW: {os.path.join(outdir, "DIPOLE_ORIGIN.out")}')
    else:
        print(f'[{mol}] skip dipole_origin_bisect (no {rho_na_path})')

    # ── FDBM dest: pyridine SSOT project (not scipy sample) ──
    grid_spec, origin, ngrid, step = _fft_friendly_grid_spec(
        atomPos, args.step, args.margin, z_extra=6.0)
    nx, ny, nz = [int(x) for x in ngrid]
    prep = afm_utils.allelectron_cube_to_fdbm_grid(
        d_cube['rho_scf'], d_cube['origin'], d_cube['step'], atomPos, d_cube['atomZ'],
        origin, step, ngrid, rc_na=0.6, R_sphere=0.6, verbosity=1)
    V_cube = afm.fft_poisson_cpu(prep['rho_diff'], step)

    tip_tot, tip_del = afm_utils.get_tip_densities(
        tip_mode='co', target_shape=(nx, ny, nz), step=step, margin=args.margin,
        basis=getattr(args, 'basis', '3ob-3-1'), output_dir=outdir, backend='dftb')
    tip_info = _plot_tip(tip_tot, tip_del, outdir, 'co', step)
    E_ES = afm.compute_es_conv_field(V_cube, tip_del, step, tip_rolled=True, return_grads=False)

    print(f'[{mol}] DFTB stock V_ES control...')
    basis_hsd = get_dftb_basis_path(getattr(args, 'basis', '3ob-3-1'))
    res_stock = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, os.path.join(outdir, 'dftb_work_stock'),
        grid_spec=grid_spec, step=step, verbosity=0)
    V_stock = res_stock['V_ES']
    E_ES_stock = afm.compute_es_conv_field(V_stock, tip_del, step, tip_rolled=True, return_grads=False)

    lines_f, png_f = afm_utils.plot_cube_es_chain_diag(
        prep['rho_scf'], prep['rho_diff'], V_cube, E_ES, tip_tot, tip_del,
        origin, step, atomPos,
        os.path.join(outdir, 'es_chain_fdbm_grid.png'),
        z_above=z_above,
        title=f'{mol} FDBM: clamp→element-invariant compact + cube-node project + DFTB V',
        compare_VES=V_stock, compare_label='DFTB V_ES')

    com = atomPos.mean(0)
    qg, p_gauss = grid_moments(d_cube['rho_diff'], d_cube['origin'], d_cube['step'])
    qc, p_cl = grid_moments(clamp['rho_diff'], d_cube['origin'], d_cube['step'])
    lines_stock_E = []
    mol_z = float(atomPos[:, 2].mean())
    z_coords = origin[2] + np.arange(nz) * step
    ix_c = float(np.clip((com[0] - origin[0]) / step, 0, nx - 1))
    iy_c = float(np.clip((com[1] - origin[1]) / step, 0, ny - 1))
    for za in z_above:
        iz = int(np.clip(np.argmin(np.abs(z_coords - (mol_z + za))), 0, nz - 1))
        for name, field in [('E_ES cube SSOT', E_ES), ('E_ES DFTB', E_ES_stock)]:
            sl = field[:, :, iz]
            mx = afm_utils.mirror_asymmetry_2d(sl, axis=0, center=ix_c)
            my = afm_utils.mirror_asymmetry_2d(sl, axis=1, center=iy_c)
            lines_stock_E.append(f'XY z+{za:.1f}  {name:16s}  mX={mx:.3e}  mY={my:.3e}')

    all_lines = [
        f'ES diag {mol}',
        f'cube_dir={cube_dir}',
        'SSOT: V_ES=fft_poisson(Δρ); Δρ=clamp→element-invariant compact_NA; cube-node project (NOT scipy sample)',
        f'LEGACY Gauss Δρ q={qg:.3e} |pxy|={np.hypot(p_gauss[0]-qg*com[0], p_gauss[1]-qg*com[1]):.3e}',
        f'CLAMP Δρ q={qc:.3e} |pxy|={np.hypot(p_cl[0]-qc*com[0], p_cl[1]-qc*com[1]):.3e}',
        f'project Δρ q={prep["q_diff"]:.3e} p_diff={prep["p_diff"]}',
        f'FDBM grid={nx}x{ny}x{nz} step={step}',
        f'tip peak={tip_info["peak"]} mX={tip_info["mirrorX"]:.3e} mY={tip_info["mirrorY"]:.3e}',
        '',
        '=== native Gaussian NA (LEGACY) ===',
        *lines_nat,
        '',
        '=== native clamp→compact (SSOT) ===',
        *lines_cl,
        '',
        '=== Δρ / NA origin bisect (canonical; see doc/Caveats.md) ===',
        *(lines_orig if lines_orig else ['(skipped — no rho_NA.cube)']),
        '',
        '=== FDBM clamp+project (panel path) ===',
        *lines_f,
        '',
        '=== E_ES cube vs DFTB ===',
        *lines_stock_E,
        f'REVIEW: {png_nat}',
        f'REVIEW: {png_cl}',
        *( [f'REVIEW: {png_orig}'] if png_orig else [] ),
        f'REVIEW: {png_f}',
        f'REVIEW: {tip_info["path"]}',
    ]
    sum_path = os.path.join(outdir, 'ES_ASYM.out')
    open(sum_path, 'w').write('\n'.join(all_lines) + '\n')
    print(f'REVIEW: {sum_path}')
    return all_lines


def run_fukui_es_diag(args):
    """ES asymmetry diagnostics for Fukui panel molecules (or --molecule subset)."""
    mols = FUKUI_PANEL
    if getattr(args, 'molecule', None):
        want = set(args.molecule)
        mols = [(m, x) for m, x in FUKUI_PANEL if m in want]
        missing = want - {m for m, _ in mols}
        if missing:
            raise ValueError(f'Unknown --molecule {missing}; choose from {[m for m, _ in FUKUI_PANEL]}')
    os.makedirs(args.outdir, exist_ok=True)
    panel_lines = [f'Fukui ES diag  outdir={args.outdir}', '']
    for mol, xyz in mols:
        run_fukui_es_diag_one(mol, xyz, args)
        panel_lines.append(f'REVIEW: {os.path.join(args.outdir, mol, "es_diag", "es_chain_fdbm_grid.png")}')
        panel_lines.append(f'REVIEW: {os.path.join(args.outdir, mol, "es_diag", "dipole_origin_bisect.png")}')
        panel_lines.append(f'REVIEW: {os.path.join(args.outdir, mol, "es_diag", "ES_ASYM.out")}')
    panel_sum = os.path.join(args.outdir, 'SUMMARY_es_diag.out')
    open(panel_sum, 'w').write('\n'.join(panel_lines) + '\n')
    print(f'\nREVIEW: {panel_sum}')


def main():
    parser = argparse.ArgumentParser(description="FDBM diagnostic: tip_mode gaussian|co|both")
    parser.add_argument('--xyz', default='data/xyz/benzene.xyz')
    parser.add_argument('--basis', default='mio-1-1')
    parser.add_argument('--step', type=float, default=0.15)
    parser.add_argument('--margin', type=float, default=4.0)
    parser.add_argument('--tip-mode', default='both', choices=['gaussian', 'co', 'both'])
    parser.add_argument('--outdir', default='debug/afm_fdbm_diag')
    parser.add_argument('--K_LAT', type=float, default=0.5,
                        help='Lateral PP stiffness in N/m (GUI units). Converted to eV/Å² internally. Hapala default 0.5 N/m ≈ 0.031 eV/Å². Do NOT pass eV/Å² here.')
    parser.add_argument('--K_RAD', type=float, default=20.0, help='Radial stiffness [eV/Å²]')
    parser.add_argument('--bond_length', type=float, default=3.0, help='CO tip bond length [Å]')
    parser.add_argument('--h_min', type=float, default=2.0)
    parser.add_argument('--h_max', type=float, default=5.5)
    parser.add_argument('--h_step', type=float, default=0.25)
    parser.add_argument('--amp', type=float, default=1.0)
    parser.add_argument('--scan_margin', type=float, default=2.0)
    parser.add_argument('--height', type=float, default=2.5, help='XY stage slice height above mol [Å]')
    parser.add_argument('--cmap', default='seismic')
    parser.add_argument('--df_cmap', default='gray')
    parser.add_argument('--ptcda-stock-vs-sa', action='store_true',
                        help='PTCDA: stock 3ob vs SA-prolonged with Ez-fitted Pauli A,β')
    parser.add_argument('--sa-params', default='debug/dftb_basis_sa_ptcda/PTCDA_sa_params.json')
    parser.add_argument('--fukui-panel', action='store_true',
                        help='Fukui pySCF cubes: cube vs DFTB stock vs prolonged Slater-tail')
    parser.add_argument('--molecule', nargs='*', default=None,
                        help='Subset of Fukui panel names (with --fukui-panel)')
    parser.add_argument('--replot-fukui-per-image', action='store_true',
                        help='Replot existing Fukui panel npz with per-image color scale into */per_image/')
    parser.add_argument('--fukui-es-diag', action='store_true',
                        help='Cube ES chain diagnostics (ρ, Δρ, V_ES, E_ES, tip) + mirror metrics')
    parser.add_argument('--z-above', nargs=2, type=float, default=[1.0, 5.0],
                        help='ES-diag slice heights above mol [Å]')
    args = parser.parse_args()

    if args.fukui_es_diag:
        args.basis = '3ob-3-1'
        if args.outdir == 'debug/afm_fdbm_diag':
            args.outdir = 'debug/fdbm_fukui_panel_flat'
        args.z_above = tuple(args.z_above)
        run_fukui_es_diag(args)
        return

    if args.replot_fukui_per_image:
        out = args.outdir if args.outdir != 'debug/afm_fdbm_diag' else 'debug/fdbm_fukui_panel'
        replot_fukui_per_image(out, molecules=args.molecule, cmap=args.cmap, df_cmap=args.df_cmap)
        return

    if args.fukui_panel:
        args.basis = '3ob-3-1'
        args.tip_mode = 'co'
        if args.outdir == 'debug/afm_fdbm_diag':
            args.outdir = 'debug/fdbm_fukui_panel'
        if args.h_min == 2.0 and args.h_max == 5.5 and args.h_step == 0.25:
            args.h_min, args.h_max, args.h_step = 2.5, 5.7, 0.4
        run_fukui_panel(args)
        return

    if args.ptcda_stock_vs_sa:
        args.xyz = args.xyz if 'PTCDA' in args.xyz else 'data/xyz/PTCDA.xyz'
        args.basis = '3ob-3-1'
        args.tip_mode = 'co'
        if args.outdir == 'debug/afm_fdbm_diag':
            args.outdir = 'debug/fdbm_ptcda_stock_vs_sa'
        # ~8 heights in typical CO-AFM contrast window
        if args.h_min == 3.0 and args.h_max == 5.5 and args.h_step == 0.5:
            args.h_min, args.h_max, args.h_step = 2.5, 5.7, 0.4
        run_ptcda_stock_vs_sa(args)
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.outdir, exist_ok=True)
    modes = ['gaussian', 'co'] if args.tip_mode == 'both' else [args.tip_mode]
    results = {}
    lines = []
    lines.append(f"FDBM diagnostic  xyz={args.xyz}  basis={args.basis}  step={args.step}")
    lines.append(f"outdir={os.path.abspath(args.outdir)}")
    lines.append("")

    for mode in modes:
        results[mode] = _run_one(args.xyz, args.basis, args.step, args.margin, mode, args.outdir, args)
        t = results[mode]['tip']
        lines.append(f"[{mode}] tip peak={t['peak']} q={t['q']:.4f} Δq={t['dq']:.4f} "
                     f"mirrorX={t['mirrorX']:.3e} mirrorY={t['mirrorY']:.3e}  df_asym={results[mode]['df_asym']:.3e}")

    if set(modes) == {'gaussian', 'co'}:
        # Side-by-side df at mid height
        rg, rc = results['gaussian'], results['co']
        ih = min(rg['df'].shape[2], rc['df'].shape[2]) // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, data, title in [
            (axes[0], rg['df'][:, :, ih], 'df gaussian'),
            (axes[1], rc['df'][:, :, ih], 'df co'),
            (axes[2], rc['df'][:, :, ih] - rg['df'][:, :, ih], 'df (co − gaussian)'),
        ]:
            # resample if shapes differ
            a = data
            vmax = np.percentile(np.abs(a), 99) or 1e-30
            im = ax.imshow(a.T, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
            ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(
            f"df compare @ ih={ih}  h={rg['heights'][ih]:.2f}Å\n"
            + _grid_caption(args.step, rg['df'].shape, rg.get('origin'),
                            f'tip_mode=gaussian|co  mol={os.path.basename(args.xyz)}'),
            fontsize=10)
        fig.tight_layout()
        cmp_path = os.path.join(args.outdir, 'compare_df_mid.png')
        fig.savefig(cmp_path, dpi=140); plt.close(fig)
        lines.append(f"compare: {cmp_path}")

        # Assert CO tip quality in SUMMARY
        assert results['co']['tip']['peak'] == (0, 0, 0), results['co']['tip']['peak']
        assert abs(results['co']['tip']['q'] - 10.0) < 1.5
        assert results['co']['tip']['mirrorX'] < 0.05

    summary = os.path.join(args.outdir, 'SUMMARY.out')
    with open(summary, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        f.write('\nREVIEW checklist:\n')
        f.write('  1. tip_co.png: XY@z=0 circular (fftshift center); XZ/YZ show O–C along z\n')
        f.write('  2. tip_co peak must be (0,0,0); q≈10; mirrorX << 0.05\n')
        f.write('  3. stage_*.png: Pauli/ES/vdw localized on molecule, not sheared\n')
        f.write('  4. df_co vs df_gaussian: both show molecule; CO sharper features\n')
        f.write('  5. compare_df_mid.png: difference map localized, not a global tilt\n')
    print('\n'.join(lines))
    print(f"\nREVIEW: {summary}")
    print(f"REVIEW: {os.path.abspath(args.outdir)}/")
    print(f"\nDone. Open folder: {os.path.abspath(args.outdir)}")

if __name__ == '__main__':
    main()
