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
    import matplotlib.pyplot as plt
    peak = np.unravel_index(int(np.argmax(np.abs(rho))), rho.shape)
    q = float(rho.sum() * step**3)
    qd = float(rho_d.sum() * step**3)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # After roll: O at (0,0,0) — show wrapped neighborhoods via fftshift for display
    def show_wrap(ax, sl, title):
        im = ax.imshow(np.fft.fftshift(sl).T, origin='lower', cmap='magma')
        ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046)
    show_wrap(axes[0, 0], rho[:, :, 0], f'{tag} total XY@z=0 (fftshift)')
    show_wrap(axes[0, 1], rho[:, 0, :], f'{tag} total XZ@y=0')
    show_wrap(axes[0, 2], rho[0, :, :], f'{tag} total YZ@x=0')
    show_wrap(axes[1, 0], rho_d[:, :, 0], f'{tag} delta XY@z=0')
    show_wrap(axes[1, 1], rho_d[:, 0, :], f'{tag} delta XZ@y=0')
    show_wrap(axes[1, 2], rho_d[0, :, :], f'{tag} delta YZ@x=0')
    mx = _mirror_asym(np.fft.fftshift(rho[:, :, 0]), 0)
    my = _mirror_asym(np.fft.fftshift(rho[:, :, 0]), 1)
    fig.suptitle(
        f'Tip {tag}: peak={peak} q={q:.3f} Δq={qd:.4f}  XY mX={mx:.2e} mY={my:.2e}\n'
        + _grid_caption(step, rho.shape),
        fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, f'tip_{tag}.png')
    fig.savefig(path, dpi=140); plt.close(fig)
    return {'peak': peak, 'q': q, 'dq': qd, 'mirrorX': mx, 'mirrorY': my, 'path': path}

def _plot_stages(fields, origin, step, atomPos, outdir, tag, z_above=2.5):
    import matplotlib.pyplot as plt
    mol_z = float(atomPos[:, 2].max())
    nz = next(iter(fields.values())).shape[2]
    z_coords = origin[2] + np.arange(nz) * step
    iz = int(np.clip(np.argmin(np.abs(z_coords - (mol_z + z_above))), 0, nz - 1))
    cx = int(np.clip(round((atomPos[:, 0].mean() - origin[0]) / step), 0, fields['rho_scf'].shape[0] - 1))
    names = ['rho_scf', 'E_pauli', 'E_ES', 'E_vdw', 'E_total', 'Fz']
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.5))
    for col, name in enumerate(names):
        data = fields[name]
        xy = data[:, :, iz]
        xz = data[:, data.shape[1] // 2, :]
        for row, (sl, lab) in enumerate([(xy, f'XY z={z_coords[iz]-mol_z:.1f}Å'), (xz, 'XZ mid-y')]):
            ax = axes[row, col]
            vmax = np.percentile(np.abs(sl), 99) or 1e-30
            cmap = 'magma' if name.startswith('rho') or name == 'E_pauli' else 'seismic'
            if name == 'E_vdw':
                im = ax.imshow(sl.T, origin='lower', cmap='viridis')
            else:
                im = ax.imshow(sl.T, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_title(f'{name}\n{lab}', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
            if row == 0:
                ax.text(0.02, 0.98, f'mX={_mirror_asym(sl,0):.2e}', transform=ax.transAxes,
                        va='top', fontsize=7, color='w',
                        bbox=dict(boxstyle='round', fc='k', alpha=0.4))
    shape = next(iter(fields.values())).shape
    fig.suptitle(
        f'Stages tip_mode={tag}  (iz={iz}, cx={cx})\n' + _grid_caption(step, shape, origin),
        fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, f'stage_{tag}.png')
    fig.savefig(path, dpi=130); plt.close(fig)
    return path

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


def _run_from_density(tag, rho_scf, V_ES, atomPos, atomTypes, origin, step, ngrid, A, beta, tip_mode, outdir, args):
    """PP-AFM scan from precomputed ρ / V_ES with explicit Pauli A,β.

    Display heights = [h_min, h_max] step h_step. PP scan covers ±amp so ``compute_df_amp``
    is well-defined at the display edges. Plots gated by ``args._plots`` / ``args.plots``.
    """
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils

    plots = getattr(args, '_plots', None)
    if plots is None:
        raw = getattr(args, 'plots', None)
        if raw is None:
            plots = {'compare', 'stage', 'tip', 'df', 'fz'}  # diagnostic script defaults: keep old richness
        else:
            parts = [p.strip().lower() for p in str(raw).replace(';', ',').split(',') if p.strip()]
            if 'all' in parts or 'debug' in parts:
                plots = {'compare', 'stage', 'tip', 'df', 'fz', 'per_image'}
            elif 'none' in parts or 'off' in parts:
                plots = set()
            else:
                plots = set(parts)

    nx, ny, nz = rho_scf.shape
    mol_z = float(atomPos[:, 2].max())
    sub = os.path.join(outdir, tag)
    os.makedirs(sub, exist_ok=True)
    print(f"\n=== {tag}  A={A:.3f} β={beta:.4f}  tip={tip_mode}  grid={nx}x{ny}x{nz} ===")

    rho_tip, rho_tip_d = afm_utils.get_tip_densities(
        tip_mode=tip_mode, target_shape=(nx, ny, nz), step=step, margin=args.margin,
        basis=args.basis, output_dir=sub, backend='dftb',
    )
    tip_info = _plot_tip(rho_tip, rho_tip_d, outdir, tag, step) if 'tip' in plots else {
        'peak': None, 'q': float(rho_tip.sum() * step**3), 'dq': float(rho_tip_d.sum() * step**3),
        'mirrorX': None, 'mirrorY': None, 'path': None,
    }

    overlap = afm.compute_pauli_overlap(rho_scf, rho_tip, step, tip_rolled=True)
    E_pauli = afm.scale_pauli_field(overlap, step, A, beta, return_grads=False)
    if V_ES is not None:
        E_ES = afm.compute_es_conv_field(V_ES, rho_tip_d, step, tip_rolled=True, return_grads=False)
    else:
        E_ES = np.zeros_like(E_pauli)
        print(f"  WARNING: no V_ES — ES term zeroed")
    E_vdw = afm.compute_dispersion_grid(atomPos, atomTypes, origin, step, ngrid, C6_CO=30.0, return_grads=False)
    E_total = E_pauli + E_ES + E_vdw
    afmulator = afm.AFMulator(use_morse=False, nloc=32)
    F_total = afmulator.compute_gradient_cl(E_total, step, bAlloc=True)

    fields = {
        'rho_scf': rho_scf, 'E_pauli': E_pauli, 'E_ES': E_ES, 'E_vdw': E_vdw,
        'E_total': E_total, 'Fz': -F_total[..., 2],
    }
    stage_path = _plot_stages(fields, origin, step, atomPos, outdir, tag, z_above=args.height) if 'stage' in plots else None

    scan_xs = np.arange(float(atomPos[:, 0].min() - args.scan_margin),
                        float(atomPos[:, 0].max() + args.scan_margin), step, dtype=np.float32)
    scan_ys = np.arange(float(atomPos[:, 1].min() - args.scan_margin),
                        float(atomPos[:, 1].max() + args.scan_margin), step, dtype=np.float32)
    # Display window = df probe heights; optionally amp-align Fz at h−amp
    def _h_stack(h0, h1, dz):
        n = int(round((h1 - h0) / dz)) + 1
        return (h0 + np.arange(n, dtype=np.float64) * dz).astype(np.float32)

    amp = float(args.amp)
    amp_align = not bool(getattr(args, 'no_amp_align', False))
    h_df = _h_stack(args.h_min, args.h_max, args.h_step)
    h_Fz = (h_df.astype(np.float64) - amp).astype(np.float32) if amp_align else h_df
    # PP scan must cover df window ±amp (and thus also Fz@h−amp when aligned)
    h_scan = _h_stack(float(h_df[0]) - amp, float(h_df[-1]) + amp, args.h_step)
    if len(h_scan) < 3:
        raise ValueError(f'Need denser heights for df amp={amp}: got {len(h_scan)} scan points')
    K_LAT_eV = afm.stiffness_Nm_to_eVA2(args.K_LAT)  # CLI: N/m → internal eV/Å²
    print(f"  K_LAT={args.K_LAT:.3f} N/m → {K_LAT_eV:.4f} eV/Å²  K_RAD={args.K_RAD} eV/Å²  L={args.bond_length} Å")
    print(f"  df h=[{float(h_df[0]):.2f},{float(h_df[-1]):.2f}]  "
          f"Fz h=[{float(h_Fz[0]):.2f},{float(h_Fz[-1]):.2f}]  "
          f"scan=[{float(h_scan[0]):.2f},{float(h_scan[-1]):.2f}] dz={args.h_step} amp={amp} align={amp_align}")
    afmulator.setup_fdbm_grid(F_total, origin, step)
    FEs, tip_disp = afmulator.scan_fdbm(
        scan_xs, scan_ys, h_scan, mol_z=mol_z,
        K_LAT=K_LAT_eV, K_RAD=args.K_RAD, bond_length=args.bond_length,
        ppm_mode=True, use_fire=True,
    )
    Fz_full = FEs[:, :, :, 2]
    dz = float(h_scan[1] - h_scan[0])
    df_full = afm.compute_df_amp(Fz_full, dz, amp=amp)
    idx_df = [int(np.argmin(np.abs(h_scan - h))) for h in h_df]
    idx_Fz = [int(np.argmin(np.abs(h_scan - h))) for h in h_Fz]
    Fz = Fz_full[:, :, idx_Fz]
    df = df_full[:, :, idx_df]
    heights = h_df  # column labels = df probe height
    x_ext = [float(scan_xs[0]), float(scan_xs[-1])]
    y_ext = [float(scan_ys[0]), float(scan_ys[-1])]
    if 'df' in plots:
        afm_utils.plot_grid_Fz(df, heights, f'df {tag} A={A:.2f} β={beta:.3f}', f'df_{tag}.png',
                               x_ext=x_ext, y_ext=y_ext, save_dir=outdir, cmap=args.df_cmap)
    if 'fz' in plots:
        afm_utils.plot_grid_Fz(Fz, h_Fz, f'Fz {tag} A={A:.2f} β={beta:.3f}', f'Fz_{tag}.png',
                               x_ext=x_ext, y_ext=y_ext, save_dir=outdir, cmap=args.cmap)
    ih = len(heights) // 2
    print(f"  df @h={heights[ih]:.2f} / Fz @h={float(h_Fz[ih]):.2f}: "
          f"df=[{df.min():.3e},{df.max():.3e}] Fz=[{Fz.min():.3e},{Fz.max():.3e}]")
    if stage_path:
        print(f"  REVIEW: {stage_path}")
    return {
        'tip': tip_info, 'stage_path': stage_path, 'df': df, 'Fz': Fz, 'heights': heights,
        'heights_Fz': h_Fz, 'amp_align': amp_align,
        'scan_xs': scan_xs, 'scan_ys': scan_ys, 'atomPos': atomPos, 'origin': origin, 'step': step,
        'A': A, 'beta': beta, 'tag': tag, 'h_scan': h_scan,
    }


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
            pa['A'], pa['beta'], 'co', args.outdir, args)
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


# Fukui pySCF PBE/def2-SVP panel — cube FDBM vs DFTB stock vs prolonged
# USER 2026-07-24: geometries enforced flat (z=0); cubes under FUKUI_CUBE_ROOTS
# USER 2026-07-27: new H-bonded dimers under …/Fukui_AFM/new (xyz always; cubes when present)
FUKUI_CUBE_ROOTS = [
    '/home/prokop/SIMULATIONS/Fukui_AFM/new',
    '/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster',
]
FUKUI_CUBE_ROOT = FUKUI_CUBE_ROOTS[0]  # preferred / error-message default
_NEW = '/home/prokop/SIMULATIONS/Fukui_AFM/new'
FUKUI_PANEL = [
    ('adenine-uracil', f'{_NEW}/adenine-uracil_PBE_def2-SVP/adenine-uracil_opt.xyz'),
    ('adenine-uracil-iso', f'{_NEW}/adenine-uracil-iso_PBE_def2-SVP/adenine-uracil-iso_opt.xyz'),
    ('azaindol_dimer', f'{_NEW}/azaindol_dimer_PBE_def2-SVP/azaindol_dimer_opt.xyz'),
    ('azaindol_isodimer', f'{_NEW}/azaindol_isodimer_PBE_def2-SVP/azaindol_isodimer_opt.xyz'),
    ('benzoicacid_dimer', 'data/xyz/benzoicacid_dimer.xyz'),
    ('benzoicamid_dimer', 'data/xyz/benzoicamid_dimer.xyz'),
    ('pentacene', 'data/xyz/pentacene.xyz'),
    ('PTCDA', 'data/xyz/PTCDA.xyz'),
    ('phtalo_1-dftb-relax', 'data/xyz/phtalo_1.xyz'),
    ('phtalo_2-dftb-relax', 'data/xyz/phtalo_2.xyz'),
]


def _fukui_cube_dir(mol):
    """First root that has rho_N.cube for ``<mol>_PBE_def2-SVP``, else preferred path."""
    tag = f'{mol}_PBE_def2-SVP'
    for root in FUKUI_CUBE_ROOTS:
        d = os.path.join(root, tag)
        if os.path.isfile(os.path.join(d, 'rho_N.cube')):
            return d
    return os.path.join(FUKUI_CUBE_ROOTS[0], tag)


def _fft_friendly_grid_spec(atomPos, step, margin, z_extra):
    """FDBM grid for Fukui panel: COM-centered XY, z-symmetric vacuum (FFT-friendly ngrid).

    ``z_extra`` is interpreted as half-span vacuum above AND below the mol plane
    (total Lz ≈ 2*z_extra), NOT the old setup_density_grid one-sided +z pad.
    """
    from spammm.SPM import AFM_utils as afm_utils
    z_vac = float(z_extra) if z_extra is not None else 7.0
    return afm_utils.make_fdbm_grid_com_zsym(atomPos, step, margin, z_vac=z_vac)


def run_fukui_one(mol, xyz_rel, args):
    """One molecule: DFT-cube FDBM + DFTB stock 3ob + DFTB Slater-tail prolonged (dual ES)."""
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import (
        parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list,
    )
    import spammm.atomicUtils as au

    os.environ['SPAMMM_AFM_CPU_FFT'] = '1'
    _ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cube_dir = _fukui_cube_dir(mol)
    xyz = xyz_rel if os.path.isabs(xyz_rel) else os.path.join(_ROOT, xyz_rel)
    outdir = os.path.join(args.outdir, mol)
    os.makedirs(outdir, exist_ok=True)

    ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
    pos, _, names, _, _ = au.load_xyz(xyz)
    atomPos_xyz = np.array(pos, dtype=np.float64)
    enames = list(names)
    atomTypes_xyz = np.array([ELEM_Z.get(e, 6) for e in enames], dtype=np.int32)

    lines = [
        f'Fukui FDBM panel: {mol}',
        f'cube_dir={cube_dir}',
        f'xyz={xyz}',
        f'step={args.step}  tip=co  basis=3ob-3-1',
        'variants: cube (pySCF ρ_N) | stock 3ob | prolonged Slater-tail (Pauli only; ES=stock)',
        '',
    ]
    print(f'\n######## {mol} ########')

    # ── Cube density (atoms from cube header = density frame) ──
    # All-electron SSOT (pyridine): clamp→compact NA + GridsOCL project — NOT Gaussian NA
    # and NOT scipy resample_field_to_grid (breaks ∫Δρ / dipole).
    rho_cube = os.path.join(cube_dir, 'rho_N.cube')
    has_cube = os.path.isfile(rho_cube)
    if has_cube:
        d_cube = afm_utils.get_density_from_cube(rho_cube, use_esp_cube=False, verbosity=0)
        atomPos = np.asarray(d_cube['atomPos'], dtype=np.float64)
        atomZ = np.asarray(d_cube['atomZ'], dtype=np.float64)
        atomTypes = np.array([int(round(z)) for z in atomZ], dtype=np.int32)
        if len(atomPos) != len(atomPos_xyz):
            print(f'  WARNING: cube natom={len(atomPos)} != xyz {len(atomPos_xyz)}; using cube')
    else:
        print(f'  WARNING: no rho_N.cube under {cube_dir} — DFTB stock+prolonged only (no DFT cube row)')
        lines.append(f'WARNING: missing {rho_cube} — skipped DFT cube variant')
        atomPos = atomPos_xyz
        atomTypes = atomTypes_xyz
        atomZ = atomTypes.astype(np.float64)
        d_cube = None

    grid_spec, origin, ngrid, step = _fft_friendly_grid_spec(
        atomPos, args.step, args.margin, z_extra=6.0)
    nx, ny, nz = [int(x) for x in ngrid]
    lines.append(f'grid={nx}x{ny}x{nz} step={step} origin={origin}')

    rho_cube_g = V_ES_cube = None
    if has_cube:
        lines.append('cube ES: clamp→element-invariant compact_NA + cube-node GridsOCL.project (no dipole strip)')
        cube_prep = afm_utils.allelectron_cube_to_fdbm_grid(
            d_cube['rho_scf'], d_cube['origin'], d_cube['step'], atomPos, atomZ,
            origin, step, ngrid, rc_na=0.6, R_sphere=0.6, verbosity=0)
        rho_cube_g = cube_prep['rho_scf']
        rho_diff_g = cube_prep['rho_diff']
        dV = step ** 3
        V_ES_cube = afm.fft_poisson_cpu(rho_diff_g, step)
        lines.append(
            f'cube q_scf={float(rho_cube_g.sum()*dV):.2f} q_diff={cube_prep["q_diff"]:.3e} '
            f'p_diff={cube_prep["p_diff"]}  (native ∫Δρ={cube_prep["clamp"]["Q_diff"]:.3e})')
    else:
        dV = step ** 3
        lines.append('cube ES: SKIPPED (no rho_N.cube)')
    # ── DFTB stock on same grid ──
    basis_hsd = get_dftb_basis_path('3ob-3-1')
    args.basis = '3ob-3-1'
    work_stock = os.path.join(outdir, 'dftb_work_stock')
    print(f'\n[{mol}] DFTB stock 3ob...')
    res_stock = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, work_stock, grid_spec=grid_spec,
        step=step, verbosity=0)
    rho_stock = res_stock['rho_scf']
    V_ES = res_stock['V_ES']
    dens_dir = os.path.join(_ROOT, 'debug', 'densities')
    os.makedirs(dens_dir, exist_ok=True)
    np.save(os.path.join(dens_dir, f'rho_{mol}_dftb_3ob.npy'), rho_stock)
    np.savez(os.path.join(dens_dir, f'rho_{mol}_dftb_3ob.meta.npz'),
             origin=origin, ngrid=ngrid, step=step,
             atom_pos=atomPos, atom_names=np.array(enames[:len(atomPos)] if len(enames) >= len(atomPos) else enames))
    lines.append(f'stock q_scf={float(rho_stock.sum()*dV):.2f}')

    # ── Prolonged Slater-tail Pauli (dual basis: stock V_ES) ──
    basis_data = parse_wfc_hsd(basis_hsd)
    basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
    proj_prolonged = make_slater_tail_species_list(basis_ang)
    work_prol = os.path.join(outdir, 'dftb_work_prolonged')
    print(f'\n[{mol}] DFTB prolonged Slater-tail (Pauli ρ)...')
    res_prol = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, work_prol, grid_spec=grid_spec,
        step=step, verbosity=0, projection_basis_ang=proj_prolonged)
    rho_prol = res_prol['rho_scf']
    np.save(os.path.join(dens_dir, f'rho_{mol}_dftb_3ob_prolonged.npy'), rho_prol)
    np.savez(os.path.join(dens_dir, f'rho_{mol}_dftb_3ob_prolonged.meta.npz'),
             origin=origin, ngrid=ngrid, step=step, atom_pos=atomPos)
    lines.append(f'prolonged q_scf={float(rho_prol.sum()*dV):.2f} (NOT charge-normalized; Pauli only)')

    # EVALUATION SSOT: one transferable (A,β) for all molecules and all ρ rows.
    # Do NOT use PTCDA_PAULI_FIT / per-mol fits here — those are for fitting scripts only
    # (run_ptcda_stock_vs_sa). See doc/AGENTS/skills/afm-plotting/SKILL.md § Pauli A,β.
    basis_key = getattr(args, 'basis', None) or '3ob-3-1'
    pa = dict(afm.PAULI_FITTED_DEFAULTS.get(basis_key, afm.PAULI_FITTED_DEFAULTS['3ob-3-1']))
    A, beta = float(pa['A']), float(pa['beta'])
    lines.append(f'Pauli EVAL defaults ({basis_key}): A={A:.3f} β={beta:.4f}  '
                 f'(same for cube|stock|prolonged; no per-molecule override)')

    variants = {}
    specs = []
    if has_cube:
        specs.append(('cube', rho_cube_g, V_ES_cube, A, beta))
    specs += [
        ('stock', rho_stock, V_ES, A, beta),
        ('prolonged', rho_prol, V_ES, A, beta),
    ]
    for key, rho, Ves, A_i, beta_i in specs:
        r = _run_from_density(key, rho, Ves, atomPos, atomTypes, origin, step, ngrid,
                              A_i, beta_i, 'co', outdir, args)
        variants[key] = r
        lines.append(f'[{key}] A={A_i:.3f} β={beta_i:.4f}  df=[{r["df"].min():.3e},{r["df"].max():.3e}]')

    heights = variants['stock']['heights']
    # Row order: DFT cube → prolonged (DFT-like tails) → stock (short-range)
    row_specs = []
    if has_cube:
        row_specs += [
            ('df', 'cube', f'df  DFT cube\nA={A:.1f} β={beta:.2f}', args.df_cmap),
        ]
    row_specs += [
        ('df', 'prolonged', f'df  prolonged\nA={A:.1f} β={beta:.2f}', args.df_cmap),
        ('df', 'stock', f'df  stock 3ob\nA={A:.1f} β={beta:.2f}', args.df_cmap),
    ]
    if has_cube:
        row_specs.append(('Fz', 'cube', 'Fz  DFT cube', args.cmap))
    row_specs += [
        ('Fz', 'prolonged', 'Fz  prolonged', args.cmap),
        ('Fz', 'stock', 'Fz  stock 3ob', args.cmap),
    ]
    title = (f'{mol} FDBM CO tip | '
             + ('DFT cube → ' if has_cube else '(no cube) ')
             + f'prolonged → stock 3ob\n'
             f'DUAL BASIS: prolonged ρ → Pauli only; ES = stock Δρ'
             + ('  |  PBE/def2-SVP cubes' if has_cube else '  |  DFTB-only (rho_N.cube missing)'))
    amp = float(getattr(args, 'amp', 1.0) or 1.0)
    amp_align = not bool(getattr(args, 'no_amp_align', False))
    cmp_name = 'compare_cube_stock_prolonged.png' if has_cube else 'compare_stock_prolonged.png'
    cmp = os.path.join(outdir, cmp_name)
    afm_utils.plot_afm_variant_height_strip(
        variants, row_specs, heights, cmp, scale='per_column', title=title,
        dpi=140, amp=amp, amp_align=amp_align)
    lines.append(f'REVIEW: {cmp}')
    per_dir = os.path.join(outdir, 'per_image')
    os.makedirs(per_dir, exist_ok=True)
    cmp_pi = os.path.join(per_dir, cmp_name)
    afm_utils.plot_afm_variant_height_strip(
        variants, row_specs, heights, cmp_pi, scale='per_image', title=title,
        dpi=140, amp=amp, amp_align=amp_align)
    lines.append(f'REVIEW: {cmp_pi}')
    lines.append(f'height SSOT: df=[{float(heights[0]):.2f},{float(heights[-1]):.2f}] dz={args.h_step} '
                 f'amp={amp} amp_align={amp_align}  (Fz@h−amp when align)')
    save_kw = dict(
        heights=heights,
        df_stock=variants['stock']['df'], Fz_stock=variants['stock']['Fz'],
        df_prolonged=variants['prolonged']['df'], Fz_prolonged=variants['prolonged']['Fz'],
        A_stock=A, beta_stock=beta,
        A_prolonged=A, beta_prolonged=beta,
        has_cube=has_cube, cube_dir=cube_dir, xyz=xyz,
    )
    if has_cube:
        save_kw.update(
            df_cube=variants['cube']['df'], Fz_cube=variants['cube']['Fz'],
            A_cube=A, beta_cube=beta)
    np.savez(os.path.join(outdir, 'scan_cube_stock_prolonged.npz'), **save_kw)
    summary = os.path.join(outdir, 'SUMMARY.out')
    lines += ['', f'REVIEW: {summary}', f'heights={list(map(float, heights))}']
    open(summary, 'w').write('\n'.join(lines) + '\n')
    print(f'REVIEW: {summary}')
    return variants


def replot_fukui_per_image(panel_dir, molecules=None, cmap='seismic', df_cmap='gray'):
    """Replot existing scan_*.npz strips with per-image color scale (does not overwrite commons)."""
    from spammm.SPM import AFM_utils as afm_utils
    mols = molecules or [m for m, _ in FUKUI_PANEL]
    lines = [f'Replot per_image from {panel_dir}', '']
    for mol in mols:
        npz = os.path.join(panel_dir, mol, 'scan_cube_stock_prolonged.npz')
        if not os.path.isfile(npz):
            print(f'  skip {mol}: missing {npz}')
            continue
        d = np.load(npz)
        heights = d['heights']
        variants = {
            'cube': {'df': d['df_cube'], 'Fz': d['Fz_cube']},
            'stock': {'df': d['df_stock'], 'Fz': d['Fz_stock']},
            'prolonged': {'df': d['df_prolonged'], 'Fz': d['Fz_prolonged']},
        }
        A_c, b_c = float(d['A_cube']), float(d['beta_cube'])
        A_s, b_s = float(d['A_stock']), float(d['beta_stock'])
        A_p, b_p = float(d['A_prolonged']), float(d['beta_prolonged'])
        row_specs = [
            ('df', 'cube', f'df  DFT cube\nA={A_c:.1f} β={b_c:.2f}', df_cmap),
            ('df', 'prolonged', f'df  prolonged\nA={A_p:.1f} β={b_p:.2f}', df_cmap),
            ('df', 'stock', f'df  stock 3ob\nA={A_s:.1f} β={b_s:.2f}', df_cmap),
            ('Fz', 'cube', 'Fz  DFT cube', cmap),
            ('Fz', 'prolonged', 'Fz  prolonged', cmap),
            ('Fz', 'stock', 'Fz  stock 3ob', cmap),
        ]
        title = (f'{mol} FDBM CO tip | DFT cube → prolonged → stock\n'
                 f'(replot — per-image color scale)')
        per_dir = os.path.join(panel_dir, mol, 'per_image')
        os.makedirs(per_dir, exist_ok=True)
        out = os.path.join(per_dir, 'compare_cube_stock_prolonged.png')
        afm_utils.plot_afm_variant_height_strip(
            variants, row_specs, heights, out, scale='per_image', title=title, dpi=140)
        # Also refresh shared-scale strip (same row order)
        out_col = os.path.join(panel_dir, mol, 'compare_cube_stock_prolonged.png')
        afm_utils.plot_afm_variant_height_strip(
            variants, row_specs, heights, out_col, scale='per_column',
            title=f'{mol} FDBM CO tip | DFT cube → prolonged → stock', dpi=140)
        lines.append(f'REVIEW: {out_col}')
        lines.append(f'REVIEW: {out}')
    sum_path = os.path.join(panel_dir, 'SUMMARY_per_image.out')
    open(sum_path, 'w').write('\n'.join(lines) + '\n')
    print(f'REVIEW: {sum_path}')
    return lines


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


def run_fukui_panel(args):
    """Run Fukui molecule panel (all or --molecule subset)."""
    mols = FUKUI_PANEL
    if getattr(args, 'molecule', None):
        want = set(args.molecule)
        mols = [(m, x) for m, x in FUKUI_PANEL if m in want]
        missing = want - {m for m, _ in mols}
        if missing:
            raise ValueError(f'Unknown --molecule {missing}; choose from {[m for m,_ in FUKUI_PANEL]}')
    os.makedirs(args.outdir, exist_ok=True)
    panel_lines = [f'Fukui FDBM panel  outdir={args.outdir}', '']
    for mol, xyz in mols:
        run_fukui_one(mol, xyz, args)
        panel_lines.append(f'REVIEW: {os.path.join(args.outdir, mol, "compare_cube_stock_prolonged.png")}')
        panel_lines.append(f'REVIEW: {os.path.join(args.outdir, mol, "compare_stock_prolonged.png")}')
        panel_lines.append(f'REVIEW: {os.path.join(args.outdir, mol, "per_image", "compare_cube_stock_prolonged.png")}')
        panel_lines.append(f'REVIEW: {os.path.join(args.outdir, mol, "SUMMARY.out")}')
    panel_sum = os.path.join(args.outdir, 'SUMMARY.out')
    open(panel_sum, 'w').write('\n'.join(panel_lines) + '\n')
    print(f'\nREVIEW: {panel_sum}')
    print(f'REVIEW: {os.path.abspath(args.outdir)}/')


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
