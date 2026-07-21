#!/usr/bin/env python3
"""Kriging vs FDBM-from-cubes: z-alignment first, then same-XY maps.

Default pair for Pauli fit (small tip ES): pyridine ``N-h`` + ``CO_O`` (O-down).
Probe sites: N and para-C (opposite ring C); marked on XY plots; z-profiles overlaid.

Usage:
  python tests/SPM/testplot_kriging_vs_fdbm_cube.py
  python tests/SPM/testplot_kriging_vs_fdbm_cube.py --endgroup N-h --tip CO_O
  python tests/SPM/testplot_kriging_vs_fdbm_cube.py --endgroup HHO-h-p_1 --tip H2O_O
"""
from __future__ import annotations
import argparse, os, time
import numpy as np

os.environ.setdefault('SPAMMM_AFM_CPU_FFT', '1')


def _outdir(name='testplot_kriging_vs_fdbm_cube'):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'debug', name))
    os.makedirs(root, exist_ok=True)
    return root


def _iz(origin_z, step, z, nz):
    return int(np.clip(round((z - origin_z) / step), 0, nz - 1))


def _grid_bbox(origin, step, shape):
    nx, ny, nz = shape[:3]
    return dict(
        xmin=float(origin[0]), xmax=float(origin[0] + (nx - 1) * step),
        ymin=float(origin[1]), ymax=float(origin[1] + (ny - 1) * step),
        zmin=float(origin[2]), zmax=float(origin[2] + (nz - 1) * step),
        nx=int(nx), ny=int(ny), nz=int(nz), step=float(step), origin=np.asarray(origin, float),
    )


def _intersection_xy(b1, b2):
    return (max(b1['xmin'], b2['xmin']), min(b1['xmax'], b2['xmax']),
            max(b1['ymin'], b2['ymin']), min(b1['ymax'], b2['ymax']))


def _sample_line(vol, origin, step, xy, zs):
    from scipy.ndimage import map_coordinates
    ix = (xy[0] - origin[0]) / step
    iy = (xy[1] - origin[1]) / step
    out = np.empty(len(zs), dtype=np.float64)
    for i, z in enumerate(zs):
        iz = (z - origin[2]) / step
        out[i] = map_coordinates(vol, [[ix], [iy], [iz]], order=1, mode='nearest')[0]
    return out


def _xy_slice_common(vol, origin, step, z, bbox_xy, nxy, order=1):
    from scipy.ndimage import map_coordinates
    xmin, xmax, ymin, ymax = bbox_xy
    xs = np.linspace(xmin, xmax, nxy)
    ys = np.linspace(ymin, ymax, nxy)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    iz = (z - origin[2]) / step
    coords = np.stack([(X - origin[0]) / step, (Y - origin[1]) / step, np.full_like(X, iz)], 0)
    return xs, ys, map_coordinates(vol, coords, order=order, mode='constant', cval=np.nan)


def _probe_sites(apos, atomZ):
    """Prefer N + para-C (pyridine); else O+COM; else COM."""
    apos = np.asarray(apos, float)
    atomZ = np.asarray(atomZ, float)
    xy_com = apos[:, :2].mean(axis=0)
    n_mask = np.isclose(atomZ, 7)
    c_mask = np.isclose(atomZ, 6)
    o_mask = np.isclose(atomZ, 8)
    sites = []
    if np.any(n_mask) and np.any(c_mask):
        xy_N = apos[n_mask, :2].mean(axis=0)
        Cs = apos[c_mask, :2]
        xy_para = Cs[np.argmax(np.linalg.norm(Cs - xy_N[None, :], axis=1))]
        sites.append(('N', xy_N, dict(color='tab:blue', marker='o')))
        sites.append(('para-C', xy_para, dict(color='tab:orange', marker='s')))
    elif np.any(o_mask):
        sites.append(('O', apos[o_mask, :2].mean(axis=0), dict(color='tab:red', marker='o')))
        sites.append(('COM', xy_com, dict(color='k', marker='+')))
    else:
        sites.append(('COM', xy_com, dict(color='k', marker='+')))
    return sites, xy_com


def _mark_atoms(ax, apos, atomZ, sites):
    el = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
    for Z, p in zip(atomZ, apos):
        ax.plot(p[0], p[1], 'k.', ms=3, alpha=0.45)
        if int(Z) != 1:
            ax.annotate(el.get(int(Z), str(int(Z))), (p[0], p[1]), fontsize=6, color='0.35',
                        xytext=(2, 2), textcoords='offset points')
    for lab, xy, st in sites:
        ax.plot(xy[0], xy[1], st['marker'], color=st['color'], ms=9, mfc='none', mew=2)
        ax.annotate(lab, (xy[0], xy[1]), color=st['color'], fontsize=8, fontweight='bold',
                    xytext=(4, 4), textcoords='offset points')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--endgroup', default='N-h', help='pyridine in Mithun set = N-h')
    p.add_argument('--tip', default='CO_O', help='CO O-down — small tip multipole for Pauli fit')
    p.add_argument('--step', type=float, default=0.1)
    p.add_argument('--z_min', type=float, default=-15.0)
    p.add_argument('--z_max', type=float, default=15.0,
                   help='AFM grid z extent; with z_symmetric, uses ±(z_max-z_min)/2 about molecule')
    p.add_argument('--z_ref', type=float, default=12.0,
                   help='far-field zero for E/ES/vdW [Å] (default 12)')
    p.add_argument('--z_offset', type=float, default=0.0)
    p.add_argument('--z_slices', type=str,
                   default='2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,4.0,4.5',
                   help='XY slice heights [Å]; dense 2.5–3.5 by default')
    p.add_argument('--A', type=float, default=None)
    p.add_argument('--beta', type=float, default=None)
    p.add_argument('--fit_pauli', action='store_true', default=True,
                   help='fit A,β so Pauli≈Kriging E (ignore ES/vdW); then rebuild E_total/F')
    p.add_argument('--no_fit_pauli', action='store_true', help='keep default/CLI A,β; skip fit')
    p.add_argument('--fit_mode', choices=('contact', 'residual'), default='contact',
                   help='contact: _fit_pauli_powerlaw (E_ref>0 wall); residual: _fit_pauli_powerlaw_residual (signed Kriging−ES−vdW)')
    p.add_argument('--fit_residual', action='store_true',
                   help='alias for --fit_mode residual (Kriging−ES−vdW; AFM heights)')
    p.add_argument('--fit_zmin', type=float, default=None,
                   help='fit z min [Å]; default 1.5 (contact) or 2.5 (residual)')
    p.add_argument('--fit_zmax', type=float, default=None,
                   help='fit z max [Å]; default 2.0 (contact) or 5.0 (residual)')
    p.add_argument('--sigma_na', type=float, default=0.3,
                   help='Gaussian ρ_NA width [Å] for Δρ=ρ_scf−ρ_NA (default 0.3)')
    p.add_argument('--nxy', type=int, default=120)
    p.add_argument('--outdir', default=None)
    args = p.parse_args()
    if args.no_fit_pauli:
        args.fit_pauli = False
    if args.fit_residual:
        args.fit_mode = 'residual'
    if args.fit_zmin is None:
        args.fit_zmin = 2.5 if args.fit_mode == 'residual' else 1.5
    if args.fit_zmax is None:
        args.fit_zmax = 5.0 if args.fit_mode == 'residual' else 2.0

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from spammm.SPM.KrigingGridFF import (
        load_clean_points, load_zscan, demo_paths, interpolate_volume_and_forces,
        grid_origin_step, MITHUN_FUKUI,
    )
    from spammm.SPM.AFM_utils import (
        build_fdbm_grid_from_cubes, _fit_pauli_powerlaw, _fit_pauli_powerlaw_residual, _plot_pauli_fit,
    )
    from spammm.SPM import AFM as afm_mod
    _fit_fn = _fit_pauli_powerlaw_residual if args.fit_mode == 'residual' else _fit_pauli_powerlaw
    _fit_fn_name = _fit_fn.__name__

    tag = f'{args.endgroup}-{args.tip}'
    if abs(float(args.sigma_na) - 0.3) > 1e-9:
        tag = f'{tag}_sigNA{args.sigma_na:g}'
    Lz_req = float(args.z_max) - float(args.z_min)
    if abs(Lz_req - 30.0) > 0.5:  # default −5…25 → Lz=30
        tag = f'{tag}_Lz{Lz_req:g}'
    outdir = args.outdir or _outdir(f'testplot_kriging_vs_fdbm_cube/{tag}')
    z_slices = [float(x) for x in args.z_slices.split(',') if x.strip()]
    lines = [f'{tag}  alignment + N/para-C probes (Pauli-fit pair)', '']

    # ── Kriging ───────────────────────────────────────────────────────────────
    cache_F = os.path.join(outdir, f'{tag}_kriging_F.npy')
    cache_m = os.path.join(outdir, f'{tag}_kriging_meta.npz')
    if os.path.isfile(cache_F) and os.path.isfile(cache_m):
        F_k = np.load(cache_F); meta = np.load(cache_m)
        print(f'[kriging] cache {cache_F}')
    else:
        pts_p, zs_p = demo_paths(args.endgroup, args.tip)
        if not pts_p.is_file() or not zs_p.is_file():
            raise FileNotFoundError(f'Missing {pts_p} or {zs_p}')
        _, points_xy = load_clean_points(pts_p)
        zscan = load_zscan(zs_p)
        print(f'[kriging] interpolating {tag} npts={len(points_xy)} nz={zscan.shape[1]} …')
        t0 = time.time()
        xs, ys, zs, F_k, _ = interpolate_volume_and_forces(
            points_xy, zscan, nx=50, ny=50, nz=zscan.shape[1], z0=1.6, dz=args.step,
            R_basis=8.0, kind='kriging', dx=args.step, dy=args.step, to_eV=True)
        origin_k, step_k = grid_origin_step(xs, ys, zs)
        np.save(cache_F, F_k.astype(np.float32))
        np.savez(cache_m, xs=xs, ys=ys, zs=zs, origin=origin_k, step=step_k)
        meta = dict(xs=xs, ys=ys, zs=zs, origin=origin_k, step=step_k)
        print(f'[kriging] built {F_k.shape} in {time.time()-t0:.1f}s')

    zs_k = meta['zs']
    origin_k, step_k = meta['origin'], float(meta['step'])
    bk = _grid_bbox(origin_k, step_k, F_k.shape)

    # ── FDBM ──────────────────────────────────────────────────────────────────
    samp_dir = os.path.join(MITHUN_FUKUI, 'neutral', args.endgroup)
    tip_dir = os.path.join(MITHUN_FUKUI, 'neutral', args.tip)
    if not os.path.isdir(samp_dir) or not os.path.isdir(tip_dir):
        raise FileNotFoundError(f'Missing cubes {samp_dir} / {tip_dir}')
    t1 = time.time()
    fdbm = build_fdbm_grid_from_cubes(
        samp_dir, tip_dir, step=args.step, margin_xy=4.0,
        z_min=args.z_min, z_max=args.z_max, sigma_na=float(args.sigma_na),
        A_pauli=args.A, beta_pauli=args.beta, tip_name=args.tip, verbosity=0,
        clamp_cores=True, use_gpu_project=True, z_symmetric=True,
    )
    print(f'[fdbm] {fdbm["ngrid"]} in {time.time()-t1:.1f}s')
    origin_f, step_f = fdbm['origin'], fdbm['step']
    bf = _grid_bbox(origin_f, step_f, fdbm['E_total'].shape)
    apos, atomZ = fdbm['atomPos'], fdbm['atomZ']
    sites, xy_com = _probe_sites(apos, atomZ)

    xmin, xmax, ymin, ymax = _intersection_xy(bk, bf)
    if xmax <= xmin or ymax <= ymin:
        raise RuntimeError(f'No XY overlap Kriging↔FDBM')
    bbox_xy = (xmin, xmax, ymin, ymax)
    ext = [xmin, xmax, ymin, ymax]

    lines += [
        f'pair: sample={args.endgroup} (pyridine if N-h)  tip={args.tip}  reoriented={fdbm.get("tip_reoriented")}  sigma_na={args.sigma_na:g}Å',
        f'probes: {[(lab, [float(x) for x in xy]) for lab, xy, _ in sites]}',
        f'Kriging XY [{bk["xmin"]:.2f},{bk["xmax"]:.2f}]×[{bk["ymin"]:.2f},{bk["ymax"]:.2f}] z[{bk["zmin"]:.1f},{bk["zmax"]:.1f}]',
        f'FDBM    XY [{bf["xmin"]:.2f},{bf["xmax"]:.2f}]×[{bf["ymin"]:.2f},{bf["ymax"]:.2f}] z[{bf["zmin"]:.1f},{bf["zmax"]:.1f}]',
        f'plot XY intersection [{xmin:.2f},{xmax:.2f}]×[{ymin:.2f},{ymax:.2f}]',
        '',
    ]

    zf = origin_f[2] + step_f * np.arange(fdbm['E_total'].shape[2])
    E_ref = float(fdbm['E_total'][:, :, _iz(origin_f[2], step_f, args.z_ref, fdbm['E_total'].shape[2])].mean())

    profiles = {}
    for label, xy, style in sites:
        Ek0 = _sample_line(F_k[:, :, :, 3], origin_k, step_k, xy, zs_k)
        Ek0 = Ek0 - Ek0[-1]
        Fzk = _sample_line(F_k[:, :, :, 2], origin_k, step_k, xy, zs_k)
        Ef = _sample_line(fdbm['E_total'], origin_f, step_f, xy, zf) - E_ref
        Fzf = _sample_line(fdbm['F_total'][:, :, :, 2], origin_f, step_f, xy, zf)
        Ep = _sample_line(fdbm['E_pauli'], origin_f, step_f, xy, zf)
        Ee_raw = _sample_line(fdbm['E_es'], origin_f, step_f, xy, zf)
        Ee = Ee_raw - float(np.interp(args.z_ref, zf, Ee_raw))
        Ev_raw = _sample_line(fdbm['E_vdw'], origin_f, step_f, xy, zf)
        Ev = Ev_raw - float(np.interp(args.z_ref, zf, Ev_raw))
        profiles[label] = dict(xy=xy, style=style, Ek0=Ek0, Fzk=Fzk, Ef=Ef, Fzf=Fzf, Ep=Ep,
                               Ee=Ee, Ee_raw=Ee_raw, Ev=Ev, Ev_raw=Ev_raw,
                               overlap=_sample_line(fdbm['overlap_raw'], origin_f, step_f, xy, zf))
        i2 = int(np.argmin(np.abs(zf - 2.0)))
        lines.append(
            f'  {label}: @z≈2.0 (pre-fit) Pauli={Ep[i2]:+.4f} ES={Ee[i2]:+.4f} '
            f'|ES|/(|Pauli|+ε)={abs(Ee[i2])/(abs(Ep[i2])+1e-12):.3f}'
        )
    lines.append('')

    # ── Fit A,β then rebuild FDBM E/F (must happen before any plots) ───────────
    A0 = float(fdbm['A_pauli'])
    beta0 = float(fdbm['beta_pauli'])
    A_pauli, beta_pauli = A0, beta0

    def _apply_pauli(A, beta, *, rebuild_profiles=True):
        """E_pauli=A·S^β; E_total=Pauli+ES+vdW; F=−∇E. Updates fdbm + profiles + E_ref."""
        nonlocal E_ref, A_pauli, beta_pauli
        Ep_new = afm_mod.scale_pauli_field(fdbm['overlap_raw'], step_f, A, beta, return_grads=False)
        Et_new = (Ep_new + fdbm['E_es'] + fdbm['E_vdw']).astype(np.float32)
        # fail-fast: composition identity
        err = float(np.max(np.abs(Et_new - (Ep_new + fdbm['E_es'] + fdbm['E_vdw']))))
        if err > 1e-6:
            raise RuntimeError(f'E_total ≠ Pauli+ES+vdW (max|Δ|={err})')
        F_new = fdbm['afmulator'].compute_gradient_cl(Et_new, step_f, bAlloc=True)
        fdbm['E_pauli'], fdbm['E_total'], fdbm['F_total'] = Ep_new, Et_new, F_new
        fdbm['A_pauli'], fdbm['beta_pauli'] = float(A), float(beta)
        A_pauli, beta_pauli = float(A), float(beta)
        E_ref = float(Et_new[:, :, _iz(origin_f[2], step_f, args.z_ref, Et_new.shape[2])].mean())
        if rebuild_profiles:
            for label, xy, _st in sites:
                pr = profiles[label]
                pr['Ep'] = _sample_line(Ep_new, origin_f, step_f, xy, zf)
                pr['Ef'] = _sample_line(Et_new, origin_f, step_f, xy, zf) - E_ref
                pr['Fzf'] = _sample_line(F_new[:, :, :, 2], origin_f, step_f, xy, zf)
                # composition on-profile (same zero for Ef): Ef ≈ Ep + Ee + Ev − E_ref
                # Ee already zeroed at z_ref; Ev usually ~0 there
        print(f'[pauli] applied A={A_pauli:.4f} β={beta_pauli:.4f}  '
              f'E_pauli∈[{Ep_new.min():.3e},{Ep_new.max():.3e}]  E_ref={E_ref:.4f}')

    if args.fit_pauli and args.A is None and args.beta is None:
        z_lo, z_hi = float(args.fit_zmin), float(args.fit_zmax)
        # contact: Pauli ≈ Kriging E (E>0 wall). residual: Pauli ≈ Kriging−ES−vdW (signed AFM range).
        use_resid = (args.fit_mode == 'residual')
        fit_mode = 'Kriging−ES−vdW residual (signed)' if use_resid else 'Kriging E contact wall (E_ref>0)'
        lines += [
            f'Pauli FIT → {fit_mode}  z∈[{z_lo},{z_hi}]  (AFM_utils.{_fit_fn_name})',
            f'NOTE: default A={A0:.2f} β={beta0:.4f} = PAULI_FITTED_DEFAULTS[pyscf_6-31g*] '
            f'from GAUSSIAN tip σ=0.7Å (NOT real CO)',
        ]
        per_site, o_pool, e_pool, z_pool = [], [], [], []
        for label, xy, style in sites:
            pr = profiles[label]
            Ek_on_f = np.interp(zf, zs_k, pr['Ek0'], left=np.nan, right=np.nan)
            E_tgt = (Ek_on_f - pr['Ee'] - pr['Ev']) if use_resid else Ek_on_f
            try:
                A_s, b_s, r2_s, _ = _fit_fn(zf, pr['overlap'], E_tgt, z_min=z_lo, z_max=z_hi)
                per_site.append((label, float(A_s), float(b_s), float(r2_s)))
                lines.append(f'  fit {label}: A={A_s:.4f}  beta={b_s:.4f}  R²={r2_s:.4f}')
                m = ((zf >= z_lo) & (zf <= z_hi) & np.isfinite(E_tgt) & (pr['overlap'] > 1e-15))
                if not use_resid:
                    m = m & (E_tgt > 1e-15)
                o_pool.append(pr['overlap'][m]); e_pool.append(E_tgt[m]); z_pool.append(zf[m])
                _plot_pauli_fit(
                    zf, E_tgt, A_s * (pr['overlap'] ** b_s), A_s, b_s,
                    os.path.join(outdir, f'pauli_fit_{label}.png'),
                    f'{tag} {label}', z_min=z_lo, z_max=z_hi,
                    ref_label=('Kriging−ES−vdW' if use_resid else 'Kriging E'))
            except ValueError as ex:
                lines.append(f'  fit {label}: FAILED ({ex})')
        if o_pool:
            o_all = np.concatenate(o_pool); e_all = np.concatenate(e_pool); z_all = np.concatenate(z_pool)
            A_p, b_p, r2_p, _ = _fit_fn(z_all, o_all, e_all, z_min=z_lo, z_max=z_hi)
            _apply_pauli(float(A_p), float(b_p))
            lines.append(f'  fit POOLED: A={A_pauli:.4f}  beta={beta_pauli:.4f}  R²={r2_p:.4f}')
            # post-fit channel check at z≈2
            for label, xy, _st in sites:
                pr = profiles[label]
                i2 = int(np.argmin(np.abs(zf - 2.0)))
                lines.append(
                    f'  post-fit {label} @z≈2: Pauli={pr["Ep"][i2]:+.4f} ES={pr["Ee"][i2]:+.4f} '
                    f'FDBM_E={pr["Ef"][i2]:+.4f}  (expect FDBM≈Pauli+ES+vdW)'
                )
            import json
            with open(os.path.join(outdir, 'pauli_fit.json'), 'w') as jf:
                json.dump({
                    'tag': tag, 'tip': args.tip, 'endgroup': args.endgroup,
                    'A': A_pauli, 'beta': beta_pauli, 'R2_pooled': float(r2_p),
                    'fit_target': 'Kriging-ES-vdW' if use_resid else 'Kriging',
                    'fit_mode': args.fit_mode,
                    'fit_fn': _fit_fn_name,
                    'sigma_na': float(args.sigma_na),
                    'z_min': z_lo, 'z_max': z_hi,
                    'A_default_gaussian_tip': A0, 'beta_default_gaussian_tip': beta0,
                    'per_site': [{'label': L, 'A': a, 'beta': b, 'R2': r} for L, a, b, r in per_site],
                    'note': 'Pauli fitted to Kriging E irrespective of ES; E_total rebuilt after fit.',
                }, jf, indent=2)
            print(f'[fit] POOLED A={A_pauli:.4f} beta={beta_pauli:.4f} R²={r2_p:.4f}  '
                  f'target={fit_mode}  (Gaussian-tip default was A={A0:.2f} β={beta0:.4f})')
        lines.append('')
    elif args.A is not None or args.beta is not None:
        _apply_pauli(float(args.A if args.A is not None else A0),
                     float(args.beta if args.beta is not None else beta0))

    A_pauli = float(fdbm['A_pauli'])
    beta_pauli = float(fdbm['beta_pauli'])
    pauli_lbl = f'Pauli A={A_pauli:.2f} β={beta_pauli:.4f}'
    lines.insert(1, f'Pauli params: A={A_pauli:.4f}  beta={beta_pauli:.4f}  '
                 f'(Gaussian-tip default was A={A0:.2f} β={beta0:.4f})')
    lines.insert(2, f'Fit mode: {args.fit_mode} ({_fit_fn_name})')
    lines.insert(3, '')

    n_sites = len(sites)
    fig, axes = plt.subplots(n_sites, 2, figsize=(11, 3.8 * n_sites), sharex=True)
    if n_sites == 1:
        axes = np.asarray(axes).reshape(1, -1)
    for row, (label, xy, style) in enumerate(sites):
        pr = profiles[label]
        zfp = zf + args.z_offset
        axE, axF = axes[row]
        axE.plot(zs_k, pr['Ek0'], 'k-', lw=2, label='Kriging E')
        axE.plot(zfp, pr['Ef'], 'r-', lw=1.5, label='FDBM E')
        axE.plot(zfp, pr['Ep'], 'r--', alpha=0.7, label=pauli_lbl)
        axE.plot(zfp, pr['Ee'], 'b--', alpha=0.7, label='ES')
        axE.plot(zfp, pr['Ev'], 'g--', alpha=0.7, label='vdW')
        axE.axvline(0, color='gray', ls=':'); axE.set_ylabel('E [eV]')
        axE.set_title(f'E(z) {label} ({xy[0]:.2f},{xy[1]:.2f})', color=style['color'])
        axE.legend(fontsize=7); axE.set_xlim(-1, 8); axE.grid(True, alpha=0.3)
        axF.plot(zs_k, pr['Fzk'], 'k-', lw=2, label='Kriging Fz')
        axF.plot(zfp, pr['Fzf'], 'r-', lw=1.5, label='FDBM Fz')
        axF.axvline(0, color='gray', ls=':'); axF.set_ylabel('Fz [eV/Å]')
        axF.set_title(f'Fz(z) {label}', color=style['color'])
        axF.legend(fontsize=7); axF.set_xlim(-1, 8); axF.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('z [Å]'); axes[-1, 1].set_xlabel('z [Å]')
    fig.suptitle(f'{tag}  z-profiles  Δz={args.z_offset}  {pauli_lbl}', fontsize=11)
    fig.tight_layout()
    prof_path = os.path.join(outdir, 'z_profiles_overlay.png')
    fig.savefig(prof_path, dpi=140); plt.close(fig)
    print(f'Saved {prof_path}')

    fig, axes = plt.subplots(n_sites, 2, figsize=(11, 3.6 * n_sites))
    if n_sites == 1:
        axes = np.asarray(axes).reshape(1, -1)
    for row, (label, xy, style) in enumerate(sites):
        pr = profiles[label]
        zfp = zf + args.z_offset
        for col, (xlim, title) in enumerate([((1.4, 3.2), 'contact zoom'), ((2.5, 6.0), 'AFM zoom')]):
            ax = axes[row, col]
            ax.plot(zs_k, pr['Ek0'], 'k-', lw=2, label='Kriging E')
            ax.plot(zfp, pr['Ef'], 'r-', lw=1.5, label='FDBM E')
            ax.plot(zfp, pr['Ep'], 'r--', alpha=0.8, label=pauli_lbl)
            ax.plot(zfp, pr['Ee'], 'b--', alpha=0.8, label='ES')
            ax.set_xlim(*xlim)
            m = (zs_k >= xlim[0]) & (zs_k <= xlim[1])
            mf = (zfp >= xlim[0]) & (zfp <= xlim[1])
            ys = np.concatenate([pr['Ek0'][m], pr['Ef'][mf], pr['Ep'][mf], pr['Ee'][mf]])
            if ys.size:
                lo, hi = np.nanpercentile(ys, [1, 99])
                pad = 0.05 * (hi - lo + 1e-6)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f'{label} — {title}', color=style['color'])
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            if row == n_sites - 1:
                ax.set_xlabel('z [Å]')
            if col == 0:
                ax.set_ylabel('E [eV]')
    fig.suptitle(f'{tag}  contact / AFM (linear)  {pauli_lbl}', fontsize=11)
    fig.tight_layout()
    zoom_path = os.path.join(outdir, 'z_profiles_zoom.png')
    fig.savefig(zoom_path, dpi=140); plt.close(fig)
    print(f'Saved {zoom_path}')

    # ── ES far-field: z=2.5 → end of cell (raw vs zeroed at z_ref / z_end) ─────
    z_end = float(zf[-1])
    z_lo_es = 2.5
    m_es = (zf >= z_lo_es)
    Ee_mean = fdbm['E_es'].reshape(-1, fdbm['E_es'].shape[2]).mean(axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    # (0,0) raw ES at probes + XY-mean
    ax = axes[0, 0]
    ax.plot(zf[m_es], Ee_mean[m_es], 'k-', lw=2, label='XY-mean ES (raw)')
    for label, xy, style in sites:
        pr = profiles[label]
        ax.plot(zf[m_es], pr['Ee_raw'][m_es], '-', color=style['color'], lw=1.4, label=f'ES raw {label}')
    ax.axhline(0, color='gray', ls=':'); ax.axvline(args.z_ref, color='orange', ls='--', alpha=0.8, label=f'z_ref={args.z_ref}')
    ax.set_ylabel('E_ES [eV]'); ax.set_title('ES raw (no zero)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    # (0,1) ES − ES(z_end) on-site — remove far monopole/FFT floor at cell end
    ax = axes[0, 1]
    for label, xy, style in sites:
        pr = profiles[label]
        e0 = float(pr['Ee_raw'][-1])
        ax.plot(zf[m_es], pr['Ee_raw'][m_es] - e0, '-', color=style['color'], lw=1.5,
                label=f'{label} − ES(z_end={z_end:.1f})')
    ax.plot(zf[m_es], Ee_mean[m_es] - float(Ee_mean[-1]), 'k--', lw=1.2, label='XY-mean − end')
    ax.axhline(0, color='gray', ls=':')
    ax.set_ylabel('ΔE_ES [eV]'); ax.set_title('ES − ES(cell end)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    # (1,0) ES − ES(z_ref)  (what profiles use)
    ax = axes[1, 0]
    for label, xy, style in sites:
        pr = profiles[label]
        ax.plot(zf[m_es], pr['Ee'][m_es], '-', color=style['color'], lw=1.5, label=f'{label} − ES(z_ref)')
    ax.axhline(0, color='gray', ls=':'); ax.axvline(args.z_ref, color='orange', ls='--', alpha=0.8)
    ax.set_xlabel('z [Å]'); ax.set_ylabel('ΔE_ES [eV]')
    ax.set_title(f'ES − ES(z_ref={args.z_ref})'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    # (1,1) channels at N: Pauli, ES_zeroed, vdW, FDBM vs Kriging  z≥2.5
    ax = axes[1, 1]
    lab0, xy0, st0 = sites[0]
    pr = profiles[lab0]
    ax.plot(zs_k, pr['Ek0'], 'k-', lw=2, label='Kriging E')
    ax.plot(zf[m_es], pr['Ef'][m_es], 'r-', lw=1.3, label='FDBM E')
    ax.plot(zf[m_es], pr['Ep'][m_es], 'r--', alpha=0.8, label='Pauli')
    ax.plot(zf[m_es], pr['Ee'][m_es], 'b--', alpha=0.8, label='ES−ref')
    ax.plot(zf[m_es], pr['Ev'][m_es], 'g--', alpha=0.8, label='vdW−ref')
    ax.axhline(0, color='gray', ls=':'); ax.set_xlim(z_lo_es, z_end)
    ax.set_xlabel('z [Å]'); ax.set_ylabel('E [eV]')
    ax.set_title(f'{lab0} channels z≥{z_lo_es}'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    Lz = float(fdbm['ngrid'][2]) * float(step_f)
    fig.suptitle(f'{tag}  ES far-field  Lz={Lz:.1f}Å  z∈[{args.z_min},{args.z_max}]  '
                 f'grid z_end={z_end:.1f}', fontsize=11)
    fig.tight_layout()
    es_far_path = os.path.join(outdir, 'z_profiles_ES_far.png')
    fig.savefig(es_far_path, dpi=140); plt.close(fig)
    print(f'Saved {es_far_path}')
    lines += [
        f'ES far-field plot: z={z_lo_es}→{z_end:.1f}  Lz={Lz:.1f}Å  z_ref={args.z_ref}',
        f'  XY-mean ES raw @z_ref={float(np.interp(args.z_ref, zf, Ee_mean)):+.4f}  @z_end={float(Ee_mean[-1]):+.4f}',
        '',
    ]

    # Contact zoom LOG scale — slope ≈ β of E_Pauli = A · overlap^β (visually exponential in z)
    eps_log = 1e-6
    xlim_c = (1.5, 3.2)
    fig, axes = plt.subplots(1, max(n_sites, 1), figsize=(5.2 * max(n_sites, 1), 4.2), sharey=True)
    if n_sites == 1:
        axes = [axes]
    for ax, (label, xy, style) in zip(axes, sites):
        pr = profiles[label]
        zfp = zf + args.z_offset
        mk = (zs_k >= xlim_c[0]) & (zs_k <= xlim_c[1]) & (pr['Ek0'] > eps_log)
        mp = (zfp >= xlim_c[0]) & (zfp <= xlim_c[1]) & (pr['Ep'] > eps_log)
        me = (zfp >= xlim_c[0]) & (zfp <= xlim_c[1]) & (pr['Ef'] > eps_log)
        ax.semilogy(zs_k[mk], pr['Ek0'][mk], 'k-', lw=2.2, label='Kriging E')
        ax.semilogy(zfp[mp], pr['Ep'][mp], 'r--', lw=1.8, label=pauli_lbl)
        ax.semilogy(zfp[me], pr['Ef'][me], 'r-', lw=1.2, alpha=0.6, label='FDBM E')
        ax.set_xlim(*xlim_c)
        ax.set_xlabel('z [Å]'); ax.set_ylabel('E [eV] (log)')
        ax.set_title(f'{label} contact LOG', color=style['color'])
        ax.legend(fontsize=7); ax.grid(True, which='both', alpha=0.35)
        ax.text(0.98, 0.02, f'E_Pauli=A·S^β\nA={A_pauli:.2f}\nβ={beta_pauli:.4f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    fig.suptitle(f'{tag}  contact LOG — slope guides β; vertical shift guides A', fontsize=11)
    fig.tight_layout()
    log_path = os.path.join(outdir, 'z_profiles_contact_log.png')
    fig.savefig(log_path, dpi=140); plt.close(fig)
    print(f'Saved {log_path}')

    both_path = None
    if n_sites >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        # linear contact, linear AFM, log contact
        panels = [
            (axes[0], (1.4, 3.2), 'contact linear', False),
            (axes[1], (2.5, 6.0), 'AFM linear', False),
            (axes[2], (1.5, 3.2), 'contact LOG', True),
        ]
        for ax, xlim, title, use_log in panels:
            ys_all = []
            for label, xy, style in sites:
                pr = profiles[label]
                zfp = zf + args.z_offset
                if use_log:
                    mk = (zs_k >= xlim[0]) & (zs_k <= xlim[1]) & (pr['Ek0'] > eps_log)
                    mp = (zfp >= xlim[0]) & (zfp <= xlim[1]) & (pr['Ep'] > eps_log)
                    ax.semilogy(zs_k[mk], pr['Ek0'][mk], '-', color=style['color'], lw=2, label=f'Kriging {label}')
                    ax.semilogy(zfp[mp], pr['Ep'][mp], '--', color=style['color'], lw=1.3, alpha=0.85,
                                label=f'Pauli {label}')
                else:
                    ax.plot(zs_k, pr['Ek0'], '-', color=style['color'], lw=2, label=f'Kriging {label}')
                    ax.plot(zfp, pr['Ep'], '--', color=style['color'], lw=1.3, alpha=0.85,
                            label=f'Pauli {label}')
                    m = (zs_k >= xlim[0]) & (zs_k <= xlim[1])
                    mf = (zfp >= xlim[0]) & (zfp <= xlim[1])
                    ys_all.append(pr['Ek0'][m]); ys_all.append(pr['Ep'][mf])
            ax.set_xlim(*xlim); ax.set_xlabel('z [Å]')
            ax.set_ylabel('E [eV]' + (' (log)' if use_log else ''))
            if not use_log and ys_all:
                ys = np.concatenate(ys_all)
                lo, hi = np.nanpercentile(ys, [1, 99])
                pad = 0.08 * (hi - lo + 1e-6)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(title); ax.legend(fontsize=6); ax.grid(True, which='both', alpha=0.3)
        fig.suptitle(f'{tag}  N vs para-C  {pauli_lbl}', fontsize=11)
        fig.tight_layout()
        both_path = os.path.join(outdir, 'z_profiles_N_vs_paraC.png')
        fig.savefig(both_path, dpi=140); plt.close(fig)
        print(f'Saved {both_path}')

    n_rows = len(z_slices)
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, n_rows * 2.4))
    if n_rows == 1:
        axes = np.asarray(axes).reshape(1, -1)
    for i, z in enumerate(z_slices):
        z_f = z - args.z_offset
        _, _, Ek = _xy_slice_common(F_k[:, :, :, 3], origin_k, step_k, z, bbox_xy, args.nxy)
        _, _, Fzk = _xy_slice_common(F_k[:, :, :, 2], origin_k, step_k, z, bbox_xy, args.nxy)
        _, _, Ef = _xy_slice_common(fdbm['E_total'] - E_ref, origin_f, step_f, z_f, bbox_xy, args.nxy)
        _, _, Fzf = _xy_slice_common(fdbm['F_total'][:, :, :, 2], origin_f, step_f, z_f, bbox_xy, args.nxy)
        panels = [(Ek.T, f'Krig E z={z:.2f}'), (Ef.T, f'FDBM E z_f={z_f:.2f}'),
                  ((Ef - Ek).T, 'E residual'), (Fzk.T, 'Krig Fz'), (Fzf.T, 'FDBM Fz')]
        m = np.isfinite(Ek) & np.isfinite(Ef)
        corrE = float(np.corrcoef(Ek[m], Ef[m])[0, 1]) if m.sum() > 10 else float('nan')
        m2 = np.isfinite(Fzk) & np.isfinite(Fzf)
        corrFz = float(np.corrcoef(Fzk[m2], Fzf[m2])[0, 1]) if m2.sum() > 10 else float('nan')
        lines.append(f'XY z_K={z:.2f}: corrE={corrE:+.3f} corrFz={corrFz:+.3f}')
        for ax, (arr, title) in zip(axes[i], panels):
            v = max(float(np.nanpercentile(np.abs(arr), 99)), 1e-12)
            im = ax.imshow(arr, origin='lower', extent=ext, cmap='RdBu_r', vmin=-v, vmax=v, aspect='equal')
            _mark_atoms(ax, apos, atomZ, sites)
            ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'{tag}  same XY  post-fit {pauli_lbl}', fontsize=11)
    fig.tight_layout()
    xy_path = os.path.join(outdir, 'GridFF_E_Fz_sameXY.png')
    fig.savefig(xy_path, dpi=140); plt.close(fig)
    print(f'Saved {xy_path}')

    # Channel map at mid of dense contact band (z≈3.0)
    z_ch = 3.0 if any(abs(z - 3.0) < 1e-9 for z in z_slices) else z_slices[len(z_slices) // 2]
    z_f = z_ch - args.z_offset
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, key, title in zip(axes, ['E_pauli', 'E_es', 'E_vdw', 'E_total'],
                              ['Pauli', 'ES', 'vdW', 'E−Eref']):
        vol = fdbm[key] if key != 'E_total' else (fdbm['E_total'] - E_ref)
        _, _, arr = _xy_slice_common(vol, origin_f, step_f, z_f, bbox_xy, args.nxy)
        v = max(float(np.nanpercentile(np.abs(arr), 99)), 1e-12)
        im = ax.imshow(arr.T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-v, vmax=v, aspect='equal')
        _mark_atoms(ax, apos, atomZ, sites)
        ax.set_title(f'{title} z_F={z_f:.2f}'); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'FDBM channels  {tag}  {pauli_lbl}', fontsize=11)
    fig.tight_layout()
    ch_path = os.path.join(outdir, 'FDBM_channels_sameXY.png')
    fig.savefig(ch_path, dpi=140); plt.close(fig)
    print(f'Saved {ch_path}')

    lines += [
        '',
        'REVIEW:',
        '  1. z_profiles_ES_far.png — ES raw / −zend / −z_ref, z=2.5→cell end',
        '  2. z_profiles_contact_log.png — LOG contact: slope↔β, offset↔A',
        '  3. z_profiles_zoom.png — linear contact / AFM',
        '  4. GridFF_E_Fz_sameXY.png — atoms marked',
        f'  5. Pauli now: A={A_pauli:.4f}  beta={beta_pauli:.4f}',
        f'  6. {outdir}',
    ]
    summary = os.path.join(outdir, 'SUMMARY.out')
    with open(summary, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'REVIEW: {summary}')
    print(f'REVIEW: {es_far_path}')
    print(f'REVIEW: {prof_path}')
    print(f'REVIEW: {zoom_path}')
    print(f'REVIEW: {log_path}')
    if both_path:
        print(f'REVIEW: {both_path}')
    print(f'REVIEW: {xy_path}')
    print(f'REVIEW: {ch_path}')
    print(f'REVIEW: {outdir}/')


if __name__ == '__main__':
    main()
