#!/usr/bin/env python3
"""Headless reproduction of GUI ModularAFMPipeline Stage 4 — PP elasticity debug.

Reuses Stage-3 F_total from a GUI cache (or recomputes via ModularAFMPipeline).
Focus: tip deflection |dxy| at each height for soft vs stiff K_LAT.

Units: internal K_LAT is eV/Å². Classic Hapala = 0.5 N/m ≈ 0.031 eV/Å².
Old GUI bug: unlabeled spin default 0.5 as eV/Å² ≈ 8 N/m → rigid tip.
GUI now enters N/m; see afm.stiffness_Nm_to_eVA2().

Usage:
  PYOPENCL_CTX=0 python tests/SPM/testplot_pp_relax_gui.py \\
      --cache /tmp/afm_gui_XXXX --xyz data/xyz/benzene.xyz

Outputs under debug/afm_pp_relax_gui/:
  tip_dxy_K*.png, compare_Klat.png, Fz_relax_vs_rigid.png, SUMMARY.out
"""
import os, sys, argparse, glob, numpy as np

os.environ.setdefault('PYOPENCL_CTX', '0')


def _find_latest_gui_cache():
    cands = sorted(glob.glob('/tmp/afm_gui_*/cache_stage3_potentials.npz'), key=os.path.getmtime, reverse=True)
    return os.path.dirname(cands[0]) if cands else None


def _scan_grid(atomPos, scan_range=3.0, scan_step=0.1, hmin=2.8, hmax=3.6, hstep=0.1):
    """Match ModularAFMPipeline / GUI scan construction."""
    x0 = float(atomPos[:, 0].min() - scan_range)
    x1 = float(atomPos[:, 0].max() + scan_range)
    y0 = float(atomPos[:, 1].min() - scan_range)
    y1 = float(atomPos[:, 1].max() + scan_range)
    scan_xs = np.arange(x0, x1 + 0.5 * scan_step, scan_step, dtype=np.float32)
    scan_ys = np.arange(y0, y1 + 0.5 * scan_step, scan_step, dtype=np.float32)
    heights = np.arange(hmin, hmax + 0.5 * hstep, hstep, dtype=np.float32)
    return scan_xs, scan_ys, heights


def _sample_F_unrelaxed(F_total, origin, step, scan_xs, scan_ys, heights, mol_z):
    """Trilinear sample of F_total at rigid (unrelaxed) tip positions — no PP."""
    from scipy.ndimage import map_coordinates
    nx_s, ny_s, nz_s = len(scan_xs), len(scan_ys), len(heights)
    XX, YY, ZZ = np.meshgrid(scan_xs, scan_ys, heights + mol_z, indexing='ij')
    coords = np.vstack([
        ((XX - origin[0]) / step).ravel(),
        ((YY - origin[1]) / step).ravel(),
        ((ZZ - origin[2]) / step).ravel(),
    ])
    out = np.zeros((nx_s, ny_s, nz_s, 4), np.float32)
    for a in range(4):
        out[..., a] = map_coordinates(F_total[..., a], coords, order=1, mode='nearest').reshape(nx_s, ny_s, nz_s)
    return out


def _run_relax(afmulator, F_total, origin, step, scan_xs, scan_ys, heights, mol_z,
               K_LAT, mode, bond_length=4.0, K_RAD=20.0):
    from spammm.SPM import AFM as afm
    afmulator.setup_fdbm_grid(F_total, origin, step)
    relax_pars = [0.1, 0.1, 0.03, 0.1]
    if mode == 'ppm':
        FEs, tip = afmulator.scan_fdbm(
            scan_xs, scan_ys, heights, mol_z=mol_z,
            K_LAT=K_LAT, K_RAD=K_RAD, bond_length=bond_length,
            relax_pars=relax_pars, ppm_mode=True, use_fire=True)
    elif mode == '2d':
        FEs, tip = afmulator.scan_fdbm_2d(
            scan_xs, scan_ys, heights, mol_z=mol_z, K_LAT=K_LAT)
    else:
        raise ValueError(mode)
    df = afm.compute_df(FEs[..., 2], float(heights[1] - heights[0]) if len(heights) > 1 else 0.1)
    return FEs, tip, df


def _plot_height_stack(fields, heights, title, path, cmap='magma', sym=False, extent=None):
    import matplotlib.pyplot as plt
    nz = len(heights)
    ncols = min(nz, 5)
    nrows = int(np.ceil(nz / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.6 * nrows), squeeze=False)
    for iz, h in enumerate(heights):
        r, c = divmod(iz, ncols)
        ax = axes[r][c]
        sl = fields[:, :, iz]
        kw = dict(origin='lower', cmap=cmap)
        if extent is not None:
            kw['extent'] = extent
        if sym:
            v = float(np.max(np.abs(sl))) or 1e-30
            im = ax.imshow(sl.T, vmin=-v, vmax=v, **kw)
        else:
            im = ax.imshow(sl.T, **kw)
        ax.set_title(f'h={h:.2f}  max={np.max(np.abs(sl)):.3g}', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    for k in range(nz, nrows * ncols):
        r, c = divmod(k, ncols); axes[r][c].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f'REVIEW: {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=None, help='GUI /tmp/afm_gui_* dir with stage3 cache')
    ap.add_argument('--xyz', default='data/xyz/benzene.xyz')
    ap.add_argument('--outdir', default='debug/afm_pp_relax_gui')
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--K', default='0.5,0.1,0.03', help='comma K_LAT values [eV/Å²]')
    ap.add_argument('--modes', default='ppm,2d', help='ppm and/or 2d')
    ap.add_argument('--bond_length', type=float, default=4.0)
    ap.add_argument('--hmin', type=float, default=2.8)
    ap.add_argument('--hmax', type=float, default=3.6)
    ap.add_argument('--hstep', type=float, default=0.1)
    ap.add_argument('--scan_range', type=float, default=3.0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import spammm.atomicUtils as au
    from spammm.SPM import AFM as afm

    os.makedirs(args.outdir, exist_ok=True)
    lines = []
    lines.append('PP elasticity diagnostic (GUI ModularAFMPipeline Stage-4 path)')
    lines.append(f'xyz={args.xyz}  bond_length={args.bond_length}  (GUI compose default=4.0)')
    lines.append('Classic Hapala PP: K_LAT ≈ 0.03 eV/Å² (~0.5 N/m). GUI default K_LAT=0.5 is ~17× stiffer.')
    lines.append('')

    pos, _, names, _, _ = au.load_xyz(args.xyz)
    atomPos = np.asarray(pos, float)
    mol_z = float(atomPos[:, 2].max())
    scan_xs, scan_ys, heights = _scan_grid(
        atomPos, scan_range=args.scan_range, hmin=args.hmin, hmax=args.hmax, hstep=args.hstep)
    extent = [float(scan_xs[0]), float(scan_xs[-1]), float(scan_ys[0]), float(scan_ys[-1])]
    lines.append(f'mol_z={mol_z:.3f}  scan={len(scan_xs)}x{len(scan_ys)}  heights={list(np.round(heights, 2))}')

    cache = args.cache or _find_latest_gui_cache()
    if cache is None or not os.path.exists(os.path.join(cache, 'cache_stage3_potentials.npz')):
        raise SystemExit('No GUI stage3 cache found. Pass --cache /tmp/afm_gui_XXX after a GUI run.')
    lines.append(f'cache={cache}')
    c3 = np.load(os.path.join(cache, 'cache_stage3_potentials.npz'))
    F_total = c3['F_total']
    c2 = np.load(os.path.join(cache, 'cache_stage2_grids.npz'))
    origin = np.asarray(c2['origin'], float)
    ngrid = np.asarray(c2['ngrid'], int)
    step = args.step
    # Infer step from span if possible
    # ModularPipeline stores step on pipe; GUI uses afm_step_spin. Prefer matching L/n.
    # origin + n*step ≈ extent; use given --step (GUI default 0.1)
    lines.append(f'F_total shape={F_total.shape} origin={origin} ngrid={ngrid} step={step}')
    lines.append(f'|F| near nuclei huge (ok); scan-plane forces matter for PP:')

    F_rigid = _sample_F_unrelaxed(F_total, origin, step, scan_xs, scan_ys, heights, mol_z)
    for iz, h in enumerate(heights):
        Fx, Fy, Fz = F_rigid[:, :, iz, 0], F_rigid[:, :, iz, 1], F_rigid[:, :, iz, 2]
        Fl = np.hypot(Fx, Fy)
        lines.append(
            f'  rigid@h={h:.2f}: |Fxy|_max={Fl.max():.4f}  |Fz|_max={np.abs(Fz).max():.4f}  '
            f'Fxy_rms={np.sqrt((Fl**2).mean()):.4f} eV/Å')

    afmulator = afm.AFMulator(use_morse=False, nloc=32)  # same as ModularPipeline stage4
    Ks = [float(x) for x in args.K.split(',') if x.strip()]
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    results = {}

    for mode in modes:
        for K in Ks:
            tag = f'{mode}_K{K:g}'
            print(f'\n=== relax {tag} bond_length={args.bond_length} ===')
            FEs, tip, df = _run_relax(
                afmulator, F_total, origin, step, scan_xs, scan_ys, heights, mol_z,
                K_LAT=K, mode=mode, bond_length=args.bond_length)
            dxy = np.hypot(tip['dx'], tip['dy'])
            results[tag] = dict(FEs=FEs, tip=tip, df=df, dxy=dxy, K=K, mode=mode)
            lines.append(f'\n[{tag}]')
            for iz, h in enumerate(heights):
                lines.append(
                    f'  h={h:.2f}: |dxy|_max={dxy[:,:,iz].max():.4f}Å  |dxy|_rms={np.sqrt((dxy[:,:,iz]**2).mean()):.4f}Å  '
                    f'df=[{df[:,:,iz].min():.3g},{df[:,:,iz].max():.3g}]  '
                    f'Fz_r=[{FEs[:,:,iz,2].min():.3g},{FEs[:,:,iz,2].max():.3g}]')
            _plot_height_stack(
                dxy, heights,
                f'Tip |dxy|  {tag}  bondL={args.bond_length}Å  (PP elasticity)',
                os.path.join(args.outdir, f'tip_dxy_{tag}.png'),
                cmap='magma', extent=extent)
            _plot_height_stack(
                df, heights,
                f'df after PP  {tag}',
                os.path.join(args.outdir, f'df_{tag}.png'),
                cmap='afmhot', extent=extent)
            _plot_height_stack(
                -FEs[..., 2], heights,
                f'Fz relaxed (repulsive +)  {tag}',
                os.path.join(args.outdir, f'Fz_relax_{tag}.png'),
                cmap='seismic', sym=True, extent=extent)

    # Side-by-side compare |dxy| at lowest height for all K
    iz0 = 0
    h0 = float(heights[iz0])
    tags = list(results.keys())
    fig, axes = plt.subplots(1, len(tags), figsize=(3.2 * len(tags), 3.4), squeeze=False)
    for i, tag in enumerate(tags):
        ax = axes[0, i]
        sl = results[tag]['dxy'][:, :, iz0]
        im = ax.imshow(sl.T, origin='lower', cmap='magma', extent=extent)
        ax.set_title(f'{tag}\nmax={sl.max():.3f}Å', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'PP tip |dxy| @ h={h0:.2f}Å — soft K should show large deflection / sharp PP contrast', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    cpath = os.path.join(args.outdir, 'compare_Klat_dxy.png')
    fig.savefig(cpath, dpi=140); plt.close(fig)
    lines.append(f'\nREVIEW: {cpath}')
    print(f'REVIEW: {cpath}')

    # Rigid Fz vs relaxed Fz at softest K (if present)
    soft = None
    for tag in tags:
        if results[tag]['K'] == min(Ks) and results[tag]['mode'] == modes[0]:
            soft = tag
    if soft is not None:
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for col, iz in enumerate([0, len(heights)//2, -1]):
            h = float(heights[iz])
            Fr = -F_rigid[:, :, iz, 2]
            Fre = -results[soft]['FEs'][:, :, iz, 2]
            dxy = results[soft]['dxy'][:, :, iz]
            for row, (sl, name, cmap, sym) in enumerate([
                (Fr, 'Fz rigid (no PP)', 'seismic', True),
                (Fre, f'Fz relaxed {soft}', 'seismic', True),
            ]):
                ax = axes[row, col]
                v = float(np.max(np.abs(sl))) or 1e-30
                im = ax.imshow(sl.T, origin='lower', cmap=cmap, vmin=-v, vmax=v, extent=extent)
                ax.set_title(f'{name}\nh={h:.2f}', fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.046)
            # overlay tip displacement quiver on bottom? separate third row used for dxy in col
        # replace row1 last with note — add dxy row via second figure already
        fig.suptitle('Rigid tip Fz vs PP-relaxed Fz (same F_total) — sharp edges need |dxy|≳0.3Å', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p2 = os.path.join(args.outdir, 'Fz_rigid_vs_relax.png')
        fig.savefig(p2, dpi=140); plt.close(fig)
        lines.append(f'REVIEW: {p2}')
        print(f'REVIEW: {p2}')

    lines.append('\n=== INTERPRETATION ===')
    lines.append('If |dxy|_max ≪ 0.2 Å at K_LAT=0.5 → tip behaves rigid (blunt contrast).')
    lines.append('If |dxy|_max ≳ 0.5–1 Å at K_LAT=0.03 → PP elasticity works; GUI K is too stiff.')
    lines.append('If |dxy| stays tiny even at K=0.03 → force sampling / height / bond_length bug (probe in vacuum).')
    lines.append('DEFAULT Morse path uses stiffness≈0.03; GUI FDBM path uses 0.5.')

    out = os.path.join(args.outdir, 'SUMMARY.out')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nREVIEW: {out}')


if __name__ == '__main__':
    main()
