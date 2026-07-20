#!/usr/bin/env python3
"""DFT Kriging GridFF → PP-AFM images (ppafm test_relax_kriging equivalent).

Reproduces Mithun DFT z-scan → Kriging GridFF → AFMulator.scan_fdbm for visual review.

Usage:
  python tests/SPM/testplot_kriging_relax.py
  python tests/SPM/testplot_kriging_relax.py --endgroup HHO-h-p_1 --tip H2O_O --dx 0.1

Outputs → debug/testplot_kriging_relax/ :
  GridFF_approach_slices.png, OutFz_*.png, df_*.png, GridFF_vs_OutFz.png, SUMMARY.out
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '0')


def _outdir():
    root = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'testplot_kriging_relax')
    os.makedirs(root, exist_ok=True)
    return os.path.abspath(root)


def _plot_gridff(xs, ys, zs, F_afm, out_path, n_show=9):
    """F_afm (nx,ny,nz,4) → E,Fx,Fy,Fz approach strips."""
    import matplotlib.pyplot as plt
    n_show = min(n_show, len(zs))
    fig, axes = plt.subplots(n_show, 4, figsize=(14, n_show * 2.0))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    names = ['E', 'Fx', 'Fy', 'Fz']
    for i in range(n_show):
        iz = int(round(i / max(n_show - 1, 1) * (len(zs) - 1)))
        for j, name in enumerate(names):
            ax = axes[i, j]
            # F_afm[ix,iy,iz] → imshow expects (ny,nx) with origin lower → arr.T if we index [ix,iy]
            arr = F_afm[:, :, iz, j].T
            vlim = max(float(np.percentile(np.abs(arr), 99)), 1e-9)
            im = ax.imshow(arr, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vlim, vmax=vlim, aspect='equal')
            ax.set_title(f'{name} z={zs[iz]:.2f}Å', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('Kriging GridFF (DFT) — E / Fx / Fy / Fz', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'Saved {out_path}')


def _plot_outfz_panel(Fz, heights, xs, ys, out_path, title, cmap='gray', ncols=9):
    import matplotlib.pyplot as plt
    nz = Fz.shape[2]
    nrows = int(np.ceil(nz / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 1.9))
    axes = np.atleast_2d(axes)
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    for k in range(nz):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        arr = Fz[:, :, k].T
        vlim = max(float(np.percentile(np.abs(arr), 99)), 1e-9)
        ax.imshow(arr, origin='lower', extent=extent, cmap=cmap, vmin=-vlim, vmax=vlim, aspect='equal')
        ax.set_title(f'z={heights[k]:.1f}Å', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(nz, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis('off')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'Saved {out_path}')


def _plot_comparison(xs, ys, zs, F_afm, Fz_relax, heights, out_path, n_rows=6):
    """Side-by-side GridFF E, Fz_grid, OutFz at matching heights."""
    import matplotlib.pyplot as plt
    n_rows = min(n_rows, len(heights), len(zs))
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, n_rows * 2.0))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    for i in range(n_rows):
        ih = int(round(i / max(n_rows - 1, 1) * (len(heights) - 1)))
        h = float(heights[ih])
        iz = int(np.clip(np.argmin(np.abs(zs - h)), 0, len(zs) - 1))
        panels = [
            (F_afm[:, :, iz, 3].T, f'E grid z={zs[iz]:.2f}', 'RdBu_r'),
            (F_afm[:, :, iz, 2].T, f'Fz grid z={zs[iz]:.2f}', 'RdBu_r'),
            (Fz_relax[:, :, ih].T, f'OutFz probe z={h:.2f}', 'RdBu_r'),
        ]
        for j, (arr, title, cmap) in enumerate(panels):
            ax = axes[i, j]
            vlim = max(float(np.percentile(np.abs(arr), 99)), 1e-9)
            im = ax.imshow(arr, origin='lower', extent=extent, cmap=cmap, vmin=-vlim, vmax=vlim, aspect='equal')
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('GridFF vs relaxed OutFz (DFT Kriging)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    p = argparse.ArgumentParser(description='Kriging DFT GridFF → PP-AFM visual demo')
    p.add_argument('--endgroup', default='HHO-h-p_1')
    p.add_argument('--tip', default='H2O_O')
    p.add_argument('--dx', type=float, default=0.1, help='Isotropic grid step [Å]')
    p.add_argument('--R', type=float, default=8.0, help='Kriging support radius')
    p.add_argument('--klat', type=str, default='0.5,1.0,2.0', help='Lateral stiffness N/m (comma list)')
    p.add_argument('--bond_length', type=float, default=4.0)
    p.add_argument('--h_min', type=float, default=3.5, help='Probe height min [Å] (world z if mol_z=0)')
    p.add_argument('--h_max', type=float, default=5.5)
    p.add_argument('--h_step', type=float, default=0.2)
    p.add_argument('--cache', type=int, default=1, help='Cache GridFF npy in outdir')
    p.add_argument('--outdir', default=None)
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')

    from spammm.SPM.KrigingGridFF import (
        load_clean_points, load_zscan, demo_paths, interpolate_volume_and_forces, grid_origin_step,
    )
    from spammm.SPM.AFM import AFMulator, stiffness_Nm_to_eVA2, compute_df_amp
    from spammm.SPM import AFM_utils as afm_utils

    outdir = args.outdir or _outdir()
    os.makedirs(outdir, exist_ok=True)
    tag = f'{args.endgroup}-{args.tip}'
    cache_F = os.path.join(outdir, f'{tag}_F_afm.npy')
    cache_meta = os.path.join(outdir, f'{tag}_meta.npz')

    points_path, zscan_path = demo_paths(args.endgroup, args.tip)
    print(f'[kriging] points={points_path}')
    print(f'[kriging] zscan={zscan_path}')
    _, points_xy = load_clean_points(points_path)
    zscan_vals = load_zscan(zscan_path)
    nz = zscan_vals.shape[1]
    z0, dz = 1.6, args.dx
    print(f'[kriging] N={points_xy.shape[0]} n_z={nz} dx={args.dx} R={args.R}')

    t0 = time.time()
    if args.cache and os.path.isfile(cache_F) and os.path.isfile(cache_meta):
        print(f'[kriging] Loading cached GridFF: {cache_F}')
        F_afm = np.load(cache_F)
        meta = np.load(cache_meta)
        xs, ys, zs = meta['xs'], meta['ys'], meta['zs']
        origin, step = meta['origin'], float(meta['step'])
    else:
        print('[kriging] Interpolating (may take a few minutes)...')
        xs, ys, zs, F_afm, _vol = interpolate_volume_and_forces(
            points_xy, zscan_vals, nx=50, ny=50, nz=nz, z0=z0, dz=dz,
            R_basis=args.R, kind='kriging', dx=args.dx, dy=args.dx, to_eV=True, verbose=False,
        )
        origin, step = grid_origin_step(xs, ys, zs)
        if args.cache:
            np.save(cache_F, F_afm.astype(np.float32))
            np.savez(cache_meta, xs=xs, ys=ys, zs=zs, origin=origin, step=step)
            print(f'[kriging] Cached → {cache_F}')
    print(f'[kriging] F_afm={F_afm.shape} origin={origin} step={step:.4f}  ({time.time()-t0:.1f}s)')
    print(f'[kriging] E[eV] ∈ [{F_afm[...,3].min():.4f}, {F_afm[...,3].max():.4f}]')

    gpath = os.path.join(outdir, 'GridFF_approach_slices.png')
    _plot_gridff(xs, ys, zs, F_afm, gpath)

    # Probe heights inside GridFF z-range (mol_z=0 → world z = probe height)
    heights = np.arange(args.h_min, args.h_max + 0.5 * args.h_step, args.h_step, dtype=np.float32)
    heights = heights[(heights >= float(zs[0]) + step) & (heights <= float(zs[-1]) - step)]
    if len(heights) < 2:
        raise RuntimeError(f'No probe heights in grid z=[{zs[0]},{zs[-1]}]; adjust --h_min/--h_max')
    scan_xs = xs.astype(np.float32)
    scan_ys = ys.astype(np.float32)
    print(f'[relax] scan {len(scan_xs)}x{len(scan_ys)}x{len(heights)}  h={heights[0]:.2f}…{heights[-1]:.2f}')

    afm = AFMulator(use_morse=False, nloc=32)
    F32 = np.ascontiguousarray(F_afm, dtype=np.float32)
    afm.setup_fdbm_grid(F32, origin, step)

    klat_Nm_list = [float(x) for x in args.klat.split(',') if x.strip()]
    lines = [
        f'Kriging DFT → PP-AFM  {tag}',
        f'points={points_path}',
        f'zscan={zscan_path}',
        f'F_afm={F_afm.shape} step={step:.4f} R={args.R} kind=kriging',
        f'E[eV] min={F_afm[...,3].min():.6g} max={F_afm[...,3].max():.6g}',
        f'scan heights={list(np.round(heights, 2))}',
        f'bond_length={args.bond_length}',
        '',
    ]

    baseline_Fz = None
    for klat_Nm in klat_Nm_list:
        K_LAT = stiffness_Nm_to_eVA2(klat_Nm)
        print(f'[relax] klat={klat_Nm} N/m → {K_LAT:.5f} eV/Å²')
        FEs, tip_disp = afm.scan_fdbm(
            scan_xs, scan_ys, heights, mol_z=0.0,
            K_LAT=K_LAT, K_RAD=20.0, bond_length=args.bond_length,
            ppm_mode=True, use_fire=True,
        )
        Fz = FEs[:, :, :, 2]
        df = compute_df_amp(Fz, float(heights[1] - heights[0]), amp=1.0)
        safe = f'klat{klat_Nm:g}'.replace('.', 'p')
        _plot_outfz_panel(Fz, heights, xs, ys, os.path.join(outdir, f'OutFz_{safe}.png'),
                          title=f'OutFz  klat={klat_Nm:g} N/m  {tag}', cmap='gray')
        afm_utils.plot_grid_Fz(
            df, heights, f'df klat={klat_Nm:g} N/m  {tag}', f'df_{safe}.png',
            x_ext=[float(xs[0]), float(xs[-1])], y_ext=[float(ys[0]), float(ys[-1])],
            save_dir=outdir, cmap='gray',
        )
        lines.append(f'klat={klat_Nm:g} N/m  Fz∈[{Fz.min():.4g},{Fz.max():.4g}]  df∈[{df.min():.4g},{df.max():.4g}]')
        if baseline_Fz is None:
            baseline_Fz = Fz
            _plot_comparison(xs, ys, zs, F_afm, Fz, heights,
                             os.path.join(outdir, 'GridFF_vs_OutFz.png'))

    # rigid-ish: very stiff lateral
    K_stiff = stiffness_Nm_to_eVA2(50.0)
    print(f'[relax] nearly-rigid klat=50 N/m')
    FEs_r, _ = afm.scan_fdbm(
        scan_xs, scan_ys, heights, mol_z=0.0,
        K_LAT=K_stiff, K_RAD=20.0, bond_length=args.bond_length,
        ppm_mode=True, use_fire=True,
    )
    _plot_outfz_panel(FEs_r[:, :, :, 2], heights, xs, ys,
                      os.path.join(outdir, 'OutFz_rigidish.png'),
                      title=f'OutFz nearly-rigid (50 N/m)  {tag}', cmap='gray')

    summary = os.path.join(outdir, 'SUMMARY.out')
    lines += [
        '',
        'REVIEW checklist:',
        '  1. GridFF_approach_slices.png — E/Fz should show molecular features; vacuum smooth (no dotted artifacts)',
        '  2. OutFz_klat0p5.png — soft tip, more contrast / tip bending vs stiff',
        '  3. GridFF_vs_OutFz.png — OutFz related to Fz_grid but smoothed/shifted by PP relax',
        '  4. Compare visually to ppafm: data/mithun_afm_scans/relax_test_hho/HHO-h-p_1_H2O_O/',
        f'  5. Folder: {outdir}',
    ]
    text = '\n'.join(lines) + '\n'
    with open(summary, 'w') as f:
        f.write(text)
    print(text)
    print(f'REVIEW: {summary}')
    print(f'REVIEW: {outdir}/')


if __name__ == '__main__':
    main()
