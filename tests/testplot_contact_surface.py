#!/usr/bin/env python3
"""
testplot_contact_surface.py — GPU contact-surface comparison vs all-atom reference.

Methods (all heavy work on PyOpenCL):
  1) cs_brute_plqh_points — Morse reference (tip charge=0)
  2) Separable B-spline×poly — global CG fit (dense xy grid, h0 height map)
  3) Radial PIC — GPU CG fit + 16×16 tiled eval

Run:  python tests/testplot_contact_surface.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from spammm.surfaces.ContactSurface import (
    ContactSurfaceCL, SeparableParams, PICParams,
    load_atom_data, make_fit_grid, eval_slice_map_gpu, pic_grid_to_map, bspline_n_intervals,
    select_contact_atoms,
)

PTCDA = os.path.join(_proj, 'data', 'xyz', 'PTCDA.xyz')
PLOT_DIR = os.path.join(_proj, 'debug', 'testplot_contact_surface')
MARGIN = 2.0
ALPHA_MORSE = 1.8
R_DAMP = 0.1
POLY_R = 10.0
M_START = 4
NZ = 5
PIC_CELL = 10.0
BSPL_DX = 0.4
FIT_DX, FIT_DY, FIT_DZ = BSPL_DX, BSPL_DX, 0.2
PLOT_DX = 0.1
Z_OFFSET = 3.0
FIT_Z_HALF = 0.2
FZ_ACTIVE_THRESH = 0.01


def _rmse(a, b):
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def _active_rmse(a, b, ref, thresh=FZ_ACTIVE_THRESH):
    mask = np.abs(ref) > thresh
    if not np.any(mask):
        return float('nan')
    return _rmse(a[mask], b[mask])


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f'Loading single PTCDA from {PTCDA} …')
    apos, reqs, enames, lvec, qs = load_atom_data(PTCDA)
    zmax = float(np.max(apos[:, 2]))
    zmin = float(np.min(apos[:, 2]))
    z_scan = zmax + Z_OFFSET
    print(f'  atoms={len(apos)}  z=[{zmin:.2f},{zmax:.2f}]  z_scan={z_scan:.2f}')

    x0f = float(apos[:, 0].min()) - MARGIN
    x1f = float(apos[:, 0].max()) + MARGIN
    y0f = float(apos[:, 1].min()) - MARGIN
    y1f = float(apos[:, 1].max()) + MARGIN
    print(f'  bbox xy=[{x0f:.1f},{x1f:.1f}]×[{y0f:.1f},{y1f:.1f}] Å  margin={MARGIN}')

    z0_fit = z_scan - FIT_Z_HALF
    z1_fit = z_scan + FIT_Z_HALF
    fit_pts = make_fit_grid(x0f, x1f, y0f, y1f, z0_fit, z1_fit, FIT_DX, FIT_DY, FIT_DZ)
    bspl_nx = bspline_n_intervals(x1f - x0f, BSPL_DX)
    bspl_ny = bspline_n_intervals(y1f - y0f, BSPL_DX)
    print(f'  fit samples: {len(fit_pts)}  bspline grid {bspl_nx}×{bspl_ny}  step={BSPL_DX} Å  poly powers={[M_START * 2**k for k in range(NZ)]}')

    ocl = ContactSurfaceCL(nloc=64)
    ocl.setup_atoms(apos.astype(np.float32), reqs.astype(np.float32), alpha_morse=ALPHA_MORSE, r_damp=R_DAMP, plqh=(1.0, 1.0, 0.0, 0.0))
    print('  OpenCL kernels loaded (common+Forces+contact_surface)')

    t0 = time.perf_counter()
    E_ref, _ = ocl.eval_brute(fit_pts)
    print(f'  GPU brute fit reference: {time.perf_counter()-t0:.2f}s')

    sep = SeparableParams(x0f, y0f, BSPL_DX, BSPL_DX, bspl_nx, bspl_ny, poly_R=POLY_R, m_start=M_START, nz=NZ, apos=apos)
    t0 = time.perf_counter()
    rmse_sep_cg = ocl.fit_separable_cg(sep, fit_pts, E_ref, apos=apos, n_iter=80, bPrint=True)
    print(f'  separable CG fit RMSE={rmse_sep_cg:.4e}  ({time.perf_counter()-t0:.2f}s)  coeffs={sep.n_coeff}')

    contact_idx = select_contact_atoms(apos, z_local=1.2, xy_radius=14.0)
    print(f'  contact atoms for PIC: {len(contact_idx)} / {len(apos)}')

    pic = PICParams(apos, contact_idx, poly_R=POLY_R, m_start=M_START, nz=4, cell_size=PIC_CELL, bounds=(x0f, y0f, x1f, y1f))
    t0 = time.perf_counter()
    rmse_pic_fit = ocl.fit_pic_cg(pic, fit_pts, E_ref, n_iter=80, reg=1e-4, bPrint=True)
    print(f'  PIC CG fit RMSE={rmse_pic_fit:.4e}  atoms={pic.nat}')

    def eval_brute(pts):
        E, F = ocl.eval_brute(pts)
        return E.astype(np.float64), F.astype(np.float64)

    def eval_sep(pts):
        E, F = ocl.eval_separable(pts, sep)
        return E.astype(np.float64), F.astype(np.float64)

    def eval_pic(pts):
        E, F = ocl.eval_pic(pts, pic)
        return E.astype(np.float64), F.astype(np.float64)

    print(f'  computing Fz maps (step={PLOT_DX} Å, tight bbox) …')
    t0 = time.perf_counter()
    xs, ys, Fz_ref = eval_slice_map_gpu(ocl, eval_brute, x0f, x1f, y0f, y1f, z_scan, PLOT_DX, PLOT_DX)
    print(f'    brute slice: {time.perf_counter()-t0:.2f}s')
    t0 = time.perf_counter()
    _, _, Fz_sep = eval_slice_map_gpu(ocl, eval_sep, x0f, x1f, y0f, y1f, z_scan, PLOT_DX, PLOT_DX)
    print(f'    separable-CG slice: {time.perf_counter()-t0:.2f}s')
    t0 = time.perf_counter()
    nx, ny = len(xs), len(ys)
    _, F_pic = ocl.eval_pic_grid(x0f, y0f, z_scan, PLOT_DX, PLOT_DX, nx, ny, pic)
    Fz_pic = pic_grid_to_map(F_pic, nx, ny)
    print(f'    PIC tiled slice: {time.perf_counter()-t0:.2f}s')

    rmse_sep = _rmse(Fz_sep, Fz_ref)
    rmse_pic = _rmse(Fz_pic, Fz_ref)
    rmse_sep_act = _active_rmse(Fz_sep, Fz_ref, Fz_ref)
    rmse_pic_act = _active_rmse(Fz_pic, Fz_ref, Fz_ref)
    print(f'  slice RMSE Fz: separable={rmse_sep:.4e}  PIC={rmse_pic:.4e}')
    print(f'  slice RMSE Fz (active |Fz|>{FZ_ACTIVE_THRESH}): separable={rmse_sep_act:.4e}  PIC={rmse_pic_act:.4e}')

    extent = [x0f, x1f, y0f, y1f]
    vmax = float(np.percentile(np.abs(Fz_ref), 99))
    contact_xy = apos[contact_idx, :2]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    titles = ['(1) GPU all-atom Fz', '(2) B-spline×poly Fz', '(3) PIC tiled Fz']
    data = [Fz_ref, Fz_sep, Fz_pic]
    for ax, Fz, title in zip(axes[0], data, titles):
        im = ax.imshow(Fz.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        ax.scatter(contact_xy[:, 0], contact_xy[:, 1], c='k', s=4, marker='.', alpha=0.35, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
        plt.colorbar(im, ax=ax, fraction=0.046, label='Fz [eV/Å]')

    err_sep = Fz_sep - Fz_ref
    err_pic = Fz_pic - Fz_ref
    ev = max(_rmse(err_sep, np.zeros_like(err_sep)), _rmse(err_pic, np.zeros_like(err_pic)), 1e-12)
    for ax, err, title in zip(axes[1, :2], [err_sep, err_pic], ['(2) error vs ref', '(3) error vs ref']):
        im = ax.imshow(err.T, origin='lower', extent=extent, cmap='coolwarm', vmin=-3*ev, vmax=3*ev, aspect='equal')
        ax.scatter(contact_xy[:, 0], contact_xy[:, 1], c='k', s=4, marker='.', alpha=0.35, linewidths=0)
        ax.set_title(f'{title}  RMSE={_rmse(err, np.zeros_like(err)):.3g}')
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
        plt.colorbar(im, ax=ax, fraction=0.046, label='ΔFz')

    ax = axes[1, 2]
    ax.scatter(contact_xy[:, 0], contact_xy[:, 1], c=apos[contact_idx, 2], s=8, marker='.', cmap='viridis', alpha=0.8)
    ax.set_xlim(x0f, x1f); ax.set_ylim(y0f, y1f)
    ax.set_aspect('equal')
    ax.set_title(f'Contact atoms (n={len(contact_idx)})')
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, fraction=0.046).set_label('z [Å]')

    fig.suptitle(f'GPU contact surface — PTCDA  bbox+{MARGIN}Å  z_scan={z_scan:.1f}Å  bspl={BSPL_DX}Å', fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, 'contact_surface_comparison.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'REVIEW: {out_png}')

    out_txt = os.path.join(PLOT_DIR, 'contact_surface_summary.out')
    with open(out_txt, 'w') as f:
        f.write(f'molecule: PTCDA  atoms={len(apos)}\n')
        f.write(f'bbox: x=[{x0f:.2f},{x1f:.2f}] y=[{y0f:.2f},{y1f:.2f}] margin={MARGIN}\n')
        f.write(f'contact_atoms: {len(contact_idx)}\n')
        f.write(f'fit_samples: {len(fit_pts)}  z_range=[{z0_fit:.2f},{z1_fit:.2f}] dz={FIT_DZ}\n')
        f.write(f'bspline_grid: {bspl_nx}×{bspl_ny}  step={BSPL_DX} Å  poly_R={POLY_R}  powers={[M_START*2**k for k in range(NZ)]}\n')
        f.write(f'separable_CG_rmse={rmse_sep_cg:.6e}\n')
        f.write(f'pic_fit_rmse={rmse_pic_fit:.6e}\n')
        f.write(f'slice_rmse_Fz: separable={rmse_sep:.6e}  pic={rmse_pic:.6e}\n')
        f.write(f'slice_rmse_Fz_active: separable={rmse_sep_act:.6e}  pic={rmse_pic_act:.6e}\n')
    print(f'REVIEW: {out_txt}')
    print('Done.')


if __name__ == '__main__':
    main()
