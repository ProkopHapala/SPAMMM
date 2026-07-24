#!/usr/bin/env python3
"""
testplot_contact_surface.py — GPU contact surface: wide z fit + raw E/Fz parity.

Phase 1a: fit separable coeffs on adaptive z-stack z_max+[1.2..6.0]Å (dz 0.1→1.0Å)
          with Boltzmann weights emphasizing low-energy (vdW-well) samples.
Phase 1b: unrelaxed E/Fz parity vs brute at close heights and 2–6 Å z-stack.
Phase 2 (optional): PP-relaxed scan — 3D img_FF vs quasi-2D (RUN_CONTACT_PP=1).

Run:  python tests/testplot_contact_surface.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

os.environ.setdefault('PYOPENCL_CTX', '0')

from spammm.surfaces.ContactSurface import (
    load_atom_data, eval_slice_map_gpu, select_contact_atoms, make_fit_z_planes_adaptive, boltzmann_fit_weights,
    poly_z_doubling_modes, poly_z_mode_powers, PICParams,
)

PTCDA = os.path.join(_proj, 'data', 'xyz', 'PTCDA.xyz')
PARAMS = os.path.join(_proj, 'data', 'ElementTypes.dat')
PLOT_DIR = os.path.join(_proj, 'debug', 'testplot_contact_surface')
MARGIN = 4.0
SCAN_MARGIN = 4.0
# Spherical Morse-R0 contact h₀; fit offsets are above h0_max (not bare atom zmax)
H0_MODE = 'spheres'
H0_R_SCALE = 0.75  # <1 so z0 clamp sits in hard repulsion (1.0 = Morse well → no wall)
FIT_Z_LO, FIT_Z_HI = 0.05, 4.0
FIT_DZ_LO, FIT_DZ_HI = 0.1, 0.8
FIT_BOLTZMANN_T = None
FIT_FORCE_WEIGHT = 1.0
FIT_Z_PLANES = make_fit_z_planes_adaptive(FIT_Z_LO, FIT_Z_HI, FIT_DZ_LO, FIT_DZ_HI)
ALPHA_MORSE = 1.8
R_DAMP = 0.1
POLY_Z0 = 0.0
POLY_R = 4.0
M_START = 4
NZ = 6
PIC_CELL = 10.0
PIC_POLY_R = POLY_R
PIC_NZ = 5
PIC_Z_LOCAL = 1.2
PIC_XY_RADIUS = 14.0
PIC_REG = 1e-2
BSPL_DX = 0.2
FIT_DX, FIT_DY = BSPL_DX, BSPL_DX
# Absolute h = z−zmax; tip_R=0 → R0≈1.7–2.0 Å, so these sit at/above contact
CLOSE_Z_OFFS = (2.0, 2.5, 3.0)
PROFILE_Z_OFFS = np.arange(1.5, 7.01, 0.05)
ZSTACK_Z_OFFS = tuple(np.arange(2.0, 6.01, 0.5))
FAR_Z_OFF = 3.5
PLOT_DX = 0.05
ZSTACK_PLOT_DX = 0.15
FZ_ACTIVE_THRESH = 0.01
DX_SCAN = 0.2
DX_GRID = 0.2
Z_TOP = 12.0
NZ_SCAN = 25
DTIP = -0.15


def _rmse(a, b):
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def _active_rmse(a, b, ref, thresh=FZ_ACTIVE_THRESH):
    mask = np.abs(ref) > thresh
    if not np.any(mask):
        return float('nan')
    return _rmse(a[mask], b[mask])


def _grid_n(apos, margin, z_top, dx):
    mn, mx = apos.min(axis=0), apos.max(axis=0)
    nx = int(np.ceil(((mx[0] - mn[0]) + 2 * margin) / dx))
    ny = int(np.ceil(((mx[1] - mn[1]) + 2 * margin) / dx))
    nz = int(np.ceil(((mx[2] - mn[2]) + margin / 2 + z_top) / dx))
    return (max(8, nx), max(8, ny), max(8, nz))


def _eval_slice_maps(eval_fn, x0, x1, y0, y1, z_scan, dx, dy):
    xs = np.arange(x0, x1 + 1e-9, dx)
    ys = np.arange(y0, y1 + 1e-9, dy)
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_scan)], axis=1)
    E, F = eval_fn(pts)
    return xs, ys, E.reshape(nx, ny), F[:, 2].reshape(nx, ny)


def plot_fit_weighting(sep, zmax):
    """Plot Boltzmann fit weights w(E) and per-z-plane summary."""
    w = np.asarray(sep.fit_sample_weights, dtype=np.float64)
    E = np.asarray(sep.fit_E_ref, dtype=np.float64)
    z_off = np.asarray(sep.fit_pts_z, dtype=np.float64) - zmax
    T = sep.fit_boltzmann_T
    E_min = float(E.min())
    z_planes = np.asarray(sep.fit_z_offsets, dtype=np.float64)
    dz_planes = np.diff(z_planes, prepend=z_planes[0])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    n_show = min(len(E), 12000)
    idx = np.linspace(0, len(E) - 1, n_show, dtype=int)
    sc = ax.scatter(E[idx], w[idx], c=z_off[idx], s=3, cmap='viridis', alpha=0.35, linewidths=0)
    E_grid = np.linspace(E.min(), np.percentile(E, 99.5), 300)
    ax.plot(E_grid, np.exp(-(E_grid - E_min) / T), 'r-', lw=1.5, label=f'w=exp(-(E-E_min)/T)')
    ax.set_xlabel('E_ref [eV]')
    ax.set_ylabel('weight w')
    ax.set_title('Boltzmann weights vs reference energy')
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(sc, ax=ax, label='z - zmax [Å]', fraction=0.046)

    ax = axes[0, 1]
    w_plane = []
    for zp in z_planes:
        m = np.abs(z_off - zp) < 0.5 * np.interp(zp, z_planes, np.maximum(np.diff(z_planes, prepend=z_planes[0]), 1e-9))
        w_plane.append(float(np.median(w[m])) if np.any(m) else float('nan'))
    ax.plot(z_planes, w_plane, 'o-', ms=5, label='median w per z plane')
    ax.axvspan(FIT_Z_LO, FIT_Z_HI, color='0.9', alpha=0.5, label='fit interval')
    ax.set_xlabel('z - zmax [Å]')
    ax.set_ylabel('median weight')
    ax.set_title('Weight vs z plane')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(z_planes, dz_planes, 's-', ms=5, color='C1')
    ax.set_xlabel('z - zmax [Å]')
    ax.set_ylabel('local dz [Å]')
    ax.set_title(f'Adaptive z sampling ({len(z_planes)} planes)')
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.02, 0.95, f'w_i = exp(-(E_i - E_min) / T),  normalized to max(w)=1\nE_min = {E_min:.4f} eV\nT = {T:.4f} eV\nfit z ∈ [{FIT_Z_LO:.1f}, {FIT_Z_HI:.1f}] Å  dz: {FIT_DZ_LO:.1f}→{FIT_DZ_HI:.1f} Å\nn_samples = {len(E):,}  n_planes = {len(z_planes)}', va='top', fontsize=10, family='monospace')
    fig.suptitle('Contact-surface fit weighting function', fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, 'contact_surface_fit_weights.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'REVIEW: {out_png}')

    out_txt = os.path.join(PLOT_DIR, 'contact_surface_fit_weights.out')
    with open(out_txt, 'w') as f:
        f.write(f'Boltzmann fit weights: T={T:.6f} eV  E_min={E_min:.6f} eV\n')
        f.write(f'z_planes offsets Å: {list(np.round(z_planes, 4))}\n')
        f.write('z_off  dz  median_w  n_pts\n')
        for iz, zp in enumerate(z_planes):
            dz = float(z_planes[iz] - z_planes[iz - 1]) if iz > 0 else 0.0
            m = np.abs(z_off - zp) < 0.5 * (dz if iz > 0 else float(z_planes[1] - z_planes[0]))
            f.write(f'{zp:6.3f} {dz:5.3f} {w_plane[iz]:8.4e} {int(m.sum()):6d}\n')
    print(f'REVIEW: {out_txt}')


def phase1_fit_close(afm, apos):
    """Fit on adaptive z-stack with Boltzmann weights; return sep + bbox."""
    zmax = float(np.max(apos[:, 2]))
    x0f = float(apos[:, 0].min()) - MARGIN
    x1f = float(apos[:, 0].max()) + MARGIN
    y0f = float(apos[:, 1].min()) - MARGIN
    y1f = float(apos[:, 1].max()) + MARGIN
    print(f'  Phase1 fit: h0_mode={H0_MODE}  AFM brute  z∈[{FIT_Z_LO},{FIT_Z_HI}]Å above contact  dz {FIT_DZ_LO}→{FIT_DZ_HI}Å  {len(FIT_Z_PLANES)} planes  dx={BSPL_DX}Å  Boltzmann T={FIT_BOLTZMANN_T or "auto"}  force_weight={FIT_FORCE_WEIGHT} (E,Fx,Fy,Fz)')
    sep = afm.fit_contact_surface(margin=MARGIN, bspl_dx=BSPL_DX, poly_R=POLY_R, poly_z0=POLY_Z0, m_start=M_START, nz=NZ, fit_z_adaptive=(FIT_Z_LO, FIT_Z_HI, FIT_DZ_LO, FIT_DZ_HI), fit_dx=FIT_DX, fit_dy=FIT_DY, fit_boltzmann=True, fit_boltzmann_T=FIT_BOLTZMANN_T, fit_force_weight=FIT_FORCE_WEIGHT, n_iter=120, brute_ref='afm', bPrint=True, h0_mode=H0_MODE, h0_R_scale=H0_R_SCALE)
    h0 = sep.h0_map
    z_ref = float(getattr(sep, 'z_ref', zmax))
    print(f'  h0: min={h0.min():.3f} max={h0.max():.3f} std={h0.std():.4f}  z_ref={z_ref:.3f}  (spheres → corrugated envelope; dz=z−h₀)')
    plot_fit_weighting(sep, zmax)

    out_txt = os.path.join(PLOT_DIR, 'contact_surface_fit.out')
    with open(out_txt, 'w') as f:
        f.write(f'molecule: PTCDA  atoms={len(apos)}  zmax={zmax:.3f}  h0_mode={H0_MODE}  h0_R_scale={H0_R_SCALE}\n')
        f.write(f'fit reference: AFM evalMorseC_QZs_toImg-equivalent brute (cMs + tip charges)\n')
        f.write(f'fit z adaptive offsets above contact z_ref=[{FIT_Z_LO},{FIT_Z_HI}] dz={FIT_DZ_LO}..{FIT_DZ_HI}  planes={list(np.round(FIT_Z_PLANES, 4))}\n')
        f.write(f'z_ref={z_ref:.3f}  abs z=[{z_ref + FIT_Z_LO:.2f},{z_ref + FIT_Z_HI:.2f}]  poly_z0={POLY_Z0:.3f} poly_R={POLY_R:.3f} Boltzmann T={sep.fit_boltzmann_T:.4f} eV  force_weight={FIT_FORCE_WEIGHT} (E,Fx,Fy,Fz)\n')
        f.write(f'bspl={sep.ncx}x{sep.ncy}  nz={NZ}  n_coeff={sep.n_coeff}\n')
        f.write(f'h0 min={h0.min():.4f} max={h0.max():.4f} std={h0.std():.6f}\n')
    print(f'REVIEW: {out_txt}')
    return sep, zmax, (x0f, x1f, y0f, y1f)


def plot_z_basis():
    """Plot separable z basis φ_k(dz) used in fit (mirror of contact_surface.cl)."""
    dz = np.linspace(0.0, 12.0, 481)
    phi, dphi, t, x = poly_z_doubling_modes(dz, poly_R=POLY_R, poly_z0=POLY_Z0, m_start=M_START, nz=NZ)
    powers = poly_z_mode_powers(M_START, NZ)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
    for k, p in enumerate(powers):
        axes[0].plot(dz, phi[:, k], lw=1.2, label=f'φ_{k}(dz)=t^{p}')
    axes[0].axvline(POLY_Z0, color='0.4', ls=':', lw=0.9, alpha=0.8, label=f'z0={POLY_Z0:.1f}Å (φ=1 below)')
    axes[0].axvline(POLY_Z0 + POLY_R, color='k', ls='--', lw=0.8, alpha=0.6, label=f'z0+Rc={POLY_Z0 + POLY_R:.1f}Å (φ→0)')
    axes[0].axvspan(FIT_Z_LO, FIT_Z_HI, color='0.9', alpha=0.5, label='fit z interval')
    axes[0].set_ylabel('φ_k(dz)')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
    axes[0].grid(alpha=0.25)
    axes[0].set_title(f'Z basis: t=1−clip((dz−z0)/Rc,0,1), z0={POLY_Z0:.1f}Å, Rc={POLY_R:.1f}Å, nz={NZ}')
    for k, p in enumerate(powers):
        axes[1].plot(dz, dphi[:, k], lw=1.0, label=f'dφ_{k}/dz')
    axes[1].axvline(POLY_Z0, color='0.4', ls=':', lw=0.9, alpha=0.8)
    axes[1].axvline(POLY_Z0 + POLY_R, color='k', ls='--', lw=0.8, alpha=0.6)
    axes[1].axvspan(FIT_Z_LO, FIT_Z_HI, color='0.9', alpha=0.5)
    axes[1].set_ylabel('dφ_k/dz [Å⁻¹]')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7)
    axes[1].grid(alpha=0.25)
    axes[2].plot(dz, t, 'k-', lw=1.2, label='t = 1 − (dz−z0)/Rc')
    axes[2].plot(dz, x, 'C4--', lw=1.0, label='x = (dz−z0)/Rc')
    axes[2].axvspan(FIT_Z_LO, FIT_Z_HI, color='0.9', alpha=0.5)
    axes[2].set_xlabel('dz = z − h0(x,y)  [Å]  (flat PTCDA: dz ≈ z − zmax)')
    axes[2].set_ylabel('t, x')
    axes[2].legend(loc='best', fontsize=8)
    axes[2].grid(alpha=0.25)
    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, 'contact_surface_z_basis.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {out_png}')

    out_txt = os.path.join(PLOT_DIR, 'contact_surface_z_basis.out')
    with open(out_txt, 'w') as f:
        f.write(f'poly_z0 = {POLY_Z0} Å  poly_R/span (Rc) = {POLY_R} Å; field clipped to zero for dz > z0+Rc = {POLY_Z0 + POLY_R} Å\n')
        f.write(f'm_start = {M_START}  nz = {NZ}  mode powers t^p: {powers}\n')
        f.write(f'fit z offsets: [{FIT_Z_LO}, {FIT_Z_HI}] Å\n\n')
        f.write('Border / instability notes:\n')
        f.write('  1. New coordinate: t=1 at lower fit edge, t=0 at upper fit edge; all z-modes are used across the sampled interval.\n')
        f.write('  2. At dz=z0, all φ_k=1 and dφ_k/dz starts finite; below z0 the basis is clamped flat, so forces are intentionally zero there.\n')
        f.write('  3. Hard cutoff dz>z0+Rc: φ≡0 → E and dE/dz jump to 0; do not extrapolate beyond upper fit edge.\n')
        f.write('  4. xy B-spline: cubic open ends; stencil clamped at bbox edges → ripples near margin.\n')
        f.write('  5. fmax(z-h0,0): kink at h0 if h0 varies in xy.\n\n')
        f.write('dz  t  phi_0..phi_{:d}\n'.format(NZ - 1))
        for zoff in list(FIT_Z_PLANES) + [POLY_Z0 + POLY_R, POLY_Z0 + POLY_R + 0.5]:
            ph, _, tv, _ = poly_z_doubling_modes([zoff], poly_R=POLY_R, poly_z0=POLY_Z0, m_start=M_START, nz=NZ)
            f.write(f'{zoff:5.2f} {tv[0]:6.4f} ' + ' '.join(f'{v:.4e}' for v in ph[0]) + '\n')
    print(f'REVIEW: {out_txt}')


def phase1_close_parity(afm, apos, sep, zmax, bbox):
    """Unrelaxed E/Fz parity at close z — Pauli repulsion, atom/bond structure."""
    x0f, x1f, y0f, y1f = bbox
    extent = [x0f, x1f, y0f, y1f]
    contact_idx = select_contact_atoms(apos, z_local=1.2, xy_radius=14.0)
    contact_xy = apos[contact_idx, :2]

    def eval_brute(pts):
        E, F = afm._brute_afm_morse_c_queries(pts)
        return E.astype(np.float64), F.astype(np.float64)

    def eval_sep(pts):
        E, F = afm._cs_fit_helper().eval_separable(pts, sep)
        return E.astype(np.float64), F.astype(np.float64)

    n_z = len(CLOSE_Z_OFFS)
    fig, axes = plt.subplots(4, n_z, figsize=(4.2 * n_z, 14))
    summary_lines = []
    for col, zoff in enumerate(CLOSE_Z_OFFS):
        z_scan = zmax + zoff
        _, _, E_ref, Fz_ref = _eval_slice_maps(eval_brute, x0f, x1f, y0f, y1f, z_scan, PLOT_DX, PLOT_DX)
        _, _, E_sep, Fz_sep = _eval_slice_maps(eval_sep, x0f, x1f, y0f, y1f, z_scan, PLOT_DX, PLOT_DX)
        rmse_E = _rmse(E_sep, E_ref)
        rmse_Fz = _rmse(Fz_sep, Fz_ref)
        rmse_Fz_act = _active_rmse(Fz_sep, Fz_ref, Fz_ref)
        summary_lines.append(f'z_off={zoff:.1f}Å  z_abs={z_scan:.2f}  rmse_E={rmse_E:.4e}  rmse_Fz={rmse_Fz:.4e}  active_Fz={rmse_Fz_act:.4e}')
        print(f'  close parity {summary_lines[-1]}')

        ev = max(float(np.percentile(E_ref, 99)), 1e-9)
        fv = max(float(np.percentile(np.abs(Fz_ref), 99)), 1e-9)
        for row, (data, title, vmin, vmax, cmap, label) in enumerate([
            (E_ref, 'brute E', 0, ev, 'magma_r', 'E [eV]'),
            (E_sep, 'fit E', 0, ev, 'magma_r', 'E [eV]'),
            (Fz_ref, 'brute Fz', -fv, fv, 'RdBu_r', 'Fz [eV/Å]'),
            (Fz_sep, 'fit Fz', -fv, fv, 'RdBu_r', 'Fz [eV/Å]'),
        ]):
            ax = axes[row, col]
            im = ax.imshow(data.T, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            ax.scatter(contact_xy[:, 0], contact_xy[:, 1], c='w', s=2, marker='.', alpha=0.4, linewidths=0)
            if row == 0:
                ax.set_title(f'z_max+{zoff:.1f}Å', fontsize=10)
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.set_xlabel('x [Å]')
            plt.colorbar(im, ax=ax, fraction=0.046, label=label)

    fig.suptitle(f'Phase1 close-contact unrelaxed parity — PTCDA  fit z∈[{FIT_Z_LO:.1f},{FIT_Z_HI:.1f}]Å  bspl={BSPL_DX}Å', fontsize=12)
    plt.tight_layout()
    out_close = os.path.join(PLOT_DIR, 'contact_surface_close_parity.png')
    fig.savefig(out_close, dpi=150)
    fig.savefig(os.path.join(PLOT_DIR, 'contact_surface_comparison.png'), dpi=150)
    plt.close(fig)
    print(f'REVIEW: {out_close}')
    print(f'REVIEW: {os.path.join(PLOT_DIR, "contact_surface_comparison.png")}')

    fig2, axes2 = plt.subplots(2, n_z, figsize=(4.2 * n_z, 7))
    for col, zoff in enumerate(CLOSE_Z_OFFS):
        z_scan = zmax + zoff
        _, _, E_ref, Fz_ref = _eval_slice_maps(eval_brute, x0f, x1f, y0f, y1f, z_scan, PLOT_DX, PLOT_DX)
        _, _, E_sep, Fz_sep = _eval_slice_maps(eval_sep, x0f, x1f, y0f, y1f, z_scan, PLOT_DX, PLOT_DX)
        err_E, err_Fz = E_sep - E_ref, Fz_sep - Fz_ref
        for row, (err, title, ref) in enumerate([(err_E, 'ΔE fit−brute', E_ref), (err_Fz, 'ΔFz fit−brute', Fz_ref)]):
            ax = axes2[row, col]
            ev = max(_rmse(err, np.zeros_like(err)), 1e-12)
            if row == 1:
                ev = max(float(np.percentile(np.abs(Fz_ref), 99)) * 0.1, ev)
            im = ax.imshow(err.T, origin='lower', extent=extent, cmap='coolwarm', vmin=-3*ev, vmax=3*ev, aspect='equal')
            ax.scatter(contact_xy[:, 0], contact_xy[:, 1], c='k', s=2, marker='.', alpha=0.35, linewidths=0)
            if row == 0:
                ax.set_title(f'z_max+{zoff:.1f}Å')
            ax.set_xlabel('x [Å]')
            rm = _rmse(err, np.zeros_like(err))
            ax.set_ylabel(f'{title}\nRMSE={rm:.3g}')
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out_err = os.path.join(PLOT_DIR, 'contact_surface_close_errors.png')
    fig2.savefig(out_err, dpi=150)
    plt.close(fig2)
    print(f'REVIEW: {out_err}')

    z_far = zmax + FAR_Z_OFF
    xs, ys, Fz_ref = eval_slice_map_gpu(None, eval_brute, x0f, x1f, y0f, y1f, z_far, 0.1, 0.1)
    _, _, Fz_sep = eval_slice_map_gpu(None, eval_sep, x0f, x1f, y0f, y1f, z_far, 0.1, 0.1)
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4.5))
    vmax = float(np.percentile(np.abs(Fz_ref), 99))
    for ax, Fz, title in zip(axes3, [Fz_ref, Fz_sep, Fz_sep - Fz_ref], ['brute Fz far', 'fit Fz far', 'ΔFz far']):
        v = vmax if title != 'ΔFz far' else max(_rmse(Fz, np.zeros_like(Fz)), 1e-12)
        cmap, vmin = ('RdBu_r', -vmax) if title != 'ΔFz far' else ('coolwarm', -3*v)
        im = ax.imshow(Fz.T, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=v if title != 'ΔFz far' else 3*v, aspect='equal')
        ax.set_title(f'{title}  z_max+{FAR_Z_OFF}Å')
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out_far = os.path.join(PLOT_DIR, 'contact_surface_far_field.png')
    fig3.savefig(out_far, dpi=150)
    plt.close(fig3)
    print(f'REVIEW: {out_far}')

    out_sum = os.path.join(PLOT_DIR, 'contact_surface_summary.out')
    with open(out_sum, 'w') as f:
        f.write('Phase1 close-contact unrelaxed parity (E + Fz vs AFM brute Morse/Coulomb reference)\n')
        f.write(f'fit_z adaptive offsets Å above zmax: [{FIT_Z_LO},{FIT_Z_HI}]  planes={list(np.round(FIT_Z_PLANES, 4))}\n')
        f.write(f'bspl_dx={BSPL_DX}  nz={NZ}  n_coeff={sep.n_coeff}\n')
        for line in summary_lines:
            f.write(line + '\n')
        f.write(f'far_field z_off={FAR_Z_OFF} rmse_Fz={_rmse(Fz_sep, Fz_ref):.6e}\n')
    print(f'REVIEW: {out_sum}')
    return summary_lines


def phase1_z_profile(afm, apos, sep, zmax):
    """1D z profile above a central atom: exposes effective z shifts directly."""
    i0 = int(np.argmin(np.linalg.norm(apos[:, :2] - apos[:, :2].mean(axis=0), axis=1)))
    x0, y0 = float(apos[i0, 0]), float(apos[i0, 1])
    z_abs = zmax + np.asarray(PROFILE_Z_OFFS, dtype=np.float64)
    pts = np.column_stack([np.full_like(z_abs, x0), np.full_like(z_abs, y0), z_abs]).astype(np.float32)
    E_ref, F_ref = afm._brute_afm_morse_c_queries(pts)
    E_fit, F_fit = afm._cs_fit_helper().eval_separable(pts, sep)
    Fz_ref = F_ref[:, 2].astype(np.float64)
    Fz_fit = F_fit[:, 2].astype(np.float64)
    E_ref = E_ref.astype(np.float64)
    E_fit = E_fit.astype(np.float64)
    z = np.asarray(PROFILE_Z_OFFS, dtype=np.float64)
    T = sep.fit_boltzmann_T
    w_prof, _, E_shift = boltzmann_fit_weights(E_ref, T=T)
    E_well = float(E_ref.min())
    e_vmin = 1.5 * E_well
    e_vmax = 2.0 * (-e_vmin)

    shifts = np.linspace(-1.0, 1.0, 401)
    mask = (z >= 1.0) & (z <= 6.0)
    best = (float('inf'), 0.0)
    for dz in shifts:
        shifted = np.interp(z[mask], z + dz, Fz_fit, left=np.nan, right=np.nan)
        ok = np.isfinite(shifted)
        if np.any(ok):
            err = _rmse(shifted[ok], Fz_ref[mask][ok])
            if err < best[0]:
                best = (err, float(dz))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
    axw = axes[0].twinx()
    axw.fill_between(z, 0.0, w_prof, color='C2', alpha=0.25, label='Boltzmann weight')
    axw.plot(z, w_prof, color='C2', lw=1.2, label=f'w=exp(-(E-E_min)/T), T={T:.3f} eV')
    axw.set_ylabel('fit weight w', color='C2')
    axw.tick_params(axis='y', labelcolor='C2')
    axw.set_ylim(0.0, 1.05)
    axes[0].plot(z, E_ref, ':', lw=1.8, color='C0', label='AFM brute E')
    axes[0].plot(z, E_fit, '-', lw=1.0, color='C1', label='B-spline quasi-2D E')
    axes[0].axvspan(FIT_Z_LO, FIT_Z_HI, color='0.9', alpha=0.5, label='fit interval')
    axes[0].set_ylabel('E [eV]')
    axes[0].set_ylim(e_vmin, e_vmax)
    h0, l0 = axes[0].legend(loc='upper left', fontsize=8), axw.legend(loc='upper right', fontsize=8)
    axes[0].add_artist(h0)
    axes[0].grid(alpha=0.25)
    axes[1].plot(z, Fz_ref, ':', lw=1.8, label='AFM brute Fz')
    axes[1].plot(z, Fz_fit, '-', lw=1.0, label='B-spline quasi-2D Fz')
    axes[1].axvspan(FIT_Z_LO, FIT_Z_HI, color='0.9', alpha=0.5, label='fit interval')
    axes[1].axhline(0.0, c='k', lw=0.7, alpha=0.5)
    axes[1].set_ylabel('Fz [eV/Å]')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[2].plot(z, w_prof, 'C2-', lw=1.5)
    axes[2].axvspan(FIT_Z_LO, FIT_Z_HI, color='0.9', alpha=0.5)
    axes[2].set_ylabel('weight w')
    axes[2].set_xlabel(f'z - zmax [Å] above atom {i0} at ({x0:.2f},{y0:.2f})')
    axes[2].grid(alpha=0.25)
    fig.suptitle(f'Central-atom z profile — best Fz z-shift dz={best[1]:+.3f}Å RMSE={best[0]:.3g}  E_well={E_well:.3f} eV  Eylim=[{e_vmin:.2f},{e_vmax:.2f}]', fontsize=11)
    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, 'contact_surface_z_profile.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'REVIEW: {out_png}')

    out_txt = os.path.join(PLOT_DIR, 'contact_surface_z_profile.out')
    with open(out_txt, 'w') as f:
        f.write(f'central_atom_index={i0} xy=({x0:.6f},{y0:.6f}) zmax={zmax:.6f}\n')
        f.write(f'fit_z=[{FIT_Z_LO},{FIT_Z_HI}]  planes={list(np.round(FIT_Z_PLANES, 4))}\n')
        f.write(f'boltzmann_T={T:.6f} eV  E_shift={E_shift:.6f} eV  E_well={E_well:.6f} eV\n')
        f.write(f'E_plot_ylim=[{e_vmin:.4f}, {e_vmax:.4f}]  (1.5*E_well .. 2*(-1.5*E_well))\n')
        f.write(f'best_Fz_vertical_shift_fit_to_ref={best[1]:+.6f}Å rmse={best[0]:.6e}\n')
        f.write('z_off  w  E_ref  E_fit  dE  Fz_ref  Fz_fit  dFz\n')
        for iz in range(0, len(z), 10):
            f.write(f'{z[iz]:7.3f} {w_prof[iz]:7.4f} {E_ref[iz]: .8e} {E_fit[iz]: .8e} {E_fit[iz]-E_ref[iz]: .8e} {Fz_ref[iz]: .8e} {Fz_fit[iz]: .8e} {Fz_fit[iz]-Fz_ref[iz]: .8e}\n')
    print(f'REVIEW: {out_txt}')
    return best


def phase1_zstack_slices(afm, apos, sep, zmax, bbox):
    """E and Fz parity slices: rows are brute, fit, fit-brute error; columns are z offsets."""
    x0f, x1f, y0f, y1f = bbox
    extent = [x0f, x1f, y0f, y1f]
    contact_idx = select_contact_atoms(apos, z_local=1.2, xy_radius=14.0)
    contact_xy = apos[contact_idx, :2]

    def eval_brute(pts):
        E, F = afm._brute_afm_morse_c_queries(pts)
        return E.astype(np.float64), F.astype(np.float64)

    def eval_sep(pts):
        E, F = afm._cs_fit_helper().eval_separable(pts, sep)
        return E.astype(np.float64), F.astype(np.float64)

    metrics = []
    E_refs, E_fits, Fz_refs, Fz_fits = [], [], [], []
    for zoff in ZSTACK_Z_OFFS:
        z_scan = zmax + float(zoff)
        _, _, E_ref, Fz_ref = _eval_slice_maps(eval_brute, x0f, x1f, y0f, y1f, z_scan, ZSTACK_PLOT_DX, ZSTACK_PLOT_DX)
        _, _, E_fit, Fz_fit = _eval_slice_maps(eval_sep, x0f, x1f, y0f, y1f, z_scan, ZSTACK_PLOT_DX, ZSTACK_PLOT_DX)
        E_refs.append(E_ref); E_fits.append(E_fit); Fz_refs.append(Fz_ref); Fz_fits.append(Fz_fit)
        metrics.append((float(zoff), _rmse(E_fit, E_ref), _rmse(Fz_fit, Fz_ref), _active_rmse(Fz_fit, Fz_ref, Fz_ref), float(np.percentile(np.abs(Fz_ref), 99))))

    def plot_stack(refs, fits, quantity, unit, cmap, symmetric):
        n_z = len(ZSTACK_Z_OFFS)
        fig, axes = plt.subplots(3, n_z, figsize=(2.7 * n_z, 8.2))
        for col, zoff in enumerate(ZSTACK_Z_OFFS):
            ref = refs[col]
            fit = fits[col]
            err = fit - ref
            if symmetric:
                vmax = max(float(np.percentile(np.abs(ref), 99)), float(np.percentile(np.abs(fit), 99)), 1e-12)
                vmin = -vmax
            else:
                vmax = max(float(np.percentile(ref, 99)), float(np.percentile(fit, 99)), 1e-12)
                vmin = min(float(np.percentile(ref, 1)), float(np.percentile(fit, 1)), 0.0)
            ev = max(float(np.percentile(np.abs(err), 99)), _rmse(fit, ref), 1e-12)
            for row, (data, title, cmap_i, vmin_i, vmax_i) in enumerate([
                (ref, f'AFM brute {quantity}', cmap, vmin, vmax),
                (fit, f'B-spline quasi-2D {quantity}', cmap, vmin, vmax),
                (err, 'fit - brute', 'coolwarm', -ev, ev),
            ]):
                ax = axes[row, col]
                im = ax.imshow(data.T, origin='lower', extent=extent, cmap=cmap_i, vmin=vmin_i, vmax=vmax_i, aspect='equal')
                ax.scatter(contact_xy[:, 0], contact_xy[:, 1], c='k' if row == 2 else 'w', s=1.4, marker='.', alpha=0.35, linewidths=0)
                if row == 0:
                    ax.set_title(f'{zoff:.1f}Å', fontsize=8)
                if col == 0:
                    ax.set_ylabel(title, fontsize=8)
                ax.tick_params(labelsize=5)
                plt.colorbar(im, ax=ax, fraction=0.046, label=unit)
        fig.suptitle(f'Raw {quantity} parity vs height — fit interval {FIT_Z_LO:.1f}..{FIT_Z_HI:.1f}Å, bspl={BSPL_DX}Å', fontsize=11)
        plt.tight_layout()
        out_png = os.path.join(PLOT_DIR, f'contact_surface_zstack_{quantity}_parity.png')
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f'REVIEW: {out_png}')

    plot_stack(E_refs, E_fits, 'E', 'E [eV]', 'magma_r', False)
    plot_stack(Fz_refs, Fz_fits, 'Fz', 'Fz [eV/Å]', 'RdBu_r', True)

    out_txt = os.path.join(PLOT_DIR, 'contact_surface_zstack_Fz_parity.out')
    with open(out_txt, 'w') as f:
        f.write('Raw Fz parity slices: AFM brute vs B-spline quasi-2D\n')
        f.write(f'fit_z=[{FIT_Z_LO},{FIT_Z_HI}]  plotted_z_offsets={ZSTACK_Z_OFFS}  plot_dx={ZSTACK_PLOT_DX}\n')
        f.write('z_off  rmse_E  rmse_Fz  active_rmse_Fz  p99_abs_Fz_ref\n')
        for zoff, rm_E, rm_Fz, active, p99 in metrics:
            f.write(f'{zoff:5.2f} {rm_E:.8e} {rm_Fz:.8e} {active:.8e} {p99:.8e}\n')
    print(f'REVIEW: {out_txt}')
    return metrics


def phase2_pp_afm_parity(sep, apos):
    """Phase 2: PP-relaxed scan — 3D img_FF vs quasi-2D replacement (same sep, same grid)."""
    from spammm.SPM.AFM import AFMulator, compute_df

    print('  Phase2: PP-AFM relaxed scan parity (reuse Phase1 sep, no refit)')
    afm = AFMulator(use_morse=True, use_fire=False)
    afm.load_molecule(PTCDA)
    afm.assign_params(params_path=PARAMS, tip_R=0.0, tip_E=1.0)
    afm.tipQs[:] = 0.0
    n_grid = _grid_n(apos, MARGIN, Z_TOP, DX_GRID)
    afm.setup_grid(n=n_grid, margin=MARGIN, z_top=Z_TOP, shift_atoms=False)
    afm.fit_contact_surface(sep=sep)

    nxy, scan_p0, scan_da, scan_db, extent, mol_z = afm.scan_bbox(margin=SCAN_MARGIN, dx=DX_SCAN)
    scan_kw = dict(nxy=nxy, nz=NZ_SCAN, dtip=DTIP, scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db)
    z0_tip = float(scan_p0[2])
    h_tip = z0_tip + np.arange(NZ_SCAN) * DTIP - mol_z
    h_probe = h_tip + float(afm.dpos0[2])
    print(f'  scan {nxy[0]}×{nxy[1]}×{NZ_SCAN}  margin={SCAN_MARGIN}Å  extent={extent}')
    print(f'  dpos0={afm.dpos0}  h_tip∈[{h_tip[0]:.2f},{h_tip[-1]:.2f}]  h_probe∈[{h_probe[0]:.2f},{h_probe[-1]:.2f}] Å above zmax')

    afm.make_forcefield()
    FEs_3d, _ = afm.run_scan(**scan_kw)
    FEs_cs, _ = afm.run_scan_contact(**scan_kw)
    FEs_raw3d, _ = afm.get_raw_FE(**scan_kw)
    FEs_rawcs, _ = afm.get_raw_FE_contact(**scan_kw)
    Fz_3d, Fz_cs = FEs_3d[:, :, :, 2], FEs_cs[:, :, :, 2]
    Fz_raw3d, Fz_rawcs = FEs_raw3d[:, :, :, 2], FEs_rawcs[:, :, :, 2]
    df_3d, df_cs = compute_df(Fz_3d, abs(DTIP)), compute_df(Fz_cs, abs(DTIP))

    # centre pixel z-alignment: Fz vs probe height (not tip height)
    i0 = int(np.argmin(np.abs(np.arange(nxy[0]) * scan_da[0] + scan_p0[0] - apos[:, 0].mean())))
    j0 = int(np.argmin(np.abs(np.arange(nxy[1]) * scan_db[1] + scan_p0[1] - apos[:, 1].mean())))
    figz, axz = plt.subplots(1, 1, figsize=(8, 5))
    axz.plot(h_probe, Fz_raw3d[i0, j0, :], 'o-', ms=3, label='3D raw')
    axz.plot(h_probe, Fz_rawcs[i0, j0, :], 's-', ms=3, label='quasi-2D raw')
    axz.plot(h_probe, Fz_3d[i0, j0, :], 'o--', ms=3, label='3D PP')
    axz.plot(h_probe, Fz_cs[i0, j0, :], 's--', ms=3, label='quasi-2D PP')
    axz.axhline(0.0, c='k', lw=0.7, alpha=0.4)
    axz.axvspan(FIT_Z_LO, FIT_Z_HI, color='0.9', alpha=0.4, label='fit z interval')
    axz.set_xlabel('probe height h_probe = h_tip + dpos0_z  [Å above zmax]')
    axz.set_ylabel('Fz [eV/Å]')
    axz.legend(loc='best', fontsize=8)
    axz.grid(alpha=0.25)
    figz.suptitle(f'Scan z-alignment at molecule centre — dpos0_z={afm.dpos0[2]:.1f}Å', fontsize=11)
    plt.tight_layout()
    out_z = os.path.join(PLOT_DIR, 'contact_surface_scan_z_alignment.png')
    figz.savefig(out_z, dpi=150)
    plt.close(figz)
    print(f'REVIEW: {out_z}')

    sel = [i for i in [0, 4, 8, 12, 16, 20, 24] if i < NZ_SCAN]
    fig, axes = plt.subplots(3, len(sel), figsize=(2.6 * len(sel), 7.8))
    for col, iz in enumerate(sel):
        d3, dcs = Fz_3d[:, :, iz], Fz_cs[:, :, iz]
        v = max(float(np.percentile(np.abs(d3), 99)), float(np.percentile(np.abs(dcs), 99)), 1e-6)
        for row, (data, title) in enumerate([
            (d3, '3D img_FF PP'),
            (dcs, 'quasi-2D PP'),
            (dcs - d3, 'Δ (2D−3D)'),
        ]):
            ax = axes[row, col]
            d = data.T
            if row < 2:
                im = ax.imshow(d, origin='lower', extent=extent, cmap='bwr', vmin=-v, vmax=v, aspect='equal')
            else:
                ev = max(v, _rmse(dcs, d3), 1e-6)
                im = ax.imshow(d, origin='lower', extent=extent, cmap='bwr', vmin=-ev, vmax=ev, aspect='equal')
            if row == 0:
                ax.set_title(f'tip {h_tip[iz]:.1f}Å\nprobe {h_probe[iz]:.1f}Å', fontsize=7)
            if col == 0:
                ax.set_ylabel(title, fontsize=8)
            ax.tick_params(labelsize=5)
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Phase2 PP-relaxed Fz — shared ±vmax per column; tip/probe heights (dpos0_z={afm.dpos0[2]:.1f}Å)', fontsize=10)
    plt.tight_layout()
    out_fz = os.path.join(PLOT_DIR, 'pp_afm_parity_Fz_relaxed.png')
    fig.savefig(out_fz, dpi=150)
    plt.close(fig)
    print(f'REVIEW: {out_fz}')

    iz_df = min(12, NZ_SCAN - 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axes, [df_3d[:, :, iz_df], df_cs[:, :, iz_df], df_cs[:, :, iz_df] - df_3d[:, :, iz_df]], ['df 3D', 'df quasi-2D', 'Δdf']):
        d = data.T
        v = max(float(np.percentile(np.abs(d), 99)), 1e-6)
        im = ax.imshow(d, origin='lower', extent=extent, cmap='bwr', vmin=-v, vmax=v, aspect='equal')
        ax.set_title(f'{title}  probe h≈{h_probe[iz_df]:.1f}Å (tip {h_tip[iz_df]:.1f}Å)')
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out_df = os.path.join(PLOT_DIR, 'pp_afm_parity_df_relaxed.png')
    fig.savefig(out_df, dpi=150)
    plt.close(fig)
    print(f'REVIEW: {out_df}')

    rmse_fz = [_rmse(Fz_cs[:, :, iz], Fz_3d[:, :, iz]) for iz in range(NZ_SCAN)]
    out_pp = os.path.join(PLOT_DIR, 'pp_afm_parity_summary.out')
    with open(out_pp, 'w') as f:
        f.write('Phase2: PP-relaxed Fz/df — 3D img_FF vs quasi-2D (close-contact SeparableParams)\n')
        f.write(f'scan margin={SCAN_MARGIN}Å  dx={DX_SCAN}  grid={n_grid}  n_coeff={sep.n_coeff}\n')
        f.write(f'dpos0={afm.dpos0.tolist()}  h_tip range [{h_tip[0]:.2f},{h_tip[-1]:.2f}]  h_probe range [{h_probe[0]:.2f},{h_probe[-1]:.2f}]\n')
        f.write(f'mean RMSE Fz={np.mean(rmse_fz):.6e}  max={np.max(rmse_fz):.6e}\n')
        f.write('iz  h_tip  h_probe  rmse_Fz  center_Fz_3d  center_Fz_cs\n')
        for iz, r in enumerate(rmse_fz):
            f.write(f'  iz={iz:2d} h_tip={h_tip[iz]:5.2f} h_probe={h_probe[iz]:5.2f} rmse_Fz={r:.6e}  Fz3d={Fz_3d[i0,j0,iz]: .6e} Fzcs={Fz_cs[i0,j0,iz]: .6e}\n')
    print(f'REVIEW: {out_pp}')
    print(f'  Phase2 mean RMSE Fz={np.mean(rmse_fz):.4e}')
    return rmse_fz


    return rmse_fz


def plot_pic_atom_selection(apos, pic, bbox, enames=None):
    """Diagnostic: which atoms carry PIC radial basis; bucket grid + Rc support."""
    x0f, x1f, y0f, y1f = bbox
    idx_pic = np.asarray(pic.indices, dtype=int)
    mask = np.zeros(len(apos), dtype=bool)
    mask[idx_pic] = True
    pic_pos = apos[idx_pic]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    ax.scatter(apos[~mask, 0], apos[~mask, 1], c='0.75', s=8, marker='o', label=f'excluded ({(~mask).sum()})', linewidths=0)
    sc = ax.scatter(pic_pos[:, 0], pic_pos[:, 1], c=pic_pos[:, 2], s=40, cmap='viridis', edgecolors='k', linewidths=0.4, label=f'PIC basis ({len(idx_pic)})')
    for x in np.arange(x0f, x1f + 1e-9, pic.cell_size):
        ax.axvline(x, color='0.85', lw=0.4)
    for y in np.arange(y0f, y1f + 1e-9, pic.cell_size):
        ax.axhline(y, color='0.85', lw=0.4)
    th = np.linspace(0, 2 * np.pi, 48)
    for p in pic_pos[:: max(1, len(pic_pos) // 40)]:
        ax.plot(p[0] + PIC_POLY_R * np.cos(th), p[1] + PIC_POLY_R * np.sin(th), 'r-', lw=0.25, alpha=0.35)
    ax.set_xlim(x0f, x1f); ax.set_ylim(y0f, y1f)
    ax.set_aspect('equal')
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    ax.set_title('PIC atoms (color=z) + bucket grid + Rc circles')
    ax.legend(loc='upper right', fontsize=7)
    plt.colorbar(sc, ax=ax, label='z [Å]', fraction=0.046)

    ax = axes[1]
    ax.scatter(apos[~mask, 0], apos[~mask, 2], c='0.75', s=8, linewidths=0)
    ax.scatter(pic_pos[:, 0], pic_pos[:, 2], c='C3', s=35, edgecolors='k', linewidths=0.3)
    ax.set_xlabel('x [Å]'); ax.set_ylabel('z [Å]')
    ax.set_title('Side view (xz)')
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.axis('off')
    lines = [
        f'PIC contact atom selection',
        f'total atoms: {len(apos)}  →  PIC basis atoms: {len(idx_pic)}',
        f'select_contact_atoms(z_local={PIC_Z_LOCAL}Å, xy_radius={PIC_XY_RADIUS}Å)',
        f'radial Rc={PIC_POLY_R}Å  nmodes={PIC_NZ}  m_start={M_START}',
        f'bucket cell={PIC_CELL}Å  grid={pic.nbx}×{pic.nby}',
        f'coeffs: {pic.nat * PIC_NZ} (= nat × nmodes)',
        '',
        'PIC atom indices (full molecule):',
    ]
    for i, gi in enumerate(idx_pic):
        nm = enames[gi] if enames is not None else '?'
        p = apos[gi]
        lines.append(f'  [{i:3d}] mol#{gi:3d} {nm:2s}  ({p[0]:7.3f},{p[1]:7.3f},{p[2]:7.3f})')
    ax.text(0.02, 0.98, '\n'.join(lines), va='top', fontsize=7, family='monospace')
    fig.suptitle('PIC basis atom coverage — radial modes anchored on contact shell', fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, 'contact_surface_pic_atoms.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'REVIEW: {out_png}')

    out_txt = os.path.join(PLOT_DIR, 'contact_surface_pic_atoms.out')
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'REVIEW: {out_txt}')


def phase_pic_fit_and_parity(afm, apos, enames, bbox):
    """Fit PIC on contact atoms; unrelaxed E/Fz parity vs brute."""
    x0f, x1f, y0f, y1f = bbox
    extent = [x0f, x1f, y0f, y1f]
    zmax = float(np.max(apos[:, 2]))
    print(f'  PIC fit: z∈[{FIT_Z_LO},{FIT_Z_HI}]Å  Rc={PIC_POLY_R}Å  nmodes={PIC_NZ}  cell={PIC_CELL}Å  z_local={PIC_Z_LOCAL}Å')
    pic = afm.fit_pic_contact_surface(margin=MARGIN, poly_R=PIC_POLY_R, m_start=M_START, nz=PIC_NZ, cell_size=PIC_CELL, z_local=PIC_Z_LOCAL, xy_radius=PIC_XY_RADIUS, fit_z_adaptive=(FIT_Z_LO, FIT_Z_HI, FIT_DZ_LO, FIT_DZ_HI), fit_dx=FIT_DX, fit_dy=FIT_DY, fit_boltzmann=True, fit_boltzmann_T=FIT_BOLTZMANN_T, n_iter=100, reg=PIC_REG, brute_ref='afm', bPrint=True)
    plot_pic_atom_selection(apos, pic, bbox, enames=enames)

    def eval_brute(pts):
        E, F = afm._brute_afm_morse_c_queries(pts)
        return E.astype(np.float64), F.astype(np.float64)

    def eval_pic(pts):
        E, F = afm._cs_fit_helper().eval_pic(pts, pic)
        return E.astype(np.float64), F.astype(np.float64)

    contact_xy = apos[pic.indices, :2]
    n_z = len(CLOSE_Z_OFFS)
    fig, axes = plt.subplots(4, n_z, figsize=(4.2 * n_z, 14))
    summary = []
    for col, zoff in enumerate(CLOSE_Z_OFFS):
        z_scan = zmax + zoff
        _, _, E_ref, Fz_ref = _eval_slice_maps(eval_brute, x0f, x1f, y0f, y1f, z_scan, PLOT_DX, PLOT_DX)
        _, _, E_pic, Fz_pic = _eval_slice_maps(eval_pic, x0f, x1f, y0f, y1f, z_scan, PLOT_DX, PLOT_DX)
        rmse_E = _rmse(E_pic, E_ref)
        rmse_Fz = _rmse(Fz_pic, Fz_ref)
        summary.append(f'z_off={zoff:.1f}Å  rmse_E={rmse_E:.4e}  rmse_Fz={rmse_Fz:.4e}')
        print(f'  PIC close parity {summary[-1]}')
        ev = max(float(np.percentile(E_ref, 99)), 1e-9)
        fv = max(float(np.percentile(np.abs(Fz_ref), 99)), 1e-9)
        for row, (data, title, vmin, vmax, cmap, label) in enumerate([
            (E_ref, 'brute E', 0, ev, 'magma_r', 'E [eV]'),
            (E_pic, 'PIC E', 0, ev, 'magma_r', 'E [eV]'),
            (Fz_ref, 'brute Fz', -fv, fv, 'RdBu_r', 'Fz [eV/Å]'),
            (Fz_pic, 'PIC Fz', -fv, fv, 'RdBu_r', 'Fz [eV/Å]'),
        ]):
            ax = axes[row, col]
            im = ax.imshow(data.T, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            ax.scatter(contact_xy[:, 0], contact_xy[:, 1], c='w', s=4, marker='.', alpha=0.5, linewidths=0)
            if row == 0:
                ax.set_title(f'z_max+{zoff:.1f}Å')
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.set_xlabel('x [Å]')
            plt.colorbar(im, ax=ax, fraction=0.046, label=label)
    fig.suptitle(f'PIC unrelaxed parity — {pic.nat} atoms × {PIC_NZ} modes  Rc={PIC_POLY_R}Å', fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, 'contact_surface_pic_close_parity.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'REVIEW: {out_png}')

    out_sum = os.path.join(PLOT_DIR, 'contact_surface_pic_summary.out')
    with open(out_sum, 'w') as f:
        f.write('PIC unrelaxed close-contact parity vs AFM brute\n')
        f.write(f'nat={pic.nat} nmodes={PIC_NZ} Rc={PIC_POLY_R} cell={PIC_CELL}\n')
        for line in summary:
            f.write(line + '\n')
    print(f'REVIEW: {out_sum}')
    return pic


def phase3_pp_afm_pic(pic, apos):
    """PP-relaxed scan: 3D img_FF vs radial PIC."""
    from spammm.SPM.AFM import AFMulator, compute_df

    print('  Phase3 PIC: PP-AFM relaxed scan parity (reuse PIC fit, no refit)')
    afm = AFMulator(use_morse=True, use_fire=False)
    afm.load_molecule(PTCDA)
    afm.assign_params(params_path=PARAMS, tip_R=0.0, tip_E=1.0)
    afm.tipQs[:] = 0.0
    n_grid = _grid_n(apos, MARGIN, Z_TOP, DX_GRID)
    afm.setup_grid(n=n_grid, margin=MARGIN, z_top=Z_TOP, shift_atoms=False)
    afm.fit_pic_contact_surface(pic=pic)

    nxy, scan_p0, scan_da, scan_db, extent, mol_z = afm.scan_bbox(margin=SCAN_MARGIN, dx=DX_SCAN)
    scan_kw = dict(nxy=nxy, nz=NZ_SCAN, dtip=DTIP, scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db)
    z0_tip = float(scan_p0[2])
    h_tip = z0_tip + np.arange(NZ_SCAN) * DTIP - mol_z
    h_probe = h_tip + float(afm.dpos0[2])
    print(f'  scan {nxy[0]}×{nxy[1]}×{NZ_SCAN}  PIC nat={pic.nat}')

    afm.make_forcefield()
    FEs_3d, _ = afm.run_scan(**scan_kw)
    FEs_pic, _ = afm.run_scan_pic(**scan_kw)
    Fz_3d, Fz_pic = FEs_3d[:, :, :, 2], FEs_pic[:, :, :, 2]
    df_3d, df_pic = compute_df(Fz_3d, abs(DTIP)), compute_df(Fz_pic, abs(DTIP))

    sel = [i for i in [0, 4, 8, 12, 16, 20, 24] if i < NZ_SCAN]
    fig, axes = plt.subplots(3, len(sel), figsize=(2.6 * len(sel), 7.8))
    for col, iz in enumerate(sel):
        d3, dp = Fz_3d[:, :, iz], Fz_pic[:, :, iz]
        v = max(float(np.percentile(np.abs(d3), 99)), float(np.percentile(np.abs(dp), 99)), 1e-6)
        for row, (data, title) in enumerate([(d3, '3D img_FF PP'), (dp, 'PIC PP'), (dp - d3, 'Δ (PIC−3D)')]):
            ax = axes[row, col]
            d = data.T
            if row < 2:
                im = ax.imshow(d, origin='lower', extent=extent, cmap='bwr', vmin=-v, vmax=v, aspect='equal')
            else:
                ev = max(v, _rmse(dp, d3), 1e-6)
                im = ax.imshow(d, origin='lower', extent=extent, cmap='bwr', vmin=-ev, vmax=ev, aspect='equal')
            if row == 0:
                ax.set_title(f'probe {h_probe[iz]:.1f}Å', fontsize=7)
            if col == 0:
                ax.set_ylabel(title, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Phase3 PP-relaxed Fz — 3D vs PIC ({pic.nat} basis atoms)', fontsize=10)
    plt.tight_layout()
    out_fz = os.path.join(PLOT_DIR, 'pp_afm_parity_Fz_pic_relaxed.png')
    fig.savefig(out_fz, dpi=150)
    plt.close(fig)
    print(f'REVIEW: {out_fz}')

    rmse_fz = [_rmse(Fz_pic[:, :, iz], Fz_3d[:, :, iz]) for iz in range(NZ_SCAN)]
    out_pp = os.path.join(PLOT_DIR, 'pp_afm_parity_pic_summary.out')
    with open(out_pp, 'w') as f:
        f.write('Phase3: PP-relaxed Fz — 3D img_FF vs radial PIC\n')
        f.write(f'PIC nat={pic.nat} nmodes={PIC_NZ} Rc={PIC_POLY_R}\n')
        f.write(f'mean RMSE Fz={np.mean(rmse_fz):.6e}  max={np.max(rmse_fz):.6e}\n')
    print(f'REVIEW: {out_pp}')
    print(f'  Phase3 PIC mean RMSE Fz={np.mean(rmse_fz):.4e}')
    return rmse_fz


# ---------------------------------------------------------------------------
# Toy systems: rigid FF parity (brute vs contact-sep vs GridFF) — no PP relax
# ---------------------------------------------------------------------------

TOY_DIR = os.path.join(PLOT_DIR, 'toys')

# PTCDA-tuned atom-z legacy vs assembly-old vs spherical contact (intended)
FIT_KNOBS = {
    'atomz_ptcda': dict(margin=4.0, bspl_dx=0.2, poly_R=5.0, poly_z0=1.0, m_start=4, nz=6,
                        fit_z_adaptive=(1.0, 6.0, 0.1, 1.0), fit_force_weight=1.0, n_iter=80,
                        h0_mode='atom_z'),
    'atomz_assembly_old': dict(margin=3.0, bspl_dx=0.2, poly_R=10.0, poly_z0=0.0, m_start=4, nz=5,
                               fit_z_adaptive=(1.0, 5.0, 0.2, 0.8), fit_force_weight=0.5, n_iter=60,
                               h0_mode='atom_z'),
    'spheres': dict(margin=4.0, bspl_dx=0.2, poly_R=4.0, poly_z0=0.0, m_start=4, nz=6,
                    fit_z_adaptive=(0.05, 4.0, 0.1, 0.8), fit_force_weight=1.0, n_iter=80,
                    h0_mode='spheres', h0_R_scale=0.75),
}

TOY_XY_ZOFFS = (2.0, 3.0, 4.0)  # h_probe = z - zmax for xy slices
TOY_PROFILE_H = np.arange(0.8, 8.01, 0.05)


def _write_toy_xyz(path, apos, enames, qs):
    apos = np.asarray(apos, dtype=np.float64).reshape(-1, 3)
    qs = np.asarray(qs, dtype=np.float64).reshape(-1)
    with open(path, 'w') as f:
        f.write(f'{len(apos)}\n')
        f.write('toy contact-surface FF parity\n')
        for i, en in enumerate(enames):
            x, y, z = apos[i]
            f.write(f'{en} {x:.6f} {y:.6f} {z:.6f} {qs[i]:.6f}\n')


def _make_afm_from_xyz(xyz_path, tip_R=0.0, tip_E=1.0, zero_tip_q=True):
    from spammm.SPM.AFM import AFMulator
    afm = AFMulator(use_morse=True, use_fire=False)
    afm.load_molecule(xyz_path)
    afm.assign_params(params_path=PARAMS, tip_R=tip_R, tip_E=tip_E)
    if zero_tip_q:
        afm.tipQs[:] = 0.0
    return afm


def _fit_sep(afm, knobs):
    kw = dict(knobs)
    return afm.fit_contact_surface(**kw, fit_boltzmann=True, brute_ref='afm', bPrint=True)


def _ensure_gridff(afm, dx=0.25, margin=4.0, z_top=10.0):
    """Build 3D img_FF once in world frame (shift_atoms=False)."""
    if getattr(afm, '_toy_gridff_ready', False):
        return
    n_grid = _grid_n(afm.atoms_arr[:, :3], margin, z_top, dx)
    afm.setup_grid(n=n_grid, margin=margin, z_top=z_top, shift_atoms=False)
    afm.make_forcefield()
    afm._toy_gridff_ready = True
    print(f'  GridFF ready n={tuple(int(v) for v in afm.n)} dx≈{dx}Å (world frame)')


def _sample_gridff_raw(afm, queries):
    """Sample existing GridFF at absolute probe positions via get_raw_FE (tip+dpos0)."""
    queries = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, 3)
    d0 = float(afm.dpos0[2])
    out_E = np.zeros(len(queries), dtype=np.float64)
    out_Fz = np.zeros(len(queries), dtype=np.float64)
    xy = np.round(queries[:, :2], 6)
    keys = {}
    for i, (x, y) in enumerate(xy):
        keys.setdefault((float(x), float(y)), []).append(i)
    zd = np.zeros(3, dtype=np.float32)
    for (x, y), idxs in keys.items():
        zs = queries[idxs, 2]
        order = np.argsort(-zs)  # top → bottom
        idxs_o = [idxs[j] for j in order]
        zs_o = zs[order]
        if len(zs_o) == 1:
            dtip, nz, hp0 = -0.05, 1, float(zs_o[0])
        else:
            dtip, nz, hp0 = float(zs_o[1] - zs_o[0]), len(zs_o), float(zs_o[0])
        scan0 = np.array([x, y, hp0 - d0], dtype=np.float32)
        Fraw, _ = afm.get_raw_FE(nxy=(1, 1), nz=nz, dtip=dtip, scan_p0=scan0, scan_da=zd, scan_db=zd)
        for k, ii in enumerate(idxs_o):
            out_E[ii] = float(Fraw[0, 0, k, 3])
            out_Fz[ii] = float(Fraw[0, 0, k, 2])
    return out_E, out_Fz


def run_one_toy(name, apos, enames, qs, knobs_name, knobs, tip_R=0.0, tip_E=1.0):
    """Rigid FF: brute vs contact-sep vs GridFF — xy maps + E/Fz(z) at atom.
    tip_R=0, tip_E=1 matches PTCDA harness (deeper Morse well; easier visual parity)."""
    sub = os.path.join(TOY_DIR, f'{name}_{knobs_name}')
    os.makedirs(sub, exist_ok=True)
    xyz = os.path.join(sub, 'toy.xyz')
    _write_toy_xyz(xyz, apos, enames, qs)

    afm = _make_afm_from_xyz(xyz, tip_R=tip_R, tip_E=tip_E, zero_tip_q=True)
    apos_w = afm.atoms_arr[:, :3].copy()
    zmax = float(apos_w[:, 2].max())
    R0 = float(afm.cLJs_arr[0, 0])  # Morse R0 of first atom
    print(f'\n=== TOY {name} / knobs={knobs_name}  nat={len(apos_w)} zmax={zmax:.3f} R0[0]={R0:.3f} ===')
    print(f'  knobs: poly_z0={knobs["poly_z0"]} poly_R={knobs["poly_R"]} nz={knobs["nz"]} '
          f'bspl_dx={knobs["bspl_dx"]} fit_z={knobs["fit_z_adaptive"]} force_w={knobs["fit_force_weight"]}')

    sep = _fit_sep(afm, knobs)
    h0 = sep.h0_map
    print(f'  sep: bspl={sep.ncx}x{sep.ncy} nz={sep.nz} n_coeff={sep.n_coeff}  '
          f'h0=[{h0.min():.3f},{h0.max():.3f}]  poly_z0={sep.poly_z0} poly_R={sep.poly_R}  '
          f'h0_mode={getattr(sep, "h0_mode", "?")} z_ref={getattr(sep, "z_ref", float("nan")):.3f}')

    margin = float(knobs['margin'])
    _ensure_gridff(afm, dx=0.25, margin=margin + 1.0, z_top=10.0)

    # --- 1D profile above atom 0 (and midpoint for 2-atom) ---
    sites = [('atom0', apos_w[0, :2])]
    if len(apos_w) >= 2:
        mid = 0.5 * (apos_w[0, :2] + apos_w[1, :2])
        sites.append(('mid', mid))
        sites.append(('atom1', apos_w[1, :2]))

    hp = TOY_PROFILE_H
    fig, axes = plt.subplots(2, len(sites), figsize=(4.2 * len(sites), 7.2), sharex=True, squeeze=False)
    summary = []
    z_ref = float(getattr(sep, 'z_ref', zmax))
    # fit_z_adaptive offsets are above z_ref (=h0_max for spheres, zmax for atom_z)
    flo = (z_ref - zmax) + knobs['fit_z_adaptive'][0]
    fhi = (z_ref - zmax) + knobs['fit_z_adaptive'][1]
    for si, (sname, xy) in enumerate(sites):
        x, y = float(xy[0]), float(xy[1])
        zq = np.column_stack([np.full(len(hp), x), np.full(len(hp), y), zmax + hp]).astype(np.float32)
        E_br, F_br = afm._brute_afm_morse_c_queries(zq)
        E_cs, F_cs = afm._cs_fit_helper().eval_separable(zq, sep)
        E_g, Fz_g = _sample_gridff_raw(afm, zq)
        Fz_br, Fz_cs = F_br[:, 2], F_cs[:, 2]

        mask = (hp >= flo) & (hp <= fhi)
        shifts = np.linspace(-1.5, 1.5, 601)
        best = (float('inf'), 0.0)
        for dz in shifts:
            shifted = np.interp(hp[mask], hp + dz, Fz_cs, left=np.nan, right=np.nan)
            ok = np.isfinite(shifted)
            if np.any(ok):
                err = _rmse(shifted[ok], Fz_br[mask][ok])
                if err < best[0]:
                    best = (err, float(dz))

        i_well_br = int(np.argmin(np.where(mask, E_br, np.inf)))
        i_well_cs = int(np.argmin(np.where(mask, E_cs, np.inf)))
        line = (f'{sname} xy=({x:.2f},{y:.2f})  Ewell_br={E_br[i_well_br]:.4f}@{hp[i_well_br]:.2f}Å  '
                f'Ewell_cs={E_cs[i_well_cs]:.4f}@{hp[i_well_cs]:.2f}Å  '
                f'Δh_well={hp[i_well_cs]-hp[i_well_br]:+.3f}Å  best_Fz_shift={best[1]:+.3f}Å  '
                f'rmse_E_fit={_rmse(E_cs[mask], E_br[mask]):.3e} rmse_Fz_fit={_rmse(Fz_cs[mask], Fz_br[mask]):.3e}  '
                f'rmse_E_g_fit={_rmse(E_g[mask], E_br[mask]):.3e} rmse_Fz_g_fit={_rmse(Fz_g[mask], Fz_br[mask]):.3e}')
        summary.append(line)
        print(f'  {line}')

        axE, axF = axes[0, si], axes[1, si]
        axE.plot(hp, E_br, 'k-', lw=1.8, label='brute')
        axE.plot(hp, E_cs, 'C0--', lw=1.3, label='contact-sep')
        axE.plot(hp, E_g, 'C1-.', lw=1.2, label='GridFF raw')
        axE.axvline(R0, color='0.5', ls=':', lw=0.8, label=f'R0={R0:.2f}')
        axE.axvline(h0.max() - zmax, color='C2', ls='-.', lw=1.0, alpha=0.8, label=f'h0_max−zmax={h0.max()-zmax:.2f}')
        axE.axvline(sep.poly_z0 + (h0.max() - zmax), color='C3', ls=':', lw=0.8, alpha=0.7, label='h0+poly_z0')
        axE.axvline(sep.poly_z0 + sep.poly_R + (h0.max() - zmax), color='C3', ls='--', lw=0.7, alpha=0.5, label='h0+z0+Rc')
        axE.axvspan(flo, fhi, color='0.9', alpha=0.45)
        # Zoom to fit-window energy scale (Pauli wall below z0 dwarfs meV well on full scale)
        e_win = np.concatenate([E_br[mask], E_cs[mask], E_g[mask]])
        e_lo = float(np.min(e_win))
        e_hi = float(np.percentile(e_win, 99))
        pad = max(0.05 * (e_hi - e_lo), 1e-3)
        axE.set_ylim(e_lo - pad, e_hi + pad)
        axE.set_xlim(flo - 0.2, fhi + 0.5)
        axE.set_title(f'{sname}')
        axE.set_ylabel('E [eV]')
        axE.legend(fontsize=6); axE.grid(True, alpha=0.3); axE.axhline(0, color='k', lw=0.4)

        axF.plot(hp, Fz_br, 'k-', lw=1.8, label='brute Fz')
        axF.plot(hp, Fz_cs, 'C0--', lw=1.3, label='contact Fz')
        axF.plot(hp, Fz_g, 'C1-.', lw=1.2, label='GridFF Fz')
        axF.axvline(R0, color='0.5', ls=':', lw=0.8)
        axF.axvspan(flo, fhi, color='0.9', alpha=0.45)
        f_win = np.concatenate([Fz_br[mask], Fz_cs[mask], Fz_g[mask]])
        fv = max(float(np.percentile(np.abs(f_win), 99)), 1e-6)
        axF.set_ylim(-1.2 * fv, 1.2 * fv)
        axF.set_xlim(flo - 0.2, fhi + 0.5)
        axF.set_xlabel('h_probe = z − zmax [Å]  (zoom=fit window)')
        axF.set_ylabel('Fz [eV/Å]')
        axF.legend(fontsize=6); axF.grid(True, alpha=0.3); axF.axhline(0, color='k', lw=0.4)

    fig.suptitle(f'{name} / {knobs_name}  rigid FF  poly_z0={sep.poly_z0} Rc={sep.poly_R} nz={sep.nz}  '
                 f'bspl={sep.ncx}×{sep.ncy}@{sep.dx}Å', fontsize=10)
    fig.tight_layout()
    out_prof = os.path.join(sub, 'rigid_EFz_profiles.png')
    fig.savefig(out_prof, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {out_prof}')

    # --- XY slices: brute | contact | Δ  (GridFF≈brute checked on profiles) ---
    x0f = float(apos_w[:, 0].min()) - margin
    x1f = float(apos_w[:, 0].max()) + margin
    y0f = float(apos_w[:, 1].min()) - margin
    y1f = float(apos_w[:, 1].max()) + margin
    extent = [x0f, x1f, y0f, y1f]
    dx_plot = 0.1

    def eval_brute(pts):
        E, F = afm._brute_afm_morse_c_queries(pts)
        return E.astype(np.float64), F.astype(np.float64)

    def eval_sep(pts):
        E, F = afm._cs_fit_helper().eval_separable(pts, sep)
        return E.astype(np.float64), F.astype(np.float64)

    n_z = len(TOY_XY_ZOFFS)
    fig, axes = plt.subplots(3, n_z, figsize=(3.6 * n_z, 9.5), squeeze=False)
    for col, zoff in enumerate(TOY_XY_ZOFFS):
        z_scan = zmax + zoff
        _, _, _, Fz_br = _eval_slice_maps(eval_brute, x0f, x1f, y0f, y1f, z_scan, dx_plot, dx_plot)
        _, _, _, Fz_cs = _eval_slice_maps(eval_sep, x0f, x1f, y0f, y1f, z_scan, dx_plot, dx_plot)
        dFz = Fz_cs - Fz_br
        fv = max(float(np.percentile(np.abs(Fz_br), 99)), 1e-9)
        ev = max(float(np.percentile(np.abs(dFz), 99)), 1e-12)
        for row, (data, ylab, vmin, vmax) in enumerate([
            (Fz_br, 'brute Fz', -fv, fv),
            (Fz_cs, 'contact Fz', -fv, fv),
            (dFz, 'Δ Fz (cs−br)', -3 * ev, 3 * ev),
        ]):
            ax = axes[row, col]
            im = ax.imshow(data.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
            ax.scatter(apos_w[:, 0], apos_w[:, 1], c='k', s=18, marker='x', zorder=5)
            if row == 0:
                ax.set_title(f'h={zoff:.1f}Å', fontsize=9)
            if col == 0:
                ax.set_ylabel(ylab, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
        print(f'  xy h={zoff:.1f}: rmse_Fz(cs)={_rmse(Fz_cs, Fz_br):.3e}')
    fig.suptitle(f'{name}/{knobs_name} rigid Fz(xy) — brute | contact-sep | Δ  (GridFF on profiles)', fontsize=11)
    fig.tight_layout()
    out_xy = os.path.join(sub, 'rigid_Fz_xy.png')
    fig.savefig(out_xy, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {out_xy}')

    out_txt = os.path.join(sub, 'SUMMARY.out')
    with open(out_txt, 'w') as f:
        f.write(f'toy={name} knobs={knobs_name}\n')
        f.write(f'nat={len(apos_w)} zmax={zmax:.4f} R0[0]={R0:.4f} tipQs=0\n')
        f.write(f'poly_z0={sep.poly_z0} poly_R={sep.poly_R} nz={sep.nz} m_start={sep.m_start}\n')
        f.write(f'bspl={sep.ncx}x{sep.ncy} dx={sep.dx} n_coeff={sep.n_coeff}\n')
        f.write(f'h0 min/max={h0.min():.4f}/{h0.max():.4f}\n')
        f.write(f'fit_z_adaptive={knobs["fit_z_adaptive"]} force_weight={knobs["fit_force_weight"]}\n')
        f.write('NOTE: dz_basis = z - h0(x,y) - poly_z0; field→0 for dz > poly_z0+poly_R\n')
        f.write('NOTE: rigid FF only (eval_separable / brute / get_raw_FE) — no PP relax\n')
        for line in summary:
            f.write(line + '\n')
        f.write(f'REVIEW: {out_prof}\nREVIEW: {out_xy}\n')
    print(f'REVIEW: {out_txt}')
    return summary


def run_toys():
    """1-atom (q=0) and 2-atom (charged) rigid FF parity under both fit knob sets."""
    os.makedirs(TOY_DIR, exist_ok=True)
    print(f'Toy rigid FF parity → {TOY_DIR}')
    print('Method recall: E(x,y,z)=Σ_kz B_xy(x,y)·c_kz · φ_kz(dz), dz=z−h0(x,y)−poly_z0')
    print('  φ: doubling poly in t=1−clip((dz−poly_z0)/poly_R); B: cubic B-spline on ncx×ncy knots')

    cases = [
        ('C1_q0', [[0., 0., 0.]], ['C'], [0.0]),
        ('C2_qpm', [[-1.2, 0., 0.], [1.2, 0., 0.]], ['C', 'C'], [+0.5, -0.5]),
    ]
    # Focus: atom_z (soft) vs spheres (intended contact). Skip full matrix of old assembly.
    knobs_run = {k: FIT_KNOBS[k] for k in ('atomz_ptcda', 'spheres')}
    all_lines = []
    for name, apos, enames, qs in cases:
        for knobs_name, knobs in knobs_run.items():
            lines = run_one_toy(name, apos, enames, qs, knobs_name, knobs)
            all_lines.extend([f'[{name}/{knobs_name}] {L}' for L in lines])

    index = os.path.join(TOY_DIR, 'INDEX.out')
    with open(index, 'w') as f:
        f.write('Rigid FF toys. Softness root cause: legacy h0=max atom-z (not spherical contact).\n')
        f.write('spheres: h0 = ray vs Morse-R0 spheres; fit offsets above h0_max.\n')
        f.write('Compare Δh_well / Fz shape: spheres should be much less soft than atomz_*.\n\n')
        for L in all_lines:
            f.write(L + '\n')
    print(f'REVIEW: {index}')
    print('Toys done.')


def main_ptcda():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f'PTCDA contact surface — close-contact fit + parity → {PLOT_DIR}')
    apos, reqs, enames, lvec, qs = load_atom_data(PTCDA)
    print(f'  atoms={len(apos)}  z=[{apos[:,2].min():.2f},{apos[:,2].max():.2f}]')

    from spammm.SPM.AFM import AFMulator
    afm = AFMulator(use_morse=True, use_fire=False)
    afm.load_molecule(PTCDA)
    afm.assign_params(params_path=PARAMS, tip_R=0.0, tip_E=1.0)
    afm.tipQs[:] = 0.0

    sep, zmax, bbox = phase1_fit_close(afm, apos)
    plot_z_basis()
    phase1_close_parity(afm, apos, sep, zmax, bbox)
    phase1_z_profile(afm, apos, sep, zmax)
    phase1_zstack_slices(afm, apos, sep, zmax, bbox)
    if os.environ.get('RUN_CONTACT_PP', '0') == '1':
        phase2_pp_afm_parity(sep, apos)
    else:
        print('Skipping Phase2 PP-relaxed parity by default; set RUN_CONTACT_PP=1 after raw E/Fz parity is satisfactory.')

    print('--- PIC (radial atom basis) ---')
    if os.environ.get('RUN_CONTACT_PIC', '0') == '1':
        pic = phase_pic_fit_and_parity(afm, apos, enames, bbox)
        if os.environ.get('RUN_CONTACT_PP', '0') == '1':
            phase3_pp_afm_pic(pic, apos)
    else:
        print('Skipping PIC by default; set RUN_CONTACT_PIC=1 to fit/compare radial PIC.')
    print('Done.')


def main():
    import argparse
    p = argparse.ArgumentParser(description='Contact-surface fit/parity (PTCDA or toy rigid FF)')
    p.add_argument('--toys', action='store_true',
                   help='1-atom / 2-atom rigid FF: brute vs contact vs GridFF (PTCDA + assembly knobs)')
    args = p.parse_args()
    if args.toys:
        run_toys()
    else:
        main_ptcda()


if __name__ == '__main__':
    main()
