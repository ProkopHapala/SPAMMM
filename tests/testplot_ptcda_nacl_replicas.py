#!/usr/bin/env python
"""PTCDA@NaCl FAF: 256×256 rigid-body imaging over 2×2 unit cell (PBC).

AFM-like: one distant anhydride O is spring-pinned at the scan (x,y,z_pin).
Molecule starts flat with that O on the tip. Relax with GPU Newton then FIRE.
Plots E, |F|, Fz, |τ|, COM_z, tilt, converged, iters, and peak |F| over the
trajectory (crash / hard-contact diagnostic).

Run (needs NVIDIA OpenCL visible):
  python tests/testplot_ptcda_nacl_replicas.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from spammm.plotUtils import plot_2d_scalar
from spammm.SPM.AFM_utils import imshow_afm
from spammm.surfaces.FoldedRigid import (
    fit_folded_for_molecule, save_fit, load_fit, setup_rigid_folded_replicas,
    NACL_SUBSTRATE, Z_SURF_TOP, LATTICE_A,
)

PTCDA = os.path.join(_ROOT, 'data', 'xyz', 'PTCDA.xyz')
OUT = os.path.join(_ROOT, 'debug', 'testplot_ptcda_nacl_replicas')
FIT_CACHE = os.path.join(OUT, 'ptcda_nacl_faf.npz')
FIT_FALLBACK = os.path.join(_ROOT, 'debug', 'test_relax_ptcda_faf', 'ptcda_nacl_faf.npz')

# Scan: NaCl 2×2 cell (a=4 Å → L=8 Å), 256×256 pixels, PBC
NXY = 256
NCELL = 2
PIN_ATOM = 24          # anhydride O at one end (same as test_manipulation_ptcda_nacl)
Z_PIN_REL = 6.0        # tip height above surface [Å] — holds molecule off hard crash
K_SPRING = 10.0        # eV/Å²
NEWTON_NITER = 80
FIRE_NITER = 4000
NEWTON_EPS = 0.1
NEWTON_TRUST = 0.1
NEWTON_LAMBDA_DAMPED = 1.0
NEWTON_LAMBDA_FINISH = 1e-2
FIRE_DT = 0.02
FIRE_MASS = 4.0        # reduced effective mass; inertia is scaled consistently
F_TOL = 1e-4
T_TOL = 1e-4


def _ensure_fit():
    os.makedirs(OUT, exist_ok=True)
    for path in (FIT_CACHE, FIT_FALLBACK):
        if os.path.isfile(path):
            print(f'[FAF] load {path}', flush=True)
            fit = load_fit(path)
            if path != FIT_CACHE:
                save_fit(fit, FIT_CACHE)
            return fit
    print('[FAF] fitting PTCDA on NaCl …', flush=True)
    t0 = time.perf_counter()
    fit = fit_folded_for_molecule(PTCDA, substrate_file=NACL_SUBSTRATE, z_range_rel=(1.2, 8.0))
    print(f'[FAF] done {time.perf_counter()-t0:.1f}s  ntypes={fit["coeffs"].shape[0]} '
          f'nbasis={fit["coeffs"].shape[1]}', flush=True)
    save_fit(fit, FIT_CACHE)
    print(f'REVIEW: {FIT_CACHE}', flush=True)
    return fit


def _tilt_deg(quats):
    qx, qy = quats[:, 0], quats[:, 1]
    cz = 1.0 - 2.0 * (qx * qx + qy * qy)
    return np.degrees(np.arccos(np.clip(np.abs(cz), 0.0, 1.0)))


def _reshape(flat, nx, ny):
    return np.asarray(flat, dtype=np.float64).reshape(ny, nx)


def _setup(fit, xs, ys, quat=None):
    z_pin = Z_SURF_TOP + Z_PIN_REL
    return setup_rigid_folded_replicas(
        fit, xs, ys, z_init=Z_PIN_REL, quats=quat, mass_trans=FIRE_MASS, pin_atom_idx=PIN_ATOM, z_pin=z_pin, k_spring=K_SPRING,
    )


def _constraint_reaction(out, xs, ys):
    """Force exerted by the pinned atom on the external constraint/tip."""
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    anchor = np.column_stack((XX.ravel(), YY.ravel(), np.full(XX.size, Z_SURF_TOP + Z_PIN_REL)))
    return K_SPRING * (out['atom_positions'][:, PIN_ATOM, :3] - anchor)


def _summarize(out, label, xs=None, ys=None, f_tol=F_TOL, t_tol=T_TOL):
    F = out['body_force'][:, :3]
    T = out['body_torque'][:, :3]
    E = out['body_force'][:, 3]
    Fmag = np.linalg.norm(F, axis=1)
    Tmag = np.linalg.norm(T, axis=1)
    conv = (Fmag < f_tol) & (Tmag < t_tol)
    iters = out['body_torque'][:, 3]
    Fmax = out['lin_mom'][:, 3]  # peak |F| written by kernel into vposs.w
    print(f'[{label}] n={len(E)}  frac_conv={conv.mean():.3f}  '
          f'|F| med={np.median(Fmag):.3e} max={Fmag.max():.3e}  '
          f'Fmax_traj med={np.median(Fmax):.3e} max={Fmax.max():.3e}  '
          f'|τ| med={np.median(Tmag):.3e} max={Tmag.max():.3e}  '
          f'E med={np.median(E):.4f}  '
          f'iters med={np.median(iters):.0f} max={iters.max():.0f}  '
          f'COM_z med={np.median(out["pos"][:,2]) - Z_SURF_TOP:.3f}', flush=True)
    maps = dict(F=F, T=T, E=E, Fmag=Fmag, Tmag=Tmag, conv=conv.astype(np.float64),
                iters=iters, Fmax=Fmax, pos=out['pos'], quats=out['quats'])
    if xs is not None and ys is not None:
        maps['Fconstraint'] = _constraint_reaction(out, xs, ys)
    return maps


def run_staged_newton(rbd, niter=NEWTON_NITER, f_tol=F_TOL, t_tol=T_TOL):
    """Basin-preserving damped Newton, then fast Newton for unfinished pixels."""
    out = rbd.run_folded_newton_replicas(niter=niter, eps_t=NEWTON_EPS, eps_r=NEWTON_EPS, trust0=NEWTON_TRUST, lambda0=NEWTON_LAMBDA_DAMPED, f_tol=f_tol, t_tol=t_tol)
    Fmag = np.linalg.norm(out['body_force'][:, :3], axis=1)
    Tmag = np.linalg.norm(out['body_torque'][:, :3], axis=1)
    nfail = int(np.count_nonzero((Fmag >= f_tol) | (Tmag >= t_tol)))
    print(f'[Newton damped] unconverged after {niter}: {nfail}/{len(Fmag)}', flush=True)
    if nfail:
        out = rbd.run_folded_newton_replicas(niter=niter, eps_t=NEWTON_EPS, eps_r=NEWTON_EPS, trust0=NEWTON_TRUST, lambda0=NEWTON_LAMBDA_FINISH, f_tol=f_tol, t_tol=t_tol)
    return out


def _plot_maps(maps, xs, ys, prefix, extent):
    nx, ny = len(xs), len(ys)
    panels = [
        ('E', _reshape(maps['E'], nx, ny), 'eV', 'viridis', False),
        ('Fmag', _reshape(maps['Fmag'], nx, ny), 'eV/Å', 'hot', False),
        ('Fmax_traj', _reshape(maps['Fmax'], nx, ny), 'eV/Å', 'hot', False),
        ('Fz', _reshape(maps['F'][:, 2], nx, ny), 'eV/Å', 'RdBu_r', True),
        ('Fconstraint_z', _reshape(maps['Fconstraint'][:, 2], nx, ny), 'eV/Å', 'RdBu_r', True),
        ('Fconstraint_mag', _reshape(np.linalg.norm(maps['Fconstraint'], axis=1), nx, ny), 'eV/Å', 'hot', False),
        ('Tmag', _reshape(maps['Tmag'], nx, ny), 'eV', 'hot', False),
        ('COM_z', _reshape(maps['pos'][:, 2] - Z_SURF_TOP, nx, ny), 'Å above surf', 'viridis', False),
        ('tilt_deg', _reshape(_tilt_deg(maps['quats']), nx, ny), 'deg', 'magma', False),
        ('converged', _reshape(maps['conv'], nx, ny), '0/1', 'gray', False),
        ('iters', _reshape(maps['iters'], nx, ny), 'steps', 'viridis', False),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(19, 14))
    for ax, (name, data, zlab, cmap, sym) in zip(axes.ravel(), panels):
        finite = data[np.isfinite(data)]
        if sym:
            v = max(abs(finite.min()), abs(finite.max()), 1e-12) if len(finite) else 1.0
            vmin, vmax = -v, v
        else:
            vmin, vmax = (finite.min(), finite.max()) if len(finite) else (0.0, 1.0)
            if vmax - vmin < 1e-30:
                vmax = vmin + 1.0
        if name in ('E', 'Fz', 'Fconstraint_z', 'Fconstraint_mag'):
            im = imshow_afm(ax, data, extent=extent, cmap=cmap, symmetric=sym, title='', colorbar=False, transpose=False)
        else:
            im = ax.imshow(data, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'{prefix}: {name}')
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(OUT, f'{prefix}_maps.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {path}', flush=True)
    for name, data, zlab, cmap, sym in panels[:3]:
        if name == 'E':
            fig, ax = plt.subplots(figsize=(6, 5))
            imshow_afm(ax, data, extent=extent, cmap=cmap, symmetric=False, title=f'{prefix} {name}', transpose=False)
        else:
            fig = plot_2d_scalar(data, extent, title=f'{prefix} {name}', z_label=zlab, cmap=cmap, symmetric=sym)
        p = os.path.join(OUT, f'{prefix}_{name}.png')
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f'REVIEW: {p}', flush=True)


def _tilt_one(q):
    qx, qy = float(q[0]), float(q[1])
    cz = 1.0 - 2.0 * (qx * qx + qy * qy)
    return float(np.degrees(np.arccos(np.clip(abs(cz), 0.0, 1.0))))


def _setup_pinned_single(fit, tip_xy, pin_idx=PIN_ATOM, z_pin_rel=Z_PIN_REL, k_spring=K_SPRING):
    """One-replica production-kernel setup with O spring-pinned at the tip."""
    xs = np.array([tip_xy[0]], dtype=np.float32)
    ys = np.array([tip_xy[1]], dtype=np.float32)
    z_tip = Z_SURF_TOP + float(z_pin_rel)
    return setup_rigid_folded_replicas(
        fit, xs, ys, z_init=z_pin_rel, mass_trans=FIRE_MASS, pin_atom_idx=pin_idx, z_pin=z_tip, k_spring=k_spring,
    )


def _record_snapshot(out):
    F = out['body_force'][0, :3]
    T = out['body_torque'][0, :3]
    E = float(out['body_force'][0, 3])
    return {
        'E': E,
        'Fmag': float(np.linalg.norm(F)),
        'Tmag': float(np.linalg.norm(T)),
        'Fz': float(F[2]),
        'com': out['pos'][0, :3].copy(),
        'quat': out['quats'][0].copy(),
        'tilt': _tilt_one(out['quats'][0]),
        'atoms': out['atom_positions'][0][:, :3].copy(),
    }


def record_newton_traj(rbd, niter=NEWTON_NITER, f_tol=F_TOL, t_tol=T_TOL):
    """Record every Newton iteration while persistent trust/lambda state remains on GPU."""
    out = rbd.run_folded_newton_replicas(niter=0, eps_t=NEWTON_EPS, eps_r=NEWTON_EPS, f_tol=f_tol, t_tol=t_tol, trust0=NEWTON_TRUST, lambda0=NEWTON_LAMBDA_DAMPED)
    snaps = [_record_snapshot(out)]
    for _ in range(int(niter)):
        out = rbd.run_folded_newton_replicas(niter=1, eps_t=NEWTON_EPS, eps_r=NEWTON_EPS, f_tol=f_tol, t_tol=t_tol, trust0=NEWTON_TRUST, lambda0=NEWTON_LAMBDA_DAMPED)
        snaps.append(_record_snapshot(out))
        if snaps[-1]['Fmag'] < f_tol and snaps[-1]['Tmag'] < t_tol:
            break
    return snaps


def record_fire_traj(rbd, n_steps=100, dt=FIRE_DT):
    """Record one-step launches while persistent FIRE dt/damping state remains on GPU."""
    rbd.run_folded_replicas(0, dt, lin_damp=0.1, ang_damp=0.1, fire=True)
    snaps = [_record_snapshot(rbd.download_outputs())]
    for _ in range(int(n_steps)):
        rbd.run_folded_replicas(1, dt, lin_damp=0.1, ang_damp=0.1, fire=True)
        snaps.append(_record_snapshot(rbd.download_outputs()))
    return snaps


def _pick_fail_pixels(conv, iters, xs, ys, n_pick=5, win=21, min_sep=20):
    """Pick non-converged pixels from densest failure neighborhoods."""
    ny, nx = len(ys), len(xs)
    fail = (conv.reshape(ny, nx) < 0.5)
    it = iters.reshape(ny, nx)
    # prefer pixels that hit the Newton budget
    hard = fail & (it >= NEWTON_NITER - 1)
    selection = 'budget failures'
    if not np.any(hard):
        if np.any(fail):
            hard = fail
            selection = 'non-converged pixels'
        else:
            hard = it >= np.percentile(it, 99.5)
            selection = 'highest-iteration converged pixels'
    ker = np.ones((win, win), dtype=np.float64)
    # density via FFT convolution (no scipy required)
    from numpy.fft import rfft2, irfft2
    F = rfft2(hard.astype(np.float64), s=(ny + win - 1, nx + win - 1))
    K = rfft2(ker, s=(ny + win - 1, nx + win - 1))
    dens = irfft2(F * K, s=(ny + win - 1, nx + win - 1))[:ny, :nx]
    dens = np.where(hard, dens, -1.0)
    picks = []
    dens_work = dens.copy()
    for _ in range(n_pick):
        iy, ix = np.unravel_index(int(np.argmax(dens_work)), dens_work.shape)
        if dens_work[iy, ix] < 0:
            break
        picks.append((int(ix), int(iy), float(xs[ix]), float(ys[iy]), float(dens[iy, ix])))
        y0, y1 = max(0, iy - min_sep), min(ny, iy + min_sep + 1)
        x0, x1 = max(0, ix - min_sep), min(nx, ix + min_sep + 1)
        dens_work[y0:y1, x0:x1] = -1.0
    return picks, dens, hard, selection


def _plot_traj(snaps, title, fname):
    steps = np.arange(len(snaps))
    E = [s['E'] for s in snaps]
    F = [s['Fmag'] for s in snaps]
    T = [s['Tmag'] for s in snaps]
    z = [s['com'][2] - Z_SURF_TOP for s in snaps]
    tilt = [s['tilt'] for s in snaps]
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(title, fontsize=12)
    axes[0].plot(steps, E, 'b-'); axes[0].set_ylabel('E [eV]'); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(steps, np.maximum(F, 1e-16), 'r-'); axes[1].set_ylabel('|F|'); axes[1].grid(True, alpha=0.3)
    axes[1].axhline(F_TOL, color='k', ls='--', lw=0.8, label='f_tol')
    axes[2].semilogy(steps, np.maximum(T, 1e-16), 'g-'); axes[2].set_ylabel('|τ|'); axes[2].grid(True, alpha=0.3)
    axes[2].axhline(T_TOL, color='k', ls='--', lw=0.8)
    axes[3].plot(steps, z, 'm-'); axes[3].set_ylabel('COM_z [Å]'); axes[3].grid(True, alpha=0.3)
    axes[4].plot(steps, tilt, 'c-'); axes[4].set_ylabel('tilt [deg]'); axes[4].set_xlabel('iteration')
    axes[4].grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {path}', flush=True)


def _plot_xz_snapshots(snaps, enames, tip_xy, title, fname, n_show=6):
    idxs = np.linspace(0, len(snaps) - 1, min(n_show, len(snaps)), dtype=int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(3.2 * len(idxs), 4), sharey=True)
    if len(idxs) == 1:
        axes = [axes]
    for ax, i in zip(axes, idxs):
        atoms = snaps[i]['atoms']
        for e, p in zip(enames, atoms):
            c = {'C': '#444', 'O': 'red', 'H': '#aaa'}.get(e, 'purple')
            ax.scatter(p[0], p[2], c=c, s=18, edgecolors='k', linewidths=0.3)
        ax.axhline(Z_SURF_TOP, color='gray', ls=':', lw=0.8)
        ax.scatter([tip_xy[0]], [Z_SURF_TOP + Z_PIN_REL], c='cyan', marker='x', s=60, zorder=10)
        ax.set_title(f'it={i}')
        ax.set_xlabel('x [Å]')
        ax.set_aspect('equal')
    axes[0].set_ylabel('z [Å]')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f'REVIEW: {path}', flush=True)


def run_fail_trajectories(fit, xs, ys, maps_n, extent):
    """Diagnose failed or highest-iteration Newton pixels with Newton/FIRE trajectories."""
    print(f'\n=== Difficult-pixel trajectories (Newton budget={NEWTON_NITER}, FIRE first 100) ===', flush=True)
    picks, dens, hard, selection = _pick_fail_pixels(maps_n['conv'], maps_n['iters'], xs, ys, n_pick=5)
    # overview: failure density + selected tips
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dens, origin='lower', extent=extent, cmap='magma', aspect='equal')
    for k, (ix, iy, x, y, d) in enumerate(picks):
        ax.plot(x, y, 'c*', ms=14)
        ax.text(x, y, f' P{k}', color='cyan', fontsize=9, va='bottom')
    ax.set_title(f'Newton {selection} density + selected tips')
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    fig.colorbar(im, ax=ax, label='selected neighbors')
    fig.tight_layout()
    p = os.path.join(OUT, 'fail_pick_map.png')
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f'REVIEW: {p}', flush=True)

    lines = [f'Newton niter budget = {NEWTON_NITER}', f'pixel selection = {selection}', f'FIRE traj steps = 100', '']
    for k, (ix, iy, x, y, d) in enumerate(picks):
        print(f'[P{k}] tip=({x:.3f},{y:.3f}) ix={ix} iy={iy} selected_density={d:.0f}', flush=True)
        lines.append(f'P{k} tip=({x:.4f},{y:.4f}) ix={ix} iy={iy} dens={d:.1f}')

        # Newton full trajectory (persistent state across one-step launches)
        rbd = _setup_pinned_single(fit, (x, y))
        snaps_n = record_newton_traj(rbd, niter=NEWTON_NITER)
        _plot_traj(snaps_n, f'Newton traj P{k} tip=({x:.2f},{y:.2f}) n={len(snaps_n)-1}', f'traj_newton_P{k}.png')
        _plot_xz_snapshots(snaps_n, fit['enames'], (x, y), f'Newton XZ P{k}', f'traj_newton_P{k}_xz.png')
        lines.append(f'  Newton steps={len(snaps_n)-1} E0={snaps_n[0]["E"]:.4f} E1={snaps_n[-1]["E"]:.4f} '
                     f'|F|={snaps_n[-1]["Fmag"]:.3e} |τ|={snaps_n[-1]["Tmag"]:.3e} tilt={snaps_n[-1]["tilt"]:.1f}')
        ip = iy * len(xs) + ix
        dpos = float(np.max(np.abs(snaps_n[-1]['com'] - maps_n['pos'][ip, :3])))
        dq = min(float(np.max(np.abs(snaps_n[-1]['quat'] - maps_n['quats'][ip]))), float(np.max(np.abs(snaps_n[-1]['quat'] + maps_n['quats'][ip]))))
        lines.append(f'  Newton one-replica vs full-map parity max|dpos|={dpos:.3e} max|dquat|={dq:.3e}')

        # FIRE first 100 from fresh IC
        rbd = _setup_pinned_single(fit, (x, y))
        snaps_f = record_fire_traj(rbd, n_steps=100)
        _plot_traj(snaps_f, f'FIRE traj P{k} tip=({x:.2f},{y:.2f}) first 100', f'traj_fire_P{k}.png')
        _plot_xz_snapshots(snaps_f, fit['enames'], (x, y), f'FIRE XZ P{k}', f'traj_fire_P{k}_xz.png')
        lines.append(f'  FIRE100 E0={snaps_f[0]["E"]:.4f} E1={snaps_f[-1]["E"]:.4f} '
                     f'|F|={snaps_f[-1]["Fmag"]:.3e} |τ|={snaps_f[-1]["Tmag"]:.3e} tilt={snaps_f[-1]["tilt"]:.1f}')

    outp = os.path.join(OUT, 'fail_traj.out')
    with open(outp, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'REVIEW: {outp}', flush=True)


def main(traj_only=False):
    print('=== PTCDA@NaCl FAF replicas (pinned O) : Newton then FIRE ===', flush=True)
    import pyopencl as cl
    plats = [(p.name, [d.name for d in p.get_devices()]) for p in cl.get_platforms()]
    print(f'[OpenCL] {plats}', flush=True)
    if not any('nvidia' in (p[0] + ' '.join(p[1])).lower() for p in plats):
        raise RuntimeError('NVIDIA OpenCL device not visible — re-run with unrestricted Shell')

    fit = _ensure_fit()
    enames = fit['enames']
    print(f'[pin] atom {PIN_ATOM} ({enames[PIN_ATOM]})  z_pin={Z_SURF_TOP+Z_PIN_REL:.2f}  k={K_SPRING}', flush=True)
    print(f'[Newton] staged {NEWTON_NITER}+{NEWTON_NITER}, eps={NEWTON_EPS:g}, trust={NEWTON_TRUST:g}, '
          f'lambda floor {NEWTON_LAMBDA_DAMPED:g}->{NEWTON_LAMBDA_FINISH:g}', flush=True)
    print(f'[FIRE] effective mass={FIRE_MASS:g}  dt={FIRE_DT:g}  budget={FIRE_NITER}', flush=True)

    L = NCELL * LATTICE_A
    xs = np.linspace(0.0, L, NXY, endpoint=False, dtype=np.float32)
    ys = np.linspace(0.0, L, NXY, endpoint=False, dtype=np.float32)
    extent = [0.0, L, 0.0, L]
    print(f'[scan] {NXY}×{NXY} tip over {L}×{L} Å (NaCl {NCELL}×{NCELL}), flat PTCDA', flush=True)

    n_rep = NXY * NXY

    # --- Newton (needed for fail mask + maps) ---
    rbd = _setup(fit, xs, ys)
    t0 = time.perf_counter()
    out_n = run_staged_newton(rbd)
    dt_n = time.perf_counter() - t0
    print(f'[Newton] wall={dt_n:.2f}s  throughput={n_rep/dt_n:.1e} mol/s', flush=True)
    maps_n = _summarize(out_n, 'Newton', xs, ys)
    if not traj_only:
        _plot_maps(maps_n, xs, ys, 'newton', extent)

    # Fail-pixel trajectories (always — this is the diagnostic ask)
    run_fail_trajectories(fit, xs, ys, maps_n, extent)

    if traj_only:
        print('Done (traj_only).', flush=True)
        return

    # --- FIRE ---
    rbd = _setup(fit, xs, ys)
    t0 = time.perf_counter()
    rbd.run_folded_replicas(FIRE_NITER, FIRE_DT, lin_damp=0.1, ang_damp=0.1, fire=True)
    out_f = rbd.download_outputs()
    dt_f = time.perf_counter() - t0
    print(f'[FIRE] wall={dt_f:.2f}s  throughput={n_rep/dt_f:.1e} mol/s  budget={FIRE_NITER}', flush=True)
    maps_f = _summarize(out_f, 'FIRE', xs, ys)
    _plot_maps(maps_f, xs, ys, 'fire', extent)

    out_path = os.path.join(OUT, 'summary.out')
    with open(out_path, 'w') as f:
        f.write(f'PTCDA@NaCl FAF {NXY}x{NXY} tip-scan over {L}x{L} A\n')
        f.write(f'pin O={PIN_ATOM} z_pin_rel={Z_PIN_REL} k={K_SPRING}\n')
        f.write(f'effective_mass={FIRE_MASS} (inertia scaled consistently)\n')
        f.write(f'Newton niter={NEWTON_NITER} frac_conv={maps_n["conv"].mean():.4f} '
                f'wall={dt_n:.4f}s throughput={n_rep/dt_n:.4e}/s '
                f'iters_med={np.median(maps_n["iters"]):.0f} '
                f'F_final_med={np.median(maps_n["Fmag"]):.4e} F_final_max={maps_n["Fmag"].max():.4e} '
                f'T_final_med={np.median(maps_n["Tmag"]):.4e} T_final_max={maps_n["Tmag"].max():.4e} '
                f'Fmax_med={np.median(maps_n["Fmax"]):.4e} Fmax_max={maps_n["Fmax"].max():.4e}\n')
        f.write(f'FIRE niter={FIRE_NITER} frac_conv={maps_f["conv"].mean():.4f} '
                f'wall={dt_f:.4f}s throughput={n_rep/dt_f:.4e}/s '
                f'iters_med={np.median(maps_f["iters"]):.0f} '
                f'F_final_med={np.median(maps_f["Fmag"]):.4e} F_final_max={maps_f["Fmag"].max():.4e} '
                f'T_final_med={np.median(maps_f["Tmag"]):.4e} T_final_max={maps_f["Tmag"].max():.4e} '
                f'Fmax_med={np.median(maps_f["Fmax"]):.4e} Fmax_max={maps_f["Fmax"].max():.4e}\n')
    print(f'REVIEW: {out_path}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    traj_only = '--traj' in sys.argv or '--traj-only' in sys.argv
    main(traj_only=traj_only)
