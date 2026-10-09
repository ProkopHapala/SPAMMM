#!/usr/bin/env python3
"""DEBUG: sequential core-then-mesh fit of a cached FDBM field (no GPU, no FDBM rerun).

1) core: (1 - r/R_k)_+^N radial modes on atoms + bond centers, cutoffs R_k geometric Rmax..Rlow,
   plain LSQ on the shell rin < r_min < rcut, probe side z > mol_z + z0.
2) mesh: weighted P-spline (fit_coremesh_lsq, no atoms) on ref-core for r_min > rin (no outer cut).
3) evaluation on the dense xz cut + deep-zone (r_min<rin) hole check + plots.
Inputs: debug/testplot_fdbm_pme_debug/<mol>_h0.5/{coremesh_samples.npz (cc) | joint/joint_samples.npz (azaindol)}.
Recreated from the lost /tmp/core_then_mesh.py (section 13 of doc/Tasks/ContactPME_CoreMesh_Fit_Design.md).
"""
import os, sys, time, argparse
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm.surfaces.CoarseMesh import fit_coremesh_lsq
from spammm.surfaces.PICCore import (POLY_CORE_R, POLY_CORE_N, POLY_ATOM_SLOTS, POLY_BOND_SLOTS,
                                    core_centers_atoms_bonds, fit_core_poly_shell, eval_core_poly)

ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'testplot_fdbm_pme_debug')
ap = argparse.ArgumentParser()
ap.add_argument('mol', choices=['cc', 'azaindol'])
ap.add_argument('--N', type=int, default=8); ap.add_argument('--nat', type=int, default=3); ap.add_argument('--nbd', type=int, default=2)
ap.add_argument('--Rmax', type=float, default=9.0); ap.add_argument('--Rlow', type=float, default=3.5)
ap.add_argument('--rin', type=float, default=2.5); ap.add_argument('--rcut', type=float, default=4.0); ap.add_argument('--z0', type=float, default=0.5)
ap.add_argument('--dbond', type=float, default=1.6); ap.add_argument('--hm', type=float, default=1.0); ap.add_argument('--lam', type=float, default=1e-3)
ap.add_argument('--fslices', action='store_true', help='plot unrelaxed Fz xy slices ref/model/diff')
args = ap.parse_args()
t0 = time.time()

# ---- load cached samples (0.25 Å grid, ij-ordered pts)
d = np.load(os.path.join(ROOT, 'cc_h0.5', 'coremesh_samples.npz') if args.mol == 'cc' else os.path.join(ROOT, 'azaindol_h0.5', 'joint', 'joint_samples.npz'))
A = d['apos'].astype(float); lo = d['lo'].astype(float); ns_s = tuple(int(n) for n in d['ns_s']); dg = float(d['h'])/int(d['ss'])
E3 = d['E_ss'].reshape(ns_s).astype(float); P3 = d['pts_ss'].reshape(*ns_s, 3).astype(float)
mol_z = A[:, 2].max()
outdir = os.path.join(ROOT, f'{args.mol}_h0.5', 'core_then_mesh'); os.makedirs(outdir, exist_ok=True)
tag = f'at{args.nat}bd{args.nbd}_N{args.N}_R{args.Rmax}-{args.Rlow}_hm{args.hm}_lam{args.lam:g}'

# ---- centers + basis (PICCore poly8 SSOT; slots atoms {0,1,2}, bonds {0,2})
assert args.N == POLY_CORE_N and args.Rmax == POLY_CORE_R[0] and args.Rlow == POLY_CORE_R[2] and args.nat == len(POLY_ATOM_SLOTS) and args.nbd == len(POLY_BOND_SLOTS), 'fixed poly8 recipe'
centers, is_bond = core_centers_atoms_bonds(A, dbond=args.dbond)
B = centers[is_bond]
print(f'{args.mol}: {len(A)} atoms, {len(B)} bond centers (d<{args.dbond}); atom modes {args.nat}, bond modes {args.nbd}, N={args.N}', flush=True)
print(f'mode cutoffs R_k: atoms {np.round(POLY_CORE_R[list(POLY_ATOM_SLOTS)], 3)}  bonds {np.round(POLY_CORE_R[list(POLY_BOND_SLOTS)], 3)}', flush=True)
def rmin_of(p): return np.min(np.linalg.norm(p[:, None, :] - A[None], axis=-1), axis=1)
def core_eval(p, coef): return eval_core_poly(np.asarray(p), centers, coef)[0]

# ---- 1) core LSQ on shell
P = P3.reshape(-1, 3); E = E3.ravel(); rm = rmin_of(P); up = P[:, 2] > mol_z + args.z0
mcore = (rm > args.rin) & (rm < args.rcut) & up
fit = fit_core_poly_shell(A, centers, is_bond, pts=P, E=E, rin=args.rin, rcut=args.rcut, z_min=mol_z + args.z0)
coef = fit.coeffs
Core3 = core_eval(P, coef).reshape(ns_s)

# ---- 2) mesh on residual (weighted, r_min>rin probe side), node spacing hm = s*dg
s = int(round(args.hm/dg)); sl = tuple(slice(0, ((n - 1)//s)*s + 1) for n in ns_s)
wm = ((rm > args.rin) & up).reshape(ns_s)[sl].astype(float)
def fit_mesh(T):
    mesh, _, diag = fit_coremesh_lsq(T[sl], np.empty((0, 3)), lo, args.hm, [], [], s=s, weights=wm, lam=args.lam)
    return mesh.coeffs
def mesh_eval(c, p): return map_coordinates(c, ((p - lo)/args.hm).T, order=3, mode='nearest', prefilter=False)
tm = time.time(); c_res = fit_mesh(E3 - Core3); print(f'[mesh:ref-core] hm={args.hm} (s={s}) nodes {c_res.shape} ({time.time()-tm:.1f}s)', flush=True)
tm = time.time(); c_spl = fit_mesh(E3); print(f'[mesh:spline-only baseline] ({time.time()-tm:.1f}s)', flush=True)

# ---- 3a) dense xz cut table
xd, zd, Ed = d['dense_x'], d['dense_z'], d['E_dense']; yc = np.median(A[:, 1])
XD, ZD = np.meshgrid(xd, zd, indexing='ij'); Q = np.stack([XD.ravel(), np.full(XD.size, yc), ZD.ravel()], 1)
rq = rmin_of(Q); Eq = Ed.ravel(); ok = rq > args.rin
Mc = core_eval(Q, coef); Mcm = Mc + mesh_eval(c_res, Q); Ms = mesh_eval(c_spl, Q)
print(f'\n[eval] dense xz cut y={yc:.2f}, {ok.sum()} points with r_min>{args.rin}\nband                        n |  core only  core+mesh spline only   (rms / max, meV)', flush=True)
for name, m in (('all r>rin', ok), ('E<0.3', ok & (Eq < 0.3)), ('r 2.5-3.0', ok & (rq < 3.0)), ('r 3.0-4.0', (rq >= 3.0) & (rq < 4.0)), ('r >4.0', rq >= 4.0)):
    print(f'{name:20s} {m.sum():8d} | ' + '  '.join(f'{1e3*np.sqrt(np.mean((X[m]-Eq[m])**2)):6.2f}/{1e3*np.abs(X[m]-Eq[m]).max():5.1f}' for X in (Mc, Mcm, Ms)), flush=True)

# ---- 3b) deep zone r_min<rin: holes = ref>0.5 but model<0.3 (PP would fall in)
Mfull = (Core3.ravel() + mesh_eval(c_res, P)).reshape(ns_s)
for zg in (0.5, 1.5, 2.0):
    for a_, b_ in ((2.0, 2.5), (1.5, 2.0)):
        s_ = (rm >= a_) & (rm < b_) & (P[:, 2] > mol_z + zg); hi = s_ & (E > 0.5)
        if not s_.any(): continue
        hole = hi & (Mfull.ravel() < 0.3)
        print(f'[hole] z>{zg} r_min [{a_},{b_}) n={s_.sum():5d} ref>0.5: {hi.sum():5d}  HOLES(ref>0.5 & model<0.3): {hole.sum():4d}  model min where ref>0.5: {Mfull.ravel()[hi].min() if hi.any() else np.nan:8.3f}', flush=True)

# ---- plots: xz slice of 3D grid INCLUDING the excluded zone (not masked), clipped to reachable range
iy = int(np.argmin(np.abs(P3[0, :, 0, 1] - yc))); xs = P3[:, iy, 0, 0]; zs = P3[0, iy, :, 2]
kz = zs > mol_z + args.z0; Rs = rm.reshape(ns_s)[:, iy, kz]
Es = E3[:, iy, kz]; Ms_ = Mfull[:, iy, kz]; Cs = Core3[:, iy, kz]; holes = (Es > 0.5) & (Ms_ < 0.3)
ext = [xs[0], xs[-1], zs[kz][0], zs[kz][-1]]
fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True); axs = axs.ravel()
for ax, F, ttl, kw in ((axs[0], Es, 'ref', dict(vmin=-0.1, vmax=1.0, cmap='viridis')), (axs[1], Ms_, 'core+mesh (model)', dict(vmin=-0.1, vmax=1.0, cmap='viridis')),
                       (axs[2], Cs, 'core only', dict(vmin=-0.1, vmax=1.0, cmap='viridis')), (axs[3], Ms_ - Es, 'model - ref  [±0.1 eV]', dict(vmin=-0.1, vmax=0.1, cmap='seismic'))):
    im = ax.imshow(F.T, origin='lower', extent=ext, aspect='equal', interpolation='nearest', **kw); fig.colorbar(im, ax=ax, shrink=0.8)
    cs = ax.contour(xs, zs[kz], Rs.T, levels=[2.0, args.rin, args.rcut], colors=['w', 'k', 'gray'], linewidths=1.0, linestyles=['--', '-', ':'])
    ax.plot(*np.array(np.meshgrid(xs, zs[kz], indexing='ij'))[:, holes], 'rx', ms=4)
    ax.plot(A[:, 0], A[:, 2], 'k.', ms=6); ax.set_title(ttl)
fig.suptitle(f'{args.mol} y={P3[0, iy, 0, 1]:.2f}  E clipped [-0.1,1.0] eV; contours r_min=2.0 (white --), {args.rin} (black), {args.rcut} (grey :); red x = HOLE (ref>0.5 & model<0.3)')
fig.tight_layout(); f1 = os.path.join(outdir, f'deepzone_xz_{tag}.png'); fig.savefig(f1, dpi=110); plt.close(fig)

# rays over atom0 and over molecule center, from z=mol_z+1 (inside the excluded zone)
zr = d['ray_z']; fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for k, ax in enumerate(axs):
    pr = d['ray_pts'][k].astype(float); rr = rmin_of(pr); Mr = core_eval(pr, coef) + mesh_eval(c_res, pr)
    ax.plot(zr, d['E_ray'][k], 'k-', lw=2, label='ref'); ax.plot(zr, Mr, 'r-', label='core+mesh'); ax.plot(zr, core_eval(pr, coef), 'b--', lw=1, label='core only')
    ax.plot(zr, mesh_eval(c_spl, pr), 'g:', label='spline only')
    for rv, ls in ((2.0, '--'), (args.rin, '-')):
        if np.any(rr < rv): ax.axvline(zr[np.argmax(rr >= rv)], color='gray', ls=ls, lw=0.8)
    ax.set_ylim(-0.1, 1.0); ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('z [Å]'); ax.set_title(f'ray {k} at xy={pr[0, :2].round(2)}  (vlines: r_min=2.0 --, {args.rin} -)'); ax.legend()
fig.tight_layout(); f2 = os.path.join(outdir, f'deepzone_rays_{tag}.png'); fig.savefig(f2, dpi=110); plt.close(fig)

# ---- Fz xy slices at AFM heights (Fz = -dE/dz by FD, consistent for ref and models)
if args.fslices:
    Sfull = mesh_eval(c_spl, P).reshape(ns_s)
    Fz_ref, Fz_cm, Fz_sp = [-np.gradient(G, dg, axis=2, edge_order=2) for G in (E3, Mfull, Sfull)]
    ix = np.where(np.abs(P3[:, 0, 0, 0]) < 6.5)[0]; jy = np.where(np.abs(P3[0, :, 0, 1]) < 6.5)[0]
    hs = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    fig, axs = plt.subplots(5, len(hs), figsize=(2.9 * len(hs), 13.5))
    for j, hh in enumerate(hs):
        iz = int(np.argmin(np.abs(P3[0, 0, :, 2] - (mol_z + hh)))); zh = P3[0, 0, iz, 2] - mol_z
        rows = [('FDBM', Fz_ref), ('core+mesh', Fz_cm), ('core+mesh - ref', Fz_cm - Fz_ref),
                ('spline only', Fz_sp), ('spline - ref', Fz_sp - Fz_ref)]
        for i, (nm, G) in enumerate(rows):
            A_ = G[np.ix_(ix, jy, [iz])][:, :, 0]; ext = [P3[ix[0], 0, 0, 0], P3[ix[-1], 0, 0, 0], P3[0, jy[0], 0, 1], P3[0, jy[-1], 0, 1]]
            if '-' in nm:
                v = max(0.01, np.nanpercentile(np.abs(A_), 99)); kw = dict(cmap='bwr', vmin=-v, vmax=v)
            else:
                v = np.nanpercentile(np.abs(A_), 99.5); kw = dict(cmap='bwr', vmin=-v, vmax=v)
            im = axs[i, j].imshow(A_.T, origin='lower', extent=ext, aspect='equal', **kw)
            axs[i, j].plot(A[:, 0], A[:, 1], 'k.', ms=1.5); fig.colorbar(im, ax=axs[i, j], shrink=0.7)
            if i == 0: axs[i, j].set_title(f'h={zh:.2f} Å')
            if j == 0: axs[i, j].set_ylabel(nm)
    fig.suptitle(f'{args.mol} unrelaxed Fz slices  (diff rows: symmetric per-panel clim)')
    f3 = os.path.join(outdir, f'fz_slices_{tag}.png'); fig.tight_layout(); fig.savefig(f3, dpi=120); plt.close(fig)
    for j, hh in enumerate(hs):
        iz = int(np.argmin(np.abs(P3[0, 0, :, 2] - (mol_z + hh)))); m = np.ix_(ix, jy, [iz])
        print(f'[Fz h={hh:.2f}] ref|{np.abs(Fz_ref[m]).mean():.3f}| eV/A  core+mesh rms_err={1e3*np.sqrt(np.mean((Fz_cm[m]-Fz_ref[m])**2)):.1f} meV/A  spline={1e3*np.sqrt(np.mean((Fz_sp[m]-Fz_ref[m])**2)):.1f} meV/A', flush=True)
    print(f'REVIEW: {os.path.abspath(f3)}', flush=True)
print(f'REVIEW: {os.path.abspath(f1)}\nREVIEW: {os.path.abspath(f2)}\ndone {time.time()-t0:.1f}s', flush=True)
