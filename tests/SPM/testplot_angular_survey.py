#!/usr/bin/env python3
"""DEBUG: basis survey — atom-only radial vs SP-angular vs atom+bond centers.

Fits CORE ONLY (no mesh) on the cached azaindol FDBM field, shell 2.5<r_min<4.0,
z>mol_z+0.5. Reports shell rms and plots the residual (ref - core): xz cut +
Fz xy slices at AFM heights — the field a mesh would have to absorb.

Variants (slots = coeffs/center):
  at3bd2    31 centers: atoms s x R{9,5.6,3.5}, bonds s x R{9,3.5}        -> 77 c
  at3       atoms only s x {9,5.6,3.5}                                   -> 45 c
  atSP2     atoms s,p x R{9,3.5}      (8/atom)                           -> 120 c
  atSP3     atoms s,p x R{9,5.6,3.5}  (12/atom)                          -> 180 c
  atSP4     atoms s,p x R{9,5.6,3.5,2.4} (16/atom)                       -> 240 c
  at5bd3    atoms s x 5 slots {9,5.6,3.5,2.8,2.3}, bonds s x {9,3.5,2.4} -> 123 c
"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm.surfaces.PICCore import core_centers_atoms_bonds, eval_core_poly

d = np.load(os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'testplot_fdbm_pme_debug', 'azaindol_h0.5', 'joint', 'joint_samples.npz'))
A = d['apos'].astype(float); lo = d['lo'].astype(float); ns_s = tuple(int(n) for n in d['ns_s']); dg = float(d['h'])/int(d['ss'])
E3 = d['E_ss'].reshape(ns_s).astype(float); P3 = d['pts_ss'].reshape(*ns_s, 3).astype(float)
P = P3.reshape(-1, 3); E = E3.ravel(); mol_z = A[:, 2].max()
rm = np.min(np.linalg.norm(P[:, None, :] - A[None], axis=-1), axis=1)
mfit = (rm > 2.5) & (rm < 4.0) & (P[:, 2] > mol_z + 0.5)
print(f'azaindol: {len(A)} atoms, {mfit.sum()} fit samples in shell', flush=True)

R5 = np.array([9.0, 5.612486080160911, 3.5, 2.8, 2.3])
N = 8
def phiN(r, R): u = np.clip(1 - r[..., None]/np.asarray(R), 0, None); return u**N   # (np, nc, nm)

def design_s(p, C, R):                                  # s-only: (np, nc*nm)
    r = np.linalg.norm(p[:, None, :] - C[None], axis=-1)
    return phiN(r, R).reshape(len(p), -1)

def design_sp(p, C, R):                                 # s,px,py,pz x radial: (np, nc*nm*4)
    v = p[:, None, :] - C[None]; r = np.linalg.norm(v, axis=-1)
    u = np.zeros_like(v); ok_ = r > 1e-9; u[ok_] = v[ok_] / r[ok_, None]   # unit dir
    ph = phiN(r, R)                                     # (np, nc, nm)
    return np.concatenate([ph] + [ph * u[..., a, None] for a in range(3)], axis=-1).reshape(len(p), -1)

C_at, is_b = core_centers_atoms_bonds(A, dbond=1.6); B = C_at[is_b]
variants = {}
variants['at3bd2'] = lambda p: np.hstack([design_s(p, A, R5[:3]), design_s(p, B, R5[[0, 2]])])
variants['at3']    = lambda p: design_s(p, A, R5[:3])
variants['atSP2']  = lambda p: design_sp(p, A, R5[[0, 2]])
variants['atSP3']  = lambda p: design_sp(p, A, R5[:3])
variants['atSP4']  = lambda p: design_sp(p, A, R5[:4])
variants['at5bd3'] = lambda p: np.hstack([design_s(p, A, R5[:5]), design_s(p, B, R5[[0, 2, 3]])])

out = {}
for name, mk in variants.items():
    t0 = time.time(); M = mk(P[mfit]); c, *_ = np.linalg.lstsq(M, E[mfit], rcond=1e-12)
    ec = M @ c - E[mfit]
    Ev = np.concatenate([mk(P[i:i+20000]) @ c for i in range(0, len(P), 20000)]).reshape(ns_s)
    out[name] = Ev
    print(f'{name:8s} ncoef={M.shape[1]:4d} cond={np.linalg.cond(M):.2e} shell rms={1e3*np.sqrt(np.mean(ec**2)):6.2f} meV max={1e3*np.abs(ec).max():6.1f} meV ({time.time()-t0:.1f}s)', flush=True)

# ---- plots: residual xz (y=mol plane) + residual Fz slices at h=2.5,3.0,3.5
yc = np.median(A[:, 1]); iy = int(np.argmin(np.abs(P3[0, :, 0, 1] - yc)))
xs = P3[:, iy, 0, 0]; zs = P3[0, iy, :, 2]; kz = zs > mol_z + 0.5
ix = np.where(np.abs(P3[:, 0, 0, 0]) < 6.5)[0]; jy = np.where(np.abs(P3[0, :, 0, 1]) < 6.5)[0]
hs = [2.5, 3.0, 3.5]; izz = [int(np.argmin(np.abs(zs - (mol_z + h_)))) for h_ in hs]
nv = len(out); fig, axs = plt.subplots(nv, 1 + len(hs), figsize=(3.1 * (1 + len(hs)), 2.9 * nv), squeeze=False)
for i, (name, Ev) in enumerate(out.items()):
    R = E3 - Ev
    ax = axs[i, 0]; Rxz = R[:, iy, kz]; v = np.nanpercentile(np.abs(Rxz), 99.5)
    ax.imshow(Rxz.T, origin='lower', extent=[xs[0], xs[-1], zs[kz][0], zs[kz][-1]], aspect='equal', cmap='seismic', vmin=-v, vmax=v)
    ax.plot(A[:, 0], A[:, 2], 'k.', ms=3); ax.set_ylabel(f'{name}\nresid E xz (±{v:.3f})', fontsize=8)
    Fz = -np.gradient(R, dg, axis=2, edge_order=2)
    for j, (h_, iz) in enumerate(zip(hs, izz)):
        S = Fz[np.ix_(ix, jy, [iz])][:, :, 0]; v = max(0.005, np.nanpercentile(np.abs(S), 99.5))
        ax = axs[i, 1 + j]; ax.imshow(S.T, origin='lower', extent=[xs[ix[0]], xs[ix[-1]], P3[0, jy[0], 0, 1], P3[0, jy[-1], 0, 1]], aspect='equal', cmap='bwr', vmin=-v, vmax=v)
        ax.plot(A[:, 0], A[:, 1], 'k.', ms=2); ax.set_title(f'resid Fz h={zs[iz]-mol_z:.2f} ±{v*1e3:.0f}m', fontsize=8)
        rms = 1e3*np.sqrt(np.mean(S**2))
        ax.set_xlabel(f'rms {rms:.1f} meV/A', fontsize=8)
fig.suptitle('azaindol residual (FDBM − core-only); xz = E [eV], slices = Fz [eV/Å]')
f = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'testplot_fdbm_pme_debug', 'azaindol_h0.5', 'angular_survey.png')
fig.tight_layout(); fig.savefig(f, dpi=120); plt.close(fig)
print('REVIEW:', os.path.abspath(f), flush=True)
