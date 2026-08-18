#!/usr/bin/env python3
"""Reproduce the fig3 symmetric trimer NDR with PME8 (8-site solver).

Runs the same trimer geometry/params as testplot_charge_rings_trimer.py
but with PauliSolverCL8 (3 active + 5 spectators) and compares NDR.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spammm.quantum import pauli_scan as ps
from spammm.quantum.PauliSolverCL import PauliSolverCL
from spammm.quantum.PauliSolverCL8 import PauliSolverCL8

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_trimer_pme8')
os.makedirs(OUT, exist_ok=True)

# ── Trimer params (same as testplot_charge_rings_trimer.py) ──────────────────
params = ps.symmetric_trimer_params(npix=100, VBias=0.85)
spos3, rots3, angles = ps.make_site_geom(params)
print(f"Trimer sites:\n{spos3}", flush=True)
print(f"Q0={params['Q0']} Qzz={params['Qzz']} W={params['W']} Esite={params['Esite']} "
      f"Temp={params['Temp']} z_tip={params['z_tip']}", flush=True)

# ── Build 8-site embed: 3 active + 5 spectators ──────────────────────────────
n_active = 3
spos8 = np.zeros((8, 4), dtype=np.float64)
spos8[:n_active] = spos3[:n_active]  # copy x,y,z,E
# 5 spectators far away, high energy
for i in range(n_active, 8):
    spos8[i] = [1e3, 1e3, 0.0, 1e3]
rots8 = np.tile(np.eye(3, dtype=np.float32), (8, 1, 1))
rots8[:n_active] = rots3[:n_active]

# Wij: only among 3 active sites
Wij8 = np.zeros((8, 8), dtype=np.float32)
W = float(params['W'])
for i in range(n_active):
    for j in range(i+1, n_active):
        Wij8[i, j] = Wij8[j, i] = W

# ── Run PME4 (reference) ─────────────────────────────────────────────────────
print("\n=== PME4 (reference) ===", flush=True)
sol4 = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=False)
xv4 = ps.scan_xV(sol4, spos3, rots3, params, nx=120, nV=100, Vmin=0.0, Vmax=0.85)
STM4, dIdV4 = xv4['STM'], xv4['dIdV']
ndr4_min = float(dIdV4.min())
ndr4_frac = float(np.mean(dIdV4 < 0))
print(f"xV I: [{STM4.min():.3e}, {STM4.max():.3e}]  dIdV: [{dIdV4.min():.3e}, {dIdV4.max():.3e}]  "
      f"NDR frac={ndr4_frac:.3f} min={ndr4_min:.3e}", flush=True)

# ── Run PME8 ─────────────────────────────────────────────────────────────────
print("\n=== PME8 ===", flush=True)
sol8 = PauliSolverCL8(nSingle=8, preferred_vendor='nvidia', bPrint=False, max_iter=5000, tol=1e-7)

def run_pme8_xV(solver, spos, rots, params, Wij, nx=120, nV=100, Vmin=0.0, Vmax=0.85):
    zT = float(params['z_tip']) + float(params['Rtip'])
    start = (float(params['p1_x']), float(params['p1_y']))
    end   = (float(params['p2_x']), float(params['p2_y']))
    pTips, ts, dist = ps.make_pTips_line(start, end, npts=nx, zT=zT)
    Vbiases = np.linspace(Vmin, Vmax, nV, dtype=np.float64)
    pTips_rep = np.tile(pTips, (nV, 1))
    Vtips_rep = np.repeat(Vbiases, nx).astype(np.float32)
    cpp = ps.make_cpp_params(params); cpp = cpp.copy(); cpp[6] = 0.0
    cs, order = ps.make_quadrupole_Coeffs(params['Q0'], params['Qzz'])
    ps.configure_leads(solver, params)
    I, *_ = solver.scan_current_tip(
        pTips=pTips_rep, Vtips=Vtips_rep, pSites=spos, params=cpp, order=order, cs=cs,
        rots=rots, Wij=Wij, return_probs=False, return_state_energies=False)
    STM = np.nan_to_num(I.reshape(nV, nx), nan=0.0, posinf=0.0, neginf=0.0)
    dIdV = np.gradient(STM, Vbiases, axis=0)
    return dict(STM=STM, dIdV=dIdV, Vbiases=Vbiases, extent=[0, dist, Vmin, Vmax], dist=dist)

xv8 = run_pme8_xV(sol8, spos8, rots8, params, Wij8, nx=120, nV=100, Vmin=0.0, Vmax=0.85)
STM8, dIdV8 = xv8['STM'], xv8['dIdV']
ndr8_min = float(dIdV8.min())
ndr8_frac = float(np.mean(dIdV8 < 0))
print(f"xV I: [{STM8.min():.3e}, {STM8.max():.3e}]  dIdV: [{dIdV8.min():.3e}, {dIdV8.max():.3e}]  "
      f"NDR frac={ndr8_frac:.3f} min={ndr8_min:.3e}", flush=True)

# ── Parity ───────────────────────────────────────────────────────────────────
diff = np.abs(STM4 - STM8)
print(f"\nParity: max|dSTM|={diff.max():.4e}, max|ddIdV|={np.abs(dIdV4-dIdV8).max():.4e}", flush=True)
print(f"NDR: PME4 min={ndr4_min:.3e}, PME8 min={ndr8_min:.3e}", flush=True)

# ── Plot comparison ──────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex='col')
ext = xv4['extent']
# PME4 STM
im00 = axs[0,0].imshow(STM4, origin='lower', extent=ext, aspect='auto', cmap='inferno')
axs[0,0].set_ylabel('V [V]'); axs[0,0].set_title('PME4 STM (reference)')
fig.colorbar(im00, ax=axs[0,0], fraction=0.046)
# PME8 STM
im01 = axs[0,1].imshow(STM8, origin='lower', extent=ext, aspect='auto', cmap='inferno')
axs[0,1].set_title('PME8 STM')
fig.colorbar(im01, ax=axs[0,1], fraction=0.046)
# PME4 dIdV
vmax4 = 0.5 * max(np.abs(dIdV4).max(), 1e-30)
im10 = axs[1,0].imshow(dIdV4, origin='lower', extent=ext, aspect='auto', cmap='bwr', vmin=-vmax4, vmax=vmax4)
axs[1,0].set_xlabel('dist [Å]'); axs[1,0].set_ylabel('V [V]')
axs[1,0].set_title(f'PME4 dIdV (NDR min={ndr4_min:.2e})')
fig.colorbar(im10, ax=axs[1,0], fraction=0.046)
# PME8 dIdV
vmax8 = 0.5 * max(np.abs(dIdV8).max(), 1e-30)
im11 = axs[1,1].imshow(dIdV8, origin='lower', extent=ext, aspect='auto', cmap='bwr', vmin=-vmax8, vmax=vmax8)
axs[1,1].set_xlabel('dist [Å]')
axs[1,1].set_title(f'PME8 dIdV (NDR min={ndr8_min:.2e})')
fig.colorbar(im11, ax=axs[1,1], fraction=0.046)
# site positions along cut
start = (float(params['p1_x']), float(params['p1_y']))
end   = (float(params['p2_x']), float(params['p2_y']))
cut_dir = np.array(end) - np.array(start); cut_dir /= np.linalg.norm(cut_dir)
site_d = np.dot(spos3[:n_active, :2] - np.array(start), cut_dir)
for d_s in site_d:
    if 0 <= d_s <= xv4['dist']:
        for ax in axs[:,0]: ax.axvline(d_s, color='cyan', ls=':', lw=0.7, alpha=0.6)
        for ax in axs[:,1]: ax.axvline(d_s, color='cyan', ls=':', lw=0.7, alpha=0.6)
fig.suptitle('Trimer NDR: PME4 vs PME8  (Qzz=0, W=0.05, Esite=-0.09, T=2.6K)', fontsize=12)
fig.tight_layout()
path = os.path.join(OUT, 'trimer_pme4_vs_pme8.png')
fig.savefig(path, dpi=140); plt.close()
print(f"\nREVIEW: {path}", flush=True)

# ── I(V) at site 0 position ──────────────────────────────────────────────────
# Find pixel closest to site 0 along cut
idx_s0 = np.argmin(np.abs(np.linspace(0, xv4['dist'], 120) - site_d[0]))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xv4['Vbiases'] if 'Vbiases' in xv4 else np.linspace(0, 0.85, 100), STM4[:, idx_s0], 'r-', label='PME4 I(V)')
ax.plot(xv8['Vbiases'], STM8[:, idx_s0], 'b--', label='PME8 I(V)')
ax.set_xlabel('V [V]'); ax.set_ylabel('I'); ax.set_title(f'I(V) at site 0 (dist={site_d[0]:.1f}Å)')
ax.legend()
fig.tight_layout()
path_iv = os.path.join(OUT, 'trimer_IV_site0.png')
fig.savefig(path_iv, dpi=140); plt.close()
print(f"REVIEW: {path_iv}", flush=True)

# ── XY top-view comparison at several biases ─────────────────────────────────
def run_pme8_xy(solver, spos, rots, params, Wij, npix=None, L=None, Vb=None):
    npix = int(npix or params['npix']); L = float(L or params['L'])
    zT = float(params['z_tip']) + float(params['Rtip'])
    pTips, Xs, Ys = ps.makePosXY(n=npix, L=L, p0=(0.0, 0.0, zT))
    Vtips = np.full(len(pTips), float(Vb or params['VBias']), dtype=np.float32)
    cpp = ps.make_cpp_params(params); cpp = cpp.copy(); cpp[6] = 0.0
    cs, order = ps.make_quadrupole_Coeffs(params['Q0'], params['Qzz'])
    ps.configure_leads(solver, params)
    I, *_ = solver.scan_current_tip(
        pTips=pTips, Vtips=Vtips, pSites=spos, params=cpp, order=order, cs=cs,
        rots=rots, Wij=Wij, return_probs=False, return_state_energies=False)
    STM = np.nan_to_num(I.reshape(npix, npix), nan=0.0, posinf=0.0, neginf=0.0)
    # dIdV via finite difference
    dQ = float(params.get('dQ', 0.02))
    Vtips2 = np.full(len(pTips), float(Vb or params['VBias']) + dQ, dtype=np.float32)
    I2, *_ = solver.scan_current_tip(
        pTips=pTips, Vtips=Vtips2, pSites=spos, params=cpp, order=order, cs=cs,
        rots=rots, Wij=Wij, return_probs=False, return_state_energies=False)
    dIdV = (np.nan_to_num(I2.reshape(npix,npix)) - STM) / dQ
    return STM, dIdV

Vs_xy = [0.50, 0.69, 0.79, 0.85]
print(f"\n=== XY top-view comparison at V={Vs_xy} ===", flush=True)
extent_xy = [-float(params['L']), float(params['L'])]*2

fig, axs = plt.subplots(2, len(Vs_xy)*2, figsize=(2.0*len(Vs_xy)*2, 5.0))
for j, Vb in enumerate(Vs_xy):
    # PME4
    p4 = dict(params); p4['VBias'] = Vb
    m4 = ps.scan_xy(sol4, spos3, rots3, p4)
    STM4xy, dIdV4xy = m4['STM'], m4['dIdV']
    # PME8
    STM8xy, dIdV8xy = run_pme8_xy(sol8, spos8, rots8, params, Wij8, Vb=Vb)
    print(f"  V={Vb:.2f}  PME4 dIdV=[{dIdV4xy.min():.2e},{dIdV4xy.max():.2e}]  "
          f"PME8 dIdV=[{dIdV8xy.min():.2e},{dIdV8xy.max():.2e}]", flush=True)
    # STM row
    ax0 = axs[0, j*2]
    ax0.imshow(STM4xy, origin='lower', extent=extent_xy, cmap='inferno')
    ax0.plot(spos3[:n_active,0], spos3[:n_active,1], 'c+', ms=6, mew=1)
    ax0.set_title(f'PME4 V={Vb:.2f}', fontsize=9)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax1 = axs[0, j*2+1]
    ax1.imshow(STM8xy, origin='lower', extent=extent_xy, cmap='inferno')
    ax1.plot(spos3[:n_active,0], spos3[:n_active,1], 'c+', ms=6, mew=1)
    ax1.set_title(f'PME8 V={Vb:.2f}', fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])
    # dIdV row (shared scale per column pair)
    sc = max(np.abs(dIdV4xy).max(), np.abs(dIdV8xy).max(), 1e-30)
    ax2 = axs[1, j*2]
    ax2.imshow(dIdV4xy, origin='lower', extent=extent_xy, cmap='bwr', vmin=-sc, vmax=sc)
    ax2.plot(spos3[:n_active,0], spos3[:n_active,1], 'k+', ms=6, mew=1)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax3 = axs[1, j*2+1]
    ax3.imshow(dIdV8xy, origin='lower', extent=extent_xy, cmap='bwr', vmin=-sc, vmax=sc)
    ax3.plot(spos3[:n_active,0], spos3[:n_active,1], 'k+', ms=6, mew=1)
    ax3.set_xticks([]); ax3.set_yticks([])
axs[0,0].set_ylabel('STM', fontsize=10)
axs[1,0].set_ylabel('dIdV', fontsize=10)
fig.suptitle('Trimer XY: PME4 vs PME8  (Qzz=0, W=0.05, Esite=-0.09, T=2.6K)', fontsize=11)
fig.tight_layout()
path_xy = os.path.join(OUT, 'trimer_xy_pme4_vs_pme8.png')
fig.savefig(path_xy, dpi=140); plt.close()
print(f"REVIEW: {path_xy}", flush=True)
