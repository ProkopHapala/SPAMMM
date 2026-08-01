#!/usr/bin/env python3
"""L2: Square tetramer PME — xV scan + xy map (NDR regime). Single SSOT: params dict."""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    from spammm.quantum import pauli_scan as ps
    from spammm.quantum.PauliSolverCL import PauliSolverCL

    # SSOT: all values come from this params dict — no hardcoded numbers in plots
    params = ps.ruslan_default_params(
        geometry_file='data/charge_rings/square_tetramer.txt',
        nsite=4, Qzz=0.0, W=0.05,
        VBias=1.2, Temp=2.6, z_tip=6.0, zV0=-0.9, zVd=20.0,
        npix=100,
        p1_x=-15.0, p1_y=0.0, p2_x=15.0, p2_y=0.0,
    )
    Vmin, Vmax = 0.5, 1.5  # xV scan range
    VBias = float(params['VBias'])  # xy slice voltage + xV horizontal line
    p1 = (float(params['p1_x']), float(params['p1_y']))
    p2 = (float(params['p2_x']), float(params['p2_y']))

    spos, rots, _ = ps.make_site_geom(params)
    print(f"[tetramer] sites:\n{spos[:,:3]}", flush=True)
    sol = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=False)

    outdir = 'debug/test_pme_tetramer'
    os.makedirs(outdir, exist_ok=True)

    # --- xV scan ---
    print(f"[tetramer] xV scan Vmin={Vmin} Vmax={Vmax} ...", flush=True)
    xv = ps.scan_xV(sol, spos, rots, params, nx=100, nV=80, Vmin=Vmin, Vmax=Vmax, return_probs=True)
    STM_xv, dIdV_xv = xv['STM'], xv['dIdV']
    V = xv['Vbiases']
    x = xv['dist_axis']
    ndr_min = float(dIdV_xv.min())
    print(f"[tetramer] xV STM range=[{STM_xv.min():.3e},{STM_xv.max():.3e}] dIdV min={ndr_min:.3e} NDR={'YES' if ndr_min<-1e-10 else 'no'}", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axs[0].imshow(STM_xv, aspect='auto', origin='lower', extent=[x[0], x[-1], V[0], V[-1]], cmap='inferno')
    axs[0].axhline(VBias, color='cyan', lw=1.5, ls='--', label=f'XY @ V={VBias:.2f}')
    axs[0].legend(loc='upper right', fontsize=8)
    axs[0].set_title(f'xV STM  square_tetramer  Qzz=0 W=0.05 Esite={params["Esite"]:.2f}')
    axs[0].set_xlabel('distance along cut [Å]'); axs[0].set_ylabel('V [V]')
    fig.colorbar(im0, ax=axs[0], fraction=0.046)
    sc = max(np.abs(dIdV_xv).max(), 1e-30)
    im1 = axs[1].imshow(dIdV_xv, aspect='auto', origin='lower', extent=[x[0], x[-1], V[0], V[-1]], cmap='bwr', vmin=-sc, vmax=sc)
    axs[1].axhline(VBias, color='cyan', lw=1.5, ls='--')
    axs[1].set_title(f'dI/dV  NDR={ndr_min<0}  min={ndr_min:.2e}')
    axs[1].set_xlabel('distance along cut [Å]'); axs[1].set_ylabel('V [V]')
    fig.colorbar(im1, ax=axs[1], fraction=0.046)
    fig.tight_layout()
    path_xv = os.path.join(outdir, f'tetramer_xV_{Vmin}_{Vmax}.png')
    fig.savefig(path_xv, dpi=100); plt.close(fig)
    print(f"REVIEW: {path_xv}", flush=True)

    # --- xy map at VBias ---
    print(f"[tetramer] xy scan at V={VBias} ...", flush=True)
    xy = ps.scan_xy(sol, spos, rots, params, return_probs=False)
    STM_xy, dIdV_xy = xy['STM'], xy['dIdV']
    print(f"[tetramer] xy STM range=[{STM_xy.min():.3e},{STM_xy.max():.3e}]", flush=True)

    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs2[0].imshow(STM_xy, origin='lower', extent=xy['extent'], cmap='inferno')
    axs2[0].plot(spos[:,0], spos[:,1], 'c+', ms=10, mew=1.5)
    axs2[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'w-', lw=1.5, label='xV cut')
    axs2[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'wo', ms=4)
    axs2[0].legend(loc='upper right', fontsize=8)
    axs2[0].set_title(f'xy STM  V={VBias:.2f}  square_tetramer')
    axs2[0].set_xlabel('x [Å]'); axs2[0].set_ylabel('y [Å]')
    fig2.colorbar(im0, ax=axs2[0], fraction=0.046)
    if dIdV_xy is not None:
        sc2 = max(np.abs(dIdV_xy).max(), 1e-30)
        im1 = axs2[1].imshow(dIdV_xy, origin='lower', extent=xy['extent'], cmap='bwr', vmin=-sc2, vmax=sc2)
        axs2[1].plot(spos[:,0], spos[:,1], 'k+', ms=10, mew=1.5)
        axs2[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'w-', lw=1.5)
        axs2[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'wo', ms=4)
        axs2[1].set_title(f'dI/dV (rings / NDR)  V={VBias:.2f}')
        axs2[1].set_xlabel('x [Å]'); axs2[1].set_ylabel('y [Å]')
        fig2.colorbar(im1, ax=axs2[1], fraction=0.046)
    fig2.tight_layout()
    path_xy = os.path.join(outdir, f'tetramer_xy_V{VBias:.2f}.png')
    fig2.savefig(path_xy, dpi=100); plt.close(fig2)
    print(f"REVIEW: {path_xy}", flush=True)

    print("[tetramer] done", flush=True)

if __name__ == '__main__':
    main()
