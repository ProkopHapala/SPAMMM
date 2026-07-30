#!/usr/bin/env python3
"""Visual demo: fig3 circle trimer — charging rings + xV with NDR.

Parametric regime from ppafm ``fig3_data/fig_1/params.json`` / ``pauli_scan_results``:
nsite=3, R=5.77, Qzz=0 (monopole), Esite=-0.09, W=0.05, diagonal line cut.

  PYTHONPATH=. python tests/quantum/testplot_charge_rings_trimer.py
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from spammm.quantum.PauliSolverCL import PauliSolverCL
from spammm.quantum import pauli_scan as ps

OUT = os.path.join(_REPO, 'debug', 'testplot_charge_rings_trimer')
os.makedirs(OUT, exist_ok=True)


def main():
    t0 = time.time()
    params = ps.symmetric_trimer_params(npix=100, VBias=0.85)
    spos, rots, angles = ps.make_site_geom(params)
    print(f'trimer sites (x,y,z,E):\n{spos}')
    print(f'Q0={params["Q0"]} Qzz={params["Qzz"]} W={params["W"]} Esite={params["Esite"]} '
          f'phiRot={params["phiRot"]:.4f} z_tip={params["z_tip"]} Temp={params["Temp"]}')
    print(f'xV cut: ({params["p1_x"]},{params["p1_y"]}) → ({params["p2_x"]},{params["p2_y"]})')

    solver = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=True)
    name = solver.ctx.devices[0].name.lower()
    assert any(k in name for k in ('nvidia', 'geforce', 'rtx', 'quadro')), solver.ctx.devices[0].name
    print(f'device: {solver.ctx.devices[0].name}')

    # ---- xV (0 → 0.85 V like test_1_xV reference) ----
    print('--- xV ---')
    xv = ps.scan_xV(solver, spos, rots, params, nx=120, nV=100, Vmin=0.0, Vmax=0.85)
    STM, dIdV = xv['STM'], xv['dIdV']
    ndr_frac = float(np.mean(dIdV < 0))
    ndr_min = float(dIdV.min())
    print(f'xV I: [{STM.min():.3e}, {STM.max():.3e}]  dIdV: [{dIdV.min():.3e}, {dIdV.max():.3e}]  '
          f'NDR fraction={ndr_frac:.3f} min={ndr_min:.3e}')
    assert STM.max() > 0 and np.isfinite(STM).all()
    assert ndr_min < 0, 'expected negative dI/dV (NDR) in fig3 trimer regime'

    fig, axs = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    ext = xv['extent']
    # distance along cut for x-axis like reference (0..~42 Å) — use projected x from extent corners
    im0 = axs[0].imshow(STM, origin='lower', extent=ext, aspect='auto', cmap='inferno')
    axs[0].set_ylabel('V [V]'); axs[0].set_title('symmetric trimer PME  Qzz=0  W=0.05 (STM)')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    vmax = 0.5 * np.nanmax(np.abs(dIdV))
    vmax = vmax if vmax > 0 else 1.0
    im1 = axs[1].imshow(dIdV, origin='lower', extent=ext, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
    axs[1].set_xlabel('x [Å] (diagonal cut projected)'); axs[1].set_ylabel('V [V]')
    axs[1].set_title(f'dI/dV  (NDR blue; min={ndr_min:.2e})')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    for sx in spos[:, 0]:
        for ax in axs:
            ax.axvline(sx, color='cyan', ls=':', lw=0.7, alpha=0.6)
    fig.tight_layout()
    path_xv = os.path.join(OUT, 'symmetric_trimer_xV_NDR.png')
    fig.savefig(path_xv, dpi=140); plt.close(fig)
    print(f'REVIEW: {path_xv}')

    # ---- xy at V_slice and a few biases ----
    print('--- xy ---')
    Vs = [0.50, 0.69, 0.79, 0.90]
    maps = []
    for V in Vs:
        p = dict(params); p['VBias'] = V
        m = ps.scan_xy(solver, spos, rots, p)
        print(f'  V={V:.2f} Imax={m["STM"].max():.3e} dIdV∈[{m["dIdV"].min():.3e},{m["dIdV"].max():.3e}]')
        maps.append(m)

    nV = len(Vs)
    fig, axs = plt.subplots(2, nV, figsize=(2.4 * nV, 5.0))
    for j, (V, m) in enumerate(zip(Vs, maps)):
        axs[0, j].imshow(m['STM'], origin='lower', extent=m['extent'], cmap='inferno')
        axs[0, j].plot(spos[:, 0], spos[:, 1], 'c+', ms=8, mew=1.2)
        axs[0, j].plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'w-', lw=1.0)
        axs[0, j].set_title(f'V={V:.2f}')
        if j == 0: axs[0, j].set_ylabel('STM')
        sc = max(np.nanmax(np.abs(m['dIdV'])), 1e-30)
        axs[1, j].imshow(m['dIdV'], origin='lower', extent=m['extent'], cmap='bwr', vmin=-sc, vmax=sc)
        axs[1, j].plot(spos[:, 0], spos[:, 1], 'k+', ms=8, mew=1.2)
        axs[1, j].plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'k-', lw=1.0)
        if j == 0: axs[1, j].set_ylabel('dI/dV rings')
        axs[1, j].set_xlabel('x [Å]')
    fig.suptitle('symmetric trimer xy  Qzz=0 W=0.05 (PME.cl)', fontsize=12)
    fig.tight_layout()
    path_xy = os.path.join(OUT, 'symmetric_trimer_xy_Vstack.png')
    fig.savefig(path_xy, dpi=130); plt.close(fig)
    print(f'REVIEW: {path_xy}')

    # close-up at V=0.69
    mid = maps[Vs.index(0.69)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].imshow(mid['STM'], origin='lower', extent=mid['extent'], cmap='inferno')
    axs[0].plot(spos[:, 0], spos[:, 1], 'c+', ms=10, mew=1.5)
    axs[0].plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'w-', lw=1.5, label='xV cut')
    axs[0].legend(fontsize=8)
    axs[0].set_title('STM V=0.69'); axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
    sc = np.nanmax(np.abs(mid['dIdV']))
    axs[1].imshow(mid['dIdV'], origin='lower', extent=mid['extent'], cmap='bwr', vmin=-sc, vmax=sc)
    axs[1].plot(spos[:, 0], spos[:, 1], 'k+', ms=10, mew=1.5)
    axs[1].plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'k-', lw=1.5)
    axs[1].set_title('dI/dV charging rings (+NDR blue)')
    axs[1].set_xlabel('x')
    fig.tight_layout()
    path_m = os.path.join(OUT, 'symmetric_trimer_xy_V0.69.png')
    fig.savefig(path_m, dpi=140); plt.close(fig)
    print(f'REVIEW: {path_m}')
    print(f'DONE in {time.time()-t0:.1f}s → {OUT}')


if __name__ == '__main__':
    main()
