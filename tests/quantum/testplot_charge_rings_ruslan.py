#!/usr/bin/env python3
"""Visual demo: Ruslan 2-site charging rings — xV diamonds + xy maps vs bias.

Reproduces the ppafm NTCDA Ruslan_long situation (E≈−0.1 eV, W=0.05 eV)
with OpenCL PauliSolverCL / PME.cl.

  PYTHONPATH=. python tests/quantum/testplot_charge_rings_ruslan.py
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

OUT = os.path.join(_REPO, 'debug', 'testplot_charge_rings_ruslan')
os.makedirs(OUT, exist_ok=True)


def _assert_nvidia(solver):
    name = solver.ctx.devices[0].name
    low = name.lower()
    assert any(k in low for k in ('nvidia', 'geforce', 'rtx', 'quadro')), f'Expected NVIDIA, got {name}'
    print(f'device: {name}')


def plot_xV(res, title, path, sdIdV=0.5):
    STM, dIdV = res['STM'], res['dIdV']
    extent = res['extent']
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    im0 = axs[0].imshow(STM, origin='lower', extent=extent, aspect='auto', cmap='inferno')
    axs[0].set_ylabel('V [V]')
    axs[0].set_title(f'{title} (STM)')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    vmax = sdIdV * np.nanmax(np.abs(dIdV)) if np.nanmax(np.abs(dIdV)) > 0 else 1.0
    im1 = axs[1].imshow(dIdV, origin='lower', extent=extent, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
    axs[1].set_xlabel('x [Å]')
    axs[1].set_ylabel('V [V]')
    axs[1].set_title('dI/dV')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    # mark site x positions
    for sx in res['spos'][:, 0]:
        for ax in axs:
            ax.axvline(sx, color='cyan', ls=':', lw=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {path}')


def plot_xy_stack(maps, Vs, extent, spos, path, W):
    nV = len(Vs)
    fig, axs = plt.subplots(2, nV, figsize=(2.2 * nV, 4.6))
    if nV == 1:
        axs = np.array([[axs[0]], [axs[1]]])
    for j, (V, m) in enumerate(zip(Vs, maps)):
        STM, dIdV = m['STM'], m['dIdV']
        im0 = axs[0, j].imshow(STM, origin='lower', extent=extent, cmap='inferno')
        axs[0, j].set_title(f'V={V:.2f}')
        axs[0, j].plot(spos[:, 0], spos[:, 1], 'c+', ms=8, mew=1.2)
        if j == 0:
            axs[0, j].set_ylabel('STM  y [Å]')
        sc = np.nanmax(np.abs(dIdV)) if dIdV is not None else 1.0
        sc = sc if sc > 0 else 1.0
        axs[1, j].imshow(dIdV, origin='lower', extent=extent, cmap='bwr', vmin=-sc, vmax=sc)
        axs[1, j].plot(spos[:, 0], spos[:, 1], 'k+', ms=8, mew=1.2)
        if j == 0:
            axs[1, j].set_ylabel('dI/dV  y [Å]')
        axs[1, j].set_xlabel('x [Å]')
    fig.suptitle(f'Ruslan_long xy vs bias  W={W:.3f} eV  (PME.cl)', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f'REVIEW: {path}')


def main():
    t0 = time.time()
    geom = ps.default_geometry_path('Ruslan_long.txt')
    spos, rots, angles = ps.load_site_geometry(geom)
    print(f'geometry: {geom}')
    print(f'sites (x,y,z,E):\n{spos}')

    # Match ppafm Bakup_NTCDA / NTCDA Ruslan_long solver_0 W=0.05
    params = ps.ruslan_default_params(
        W=0.05,
        Esite=-0.09,
        VBias=2.0,
        npix=100,          # slightly coarser than 200 for demo speed
        L=20.0,
        p1_x=-15.0,
        p2_x=15.0,
        dQ=0.02,
    )
    print(f'params: Esite={params["Esite"]} W={params["W"]} Temp={params["Temp"]}K '
          f'GammaT={params["GammaT"]} Q0={params["Q0"]} Qzz={params["Qzz"]}')

    solver = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=True)
    _assert_nvidia(solver)

    # ---- xV charging diamonds ----
    print('--- xV scan ---')
    xv = ps.scan_xV(solver, spos, rots, params, nx=100, nV=100, Vmin=0.0, Vmax=2.0)
    print(f'xV STM: min={xv["STM"].min():.3e} max={xv["STM"].max():.3e}')
    print(f'xV dIdV: min={xv["dIdV"].min():.3e} max={xv["dIdV"].max():.3e}')
    assert np.isfinite(xv['STM']).all() and xv['STM'].max() > 0
    plot_xV(xv, f'Ruslan_long PME  W={params["W"]:.3f} eV', os.path.join(OUT, 'Ruslan_long_xV_W0.050.png'))

    # ---- xy maps at increasing bias (charging rings) ----
    print('--- xy stack ---')
    Vs = [0.50, 0.70, 0.90, 1.00, 1.10, 1.30, 1.50]
    maps = []
    for V in Vs:
        p = dict(params)
        p['VBias'] = V
        m = ps.scan_xy(solver, spos, rots, p)
        print(f'  V={V:.2f}  Imax={m["STM"].max():.3e}  dIdVmax={np.nanmax(np.abs(m["dIdV"])):.3e}')
        maps.append(m)
    plot_xy_stack(maps, Vs, maps[0]['extent'], spos, os.path.join(OUT, 'Ruslan_long_xy_Vstack_W0.050.png'), params['W'])

    # Also save a single mid-bias pair for closer look
    mid = maps[Vs.index(1.00)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].imshow(mid['STM'], origin='lower', extent=mid['extent'], cmap='inferno')
    axs[0].plot(spos[:, 0], spos[:, 1], 'c+', ms=10, mew=1.5)
    axs[0].set_title('STM V=1.00 W=0.050')
    axs[0].set_xlabel('x [Å]'); axs[0].set_ylabel('y [Å]')
    sc = np.nanmax(np.abs(mid['dIdV']))
    axs[1].imshow(mid['dIdV'], origin='lower', extent=mid['extent'], cmap='bwr', vmin=-sc, vmax=sc)
    axs[1].plot(spos[:, 0], spos[:, 1], 'k+', ms=10, mew=1.5)
    axs[1].set_title('dI/dV (charging rings)')
    axs[1].set_xlabel('x [Å]')
    fig.tight_layout()
    path = os.path.join(OUT, 'Ruslan_long_xy_V1.00_W0.050.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'REVIEW: {path}')

    print(f'DONE in {time.time()-t0:.1f}s  → {OUT}')


if __name__ == '__main__':
    main()
