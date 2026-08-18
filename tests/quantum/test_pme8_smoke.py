#!/usr/bin/env python3
"""Smoke test: PME8.cl compiles and matches PME.cl on 4-site square tetramer.

Compares PauliSolverCL (4-site) vs PauliSolverCL8 (8-site, 4 active + 4 spectators)
on the square_tetramer geometry. Verifies current parity within float tolerance.
"""
import os, sys
import numpy as np

def main():
    from spammm.quantum import pauli_scan as ps
    from spammm.quantum.PauliSolverCL import PauliSolverCL
    from spammm.quantum.PauliSolverCL8 import PauliSolverCL8

    # 4-site square geometry
    params = ps.ruslan_default_params(
        geometry_file='data/charge_rings/square_tetramer.txt',
        nsite=4, Qzz=0.0, W=0.05,
        VBias=1.2, Temp=2.6, z_tip=6.0, zV0=-0.9, zVd=20.0,
        npix=20, L=15.0,
    )
    spos4, rots4, _ = ps.make_site_geom(params)
    print(f"[smoke] 4-site geometry:\n{spos4[:,:3]}", flush=True)

    # --- PME4 reference ---
    print("[smoke] Building PME4 solver...", flush=True)
    sol4 = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=False)
    out4 = ps.scan_xy(sol4, spos4, rots4, params)
    I4 = out4['STM']
    print(f"[smoke] PME4 I range=[{I4.min():.4e},{I4.max():.4e}]", flush=True)

    # --- PME8: embed 4 active + 4 spectators ---
    # Pad spos to 8 sites: 4 real + 4 far/high-E spectators
    spos8 = np.zeros((8, 4), dtype=np.float64)
    spos8[:4] = spos4[:4]
    for i in range(4, 8):
        spos8[i] = [1e3*(i+1), 1e3*(i+1), 0.0, 1e3]  # far, high E
    rots8 = np.tile(np.eye(3, dtype=np.float32), (8, 1, 1))
    rots8[:4] = rots4[:4]

    # Wij: only among first 4 sites
    Wij8 = np.zeros((8, 8), dtype=np.float32)
    W = float(params['W'])
    for i in range(4):
        for j in range(i+1, 4):
            Wij8[i, j] = Wij8[j, i] = W

    print("[smoke] Building PME8 solver...", flush=True)
    sol8 = PauliSolverCL8(nSingle=8, preferred_vendor='nvidia', bPrint=False,
                          max_iter=5000, tol=1e-7)

    # Run scan_xy equivalent manually (pauli_scan.scan_xy is hardcoded to PME4 embed)
    npix = int(params['npix'])
    L = float(params['L'])
    zT = float(params['z_tip']) + float(params['Rtip'])
    pTips, Xs, Ys = ps.makePosXY(n=npix, L=L, p0=(0.0, 0.0, zT))
    Vtips = np.full(len(pTips), float(params['VBias']), dtype=np.float32)
    cpp = ps.make_cpp_params(params)
    cpp = cpp.copy(); cpp[6] = 0.0  # W=0 in cpp, use Wij instead
    cs, order = ps.make_quadrupole_Coeffs(params['Q0'], params['Qzz'])
    ps.configure_leads(sol8, params)

    I8_raw, Es8, Ts8, Probs8, StateEs8 = sol8.scan_current_tip(
        pTips=pTips, Vtips=Vtips, pSites=spos8, params=cpp, order=order, cs=cs,
        rots=rots8, Wij=Wij8, return_probs=True, return_state_energies=True,
    )
    I8 = I8_raw.reshape(npix, npix)
    I8 = np.nan_to_num(I8, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[smoke] PME8 I range=[{I8.min():.4e},{I8.max():.4e}]", flush=True)

    # --- Parity check ---
    abs_diff = np.abs(I4 - I8)
    max_abs = abs_diff.max()
    max_rel = (abs_diff / (np.abs(I4) + 1e-30)).max()
    print(f"\n[smoke] Parity: max_abs_diff={max_abs:.4e}, max_rel_diff={max_rel:.4e}", flush=True)

    # Check probability normalisation
    if Probs8 is not None:
        P_sums = Probs8.sum(axis=1)
        print(f"[smoke] P sum range=[{P_sums.min():.6f},{P_sums.max():.6f}] (should be ~1.0)", flush=True)

    # Tolerance: PME8 uses iterative solver, so allow more slack than PME4
    tol_abs = 1e-3
    if max_abs < tol_abs:
        print(f"[smoke] PASS: max_abs_diff < {tol_abs}", flush=True)
        return 0
    else:
        print(f"[smoke] FAIL: max_abs_diff={max_abs} >= {tol_abs}", flush=True)
        # Print worst pixels
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"[smoke] Worst pixel at {idx}: I4={I4[idx]:.6e}, I8={I8[idx]:.6e}", flush=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
