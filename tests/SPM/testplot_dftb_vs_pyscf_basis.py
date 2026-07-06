#!/usr/bin/env python3
"""
Compare DFTB 3ob STO basis vs pySCF def2-SVP GTO basis for C, N, O.

Plots radial functions R_l(r) in log scale from 0 to 4 Å.
Both raw and peak-normalized versions are shown.
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyscf import gto

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, _ROOT)
from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd, evaluate_sto_1d

BOHR2ANG = 0.529177210903
WFC_PATH = os.path.join(_ROOT, 'spammm', 'quantum', 'DFTB', 'data', 'wfc.3ob-3-1.hsd')
OUT_DIR  = os.path.join(_ROOT, 'debug', 'plot_3ob_basis_tails')
os.makedirs(OUT_DIR, exist_ok=True)

basis_data = parse_wfc_hsd(WFC_PATH)

# Radial grid in Angstrom, convert to Bohr for evaluation
r_ang = np.linspace(0.001, 4.0, 2000)
r_bohr = r_ang / BOHR2ANG

# ── GTO radial evaluation ──
def eval_gto_radial(r_bohr, l, exps, coeffs):
    """R_l(r) = r^l * sum_i c_i * exp(-alpha_i * r^2).  Units: Bohr, exps in Bohr^-2."""
    r = np.asarray(r_bohr, dtype=np.float64)
    result = np.zeros_like(r)
    for a, c in zip(exps, coeffs):
        result += c * np.exp(-a * r**2)
    return r**l * result

def get_pyscf_basis(elem, basis_name='def2-svp'):
    """Returns list of (l, exps, coeffs) for each basis function."""
    b = gto.basis.load(basis_name, elem)
    funcs = []
    for f in b:
        l = f[0]
        prims = f[1:]
        exps = np.array([p[0] for p in prims])
        coeffs = np.array([p[1] for p in prims])
        funcs.append((l, exps, coeffs))
    return funcs

ELEMENTS = ['C', 'N', 'O']
SHELL_NAMES = {0: 's', 1: 'p', 2: 'd'}

# ── Plot: raw values (log scale) ──
fig, axes = plt.subplots(len(ELEMENTS), 2, figsize=(14, 4*len(ELEMENTS)), squeeze=False)

# ── Plot: peak-normalized (log scale) ──
fig_n, axes_n = plt.subplots(len(ELEMENTS), 2, figsize=(14, 4*len(ELEMENTS)), squeeze=False)

for i_elem, elem in enumerate(ELEMENTS):
    # DFTB 3ob
    sp = basis_data[elem]
    dftb_funcs = []
    for orb in sp['orbitals']:
        l = orb['AngularMomentum']
        exps = np.asarray(orb['Exponents'], dtype=np.float64)
        coeffs = np.asarray(orb['Coefficients'], dtype=np.float64)
        dftb_funcs.append((l, exps, coeffs, 'DFTB'))

    # pySCF def2-SVP
    pyscf_funcs = get_pyscf_basis(elem, 'def2-svp')

    for i_col, target_l in enumerate([0, 1]):
        ax = axes[i_elem, i_col]
        ax_n = axes_n[i_elem, i_col]
        shell = SHELL_NAMES[target_l]

        # Plot DFTB (single contracted function per l)
        for l, exps, coeffs, tag in dftb_funcs:
            if l != target_l:
                continue
            R = evaluate_sto_1d(r_bohr, l, exps, coeffs)
            label = f"DFTB {shell} (contracted, {len(exps)} prim)"
            ax.semilogy(r_ang, np.abs(R) + 1e-30, color='blue', lw=2, label=label)
            # Normalized
            peak = np.max(np.abs(R))
            if peak > 0:
                ax_n.semilogy(r_ang, np.abs(R) / peak + 1e-30, color='blue', lw=2, label=label)

        # Plot pySCF functions
        colors_pyscf = ['red', 'orange', 'darkred']
        i_func = 0
        for l, exps, coeffs in pyscf_funcs:
            if l != target_l:
                continue
            R = eval_gto_radial(r_bohr, l, exps, coeffs)
            nPrim = len(exps)
            if nPrim == 1:
                label = f"pySCF {shell}{i_func+1} (uncontracted, exp={exps[0]:.3f})"
            else:
                label = f"pySCF {shell}{i_func+1} (contracted, {nPrim} prim)"
            color = colors_pyscf[i_func % len(colors_pyscf)]
            ax.semilogy(r_ang, np.abs(R) + 1e-30, color=color, lw=1.5, ls='--', label=label)
            # Normalized
            peak = np.max(np.abs(R))
            if peak > 0:
                ax_n.semilogy(r_ang, np.abs(R) / peak + 1e-30, color=color, lw=1.5, ls='--', label=label)
            i_func += 1

        ax.set_title(f"{elem} {shell}-orbital (raw)")
        ax.set_xlabel("r [Å]")
        ax.set_ylabel("|R_l(r)| [log]")
        ax.set_xlim(0, 4)
        ax.set_ylim(1e-4, 1e2)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax_n.set_title(f"{elem} {shell}-orbital (peak-normalized)")
        ax_n.set_xlabel("r [Å]")
        ax_n.set_ylabel("|R_l(r)| / max [log]")
        ax_n.set_xlim(0, 4)
        ax_n.set_ylim(1e-4, 1e2)
        ax_n.legend(fontsize=7)
        ax_n.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'dftb_vs_pyscf_raw.png'), dpi=150)
print(f"Saved: {OUT_DIR}/dftb_vs_pyscf_raw.png")

plt.tight_layout()
fig_n.savefig(os.path.join(OUT_DIR, 'dftb_vs_pyscf_normalized.png'), dpi=150)
print(f"Saved: {OUT_DIR}/dftb_vs_pyscf_normalized.png")

# ── Print comparison table ──
print("\n=== Basis comparison ===")
for elem in ELEMENTS:
    print(f"\n--- {elem} ---")
    sp = basis_data[elem]
    for orb in sp['orbitals']:
        l = orb['AngularMomentum']
        exps = np.asarray(orb['Exponents'])
        print(f"  DFTB l={l}: exps={exps} Bohr^-1, min_exp={exps.min():.3f}")
    pyscf_funcs = get_pyscf_basis(elem, 'def2-svp')
    for l, exps, coeffs in pyscf_funcs:
        if l > 1:
            continue
        print(f"  pySCF l={l}: exps={exps} Bohr^-2, min_exp={exps.min():.3f}")
