#!/usr/bin/env python3
"""
Plot 3ob basis radial functions in log scale and fit corrected exponential tails.

The 3ob basis is contracted for fast DFTB calculations (short range, hard cutoff at 6 Bohr).
For AFM density projection we want orbitals that decay nicely exponentially at large r.

Strategy:
  1. Parse wfc.3ob-3-1.hsd to get STO parameters (exponents, coefficients)
  2. Evaluate R_l(r) on a fine grid, plot |R_l(r)| in log scale
  3. Fit a single STO tail A * r^l * exp(-alpha * r) to the asymptotic region
  4. Create modified basis: original for r < r_match, smooth transition to fitted exponential for r > r_match
  5. Write modified wfc file and plot comparison

Usage:
  python plot_3ob_basis_tails.py
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, _ROOT)

from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd, evaluate_sto_1d

BOHR2ANG = 0.529177210903  # Bohr to Angstrom

WFC_PATH = os.path.join(_ROOT, 'spammm', 'quantum', 'DFTB', 'data', 'wfc.3ob-3-1.hsd')
OUT_DIR  = os.path.join(_ROOT, 'debug', 'plot_3ob_basis_tails')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Parse basis ──
basis_data = parse_wfc_hsd(WFC_PATH)
print(f"Parsed {len(basis_data)} species: {list(basis_data.keys())}")

# ── 2. Evaluate and plot ──
# Work in Angstrom for plotting; convert to Bohr for STO evaluation
r_ang = np.linspace(0.0, 4.0, 2000)  # Angstrom, plot range 0-4 Å
r_bohr = r_ang / BOHR2ANG  # convert to Bohr for evaluate_sto_1d
r_safe = np.where(r_bohr < 1e-10, 1e-10, r_bohr)  # avoid log(0)

# Fit range in Bohr (0.5-1.0 Å)
R_FIT_START_ANG = 0.5
R_FIT_END_ANG   = 1.0
R_FIT_START_BOHR = R_FIT_START_ANG / BOHR2ANG
R_FIT_END_BOHR   = R_FIT_END_ANG / BOHR2ANG
# Match point: start of fit region
R_MATCH_ANG = R_FIT_START_ANG
R_MATCH_BOHR = R_FIT_START_BOHR

ELEMENTS_TO_PLOT = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
COLORS = {'s': 'blue', 'p': 'red', 'd': 'green'}
LS_ORIG = '-'
LS_CORR = '--'

def fit_exponential_tail(r, R, l, r_match, r_fit_start, r_fit_end):
    """Fit A * r^l * exp(-alpha * r) to data in [r_fit_start, r_fit_end] region (Bohr units).

    We fit log(|R|/r^l) = log|A| - alpha * r  (linear in r)
    Then return (A, alpha) such that the tail matches at r_match.
    """
    mask = (r >= r_fit_start) & (r <= r_fit_end) & (np.abs(R) > 1e-30)
    if mask.sum() < 5:
        print(f"  WARNING: not enough points for tail fit (l={l}, fit range [{r_fit_start:.3f}, {r_fit_end:.3f}] Bohr)")
        return None, None

    r_fit = r[mask]
    R_fit = R[mask]
    # log|R| - l*log(r) = log|A| - alpha*r
    y = np.log(np.abs(R_fit)) - l * np.log(r_fit)
    # Linear fit: y = a + b*r  =>  A = exp(a), alpha = -b
    A_mat = np.vstack([np.ones_like(r_fit), r_fit]).T
    sol, res, rank, sv = np.linalg.lstsq(A_mat, y, rcond=None)
    logA, alpha = sol[0], -sol[1]
    A = np.sign(R_fit[0]) * np.exp(logA)

    # Renormalize A so the tail matches the original at r_match
    idx_match = np.argmin(np.abs(r - r_match))
    R_orig_at_match = R[idx_match]
    R_tail_at_match = A * r_match**l * np.exp(-alpha * r_match)
    if abs(R_tail_at_match) > 1e-30:
        A = A * R_orig_at_match / R_tail_at_match

    return A, alpha

def smooth_blend(r, r_match, r_blend):
    """Smoothstep blend: 0 for r < r_match, 1 for r > r_match + r_blend."""
    t = np.clip((r - r_match) / r_blend, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)  # smoothstep

def corrected_radial(r, l, exps, coeffs, A_tail, alpha_tail, r_match, r_blend):
    """Original STO blended to single exponential tail."""
    R_orig = evaluate_sto_1d(r, l, exps, coeffs)
    if A_tail is None or alpha_tail is None:
        return R_orig
    R_tail = A_tail * r**l * np.exp(-alpha_tail * r)
    w = smooth_blend(r, r_match, r_blend)
    return (1.0 - w) * R_orig + w * R_tail

# ── Plot each element ──
fig, axes = plt.subplots(len(ELEMENTS_TO_PLOT), 1, figsize=(12, 4*len(ELEMENTS_TO_PLOT)), squeeze=False)

corrected_basis = {}  # store modified parameters for output

for i_elem, elem in enumerate(ELEMENTS_TO_PLOT):
    if elem not in basis_data:
        print(f"  Skipping {elem} (not in basis)")
        continue
    sp = basis_data[elem]
    ax = axes[i_elem, 0]

    corrected_orbitals = []
    for orb in sp['orbitals']:
        l = orb['AngularMomentum']
        exps = np.asarray(orb['Exponents'], dtype=np.float64)
        coeffs = np.asarray(orb['Coefficients'], dtype=np.float64)  # (nPow, nAlpha)
        cutoff = orb['Cutoff']  # 6.0 Bohr

        # Original STO (evaluated in Bohr)
        R_orig = evaluate_sto_1d(r_safe, l, exps, coeffs)

        # Fit tail in narrow range [0.5, 1.0] Å = [R_FIT_START_BOHR, R_FIT_END_BOHR] Bohr
        # Match from r_match = 0.5 Å, blend over 0.3 Å
        r_match = R_MATCH_BOHR
        r_blend = 0.3 / BOHR2ANG  # 0.3 Å in Bohr
        A_tail, alpha_tail = fit_exponential_tail(r_safe, R_orig, l, r_match, R_FIT_START_BOHR, R_FIT_END_BOHR)

        if A_tail is not None:
            R_corr = corrected_radial(r_safe, l, exps, coeffs, A_tail, alpha_tail, r_match, r_blend)
        else:
            R_corr = R_orig

        # Plot in Angstrom
        label_orig = f"{elem} {l} (orig)"
        label_corr = f"{elem} {l} (corr)"
        color = COLORS.get(['s','p','d'][l] if l < 3 else 's', 'black')
        ax.semilogy(r_ang, np.abs(R_orig) + 1e-30, color=color, ls=LS_ORIG, lw=1.5, label=label_orig)
        if A_tail is not None:
            ax.semilogy(r_ang, np.abs(R_corr) + 1e-30, color=color, ls=LS_CORR, lw=1.5, label=label_corr)
            # Also plot the pure tail
            R_pure_tail = A_tail * r_safe**l * np.exp(-alpha_tail * r_safe)
            ax.semilogy(r_ang, np.abs(R_pure_tail) + 1e-30, color=color, ls=':', lw=0.8, alpha=0.5, label=f"{elem} {l} (tail fit)")

        # Mark fit range and original cutoff (in Angstrom)
        ax.axvspan(R_FIT_START_ANG, R_FIT_END_ANG, color='yellow', alpha=0.15)
        ax.axvline(cutoff * BOHR2ANG, color='gray', ls='--', lw=0.5, alpha=0.5)

        print(f"  {elem} l={l}: nAlpha={len(exps)}, exps={exps}, alpha_tail={alpha_tail:.4f} Bohr^-1" if alpha_tail else f"  {elem} l={l}: tail fit failed")

        # Store corrected orbital info
        corr_orb = dict(orb)
        if A_tail is not None and alpha_tail is not None:
            corr_orb['tail_A'] = A_tail
            corr_orb['tail_alpha'] = alpha_tail
            corr_orb['tail_r_match'] = r_match
            corr_orb['tail_r_blend'] = r_blend
            corr_orb['Cutoff'] = 15.0  # extend cutoff
        corrected_orbitals.append(corr_orb)

    corrected_basis[elem] = {'AtomicNumber': sp['AtomicNumber'], 'orbitals': corrected_orbitals}

    ax.set_title(f"{elem} (Z={sp['AtomicNumber']})")
    ax.set_xlabel("r [Å]")
    ax.set_ylabel("|R_l(r)| [log]")
    ax.set_xlim(0, 4)
    ax.set_ylim(1e-4, 1e2)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '3ob_basis_log_tails.png'), dpi=150)
print(f"\nSaved plot: {os.path.join(OUT_DIR, '3ob_basis_log_tails.png')}")

# ── 3. Write modified wfc HSD file ──
def write_modified_wfc(corrected_basis, path):
    """Write modified wfc HSD with extended cutoffs and tail parameters as comments."""
    with open(path, 'w') as f:
        for elem, sp in corrected_basis.items():
            f.write(f"{elem} {{\n")
            f.write(f"  AtomicNumber = {sp['AtomicNumber']}\n")
            for orb in sp['orbitals']:
                l = orb['AngularMomentum']
                exps = orb['Exponents']
                coeffs = orb['Coefficients']
                cutoff = orb['Cutoff']
                f.write(f"  Orbital {{\n")
                f.write(f"    AngularMomentum = {l}\n")
                f.write(f"    Occupation = {orb['Occupation']:.15E}\n")
                f.write(f"    Cutoff = {cutoff:.2f}\n")
                if 'tail_alpha' in orb:
                    f.write(f"    # Tail correction: A={orb['tail_A']:.6e}, alpha={orb['tail_alpha']:.6f}, r_match={orb['tail_r_match']:.1f}\n")
                f.write(f"    Exponents {{\n")
                # Write exponents as space-separated
                f.write("    " + " ".join(f"{e:.15E}" for e in exps) + "\n")
                f.write(f"    }}\n")
                f.write(f"    Coefficients {{\n")
                # Coefficients: (nPow, nAlpha) in Fortran column-major
                nPow, nAlpha = coeffs.shape
                flat = coeffs.flatten(order='F')
                for j in range(nAlpha):
                    f.write("    " + " ".join(f"{c:.15E}" for c in coeffs[:, j]) + "\n")
                f.write(f"    }}\n")
                f.write(f"  }}\n")
            f.write(f"}}\n\n")
    print(f"Written modified basis: {path}")

write_modified_wfc(corrected_basis, os.path.join(OUT_DIR, 'wfc.3ob-3-1.corrected.hsd'))

# ── 4. Summary plot: all s-orbitals together ──
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
for elem in ELEMENTS_TO_PLOT:
    if elem not in basis_data:
        continue
    sp = basis_data[elem]
    for orb in sp['orbitals']:
        l = orb['AngularMomentum']
        if l != 0:
            continue
        exps = np.asarray(orb['Exponents'], dtype=np.float64)
        coeffs = np.asarray(orb['Coefficients'], dtype=np.float64)
        R = evaluate_sto_1d(r_safe, l, exps, coeffs)
        ax1.semilogy(r_ang, np.abs(R) + 1e-30, label=f"{elem} s (orig)")
        # Corrected
        if elem in corrected_basis:
            corr_orb = corrected_basis[elem]['orbitals'][0]  # s orbital
            if 'tail_alpha' in corr_orb:
                R_c = corrected_radial(r_safe, l, exps, coeffs,
                                       corr_orb['tail_A'], corr_orb['tail_alpha'],
                                       corr_orb['tail_r_match'], corr_orb['tail_r_blend'])
                ax2.semilogy(r_ang, np.abs(R_c) + 1e-30, label=f"{elem} s (corr)")

ax1.set_title("Original 3ob s-orbitals (log scale)")
ax1.set_xlabel("r [Å]")
ax1.set_ylabel("|R(r)|")
ax1.set_xlim(0, 4)
ax1.set_ylim(1e-4, 1e2)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.axvline(6.0 * BOHR2ANG, color='gray', ls='--', lw=0.5, label='cutoff=6 Bohr')
ax1.axvspan(R_FIT_START_ANG, R_FIT_END_ANG, color='yellow', alpha=0.15)

ax2.set_title("Corrected 3ob s-orbitals (log scale)")
ax2.set_xlabel("r [Å]")
ax2.set_ylabel("|R(r)|")
ax2.set_xlim(0, 4)
ax2.set_ylim(1e-4, 1e2)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.axvspan(R_FIT_START_ANG, R_FIT_END_ANG, color='yellow', alpha=0.15)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '3ob_s_comparison.png'), dpi=150)
print(f"Saved plot: {os.path.join(OUT_DIR, '3ob_s_comparison.png')}")

print("\n=== Summary ===")
print(f"Original basis: {WFC_PATH}")
print(f"Corrected basis: {os.path.join(OUT_DIR, 'wfc.3ob-3-1.corrected.hsd')}")
print(f"Plots: {OUT_DIR}/")
print("\nKey observations:")
print("  - Original 3ob basis has hard cutoff at 6 Bohr")
print("  - Contraction causes non-exponential behavior at large r")
print("  - Corrected basis: matches original near core, exponential tail at large r")
print("  - Cutoff extended to 15 Bohr for AFM density projection")
