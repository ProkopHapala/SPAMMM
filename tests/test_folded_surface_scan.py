#!/usr/bin/env python3
"""
test_folded_surface_scan.py — Fit folded basis to NaCl surface potential.

Morse (Pauli+London) and Coulomb are INDEPENDENT problems, fit and plotted separately.
  - Morse is charge-independent → only 2 combinations (Na site, Cl site)
  - Coulomb depends on charge → 4 combinations (2 sites × 2 charges)

GPU is used ONLY to precompute brute-force reference profiles.
All fitting, evaluation, and plotting is done in pure Python/numpy.

Each plot: top = reference + fit + error (residual), bottom = basis functions.

Run:  python tests/test_folded_surface_scan.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from spammm.forcefields.MolecularDynamics import MolecularDynamics
from spammm.AtomicSystem import AtomicSystem
from spammm.surfaces.Ewald2D import Ewald2D

# =============================================================================
# Configuration
# =============================================================================

NACL_SUBSTRATE = os.path.join(_proj_root, 'data', 'substrates', 'NaCl_1x1_L3.xyz')
PLOT_DIR = os.path.join(_proj_root, 'debug', 'test_folded_surface_scan')

R_O = 1.7500; SQRT_E_O = np.sqrt(0.00260184625)
PROBES = {
    'O_neg': {'REQ': np.array([[R_O, SQRT_E_O, -0.5, 0.0]], dtype=np.float32), 'label': 'O (Q=-0.5)'},
    'O_pos': {'REQ': np.array([[R_O, SQRT_E_O, +0.5, 0.0]], dtype=np.float32), 'label': 'O (Q=+0.5)'},
}

Z_SURF_TOP = -3.25
SITES = {'Na': np.array([0.0, 0.0]), 'Cl': np.array([2.0, 2.0])}
LATTICE_A = 4.0

Z_SCAN_REL = np.linspace(0.3, 10.0, 120)
FIT_Z_RANGE = (1.5, 8.0)
FIT_NXY = 32
FIT_NZ_SAMP = 60

MORSE_NPBC = (4, 4, 0)
MORSE_ALPHA = 1.8

# Z-basis decay rates matched to physics
MORSE_ALPHAS = np.array([1.0, 1.8, 2.7, 3.6, 5.0], dtype=np.float64)  # /Å
COULOMB_ALPHAS = np.array([0.0, 0.3, 0.6, 1.0, 1.5], dtype=np.float64)  # /Å (0=constant; Ewald lateral avg=0 so constant is legitimate)

# Polynomial basis: (1 - x/R)^n  where x = z - z0, n = 2^m
# Effective decay rate near x=0 is n/R
POLY_R = 14.0  # cutoff [Å] — compact support, zero beyond R
MORSE_POLY_POWERS = np.array([4, 8, 16, 32, 64], dtype=np.float64)  # α_eff = n/R = 0.29..4.57 /Å
COULOMB_POLY_POWERS = np.array([0, 4, 8, 16, 32], dtype=np.float64)    # n=0=constant, α_eff = 0..2.29 /Å

NU = 4; NV = 4


# =============================================================================
# GPU reference: compute brute-force Morse on a grid
# =============================================================================

def _make_transforms(positions):
    T = np.zeros((len(positions), 3, 4), dtype=np.float32)
    T[:, 0, 0] = 1.0; T[:, 1, 1] = 1.0; T[:, 2, 2] = 1.0
    T[:, :, 3] = positions
    return T.reshape(-1, 12)

def _z_scan_positions(site_xy, z_rel):
    n = len(z_rel)
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 0] = site_xy[0]; pos[:, 1] = site_xy[1]
    pos[:, 2] = Z_SURF_TOP + z_rel
    return pos

def _grid_positions(z_range_rel, nxy, nz_samp):
    us = np.linspace(0.0, 1.0, nxy, endpoint=False, dtype=np.float32)
    vs = np.linspace(0.0, 1.0, nxy, endpoint=False, dtype=np.float32)
    zs = np.linspace(Z_SURF_TOP + z_range_rel[0], Z_SURF_TOP + z_range_rel[1], nz_samp, dtype=np.float32)
    pos = []
    for z in zs:
        for v in vs:
            for u in us:
                pos.append([u * LATTICE_A, v * LATTICE_A, z])
    return np.array(pos, dtype=np.float32), us, vs, zs

def compute_morse_reference(probe_REQ, positions, nPBC=(4,4,0), alpha_morse=1.8):
    md = MolecularDynamics(nloc=32, debug_build_options='-DDBG_UFF=0')
    md.init_rigid_molecule_batch(np.zeros((1,3), dtype=np.float32), probe_REQ, nSystems=max(len(positions), 1))
    md.set_surface(NACL_SUBSTRATE, nPBC=nPBC, alpha_morse=alpha_morse, bMacro=True)
    T = _make_transforms(positions)
    out = md.eval_rigid_getSurfMorse_components(T, chunk_size=md.nSystems, components=('pauli', 'london', 'coulomb'))
    return out


# =============================================================================
# Pure Python folded basis: fit and evaluate
# =============================================================================

def _build_basis_params(z_params, nu=NU, nv=NV, basis_type='exp', exclude_00=False):
    """Build (nbasis, 4) array of (iu, iv, z_param, z0). z_param = alpha (exp) or n_power (poly).
    exclude_00: skip (0,0) lateral mode (lateral average) — needed for Coulomb where
    finite-cluster lateral avg has same sign at Na/Cl but potential has opposite signs."""
    params = []
    z0 = Z_SURF_TOP + FIT_Z_RANGE[0]
    for iu in range(nu):
        for iv in range(nv):
            if exclude_00 and iu == 0 and iv == 0:
                continue
            for p in z_params:
                params.append((float(iu), float(iv), float(p), float(z0)))
    return np.array(params, dtype=np.float64)


def _basis_matrix(uvz, basis_params, basis_type='exp', R=POLY_R):
    """Evaluate basis functions at (u,v,z) points. Returns (N, nbasis) matrix.
    basis_type='exp':  bz = exp(-alpha * max(0, z-z0))
    basis_type='poly': bz = (1 - x)^n  where x = max(0, (z-z0)/R), clamped to [0,1]"""
    uvz = np.asarray(uvz, dtype=np.float64)
    bp = np.asarray(basis_params, dtype=np.float64)
    u = uvz[:, 0][:, None]; v = uvz[:, 1][:, None]; z = uvz[:, 2][:, None]
    ku = bp[None, :, 0]; kv = bp[None, :, 1]; zp = bp[None, :, 2]; z0 = bp[None, :, 3]
    bx = np.cos((2.0 * np.pi) * ku * u)
    by = np.cos((2.0 * np.pi) * kv * v)
    dz = np.maximum(0.0, z - z0)
    if basis_type == 'poly':
        x = np.minimum(dz / R, 1.0)
        bz = (1.0 - x) ** zp
    else:
        bz = np.exp(-zp * dz)
    return bx * by * bz


def _xyz_to_uvz(xyz):
    """Convert (x,y,z) to fractional (u,v,z) using lattice vectors."""
    u = xyz[:, 0] / LATTICE_A
    v = xyz[:, 1] / LATTICE_A
    return np.stack([u, v, xyz[:, 2]], axis=1)


def fit_component(uvz_samples, E_ref, z_params, fit_mask=None, basis_type='exp', R=POLY_R, exclude_00=False):
    """Fit folded basis coefficients to reference energy using lstsq.
    z_params = alphas (exp) or powers (poly). Returns (coeffs, basis_params, rmse_fit)."""
    bp = _build_basis_params(z_params, basis_type=basis_type, exclude_00=exclude_00)
    Phi = _basis_matrix(uvz_samples, bp, basis_type=basis_type, R=R)
    if fit_mask is not None:
        w = fit_mask.astype(np.float64)
    else:
        w = np.ones(len(E_ref), dtype=np.float64)
    Phiw = Phi * w[:, None]
    yw = E_ref * w
    S, *_ = np.linalg.lstsq(Phiw, yw, rcond=None)
    pred = Phi @ S
    if fit_mask is not None:
        rmse = np.sqrt(np.sum((pred - E_ref)**2 * fit_mask) / max(fit_mask.sum(), 1))
    else:
        rmse = np.sqrt(np.mean((pred - E_ref)**2))
    return S, bp, rmse


def eval_fit(uvz, coeffs, basis_params, basis_type='exp', R=POLY_R):
    """Evaluate fitted potential at (u,v,z) points."""
    return _basis_matrix(uvz, basis_params, basis_type=basis_type, R=R) @ coeffs


# =============================================================================
# Plotting: top = ref + fit + error, bottom = basis functions
# =============================================================================

def _safe_name(s):
    return s.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')

def _set_ylim_symmetric(ax, E_ref, z_rel=None, fit_range=None, E_mask=None):
    """Set y-limits symmetric: [-vmax, +vmax] where vmax = max|E_ref| over fit region."""
    if z_rel is not None and fit_range is not None:
        mask = (z_rel >= fit_range[0]) & (z_rel <= fit_range[1])
        if E_mask is not None: mask = mask & E_mask
        E_sub = E_ref[mask]
    else:
        E_sub = E_ref
    vmax = float(np.max(np.abs(E_sub)))
    ax.set_ylim(-vmax * 1.2, vmax * 1.2)

def _z_basis_values(z_rel, z_params, basis_type='exp', R=POLY_R):
    """Compute z-basis functions for plotting."""
    z0 = FIT_Z_RANGE[0]
    dz = np.maximum(0.0, z_rel - z0)
    if basis_type == 'poly':
        x = np.minimum(dz / R, 1.0)
        return np.stack([(1.0 - x) ** n for n in z_params], axis=1)
    else:
        return np.stack([np.exp(-a * dz) for a in z_params], axis=1)


def plot_fit_with_basis(z_rel, E_ref, E_fit, z_params, component_label, site_name, probe_label, save_dir, fit_range=None, basis_type='exp', R=POLY_R, E_mask=None):
    """Top: reference + fit + error. Bottom: basis functions.
    E_mask: boolean array — True=valid (plotted thick), False=excluded (plotted thin)."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    ax_top.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    # Reference: dotted, thick where valid, thin where excluded
    if E_mask is not None:
        ax_top.plot(z_rel[E_mask], E_ref[E_mask], 'b:', label='Reference (valid)', linewidth=1.5)
        ax_top.plot(z_rel[~E_mask], E_ref[~E_mask], 'b:', linewidth=0.5, alpha=0.4)
    else:
        ax_top.plot(z_rel, E_ref, 'b:', label='Reference', linewidth=1.5)
    # Fit: solid, thin
    ax_top.plot(z_rel, E_fit, 'r-', label=f'Fit ({basis_type})', linewidth=0.5)
    diff = E_fit - E_ref
    ax_top.plot(z_rel, diff, 'g-', label='Error (fit−ref)', linewidth=1, alpha=0.7)
    if fit_range is not None:
        ax_top.axvspan(fit_range[0], fit_range[1], alpha=0.1, color='orange', label='Fit region')
        mask = (z_rel >= fit_range[0]) & (z_rel <= fit_range[1])
        if E_mask is not None: mask = mask & E_mask
        rmse = np.sqrt(np.mean(diff[mask]**2)) if mask.any() else float('nan')
    else:
        rmse = np.sqrt(np.mean(diff**2))
    ax_top.set_ylabel('Energy [eV]')
    ax_top.set_title(f'{component_label} [{basis_type}]: {probe_label} above {site_name} (fit RMSE={rmse:.6f} eV)')
    ax_top.legend(loc='best', fontsize=8)
    _set_ylim_symmetric(ax_top, E_ref, z_rel=z_rel, fit_range=fit_range, E_mask=E_mask)
    # --- Bottom: basis functions ---
    bz_vals = _z_basis_values(z_rel, z_params, basis_type=basis_type, R=R)
    cmap = plt.cm.plasma
    for i in range(len(z_params)):
        p = z_params[i]
        if basis_type == 'poly':
            lbl = f'n={int(p)} (α_eff={p/R:.1f})'
        else:
            lbl = f'α={p:.1f}'
        ax_bot.plot(z_rel, bz_vals[:, i], color=cmap(i / max(len(z_params)-1, 1)), label=lbl, linewidth=1.5)
    ax_bot.axvspan(FIT_Z_RANGE[0], FIT_Z_RANGE[1], alpha=0.1, color='orange')
    ax_bot.set_xlabel('z above surface [Å]'); ax_bot.set_ylabel('Basis amplitude')
    ax_bot.set_title(f'Z-basis functions ({component_label}, {basis_type})')
    ax_bot.legend(fontsize=8, ncol=len(z_params))
    fig.tight_layout()
    fname = f'{_safe_name(component_label)}_{basis_type}_{_safe_name(probe_label)}_above_{site_name}.png'
    fig.savefig(os.path.join(save_dir, fname), dpi=120)
    plt.close(fig)


def plot_basis_correspondence(save_dir, R=POLY_R):
    """Show how (1-x/R)^n approximates exp(-alpha*x) with alpha = n/R.
    For each pair, plot exp, poly, and their difference."""
    x = np.linspace(0, R*1.2, 300)
    # Matched pairs: (alpha, n) where alpha ≈ n/R
    pairs = [
        # Morse range
        (1.0,  5,  'Morse'), (1.8,  9,  'Morse'), (2.7, 14, 'Morse'), (3.6, 18, 'Morse'), (5.0, 25, 'Morse'),
        # Powers of 2 (actual poly basis)
        (0.4,  2,  'poly'),  (0.8,  4,  'poly'),  (1.6,  8,  'poly'),  (3.2, 16, 'poly'),  (6.4, 32, 'poly'),
        # Coulomb range
        (0.3,  2,  'Coulomb'), (0.6,  3,  'Coulomb'), (1.0,  5,  'Coulomb'), (1.5,  8,  'Coulomb'),
        # Poly Coulomb
        (0.2,  1,  'poly'), (0.4,  2,  'poly'), (0.8,  4,  'poly'), (1.6,  8,  'poly'),
    ]
    # Deduplicate by (alpha, n)
    seen = set(); pairs_unique = []
    for a, n, tag in pairs:
        key = (round(a, 2), n)
        if key not in seen:
            seen.add(key); pairs_unique.append((a, n, tag))

    n_pairs = len(pairs_unique)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 2.2*n_pairs), sharex=True)
    if n_pairs == 1: axes = [axes]
    for ax, (alpha, n, tag) in zip(axes, pairs_unique):
        exp_val = np.exp(-alpha * x)
        x_clip = np.clip(x / R, 0, 1)
        poly_val = (1 - x_clip) ** n
        ax.plot(x, exp_val, 'b-', label=f'exp(−{alpha:.1f}·x)', linewidth=2)
        ax.plot(x, poly_val, 'r--', label=f'(1−x/R)^{n}, α_eff={n/R:.2f}', linewidth=2)
        ax.plot(x, poly_val - exp_val, 'g:', label='diff (poly−exp)', linewidth=1, alpha=0.7)
        ax.axvline(R, color='gray', linewidth=0.5, linestyle=':', label=f'R={R}')
        ax.set_ylabel('amplitude')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_title(f'α={alpha:.1f}/Å  vs  n={n} (α_eff={n/R:.2f}/Å)  [{tag}]', fontsize=9)
    axes[-1].set_xlabel('x = z − z₀  [Å]')
    fig.suptitle(f'Correspondence: exp(−α·x)  vs  (1−x/R)^n   [R={R} Å,  α_eff = n/R]', fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(save_dir, 'basis_correspondence.png'), dpi=120)
    plt.close(fig)
    print(f'  Saved basis_correspondence.png ({n_pairs} pairs)')


# =============================================================================
# Main: Morse and Coulomb as independent problems
# =============================================================================

def _fit_both(grid_uvz, E_ref_grid, exp_params, poly_params, label, fit_mask=None, exclude_00=False):
    """Fit both exp and poly bases, print diagnostics, return results dict."""
    results = {}
    for bt, params in [('exp', exp_params), ('poly', poly_params)]:
        S, bp, rmse = fit_component(grid_uvz, E_ref_grid, params, fit_mask=fit_mask, basis_type=bt, exclude_00=exclude_00)
        results[bt] = {'S': S, 'bp': bp, 'rmse': rmse, 'params': params}
        n_used = int(fit_mask.sum()) if fit_mask is not None else len(E_ref_grid)
        print(f'  {label} [{bt}]: nbasis={len(bp)} npoints={n_used} RMSE={rmse:.6f} eV')
        # Print (0,0) mode coefficients
        for ib in range(len(bp)):
            iu, iv, zp, z0 = bp[ib]
            if int(iu) == 0 and int(iv) == 0 and abs(S[ib]) > 1e-8:
                if bt == 'poly':
                    print(f'    n={int(zp)} (α_eff={zp/POLY_R:.1f}): coeff={S[ib]:+.6e}')
                else:
                    print(f'    α={zp:.2f}: coeff={S[ib]:+.6e}')
    return results

def _eval_rmse(z_rel, E_ref, E_fit, E_mask=None):
    """RMSE over fit z-range, optionally also masked by energy threshold."""
    mask = (z_rel >= FIT_Z_RANGE[0]) & (z_rel <= FIT_Z_RANGE[1])
    if E_mask is not None:
        mask = mask & E_mask
    return np.sqrt(np.mean((E_fit - E_ref)[mask]**2)) if mask.any() else float('nan')

def run_scan():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f'[folded_scan] Output: {PLOT_DIR}')
    print(f'[folded_scan] Fit region: {FIT_Z_RANGE[0]}-{FIT_Z_RANGE[1]} Å above surface')
    print(f'[folded_scan] Morse exp alphas: {MORSE_ALPHAS.tolist()}')
    print(f'[folded_scan] Morse poly powers: {MORSE_POLY_POWERS.tolist()} (R={POLY_R} Å)')
    print(f'[folded_scan] Coulomb exp alphas: {COULOMB_ALPHAS.tolist()}')
    print(f'[folded_scan] Coulomb poly powers: {COULOMB_POLY_POWERS.tolist()} (R={POLY_R} Å)')
    print(f'[folded_scan] Lateral basis: nu={NU} nv={NV} ({NU*NV} lateral modes)')

    # =========================================================================
    # PART A: MORSE (charge-independent → 1 probe, 2 sites)
    # =========================================================================
    print('\n' + '='*70)
    print('PART A: MORSE FIT (Pauli + London, charge-independent)')
    print('='*70)

    morse_probe = PROBES['O_neg']  # charge doesn't affect Morse
    morse_probe_label = 'O (Morse, Q irrelevant)'

    # --- GPU reference: z-scans at Na and Cl sites ---
    print('\n--- GPU reference: Morse z-scans ---')
    morse_zscan = {}
    all_pos = []
    for site_xy in SITES.values():
        all_pos.append(_z_scan_positions(site_xy, Z_SCAN_REL))
    all_pos = np.concatenate(all_pos, axis=0)
    out = compute_morse_reference(morse_probe['REQ'], all_pos)
    idx = 0
    for site_name in SITES:
        n = len(Z_SCAN_REL)
        morse_zscan[site_name] = {ck: out[ck][idx:idx+n] for ck in ['pauli', 'london', 'coulomb', 'total']}
        morse_ref = morse_zscan[site_name]['pauli'] + morse_zscan[site_name]['london']
        E_min = morse_ref.min(); z_min = Z_SCAN_REL[np.argmin(morse_ref)]
        print(f'  Morse above {site_name}: E_min={E_min:.6f} eV at z={z_min:.3f} Å')
        for ck in ['pauli', 'london']:
            arr = morse_zscan[site_name][ck]
            mask = (Z_SCAN_REL > 2.0) & (Z_SCAN_REL < 5.0) & (np.abs(arr) > 1e-10)
            if mask.sum() > 3:
                slope = np.polyfit(Z_SCAN_REL[mask], np.log(np.abs(arr[mask])), 1)[0]
                print(f'    {ck:>8s}: decay ~ {abs(slope):.3f} /Å')
        idx += n

    # --- GPU reference: grid for fitting ---
    print('\n--- GPU reference: Morse grid ---')
    grid_pos, grid_us, grid_vs, grid_zs = _grid_positions(FIT_Z_RANGE, FIT_NXY, FIT_NZ_SAMP)
    print(f'  Grid: {FIT_NXY}x{FIT_NXY}x{FIT_NZ_SAMP} = {len(grid_pos)} points')
    out_grid = compute_morse_reference(morse_probe['REQ'], grid_pos)
    morse_ref_grid = out_grid['pauli'] + out_grid['london']
    print(f'  Morse ref range [{morse_ref_grid.min():.4f}, {morse_ref_grid.max():.4f}] eV')

    # --- Energy-based mask: exclude high-repulsive region ---
    E_min = morse_ref_grid.min()
    morse_mask = morse_ref_grid < -E_min  # only fit well + near-zero region
    print(f'  Morse mask: E_min={E_min:.6f} eV, threshold={-E_min:.6f} eV, kept={morse_mask.sum()}/{len(morse_mask)} points')

    # --- Pure Python fit: both exp and poly ---
    print('\n--- Fit Morse (pure Python, exp vs poly) ---')
    grid_uvz = _xyz_to_uvz(grid_pos)
    morse_results = _fit_both(grid_uvz, morse_ref_grid, MORSE_ALPHAS, MORSE_POLY_POWERS, 'Morse', fit_mask=morse_mask)

    # --- Evaluate and plot: 2 sites, separate exp and poly plots ---
    print('\n--- Plot Morse fits ---')
    for site_name, site_xy in SITES.items():
        pos = _z_scan_positions(site_xy, Z_SCAN_REL)
        uvz = _xyz_to_uvz(pos)
        E_ref = morse_zscan[site_name]['pauli'] + morse_zscan[site_name]['london']
        E_min_z = E_ref.min()
        z_mask = E_ref < -E_min_z  # same threshold as grid mask
        for bt in ['exp', 'poly']:
            E_fit = eval_fit(uvz, morse_results[bt]['S'], morse_results[bt]['bp'], basis_type=bt)
            rmse = _eval_rmse(Z_SCAN_REL, E_ref, E_fit, E_mask=z_mask)
            print(f'  Morse [{bt}] above {site_name}: RMSE={rmse:.6f} eV (masked: {int(z_mask.sum())}/{len(z_mask)} pts)')
            plot_fit_with_basis(Z_SCAN_REL, E_ref, E_fit, morse_results[bt]['params'], 'Morse', site_name, morse_probe_label, PLOT_DIR, fit_range=FIT_Z_RANGE, basis_type=bt, E_mask=z_mask)

    # =========================================================================
    # PART B: COULOMB (charge-dependent → 2 probes × 2 sites = 4 combos)
    # =========================================================================
    print('\n' + '='*70)
    print('PART B: COULOMB FIT (electrostatic, charge-dependent)')
    print('='*70)

    # --- Ewald2D reference: periodic Coulomb (replaces finite-cluster GPU brute-force) ---
    # Ewald gives zero lateral average for charge-neutral cell → (0,0) modes are legitimate.
    print('\n--- Ewald2D setup ---')
    mol = AtomicSystem(fname=NACL_SUBSTRATE)
    apos = np.asarray(mol.apos, dtype=float)
    surf_rx, surf_ry, surf_rz = apos[:, 0], apos[:, 1], apos[:, 2]
    surf_q = np.asarray(mol.qs, dtype=float)
    surf_lvec = np.asarray(mol.lvec, dtype=float)
    a_vec = surf_lvec[0, :2]; b_vec = surf_lvec[1, :2]
    EWALD_N_HARM = 6
    ew = Ewald2D(a_vec, b_vec, surf_rx, surf_ry, surf_rz, surf_q, n_harm=EWALD_N_HARM)
    print(f'  N_ions={len(surf_q)} Q_tot={np.sum(surf_q):.6f} n_harm={EWALD_N_HARM} N_G={ew.N_G}')

    # --- Ewald z-scans for both probes (phi * q_probe = energy) ---
    print('\n--- Ewald reference: Coulomb z-scans ---')
    coulomb_zscan = {}  # [probe_key][site_name]
    z_abs_scan = Z_SURF_TOP + Z_SCAN_REL  # absolute z for Ewald
    for pk, probe in PROBES.items():
        q_probe = float(probe['REQ'][0, 2])
        coulomb_zscan[pk] = {}
        for site_name, site_xy in SITES.items():
            phi = ew.phi_full_1d(site_xy[0], site_xy[1], z_abs_scan)
            E = phi * q_probe  # eV/e * e = eV
            coulomb_zscan[pk][site_name] = E
            print(f'  {pk} above {site_name}: Coulomb range [{E.min():.4f}, {E.max():.4f}] eV')

    # --- Ewald grid for fitting ---
    print('\n--- Ewald reference: Coulomb grid ---')
    coulomb_grid_refs = {}
    grid_z_abs = np.linspace(Z_SURF_TOP + FIT_Z_RANGE[0], Z_SURF_TOP + FIT_Z_RANGE[1], FIT_NZ_SAMP, dtype=np.float64)
    for pk, probe in PROBES.items():
        q_probe = float(probe['REQ'][0, 2])
        E_grid = np.zeros((FIT_NZ_SAMP, FIT_NXY, FIT_NXY), dtype=np.float64)
        us = np.linspace(0.0, 1.0, FIT_NXY, endpoint=False)
        vs = np.linspace(0.0, 1.0, FIT_NXY, endpoint=False)
        for iz, z_abs in enumerate(grid_z_abs):
            X, Y = np.meshgrid(us * LATTICE_A, vs * LATTICE_A)
            phi = ew.phi_vacuum_xy(X, Y, z_abs)  # vacuum: z > z_max(ions) = -3.25, our z >= -1.75
            E_grid[iz] = phi * q_probe
        coulomb_grid_refs[pk] = E_grid.ravel()
        print(f'  {pk}: Coulomb grid range [{coulomb_grid_refs[pk].min():.4f}, {coulomb_grid_refs[pk].max():.4f}] eV')

    # --- Fit Coulomb (Ewald reference, no lateral avg subtraction needed) ---
    print('\n--- Fit Coulomb (pure Python, exp vs poly) ---')
    coulomb_results = {}
    for pk, probe in PROBES.items():
        coulomb_results[pk] = _fit_both(grid_uvz, coulomb_grid_refs[pk], COULOMB_ALPHAS, COULOMB_POLY_POWERS, f'Coulomb {pk}')

    # --- Evaluate and plot: 4 combos, separate exp and poly plots ---
    print('\n--- Plot Coulomb fits ---')
    for pk, probe in PROBES.items():
        for site_name, site_xy in SITES.items():
            pos = _z_scan_positions(site_xy, Z_SCAN_REL)
            uvz = _xyz_to_uvz(pos)
            E_ref = coulomb_zscan[pk][site_name]
            for bt in ['exp', 'poly']:
                E_fit = eval_fit(uvz, coulomb_results[pk][bt]['S'], coulomb_results[pk][bt]['bp'], basis_type=bt)
                rmse = _eval_rmse(Z_SCAN_REL, E_ref, E_fit)
                pos_far = _z_scan_positions(site_xy, np.array([100.0]))
                E_fit_far = eval_fit(_xyz_to_uvz(pos_far), coulomb_results[pk][bt]['S'], coulomb_results[pk][bt]['bp'], basis_type=bt)[0]
                print(f'  Coulomb [{bt}] {pk} above {site_name}: RMSE={rmse:.6f} eV  E_ref(z=10)={E_ref[-1]:.6f}  E_fit(z=100)={E_fit_far:.6f}')
                plot_fit_with_basis(Z_SCAN_REL, E_ref, E_fit, coulomb_results[pk][bt]['params'], 'Coulomb', site_name, probe['label'], PLOT_DIR, fit_range=FIT_Z_RANGE, basis_type=bt)

    # =========================================================================
    # Summary
    # =========================================================================
    print('\n' + '='*70)
    print('SUMMARY (fit-region RMSE)')
    print('='*70)
    print(f'{"Component":>10s} {"Probe":>8s} {"Site":>4s} {"exp RMSE":>12s} {"poly RMSE":>12s}')
    print('-' * 52)
    for site_name in SITES:
        pos = _z_scan_positions(SITES[site_name], Z_SCAN_REL)
        uvz = _xyz_to_uvz(pos)
        E_ref = morse_zscan[site_name]['pauli'] + morse_zscan[site_name]['london']
        z_mask = E_ref < -E_ref.min()
        E_exp = eval_fit(uvz, morse_results['exp']['S'], morse_results['exp']['bp'], basis_type='exp')
        E_poly = eval_fit(uvz, morse_results['poly']['S'], morse_results['poly']['bp'], basis_type='poly')
        rmse_exp = _eval_rmse(Z_SCAN_REL, E_ref, E_exp, E_mask=z_mask)
        rmse_poly = _eval_rmse(Z_SCAN_REL, E_ref, E_poly, E_mask=z_mask)
        print(f'{"Morse":>10s} {"—":>8s} {site_name:>4s} {rmse_exp:>12.6f} {rmse_poly:>12.6f}')
    for pk, probe in PROBES.items():
        for site_name in SITES:
            pos = _z_scan_positions(SITES[site_name], Z_SCAN_REL)
            uvz = _xyz_to_uvz(pos)
            E_ref = coulomb_zscan[pk][site_name]
            E_exp = eval_fit(uvz, coulomb_results[pk]['exp']['S'], coulomb_results[pk]['exp']['bp'], basis_type='exp')
            E_poly = eval_fit(uvz, coulomb_results[pk]['poly']['S'], coulomb_results[pk]['poly']['bp'], basis_type='poly')
            rmse_exp = _eval_rmse(Z_SCAN_REL, E_ref, E_exp)
            rmse_poly = _eval_rmse(Z_SCAN_REL, E_ref, E_poly)
            print(f'{"Coulomb":>10s} {pk:>8s} {site_name:>4s} {rmse_exp:>12.6f} {rmse_poly:>12.6f}')

    # =========================================================================
    # PART C: Basis correspondence plot
    # =========================================================================
    print('\n' + '='*70)
    print('PART C: BASIS CORRESPONDENCE (exp vs poly)')
    print('='*70)
    print(f'  Relation: α_eff = n/R  (R={POLY_R} Å)')
    print('  Plotting exp(−α·x) vs (1−x/R)^n for matched pairs...')
    plot_basis_correspondence(PLOT_DIR, R=POLY_R)

    print(f'\n[folded_scan] All plots saved to: {PLOT_DIR}')


if __name__ == '__main__':
    run_scan()
