"""
PICCore.py — Compact atom-centered radial core for the contact_pme particle-mesh backend.

Fits the analytic soft-core residual v_i^S(r) on [r_lo_i, r_cut] with doubling-power
basis functions phi_m(r) = t(r)^p_m (p_m = 4, 8, 16, 32, 64), where
t = (r_cut - r)/(r_cut - r_lo_i). Every mode is exactly zero for r >= r_cut.

Contract version 2. The core residual alone need not contain a physical well;
only V_mesh + V_core must reproduce the repulsive wall, well, and tail.

Energy and force are always paired: every evaluator returns (E, F) with F = -∇E.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from spammm.surfaces.PMESplit import SplitParams, soft_core_split
from spammm.surfaces.ContactSurface import build_pic_buckets

# Doubling-power exponents (contract: p_m = 4, 8, 16, 32, 64)
CORE_POWERS = np.array([4, 8, 16, 32, 64], dtype=np.int64)
N_MODES = len(CORE_POWERS)


@dataclass
class CoreFit:
    """Per-atom core fit result."""
    coeffs: np.ndarray       # (na, N_MODES) float64 — raw-power coefficients
    r_lo: np.ndarray         # (na,) per-atom inner radius
    r_cut: float             # global outer cutoff
    powers: np.ndarray       # (N_MODES,) exponents
    basis: str               # 'raw' or 'hierarchical' (per-atom choice)
    cond_raw: np.ndarray     # (na,) condition number of raw-power design matrix
    cond_hier: np.ndarray    # (na,) condition number of hierarchical design matrix
    train_rmse_E: np.ndarray # (na,) training energy RMSE
    train_rmse_F: np.ndarray # (na,) training radial-force RMSE
    held_rmse_E: np.ndarray  # (na,) held-out energy RMSE (NaN if no held-out)
    held_rmse_F: np.ndarray  # (na,) held-out radial-force RMSE
    held_max_E: np.ndarray   # (na,) held-out max |dE|
    held_max_F: np.ndarray   # (na,) held-out max |dF|
    worst_r: np.ndarray      # (na,) worst held-out radius


# ── basis functions ─────────────────────────────────────────────────────────

def core_basis(r, r_lo, r_cut, powers=CORE_POWERS):
    """phi_m(r) = t^p_m and dphi_m/dr for the raw doubling-power basis.

    Returns (phi, dphi) each shape (..., N_MODES).
    For r >= r_cut: phi = 0, dphi = 0 (exact cutoff).
    For r <= r_lo: t = 1 (phi = 1, dphi = 0 — but domain is r >= r_lo).
    """
    r = np.asarray(r, dtype=np.float64)
    r_lo = np.asarray(r_lo, dtype=np.float64)
    D = r_cut - r_lo
    t = (r_cut - r) / D
    t = np.clip(t, 0.0, 1.0)
    active = (r >= r_lo) & (r < r_cut)
    phi = np.where(active[..., None], t[..., None] ** powers[None, :], 0.0)
    dphi = np.where(active[..., None], -powers[None, :] * t[..., None] ** (powers[None, :] - 1) / D, 0.0)
    return phi, dphi


def _hierarchical_transform(powers=CORE_POWERS):
    """Build the hierarchical basis transform matrix H such that phi_hier = H @ phi_raw.

    Hierarchical modes: t^4, t^8 - t^4, t^16 - t^8, t^32 - t^16, t^64 - t^32.
    H is (N_MODES, N_MODES) upper-triangular.
    """
    n = len(powers)
    H = np.zeros((n, n), dtype=np.float64)
    H[0, 0] = 1.0
    for i in range(1, n):
        H[i, i] = 1.0
        H[i, i - 1] = -1.0
    return H


# ── fit_core_1d ─────────────────────────────────────────────────────────────

def _sample_radii(r_lo, r_cut, n_shells=300, n_endpoint=30, rng=None):
    """Nonuniform shell sample points on [r_lo, r_cut] with dense endpoints.

    Concentrates samples near r_lo (steep repulsive wall) and r_cut (cutoff join).
    Returns sorted unique array of radii.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    D = r_cut - r_lo
    interior_n = n_shells - 2 * n_endpoint
    u_interior = rng.beta(0.5, 0.5, interior_n)  # U-shaped: dense near both endpoints
    r_interior = r_lo + D * u_interior
    r_lo_dense = r_lo + D * rng.uniform(0, 0.03, n_endpoint)
    r_cut_dense = r_lo + D * (1.0 - rng.uniform(0, 0.03, n_endpoint))
    r = np.concatenate([r_interior, r_lo_dense, r_cut_dense])
    return np.sort(np.unique(r))


def fit_core_1d(p: SplitParams, n_shells=300, n_endpoint=30, n_holdout=80,
                row_scale=1.0, deriv_row_scale=1.0, seed=42):
    """Fit per-atom core coefficients for v_i^S(r) on [r_lo_i, r_cut].

    Uses energy rows AND radial-derivative rows (dv_S/dr) at ALL training radii
    in a weighted least-squares system. Includes dense endpoint samples.
    Reports condition numbers for both raw and hierarchical bases.

    Chooses the basis (raw vs hierarchical) by condition number AND held-out
    energy-plus-radial-force error (NOT training energy alone).

    Returns CoreFit with per-atom coefficients, conditioning, and error metrics.
    """
    rng = np.random.default_rng(seed)
    r_lo_arr = np.atleast_1d(p.r_lo).astype(np.float64)
    r_cut = float(p.r_cut)
    na = p.na
    coeffs = np.zeros((na, N_MODES), dtype=np.float64)
    cond_raw_arr = np.zeros(na)
    cond_hier_arr = np.zeros(na)
    train_rmse_E = np.zeros(na)
    train_rmse_F = np.zeros(na)
    held_rmse_E = np.full(na, np.nan)
    held_rmse_F = np.full(na, np.nan)
    held_max_E = np.full(na, np.nan)
    held_max_F = np.full(na, np.nan)
    worst_r = np.full(na, np.nan)
    H = _hierarchical_transform()

    R0_arr = np.atleast_1d(p.R0).astype(np.float64)
    E0_arr = np.atleast_1d(p.E0).astype(np.float64)
    q_arr = np.atleast_1d(p.q).astype(np.float64)
    for i in range(na):
        r_lo_i = float(r_lo_arr[i])
        pi = SplitParams(R0=np.array([R0_arr[i]]), E0=np.array([E0_arr[i]]),
                         q=np.array([q_arr[i]]), alpha=p.alpha, q_tip=p.q_tip,
                         r_damp=p.r_damp, r_cut=r_cut)
        r_all = _sample_radii(r_lo_i, r_cut, n_shells, n_endpoint, rng)
        if len(r_all) > n_holdout + 10:
            hold_idx = rng.choice(len(r_all), size=n_holdout, replace=False)
            train_mask = np.ones(len(r_all), dtype=bool)
            train_mask[hold_idx] = False
            r_train = r_all[train_mask]
            r_hold = r_all[hold_idx]
        else:
            r_train = r_all
            r_hold = np.array([], dtype=np.float64)

        s_train = soft_core_split(r_train, pi)
        v_S = s_train['v_S']
        dv_S = s_train['dv_S_dr']
        phi_train, dphi_train = core_basis(r_train, r_lo_i, r_cut)

        # Energy rows + derivative rows at ALL training radii
        A_E = phi_train * row_scale
        b_E = v_S * row_scale
        A_D = dphi_train * deriv_row_scale
        b_D = dv_S * deriv_row_scale
        A = np.vstack([A_E, A_D])
        b = np.concatenate([b_E, b_D])

        cond_raw = float(np.linalg.cond(A))
        A_hier = A @ H.T
        cond_hier = float(np.linalg.cond(A_hier))

        c_raw, *_ = np.linalg.lstsq(A, b, rcond=None)
        c_hier, *_ = np.linalg.lstsq(A_hier, b, rcond=None)
        c_raw_from_hier = H.T @ c_hier

        if len(r_hold) > 0:
            s_hold = soft_core_split(r_hold, pi)
            v_S_hold = s_hold['v_S']
            dv_S_hold = s_hold['dv_S_dr']
            phi_hold, dphi_hold = core_basis(r_hold, r_lo_i, r_cut)
            E_raw = phi_hold @ c_raw
            F_raw = dphi_hold @ c_raw
            metric_raw = np.sqrt(np.mean((E_raw - v_S_hold)**2) + np.mean((F_raw - dv_S_hold)**2))
            E_hier = phi_hold @ c_raw_from_hier
            F_hier = dphi_hold @ c_raw_from_hier
            metric_hier = np.sqrt(np.mean((E_hier - v_S_hold)**2) + np.mean((F_hier - dv_S_hold)**2))
            chosen = c_raw_from_hier if metric_hier < metric_raw else c_raw
        else:
            chosen = c_raw_from_hier if cond_hier < cond_raw else c_raw

        coeffs[i] = chosen
        E_train = phi_train @ chosen
        F_train = dphi_train @ chosen
        train_rmse_E[i] = float(np.sqrt(np.mean((E_train - v_S)**2)))
        train_rmse_F[i] = float(np.sqrt(np.mean((F_train - dv_S)**2)))

        if len(r_hold) > 0:
            s_hold = soft_core_split(r_hold, pi)
            v_S_hold = s_hold['v_S']
            dv_S_hold = s_hold['dv_S_dr']
            phi_hold, dphi_hold = core_basis(r_hold, r_lo_i, r_cut)
            E_hold = phi_hold @ chosen
            F_hold = dphi_hold @ chosen
            err_E = E_hold - v_S_hold
            err_F = F_hold - dv_S_hold
            held_rmse_E[i] = float(np.sqrt(np.mean(err_E**2)))
            held_rmse_F[i] = float(np.sqrt(np.mean(err_F**2)))
            held_max_E[i] = float(np.max(np.abs(err_E)))
            held_max_F[i] = float(np.max(np.abs(err_F)))
            combined_err = np.abs(err_E) + np.abs(err_F)
            worst_idx = int(np.argmax(combined_err))
            worst_r[i] = float(r_hold[worst_idx])

        cond_raw_arr[i] = cond_raw
        cond_hier_arr[i] = cond_hier

    return CoreFit(coeffs=coeffs, r_lo=r_lo_arr, r_cut=r_cut, powers=CORE_POWERS,
                   basis='raw_or_hier_per_atom', cond_raw=cond_raw_arr, cond_hier=cond_hier_arr,
                   train_rmse_E=train_rmse_E, train_rmse_F=train_rmse_F,
                   held_rmse_E=held_rmse_E, held_rmse_F=held_rmse_F,
                   held_max_E=held_max_E, held_max_F=held_max_F, worst_r=worst_r)


# ── eval_core ───────────────────────────────────────────────────────────────

def eval_core(queries, atom_pos, fit: CoreFit, r_cut=None):
    """Batch evaluation of the compact core field at query points.

    Uses XY buckets (build_pic_buckets) with cell_size >= r_cut and 3×3 lookup.
    Returns (E, F) with E shape (nq,) and F shape (nq, 3). F = -∇E.

    Correctness-first: no silent candidate truncation. Raises ValueError on
    domain violation (r < r_lo) and RuntimeError on invalid bucket atom index.
    """
    queries = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    atom_pos = np.asarray(atom_pos, dtype=np.float64).reshape(-1, 3)
    nq = len(queries)
    na = len(atom_pos)
    rc = float(r_cut if r_cut is not None else fit.r_cut)
    assert na == fit.coeffs.shape[0], f"atom count mismatch: {na} vs {fit.coeffs.shape[0]}"

    margin = rc
    x0 = float(atom_pos[:, 0].min()) - margin
    y0 = float(atom_pos[:, 1].min()) - margin
    x1 = float(atom_pos[:, 0].max()) + margin
    y1 = float(atom_pos[:, 1].max()) + margin
    cell_size = rc  # >= r_cut (exact equality satisfies >=)
    bucket_atoms, bucket_offsets, nbx, nby = build_pic_buckets(atom_pos, x0, y0, x1, y1, cell_size)

    E = np.zeros(nq, dtype=np.float64)
    F = np.zeros((nq, 3), dtype=np.float64)

    for iq in range(nq):
        qx, qy, qz = queries[iq]
        bx = int((qx - x0) / cell_size)
        by = int((qy - y0) / cell_size)
        bx = min(max(bx, 0), nbx - 1)
        by = min(max(by, 0), nby - 1)
        for dy in range(-1, 2):
            iy = by + dy
            if iy < 0 or iy >= nby:
                continue
            for dx in range(-1, 2):
                ix = bx + dx
                if ix < 0 or ix >= nbx:
                    continue
                bid = iy * nbx + ix
                i0 = int(bucket_offsets[bid])
                i1 = int(bucket_offsets[bid + 1])
                for ia_idx in range(i0, i1):
                    ia = int(bucket_atoms[ia_idx])
                    if ia < 0 or ia >= na:
                        raise RuntimeError(f"Invalid atom index {ia} in bucket {bid} (fail-loud)")
                    dp = queries[iq] - atom_pos[ia]
                    r = float(np.linalg.norm(dp))
                    if r >= rc:
                        continue
                    r_lo_i = float(fit.r_lo[ia])
                    if r < r_lo_i:
                        raise ValueError(f"Domain violation: r={r:.6f} < r_lo={r_lo_i:.6f} for atom {ia} at query {iq}")
                    phi, dphi = core_basis(np.array([r]), r_lo_i, rc, fit.powers)
                    c = fit.coeffs[ia]
                    E[iq] += float(phi[0] @ c)
                    if r > 1e-30:
                        F[iq] -= float(dphi[0] @ c) * dp / r

    return E, F


def eval_core_direct(queries, atom_pos, fit: CoreFit, r_cut=None):
    """Direct all-atom sum (no buckets) — oracle for completeness validation.

    Returns (E, F) with E shape (nq,) and F shape (nq, 3). F = -∇E.
    """
    queries = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    atom_pos = np.asarray(atom_pos, dtype=np.float64).reshape(-1, 3)
    nq = len(queries)
    na = len(atom_pos)
    rc = float(r_cut if r_cut is not None else fit.r_cut)
    assert na == fit.coeffs.shape[0], f"atom count mismatch: {na} vs {fit.coeffs.shape[0]}"
    E = np.zeros(nq, dtype=np.float64)
    F = np.zeros((nq, 3), dtype=np.float64)
    for iq in range(nq):
        for ia in range(na):
            dp = queries[iq] - atom_pos[ia]
            r = float(np.linalg.norm(dp))
            if r >= rc:
                continue
            r_lo_i = float(fit.r_lo[ia])
            if r < r_lo_i:
                raise ValueError(f"Domain violation: r={r:.6f} < r_lo={r_lo_i:.6f} for atom {ia} at query {iq}")
            phi, dphi = core_basis(np.array([r]), r_lo_i, rc, fit.powers)
            c = fit.coeffs[ia]
            E[iq] += float(phi[0] @ c)
            if r > 1e-30:
                F[iq] -= float(dphi[0] @ c) * dp / r
    return E, F


# ── combined evaluation (V_mesh_direct + V_core) ────────────────────────────

def eval_core_and_soft(queries, atom_pos, p: SplitParams, fit: CoreFit, r_cut=None):
    """Combined direct-soft value (v_L direct sum) + fitted core.

    For validation: V_combined = Σ_i v_L_i(r) + Σ_i core_i(r).
    The mesh interpolation is NOT included (this is the direct reference
    for the combined wall/well/tail plot).

    Returns (E, F) with E shape (nq,) and F shape (nq, 3). F = -∇E.
    """
    queries = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    atom_pos = np.asarray(atom_pos, dtype=np.float64).reshape(-1, 3)
    nq = len(queries)
    na = len(atom_pos)
    rc = float(r_cut if r_cut is not None else fit.r_cut)
    E = np.zeros(nq, dtype=np.float64)
    F = np.zeros((nq, 3), dtype=np.float64)
    R0_arr = np.atleast_1d(p.R0).astype(np.float64)
    E0_arr = np.atleast_1d(p.E0).astype(np.float64)
    q_arr = np.atleast_1d(p.q).astype(np.float64)
    r_lo_arr = np.atleast_1d(p.r_lo).astype(np.float64)
    for iq in range(nq):
        for ia in range(na):
            dp = queries[iq] - atom_pos[ia]
            r = float(np.linalg.norm(dp))
            r_lo_i = float(r_lo_arr[ia])
            if r < r_lo_i:
                raise ValueError(f"Domain violation: r={r:.6f} < r_lo={r_lo_i:.6f} for atom {ia} at query {iq}")
            pi = SplitParams(R0=np.array([R0_arr[ia]]), E0=np.array([E0_arr[ia]]),
                             q=np.array([q_arr[ia]]), alpha=p.alpha, q_tip=p.q_tip,
                             r_damp=p.r_damp, r_cut=rc)
            s = soft_core_split(np.array([r]), pi)
            E[iq] += float(s['v_L'][0])
            if r > 1e-30:
                F[iq] -= float(s['dv_L_dr'][0]) * dp / r
            if r < rc:
                phi, dphi = core_basis(np.array([r]), r_lo_i, rc, fit.powers)
                c = fit.coeffs[ia]
                E[iq] += float(phi[0] @ c)
                if r > 1e-30:
                    F[iq] -= float(dphi[0] @ c) * dp / r
    return E, F
