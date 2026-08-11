"""
PICCore.py — Compact atom-centered radial core for the contact_pme particle-mesh backend.

Fits the analytic soft-core residual v_i^S(r) on [r_lo_i, r_b_i] with doubling-power
basis functions phi_m(r) = t(r)^p_m, where t = (r_b - r)/(r_b - r_lo).
Default powers p_m = (2, 4, 8, 16, 32) — every mode has φ(r_b)=φ'(r_b)=0.

Default split target is the compact residual (PMESplit split_mode='paw'/'hermite'/'plateau'):
  v_S = (1-W)(v - C), compact to r_b = R0 + Δ_b.

Fit uses Boltzmann weights on total v(r) (like ContactSurface.boltzmann_fit_weights)
plus separate E/F block normalization so the well is not lost to derivative scale.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from spammm.surfaces.PMESplit import SplitParams, soft_core_split, precompute_split_cache
from spammm.surfaces.ContactSurface import build_pic_buckets, boltzmann_fit_weights

# Doubling-power exponents — p0=2 keeps φ=φ'=0 at cutoff and resolves the well better
# than the legacy (4,8,16,32,64) which collapses mid-shell.
CORE_POWERS = np.array([2, 4, 8, 16, 32], dtype=np.int64)
N_MODES = len(CORE_POWERS)


@dataclass
class CoreFit:
    """Per-atom core fit result."""
    coeffs: np.ndarray       # (na, N_MODES) float64 — raw-power coefficients
    r_lo: np.ndarray         # (na,) per-atom inner radius (r_min)
    r_b: np.ndarray          # (na,) per-atom outer core cutoff (plateau r_b or legacy r_cut)
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

    @property
    def r_cut(self) -> float:
        """Backward-compat: global neighbor cutoff = max_i(r_b)."""
        return float(np.max(self.r_b))


# ── basis functions ─────────────────────────────────────────────────────────

def core_basis(r, r_lo, r_b, powers=CORE_POWERS):
    """phi_m(r) = t^p_m and dphi_m/dr for the raw doubling-power basis.

    t = (r_b - r)/(r_b - r_lo). Returns (phi, dphi) each shape (..., N_MODES).
    For r >= r_b: phi = 0, dphi = 0 (exact cutoff).
    For r <= r_lo: t clipped to 1 → phi = 1^{p}=1, dphi = 0 (AFM close-approach clamp).
    """
    r = np.asarray(r, dtype=np.float64)
    r_lo = np.asarray(r_lo, dtype=np.float64)
    r_b = np.asarray(r_b, dtype=np.float64)
    D = r_b - r_lo
    t = (r_b - r) / D
    t = np.clip(t, 0.0, 1.0)
    active = r < r_b
    # dphi only in open (r_lo, r_b); flat clamp below r_lo
    d_active = (r > r_lo) & (r < r_b)
    phi = np.where(active[..., None], t[..., None] ** powers[None, :], 0.0)
    dphi = np.where(d_active[..., None], -powers[None, :] * t[..., None] ** (powers[None, :] - 1) / D, 0.0)
    return phi, dphi


def _hierarchical_transform(powers=CORE_POWERS):
    """Build the hierarchical basis transform matrix H such that phi_hier = H @ phi_raw.

    Hierarchical modes: t^p0, t^p1 - t^p0, ...  H is (N_MODES, N_MODES) upper-triangular.
    """
    n = len(powers)
    H = np.zeros((n, n), dtype=np.float64)
    H[0, 0] = 1.0
    for i in range(1, n):
        H[i, i] = 1.0
        H[i, i - 1] = -1.0
    return H


# ── fit_core_1d ─────────────────────────────────────────────────────────────

def _sample_radii(r_lo, r_b, n_shells=300, n_endpoint=30, rng=None):
    """Nonuniform shell sample points on [r_lo, r_b] with dense endpoints."""
    if rng is None:
        rng = np.random.default_rng(42)
    D = r_b - r_lo
    interior_n = n_shells - 2 * n_endpoint
    u_interior = rng.beta(0.5, 0.5, interior_n)
    r_interior = r_lo + D * u_interior
    r_lo_dense = r_lo + D * rng.uniform(0, 0.03, n_endpoint)
    r_b_dense = r_lo + D * (1.0 - rng.uniform(0, 0.03, n_endpoint))
    r = np.concatenate([r_interior, r_lo_dense, r_b_dense])
    return np.sort(np.unique(r))


def _atom_outer(p: SplitParams, i: int = 0) -> float:
    """Per-atom outer core radius: r_b for compact splits, r_cut for legacy rho."""
    from spammm.surfaces.PMESplit import _COMPACT_SPLIT_MODES
    if p.split_mode in _COMPACT_SPLIT_MODES:
        return float(np.atleast_1d(p.r_b)[i])
    return float(p.r_cut)


def fit_core_1d(p: SplitParams, n_shells=300, n_endpoint=30, n_holdout=80,
                powers=None, seed=42, boltzmann_T=None):
    """Fit per-atom core coefficients for v_i^S(r) on [r_lo_i, r_b_i].

    Uses energy rows AND radial-derivative rows at ALL training radii with:
      - Boltzmann weights w = exp(-(v-v_min)/T) from total potential v(r)
      - Separate E/F block normalization λ_E=1/std(v_S), λ_F=1/std(dv_S)

    Returns CoreFit with per-atom coefficients, conditioning, and error metrics.
    """
    if powers is None:
        powers = CORE_POWERS
    powers = np.asarray(powers, dtype=np.int64)
    n_modes = len(powers)
    rng = np.random.default_rng(seed)
    r_lo_arr = np.atleast_1d(p.r_lo).astype(np.float64)
    na = p.na
    r_b_arr = np.array([_atom_outer(p, i) for i in range(na)], dtype=np.float64)
    coeffs = np.zeros((na, n_modes), dtype=np.float64)
    cond_raw_arr = np.zeros(na)
    cond_hier_arr = np.zeros(na)
    train_rmse_E = np.zeros(na)
    train_rmse_F = np.zeros(na)
    held_rmse_E = np.full(na, np.nan)
    held_rmse_F = np.full(na, np.nan)
    held_max_E = np.full(na, np.nan)
    held_max_F = np.full(na, np.nan)
    worst_r = np.full(na, np.nan)
    H = _hierarchical_transform(powers)

    for i in range(na):
        r_lo_i = float(r_lo_arr[i])
        r_b_i = float(r_b_arr[i])
        pi = p.with_atom(i)
        cache = precompute_split_cache(pi)
        r_all = _sample_radii(r_lo_i, r_b_i, n_shells, n_endpoint, rng)
        if len(r_all) > n_holdout + 10:
            hold_idx = rng.choice(len(r_all), size=n_holdout, replace=False)
            train_mask = np.ones(len(r_all), dtype=bool)
            train_mask[hold_idx] = False
            r_train = r_all[train_mask]
            r_hold = r_all[hold_idx]
        else:
            r_train = r_all
            r_hold = np.array([], dtype=np.float64)

        s_train = soft_core_split(r_train, pi, cache=cache)
        v_S = s_train['v_S']
        dv_S = s_train['dv_S_dr']
        v_tot = s_train['v']
        phi_train, dphi_train = core_basis(r_train, r_lo_i, r_b_i, powers)

        # Boltzmann weights on total potential (emphasize vdW well)
        w_b, T_used, _ = boltzmann_fit_weights(v_tot, T=boltzmann_T)
        w_b = np.asarray(w_b, dtype=np.float64)
        # E/F block normalization
        E_scale = max(float(np.std(v_S)), 1e-12)
        F_scale = max(float(np.std(dv_S)), 1e-12)
        lam_E = 1.0 / E_scale
        lam_F = 1.0 / F_scale
        sw = np.sqrt(w_b)
        A_E = phi_train * (sw * lam_E)[:, None]
        b_E = v_S * (sw * lam_E)
        A_D = dphi_train * (sw * lam_F)[:, None]
        b_D = dv_S * (sw * lam_F)
        A = np.vstack([A_E, A_D])
        b = np.concatenate([b_E, b_D])

        cond_raw = float(np.linalg.cond(A))
        A_hier = A @ H.T
        cond_hier = float(np.linalg.cond(A_hier))

        c_raw, *_ = np.linalg.lstsq(A, b, rcond=None)
        c_hier, *_ = np.linalg.lstsq(A_hier, b, rcond=None)
        c_raw_from_hier = H.T @ c_hier

        if len(r_hold) > 0:
            s_hold = soft_core_split(r_hold, pi, cache=cache)
            v_S_hold = s_hold['v_S']
            dv_S_hold = s_hold['dv_S_dr']
            phi_hold, dphi_hold = core_basis(r_hold, r_lo_i, r_b_i, powers)
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
            s_hold = soft_core_split(r_hold, pi, cache=cache)
            v_S_hold = s_hold['v_S']
            dv_S_hold = s_hold['dv_S_dr']
            phi_hold, dphi_hold = core_basis(r_hold, r_lo_i, r_b_i, powers)
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

    return CoreFit(coeffs=coeffs, r_lo=r_lo_arr, r_b=r_b_arr, powers=powers,
                   basis='raw_or_hier_per_atom', cond_raw=cond_raw_arr, cond_hier=cond_hier_arr,
                   train_rmse_E=train_rmse_E, train_rmse_F=train_rmse_F,
                   held_rmse_E=held_rmse_E, held_rmse_F=held_rmse_F,
                   held_max_E=held_max_E, held_max_F=held_max_F, worst_r=worst_r)


# ── eval_core ───────────────────────────────────────────────────────────────

def eval_core(queries, atom_pos, fit: CoreFit, r_cut=None):
    """Batch evaluation of the compact core field at query points.

    Uses XY buckets (build_pic_buckets) with cell_size >= r_core_max and 3×3 lookup.
    Returns (E, F) with E shape (nq,) and F shape (nq, 3). F = -∇E.
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
    cell_size = rc
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
                    r_b_i = float(fit.r_b[ia])
                    if r >= r_b_i:
                        continue
                    r_lo_i = float(fit.r_lo[ia])
                    phi, dphi = core_basis(np.array([r]), r_lo_i, r_b_i, fit.powers)
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
    assert na == fit.coeffs.shape[0], f"atom count mismatch: {na} vs {fit.coeffs.shape[0]}"
    E = np.zeros(nq, dtype=np.float64)
    F = np.zeros((nq, 3), dtype=np.float64)
    for iq in range(nq):
        for ia in range(na):
            dp = queries[iq] - atom_pos[ia]
            r = float(np.linalg.norm(dp))
            r_b_i = float(fit.r_b[ia])
            if r >= r_b_i:
                continue
            r_lo_i = float(fit.r_lo[ia])
            phi, dphi = core_basis(np.array([r]), r_lo_i, r_b_i, fit.powers)
            c = fit.coeffs[ia]
            E[iq] += float(phi[0] @ c)
            if r > 1e-30:
                F[iq] -= float(dphi[0] @ c) * dp / r
    return E, F


# ── combined evaluation (V_mesh_direct + V_core) ────────────────────────────

def eval_core_and_soft(queries, atom_pos, p: SplitParams, fit: CoreFit, r_cut=None):
    """Combined direct-soft value (v_L direct sum) + fitted core.

    For validation: V_combined = Σ_i v_L_i(r) + Σ_i core_i(r).
    Returns (E, F) with E shape (nq,) and F shape (nq, 3). F = -∇E.
    """
    queries = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    atom_pos = np.asarray(atom_pos, dtype=np.float64).reshape(-1, 3)
    nq = len(queries)
    na = len(atom_pos)
    E = np.zeros(nq, dtype=np.float64)
    F = np.zeros((nq, 3), dtype=np.float64)
    r_lo_arr = np.atleast_1d(p.r_lo).astype(np.float64)
    for iq in range(nq):
        for ia in range(na):
            dp = queries[iq] - atom_pos[ia]
            r = float(np.linalg.norm(dp))
            r_lo_i = float(r_lo_arr[ia])
            pi = p.with_atom(ia)
            s = soft_core_split(np.array([r]), pi)
            E[iq] += float(s['v_L'][0])
            if r > 1e-30:
                F[iq] -= float(s['dv_L_dr'][0]) * dp / r
            r_b_i = float(fit.r_b[ia])
            if r < r_b_i:
                phi, dphi = core_basis(np.array([r]), r_lo_i, r_b_i, fit.powers)
                c = fit.coeffs[ia]
                E[iq] += float(phi[0] @ c)
                if r > 1e-30:
                    F[iq] -= float(dphi[0] @ c) * dp / r
    return E, F
