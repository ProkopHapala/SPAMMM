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
# Deterministic radial grid for the GPU core fit (cs_fit_core_paw). Same count in the kernel.
CORE_FIT_NSAMP = 32


@dataclass
class CoreFit:
    """Per-atom core fit result."""
    coeffs: np.ndarray       # (na, N_MODES) float64 — raw-power coefficients
    r_lo: np.ndarray         # (na,) per-atom inner radius (r_min)
    r_b: np.ndarray          # (na,) per-atom outer core cutoff (plateau r_b or legacy r_cut)
    powers: np.ndarray       # (N_MODES,) exponents
    basis: str               # 'raw'/'hierarchical' raw powers, or independently C2 'smooth'
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

def core_basis(r, r_lo, r_b, powers=CORE_POWERS, smooth=False):
    """phi_m(r) = t^p_m and dphi_m/dr for the raw doubling-power basis.

    t = (r_b - r)/(r_b - r_lo). Returns (phi, dphi) each shape (..., N_MODES).
    For r >= r_b: phi = 0, dphi = 0 (exact cutoff).
    For r <= r_lo: t clipped to 1 → phi = 1^{p}=1, dphi = 0 (AFM close-approach clamp).
    smooth=True: map t through quintic smootherstep before taking powers. All
    modes then have C2 joins independently; no coefficient constraints consume
    fitting degrees of freedom. This is a distinct, explicitly tagged basis.
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
    dt = -np.ones_like(t)/D
    if smooth:
        dt *= 30*t*t*(1-t)*(1-t)
        t = t*t*t*(10+t*(-15+6*t))
    phi = np.where(active[..., None], t[..., None] ** powers[None, :], 0.0)
    dphi = np.where(d_active[..., None], powers[None, :] * t[..., None] ** (powers[None, :] - 1) * dt[..., None], 0.0)
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


def sample_core_shells(apos, r_lo, r_b, n_per_atom=256, seed=42):
    """Uniform-r shell samples per atom in [max(0.4, 0.7*r_lo), r_b] — training points
    for field-sample core fits (dense near the wall where the mesh fails)."""
    rng = np.random.default_rng(seed)
    apos = np.asarray(apos, np.float64).reshape(-1, 3)
    r_lo = np.broadcast_to(np.asarray(r_lo, np.float64), (len(apos),))
    r_b = np.broadcast_to(np.asarray(r_b, np.float64), (len(apos),))
    r_min = np.maximum(0.4, 0.7 * r_lo)
    r = r_min[:, None] + (r_b - r_min)[:, None] * rng.random((len(apos), n_per_atom))
    v = rng.normal(size=(len(apos), n_per_atom, 3))
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    return (apos[:, None, :] + v * r[:, :, None]).reshape(-1, 3)


def fit_cores_from_samples(apos, xyz, E_samp, r_lo, r_b, F_samp=None, powers=CORE_POWERS,
                           weights=None, boltzmann_T=0.5, tikhonov=1e-6, bPrint=False, return_design=False, smooth=False):
    """Joint sparse least-squares fit of ALL atoms' compact cores from field samples.

    The ContactPME zero-core path (cpm_params_from_samples) puts the whole field
    on the mesh — the tricubic then needs h≲0.2 Å to resolve the repulsive wall
    and shows sharp-dot artifacts over atoms at larger h. This is the missing
    PAW piece: fit per-atom compact cores Σ_m c_im φ_m(r_i) (t=(r_b−r)/(r_b−r_lo),
    powers 2,4,8,16,32, zero at r_b) directly against arbitrary (E[,F]) samples
    (e.g. AFMulator.sample_fdbm), subtract them, and let a COARSE mesh (0.35–1.0 Å)
    carry only the smooth residual — the actual PME design.

    Joint sparse LSQ over the full (na·nm) coeff vector — atoms overlap within
    r_b so per-atom solves are not independent. Normal equations on a CSR design
    (na·5 ≈ 200 unknowns → dense solve is trivial). Boltzmann weights on E keep
    the deep repulsive interior (r≪r_lo, E≫) from dominating: the AFM-relevant
    range E≲1 eV decides the fit. Unknowns for un-sampled atoms are pinned to ~0
    by the Tikhonov term.

    Args:
        apos: (na,3) atom positions.
        xyz: (ns,3) sample positions (sample_core_shells + a context cloud).
        E_samp: (ns,) field energy at xyz.
        F_samp: optional (ns,3) field force — adds 3 derivative rows per sample
                (weight-normalized like fit_core_1d).
        r_lo, r_b: scalars or (na,) core bounds — must match SplitParams
                   (r_lo=R0−Δ_in, r_b=R0+Δ_b, span=Δ_in+Δ_b kernel ABI).
        weights: optional (ns,) row weights; default Boltzmann(E, T=boltzmann_T).
        return_design: return CSR energy/force matrices without solving, for the
                       joint weighted core/mesh fit (same compact basis).
        smooth: use five independent C2 modes instead of constraining raw powers.
    Returns:
        CoreFit (basis='raw', uniform-span) — ready for ContactPMEParams.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    apos = np.asarray(apos, np.float64).reshape(-1, 3)
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    E = np.asarray(E_samp, np.float64).ravel()
    na, nm, ns = len(apos), len(powers), len(xyz)
    assert E.shape == (ns,), f'E_samp shape {E.shape} vs ({ns},)'
    r_lo = np.broadcast_to(np.asarray(r_lo, np.float64), (na,)).copy()
    r_b = np.broadcast_to(np.asarray(r_b, np.float64), (na,)).copy()
    powers = np.asarray(powers, np.int64)
    # Previously: query_ball_point + a Python list for every sample/atom pair.
    # Sparse pairs are assembled in scipy, without a samples×atoms distance array.
    pairs = cKDTree(xyz).sparse_distance_matrix(cKDTree(apos), float(r_b.max()), output_type='coo_matrix')
    s_idx, a_idx = pairs.row, pairs.col
    dvec = xyz[s_idx] - apos[a_idx]
    r = np.linalg.norm(dvec, axis=1)
    keep = r < r_b[a_idx]
    s_idx, a_idx, r, dvec = s_idx[keep], a_idx[keep], r[keep], dvec[keep]
    phi, dphi = core_basis(r, r_lo[a_idx], r_b[a_idx], powers, smooth=smooth)
    col = (a_idx[:, None] * nm + np.arange(nm)).ravel()
    A = coo_matrix((phi.ravel(), (np.repeat(s_idx, nm), col)), shape=(ns, na * nm)).tocsr()
    Af = None
    if F_samp is not None:
        # Force rows: F_c = -dE/dx_c = -Σ_m c_m (dφ_m/dr) u_c, u=(xyz−apos)/r.
        # Rows 3s+c; per pair flat order (c, m) matching val_F (p, c, m).
        F = np.asarray(F_samp, np.float64).reshape(-1, 3)
        assert F.shape == (ns, 3)
        npair = len(s_idx)
        u = dvec / np.maximum(r, 1e-30)[:, None]                                    # (npair,3)
        row_F = np.repeat(s_idx[:, None] * 3 + np.arange(3)[None, :], nm, axis=1).ravel()  # (npair*3*nm,)
        col_F = np.tile(col.reshape(npair, nm), (1, 3)).ravel()
        val_F = (-dphi[:, None, :] * u[:, :, None]).ravel()                         # (p,c,m) → -dφ·u_c
        Af = coo_matrix((val_F, (row_F, col_F)), shape=(3 * ns, na * nm)).tocsr()
    if return_design:
        return A, Af
    w = np.asarray(weights, np.float64).ravel() if weights is not None else boltzmann_fit_weights(E, T=boltzmann_T)[0]
    Aw = A.multiply(w[:, None]).tocsr()
    G = (A.T @ Aw).toarray()
    rhs = A.T @ (w * E)
    if F_samp is not None:
        lam_F = 1.0 / max(float(np.std(F)), 1e-12)
        lam_E = 1.0 / max(float(np.std(E)), 1e-12)
        G *= lam_E ** 2
        rhs *= lam_E ** 2
        Awf = Af.multiply(np.repeat(w, 3)[:, None]).tocsr()
        G += (Af.T @ Awf).toarray() * lam_F ** 2
        rhs += Af.T @ (np.repeat(w, 3) * F.ravel()) * lam_F ** 2
    # ridge relative to mean diagonal — scale-invariant, damps the near-nullspace
    # (atoms whose core shells are barely sampled) that otherwise explodes coeffs
    G += (tikhonov * float(np.trace(G)) / (na * nm)) * np.eye(na * nm)
    c = np.linalg.solve(G, rhs)
    coeffs = c.reshape(na, nm)
    E_pred = A @ c
    res = E_pred - E
    if bPrint:
        wrmse = np.sqrt(np.sum(w * res**2) / np.sum(w))
        print(f'fit_cores_from_samples: na={na} modes={nm} unknowns={na*nm} '
              f'samples={ns} pairs={len(s_idx)} rmse_E={np.sqrt(np.mean(res**2)):.4e} '
              f'wrmse_E={wrmse:.4e} cond(G)={np.linalg.cond(G):.2e}')
    z = np.full(na, np.sqrt(np.mean(res ** 2)))
    return CoreFit(coeffs=coeffs, r_lo=r_lo, r_b=r_b, powers=powers, basis='smooth' if smooth else 'raw',
                   cond_raw=np.full(na, np.linalg.cond(G)), cond_hier=np.zeros(na),
                   train_rmse_E=z, train_rmse_F=np.zeros(na), held_rmse_E=np.full(na, np.nan),
                   held_rmse_F=np.full(na, np.nan), held_max_E=np.full(na, np.nan),
                   held_max_F=np.full(na, np.nan), worst_r=np.full(na, np.nan))


def _chebyshev_u(n):
    """Open nodes on (0,1), clustered at both ends. Shared by the GPU kernel."""
    k = np.arange(n, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / n))


def _p95_rows(v):
    """Linear percentile 95 along axis 1. Matches np.percentile(..., 95)."""
    s = np.sort(v, axis=1)
    n = s.shape[1]
    pos = 0.95 * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return (1.0 - frac) * s[:, lo] + frac * s[:, hi]


def paw_coeffs_batch(p: SplitParams):
    """Even-poly PAW coefficients for every atom. One batched 3×3 solve, no atom loop.

    Same a0 minimization as PMESplit._paw_even_coeffs.
    Returns a0, a2, a4, a6, r_b, each shape (na,).
    """
    from spammm.surfaces.PMESplit import combined_atom_potential
    r_b = np.atleast_1d(np.asarray(p.r_b, dtype=np.float64))
    Vc, Gc, Hc = combined_atom_potential(r_b, p)
    Vc, Gc, Hc = np.atleast_1d(Vc), np.atleast_1d(Gc), np.atleast_1d(Hc)
    D = r_b
    D2, D3, D4, D5, D6 = D*D, D**3, D**4, D**5, D**6
    A = np.empty((len(D), 3, 3), dtype=np.float64)
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = D2, D4, D6
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = 2*D, 4*D3, 6*D5
    A[:, 2, 0], A[:, 2, 1], A[:, 2, 2] = 2.0, 12*D2, 30*D4
    # rhs must be (na, 3, 1). A (na, 3) right-hand side is read as m=na, n=3.
    a_at_0 = np.linalg.solve(A, np.stack([Vc, Gc, Hc], axis=1)[..., None])[..., 0]
    a_hom = np.linalg.solve(A, np.tile(np.array([-1.0, 0.0, 0.0]), (len(D), 1))[..., None])[..., 0]
    s = np.sqrt(0.6)
    xs = 0.5 * D[:, None] * (1.0 + np.array([-s, 0.0, s]))
    ws = 0.5 * D[:, None] * np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
    xs2, xs4 = xs*xs, xs**4
    u = 2*a_at_0[:, 0:1] + 12*a_at_0[:, 1:2]*xs2 + 30*a_at_0[:, 2:3]*xs4
    v = 2*a_hom[:, 0:1] + 12*a_hom[:, 1:2]*xs2 + 30*a_hom[:, 2:3]*xs4
    denom = np.sum(ws * v * v, axis=1)
    a0 = np.where(denom < 1e-30, 0.0, -np.sum(ws * u * v, axis=1) / np.maximum(denom, 1e-30))
    a246 = np.linalg.solve(A, np.stack([Vc - a0, Gc, Hc], axis=1)[..., None])[..., 0]
    return a0, a246[:, 0], a246[:, 1], a246[:, 2], r_b


def fit_core_paw_grid(p: SplitParams, n_samp=CORE_FIT_NSAMP):
    """Float64 normal equations of the PAW core on the Chebyshev grid.

    Oracle for cs_fit_core_paw. Same weights as fit_core_1d (Boltzmann on v,
    separate E/F scales) but a fixed grid instead of the random shell sample.
    Returns coefficients (na, 5).
    """
    assert p.split_mode == 'paw', f'fit_core_paw_grid is the paw GPU path, got {p.split_mode}'
    a0, a2, a4, a6, r_b = paw_coeffs_batch(p)
    r_lo = np.atleast_1d(np.asarray(p.r_lo, dtype=np.float64))
    R0 = np.atleast_1d(np.asarray(p.R0, dtype=np.float64))
    E0 = np.atleast_1d(np.asarray(p.E0, dtype=np.float64))
    q = np.atleast_1d(np.asarray(p.q, dtype=np.float64))
    u = _chebyshev_u(n_samp)
    r = r_lo[:, None] + (r_b - r_lo)[:, None] * u[None, :]
    # Morse + Coulomb, broadcast over samples. combined_atom_potential wants a SplitParams
    # whose R0/E0/q line up with r. Evaluate per the closed form so one atom axis stays.
    K = -float(p.alpha)
    R2 = float(p.r_damp) ** 2
    e = np.exp(K * (r - R0[:, None]))
    e2 = e * e
    v = E0[:, None] * (e2 - 2.0 * e)
    dv = 2.0 * K * E0[:, None] * (e2 - e)
    cc = 14.3996448915 * q[:, None] * float(p.q_tip)
    s2 = r * r + R2
    inv_s = 1.0 / np.sqrt(s2)
    v = v + cc * inv_s
    dv = dv - cc * r * inv_s ** 3
    r2, r4, r6 = r*r, r**4, r**6
    P = a0[:, None] + a2[:, None]*r2 + a4[:, None]*r4 + a6[:, None]*r6
    dP = (2*a2[:, None])*r + (4*a4[:, None])*r2*r + (6*a6[:, None])*r4*r
    v_S = v - P
    dv_S = dv - dP
    D = (r_b - r_lo)[:, None]
    t = np.clip((r_b[:, None] - r) / D, 0.0, 1.0)
    powers = CORE_POWERS.astype(np.float64)
    phi = t[:, :, None] ** powers[None, None, :]
    dphi = -powers[None, None, :] * t[:, :, None] ** (powers - 1.0)[None, None, :] / D[:, :, None]
    E_shift = v.min(axis=1)
    p95 = _p95_rows(v)
    T = np.maximum((p95 - E_shift) / 3.0, 0.05)
    w = np.exp(-(v - E_shift[:, None]) / T[:, None])
    w /= np.maximum(w.max(axis=1, keepdims=True), 1e-16)
    E_scale = np.maximum(v_S.std(axis=1), 1e-12)
    F_scale = np.maximum(dv_S.std(axis=1), 1e-12)
    wE = w * (1.0 / E_scale[:, None]) ** 2
    wF = w * (1.0 / F_scale[:, None]) ** 2
    # G_mn = Σ wE phi_m phi_n + wF dphi_m dphi_n
    G = np.einsum('ak,akm,akn->amn', wE, phi, phi) + np.einsum('ak,akm,akn->amn', wF, dphi, dphi)
    rhs = np.einsum('ak,akm,ak->am', wE, phi, v_S) + np.einsum('ak,akm,ak->am', wF, dphi, dv_S)
    return np.linalg.solve(G, rhs[..., None])[..., 0]


# ── field-oracle PAW core fit (FDBM) ─────────────────────────────────────────
# The production split (fit_core_1d / fit_core_paw_grid / cs_fit_core_paw) needs the
# analytic per-atom oracle v_i(r) = Morse+Coulomb. For an arbitrary sampled field
# (FDBM) no analytic v exists — but the same construction applies with a radial
# oracle extracted from the field (see doc/Reports/ContactPME_Split_Rcut_Locality
# _2026-10-04.md: the split IS the damping; per-atom 1D fits, never a joint LSQ):
#   v_i(r) = robust median profile of E over samples where atom i is the nearest
#            atom and the direction is inside the probe cone (dz/r > cone_min)
#   P_i(r) = even soft poly C2-matched to v_i at r_b (paw_even_potential coeffs)
#   v_S,i  = v_i − P_i → t^p basis by weighted E+F normal equations (per atom)
#   mesh   = E − Σ_i v_S,i — smooth by construction inside the wall (bg + P_i).
# The oracle is defined only where samples exist (r ≳ 1.5 Å in the up cone);
# below the sampled radius v_S is flat-extended and the residual must be masked
# (residual taper) — the wall anisotropy there is fundamentally non-radial.

class FieldOracle:
    """Per-atom radial oracle + PAW split tables for one field.

    Attributes per atom i: r_prof/v_prof (median profile knots), paw=(a0..a6),
    r_min (first covered radius), r_b. eval_vs(xyz) sums v_S,i(r_i) over atoms.
    """
    __slots__ = ('r_prof', 'v_prof', 'paw', 'r_min', 'r_b', 'r_lo')

    def __init__(self, r_prof, v_prof, paw, r_min, r_b, r_lo):
        self.r_prof = r_prof          # (na, nk) profile knot radii (NaN-padded)
        self.v_prof = v_prof          # (na, nk) median E profile at r_prof
        self.paw = paw                # (na, 4) a0,a2,a4,a6
        self.r_min = r_min            # (na,) first covered radius (inf = no data)
        self.r_b = r_b                # (na,)
        self.r_lo = r_lo              # (na,)

    def vs_atom(self, i, r):
        """v_S,i(r) = v_i(r) − P_i(r) inside r_b, 0 outside; v_i flat-extended
        below r_min (deep region — mask residual there, it is un-sampled)."""
        r = np.asarray(r, np.float64)
        if not np.isfinite(self.r_min[i]):
            return np.zeros_like(r)
        a0, a2, a4, a6 = self.paw[i]
        r2 = r * r
        P = a0 + a2 * r2 + a4 * r2 * r2 + a6 * r2 * r2 * r2
        rp = self.r_prof[i]; vp = self.v_prof[i]
        ok = np.isfinite(rp)
        v = np.interp(r, rp[ok], vp[ok])          # flat-extend outside coverage
        return np.where(r < self.r_b[i], v - P, 0.0)


def _binned_median(idx, val, nbins):
    """Lower median per bin, vectorized (sort once). NaN for empty bins."""
    out = np.full(nbins, np.nan)
    order = np.argsort(idx, kind='stable')
    ii, vv = idx[order], val[order]
    cnt = np.bincount(ii, minlength=nbins)
    ends = np.cumsum(cnt)
    starts = ends - cnt
    mid = starts + (cnt - 1) // 2
    out[cnt > 0] = vv[mid[cnt > 0]]
    return out


def _spline_irls_profile(r, resid, n_int=8, k=3, iters=5):
    """Robust smooth 1D fit of resid(r): B-spline LSQ + Huber IRLS.
    The resid distribution is one-sided-skewed (occasional huge positives toward
    neighbor walls) — IRLS downweights them; the spline is smooth by construction
    so dv/dr is usable directly. Returns (BSpline, r_lo_covered, r_hi_covered)."""
    from scipy.interpolate import BSpline
    order = np.argsort(r)
    rm, y = r[order], resid[order]
    lo, hi = float(rm[0]), float(rm[-1])
    kn = np.concatenate([np.full(k, lo), np.linspace(lo, hi, n_int), np.full(k, hi)])
    B = BSpline.design_matrix(rm, kn, k).toarray()
    w = np.ones(len(rm))
    c = np.zeros(B.shape[1])
    for _ in range(iters):
        Bw = B * w[:, None]
        G = Bw.T @ B
        c = np.linalg.solve(G + 1e-10 * np.eye(len(c)), Bw.T @ y)
        res = y - B @ c
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        w = 1.0 / np.maximum(1.0, np.abs(res) / (1.345 * s))
    return BSpline(kn, c, k), lo, hi


def _profile_tail(r_prof, v_prof, r_b, span=0.8):
    """(Vc,Gc,Hc) at r_b from a quadratic fit of the profile's outer ~span Å."""
    m = r_prof > r_b - span
    if m.sum() < 3:
        m = np.arange(len(r_prof)) >= max(0, len(r_prof) - 3)
    a, b, c = np.polyfit(r_prof[m], v_prof[m], 2)
    return a * r_b * r_b + b * r_b + c, 2 * a * r_b + b, 2 * a


def fit_cores_paw_field(apos, xyz, E, sp, *, cone_min=0.3, dr=0.05, n_jacobi=20,
                        lam=0.5, n_warm=10, n_samp=CORE_FIT_NSAMP, powers=CORE_POWERS,
                        bPrint=False):
    """Per-atom PAW core fit from arbitrary field samples — field-oracle analog
    of fit_core_paw_grid / cs_fit_core_paw (which use the analytic Morse oracle).

    The field is NOT a sum of isolated radial potentials, so v_i(r) is extracted
    by DAMPED Jacobi decomposition: v_i ← (1−λ)v_i + λ·median-profile(E − Σ_{j≠i}ṽ_j)
    over samples where atom i is nearest and inside the probe cone (dz/r > cone_min).
    Undamped iteration oscillates ±~60 eV on the steep wall — λ≈0.5, ~20 iters
    converges to ~2% of the wall (verified on synthetic additive field).

    Args:
        apos: (na,3) atom positions (heavy atoms, matching sp order).
        xyz, E: (ns,3),(ns,) field samples covering the probe domain.
        sp: SplitParams — only r_lo/r_b radial bounds are used (v NOT analytic).
    Returns:
        (CoreFit, FieldOracle): CoreFit feeds ContactPMEParams; FieldOracle.eval
        gives Σ_i v_S,i for the mesh residual E − Σv_S (the pseudoization).
    """
    from scipy.spatial import cKDTree
    from spammm.surfaces.PMESplit import _paw_even_coeffs
    apos = np.asarray(apos, np.float64).reshape(-1, 3)
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    E = np.asarray(E, np.float64).ravel()
    na = len(apos)
    r_lo = np.broadcast_to(np.atleast_1d(np.asarray(sp.r_lo, np.float64)), (na,)).copy()
    r_b = np.broadcast_to(np.atleast_1d(np.asarray(sp.r_b, np.float64)), (na,)).copy()
    powers = np.asarray(powers, np.int64)
    rb_max = float(r_b.max())
    # ── sample↔atom pair list (within r_b), plus nearest + cone masks ──
    _, nearest = cKDTree(apos).query(xyz, k=1)
    nbrs = cKDTree(apos).query_ball_point(xyz, r=rb_max)
    pairs = [(s, ia) for s, ats in enumerate(nbrs) for ia in ats]
    s_idx = np.array([p[0] for p in pairs], np.int64)
    a_idx = np.array([p[1] for p in pairs], np.int64)
    dvec = xyz[s_idx] - apos[a_idx]
    r_pair = np.linalg.norm(dvec, axis=1)
    keep = r_pair < r_b[a_idx]
    s_idx, a_idx, r_pair, dvec = s_idx[keep], a_idx[keep], r_pair[keep], dvec[keep]
    # ── shared radial grid; Jacobi decomposition of E into per-atom walls ──
    rgrid = np.arange(0.5 * dr, rb_max + dr, dr)
    nb = len(rgrid)
    V = np.zeros((na, nb))
    cov = np.zeros((na, nb), dtype=bool)
    dz = xyz[:, 2][s_idx] - apos[a_idx, 2]
    cone_pair = dz / np.maximum(r_pair, 1e-9) > cone_min
    own = nearest[s_idx] == a_idx                               # sample nearest this atom
    splines = [None] * na
    for it in range(n_jacobi):
        # flat-extended profiles for residual evaluation (NaN outside coverage)
        Vext = V.copy()
        for i in range(na):
            c = cov[i]
            if c.any():
                Vext[i] = np.interp(rgrid, rgrid[c], V[i, c])    # fill gaps + flat-extend
        # vectorized per-pair interp of Vext on rgrid (grid uniform, dr spacing)
        fr = np.clip((r_pair - rgrid[0]) / dr, 0.0, nb - 1.001)
        i0 = fr.astype(np.int64); w1 = fr - i0
        v_pair = Vext[a_idx, i0] * (1.0 - w1) + Vext[a_idx, i0 + 1] * w1
        S_tot = np.bincount(s_idx, weights=v_pair, minlength=len(xyz))
        sel = own & cone_pair
        for i in range(na):
            m_pair = sel & (a_idx == i)
            if m_pair.sum() < 8:
                continue
            sm, rm = s_idx[m_pair], r_pair[m_pair]
            resid = E[sm] - S_tot[sm] + np.interp(rm, rgrid, Vext[i])
            ib = np.clip((rm - rgrid[0]) / dr, 0, nb - 1).astype(np.int64)
            cnew = np.bincount(ib, minlength=nb) > 0
            if it >= n_warm:
                spl, lo_p, hi_p = _spline_irls_profile(rm, resid)
                Vnew = np.full(nb, np.nan)
                Vnew[cnew] = spl(np.clip(rgrid[cnew], lo_p, hi_p))
                splines[i] = spl
            else:
                Vnew = _binned_median(ib, resid, nb)
            upd = cnew & np.isfinite(Vnew)
            # damped update — undamped Jacobi oscillates ±~60 eV on the steep wall
            V[i][upd] = (1.0 - lam) * V[i][upd] + lam * Vnew[upd]
            cov[i] |= cnew
    u = _chebyshev_u(n_samp)
    r_cheb = r_lo[:, None] + (r_b - r_lo)[:, None] * u[None, :]
    D = (r_b - r_lo)[:, None]
    t = np.clip((r_b[:, None] - r_cheb) / D, 0.0, 1.0)
    pf = powers.astype(np.float64)
    phi = t[:, :, None] ** pf[None, None, :]
    dphi = -pf[None, None, :] * t[:, :, None] ** (pf - 1.0)[None, None, :] / D[:, :, None]
    coeffs = np.zeros((na, len(powers)))
    orc = dict(r_prof=np.full((na, len(rgrid)), np.nan), v_prof=np.full((na, len(rgrid)), np.nan),
               paw=np.zeros((na, 4)), r_min=np.full(na, np.inf), r_b=r_b.copy(), r_lo=r_lo.copy())
    for i in range(na):
        spl = splines[i]
        cov_i = np.isfinite(V[i]) & cov[i]
        if spl is None or cov_i.sum() < 4:
            continue
        prof_r = rgrid[cov_i]
        lo_p, hi_p = float(prof_r[0]), float(prof_r[-1])
        # (Vc,Gc,Hc) at r_b from a quadratic LSQ over the outer ~0.8 A of COVERAGE —
        # raw spline derivatives at the edge knot are unstable (boundary interval),
        # and a wrong Hc makes P(r_b) != v(r_b) -> rim discontinuity in v_S at r_b.
        rt = np.linspace(hi_p - 0.85, hi_p - 0.05, 24)
        qa, qb_, qc = np.polyfit(rt, spl(rt), 2)
        Vc = qa * r_b[i]**2 + qb_ * r_b[i] + qc
        Gc = 2 * qa * r_b[i] + qb_; Hc = 2 * qa
        a0, a2, a4, a6, _ = _paw_even_coeffs(r_b[i], Vc, Gc, Hc, a0=None)
        in_r = (rgrid >= lo_p) & (rgrid <= hi_p)
        orc['r_prof'][i] = rgrid
        vp = np.empty(nb)
        vp[in_r] = spl(rgrid[in_r])
        vp[~in_r] = np.where(rgrid[~in_r] < lo_p, spl(lo_p), spl(hi_p))  # flat-extend
        orc['v_prof'][i] = vp
        orc['paw'][i] = (a0, a2, a4, a6)
        orc['r_min'][i] = lo_p
        r = np.clip(r_cheb[i], lo_p, hi_p)
        v = spl(r)
        dvr = spl(r, 1)
        r2 = r * r
        P = a0 + a2 * r2 + a4 * r2 * r2 + a6 * r2 * r2 * r2
        dP = 2 * a2 * r + 4 * a4 * r2 * r + 6 * a6 * r2 * r2 * r
        v_S = v - P
        dv_S = dvr - dP
        # same weighting as fit_core_paw_grid / cs_fit_core_paw
        E_shift = v.min()
        p95 = float(np.percentile(v, 95))
        T = max((p95 - E_shift) / 3.0, 0.05)
        w = np.exp(-(v - E_shift) / T); w /= max(w.max(), 1e-16)
        wE = w / max(v_S.std(), 1e-12) ** 2
        wF = w / max(dv_S.std(), 1e-12) ** 2
        G = np.einsum('k,km,kn->mn', wE, phi[i], phi[i]) + np.einsum('k,km,kn->mn', wF, dphi[i], dphi[i])
        rhs = np.einsum('k,km,k->m', wE, phi[i], v_S) + np.einsum('k,km,k->m', wF, dphi[i], dv_S)
        coeffs[i] = np.linalg.solve(G, rhs)
    if bPrint:
        n_ok = int(np.isfinite(orc['r_min']).sum())
        print(f'fit_cores_paw_field: {n_ok}/{na} atoms with in-cone profile '
              f'(jacobi={n_jacobi}); |c| max={np.abs(coeffs).max():.3f}')
    z = np.zeros(na)
    fit = CoreFit(coeffs=coeffs, r_lo=r_lo, r_b=r_b, powers=powers, basis='raw',
                  cond_raw=z, cond_hier=z, train_rmse_E=z, train_rmse_F=z,
                  held_rmse_E=z, held_rmse_F=z, held_max_E=z, held_max_F=z, worst_r=z)
    return fit, FieldOracle(orc['r_prof'], orc['v_prof'], orc['paw'], orc['r_min'], orc['r_b'], orc['r_lo'])


def eval_vs_oracle(xyz, apos, oracle: FieldOracle):
    """Σ_i v_S,i^oracle(r_i) at query points (for the mesh residual E − Σv_S).

    Vectorized (cKDTree pair list + bincount). Returns (vs, cov) where cov[q] =
    min_i (r_iq − r_min_i) over atoms within r_b — negative marks queries inside
    an atom's un-sampled deep region (mask/taper the residual there)."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    apos = np.asarray(apos, np.float64).reshape(-1, 3)
    nq = len(xyz)
    nbrs = cKDTree(apos).query_ball_point(xyz, r=float(np.max(oracle.r_b)))
    pairs = [(s, ia) for s, ats in enumerate(nbrs) for ia in ats]
    if len(pairs) == 0:
        return np.zeros(nq), np.full(nq, np.inf)
    s_idx = np.array([p[0] for p in pairs], np.int64)
    a_idx = np.array([p[1] for p in pairs], np.int64)
    dvec = xyz[s_idx] - apos[a_idx]
    r = np.linalg.norm(dvec, axis=1)
    rb = oracle.r_b[a_idx]
    keep = r < rb
    s_idx, a_idx, r, rb = s_idx[keep], a_idx[keep], r[keep], rb[keep]
    a0, a2, a4, a6 = oracle.paw[a_idx].T
    r2 = r * r
    P = a0 + a2 * r2 + a4 * r2 * r2 + a6 * r2 * r2 * r2
    v = np.empty(len(r))
    for i in np.unique(a_idx):                                 # per-atom interp
        mm = a_idx == i
        rp, vp = oracle.r_prof[i], oracle.v_prof[i]
        ok = np.isfinite(rp)
        v[mm] = np.interp(r[mm], rp[ok], vp[ok]) if ok.any() else 0.0
    vs_pair = np.where(np.isfinite(oracle.r_min[a_idx]), v - P, 0.0)
    vs = np.bincount(s_idx, weights=vs_pair, minlength=nq)
    cov_d = r - np.where(np.isfinite(oracle.r_min[a_idx]), oracle.r_min[a_idx], np.inf)
    cov = np.full(nq, np.inf)
    np.minimum.at(cov, s_idx, cov_d)
    return vs, cov


def inpaint_residual(resid_nodes, ns, mask, n_iter=400):
    """Harmonic (Laplace) fill of a residual field inside masked grid cells.

    Inside an atom's un-sampled deep ball (r < r_min of the field oracle) the
    residual E - Σv_S still contains the keV wall — B-spline prefiltering then
    rings ±tens of eV into the AFM domain. The PAW-consistent fix is to give
    the mesh a SMOOTH continuation there (like Σv_L = P_i + bg in production):
    solve ∇²g = 0 on the mask with Dirichlet BC = residual on the covered
    boundary (Jacobi iteration, vectorized slices). Masked cells at the grid
    border are excluded (no wrap-around).
    Returns flattened (nx*ny*nz,) float64 residual with the mask inpainted."""
    g = np.asarray(resid_nodes, np.float64).reshape(tuple(ns)).copy()
    m = np.asarray(mask, bool).reshape(tuple(ns))
    m_in = m[1:-1, 1:-1, 1:-1]
    if not m_in.any():
        return g.ravel()
    gi = g[1:-1, 1:-1, 1:-1]
    gi[m_in] = 0.0                                               # seed
    for _ in range(n_iter):
        g6 = (g[:-2, 1:-1, 1:-1] + g[2:, 1:-1, 1:-1] + g[1:-1, :-2, 1:-1] +
              g[1:-1, 2:, 1:-1] + g[1:-1, 1:-1, :-2] + g[1:-1, 1:-1, 2:]) * (1.0 / 6.0)
        gi[m_in] = g6[m_in]
    return g.ravel()


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
                    phi, dphi = core_basis(np.array([r]), r_lo_i, r_b_i, fit.powers, smooth=fit.basis == 'smooth')
                    c = fit.coeffs[ia]
                    E[iq] += float(phi[0] @ c)
                    if r > 1e-30:
                        F[iq] -= float(dphi[0] @ c) * dp / r

    return E, F


def eval_core_fast(queries, atom_pos, fit: CoreFit):
    """Vectorized eval_core — same math, cKDTree pair list + bincount instead of the
    per-query Python loop. Use for mesh-node residual evaluation (~1e5 queries)."""
    from scipy.spatial import cKDTree
    queries = np.asarray(queries, np.float64).reshape(-1, 3)
    atom_pos = np.asarray(atom_pos, np.float64).reshape(-1, 3)
    nq = len(queries)
    r_max = float(np.max(fit.r_b))
    pairs = cKDTree(queries).sparse_distance_matrix(cKDTree(atom_pos), r_max, output_type='coo_matrix')
    if pairs.nnz == 0:
        return np.zeros(nq), np.zeros((nq, 3))
    s_idx, a_idx = pairs.row, pairs.col
    dvec = queries[s_idx] - atom_pos[a_idx]
    r = np.linalg.norm(dvec, axis=1)
    keep = r < np.asarray(fit.r_b)[a_idx]
    s_idx, a_idx, r, dvec = s_idx[keep], a_idx[keep], r[keep], dvec[keep]
    phi, dphi = core_basis(r, np.asarray(fit.r_lo)[a_idx], np.asarray(fit.r_b)[a_idx], fit.powers, smooth=fit.basis == 'smooth')
    c = np.asarray(fit.coeffs)[a_idx]                            # (npair, nm)
    E = np.bincount(s_idx, weights=np.einsum('ij,ij->i', phi, c), minlength=nq)
    u = dvec / np.maximum(r, 1e-30)[:, None]
    f_mag = -np.einsum('ij,ij->i', dphi, c)                      # F = -dE/dr * u
    F = np.stack([np.bincount(s_idx, weights=f_mag * u[:, k], minlength=nq) for k in range(3)], axis=1)
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
            phi, dphi = core_basis(np.array([r]), r_lo_i, r_b_i, fit.powers, smooth=fit.basis == 'smooth')
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
                phi, dphi = core_basis(np.array([r]), r_lo_i, r_b_i, fit.powers, smooth=fit.basis == 'smooth')
                c = fit.coeffs[ia]
                E[iq] += float(phi[0] @ c)
                if r > 1e-30:
                    F[iq] -= float(dphi[0] @ c) * dp / r
    return E, F
