"""
PMESplit.py — Atomwise soft-core split for the contact_pme particle-mesh backend.

Physics oracle: getMorsePLQH / cs_brute_plqh_points (kernels/Forces.cl:235-249,
kernels/contact_surface.cl:284-330). Contract version 2.

The combined radial potential per atom i is:
  v_i(r) = E0_i [exp(2K(r-R0_i)) - 2 exp(K(r-R0_i))]
         + COULOMB_CONST * q_i * q_tip / sqrt(r^2 + R2damp)
where K = -alpha < 0 (global tip property), R2damp = r_damp^2.

Force: F_i = -dv_i/dr * (r-R_i)/r   (probe force, dp = pos - R_i)

The soft-core split maps r → rho_i(r) via a quintic Hermite polynomial on
[r_lo_i, r_cut], making v_L = v(rho) C² and flat at the inner boundary while
v_S = v - v_L and its first two derivatives vanish at r_cut.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

COULOMB_CONST = 14.3996448915  # [eV*Ang/e^2], matches kernels/common.cl


@dataclass
class SplitParams:
    """Per-atom and global tip parameters for the combined radial potential and soft-core split."""
    R0: np.ndarray       # (na,) Morse equilibrium radius per atom
    E0: np.ndarray       # (na,) Morse well depth per atom
    q: np.ndarray        # (na,) atomic charge per atom (from mol.qs, NOT ElementTypes.dat)
    alpha: float         # tip Morse stiffness (>0); K = -alpha
    q_tip: float         # tip charge
    r_damp: float        # Coulomb damping radius
    r_cut: float = 6.0   # outer split radius [Ang]

    @property
    def K(self) -> float: return -self.alpha

    @property
    def R2damp(self) -> float: return self.r_damp ** 2

    @property
    def r_lo(self) -> np.ndarray: return self.R0 - 0.5

    @property
    def na(self) -> int: return len(np.atleast_1d(self.R0))


# ── combined radial potential ──────────────────────────────────────────────

def combined_atom_potential(r, p: SplitParams):
    """v(r), dv/dr, d²v/dr² for the combined Morse + damped-Coulomb potential.

    Vectorized: r is broadcastable with p.R0, p.E0, p.q.
    Matches getMorsePLQH with PLQH=(1,1,q_tip,0) exactly.
    """
    K = -p.alpha
    R2damp = p.r_damp ** 2
    R0, E0, q = p.R0, p.E0, p.q
    r = np.asarray(r, dtype=np.float64)
    # Morse: e = exp(K*(r-R0)), v = E0*(e^2 - 2*e)
    e = np.exp(K * (r - R0))
    e2 = e * e
    v_morse = E0 * (e2 - 2.0 * e)
    dv_morse = 2.0 * K * E0 * (e2 - e)
    d2v_morse = 2.0 * K * K * E0 * (2.0 * e2 - e)
    # Damped Coulomb: v = cc / sqrt(r^2 + R2damp)
    s2 = r * r + R2damp
    inv_s = 1.0 / np.sqrt(s2)
    inv_s3 = inv_s ** 3
    inv_s5 = inv_s3 * inv_s * inv_s
    cc = COULOMB_CONST * q * p.q_tip
    v_coul = cc * inv_s
    dv_coul = -cc * r * inv_s3
    d2v_coul = -cc * (R2damp - 2.0 * r * r) * inv_s5
    v = v_morse + v_coul
    dvdr = dv_morse + dv_coul
    d2vdr2 = d2v_morse + d2v_coul
    return v, dvdr, d2vdr2


# ── quintic Hermite softened coordinate ────────────────────────────────────

def _quintic_f(u):
    """f(u) = 6u³ - 8u⁴ + 3u⁵ and derivatives w.r.t. u. u in [0,1]."""
    u2 = u * u; u3 = u2 * u; u4 = u3 * u; u5 = u4 * u
    f = 6.0 * u3 - 8.0 * u4 + 3.0 * u5
    df = 18.0 * u2 - 32.0 * u3 + 15.0 * u4
    d2f = 36.0 * u - 96.0 * u2 + 60.0 * u3
    return f, df, d2f


def softened_rho(r, r_lo, r_cut):
    """Quintic Hermite softened radial coordinate rho(r) with C² joins.

    - rho(r) = r_lo for r <= r_lo
    - On [r_lo, r_cut]: quintic Hermite satisfying rho(r_lo)=r_lo, rho'(r_lo)=rho''(r_lo)=0,
      rho(r_cut)=r_cut, rho'(r_cut)=1, rho''(r_cut)=0
    - rho(r) = r for r >= r_cut

    Returns (rho, drho/dr, d²rho/dr²). Vectorized; r_lo and r_cut broadcast with r.
    """
    r = np.asarray(r, dtype=np.float64)
    r_lo = np.asarray(r_lo, dtype=np.float64)
    r_cut = np.asarray(r_cut, dtype=np.float64)
    D = r_cut - r_lo
    u = np.clip((r - r_lo) / D, 0.0, 1.0)
    f, df, d2f = _quintic_f(u)
    rho_inner = r_lo + D * f
    drho_inner = df          # drho/dr = D*df/du * du/dr = D*df * (1/D) = df
    d2rho_inner = d2f / D    # d²rho/dr² = d²f/du² * (du/dr)² = d2f / D² ... but drho/dr=df(u),
    # so d²rho/dr² = d(df)/du * du/dr = d2f * (1/D) = d2f / D
    rho = np.where(r <= r_lo, r_lo, np.where(r >= r_cut, r, rho_inner))
    drho = np.where(r <= r_lo, 0.0, np.where(r >= r_cut, 1.0, drho_inner))
    d2rho = np.where(r <= r_lo, 0.0, np.where(r >= r_cut, 0.0, d2rho_inner))
    return rho, drho, d2rho


# ── soft-core split ────────────────────────────────────────────────────────

def soft_core_split(r, p: SplitParams):
    """Compute v, v_L, v_S and their first/second radial derivatives.

    v_L(r) = v(rho(r)), v_S(r) = v(r) - v_L(r) on domain r >= r_lo.
    For r >= r_cut: v_L = v, v_S = 0 (all derivatives vanish).
    For r <= r_lo: v_L = v(r_lo) (constant), v_S undefined (domain violation).

    Returns dict: v, v_L, v_S, dvdr, dv_L_dr, dv_S_dr, d2v, d2v_L, d2v_S.
    """
    r = np.asarray(r, dtype=np.float64)
    r_lo = p.r_lo
    # Direct potential at r
    v, dvdr, d2vdr2 = combined_atom_potential(r, p)
    # Softened coordinate
    rho, drho, d2rho = softened_rho(r, r_lo, p.r_cut)
    # Potential at rho(r)
    v_rho, dv_rho, d2v_rho = combined_atom_potential(rho, p)
    # Chain rule: v_L = v(rho(r))
    v_L = v_rho
    dv_L_dr = dv_rho * drho
    d2v_L_dr2 = d2v_rho * drho ** 2 + dv_rho * d2rho
    # Split
    v_S = v - v_L
    dv_S_dr = dvdr - dv_L_dr
    d2v_S_dr2 = d2vdr2 - d2v_L_dr2
    return dict(v=v, v_L=v_L, v_S=v_S, dvdr=dvdr, dv_L_dr=dv_L_dr, dv_S_dr=dv_S_dr,
                d2v=d2vdr2, d2v_L=d2v_L_dr2, d2v_S=d2v_S_dr2)


# ── domain violation ───────────────────────────────────────────────────────

def domain_violation_mask(r, p: SplitParams):
    """Boolean mask: True where r < r_lo_i (model-domain violation)."""
    r = np.asarray(r, dtype=np.float64)
    return r < p.r_lo


def check_domain(r, p: SplitParams):
    """Detect domain violations. Returns (violation_mask, min_r, offending_indices).

    Raises ValueError if any violation is found (fail-loud). Use domain_violation_mask
    for non-raising checks.
    """
    mask = domain_violation_mask(r, p)
    if np.any(mask):
        idx = np.where(mask)[0]
        min_r = float(np.min(r[mask]))
        raise ValueError(f"Domain violation: r={min_r:.6f} < r_lo at indices {idx.tolist()}")
    return mask, None, None


# ── r_cut sweep ────────────────────────────────────────────────────────────

def r_cut_candidates(p: SplitParams, candidates=(4.0, 5.0, 6.0)):
    """Evaluate r_cut candidates. Reject any with r_cut <= max_i(r_lo_i).

    Returns (valid, rejected, r_lo_max). valid is sorted ascending (smallest first
    for selection by downstream gates).
    """
    r_lo_max = float(np.max(p.r_lo))
    valid = sorted([rc for rc in candidates if rc > r_lo_max])
    rejected = [rc for rc in candidates if rc <= r_lo_max]
    return valid, rejected, r_lo_max


# ── force from energy (paired E/F) ─────────────────────────────────────────

def eval_atom_ef(queries, atom_pos, p: SplitParams):
    """Energy and force on probe at query positions from a single atom.

    F = -dv/dr * (query - atom_pos) / r. Returns (E, F) with E shape (nq,) and F (nq, 3).
    p must have scalar (single-atom) R0, E0, q.
    """
    dp = np.asarray(queries, dtype=np.float64).reshape(-1, 3) - np.asarray(atom_pos, dtype=np.float64)
    r = np.linalg.norm(dp, axis=-1)
    v, dvdr, _ = combined_atom_potential(r, p)
    r_safe = np.where(r > 1e-30, r, 1.0)
    F = (-dvdr / r_safe)[:, None] * dp
    F = np.where((r > 1e-30)[:, None], F, 0.0)
    return v, F
