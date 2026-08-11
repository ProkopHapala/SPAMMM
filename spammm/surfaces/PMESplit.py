"""
PMESplit.py — Atomwise long/short split for the contact_pme particle-mesh backend.

Physics oracle: getMorsePLQH / cs_brute_plqh_points (kernels/Forces.cl:235-249,
kernels/contact_surface.cl:284-330).

The combined radial potential per atom i is:
  v_i(r) = E0_i [exp(2K(r-R0_i)) - 2 exp(K(r-R0_i))]
         + COULOMB_CONST * q_i * q_tip / sqrt(r^2 + R2damp)
where K = -alpha < 0 (global tip property), R2damp = r_damp^2.

Force: F_i = -dv_i/dr * (r-R_i)/r   (probe force, dp = pos - R_i)

Split modes (1D / Python experiment — choose by smoothness metrics before OpenCL):
  'paw'      Preferred: even polynomial in r (smooth at r=0 as 3D radial field)
             P(r)=a0+a2 r²+a4 r⁴+a6 r⁶, C²-matched to v at r_b=R0+Δ_b.
             Free a0 chosen to minimize ∫(P'')² on [0,r_b]. No W-blend.
  'hermite'  Soft poly in s=r-r_lo: P'=P''=0 at r_lo, C² match at r_b.
             Weaker at origin for mesh rasterization than 'paw'.
  'plateau'  v_L = C + W(v-C). WARNING: dv_L/dr contains W'(v-C), which
             invents force bumps in the switch interval (not true damping).
  'rho'      legacy coordinate map v_L=v(rho(r)); often max rho'>1 (steepening).
  'softcore' hermite soft poly with a0 seeded by v(sqrt(r_lo^2+a^2)).

Geometry (R0-relative): r_lo=R0-Δ_in, r_a=R0+Δ_a, r_b=R0+Δ_b.
  Default Δ_in=1.0 so r_lo sits below CLI AFM closest approach (~2.7 Å with amp=1).
  For 'paw', tip/core domain still uses r_lo; soft long-range P is valid for r≥0.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

COULOMB_CONST = 14.3996448915  # [eV*Ang/e^2], matches kernels/common.cl

# Default R0-relative geometry (Å)
DEFAULT_DELTA_IN = 1.0   # r_lo=R0-Δ_in; must sit below AFM closest approach (~2.7Å for CLI amp=1)
DEFAULT_DELTA_A = 0.5
DEFAULT_DELTA_B = 2.0
DEFAULT_SOFTCORE_A = 1.0

# Modes with compact outer radius r_b = R0+Δ_b (not legacy r_cut)
_COMPACT_SPLIT_MODES = ('paw', 'hermite', 'plateau', 'softcore')


@dataclass
class SplitParams:
    """Per-atom and global tip parameters for the combined radial potential and split."""
    R0: np.ndarray
    E0: np.ndarray
    q: np.ndarray
    alpha: float
    q_tip: float
    r_damp: float
    split_mode: str = 'paw'  # 'paw' | 'hermite' | 'plateau' | 'rho' | 'softcore'
    delta_in: float = DEFAULT_DELTA_IN
    delta_a: float = DEFAULT_DELTA_A
    delta_b: float = DEFAULT_DELTA_B
    r_cut: float = 6.0
    softcore_a: float = DEFAULT_SOFTCORE_A

    @property
    def K(self) -> float: return -self.alpha

    @property
    def R2damp(self) -> float: return self.r_damp ** 2

    @property
    def r_lo(self) -> np.ndarray:
        return np.asarray(self.R0, dtype=np.float64) - float(self.delta_in)

    @property
    def r_a(self) -> np.ndarray:
        return np.asarray(self.R0, dtype=np.float64) + float(self.delta_a)

    @property
    def r_b(self) -> np.ndarray:
        return np.asarray(self.R0, dtype=np.float64) + float(self.delta_b)

    @property
    def r_core_max(self) -> float:
        if self.split_mode in _COMPACT_SPLIT_MODES:
            return float(np.max(self.r_b))
        return float(self.r_cut)

    @property
    def na(self) -> int: return len(np.atleast_1d(self.R0))

    def with_atom(self, i: int) -> 'SplitParams':
        R0 = np.atleast_1d(self.R0); E0 = np.atleast_1d(self.E0); q = np.atleast_1d(self.q)
        return SplitParams(R0=np.array([float(R0[i])]), E0=np.array([float(E0[i])]),
                           q=np.array([float(q[i])]), alpha=self.alpha, q_tip=self.q_tip,
                           r_damp=self.r_damp, split_mode=self.split_mode,
                           delta_in=self.delta_in, delta_a=self.delta_a, delta_b=self.delta_b,
                           r_cut=self.r_cut, softcore_a=self.softcore_a)


def combined_atom_potential(r, p: SplitParams):
    """v(r), dv/dr, d²v/dr² — Morse + damped Coulomb. Matches getMorsePLQH PLQH=(1,1,q_tip,0)."""
    K = -p.alpha
    R2damp = p.r_damp ** 2
    R0, E0, q = p.R0, p.E0, p.q
    r = np.asarray(r, dtype=np.float64)
    e = np.exp(K * (r - R0))
    e2 = e * e
    v_morse = E0 * (e2 - 2.0 * e)
    dv_morse = 2.0 * K * E0 * (e2 - e)
    d2v_morse = 2.0 * K * K * E0 * (2.0 * e2 - e)
    s2 = r * r + R2damp
    inv_s = 1.0 / np.sqrt(s2)
    inv_s3 = inv_s ** 3
    inv_s5 = inv_s3 * inv_s * inv_s
    cc = COULOMB_CONST * q * p.q_tip
    v_coul = cc * inv_s
    dv_coul = -cc * r * inv_s3
    d2v_coul = -cc * (R2damp - 2.0 * r * r) * inv_s5
    return v_morse + v_coul, dv_morse + dv_coul, d2v_morse + d2v_coul


def _smoothstep_S(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x
    S = 10.0 * x3 - 15.0 * x4 + 6.0 * x5
    dS = 30.0 * x2 - 60.0 * x3 + 30.0 * x4
    d2S = 60.0 * x - 180.0 * x2 + 120.0 * x3
    return S, dS, d2S


def activation_W(r, r_a, r_b):
    """W(r): 0 below r_a, 1 above r_b, C² quintic in between. Used by plateau only."""
    r = np.asarray(r, dtype=np.float64)
    r_a = np.asarray(r_a, dtype=np.float64)
    r_b = np.asarray(r_b, dtype=np.float64)
    D = r_b - r_a
    x = (r - r_a) / D
    S, dS, d2S = _smoothstep_S(x)
    W = np.where(r <= r_a, 0.0, np.where(r >= r_b, 1.0, S))
    dW = np.where((r > r_a) & (r < r_b), dS / D, 0.0)
    d2W = np.where((r > r_a) & (r < r_b), d2S / (D * D), 0.0)
    return W, dW, d2W


def _quintic_f(u):
    u2 = u * u; u3 = u2 * u; u4 = u3 * u; u5 = u4 * u
    f = 6.0 * u3 - 8.0 * u4 + 3.0 * u5
    df = 18.0 * u2 - 32.0 * u3 + 15.0 * u4
    d2f = 36.0 * u - 96.0 * u2 + 60.0 * u3
    return f, df, d2f


def softened_rho(r, r_lo, r_cut):
    """Legacy quintic Hermite softened coordinate rho(r)."""
    r = np.asarray(r, dtype=np.float64)
    r_lo = np.asarray(r_lo, dtype=np.float64)
    r_cut = np.asarray(r_cut, dtype=np.float64)
    D = r_cut - r_lo
    u = np.clip((r - r_lo) / D, 0.0, 1.0)
    f, df, d2f = _quintic_f(u)
    rho_inner = r_lo + D * f
    rho = np.where(r <= r_lo, r_lo, np.where(r >= r_cut, r, rho_inner))
    drho = np.where(r <= r_lo, 0.0, np.where(r >= r_cut, 1.0, df))
    d2rho = np.where(r <= r_lo, 0.0, np.where(r >= r_cut, 0.0, d2f / D))
    return rho, drho, d2rho


# ── PAW even-polynomial soft replacement (smooth at r=0) ───────────────────

def _minimize_1d(roughness, lo, hi, n_grid=81, n_refine=30):
    """Coarse grid + golden-section refine for scalar roughness(a0)."""
    grid = np.linspace(lo, hi, n_grid)
    R = np.array([roughness(a) for a in grid])
    i0 = int(np.argmin(R))
    a, b = grid[max(0, i0 - 1)], grid[min(len(grid) - 1, i0 + 1)]
    for _ in range(n_refine):
        m1 = a + 0.382 * (b - a); m2 = a + 0.618 * (b - a)
        if roughness(m1) < roughness(m2):
            b = m2
        else:
            a = m1
    return 0.5 * (a + b)


def _paw_even_coeffs(r_b, Vc, Gc, Hc, a0=None):
    """P(r)=a0+a2 r²+a4 r⁴+a6 r⁶; C²-match (Vc,Gc,Hc) at r_b; even → smooth at 0.

    If a0 is None, choose a0 minimizing ∫_0^{r_b} (P'')² dr (least curvature).
    """
    D = float(r_b)
    assert D > 0.0, f'paw r_b must be > 0, got {D}'
    A = np.array([
        [D**2,   D**4,    D**6],
        [2*D,    4*D**3,  6*D**5],
        [2.0,    12*D**2, 30*D**4],
    ], dtype=np.float64)

    def coeffs_for_a0(a0_val):
        return np.linalg.solve(A, np.array([Vc - a0_val, Gc, Hc], dtype=np.float64))

    if a0 is None:
        a_at_0 = coeffs_for_a0(0.0)
        a_hom = np.linalg.solve(A, np.array([-1.0, 0.0, 0.0]))
        # 3-point Gauss–Legendre on [0, D]
        xs = 0.5 * D * (1.0 + np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]))
        ws = 0.5 * D * np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

        def roughness(a0_val):
            a2, a4, a6 = a_at_0 + a0_val * a_hom
            p2 = 2 * a2 + 12 * a4 * xs**2 + 30 * a6 * xs**4
            return float(np.sum(ws * p2 * p2))

        span = max(abs(Vc), abs(Hc) * D * D, abs(Gc) * D, 1e-6)
        a0 = _minimize_1d(roughness, Vc - 20 * span, Vc + 5 * span)
    a2, a4, a6 = coeffs_for_a0(a0)
    return float(a0), float(a2), float(a4), float(a6), D


def paw_even_potential(r, r_b, Vc, Gc, Hc, a0=None):
    """Evaluate even soft poly P(r) and derivatives. Returns (P, dP, d2P, meta)."""
    r = np.asarray(r, dtype=np.float64)
    a0, a2, a4, a6, D = _paw_even_coeffs(r_b, Vc, Gc, Hc, a0=a0)
    r2 = r * r; r4 = r2 * r2; r6 = r4 * r2
    P = a0 + a2 * r2 + a4 * r4 + a6 * r6
    dP = 2 * a2 * r + 4 * a4 * r2 * r + 6 * a6 * r4 * r
    d2P = 2 * a2 + 12 * a4 * r2 + 30 * a6 * r4
    return P, dP, d2P, dict(a0=a0, a2=a2, a4=a4, a6=a6, D=D)


# ── Hermite soft poly in (r-r_lo) — weaker at origin than paw ───────────────

def _hermite_soft_coeffs(r_lo, r_b, Vc, Gc, Hc, a0=None):
    """P(s)=a0+a3 s³+a4 s⁴+a5 s⁵, s=r-r_lo; P'=P''=0 at r_lo; match (Vc,Gc,Hc) at r_b.

    If a0 is None, choose a0 minimizing ∫(P'')² on [r_lo,r_b] (least curvature).
    """
    D = float(r_b - r_lo)
    A = np.array([
        [D**3,   D**4,    D**5],
        [3*D**2, 4*D**3,  5*D**4],
        [6*D,    12*D**2, 20*D**3],
    ], dtype=np.float64)

    def coeffs_for_a0(a0_val):
        return np.linalg.solve(A, np.array([Vc - a0_val, Gc, Hc], dtype=np.float64))

    if a0 is None:
        a_at_0 = coeffs_for_a0(0.0)
        a_hom = np.linalg.solve(A, np.array([-1.0, 0.0, 0.0]))
        xs = 0.5 * D * (1.0 + np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]))
        ws = 0.5 * D * np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

        def roughness(a0_val):
            a3, a4, a5 = a_at_0 + a0_val * a_hom
            p2 = 6 * a3 * xs + 12 * a4 * xs**2 + 20 * a5 * xs**3
            return float(np.sum(ws * p2 * p2))

        span = max(abs(Vc), abs(Hc) * D * D, 1e-6)
        a0 = _minimize_1d(roughness, Vc - 20 * span, Vc + 5 * span)
    a3, a4, a5 = coeffs_for_a0(a0)
    return float(a0), float(a3), float(a4), float(a5), D


def hermite_soft_potential(r, r_lo, r_b, Vc, Gc, Hc, a0=None):
    """Evaluate soft polynomial P(r) and derivatives. Returns (P, dP, d2P, meta)."""
    r = np.asarray(r, dtype=np.float64)
    a0, a3, a4, a5, D = _hermite_soft_coeffs(r_lo, r_b, Vc, Gc, Hc, a0=a0)
    s = r - r_lo
    P = a0 + a3 * s**3 + a4 * s**4 + a5 * s**5
    dP = 3 * a3 * s**2 + 4 * a4 * s**3 + 5 * a5 * s**4
    d2P = 6 * a3 * s + 12 * a4 * s**2 + 20 * a5 * s**3
    return P, dP, d2P, dict(a0=a0, a3=a3, a4=a4, a5=a5, D=D)


def _soft_core_split_plateau(r, p: SplitParams):
    """v_L=C+W(v-C). CAUTION: dv_L/dr = W'(v-C)+W v' invents force bumps."""
    r = np.asarray(r, dtype=np.float64)
    r_a, r_b = p.r_a, p.r_b
    v, dvdr, d2vdr2 = combined_atom_potential(r, p)
    C, _, _ = combined_atom_potential(r_a, p)
    W, dW, d2W = activation_W(r, r_a, r_b)
    vC = v - C
    v_L = C + W * vC
    dv_L_dr = dW * vC + W * dvdr
    d2v_L_dr2 = d2W * vC + 2.0 * dW * dvdr + W * d2vdr2
    return dict(v=v, v_L=v_L, v_S=(1.0 - W) * vC, dvdr=dvdr, dv_L_dr=dv_L_dr,
                dv_S_dr=dvdr - dv_L_dr, d2v=d2vdr2, d2v_L=d2v_L_dr2,
                d2v_S=d2vdr2 - d2v_L_dr2, C=C, W=W)


def _soft_core_split_rho(r, p: SplitParams):
    r = np.asarray(r, dtype=np.float64)
    v, dvdr, d2vdr2 = combined_atom_potential(r, p)
    rho, drho, d2rho = softened_rho(r, p.r_lo, p.r_cut)
    v_rho, dv_rho, d2v_rho = combined_atom_potential(rho, p)
    v_L = v_rho
    dv_L_dr = dv_rho * drho
    d2v_L_dr2 = d2v_rho * drho ** 2 + dv_rho * d2rho
    return dict(v=v, v_L=v_L, v_S=v - v_L, dvdr=dvdr, dv_L_dr=dv_L_dr,
                dv_S_dr=dvdr - dv_L_dr, d2v=d2vdr2, d2v_L=d2v_L_dr2,
                d2v_S=d2vdr2 - d2v_L_dr2)


def _soft_core_split_paw(r, p: SplitParams, a0=None):
    """Even-poly PAW: P=a0+a2 r²+… smooth at r=0, C² join at r_b, no W-blend."""
    r = np.asarray(r, dtype=np.float64)
    r_b = float(np.asarray(p.r_b).reshape(-1)[0])
    v, dvdr, d2vdr2 = combined_atom_potential(r, p)
    Vb, Gb, Hb = combined_atom_potential(np.array([r_b]), p)
    P, dP, d2P, meta = paw_even_potential(r, r_b, float(Vb[0]), float(Gb[0]), float(Hb[0]), a0=a0)
    inside = r < r_b
    v_L = np.where(inside, P, v)
    dv_L = np.where(inside, dP, dvdr)
    d2v_L = np.where(inside, d2P, d2vdr2)
    return dict(v=v, v_L=v_L, v_S=v - v_L, dvdr=dvdr, dv_L_dr=dv_L, dv_S_dr=dvdr - dv_L,
                d2v=d2vdr2, d2v_L=d2v_L, d2v_S=d2vdr2 - d2v_L, paw_meta=meta)


def _soft_core_split_hermite(r, p: SplitParams, a0=None):
    """PAW-like soft poly in (r-r_lo): no W-blend, C² join at r_b, flat at r_lo."""
    r = np.asarray(r, dtype=np.float64)
    r_lo = float(np.asarray(p.r_lo).reshape(-1)[0])
    r_b = float(np.asarray(p.r_b).reshape(-1)[0])
    v, dvdr, d2vdr2 = combined_atom_potential(r, p)
    Vb, Gb, Hb = combined_atom_potential(np.array([r_b]), p)
    P, dP, d2P, meta = hermite_soft_potential(r, r_lo, r_b, float(Vb[0]), float(Gb[0]), float(Hb[0]), a0=a0)
    inside = r < r_b
    v_L = np.where(inside, P, v)
    dv_L = np.where(inside, dP, dvdr)
    d2v_L = np.where(inside, d2P, d2vdr2)
    return dict(v=v, v_L=v_L, v_S=v - v_L, dvdr=dvdr, dv_L_dr=dv_L, dv_S_dr=dvdr - dv_L,
                d2v=d2vdr2, d2v_L=d2v_L, d2v_S=d2vdr2 - d2v_L, hermite_meta=meta)


def _soft_core_split_softcore(r, p: SplitParams):
    """Hermite soft poly with a0 seeded by softcore distance at r_lo."""
    r_lo = float(np.asarray(p.r_lo).reshape(-1)[0])
    a = float(p.softcore_a)
    rho_lo = np.sqrt(r_lo * r_lo + a * a)
    C_seed, _, _ = combined_atom_potential(np.array([rho_lo]), p)
    return _soft_core_split_hermite(r, p, a0=float(C_seed[0]))


def soft_core_split(r, p: SplitParams):
    """Compute v, v_L, v_S and radial derivatives. Modes: paw|hermite|plateau|rho|softcore."""
    mode = p.split_mode
    if mode == 'paw':
        return _soft_core_split_paw(r, p)
    if mode == 'hermite':
        return _soft_core_split_hermite(r, p)
    if mode == 'plateau':
        return _soft_core_split_plateau(r, p)
    if mode == 'rho':
        return _soft_core_split_rho(r, p)
    if mode == 'softcore':
        return _soft_core_split_softcore(r, p)
    raise ValueError(f"Unknown split_mode={mode!r}; use 'paw'|'hermite'|'plateau'|'rho'|'softcore'")


def split_smoothness_metrics(r, s, r_lo=None, r_b=None):
    """PAW goal metrics: is v_L smoother than v, or does it invent bumps?

    force_overshoot = max(0, |v_L'| - |v'|)  — must be ~0 (no amplification)
    hf_ratio = RMS(d²v_L)/RMS(d²v)           — <1 means curvature damping
    n_extrema_*                              — wiggle count via force zero-crossings
    """
    r = np.asarray(r, dtype=np.float64)
    dvL = np.asarray(s['dv_L_dr'], dtype=np.float64)
    dv = np.asarray(s['dvdr'], dtype=np.float64)
    d2vL = np.asarray(s['d2v_L'], dtype=np.float64)
    d2v = np.asarray(s['d2v'], dtype=np.float64)
    vS = np.asarray(s['v_S'], dtype=np.float64)
    dvS = np.asarray(s['dv_S_dr'], dtype=np.float64)
    mask = np.ones(len(r), dtype=bool) if r_lo is None else ((r >= r_lo) & (r <= (r_b if r_b is not None else r[-1])))

    def n_extrema(y):
        dy = np.diff(y[mask])
        return int(np.sum(dy[:-1] * dy[1:] < 0)) if len(dy) > 1 else 0

    def rms(x):
        return float(np.sqrt(np.mean(x[mask]**2))) if np.any(mask) else 0.0

    rms_d2v = rms(d2v)
    overshoot = float(np.max(np.maximum(np.abs(dvL[mask]) - np.abs(dv[mask]), 0.0))) if np.any(mask) else 0.0
    return dict(
        max_abs_d2_vL=float(np.max(np.abs(d2vL[mask]))) if np.any(mask) else 0.0,
        max_abs_d2_v=float(np.max(np.abs(d2v[mask]))) if np.any(mask) else 0.0,
        force_overshoot=overshoot,
        n_extrema_vL=n_extrema(dvL),
        n_extrema_vS=n_extrema(dvS),
        max_abs_vS=float(np.max(np.abs(vS[mask]))) if np.any(mask) else 0.0,
        max_abs_dvS=float(np.max(np.abs(dvS[mask]))) if np.any(mask) else 0.0,
        hf_ratio=(rms(d2vL) / rms_d2v) if rms_d2v > 1e-30 else np.nan,
        rms_d2_vL=rms(d2vL),
        rms_d2_v=rms_d2v,
    )


def domain_violation_mask(r, p: SplitParams):
    return np.asarray(r, dtype=np.float64) < p.r_lo


def check_domain(r, p: SplitParams):
    mask = domain_violation_mask(r, p)
    if np.any(mask):
        idx = np.where(mask)[0]
        raise ValueError(f"Domain violation: r={float(np.min(r[mask])):.6f} < r_lo at indices {idx.tolist()}")
    return mask, None, None


def r_cut_candidates(p: SplitParams, candidates=(4.0, 5.0, 6.0)):
    r_lo_max = float(np.max(p.r_lo))
    valid = sorted([rc for rc in candidates if rc > r_lo_max])
    rejected = [rc for rc in candidates if rc <= r_lo_max]
    return valid, rejected, r_lo_max


def delta_b_candidates(p: SplitParams, candidates=(1.5, 2.0, 2.5)):
    valid = sorted([db for db in candidates if db > float(p.delta_a)])
    rejected = [db for db in candidates if db <= float(p.delta_a)]
    return valid, rejected


def eval_atom_ef(queries, atom_pos, p: SplitParams):
    dp = np.asarray(queries, dtype=np.float64).reshape(-1, 3) - np.asarray(atom_pos, dtype=np.float64)
    r = np.linalg.norm(dp, axis=-1)
    v, dvdr, _ = combined_atom_potential(r, p)
    r_safe = np.where(r > 1e-30, r, 1.0)
    F = (-dvdr / r_safe)[:, None] * dp
    return v, np.where((r > 1e-30)[:, None], F, 0.0)
