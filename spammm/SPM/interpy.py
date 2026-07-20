"""
interpy.py — Wendland C2 compact RBF / covariance kernels for Kriging & RBF.

Essence: Shared basis φ(r), dφ/dr, and pairwise distances for AFM z-scan interpolation.
Design: Compact support; optional per-point radii (wendland_c2_varR). Ported from ppafm
  pyProbeParticle/interpy.py — NumPy/SciPy only.
Open issues / caveats:
  - Fz for GridFF is finite-Δz in KrigingGridFF, not analytic in z (xy-only interpolators).
  - See doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import solve  # noqa: F401 — kept for parity with ppafm; unused here


def wendland_c2(r, R_basis, C=1.0):
    """Wendland C2 compactly supported RBF."""
    r = np.abs(r)
    mask = r < R_basis
    t = r[mask] / R_basis
    t1 = 1.0 - t
    t2 = t1 * t1
    t4 = t2 * t2
    out = np.zeros_like(r)
    out[mask] = t4 * (4.0 * t + C)
    return out


def wendland_c2_varR(r, R, C=1.0):
    """Wendland C2 for elementwise radii (same shape / broadcastable as r)."""
    r = np.abs(r)
    R = np.asarray(R, dtype=float)
    out = np.zeros_like(r, dtype=float)
    mask = r < R
    if np.any(mask):
        t = r[mask] / R[mask]
        t1 = 1.0 - t
        t2 = t1 * t1
        t4 = t2 * t2
        out[mask] = t4 * (4.0 * t + C)
    return out


def compact_c2_covariance(r, R_basis):
    """Compactly supported C2 Wendland used as covariance C(r)."""
    return wendland_c2(r, R_basis)


def compact_c2_variogram(r, R_basis):
    """Variogram gamma(r) = C(0) - C(r)."""
    return 1.0 - compact_c2_covariance(r, R_basis)


def wendland_c2_deriv(r, R_basis, C=1.0):
    """Analytical dφ/dr of Wendland C2; 0 for r >= R."""
    r = np.abs(r)
    out = np.zeros_like(r, dtype=float)
    mask = r < R_basis
    if np.any(mask):
        t = r[mask] / R_basis
        t1 = 1.0 - t
        out[mask] = (1.0 / R_basis) * (t1 ** 3) * (4.0 - 4.0 * C - 20.0 * t)
    return out


def wendland_c2_deriv_varR(r, R, C=1.0):
    """Analytical dφ/dr with elementwise radii."""
    r = np.abs(r)
    R = np.asarray(R, dtype=float)
    out = np.zeros_like(r, dtype=float)
    mask = r < R
    if np.any(mask):
        t = r[mask] / R[mask]
        t1 = 1.0 - t
        out[mask] = (1.0 / R[mask]) * (t1 ** 3) * (4.0 - 4.0 * C - 20.0 * t)
    return out


def pairwise_distances(points1, points2):
    """Distances between all pairs: (N,D) x (M,D) -> (N,M)."""
    return np.sqrt(np.sum((points1[:, np.newaxis, :] - points2[np.newaxis, :, :]) ** 2, axis=-1))
