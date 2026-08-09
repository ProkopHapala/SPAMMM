"""
test_clamp_consistency.py — L0: contact kernel clamp consistency (F=−∇E for s<0).

The contact-surface z-basis (poly_z_doubling_modes in ContactSurface.py, matching
contact_surface.cl:poly_z_doubling_modes) clamps s = dz - poly_z0 to [0, Rc] via
fmax/clip. When s < 0 (below the contact surface):
  - E is constant (phi = t^m = 1^m = 1 for all modes, since x=0 → t=1)
  - F = -dE/dz = 0 (dphi = 0, since active=false)

This test verifies that:
1. For s < 0: phi is constant (same value for all s < 0) and dphi = 0
2. For s > 0 (active region): finite-difference F = -dE/dz matches analytical dphi
3. For s < 0: finite-difference F ≈ 0 (E is constant → no force)
4. The transition at s=0 is consistent (no spurious nonzero F with constant E)
"""
import numpy as np
import pytest

from spammm.surfaces.ContactSurface import poly_z_doubling_modes


def test_clamp_s_negative_phi_constant_dphi_zero():
    """For s < 0: phi is constant (all modes = 1) and dphi = 0 (no force)."""
    s_neg = np.array([-0.5, -1.0, -2.0, -5.0, -10.0], dtype=np.float64)
    phi, dphi, t, x = poly_z_doubling_modes(s_neg, poly_R=10.0, m_start=4, nz=5, poly_z0=0.0)
    # x = clip(s/R, 0, 1) = 0 for all s < 0
    assert np.all(x == 0.0), f"x={x}, expected all 0 for s<0"
    # t = 1 - x = 1
    assert np.allclose(t, 1.0), f"t={t}, expected all 1 for s<0"
    # phi = t^m = 1 for all modes (constant)
    for k in range(phi.shape[1]):
        assert np.allclose(phi[:, k], 1.0), f"phi[:,{k}]={phi[:,k]}, expected all 1"
    # dphi = 0 (no force in clamped region)
    assert np.allclose(dphi, 0.0), f"dphi={dphi}, expected all 0 for s<0"


def test_clamp_s_negative_finite_difference_force_zero():
    """Finite-difference F = -dE/dz ≈ 0 for s < 0 (E is constant in clamped region)."""
    # Sample s values densely in the clamped region
    ds = 1e-4
    s_center = np.array([-1.0, -2.0, -5.0], dtype=np.float64)
    for s0 in s_center:
        s_vals = np.array([s0 - ds, s0, s0 + ds], dtype=np.float64)
        phi, dphi, _, _ = poly_z_doubling_modes(s_vals, poly_R=10.0, m_start=4, nz=5, poly_z0=0.0)
        # E = sum of phi across modes (with unit coefficients)
        E = phi.sum(axis=1)
        # F = -dE/dz via central finite difference
        F_fd = -(E[2] - E[0]) / (2 * ds)
        assert abs(F_fd) < 1e-6, f"s={s0}: finite-diff F={F_fd:.3e}, expected ~0 (E constant)"


def test_clamp_active_region_finite_difference_matches_analytical():
    """For s > 0 (active): finite-difference F = -dE/dz matches analytical dphi."""
    ds = 1e-5
    s_test = np.array([0.5, 1.0, 3.0, 5.0, 8.0], dtype=np.float64)
    poly_R = 10.0
    for s0 in s_test:
        s_vals = np.array([s0 - ds, s0, s0 + ds], dtype=np.float64)
        phi, dphi, _, _ = poly_z_doubling_modes(s_vals, poly_R=poly_R, m_start=4, nz=5, poly_z0=0.0)
        E = phi.sum(axis=1)
        F_fd = -(E[2] - E[0]) / (2 * ds)
        F_analytical = -dphi[1].sum()  # F = -dE/dz = -sum(dphi)
        rel_err = abs(F_fd - F_analytical) / max(abs(F_analytical), 1e-30)
        assert rel_err < 1e-4, f"s={s0}: fd={F_fd:.6e}, analytical={F_analytical:.6e}, rel_err={rel_err:.3e}"


def test_clamp_transition_consistency():
    """At s=0 boundary: no spurious nonzero F with constant E on the clamped side."""
    # Sample just below and just above s=0
    ds = 1e-5
    s_below = np.array([-ds, -2 * ds, -3 * ds], dtype=np.float64)
    s_above = np.array([ds, 2 * ds, 3 * ds], dtype=np.float64)
    phi_below, dphi_below, _, _ = poly_z_doubling_modes(s_below, poly_R=10.0, m_start=4, nz=5)
    phi_above, dphi_above, _, _ = poly_z_doubling_modes(s_above, poly_R=10.0, m_start=4, nz=5)
    # Below: E constant, F=0
    E_below = phi_below.sum(axis=1)
    assert np.allclose(E_below, E_below[0]), "E not constant below s=0"
    assert np.allclose(dphi_below, 0.0), "dphi not 0 below s=0"
    # Above: F should be nonzero (active region) — verifies the clamp actually transitions
    E_above = phi_above.sum(axis=1)
    # E should be decreasing (approaching 0 as s→Rc) — derivative should be negative
    dE_above = np.diff(E_above)
    assert np.all(dE_above < 0), f"E not decreasing above s=0: dE={dE_above}"


def test_clamp_with_poly_z0_offset():
    """Clamp works correctly with nonzero poly_z0 (s = dz - poly_z0)."""
    poly_z0 = 2.0
    poly_R = 10.0
    # s = dz - poly_z0; s < 0 when dz < poly_z0
    dz_below = np.array([0.0, 1.0, 1.5], dtype=np.float64)  # s = -2, -1, -0.5
    dz_above = np.array([3.0, 5.0, 8.0], dtype=np.float64)   # s = 1, 3, 6
    phi_b, dphi_b, _, _ = poly_z_doubling_modes(dz_below, poly_R=poly_R, m_start=4, nz=5, poly_z0=poly_z0)
    phi_a, dphi_a, _, _ = poly_z_doubling_modes(dz_above, poly_R=poly_R, m_start=4, nz=5, poly_z0=poly_z0)
    # Below poly_z0: constant E, zero F
    E_b = phi_b.sum(axis=1)
    assert np.allclose(E_b, E_b[0]), f"E not constant below poly_z0: {E_b}"
    assert np.allclose(dphi_b, 0.0), f"dphi not 0 below poly_z0: {dphi_b}"
    # Above poly_z0: active (F nonzero)
    assert not np.allclose(dphi_a, 0.0), "dphi all 0 above poly_z0 — should be active"


def test_clamp_e_constant_not_nonzero_force():
    """Explicitly verify: when E is constant (s<0), F=0 — not the inconsistent case
    where E is constant but F is nonzero (which would violate F=-∇E)."""
    s = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
    phi, dphi, _, _ = poly_z_doubling_modes(s, poly_R=10.0, m_start=4, nz=5, poly_z0=0.0)
    E = phi.sum(axis=1)
    # E is constant
    assert np.allclose(np.diff(E), 0.0), "E varies in clamped region"
    # F (dphi) is zero — consistent with constant E
    assert np.allclose(dphi, 0.0), "F nonzero while E constant — F=-∇E violated"
