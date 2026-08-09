"""
test_bspline_boundary.py — L0: cubic B-spline prefilter boundary correctness.

Verifies that _bspline_prefilter_1d / _bspline_prefilter_2d produce control
coefficients that exactly reproduce nodal values at EVERY node — including the
first and last (boundary) — when interpolated with the zero-padding convention
used by cs_interp_h0 (kernel) / interp_h0 (Python).

The previous causal/anti-causal IIR (Unser infinite-signal filter) gave ~1e-16
interior error but O(1e-2) error at the first node. The tridiagonal solve
(inverse of the finite zero-padded interpolation matrix) is exact everywhere.
"""
import numpy as np
import pytest
from scipy.linalg import solve_banded

from spammm.surfaces.ContactSurface import (
    _bspline_prefilter_1d, _bspline_prefilter_2d, interp_h0, build_contact_height_map,
)


def _eval_bspline_1d(i, coeffs):
    """Evaluate cubic B-spline at node i with zero-padding (matches cs_interp_h0 1D)."""
    s = 0.0
    weights = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0]
    for k, ii in enumerate([i - 1, i, i + 1, i + 2]):
        if 0 <= ii < len(coeffs):
            s += weights[k] * coeffs[ii]
    return s


@pytest.mark.parametrize("n", [5, 10, 20, 50])
def test_prefilter_1d_interior_exact(n):
    """Interior nodes reproduce nodal values to machine precision."""
    xs = np.arange(n) * 0.3
    data = np.sin(xs) + 0.5 * np.cos(xs * 0.7)
    coeffs = _bspline_prefilter_1d(data)
    recon = np.array([_eval_bspline_1d(i, coeffs) for i in range(n)])
    err = np.abs(recon - data)
    assert err[1:-1].max() < 1e-10, f"interior max err {err[1:-1].max():.3e}"


@pytest.mark.parametrize("n", [5, 10, 20, 50])
def test_prefilter_1d_boundary_exact(n):
    """Boundary nodes (first and last) reproduce nodal values to machine precision.

    This is the key fix: the old IIR prefilter had O(1e-2) error at the first node.
    """
    xs = np.arange(n) * 0.3
    data = np.sin(xs) + 0.5 * np.cos(xs * 0.7)
    coeffs = _bspline_prefilter_1d(data)
    recon = np.array([_eval_bspline_1d(i, coeffs) for i in range(n)])
    err = np.abs(recon - data)
    assert err[0] < 1e-10, f"first node err {err[0]:.3e}"
    assert err[-1] < 1e-10, f"last node err {err[-1]:.3e}"


@pytest.mark.parametrize("ncx,ncy", [(6, 7), (10, 10), (15, 8)])
def test_prefilter_2d_all_nodes_exact(ncx, ncy):
    """2D separable prefilter: all nodes (incl. boundaries) reproduce via interp_h0."""
    dx = 0.4
    x0, y0 = 0.0, 0.0
    xs = np.arange(ncx) * dx
    ys = np.arange(ncy) * dx
    # h0_2d stored as (ncy, ncx) C-order, flat index jy*ncx+ix (matches build_contact_height_map)
    h0_2d = np.sin(xs[None, :]) + 0.3 * np.cos(ys[:, None] * 0.8)
    coeffs_2d = _bspline_prefilter_2d(h0_2d)
    cflat = coeffs_2d.reshape(-1)
    maxerr = 0.0
    for ix in range(ncx):
        for iy in range(ncy):
            v = interp_h0(float(xs[ix]), float(ys[iy]), cflat, dx, x0, y0, ncx, ncy)
            maxerr = max(maxerr, abs(v - h0_2d[iy, ix]))
    assert maxerr < 1e-10, f"2D max node err {maxerr:.3e}"


def test_prefilter_matches_tridiag_solve():
    """Prefilter output must match direct tridiagonal solve (the exact inverse)."""
    n = 20
    xs = np.arange(n) * 0.3
    data = np.sin(xs) + 0.5 * np.cos(xs * 0.7)
    coeffs = _bspline_prefilter_1d(data)
    # Direct tridiagonal solve: A c = d, A=tridiag(1/6, 4/6, 1/6)
    ab = np.zeros((3, n))
    ab[0, 1:] = 1.0 / 6.0
    ab[1, :] = 4.0 / 6.0
    ab[2, :-1] = 1.0 / 6.0
    ref = solve_banded((1, 1), ab, data)
    assert np.allclose(coeffs, ref, atol=1e-12), f"max diff {np.abs(coeffs - ref).max():.3e}"


def test_build_contact_height_map_returns_dict():
    """build_contact_height_map returns dict with h0_samples and h0_coeffs."""
    apos = np.array([[0, 0, 1.0], [1, 0, 1.5], [0, 1, 1.2], [1, 1, 1.8]], dtype=np.float64)
    Rs = np.array([2.0, 2.0, 2.0, 2.0])
    result = build_contact_height_map(apos, -2, -2, 0.5, 0.5, 10, 10, r_xy=8.0, Rs=Rs)
    assert isinstance(result, dict)
    assert 'h0_samples' in result and 'h0_coeffs' in result
    assert result['h0_samples'].shape == (100,)
    assert result['h0_coeffs'].shape == (100,)


def test_h0_coeffs_reproduce_samples_at_nodes():
    """h0_coeffs from build_contact_height_map reproduce h0_samples at every node via interp_h0."""
    apos = np.array([[0, 0, 1.0], [1, 0, 1.5], [0, 1, 1.2], [1, 1, 1.8]], dtype=np.float64)
    Rs = np.array([2.0, 2.0, 2.0, 2.0])
    ncx, ncy, dx, x0, y0 = 10, 10, 0.5, -2.0, -2.0
    result = build_contact_height_map(apos, x0, y0, dx, dx, ncx, ncy, r_xy=8.0, Rs=Rs)
    samples = result['h0_samples'].reshape(ncy, ncx)
    cflat = result['h0_coeffs']
    maxerr = 0.0
    for ix in range(ncx):
        for iy in range(ncy):
            cx = x0 + ix * dx
            cy = y0 + iy * dx
            v = interp_h0(float(cx), float(cy), cflat, dx, x0, y0, ncx, ncy)
            maxerr = max(maxerr, abs(v - samples[iy, ix]))
    assert maxerr < 1e-6, f"contact height map node err {maxerr:.3e}"


def test_h0_samples_are_physical_values():
    """h0_samples contain the raw nodal heights (physical contact values), not coefficients.

    Coefficients can overshoot/undershoot the nodal range. Samples must stay within
    [zmin, zmax+Rmax] for the sphere-envelope mode.
    """
    apos = np.array([[0, 0, 1.0], [1, 0, 1.5], [0, 1, 1.2], [1, 1, 1.8]], dtype=np.float64)
    Rs = np.array([2.0, 2.0, 2.0, 2.0])
    result = build_contact_height_map(apos, -2, -2, 0.5, 0.5, 10, 10, r_xy=8.0, Rs=Rs)
    samples = result['h0_samples']
    coeffs = result['h0_coeffs']
    # Samples range: zmin to zmax + Rmax (sphere envelope)
    zmin = float(apos[:, 2].min())
    zmax_Rmax = float(apos[:, 2].max()) + float(Rs.max())
    assert samples.min() >= zmin - 1e-6
    assert samples.max() <= zmax_Rmax + 1e-6
    # Coefficients CAN exceed this range (B-spline control points overshoot)
    # — this is exactly why h0_min/h0_max must come from samples, not coeffs.
    coeff_range = float(coeffs.max()) - float(coeffs.min())
    sample_range = float(samples.max()) - float(samples.min())
    # Coefficients typically have a wider range than samples
    assert coeff_range >= sample_range - 1e-6
