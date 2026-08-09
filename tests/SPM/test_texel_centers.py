"""
test_texel_centers.py — L0: verify _make_dinv_axis_aligned produces half-texel offset.

OpenCL read_imagef with normalized coords maps u=(i+0.5)/n to texel center i.
Without the +0.5/n offset in dinvA/B/C .w, u=i/n lands on texel edges →
systematic half-voxel shift in every GridFF interpolation.

This test verifies the half-texel correction is present in the .w component
of dinvA, dinvB, dinvC produced by _make_dinv_axis_aligned and _make_dinv_lvec.
"""
import numpy as np
import pytest

from spammm.SPM.AFM import AFMulator


def test_dinv_axis_aligned_half_texel_offset():
    """_make_dinv_axis_aligned: .w contains +0.5/n half-texel offset (not 0)."""
    L = (10.0, 20.0, 30.0)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    n = (50, 100, 150)
    dinvA, dinvB, dinvC = AFMulator._make_dinv_axis_aligned(L, origin, n)
    nx, ny, nz = n
    # .w = -origin/L + 0.5/n; with origin=0 → .w = 0.5/n (float32 tolerance)
    assert abs(dinvA[3] - 0.5 / nx) < 1e-6, f"dinvA.w={dinvA[3]}, expected 0.5/{nx}={0.5/nx}"
    assert abs(dinvB[3] - 0.5 / ny) < 1e-6, f"dinvB.w={dinvB[3]}, expected 0.5/{ny}={0.5/ny}"
    assert abs(dinvC[3] - 0.5 / nz) < 1e-6, f"dinvC.w={dinvC[3]}, expected 0.5/{nz}={0.5/nz}"


def test_dinv_axis_aligned_nonzero_origin():
    """_make_dinv_axis_aligned: .w = -origin/L + 0.5/n for nonzero origin."""
    L = (10.0, 20.0, 30.0)
    origin = np.array([-5.0, 3.0, 1.0], dtype=np.float32)
    n = (50, 100, 150)
    dinvA, dinvB, dinvC = AFMulator._make_dinv_axis_aligned(L, origin, n)
    nx, ny, nz = n
    expected_w = [-origin[i] / L[i] + 0.5 / n[i] for i in range(3)]
    assert abs(dinvA[3] - expected_w[0]) < 1e-6
    assert abs(dinvB[3] - expected_w[1]) < 1e-6
    assert abs(dinvC[3] - expected_w[2]) < 1e-6


def test_dinv_axis_aligned_no_zero_offset():
    """The .w component must NOT be zero (the old buggy behavior without half-texel)."""
    L = (10.0, 20.0, 30.0)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    n = (50, 100, 150)
    dinvA, dinvB, dinvC = AFMulator._make_dinv_axis_aligned(L, origin, n)
    # With origin=0, old bug would give .w=0; fix gives .w=0.5/n (nonzero)
    assert dinvA[3] != 0.0, "dinvA.w is 0 — half-texel offset missing"
    assert dinvB[3] != 0.0, "dinvB.w is 0 — half-texel offset missing"
    assert dinvC[3] != 0.0, "dinvC.w is 0 — half-texel offset missing"


def test_dinv_lvec_half_texel_offset():
    """_make_dinv_lvec: .w contains +0.5/n half-texel offset."""
    invMT = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    n = (50, 100, 150)
    dinvA, dinvB, dinvC = AFMulator._make_dinv_lvec(invMT, n)
    nx, ny, nz = n
    assert abs(dinvA[3] - 0.5 / nx) < 1e-6, f"dinvA.w={dinvA[3]}, expected 0.5/{nx}"
    assert abs(dinvB[3] - 0.5 / ny) < 1e-6, f"dinvB.w={dinvB[3]}, expected 0.5/{ny}"
    assert abs(dinvC[3] - 0.5 / nz) < 1e-6, f"dinvC.w={dinvC[3]}, expected 0.5/{nz}"


def test_dinv_axis_aligned_diagonal():
    """_make_dinv_axis_aligned: diagonal components are 1/L (axis-aligned)."""
    L = (10.0, 20.0, 30.0)
    origin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    n = (50, 100, 150)
    dinvA, dinvB, dinvC = AFMulator._make_dinv_axis_aligned(L, origin, n)
    assert abs(dinvA[0] - 1.0 / L[0]) < 1e-6
    assert abs(dinvB[1] - 1.0 / L[1]) < 1e-6
    assert abs(dinvC[2] - 1.0 / L[2]) < 1e-6
    # Off-diagonal components are 0
    assert dinvA[1] == 0.0 and dinvA[2] == 0.0
    assert dinvB[0] == 0.0 and dinvB[2] == 0.0
    assert dinvC[0] == 0.0 and dinvC[1] == 0.0


def test_dinv_maps_voxel_to_texel_center():
    """End-to-end: dinv maps voxel i to normalized coord (i+0.5)/n (texel center)."""
    L = (10.0, 20.0, 30.0)
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    n = (50, 100, 150)
    dinvA, dinvB, dinvC = AFMulator._make_dinv_axis_aligned(L, origin, n)
    nx, ny, nz = n
    step = [L[i] / n[i] for i in range(3)]
    # Voxel i at position pos = origin + i*step
    for i in [0, 1, 25, 49]:
        pos = origin[0] + i * step[0]
        coord = pos * dinvA[0] + 1.0 * dinvA[3]  # dot([pos,1], dinvA)
        expected = (i + 0.5) / nx
        assert abs(coord - expected) < 1e-5, f"voxel {i}: coord={coord}, expected {expected}"
