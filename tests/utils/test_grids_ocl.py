#!/usr/bin/env python3
"""L0: GridsOCL project_density preserves charge + dipole; Gaussian splat integrates to Z."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _moments_src_centers(rho, origin, step):
    from spammm.utils.GridsOCL import grid_moments_centers
    return grid_moments_centers(rho, origin, step)


def _moments_dst_corners(rho, origin, step):
    from spammm.utils.GridsOCL import grid_moments
    return grid_moments(rho, origin, step)


@pytest.fixture(scope='module')
def grids():
    from spammm.utils.GridsOCL import GridsOCL
    return GridsOCL(nloc=64, preferred_vendor='nvidia')


def test_project_preserves_charge_and_dipole(grids):
    """Fine blob → coarse grid: ∫ρ and p = ∫ρ r must match (within float32)."""
    step_s = 0.05
    origin_s = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
    n_s = (40, 40, 40)
    # asymmetric density blob (finite dipole)
    xs = origin_s[0] + step_s * (np.arange(n_s[0]) + 0.5)
    ys = origin_s[1] + step_s * (np.arange(n_s[1]) + 0.5)
    zs = origin_s[2] + step_s * (np.arange(n_s[2]) + 0.5)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    # two Gaussians of opposite? no — same sign offset → monopole + dipole
    rho = (2.5 * np.exp(-((X + 0.3) ** 2 + (Y - 0.1) ** 2 + (Z + 0.2) ** 2) / (2 * 0.15 ** 2))
           + 1.0 * np.exp(-((X - 0.4) ** 2 + Y ** 2 + (Z - 0.35) ** 2) / (2 * 0.12 ** 2)))
    rho = rho.astype(np.float32)

    step_d = 0.1
    origin_d = np.array([-1.5, -1.5, -1.5], dtype=np.float64)
    n_d = (32, 32, 32)  # covers support with margin

    q0, p0 = _moments_src_centers(rho, origin_s, step_s)
    assert q0 > 0.1

    dst = grids.project_density(rho, origin_s, step_s, origin_d, step_d, n_d)
    q1, p1 = _moments_dst_corners(dst, origin_d, step_d)

    print(f'[project] q_src={q0:.6f} q_dst={q1:.6f}  Δq={q1-q0:.3e}')
    print(f'[project] p_src={p0} p_dst={p1}  Δp={p1-p0}')

    assert abs(q1 - q0) / q0 < 1e-4, f'charge not preserved: {q0} → {q1}'
    # dipole: absolute tolerance relative to |p| scale
    scale = max(np.linalg.norm(p0), 1e-6)
    assert np.linalg.norm(p1 - p0) / scale < 2e-3, f'dipole not preserved: {p0} → {p1}'


def test_project_with_translation(grids):
    """Affine translation t should shift dipole by q*t."""
    step = 0.1
    origin_s = np.array([0.0, 0.0, 0.0])
    n_s = (20, 20, 20)
    xs = origin_s[0] + step * (np.arange(n_s[0]) + 0.5)
    ys = origin_s[1] + step * (np.arange(n_s[1]) + 0.5)
    zs = origin_s[2] + step * (np.arange(n_s[2]) + 0.5)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    rho = np.exp(-((X - 1.0) ** 2 + (Y - 1.0) ** 2 + (Z - 1.0) ** 2) / (2 * 0.2 ** 2)).astype(np.float32)

    t = np.array([0.5, -0.25, 0.75], dtype=np.float32)
    origin_d = np.array([-0.5, -0.5, -0.5])
    n_d = (40, 40, 40)

    q0, p0 = _moments_src_centers(rho, origin_s, step)
    dst = grids.project_density(rho, origin_s, step, origin_d, step, n_d, t=t)
    q1, p1 = _moments_dst_corners(dst, origin_d, step)

    p_expect = p0 + q0 * t.astype(np.float64)
    assert abs(q1 - q0) / q0 < 1e-4
    assert np.linalg.norm(p1 - p_expect) / max(np.linalg.norm(p_expect), 1e-6) < 2e-3


def test_cube_node_adapter_preserves_charge_and_dipole(grids):
    """Cube nodes must be adapted to the center-source convention of the OpenCL kernel."""
    from spammm.SPM.AFM_utils import project_density_to_grid
    from spammm.utils.GridsOCL import grid_moments

    step_s = np.array([0.08, 0.09, 0.11])
    origin_s = np.array([-1.2, -1.1, -1.3])
    n_s = (30, 28, 24)
    xs = origin_s[0] + step_s[0] * np.arange(n_s[0])
    ys = origin_s[1] + step_s[1] * np.arange(n_s[1])
    zs = origin_s[2] + step_s[2] * np.arange(n_s[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    rho = np.exp(-((X - 0.21) ** 2 + (Y + 0.17) ** 2 + (Z - 0.09) ** 2) / (2 * 0.18 ** 2)).astype(np.float32)

    origin_d = np.array([-1.6, -1.5, -1.7])
    step_d = 0.1
    n_d = (36, 34, 36)
    q0, p0 = grid_moments(rho, origin_s, step_s)
    dst, same_grids = project_density_to_grid(rho, origin_s, step_s, origin_d, step_d, n_d, grids=grids)
    q1, p1 = grid_moments(dst, origin_d, step_d)

    assert same_grids is grids
    assert abs(q1 - q0) / q0 < 1e-4
    assert np.linalg.norm(p1 - p0) / max(np.linalg.norm(p0), 1e-6) < 2e-3


def test_gaussian_splat_integral(grids):
    """∫ ρ_NA ≈ Σ Z for one atom on a fine enough grid."""
    step = 0.1
    origin = np.array([-3.0, -3.0, -3.0])
    n = (60, 60, 60)
    grid = np.zeros(n, dtype=np.float32)
    Z = 8.0
    grids.splat_gaussians(grid, origin, step, [[0.0, 0.0, 0.0]], [Z], sigma=0.5, sign=1.0, nsig=5.0)
    q = float(grid.sum() * step ** 3)
    print(f'[gauss] ∫ρ={q:.6f}  Z={Z}')
    assert abs(q - Z) / Z < 0.02  # truncation + float32


def test_vs_map_coordinates_charge(grids):
    """Projection charge closer to exact than scipy map_coordinates sampling."""
    from scipy.ndimage import map_coordinates
    from spammm.SPM.AFM_utils import resample_field_to_grid

    step_s = 0.05292
    origin_s = np.array([-2.0, -2.0, -2.0])
    n_s = (80, 80, 80)
    xs = origin_s[0] + step_s * (np.arange(n_s[0]) + 0.5)
    ys = origin_s[1] + step_s * (np.arange(n_s[1]) + 0.5)
    zs = origin_s[2] + step_s * (np.arange(n_s[2]) + 0.5)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    rho = np.exp(-((X - 0.2) ** 2 + (Y + 0.15) ** 2 + (Z - 0.1) ** 2) / (2 * 0.25 ** 2)).astype(np.float32)

    step_d = 0.1
    origin_d = np.array([-2.5, -2.5, -2.5])
    n_d = (50, 50, 50)

    q0, p0 = _moments_src_centers(rho, origin_s, step_s)
    dst_p = grids.project_density(rho, origin_s, step_s, origin_d, step_d, n_d)
    q_p, p_p = _moments_dst_corners(dst_p, origin_d, step_d)

    dst_s = resample_field_to_grid(rho, origin_s, step_s, origin_d, step_d, n_d)
    # resample samples density values — moments with centers on dest
    q_s, p_s = _moments_src_centers(dst_s, origin_d, step_d)

    print(f'[compare] q0={q0:.6f}  project={q_p:.6f}  sample={q_s:.6f}')
    print(f'[compare] |Δq|_project={abs(q_p-q0):.3e}  |Δq|_sample={abs(q_s-q0):.3e}')
    assert abs(q_p - q0) < abs(q_s - q0) * 0.5 or abs(q_p - q0) / q0 < 1e-3
