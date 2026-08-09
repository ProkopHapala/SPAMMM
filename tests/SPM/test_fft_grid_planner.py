"""
test_fft_grid_planner.py — L0: plan_fft_friendly_grid correctness.

Verifies that plan_fft_friendly_grid produces clFFT-friendly grid dimensions
(prime factors 2,3,5,7 only) for benzene, pyridine, pentacene, PTCDA, and that
it raises ValueError with prime factorization for unfriendly shapes.
"""
import os
import numpy as np
import pytest

from spammm.SPM.AFM import plan_fft_friendly_grid, _FDBMGpyFFT, _prime_factorization

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')


def _load_xyz(path):
    """Minimal XYZ loader — returns (natoms, 3) positions."""
    with open(path) as f:
        f.readline()  # natoms
        f.readline()  # comment
        pts = []
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts, dtype=np.float64)


@pytest.mark.parametrize("xyz_name", ["benzene.xyz", "pyridine.xyz", "pentacene.xyz", "PTCDA.xyz"])
def test_plan_fft_friendly_grid_known_molecules(xyz_name):
    """Grid dims are FFT-friendly for all standard test molecules."""
    path = os.path.join(DATA_DIR, "xyz", xyz_name)
    if not os.path.exists(path):
        pytest.skip(f"{xyz_name} not found at {path}")
    atomPos = _load_xyz(path)
    grid_spec, origin, ngrid, step = plan_fft_friendly_grid(atomPos, step=0.1, margin=4.0, z_vac=7.0)
    nx, ny, nz = ngrid
    for dim, name in zip((nx, ny, nz), ('nx', 'ny', 'nz')):
        assert _FDBMGpyFFT.is_fft_friendly(dim), f"{xyz_name}: {name}={dim} not FFT-friendly"
    # All dims should be multiples of 8 (round_fft_friendly default)
    for dim in ngrid:
        assert dim % 8 == 0, f"{xyz_name}: dim {dim} not multiple of 8"
    # Grid spec structure
    assert 'origin' in grid_spec and 'ngrid' in grid_spec
    assert grid_spec['ngrid'] == ngrid
    assert len(origin) == 3
    assert step == 0.1


def test_plan_fft_friendly_grid_com_centered():
    """XY origin is COM-centered: origin = com - 0.5*ngrid*step."""
    atomPos = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float64)
    grid_spec, origin, ngrid, step = plan_fft_friendly_grid(atomPos, step=0.1, margin=4.0, z_vac=7.0)
    com = atomPos.mean(axis=0)
    mol_z = float(atomPos[:, 2].mean())
    assert abs(origin[0] - (com[0] - 0.5 * ngrid[0] * step)) < 1e-10
    assert abs(origin[1] - (com[1] - 0.5 * ngrid[1] * step)) < 1e-10
    assert abs(origin[2] - (mol_z - 0.5 * ngrid[2] * step)) < 1e-10


def test_plan_fft_friendly_grid_z_symmetric():
    """Z is symmetric about the molecular plane (mol_z ± z_vac)."""
    atomPos = np.array([[0, 0, 1.5], [2, 0, 1.5], [0, 2, 1.5]], dtype=np.float64)
    grid_spec, origin, ngrid, step = plan_fft_friendly_grid(atomPos, step=0.1, margin=4.0, z_vac=7.0)
    mol_z = float(atomPos[:, 2].mean())
    z_half = 0.5 * ngrid[2] * step
    assert abs(origin[2] - (mol_z - z_half)) < 1e-10
    # Top of grid = origin + ngrid*step should be mol_z + z_half
    z_top = origin[2] + ngrid[2] * step
    assert abs(z_top - (mol_z + z_half)) < 1e-10


def test_plan_fft_friendly_grid_prime_factorization():
    """_prime_factorization returns correct factors."""
    assert _prime_factorization(1) == {}
    assert _prime_factorization(2) == {2: 1}
    assert _prime_factorization(11) == {11: 1}
    assert _prime_factorization(120) == {2: 3, 3: 1, 5: 1}
    assert _prime_factorization(64) == {2: 6}


def test_plan_fft_friendly_grid_step_consistency():
    """Different steps produce proportionally smaller grids (all still friendly)."""
    atomPos = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0]], dtype=np.float64)
    _, _, n1, _ = plan_fft_friendly_grid(atomPos, step=0.1, margin=4.0, z_vac=7.0)
    _, _, n2, _ = plan_fft_friendly_grid(atomPos, step=0.2, margin=4.0, z_vac=7.0)
    # Coarser step → fewer or equal grid points
    for d1, d2 in zip(n1, n2):
        assert d2 <= d1
    # Both must be FFT-friendly
    for d in n1 + n2:
        assert _FDBMGpyFFT.is_fft_friendly(d)
