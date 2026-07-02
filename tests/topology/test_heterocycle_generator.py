"""Tests for heterocycle_generator: sparse grid -> AtomicSystem."""
import numpy as np
import pytest
from spammm.topology.heterocycle_generator import build_geometry, build_atomic_system, EXAMPLES


class TestBuildGeometry:
    def test_naphthalene_atom_count(self):
        """Naphthalene: 10 atoms (4 E + 6 D)."""
        pos, enames, eo, subtypes, row_info = build_geometry(EXAMPLES['naphthalene'])
        assert len(pos) == 10, f"Naphthalene should have 10 atoms, got {len(pos)}"

    def test_naphthalene_all_carbon(self):
        """Naphthalene: all atoms are C."""
        pos, enames, eo, subtypes, row_info = build_geometry(EXAMPLES['naphthalene'])
        assert all(e == 'C' for e in enames), f"All should be C: {enames}"

    def test_purin_has_nitrogen(self):
        """Purin: should contain N atoms."""
        pos, enames, eo, subtypes, row_info = build_geometry(EXAMPLES['purin'])
        assert 'N' in enames, f"Purin should have N: {enames}"

    def test_coordinates_finite(self):
        """All coordinates should be finite numbers."""
        for name, system in EXAMPLES.items():
            pos, enames, eo, subtypes, row_info = build_geometry(system)
            assert np.all(np.isfinite(pos)), f"{name}: non-finite coordinates"

    def test_z_is_zero(self):
        """All atoms should be in the xy plane (z=0)."""
        pos, enames, eo, subtypes, row_info = build_geometry(EXAMPLES['naphthalene'])
        assert np.allclose(pos[:, 2], 0.0), f"z-coordinates not zero: {pos[:,2]}"


class TestBuildAtomicSystem:
    def test_naphthalene_bonds(self):
        """Naphthalene should have 11 bonds."""
        atoms = build_atomic_system(EXAMPLES['naphthalene'])
        assert atoms.bonds is not None
        assert len(atoms.bonds) == 11, f"Naphthalene should have 11 bonds, got {len(atoms.bonds)}"

    def test_naphthalene_natoms(self):
        """Naphthalene: 10 atoms."""
        atoms = build_atomic_system(EXAMPLES['naphthalene'])
        assert atoms.natoms == 10, f"Expected 10 atoms, got {atoms.natoms}"

    def test_purin_natoms(self):
        """Purin: check atom count (E:CN=2 + D:|NC|.|=5 + E:Nn=2 = 9)."""
        atoms = build_atomic_system(EXAMPLES['purin'])
        assert atoms.natoms == 9, f"Purin should have 9 atoms, got {atoms.natoms}"

    def test_enames_original_preserved(self):
        """_enames_original should preserve case for sp2/sp3 distinction."""
        atoms = build_atomic_system(EXAMPLES['purin'])
        eo = getattr(atoms, '_enames_original', None)
        assert eo is not None, "_enames_original not set"
        assert any(e != e.upper() for e in eo), f"Original case not preserved: {eo}"

    def test_all_examples_build(self):
        """All built-in examples should build without error."""
        for name, system in EXAMPLES.items():
            try:
                atoms = build_atomic_system(system)
                assert atoms.natoms > 0, f"{name}: no atoms"
            except Exception as e:
                pytest.fail(f"{name}: build failed: {e}")

    def test_bonds_are_symmetric(self):
        """Bonds should have i < j (sorted)."""
        atoms = build_atomic_system(EXAMPLES['naphthalene'])
        for i, j in atoms.bonds:
            assert i < j, f"Bond not sorted: ({i},{j})"
