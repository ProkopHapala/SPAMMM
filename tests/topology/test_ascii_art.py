"""Tests for ascii_art_heterocycle: ASCII art -> AtomicSystem + Kekule solver."""
import numpy as np
import pytest
from spammm.topology.ascii_art_heterocycle import parse_ascii_art, ASCII_EXAMPLES
from spammm.topology.KekulePure import run_kekule_solver, mol_bond_types


class TestParseAsciiArt:
    def test_naphthalene_dimer_natoms(self):
        """Naphthalene (dimer format): 10 atoms."""
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        assert atoms.natoms == 10, f"Expected 10 atoms, got {atoms.natoms}"

    def test_naphthalene2_single_natoms(self):
        """Naphthalene (single-atom format): 10 atoms."""
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene2'])
        assert atoms.natoms == 10, f"Expected 10 atoms, got {atoms.natoms}"

    def test_all_examples_parse(self):
        """All ASCII examples should parse without error."""
        for name, art in ASCII_EXAMPLES.items():
            try:
                atoms = parse_ascii_art(art)
                assert atoms.natoms > 0, f"{name}: no atoms"
            except Exception as e:
                pytest.fail(f"{name}: parse failed: {e}")

    def test_bonds_not_empty(self):
        """Naphthalene should have bonds."""
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        assert atoms.bonds is not None and len(atoms.bonds) > 0, "No bonds generated"

    def test_enames_original_set(self):
        """_enames_original should be set for sp2/sp3 distinction."""
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        eo = getattr(atoms, '_enames_original', None)
        assert eo is not None, "_enames_original not set"

    def test_coordinates_finite(self):
        """All coordinates should be finite."""
        for name, art in ASCII_EXAMPLES.items():
            atoms = parse_ascii_art(art)
            assert np.all(np.isfinite(atoms.apos)), f"{name}: non-finite coordinates"


class TestRunKekuleSolver:
    def test_naphthalene_solve(self):
        """Naphthalene Kekule solver should converge and satisfy valence."""
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        atoms.neighs()
        r = run_kekule_solver(atoms, Kval=50.0, Kloc=5.0, Karo=0.5, allow_aromatic=True)
        assert r['err'] is None, f"Solver error: {r['err']}"
        rep = r['report']
        assert 'max_err' in rep
        assert rep['max_err'] < 1e-6, f"Valence constraint not satisfied: max_err={rep['max_err']}"

    def test_bond_orders_in_bounds(self):
        """All snapped bond orders should be in {0, 0.5, 1}."""
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        atoms.neighs()
        r = run_kekule_solver(atoms, Kval=50.0, Kloc=5.0, Karo=0.5, allow_aromatic=True)
        bo = r['bo_snap']
        heavy_bonds = [i for i in range(len(bo)) if bo[i] != 0.0 or True]
        valid = {0.0, 0.5, 1.0}
        for b in bo:
            assert any(abs(b - v) < 0.15 for v in valid), f"Bond order {b} not near 0/0.5/1"

    def test_bond_classification(self):
        """Solver report should have single/aromatic/double counts."""
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        atoms.neighs()
        r = run_kekule_solver(atoms, Kval=50.0, Kloc=5.0, Karo=0.5, allow_aromatic=True)
        rep = r['report']
        assert rep['single'] + rep['aromatic'] + rep['double'] > 0, "No bonds classified"


class TestMolBondTypes:
    def test_all_single_without_kekule(self):
        """Without kekule, all bond types should be 1 (single)."""
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        bt = mol_bond_types(atoms, bo_snap=None, kekule=False)
        assert bt is not None
        assert np.all(bt == 1), f"All should be single: {bt}"

    def test_aromatic_classification(self):
        """With aromatic allowed, ~0.5 bonds should be type 4."""
        atoms = parse_ascii_art(ASCII_EXAMPLES['naphthalene'])
        atoms.neighs()
        r = run_kekule_solver(atoms, Kval=50.0, Kloc=5.0, Karo=0.5, allow_aromatic=True)
        bt = mol_bond_types(atoms, bo_snap=r['bo_snap'], allow_aromatic=True, kekule=True)
        assert bt is not None
        # Should have at least some aromatic (4) or double (2) bonds
        assert np.any(bt >= 2), f"All bonds single despite kekule: {bt}"
