"""Tests for KekulePure pi-bond-order optimizer."""
import numpy as np
import pytest
from spammm.AtomicSystem import AtomicSystem
from spammm.topology.KekulePure import KekulePure, make_n_pi, make_pi_mask, optimize_pi_bonds


def _make_benzene():
    """Benzene: 6 C atoms in a hexagon, 6 bonds."""
    a = 1.42
    angles = np.arange(6) * (np.pi / 3.0)
    apos = np.column_stack([a * np.cos(angles), a * np.sin(angles), np.zeros(6)])
    bonds = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]], dtype=np.int32)
    sys = AtomicSystem(apos=apos, enames=['C']*6)
    sys.bonds = bonds
    sys.natoms = 6
    return sys


def _make_naphthalene():
    """Naphthalene: 10 C atoms, 11 bonds (two fused hexagons)."""
    a = 1.42
    s3 = np.sqrt(3.0)
    apos = np.array([
        [0,      0, 0], [a,       0, 0],
        [-a/2,  a*s3/2, 0], [a*1.5, a*s3/2, 0],
        [-a/2, -a*s3/2, 0], [a*1.5, -a*s3/2, 0],
        [a*2.5,  a*s3/2, 0], [a*2.5, -a*s3/2, 0],
        [a*3.5,  0, 0], [a*2.0, 0, 0],
    ])
    bonds = np.array([
        [0,2],[0,4],[0,1],[1,3],[1,5],[2,9],[3,6],[4,9],[5,9],[6,8],[7,8]
    ], dtype=np.int32)
    bonds = np.array([
        [0,2],[0,4],[0,1],[1,3],[1,5],[2,9],[4,9],[3,6],[5,7],[6,8],[7,8]
    ], dtype=np.int32)
    sys = AtomicSystem(apos=apos, enames=['C']*10)
    sys.bonds = bonds
    sys.natoms = 10
    return sys


class TestKekulePureBenzene:
    def test_valence_constraint(self):
        """After solve, sum of pi bond orders at each atom == n_pi (1 for sp2 C)."""
        sys = _make_benzene()
        k = KekulePure(sys, n_pi=np.ones(6), Kval=50.0, Karo=0.5, allow_aromatic=True)
        k.solve_quadratic(Kloc=0.0)
        val = k.project_valence()
        assert np.allclose(val, 1.0, atol=1e-6), f"Valence constraint violated: {val}"

    def test_bond_order_bounds(self):
        """All pi bond orders in [0, 1]."""
        sys = _make_benzene()
        k = KekulePure(sys, n_pi=np.ones(6), Kval=50.0, Karo=0.5, allow_aromatic=True)
        k.solve_quadratic(Kloc=0.0)
        bo = k.pi_bond_orders()
        assert np.all(bo >= -1e-10) and np.all(bo <= 1.0 + 1e-10), f"Bond orders out of [0,1]: {bo}"

    def test_benzene_aromatic(self):
        """Benzene with aromatic allowed should give all bonds ~0.5 (aromatic)."""
        sys = _make_benzene()
        k = KekulePure(sys, n_pi=np.ones(6), Kval=50.0, Karo=0.5, allow_aromatic=True)
        k.solve_quadratic(Kloc=0.0)
        bo = k.pi_bond_orders()
        assert np.allclose(bo, 0.5, atol=0.1), f"Benzene should be aromatic (all ~0.5): {bo}"

    def test_benzene_discrete(self):
        """Benzene with aromatic=False, after snap, should give 3 single + 3 double."""
        sys = _make_benzene()
        k = KekulePure(sys, n_pi=np.ones(6), Kval=50.0, Karo=0.5, allow_aromatic=False)
        k.solve_quadratic(Kloc=0.0)
        k.Kloc = 5.0
        k.solve_snap(niter=50)
        k.snap()
        cls = k.classify()
        n_single = int(np.sum(cls == 0))
        n_double = int(np.sum(cls == 2))
        n_aromatic = int(np.sum(cls == 1))
        assert n_single == 3, f"Expected 3 single bonds, got {n_single}"
        assert n_double == 3, f"Expected 3 double bonds, got {n_double}"
        assert n_aromatic == 0, f"Expected 0 aromatic bonds, got {n_aromatic}"

    def test_total_bond_orders(self):
        """Total bond orders = 1 + pi."""
        sys = _make_benzene()
        k = KekulePure(sys, n_pi=np.ones(6), Kval=50.0, Karo=0.5, allow_aromatic=True)
        k.solve_quadratic(Kloc=0.0)
        total = k.total_bond_orders()
        bo = k.pi_bond_orders()
        assert np.allclose(total, 1.0 + bo)


class TestKekulePureNaphthalene:
    def test_valence_constraint(self):
        """Naphthalene: all atoms should satisfy valence constraint after solve."""
        sys = _make_naphthalene()
        k = KekulePure(sys, n_pi=np.ones(10), Kval=50.0, Karo=0.5, allow_aromatic=True)
        k.solve_quadratic(Kloc=0.0)
        val = k.project_valence()
        assert np.allclose(val, 1.0, atol=1e-6), f"Valence constraint violated: {val}"

    def test_naphthalene_bond_count(self):
        """Naphthalene has 11 bonds."""
        sys = _make_naphthalene()
        assert len(sys.bonds) == 11, f"Expected 11 bonds, got {len(sys.bonds)}"


class TestMakeNPi:
    def test_uppercase_sp2(self):
        """Uppercase element symbols -> n_pi=1 (sp2)."""
        sys = AtomicSystem(apos=np.zeros((3,3)), enames=['C','N','O'])
        sys.natoms = 3
        n_pi = make_n_pi(sys)
        assert np.allclose(n_pi, 1.0), f"Uppercase C/N/O should be sp2: {n_pi}"

    def test_lowercase_sp3(self):
        """Lowercase element symbols -> n_pi=0 (sp3)."""
        sys = AtomicSystem(apos=np.zeros((3,3)), enames=['c','n','o'])
        sys.natoms = 3
        n_pi = make_n_pi(sys)
        assert np.allclose(n_pi, 0.0), f"Lowercase c/n/o should be sp3: {n_pi}"

    def test_mixed(self):
        """Mixed case: uppercase=sp2, lowercase=sp3."""
        sys = AtomicSystem(apos=np.zeros((4,3)), enames=['C','c','N','n'])
        sys.natoms = 4
        n_pi = make_n_pi(sys)
        expected = [1.0, 0.0, 1.0, 0.0]
        assert np.allclose(n_pi, expected), f"Mixed case n_pi: {n_pi}, expected {expected}"


class TestMakePiMask:
    def test_default_elements(self):
        """Default mask includes C, N, O."""
        sys = AtomicSystem(apos=np.zeros((4,3)), enames=['C','N','O','H'])
        sys.natoms = 4
        mask = make_pi_mask(sys)
        assert mask.tolist() == [True, True, True, False], f"Pi mask: {mask}"


class TestOptimizePiBonds:
    def test_runs_and_returns_finite(self):
        """optimize_pi_bonds should run and return finite bond orders."""
        sys = _make_benzene()
        k = optimize_pi_bonds(sys, n_pi=np.ones(6), Kval=1.0, Karo=0.3, dt=0.1, nmax=500)
        bo = k.pi_bond_orders()
        assert np.all(np.isfinite(bo)), f"Non-finite bond orders: {bo}"
        assert np.all(bo >= -1e-10) and np.all(bo <= 1.0 + 1e-10), f"Bond orders out of [0,1]: {bo}"
