"""
test_fragmentation.py — Tests for molecular graph segmentation algorithms.

Tests connected components, bridges, articulation points, and local bridges
on AtomicGraph. All tests are headless (L1 only).

Run:
  pytest tests/topology/test_fragmentation.py
  pytest tests/topology/test_fragmentation.py -k bridges
"""

import pytest
import numpy as np

from spammm.topology.AtomicGraph import AtomicGraph, Atom, Bond


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_hex_ring(graph, cx=0.0, cy=0.0, r=1.42):
    """Create a 6-membered ring of C atoms at hex positions. Returns list of 6 atoms."""
    atoms = []
    for i in range(6):
        ang = i * np.pi / 3
        pos = np.array([cx + r * np.cos(ang), cy + r * np.sin(ang), 0.0])
        atoms.append(graph.add_atom(pos, 'C', 6, npi=1))
    for i in range(6):
        graph.add_bond(atoms[i], atoms[(i + 1) % 6])
    return atoms


def _make_chain(graph, start_pos, n, step=1.42, ename='C', atype=6):
    """Create a linear chain of n atoms. Returns list of atoms."""
    atoms = []
    for i in range(n):
        pos = np.array([start_pos[0] + i * step, start_pos[1], start_pos[2]])
        atoms.append(graph.add_atom(pos, ename, atype, npi=1))
    for i in range(n - 1):
        graph.add_bond(atoms[i], atoms[i + 1])
    return atoms


def _bond_ids(bonds):
    """Return set of frozenset(atom_id pairs) from list of Bond objects."""
    return {frozenset((b.a._id, b.b._id)) for b in bonds}


def _atom_ids(atoms):
    """Return set of atom _ids from list of Atom objects."""
    return {a._id for a in atoms}


# ═══════════════════════════════════════════════════════════════════════════════
# 01. Connected Components
# ═══════════════════════════════════════════════════════════════════════════════

class TestConnectedComponents:

    def test_empty_graph(self):
        g = AtomicGraph()
        assert g.find_connected_components() == []

    def test_single_atom(self):
        g = AtomicGraph()
        a = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        comps = g.find_connected_components()
        assert len(comps) == 1
        assert len(comps[0]) == 1

    def test_two_disconnected_atoms(self):
        g = AtomicGraph()
        g.add_atom(np.array([0, 0, 0]), 'C', 6)
        g.add_atom(np.array([10, 0, 0]), 'C', 6)
        comps = g.find_connected_components()
        assert len(comps) == 2

    def test_two_disconnected_molecules(self):
        g = AtomicGraph()
        _make_hex_ring(g, cx=0, cy=0)
        _make_hex_ring(g, cx=10, cy=0)
        comps = g.find_connected_components()
        assert len(comps) == 2
        assert len(comps[0]) == 6
        assert len(comps[1]) == 6

    def test_connected_ring(self):
        g = AtomicGraph()
        _make_hex_ring(g)
        comps = g.find_connected_components()
        assert len(comps) == 1
        assert len(comps[0]) == 6

    def test_two_rings_connected_by_bond(self):
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=5, cy=0)
        g.add_bond(r1[0], r2[0])
        comps = g.find_connected_components()
        assert len(comps) == 1
        assert len(comps[0]) == 12

    def test_ignores_dead_atoms(self):
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        g.add_bond(a1, a2)
        a3 = g.add_atom(np.array([10, 0, 0]), 'C', 6)
        g.remove_atom(a3, soft=True)
        comps = g.find_connected_components()
        assert len(comps) == 1
        assert len(comps[0]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 02. Bridges
# ═══════════════════════════════════════════════════════════════════════════════

class TestBridges:

    def test_empty_graph(self):
        g = AtomicGraph()
        assert g.find_bridges() == []

    def test_single_bond(self):
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        g.add_bond(a1, a2)
        bridges = g.find_bridges()
        assert len(bridges) == 1

    def test_chain_all_bridges(self):
        """In a chain A-B-C-D, all 3 bonds are bridges."""
        g = AtomicGraph()
        atoms = _make_chain(g, [0, 0, 0], 4)
        bridges = g.find_bridges()
        assert len(bridges) == 3

    def test_ring_no_bridges(self):
        """In a ring (cycle), no bond is a bridge."""
        g = AtomicGraph()
        _make_hex_ring(g)
        bridges = g.find_bridges()
        assert len(bridges) == 0

    def test_two_rings_connected_by_bond(self):
        """Two rings connected by single bond: only the connecting bond is a bridge."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=5, cy=0)
        connecting = g.add_bond(r1[0], r2[0])
        bridges = g.find_bridges()
        assert len(bridges) == 1
        assert bridges[0] is connecting

    def test_two_rings_connected_by_chain(self):
        """Two rings connected by 2-bond chain: both chain bonds are bridges."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=5, cy=0)
        linker1 = g.add_atom(np.array([2.5, 0, 0]), 'C', 6, npi=1)
        linker2 = g.add_atom(np.array([3.5, 0, 0]), 'C', 6, npi=1)
        g.add_bond(r1[0], linker1)
        g.add_bond(linker1, linker2)
        g.add_bond(linker2, r2[0])
        bridges = g.find_bridges()
        assert len(bridges) == 3

    def test_fused_rings_no_bridges(self):
        """Naphthalene (2 fused rings): no bridges — shared edge is not a bridge."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        # Second ring shares atoms r1[1] and r1[2]
        atoms2 = [r1[1], r1[2]]
        for i in range(4):
            ang = (2 + i + 1) * np.pi / 3
            pos = np.array([np.cos(ang) * 1.42, np.sin(ang) * 1.42, 0.0])
            atoms2.append(g.add_atom(pos, 'C', 6, npi=1))
        g.add_bond(atoms2[1], atoms2[2])
        g.add_bond(atoms2[2], atoms2[3])
        g.add_bond(atoms2[3], atoms2[4])
        g.add_bond(atoms2[4], atoms2[5])
        g.add_bond(atoms2[5], atoms2[0])
        bridges = g.find_bridges()
        assert len(bridges) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 03. Articulation Points
# ═══════════════════════════════════════════════════════════════════════════════

class TestArticulationPoints:

    def test_empty_graph(self):
        g = AtomicGraph()
        assert g.find_articulation_points() == []

    def test_single_bond_no_ap(self):
        """A-B: neither atom is an AP (removing either leaves 1 atom = 1 component)."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        g.add_bond(a1, a2)
        aps = g.find_articulation_points()
        assert len(aps) == 0

    def test_chain_aps(self):
        """Chain A-B-C-D: B and C are articulation points."""
        g = AtomicGraph()
        atoms = _make_chain(g, [0, 0, 0], 4)
        aps = g.find_articulation_points()
        ap_ids = {a._id for a in aps}
        assert ap_ids == {atoms[1]._id, atoms[2]._id}

    def test_ring_no_aps(self):
        """Ring: no articulation points."""
        g = AtomicGraph()
        _make_hex_ring(g)
        aps = g.find_articulation_points()
        assert len(aps) == 0

    def test_two_rings_connected_by_bond(self):
        """Two rings connected by bond: the two endpoint atoms of the bridge are APs."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=5, cy=0)
        g.add_bond(r1[0], r2[0])
        aps = g.find_articulation_points()
        ap_ids = {a._id for a in aps}
        assert ap_ids == {r1[0]._id, r2[0]._id}


# ═══════════════════════════════════════════════════════════════════════════════
# 04. Local Bridges
# ═══════════════════════════════════════════════════════════════════════════════

class TestLocalBridges:

    def test_empty_graph(self):
        g = AtomicGraph()
        assert g.find_local_bridges() == []

    def test_ring_local_bridges_max_dist_3(self):
        """In a 6-ring, alternate path around ring is length 5 > 3 → all bonds are local bridges."""
        g = AtomicGraph()
        _make_hex_ring(g)
        lb = g.find_local_bridges(max_dist=3)
        assert len(lb) == 6

    def test_ring_no_local_bridges_max_dist_5(self):
        """In a 6-ring, alternate path is length 5 ≤ 5 → no local bridges at max_dist=5."""
        g = AtomicGraph()
        _make_hex_ring(g)
        lb = g.find_local_bridges(max_dist=5)
        assert len(lb) == 0

    def test_chain_all_local_bridges(self):
        """Chain bonds: no alternate path → all are local bridges."""
        g = AtomicGraph()
        _make_chain(g, [0, 0, 0], 4)
        lb = g.find_local_bridges(max_dist=3)
        assert len(lb) == 3

    def test_local_bridge_not_global_bridge(self):
        """Two rings connected by direct bond AND a long chain.
        The direct bond is a local bridge (alt path > 3) but NOT a global bridge."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=10, cy=0)
        # Direct bond between rings
        direct = g.add_bond(r1[0], r2[0])
        # Long alternative path: r1[1] - a - b - c - d - r2[1]
        p = [r1[1]]
        for i in range(4):
            p.append(g.add_atom(np.array([2 + i * 1.5, 3, 0]), 'C', 6, npi=1))
        p.append(r2[1])
        for i in range(len(p) - 1):
            g.add_bond(p[i], p[i + 1])
        # Direct bond should NOT be a global bridge
        bridges = g.find_bridges()
        assert frozenset((direct.a._id, direct.b._id)) not in _bond_ids(bridges)
        # Direct bond SHOULD be a local bridge (alt path length > 3)
        local_bridges = g.find_local_bridges(max_dist=3)
        assert frozenset((direct.a._id, direct.b._id)) in _bond_ids(local_bridges)

    def test_max_dist_2_triangle(self):
        """In a triangle (3-ring), no bond is a local bridge at max_dist=2
        (each bond has alt path of length 2)."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1, 0, 0]), 'C', 6)
        a3 = g.add_atom(np.array([0.5, 0.87, 0]), 'C', 6)
        g.add_bond(a1, a2)
        g.add_bond(a2, a3)
        g.add_bond(a3, a1)
        lb = g.find_local_bridges(max_dist=2)
        assert len(lb) == 0

    def test_max_dist_2_no_common_neighbor(self):
        """A-B-C chain: bond A-B has no common neighbor with B (C is B's neighbor
        but not A's) → local bridge at max_dist=2? No: alt path A-?-B has length 2
        only if A and B share a neighbor. They don't, so it IS a local bridge at max_dist=2."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        a3 = g.add_atom(np.array([2.84, 0, 0]), 'C', 6)
        g.add_bond(a1, a2)
        g.add_bond(a2, a3)
        lb = g.find_local_bridges(max_dist=2)
        # Both bonds have no alternate path of length <= 2
        assert len(lb) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 05. Combined / Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestCombined:

    def test_bridges_are_subset_of_local_bridges(self):
        """Every global bridge is also a local bridge (no alt path at all)."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=5, cy=0)
        chain = _make_chain(g, [3, 0, 0], 2)
        g.add_bond(r1[0], chain[0])
        g.add_bond(chain[-1], r2[0])
        bridges = g.find_bridges()
        local_bridges = g.find_local_bridges(max_dist=3)
        bridge_ids = _bond_ids(bridges)
        local_ids = _bond_ids(local_bridges)
        assert bridge_ids.issubset(local_ids), \
            f"Bridges {bridge_ids} not subset of local bridges {local_ids}"

    def test_disconnected_fragments_with_bridge(self):
        """3 disconnected fragments: 2 rings + 1 chain, with one ring-chain connection."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=10, cy=0)  # disconnected
        chain = _make_chain(g, [3, 0, 0], 3)
        g.add_bond(r1[0], chain[0])
        # Components: {r1 + chain} and {r2}
        comps = g.find_connected_components()
        assert len(comps) == 2
        # Bridges: all chain bonds + ring-chain bond
        bridges = g.find_bridges()
        assert len(bridges) == 3  # r1[0]-chain[0], chain[0]-chain[1], chain[1]-chain[2]


# ═══════════════════════════════════════════════════════════════════════════════
# 06. Biconnected Components
# ═══════════════════════════════════════════════════════════════════════════════

class TestBiconnectedComponents:

    def test_empty_graph(self):
        g = AtomicGraph()
        assert g.find_biconnected_components() == []

    def test_single_bond(self):
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        g.add_bond(a1, a2)
        blocks = g.find_biconnected_components()
        assert len(blocks) == 1
        assert len(blocks[0][1]) == 1  # 1 bond

    def test_ring_one_block(self):
        """A 6-ring is a single biconnected component."""
        g = AtomicGraph()
        _make_hex_ring(g)
        blocks = g.find_biconnected_components()
        assert len(blocks) == 1
        assert len(blocks[0][0]) == 6  # 6 atoms
        assert len(blocks[0][1]) == 6  # 6 bonds

    def test_two_rings_connected_by_bond(self):
        """Two rings + bridge bond = 3 blocks (2 ring blocks + 1 bridge block)."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=5, cy=0)
        g.add_bond(r1[0], r2[0])
        blocks = g.find_biconnected_components()
        assert len(blocks) == 3
        bond_counts = sorted([len(b) for _, b in blocks])
        assert bond_counts == [1, 6, 6]

    def test_fused_rings_one_block(self):
        """Naphthalene (fused rings) = 1 biconnected component."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        atoms2 = [r1[1], r1[2]]
        for i in range(4):
            ang = (2 + i + 1) * np.pi / 3
            pos = np.array([np.cos(ang) * 1.42, np.sin(ang) * 1.42, 0.0])
            atoms2.append(g.add_atom(pos, 'C', 6, npi=1))
        g.add_bond(atoms2[1], atoms2[2])
        g.add_bond(atoms2[2], atoms2[3])
        g.add_bond(atoms2[3], atoms2[4])
        g.add_bond(atoms2[4], atoms2[5])
        g.add_bond(atoms2[5], atoms2[0])
        blocks = g.find_biconnected_components()
        assert len(blocks) == 1
        assert len(blocks[0][0]) == 10  # naphthalene has 10 C atoms

    def test_disconnected_rings(self):
        """Two disconnected rings = 2 blocks."""
        g = AtomicGraph()
        _make_hex_ring(g, cx=0, cy=0)
        _make_hex_ring(g, cx=10, cy=0)
        blocks = g.find_biconnected_components()
        assert len(blocks) == 2
        assert all(len(b) == 6 for _, b in blocks)


# ═══════════════════════════════════════════════════════════════════════════════
# 07. Fragments (split by bridges)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFragments:

    def test_empty_graph(self):
        g = AtomicGraph()
        frags, cuts = g.find_fragments()
        assert frags == []
        assert cuts == []

    def test_single_ring_one_fragment(self):
        """A single ring has no bridges → 1 fragment, 0 cuts."""
        g = AtomicGraph()
        _make_hex_ring(g)
        frags, cuts = g.find_fragments()
        assert len(frags) == 1
        assert len(cuts) == 0
        assert len(frags[0]) == 6

    def test_two_rings_split_into_two(self):
        """Two rings connected by a bridge bond → 2 fragments, 1 cut."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=5, cy=0)
        connecting = g.add_bond(r1[0], r2[0])
        frags, cuts = g.find_fragments()
        assert len(frags) == 2
        assert len(cuts) == 1
        assert cuts[0] is connecting
        # Each fragment should have 6 atoms
        assert sorted(len(f) for f in frags) == [6, 6]

    def test_ch_bonds_not_bridges(self):
        """C-H bonds must not be considered bridges."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6, npi=1)
        h1 = g.add_atom(np.array([0, 1, 0]), 'H', 1)
        g.add_bond(a1, h1)
        a2 = g.add_atom(np.array([3, 0, 0]), 'C', 6, npi=1)
        h2 = g.add_atom(np.array([3, 1, 0]), 'H', 1)
        g.add_bond(a2, h2)
        g.add_bond(a1, a2)
        frags, cuts = g.find_fragments()
        # Single fragment: C-C is a bridge but min_size=2, each side has 1 heavy → merged
        assert len(frags) == 1
        assert len(cuts) == 0

    def test_min_size_merges_small_fragments(self):
        """Two rings connected by bond: min_size=2 keeps both, min_size=7 merges smaller ring."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)   # 6 heavy atoms
        r2 = _make_hex_ring(g, cx=5, cy=0)   # 6 heavy atoms
        g.add_bond(r1[0], r2[0])
        # With min_size=2: both rings (6 >= 2) stay separate → 2 fragments, 1 cut
        frags2, cuts2 = g.find_fragments(min_size=2)
        assert len(frags2) == 2
        assert len(cuts2) == 1
        # With min_size=7: both rings (6 < 7) try to merge → 1 fragment, 0 cuts
        frags7, cuts7 = g.find_fragments(min_size=7)
        assert len(frags7) == 1
        assert len(cuts7) == 0

    def test_three_rings_chain(self):
        """Ring-chain-ring: chain bonds are bridges, splits into 3 fragments."""
        g = AtomicGraph()
        r1 = _make_hex_ring(g, cx=0, cy=0)
        r2 = _make_hex_ring(g, cx=10, cy=0)
        c1 = g.add_atom(np.array([5, 0, 0]), 'C', 6, npi=1)
        g.add_bond(r1[0], c1)
        g.add_bond(c1, r2[0])
        frags, cuts = g.find_fragments(min_size=2)
        # c1 is 1 heavy atom → merged into one of the rings with min_size=2
        # So we get 2 fragments (ring1+c1, ring2) and 1 cut
        assert len(frags) == 2
        assert len(cuts) == 1
