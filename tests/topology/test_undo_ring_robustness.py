#!/usr/bin/env python3
"""
L0 tests for GUI Editor Ring Drawing & Undo Robustness (task: GUI_Editor_RingDrawing_Robustness).

Verifies that EditorSnapshot preserves full graph fidelity (IDs, pins, parents,
charges, bond orders, float64 positions, hex_tiles) and that undo + add_ring
no longer produces overlapping disconnected topology.

These tests are designed to FAIL on the old PackedMolecule-based undo and PASS
after the EditorSnapshot fix.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend, OVERLAP_TOL
from spammm.topology.AtomicGraph import AtomicGraph, Atom, Bond
from spammm.topology.PackedMolecule import EditorSnapshot
from spammm import elements


# ── Helpers ──────────────────────────────────────────────────────────────────

def _alive_heavy(graph):
    return [a for a in graph.atoms.values() if a.alive and a.npi != -1]

def _alive_h(graph):
    return [a for a in graph.atoms.values() if a.alive and a.npi == -1]

def _min_heavy_dist(graph):
    heavy = _alive_heavy(graph)
    if len(heavy) < 2: return float('inf')
    dmin = float('inf')
    for i in range(len(heavy)):
        for j in range(i + 1, len(heavy)):
            d = np.linalg.norm(heavy[i].pos - heavy[j].pos)
            if d < dmin: dmin = d
    return dmin

def _min_h_to_nonparent_heavy_dist(graph):
    h_atoms = _alive_h(graph)
    heavy = _alive_heavy(graph)
    dmin = float('inf')
    for h in h_atoms:
        for a in heavy:
            if a is h.parent: continue
            d = np.linalg.norm(h.pos - a.pos)
            if d < dmin: dmin = d
    return dmin


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEditorSnapshotFidelity:
    """EditorSnapshot must preserve every authoritative graph field."""

    def test_01_snapshot_preserves_ids_pins_parents_charges_bondorders(self):
        """Round-trip: graph → EditorSnapshot → graph preserves all fields."""
        b = MoleculeEditorBackend()
        b.add_ring(0, 0)  # C6H6 with pins, H caps, aromatic bonds
        # Set a non-default charge and bond order to verify fidelity
        heavy = _alive_heavy(b.graph)
        heavy[0].charge = 0.5
        bonds = [bd for bd in b.graph.bonds.values() if bd.alive]
        bonds[0].order = 2.0
        b.hex_tiles.add((0, 0))
        # Capture pre-snapshot state
        pre_atoms = {a._id: (a.ename, a.atype, a.pos.copy(), a.pin, a.parent._id if a.parent else -1, a.npi, a.charge)
                     for a in b.graph.atoms.values() if a.alive}
        pre_bonds = {bd._id: (bd.a._id, bd.b._id, bd.order) for bd in b.graph.bonds.values() if bd.alive}
        pre_hex = set(b.hex_tiles)
        # Snapshot + restore
        snap = EditorSnapshot.from_graph(b.graph, b.hex_tiles)
        g2, hex2 = snap.to_graph()
        # Verify atoms
        for aid, (ename, atype, pos, pin, pid, npi, charge) in pre_atoms.items():
            a2 = g2.atoms.get(aid)
            assert a2 is not None, f"Atom {aid} missing after restore"
            assert a2.ename == ename
            assert a2.atype == atype
            assert np.allclose(a2.pos, pos), f"Atom {aid} pos mismatch: {a2.pos} vs {pos}"
            assert a2.pin == pin, f"Atom {aid} pin mismatch: {a2.pin} vs {pin}"
            expected_parent = g2.atoms.get(pid) if pid >= 0 else None
            assert a2.parent is expected_parent or (a2.parent is not None and a2.parent._id == pid), \
                f"Atom {aid} parent mismatch: got {a2.parent}, expected _id={pid}"
            assert a2.npi == npi
            assert abs(a2.charge - charge) < 1e-12, f"Atom {aid} charge mismatch: {a2.charge} vs {charge}"
        # Verify bonds
        for bid, (aid_a, aid_b, order) in pre_bonds.items():
            bd2 = g2.bonds.get(bid)
            assert bd2 is not None, f"Bond {bid} missing after restore"
            assert {bd2.a._id, bd2.b._id} == {aid_a, aid_b}
            assert abs(bd2.order - order) < 1e-12, f"Bond {bid} order mismatch: {bd2.order} vs {order}"
        # Verify hex_tiles
        assert hex2 == pre_hex, f"hex_tiles mismatch: {hex2} vs {pre_hex}"
        # Verify pin cache rebuilt
        assert len(g2._pin_to_atom) == len([a for a in g2.atoms.values() if a.alive and a.pin is not None])

    def test_02_float64_precision_preserved(self):
        """Positions must survive round-trip at float64 precision (not float32)."""
        b = MoleculeEditorBackend()
        b.add_ring(0, 0)
        # Set a position with sub-float32 resolution differences
        a = _alive_heavy(b.graph)[0]
        a.pos = np.array([1.0 + 1e-7, 2.0 + 3e-7, 0.0], dtype=np.float64)
        snap = EditorSnapshot.from_graph(b.graph, b.hex_tiles)
        g2, _ = snap.to_graph()
        a2 = g2.atoms.get(a._id)
        assert np.allclose(a2.pos, a.pos, atol=1e-12), f"float64 precision lost: {a2.pos} vs {a.pos}"

    def test_03_pin_cache_duplicate_rejected(self):
        """rebuild_pin_cache_from_atoms must raise on duplicate pins."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6, pin=(0.0, 0.0), npi=1)
        a2 = g.add_atom(np.array([1, 0, 0]), 'C', 6, pin=(1.0, 0.0), npi=1)
        # Force duplicate pin
        a2.pin = (0.0, 0.0)
        g._pin_to_atom.clear()
        with pytest.raises(RuntimeError, match="duplicate pin"):
            g.rebuild_pin_cache_from_atoms()


class TestUndoRingRobustness:
    """The primary reproduction: undo + add_ring must not create overlapping atoms."""

    def test_10_undo_then_adjacent_ring_produces_naphthalene(self):
        """add_ring(0,0) → snapshot/restore → add_ring(1,0) = C10H8, not overlapping C12H12."""
        b = MoleculeEditorBackend()
        b.add_ring(0, 0)
        heavy_before = len(_alive_heavy(b.graph))
        assert heavy_before == 6
        # Snapshot (simulates _push_undo before a mutation)
        snap = EditorSnapshot.from_graph(b.graph, b.hex_tiles)
        # Simulate a mutation (add free atom) then undo
        b._append_atom(pos=[10.0, 10.0, 0.0], ename='C', pin=None, npi=1)
        b._sync_sys()
        # Undo
        g2, hex2 = snap.to_graph()
        b.graph = g2
        b.hex_tiles = hex2
        b._rings_dirty = True
        b._sync_sys()
        # After undo, pins must be restored
        assert len(b.graph._pin_to_atom) == 6, f"Pin cache empty after undo: {len(b.graph._pin_to_atom)}"
        # Draw adjacent ring
        b.add_ring(1, 0)
        # Must be naphthalene C10H8, not overlapping C12
        heavy = _alive_heavy(b.graph)
        assert len(heavy) == 10, f"Expected 10 heavy atoms (naphthalene), got {len(heavy)}"
        h = _alive_h(b.graph)
        assert len(h) == 8, f"Expected 8 H caps, got {len(h)}"
        # No overlapping atoms
        dmin = _min_heavy_dist(b.graph)
        assert dmin > OVERLAP_TOL, f"Heavy atoms overlap: min dist={dmin:.4f} Å"
        # Graph validation passes
        b.graph.validate(overlap_tol=OVERLAP_TOL)

    def test_11_undo_then_same_ring_no_duplicates(self):
        """Undo then re-draw the SAME ring must not duplicate atoms."""
        b = MoleculeEditorBackend()
        b.add_ring(0, 0)
        snap = EditorSnapshot.from_graph(b.graph, b.hex_tiles)
        # Mutate + undo
        b._append_atom(pos=[10.0, 10.0, 0.0], ename='C', pin=None, npi=1)
        g2, hex2 = snap.to_graph()
        b.graph = g2
        b.hex_tiles = hex2
        b._sync_sys()
        # Re-draw same ring — should be idempotent (all 6 nodes already pinned)
        b.add_ring(0, 0)
        heavy = _alive_heavy(b.graph)
        assert len(heavy) == 6, f"Re-drawing same ring after undo duplicated atoms: got {len(heavy)}"
        dmin = _min_heavy_dist(b.graph)
        assert dmin > OVERLAP_TOL, f"Heavy atoms overlap: min dist={dmin:.4f} Å"

    def test_12_h_caps_not_inside_bonds_after_undo_redraw(self):
        """After undo + adjacent ring, no H cap should be inside a C-C bond."""
        b = MoleculeEditorBackend()
        b.add_ring(0, 0)
        snap = EditorSnapshot.from_graph(b.graph, b.hex_tiles)
        b._append_atom(pos=[10.0, 10.0, 0.0], ename='C', pin=None, npi=1)
        g2, hex2 = snap.to_graph()
        b.graph = g2
        b.hex_tiles = hex2
        b._sync_sys()
        b.add_ring(1, 0)
        a_CC = b.a_CC
        dmin_h = _min_h_to_nonparent_heavy_dist(b.graph)
        # H should not be closer than ~half a C-C bond to a non-parent heavy atom
        assert dmin_h > a_CC * 0.5, f"H cap too close to non-parent heavy atom: {dmin_h:.4f} Å (a_CC*0.5={a_CC*0.5:.4f})"


class TestOverlapGuard:
    """The fail-fast overlap guard in add_ring."""

    def test_20_overlap_guard_raises_on_stale_cache(self):
        """If pin cache is stale but a heavy atom sits at a node, add_ring must raise."""
        b = MoleculeEditorBackend()
        b.add_ring(0, 0)
        # Simulate stale pin cache (as would happen with the old lossy undo)
        b.graph._pin_to_atom.clear()
        # Now add_ring should detect the overlap and raise
        with pytest.raises(RuntimeError, match="overlaps existing"):
            b.add_ring(0, 0)


class TestGraphValidation:
    """The validate() helper catches corruption."""

    def test_30_validate_catches_overlapping_atoms(self):
        g = AtomicGraph()
        g.add_atom(np.array([0, 0, 0]), 'C', 6, npi=1)
        g.add_atom(np.array([0.1, 0, 0]), 'C', 6, npi=1)  # 0.1 Å apart
        with pytest.raises(RuntimeError, match="overlap"):
            g.validate(overlap_tol=0.3)

    def test_31_validate_catches_dead_parent(self):
        g = AtomicGraph()
        c = g.add_atom(np.array([0, 0, 0]), 'C', 6, npi=1)
        h = g.add_atom(np.array([0, 1, 0]), 'H', 1, parent=c, npi=-1)
        g.remove_atom(c)  # soft-delete parent
        with pytest.raises(RuntimeError, match="dead parent"):
            g.validate()

    def test_32_validate_passes_on_clean_naphthalene(self):
        b = MoleculeEditorBackend()
        b.add_ring(0, 0)
        b.add_ring(1, 0)
        b.graph.validate(overlap_tol=OVERLAP_TOL)  # should not raise


class TestNormalPathRegression:
    """Normal (no-undo) ring drawing still works."""

    def test_40_adjacent_ring_without_undo(self):
        """add_ring(0,0) + add_ring(1,0) without undo = C10H8 (regression check)."""
        b = MoleculeEditorBackend()
        b.add_ring(0, 0)
        b.add_ring(1, 0)
        heavy = _alive_heavy(b.graph)
        assert len(heavy) == 10, f"Expected 10 heavy atoms, got {len(heavy)}"
        h = _alive_h(b.graph)
        assert len(h) == 8, f"Expected 8 H caps, got {len(h)}"
        b.graph.validate(overlap_tol=OVERLAP_TOL)

    def test_41_identity_preserved_across_undo(self):
        """Atom._id values that existed before undo remain valid after undo."""
        b = MoleculeEditorBackend()
        b.add_ring(0, 0)
        pre_ids = {a._id for a in b.graph.atoms.values() if a.alive}
        snap = EditorSnapshot.from_graph(b.graph, b.hex_tiles)
        b._append_atom(pos=[10.0, 10.0, 0.0], ename='C', pin=None, npi=1)
        g2, hex2 = snap.to_graph()
        b.graph = g2
        b.hex_tiles = hex2
        b._sync_sys()
        post_ids = {a._id for a in b.graph.atoms.values() if a.alive}
        assert pre_ids == post_ids, f"Identity lost: missing={pre_ids - post_ids}, extra={post_ids - pre_ids}"
