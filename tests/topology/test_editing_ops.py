"""
test_editing_ops.py — Headless tests for all molecular editing operations.

Two-layer testing:
  L1 (always): TopologySnapshot diff assertions — fast, deterministic, no display
  L2 (--visual): before/after PNG rendering for human review

Run:
  pytest tests/topology/test_editing_ops.py                    # L1 only
  pytest tests/topology/test_editing_ops.py --visual           # L1 + L2
  pytest tests/topology/test_editing_ops.py --visual -k ring   # filter to ring tests

Visual output goes to: debug/test_editing_ops/
"""

import os
import pytest
import numpy as np
import copy

from spammm.topology.KekuleBackend import KekuleBackend
from spammm.topology.AtomicGraph import AtomicGraph, Atom, Bond
from spammm import elements
from tests.helpers.topology_test import TopologySnapshot, TopologyDiff, render_before_after


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_graph_copy(backend):
    """Deep-copy the graph for before/after rendering (graph is mutable)."""
    return copy.deepcopy(backend.graph)


def _maybe_render(visual_output_dir, test_name, graph_before, backend,
                  highlight_atoms=None, cursor_pos=None, diff=None,
                  title_before='Before', title_after='After'):
    """Render before/after PNG if --visual flag is set."""
    if visual_output_dir is None:
        return
    savepath = os.path.join(visual_output_dir, f'{test_name}.png')
    render_before_after(
        graph_before, backend.graph, savepath,
        title_before=title_before, title_after=title_after,
        highlight_atoms=highlight_atoms, cursor_pos=cursor_pos, diff=diff,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 01. Atom-Level Operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestAddAtom:
    """Add atom at grid position / arbitrary position."""

    def test_01_add_atom_at_grid_position(self, visual_output_dir):
        """Add single C atom at grid origin (0,0).

        Demonstrates: set_atom_type on empty grid node creates new atom.
        Before: empty graph. After: 1 C atom at origin.
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.set_atom_type((0.0, 0.0), 'C')

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        diff.assert_counts('add_atom_grid', added_atoms=1, added_bonds=0)
        _maybe_render(visual_output_dir, 'test_01_add_atom_grid',
                      graph_before, b, diff=diff,
                      title_before='Before: empty graph',
                      title_after='After: 1 C atom at origin')

    def test_02_add_atom_at_arbitrary_position(self, visual_output_dir):
        """Add C atom at off-grid position (1.0, 2.0, 0.0).

        Demonstrates: add_atom_at_position creates atom not on hex grid.
        Before: empty graph. After: 1 C atom at (1,2).
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.add_atom_at_position([1.0, 2.0, 0.0], 'C', npi=1)

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        diff.assert_counts('add_atom_pos', added_atoms=1)
        _maybe_render(visual_output_dir, 'test_02_add_atom_pos',
                      graph_before, b, diff=diff,
                      title_before='Before: empty graph',
                      title_after='After: 1 C atom at (1,2)')


class TestRemoveAtom:
    """Remove atom by grid key / by index."""

    def test_03_remove_atom_by_grid_key(self, visual_output_dir):
        """Remove C atom from benzene by grid key (simulates right-click on atom).

        Demonstrates: remove_atom deletes atom + its bonds.
        Before: benzene C6 (6 atoms, 6 bonds). After: 5 atoms, 4 bonds (1 atom + 2 bonds removed).
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        heavy = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C']
        node_key = heavy[0].pin
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.remove_atom(node_key)

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.removed_atoms) >= 1, f'Expected >=1 removed atoms, got {diff.removed_atoms}'
        assert len(diff.removed_bonds) >= 2, f'Expected >=2 removed bonds, got {diff.removed_bonds}'
        _maybe_render(visual_output_dir, 'test_03_remove_atom',
                      graph_before, b, diff=diff,
                      highlight_atoms={heavy[0]._id},
                      cursor_pos=tuple(heavy[0].pos[:2]),
                      title_before='Before: benzene C6 (click on atom to remove)',
                      title_after='After: 1 C atom removed (5 atoms, 4 bonds)')

    def test_04_remove_atom_by_index(self, visual_output_dir):
        """Remove atom by index (simulates programmatic deletion).

        Demonstrates: remove_atom_by_index removes atom at index 0.
        Before: benzene C6. After: 5 atoms, 4 bonds.
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.remove_atom_by_index(0)

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.removed_atoms) >= 1
        _maybe_render(visual_output_dir, 'test_04_remove_atom_by_idx',
                      graph_before, b, diff=diff,
                      title_before='Before: benzene C6',
                      title_after='After: atom[0] removed (5 atoms, 4 bonds)')


class TestChangeElementType:
    """Change element type at grid node."""

    def test_05_change_element_C_to_N(self, visual_output_dir):
        """Change one C atom to N in benzene (simulates element picker in GUI).

        Demonstrates: set_atom_type on existing atom changes element, no topology change.
        Before: benzene C6 (all black). After: C5N (one blue N atom at same position).
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        heavy = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C']
        node_key = heavy[0].pin
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.set_atom_type(node_key, 'N')

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        diff.assert_counts('change_element', added_atoms=0, removed_atoms=0)
        a = b.graph._pin_to_atom.get(node_key)
        assert a.ename == 'N', f'Expected N, got {a.ename}'
        _maybe_render(visual_output_dir, 'test_05_change_element',
                      graph_before, b, diff=diff,
                      highlight_atoms={heavy[0]._id},
                      title_before='Before: benzene C6 (highlight = atom to change)',
                      title_after='After: C→N (blue atom is nitrogen)')


class TestChangeHybridization:
    """Change hybridization (npi)."""

    def test_06_change_hybridization_sp2_to_sp3(self, visual_output_dir):
        """Change one C from sp2 to sp3 in benzene (adds extra H cap).

        Demonstrates: set_atom_valency changes hybridization, H caps adjusted.
        Before: C_sp2 with 1 H cap (CH). After: C_sp3 with 2 H caps (CH2).
        """
        b = KekuleBackend()
        b.add_ring(0, 0)  # sp2 with auto H caps
        heavy = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C']
        node_key = heavy[0].pin
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.set_atom_valency(node_key, 0)  # sp3

        snap_after = TopologySnapshot(b.graph)
        a = b.graph._pin_to_atom.get(node_key)
        assert 'sp3' in a.subtype, f'Expected sp3 subtype, got {a.subtype}'
        _maybe_render(visual_output_dir, 'test_06_change_hybridization',
                      graph_before, b,
                      highlight_atoms={heavy[0]._id},
                      title_before='Before: C_sp2 (1 H cap, CH group)',
                      title_after='After: C_sp3 (2 H caps, CH2 group)')


# ═══════════════════════════════════════════════════════════════════════════════
# 02. Bond-Level Operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestInsertAtomIntoBond:
    """Insert atom into bond (A-B → A-C-B)."""

    def test_07_insert_atom_into_bond(self, visual_output_dir):
        """Insert C atom into middle of a C-C bond in benzene.

        Demonstrates: insert_atom_into_bond splits bond A-B into A-C-B.
        Before: benzene with 6 C-C bonds. After: 7 atoms, original bond replaced by 2 new bonds.
        The new C atom (green) sits at bond midpoint, original atoms pushed aside.
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        bonds = [bd for bd in b.graph.bonds.values() if bd.alive]
        bond = bonds[0]
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        new_atom = b.insert_atom_into_bond(bond, 'C')

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        diff.assert_counts('insert_into_bond',
                           added_atoms=1, removed_atoms=0,
                           added_bonds=2, removed_bonds=1)
        _maybe_render(visual_output_dir, 'test_07_insert_into_bond',
                      graph_before, b, diff=diff,
                      cursor_pos=tuple((bond.a.pos + bond.b.pos)[:2] / 2.0),
                      title_before='Before: benzene (cursor = bond midpoint to insert at)',
                      title_after='After: C atom inserted into bond (A-C-B, green=new atom)')


class TestCollapseBond:
    """Collapse bond (A-B → survivor, neighbor bonds transferred)."""

    def test_08_collapse_bond(self, visual_output_dir):
        """Collapse a C-C bond in benzene — one atom removed, survivor moves to center.

        Demonstrates: collapse_bond merges two atoms into one. Atom farther from mouse survives.
        Before: benzene C6. After: 5 atoms, survivor at bond midpoint, neighbor bonds transferred.
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        bonds = [bd for bd in b.graph.bonds.values() if bd.alive]
        bond = bonds[0]
        atom_a, atom_b = bond.a, bond.b
        mouse_pos = atom_a.pos[:2] + np.array([0.01, 0.01])
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        survivor = b.collapse_bond(bond, mouse_pos)

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.removed_atoms) >= 1, f'Expected >=1 removed, got {diff.removed_atoms}'
        assert len(diff.removed_bonds) >= 1, f'Expected >=1 removed bonds, got {diff.removed_bonds}'
        _maybe_render(visual_output_dir, 'test_08_collapse_bond',
                      graph_before, b, diff=diff,
                      cursor_pos=tuple(mouse_pos),
                      title_before='Before: benzene (cursor near atom A, A will be removed)',
                      title_after='After: bond collapsed (B survives at midpoint, A removed)')


class TestAddRemoveBond:
    """Add/remove bond directly on AtomicGraph."""

    def test_09_add_bond(self):
        """Add bond between two atoms on AtomicGraph."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        assert len(g.bonds) == 0
        b = g.add_bond(a1, a2)
        assert len(g.bonds) == 1
        assert b.a is a1 and b.b is a2

    def test_10_add_bond_idempotent(self):
        """Adding same bond twice returns existing bond."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        b1 = g.add_bond(a1, a2)
        b2 = g.add_bond(a1, a2)
        assert b1 is b2, 'add_bond should return existing bond'

    def test_11_remove_bond(self):
        """Remove bond from AtomicGraph."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        b = g.add_bond(a1, a2)
        g.remove_bond(b)
        g.cleanup_invalid()
        assert len(g.bonds) == 0


class TestRecalcBonds:
    """Recalculate all bonds from distance."""

    def test_12_recalc_bonds(self, visual_output_dir):
        """Remove all bonds from benzene, then recalc from distances.

        Demonstrates: recalc_bonds rebuilds bond topology from atom positions.
        Before: 6 C atoms with no bonds (just positions). After: 6 C-C bonds restored.
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        for bd in list(b.graph.bonds.values()):
            b.graph.remove_bond(bd)
        b.graph.cleanup_invalid()
        assert len(b.graph.bonds) == 0
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.recalc_bonds()

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.added_bonds) > 0, 'recalc_bonds should create bonds'
        _maybe_render(visual_output_dir, 'test_12_recalc_bonds',
                      graph_before, b, diff=diff,
                      title_before='Before: 6 C atoms, no bonds',
                      title_after='After: recalc_bonds restored 6 C-C bonds (blue=new bonds)')


# ═══════════════════════════════════════════════════════════════════════════════
# 03. Ring/Hex Operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestAddRing:
    """Add hex ring at axial position."""

    def test_13_add_ring_single(self, visual_output_dir):
        """Add single benzene ring at (0,0) — creates C6H6.

        Demonstrates: add_ring creates 6 C atoms in hexagon + 6 H caps + bonds.
        Before: empty graph. After: benzene C6H6 (12 atoms, 12 bonds).
        """
        b = KekuleBackend()
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.add_ring(0, 0)

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        diff.assert_counts('add_ring', added_atoms=12, added_bonds=12)
        assert (0, 0) in b.hex_tiles
        _maybe_render(visual_output_dir, 'test_13_add_ring',
                      graph_before, b, diff=diff,
                      title_before='Before: empty graph',
                      title_after='After: benzene C6H6 (green=new atoms)')

    def test_14_add_ring_adjacent_shares_atoms(self, visual_output_dir):
        """Add ring at (1,0) adjacent to existing (0,0) — naphthalene.

        Demonstrates: adjacent hex rings share 2 atoms (not duplicated).
        Before: benzene C6 (1 ring). After: naphthalene C10 (2 fused rings, 2 shared atoms).
        """
        b = KekuleBackend()
        b.add_ring(0, 0)
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.add_ring(1, 0)

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.added_atoms) > 0, 'Adjacent ring should add atoms'
        heavy = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C']
        assert len(heavy) == 10, f'Expected 10 C atoms (2 shared), got {len(heavy)}'
        _maybe_render(visual_output_dir, 'test_14_add_ring_adjacent',
                      graph_before, b, diff=diff,
                      title_before='Before: benzene C6 (1 ring)',
                      title_after='After: naphthalene C10 (2 fused rings, green=new atoms)')


class TestRemoveRing:
    """Remove hex ring."""

    def test_15_remove_ring(self, visual_output_dir):
        """Remove ring at (0,0) — all 6 C atoms removed (Hex1 paint mode).

        Demonstrates: remove_ring in Hex1 mode removes all atoms at 6 nodes.
        Before: benzene C6. After: empty graph.
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.remove_ring(0, 0)

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.removed_atoms) == 6, f'Expected 6 removed, got {len(diff.removed_atoms)}'
        assert (0, 0) not in b.hex_tiles
        _maybe_render(visual_output_dir, 'test_15_remove_ring',
                      graph_before, b, diff=diff,
                      title_before='Before: benzene C6',
                      title_after='After: ring removed (empty graph)')

    def test_16_remove_ring_hex2_preserves_shared(self, visual_output_dir):
        """Remove ring in Hex2 (toggle) mode — shared atoms preserved.

        Demonstrates: Hex2 mode preserves atoms shared with adjacent rings.
        Before: naphthalene C10 (2 rings). After: benzene C6 (shared 2 atoms preserved, 4 removed).
        """
        b = KekuleBackend()
        b.hex_mode = 'Hex2'
        b.auto_h_cap = False
        b.add_ring(0, 0)
        b.add_ring(1, 0)
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.remove_ring(0, 0)

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.removed_atoms) < 6, f'Hex2 should preserve shared atoms, removed {len(diff.removed_atoms)}'
        _maybe_render(visual_output_dir, 'test_16_remove_ring_hex2',
                      graph_before, b, diff=diff,
                      title_before='Before: naphthalene C10 (2 rings, Hex2 mode)',
                      title_after='After: 1 ring removed, shared atoms preserved (C6 remains)')


class TestToggleRing:
    """Toggle ring (add if absent, remove if present)."""

    def test_17_toggle_ring_add_then_remove(self, visual_output_dir):
        """Toggle ring ON then OFF — demonstrates idempotent toggle.

        Demonstrates: toggle_ring adds if absent, removes if present.
        Before: benzene C6 (toggle ON). After: empty (toggle OFF).
        """
        b = KekuleBackend()
        b.auto_h_cap = False

        b.toggle_ring(0, 0)
        assert (0, 0) in b.hex_tiles
        graph_after_add = _save_graph_copy(b)
        snap_after_add = TopologySnapshot(b.graph)

        b.toggle_ring(0, 0)
        assert (0, 0) not in b.hex_tiles
        snap_after_remove = TopologySnapshot(b.graph)

        diff = snap_after_add.diff(snap_after_remove)
        assert len(diff.removed_atoms) == 6, f'Expected 6 removed on toggle off, got {len(diff.removed_atoms)}'
        _maybe_render(visual_output_dir, 'test_17_toggle_ring',
                      graph_after_add, b, diff=diff,
                      title_before='Before: toggle ON (benzene C6)',
                      title_after='After: toggle OFF (empty)')


class TestDetectRings:
    """Detect rings from bond graph."""

    def test_18_detect_rings_mixed(self, visual_output_dir):
        """Detect rings in a mixed graph: naphthalene + extra bonded atoms without rings.

        Demonstrates: detect_geometry_rings finds 6-membered rings from bond topology.
        Before: naphthalene (2 fused rings) + 2 extra C atoms bonded to edge (no ring).
        After: 2 rings detected (magenta hex outlines), extra atoms correctly excluded.
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        b.add_ring(1, 0)
        # Add 2 extra atoms bonded to an edge atom (not part of any ring)
        heavy = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C']
        edge_atom = heavy[0]
        extra1 = b.graph.add_atom(edge_atom.pos + np.array([0.0, 2.5, 0.0]), 'C', 6, subtype='C_sp2')
        b.graph.add_bond(edge_atom, extra1)
        extra2 = b.graph.add_atom(extra1.pos + np.array([1.42, 0.0, 0.0]), 'C', 6, subtype='C_sp2')
        b.graph.add_bond(extra1, extra2)
        b.graph.sync_neighbor_lists()
        # Clear any existing rings to show detection creating them
        for r in list(b.graph.rings.values()):
            r.alive = False
        b.graph.cleanup_invalid()
        graph_before = _save_graph_copy(b)

        b._rings_dirty = True
        rings = b.detect_geometry_rings()

        assert len(rings) >= 2, f'Expected >=2 detected rings, got {len(rings)}'
        for r in rings:
            assert len(r.atoms) == 6, f'Ring should have 6 atoms, got {len(r.atoms)}'
        _maybe_render(visual_output_dir, 'test_18_detect_rings',
                      graph_before, b,
                      title_before='Before: naphthalene + 2 extra C (no ring objects)',
                      title_after='After: 2 rings detected (magenta outlines), extra atoms excluded')

    def test_19_detect_rings_naphthalene(self):
        """Two adjacent rings → detect >=2 rings."""
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        b.add_ring(1, 0)
        b._rings_dirty = True
        rings = b.detect_geometry_rings()
        assert len(rings) >= 2, f'Expected >=2 rings for naphthalene, got {len(rings)}'


# ═══════════════════════════════════════════════════════════════════════════════
# 04. Hydrogen Cap Management
# ═══════════════════════════════════════════════════════════════════════════════

class TestHCaps:
    """H cap management."""

    def test_20_add_h_caps(self, visual_output_dir):
        """Add H caps to bare benzene C6 — each C gets 1 H (sp2, 2 heavy neighbors).

        Demonstrates: add_h_caps adds H atoms to undercoordinated heavy atoms.
        Before: benzene C6 (no H, 6 atoms). After: benzene C6H6 (12 atoms, 6 new H caps).
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.add_h_caps()

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.added_atoms) == 6, f'Expected 6 H caps, got {len(diff.added_atoms)}'
        _maybe_render(visual_output_dir, 'test_20_add_h_caps',
                      graph_before, b, diff=diff,
                      title_before='Before: benzene C6 (no H caps)',
                      title_after='After: C6H6 (6 H caps added, green=new H atoms)')

    def test_21_remove_h_caps(self, visual_output_dir):
        """Remove all H caps from benzene C6H6.

        Demonstrates: remove_h_caps strips all H_cap atoms.
        Before: benzene C6H6 (12 atoms). After: benzene C6 (6 atoms, H removed).
        """
        b = KekuleBackend()
        b.add_ring(0, 0)  # auto H caps
        h_before = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'H']
        assert len(h_before) > 0
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.remove_h_caps()

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.removed_atoms) == len(h_before), f'Expected {len(h_before)} removed, got {len(diff.removed_atoms)}'
        _maybe_render(visual_output_dir, 'test_21_remove_h_caps',
                      graph_before, b, diff=diff,
                      title_before='Before: benzene C6H6 (with H caps)',
                      title_after='After: C6 (H caps removed)')

    def test_22_adjust_h_fixes_missing(self, visual_output_dir):
        """Remove some H caps from benzene, then adjust_h restores them.

        Demonstrates: adjust_h = remove_h_caps + add_h_caps. Fixes missing/displaced H atoms.
        Before: benzene C6H6 with 3 H caps removed (C6H3, undercoordinated). After: C6H6 restored.
        """
        b = KekuleBackend()
        b.add_ring(0, 0)  # benzene C6H6 with auto H caps
        # Remove 3 H caps to simulate damaged structure
        h_atoms = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'H']
        assert len(h_atoms) == 6, f'Expected 6 H caps, got {len(h_atoms)}'
        for h in h_atoms[:3]:
            for bd in h.bonds:
                bd.alive = False
            h.alive = False
        b.graph.cleanup_invalid()
        b.graph.sync_neighbor_lists()
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)
        h_remaining = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'H']
        assert len(h_remaining) == 3, f'Expected 3 H after removal, got {len(h_remaining)}'

        b.adjust_h()

        snap_after = TopologySnapshot(b.graph)
        h_final = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'H']
        assert len(h_final) == 6, f'adjust_h should restore 6 H caps, got {len(h_final)}'
        assert snap_after.n_atoms == 12, f'Expected 12 atoms (C6H6), got {snap_after.n_atoms}'
        _maybe_render(visual_output_dir, 'test_22_adjust_h',
                      graph_before, b,
                      title_before='Before: benzene C6H3 (3 H caps removed, undercoordinated)',
                      title_after='After: adjust_h restored C6H6 (6 H caps, green=new H)')


# ═══════════════════════════════════════════════════════════════════════════════
# 05. Picking / Selection
# ═══════════════════════════════════════════════════════════════════════════════

class TestPicking:
    """Pick atom/bond/ring by position (simulates mouse click)."""

    def test_23_pick_atom(self):
        """Pick atom at its position — returns correct atom."""
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        heavy = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C'][0]
        picked = b.pick_atom(heavy.pos, radius=0.5)
        assert picked is not None
        assert picked._id == heavy._id

    def test_24_pick_atom_miss(self):
        """Pick at empty position — returns None."""
        b = KekuleBackend()
        b.add_ring(0, 0)
        picked = b.pick_atom(np.array([100.0, 100.0, 0.0]), radius=0.5)
        assert picked is None

    def test_25_pick_bond(self):
        """Pick bond at its midpoint — returns correct bond."""
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        bonds = [bd for bd in b.graph.bonds.values() if bd.alive]
        bond = bonds[0]
        center = (bond.a.pos + bond.b.pos) / 2
        picked = b.pick_bond(center, radius=0.5)
        assert picked is not None
        assert picked._id == bond._id

    def test_26_pick_ring(self):
        """Pick ring at its center — returns correct ring."""
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        b._rings_dirty = True
        b.detect_geometry_rings()
        rings = list(b.graph.rings.values())
        assert len(rings) > 0
        ring = rings[0]
        picked = b.pick_ring(ring.cog, radius=1.0)
        assert picked is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 06. Graph Maintenance
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphMaintenance:
    """cleanup, sync, neighbor lists, to_arrays."""

    def test_27_cleanup_invalid(self):
        """Soft-deleted atoms are removed by cleanup_invalid."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        g.add_bond(a1, a2)
        g.remove_atom(a2, soft=True)
        assert a2._id in g.atoms
        g.cleanup_invalid()
        assert a2._id not in g.atoms
        assert len(g.bonds) == 0

    def test_28_sync_neighbor_lists(self):
        """sync_neighbor_lists rebuilds from alive bonds."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        g.add_bond(a1, a2)
        g.sync_neighbor_lists()
        assert a2 in a1.neighbors
        assert a1 in a2.neighbors
        b = g.get_bond(a1, a2)
        g.remove_bond(b)
        g.cleanup_invalid()
        g.sync_neighbor_lists()
        assert a2 not in a1.neighbors
        assert a1 not in a2.neighbors

    def test_29_to_arrays(self):
        """to_arrays exports alive atoms/bonds only."""
        g = AtomicGraph()
        a1 = g.add_atom(np.array([0, 0, 0]), 'C', 6)
        a2 = g.add_atom(np.array([1.42, 0, 0]), 'C', 6)
        a3 = g.add_atom(np.array([2.84, 0, 0]), 'C', 6)
        g.add_bond(a1, a2)
        g.add_bond(a2, a3)
        g.remove_atom(a3, soft=True)
        g.cleanup_invalid()
        atom_list, enames, apos, atypes, bonds, bond_list, ring_list = g.to_arrays()
        assert len(atom_list) == 2
        assert len(bonds) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 07. I/O Operations
# ═══════════════════════════════════════════════════════════════════════════════

class TestIO:
    """Save/load XYZ roundtrip."""

    def test_30_xyz_roundtrip(self, tmp_path, visual_output_dir):
        """Save naphthalene to XYZ, load into fresh backend — verify topology reconstructed.

        Demonstrates: save_xyz writes atom positions, load_xyz reads them back and rebuilds graph.
        Before: original naphthalene C10H8 (with bonds). After: loaded from XYZ (atoms restored, bonds rebuilt via recalc).
        Shows that the XYZ format preserves atom positions and the graph can be reconstructed.
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        b.add_ring(1, 0)
        b.adjust_h()
        snap_original = TopologySnapshot(b.graph)
        graph_before = _save_graph_copy(b)

        fname = str(tmp_path / 'test.xyz')
        b.save_xyz(fname)

        b2 = KekuleBackend()
        b2.load_xyz(fname)
        # load_xyz loads both heavy and H atoms from XYZ
        # recalc_bonds to get full bond topology (load_xyz only creates nearest-heavy bonds)
        b2.recalc_bonds()

        snap_loaded = TopologySnapshot(b2.graph)
        assert snap_original.n_atoms == snap_loaded.n_atoms, \
            f'Atom count mismatch: {snap_original.n_atoms} vs {snap_loaded.n_atoms}'
        c_bonds_orig = sum(1 for p in snap_original.bonds if all(
            snap_original.atoms[aid][0] != 'H' for aid in p))
        c_bonds_recalc = sum(1 for p in snap_loaded.bonds if all(
            snap_loaded.atoms[aid][0] != 'H' for aid in p))
        assert c_bonds_orig == c_bonds_recalc, \
            f'C-C bond count mismatch: {c_bonds_orig} vs {c_bonds_recalc}'
        _maybe_render(visual_output_dir, 'test_30_xyz_roundtrip',
                      graph_before, b2,
                      title_before='Before: original naphthalene C10H8',
                      title_after='After: loaded from XYZ + recalc_bonds (reconstructed)')


# ═══════════════════════════════════════════════════════════════════════════════
# 08. Ribbon Builders
# ═══════════════════════════════════════════════════════════════════════════════

class TestRibbonBuilder:
    """Build zigzag ribbon."""

    def test_31_build_zigzag_ribbon(self, visual_output_dir):
        """Build finite zigzag graphene ribbon (3 chains, 3 cells) with H passivation via adjust_h.

        Demonstrates: build_zigzag_ribbon creates graphene strip, then adjust_h adds H caps
        using the same method as everywhere else (geometry-based, not fixed offsets).
        Before: empty graph. After: finite zigzag ribbon with C atoms + properly oriented H caps.
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        # Build ribbon without built-in passivation, use standard adjust_h instead
        b.build_zigzag_ribbon(width_chains=3, length_cells=3, passivation=None,
                              bPeriodicX=False, side_passivation=None)
        b._sync_sys()
        b.adjust_h()

        snap_after = TopologySnapshot(b.graph)
        diff = snap_before.diff(snap_after)
        assert len(diff.added_atoms) > 0, 'Ribbon should add atoms'
        c_atoms = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C']
        h_atoms = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'H']
        assert len(c_atoms) > 0, 'Should have C atoms'
        assert len(h_atoms) > 0, 'Should have H passivation'
        _maybe_render(visual_output_dir, 'test_31_zigzag_ribbon',
                      graph_before, b, diff=diff,
                      title_before='Before: empty graph',
                      title_after='After: zigzag ribbon + adjust_h (green=new, C=black, H=gray)')


# ═══════════════════════════════════════════════════════════════════════════════
# 09. Multi-Step Editing
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiStep:
    """Sequence of edits, verify final topology."""

    def test_32_build_pah_then_edit(self, visual_output_dir):
        """Build naphthalene, change one C to N (simulates heteroatom substitution).

        Demonstrates: multi-step edit — build structure, then change element.
        Before: naphthalene C10 (all carbon). After: C9N (one C replaced by N, H caps adjusted).
        """
        b = KekuleBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        b.add_ring(1, 0)
        heavy = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C']
        graph_before = _save_graph_copy(b)
        snap_before = TopologySnapshot(b.graph)

        b.set_atom_type(heavy[0].pin, 'N')
        b.adjust_h()

        snap_after = TopologySnapshot(b.graph)
        heavy_before = {aid: e for aid, (e, st) in snap_before.atoms.items() if e != 'H'}
        heavy_after  = {aid: e for aid, (e, st) in snap_after.atoms.items()  if e != 'H'}
        assert set(heavy_before.keys()) == set(heavy_after.keys()), 'Heavy atom IDs should be unchanged'
        assert heavy_after[heavy[0]._id] == 'N', f'Expected N, got {heavy_after[heavy[0]._id]}'
        _maybe_render(visual_output_dir, 'test_32_pah_edit',
                      graph_before, b,
                      highlight_atoms={heavy[0]._id},
                      title_before='Before: naphthalene C10 (highlight = atom to change)',
                      title_after='After: C→N substitution (blue=N, H caps adjusted)')

    def test_33_add_remove_add_ring(self, visual_output_dir):
        """Add ring, remove it, add it again — verify idempotent reconstruction.

        Demonstrates: add→remove→add cycle produces same topology.
        Before: benzene C6 (re-added after remove). After: same benzene C6 (identical counts).
        Shows that the graph correctly handles create/destroy/create cycles.
        """
        b = KekuleBackend()
        b.auto_h_cap = False

        b.add_ring(0, 0)
        snap_after_first_add = TopologySnapshot(b.graph)

        b.remove_ring(0, 0)
        snap_after_remove = TopologySnapshot(b.graph)
        assert snap_after_remove.n_atoms == 0, f'After remove: expected 0 atoms, got {snap_after_remove.n_atoms}'
        assert snap_after_remove.n_bonds == 0, f'After remove: expected 0 bonds, got {snap_after_remove.n_bonds}'

        b.add_ring(0, 0)
        snap_after_second_add = TopologySnapshot(b.graph)

        assert snap_after_first_add.n_atoms == snap_after_second_add.n_atoms
        assert snap_after_first_add.n_bonds == snap_after_second_add.n_bonds
        # For visual: show the final state vs empty (the "before" is after remove = empty)
        graph_empty = AtomicGraph()  # empty graph to represent the "after remove" state
        _maybe_render(visual_output_dir, 'test_33_add_remove_add',
                      graph_empty, b,
                      title_before='Before: empty (after add→remove cycle)',
                      title_after='After: re-added ring (same as first add)')


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Electron Pairs (AtomicSystem.add_electron_pairs)
# ═══════════════════════════════════════════════════════════════════════════════

# (name, xyz_file, expected_epairs, description)
_EPAIR_MOLECULES = [
    ('H2O',        'H2O.xyz',        2, 'O sp3: 2 lone pairs'),
    ('NH3',        'NH3.xyz',        1, 'N sp3: 1 lone pair'),
    ('CH2O',       'CH2O.xyz',       2, 'O sp2 (C=O): 2 lone pairs'),
    ('HCN',        'HCN.xyz',        1, 'N sp (C≡N): 1 lone pair'),
    ('pyridine',   'pyridine.xyz',   1, 'N sp2 aromatic: 1 lone pair'),
    ('HCOOH',      'HCOOH.xyz',      4, '2×O (C=O + C-O-H): 4 lone pairs'),
    ('CH2NH',      'CH2NH.xyz',      1, 'N sp2 (C=NH): 1 lone pair'),
    ('pyrrole',    'pyrrole.xyz',    1, 'N sp3 (pyrrolic NH): 1 lone pair'),
    ('formamide',  'formamide.xyz',  3, 'O sp2 (C=O) + N sp2 (C-NH2): 2+1 = 3 lone pairs'),
]


def _load_sys(xyz_path):
    """Load XYZ into AtomicSystem, find bonds, build neighbors."""
    from spammm.AtomicSystem import AtomicSystem
    sys = AtomicSystem(fname=xyz_path)
    sys.findBonds()
    sys.neighs()
    return sys


def _count_epairs(sys):
    return sum(1 for e in sys.enames if e == 'E')


def _save_combined_xyz(sys_before, sys_after, savepath):
    """Save before and after geometries into a single XYZ file (before first, then after)."""
    from spammm.atomicUtils import saveXYZ
    # Save before
    saveXYZ(sys_before.enames, sys_before.apos, savepath, qs=sys_before.qs, mode='w', comment='before add_electron_pairs')
    # Append after
    saveXYZ(sys_after.enames, sys_after.apos, savepath, qs=sys_after.qs, mode='a', comment='after add_electron_pairs')


class TestElectronPairs:
    """Test add_electron_pairs on various small molecules with N and O atoms.

    Parametrized over a list of molecules. For each:
    - L1: assert correct number of electron pairs (E atoms) added
    - L2 (--visual): save multi-projection PNG (xy, xz, yz × before/after) and combined XYZ
    """

    @pytest.mark.parametrize('name, xyz_file, expected_ep, desc', _EPAIR_MOLECULES)
    def test_epair(self, name, xyz_file, expected_ep, desc, xyz, visual_output_dir):
        """Add electron pairs to {name}: {desc}.

        Before: {name} from XYZ. After: {name} + {expected_ep} electron pairs (E atoms).
        """
        sys = _load_sys(xyz(xyz_file))
        n_before = len(sys.apos)
        sys_before = copy.deepcopy(sys)

        sys.add_electron_pairs()

        n_after = len(sys.apos)
        n_ep = _count_epairs(sys)
        assert n_ep == expected_ep, f'{name}: expected {expected_ep} electron pairs, got {n_ep} ({desc})'
        assert n_after == n_before + expected_ep, f'{name}: expected {n_before + expected_ep} atoms, got {n_after}'

        if visual_output_dir:
            from tests.helpers.topology_test import render_sys_multiview
            # Multi-projection PNG
            png_path = os.path.join(visual_output_dir, f'test_epair_{name}.png')
            render_sys_multiview(sys_before, sys, png_path,
                                 title_before=f'Before: {name} ({n_before} atoms)',
                                 title_after=f'After: {name} + {n_ep} EP ({n_after} atoms)')
            # Combined XYZ (before + after in one file)
            xyz_path = os.path.join(visual_output_dir, f'test_epair_{name}.xyz')
            _save_combined_xyz(sys_before, sys, xyz_path)
