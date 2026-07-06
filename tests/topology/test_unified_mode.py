"""
test_unified_mode.py — Headless tests for Unified mode backend methods and pick priority.

Tests:
  - cycle_atom_type: C→N→O→C cycling
  - cycle_bond_order: 1.0→1.5→2.0→3.0→1.0 cycling
  - resolve_unified_target: pick priority (atom > bond > hex > empty)
  - Simulated click dispatch (LMB on atom/bond/hex/empty, RMB on atom/bond/hex)

Run:
  pytest tests/topology/test_unified_mode.py -v
"""

import pytest
import numpy as np
from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend


class TestCycleAtomType:
    """Test cycle_atom_type: C→N→O→C."""

    def test_cycle_C_to_N(self):
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        c_atom = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C'][0]
        b.cycle_atom_type(c_atom._id)
        assert c_atom.ename == 'N'

    def test_cycle_N_to_O(self):
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        c_atom = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C'][0]
        b.cycle_atom_type(c_atom._id)  # C→N
        b.cycle_atom_type(c_atom._id)  # N→O
        assert c_atom.ename == 'O'

    def test_cycle_O_back_to_C(self):
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        c_atom = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C'][0]
        b.cycle_atom_type(c_atom._id)  # C→N
        b.cycle_atom_type(c_atom._id)  # N→O
        b.cycle_atom_type(c_atom._id)  # O→C
        assert c_atom.ename == 'C'

    def test_cycle_full_loop(self):
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        c_atom = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C'][0]
        original = c_atom.ename
        for _ in range(3):
            b.cycle_atom_type(c_atom._id)
        assert c_atom.ename == original

    def test_cycle_dead_atom_noop(self):
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        c_atom = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C'][0]
        b.graph.remove_atom(c_atom)
        b.cycle_atom_type(c_atom._id)  # should be no-op
        assert not c_atom.alive


class TestCycleBondOrder:
    """Test cycle_bond_order: 1.0→1.5→2.0→3.0→1.0."""

    def test_cycle_aromatic_to_double(self):
        """add_ring creates sp2 atoms → bonds default to order=1.5 (aromatic)."""
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        assert bond.order == 1.5  # sp2-sp2 bonds default to aromatic
        b.cycle_bond_order(bond)
        assert bond.order == 2.0

    def test_cycle_double_to_triple(self):
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        b.cycle_bond_order(bond)  # 1.5→2.0
        b.cycle_bond_order(bond)  # 2.0→3.0
        assert bond.order == 3.0

    def test_cycle_triple_back_to_aromatic(self):
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        b.cycle_bond_order(bond)  # 1.5→2.0
        b.cycle_bond_order(bond)  # 2.0→3.0
        b.cycle_bond_order(bond)  # 3.0→1.0
        b.cycle_bond_order(bond)  # 1.0→1.5
        assert bond.order == 1.5

    def test_cycle_from_single(self):
        """Test cycling starting from a single bond (order=1.0)."""
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        bond.order = 1.0  # force single
        b.cycle_bond_order(bond)
        assert bond.order == 1.5

    def test_cycle_dead_bond_noop(self):
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        b.graph.remove_bond(bond)
        original_order = bond.order
        b.cycle_bond_order(bond)
        assert bond.order == original_order  # unchanged


class TestResolveUnifiedTarget:
    """Test pick priority: atom > bond > hex > empty.

    These tests simulate the resolve_unified_target logic from SPAMMM_GUI
    using only the backend, no Qt/Vispy required.
    """

    def _resolve(self, backend, p_world, pick_radius=0.5):
        """Replicate resolve_unified_target logic without GUI."""
        atom = backend.pick_atom(p_world, radius=pick_radius)
        if atom: return ('atom', atom)
        bond = backend.pick_bond(p_world, radius=pick_radius)
        if bond: return ('bond', bond)
        if hasattr(backend, 'snap_to_ring'):
            q, r = backend.snap_to_ring(p_world[0], p_world[1])
            # Check distance to hex center — snap_to_ring always returns nearest
            from spammm.topology.HexGrid import snap_to_grid
            ring_nodes = backend.grid.ring_nodes(q, r)
            center = np.mean([snap_to_grid(n)[:2] for n in ring_nodes], axis=0)
            dist = np.linalg.norm(center - p_world[:2])
            if dist < backend.a_CC * 0.5: return ('hex', (q, r))
        return ('empty', p_world)

    def test_pick_atom_over_bond(self):
        """When cursor is on an atom that's also near a bond midpoint, atom wins."""
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        atom = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C'][0]
        t_type, t = self._resolve(b, atom.pos)
        assert t_type == 'atom'
        assert t._id == atom._id

    def test_pick_bond_when_no_atom(self):
        """When cursor is on bond midpoint (not near any atom), bond wins."""
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        center = (bond.a.pos + bond.b.pos) / 2
        # Move slightly off the atoms to avoid atom pick — center is equidistant
        t_type, t = self._resolve(b, center, pick_radius=0.3)
        assert t_type == 'bond'

    def test_pick_hex_when_no_atom_no_bond(self):
        """When cursor is far from atoms/bonds but near hex center, hex wins."""
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        # Hex center is at the cog of the ring — find it
        b._rings_dirty = True
        b.detect_geometry_rings()
        ring = list(b.graph.rings.values())[0]
        cog = ring.cog
        # Use small pick radius so we don't pick atoms/bonds
        t_type, t = self._resolve(b, cog, pick_radius=0.01)
        assert t_type == 'hex'

    def test_pick_empty_far_away(self):
        """When cursor is far from any hex center, empty wins.
        The hex grid tiles all of 2D space, so 'far' means between hex centers
        (e.g. at a vertex position that's > a_CC*0.5 from any center)."""
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        # Position (1.0, 0.0) is at a hexagon vertex — ~1.0 Å from nearest center
        far = np.array([1.0, 0.0, 0.0])
        t_type, t = self._resolve(b, far, pick_radius=0.01)
        assert t_type == 'empty'


class TestSimulatedUnifiedClicks:
    """Simulate Unified mode click dispatch using backend operations directly.

    These tests replicate what on_mouse_press does in Unified mode,
    but call backend methods directly (no Qt/Vispy event objects).
    """

    def _setup(self):
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        b.add_ring(0, 0)
        return b

    def test_lmb_on_atom_cycles_type(self):
        """LMB on atom → cycle_atom_type."""
        b = self._setup()
        atom = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C'][0]
        assert atom.ename == 'C'
        b.cycle_atom_type(atom._id)
        assert atom.ename == 'N'

    def test_lmb_on_bond_cycles_order(self):
        """LMB on bond → cycle_bond_order."""
        b = self._setup()
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        initial_order = bond.order  # 1.5 for sp2-sp2 ring bonds
        b.cycle_bond_order(bond)
        assert bond.order != initial_order

    def test_lmb_on_hex_adds_ring(self):
        """LMB on hex center → add_ring (new adjacent ring)."""
        b = self._setup()
        n_before = len([a for a in b.graph.atoms.values() if a.alive])
        b.add_ring(1, 0)
        n_after = len([a for a in b.graph.atoms.values() if a.alive])
        assert n_after > n_before

    def test_lmb_on_empty_adds_atom(self):
        """LMB on empty space → _append_atom."""
        b = self._setup()
        n_before = len([a for a in b.graph.atoms.values() if a.alive])
        b._append_atom(pos=[10.0, 10.0, 0.0], ename='C', pin=None, parent=None, npi=1)
        n_after = len([a for a in b.graph.atoms.values() if a.alive])
        assert n_after == n_before + 1

    def test_rmb_on_atom_deletes(self):
        """RMB on atom → remove_atom_by_id."""
        b = self._setup()
        atom = [a for a in b.graph.atoms.values() if a.alive and a.ename == 'C'][0]
        atom_id = atom._id
        b.remove_atom_by_id(atom_id)
        assert not b.graph.atoms[atom_id].alive

    def test_rmb_on_bond_deletes(self):
        """RMB on bond → delete_bond."""
        b = self._setup()
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        bond_id = bond._id
        b.delete_bond(bond)
        assert not b.graph.bonds[bond_id].alive

    def test_rmb_on_hex_removes_ring(self):
        """RMB on hex center → remove_ring."""
        b = self._setup()
        n_before = len([a for a in b.graph.atoms.values() if a.alive])
        b.remove_ring(0, 0)
        n_after = len([a for a in b.graph.atoms.values() if a.alive])
        assert n_after < n_before

    def test_ctrl_lmb_on_bond_inserts_atom(self):
        """Ctrl+LMB on bond → insert_atom_into_bond."""
        b = self._setup()
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        n_atoms_before = len([a for a in b.graph.atoms.values() if a.alive])
        new_atom = b.insert_atom_into_bond(bond, 'C', push_aside=True)
        n_atoms_after = len([a for a in b.graph.atoms.values() if a.alive])
        assert n_atoms_after == n_atoms_before + 1
        assert new_atom.alive

    def test_ctrl_rmb_on_bond_collapses(self):
        """Ctrl+RMB on bond → collapse_bond."""
        b = self._setup()
        bond = [bd for bd in b.graph.bonds.values() if bd.alive][0]
        n_atoms_before = len([a for a in b.graph.atoms.values() if a.alive])
        center = (bond.a.pos + bond.b.pos) / 2
        survivor = b.collapse_bond(bond, center[:2])
        n_atoms_after = len([a for a in b.graph.atoms.values() if a.alive])
        assert n_atoms_after == n_atoms_before - 1
        assert survivor.alive

    def test_full_unified_workflow(self):
        """End-to-end: add ring → cycle atom → cycle bond → add atom → delete bond."""
        b = MoleculeEditorBackend()
        b.auto_h_cap = False
        # 1. Add hex ring
        b.add_ring(0, 0)
        atoms = [a for a in b.graph.atoms.values() if a.alive]
        bonds = [bd for bd in b.graph.bonds.values() if bd.alive]
        assert len(atoms) == 6
        assert len(bonds) == 6
        # 2. Cycle atom type C→N
        atom = atoms[0]
        b.cycle_atom_type(atom._id)
        assert atom.ename == 'N'
        # 3. Cycle bond order (ring bonds start at 1.5=aromatic → 2.0=double)
        bond = bonds[0]
        b.cycle_bond_order(bond)
        assert bond.order == 2.0
        # 4. Add free atom
        b._append_atom(pos=[5.0, 5.0, 0.0], ename='O', pin=None, parent=None, npi=1)
        assert len([a for a in b.graph.atoms.values() if a.alive]) == 7
        # 5. Delete a bond
        b.delete_bond(bonds[1])
        assert not bonds[1].alive
