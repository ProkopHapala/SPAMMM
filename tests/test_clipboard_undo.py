#!/usr/bin/env python3
"""
Headless test for clipboard copy/paste and undo system.

Tests PackedMolecule-based copy/paste logic (without Qt clipboard) and
UndoStack push/pop with graph restoration.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spammm.topology.AtomicGraph import AtomicGraph
from spammm.topology.KekuleBackend import KekuleBackend
from spammm.topology.PackedMolecule import PackedMolecule, UndoStack, _z_to_ename

DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug', 'test_clipboard_undo')
os.makedirs(DEBUG_DIR, exist_ok=True)


def build_benzene_backend():
    kb = KekuleBackend()
    a_CC = 1.42
    angles = np.arange(6) * (np.pi / 3.0)
    carbons = []
    for i in range(6):
        pos = np.array([a_CC * np.cos(angles[i]), a_CC * np.sin(angles[i]), 0.0])
        c = kb.graph.add_atom(pos, 'C', 6, pin=None, parent=None, npi=1)
        carbons.append(c)
    for i in range(6):
        kb.graph.add_bond(carbons[i], carbons[(i + 1) % 6])
    for i in range(6):
        hpos = carbons[i].pos * 1.8
        kb.graph.add_atom(hpos, 'H', 1, pin=None, parent=carbons[i], npi=-1)
    kb.graph.sync_neighbor_lists()
    kb._sync_sys()
    return kb


def get_summary(kb):
    atom_list, enames, apos, atypes, bonds, _, _ = kb.graph.to_arrays()
    return len(atom_list), len(bonds), list(enames), apos.copy(), bonds.copy()


def test_copy_paste():
    print("=== Test 1: Copy/paste with PackedMolecule ===")
    kb = build_benzene_backend()
    n0, nb0, es0, pos0, bonds0 = get_summary(kb)
    print(f"Original: {n0} atoms, {nb0} bonds")

    # Simulate copy: select first 3 C atoms (indices 0,1,2)
    packed = PackedMolecule.from_graph(kb.graph, atom_indices=[0, 1, 2])
    print(f"Copied: {packed} (etype={list(packed.etype)}, bonds={packed.bonds.tolist()})")
    # Should have 3 C atoms and 2 bonds (0-1, 1-2)
    assert len(packed.etype) == 3, f"Expected 3 atoms, got {len(packed.etype)}"
    assert all(z == 6 for z in packed.etype), "All should be C"
    assert len(packed.bonds) == 2, f"Expected 2 internal bonds, got {len(packed.bonds)}"

    # Simulate paste: add atoms from packed to graph
    new_atoms = []
    for i in range(len(packed.etype)):
        z = int(packed.etype[i])
        ename = _z_to_ename(z)
        npi_i = int(packed.npi[i])
        pos = list(packed.apos[i].copy())
        a = kb._append_atom(pos=pos, ename=ename, pin=None, parent=None, npi=npi_i)
        new_atoms.append(a)
    for col in packed.bonds:
        i, j = int(col[0]), int(col[1])
        kb.graph.add_bond(new_atoms[i], new_atoms[j])
    kb.graph.sync_neighbor_lists()
    kb._sync_sys()

    n1, nb1, es1, pos1, bonds1 = get_summary(kb)
    print(f"After paste: {n1} atoms, {nb1} bonds")
    assert n1 == n0 + 3, f"Expected {n0+3} atoms, got {n1}"
    assert nb1 == nb0 + 2, f"Expected {nb0+2} bonds, got {nb1}"
    print("  PASSED ✓")


def test_undo_stack():
    print("\n=== Test 2: UndoStack (push → mutate → undo) ===")
    kb = build_benzene_backend()
    stack = UndoStack(maxlen=50)
    n0, nb0, es0, pos0, bonds0 = get_summary(kb)
    print(f"Initial: {n0} atoms, {nb0} bonds")

    # Push state before mutation
    stack.push(PackedMolecule.from_graph(kb.graph))
    assert len(stack) == 1

    # Mutate: delete 2 atoms
    atom_list = [a for a in kb.graph.atoms.values() if a.alive]
    kb.graph.remove_atom(atom_list[0])
    kb.graph.remove_atom(atom_list[1])
    kb.graph.sync_neighbor_lists()
    kb._sync_sys()
    n1, nb1, es1, pos1, bonds1 = get_summary(kb)
    print(f"After delete: {n1} atoms, {nb1} bonds")
    assert n1 < n0, "Atom count should decrease"

    # Undo: restore from stack
    packed = stack.pop()
    assert packed is not None, "Stack should not be empty"
    kb.graph = packed.to_graph()
    kb._sync_sys()
    n2, nb2, es2, pos2, bonds2 = get_summary(kb)
    print(f"After undo: {n2} atoms, {nb2} bonds")
    assert n2 == n0, f"Atom count should be restored to {n0}, got {n2}"
    assert nb2 == nb0, f"Bond count should be restored to {nb0}, got {nb2}"
    assert np.allclose(pos2, pos0, atol=1e-4), "Positions should match original"
    print("  PASSED ✓")


def test_undo_multiple():
    print("\n=== Test 3: Multiple undo levels ===")
    kb = build_benzene_backend()
    stack = UndoStack(maxlen=50)
    n0, nb0, es0, pos0, bonds0 = get_summary(kb)

    # Push state, then add an atom
    stack.push(PackedMolecule.from_graph(kb.graph))
    kb.graph.add_atom(np.array([5.0, 5.0, 0.0]), 'C', 6, pin=None, parent=None, npi=1)
    kb.graph.sync_neighbor_lists()
    kb._sync_sys()
    n1, _, _, _, _ = get_summary(kb)
    assert n1 == n0 + 1

    # Push state, then add another atom
    stack.push(PackedMolecule.from_graph(kb.graph))
    kb.graph.add_atom(np.array([6.0, 6.0, 0.0]), 'C', 6, pin=None, parent=None, npi=1)
    kb.graph.sync_neighbor_lists()
    kb._sync_sys()
    n2, _, _, _, _ = get_summary(kb)
    assert n2 == n0 + 2

    # Undo once → should go back to n1
    packed = stack.pop()
    kb.graph = packed.to_graph()
    kb._sync_sys()
    n3, _, _, _, _ = get_summary(kb)
    assert n3 == n1, f"Expected {n1} after 1 undo, got {n3}"

    # Undo again → should go back to n0
    packed = stack.pop()
    kb.graph = packed.to_graph()
    kb._sync_sys()
    n4, _, _, _, _ = get_summary(kb)
    assert n4 == n0, f"Expected {n0} after 2 undos, got {n4}"

    # Stack should be empty
    assert stack.pop() is None, "Stack should be empty"
    print("  PASSED ✓")


def test_undo_rolling_buffer():
    print("\n=== Test 4: Rolling buffer (maxlen=3) ===")
    stack = UndoStack(maxlen=3)
    g = AtomicGraph()
    for i in range(5):
        g.add_atom(np.array([i, 0, 0]), 'C', 6, npi=1)
        stack.push(PackedMolecule.from_graph(g))
    assert len(stack) == 3, f"Expected 3 (maxlen), got {len(stack)}"
    # Pop should give the last state (5 atoms)
    packed = stack.pop()
    assert len(packed.etype) == 5, f"Expected 5 atoms, got {len(packed.etype)}"
    # Pop should give 4 atoms
    packed = stack.pop()
    assert len(packed.etype) == 4, f"Expected 4 atoms, got {len(packed.etype)}"
    # Pop should give 3 atoms (oldest surviving)
    packed = stack.pop()
    assert len(packed.etype) == 3, f"Expected 3 atoms, got {len(packed.etype)}"
    # Stack empty
    assert stack.pop() is None
    print("  PASSED ✓")


def test_plot():
    print("\n=== Generating debug plot ===")
    kb = build_benzene_backend()
    _, es0, pos0, _, bonds0, _, _ = kb.graph.to_arrays()

    # Simulate undo: push, delete, restore
    stack = UndoStack(maxlen=10)
    stack.push(PackedMolecule.from_graph(kb.graph))
    atom_list = [a for a in kb.graph.atoms.values() if a.alive]
    kb.graph.remove_atom(atom_list[0])
    kb.graph.sync_neighbor_lists()
    kb._sync_sys()
    _, es_del, pos_del, _, bonds_del, _, _ = kb.graph.to_arrays()

    packed = stack.pop()
    kb.graph = packed.to_graph()
    kb._sync_sys()
    _, es_undo, pos_undo, _, bonds_undo, _, _ = kb.graph.to_arrays()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, apos, bonds, es) in zip(axes, [
        ("Original", pos0, bonds0, es0),
        ("After delete", pos_del, bonds_del, es_del),
        ("After undo", pos_undo, bonds_undo, es_undo),
    ]):
        ax.set_title(title)
        ax.set_aspect('equal')
        colors = ['gray' if e == 'H' else 'black' for e in es]
        ax.scatter(apos[:, 0], apos[:, 1], c=colors, s=100, zorder=5)
        for col in bonds:
            i, j = int(col[0]), int(col[1])
            ax.plot([apos[i, 0], apos[j, 0]], [apos[i, 1], apos[j, 1]], 'b-', zorder=1)
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
    plt.tight_layout()
    fname = os.path.join(DEBUG_DIR, 'clipboard_undo.png')
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved {fname}")


if __name__ == '__main__':
    test_copy_paste()
    test_undo_stack()
    test_undo_multiple()
    test_undo_rolling_buffer()
    test_plot()
    print("\n=== ALL TESTS PASSED ✓ ===")
