#!/usr/bin/env python3
"""
Headless test for PackedMolecule: graph round-trip, text round-trip, npz save/load.

Builds a small molecule (benzene-like 6 C + 6 H), packs it, unpacks it,
and verifies all fields match. Outputs diagnostic plots to debug/.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spammm.topology.AtomicGraph import AtomicGraph
from spammm.topology.PackedMolecule import PackedMolecule, UndoStack

DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug', 'test_packed_molecule')
os.makedirs(DEBUG_DIR, exist_ok=True)


def build_benzene_graph():
    """Build a benzene-like graph: 6 C atoms in hexagon + 6 H caps."""
    g = AtomicGraph()
    a_CC = 1.42
    angles = np.arange(6) * (np.pi / 3.0)
    carbons = []
    for i in range(6):
        pos = np.array([a_CC * np.cos(angles[i]), a_CC * np.sin(angles[i]), 0.0])
        c = g.add_atom(pos, 'C', 6, pin=None, parent=None, npi=1)
        carbons.append(c)
    # Ring bonds
    for i in range(6):
        g.add_bond(carbons[i], carbons[(i + 1) % 6])
    # H caps
    for i in range(6):
        hpos = carbons[i].pos * 1.8  # push H outward
        h = g.add_atom(hpos, 'H', 1, pin=None, parent=carbons[i], npi=-1)
    g.sync_neighbor_lists()
    return g


def test_graph_roundtrip():
    print("=== Test 1: Graph round-trip ===")
    g_orig = build_benzene_graph()
    atom_list, enames, apos, atypes, bonds, _, _ = g_orig.to_arrays()
    print(f"Original: {len(atom_list)} atoms, {len(bonds)} bonds")
    print(f"  enames: {list(enames)}")
    print(f"  atypes: {list(atypes)}")
    print(f"  bonds:  {bonds.tolist()}")

    # Pack
    packed = PackedMolecule.from_graph(g_orig)
    print(f"Packed: {packed}")
    print(f"  etype: {list(packed.etype)}")
    print(f"  apos shape: {packed.apos.shape}")
    print(f"  bonds: {packed.bonds.tolist()}")
    print(f"  npi:   {list(packed.npi)}")

    # Verify etype matches atypes
    assert np.array_equal(packed.etype, atypes.astype(np.int32)), "etype mismatch!"
    # Verify positions match
    assert np.allclose(packed.apos, apos.astype(np.float32), atol=1e-4), "apos mismatch!"
    # Verify bonds match
    assert np.array_equal(packed.bonds, bonds.astype(np.int32)), "bonds mismatch!"
    # Verify npi: C atoms should be 1 (sp2), H atoms should be -1 (H_cap)
    expected_npi = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1], dtype=np.int8)
    assert np.array_equal(packed.npi, expected_npi), f"npi mismatch: {list(packed.npi)} vs {list(expected_npi)}"

    # Unpack to new graph
    g_new = packed.to_graph()
    atom_list2, enames2, apos2, atypes2, bonds2, _, _ = g_new.to_arrays()
    print(f"Restored: {len(atom_list2)} atoms, {len(bonds2)} bonds")
    print(f"  enames: {list(enames2)}")
    print(f"  bonds:  {bonds2.tolist()}")

    # Verify round-trip
    assert len(atom_list2) == len(atom_list), f"atom count mismatch: {len(atom_list2)} vs {len(atom_list)}"
    assert np.array_equal(enames2, enames), "enames mismatch after round-trip!"
    assert np.allclose(apos2, apos, atol=1e-4), "apos mismatch after round-trip!"
    assert len(bonds2) == len(bonds), f"bond count mismatch: {len(bonds2)} vs {len(bonds)}"

    # Verify H cap parent reconstruction
    h_atoms = [a for a in atom_list2 if a.ename == 'H']
    for h in h_atoms:
        assert h.parent is not None, f"H atom {h._id} has no parent!"
        assert h.parent.ename == 'C', f"H parent is {h.parent.ename}, expected C!"
        assert h.npi == -1, f"H npi is {h.npi}, expected -1 (H_cap)!"

    print("  PASSED ✓")
    return g_orig, packed


def test_subset_packing():
    print("\n=== Test 2: Subset packing (selected atoms only) ===")
    g = build_benzene_graph()
    atom_list, enames, apos, atypes, bonds, _, _ = g.to_arrays()

    # Select only the 6 C atoms (indices 0-5)
    packed_sub = PackedMolecule.from_graph(g, atom_indices=[0, 1, 2, 3, 4, 5])
    print(f"Subset packed: {packed_sub}")
    print(f"  etype: {list(packed_sub.etype)}")
    print(f"  bonds: {packed_sub.bonds.tolist()}")

    assert len(packed_sub.etype) == 6, f"Expected 6 atoms, got {len(packed_sub.etype)}"
    assert all(z == 6 for z in packed_sub.etype), "All should be C (Z=6)"
    # Should have 6 ring bonds (all internal)
    assert len(packed_sub.bonds) == 6, f"Expected 6 bonds, got {len(packed_sub.bonds)}"

    print("  PASSED ✓")


def test_text_roundtrip():
    print("\n=== Test 3: Text round-trip (XYZ and MOL) ===")
    g = build_benzene_graph()
    packed = PackedMolecule.from_graph(g)

    # XYZ text
    xyz_text = packed.to_xyz_text()
    print(f"XYZ text ({len(xyz_text)} chars):")
    print(xyz_text[:200])
    packed_xyz = PackedMolecule.from_text(xyz_text)
    assert packed_xyz is not None, "from_text returned None for XYZ!"
    print(f"From XYZ: {packed_xyz}")
    assert len(packed_xyz.etype) == len(packed.etype), "atom count mismatch in XYZ round-trip!"
    assert np.allclose(packed_xyz.apos, packed.apos, atol=1e-3), "apos mismatch in XYZ round-trip!"
    # XYZ has no bonds
    assert len(packed_xyz.bonds) == 0, "XYZ should have 0 bonds!"

    # MOL text
    mol_text = packed.to_mol_text()
    print(f"MOL text ({len(mol_text)} chars):")
    print(mol_text[:300])
    packed_mol = PackedMolecule.from_text(mol_text)
    assert packed_mol is not None, "from_text returned None for MOL!"
    print(f"From MOL: {packed_mol}")
    assert len(packed_mol.etype) == len(packed.etype), "atom count mismatch in MOL round-trip!"
    assert np.allclose(packed_mol.apos, packed.apos, atol=1e-3), "apos mismatch in MOL round-trip!"
    # MOL should have bonds
    assert len(packed_mol.bonds) == len(packed.bonds), f"bond count mismatch: {len(packed_mol.bonds)} vs {len(packed.bonds)}"

    print("  PASSED ✓")


def test_npz_save_load():
    print("\n=== Test 4: NPZ save/load ===")
    g = build_benzene_graph()
    packed = PackedMolecule.from_graph(g)
    fname = os.path.join(DEBUG_DIR, 'benzene.npz')
    packed.save_npz(fname)
    print(f"Saved to {fname} ({os.path.getsize(fname)} bytes)")
    packed_loaded = PackedMolecule.load_npz(fname)
    print(f"Loaded: {packed_loaded}")
    assert np.array_equal(packed.etype, packed_loaded.etype), "etype mismatch in NPZ!"
    assert np.allclose(packed.apos, packed_loaded.apos), "apos mismatch in NPZ!"
    assert np.array_equal(packed.bonds, packed_loaded.bonds), "bonds mismatch in NPZ!"
    assert np.array_equal(packed.npi, packed_loaded.npi), "npi mismatch in NPZ!"
    print("  PASSED ✓")


def test_undo_stack():
    print("\n=== Test 5: UndoStack ===")
    stack = UndoStack(maxlen=3)
    g = build_benzene_graph()
    p1 = PackedMolecule.from_graph(g)
    p2 = PackedMolecule.from_graph(g)
    p3 = PackedMolecule.from_graph(g)
    p4 = PackedMolecule.from_graph(g)

    stack.push(p1)
    stack.push(p2)
    stack.push(p3)
    stack.push(p4)  # should drop p1 (maxlen=3)
    assert len(stack) == 3, f"Expected 3, got {len(stack)}"
    popped = stack.pop()
    assert popped is p4, "Should pop p4 (last pushed)"
    popped = stack.pop()
    assert popped is p3, "Should pop p3"
    popped = stack.pop()
    assert popped is p2, "Should pop p2"
    popped = stack.pop()
    assert popped is None, "Stack should be empty"
    print("  PASSED ✓")


def test_plot():
    print("\n=== Generating debug plot ===")
    g = build_benzene_graph()
    packed = PackedMolecule.from_graph(g)
    g_restored = packed.to_graph()
    _, enames, apos_orig, _, bonds_orig, _, _ = g.to_arrays()
    _, enames2, apos_new, _, bonds_new, _, _ = g_restored.to_arrays()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, apos, bonds, es) in zip(axes, [
        ("Original", apos_orig, bonds_orig, enames),
        ("Restored", apos_new, bonds_new, enames2),
    ]):
        ax.set_title(title)
        ax.set_aspect('equal')
        colors = ['gray' if e == 'H' else 'black' for e in es]
        ax.scatter(apos[:, 0], apos[:, 1], c=colors, s=100, zorder=5)
        for i, e in enumerate(es):
            ax.annotate(e, (apos[i, 0], apos[i, 1]), fontsize=8, ha='center', va='bottom')
        for col in bonds:
            i, j = int(col[0]), int(col[1])
            ax.plot([apos[i, 0], apos[j, 0]], [apos[i, 1], apos[j, 1]], 'b-', zorder=1)
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
    plt.tight_layout()
    fname = os.path.join(DEBUG_DIR, 'roundtrip.png')
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"  Saved {fname}")


if __name__ == '__main__':
    g_orig, packed = test_graph_roundtrip()
    test_subset_packing()
    test_text_roundtrip()
    test_npz_save_load()
    test_undo_stack()
    test_plot()
    print("\n=== ALL TESTS PASSED ✓ ===")
