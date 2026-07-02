#!/usr/bin/env python3
"""
Headless test for multi-format export/import via KekuleBackend.

Builds a benzene-like molecule in AtomicGraph, exports to .xyz/.mol/.mol2,
then imports each back and verifies topology matches.
"""

import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spammm.topology.KekuleBackend import KekuleBackend
from spammm.topology.AtomicGraph import AtomicGraph

DEBUG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'debug', 'test_export_import')
os.makedirs(DEBUG_DIR, exist_ok=True)


def build_benzene_backend():
    """Build a KekuleBackend with a benzene-like structure."""
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


def get_topology_summary(kb):
    """Return (natoms, nbonds, enames, apos, bonds) for comparison."""
    atom_list, enames, apos, atypes, bonds, _, _ = kb.graph.to_arrays()
    return len(atom_list), len(bonds), list(enames), apos.copy(), bonds.copy()


def test_xyz_export_import():
    print("=== Test 1: XYZ export/import ===")
    kb = build_benzene_backend()
    n0, nb0, es0, pos0, bonds0 = get_topology_summary(kb)
    print(f"Original: {n0} atoms, {nb0} bonds")

    fname = os.path.join(DEBUG_DIR, 'test.xyz')
    kb.save_structure(fname)
    print(f"Saved to {fname} ({os.path.getsize(fname)} bytes)")

    kb2 = KekuleBackend()
    kb2.load_structure(fname)
    n1, nb1, es1, pos1, bonds1 = get_topology_summary(kb2)
    print(f"Loaded: {n1} atoms, {nb1} bonds")

    assert n1 == n0, f"atom count mismatch: {n1} vs {n0}"
    # XYZ has no bonds — load_xyz uses _create_bond_to_nearest_heavy heuristic
    # So bonds may differ. Just check atoms match.
    assert np.allclose(pos1, pos0, atol=1e-3), "positions mismatch!"
    print("  PASSED ✓ (atoms match; bonds use heuristic for XYZ)")


def test_mol_export_import():
    print("\n=== Test 2: MOL export/import ===")
    kb = build_benzene_backend()
    n0, nb0, es0, pos0, bonds0 = get_topology_summary(kb)
    print(f"Original: {n0} atoms, {nb0} bonds, enames={es0}")

    fname = os.path.join(DEBUG_DIR, 'test.mol')
    kb.save_structure(fname)
    print(f"Saved to {fname} ({os.path.getsize(fname)} bytes)")
    with open(fname) as f:
        print("  File contents (first 300 chars):")
        print(f.read()[:300])

    kb2 = KekuleBackend()
    kb2.load_structure(fname)
    n1, nb1, es1, pos1, bonds1 = get_topology_summary(kb2)
    print(f"Loaded: {n1} atoms, {nb1} bonds, enames={es1}")

    # MOL preserves bonds between heavy atoms
    assert n1 == n0, f"atom count mismatch: {n1} vs {n0}"
    # Check heavy atom positions match
    heavy_mask0 = np.array([e not in ('H', 'E') for e in es0])
    heavy_mask1 = np.array([e not in ('H', 'E') for e in es1])
    pos0_heavy = pos0[heavy_mask0]
    pos1_heavy = pos1[heavy_mask1]
    assert len(pos0_heavy) == len(pos1_heavy), f"heavy atom count mismatch: {len(pos0_heavy)} vs {len(pos1_heavy)}"
    assert np.allclose(pos1_heavy, pos0_heavy, atol=1e-3), "heavy atom positions mismatch!"
    # MOL should preserve bonds (6 ring bonds between C atoms)
    assert nb1 == nb0, f"bond count mismatch: {nb1} vs {nb0}"
    print("  PASSED ✓")


def test_mol2_export_import():
    print("\n=== Test 3: MOL2 export/import ===")
    kb = build_benzene_backend()
    n0, nb0, es0, pos0, bonds0 = get_topology_summary(kb)
    print(f"Original: {n0} atoms, {nb0} bonds, enames={es0}")

    fname = os.path.join(DEBUG_DIR, 'test.mol2')
    kb.save_structure(fname)
    print(f"Saved to {fname} ({os.path.getsize(fname)} bytes)")
    with open(fname) as f:
        print("  File contents (first 400 chars):")
        print(f.read()[:400])

    kb2 = KekuleBackend()
    kb2.load_structure(fname)
    n1, nb1, es1, pos1, bonds1 = get_topology_summary(kb2)
    print(f"Loaded: {n1} atoms, {nb1} bonds, enames={es1}")

    assert n1 == n0, f"atom count mismatch: {n1} vs {n0}"
    heavy_mask0 = np.array([e not in ('H', 'E') for e in es0])
    heavy_mask1 = np.array([e not in ('H', 'E') for e in es1])
    pos0_heavy = pos0[heavy_mask0]
    pos1_heavy = pos1[heavy_mask1]
    assert len(pos0_heavy) == len(pos1_heavy), f"heavy atom count mismatch: {len(pos0_heavy)} vs {len(pos1_heavy)}"
    assert np.allclose(pos1_heavy, pos0_heavy, atol=1e-3), "heavy atom positions mismatch!"
    assert nb1 == nb0, f"bond count mismatch: {nb1} vs {nb0}"
    print("  PASSED ✓")


def test_plot():
    print("\n=== Generating debug plot ===")
    kb = build_benzene_backend()
    _, es0, pos0, _, bonds0, _, _ = kb.graph.to_arrays()

    # Export to MOL and re-import
    fname = os.path.join(DEBUG_DIR, 'plot_test.mol')
    kb.save_structure(fname)
    kb2 = KekuleBackend()
    kb2.load_structure(fname)
    _, es1, pos1, _, bonds1, _, _ = kb2.graph.to_arrays()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, apos, bonds, es) in zip(axes, [
        ("Original", pos0, bonds0, es0),
        ("MOL round-trip", pos1, bonds1, es1),
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
    fname_plot = os.path.join(DEBUG_DIR, 'export_import.png')
    plt.savefig(fname_plot, dpi=100)
    plt.close()
    print(f"  Saved {fname_plot}")


if __name__ == '__main__':
    test_xyz_export_import()
    test_mol_export_import()
    test_mol2_export_import()
    test_plot()
    print("\n=== ALL TESTS PASSED ✓ ===")
