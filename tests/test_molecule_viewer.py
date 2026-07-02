"""
test_molecule_viewer.py — Test MoleculeViewer standalone and offscreen rendering.

Tests:
  1. Load a molecule file and verify data arrays
  2. Render offscreen and verify image shape/dtype
  3. Test auto_fit doesn't crash
  4. Test label modes
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spammm.GUI.MoleculeViewer import MoleculeViewer, default_atom_colors, default_atom_sizes, compute_bond_segments


def test_load_and_data():
	"""Test loading a molecule file sets data correctly."""
	viewer = MoleculeViewer(bgcolor='white', show=False)
	system = viewer.load_file('data/xyz/benzene.xyz')
	assert len(viewer._pos) > 0, "No atoms loaded"
	assert viewer._pos.shape[1] == 3, f"Expected (n,3) shape, got {viewer._pos.shape}"
	assert viewer._colors is not None, "Colors not set"
	assert viewer._sizes is not None, "Sizes not set"
	assert viewer._enames is not None, "Enames not set"
	print(f"  OK: loaded {len(viewer._pos)} atoms, {len(viewer._bonds) if viewer._bonds is not None else 0} bonds")
	return viewer


def test_offscreen_render():
	"""Test offscreen rendering produces valid RGBA image."""
	viewer = MoleculeViewer(bgcolor='white', show=False)
	viewer.load_file('data/xyz/H2O.xyz')
	img = viewer.render_offscreen(size=256)
	assert img is not None, "render_offscreen returned None"
	assert img.shape == (256, 256, 4), f"Expected (256,256,4), got {img.shape}"
	assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"
	# Check it's not all background (has some non-white pixels)
	non_white = np.any(img[:, :, :3] < 250, axis=2).sum()
	assert non_white > 10, f"Image appears empty (only {non_white} non-white pixels)"
	print(f"  OK: rendered {img.shape} image, {non_white} non-white pixels")


def test_auto_fit():
	"""Test auto_fit doesn't crash and sets reasonable camera distance."""
	viewer = MoleculeViewer(bgcolor='white', show=False)
	viewer.load_file('data/xyz/PTCDA.xyz')
	viewer.auto_fit()
	d = viewer.view.camera.distance
	assert d > 0, f"Camera distance should be positive, got {d}"
	print(f"  OK: auto_fit set distance={d:.1f}")


def test_label_modes():
	"""Test label mode switching."""
	viewer = MoleculeViewer(bgcolor='white', show=False)
	viewer.load_file('data/xyz/CH4.xyz')
	viewer.set_label_mode('Element+Index')
	assert viewer.text_labels.visible, "Labels should be visible in Element+Index mode"
	viewer.set_label_mode('none')
	assert not viewer.text_labels.visible, "Labels should be hidden in none mode"
	print("  OK: label modes work")


def test_default_colors():
	"""Test default_atom_colors returns valid RGBA."""
	enames = ['C', 'H', 'O', 'N']
	cols = default_atom_colors(enames)
	assert cols.shape == (4, 4), f"Expected (4,4), got {cols.shape}"
	assert (cols[:, 3] == 1.0).all(), "Alpha should be 1.0"
	print(f"  OK: default_atom_colors for {enames} -> {cols[:, :3].tolist()}")


def test_bond_segments():
	"""Test compute_bond_segments."""
	pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
	bonds = np.array([[0, 1], [0, 2]])
	segs = compute_bond_segments(pos, bonds)
	assert segs.shape == (4, 3), f"Expected (4,3), got {segs.shape}"
	np.testing.assert_array_equal(segs[0], pos[0])
	np.testing.assert_array_equal(segs[1], pos[1])
	print("  OK: bond segments computed correctly")


def test_empty_bonds():
	"""Test compute_bond_segments with None bonds."""
	pos = np.array([[0, 0, 0]], dtype=np.float32)
	segs = compute_bond_segments(pos, None)
	assert segs.shape == (0, 3), f"Expected (0,3), got {segs.shape}"
	print("  OK: empty bonds handled")


def run_all():
	tests = [
		("default_atom_colors", test_default_colors),
		("compute_bond_segments", test_bond_segments),
		("empty_bonds", test_empty_bonds),
		("load_and_data", test_load_and_data),
		("offscreen_render", test_offscreen_render),
		("auto_fit", test_auto_fit),
		("label_modes", test_label_modes),
	]
	passed = 0
	failed = 0
	for name, fn in tests:
		try:
			print(f"[TEST] {name}...")
			fn()
			passed += 1
		except Exception as e:
			print(f"  FAIL: {e}")
			import traceback
			traceback.print_exc()
			failed += 1
	print(f"\n{'='*40}")
	print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
	return failed == 0


if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	success = run_all()
	sys.exit(0 if success else 1)
