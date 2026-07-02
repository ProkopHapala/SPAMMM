"""
test_directory_navigator.py — Test DirectoryNavigator directory reading and navigation.
"""

import sys, os, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spammm.GUI.DirectoryNavigator import DirectoryNavigator


def test_read_dir():
	"""Test reading the data/xyz directory."""
	nav = DirectoryNavigator('data/xyz')
	assert len(nav.file_names) > 0, "No files found in data/xyz"
	assert all(f.endswith('.xyz') for f in nav.file_names), "All files should be .xyz"
	assert '..' in nav.sub_dirs, "Parent dir should be in sub_dirs"
	print(f"  OK: found {len(nav.file_names)} files, {len(nav.sub_dirs)} dirs in data/xyz")


def test_extensions():
	"""Test that only .xyz/.mol/.mol2 files are listed."""
	with tempfile.TemporaryDirectory() as tmpdir:
		# Create various files
		for name in ['mol1.xyz', 'mol2.mol2', 'mol3.mol', 'readme.txt', 'data.json', 'script.py']:
			open(os.path.join(tmpdir, name), 'w').close()
		# Create a subdirectory
		os.mkdir(os.path.join(tmpdir, 'subdir'))
		nav = DirectoryNavigator(tmpdir)
		assert len(nav.file_names) == 3, f"Expected 3 molecule files, got {len(nav.file_names)}: {nav.file_names}"
		assert 'subdir' in nav.sub_dirs, "subdir should be in sub_dirs"
		assert 'readme.txt' not in nav.file_names, "txt files should be excluded"
		print(f"  OK: filtered to {nav.file_names}")


def test_navigate_parent():
	"""Test navigating to parent directory."""
	nav = DirectoryNavigator('data/xyz')
	parent = nav.parent_dir()
	nav.navigate_to('..')
	assert nav.work_dir == os.path.abspath('data'), f"Expected data dir, got {nav.work_dir}"
	print(f"  OK: navigated to parent {nav.work_dir}")


def test_navigate_subdir():
	"""Test navigating into a subdirectory."""
	nav = DirectoryNavigator('data')
	nav.navigate_to('xyz')
	assert 'benzene.xyz' in nav.file_names, f"benzene.xyz not found in {nav.file_names}"
	print(f"  OK: navigated to {nav.work_dir}, found {len(nav.file_names)} files")


def test_navigate_back():
	"""Test navigating to parent and back."""
	nav = DirectoryNavigator('data/xyz')
	original = nav.work_dir
	nav.navigate_to('..')
	nav.navigate_to('xyz')
	assert nav.work_dir == original, f"Expected {original}, got {nav.work_dir}"
	print(f"  OK: round-trip navigation works")


def test_empty_dir():
	"""Test empty directory."""
	with tempfile.TemporaryDirectory() as tmpdir:
		nav = DirectoryNavigator(tmpdir)
		assert len(nav.file_names) == 0, "Empty dir should have 0 files"
		assert nav.sub_dirs == ['..'], f"Empty dir should only have '..', got {nav.sub_dirs}"
		print("  OK: empty directory handled")


def run_all():
	tests = [
		("read_dir", test_read_dir),
		("extensions", test_extensions),
		("navigate_parent", test_navigate_parent),
		("navigate_subdir", test_navigate_subdir),
		("navigate_back", test_navigate_back),
		("empty_dir", test_empty_dir),
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
