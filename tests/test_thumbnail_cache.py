"""
test_thumbnail_cache.py — Test ThumbnailCache lazy rendering.

Tests:
  1. Set files and verify queue populated
  2. Update renders thumbnails one at a time
  3. is_ready / get_image work correctly
  4. Invalidate clears cache
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spammm.GUI.ThumbnailCache import ThumbnailCache


def test_set_files():
	"""Test set_files populates queue."""
	cache = ThumbnailCache(thumb_size=128)
	cache.set_files(['data/xyz/H2O.xyz', 'data/xyz/benzene.xyz', 'data/xyz/CH4.xyz'])
	assert cache.pending_count() == 3, f"Expected 3 pending, got {cache.pending_count()}"
	assert cache.ready_count() == 0, f"Expected 0 ready, got {cache.ready_count()}"
	print(f"  OK: 3 files queued, 0 ready")


def test_update_one():
	"""Test update renders one thumbnail."""
	cache = ThumbnailCache(thumb_size=128)
	cache.set_files(['data/xyz/H2O.xyz', 'data/xyz/benzene.xyz'])
	rendered = cache.update(max_per_call=1)
	assert len(rendered) == 1, f"Expected 1 rendered, got {len(rendered)}"
	assert cache.is_ready(rendered[0]), "Rendered index should be ready"
	assert cache.pending_count() == 1, f"Expected 1 pending, got {cache.pending_count()}"
	img = cache.get_image(rendered[0])
	assert img is not None, "Image should not be None"
	assert img.shape == (128, 128, 4), f"Expected (128,128,4), got {img.shape}"
	print(f"  OK: rendered 1 thumbnail, shape={img.shape}")


def test_update_all():
	"""Test rendering all thumbnails."""
	cache = ThumbnailCache(thumb_size=64)
	cache.set_files(['data/xyz/H2O.xyz', 'data/xyz/CO.xyz', 'data/xyz/HF.xyz'])
	while cache.has_pending():
		cache.update(max_per_call=1)
	assert cache.ready_count() == 3, f"Expected 3 ready, got {cache.ready_count()}"
	assert not cache.has_pending(), "Queue should be empty"
	print(f"  OK: all 3 thumbnails rendered")


def test_invalidate():
	"""Test invalidate clears cache."""
	cache = ThumbnailCache(thumb_size=64)
	cache.set_files(['data/xyz/H2O.xyz'])
	cache.update(max_per_call=1)
	assert cache.ready_count() == 1
	cache.invalidate()
	assert cache.ready_count() == 0, "Cache should be empty after invalidate"
	assert not cache.has_pending(), "Queue should be empty after invalidate"
	print("  OK: invalidate works")


def test_failed_render():
	"""Test that a failed render produces a placeholder, not a crash."""
	cache = ThumbnailCache(thumb_size=64)
	cache.set_files(['nonexistent/file.xyz'])
	rendered = cache.update(max_per_call=1)
	assert len(rendered) == 1, "Should have rendered (placeholder) even on failure"
	img = cache.get_image(rendered[0])
	assert img is not None, "Placeholder should exist"
	assert img.shape == (64, 64, 4), f"Expected (64,64,4), got {img.shape}"
	print("  OK: failed render produces placeholder")


def run_all():
	tests = [
		("set_files", test_set_files),
		("update_one", test_update_one),
		("update_all", test_update_all),
		("invalidate", test_invalidate),
		("failed_render", test_failed_render),
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
