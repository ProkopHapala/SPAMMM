"""
ThumbnailCache.py — Offscreen rendering of molecule thumbnails + lazy job queue.

Renders each molecule to a small RGBA image using MoleculeViewer.render_offscreen().
Stores results in memory (no disk cache). Uses a job queue for lazy rendering:
one thumbnail per update() call, called from QTimer in the browser.

For Phase 1 (prototype): stores individual numpy arrays, no texture atlas yet.
Phase 2 will add TextureAtlas for batch rendering.
"""

import numpy as np
from collections import deque

from spammm.GUI.MoleculeViewer import MoleculeViewer, default_atom_colors, default_atom_sizes
from spammm.AtomicSystem import AtomicSystem


class ThumbnailCache:
	"""Lazy-rendered in-memory thumbnail cache for molecular files.

	Args:
		thumb_size:  rendered thumbnail size in pixels (square)
		bgcolor:     background color for rendering
	"""

	def __init__(self, thumb_size=256, bgcolor='white'):
		self.thumb_size = thumb_size
		self.bgcolor = bgcolor
		self._file_paths = []
		self._images = {}       # index -> (H,W,4) uint8 RGBA
		self._render_queue = deque()
		self._viewer = None     # lazily created MoleculeViewer for offscreen rendering

	def set_files(self, file_paths):
		"""Set the list of molecule files to render. Resets cache and queue."""
		self._file_paths = list(file_paths)
		self._images = {}
		self._render_queue = deque(range(len(self._file_paths)))

	def get_image(self, index):
		"""Return RGBA numpy array for thumbnail, or None if not yet rendered."""
		return self._images.get(index, None)

	def is_ready(self, index):
		"""True if thumbnail has been rendered."""
		return index in self._images

	def update(self, max_per_call=1):
		"""Render up to max_per_call thumbnails from the queue.

		Returns:
			list of indices that were rendered this call (empty if queue was empty)
		"""
		rendered = []
		for _ in range(max_per_call):
			if not self._render_queue:
				break
			idx = self._render_queue.popleft()
			try:
				img = self._render_molecule(self._file_paths[idx])
				self._images[idx] = img
				rendered.append(idx)
			except Exception as e:
				print(f"ThumbnailCache: failed to render {self._file_paths[idx]}: {e}")
				# Store a placeholder (gray) so we don't retry
				self._images[idx] = np.full((self.thumb_size, self.thumb_size, 4), 128, dtype=np.uint8)
				rendered.append(idx)
		return rendered

	def has_pending(self):
		"""True if there are thumbnails still waiting to be rendered."""
		return len(self._render_queue) > 0

	def pending_count(self):
		"""Number of thumbnails waiting in queue."""
		return len(self._render_queue)

	def ready_count(self):
		"""Number of thumbnails already rendered."""
		return len(self._images)

	def invalidate(self):
		"""Clear all cached thumbnails."""
		self._images = {}
		self._render_queue.clear()

	def _get_viewer(self):
		"""Lazily create an offscreen MoleculeViewer for rendering."""
		if self._viewer is None:
			self._viewer = MoleculeViewer(bgcolor=self.bgcolor, show=True)
			# Immediately hide the window — we only need the GL context
			self._viewer.canvas.native.hide()
		return self._viewer

	def _render_molecule(self, file_path):
		"""Load molecule and render to RGBA numpy array.

		Returns:
			(thumb_size, thumb_size, 4) uint8 RGBA
		"""
		viewer = self._get_viewer()
		viewer.load_file(file_path)
		img = viewer.render_offscreen(self.thumb_size)
		return img
