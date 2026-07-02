"""
MolecularBrowserVispy.py — ACDSee-style molecular file browser using VisPy.

Two modes:
  - BROWSE: thumbnail grid of molecules in current directory
  - VIEW:   interactive 3D view of selected molecule (via MoleculeViewer)

Navigation:
  - Arrow keys: move cursor in grid
  - Enter:      switch BROWSE <-> VIEW
  - Backspace:  go to parent directory
  - Mouse click: select thumbnail or navigate directory
  - Mouse wheel: scroll grid
  - Esc:        quit / back to browse

Phase 1 implementation: uses VisPy ImageVisual per thumbnail (simple, not yet optimized).
Phase 2 will replace with instanced quads + texture atlas.
"""

import sys, os, argparse
import numpy as np
from collections import deque

from PyQt5 import QtWidgets, QtCore, QtGui
import vispy
vispy.use('pyqt5')
from vispy import scene
from vispy.scene import visuals

from spammm.GUI.DirectoryNavigator import DirectoryNavigator
from spammm.GUI.ThumbnailCache import ThumbnailCache
from spammm.GUI.MoleculeViewer import MoleculeViewer


class MolecularBrowserVispy(QtWidgets.QMainWindow):
	"""ACDSee-style molecular file browser using VisPy + PyQt5."""

	# Layout constants
	THUMB_SIZE = 200       # display size of each thumbnail (pixels)
	THUMB_RENDER = 256     # offscreen render resolution
	THUMB_SPACING = 20     # spacing between thumbnails
	MARGIN = 20            # margin around grid
	DIR_BAR_HEIGHT = 40    # height of directory button bar
	LABEL_HEIGHT = 20      # height of filename label under thumbnail

	def __init__(self, start_dir='.', title='Molecular Browser'):
		super().__init__()
		self.setWindowTitle(title)
		self.resize(1200, 800)

		# State
		self.navigator = DirectoryNavigator(start_dir)
		self.thumb_cache = ThumbnailCache(thumb_size=self.THUMB_RENDER)
		self.thumb_cache.set_files(self.navigator.file_paths)

		self.mode = 'BROWSE'  # 'BROWSE' or 'VIEW'
		self.cursor_index = 0  # selected thumbnail index
		self.scroll_y = 0     # grid scroll offset
		self.grid_cols = 0    # computed on resize
		self.grid_rows = 0

		# Thumbnail ImageVisuals (one per molecule, Phase 1)
		self.thumb_visuals = []
		self.thumb_labels = None  # VisPy Text visual for all labels
		self.dir_buttons = []     # list of (name, rect) for subdirectory buttons
		self.cursor_rect = None   # Line visual for cursor outline

		# 3D viewer for VIEW mode
		self.viewer = None

		# Build UI
		self._init_ui()
		self._update_grid()
		self._start_thumbnail_timer()

	def _init_ui(self):
		"""Create the VisPy canvas and connect events."""
		self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', show=True, size=(1000, 700))
		self.grid_view = self.canvas.central_widget.add_view()
		self.grid_view.camera = scene.PanZoomCamera(aspect=1)
		self.grid_view.camera.interactive = True

		# Text for path bar and status
		self.path_text = visuals.Text(parent=self.grid_view.scene, color='black', font_size=12, anchor_x='left', anchor_y='top')
		self.status_text = visuals.Text(parent=self.grid_view.scene, color=(0.3, 0.3, 0.3), font_size=10, anchor_x='left', anchor_y='bottom')
		self.status_text.visible = False  # shown in status bar instead

		# Cursor outline (green rectangle)
		self.cursor_line = visuals.Line(parent=self.grid_view.scene, color='lime', width=2.0, antialias=True, method='gl')
		self.cursor_line.set_gl_state('translucent', depth_test=False)

		# Directory button text
		self.dir_text = visuals.Text(parent=self.grid_view.scene, color=(0.1, 0.4, 0.8), font_size=11, anchor_x='left', anchor_y='center')

		# All labels for thumbnails (single Text visual)
		self.thumb_labels = visuals.Text(parent=self.grid_view.scene, color='black', font_size=9, anchor_x='center', anchor_y='top')

		# Connect events
		self.canvas.events.key_press.connect(self._on_key_press)
		self.canvas.events.mouse_press.connect(self._on_mouse_press)
		self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
		self.canvas.events.resize.connect(self._on_resize)

		# Layout: canvas as central widget
		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(self.canvas.native)
		central = QtWidgets.QWidget()
		central.setLayout(layout)
		self.setCentralWidget(central)

		# Status bar
		self.statusBar().showMessage("Arrow keys: navigate | Enter: view | Backspace: parent dir | Esc: quit")

	def _start_thumbnail_timer(self):
		"""Start QTimer for lazy thumbnail rendering."""
		self.thumb_timer = QtCore.QTimer(self)
		self.thumb_timer.timeout.connect(self._on_thumb_timer)
		self.thumb_timer.start(16)  # ~60fps

	def _on_thumb_timer(self):
		"""Render one thumbnail per timer tick, update display."""
		if self.mode != 'BROWSE':
			return
		if not self.thumb_cache.has_pending():
			self.thumb_timer.stop()
			return
		rendered = self.thumb_cache.update(max_per_call=1)
		if rendered:
			self._update_thumbnail_visuals()

	def _update_grid(self):
		"""Recompute grid layout and recreate thumbnail visuals."""
		canvas_w, canvas_h = self.canvas.size
		usable_w = canvas_w - 2 * self.MARGIN
		usable_h = canvas_h - self.DIR_BAR_HEIGHT - 2 * self.MARGIN

		# Compute columns that fit
		col_w = self.THUMB_SIZE + self.THUMB_SPACING
		self.grid_cols = max(1, usable_w // col_w)

		n_files = len(self.navigator)
		self.grid_rows = (n_files + self.grid_cols - 1) // self.grid_cols

		# Compute grid positions (numpy)
		n = n_files
		if n == 0:
			self._grid_positions = np.zeros((0, 2), dtype=np.float32)
		else:
			cols = np.arange(n) % self.grid_cols
			rows = np.arange(n) // self.grid_cols
			x = cols * col_w + self.MARGIN + self.THUMB_SIZE / 2
			y = -rows * col_w - self.MARGIN - self.THUMB_SIZE / 2 - self.DIR_BAR_HEIGHT
			self._grid_positions = np.column_stack([x, y])

		# Update camera to show full grid
		if n > 0:
			total_w = self.grid_cols * col_w
			total_h = self.grid_rows * col_w
			self.grid_view.camera.rect = (0, -total_h - self.MARGIN, total_w + self.MARGIN, total_h + self.DIR_BAR_HEIGHT + self.MARGIN)

		# Update directory buttons
		self._update_dir_buttons()

		# Update thumbnail visuals
		self._update_thumbnail_visuals()

		# Update path text
		self.path_text.pos = (self.MARGIN, -10)
		self.path_text.text = f"📁 {self.navigator.work_dir}  ({n_files} molecules)"

		# Update cursor
		self._update_cursor()

	def _update_dir_buttons(self):
		"""Update directory button positions and text."""
		dirs = self.navigator.sub_dirs
		if not dirs:
			self.dir_text.text = ''
			self.dir_buttons = []
			return

		labels = []
		positions = []
		self.dir_buttons = []
		x = self.MARGIN
		y = -self.DIR_BAR_HEIGHT / 2
		for i, name in enumerate(dirs):
			w = len(name) * 8 + 20  # rough text width
			self.dir_buttons.append((name, (x, y - 10, x + w, y + 10)))
			labels.append(f"[{name}]" if name == '..' else name)
			positions.append((x + w / 2, y))
			x += w + self.THUMB_SPACING

		self.dir_text.pos = np.array(positions, dtype=np.float32)
		self.dir_text.text = labels

	def _update_thumbnail_visuals(self):
		"""Create/update ImageVisuals for all thumbnails."""
		# Remove old visuals
		for v in self.thumb_visuals:
			v.parent = None
		self.thumb_visuals = []

		n = len(self.navigator)
		if n == 0:
			self.thumb_labels.text = ''
			return

		# Create ImageVisual for each rendered thumbnail
		ready_indices = []
		lbl_pos = []
		lbl_txt = []

		for i in range(n):
			img = self.thumb_cache.get_image(i)
			if img is not None:
				# Create ImageVisual at grid position
				pos = self._grid_positions[i]
				# ImageVisual centered at pos: need to offset by half thumb size
				# VisPy Image uses pos as bottom-left corner
				x = pos[0] - self.THUMB_SIZE / 2
				y = pos[1] - self.THUMB_SIZE / 2
				iv = visuals.Image(img, parent=self.grid_view.scene)
				iv.transform = scene.STTransform(translate=(x, y), scale=(self.THUMB_SIZE / img.shape[1], self.THUMB_SIZE / img.shape[0]))
				iv.set_gl_state('translucent', depth_test=False)
				self.thumb_visuals.append(iv)
				ready_indices.append(i)

				# Label position (below thumbnail)
				lbl_pos.append((pos[0], pos[1] - self.THUMB_SIZE / 2 - 5))
				lbl_txt.append(self.navigator.file_names[i])

		# Update labels
		if lbl_pos:
			self.thumb_labels.pos = np.array(lbl_pos, dtype=np.float32)
			self.thumb_labels.text = lbl_txt
		else:
			self.thumb_labels.text = ''

		# Update status bar
		self.statusBar().showMessage(
			f"📁 {self.navigator.work_dir} | {n} molecules | "
			f"{self.thumb_cache.ready_count()} thumbnails rendered | "
			f"{self.thumb_cache.pending_count()} pending | "
			f"Arrow keys: navigate | Enter: view | Backspace: parent dir"
		)

	def _update_cursor(self):
		"""Update cursor rectangle outline at selected thumbnail."""
		n = len(self.navigator)
		if n == 0 or self.cursor_index >= n:
			self.cursor_line.set_data(np.zeros((0, 2), dtype=np.float32))
			return
		pos = self._grid_positions[self.cursor_index]
		s = self.THUMB_SIZE / 2 + 4
		rect = np.array([
			[pos[0] - s, pos[1] - s],
			[pos[0] + s, pos[1] - s],
			[pos[0] + s, pos[1] + s],
			[pos[0] - s, pos[1] + s],
			[pos[0] - s, pos[1] - s],
		], dtype=np.float32)
		self.cursor_line.set_data(rect)

	# --- Mode switching ---

	def enter_view_mode(self):
		"""Switch to 3D view of selected molecule (opens in separate window)."""
		n = len(self.navigator)
		if n == 0 or self.cursor_index >= n:
			return
		self.mode = 'VIEW'

		# Viewer opens in its own window — browser keeps showing thumbnails
		filepath = self.navigator.file_paths[self.cursor_index]
		fname = self.navigator.file_names[self.cursor_index]
		self.setWindowTitle(f"Viewing: {fname}")

		# Create viewer with own canvas (separate window)
		self.viewer = MoleculeViewer(bgcolor='white', show=True)
		self.viewer.load_file(filepath)
		self.viewer.set_label_mode('Element+Index')

		# Forward viewer key presses to browser so Enter/Esc works from viewer window
		self.viewer.canvas.events.key_press.connect(self._on_viewer_key_press)
		# Hook Qt native closeEvent to detect X button / Alt+F4
		native = self.viewer.canvas.native
		self._orig_viewer_close = native.closeEvent
		native.closeEvent = self._on_viewer_native_close

		self.statusBar().showMessage(f"Viewing: {fname} | Esc/Enter: back to browse")

	def _on_viewer_key_press(self, event):
		"""Handle Enter/Esc from viewer window to return to browse."""
		if event.key in ('Enter', 'Escape'):
			self.exit_view_mode()

	def _on_viewer_native_close(self, event):
		"""Qt native closeEvent hook — fires on X button, Alt+F4, etc."""
		# Restore original closeEvent first to avoid re-entry
		orig = getattr(self, '_orig_viewer_close', None)
		if orig is not None:
			self.viewer.canvas.native.closeEvent = orig
			self._orig_viewer_close = None
			orig(event)
		else:
			event.accept()
		# Return to browse mode if still in VIEW
		if self.mode == 'VIEW':
			self.mode = 'BROWSE'
			self.viewer = None
			self.setWindowTitle('Molecular Browser')
			self._update_thumbnail_visuals()
			self._start_thumbnail_timer()
			self.raise_()
			self.activateWindow()
			self.canvas.native.setFocus()

	def exit_view_mode(self):
		"""Return to browse mode (close viewer window)."""
		if self.viewer is not None:
			# Restore original closeEvent before closing to avoid re-entry
			native = self.viewer.canvas.native
			if hasattr(self, '_orig_viewer_close') and self._orig_viewer_close is not None:
				native.closeEvent = self._orig_viewer_close
				self._orig_viewer_close = None
			try: self.viewer.canvas.events.key_press.disconnect(self._on_viewer_key_press)
			except Exception: pass
			self.viewer.canvas.close()
			self.viewer = None

		self.mode = 'BROWSE'
		self.setWindowTitle('Molecular Browser')
		self._update_thumbnail_visuals()
		self._start_thumbnail_timer()

		# Restore focus to browser window
		self.raise_()
		self.activateWindow()
		self.canvas.native.setFocus()

	# --- Input handlers ---

	def _on_key_press(self, event):
		if event.key == 'Escape':
			if self.mode == 'VIEW':
				self.exit_view_mode()
			else:
				self.close()
			return

		if self.mode == 'VIEW':
			# In view mode, only Enter/Esc return to browse
			if event.key == 'Enter':
				self.exit_view_mode()
			return

		# BROWSE mode
		n = len(self.navigator)
		if n == 0 and event.key != 'Backspace':
			return

		if event.key == 'Left':
			self.cursor_index = max(0, self.cursor_index - 1)
			self._update_cursor()
		elif event.key == 'Right':
			self.cursor_index = min(n - 1, self.cursor_index + 1)
			self._update_cursor()
		elif event.key == 'Up':
			self.cursor_index = max(0, self.cursor_index - self.grid_cols)
			self._update_cursor()
		elif event.key == 'Down':
			self.cursor_index = min(n - 1, self.cursor_index + self.grid_cols)
			self._update_cursor()
		elif event.key == 'Enter':
			self.enter_view_mode()
		elif event.key == 'Backspace':
			self._navigate_parent()

	def _on_mouse_press(self, event):
		if self.mode != 'BROWSE':
			return
		# Convert screen pos to grid coordinates
		if event.button != 1:  # left click only
			return
		pos = self.grid_view.camera.transform.imap(event.pos[:2])

		# Check directory buttons first
		for name, rect in self.dir_buttons:
			if rect[0] <= pos[0] <= rect[2] and rect[3] <= pos[1] <= rect[1]:
				self._navigate_to(name)
				return

		# Check thumbnails
		n = len(self.navigator)
		for i in range(n):
			p = self._grid_positions[i]
			s = self.THUMB_SIZE / 2
			if abs(pos[0] - p[0]) < s and abs(pos[1] - p[1]) < s:
				self.cursor_index = i
				self._update_cursor()
				# Double-click would enter view; single click just selects
				return

	def _on_mouse_wheel(self, event):
		if self.mode != 'BROWSE':
			return
		# PanZoomCamera handles wheel for zoom; we want scroll
		# For Phase 1: let camera handle it
		pass

	def _on_resize(self, event):
		if self.mode == 'BROWSE':
			self._update_grid()

	# --- Navigation ---

	def _navigate_to(self, dir_name):
		"""Navigate to a subdirectory."""
		self.navigator.navigate_to(dir_name)
		self.thumb_cache.set_files(self.navigator.file_paths)
		self.cursor_index = 0
		self._update_grid()
		self._start_thumbnail_timer()

	def _navigate_parent(self):
		"""Navigate to parent directory."""
		self._navigate_to('..')


# --- CLI entry point ---

def main():
	parser = argparse.ArgumentParser(description='ACDSee-style molecular file browser')
	parser.add_argument('--dir', default='.', help='Starting directory')
	parser.add_argument('--title', default='Molecular Browser', help='Window title')
	args = parser.parse_args()

	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
	browser = MolecularBrowserVispy(start_dir=args.dir, title=args.title)
	browser.show()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()
