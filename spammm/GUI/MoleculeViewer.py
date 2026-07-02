"""
MoleculeViewer.py — Standalone modular 3D molecular viewer using VisPy.

Three composable layers (enabled via constructor flags):
  - rendering:  always on (atoms, bonds, labels, camera, GL state)
  - interaction: optional (picking, dragging, selection — Qt signals)
  - editing:    optional (ring preview, bond creation, hex grid — for SPAMMM_GUI)

Usage contexts:
  1. Standalone:   MoleculeViewer() creates own SceneCanvas
  2. Embedded:     MoleculeViewer(canvas=existing_canvas) attaches to existing canvas
  3. Offscreen:    MoleculeViewer(show=False) + render_offscreen() for thumbnails

Does NOT depend on AtomScene/VispyUtils — fully independent module.
"""

import numpy as np
from PyQt5 import QtCore
import vispy
vispy.use('pyqt5')
from vispy import scene
from vispy.scene import visuals

from spammm import elements
from spammm import atomicUtils as au
from spammm.AtomicSystem import AtomicSystem


def _as_f32(x):
	return np.asarray(x, dtype=np.float32)


def default_atom_colors(enames):
	"""Get RGBA colors for element names from elements.ELEMENT_DICT."""
	cols = np.zeros((len(enames), 4), dtype=np.float32)
	for i, e in enumerate(enames):
		r, g, b = elements.hex_to_float_rgb(elements.ELEMENT_DICT[e][8])
		cols[i] = (r, g, b, 1.0)
	return cols


def default_atom_sizes(enames, scale=40.0):
	"""Get marker sizes from covalent radii * scale.

	VisPy markers are in pixels. scale=40 gives reasonable sphere sizes
	(C~29px, H~15px) at typical camera distances.
	"""
	sizes = np.zeros(len(enames), dtype=np.float32)
	for i, e in enumerate(enames):
		sizes[i] = elements.ELEMENT_DICT[e][6] * scale
	return sizes


def compute_bond_segments(pos, bonds):
	"""Convert (n,2) bond index array to (2n,3) line segment endpoints."""
	if bonds is None or len(bonds) == 0:
		return np.zeros((0, 3), dtype=np.float32)
	bonds = np.asarray(bonds)
	segs = np.empty((bonds.shape[0] * 2, 3), dtype=np.float32)
	segs[0::2] = pos[bonds[:, 0]]
	segs[1::2] = pos[bonds[:, 1]]
	return segs


class MoleculeViewer(QtCore.QObject):
	"""Standalone modular 3D molecular viewer.

	Args:
		bgcolor:     canvas background color
		canvas:      existing SceneCanvas to attach to (None = create own)
		interaction: enable picking/dragging/selection signals
		editing:     enable ring preview, bond creation, hex grid (for editor)
		backend:     optional backend object for authoritative geometry
		show:        show canvas window (False for offscreen/headless)
	"""

	# Signals (only meaningful if interaction=True)
	sig_atom_picked = QtCore.pyqtSignal(int)
	sig_drag_state = QtCore.pyqtSignal(int, int, object)  # active(0/1), atom_id, pos3
	sig_atom_moved = QtCore.pyqtSignal(int, object)       # atom_id, pos3
	sig_rmb_remove = QtCore.pyqtSignal(int)
	sig_selection_changed = QtCore.pyqtSignal(object)     # set of atom ids
	sig_camera_changed = QtCore.pyqtSignal()
	sig_link_bond = QtCore.pyqtSignal(int, int)
	sig_atom_clicked = QtCore.pyqtSignal(int)

	def __init__(self, *, bgcolor='white', canvas=None, interaction=False, editing=False, backend=None, show=False):
		super().__init__()
		self.backend = backend
		self._interaction = interaction
		self._editing = editing

		# Canvas: create own or attach to existing
		if canvas is not None:
			self.canvas = canvas
			self._owns_canvas = False
		else:
			self.canvas = scene.SceneCanvas(keys='interactive', bgcolor=bgcolor, show=show, size=(512, 512))
			self._owns_canvas = True

		self.view = self.canvas.central_widget.add_view()
		# Default: 3D turntable camera
		self.view.camera = scene.TurntableCamera(fov=30, distance=20, elevation=30, azimuth=30)
		self.view.camera.interactive = True

		# --- Rendering visuals (always on) ---
		self.bond_lines = visuals.Line(parent=self.view.scene, color=(0.3, 0.3, 0.3, 0.8), width=2.0, antialias=True)
		self.atom_markers = visuals.Markers(parent=self.view.scene)
		self.text_labels = visuals.Text(parent=self.view.scene, color='black', font_size=10, anchor_x='left', anchor_y='bottom')
		self.axes = visuals.XYZAxis(parent=self.view.scene)

		# GL state
		self.atom_markers.set_gl_state('translucent', depth_test=True)
		self.bond_lines.set_gl_state('translucent', depth_test=True)
		self.text_labels.set_gl_state('translucent', depth_test=False)

		# --- Interaction visuals (only if interaction=True) ---
		if interaction:
			self.hover_marker = visuals.Markers(parent=self.view.scene)
			self.hover_marker.set_gl_state('translucent', depth_test=False)
			self.hover_marker.visible = False
			self.selection_rect = None  # lazily created
			self._selected_ids = set()
			self._pick_radius = 10.0
			self._drag_id = None
			self._drag_offset = None
			self._lock_drag = False
			self._link_mode = False
			self.link_line = visuals.Line(parent=self.view.scene, color=(0.2, 0.8, 0.2, 0.8), width=3.0, antialias=True, method='gl')
			self.link_line.visible = False

		# --- Editing visuals (only if editing=True) ---
		if editing:
			self.ring_preview_line = visuals.Line(parent=self.view.scene, color=(0.2, 0.8, 0.8, 0.6), width=2.0, antialias=True, method='gl')
			self.ring_preview_line.visible = False

		# Data state
		self._pos = np.zeros((0, 3), dtype=np.float32)
		self._colors = None
		self._sizes = None
		self._bonds = None
		self._enames = None
		self._atom_ids = np.zeros((0,), dtype=np.int64)
		self._label_mode = 'none'
		self._bgcolor = bgcolor

		# Connect camera change signal (VisPy camera emits 'changed' event)
		try:
			self.view.camera.events.changed.connect(self._on_camera_changed)
		except Exception:
			pass  # Some camera types may not have 'changed' event

	def _on_camera_changed(self, event=None):
		self.sig_camera_changed.emit()

	# --- Data ---

	def set_data(self, pos, colors=None, sizes=None, bonds=None, forces=None, enames=None, atom_ids=None):
		"""Set molecule data and redraw.

		Args:
			pos:      (n,3) float32 positions
			colors:   (n,4) float32 RGBA, or None for default element colors
			sizes:    (n,) float32 marker sizes, or None for default
			bonds:    (m,2) int bond indices, or None
			forces:   (n,3) float32 forces (optional, not yet rendered)
			enames:   list of element name strings (for default colors/sizes/labels)
			atom_ids: (n,) int64 stable atom IDs (for interaction)
		"""
		self._pos = _as_f32(pos)
		self._enames = enames
		self._bonds = bonds
		self._forces = forces

		if atom_ids is not None:
			self._atom_ids = np.asarray(atom_ids, dtype=np.int64)
		else:
			self._atom_ids = np.arange(len(self._pos), dtype=np.int64)

		# Colors
		if colors is not None:
			self._colors = _as_f32(colors)
		elif enames is not None:
			self._colors = default_atom_colors(enames)
		else:
			self._colors = np.tile((0.5, 0.5, 0.5, 1.0), (len(self._pos), 1)).astype(np.float32)

		# Sizes
		if sizes is not None:
			self._sizes = _as_f32(sizes)
		elif enames is not None:
			self._sizes = default_atom_sizes(enames)
		else:
			self._sizes = np.full(len(self._pos), 10.0, dtype=np.float32)

		self._redraw()

	def load_file(self, filepath, orient_pca=True):
		"""Load molecule from .xyz/.mol/.mol2 file and set_data.

		Args:
			filepath:    path to molecule file
			orient_pca:  if True, center at COG and rotate so principal axes align with screen
		"""
		system = AtomicSystem(fname=filepath)
		if system.bonds is None:
			system.findBonds()
		pos = system.apos.astype(np.float32).copy()
		bonds = system.bonds
		enames = system.enames
		# PCA orientation: center at COG first, then rotate to principal axes
		if orient_pca and len(pos) > 2:
			cog = pos.mean(axis=0)
			pos -= cog  # center at origin (matching C++ addToPos(COG*-1))
			au.orientPCA(pos)  # rotate in-place around origin
		self.set_data(pos, enames=enames, bonds=bonds)
		self.auto_fit()
		return system

	def _redraw(self):
		"""Update all visuals from current data."""
		n = len(self._pos)
		if n == 0:
			self.atom_markers.set_data(np.zeros((0, 3), dtype=np.float32))
			self.bond_lines.set_data(np.zeros((0, 3), dtype=np.float32))
			self.text_labels.text = ['']
			self.text_labels.pos = np.zeros((1, 3), dtype=np.float32)
			self.text_labels.visible = False
			return

		# Atoms
		self.atom_markers.set_data(self._pos, face_color=self._colors, size=self._sizes, edge_width=0.5, edge_color='black', symbol='disc')

		# Bonds
		segs = compute_bond_segments(self._pos, self._bonds)
		self.bond_lines.set_data(segs)

		# Labels
		self._update_labels()

	def _update_labels(self):
		"""Update text labels based on _label_mode."""
		if self._label_mode == 'none' or self._enames is None:
			self.text_labels.text = ['']
			self.text_labels.pos = np.zeros((1, 3), dtype=np.float32)
			self.text_labels.visible = False
			return

		idx = np.arange(len(self._pos))
		lbl_pos = []
		lbl_txt = []

		if self._label_mode == 'Element+Index':
			for i, e in enumerate(self._enames):
				lbl_pos.append(self._pos[i])
				lbl_txt.append(f"{e}{i}")
		elif self._label_mode == 'Element':
			for i, e in enumerate(self._enames):
				lbl_pos.append(self._pos[i])
				lbl_txt.append(e)

		if lbl_pos:
			self.text_labels.pos = np.array(lbl_pos, dtype=np.float32)
			self.text_labels.text = lbl_txt
			self.text_labels.visible = True
		else:
			self.text_labels.visible = False

	# --- Camera ---

	def auto_fit(self):
		"""Fit camera to molecule bounding box.

		After PCA orientation, molecule is centered at origin with the two
		largest principal axes aligned to screen X and Y. Camera looks
		down Z (elevation=90, azimuth=0) so screen XY = world XY.
		Sets camera distance so the molecule's XY bounding box fills the viewport.
		"""
		if len(self._pos) == 0:
			return
		# After PCA, molecule is centered at origin. Compute XY bounding box.
		span = np.ptp(self._pos, axis=0)
		# Screen-plane spans (X, Y) after PCA + camera looking down Z
		max_span = max(span[0], span[1])
		if max_span < 1e-6:
			max_span = 5.0
		# Look straight down Z so PCA X,Y = screen X,Y
		self.view.camera.azimuth = 0
		self.view.camera.elevation = 90
		self.view.camera.center = (0.0, 0.0, 0.0)
		# C++ BrowserView: zoom_fit = maxspan * 0.7 (half-height of view)
		# VisPy TurntableCamera: visible half-height = distance * tan(fov/2)
		fov = self.view.camera.fov or 30
		if fov < 1:
			# Orthographic: distance IS the half-height
			distance = max_span * 0.7
		else:
			fov_rad = np.radians(fov)
			distance = (max_span * 0.7) / np.tan(fov_rad / 2)
		self.view.camera.distance = distance

	def set_camera_2d(self):
		"""Switch to orthographic top-down view (elevation=90, fov=0)."""
		self.view.camera = scene.TurntableCamera(fov=0, distance=self.view.camera.distance, elevation=90, azimuth=0)
		self.view.camera.interactive = True
		try: self.view.camera.events.changed.connect(self._on_camera_changed)
		except Exception: pass

	def set_camera_3d(self):
		"""Switch to perspective 3D view (fov=30, free rotation)."""
		self.view.camera = scene.TurntableCamera(fov=30, distance=self.view.camera.distance, elevation=30, azimuth=30)
		self.view.camera.interactive = True
		try: self.view.camera.events.changed.connect(self._on_camera_changed)
		except Exception: pass

	def set_zoom(self, zoom):
		"""Set camera distance (inverse zoom)."""
		self.view.camera.distance = max(1e-4, float(zoom))

	# --- Labels ---

	def set_label_mode(self, mode):
		"""Set label mode: 'none', 'Element', 'Element+Index'."""
		self._label_mode = mode
		self._update_labels()

	# --- Offscreen rendering ---

	def render_offscreen(self, size=256, supersample=4):
		"""Render current scene to RGBA numpy array with supersampling.

		Renders at native canvas size (which centers correctly), then crops to
		square and downscales to size² for antialiasing.
		Note: canvas.render(size=) doesn't adjust camera projection, causing
		off-center output. Rendering at native size avoids this bug.

		Args:
			size:         final output size in pixels (square)
			supersample:  unused (native canvas is already larger than size)

		Returns:
			(size, size, 4) uint8 RGBA numpy array
		"""
		img = self.canvas.render(alpha=True)
		if img is None:
			return np.zeros((size, size, 4), dtype=np.uint8)
		img = np.asarray(img, dtype=np.uint8)
		h, w = img.shape[:2]
		# Crop to square (centered)
		s = min(h, w)
		y0 = (h - s) // 2
		x0 = (w - s) // 2
		img = img[y0:y0+s, x0:x0+s]
		# Downscale to target size using block average if integer multiple
		if s > size and s % size == 0:
			ss = s // size
			img = img.reshape(size, ss, size, ss, 4).mean(axis=(1, 3)).astype(np.uint8)
		elif s != size:
			# Simple stride-based downscale
			ys = np.linspace(0, s-1, size).astype(int)
			xs = np.linspace(0, s-1, size).astype(int)
			img = img[ys][:, xs]
		return img

	# --- Interaction (only if interaction=True) ---

	def set_selection_mode(self, enabled):
		if not self._interaction:
			return
		self._selection_mode = enabled

	def lock_drag(self, locked):
		if not self._interaction:
			return
		self._lock_drag = locked

	# --- Utility ---

	def show(self):
		"""Show the canvas window (standalone mode)."""
		if self._owns_canvas:
			self.canvas.show()

	@property
	def native(self):
		"""Return native Qt widget for embedding in layouts."""
		return self.canvas.native

	def run(self):
		"""Run the VisPy app (standalone mode)."""
		if self._owns_canvas:
			self.canvas.show()
			from vispy.app import use
			app = self.canvas.app
			app.run()


# --- CLI entry point ---

def main():
	import sys, argparse
	parser = argparse.ArgumentParser(description='Standalone 3D molecule viewer')
	parser.add_argument('file', help='Molecule file (.xyz, .mol, .mol2)')
	parser.add_argument('--labels', default='none', choices=['none', 'Element', 'Element+Index'], help='Label mode')
	parser.add_argument('--bg', default='white', help='Background color')
	args = parser.parse_args()

	viewer = MoleculeViewer(bgcolor=args.bg, show=True)
	viewer.load_file(args.file)
	viewer.set_label_mode(args.labels)
	viewer.auto_fit()
	viewer.run()

if __name__ == '__main__':
	main()
