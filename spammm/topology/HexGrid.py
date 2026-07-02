"""
HexGrid.py — Hexagonal grid with transform support (offset, rotation, transpose).

Auxiliary ruler for hex drawing and atom snapping. Decoupled from molecular topology.
Transform: transpose → rotate → offset (applied to base honeycomb positions).
"""

import numpy as np

s3 = np.sqrt(3.0)

def snap_to_grid(pos_xy):
	"""Round a 2D position to 4 decimal places → tuple key for dict lookup."""
	return (round(float(pos_xy[0]), 4), round(float(pos_xy[1]), 4))

class HexGrid:
	"""Honeycomb grid with optional offset, rotation, and transpose transforms.

	Parameters
	----------
	a_CC : float
		C-C bond length (circumradius of hexagon), default 1.42 Å.
	offset : (float, float)
		(x, y) offset of grid origin in world coords.
	rotation : float
		Rotation angle in radians (counter-clockwise).
	transpose : bool
		Swap x↔y axes (reflect along y=x line). NOT a symmetry of the hex grid,
		so this produces a visible change (unlike mirroring along x or y).
	"""

	def __init__(self, a_CC=1.42, offset=(0.0, 0.0), rotation=0.0, transpose=False):
		self.a_CC = a_CC
		self.offset = np.array(offset, dtype=np.float64)
		self.rotation = rotation
		self.transpose = transpose

	# ── Transform helpers ──────────────────────────────────────────────

	def _base_ring_nodes(self, q, r):
		"""Base honeycomb node positions (no transform) for axial (q, r)."""
		a = self.a_CC
		cx = a * s3 * (q + r * 0.5)
		cy = a * 1.5 * r
		angles = np.arange(6) * (np.pi / 3.0) + np.pi / 6.0
		return np.column_stack([cx + a * np.cos(angles), cy + a * np.sin(angles)])

	def _to_world(self, local_xy):
		"""Transform grid-local coords → world coords (transpose → rotate → offset)."""
		x, y = local_xy[0], local_xy[1]
		if self.transpose: x, y = y, x
		if self.rotation != 0.0:
			c, s = np.cos(self.rotation), np.sin(self.rotation)
			x, y = c * x - s * y, s * x + c * y
		return np.array([x + self.offset[0], y + self.offset[1]])

	def _to_local(self, world_xy):
		"""Inverse transform: world coords → grid-local coords."""
		x = world_xy[0] - self.offset[0]
		y = world_xy[1] - self.offset[1]
		if self.rotation != 0.0:
			c, s = np.cos(-self.rotation), np.sin(-self.rotation)
			x, y = c * x - s * y, s * x + c * y
		if self.transpose: x, y = y, x
		return np.array([x, y])

	# ── Public API ─────────────────────────────────────────────────────

	def ring_nodes(self, q, r):
		"""Return (6, 2) array of world-space node positions for hexagon at axial (q, r)."""
		base = self._base_ring_nodes(q, r)
		return np.array([self._to_world(node) for node in base])

	def snap_to_ring(self, x, y):
		"""Find axial (q, r) of hexagon whose center is closest to world (x, y)."""
		lx, ly = self._to_local(np.array([x, y]))
		a = self.a_CC
		r_exact = ly / (1.5 * a)
		q_exact = lx / (s3 * a) - r_exact * 0.5
		return int(round(q_exact)), int(round(r_exact))

	def snap_to_node(self, x, y, tol=0.2):
		"""Find nearest grid node key to world (x, y), or None if farther than tol."""
		q, r = self.snap_to_ring(x, y)
		best_node = None
		min_dist = float('inf')
		for dq, dr in [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
			nodes = self.ring_nodes(q + dq, r + dr)
			for node in nodes:
				d = np.sqrt((node[0] - x)**2 + (node[1] - y)**2)
				if d < min_dist:
					min_dist = d
					best_node = node
		if min_dist < tol:
			return snap_to_grid(best_node)
		return None

	def get_guide_points(self, qrange=(-10, 10), rrange=(-10, 10)):
		"""Return (N, 2) array of unique node positions in world coords."""
		nodes = set()
		for q in range(qrange[0], qrange[1] + 1):
			for r in range(rrange[0], rrange[1] + 1):
				for node in self.ring_nodes(q, r):
					nodes.add(snap_to_grid(node))
		return np.array(list(nodes))

	# ── Transform mutators (invalidate pins) ───────────────────────────

	def toggle_transpose(self):
		"""Swap x↔y axes (reflect along y=x). Returns self for chaining."""
		self.transpose = not self.transpose
		return self

	def rotate(self, angle_rad):
		"""Add rotation angle (radians). Returns self for chaining."""
		self.rotation += angle_rad
		return self

	def set_rotation(self, angle_rad):
		"""Set absolute rotation angle (radians). Returns self for chaining."""
		self.rotation = angle_rad
		return self

	def shift(self, dx, dy):
		"""Shift grid offset by (dx, dy). Returns self for chaining."""
		self.offset = self.offset + np.array([dx, dy])
		return self

	def set_offset(self, ox, oy):
		"""Set absolute offset. Returns self for chaining."""
		self.offset = np.array([ox, oy], dtype=np.float64)
		return self

	def reset_transform(self):
		"""Reset transform to identity (no offset, no rotation, no flip)."""
		self.offset = np.array([0.0, 0.0], dtype=np.float64)
		self.rotation = 0.0
		self.transpose = False
