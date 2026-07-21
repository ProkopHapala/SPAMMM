"""
VispyUtils.py — Reusable VisPy 3D molecular visualization widgets.

Purpose: Provide AtomScene, a self-contained VisPy widget for rendering atoms
as spheres, bonds as cylinders, forces as arrows, and labels as text. Handles
picking, dragging, camera controls, and selection management.

Key functionality:
  - AtomScene: 3D scene with atoms, bonds, forces, labels, cell box
  - Atom picking (click to select), multi-select, drag-to-move
  - Camera: turntable arcball, perspective/orthographic toggle
  - Force visualization: render atomic forces as colored arrows
  - Bond color by length or by type
  - Efficient GPU batch updates (positions, colors, bonds)

Role in SPAMMM: The rendering engine. Used by KekuleExplorerGUI as the main 3D
canvas and by any module that needs molecular visualization (AFMExtension,
MolecularBrowser when ported to VisPy).
"""

import numpy as np

from PyQt5 import QtCore
import vispy

_VERBOSE = 2
def debug_print(level, msg):
    if _VERBOSE >= level: print(msg)
vispy.use('pyqt5')
from vispy import scene
from vispy.scene import visuals
from vispy.color import Colormap


def _as_f32(x):
    return np.asarray(x, dtype=np.float32)


def compute_bond_colors_by_length(bonds, pos, color_range=(0.0, 1.0)):
    """Compute bond colors based on bond length (blue=short, red=long).
    
    Args:
        bonds: List/array of (ia, ja) bond indices
        pos: (n,3) array of positions
        color_range: (min, max) color values for blue-red mapping
    
    Returns:
        bond_segs: (2*m, 3) array of segment endpoints
        bond_colors: (2*m, 4) array of RGBA colors
    """
    bond_lengths = []
    for b in bonds:
        ia, ja = b
        p1, p2 = pos[ia], pos[ja]
        d = np.linalg.norm(p1 - p2)
        bond_lengths.append(d)
    bond_lengths = np.array(bond_lengths)
    vmin, vmax = bond_lengths.min(), bond_lengths.max()
    
    bond_segs = []
    bond_colors = []
    for i, b in enumerate(bonds):
        ia, ja = b
        p1, p2 = pos[ia], pos[ja]
        bond_segs.append(p1)
        bond_segs.append(p2)
        
        if abs(vmax - vmin) < 1e-4:
            f = 0.5
        else:
            f = (bond_lengths[i] - vmin) / (vmax - vmin)
        color = (f, 0.0, 1.0 - f, 0.8)
        bond_colors.append(color)
        bond_colors.append(color)
    
    return np.array(bond_segs, dtype=np.float32), np.array(bond_colors, dtype=np.float32)


def compute_bond_colors_by_delta(bonds, pos, delta, scale=0.08):
    """Bond colors from Δlength vs reference frame [Å]. Blue=shortened, red=elongated."""
    bond_segs, bond_colors = [], []
    scale = max(float(scale), 1e-6)
    for i, b in enumerate(bonds):
        ia, ja = int(b[0]), int(b[1])
        bond_segs.extend([pos[ia], pos[ja]])
        t = float(np.clip(delta[i] / scale, -1.0, 1.0))
        f = 0.5 + 0.5 * t
        bond_colors.extend([[f, 0.25, 1.0 - f, 1.0], [f, 0.25, 1.0 - f, 1.0]])
    return np.array(bond_segs, dtype=np.float32), np.array(bond_colors, dtype=np.float32)


def generate_atom_labels(label_mode, pos, enames, atom_npi=None, backend=None, bonds=None):
    """Generate text labels for atoms based on label_mode.
    
    Args:
        label_mode: String specifying label type
        pos: (n,3) array of positions
        enames: List/array of element names
        atom_npi: Optional list of npi values (-1=H_cap, 0=sp3, 1=sp2, 2=sp)
        backend: Optional backend object
        bonds: Optional list of bonds for bond length labels
    
    Returns:
        lbl_pos: List of label positions
        lbl_texts: List of label text strings
    """
    lbl_pos = []
    lbl_texts = []
    
    if label_mode == 'Element+Index':
        for i, e in enumerate(enames):
            if e != 'H':
                lbl_pos.append(pos[i])
                lbl_texts.append(f"{e}{i}")
    elif label_mode == 'Atomic Type':
        _npi_label = {0: 'sp3', 1: 'sp2', 2: 'sp', -1: 'H_cap'}
        for i, npi in enumerate(atom_npi or []):
            if enames[i] != 'H':
                lbl_pos.append(pos[i])
                lbl_texts.append(_npi_label.get(npi, str(npi)))
    elif label_mode == 'Pi Orbitals':
        for i in range(len(enames)):
            if i < len(atom_npi or []):
                if enames[i] != 'H':
                    lbl_pos.append(pos[i])
                    lbl_texts.append(str(atom_npi[i]))
    elif label_mode == 'Z-Height':
        for i, e in enumerate(enames):
            if e != 'H':
                lbl_pos.append(pos[i])
                lbl_texts.append(f"{pos[i, 2]:.2f}")
    elif label_mode == 'Charge':
        charges = backend.atom_charges if backend is not None else None
        for i in range(len(enames)):
            lbl_pos.append(pos[i])
            q = charges[i] if charges is not None and i < len(charges) else 0.0
            lbl_texts.append(f"{q:+.3f}")
    elif label_mode == 'Bond Lengths':
        if bonds:
            for b in bonds:
                ia, ja = b
                p1, p2 = pos[ia], pos[ja]
                d = np.linalg.norm(p1 - p2)
                lbl_pos.append((p1 + p2) * 0.5)
                lbl_texts.append(f"{d:.3f}")
    
    return lbl_pos, lbl_texts


class AtomScene(QtCore.QObject):
    """Reusable Vispy widget for atoms (+ optional bonds) with orthographic top-down view.

    This is intended as a generic MD/FF viewer: operates on arrays (positions, colors, sizes, bonds)
    and optional per-atom vectors (forces) for visualization.

    Picking/dragging is implemented in a pseudo-2D mode:
    - camera is fixed to top-down view
    - drag moves atoms in XY plane (Z unchanged unless you set it externally)
    """

    sig_atom_picked = QtCore.pyqtSignal(int)  # Atom._id
    sig_drag_state = QtCore.pyqtSignal(int, int, object)  # active(0/1), Atom._id, pos3
    sig_atom_moved = QtCore.pyqtSignal(int, object)  # Atom._id, pos3 - emitted during drag
    sig_rmb_remove = QtCore.pyqtSignal(int)  # Atom._id to remove
    sig_selection_changed = QtCore.pyqtSignal(object)  # set of selected Atom._ids
    sig_camera_changed = QtCore.pyqtSignal()  # camera changed (zoom/pan/rotate)
    sig_link_bond = QtCore.pyqtSignal(int, int)  # from_id, to_id — Ctrl+drag bond creation
    sig_link_to_pos = QtCore.pyqtSignal(int, float, float)  # from_id, x, y — Ctrl+drag to empty space (create atom + bond)
    sig_atom_clicked = QtCore.pyqtSignal(int)  # Atom._id — click without drag (for type change)

    def __init__(self, *, bgcolor='white', backend=None):
        super().__init__(parent=None)
        self.backend = backend  # Reference to backend for authoritative geometry

        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor=bgcolor, show=False)
        self.view = self.canvas.central_widget.add_view()
        # Ortho top-down like MolecularPlacerVisPy
        self.view.camera = scene.TurntableCamera(fov=0, distance=80, elevation=90, azimuth=0)
        self.view.camera.interactive = False  # disable default camera mouse handling; we handle RMB explicitly

        self._cam_debug = 0
        self._rmb_down = False
        self._rmb_last = None
        self._cam_rot_speed = 0.3  # deg per pixel
        self._cam_zoom_speed = 0.12
        self._cam_zoom_min = 1e-4
        self._cam_zoom_max = 1e+4
        self._cam_pan_speed = 2.0  # world units per key press

        # Draw ordering: radius behind everything, then bboxes/links/lines, then atom centers, then labels.
        self.radius_markers = visuals.Markers(parent=self.view.scene)
        self.bbox_lines = visuals.Line(parent=self.view.scene, color=(0.8, 0.0, 0.0, 0.9), width=2.0, antialias=True, method='gl')
        self.inbox_lines = visuals.Line(parent=self.view.scene, color=(0.0, 0.0, 0.0, 0.35), width=1.0, antialias=True, method='gl')
        self.halo_lines  = visuals.Line(parent=self.view.scene, color=(0.8, 0.1, 0.8, 0.35), width=1.0, antialias=True, method='gl')
        self.neigh_lines = visuals.Line(parent=self.view.scene, color=(0.2, 0.2, 0.2, 0.65), width=1.0, antialias=True, method='gl')
        self.port_lines = visuals.Line(parent=self.view.scene, color=(1.0, 0.55, 0.0, 0.55), width=1.0, antialias=True, method='gl')
        self.port_target_lines = visuals.Line(parent=self.view.scene, color=(0.0, 0.7, 0.9, 0.55), width=1.0, antialias=True, method='gl')
        self.dpos_lines = visuals.Line(parent=self.view.scene, color=(0.9, 0.0, 0.0, 0.75), width=1.6, antialias=True, method='gl')
        self.dpos_neigh_lines = visuals.Line(parent=self.view.scene, color=(0.2, 0.2, 1.0, 0.75), width=1.6, antialias=True, method='gl')
        self.bond_lines = visuals.Line(parent=self.view.scene, color='gray', width=1.5, antialias=True, method='gl')
        self.bond_colored_lines = visuals.Line(parent=self.view.scene, color='gray', width=3.0, antialias=True, method='gl')
        self.ch_bond_lines = visuals.Line(parent=self.view.scene, color=(0.4, 0.4, 0.4, 0.6), width=1.0, antialias=True, method='gl')
        self.hbond_lines = visuals.Line(parent=self.view.scene, color=(0.8, 0.2, 0.8, 0.5), width=1.5, antialias=True, method='gl')
        self.force_lines = visuals.Line(parent=self.view.scene, color=(1, 0, 0, 0.8), width=2.0, antialias=True, method='gl')
        self.bond_order_lines = visuals.Line(parent=self.view.scene, color='gray', width=2.0, antialias=True, method='gl')
        self.bond_order_text = visuals.Text(parent=self.view.scene, color='darkblue', font_size=9, anchor_x='center', anchor_y='center')
        self.atom_markers = visuals.Markers(parent=self.view.scene)
        self.axes = visuals.XYZAxis(parent=self.view.scene)
        self.text_labels = visuals.Text(parent=self.view.scene, color='black', font_size=10, anchor_x='left', anchor_y='bottom')
        # Hover visuals for debug visualization
        self.hover_bond_line = visuals.Line(parent=self.view.scene, color='lime', width=4.0, antialias=True, method='gl')
        self.hover_ring_lines = visuals.Line(parent=self.view.scene, color='cyan', width=2.0, antialias=True, method='gl')
        self.hover_ring_markers = visuals.Markers(parent=self.view.scene)
        self.hover_ring_text = visuals.Text(parent=self.view.scene, color='cyan', font_size=12, anchor_x='center', anchor_y='center')
        self.hover_atom_marker = visuals.Markers(parent=self.view.scene)
        # Link line for Ctrl+drag bond creation (rubber-band)
        self.link_line = visuals.Line(parent=self.view.scene, color=(0.2, 0.8, 0.2, 0.8), width=3.0, antialias=True, method='gl')
        self.link_line.visible = False
        # Ring preview ghost (n-gon outline shown when hovering bond in Ring mode)
        self.ring_preview_line = visuals.Line(parent=self.view.scene, color=(0.2, 0.8, 0.8, 0.6), width=2.0, antialias=True, method='gl')
        self.ring_preview_line.visible = False
        # Selection rectangle (Line visual) - create lazily to avoid initialization issues
        self.selection_rect = None
        # Selection AABB + sticky δ (move) / φ (rotate) corner handles
        self.sel_bbox_lines = visuals.Line(parent=self.view.scene, color=(0.2, 0.55, 1.0, 0.95), width=2.0, antialias=True, method='gl')
        self.sel_handle_markers = visuals.Markers(parent=self.view.scene)
        self.sel_handle_text = visuals.Text(parent=self.view.scene, color='black', font_size=14, anchor_x='center', anchor_y='center')
        self.sel_bbox_lines.visible = False
        self.sel_handle_markers.visible = False
        self.sel_handle_text.visible = False
        # Fragment highlight visuals (dedicated — not shared with hex grid)
        self.frag_lines = visuals.Line(parent=self.view.scene, color=(1.0, 0.5, 0.0, 0.9), width=5.0, antialias=True, method='gl')
        self.frag_bbox_lines = visuals.Line(parent=self.view.scene, color=(0.2, 0.8, 0.2, 0.6), width=1.5, antialias=True, method='gl')
        self.frag_lines.visible = False
        self.frag_bbox_lines.visible = False

        # Enforce z-order when supported
        for o, v in enumerate((self.radius_markers, self.bbox_lines, self.inbox_lines, self.halo_lines, self.neigh_lines, self.port_lines, self.port_target_lines, self.dpos_lines, self.dpos_neigh_lines, self.bond_lines, self.bond_colored_lines, self.ch_bond_lines, self.hbond_lines, self.force_lines, self.bond_order_lines, self.atom_markers, self.axes, self.text_labels, self.bond_order_text, self.hover_bond_line, self.hover_ring_lines, self.hover_ring_markers, self.hover_ring_text, self.hover_atom_marker, self.link_line, self.ring_preview_line, self.sel_bbox_lines, self.sel_handle_markers, self.sel_handle_text, self.frag_lines, self.frag_bbox_lines)):
            if hasattr(v, 'order'):
                v.order = int(o)

        # GL state: radius translucent and never blocks other overlays
        try:
            self.radius_markers.set_gl_state('translucent', depth_test=False)
            for v in (self.bbox_lines, self.inbox_lines, self.halo_lines, self.neigh_lines, self.port_lines, self.port_target_lines, self.dpos_lines, self.dpos_neigh_lines, self.bond_lines, self.bond_colored_lines, self.ch_bond_lines, self.hbond_lines, self.force_lines, self.bond_order_lines, self.hover_bond_line, self.hover_ring_lines, self.link_line, self.ring_preview_line, self.sel_bbox_lines, self.frag_lines, self.frag_bbox_lines):
                v.set_gl_state('translucent', depth_test=False)
            self.atom_markers.set_gl_state('translucent', depth_test=False)
            self.hover_ring_markers.set_gl_state('translucent', depth_test=False)
            self.hover_atom_marker.set_gl_state('translucent', depth_test=False)
            self.sel_handle_markers.set_gl_state('translucent', depth_test=False)
            self.text_labels.set_gl_state('translucent', depth_test=False)
            self.bond_order_text.set_gl_state('translucent', depth_test=False)
            self.hover_ring_text.set_gl_state('translucent', depth_test=False)
            self.sel_handle_text.set_gl_state('translucent', depth_test=False)
        except Exception:
            pass

        self._pos = np.zeros((0, 3), dtype=np.float32)
        self._colors = None
        self._sizes = None
        self._bonds = None
        self._forces = None
        self._radius = None
        self._bond_orders = None        # (m,) pi bond orders for heavy-atom bonds
        self._bond_order_bonds = None   # (m,2) bond indices matching _bond_orders
        self._show_bond_order_labels = False
        self._atom_ids = np.zeros((0,), dtype=np.int64)  # Atom._id parallel to _pos
        self._id_to_idx = {}  # Atom._id → array index, rebuilt on set_data

        self._render_mask = None
        self._group_size = 64
        self._color_by_group = False
        self._colors_base = None

        self._show_radius = False
        self._show_bboxes = False
        self._show_inbox_links = False
        self._show_halo_links = False
        self._show_neigh_bonds = False
        self._show_port_tips = False
        self._show_port_targets = False
        self._show_dpos = False
        self._show_dpos_neigh = False
        self._show_axes = True
        self._bboxes_min = None
        self._bboxes_max = None
        self._inbox_link_segs = None
        self._halo_link_segs = None
        self._inbox_link_gid = None
        self._halo_link_gid = None

        self._neigh_segs = None
        self._port_tip_segs = None
        self._port_target_segs = None
        self._dpos_segs = None
        self._dpos_neigh_segs = None

        # Fragment highlight data (persists across _redraw calls)
        self._frag_bond_segs = None       # (2*m, 3) array of bond segment endpoints
        self._frag_bond_colors = None     # (2*m, 4) RGBA per vertex
        self._frag_bbox_segs = None       # (2*k, 3) AABB edge segments
        self._frag_bbox_colors = None     # (2*k, 4) RGBA per vertex

        self._label_mode = 'none'  # none|global|local|pair|radius
        self._labels_text = None

        self._marker_style = 'disc'      # vispy marker name
        self._radius_style = 'disc'

        self._pick_active = False
        self._pick_idx = -1
        self._pick_id = -1  # Atom._id of picked atom (-1 = none)
        self._pick_z = 0.0
        self._press_pos = None
        self.lock_drag = False       # Set True externally to suppress all atom drag
        self.pick_radius = 0.5       # Max distance (Angstroms) for RMB atom picking

        self._pick_mode = '2d'   # '2d' or '3d'
        self._lock_top_view = True
        self._clamp_xy = False
        self._fixed = None
        self._allow_mouse_orbit = False  # v1: keyboard rotate only
        self._depth_test = False
        self._view_debug = False
        self._cam_rot_key_speed = 5.0  # deg per arrow key in 3D

        # Debug: mouse ray + hit point (for transform sanity)
        self.debug_ray_line = visuals.Line(parent=self.view.scene, color=(1.0, 0.2, 0.2, 0.85), width=2.0, antialias=True, method='gl')
        self.debug_ray_line.visible = False
        self.debug_hit_marker = visuals.Markers(parent=self.view.scene)
        self.debug_hit_marker.visible = False
        try:
            self.debug_ray_line.set_gl_state('translucent', depth_test=False)
            self.debug_hit_marker.set_gl_state('translucent', depth_test=False)
        except Exception:
            pass

        # Selection state — uses Atom._id for robustness across topology changes
        self._selection_mode = False  # If True, Select-mode interaction (add/remove + bbox handles)
        self._selected_ids = set()  # set of Atom._id
        self._selection_start = None
        self._selection_end = None
        self._selected_colors_backup = None  # Store original colors for selected atoms
        # Sticky selection transform: None | 'move' (δ) | 'rotate' (φ)
        self._xform_mode = None
        self._xform_last_xy = None
        self._xform_com = None
        self._xform_last_angle = None
        self._sel_handle_move = None   # (2,) world xy of δ corner
        self._sel_handle_rot = None    # (2,) world xy of φ corner
        self._sel_handle_r = 0.55      # pick radius for handles (Å)
        self._sel_bbox_pad = 0.45
        self._box_sel_mode = 'add'     # 'add' | 'remove' for rubber-band box

        self._drag_plane_p0 = None
        self._drag_plane_n = None

        # Link mode state — Ctrl+drag to create bonds between atoms
        self._link_active = False
        self._link_from_id = -1
        self._link_mode = False  # Set True externally when in Atom mode (enables Ctrl+drag bond creation)
        self._last_ctrl = False  # Store Ctrl state from press for RMB bridge removal

        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_wheel.connect(self._on_mouse_wheel)
        self.canvas.events.key_press.connect(self._on_key_press)

        self._apply_camera_mode()

    @property
    def widget(self):
        return self.canvas.native

    def _sync_fixed_mask(self):
        """Keep _fixed length matched to _pos (topology changes invalidate old FF masks)."""
        n = int(self._pos.shape[0])
        if (self._fixed is None) or (self._fixed.shape[0] != n):
            self._fixed = np.zeros((n,), dtype=bool)

    def set_data(self, pos, *, colors=None, sizes=None, bonds=None, forces=None, force_scale=1.0):
        pos = _as_f32(pos)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"AtomScene.set_data: pos.shape={pos.shape} expected (n,3)")
        # If backend is provided, use its authoritative positions as reference (not copy)
        # But keep a copy for rendering to avoid issues during camera changes
        if self.backend is not None:
            self._pos = self.backend.sys.apos.astype(np.float32).copy()
            # Build _atom_ids and _id_to_idx from backend for ID-based picking
            if hasattr(self.backend, '_atom_ids') and self.backend._atom_ids is not None:
                self._atom_ids = self.backend._atom_ids.copy()
            else:
                self._atom_ids = np.arange(len(self._pos), dtype=np.int64)
            self._id_to_idx = {int(aid): i for i, aid in enumerate(self._atom_ids)}
        else:
            self._pos = pos
            self._atom_ids = np.arange(len(pos), dtype=np.int64)
            self._id_to_idx = {i: i for i in range(len(pos))}
        self._render_mask = None
        self._sync_fixed_mask()
        n = int(self._pos.shape[0])
        # colors/sizes must match _pos (backend may differ from the pos argument)
        if colors is not None:
            colors = _as_f32(colors)
            if colors.shape[0] != n:
                raise ValueError(f"AtomScene.set_data: colors.shape[0]={colors.shape[0]} != n_pos={n}")
        if sizes is not None:
            sizes = _as_f32(sizes)
            if sizes.shape[0] != n:
                raise ValueError(f"AtomScene.set_data: sizes.shape[0]={sizes.shape[0]} != n_pos={n}")
        self._colors = None if colors is None else colors
        self._colors_base = None if colors is None else colors.copy()
        self._sizes = None if sizes is None else sizes
        self._bonds = None if bonds is None else np.asarray(bonds, dtype=np.int32)
        self._forces = None if forces is None else _as_f32(forces)
        self._force_scale = float(force_scale)
        self._redraw()

    def set_marker_style(self, style='disc'):
        style = str(style)
        self._marker_style = style
        self._redraw()

    def set_bond_orders(self, bonds, bond_orders, show_labels=False):
        """Set pi bond orders for visual rendering of double/aromatic bonds.

        Args:
            bonds: (m,2) int array of bond indices into self._pos
            bond_orders: (m,) float array of pi bond orders (0=single, 0.5=aromatic, 1=double)
            show_labels: if True, render bond order values as text at midpoints
        """
        if bonds is None or bond_orders is None:
            self._bond_order_bonds = None
            self._bond_orders = None
        else:
            self._bond_order_bonds = np.asarray(bonds, dtype=np.int32)
            self._bond_orders = np.asarray(bond_orders, dtype=np.float32)
        self._show_bond_order_labels = show_labels
        self._redraw()

    def set_bond_order_labels(self, show):
        """Toggle bond order label display."""
        self._show_bond_order_labels = show
        self._redraw()

    def set_radius_style(self, style='disc'):
        style = str(style)
        self._radius_style = style
        self._redraw()

    def set_radius(self, radius):
        r = np.asarray(radius, dtype=np.float32)
        if r.shape != (self._pos.shape[0],):
            raise ValueError(f"AtomScene.set_radius: radius.shape={r.shape} expected ({self._pos.shape[0]},)")
        self._radius = r
        self._redraw()

    def set_render_mask(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self._pos.shape[0],):
            raise ValueError(f"AtomScene.set_render_mask: mask.shape={mask.shape} expected ({self._pos.shape[0]},)")
        self._render_mask = mask.copy()
        self._redraw()

    def set_group_size(self, group_size):
        self._group_size = int(group_size)

    def set_color_by_group(self, enable):
        self._color_by_group = bool(enable)
        self._redraw()

    def set_show_radius(self, enable):
        self._show_radius = bool(enable)
        self._redraw()

    def set_show_bboxes(self, enable):
        self._show_bboxes = bool(enable)
        self._redraw()

    def set_show_inbox_links(self, show):
        self._show_inbox_links = bool(show)
        self._redraw()

    def set_show_halo_links(self, show):
        self._show_halo_links = bool(show)
        self._redraw()

    def set_show_neigh_bonds(self, show):
        self._show_neigh_bonds = bool(show)
        self._redraw()

    def set_show_port_tips(self, show):
        self._show_port_tips = bool(show)
        self._redraw()

    def set_show_port_targets(self, show):
        self._show_port_targets = bool(show)
        self._redraw()

    def set_show_dpos(self, show):
        self._show_dpos = bool(show)
        self._redraw()

    def set_show_dpos_neigh(self, show):
        self._show_dpos_neigh = bool(show)
        self._redraw()

    def set_show_axes(self, enable):
        self._show_axes = bool(enable)
        self.axes.visible = bool(enable)
        self.canvas.update()

    def set_inbox_links(self, segs, *, gids=None):
        if segs is None:
            self._inbox_link_segs = None
            self._inbox_link_gid = None
        else:
            s = _as_f32(segs)
            if s.ndim != 2 or s.shape[1] != 3 or (s.shape[0] % 2) != 0:
                raise ValueError(f"AtomScene.set_inbox_links: segs.shape={s.shape} expected (2*m,3)")
            self._inbox_link_segs = s
            if gids is None:
                self._inbox_link_gid = None
            else:
                g = np.asarray(gids, dtype=np.int32)
                if g.shape != (s.shape[0] // 2,):
                    raise ValueError(f"AtomScene.set_inbox_links: gids.shape={g.shape} expected ({s.shape[0]//2},)")
                self._inbox_link_gid = g
        self._redraw()

    def set_halo_links(self, segs, *, gids=None):
        if segs is None:
            self._halo_link_segs = None
            self._halo_link_gid = None
        else:
            s = _as_f32(segs)
            if s.ndim != 2 or s.shape[1] != 3 or (s.shape[0] % 2) != 0:
                raise ValueError(f"AtomScene.set_halo_links: segs.shape={s.shape} expected (2*m,3)")
            self._halo_link_segs = s
            if gids is None:
                self._halo_link_gid = None
            else:
                g = np.asarray(gids, dtype=np.int32)
                if g.shape != (s.shape[0] // 2,):
                    raise ValueError(f"AtomScene.set_halo_links: gids.shape={g.shape} expected ({s.shape[0]//2},)")
                self._halo_link_gid = g
        self._redraw()

    def set_neigh_bonds(self, segs):
        if segs is None:
            self._neigh_segs = None
        else:
            s = _as_f32(segs)
            if s.ndim != 2 or s.shape[1] != 3 or (s.shape[0] % 2) != 0:
                raise ValueError(f"AtomScene.set_neigh_bonds: segs.shape={s.shape} expected (2*m,3)")
            self._neigh_segs = s
        self._redraw()

    def set_port_tips(self, segs):
        if segs is None:
            self._port_tip_segs = None
        else:
            s = _as_f32(segs)
            if s.ndim != 2 or s.shape[1] != 3 or (s.shape[0] % 2) != 0:
                raise ValueError(f"AtomScene.set_port_tips: segs.shape={s.shape} expected (2*m,3)")
            self._port_tip_segs = s
        self._redraw()

    def set_port_targets(self, segs):
        if segs is None:
            self._port_target_segs = None
        else:
            s = _as_f32(segs)
            if s.ndim != 2 or s.shape[1] != 3 or (s.shape[0] % 2) != 0:
                raise ValueError(f"AtomScene.set_port_targets: segs.shape={s.shape} expected (2*m,3)")
            self._port_target_segs = s
        self._redraw()

    def set_dpos(self, segs):
        if segs is None:
            self._dpos_segs = None
        else:
            s = _as_f32(segs)
            if s.ndim != 2 or s.shape[1] != 3 or (s.shape[0] % 2) != 0:
                raise ValueError(f"AtomScene.set_dpos: segs.shape={s.shape} expected (2*m,3)")
            self._dpos_segs = s
        self._redraw()

    def set_dpos_neigh(self, segs):
        if segs is None:
            self._dpos_neigh_segs = None
        else:
            s = _as_f32(segs)
            if s.ndim != 2 or s.shape[1] != 3 or (s.shape[0] % 2) != 0:
                raise ValueError(f"AtomScene.set_dpos_neigh: segs.shape={s.shape} expected (2*m,3)")
            self._dpos_neigh_segs = s
        self._redraw()

    def set_bboxes(self, bmin, bmax):
        bmin = _as_f32(bmin)
        bmax = _as_f32(bmax)
        if bmin.shape != bmax.shape or bmin.ndim != 2 or bmin.shape[1] != 4:
            raise ValueError(f"AtomScene.set_bboxes: bmin.shape={bmin.shape} bmax.shape={bmax.shape} expected (ng,4)")
        self._bboxes_min = bmin
        self._bboxes_max = bmax
        self._redraw()

    def set_frag_highlights(self, bond_segs=None, bond_colors=None, bbox_segs=None, bbox_colors=None):
        """Set fragment highlight overlays. Pass None to clear.
        bond_segs: (2*m, 3) array of bond segment endpoints
        bond_colors: (2*m, 4) RGBA per vertex, or single tuple for all
        bbox_segs: (2*k, 3) array of AABB edge segments
        bbox_colors: (2*k, 4) RGBA per vertex, or single tuple for all"""
        self._frag_bond_segs = _as_f32(bond_segs) if bond_segs is not None else None
        self._frag_bbox_segs = _as_f32(bbox_segs) if bbox_segs is not None else None
        if bond_colors is not None and not isinstance(bond_colors, tuple):
            self._frag_bond_colors = np.asarray(bond_colors, dtype=np.float32)
        else:
            self._frag_bond_colors = bond_colors
        if bbox_colors is not None and not isinstance(bbox_colors, tuple):
            self._frag_bbox_colors = np.asarray(bbox_colors, dtype=np.float32)
        else:
            self._frag_bbox_colors = bbox_colors
        self._redraw()

    def set_label_mode(self, mode):
        mode = str(mode).lower()
        if mode not in ('none', 'global', 'local', 'pair', 'radius'):
            raise ValueError(f"AtomScene.set_label_mode: mode={mode} expected none|global|local|pair|radius")
        self._label_mode = mode
        self._labels_text = None
        self._redraw()

    def _px_per_world_ortho(self):
        """Pixels per 1 world unit for TurntableCamera with fov=0.

        To keep glyph size independent of camera orientation, use only zoom (scale_factor) and viewport.
        """
        cam = self.view.camera
        if cam is None:
            return 1.0
        sf = float(getattr(cam, 'scale_factor', 1.0))
        if (not np.isfinite(sf)) or (sf <= 1e-12):
            return 1.0
        tr = self.view.scene.transform
        p0 = np.array(tr.imap((0.0, 0.0, 0.0)), dtype=np.float32)
        p1 = np.array(tr.imap((1.0, 0.0, 0.0)), dtype=np.float32)
        world_len = float(np.linalg.norm(p1[:2] - p0[:2]))
        if (not np.isfinite(world_len)) or (world_len <= 1e-12):
            return 1.0
        return 1.0 / world_len

    def get_zoom(self):
        cam = self.view.camera
        if cam is None:
            return 1.0
        sf = getattr(cam, 'scale_factor', 1.0)
        return float(sf)

    def set_zoom(self, zoom):
        cam = self.view.camera
        if cam is None:
            return
        z = float(zoom)
        if z < self._cam_zoom_min:
            z = self._cam_zoom_min
        if z > self._cam_zoom_max:
            z = self._cam_zoom_max
        cam.scale_factor = z
        self._redraw()
        self.sig_camera_changed.emit()

    def reset_view(self):
        cam = self.view.camera
        if cam is None:
            return
        cam.fov = 0
        cam.azimuth = 0
        cam.elevation = 90
        cam.roll = 0
        cam.scale_factor = 1.0
        self.canvas.update()

    def set_pick_mode(self, mode):
        mode = str(mode).lower()
        if mode not in ('2d', '3d'):
            raise ValueError(f"AtomScene.set_pick_mode: mode={mode} expected '2d'|'3d'")
        self._pick_mode = mode

    def set_lock_top_view(self, lock):
        self._lock_top_view = bool(lock)
        self._apply_camera_mode()

    def set_allow_mouse_orbit(self, allow):
        self._allow_mouse_orbit = bool(allow)

    def set_view_debug(self, enabled):
        self._view_debug = bool(enabled)
        if not self._view_debug:
            self.debug_ray_line.visible = False
            self.debug_hit_marker.visible = False
            self.debug_ray_line.set_data(np.zeros((0, 3), dtype=np.float32))
            self.debug_hit_marker.set_data(np.zeros((0, 3), dtype=np.float32))

    def set_depth_test(self, enabled):
        """Enable GL depth test on atoms/bonds (for tilted 3D view). Overlays stay depth-free."""
        self._depth_test = bool(enabled)
        try:
            dt = self._depth_test
            self.atom_markers.set_gl_state('translucent', depth_test=dt)
            for v in (self.bond_lines, self.bond_colored_lines, self.ch_bond_lines, self.hbond_lines, self.force_lines, self.bond_order_lines):
                v.set_gl_state('translucent', depth_test=dt)
        except Exception:
            pass

    def set_camera_preset(self, name):
        """Ortho standard views. name: top|bottom|front|back|left|right. Keeps fov=0."""
        presets = {
            'top':    (90.0, 0.0),
            'bottom': (-90.0, 0.0),
            'front':  (0.0, 0.0),
            'back':   (0.0, 180.0),
            'left':   (0.0, -90.0),
            'right':  (0.0, 90.0),
        }
        key = str(name).lower()
        if key not in presets:
            raise ValueError(f"AtomScene.set_camera_preset: unknown {name!r}")
        el, az = presets[key]
        cam = self.view.camera
        if cam is None:
            return
        cam.fov = 0
        cam.elevation = el
        cam.azimuth = az
        cam.roll = 0
        self.canvas.update()
        self.sig_camera_changed.emit()
        self._cam_print(f'preset:{key}')

    def update_view_debug(self, mouse_pos, hit=None):
        """Draw mouse ray and optional hit point (world)."""
        if not self._view_debug:
            return
        r0, rd = self._ray_from_mouse(mouse_pos)
        # Segment through scene: use camera center distance scale
        cam = self.view.camera
        dist = float(getattr(cam, 'distance', 50.0) or 50.0)
        p0 = r0 - rd * dist
        p1 = r0 + rd * dist
        self.debug_ray_line.set_data(np.vstack([p0, p1]).astype(np.float32))
        self.debug_ray_line.visible = True
        if hit is not None:
            h = np.asarray(hit, dtype=np.float32).reshape(1, 3)
            self.debug_hit_marker.set_data(pos=h, symbol='disc', size=12, face_color=(1, 0.3, 0.1, 1), edge_color='black', edge_width=1)
            self.debug_hit_marker.visible = True
        else:
            self.debug_hit_marker.set_data(np.zeros((0, 3), dtype=np.float32))
            self.debug_hit_marker.visible = False

    def set_camera_debug(self, level=1):
        self._cam_debug = int(level)

    def set_clamp_xy(self, clamp):
        self._clamp_xy = bool(clamp)

    def set_fixed_mask(self, fixed):
        fixed = np.asarray(fixed, dtype=bool)
        if fixed.shape != (self._pos.shape[0],):
            raise ValueError(f"AtomScene.set_fixed_mask: fixed.shape={fixed.shape} expected ({self._pos.shape[0]},)")
        self._fixed = fixed.copy()

    def toggle_fixed(self, i):
        i = int(i)
        if i < 0 or i >= self._pos.shape[0]:
            raise ValueError(f"AtomScene.toggle_fixed: i={i} out of range")
        self._fixed[i] = ~self._fixed[i]
        return bool(self._fixed[i])

    def is_fixed(self, i):
        i = int(i)
        if self._fixed is None:
            return False
        return bool(self._fixed[i])

    def update_positions(self, pos):
        pos = _as_f32(pos)
        if pos.shape != self._pos.shape:
            raise ValueError(f"AtomScene.update_positions: pos.shape={pos.shape} != current {self._pos.shape}")
        self._pos = pos
        self._redraw()

    def _apply_camera_mode(self):
        cam = self.view.camera
        if cam is None:
            return
        cam.interactive = False
        if self._lock_top_view:
            cam.fov = 0
            cam.elevation = 90
            cam.azimuth = 0
            cam.roll = 0
        # leave distance/center as is

    def _cam_print(self, tag):
        if int(self._cam_debug) <= 0:
            return
        cam = self.view.camera
        if cam is None:
            return
        print(f"[CAM] {tag} az={float(cam.azimuth):.3f} el={float(cam.elevation):.3f} dist={float(cam.distance):.3f}")

    def _cam_rotate(self, dx_px, dy_px):
        if self._lock_top_view:
            return
        cam = self.view.camera
        if cam is None:
            return
        cam.azimuth = float(cam.azimuth) + float(dx_px) * float(self._cam_rot_speed)
        cam.elevation = float(cam.elevation) + float(dy_px) * float(self._cam_rot_speed)
        if cam.elevation > 90.0:
            cam.elevation = 90.0
        if cam.elevation < -90.0:
            cam.elevation = -90.0
        # Camera-only update — do NOT _redraw() or emit sig_camera_changed here
        # (that triggers refresh_view during mouse_move → VisPy EventEmitter loop).
        self.canvas.update()
        self._cam_print('rotate')

    def _cam_rotate_keys(self, daz, del_):
        """Rotate by absolute degrees (keyboard)."""
        if self._lock_top_view:
            return
        cam = self.view.camera
        if cam is None:
            return
        cam.azimuth = float(cam.azimuth) + float(daz)
        cam.elevation = float(np.clip(float(cam.elevation) + float(del_), -90.0, 90.0))
        self.canvas.update()
        self._cam_print('rotate_key')

    def _cam_zoom(self, delta):
        cam = self.view.camera
        if cam is None:
            return
        # Orthographic zoom: change camera scale_factor (distance does not change zoom for fov=0).
        z0 = float(getattr(cam, 'scale_factor', 1.0))
        s = float(np.exp(-float(delta) * float(self._cam_zoom_speed)))
        z1 = z0 * s
        if z1 < self._cam_zoom_min:
            z1 = self._cam_zoom_min
        if z1 > self._cam_zoom_max:
            z1 = self._cam_zoom_max
        cam.scale_factor = z1
        self.canvas.update()
        if int(self._cam_debug) > 0:
            print(f"[CAM] zoom delta={float(delta):.6g} scale:{z0:.6g}->{z1:.6g}")

    def _cam_pan(self, dx, dy):
        cam = self.view.camera
        if cam is None:
            return
        # Pan camera center by dx, dy in world units
        center = np.array(cam.center)
        center[0] += float(dx) * float(self._cam_pan_speed)
        center[1] += float(dy) * float(self._cam_pan_speed)
        cam.center = tuple(center)
        self.canvas.update()
        if int(self._cam_debug) > 0:
            print(f"[CAM] pan dx={float(dx):.3f} dy={float(dy):.3f} center={tuple(center)}")

    def _ray_from_mouse(self, mouse_pos, z0=0.0, z1=1.0):
        # mouse_pos in canvas pixels
        # If the view is shifted (e.g. in a Grid), we must use local coordinates
        view_pos = np.array(mouse_pos) - self.view.pos[:2]
        tr = self.view.scene.transform
        p0 = np.array(tr.imap((view_pos[0], view_pos[1], float(z0)))[:3], dtype=np.float32)
        p1 = np.array(tr.imap((view_pos[0], view_pos[1], float(z1)))[:3], dtype=np.float32)
        d = p1 - p0
        dn = float(np.linalg.norm(d))
        if dn <= 1e-20:
            d = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            d /= dn
        return p0, d

    def _pick_idx_from_ray(self, r0, rd):
        # closest point distance^2 to ray for each atom
        # d2 = |(p-r0) - rd*dot(rd,(p-r0))|^2
        self._sync_fixed_mask()
        valid = np.isfinite(self._pos).all(axis=1)
        if self._fixed is not None:
            valid &= (~self._fixed)
        if not np.any(valid):
            return -1, 1e30
        dp = self._pos - r0[None, :]
        t = np.sum(dp * rd[None, :], axis=1)
        q = dp - rd[None, :] * t[:, None]
        d2 = np.sum(q * q, axis=1)
        d2 = np.where(valid, d2, 1e30)
        i = int(np.argmin(d2))
        return i, float(d2[i])

    def _validated_segs(self, tag, segs):
        s = np.asarray(segs, dtype=np.float32)
        if s.ndim != 2 or s.shape[1] != 3 or (s.shape[0] % 2) != 0:
            raise ValueError(f"{tag}: segs.shape={s.shape} expected (2*m,3)")
        if not np.isfinite(s).all():
            bad = np.where(~np.isfinite(s))[0][:10]
            raise ValueError(f"{tag}: non-finite entries at idx={bad.tolist()}")
        return s

    def _line_set(self, tag, visual, segs, *, color=None, width=None, connect=None):
        if segs is None:
            visual.set_data(np.zeros((0, 3), dtype=np.float32))
            return
        try:
            s = self._validated_segs(tag, segs)
        except Exception as e:
            print(f"[VISPY-SEG-ERR] {tag} validation failed: {e}")
            visual.set_data(np.zeros((0, 3), dtype=np.float32))
            return
        if s.size == 0:
            visual.set_data(np.zeros((0, 3), dtype=np.float32))
            return
        conn = connect
        if conn is None:
            conn = np.zeros((s.shape[0],), dtype=bool); conn[0::2] = True
        # validate color length if array-like
        if hasattr(color, 'shape') and hasattr(color, '__len__'):
            clen = int(len(color))
            if clen not in (0, s.shape[0]):
                print(f"[VISPY-SEG-ERR] {tag} color length mismatch: len(color)={clen} verts={s.shape[0]}")
                color = None
        try:
            visual.set_data(s, connect=conn, color=color, width=width)
        except Exception as e:
            stats = {
                'tag': tag,
                'segs_shape': s.shape,
                'segs_min': float(np.min(s)) if s.size else None,
                'segs_max': float(np.max(s)) if s.size else None,
                'connect_shape': getattr(conn, 'shape', None),
                'color_shape': getattr(color, 'shape', None) if hasattr(color, 'shape') else None,
                'width': width,
            }
            print(f"[VISPY-LINE-ERR] {tag} set_data failed: {e} | stats={stats}")
            visual.set_data(np.zeros((0, 3), dtype=np.float32))

    def _intersect_ray_plane(self, r0, rd, p0, n):
        denom = float(np.dot(rd, n))
        if abs(denom) < 1e-12:
            return None
        t = float(np.dot((p0 - r0), n) / denom)
        return r0 + rd * t

    def _redraw(self):
        # Don't sync _pos from backend here - causes stale bond index issues during camera changes
        # The GUI should call refresh_view() when atoms actually change
        # Camera changes should only re-render, not re-sync data
        self._sync_fixed_mask()
        # Drop stale per-atom arrays (e.g. overlay redraw before set_data after topology change)
        n = int(self._pos.shape[0])
        if self._colors is not None and self._colors.shape[0] != n:
            debug_print(1, f"[REDRAW] drop stale _colors {self._colors.shape[0]} vs n={n}")
            self._colors = None
            self._colors_base = None
        if self._sizes is not None and self._sizes.shape[0] != n:
            debug_print(1, f"[REDRAW] drop stale _sizes {self._sizes.shape[0]} vs n={n}")
            self._sizes = None

        if self._pos.size == 0:
            idx = np.array([], dtype=int)
        else:
            m = self._render_mask
            if m is None:
                idx = np.arange(self._pos.shape[0], dtype=int)
            else:
                idx = np.where(m)[0].astype(int)

        if idx.size == 0:
            self.atom_markers.set_data(np.zeros((0, 3), dtype=np.float32))
            self.radius_markers.set_data(np.zeros((0, 3), dtype=np.float32))
            self.bbox_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.inbox_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.halo_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.neigh_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.port_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.port_target_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.dpos_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.dpos_neigh_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.bond_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.bond_colored_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.ch_bond_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.hbond_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.force_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            return

        # Colors
        if self._colors is None:
            # Default coloring
            face_color = np.zeros((idx.size, 4), dtype=np.float32)
            face_color[:, 0] = 0.5
            face_color[:, 1] = 0.5
            face_color[:, 2] = 0.5
            face_color[:, 3] = 1.0

        if self._color_by_group:
            # deterministic HSV-like palette per group
            g = (idx // int(self._group_size)).astype(np.int32)
            c = np.empty((idx.size, 4), dtype=np.float32)
            for i, gi in enumerate(g):
                h = float((gi * 0.61803398875) % 1.0)
                r = abs(h * 6.0 - 3.0) - 1.0
                g1 = 2.0 - abs(h * 6.0 - 2.0)
                b = 2.0 - abs(h * 6.0 - 4.0)
                rgb = np.clip(np.array([r, g1, b], dtype=np.float32), 0.0, 1.0)
                c[i, :3] = rgb
                c[i, 3] = 0.9
            face_color = c
        else:
            if self._colors is None:
                face_color = (0.2, 0.2, 0.2, 1.0)
            else:
                face_color = self._colors[idx]
        if self._sizes is None:
            size = 8.0
            size = np.full((idx.size,), float(size), dtype=np.float32)
        else:
            size = _as_f32(self._sizes[idx])

        # marker style (disc/square, etc.)
        try:
            self.atom_markers.set_data(self._pos[idx], face_color=face_color, size=size, edge_width=0.5, edge_color='black', symbol=self._marker_style)
        except TypeError:
            # older vispy uses 'marker' kw
            self.atom_markers.set_data(self._pos[idx], face_color=face_color, size=size, edge_width=0.5, edge_color='black', marker=self._marker_style)

        if self._show_radius and (self._radius is not None):
            # world radius -> exact screen size scaling for orthographic camera (depends only on zoom)
            r = np.maximum(self._radius[idx], 0.0)
            px_per_world = float(self._px_per_world_ortho())
            sizeR = (2.0 * r * px_per_world).astype(np.float32)
            colR = np.zeros((idx.size, 4), dtype=np.float32)
            colR[:, :3] = face_color[:, :3] if isinstance(face_color, np.ndarray) else np.array(face_color[:3], dtype=np.float32)[None, :]
            colR[:, 3] = 0.10
            try:
                self.radius_markers.set_data(self._pos[idx], face_color=colR, size=sizeR, edge_width=0.0, symbol=self._radius_style)
            except TypeError:
                self.radius_markers.set_data(self._pos[idx], face_color=colR, size=sizeR, edge_width=0.0, marker=self._radius_style)
        else:
            self.radius_markers.set_data(np.zeros((0, 3), dtype=np.float32))

        # Debug link lines
        if self._show_inbox_links and (self._inbox_link_segs is not None) and (self._inbox_link_segs.size > 0):
            if self._color_by_group and (self._inbox_link_gid is not None):
                gid = self._inbox_link_gid
                col = np.empty((gid.size * 2, 4), dtype=np.float32)
                for i, gi in enumerate(gid):
                    h = float((int(gi) * 0.61803398875) % 1.0)
                    r = abs(h * 6.0 - 3.0) - 1.0
                    g1 = 2.0 - abs(h * 6.0 - 2.0)
                    b = 2.0 - abs(h * 6.0 - 4.0)
                    rgb = np.clip(np.array([r, g1, b], dtype=np.float32), 0.0, 1.0)
                    col[2*i+0, :3] = rgb; col[2*i+1, :3] = rgb
                    col[2*i+0, 3] = 0.35; col[2*i+1, 3] = 0.35
                self._line_set("inbox_links", self.inbox_lines, self._inbox_link_segs, color=col)
            else:
                self._line_set("inbox_links", self.inbox_lines, self._inbox_link_segs)
        else:
            self.inbox_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        if self._show_halo_links and (self._halo_link_segs is not None) and (self._halo_link_segs.size > 0):
            if self._color_by_group and (self._halo_link_gid is not None):
                gid = self._halo_link_gid
                col = np.empty((gid.size * 2, 4), dtype=np.float32)
                for i, gi in enumerate(gid):
                    h = float((int(gi) * 0.61803398875) % 1.0)
                    r = abs(h * 6.0 - 3.0) - 1.0
                    g1 = 2.0 - abs(h * 6.0 - 2.0)
                    b = 2.0 - abs(h * 6.0 - 4.0)
                    rgb = np.clip(np.array([r, g1, b], dtype=np.float32), 0.0, 1.0)
                    col[2*i+0, :3] = rgb; col[2*i+1, :3] = rgb
                    col[2*i+0, 3] = 0.35; col[2*i+1, 3] = 0.35
                self._line_set("halo_links", self.halo_lines, self._halo_link_segs, color=col)
            else:
                self._line_set("halo_links", self.halo_lines, self._halo_link_segs)
        else:
            self.halo_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        if self._show_neigh_bonds and (self._neigh_segs is not None) and (self._neigh_segs.size > 0):
            self._line_set("neigh_bonds", self.neigh_lines, self._neigh_segs)
        else:
            self.neigh_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        if self._show_port_tips and (self._port_tip_segs is not None) and (self._port_tip_segs.size > 0):
            self._line_set("port_tips", self.port_lines, self._port_tip_segs)
        else:
            self.port_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        if self._show_port_targets and (self._port_target_segs is not None) and (self._port_target_segs.size > 0):
            self._line_set("port_targets", self.port_target_lines, self._port_target_segs)
        else:
            self.port_target_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        if self._show_dpos and (self._dpos_segs is not None) and (self._dpos_segs.size > 0):
            self._line_set("dpos", self.dpos_lines, self._dpos_segs)
        else:
            self.dpos_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        if self._show_dpos_neigh and (self._dpos_neigh_segs is not None) and (self._dpos_neigh_segs.size > 0):
            self._line_set("dpos_neigh", self.dpos_neigh_lines, self._dpos_neigh_segs)
        else:
            self.dpos_neigh_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        # Bonds: draw segment pairs (normal bonds - GUI handles CH/H-bonds separately)
        if (self._bonds is not None) and (self._bonds.size > 0):
            b = self._bonds
            if m is not None:
                mb = m[b[:, 0]] & m[b[:, 1]]
                b = b[mb]
            if b.size > 0:
                segs = np.empty((b.shape[0] * 2, 3), dtype=np.float32)
                segs[0::2] = self._pos[b[:, 0]]
                segs[1::2] = self._pos[b[:, 1]]
                self._line_set("bonds", self.bond_lines, segs, color=(0.3, 0.3, 0.3, 0.8), width=1.5)
            else:
                self.bond_lines.set_data(np.zeros((0, 3), dtype=np.float32))
        else:
            self.bond_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        # During sticky δ/φ: keep CH/H-bond/BO/label/frag overlays hidden (stale vs moving atoms)
        if self._xform_mode is not None:
            self.hide_xform_overlays()
            self.canvas.update()
            return

        # Bond orders: render double bonds as parallel lines, aromatic in green
        if (self._bond_order_bonds is not None) and (self._bond_orders is not None) and (len(self._bond_orders) > 0):
            b = self._bond_order_bonds
            bo = self._bond_orders
            pos = self._pos
            # Build segments: single=1 line, aromatic=1 green line, double=2 parallel lines
            seg_list = []
            col_list = []
            mid_list = []
            label_list = []
            for ib in range(len(bo)):
                i, j = b[ib]
                p0 = pos[i]
                p1 = pos[j]
                mid = 0.5 * (p0 + p1)
                d = p1 - p0
                dl = np.linalg.norm(d)
                if dl < 1e-10:
                    continue
                d /= dl
                # perpendicular in xy plane
                perp = np.array([-d[1], d[0], 0.0], dtype=np.float32)
                perp /= (np.linalg.norm(perp) + 1e-20)
                offset = 0.12  # parallel line offset in Angstroms
                pi_bo = bo[ib]
                total = 1.0 + pi_bo
                if abs(total - 1.5) < 0.15:
                    # aromatic: single green line, medium width
                    seg_list.append([p0, p1])
                    col_list.append((0.0, 0.6, 0.0, 0.9))
                elif total > 2.5:
                    # triple: three parallel lines
                    seg_list.append([p0, p1])
                    col_list.append((0.2, 0.2, 0.2, 0.9))
                    seg_list.append([p0 + perp * offset, p1 + perp * offset])
                    col_list.append((0.2, 0.2, 0.2, 0.9))
                    seg_list.append([p0 - perp * offset, p1 - perp * offset])
                    col_list.append((0.2, 0.2, 0.2, 0.9))
                elif total > 1.7:
                    # double: two parallel lines
                    seg_list.append([p0 + perp * offset, p1 + perp * offset])
                    col_list.append((0.2, 0.2, 0.2, 0.9))
                    seg_list.append([p0 - perp * offset, p1 - perp * offset])
                    col_list.append((0.2, 0.2, 0.2, 0.9))
                else:
                    # single: one thin line
                    seg_list.append([p0, p1])
                    col_list.append((0.3, 0.3, 0.3, 0.8))
                mid_list.append(mid)
                label_list.append(f"{pi_bo:.2f}")
            if seg_list:
                segs = np.array(seg_list, dtype=np.float32).reshape(-1, 3)
                colors = np.array(col_list, dtype=np.float32)
                # Repeat colors for each vertex in each segment pair
                vert_colors = np.repeat(colors, 2, axis=0)
                self._line_set("bond_orders", self.bond_order_lines, segs, color=vert_colors, width=2.0)
                self.bond_order_lines.visible = True
            else:
                self.bond_order_lines.set_data(np.zeros((0, 3), dtype=np.float32))
                self.bond_order_lines.visible = False
            # Bond order labels
            if self._show_bond_order_labels and mid_list:
                self.bond_order_text.text = label_list
                self.bond_order_text.pos = np.array(mid_list, dtype=np.float32)
                self.bond_order_text.visible = True
            else:
                self.bond_order_text.visible = False
        else:
            self.bond_order_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.bond_order_lines.visible = False
            self.bond_order_text.visible = False

        # Forces: per-atom line from pos to pos+f*scale
        if self._forces is not None:
            f = self._forces
            if f.shape != self._pos.shape:
                raise ValueError(f"AtomScene._redraw: forces.shape={f.shape} expected {self._pos.shape}")
            segs = np.empty((idx.size * 2, 3), dtype=np.float32)
            segs[0::2] = self._pos[idx]
            segs[1::2] = self._pos[idx] + f[idx] * self._force_scale
            self._line_set("forces", self.force_lines, segs)
        else:
            self.force_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        # Bounding boxes (clusters)
        if self._show_bboxes and (self._bboxes_min is not None) and (self._bboxes_max is not None):
            bmin = self._bboxes_min
            bmax = self._bboxes_max
            ng = int(bmin.shape[0])
            segs = []
            connect = []
            for ig in range(ng):
                mn = bmin[ig, :3]; mx = bmax[ig, :3]
                v = np.array([
                    [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
                    [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
                    [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
                    [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
                ], dtype=np.float32)
                e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                for (a,b) in e:
                    segs.append(v[a]); segs.append(v[b])
                    connect.append(True); connect.append(False)
            if len(segs) > 0:
                segs = np.asarray(segs, dtype=np.float32)
                connect = np.asarray(connect, dtype=bool)
                self.bbox_lines.set_data(segs, connect=connect)
            else:
                self.bbox_lines.set_data(np.zeros((0, 3), dtype=np.float32))
        else:
            self.bbox_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        # Labels - only manage internally if no backend (GUI manages labels externally)
        if self.backend is None:
            if self._label_mode == 'none':
                self.text_labels.text = ['']
                self.text_labels.pos = np.zeros((1, 3), dtype=np.float32)
                self.text_labels.visible = False
            else:
                if self._labels_text is None:
                    txt = []
                    for ii in idx:
                        if self._label_mode == 'global':
                            txt.append(str(int(ii)))
                        elif self._label_mode == 'local':
                            txt.append(str(int(ii % int(self._group_size))))
                        elif self._label_mode == 'radius':
                            if self._radius is None:
                                txt.append('nan')
                            else:
                                txt.append(f"{float(self._radius[ii]):.2f}")
                        else:
                            txt.append(f"{int(ii//int(self._group_size))},{int(ii%int(self._group_size))}")
                    self._labels_text = txt
                self.text_labels.text = self._labels_text
                self.text_labels.pos = (self._pos[idx] + np.array([0.02, 0.02, 0.02], dtype=np.float32)[None, :]).astype(np.float32)
                self.text_labels.visible = True

        # Fragment highlight bonds (dedicated visual — persists across redraws)
        if self._frag_bond_segs is not None and self._frag_bond_segs.size > 0:
            self._line_set("frag_bonds", self.frag_lines, self._frag_bond_segs, color=self._frag_bond_colors, width=5.0)
            self.frag_lines.visible = True
        else:
            self.frag_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.frag_lines.visible = False

        # Fragment bounding boxes (dedicated visual)
        if self._frag_bbox_segs is not None and self._frag_bbox_segs.size > 0:
            self._line_set("frag_bboxes", self.frag_bbox_lines, self._frag_bbox_segs, color=self._frag_bbox_colors, width=1.5)
            self.frag_bbox_lines.visible = True
        else:
            self.frag_bbox_lines.set_data(np.zeros((0, 3), dtype=np.float32))
            self.frag_bbox_lines.visible = False

        self.canvas.update()

    def _mouse_to_world_xy(self, mouse_pos, z=0.0):
        # Works best with top-down orthographic camera.
        # mouse_pos is in canvas pixels (x,y).
        tr = self.view.scene.transform
        p = tr.imap((mouse_pos[0], mouse_pos[1], float(z)))
        return np.array([p[0], p[1]], dtype=np.float32)

    def _pick_idx_from_mouse(self, pos):
        if self._pos.shape[0] == 0:
            return -1
        self._sync_fixed_mask()
        if self._pick_mode == '2d':
            r0, rd = self._ray_from_mouse(pos)
            xy = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
            if xy is None:
                return -1
            xy = xy[:2]
            valid = np.isfinite(self._pos).all(axis=1)
            if self._fixed is not None:
                valid &= (~self._fixed)
            if not np.any(valid):
                return -1
            d2 = np.sum((self._pos[:, :2] - xy[None, :]) ** 2, axis=1)
            d2 = np.where(valid, d2, 1e30)
            return int(np.argmin(d2))
        r0, rd = self._ray_from_mouse(pos)
        i, _ = self._pick_idx_from_ray(r0, rd)
        return i

    def _pick_idx_with_dist(self, pos):
        """Like _pick_idx_from_mouse but also returns squared distance. Returns (idx, d2)."""
        if self._pos.shape[0] == 0:
            return -1, 1e30
        self._sync_fixed_mask()
        if self._pick_mode == '2d':
            r0, rd = self._ray_from_mouse(pos)
            xy = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
            if xy is None:
                return -1, 1e30
            xy = xy[:2]
            valid = np.isfinite(self._pos).all(axis=1)
            if self._fixed is not None:
                valid &= (~self._fixed)
            if not np.any(valid):
                return -1, 1e30
            d2 = np.sum((self._pos[:, :2] - xy[None, :]) ** 2, axis=1)
            d2 = np.where(valid, d2, 1e30)
            i = int(np.argmin(d2))
            return i, float(d2[i])
        r0, rd = self._ray_from_mouse(pos)
        i, d = self._pick_idx_from_ray(r0, rd)
        return i, d * d

    def _idx_to_id(self, idx):
        """Convert array index to Atom._id. Returns -1 if invalid."""
        if idx < 0 or idx >= len(self._atom_ids):
            return -1
        return int(self._atom_ids[idx])

    def _id_to_idx_safe(self, atom_id):
        """Convert Atom._id to array index. Returns -1 if not found."""
        return self._id_to_idx.get(int(atom_id), -1)

    def _pick_id_from_mouse(self, pos, max_dist=0.5):
        """Pick atom by mouse position with distance threshold.
        Returns (Atom._id, distance) or (-1, inf) if no atom within threshold."""
        idx, d2 = self._pick_idx_with_dist(pos)
        if idx < 0 or d2 > max_dist * max_dist:
            return -1, float('inf')
        return self._idx_to_id(idx), float(np.sqrt(d2))

    def _bond_near_mouse(self, pos, max_dist=None):
        """True if a bond midpoint is within max_dist of the mouse ray (or XY hit in 2d)."""
        if self._bonds is None or len(self._bonds) == 0 or self._pos.shape[0] == 0:
            return False
        max_dist = self.pick_radius if max_dist is None else float(max_dist)
        r0, rd = self._ray_from_mouse(pos)
        bonds = np.asarray(self._bonds)
        for a, b in bonds:
            ia, ib = int(a), int(b)
            if ia < 0 or ib < 0 or ia >= len(self._pos) or ib >= len(self._pos):
                continue
            c = 0.5 * (self._pos[ia] + self._pos[ib])
            if self._pick_mode == '2d':
                xy = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0.0, 0.0, 1.0]))
                if xy is None:
                    continue
                d = float(np.linalg.norm(c[:2] - xy[:2]))
            else:
                dp = c - r0
                q = dp - rd * float(np.dot(dp, rd))
                d = float(np.linalg.norm(q))
            if d <= max_dist:
                return True
        return False

    def _on_mouse_press(self, ev):
        # Store Ctrl state for RMB bridge removal in GUI
        self._last_ctrl = 'Control' in ev.modifiers if isinstance(ev.modifiers, (tuple, list)) else False
        if ev.button in (2, 3):
            if self._selection_mode:
                # Sticky xform: RMB also commits/exits
                if self._xform_mode is not None:
                    self.exit_xform(commit=True)
                    ev.handled = True
                    return
                # RMB on atom → remove from selection; empty → box-remove drag
                atom_id, dist = self._pick_id_from_mouse(ev.pos, max_dist=self.pick_radius)
                if atom_id >= 0:
                    if atom_id in self._selected_ids:
                        self._selected_ids.discard(atom_id)
                        if not self._selected_ids:
                            self.exit_xform(commit=False)
                        self._highlight_selected()
                        self._update_selection_bbox()
                        self.sig_selection_changed.emit(self._selected_ids.copy())
                        debug_print(2, f"[SEL] RMB remove id={atom_id} n={len(self._selected_ids)}")
                    ev.handled = True
                    return
                if self.selection_rect is None:
                    self.selection_rect = visuals.Line(parent=self.view.scene, color=(0.9, 0.3, 0.3, 1.0), width=2.0)
                    self.selection_rect.visible = False
                p = self._mouse_xy_on_z0(ev.pos)
                self._selection_start = np.asarray(p, dtype=np.float32)
                self._selection_end = self._selection_start.copy()
                self._box_sel_mode = 'remove'
                self.selection_rect.visible = True
                self._update_selection_rect()
                ev.handled = True
                return
            atom_id, dist = self._pick_id_from_mouse(ev.pos, max_dist=self.pick_radius)
            if atom_id >= 0:
                if int(self._cam_debug) > 0:
                    idx = self._id_to_idx_safe(atom_id)
                    print(f"[RMB] hit id={atom_id} idx={idx} d={dist:.3f} pos=({self._pos[idx,0]:.2f},{self._pos[idx,1]:.2f})")
                self.sig_rmb_remove.emit(atom_id)
                ev.handled = True
                return
            else:
                if int(self._cam_debug) > 0:
                    r0, rd = self._ray_from_mouse(ev.pos)
                    xy = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
                    if xy is not None and self._pos.shape[0] > 0:
                        xy = xy[:2]
                        d2 = np.sum((self._pos[:, :2] - xy[None, :]) ** 2, axis=1)
                        best = int(np.argmin(d2))
                        print(f"[RMB] miss mouse=({xy[0]:.2f},{xy[1]:.2f}) closest_d={np.sqrt(d2[best]):.3f} radius={self.pick_radius}")
                    else:
                        print(f"[RMB] miss (no atoms or ray miss)")
            # 3D: RMB-drag rotate when no atom/bond under cursor (lowest priority)
            if not self._lock_top_view:
                if self._bond_near_mouse(ev.pos):
                    return  # let GUI delete bond / other RMB actions
                self._rmb_down = True
                self._rmb_last = np.array(ev.pos, dtype=np.float32)
                self._cam_print('rmb_down')
                ev.handled = True
                return
            # 2D: leave miss for GUI (hex/bond/empty)
            return
        if ev.button != 1:
            return

        # ── Select mode LMB: handles / sticky xform / add-to-selection ──
        if self._selection_mode:
            handle = self._pick_sel_handle(ev.pos)
            if self._xform_mode is not None:
                # Click opposite handle → switch mode; same/elsewhere → commit exit
                if handle is not None and handle != self._xform_mode:
                    self.enter_xform(handle, mouse_pos=ev.pos)
                else:
                    self.exit_xform(commit=True)
                ev.handled = True
                return
            if handle is not None:
                self.enter_xform(handle, mouse_pos=ev.pos)
                ev.handled = True
                return
            atom_id, dist = self._pick_id_from_mouse(ev.pos, max_dist=self.pick_radius)
            if atom_id >= 0:
                self._selected_ids.add(atom_id)
                self._highlight_selected()
                self._update_selection_bbox()
                self.sig_selection_changed.emit(self._selected_ids.copy())
                debug_print(2, f"[SEL] LMB add id={atom_id} n={len(self._selected_ids)}")
                ev.handled = True
                return
            # Empty LMB: optional box-add
            if self.selection_rect is None:
                self.selection_rect = visuals.Line(parent=self.view.scene, color=(0.2, 0.6, 1.0, 1.0), width=2.0)
                self.selection_rect.visible = False
            p = self._mouse_xy_on_z0(ev.pos)
            self._selection_start = np.asarray(p, dtype=np.float32)
            self._selection_end = self._selection_start.copy()
            self._box_sel_mode = 'add'
            self.selection_rect.visible = True
            self._update_selection_rect()
            ev.handled = True
            return

        # Ctrl+LMB in link mode: start bond creation drag (rubber-band line)
        debug_print(2, f"[SCENE_PRESS] link_mode={self._link_mode} ctrl={self._last_ctrl} button={ev.button}")
        if self._link_mode and self._last_ctrl:
            i, d2 = self._pick_idx_with_dist(ev.pos)
            debug_print(2, f"[SCENE_PRESS] ctrl+link check: idx={i} d2={d2:.3f} radius²={self.pick_radius**2:.3f}")
            if i >= 0 and d2 <= self.pick_radius ** 2:
                atom_id = self._idx_to_id(i)
                self._link_active = True
                self._link_from_id = atom_id
                self._pick_idx = -1  # Prevent GUI on_mouse_press from reading stale pick
                self._pick_id = -1
                # Show rubber-band line from atom to current mouse pos
                r0, rd = self._ray_from_mouse(ev.pos)
                p = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
                if p is not None:
                    start = self._pos[i][:2]
                    end = p[:2]
                    self.link_line.set_data(pos=np.array([start, end], dtype=np.float32), color=(0.2, 0.8, 0.2, 0.8))
                    self.link_line.visible = True
                ev.handled = True
                return
            # Ctrl pressed but no atom under cursor → fall through to normal behavior

        # External lock: suppress all drag (e.g. Ring mode)
        if self.lock_drag:
            # Do not stash _pick_idx — release would treat it as a drag (no _press_pos).
            self._pick_idx = -1
            self._pick_id = -1
            self._pick_active = False
            return

        i, d2 = self._pick_idx_with_dist(ev.pos)
        if i < 0 or d2 > self.pick_radius ** 2:
            # No atom within pick radius — clear stale pick state and let GUI handler run
            debug_print(2, f"[SCENE_PRESS] no atom within radius (best_d2={d2:.3f} radius²={self.pick_radius**2:.3f}) — falling through to GUI")
            self._pick_idx = -1
            self._pick_id = -1
            return
        atom_id = self._idx_to_id(i)
        self._press_pos = np.array(ev.pos, dtype=np.float32)  # for click-vs-drag detection
        debug_print(2, f"[SCENE_PRESS] picked atom_id={atom_id} idx={i} d={np.sqrt(d2):.3f} — starting drag")

        # If atoms are selected and we click on one of them, drag all selected
        if self._selected_ids and atom_id in self._selected_ids:
            self._pick_active = True
            self._pick_idx = i  # Track which atom was clicked for delta calculation
            self._pick_id = atom_id
            self._pick_z = 0.0 if self._clamp_xy else float(self._pos[i, 2])
            # Store initial positions of all selected atoms (map IDs to current indices)
            self._selected_initial_pos = {}
            for aid in self._selected_ids:
                idx = self._id_to_idx_safe(aid)
                if idx >= 0:
                    self._selected_initial_pos[idx] = self._pos[idx].copy()
            self.sig_atom_picked.emit(atom_id)
            self.sig_drag_state.emit(1, atom_id, self._pos[i].copy())
            if int(self._cam_debug) > 0:
                print(f"[DRAG] down selected={len(self._selected_ids)} atoms, anchor id={atom_id}")
            if self._pick_mode == '3d':
                r0, rd = self._ray_from_mouse(ev.pos)
                self._drag_plane_p0 = self._pos[i].copy()
                self._drag_plane_n = rd.copy()
            ev.handled = True
            return

        if self.is_fixed(i):
            # still allow pick, but not drag
            self._pick_active = False
            self._pick_idx = i
            self._pick_id = atom_id
            self.sig_atom_picked.emit(atom_id)
            ev.handled = True
            return

        self._pick_active = True
        self._pick_idx = i
        self._pick_id = atom_id
        self._pick_z = 0.0 if self._clamp_xy else float(self._pos[i, 2])
        self.sig_atom_picked.emit(atom_id)
        self.sig_drag_state.emit(1, atom_id, self._pos[i].copy())
        if int(self._cam_debug) > 0:
            print(f"[DRAG] down id={atom_id} pos=({self._pos[i,0]:.3f},{self._pos[i,1]:.3f},{self._pos[i,2]:.3f}) mode={self._pick_mode}")

        if self._pick_mode == '3d':
            r0, rd = self._ray_from_mouse(ev.pos)
            self._drag_plane_p0 = self._pos[i].copy()
            self._drag_plane_n = rd.copy()
        ev.handled = True

    def _on_mouse_release(self, ev):
        if ev.button in (2, 3):
            if self._selection_mode and self.selection_rect is not None and self.selection_rect.visible:
                # Finalize selection
                self.selection_rect.visible = False
                self._finalize_selection(mode=getattr(self, '_box_sel_mode', 'remove'))
                ev.handled = True
                return
            self._rmb_down = False
            self._rmb_last = None
            self._cam_print('rmb_up')
            ev.handled = True
            return
        # LMB box-add release in Select mode
        if ev.button == 1 and self._selection_mode and self.selection_rect is not None and self.selection_rect.visible:
            self.selection_rect.visible = False
            self._finalize_selection(mode=getattr(self, '_box_sel_mode', 'add'))
            ev.handled = True
            return
        # Handle link mode release (Ctrl+drag bond creation)
        if self._link_active:
            self.link_line.visible = False
            target_id, _ = self._pick_id_from_mouse(ev.pos, max_dist=self.pick_radius)
            from_id = self._link_from_id
            self._link_active = False
            self._link_from_id = -1
            self.hover_atom_marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            if target_id >= 0 and target_id != from_id:
                debug_print(2, f"[LINK_RELEASE] bond from={from_id} to={target_id}")
                self.sig_link_bond.emit(from_id, target_id)
            elif target_id == from_id:
                # Released on same atom = click → change type
                self.sig_atom_clicked.emit(from_id)
            else:
                # Released on empty space → create new atom at release position + bond (2D only)
                if self._pick_mode != '2d':
                    ev.handled = True
                    return
                r0, rd = self._ray_from_mouse(ev.pos)
                p = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
                if p is not None:
                    debug_print(2, f"[LINK_RELEASE] new atom from={from_id} at=({p[0]:.2f},{p[1]:.2f})")
                    self.sig_link_to_pos.emit(from_id, float(p[0]), float(p[1]))
            ev.handled = True
            return
        self._pick_active = False
        if self._pick_idx >= 0 and hasattr(self, '_press_pos') and self._press_pos is not None:
            # Click-vs-drag: if mouse barely moved, treat as click (emit sig_atom_clicked)
            moved = np.linalg.norm(np.array(ev.pos, dtype=np.float32) - self._press_pos)
            if moved < 3:
                debug_print(2, f"[SCENE_RELEASE] click (moved={moved:.1f}px) → sig_atom_clicked atom_id={self._pick_id}")
                self.sig_atom_clicked.emit(int(self._pick_id))
            else:
                debug_print(2, f"[SCENE_RELEASE] drag (moved={moved:.1f}px) → sig_drag_state atom_id={self._pick_id}")
                self.sig_drag_state.emit(0, int(self._pick_id), self._pos[int(self._pick_idx)].copy())
            if int(self._cam_debug) > 0:
                i = int(self._pick_idx)
                print(f"[DRAG] up id={self._pick_id} pos=({self._pos[i,0]:.3f},{self._pos[i,1]:.3f},{self._pos[i,2]:.3f})")
        self._pick_idx = -1
        self._pick_id = -1
        self._press_pos = None
        self._drag_plane_p0 = None
        self._drag_plane_n = None
        # Clean up selected initial positions
        if hasattr(self, '_selected_initial_pos'):
            del self._selected_initial_pos

    def _on_mouse_move(self, ev):
        # Sticky δ/φ transform follows mouse with no button hold
        if self._xform_mode is not None:
            self._apply_xform_mouse(ev.pos)
            ev.handled = True
            return
        if self._link_active:
            # Update rubber-band line endpoint to mouse world position
            r0, rd = self._ray_from_mouse(ev.pos)
            p = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
            if p is not None:
                from_idx = self._id_to_idx_safe(self._link_from_id)
                if from_idx >= 0:
                    start = self._pos[from_idx][:2]
                    end = p[:2]
                    # Check if hovering over a different atom → highlight green
                    target_id, _ = self._pick_id_from_mouse(ev.pos, max_dist=self.pick_radius)
                    if target_id >= 0 and target_id != self._link_from_id:
                        tgt_idx = self._id_to_idx_safe(target_id)
                        if tgt_idx >= 0:
                            self.hover_atom_marker.set_data(
                                pos=self._pos[tgt_idx:tgt_idx+1].astype(np.float32),
                                symbol='ring', size=18, edge_width=2,
                                face_color=(0, 0, 0, 0), edge_color=(0.2, 0.8, 0.2, 1.0))
                            self.link_line.set_data(pos=np.array([start, end], dtype=np.float32), color=(0.2, 0.8, 0.2, 0.9))
                        else:
                            self.hover_atom_marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
                            self.link_line.set_data(pos=np.array([start, end], dtype=np.float32), color=(0.5, 0.5, 0.5, 0.6))
                    else:
                        self.hover_atom_marker.set_data(pos=np.zeros((0, 3), dtype=np.float32))
                        self.link_line.set_data(pos=np.array([start, end], dtype=np.float32), color=(0.5, 0.5, 0.5, 0.6))
            ev.handled = True
            return
        if self._selection_mode and self.selection_rect is not None and self.selection_rect.visible:
            # Update selection rectangle - use same method as picking for consistency
            r0, rd = self._ray_from_mouse(ev.pos)
            p = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
            if p is not None:
                self._selection_end = np.array([p[0], p[1]], dtype=np.float32)
            else:
                self._selection_end = self._mouse_to_world_xy(ev.pos, z=0.0)
            self._update_selection_rect()
            ev.handled = True
            return
        if self._rmb_down:
            if self._rmb_last is not None:
                cur = np.array(ev.pos, dtype=np.float32)
                d = cur - self._rmb_last
                self._rmb_last = cur
                self._cam_rotate(d[0], d[1])
            ev.handled = True
            return
        if not self._pick_active:
            return
        i = self._pick_idx

        # Use authoritative geometry directly if backend is available
        if self.backend is not None:
            p = self.backend.sys.apos
        else:
            p = self._pos.copy()

        if self._pick_mode == '2d':
            # Use ray casting for consistent coordinate handling with axis widgets
            r0, rd = self._ray_from_mouse(ev.pos)
            new_xy = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
            if new_xy is not None:
                # If dragging selected atoms, move all of them
                if self._selected_ids and hasattr(self, '_selected_initial_pos'):
                    if i >= 0:
                        # Clicked on an atom - use its position as reference
                        delta = new_xy[:2] - self._selected_initial_pos[i][:2]
                    else:
                        # Selection mode - calculate delta from mouse movement
                        r0_start, rd_start = self._ray_from_mouse(self._drag_start_mouse)
                        start_xy = self._intersect_ray_plane(r0_start, rd_start, np.zeros(3), np.array([0,0,1]))
                        if start_xy is not None:
                            delta = new_xy[:2] - start_xy[:2]
                        else:
                            delta = np.array([0.0, 0.0])
                    for idx in self._selected_initial_pos:
                        p[idx, 0] = self._selected_initial_pos[idx][0] + delta[0]
                        p[idx, 1] = self._selected_initial_pos[idx][1] + delta[1]
                        p[idx, 2] = self._selected_initial_pos[idx][2]
                else:
                    p[i, 0] = new_xy[0]
                    p[i, 1] = new_xy[1]
                    p[i, 2] = self._pick_z
        else:
            if (self._drag_plane_p0 is None) or (self._drag_plane_n is None):
                return
            r0, rd = self._ray_from_mouse(ev.pos)
            x = self._intersect_ray_plane(r0, rd, self._drag_plane_p0, self._drag_plane_n)
            if x is None:
                return
            if self._clamp_xy:
                x[2] = 0.0
            # If dragging selected atoms, move all of them
            if self._selected_ids and hasattr(self, '_selected_initial_pos'):
                delta = x - self._selected_initial_pos[i]
                for idx in self._selected_initial_pos:
                    p[idx] = self._selected_initial_pos[idx] + delta
            else:
                p[i, :] = x

        # If using backend proxy, update _pos for rendering (but backend is authoritative)
        # Always copy — a view into sys.apos desyncs _colors/_fixed after topology changes.
        if self.backend is not None:
            self._pos = self.backend.sys.apos.astype(np.float32).copy()
        else:
            self._pos = p
        # Emit signal for parent to track drag position
        self.sig_atom_moved.emit(int(self._pick_id), self._pos[i].copy())
        self._redraw()
        ev.handled = True

    def _on_mouse_wheel(self, ev):
        # Manual zoom (do not rely on camera.interactive)
        delta = None
        raw = {}
        if hasattr(ev, 'delta') and (ev.delta is not None):
            raw['delta'] = ev.delta
            d = ev.delta
            try:
                delta = float(d[1])
            except Exception:
                try:
                    delta = float(d)
                except Exception:
                    delta = None
        elif hasattr(ev, 'delta_y'):
            raw['delta_y'] = ev.delta_y
            delta = float(ev.delta_y)
        elif hasattr(ev, 'dy'):
            raw['dy'] = ev.dy
            delta = float(ev.dy)
        elif hasattr(ev, 'step'):
            raw['step'] = ev.step
            delta = float(ev.step)

        # fallback: if tuple and y is 0, try x
        if (delta is not None) and (abs(delta) < 1e-12) and isinstance(raw.get('delta', None), (tuple, list)):
            try:
                delta = float(raw['delta'][0])
            except Exception:
                pass

        # normalize common wheel conventions (some give +-120 per notch)
        if delta is None:
            if int(self._cam_debug) > 0:
                print(f"[WHEEL] no-delta fields={list(raw.keys())}")
            ev.handled = True
            return

        if abs(delta) > 50.0:
            delta /= 120.0
        if int(self._cam_debug) > 0:
            print(f"[WHEEL] delta={float(delta):.6g} raw={raw}")
        if abs(delta) < 1e-12:
            ev.handled = True
            return
        self._cam_zoom(delta)
        ev.handled = True

    def _on_key_press(self, ev):
        """Camera pan (2D / Shift) or rotate (3D); digit presets for ortho views."""
        if int(self._cam_debug) > 0:
            print(f"[KEY] key={ev.key} text={ev.text}")
        mods = ev.modifiers if isinstance(ev.modifiers, (tuple, list)) else ()
        shift = 'Shift' in mods

        # View presets: 5=Top (default), 0=Bottom, 8=Back, 2=Front, 4=Left, 6=Right
        preset_map = {
            '5': 'top', 'KP_5': 'top',
            '0': 'bottom', 'KP_0': 'bottom',
            '8': 'back', 'KP_8': 'back',
            '2': 'front', 'KP_2': 'front',
            '4': 'left', 'KP_4': 'left',
            '6': 'right', 'KP_6': 'right',
        }
        key = ev.key
        if key in preset_map:
            # In locked 2D top view only Top preset is allowed
            if self._lock_top_view and preset_map[key] != 'top':
                ev.handled = True
                return
            self.set_camera_preset(preset_map[key])
            ev.handled = True
            return

        arrow_keys = ('ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Up', 'Down', 'Left', 'Right')
        if key not in arrow_keys:
            return

        # Map to direction
        if key in ('ArrowUp', 'Up'):
            pan_dx, pan_dy, rot_az, rot_el = 0, 1, 0, self._cam_rot_key_speed
        elif key in ('ArrowDown', 'Down'):
            pan_dx, pan_dy, rot_az, rot_el = 0, -1, 0, -self._cam_rot_key_speed
        elif key in ('ArrowLeft', 'Left'):
            pan_dx, pan_dy, rot_az, rot_el = -1, 0, -self._cam_rot_key_speed, 0
        else:
            pan_dx, pan_dy, rot_az, rot_el = 1, 0, self._cam_rot_key_speed, 0

        if self._lock_top_view or shift:
            self._cam_pan(pan_dx, pan_dy)
        else:
            self._cam_rotate_keys(rot_az, rot_el)
        ev.handled = True

    def _mouse_xy_on_z0(self, mouse_pos):
        r0, rd = self._ray_from_mouse(mouse_pos)
        p = self._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0.0, 0.0, 1.0]))
        if p is not None:
            return np.array([p[0], p[1]], dtype=np.float64)
        return np.asarray(self._mouse_to_world_xy(mouse_pos, z=0.0), dtype=np.float64)

    def _selection_indices(self):
        idxs = []
        for aid in self._selected_ids:
            i = self._id_to_idx_safe(aid)
            if i >= 0:
                idxs.append(i)
        return idxs

    def _selection_com(self):
        idxs = self._selection_indices()
        if not idxs:
            return np.zeros(3, dtype=np.float64)
        return self._pos[idxs].mean(axis=0).astype(np.float64)

    def _hide_selection_bbox(self):
        self.sel_bbox_lines.visible = False
        self.sel_handle_markers.visible = False
        self.sel_handle_text.visible = False
        self._sel_handle_move = None
        self._sel_handle_rot = None
        self.sel_bbox_lines.set_data(np.zeros((0, 3), dtype=np.float32))
        self.sel_handle_markers.set_data(np.zeros((0, 3), dtype=np.float32))
        self.sel_handle_text.text = ''

    def _update_selection_bbox(self):
        """Draw AABB + δ (BL) / φ (TR) handles around selected atoms."""
        idxs = self._selection_indices()
        if not idxs:
            self._hide_selection_bbox()
            return
        pts = self._pos[idxs]
        pad = float(self._sel_bbox_pad)
        xmin = float(pts[:, 0].min()) - pad
        xmax = float(pts[:, 0].max()) + pad
        ymin = float(pts[:, 1].min()) - pad
        ymax = float(pts[:, 1].max()) + pad
        if xmax - xmin < 0.8:
            c = 0.5 * (xmin + xmax); xmin, xmax = c - 0.4, c + 0.4
        if ymax - ymin < 0.8:
            c = 0.5 * (ymin + ymax); ymin, ymax = c - 0.4, c + 0.4
        z = float(pts[:, 2].mean()) + 0.05
        box = np.array([[xmin, ymin, z], [xmax, ymin, z], [xmax, ymax, z], [xmin, ymax, z], [xmin, ymin, z]], dtype=np.float32)
        self.sel_bbox_lines.set_data(pos=box, color=(0.15, 0.55, 1.0, 0.95), width=2.0)
        self.sel_bbox_lines.visible = True
        self._sel_handle_move = np.array([xmin, ymin], dtype=np.float64)
        self._sel_handle_rot = np.array([xmax, ymax], dtype=np.float64)
        hpos = np.array([[xmin, ymin, z], [xmax, ymax, z]], dtype=np.float32)
        # Active handle: larger + warm; idle: cool cyan / magenta
        move_on = self._xform_mode == 'move'
        rot_on = self._xform_mode == 'rotate'
        cols = np.array([
            [1.0, 0.85, 0.1, 1.0] if move_on else [0.15, 0.75, 0.95, 1.0],
            [1.0, 0.85, 0.1, 1.0] if rot_on else [0.85, 0.25, 0.9, 1.0],
        ], dtype=np.float32)
        sizes = np.array([22.0 if move_on else 14.0, 22.0 if rot_on else 14.0], dtype=np.float32)
        try:
            self.sel_handle_markers.set_data(hpos, face_color=cols, size=sizes, edge_width=1.5, edge_color='black', symbol='square')
        except TypeError:
            self.sel_handle_markers.set_data(hpos, face_color=cols, size=sizes, edge_width=1.5, edge_color='black', marker='square')
        self.sel_handle_markers.visible = True
        self.sel_handle_text.text = ['δ', 'φ']
        self.sel_handle_text.pos = hpos
        self.sel_handle_text.color = np.array([(0, 0, 0, 1), (0, 0, 0, 1)], dtype=np.float32)
        self.sel_handle_text.visible = True
        self.canvas.update()

    def _pick_sel_handle(self, mouse_pos):
        """Return 'move'|'rotate'|None for δ/φ corner under cursor."""
        if self._sel_handle_move is None or self._sel_handle_rot is None:
            return None
        xy = self._mouse_xy_on_z0(mouse_pos)
        r2 = float(self._sel_handle_r) ** 2
        d_move = float(np.sum((xy - self._sel_handle_move) ** 2))
        d_rot = float(np.sum((xy - self._sel_handle_rot) ** 2))
        if d_move <= r2 and d_move <= d_rot:
            return 'move'
        if d_rot <= r2:
            return 'rotate'
        return None

    def hide_xform_overlays(self):
        """Hide CH/H-bond/label/hover/frag overlays during sticky move/rotate (stale otherwise)."""
        empty = np.zeros((0, 3), dtype=np.float32)
        for v in (self.ch_bond_lines, self.hbond_lines, self.bond_order_lines, self.bond_colored_lines,
                  self.force_lines, self.hover_bond_line, self.hover_ring_lines, self.link_line,
                  self.ring_preview_line, self.frag_lines, self.frag_bbox_lines):
            v.set_data(empty)
            v.visible = False
        self.bond_order_text.visible = False
        self.text_labels.visible = False
        self.hover_ring_text.visible = False
        self.hover_ring_text.text = ''
        self.hover_atom_marker.set_data(empty)
        self.hover_ring_markers.set_data(empty)

    def enter_xform(self, mode, mouse_pos=None):
        """Sticky transform toggle: mode='move'|'rotate'. Follows mouse until exit_xform."""
        if mode not in ('move', 'rotate') or not self._selected_ids:
            return
        self._xform_mode = mode
        xy = self._mouse_xy_on_z0(mouse_pos) if mouse_pos is not None else self._selection_com()[:2]
        self._xform_last_xy = np.asarray(xy, dtype=np.float64)
        self._xform_com = self._selection_com()
        d = self._xform_last_xy - self._xform_com[:2]
        self._xform_last_angle = float(np.arctan2(d[1], d[0]))
        debug_print(2, f"[XFORM] enter mode={mode} n={len(self._selected_ids)}")
        self.hide_xform_overlays()
        self._update_selection_bbox()
        self.sig_drag_state.emit(1, -1, self._xform_com.astype(np.float32))

    def exit_xform(self, commit=True):
        """End sticky move/rotate; optionally commit positions to graph via drag-end signal."""
        if self._xform_mode is None:
            return
        mode = self._xform_mode
        self._xform_mode = None
        self._xform_last_xy = None
        self._xform_last_angle = None
        debug_print(2, f"[XFORM] exit mode={mode} commit={commit}")
        if commit and self._selected_ids:
            # atom_id=-1 → sync only (no single-atom merge heuristic)
            self.sig_drag_state.emit(0, -1, self._selection_com().astype(np.float32))
        self._update_selection_bbox()

    def _apply_xform_mouse(self, mouse_pos):
        """Follow mouse while sticky move/rotate is active (no button hold)."""
        if self._xform_mode is None or not self._selected_ids:
            return
        xy = self._mouse_xy_on_z0(mouse_pos)
        idxs = self._selection_indices()
        if not idxs:
            return
        p = self.backend.sys.apos if self.backend is not None else self._pos
        if self._xform_mode == 'move':
            if self._xform_last_xy is None:
                self._xform_last_xy = xy
                return
            delta = xy - self._xform_last_xy
            self._xform_last_xy = xy
            for i in idxs:
                p[i, 0] += delta[0]
                p[i, 1] += delta[1]
            if self._xform_com is not None:
                self._xform_com[0] += delta[0]
                self._xform_com[1] += delta[1]
        else:  # rotate about selection COM in XY
            com = self._xform_com if self._xform_com is not None else self._selection_com()
            d = xy - com[:2]
            ang = float(np.arctan2(d[1], d[0]))
            if self._xform_last_angle is None:
                self._xform_last_angle = ang
                return
            dang = ang - self._xform_last_angle
            # unwrap
            if dang > np.pi: dang -= 2 * np.pi
            if dang < -np.pi: dang += 2 * np.pi
            self._xform_last_angle = ang
            c, s = np.cos(dang), np.sin(dang)
            for i in idxs:
                dx = p[i, 0] - com[0]
                dy = p[i, 1] - com[1]
                p[i, 0] = com[0] + c * dx - s * dy
                p[i, 1] = com[1] + s * dx + c * dy
        if self.backend is not None:
            self._pos = self.backend.sys.apos.astype(np.float32).copy()
        else:
            self._pos = np.asarray(p, dtype=np.float32)
        self._redraw()
        self._highlight_selected()
        self._update_selection_bbox()

    def _update_selection_rect(self):
        """Update selection rectangle visualization (Line visual)."""
        if self._selection_start is None or self._selection_end is None:
            return
        x0, y0 = self._selection_start
        x1, y1 = self._selection_end
        # Ensure proper ordering
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        # Create rectangle vertices (5 points to close the loop)
        vertices = np.array([
            [x_min, y_min, 0],
            [x_max, y_min, 0],
            [x_max, y_max, 0],
            [x_min, y_max, 0],
            [x_min, y_min, 0]
        ], dtype=np.float32)
        self.selection_rect.set_data(pos=vertices)
        self.canvas.update()

    def _finalize_selection(self, mode='add'):
        """Finalize box selection. mode='add' unions; mode='remove' subtracts."""
        if self._selection_start is None or self._selection_end is None:
            return
        x0, y0 = self._selection_start
        x1, y1 = self._selection_end
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        # Tiny drag = click, not a box
        if (x_max - x_min) < 1e-3 and (y_max - y_min) < 1e-3:
            self._selection_start = None
            self._selection_end = None
            return
        hit = set()
        for i in range(len(self._pos)):
            x, y = self._pos[i, 0], self._pos[i, 1]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                aid = self._idx_to_id(i)
                if aid >= 0:
                    hit.add(aid)
        if mode == 'remove':
            self._selected_ids -= hit
        else:
            self._selected_ids |= hit
        self._highlight_selected()
        self._update_selection_bbox()
        self.sig_selection_changed.emit(self._selected_ids.copy())
        self._selection_start = None
        self._selection_end = None

    def _highlight_selected(self):
        """Highlight selected atoms by changing their color."""
        if self._colors is None:
            return
        # Restore original colors (only if sizes match)
        if self._selected_colors_backup is not None and self._selected_colors_backup.shape == self._colors.shape:
            self._colors[:] = self._selected_colors_backup
        # Store original colors if first selection or if sizes don't match
        else:
            self._selected_colors_backup = self._colors.copy()
        # Highlight selected atoms (map IDs to current indices)
        for aid in self._selected_ids:
            idx = self._id_to_idx_safe(aid)
            if idx >= 0 and idx < len(self._colors):
                self._colors[idx] = (1.0, 0.5, 0.0, 1.0)  # Orange highlight
        self.atom_markers.set_data(self._pos, edge_color=None, face_color=self._colors, size=self._sizes)
        self.canvas.update()

    def set_selection_mode(self, enabled):
        """Enable or disable selection mode."""
        self._selection_mode = enabled
        # Clear selection when exiting selection mode (commits sticky xform if active)
        if not enabled:
            self.clear_selection()

    def get_selected_ids(self):
        """Return set of selected Atom._ids."""
        return self._selected_ids.copy()

    def get_selected_indices(self):
        """Return set of selected atom array indices (DEPRECATED: use get_selected_ids)."""
        return {self._id_to_idx_safe(aid) for aid in self._selected_ids if self._id_to_idx_safe(aid) >= 0}

    def set_selected_ids(self, ids):
        """Set selected atoms by Atom._id."""
        self._selected_ids = set(ids)
        self._highlight_selected()
        self._update_selection_bbox()
        self.sig_selection_changed.emit(self._selected_ids)

    def set_selected_indices(self, indices):
        """Set selected atom indices (DEPRECATED: use set_selected_ids)."""
        self._selected_ids = {self._idx_to_id(i) for i in indices if self._idx_to_id(i) >= 0}
        self._highlight_selected()
        self._update_selection_bbox()
        self.sig_selection_changed.emit(self._selected_ids)

    def clear_selection(self):
        """Clear selection and restore original colors."""
        self.exit_xform(commit=True)
        self._selected_ids.clear()
        self._hide_selection_bbox()
        if self._selected_colors_backup is not None:
            # Check shape consistency before restoring (atoms may have been added/removed)
            if self._colors is not None and self._selected_colors_backup.shape == self._colors.shape:
                self._colors[:] = self._selected_colors_backup
                self.atom_markers.set_data(self._pos, edge_color=None, face_color=self._colors, size=self._sizes)
            else:
                # Shape mismatch - refresh colors from backend
                self._selected_colors_backup = None
        self.canvas.update()
        self.sig_selection_changed.emit(set())


def normalize_scalar_field(vals, vmin=None, vmax=None, symmetric=False):
    a = np.asarray(vals, dtype=np.float64)
    finite = np.isfinite(a)
    if not np.any(finite):
        raise ValueError("normalize_scalar_field(): field has no finite values")
    if symmetric:
        if vmin is None and vmax is None:
            vmax = float(np.max(np.abs(a[finite])))
            vmin = -vmax
        elif vmin is None:
            vmin = -float(vmax)
        elif vmax is None:
            vmax = -float(vmin)
    else:
        if vmin is None:
            vmin = float(np.min(a[finite]))
        if vmax is None:
            vmax = float(np.max(a[finite]))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"normalize_scalar_field(): invalid limits vmin={vmin} vmax={vmax}")
    if vmax <= vmin:
        if vmax == vmin:
            out = np.zeros_like(a, dtype=np.float32)
            out[~finite] = 0.0
            return out, float(vmin), float(vmax)
        raise ValueError(f"normalize_scalar_field(): vmax={vmax} must be > vmin={vmin}")
    out = np.zeros_like(a, dtype=np.float32)
    out[finite] = np.clip((a[finite] - vmin) / (vmax - vmin), 0.0, 1.0)
    out[~finite] = 0.0
    return out, float(vmin), float(vmax)


def make_grid_mesh_data(xs, ys, zs, colors=None, mask=None):
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    zs = np.asarray(zs, dtype=np.float32)
    if zs.shape != (len(xs), len(ys)):
        raise ValueError(f"make_grid_mesh_data(): zs.shape={zs.shape} expected ({len(xs)},{len(ys)})")
    if mask is None:
        mask = np.isfinite(zs)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != zs.shape:
            raise ValueError(f"make_grid_mesh_data(): mask.shape={mask.shape} expected {zs.shape}")
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    verts = np.stack([X, Y, zs], axis=2).reshape(-1, 3).astype(np.float32)
    if colors is not None:
        cols = np.asarray(colors, dtype=np.float32)
        if cols.shape[:2] != zs.shape:
            raise ValueError(f"make_grid_mesh_data(): colors.shape[:2]={cols.shape[:2]} expected {zs.shape}")
        cols = cols.reshape(-1, cols.shape[-1]).astype(np.float32)
    else:
        cols = None
    faces = []
    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            i00 = ix * len(ys) + iy
            i10 = (ix + 1) * len(ys) + iy
            i01 = ix * len(ys) + (iy + 1)
            i11 = (ix + 1) * len(ys) + (iy + 1)
            if mask[ix, iy] and mask[ix + 1, iy] and mask[ix + 1, iy + 1]:
                faces.append((i00, i10, i11))
            if mask[ix, iy] and mask[ix + 1, iy + 1] and mask[ix, iy + 1]:
                faces.append((i00, i11, i01))
    faces = np.asarray(faces, dtype=np.uint32)
    return verts, faces, cols


def colormap_rgba(vals, cmap='coolwarm', vmin=None, vmax=None, symmetric=False, alpha=1.0):
    t, vmin, vmax = normalize_scalar_field(vals, vmin=vmin, vmax=vmax, symmetric=symmetric)
    cm = Colormap(cmap) if isinstance(cmap, (list, tuple)) else vispy.color.get_colormap(cmap)
    mapped = cm.map(t.ravel())
    rgba = np.asarray(mapped, dtype=np.float32).reshape(t.shape + (4,))
    rgba[..., 3] = float(alpha)
    return rgba, vmin, vmax


def make_surface_mesh(xs, ys, zs, scalar=None, cmap='coolwarm', vmin=None, vmax=None, symmetric=False, alpha=1.0, mask=None):
    if scalar is None:
        rgba = None
        clim = None
    else:
        rgba, vmin, vmax = colormap_rgba(scalar, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, alpha=alpha)
        clim = (vmin, vmax)
    verts, faces, cols = make_grid_mesh_data(xs, ys, zs, colors=rgba, mask=mask)
    return {'vertices': verts, 'faces': faces, 'vertex_colors': cols, 'clim': clim}


def create_surface_visual(parent, mesh_data, shading='smooth'):
    v = np.asarray(mesh_data['vertices'], dtype=np.float32)
    f = np.asarray(mesh_data['faces'], dtype=np.uint32)
    if len(v) == 0 or len(f) == 0:
        raise ValueError(f"create_surface_visual(): empty mesh vertices={v.shape} faces={f.shape}")
    vc = mesh_data.get('vertex_colors', None)
    if vc is not None:
        return visuals.Mesh(vertices=v, faces=f, vertex_colors=np.asarray(vc, dtype=np.float32), shading=shading, parent=parent)
    return visuals.Mesh(vertices=v, faces=f, color=(0.7, 0.7, 0.9, 1.0), shading=shading, parent=parent)


def render_surface_png(out_path, mesh_data, atom_points=None, atom_colors=None, atom_sizes=None, title=None, bgcolor='white', azimuth=-60.0, elevation=35.0, scale=1.2):
    canvas = scene.SceneCanvas(keys=None, bgcolor=bgcolor, show=False, size=(1200, 900))
    view = canvas.central_widget.add_view()
    view.camera = scene.TurntableCamera(fov=0.0, elevation=float(elevation), azimuth=float(azimuth))
    mesh = create_surface_visual(view.scene, mesh_data, shading='smooth')
    mesh.set_gl_state('translucent', depth_test=True)
    if atom_points is not None:
        pts = np.asarray(atom_points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"render_surface_png(): atom_points.shape={pts.shape} expected (n,3)")
        mk = visuals.Markers(parent=view.scene)
        mk.set_data(pts, face_color=atom_colors if atom_colors is not None else (0.1, 0.1, 0.1, 0.5), size=atom_sizes if atom_sizes is not None else 8.0, edge_width=0.0)
    if title:
        visuals.Text(text=str(title), pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32), color='black', font_size=12, parent=view.scene)
    vv = np.asarray(mesh_data['vertices'], dtype=np.float32)
    ctr = vv.mean(axis=0)
    ext = np.max(vv, axis=0) - np.min(vv, axis=0)
    rad = float(np.max(ext[:2]))
    if rad <= 1e-6:
        rad = 1.0
    view.camera.center = tuple(ctr.tolist())
    view.camera.scale_factor = float(rad * scale)
    img = canvas.render(alpha=False)
    from vispy.io import write_png
    write_png(out_path, img)
    return img


def create_heatmap_window(data_2d, extent, title="Heatmap", cmap='bwr', symmetric=True, atom_pos=None, atom_types=None):
    """Create a VisPy window to display 2D heatmap (orbital/density) with optional atom overlay.

    Args:
        data_2d: 2D numpy array (ny, nx) of scalar values where data[i,j] corresponds to (x[j], y[i])
        extent: [xmin, xmax, ymin, ymax] in world coordinates
        title: Window title
        cmap: Colormap name ('bwr', 'hot', 'viridis', etc.)
        symmetric: If True, colormap is symmetric around zero
        atom_pos: Optional (n,3) array of atom positions
        atom_types: Optional array of atom types for coloring

    Returns:
        (canvas, view) tuple for further manipulation if needed
    """
    from PyQt5 import QtWidgets
    data = np.asarray(data_2d, dtype=np.float32)
    ny, nx = data.shape

    # Create colormap
    if symmetric:
        vmax = max(abs(np.min(data)), abs(np.max(data)))
        if vmax < 1e-30:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin, vmax = np.min(data), np.max(data)
        if vmax - vmin < 1e-30:
            vmax = vmin + 1.0

    # Generate RGBA colors
    rgba, _, _ = colormap_rgba(data.ravel(), cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, alpha=1.0)
    rgba = rgba.reshape(ny, nx, 4).astype(np.float32)

    # Create mesh vertices for image (quad grid)
    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    # Create vertices (nx*ny grid points)
    verts = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=1).astype(np.float32)

    # Create faces (two triangles per grid cell)
    n_cells = (nx - 1) * (ny - 1)
    faces = np.zeros((n_cells * 2, 3), dtype=np.uint32)
    for i in range(nx - 1):
        for j in range(ny - 1):
            cell_idx = i * (ny - 1) + j
            v00 = i * ny + j
            v01 = i * ny + (j + 1)
            v10 = (i + 1) * ny + j
            v11 = (i + 1) * ny + (j + 1)
            faces[cell_idx * 2 + 0] = [v00, v01, v10]
            faces[cell_idx * 2 + 1] = [v01, v11, v10]

    # Map colors to vertices (use corner colors)
    vertex_colors = rgba.reshape(nx * ny, 4)

    # Create window
    canvas = scene.SceneCanvas(keys=None, bgcolor='white', show=True, size=(800, 600))
    canvas.title = str(title)
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1.0)

    # Create heatmap mesh
    mesh = visuals.Mesh(vertices=verts, faces=faces, vertex_colors=vertex_colors, shading='flat', parent=view.scene)
    mesh.set_gl_state('translucent', depth_test=False)

    # Add atoms if provided
    if atom_pos is not None:
        pos = np.asarray(atom_pos, dtype=np.float32)
        print(f"DEBUG create_heatmap_window atom_pos: pos.shape={pos.shape}, pos.ndim={pos.ndim}")
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"atom_pos.shape={pos.shape} expected (n,3)")
        # Project to 2D (use x,y, set z=0 for visibility)
        pos_2d = pos.copy()
        pos_2d[:, 2] = 0.0

        # Color atoms by type if types provided, else green
        if atom_types is not None:
            from spammm import elements
            colors = []
            for atype in atom_types:
                c = elements.getColor(atype)
                colors.append((c[0], c[1], c[2], 1.0))
            colors = np.array(colors, dtype=np.float32)
            print(f"DEBUG create_heatmap_window atom_types: atom_types={atom_types}, colors.shape={colors.shape}")
        else:
            colors = (0.0, 0.5, 0.0, 1.0)
            print(f"DEBUG create_heatmap_window atom_types: using default colors")

        print(f"DEBUG create_heatmap_window: calling atom_markers.set_data with pos_2d.shape={pos_2d.shape}, face_color type={type(colors)}")
        atom_markers = visuals.Markers(parent=view.scene)
        atom_markers.set_data(pos_2d, face_color=colors, size=5.0, edge_width=0.5, edge_color='black')

    # Center camera on extent
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    view.camera.center = (cx, cy, 0.0)
    view.camera.rect = (-w/2, -h/2, w, h)

    return canvas, view
