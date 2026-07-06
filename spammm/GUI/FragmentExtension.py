"""
FragmentExtension.py — GUI panel for molecular graph fragmentation analysis.

Provides buttons to run connected components, bridges, articulation points,
local bridges, biconnected components, and rotatable bonds on the current
AtomicGraph. Results are displayed as text, highlighted in the 3D scene
with per-fragment colors, AABB bounding boxes, and a clickable fragment list.

Wired through ExtensionManager as 'fragments' extension.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np

from .ExtensionManager import UIComponents

# Distinct color palette for fragments (RGBA)
FRAGMENT_COLORS = [
    (1.0, 0.2, 0.2, 1.0),  # red
    (0.2, 0.4, 1.0, 1.0),  # blue
    (0.2, 0.8, 0.2, 1.0),  # green
    (1.0, 0.6, 0.0, 1.0),  # orange
    (0.8, 0.2, 0.8, 1.0),  # magenta
    (0.0, 0.7, 0.7, 1.0),  # cyan
    (0.9, 0.9, 0.2, 1.0),  # yellow
    (0.5, 0.3, 0.1, 1.0),  # brown
]


def build_ui(window):
    """Build Fragments analysis panel for the SPAMMM GUI.
    Returns ExtensionManager.UIComponents.
    """
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setSpacing(2)
    layout.setContentsMargins(2, 2, 2, 2)

    # --- Main button: Fragments (split by bridges) ---
    row0 = QtWidgets.QHBoxLayout()
    window.frag_split_btn = QtWidgets.QPushButton("Fragments")
    window.frag_split_btn.setMaximumWidth(90)
    window.frag_split_btn.clicked.connect(lambda: _on_fragments(window))
    row0.addWidget(window.frag_split_btn)

    window.frag_min_size_spin = QtWidgets.QSpinBox()
    window.frag_min_size_spin.setRange(1, 50)
    window.frag_min_size_spin.setValue(2)
    window.frag_min_size_spin.setMaximumWidth(40)
    row0.addWidget(QtWidgets.QLabel("min:"))
    row0.addWidget(window.frag_min_size_spin)
    row0.addStretch()
    layout.addLayout(row0)

    # --- Analysis buttons row 1 ---
    row1 = QtWidgets.QHBoxLayout()
    window.frag_components_btn = QtWidgets.QPushButton("Components")
    window.frag_components_btn.setMaximumWidth(90)
    window.frag_components_btn.clicked.connect(lambda: _on_components(window))
    row1.addWidget(window.frag_components_btn)

    window.frag_bridges_btn = QtWidgets.QPushButton("Bridges")
    window.frag_bridges_btn.setMaximumWidth(70)
    window.frag_bridges_btn.clicked.connect(lambda: _on_bridges(window))
    row1.addWidget(window.frag_bridges_btn)

    window.frag_ap_btn = QtWidgets.QPushButton("Articulation")
    window.frag_ap_btn.setMaximumWidth(90)
    window.frag_ap_btn.clicked.connect(lambda: _on_articulation(window))
    row1.addWidget(window.frag_ap_btn)
    row1.addStretch()
    layout.addLayout(row1)

    # --- Analysis buttons row 2 ---
    row2 = QtWidgets.QHBoxLayout()
    window.frag_local_btn = QtWidgets.QPushButton("Local Bridges")
    window.frag_local_btn.setMaximumWidth(100)
    window.frag_local_btn.clicked.connect(lambda: _on_local_bridges(window))
    row2.addWidget(window.frag_local_btn)

    window.frag_local_dist_spin = QtWidgets.QSpinBox()
    window.frag_local_dist_spin.setRange(2, 20)
    window.frag_local_dist_spin.setValue(3)
    window.frag_local_dist_spin.setMaximumWidth(40)
    row2.addWidget(window.frag_local_dist_spin)
    row2.addStretch()
    layout.addLayout(row2)

    # --- Analysis buttons row 3 ---
    row3 = QtWidgets.QHBoxLayout()
    window.frag_biconn_btn = QtWidgets.QPushButton("BiComponents")
    window.frag_biconn_btn.setMaximumWidth(100)
    window.frag_biconn_btn.clicked.connect(lambda: _on_biconnected(window))
    row3.addWidget(window.frag_biconn_btn)
    row3.addStretch()
    layout.addLayout(row3)

    # --- Clear button ---
    window.frag_clear_btn = QtWidgets.QPushButton("Clear Highlight")
    window.frag_clear_btn.setMaximumWidth(100)
    window.frag_clear_btn.clicked.connect(lambda: _on_clear(window))
    layout.addWidget(window.frag_clear_btn)

    # --- Results display ---
    window.frag_results_label = QtWidgets.QLabel("Results: ---")
    window.frag_results_label.setWordWrap(True)
    window.frag_results_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    window.frag_results_label.setStyleSheet("font-family: monospace; font-size: 10px;")
    layout.addWidget(window.frag_results_label)

    # --- Fragment list ---
    window.frag_list_label = QtWidgets.QLabel("Fragments:")
    layout.addWidget(window.frag_list_label)
    window.frag_list = QtWidgets.QListWidget()
    window.frag_list.setMaximumHeight(120)
    window.frag_list.itemClicked.connect(lambda item: _on_fragment_clicked(window, item))
    layout.addWidget(window.frag_list)

    layout.addStretch()

    # Initialize state
    window._frag_overlay = None       # dict passed to scene.set_frag_highlights
    window._frag_component_colors = None  # {atom_id: RGBA tuple} for atom coloring
    window._frag_data = None          # list of (label, set_of_atom_ids, color) for list clicks

    view_modes = [
        ('Fragment Highlight', lambda: _toggle_highlight(window)),
    ]

    return UIComponents(panel=panel, edit_modes=[], view_modes=view_modes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_graph(window):
    backend = getattr(window, 'backend', None)
    if backend is None: return None
    return backend.graph


def _get_pos_and_idx_map(window):
    """Return (apos, idx_map) from backend.sys."""
    backend = window.backend
    pos = backend.sys.apos.astype(np.float32)
    idx_map = getattr(backend, '_atom_idx_map', {})
    return pos, idx_map


def _aabb_edges(positions, pad=0.3):
    """Compute AABB edges from (n,3) positions. Returns (2*k, 3) segment array."""
    if len(positions) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mn = positions.min(axis=0) - pad
    mx = positions.max(axis=0) + pad
    v = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float32)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    segs = []
    for a, b in edges:
        segs.append(v[a]); segs.append(v[b])
    return np.array(segs, dtype=np.float32)


def _bonds_to_segs(bonds, pos, idx_map, colors=None):
    """Convert list of Bond objects to (2*m, 3) segment array + (2*m, 4) color array."""
    segs = []
    cols = []
    for b in bonds:
        i1, i2 = idx_map.get(b.a._id), idx_map.get(b.b._id)
        if i1 is None or i2 is None: continue
        segs.append(pos[i1]); segs.append(pos[i2])
        if colors is not None:
            c = colors
            cols.append(c); cols.append(c)
    if not segs:
        return np.zeros((0, 3), dtype=np.float32), None
    segs = np.array(segs, dtype=np.float32)
    if cols:
        cols = np.array(cols, dtype=np.float32)
    else:
        cols = None
    return segs, cols


def _set_overlay(window, bond_segs=None, bond_colors=None, bbox_segs=None, bbox_colors=None, component_colors=None):
    """Store overlay data and trigger refresh."""
    window._frag_overlay = {}
    if bond_segs is not None:
        window._frag_overlay['bond_segs'] = bond_segs
    if bond_colors is not None:
        window._frag_overlay['bond_colors'] = bond_colors
    if bbox_segs is not None:
        window._frag_overlay['bbox_segs'] = bbox_segs
    if bbox_colors is not None:
        window._frag_overlay['bbox_colors'] = bbox_colors
    window._frag_component_colors = component_colors
    if hasattr(window, 'refresh_view'):
        window.refresh_view()


def _populate_frag_list(window, fragments):
    """Populate the fragment list widget. fragments = list of (label, set_of_atom_ids, color)."""
    window.frag_list.clear()
    window._frag_data = fragments
    for i, (label, atom_ids, color) in enumerate(fragments):
        item = QtWidgets.QListWidgetItem(f"{i}: {label}")
        # Set color swatch via background
        r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
        item.setBackground(QtGui.QColor(r, g, b, 120))
        window.frag_list.addItem(item)


def _format_bonds(bonds):
    parts = []
    for b in bonds:
        parts.append(f"{b.a._id}-{b.b._id}({b.a.ename}{b.b.ename})")
    return ", ".join(parts)


def _format_atoms(atoms):
    parts = []
    for a in atoms:
        parts.append(f"{a._id}:{a.ename}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _on_components(window):
    g = _get_graph(window)
    if g is None: return
    pos, idx_map = _get_pos_and_idx_map(window)
    comps = g.find_connected_components()
    sizes = sorted([len(c) for c in comps], reverse=True)
    # Per-fragment atom colors + bounding boxes
    component_colors = {}
    bbox_segs = []
    bbox_colors = []
    frag_list = []
    for i, comp in enumerate(comps):
        c = FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)]
        for a in comp:
            component_colors[a._id] = c
        comp_pos = np.array([a.pos for a in comp], dtype=np.float32)
        edges = _aabb_edges(comp_pos)
        if edges.size > 0:
            bbox_segs.append(edges)
            bbox_colors.extend([c] * (len(edges)))
        frag_list.append((f"Comp {i} ({len(comp)} atoms)", {a._id for a in comp}, c))
    bbox_segs = np.concatenate(bbox_segs, axis=0) if bbox_segs else None
    bbox_colors = np.array(bbox_colors, dtype=np.float32) if bbox_colors else None
    _set_overlay(window, bbox_segs=bbox_segs, bbox_colors=bbox_colors, component_colors=component_colors)
    _populate_frag_list(window, frag_list)
    window.frag_results_label.setText(f"Components: {len(comps)} (sizes: {sizes})")


def _on_bridges(window):
    g = _get_graph(window)
    if g is None: return
    pos, idx_map = _get_pos_and_idx_map(window)
    bridges = g.find_bridges()
    bridge_color = (1.0, 0.5, 0.0, 0.9)  # orange
    bond_segs, bond_colors = _bonds_to_segs(bridges, pos, idx_map, colors=bridge_color)
    _set_overlay(window, bond_segs=bond_segs, bond_colors=bond_colors)
    window.frag_list.clear()
    window.frag_results_label.setText(f"Bridges: {len(bridges)}\n{_format_bonds(bridges)}")


def _on_articulation(window):
    g = _get_graph(window)
    if g is None: return
    aps = g.find_articulation_points()
    # Highlight articulation atoms by enlarging them (via component_colors override)
    component_colors = {}
    ap_color = (1.0, 0.2, 0.2, 1.0)  # red
    for a in aps:
        component_colors[a._id] = ap_color
    _set_overlay(window, component_colors=component_colors)
    window.frag_list.clear()
    window.frag_results_label.setText(f"Articulation points: {len(aps)}\n{_format_atoms(aps)}")


def _on_local_bridges(window):
    g = _get_graph(window)
    if g is None: return
    pos, idx_map = _get_pos_and_idx_map(window)
    max_dist = window.frag_local_dist_spin.value()
    lb = g.find_local_bridges(max_dist=max_dist)
    lb_color = (0.8, 0.2, 0.8, 0.9)  # magenta
    bond_segs, bond_colors = _bonds_to_segs(lb, pos, idx_map, colors=lb_color)
    _set_overlay(window, bond_segs=bond_segs, bond_colors=bond_colors)
    window.frag_list.clear()
    window.frag_results_label.setText(f"Local bridges (d≤{max_dist}): {len(lb)}\n{_format_bonds(lb)}")


def _on_biconnected(window):
    g = _get_graph(window)
    if g is None: return
    pos, idx_map = _get_pos_and_idx_map(window)
    blocks = g.find_biconnected_components()
    # Per-block colors + bounding boxes + bond highlights
    component_colors = {}
    bbox_segs = []
    bbox_colors = []
    bond_segs = []
    bond_colors = []
    frag_list = []
    for i, (atoms, bonds) in enumerate(blocks):
        c = FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)]
        is_bridge = len(bonds) == 1
        label = f"{'Bridge' if is_bridge else 'Ring'} {i} ({len(atoms)} atoms, {len(bonds)} bonds)"
        for a in atoms:
            component_colors[a._id] = c
        block_pos = np.array([a.pos for a in atoms], dtype=np.float32)
        edges = _aabb_edges(block_pos)
        if edges.size > 0:
            bbox_segs.append(edges)
            bbox_colors.extend([c] * len(edges))
        # Highlight bonds with block color (thicker for bridges)
        for b in bonds:
            i1, i2 = idx_map.get(b.a._id), idx_map.get(b.b._id)
            if i1 is not None and i2 is not None:
                bond_segs.append(pos[i1]); bond_segs.append(pos[i2])
                bond_colors.append(c); bond_colors.append(c)
        frag_list.append((label, {a._id for a in atoms}, c))
    bbox_segs = np.concatenate(bbox_segs, axis=0) if bbox_segs else None
    bbox_colors = np.array(bbox_colors, dtype=np.float32) if bbox_colors else None
    bond_segs = np.array(bond_segs, dtype=np.float32) if bond_segs else None
    bond_colors = np.array(bond_colors, dtype=np.float32) if bond_colors else None
    _set_overlay(window, bond_segs=bond_segs, bond_colors=bond_colors, bbox_segs=bbox_segs, bbox_colors=bbox_colors, component_colors=component_colors)
    _populate_frag_list(window, frag_list)
    sizes = sorted([len(bonds) for _, bonds in blocks], reverse=True)
    window.frag_results_label.setText(
        f"BiComponents: {len(blocks)} (bond counts: {sizes})\n"
        f"  ring blocks: {len([b for _, b in blocks if len(b) > 1])}, "
        f"bridge blocks: {len([b for _, b in blocks if len(b) == 1])}"
    )


def _on_fragments(window):
    g = _get_graph(window)
    if g is None: return
    pos, idx_map = _get_pos_and_idx_map(window)
    min_size = window.frag_min_size_spin.value()
    fragments, cut_bridges = g.find_fragments(min_size=min_size)
    # Per-fragment colors + bounding boxes + cut bridge highlights
    component_colors = {}
    bbox_segs = []
    bbox_colors = []
    frag_list = []
    for i, frag in enumerate(fragments):
        c = FRAGMENT_COLORS[i % len(FRAGMENT_COLORS)]
        n_heavy = sum(1 for a in frag if a.ename != 'H')
        for a in frag:
            component_colors[a._id] = c
        frag_pos = np.array([a.pos for a in frag], dtype=np.float32)
        edges = _aabb_edges(frag_pos)
        if edges.size > 0:
            bbox_segs.append(edges)
            bbox_colors.extend([c] * len(edges))
        frag_list.append((f"Frag {i} ({n_heavy} heavy, {len(frag)} total)", {a._id for a in frag}, c))
    bbox_segs = np.concatenate(bbox_segs, axis=0) if bbox_segs else None
    bbox_colors = np.array(bbox_colors, dtype=np.float32) if bbox_colors else None
    # Cut bridges as thick orange lines
    bridge_color = (1.0, 0.5, 0.0, 0.9)
    bond_segs, bond_colors = _bonds_to_segs(cut_bridges, pos, idx_map, colors=bridge_color)
    _set_overlay(window, bond_segs=bond_segs, bond_colors=bond_colors, bbox_segs=bbox_segs, bbox_colors=bbox_colors, component_colors=component_colors)
    _populate_frag_list(window, frag_list)
    sizes = sorted([sum(1 for a in f if a.ename != 'H') for f in fragments], reverse=True)
    window.frag_results_label.setText(
        f"Fragments: {len(fragments)} (heavy sizes: {sizes})\n"
        f"Cut bridges: {len(cut_bridges)}\n{_format_bonds(cut_bridges)}"
    )


def _on_clear(window):
    window._frag_overlay = None
    window._frag_component_colors = None
    window._frag_data = None
    window.frag_list.clear()
    window.frag_results_label.setText("Results: ---")
    if hasattr(window, 'refresh_view'):
        window.refresh_view()


def _on_fragment_clicked(window, item):
    """Select atoms belonging to clicked fragment."""
    if window._frag_data is None: return
    row = window.frag_list.row(item)
    if row < 0 or row >= len(window._frag_data): return
    label, atom_ids, color = window._frag_data[row]
    # Select atoms in the scene
    if hasattr(window, 'scene') and hasattr(window.scene, 'set_selected_ids'):
        window.scene.set_selected_ids(atom_ids)
    window.frag_results_label.setText(f"Selected: {label} ({len(atom_ids)} atoms)")


def _toggle_highlight(window):
    pass
