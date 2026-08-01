"""ReactionCoordinateExtension.py — GUI panel for H-bond reaction-coordinate exploration on AtomicGraph.

Separate extension (not Kekule/ASCII): import geometry, run DFTB scan methods, scrub trajectory
on slider with bond Δ coloring, optional Mulliken ESP animation (`rc_esp_view` + `mpl_blit`).

- **SSOT geometry for scripts:** `build_ascii_hbond_system` in `rc_scan_gui_script` — not raw stripped ASCII.
- **Work dir:** `debug/rc_scan/`; cached review npz under `debug/testplot_rc_scan_gui/`.
- **Docs:** `doc/Topics/ReactionCoordinateScan.md`
"""
from PyQt5 import QtWidgets, QtCore
import numpy as np
import os

from .ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH, GridPlacer
from spammm.GUI.EditModeHandlers import EditModeHandler
from spammm.GUI.VispyUtils import compute_bond_colors_by_length, compute_bond_colors_by_delta
from spammm.topology.hbond_utils import find_hbonds_graph, default_mapping
from spammm.topology.scan_dataset import ScanDataset
from spammm.quantum.coordinate_scan import build_frame, build_control_grid, interpolate_all_atoms, run_rigid_dftb_scan, run_pm_neb, dataset_from_frames, DEFAULT_DS
from spammm.GUI.rc_esp_view import open_rc_esp_animation, update_rc_esp_frame


def build_ui(window):
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    g_geo = GridPlacer(cols=6)
    window.rc_import_btn = QtWidgets.QPushButton("Import from Graph")
    window.rc_import_btn.clicked.connect(lambda: import_from_graph(window))
    g_geo.add(window.rc_import_btn, span=3)
    window.rc_sync_btn = QtWidgets.QPushButton("Sync to Graph")
    window.rc_sync_btn.clicked.connect(lambda: sync_to_graph(window))
    g_geo.add(window.rc_sync_btn, span=3)
    layout.addLayout(g_geo.layout())

    window.rc_hbond_label = QtWidgets.QLabel("H-bonds: (import first)")
    layout.addWidget(window.rc_hbond_label)
    g_hb = GridPlacer(cols=6)
    window.rc_refresh_hb_btn = QtWidgets.QPushButton("Refresh H-bonds")
    window.rc_refresh_hb_btn.clicked.connect(lambda: refresh_hbonds(window))
    g_hb.add(window.rc_refresh_hb_btn, span=2)
    window.rc_hbond_spin = QtWidgets.QSpinBox()
    window.rc_hbond_spin.setMinimum(0)
    window.rc_hbond_spin.setMaximum(99)
    g_hb.add_pair("pair", window.rc_hbond_spin, label_span=1, input_span=1)
    window.rc_all_hbonds_chk = QtWidgets.QCheckBox("All H-bonds (symmetric)")
    window.rc_all_hbonds_chk.setChecked(True)
    window.rc_all_hbonds_chk.setToolTip("Move all detected H-bonds together (shared control u)")
    window.rc_all_hbonds_chk.stateChanged.connect(lambda _: refresh_hbonds(window))
    g_hb.add(window.rc_all_hbonds_chk, span=2)
    layout.addLayout(g_hb.layout())

    g_opts = GridPlacer(cols=6)
    window.rc_relax_endpoints_chk = QtWidgets.QCheckBox("Relax endpoints (DFTB)")
    window.rc_relax_endpoints_chk.setChecked(True)
    window.rc_relax_endpoints_chk.setToolTip("DFTB geometry opt at u=0 and u=1 before interpolation")
    g_opts.add(window.rc_relax_endpoints_chk, span=6)
    layout.addLayout(g_opts.layout())

    g_dx = GridPlacer(cols=6)
    window.rc_dx_spin = QtWidgets.QDoubleSpinBox()
    window.rc_dx_spin.setRange(0.01, 1.0)
    window.rc_dx_spin.setSingleStep(0.01)
    window.rc_dx_spin.setValue(DEFAULT_DS)
    g_dx.add_pair("dx:", window.rc_dx_spin, label_span=1, input_span=2)
    window.rc_method_combo = QtWidgets.QComboBox()
    window.rc_method_combo.addItems(["rigid DFTB", "pm-NEB SP", "pm-NEB relaxed"])
    g_dx.add(window.rc_method_combo, span=3)
    layout.addLayout(g_dx.layout())

    g_run = GridPlacer(cols=6)
    window.rc_run_btn = QtWidgets.QPushButton("Run scan")
    window.rc_run_btn.clicked.connect(lambda: run_scan(window))
    g_run.add(window.rc_run_btn, span=2)
    window.rc_preview_btn = QtWidgets.QPushButton("Preview path")
    window.rc_preview_btn.setToolTip("Rigid endpoints: only H atoms move (fast). Use Run scan + pm-NEB relaxed for full geometry.")
    window.rc_preview_btn.clicked.connect(lambda: run_preview_scan(window))
    g_run.add(window.rc_preview_btn, span=2)
    window.rc_save_btn = QtWidgets.QPushButton("Save npz")
    window.rc_save_btn.clicked.connect(lambda: save_npz(window))
    g_run.add(window.rc_save_btn, span=1)
    window.rc_load_btn = QtWidgets.QPushButton("Load npz")
    window.rc_load_btn.clicked.connect(lambda: load_npz(window))
    g_run.add(window.rc_load_btn, span=1)
    layout.addLayout(g_run.layout())

    g_esp = GridPlacer(cols=6)
    window.rc_esp_z_spin = QtWidgets.QDoubleSpinBox()
    window.rc_esp_z_spin.setRange(-20.0, 20.0)
    window.rc_esp_z_spin.setSingleStep(0.5)
    window.rc_esp_z_spin.setValue(2.0)
    window.rc_esp_z_spin.setToolTip("Height above molecular center for ESP slice (Å)")
    g_esp.add_pair("ESP z:", window.rc_esp_z_spin, label_span=1, input_span=1)
    window.rc_esp_n_spin = QtWidgets.QSpinBox()
    window.rc_esp_n_spin.setRange(32, 400)
    window.rc_esp_n_spin.setValue(128)
    window.rc_esp_n_spin.setToolTip("Grid resolution (max axis)")
    g_esp.add_pair("grid:", window.rc_esp_n_spin, label_span=1, input_span=1)
    window.rc_esp_btn = QtWidgets.QPushButton("ESP animation")
    window.rc_esp_btn.setToolTip("Precompute Coulomb ESP from Mulliken charges; animate with scan slider")
    window.rc_esp_btn.clicked.connect(lambda: open_rc_esp_animation(window))
    g_esp.add(window.rc_esp_btn, span=2)
    layout.addLayout(g_esp.layout())

    window.rc_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    window.rc_slider.setMinimum(0)
    window.rc_slider.setMaximum(0)
    window.rc_slider.valueChanged.connect(lambda v: show_scan_frame(window, v))
    layout.addWidget(window.rc_slider)
    window.rc_status = QtWidgets.QLabel("Status: idle")
    layout.addWidget(window.rc_status)

    window.rc_dataset = None
    window.rc_hbonds = []
    window.rc_mapping = []
    window.rc_reference_apo = None
    window.rc_preview_apo = None
    window._rc_esp_view = None
    window._rc_esp_stack = None
    window._rc_esp_cache_key = None

    class RCPinMode(EditModeHandler):
        status_msg = "RC pin — click atoms to toggle spatial constraint"
        def on_atom_click(self, atom_id):
            handle_rc_pin_click(window, atom_id)

    window.register_mode_handler('rc_pin', RCPinMode(window))
    edit_modes = [('RC pin', lambda: window.set_edit_mode('rc_pin'))]
    return UIComponents(panel=panel, edit_modes=edit_modes)


def import_from_graph(window):
    """Snapshot geometry + H-bonds from backend (Import from Graph button)."""
    backend = window.backend
    backend.ensure_sys()
    window.rc_reference_apo = backend.sys.apos.copy()
    refresh_hbonds(window)
    window.rc_status.setText(f"Imported {backend.sys.natoms} atoms")


def _n_scan_hbonds(window):
    """Number of H-bonds in the active scan (matches `_scan_hbonds()` length)."""
    n = len(window.rc_hbonds)
    if n == 0:
        return 0
    if getattr(window, 'rc_all_hbonds_chk', None) and window.rc_all_hbonds_chk.isChecked():
        return n
    return 1


def refresh_hbonds(window):
    """Detect H-bonds on current backend.sys (Refresh H-bonds button)."""
    window.rc_hbonds = find_hbonds_graph(window.backend, bPrint=False)
    n = len(window.rc_hbonds)
    window.rc_hbond_spin.setMaximum(max(0, n - 1))
    n_scan = _n_scan_hbonds(window)
    window.rc_mapping = default_mapping(n_scan, m=1) if n_scan > 0 else []
    label = f"H-bonds: {n} detected"
    if n_scan > 1:
        label += f" (scan all, mapping={window.rc_mapping})"
    elif n == 1:
        label += f" (mapping={window.rc_mapping})"
    window.rc_hbond_label.setText(label)
    if n == 0:
        window.rc_status.setText("No H-bonds found")


def configure_scan(window, pair=0, dx=None, method=None, all_hbonds=None, relax_endpoints=None):
    """Set widget values for scan setup (same fields user edits before Run)."""
    if all_hbonds is not None and hasattr(window, 'rc_all_hbonds_chk'):
        window.rc_all_hbonds_chk.setChecked(bool(all_hbonds))
    if pair is not None:
        window.rc_hbond_spin.setValue(int(pair))
    if dx is not None:
        window.rc_dx_spin.setValue(float(dx))
    if method is not None:
        window.rc_method_combo.setCurrentText(method)
    if relax_endpoints is not None and hasattr(window, 'rc_relax_endpoints_chk'):
        window.rc_relax_endpoints_chk.setChecked(bool(relax_endpoints))
    refresh_hbonds(window)


def _active_hbonds(window):
    if not window.rc_hbonds:
        return []
    i = window.rc_hbond_spin.value()
    return [window.rc_hbonds[i]] if i < len(window.rc_hbonds) else []


def _scan_hbonds(window):
    """H-bonds included in scan (one pair or all symmetric)."""
    if not window.rc_hbonds:
        return []
    if getattr(window, 'rc_all_hbonds_chk', None) and window.rc_all_hbonds_chk.isChecked():
        return list(window.rc_hbonds)
    return _active_hbonds(window)


def _scan_topology(window):
    backend = window.backend
    backend.ensure_sys()
    sys = backend.sys
    bonds = np.asarray(sys.bonds, dtype=np.int32) if sys.bonds is not None else np.zeros((0, 2), dtype=np.int32)
    from spammm import elements as el
    etype = np.array([el.ELEMENT_DICT[e][0] for e in sys.enames], dtype=np.int32)
    atom_ids = backend._atom_ids.copy() if getattr(backend, '_atom_ids', None) is not None and len(backend._atom_ids) == sys.natoms else np.arange(sys.natoms, dtype=np.int64)
    return sys, bonds, etype, atom_ids


def _apply_dataset_to_ui(window, ds, status_msg):
    window.rc_dataset = ds
    window.rc_slider.setMaximum(max(0, ds.nframes - 1))
    window.rc_slider.setValue(0)
    show_scan_frame(window, 0)
    window.rc_status.setText(status_msg)


def run_preview_scan(window):
    """Build scan trajectory geometry without DFTB (Preview path button / fast GUI script)."""
    backend = window.backend
    backend.ensure_sys()
    if window.rc_reference_apo is None:
        window.rc_reference_apo = backend.sys.apos.copy()
    hb = _scan_hbonds(window)
    if not hb:
        window.rc_status.setText("Select an H-bond first")
        return None
    dx = window.rc_dx_spin.value()
    method = window.rc_method_combo.currentText()
    sys, bonds, etype, atom_ids = _scan_topology(window)
    m = max(window.rc_mapping) + 1
    meta = dict(dx=dx, mapping=list(window.rc_mapping), hbond_records=[h.to_dict() for h in hb])
    if method.startswith('pm'):
        u1d = build_control_grid([(0.0, 1.0)], dx=dx)[:, 0]
        apos_start = build_frame(window.rc_reference_apo, hb, np.zeros(m), window.rc_mapping)
        apos_end = build_frame(window.rc_reference_apo, hb, np.ones(m), window.rc_mapping)
        apos_stack = interpolate_all_atoms(apos_start, apos_end, u1d)
        controls = u1d[:, np.newaxis]
        meta['scan_type'] = 'pm_neb_preview'
        meta['rigid_endpoints'] = True
    else:
        controls = build_control_grid([(0.0, 1.0)], dx=dx)
        apos_stack = np.array([build_frame(window.rc_reference_apo, hb, u_row, window.rc_mapping) for u_row in controls])
        meta['scan_type'] = 'rigid_preview'
    energies = np.full(len(apos_stack), np.nan)
    ds = dataset_from_frames(etype, bonds, atom_ids, apos_stack, controls, energies, meta)
    _apply_dataset_to_ui(window, ds, f"Preview: {ds.nframes} frames ({ds.meta.get('scan_type')}) — rigid H only")
    return ds


def run_scan(window):
    """Run DFTB scan (Run scan button)."""
    backend = window.backend
    backend.ensure_sys()
    sys = backend.sys
    if window.rc_reference_apo is None:
        window.rc_reference_apo = sys.apos.copy()
    hb = _scan_hbonds(window)
    if not hb:
        window.rc_status.setText("Select an H-bond first")
        return None
    dx = window.rc_dx_spin.value()
    bonds, etype, atom_ids = _scan_topology(window)[1:]
    work_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'rc_scan')
    method = window.rc_method_combo.currentText()
    relax_endpoints = method == 'pm-NEB relaxed' or (getattr(window, 'rc_relax_endpoints_chk', None) and window.rc_relax_endpoints_chk.isChecked() and method.startswith('pm'))
    try:
        if method.startswith('pm'):
            ds = run_pm_neb(sys.enames, window.rc_reference_apo, hb, window.rc_mapping, dx=dx, relax_endpoints=relax_endpoints, run_sp=(method == 'pm-NEB SP'), etype=etype, bonds=bonds, atom_ids=atom_ids, work_dir=work_dir, verbose=True, on_fail='skip')
        else:
            ds = run_rigid_dftb_scan(sys.enames, window.rc_reference_apo, hb, window.rc_mapping, dx=dx, etype=etype, bonds=bonds, atom_ids=atom_ids, work_dir=work_dir, verbose=True, on_fail='skip')
    except Exception as exc:
        window.rc_status.setText(f"Scan failed: {exc}")
        raise
    _apply_dataset_to_ui(window, ds, f"Scan done: {ds.nframes} frames ({ds.meta.get('scan_type')})")
    return ds


def enable_bond_length_visualization(window, on=True):
    """Toggle bond-length coloring (View menu shortcut equivalent)."""
    window.bond_viz_mode = bool(on)
    if window.rc_preview_apo is not None:
        apply_preview_to_scene(window, window.rc_preview_apo)


def apply_preview_to_scene(window, apos, frame_idx=None):
    """Update Vispy scene from preview coordinates incl. bonds + H-bonds."""
    if apos is None or not hasattr(window, 'scene'):
        return
    window.rc_preview_apo = np.asarray(apos, dtype=float).copy()
    pos = window.rc_preview_apo.astype(np.float32)
    window.scene.update_positions(pos)
    sys = window.backend.sys
    if sys.bonds is None:
        return
    is_heavy = np.array([sys.enames[i] != 'H' for i in range(len(sys.enames))])
    bonds_heavy = [b for b in sys.bonds if is_heavy[b[0]] and is_heavy[b[1]]]
    heavy_bi = [bi for bi, b in enumerate(sys.bonds) if is_heavy[b[0]] and is_heavy[b[1]]]
    bonds_h = [b for b in sys.bonds if not (is_heavy[b[0]] and is_heavy[b[1]])]
    ds = window.rc_dataset
    fi = int(frame_idx) if frame_idx is not None else 0
    if getattr(window, 'bond_viz_mode', False) and bonds_heavy:
        if ds is not None and fi < ds.nframes and len(heavy_bi):
            delta = ds.bond_len[fi, heavy_bi] - ds.bond_len[0, heavy_bi]
            bond_segs, bond_colors = compute_bond_colors_by_delta(bonds_heavy, pos, delta, scale=0.06)
        else:
            bond_segs, bond_colors = compute_bond_colors_by_length(bonds_heavy, pos)
        window.scene._line_set("bonds-colored", window.scene.bond_colored_lines, bond_segs, color=bond_colors, width=5.0)
        window.scene.bond_colored_lines.visible = True
        window.scene.bond_lines.visible = False
    elif bonds_heavy:
        h_segs = pos[np.array(bonds_heavy)].reshape(-1, 3)
        window.scene._line_set("bonds", window.scene.bond_lines, h_segs, color=(0.5, 0.5, 0.5, 0.8), width=2.0)
        window.scene.bond_colored_lines.visible = False
        window.scene.bond_lines.visible = True
    if bonds_h:
        ch_segs = pos[np.array(bonds_h)].reshape(-1, 3)
        window.scene._line_set("CH-bonds", window.scene.ch_bond_lines, ch_segs, color=(0.4, 0.4, 0.4, 0.6), width=1.0)
    sys.neighs()
    hbonds = sys.find_hbonds(bPrint=False)
    if hbonds:
        hb_segs = []
        for d, h, a, dist, ang in hbonds:
            hb_segs.append(pos[h])
            hb_segs.append(pos[a])
        window.scene._line_set("H-bonds", window.scene.hbond_lines, np.array(hb_segs, dtype=np.float32), color=(0.8, 0.2, 0.8, 0.5), width=1.5)
    if hasattr(window.scene, 'canvas'):
        window.scene.canvas.update()


def show_scan_frame(window, frame_idx):
    """Show scan frame on slider (same as rc_slider valueChanged)."""
    ds = window.rc_dataset
    if ds is None:
        return
    fi = int(frame_idx)
    apply_preview_to_scene(window, ds.frame(fi), frame_idx=fi)
    e = ds.energies_ev[fi]
    etxt = f" E={e:.4f} eV" if np.isfinite(e) else ""
    u = ds.controls[fi]
    window.rc_status.setText(f"Frame {fi}/{ds.nframes-1} u={u}{etxt}")
    update_rc_esp_frame(window, fi)


def sync_to_graph(window):
    """Write preview frame to AtomicGraph (Sync to Graph button)."""
    apos = window.rc_preview_apo if window.rc_preview_apo is not None else (window.rc_dataset.frame(window.rc_slider.value()) if window.rc_dataset else None)
    if apos is None:
        window.rc_status.setText("Nothing to sync")
        return
    backend = window.backend
    if not backend.graph.atoms:
        backend.sys.apos = apos.copy()
        window.refresh_view()
        window.rc_status.setText("Synced frame to sys (ASCII load)")
        return
    backend.ensure_sys()
    atom_list, _, _, _, _, _, _ = backend.graph.to_arrays()
    for i, atom in enumerate(atom_list):
        atom.pos = apos[i].copy()
    backend._sync_sys()
    window.refresh_view()
    window.rc_status.setText("Synced full frame to graph")


def save_npz(window):
    if window.rc_dataset is None:
        return
    path, _ = QtWidgets.QFileDialog.getSaveFileName(window, "Save ScanDataset", "scan.npz", "NumPy zip (*.npz)")
    if path:
        window.rc_dataset.save_npz(path)
        window.rc_status.setText(f"Saved {path}")


def load_npz_path(window, path):
    """Load ScanDataset from path (no file dialog)."""
    ds = ScanDataset.load_npz(path)
    _apply_dataset_to_ui(window, ds, f"Loaded {path}")
    return ds


def load_npz(window):
    path, _ = QtWidgets.QFileDialog.getOpenFileName(window, "Load ScanDataset", "", "NumPy zip (*.npz)")
    if path:
        load_npz_path(window, path)


def handle_rc_pin_click(window, atom_idx):
    pinned = window.backend.toggle_constraint_by_index(atom_idx)
    mask = window.backend.constraint_mask()
    if hasattr(window, 'scene'):
        window.scene.set_fixed_mask(mask)
    state = "pinned" if pinned else "unpinned"
    window.rc_status.setText(f"Atom {atom_idx} {state}")
