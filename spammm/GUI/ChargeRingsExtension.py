"""ChargeRingsExtension — GUI for Pauli Master Equation charge-ring STM (PME.cl).

Panel: load/save JSON params, spinbox edits, Calc XY / xV / 1D with cut-line
overlay on xy and many-body state probability plots.

Uses ``spammm.quantum.pauli_scan`` + ``PauliSolverCL``. Sibling of AFMExtension
(not stuffed into FDBM). Audit: ``doc/TopicalAudit/ChargeRings_PME.md``.
"""
from __future__ import annotations

import json
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore

from .ExtensionManager import UIComponents
from .CollapsibleSection import CollapsibleSection
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH, AutoGridPlacer

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'charge_rings')

# Keys exposed as spinboxes (label, key, min, max, step, decimals)
_SPIN_SPEC = [
    ('nsite', 'nsite', 1, 4, 1, 0),
    ('radius', 'radius', 1.0, 20.0, 0.1, 2),
    ('phiRot', 'phiRot', -6.3, 6.3, 0.1, 3),
    ('Esite', 'Esite', -1.0, 1.0, 0.01, 3),
    ('W', 'W', 0.0, 0.5, 0.01, 3),
    ('Q0', 'Q0', 0.0, 5.0, 0.1, 2),
    ('Qzz', 'Qzz', -20.0, 20.0, 0.5, 2),
    ('VBias', 'VBias', 0.0, 3.0, 0.05, 3),
    ('z_tip', 'z_tip', 1.0, 15.0, 0.5, 2),
    ('zV0', 'zV0', -5.0, 5.0, 0.1, 2),
    ('Temp', 'Temp', 0.1, 100.0, 0.5, 2),
    ('GammaT', 'GammaT', 1e-4, 1.0, 0.01, 4),
    ('decay', 'decay', 0.05, 2.0, 0.05, 3),
    ('L', 'L', 5.0, 40.0, 1.0, 1),
    ('npix', 'npix', 20, 200, 10, 0),
    ('p1_x', 'p1_x', -40.0, 40.0, 0.5, 2),
    ('p1_y', 'p1_y', -40.0, 40.0, 0.5, 2),
    ('p2_x', 'p2_x', -40.0, 40.0, 0.5, 2),
    ('p2_y', 'p2_y', -40.0, 40.0, 0.5, 2),
    ('nx', 'nx', 20, 200, 10, 0),
    ('nV', 'nV', 20, 200, 10, 0),
    ('Vmin', 'Vmin', 0.0, 2.0, 0.05, 3),
    ('Vmax', 'Vmax', 0.1, 3.0, 0.05, 3),
]


def build_ui(window):
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    # --- Preset ---
    g0 = AutoGridPlacer(cols=4)
    window.cr_preset = QtWidgets.QComboBox()
    window.cr_preset.addItems(['symmetric_trimer', 'fig3_trimer', 'Ruslan_long', 'Ruslan_short', 'square_tetramer'])
    g0.add_pair("Preset:", window.cr_preset)
    window.cr_load_preset_btn = QtWidgets.QPushButton('Load preset')
    window.cr_load_preset_btn.clicked.connect(lambda: load_preset(window))
    g0.add(window.cr_load_preset_btn)
    layout.addLayout(g0.layout())

    # --- JSON I/O ---
    g1 = AutoGridPlacer(cols=4)
    window.cr_load_json_btn = QtWidgets.QPushButton('Load JSON…')
    window.cr_load_json_btn.clicked.connect(lambda: load_json(window))
    g1.add(window.cr_load_json_btn)
    window.cr_save_json_btn = QtWidgets.QPushButton('Save JSON…')
    window.cr_save_json_btn.clicked.connect(lambda: save_json(window))
    g1.add(window.cr_save_json_btn)
    layout.addLayout(g1.layout())

    # --- Param spins (scrollable) ---
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setMaximumHeight(280)
    spin_host = QtWidgets.QWidget()
    g_spins = AutoGridPlacer(cols=4)
    window.cr_spins = {}
    for label, key, lo, hi, step, dec in _SPIN_SPEC:
        if dec == 0:
            sp = QtWidgets.QSpinBox()
            sp.setRange(int(lo), int(hi))
            sp.setSingleStep(int(step))
        else:
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
        sp.setMaximumWidth(90)
        window.cr_spins[key] = sp
        g_spins.add_pair(label, sp)
    spin_host.setLayout(g_spins.layout())
    scroll.setWidget(spin_host)
    layout.addWidget(scroll)

    window.cr_bMirror = QtWidgets.QCheckBox('bMirror')
    window.cr_bMirror.setChecked(True)
    window.cr_bRamp = QtWidgets.QCheckBox('bRamp')
    window.cr_bRamp.setChecked(True)
    g_chk = AutoGridPlacer(cols=4)
    g_chk.add(window.cr_bMirror)
    g_chk.add(window.cr_bRamp)
    layout.addLayout(g_chk.layout())

    # --- Calc buttons ---
    g_calc = AutoGridPlacer(cols=4)
    window.cr_xy_btn = QtWidgets.QPushButton('Calc XY')
    window.cr_xy_btn.setToolTip('Constant-VBias xy STM + dI/dV; overlay xV cut line')
    window.cr_xy_btn.clicked.connect(lambda: calc_xy(window))
    g_calc.add(window.cr_xy_btn)
    window.cr_xv_btn = QtWidgets.QPushButton('Calc xV')
    window.cr_xv_btn.setToolTip('Line×voltage scan (diamonds / NDR) + state probs')
    window.cr_xv_btn.clicked.connect(lambda: calc_xv(window))
    g_calc.add(window.cr_xv_btn)
    window.cr_1d_btn = QtWidgets.QPushButton('Calc 1D')
    window.cr_1d_btn.setToolTip('Fixed-V line cut + many-body P(s)')
    window.cr_1d_btn.clicked.connect(lambda: calc_1d(window))
    g_calc.add(window.cr_1d_btn)
    layout.addLayout(g_calc.layout())

    window.cr_status = QtWidgets.QLabel('Status: Load preset or JSON, then Calc')
    window.cr_status.setWordWrap(True)
    window.cr_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(window.cr_status)
    layout.addStretch()

    # --- Lattice PME8 (periodic) — Temirov rectangular-lattice mode ---
    _build_lattice_ui(window, layout)

    window._cr_solver = None
    window._cr_xy = None
    window._cr_xv = None
    load_preset(window)  # default symmetric_trimer
    return UIComponents(panel=panel, edit_modes=[], view_modes=[])


# ===========================================================================
# Lattice PME8 (periodic) — Temirov rectangular-lattice interactive mode
# Backend: spammm/quantum/pme_lattice.py (see its docstring for conventions).
# ===========================================================================
_LAT_SPIN_SPEC = [  # (label, key, min, max, step, decimals)
    ('VBias', 'VBias', 1.0, 3.0, 0.005, 3),
    ('z_charge', 'z_charge', 5.0, 30.0, 0.05, 2),
    ('Qzz', 'Qzz', -100.0, 100.0, 0.2, 2),
    ('zVd', 'zVd', 0.0, 60.0, 0.1, 2),
    ('E_ref', 'E_ref', -0.3, 0.3, 0.0005, 4),
    ('W_a meV', 'W_a_meV', 0.0, 150.0, 0.5, 2),
    ('W_b meV', 'W_b_meV', 0.0, 150.0, 0.5, 2),
    ('W_ab meV', 'W_ab_meV', 0.0, 150.0, 0.5, 2),
    ('px', 'px', -5.0, 5.0, 0.05, 3),
    ('py', 'py', -5.0, 5.0, 0.05, 3),
    ('pz', 'pz', -5.0, 5.0, 0.05, 3),
    ('Qxx', 'Qxx', -500.0, 500.0, 1.0, 1),
    ('Qyy', 'Qyy', -500.0, 500.0, 1.0, 1),
    ('Qxy', 'Qxy', -500.0, 500.0, 1.0, 1),
    ('Qxz', 'Qxz', -500.0, 500.0, 1.0, 1),
    ('Qyz', 'Qyz', -500.0, 500.0, 1.0, 1),
    ('Q0', 'Q0', 0.0, 5.0, 0.1, 2),
    ('Rtip', 'Rtip', 0.5, 10.0, 0.1, 2),
    ('zV0', 'zV0', -5.0, 5.0, 0.1, 2),
    ('Temp', 'Temp', 0.1, 100.0, 0.5, 2),
    ('GammaT', 'GammaT', 1e-4, 1.0, 0.01, 4),
    ('decay', 'decay', 0.05, 2.0, 0.05, 3),
    ('npix', 'npix', 17, 257, 16, 0),
    ('ref R px', 'ring_ref_px', 0.0, 120.0, 1.0, 1),
    ('cut_Vmin', 'cut_Vmin', 0.5, 3.0, 0.005, 3),
    ('cut_Vmax', 'cut_Vmax', 0.5, 3.0, 0.005, 3),
    ('cut_dV', 'cut_dV', 0.001, 0.1, 0.005, 3),
    ('cut_nfrac', 'cut_nfrac', 11, 161, 10, 0),
    ('n1', 'n1', 1, 4, 1, 0),
    ('n2', 'n2', 1, 4, 1, 0),
]

_LAT_COMBO_SPEC = [  # (label, key, items)
    ('solver', 'solver', ['auto', 'pme4', 'pme8']),
    ('method', 'stationary_method', ['gmres', 'bicgstab', 'jacobi', 'euler']),
    ('precision', 'stationary_precision', ['double', 'single']),
]


def _build_lattice_ui(window, layout):
    """Collapsible 'Lattice PME8 (periodic)' section appended below cluster controls."""
    sec = CollapsibleSection('Lattice PME8 (periodic)', collapsed=False)
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    apply_tight(lay)

    g_btn = AutoGridPlacer(cols=4)
    window.lat_live_chk = QtWidgets.QCheckBox('Auto-update')
    window.lat_live_chk.setChecked(True)
    window.lat_live_chk.setToolTip('Recompute the map automatically 0.35 s after any parameter change')
    g_btn.add(window.lat_live_chk)
    window.lat_load_btn = QtWidgets.QPushButton('Load lattice JSON…')
    window.lat_load_btn.clicked.connect(lambda: lat_load_json(window))
    g_btn.add(window.lat_load_btn)
    window.lat_save_btn = QtWidgets.QPushButton('Save lattice JSON…')
    window.lat_save_btn.clicked.connect(lambda: lat_save_json(window))
    g_btn.add(window.lat_save_btn)
    window.lat_map_btn = QtWidgets.QPushButton('Map')
    window.lat_map_btn.setToolTip('dI/dV + current map at npix')
    window.lat_map_btn.clicked.connect(lambda: lat_run_map(window))
    g_btn.add(window.lat_map_btn)
    window.lat_hires_btn = QtWidgets.QPushButton('Hi-res')
    window.lat_hires_btn.setToolTip('One-shot map at npix=257')
    window.lat_hires_btn.clicked.connect(lambda: lat_run_map(window, npix=257))
    g_btn.add(window.lat_hires_btn)
    window.lat_cuts_btn = QtWidgets.QPushButton('Cuts')
    window.lat_cuts_btn.setToolTip('long/short/diagonal x-V cuts vs experiment')
    window.lat_cuts_btn.clicked.connect(lambda: lat_run_cuts(window))
    g_btn.add(window.lat_cuts_btn)
    lay.addLayout(g_btn.layout())

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setMaximumHeight(220)
    spin_host = QtWidgets.QWidget()
    g_sp = AutoGridPlacer(cols=4)
    window.lat_spins = {}
    for label, key, lo, hi, step, dec in _LAT_SPIN_SPEC:
        if dec == 0:
            sp = QtWidgets.QSpinBox(); sp.setRange(int(lo), int(hi)); sp.setSingleStep(int(step))
        else:
            sp = QtWidgets.QDoubleSpinBox(); sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(dec)
        sp.setMaximumWidth(SPIN_MAX_WIDTH)
        sp.valueChanged.connect(lambda *a: _lat_live_kick(window))
        if key in ('n1', 'n2'):
            sp.valueChanged.connect(lambda *a: _lat_update_solver_combos(window))
        window.lat_spins[key] = sp
        g_sp.add_pair(label, sp)
    spin_host.setLayout(g_sp.layout())
    scroll.setWidget(spin_host)
    lay.addWidget(scroll)

    g_chk = AutoGridPlacer(cols=4)
    window.lat_occ_chk = QtWidgets.QCheckBox('occupied_env')
    window.lat_occ_chk.setChecked(True)
    window.lat_occ_chk.setToolTip('E_i = E_ref - Wij row sum (occupied-environment convention)')
    window.lat_occ_chk.stateChanged.connect(lambda *a: _lat_live_kick(window))
    g_chk.add(window.lat_occ_chk)
    window.lat_bMirror_chk = QtWidgets.QCheckBox('bMirror')
    window.lat_bMirror_chk.setChecked(True)
    window.lat_bMirror_chk.stateChanged.connect(lambda *a: _lat_live_kick(window))
    g_chk.add(window.lat_bMirror_chk)
    window.lat_bRamp_chk = QtWidgets.QCheckBox('bRamp')
    window.lat_bRamp_chk.setChecked(True)
    window.lat_bRamp_chk.stateChanged.connect(lambda *a: _lat_live_kick(window))
    g_chk.add(window.lat_bRamp_chk)
    for label, key, items in _LAT_COMBO_SPEC:
        cb = QtWidgets.QComboBox()
        cb.addItems(items)
        cb.currentTextChanged.connect((lambda *a: _lat_solver_changed(window)) if key == 'solver' else (lambda *a: _lat_live_kick(window)))
        setattr(window, f'lat_{key}_combo', cb)
        g_chk.add_pair(label, cb)
    lay.addLayout(g_chk.layout())

    window.lat_status = QtWidgets.QLabel('Lattice: load a lattice_pme8 preset JSON')
    window.lat_status.setWordWrap(True)
    window.lat_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    lay.addWidget(window.lat_status)

    sec.setContent(host)
    layout.addWidget(sec)

    window._lat_params = None
    window._lat_model = None
    window._lat_solvers = {}
    window._lat_geo_key = None
    window._lat_map_dlg = None
    window._lat_cuts_dlg = None
    window._lat_cut_lines = None
    window._lat_map_art = None
    window._lat_map_suptitle = None
    window._lat_cuts_art = None
    window._lat_busy = False
    window._lat_pending = None
    window._lat_eff_solver = None
    window._lat_timer = QtCore.QTimer(window if isinstance(window, QtCore.QObject) else None)
    window._lat_timer.setSingleShot(True)
    window._lat_timer.setInterval(350)
    window._lat_timer.timeout.connect(lambda: lat_run_map(window))


def _lat_status(window, msg, error=False):
    window.lat_status.setStyleSheet('color: red' if error else '')
    window.lat_status.setText(f'Lattice: {msg}')


def _lat_nsites(window, p):
    """n1*n2*nbasis for the params' geometry (model geometry if already built)."""
    g = window._lat_model.geometry if window._lat_model is not None else json.loads(open(p['geometry_file']).read())
    return int(p['supercell'][0]) * int(p['supercell'][1]) * len(g['basis_A'])


def _lat_check_solver(window, p):
    """Error string for invalid solver/supercell combos, else None."""
    if p.get('solver', 'auto') != 'pme4':
        return None
    n = _lat_nsites(window, p)
    return f'pme4 needs ≤4 sites: set n1=n2=2 or solver=auto/pme8 (now {n})' if n > 4 else None


def _lat_update_solver_combos(window):
    """method/precision only apply to pme8 — disable when effective solver is pme4.
    On an effective-solver CHANGE, clamp npix to the interactive range
    (pme4 fast → >=129; pme8 slow → <=65)."""
    if getattr(window, 'lat_stationary_method_combo', None) is None:
        return
    p = _lat_params(window)
    kind = p.get('solver', 'auto')
    if kind == 'auto' and window._lat_params is not None:
        try:
            kind = 'pme4' if _lat_nsites(window, p) <= 4 else 'pme8'
        except Exception:
            kind = 'pme8'
    if kind in ('pme4', 'pme8') and getattr(window, '_lat_eff_solver', None) != kind and window._lat_params is not None:
        npix_sp = window.lat_spins['npix']
        if kind == 'pme4' and npix_sp.value() < 129:
            npix_sp.setValue(129)
            _lat_status(window, f'{kind} selected → npix set to 129')
        elif kind == 'pme8' and npix_sp.value() > 65:
            npix_sp.setValue(65)
            _lat_status(window, f'{kind} selected → npix set to 65')
    if kind in ('pme4', 'pme8'):
        window._lat_eff_solver = kind
    eff4 = kind == 'pme4'
    window.lat_stationary_method_combo.setEnabled(not eff4)
    window.lat_stationary_precision_combo.setEnabled(not eff4)


def _lat_solver_changed(window):
    """solver combo changed: pme4 needs <=4 sites — auto-set 2x2, then update/kick."""
    if window.lat_solver_combo.currentText() == 'pme4' and window._lat_params is not None and window._lat_params.get('geometry_file'):
        p = _lat_params(window)
        if _lat_nsites(window, p) > 4:
            window.lat_spins['n1'].setValue(2)
            window.lat_spins['n2'].setValue(2)
            _lat_status(window, 'pme4 selected → supercell set to 2×2')
    _lat_update_solver_combos(window)
    _lat_live_kick(window)


def _lat_params(window):
    """Full param dict: last loaded preset merged with current widget values."""
    p = dict(getattr(window, '_lat_params', None) or {})
    for key, sp in window.lat_spins.items():
        p[key] = int(sp.value()) if isinstance(sp, QtWidgets.QSpinBox) else float(sp.value())
    p['occupied_env_energy'] = window.lat_occ_chk.isChecked()
    p['bMirror'] = window.lat_bMirror_chk.isChecked()
    p['bRamp'] = window.lat_bRamp_chk.isChecked()
    for _, key, _ in _LAT_COMBO_SPEC:
        p[key] = getattr(window, f'lat_{key}_combo').currentText()
    p['supercell'] = [p.pop('n1'), p.pop('n2')]
    p['model'] = 'lattice_pme8'
    return p


def _lat_params_to_widgets(window, p):
    for key, sp in window.lat_spins.items():
        v = p.get('supercell', [None, None])[0] if key == 'n1' else (p.get('supercell', [None, None])[1] if key == 'n2' else p.get(key))
        if v is None:
            continue
        sp.blockSignals(True)
        sp.setValue(int(v) if isinstance(sp, QtWidgets.QSpinBox) else float(v))
        sp.blockSignals(False)
    for _, key, items in _LAT_COMBO_SPEC:
        if key in p:
            cb = getattr(window, f'lat_{key}_combo')
            cb.blockSignals(True)
            cb.setCurrentText(str(p[key]))
            cb.blockSignals(False)
    for key, chk in (('occupied_env_energy', window.lat_occ_chk), ('bMirror', window.lat_bMirror_chk), ('bRamp', window.lat_bRamp_chk)):
        if key in p:
            chk.setChecked(bool(p[key]))


def _lat_preset_dir():
    return os.environ.get('SPAMMM_LATTICE_PRESETS', _DATA)


def lat_load_json(window, path=None):
    from spammm.quantum import pme_lattice as pl
    if not path:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Load lattice JSON', _lat_preset_dir(), 'JSON (*.json)')
    if not path:
        return
    try:
        p = pl.load_lattice_params(path)
        window._lat_params = p
        _lat_params_to_widgets(window, p)
        _lat_update_solver_combos(window)
        _lat_status(window, f'Loaded {os.path.basename(path)}')
        _lat_live_kick(window)
    except Exception as e:
        _lat_status(window, f'Load FAILED: {e}')
        import traceback; traceback.print_exc()


def lat_save_json(window):
    path, _ = QtWidgets.QFileDialog.getSaveFileName(window, 'Save lattice JSON', _lat_preset_dir(), 'JSON (*.json)')
    if not path:
        return
    try:
        p = {k: v for k, v in _lat_params(window).items() if not k.startswith('_')}
        with open(path, 'w') as f:
            json.dump(p, f, indent=2)
        _lat_status(window, f'Saved {os.path.basename(path)}')
    except Exception as e:
        _lat_status(window, f'Save FAILED: {e}')
        import traceback; traceback.print_exc()


def _lat_live_kick(window):
    if getattr(window, 'lat_live_chk', None) is None or not window.lat_live_chk.isChecked():
        return
    if window._lat_params is None:
        return
    window._lat_timer.start()


def _lat_model(window, params):
    """Reuse model when geometry/supercell unchanged; solvers cached per window."""
    from spammm.quantum import pme_lattice as pl
    geo_key = (params.get('geometry_file'), tuple(params.get('supercell', [2, 4])))
    if window._lat_model is None or window._lat_geo_key != geo_key:
        window._lat_model = pl.LatticePME(params, solvers=window._lat_solvers)
        window._lat_geo_key = geo_key
    else:
        window._lat_model.configure(params)
    return window._lat_model


def _lat_dialog(window, attr, title, figsize=(11, 7)):
    """Persistent plot dialog; created once, artists updated in place afterwards."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    dlg = getattr(window, attr, None)
    if dlg is None:
        fig = Figure(figsize=figsize)
        axes = fig.subplots(2, 3)
        dlg = QtWidgets.QDialog(window if isinstance(window, QtWidgets.QWidget) else None)
        vbox = QtWidgets.QVBoxLayout(dlg)
        apply_tight(vbox)
        canvas = FigureCanvas(fig)
        vbox.addWidget(canvas)
        dlg.resize(1100, 700)
        setattr(window, attr, dlg)
        setattr(window, attr + '_fig', fig)
        setattr(window, attr + '_axes', axes)
        setattr(window, attr + '_canvas', canvas)
        dlg.finished.connect(lambda *a: _lat_dlg_finished(window, attr))
        dlg.setWindowTitle(title)
        dlg.show()
    return dlg, getattr(window, attr + '_fig'), getattr(window, attr + '_axes'), getattr(window, attr + '_canvas')


def _lat_dlg_finished(window, attr):
    """Drop the dialog AND its cached artists so a reopened window gets fresh artists."""
    for a in (attr, attr + '_fig', attr + '_axes', attr + '_canvas', attr.replace('_dlg', '_art'), attr.replace('_dlg', '_suptitle')):
        setattr(window, a, None)


def _lat_finish(window):
    """Clear busy flag; run at most one queued recompute requested during compute."""
    window._lat_busy = False
    pend, window._lat_pending = window._lat_pending, None
    if pend:
        kind, arg = pend
        lat_run_map(window, arg) if kind == 'map' else lat_run_cuts(window)


def _lat_fail(window, kind, e):
    """Red status + red FAILED suptitle on the map dialog if it exists."""
    _lat_status(window, f'{kind} FAILED: {e}', error=True)
    if getattr(window, '_lat_map_dlg', None) is not None and getattr(window, '_lat_map_suptitle', None) is not None:
        window._lat_map_suptitle.set_text(f'FAILED: {e}')
        window._lat_map_suptitle.set_color('red')
        window._lat_map_dlg_canvas.draw_idle()
    import traceback; traceback.print_exc()


def lat_run_map(window, npix=None):
    import time
    if getattr(window, '_lat_busy', False):
        window._lat_pending = ('map', npix)
        return
    try:
        if window._lat_params is None:
            _lat_status(window, 'load a lattice preset first')
            return
        window._lat_busy = True
        _lat_status(window, 'computing…')
        QtWidgets.QApplication.processEvents()
        t0 = time.monotonic()
        p = _lat_params(window)
        err = _lat_check_solver(window, p)
        if err:
            raise ValueError(err)
        model = _lat_model(window, p)
        npix = int(npix if npix is not None else p.get('npix', 65))
        raw, mV = model.experiment_image(float(p['VBias']))
        if raw is None:
            W = H = int(p.get('image_size_px', 256))
            exp = None
        else:
            H, W = raw.shape
        scan = model.scan_map(float(p['VBias']), npix, W, H)
        window._lat_last_scan = scan
        if raw is not None:
            exp = model.experiment_on_grid(raw, scan['pixelxy'], scan['dIdV'].shape)
        step = float(scan['pixels'][0][1] - scan['pixels'][0][0])
        scores = model.image_scores(scan['dIdV'], exp, step) if exp is not None else None
        _lat_show_map(window, model, p, scan, exp, raw, mV, scores, W, H)
        _lat_status(window, f"Map npix={npix} V={p['VBias']:.3f} in {time.monotonic()-t0:.1f}s" + (f" signed={scores['signed']:.3f} magn={scores['magnitude']:.3f}" if scores else ' (no experiment)'))
    except Exception as e:
        _lat_fail(window, 'Map', e)
    finally:
        _lat_finish(window)


def _lat_show_map(window, model, p, scan, exp, raw, mV, scores, W, H):
    import matplotlib
    matplotlib.use('Qt5Agg')
    extent = [0, W - 1, H - 1, 0]
    dlg, fig, axes, canvas = _lat_dialog(window, '_lat_map_dlg', 'Lattice PME8 map')
    art = getattr(window, '_lat_map_art', None)
    if art is None:
        art = dict(ims={}, extra=[])
        window._lat_map_art = art
    d = scan['dIdV']
    sc = max(float(np.abs(d).max()), 1e-30)
    panels = [
        (raw, 'gray', None, f'experiment {mV} mV' if raw is not None else 'no experiment'),
        (d, 'bwr', (-sc, sc), 'PME dI/dV'),
        (scan['I'], 'inferno', None, 'PME current'),
    ]
    step = float(scan['pixels'][0][1] - scan['pixels'][0][0])
    for i, (img, cmap, clim, title) in enumerate(panels):
        ax = axes.flat[i]
        im = art['ims'].get(i)
        if img is None:
            img = np.zeros((2, 2))
        if im is None:
            im = ax.imshow(img, origin='upper', extent=extent, cmap=cmap, aspect='equal')
            art['ims'][i] = im
        else:
            im.set_data(img)
        if cmap == 'gray' and raw is not None:
            lo, hi = np.percentile(raw, [1, 99])
            im.set_clim(lo, hi)
        elif clim is not None:
            im.set_clim(*clim)
        else:
            im.autoscale()
        ax.set_title(title, fontsize=9)
    # overlay panel: experiment gray + PME dI/dV contours + site markers + cut lines
    ax = axes.flat[3]
    ax.clear()
    if raw is not None:
        lo, hi = np.percentile(raw, [1, 99])
        ax.imshow(raw, origin='upper', extent=extent, cmap='gray', vmin=lo, vmax=hi, aspect='equal')
    else:
        ax.imshow(np.zeros((2, 2)), origin='upper', extent=extent, cmap='gray', aspect='equal')
    px, py = scan['pixels']
    ax.contour(px, py, d, levels=[0.3 * sc], colors=['cyan'], linewidths=0.8)
    ax.contour(px, py, d, levels=[-0.3 * sc], colors=['magenta'], linewidths=0.8)
    sites = model.site_pixels(W, H)
    ax.plot(sites[:, 0], sites[:, 1], 'y+', ms=5, mew=0.8)
    r0 = float(p.get('ring_ref_px', 0.))  # fixed reference-ring radius for visual shift checks
    if r0 > 0:
        th = np.linspace(0, 2 * np.pi, 64)
        for sx, sy in sites:
            ax.plot(sx + r0 * np.cos(th), sy + r0 * np.sin(th), 'y--', lw=0.7)
    for line in (window._lat_cut_lines or []):
        ax.plot(line[:, 0], line[:, 1], 'w-', lw=0.7, alpha=0.8)
    ax.set_title('overlay + sites', fontsize=9)
    # band-pass panels (4 px feature, signed polarity)
    from spammm.quantum.pme_lattice import bandpass_image
    for i, img, title in ((4, exp, 'exp band-pass 4px'), (5, d, 'PME band-pass 4px')):
        ax = axes.flat[i]
        im = art['ims'].get(i)
        bp = bandpass_image(img, step) if img is not None else np.zeros((2, 2))
        bsc = max(float(np.abs(bp).max()), 1e-30)
        if im is None:
            im = ax.imshow(bp, origin='upper', extent=extent, cmap='bwr', vmin=-bsc, vmax=bsc, aspect='equal')
            art['ims'][i] = im
        else:
            im.set_data(bp); im.set_clim(-bsc, bsc)
        ax.set_title(title if img is not None else 'no experiment', fontsize=9)
    for ax in axes.flat:
        ax.set_xlim(0, W - 1); ax.set_ylim(H - 1, 0)
        ax.tick_params(labelsize=7)
    st = getattr(window, '_lat_map_suptitle', None)
    txt = (f"V={p['VBias']:.3f}  " + (f"signed={scores['signed']:.3f} magn={scores['magnitude']:.3f}  " if scores else 'no experiment  ') +
           f"px={p.get('px',0):.2f} py={p.get('py',0):.2f} Qxy={p.get('Qxy',0):.2f} Qxx-Qyy={p.get('Qxx',0)-p.get('Qyy',0):.2f} W=({p.get('W_a_meV',0):.1f},{p.get('W_b_meV',0):.1f},{p.get('W_ab_meV',0):.1f})meV")
    if st is None:
        window._lat_map_suptitle = fig.suptitle(txt, fontsize=9)
    else:
        st.set_text(txt)
        st.set_color('black')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    canvas.draw_idle()


def lat_run_cuts(window):
    import time
    if getattr(window, '_lat_busy', False):
        window._lat_pending = ('cuts', None)
        return
    try:
        if window._lat_params is None:
            _lat_status(window, 'load a lattice preset first')
            return
        window._lat_busy = True
        _lat_status(window, 'computing…')
        QtWidgets.QApplication.processEvents()
        t0 = time.monotonic()
        p = _lat_params(window)
        err = _lat_check_solver(window, p)
        if err:
            raise ValueError(err)
        model = _lat_model(window, p)
        Vmin, Vmax = float(p['cut_Vmin']), float(p['cut_Vmax'])
        step = max(5, round(float(p['cut_dV']) * 1000 / 5) * 5)  # mV; cut_dV is the voltage-grid spacing
        voltages = np.arange(round(Vmin * 1000 / 5) * 5, round(Vmax * 1000 / 5) * 5 + 1, step) / 1000
        nfrac = int(p.get('cut_nfrac', 41))
        cuts = model.scan_cuts(voltages, nfrac)
        exp, mV_used = model.sample_experiment_cuts(voltages, nfrac)
        corr = model.cut_correlations(cuts['dIdV'], exp) if exp is not None else None
        window._lat_cut_lines = [l for fam in cuts['pixels'] for l in fam]
        _lat_show_cuts(window, cuts, exp, corr)
        _lat_status(window, f"Cuts {len(voltages)}V x {nfrac} in {time.monotonic()-t0:.1f}s" + (f" corr={['%.3f' % c for c in corr]}" if corr else ' (no experiment)'))
        if window._lat_map_dlg is not None:
            lat_run_map(window)  # refresh overlay with cut lines
    except Exception as e:
        _lat_fail(window, 'Cuts', e)
    finally:
        _lat_finish(window)


def _lat_show_cuts(window, cuts, exp, corr):
    import matplotlib
    matplotlib.use('Qt5Agg')
    dlg, fig, axes, canvas = _lat_dialog(window, '_lat_cuts_dlg', 'Lattice PME8 cuts')
    art = getattr(window, '_lat_cuts_art', None)
    if art is None:
        art = dict(ims={})
        window._lat_cuts_art = art
    extent = [0, 1, cuts['voltages'][0], cuts['voltages'][-1]]
    for f, name in enumerate(cuts['names']):
        title = f"{name} corr={corr[f]:.3f}" if corr is not None else f'{name} (no experiment)'
        for row, img in ((0, exp[f] if exp is not None else None), (1, cuts['dIdV'][f])):
            ax = axes[row, f]
            key = (row, f)
            data = img if img is not None else np.zeros((2, 2))
            im = art['ims'].get(key)
            if row == 0 and exp is not None:
                lo, hi = np.percentile(exp[f], [1, 99])
                clim = (lo, hi)
            else:
                m = max(float(np.abs(data).max()), 1e-30)
                clim = (-m, m)
            if im is None:
                im = ax.imshow(data, origin='lower', extent=extent, aspect='auto', cmap='gray' if row == 0 else 'bwr', vmin=clim[0], vmax=clim[1])
                art['ims'][key] = im
            else:
                im.set_data(data); im.set_clim(*clim)
            ax.set_title(title if row == 0 else f'PME dI/dV {name}', fontsize=9)
            ax.tick_params(labelsize=7)
            if f == 0:
                ax.set_ylabel('V [V]')
            if row == 1:
                ax.set_xlabel('fraction')
    fig.tight_layout()
    canvas.draw_idle()


def _set_status(window, msg):
    window.cr_status.setText(f'Status: {msg}')


def params_from_widgets(window):
    p = {}
    for key, sp in window.cr_spins.items():
        p[key] = sp.value()
    p['bMirror'] = window.cr_bMirror.isChecked()
    p['bRamp'] = window.cr_bRamp.isChecked()
    # scan helpers expect these names
    p['zVd'] = p.get('zVd', 20.0)
    if 'zVd' not in window.cr_spins:
        p['zVd'] = getattr(window, '_cr_zVd', 20.0)
    p['Rtip'] = getattr(window, '_cr_Rtip', 3.0)
    p['GammaS'] = p.get('GammaT', 0.01)
    p['dQ'] = 0.02
    p['zQd'] = 0.0
    p['phi0_ax'] = getattr(window, '_cr_phi0_ax', 0.0)
    return p


def params_to_widgets(window, params):
    for key, sp in window.cr_spins.items():
        if key not in params:
            continue
        v = params[key]
        sp.blockSignals(True)
        if isinstance(sp, QtWidgets.QSpinBox):
            sp.setValue(int(v))
        else:
            sp.setValue(float(v))
        sp.blockSignals(False)
    if 'bMirror' in params:
        window.cr_bMirror.setChecked(bool(params['bMirror']))
    if 'bRamp' in params:
        window.cr_bRamp.setChecked(bool(params['bRamp']))
    window._cr_zVd = float(params.get('zVd', 20.0))
    window._cr_Rtip = float(params.get('Rtip', 3.0))
    window._cr_phi0_ax = float(params.get('phi0_ax', 0.0))
    # ensure nx/nV/Vmin/Vmax defaults
    if 'nx' not in params and 'nx' in window.cr_spins:
        window.cr_spins['nx'].setValue(100)
    if 'nV' not in params and 'nV' in window.cr_spins:
        window.cr_spins['nV'].setValue(80)
    if 'Vmin' not in params and 'Vmin' in window.cr_spins:
        window.cr_spins['Vmin'].setValue(0.0)
    if 'Vmax' not in params and 'Vmax' in window.cr_spins:
        window.cr_spins['Vmax'].setValue(float(params.get('VBias', 0.85)))


def load_preset(window):
    from spammm.quantum import pauli_scan as ps
    name = window.cr_preset.currentText()
    try:
        if name == 'symmetric_trimer':
            p = ps.symmetric_trimer_params()
            path = ps.default_geometry_path('symmetric_trimer.json')
            if os.path.isfile(path):
                p.update(ps.load_json_params(path))
        elif name == 'fig3_trimer':
            p = ps.fig3_trimer_params()
        elif name.startswith('Ruslan'):
            geom = ps.default_geometry_path(f'{name}.txt')
            p = ps.ruslan_default_params(geometry_file=geom, nsite=2)
            spos, _, _ = ps.load_site_geometry(geom)
            p['nsite'] = len(spos)
        elif name == 'square_tetramer':
            geom = ps.default_geometry_path('square_tetramer.txt')
            # NDR regime: match fig3_trimer params (Qzz=0, low VBias, low Temp)
            # ruslan_default_params has VBias=2.0 which is too high — washes out NDR
            p = ps.ruslan_default_params(geometry_file=geom, nsite=4, Qzz=0.0, W=0.05,
                                         VBias=0.85, Temp=2.6, z_tip=6.0, zV0=-0.9, zVd=20.0)
        else:
            p = ps.symmetric_trimer_params()
        p.setdefault('nx', 100)
        p.setdefault('nV', 80)
        p.setdefault('Vmin', 0.0)
        p.setdefault('Vmax', float(p.get('VBias', 0.85)))
        params_to_widgets(window, p)
        window._cr_params_extra = {k: v for k, v in p.items() if k not in window.cr_spins}
        _set_status(window, f'Loaded preset {name}')
    except Exception as e:
        _set_status(window, f'Preset FAILED: {e}')
        import traceback; traceback.print_exc()


def load_json(window):
    path, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Load charge-rings JSON', _DATA, 'JSON (*.json)')
    if not path:
        return
    from spammm.quantum import pauli_scan as ps
    try:
        p = ps.load_json_params(path)
        p.setdefault('nx', 100); p.setdefault('nV', 80)
        p.setdefault('Vmin', 0.0); p.setdefault('Vmax', float(p.get('VBias', 1.0)))
        params_to_widgets(window, p)
        window._cr_params_extra = p
        _set_status(window, f'Loaded {os.path.basename(path)}')
    except Exception as e:
        _set_status(window, f'Load FAILED: {e}')


def save_json(window):
    path, _ = QtWidgets.QFileDialog.getSaveFileName(window, 'Save charge-rings JSON', _DATA, 'JSON (*.json)')
    if not path:
        return
    p = params_from_widgets(window)
    extra = getattr(window, '_cr_params_extra', {}) or {}
    for k in ('zVd', 'Rtip', 'phi0_ax', 'geometry_file', 'GammaS', 'dQ', 'zQd'):
        if k in extra:
            p[k] = extra[k]
    p['zVd'] = getattr(window, '_cr_zVd', p.get('zVd', 20.0))
    p['Rtip'] = getattr(window, '_cr_Rtip', 3.0)
    p['phi0_ax'] = getattr(window, '_cr_phi0_ax', 0.0)
    # drop GUI-only keys from core file if desired — keep nx/nV/Vmin/Vmax for reload
    try:
        with open(path, 'w') as f:
            json.dump(p, f, indent=2)
        _set_status(window, f'Saved {os.path.basename(path)}')
    except Exception as e:
        _set_status(window, f'Save FAILED: {e}')


def _get_solver(window):
    if getattr(window, '_cr_solver', None) is None:
        from spammm.quantum.PauliSolverCL import PauliSolverCL
        window._cr_solver = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=False)
        name = window._cr_solver.ctx.devices[0].name
        _set_status(window, f'OpenCL: {name}')
    return window._cr_solver


def _site_geom(window, params):
    from spammm.quantum import pauli_scan as ps
    if params.get('geometry_file'):
        return ps.load_site_geometry(params['geometry_file'])
    # circle from widgets
    return ps.make_site_geom(params)


def _full_params(window):
    p = params_from_widgets(window)
    extra = getattr(window, '_cr_params_extra', {}) or {}
    p['zVd'] = getattr(window, '_cr_zVd', extra.get('zVd', 20.0))
    p['Rtip'] = getattr(window, '_cr_Rtip', extra.get('Rtip', 3.0))
    p['phi0_ax'] = getattr(window, '_cr_phi0_ax', extra.get('phi0_ax', 0.0))
    if 'geometry_file' in extra:
        p['geometry_file'] = extra['geometry_file']
    p['GammaS'] = p.get('GammaT', 0.01)
    p['dQ'] = 0.02
    p['zQd'] = 0.0
    return p


def calc_xy(window):
    from spammm.quantum import pauli_scan as ps
    from .plotutils import show_in_plot_window
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.figure import Figure
    try:
        params = _full_params(window)
        spos, rots, _ = _site_geom(window, params)
        solver = _get_solver(window)
        _set_status(window, 'Running xy…')
        QtWidgets.QApplication.processEvents()
        xy = ps.scan_xy(solver, spos, rots, params, return_probs=False)
        window._cr_xy = xy
        fig = Figure(figsize=(9, 4))
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)
        im0 = ax0.imshow(xy['STM'], origin='lower', extent=xy['extent'], cmap='inferno')
        ax0.plot(spos[:, 0], spos[:, 1], 'c+', ms=10, mew=1.5)
        # cut line from p1→p2
        ax0.plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'w-', lw=1.5, label='xV/1D cut')
        ax0.legend(loc='upper right', fontsize=8)
        ax0.set_title(f"STM V={params['VBias']:.3f}")
        ax0.set_xlabel('x [Å]'); ax0.set_ylabel('y [Å]')
        fig.colorbar(im0, ax=ax0, fraction=0.046)
        if xy['dIdV'] is not None:
            sc = max(np.nanmax(np.abs(xy['dIdV'])), 1e-30)
            im1 = ax1.imshow(xy['dIdV'], origin='lower', extent=xy['extent'], cmap='bwr', vmin=-sc, vmax=sc)
            ax1.plot(spos[:, 0], spos[:, 1], 'k+', ms=10, mew=1.5)
            ax1.plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'k-', lw=1.5)
            ax1.set_title('dI/dV (rings / NDR)')
            ax1.set_xlabel('x [Å]')
            fig.colorbar(im1, ax=ax1, fraction=0.046)
        fig.tight_layout()
        show_in_plot_window(window, fig, title='ChargeRings XY', attr='_cr_xy_window')
        _set_status(window, f"XY done Imax={xy['STM'].max():.3e}")
    except Exception as e:
        _set_status(window, f'XY FAILED: {e}')
        import traceback; traceback.print_exc()


def calc_xv(window):
    from spammm.quantum import pauli_scan as ps
    from .plotutils import show_in_plot_window
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.figure import Figure
    try:
        params = _full_params(window)
        spos, rots, _ = _site_geom(window, params)
        solver = _get_solver(window)
        nx = int(params.get('nx', 100))
        nV = int(params.get('nV', 80))
        Vmin = float(params.get('Vmin', 0.0))
        Vmax = float(params.get('Vmax', params['VBias']))
        _set_status(window, 'Running xV…')
        QtWidgets.QApplication.processEvents()
        xv = ps.scan_xV(solver, spos, rots, params, nx=nx, nV=nV, Vmin=Vmin, Vmax=Vmax, return_probs=True)
        window._cr_xv = xv

        fig = Figure(figsize=(8, 8))
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 1, 2)
        # use distance along cut for x axis
        x0, x1 = 0.0, xv['dist']
        ext = [x0, x1, Vmin, Vmax]
        im0 = ax0.imshow(xv['STM'], origin='lower', extent=ext, aspect='auto', cmap='inferno')
        ax0.set_ylabel('V [V]'); ax0.set_title('xV STM')
        fig.colorbar(im0, ax=ax0, fraction=0.046)
        sc = 0.5 * max(np.nanmax(np.abs(xv['dIdV'])), 1e-30)
        im1 = ax1.imshow(xv['dIdV'], origin='lower', extent=ext, aspect='auto', cmap='bwr', vmin=-sc, vmax=sc)
        ax1.set_xlabel('distance along cut [Å]'); ax1.set_ylabel('V [V]')
        ax1.set_title(f"dI/dV  NDR min={xv['dIdV'].min():.2e}")
        fig.colorbar(im1, ax=ax1, fraction=0.046)
        fig.tight_layout()
        show_in_plot_window(window, fig, title='ChargeRings xV', attr='_cr_xv_window')

        # state probabilities (active subspace masks 0..2^n_act-1)
        if xv.get('probs') is not None:
            _plot_state_probs_xV(window, xv)
        _set_status(window, f"xV done Imax={xv['STM'].max():.3e} NDRmin={xv['dIdV'].min():.2e}")
    except Exception as e:
        _set_status(window, f'xV FAILED: {e}')
        import traceback; traceback.print_exc()


def _plot_state_probs_xV(window, xv):
    from .plotutils import show_in_plot_window
    from matplotlib.figure import Figure
    P = xv['probs']  # [nV, nx, nStates]
    n_act = int(xv.get('n_active', 3))
    n_show = 1 << n_act  # only states without spectator occupation
    nV, nx, nSt = P.shape
    ncol = min(4, n_show)
    nrow = (n_show + ncol - 1) // ncol
    fig = Figure(figsize=(2.4 * ncol, 2.0 * nrow))
    ext = [0.0, xv['dist'], xv['Vbiases'][0], xv['Vbiases'][-1]]
    for s in range(n_show):
        ax = fig.add_subplot(nrow, ncol, s + 1)
        ax.imshow(P[:, :, s], origin='lower', extent=ext, aspect='auto', cmap='magma', vmin=0, vmax=1)
        ax.set_title(format(s, f'0{n_act}b'), fontsize=9)
        if s % ncol == 0:
            ax.set_ylabel('V')
        if s // ncol == nrow - 1:
            ax.set_xlabel('s [Å]')
    fig.suptitle('Many-body state probabilities P(s,V)', fontsize=11)
    fig.tight_layout()
    show_in_plot_window(window, fig, title='ChargeRings state probs (xV)', attr='_cr_prob_window')


def calc_1d(window):
    from spammm.quantum import pauli_scan as ps
    from .plotutils import show_in_plot_window
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.figure import Figure
    try:
        params = _full_params(window)
        spos, rots, _ = _site_geom(window, params)
        solver = _get_solver(window)
        nx = int(params.get('nx', 100))
        _set_status(window, 'Running 1D…')
        QtWidgets.QApplication.processEvents()
        r = ps.scan_1d(solver, spos, rots, params, nx=nx, return_probs=True)

        fig = Figure(figsize=(8, 6))
        ax0 = fig.add_subplot(2, 1, 1)
        ax0.plot(r['dist_axis'], r['I'], 'k-')
        ax0.set_ylabel('I'); ax0.set_title(f"1D cut V={r['VBias']:.3f}  ({r['start']}→{r['end']})")
        ax0.grid(True, alpha=0.3)
        ax1 = fig.add_subplot(2, 1, 2)
        n_act = int(r.get('n_active', 3))
        n_show = 1 << n_act
        P = r['probs']
        for s in range(n_show):
            ax1.plot(r['dist_axis'], P[:, s], label=format(s, f'0{n_act}b'))
        ax1.set_xlabel('distance along cut [Å]'); ax1.set_ylabel('P_state')
        ax1.legend(ncol=4, fontsize=7, loc='upper right')
        ax1.set_title('Many-body state probabilities')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        show_in_plot_window(window, fig, title='ChargeRings 1D + probs', attr='_cr_1d_window')

        # also refresh xy with cut if we have a previous xy at same V, else light xy for overlay context
        if window._cr_xy is None or abs(params['VBias'] - float(getattr(window, '_cr_xy_V', -1))) > 1e-6:
            xy = ps.scan_xy(solver, spos, rots, params)
            window._cr_xy = xy
            window._cr_xy_V = params['VBias']
        _show_xy_with_cut(window, window._cr_xy, spos, params)
        _set_status(window, f"1D done Imax={r['I'].max():.3e}")
    except Exception as e:
        _set_status(window, f'1D FAILED: {e}')
        import traceback; traceback.print_exc()


def _show_xy_with_cut(window, xy, spos, params):
    from .plotutils import show_in_plot_window
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(xy['STM'], origin='lower', extent=xy['extent'], cmap='inferno')
    ax.plot(spos[:, 0], spos[:, 1], 'c+', ms=10, mew=1.5)
    ax.plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'w-', lw=2, label='1D/xV cut')
    ax.legend(loc='upper right')
    ax.set_title('XY + cut line'); ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout()
    show_in_plot_window(window, fig, title='ChargeRings cut on XY', attr='_cr_cut_window')
