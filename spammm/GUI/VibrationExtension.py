"""
VibrationExtension.py — GUI panel for normal-mode analysis (ExtensionManager: `vibrations`).

Compute runs in a background QThread; results appear in a clickable table. One plot
window is reused per session — row click updates the figure (no multi-window flood).
Backend and display units can be changed without recomputing the Hessian.
"""

from __future__ import annotations

import copy
import os

from PyQt5 import QtCore, QtWidgets

from .ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH, AutoGridPlacer
from .plotutils import show_in_plot_window
from spammm.AtomicSystem import AtomicSystem
from spammm.dynamics.Vibrations import run_vibrations, FREQ_UNIT_LABELS, format_freq, format_E_zpe
from spammm.dynamics.VibrationPlot import make_mode_figure

_BACKEND_MAP = {'UFF': 'uff', 'SPFF': 'spff', 'DFTB': 'dftb'}
_UNIT_ITEMS = [('cm⁻¹', 'cm-1'), ('meV', 'meV'), ('THz', 'THz'), ('kcal/mol', 'kcal/mol')]


class _VibWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, mol, backend, parent=None):
        super().__init__(parent)
        self._mol = mol
        self._backend = backend

    def run(self):
        try:
            result = run_vibrations(self._mol, backend=self._backend)
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


def _unit_key(window) -> str:
    return window.vib_unit_combo.currentData()


def _table_headers(unit: str):
    flab = FREQ_UNIT_LABELS[unit]
    elab = flab if unit in ('meV', 'kcal/mol') else 'eV'
    return ['#', f'freq/{flab}', f'E_zpe/{elab}', 'f_xy', 'f_z', 'character']


def build_ui(window):
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    g1 = AutoGridPlacer(cols=4)
    window.vib_backend_combo = QtWidgets.QComboBox()
    window.vib_backend_combo.addItems(['UFF', 'SPFF', 'DFTB'])
    window.vib_backend_combo.setCurrentText('UFF')
    window.vib_backend_combo.setToolTip('UFF: GPU force field. SPFF: pi-aware FF. DFTB: native Hessian via DFTB+.')
    window.vib_backend_combo.setMaximumWidth(72)
    g1.add_pair("Backend:", window.vib_backend_combo)
    window.vib_unit_combo = QtWidgets.QComboBox()
    for label, key in _UNIT_ITEMS:
        window.vib_unit_combo.addItem(label, key)
    window.vib_unit_combo.setMaximumWidth(80)
    window.vib_unit_combo.currentIndexChanged.connect(lambda _: _on_unit_changed(window))
    g1.add_pair("Units:", window.vib_unit_combo)
    window.vib_compute_btn = QtWidgets.QPushButton("Compute Modes")
    window.vib_compute_btn.clicked.connect(lambda: _on_compute(window))
    g1.add(window.vib_compute_btn)
    layout.addLayout(g1.layout())

    window.vib_status_label = QtWidgets.QLabel("Status: Ready — click a row to plot")
    window.vib_status_label.setWordWrap(True)
    window.vib_status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(window.vib_status_label)

    window.vib_mode_table = QtWidgets.QTableWidget(0, 6)
    window.vib_mode_table.setHorizontalHeaderLabels(_table_headers('cm-1'))
    window.vib_mode_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    window.vib_mode_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    window.vib_mode_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    window.vib_mode_table.verticalHeader().setVisible(False)
    window.vib_mode_table.setAlternatingRowColors(True)
    window.vib_mode_table.setEnabled(False)
    window.vib_mode_table.cellClicked.connect(lambda row, _col: _show_mode_figure(window, row))
    hh = window.vib_mode_table.horizontalHeader()
    hh.setStretchLastSection(True)
    hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
    layout.addWidget(window.vib_mode_table, stretch=1)

    layout.addStretch()
    window.vib_result = None
    window._vib_worker = None
    return UIComponents(panel=panel)


def _mol_from_window(window):
    sys = window.backend.sys if window.backend is not None else None
    if sys is None or sys.apos is None or len(sys.apos) == 0:
        return None
    mol = AtomicSystem(apos=sys.apos.copy(), enames=list(sys.enames), bonds=copy.deepcopy(sys.bonds) if sys.bonds is not None else None, lvec=copy.deepcopy(sys.lvec) if getattr(sys, 'lvec', None) is not None else None)
    if mol.bonds is None:
        mol.findBonds()
    if mol.ngs is None:
        mol.neighs()
    return mol


def _backend_key(window):
    return _BACKEND_MAP[window.vib_backend_combo.currentText()]


def _set_busy(window, busy: bool):
    window.vib_compute_btn.setEnabled(not busy)
    window.vib_backend_combo.setEnabled(not busy)
    window.vib_unit_combo.setEnabled(not busy)
    if busy:
        window.vib_mode_table.setEnabled(False)


def _fill_mode_table(window, result):
    unit = _unit_key(window)
    tbl = window.vib_mode_table
    tbl.setHorizontalHeaderLabels(_table_headers(unit))
    tbl.setRowCount(0)
    tbl.setRowCount(len(result.mode_info))
    for row, m in enumerate(result.mode_info):
        vals = [str(m.index), format_freq(m.freq_cm1, unit), format_E_zpe(m.freq_cm1, unit), f'{m.f_xy:.3f}', f'{m.f_z:.3f}', m.character]
        for col, text in enumerate(vals):
            item = QtWidgets.QTableWidgetItem(text)
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter if col < 5 else QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            tbl.setItem(row, col, item)
    tbl.setEnabled(len(result.mode_info) > 0)
    if len(result.mode_info) > 0:
        tbl.selectRow(0)


def _on_unit_changed(window):
    if getattr(window, 'vib_result', None) is not None:
        _fill_mode_table(window, window.vib_result)
        row = window.vib_mode_table.currentRow()
        if row >= 0:
            _show_mode_figure(window, row)


def _on_compute(window):
    mol = _mol_from_window(window)
    if mol is None:
        window.vib_status_label.setText('Status: No molecule loaded')
        return
    backend = _backend_key(window)
    if backend == 'dftb' and not os.environ.get('DFTB_EXE'):
        window.vib_status_label.setText('Status: DFTB_EXE not set')
        return
    if backend in ('uff', 'spff'):
        try:
            import pyopencl  # noqa: F401
        except ImportError:
            window.vib_status_label.setText('Status: pyopencl required for force-field Hessian')
            return
    if window._vib_worker is not None and window._vib_worker.isRunning():
        window.vib_status_label.setText('Status: Computation already running')
        return
    _set_busy(window, True)
    nat = len(mol.enames)
    window.vib_status_label.setText(f'Status: Computing Hessian ({backend}, N={nat})…')
    QtWidgets.QApplication.processEvents()
    worker = _VibWorker(mol, backend, parent=window)
    window._vib_worker = worker

    def _done(result):
        _set_busy(window, False)
        window.vib_result = result
        _fill_mode_table(window, result)
        nm = len(result.mode_info)
        window.vib_status_label.setText(f'Status: {nm} modes ({backend}) — click a row to plot')

    def _fail(msg):
        _set_busy(window, False)
        window.vib_result = None
        window.vib_mode_table.setRowCount(0)
        window.vib_mode_table.setEnabled(False)
        window.vib_status_label.setText('Status: FAILED')
        QtWidgets.QMessageBox.critical(window, 'Vibrations', msg)

    worker.finished_ok.connect(_done)
    worker.failed.connect(_fail)
    worker.start()


def _show_mode_figure(window, mode_index: int):
    result = getattr(window, 'vib_result', None)
    if result is None or mode_index < 0 or mode_index >= len(result.mode_info):
        return
    unit = _unit_key(window)
    import matplotlib
    matplotlib.use('Qt5Agg')
    fig = make_mode_figure(result, mode_index, unit=unit)
    mi = result.mode_info[mode_index]
    flab = result.mode_freq_label(mode_index, unit)
    title = f"mode {mode_index}: {flab} ({mi.character})"
    show_in_plot_window(window, fig, title=title, attr='_vib_plot_window')
    window.vib_status_label.setText(f'Status: mode {mode_index} — {flab} ({mi.character})')
