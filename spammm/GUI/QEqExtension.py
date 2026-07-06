"""
QEqExtension.py — GUI extension for Charge Equilibration (QEq) via direct matrix solve.

Provides a panel to compute partial atomic charges using the Rappe & Goddard QEq
method with Cholesky+Schur (default) or LU (backup) solvers.

Wired through ExtensionManager as 'qeq' extension.
"""

from PyQt5 import QtWidgets, QtCore
import numpy as np

from .ExtensionManager import UIComponents
from spammm.forcefields.QEq import solve, solve_cholesky, solve_lu, KE
from spammm.topology.FFparams import read_element_types
import os

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')


def build_ui(window):
    """Build QEq panel for the SPAMMM GUI.

    Returns ExtensionManager.UIComponents.
    """
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setSpacing(2)
    layout.setContentsMargins(2, 2, 2, 2)

    # --- Method selector ---
    row1 = QtWidgets.QHBoxLayout()
    row1.addWidget(QtWidgets.QLabel("Method:"))
    window.qeq_method_combo = QtWidgets.QComboBox()
    window.qeq_method_combo.addItems(['cholesky', 'lu'])
    window.qeq_method_combo.setMaximumWidth(100)
    row1.addWidget(window.qeq_method_combo)
    row1.addStretch()
    layout.addLayout(row1)

    # --- Q_target ---
    row2 = QtWidgets.QHBoxLayout()
    row2.addWidget(QtWidgets.QLabel("Q_total:"))
    window.qeq_qtarget_spin = QtWidgets.QDoubleSpinBox()
    window.qeq_qtarget_spin.setRange(-100.0, 100.0)
    window.qeq_qtarget_spin.setSingleStep(0.1)
    window.qeq_qtarget_spin.setValue(0.0)
    window.qeq_qtarget_spin.setMaximumWidth(80)
    row2.addWidget(window.qeq_qtarget_spin)
    row2.addStretch()
    layout.addLayout(row2)

    # --- Solve button ---
    window.qeq_solve_btn = QtWidgets.QPushButton("Solve QEq")
    window.qeq_solve_btn.clicked.connect(lambda: _on_solve(window))
    layout.addWidget(window.qeq_solve_btn)

    # --- Plot ESP button + z-height ---
    row_esp = QtWidgets.QHBoxLayout()
    row_esp.addWidget(QtWidgets.QLabel("ESP z:"))
    window.qeq_esp_z_spin = QtWidgets.QDoubleSpinBox()
    window.qeq_esp_z_spin.setRange(-10.0, 20.0)
    window.qeq_esp_z_spin.setSingleStep(0.5)
    window.qeq_esp_z_spin.setValue(3.0)
    window.qeq_esp_z_spin.setMaximumWidth(60)
    row_esp.addWidget(window.qeq_esp_z_spin)
    window.qeq_plot_esp_btn = QtWidgets.QPushButton("Plot ESP")
    window.qeq_plot_esp_btn.clicked.connect(lambda: _on_plot_esp(window))
    row_esp.addWidget(window.qeq_plot_esp_btn)
    row_esp.addStretch()
    layout.addLayout(row_esp)

    # --- Status ---
    window.qeq_status_label = QtWidgets.QLabel("Status: Ready")
    window.qeq_status_label.setWordWrap(True)
    window.qeq_status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(window.qeq_status_label)

    # --- Charge display ---
    window.qeq_charge_label = QtWidgets.QLabel("Charges: ---")
    window.qeq_charge_label.setWordWrap(True)
    window.qeq_charge_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(window.qeq_charge_label)

    layout.addStretch()

    view_modes = [
        ('Charge Labels', lambda: _toggle_charge_labels(window)),
    ]

    return UIComponents(panel=panel, edit_modes=[], view_modes=view_modes)


def _on_solve(window):
    """Run QEq charge equilibration on the current backend system."""
    sys = window.backend.sys if window.backend is not None else None
    if sys is None or sys.apos is None or len(sys.apos) == 0:
        window.qeq_status_label.setText("Status: No molecule loaded")
        return
    try:
        pos = np.asarray(sys.apos, dtype=np.float64)
        enames = sys.enames

        etypes = read_element_types(os.path.join(_DATA_PATH, 'ElementTypes.dat'))
        chi = np.array([etypes[e].Eaff for e in enames], dtype=np.float64)
        hardness = np.array([etypes[e].Ehard for e in enames], dtype=np.float64)

        missing = [e for e in enames if e not in etypes or not etypes[e].bQEq]
        if missing:
            window.qeq_status_label.setText(f"Status: Missing QEq params for: {missing}")
            return

        method = window.qeq_method_combo.currentText()
        Q_target = window.qeq_qtarget_spin.value()

        q = solve(pos, chi, hardness, Q_target=Q_target, method=method)

        window.backend.set_atom_charges(q)
        window.qeq_charges = q
        q_str = np.round(q, 4).tolist()
        window.qeq_charge_label.setText(f"Charges: {q_str}")
        window.qeq_status_label.setText(
            f"Status: Solved ({method}) — sum={q.sum():.2e} |q|_max={np.abs(q).max():.4f}"
        )

        if hasattr(window, 'set_label_mode'):
            window.set_label_mode('Charge')

        if hasattr(window, 'refresh_view'):
            window.refresh_view()
    except Exception as e:
        window.qeq_status_label.setText(f"Status: QEq FAILED: {e}")
        import traceback; traceback.print_exc()


def _on_plot_esp(window):
    """Plot electrostatic potential from QEq charges on a 2D grid above the molecule."""
    from .plotutils import compute_grid_extent, make_2d_grid, plot_2d_scalar, show_in_plot_window

    q = getattr(window, 'qeq_charges', None)
    if q is None:
        window.qeq_status_label.setText("Status: Solve QEq first")
        return
    sys = window.backend.sys if window.backend is not None else None
    if sys is None or sys.apos is None or len(sys.apos) == 0:
        window.qeq_status_label.setText("Status: No molecule loaded")
        return
    try:
        pos = np.asarray(sys.apos, dtype=np.float64)
        enames = sys.enames
        z_height = window.qeq_esp_z_spin.value()

        grid_origin, size_xy, center_z = compute_grid_extent(pos)
        points, extent, nx, ny = make_2d_grid(grid_origin, size_xy, center_z, z_height)

        # ESP = sum_i q_i * KE / |r - r_i|  (Coulomb potential in eV, distances in Angstrom)
        r = points[:, None, :] - pos[None, :, :]  # (npoints, natoms, 3)
        d = np.linalg.norm(r, axis=2)              # (npoints, natoms)
        esp = KE * (q[None, :] / d).sum(axis=1)     # (npoints,)
        data_2d = esp.reshape(ny, nx)

        fig = plot_2d_scalar(data_2d, extent, title=f"Electrostatic Potential  z={z_height:.1f}Å",
                             z_label='ESP (eV)', cmap='seismic', symmetric=True,
                             apos=pos, enames=enames)
        show_in_plot_window(window, fig, title=f"ESP (z={z_height:.1f}Å)", attr='_qeq_plot_window')

        vmax = max(abs(data_2d.min()), abs(data_2d.max()))
        window.qeq_status_label.setText(f"Status: ESP plotted at z={z_height:.1f}Å (range ±{vmax:.4f} eV)")
    except Exception as e:
        window.qeq_status_label.setText(f"Status: ESP plot FAILED: {e}")
        import traceback; traceback.print_exc()


def _toggle_charge_labels(window):
    """Toggle charge label display in the scene."""
    window.show_charge_labels = not getattr(window, 'show_charge_labels', False)
    if hasattr(window, 'refresh_view'):
        window.refresh_view()
