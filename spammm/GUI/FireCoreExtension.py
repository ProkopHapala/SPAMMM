"""
FireCoreExtension.py — GUI extension for FireCore SCF, orbital/density plotting.

Provides UI panel for running FireCore SCF computation, plotting molecular
orbitals, electron density, and delta-rho on a 2D grid above the molecule.

Wired through ExtensionManager as 'firecore' extension.
"""

import os
import numpy as np
from PyQt5 import QtWidgets

from .ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH, AutoGridPlacer
from spammm.GUI import VispyUtils as vu
from spammm import elements


def build_ui(window):
    """Build FireCore panel for the SPAMMM GUI.

    Returns ExtensionManager.UIComponents.
    """
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    scf_btn = QtWidgets.QPushButton("Compute SCF")
    scf_btn.clicked.connect(lambda: _on_compute_scf(window))
    layout.addWidget(scf_btn)

    window.orbital_info_label = QtWidgets.QLabel("Orbitals: Not computed")
    window.orbital_info_label.setWordWrap(True)
    layout.addWidget(window.orbital_info_label)

    g1 = AutoGridPlacer(cols=4)
    window.z_height_spinbox = window.spinBox(2.0, 0.5, vmin=-10.0, vmax=20.0)
    g1.add_pair("Z:", window.z_height_spinbox)
    window.orbital_spinbox = window.spinBox(0, vmin=0, vmax=999, enabled=False, callback=lambda v: _update_orbital_energy_label(window, v), int_mode=True)
    g1.add_pair("Orb:", window.orbital_spinbox)
    layout.addLayout(g1.layout())

    g2 = AutoGridPlacer(cols=4)
    window.plot_orb_btn = QtWidgets.QPushButton("Plot Orb")
    window.plot_orb_btn.setEnabled(False)
    window.plot_orb_btn.clicked.connect(lambda: _on_plot_orbital(window))
    g2.add(window.plot_orb_btn)
    window.plot_density_btn = QtWidgets.QPushButton("Plot Dens")
    window.plot_density_btn.setEnabled(False)
    window.plot_density_btn.clicked.connect(lambda: _on_plot_density(window))
    g2.add(window.plot_density_btn)
    window.plot_delta_btn = QtWidgets.QPushButton("Plot Delta")
    window.plot_delta_btn.setEnabled(False)
    window.plot_delta_btn.clicked.connect(lambda: _on_plot_delta_rho(window))
    g2.add(window.plot_delta_btn)
    layout.addLayout(g2.layout())

    fdata_btn = QtWidgets.QPushButton("Set Fdata")
    fdata_btn.clicked.connect(lambda: _on_set_fdata_path(window))
    layout.addWidget(fdata_btn)

    view_modes = [
        ('Molecular Orbital', lambda: window.set_view_mode('orbital')),
        ('Density',           lambda: window.set_view_mode('density')),
        ('Delta-Rho',         lambda: window.set_view_mode('delta_rho')),
    ]
    return UIComponents(panel=panel, view_modes=view_modes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_extent_from_geometry(apos, padding_factor=0.1):
    """Compute grid extent and origin from atomic positions."""
    if len(apos) > 0:
        apos_2d = apos[:, :2]
        min_pos = apos_2d.min(axis=0)
        max_pos = apos_2d.max(axis=0)
        center_z = apos[:, 2].mean()
        padding = (max_pos - min_pos) * padding_factor
        grid_origin = min_pos - padding
        size = (max_pos - min_pos + 2 * padding).max()
    else:
        raise ValueError("Cannot compute grid extent: no atoms in system. Add atoms first.")
    return grid_origin, size, center_z

def _make_2d_grid(grid_origin, size, center_z, z_height, n=100):
    """Generate 2D grid points for projection."""
    xs = np.linspace(grid_origin[0], grid_origin[0] + size, n)
    ys = np.linspace(grid_origin[1], grid_origin[1] + size, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X) + (center_z + z_height)
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    extent = [grid_origin[0], grid_origin[0] + size, grid_origin[1], grid_origin[1] + size]
    return points, extent, n

def _evaluate_on_grid(window, points, what, orb_index=None):
    """Call FireCore evaluator on grid points."""
    fc = window.extensions.require('firecore')
    points_f64 = points.astype(np.float64)
    if what == 'orbital':
        if orb_index is None:
            raise ValueError("orb_index required for orbital evaluation")
        return fc.orb2points(points_f64, iMO=int(orb_index + 1), ikpoint=1)
    elif what == 'density':
        return fc.dens2points(points_f64, f_den=1.0, f_den0=0.0)
    elif what == 'delta_rho':
        return fc.dens2points(points_f64, f_den=1.0, f_den0=-1.0)
    else:
        raise ValueError(f"Unknown what={what}, expected 'orbital', 'density', or 'delta_rho'")

def _plot_2d_projection(data_2d, extent, title, cmap, symmetric, atom_pos, atom_types):
    """Plot 2D projection using VisPy heatmap."""
    canvas, view = vu.create_heatmap_window(data_2d, extent, title=title, cmap=cmap, symmetric=symmetric, atom_pos=atom_pos, atom_types=atom_types)
    return canvas, view


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _update_orbital_energy_label(window, value):
    """Update orbital energy label when spinbox value changes."""
    if hasattr(window, '_eigen'):
        idx = int(value)
        if 0 <= idx < len(window._eigen):
            window.orbital_info_label.setText(f"Orbital {idx} E: {window._eigen[idx]:.3f} eV")
        else:
            window.orbital_info_label.setText("Orbital: Invalid index")

def _on_compute_scf(window):
    """Run FireCore SCF and store eigenvalues/wfcoef on window."""
    if len(window.backend.sys.apos) == 0:
        QtWidgets.QMessageBox.warning(window, "Warning", "No atoms to compute orbitals for.")
        return

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    FDATA_DIR = os.path.join(_THIS_DIR, "Fdata")
    FDATA_TARGET = window.fdata_path

    if not os.path.exists(FDATA_DIR):
        if os.path.exists(FDATA_TARGET):
            os.symlink(FDATA_TARGET, FDATA_DIR)
        else:
            msg = f"Neither {FDATA_DIR} nor {FDATA_TARGET} exists. Please download Fdata_HC_minimal from fireball-qmd.github.io"
            QtWidgets.QMessageBox.critical(window, "Fdata Error", msg)
            return

    window.statusBar().showMessage("Running FireCore SCF...")
    QtWidgets.QApplication.processEvents()
    try:
        atypes = np.array([elements.ELEMENT_DICT[e][0] for e in window.backend.sys.enames], dtype=np.int32)
        apos = np.array(window.backend.sys.apos, dtype=np.float64)
        fc = window.extensions.require('firecore')
        fc.setVerbosity(0)
        fc.initialize(atomType=atypes, atomPos=apos)
        fc.evalForce(apos, nmax_scf=200)
        dims = fc.get_HS_dims()
        norb = int(dims.norbitals)
        window._eigen = fc.get_eigen(ikp=1, norb=norb)
        window._wfcoef = fc.get_wfcoef(norb=norb)
        window._norb = norb
        occ = np.where(window._eigen < 0.0)[0]
        window._homo = int(occ[-1]) if len(occ) > 0 else len(window._eigen) // 2 - 1
        window._lumo = window._homo + 1

        valence_dict = {'H': 1, 'C': 4, 'N': 5, 'O': 6}
        total_electrons = sum([valence_dict.get(e, 0) for e in window.backend.sys.enames])
        occupied_orbitals = total_electrons // 2

        info_text = (f"Total Orbitals: {norb}\n"
                     f"HOMO: {window._homo + 1} (E={window._eigen[window._homo]:.3f} eV)\n"
                     f"LUMO: {window._lumo + 1} (E={window._eigen[window._lumo]:.3f} eV)\n"
                     f"Occupied: {occupied_orbitals} (e-/2)")
        window.orbital_info_label.setText(info_text)

        window.orbital_spinbox.setEnabled(True)
        window.setSpinBox(window.orbital_spinbox, vmin=0, vmax=norb-1, value=window._homo)
        window.plot_orb_btn.setEnabled(True)
        window.plot_density_btn.setEnabled(True)
        window.plot_delta_btn.setEnabled(True)

        _update_orbital_energy_label(window, window._homo)

        msg = f"SCF done. HOMO={window._homo + 1} E={window._eigen[window._homo]:.3f} eV  LUMO={window._lumo + 1} E={window._eigen[window._lumo]:.3f} eV"
        window.statusBar().showMessage(msg)
        QtWidgets.QMessageBox.information(window, "SCF Done", f"HOMO={window._homo + 1} E={window._eigen[window._homo]:.3f} eV\nLUMO={window._lumo + 1} E={window._eigen[window._lumo]:.3f} eV")
    except Exception as e:
        window._raise(f"SCF FAILED: {e}", title="SCF Error")

def _on_plot_orbital(window):
    """Plot orbital at the index selected in spinbox."""
    if not hasattr(window, '_eigen'):
        QtWidgets.QMessageBox.information(window, "Info", "Please run Compute SCF first.")
        return
    mo_idx = window.orbital_spinbox.value()
    if mo_idx < 0 or mo_idx >= len(window._eigen):
        QtWidgets.QMessageBox.warning(window, "Warning", f"Invalid orbital index: {mo_idx}")
        return
    z_height = window.z_height_spinbox.value()
    window.statusBar().showMessage(f"Projecting MO {mo_idx + 1}...")
    QtWidgets.QApplication.processEvents()
    try:
        apos = window.backend.sys.apos
        grid_origin, size, center_z = _compute_extent_from_geometry(apos)
        points, extent, n = _make_2d_grid(grid_origin, size, center_z, z_height)
        flat_data = _evaluate_on_grid(window, points, 'orbital', orb_index=mo_idx)
        data_2d = np.asarray(flat_data, dtype=np.float64).reshape(n, n)
        E = window._eigen[mo_idx]
        pos = apos.astype(np.float32)
        enames = window.backend.sys.enames
        _plot_2d_projection(data_2d, extent, title=f"MO {mo_idx + 1} E={E:+.3f} eV  z={z_height:.1f}Å", cmap='bwr', symmetric=True, atom_pos=pos, atom_types=enames)
        window.statusBar().showMessage(f"Plotted MO {mo_idx + 1}")
    except Exception as e:
        window._raise(f"Plot FAILED: {e}", title="Plot Error")

def _on_plot_density(window):
    """Plot electron density on 2D grid."""
    if not hasattr(window, '_eigen'):
        QtWidgets.QMessageBox.information(window, "Info", "Please run Compute SCF first.")
        return
    window.statusBar().showMessage("Computing electron density...")
    QtWidgets.QApplication.processEvents()
    try:
        apos = window.backend.sys.apos
        z_height = window.z_height_spinbox.value()
        grid_origin, size, center_z = _compute_extent_from_geometry(apos)
        points, extent, n = _make_2d_grid(grid_origin, size, center_z, z_height)
        flat_data = _evaluate_on_grid(window, points, 'density')
        data_2d = np.asarray(flat_data, dtype=np.float64).reshape(n, n)
        pos = apos.astype(np.float32)
        enames = list(window.backend.sys.enames)
        _plot_2d_projection(data_2d, extent, title=f"Electron Density (z={z_height:.1f}Å)", cmap='bwr', symmetric=False, atom_pos=pos, atom_types=enames)
        window.statusBar().showMessage("Density plotted")
    except Exception as e:
        window._raise(f"Density plot FAILED: {e}", title="Plot Error")

def _on_plot_delta_rho(window):
    """Plot delta-rho (rho_SCF - rho_NA) on 2D grid."""
    if not hasattr(window, '_eigen'):
        QtWidgets.QMessageBox.information(window, "Info", "Please run Compute SCF first.")
        return
    window.statusBar().showMessage("Computing delta-rho (rho_SCF - rho_NA)...")
    QtWidgets.QApplication.processEvents()
    try:
        apos = window.backend.sys.apos
        z_height = window.z_height_spinbox.value()
        grid_origin, size, center_z = _compute_extent_from_geometry(apos)
        points, extent, n = _make_2d_grid(grid_origin, size, center_z, z_height)
        flat_data = _evaluate_on_grid(window, points, 'delta_rho')
        data_2d = np.asarray(flat_data, dtype=np.float64).reshape(n, n)
        pos = apos.astype(np.float32)
        enames = list(window.backend.sys.enames)
        _plot_2d_projection(data_2d, extent, title=f"Delta-Rho (z={z_height:.1f}Å)", cmap='bwr', symmetric=True, atom_pos=pos, atom_types=enames)
        window.statusBar().showMessage("Delta-rho plotted")
    except Exception as e:
        window._raise(f"Delta-rho plot FAILED: {e}", title="Plot Error")

def _on_set_fdata_path(window):
    """Open dialog to set Fdata path and save to settings."""
    selected = window.fileDialog(mode="directory", title="Select Fdata Directory", start_dir=window.fdata_path)
    if selected:
        window.fdata_path = selected
        window.settings.setValue("fdata_path", selected)
        window.extensions.set_config('firecore', 'fdata_dir', selected)
        window.extensions.save_config()
        window.statusBar().showMessage(f"Fdata path set to: {selected}")
        QtWidgets.QMessageBox.information(window, "Settings Saved", f"Fdata path set to:\n{selected}")
