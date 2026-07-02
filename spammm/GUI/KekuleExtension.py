"""
KekuleExtension.py — GUI extension for heterocycle generation and Kekule bond-order optimization.

Purpose: Provide UI panel for generating heterocycle structures from ASCII art or
sparse grid descriptions, running the Kekule pi-bond-order optimizer, and exporting
results. Built on top of topology.ascii_art_heterocycle and topology.heterocycle_generator.

Key functionality:
  - build_ui(window) -> UIComponents with panel + edit modes + view modes
  - Panel: ASCII art text editor, example dropdown, Generate button, Kekule solver controls
  - Generate: parse ASCII art -> AtomicSystem -> load into backend
  - Solve: run two-phase Kekule solver, display bond orders
  - Export: save XYZ/MOL files

Role in SPAMMM: Wired through ExtensionManager as 'kekule' extension.
"""

from PyQt5 import QtWidgets, QtCore
import numpy as np

from .ExtensionManager import UIComponents
from spammm.topology.ascii_art_heterocycle import parse_ascii_art, run_kekule_solver, ASCII_EXAMPLES, mol_bond_types, export_mol, resolve_hbond_pairs, _build_target_valence, jacobi_relax_bond_lengths
from spammm.topology.KekulePure import make_n_pi


def build_ui(window):
    """Build Kekule heterocycle panel for the SPAMMM GUI.

    Returns ExtensionManager.UIComponents.
    """
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setSpacing(2)
    layout.setContentsMargins(2, 2, 2, 2)

    # --- Row 1: Example dropdown + Load button ---
    row1 = QtWidgets.QHBoxLayout()
    row1.addWidget(QtWidgets.QLabel("Example:"))
    window.kek_example_combo = QtWidgets.QComboBox()
    window.kek_example_combo.addItems(sorted(ASCII_EXAMPLES.keys()))
    window.kek_example_combo.setMaximumWidth(150)
    row1.addWidget(window.kek_example_combo)
    window.kek_load_example_btn = QtWidgets.QPushButton("Load")
    window.kek_load_example_btn.setMaximumWidth(60)
    window.kek_load_example_btn.clicked.connect(lambda: _on_load_example(window))
    row1.addWidget(window.kek_load_example_btn)
    row1.addStretch()
    layout.addLayout(row1)

    # --- Row 2: ASCII art text editor ---
    window.kek_ascii_edit = QtWidgets.QPlainTextEdit()
    window.kek_ascii_edit.setMaximumHeight(120)
    window.kek_ascii_edit.setPlaceholderText("Enter ASCII art here or load an example...")
    layout.addWidget(window.kek_ascii_edit)

    # --- Row 3: Generate button ---
    row3 = QtWidgets.QHBoxLayout()
    window.kek_generate_btn = QtWidgets.QPushButton("Generate")
    window.kek_generate_btn.setMaximumWidth(80)
    window.kek_generate_btn.clicked.connect(lambda: _on_generate(window))
    row3.addWidget(window.kek_generate_btn)
    window.kek_hydrogens_chk = QtWidgets.QCheckBox("Add H")
    window.kek_hydrogens_chk.setChecked(True)
    row3.addWidget(window.kek_hydrogens_chk)
    window.kek_relax_spin = QtWidgets.QSpinBox()
    window.kek_relax_spin.setRange(0, 100)
    window.kek_relax_spin.setValue(0)
    window.kek_relax_spin.setMaximumWidth(50)
    window.kek_relax_spin.setToolTip("Jacobi bond-length relaxation steps (0=off)")
    row3.addWidget(QtWidgets.QLabel("Relax:"))
    row3.addWidget(window.kek_relax_spin)
    row3.addStretch()
    layout.addLayout(row3)

    # --- Separator ---
    sep1 = QtWidgets.QFrame()
    sep1.setFrameShape(QtWidgets.QFrame.HLine)
    layout.addWidget(sep1)

    # --- Row 4: Kekule solver parameters ---
    row4 = QtWidgets.QHBoxLayout()
    row4.addWidget(QtWidgets.QLabel("Kval:"))
    window.kek_kval_spin = QtWidgets.QDoubleSpinBox()
    window.kek_kval_spin.setRange(0.1, 1000.0)
    window.kek_kval_spin.setValue(50.0)
    window.kek_kval_spin.setMaximumWidth(60)
    row4.addWidget(window.kek_kval_spin)
    row4.addWidget(QtWidgets.QLabel("Kloc:"))
    window.kek_kloc_spin = QtWidgets.QDoubleSpinBox()
    window.kek_kloc_spin.setRange(0.0, 100.0)
    window.kek_kloc_spin.setValue(5.0)
    window.kek_kloc_spin.setMaximumWidth(60)
    row4.addWidget(window.kek_kloc_spin)
    row4.addWidget(QtWidgets.QLabel("Karo:"))
    window.kek_karo_spin = QtWidgets.QDoubleSpinBox()
    window.kek_karo_spin.setRange(0.0, 10.0)
    window.kek_karo_spin.setSingleStep(0.1)
    window.kek_karo_spin.setValue(0.5)
    window.kek_karo_spin.setMaximumWidth(60)
    row4.addWidget(window.kek_karo_spin)
    row4.addStretch()
    layout.addLayout(row4)

    # --- Row 5: Aromatic + Solve ---
    row5 = QtWidgets.QHBoxLayout()
    window.kek_aromatic_chk = QtWidgets.QCheckBox("Aromatic")
    window.kek_aromatic_chk.setChecked(True)
    row5.addWidget(window.kek_aromatic_chk)
    window.kek_solve_btn = QtWidgets.QPushButton("Solve ASCII")
    window.kek_solve_btn.setMaximumWidth(90)
    window.kek_solve_btn.clicked.connect(lambda: _on_solve(window))
    row5.addWidget(window.kek_solve_btn)
    window.kek_solve_current_btn = QtWidgets.QPushButton("Solve Current")
    window.kek_solve_current_btn.setMaximumWidth(100)
    window.kek_solve_current_btn.clicked.connect(lambda: _on_solve_current(window))
    row5.addWidget(window.kek_solve_current_btn)
    row5.addStretch()
    layout.addLayout(row5)

    # --- Row 6: Export ---
    row6 = QtWidgets.QHBoxLayout()
    window.kek_export_xyz_btn = QtWidgets.QPushButton("Export XYZ")
    window.kek_export_xyz_btn.setMaximumWidth(90)
    window.kek_export_xyz_btn.clicked.connect(lambda: _on_export_xyz(window))
    row6.addWidget(window.kek_export_xyz_btn)
    window.kek_export_mol_btn = QtWidgets.QPushButton("Export MOL")
    window.kek_export_mol_btn.setMaximumWidth(90)
    window.kek_export_mol_btn.clicked.connect(lambda: _on_export_mol(window))
    row6.addWidget(window.kek_export_mol_btn)
    row6.addStretch()
    layout.addLayout(row6)

    # --- Status ---
    window.kek_status_label = QtWidgets.QLabel("Status: Ready")
    window.kek_status_label.setWordWrap(True)
    window.kek_status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(window.kek_status_label)

    # --- Bond order display ---
    window.kek_bond_order_label = QtWidgets.QLabel("Bond orders: ---")
    window.kek_bond_order_label.setWordWrap(True)
    layout.addWidget(window.kek_bond_order_label)

    layout.addStretch()

    # View modes: toggle bond order labels
    view_modes = [
        ('Bond Order Labels', lambda: _toggle_bond_order_labels(window)),
    ]

    return UIComponents(panel=panel, edit_modes=[], view_modes=view_modes)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _on_load_example(window):
    """Load an ASCII art example into the text editor."""
    name = window.kek_example_combo.currentText()
    art = ASCII_EXAMPLES.get(name, "")
    window.kek_ascii_edit.setPlainText(art.strip())
    window.kek_status_label.setText(f"Status: Loaded example '{name}'")


def _get_backend_sys(window):
    """Get AtomicSystem from the window's backend."""
    if hasattr(window, 'backend') and window.backend is not None:
        return window.backend.sys
    return None


def _on_generate(window):
    """Parse ASCII art and load the resulting AtomicSystem into the backend."""
    text = window.kek_ascii_edit.toPlainText().strip()
    if not text:
        window.kek_status_label.setText("Status: No ASCII art entered")
        return
    try:
        atoms = parse_ascii_art(text)
        atoms.neighs()
        n_pi0 = make_n_pi(atoms)
        if window.kek_hydrogens_chk.isChecked():
            tv = _build_target_valence(atoms, n_pi0)
            atoms.add_capping_h_sp2(target_valence=tv)
            enames_original = getattr(atoms, '_enames_original', None)
            if (enames_original is not None) and (len(enames_original) < len(atoms.apos)):
                atoms._enames_original = list(enames_original) + ['H'] * (len(atoms.apos) - len(enames_original))
            atoms.neighs()
        resolve_hbond_pairs(atoms)
        n_relax = window.kek_relax_spin.value()
        if n_relax > 0:
            jacobi_relax_bond_lengths(atoms, n_iters=n_relax)
        # Store on window for later use
        window.kek_atoms = atoms
        window.kek_bo_snap = None
        window.bond_orders = None
        window.bond_order_bonds = None
        # Load into backend if available
        if hasattr(window, 'backend') and window.backend is not None:
            window.backend.sys = atoms
            if hasattr(window, 'refresh_view'):
                window.refresh_view()
        n_bonds = len(atoms.bonds) if atoms.bonds is not None else 0
        window.kek_status_label.setText(f"Status: Generated {atoms.natoms} atoms, {n_bonds} bonds")
        window.kek_bond_order_label.setText("Bond orders: ---")
    except Exception as e:
        window.kek_status_label.setText(f"Status: Generate FAILED: {e}")
        import traceback; traceback.print_exc()


def _store_bond_orders_on_window(window, atoms, r):
    """Store solver results on window for rendering in refresh_view."""
    bo_snap = r['bo_snap']
    window.kek_bo_snap = bo_snap
    # Extract pi-active bonds (both atoms have n_pi > 0) and their bond orders
    bonds_all = np.asarray(atoms.bonds, dtype=np.int32)
    n_pi = r['n_pi']
    is_pi = np.asarray(n_pi) > 0
    pi_mask = is_pi[bonds_all[:, 0]] & is_pi[bonds_all[:, 1]] if len(bonds_all) else np.zeros(0, dtype=bool)
    window.bond_order_bonds = bonds_all[pi_mask] if len(pi_mask) else None
    window.bond_orders = bo_snap[pi_mask] if len(pi_mask) else None


def _on_solve(window):
    """Run the Kekule solver on the ASCII-generated system."""
    atoms = getattr(window, 'kek_atoms', None)
    if atoms is None:
        window.kek_status_label.setText("Status: Generate first")
        return
    try:
        Kval = window.kek_kval_spin.value()
        Kloc = window.kek_kloc_spin.value()
        Karo = window.kek_karo_spin.value()
        allow_aromatic = window.kek_aromatic_chk.isChecked()
        r = run_kekule_solver(atoms, Kval=Kval, Kloc=Kloc, Karo=Karo, allow_aromatic=allow_aromatic)
        if r['err'] is not None:
            print(f"[KEKULE] Solve ERROR: {r['err']}")
            tb = r['report'].get('traceback', '')
            if tb: print(tb)
            window.kek_status_label.setText(f"Status: Solve ERROR: {r['err']}")
            return
        _store_bond_orders_on_window(window, atoms, r)
        rep = r['report']
        bo_str = np.round(window.bond_orders, 2).tolist() if window.bond_orders is not None else []
        window.kek_bond_order_label.setText(f"Bond orders: {bo_str}")
        window.kek_status_label.setText(
            f"Status: Solved — single={rep['single']} aromatic={rep['aromatic']} double={rep['double']}"
        )
        if hasattr(window, 'refresh_view'):
            window.refresh_view()
    except Exception as e:
        window.kek_status_label.setText(f"Status: Solve FAILED: {e}")
        import traceback; traceback.print_exc()


def _on_solve_current(window):
    """Run the Kekule solver on the current backend system (drawn molecule)."""
    sys = _get_backend_sys(window)
    if sys is None or sys.apos is None or len(sys.apos) == 0:
        window.kek_status_label.setText("Status: No molecule loaded")
        return
    try:
        if sys.bonds is None:
            sys.findBonds()
        sys.neighs()
        Kval = window.kek_kval_spin.value()
        Kloc = window.kek_kloc_spin.value()
        Karo = window.kek_karo_spin.value()
        allow_aromatic = window.kek_aromatic_chk.isChecked()
        # Use n_pi from backend (user-set pi orbitals), not element case
        n_pi = np.asarray(window.backend.atom_npi, dtype=float) if hasattr(window.backend, 'atom_npi') else None
        r = run_kekule_solver(sys, Kval=Kval, Kloc=Kloc, Karo=Karo, allow_aromatic=allow_aromatic, n_pi=n_pi)
        if r['err'] is not None:
            print(f"[KEKULE] Solve ERROR: {r['err']}")
            tb = r['report'].get('traceback', '')
            if tb: print(tb)
            window.kek_status_label.setText(f"Status: Solve ERROR: {r['err']}")
            return
        _store_bond_orders_on_window(window, sys, r)
        window.kek_atoms = sys  # store for export
        rep = r['report']
        bo_str = np.round(window.bond_orders, 2).tolist() if window.bond_orders is not None else []
        window.kek_bond_order_label.setText(f"Bond orders: {bo_str}")
        window.kek_status_label.setText(
            f"Status: Solved — single={rep['single']} aromatic={rep['aromatic']} double={rep['double']}"
        )
        if hasattr(window, 'refresh_view'):
            window.refresh_view()
    except Exception as e:
        window.kek_status_label.setText(f"Status: Solve FAILED: {e}")
        import traceback; traceback.print_exc()


def _on_export_xyz(window):
    """Export the current system to XYZ file."""
    atoms = getattr(window, 'kek_atoms', None)
    if atoms is None:
        window.kek_status_label.setText("Status: Generate first")
        return
    fname, _ = QtWidgets.QFileDialog.getSaveFileName(window, "Save XYZ", "", "XYZ files (*.xyz)")
    if not fname:
        return
    try:
        atoms.saveXYZ(fname)
        window.kek_status_label.setText(f"Status: Saved XYZ: {fname}")
    except Exception as e:
        window.kek_status_label.setText(f"Status: Export XYZ FAILED: {e}")
        print(f"[KEKULE] Export XYZ FAILED: {e}")


def _on_export_mol(window):
    """Export the current system to MOL file."""
    atoms = getattr(window, 'kek_atoms', None)
    if atoms is None:
        window.kek_status_label.setText("Status: Generate first")
        return
    fname, _ = QtWidgets.QFileDialog.getSaveFileName(window, "Save MOL", "", "MOL files (*.mol)")
    if not fname:
        return
    try:
        bo_snap = getattr(window, 'kek_bo_snap', None)
        bt = mol_bond_types(atoms, bo_snap=bo_snap, allow_aromatic=window.kek_aromatic_chk.isChecked(), kekule=bo_snap is not None)
        atoms.save_mol(fname, bond_types=bt)
        window.kek_status_label.setText(f"Status: Saved MOL: {fname}")
    except Exception as e:
        window.kek_status_label.setText(f"Status: Export MOL FAILED: {e}")
        print(f"[KEKULE] Export MOL FAILED: {e}")


def _toggle_bond_order_labels(window):
    """Toggle bond order label display in the scene."""
    window.show_bond_order_labels = not window.show_bond_order_labels
    if hasattr(window, 'refresh_view'):
        window.refresh_view()
