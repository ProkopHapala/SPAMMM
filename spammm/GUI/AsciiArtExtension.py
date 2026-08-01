"""AsciiArtExtension — GUI extension for ASCII art heterocycle generation."""
from PyQt5 import QtWidgets, QtCore
import numpy as np

from .ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH, GridPlacer
from spammm.topology.ascii_art_heterocycle import parse_ascii_art, ASCII_EXAMPLES, resolve_hbond_pairs, _build_target_valence, jacobi_relax_bond_lengths
from spammm.topology.KekulePure import make_n_pi


def build_ui(window):
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    g1 = GridPlacer(cols=6)
    window.kek_example_combo = QtWidgets.QComboBox()
    window.kek_example_combo.addItems(sorted(ASCII_EXAMPLES.keys()))
    window.kek_example_combo.setMaximumWidth(150)
    g1.add_pair("Example:", window.kek_example_combo, label_span=1, input_span=3)
    window.kek_load_example_btn = QtWidgets.QPushButton("Load")
    window.kek_load_example_btn.clicked.connect(lambda: load_ascii_example(window))
    g1.add(window.kek_load_example_btn, span=2)
    layout.addLayout(g1.layout())

    window.kek_ascii_edit = QtWidgets.QPlainTextEdit()
    window.kek_ascii_edit.setMaximumHeight(120)
    window.kek_ascii_edit.setPlaceholderText("Enter ASCII art here or load an example...")
    layout.addWidget(window.kek_ascii_edit)

    g3 = GridPlacer(cols=6)
    window.kek_generate_btn = QtWidgets.QPushButton("Generate")
    window.kek_generate_btn.clicked.connect(lambda: generate_ascii_molecule(window))
    g3.add(window.kek_generate_btn, span=2)
    window.kek_hydrogens_chk = QtWidgets.QCheckBox("Add H")
    window.kek_hydrogens_chk.setChecked(True)
    g3.add(window.kek_hydrogens_chk, span=1)
    window.kek_relax_spin = QtWidgets.QSpinBox()
    window.kek_relax_spin.setRange(0, 100)
    window.kek_relax_spin.setValue(0)
    window.kek_relax_spin.setMaximumWidth(50)
    window.kek_relax_spin.setToolTip("Jacobi bond-length relaxation steps (0=off)")
    g3.add_pair("Relax:", window.kek_relax_spin, label_span=1, input_span=2)
    layout.addLayout(g3.layout())

    window.ascii_status_label = QtWidgets.QLabel("Status: Ready")
    window.ascii_status_label.setWordWrap(True)
    window.ascii_status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(window.ascii_status_label)
    layout.addStretch()
    return UIComponents(panel=panel, edit_modes=[], view_modes=[])


def load_ascii_example(window, name=None):
    """Load ASCII example into editor (same as Load button)."""
    name = name or window.kek_example_combo.currentText()
    idx = window.kek_example_combo.findText(name)
    if idx >= 0:
        window.kek_example_combo.setCurrentIndex(idx)
    art = ASCII_EXAMPLES.get(name, "")
    window.kek_ascii_edit.setPlainText(art)
    window.ascii_status_label.setText(f"Status: Loaded example '{name}'")


def generate_ascii_molecule(window):
    """Parse ASCII art and load into backend (same as Generate button)."""
    text = window.kek_ascii_edit.toPlainText().strip()
    if not text:
        window.ascii_status_label.setText("Status: No ASCII art entered")
        return False
    try:
        atoms = parse_ascii_art(text, hbond_length=3.0)
        atoms.neighs()
        n_pi0 = make_n_pi(atoms)
        if window.kek_hydrogens_chk.isChecked():
            tv = _build_target_valence(atoms, n_pi0)
            atoms.add_capping_h_sp2(target_valence=tv)
            enames_original = getattr(atoms, '_enames_original', None)
            if (enames_original is not None) and (len(enames_original) < len(atoms.apos)):
                atoms._enames_original = list(enames_original) + ['H'] * (len(atoms.apos) - len(enames_original))
            atoms.neighs()
        n_relax = window.kek_relax_spin.value()
        if n_relax > 0:
            jacobi_relax_bond_lengths(atoms, n_iters=n_relax, bmix=0.3)
        resolve_hbond_pairs(atoms)
        window.kek_atoms = atoms
        window.kek_bo_snap = None
        window.bond_orders = None
        window.bond_order_bonds = None
        if hasattr(window, 'backend') and window.backend is not None:
            window.backend.sys = atoms
            if hasattr(window, 'refresh_view'):
                window.refresh_view()
        n_bonds = len(atoms.bonds) if atoms.bonds is not None else 0
        window.ascii_status_label.setText(f"Status: Generated {atoms.natoms} atoms, {n_bonds} bonds")
        return True
    except Exception as e:
        window.ascii_status_label.setText(f"Status: Generate FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
