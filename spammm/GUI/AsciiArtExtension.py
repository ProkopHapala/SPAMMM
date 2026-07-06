"""
AsciiArtExtension.py — GUI extension for ASCII art heterocycle generation.

Provides a panel to build molecular structures from ASCII art input.
Parse, add hydrogens, relax bond lengths, and load into the backend.

Wired through ExtensionManager as 'ascii' extension.
"""

from PyQt5 import QtWidgets, QtCore
import numpy as np

from .ExtensionManager import UIComponents
from spammm.topology.ascii_art_heterocycle import parse_ascii_art, ASCII_EXAMPLES, resolve_hbond_pairs, _build_target_valence, jacobi_relax_bond_lengths
from spammm.topology.KekulePure import make_n_pi


def build_ui(window):
    """Build ASCII art heterocycle panel for the SPAMMM GUI.

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

    # --- Status ---
    window.ascii_status_label = QtWidgets.QLabel("Status: Ready")
    window.ascii_status_label.setWordWrap(True)
    window.ascii_status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(window.ascii_status_label)

    layout.addStretch()

    return UIComponents(panel=panel, edit_modes=[], view_modes=[])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _on_load_example(window):
    """Load an ASCII art example into the text editor."""
    name = window.kek_example_combo.currentText()
    art = ASCII_EXAMPLES.get(name, "")
    window.kek_ascii_edit.setPlainText(art.strip())
    window.ascii_status_label.setText(f"Status: Loaded example '{name}'")


def _on_generate(window):
    """Parse ASCII art and load the resulting AtomicSystem into the backend."""
    text = window.kek_ascii_edit.toPlainText().strip()
    if not text:
        window.ascii_status_label.setText("Status: No ASCII art entered")
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
        # Store on window for later use by Kekule solver extension
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
        window.ascii_status_label.setText(f"Status: Generated {atoms.natoms} atoms, {n_bonds} bonds")
    except Exception as e:
        window.ascii_status_label.setText(f"Status: Generate FAILED: {e}")
        import traceback; traceback.print_exc()
