"""
DFTBExtension.py — GUI extension for DFTB+ geometry relaxation.

Provides UI panel for running DFTB+ relaxation on the current molecule.

Wired through ExtensionManager as 'dftb' extension.
"""

from PyQt5 import QtWidgets

from .ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH


def build_ui(window):
    """Build DFTB+ panel for the SPAMMM GUI.

    Returns ExtensionManager.UIComponents.
    """
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    relax_btn = QtWidgets.QPushButton("Relax (DFTB+)")
    relax_btn.clicked.connect(lambda: _on_relax(window))
    layout.addWidget(relax_btn)

    window.dftb_status_label = QtWidgets.QLabel("Status: Ready")
    window.dftb_status_label.setWordWrap(True)
    layout.addWidget(window.dftb_status_label)

    return UIComponents(panel=panel)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _on_relax(window):
    """Run DFTB+ geometry relaxation on the current backend system."""
    window.dftb_status_label.setText("Status: Relaxing...")
    window.statusBar().showMessage("Relaxing... please wait")
    QtWidgets.QApplication.processEvents()

    try:
        E, forces, lvs = window.backend.run_relaxation(workdir='debug/gui_relax')
        msg = f"Relaxation done. E = {E:.4f} eV"
        window.statusBar().showMessage(msg)
        window.dftb_status_label.setText(f"Status: Done\nE = {E:.4f} eV")
        window.refresh_view()
    except Exception as e:
        msg = f"Relaxation FAILED: {e}"
        window.statusBar().showMessage(msg)
        window.dftb_status_label.setText(f"Status: FAILED\n{e}")
        window._raise(msg, title="Relaxation Error")
