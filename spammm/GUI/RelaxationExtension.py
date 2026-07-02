"""
RelaxationExtension.py — GUI extension for molecular relaxation with forcefields.

Purpose: Provide UI panel and edit/view modes for the relaxation extension.
Built on top of RelaxationController (logic) and ExtensionManager.UIComponents (UI).

Key functionality:
  - build_ui(window) → UIComponents with panel + edit modes + view modes
  - Panel: FF type, Build, Relax N steps, Interactive toggle, Pin/Unpin buttons
  - Edit mode "Pin/Unpin": click atoms to toggle pin state
  - View mode "Show Forces": display force vectors on atoms
  - Interactive mode: QTimer-driven per-frame relaxation with live Vispy update

Role in SPAMMM: Wired through ExtensionManager as 'relax' extension.
The extension creates a RelaxationController on the window and connects UI widgets to it.
GUI hooks (set_edit_mode, handle_click, on_key_press) will be added to SPAMMM_GUI later.
"""

from PyQt5 import QtWidgets, QtCore
import numpy as np

from ..forcefields.RelaxationController import RelaxationController, DEFAULT_DT, DEFAULT_DAMP, DEFAULT_FLIMIT
from .ExtensionManager import UIComponents


def build_ui(window):
    """Build relaxation panel for KekuleExplorerGUI.

    Returns ExtensionManager.UIComponents.
    """
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setSpacing(3)
    layout.setContentsMargins(2, 2, 2, 2)

    # Create controller instance on the window
    if not hasattr(window, 'relax_controller'):
        window.relax_controller = RelaxationController()
    ctrl = window.relax_controller

    # --- FF type selector ---
    row_ff = QtWidgets.QHBoxLayout()
    row_ff.addWidget(QtWidgets.QLabel("FF:"))
    window.relax_ff_combo = QtWidgets.QComboBox()
    window.relax_ff_combo.addItems(["SPFF", "UFF"])
    row_ff.addWidget(window.relax_ff_combo)
    layout.addLayout(row_ff)

    # --- Build FF button ---
    window.relax_build_btn = QtWidgets.QPushButton("Build FF")
    window.relax_build_btn.clicked.connect(lambda: _on_build_ff(window))
    layout.addWidget(window.relax_build_btn)

    # --- Relaxation parameters ---
    row_params = QtWidgets.QHBoxLayout()
    row_params.addWidget(QtWidgets.QLabel("Steps:"))
    window.relax_steps_spin = QtWidgets.QSpinBox()
    window.relax_steps_spin.setRange(1, 100000)
    window.relax_steps_spin.setValue(100)
    row_params.addWidget(window.relax_steps_spin)
    row_params.addWidget(QtWidgets.QLabel("dt:"))
    window.relax_dt_spin = QtWidgets.QDoubleSpinBox()
    window.relax_dt_spin.setRange(0.001, 1.0)
    window.relax_dt_spin.setSingleStep(0.005)
    window.relax_dt_spin.setValue(DEFAULT_DT)
    row_params.addWidget(window.relax_dt_spin)
    row_params.addWidget(QtWidgets.QLabel("damp:"))
    window.relax_damp_spin = QtWidgets.QDoubleSpinBox()
    window.relax_damp_spin.setRange(0.0, 1.0)
    window.relax_damp_spin.setSingleStep(0.05)
    window.relax_damp_spin.setValue(DEFAULT_DAMP)
    row_params.addWidget(window.relax_damp_spin)
    layout.addLayout(row_params)

    # --- Relax button ---
    window.relax_run_btn = QtWidgets.QPushButton("Relax")
    window.relax_run_btn.setEnabled(False)
    window.relax_run_btn.clicked.connect(lambda: _on_relax(window))
    layout.addWidget(window.relax_run_btn)

    # --- Interactive mode ---
    window.relax_interactive_btn = QtWidgets.QPushButton("Interactive")
    window.relax_interactive_btn.setCheckable(True)
    window.relax_interactive_btn.setEnabled(False)
    window.relax_interactive_btn.clicked.connect(lambda checked: _on_interactive(window, checked))
    layout.addWidget(window.relax_interactive_btn)

    # --- Energy display ---
    window.relax_energy_label = QtWidgets.QLabel("Energy: ---")
    window.relax_energy_label.setWordWrap(True)
    layout.addWidget(window.relax_energy_label)

    # --- Separator ---
    sep = QtWidgets.QFrame()
    sep.setFrameShape(QtWidgets.QFrame.HLine)
    layout.addWidget(sep)

    # --- Pin controls ---
    row_pin = QtWidgets.QHBoxLayout()
    window.relax_pin_btn = QtWidgets.QPushButton("Pin Selected")
    window.relax_pin_btn.setEnabled(False)
    window.relax_pin_btn.clicked.connect(lambda: _on_pin_selected(window))
    row_pin.addWidget(window.relax_pin_btn)
    window.relax_unpin_btn = QtWidgets.QPushButton("Unpin All")
    window.relax_unpin_btn.setEnabled(False)
    window.relax_unpin_btn.clicked.connect(lambda: _on_unpin_all(window))
    row_pin.addWidget(window.relax_unpin_btn)
    layout.addLayout(row_pin)

    # --- Status ---
    window.relax_status_label = QtWidgets.QLabel("Status: Not built")
    window.relax_status_label.setWordWrap(True)
    layout.addWidget(window.relax_status_label)

    layout.addStretch()

    # Edit modes: Pin/Unpin lets user click atoms to toggle pin
    edit_modes = [
        ('Pin/Unpin', lambda: _set_edit_mode_pin(window)),
    ]

    # View modes: Show forces toggles force vector display
    view_modes = [
        ('Show Forces', lambda: _toggle_show_forces(window)),
    ]

    return UIComponents(panel=panel, edit_modes=edit_modes, view_modes=view_modes)


# ---------------------------------------------------------------------------
# Callbacks (called from panel buttons — no direct GUI modification beyond panel)
# ---------------------------------------------------------------------------

def _get_sys(window):
    """Get AtomicSystem from the window's backend."""
    if hasattr(window, 'backend') and window.backend is not None:
        return window.backend.sys
    return None


def _on_build_ff(window):
    """Build forcefield from current molecular system."""
    ctrl = window.relax_controller
    sys = _get_sys(window)
    if sys is None or sys.apos is None or len(sys.apos) == 0:
        window.relax_status_label.setText("Status: No molecule loaded")
        return
    ff_type = window.relax_ff_combo.currentText().lower()
    try:
        window.relax_status_label.setText("Building FF...")
        QtWidgets.QApplication.processEvents()
        info = ctrl.build_ff(sys, ff_type=ff_type)
        window.relax_status_label.setText(f"Status: Built {ff_type.upper()} ({info['natoms']} atoms)")
        window.relax_run_btn.setEnabled(True)
        window.relax_interactive_btn.setEnabled(True)
        window.relax_pin_btn.setEnabled(True)
        window.relax_unpin_btn.setEnabled(True)
    except Exception as e:
        window.relax_status_label.setText(f"Status: Build FAILED: {e}")
        import traceback; traceback.print_exc()


def _on_relax(window):
    """Run batch relaxation and update view."""
    ctrl = window.relax_controller
    if not ctrl.is_built:
        return
    nsteps = window.relax_steps_spin.value()
    dt = window.relax_dt_spin.value()
    damp = window.relax_damp_spin.value()
    try:
        window.relax_status_label.setText(f"Relaxing {nsteps} steps...")
        QtWidgets.QApplication.processEvents()
        E = ctrl.relax_n(nsteps=nsteps, dt=dt, damp=damp)
        state = ctrl.get_state()
        # Update backend positions
        sys = _get_sys(window)
        if sys is not None:
            sys.apos[:, :3] = state['positions']
        window.relax_energy_label.setText(f"Energy: {E:.6f}")
        window.relax_status_label.setText(f"Status: Relaxed ({nsteps} steps)")
        # Request view refresh — GUI will connect this later
        if hasattr(window, 'refresh_view'):
            window.refresh_view()
    except Exception as e:
        window.relax_status_label.setText(f"Status: Relax FAILED: {e}")
        import traceback; traceback.print_exc()


def _on_interactive(window, checked):
    """Start/stop interactive relaxation timer."""
    ctrl = window.relax_controller
    if not ctrl.is_built:
        return
    if checked:
        if not hasattr(window, '_relax_timer'):
            window._relax_timer = QtCore.QTimer(window)
            window._relax_timer.timeout.connect(lambda: _interactive_tick(window))
        dt = window.relax_dt_spin.value()
        damp = window.relax_damp_spin.value()
        ctrl.md.set_md_params(dt=dt, damp=damp, Flimit=DEFAULT_FLIMIT)
        window._relax_timer.start(30)  # ~30ms = ~33fps
        window.relax_status_label.setText("Status: Interactive ON")
        window.relax_interactive_btn.setText("Stop Interactive")
    else:
        if hasattr(window, '_relax_timer'):
            window._relax_timer.stop()
        window.relax_status_label.setText("Status: Interactive OFF")
        window.relax_interactive_btn.setText("Interactive")


def _interactive_tick(window):
    """Single interactive relaxation step + view update."""
    ctrl = window.relax_controller
    if not ctrl.is_built:
        return
    try:
        ctrl.relax_step()
        pos = ctrl.get_positions()
        # Update scene positions directly for fast feedback
        if hasattr(window, 'scene'):
            window.scene.update_positions(pos.astype(np.float32))
        # Update backend positions
        sys = _get_sys(window)
        if sys is not None:
            sys.apos[:, :3] = pos
        # Update energy display (less frequently to avoid flicker)
        if hasattr(window, '_relax_tick_count'):
            window._relax_tick_count += 1
        else:
            window._relax_tick_count = 1
        if window._relax_tick_count % 10 == 0:
            E = ctrl.get_energy()
            window.relax_energy_label.setText(f"Energy: {E:.6f}")
    except Exception as e:
        # Stop timer on error to prevent spam
        if hasattr(window, '_relax_timer'):
            window._relax_timer.stop()
        window.relax_status_label.setText(f"Status: Interactive ERROR: {e}")
        window.relax_interactive_btn.setChecked(False)
        window.relax_interactive_btn.setText("Interactive")
        import traceback; traceback.print_exc()


def _on_pin_selected(window):
    """Pin all selected atoms to their current positions."""
    ctrl = window.relax_controller
    if not ctrl.is_built:
        return
    # Get selected atom IDs from scene, convert to indices
    if hasattr(window, 'scene') and hasattr(window.scene, '_selected_ids'):
        selected_ids = window.scene._selected_ids
        if not selected_ids:
            window.relax_status_label.setText("Status: No atoms selected")
            return
        # Convert Atom._id to array index
        id_to_idx = window.scene._id_to_idx
        indices = [id_to_idx[aid] for aid in selected_ids if aid in id_to_idx]
        if not indices:
            window.relax_status_label.setText("Status: Selected atoms not found in scene")
            return
        ctrl.pin_selected(indices)
        # Update scene fixed mask
        mask = ctrl.get_pinned_mask()
        window.scene.set_fixed_mask(mask)
        n = len(indices)
        window.relax_status_label.setText(f"Status: Pinned {n} atoms")
    else:
        window.relax_status_label.setText("Status: No selection available")


def _on_unpin_all(window):
    """Remove all pin constraints."""
    ctrl = window.relax_controller
    if not ctrl.is_built:
        return
    ctrl.clear_pins()
    # Clear scene fixed mask
    if hasattr(window, 'scene'):
        mask = np.zeros(ctrl.natoms, dtype=bool)
        window.scene.set_fixed_mask(mask)
    window.relax_status_label.setText("Status: All pins cleared")


def _set_edit_mode_pin(window):
    """Set edit mode to pin/unpin — GUI will hook this into set_edit_mode later."""
    if hasattr(window, 'set_edit_mode'):
        window.set_edit_mode('pin_unpin')
    window.relax_status_label.setText("Status: Pin/Unpin mode — click atoms to toggle")


def _toggle_show_forces(window):
    """Toggle force vector display in the scene."""
    ctrl = window.relax_controller
    if not ctrl.is_built:
        return
    if not hasattr(window, '_relax_show_forces'):
        window._relax_show_forces = False
    window._relax_show_forces = not window._relax_show_forces
    if window._relax_show_forces:
        forces = ctrl.get_forces()
        pos = ctrl.get_positions()
        if hasattr(window, 'scene'):
            window.scene.set_data(pos.astype(np.float32), forces=forces[:, :3], force_scale=0.1)
    else:
        if hasattr(window, 'scene'):
            window.scene.set_data(window.scene._pos)


# ---------------------------------------------------------------------------
# Hooks for GUI integration (to be called from SPAMMM_GUI later)
# ---------------------------------------------------------------------------

def handle_pin_click(window, atom_idx):
    """Toggle pin on an atom by index. Called from GUI handle_click in pin_unpin mode.

    Args:
        window: KekuleExplorerWindow
        atom_idx: array index of clicked atom
    """
    ctrl = window.relax_controller
    if not ctrl.is_built:
        return
    pinned = ctrl.toggle_pin(atom_idx)
    # Update scene fixed mask
    mask = ctrl.get_pinned_mask()
    if hasattr(window, 'scene'):
        window.scene.set_fixed_mask(mask)
    state = "pinned" if pinned else "unpinned"
    window.relax_status_label.setText(f"Status: Atom {atom_idx} {state}")


def handle_key_pin(window, atom_idx):
    """Toggle pin on hovered atom via keyboard shortcut 'P'.

    Args:
        window: KekuleExplorerWindow
        atom_idx: array index of hovered atom, or -1 if none
    """
    if atom_idx < 0:
        return
    handle_pin_click(window, atom_idx)
