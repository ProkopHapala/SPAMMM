"""
FFExtension.py — GUI extension for forcefield-based relaxation and MD.

Purpose: Provide UI panel and edit/view modes for the relaxation extension.
Built on top of FFController (logic) and ExtensionManager.UIComponents (UI).

Key functionality:
  - build_ui(window) → UIComponents with panel + edit modes + view modes
  - Panel: FF type, Build, Relax N steps, Interactive toggle, Pin/Unpin buttons
  - Edit mode "Pin/Unpin": click atoms to toggle pin state
  - View mode "Show Forces": display force vectors on atoms
  - Interactive mode: QTimer-driven per-frame relaxation with live Vispy update

Role in SPAMMM: Wired through ExtensionManager as 'relax' extension.
The extension creates a FFController on the window and connects UI widgets to it.
GUI hooks (set_edit_mode, handle_click, on_key_press) will be added to SPAMMM_GUI later.
"""

from PyQt5 import QtWidgets, QtCore
import numpy as np

from ..forcefields.FFController import FFController, DEFAULT_DT, DEFAULT_DAMP, DEFAULT_FLIMIT
from .ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH, AutoGridPlacer


def build_ui(window):
    """Build forcefield panel for KekuleExplorerGUI.

    Returns ExtensionManager.UIComponents.
    """
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    # Create controller instance on the window
    if not hasattr(window, 'ff_controller'):
        window.ff_controller = FFController()

    # --- Row 1: FF type + Build ---
    g1 = AutoGridPlacer(cols=4)
    window.relax_ff_combo = QtWidgets.QComboBox()
    window.relax_ff_combo.addItems(["SPFF", "UFF"])
    window.relax_ff_combo.setMaximumWidth(60)
    g1.add_pair("FF:", window.relax_ff_combo)
    window.relax_build_btn = QtWidgets.QPushButton("Build")
    window.relax_build_btn.clicked.connect(lambda: _on_build_ff(window))
    g1.add(window.relax_build_btn)
    layout.addLayout(g1.layout())

    # --- Row 2: nSteps + Step button ---
    g2 = AutoGridPlacer(cols=4)
    window.relax_steps_spin = QtWidgets.QSpinBox()
    window.relax_steps_spin.setRange(1, 100000)
    window.relax_steps_spin.setValue(100)
    window.relax_steps_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g2.add_pair("nSteps:", window.relax_steps_spin)
    window.relax_step_btn = QtWidgets.QPushButton("Step")
    window.relax_step_btn.clicked.connect(lambda: _on_step(window))
    g2.add(window.relax_step_btn)
    layout.addLayout(g2.layout())

    # --- Row 3: dt + damp ---
    g3 = AutoGridPlacer(cols=4)
    window.relax_dt_spin = QtWidgets.QDoubleSpinBox()
    window.relax_dt_spin.setRange(0.001, 1.0)
    window.relax_dt_spin.setSingleStep(0.005)
    window.relax_dt_spin.setValue(DEFAULT_DT)
    window.relax_dt_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g3.add_pair("dt:", window.relax_dt_spin)
    window.relax_damp_spin = QtWidgets.QDoubleSpinBox()
    window.relax_damp_spin.setRange(0.0, 1.0)
    window.relax_damp_spin.setSingleStep(0.05)
    window.relax_damp_spin.setValue(DEFAULT_DAMP)
    window.relax_damp_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g3.add_pair("damp:", window.relax_damp_spin)
    layout.addLayout(g3.layout())

    # --- Row 4: Relax + Interactive + Serial checkbox ---
    g4 = AutoGridPlacer(cols=4)
    window.relax_run_btn = QtWidgets.QPushButton("Relax")
    window.relax_run_btn.clicked.connect(lambda: _on_relax(window))
    g4.add(window.relax_run_btn)
    from .ShortcutRegistry import encode_keystroke
    window.relax_interactive_btn = QtWidgets.QPushButton(f"Interactive [{encode_keystroke('Space')}]")
    window.relax_interactive_btn.setCheckable(True)
    window.relax_interactive_btn.clicked.connect(lambda checked: _on_interactive(window, checked))
    g4.add(window.relax_interactive_btn)
    window.relax_serial_chk = QtWidgets.QCheckBox("Serial")
    window.relax_serial_chk.setChecked(True)
    window.relax_serial_chk.setToolTip("Use single-kernel local-memory relaxation (150x faster for small molecules)")
    window.relax_serial_chk.toggled.connect(lambda checked: _on_serial_toggled(window, checked))
    g4.add(window.relax_serial_chk)
    window.relax_debug_chk = QtWidgets.QCheckBox("Debug FF")
    window.relax_debug_chk.setChecked(False)
    window.relax_debug_chk.setToolTip("Print all arrays sent to GPU after build (atom types, npi, neighbors, bond params, etc.)")
    g4.add(window.relax_debug_chk)
    layout.addLayout(g4.layout())

    # --- Energy + Fmax display ---
    window.relax_energy_label = QtWidgets.QLabel("E: ---  | Fmax: ---")
    window.relax_energy_label.setWordWrap(True)
    layout.addWidget(window.relax_energy_label)

    # --- Separator ---
    sep = QtWidgets.QFrame()
    sep.setFrameShape(QtWidgets.QFrame.HLine)
    layout.addWidget(sep)

    # --- Pin controls ---
    g_pin = AutoGridPlacer(cols=4)
    window.relax_pin_btn = QtWidgets.QPushButton("Pin Sel")
    window.relax_pin_btn.setEnabled(False)
    window.relax_pin_btn.clicked.connect(lambda: _on_pin_selected(window))
    g_pin.add(window.relax_pin_btn)
    window.relax_unpin_btn = QtWidgets.QPushButton("Unpin All")
    window.relax_unpin_btn.setEnabled(False)
    window.relax_unpin_btn.clicked.connect(lambda: _on_unpin_all(window))
    g_pin.add(window.relax_unpin_btn)
    layout.addLayout(g_pin.layout())

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

def _on_serial_toggled(window, checked):
    """Toggle serial kernel mode. Shows status; auto-fallback if molecule too large."""
    ctrl = window.ff_controller
    if checked and ctrl.is_built:
        if not ctrl._can_use_serial(ctrl.enable_nonbond):
            window.relax_serial_chk.setChecked(False)
            window.relax_status_label.setText("Status: Serial unavailable (too large or non-bonded) — using batch")
        else:
            window.relax_status_label.setText("Status: Serial kernel ON")
    elif not checked:
        window.relax_status_label.setText("Status: Batch kernel mode")

# npi → SPFF atom type name mapping (for aromatic GUI systems)
_NPI_TO_TYPE = {
    ('C', 0): 'C_3', ('C', 1): 'C_R', ('C', 2): 'C_1',
    ('N', 0): 'N_3', ('N', 1): 'N_R', ('N', 2): 'N_1',
    ('O', 0): 'O_3', ('O', 1): 'O_R', ('O', 2): 'O_1',
}

def _get_sys(window):
    """Get AtomicSystem from the window's backend, with synced atom types."""
    if hasattr(window, 'backend') and window.backend is not None:
        backend = window.backend
        # Sync sys from graph (rebuild arrays from authoritative graph)
        if hasattr(backend, '_sync_sys'):
            backend._sync_sys()
        sys = backend.sys
        # Set atom_types_spff from npi values if available
        if hasattr(backend, 'atom_npi') and sys is not None and len(sys.enames) > 0:
            npis = backend.atom_npi
            enames = sys.enames
            types = []
            for i, e in enumerate(enames):
                e = str(e)
                if e == 'H':
                    types.append('H')
                elif i < len(npis):
                    npi = int(npis[i])
                    types.append(_NPI_TO_TYPE.get((e, npi), e))
                else:
                    types.append(e)
            sys.atom_types_spff = types
        return sys
    return None


def _sync_positions_to_graph(window, positions):
    """Propagate relaxed positions back to the authoritative AtomicGraph."""
    if hasattr(window, 'backend') and window.backend is not None:
        backend = window.backend
        if hasattr(backend, 'graph') and hasattr(backend.graph, 'update_positions_from_array'):
            try:
                backend.graph.update_positions_from_array(positions)
                backend._sync_sys()  # rebuild sys from graph so everything is consistent
            except ValueError as e:
                # Atom count mismatch — topology changed, FF is stale
                pass


def _debug_print_ff(ctrl):
    """Print all arrays going to GPU for debugging."""
    spff = ctrl.spff
    md = ctrl.md
    print("=" * 60)
    print("DEBUG FF: Arrays sent to GPU")
    print("=" * 60)
    print(f"  natoms={spff.natoms}  nnode={spff.nnode}  ncap={spff.ncap}  nvecs={spff.nvecs}  ntors={spff.ntors}")
    print(f"  md.nSystems={md.nSystems}  md.nvecs={md.nvecs}  md.nnode={md.nnode}")
    print(f"  _can_use_serial={ctrl._can_use_serial(ctrl.enable_nonbond)}  enable_nonbond={ctrl.enable_nonbond}")
    # Atom type assignment
    sys = getattr(ctrl, '_last_sys', None)
    if sys is not None:
        print(f"  sys.enames:   {list(sys.enames)}")
        print(f"  sys.atypes:   {sys.atypes}")
        print(f"  atom_types_spff: {getattr(sys, 'atom_types_spff', None)}")
    print(f"  spff.npi_list: {spff.npi_list}")
    print(f"  spff.nep_list: {spff.nep_list}")
    print(f"  spff.isNode:   {getattr(spff, 'isNode', 'N/A')}")
    # Full arrays via SPFF.printArrays
    spff.printArrays()
    # Per-atom detail: neighbors, bond params, pi positions
    print("-" * 60)
    print("Per-atom detail (all atoms):")
    for ia in range(spff.natoms):
        pos = spff.apos[ia]
        ngh = spff.neighs[ia]
        atype = spff.atypes[ia]
        npi = spff.npi_list[ia] if ia < len(spff.npi_list) else '?'
        if ia < spff.nnode:
            ap = spff.apars[ia]
            bL = spff.bLs[ia]
            bK = spff.bKs[ia]
            Ksp = spff.Ksp[ia]
            Kpp = spff.Kpp[ia]
            pi = spff.pipos[ia]
            print(f"  [{ia:2d}] NODE type={atype} npi={npi} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) neighs={ngh}")
            print(f"       apars=({ap[0]:.4f},{ap[1]:.4f},{ap[2]:.4f},{ap[3]:.4f}) bLs={bL} bKs={bK} Ksp={Ksp} Kpp={Kpp}")
            print(f"       pipos=({pi[0]:.4f},{pi[1]:.4f},{pi[2]:.4f})")
        else:
            print(f"  [{ia:2d}] CAP  type={atype} npi={npi} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) neighs={ngh}")
    # Pi orbital positions in apos (indices natoms..natoms+nnode)
    print("-" * 60)
    print("Pi orbital vectors in apos[natoms:]:")
    for i in range(spff.nnode):
        idx = spff.natoms + i
        if idx < spff.nvecs:
            v = spff.apos[idx]
            print(f"  pi[{i}] (atom {i}): ({v[0]:.4f},{v[1]:.4f},{v[2]:.4f})")
    # Back neighbors
    if hasattr(spff, 'back_neighs') and spff.back_neighs is not None:
        print("-" * 60)
        print("Back neighbors (all):")
        for i in range(len(spff.back_neighs)):
            if np.any(spff.back_neighs[i] >= 0):
                print(f"  [{i:2d}] {spff.back_neighs[i]}")
    print("=" * 60)


def _on_build_ff(window):
    """Build forcefield from current molecular system."""
    ctrl = window.ff_controller
    sys = _get_sys(window)
    if sys is None or sys.apos is None or len(sys.apos) == 0:
        window.relax_status_label.setText("Status: No molecule loaded")
        return
    ff_type = window.relax_ff_combo.currentText().lower()
    try:
        import time
        window.relax_status_label.setText("Building FF (compiling kernels)...")
        QtWidgets.QApplication.processEvents()
        t0 = time.time()
        info = ctrl.build_ff(sys, ff_type=ff_type)
        t_build = time.time() - t0
        ctrl._last_sys = sys  # store for debug printing
        # Check serial availability
        can_serial = ctrl._can_use_serial(ctrl.enable_nonbond)
        if not can_serial:
            window.relax_serial_chk.setChecked(False)
            window.relax_status_label.setText(f"Status: Built {ff_type.upper()} ({info['natoms']} atoms) — batch mode (too large for serial)")
        else:
            window.relax_serial_chk.setChecked(True)
            window.relax_status_label.setText(f"Status: Built {ff_type.upper()} ({info['natoms']} atoms) — serial mode")
        print(f"[FF] Build took {t_build:.2f}s | natoms={info['natoms']} nnode={info['nnode']} nvecs={info['nvecs']} | serial={can_serial}")
        if window.relax_debug_chk.isChecked():
            _debug_print_ff(ctrl)
        window.relax_pin_btn.setEnabled(True)
        window.relax_unpin_btn.setEnabled(True)
    except Exception as e:
        window.relax_status_label.setText(f"Status: Build FAILED: {e}")
        import traceback; traceback.print_exc()


def _ensure_built(window):
    """Auto-build FF if not already built, or rebuild if topology changed.
    Returns True if built or already built."""
    ctrl = window.ff_controller
    sys = _get_sys(window)
    if sys is None or sys.apos is None or len(sys.apos) == 0:
        window.relax_status_label.setText("Status: No molecule loaded")
        return False
    # Check if FF is stale (atom count mismatch → topology changed)
    if ctrl.is_built and hasattr(ctrl, 'natoms') and ctrl.natoms != len(sys.apos):
        print(f"[FF] Topology changed: natoms {ctrl.natoms} → {len(sys.apos)}, rebuilding FF")
        ctrl.teardown()
    if ctrl.is_built:
        # Sync current positions from authoritative AtomGraph → GPU
        # (user may have dragged atoms since last build/relax)
        ctrl.update_positions(sys.apos[:, :3])
        return True
    ff_type = window.relax_ff_combo.currentText().lower()
    try:
        import time
        window.relax_status_label.setText("Building FF (compiling kernels)...")
        QtWidgets.QApplication.processEvents()
        t0 = time.time()
        info = ctrl.build_ff(sys, ff_type=ff_type)
        t_build = time.time() - t0
        ctrl._last_sys = sys
        can_serial = ctrl._can_use_serial(ctrl.enable_nonbond)
        if not can_serial:
            window.relax_serial_chk.setChecked(False)
        else:
            window.relax_serial_chk.setChecked(True)
        print(f"[FF] Build took {t_build:.2f}s | natoms={info['natoms']} nnode={info['nnode']} nvecs={info['nvecs']} | serial={can_serial}")
        if window.relax_debug_chk.isChecked():
            _debug_print_ff(ctrl)
        window.relax_pin_btn.setEnabled(True)
        window.relax_unpin_btn.setEnabled(True)
        return True
    except Exception as e:
        window.relax_status_label.setText(f"Status: Build FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


def _on_step(window):
    """Run nSteps relaxation steps and update view."""
    if not _ensure_built(window):
        return
    ctrl = window.ff_controller
    nsteps = window.relax_steps_spin.value()
    dt = window.relax_dt_spin.value()
    damp = window.relax_damp_spin.value()
    try:
        import time
        t0 = time.time()
        E = ctrl.relax_n(nsteps=nsteps, dt=dt, damp=damp)
        t_step = time.time() - t0
        fmax = ctrl.get_fmax()
        state = ctrl.get_state()
        # Propagate relaxed positions back to authoritative AtomicGraph
        _sync_positions_to_graph(window, state['positions'])
        print(f"[FF] Step took {t_step:.3f}s ({nsteps} steps)")
        window.relax_energy_label.setText(f"E: {E:.6f}  | Fmax: {fmax:.4f}")
        window.relax_status_label.setText(f"Status: Stepped ({nsteps} steps)")
        if hasattr(window, 'refresh_view'):
            window.refresh_view()
    except Exception as e:
        window.relax_status_label.setText(f"Status: Step FAILED: {e}")
        import traceback; traceback.print_exc()


def _on_relax(window):
    """Run relaxation until convergence, updating view every nSteps."""
    if not _ensure_built(window):
        return
    ctrl = window.ff_controller
    dt = window.relax_dt_spin.value()
    damp = window.relax_damp_spin.value()
    nsteps = window.relax_steps_spin.value()
    max_steps = nsteps * 100  # safeguard: allow up to 100 batches
    try:
        can_serial = ctrl._can_use_serial(ctrl.enable_nonbond)
        use_serial = window.relax_serial_chk.isChecked() and can_serial
        mode_str = "serial" if use_serial else "batch"
        if not use_serial and window.relax_serial_chk.isChecked():
            print(f"[FF] Serial checkbox checked but _can_use_serial=False (nvecs={ctrl.md.nvecs}, nnode={ctrl.md.nnode}, nSystems={ctrl.md.nSystems}, nonbond={ctrl.enable_nonbond})")
        window.relax_status_label.setText(f"Relaxing ({mode_str}) to convergence...")
        QtWidgets.QApplication.processEvents()
        import time
        t0 = time.time()
        def _cb(step, E, fmax):
            window.relax_energy_label.setText(f"E: {E:.6f}  | Fmax: {fmax:.4f}")
            window.relax_status_label.setText(f"Relaxing ({mode_str})... step={step} fmax={fmax:.4f}")
            QtWidgets.QApplication.processEvents()
            # Update view every batch
            state = ctrl.get_state()
            _sync_positions_to_graph(window, state['positions'])
            if hasattr(window, 'refresh_view'):
                window.refresh_view()
            return True
        result = ctrl.relax_until_converged(max_steps=max_steps, dt=dt, damp=damp, callback=_cb, batch_size=nsteps)
        t_relax = time.time() - t0
        state = ctrl.get_state()
        _sync_positions_to_graph(window, state['positions'])
        print(f"[FF] Relax took {t_relax:.2f}s ({mode_str}, {result['nsteps']} steps)")
        if result['converged']:
            window.relax_status_label.setText(f"Status: Converged ({result['nsteps']} steps, fmax={result['fmax']:.4f})")
        else:
            window.relax_status_label.setText(f"Status: Max steps ({result['nsteps']}, fmax={result['fmax']:.4f})")
        window.relax_energy_label.setText(f"E: {result['energy']:.6f}  | Fmax: {result['fmax']:.4f}")
        if hasattr(window, 'refresh_view'):
            window.refresh_view()
    except Exception as e:
        window.relax_status_label.setText(f"Status: Relax FAILED: {e}")
        import traceback; traceback.print_exc()


def _on_interactive(window, checked):
    """Start/stop interactive relaxation timer."""
    if checked and not _ensure_built(window):
        window.relax_interactive_btn.setChecked(False)
        return
    ctrl = window.ff_controller
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
        window.relax_interactive_btn.setText(f"Stop Interactive [{encode_keystroke('Space')}]")
    else:
        if hasattr(window, '_relax_timer'):
            window._relax_timer.stop()
        window.relax_status_label.setText("Status: Interactive OFF")
        window.relax_interactive_btn.setText(f"Interactive [{encode_keystroke('Space')}]")


def _interactive_tick(window):
    """Interactive relaxation: run nSteps per tick + view update."""
    ctrl = window.ff_controller
    if not ctrl.is_built:
        return
    try:
        nsteps = window.relax_steps_spin.value()
        dt = window.relax_dt_spin.value()
        damp = window.relax_damp_spin.value()
        if window.relax_serial_chk.isChecked() and ctrl._can_use_serial(ctrl.enable_nonbond):
            ctrl.md.relax_serial(nsteps=nsteps, dt=dt, damp=damp, Flimit=DEFAULT_FLIMIT)
        else:
            ctrl.md.set_md_params(dt=dt, damp=damp, Flimit=DEFAULT_FLIMIT)
            ctrl.md.relax_batch(nsteps=nsteps, do_nb=ctrl.enable_nonbond)
        pos = ctrl.get_positions()
        # Update scene positions directly for fast feedback
        if hasattr(window, 'scene'):
            window.scene.update_positions(pos.astype(np.float32))
        # Propagate to authoritative AtomicGraph
        _sync_positions_to_graph(window, pos)
        # Update energy display (less frequently to avoid flicker)
        if hasattr(window, '_relax_tick_count'):
            window._relax_tick_count += 1
        else:
            window._relax_tick_count = 1
        if window._relax_tick_count % 10 == 0:
            E = ctrl.get_energy()
            fmax = ctrl.get_fmax()
            window.relax_energy_label.setText(f"E: {E:.6f}  | Fmax: {fmax:.4f}")
    except Exception as e:
        # Stop timer on error to prevent spam
        if hasattr(window, '_relax_timer'):
            window._relax_timer.stop()
        window.relax_status_label.setText(f"Status: Interactive ERROR: {e}")
        window.relax_interactive_btn.setChecked(False)
        window.relax_interactive_btn.setText(f"Interactive [{encode_keystroke('Space')}]")
        import traceback; traceback.print_exc()


def _on_pin_selected(window):
    """Pin all selected atoms to their current positions."""
    ctrl = window.ff_controller
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
    ctrl = window.ff_controller
    if not ctrl.is_built:
        return
    if hasattr(window, 'backend'):
        window.backend.clear_constraints()
        mask = window.backend.constraint_mask()
    else:
        mask = np.zeros(ctrl.natoms, dtype=bool)
    ctrl.set_pinned(mask, window.scene._pos.copy() if hasattr(window, 'scene') else None)
    if hasattr(window, 'scene'):
        window.scene.set_fixed_mask(mask)
    window.relax_status_label.setText("Status: All pins cleared")


def _set_edit_mode_pin(window):
    """Set edit mode to pin/unpin — GUI will hook this into set_edit_mode later."""
    if hasattr(window, 'set_edit_mode'):
        window.set_edit_mode('pin_unpin')
    window.relax_status_label.setText("Status: Pin/Unpin mode — click atoms to toggle")


def _toggle_show_forces(window):
    """Toggle force vector display in the scene."""
    ctrl = window.ff_controller
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
    ctrl = window.ff_controller
    if not ctrl.is_built:
        return
    if hasattr(window, 'backend'):
        pinned = window.backend.toggle_constraint_by_index(atom_idx)
        mask = window.backend.constraint_mask()
        ctrl.set_pinned(mask, window.scene._pos.copy() if hasattr(window, 'scene') else None)
    else:
        pinned = ctrl.toggle_pin(atom_idx)
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
