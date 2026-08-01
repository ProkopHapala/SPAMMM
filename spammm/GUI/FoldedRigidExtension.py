"""FoldedRigidExtension.py — GUI extension for folded-basis rigid-body surface simulation.

Provides a panel to:
  - fit folded-basis coefficients for the current editor molecule on a substrate,
  - set up a rigid-body dynamics object,
  - relax the molecule on the surface,
  - run lateral and relaxed (pinned-atom) scans,
  - toggle a substrate overlay in the main scene.

It reuses the high-level workflow functions in spammm.surfaces.FoldedRigid and the
RigidBodyDynamics / SPFF_cl forcefield backend.
"""

from PyQt5 import QtWidgets, QtCore
import numpy as np
import hashlib
import os
import tempfile
import traceback

from vispy import scene as vscene

from .ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH, AutoGridPlacer
from .EditModeHandlers import EditModeHandler
from .CollapsibleSection import CollapsibleSection
from .VispyUtils import make_grid_mesh_data, colormap_rgba, update_faf_map_overlay
from spammm.surfaces import FoldedRigid
from spammm.topology.FFparams import load_xyz_with_REQs


# Default substrate shipped with the project
DEFAULT_SUBSTRATE = FoldedRigid.NACL_SUBSTRATE

# Color/size for substrate overlay (same convention as surface_plots)
# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _status(window, msg):
    window.fr_status_label.setText(msg)
    QtWidgets.QApplication.processEvents()


def _fit_cache_dir():
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'debug', 'folded_rigid', 'fits')
    os.makedirs(d, exist_ok=True)
    return d


def _fit_cache_key(sys, sub_file, nu, nv, alpha, z_min, z_max):
    """Stable hash key for a fit based on geometry, substrate and fit settings."""
    apos = np.asarray(sys.apos, dtype=np.float64)
    apos = np.round(apos - apos.mean(axis=0), 6)
    qs = np.asarray(sys.qs, dtype=np.float64)
    Rs = np.asarray(sys.Rs, dtype=np.float64)
    h = hashlib.md5()
    h.update(apos.tobytes())
    h.update(np.array(sys.enames, dtype=object).tobytes())
    h.update(np.round(qs, 6).tobytes())
    h.update(np.round(Rs, 6).tobytes())
    h.update(os.path.abspath(sub_file).encode('utf-8'))
    h.update(repr((nu, nv, alpha, z_min, z_max)).encode('utf-8'))
    return h.hexdigest()


def _get_sys(window):
    """Get backend AtomicSystem, synced and pre-initialized for XYZ export."""
    backend = getattr(window, 'backend', None)
    if backend is None:
        return None
    if hasattr(backend, '_sync_sys'):
        backend._sync_sys()
    sys = backend.sys
    if sys is None or sys.apos is None or len(sys.apos) == 0:
        return None
    # _sync_sys now sets qs/Rs/aux_labels from graph; fallback to preinitialize if still missing
    if sys.qs is None or sys.Rs is None or sys.aux_labels is None:
        sys.preinitialize_atomic_properties()
    return sys


def _export_molecule_to_temp(window):
    """Write current editor molecule to a temporary XYZ file."""
    sys = _get_sys(window)
    if sys is None:
        raise ValueError("No molecule to export")
    fd, fname = tempfile.mkstemp(suffix='.xyz')
    os.close(fd)
    sys.saveXYZ(fname, blvec=False, comment='FoldedRigid temp molecule')
    return fname


def _ensure_backend_matched(window, atom_positions):
    """If the backend graph has a different number of atoms than the RBD result,
    rebuild it from the fit result so positions can be synced."""
    backend = window.backend
    rbd = getattr(window, 'fr_rbd', None)
    fit = getattr(window, 'fr_fit_result', None)
    atom_list = [a for a in backend.graph.atoms.values() if a.alive]
    if len(atom_list) == len(atom_positions):
        return
    if rbd is None or fit is None:
        return
    from spammm.topology import AtomicGraph
    from spammm import elements as _elements
    enames = fit.get('enames') or rbd.enames
    apos_mol = fit.get('apos_mol')
    if enames is None or apos_mol is None:
        return
    new_graph = AtomicGraph()
    for i, e in enumerate(enames):
        pos = np.array(apos_mol[i], dtype=np.float64)
        new_graph.add_atom(pos, e, _elements.ELEMENT_DICT[e][0])
    backend.graph = new_graph


def _update_graph(window, atom_positions):
    """Propagate relaxed atom positions back to the authoritative AtomicGraph."""
    backend = window.backend
    if hasattr(backend, 'graph') and hasattr(backend.graph, 'update_positions_from_array'):
        _ensure_backend_matched(window, atom_positions)
        backend.graph.update_positions_from_array(atom_positions)
    if hasattr(backend, '_sync_sys'):
        backend._sync_sys()
    if hasattr(window, 'refresh_view'):
        window.refresh_view()


def _molecule_bbox(sys, pad=5.0):
    """Return (xmin, xmax, ymin, ymax) for the current molecule, padded."""
    apos = sys.apos
    if apos is None or len(apos) == 0:
        return -pad, pad, -pad, pad
    return (
        float(apos[:, 0].min() - pad), float(apos[:, 0].max() + pad),
        float(apos[:, 1].min() - pad), float(apos[:, 1].max() + pad),
    )


def _load_and_replicate_substrate(window):
    """Load substrate and replicate it to cover the current molecule area."""
    sub_file = window.fr_substrate_edit.text()
    if not sub_file or not os.path.isfile(sub_file):
        raise ValueError(f"Substrate file not found: {sub_file!r}")
    apos, _, enames, _, lvec = load_xyz_with_REQs(sub_file)
    if lvec is None:
        raise ValueError(f"Substrate file {sub_file} must contain lattice vectors")
    sys = _get_sys(window)
    x_min, x_max, y_min, y_max = _molecule_bbox(sys)
    rep_pos, rep_names = FoldedRigid.replicate_substrate(apos, enames, lvec, (x_min, x_max), (y_min, y_max))
    return rep_pos, rep_names


def _update_substrate_overlay(window):
    """Create/update the FAF substrate potential map overlay (shared VispyUtils.update_faf_map_overlay).

    Shows the same FAF heatmap as demo_pairff.py / RigidBodyVispy._recompute_map,
    using potential_to_rgba (display SSOT).
    """
    fit = getattr(window, 'fr_fit_result', None)
    if fit is None:
        return
    z = FoldedRigid.Z_SURF_TOP + float(window.fr_z_spin.value())
    span = float(window.fr_potential_span_spin.value())
    extent = (-span, span, -span, span)
    visible = getattr(window, 'fr_show_substrate', False)
    img = update_faf_map_overlay(window.scene, fit, z, extent,
                                 image_attr='fr_substrate_map', visible=visible)
    window.fr_substrate_map = img


def _toggle_substrate(window, checked=None):
    """Toggle substrate overlay visibility."""
    if checked is None:
        checked = not getattr(window, 'fr_show_substrate', False)
    window.fr_show_substrate = checked
    if checked:
        try:
            _update_substrate_overlay(window)
        except Exception as e:
            _status(window, f"Substrate overlay failed: {e}")
            window.fr_show_substrate = False
    elif hasattr(window, 'fr_substrate_map'):
        window.fr_substrate_map.visible = False


def _get_potential_type_idx(window):
    """Get atom type index for potential slice from combo box."""
    fit = getattr(window, 'fr_fit_result', None)
    if fit is None:
        return 0
    unique_REQs = fit.get('unique_REQs')
    if unique_REQs is None:
        return 0
    idx = window.fr_potential_type_combo.currentIndex()
    if idx < 0 or idx >= len(unique_REQs):
        idx = 0
    return idx


def _update_potential_overlay(window):
    """Create/update the vispy potential slice mesh in the main scene (shared VispyUtils)."""
    fit = getattr(window, 'fr_fit_result', None)
    if fit is None:
        raise ValueError("No fit loaded — Load Fit first")
    ityp = _get_potential_type_idx(window)
    z = FoldedRigid.Z_SURF_TOP + float(window.fr_z_spin.value())
    span = float(window.fr_potential_span_spin.value())
    n = int(window.fr_potential_n_spin.value())
    a, b, E = FoldedRigid.eval_folded_potential_slice(fit, ityp, plane='xy', fixed_val=z, extent=(-span, span, -span, span), n=n)
    rgba, vmin, vmax = colormap_rgba(E, cmap='bwr', symmetric=True, alpha=0.6)
    xs, ys = a, b
    zs = np.full((n, n), z, dtype=np.float32)
    visible = getattr(window, 'fr_show_potential', False)
    mesh = update_potential_overlay(window.scene, xs, ys, zs, rgba,
                                    mesh_attr='fr_potential_mesh', visible=visible)
    window.fr_potential_mesh = mesh


def _toggle_potential(window, checked=None):
    """Toggle potential slice overlay visibility."""
    if checked is None:
        checked = not getattr(window, 'fr_show_potential', False)
    window.fr_show_potential = checked
    if checked:
        try:
            _update_potential_overlay(window)
            _status(window, "Potential slice shown")
        except Exception as e:
            _status(window, f"Potential overlay failed: {e}")
            traceback.print_exc()
            window.fr_show_potential = False
    elif hasattr(window, 'fr_potential_mesh'):
        window.fr_potential_mesh.visible = False


def _on_potential_type_changed(window, idx):
    """Refresh potential overlay when atom type combo changes."""
    if getattr(window, 'fr_show_potential', False):
        try:
            _update_potential_overlay(window)
        except Exception as e:
            _status(window, f"Potential update failed: {e}")


def _populate_potential_type_combo(window, fit):
    """Populate the atom type combo from fit's unique_REQs."""
    combo = window.fr_potential_type_combo
    combo.blockSignals(True)
    combo.clear()
    unique_REQs = fit.get('unique_REQs')
    enames = fit.get('enames', [])
    atom_type_ids = fit.get('atom_type_ids', [])
    if unique_REQs is not None:
        for i, req in enumerate(unique_REQs):
            R, E, Q = req[0], req[1], req[2]
            label = f"type{i} R={R:.2f} Q={Q:+.2f}"
            if i < len(atom_type_ids):
                count = int(np.sum(atom_type_ids == i))
                label += f" ({count}x)"
            combo.addItem(label)
    combo.blockSignals(False)


# ---------------------------------------------------------------------------
# Workflow callbacks
# ---------------------------------------------------------------------------

def _on_fit(window):
    """Fit folded basis coefficients for current molecule and substrate (cached)."""
    try:
        sys = _get_sys(window)
        sub_file = window.fr_substrate_edit.text()
        nu = int(window.fr_nu_spin.value())
        nv = int(window.fr_nv_spin.value())
        alpha = float(window.fr_alpha_spin.value())
        z_min = float(window.fr_zmin_spin.value())
        z_max = float(window.fr_zmax_spin.value())
        # Cache path keyed on geometry+substrate+params (custom substrate support)
        key = _fit_cache_key(sys, sub_file, nu, nv, alpha, z_min, z_max)
        cache_path = os.path.join(_fit_cache_dir(), f"{key}.npz")
        # Build molecule tuple with QEq charges — sys.qs are valence electron
        # counts, NOT QEq charges. Without QEq, Coulomb is zero (no checkerboard).
        from spammm.forcefields.QEq import compute_qeq_reqs
        apos = np.asarray(sys.apos, dtype=np.float32)
        enames = list(sys.enames)
        REQs = compute_qeq_reqs(apos, enames, name='editor_mol')
        _status(window, f"Fitting folded basis (QEq Q=[{REQs[:, 2].min():.3f},{REQs[:, 2].max():.3f}])...")
        fit = FoldedRigid.load_or_fit_faf(
            (apos, enames, REQs), mol_name='editor_mol',
            substrate_file=sub_file, z_range_rel=(z_min, z_max),
            fit_path=cache_path,
            nu=nu, nv=nv, nPBC=(4, 4, 0), alpha_morse=alpha,
            custom_alphas=FoldedRigid.COMBINED_ALPHAS,
        )
        window.fr_fit_result = fit
        _populate_potential_type_combo(window, fit)
        _on_setup(window)
        _status(window, f"Fit complete: {fit['coeffs'].shape}")
    except Exception as e:
        _status(window, f"Fit failed: {e}")
        traceback.print_exc()


def _on_setup(window):
    """Create a RigidBodyDynamics instance from the fitted basis."""
    try:
        fit = getattr(window, 'fr_fit_result', None)
        if fit is None:
            raise ValueError("Run Fit first")
        _status(window, "Setting up rigid body...")
        rbd = FoldedRigid.setup_rigid_folded(
            None,
            fit,
            z_init=float(window.fr_z_spin.value()),
            xy_init=(float(window.fr_x_spin.value()), float(window.fr_y_spin.value())),
            quats=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            mass_trans=1.0,
        )
        window.fr_rbd = rbd
        _status(window, f"Setup complete: {rbd.num_atoms} atoms")
    except Exception as e:
        _status(window, f"Setup failed: {e}")
        traceback.print_exc()


def _on_relax(window):
    """Run a relaxation and update the editor view with the final geometry."""
    try:
        rbd = getattr(window, 'fr_rbd', None)
        if rbd is None:
            raise ValueError("Run Setup first")
        n_steps = int(window.fr_nsteps_spin.value())
        dt = float(window.fr_dt_spin.value())
        lin_damp = float(window.fr_lin_damp_spin.value())
        ang_damp = float(window.fr_ang_damp_spin.value())
        record_interval = max(1, n_steps // 50)
        _status(window, f"Relaxing {n_steps} steps...")
        traj = FoldedRigid.relax_folded(
            rbd, n_steps=n_steps, dt=dt, lin_damp=lin_damp, ang_damp=ang_damp,
            record_interval=record_interval,
        )
        window.fr_relax_traj = traj
        final_pos = traj['atom_positions'][-1]
        _update_graph(window, final_pos)
        E = float(traj['energies'][-1])
        F = float(traj['forces'][-1])
        T = float(traj['torques'][-1])
        window.fr_energy_label.setText(f"E: {E:.6f}  |F|: {F:.4f}  |T|: {T:.4f}")
        _status(window, f"Relaxation complete ({len(traj['energies'])} records)")
    except Exception as e:
        _status(window, f"Relax failed: {e}")
        traceback.print_exc()


def _on_lateral_scan(window):
    """Run a small lateral force map around the current COM."""
    try:
        rbd = getattr(window, 'fr_rbd', None)
        if rbd is None:
            raise ValueError("Run Setup first")
        x0 = float(window.fr_x_spin.value())
        y0 = float(window.fr_y_spin.value())
        z = FoldedRigid.Z_SURF_TOP + float(window.fr_z_scan_spin.value())
        span = float(window.fr_scan_span_spin.value())
        n = int(window.fr_scan_n_spin.value())
        xs = np.linspace(x0 - span, x0 + span, n)
        ys = np.linspace(y0 - span, y0 + span, n)
        _status(window, f"Lateral scan {n}x{n}...")
        scan = FoldedRigid.lateral_scan(rbd, xs, ys, z, n_relax=50, dt=0.01)
        window.fr_scan_result = scan
        _status(window, f"Scan complete ({n}x{n})")
        _plot_scan_dialog(window, scan)
    except Exception as e:
        _status(window, f"Scan failed: {e}")
        traceback.print_exc()


def _on_relaxed_scan(window):
    """Run a pinned-atom relaxed scan along the x-axis."""
    try:
        rbd = getattr(window, 'fr_rbd', None)
        if rbd is None:
            raise ValueError("Run Setup first")
        pin_idx = getattr(window, '_fr_pin_idx', 0)
        if pin_idx < 0 or pin_idx >= rbd.num_atoms:
            pin_idx = 0
        x0 = float(window.fr_x_spin.value())
        y0 = float(window.fr_y_spin.value())
        z = FoldedRigid.Z_SURF_TOP + float(window.fr_z_spin.value())
        span = float(window.fr_scan_span_spin.value())
        n = int(window.fr_scan_n_spin.value())
        path = np.array([[x0 + (i / max(1, n - 1)) * span, y0, z] for i in range(n)], dtype=np.float32)
        _status(window, f"Relaxed scan (pin {pin_idx}) {n} points...")
        traj = FoldedRigid.relaxed_scan(rbd, pin_idx, path, k_spring=5.0, n_relax=200, dt=0.005)
        window.fr_relaxed_scan_traj = traj
        final_pos = traj['atom_positions'][-1]
        _update_graph(window, final_pos)
        _status(window, f"Relaxed scan complete ({len(traj['positions'])} records)")
    except Exception as e:
        _status(window, f"Relaxed scan failed: {e}")
        traceback.print_exc()


def _on_plot_relax(window):
    """Plot the last relaxation trajectory."""
    traj = getattr(window, 'fr_relax_traj', None)
    if traj is None:
        _status(window, "No relaxation trajectory to plot")
        return
    _plot_relaxation_dialog(window, traj)


def _on_save_fit(window):
    """Save current fit result to a user-selected .npz file."""
    fit = getattr(window, 'fr_fit_result', None)
    if fit is None:
        _status(window, "No fit result to save")
        return
    fname = window.fileDialog(mode="save", title="Save Fit Parameters", filter_str="Numpy Files (*.npz);;All Files (*)")
    if fname:
        if not fname.endswith('.npz'):
            fname += '.npz'
        FoldedRigid.save_fit(fit, fname)
        _status(window, f"Saved fit to {fname}")


def _load_fit_path(window, fname):
    """Load fit result from a .npz file and set up the rigid body."""
    window.fr_fit_result = FoldedRigid.load_fit(fname)
    _populate_potential_type_combo(window, window.fr_fit_result)
    _on_setup(window)
    _status(window, f"Loaded fit: {window.fr_fit_result['coeffs'].shape}")


def _on_load_fit(window):
    """Load fit result from a user-selected .npz file."""
    fname = window.fileDialog(mode="open", title="Load Fit Parameters", filter_str="Numpy Files (*.npz);;All Files (*)")
    if fname:
        try:
            _load_fit_path(window, fname)
        except Exception as e:
            _status(window, f"Load fit failed: {e}")
            traceback.print_exc()


def _run_n(window, n_steps=None, dt=None, lin_damp=None, ang_damp=None, update=True):
    """Run n_steps of folded rigid dynamics and optionally update the GUI."""
    rbd = getattr(window, 'fr_rbd', None)
    if rbd is None:
        raise ValueError("No RBD set up. Load a fit or run Fit first.")
    if n_steps is None:
        n_steps = int(window.fr_niter_spin.value())
    dt = float(window.fr_dt_spin.value()) if dt is None else float(dt)
    lin_damp = float(window.fr_lin_damp_spin.value()) if lin_damp is None else float(lin_damp)
    ang_damp = float(window.fr_ang_damp_spin.value()) if ang_damp is None else float(ang_damp)
    rbd.run_folded(n_steps, dt, lin_damp=lin_damp, ang_damp=ang_damp)
    out = rbd.download_outputs()
    atom_pos = out['atom_positions'][0]
    E = float(atom_pos[:, 3].sum())
    f = out['body_force'][0]
    tq = out['body_torque'][0]
    F = float(np.linalg.norm(f[:3]))
    T = float(np.linalg.norm(tq[:3]))
    if update:
        _update_graph(window, atom_pos[:, :3])
        window.fr_energy_label.setText(f"E: {E:.6f}  |F|: {F:.4f}  |T|: {T:.4f}")
    return out, E, F, T


def _on_run(window):
    """Toggle continuous play/pause simulation loop."""
    if getattr(window, '_fr_timer_running', False):
        _on_stop(window)
        return
    try:
        rbd = getattr(window, 'fr_rbd', None)
        if rbd is None:
            raise ValueError("No RBD set up. Load a fit or run Fit first.")
        window.set_edit_mode('fr_manip')
        if not hasattr(window, '_fr_timer'):
            window._fr_timer = QtCore.QTimer(window)
            window._fr_timer.timeout.connect(lambda: _on_timer(window))
        interval_ms = max(20, int(0.1 * window.fr_niter_spin.value()))
        window._fr_timer.start(interval_ms)
        window._fr_timer_running = True
        window.fr_run_btn.setText("Stop")
        window.fr_run_btn.setToolTip("Stop continuous simulation")
        _status(window, "Continuous run started")
    except Exception as e:
        _status(window, f"Run failed: {e}")
        traceback.print_exc()


def _on_stop(window):
    """Stop the continuous simulation timer."""
    if hasattr(window, '_fr_timer') and window._fr_timer is not None:
        window._fr_timer.stop()
    window._fr_timer_running = False
    window.fr_run_btn.setText("Run")
    window.fr_run_btn.setToolTip("Run continuous simulation")
    _status(window, "Run stopped")


def _on_timer(window):
    """Advance the continuous simulation by one tick."""
    try:
        chunk = int(window.fr_niter_spin.value())
        fconv = float(window.fr_fconv_spin.value())
        out, E, F, T = _run_n(window, n_steps=max(1, chunk))
        if fconv > 0.0 and F < fconv:
            _on_stop(window)
            _status(window, f"Run converged F={F:.4g}")
            return
        if getattr(window, '_fr_pin_idx', -1) >= 0:
            _status(window, f"drag | E={E:.4f} |F|={F:.4f} |T|={T:.4f}")
    except Exception as e:
        _on_stop(window)
        _status(window, f"Run failed: {e}")
        traceback.print_exc()


def _on_step(window):
    """Run a single dynamics step."""
    try:
        _run_n(window, 1)
        _status(window, "Step")
    except Exception as e:
        _status(window, f"Step failed: {e}")
        traceback.print_exc()


def _on_pick_substrate(window):
    """Open file dialog to select substrate XYZ."""
    fname = window.fileDialog(mode="open", title="Select Substrate XYZ", filter_str="XYZ Files (*.xyz);;All Files (*)")
    if fname:
        window.fr_substrate_edit.setText(fname)


# ---------------------------------------------------------------------------
# Matplotlib plot dialogs
# ---------------------------------------------------------------------------

def _plot_relaxation_dialog(window, traj):
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        fig = Figure(figsize=(10, 9), dpi=100)
        axes = fig.subplots(3, 1, sharex=True)
        steps = np.arange(len(traj['energies']))
        axes[0].plot(steps, traj['energies'], 'b-')
        axes[0].set_ylabel('Energy [eV]')
        axes[0].set_title('Folded Rigid Relaxation')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(steps, traj['forces'], 'r-')
        axes[1].set_ylabel('|Force| [eV/Å]')
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(steps, traj['torques'], 'g-')
        axes[2].set_ylabel('|Torque| [eV]')
        axes[2].set_xlabel('Step')
        axes[2].grid(True, alpha=0.3)
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        plot_window = QtWidgets.QDialog(window)
        plot_window.setWindowTitle("Folded Rigid Relaxation")
        layout = QtWidgets.QVBoxLayout(plot_window)
        layout.addWidget(canvas)
        plot_window.resize(800, 700)
        plot_window.show()
        if not hasattr(window, '_fr_plot_windows'):
            window._fr_plot_windows = []
        window._fr_plot_windows.append(plot_window)
    except Exception as e:
        _status(window, f"Plot failed: {e}")
        traceback.print_exc()


def _plot_scan_dialog(window, scan):
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        X, Y = scan['X'], scan['Y']
        Fz = scan['Fz']
        Fmag = np.sqrt(scan['Fx']**2 + scan['Fy']**2 + Fz**2)
        E = scan['E']

        fig = Figure(figsize=(18, 5), dpi=100)
        axes = fig.subplots(1, 3)
        for ax, data, title, cmap in [
            (axes[0], Fz, 'Fz [eV/Å]', 'RdBu_r'),
            (axes[1], Fmag, '|F| [eV/Å]', 'hot'),
            (axes[2], E, 'Energy [eV]', 'viridis'),
        ]:
            im = ax.pcolormesh(X, Y, data, shading='auto', cmap=cmap)
            ax.set_aspect('equal')
            ax.set_xlabel('X [Å]')
            ax.set_ylabel('Y [Å]')
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        plot_window = QtWidgets.QDialog(window)
        plot_window.setWindowTitle("Folded Rigid Lateral Scan")
        layout = QtWidgets.QVBoxLayout(plot_window)
        layout.addWidget(canvas)
        plot_window.resize(1200, 450)
        plot_window.show()
        if not hasattr(window, '_fr_plot_windows'):
            window._fr_plot_windows = []
        window._fr_plot_windows.append(plot_window)
    except Exception as e:
        _status(window, f"Scan plot failed: {e}")
        traceback.print_exc()


def _on_xz_slice(window):
    """Compute and plot XZ side-view potential slice."""
    _plot_side_slice(window, 'xz')


def _on_yz_slice(window):
    """Compute and plot YZ side-view potential slice."""
    _plot_side_slice(window, 'yz')


def _plot_side_slice(window, plane):
    """Compute and plot a side-view (XZ or YZ) potential slice in a matplotlib dialog."""
    try:
        fit = getattr(window, 'fr_fit_result', None)
        if fit is None:
            raise ValueError("No fit loaded — Load Fit first")
        ityp = _get_potential_type_idx(window)
        fixed_val = float(window.fr_x_spin.value()) if plane == 'xz' else float(window.fr_y_spin.value())
        span_xy = float(window.fr_potential_span_spin.value())
        z_min = FoldedRigid.Z_SURF_TOP + float(window.fr_zmin_spin.value()) - 1.0
        z_max = FoldedRigid.Z_SURF_TOP + float(window.fr_zmax_spin.value()) + 2.0
        n = int(window.fr_potential_n_spin.value())
        a, b, E = FoldedRigid.eval_folded_potential_slice(fit, ityp, plane=plane, fixed_val=fixed_val, extent=(-span_xy, span_xy, z_min, z_max), n=n)
        _plot_potential_slice_dialog(window, a, b, E, plane, fixed_val, ityp, fit)
    except Exception as e:
        _status(window, f"Slice failed: {e}")
        traceback.print_exc()


def _plot_potential_slice_dialog(window, a, b, E, plane, fixed_val, ityp, fit):
    """Show a 2D potential slice in a matplotlib dialog."""
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        unique_REQs = fit.get('unique_REQs')
        type_label = f'type {ityp}'
        if unique_REQs is not None and ityp < len(unique_REQs):
            req = unique_REQs[ityp]
            type_label = f'R={req[0]:.2f} E={req[1]:.4f} Q={req[2]:.2f}'

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.subplots(1, 1)
        A, B = np.meshgrid(a, b, indexing='ij')
        im = ax.pcolormesh(A, B, E, shading='auto', cmap='bwr', vmin=-max(abs(E.min()), abs(E.max())), vmax=max(abs(E.min()), abs(E.max())))
        ax.set_aspect('equal')
        fixed_label = 'y' if plane == 'xz' else 'x'
        h_label = 'x' if plane == 'xz' else 'y'
        ax.set_xlabel(f'{h_label} [Å]')
        ax.set_ylabel('z [Å]')
        ax.set_title(f'Potential slice {plane} at {fixed_label}={fixed_val:.2f} ({type_label})')
        ax.axhline(y=FoldedRigid.Z_SURF_TOP, color='gray', linestyle='--', alpha=0.5, label='surface top')
        fig.colorbar(im, ax=ax, label='Energy [eV]')
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        plot_window = QtWidgets.QDialog(window)
        plot_window.setWindowTitle(f'Potential {plane.upper()} Slice')
        layout = QtWidgets.QVBoxLayout(plot_window)
        layout.addWidget(canvas)
        plot_window.resize(800, 600)
        plot_window.show()
        if not hasattr(window, '_fr_plot_windows'):
            window._fr_plot_windows = []
        window._fr_plot_windows.append(plot_window)
    except Exception as e:
        _status(window, f"Potential plot failed: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Edit mode handlers
# ---------------------------------------------------------------------------

def _build_edit_mode_handlers(window):
    class FRPinMode(EditModeHandler):
        status_msg = "FoldedRigid: click atom to select pin atom"
        def on_atom_click(self, atom_id):
            idx = window.scene._id_to_idx.get(atom_id, -1)
            window._fr_pin_idx = idx
            window._fr_pin_atom_id = atom_id
            _status(window, f"Pin atom selected: {atom_id} (idx={idx})")

    class FRComMode(EditModeHandler):
        status_msg = "FoldedRigid: click to set COM (x,y)"
        def on_press(self, event, p_world, ctrl):
            if event.button == 1:
                window.fr_x_spin.setValue(float(p_world[0]))
                window.fr_y_spin.setValue(float(p_world[1]))
                _status(window, f"COM set to ({p_world[0]:.2f}, {p_world[1]:.2f})")

    class FRManipMode(EditModeHandler):
        status_msg = "LMB: Pick & drag atom to pull molecule | release LMB to drop"
        lock_drag = True
        capture_move = True

        def on_activate(self):
            if getattr(window, 'fr_rbd', None) is None:
                _status(window, "FR Manip: load a fit and press Setup first")
            self._pin = -1

        def _pick_idx(self, event):
            atom_id, dist = window.scene._pick_id_from_mouse(event.pos, max_dist=window.pick_radius)
            if atom_id < 0:
                return -1
            return window.scene._id_to_idx_safe(atom_id)

        def _set_anchors(self, idx, target):
            rbd = window.fr_rbd
            if rbd is None:
                return
            anchors = rbd.anchors.copy()
            anchors[:, 3] = -1.0
            if idx >= 0:
                anchors[idx, :3] = target
                anchors[idx, 3] = float(window.fr_k_spring_spin.value())
            rbd.update_anchors(anchors)

        def _closest_point_on_ray(self, atom_pos, r0, rd):
            rd = np.asarray(rd, dtype=np.float64)
            rd2 = float(np.dot(rd, rd))
            if rd2 < 1e-12:
                return np.asarray(r0, dtype=np.float64)
            t = float(np.dot(atom_pos - r0, rd) / rd2)
            return r0 + t * rd

        def on_press(self, event, p_world, ctrl):
            if event.button != 1:
                return
            idx = self._pick_idx(event)
            if idx < 0:
                return
            self._pin = idx
            window._fr_pin_idx = idx
            window._fr_pin_atom_id = window.scene._idx_to_id(idx)
            atom_pos = window.scene._pos[idx].astype(np.float64)
            r0, rd = window.scene._ray_from_mouse(event.pos)
            target = self._closest_point_on_ray(atom_pos, r0, rd).astype(np.float32)
            self._set_anchors(idx, target)
            _status(window, f"Pinned atom {window._fr_pin_atom_id} (idx={idx}); drag to pull")

        def on_move(self, p_world, r0=None, rd=None):
            if self._pin < 0:
                return
            rbd = window.fr_rbd
            if rbd is None:
                return
            atom_pos = window.scene._pos[self._pin].astype(np.float64)
            if r0 is not None and rd is not None:
                target = self._closest_point_on_ray(atom_pos, r0, rd)
            else:
                target = np.array([p_world[0], p_world[1], atom_pos[2]], dtype=np.float64)
            self._set_anchors(self._pin, target.astype(np.float32))

        def on_release(self, event, p_world, ctrl):
            if event.button != 1:
                return
            if self._pin < 0:
                return
            self._set_anchors(-1, np.zeros(3, dtype=np.float32))
            self._pin = -1
            window._fr_pin_idx = -1
            window._fr_pin_atom_id = -1
            window.scene._pick_active = False
            _status(window, "Pin released")

        def on_rmb_atom(self, atom_id, ctrl):
            pass

    window.register_mode_handler('fr_pin', FRPinMode(window))
    window.register_mode_handler('fr_com', FRComMode(window))
    window.register_mode_handler('fr_manip', FRManipMode(window))


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------

def build_ui(window):
    """Build the Folded Rigid extension panel."""
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    # State defaults
    window.fr_fit_result = None
    window.fr_rbd = None
    window.fr_relax_traj = None
    window.fr_scan_result = None
    window.fr_relaxed_scan_traj = None
    window.fr_show_substrate = False
    window.fr_show_potential = False
    window._fr_pin_idx = 0
    window._fr_pin_atom_id = -1
    window._fr_plot_windows = []
    window._fr_drag_pin = False

    # ------------------------------------------------------------------
    # Main dynamics controls (always visible)
    # ------------------------------------------------------------------
    g_main = AutoGridPlacer(cols=4)
    window.fr_load_fit_btn = QtWidgets.QPushButton("Load Fit")
    window.fr_load_fit_btn.setToolTip("Load pre-fitted .npz and set up")
    window.fr_load_fit_btn.clicked.connect(lambda: _on_load_fit(window))
    g_main.add(window.fr_load_fit_btn)
    window.fr_run_btn = QtWidgets.QPushButton("Run")
    window.fr_run_btn.setToolTip("Run n_iters steps (or until Fconv)")
    window.fr_run_btn.clicked.connect(lambda: _on_run(window))
    g_main.add(window.fr_run_btn)
    window.fr_step_btn = QtWidgets.QPushButton("Step")
    window.fr_step_btn.setToolTip("Run one dynamics step")
    window.fr_step_btn.clicked.connect(lambda: _on_step(window))
    g_main.add(window.fr_step_btn)
    layout.addLayout(g_main.layout())

    # COM / drag spring
    g_pos = AutoGridPlacer(cols=4)
    window.fr_x_spin = QtWidgets.QDoubleSpinBox(); window.fr_x_spin.setRange(-100.0, 100.0); window.fr_x_spin.setSingleStep(0.1); window.fr_x_spin.setValue(0.0); window.fr_x_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_pos.add_pair("x:", window.fr_x_spin)
    window.fr_y_spin = QtWidgets.QDoubleSpinBox(); window.fr_y_spin.setRange(-100.0, 100.0); window.fr_y_spin.setSingleStep(0.1); window.fr_y_spin.setValue(0.0); window.fr_y_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_pos.add_pair("y:", window.fr_y_spin)
    window.fr_z_spin = QtWidgets.QDoubleSpinBox(); window.fr_z_spin.setRange(0.1, 50.0); window.fr_z_spin.setSingleStep(0.1); window.fr_z_spin.setValue(2.5); window.fr_z_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_pos.add_pair("z:", window.fr_z_spin)
    window.fr_k_spring_spin = QtWidgets.QDoubleSpinBox(); window.fr_k_spring_spin.setRange(0.01, 1000.0); window.fr_k_spring_spin.setSingleStep(0.5); window.fr_k_spring_spin.setValue(2.0); window.fr_k_spring_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_pos.add_pair("k:", window.fr_k_spring_spin)
    layout.addLayout(g_pos.layout())

    # Dynamics parameters
    g_dyn = AutoGridPlacer(cols=4)
    window.fr_niter_spin = QtWidgets.QSpinBox(); window.fr_niter_spin.setRange(1, 10000000); window.fr_niter_spin.setValue(250); window.fr_niter_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_dyn.add_pair("n_iter:", window.fr_niter_spin)
    window.fr_dt_spin = QtWidgets.QDoubleSpinBox(); window.fr_dt_spin.setRange(0.0001, 1.0); window.fr_dt_spin.setSingleStep(0.001); window.fr_dt_spin.setValue(0.02); window.fr_dt_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_dyn.add_pair("dt:", window.fr_dt_spin)
    window.fr_fconv_spin = QtWidgets.QDoubleSpinBox(); window.fr_fconv_spin.setRange(0.0, 100.0); window.fr_fconv_spin.setSingleStep(0.001); window.fr_fconv_spin.setValue(0.0); window.fr_fconv_spin.setDecimals(5); window.fr_fconv_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_dyn.add_pair("Fconv:", window.fr_fconv_spin)
    window.fr_lin_damp_spin = QtWidgets.QDoubleSpinBox(); window.fr_lin_damp_spin.setRange(0.0, 1.0); window.fr_lin_damp_spin.setSingleStep(0.01); window.fr_lin_damp_spin.setValue(0.95); window.fr_lin_damp_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_dyn.add_pair("ld:", window.fr_lin_damp_spin)
    window.fr_ang_damp_spin = QtWidgets.QDoubleSpinBox(); window.fr_ang_damp_spin.setRange(0.0, 1.0); window.fr_ang_damp_spin.setSingleStep(0.01); window.fr_ang_damp_spin.setValue(0.90); window.fr_ang_damp_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_dyn.add_pair("ad:", window.fr_ang_damp_spin)
    layout.addLayout(g_dyn.layout())

    # Overlay / energy
    g_ov = AutoGridPlacer(cols=4)
    window.fr_show_substrate_chk = QtWidgets.QCheckBox("Substrate")
    window.fr_show_substrate_chk.setChecked(False)
    window.fr_show_substrate_chk.stateChanged.connect(lambda state: _toggle_substrate(window, state == QtCore.Qt.Checked))
    g_ov.add(window.fr_show_substrate_chk)
    window.fr_show_potential_chk = QtWidgets.QCheckBox("Potential")
    window.fr_show_potential_chk.setChecked(False)
    window.fr_show_potential_chk.stateChanged.connect(lambda state: _toggle_potential(window, state == QtCore.Qt.Checked))
    g_ov.add(window.fr_show_potential_chk)
    layout.addLayout(g_ov.layout())

    # Potential slice controls
    g_pot = AutoGridPlacer(cols=4)
    window.fr_potential_type_combo = QtWidgets.QComboBox()
    window.fr_potential_type_combo.addItem("O (q=-0.4)")
    window.fr_potential_type_combo.addItem("H (q=+0.2)")
    window.fr_potential_type_combo.currentIndexChanged.connect(lambda idx: _on_potential_type_changed(window, idx))
    g_pot.add_pair("type:", window.fr_potential_type_combo)
    window.fr_potential_span_spin = QtWidgets.QDoubleSpinBox(); window.fr_potential_span_spin.setRange(1.0, 50.0); window.fr_potential_span_spin.setSingleStep(0.5); window.fr_potential_span_spin.setValue(8.0); window.fr_potential_span_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_pot.add_pair("span:", window.fr_potential_span_spin)
    window.fr_potential_n_spin = QtWidgets.QSpinBox(); window.fr_potential_n_spin.setRange(8, 256); window.fr_potential_n_spin.setValue(64); window.fr_potential_n_spin.setMaximumWidth(50)
    g_pot.add_pair("n:", window.fr_potential_n_spin)
    layout.addLayout(g_pot.layout())

    # Slice buttons
    g_slice = AutoGridPlacer(cols=4)
    window.fr_xz_slice_btn = QtWidgets.QPushButton("XZ Slice")
    window.fr_xz_slice_btn.setToolTip("Plot XZ side-view potential slice at current y")
    window.fr_xz_slice_btn.clicked.connect(lambda: _on_xz_slice(window))
    g_slice.add(window.fr_xz_slice_btn)
    window.fr_yz_slice_btn = QtWidgets.QPushButton("YZ Slice")
    window.fr_yz_slice_btn.setToolTip("Plot YZ side-view potential slice at current x")
    window.fr_yz_slice_btn.clicked.connect(lambda: _on_yz_slice(window))
    g_slice.add(window.fr_yz_slice_btn)
    layout.addLayout(g_slice.layout())

    window.fr_energy_label = QtWidgets.QLabel("E: ---  |F|: ---  |T|: ---")
    window.fr_energy_label.setWordWrap(True)
    layout.addWidget(window.fr_energy_label)

    window.fr_status_label = QtWidgets.QLabel("Status: idle")
    window.fr_status_label.setWordWrap(True)
    layout.addWidget(window.fr_status_label)

    # ------------------------------------------------------------------
    # Advanced section: fitting, scans, plotting
    # ------------------------------------------------------------------
    adv = CollapsibleSection("Fitting & Scans", collapsed=True)
    adv_layout = QtWidgets.QVBoxLayout()
    apply_tight(adv_layout)
    adv_widget = QtWidgets.QWidget()
    adv_widget.setLayout(adv_layout)

    # Substrate file
    g_sub = AutoGridPlacer(cols=4)
    window.fr_substrate_edit = QtWidgets.QLineEdit(DEFAULT_SUBSTRATE)
    window.fr_substrate_edit.setReadOnly(True)
    g_sub.add_pair("Substrate:", window.fr_substrate_edit)
    window.fr_substrate_btn = QtWidgets.QPushButton("...")
    window.fr_substrate_btn.clicked.connect(lambda: _on_pick_substrate(window))
    g_sub.add(window.fr_substrate_btn)
    adv_layout.addLayout(g_sub.layout())

    # Fit parameters
    g_fit = AutoGridPlacer(cols=4)
    window.fr_nu_spin = QtWidgets.QSpinBox(); window.fr_nu_spin.setRange(1, 32); window.fr_nu_spin.setValue(4); window.fr_nu_spin.setMaximumWidth(50)
    g_fit.add_pair("nu:", window.fr_nu_spin)
    window.fr_nv_spin = QtWidgets.QSpinBox(); window.fr_nv_spin.setRange(1, 32); window.fr_nv_spin.setValue(4); window.fr_nv_spin.setMaximumWidth(50)
    g_fit.add_pair("nv:", window.fr_nv_spin)
    window.fr_alpha_spin = QtWidgets.QDoubleSpinBox(); window.fr_alpha_spin.setRange(0.1, 10.0); window.fr_alpha_spin.setSingleStep(0.1); window.fr_alpha_spin.setValue(1.8); window.fr_alpha_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_fit.add_pair("α:", window.fr_alpha_spin)
    adv_layout.addLayout(g_fit.layout())

    # Fit z range
    g_zr = AutoGridPlacer(cols=4)
    window.fr_zmin_spin = QtWidgets.QDoubleSpinBox(); window.fr_zmin_spin.setRange(0.1, 20.0); window.fr_zmin_spin.setSingleStep(0.1); window.fr_zmin_spin.setValue(1.5); window.fr_zmin_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    window.fr_zmax_spin = QtWidgets.QDoubleSpinBox(); window.fr_zmax_spin.setRange(0.1, 50.0); window.fr_zmax_spin.setSingleStep(0.1); window.fr_zmax_spin.setValue(8.0); window.fr_zmax_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_zr.add(QtWidgets.QLabel("fit z:"), span=2)
    g_zr.add(window.fr_zmin_spin)
    g_zr.add(QtWidgets.QLabel("-"), span=1)
    g_zr.add(window.fr_zmax_spin)
    adv_layout.addLayout(g_zr.layout())

    # Advanced action buttons
    g_adv_btn = AutoGridPlacer(cols=4)
    window.fr_fit_btn = QtWidgets.QPushButton("Fit")
    window.fr_fit_btn.setToolTip("Fit folded basis for current molecule + substrate")
    window.fr_fit_btn.clicked.connect(lambda: _on_fit(window))
    g_adv_btn.add(window.fr_fit_btn)
    window.fr_save_fit_btn = QtWidgets.QPushButton("Save Fit")
    window.fr_save_fit_btn.setToolTip("Save fit parameters to .npz")
    window.fr_save_fit_btn.clicked.connect(lambda: _on_save_fit(window))
    g_adv_btn.add(window.fr_save_fit_btn)
    adv_layout.addLayout(g_adv_btn.layout())

    # Scan parameters
    g_scan = AutoGridPlacer(cols=4)
    window.fr_scan_span_spin = QtWidgets.QDoubleSpinBox(); window.fr_scan_span_spin.setRange(0.1, 50.0); window.fr_scan_span_spin.setSingleStep(0.1); window.fr_scan_span_spin.setValue(2.0); window.fr_scan_span_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_scan.add_pair("span:", window.fr_scan_span_spin)
    window.fr_scan_n_spin = QtWidgets.QSpinBox(); window.fr_scan_n_spin.setRange(2, 64); window.fr_scan_n_spin.setValue(5); window.fr_scan_n_spin.setMaximumWidth(50)
    g_scan.add_pair("n:", window.fr_scan_n_spin)
    window.fr_z_scan_spin = QtWidgets.QDoubleSpinBox(); window.fr_z_scan_spin.setRange(0.1, 50.0); window.fr_z_scan_spin.setSingleStep(0.1); window.fr_z_scan_spin.setValue(3.0); window.fr_z_scan_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_scan.add_pair("scan z:", window.fr_z_scan_spin)
    adv_layout.addLayout(g_scan.layout())

    # Scan buttons
    g_scan_btn = AutoGridPlacer(cols=4)
    window.fr_lat_scan_btn = QtWidgets.QPushButton("Lateral Scan")
    window.fr_lat_scan_btn.setToolTip("Lateral force map at fixed z")
    window.fr_lat_scan_btn.clicked.connect(lambda: _on_lateral_scan(window))
    g_scan_btn.add(window.fr_lat_scan_btn)
    window.fr_relaxed_scan_btn = QtWidgets.QPushButton("Relaxed Scan")
    window.fr_relaxed_scan_btn.setToolTip("Drag pinned atom along +x")
    window.fr_relaxed_scan_btn.clicked.connect(lambda: _on_relaxed_scan(window))
    g_scan_btn.add(window.fr_relaxed_scan_btn)
    adv_layout.addLayout(g_scan_btn.layout())

    adv.setContent(adv_widget)
    layout.addWidget(adv)
    layout.addStretch()

    # Register mode handlers
    _build_edit_mode_handlers(window)

    edit_modes = [
        ('FR Pin', lambda: window.set_edit_mode('fr_pin')),
        ('FR COM', lambda: window.set_edit_mode('fr_com')),
        ('FR Manip', lambda: window.set_edit_mode('fr_manip')),
    ]
    view_modes = [
        ('Show Substrate', lambda: _toggle_substrate(window)),
        ('Show Potential', lambda: _toggle_potential(window)),
    ]
    help_text = {
        'Load Fit': 'Load a pre-fitted .npz file and set up the rigid body.',
        'Run': 'Run n_iter dynamics steps (or until Fconv).',
        'Step': 'Run one dynamics step.',
        'x/y/z': 'Initial COM (z is relative to substrate top; ~2.5 Å for H2O/NaCl).',
        'k': 'Spring constant for the pinned atom in FR Manip mode.',
        'n_iter': 'Dynamics steps per continuous-run tick (default 100).',
        'dt': 'Time step for the rigid-body integrator.',
        'Fconv': 'Stop Run when total body force magnitude is below this.',
        'ld': 'Linear damping factor (0 = no damping, 1 = frozen).',
        'ad': 'Angular damping factor.',
        'Substrate': 'Toggle the substrate overlay in the 3D scene.',
        'Potential': 'Toggle the XY potential slice overlay (at current z) in the 3D scene.',
        'type': 'Atom type for potential slice (O = type 0, H = type 1).',
        'span': 'Half-width of the potential slice grid in Å.',
        'n': 'Grid resolution for potential slice.',
        'XZ Slice': 'Plot XZ side-view potential at current y (matplotlib dialog).',
        'YZ Slice': 'Plot YZ side-view potential at current x (matplotlib dialog).',
        'Fitting & Scans': 'Advanced: fit basis, save/load .npz, and run lateral/pinned scans.',
    }

    return UIComponents(panel=panel, edit_modes=edit_modes, view_modes=view_modes, help_text=help_text)


def prepare_folded_rigid(window, mol=None, fit=None, substrate=None, run=False, step=False, manip=False, n=None, dt=None, x=None, y=None, z=2.5):
    """Programmatically prepare the FoldedRigid extension for a run.

    This is the GUI-script equivalent of the user clicks: load molecule,
    load (or fit) substrate potential, optionally run/step.
    """
    from .gui_script_utils import load_molecule, set_spin_value, set_line_edit, process_events, expand_extension_panel
    from .gui_script_utils import set_edit_mode

    expand_extension_panel(window, 'folded_rigid', open=True)

    if x is not None:
        set_spin_value(window.fr_x_spin, x)
    if y is not None:
        set_spin_value(window.fr_y_spin, y)
    if z is not None:
        set_spin_value(window.fr_z_spin, z)
    if n is not None:
        set_spin_value(window.fr_niter_spin, n)
    if dt is not None:
        set_spin_value(window.fr_dt_spin, dt)

    if mol is not None:
        load_molecule(window, mol)
    if substrate is not None:
        set_line_edit(window.fr_substrate_edit, substrate)

    if fit is not None:
        _load_fit_path(window, fit)
    else:
        _on_fit(window)

    if run:
        _on_run(window)
    elif step:
        _on_step(window)

    if manip:
        set_edit_mode(window, 'fr_manip')

    process_events(window)
    return getattr(window, 'fr_rbd', None)
