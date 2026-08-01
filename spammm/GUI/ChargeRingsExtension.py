"""ChargeRingsExtension — GUI for Pauli Master Equation charge-ring STM (PME.cl).

Panel: load/save JSON params, spinbox edits, Calc XY / xV / 1D with cut-line
overlay on xy and many-body state probability plots.

Uses ``spammm.quantum.pauli_scan`` + ``PauliSolverCL``. Sibling of AFMExtension
(not stuffed into FDBM). Audit: ``doc/TopicalAudit/ChargeRings_PME.md``.
"""
from __future__ import annotations

import json
import os
import numpy as np
from PyQt5 import QtWidgets, QtCore

from .ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH, AutoGridPlacer

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'charge_rings')

# Keys exposed as spinboxes (label, key, min, max, step, decimals)
_SPIN_SPEC = [
    ('nsite', 'nsite', 1, 4, 1, 0),
    ('radius', 'radius', 1.0, 20.0, 0.1, 2),
    ('phiRot', 'phiRot', -6.3, 6.3, 0.1, 3),
    ('Esite', 'Esite', -1.0, 1.0, 0.01, 3),
    ('W', 'W', 0.0, 0.5, 0.01, 3),
    ('Q0', 'Q0', 0.0, 5.0, 0.1, 2),
    ('Qzz', 'Qzz', -20.0, 20.0, 0.5, 2),
    ('VBias', 'VBias', 0.0, 3.0, 0.05, 3),
    ('z_tip', 'z_tip', 1.0, 15.0, 0.5, 2),
    ('zV0', 'zV0', -5.0, 5.0, 0.1, 2),
    ('Temp', 'Temp', 0.1, 100.0, 0.5, 2),
    ('GammaT', 'GammaT', 1e-4, 1.0, 0.01, 4),
    ('decay', 'decay', 0.05, 2.0, 0.05, 3),
    ('L', 'L', 5.0, 40.0, 1.0, 1),
    ('npix', 'npix', 20, 200, 10, 0),
    ('p1_x', 'p1_x', -40.0, 40.0, 0.5, 2),
    ('p1_y', 'p1_y', -40.0, 40.0, 0.5, 2),
    ('p2_x', 'p2_x', -40.0, 40.0, 0.5, 2),
    ('p2_y', 'p2_y', -40.0, 40.0, 0.5, 2),
    ('nx', 'nx', 20, 200, 10, 0),
    ('nV', 'nV', 20, 200, 10, 0),
    ('Vmin', 'Vmin', 0.0, 2.0, 0.05, 3),
    ('Vmax', 'Vmax', 0.1, 3.0, 0.05, 3),
]


def build_ui(window):
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    # --- Preset ---
    g0 = AutoGridPlacer(cols=4)
    window.cr_preset = QtWidgets.QComboBox()
    window.cr_preset.addItems(['symmetric_trimer', 'fig3_trimer', 'Ruslan_long', 'Ruslan_short', 'square_tetramer'])
    g0.add_pair("Preset:", window.cr_preset)
    window.cr_load_preset_btn = QtWidgets.QPushButton('Load preset')
    window.cr_load_preset_btn.clicked.connect(lambda: load_preset(window))
    g0.add(window.cr_load_preset_btn)
    layout.addLayout(g0.layout())

    # --- JSON I/O ---
    g1 = AutoGridPlacer(cols=4)
    window.cr_load_json_btn = QtWidgets.QPushButton('Load JSON…')
    window.cr_load_json_btn.clicked.connect(lambda: load_json(window))
    g1.add(window.cr_load_json_btn)
    window.cr_save_json_btn = QtWidgets.QPushButton('Save JSON…')
    window.cr_save_json_btn.clicked.connect(lambda: save_json(window))
    g1.add(window.cr_save_json_btn)
    layout.addLayout(g1.layout())

    # --- Param spins (scrollable) ---
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setMaximumHeight(280)
    spin_host = QtWidgets.QWidget()
    g_spins = AutoGridPlacer(cols=4)
    window.cr_spins = {}
    for label, key, lo, hi, step, dec in _SPIN_SPEC:
        if dec == 0:
            sp = QtWidgets.QSpinBox()
            sp.setRange(int(lo), int(hi))
            sp.setSingleStep(int(step))
        else:
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
        sp.setMaximumWidth(90)
        window.cr_spins[key] = sp
        g_spins.add_pair(label, sp)
    spin_host.setLayout(g_spins.layout())
    scroll.setWidget(spin_host)
    layout.addWidget(scroll)

    window.cr_bMirror = QtWidgets.QCheckBox('bMirror')
    window.cr_bMirror.setChecked(True)
    window.cr_bRamp = QtWidgets.QCheckBox('bRamp')
    window.cr_bRamp.setChecked(True)
    g_chk = AutoGridPlacer(cols=4)
    g_chk.add(window.cr_bMirror)
    g_chk.add(window.cr_bRamp)
    layout.addLayout(g_chk.layout())

    # --- Calc buttons ---
    g_calc = AutoGridPlacer(cols=4)
    window.cr_xy_btn = QtWidgets.QPushButton('Calc XY')
    window.cr_xy_btn.setToolTip('Constant-VBias xy STM + dI/dV; overlay xV cut line')
    window.cr_xy_btn.clicked.connect(lambda: calc_xy(window))
    g_calc.add(window.cr_xy_btn)
    window.cr_xv_btn = QtWidgets.QPushButton('Calc xV')
    window.cr_xv_btn.setToolTip('Line×voltage scan (diamonds / NDR) + state probs')
    window.cr_xv_btn.clicked.connect(lambda: calc_xv(window))
    g_calc.add(window.cr_xv_btn)
    window.cr_1d_btn = QtWidgets.QPushButton('Calc 1D')
    window.cr_1d_btn.setToolTip('Fixed-V line cut + many-body P(s)')
    window.cr_1d_btn.clicked.connect(lambda: calc_1d(window))
    g_calc.add(window.cr_1d_btn)
    layout.addLayout(g_calc.layout())

    window.cr_status = QtWidgets.QLabel('Status: Load preset or JSON, then Calc')
    window.cr_status.setWordWrap(True)
    window.cr_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    layout.addWidget(window.cr_status)
    layout.addStretch()

    window._cr_solver = None
    window._cr_xy = None
    window._cr_xv = None
    load_preset(window)  # default symmetric_trimer
    return UIComponents(panel=panel, edit_modes=[], view_modes=[])


def _set_status(window, msg):
    window.cr_status.setText(f'Status: {msg}')


def params_from_widgets(window):
    p = {}
    for key, sp in window.cr_spins.items():
        p[key] = sp.value()
    p['bMirror'] = window.cr_bMirror.isChecked()
    p['bRamp'] = window.cr_bRamp.isChecked()
    # scan helpers expect these names
    p['zVd'] = p.get('zVd', 20.0)
    if 'zVd' not in window.cr_spins:
        p['zVd'] = getattr(window, '_cr_zVd', 20.0)
    p['Rtip'] = getattr(window, '_cr_Rtip', 3.0)
    p['GammaS'] = p.get('GammaT', 0.01)
    p['dQ'] = 0.02
    p['zQd'] = 0.0
    p['phi0_ax'] = getattr(window, '_cr_phi0_ax', 0.0)
    return p


def params_to_widgets(window, params):
    for key, sp in window.cr_spins.items():
        if key not in params:
            continue
        v = params[key]
        sp.blockSignals(True)
        if isinstance(sp, QtWidgets.QSpinBox):
            sp.setValue(int(v))
        else:
            sp.setValue(float(v))
        sp.blockSignals(False)
    if 'bMirror' in params:
        window.cr_bMirror.setChecked(bool(params['bMirror']))
    if 'bRamp' in params:
        window.cr_bRamp.setChecked(bool(params['bRamp']))
    window._cr_zVd = float(params.get('zVd', 20.0))
    window._cr_Rtip = float(params.get('Rtip', 3.0))
    window._cr_phi0_ax = float(params.get('phi0_ax', 0.0))
    # ensure nx/nV/Vmin/Vmax defaults
    if 'nx' not in params and 'nx' in window.cr_spins:
        window.cr_spins['nx'].setValue(100)
    if 'nV' not in params and 'nV' in window.cr_spins:
        window.cr_spins['nV'].setValue(80)
    if 'Vmin' not in params and 'Vmin' in window.cr_spins:
        window.cr_spins['Vmin'].setValue(0.0)
    if 'Vmax' not in params and 'Vmax' in window.cr_spins:
        window.cr_spins['Vmax'].setValue(float(params.get('VBias', 0.85)))


def load_preset(window):
    from spammm.quantum import pauli_scan as ps
    name = window.cr_preset.currentText()
    try:
        if name == 'symmetric_trimer':
            p = ps.symmetric_trimer_params()
            path = ps.default_geometry_path('symmetric_trimer.json')
            if os.path.isfile(path):
                p.update(ps.load_json_params(path))
        elif name == 'fig3_trimer':
            p = ps.fig3_trimer_params()
        elif name.startswith('Ruslan'):
            geom = ps.default_geometry_path(f'{name}.txt')
            p = ps.ruslan_default_params(geometry_file=geom, nsite=2)
            spos, _, _ = ps.load_site_geometry(geom)
            p['nsite'] = len(spos)
        elif name == 'square_tetramer':
            geom = ps.default_geometry_path('square_tetramer.txt')
            # NDR regime: match fig3_trimer params (Qzz=0, low VBias, low Temp)
            # ruslan_default_params has VBias=2.0 which is too high — washes out NDR
            p = ps.ruslan_default_params(geometry_file=geom, nsite=4, Qzz=0.0, W=0.05,
                                         VBias=0.85, Temp=2.6, z_tip=6.0, zV0=-0.9, zVd=20.0)
        else:
            p = ps.symmetric_trimer_params()
        p.setdefault('nx', 100)
        p.setdefault('nV', 80)
        p.setdefault('Vmin', 0.0)
        p.setdefault('Vmax', float(p.get('VBias', 0.85)))
        params_to_widgets(window, p)
        window._cr_params_extra = {k: v for k, v in p.items() if k not in window.cr_spins}
        _set_status(window, f'Loaded preset {name}')
    except Exception as e:
        _set_status(window, f'Preset FAILED: {e}')
        import traceback; traceback.print_exc()


def load_json(window):
    path, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Load charge-rings JSON', _DATA, 'JSON (*.json)')
    if not path:
        return
    from spammm.quantum import pauli_scan as ps
    try:
        p = ps.load_json_params(path)
        p.setdefault('nx', 100); p.setdefault('nV', 80)
        p.setdefault('Vmin', 0.0); p.setdefault('Vmax', float(p.get('VBias', 1.0)))
        params_to_widgets(window, p)
        window._cr_params_extra = p
        _set_status(window, f'Loaded {os.path.basename(path)}')
    except Exception as e:
        _set_status(window, f'Load FAILED: {e}')


def save_json(window):
    path, _ = QtWidgets.QFileDialog.getSaveFileName(window, 'Save charge-rings JSON', _DATA, 'JSON (*.json)')
    if not path:
        return
    p = params_from_widgets(window)
    extra = getattr(window, '_cr_params_extra', {}) or {}
    for k in ('zVd', 'Rtip', 'phi0_ax', 'geometry_file', 'GammaS', 'dQ', 'zQd'):
        if k in extra:
            p[k] = extra[k]
    p['zVd'] = getattr(window, '_cr_zVd', p.get('zVd', 20.0))
    p['Rtip'] = getattr(window, '_cr_Rtip', 3.0)
    p['phi0_ax'] = getattr(window, '_cr_phi0_ax', 0.0)
    # drop GUI-only keys from core file if desired — keep nx/nV/Vmin/Vmax for reload
    try:
        with open(path, 'w') as f:
            json.dump(p, f, indent=2)
        _set_status(window, f'Saved {os.path.basename(path)}')
    except Exception as e:
        _set_status(window, f'Save FAILED: {e}')


def _get_solver(window):
    if getattr(window, '_cr_solver', None) is None:
        from spammm.quantum.PauliSolverCL import PauliSolverCL
        window._cr_solver = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=False)
        name = window._cr_solver.ctx.devices[0].name
        _set_status(window, f'OpenCL: {name}')
    return window._cr_solver


def _site_geom(window, params):
    from spammm.quantum import pauli_scan as ps
    if params.get('geometry_file'):
        return ps.load_site_geometry(params['geometry_file'])
    # circle from widgets
    return ps.make_site_geom(params)


def _full_params(window):
    p = params_from_widgets(window)
    extra = getattr(window, '_cr_params_extra', {}) or {}
    p['zVd'] = getattr(window, '_cr_zVd', extra.get('zVd', 20.0))
    p['Rtip'] = getattr(window, '_cr_Rtip', extra.get('Rtip', 3.0))
    p['phi0_ax'] = getattr(window, '_cr_phi0_ax', extra.get('phi0_ax', 0.0))
    if 'geometry_file' in extra:
        p['geometry_file'] = extra['geometry_file']
    p['GammaS'] = p.get('GammaT', 0.01)
    p['dQ'] = 0.02
    p['zQd'] = 0.0
    return p


def calc_xy(window):
    from spammm.quantum import pauli_scan as ps
    from .plotutils import show_in_plot_window
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.figure import Figure
    try:
        params = _full_params(window)
        spos, rots, _ = _site_geom(window, params)
        solver = _get_solver(window)
        _set_status(window, 'Running xy…')
        QtWidgets.QApplication.processEvents()
        xy = ps.scan_xy(solver, spos, rots, params, return_probs=False)
        window._cr_xy = xy
        fig = Figure(figsize=(9, 4))
        ax0 = fig.add_subplot(1, 2, 1)
        ax1 = fig.add_subplot(1, 2, 2)
        im0 = ax0.imshow(xy['STM'], origin='lower', extent=xy['extent'], cmap='inferno')
        ax0.plot(spos[:, 0], spos[:, 1], 'c+', ms=10, mew=1.5)
        # cut line from p1→p2
        ax0.plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'w-', lw=1.5, label='xV/1D cut')
        ax0.legend(loc='upper right', fontsize=8)
        ax0.set_title(f"STM V={params['VBias']:.3f}")
        ax0.set_xlabel('x [Å]'); ax0.set_ylabel('y [Å]')
        fig.colorbar(im0, ax=ax0, fraction=0.046)
        if xy['dIdV'] is not None:
            sc = max(np.nanmax(np.abs(xy['dIdV'])), 1e-30)
            im1 = ax1.imshow(xy['dIdV'], origin='lower', extent=xy['extent'], cmap='bwr', vmin=-sc, vmax=sc)
            ax1.plot(spos[:, 0], spos[:, 1], 'k+', ms=10, mew=1.5)
            ax1.plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'k-', lw=1.5)
            ax1.set_title('dI/dV (rings / NDR)')
            ax1.set_xlabel('x [Å]')
            fig.colorbar(im1, ax=ax1, fraction=0.046)
        fig.tight_layout()
        show_in_plot_window(window, fig, title='ChargeRings XY', attr='_cr_xy_window')
        _set_status(window, f"XY done Imax={xy['STM'].max():.3e}")
    except Exception as e:
        _set_status(window, f'XY FAILED: {e}')
        import traceback; traceback.print_exc()


def calc_xv(window):
    from spammm.quantum import pauli_scan as ps
    from .plotutils import show_in_plot_window
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.figure import Figure
    try:
        params = _full_params(window)
        spos, rots, _ = _site_geom(window, params)
        solver = _get_solver(window)
        nx = int(params.get('nx', 100))
        nV = int(params.get('nV', 80))
        Vmin = float(params.get('Vmin', 0.0))
        Vmax = float(params.get('Vmax', params['VBias']))
        _set_status(window, 'Running xV…')
        QtWidgets.QApplication.processEvents()
        xv = ps.scan_xV(solver, spos, rots, params, nx=nx, nV=nV, Vmin=Vmin, Vmax=Vmax, return_probs=True)
        window._cr_xv = xv

        fig = Figure(figsize=(8, 8))
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 1, 2)
        # use distance along cut for x axis
        x0, x1 = 0.0, xv['dist']
        ext = [x0, x1, Vmin, Vmax]
        im0 = ax0.imshow(xv['STM'], origin='lower', extent=ext, aspect='auto', cmap='inferno')
        ax0.set_ylabel('V [V]'); ax0.set_title('xV STM')
        fig.colorbar(im0, ax=ax0, fraction=0.046)
        sc = 0.5 * max(np.nanmax(np.abs(xv['dIdV'])), 1e-30)
        im1 = ax1.imshow(xv['dIdV'], origin='lower', extent=ext, aspect='auto', cmap='bwr', vmin=-sc, vmax=sc)
        ax1.set_xlabel('distance along cut [Å]'); ax1.set_ylabel('V [V]')
        ax1.set_title(f"dI/dV  NDR min={xv['dIdV'].min():.2e}")
        fig.colorbar(im1, ax=ax1, fraction=0.046)
        fig.tight_layout()
        show_in_plot_window(window, fig, title='ChargeRings xV', attr='_cr_xv_window')

        # state probabilities (active subspace masks 0..2^n_act-1)
        if xv.get('probs') is not None:
            _plot_state_probs_xV(window, xv)
        _set_status(window, f"xV done Imax={xv['STM'].max():.3e} NDRmin={xv['dIdV'].min():.2e}")
    except Exception as e:
        _set_status(window, f'xV FAILED: {e}')
        import traceback; traceback.print_exc()


def _plot_state_probs_xV(window, xv):
    from .plotutils import show_in_plot_window
    from matplotlib.figure import Figure
    P = xv['probs']  # [nV, nx, nStates]
    n_act = int(xv.get('n_active', 3))
    n_show = 1 << n_act  # only states without spectator occupation
    nV, nx, nSt = P.shape
    ncol = min(4, n_show)
    nrow = (n_show + ncol - 1) // ncol
    fig = Figure(figsize=(2.4 * ncol, 2.0 * nrow))
    ext = [0.0, xv['dist'], xv['Vbiases'][0], xv['Vbiases'][-1]]
    for s in range(n_show):
        ax = fig.add_subplot(nrow, ncol, s + 1)
        ax.imshow(P[:, :, s], origin='lower', extent=ext, aspect='auto', cmap='magma', vmin=0, vmax=1)
        ax.set_title(format(s, f'0{n_act}b'), fontsize=9)
        if s % ncol == 0:
            ax.set_ylabel('V')
        if s // ncol == nrow - 1:
            ax.set_xlabel('s [Å]')
    fig.suptitle('Many-body state probabilities P(s,V)', fontsize=11)
    fig.tight_layout()
    show_in_plot_window(window, fig, title='ChargeRings state probs (xV)', attr='_cr_prob_window')


def calc_1d(window):
    from spammm.quantum import pauli_scan as ps
    from .plotutils import show_in_plot_window
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.figure import Figure
    try:
        params = _full_params(window)
        spos, rots, _ = _site_geom(window, params)
        solver = _get_solver(window)
        nx = int(params.get('nx', 100))
        _set_status(window, 'Running 1D…')
        QtWidgets.QApplication.processEvents()
        r = ps.scan_1d(solver, spos, rots, params, nx=nx, return_probs=True)

        fig = Figure(figsize=(8, 6))
        ax0 = fig.add_subplot(2, 1, 1)
        ax0.plot(r['dist_axis'], r['I'], 'k-')
        ax0.set_ylabel('I'); ax0.set_title(f"1D cut V={r['VBias']:.3f}  ({r['start']}→{r['end']})")
        ax0.grid(True, alpha=0.3)
        ax1 = fig.add_subplot(2, 1, 2)
        n_act = int(r.get('n_active', 3))
        n_show = 1 << n_act
        P = r['probs']
        for s in range(n_show):
            ax1.plot(r['dist_axis'], P[:, s], label=format(s, f'0{n_act}b'))
        ax1.set_xlabel('distance along cut [Å]'); ax1.set_ylabel('P_state')
        ax1.legend(ncol=4, fontsize=7, loc='upper right')
        ax1.set_title('Many-body state probabilities')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        show_in_plot_window(window, fig, title='ChargeRings 1D + probs', attr='_cr_1d_window')

        # also refresh xy with cut if we have a previous xy at same V, else light xy for overlay context
        if window._cr_xy is None or abs(params['VBias'] - float(getattr(window, '_cr_xy_V', -1))) > 1e-6:
            xy = ps.scan_xy(solver, spos, rots, params)
            window._cr_xy = xy
            window._cr_xy_V = params['VBias']
        _show_xy_with_cut(window, window._cr_xy, spos, params)
        _set_status(window, f"1D done Imax={r['I'].max():.3e}")
    except Exception as e:
        _set_status(window, f'1D FAILED: {e}')
        import traceback; traceback.print_exc()


def _show_xy_with_cut(window, xy, spos, params):
    from .plotutils import show_in_plot_window
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(xy['STM'], origin='lower', extent=xy['extent'], cmap='inferno')
    ax.plot(spos[:, 0], spos[:, 1], 'c+', ms=10, mew=1.5)
    ax.plot([params['p1_x'], params['p2_x']], [params['p1_y'], params['p2_y']], 'w-', lw=2, label='1D/xV cut')
    ax.legend(loc='upper right')
    ax.set_title('XY + cut line'); ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout()
    show_in_plot_window(window, fig, title='ChargeRings cut on XY', attr='_cr_cut_window')
