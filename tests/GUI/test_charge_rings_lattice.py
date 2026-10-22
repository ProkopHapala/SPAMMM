#!/usr/bin/env python3
"""L0+L2 GUI smoke: ChargeRings lattice mode builds, loads 7T preset, computes
map (npix=33) and cuts (nfrac=21, 3 voltages), saves figures to
debug/charge_rings_lattice/{map,cuts}.png.

Run: QT_QPA_PLATFORM=offscreen pytest tests/GUI/test_charge_rings_lattice.py -s
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PyQt5 import QtWidgets

PRESET = os.environ.get('SPAMMM_LATTICE_PRESET', '/home/prokop/git/SPAMMM_export_for_ChaoLi/results/temirov_gui_presets/temirov_7T.json')
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'debug', 'charge_rings_lattice')

pytestmark = pytest.mark.gpu


class _MockWindow:
    """Stand-in for SPAMMM_GUI window (no scene/backend needed for lattice mode)."""
    def __init__(self):
        self.backend = type('B', (), dict(sys=None, graph=None))()
        self._mode_handlers = {}
    def register_mode_handler(self, name, handler):
        self._mode_handlers[name] = handler
    def refresh_view(self):
        pass
    def statusBar(self):
        return type('SB', (), dict(showMessage=lambda *a, **k: None))()


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


@pytest.fixture
def cr_window(qapp):
    from spammm.GUI.ChargeRingsExtension import build_ui
    w = _MockWindow()
    w._ui = build_ui(w)  # keep panel alive — unparented QWidget would be GC'd
    return w


def test_lattice_map_and_cuts(cr_window):
    from spammm.GUI import ChargeRingsExtension as CR
    import matplotlib
    matplotlib.use('Agg')
    w = cr_window
    assert w.lat_map_btn is not None and w.lat_cuts_btn is not None
    CR.lat_load_json(w, PRESET)
    assert w._lat_params is not None and w._lat_params['model'] == 'lattice_pme8'
    # map at npix=33
    w.lat_spins['npix'].setValue(33)
    CR.lat_run_map(w)
    assert w._lat_last_scan is not None and np.isfinite(w._lat_last_scan['dIdV']).all()
    os.makedirs(OUTDIR, exist_ok=True)
    w._lat_map_dlg_fig.savefig(os.path.join(OUTDIR, 'map.png'), dpi=110)
    print(f'REVIEW: {os.path.join(OUTDIR, "map.png")}', flush=True)
    # cuts: nfrac=21, 3 voltages
    w.lat_spins['cut_nfrac'].setValue(21)
    w.lat_spins['cut_Vmin'].setValue(1.60)
    w.lat_spins['cut_Vmax'].setValue(1.70)
    CR.lat_run_cuts(w)
    assert w._lat_cut_lines is not None
    w._lat_cuts_dlg_fig.savefig(os.path.join(OUTDIR, 'cuts.png'), dpi=110)
    print(f'REVIEW: {os.path.join(OUTDIR, "cuts.png")}', flush=True)
    assert 'FAILED' not in w.lat_status.text()


def test_lattice_auto_update_and_solver_switch(cr_window):
    """(a) Auto-update fires on spin change; (b) pme4 auto-sets 2x2 + disables
    method/precision; (c) pme4 + n2=4 shows red error; (d) closed map dialog
    gets fresh artists on recompute."""
    from PyQt5.QtTest import QTest
    from spammm.GUI import ChargeRingsExtension as CR
    w = cr_window
    assert w.lat_live_chk.isChecked()  # Auto-update on by default
    CR.lat_load_json(w, PRESET)
    w.lat_spins['npix'].setValue(33)
    QTest.qWait(1500)
    assert w._lat_last_scan is not None and 'FAILED' not in w.lat_status.text()
    # (a) VBias spin change triggers an auto map recompute at the new bias
    V0 = w.lat_spins['VBias'].value()
    w.lat_spins['VBias'].setValue(V0 + 0.05)
    QTest.qWait(1500)
    assert abs(w._lat_last_scan['bias'] - (V0 + 0.05)) < 1e-9, f"auto-update did not recompute: {w._lat_last_scan['bias']} != {V0 + 0.05}"
    # (b) solver=pme4 auto-sets supercell 2x2, disables method/precision, npix->129, map OK
    w.lat_solver_combo.setCurrentText('pme4')
    assert w.lat_spins['n1'].value() == 2 and w.lat_spins['n2'].value() == 2
    assert w.lat_spins['npix'].value() == 129
    assert not w.lat_stationary_method_combo.isEnabled() and not w.lat_stationary_precision_combo.isEnabled()
    QTest.qWait(1500)
    assert 'FAILED' not in w.lat_status.text() and w._lat_last_scan is not None
    # (c) n2=4 with pme4 -> red error, no escaping exception
    w.lat_spins['n2'].setValue(4)
    QTest.qWait(1500)
    assert 'pme4 needs' in w.lat_status.text() and 'red' in w.lat_status.styleSheet()
    # restore: solver=pme8 re-enables combos
    w.lat_spins['n2'].setValue(2)
    w.lat_solver_combo.setCurrentText('pme8')
    assert w.lat_stationary_method_combo.isEnabled() and w.lat_stationary_precision_combo.isEnabled()
    QTest.qWait(1500)
    assert 'FAILED' not in w.lat_status.text()
    # (d) close map dialog -> cached artists reset; recompute creates a new figure
    old_dlg, old_fig = w._lat_map_dlg, w._lat_map_dlg_fig
    assert old_dlg is not None
    old_dlg.done(0)
    assert w._lat_map_dlg is None and w._lat_map_art is None and w._lat_map_dlg_fig is None
    CR.lat_run_map(w)
    assert w._lat_map_dlg is not None and w._lat_map_dlg_fig is not old_fig
    assert w._lat_map_art['ims'][0].figure is w._lat_map_dlg_fig


def test_lattice_hires_and_circles(cr_window):
    """Hi-res 257 map, large in-plane quadrupole, ref-R overlay circles, timings."""
    import time
    from PyQt5.QtTest import QTest
    from spammm.GUI import ChargeRingsExtension as CR
    w = cr_window
    CR.lat_load_json(w, PRESET)
    w.lat_spins['npix'].setValue(33)
    QTest.qWait(1500)  # settle the load-triggered auto map
    # pme4 switch -> 2x2 + npix 129
    w.lat_solver_combo.setCurrentText('pme4')
    assert w.lat_spins['npix'].value() == 129
    QTest.qWait(1500)
    t = time.monotonic(); CR.lat_run_map(w); t129 = time.monotonic() - t
    assert 'FAILED' not in w.lat_status.text()
    t = time.monotonic(); CR.lat_run_map(w, npix=257); t257 = time.monotonic() - t
    assert 'FAILED' not in w.lat_status.text()
    assert w._lat_last_scan['dIdV'].shape == (257, 257)
    print(f'TIME pme4 map npix=129: {t129:.2f}s   npix=257: {t257:.2f}s', flush=True)
    # large in-plane quadrupole + ref-R circles draw without error
    w.lat_spins['Qxy'].setValue(300)
    w.lat_spins['ring_ref_px'].setValue(20)
    CR.lat_run_map(w)
    assert 'FAILED' not in w.lat_status.text()
    assert any(l.get_linestyle() == '--' for l in w._lat_map_dlg_axes.flat[3].lines)
    # back to pme8 -> npix clamped to 65; time one pme8 map
    w.lat_solver_combo.setCurrentText('pme8')
    assert w.lat_spins['npix'].value() == 65
    QTest.qWait(1500)  # flush queued auto-map before the timed call
    t = time.monotonic(); CR.lat_run_map(w); t65 = time.monotonic() - t
    assert 'FAILED' not in w.lat_status.text()
    print(f'TIME pme8 map npix=65: {t65:.2f}s', flush=True)
