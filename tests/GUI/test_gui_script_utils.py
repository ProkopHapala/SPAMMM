"""L0 tests for programmatic GUI helpers."""
import os
import pytest

pytest.importorskip('PyQt5')


@pytest.fixture(scope='module')
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_set_slider_value_blocks_signals(qapp):
    from PyQt5.QtWidgets import QSlider
    from spammm.GUI.gui_script_utils import set_slider_value
    calls = []
    slider = QSlider()
    slider.valueChanged.connect(calls.append)
    set_slider_value(slider, 3)
    assert slider.value() == 3
    assert calls == []


def test_configure_scan_single_hbond_mapping(qapp):
    from spammm.GUI.SPAMMM_GUI import SPAMMMWindow
    from spammm.GUI.ReactionCoordinateExtension import configure_scan, _scan_hbonds
    from spammm.quantum.hbond_scan import build_ascii_hbond_system
    window = SPAMMMWindow()
    window.backend.sys = build_ascii_hbond_system('2Quinolone')
    window.refresh_view()
    configure_scan(window, pair=0, all_hbonds=False)
    assert len(window.rc_hbonds) >= 2
    assert len(window.rc_mapping) == 1
    assert len(_scan_hbonds(window)) == len(window.rc_mapping)
