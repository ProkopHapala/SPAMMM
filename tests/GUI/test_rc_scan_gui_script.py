"""L0 smoke test for RC scan GUI prepare script (offscreen Qt)."""
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


def test_prepare_rc_scan_review_offscreen(qapp):
    from spammm.GUI.SPAMMM_GUI import SPAMMMWindow
    from spammm.GUI.gui_script_runner import run_gui_script
    import os
    window = SPAMMMWindow()
    script = os.path.join(os.path.dirname(__file__), '..', '..', 'spammm', 'GUI', 'gui_scripts', 'rc_scan_review.py')
    ds = run_gui_script(window, script, ['--preview', '--no-cache', '--dx', '0.5'])
    assert ds.nframes >= 3
    assert ds.meta.get('scan_type') == 'pm_neb_preview'
    assert window.rc_dataset is ds
    assert window.rc_slider.maximum() == ds.nframes - 1
    assert window.rc_preview_apo is not None
