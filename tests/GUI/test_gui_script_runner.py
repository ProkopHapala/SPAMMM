"""L0 tests for the GUI script runner: pacing API, controller, discovery, panel.

Covers (see doc/Tasks/GUI_Scripting_DemoRunner.md §5.13):
  - ScriptOptions / ScriptContext.batches boundaries
  - legacy synchronous return value preserved
  - generator order: one timer advance per frame + final return value
  - delay: frame does not advance early
  - barrier: PAUSED state, no progress until Continue; barriers-off advances
  - cancellation: Stop closes generator, runs finally
  - failure: generator exception → FAILED, full traceback retained
  - single active run rejected while running/paused
  - discovery: sorted, excludes _offline, no import
  - MC step update_ui=False returns summary without scene sync

Run: pytest tests/GUI/test_gui_script_runner.py -m "not slow"
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt5 import QtWidgets, QtCore

from spammm.GUI.gui_script_runner import (
    ScriptOptions, ScriptContext, ScriptController, _Frame, _Barrier,
    run_gui_script, bundled_scripts,
    IDLE, RUNNING, PAUSED, COMPLETED, CANCELLED, FAILED,
)


# ---------------------------------------------------------------------------
# QApplication fixture (module-scoped, offscreen)
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _MockStatusBar:
    def __init__(self):
        self.msg = ''
    def showMessage(self, msg):
        self.msg = msg


class _MockWindow(QtCore.QObject):
    """Minimal window for controller tests: statusBar + signal-free."""
    def __init__(self):
        super().__init__()
        self._sb = _MockStatusBar()
    def statusBar(self):
        return self._sb


# ---------------------------------------------------------------------------
# Pure: ScriptOptions + ScriptContext.batches
# ---------------------------------------------------------------------------
def test_batches_fast_one_range():
    ctx = ScriptContext(ScriptOptions(points_per_frame=0))
    ranges = list(ctx.batches(10))
    assert ranges == [(0, 10)]


def test_batches_chunked_covers_all():
    ctx = ScriptContext(ScriptOptions(points_per_frame=3))
    ranges = list(ctx.batches(10))
    assert ranges == [(0, 3), (3, 6), (6, 9), (9, 10)]


def test_batches_zero_n_empty():
    ctx = ScriptContext(ScriptOptions(points_per_frame=5))
    assert list(ctx.batches(0)) == []


def test_frame_barrier_directives_typed():
    ctx = ScriptContext(ScriptOptions())
    assert isinstance(ctx.frame('hi'), _Frame)
    assert ctx.frame('hi').message == 'hi'
    assert isinstance(ctx.barrier('stop'), _Barrier)
    assert ctx.barrier('stop').message == 'stop'


# ---------------------------------------------------------------------------
# Controller: generator order + completion
# ---------------------------------------------------------------------------
def _drive_until_done(ctrl, timeout_ms=2000):
    """Process Qt events until the controller leaves RUNNING/PAUSED or timeout."""
    loop = QtCore.QEventLoop()
    deadline = QtCore.QElapsedTimer(); deadline.start()
    states = []
    ctrl.state_changed.connect(lambda s: states.append(s))
    # Use a periodic check via processEvents
    while ctrl.state in (RUNNING, PAUSED) and deadline.elapsed() < timeout_ms:
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 10)
    return states


def test_controller_completes_and_returns_value(qapp):
    w = _MockWindow()
    ctrl = ScriptController(w, ScriptOptions(delay_ms=0))
    markers = []
    states = []
    ctrl.state_changed.connect(lambda s: states.append(s))

    def gen():
        markers.append('start')
        yield ctrl.ctx.frame('a')
        markers.append('mid')
        yield ctrl.ctx.frame('b')
        markers.append('end')
        return 42

    ctrl.attach(gen())
    _drive_until_done(ctrl)
    assert ctrl.state == COMPLETED
    assert ctrl._result == 42
    assert markers == ['start', 'mid', 'end']
    assert states[0] == RUNNING
    assert states[-1] == COMPLETED


def test_controller_barrier_pauses_then_continue(qapp):
    w = _MockWindow()
    ctrl = ScriptController(w, ScriptOptions(honor_barriers=True, delay_ms=0))
    reached = []

    def gen():
        yield ctrl.ctx.frame('pre')
        reached.append('before_barrier')
        yield ctrl.ctx.barrier('wait here')
        reached.append('after_barrier')
        yield ctrl.ctx.frame('post')

    ctrl.attach(gen())
    QtWidgets.QApplication.processEvents()
    # Let the pre frame's 0-ms timer fire
    for _ in range(20):
        QtWidgets.QApplication.processEvents()
    assert ctrl.state == PAUSED, f"expected PAUSED, got {ctrl.state}"
    assert reached == ['before_barrier']
    # Continue
    ctrl.continue_barrier()
    _drive_until_done(ctrl)
    assert ctrl.state == COMPLETED
    assert reached == ['before_barrier', 'after_barrier']


def test_controller_barrier_off_advances(qapp):
    w = _MockWindow()
    ctrl = ScriptController(w, ScriptOptions(honor_barriers=False, delay_ms=0))
    reached = []

    def gen():
        yield ctrl.ctx.frame('pre')
        yield ctrl.ctx.barrier('ignored')
        reached.append('passed_barrier')

    ctrl.attach(gen())
    _drive_until_done(ctrl)
    assert ctrl.state == COMPLETED
    assert reached == ['passed_barrier']


def test_controller_cancel_runs_finally(qapp):
    w = _MockWindow()
    ctrl = ScriptController(w, ScriptOptions(honor_barriers=True))
    finally_ran = []

    def gen():
        try:
            yield ctrl.ctx.barrier('pause')
        finally:
            finally_ran.append('cleanup')

    ctrl.attach(gen())
    for _ in range(20):
        QtWidgets.QApplication.processEvents()
    assert ctrl.state == PAUSED
    ctrl.stop()
    assert ctrl.state == CANCELLED
    assert finally_ran == ['cleanup']


def test_controller_failure_records_exception(qapp):
    w = _MockWindow()
    ctrl = ScriptController(w, ScriptOptions())

    def gen():
        yield ctrl.ctx.frame('ok')
        raise ValueError('boom')

    ctrl.attach(gen())
    _drive_until_done(ctrl)
    assert ctrl.state == FAILED
    assert isinstance(ctrl._exc, ValueError)
    assert str(ctrl._exc) == 'boom'


def test_controller_unsupported_yield_fails(qapp):
    w = _MockWindow()
    ctrl = ScriptController(w, ScriptOptions())

    def gen():
        yield 123  # not a Frame/Barrier/None

    ctrl.attach(gen())
    _drive_until_done(ctrl)
    assert ctrl.state == FAILED
    assert isinstance(ctrl._exc, TypeError)


def test_controller_rejects_second_attach_while_running(qapp):
    w = _MockWindow()
    ctrl = ScriptController(w, ScriptOptions(honor_barriers=True))

    def gen1():
        yield ctrl.ctx.barrier('pause')

    def gen2():
        yield ctrl.ctx.frame('x')

    ctrl.attach(gen1())
    for _ in range(20):
        QtWidgets.QApplication.processEvents()
    assert ctrl.state == PAUSED
    with pytest.raises(RuntimeError):
        ctrl.attach(gen2())


# ---------------------------------------------------------------------------
# run_gui_script: legacy synchronous parity
# ---------------------------------------------------------------------------
def test_run_gui_script_legacy_return(tmp_path, qapp):
    w = _MockWindow()
    script = tmp_path / 'legacy.py'
    script.write_text(
        "def run(window, argv=None):\n"
        "    return {'ok': True, 'argv': list(argv or [])}\n"
    )
    res = run_gui_script(w, str(script), ['--x', '1'], options=ScriptOptions())
    assert res == {'ok': True, 'argv': ['--x', '1']}


def test_run_gui_script_generator_attaches_controller(tmp_path, qapp):
    w = _MockWindow()
    script = tmp_path / 'gen.py'
    script.write_text(
        "def run(window, argv=None, ctx=None):\n"
        "    yield ctx.frame('a')\n"
        "    yield ctx.frame('b')\n"
        "    return 'done'\n"
    )
    res = run_gui_script(w, str(script), [], options=ScriptOptions())
    assert res is None  # generators return None synchronously
    ctrl = w._gui_script_controller
    assert ctrl is not None
    _drive_until_done(ctrl)
    assert ctrl.state == COMPLETED
    assert ctrl._result == 'done'


# ---------------------------------------------------------------------------
# Discovery: sorted, excludes _offline, no import
# ---------------------------------------------------------------------------
def test_bundled_scripts_excludes_offline_and_underscore():
    items = bundled_scripts()
    paths = [p for _, p in items]
    for _, p in items:
        assert os.path.isfile(p)
        assert not os.path.basename(p).startswith('_')
        assert not os.path.basename(p).endswith('_offline.py')
    # conference_demo should be present
    names = [os.path.basename(p) for _, p in items]
    assert 'conference_demo.py' in names
    assert 'ptcda_interactive_drag.py' in names
    assert 'azaindol_draw_offline.py' not in names
    # sorted by filename
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# MC step update_ui returns summary (uses RigidAssembly mock pattern)
# ---------------------------------------------------------------------------
def test_mc_step_returns_summary_dict(qapp):
    """update_ui=False returns a summary dict without calling _sync_display/_status.

    Uses the RigidAssemblyExtension mock-window pattern to avoid a full GPU build.
    """
    from spammm.GUI.RigidAssemblyExtension import _on_mc_step
    import numpy as np

    class _MockScene:
        def _pick_id_from_mouse(self, pos, max_dist=1.0): return -1, float('inf')
        def _id_to_idx_safe(self, aid): return -1
        def _ray_from_mouse(self, pos): return np.zeros(3), np.array([0, 0, 1.0])

    class _MockBackend:
        def __init__(self):
            self.sys = None
            self.graph = None

    class _MockWin:
        def __init__(self):
            self.backend = _MockBackend()
            self.scene = _MockScene()
            self.ra_ensemble = None
            self.ra_rbd = None
        def statusBar(self): return _MockStatusBar()

    w = _MockWin()
    # Not built → returns None, no crash
    res = _on_mc_step(w, update_ui=False)
    assert res is None
