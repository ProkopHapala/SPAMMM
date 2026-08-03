"""gui_script_runner.py — load and drive GUI control scripts with optional pacing.

Two script contracts, both loaded after ``window.show()``:

- **Legacy synchronous** — ``def run(window, argv=None)`` returning a value. Used by
  ``./run_gui.sh --script`` and existing scripts (rc_scan_review, azaindol_draw_demo).
  The return value is preserved for tests like ``test_rc_scan_gui_script.py``.
- **Paced generator** — ``def run(window, argv=None, ctx=None)`` containing ``yield``.
  The runner attaches the generator to a single-shot ``QTimer`` controller on the Qt
  main thread. Each ``yield ctx.frame(...)`` / ``yield ctx.barrier(...)`` is one
  repaint/control boundary. This is the cooperative model: arbitrary Python/Qt/OpenCL
  work runs between yields; the runner never preem a kernel or a Qt call.

Scripts live in ``demos/gui_scripts/`` (centralized). ``bundled_scripts()`` auto-discovers
them for the Scripts → Bundled menu and the Script Runner panel. No manual registration.

Design (see ``doc/Tasks/GUI_Scripting_DemoRunner.md``):
- ``ScriptOptions`` is the SSOT for presentation pacing (delay, points-per-frame,
  barriers). Defaults are fast mode.
- ``ScriptContext`` exposes only ``batches()``, ``frame()``, ``barrier()``. There is
  deliberately no ``chunk()`` (work has already run by the time of yield) and no
  ``sleep()`` (use ``yield ctx.frame(..., delay_ms=N)``).
- One controller per window (``window._gui_script_controller``); states
  IDLE/RUNNING/PAUSED/COMPLETED/CANCELLED/FAILED. Continue/Stop are shared by the
  Script Runner extension panel, the Scripts menu, and the F8 shortcut.
- Backward compatible: a non-generator ``run`` still returns its value synchronously.
"""
import importlib.util
import inspect
import os
import shlex
import sys
import traceback

from PyQt5 import QtCore, QtWidgets


# ---------------------------------------------------------------------------
# ScriptOptions — immutable presentation pacing SSOT
# ---------------------------------------------------------------------------
class ScriptOptions:
    """Presentation pacing for a script launch. Independent of script argv.

    Defaults are fast mode: one batch, no delay, barriers ignored.
    """
    __slots__ = ('delay_ms', 'points_per_frame', 'honor_barriers')

    def __init__(self, delay_ms=0, points_per_frame=0, honor_barriers=False):
        self.delay_ms = int(delay_ms)
        self.points_per_frame = int(points_per_frame)
        self.honor_barriers = bool(honor_barriers)


# ---------------------------------------------------------------------------
# Yielded directives — small typed objects so mistakes fail loud
# ---------------------------------------------------------------------------
class _Frame:
    """One repaint/control boundary. Yielded by ``ctx.frame(...)``."""
    __slots__ = ('message', 'delay_ms')

    def __init__(self, message=None, delay_ms=None):
        self.message = message
        self.delay_ms = delay_ms


class _Barrier:
    """Pause until Continue. Yielded by ``ctx.barrier(...)``."""
    __slots__ = ('message',)

    def __init__(self, message=None):
        self.message = message


# ---------------------------------------------------------------------------
# ScriptContext — minimal and precise pacing API passed to generator scripts
# ---------------------------------------------------------------------------
class ScriptContext:
    """Pacing context created by the runner and passed to the script.

    ``batches(n)`` is a generator yielding ``(i0, i1)`` ranges. ``points_per_frame<=0``
    gives one ``(0, n)`` range; otherwise ranges have at most that many points. The
    script defines what a point means (MC step, MD step, tile, iteration).
    """
    __slots__ = ('delay_ms', 'points_per_frame', 'honor_barriers')

    def __init__(self, options):
        self.delay_ms = options.delay_ms
        self.points_per_frame = options.points_per_frame
        self.honor_barriers = options.honor_barriers

    def batches(self, n):
        """Yield (i0, i1) index ranges covering [0, n)."""
        n = int(n)
        if n <= 0:
            return
        step = self.points_per_frame if self.points_per_frame > 0 else n
        i = 0
        while i < n:
            j = min(i + step, n)
            yield (i, j)
            i = j

    def frame(self, message=None, delay_ms=None):
        """Yield this for one repaint/control boundary."""
        return _Frame(message=message, delay_ms=delay_ms)

    def barrier(self, message=None):
        """Yield this to pause until Continue (only when honor_barriers is set)."""
        return _Barrier(message=message)


# ---------------------------------------------------------------------------
# Controller states
# ---------------------------------------------------------------------------
IDLE = 'IDLE'
RUNNING = 'RUNNING'
PAUSED = 'PAUSED'
COMPLETED = 'COMPLETED'
CANCELLED = 'CANCELLED'
FAILED = 'FAILED'


# ---------------------------------------------------------------------------
# ScriptController — drives one generator on a single-shot QTimer
# ---------------------------------------------------------------------------
class ScriptController(QtCore.QObject):
    """Drives one generator script on the Qt main thread.

    States: IDLE → RUNNING ↔ PAUSED → COMPLETED / CANCELLED / FAILED.
    Emits ``state_changed(state)``, ``finished(value)``, ``failed(exc)``,
    ``cancelled()`` for tests and future UI.
    """
    state_changed = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(object)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, window, options=None):
        super().__init__(window)
        self.window = window
        self.options = options or ScriptOptions()
        self.ctx = ScriptContext(self.options)
        self._gen = None
        self._state = IDLE
        self._result = None
        self._exc = None
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_tick)

    # ---- state ---------------------------------------------------------
    @property
    def state(self):
        return self._state

    def _set_state(self, s):
        if self._state == s:
            return
        self._state = s
        self.state_changed.emit(s)

    # ---- lifecycle -----------------------------------------------------
    def attach(self, gen):
        """Attach a freshly created generator and arm the first 0-ms tick."""
        if self._state in (RUNNING, PAUSED):
            raise RuntimeError(f"Script already {self._state}; stop first")
        self._gen = gen
        self._result = None
        self._exc = None
        self._set_state(RUNNING)
        self._timer.start(0)

    def continue_barrier(self):
        """Advance a paused barrier (shared by panel, menu, F8)."""
        if self._state != PAUSED:
            return
        self._set_state(RUNNING)
        self._timer.start(0)

    def stop(self):
        """Request cancellation. Closes the generator so its finally blocks run."""
        if self._state in (IDLE, COMPLETED, CANCELLED, FAILED):
            return
        self._timer.stop()
        if self._gen is not None:
            try:
                self._gen.close()
            except Exception:
                pass  # swallowed: stop must not mask the original failure
            self._gen = None
        self._set_state(CANCELLED)
        self.cancelled.emit()

    def close(self):
        """Release references (called on window close / completion)."""
        self._timer.stop()
        self._gen = None

    # ---- driver --------------------------------------------------------
    def _on_tick(self):
        if self._state != RUNNING or self._gen is None:
            return
        try:
            directive = next(self._gen)
        except StopIteration as stop:
            self._result = stop.value
            self._gen = None
            self._set_state(COMPLETED)
            self.finished.emit(self._result)
            return
        except Exception as exc:
            self._exc = exc
            self._gen = None
            self._set_state(FAILED)
            # Fail loud: full traceback to stderr, never reduced to a status string
            sys.excepthook(type(exc), exc, exc.__traceback__)
            self.failed.emit(exc)
            return

        # Interpret the yielded directive
        if directive is None:
            self._schedule_frame(None, self.options.delay_ms)
        elif isinstance(directive, _Frame):
            msg = directive.message
            delay = self.options.delay_ms if directive.delay_ms is None else int(directive.delay_ms)
            self._schedule_frame(msg, delay)
        elif isinstance(directive, _Barrier):
            msg = directive.message
            if self.options.honor_barriers:
                self._set_state(PAUSED)
                if msg:
                    self._status(msg)
            else:
                self._schedule_frame(msg, 0)
        else:
            # Any other yielded value is a mistake — fail loud
            self._gen = None
            exc = TypeError(f"Script yielded unsupported value: {directive!r} "
                            f"(expected ctx.frame()/ctx.barrier()/None)")
            self._exc = exc
            self._set_state(FAILED)
            sys.excepthook(type(exc), exc, exc.__traceback__)
            self.failed.emit(exc)

    def _schedule_frame(self, message, delay_ms):
        if message:
            self._status(message)
        if delay_ms and delay_ms > 0:
            self._timer.start(int(delay_ms))
        else:
            self._timer.start(0)

    def _status(self, msg):
        sb = self.window.statusBar()
        if sb is not None:
            sb.showMessage(msg)
        # Also surface to stdout so long-running-script progress is visible
        print(f"[script] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------
def _load_script_module(script_path):
    """Import *script_path* as an isolated module; return (module, run_fn)."""
    script_path = os.path.abspath(script_path)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"GUI script not found: {script_path}")
    name = f"spammm_gui_script_{os.path.splitext(os.path.basename(script_path))[0]}"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GUI script: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    run_fn = getattr(mod, 'run', None)
    if run_fn is None:
        raise RuntimeError(f"GUI script must define run(window, argv=None[, ctx=None]): {script_path}")
    return mod, run_fn


def _accepts_ctx(run_fn):
    """True if run_fn's signature accepts a ctx keyword (positional or **kwargs)."""
    try:
        sig = inspect.signature(run_fn)
    except (ValueError, TypeError):
        return False
    params = sig.parameters
    if 'ctx' in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def run_gui_script(window, script_path, script_argv=None, options=None):
    """Execute *script_path*.

    Legacy synchronous scripts return their value unchanged. Generator scripts are
    attached to ``window._gui_script_controller`` and driven by a single-shot QTimer;
    this function returns ``None`` for generators (completion is asynchronous).

    ``options`` defaults to fast mode. Pass ``ScriptOptions(delay_ms=..., ...`` for
    paced launches from the panel or CLI ``--script-*`` flags.
    """
    options = options or ScriptOptions()
    mod, run_fn = _load_script_module(script_path)
    script_argv = list(script_argv or [])
    ctx = ScriptContext(options)
    if _accepts_ctx(run_fn):
        result = run_fn(window, script_argv, ctx=ctx)
    else:
        result = run_fn(window, script_argv)
    # If the call returned a generator, drive it on the controller.
    if inspect.isgenerator(result):
        ctrl = _ensure_controller(window, options)
        ctrl.attach(result)
        return None
    # Legacy synchronous return value
    return result


def _ensure_controller(window, options):
    """Get or create the per-window ScriptController with the given options."""
    ctrl = getattr(window, '_gui_script_controller', None)
    if ctrl is None:
        ctrl = ScriptController(window, options)
        window._gui_script_controller = ctrl
    else:
        ctrl.options = options
        ctrl.ctx = ScriptContext(options)
    return ctrl


# ---------------------------------------------------------------------------
# Bundled script discovery (no import during discovery)
# ---------------------------------------------------------------------------
def bundled_scripts(gui_scripts_dir=None):
    """Return list of (display_name, abs_path) for bundled GUI scripts, sorted.

    Excludes ``_*.py`` and ``*_offline.py``. Does not import any module.
    Scripts live in ``demos/gui_scripts/`` (centralized demo scripts directory).
    """
    if gui_scripts_dir is None:
        # demos/gui_scripts/ relative to repo root (two levels up from spammm/GUI/)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gui_scripts_dir = os.path.join(repo_root, 'demos', 'gui_scripts')
    if not os.path.isdir(gui_scripts_dir):
        return []
    out = []
    for fname in sorted(os.listdir(gui_scripts_dir)):
        if not fname.endswith('.py'):
            continue
        if fname.startswith('_'):
            continue
        if fname.endswith('_offline.py'):
            continue
        stem = fname[:-3]
        display = stem.replace('_', ' ').title()
        out.append((display, os.path.abspath(os.path.join(gui_scripts_dir, fname))))
    return out


# ---------------------------------------------------------------------------
# Per-script QSettings persistence (argv + pacing), keyed by stable path hash
# ---------------------------------------------------------------------------
import hashlib

_SETTINGS_ORG = "FireCore"
_SETTINGS_APP = "KekuleExplorer"
_SETTINGS_GROUP_PREFIX = "gui_script/"


def _path_key(path):
    h = hashlib.sha1(os.path.abspath(path).encode('utf-8')).hexdigest()[:16]
    return _SETTINGS_GROUP_PREFIX + h


def load_script_settings(path):
    """Return remembered {argv, delay_ms, points_per_frame, honor_barriers} or {}."""
    s = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.beginGroup(_path_key(path))
    out = {}
    if s.contains('path'):
        out['path'] = s.value('path')
        out['argv'] = s.value('argv', '', type=str)
        out['delay_ms'] = int(s.value('delay_ms', 0, type=int))
        out['points_per_frame'] = int(s.value('points_per_frame', 0, type=int))
        out['honor_barriers'] = bool(s.value('honor_barriers', False, type=bool))
    s.endGroup()
    return out


def save_script_settings(path, argv, delay_ms, points_per_frame, honor_barriers):
    """Persist argv + pacing for a script path."""
    s = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.beginGroup(_path_key(path))
    s.setValue('path', os.path.abspath(path))
    s.setValue('argv', argv)
    s.setValue('delay_ms', int(delay_ms))
    s.setValue('points_per_frame', int(points_per_frame))
    s.setValue('honor_barriers', bool(honor_barriers))
    s.endGroup()


def recent_script_paths():
    """Return absolute paths of all scripts with saved settings, sorted by path."""
    s = QtCore.QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.beginGroup(_SETTINGS_GROUP_PREFIX.rstrip('/'))
    paths = []
    for grp in s.childGroups():
        s.beginGroup(grp)
        if s.contains('path'):
            paths.append(str(s.value('path')))
        s.endGroup()
    s.endGroup()
    return sorted(set(paths))


# ---------------------------------------------------------------------------
# Script Runner extension panel (build_ui)
# ---------------------------------------------------------------------------
from spammm.GUI.ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, AutoGridPlacer, tight_button, tight_spin
from spammm.GUI.CollapsibleSection import CollapsibleSection


def build_ui(window):
    """Build the Script Runner extension panel and attach controller + actions.

    Reuses the existing extension lifecycle. The panel is operational UI:
    script selector, raw argv, pacing controls, Run/Continue/Stop, state label.
    """
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    # --- script selector + Browse ---
    g_sel = AutoGridPlacer(cols=4)
    window.sr_script_combo = QtWidgets.QComboBox()
    window.sr_script_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    window.sr_script_combo.setToolTip('Bundled and recently browsed scripts')
    g_sel.add_pair('Script:', window.sr_script_combo)
    window.sr_browse_btn = tight_button('Browse')
    g_sel.add(window.sr_browse_btn)
    layout.addLayout(g_sel.layout())

    # --- argv line ---
    g_argv = AutoGridPlacer(cols=4)
    window.sr_argv_edit = QtWidgets.QLineEdit()
    window.sr_argv_edit.setPlaceholderText('argv line (shlex)')
    window.sr_argv_edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    g_argv.add_pair('argv:', window.sr_argv_edit)
    layout.addLayout(g_argv.layout())

    # --- pacing controls ---
    g_pace = AutoGridPlacer(cols=4)
    window.sr_delay_spin = tight_spin(value=0, step=50, vmin=0, vmax=60000, decimals=0, int_mode=True)
    g_pace.add_pair('Delay ms:', window.sr_delay_spin)
    window.sr_ppf_spin = tight_spin(value=0, step=1, vmin=0, vmax=100000, decimals=0, int_mode=True)
    g_pace.add_pair('Pts/frame:', window.sr_ppf_spin)
    window.sr_barrier_chk = QtWidgets.QCheckBox('Barriers')
    g_pace.add(window.sr_barrier_chk)
    layout.addLayout(g_pace.layout())

    # --- Run / Continue / Stop + state ---
    # State label is on its OWN row below the buttons: its text changes during
    # simulation (state: IDLE -> RUNNING [script] -> COMPLETED [script]) and would
    # otherwise grow its grid column, shrinking the button columns and clipping
    # button labels. Own row keeps button column widths static.
    g_btn = AutoGridPlacer(cols=4)
    window.sr_run_btn = tight_button('Run')
    window.sr_continue_btn = tight_button('Continue')
    window.sr_stop_btn = tight_button('Stop')
    g_btn.add(window.sr_run_btn)
    g_btn.add(window.sr_continue_btn)
    g_btn.add(window.sr_stop_btn)
    g_btn.newrow()
    window.sr_state_label = QtWidgets.QLabel('state: IDLE')
    window.sr_state_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    g_btn.add(window.sr_state_label, span=g_btn._cols)
    layout.addLayout(g_btn.layout())

    # --- wire signals ---
    window.sr_browse_btn.clicked.connect(lambda: _on_browse(window))
    window.sr_run_btn.clicked.connect(lambda: _on_run(window))
    window.sr_continue_btn.clicked.connect(lambda: _on_continue(window))
    window.sr_stop_btn.clicked.connect(lambda: _on_stop(window))
    window.sr_script_combo.currentIndexChanged.connect(lambda: _on_script_selected(window))

    # Ensure controller exists and reflect its state
    _ensure_controller(window, ScriptOptions())
    window._gui_script_controller.state_changed.connect(lambda s: _refresh_state(window))
    window._gui_script_controller.finished.connect(lambda v: _refresh_state(window, done=True))
    window._gui_script_controller.failed.connect(lambda e: _refresh_state(window, done=True))
    window._gui_script_controller.cancelled.connect(lambda: _refresh_state(window, done=True))

    # Populate selector and restore last selection
    _populate_combo(window)
    _refresh_state(window)

    return UIComponents(panel=panel, help_text=_help_text())


def _help_text():
    return ('Script Runner — drive GUI scripts with optional pacing.\n'
            'Delay ms: pause after each frame (0 = fast). '
            'Pts/frame: chunk size for ctx.batches() (0 = one batch). '
            'Barriers: honor ctx.barrier() pauses.\n'
            'F8 = Continue. Scripts are trusted Python with full window access.')


def _populate_combo(window):
    """Fill the script combo with bundled + recent external scripts."""
    combo = window.sr_script_combo
    combo.blockSignals(True)
    combo.clear()
    seen = set()
    for display, path in bundled_scripts():
        if path in seen:
            continue
        seen.add(path)
        combo.addItem(display, path)
    for path in recent_script_paths():
        if path in seen:
            continue
        seen.add(path)
        combo.addItem(os.path.basename(path), path)
    combo.blockSignals(False)


def _on_browse(window):
    start = window.work_dir if hasattr(window, 'work_dir') else os.getcwd()
    path, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Select GUI script', start, 'Python (*.py)')
    if not path:
        return
    path = os.path.abspath(path)
    # Add to combo if new
    combo = window.sr_script_combo
    idx = combo.findData(path)
    if idx < 0:
        combo.blockSignals(True)
        combo.addItem(os.path.basename(path), path)
        combo.blockSignals(False)
        idx = combo.count() - 1
    combo.setCurrentIndex(idx)
    _on_script_selected(window)


def _on_script_selected(window):
    """Restore remembered argv for the selected script.

    Pacing (delay/ppf/barriers) is NOT restored here — those are global presentation
    settings the user sets once. Restoring them per-script would silently overwrite
    the user's manually set delay when selecting a script from the menu.
    """
    path = window.sr_script_combo.currentData()
    if not path:
        return
    st = load_script_settings(path)
    window.sr_argv_edit.setText(st.get('argv', ''))


def _on_run(window):
    """Save settings, build ScriptOptions, and launch the selected script."""
    path = window.sr_script_combo.currentData()
    if not path or not os.path.isfile(path):
        window.statusBar().showMessage('Script Runner: no script selected')
        return
    ctrl = getattr(window, '_gui_script_controller', None)
    if ctrl is not None and ctrl.state in (RUNNING, PAUSED):
        window.statusBar().showMessage(f'Script Runner: already {ctrl.state} — stop first')
        return
    argv = window.sr_argv_edit.text().strip()
    delay = int(window.sr_delay_spin.value())
    ppf = int(window.sr_ppf_spin.value())
    barriers = bool(window.sr_barrier_chk.isChecked())
    save_script_settings(path, argv, delay, ppf, barriers)
    options = ScriptOptions(delay_ms=delay, points_per_frame=ppf, honor_barriers=barriers)
    _ensure_controller(window, options)
    print(f"[Script Runner] run {path} argv={shlex.split(argv)} delay={delay} ppf={ppf} barriers={barriers}", flush=True)
    try:
        run_gui_script(window, path, shlex.split(argv), options=options)
    except Exception as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)
        # Use the existing GUI error mechanism for interactive launches
        if hasattr(window, '_raise'):
            window._raise(f'Script Runner: FAILED {exc}', title='Script Runner Error')
        else:
            window.statusBar().showMessage(f'Script Runner: FAILED {exc}')
    _refresh_state(window)


def _on_continue(window):
    ctrl = getattr(window, '_gui_script_controller', None)
    if ctrl is not None:
        ctrl.continue_barrier()


def _on_stop(window):
    ctrl = getattr(window, '_gui_script_controller', None)
    if ctrl is not None:
        ctrl.stop()
    _refresh_state(window)


def _refresh_state(window, done=False):
    """Update Continue/Stop enabled state and the state label."""
    ctrl = getattr(window, '_gui_script_controller', None)
    state = ctrl.state if ctrl is not None else IDLE
    window.sr_continue_btn.setEnabled(state == PAUSED)
    window.sr_stop_btn.setEnabled(state in (RUNNING, PAUSED))
    window.sr_run_btn.setEnabled(state not in (RUNNING, PAUSED))
    script = window.sr_script_combo.currentData() if hasattr(window, 'sr_script_combo') else ''
    sname = os.path.basename(script) if script else ''
    window.sr_state_label.setText(f'state: {state}' + (f'  [{sname}]' if sname else ''))
