---
type: Task
title: GUI scripting & demo runner — conference demo of full SPAMMM workflow (build → greedy assembly → AFM/BR-STM/PME)
status: design proposed — awaiting USER approval before implementation
tags: [gui, scripting, demo, conference, pacing, menu, automation, presentation]
timestamp: 2026-08-01
related: [GUI_DrawDemo_Scripts.md, GUI_TightLayout.md, Consolidate_GUI_CLI_Backend_Input_Protocol.md]
skills: [code-reuse, doc-read-navigate]
---

# GUI scripting & demo runner

## 1. Goal

Produce a **conference demo** of `SPAMMM_GUI` running a complex workflow end-to-end,
recordable as a video:

1. **Build** a molecule (editor ring/atom ops).
2. **Optimize on surface** — greedy assembly of 4 PTCDA molecules (MC/GA rigid-body
   packing on a substrate).
3. **Generate AFM**, **BR-STM** (bond-resolved), and **PME charge-rings** images of the
   resulting assembly.

The demo must be driven by a **GUI script** (Python) that:

- can be **selected from a menu inside the running GUI** (new extension/menu entry), and
- can equally be **run directly from the terminal** via the existing `--script` arg.

Normally scripts run **as fast as possible** (no delay). For demonstration the same
script must be able to run **slowly / paced** so an audience can follow each step, with
primitives to:

- set a **delay per frame**,
- set **how many compute points per frame** (chunk size before a view update),
- emit an explicit **wait / barrier** command (pause until a condition or user input).

The scripting system should be **maximally general and powerful** — able to drive any
operation the GUI exposes. This document inventories what already exists, assesses its
generality, and poses the open design questions. A better model will answer them and
propose an efficient, general, elegant (simple), and unlimited solution.

## 2. Inventory — what already exists

### 2.1 Script runner + CLI entry (DONE, working)

`spammm/GUI/gui_script_runner.py` — 23-line loader. Any `.py` defining
`run(window, argv=None)` is executed after `window.show()`.

`spammm/GUI/SPAMMM_GUI.py` lines 1620–1640 — `--script`/`-s` CLI arg already wired.
Args after `--` are forwarded to the script. **Terminal-arg execution already works:**

```
./run_gui.sh --script spammm/GUI/gui_scripts/NAME.py -- --opt val
```

### 2.2 Widget-helper library (DONE, ~305 lines)

`spammm/GUI/gui_script_utils.py` (imported as `GSU`):

- **Widget setters** (all pump `processEvents`): `set_edit_mode`, `set_atom_combo`,
  `set_ring_size`, `set_spin_value`, `set_combo_text`, `set_check`, `click_button`,
  `set_auto_h_cap`, `set_line_edit`, `set_slider_value`.
- **Panel control**: `expand_extension_panel` (open/close a `CollapsibleSection` by
  registry key or display title — reads `window._extension_sections`).
- **Molecule**: `load_molecule`.
- **Demo overlays**: `apply_demo_overlays` — drives VisPy hover/cursor/ring-preview
  chrome the same way `EditModeHandlers` do on real mouse move (so GIF frames show
  foreshadow rings, bond hover, δ/φ handles). Caveat: never `set_data` empty arrays on
  `ring_preview_line` (offscreen render can segfault).
- **Capture**: `capture_canvas_png` (VisPy viewport), `capture_window_png` (full window,
  composites `canvas.render()` into the Qt grab — grab alone often blanks OpenGL on
  offscreen/Wayland), `frames_to_gif` (Pillow pack of ordered PNGs).

### 2.3 Existing scripts (5, all use the `run(window, argv)` contract)

`spammm/GUI/gui_scripts/`:

- `azaindol_draw_demo.py` — **closest precedent**: full draw workflow → PNG frames →
  GIF via a `GuiHost` that wraps `window`.
- `azaindol_draw_offline.py` — headless (no Qt), same shared sequence.
- `folded_rigid_setup.py` — load mol + folded substrate, start FoldedRigid.
- `rc_scan_review.py`, `rc_scan_offline.py` — reaction-coordinate scan.

The **`GuiHost` pattern** (`azaindol_draw_demo.py` lines 19–121) is the abstraction to
generalize: a host object that wraps `window` and exposes **semantic ops**
(`add_hex`, `select_ids`, `translate_selected`, `snapshot`, `set_auto_h`, …) so the
**sequence is UI-agnostic** and can be replayed against either a live window or an
offline backend.

### 2.4 Extension system (DONE, declarative registry)

`spammm/GUI/ExtensionManager.py` — `EXTENSION_REGISTRY` maps keys → module + `build_ui`.
Already registers `afm`, `rigid_assembly`, `charge_rings`, `folded_rigid`, `ff`, `dftb`,
`firecore`, `kekule`, `qeq`, `fragments`, `reaction_coord`, `vibrations`, `povray`, …
Each extension's panel is a `CollapsibleSection` registered in
`window._extension_sections` (key → section). Scripts can open them via
`GSU.expand_extension_panel`.

### 2.5 All target demo operations already exist as GUI-callable functions

| Demo step | Function | Location |
|---|---|---|
| Build 4×PTCDA on grid | `_on_build(window)` | `RigidAssemblyExtension.py:189` |
| Greedy MC step (one) | `_on_mc_step(window)` | `RigidAssemblyExtension.py:285` (reuses `greedy_energy_step`) |
| Greedy MC run (loop) | `_on_mc_run(window)` | `RigidAssemblyExtension.py:321` |
| AFM full pipeline | `run_afm_full_pipeline(window)` | `AFMExtension.py:730` |
| STM | `run_stm(window)` | `AFMExtension.py:754` |
| BR-STM (bond-resolved) | `run_br_stm(window)` | `AFMExtension.py:797` |
| PME charge rings XY | `_on_pme_scan_xy(window)` | `RigidAssemblyExtension.py:510` (reuses `pauli_scan.scan_xy`) |

**No new compute logic is needed** — a demo script just orchestrates these.

### 2.6 Menu system (MINIMAL — only Settings)

`SPAMMM_GUI.py:767–769` — only `self.settings_menu = self.menuBar().addMenu("Settings")`.
**No "Scripts" menu exists yet.** This is the one missing wiring for in-GUI script
selection.

### 2.7 Pacing / timing (MISSING for demos)

Scripts run **synchronously in the main thread**. `GSU.process_events` pumps repaints
but introduces no delay. Long ops (AFM S1–S4, greedy MC loop) freeze the UI between
`processEvents` calls.

Existing `QTimer` precedent for **non-blocking** paced execution:
`FFExtension.py:414–436` (`_on_interactive` → 30 ms timer → `_interactive_tick` runs N
steps then updates view). This is the pattern used for live relaxation. **No equivalent
exists for scripts.**

## 3. Capability assessment

The infrastructure is **~80 % there**. The runner, CLI arg, helper library,
capture/GIF, and every target operation already exist. What is missing:

1. A **Scripts menu** in the GUI (auto-list `gui_scripts/*.py`, run via
   `run_gui_script`).
2. A **demo pacing API** so the same script can run fast (headless/CI) or slow
   (conference) without code changes.
3. A clean answer to the **synchronous-blocking problem** for smooth live visuals.

The current scripts are **general** (any Python, full `window` access) but **not
maximally elegant**: they hardcode widget attribute names, have no standard
pacing/narration hooks, and block the event loop.

## 4. Open design questions & challenges

### Q1 — Scripts menu: auto-discover vs register?

Should the menu auto-scan `gui_scripts/*.py` (zero boilerplate, but no metadata — names
from filenames), or should scripts declare metadata (title, description, arg schema) in
a header/registry so the menu shows human-readable entries + can prompt for args? What
is the simplest convention that stays general?

### Q2 — Pacing API design (the core problem)

The user wants three pacing primitives usable **inside the script**:

- **delay per frame** (sleep N ms between steps),
- **points-per-frame** (run N compute units, then update view — e.g. N MC steps per
  visual frame),
- **wait / barrier** (explicit pause until condition or user input).

How should these be exposed? Options:

- (a) A `DemoContext`/`host` object passed to `run(window, ctx, argv)` with
  `ctx.sleep()`, `ctx.points_per_frame()`, `ctx.wait()`, `ctx.snapshot()`.
- (b) Module-level globals in `gui_script_utils` (e.g. `GSU.set_pacing(delay=…)`) —
  simple but stateful/global.
- (c) A generator/coroutine script that `yield`s between steps and the runner drives it
  on a `QTimer` (each `yield` = one frame, naturally non-blocking).
- (d) Decorator/DSL.

Which is most general, simplest, and keeps the same script runnable fast from terminal
and slow from GUI?

### Q3 — Synchronous blocking vs non-blocking execution

`run_gui_script` runs synchronously → long compute freezes the UI. For a smooth
conference demo, should the script run:

- (a) In a `QThread` worker (compute off-main-thread, but Qt widget access from worker
  is unsafe — need signals)?
- (b) As a `QTimer`-driven generator (script yields, runner schedules next step on
  timer — cooperative, no threading issues, but script must be written as generator)?
- (c) Synchronous but with frequent `processEvents` + the compute itself chunked (e.g.
  `_on_mc_step` already does one step; call it N times with `processEvents` between)?

Note: the GPU/OpenCL compute (greedy MC, AFM pipeline) is itself blocking per-call.
True smoothness needs the compute to be chunkable (it is — `_on_mc_step` is one step,
AFM has stages S1–S6). How to best exploit this chunkability for pacing?

### Q4 — Single script, two modes (fast terminal / slow GUI demo)?

The user wants **the same script** runnable (1) fast from terminal `--script` and (2)
slow with visible frames in GUI menu. How should pacing default? Should the script read
a `--demo`/`--delay`/`--points-per-frame` flag (already possible via `argv`), or should
the menu runner inject a default demo context while the CLI runner injects a no-op fast
context? What is the SSOT for pacing defaults?

### Q5 — Narration & camera for conference quality

Beyond pacing, a good demo needs: status-bar messages, view fits/zooms, highlights,
maybe audio cues. `azaindol_draw_demo` already has `snapshot(name, title=…)` and
`apply_demo_overlays`. Should there be a standard `narrate(text)` / `focus_on(atoms)` /
`highlight(ids)` API in the host, or is `window.statusBar().showMessage` +
`scene.fit_to_atoms` enough? How much to formalize vs leave as raw `window` access?

### Q6 — Generality ceiling: "run any operation within the GUI"

The user wants the scripting system to be **unlimited** (any GUI operation scriptable).
Current approach = raw Python calling `window.*` and extension functions. Is that
sufficient, or do we need:

- A whitelist/command protocol (like the FireCore `ScriptRunner.js` mentioned in
  `doc/Topics/GUI_DrawDemo_Scripts.md`)?
- An event-record/replay (record user clicks → replay)?
- Just better host abstraction covering all extensions?

The doc notes FireCore uses a whitelist; SPAMMM chose "Python hosts instead." Should we
keep raw-Python (maximally powerful, minimally safe) or add structure?

### Q7 — Where does the demo pacing logic live?

Per `AGENTS.md` SoC + `code-reuse` skill: pacing/capture belongs in
`gui_script_utils` (shared), the menu wiring in `SPAMMM_GUI.create_menus`, the demo
**sequence** in a new script under `gui_scripts/`. Should the pacing context be a new
class in `gui_script_utils` (e.g. `DemoHost`/`ScriptContext`), or extend the existing
`GuiHost` pattern from `azaindol_draw_demo.py`? Avoid new files unless necessary —
`gui_script_utils.py` is the natural home.

## 5. Proposed design

### 5.1 Decision summary

| Question | Decision | Why |
|---|---|---|
| Q1 — discovery/UI | Real tight Script Runner extension panel + Scripts menu; auto-discover bundled scripts and browse any `.py`; no script registry | Persistent controls are better for conference operation; no duplicate metadata or eager imports |
| Q2 — pacing | `ScriptContext` + explicit yielded directives | Precise, testable control without global state or sleeps |
| Q3 — execution | `QTimer`-driven Python generator on the Qt main thread | Non-blocking between work chunks, Qt-safe, cancellable, no nested event loop |
| Q4 — fast/demo parity | One `ScriptOptions` SSOT used by CLI and panel; panel remembers settings per script | Same script and runner; reproducible conference settings without changing script argv |
| Q5 — presentation helpers | Formalize only `frame()` and `barrier()`; reuse existing GSU camera/overlay/capture helpers | Avoid a second overlapping GUI abstraction |
| Q6 — generality | Trusted unrestricted Python with raw `window` access; no whitelist/DSL | A command protocol can only reduce power and duplicate the Python API |
| Q7 — placement | Runner/controller + extension `build_ui` in `gui_script_runner.py`; GSU helpers stay separate; menu bridge in `SPAMMM_GUI.py` | One responsibility per existing module; the real extension needs no additional module |

The essential model is **cooperative scripting**: each script may perform arbitrary
Python/Qt/OpenCL work, but it yields at explicit visual boundaries. The runner never
attempts to preempt Python, Qt, or an OpenCL call.

### 5.2 Why QTimer + generator is the correct execution model

#### Reject `QThread`

GUI scripts intentionally access Qt widgets and the VisPy scene directly. Moving the
script to a worker thread makes those accesses invalid and would require a signal/slot
proxy for every operation. OpenCL contexts/queues and extension state also have existing
thread affinity assumptions. A worker thread would therefore make unrestricted scripts
less powerful and much more complex. Pure compute may independently gain worker support
later, but that is a compute-API concern, not the script runner.

#### Reject nested `QEventLoop` / `time.sleep`

`time.sleep` freezes painting and input. A nested `QEventLoop` preserves linear syntax
but permits arbitrary re-entry while a script's stack is suspended, complicates stop and
error unwinding, and is fragile when scripts open dialogs. It is inappropriate as the
foundation for a general runner.

#### Use a single-shot `QTimer`

A generator preserves local variables and control flow between yields. The controller
advances it once, handles the yielded directive, then schedules the next advance with a
single-shot `QTimer`. All script and widget code remains on the GUI thread. At a yield,
Qt naturally paints, handles input, and can accept Continue/Stop. This is the same
cooperative principle already used by the interactive FF timer, generalized for scripts.

**Hard limit:** cancellation and responsiveness exist only at yields. A blocking DFTB,
Python, or OpenCL call cannot be interrupted by the runner. Long workflows must expose
natural chunks: one MC step, one MD batch, one AFM stage, one scan tile, etc. If one chunk
is too slow for acceptable GUI responsiveness, that compute API—not the script runner—
must be made finer-grained.

### 5.3 Backward-compatible script contract

Existing scripts remain valid and preserve their synchronous return value:

```python
def run(window, argv=None):
    ...
    return result
```

New paced scripts append an optional keyword and contain at least one `yield`:

```python
def run(window, argv=None, ctx=None):
    ...
    yield ctx.frame("Visible step complete")
    ...
```

Appending `ctx` preserves the established positional meaning of `argv`. The runner uses
`inspect.signature` and passes `ctx=...` only when accepted (including `**kwargs`). It
then inspects the returned object:

- normal value → legacy synchronous completion; return it unchanged (required by the
  existing `test_rc_scan_gui_script.py`),
- generator → attach it to the controller and advance it by QTimer,
- coroutine / arbitrary iterator → reject loudly; asynchronous frameworks and implicit
  iterator semantics are deliberately out of scope.

Generator support is **opt-in**, so no existing script must be rewritten. A legacy
script selected from the menu still runs, but cannot be paced or stopped until it
returns; the launcher must state this clearly.

### 5.4 `ScriptOptions`: one pacing SSOT

The runner receives one immutable options object, independent of scientific script
arguments:

```python
ScriptOptions(
    delay_ms=0,             # delay after each Frame; 0 = no deliberate delay
    points_per_frame=0,     # 0 = all remaining points in one batch (fast)
    honor_barriers=False,   # False = Barrier behaves as a zero-delay Frame
)
```

Defaults are the required **fast mode**. Positive `points_per_frame` is presentation
chunking. The script's own `argparse` remains the SSOT for scientific/task parameters;
presentation options are runner options and must not be duplicated in each script.

CLI options belong before `--`; arguments after `--` remain untouched script argv:

```bash
# Fast defaults
./run_gui.sh --script spammm/GUI/gui_scripts/conference_demo.py -- --n-step 200

# Paced, five MC steps per visual frame, explicit barriers enabled
./run_gui.sh --script spammm/GUI/gui_scripts/conference_demo.py \
  --script-delay-ms 300 --script-points-per-frame 5 --script-barriers \
  -- --n-step 200
```

The Script Runner panel edits the same three fields and **remembers them per absolute
script path** with the existing `QSettings`, together with that script's raw argv. Use a
stable path hash as the settings-group key and store the path inside the group for
inspection. Persistence is UI convenience; the runtime `ScriptOptions` object remains
authoritative. CLI launches do not read remembered GUI values and remain deterministic.

### 5.5 `ScriptContext`: minimal and precise API

`ScriptContext` is created by the runner and passed to the script. It contains only
execution/pacing semantics:

```python
class ScriptContext:
    delay_ms: int
    points_per_frame: int
    honor_barriers: bool

    def batches(self, n): ...
    def frame(self, message=None, delay_ms=None): ...
    def barrier(self, message=None): ...
```

Semantics:

- `ctx.batches(n)` yields `(i0, i1)` ranges. `points_per_frame <= 0` gives one `(0, n)`
  range; otherwise ranges have at most that many points. **The script defines what a
  point means** (MC step, MD step, pixel/tile, optimizer iteration). The runner cannot
  infer or enforce this externally.
- `yield ctx.frame(message)` is one repaint/control boundary. The runner updates the
  status bar and schedules the next generator advance after the configured delay;
  returning to Qt lets already-updated widgets/VisPy repaint. It does **not** call
  `window.refresh_view()` implicitly because that may be expensive or alter an extension
  plot. The script invokes refresh only when its operation does not already do so.
  `delay_ms=` overrides the global delay for this one frame.
- `yield ctx.barrier(message)` pauses only when `honor_barriers=True`; otherwise it acts
  as a zero-delay frame. Continue comes from the extension panel, Scripts menu, or their
  shared `F8` shortcut—never a modal dialog or nested event loop.
- `yield None` may mean a default frame for concise scripts; any other yielded value is a
  `TypeError` so mistakes fail loudly.

There is deliberately no `chunk()` directive: by the time a generator yields, the work
has already happened. A `yield ctx.chunk(10)` API would falsely imply that the runner can
schedule ten points. `batches()` puts chunking at the only place where it can be correct:
around the work loop.

There is also deliberately no `ctx.sleep()`: timed waiting is `yield ctx.frame(...,
delay_ms=N)`, which returns control to Qt rather than sleeping.

### 5.6 Work-unit contract: make points-per-frame real

`ctx.batches()` controls when the script yields, but it cannot suppress painting or
`processEvents()` performed *inside* a called operation. The current
`RigidAssemblyExtension._on_mc_step(window)` calls `_sync_display` on acceptance and
`_status` (which pumps Qt) on every step. Repeating that callback five times therefore
does not mean five compute points followed by one frame.

Make one surgical, backward-compatible generalization:

```python
def _on_mc_step(window, update_ui=True):
    ...                         # always update ensemble/GPU/energy/counters
    if update_ui:
        _sync_display(window)   # always sync current final pose, even if this step rejected
        _status(window, ...)
    return summary              # E0, Ebest, E, accepted, step count
```

- Existing button and `_on_mc_run` calls keep `update_ui=True` and behave unchanged.
- A paced script calls `update_ui=False` for intermediate points and `True` for the last
  point in each `ctx.batches()` range.
- The last point refreshes even when rejected, so an accepted pose from earlier in the
  batch is still displayed.
- The returned summary lets the script print accepted-step and energy-decrease progress
  once per frame without reading incidental widget text.
- No scientific compute is duplicated or moved into the script.

This opt-in `update_ui` pattern is the general contract for other chunkable operations:
the workhorse owns compute and authoritative state; the caller chooses whether this call
is also a visual boundary. Do not force every extension to adopt it pre-emptively—add it
only where a paced script needs multiple work units per frame.

### 5.7 Presentation API: reuse, do not duplicate

`ScriptContext` must not become a second `GuiHost` or duplicate
`gui_script_utils.py`. Presentation uses existing pieces:

- narration at a scheduling boundary: `yield ctx.frame("Running greedy assembly…")`,
- widgets/panels: `GSU.set_*`, `GSU.click_button`, `GSU.expand_extension_panel`,
- camera: `window.scene.fit_to_atoms(...)`,
- hover/highlight chrome: `GSU.apply_demo_overlays(...)`,
- PNG/GIF: `GSU.capture_window_png`, `capture_canvas_png`, `frames_to_gif`.

If repeated camera operations emerge from multiple real scripts, add them to GSU then;
do not predict a broad `focus_on`/`highlight` facade now. The existing `GuiHost` remains
appropriate for a sequence shared with an offline renderer, but it is not required for
every GUI-only script.

### 5.8 Controller state and failure semantics

Only one generator script may be active per window. The controller is retained as
`window._gui_script_controller` and has explicit states:

```
IDLE → RUNNING ↔ PAUSED → COMPLETED
                  ↘       CANCELLED
                   FAILED
```

- **Continue** advances a paused barrier through one controller method shared by the
  extension button, Scripts menu action, and `F8` shortcut.
- **Stop** requests cancellation and calls `generator.close()` at the next yield so the
  script's `finally` blocks run. It cannot abort a currently executing kernel/function.
- Starting another script while one is RUNNING/PAUSED is rejected; do not silently stop
  or replace active scientific work.
- Completion stores the generator's `StopIteration.value`, updates status, restores menu
  actions, and releases module/generator references.
- Failure stops the timer, stores the exception/traceback, prints the full traceback via
  `sys.excepthook`, updates the panel/status bar, and for interactive launches uses the
  existing GUI error mechanism. Asynchronous errors must never be reduced to a status
  string.
- The controller exposes finished/failed/cancelled Qt signals for tests and future UI;
  scripts do not depend on those signals.
- Closing the main window closes the active generator. No forceful thread termination is
  involved.

A single-shot timer is preferable to a repeating timer: the next interval is selected
from the yielded frame, no tick can accumulate while a work chunk is running, and a
barrier simply does not arm the timer.

### 5.9 Script Runner extension panel + menu bridge

The USER chose a **real extension panel**, not a menu-only launcher. Register
`gui_scripts` in `ExtensionManager.EXTENSION_REGISTRY`, enabled by default, with module
`spammm.GUI.gui_script_runner` and its `build_ui(window)`. This reuses the existing
extension lifecycle and `CollapsibleSection`; no separate extension module is needed.
The panel is operational UI, not an empty container.

Keep the layout maximally tight:

```
Script Runner
[ script/path combo                         ] [Browse]
[ argv line                                           ]
Delay [ 300 ms ]   Points/frame [ 5 ]   [x] Barriers
[Run] [Continue] [Stop]                 state: PAUSED
```

- The combo contains bundled scripts plus previously browsed external scripts.
- Selecting a script loads its remembered argv and pacing values.
- Browse accepts any trusted `.py`, normalizes the absolute path, adds it to recent
  scripts, selects it, and persists it.
- Run saves current settings, builds `ScriptOptions`, and invokes the same
  `run_gui_script` used by CLI.
- Continue is enabled only at a Barrier; Stop is enabled while RUNNING/PAUSED.
- The state label shows IDLE/RUNNING/PAUSED/COMPLETED/CANCELLED/FAILED and the current
  script, without relying only on transient status-bar text.

`SPAMMMWindow.create_menus` adds a bridge, not a duplicate configuration UI:

```
Scripts
├── Open Script Runner            # expand/focus the extension panel
├── Select Script…                # browse, then populate/expand the panel
├── Bundled
│   ├── Azaindol Draw Demo        # select in panel; do not auto-run
│   ├── Folded Rigid Setup
│   ├── RC Scan Review
│   └── Conference Demo
├── Refresh Scripts
├── ----------------------------
├── Continue                 F8   # enabled only at Barrier
└── Stop                          # enabled while RUNNING/PAUSED
```

Selecting a bundled menu item opens the panel with that script selected rather than
immediately starting scientific work; Run remains an explicit action using visible
settings. `F8` is currently conflict-free and should be registered through
`ShortcutRegistry` under group `Script Runner`, invoking the same Continue method as the
panel and menu. Do not add a second shortcut dispatch path.

Bundled discovery scans `spammm/GUI/gui_scripts/` on refresh:

- sorted `*.py`, excluding `_*.py`, `__pycache__`, and `*_offline.py`,
- display name = humanized filename stem,
- tooltip/data = absolute path,
- no import during discovery (avoids startup cost and top-level side effects),
- selected file is imported only on Run.

Do not build metadata parsing, an argument-schema language, or a duplicate
`SCRIPT_REGISTRY`. Scripts already own `argparse`; the panel uses one raw argv line
parsed with `shlex.split`, supporting arbitrary arguments without reimplementing
argparse in Qt.

Per the USER decision, argv/delay/points/barriers are remembered **per absolute script
path** in QSettings. A stable path hash keys each settings group; the group stores the
path and values. The combo's recent external scripts are reconstructed from those groups.
CLI launches never consume these GUI-persisted defaults.

### 5.10 Trusted unrestricted Python is the general solution

GUI scripts are local, trusted Python code. They receive the real `window`, may import
any SPAMMM or third-party module, call backend/extension workhorse functions, manipulate
widgets, and use normal Python control flow. This is the concrete meaning of
**unlimited**.

Do not add a whitelist command language:

- it necessarily exposes less than Python,
- every GUI capability would require duplicate command plumbing,
- it creates a second API that drifts from real implementation,
- it cannot express arbitrary scientific branching/composition elegantly.

Do not make pixel event recording/replay the foundation: coordinates and widget layout
are brittle. For demonstration parity, scripts should set widgets and invoke their
buttons where accessible (`GSU.click_button`), or call the same semantic slot/workhorse
when a button is not retained on `window`. Event recording can later generate a draft
Python script, but replay should remain semantic.

Raw Python is intentionally not sandboxed. The extension panel should label external
scripts as trusted code; pretending to sandbox imports while handing over `window` would
be false security.

### 5.11 Conference demo skeleton

The script configures the same GUI widgets a presenter would use, chunks only naturally
chunkable work, and yields around monolithic stages:

```python
import argparse
from spammm.GUI import gui_script_utils as GSU
from spammm.GUI import RigidAssemblyExtension as RA
from spammm.GUI import AFMExtension as AFM


def run(window, argv=None, ctx=None):
    p = argparse.ArgumentParser()
    p.add_argument('--n-step', type=int, default=200)
    args = p.parse_args(argv or [])

    yield ctx.frame('Configuring 4×PTCDA assembly…')
    GSU.expand_extension_panel(window, 'rigid_assembly')
    GSU.set_combo_text(window.ra_source_combo, 'From file')
    GSU.set_combo_text(window.ra_mol_combo, 'PTCDA')
    GSU.set_spin_value(window.ra_nmol_spin, 4)
    GSU.click_button(window.ra_build_btn)
    yield ctx.frame('Built 4×PTCDA')

    for i0, i1 in ctx.batches(args.n_step):
        for i in range(i0, i1):
            step = RA._on_mc_step(window, update_ui=(i + 1 == i1))
        print(f"MC {i1}/{args.n_step}: E={step['E']:.6f} accepted={step['accepted']}", flush=True)
        yield ctx.frame(f"Greedy assembly: {i1}/{args.n_step}, E={step['E']:.6f}")

    yield ctx.barrier('Assembly ready — Continue to AFM')
    GSU.expand_extension_panel(window, 'afm')

    # Stage calls are blocking but give meaningful boundaries and visible status.
    yield ctx.frame('AFM S1: DFTB+ SCF…')
    AFM.run_afm_stage1(window)
    yield ctx.frame('AFM S1 complete; projecting density…')
    AFM.run_afm_stage2(window)
    yield ctx.frame('AFM S2 complete; building potentials…')
    AFM.run_afm_stage3(window)
    yield ctx.frame('AFM S3 complete; relaxing probe particle…')
    AFM.run_afm_stage4(window)
    yield ctx.frame('AFM image complete')

    yield ctx.barrier('Continue to bond-resolved STM')
    AFM.run_br_stm(window)
    yield ctx.frame('BR-STM complete')

    yield ctx.barrier('Continue to PME charge rings')
    RA._on_pme_scan_xy(window)
    yield ctx.frame('PME charge-rings image complete')
```

The final implementation should prefer retained button attributes for widget parity.
AFM product/stage buttons are currently local variables in `build_ui`, so direct AFM
workhorse calls are necessary unless those buttons are exposed on `window`. This is a
small GUI-accessibility issue, not a reason to invent a command registry.

The script must print unbuffered start/progress/completion messages in addition to GUI
status. For greedy optimization, report accepted steps and energy decrease as required
by the long-running-script rule. Existing callbacks may need their status output exposed
to stdout; pacing must not hide it.

### 5.12 Performance implications

- Fast default (`delay_ms=0`, `points_per_frame=0`, barriers off) creates one work batch
  and no deliberate waits. Generator/timer overhead is one event-loop turn per explicit
  frame, negligible beside scientific compute.
- Demo mode uses a positive batch size chosen so one batch fits the desired frame budget.
  The runner performs no per-atom/per-trial Python compute; it only orchestrates existing
  GPU-backed workhorse calls.
- `QApplication.processEvents()` should not be called redundantly from the controller;
  returning to the Qt event loop after a yield is the repaint mechanism. Existing GSU
  setters may still pump events for widget parity.
- Do not capture PNGs on every live frame unless recording was explicitly requested;
  rendering/capture is independent of pacing.

### 5.13 Verification design

Use existing GUI test infrastructure; no GPU/DFTB is needed for runner tests.

1. **Legacy parity:** existing `test_rc_scan_gui_script.py` still receives the synchronous
   script's returned dataset.
2. **Generator order:** a tiny fixture generator appends markers around two Frames;
   verify one timer advance per frame and final return value.
3. **Fast batching:** `points_per_frame=0` yields one `(0, n)` batch; positive values
   cover every index exactly once with the expected boundaries.
4. **Delay:** use Qt test waiting/signals—not wall-clock sleeps—to verify a frame does not
   advance early.
5. **Barrier:** verify PAUSED state and no progress until Continue; barriers-off advances
   immediately.
6. **Cancellation:** Stop closes the generator, executes `finally`, and emits cancelled.
7. **Failure:** generator exception records the original exception, emits failed, prints
   a full traceback, and returns controller/panel/menu state to idle.
8. **Single active run:** second launch is rejected while running/paused.
9. **Discovery:** bundled list is sorted, excludes `*_offline.py`, and does not import
   modules.
10. **Panel persistence:** switching between two paths restores separate argv and pacing
    values; CLI options remain independent of QSettings.
11. **Shared Continue:** panel button, Scripts menu, and `F8` invoke the same controller
    method and have synchronized enabled state.
12. **MC batching:** `update_ui=False` advances authoritative pose/energy without scene or
    event updates; the final `update_ui=True` syncs the batch's final pose even if its own
    trial is rejected.
13. **Offscreen smoke:** launch one generator script with 0-ms Frames and verify the Qt
    event loop completes it.
14. **L2 conference review:** run the real workflow with `--develop -s`; review explicit
    screenshots/video and scientific output. USER confirmation is required before task
    status can become done.

## 6. Constraints (from AGENTS.md)

- **KISS / AHA / YAGNI** — simplest solution that works; no hasty abstractions; surgical
  edits.
- **DRY** — inventory first (done above); generalize rather than duplicate.
- **SoC** — Compute / plotting / Backend / CLI / GUI in separate modules. Execution,
  pacing, and the Script Runner extension UI belong to `gui_script_runner.py`;
  widget/capture helpers remain in `gui_script_utils.py`; demo sequences live in
  `gui_scripts/`; the thin menu bridge belongs to `SPAMMM_GUI.create_menus`.
- **SSOT** — `ScriptOptions` is the one source for presentation pacing; each script's
  `argparse` remains the source for scientific/task inputs.
- **Fail Fast** — no silent fallbacks; asynchronous errors retain and print their full
  traceback.
- **No unnecessary files** — extend existing infrastructure modules. The requested
  conference workflow itself is one legitimate new script under `gui_scripts/`.
- **Long-running scripts MUST print unbuffered progress** — pacing must not swallow
  status output.
- **Never mark "done" without USER confirmation.**

## 7. Implementation deliverables after approval

1. Extend `gui_script_runner.py` with `ScriptOptions`, `ScriptContext`, typed Frame /
   Barrier directives, and the single-shot-QTimer controller while preserving legacy
   synchronous returns.
2. Add the three runner options to `SPAMMM_GUI.py` CLI parsing and start generator scripts
   before `app.exec_()` by arming their first 0-ms timer.
3. Register the enabled-by-default Script Runner extension; implement its tight selector,
   raw argv, pacing, per-script QSettings, Run/Continue/Stop, and state label in
   `gui_script_runner.py`. Add only the Scripts menu bridge and shared `F8` Continue
   shortcut wiring to `SPAMMM_GUI.py` / `ShortcutRegistry`.
4. Generalize rigid-assembly MC stepping with `update_ui=True` default + returned summary,
   then add one conference workflow script under `gui_scripts/` using existing GSU and
   RigidAssembly/AFM workhorses—no duplicated compute.
5. Add focused L0 runner/panel/menu tests, then run the real L2 conference workflow and
   present artifacts to the USER for confirmation.
6. Update `gui_scripts/README.md` and `doc/Topics/GUI_DrawDemo_Scripts.md` after the
   implementation is verified.

## 8. Related

- Topical SSOT: `doc/Topics/GUI_DrawDemo_Scripts.md` (azaindol draw demo architecture)
- Script launcher notes: `doc/Topics/Takeways.md`
- `spammm/GUI/gui_scripts/README.md` — adding-a-script guide
- `spammm/GUI/FFExtension.py:414` — `QTimer` interactive-relaxation precedent
- `spammm/GUI/azaindol_draw_sequence.py` — shared sequence SSOT (offline + GUI hosts)
