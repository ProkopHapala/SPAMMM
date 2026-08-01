# Control scripts (`gui_scripts/`)

One-command Python scripts launched by the GUI after startup via
`./run_gui.sh --script spammm/GUI/gui_scripts/NAME.py [-- extra args]`, or selected
from the **Scripts** menu / **Script Runner** extension panel. They programmatically
click the same buttons a user would click, so they are not headless; use
`rc_scan_offline.py` / `azaindol_draw_offline.py` if you need the same pipeline
without Qt.

## Two contracts

- **Legacy synchronous** — `def run(window, argv=None)` returns a value. Runs to
  completion; cannot be paced or stopped mid-run.
- **Paced generator** — `def run(window, argv=None, ctx=None)` containing `yield`.
  `ctx` is a `ScriptContext` (see `spammm.GUI.gui_script_runner`). Yield
  `ctx.frame(msg)` for a repaint/control boundary, `ctx.barrier(msg)` to pause until
  Continue (F8), and use `for i0, i1 in ctx.batches(n):` to chunk work into
  points-per-frame. The runner drives the generator on a single-shot QTimer; the
  same script runs fast from CLI (default options) or slow from the panel.

Presentation pacing is set by the runner, not the script:
`--script-delay-ms N`, `--script-points-per-frame M`, `--script-barriers`.
The panel remembers these per script in QSettings.

## File index

- **conference_demo.py** — Paced generator: build 4×PTCDA → greedy MC assembly →
  AFM (S1–S4) → BR-STM → PME charge rings. Conference workflow SSOT.
  - Example (fast): `./run_gui.sh --script spammm/GUI/gui_scripts/conference_demo.py -- --n-step 200`
  - Example (paced): `./run_gui.sh --script spammm/GUI/gui_scripts/conference_demo.py --script-delay-ms 300 --script-points-per-frame 5 --script-barriers -- --n-step 200`
- **folded_rigid_setup.py** — Load a molecule, fit or load a folded substrate
  potential, and start the `FoldedRigid` extension in run or manip (drag) mode.
  - Arguments: `--mol`, `--substrate`, `--fit`, `--fit-fit`, `--run`, `--step`,
    `--manip`, `--n`, `--dt`, `--x`, `--y`, `--z`, `--k`
  - Example: `./run_gui.sh --script spammm/GUI/gui_scripts/folded_rigid_setup.py -- --mol data/xyz/H2O.xyz --fit data/fits/h2o_nacl.npz --run`
- **rc_scan_review.py** — 2Quinolone symmetric pm-NEB relaxed review; `--preview` loads cached npz
- **rc_scan_offline.py** — Same reaction-coordinate path without Qt (`PYTHONPATH=. python3 …`)
- **azaindol_draw_offline.py** — Headless hex→pent→N→H→copy/paste/rotate dimer; one SVG per step
  - Example: `PYTHONPATH=. python spammm/GUI/gui_scripts/azaindol_draw_offline.py`
- **azaindol_draw_demo.py** — Same sequence via live GUI tools; full-window PNG frames → GIF
  - Example: `./run_gui.sh --script spammm/GUI/gui_scripts/azaindol_draw_demo.py`
  - `--canvas-only` for VisPy viewport only; `--zoom-out 2` (default) fits selection box

## Adding a new script

1. Implement `run(window, argv=None)` (legacy) or `run(window, argv=None, ctx=None)`
   (paced generator). Drop the file in this folder; it appears in Scripts → Bundled.
2. Use helpers from `spammm.GUI.gui_script_utils` (`load_molecule`, `set_spin_value`,
   `set_edit_mode`, `expand_extension_panel`, etc.).
3. For chunkable work, expose an `update_ui=True` default on the workhorse and call
   with `update_ui=False` for intermediate points (see `RigidAssemblyExtension._on_mc_step`).

## Related

- `spammm/GUI/gui_script_runner.py` — runner, `ScriptOptions`, `ScriptContext`,
  `ScriptController`, Script Runner extension panel, bundled discovery
- `spammm/GUI/gui_script_utils.py` — widget helpers + demo overlays / PNG+GIF capture
- `spammm/GUI/azaindol_draw_sequence.py` — shared azaindol draw ops (offline + GUI)
- `spammm/GUI/SPAMMM_GUI.py` — main window, `--script` / `-s` runner, Scripts menu, F8
- Design: [`doc/Tasks/GUI_Scripting_DemoRunner.md`](../../../doc/Tasks/GUI_Scripting_DemoRunner.md)
- Topical SSOT: [`doc/Topics/GUI_DrawDemo_Scripts.md`](../../../doc/Topics/GUI_DrawDemo_Scripts.md)
- RC sibling pattern: [`doc/Topics/ReactionCoordinateScan.md`](../../../doc/Topics/ReactionCoordinateScan.md)
