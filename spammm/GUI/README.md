# GUI/

PyQt5 GUI for molecular editing and AFM simulation. Main window combines VisPy 3D scene with lazy-loaded extension panels.

## Naming

| Current | Role |
|---------|------|
| **SPAMMMWindow** (`SPAMMM_GUI.py`) | Main window — molecular editor + extensions |
| **MoleculeEditorBackend** | Topology editing backend (not the Kekule solver) |
| **KekuleExtension** | GUI panel for **KekulePure** bond-order solver only |
| ~~KekuleExplorerWindow~~ | Legacy alias → `SPAMMMWindow` |

## Core

- **SPAMMM_GUI.py** — Main window; `--script` / `-s` runs control scripts after show (see `gui_script_runner.py`)
- **BaseGUI.py** — Reusable PyQt5 widget helpers
- **ExtensionManager.py** — Lazy-loading extensions (AFM, FF, QEq, Vibrations, Kekule, ASCII, **reaction_coord**, …)
- **VispyUtils.py** — VisPy ortho scene (`AtomScene`): atoms/bonds, picking, drag, camera presets, **2D/3D view** (`lock_top` / depth test / RMB orbit)
- **EditModeHandlers.py** — Per-mode mouse dispatch; Ring mode 3D ray pick (atom/bond/COG)
- **Manipulate mode** — canonical context-dispatched rigid interaction mode; Build/Setup
  selects the explicit `rigid_assembly` or `folded_rigid` adapter. Shift+LMB owns the
  backend spatial-constraint toggle.
- **plotutils.py** — Qt 2D plot dialog; re-exports `spammm.plotUtils` grid/ESP helpers

**View mode:** default top-down planar edit (`b2Dview=True`). `Enter` toggles ortho 3D (hex/empty disabled; Ring/atom/bond OK). See `doc/GUI_CHEATSHEET.md`, `doc/Tasks/GUI_Editor_3D_ViewMode.md`.
- **mpl_blit.py** — Fast matplotlib updates via blit on embedded Qt canvas; caveats in `doc/Takeways.md`

## Extensions

- **AFMExtension.py** — FDBM AFM/STM panel (GUI uses ModularPipeline only). **K_LAT in N/m** → eV/Å² internally. Default plot z=3.0 Å; atom overlay checkbox replots live; atoms as `'.'` dots. Prefer grid step ≤0.1 Å. Perf via pipeline `SPAMMM_AFM_FAST_S3` — see `doc/Tasks/PerfBenchmark_FDBM.md`, `doc/Tasks/AFMTesting.md`.
- **KekuleExtension.py** — Kekulé π-bond-order solver panel
- **FoldedRigidExtension.py** — Folded-basis rigid-body manipulation panel (load molecule, fit/load substrate potential, drag atoms)
- **ChargeRingsExtension.py** — PME charge-ring STM: JSON params, Calc XY/xV/1D, cut-line overlay, many-body state probs (`doc/TopicalAudit/ChargeRings_PME.md`); sites still abstract until pose SSOT (`RigidMoleculePose_SSOT.md`)
- **RigidAssemblyExtension.py** — Unified rigid-body panel: **Manipulate** maps scene atoms
  through PairFF dummy sites and relaxes all molecules concurrently with FAF; its probe map
  is the explicit `Probe E: static bodies + substrate` diagnostic with cached Total/PairFF/FAF
  layers. **MC/GA** reproduces the deterministic testplot initialization, **PME** uses
  ensemble CoM + R(q). `RigidEnsemble` is the pose authority; GPU and `_mb_*` mirrors are
  checked/synchronized. L0 tests: `tests/GUI/test_rigid_assembly_extension.py`.
- **RigidBodyVispy.py** — Standalone Vispy+Qt viewer for **PairFF** (FIRE, click-to-select active, map = PairFF[+FAF]); used by `demos/demo_pairff.py` — superseded for main-GUI use by `RigidAssemblyExtension` (`PairFF_GUI_Integration.md`)
- **AsciiArtExtension.py** — ASCII art → molecule; must match `build_ascii_hbond_system` pipeline for DFTB scans
- **ReactionCoordinateExtension.py** — H-bond RC scan: import graph, DFTB methods, slider, bond viz, ESP animation
- **rc_esp_view.py** — Blitted ESP heatmap synced to RC slider (uses `mpl_blit.py`)
- **rc_scan_gui_script.py** — Headless setup helper for review scripts
- **ChargeRingsExtension.py** — PME charge-ring STM panel (JSON I/O, xy/xV/1D, state probs)
- **FragmentExtension.py**, **DFTBExtension.py** — fragment browser / DFTB panel

## Control scripts (`demos/gui_scripts/`)

Scripts live in `demos/gui_scripts/` (centralized). Run via
`./run_gui.sh --script demos/gui_scripts/NAME.py [-- extra args]`
or select from the **Scripts → Bundled** menu (auto-discovered).
See [`demos/gui_scripts/README.md`](../../demos/gui_scripts/README.md) for full index.

- **folded_rigid_setup.py** — Folded-rigid molecule-on-surface manipulation: load molecule, fit/load potential, run/drag
- **conference_demo.py** — 4×PTCDA windmill candidate → untouched default 3ob DFTB+ AFM → LUMO BR-STM
- **ptcda_drag_demo.py** — Automatic 2×PTCDA stick-slip drag on NaCl → GIF + MP4 (`--format both`)
- **ptcda_interactive_drag.py** — 4×QEq-PTCDA/NaCl setup for all-mobile O-anchor stick-slip review
- **rc_scan_review.py** — 2Quinolone symmetric pm-NEB relaxed review; `--preview` loads cached npz
- **rc_scan_offline.py** — Same DFTB path without Qt (`PYTHONPATH=. python3 …`)
- **azaindol_draw_demo.py** — Live Ring/Atom/Select draw → full-window PNG frames → GIF (`--zoom-out`, `--canvas-only`)
- **azaindol_draw_offline.py** — Same sequence headless → one SVG per step (`PYTHONPATH=. python …`)

Shared sequence: `azaindol_draw_sequence.py`. Topical docs: `doc/Topics/ReactionCoordinateScan.md`, `doc/Topics/GUI_DrawDemo_Scripts.md`. Folder index: `gui_scripts/README.md`.
