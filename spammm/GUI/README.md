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
- **VispyUtils.py** — VisPy 3D scene (`AtomScene`), bond-length / Δlength coloring
- **EditModeHandlers.py** — Per-mode mouse dispatch (incl. `rc_pin` spatial constraints)
- **plotutils.py** — Qt 2D plot dialog; re-exports `spammm.plotUtils` grid/ESP helpers
- **mpl_blit.py** — Fast matplotlib updates via blit on embedded Qt canvas; caveats in `doc/Takeways.md`

## Extensions

- **AFMExtension.py** — FDBM AFM/STM panel (GUI uses ModularPipeline only). **K_LAT in N/m** → eV/Å² internally. Default plot z=3.0 Å; atom overlay checkbox replots live; atoms as `'.'` dots. Prefer grid step ≤0.1 Å. Perf via pipeline `SPAMMM_AFM_FAST_S3` — see `doc/Tasks/PerfBenchmark_FDBM.md`, `doc/Tasks/AFMTesting.md`.
- **KekuleExtension.py** — Kekulé π-bond-order solver panel
- **FoldedRigidExtension.py** — Folded-basis rigid-body manipulation panel (load molecule, fit/load substrate potential, drag atoms)
- **AsciiArtExtension.py** — ASCII art → molecule; must match `build_ascii_hbond_system` pipeline for DFTB scans
- **ReactionCoordinateExtension.py** — H-bond RC scan: import graph, DFTB methods, slider, bond viz, ESP animation
- **rc_esp_view.py** — Blitted ESP heatmap synced to RC slider (uses `mpl_blit.py`)
- **rc_scan_gui_script.py** — Headless setup helper for review scripts
- **FFExtension.py**, **QEqExtension.py**, **VibrationExtension.py** — FF relax / charge / normal-mode panels
- **FragmentExtension.py**, **DFTBExtension.py** — fragment browser / DFTB panel

## Control scripts (`gui_scripts/`)

Run via `./run_gui.sh --script spammm/GUI/gui_scripts/NAME.py [-- extra args]`:

- **folded_rigid_setup.py** — Folded-rigid molecule-on-surface manipulation: load molecule, fit/load potential, run/drag
- **rc_scan_review.py** — 2Quinolone symmetric pm-NEB relaxed review; `--preview` loads cached npz
- **rc_scan_offline.py** — Same DFTB path without Qt (`PYTHONPATH=. python3 …`)

Topical doc: `doc/Topics/ReactionCoordinateScan.md`
