---
type: TopicalAudit
title: GUI Draw Demo Scripts (SVG + GIF)
tags: [gui, editor, svg, gif, screenshot, azaindol, scripting, presentation]
timestamp: 2026-07-29
---

# GUI Draw Demo Scripts (SVG + GIF)

## Summary

Reproduce a full molecular-editor workflow as a **shared step sequence** that drives the same APIs a user hits in `SPAMMM_GUI` (Ring / Atom / Select, Auto H, Ctrl-C/V, δ/φ transforms). Two runners share that sequence:

1. **Offline SVG** — headless `MoleculeEditorBackend` + matplotlib ball-stick → one `.svg` (and `.png` sibling) per step under `debug/azaindol_draw_offline/`.
2. **GUI GIF** — live window via `./run_gui.sh --script …` → VisPy hover/cursor chrome + full-window PNG frames → animated GIF under `debug/azaindol_draw_demo/`.

Motivation: systematic GUI debugging, reproducible presentation of how rings/N-substitution/copy-paste look in the real UI, and parity between semantic 2D illustrations (SVG) and pixel screenshots (GIF).

**Reference molecule:** 7-azaindole monomer → H-bond dimer (`data/xyz/azaindol.xyz`, `data/xyz/azaindol_dimer.xyz`).

## Architecture

```
                    azaindol_draw_sequence.run_azaindol_draw(host)
                    ┌─────────────────┴─────────────────┐
                    │  shared ops (add_hex, adj ring,   │
                    │  set N, H caps, copy/paste, δ/φ)  │
                    └─────────────────┬─────────────────┘
          SequenceHost.snapshot()     │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  OfflineHost                     GuiHost
  render_editor_svg()             apply_demo_overlays() +
  → debug/.../*.svg               capture_window_png() → GIF
```

| Role | Location | Status | Notes |
|------|----------|--------|-------|
| Shared sequence SSOT | `spammm/GUI/azaindol_draw_sequence.py` | **active** | `run_azaindol_draw`, `render_editor_svg`, geometry helpers |
| GUI helpers | `spammm/GUI/gui_script_utils.py` | **active** | `apply_demo_overlays`, `capture_canvas_png`, `capture_window_png`, `frames_to_gif` |
| Script runner | `spammm/GUI/gui_script_runner.py` | **active** | loads `run(window, argv)` after `show()` |
| Offline entry | `spammm/GUI/gui_scripts/azaindol_draw_offline.py` | **active** | no Qt |
| GUI entry | `spammm/GUI/gui_scripts/azaindol_draw_demo.py` | **active** | widget-parity + GIF |
| Editor backend | `spammm/topology/MoleculeEditorBackend.py` | **active** | ops both hosts call |
| VisPy scene | `spammm/GUI/VispyUtils.py` | **active** | `fit_to_atoms`, selection δ/φ, `ring_preview_line` |
| Mode handlers | `spammm/GUI/EditModeHandlers.py` | **active** | Ring foreshadow / hover semantics mirrored by overlays |

## Tutorial

### Offline SVG sequence

```bash
PYTHONPATH=. python spammm/GUI/gui_scripts/azaindol_draw_offline.py
# optional: --relax  (DFTB after H caps)  --out DIR  --save-xyz PATH
```

Artifacts: `debug/azaindol_draw_offline/{00_empty,…,09_done}.svg` (+ `.png` siblings), `azaindol_dimer_drawn.xyz`.

### GUI demo → GIF

```bash
./run_gui.sh --script spammm/GUI/gui_scripts/azaindol_draw_demo.py
./run_gui.sh --script spammm/GUI/gui_scripts/azaindol_draw_demo.py -- --zoom-out 2 --gif-ms 550
./run_gui.sh --script spammm/GUI/gui_scripts/azaindol_draw_demo.py -- --canvas-only   # VisPy only
```

Artifacts: `debug/azaindol_draw_demo/*.png`, `azaindol_draw_demo.gif`, `azaindol_dimer_drawn.xyz`.

Offscreen smoke (CI / agents):

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=. python3 -c "
from PyQt5 import QtWidgets
from spammm.GUI.SPAMMM_GUI import SPAMMMWindow
from spammm.GUI.gui_script_runner import run_gui_script
app = QtWidgets.QApplication([]); w = SPAMMMWindow(); w.resize(1280,900); w.show()
run_gui_script(w, 'spammm/GUI/gui_scripts/azaindol_draw_demo.py', ['--gif-ms','500'])
"
```

## Frame story (mouse actions)

Order matters for UX: **hover / foreshadow before click**, then materialize.

| Frame | Intent |
|-------|--------|
| `00_empty` | Ring mode, Auto H off |
| `00b_hover_hex` | Cursor + orange hex nodes + cyan hex foreshadow |
| `01_hex` | Click → hex on grid |
| `01b_hover_bond` | Hover fusion bond: lime bond + **cyan 5-ring foreshadow** |
| `02_pentagon` | Click → fused pentagon |
| `02b_hover_N1` | Atom mode N — hover pyridine site |
| `02c_N1_done` | First C→N |
| `02d_hover_N2` | Hover pyrrole site |
| `03_azaindol_skel` | Second C→N (`nπ=0` for NH) |
| `04_hydrogens` | Auto H / `add_h_caps` |
| `05_*` … `09_done` | Select → copy → paste → φ rotate → δ translate → dimer |

Dimer pose: 180° about selection COM + COM shift from reference `azaindol_dimer.xyz` (heavy-atom COMs).

## Two render paths (what each is for)

| | Offline SVG | GUI PNG/GIF |
|--|-------------|-------------|
| **Purpose** | Crisp presentation / versionable slides | Prove real UI + debug chrome |
| **Compute** | Backend only | Same ops through `window.backend` + mode widgets |
| **Overlays** | matplotlib: cursor, hex hover, bond, ring preview, δ/φ AABB | VisPy: `cursor_markers`, `hover_markers`, `hover_bond_line`, `ring_preview_line`, selection handles |
| **Window chrome** | No | Yes (sidebar/status); OS titlebar via `grabWindow` when platform allows |
| **OpenGL caveat** | N/A | `capture_window_png` **composites** `canvas.render()` into the Qt grab (grab alone often blanks GL) |

## Design notes & pitfalls

- **Same tools as the user:** GUI host sets Ring/Atom/Select, element combo, Auto H, then calls the same backend methods `EditModeHandlers` use (`add_ring`, `add_adjacent_ring`, `set_atom_type_by_id`, `copy_selected_atoms` / `paste_copied_atoms`, `translate_selected` / `rotate_selected`).
- **Atom `_id` is global** across `MoleculeEditorBackend` instances — scripts must pick bonds/atoms by geometry (e.g. rightmost hex edge), not hardcoded IDs.
- **Pyrrole NH:** after C→N on the pentagon, set `npi=0` so `add_h_caps` adds H (`target_σ = 3`); pyridine N stays `npi=1`.
- **VisPy `ring_preview_line`:** never `set_data` an empty array then refill — offscreen `canvas.render()` can segfault. `apply_demo_overlays` only toggles visibility / replaces with real vertices.
- **Camera:** `AtomScene.fit_to_atoms(margin=…)` with `--zoom-out` (default 2) and a minimum `scale_factor` so early single-hex frames stay readable.
- **Pattern reuse:** same `./run_gui.sh --script` contract as RC scan / folded-rigid (`run(window, argv=None)`).

## Related docs

- Index: [topical_audit.md §1f–1g](../topical_audit.md)
- Cheatsheet (modes / δφ): [GUI_CHEATSHEET.md](../GUI_CHEATSHEET.md)
- Script launcher notes: [Takeways.md](../Takeways.md)
- Folder indexes: [spammm/GUI/README.md](../../spammm/GUI/README.md), [gui_scripts/README.md](../../spammm/GUI/gui_scripts/README.md)
- Sibling pattern: [ReactionCoordinateScan.md](ReactionCoordinateScan.md)
- Editor ring ops: [ARCHITECTURE_ROADMAP.md](../ARCHITECTURE_ROADMAP.md) (N-gon placement)
- FireCore analogue (not ported): `ScriptRunner.js` whitelist commands — SPAMMM uses Python hosts instead

## Open issues

- No dedicated pytest yet for azaindol draw (RC has `test_rc_scan_gui_script.py`); offscreen script run is the smoke check.
- AFM/STM after dimer left out of this demo by design.
- Wayland / some compositors may omit OS titlebar in `grabWindow`; client area + sidebar still captured.
