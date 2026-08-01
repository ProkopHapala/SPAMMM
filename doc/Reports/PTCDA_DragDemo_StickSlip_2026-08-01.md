---
type: Report
title: PTCDA drag stick-slip demo on NaCl — GUI script with FAF substrate visualization (session 2026-08-01)
status: delivered — USER confirmed GIF quality
tags: [PairFF, FAF, drag, stick-slip, PTCDI, NaCl, GUI-script, visualization, VisPy, substrate]
timestamp: 2026-08-01
related:
  - doc/Tasks/Drag_Demo_Issues.md
  - doc/Tasks/GUI_Scripting_DemoRunner.md
  - doc/Tasks/GUI_Script_AFM_Parity_Bug.md
  - doc/Tasks/PairFF_MapDisplay_SSOT.md
  - doc/TopicalAudit/PairFF_RigidBody.md
  - doc/Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md
skills: [code-reuse, doc-read-navigate, centralized-plotting]
---

# PTCDA drag stick-slip demo on NaCl — GUI script with FAF substrate visualization

**Status:** delivered — USER confirmed the final GIF shows real stick-slip with substrate, anchor line, and growing trajectories.  
**Artifacts:** `debug/ptcda_drag_demo/ptcda_drag_demo.gif`, `frame_first.png`, `frame_last.png`  
**Script:** `spammm/GUI/gui_scripts/ptcda_drag_demo.py`  
**Run:** `./run_gui.sh --script spammm/GUI/gui_scripts/ptcda_drag_demo.py`

---

## 1. User goal

Produce a **conference-quality GIF** demonstrating **stick-slip dragging of a PTCDA molecule on NaCl**:

1. Build 2×PTCDA on NaCl via the **GUI** (not a standalone script).
2. Programmatically drag one molecule's **corner oxygen** across the surface.
3. Show **real lattice-periodic stick-slip** (not artificial spring-induced jumps).
4. Render the **FAF substrate potential map** (NaCl corrugation) under the molecules — the same visualization as `demos/demo_pairff.py`.
5. Show the **anchor spring line** (atom → target) and **growing trajectory lines** for the dragged atom, anchor target, and opposite corner atom.
6. The script must use the **same code path as interactive mouse dragging** — it is a GUI script, not an offline bypass.

### User preferences (documented for future sessions)

- **Reuse existing visualization code.** Never reinvent rendering. The FAF substrate map must use `potential_to_rgba` from `RigidBodyVispy` (display SSOT) — the same function `demo_pairff.py` uses. Do NOT draw giant Na/Cl atom markers instead of the potential heatmap.
- **GUI scripts simulate user actions.** A GUI script must call the same functions the GUI buttons / mouse handlers call. Bypassing the GUI with direct `rbd.run_multimol_md()` + matplotlib rendering defeats the purpose.
- **Do not move the camera during capture.** `capture_canvas_png(fit=False)` — the viewport must stay fixed so the viewer can follow the motion.
- **Show the anchor.** A red line from the dragged atom to the anchor target + a red cross marker at the target. This must also work during interactive mouse dragging, not only in the demo script.
- **Show growing trajectories.** Three colored polylines (dragged atom=red, anchor target=blue, opposite corner=green) that grow as the drag progresses, appending one point per frame.
- **Drag the corner oxygen**, not the central anhydride O. O[29] (top-right corner, x=5.85, y=2.20) is the default; O[27] (bottom-left) is the opposite corner.

---

## 2. Problems and caveats — what went wrong

### 2.1 First attempt: standalone matplotlib script (`tests/run_drag_demo.py`)

**User feedback:** Stick-slip looked artificial, no surface visible, other molecules didn't interact with substrate, should be a GUI script not standalone.

**Root cause:** The script bypassed the GUI entirely — direct `rbd.run_multimol_md()` calls, matplotlib rendering with colored circles, no NaCl lattice or FAF heatmap. The stick-slip was real physics but visually unconvincing because the surface was invisible and the spring was too weak/fast.

### 2.2 Second attempt: GUI script with Na/Cl giant markers

**User feedback (angry):** "are you even listening to me? I said use the surface visualization from `demo_pairff.py`! You implemented some your own CRAP! Huge atoms bigger than the whole PTCDA molecule and no map of the potential!"

**Root cause:** I added Na/Cl ion markers (`update_substrate_overlay` with disc markers size=120) to `VispyUtils` instead of using the **FAF potential heatmap** that `demo_pairff.py` shows via `RigidBodyVispy._recompute_map` + `potential_to_rgba`. The user had explicitly said to reuse `demo_pairff.py`'s substrate visualization. I did not listen.

**Fix:** Replaced `update_substrate_overlay` (markers) with `update_faf_map_overlay` (VisPy Image visual + `potential_to_rgba` display SSOT + `eval_folded_potential_grid`). This shows the same colored potential landscape as `demo_pairff.py`.

### 2.3 Camera moving during capture

**User feedback:** "Do not move the camera, it makes it difficult to follow the motion when the camera moves with the molecule."

**Root cause:** `GSU.capture_canvas_png()` defaults to `fit=True`, which calls `fit_to_atoms()` every frame, recentering on the molecule as it drifts.

**Fix:** `capture_canvas_png(..., fit=False)`. One `fit_to_atoms` at the start (after build), then the camera is locked for the entire drag.

### 2.4 No anchor / cursor visible

**User feedback:** "I still do not see the position of the anchor. If you cannot make mouse cursor where you want, at least render the real position of the anchor dragging the atom, and indicate the atom. We should draw a line connecting the anchor and anchored atom during dragging (also when user drags the atom by mouse)."

**Root cause:** The demo script set the anchor via `_set_anchors()` but never rendered it. The RA drag handler (`RAManipMode`) also had no anchor visualization — unlike `RigidBodyVispy` which has `anchor_line` + `anchor_marker`.

**Fix:** Added `_update_anchor_visuals(window, atom_pos, target)` to `RigidAssemblyExtension.py` — draws a red line (atom → target) + red cross marker (at target). Wired into `on_press`, `on_move`, `on_release` so it works for both interactive mouse dragging and the demo script.

### 2.5 Wrong atom dragged (central O instead of corner O)

**User feedback:** "We want to drag the corner oxygen not the central oxygen."

**Root cause:** Default `--anchor-atom` was 25 (central anhydride O at x=5.73, y=0). The corner O is O[29] at (5.85, 2.20).

**Fix:** Changed default to `--anchor-atom 29`. Added `--opposite-atom 27` for the opposite corner trajectory.

### 2.6 No trajectory history

**User feedback:** "Make visible also the whole trajectory of the dragged atom, the anchor point and the opposite corner atom, so there should be 3 lines, the line gradually grows as we drag it and new points are appended."

**Fix:** Three VisPy Line visuals (`connect='strip'`) with growing point arrays, appended one point per frame in `capture_frame()`. Colors: red (dragged O), blue (anchor target), green (opposite corner O).

---

## 3. What was delivered

### 3.1 Final GIF

`debug/ptcda_drag_demo/ptcda_drag_demo.gif` — 81 frames, 682×709 px, ~6.5 MB.

Shows:
- 2×PTCDA molecules on NaCl (FAF potential heatmap background, colored by `potential_to_rgba`)
- Red line from dragged corner O[29] to anchor target (spring visualization)
- Red cross marker at anchor target
- Three growing trajectory polylines (red/blue/green)
- Stick-slip physics: E oscillates between -2.38 eV (stuck) and -1.56 eV (loaded), period = 4.0 Å = one NaCl lattice cell

### 3.2 Physics results

| Parameter | Value |
|-----------|-------|
| Molecules | 2×PTCDA |
| Substrate | NaCl (FAF folded basis) |
| Anchor atom | O[29] (top-right corner carbonyl) |
| Spring constant | 0.2 (PairFF internal units) |
| Drag step | 0.2 Å |
| Total drag | 16.0 Å (80 steps) |
| FIRE relaxation | 200 steps per drag step |
| Stick-slip period | 4.0 Å (= one NaCl lattice cell) |
| E_stuck | -2.38 eV |
| E_loaded (before slip) | -1.56 eV |
| Slip event | Step 18: E jumps -1.56 → -2.36 |

The 4.0 Å period confirms the stick-slip is **lattice-periodic** — the molecule locks into each NaCl site, the spring loads, then it slips to the next site. This is real surface physics, not a spring artifact.

---

## 4. Code changes

### 4.1 Shared FAF map overlay (`spammm/GUI/VispyUtils.py`)

**New function:** `update_faf_map_overlay(scene, fit, z_eval, extent, step=0.1, probe_type='auto', image_attr='faf_map_image', visible=True)`

Creates/updates a VisPy Image visual showing the FAF substrate potential. Reuses:
- `spammm.GUI.RigidBodyVispy.potential_to_rgba` (display SSOT, vmax=|Emin|)
- `spammm.surfaces.FoldedRigid.eval_folded_potential_grid` (potential evaluation)

This is the shared substrate visualization that `FoldedRigidExtension`, `RigidAssemblyExtension`, and demo scripts all use. Any extension that has a FAF fit can now show the substrate map in the main AtomScene with one call.

### 4.2 Anchor visualization (`spammm/GUI/RigidAssemblyExtension.py`)

**New function:** `_update_anchor_visuals(window, atom_world_pos, anchor_target)`

Draws:
- Red line from the anchored atom to the anchor target (spring visualization)
- Red cross marker at the anchor target position

Cached on `window.ra_anchor_line` / `window.ra_anchor_marker`. Wired into `RAManipMode.on_press/on_move/on_release` so it works for interactive mouse dragging too — not just the demo script.

### 4.3 Substrate overlay in RA build (`spammm/GUI/RigidAssemblyExtension.py`)

**New function:** `_update_ra_substrate_overlay(window)`

Called after `_on_build` when FAF is enabled. Loads the FAF fit, computes the assembly extent, and calls `update_faf_map_overlay` to show the NaCl potential map under the molecules.

### 4.4 FoldedRigidExtension refactored (`spammm/GUI/FoldedRigidExtension.py`)

`_update_substrate_overlay` and `_update_potential_overlay` now call the shared `VispyUtils` functions instead of inline VisPy code. Removed duplicated `_SUB_COLORS` dict.

### 4.5 Demo script (`spammm/GUI/gui_scripts/ptcda_drag_demo.py`)

Complete rewrite. Uses the GUI code path:
- `GSU.click_button(window.ra_build_btn)` — build via GUI
- `_set_anchors(window, idx, target)` — same as drag handler `on_press`
- `rbd.run_multimol_md(n_relax, dt, fire=True)` — same as drag handler `on_move`
- `_sync_ensemble_from_gpu` + `_sync_display` — updates VisPy canvas
- `GSU.capture_canvas_png(..., fit=False)` — fixed camera
- `_update_anchor_visuals` — anchor line + marker
- Three growing trajectory lines (red/blue/green)

---

## 5. General takeaways

### 5.1 GUI scripts must use the GUI code path

A GUI script is a **thin orchestrator** that calls the same functions the GUI buttons and mouse handlers call. It must NOT bypass the GUI with direct `rbd.*` calls + custom rendering. This is the entire point of the GUI scripting system — reproducible demonstrations that are debuggable by repeating the same clicks manually.

**Pattern:** `GSU.click_button(window.ra_build_btn)` ✓ · `rbd.run_multimol_md()` + matplotlib ✗

### 5.2 Reuse existing visualization — never reinvent

Before writing any rendering code, check what `demo_*.py` scripts and existing extensions already show. The FAF substrate visualization existed in `RigidBodyVispy._recompute_map` + `potential_to_rgba` long before this session. The correct approach was to extract it to a shared function in `VispyUtils` and call it from the main scene — not to draw Na/Cl disc markers from scratch.

**Pattern:** `from spammm.GUI.RigidBodyVispy import potential_to_rgba` ✓ · `Markers(size=120, symbol='disc')` ✗

### 5.3 Camera must stay fixed during capture

`GSU.capture_canvas_png()` defaults to `fit=True` (refits every frame). For any animation where the molecule moves, use `fit=False` after the initial fit. A moving camera makes it impossible to follow the motion.

### 5.4 Anchor visualization belongs in the drag handler, not the script

The red spring line + cross marker was added to `RAManipMode.on_press/on_move/on_release` — not just the demo script. This means interactive mouse dragging also shows the anchor now. Demo-script-only visuals are a code smell; if it's useful for the demo, it's useful for the user.

### 5.5 Listen to the user

The user explicitly said "use the surface visualization from `demo_pairff.py`" multiple times. I ignored this and implemented my own Na/Cl marker visualization. This wasted a full iteration cycle. When the user names a specific file or function, read it first and reuse it — do not invent an alternative.

---

## 6. File inventory

| File | Role |
|------|------|
| `spammm/GUI/gui_scripts/ptcda_drag_demo.py` | Demo script: GUI build → drag → GIF |
| `spammm/GUI/VispyUtils.py` | Shared `update_faf_map_overlay` (FAF potential heatmap) |
| `spammm/GUI/RigidAssemblyExtension.py` | `_update_anchor_visuals`, `_update_ra_substrate_overlay` |
| `spammm/GUI/FoldedRigidExtension.py` | Refactored to use shared VispyUtils functions |
| `spammm/GUI/RigidBodyVispy.py` | `potential_to_rgba` (display SSOT, unchanged) |
| `demos/demo_pairff.py` | Reference for FAF substrate visualization (unchanged) |
| `debug/ptcda_drag_demo/` | GIF + first/last frames |
