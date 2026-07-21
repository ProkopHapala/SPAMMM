# Task: SPAMMM_GUI Editor — Orthographic 3D View Mode

**Status:** Done (USER confirmed 2026-07-21 — Ring/atom/bond hover in tilted view OK)  
**Priority:** P2 (GUI — human ToDo “3D view”)  
**Decision:** 3D is an **editor view mode** (not a separate app). Hex (and empty/link-to-empty) gated by `b2Dview`; Ring atom/bond/COG work in 3D.  
**Related:** `doc/GUI.desing.md`, `doc/GUI_CHEATSHEET.md`, `doc/GUI_topology_edit.desing.md`, `spammm/GUI/VispyUtils.py` (`AtomScene`), `spammm/GUI/SPAMMM_GUI.py`, `doc/ToDo/ToDo.human.md`, `doc/ToDo/ToDO_GUI.md`

> Marked Done after USER confirmation (2026-07-21).

---

## 1. Goal

Add a fast, orthographic **3D inspect/edit-lite** mode inside `SPAMMM_GUI`, while keeping the current **top-down 2D edit** workflow as the default and unbroken.

Primary use: assembly / stacking / multilayer debugging, quick standard views, correct picking under tilted cameras. **Hex grid** placement stays 2D-only; Ring atom/bond/COG work in 3D.

---

## 2. Locked USER decisions (SSOT)

| Decision | Choice |
|----------|--------|
| Architecture | **Editor view mode** with `b2Dview` |
| Projection | **Orthographic only** (`fov=0`) in both modes |
| `b2Dview` | bool, **True by default**; checkbox in Editors panel |
| **Enter** | Toggle `b2Dview` (2D ↔ 3D) |
| **Space** | Toggle run/stop interactive FF / simulation (`relax_interactive_btn` if present) |
| Mouse orbit | **RMB-drag on empty** in 3D (lowest priority after atom/bond). Arrows also rotate. Optional later: gizmo / Alt |
| Arrow keys | **2D:** pan (current). **3D:** rotate (az/el). **Shift+arrows:** pan in both |
| Future orbit | Optional later: gizmo corner, or hold unused modifier (Alt preferred — Ctrl=link, Shift=pan) |
| View presets | Digit / numpad keys; **`5` = Top (XY)** = default |
| Multi-viewport | **Deferred** — see §7 (VisPy shared-visual is hard; not worth it now) |
| Depth test | **On when `b2Dview=False`**; off in 2D (current overlay style) |
| “XY-clamped drag” | **Not a product feature** — see §2.1 clarification |
| Hex / empty / link-to-empty | Only when `b2Dview=True` |
| Atom / bond pick & drag | Work in both; 3D uses ray / view-plane drag |
| Rect select in 3D | Screen-space (Phase 2 polish if needed) |
| Debug | Cursor + mouse ray line (and optional triad) to validate transforms |

### 2.1 Clarification: no “XY-clamped drag” mode

Earlier discussion mentioned `_clamp_xy` / “XY-clamped drag”. That is **not** needed as a user-facing mode:

| Context | Behavior |
|---------|----------|
| **2D edit (`b2Dview`)** | Mouse → ray ∩ **z=0** (or keep atom’s z while moving XY). Planar construction. |
| **3D atom drag** | Drag in **view plane** through the atom (existing `pick_mode='3d'`). |
| **FF interactive pull** | Spring target = **nearest point on mouse ray** (future/align with `ToDO_GUI.md`). |
| **Bond draw** | Atom↔atom only; no empty spawn in 3D. |
| **Ring draw** | Bond-based; ring plane uses **z as normal** for planar rings when relevant; hex-grid branch 2D-only. |

No separate “clamp to XY while tilted” toggle.

---

## 3. Current baseline

- `AtomScene`: VisPy + `TurntableCamera`, already **`fov=0`**, default top-down; `_lock_top_view=True`, `_pick_mode='2d'`.
- Hooks exist: `set_lock_top_view()`, `set_pick_mode('2d'|'3d')`, 3D drag plane.
- GUI handlers historically always **ray ∩ z=0**.
- Graph `pick_*` use full 3D distance.
- **Depth test is currently OFF** on atoms/bonds (`set_gl_state(..., depth_test=False)`) — correct for flat 2D overlay; must enable in 3D.
- `MoleculeViewer` / MolBrowser: separate path; out of scope for Phase 1.

---

## 4. Control map (locked)

### 4.1 Modes

| | `b2Dview=True` (default) | `b2Dview=False` |
|--|--------------------------|-----------------|
| Camera | Locked top, az=0, el=90 | Unlocked; ortho; keyboard rotate + presets |
| Pick | `'2d'` (XY / z=0) | `'3d'` (ray–atom) |
| Depth test | Off | On (atoms + bonds) |
| Hex / empty / link-to-empty | Enabled | **Disabled** (status hint) |
| Mouse orbit | Off | Off (v1) |

### 4.2 View presets (Z-up; `5` = Top)

```
  7         8 Back      9
  4 Left    5 Top       6 Right
  1         2 Front     3
            0 Bottom
```

| Key (also KP_*) | View | elevation | azimuth |
|-----------------|------|-----------|---------|
| **5** | **Top (+Z)** | 90 | 0 |
| 0 | Bottom (−Z) | −90 | 0 |
| 8 | Back | 0 | 180 |
| 2 | Front | 0 | 0 |
| 4 | Left | 0 | −90 |
| 6 | Right | 0 | 90 |

Keep Ring mode **KP_ADD / KP_SUBTRACT** for ring size.

### 4.3 Feature gate

```text
if not gui.b2Dview and op in (hex, empty, link_to_empty):
    status "2D-only — press Enter for 2D view"
```

---

## 5. Multi-viewport — deferred (why)

VisPy does **not** cleanly support one Visual tree in multiple ViewBoxes (single-parent nodes; gallery duplicates visuals; issues #1124 / #1666). True shared scene is high cost / fragile.

**Decision:** skip multi-viewport for now. Single editor viewport + keyboard presets is enough. Revisit later only if assembly debugging strongly needs side-by-side Top/Front (then: synced duplicate visuals or second canvas — not shared SubScene).

---

## 6. Implementation plan

### Phase 1 (this session) — core

1. `b2Dview` + checkbox + `apply_view_mode()`
2. Enter ↔ view; Space ↔ interactive FF run
3. Gate hex / empty / link-to-empty
4. Presets `5/0/8/2/4/6` (+ KP_*)
5. Arrows: pan (2D) / rotate (3D); Shift+arrows pan
6. No mouse orbit (`_allow_mouse_orbit=False`)
7. Depth test on in 3D
8. Debug cursor + ray line
9. 3D GUI pick path for bond (ray); atoms already via scene

### Phase 2 — polish

- Screen-space rect select in 3D
- Rubber-band link line in 3D (not forced to z=0)
- Optional Alt+drag orbit / corner gizmo
- Cheatsheet / design doc sync

### Out of scope

- Multi-viewport, perspective, hex in 3D, MolBrowser merge

---

## 7. Files

| File | Role |
|------|------|
| `spammm/GUI/VispyUtils.py` | presets, arrows, depth_test, debug ray, no mouse orbit |
| `spammm/GUI/SPAMMM_GUI.py` | `b2Dview`, Enter/Space, checkbox, mouse path |
| `spammm/GUI/EditModeHandlers.py` | gate hex/empty |
| `doc/GUI_CHEATSHEET.md` | keys |
| `doc/Tasks/GUI_Editor_3D_ViewMode.md` | this file |

---

## 8. Success criteria

- Startup 2D identical to today.
- Enter → 3D ortho; hex/empty disabled; atom/bond still work; depth occlusion OK.
- `5` → Top; arrows rotate in 3D; Space toggles interactive FF when panel exists.
- Debug ray matches picks.
- No 2D regression.

---

## Session notes

**2026-07-21** — Design; USER locked Enter/Space, keyboard rotate (no mouse orbit), `5`=Top, defer multi-view, depth_test in 3D, no XY-clamp product mode.  
**2026-07-21 (evening)** — Fixes: (1) EventEmitter loop from `sig_camera_changed→refresh_view` during RMB rotate — camera moves use `canvas.update()` only. (2) Ring mode 3D: ray pick for bonds/atoms; side/preview mouse = ray∩z=0. (3) Ring `lock_drag` no longer leaves fake drag release.
**2026-07-21 (later)** — Ring hover: closest atom/bond/ring-COG along ray; only hex 2D-blocked. USER confirmed → docs/ToDos updated; **Status: Done**.
