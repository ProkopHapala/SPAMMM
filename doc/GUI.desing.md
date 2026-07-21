# SPAMMM GUI — Kekule Structure Explorer

## Visual Design Principles

This is **serious scientific software**, not a marketing webpage. The GUI must give the user great power to control everything, fitting hierarchical levels of deeper and deeper control into a limited-size window in an ergonomic way.

**Screen space, user attention, and user time are limited resources. We must use them efficiently and not waste them on bullshit.**

1. **Fast and snappy** — UI must respond instantly. No animations, no transitions, no multi-frame effects. Show/hide is `setVisible(True/False)`, period. If something can be done in 0 ms, do it in 0 ms.
2. **Minimalist** — No decoration, no margins, no padding, no rounded corners, no gradients, no shadows. Every pixel must earn its place by carrying information or enabling control. Qt default styles are already too bloated — strip them where possible.
3. **Highly informative** — The user must see all relevant state at a glance: cache status, dirty flags, numerical ranges, error messages. Status labels and diagnostic output are first-class citizens, not afterthoughts.
4. **Hierarchical control** — Collapsible sections give access to deeper levels of control without cluttering the default view. Common operations are visible; advanced parameters are one click away. The hierarchy is: main buttons → parameters → visualization → advanced/diagnostics.
5. **No wasted space** — Side panel is fixed-width, wrapped in a `QScrollArea`. Expanding sections scrolls, never resizes the window. Compact spacing, no empty gaps, no unnecessary group box borders.
6. **No bullshit** — No tooltips that state the obvious. No confirmation dialogs for reversible actions. No progress bars for sub-second operations. No "welcome" screens. The user is a scientist who knows what they want.
8. **Keyboard-first** — Mouse is for the 3D viewport. Side panel controls should be reachable via keyboard (Tab navigation, Enter to apply). Shortcuts for frequent operations.

---

## Overview

Interactive molecular editor for designing carbon-based nanostructures (graphene, nanoribbons, functionalized edges, 2D heterostructures). Built on PyQt5 + Vispy, with a hexagonal grid for graphene-like scaffolding and a real-time 3D viewport.

**Launch:** `./run_gui.sh`

---

## Architecture (3 layers)

| Layer | Class | File | Role |
|-------|-------|------|------|
| Topology | `AtomicGraph` | `spammm/topology/AtomicGraph.py` | Authoritative object graph: `Atom`/`Bond`/`Ring` with stable `_id`, soft-delete |
| Backend | `KekuleBackend` | `spammm/topology/KekuleBackend.py` | Bridge: exports dense numpy arrays + ID↔index mappings for rendering/FF |
| Rendering | `AtomScene` | `spammm/GUI/VispyUtils.py` | Vispy canvas: atom markers, bonds, labels, picking, drag, selection |
| Controller | `KekuleExplorerWindow` | `spammm/GUI/SPAMMM_GUI.py` | PyQt5 main window: side panel, menus, signal dispatch, mode handling |
| Grid | `HexGrid` | `spammm/topology/HexGrid.py` | Honeycomb ruler with offset/rotation/transpose transforms |

**Key design principle:** All GUI signals and operations use `Atom._id` (stable, never reindexed), not array indices (ephemeral, rebuilt on every `_sync_sys()`). See `doc/GUI_topology_edit.desing.md` for internal details.

---

## Edit Modes

| Mode | LMB | Ctrl+LMB | RMB | Ctrl+RMB | Description |
|------|-----|----------|-----|----------|-------------|
| **Unified** | Atom: click=cycle type / drag=move; Bond: cycle order; Hex: add ring; Empty: add atom | Atom: drag to atom=bond / drag to empty=new atom+bond; Bond: insert atom (push aside) | Atom: delete; Bond: delete; Hex: remove ring (preserve shared) | Atom: delete+bridge; Bond: collapse | Context-sensitive: target determined by cursor position (atom > bond > hex > empty). Default mode. |
| **Hex1** (paint) | Add hex ring | — | Remove hex ring | — | Force add/remove; shared atoms between rings are also removed |
| **Hex2** (toggle) | Add hex ring | — | Remove hex ring (preserve shared) | — | Like Hex1 but preserves atoms shared with neighboring rings |
| **Atom** | Click: change type / Drag: move atom / Empty: add atom | Drag to atom: create bond / Drag to empty: new atom + bond | Delete atom | Delete atom + bridge neighbors | Free atom placement, drag-to-bond, type change |
| **Bond** | Insert atom (no push) | Insert atom (push aside) | Delete bond | Collapse bond (merge atoms) | Edit existing bonds with Ctrl for atom adjustment |
| **Ring** | Add n-gon ring on bond | — | Delete bond | — | Click bond to create n-membered ring (default 5) sharing that bond; ring size spinbox + numpad +/- in side panel; ghost preview on hover shows ring on mouse side |
| **pi** | Click: cycle pi-orbital count (0→1→2→0) / Empty: add atom | — | Delete atom | — | Adjust hybridization (sp3→sp2→sp) on clicked atom |
| **Select** | Drag selected atoms | — | Rectangle select / Delete | — | RMB drag = selection box; Delete = remove selected; Ctrl-C/V = copy/paste |

### Mode handler architecture

Each edit mode is implemented as an `EditModeHandler` subclass in `spammm/GUI/EditModeHandlers.py`. The base class defines overridable methods (`on_press`, `on_move`, `on_atom_click`, `on_link`, `on_rmb_atom`, `on_activate`) and class attributes (`link_mode`, `lock_drag`, `selection_mode`, `ring_size_visible`, `status_msg`). The GUI controller (`KekuleExplorerWindow`) dispatches signals to the active handler via a `mode_handlers` dict. Extensions register minimal `EditModeHandler` instances.

**Key class attributes set by each mode:**

| Mode | `link_mode` | `lock_drag` | `selection_mode` |
|------|-----------|-----------|-----------------|
| Unified | True | False | False |
| Atom | True | False | False |
| Bond | False | True | False |
| Select | False | False | True |

### Atom types
- Available: C, N, O (selectable via combo box)
- Each atom has an `npi` integer field (pi-orbital count): `-1`=H_cap, `0`=sp3, `1`=sp2 (default), `2`=sp
- H caps auto-added/removed when `Auto H` is enabled (default: on)

---

## Hexagonal Grid

The grid is a honeycomb lattice with configurable transform:

| Parameter | Control | Default | Description |
|-----------|---------|---------|-------------|
| `a_CC` | Spinbox | 1.42 Å | C-C bond length (hexagon circumradius) |
| Rotation | Spinbox (degrees) | 0° | Counter-clockwise rotation of grid |
| Offset X/Y | Spinbox | 0, 0 | Grid origin shift (in Å or grid units) |
| Transpose | Toggle button | Off | Swap x↔y axes (reflect along y=x) |
| Reset | Button | — | Reset all transforms to identity |

Grid transform changes call `reassign_pins()` which re-snaps all alive heavy atoms to the new grid node positions. Atoms not near any node get `pin=None`.

**Grid guide dots:** Faint markers show all grid nodes in range — visual aid for hex placement.

**Hex ring placement:** Click in Hex1/Hex2 mode snaps to the nearest hexagon center (axial coordinates q, r). The 6 ring nodes are placed at the hexagon vertices. Shared nodes between adjacent hexagons are automatically detected (same grid pin key).

---

## Ribbon Builder

Pre-built nanoribbon generator for quick starts:

### Single ribbon
- **Rows:** Number of zigzag chains (width)
- **Bottom/Top passivation:** String encoding (e.g. `nnHHoo`)
  - `n` = NH, `N` = N, `o` = C=O, `O` = O, `H` = CH, `h` = C-OH
- Generates a periodic zigzag graphene nanoribbon with specified edge passivation

### Two-ribbon (stacked)
- Two ribbons stacked with an H-bond gap
- **R2 rows:** Width of second ribbon
- **H-bond spacing:** Vertical distance between ribbons (default 3.0 Å)
- Each ribbon has independent passivation strings

---

## Atom Interaction

### Mouse
| Button | Action |
|--------|--------|
| **LMB** | Mode-dependent (add/toggle/insert — see Edit Modes) |
| **Ctrl+LMB drag** (Unified/Atom mode) | Drag from atom to atom: create bond (rubber-band line, green target highlight). Drag from atom to empty: create new atom at release position + bond to source atom. |
| **RMB** | Mode-dependent (remove/collapse/select — see Edit Modes) |
| **Ctrl+RMB** (Unified/Atom mode) | Delete atom + bridge its 2 heavy neighbors with a new bond |
| **Middle-click** | Toggle H state on nearest atom |
| **Scroll** | Zoom in/out |
| **RMB drag** (Select mode) | Rectangle selection |
| **LMB drag** (Unified/Atom/Select mode) | Drag atom(s) to new position. Drop atom on top of another → merge (target survives, bonds transfer) |
| **LMB click** (Unified/Atom/Pi mode) | Click atom without dragging: cycle atom type (C→N→O→C) or cycle pi orbitals |

### Keyboard
| Key | Action |
|-----|--------|
| **Delete** | Remove all selected atoms (Select mode) |
| **Ctrl+C** | Copy selected atoms to clipboard |
| **Ctrl+V** | Paste copied atoms at original positions (duplicate) |
| **Ctrl+Z** | Undo last topology change |
| **Arrow keys** | Pan camera |

### Picking
- **Pick radius** (spinbox, default 0.5 Å): Maximum distance from mouse to atom for pick to register
- RMB picking uses ray-plane intersection with distance threshold (same transform as hover)
- Hover highlights nearest atom (yellow) and shows hex ring outline (cyan) in Hex modes

---

## Visualization

### Labels
| Mode | Shows |
|------|-------|
| Element+Index | e.g. `C3`, `N7` |
| Atomic Type | Z number |
| Pi Orbitals | Number of pi electrons (0/1/2) |
| Z-Height | z-coordinate |
| Charge | Partial charge (if computed) |
| Bond Lengths | C-C distance annotations |

### Overlays
- **Bond colors:** Toggle to color bonds by length (shorter=red, longer=blue)
- **Debug view:** Grid guide dots, node→atom mapping lines, hex ring outlines
- **Force arrows:** Red vectors showing atomic forces (when available from relaxation/SCF)

### Camera
- Orthographic top-down (TurntableCamera with `fov=0`, `elevation=90°`)
- RMB drag (non-Select modes) = rotate camera around z-axis
- Scroll = zoom (logarithmic scale)
- Arrow keys = pan

---

## File I/O

| Operation | Description |
|-----------|-------------|
| **Export XYZ** | Save current structure to `.xyz` file |
| **Show XYZ** | Print XYZ to console / dialog |
| **Load XYZ** | Load structure from `.xyz` file; atoms placed at actual positions, pins assigned if near grid nodes |
| **Screenshot** | Save Vispy canvas screenshot (via matplotlib right-click menu on plots) |

---

## Extensions

Extensions are dynamically loaded and add collapsible panels + custom edit/view modes to the side panel. Available extensions:

| Extension | Purpose | Dependencies |
|-----------|---------|--------------|
| **FireCore** | Fireball DFT calculations (SCF, forces, relaxation) | `fdata_dir` |
| **DFTB** | DFTB+ tight-binding calculations | — |
| **AFM** | AFM/STM image simulation (OpenCL) | `pyopencl` |
| **FF** (`ff`) | SPFF/UFF GPU relaxation panel | `pyopencl` |
| **Vibrations** (`vibrations`) | Normal modes: DFTB or UFF/SPFF Hessian, clickable mode table + plot | `pyopencl` for FF; `DFTB_EXE` for DFTB |
| **QEq** (`qeq`) | Charge equilibration (Cholesky/LU) | — |
| **SPFF** | Simple Pair Force Field (classical MD) | — |
| **Grid** | OpenCL grid projector for electron density | `pyopencl`, `fdata_dir` |
| **Psi4** | Psi4 quantum chemistry integration | `psi4` |
| **pySCF** | pySCF quantum chemistry integration | `pyscf` |
| **MolDyn** | Molecular dynamics simulation | `pyopencl` |
| **POVray** | POV-Ray ray-tracing export | — |

Extensions that fail to load (missing dependency) show a grayed-out panel with the reason.

---

## Topology Operations (Backend)

| Operation | Method | Notes |
|-----------|--------|-------|
| Add hex ring | `add_ring(q, r)` | Places 6 C atoms at grid nodes, bonds them |
| Remove hex ring | `remove_ring(q, r)` | Soft-deletes ring atoms (Hex2 preserves shared) |
| Add free atom | `_append_atom(pos, ename)` | Places atom at exact position, bonds to nearest heavy |
| Delete atom | `remove_atom_by_id(id)` | O(1) soft-delete + H cap cleanup + `adjust_h()` |
| Delete atom + bridge | `remove_atom_with_bridge(id)` | Remove atom + connect its 2 heavy neighbors (inverse of insert) |
| Batch delete | `remove_atoms_by_id(ids)` | Collect all, soft-delete each, one `_sync_sys()` |
| Change atom type | `set_atom_type_by_id(id, ename)` | Mutates element, adjusts H caps |
| Create bond | `graph.add_bond(a, b)` | Idempotent — returns existing if already bonded. Local neighbor update. |
| Delete bond | `delete_bond(bond)` | Soft-delete bond only, keep both atoms |
| Insert into bond | `insert_atom_into_bond(bond, ename, push_aside)` | Splits bond with new atom; `push_aside` controls if A/B move |
| Collapse bond | `collapse_bond(bond, pos)` | Merges two atoms into one at given position |
| Merge atoms (drag) | `merge_atoms(dragged_id, target_id)` | Drag atom onto another → target survives, dragged atom's heavy bonds transfer (no duplicates), H caps readjusted |
| Add adjacent ring | `add_adjacent_ring(bond, n_members, ename, side)` | Creates n-membered ring sharing picked bond as one edge; `side` (+1/-1) from mouse position |
| Ring preview | `compute_adjacent_ring_positions(bond, n, side)` | Pure geometry: returns n-gon vertex positions without creating atoms (for hover preview) |
| Adjust H caps | `adjust_h()` | Remove all H caps → re-add based on valency |
| Recalculate bonds | `recalc_bonds()` | Distance-based bond detection (dangerous: may create spurious bonds) |
| Snap to pins | `snap_atoms_to_pins()` | Reset all atom positions to their grid pin coordinates |
| Reassign pins | `reassign_pins()` | Re-snap all atoms to current grid (after transform change) |

### Hydrogen cap management
- `Auto H` (default on): Automatically adds/removes H atoms to satisfy valency
- H caps are `Atom` objects with `npi=-1` and `parent=<heavy_atom>`
- `adjust_h()` = `remove_h_caps()` + `add_h_caps()` + `_sync_sys()`
- Removing a heavy atom also removes its H caps

### Passivation groups
Pre-defined edge terminations for ribbons:

| Code | Group | Atoms added |
|------|-------|-------------|
| `n` | NH | N + H |
| `N` | N | N (replaces C) |
| `o` | C=O | O (double bond) |
| `O` | O | O (replaces C) |
| `H` | CH | H (keeps C) |
| `h` | C-OH | O + H (hydroxyl) |

---

## Key Caveats

1. **Coordinate transforms:** All mouse→world conversions must use `_ray_from_mouse` + `_intersect_ray_plane` (accounts for grid layout offset). Never use `_mouse_to_world_xy` for picking — it's missing the `view.pos` offset.

2. **Index vs ID:** Array indices are ephemeral (rebuilt on every `_sync_sys()`). All signals and persistent state use `Atom._id`. Mapping arrays (`_atom_ids`, `_id_to_idx`) bridge the two but are only valid within one render frame.

3. **Soft-delete:** Atoms are marked `alive=False`, not removed from dict. `to_arrays()` filters them out. `cleanup_invalid()` is deferred — not called after every deletion. Neighbor queries must filter by `n.alive` and `n.npi != -1` (exclude H caps).

4. **Local neighbor updates:** `add_bond` and `remove_bond` update `atom.neighbors` in-place (O(degree)). `add_bond` is idempotent — if a dead bond exists between the pair, it revives it instead of creating a duplicate. Global `sync_neighbor_lists()` is only needed for bulk operations (ring add/remove, `collapse_bond`).

5. **Double event handling:** Both `AtomScene` and `SPAMMM_GUI` handle `mouse_press`. The GUI handler checks `event.handled` and skips if the scene already handled it (e.g. atom picked, link mode active). `ev.handled = True` does NOT stop other callbacks in Vispy — the GUI must explicitly check it.

6. **`adjust_h()` reorders arrays:** H caps are removed and re-added, completely changing array ordering. This is why ID-based operations are essential — indices before `adjust_h()` are invalid after.

7. **Undo stack:** `UndoStack(maxlen=100)` stores `PackedMolecule` snapshots. `_push_undo()` is called before all graph mutations. Ctrl+Z restores previous state.

8. **Clipboard:** `PackedMolecule` serializes selected atoms + internal bonds. Ctrl+C stores packed + Qt clipboard (MOL/XYZ text). Ctrl+V rebuilds from packed.

9. **Drag-to-merge:** In Atom mode, dragging an atom onto another (within `pick_radius`) triggers `merge_atoms` — the non-dragged atom survives, all heavy-atom bonds transfer (idempotent `add_bond` prevents duplicates), H caps are removed and readjusted. Undo (Ctrl+Z) reverts.

10. **Ring mode ghost preview:** Hovering a bond in Ring mode shows a cyan n-gon outline on the mouse side of the bond. Numpad +/- changes ring size (3–12), synced with spinbox. In **3D view**, pick is closest of bond / atom / ring-COG along the mouse ray; side uses ray ∩ z=0; hex placement is 2D-only.

11. **Ortho 2D/3D view (`b2Dview`):** Default locked Top. `Enter` (or Editors checkbox) unlocks free ortho view; `RMB`-drag on empty rotates; digit keys for presets (`5`=Top). Depth test on in 3D. Details: `doc/Tasks/GUI_Editor_3D_ViewMode.md`, `doc/GUI_CHEATSHEET.md`.

See `doc/GUI_topology_edit.desing.md` for detailed internal design and bug history.

---

## Key Files

| File | Role |
|------|------|
| `spammm/GUI/SPAMMM_GUI.py` | Main window, side panel, signal dispatch, mode handler registry |
| `spammm/GUI/VispyUtils.py` | Vispy `AtomScene`: rendering, picking, drag, selection, camera, click-vs-drag detection |
| `spammm/GUI/EditModeHandlers.py` | `EditModeHandler` class hierarchy: per-mode logic (Unified, Atom, Bond, Hex, Pi, Select, Ring) |
| `spammm/GUI/BaseGUI.py` | PyQt5 widget factory base class (buttons, spinboxes, etc.) |
| `spammm/GUI/ExtensionManager.py` | Dynamic extension loading and UI integration |
| `spammm/GUI/CollapsibleSection.py` | Collapsible side panel sections |
| `spammm/topology/KekuleBackend.py` | Editing engine: all topology operations, H cap management, ribbon building |
| `spammm/topology/AtomicGraph.py` | Authoritative topology: `Atom`/`Bond`/`Ring` classes, soft-delete, `to_arrays()` |
| `spammm/topology/HexGrid.py` | Honeycomb grid with transform support (offset, rotation, transpose) |
