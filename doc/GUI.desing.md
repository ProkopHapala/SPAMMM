# SPAMMM GUI — Kekule Structure Explorer

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

| Mode | LMB | RMB | Description |
|------|-----|-----|-------------|
| **Hex1** (paint) | Add hex ring | Remove hex ring | Force add/remove; shared atoms between rings are also removed |
| **Hex2** (toggle) | Add hex ring | Remove hex ring (preserve shared) | Like Hex1 but preserves atoms shared with neighboring rings |
| **Atom** | Add atom / change type | Delete atom | Free placement at click position; if near existing atom, changes its type |
| **Bond** | Insert atom into bond | Collapse bond (merge two atoms into one) | Click on a bond to split it with a new atom; RMB to merge |
| **pi** | Cycle pi-orbital count (0→1→2→0) | Delete atom | Adjust hybridization (sp3→sp2→sp) on clicked atom |
| **Select** | Drag selected atoms | Rectangle select / Delete | RMB drag = selection box; Delete = remove selected; Ctrl-C/V = copy/paste |

### Atom types
- Available: C, N, O (selectable via combo box)
- Each atom gets a subtype (e.g. `C_sp2`, `N_sp3`, `H_cap`) inferred from element and bonding
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
| **RMB** | Mode-dependent (remove/collapse/select — see Edit Modes) |
| **Middle-click** | Toggle H state on nearest atom |
| **Scroll** | Zoom in/out |
| **RMB drag** (Select mode) | Rectangle selection |
| **LMB drag** (Atom/Select mode) | Drag atom(s) to new position |

### Keyboard
| Key | Action |
|-----|--------|
| **Delete** | Remove all selected atoms (Select mode) |
| **Ctrl+C** | Copy selected atoms to clipboard |
| **Ctrl+V** | Paste copied atoms at original positions (duplicate) |
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
| Batch delete | `remove_atoms_by_id(ids)` | Collect all, soft-delete each, one `_sync_sys()` |
| Change atom type | `set_atom_type_by_id(id, ename)` | Mutates element, adjusts H caps |
| Insert into bond | `insert_atom_into_bond(bond, ename)` | Splits bond with new atom |
| Collapse bond | `collapse_bond(bond, pos)` | Merges two atoms into one at given position |
| Adjust H caps | `adjust_h()` | Remove all H caps → re-add based on valency |
| Recalculate bonds | `recalc_bonds()` | Distance-based bond detection (dangerous: may create spurious bonds) |
| Snap to pins | `snap_atoms_to_pins()` | Reset all atom positions to their grid pin coordinates |
| Reassign pins | `reassign_pins()` | Re-snap all atoms to current grid (after transform change) |

### Hydrogen cap management
- `Auto H` (default on): Automatically adds/removes H atoms to satisfy valency
- H caps are `Atom` objects with `subtype='H_cap'` and `parent=<heavy_atom>`
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

3. **Soft-delete:** Atoms are marked `alive=False`, not removed from dict. `to_arrays()` filters them out. `cleanup_invalid()` is deferred — not called after every deletion. Neighbor queries must filter by `n.alive`.

4. **Double event handling:** Both `AtomScene` and `SPAMMM_GUI` handle `mouse_press`. Mode checks prevent conflicts. `ev.handled = True` does NOT stop other callbacks in Vispy.

5. **`adjust_h()` reorders arrays:** H caps are removed and re-added, completely changing array ordering. This is why ID-based operations are essential — indices before `adjust_h()` are invalid after.

See `doc/GUI_topology_edit.desing.md` for detailed internal design and bug history.

---

## Key Files

| File | Role |
|------|------|
| `spammm/GUI/SPAMMM_GUI.py` | Main window, side panel, signal dispatch, mode handling |
| `spammm/GUI/VispyUtils.py` | Vispy `AtomScene`: rendering, picking, drag, selection, camera |
| `spammm/GUI/BaseGUI.py` | PyQt5 widget factory base class (buttons, spinboxes, etc.) |
| `spammm/GUI/ExtensionManager.py` | Dynamic extension loading and UI integration |
| `spammm/GUI/CollapsibleSection.py` | Collapsible side panel sections |
| `spammm/topology/KekuleBackend.py` | Editing engine: all topology operations, H cap management, ribbon building |
| `spammm/topology/AtomicGraph.py` | Authoritative topology: `Atom`/`Bond`/`Ring` classes, soft-delete, `to_arrays()` |
| `spammm/topology/HexGrid.py` | Honeycomb grid with transform support (offset, rotation, transpose) |
