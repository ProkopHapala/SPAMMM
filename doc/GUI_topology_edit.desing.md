# GUI Topology Editing — System Design & Caveats

## Overview

The GUI molecular editor has three layers with different data representations:

1. **AtomicGraph** — authoritative topology (dicts of `Atom`/`Bond`/`Ring` objects with stable `_id`)
2. **KekuleBackend** — bridge layer (dense numpy arrays for Vispy/FF + ephemeral mapping arrays)
3. **AtomScene (Vispy)** — rendering layer (dense arrays, mouse picking, drag, selection)

The central design challenge: **topology operations need flexibility (add/remove atoms, change bonds)**, while **rendering and force-field computation need compact dense arrays**. These two concerns are **decoupled**.

---

## Layer 1: AtomicGraph — Authoritative Topology

### Data structures

- `atoms: dict[int, Atom]` — keyed by `Atom._id` (stable unique counter, never resets)
- `bonds: dict[int, Bond]` — keyed by `Bond._id`
- `rings: dict[int, Ring]` — keyed by `Ring._id`
- `_pin_to_atom: dict[tuple, Atom]` — grid pin → atom (for grid-based placement)

### Atom identity

- `Atom._id` is a **monotonically increasing counter** (`Atom._counter += 1` on each `__init__`).
- It is **not an array index** — it can grow to billions without issue.
- It **never resets** — even after all atoms are deleted, new atoms get higher IDs.
- It serves as the **stable identity** across topology changes.

### Soft-delete (dead-slot pattern)

- `Atom.alive` (bool) — `True` by default, `False` when soft-deleted.
- `Bond.alive` and `Ring.alive` — same pattern.
- `remove_atom(soft=True)` flips `alive=False`, marks all its bonds dead, and **removes the atom from `_pin_to_atom` cache** (sets `atom.pin = None`). O(1) + O(degree).
- Dead atoms **stay in `atoms` dict** but are filtered out by `to_arrays()`.
- `cleanup_invalid()` prunes all dead objects from dicts — **deferred, not called after every deletion**.
- **Pin cache cleanup on soft-delete** ensures dead atoms don't block re-adding live atoms at the same grid node (e.g., add→remove→add ring cycle).

### Why soft-delete?

- **O(1) deletion** — no dict rebuilding, no index shifting.
- **Batch delete is trivially correct** — collect `_id`s, flip `alive=False` on each, one `_sync_sys()` at end.
- **No index instability** — the `_id`→`Atom` mapping in `atoms` dict never changes during soft-delete.
- Dead slots waste some memory, but for ~100-1000 atoms this is negligible. Dict growth is handled by Python's internal reallocation (exponential/golden ratio strategy).

### When to call `cleanup_invalid()`

- **Not** after every deletion — this was the old behavior, causing O(N) overhead per delete.
- Only when explicitly needed (e.g., "compact" button, or when dead fraction > threshold).
- Currently deferred indefinitely — dead atoms are simply filtered by `to_arrays()`.

### When to call `sync_neighbor_lists()`

- `Atom.neighbors` is a **derived** list rebuilt from alive bonds.
- `add_bond` and `remove_bond` now update `neighbors` **locally** (O(degree)) — no global sync needed for single bond add/remove.
- `add_bond` is **idempotent**: if a dead bond exists between the pair, it revives it (`alive=True`, restores neighbors) instead of creating a duplicate.
- After **soft-delete only**: neighbor lists become stale (contain dead atoms), but consumers filter by `n.alive`. No sync needed.
- After **collapse_bond**: topology changes, so `sync_neighbor_lists()` is called.
- `sync_neighbor_lists()` is only needed for **bulk operations** where many bonds change at once.

### `h_children()` is O(N)

- Iterates all atoms to find children by `parent` attribute and `npi == -1` (H_cap).
- For ~100 atoms this is fine. If it becomes a bottleneck, store a `children: list` on each `Atom`.

### Atom hybridization: `npi` (int, not string)

- `Atom.npi` is an **integer** representing pi-orbital count:
  - `npi = -1` → H_cap (hydrogen cap, not a hybridization state)
  - `npi = 0`  → sp3 (tetrahedral, 0 pi orbitals)
  - `npi = 1`  → sp2 (trigonal planar, 1 pi orbital) — **default**
  - `npi = 2`  → sp (linear, 2 pi orbitals)
- **No string subtypes** — the old `subtype` field (e.g., `'C_sp2'`, `'H_cap'`) has been fully replaced by `npi`.
- H caps are identified by `npi == -1` (not by string matching).
- `add_atom(pos, ename, atype, pin=None, parent=None, npi=1)` — `npi` parameter replaces old `subtype`.

---

## Layer 2: KekuleBackend — Bridge (Dense Arrays + Mappings)

### `_sync_sys()` — Export from graph to dense arrays

Called whenever consumers need fresh dense arrays. Rebuilds everything from `AtomicGraph.to_arrays()`:

```
atom_list, enames, apos, atypes, bonds, bond_list, ring_list = graph.to_arrays()
```

### 4 ephemeral mapping arrays

Built on each `_sync_sys()` call, **not maintained during topology ops**:

| Array | Type | Direction | Description |
|-------|------|-----------|-------------|
| `_atom_ids` | `np.int64[N]` | dense→graph | `_atom_ids[i]` = `Atom._id` at dense index `i` |
| `_atom_idx_map` | `dict[int,int]` | graph→dense | `_atom_idx_map[Atom._id]` = dense index `i` |
| `_bond_ids` | `np.int64[B]` | dense→graph | `_bond_ids[i]` = `Bond._id` at dense index `i` |
| `_bond_idx_map` | `dict[int,int]` | graph→dense | `_bond_idx_map[Bond._id]` = dense index `i` |

These are **stale between `_sync_sys()` calls** — but nobody uses them during that window. The GUI stores `_id` (stable), not dense index (ephemeral). The mapping is only used at the boundary (picking → emit `_id`, or `_id` → look up position in dense array), and it's always fresh because `refresh_view()` calls `_sync_sys()` before the scene reads arrays.

### ID-based removal methods

- `remove_atom_by_id(atom_id)` — O(1) lookup in `graph.atoms`, soft-delete, one `_sync_sys()`.
- `remove_atoms_by_id(atom_ids)` — collect all atoms + H children first, soft-delete each, one `_sync_sys()`.
- `set_atom_type_by_id(atom_id, ename)` — O(1) lookup, mutate, one `_sync_sys()`.

### `adjust_h()` — H cap management

- `remove_h_caps()` — soft-deletes all H caps (atoms with `npi == -1`, flip `alive=False`). No `cleanup_invalid()`.
- `add_h_caps()` — appends new H `Atom` objects to graph with `npi=-1`. Uses `n.alive` and `n.npi != -1` filters in neighbor queries to handle stale neighbor lists.
- `adjust_h()` = `remove_h_caps()` + `add_h_caps()` + `_sync_sys()`.
- Old dead H caps stay in dict but are filtered by `to_arrays()`.
- `_get_element_default_npi(element)` — returns `-1` for H, `1` (sp2) for all others. Replaces old `_get_element_default_subtype()`.
- `atom_npi` property — returns list of `a.npi` for all alive atoms. Replaces old `atom_subtype`.
- `set_atom_npi_by_id(atom_id, npi)` — sets `npi` directly on atom by stable `_id`. Replaces old `set_atom_subtype_by_id()`.
- `_target_sigma(element, npi)` — electron counting: `nsigma = nval - npi - nepair`. Uses `npi` directly (no string parsing).

### Caveat: neighbor queries after soft-delete

After soft-delete, `Atom.neighbors` lists may contain dead atoms. **All neighbor queries must filter by `n.alive`**:

```python
heavy_neighbors = [n for n in a.neighbors if n.alive and n.npi != -1]
```

This is the key insight that allows deferring `sync_neighbor_lists()` — consumers simply skip dead neighbors and H caps by checking `n.alive` and `n.npi != -1`.

---

## Layer 3: AtomScene (Vispy) — Rendering & Interaction

### Data stored in scene

- `_pos: np.float32[N,3]` — copy of `backend.sys.apos` (dense, only alive atoms)
- `_atom_ids: np.int64[N]` — copy of `backend._atom_ids` (parallel to `_pos`)
- `_id_to_idx: dict[int,int]` — `Atom._id` → dense index (rebuilt on `set_data()`)
- `_selected_ids: set[int]` — selection as set of `Atom._id` (not indices!)

### Signal contract — all signals emit `Atom._id`

| Signal | Payload | Description |
|--------|---------|-------------|
| `sig_rmb_remove` | `int` (Atom._id) | RMB click on atom → request deletion |
| `sig_atom_picked` | `int` (Atom._id) | LMB click on atom |
| `sig_drag_state` | `(int, int, object)` | (active 0/1, Atom._id, pos3) — emitted on drag end |
| `sig_atom_moved` | `(int, object)` | (Atom._id, pos3) during drag |
| `sig_selection_changed` | `set[int]` | set of Atom._ids |
| `sig_link_bond` | `(int, int)` | (from_id, to_id) — Ctrl+drag from atom to atom: create bond |
| `sig_link_to_pos` | `(int, float, float)` | (from_id, x, y) — Ctrl+drag from atom to empty space: create new atom + bond |
| `sig_atom_clicked` | `int` (Atom._id) | LMB click on atom without drag → cycle atom type |

### ID ↔ index conversion helpers

- `_idx_to_id(idx)` — `self._atom_ids[idx]` (array lookup, O(1))
- `_id_to_idx_safe(atom_id)` — `self._id_to_idx.get(atom_id, -1)` (dict lookup, O(1))

These are **only valid after `set_data()`** has been called (which rebuilds both from backend).

---

## Caveat 1: Coordinate Transform Bug (Critical)

### The bug

Two different methods existed for converting mouse pixel coordinates to world coordinates:

1. **`_ray_from_mouse(mouse_pos)`** — correctly subtracts `self.view.pos[:2]` (the view's offset within the grid layout):
   ```python
   view_pos = np.array(mouse_pos) - self.view.pos[:2]
   tr = self.view.scene.transform
   p0 = tr.imap((view_pos[0], view_pos[1], z0))
   ```

2. **`_mouse_to_world_xy(mouse_pos, z)`** — does **NOT** subtract `self.view.pos[:2]`:
   ```python
   tr = self.view.scene.transform
   p = tr.imap((mouse_pos[0], mouse_pos[1], z))  # BUG: missing view.pos offset
   ```

### Impact

- **Hover** used `_ray_from_mouse` + `_intersect_ray_plane` → correct world coordinates.
- **RMB picking** used `_mouse_to_world_xy` → **shifted world coordinates** by the view's grid offset.
- Result: clicking directly on an atom produced a consistent ~0.6-0.7 Å offset, causing the wrong atom (or no atom) to be picked.
- The user saw the hover highlight the correct atom, but RMB deleted a different one — a very confusing symptom.

### Fix

All 2D picking methods (`_pick_idx_from_mouse`, `_pick_idx_with_dist`) now use `_ray_from_mouse` + `_intersect_ray_plane`, same as hover. The broken `_mouse_to_world_xy` is no longer used for picking.

### Lesson

**Never use two different coordinate transform paths for the same logical operation.** Always use one consistent method. If the view is embedded in a grid layout, all transforms must account for `view.pos`.

---

## Caveat 2: Distance Threshold for Picking

### The bug

`_pick_idx_from_mouse` returned the **globally closest atom** with no distance threshold. If the mouse was between two atoms, it picked whichever was mathematically closer — even if the distance was 0.7+ Å (far from any atom).

### Fix

RMB picking now uses `_pick_id_from_mouse(pos, max_dist=self.pick_radius)`:
- Picks closest atom
- **Rejects if distance > `pick_radius`** (default 0.5 Å, configurable via spinbox)
- Returns `(atom_id, distance)` or `(-1, inf)` if no atom within threshold

The `pick_radius` is synced from the GUI's spinbox to `scene.pick_radius` on init and on value change.

### Lesson

**Always have a distance threshold for picking.** Without it, clicking empty space picks the nearest atom, which is rarely what the user wants. The threshold should match the visual atom size and be user-configurable.

---

## Caveat 3: Index vs ID — Why Indices Break

### The problem with array indices

Dense array indices are **ephemeral** — they change whenever `_sync_sys()` rebuilds arrays:
- Deleting atom at index 5 shifts all atoms after it.
- `adjust_h()` removes and re-adds H caps, completely reordering the array.
- Batch deletion of multiple atoms causes index shifting between each deletion.

### How it broke (old code)

1. User RMB clicks → scene picks index `i` → emits `i`
2. GUI receives index `i` → calls `backend.remove_atom_by_index(i)`
3. Backend removes atom, rebuilds arrays → **all indices shift**
4. Next RMB click uses stale `_pos` array → picks wrong atom

### How it works now (ID-based)

1. User RMB clicks → scene picks index `i` → converts to `atom_id = _atom_ids[i]` → emits `atom_id`
2. GUI receives `atom_id` → calls `backend.remove_atom_by_id(atom_id)`
3. Backend looks up `graph.atoms[atom_id]` (O(1) dict lookup) → soft-deletes
4. `_sync_sys()` rebuilds arrays + mappings → scene gets fresh data
5. Next RMB click uses fresh `_atom_ids` array → correct mapping

### Key insight

The **dense index is only valid within a single render frame**. Between frames, `_sync_sys()` may have rebuilt arrays. The `Atom._id` is the only stable reference. The mapping (`_atom_ids`, `_id_to_idx`) bridges the two worlds but must be rebuilt on every export.

---

## Caveat 4: Dead/Alive Atoms and Stale Neighbor Lists

### The problem

After soft-delete (`atom.alive = False`), the atom's neighbors still have it in their `neighbors` lists. If code iterates `a.neighbors` without filtering, it will process dead atoms.

### The solution

**All neighbor queries must filter by `n.alive`:**

```python
heavy_neighbors = [n for n in a.neighbors if n.alive and n.npi != -1]
```

This allows deferring `sync_neighbor_lists()` — the O(N+B) global rebuild — until it's actually needed (after bond creation, not after deletion).

### When `sync_neighbor_lists()` IS needed

- After `add_h_caps()` — new bonds were created (though `add_bond` now updates locally, bulk sync is a safety net).
- After `collapse_bond()` — topology changed, multiple bonds marked dead + created.
- **NOT** after `remove_atom_by_id()` — soft-delete + `n.alive` filter suffices.
- **NOT** after `delete_bond()` — `remove_bond` updates neighbors locally.
- **NOT** after `insert_atom_into_bond()` — `add_bond` updates neighbors locally.
- **NOT** after `add_adjacent_ring()` — `add_bond` updates neighbors locally.
- **NOT** after `merge_atoms()` — `add_bond` (idempotent revive) + `remove_atom` handle neighbors locally.

---

## Caveat 5: Double Event Handling

### The issue

Both `AtomScene._on_mouse_press` and `SPAMMM_GUI.on_mouse_press` are connected to the **same** `canvas.events.mouse_press` event. In Vispy, `ev.handled = True` does **not** stop other callbacks — both handlers fire.

### How it's handled

The GUI handler (`SPAMMM_GUI.on_mouse_press`) explicitly checks `event.handled` at the top and returns early if the scene already handled the event. This prevents double-fire.

**Event flow for LMB on atom (Unified/Atom mode):**
1. `AtomScene._on_mouse_press` picks atom (within `pick_radius`), sets `ev.handled = True`, starts drag state
2. `SPAMMM_GUI.on_mouse_press` sees `event.handled = True` → returns early (no double-fire)
3. On release: `AtomScene._on_mouse_release` differentiates click vs drag:
   - **Click** (mouse moved < 3px): emits `sig_atom_clicked(atom_id)` → `on_atom_clicked` → `EditModeHandler.on_atom_click` → `cycle_atom_type`
   - **Drag** (mouse moved ≥ 3px): emits `sig_drag_state(0, atom_id, pos)` → `on_drag_state` → sync positions / merge

**Event flow for LMB on empty space / bond / hex (Unified mode):**
1. `AtomScene._on_mouse_press` finds no atom within `pick_radius` → does NOT set `ev.handled`, returns
2. `SPAMMM_GUI.on_mouse_press` runs → `EditModeHandler.on_press` → `resolve_target` → dispatch to bond/hex/empty action

**Event flow for Ctrl+LMB on atom (Unified/Atom mode):**
1. `AtomScene._on_mouse_press` sees `_link_mode and _last_ctrl` → picks atom, starts link mode, sets `ev.handled = True`
2. `SPAMMM_GUI.on_mouse_press` sees `event.handled = True` → returns early
3. On release: `AtomScene._on_mouse_release` checks link target:
   - **On another atom**: emits `sig_link_bond(from_id, to_id)` → `on_link_bond` → `EditModeHandler.on_link` → `create_bond`
   - **On empty space**: emits `sig_link_to_pos(from_id, x, y)` → `on_link_to_pos` → create new atom at (x,y) + bond to source
   - **On same atom**: emits `sig_atom_clicked(from_id)` → cycle type

### Lesson

When multiple handlers are connected to the same event, ensure they don't conflict. The GUI must explicitly check `event.handled` since Vispy does not enforce it.

---

## Data Flow Summary

```
User RMB click on atom
    │
    ▼
AtomScene._on_mouse_press(ev)
    │  _pick_id_from_mouse(ev.pos, max_dist=pick_radius)
    │  → uses _ray_from_mouse + _intersect_ray_plane (correct transform)
    │  → returns (atom_id, distance)
    │
    ▼
sig_rmb_remove.emit(atom_id)  ← stable Atom._id
    │
    ▼
SPAMMM_GUI.on_atom_remove(atom_id)
    │  backend.remove_atom_by_id(atom_id)
    │
    ▼
KekuleBackend.remove_atom_by_id(atom_id)
    │  a = graph.atoms.get(atom_id)     ← O(1) dict lookup
    │  for h in graph.h_children(a):
    │      graph.remove_atom(h)          ← soft-delete H caps
    │  graph.remove_atom(a)              ← soft-delete atom
    │  adjust_h()                        ← remove/re-add H caps
    │  _sync_sys()                       ← rebuild dense arrays + 4 mappings
    │
    ▼
SPAMMM_GUI.refresh_view()
    │  scene.set_data(pos, colors, sizes, bonds)
    │
    ▼
AtomScene.set_data()
    │  _pos = backend.sys.apos.copy()
    │  _atom_ids = backend._atom_ids.copy()
    │  _id_to_idx = {id: idx for ...}
    │
    ▼
Vispy renders fresh scene
```

---

## Key Files

| File | Role |
|------|------|
| `spammm/topology/AtomicGraph.py` | Authoritative topology: `Atom`, `Bond`, `Ring` classes, soft-delete, `to_arrays()`, `cleanup_invalid()`, `sync_neighbor_lists()`, local neighbor updates in `add_bond`/`remove_bond` |
| `spammm/topology/KekuleBackend.py` | Bridge: `_sync_sys()` (export + 4 mappings), `remove_atom_by_id()`, `adjust_h()`, `add_h_caps()`, `merge_atoms()`, `add_adjacent_ring()`, `compute_adjacent_ring_positions()` |
| `spammm/GUI/VispyUtils.py` | Rendering: `AtomScene` class, picking, drag, selection, click-vs-drag detection, link mode, signals (all emit `Atom._id`) |
| `spammm/GUI/SPAMMM_GUI.py` | Controller: signal handlers, mode handler registry, `on_atom_clicked()`, `on_link_to_pos()`, `refresh_view()` |
| `spammm/GUI/EditModeHandlers.py` | `EditModeHandler` class hierarchy: `UnifiedMode`, `AtomMode`, `BondMode`, `HexMode`, `PiMode`, `SelectMode`, `RingMode` |

---

## Testing Checklist

- [ ] RMB click on atom → correct atom deleted (verify by position/element)
- [ ] RMB click on empty space → no deletion
- [ ] Batch RMB deletion (Select mode + Delete key) → all selected atoms removed
- [ ] Drag atom → position updates in graph and rendering
- [ ] Add hex ring → H caps auto-added → delete atom → H caps adjusted
- [ ] Change atom type (LMB click on existing atom) → correct atom changed
- [ ] Copy/paste selection → pasted atoms have new `_id`s
- [ ] Undo (Ctrl+Z) → graph restored to previous state
- [ ] Export/Import .xyz/.mol/.mol2 → round-trip preserves atoms, bonds, positions
- [ ] Pi-mode cycling (sp3→sp2→sp→sp3) → `npi` changes correctly, H caps adjust
- [ ] Add→remove→add ring cycle → atoms re-created correctly (pin cache not stale)
- [ ] Ring mode: LMB on bond → n-gon created on mouse side, correct size
- [ ] Ring mode: numpad +/- changes ring size, spinbox syncs, ghost preview updates
- [ ] Ring mode: ghost preview shows on correct side when mouse crosses bond
- [ ] Drag-to-merge: drag atom onto another → target survives, bonds transfer (no duplicates)
- [ ] Drag-to-merge: H caps readjusted after merge
- [ ] Drag-to-merge: undo (Ctrl+Z) reverts merge
- [x] Ctrl+LMB drag (Unified/Atom mode): rubber-band bond creation between two atoms
- [x] Ctrl+LMB drag to empty (Unified/Atom mode): create new atom at release position + bond to source
- [x] Ctrl+RMB (Unified/Atom mode): remove atom + bridge 2 heavy neighbors
- [x] Bond mode: LMB insert (no push), Ctrl+LMB insert (push aside), RMB delete, Ctrl+RMB collapse
- [x] Unified mode: click atom → cycle type C→N→O→C (via click-vs-drag detection)
- [x] Unified mode: drag atom → move (via click-vs-drag detection)
- [x] Unified mode: click bond → cycle order
- [x] Unified mode: click hex → add ring
- [x] Unified mode: click empty → add free atom
- [x] Unified mode: Ctrl+drag atom to empty → new atom + bond

---

## PackedMolecule — Dense Snapshot for Clipboard & Undo

### Purpose

`PackedMolecule` is a compact, serializable snapshot of an `AtomicGraph` using dense numpy arrays. Used for:
- **Clipboard** (Ctrl+C/Ctrl+V) — copy selected atoms + internal bonds
- **Undo stack** (Ctrl+Z) — rolling buffer of graph states
- **Binary I/O** — `.npz` save/load

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `etype` | `int32[N]` | Atomic numbers (Z) |
| `apos` | `float32[N,3]` | Positions in Angstrom |
| `bonds` | `int32[B,2]` | 0-based index pairs |
| `npi` | `int8[N]` | Pi-orbital count (-1=H_cap, 0=sp3, 1=sp2, 2=sp) |

Memory: ~21 bytes/atom + 8 bytes/bond. 1000 atoms + 2000 bonds ≈ 37 KB.

### Key methods

- `from_graph(graph, atom_indices=None)` — extract all or subset of atoms (with internal bonds only)
- `to_graph()` — rebuild fresh `AtomicGraph` from packed data (parent reconstructed by distance for H caps)
- `to_xyz_text()` / `to_mol_text()` — text export for clipboard interoperability
- `from_text(text)` — parse `.xyz` or `.mol` text (auto-detect format)
- `save_npz(fname)` / `load_npz(fname)` — binary I/O

### UndoStack

- `UndoStack(maxlen=100)` — rolling `deque` of `PackedMolecule` snapshots
- `push(packed)` — O(1) append, auto-evicts oldest when full
- `pop()` — O(1) pop, returns `PackedMolecule` or `None` if empty
- `enabled` flag — can be disabled to skip pushes during batch operations
- GUI calls `_push_undo()` before all graph mutations (delete, paste, atom type change, pi cycle, handle_click, reset_offsets, adjust_h, recalc_bonds)

### Clipboard flow

1. **Ctrl+C**: `copy_selected_atoms()` → `PackedMolecule.from_graph(graph, selected_indices)` → store in `self.copied_packed` + put MOL/XYZ text on Qt clipboard
2. **Ctrl+V**: `paste_copied_atoms()` → use `self.copied_packed` (or parse Qt clipboard via `from_text()`) → `_append_atom(npi=...)` for each atom → re-create internal bonds → select pasted atoms

---

## Unified Mode — Combined Hex/Atom/Bond Editing (Implemented)

### Motivation

The GUI has 8 edit modes (Unified, Hex1, Hex2, Atom, Bond, Ring, pi, Select). Unified mode combines the most frequent operations into a single context-sensitive mode, reducing mode-switching friction. It is the default mode.

### Architecture

Unified mode is implemented as `UnifiedMode(EditModeHandler)` in `spammm/GUI/EditModeHandlers.py`. It uses `resolve_target(p_world)` to determine what's under the cursor (atom > bond > hex > empty priority) and dispatches accordingly.

**Key design**: Atom interactions (click, drag, Ctrl+drag) are handled by `AtomScene` via signals. Bond/hex/empty interactions are handled by the GUI handler (`on_mouse_press` → `UnifiedMode.on_press`). This split is necessary because the scene manages low-level mouse state (pick, drag, link) while the GUI handler manages high-level topology operations.

### Target Resolution

```
def resolve_target(p_world):
    1. atom = backend.pick_atom(p_world, radius=pick_radius)
       if atom: return ('atom', atom)
    2. bond = backend.pick_bond(p_world, radius=pick_radius)
       if bond: return ('bond', bond)
    3. q, r = backend.snap_to_ring(p_world[0], p_world[1])
       if near hex center: return ('hex', (q, r))
    4. return ('empty', p_world)
```

### Action Dispatch

| Target | LMB | Ctrl+LMB | RMB | Ctrl+RMB |
|--------|-----|----------|-----|----------|
| **Atom** | Click: cycle type (C→N→O→C) / Drag: move | Drag to atom: bond / Drag to empty: new atom+bond | Delete | Delete + bridge |
| **Bond** | Cycle order (1→1.5→2→3→1) | Insert atom (push aside) | Delete | Collapse |
| **Hex** | Add ring | — | Remove ring (preserve shared) | — |
| **Empty** | Add free atom | — | — | — |

### Signal Flow (LMB on atom)

1. `AtomScene._on_mouse_press`: picks atom within `pick_radius`, stores `_press_pos`, sets `ev.handled = True`
2. `SPAMMM_GUI.on_mouse_press`: sees `event.handled = True` → returns (no double-fire)
3. `AtomScene._on_mouse_release`: differentiates click vs drag:
   - **Click** (moved < 3px): `sig_atom_clicked(atom_id)` → `on_atom_clicked` → `UnifiedMode.on_atom_click` → `cycle_atom_type`
   - **Drag** (moved ≥ 3px): `sig_drag_state(0, atom_id, pos)` → `on_drag_state` → sync positions / merge

### Signal Flow (Ctrl+LMB on atom)

1. `AtomScene._on_mouse_press`: `_link_mode and _last_ctrl` → starts link mode, sets `ev.handled = True`
2. `SPAMMM_GUI.on_mouse_press`: sees `event.handled = True` → returns
3. `AtomScene._on_mouse_release`: checks link target:
   - **Another atom**: `sig_link_bond(from_id, to_id)` → `on_link_bond` → `UnifiedMode.on_link` → `create_bond`
   - **Empty space**: `sig_link_to_pos(from_id, x, y)` → `on_link_to_pos` → create new atom at (x,y) + bond to source
   - **Same atom**: `sig_atom_clicked(from_id)` → cycle type

### Signal Flow (LMB on bond/hex/empty)

1. `AtomScene._on_mouse_press`: no atom within `pick_radius` → does NOT set `ev.handled`, returns
2. `SPAMMM_GUI.on_mouse_press`: runs → `UnifiedMode.on_press` → `resolve_target` → bond/hex/empty action

### Click-vs-Drag Detection

`AtomScene._on_mouse_press` stores `_press_pos` (pixel coordinates). On `_on_mouse_release`, if the mouse moved less than 3 pixels, it's a click (`sig_atom_clicked`); otherwise it's a drag (`sig_drag_state`). This is what enables atom type cycling on click while preserving drag-to-move on the same button.

### Mode Handler Registration

`set_edit_mode(mode)` syncs scene flags from the handler's class attributes:
- `scene._link_mode = handler.link_mode` — enables Ctrl+drag bond creation
- `scene.lock_drag = handler.lock_drag` — suppresses atom dragging (Bond mode)
- `scene.selection_mode = handler.selection_mode` — enables RMB rectangle selection

**Important**: `set_edit_mode` must be called during init (after signal connections) to sync the default mode's flags. Without this, `_link_mode` stays `False` even though `UnifiedMode.link_mode = True`.

### Hover Highlighting

`UnifiedMode.on_move(p_world)` calls `resolve_target` and shows:

| Target | Hover visual | Status bar |
|--------|-------------|------------|
| **Atom** | Yellow ring | `"Atom {ename}→{next} (LMB) | Drag: Move | Ctrl+Drag: Bond | RMB: Delete | Ctrl+RMB: Bridge"` |
| **Bond** | Lime line | `"Bond {current}→{next} (LMB) | RMB: Delete | Ctrl+LMB: Insert | Ctrl+RMB: Collapse"` |
| **Hex** | Orange hexagon outline | `"Hex ring (q,r) — LMB: Add | RMB: Remove (preserve shared)"` |
| **Empty** | Red cross cursor | `"Empty — LMB: Add {cur_atom_type} atom | RMB: Nothing"` |
