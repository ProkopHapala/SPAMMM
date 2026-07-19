# Atom Identity & Index Synchronization Design

## Problem

Right-click deletion removes the wrong atom, removes other atoms, or does nothing.
Root cause: **ephemeral array indices** are used as atom identifiers across
three data representations that can fall out of sync.

## Data Representations (SSOT Hierarchy)

```
AtomicGraph (authoritative)
  └─ dict {_id: Atom}   ← stable Python objects, _id never renumbered
  └─ to_arrays() → (atom_list, enames, apos, atypes, bonds, bond_list, rings)
       ↓ _sync_sys()
AtomicSystem / sys (derived, read-only)
  └─ sys.apos, sys.enames, sys.bonds  ← numpy arrays for FF/rendering
       ↓ set_data()
AtomScene / vispy (derived, read-only)
  └─ _pos, _colors, _sizes  ← numpy arrays for GPU rendering
  └─ _atom_ids: np.array   ← NEW: parallel to _pos, maps scene index → Atom._id
```

**Authority chain**: AtomicGraph → sys → scene.
Any mutation goes to AtomicGraph first, then propagates down via `_sync_sys()` + `refresh_view()`.

## Why Deletion Fails (Root Causes)

### 1. No distance threshold in picking
`_pick_idx_from_mouse()` always returns `argmin(d2)` — clicking empty space
still "picks" the nearest atom and deletes it.

### 2. Stale index between scene and graph
- `refresh_view()` builds `sys.apos` via `to_arrays()` → scene copies to `_pos`
- Any graph mutation (e.g. `adjust_h()` adds/removes H atoms) changes `to_arrays()` ordering
- If `_pos` is not refreshed after mutation, scene index `i` ≠ graph index `i`
- `remove_atom_by_index(i)` calls `to_arrays()` **again** → uses `atom_list[i]` which may be a different atom

### 3. `delete_selected_atoms` iterates with stale indices
```python
for idx in selected:          # selected = [3, 7, 12]
    self.backend._rebuild_after_delete([idx])
```
After deleting index 3, the graph changes. Index 7 now points to a different atom.
Each iteration makes the next one wrong.

### 4. `adjust_h()` after deletion changes graph without updating scene
`remove_atom_by_index()` calls `adjust_h()` which adds/removes H atoms.
The scene's `_pos` is stale until next `refresh_view()`.
If another RMB click arrives before `refresh_view()` completes, it uses stale indices.

## Solution: ID-Based Pipeline

### Principle
**Never pass array indices across module boundaries.**
Use `Atom._id` (int) as the stable identifier.
Convert to/from array indices only at the boundary (inside `set_data` / `refresh_view`).

### Changes

#### 1. AtomScene tracks `_atom_ids` (VispyUtils.py)
```python
# In set_data():
self._atom_ids = backend.graph.to_arrays()[0]  # atom_list
self._atom_id_array = np.array([a._id for a in atom_list])
# OR: backend provides it directly via _sync_sys()
```

#### 2. Picking returns Atom._id, not index (VispyUtils.py)
```python
def _pick_id_from_mouse(self, pos, max_dist=0.5):
    """Return (Atom._id, distance) or (None, inf)."""
    idx, dist = self._pick_idx_from_mouse(pos)  # existing, but add threshold
    if idx < 0 or dist > max_dist**2:
        return None
    return int(self._atom_id_array[idx])
```

#### 3. `sig_rmb_remove` emits Atom._id (VispyUtils.py)
```python
sig_rmb_remove = QtCore.pyqtSignal(int)  # now Atom._id, not array index
```

#### 4. Backend removes by Atom._id (KekuleBackend.py)
```python
def remove_atom_by_id(self, atom_id):
    """Remove atom by stable Atom._id. No to_arrays() needed."""
    a = self.graph.atoms.get(atom_id)
    if a is None or not a.alive:
        return
    for h in self.graph.h_children(a):
        self.graph.remove_atom(h)
    self.graph.remove_atom(a)
    self.graph.cleanup_invalid()
    self.graph.sync_neighbor_lists()
    if self.auto_h_cap:
        self.adjust_h()
    self._sync_sys()
```

#### 5. Selection uses Atom._id set (VispyUtils.py + SPAMMM_GUI.py)
- `_selected_ids: set[int]` instead of `_selected_indices: set[int]`
- `get_selected_ids()` / `set_selected_ids()` 
- Drag operations map IDs to current indices via `_atom_id_array`

#### 6. `delete_selected_atoms` collects all IDs first, then deletes (SPAMMM_GUI.py)
```python
def delete_selected_atoms(self):
    ids = list(self.scene.get_selected_ids())
    for aid in ids:
        self.backend.remove_atom_by_id(aid)
    self.scene.clear_selection()
    self.refresh_view()
```

### Performance Notes

- `Atom._id` → array index: O(1) via `np.searchsorted` on sorted `_atom_id_array`
  or O(1) via dict `_id_to_idx` (built once per `set_data`)
- Array index → `Atom._id`: O(1) via `_atom_id_array[idx]`
- No iteration over all atoms needed for any operation
- Dict lookup `graph.atoms[id]` is O(1) — already used by AtomicGraph
- The only overhead is building `_atom_id_array` once per `refresh_view()` — negligible (N atoms)

### What Does NOT Change
- `to_arrays()` still returns arrays for rendering/FF — that's its purpose
- `_pos` still stores float32 positions for GPU — that's its purpose
- AtomicGraph remains the authoritative state
- The fix only changes **what identifier crosses module boundaries**: index → _id
