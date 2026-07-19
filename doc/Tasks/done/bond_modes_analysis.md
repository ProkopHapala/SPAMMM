# Bond Editing Modes — Analysis & Design Notes (v3)

## Current State

### Existing "Bond" mode
- **LMB**: `insert_atom_into_bond(bond, ename)` — splits bond A-B into A-C-B, **pushes A and B aside** to maintain bond lengths
- **RMB**: `collapse_bond(bond, mouse_pos)` — merges two atoms into one (destructive)
- **Hover**: highlights bond in lime green (`hover_bond_line`)
- **Picking**: `backend.pick_bond(pos, radius)` — finds bond whose **center** is within `radius` of mouse
- **Drag**: locked (`scene.lock_drag = True`) — no atom dragging

### Existing "Atom" mode
- **LMB click on atom**: Changes atom type (to `cur_atom_type`) — handled in `on_mouse_press`, returns early
- **LMB click on empty space**: Adds new atom at click position — falls through to `handle_click`
- **LMB drag on atom**: Moves atom — Vispy handles via `_pick_active` + `_on_mouse_move` + `sig_drag_state`
  - `_on_mouse_press` picks atom, sets `_pick_active=True`, emits `sig_atom_picked` + `sig_drag_state(1,...)`
  - `_on_mouse_move` updates `_pos[i]` to mouse world position
  - `_on_mouse_release` emits `sig_drag_state(0,...)` → GUI syncs positions back to graph
  - **Conflict**: GUI's `on_mouse_press` intercepts LMB on atom (changes type, returns) — so drag never starts in Atom mode currently. Drag only works in non-Atom modes where `on_mouse_press` doesn't intercept.
- **RMB on atom**: Deletes atom (via `sig_rmb_remove` → `on_atom_remove` → `remove_atom_by_id`)

### Existing "Atom" RMB removal
- `on_atom_remove(atom_id)` → `backend.remove_atom_by_id(atom_id)`
- `remove_atom_by_id`: soft-deletes atom + H children, calls `adjust_h()` + `_sync_sys()`
- **No neighbor handling** — bonds are just marked dead, neighbors are not connected to each other

### What's missing
1. **Create bond between two existing atoms** — no way to connect atoms that aren't bonded
2. **Delete a bond** without collapsing atoms — no way to remove just the bond
3. **Insert atom into bond without pushing atoms aside** — current insert always moves A and B
4. **Remove atom with neighbor bridging** — when removing an atom bonded to 2 heavy neighbors, option to create bond between those neighbors (bridge/collapse)

### Existing backend support
- `AtomicGraph.add_bond(a, b)` — idempotent, returns existing bond if already bonded
- `AtomicGraph.remove_bond(bond)` — soft-deletes bond only, atoms stay alive
- `AtomicGraph.pick_bond(pos, radius)` — picks by bond center distance
- `KekuleBackend.insert_atom_into_bond(bond, ename)` — inserts + pushes atoms aside
- `KekuleBackend.collapse_bond(bond, mouse_pos)` — merges atoms (transfers bonds, shifts neighbors)
- `KekuleBackend.remove_atom_by_id(atom_id)` — soft-delete atom + H children, no neighbor bridging

**NOT implemented**: `remove_atom_with_bridge(atom_id)` — remove atom and connect its 2 heavy neighbors with a new bond. This is the inverse of `insert_atom_into_bond`. Needs to be added.

---

## Revised Design (v3)

### Key decisions
1. **No new mode** — bond creation is Ctrl+drag in Atom mode, bond delete/collapse is Ctrl modifier in Bond mode
2. **Atom mode**: LMB drag = move atom (keep existing). **Ctrl+LMB drag** = create bond (rubber-band line)
3. **Atom mode RMB**: No Ctrl = just remove atom. **Ctrl+RMB** = remove atom + bridge 2 heavy neighbors
4. **Bond mode**: Ctrl controls atom adjustment (push aside / collapse)

---

### Mode: **Bond** (Ctrl controls atom adjustment)

| Action | Without Ctrl (minimal) | With Ctrl (adjust atoms) |
|--------|------------------------|--------------------------|
| **LMB** on bond | Insert atom at bond center, **don't move** A and B | Insert atom at bond center, **push A and B aside** (current behavior) |
| **RMB** on bond | **Delete bond** — remove bond only, keep both atoms | **Collapse bond** — merge two atoms into one (current behavior) |

**Rationale**: Ctrl = "also adjust surrounding atoms". Without Ctrl, operations are minimal/non-destructive.

- Hover: bond highlighted in lime green (same as now)
- `lock_drag = True` (no atom dragging)
- Status bar: `"LMB: Insert atom (Ctrl: push aside) | RMB: Delete bond (Ctrl: collapse)"`

**Backend changes needed**:
- `insert_atom_into_bond(bond, ename, push_aside=True)` — add `push_aside` param. When `False`, skip position updates of A and B.
- `delete_bond(bond)` — convenience: `graph.remove_bond(bond)` + `adjust_h()` + `_sync_sys()`
- `collapse_bond` already exists — called on Ctrl+RMB

---

### Mode: **Atom** (Ctrl+drag creates bond, Ctrl+RMB bridges neighbors)

#### LMB — drag behavior (keep existing move, add Ctrl+drag for bond)

| Action | Without Ctrl | With Ctrl |
|--------|--------------|-----------|
| **LMB click on atom** (no drag) | Change atom type (same as now) | Change atom type (same) |
| **LMB click on empty space** | Add new atom (same as now) | Add new atom (same) |
| **LMB drag from atom** | **Move atom** (existing Vispy drag) | **Create bond** — rubber-band line to target |
| **LMB drag from empty space** | Nothing | Nothing |

**Ctrl+LMB drag to create bond**:
- Press Ctrl+LMB on atom A → start rubber-band line from A to mouse (don't move A)
- Move → update rubber-band line, highlight target atom in green if hovering over one
- Release on atom B (≠ A) → `graph.add_bond(A, B)` + cleanup
- Release on empty space → cancel (no bond)
- Release on same atom A → treat as click (change type)

**How to implement**: In Vispy `_on_mouse_press`, check Ctrl modifier. If Ctrl and atom picked → enter `_link_active` state instead of `_pick_active` (don't move atom). If no Ctrl → normal drag behavior (move atom).

#### RMB — atom removal with optional neighbor bridging

| Action | Without Ctrl | With Ctrl |
|--------|--------------|-----------|
| **RMB on atom** | Remove atom + H children, bonds just die | Remove atom + H children + **bridge 2 heavy neighbors** with new bond |

**Ctrl+RMB "bridge collapse"**:
- Find heavy neighbors of atom (exclude H caps: `n.npi != -1`)
- If exactly 2 heavy neighbors (A, B) and they're not already bonded → create bond A-B, remove atom
- If 1 or 0 heavy neighbors → just remove atom (nothing to bridge)
- If 3+ heavy neighbors → just remove atom (can't bridge unambiguously)
- This is the **inverse of insert_atom_into_bond**: A-C-B becomes A-B

**Backend function needed**: `remove_atom_with_bridge(atom_id)`:
```python
def remove_atom_with_bridge(self, atom_id):
    a = self.graph.atoms.get(atom_id)
    if a is None or not a.alive: return
    # Find heavy neighbors
    heavy_neighs = [n for n in a.neighbors if n.alive and n.npi != -1]
    # Remove H children first
    for h in self.graph.h_children(a):
        self.graph.remove_atom(h)
    # Bridge if exactly 2 heavy neighbors
    if len(heavy_neighs) == 2:
        n1, n2 = heavy_neighs
        # Check they're not already bonded
        already_bonded = any(b.other(n1) is n2 for b in n1.bonds if b.alive)
        if not already_bonded:
            self.graph.add_bond(n1, n2)
    self.graph.remove_atom(a)
    if self.auto_h_cap:
        self.adjust_h()
    self._sync_sys()
```

**GUI changes**: `on_atom_remove` needs to check Ctrl modifier. But `on_atom_remove` is called from `sig_rmb_remove` signal which only passes `atom_id` — no modifier info. Options:
1. **Add modifier to signal**: Change `sig_rmb_remove` to `sig_rmb_remove(int, bool)` — `(atom_id, ctrl_pressed)`. Cleanest.
2. **Store modifier in scene**: `self._last_ctrl = bool` set in `_on_mouse_press`, read in `on_atom_remove`. Simpler, slightly hacky.
3. **New signal**: `sig_rmb_remove_bridge(int)` for Ctrl+RMB. Most explicit.

**Recommended**: Option 2 (store modifier) — minimal signal changes, easy to extend.

---

## Updated Mode List (v3)

| Mode | LMB | Ctrl+LMB | RMB | Ctrl+RMB | Description |
|------|-----|----------|-----|----------|-------------|
| Hex1 | Add ring | — | Remove ring | — | Paint hexagons (force) |
| Hex2 | Add ring | — | Remove ring (preserve shared) | — | Toggle hexagons |
| **Atom** | Click: change type / Drag: move atom | Drag: create bond | Delete atom | Delete atom + bridge neighbors | Free atom + bond creation |
| **Bond** | Insert atom (no push) | Insert atom (push aside) | Delete bond | Collapse bond | Edit existing bonds |
| pi | Cycle npi | — | Delete atom | — | Hybridization editing |
| Select | Drag selected | — | Rectangle select | — | Multi-atom operations |

**6 modes total** — same as before. Ctrl modifier adds secondary behavior in Atom and Bond modes.

---

## Edge Cases

1. **Bond already exists**: `add_bond` is idempotent — returns existing bond. No error, no duplicate.
2. **Self-bond (A→A)**: Ctrl+drag from atom, release on same atom → treated as click (change type). Check `a_id != b_id` before `add_bond`.
3. **Bond delete with H caps**: Call `adjust_h()` if `auto_h_cap` is on.
4. **Bridge collapse with 0 or 1 heavy neighbors**: Just remove atom, no bridge.
5. **Bridge collapse with 3+ heavy neighbors**: Just remove atom, no bridge (ambiguous).
6. **Bridge collapse: neighbors already bonded**: Remove atom, don't create duplicate bond.
7. **Ctrl+LMB drag from empty space**: Nothing happens (no atom to start from).
8. **Ctrl+LMB drag, release on empty space**: Cancel — no bond created.
9. **Insert without push_aside**: A and B stay in place. New C at center. Bond lengths A-C, B-C = half of A-B. Fine for rough editing.
10. **Atom drag in Atom mode currently broken**: `on_mouse_press` intercepts LMB on atom and returns early, preventing Vispy drag. Need to fix: only intercept for type change if no drag occurs. Or: let Vispy handle press, intercept on release (click vs drag distinction).

---

## Implementation Plan

### Step 1: Backend — new functions

**`KekuleBackend.py`** (~25 lines):
- `insert_atom_into_bond(bond, ename, push_aside=True)` — add `push_aside` param
- `delete_bond(bond)` — `graph.remove_bond(bond)` + `adjust_h()` + `_sync_sys()`
- `remove_atom_with_bridge(atom_id)` — remove atom + bridge 2 heavy neighbors if applicable

### Step 2: Bond mode — Ctrl modifier

**`SPAMMM_GUI.py`** (~15 lines):
- In `handle_click`, Bond mode: read Ctrl from `event.modifiers`
  - LMB: `insert_atom_into_bond(bond, ename, push_aside=ctrl)`
  - RMB: if ctrl → `collapse_bond(bond, pos)`, else → `backend.delete_bond(bond)`
- Update status bar in `set_edit_mode`

### Step 3: Atom mode — Ctrl+drag to create bond

**`VispyUtils.py`** (~40 lines):
- Add `_link_active: bool`, `_link_from_id: int`, `_link_line` visual
- New signal: `sig_link_bond(int, int)` — `(from_id, to_id)`
- In `_on_mouse_press`: if Ctrl held and atom picked → `_link_active=True`, show rubber-band line. **Don't set `_pick_active`** (no atom moving).
- In `_on_mouse_move` (while `_link_active`): update rubber-band line, highlight target atom green if near
- In `_on_mouse_release` (while `_link_active`): pick atom at release. If different → emit `sig_link_bond`. Reset.

**`SPAMMM_GUI.py`** (~15 lines):
- Connect `sig_link_bond` → `on_link_bond(a_id, b_id)`: `graph.add_bond(a, b)` + `sync_neighbor_lists()` + `adjust_h()` + `_sync_sys()` + `refresh_view()`

### Step 4: Atom mode — Ctrl+RMB bridge collapse

**`VispyUtils.py`** (~5 lines):
- Store `_last_ctrl` in `_on_mouse_press`: `self._last_ctrl = 'Control' in ev.modifiers`
- Pass to `sig_rmb_remove` or store for GUI to read

**`SPAMMM_GUI.py`** (~10 lines):
- In `on_atom_remove`: check `self.scene._last_ctrl`
  - If ctrl → `backend.remove_atom_with_bridge(atom_id)`
  - If no ctrl → `backend.remove_atom_by_id(atom_id)` (current behavior)

### Step 5: Fix Atom mode drag vs click conflict

**Current bug**: `on_mouse_press` in GUI intercepts LMB on atom (changes type, returns) — prevents Vispy drag from working in Atom mode.

**Fix**: Don't intercept in `on_mouse_press` for Atom mode. Let Vispy handle the press (start drag). On release, if atom didn't move (click) → change type. If atom moved → it was a drag (position already synced via `on_drag_state`).

This means moving the type-change logic from `on_mouse_press` to `on_drag_state(state=0)` or a new `sig_atom_clicked` signal.

---

## Files to modify (v3)

| File | Changes | Est. lines |
|------|---------|------------|
| `KekuleBackend.py` | `push_aside` param, `delete_bond`, `remove_atom_with_bridge` | ~25 |
| `SPAMMM_GUI.py` | Ctrl in Bond mode, `on_link_bond`, Ctrl in `on_atom_remove`, drag/click fix | ~40 |
| `VispyUtils.py` | `_link_active` state, rubber-band line, `sig_link_bond`, `_last_ctrl`, click vs drag | ~45 |
| `AtomicGraph.py` | No changes needed | 0 |

**Total**: ~110 lines. No new mode. No new file.
