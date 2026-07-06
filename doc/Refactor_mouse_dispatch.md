# Refactor: Mode-based Callback Dispatch for Mouse Events

## Problem

Current `on_mouse_press`, `handle_click`, and `on_mouse_move` are large if/elif chains
on `self.edit_mode`. Adding a new mode (like Unified) means inserting more branches in
3+ places. Extension modes use a separate `_ext_edit_modes` dict — inconsistent.

## Current Dispatch Flow

```
Vispy event → on_mouse_press(event)
  ├── if 'Select': return (let Vispy handle)
  ├── if 'Unified': 80-line inline dispatch → return
  ├── if Hex/Ring/Bond: pass (no drag)
  ├── else: if selected & LMB: return (let Vispy drag)
  ├── if atom picked & LMB & Atom/pi/Ring: handle inline → return
  └── fallback: handle_click(pos, action='add'/'remove', ctrl)
      └── another if/elif on edit_mode for target resolution + action
```

```
Vispy event → on_mouse_move(event)
  └── if/elif on edit_mode for hover highlighting
      ├── Unified: resolve_unified_target → 4 sub-branches
      ├── Atom/pi/Select: highlight atom
      ├── Bond/Ring: highlight bond + ring preview
      └── Hex1/Hex2: highlight ring + hex grid
```

Signal callbacks (from AtomScene):
- `sig_rmb_remove` → `on_atom_remove(atom_id)` — mode-guarded
- `sig_link_bond` → `on_link_bond(from_id, to_id)` — not mode-guarded
- `sig_atom_picked` → `on_atom_clicked(atom_id)` — mode-guarded

## Proposed Design: ModeHandler Registry

### Data Structure

```python
# Each mode registers a ModeHandler with callbacks for each event type:
ModeHandler = namedtuple('ModeHandler', [
    'on_press',    # (event, p_world, ctrl) -> handled: bool
    'on_move',     # (p_world) -> None  (hover update)
    'on_release',  # (event, p_world, ctrl) -> None  (optional)
    'on_rmb_atom', # (atom_id, ctrl) -> None  (sig_rmb_remove callback, optional)
    'on_link',     # (from_id, to_id) -> None  (sig_link_bond callback, optional)
    'on_atom_click', # (atom_id) -> None  (sig_atom_picked without drag, optional)
    'config',      # dict: {lock_drag, link_mode, selection_mode, ring_size_visible}
    'status_msg',  # str: initial status bar message
])
```

### Registry

```python
self.mode_handlers = {}  # mode_name -> ModeHandler

# Built-in modes registered in _init_mode_handlers():
self._init_mode_handlers()

# Extension modes: extensions can register via:
#   window.register_mode_handler('MyMode', ModeHandler(...))
# This replaces _ext_edit_modes dict
```

### Dispatch (simplified)

```python
def on_mouse_press(self, event):
    h = self.mode_handlers.get(self.edit_mode)
    if h is None or h.on_press is None: return
    ctrl = 'Control' in event.modifiers if isinstance(event.modifiers, (tuple, list)) else False
    r0, rd = self.scene._ray_from_mouse(event.pos)
    p_world = self.scene._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
    if p_world is None: return
    h.on_press(event, p_world, ctrl)

def on_mouse_move(self, event):
    r0, rd = self.scene._ray_from_mouse(event.pos)
    p_world = self.scene._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
    if p_world is None: return
    self.cursor_markers.set_data(...)
    self._clear_hover()
    h = self.mode_handlers.get(self.edit_mode)
    if h and h.on_move:
        h.on_move(p_world)

def on_atom_remove(self, atom_id):
    h = self.mode_handlers.get(self.edit_mode)
    if h and h.on_rmb_atom:
        h.on_rmb_atom(atom_id, self.scene._last_ctrl)

def on_link_bond(self, from_id, to_id):
    h = self.mode_handlers.get(self.edit_mode)
    if h and h.on_link:
        h.on_link(from_id, to_id)

def on_atom_clicked(self, atom_id):
    h = self.mode_handlers.get(self.edit_mode)
    if h and h.on_atom_click:
        h.on_atom_click(atom_id)
```

### set_edit_mode (simplified)

```python
def set_edit_mode(self, mode):
    self.edit_mode = mode
    h = self.mode_handlers.get(mode)
    if h is None: return  # unknown mode
    # Apply config
    cfg = h.config
    self.scene.set_selection_mode(cfg.get('selection_mode', False))
    self.scene.lock_drag = cfg.get('lock_drag', False)
    self.scene._link_mode = cfg.get('link_mode', False)
    self.ring_size_label.setVisible(cfg.get('ring_size_visible', False))
    self.ring_size_spinbox.setVisible(cfg.get('ring_size_visible', False))
    self.scene.ring_preview_line.visible = False
    if h.status_msg:
        self.statusBar().showMessage(h.status_msg)
    if mode == 'Hex1': self.backend.hex_mode = 'Hex1'
    elif mode == 'Hex2': self.backend.hex_mode = 'Hex2'
    if mode == 'pi': self.set_label_mode('Pi Orbitals')
```

## Mode Handler Implementations

### Unified
- `on_press`: resolve_unified_target → dispatch LMB/RMB per target_type
- `on_move`: resolve_unified_target → hover highlight + status bar
- `on_rmb_atom`: remove_atom_by_id or remove_atom_with_bridge (ctrl)
- `on_link`: create bond (Ctrl+drag)
- `on_atom_click`: cycle_atom_type (click without drag)
- `config`: {lock_drag: False, link_mode: True, selection_mode: False}

### Atom
- `on_press`: if atom picked & LMB → set_atom_type_by_id; else handle_click fallback
- `on_move`: highlight hovered atom
- `on_rmb_atom`: remove_atom_by_id or remove_atom_with_bridge
- `on_link`: create bond
- `on_atom_click`: set_atom_type_by_id
- `config`: {lock_drag: False, link_mode: True, selection_mode: False}

### pi
- `on_press`: if atom picked & LMB → cycle npi; else handle_click fallback
- `on_move`: highlight hovered atom
- `on_rmb_atom`: remove_atom_by_id or remove_atom_with_bridge
- `config`: {lock_drag: False, link_mode: False, selection_mode: False}

### Bond
- `on_press`: LMB → insert_atom_into_bond; RMB → delete_bond or collapse_bond (ctrl)
- `on_move`: highlight hovered bond
- `config`: {lock_drag: True, link_mode: False, selection_mode: False}

### Ring
- `on_press`: LMB → add_adjacent_ring on picked bond; RMB → delete_bond
- `on_move`: highlight hovered bond + ring preview polygon
- `config`: {lock_drag: True, link_mode: False, selection_mode: False, ring_size_visible: True}

### Hex1 / Hex2
- `on_press`: LMB → add_ring; RMB → remove_ring
- `on_move`: highlight hovered ring + hex grid nodes
- `config`: {lock_drag: True, link_mode: False, selection_mode: False}

### Select
- `on_press`: None (let Vispy handle everything)
- `on_move`: highlight hovered atom
- `on_rmb_atom`: remove_atom_by_id
- `config`: {lock_drag: False, link_mode: False, selection_mode: True}

## Implementation Steps

1. **Define ModeHandler namedtuple** at module level
2. **Add `self.mode_handlers = {}` dict** in `__init__`
3. **Write `_init_mode_handlers()`** — registers all 8 built-in modes
   - Extract existing inline logic into handler functions
   - Each handler is a method like `_unified_on_press(self, event, p_world, ctrl)`
4. **Simplify `set_edit_mode`** — use config dict from handler
5. **Simplify `on_mouse_press`** — single dispatch line
6. **Simplify `on_mouse_move`** — single dispatch + common preamble
7. **Simplify `on_atom_remove`, `on_link_bond`, `on_atom_clicked`** — dispatch via handler
8. **Update extension registration** — `register_mode_handler(name, handler)` method
   replaces `_ext_edit_modes` dict
9. **Delete `handle_click`** — its logic is now distributed into mode handlers
10. **Run tests** — `pytest tests/topology/test_unified_mode.py tests/topology/test_editing_ops.py`

## Key Design Decisions

- **Handler functions are methods** (not closures) — can access `self.backend`, `self.scene`, etc.
- **Common preamble in dispatcher** — ray casting, ctrl detection done once
- **`_clear_hover()` helper** — clears all hover visuals, called before `on_move`
- **`handle_click` eliminated** — was a second-level dispatcher, now redundant
- **Extensions use same registry** — no separate `_ext_edit_modes` path
- **Backward compat**: `on_atom_remove` etc. still exist as signal targets, just dispatch internally

## Risk Assessment

- **Medium risk**: touches all mouse interaction code
- **Mitigation**: incremental extraction — write handlers first, then switch dispatch
- **Test coverage**: 24 unified mode tests + 42 editing ops tests
- **No behavior change** — pure refactor, same logic in different structure
