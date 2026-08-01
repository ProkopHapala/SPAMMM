# Task: GUI Help Panel + Centralized Shortcut Registry

**Status:** Spec rewritten (2026-08-01) — decentralized extension-driven architecture, ready for implementation  
**Priority:** P2 (GUI UX — human ToDo "GUI Laptop" + "key-shortcuts")  
**Related:** `doc/ToDo/ToDo.human.md` (GUI Laptop, longer/key-shortcuts), `user_guide/GUI_CHEATSHEET.md`, `spammm/GUI/SPAMMM_GUI.py` (`on_key_press`), `spammm/GUI/VispyUtils.py` (`_on_key_press`), `spammm/GUI/BaseGUI.py` (`button`), `spammm/GUI/ExtensionManager.py` (`UIComponents.help_text`)

---

## 1. Goal

Two related improvements to SPAMMM_GUI user experience:

1. **In-GUI help/cheatsheet panel** — a collapsible section + Help menu showing all keyboard + mouse shortcuts, auto-collected from the registry.
2. **Centralized shortcut registry** — a generic **mechanism** (not a hardcoded list of actions) that extensions use to register their own shortcuts. The registry detects conflicts, auto-syncs button labels with Unicode keystrokes, and auto-generates help/cheatsheet content.

### Key architectural principle (CORRECTED)

The registry defines the **system/protocol/sync algorithm** — NOT specific actions. Each extension (and the main GUI) registers its own shortcuts via the registry API when it creates buttons. The registry:

- **Does NOT know a priori** what actions exist
- **Does** provide: conflict detection, Unicode encoding, label auto-sync, help generation, dispatch
- Is **extensible** — new extensions register their shortcuts, no central edits needed

---

## 2. Current shortcut inventory (audit — for migration reference only)

### Global (`SPAMMM_GUI.on_key_press`)

| Key | Modifiers | Action | Context |
|-----|-----------|--------|---------|
| `Enter` | — | Toggle 2D/3D view | Always |
| `Space` | — | Toggle FF run/stop | Always |
| `Z` | Ctrl | Undo | Always |
| `V` | Ctrl | Paste | Always |
| `C` | Ctrl | Copy selected | Select mode |
| `Delete` | — | Delete selected | Select mode |
| `+` / `=` / `KP_ADD` | — | Increase ring size | Ring mode |
| `-` / `_` / `KP_SUBTRACT` | — | Decrease ring size | Ring mode |

### Camera (`VispyUtils._on_key_press`)

| Key | Modifiers | Action | Context |
|-----|-----------|--------|---------|
| `Arrows` | — | Pan (2D) / Rotate (3D) | Always |
| `Arrows` | Shift | Pan (both modes) | Always |
| `5` / `KP_5` | — | Top view preset | Always |
| `0` / `2` / `4` / `6` / `8` | — | Bottom/Front/Left/Right/Back | 3D only |
| `Scroll` | — | Zoom | Always |

### Mouse (hardwired — NOT in registry)

| Button | Modifiers | Action | Context |
|--------|-----------|--------|---------|
| LMB | — | Add/toggle/drag (mode-dependent) | All modes |
| RMB | — | Remove/delete (mode-dependent) | All modes |
| LMB | Ctrl | Drag atom→atom: create bond | Atom/Unified |
| LMB | Ctrl | Drag to empty: create atom+bond | Unified |
| RMB | — | Drag empty: rotate view | 3D only |
| Middle | — | Toggle H at position | Atom/Pi/Unified |

---

## 3. Design: Decentralized Shortcut Registry

### 3.1 Core principle: registry = mechanism, not content

`ShortcutRegistry.py` defines:
- `ShortcutSpec` — data class for one shortcut (keys, modifiers, description, group, context_fn, callback)
- `ShortcutRegistry` — singleton store with: register (conflict check), encode (Unicode), format_label, dispatch, help generation, `--export` CLI
- Unicode modifier encoding (`⌃⇧⌥`)

`ShortcutRegistry.py` does **NOT** define:
- Which actions exist
- Which keys map to which actions
- Any specific shortcut registration

Each **extension** and the **main GUI** register their own shortcuts at their `build_ui()` / `initUI()` time.

### 3.2 Modifier encoding

| Modifier | Symbol | Unicode |
|----------|--------|---------|
| Ctrl | `⌃` | U+2303 |
| Shift | `⇧` | U+21E7 |
| Alt | `⌥` | U+2325 |

Special keys: `⏎` Enter (U+23CE), `␣` Space (U+2423), `Del` Delete, `↑↓←→` Arrows.

**Format:** `ButtonName [⌃⇧K]` — brackets delimit, modifiers prefix, key suffix.

ASCII fallback (if font fails): `^` Ctrl, `S+` Shift, `A+` Alt.

### 3.3 The registration + sync protocol

**Step 1 — Extension creates a button with a shortcut:**

```python
# In any extension's build_ui() or SPAMMM_GUI's initUI():
self.button("Undo", self.undo, layout=row, shortcut=('Z', ('Control',), "Undo last edit", "Global"))
```

The `shortcut=` argument is a tuple: `(keys, modifiers, description, group)`
- `keys`: vispy key name or list of equivalents, e.g. `'Z'`, `['Enter', 'Return']`, `['+', 'KP_ADD', '=']`
- `modifiers`: tuple of `'Control'`/`'Shift'`/`'Alt'` (empty `()` = no modifiers)
- `description`: human-readable, e.g. `"Undo last edit"` — used in help/cheatsheet
- `group`: `"Global"`, `"Camera"`, `"Ring"`, `"Select"`, or extension-defined group name

**Step 2 — `BaseGUI.button()` calls `ShortcutRegistry.register_button()`:**

```python
def button(self, text, callback=None, tooltip=None, layout=None, shortcut=None):
    btn = QtWidgets.QPushButton(text)
    if shortcut is not None:
        from spammm.GUI.ShortcutRegistry import ShortcutRegistry
        ShortcutRegistry.register_button(btn, text, shortcut)
    ...
```

**Step 3 — `ShortcutRegistry.register_button()` does the sync:**

1. Unpack `shortcut` tuple → `(keys, modifiers, description, group)`
2. **Conflict check**: is `(key, modifiers)` already registered? If yes → fail loud (print warning + raise, per Fail Fast principle). This prevents two extensions grabbing the same key.
3. Create `ShortcutSpec` and store it
4. **Auto-format label**: `btn.setText(format_label(text, spec))` → `"Undo [⌃Z]"`
5. Store `widget → spec` mapping (so label can be re-synced if needed)

**Step 4 — Key dispatch (optional, per-extension):**

Extensions that want registry-driven dispatch call `ShortcutRegistry.dispatch(event, window)` in their key handler. The registry checks `context_fn(window)` and calls the callback. Extensions that prefer their own dispatch (like VispyUtils camera, which has direction-dependent logic) can register shortcuts as documentation-only (`callback=None`) — they appear in help/cheatsheet but dispatch stays in the extension.

### 3.4 API: ShortcutRegistry

```python
class ShortcutSpec:
    keys: list[str]           # vispy key names, e.g. ['Z'] or ['Enter', 'Return']
    modifiers: tuple[str]     # ('Control',) or ()
    description: str          # "Undo last edit"
    group: str                # "Global", "Camera", or extension-defined
    context_fn: callable|None # (window) -> bool; None = always active
    callback: callable|None   # (window) -> None; None = documentation-only
    widget: QWidget|None      # the button this shortcut is bound to (for label sync)

class ShortcutRegistry:
    @classmethod
    def register(cls, keys, modifiers=(), description="", group="Global",
                 context_fn=None, callback=None, widget=None) -> ShortcutSpec
        # Conflict check: if (key, modifiers) already registered → raise ShortcutConflictError
        # Create + store ShortcutSpec, return it

    @classmethod
    def register_button(cls, btn, text, shortcut_tuple) -> ShortcutSpec
        # Unpack (keys, modifiers, description, group)
        # Call register(), auto-format btn label

    @classmethod
    def encode(cls, keys, modifiers) -> str
        # "⌃Z", "⏎", "⇧↑"

    @classmethod
    def format_label(cls, text, spec) -> str
        # "Undo [⌃Z]"

    @classmethod
    def dispatch(cls, event, window) -> bool
        # Try to match event.key + event.modifiers against registered shortcuts
        # Check context_fn(window), call callback(window) if match
        # Skip documentation-only (callback=None) entries
        # Return True if handled

    @classmethod
    def all(cls) -> list[ShortcutSpec]
        # All registered shortcuts (for help generation)

    @classmethod
    def by_group(cls) -> dict[str, list[ShortcutSpec]]
        # Grouped for help panel sections

    @classmethod
    def help_markdown(cls) -> str
        # Auto-generate keyboard table for cheatsheet

    @classmethod
    def help_html(cls) -> str
        # Auto-generate HTML for in-GUI help panel

    @classmethod
    def export_cheatsheet_section(cls) -> str
        # Markdown table for <!-- AUTOGEN:keyboard --> section

class ShortcutConflictError(RuntimeError):
    pass
```

### 3.5 Who registers what (decentralized)

| Registrant | Where | What |
|------------|-------|------|
| `SPAMMM_GUI.initUI()` | Main GUI | Enter (2D/3D), Space (FF run), Ctrl+Z (undo), Ctrl+V (paste), Ctrl+C (copy), Delete (delete sel), Numpad +/- (ring size) |
| `VispyUtils.AtomScene.__init__()` | Camera | Arrows (pan/rotate), Shift+Arrows (pan), digit presets (5/0/2/4/6/8), Scroll (zoom) — registered as documentation-only (callback=None) since dispatch is direction-dependent |
| Each extension's `build_ui()` | Extension | Whatever shortcuts that extension defines for its buttons |

**No central list.** The registry is empty at import time. It fills up as extensions call `register()` / `register_button()` during UI construction.

### 3.6 Conflict detection

When `register()` is called with `(keys, modifiers)` that already exist:
- Raise `ShortcutConflictError(f"Shortcut {encode(keys, modifiers)} already registered: {existing.description}")`
- This is a **fail-loud** check (per AGENTS.md Fail Fast principle)
- Prevents two extensions silently grabbing the same key

### 3.7 Help panel + cheatsheet auto-generation

The help panel and `--export` CLI iterate `ShortcutRegistry.all()` / `by_group()` — they automatically include whatever extensions have registered. No central maintenance.

Mouse actions stay hand-maintained (not in registry — see §7 decision 4).

---

## 4. Help panel design

### Side panel section
- `create_help_section()` in `SPAMMM_GUI.py` — CollapsibleSection "Help / Cheatsheet"
- QTextEdit (read-only) with `ShortcutRegistry.help_html()` (auto-collected from all registered shortcuts)
- Sections auto-generated by group: Global, Camera, Ring, Select, + any extension-defined groups
- Mouse actions: static HTML table (hand-maintained string constant)
- Extension `help_text` (from `UIComponents`): already wired via "?" buttons

### Help menu
- `create_menus()` adds "Help" menu with "Show Cheatsheet" → dialog with full help text
- Eventually the side-panel section may be removed in favor of menu dialog only

---

## 5. Doc reorganization: `doc/` → `user_guide/`

### Move to `user_guide/` (user-facing):

| Current location | New location | Reason |
|------------------|--------------|--------|
| `doc/KekuleSolverVisualization.md` | `user_guide/KekuleSolver_GUI.md` | Describes how to use the Kekule solver visualization in GUI — user-facing |

### Keep in `doc/` (developer-facing):

| File | Reason |
|------|--------|
| `doc/GUI.desing.md` | Architecture + design principles for developers |
| `doc/GUI_FF_Relaxation.md` | Implementation report (OpenCL kernel internals) |
| `doc/GUI_topology_edit.desing.md` | System design (AtomicGraph, soft-delete, ID mapping) |
| `doc/Tasks/GUI_Editor_3D_ViewMode.md` | Task doc with implementation decisions |
| `doc/MolecularBrowser_Report.md` | Implementation report |
| `doc/MolecularBrowser_Vispy.design.md` | Design document |
| `doc/Refactor_mouse_dispatch.md` | Refactor design |
| `doc/HowTo/VisualDebugging.md` | Testing strategy for agents |

### Update `user_guide/README.md`:
Add `GUI_CHEATSHEET.md` + `KekuleSolver_GUI.md` to index table (done in prior session).

---

## 6. Implementation plan (phased)

### Phase 1: ShortcutRegistry mechanism (generic, no specific actions)

1. Create `spammm/GUI/ShortcutRegistry.py`:
   - `ShortcutSpec` data class
   - `ShortcutRegistry` singleton: `register()`, `register_button()`, `encode()`, `format_label()`, `dispatch()`, `help_markdown()`, `help_html()`, `export_cheatsheet_section()`
   - `ShortcutConflictError` exception
   - Unicode modifier encoding
   - `--export` CLI
   - **NO default registrations** — empty at import time
2. Modify `BaseGUI.button()` — add `shortcut=` param, call `register_button()` (already done in prior session, API matches)
3. L0 test: verify registry mechanism (register, conflict detection, encoding, dispatch, export)

### Phase 2: Extensions register their shortcuts (decentralized)

1. `SPAMMM_GUI.initUI()` — register global shortcuts (Enter, Space, Ctrl+Z, Ctrl+V, Ctrl+C, Delete, Numpad +/-) via `button(shortcut=...)` or direct `register()` calls
2. `SPAMMM_GUI.on_key_press()` — dispatch via `ShortcutRegistry.dispatch(event, self)` first, fall through to mode-specific handling
3. `VispyUtils.AtomScene.__init__()` — register camera shortcuts as documentation-only (callback=None)
4. Each extension's `build_ui()` — add `shortcut=` to button calls where applicable
5. L0 test: verify all expected shortcuts are registered after GUI construction

### Phase 3: Help panel + Help menu

1. Add `create_help_section()` to `SPAMMM_GUI.py` — CollapsibleSection with auto-generated help
2. Add "Help" menu to `create_menus()` — "Show Cheatsheet" dialog
3. Wire extension `help_text` into the help panel
4. L0 test: verify help section constructs, contains expected shortcuts

### Phase 4: Doc sync + reorganization

1. Move `doc/KekuleSolverVisualization.md` → `user_guide/KekuleSolver_GUI.md`
2. Run `python -m spammm.GUI.ShortcutRegistry --export` to regenerate cheatsheet keyboard section
3. Verify `user_guide/GUI_CHEATSHEET.md` autogen markers work

---

## 7. USER decisions (resolved 2026-08-01)

1. **Modifier symbols:** ✅ Unicode — `⌃⇧⌥` (with ASCII fallback `^S+A+`).
2. **Help panel location:** ✅ Both — CollapsibleSection in side panel + Help menu dialog. Side-panel section may be removed eventually.
3. **Cheatsheet generation:** ✅ Auto-generate from registry via `--export` script. `GUI_CHEATSHEET.md` gets `<!-- AUTOGEN:keyboard START/END -->` markers.
4. **Mouse actions in registry:** ✅ Keep hardwired — mouse actions stay in mode handlers, NOT in registry. Documented in cheatsheet (hand-maintained table) + help panel (static section) + `doc/GUI.desing.md`.
5. **Architecture (CORRECTED 2026-08-01):** ✅ Registry = generic mechanism only. Extensions register their own shortcuts. No central hardcoded action list. Conflict detection via fail-loud `ShortcutConflictError`.

---

## 8. Files to create/modify

| File | Action |
|------|--------|
| `spammm/GUI/ShortcutRegistry.py` | **Create** — generic registry mechanism (ShortcutSpec, register, conflict check, encode, dispatch, help gen, --export). NO specific actions. |
| `spammm/GUI/BaseGUI.py` | Modify `button()` — add `shortcut=` param, call `register_button()` (done, API matches) |
| `spammm/GUI/SPAMMM_GUI.py` | Register global shortcuts in `initUI()`, dispatch via registry in `on_key_press()`, add `create_help_section()`, add Help menu |
| `spammm/GUI/VispyUtils.py` | Register camera shortcuts as documentation-only in `AtomScene.__init__()` |
| `spammm/GUI/FFExtension.py` | Add `shortcut=` to "Interactive" button (Space) |
| Other extensions | Add `shortcut=` to buttons with key bindings (as applicable) |
| `tests/GUI/test_shortcut_registry.py` | **Create** — L0 tests for registry mechanism |
| `user_guide/GUI_CHEATSHEET.md` | Auto-generate keyboard section via `--export` (markers already added) |
| `user_guide/README.md` | Add cheatsheet + KekuleSolver to index (done in prior session) |
| `doc/KekuleSolverVisualization.md` → `user_guide/KekuleSolver_GUI.md` | **Move** |
