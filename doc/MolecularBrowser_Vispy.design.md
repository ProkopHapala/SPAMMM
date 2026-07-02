# Molecular Browser (VisPy) — Design Document

## Overview

ACDSee-style 3D molecular file browser built with **VisPy + PyQt5**. Browse directories of `.xyz`/`.mol`/`.mol2` files as a thumbnail grid of pre-rendered 3D molecule images. Click/Enter to switch to interactive 3D view. Inspired by FireCore's `BrowserView.h` (C++/SDL/OpenGL) but implemented in Python with VisPy for portability and integration with SPAMMM's existing `AtomScene`.

**Goal:** High-performance browser that can handle thousands of molecules with minimal Python overhead. Core strategy: **pre-render thumbnails to GPU textures, display as instanced sprites**.

**Modularity:** The 3D viewer (`MoleculeViewer`) is a standalone class usable independently, inside the browser, or embedded in `SPAMMM_GUI.py`'s editor. The browser calls the viewer when user hits Enter on a thumbnail. The same viewer can be toggled in the editor (which is 2D by default) for 3D inspection.

---

## Reference Implementations

| Source | File | What to borrow |
|--------|------|----------------|
| FireCore C++ | `cpp/apps/MolecularEditor/BrowserView.h` | Thumbnail rendering, grid layout, dir navigation, cursor, hit-testing |
| FireCore C++ | `cpp/apps/MolecularEditor/MolView.h` | Auto-fit zoom, label toggles, lightweight 3D view |
| FireCore C++ | `cpp/apps/MolecularEditor/MolecularBrowser.cpp` | BROWSE↔VIEW mode switching, shared resources |
| FireCore C++ | `cpp/common_SDL/browser_sdl.h` | Lazy thumbnail job queue (one per frame) |
| SPAMMM Python | `spammm/GUI/MolecularBrowser.py` | Existing PyOpenGL browser (being replaced) |
| SPAMMM Python | `spammm/GUI/VispyUtils.py` | `AtomScene` — base for `MoleculeViewer` (rendering + interaction split) |
| SPAMMM Python | `spammm/GUI/SPAMMM_GUI.py` | `KekuleExplorerWindow` — embeds `AtomScene`, will embed `MoleculeViewer` for 3D toggle |
| SPAMMM Python | `spammm/GUI/BaseGUI.py` | PyQt5 widget factory helpers |
| SPAMMM Python | `spammm/AtomicSystem.py` | Molecule loading (.xyz/.mol/.mol2), bond finding |
| SPAMMM Python | `spammm/elements.py` | `ELEMENT_DICT`, `hex_to_float_rgb`, atom colors/radii |

---

## Architecture

```
MolecularBrowserVispy (QMainWindow)
│
├── DirectoryNavigator
│   ├── work_dir: str
│   ├── file_names: List[str]       # .xyz/.mol/.mol2 files
│   ├── sub_dirs: List[str]         # subdirectories (includes "..")
│   ├── navigate_to(dir: str)
│   └── read_dir() -> (files, dirs)
│
├── ThumbnailCache
│   ├── textures: Dict[int, np.ndarray]    # mol_index -> (H,W,4) RGBA
│   ├── atlas: TextureAtlas                 # packed GPU texture
│   ├── render_queue: deque[int]            # indices pending render
│   ├── render_one() -> bool                # render next in queue, return True if rendered
│   ├── get_thumbnail(i) -> (uv_rect, tex_id)
│   └── invalidate()                        # clear all on dir change
│
├── BrowserCanvas (VisPy SceneCanvas)
│   ├── BROWSE mode:
│   │   ├── ThumbnailGrid (instanced quads + atlas)
│   │   │   ├── positions: np.ndarray (N,2)  # grid layout, numpy-computed
│   │   │   ├── uv_offsets: np.ndarray (N,4) # per-instance atlas UVs
│   │   │   ├── sizes: np.ndarray (N,2)      # per-instance quad size
│   │   │   └── Visual: instanced textured quads (custom or vispy Compound)
│   │   ├── DirButtons (rectangles + text)
│   │   ├── PathBar (text)
│   │   ├── Cursor (line rectangle outline)
│   │   └── Labels (VisPy Text visual — filenames under thumbnails)
│   │
│   └── VIEW mode:
│       └── MoleculeViewer (standalone, reusable)
│           ├── set_data(pos, colors, sizes, bonds)
│           ├── auto_fit_zoom()           # from MolView.h pattern
│           └── interactive camera (turntable)
│
├── InputHandler
│   ├── keyboard: arrows (cursor), Enter (toggle mode), Backspace (parent dir)
│   ├── mouse: click (select/navigate), wheel (scroll)
│   └── hit_test(mx, my) -> (type, index)
│
└── StatusBar / InfoBar
    ├── current dir path
    ├── molecule count
    └── selected molecule info (name, atoms, bonds)
```

---

## Modular 3D Viewer Architecture

The 3D viewer is a **separate, standalone class** (`MoleculeViewer`) that can be used in three contexts:

1. **Standalone:** `python -m spammm.GUI.MoleculeViewer --file molecule.xyz`
2. **Inside the browser:** Browser switches to `MoleculeViewer` when user hits Enter on a thumbnail
3. **Inside `SPAMMM_GUI.py`:** Editor toggles between 2D top-down (current `AtomScene`) and 3D perspective (`MoleculeViewer`)

### Current problem: `AtomScene` mixes rendering + editing

`AtomScene` in `VispyUtils.py` currently bundles:
- **Rendering:** atoms (markers), bonds (lines), forces, labels, camera, GL state
- **Editing interaction:** picking, dragging, bond creation (Ctrl+drag), ring preview, selection rect, RMB remove

This makes it hard to reuse as a pure viewer. The solution is to split into layers:

```
MoleculeViewer (standalone 3D viewer)
│
├── MolRenderVisuals        # Pure rendering — no interaction, no signals
│   ├── atom_markers        # VisPy Markers
│   ├── bond_lines          # VisPy Line
│   ├── force_lines         # VisPy Line (optional)
│   ├── text_labels         # VisPy Text (optional)
│   ├── bbox_lines          # VisPy Line (optional)
│   └── camera              # TurntableCamera (3D) or PanZoomCamera (2D)
│
├── MolInteractionHandler   # Optional — adds picking, dragging, selection
│   ├── sig_atom_picked
│   ├── sig_drag_state
│   ├── sig_rmb_remove
│   ├── sig_link_bond
│   ├── sig_atom_clicked
│   └── sig_selection_changed
│   (wired via composition, not inheritance)
│
└── MolEditHandler          # Optional — adds ring preview, bond creation, hex grid
    ├── ring_preview_line
    ├── link_line
    └── hex_grid_markers
    (used only by SPAMMM_GUI editor)
```

### Composition pattern

```python
# Standalone viewer (browser or CLI):
viewer = MoleculeViewer(canvas, interaction=False)
viewer.set_data(pos, colors, sizes, bonds)
viewer.auto_fit()

# SPAMMM_GUI editor (full editing):
viewer = MoleculeViewer(canvas, interaction=True, editing=True)
viewer.backend = self.backend  # KekuleBackend for authoritative geometry
viewer.sig_drag_state.connect(self.on_drag_state)
viewer.sig_rmb_remove.connect(self.on_atom_remove)
# ... same signals as current AtomScene
```

### Migration path from `AtomScene`

`AtomScene` is not deleted — it becomes a thin compatibility wrapper:

```python
class AtomScene(QtCore.QObject):
    """Deprecated — use MoleculeViewer with interaction=True, editing=True."""
    def __init__(self, *, bgcolor='white', backend=None):
        self._viewer = MoleculeViewer(bgcolor=bgcolor, interaction=True, editing=True)
        self._viewer.backend = backend
        # Forward all signals
        self.sig_atom_picked = self._viewer.sig_atom_picked
        # ...
    
    def set_data(self, *args, **kwargs):
        self._viewer.set_data(*args, **kwargs)
```

This lets `SPAMMM_GUI.py` keep working unchanged during migration, then gradually switch to `MoleculeViewer` directly.

---

## Component Details

### 1. DirectoryNavigator

Thin wrapper around `os.listdir` with file extension filtering. Mirrors FireCore's `Browser::readDir`.

```python
class DirectoryNavigator:
    work_dir: str
    file_names: list[str]    # sorted, filtered by extension
    sub_dirs: list[str]      # includes ".." as first entry

    EXTENSIONS = {'.xyz', '.mol', '.mol2'}

    def read_dir(self) -> None
    def navigate_to(self, dir_name: str) -> None  # handles "..", absolute, relative
    def parent_dir(self) -> str
```

**Data flow:** `navigate_to()` → `read_dir()` → `ThumbnailCache.invalidate()` + populate render queue → `BrowserCanvas` redraws.

### 2. ThumbnailCache

**Core performance component.** Pre-renders each molecule to a small RGBA image, packs into a texture atlas.

#### Rendering pipeline (per molecule)

1. Load molecule: `AtomicSystem(fname=path)` → get `apos`, `enames`, `bonds`
2. Auto-center: `apos -= apos.mean(axis=0)` (subtract COG)
3. Auto-orient (optional): align principal axes (like FireCore's `FindRotation`)
4. Find bonds if needed: `system.findBonds()`
5. Compute auto-fit zoom: `max_span = max(ptp(apos, axis=0))`; `zoom = max_span * 0.7`
6. Render to offscreen image using a **dedicated lightweight VisPy canvas**:
   - Create `SceneCanvas(size=(thumb_size, thumb_size), show=False)`
   - Add `AtomScene` or simpler renderer with markers (atoms) + lines (bonds)
   - Set camera distance/zoom from auto-fit
   - `canvas.render()` → numpy array `(H, W, 4)` RGBA
7. Store in cache: `textures[i] = rendered_array`

#### Offscreen rendering approach

Two options:

**Option A: VisPy `SceneCanvas.render()` (preferred)**
- VisPy supports offscreen rendering via `canvas.render(alpha=True)`
- Returns numpy array directly — no OpenGL FBO management
- Can use a single shared `SceneCanvas` instance, swap data per molecule
- Pro: pure VisPy, no raw OpenGL
- Con: `canvas.render()` may have overhead per call (context switching)

**Option B: Raw OpenGL FBO (like existing MolecularBrowser.py)**
- Use `glGenFramebuffers` + `glFramebufferTexture2D`
- Render with existing sphere/cylinder shaders
- Pro: maximum control, minimal overhead
- Con: raw OpenGL, duplicates VisPy rendering logic

**Decision:** Start with Option A (VisPy `canvas.render()`). If profiling shows it's too slow, fall back to Option B or hybrid (VisPy for 3D view, raw GL for thumbnails).

#### Texture atlas

Pack thumbnails into a single large texture (e.g., 2048×2048) to enable batch rendering:

```
Atlas 2048×2048, thumb 256×256 → 8×8 = 64 thumbnails per atlas
For >64 molecules: multiple atlases or larger texture (4096×4096 = 256 thumbs)
```

```python
class TextureAtlas:
    tex_size: int = 2048          # atlas texture size
    thumb_size: int = 256         # individual thumbnail size
    cols: int                     # tex_size // thumb_size
    rows: int                     # tex_size // thumb_size
    capacity: int                 # cols * rows
    gpu_texture: vispy.gloo.Texture2D
    used: int = 0                 # next free slot

    def add(self, image: np.ndarray) -> tuple[float, float, float, float]
        # Returns (u0, v0, u1, v1) UV rect in atlas
    def clear(self) -> None
```

**UV computation:** All thumbnail positions in atlas precomputed as numpy array:
```python
# For N thumbnails in grid layout:
cols = N_atlas_cols
u0 = (indices % cols) * thumb_size / atlas_size
v0 = (indices // cols) * thumb_size / atlas_size
u1 = u0 + thumb_size / atlas_size
v1 = v0 + thumb_size / atlas_size
uv_rects = np.column_stack([u0, v0, u1, v1])  # (N, 4)
```

#### Lazy rendering (job queue)

Following `browser_sdl.h` pattern — render one thumbnail per frame to avoid blocking UI:

```python
class ThumbnailCache:
    render_queue: deque[int]     # molecule indices pending render
    placeholders: set[int]       # indices showing placeholder

    def update(self) -> bool:
        """Render one thumbnail per call. Returns True if something was rendered."""
        if not self.render_queue:
            return False
        idx = self.render_queue.popleft()
        image = self._render_molecule(idx)
        uv = self.atlas.add(image)
        self.uv_rects[idx] = uv
        return True
```

Called from a QTimer (e.g., every 16ms = 60fps): if queue not empty, render one. This gives smooth UI while thumbnails populate progressively.

#### Disk cache (future extension, NOT implemented now)

**Not implementing disk cache for now** — rendering should be fast enough with in-memory lazy rendering. Disk cache would pollute the filesystem. If needed in the future, it could save rendered thumbnails to `~/.cache/spammm/thumbs/` keyed by file hash, but this is explicitly deferred.

### 3. BrowserCanvas

The main VisPy canvas. Two modes rendered into the same `SceneCanvas`:

#### BROWSE mode — thumbnail grid

**Grid layout (numpy-computed, no Python loops per frame):**
```python
def compute_grid(n_items, n_cols, thumb_size, spacing, canvas_size, scroll_y):
    rows = np.arange(n_items) // n_cols
    cols = np.arange(n_items) % n_cols
    x = cols * (thumb_size + spacing) + margin
    y = rows * (thumb_size + spacing) + scroll_y
    return np.column_stack([x, y])  # (N, 2) screen positions
```

**Rendering options:**

**Option A: VisPy Markers (simplest)**
- Use `visuals.Markers` with per-point texture coordinates (if supported)
- One draw call for all visible thumbnails
- Limitation: VisPy markers may not support per-instance textures easily

**Option B: Custom instanced quad visual (most performant)**
- Create a single quad mesh (4 vertices)
- Use VisPy's `gloo` or custom GLSL program with instanced rendering
- Per-instance attributes: position (offset), size (scale), UV rect (atlas coords)
- Vertex shader: `gl_Position = proj * vec4(pos + offset + a_quad * size, 0, 1)`
- Fragment shader: `texture(atlas, mix(uv0, uv1, a_uv))`
- This is the **ideal approach** — one draw call for all thumbnails

**Option C: Multiple ImageVisuals (fallback)**
- One `visuals.Image` per thumbnail
- Simple but N draw calls, N texture binds
- Acceptable for <100 molecules, not scalable

**Decision:** Use Option B (custom instanced quads) for performance. Fall back to Option C for initial prototype if instanced quads prove complex.

**Viewport culling:** Only render thumbnails whose grid position is within canvas bounds:
```python
visible = (y > -thumb_size) & (y < canvas_h) & (x > -thumb_size) & (x < canvas_w)
positions_visible = positions[visible]
uv_rects_visible = uv_rects[visible]
```

**Cursor:** Green rectangle outline drawn at selected thumbnail position. Updated on key/mouse navigation.

**Text labels:** VisPy `visuals.Text` for filenames under thumbnails. Batch all labels in one Text visual.

**Directory buttons:** Rectangles + text for subdirectories at top. Click to navigate.

#### VIEW mode — interactive 3D

Uses `MoleculeViewer` (standalone class, see Modular 3D Viewer Architecture above):

```python
def enter_view_mode(self, mol_index):
    system = self.navigator.load_molecule(mol_index)
    pos = system.apos.astype(np.float32)
    colors = np.array([elements.hex_to_float_rgb(elements.ELEMENT_DICT[e][8]) + (1.0,) for e in system.enames])
    sizes = np.array([elements.ELEMENT_DICT[e][6] * 10 for e in system.enames])
    bonds = system.bonds if system.bonds is not None else system.findBonds()[0]
    self.viewer.set_data(pos, colors=colors, sizes=sizes, bonds=bonds)
    self.viewer.auto_fit()
```

`MoleculeViewer` is the same class used standalone and in `SPAMMM_GUI.py`.

### 4. InputHandler

| Input | BROWSE mode | VIEW mode |
|-------|-------------|-----------|
| Left/Right arrows | Move cursor left/right | Rotate camera |
| Up/Down arrows | Move cursor up/down | Rotate camera |
| Enter | Switch to VIEW mode | Switch back to BROWSE |
| Backspace | Navigate to parent dir | — |
| Mouse click | Select thumbnail or navigate dir | Pick atom |
| Mouse wheel | Scroll grid | Zoom |
| Esc | Quit | Back to BROWSE |

**Hit testing** (from `BrowserView.h` pattern):
```python
def hit_test(self, mx, my):
    # Check dir buttons first
    for i, rect in enumerate(self.dir_rects):
        if point_in_rect(mx, my, rect):
            return ('dir', i)
    # Check thumbnails
    for i, rect in enumerate(self.thumb_rects):
        if point_in_rect(mx, my, rect):
            return ('mol', i)
    return ('none', -1)
```

Optimized with numpy: compute all rect distances at once.

---

## Performance Strategy

### Minimizing Python overhead

| Operation | Approach | Python calls per frame |
|-----------|----------|----------------------|
| Grid layout | Precompute numpy array, update only on scroll/resize | 0 (cached) |
| Thumbnail rendering | One instanced draw call with atlas | 1 draw call |
| Text labels | Single `Text` visual with all labels | 1 draw call |
| Thumbnail generation | One per frame via job queue (not blocking) | 1 render call |
| Molecule loading | On directory change only (not per-frame) | 0 |
| Hit testing | Numpy vectorized rect check | 1 numpy op |

### Memory budget

| Component | Size | Notes |
|-----------|------|-------|
| Thumbnail atlas 2048² | 16 MB (RGBA) | 64 thumbnails at 256² |
| Thumbnail atlas 4096² | 64 MB (RGBA) | 256 thumbnails at 256² |
| Molecule data (loaded) | ~1 KB per small molecule | Only loaded molecules kept in memory |
| VisPy canvas | ~few MB | GPU textures + buffers |

### Scalability

- **100 molecules:** Single 2048² atlas, instant
- **1000 molecules:** Multiple atlases or 4096², lazy rendering (one per frame = ~17s at 60fps, but progressive display)
- **10000 molecules:** Need virtual scrolling (only load/render visible range), disk cache for thumbnails

---

## File Structure

```
spammm/GUI/
├── MoleculeViewer.py           # NEW — standalone modular 3D viewer (rendering + optional interaction + optional editing)
├── MolecularBrowserVispy.py    # NEW — main browser widget (QMainWindow), uses MoleculeViewer for VIEW mode
├── ThumbnailCache.py           # NEW — offscreen rendering + texture atlas (uses MoleculeViewer for rendering)
├── DirectoryNavigator.py       # NEW — directory reading + navigation
├── BrowserGrid.py              # NEW — instanced quad grid rendering (VisPy)
├── VispyUtils.py               # EXISTING — AtomScene (becomes thin wrapper around MoleculeViewer)
├── BaseGUI.py                  # EXISTING — PyQt5 widget helpers
├── SPAMMM_GUI.py               # EXISTING — KekuleExplorerWindow (will toggle MoleculeViewer for 3D mode)
├── MolecularBrowser.py         # EXISTING — old PyOpenGL browser (deprecated)
├── GLGUI.py                    # EXISTING — old PyOpenGL infrastructure (deprecated)
└── shaders/                    # EXISTING — GLSL shaders
    ├── thumbnail_quad.glslv    # NEW — instanced quad vertex shader
    └── thumbnail_quad.glslf    # NEW — instanced quad fragment shader
```

---

## API Sketch

### MoleculeViewer

```python
class MoleculeViewer(QtCore.QObject):
    """Standalone modular 3D molecular viewer.
    
    Three layers, enabled via constructor flags:
    - rendering: always on (atoms, bonds, labels, camera)
    - interaction: optional (picking, dragging, selection)
    - editing: optional (ring preview, bond creation, hex grid)
    """
    # Signals (only active if interaction=True)
    sig_atom_picked = QtCore.pyqtSignal(int)
    sig_drag_state = QtCore.pyqtSignal(int, int, object)
    sig_atom_moved = QtCore.pyqtSignal(int, object)
    sig_rmb_remove = QtCore.pyqtSignal(int)
    sig_selection_changed = QtCore.pyqtSignal(object)
    sig_camera_changed = QtCore.pyqtSignal()
    sig_link_bond = QtCore.pyqtSignal(int, int)
    sig_atom_clicked = QtCore.pyqtSignal(int)

    def __init__(self, *, bgcolor='white', canvas=None, interaction=False, editing=False, backend=None):
        # If canvas=None, creates own SceneCanvas (standalone mode)
        # If canvas provided, attaches to existing canvas (embedded mode)
        ...

    # --- Data ---
    def set_data(self, pos, colors=None, sizes=None, bonds=None, forces=None, radius=None, atom_ids=None):
        ...
    def auto_fit(self) -> None:
        # Auto-center + auto-zoom from bounding box
        ...

    # --- Camera ---
    def set_camera_2d(self) -> None:   # orthographic top-down (elevation=90)
    def set_camera_3d(self) -> None:   # perspective, free rotation
    def set_zoom(self, zoom: float) -> None

    # --- Labels ---
    def set_label_mode(self, mode: str) -> None  # 'none', 'Element+Index', 'Pi Orbitals', etc.

    # --- Rendering (always available) ---
    def render_offscreen(self, size=256) -> np.ndarray:
        # Render current scene to RGBA numpy array (for thumbnails)
        # Uses canvas.render() if standalone, or FBO if embedded
        ...

    # --- Interaction (only if interaction=True) ---
    def set_selection_mode(self, enabled: bool) -> None
    def lock_drag(self, locked: bool) -> None

    # --- Editing (only if editing=True) ---
    def set_ring_preview(self, ...) -> None
    def set_link_mode(self, enabled: bool) -> None
```

### MolecularBrowserVispy

```python
class MolecularBrowserVispy(BaseGUI):
    """ACDSee-style molecular file browser using VisPy."""

    def __init__(self, start_dir='.', thumb_size=256, title='Molecular Browser'):
        ...

    # --- Mode switching ---
    def enter_browse_mode(self) -> None
    def enter_view_mode(self, mol_index: int) -> None

    # --- Directory ---
    def navigate_to(self, dir_name: str) -> None

    # --- Thumbnail rendering loop ---
    def _on_thumbnail_timer(self) -> None
        # Called by QTimer every ~16ms
        # Renders one thumbnail from queue if available
        # Updates grid visual with new atlas UVs

    # --- Input ---
    def _on_key_press(self, event) -> None
    def _on_mouse_press(self, event) -> None
    def _on_mouse_wheel(self, event) -> None

    # --- Rendering ---
    def _draw_browse(self) -> None
    def _draw_view(self) -> None
```

### ThumbnailCache

```python
class ThumbnailCache:
    def __init__(self, thumb_size=256, atlas_size=2048):
        ...

    def set_molecules(self, file_paths: list[str]) -> None
        # Populate render_queue with all indices

    def update(self) -> bool
        # Render one thumbnail from queue
        # Returns True if something was rendered

    def get_uv_rect(self, index: int) -> np.ndarray
        # Returns (u0, v0, u1, v1) atlas UV coordinates for thumbnail

    def is_ready(self, index: int) -> bool
        # True if thumbnail has been rendered

    def invalidate(self) -> None
        # Clear all thumbnails (on directory change)

    def _render_molecule(self, file_path: str) -> np.ndarray
        # Load molecule, render to offscreen VisPy canvas
        # Returns RGBA image array (thumb_size, thumb_size, 4)
```

### DirectoryNavigator

```python
class DirectoryNavigator:
    EXTENSIONS = {'.xyz', '.mol', '.mol2'}

    def __init__(self, start_dir='.'):
        self.work_dir = os.path.abspath(start_dir)
        self.file_paths: list[str] = []
        self.file_names: list[str] = []
        self.sub_dirs: list[str] = []

    def read_dir(self) -> None
        # Populate file_names, file_paths, sub_dirs

    def navigate_to(self, dir_name: str) -> None
        # Handle "..", absolute paths, relative paths

    def parent_dir(self) -> str
```

---

## Integration with SPAMMM GUI

Three usage contexts for `MoleculeViewer`:

### 1. Standalone viewer
```bash
python -m spammm.GUI.MoleculeViewer --file molecule.xyz
```
Pure 3D view, no editing. Camera controls only (rotate, zoom, pan).

### 2. Inside the browser
Browser calls `MoleculeViewer` when user hits Enter on a thumbnail. No editing — just viewing. Esc returns to browse grid.

### 3. Inside `SPAMMM_GUI.py` (KekuleExplorerWindow)
The editor is 2D top-down by default (current `AtomScene` with orthographic camera, elevation=90). A toggle button switches to 3D perspective mode using `MoleculeViewer` with `interaction=True, editing=True`:

```python
# In KekuleExplorerWindow:
def toggle_3d_view(self):
    if self._3d_mode:
        # Back to 2D
        self.scene = AtomScene(bgcolor='white', backend=self.backend)  # or MoleculeViewer with 2D camera
        self._3d_mode = False
    else:
        # Switch to 3D
        self.viewer_3d = MoleculeViewer(canvas=self.scene.canvas, interaction=True, editing=True)
        self.viewer_3d.backend = self.backend
        self.viewer_3d.set_data(...)  # same data as 2D scene
        self.viewer_3d.set_camera_3d()  # perspective, free rotation
        self._3d_mode = True
```

The same signals (`sig_drag_state`, `sig_rmb_remove`, etc.) work in both modes, so the editor logic doesn't change.

### 4. Browser as extension in editor
The browser itself can also be launched from within `SPAMMM_GUI.py` as a dialog or dock widget to browse molecule databases and import selected molecules into the editor.

---

## Implementation Phases

### Phase 1: Minimal viable browser (prototype)
- `MoleculeViewer` — extract from `AtomScene`, split rendering vs interaction vs editing
- `DirectoryNavigator` — read dir, filter extensions
- `ThumbnailCache` — render molecules via `MoleculeViewer` + `SceneCanvas.render()`, store as numpy arrays
- `BrowserCanvas` — simple grid using VisPy `ImageVisual` per thumbnail (Option C)
- BROWSE↔VIEW mode switching with `MoleculeViewer` for 3D view
- Keyboard navigation + mouse click
- `AtomScene` becomes thin wrapper around `MoleculeViewer` (backward compat for `SPAMMM_GUI.py`)

### Phase 2: Performance optimization
- Replace per-thumbnail `ImageVisual` with instanced quads + texture atlas (Option B)
- Lazy rendering via QTimer job queue
- Viewport culling
- Numpy-vectorized hit testing

### Phase 3: Polish
- Subdirectory navigation buttons
- Path bar
- Status bar with molecule info
- Auto-orientation (principal axis alignment) for thumbnails
- Filtering/search by formula/name
- Multi-select + batch operations
- Disk cache for thumbnails (future, not implemented now — would pollute disk)

---

## Key Design Principles (from AGENTS.md)

- **KISS:** Simplest solution that works. One-liner > ten-liner.
- **SoC:** Separate ThumbnailCache (rendering), DirectoryNavigator (IO), BrowserCanvas (display), InputHandler (control)
- **SSOT:** AtomicSystem is the single source for molecule data. ThumbnailCache owns thumbnail textures.
- **Fail Fast:** No silent try/except. Missing files crash with stack trace.
- **Performance:** Preallocate numpy arrays, minimize per-frame Python, batch GPU calls.
- **DRY:** Reuse `MoleculeViewer` for 3D view everywhere (standalone, browser, editor), reuse `BaseGUI` for widgets, reuse `AtomicSystem` for loading.
- **Modularity:** `MoleculeViewer` is composable — rendering, interaction, and editing are separate layers that can be enabled independently.
