# Molecular Browser — Implementation Report

## Overview

This report documents the implementation of the modular 3D molecular viewer and browser (`MolecularBrowserVispy`), covering what was built, problems encountered and solved, and remaining issues with proposed solutions.

## Architecture

### Components

- **`MoleculeViewer.py`** — Standalone 3D molecule viewer using VisPy (PyQt5 backend). Renders atoms as markers, bonds as line segments, and optional text labels. Supports offscreen rendering for thumbnails.
- **`MolecularBrowserVispy.py`** — ACDSee-style file browser. Shows a grid of molecule thumbnails rendered offscreen. Enter opens selected molecule in a separate `MoleculeViewer` window.
- **`ThumbnailCache.py`** — Lazy job queue for offscreen thumbnail rendering. Renders one thumbnail per timer tick using a shared hidden `MoleculeViewer` instance.
- **`DirectoryNavigator.py`** — Filesystem navigation (list files, navigate subdirectories/parent).

### Data Flow

```
DirectoryNavigator → file list
    ↓
ThumbnailCache → MoleculeViewer.render_offscreen() → RGBA numpy arrays
    ↓
MolecularBrowserVispy → ImageVisual per thumbnail → VisPy canvas grid
    ↓
Enter key → MoleculeViewer (separate window) → interactive 3D view
```

## What Was Implemented

### 1. White Background

Changed all background colors from black to white across:
- Browser canvas (`MolecularBrowserVispy.py`): `bgcolor='white'`, text colors changed to black/dark
- Thumbnail viewer (`ThumbnailCache.py`): `bgcolor='white'`
- Standalone viewer (`MoleculeViewer.py`): `bgcolor='white'`

### 2. PCA-Based Orientation + Bounding Box Fit

**PCA alignment**: In `MoleculeViewer.load_file()`, the molecule is:
1. Centered at its center of gravity (COG): `pos -= cog`
2. Rotated to principal axes using `atomicUtils.orientPCA(pos)` (in-place rotation)

After PCA, the two largest spatial spreads are aligned to world X and Y axes, and the smallest spread (typically molecular plane normal) is along Z.

**Camera alignment**: In `auto_fit()`, the camera is set to look straight down Z:
- `azimuth = 0, elevation = 90` — so screen XY = world XY
- Camera center = `(0, 0, 0)` (molecule already centered at origin)
- Distance computed from the larger of X/Y spans: `distance = (max_span * 0.7) / tan(fov/2)`

This ensures the two largest principal axes map directly to screen X and Y, and the molecule fills the viewport.

**Verification**: Benzene after PCA has spans (4.87, 4.69, ~0) — planar molecule correctly oriented. Porphyrin (square) centered at (128, 128) in 256px thumbnail.

### 3. Atom Sizing

`default_atom_sizes` uses covalent radii × scale factor. Scale was increased from 10.0 to 40.0, giving:
- Carbon: ~29px diameter
- Hydrogen: ~15px diameter
- Oxygen: ~27px diameter

### 4. Antialiased Thumbnails via Native Render + Downscale

`render_offscreen()` renders at the canvas native size (512×512) — which centers correctly — then crops to square and downscales to the target size (256×256) using block averaging (2×2 blocks → 1 pixel).

**Why not `canvas.render(size=)`?**: VisPy's `canvas.render(size=(W,H))` renders at the requested resolution but the camera projection matrix is still computed from the canvas's actual size. This causes the molecule to appear off-center (bottom-left corner) in the output. Rendering at native size and downscaling avoids this bug.

### 5. Browser/Viewer Window Management

- **Enter in browser**: Opens molecule in separate `MoleculeViewer` window. Browser keeps showing thumbnails (not hidden).
- **Enter/Esc in viewer**: Closes viewer, returns to browse mode. Viewer's `key_press` event is forwarded to browser.
- **X button / Alt+F4 on viewer**: Detected via Qt native `closeEvent` monkey-patch. Automatically returns to browse mode and restores keyboard focus to browser window.
- **Focus restoration**: After viewer closes (by any means), browser calls `raise_()`, `activateWindow()`, `canvas.native.setFocus()`.

## Problems Encountered and Solutions

### Problem 1: Thumbnails Off-Center

**Symptom**: Molecules appeared in bottom-left corner of thumbnails instead of centered.

**Root cause**: `canvas.render(size=(256,256))` doesn't adjust the camera projection matrix. The camera still uses the original canvas size (512×512) for projection, causing a mismatch between render resolution and camera framing.

**Solution**: Render at native canvas size (512×512), crop to centered square, then downscale to target size. This ensures camera projection matches the rendered image.

### Problem 2: PCA Not Visible in Thumbnails

**Symptom**: Thumbnails showed molecules at odd angles, not aligned to principal axes.

**Root cause**: Camera had `elevation=30, azimuth=30` (tilted view). Even though PCA correctly aligned the molecule's principal axes to world X/Y, the tilted camera meant screen X/Y ≠ world X/Y.

**Solution**: Set camera to `elevation=90, azimuth=0` (look straight down Z) in `auto_fit()`. Now PCA-aligned world X/Y map directly to screen X/Y.

### Problem 3: Browser Loses Focus When Viewer Closes

**Symptom**: After closing viewer window (X button), browser couldn't receive keyboard input until Enter was pressed again.

**Root cause**: 
1. VisPy's `canvas.events.close` doesn't fire when Qt X button is clicked — it only fires on programmatic `canvas.close()`.
2. No explicit focus restoration after viewer window closed.

**Solution**: 
1. Monkey-patch Qt native `closeEvent` on viewer's canvas window to detect X button / Alt+F4.
2. In the close handler: restore original `closeEvent`, reset to BROWSE mode, call `raise_()`/`activateWindow()`/`setFocus()`.
3. In `exit_view_mode()`: restore original `closeEvent` before calling `canvas.close()` to avoid re-entry.

### Problem 4: Enter Not Working in Viewer Window

**Symptom**: Pressing Enter in viewer window did nothing.

**Root cause**: Key press handler was connected to the **browser** canvas. The viewer window has its own canvas. Key events in the viewer window never reached the browser's handler.

**Solution**: Connect `viewer.canvas.events.key_press` → `_on_viewer_key_press` which handles Enter/Esc by calling `exit_view_mode()`.

## Remaining Issues

### Issue 1: Thumbnail Rendering Performance

**Symptom**: Thumbnails render slowly (one at a time, ~100-200ms each). Browser feels laggy during initial load and when navigating directories.

**Root causes**:

1. **One MoleculeViewer per thumbnail render cycle**: `ThumbnailCache._get_viewer()` reuses a single viewer, but each `_render_molecule()` call does:
   - `AtomicSystem(fname=...)` — file I/O + bond finding (Python loop)
   - `viewer.load_file()` — PCA computation (numpy SVD on positions)
   - `viewer.render_offscreen()` — full GL render + numpy downscale
   - All on the main thread, blocking the UI

2. **One thumbnail per timer tick**: `ThumbnailCache.update(max_per_call=1)` renders only 1 thumbnail per 50ms timer tick. For 27 files, that's ~1.4 seconds minimum, but each render can take 100-200ms, so total is 3-5 seconds.

3. **ImageVisual recreation**: `_update_thumbnail_visuals()` destroys and recreates all `ImageVisual` objects every time a new thumbnail is ready. This causes GL state churn (shader compilation, texture uploads).

4. **No disk cache**: Thumbnails are re-rendered from scratch every time the directory is visited.

5. **No threading**: All rendering happens on the main Qt thread, blocking UI events.

**Proposed solutions**:

- **Reuse ImageVisuals**: Instead of destroying/recreating, update existing `ImageVisual.set_data()` when a new thumbnail arrives. Avoids shader recompilation.
- **Batch rendering**: Render multiple thumbnails per timer tick (e.g., `max_per_call=3`). Tune based on frame time budget.
- **Disk cache**: Save rendered thumbnails as PNG files in a `.thumbs/` subdirectory. Check modification time vs source file to invalidate.
- **Background thread**: Move file I/O and PCA computation to a QThread. GL rendering must stay on main thread (OpenGL context is thread-affine), but data preparation can be parallelized.
- **Texture atlas (Phase 2)**: Pack all thumbnails into a single texture atlas. Use instanced quad rendering instead of individual `ImageVisual` objects. Single draw call for entire grid.
- **Preload visible thumbnails first**: Render thumbnails in viewport first (visible rows), then off-screen ones. Prioritize by scroll position.

### Issue 2: Bond Rendering Quality

**Current implementation**: `compute_bond_segments()` in `MoleculeViewer.py:54-62` converts `(m,2)` bond indices into `(2m,3)` endpoint coordinates:

```python
segs[0::2] = pos[bonds[:, 0]]  # even indices = bond start
segs[1::2] = pos[bonds[:, 1]]  # odd indices = bond end
```

This produces a flat array of 2m vertices. The `visuals.Line` is created without specifying `connect='segments'`:

```python
self.bond_lines = visuals.Line(parent=self.view.scene, color=(0.3, 0.3, 0.3, 0.8), width=2.0, antialias=True)
```

**The problem**: VisPy `Line` with default `connect='strip'` treats the vertex array as a **continuous polyline** — it connects vertex 0→1→2→3→4→..., meaning:
- Vertex 1 (end of bond 0) connects to vertex 2 (start of bond 1) — **spurious bond**
- Vertex 3 (end of bond 1) connects to vertex 4 (start of bond 2) — **spurious bond**
- etc.

This draws extra line segments between the end of one bond and the start of the next, creating a zigzag pattern connecting all bonds in sequence. This is why bonds look "strange" — there are phantom lines between non-bonded atoms.

**The fix** (not yet implemented): Set `connect='segments'` when creating the Line visual, or when calling `set_data()`:

```python
self.bond_lines = visuals.Line(parent=self.view.scene, color=..., width=2.0, antialias=True, connect='segments')
```

With `connect='segments'`, VisPy treats each pair of vertices as an independent line segment: (0,1), (2,3), (4,5), ... — exactly one segment per bond, no spurious connections.

Alternatively, use `connect=np.array([...])` to explicitly specify pairs.

**Additional improvement**: For nicer bond rendering, use per-vertex colors so each half of a bond is colored by the respective atom (CPK style). This requires:
- Splitting each bond into two segments (atom A → midpoint, midpoint → atom B)
- Setting vertex colors accordingly
- Using `Line(color=vertex_colors_array)` with `connect='segments'`

## File Summary

| File | Role |
|------|------|
| `spammm/GUI/MoleculeViewer.py` | 3D viewer: atoms, bonds, labels, PCA, auto-fit, offscreen render |
| `spammm/GUI/MolecularBrowserVispy.py` | Browser: thumbnail grid, navigation, viewer window management |
| `spammm/GUI/ThumbnailCache.py` | Lazy thumbnail rendering queue |
| `spammm/GUI/DirectoryNavigator.py` | Filesystem navigation |
| `spammm/atomicUtils.py` | PCA utilities (`rotMatPCA`, `orientPCA`) |

## Test Status

All existing tests pass:
- `tests/test_molecule_viewer.py`: 7/7 passed
- `tests/test_thumbnail_cache.py`: 5/5 passed
- `tests/test_directory_navigator.py`: 6/6 passed
