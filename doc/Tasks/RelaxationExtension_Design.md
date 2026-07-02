# Relaxation Extension Design

## Goal

Build a GUI extension that:
1. Builds a forcefield (SPFF or UFF) from the current molecular structure
2. Runs one or several loops of relaxation (FIRE / damped MD)
3. Supports interactive per-frame relaxation with real-time Vispy visualization
4. Supports constraints — pinning selected atoms via Pin/Unpin button or click

---

## What We Already Have

### Extension System (`ExtensionManager.py`)
- `EXTENSION_REGISTRY` already lists `spff` and `moldyn` extensions (both disabled by default)
- `UIComponents` pattern: each extension returns `(panel, edit_modes, view_modes)`
- `build_ui(name, window)` imports module, calls its `build_ui()` function
- Extensions get a `CollapsibleSection` in the sidebar automatically
- **What's missing**: No `build_ui()` function exists yet in `SPFF.py` or `MolecularDynamics.py` (grep found none)

### Forcefield Building
- **`SPFF.py`**: `SPFF` class converts `AtomicSystem` → SPFF GPU arrays via `toSPFFsp3_loc()`. Handles topology, neighbors, bond params, pi-orbital directions. Uses `FFparams.py` for parameter loading.
- **`UFFbuilder.py`**: `UFF_Builder` class converts `AtomicSystem` → UFF arrays via `build()` + `get_arrays()`. Enumerates bonds, angles, dihedrals, inversions. Assigns UFF atom types with complex rules (aromaticity, resonance, conjugation).
- **`FFparams.py`**: Parses `ElementTypes.dat`, `AtomTypes.dat`, `BondTypes.dat`. Provides `SPFFparams` for parameter lookup. Used by both SPFF and UFFbuilder.
- **`AtomicSystem.py`**: Flat-array molecular representation (`apos`, `enames`, `bonds`, `ngs`). Has `findBonds()`, `neighs()`. This is the input to both FF builders.

### MD / Relaxation Engine (`MolecularDynamics.py`)
- `MolecularDynamics` class: GPU-accelerated MD using OpenCL
- **`relax(nsteps, dt, damp, Flimit)`** — runs nsteps of damped MD on GPU, returns final energy
- **`run_step_basic()`** / **`run_step_rot()`** — single MD step (cleanForce → getSPFF → updateAtoms)
- **`get_positions()`** — downloads `(natoms, 3)` positions from GPU
- **`get_forces()`** — downloads `(natoms, 4)` forces from GPU
- **`get_total_energy()`** — downloads and sums energy component
- **`download_results()`** — gets both positions and forces in one call
- **`set_md_params(dt, damp, Flimit)`** — updates MD parameters on GPU
- `kernel_params['mask']` exists as `int4` `[1,1,1,1]` — currently a dimension mask, **not** per-atom. Could be extended or a new per-atom mask buffer added.
- Loads kernels: `common.cl`, `Forces.cl`, `SPFF.cl`, `gridFF.cl`, `surface.cl`, `nonbonded.cl`

### Visualization (`VispyUtils.py` — `AtomScene`)
- **`set_data(pos, forces=..., force_scale=...)`** — can display per-atom force vectors as red lines
- **`update_positions(pos)`** — fast position-only update, calls `_redraw()`
- **`_fixed` boolean mask** — already implemented!
  - `set_fixed_mask(fixed)` — set entire mask
  - `toggle_fixed(i)` — toggle single atom
  - `is_fixed(i)` — query
  - Fixed atoms are excluded from picking/dragging (lines 673, 988, 1010, 1142)
- **Selection system**: `_selected_ids` (set of Atom._id), `_selection_mode`, `sig_selection_changed` signal
- **`force_lines`** visual — Line object for force vectors, already in draw pipeline

### GUI (`SPAMMM_GUI.py`)
- `run_relaxation()` — currently calls `backend.run_relaxation()` which uses DFTB+ (external). This is the **DFTB path**, not the GPU FF path.
- `refresh_view()` — full scene update from `backend.sys`
- `on_drag_state()` — syncs positions after drag, calls `scene.update_positions()`
- `on_key_press()` — keyboard shortcut handler
- `_build_extension_panels()` — dynamically builds UI from extensions
- `set_edit_mode()` — dispatches to extension edit mode callbacks
- Selection state managed via `scene._selected_ids`

---

## Design

### Architecture Overview

```
SPAMMM_GUI (KekuleExplorerWindow)
  └── ExtensionManager loads 'relax' extension
        └── RelaxationExtension.build_ui(window) → UIComponents
              ├── Panel: FF type selector, Build FF, Relax N steps, Interactive toggle
              ├── Edit mode: "Pin/Unpin" — click atoms to toggle pin
              └── View mode: "Show Forces" — display force vectors

RelaxationController (logic, no Qt dependency)
  ├── build_ff(AtomicSystem, ff_type='spff'|'uff') → sets up MD engine
  ├── relax_step() → one MD step on GPU
  ├── relax_n(nsteps) → batch relaxation
  ├── get_state() → {positions, forces, energy}
  ├── set_pinned(mask) → propagate constraint to MD
  └── teardown() → release GPU resources

AtomScene (VispyUtils.py) — already has:
  ├── _fixed mask (pinning visualization + interaction block)
  ├── update_positions() (fast feedback)
  └── set_data(forces=...) (force vector display)
```

### New Files

1. **`spammm/GUI/RelaxationExtension.py`** — Extension module
   - `build_ui(window)` → `UIComponents` with panel + edit/view modes
   - Creates `RelaxationController` instance on the window
   - Panel widgets:
     - FF type combo (SPFF / UFF)
     - "Build FF" button → calls `controller.build_ff(backend.sys, ff_type)`
     - "Relax 100 steps" button → calls `controller.relax_n(100)` then updates view
     - "Interactive" checkbox → starts/stops QTimer loop
     - "Pin/Unpin" button → toggles pin on selected atoms
     - "Clear Pins" button
     - Energy display label
   - Edit mode "Pin/Unpin": click atom → toggle `scene.toggle_fixed(idx)`, update controller

2. **`spammm/forcefields/RelaxationController.py`** — Pure logic controller
   - Wraps `MolecularDynamics` + `SPFF` / `UFFbuilder`
   - Methods: `build_ff()`, `relax_step()`, `relax_n()`, `get_state()`, `set_pinned()`
   - No Qt imports — pure Python/NumPy/OpenCL

### Registration

Add to `EXTENSION_REGISTRY` in `ExtensionManager.py`:
```python
'relax': dict(
    module='spammm.GUI.RelaxationExtension', class_name=None,
    dependencies=['pyopencl'], req_paths=[],
    build_ui='build_ui',
),
```
Enable in `DEFAULT_CONFIG`:
```python
'relax': dict(enabled=True),
```

---

## Component Details

### 1. Forcefield Building

**SPFF path** (covalent, sp3 + pi):
```
AtomicSystem → SPFF.toSPFFsp3_loc(sys) → SPFF arrays (apos, REQs, neighs, bLs, bKs, ...)
             → MolecularDynamics.realloc(natoms, nnode, ...)
             → MolecularDynamics packs arrays to GPU buffers
```

**UFF path** (bonds, angles, dihedrals, inversions):
```
AtomicSystem → UFF_Builder(sys) → builder.build() → builder.get_arrays()
             → MolecularDynamics.realloc(...) → pack UFF arrays to GPU
```

**Key consideration**: `SPFF.toSPFFsp3_loc()` handles atom reordering (`_ensure_node_first`), pi-orbital propagation, exclusion lists. This is complex but already working. The controller just orchestrates.

**Rebuild on topology change**: If user adds/removes atoms or bonds, FF must be rebuilt. Detect via `sig_geometry_changed` signal (already emitted from GUI).

### 2. Relaxation Loops

**Batch mode** (one-shot N steps):
```python
E = controller.relax_n(nsteps=100, dt=0.01, damp=0.95)
pos = controller.get_positions()
forces = controller.get_forces()
scene.update_positions(pos)
scene.set_data(pos, forces=forces, force_scale=scale)
```

**Interactive mode** (per-frame, ~30ms QTimer):
```python
# Each tick:
controller.relax_step()
pos = controller.get_positions()
forces = controller.get_forces()
scene.update_positions(pos)
# Optionally update forces less frequently (every 5th frame) for perf
```

**Performance**: GPU step is sub-millisecond for small molecules. Position download is the bottleneck (~0.1ms for 100 atoms). `update_positions()` calls `_redraw()` which rebuilds Vispy markers — this is the main cost. For 100 atoms this is fast enough for 30fps interactive.

**Optimization for later**: Skip force vector display in interactive mode (forces change rapidly, visually noisy). Only show positions. Or update forces every Nth frame.

### 3. Constraint System — Atom Pinning

**What exists**:
- `AtomScene._fixed` boolean mask — prevents picking/dragging of fixed atoms
- `AtomScene.set_fixed_mask()`, `toggle_fixed()`, `is_fixed()` — API ready
- `MolecularDynamics.kernel_params['mask']` — int4, not per-atom

**What's needed**:

**Option A (recommended): Per-atom force mask on GPU**
- Add a `pinned` buffer (int32 array, 1=pinned, 0=free) to `MolecularDynamics`
- Upload to GPU alongside other arrays
- Modify `updateAtomsSPFFf4` kernel: if `pinned[i]`, skip force/velocity update (position stays)
- **Pros**: Clean, zero CPU overhead, positions never move on GPU
- **Cons**: Requires kernel modification in `SPFF.cl` / `Forces.cl`

**Option B (quick, no kernel change): Post-step position restore on CPU**
- After each `run_step_basic()`, download positions, restore pinned atoms to saved positions, re-upload
- **Pros**: No kernel changes, works immediately
- **Cons**: Extra GPU↔CPU transfers per step, breaks interactive performance

**Option C (hybrid): Zero forces on pinned atoms before integrator**
- After `getSPFFf4` (force evaluation), run a small kernel that zeros forces on pinned atoms
- The `updateAtomsSPFFf4` integrator then naturally doesn't move them
- **Pros**: Minimal kernel change (new tiny kernel), clean separation
- **Cons**: One extra kernel launch per step (negligible)

**Recommended**: Option A for production, Option C as quick start. Option B only for prototyping.

**Pin/Unpin UX**:
- Edit mode "Pin/Unpin": left-click atom toggles pin state
- Button "Pin Selected": pins all atoms in `scene._selected_ids`
- Button "Unpin All": clears all pins
- Visual: pinned atoms get distinct color (e.g., red outline or different marker)
- `scene._fixed` already prevents dragging pinned atoms — good

### 4. Interactive Visualization Feedback

**Data flow per frame**:
```
GPU step (run_step_basic)
  → get_positions()  [GPU→CPU, ~0.1ms for 100 atoms]
  → scene.update_positions(pos)  [Vispy redraw, ~1-5ms]
  → canvas.update()  [GPU render]
```

**Force visualization**:
- `scene.set_data(forces=forces, force_scale=0.1)` already supported
- `force_lines` visual draws red lines from each atom along force vector
- Toggle via view mode "Show Forces"
- In interactive mode, updating forces every frame may be visually noisy — consider:
  - Show forces only when not relaxing (static display)
  - Or show every Nth frame
  - Or show magnitude as atom color instead of vectors

**Sync back to backend**:
- After relaxation, update `backend.sys.apos` with relaxed positions
- Emit `sig_geometry_changed` to trigger bond re-evaluation
- Push undo state before relaxation starts

### 5. Edit Mode: Pin/Unpin

Register as edit mode in `UIComponents.edit_modes`:
```python
edit_modes = [('Pin/Unpin', lambda: window.set_edit_mode('pin_unpin'))]
```

In `SPAMMM_GUI.set_edit_mode()`, when mode is `'pin_unpin'`:
- `handle_click()` should toggle `scene.toggle_fixed(idx)` instead of adding atoms
- Pinned atoms get visual indicator (color change or marker style)
- Controller's pinned mask updated: `controller.set_pinned(scene._fixed)`

**Keyboard shortcut**: 'P' key to toggle pin on hovered/selected atom (add to `on_key_press`).

---

## Implementation Phases

### Phase 1: Basic Relaxation Extension (batch mode)
1. Create `RelaxationExtension.py` with `build_ui()`
2. Create `RelaxationController.py` with `build_ff()`, `relax_n()`, `get_state()`
3. Register in `ExtensionManager.py`
4. Panel: FF type, Build, Relax N steps, Energy display
5. After relaxation: update `backend.sys.apos`, `refresh_view()`
6. No constraints yet

### Phase 2: Atom Pinning
1. Add `set_pinned(mask)` to `RelaxationController`
2. Implement Option C (zero-force kernel for pinned atoms) or Option A (kernel mod)
3. Add "Pin/Unpin" edit mode to GUI
4. Add "Pin Selected" / "Unpin All" buttons
5. Visual indicator for pinned atoms
6. Keyboard shortcut 'P'

### Phase 3: Interactive Relaxation
1. Add QTimer-driven loop in extension panel
2. Per-frame: `relax_step()` → `update_positions()` → `canvas.update()`
3. Optional force vector display
4. Start/stop controls
5. Performance tuning (frame rate, force update frequency)

### Phase 4: Polish
1. Auto-rebuild FF on topology change (`sig_geometry_changed`)
2. Undo support (push state before relaxation)
3. Convergence indicator (force norm / energy change)
4. Multiple FF presets (SPFF, UFF, combined SPFF+UFF)
5. Save/load relaxed geometry

---

## Key Files to Modify

| File | Change |
|------|--------|
| `spammm/GUI/ExtensionManager.py` | Add `'relax'` to `EXTENSION_REGISTRY` + `DEFAULT_CONFIG` |
| `spammm/GUI/RelaxationExtension.py` | **New** — `build_ui()`, panel, edit/view modes |
| `spammm/forcefields/RelaxationController.py` | **New** — FF build + MD orchestration + constraints |
| `spammm/GUI/SPAMMM_GUI.py` | Add `'pin_unpin'` handling in `set_edit_mode()` / `handle_click()` |
| `spammm/forcefields/MolecularDynamics.py` | Add `set_pinned()` + pinned buffer upload |
| `kernels/SPFF.cl` or `kernels/Forces.cl` | Add pinned-atom force zeroing (Option A/C) |
| `spammm/GUI/VispyUtils.py` | Visual indicator for pinned atoms (color/marker) |

## Key Files to Reuse (No Changes)

| File | Role |
|------|------|
| `spammm/forcefields/SPFF.py` | FF topology builder — call `toSPFFsp3_loc()` |
| `spammm/forcefields/UFFbuilder.py` | UFF topology builder — call `build()` + `get_arrays()` |
| `spammm/topology/FFparams.py` | Parameter loading — used by SPFF/UFFbuilder |
| `spammm/AtomicSystem.py` | Input data structure |
| `spammm/GUI/VispyUtils.py` | `update_positions()`, `set_fixed_mask()`, `toggle_fixed()`, force display |
