---
type: Report
title: Static-obstacle drag demo — dimer split, frozen molecules, combined PairFF+FAF probe map (session 2026-08-03)
status: delivered — USER confirmed GIF quality and map correctness
tags: [PairFF, FAF, drag, static-obstacle, body-state, mixed-species, NaCl, GUI-script, visualization, VisPy, benzoic-acid-dimer, connected-components]
timestamp: 2026-08-03
related:
  - doc/Tasks/RigidAssembly_StaticMols_PotentialMap.md
  - doc/Reports/PTCDA_DragDemo_StickSlip_2026-08-01.md
  - doc/Tasks/PairFF_MultiBody_Kernel.md
  - doc/Tasks/RigidMoleculePose_SSOT.md
  - doc/TopicalAudit/PairFF_RigidBody.md
skills: [code-reuse, doc-read-navigate, molecular-structure-sync, numerical-parity]
---

# Static-obstacle drag demo — dimer split, frozen molecules, combined PairFF+FAF probe map

**Status:** delivered — USER confirmed the final GIF shows correct bonds, FAF substrate, Pauli repulsion, and static/dynamic toggle.
**Artifacts:** `debug/static_obstacle_drag_demo/static_obstacle_drag_demo.gif`, `.mp4`, `frame_first.png`, `frame_last.png`
**Script:** `demos/gui_scripts/static_obstacle_drag_demo.py`
**Run:** `./run_gui.sh --script demos/gui_scripts/static_obstacle_drag_demo.py`

---

## 1. User goal

Demonstrate dragging a dynamic molecule through a field of static (frozen) obstacles on a NaCl substrate:

1. Load a **benzoic acid dimer** from XYZ and split it into two rigid bodies via **connected components** of the molecular graph.
2. Freeze one body as a **static obstacle** — it keeps its pose exactly but remains a PairFF partner.
3. Drag the other body **toward** the static obstacle so they interact (not away).
4. Show the **combined PairFF(static) + FAF(NaCl) probe map** as a background overlay.
5. Mid-drag, **toggle the static molecule to dynamic and back** to demonstrate the static↔dynamic switching.
6. Produce a GIF/MP4 from captured VisPy canvas frames.

---

## 2. Implementation

### 2.1 Body-state gating (kernels 14 + 15)

- **Kernel 15** (`rigid_body_pairff_multimol_kernel`): Added `body_state` buffer. Static bodies (state=0) skip force/torque integration. Deleted bodies (state<0) are skipped in partner loops.
- **Kernel 14** (`rigid_body_pairff_energy_replica_kernel`): Added `body_state` buffer. Deleted active bodies write zero energy. Deleted partners are skipped in the energy sum.
- **`RigidBodyPairFF`**: `_body_state_host` array + `body_state` OpenCL buffer. `set_body_states`/`set_body_state` methods. `from_molecules` initializes all bodies to dynamic (+1).

### 2.2 Mixed-species assembly

- **`build_mixed_species_assembly`** in `RigidBodyUtils.py`: round-robin body ordering, deterministic interleaved by copy.
- RA panel accepts comma-separated molecule names (e.g. `NTCDI,uracil`).
- **Factorized PLQH per-pack**: `_folded_plqh_all_sites` builds PLQH per-pack from runtime `REQ_base` when the fit's `plqh_override` doesn't match a pack's atom count.

### 2.3 Dimer split via connected components

The demo loads `benzoicacid_dimer.xyz` (30 atoms) into the editor graph, then uses the **"From editor"** source in the RA panel. This calls `graph_to_rigid_fragments` → `AtomicGraph.find_connected_components()` which splits the dimer into 2 rigid bodies (15 atoms each) by BFS over alive bonds.

### 2.4 Combined probe map

`compute_combined_probe_map` in `RigidBodyUtils.py` computes:
```
E_map(x,y; z_probe) = E_PairFF(static molecules) + E_FAF(NaCl)
```
Dynamic molecules are excluded — only frozen (static) bodies contribute to the PairFF part. The map is cached and only recomputed on build/toggle/delete/probe-change/explicit recompute.

### 2.5 GUI controls

- **Shift+LMB**: toggle dynamic↔static on any live molecule
- **RMB**: soft-delete a molecule (rejected on last live body)
- **LMB on static**: no anchor attached (reports "molecule is static")
- **State counts label**: `dyn=N stat=N del=N` in the RA panel
- **Probe controls**: H+/O− presets, R0/E0/Q/z_probe spinboxes, map checkbox, recompute button

---

## 3. Problems and caveats — what went wrong

### 3.1 Scrambled atom types and bonds (graph rebuild bug)

**Symptom:** After building from editor with a dimer, atom colors and bonds were visually scrambled — random colors, wrong bond connections.

**Root cause:** `_ensure_backend_matched` in `RigidAssemblyExtension.py` only checked if the atom *count* matched between the editor graph and the assembly. When loading a dimer from the editor, the graph has atoms in XYZ file order, but `graph_to_rigid_fragments` splits them by BFS order within each connected component. Since both have 30 atoms, the count check passed and `update_positions_from_array` assigned assembly-ordered positions to graph-ordered atoms — scrambling bonds and colors.

**Fix:** `_ensure_backend_matched` now also checks that the enames sequences match. If they differ (same count but different order), the graph is rebuilt from the assembly's atom order with correct bonds from `ra_bonds0`.

**Lesson:** Atom count equality is necessary but not sufficient for graph-assembly synchronization. The enames sequence (or a stable atom-ID mapping) must also match. See `doc/Takeways.md` → "Graph rebuild enames check".

### 3.2 Missing FAF substrate and Pauli repulsion in map

**Symptom:** The combined map was uniformly blue — no NaCl corrugation, no Pauli repulsion from the frozen molecule.

**Root cause:** The "From editor" source path had `faf_enabled = source != 'From editor'` which hardcoded FAF off for editor builds. The demo used "From editor" so FAF was never attached. Additionally, `load_or_fit_faf` was called with `mol_name=None`, causing `None.lower()` AttributeError (silently caught by the try/except in `_on_build`).

**Fix:**
- Removed the `source != 'From editor'` exclusion — FAF is now enabled for editor builds.
- Added FAF fitting for editor fragments: fits on the first fragment's atoms with `mol_name='editor_frag0'` for cache filename.
- Fixed `load_or_fit_faf` call to use a valid string `mol_name`.

### 3.3 White box around static molecule (double overlay)

**Symptom:** A visible white rectangle appeared around the combined map overlay.

**Root cause:** Both the FAF substrate overlay and the combined map overlay were visible simultaneously, creating a visible boundary where their extents differed.

**Fix:** `_recompute_ra_combined_map` now hides the FAF substrate overlay when the combined map is shown, and uses the same `order = -1` z-depth.

### 3.4 Drag direction away from obstacle

**Symptom:** The first version dragged the dynamic molecule outward, away from the static obstacle — they never interacted.

**Fix:** The demo now computes the direction from the dynamic body's CoM to the static body's CoM and drags along that vector.

### 3.5 Warm map recompute time

The warm recompute time for the CPU-based map is ~254ms, exceeding the 0.2s budget. This is not a blocker for the demo (recompute is event-driven, not per-frame), but a GPU map kernel may be needed for interactive use. See task doc §7.

---

## 4. Test coverage

**`tests/test_body_state.py`** (8 tests):
- `test_all_dynamic_matches_default` — kernel-15 state buffer doesn't change dynamics when all dynamic
- `test_frozen_body_does_not_move` — static body pose invariant, velocity/FIRE zero
- `test_static_partner_still_interacts` — dynamic force differs with static vs deleted partner
- `test_deletion_matches_rebuilt_reference` — kernel-15 with deleted body matches rebuilt reference
- `test_kernel14_deletion_matches_rebuilt_reference` — kernel-14 energy with deleted body matches rebuilt reference
- `test_kernel14_all_dynamic_matches_default` — kernel-14 regression
- `test_mixed_species_build_and_faf_step` — 4-species assembly + FAF step is finite
- `test_combined_map_decomposition_and_invalidation` — E_sum == E_pairff + E_faf, invalidation on state/probe change

**`tests/GUI/test_rigid_assembly_extension.py`** (13 tests, 2 new):
- `test_ra_gesture_state_toggle_and_delete` — Shift+LMB toggles, RMB deletes, last-live-body protection
- `test_ra_mixed_species_build` — comma-separated species build + FAF

All 21 tests pass. Run: `pytest tests/test_body_state.py tests/GUI/test_rigid_assembly_extension.py -m "not slow"`

---

## 5. Files changed

| File | Change |
|---|---|
| `kernels/rigid.cl` | `body_state` gate in kernels 14 + 15 |
| `spammm/forcefields/RigidBodyDynamics.py` | `_body_state_host`, `body_state` buffer, `set_body_states`/`set_body_state`, factorized PLQH per-pack |
| `spammm/forcefields/RigidBodyUtils.py` | `build_mixed_species_assembly`, `compute_combined_probe_map` |
| `spammm/GUI/RigidAssemblyExtension.py` | state helpers, gestures, probe map overlay, FAF for editor builds, enames check in graph rebuild |
| `demos/gui_scripts/static_obstacle_drag_demo.py` | new demo script |
| `tests/test_body_state.py` | new test file (8 tests) |
| `tests/GUI/test_rigid_assembly_extension.py` | 2 new tests |

---

## 6. Reproduction

```bash
# Run the demo (produces GIF + MP4 + first/last PNG)
./run_gui.sh --script demos/gui_scripts/static_obstacle_drag_demo.py

# Custom drag distance and relaxation
./run_gui.sh --script demos/gui_scripts/static_obstacle_drag_demo.py -- --drag-x 10 --n-relax 300

# Run tests
pytest tests/test_body_state.py tests/GUI/test_rigid_assembly_extension.py -m "not slow"
```

**Output:** `debug/static_obstacle_drag_demo/static_obstacle_drag_demo.gif` (682×709, ~3.4 MB), `.mp4` (~94 KB)

**Energy trace:** E starts at -1.62 eV (relaxed), rises as the dynamic molecule approaches the static obstacle (Pauli repulsion), drops to -1.68 eV when the static molecule is toggled to dynamic (both relax), then rises again as the drag continues past the obstacle.
