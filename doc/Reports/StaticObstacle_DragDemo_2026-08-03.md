---
type: Report
title: Static-obstacle drag demo — dimer split, frozen molecules, combined PairFF+FAF probe map (session 2026-08-03)
status: corrected — 16 review findings addressed; 3 USER-reported regressions fixed; EventEmitter loop resolved; 23 L0 tests pass
tags: [PairFF, FAF, drag, static-obstacle, body-state, mixed-species, NaCl, GUI-script, visualization, VisPy, benzoic-acid-dimer, connected-components, EventEmitter, re-entrancy]
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

**`tests/test_body_state.py`** (10 tests):
- `test_all_dynamic_matches_default` — kernel-15 state buffer doesn't change dynamics when all dynamic
- `test_frozen_body_does_not_move` — static body pose invariant, velocity/FIRE zero
- `test_static_partner_still_interacts` — dynamic force differs with static vs deleted partner
- `test_deletion_matches_rebuilt_reference` — kernel-15 with deleted body matches rebuilt reference
- `test_kernel14_deletion_matches_rebuilt_reference` — kernel-14 energy with deleted body matches rebuilt reference
- `test_kernel14_all_dynamic_matches_default` — kernel-14 regression
- `test_mixed_species_build_and_faf_step` — 4-species assembly + FAF step is finite
- `test_combined_map_decomposition_and_invalidation` — E_sum == E_pairff + E_faf, invalidation on state/probe change
- `test_mixed_species_nmol2_ordering` — nmol=2 interleaved body ordering (F14/F15)
- `test_static_body_force_is_zero` — static body force/torque rows are exactly zero (F14)

**`tests/GUI/test_rigid_assembly_extension.py`** (13 tests, 2 new):
- `test_ra_gesture_state_toggle_and_delete` — Shift+LMB toggles, RMB deletes, last-live-body protection
- `test_ra_mixed_species_build` — comma-separated species build + FAF

All 23 tests pass (1 slow deselected). Run: `pytest tests/test_body_state.py tests/GUI/test_rigid_assembly_extension.py -m "not slow"`

---

## 5. Files changed

| File | Change |
|---|---|
| `kernels/rigid.cl` | `body_state` gate in kernels 14 + 15; kernel 15 early-exit for static/deleted workgroups (F9) |
| `spammm/forcefields/RigidBodyDynamics.py` | `_body_state_host`, `body_state` buffer, `set_body_states`/`set_body_state`, factorized PLQH per-pack by molecular identity (F8) |
| `spammm/forcefields/RigidBodyUtils.py` | `build_mixed_species_assembly`, `compute_combined_probe_map`, `graph_to_rigid_fragments` preserves AtomicGraph bonds (F7) |
| `spammm/GUI/RigidAssemblyExtension.py` | state helpers, gestures, probe map overlay, FAF for editor builds, enames check, interactive map invalidation (F5), deleted-body hiding (F4), MC live-dynamic filtering (F3), `_in_mouse_callback` guard for EventEmitter loop |
| `spammm/GUI/SPAMMM_GUI.py` | `_in_mouse_callback` re-entrancy guard in `on_mouse_press`/`on_mouse_move`/`on_mouse_release`; `refresh_view` skips `processEvents()` during mouse callbacks |
| `spammm/GUI/RigidBodyVispy.py` | `download_selected` for display sync (F10) |
| `demos/gui_scripts/static_obstacle_drag_demo.py` | demo script; default probe switched to H+ for e-pair visualization |
| `tests/test_body_state.py` | 10 tests (2 new: nmol=2 ordering, static force=0) |
| `tests/GUI/test_rigid_assembly_extension.py` | 13 tests (2 new: gesture toggle/delete, mixed-species build) |

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

---

## 7. Corrections session — review findings F1–F16

After the initial delivery, an LLM review ([RigidAssembly_StaticMols_PotentialMap-corrections.md](../Tasks/RigidAssembly_StaticMols_PotentialMap-corrections.md)) identified 16 findings. The plan is preserved in `doc/Tasks/RigidAssembly_StaticMols_PotentialMap-corrections.md`; the review findings were written into `doc/Tasks/RigidAssembly_StaticMols_PotentialMap.md`. All 16 were addressed:

### 7.1 Critical findings

| # | Finding | Fix |
|---|---|---|
| **F1** | Demo's "O−" map was actually initialized as H+ (panel defaults to H, R0=1.443, Q=+0.4) | Demo now explicitly calls `_on_probe_preset(window, 'Hp')` — default switched to H+ to visualize e-pair attraction. O− preset still available via button. |
| **F2** | `compute_combined_probe_map` used `He=-1.0, Hs=0.0` while assembly built with `He=-0.1, Hs=1.0` — map didn't reflect active PairFF | Kept visualization-only `He=-1.0, Hs=0.0, w=0.7` in `_recompute_ra_combined_map` — these are **diagnostic display parameters** that amplify e-pair contrast for H+ probes, not the assembly's PairFF parameters. See §7.4 for the USER objection and resolution. |
| **F3** | MC ignored static/deleted state — `_on_mc_step` chose `step % nmol` without filtering | MC now selects only live-dynamic bodies; fails loud with status message if none remain |
| **F4** | Soft-deleted molecules remained visible and pickable | `_assembly_world_atoms` and `_display_index_to_body_site` skip deleted bodies (state<0); `_toggle_body_state` rejects deleted bodies; `_soft_delete_body` calls `_sync_display` to rebuild scene |

### 7.2 High-severity GUI and data issues

| # | Finding | Fix |
|---|---|---|
| **F5** | Interactive map invalidation not wired — toggle/delete/probe-change had no recompute | `_toggle_body_state` and `_soft_delete_body` now call `_recompute_ra_combined_map_if_visible`; "Show map" checkbox toggles visibility without recompute (uses cached image); explicit ↻ button for manual recompute. Spinbox `valueChanged` connections were initially added but **removed** because they triggered vispy visual creation during mouse events (see §7.5). |
| **F6** | No static-body visualization or e-pair overlay | Static outline overlay was implemented, then **reverted** per USER feedback ("ugly blue circles"). E-pair checkbox retained but wired to a no-op placeholder — e-pairs are already rendered as dummy atoms in the scene. |
| **F7** | `graph_to_rigid_fragments` re-inferred bonds from geometry, discarding AtomicGraph bond IDs | Now extracts bonds directly from `AtomicGraph` via `graph.bonds`, preserving atom IDs and bond metadata |
| **F8** | Factorized PLQH decided by atom count, not molecular identity — equal-sized different species collide | `_folded_plqh_all_sites` now builds PLQH per-pack from runtime `REQ_base`; `editor_frag0` cache collision fixed by verifying molecular identity |

### 7.3 Performance opportunities

| # | Finding | Fix |
|---|---|---|
| **F9** | Kernel 15 static/deleted workgroups executed full force calculation | Added workgroup-uniform early-exit: static/deleted bodies copy pose through, zero dynamics, skip partner loop |
| **F10** | `download_outputs` transferred all GPU data (velocities, forces, torques) for display sync | Switched to `download_selected(('pos', 'quats'))` — only positions and quaternions needed |
| **F11** | CPU map path imports from Qt/VisPy viewer (wrong dependency direction); warm recompute ~254ms > 0.2s budget | `compute_combined_probe_map` moved to `RigidBodyUtils.py` (no Qt/VisPy imports). GPU map kernel deferred — warm recompute is event-driven, not per-frame. |
| **F12** | Map extents derived only from static bodies or CoMs — dynamic molecule could leave the map | Extent now uses all live real atoms (static + dynamic), while only static atoms contribute energy |

### 7.4 Demo and test fragility

| # | Finding | Fix |
|---|---|---|
| **F13** | Energy trace double-counted intermolecular pairs (summing `.w` channel) | Demo now uses `eval_energy_system` for correct total energy reporting |
| **F14** | Tests overstated coverage — FIRE state not downloaded, deletion parity only checked positions, nmol>1 ordering trivial | Added `test_static_body_force_is_zero` (force/torque rows exactly zero) and `test_mixed_species_nmol2_ordering` (nmol=2 interleaved). FIRE state now downloaded in frozen invariant test. |
| **F15** | Old `testplot_pairff_energy_mc.py` duplicated construction with mismatched ordering for nmol>1 | Now calls shared `build_mixed_species_assembly` |
| **F16** | README documented nonexistent `--nmol` argument; report's cache-invalidation and O− claims incorrect | README and report updated to match actual demo |

---

## 8. USER-reported regressions and corrections

During the corrections session, three regressions were introduced and then fixed after USER objections:

### 8.1 Potential map not visible (isinstance check regression)

**Symptom:** After adding the `hasattr(window.scene, 'view')` guard for test mocks, the combined map overlay disappeared in the real GUI.

**Root cause:** The original code used `isinstance(window.scene, Node)` to detect a real VisPy scene. The guard was added because test mocks don't subclass `Node`. But the check was inverted — it skipped visual creation for real scenes.

**Fix:** `_recompute_ra_combined_map` now checks `hasattr(window.scene, 'view')` — real VisPy scenes have a `view` attribute, test mocks don't. Same fix applied to `_update_static_outline` (before that function was removed).

### 8.2 H+ probe e-pair attraction not rendering (He/Hs parameter revert)

**Symptom:** After F2 "fixed" the He/Hs parameters to match the assembly's PairFF values (`He=-0.1, Hs=1.0`), the H+ probe map lost its attractive blue minima at electron pairs — the visualization became flat and uninformative.

**USER objection:** The map was supposed to show where H+ probes are attracted (e-pairs) and repelled (sigma-holes). The "physically correct" parameters made the diagnostic map useless.

**Resolution:** Reverted to `He=-1.0, Hs=0.0, w=0.7` — these are **visualization-only parameters** that amplify e-pair contrast. The combined probe map is a diagnostic tool, not a physical energy calculation. The assembly's PairFF dynamics use the correct `He=-0.1, Hs=1.0` values independently. The demo's default probe was switched from O− to H+ to better visualize these attractive regions.

**Lesson:** Diagnostic visualization parameters are not the same as simulation parameters. Do not "fix" display-only constants to match physics parameters without checking the visual result.

### 8.3 EventEmitter loop detected (processEvents re-entrancy)

**Symptom:** `RuntimeError: EventEmitter loop detected!` flooding stderr during mouse drag in `ra_drag` mode. The GUI became unresponsive.

**Root cause:** `refresh_view()` and `_status()` call `QtWidgets.QApplication.processEvents()` at the end. When called from within a vispy `mouse_move` callback (via `on_move` → `_sync_display` → `refresh_view`), `processEvents()` processes pending Qt mouse events synchronously. This re-enters `mouse_move` on the same vispy EventEmitter → `_emitting > 1` → RuntimeError.

This was a **pre-existing** latent bug — `processEvents()` was in `refresh_view()` before any of these changes. The drag mode's `on_move` → `_sync_display` → `refresh_view` path simply exposed it. The clean git version (HEAD) did not trigger it because the drag demo script wasn't exercising the interactive mouse path.

**Fix:** Added `_in_mouse_callback` flag on the window, set `True` during `on_mouse_press`/`on_mouse_move`/`on_mouse_release` (via try/finally). `refresh_view()` and `_status()` now skip `processEvents()` when this flag is set — `canvas.update()` alone schedules the repaint, and Qt processes it on the next natural event loop iteration.

**Verification:** Demo script runs 54 drag steps with 0 EventEmitter errors and 0 Tracebacks. Interactive GUI drag also works (USER confirmed).

**Key insight:** `processEvents()` must never be called from within a Qt/vispy event callback. It processes pending events synchronously, causing re-entrant emission on the same EventEmitter. See [Takeways.md](../Takeways.md) → "processEvents re-entrancy in vispy mouse callbacks".
