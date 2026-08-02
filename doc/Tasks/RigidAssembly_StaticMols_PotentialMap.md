---
type: Task
title: Static rigid molecules and combined PairFF+FAF 2D probe map in SPAMMM_GUI
status: implemented — USER confirmed GIF quality and map correctness (2026-08-03); L0 tests pass; warm map recompute ~254ms exceeds 0.2s budget (GPU kernel deferred)
tags: [GUI, rigid-body, PairFF, FAF, OpenCL, visualization, drag, mixed-species]
timestamp: 2026-08-03
related: [MultiMol_FAF_ConcurrentRelax.md, PairFF_FAF_Substrate.md, PairFF_MultiBody_Kernel.md, PairFF_MapDisplay_SSOT.md, RigidMoleculePose_SSOT.md, ../Reports/StaticObstacle_DragDemo_2026-08-03.md]
skills: [code-reuse, doc-read-navigate, gpu-optimize, running-tests, numerical-parity]
---

# Static rigid molecules and combined PairFF+FAF probe map

## 0. Required result

Extend the existing **Rigid Assembly** workflow inside the main `SPAMMM_GUI`; do not
create a second VisPy window or a parallel forcefield implementation.

The result must provide:

1. Several rigid molecules in one scene, including **different species in the same
   assembly** (initial demonstration: NTCDI, TBTAP, uracil, and benzoic acid).
2. Per-molecule **dynamic/static** state. Static molecules retain their pose exactly but
   remain PairFF partners of dynamic molecules.
3. Existing concurrent behavior among dynamic molecules: dragging one dynamic molecule
   lets the other dynamic molecules respond.
4. A cached background field for the selected test probe:

   \[
   E_\mathrm{map}(x,y;z_\mathrm{probe}) =
   E_\mathrm{PairFF}^{\mathrm{static\ molecules}} +
   E_\mathrm{FAF}^{\mathrm{NaCl}}
   \]

   Dynamic molecules are intentionally excluded from the map so it does not need to be
   recomputed during every drag frame.
5. The **O− probe preset as the default**, with the same editable probe parameters and
   display scaling used by `demo_pairff.py`.
6. Optional e-pair and sigma-hole display.
7. A thin GUI script demo derived from `demos/gui_scripts/ptcda_drag_demo.py`.

The implementation remains **unverified** until L0 numerical tests pass and the USER
reviews the L2 GUI capture. Do not mark this task fixed/resolved/done before that review.

---

## 1. Exact GUI behavior

### 1.1 Build a mixed assembly

Keep the existing compact Build section. Make the existing molecule combo editable and
accept either one name or a comma-separated list:

```text
PTCDA
NTCDI,TBTAP,uracil,benzoic_acid
```

`nmol` means **copies per listed species**. With `nmol=1`, the second example creates four
bodies. With `nmol=2`, it creates eight. Body order must be deterministic and
round-robin/interleaved:

```text
copy 0: NTCDI, TBTAP, uracil, benzoic_acid
copy 1: NTCDI, TBTAP, uracil, benzoic_acid
```

The molecule tuple, `tid`, bonds, FAF parameters, and display atom range must use this
same order. Do not copy the current mixed-species test harness ordering blindly: its
`molecules` list is grouped by species while its `bonds0` construction cycles species.

Reuse:

- `MOL_PATHS` and `load_molecule`; add the already existing
  `data/xyz/benzoic_acid.xyz` entry.
- `RigidBodyPairFF.from_molecules`, which already accepts a different
  `(apos, enames, REQs)` tuple for every body.
- `grid_pos` and the current deterministic jitter/quaternion initialization.
- `RigidEnsemble.from_poses` as the pose authority.

Do not add an "Add molecule" sub-dialog or a second scene builder for the first version.

### 1.2 Mouse controls in the existing `RA Drag` mode

Plain LMB is already needed for dragging, so it must not also toggle state.

| Gesture | Required action |
|---|---|
| **LMB drag** on a dynamic molecule | Existing anchor-spring drag; all dynamic bodies may relax |
| **LMB** on a static molecule | Do not attach an anchor; report that the molecule is static |
| **Shift+LMB click** on any live molecule | Toggle dynamic ↔ static |
| **RMB click** on a molecule | Soft-delete it from rendering and all physics interactions |

State changes operate on the whole rigid body selected through any of its real atoms.
Before dynamic → static, synchronize the latest GPU pose into `RigidEnsemble`; that exact
pose becomes the frozen pose. Both state transitions clear that body's velocity and FIRE
state so stale momentum cannot reappear.

If no dynamic bodies remain, the scene is valid but drag/MC must fail loud with a useful
status message. RMB deletion of the last live body must be rejected.

### 1.3 Visual state

- Preserve normal element/charge colors.
- Draw a dedicated outline overlay on real atoms of static molecules (neutral cool-gray
  or light-blue edge, transparent face). This is preferable to recoloring atoms because
  it survives ordinary `AtomScene` refreshes and preserves chemical identity.
- Hide deleted molecules, their bonds, dummy sites, and map contribution.
- The status line reports counts, for example:
  `dynamic=2 static=2 deleted=0`.

The state outline belongs to `RigidAssemblyExtension`; do not add rigid-body policy to the
generic `AtomScene`.

### 1.4 Map and probe controls

Add one tight "Probe map" row/group to the Rigid Assembly panel, following the existing
layout policy (`apply_tight`, `AutoGridPlacer`, bounded widget widths):

- H+ and O− preset buttons.
- Element combo to fill R0/E0.
- Editable R0, E0, and Q.
- Fixed `z_probe` relative to `Z_SURF_TOP` (default `3.0 Å`).
- `map` checkbox.
- `e-pairs` checkbox.
- Explicit `↻` recompute button.

Default: **O−**, using the O atom R0/E0 values and `Q=-0.4 e`.

`z_probe` must be independent of the moving active-body CoM. Following a dynamic body's
height would invalidate the cached map during every frame.

### 1.5 Map invalidation contract

The map is recomputed only when its inputs change:

- assembly build;
- dynamic/static toggle;
- molecule deletion;
- probe R0/E0/Q or preset change;
- `z_probe`, grid extent, or grid step change;
- explicit `↻`.

The map is **not** recomputed on:

- mouse move;
- relaxation/MD frame;
- dynamic-body pose update;
- ordinary scene redraw;
- drag release, unless another map input changed.

Hide/show must reuse the cached image; checking the map box after a pure hide must not
evaluate the field again.

---

## 2. Confirmed implementation inventory

### 2.1 Main GUI rigid assembly

`spammm/GUI/RigidAssemblyExtension.py` already provides:

- `RigidEnsemble` pose ownership and checked GPU/host synchronization.
- single-species `_on_build`;
- `_display_index_to_body_site` for real-display atom → PairFF site mapping;
- `_set_anchors`, `_make_ramdrag_mode`, and `run_multimol_md`;
- `_sync_ensemble_from_gpu` and `_sync_display`;
- a local pattern for lazily created VisPy overlays (`ra_anchor_line/marker`);
- FAF-only `_update_ra_substrate_overlay`.

The current drag path uses kernel 15, so all bodies integrate concurrently.
`set_active_body(body)` called by the handler only selects the legacy active view; it is
not a per-body frozen-state API.

### 2.2 Forcefield and GPU paths

`spammm/forcefields/RigidBodyDynamics.py` already provides:

- mixed per-body packs in `RigidBodyPairFF.from_molecules`;
- flat offsets `mols`, body poses, REQ/type arrays, anchors, and ping-pong state;
- `run_multimol_md(..., faf=True)` using
  `rigid_body_pairff_multimol_kernel` (kernel 15);
- `attach_pairff_faf` and tensor-materialized FAF coefficients;
- `world_sites_all_bodies` and `_body_sites_world`;
- `_folded_types_all_sites`, including concatenated typed IDs.

`kernels/rigid.cl` kernel 15 launches one workgroup per body. Every workgroup gathers
PairFF forces from all partner bodies, adds FAF/anchors, then integrates its own pose.
There is currently no per-body integration gate.

Kernel 14 (`rigid_body_pairff_energy_replica_kernel`) is the MC/GA energy evaluator.
Deletion must also be respected there, otherwise a visually deleted body remains in MC
energies.

### 2.3 Existing high-level state metadata

`RigidEnsemble.RigidBody` already has:

- `active`: currently unused outside metadata;
- `alive`: currently unused outside metadata;
- stable body `id`.

Use these as the high-level source of truth:

| `alive` | `active` | Meaning |
|---:|---:|---|
| 1 | 1 | dynamic, rendered, interacts |
| 1 | 0 | static, rendered, interacts, contributes to map |
| 0 | any | deleted, hidden, excluded from interactions/map |

Do not create an independent GUI `frozen_mask` that can disagree with the ensemble.
`active_body` remains the selected-body index for legacy sequential/view paths and must
not be overloaded with this meaning.

### 2.4 Existing map and display path

`spammm/GUI/RigidBodyVispy.py` already has:

- `compute_potential_map_unified`;
- combined `_recompute_map`;
- H+/O− probe controls;
- `potential_to_rgba`, the established display SSOT;
- cyan e-pair / magenta sigma-hole markers and dummy bonds.

`spammm/surfaces/FoldedRigid.py` already has:

- `eval_folded_potential_grid`;
- factorized probe evaluation through `atom_REQH`;
- `faf_type_idx_for_probe` for typed fits.

`spammm/GUI/VispyUtils.py` already has the main scene and the image-transform code in
`update_faf_map_overlay`.

Important corrections to the old draft:

1. Both existing map components are **CPU NumPy**, not GPU.
2. `compute_potential_map_unified` intentionally omits real-atom Coulomb. The standalone
   demo therefore shows a diagnostic compact-exp/dummy-site PairFF map, not the complete
   kernel-15 energy.
3. The current main-GUI factorized FAF overlay requests `component='coulomb_phi'`; it is
   not the same total O− probe energy used by `RigidBodyVispy._recompute_map`.
4. Importing compute from a viewer into generic visualization would preserve the current
   backwards dependency. Shared compute should live outside the viewer.

### 2.5 Existing mixed-species FAF path and its caveat

The factorized NaCl `coeffs4`, basis, and lattice arrays in the cached NTCDI, TBTAP,
uracil, and PTCDA fits are identical: they describe the substrate. Per-atom PLQH is the
molecule-specific part.

For mixed species, do not apply the first fit's `atom_plqh` array to every body. Current
`_folded_plqh_all_sites(..., plqh_override=...)` assumes that override matches each pack;
it fails for different atom counts and is wrong for equal-sized different species.
Build PLQH per pack from that pack's runtime `REQ_base` (or support one explicitly
concatenated per-body override) and then materialize the existing shared NaCl basis.

Typed FAF remains supported by concatenating `remap_fit_for_molecule(...).atom_type_ids`
in exact body order, but factorized FAF is the default for this mixed-species GUI demo.

---

## 3. Proposed architecture

### 3.1 One derived GPU body-state buffer

Add one `int32[n_mols]` RBD mirror derived from `RigidEnsemble`:

```text
+1 dynamic
 0 static
-1 deleted
```

Host API:

- `set_body_states(states)` validates length/values, uploads into a persistent buffer,
  clears velocities/FIRE for non-dynamic bodies, and does not reallocate molecule data.
- `set_body_state(body, state)` is the one-body convenience wrapper.
- Rigid Assembly always updates ensemble metadata first, then derives/uploads the buffer.

Defaults are all `+1`, preserving every existing caller.

Do not add this buffer to kernels 12/13 in the first implementation. The requested main
GUI uses kernel 15; changing legacy sequential kernels expands regression risk without
providing this feature.

### 3.2 Kernel 15 behavior

Add `body_state` to the existing kernel; do not create a copied kernel.

- A dynamic body's workgroup executes the current force, FAF, anchor, FIRE/Euler, and
  write-back path unchanged.
- A static or deleted body's workgroup copies pose through unchanged and writes zero
  velocities/FIRE state. The branch is uniform for the entire workgroup.
- In a dynamic body's partner loop:
  - dynamic and static partners are included;
  - deleted partners are skipped.

Thus static-static work is not recomputed, while every dynamic body still gathers the
force from every live static/dynamic partner. Frozen reaction forces are not required for
the first GUI version; if later needed for diagnostics, add a separate opt-in reduction
rather than paying for it every drag step.

Changing values in the persistent state buffer must not invalidate cached launchers; the
buffer object is unchanged.

### 3.3 Kernel 14 / MC behavior

- Round-robin MC chooses only live dynamic bodies as `moved`.
- Static bodies remain frozen partners in the evaluated energy.
- Deleted bodies are skipped in active and partner energy terms.
- With an all-dynamic/all-live state vector, kernel-14 output must match the pre-change
  path within the existing float32 tolerance.

If body deletion is deferred from MC, the GUI must disable MC after deletion and say so
explicitly. Silent ghost interactions are not acceptable.

### 3.4 Shared mixed-species assembly helper

Generalize the existing mixed-species construction pattern into
`spammm/forcefields/RigidBodyUtils.py`; both the GUI extension and
`tests/testplot_pairff_energy_mc.py` should call it. It returns aligned:

```text
molecules, tids, bonds_list, species records
```

Do not maintain a second loader/parser inside the demo script. The demo only sets GUI
controls and invokes extension workhorses.

### 3.5 Shared headless map compute

Move/generalize the existing PairFF map workhorse into the existing
`RigidBodyUtils.py` (no new module):

- input: explicit static world sites/REQ/types, probe REQ, grid, optional FAF fit;
- output: raw `E_pairff`, raw `E_faf`, raw sum, axes, and extent;
- no Qt/VisPy imports;
- retain a compatibility import/wrapper in `RigidBodyVispy.py` so existing code does not
  break.

For factorized FAF, pass the probe's full
`[R0, sqrt(E0), Q, 0]` through `atom_REQH`; do not show only `coulomb_phi`.

The initial map must preserve the established `demo_pairff.py` PairFF semantics,
including its current omission of real-atom Coulomb, so the main GUI and standalone
viewer agree. Label/document it as a **diagnostic probe map**. Adding Coulomb is a
separate parity/display decision because it changes the USER-approved appearance.

In `VispyUtils.py`, factor only the generic "raw 2D array + extent → cached VisPy Image"
operation out of `update_faf_map_overlay`. Both FAF-only and combined callers then reuse
one transform/z-order implementation and `potential_to_rgba`.

### 3.6 CPU now, measured GPU decision

The old draft's claim that the CPU path is always sub-second was not established. A local
representative measurement for 104 mixed sites on a `299×281` grid gave:

- first call about 1.55 s (dominated by the dynamic import of `fit_radial.py`);
- warm calls about 0.15 s for the PairFF component.

Therefore:

1. First remove repeated import/setup overhead and use the existing vectorized CPU path.
2. Recompute only on the invalidation events in §1.5.
3. Measure total PairFF+FAF recompute time on the target NVIDIA workstation.
4. If warm recompute exceeds the agreed interaction budget (proposed: 0.2 s for the
   default scene/grid), implement a follow-up GPU map evaluator: one work-item per pixel,
   reading cached frozen world sites and reusing `compact_exp_pair_EF` plus the folded
   basis helpers. Do not port the NumPy loops speculatively.

This answers “is the map recomputed on GPU?” precisely: **not today, and not in the
minimal first implementation unless the measured event-driven CPU path misses the
budget**.

### 3.7 E-pair and sigma-hole overlay

Keep this overlay local to `RigidAssemblyExtension`, like the existing anchor visuals.

- Cyan marker: type 1 e-pair.
- Magenta marker: type 2 sigma-hole.
- Optional faint segment to the associated real host atom.
- Build/cache dummy→host indices once per pack; do not run a nearest-real nested loop on
  every frame.
- Update world positions from the already synchronized ensemble poses; do not add another
  GPU download.
- Hide the visuals when the checkbox is off or the body is deleted.

Move/share the e-pair/sigma visual constants/helper from `RigidBodyVispy` rather than
copying color and size formulas.

---

## 4. Delivery phases

### Phase A — state semantics and numerical guards

1. Add failing L0 tests for all-dynamic parity, frozen invariance, live-static
   interaction, and deleted-body exclusion.
2. Add `RigidEnsemble` metadata → RBD state synchronization.
3. Gate kernel 15 and kernel 14 as specified.
4. Run the focused GPU tests on an NVIDIA device.

### Phase B — mixed species and FAF

1. Add the shared mixed-species assembly helper.
2. Add `benzoic_acid` to the existing path table.
3. Make the existing combo accept comma-separated species.
4. Correct factorized PLQH construction per pack.
5. Build and run one finite kernel-15+FAF step for
   NTCDI+TBTAP+uracil+benzoic_acid.

### Phase C — interaction and visualization

1. Add Shift+LMB toggle and RMB soft delete to `RA Drag`.
2. Add static outlines and state counts.
3. Extract/reuse shared probe controls and headless map composition.
4. Add the combined cached map.
5. Add the e-pair/sigma-hole overlay.

### Phase D — thin GUI demo

Create the explicitly requested
`demos/gui_scripts/static_obstacle_drag_demo.py`, reusing the pacing/capture and workhorse
calls from `ptcda_drag_demo.py`:

1. Build one each of NTCDI, TBTAP, uracil, and benzoic acid on NaCl.
2. Leave one selected molecule dynamic and freeze the obstacles.
3. Enable O− combined map and e-pairs.
4. Drag the dynamic molecule with the existing anchor/FIRE path.
5. Print unbuffered progress: species/states, map timing/range, accepted drag progress,
   static maximum displacement, and final paths.
6. Produce first/last PNG plus GIF (MP4 optional).

Run:

```bash
./run_gui.sh --script demos/gui_scripts/static_obstacle_drag_demo.py
```

The script contains orchestration only; state, map, build, and visualization logic stays
in existing shared modules/extension code.

---

## 5. Expected change footprint

| File | Surgical responsibility |
|---|---|
| `kernels/rigid.cl` | body-state gate in kernels 15 and 14; no copied kernels |
| `spammm/forcefields/RigidEnsemble.py` | clarify `active`/`alive` semantics; no new pose store |
| `spammm/forcefields/RigidBodyDynamics.py` | persistent body-state mirror/API; mixed factorized PLQH correction |
| `spammm/forcefields/RigidBodyUtils.py` | shared mixed-species builder and headless map composition |
| `spammm/GUI/VispyUtils.py` | reusable raw-array image overlay and shared probe/dummy visual helpers |
| `spammm/GUI/RigidBodyVispy.py` | consume shared map/probe helpers while preserving public behavior |
| `spammm/GUI/RigidAssemblyExtension.py` | tight controls, gestures, state/display sync, combined-map/e-pair orchestration |
| `demos/gui_scripts/static_obstacle_drag_demo.py` | thin requested demo |
| `tests/GUI/test_rigid_assembly_extension.py` | GUI/state/map/mixed-species L0 tests |
| existing PairFF GPU test file | kernel 14/15 parity and invariants |
| `user_guide/RigidAssembly_GUI.md`, topical audit | update only after implementation/testing; status remains unverified pending USER review |

No force law is duplicated. No new pose authority, VisPy window, general-purpose module,
or per-frame map computation is introduced.

---

## 6. Verification and acceptance

### L0 — automatic

1. **All-dynamic regression:** three-body kernel-15 state after one exact (`batch=1`) step
   matches the pre-state-buffer reference within `atol=rtol=2e-6`.
2. **Frozen invariant:** after anchored relaxation,
   static `pos/qrot` are unchanged (target `atol=1e-7`) and their velocity/FIRE rows are
   exactly zero; at least one dynamic body moves.
3. **Static interaction:** dynamic force/trajectory differs when the same frozen partner
   is live versus deleted, proving the static body still interacts.
4. **Deletion parity:** a three-body scene with body 1 deleted matches a physically rebuilt
   two-body reference for live-body forces/energy.
5. **Mixed FAF:** four requested species have aligned tids/packs/bonds; per-site FAF data
   length equals `total_atoms`; one kernel-15+FAF step is finite.
6. **Map decomposition:** `E_sum == E_pairff_static + E_faf_probe`; dynamic-only pose
   changes leave it unchanged; state/probe changes invalidate it.
7. **Standalone parity:** for the same static sites/probe/grid, shared map values and
   `potential_to_rgba` match `RigidBodyVispy` behavior.
8. **Dummy overlay:** marker counts equal the live type-1/type-2 site counts and positions
   follow the rigid transform.
9. **Gesture mapping:** Shift+LMB toggles the selected body, ordinary LMB preserves drag,
   RMB excludes the body, and the last live body cannot be removed.

### L1 — agent review

Run the focused test in develop mode without filtering stdout. Read every reported
`.out`/`.log` artifact. Record:

- OpenCL device name (must be NVIDIA; never report PoCL/CPU as GPU);
- body states and maximum static displacement;
- worst kernel parity error and index;
- mixed species and per-pack real/dummy counts;
- PairFF, FAF, and sum map ranges;
- cold/warm map recompute timings.

### L2 — USER visual review

The USER reviews PNG/GIF output and confirms:

- static molecules are unmistakable without losing element colors;
- only dynamic molecules move;
- the combined background visibly contains both NaCl corrugation and frozen-molecule
  basins;
- O− probe controls produce the expected changes;
- e-pairs/sigma-holes are readable but not visually dominant;
- layout remains maximally tight.

Only after this confirmation may task/status documentation be marked resolved/done.

---

## 7. One remaining USER decision

The minimal event-driven CPU map is measured before adding another OpenCL kernel.

**Decision:** Is a warm recompute budget of **0.2 s** acceptable for build/toggle/probe
events, with GPU map work added only if that budget is missed? If you require the 2D map
to be GPU-computed regardless of measured latency, say so explicitly; that expands the
task by one new evaluator kernel and CPU↔GPU parity tests.
