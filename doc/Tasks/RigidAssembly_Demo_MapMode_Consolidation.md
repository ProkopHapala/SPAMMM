---
type: Task
title: Rigid Assembly — demo variants, probe-map/simulation parity, map extent, post-script mode, edit-mode consolidation
status: implemented — L0 tests pass (450/451); L2 USER visual review in progress (color scale + nuclear exclusion + alt FAF layer corrected 2026-08-03)
tags: [rigid-body, PairFF, probe-map, GPU, edit-modes, GUI, demo, consolidation]
timestamp: 2026-08-03
related:
  - doc/Tasks/RigidAssembly_StaticMols_PotentialMap.md
  - doc/Tasks/RigidAssembly_StaticMols_PotentialMap-corrections.md
  - doc/Reports/StaticObstacle_DragDemo_2026-08-03.md
  - doc/Tasks/Drag_Demo_Issues.md
  - user_guide/GUI_CHEATSHEET.md
---

# Rigid Assembly — reviewed implementation specification

> **Authority:** this reviewed specification supersedes the archived initial draft at
> the end of this document. The implementation is complete; L0 tests pass (450 passed,
> 1 pre-existing unrelated failure, 2 skipped). L2 USER visual review is in progress
> after three post-implementation corrections (§9). No issue is marked resolved until
> USER confirms the final GIFs.

## 0. Decisions

| Issue | Required decision |
|---|---|
| Demo scripts | Keep **two menu-loadable GUI macro scripts**: `static_obstacle_drag_demo.py` and `static_obstacle_drag_demo_alt.py`. They take no CLI parameters. Duplication of macro/animation orchestration is acceptable; the shared core/engine remains the single source of truth for physics and GUI operations. |
| Alternate narrative | Continuous hand-off: drag body 1 toward static body 0, swap roles at closest approach, then pull body 0 **away** from newly static body 1. Recompute direction from current poses. |
| PairFF map | The raw map must use the exact PairFF data used by dynamics. No display-only `He`/`Hs` substitution is allowed. |
| Map meaning | Show the energy of one explicit test probe over **live static bodies + substrate**, not the total energy of a rigid molecule or the whole simulation. Label this scope in the UI. |
| GPU evaluator | Add one grid kernel, but first factor the PairFF site-pair primitive so the grid and dynamics call the same inline force/energy function. Copying the formula into another kernel is not acceptable parity. |
| Map extent | Pass a Rigid Assembly GUI default margin of **10 Å**; do not change the shared helper's default for unrelated callers. Expand the short axis to the view aspect ratio. |
| Weak map contrast | Keep raw energy unchanged. Provide Total/PairFF/FAF layer selection, an explicit color limit with Auto, a color legend, and a value readout. Do not add a “physics gain” control or signed log transform. |
| Post-script mode | The demo itself sets the explicit manipulation context and switches to canonical `Manipulate` mode before its final frame. Do not add `final_mode` to the script runner. |
| Edit modes | Replace the six extension entries `RA Drag`, `FR Pin`, `FR COM`, `FR Manip`, `Pin/Unpin`, and `RC pin` with one canonical `Manipulate` mode. |
| Dispatch | Use an explicit manipulation-context SSOT set by Build/Setup. Never guess from `hasattr(window, 'ra_rbd')` versus `fr_rbd`; both can coexist. |
| Constraints | `MoleculeEditorBackend.constraint_set` is the spatial-constraint SSOT. Shift+LMB toggles it and synchronizes any built FF controller. Do not retain separate FF and RC pin truths. |
| Hex modes | `Hex1`/`Hex2` are unrelated to rigid manipulation and have distinct paint/preserve semantics. Leave them unchanged in this task; audit them separately if desired. |

## 1. Code audit findings that constrain the design

### 1.1 Demo and script lifecycle

- `demos/gui_scripts/static_obstacle_drag_demo.py::run` already contains all build,
  anchor, relaxation, capture, and export orchestration. The mid-drag state toggle is a
  block inside its only drag loop. The alternate behavior should be a second
  menu-loadable GUI macro, not a CLI branch in the first script.
- GUI scripts are datafile/animation-like macros loaded from the menu and are not
  standalone programs. They may contain repeated high-level sequencing when that keeps
  each animation readable. DRY applies to the shared engine, physics, rendering, and
  testable workhorse functions they call; do not add an abstraction solely to deduplicate
  macro-script text.
- The script currently chooses the first real O site, not the site best aligned with the
  requested pull, and it reports two different energy estimates in the toggle and normal
  paths. All passes must use one anchor-selection rule and
  `RigidBodyPairFF.eval_energy_system`.
- The script deletes captured `_frame_*.png` files after encoding. Do not delete those
  plots in the implementation; place them in a variant-specific `frames/` directory and
  retain them for review.
- `gui_script_runner.ScriptOptions` is deliberately presentation-only
  (`delay_ms`, `points_per_frame`, `honor_barriers`). A physics-specific final mode does
  not belong there.

### 1.2 PairFF parameter and map mismatch

- `RigidAssemblyExtension._on_build` constructs the live force field with
  `He=-0.1`, `Hs=1.0`, `w=0.7`, `beta=1.7`.
- `RigidBodyPairFF.init_pairff` already records the live values in
  `rbd.pairff_params_host`, and the actual per-site values used by the kernel are packed
  in each `pack['REQ_ext']` / the flat `dyn_REQ` GPU buffer.
- `RigidAssemblyExtension._recompute_ra_combined_map` ignores that state and passes
  `He=-1.0`, `Hs=0.0`, `w=0.7`, `beta=1.7`.
- `RigidBodyVispy.compute_potential_map_unified` copies the REQ array and overwrites its
  dummy-site `.z/.w` values again. The map therefore has a second parameter source.
- The CPU map omits the damped real-atom Coulomb term that is present in unified PairFF
  dynamics. It is not kernel-15 energy parity even when `He/Hs/w/beta` happen to match.
- `RigidBodyUtils.compute_combined_probe_map` imports compute code from
  `spammm.GUI.RigidBodyVispy`, reversing the desired dependency direction
  (`forcefields` → `GUI`).

### 1.3 Existing GPU reuse point

- `kernels/Forces.cl::compact_exp_pair_EF` is already the shared compact-exponential
  radial primitive.
- The unified kernels in `kernels/rigid.cl` repeat the surrounding REQ mixing, cutoff,
  dummy suppression, and damped Coulomb block in multiple inner loops.
- Kernel 15 (`rigid_body_pairff_multimol_kernel`) is the Rigid Assembly dynamics path.
  A map kernel that merely repeats its mixing block would still be a second physics
  implementation. The common unit must be an inline **site-pair E/F primitive** used by
  both kernels.
- The FAF evaluator used by kernel 15 is already an inline GPU tensor evaluator.
  `FoldedRigid.materialize_factorized_coeffs` and the existing typed-fit data provide the
  correct way to prepare a single probe's substrate coefficients.

### 1.4 Extent and display facts

- `compute_combined_probe_map` currently uses atom bounds plus a 4 Å additive margin.
- `AtomScene.fit_to_atoms(margin=1.8)` uses a **multiplicative span factor**, not a
  1.8 Å additive margin, and the canvas aspect ratio can expose more of the short axis.
  The old explanation of two directly comparable margins was incorrect.
- `potential_to_rgba(Emap, vmin=None, vmax=None)` currently ignores its `vmin` and
  `vmax` arguments and always resets them to `±max(|Emin|, 0.01)`.
- The RA map cache already stores `E_sum`, `E_pairff`, `E_faf`, axes, and extent, which
  is enough for layer selection and O(1) cursor lookup without recomputation.
- The RA `e-pairs` checkbox currently calls a placeholder that does nothing. A visible,
  checkable no-op is misleading.

### 1.5 Edit-mode facts

- With the default enabled extensions, there are **14**, not 13, entries: eight built-in
  and six extension entries relevant to this audit.
- Current RA gestures are `LMB drag`, `Shift+LMB static/dynamic`, and `RMB delete`.
  They do not match the old draft's proposed RMB-toggle/Ctrl+RMB-delete table.
- `FR Pin` is a persistent atom selection used by relaxed scans. `FR Manip` creates a
  transient spring and clears the pin on release. `FR Manip` is therefore not currently
  a superset of `FR Pin`.
- `FR COM` only writes the x/y setup spinboxes; it does not move an already constructed
  rigid body.
- `RC pin` toggles `MoleculeEditorBackend.constraint_set`. `Pin/Unpin` toggles a separate
  `FFController._pinned_mask`.
- `Pin/Unpin` switches to internal mode name `pin_unpin`, but no handler is registered
  for that name; it is currently nonfunctional through the scene.
- Extension dropdown labels and internal handler names are different
  (`RA Drag` → `ra_drag`, etc.). Programmatic `set_edit_mode` does not reliably make the
  combo's visible selection the canonical state.

## 2. Scientific contract for the probe map

### 2.1 Quantity being plotted

For a real test probe with

```text
REQ_probe = [R0_probe, sqrt(E0_probe), Q_probe, 0]
```

at `p=(x,y,z_probe)`, define

```text
E_map(p) = E_PairFF_probe,static(p) + E_FAF_probe(p)
```

where:

- `E_PairFF_probe,static` sums the unified PairFF site-pair energy over every site of
  every **live static** body;
- dynamic bodies are excluded because this view diagnoses the fixed obstacle field;
- deleted bodies are excluded;
- real-real pairs include both compact-exp and damped Coulomb exactly as dynamics;
- probe-dummy pairs use the dummy `REQ.z/.w` values already packed for dynamics;
- dummy-dummy terms are zero by the same mixing rule;
- `E_FAF_probe` is the substrate energy for that same probe REQ and position.

The map intentionally excludes rigid-body `k_z`, anchor springs, MC packing penalties,
and interactions with dynamic bodies. These exclusions must appear in the map tooltip or
help text. The label should be:

> `Probe E: static bodies + substrate`

Do not call the result “simulation total energy” or imply it is the energy of the dragged
molecule.

### 2.2 Parameter SSOT

- Production evaluation receives `REQ_ext`/`dyn_REQ` directly. It must not accept
  independent `He`, `Hs`, or `w` overrides.
- `beta` comes from `rbd.pairff_params_host['beta']`; absence is a loud error for a
  unified PairFF map, not a fallback to `1.7`.
- The probe controls remain explicit because the probe is an external diagnostic object,
  not one of the simulated bodies.
- Cache a parameter snapshot with the map result: probe REQ, `z_probe`, `beta`, body-state
  revision, grid specification, FAF fit identity/mode, and the PairFF mode.

### 2.3 Parameter provenance and calibration

Repository search and git history show that the current `He=-0.1`, `Hs=1.0`, `w=0.7`,
`beta=1.7`, `epair_dist=1.4`, and `sigma_dist=1.0` defaults were introduced without a
recorded fit, citation, or validation dataset. Examples contain other illustrative values
(`He=-1.0`, epair charge `-0.2`), but they are not calibration evidence.

Therefore this consolidation must:

1. Preserve the current dynamics values; do not tune physics to improve appearance.
2. Mark them in `doc/nonbonding_forcefields.md` as **unfitted working defaults**.
3. Record units, sign conventions, dummy placement, charge-conservation behavior, and
   the fact that unified `rc` is not the compact-exp cutoff control.
4. Create a separate calibration task, not a hidden part of this GUI change.

The later calibration should fit energies **and forces** to a small, versioned reference
set of distance/orientation scans for representative donor/acceptor dimers, with held-out
chemistries and physical checks (correct minimum geometry, dissociation to zero, no
spurious deep wells). AFM contrast is a downstream validation, not the primary parameter
fit target.

### 2.4 Honest but readable visualization

Raw arrays are never transformed in the cache. Only RGBA conversion changes display.

Required compact controls:

| Control | Behavior |
|---|---|
| Layer | `Total` (default), `PairFF`, or `FAF`; switching uses cached arrays only. |
| Color ± | Numeric energy limit in the map's energy unit. |
| Auto | Sets `Color ±` from the selected layer's attractive range; if no negative values exist, use a robust finite absolute percentile and report that rule. |
| Margin Å | RA-specific default `10.0`; recompute on editing completion. |
| Recompute | Explicit immediate recompute. |

Display requirements:

- zero-centered diverging `RdBu_r`, negative attraction blue and positive repulsion red;
- saturation is allowed, but a small legend must show `-limit`, `0`, `+limit` and units;
- status text reports raw `[min,max]`, displayed limit, grid shape, and evaluation time;
- mouse hover over the map reports `(x,y,z)`, `E_total`, `E_pairff`, and `E_faf` from the
  nearest cached grid cell;
- no logarithmic transform for signed energy and no unlabeled “gain” slider;
- implement the `e-pairs` control as `Sites` showing cyan electron pairs and magenta
  sigma holes with a tiny legend, or remove the control until it works;
- show static bodies without replacing element colors: use a thin cyan halo/COM marker
  and `S<body_id>` label. Deleted bodies remain absent.

Fix `potential_to_rgba` so explicit limits are honored, and reuse it in all map views.
Factor the repeated “RGBA array + axes/extent → cached VisPy Image” operation from
`VispyUtils.update_faf_map_overlay`; do not keep a second transform implementation in
`RigidAssemblyExtension`.

## 3. GPU evaluator and map geometry

### 3.1 Shared PairFF primitive

Add one inline function beside `compact_exp_pair_EF` in `kernels/Forces.cl`, conceptually:

```text
pairff_unified_site_EF(dp, REQ_i, type_i, REQ_j, type_j, beta)
    -> force_xyz + energy
```

It owns, in the current operation order:

- `g_i/g_j`, `R0`, blunt width, `alpha`, compact cutoff, attraction mixing, and
  dummy-dummy suppression;
- the `compact_exp_pair_EF` call;
- damped Coulomb for real-real pairs.

First replace the repeated blocks in unified rigid kernels with this inline helper without
algebraic regrouping. Run pre/post one-step and replica-energy parity before adding the
map kernel. If parity exceeds the tolerances in §8, stop and diagnose the first differing
pair; do not relax tolerances.

### 3.2 Grid kernel

Add one kernel to the existing `RigidBodyPairFF` OpenCL program:

- one work-item per `(ix,iy)` pixel;
- world positions, packed REQs, and types for live static sites are contiguous inputs;
- load site tiles cooperatively into local memory and gather over them; no atomics;
- evaluate PairFF with `pairff_unified_site_EF`;
- evaluate FAF with the same inline folded tensor evaluator used by kernel 15;
- write one `float4` per pixel:
  `(E_total, E_pairff, E_faf, 0)`;
- compile with the existing RBD program, allocate output/static-site buffers persistently,
  and resize only when capacity is insufficient;
- perform one output readback per recompute. Do not launch once per pixel and do not
  transfer one result at a time.

For factorized FAF, materialize coefficients for the explicit probe REQ through the
existing factorized PLQH path. For typed FAF, resolve the probe type through the existing
typed-fit mapping. If a typed fit cannot represent the requested custom probe, fail with
a clear message rather than silently choosing the nearest type.

Host-side rigid transforms may prepare the small static-site upload on invalidation; the
scientific pair primitive and FAF evaluator must remain the GPU SSOT. Do not transform
every site separately for every pixel if a single pretransform/upload suffices.

### 3.3 CPU reference

Keep an independent NumPy evaluator only as a test/reference path:

- place it in `RigidBodyUtils.py`, not a GUI module;
- include compact-exp, damped Coulomb, dummy mixing, and FAF;
- accept already-packed REQs instead of `He/Hs/w` overrides;
- never use it for normal interactive recompute;
- retain a compatibility wrapper in `RigidBodyVispy.py` if external imports require it.

This independence is intentional: a parity test that calls the same OpenCL helper twice
cannot detect a wrong formula.

### 3.4 Grid planning and view coverage

Add/reuse one host grid planner in `RigidBodyUtils.py`:

1. Start from all live real-atom XY bounds.
2. Add the explicit RA margin (`10.0 Å` default).
3. When a real scene is available, expand only the shorter axis so the map rectangle
   covers the current view aspect ratio; never crop atom+margin bounds.
4. Snap endpoints to the requested step so axes, extent, and image transform agree.
5. Keep `indexing='xy'`, result shape `(ny,nx)`, and row 0 = `ymin`.
6. Validate finite positive step and a maximum grid size. If the request is too large,
   report the required point count and ask the user to increase step; do not silently
   coarsen scientific sampling.

Pass `margin=10.0` from the RA panel. Leave the shared default at 4 Å for compatibility.
Do not recompute on every camera pan/zoom. The larger, aspect-aware map should cover the
normal fitted view; a manual recompute handles an intentionally changed view.

### 3.5 Invalidation and responsiveness

- Parameter spinboxes recompute on `editingFinished`, not every intermediate
  `valueChanged`.
- State/probe/height/margin/step changes mark one dirty revision and schedule one
  zero-delay Qt callback after the mouse event returns. Coalesce repeated changes.
- Never call `processEvents()` recursively from the map computation or create a new
  visual inside a VisPy mouse callback.
- Dynamic-body motion alone does not dirty the static-obstacle map.
- Layer and color-limit changes only recolor cached arrays.
- Warm performance acceptance on the NVIDIA device is ≤0.20 s end-to-end for the
  default scene/grid, with kernel time and readback time reported separately. Target
  kernel time is ≤0.05 s; do not report PoCL/CPU timing as GPU performance.

## 4. Two menu-loadable GUI scripts

### 4.1 Menu-loaded macro structure

Keep these as two GUI scripts selected from the existing menu:

```text
static_obstacle_drag_demo.py
static_obstacle_drag_demo_alt.py
```

Do not add CLI arguments, command-line entry points, or runner-level `final_mode`
parameters. The menu loads the script and lets it control the existing GUI/engine
context. Script-local capture and pacing are animation glue; physics/state/map
operations remain calls to existing shared modules and extension workhorses.

The two files may repeat high-level macro sequencing when that makes the animation
easier to inspect and edit. Do not duplicate PairFF, map, rendering, or GUI-engine
implementations inside them, and do not create a script framework merely to remove
similar lines from these data/animation files.

Choose the anchor deterministically from real atoms of the selected body:

1. prefer O atoms;
2. among candidates, choose the atom with the largest projection along the requested
   pull direction (the leading site);
3. if there is no O, use the leading real atom;
4. print body, local site, element, flat site, and initial world position.

### 4.2 `static` sequence

1. Load and split the dimer; require exactly two connected components for this demo.
2. Build RA and verify ensemble/GPU/host pose parity.
3. Relax both bodies without an anchor.
4. Set body 0 static and body 1 dynamic.
5. Set the H+ probe, compute the honest GPU map, show `PairFF` first long enough to make
   directional sites legible, then return to `Total` for the drag.
6. Drag body 1 toward body 0 with no state toggle.
7. At every reported step use `eval_energy_system`; print energy, energy change, anchor
   error, and maximum displacement of the static body's position/quaternion.

### 4.3 `alternate` sequence

Run the same first pass. At closest approach:

1. release the anchor and synchronize all poses;
2. freeze body 1, unfreeze body 0, and assert the state vector is `[dynamic, static]`;
3. recompute the map so it now contains body 1, not body 0;
4. recompute the separation direction from current CoMs;
5. pull body 0 away from body 1. This continuous detach pass demonstrates that the
   obstacle role switched without forcing already-close molecules further into overlap;
6. track body 1's invariant during the second pass.

Do not reset poses between passes; the point of this variant is continuous role hand-off.

### 4.4 Final state and artifacts

Before the final yielded frame:

- release all anchors and reset dynamics state;
- synchronize ensemble and display;
- set manipulation context to `rigid_assembly`;
- call the canonical GUI utility to switch to `Manipulate`;
- leave the final map visible and current.

Use variant-specific output roots:

```text
debug/static_obstacle_drag_demo/static/
debug/static_obstacle_drag_demo/alternate/
```

Each contains retained numbered PNG frames, first/last PNG, GIF, optional MP4, and a
plain-text summary with parameters, device, map ranges/timings, state transitions,
energy trace, static-pose maximum error, and output paths. Print each artifact as
`REVIEW: <path>` with unbuffered progress.

## 5. Consolidated interaction design

### 5.1 Context SSOT

Add one explicit GUI value:

```text
active_manipulation_context = None | 'rigid_assembly' | 'folded_rigid'
```

- RA Build sets `rigid_assembly`.
- Folded Rigid Setup/Run sets `folded_rigid`.
- The active target is shown in the status/mouse hint whenever `Manipulate` is selected.
- If no context is set, rigid drag gestures fail loudly with “Build RA or Setup Folded
  Rigid first.”
- If both `ra_rbd` and `fr_rbd` exist, the explicit value wins. Never infer the target
  from attribute existence or panel visibility.

The canonical mode name and dropdown label are both `Manipulate`. `set_edit_mode` must
synchronize `edit_mode`, handler activation, scene flags, mouse hint, and combo selection
without recursive signals. Remove label→internal-name callback indirection for this mode.

### 5.2 Gesture contract

Use a small pixel threshold to distinguish click from drag. A press alone must not launch
dynamics.

| Gesture | Context | Action |
|---|---|---|
| LMB click atom | RA | Select active body; report body id and dynamic/static state. |
| LMB drag atom | RA | Pull a dynamic body with the existing anchor/FIRE path. Static/deleted bodies cannot acquire an anchor. |
| LMB click atom | Folded | Persistently select the scan pin atom. |
| LMB drag atom | Folded | Pull with the existing folded anchor; on release remove the transient spring but retain the selected scan-pin identity. |
| LMB release | Both | Release transient anchor, reset appropriate dynamics state, commit current pose to its authority. |
| RMB atom | RA only | Toggle dynamic ↔ static. |
| Ctrl+RMB atom | RA only | Soft-delete body after the existing last-live-body guard. |
| Shift+LMB click atom | Any editor/FF/RC context | Toggle the backend spatial constraint SSOT and synchronize consumers. This gesture never starts rigid drag. |
| Alt+LMB empty | Folded only | Set the **next Setup** x/y COM values and say “press Setup to apply.” It must not pretend to move a live body. |
| Plain LMB empty | Manipulate | No action. |

RMB on empty retains camera rotation in 3D. Ctrl+RMB deletion is intentionally guarded
because it is destructive. If undo for RA soft deletion is unavailable, label it
“soft-delete” and keep the data recoverable; do not physically remove packs.

### 5.3 Separation of concerns

Implement the gesture state machine once in the existing
`spammm/GUI/EditModeHandlers.py`. Keep physics-specific actions in their extensions:

- Rigid Assembly owns scene-index→body/site mapping, body state, anchors, all-mobile MD,
  ensemble synchronization, and RA visuals.
- Folded Rigid owns its pin identity, anchors, folded dynamics, COM setup fields, and
  graph update.
- The central handler dispatches only to the adapter registered for the explicit context.
  Use a minimal callback record/dict in `SPAMMM_GUI.py`; do not create a new module or a
  general plugin framework.

Reuse `_closest_point_on_ray` from one shared existing GUI utility rather than retaining
copies in both rigid extensions.

### 5.4 Constraint SSOT

Shift+LMB receives a stable atom id and calls
`MoleculeEditorBackend.toggle_constraint(atom_id)`. Then:

1. derive the dense mask through `backend.constraint_mask()`;
2. update `scene.set_fixed_mask(mask)`;
3. if `FFController` is built and atom counts match, call `set_pinned(mask, current_pos)`;
4. if counts do not match, fail loudly that the FF is stale;
5. RC reads the same backend mask and only updates its status text.

Remove the separate `Pin/Unpin` and `RC pin` dropdown entries after the consolidated
gesture is tested. Keep compatibility functions temporarily if scripts import them, but
they must delegate to the backend SSOT rather than mutate independent masks.

### 5.5 Migration and non-goals

1. Add `Manipulate` and its tests.
2. Make RA/Folded build/setup activate the explicit context.
3. Route constraints through the backend SSOT.
4. Keep old handler names registered for one compatibility period, but hide their six
   labels from the dropdown.
5. Update the cheatsheet and panel hints.
6. Remove compatibility handlers only after USER confirmation in a later task.

Do not change `Unified`, `Atom`, `Bond`, `Ring`, `pi`, `Select`, `Hex1`, or `Hex2` in
this implementation. Do not unify the RA and Folded physics objects or pose authorities.

## 6. Work packages and exact responsibilities

### Package A — tests before physics edits

- Add independent site-pair reference cases and record current kernel 14/15 results.
- Add non-default PairFF parameter cases so a hidden hardcoded default fails.
- Add mode/context/constraint tests that fail with the current six-mode behavior.

### Package B — shared PairFF primitive and GPU map

- Factor the inline PairFF E/F primitive.
- Prove kernel pre/post parity.
- Add persistent grid buffers/launcher to `RigidBodyPairFF`.
- Route RA and standalone PairFF map recompute through the GPU evaluator.
- Retain the corrected CPU reference for parity only.

### Package C — map UI, extent, and visuals

- Add the grid planner and explicit RA margin.
- Add layer/color controls, legend, value readout, coalesced invalidation, shared image
  overlay, static markers, and functional dummy-site toggle.
- Verify orientation and coverage with non-square grids and canvases.

### Package D — menu-loaded demo macros and final mode

- Keep `static_obstacle_drag_demo.py` as the static-obstacle macro and add
  `static_obstacle_drag_demo_alt.py` as the alternate hand-off macro.
- Keep both scripts directly loadable from the GUI menu, with no CLI parameterization.
- Remove the mid-drag toggle from the `static` behavior.
- Retain all review frames and leave the GUI in RA `Manipulate` context.
- Update `demos/gui_scripts/README.md` with both menu entries and their macro purpose.

### Package E — interaction consolidation

- Add canonical mode/context dispatch and click-vs-drag threshold.
- Register RA/Folded adapters.
- Make backend constraints authoritative and synchronize the FF consumer.
- Hide the six superseded extension labels after tests pass.

### Package F — documentation after verified implementation

- Update `user_guide/GUI_CHEATSHEET.md` with the exact gesture table.
- Update `doc/nonbonding_forcefields.md` with the unfitted parameter status.
- Update `doc/TopicalAudit/PairFF_RigidBody.md`, `doc/topical_audit.md`,
  `demos/PairFF_manual.md`, and relevant folder READMEs.
- Correct the obsolete “display-only He/Hs amplification is allowed” guidance in
  `doc/Caveats.md` and `doc/Takeways.md`.
- Keep every task/status field unverified until the USER accepts the L2 artifacts.

## 7. Expected implementation footprint

One new GUI macro file is intentional: `static_obstacle_drag_demo_alt.py`. No new
physics, rendering, or script-runner framework is required.

| Existing file | Responsibility |
|---|---|
| `kernels/Forces.cl` | Shared unified site-pair E/F primitive. |
| `kernels/rigid.cl` | Use shared primitive; add fused probe-grid PairFF+FAF kernel. |
| `spammm/forcefields/RigidBodyDynamics.py` | Kernel signature/launcher, persistent buffers, probe FAF preparation, timing. |
| `spammm/forcefields/RigidBodyUtils.py` | Grid planning, production orchestration, independent CPU reference; no GUI import. |
| `spammm/GUI/RigidBodyVispy.py` | Compatibility wrapper, corrected color-limit SSOT, consume GPU map. |
| `spammm/GUI/VispyUtils.py` | Shared raw-array image overlay and ray helper. |
| `spammm/GUI/RigidAssemblyExtension.py` | Honest map cache/UI/invalidation/visuals and RA manipulation adapter. |
| `spammm/GUI/FoldedRigidExtension.py` | Folded manipulation adapter; persistent pin plus transient drag semantics. |
| `spammm/GUI/EditModeHandlers.py` | One click/drag/modifier state machine. |
| `spammm/GUI/SPAMMM_GUI.py` | Canonical mode/context registry, combo synchronization, constraint orchestration. |
| `spammm/GUI/FFExtension.py` | Delegate pin operations to backend constraint SSOT. |
| `spammm/GUI/ReactionCoordinateExtension.py` | Delegate pin operations to backend constraint SSOT. |
| `demos/gui_scripts/static_obstacle_drag_demo.py` | Menu-loaded static-obstacle animation macro. |
| `demos/gui_scripts/static_obstacle_drag_demo_alt.py` | Menu-loaded continuous role hand-off animation macro. |
| `tests/test_body_state.py` | Site/map GPU parity, decomposition, state filtering, performance record. |
| `tests/GUI/test_rigid_assembly_extension.py` | Map parameter use, context, gestures, state and display behavior. |
| `tests/GUI/test_gui_script_utils.py` | Canonical programmatic mode/combo synchronization. |

## 8. Verification and acceptance

### 8.1 L0 — automatic numerical and structural checks

1. **Input-first:** packed static world positions, `REQ`, types, body states, probe REQ,
   beta, grid axes, and FAF probe coefficients match the intended host snapshots.
   Integer/type/state checks are exact.
2. **Pair microcases:** independent float64 reference versus GPU for real-real
   compact-exp, real-real Coulomb, H+↔epair, O−↔sigma-hole, dummy-dummy zero, and cutoff
   boundary. Compare energy and force; report the worst pair/index.
3. **Kernel refactor:** kernel 14 energies and one exact kernel-15 step match pre-refactor
   references (`rtol=atol=2e-6` unless the existing stricter test applies).
4. **Point parity:** selected map pixels match a one-real-site probe-body energy
   evaluation against the same static bodies, with PairFF and FAF isolated before the
   combined comparison.
5. **Map channels:** `E_total == E_pairff + E_faf` within float32 accumulation tolerance;
   all values finite.
6. **State filtering:** static bodies contribute; dynamic and deleted bodies do not.
   Moving only a dynamic body leaves values at unchanged grid coordinates invariant.
7. **Parameter honesty:** build with deliberately non-default `He/Hs/w/beta`; assert the
   map consumes packed REQs and recorded beta, and no hardcoded map parameter remains.
8. **Geometry:** non-square grid shape is `(ny,nx)`; known asymmetric point features
   appear at the correct x/y coordinate; image extent covers atom bounds plus margin.
9. **Display:** changing layer/limit does not mutate cached raw arrays or launch a kernel;
   explicit `vmin/vmax` are honored.
10. **Interaction:** explicit context wins when RA and Folded objects both exist; click
    does not launch dynamics; drag does; release clears only transient anchors; Folded
    persistent pin remains.
11. **Constraints:** Shift+LMB toggles stable atom id → backend mask → scene and built FF
    masks identically; stale atom counts raise.
12. **Demo:** both variants finish with anchors disabled, expected body states, finite
    energy, retained frames, and canonical `Manipulate`/`rigid_assembly` state.

### 8.2 L1 — agent review

Run focused tests in `--develop -s` mode without filtering stdout and read every reported
artifact. Record:

- OpenCL device name; it must be NVIDIA;
- cold compile, warm kernel, readback, RGBA/update, and end-to-end map timings;
- grid shape, static-site count, PairFF/FAF/total raw ranges, color limit;
- worst CPU↔GPU energy and force error with case/index;
- state transitions and maximum static-body position/quaternion error;
- both demo summary files and all `REVIEW:` paths.

Recommended focused commands are the existing test files, not new wrappers:

```bash
pytest tests/test_body_state.py --develop -s
pytest tests/GUI/test_rigid_assembly_extension.py --develop -s
pytest tests/GUI/test_gui_script_utils.py --develop -s
```

Then run the routine suite:

```bash
pytest -m "not slow"
```

### 8.3 L2 — USER visual review

Provide first/last PNG and GIF for both variants plus one screenshot of the consolidated
mode and map controls. The USER confirms:

- attraction/repulsion and energy units are legible without changing physics;
- Total/PairFF/FAF layers explain which component creates each feature;
- the map covers the normal fitted viewport and has correct orientation;
- electron pairs, sigma holes, and static bodies are distinguishable but not dominant;
- the static body is visually and numerically stationary in each pass;
- alternate hand-off reads clearly;
- the final GUI is immediately interactive in the visibly selected `Manipulate` mode;
- the consolidated gesture scheme is comfortable and unambiguous.

Only after showing these results and receiving explicit USER confirmation may any related
task or issue status be changed to fixed/resolved/done.

## 9. Implementation record (unverified)

The approved implementation has been applied to the shared PairFF evaluator, probe-map
launcher, RA/Folded manipulation dispatch, constraint synchronization, and the two
menu-loaded static-obstacle macros. Focused NVIDIA OpenCL checks currently pass, but the
L2 visual review and USER confirmation required by §8.3 are still pending. This record
must not be changed to fixed/resolved/done without that confirmation.

Verification recorded on 2026-08-03: `tests/test_body_state.py --develop -s` (10 passed),
`tests/GUI/test_rigid_assembly_extension.py --develop -s` (14 passed), and
`tests/GUI/test_gui_script_utils.py --develop -s` (2 passed), all with the NVIDIA GeForce
GTX 1650 selected. `pytest -m "not slow"` reached 450 passed, 2 skipped, 21 deselected,
with one unrelated existing failure: `test_prepare_rc_scan_review_offscreen` refers to
the absent legacy path `spammm/GUI/gui_scripts/rc_scan_review.py`; the maintained script is
`demos/gui_scripts/rc_scan_review.py`.

### 9.1 Post-implementation corrections (USER-reported, 2026-08-03)

Three issues were found during L2 USER visual review and corrected. None of these
changes alters the §0 decisions or the physics; they fix display and demo-script bugs.

| # | Issue | Root cause | Fix |
|---|---|---|---|
| **C1** | Map color scale oversaturated after recompute | `_recompute_ra_combined_map` set the color-limit spin to the computed `\|Emin\|` on the first call; subsequent recomputes (after body-state toggle) reused the stale positive spin value instead of recomputing `\|Emin\|` from the new data. `_on_ra_map_auto` used `np.percentile(attractive, 5)` instead of `\|Emin\|`. | `_recompute_ra_combined_map` now resets the spin to 0 (Auto) before `_update_ra_combined_map_visual`, so `vmin=Emin, vmax=\|Emin\|` is freshly computed every time. `_on_ra_map_auto` uses `max(\|Emin\|, 0.01)`. Documented as a permanent rule in `.devin/skills/centralized-plotting/SKILL.md` §"Color Scale Rule". |
| **C2** | Alt demo missing FAF component | `static_obstacle_drag_demo_alt.py` set `ra_map_layer_combo` to `'PairFF'`, excluding FAF from the display. | Changed to `'Total'` so the combined PairFF+FAF map is shown. |
| **C3** | Nuclear singularities dominate `\|Emin\|`, washing out chemical features | The infinitely deep attractive wells at real-atom nuclei (compact-exp + damped Coulomb) dominate `np.nanmin(Emap)`, making the color scale too wide. | Added `nuclear_exclusion_mask(xs, ys, z_probe, static_apos, static_types, r=1.0)` in `RigidBodyUtils.py` — a boolean mask True within 1 Å of any real atom. `compute_combined_probe_map` returns it as a 7th value. The GUI uses `E_for_lim = E[~exclude_mask]` for `vmin/vmax` estimation only; the map itself is fully finite and displayed everywhere (no NaN holes). |

**Files changed in §9.1:**

| File | Change |
|------|--------|
| `spammm/GUI/RigidAssemblyExtension.py` | Reset spin to 0 on recompute; `_on_ra_map_auto` uses `\|Emin\|`; both use `exclude_mask` for color-limit estimation |
| `spammm/GUI/RigidBodyVispy.py` | `potential_to_rgba` docstring updated: `vmin=Emin, vmax=\|Emin\|` |
| `spammm/forcefields/RigidBodyUtils.py` | New `nuclear_exclusion_mask` function; `compute_combined_probe_map` returns 7-tuple with `exclude_mask`; CPU `_compute_unified_probe_pair_map` computes normally everywhere (no NaN) |
| `kernels/rigid.cl` | `rigid_body_pairff_probe_grid` computes physical energy at every pixel (no NaN early-return); header comment notes host-side mask |
| `demos/gui_scripts/static_obstacle_drag_demo_alt.py` | Layer set to `'Total'` (was `'PairFF'`) |
| `tests/test_body_state.py` | `test_combined_map_decomposition_and_invalidation` updated for 7-tuple return + finite-only assertions |
| `.devin/skills/centralized-plotting/SKILL.md` | New "Color Scale Rule" and "Nuclear exclusion" sections documenting the user-mandated rules |

**Verification after §9.1:** 17 focused tests pass; `pytest -m "not slow"` reaches 450
passed (same 1 unrelated failure). Both demos re-run successfully (55 + 109 frames).
Artifacts regenerated at `debug/static_obstacle_drag_demo/{static,alternate}/`.

## 10. Stop conditions for the implementing agent

Stop and report instead of guessing if:

- a typed FAF fit cannot represent the custom probe;
- the shared PairFF helper changes pre-existing kernel results beyond tolerance;
- the target NVIDIA device is unavailable;
- a proposed mode action has no explicit manipulation context;
- backend and FF atom counts/order disagree;
- map sampling would exceed the validated point limit;
- parameter calibration is requested as part of this implementation;
- compatibility requires deleting an old handler or demo file.

---

## Archived initial draft — do not implement

<details>
<summary>Original discussion retained for provenance; superseded by §§0–9 above</summary>

# Rigid Assembly — 5 follow-up issues

> Spawned from the 2026-08-03 corrections session (see
> [StaticObstacle_DragDemo_2026-08-03.md](../Reports/StaticObstacle_DragDemo_2026-08-03.md)).
> This is a **spec for discussion** — no implementation until USER approves.
> Each section lists the problem, the USER's stated preference, and proposed approach.

---

## 1. Demo variants — one-static + alternating

### Problem

The current `demos/gui_scripts/static_obstacle_drag_demo.py` freezes body 0 and
drags body 1 toward it, with a mid-drag toggle (static→dynamic→static). The
USER wants two cleaner variants:

- **Variant A — one static, one dynamic:** body 0 frozen, body 1 dragged
  toward it. No mid-drag toggle. Pure static-obstacle demonstration.
- **Variant B — alternating:** first drag body 1 (with body 0 static), then
  freeze body 1, unfreeze body 0, and drag body 0 (with body 1 static). Shows
  that the "static obstacle" role can switch between molecules.

### Proposed approach

- Keep the existing `static_obstacle_drag_demo.py` as variant A (remove the
  mid-drag toggle block at lines 207–232 — it muddies the static-obstacle
  narrative).
- Add `static_obstacle_drag_demo_alt.py` (variant B): after the first drag
  pass (body 1 → body 0), freeze body 1, unfreeze body 0, recompute the probe
  map, and drag body 0 → body 1. Reuses the same anchor/relax/capture
  machinery.
- Both scripts end by switching to `ra_drag` edit mode (see §4).

### Open questions

- Should variant B drag *toward* the now-static body 1, or *away* (pulling
  body 0 off the obstacle)? USER: please clarify the intended narrative.
- Should the two variants be separate files, or one file with a `--variant
  {static,alternate}` flag? Separate files is simpler (KISS) but duplicates
  ~80% of the code. A flag avoids duplication but complicates the script.

---

## 2. Probe-map parameters must match the simulation (PairFF parity)

### Problem

The combined probe map (`_recompute_ra_combined_map` in
`spammm/GUI/RigidAssemblyExtension.py:357–359`) uses hardcoded
**visualization-only** parameters:

```python
He = -1.0   # amplified e-pair contrast
Hs = 0.0    # sigma-hole disabled
w  = 0.7
beta = 1.7
```

But the assembly's PairFF is built with **different** values
(`spammm/GUI/RigidAssemblyExtension.py:541`):

```python
He=-0.1, Hs=1.0, w=0.7, beta=1.7
```

**Consequence:** the map shows an amplified, physically inaccurate H-bond
landscape. The e-pair attraction is 10× stronger in the map than in the
simulation, and sigma-hole contributions are entirely absent from the map.

### USER's position (verbatim)

> "You say that when you used same params for visualization and for
> simulation I was complaining that it is not visible, but the point is
> that we must then adjust the parameters perhaps, not to fake the map.
> The map is diagnostics of the simulation. If the e-pair bonding sites
> are not visible as relevant depression in the diagnostic view they are
> perhaps too weak to control the dynamics. How we chose the parameters
> anyway? They seem to me just pulled-out-of-thin-air."

### Root cause: parameter provenance

The `He=-0.1, Hs=1.0` defaults are defined in
`spammm/forcefields/RigidBodyDynamics.py:1563` (`_prepare_molecule_pack`) and
`spammm/forcefields/RigidBodyDynamics.py:2145` (`init_pairff`). There is no
documentation of how these values were derived — no fitting reference, no
DFT calibration, no analytical justification. They appear to be
hand-tuned defaults.

The `He=-1.0, Hs=0.0` map values were introduced during the corrections
session to make the H-bond minima *visually* visible after the USER
objected to the flat map produced by `He=-0.1`. This was a visualization
hack, not a physics fix.

### Proposed approach

1. **Make the map use the assembly's actual PairFF parameters.** The map
   is a diagnostic of the simulation — it must reflect the real
   force field, not an amplified caricature. `_recompute_ra_combined_map`
   should read `He`, `Hs`, `w`, `beta` from `window.ra_rbd` (the built
   `RigidBodyPairFF`), not from hardcoded constants.

2. **Investigate and document the parameter provenance.** If `He=-0.1`
   is too weak to produce visible e-pair minima in the map, that is a
   **physics signal**: either (a) the parameter is wrong and should be
   re-fitted, or (b) the parameter is right and the e-pair contribution
   genuinely is weak — in which case the map correctly shows this, and
   we adjust the *colormap scaling* (not the physics) to make weak
   features visible. Faking the parameters to make the map "look right"
   hides a potential calibration issue.

3. **Colormap scaling, not parameter amplification.** If the real
   `He=-0.1` produces a map with weak contrast, use a perceptual
   colormap with adaptive vmin/vmax (e.g. percentile-based clipping) or
   a log-scale, rather than inflating the physics. The map's job is to
   show the *relative* landscape shape, not absolute depth.

4. **Document the parameters.** Add a section to
   `doc/nonbonding_forcefields.md` (or a new `doc/PairFF_parameters.md`)
   explaining: what `He` and `Hs` represent physically (e-pair and
   sigma-hole pseudo-charges), how they were chosen, and what reference
   data (if any) they were fitted against. If no reference exists, mark
   them as "unfitted defaults — pending calibration" and create a
   fitting task.

### Open questions

- Does the USER want to re-fit `He`/`Hs` against a reference (DFT H-bond
  energies, AFM frequency shift curves), or keep the current values and
  just fix the map to use them honestly?
- Should the map have a "diagnostic gain" slider (clearly labeled as
  non-physical) so the USER can amplify weak features on demand without
  changing the simulation? This is a compromise between "honest map" and
  "visible map."

---

## 3. Map extent — fill the view + GPU evaluation

### Problem

The probe map extent is computed from all live real atoms + 4 Å margin
(`spammm/forcefields/RigidBodyUtils.py:262–273`). The camera
(`fit_to_atoms`, `spammm/GUI/VispyUtils.py:709`) uses a 1.8 Å margin and
may zoom to a different extent. Result: the map covers the atoms + 4 Å,
but the visible viewport extends further — the map doesn't fill the
view.

### USER's position (verbatim)

> "I would simply just increase margin, for fast replot I would use GPU
> evaluation, since we have it implemented on GPU anyway, the point is
> not just speed (although it is) but also be sure that we use same
> codepath for the map as for the simulation."

### Proposed approach

**3a. Increase margin.** Change the default `margin` in
`compute_combined_probe_map` from 4.0 to ~10–12 Å so the map extends
well beyond the atoms and fills the typical viewport. This is the
minimal fix. The margin should be a GUI parameter (spinbox) so the USER
can adjust it without code changes.

**3b. GPU evaluation — same codepath as simulation.** This is the
deeper fix and addresses the USER's parity concern from §2 as well.
Currently:
- **Simulation** uses the GPU kernel `rigid_body_pairff_kernel` in
  `kernels/rigid.cl` (compact-exp PairFF with He/Hs mixing).
- **Map** uses a *separate CPU implementation*
  (`compute_potential_map_unified` in `spammm/GUI/RigidBodyVispy.py:182`)
  that reimplements the same physics in NumPy.

Two separate implementations of the same PairFF physics is a parity risk
(see skill:`numerical-parity`). The USER wants the map to use the **same
GPU codepath** as the simulation.

**Options for GPU map evaluation:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. New grid-eval kernel** | Write a new OpenCL kernel that evaluates the PairFF energy on a 2D grid of probe positions, reusing the same mixing functions from `rigid.cl`. | Single codepath; fast; no host-device round-trips per pixel. | New kernel to write + test + maintain. |
| **B. Probe-atom sweep** | Place a probe atom at each grid point and call the existing `eval_energy_system` (or a lightweight variant). Reuses existing kernel. | No new kernel. | N_grid kernel launches (slow for large grids); probe is a single atom, not a body — may not match the body-pair kernel path. |
| **C. Keep CPU, add parity test** | Keep the CPU map but add a strict numerical-parity test vs the GPU kernel output. | Least work. | Doesn't solve speed; two codepaths still exist. |

**Recommendation:** Option A (new grid-eval kernel) is the right long-term
fix. It gives both speed and codepath parity. The kernel would:
- Take the static body's atom positions, REQ, types, enames (already on
  GPU in `ra_rbd`).
- Take a 2D grid (xs, ys) + z_probe.
- Evaluate the compact-exp PairFF energy at each grid point using the
  same `compact_exp_force_over_r` mixing as `rigid.cl`.
- Output an `(ny, nx)` energy buffer.

This is a `port-to-opencl` task (see skill:`port-to-opencl`).

### Open questions

- Does the USER want the GPU map kernel in this task, or as a separate
  follow-up? It's a significant piece of work (new kernel + parity test).
- For the margin increase (3a): should it be a GUI spinbox, or just a
  larger hardcoded default?

---

## 4. Switch to RA Drag mode after script finishes

### Problem

When a GUI script (e.g. `static_obstacle_drag_demo.py`) finishes, the
edit mode stays whatever it was before the script started (usually
`Unified`). The USER wants to be dropped into `ra_drag` mode after a
rigid-manipulation script finishes, so they can immediately interact
with the result.

### Proposed approach

- **Script-level fix (minimal):** Each rigid-manipulation demo script
  calls `window.set_edit_mode('ra_drag')` before its final `yield`. This
  is a one-liner per script and doesn't touch the script runner.

- **Runner-level fix (broader):** Add an optional `final_mode` field to
  `ScriptOptions` (or a `ctx.set_final_mode('ra_drag')` API) that the
  script runner applies after `StopIteration`. More general but adds
  API surface. YAGNI unless multiple scripts need different final modes.

**Recommendation:** Script-level fix for now. If a second script needs
the same behavior, generalize to the runner.

### Open questions

- Should this apply to *all* rigid-assembly scripts, or only the drag
  demos? (I.e. is `ra_drag` always the right final mode, or might some
  scripts want `Select` or `Unified`?)

---

## 5. Edit-mode consolidation — too many overlapping modes

### Problem

The edit-mode dropdown has **13 modes** (8 built-in + 5 extension):

| Mode | Source | Purpose |
|------|--------|---------|
| Unified | Built-in | All-in-one context-sensitive editing |
| Hex1 | Built-in | Force add/remove hex grid nodes |
| Hex2 | Built-in | Add/remove preserving shared edges |
| Atom | Built-in | Focused atom editing |
| Bond | Built-in | Bond insert/delete |
| Ring | Built-in | Fused n-gon ring placement |
| pi | Built-in | Toggle pi-orbital participation |
| Select | Built-in | Selection + sticky transform |
| RA Drag | RigidAssemblyExtension | Rigid body drag/toggle/delete |
| FR Pin | FoldedRigidExtension | Folded rigid: click to select pin atom |
| FR COM | FoldedRigidExtension | Folded rigid: click to set COM (x,y) |
| FR Manip | FoldedRigidExtension | Folded rigid: LMB drag atom to pull molecule |
| Pin/Unpin | FFExtension | Force-field: click atoms to toggle pin |
| RC pin | ReactionCoordinateExtension | Reaction coordinate pin |

The cheatsheet (`user_guide/GUI_CHEATSHEET.md`) documents the 8 built-in
modes but **does not mention any extension modes** — a documentation gap.

### USER's position (verbatim)

> "This is what I speak about, we have too many modes for rigid movement.
> They have overlapping functionality. By combination of LMB, RMB and
> modifiers ctrl, shift, alt we should be able to combine this
> functionality into one logical system, right? Suggest how."

### Analysis: overlapping rigid-manipulation modes

The five "rigid movement" modes overlap significantly:

| Capability | RA Drag | FR Pin | FR COM | FR Manip | Pin/Unpin | RC pin |
|------------|---------|--------|--------|----------|-----------|--------|
| Pick atom by click | ✓ (LMB) | ✓ (LMB) | — | ✓ (LMB) | ✓ (LMB) | ✓ (LMB) |
| Drag atom to pull body | ✓ (LMB drag) | — | — | ✓ (LMB drag) | — | — |
| Set COM by click | — | — | ✓ (LMB) | — | — | — |
| Toggle pin on/off | — | ✓ (select) | — | — | ✓ (toggle) | ✓ (select) |
| Toggle static/dynamic | ✓ (RMB) | — | — | — | — | — |
| Delete body | ✓ (RMB+Ctrl) | — | — | — | — | — |
| Release on mouse-up | ✓ | — | — | ✓ | — | — |

**Key observations:**
- **FR Pin** and **RC pin** are nearly identical: both just select a pin
  atom by clicking. The only difference is which extension reads
  `_fr_pin_idx` vs `_rc_pin_idx`.
- **FR Manip** is a superset of **FR Pin**: it picks the pin atom *and*
  drags it. FR Pin is redundant if FR Manip can also "just select
  without dragging" (e.g. click without drag).
- **Pin/Unpin** (FFExtension) toggles pin state on click — different
  semantics (toggle vs select) but same gesture (click atom).
- **RA Drag** is the most capable: drag, toggle static, delete. But it
  doesn't do pin selection or COM setting.

### Proposed consolidation: one "Rigid Manip" mode

Merge the five rigid-movement modes into a single **"Rigid Manip"** mode
with modifier-based dispatch:

| Gesture | Modifier | Action |
|---------|----------|--------|
| LMB click on atom | — | Pick pin atom (selects which atom to anchor) |
| LMB drag on atom | — | Drag atom → pull body (spring anchor follows mouse) |
| LMB release | — | Release anchor (body free) |
| RMB click on atom | — | Toggle static ↔ dynamic (RA Drag behavior) |
| RMB click on atom | Ctrl | Delete body (RA Drag behavior) |
| LMB click on atom | Shift | Toggle pin on/off (Pin/Unpin behavior) |
| LMB click on empty | — | Set COM (x,y) at cursor (FR COM behavior) |
| LMB click on atom | Alt | Set reaction-coordinate pin (RC pin behavior) |

**Rationale:**
- **LMB = manipulate** (pick/drag/pull), **RMB = state toggle** (static/
  dynamic/delete), **modifiers = secondary actions** (pin, COM, RC).
- No mode switching needed — all rigid-body interactions available in
  one mode.
- The mode is context-aware: if `ra_rbd` exists, RA Drag semantics apply;
  if `fr_rbd` exists, FR Manip semantics apply; if `ff_controller`
  exists, Pin/Unpin semantics apply. The modifier dispatch picks the
  right target based on which extension is active.

**Migration path:**
1. Implement "Rigid Manip" as a new mode handler that dispatches based
   on modifiers + active extension.
2. Keep the old modes temporarily (deprecation period) so existing
   scripts and muscle memory still work.
3. Update cheatsheet with the consolidated mode.
4. After USER confirms the consolidated mode works, remove the old
   modes from the dropdown (keep handlers registered for script
   compatibility).

### Hex1 / Hex2 — also legacy?

**Hex1** (force add/remove) and **Hex2** (preserve shared edges) are
older hex-grid ring placement modes. The **Ring** mode generalizes ring
placement (edge, corner, hex center) and is the recommended mode per the
cheatsheet. Hex1/Hex2 may be legacy.

**Open question:** Does the USER still use Hex1/Hex2, or are they
superseded by Ring? If legacy, remove from dropdown (keep handlers).

### Documentation gap

The cheatsheet (`user_guide/GUI_CHEATSHEET.md`) must be updated to
document all extension modes (or the consolidated mode). Currently only
the 8 built-in modes are documented.

### Open questions

- Does the USER agree with the modifier scheme above? Specifically:
  Shift = toggle pin, Alt = RC pin, LMB-empty = set COM?
- Should the consolidated mode be context-aware (auto-detect active
  extension) or should the USER explicitly pick "RA" vs "FR" vs "FF"
  sub-mode via a secondary control?
- Are Hex1/Hex2 legacy? Should they be removed from the dropdown?

---

## Implementation order (proposed)

1. **§4** (post-script mode) — trivial, unblocks demo work.
2. **§1** (demo variants) — depends on §4.
3. **§3a** (increase margin) — quick fix, immediate visual improvement.
4. **§2** (map uses simulation parameters) — physics fix, may reveal
   parameter calibration issues.
5. **§3b** (GPU map kernel) — larger task, separate PR.
6. **§5** (edit-mode consolidation) — largest task, needs USER design
   approval first.

---

## Files touched (anticipated)

| File | Section | Change |
|------|---------|--------|
| `demos/gui_scripts/static_obstacle_drag_demo.py` | §1, §4 | Remove mid-drag toggle; add `set_edit_mode('ra_drag')` at end |
| `demos/gui_scripts/static_obstacle_drag_demo_alt.py` | §1 | New: alternating static/dynamic variant |
| `spammm/GUI/RigidAssemblyExtension.py` | §2, §3a | Map reads `He`/`Hs` from `ra_rbd`; increase margin |
| `spammm/forcefields/RigidBodyUtils.py` | §3a | Increase default `margin` |
| `kernels/rigid.cl` | §3b | New grid-eval kernel (separate PR) |
| `spammm/GUI/SPAMMM_GUI.py` | §5 | Consolidated mode handler |
| `spammm/GUI/RigidAssemblyExtension.py` | §5 | RA Drag → Rigid Manip dispatch |
| `spammm/GUI/FoldedRigidExtension.py` | §5 | FR modes → Rigid Manip dispatch |
| `spammm/GUI/FFExtension.py` | §5 | Pin/Unpin → Rigid Manip dispatch |
| `spammm/GUI/ReactionCoordinateExtension.py` | §5 | RC pin → Rigid Manip dispatch |
| `user_guide/GUI_CHEATSHEET.md` | §5 | Document consolidated mode + extension modes |
| `doc/nonbonding_forcefields.md` | §2 | Document He/Hs provenance |

</details>
