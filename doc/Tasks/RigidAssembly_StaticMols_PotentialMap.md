---
type: Review
title: Review findings F1–F16 for Static rigid molecules + combined PairFF+FAF probe map
status: all 16 findings addressed — see ../Reports/StaticObstacle_DragDemo_2026-08-03.md §7–§8
tags: [review, rigid-body, PairFF, FAF, GUI, body-state, mixed-species]
timestamp: 2026-08-03
related: [RigidAssembly_StaticMols_PotentialMap-corrections.md, ../Reports/StaticObstacle_DragDemo_2026-08-03.md]
---

> **Note:** This file contains the original LLM review findings (F1–F16). The original
> implementation plan is preserved in
> [RigidAssembly_StaticMols_PotentialMap-corrections.md](RigidAssembly_StaticMols_PotentialMap-corrections.md).
> All 16 findings were addressed — see
> [../Reports/StaticObstacle_DragDemo_2026-08-03.md](../Reports/StaticObstacle_DragDemo_2026-08-03.md)
> §7 (findings) and §8 (USER-reported regressions).

## Review outcome

  The benzoic-acid demo produces a convincing visual result, but the generalized feature is still fragile. Several requirements work only because the script
  calls private helpers explicitly; the equivalent interactive GUI paths are incomplete. I also found two numerical mismatches that make the displayed field
  different from what the report claims.

  ### Critical findings

  1. The demo’s “O−” map is actually initialized as H+.

     The panel defaults to element H, R0=1.443, E0=0.00191, and Q=+0.40 in spammm/GUI/RigidAssemblyExtension.py:1157. The demo never activates
     _on_probe_preset(..., 'Om') before recomputing the map in demos/gui_scripts/static_obstacle_drag_demo.py:67.

     Therefore the artifact described as an O− probe map is initialized as H+. The proper O parameters are approximately R0=1.75, E0=0.002602, Q=-0.4.

  2. The PairFF part of the map does not use the simulated force-field parameters.

     compute_combined_probe_map() defaults to He=-1.0, Hs=0.0 in spammm/forcefields/RigidBodyUtils.py:196, while the actual assembly is built with He=-0.1,
     Hs=1.0 in spammm/GUI/RigidAssemblyExtension.py:471. The GUI passes only beta.

     Consequently, the displayed e-pair attraction is ten times the configured value and sigma-hole contributions are omitted. The visually plausible field is
     not the diagnostic field of the active PairFF configuration.

  3. Static/deleted state is ignored by MC.

     _on_mc_step() chooses step % nmol without filtering live dynamic bodies in spammm/GUI/RigidAssemblyExtension.py:557. Kernel 14 skips deleted interactions,
     but state 0 does not prevent a static body from being proposed and moved.

     Effects:
      - Static molecules can move through MC.
      - Deleted bodies consume MC turns and can receive new poses.
      - No-dynamic-body scenes do not fail with the required message.
      - Packing energy still includes deleted bodies.

  4. Soft deletion does not remove the molecule from rendering and can lead to contradictory state.

     _soft_delete_body() changes GPU state and ensemble.alive, but _assembly_world_atoms() and _display_index_to_body_site() continue iterating every body in
     spammm/GUI/RigidAssemblyExtension.py:121 and spammm/GUI/RigidAssemblyExtension.py:149.

     The deleted molecule therefore remains visible and pickable. Shift+LMB on it sets its GPU state back to dynamic because _toggle_body_state() interprets
     every non-1 state as static, but ensemble.alive remains false. This breaks the claimed ensemble SSOT.

  ### High-severity GUI and data issues

  5. Interactive map invalidation is not implemented.

     The report says toggle/delete/probe changes recompute the map, but:
      - _toggle_body_state() does not recompute it.
      - _soft_delete_body() does not recompute it.
      - R0/E0/Q/z controls have no change handlers.
      - The “Show map” checkbox has no signal connection and ra_show_map is never set; see spammm/GUI/RigidAssemblyExtension.py:1180.
      - Building from the regular GUI does not create the combined map.

     The demo works because it calls _recompute_ra_combined_map() manually after every scripted toggle.

  6. Required visualization features are absent.

     Static-body visualization is still a TODO in spammm/GUI/RigidAssemblyExtension.py:140. There is no e-pair checkbox or e-pair/sigma-hole overlay. This makes
     static bodies visually indistinguishable from dynamic bodies and leaves an explicit user requirement unimplemented.

  7. The graph synchronization “fix” remains unsafe.

     Element-name sequence is not a stable atom mapping. Two permutations of carbon or hydrogen atoms can have the same enames sequence, so spammm/GUI/
     RigidAssemblyExtension.py:170 can still miss reordered atoms.

     More importantly, editor fragment bonds are re-inferred from geometry in spammm/forcefields/RigidBodyUtils.py:118, instead of being derived from the
     authoritative AtomicGraph. Rebuilding the graph then discards stable atom/bond IDs, bond orders, charges, hybridization and other metadata.

     A robust solution is to preserve the original atom IDs and exact live bonds when producing fragments, then update display positions through that explicit
     mapping.

  8. Mixed-species factorized FAF remains wrong for equal-sized different species.

     _folded_plqh_all_sites() decides whether to reuse the first fit’s atom_plqh solely by atom count in spammm/forcefields/RigidBodyDynamics.py:2350. Two
     chemically different molecules with equal real-atom counts will receive the same PLQH array.

     The current four-species test avoids this because their sizes are 26, 22, 12 and 15 atoms. It therefore cannot expose the bug already identified in the
     task specification.

     The constant editor cache name editor_frag0 adds another collision: load_or_fit_faf() loads a cache without verifying molecular identity in spammm/
     surfaces/FoldedRigid.py:464.

  ### Performance opportunities

  9. Static and deleted workgroups still execute the full force calculation.

     Kernel 15 reads state_a in kernels/rigid.cl:4216, but uses it only at the integration step in kernels/rigid.cl:4413. Static/deleted bodies still calculate
     every PairFF interaction, FAF force, anchor contribution and reduction.

     This contradicts the planned uniform fast path and wastes most computation when many obstacles are frozen. A simple workgroup-uniform branch should copy
     the pose and zero dynamics for non-dynamic bodies, while only dynamic workgroups gather partner forces.

  10. Every mouse update downloads far more GPU data than needed.

     _sync_ensemble_from_gpu() calls download_outputs(), which transfers poses, velocities, all atom positions, atom forces, body forces and torques. Only
     positions and quaternions are required. The demo then downloads apos_world a second time in demos/gui_scripts/static_obstacle_drag_demo.py:230.

     Use download_selected(('pos', 'quats')) for display synchronization and perform at most one atom-position download when the demo needs its anchor.

  11. The CPU map path misses the agreed latency budget and has the wrong dependency direction.

     The “headless” helper imports computation from the Qt/VisPy viewer in spammm/forcefields/RigidBodyUtils.py:222. Its first dynamic compact-force import
     measured about 1.46 s here; subsequent import setup was below 1 ms, while the reported warm total remains about 254 ms.

     Because the measured warm time exceeds 0.2 s, a one-work-item-per-pixel OpenCL evaluator is now justified. Before that, move the compact map computation
     out of the GUI module and reuse a generic raw-array-to-VisPy-image helper.

  12. Map extents are unstable and clip the dynamic scene.

     With static bodies, the extent is derived only from their sites; with no static bodies it is derived only from body CoMs in spammm/forcefields/
     RigidBodyUtils.py:225. Toggling state therefore changes the extent, and the moving molecule can leave the map.

     This is visible in debug/static_obstacle_drag_demo/frame_first.png: the upper dynamic molecule extends outside the field. Extent should come from all live
     real atoms—or a fixed scene grid—while only static atoms contribute energy.

  ### Demo and test fragility

  13. The reported energy trace is not a valid total energy.

     Kernel 15 stores each body’s complete active-versus-partner energy in every site’s .w channel in kernels/rigid.cl:4395. Summing apos_world[:,3] over all
     bodies in demos/gui_scripts/static_obstacle_drag_demo.py:151 counts intermolecular pair energies twice, while FAF/anchor terms are counted once.

     The qualitative trajectory may still be useful, but the values reported as eV totals are not physically consistent.

  14. The test suite overstates its coverage.

     Examples:

  - Frozen FIRE state is claimed but never downloaded in tests/test_body_state.py:51.
  - Deletion parity checks only live positions, not forces/energy.
  - Map “invalidation” directly recomputes twice; it never tests the GUI cache, state toggles, probe changes or call counts.
  - “Gesture” tests call helper functions directly rather than exercising picking/events.
  - Mixed FAF tests check finiteness, not PLQH or force/energy parity.
  - Both mixed ordering tests use nmol=1, making interleaved ordering trivial.

      Test collection finds 22 tests, not the 21 stated in the report.

  15. The old mixed-species testplot still duplicates construction and has mismatched ordering.

     It constructs molecules grouped by species but bonds and tids cycling by species in tests/testplot_pairff_energy_mc.py:234 and tests/
     testplot_pairff_energy_mc.py:317. For nmol>1, templates, names and bonds disagree. It should call the new shared builder.

  16. Documentation is inconsistent with the actual demo.

     The GUI-script README still describes NTCDI+uracil and documents a nonexistent --nmol argument in demos/gui_scripts/README.md:46. The report’s cache-
     invalidation and O− claims are also incorrect.

  ## Verification performed

  - Python compilation passed for all modified implementation/test files.
  - Three available non-GPU GUI/helper tests passed.
  - Targeted collection found 22 tests.
  - I could not independently rerun the OpenCL tests: the sandbox exposes no OpenCL platforms, and the required NVIDIA escalation timed out twice. I did not
    substitute CPU/PoCL results.

  - git diff --check HEAD^ HEAD reports one whitespace error in doc/Caveats.md.

  No files were changed during this review.

  Recommended repair order: correct the probe and map parameters first; centralize state transitions and wire GUI invalidation/rendering; make MC respect live-
  dynamic bodies; fix topology identity and per-pack PLQH; then optimize kernel workgroups, downloads and the map evaluator.
