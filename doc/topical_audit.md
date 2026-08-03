# Topical Audit — SPAMMM

## Purpose

SPAMMM (Scanning Probe AFM/STM Molecular Modeling) is a Python + PyOpenCL scientific simulation package. This file is the index for topic-based audits — each section maps where a scientific concept is implemented, its status, and relationships to other topics.

## Technology Stack

- **Language**: Python (orchestration) + OpenCL `.cl` kernels (compute)
- **Acceleration**: PyOpenCL (GPU), NumPy (CPU vectorized)
- **GUI**: VisPy (Python, OpenGL)
- **QM**: DFTB+ (subprocess/C-API), pySCF
- **No C++, no JavaScript, no Fortran, no CUDA** — all compute is Python + OpenCL

## Content Guidelines

For each topic, document:

- **Overview** — What is the scientific problem being solved?
- **Implementations** — List all locations where this topic appears (file paths)
- **Status** — Active, experimental, deprecated, or unfinished
- **Relationships** — How different implementations relate
- **Notes** — Context about why multiple implementations exist, which to use, consolidation plans

## Usage

When working on a feature:
1. Check this file first to see if related work already exists
2. Update the relevant section when adding new implementations
3. Mark old implementations as deprecated when superseded
4. Add new sections for novel scientific areas

## Scope

This is **not** a replacement for:
- API documentation (use docstrings)
- Build instructions (use README)
- Tutorial examples (use `tests/`)

This **is** a supplement to:
- Help navigate the codebase
- Preserve context for long-term project continuity
- Enable systematic consolidation of duplicate efforts

---

# TOPIC HIERARCHY

## 1. Molecular Topology

**SSOT:** `AtomicGraph` (`spammm/topology/AtomicGraph.py`) — all other representations derive from it. See skill:`molecular-structure-sync`.

### 1a. Base Graph & Connectivity
- Graph representations, bond finding, neighbor lists, ring/bridge detection
- **Key files:** `spammm/topology/AtomicGraph.py`, `spammm/AtomicSystem.py`, `spammm/atomicUtils.py`
- **Audit Document:** [molecular_topology.md](molecular_topology.md)

### 1b. Type Assignment
- Atom type assignment (sp1/sp2/sp3), VSEPR geometry, parameter loading
- **Key files:** `spammm/topology/FFparams.py`
- **Audit Document:** [molecular_topology_types.md](molecular_topology_types.md)

### 1c. Kekule Bond Order Solver
- Pi-bond order optimization, aromatic detection, feasibility precheck, multi-seed localization with 6-ring validation
- **Key files:** `spammm/topology/KekulePure.py`, `spammm/topology/MoleculeEditorBackend.py`, `spammm/GUI/KekuleExtension.py`
- **Tests:** `tests/topology/test_kekule.py`, `tests/topology/test_editing_ops.py`, `tests/topology/testplot_kekule.py`
- **Artifacts:** `debug/test_kekule/` (develop-mode PNGs)
- **Caveats:** default seed can yield valence-valid but chemically invalid patterns — use `localize_kekule(validate=True, ntrials=5)`; some ASCII examples impossible (odd π count)

### 1d. Heterocycle Generation & ASCII Art
- Heterocycle structure generation and ASCII art → 2D geometry; `:` H-bond markers
- **Key files:** `spammm/topology/heterocycle_generator.py`, `spammm/topology/ascii_art_heterocycle.py`, `spammm/GUI/AsciiArtExtension.py`
- **Tests:** `tests/topology/test_heterocycle_generator.py`, `tests/topology/test_ascii_art.py`, `tests/topology/test_kekule.py`

### 1e. H-Bond Proton Transfer (DFTB rigid scan)
- Move bridging H along donor→acceptor axis; DFTB+ single-point energy profile (ASCII pipeline)
- **Key files:** `spammm/quantum/hbond_scan.py`, `spammm/quantum/DFTB_utils.py` (`run_dftb_sp`, `save_xyz_movie`)
- **Topology deps:** `resolve_hbond_pairs()`, `build_ascii_hbond_system()` from ASCII examples with `:`
- **Tests:** `tests/topology/test_hbond_scan.py` (L0 + slow DFTB), `tests/topology/testplot_hbond_scan.py`
- **Artifacts:** `debug/test_hbond_scan/hbond_*.{png,xyz}` — XYZ comment: `s=… f=… E=… dE=…`
- **Caveats:** 0.1 Å path grid default; rigid geometry; SCC failures skipped; uses `$DFTB_EXE` not PATH `dftb+`
- **Superseded for graph/GUI work by:** §1h Reaction-coordinate scan

### 1h. Reaction-Coordinate Scan (graph + GUI + pm-NEB)
- Multi-H-bond control grids, `ScanDataset` trajectories, DFTB relax at endpoints, Mulliken charges, ESP animation
- **Key files:** `spammm/quantum/coordinate_scan.py`, `spammm/topology/scan_dataset.py`, `spammm/topology/hbond_utils.py`, `spammm/quantum/esp_grid.py`, `spammm/GUI/ReactionCoordinateExtension.py`, `spammm/GUI/rc_esp_view.py`, `spammm/GUI/mpl_blit.py`
- **DFTB:** `run_dftb_relax`, `run_dftb_sp(return_charges=True)`, `parse_mulliken_charges`, `clean_dftb_workdir`
- **GUI scripts:** `demos/gui_scripts/rc_scan_review.py`, `rc_scan_offline.py`; launcher `./run_gui.sh --script …`
- **Audit document:** [Topics/ReactionCoordinateScan.md](Topics/ReactionCoordinateScan.md)
- **Tests:** `tests/topology/test_scan_dataset.py`, `tests/GUI/test_rc_scan_gui_script.py`
- **Artifacts:** `debug/rc_scan/`, `debug/testplot_rc_scan_gui/*.npz`, `debug/rc_scan_offline/`
- **Caveats:** GUI molecule build must match `build_ascii_hbond_system` (not `.strip()` ASCII alone); `endpoints_relaxed` false if DFTB skip; blit rules in [Takeways.md](Takeways.md) and `mpl_blit.py`

### 1f. Molecular Editor (GUI backend)
- Hex-grid editing, passivation, ring ops — **not** the Kekule bond-order solver
- **Key files:** `spammm/topology/MoleculeEditorBackend.py`, `spammm/GUI/SPAMMM_GUI.py` (`SPAMMMWindow`), `spammm/GUI/EditModeHandlers.py`
- **Legacy alias:** `KekuleExplorerWindow` → `SPAMMMWindow`
- **Ring placement (3 modes, unified in `RingMode`):**
  - **Edge ring** (`add_adjacent_ring`): n-gon sharing 1 existing bond as edge; side determined by mouse position
  - **Corner ring** (`add_corner_ring`): n-gon sharing 2 existing bonds at an inner corner atom (angle < 180°); uses circumcircle through B-A-C, vector-math only (dot/cross products, no atan2); outer corners (>180°) fall through to edge/hex method; mouse direction selects which neighbor pair
  - **Hex-grid ring** (`add_ring`): snap to hex grid center, creates hexagonal ring on grid
  - Priority in RingMode: bond → corner atom → hex center
- **Draw demos:** shared sequence → SVG / GIF — [Topics/GUI_DrawDemo_Scripts.md](Topics/GUI_DrawDemo_Scripts.md)
- **Grid transforms:** transpose, flip X, flip Y — apply to both grid and atom geometry (`transform_atoms`)
- **Tests:** `tests/topology/test_editing_ops.py`, `tests/test_export_import.py`

### 1g. Editors & GUI
- Interactive molecular editor: VisPy-based GUI with topology editing
- **Key files:** `spammm/GUI/SPAMMM_GUI.py`, `spammm/GUI/VispyUtils.py`, `spammm/GUI/KekuleExtension.py`, `spammm/GUI/ReactionCoordinateExtension.py`, `spammm/GUI/mpl_blit.py`
- **Audit Document:** [molecular_topology_editors.md](molecular_topology_editors.md)
- **Feature audit:** [gui_audit.md](gui_audit.md)
- **Developer notes:** [Takeways.md](Takeways.md) (matplotlib blit, GUI vs test geometry)

### 1i. GUI Draw Demo Scripts (SVG + GIF)
- Shared editor step sequence (hex → fused pent → N → H → copy/paste → δ/φ dimer) for presentation and systematic GUI debugging
- **SSOT sequence:** `spammm/GUI/azaindol_draw_sequence.py` (`run_azaindol_draw`, `render_editor_svg`)
- **Helpers:** `spammm/GUI/gui_script_utils.py` (`apply_demo_overlays`, `capture_window_png` GL composite, `frames_to_gif`)
- **GUI scripts:** `azaindol_draw_demo.py` (live window PNG → GIF), `azaindol_draw_offline.py` (headless SVG)
- **Audit document:** [Topics/GUI_DrawDemo_Scripts.md](Topics/GUI_DrawDemo_Scripts.md)
- **Artifacts:** `debug/azaindol_draw_demo/` (PNG+GIF), `debug/azaindol_draw_offline/` (SVG+PNG)
- **Caveats:** never empty `ring_preview_line` then refill (VisPy segfault); window grab must composite `canvas.render()` into Qt pixmap

## 2. Force Fields

### 2a. Intramolecular Force Fields
- UFF (universal), SPFF (sp3 with pi-orbital nodes), **LFF** (projective Jacobi springs), rigid body dynamics
- **Key files:** `spammm/forcefields/UFF_cl.py`, `spammm/forcefields/SPFF_cl.py`, `spammm/forcefields/LFFSolver.py`, `spammm/forcefields/UFFbuilder.py`, `spammm/forcefields/SPFFbuilder.py`, `spammm/forcefields/RigidBodyDynamics.py`
- **Kernels:** `kernels/UFF.cl`, `kernels/SPFF.cl`, `kernels/LFF.cl`, `kernels/rigid.cl`
- **Controller:** `spammm/forcefields/FFController.py` (`ff_type` in `{spff,uff,lff}`)
- **Audit Documents:** [forcefields_overview.md](forcefields_overview.md), [intramolecular_forcefields.md](intramolecular_forcefields.md), [Topics/ForceFields/LFF_ProjectiveRelax.md](Topics/ForceFields/LFF_ProjectiveRelax.md)
- **Tests:** `tests/test_forcefield.py`, `tests/test_relax_serial.py`, `tests/test_relax_ptcda_faf.py`
- **Caveats (LFF):** surrogate springs (not energy-parity with UFF/SPFF); K₁₄ caps mandatory; uniform mass=1; WG≤64 atoms

### 2b. Non-Bonding Force Fields
- LJ/Morse/Coulomb, exclusion schemes
- **Kernels:** `kernels/nonbonded.cl`, `kernels/nonbonded_grid.cl`
- **Audit Document:** [nonbonding_forcefields.md](nonbonding_forcefields.md)

### 2b2. PairFF (rigid-body intermolecular)
- Compact-exp / Morse pairwise FF with epair + σ-hole dummies; shared allmol GPU buffers; active-only and concurrent all-mobile 6-DOF dynamics; optional fused FAF substrate; **energy-only replica kernel** for MC/GA optimizers; **per-body static/dynamic/deleted state gating** (kernel 14+15); **GPU probe-map kernel** (kernel 18) sharing the same inline `pairff_unified_site_EF` primitive as dynamics — codepath parity
- **Key files:** `spammm/forcefields/RigidBodyDynamics.py` (`RigidBodyPairFF`, `eval_probe_grid_gpu`), `spammm/forcefields/RigidBodyUtils.py` (`build_mixed_species_assembly`, `compute_combined_probe_map`, `nuclear_exclusion_mask`, `plan_probe_grid`), `spammm/GUI/RigidBodyVispy.py` (`potential_to_rgba`), `spammm/surfaces/FoldedRigid.py`, `demos/demo_pairff.py`
- **Kernels:** `kernels/rigid.cl` (7–13: allmol ± FAF MD; **14: `rigid_body_pairff_energy_replica_kernel`** — energy + fused clash flags, 2D launch `(nactive*64, n_replica)`, WG=64, float4 channels, ~3.1 KiB LM; **15: `rigid_body_pairff_multimol_kernel`** — concurrent all-molecule MD, ping-pong I/O, `niter` loop, optional `predict_partners`, **`body_state` gate** (static=skip integration, deleted=skip partner loop); **16: `rigid_body_pairff_multimol_persistent_kernel`** — same + atomic global barrier (experimental, requires all WGs resident); **17: `rigid_body_pairff_multimol_single_wg_kernel`** — single-WG Jacobi, exact but 1-CU only; **18: `rigid_body_pairff_probe_grid`** — 2D grid PairFF energy eval using shared `pairff_unified_site_EF` inline primitive, same params as dynamics), `kernels/Forces.cl` (`compact_exp_pair_EF`)
- **Body-state API:** `RigidBodyPairFF.set_body_state(i, state)` / `set_body_states(states)` — state: +1=dynamic, 0=static, <0=deleted. `RigidEnsemble.active/alive` is the high-level SSOT; `_body_state_host` is the GPU mirror.
- **Probe map:** `compute_combined_probe_map` returns `(E_sum, E_pairff, E_faf, xs, ys, extent, exclude_mask)`. Uses exact `beta` from `rbd.pairff_params_host` (no display-only substitution). `exclude_mask` (True within 1 Å of real atoms) used only for `vmin/vmax` estimation — map is fully finite everywhere. Color scale rule: `vmin=Emin, vmax=|Emin|` (symmetric), see `.devin/skills/centralized-plotting/SKILL.md`.
- **User manual:** [demos/PairFF_manual.md](../demos/PairFF_manual.md)
- **Topical audit:** [TopicalAudit/PairFF_RigidBody.md](TopicalAudit/PairFF_RigidBody.md) (force model); [TopicalAudit/RigidBody.md](TopicalAudit/RigidBody.md) (pose SSOT across modules)
- **Design report:** [Topics/ForceFields/PairFF.md](Topics/ForceFields/PairFF.md)
- **Tasks:** [Tasks/PairFF_MultiBody_Kernel.md](Tasks/PairFF_MultiBody_Kernel.md) (Done), [Tasks/PairFF_FAF_Substrate.md](Tasks/PairFF_FAF_Substrate.md) (Done), [Tasks/PairFF_RigidEnergy_MC_GA.md](Tasks/PairFF_RigidEnergy_MC_GA.md) (K1+P1+H1+F done; H2 SA pending), [Tasks/PairFF_MC_PythonBottleneck.md](Tasks/PairFF_MC_PythonBottleneck.md) (implementation tested; 79–89 → 0.66 ms/step, awaiting USER confirmation), [Tasks/RigidMoleculePose_SSOT.md](Tasks/RigidMoleculePose_SSOT.md) (design locked), [Tasks/MultiMol_MD_LaunchOverhead.md](Tasks/MultiMol_MD_LaunchOverhead.md) (verified — ping-pong concurrent MD, 4–135× speedup, awaiting integration), [Tasks/RigidAssembly_StaticMols_PotentialMap.md](Tasks/RigidAssembly_StaticMols_PotentialMap.md) (implemented — static/dynamic/deleted state, mixed-species, combined probe map; USER confirmed 2026-08-03), [Tasks/RigidAssembly_Demo_MapMode_Consolidation.md](Tasks/RigidAssembly_Demo_MapMode_Consolidation.md) (implemented — GPU probe kernel, honest params, consolidated Manipulate mode, two demo variants, C1–C3 corrections; L2 review in progress)
- **Driver:** `tests/testplot_pairff_energy_mc.py` → `debug/testplot_pairff_energy_mc/<mol>/` (8 molecules: PTCDA, formic acid, terephthalic acid, NTCDI, TBTAP, azaindol, uracil, adenine; **multi-species** `--mol adenine,uracil`; FAF substrate, charge colors, GIF trajectory)
- **Body-state tests:** `tests/test_body_state.py` (10 L0 tests: all-dynamic parity, frozen invariant, static interaction, deletion parity k14+k15, mixed-species FAF, map decomposition + invalidation, nmol=2 ordering, static force=0)
- **Caveats:** multi-body/FAF need unified mode + NVIDIA; ≤128 sites/molecule; the main GUI uses `RigidEnsemble` with checked GPU/`_mb_*` mirrors; greedy MC stalls in deep minima (21–75/1000 accepted) — SA (H2) needed; kernel 15 supports concurrent FAF while persistent/single-WG variants remain restricted/experimental; **graph rebuild for display must check enames match, not just count** (see [Takeways.md](Takeways.md) → "Graph rebuild enames check"); **probe map color scale must exclude nuclear singularities (1 Å around real atoms) from vmin/vmax — see `.devin/skills/centralized-plotting/SKILL.md` §"Nuclear exclusion"**

### 2c. Charge Equilibration
- QEq charge transfer method
- **Key files:** `spammm/forcefields/QEq.py`, `spammm/GUI/QEqExtension.py`

### 2d. On-Surface Assembly (Rigid SAM)
- Rigid-body packing on a fixed hexagonal unit cell: 6 C6 orientations/cell, GPU clash scoring
- **Key files:** `spammm/forcefields/Assembly.py`, `spammm/forcefields/AssemblyPlot.py`
- **Kernel:** `kernels/assembly.cl`
- **Driver:** `tests/testplot_assembly.py` → `debug/testplot_assembly/`
- **FireCore reference:** `pyBall/OCL/Assembly.py`, `tests/tMMFF/test_assembly.py`
- **Caveats:** `radius=1.0` Å default (softened for rigid search); `align_flat` pre-aligns to surface; rotation dedup CPU-only, off by default
- **Tests:** visual demo only (`testplot_assembly.py`); no pytest L0 yet

### 2e. Vibrational Analysis (normal modes)
- Hessian → rigid-mode projection → frequencies and in-plane/out-of-plane classification
- **Key files:** `spammm/dynamics/Vibrations.py`, `spammm/dynamics/VibrationPlot.py`, `spammm/forcefields/FFEvaluator.py`, `spammm/GUI/VibrationExtension.py`
- **DFTB Hessian:** `spammm/quantum/DFTB_utils.py` (`write_dftb_input_hessian`, `read_hessian`)
- **Audit Document:** [Topics/Vibrations.md](Topics/Vibrations.md)
- **Tests:** `tests/test_vibrations.py` → `debug/test_vibrations/`
- **Caveats:** UFF/SPFF absolute frequencies not calibrated; SPFF freezes pi-orbitals in FD; phonon/PBC not implemented

## 3. Surface Interactions

### 3a. GridFF & Surface Potentials
- GridFF (B-spline interpolation), FoldedAtomicFunctions (compact basis), Ewald2D
- **Key files:** `spammm/surfaces/GridFF.py`, `spammm/surfaces/GridFFRelaxedScan.py`, `spammm/surfaces/FoldedRigid.py`, `spammm/surfaces/Ewald2D.py`, `spammm/surfaces/SurfaceEwald.py`, `spammm/surfaces/Surface_utils.py`, `spammm/surfaces/SubstrateBuilder.py`
- **Kernels:** `kernels/gridFF.cl`, `kernels/surface.cl`
- **Audit Document:** [surface_interactions.md](surface_interactions.md)
- **Consolidation needed:** `GridFF.py` vs `GridFFRelaxedScan.py` — overlapping functionality
- **Tests:** `tests/test_surface.py`, `tests/test_folded_relax.py`, `tests/surfaces/ocl_GridFF_new.py`

### 3b. Folded Rigid Body Manipulation (molecule-on-surface)

Interactive rigid-body dynamics of a small molecule on a periodic substrate using a pre-fitted folded basis expansion (Pauli/London/Coulomb). The GUI supports loading the molecule, fitting or loading the substrate potential, continuous relaxation, and LMB dragging of individual atoms.

| Location | Status | Notes |
|----------|--------|-------|
| `spammm/surfaces/FoldedRigid.py` | active | `setup_rigid_folded`, `relax_folded`, `fit_folded_for_molecule`, `RigidBodyDynamics` setup |
| `spammm/GUI/FoldedRigidExtension.py` | active | Extension panel, edit modes (`fr_pin`, `fr_com`, `fr_manip`), continuous timer, drag atom picking |
| `demos/gui_scripts/folded_rigid_setup.py` | active | One-command setup script (`--run`, `--manip`, `--fit`, `--mol`) |
| `spammm/forcefields/SPFF_cl.py` | active | `fit_folded_surface_basis` with `coulomb_solver='ewald2d'` |
| `spammm/forcefields/RigidBodyDynamics.py` | active | OpenCL rigid-body state, `run_folded`, `update_anchors` |
| `tests/testplot_ptcda_nacl_replicas.py` | experimental | 256×256 pinned-PTCDA imaging; staged damped Newton, FIRE comparison, constraint-reaction maps |
| `doc/Reports/PTCDA_NaCl_Rigid_Newton_FIRE_Relaxation.md` | active | Solver rationale, continuity evidence, performance tradeoffs, reproduction, and caveats |
| `kernels/rigid.cl` | active | `rigid_body_folded_kernel` folded basis force/torque + anchor springs |
| `spammm/GUI/SPAMMM_GUI.py` | active | Mouse dispatch (`on_mouse_press/move/release`), `refresh_view` no-bonds fix, extension integration |
| `spammm/GUI/EditModeHandlers.py` | active | `on_move` ray-origin/direction, `on_release` hook base class |
| `spammm/GUI/RigidAssemblyExtension.py` | active | Unified rigid-body GUI: checked ensemble↔GPU↔host pose sync; dense-scene→dummy-site anchor mapping; kernel-15 all-mobile PairFF+FAF drag; MC/GA + PME. L0: `tests/GUI/test_rigid_assembly_extension.py` |
| `spammm/forcefields/molecule_loaders.py` | active | Shared rigid-body molecule loaders (XYZ/mol2 + QEq + bonds); reused by `testplot_pairff_energy_mc.py` + `RigidAssemblyExtension` |
| `data/fits/h2o_nacl.npz` | data | Cached H2O/NaCl folded basis fit (`nu=4, nv=4`) |

**Key design points:**
- `AtomicGraph` is the SSOT for atom positions; `RigidBodyDynamics` GPU `poss`/`qrots` remain per-algorithm working storage. **Shared rigid-pose rep now exists:** `spammm/forcefields/RigidEnsemble.py` — optional, numpy-only, poses-only (`pos`+`qrot` per molecule, stable ids). Rigid modules import it and read `get_poses()`; one-way `ensemble → AtomicGraph` on demand. See [TopicalAudit/RigidBody.md](TopicalAudit/RigidBody.md), [Tasks/RigidMoleculePose_SSOT.md](Tasks/RigidMoleculePose_SSOT.md).
- `_update_graph` rebuilds the backend graph if the atom count diverges between `backend` and `rbd`.
- `FRManipMode` uses `AtomScene._pick_id_from_mouse` for screen-space atom picking and ray-projection for anchor targets.
- Default dynamics: `k=2.0`, `dt=0.02`, `n_iter=250`, adaptive `Run` timer `max(20, 0.1·n_iter)` ms.
- Folded setup scales `I` and `I⁻¹` consistently with the requested effective mass (`I∝m_eff`, `I⁻¹∝1/m_eff`); PTCDA imaging uses `m_eff=4` for FIRE preconditioning.
- Stable float32 imaging Newton uses centered `eps=0.1`, trust `0.1`, and a staged LM floor (`1.0` for basin following, then `0.01` for finishing); persistent state across launches is mandatory.
- Plot the spring reaction `F_constraint=+k(r_pin-r_anchor)` as the tip observable; total force is a convergence residual and approaches zero.

**GUI scripts:**
```bash
./run_gui.sh --script demos/gui_scripts/folded_rigid_setup.py -- --mol data/xyz/H2O.xyz --fit data/fits/h2o_nacl.npz --run
./run_gui.sh --script demos/gui_scripts/folded_rigid_setup.py -- --mol data/xyz/H2O.xyz --fit data/fits/h2o_nacl.npz --manip
./run_gui.sh --script demos/gui_scripts/ptcda_interactive_drag.py
```

**Tests:** `tests/surfaces/test_folded_relax.py` (smoke), manual `Run`/`drag` L2 review.

**Open issues:** `Run` speed is GPU-timer-limited; the `Run` timer interval is auto-scaled with `n_iter`. RigidAssembly has numerical L0 coverage for all-mobile anchored drag, but visible NaCl stick-slip awaits USER review. Newton translation/rotation coordinates are not nondimensionalized; the replica kernel uses centered Hessian differences while the single-body workgroup kernel still uses forward differences.

## 3c. Contact Surface (quasi-2D static AFM)

Compact alternative to 3D `img_FF` for aperiodic rigid PP-AFM: **separable B-spline×poly**
(global corrugation) and **radial PIC** (per-atom compact support). **Design intent:** atom-scale
nodes (`bspl_dx~1 Å`), sphere-envelope `h₀` with `h0_R_scale~0.75`.

**Design:** [Topics/AFM/ContactSurface_Static.md](Topics/AFM/ContactSurface_Static.md)  
**Parity vs GridFF:** [Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md](Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md)  
**Caveats:** [Caveats.md](Caveats.md) §6 · **Pitfalls:** [Takeways.md](Takeways.md)  
**Module README:** [spammm/surfaces/README.md](../spammm/surfaces/README.md) · **Audit:** [TopicalAudit/AFM_ContactSurface.md](TopicalAudit/AFM_ContactSurface.md)

| Location | Status | Notes |
|----------|--------|-------|
| `spammm/surfaces/ContactSurface.py` | active | Sphere `h₀`, separable/PIC CG |
| `kernels/contact_surface.cl` | active | Brute, Av/Atv, PIC, PP contact |
| `spammm/SPM/AFM.py` | active | `fit_contact_surface`, `run_scan_contact` |
| `run_assembly_afm.py` | experimental | Coarse defaults + `--compare-dir` |
| `spammm/surfaces/GridFF.py` | active | Dense 3D classical reference |

**Parity:** E/Fz profiles vs brute improved (2026-07-24); XY vs GridFF still sharper — USER visual pending.  
L0: `tests/SPM/test_afm_contact_surface.py`. L2: `tests/testplot_contact_surface.py`, assembly compare.

**Task SSOT:** [Tasks/Fast_2p5D_AFM_ContactSurface.md](Tasks/Fast_2p5D_AFM_ContactSurface.md)

**Open issues:** USER confirm maps; residual XY sharpness; PIC re-validate; pipeline flag; elastic Phase 2.

## 4. AFM/STM Simulation

### 4a. AFM Simulation (FDBM = GUI engine)
- Morse/LJ (tests/scripts) vs **FDBM** ModularPipeline S1–S6 (GUI). Round-1+2 perf: GPU tasks/FFT, dense NA, uncompressed cache, fused ES + GPU pad/scale (`SPAMMM_AFM_FAST_S3=1` default).
- **Key files:** `spammm/SPM/AFM.py`, `spammm/SPM/AFM_utils.py`, `spammm/SPM/ModularPipeline.py`, `spammm/SPM/ScanUtils.py`, `spammm/SPM/ManipulationPathOpt.py`
- **Folder README:** `spammm/SPM/README.md`
- **Kernel:** `kernels/AFM.cl` (incl. `fdbm_*` Stage-3 helpers)
- **GUI:** `spammm/GUI/AFMExtension.py`
- **Density:** `spammm/quantum/DFTB/Grid_dftb.py`
- **CLI:** `run_spm.py` — FDBM defaults amp-align (df @ h, Fz @ h−amp); dense dz=0.1; `opt` / `smiles-afm` planar+PCA. Guide: [`user_guide/SPM_CLI.md`](../user_guide/SPM_CLI.md).
- **Perf / tests:** `doc/Tasks/PerfBenchmark_FDBM.md`, `tests/SPM/bench_fdbm.py`, `tests/SPM/test_afm_morse.py`, `tests/SPM/test_afm_fdbm.py`
- **Topical audit:** [TopicalAudit/AFM_FDBM.md](TopicalAudit/AFM_FDBM.md)
- **Overview doc:** [afm_stm_simulation.md](afm_stm_simulation.md)
- **Rigid body AFM:** `spammm/forcefields/RigidBodyAFM.py`
- **Caveats:** K_LAT N/m vs eV/Å²; prefer `step ≤ 0.1 Å` (`doc/Tasks/AFMTesting.md`)

### 4b. STM Simulation
- LCAO orbital projection, spectral function / BR-STM; vacuum tails need prolonged basis
- **Kernels:** `kernels/LCAO_grid.cl` (overlap_exp + projectors), `kernels/LCAO_STM_FGR.cl` (FGR \(H-ES\)), `kernels/LCAO_STM.cl` (Dyson/GF later)
- **Python:** `AFM_utils.compute_stm`, `compute_bond_resolved_stm`, `project_mo_stm_fgr_slice`; `Grid_dftb` orbital projectors + FGR host; ModularPipeline S6; `stm_compare.run_fgr_transfer_compare`
- **CLI:** `run_spm.py stm {orbitals,current,panel,br,fgr}`
- **Task (campaign):** [Tasks/STM_ExtendedBasis_OrbitalCompare.md](Tasks/STM_ExtendedBasis_OrbitalCompare.md) — mio vs 3ob vs prolonged vs pySCF HOMO/LUMO cubes (pentacene, PTCDA)
- **FGR transfer (\(M=H-ES\)):** [TopicalAudit/STM_FGR_Transfer.md](TopicalAudit/STM_FGR_Transfer.md) · report [Reports/STM_FGR_Transfer_H_ES_2026-07-29.md](Reports/STM_FGR_Transfer_H_ES_2026-07-29.md) · wiring [Ideas/LCAO_STM_FGR_WIRING.md](Ideas/LCAO_STM_FGR_WIRING.md) · artifacts `debug/stm_fgr_compare/`
- **Related:** prolonged WFC [Tasks/ProlongedRadialBasis_DFTB.md](Tasks/ProlongedRadialBasis_DFTB.md); Dyson later [Tasks/DysonOrbitals_DFTB_STM.md](Tasks/DysonOrbitals_DFTB_STM.md)
- **Audit Document:** [afm_stm_simulation.md](afm_stm_simulation.md) (STM sections)
- **Caveats:** no dedicated L0 STM FGR pytest yet; GUI STM/Orbitals widgets often empty; Level B EH makes \(H\propto S\) (Level A unfinished)

### 4c. Kriging GridFF ↔ FDBM Pauli
- DFT z-scan → Wendland Kriging/RBF GridFF; compare to FDBM Pauli/ES/vdW; site Pauli fits
- **Key files:** `spammm/SPM/KrigingGridFF.py`, `InterpolatorKriging.py`, `AFM_utils` Pauli fitters
- **Tasks:** [Import_KrigingGridFF.md](Tasks/Import_KrigingGridFF.md), [Pauli_A_beta_KrigingTransferability.md](Tasks/Pauli_A_beta_KrigingTransferability.md) (site maps + transferability)
- **Reports:** [Reports/Kriging_DFT_vs_DFTB_FDBM_pyridine.md](Reports/Kriging_DFT_vs_DFTB_FDBM_pyridine.md), [Kriging_FDBM_PauliFit_pyridine_2026-07-21.md](Reports/Kriging_FDBM_PauliFit_pyridine_2026-07-21.md), [Fukui_FDBM_panel_notes_2026-07-23.md](Reports/Fukui_FDBM_panel_notes_2026-07-23.md)
- **Caveats (global):** [Caveats.md](Caveats.md) — aniso cubes; AE core alias → **element-invariant** clamp NA; node vs center project; sample vs project
- **ES dipole report (SSOT):** [Reports/Cube_ES_DeltaRho_NA_dipole.md](Reports/Cube_ES_DeltaRho_NA_dipole.md) · handoff [Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md](Reports/Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md) (implemented; USER visual pending)
- **Topic:** [Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md](Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md)

### 4d. Charge rings (Pauli Master Equation)
- Many-body site occupations → STM \(I\), \(dI/dV\) (xy rings, xV diamonds, NDR)
- **Kernel:** `kernels/PME.cl` (4 sites / 16 states)
- **Python:** `spammm/quantum/PauliSolverCL.py`, `pauli_scan.py`
- **Data:** `data/charge_rings/` (Ruslan_*, fig3_trimer.json)
- **Demos:** `tests/quantum/testplot_charge_rings_{ruslan,trimer}.py` → `debug/testplot_charge_rings_*/`
- **L0:** `tests/quantum/test_pme_trimer.py` — symmetric trimer mirror + NDR
- **GUI:** `spammm/GUI/ChargeRingsExtension.py` — JSON I/O, Calc XY/xV/1D, cut line, state probs
- **Topical audit:** [TopicalAudit/ChargeRings_PME.md](TopicalAudit/ChargeRings_PME.md)
- **Task:** [Tasks/Import_ChargeRings_PME.md](Tasks/Import_ChargeRings_PME.md) — A+D+F Done; Hubbard/MQCA/MC-fit Later
- **Pose follow-on:** sites = rigid molecules — [TopicalAudit/RigidBody.md](TopicalAudit/RigidBody.md), [Tasks/RigidMoleculePose_SSOT.md](Tasks/RigidMoleculePose_SSOT.md) (design only)
- **Caveats:** pad n&lt;4 with spectators + Wij; tip μ from Vtips; Ruslan Qzz=10 vs fig3 Qzz=0 (NDR); sites still abstract (circle/JSON) until pose SSOT

## 5. QM Integration (DFTB+)

- DFTB+ integration: subprocess, C-API, parsers, OpenCL grid projection, constrained scans
- **Key files:** `spammm/quantum/DFTB/DFTBcore.py`, `spammm/quantum/DFTB/DFTBplusParser.py`, `spammm/quantum/DFTB/Grid_dftb.py`, `spammm/quantum/DFTB/basis_optimizer.py`, `spammm/quantum/DFTB_utils.py`, `spammm/quantum/hbond_scan.py`, `spammm/quantum/pySCF_utils.py`
- **Hessian (vibrations):** `DFTB_utils.write_dftb_input_hessian` — used by `dynamics/Vibrations.py`
- **Kekulé RI density (planned):** [Tasks/Kekule_ExponentialDensityFit.md](Tasks/Kekule_ExponentialDensityFit.md) — π orders → atom+bond exponentials vs DFT ρ
- **Audit Document:** [afm_stm_simulation.md](afm_stm_simulation.md) (DFTB sections)
- **Tests:** `tests/SPM/plot_dftb_vs_pyscf_basis.py`, `tests/SPM/plot_3ob_basis_tails.py`, `tests/topology/test_hbond_scan.py`

## 6. GUI & Visualization

- Main GUI: VisPy-based molecular editor with extension plugins
- **Key files:** `spammm/GUI/SPAMMM_GUI.py`, `spammm/GUI/BaseGUI.py`, `spammm/GUI/GLGUI.py`, `spammm/GUI/VispyUtils.py`, `spammm/GUI/MoleculeViewer.py`, `spammm/GUI/MolecularBrowser.py`, `spammm/GUI/ExtensionManager.py`, `spammm/GUI/EditModeHandlers.py`
- **Edit modes:** Unified (atom/bond/hex/empty), Atom, Bond, Ring (edge+corner+hex), Hex (paint/toggle), pi, Select
- **Extensions:** `KekuleExtension.py`, `AFMExtension.py`, `FFExtension.py`, `QEqExtension.py`, `VibrationExtension.py`, `RigidAssemblyExtension.py`, `FoldedRigidExtension.py`
- **Rigid Assembly extension:** `RigidAssemblyExtension.py` — mixed-species build (comma-separated names or "From editor" via connected components), per-body static/dynamic/deleted state (Shift+LMB toggle, RMB soft-delete), combined PairFF(static)+FAF probe map overlay (GPU kernel 18, honest params from `rbd.pairff_params_host`), probe controls (H+/O− presets, R0/E0/Q/z_probe), map layer combo (Total/PairFF/FAF), color-limit spin (Auto=`|Emin|` with nuclear exclusion), margin spin, state counts label. **Consolidated `Manipulate` mode** (§5 of consolidation task) replaces RA Drag/FR Pin/FR COM/FR Manip/Pin-Unpin/RC pin. **Caveat:** `_ensure_backend_matched` must check enames match, not just count — see [Takeways.md](Takeways.md).
- **Substrate visualization (FAF potential map):** `VispyUtils.update_faf_map_overlay` — shared function reusing `RigidBodyVispy.potential_to_rgba` (display SSOT, vmax=|Emin|) + `FoldedRigid.eval_folded_potential_grid`. Used by `FoldedRigidExtension`, `RigidAssemblyExtension`, and `ptcda_drag_demo.py`. Same visualization as `demos/demo_pairff.py`.
- **Anchor visualization (drag spring):** `RigidAssemblyExtension._update_anchor_visuals` — red line (atom→target) + red cross marker (at target). Wired into `RAManipMode.on_press/on_move/on_release` for interactive + script dragging.
- **GUI scripting:** `gui_script_runner.py` + `gui_script_utils.py` (GSU). Scripts call the same functions as GUI buttons. Scripts centralized in `demos/gui_scripts/` (auto-discovered by Scripts → Bundled menu). See [GUI_Scripting_DemoRunner.md](Tasks/GUI_Scripting_DemoRunner.md), [GUI_Scripts_Consolidation_2026-08-01.md](Reports/GUI_Scripts_Consolidation_2026-08-01.md).
- **Animation export:** `frames_to_gif` (PIL) + `frames_to_video` (ffmpeg H.264 `-tune animation`, 11x smaller than GIF). Supports MP4/WebM/AV1.
- **Design docs:** [GUI.desing.md](GUI.desing.md), [GUI_FF_Relaxation.md](GUI_FF_Relaxation.md), [GUI_topology_edit.desing.md](GUI_topology_edit.desing.md)
- **Audit Document:** [molecular_topology_editors.md](molecular_topology_editors.md), [gui_audit.md](gui_audit.md)
- **Reports:** [PTCDA_DragDemo_StickSlip_2026-08-01.md](Reports/PTCDA_DragDemo_StickSlip_2026-08-01.md) — drag demo with FAF substrate, anchor lines, trajectory visualization; [GUI_Scripts_Consolidation_2026-08-01.md](Reports/GUI_Scripts_Consolidation_2026-08-01.md) — scripts centralized to `demos/gui_scripts/` + MP4 export; [StaticObstacle_DragDemo_2026-08-03.md](Reports/StaticObstacle_DragDemo_2026-08-03.md) — dimer split via connected components, static/dynamic toggle, combined PairFF+FAF probe map; consolidated 2026-08-03 with GPU probe kernel, honest params, two demo variants, C1–C3 corrections (see [Tasks/RigidAssembly_Demo_MapMode_Consolidation.md](Tasks/RigidAssembly_Demo_MapMode_Consolidation.md))

---

# TOPIC DEPENDENCY GRAPH

```
              ┌─────────────────────────────┐
              │  Molecular Topology (1)      │
              │  AtomicGraph (SSOT)          │
              └──────────┬──────────────────┘
                         │
              ┌──────────▼──────────────────┐
              │  Type Assignment (1b)        │
              │  → FF Parameter Loading      │
              └──────────┬──────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────────┐
  │ Intra FF (2a)│ │ Non-Bond(2b)│ │ Surface (3)     │
  │ UFF, SPFF   │ │ LJ, Morse   │ │ GridFF, Ewald2D │
  │ RigidBody   │ │ Coulomb     │ │ FoldedRigid     │
  │             │ │             │ │ ContactSurface  │
  └──────┬──────┘ └──────┬──────┘ └──────┬──────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
              ┌──────────▼──────────────────┐
              │  AFM/STM Simulation (4)     │
              │  AFM → RigidBody → GridFF   │
              │  → Tip Relaxation → Image   │
              └──────────┬──────────────────┘
                         │
              ┌──────────▼──────────────────┐
              │  QM Integration (5)         │
              │  DFTB+ / pySCF              │
              │  → H/S matrices → orbitals  │
              └─────────────────────────────┘
```

**Key dependencies:**
- Topology (1) → Type Assignment (1b): graph structure determines hybridization
- Type Assignment (1b) → Force Fields (2): types determine FF parameters
- Force Fields (2) + Surface Interactions (3) → AFM/STM (4): AFM uses both
- QM Integration (5) → AFM/STM (4): STM requires DFTB orbital data
- GUI (6) → All topics: GUI visualizes and edits everything

---

# OPEN ISSUES & CONSOLIDATION OPPORTUNITIES

| Topic | Issue | Related Files | Priority |
|-------|-------|---------------|----------|
| **GridFF variants** | `GridFF.py` vs `GridFFRelaxedScan.py` — overlapping functionality, needs consolidation | `spammm/surfaces/GridFF.py`, `spammm/surfaces/GridFFRelaxedScan.py` | Medium |
| **Contact surface** | Sphere h₀ + coarse dx; profiles OK; XY vs GridFF USER pending | [ContactSurface_2p5D_vs_GridFF_2026-07-24.md](Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md), [Caveats.md](Caveats.md) §6 | Medium |
| **Substrate builder** | `SubstrateBuilder.py` is minimal — no CIF parsing, lattice replication, or slab cutting | `spammm/surfaces/SubstrateBuilder.py` | Low |
| **File I/O** | XYZ, MOL2 parsing in `atomicUtils.py` — no CIF or extended XYZ with Lattice support | `spammm/atomicUtils.py` | Low |
| **Vibrations** | Absolute UFF/DFTB frequencies vs experiment not calibrated; phonon bands unported | `dynamics/Vibrations.py`, `doc/FireCore_migration_codemap.md` | Low |

---

# WINDSURF CODEMAPS

* [All Windsurf Codemaps](https://windsurf.com/codemaps)

### AFM/STM Simulation
* [AFM Simulation: GPU Rigid Body Dynamics, CPU GridFF Relaxation, and Interactive GUI](https://windsurf.com/codemaps/594f7eaf-c3ab-4139-8f20-d1d2d7f8d401-fe86ab10a43f3d18)
* [AFM PyOpenCL System: Morse/LJ Path and FDBM Density-Based Path](https://windsurf.com/codemaps/9bb4c2a5-0c38-4943-abe9-254cfdcc75af-8796fe608a7d71c1)
* [AFM FDBM Pipeline: DFTB Backend & pySCF Integration Points](https://windsurf.com/codemaps/02d559c9-de47-4058-b07b-3318664b454e-fe86ab10a43f3d18)
* [Rigid Body Dynamics on Surfaces (pyOpenCL)](https://windsurf.com/codemaps/b5d9c2d2-50f0-4ba7-bc65-60db6e06e423-8796fe608a7d71c1)
* [Rigid Body Dynamics System for AFM Simulation](https://windsurf.com/codemaps/c9f13e1c-edfa-4702-814f-5036d03ea6c9-fe86ab10a43f3d18)

### STM Simulation
* [GPU Green's Function STM Implementation: Current Orbital Projection System & Planned GF Solver Integration](https://windsurf.com/codemaps/f398c2cf-5ff8-4d75-a398-c83e788e27b4-fe86ab10a43f3d18)
* [STM Simulation Pipeline: Orbital Projection & Quantum Transport](https://windsurf.com/codemaps/d0242216-c415-4f38-98f9-4c88b5dfeeb8-fe86ab10a43f3d18)
* [STM QMMM: DFTB Integration with GPU Density Projection](https://windsurf.com/codemaps/9fa40c64-e78c-42f2-9573-574936c8040d-fe86ab10a43f3d18)

### Surface Interactions
* [Interactive GridFF Scanning: PTCDA-on-CaF2 Constrained Relaxation System](https://windsurf.com/codemaps/99d506e2-223b-4ae7-bb60-8c2498fedfb9-8796fe608a7d71c1)
* [Surface Potential Evaluation: GridFF B-spline and XYZ Rigid Kernels](https://windsurf.com/codemaps/2a639fae-c9cb-407a-9d45-7b806c90c749-8796fe608a7d71c1)
* [FoldedAtomicFunctions: Surface Potential Basis Fitting System](https://windsurf.com/codemaps/c9fc44a7-57a2-47c5-906f-886fa301ccc7-8796fe608a7d71c1)
* [Molecule-Substrate Interaction Energy Scanning: Assembly, GUI Placement, Force Fields & Surface Evaluation](https://windsurf.com/codemaps/38bd3cb6-31c0-45b6-9e09-fda94257999c-8796fe608a7d71c1)
* [Molecule-on-Surface Systems: GridFF, XYZ Scanning, Surface Sampling, and Assembly](https://windsurf.com/codemaps/f8407e23-3a2e-41f1-abcf-9c15f3644c41-8796fe608a7d71c1)

### Force Fields
* [SPFF/UFF PyOpenCL Parity Infrastructure](https://windsurf.com/codemaps/8d1b056f-1502-4363-b52d-8257de4be453-8796fe608a7d71c1)
* [FitREQ_PN: Hydrogen-Bond Parameter Fitting System](https://windsurf.com/codemaps/d977d597-94b4-42c3-a92a-0cefe34a3e82-8796fe608a7d71c1)
* [FitREQ Interactive GUI: Monte Carlo Optimization & Energy Decomposition Integration](https://windsurf.com/codemaps/e25a0dfc-f9a8-42ab-b8bb-1d959037ca68-fe86ab10a43f3d18)
* [FitREQ Hydrogen Bond Fitting System - GPU-Accelerated Parameter Optimization](https://windsurf.com/codemaps/bf59a960-ac6c-4eea-b828-9bd18c3d44ac-fe86ab10a43f3d18)

### QM Integration
* [DFTB+ Python Integration: Library Interfaces, Parsers, and OpenCL Grid Projection](https://windsurf.com/codemaps/1d6b4b7c-04de-49ef-b581-12cf5bfef54a-fe86ab10a43f3d18)
* [DFTB+ Eigenvector Export for OpenCL Orbital Projection](https://windsurf.com/codemaps/845d1373-d23e-4f7d-a109-c0d8eccebea9-fe86ab10a43f3d18)
* [DFTB+ Calculation Flow: Standalone Program, C API, and Python Wrapper](https://windsurf.com/codemaps/2c157118-9d28-4a7c-a234-a49a3d464424-fe86ab10a43f3d18)
* [DFTB Reference Calculation & FDBM AFM Forcefield Comparison System](https://windsurf.com/codemaps/1153fe89-ff29-4d4b-b4a6-e97d8f37047f-fe86ab10a43f3d18)
