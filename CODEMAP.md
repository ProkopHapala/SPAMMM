# CODEMAP — SPAMMM Repository Structure

SPAMMM is a Python + PyOpenCL scientific simulation package for AFM/STM, molecular topology, forcefields, and surface interactions. No C++, no JavaScript, no Fortran — all compute is in Python orchestration + OpenCL kernels.

## Top-level

- `spammm/` — main library package
- `kernels/` — OpenCL `.cl` kernel sources
- `tests/` — pytest suite + standalone diagnostic scripts
- `data/` — molecular structures, FF parameter files, substrates
- `doc/` — documentation, design docs, topical audits, skills, protocols
- `examples/` — usage examples (density comparison, basis optimization)
- `scripts/` — utility scripts
- `run_gui.sh` — GUI launcher
- `pytest.ini` — pytest configuration (markers: `slow`, `gpu`)

## spammm/ — Core Library

### spammm/topology/ — Molecular Topology (SSOT: AtomicGraph)
- `AtomicGraph.py` — `Atom`, `Bond`, `Ring` classes; `to_arrays()`. **Authoritative** molecular structure (see skill:`molecular-structure-sync`)
- `MoleculeEditorBackend.py` — molecular editor backend: graph ↔ dense arrays; hex grid, editing ops, `_sync_sys()`, export
- `KekulePure.py` — Kekule pi-bond order solver; feasibility precheck, multi-seed localization, 6-ring validation; writes results back to `Bond.order`
- `PackedMolecule.py` — compact molecule representation
- `FFparams.py` — forcefield parameter assignment from topology
- `HexGrid.py` — hexagonal grid for graphene/2D structures
- `ascii_art_heterocycle.py` — ASCII art heterocycle builder; `:` H-bond marks, `resolve_hbond_pairs()`, `ASCII_EXAMPLES`
- `hbond_utils.py` — H-bond discovery on `AtomicGraph`/`AtomicSystem`; `HbondRecord`, `find_hbonds_graph`, control→fraction mapping
- `scan_dataset.py` — `ScanDataset` `.npz` I/O for reaction-coordinate trajectories (geometry, Mulliken charges, optional ESP stack)
- `scan_kekule.py` — C–C bond length vs control on scan trajectories (Kekulé analysis helper)
- `heterocycle_generator.py` — heterocycle structure generation

### spammm/forcefields/ — Interatomic Force Fields
- `FFController.py` — forcefield controller/orchestrator (`spff` / `uff` / `lff`)
- `FFEvaluator.py` — single-point UFF/SPFF E,F evaluator for FD Hessians
- `SPFF_cl.py` — SPFF forcefield (PyOpenCL)
- `UFF_cl.py` — UFF forcefield (PyOpenCL); fused multi-step + optional FAF
- `LFFSolver.py` — linearized projective Jacobi (K₁₂/K₁₃/K₁₄ from UFF; soft FAF)
- `SPFFbuilder.py` — SPFF topology/buffer builder
- `UFFbuilder.py` — UFF topology/buffer builder
- `RigidBodyDynamics.py` — rigid body dynamics integrator; `RigidBodyPairFF` with per-body state (dynamic/static/deleted), mixed-species packs, factorized PLQH, `body_state` GPU buffer
- `RigidBodyUtils.py` — shared helpers: `build_mixed_species_assembly` (round-robin body order), `compute_combined_probe_map` (PairFF static + FAF), `graph_to_rigid_fragments` (connected-components split)
- `RigidBodyAFM.py` — rigid body AFM tip dynamics
- `Assembly.py` — hexagonal SAM rigid-body packing search (orchestration + `AssemblyOCL`)
- `AssemblyPlot.py` — assembly top views, clash/strain diagnostics, XYZ export
- `QEq.py` — charge equilibration
- `README.md` — module index; three relax paths + Assembly

### spammm/dynamics/ — Vibrational Analysis
- `Vibrations.py` — Hessian assembly (DFTB / UFF / SPFF), rigid-mode projection, mode analysis, unit conversion
- `VibrationPlot.py` — top-view mode plots (in-plane arrows + seismic z circles)
- `README.md` — module index; see `doc/Topics/Vibrations.md`

### spammm/surfaces/ — Surface Interactions
- `GridFF.py` — grid force field (B-spline interpolation on periodic substrates)
- `ContactSurface.py` — **quasi-2D contact field** for aperiodic AFM: separable B-spline×poly + radial PIC; `ContactSurfaceCL`, `SeparableParams`, `PICParams`, CG fit, h₀ map
- `GridFFRelaxedScan.py` — relaxed scan over surface grid
- `FoldedRigid.py` — folded basis rigid body relaxation
- `Ewald2D.py` — 2D Ewald summation for surfaces
- `SurfaceEwald.py` — Ewald surface potential (GPU)
- `Surface_utils.py` — surface utility functions
- `SubstrateBuilder.py` — substrate crystal builder
- `surface_plots.py` — surface visualization (plotting only)
- `README.md` — module index; contact-surface API summary + fit knobs

### spammm/SPM/ — Scanning Probe Microscopy (AFM/STM)
- `AFM.py` — AFMulator (Morse/LJ + FDBM); contact surface helpers; **FAST_S3** fused ES + GPU pad/scale (`SPAMMM_AFM_FAST_S3`); `AFMBench`
- `AFM_utils.py` — tip densities, FDBM orchestration, `compose_and_relax_total`
- `ModularPipeline.py` — modular AFM/STM pipeline (S1–S6) with dual Stage-3 (fast/legacy)
- Perf report: `doc/Tasks/PerfBenchmark_FDBM.md`; bench: `tests/SPM/bench_fdbm.py`
- `ManipulationPathOpt.py` — manipulation path optimization
- `ScanUtils.py` — scan grid utilities

### spammm/quantum/ — Quantum Mechanics Integration
- `DFTB/DFTBcore.py` — DFTB+ core interface
- `DFTB/DFTBplusParser.py` — DFTB+ output parser
- `DFTB/Grid_dftb.py` — DFTB orbital grid projection (OpenCL)
- `DFTB/basis_optimizer.py` — DFTB basis set optimization
- `DFTB_utils.py` — DFTB+ utilities: `run_dftb_sp`, `run_dftb_relax`, `parse_mulliken_charges`, Hessian I/O, constrained scan helpers
- `coordinate_scan.py` — reaction-coordinate paths: control grids, pm-NEB (relax + interp), rigid DFTB scan → `ScanDataset`
- `esp_grid.py` — Coulomb ESP on 2D grids from atomic charges (same KE/r as QEq); stack precompute for animation
- `hbond_scan.py` — rigid DFTB H-bond proton-transfer scan for ASCII `:` systems (0.1 Å path grid)
- `pySCF_utils.py` — pySCF integration utilities

### spammm/GUI/ — Graphical User Interface (VisPy)
- `SPAMMM_GUI.py` — main GUI application
- `BaseGUI.py` — base GUI class
- `GLGUI.py` — OpenGL-based GUI
- `VispyUtils.py` — VisPy 3D rendering (`AtomScene`, atom/bond visualization)
- `MoleculeViewer.py` — molecule viewer widget
- `MolecularBrowser.py` — molecular file browser
- `MolecularBrowserVispy.py` — Vispy-based browser
- `DirectoryNavigator.py` — directory tree navigator
- `ThumbnailCache.py` — thumbnail caching for browser
- `ExtensionManager.py` — plugin/extension manager
- `KekuleExtension.py` — Kekule solver GUI extension
- `AsciiArtExtension.py` — ASCII art → molecule + Kekule + H-bond resolution GUI
- `AFMExtension.py` — FDBM AFM/STM GUI (ModularPipeline); plot z=3.0 Å default; atom overlay
- `FFExtension.py` — forcefield GUI extension
- `VibrationExtension.py` — normal-mode analysis GUI (DFTB / UFF / SPFF, clickable mode table)
- `QEqExtension.py` — QEq GUI extension
- `ReactionCoordinateExtension.py` — H-bond reaction-coordinate scan panel (DFTB, slider, bond Δ viz, ESP animation)
- `rc_esp_view.py` — blitted ESP animation synced to RC slider
- `mpl_blit.py` — reusable matplotlib blit helper for Qt embedded plots (see `doc/Takeways.md`)
- `rc_scan_gui_script.py` — programmatic RC review setup for `--script` launcher
- `gui_script_runner.py` — runs control scripts after `window.show()`
- `gui_script_utils.py` — script helpers; demo overlays; `capture_window_png` / `frames_to_gif`
- `azaindol_draw_sequence.py` — shared hex→azaindol→dimer draw sequence (SVG + GUI hosts)
- `gui_scripts/` — `rc_scan_*`, `azaindol_draw_demo.py` / `azaindol_draw_offline.py`, `folded_rigid_setup.py`, `ptcda_drag_demo.py`, `static_obstacle_drag_demo.py` (dimer split + static/dynamic toggle + combined probe map)
- `plotutils.py` — Qt 2D plot window wrapper; re-exports `spammm.plotUtils` ESP helpers
- `CollapsibleSection.py` — UI collapsible section widget
- `shaders/` — GLSL shaders

### spammm/utils/ — Utilities
- `OpenCLBase.py` — OpenCL base class (kernel caching, buffer management)
- `clUtils.py` — OpenCL utility functions
- `Lingebra_ocl.py` — linear algebra OpenCL kernels wrapper
- `test_utils.py` — test utility functions

### spammm/ — Root Modules
- `AtomicSystem.py` — atomic system container (dense arrays: `apos`, `enames`, `bonds`)
- `atomicUtils.py` — atomic utility functions (file I/O, MOL/MOL2/XYZ, bond detection)
- `elements.py` — periodic table data
- `plotUtils.py` — plotting utilities
- `config_utils.py` — configuration file handling
- `globals.py` — global constants

## kernels/ — OpenCL Kernel Sources
- `UFF.cl` — UFF forcefield kernels (incl. fused multi-step + FAF)
- `SPFF.cl` — SPFF forcefield kernels (incl. fused multi-step + FAF)
- `LFF.cl` — projective Jacobi on linearized springs (K₁₂/K₁₃/K₁₄ + FAF outer)
- `Forces.cl` — general force computation
- `AFM.cl` — AFM PP relax + FDBM Stage-3 `fdbm_*` helpers (FAST_S3)
- `gridFF.cl` — grid force field kernels
- `surface.cl` — surface interaction kernels
- `contact_surface.cl` — quasi-2D contact field: brute reference, separable Av/Atv/eval, PIC fit/eval, `relaxStrokesTiltedContact` / `relaxStrokesTiltedPIC`
- `rigid.cl` — rigid body dynamics kernels
- `assembly.cl` — rigid-body SAM packing: `emit_configuration_xyz`, `evaluate_packing_3d`
- `nonbonded.cl` — non-bonded interaction kernels
- `nonbonded_grid.cl` — grid-based non-bonded kernels
- `LCAO_STM.cl` — LCAO STM simulation kernels
- `LCAO_grid.cl` — LCAO orbital grid projection kernels
- `lingebra.cl` — linear algebra kernels
- `common.cl` — shared OpenCL utilities
- `README.md` — OpenCL kernel index and composition rules

## tests/ — Test Suite
- `conftest.py` — pytest fixtures (data paths, molecule loader, `--update-refs`)
- `test_topology.py` — bond/angle/hybridization/type assignment
- `test_forcefield.py` — UFF/SPFF optimization, NVE conservation, energy–force correspondence
- `test_relax_serial.py` — SPFF serial vs batch parity
- `test_relax_flat1.py` — flat_1 PAH vacuum/substrate timing
- `test_relax_ptcda_faf.py` — PTCDA+FAF: SPFF/UFF fused + LFF topology/sweep
- `test_body_state.py` — PairFF body-state L0 tests (all-dynamic parity, frozen invariant, static interaction, deletion parity k14+k15, mixed-species FAF, map decomposition)
- `GUI/test_rigid_assembly_extension.py` — RA extension GUI tests (build, display index, drag step, MC parity, gestures, mixed-species)
- `test_vibrations.py` — normal modes (H2O fast; benzene/PTCDA slow); `debug/test_vibrations/`
- `test_surface.py` — Ewald vs brute, GridFF, folded function
- `test_folded_relax.py` — rigid body relaxation + manipulation
- `test_lingebra.py` — linear algebra eigenvalue tests
- `testplot_tensor_parity.py` — GPU vs CPU tensor kernel parity (visual demo)
- `testplot_folded_surface_scan.py` — folded basis fitting + plots (visual demo)
- `testplot_assembly.py` — hexagonal SAM assembly search + clash/strain diagnostics (visual demo; Class 2)
- `testplot_pairff_energy_mc.py` — PairFF+FAF rigid-body greedy MC assembly (8 mols, multi-species `--mol adenine,uracil`, charge colors, GIF trajectory); `debug/testplot_pairff_energy_mc/<mol>/`
- `testplot_contact_surface.py` — contact-surface fit + parity (separable + PIC + PP relaxed); `debug/testplot_contact_surface/`
- `test_export_import.py` — MOL/MOL2/XYZ round-trip
- `test_clipboard_undo.py` — clipboard/undo operations
- `test_packed_molecule.py` — packed molecule representation
- `test_molecule_viewer.py` — molecule viewer tests
- `test_directory_navigator.py` — directory navigator tests
- `test_thumbnail_cache.py` — thumbnail cache tests
- `test_relax_serial.py` — serial relaxation
- `test_integration.py` — (removed, stubs were never implemented)
- `run_manipulation.py` — CLI relaxed scan, export .xyz movie (Class 2)
- `ref_data/` — git-tracked reference files (`.ref.json`, `.ref.xyz`)
- `helpers/` — test utility modules (`parity.py`, `geometry.py`, `scan.py`, `folded_rigid.py`, `topology_test.py`)
- `SPM/` — AFM/STM tests and plots (`test_afm_contact_surface.py`, `testplot_afm_contact_surface.py`)
- `topology/` — topology editing, Kekule (`test_kekule.py`), H-bond DFTB scan (`test_hbond_scan.py`), RC scan (`test_scan_dataset.py`, `testplot_hbond_scan.py`)
- `GUI/` — `test_rc_scan_gui_script.py` (offscreen RC review script)
- `surfaces/` — surface-specific tests
- `forcefields/` — forcefield-specific tests
- `quantum/` — quantum integration tests

## data/ — Data Files
- `mol/` — molecule files (`.mol2`)
- `xyz/` — molecule files (`.xyz`)
- `substrates/` — substrate crystal structures (`.xyz`)
- `AtomTypes.dat`, `BondTypes.dat`, `AngleTypes.dat`, `DihedralTypes.dat` — FF parameters

## doc/ — Documentation
- `Caveats.md` — recurring scientific/numerical traps (all-e Δρ/NA, corner vs center, sample vs project)
- `topical_audit.md` — cross-implementation maps per scientific topic
- `Takeways.md` — resolved pitfalls (blit, GUI vs test geometry, DFTB, **contact-surface z/sign/buffers**)
- `TEST_DESIGN.md` — test system design
- `TEST_RESULTS.md` (in `tests/`) — test results log
- `AGENTS/` — agent instructions, skills, protocols, workflows
- `HowTo/` — how-to guides
- `Tasks/` — task design documents (e.g. `ReactionCoordinateExtension_Design.md`)
- `Reports/` — durable scientific handoffs (e.g. Fukui panel ES notes)
- `TopicalAudit/` — per-topic implementation maps (e.g. `AFM_FDBM.md`)
- `Topics/AFM/ContactSurface_Static.md` — **quasi-2D AFM field** (separable + PIC): API, tutorial, parity
- `Topics/AFM/ContactSurface_Elastic.md` — elastic extension (future)
- `Topics/ForceFields/LFF_ProjectiveRelax.md` — LFF projective Jacobi (springs + FAF outer); 3rd relax path
- `Topics/Vibrations.md` — normal-mode analysis (Hessian, GUI, units, tests)
- `Topics/ReactionCoordinateScan.md` — H-bond RC scan, pm-NEB, ScanDataset, ESP animation
- `Tasks/PerfBenchmark_Relaxation.md` — UFF/SPFF/LFF relax perf + PTCDA+FAF session log
- `Ideas/` — research ideas
- Various topic-specific `.md` files (forcefields, topology, surfaces, AFM, GUI)
