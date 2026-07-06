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
- `KekuleBackend.py` — bridge: graph ↔ dense arrays; `_sync_sys()`, `save_structure()`, export
- `KekulePure.py` — Kekule pi-bond order solver; writes results back to `Bond.order`
- `PackedMolecule.py` — compact molecule representation
- `FFparams.py` — forcefield parameter assignment from topology
- `HexGrid.py` — hexagonal grid for graphene/2D structures
- `ascii_art_heterocycle.py` — ASCII art rendering of heterocycles
- `heterocycle_generator.py` — heterocycle structure generation

### spammm/forcefields/ — Interatomic Force Fields
- `FFController.py` — forcefield controller/orchestrator
- `SPFF_cl.py` — SPFF forcefield (PyOpenCL)
- `UFF_cl.py` — UFF forcefield (PyOpenCL)
- `SPFFbuilder.py` — SPFF topology/buffer builder
- `UFFbuilder.py` — UFF topology/buffer builder
- `RigidBodyDynamics.py` — rigid body dynamics integrator
- `RigidBodyAFM.py` — rigid body AFM tip dynamics
- `Assembly.py` — force assembly from kernel outputs
- `QEq.py` — charge equilibration

### spammm/surfaces/ — Surface Interactions
- `GridFF.py` — grid force field (B-spline interpolation)
- `GridFFRelaxedScan.py` — relaxed scan over surface grid
- `FoldedRigid.py` — folded basis rigid body relaxation
- `Ewald2D.py` — 2D Ewald summation for surfaces
- `SurfaceEwald.py` — Ewald surface potential
- `Surface_utils.py` — surface utility functions
- `SubstrateBuilder.py` — substrate crystal builder
- `surface_plots.py` — surface visualization (plotting only)

### spammm/SPM/ — Scanning Probe Microscopy (AFM/STM)
- `AFM.py` — AFM simulation (Morse/LJ + FDBM)
- `AFM_util.py` — AFM utilities, density-based model helpers
- `ModularPipeline.py` — modular AFM pipeline (S1-S6 stages)
- `ManipulationPathOpt.py` — manipulation path optimization
- `ScanUtils.py` — scan grid utilities

### spammm/quantum/ — Quantum Mechanics Integration
- `DFTB/DFTBcore.py` — DFTB+ core interface
- `DFTB/DFTBplusParser.py` — DFTB+ output parser
- `DFTB/Grid_dftb.py` — DFTB orbital grid projection (OpenCL)
- `DFTB/basis_optimizer.py` — DFTB basis set optimization
- `DFTB_utils.py` — DFTB utility functions
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
- `AFMExtension.py` — AFM simulation GUI extension
- `FFExtension.py` — forcefield GUI extension
- `QEqExtension.py` — QEq GUI extension
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
- `UFF.cl` — UFF forcefield kernels
- `SPFF.cl` — SPFF forcefield kernels
- `Forces.cl` — general force computation
- `AFM.cl` — AFM simulation kernels
- `gridFF.cl` — grid force field kernels
- `surface.cl` — surface interaction kernels
- `rigid.cl` — rigid body dynamics kernels
- `assembly.cl` — force assembly kernels
- `nonbonded.cl` — non-bonded interaction kernels
- `nonbonded_grid.cl` — grid-based non-bonded kernels
- `LCAO_STM.cl` — LCAO STM simulation kernels
- `LCAO_grid.cl` — LCAO orbital grid projection kernels
- `lingebra.cl` — linear algebra kernels
- `common.cl` — shared OpenCL utilities
- `INVENTORY.md` — kernel inventory documentation

## tests/ — Test Suite
- `conftest.py` — pytest fixtures (data paths, molecule loader, `--update-refs`)
- `test_topology.py` — bond/angle/hybridization/type assignment
- `test_forcefield.py` — UFF/SPFF optimization, NVE conservation
- `test_surface.py` — Ewald vs brute, GridFF, folded function
- `test_folded_relax.py` — rigid body relaxation + manipulation
- `test_lingebra.py` — linear algebra eigenvalue tests
- `test_tensor_parity.py` — GPU vs CPU tensor kernel parity (Class 2)
- `test_folded_surface_scan.py` — folded basis fitting + plots (Class 2)
- `test_export_import.py` — MOL/MOL2/XYZ round-trip
- `test_clipboard_undo.py` — clipboard/undo operations
- `test_packed_molecule.py` — packed molecule representation
- `test_molecule_viewer.py` — molecule viewer tests
- `test_directory_navigator.py` — directory navigator tests
- `test_thumbnail_cache.py` — thumbnail cache tests
- `test_relax_serial.py` — serial relaxation
- `test_integration.py` — integration test stubs
- `run_manipulation.py` — CLI relaxed scan, export .xyz movie (Class 2)
- `ref_data/` — git-tracked reference files (`.ref.json`, `.ref.xyz`)
- `helpers/` — test utility modules (`parity.py`, `geometry.py`, `scan.py`, `folded_rigid.py`, `topology_test.py`)
- `SPM/` — AFM/STM tests and plots
- `topology/` — topology editing, Kekule, heterocycle tests
- `surfaces/` — surface-specific tests
- `forcefields/` — forcefield-specific tests
- `quantum/` — quantum integration tests

## data/ — Data Files
- `mol/` — molecule files (`.mol2`)
- `xyz/` — molecule files (`.xyz`)
- `substrates/` — substrate crystal structures (`.xyz`)
- `AtomTypes.dat`, `BondTypes.dat`, `AngleTypes.dat`, `DihedralTypes.dat` — FF parameters

## doc/ — Documentation
- `topical_audit.md` — cross-implementation maps per scientific topic
- `TEST_DESIGN.md` — test system design
- `TEST_RESULTS.md` (in `tests/`) — test results log
- `AGENTS/` — agent instructions, skills, protocols, workflows
- `HowTo/` — how-to guides
- `Tasks/` — task design documents
- `Ideas/` — research ideas
- Various topic-specific `.md` files (forcefields, topology, surfaces, AFM, GUI)
