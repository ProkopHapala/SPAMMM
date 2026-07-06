# FireCore → SPAMMM Migration Codemap

## Purpose

SPAMMM is a clean, lean extraction of FireCore — stripped of C++, JavaScript, Fortran, CUDA, and experimental features, keeping only what's useful for SPM (Scanning Probe Microscopy) simulations. This document maps FireCore's topics to SPAMMM's current state, identifying what is already migrated and what may be worth porting in the future.

**FireCore topical audit location**: `/home/prokophapala/git/FireCore/doc/topical_audit/`

# POTENTIAL FUTURE PORTS

## Priority 1 — HIGH

### 1.1 RigidAtom Forcefield (XPBD — Per-Atom Rigid Body Dynamics)

**What**: Treats each atom as a rigid body with 6 DOFs (position + quaternion). Port-based constraints encode bond lengths and angles without explicit potentials. Conceptually "As-Rigid-As-Possible" for molecular graphs.

**Why interesting**: 
- Enables flexible molecule simulation without explicit angle/dihedral terms
- Natural for AFM tip–molecule interactions where local geometry matters
- Port-based topology is elegant and GPU-friendly
- User explicitly mentioned wanting to migrate this

**FireCore files to port**:
- `pyBall/XPBD_2D/XPBD_2D.py` + `XPBD_2D.cl` — 2D version (complex number rotation)
- `pyBall/XPDB_AVBD/XPDB.py` + `XPDB.cl` — 3D version (quaternion, angular velocity)
- `pyBall/XPDB_AVBD/RRsp3.py` + `RRsp3.cl` — Rigid-atom sp3 solver with port-based topology
- `pyBall/XPBD_2D/` discussion documents — design rationale

**SPAMMM target location**: `spammm/forcefields/RigidAtomFF.py` + `kernels/rigid_atom.cl`

**Effort**: Medium — Python+OpenCL code already exists, needs cleanup and integration with SPAMMM's `AtomicGraph` topology and `FFController`

**Key physics to preserve**:
- Port-based constraints (local frame → world transform → spring force)
- Analytic Procrustes rotation update (SVD of cross-covariance)
- Cluster collision detection (AABB bounding boxes)
- Heavy-ball momentum acceleration

---

### 1.2 ProjectiveDynamics (Position-Based Implicit Solver)

**What**: Implicit solver for stiff spring networks. Solves `Ax = b` at each timestep using Jacobi, Gauss-Seidel, or Cholesky. Allows 10–100x larger timesteps than explicit integration.

**Why interesting**:
- Enables stable relaxation of molecules on surfaces with large timesteps
- OpenCL kernel already exists (`truss.cl`)
- Momentum-accelerated Jacobi achieves GS-like convergence while staying parallel
- Planned for near-term migration alongside RigidAtom FF

**FireCore files to port**:
- `pyBall/pyTruss/truss.cl` — OpenCL kernels (Jacobi, Gauss-Seidel with graph coloring)
- `pyBall/pyTruss/truss_solver_ocl.py` — PyOpenCL wrapper
- `doc/py/ProjectiveDynamics/projective_dynamics.py` — Python reference

**SPAMMM target location**: `spammm/forcefields/ProjectiveDynamics.py` + `kernels/projective_dynamics.cl`

**Effort**: Medium — OpenCL kernel exists, needs integration with SPAMMM's `OpenCLBase` and `FFController`

**Key physics to preserve**:
- Sparse system matrix assembly (stiffness + inertia + damping)
- Jacobi iteration with diagonal preconditioning (fully parallel)
- Gauss-Seidel with graph coloring (3–6 colors for typical molecular graphs)
- Momentum-accelerated Jacobi (heavy-ball / Nesterov, β ≈ 0.9)
- Local memory: preload diagonal 3×3 blocks

### 1.3 Molecular Building Operations & Topology Manipulation

**What**: A toolkit for programmatic molecule construction and topology editing:
- **Polymer assembly** from monomer tokens with anchor-based linking (`assemblePolymerFromTokens`)
- **Group attachment by markers** — attach functional groups (–OH, –NH₂, =O, –CH₃, etc.) using dummy-atom marker pairs (Xe/He) that define position and orientation, then are removed after bonding (`attachGroupByMarker`)
- **Group attachment by direction** — attach fragments using forward/up reference atoms to define the local frame (`attachParsedByDirection`)
- **Bridge insertion/collapse** — insert a CH₂ bridge across a bond, or collapse a bridge atom to reform the direct bond (`insertBridge`, `collapseBridgeAt`, `collapseAllBridges`)
- **PBC replication** — replicate unit cell contents over lattice repeats (`replicateMolecule`)
- **Substrate builders** — NaCl slab/step-edge, CIF crystal builder with symmetry operations

**Why interesting**:
- SPAMMM's `KekuleBackend` has basic hex-grid ring add/remove and `collapse_bond`, but lacks programmatic group attachment, polymer assembly, and bridge operations
- Building complex molecules (functionalized PAHs, polymers on surfaces) is needed for SPM simulations
- Marker-based attachment is elegant — works like "click chemistry" for molecular graphs
- All operations are pure Python/NumPy, no GPU needed

**What SPAMMM already has**:
- `KekuleBackend.add_ring()` / `remove_ring()` — hex-grid ring operations
- `KekuleBackend.add_atom_at_position()` — arbitrary position atom insertion
- `KekuleBackend.collapse_bond()` — bond collapse (partial equivalent of `collapseBridgeAt`)
- `KekuleBackend.adjust_h()` — auto hydrogen capping
- `AtomicGraph.add_atom()` / `add_ring()` / `remove_ring()` / `detect_rings()`

**What needs to be ported** (from JS, rewrite in Python):
- `MoleculeUtils.js::assemblePolymerFromTokens()` — polymer sequence builder with head/tail anchors
- `MoleculeUtils.js::attachGroupByMarker()` — marker-pair group attachment (Xe/He dummy atoms)
- `MoleculeUtils.js::attachParsedByDirection()` — directional frame-based group attachment
- `MoleculeUtils.js::insertBridge()` / `collapseBridgeAt()` / `collapseAllBridges()` — bridge operations
- `MoleculeUtils.js::replicateMolecule()` — PBC supercell replication
- `SubstrateBuilder.py::gen_nacl_slab()` / `gen_nacl_step()` — already Python, could be directly ported
- `CrystalUtils.js` — CIF parsing, symmetry operations, unit cell builder (rewrite in Python)

**FireCore C++ reference** (algorithm only, not ported):
- `MMFFBuilder.h::splitByBond()` — fragment splitting by bond removal
- `Groups.h` — group-based force application with COG/orientation tracking

**SPAMMM target location**: `spammm/topology/MoleculeBuilder.py` (new module for building operations)

**Effort**: Medium — JS code is clean and well-structured, Python rewrite is straightforward. SubstrateBuilder is already Python.

---

### 1.4 Graph-Based Atom Selection (SMARTS-like Pattern Matching)

**What**: A selection/query system for atoms based on chemical environment, like regular expressions for molecular graphs. FireCore's `MoleculeSelection.js` implements:
- **Element/type token matching** — `C`, `N|O`, `C_sp2`, `*` (wildcard)
- **Neighbor count constraints** — `C n{N|O}={1,2}` selects carbons with 1–2 nitrogen/oxygen neighbors
- **Degree constraints** — `C deg={3}` selects carbons with exactly 3 bonds
- **Bridge candidate detection** — `selectBridgeCandidates()` finds atoms with specific heavy/hydrogen neighbor counts
- **Selection modes** — replace, add, subtract

**Query syntax example**:
```
C n{N}={1} n{O}={0}     → carbons bonded to exactly 1 N and 0 O
C_sp2 deg={3}            → sp2 carbons with 3 bonds
* n{C}={2} n{H}={2}      → any atom with 2 C and 2 H neighbors (bridge candidates)
```

**Why interesting**:
- SPAMMM has basic selection in `VispyUtils` (picking, highlighting) but no chemical-environment queries
- Essential for programmatic molecule modification: "select all sp2 carbons with a free valence, then attach a group"
- Natural complement to the molecular building operations (1.3) — you need to select where to attach before attaching
- Works on `AtomicGraph` topology — pure Python, no GPU

**What SPAMMM already has**:
- `VispyUtils` — visual selection (click, box-select, highlight)
- `AtomicSystem` — atom arrays with `atypes`, `enames`
- `AtomicGraph` — full bond topology, neighbor lists, ring detection

**What needs to be ported** (from JS, rewrite in Python):
- `MoleculeSelection.js::compileSelectQuerySpec()` — parse query string into matcher
- `MoleculeSelection.js::applySelectQuery()` — apply compiled query to molecule, return selected atom set
- `MoleculeSelection.js::selectBridgeCandidates()` / `findBridgeAtom()` / `findBondPairForInsert()` — specialized topology queries
- Token set matching (`compileTokenSetToMatcher`) — element/atom-type union matching with wildcards

**SPAMMM target location**: `spammm/topology/AtomSelection.py` (new module)

**Effort**: Low–Medium — JS code is ~260 lines, clean logic. Python rewrite on `AtomicGraph` is straightforward.

---

## Priority 2 — MEDIUM (Possible future port)

### 2.1 NEB (Nudged Elastic Band) for H-Transfer

**What**: NEB calculations for hydrogen transfer between molecules and periodic ribbons. Supports rigid scan, relaxed scan, and dry-run modes. PBC-aware with k-point sampling for ribbon systems.

**Why interesting**:
- H-transfer on surfaces is relevant to AFM tip chemistry
- PBC + k-point infrastructure could be reused for other periodic calculations
- However, this is more of a QM workflow tool than core SPM

**FireCore files to port**:
- `tests/pyFireball/neb_h_transfer_molecules.py` — molecular NEB
- `tests/pyFireball/neb_h_transfer.py` — ribbon NEB with k-points
- `tests/pyFireball/build_two_ribbons.py` — ribbon cell builder
- `tests/pyFireball/scan_constrained.py` — constrained H-transfer scan
- `tests/pyFireball/scan_LHb.py` — H-bond length scan

**SPAMMM target location**: `spammm/quantum/NEB.py` + `tests/SPM/test_neb.py`

**Effort**: Medium — needs refactoring from standalone scripts to shared module, integration with SPAMMM's DFTB interface

---

### 2.2 Nanocrystal / Phonon / Vibration Spectroscopy

**What**: Complete pipeline for computing phonon band structures and FTIR absorption spectra: Hessian via finite differences, mass matrix, rigid-mode projection, Green's function probing.

**Why interesting**:
- Could characterize substrate phonons (relevant to AFM damping)
- Parameter fitting to DFT reference Hessians
- However, primarily C++ dependent (Hessian via ctypes) — would need pure Python/OpenCL reimplementation

**FireCore files to port** (Python only):
- `pyBall/FTIR.py` — linear response FTIR, Green's function probing, parameter fitting
- `tests/tMMFF/test_diamond_phonon_bands.py` — phonon band structure
- `tests/tMMFF/test_vibration_spectra.py` — end-to-end spectrum
- `tests/tMMFF/test_hessian_fitting.py` — parameter fitting
- `doc/Topics/FTIR_Nanocrystals/Phonon_testing_guide.md` — practical guide
- `doc/Topics/FTIR_Nanocrystals/Hessian_Kspace.md` — k-space theory

**SPAMMM target location**: `spammm/quantum/Phonons.py` + `tests/test_phonons.py`

**Effort**: High — Hessian computation currently uses C++ ctypes; would need PyOpenCL reimplementation. FTIR post-processing is pure Python and easier.

**Caveat**: Not directly SPM-related. Lower priority unless substrate phonon characterization becomes needed.

---

## Priority 3 — LOW (Experimental or tangential to SPM)

### 3.1 FMM (Fast Multipole Method)

**What**: Tile-based single-layer FMM for long-range electrostatics on GPU. Replaces O(N²) Coulomb with O(N log N) using multipole expansions up to quadrupole order.

**Why not high priority**:
- Experimental in FireCore, not integrated into production MD loop
- SPAMMM uses GridFF + Ewald2D for surface electrostatics, which covers most use cases
- Only needed for very large systems (>10⁴ atoms) — unusual for SPM

**FireCore files** (if ever needed):
- `cpp/common_resources/cl/FMM.cl` — OpenCL kernel
- `cpp/common/math/Multipoles.h` — C++ math (would need Python port)
- `doc/FMM/FMM.md` — mathematical derivation

**Effort**: High — C++ math utilities need Python/NumPy reimplementation

### 3.2 Reactive RigidAtom FF (RRsp3 → Reactive Port-Based FF)

**What**: A reactive extension of the RigidAtom forcefield. FireCore's `RRsp3.py`/`RRsp3.cl` is a rigid-atom sp3 solver with port-based topology. The plan for SPAMMM is to make a reactive version:
- Replace harmonic port springs with exponential/radial terms that can dissociate
- Enable all-ports-to-all-atoms interaction (not just bonded neighbors)
- This approaches RARFF (Reactive Atomistic Force Field) physics — semi-quantum force fields using localized orbital ansätze — but within the port-based rigid-atom framework

**Why interesting**:
- Closely related to RigidAtom FF (Priority 1) — natural extension of the same codebase
- Enables bond breaking/forming during AFM manipulation simulations
- All-ports-to-all-atoms interaction captures multi-center bonding without explicit electron representation
- RRsp3 code already exists in FireCore as starting point

**FireCore files to port**:
- `pyBall/XPDB_AVBD/RRsp3.py` + `RRsp3.cl` — rigid-atom sp3 solver with port-based topology
- `cpp/common/molecular/RARFF.h` — reference for RARFF physics (C++, for algorithm reference only)
- `cpp/common/molecular/eFF.h` — electron force field reference (C++, for physics reference only)

**SPAMMM target location**: `spammm/forcefields/ReactiveRigidAtomFF.py` + `kernels/reactive_rigid_atom.cl`

**Effort**: High — requires new physics (exponential radial terms, all-ports-to-all-atoms) beyond direct port of RRsp3

**Depends on**: RigidAtom FF (1.1) must be migrated first

### 3.3 Web-Based Force Fields (WebGL/WebGPU)

**Entirely skipped** — SPAMMM is Python + PyOpenCL only, no JavaScript.

---

# SUMMARY TABLE

| # | Topic | Priority | Effort | In SPAMMM? |
|---|-------|----------|--------|------------|
| 1 | RigidAtom FF (XPBD) | **HIGH** | Medium | No — planned for near-term |
| 2 | ProjectiveDynamics | **HIGH** | Medium | No — planned for near-term |
| 3 | Molecular Building Ops (polymer, attach, bridge) | **HIGH** | Medium | Partial — KekuleBackend has ring ops only |
| 4 | Graph-Based Atom Selection (SMARTS-like) | **HIGH** | Low–Med | No — JS code to rewrite in Python |
| 5 | NEB / H-transfer | MEDIUM | Medium | No — possibly |
| 6 | Phonon / FTIR / Hessian | MEDIUM | High | No |
| 7 | FMM | LOW | High | No |
| 8 | Reactive RigidAtom FF (RRsp3) | MEDIUM | High | No — planned, depends on #1 |
| 9 | Web FF (WebGL/WebGPU) | — | — | N/A (skip) |
| 10 | FitREQ | — | — | No (no plans to migrate) |
| 11 | KekuleSolver | — | — | **Yes** — migrated as `KekulePure.py` |
| 12–32 | Topology, UFF, SPFF, RigidBody, GridFF, FAF, Ewald2D, AFM, STM, DFTB, QEq, Assembly, KekulePure | — | — | **Yes — already migrated** |

---

# MIGRATION NOTES

## Architecture differences

- FireCore's `MolWorld_sp3.h` (C++ orchestrator) → SPAMMM's `FFController.py` (Python)
- FireCore's `pyBall/OCL/` namespace → SPAMMM's `spammm/` package structure
- FireCore's `cpp/common_resources/cl/` kernels → SPAMMM's `kernels/` directory
- FireCore's `tests/tMMFF/` → SPAMMM's `tests/` with pytest structure
- FireCore uses ctypes for C++ bindings → SPAMMM is pure Python + PyOpenCL (no ctypes)

## What was intentionally left behind

- All C++ code (`cpp/`, `pyBall/FireCore.py`, `pyBall/Kekule.py` ctypes)
- All JavaScript code (`web/`)
- All Fortran code
- CUDA kernels (SPAMMM uses OpenCL only)
- WebGPU/WebGL shaders
- `MoleculeEditor2D.py` (deprecated)
- Reactive force fields (too experimental)
- Multi-language parameter parsers (SPAMMM has single Python parser)

## Dependencies for future ports

All future ports should:
1. Use `spammm/utils/OpenCLBase.py` for OpenCL context management
2. Integrate with `spammm/forcefields/FFController.py` for force dispatch
3. Use `spammm/topology/AtomicGraph.py` as topology SSOT
4. Follow SPAMMM's `kernels/` directory for `.cl` files
5. Add tests in `tests/` with pytest conventions


---


# ALREADY IN SPAMMM (No Action Needed)

These topics have been migrated from FireCore's Python/PyOpenCL code into SPAMMM. The C++/JS/Fortran counterparts are intentionally left behind.

## 1. Molecular Topology — Graph & Connectivity

| FireCore Python File | SPAMMM File | Status |
|---------------------|-------------|--------|
| `pyBall/AtomicGraph.py` | `spammm/topology/AtomicGraph.py` | **Migrated** — SSOT for molecular topology |
| `pyBall/AtomicSystem.py` | `spammm/AtomicSystem.py` | **Migrated** — array-based representation |
| `pyBall/atomicUtils.py` | `spammm/atomicUtils.py` | **Migrated** — bond/angle/dihedral utilities |

**Not ported** (C++/JS only, not relevant): `MMFFBuilderBase.h`, `LimitedGraph.h`, `EditableMolecule.js`, `MMFFLTopology.js`

## 2. Molecular Topology — Type Assignment & Parameters

| FireCore Python File | SPAMMM File | Status |
|---------------------|-------------|--------|
| `pyBall/OCL/MMFF.py` (param loading) | `spammm/topology/FFparams.py` | **Migrated** |
| `tests/tUFF/data_UFF/*.dat` | `data/*.dat` | **Migrated** — `ElementTypes.dat`, `AtomTypes.dat`, `BondTypes.dat`, `AngleTypes.dat`, `DihedralTypes.dat` |

**Not ported**: C++ `MMFFparams.h`, JS `MMParams.js` parsers

## 3. Molecular Topology — Editors (GUI)

| FireCore Python File | SPAMMM File | Status |
|---------------------|-------------|--------|
| `pyBall/KekuleBackend.py` | `spammm/topology/KekuleBackend.py` | **Migrated** |
| `pyBall/KekuleExplorerGUI.py` | `spammm/GUI/SPAMMM_GUI.py` + extensions | **Migrated** (refactored into SPAMMM GUI) |
| `pyBall/ExtensionManager.py` | `spammm/GUI/ExtensionManager.py` | **Migrated** |

**Not ported**: `MoleculeEditor2D.py` (deprecated), all JS editors (`molgui_web`, `molgui_webgpu`)

## 4. UFF (Universal Force Field)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `pyBall/OCL/UFF.py` | `spammm/forcefields/UFFbuilder.py` | **Migrated** |
| `cpp/common_resources/cl/relax_multi.cl` (UFF parts) | `kernels/UFF.cl` | **Migrated** (OpenCL kernel) |

## 5. SPFF / MMFFsp3

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `pyBall/OCL/MMFF.py` (FF evaluation) | `spammm/forcefields/SPFF_cl.py` | **Migrated** (renamed MMFFsp3 → SPFF) |
| `pyBall/OCL/MMFF.py` (topology builder) | `spammm/forcefields/SPFFbuilder.py` | **Migrated** |
| `cpp/common_resources/cl/relax_multi.cl` (MMFF parts) | `kernels/UFF.cl` (shared) | **Migrated** |

## 6. RigidBody Dynamics (Whole-Molecule Rigid Body)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `pyBall/OCL/RigidBodyDynamics.py` | `spammm/forcefields/RigidBodyDynamics.py` | **Migrated** |
| `cpp/common_resources/cl/Rigid.cl` | `kernels/rigid.cl` | **Migrated** |
| `pyBall/OCL/RigidBodyAFM.py` | `spammm/forcefields/RigidBodyAFM.py` | **Migrated** — AFM-specific rigid body |

## 7. Non-Bonded Force Field (LJ, Morse, Coulomb, H-bond)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `cpp/common_resources/cl/relax_multi.cl` (getLJQH, getMorsePLQH) | `kernels/Forces.cl` | **Migrated** |
| `pyBall/OCL/MMFF.py` (NB evaluation) | `spammm/forcefields/SPFF_cl.py` | **Migrated** |

## 8. Assembly (Rigid-Body Packing & Clash Scoring)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `pyBall/OCL/Assembly.py` | `spammm/forcefields/Assembly.py` | **Migrated** |
| `cl/Assembly.cl` | `kernels/assembly.cl` | **Migrated** |
| `tests/tMMFF/test_assembly.py` (driver) | `tests/testplot_assembly.py` | **Migrated** (orchestration + diagnostics) |
| — | `spammm/forcefields/AssemblyPlot.py` | **SPAMMM** (plotting/diagnostics; split from driver) |

## 9. GridFF (Grid Force Field — B-spline Substrate Potential)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `pyBall/OCL/GridFF.py` | `spammm/surfaces/GridFF.py` | **Migrated** |
| `pyBall/OCL/GridFFRelaxedScan.py` | `spammm/surfaces/GridFFRelaxedScan.py` | **Migrated** |
| `cpp/common_resources/cl/GridFF.cl` | `kernels/surface.cl` (GridFF parts) | **Migrated** |

## 10. FoldedAtomicFunctions (FAF — Compact Basis Substrate Potential)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `doc/py/FoldedAtomicFunctions/` | `spammm/surfaces/FoldedRigid.py` | **Migrated** |
| `cpp/common_resources/cl/Surface.cl` (folded parts) | `kernels/surface.cl` (folded parts) | **Migrated** |

## 11. Surface.cl — Unified Surface Interaction Kernel

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `cpp/common_resources/cl/Surface.cl` | `kernels/surface.cl` | **Migrated** — Morse, folded, Ewald2D, macro corrections |

## 12. Ewald2D (2D Ewald Electrostatics for Slabs)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `pyBall/OCL/SurfaceEwald.py` | `spammm/surfaces/SurfaceEwald.py` | **Migrated** |
| `pyBall/Ewald2D.py` | `spammm/surfaces/Ewald2D.py` | **Migrated** |

## 13. Surface_utils.py (GridFF Alignment & FDBM Fitting)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `pyBall/OCL/Surface_utils.py` | `spammm/surfaces/Surface_utils.py` | **Migrated** |

## 14. AFM Simulation (Morse/LJ + FDBM)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `pyBall/OCL/AFM.py` | `spammm/SPM/AFM.py` | **Migrated** |
| `pyBall/OCL/AFM_utils.py` | `spammm/SPM/AFM_utils.py` | **Migrated** |
| `cpp/common_resources/cl/AFM.cl` | `kernels/AFM.cl` | **Migrated** |
| Modular pipeline | `spammm/SPM/ModularPipeline.py` | **Migrated** |
| AFM GUI extension | `spammm/GUI/AFMExtension.py` | **Migrated** |

## 15. STM Simulation

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `cpp/common_resources/cl/LCAO_STM.cl` | `kernels/LCAO_STM.cl` | **Migrated** |
| STM pipeline | `spammm/SPM/AFM.py` / `ModularPipeline.py` | **Migrated** |

## 16. DFTB+ / QM Integration

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `pyBall/dftb_utils.py` | `spammm/quantum/DFTB_utils.py` | **Migrated** |
| DFTB+ C-API wrapper | `spammm/quantum/DFTB/DFTBcore.py` | **Migrated** |
| DFTB+ parser | `spammm/quantum/DFTB/DFTBplusParser.py` | **Migrated** |
| Grid projection | `spammm/quantum/DFTB/Grid_dftb.py` | **Migrated** |
| pySCF utils | `spammm/quantum/pySCF_utils.py` | **Migrated** |

## 17. QEq (Charge Equilibration)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| QEq implementation | `spammm/forcefields/QEq.py` | **Migrated** |
| QEq GUI | `spammm/GUI/QEqExtension.py` | **Migrated** |

## 18. REQ → PLQ Conversion

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `NBFF::evalPLQs()` / `makePLQs()` | Embedded in `SPFF_cl.py`, `RigidBodyDynamics.py` | **Migrated** |

## 19. FFController (Central Orchestrator)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `cpp/common/molecular/MolWorld_sp3.h` | `spammm/forcefields/FFController.py` | **Migrated** (replaced C++ orchestrator with Python) |

## 20. Graphene Ribbon Builder

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `doc/Topics/Kekule_Topology/GrapheneRibbonBuilder.py` | `spammm/topology/KekuleBackend.py` (ribbon functions) | **Migrated** (integrated into KekuleBackend) |

## 21. KekuleSolver (Bond-Order Optimizer)

| FireCore File | SPAMMM File | Status |
|--------------|-------------|--------|
| `doc/Topics/Kekule_Topology/KekuleSolver.py` | `spammm/topology/KekulePure.py` | **Migrated** — reimplemented as pure Python NumPy optimizer |
| `pyBall/Kekule.py` (ctypes wrapper) | Not needed | **Skipped** — KekulePure.py replaces C++ dependency |

