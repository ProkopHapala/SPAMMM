# AFM/STM Simulation

> Cross-language audit of Atomic Force Microscopy (AFM) and Scanning Tunneling Microscopy (STM) simulation components: force field computation, tip relaxation, rigid body dynamics, electronic structure, and orbital projection. See [surface_interactions.md](surface_interactions.md) for GridFF substrate potentials, [intramolecular_forcefields.md](intramolecular_forcefields.md) for SPFF/UFF, and [nonbonding_forcefields.md](nonbonding_forcefields.md) for NBFF.

**Related Windsurf Codemaps:**
- [AFM PyOpenCL System: Morse/LJ Path and FDBM Density-Based Path](https://windsurf.com/codemaps/9bb4c2a5-0c38-4943-abe9-254cfdcc75af-8796fe608a7d71c1) — AFM force evaluation using GridFF and FDBM models.
- [AFM Simulation: GPU Rigid Body Dynamics, CPU GridFF Relaxation, and Interactive GUI](https://windsurf.com/codemaps/594f7eaf-c3ab-4139-8f20-d1d2d7f8d401-fe86ab10a43f3d18) — GPU rigid-body AFM with GridFF substrate potential.
- [AFM FDBM Pipeline: DFTB Backend & pySCF Integration Points](https://windsurf.com/codemaps/02d559c9-de47-4058-b07b-3318664b454e-fe86ab10a43f3d18) — DFTB-based force-field parameter derivation for AFM.
- [DFTB Reference Calculation & FDBM AFM Forcefield Comparison System](https://windsurf.com/codemaps/1153fe89-ff29-4d4b-b4a6-e97d8f37047f-fe86ab10a43f3d18) — DFTB reference vs classical surface interaction comparison.
- [Rigid Body Dynamics on Surfaces (pyOpenCL)](https://windsurf.com/codemaps/b5d9c2d2-50f0-4ba7-bc65-60db6e06e423-8796fe608a7d71c1) — Rigid-body dynamics with GridFF surface sampling.
- [Rigid Body Dynamics System for AFM Simulation](https://windsurf.com/codemaps/c9f13e1f-edfa-4702-814f-5036d03ea6c9-fe86ab10a43f3d18) — 6-DOF rigid body AFM tip mechanics.
- [GPU Green's Function STM Implementation](https://windsurf.com/codemaps/f398c2cf-5ff8-4d75-a398-c83e788e27b4-fe86ab10a43f3d18) — Current orbital projection & planned GF solver.
- [STM Simulation Pipeline: Orbital Projection & Quantum Transport](https://windsurf.com/codemaps/d0242216-c415-4f38-98f9-4c88b5dfeeb8-fe86ab10a43f3d18) — Orbital projection and quantum transport.
- [STM QMMM: Fireball DFTB Integration with GPU Density Projection](https://windsurf.com/codemaps/9fa40c64-e78c-42f2-9573-574936c8040d-fe86ab10a43f3d18) — Fireball DFTB integration with GPU density projection.

---

## Quick Navigation

| I want to... | Go to |
|--------------|-------|
| Simulate AFM with LJ/Morse tip | [AFMulator](#1-afmulator--pyballoclafmpy) |
| Simulate AFM with rigid body molecule on surface | [RigidBodyAFM](#3-rigidbodyafm--pyballoclrigidbodyafmpy) |
| Use GPU rigid body dynamics directly | [RigidBodyDynamics](#4-rigidbodydynamics--pyballoclrigidbodydynamicspy) |
| Run AFM from the GUI | [AFMExtension](#5-afmextension--pyballafmextensionpy) |
| Run the staged AFM pipeline | [ModularAFMPipeline](#6-modularafmpipeline--pyballoclmodularpipelinepy) |
| Simulate STM current / DOS | [STM module](#7-stm-module--pyballfireballoclstmpy) |
| Project molecular orbitals onto grids | [STM_utils](#8-stm_utils--pyballfireballoclstm_utilspy) |
| Understand the C++ AFM scanner app | [MolecularEditorOCL](#9-moleculareditorocl--c-app](#9-moleculareditorocl--c-app) |
| Understand the OpenCL rigid body kernel | [Rigid.cl](#10-rigidcl--opencl-kernel) |
| Find test scripts | [Test Coverage](#test-coverage) |
| Find documentation | [Documentation](#documentation) |

---

## Architecture Overview

```
AFM Simulation Pipeline
=======================

  Geometry (xyz) ──► AFMulator ──► Force Field ──► Tip Relaxation ──► AFM Image
                       │                                                    
                       ├── LJ/Morse path (relax.cl)                       
                       ├── FDBM path (precomputed GridFF)                  
                       └── Electrostatic (point charges / QEq)             
                                                                            
  RigidBodyAFM ──► RigidBodyDynamics ──► Rigid.cl ──► GridFF surface       
       │                                                                    
       └── Molecule attached to tip via harmonic spring                     
                                                                            
  ModularAFMPipeline ──► Stage S1: Geometry + DFTB+ SCF                    
       ├── Stage S2: Grid projection (density)                              
       ├── Stage S3: Potential calculation (Pauli/vdW/electrostatic)        
       ├── Stage S4: Force relaxation                                       
       ├── Stage S5: AFM image                                              
       └── Stage S6: STM / BR-STM                                           
                                                                            
  AFMExtension ──► KekuleExplorerGUI integration                           
       └── Dirty flag system (S1-S6) for incremental recomputation          


STM Simulation Pipeline
=======================

  Fireball SCF ──► H, S matrices ──► Spectral function A(E) ──► DOS / PDOS
       │                                        │
       │                                        ├── NEGF Caroli formula ──► STM current
       │                                        └── MO overlap ──► Featureless tip current
       │
       └── Orbital projection (STM_utils) ──► Grid.cl kernels ──► ψ(r) on 3D grid
```

---

## All AFM/STM Components Compared

| Component | Language | Backend | GPU? | Status |
|-----------|----------|---------|------|--------|
| **AFMulator** | Python | PyOpenCL (relax.cl) | Yes | **Active** |
| **AFM_utils** | Python | DFTB+/pySCF/Fireball | Mixed | **Active** |
| **RigidBodyAFM** | Python | PyOpenCL (Rigid.cl) | Yes | **Active** |
| **RigidBodyDynamics** | Python | PyOpenCL (Rigid.cl) | Yes | **Active** |
| **ModularAFMPipeline** | Python | DFTB+/pySCF | Mixed | **Active** |
| **AFMExtension** | Python | PyQt5 + pipeline | Mixed | **Active** |
| **STM** | Python | Fireball SCF + numpy | No | **Active** |
| **STM_utils** | Python | PyOpenCL (Grid.cl) | Yes | **Active** |
| **RigidBodyFF** | C++ | CPU | No | **Active** |
| **MolecularEditorOCL** | C++ | SDL2/OpenCL | Yes | **Active** |
| **Rigid.cl** | OpenCL | GPU kernel | Yes | **Active** |
| **relax.cl** | OpenCL | GPU kernel | Yes | **Active** |

---

## 1. AFMulator (`pyBall/OCL/AFM.py`)

### Purpose

Core OpenCL-accelerated AFM simulator. Computes tip-sample interactions (LJ/Morse, electrostatic, Pauli repulsion), relaxes the tip to generate constant-force or constant-height images, and handles molecule loading and scan management.

### Key Class: `AFMulator(OpenCLBase)`

**File**: `pyBall/OCL/AFM.py` (~667 lines)

- **Force field computation**: LJ/Morse potentials, electrostatic convolution with point charges, C6/R^6 dispersion
- **Tip relaxation**: `run_scan()`, `relaxStrokesTilted()` kernel for probe-particle relaxation
- **FDBM support**: `setup_fdbm_grid()`, `scan_fdbm()`, `scan_fdbm_2d()` for pre-computed force fields
- **Molecule loading**: `load_molecule()`, `assign_params()` with combination rules
- **Grid management**: `setup_grid()`, `setup_grid_lvec()` for periodic systems
- **Kernels**: Uses `relax.cl` with `evalLJC_QZs_toImg`, `evalMorseC_QZs_toImg`, `relaxStrokesTilted`

### Default CO-tip Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DEFAULT_tipA` | `[1,0,0,0]` | Tip orientation A |
| `DEFAULT_tipB` | `[0,1,0,0]` | Tip orientation B |
| `DEFAULT_tipC` | `[0,0,1,-0.1]` | Tip orientation C, `.w=dtip` |
| `DEFAULT_stiffness` | `[-0.03,-0.03,-0.03,-1.0]` | Spring constants (kx,ky,kz,kt) |
| `DEFAULT_dpos0` | `[0,0,-4.0,4.0]` | Probe equilibrium offset |
| `DEFAULT_relax_pars` | `[0.5,0.1,0.02,0.5]` | Relaxation parameters |
| `DEFAULT_tipQs` | `[0,-0.1,0.1,0]` | Tip charges |
| `DEFAULT_tipQZs` | `[0,1.8,3.6,0]` | Tip charge z-positions |

### Key Methods

| Method | Purpose |
|--------|---------|
| `load_molecule(xyz_path)` | Load molecule from XYZ, assign LJ parameters |
| `setup_grid(p0, L, n)` | Set up scan grid origin, size, resolution |
| `setup_fdbm_grid(...)` | Load precomputed FDBM force field as OpenCL image |
| `run_scan(...)` | Full scan: compute forces + relax tip at each pixel |
| `scan_fdbm(...)` | Scan using FDBM precomputed force field |
| `get_raw_FE(...)` | Sample force field without probe-particle relaxation |

---

## 2. AFM Utilities (`pyBall/OCL/AFM_utils.py`)

### Purpose

High-level orchestration, plotting, and integration with QM backends (DFTB+, pySCF, Fireball) for density projection. Bridges `AFM.py` physics with electronic structure calculations.

**File**: `pyBall/OCL/AFM_utils.py` (~749 lines)

### Key Functions

| Function | Purpose |
|----------|---------|
| `compose_and_relax(...)` | Orchestrate force field composition and probe relaxation |
| `compose_and_relax_total(...)` | Full pipeline: density → potential → force → relaxation |
| `get_density_from_dftb_dense(...)` | DFTB+ density (dense matrix) |
| `get_density_from_dftb_plus(...)` | DFTB+ density (sparse) |
| `get_density_from_pyscf(...)` | pySCF density (Gaussian orbitals) |
| `get_density_from_fireball(...)` | Fireball density |
| `_project_densities(geo, evecs, basis, grid_spec)` | Shared projection: returns (rho_scf, rho_na, rho_diff) |
| `plot_slices(...)`, `save_afm_images(...)`, `plot_grid_Fz(...)` | Plotting and visualization |

### QM Backend Integration

The module provides density provider adapters that produce `rho_scf` (SCF density), `rho_na` (neutral atom density), and `rho_diff` (difference density) grids:

- **DFTB+**: Uses `Grid_dftb` backends with Slater-type orbitals, GPU-accelerated projection
- **pySCF**: Uses Gaussian-type orbitals, CPU-based evaluation
- **Fireball**: Uses Fireball SCF eigenvectors

---

## 3. RigidBodyAFM (`pyBall/OCL/RigidBodyAFM.py`)

### Purpose

High-level class for AFM simulation using GPU rigid body dynamics. The molecule is attached via a harmonic spring to a moving "tip" (anchor point) and interacts with a GridFF surface.

**File**: `pyBall/OCL/RigidBodyAFM.py` (~225 lines)

### Key Class: `RigidBodyAFM`

| Method | Purpose |
|--------|---------|
| `__init__(mol_path, gridff_path, sub_xyz, ...)` | Initialize with molecule, GridFF, substrate |
| `prepare(n_bodies, initial_positions, initial_quats)` | Set up `RigidBodyDynamics` instance |
| `set_anchor_positions(positions)` | Move tip anchor to scan positions |
| `relax_to_constraint(niter, dt)` | Relax molecule under anchor constraint |
| `sample_gridff_single_atom(...)` | Sample GridFF at scan positions with single test atom |
| `plot_gridff_diagnostics(...)` | Visualize GridFF channels and substrate atoms |

### Anchor System

Each atom can have an anchor point with a spring constant `k`. The anchor is stored as `float4(x, y, z, k)` where `k > 0` means active. The tip is simulated by setting anchors on specific atoms and moving them across scan positions.

---

## 4. RigidBodyDynamics (`pyBall/OCL/RigidBodyDynamics.py`)

### Purpose

OpenCL-accelerated rigid body dynamics engine. Simulates multiple rigid bodies with 6-DOF (3 translational + 3 rotational via quaternions). Each rigid body is simulated within a single workgroup on the GPU.

**File**: `pyBall/OCL/RigidBodyDynamics.py` (~649 lines)

### Key Class: `RigidBodyDynamics(OpenCLBase)`

| Feature | Description |
|---------|-------------|
| State | Positions (`poss`), orientations (`qrots`), linear/angular velocities (`vposs`, `vrots`) |
| Body properties | Mass (in `poss.w`), inverse inertia tensor (`I_body_inv`) |
| Atom layout | `mols[gid]` maps body index → atom range; `apos_body` stores body-frame coordinates |
| Forces | GridFF surface forces, anchor springs, external E-field |
| Integration | Damped velocity Verlet with quaternion rotation via Taylor series |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_WORKGROUP_SIZE` | 32 | Threads per workgroup (must match kernel) |
| `DEFAULT_MAX_ATOMS_PER_BODY` | 128 | Maximum atoms per rigid body |
| `DEFAULT_ALPHA_MORSE` | 1.5 | Morse potential decay parameter |

### Key Methods

| Method | Purpose |
|--------|---------|
| `from_xyz_and_grid(mol_path, gridff_path, sub_xyz, ...)` | Factory: load molecule + GridFF + substrate |
| `update_anchors(anchors)` | Upload anchor spring positions |
| `run(niter, dt)` | Execute `niter` relaxation steps |
| `download_selected(keys)` | Download specific state arrays from GPU |
| `upload_PLQs(...)` | Upload atom PLQ parameters for GridFF sampling |

---

## 5. AFMExtension (`pyBall/AFMExtension.py`)

### Purpose

AFM simulation extension for `KekuleExplorerGUI`. Uses `ModularAFMPipeline` for staged, cached computation. Dirty flag system ensures only changed stages are recomputed.

**File**: `pyBall/AFMExtension.py` (~1194 lines)

### Dirty Flag System

`AFMDirtyFlags` tracks which pipeline stages are stale. Setting a stage dirty automatically marks all downstream stages dirty.

**Dependency chain**: `geometry/basis/step → S1 → S2 → S3 → S4 → S5/S6`

| Stage | Description | Dirty trigger |
|-------|-------------|---------------|
| S1 | Geometry + DFTB+ SCF | Geometry change |
| S2 | Grid projection (density) | Step/margin change |
| S3 | Potential calculation | Pauli/vdW params change |
| S4 | Force relaxation | Scan range/heights change |
| S5 | AFM image | Scan params change |
| S6 | STM / BR-STM | MO selection change |

### Key Functions

| Function | Purpose |
|----------|---------|
| `_get_afm_geometry(window)` | Convert backend geometry to AFM format |
| `_get_pipeline_params(window)` | Snapshot current UI parameter values |
| `_ensure_pipeline(window)` | Create or recreate `ModularAFMPipeline` if needed |
| `run_afm_scan(window)` | Execute full AFM scan from GUI |
| `run_stm_scan(window)` | Execute STM scan from GUI |

---

## 6. ModularAFMPipeline (`pyBall/OCL/ModularPipeline.py`)

### Purpose

Decoupled, stage-based modular pipeline for AFM and STM simulations. Saves intermediate results to disk allowing fast, independent stage execution. Supports multiple quantum chemistry backends.

**File**: `pyBall/OCL/ModularPipeline.py` (~589 lines)

### Key Class: `ModularAFMPipeline`

| Feature | Description |
|---------|-------------|
| Backends | `'dftb'` (DFTB+ with Slater-type orbitals, GPU projection) or `'pyscf'` (pySCF with Gaussian orbitals, CPU) |
| Geometry injection | Can accept `atomPos`/`enames` directly instead of XYZ file |
| Disk caching | Each stage saves to `output_dir` for incremental re-execution |
| Scan parameters | `step`, `margin`, `z_extra`, `scan_range`, `scan_step`, `height_range`, `height_step` |

### Pipeline Stages

1. **S1**: Run DFTB+ SCF (or pySCF) → obtain eigenvectors, eigenvalues
2. **S2**: Project density onto 3D grid → `rho_scf`, `rho_na`, `rho_diff`
3. **S3**: Compute potentials (Pauli repulsion, London dispersion, electrostatic)
4. **S4**: Relax probe particle over scan grid → force field
5. **S5**: Generate AFM frequency shift images
6. **S6**: STM / BR-STM (optional, uses orbital projection)

---

## 7. STM Module (`pyBall/FireballOCL/STM.py`)

### Purpose

Shared STM transport module for Fireball-based molecular junction simulation. Provides DOS computation with Γ broadening, NEGF Caroli formula for STM current, Wolfsberg-Helmholtz hopping model, and PDOS real-space projection.

**File**: `pyBall/FireballOCL/STM.py` (~749 lines)

### Key Functions

| Function | Purpose |
|----------|---------|
| `compute_dos(atomTypes, atomPos, gamma, ...)` | Run Fireball SCF, compute spectral function PDOS, save to `.npz` |
| `build_dense_HS(dims, data, atomTypes)` | Build dense H[norb,norb] and S[norb,norb] from Fireball sparse output |
| `compute_hopping(...)` | Wolfsberg-Helmholtz hopping model for inter-system coupling |
| `build_inter_system_blocks_exp_sk(...)` | Vacuum exponential + SK angular coupling between tip and sample |
| `stm_current(...)` | Calculate STM current via NEGF Caroli formula |
| `negf_current(...)`, `negf_current_iterative(...)` | NEGF transport (direct and iterative solvers) |
| `mo_overlap_amplitude(...)` | Featureless tip approximation: MO overlap amplitude |
| `response_amplitude_map(...)` | Compute STM response amplitude map |
| `pad_HS_to_float4(...)` | Pad H/S for GridFF-like float4 layout |
| `project_pdos_to_grid(...)` | Real-space PDOS projection onto 3D grid |

### Physics

**Spectral function**: $A(E) = \Gamma / [(E - H_{\text{eff}})^2 + \Gamma^2]$ with broadening $\Gamma$.

**NEGF Caroli formula**: $I(V) = \frac{2e}{h} \int T(E, V) [f_L(E) - f_R(E)] dE$

**Wolfsberg-Helmholtz hopping**: $H_{ij} \approx \frac{1}{2} K (H_{ii} + H_{jj}) S_{ij}$ where $K \approx 1.75$.

### Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `E_pad` | 1e3 | Padding energy for unused orbitals in float4 layout |

---

## 8. STM_utils (`pyBall/FireballOCL/STM_utils.py`)

### Purpose

Utilities for orbital mapping conventions (Fortran to OpenCL), building grid specifications, projecting molecular orbitals to grids or points, and parity verification between Fortran and OpenCL implementations.

**File**: `pyBall/FireballOCL/STM_utils.py` (~826 lines)

### Orbital Ordering Conventions (Critical)

| Convention | Order | Mapping from Fortran |
|------------|-------|---------------------|
| **Fortran/Fireball** | `[s, py, pz, px]` | — (reference) |
| **OpenCL Hamiltonian** | `[s, px, py, pz]` | `_PERM_FORT_TO_HAM = [0, 3, 1, 2]` |
| **OpenCL Grid** | `[px, py, pz, s]` | `_PERM_FORT_TO_GRID = [3, 1, 2, 0]` |

Hydrogen (1 orbital): pack into Grid order as `[0, 0, 0, s]`.

### Key Functions

| Function | Purpose |
|----------|---------|
| `get_orbital_layout(sparse_data, natoms)` | Per-atom orbital counts from Z → `nzx` → `num_orb` |
| `remap_coeffs_fortran_to_grid(coeffs, norb_per)` | Remap MO coefficients Fortran → Grid kernel order |
| `remap_coeffs_fortran_to_hamiltonian(coeffs, norb_per)` | Remap MO coefficients Fortran → Hamiltonian order |
| `build_grid_spec(atom_center, Lx, Ly, Lz, nx, ny, nz, ...)` | Build grid spec for OpenCL projection |
| `project_orbital_to_grid(C, mo_idx, ...)` | Project single MO onto 3D grid (density kernel) |
| `project_orbital_to_grid_v2(C, mo_idx, ...)` | Project single MO onto 3D grid (orbital kernel, signed ψ) |
| `project_orbital_to_points(...)` | Evaluate MO at arbitrary points (SK angular) |
| `project_orbital_to_points_exp(...)` | Evaluate MO at arbitrary points (exponential radial) |
| `sparse_to_dense(...)` | Build dense H/S from sparse Fireball blocks |
| `compute_correlation_stats(...)` | Compare reference vs test orbital data |

### Grid Sampling Parity

Fortran `orb2points()` samples at voxel centers; OpenCL grid kernels sample at voxel corners. For parity, shift origin by half-step: `0.5*(dA + dB + dC)` when `voxel_center_sampling=True`.

### Sparse Block Layout

`H_blocks` from `fc.get_HS_sparse(dims, data)` is viewed as `H_blocks[iatom, ineigh, imu, inu]` where `imu` runs orbitals on iatom (row), `inu` runs orbitals on jatom (col). Self-neighbor is detected by `(neigh_j == iatom+1 && neigh_b == 0)`.

---

## 9. MolecularEditorOCL (C++ App)

### Purpose

SDL2/OpenGL C++ application for interactive AFM scanning with GPU-accelerated rigid body molecular dynamics and GridFF surface interaction.

**File**: `cpp/apps_OCL/MolecularEditorOCL/MolecularEditorOCL_scanner.cpp` (~754 lines)

### Key Class: `AppMolecularEditorOCL`

| Component | Description |
|-----------|-------------|
| `GridFF_OCL gridFFocl` | OpenCL GridFF wrapper for surface potential |
| `RigidMolecularWorldOCL clworld` | Multi-system rigid body molecular world |
| `SPFF world` | Molecular mechanics force field |
| `DynamicOpt opt` | CPU dynamics optimizer (FIRE) |

### Key Operations

- **Substrate loading**: `world.gridFF.loadCell()`, `world.gridFF.loadXYZ()`, `world.genPLQ()`
- **GPU grid evaluation**: `gridFFocl.evalGridFFs(world.gridFF, {1,1,1})`
- **GPU relaxation**: `clworld.relaxStepGPU(niter, dt)` — rigid body relaxation on GridFF surface
- **Multi-system**: Supports `nSystems` copies for parallel scanning
- **Visualization**: Real-time rendering of forces, torques, and substrate isosurfaces

### Test Script

`tests/tMolGUIapp_QMMM_multi/run_AFM.sh` runs the C++ AFM scanner:
```bash
./$name -m 1 -x common_resources/xyz/PTCDA -g common_resources/xyz/NaCl_1x1_L2 -iParalel 3 -iMO 3 -substr_iso 0.05
```

---

## 10. Rigid.cl (OpenCL Kernel)

### Purpose

OpenCL kernel for rigid body dynamics with GridFF surface interaction. Used by both Python (`RigidBodyDynamics.py`) and C++ (`MolecularEditorOCL`) paths.

**File**: `cpp/common_resources/cl/Rigid.cl` (449 lines)

### Kernels

**`rigid_body_dynamics_kernel`** (line 195):
- Basic rigid body simulation with external E-field and anchor springs
- Each workgroup handles one rigid body (32 threads, up to 128 atoms)
- Quaternion rotation via Taylor series expansion
- Damped velocity Verlet integration

**`rigid_body_gridff_kernel`** (line 312):
- GridFF surface interaction with B-spline interpolation
- Samples `BsplinePLQ` grid at atom positions via `fe3d_pbc_comb()`
- Computes per-atom forces and torques, reduces across workgroup
- Configurable damping (`md_params.x` = linear, `md_params.y` = angular)
- Debug output via `RIGID_DBG` macro

### Key Helper Functions

| Function | Line | Purpose |
|----------|------|---------|
| `quat_mult(q1, q2)` | 14 | Quaternion multiplication |
| `quat_factors_taylor(r2)` | 32 | Taylor series for sin(r/2)/r and cos(r/2) |
| `make_qrot_taylor(omega)` | 47 | Build rotation quaternion from angular velocity |
| `quat_to_a/b/c(q)` | 80-82 | Quaternion → rotation matrix rows |
| `fe3d_pbc_comb(u, n, Es, PLQH, ...)` | 156 | 3D B-spline interpolation with PBC |
| `basis(u)`, `dbasis(u)` | 116, 128 | Cubic B-spline basis and derivative |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `WORKGROUP_SIZE` | 32 | Threads per workgroup |
| `MAX_ATOMS_PER_BODY` | 128 | Max atoms per rigid body |
| `ATOMS_PER_THREAD` | 4 | Atoms processed per thread per iteration |

---

## 11. RigidBodyFF (C++ Header)

### Purpose

C++ rigid body force field manager. Projects atoms from body frame to world frame, evaluates torques, and rotates quaternions.

**File**: `cpp/common/molecular/RigidBodyFF.h` (82 lines)

### Key Class: `RigidBodyFF`

| Method | Purpose |
|--------|---------|
| `projectAtoms()` | Rotate body-frame atoms to world frame using quaternion |
| `evalTorqs()` | Evaluate torque on each rigid body from NBFF forces |
| `rotateQuaternions(dt)` | Update quaternions from angular velocity |
| `realloc(n, ps, qrots, frots, vrots, apos0, mols)` | Bind or allocate arrays |
| `makePos0s(na)` | Store initial body-frame atomic positions |

---

## 12. relax.cl (OpenCL Kernel)

### Purpose

Unified force field kernel for AFM probe-particle relaxation. Used by `AFMulator` for LJ/Morse + electrostatic force computation and tip relaxation.

**File**: `cpp/common_resources/cl/relax.cl`

### Key Kernels Used by AFMulator

| Kernel | Purpose |
|--------|---------|
| `evalLJC_QZs_toImg` | Evaluate LJ + Coulomb forces at scan positions |
| `evalMorseC_QZs_toImg` | Evaluate Morse + Coulomb forces at scan positions |
| `relaxStrokesTilted` | Relax probe particle at each scan pixel (FIRE or damped velocity) |

Build option `-DOPT_FIRE=1` enables FIRE relaxation, `-DOPT_FIRE=0` uses damped velocity.

---

## Test Coverage

### AFM Tests (`tests/tAFM/`)

| File | Purpose |
|------|---------|
| `test_fdbm.py` | FDBM force field testing |
| `test_gradient_kernel.py` | Gradient kernel validation |
| `test_gradient_simple.py` | Simple gradient tests |
| `test_gradient_visual.py` | Visual gradient debugging |
| `test_ptcda.py` | PTCDA molecule AFM simulation |
| `test_single_atom.py` | Single atom AFM |
| `afm_morse_pbc.py` | Morse potential with PBC |
| `run.sh`, `run_fdbm.sh` | Test runner scripts |

### AFM PyOpenCL FDBM Tests (`tests/tAFM/pyocl_fdbm/`)

| File | Purpose |
|------|---------|
| `run_pyocl_fdbm.py` | Main PyOpenCL FDBM runner (~63K) |
| `run_pyocl_fdbm_dftb.py` | DFTB+ integration (~62K) |
| `run_pyocl_fdbm_dftb_pentacene.py` | Pentacene-specific tests |
| `test_full_pipeline.py` | Full pipeline validation |
| `test_modular_pipeline.py` | Modular pipeline testing |
| `fit_fdbm_pauli.py` | Pauli coefficient fitting |
| `test_fit_pauli_pyscf.py` | pySCF fitting validation |
| `compare_fireball_dftb.py` | Fireball vs DFTB comparison |
| `diagnostic_forcefield.py` | Force field diagnostics |
| `compute_co_tip.py` | CO-tip parameter computation |
| `run_dftb_zscan.py` | DFTB+ z-scan |
| `plot_transition.py` | Transition plotting |

### Rigid Body AFM Tests (`tests/tSPFF/`)

| File | Purpose |
|------|---------|
| `test_rigid_afm_ml.py` | Machine learning for rigid AFM |
| `run_rigid_afm_scan.py` | Rigid AFM scan execution |
| `test_rigid_gridff_surface.py` | GridFF surface interaction |
| `test_rigid_gridff_ptcda_batch.py` | Batch PTCDA tests |
| `test_fdbm_fit_dft.py` | FDBM fit against DFT reference |
| `test_fdbm_fit_gridff_mock.py` | Mock FDBM fitting |
| `TipSplineOptimizer.py` | Tip spline optimization |
| `ManipulationPathOpt.py` | Manipulation path optimization |

### STM Tests (`tests/pyFireball/`)

| File | Purpose |
|------|---------|
| `test_stm_orbital_projection.py` | Orbital projection validation |
| `test_stm_homo.py` | HOMO STM simulation |
| `test_stm_orbital_rotated.py` | Rotated orbital tests |
| `test_stm_orbital_rotated_2mol.py` | Two-molecule STM |
| `test_stm_gf_dyson_2mol.py` | Green's function Dyson equation |
| `test_stm_gf_dyson_2mol_ocl.py` | OpenCL Green's function |
| `test_stm_dyson_fortran.py` | Fortran reference comparison |
| `test_h2o_mo_vs_ldos.py` | MO vs LDOS comparison |
| `test_h2o_orbital_comparison.py` | Orbital coefficient parity |
| `test_response_function.py` | Response function testing |
| `test_response_function_rotated.py` | Rotated response function |
| `stm_analysis.py` | STM data analysis |
| `stm_compute_dos.py` | DOS computation |

### C++ AFM Test (`tests/tMolGUIapp_QMMM_multi/`)

| File | Purpose |
|------|---------|
| `run_AFM.sh` | C++ MolecularEditorOCL AFM scanner runner |

---

## Documentation

### AFM Documentation (`doc/Topics/AFM/`)

| File | Content |
|------|---------|
| `AFM.md` | Full AFM simulation pipeline description |
| `AFM_migration_plan.md` | 2025 PyOpenCL migration plan |
| `AFM_migration.progress.md` | Migration progress tracking |
| `AFM_migration_discusion.chat.md` | Migration discussion |
| `AFM_FDBM_DFTB.chat.md` | FDBM + DFTB integration |
| `AFM_FDBM_fitting.chat.md` | FDBM parameter fitting |
| `AFM_FDBM_optimization.chat.md` | Performance optimization |
| `AFM_FDBM_profiling_optimization.chat.md` | Profiling results |
| `AFM_FDBM_pySCF.chat.md` | pySCF integration |
| `DFTB_Perturbation_Pauli.chat.md` | DFTB perturbation theory |
| `IndentationForce2D.chat.md` | 2D indentation force analysis |
| `dense_projection_integration_plan.md` | Dense projection planning |

### STM Documentation (`doc/Topics/STM/`)

| File | Content |
|------|---------|
| `STM_GF_new.chat.md` | Green's function STM workflow |
| `STM_GPU_QMMM.chat.md` | GPU-accelerated QM/MM STM |

### OnSurfaceAssembly Documentation (`doc/Topics/OnSurfaceAssembly/`)

| File | Content |
|------|---------|
| `AFM_File_Analysis_and_Architecture.md` | AFM file mapping |
| `RigidBodyAFM.chat.md` | Rigid body AFM discussion |
| `ML_AFM_manipulation.report.md` | ML for AFM manipulation |
| `GridFF_FDBM_Fitting.chat.md` | GridFF FDBM fitting |
| `GridFF_RelaxedScan_cpp_notes.guide.md` | C++ relaxed scan notes |

---

## Key Physical Constants

| Constant | Value | Source |
|----------|-------|--------|
| `COULOMB_CONST` | 14.3996448915 eV·Å/e² | `AFM.py` |
| `bohr2ang` | 0.5291772109217 | `test_fdbm_fit_dft.py` |
| `hartree2ev` | 27.211396132 | `test_fdbm_fit_dft.py` |
| `DEFAULT_ALPHA_MORSE` | 1.5 | `RigidBodyDynamics.py` |
| `DEFAULT_WORKGROUP_SIZE` | 32 | `RigidBodyDynamics.py` |
| `DEFAULT_MAX_ATOMS_PER_BODY` | 128 | `RigidBodyDynamics.py` |

---

## Detailed Interactions & Data Flow

### AFM Pipeline Data Flow (GUI → Physics → Visualization)

```
KekuleExplorerGUI
  └── AFMExtension.build_ui() creates panel widgets
       └── User clicks "Run Full AFM Pipeline"
            └── run_afm_full_pipeline(window)
                 ├── _get_afm_geometry(window)  →  atomPos, atomTypes, enames from backend
                 ├── _ensure_pipeline(window)   →  ModularAFMPipeline (created or reused)
                 │    └── ModularAFMPipeline.__init__()
                 │         ├── _init_geometry_and_grids()  →  scan_xs, scan_ys, heights
                 │         └── _init_dftb_backend()        →  GridProjector, atoms_dict
                 │
                 ├── Stage 1: pipe.stage1_scf()
                 │    └── DFTBcore.init() → run_scf() → get_dm_dense(), get_eigvecs_dense()
                 │         → cached to cache_stage1_scf.npz
                 │         → window._afm_eigvecs, window._afm_eigvals
                 │
                 ├── Stage 2: pipe.stage2_project(dm_dense)
                 │    └── projector.project_density_dense()  →  rho_scf (GPU)
                 │    └── dg.project_neutral_density()        →  rho_na
                 │    └── rho_diff = rho_scf - rho_na
                 │         → cached to cache_stage2_grids.npz
                 │         → window._afm_density
                 │
                 ├── Stage 3: pipe.stage3_potentials(rho_scf, rho_na, rho_diff)
                 │    ├── afm.fft_poisson(rho_diff)           →  V_ES
                 │    ├── afm.compute_pauli_overlap(rho_scf)  →  E_pauli_field
                 │    ├── afm.compute_es_conv_field(V_ES)     →  E_ES_field
                 │    ├── afm.compute_dispersion_grid()       →  E_vdw
                 │    └── AFMulator.compute_gradient_cl()     →  F_total (GPU)
                 │         → cached to cache_stage3_potentials.npz
                 │         → window._afm_potentials
                 │
                 ├── Stage 4: pipe.stage4_relax(F_total)
                 │    └── afm_utils.compose_and_relax_total()
                 │         └── AFMulator.scan_fdbm()  →  df, tip_disp (GPU relax.cl)
                 │         → cached to cache_stage4_relax.npz
                 │         → window._afm_results
                 │
                 ├── Stage 5 (optional): pipe.stage5_stm(eigvecs, eigvals)
                 │    └── afm_utils.compute_stm()  →  stm_grid
                 │
                 └── Stage 6 (optional): pipe.stage6_br_stm(eigvecs, eigvals, tip_disp)
                      └── afm_utils.compute_bond_resolved_stm()  →  br_stm_grid

  Visualization:
  └── plot_afm_slice(window)
       ├── Component selector: "AFM Image (df)" | "STM Signal" | "SCF Density" | ...
       ├── Z-height spinner → grid slice index
       ├── matplotlib Figure → imshow with colormap
       ├── _overlay_atoms() → atom position dots
       └── _show_in_plot_window() → reusable QDialog with FigureCanvas

  Orbital Plot:
  └── plot_orbital_map(window)
       ├── MO index from spinner
       ├── pipe.projector.project_orbital_dense_points_exp()  →  ψ(r) at 2D grid
       └── matplotlib seismic colormap (blue=neg, red=pos)
```

### Rigid Body AFM Data Flow (Alternative path)

```
RigidBodyAFM
  └── prepare()
       └── RigidBodyDynamics.from_xyz_and_grid()
            ├── Load molecule XYZ + GridFF + substrate XYZ
            ├── Compile Rigid.cl kernel
            └── Upload atom positions, PLQs, poses, quaternions to GPU
  └── set_anchor_positions(scan_positions)
       └── Upload anchor springs (x,y,z,k) to GPU
  └── relax_to_constraint(niter, dt)
       └── Enqueue rigid_body_gridff_kernel
            ├── B-spline interpolation of GridFF at atom positions
            ├── Force & torque computation per rigid body
            ├── Quaternion rotation update
            └── Damped velocity Verlet integration
       └── Download atom_positions, forces, torques
```

### C++ MolecularEditorOCL Data Flow

```
AppMolecularEditorOCL
  ├── init: Load AtomTypes.dat, BondTypes.dat
  ├── Load molecule types (XYZ) → builder.loadMolType()
  ├── Insert molecules → builder.insertMolecule()
  ├── Build SPFF → builder.toSPFF(&world)
  ├── initRigidSubstrate():
  │    ├── Load substrate cell (.lvs) + XYZ
  │    ├── genPLQ() → REQ → PLQ conversion
  │    ├── gridFFocl.evalGridFFs() → GPU GridFF evaluation
  │    └── renderSubstrate_() → OpenGL display list for isosurface
  ├── clworld.prepareBuffers() → GPU buffers for nSystems × nMols
  ├── clworld.relaxStepGPU(500, 0.5) → GPU rigid body relaxation
  └── draw() per frame:
       ├── clworld.relaxStepGPU(1, 0.5) → one GPU step
       ├── clworld.system2atoms() → download atoms
       ├── drawRigidMolSystem() → draw atoms + COG lines
       └── drawRigidMolSystemForceTorq() → draw force/torque vectors
```

### Key Interaction Points Between Components

| From | To | Interface | Data |
|------|----|-----------|------|
| `AFMExtension` | `ModularAFMPipeline` | `_ensure_pipeline(window)` | atomPos, enames, params → pipeline |
| `ModularAFMPipeline` | `AFMulator` | `stage3_potentials()`, `stage4_relax()` | F_total → AFMulator for gradient/relax |
| `ModularAFMPipeline` | `AFM_utils` | `stage4_relax()` → `compose_and_relax_total()` | F_total, scan grid → df, tip_disp |
| `ModularAFMPipeline` | `Grid_dftb` | `stage2_project()` | dm_dense → rho_scf via GPU projector |
| `AFMExtension` | `KekuleExplorerGUI` | `window.backend.sys.apos` | Geometry access |
| `AFMExtension` | `KekuleExplorerGUI` | `window.sig_geometry_changed` | Dirty flag trigger |
| `RigidBodyAFM` | `RigidBodyDynamics` | `prepare()`, `relax_to_constraint()` | Molecule + GridFF setup |
| `RigidBodyDynamics` | `Rigid.cl` | OpenCL kernel enqueue | GPU buffers (poses, quats, atoms, GridFF) |
| `AFMulator` | `relax.cl` | OpenCL kernel enqueue | Force evaluation + tip relaxation |
| `MolecularEditorOCL` | `RigidMolecularWorldOCL` | `relaxStepGPU()` | Multi-system rigid body GPU relaxation |
| `MolecularEditorOCL` | `GridFF_OCL` | `evalGridFFs()` | GPU substrate potential evaluation |
| `STM` | `FireCore` (Fortran) | `fc.init()`, `fc.SCF()`, `fc.get_HS_sparse()` | SCF → H, S matrices |
| `STM_utils` | `Grid.cl` | `project_orbital_to_grid_v2()` | MO coefficients → ψ(r) on 3D grid |

---

## Consolidation Plan: Unified Vispy GUI

### Current State: Two Separate GUIs

| Feature | KekuleExplorerGUI (Python) | MolecularEditorOCL (C++) |
|---------|---------------------------|-------------------------|
| **Framework** | PyQt5 + Vispy | SDL2 + OpenGL |
| **3D Scene** | `AtomScene` (Vispy) | Custom SDL2/GL |
| **Molecule editing** | Yes (Hex1/Hex2/Atom/Bond/pi/Select) | No (view only) |
| **AFM scanning** | Yes (via AFMExtension + ModularAFMPipeline) | Yes (via RigidMolecularWorldOCL) |
| **AFM mode** | Probe-particle (FDBM, relax.cl) | Rigid body (Rigid.cl, GridFF) |
| **STM** | Yes (via pipeline stage5/6) | No |
| **GridFF surface** | No (isolated molecule AFM) | Yes (substrate interaction) |
| **Rigid body dynamics** | No | Yes (multi-system GPU) |
| **Visualization** | matplotlib popups for 2D slices | Real-time 3D force/torque vectors |
| **Orbital visualization** | 2D matplotlib slice | No |
| **Multi-system** | No | Yes (100 systems in parallel) |
| **Manipulation** | Atom picking/dragging | Atom picking (spring force) |

### Target: Single Unified Python+Vispy GUI

#### Architecture

```
UnifiedAFMStudio (PyQt5 + Vispy)
├── 3D Scene (extended AtomScene)
│   ├── Molecule visualization (atoms, bonds, labels)
│   ├── Substrate visualization (isosurface from GridFF)
│   ├── Force/torque vector overlays
│   ├── Tip position visualization (probe particle or rigid body)
│   ├── Orbital isosurface rendering (ψ(r) from STM_utils)
│   └── Scan grid overlay (scan_xs × scan_ys × heights)
│
├── Side Panel (CollapsibleSections)
│   ├── Editors Section (from KekuleExplorerGUI)
│   │   ├── Edit mode (Hex1/Hex2/Atom/Bond/pi/Select)
│   │   ├── Atom type, auto-H, auto-bonds
│   │   └── Export XYZ
│   │
│   ├── AFM Section (from AFMExtension)
│   │   ├── Pipeline controls (S1-S6 + full pipeline)
│   │   ├── Dirty flag status display
│   │   ├── Parameters (basis, step, margin, scan, physics)
│   │   ├── Visualization (component selector, z-height, colormap)
│   │   └── STM/Orbitals (MO list, field type, BR-STM)
│   │
│   ├── Rigid Body Section (NEW — from MolecularEditorOCL)
│   │   ├── Substrate loading (.lvs + .xyz)
│   │   ├── GridFF evaluation
│   │   ├── Molecule-on-surface relaxation
│   │   ├── Anchor spring configuration
│   │   ├── Multi-system scan
│   │   └── Force/torque visualization toggle
│   │
│   └── Diagnostics Section
│       ├── Energy component panels (Pauli/ES/vdW)
│       ├── Convergence monitoring
│       └── Export results (NPZ/PNG)
│
└── 2D Plot Panel (embedded Vispy or matplotlib)
    ├── AFM image slices (df, Fz, potentials)
    ├── STM signal maps
    ├── Orbital phase maps
    └── Z-slice slider with live update
```

#### What to Copy Where

**From `AFMExtension.py` → Unified GUI:**
- **Copy directly**: `AFMDirtyFlags` class (no changes needed)
- **Copy directly**: `_get_pipeline_params()`, `_get_stm_params_from_ui()`, `_get_homo_index()`, `_update_homo_label()`
- **Copy directly**: `run_afm_full_pipeline()`, `run_afm_stage1-4()`, `run_stm()` — these are GUI-agnostic, take `window` param
- **Adapt**: `build_ui()` — restructure into CollapsibleSection within unified panel
- **Adapt**: `plot_afm_slice()`, `plot_orbital_map()`, `plot_afm_diagnostic_panel()` — replace matplotlib with Vispy 2D plots or keep matplotlib for 2D (simpler)
- **Adapt**: `_overlay_atoms()` — use Vispy markers instead of matplotlib
- **Copy directly**: `_ensure_pipeline()`, `_ensure_stages_for_component()` (pipeline management)

**From `KekuleExplorerGUI.py` → Unified GUI:**
- **Copy directly**: `AtomScene` usage pattern (Vispy scene + axes)
- **Copy directly**: Extension manager pattern (`ExtensionManager`, `_build_extension_panels()`)
- **Copy directly**: `create_editors_section()`, `create_ribbon_section()`
- **Copy directly**: `sig_geometry_changed` signal pattern
- **Adapt**: `initUI()` — extend with substrate visualization and rigid body controls

**From `MolecularEditorOCL_scanner.cpp` → Unified GUI (Python port):**
- **Port to Python**: `initRigidSubstrate()` → load .lvs + .xyz, genPLQ(), GridFF evaluation
  - Already available: `RigidBodyDynamics.from_xyz_and_grid()` does this
- **Port to Python**: `drawRigidMolSystem()` → Vispy markers + lines for rigid body atoms
  - Already available: `RigidBodyAFM` class wraps this
- **Port to Python**: `drawRigidMolSystemForceTorq()` → Vispy line vectors for forces/torques
  - Need: Add force/torque vector visualization to `AtomScene` (force_lines already exists!)
- **Port to Python**: Substrate isosurface rendering → Vispy `Volume` visual
  - Need: Convert GridFF 3D scalar field to Vispy Volume
- **Port to Python**: Multi-system parallel relaxation → `RigidBodyDynamics` already supports n_bodies
- **Port to Python**: Atom picking with spring force → `AtomScene` already has picking; add spring manipulation mode

**From `RigidBodyAFM.py` → Unified GUI:**
- **Copy directly**: `RigidBodyAFM` class as backend for rigid body AFM mode
- **Copy directly**: `prepare()`, `set_anchor_positions()`, `relax_to_constraint()`
- **Adapt**: `sample_gridff_single_atom()` → use for scan grid visualization
- **Adapt**: `plot_gridff_diagnostics()` → integrate into diagnostics section

**From `STM.py` / `STM_utils.py` → Unified GUI:**
- **Copy directly**: `compute_dos()` for Fireball SCF + DOS
- **Copy directly**: orbital projection functions from `STM_utils`
- **New**: Add 3D orbital isosurface visualization using Vispy `Volume` visual
- **New**: Add DOS/PDOS plot panel

#### New Components Needed

1. **`SubstrateVisualizer`** (Vispy): Render GridFF isosurface in 3D scene
   - Use `vispy.scene.visuals.Volume` for 3D scalar field
   - Toggle between Pauli/ES/vdW/total channels
   - Isosurface level slider

2. **`ForceVectorOverlay`** (Vispy): Draw force/torque vectors on atoms
   - `AtomScene.force_lines` already exists — extend with torque visualization
   - Color by magnitude, scale by slider

3. **`ScanGridOverlay`** (Vispy): Visualize AFM scan grid in 3D
   - Wireframe box showing scan volume
   - Current tip position marker
   - Height slice indicator

4. **`OrbitalVolumeVisualizer`** (Vispy): 3D orbital isosurface
   - Use `Volume` visual with signed colormap (seismic)
   - MO selector dropdown
   - Isosurface level slider

5. **`RigidBodyPanel`** (PyQt5): Controls for rigid body AFM mode
   - Substrate file loaders
   - Anchor atom selector
   - Spring constant slider
   - Relaxation parameters (niter, dt, damping)
   - Multi-system count

6. **`STMDOSPanel`** (PyQt5): DOS/PDOS visualization
   - Energy range slider
   - Gamma broadening
   - Contact atom selector
   - PDOS plot (matplotlib or Vispy 2D)

#### Feature List for Unified GUI

**Molecule Editing (from KekuleExplorerGUI):**
- Hex1/Hex2/Atom/Bond/pi/Select edit modes
- Auto H-capping, auto bonds
- Grid snapping
- Atom labels (element+index, type, pi orbitals, z-height, charge, bond lengths)
- Export XYZ

**AFM Simulation (from AFMExtension):**
- Full pipeline with dirty flags (S1-S6)
- Individual stage execution
- Parameter control (basis, step, margin, scan range, heights, Pauli A/beta, C6, K_LAT)
- Basis presets (mio-1-1, 3ob-3-1) with auto Pauli params
- Component visualization (df, density, potentials, forces)
- Z-slice slider with live update
- Diagnostic panel (4-panel energy components)
- Atom overlay on 2D plots

**STM Simulation (from AFMExtension + STM):**
- MO list selection (relative to HOMO or absolute)
- Field type (ldos, psi², ψ)
- Bond-resolved STM
- Orbital map with phase (seismic colormap)
- exp_beta, exp_r0 parameters
- HOMO/LUMO info display

**Rigid Body AFM (from MolecularEditorOCL + RigidBodyAFM):**
- Substrate loading (.lvs + .xyz)
- GridFF surface visualization (isosurface)
- Molecule-on-surface relaxation
- Anchor spring configuration
- Multi-system parallel scan
- Force/torque vector visualization
- CPU/GPU relaxation toggle

**3D Visualization (new + existing):**
- Real-time 3D molecule view (atoms, bonds)
- Substrate isosurface
- Force/torque vectors
- Scan grid overlay
- Orbital isosurface (3D)
- Tip/probe position tracking
- Camera controls (top/side/free view)

**Data Management:**
- Pipeline disk caching (existing)
- Export results (NPZ, PNG)
- Save/load scan configurations
- Substrate preset library

#### Implementation Priority

1. **Phase 1**: Merge AFMExtension into KekuleExplorerGUI (already done — just verify)
2. **Phase 2**: Add RigidBodyAFM panel with substrate loading and GridFF visualization
3. **Phase 3**: Add 3D orbital isosurface visualization
4. **Phase 4**: Add force/torque vector overlays in 3D scene
5. **Phase 5**: Add multi-system parallel scanning
6. **Phase 6**: Add STM DOS/PDOS panel
7. **Phase 7**: Deprecate C++ MolecularEditorOCL (all features ported to Python)

#### What Can Be Eliminated

- **C++ `MolecularEditorOCL_scanner.cpp`**: Once all features are ported to Python+Vispy, this becomes redundant. The C++ app's unique features (substrate viz, rigid body, multi-system) are all available via `RigidBodyDynamics` + `RigidBodyAFM` in Python.
- **matplotlib 2D plots**: Can eventually be replaced with Vispy 2D plots for consistency, but matplotlib is fine for now (simpler, more mature for 2D scientific plots).
- **`AFM_utils.run_afm_pipeline()`**: This is the old monolithic pipeline function. `ModularAFMPipeline` supersedes it with staged execution and caching. Can be deprecated.

---

## See Also

- [Topical Audit Index](topical_audit.md) — priority ranking, dependency graph, missing topics
- [Surface Interactions](surface_interactions.md) — GridFF, FoldedAtomicFunctions, Ewald2D, Surface.cl
- [Intramolecular Force Fields](intramolecular_forcefields.md) — SPFFsp3, UFF, ProjectiveDynamics, XPBD
- [Nonbonding Force Fields](nonbonding_forcefields.md) — NBFF, exclusion schemes, FMM
- [Force Fields Overview](forcefields_overview.md) — High-level taxonomy
- [H-transfer & DFTB+](Htransfer_Kekule_DFTB.md) — DFTB+ integration details
- [GUI Feature Audit](gui_audit.md) — visualization & editor feature matrices, VisPy consolidation plan

---

*Last updated: 2026-06-23*
