# Kernel Inventory

## Current File Structure (post-reorganization)

```
kernels/
├── common.cl              # Shared types, macros, constants, math helpers
├── Forces.cl              # Inline pairwise potential functions (not __kernel)
├── nonbonded.cl           # Non-bonded LJ/Coulomb/H-bond kernels (2nd-neighbor exclusion)
├── gridFF.cl              # Grid FF construction, B-spline interpolation, Poisson solver
├── surface.cl             # Surface electrostatics (Ewald2D, folded basis, brute-force)
├── SPFF.cl                # SPFFsp3 bonding + MD integrator (pi-orbitals, recoil)
├── UFF.cl                 # UFF bonding + simplified MD integrator (no pi-orbitals)
├── rigid.cl               # 6-DOF rigid body dynamics with quaternion integration
├── assembly.cl            # Rigid-body molecular assembly and packing evaluation
├── AFM.cl                 # AFM probe-particle relaxation and image generation
├── LCAO_grid.cl           # LCAO density/orbital projection onto real-space grids
├── LCAO_STM.cl            # STM/Dyson equation kernels
```

### Composition rules (Python harness concatenates in order)
- **SPFF MD step**: `common.cl` + `Forces.cl` + `SPFF.cl` + `nonbonded.cl` (+ `gridFF.cl` + `surface.cl` if needed)
- **UFF MD step**: `common.cl` + `Forces.cl` + `UFF.cl` + `nonbonded.cl`
- **Rigid body + GridFF**: `common.cl` + `Forces.cl` + `gridFF.cl` + `rigid.cl`
- **AFM scan**: `common.cl` + `Forces.cl` + `AFM.cl` (+ `gridFF.cl` if using precomputed field)
- **Surface scan**: `common.cl` + `Forces.cl` + `surface.cl` (+ `gridFF.cl`)
- **LCAO grid projection**: `LCAO_grid.cl` (self-contained)
- **STM/Dyson**: `LCAO_grid.cl` + `LCAO_STM.cl`
- **Assembly**: `assembly.cl` (self-contained)

---

## Forces.cl
**Purpose**: Collection of pairwise interaction functions (LJ, Morse, Coulomb, H-bond) used across multiple kernels.
- `getLJQH`: LJ 12-6 + damped Coulomb (LJQH model)
- `getMorseQH`: Morse + damped Coulomb
- `MODEL_LJQH2_PAIR`: LJ 12-6 with H2 scaling + Coulomb (parameter derivatives)
- `MODEL_MorseQ_PAIR`: Morse with H scaling + Coulomb (parameter derivatives)
- `MODEL_LJr8QH_PAIR`: r^-8 variant with H2 scaling
- Energy-only variants: `ENERGY_LJQH2_PAIR`, `ENERGY_LJr8QH2_PAIR`, `ENERGY_MorseQ_PAIR`
- Decomposition macros: `MODEL_MorseQ_PAIR_DECOMP`, `MODEL_LJQH2_PAIR_DECOMP` (split into pauli/london/hbond/electro components)

**Essence**: Low-level pairwise potential functions. These are inline helpers, not standalone kernels. Used by AFM, forcefields, and surface interactions.

---

## Assembly.cl
**Purpose**: Rigid-body molecular assembly and collision detection.
- `emit_configuration_xyz`: Apply rigid transforms to base atoms and output full configuration (1 thread = 1 atom)
- `evaluate_packing_3d`: Evaluate clash penalty for a configuration of multiple molecules on a surface. Uses tiling and local memory for pairwise distance checks. Early exit on massive clash.

**Essence**: Assembly = rigid-body transforms + collision scoring. Used for building molecular assemblies on surfaces.

---

## Rigid.cl
**Purpose**: 6-DOF rigid body dynamics with quaternion integration and GridFF sampling.
- `rigid_body_dynamics_kernel`: Basic rigid body MD with external field and anchor springs. Quaternion integration (Taylor series for small rotations). Velocity damping.
- `rigid_body_gridff_kernel`: Rigid body dynamics with GridFF surface forces. B-spline interpolation via `fe3d_pbc_comb`. Outputs atom forces and body-level force/torque.

**Essence**: Rigid body MD = quaternion rotation + translation + inertia tensor + GridFF sampling. Used for AFM manipulation and rigid-body docking.

---

## Surface.cl
**Purpose**: Surface electrostatics (Ewald2D) and folded-basis potential evaluation.
- `compute_ewald_coefficients`: Compute C_G coefficients for 2D Ewald (charge density projection to reciprocal space)
- `eval_potential_vacuum`: Evaluate Ewald potential at z > max(z_i) using C_G coefficients
- `eval_potential_full`: Full Ewald evaluation at any z using per-ion weights w[g,i]
- `eval_potential_brute`: Brute-force Coulomb sum over PBC images (validation only)
- `getSurfMorse`: Brute-force Morse surface interaction (pairwise with substrate replicas)
- `getSurfFolded`: Folded atomic function surface potential (analytic summation over z-harmonics)
- `getSurfFolded_harmonics`: Folded basis harmonic expansion

**Essence**: Surface electrostatics = Ewald2D (reciprocal space) + folded analytic potentials. Used for molecule-surface interactions.

---

## GridFF.cl
**Purpose**: B-spline grid force field construction, sampling, and convolution.
- `sample3D*`: Sample 3D grid at points (various variants: basic, grid, comb2, comb)
- `sample1D_pbc`: 1D sampling with periodic boundary
- `BsplineConv3D*`: B-spline convolution kernels (textured and non-textured)
- `make_MorseFF*`: Build Morse force field grid from atoms
- `make_Coulomb_points`: Build Coulomb potential grid from point charges
- `project_atom_on_grid_cubic_pbc`: Project atoms to grid with cubic B-spline
- `project_atoms_on_grid_quintic_pbc`: Project atoms to grid with quintic B-spline (complex output for Poisson)
- `poissonW*`: Poisson solver in Fourier space (rho_k -> V_k)
- `laplace_real_pbc`: Laplacian operator on real-space grid
- `slabPotential*`: Slab potential corrections
- `make_GridFF`: Build GridFF from atoms (B-spline interpolation)
- `getNonBond*`: Non-bonded force evaluation with GridFF (multiple variants: basic, ex2, GridFF_Bspline)
- `getSurfaceIsoSurfMorse`: Surface isosurface from Morse potential
- `getSurfaceIsoGridFF`: Surface isosurface from GridFF

**Essence**: GridFF = atom-to-grid projection + B-spline interpolation + Poisson solver + sampling. Used for precomputed surface potentials.

---

## relax.cl
**Purpose**: AFM probe-particle relaxation and image generation (AFMulator).
- `getFEinPoints*`: Sample force/energy field at points (various: basic, shifted)
- `getFEinStrokes*`: Sample along scan strokes (AFM scan lines)
- `getZisoTilted*`: Find isosurface height from force/energy field
- `relaxPoints*`: Relax probe particle positions in force field
- `relaxStrokes*`: Relax probe along scan strokes
- `relaxStrokesTilted*`: Relax with tilted tip geometry
- `relaxStrokesTilted_convZ`: Relax with Z-convolution
- `convolveZ`: Z-direction convolution
- `izoZ`: Isosurface extraction
- `evalLJC_QZs*`: Evaluate LJ + Coulomb at Z-slices (to image)
- `evalMorseC_QZs_toImg`: Morse + Coulomb to image
- `evalDispersion_toImg`: C6 dispersion to image
- `gradient_central_diff`: Compute gradient from scalar field
- `getNonBond*`: Non-bonded forces (LJ/Morse/Coulomb) for AFM tip
- `getSPFFsp3`: SPFFsp3 force evaluation
- `getSPFFf4`: SPFFsp3 with 4-neighbor topology
- `updateAtomsSPFFf4`: Update atom positions/velocities (MD step)
- `cleanForceSPFFf4`: Zero force buffers
- `gatherForceAndMove`: Gather forces and move atoms
- `updatePiPos0`: Update pi-orbital positions
- `evalPiPi`: Pi-pi interaction evaluation

**Essence**: AFMulator = probe-particle relaxation in force field + image generation. Supports LJ/Morse/Coulomb/SPFF.

---

## relax_multi.cl
**Purpose**: SPFFsp3 multi-system molecular dynamics (main MD kernel).
- `getSPFFf4`: Bonding interactions (bonds, angles, torsions, pi-pi, H-bond) with 4-neighbor topology
- `getSPFFf4_bak`: Backup variant
- `updateGroups`: Update group positions (rigid groups)
- `groupForce`: Group force evaluation
- `updateAtomsSPFFf4`: Assemble recoil forces and update positions/velocities (MD integration)
- `getNonBond*`: Non-bonded LJ/Morse/Coulomb with PBC (multiple variants: basic, ex2, GridFF_Bspline)
- `printOnGPU`: Debug print
- `getSurfFlat`: Flat surface interaction
- `cleanForceSPFFf4`: Zero force buffers
- `getShortRangeBuckets*`: Spatial bucketing for neighbor search
- `sortAtomsToBucketOverlaps`: Sort atoms into overlapping buckets
- `getNonBond_GridFF*`: Non-bonded with GridFF forces
- `sampleGridFF_Bspline_points`: Sample GridFF at points
- `getSurfaceIsoSurfMorse`: Surface isosurface from Morse
- `getSurfaceIsoGridFF`: Surface isosurface from GridFF

**Essence**: Full SPFFsp3 MD engine = bonding + non-bonding + PBC + GridFF + spatial bucketing. This is the workhorse MD kernel.

---

## UFF.cl
**Purpose**: UFF force field evaluation (bonds, angles, torsions, inversions).
- `clear_fapos_UFF`: Zero atom force buffer
- `clear_fint_UFF`: Zero interaction force buffer
- `evalBondsAndHNeigh_UFF`: Evaluate bonds and compute H-neighbor vectors
- `evalAngles_UFF`: Evaluate angle interactions
- `evalDihedrals_UFF`: Evaluate dihedral interactions
- `evalInversions_UFF`: Evaluate inversion interactions
- `assembleForces_UFF`: Assemble force pieces onto atoms
- `updateAtomsSPFFf4`: Update atoms (reused from relax_multi)
- `getNonBond*`: Non-bonded LJ/Morse/Coulomb (same as relax_multi variants)
- `getSurfMorse`: Surface Morse interaction

**Essence**: UFF = bonding (bonds/angles/torsions/inversions) + non-bonding. Separated into per-interaction kernels.

---

## Grid_dftb.cl
**Purpose**: DFTB density/orbital projection to real-space grid for FDBM/STM.
- `project_density_sparse`: Project sparse density matrix to grid (task-based)
- `count_atoms_per_block`: Count atoms per grid block for task generation
- `fill_task_atoms`: Fill task atom lists
- `compact_tasks`: Compact task list
- `project_density_sparse_tiled`: Tiled variant for better coalescing
- `project_orbital`: Project orbital to grid (task-based)
- `project_orbital_points_exp`: Project orbital to arbitrary points (exponential basis)
- `mo_overlap_points_exp_sk`: Compute MO overlap at points (STM tip response)
- `mo_overlap_points_exp_sk_2mol`: Two-molecule overlap
- `project_orbital_points`: Project orbital to points (spline basis)
- `project_orbital_dense_points`: Dense orbital projection to points
- `project_density_dense_points`: Dense density projection to points
- `project_orbital_dense`: Dense orbital projection to grid
- `project_density_dense`: Dense density projection to grid
- `response_amplitude_exp`: STM response amplitude calculation
- `solve_stm_dyson_wg`: Solve Dyson equation for STM (workgroup-local)
- `stm_gf_dyson_2mol_mo_scan`: STM Green's function Dyson scan for two molecules

**Essence**: DFTB grid projection = sparse/dense density/orbital projection + STM overlap/response. Used for FDBM density and STM simulation.

---

## relax_multi_mini.cl
**Purpose**: Stripped-down variant of relax_multi.cl (historical, needs refactoring).
- Contains similar kernels to relax_multi but with reduced functionality.
- Currently not loaded by any Python module.

**Essence**: Historical artifact. Should be merged into relax_multi.cl or removed.

---

## Summary of Redundancy

### Non-bonded kernels (LJ/Morse/Coulomb with PBC)
- `getNonBond` (neighbor-list based, `neighs`+`neighCell` int4 arrays) appears in `relax.cl`, `relax_multi.cl`, `UFF.cl` — nearly identical logic
- `getNonBond_ex2` (exclusion-list based, packed `excl` array with `EXCL_MAX`) appears in `relax_multi.cl` and `UFF.cl` — this is the 2nd-neighbor exclusion variant
- `getNonBond_GridFF_Bspline` and `getNonBond_GridFF_Bspline_ex2` appear in both `relax_multi.cl` and `UFF.cl`
- `Forces.cl` has inline functions (`getLJQH`, `getMorseQH`) that duplicate the pairwise logic embedded in all the above
- `relax_multi_mini.cl` has yet another copy of `getNonBond` + `getNonBond_ex2`

**Key difference**: `getNonBond` uses `neighs`/`neighCell` (int4 per atom, 4 neighbors max) — this is 1st-neighbor exclusion only. `getNonBond_ex2` uses a packed sorted exclusion list (`excl` with `EXCL_MAX`) — this is 2nd-neighbor exclusion. **We should keep only `ex2` variant.**

### GridFF sampling
- `sample3D*` family in `GridFF.cl` — standalone grid sampling
- `sampleGridFF_Bspline_points` in `relax_multi.cl` — same B-spline interpolation but inline
- `sampleGrid` / `sampleGrid_tex` in `relax_multi_mini.cl` — another copy
- `fe3d_pbc_comb` in `Rigid.cl` — B-spline interpolation with PBC, different interface but same math
- `getNonBond_GridFF_Bspline*` in `relax_multi.cl` and `UFF.cl` — combines non-bonded with GridFF sampling inline

### Surface interactions
- `getSurfMorse` (brute-force Morse with PBC replicas) in both `UFF.cl` and `relax_multi.cl` — identical
- `getSurfaceIsoSurfMorse` and `getSurfaceIsoGridFF` in both `relax.cl` and `relax_multi.cl` — identical
- `getSurfFolded*` in `Surface.cl` — folded analytic variant, unique
- `getSurfFlat` in `relax_multi.cl` and `relax_multi_mini.cl` — duplicated

### MD integration
- `updateAtomsSPFFf4` in `UFF.cl`: **simplified** — no `fneigh`/`bkNeighs` (no recoil assembly), no `bboxes`, no `sysneighs`/`sysbonds`. Just basic force -> velocity -> position update with constraints and damping.
- `updateAtomsSPFFf4` in `relax_multi.cl`: **full** — has recoil force gathering from `fneigh`/`bkNeighs` (for pi-orbital DOFs), bounding box constraints, inter-system bonds (`sysneighs`/`sysbonds`), pi-orbital normalization. This is the SPFF-specific integrator.
- `updateAtomsSPFFf4_rot` in `relax_multi_mini.cl`: variant with angular momentum handling for pi-orbitals
- `cleanForceSPFFf4` in both `UFF.cl` and `relax_multi.cl` — identical
- `runMD` in `relax_multi_mini.cl` — fused force+integrate kernel

**Key difference**: UFF does NOT have pi-orbital DOFs, so its `updateAtomsSPFFf4` lacks recoil assembly. SPFF has pi-orbital DOFs as additional `nnode` vectors, requiring recoil force gathering from back-neighbors.

### relax*.cl overlap
- `relax.cl` = AFM probe-particle relaxation + image generation + some SPFF/nonbonded kernels (mixed concerns)
- `relax_multi.cl` = full SPFFsp3 MD engine (bonding + non-bonded + GridFF + surface + integrator) — the workhorse
- `relax_multi_mini.cl` = stripped SPFF variant with `getSPFFf4_rot`, `runMD`, `scanNonBond*` — historical, needs merging

All three share: `cl_Mat3` typedef, `float4Zero` macro, `getNonBond*`, `cleanForceSPFFf4`, `getSPFFf4`, surface kernels. Massive code duplication.

---

## Proposed Reorganization

### Principle: orthogonal composability via text concatenation
OpenCL has no `#include` (or it's unreliable). We use text concatenation + macro replacement to assemble optimized kernels. So we split into **snippet files** (not compilable alone) that get concatenated by the Python harness.

### Proposed file structure

```
kernels/
├── common.cl           # Shared types, macros, constants
│                       #   cl_Mat3, float4Zero, COULOMB_CONST, samplers
│                       #   quat_mult, quat_to_a/b/c, make_qrot, etc.
│
├── Forces.cl  # Inline pairwise functions (not __kernel)
│                       #   getLJQH, getMorseQH, energy_LJ, energy_Morse,
│                       #   energy_Coulomb, energy_HBond
│                       #   MODEL_LJQH2_PAIR, MODEL_MorseQ_PAIR macros
│                       #   (from Forces.cl helpers)
│
├── nonbonded.cl        # __kernel getNonBond_ex2 (2nd-neighbor exclusion only)
│                       # __kernel getNonBond_GridFF_Bspline_ex2
│                       # Uses Forces.cl inline functions
│                       # Shared by UFF, SPFF, and rigid-body
│
├── gridff.cl           # Grid construction + sampling
│                       #   make_MorseFF, make_Coulomb_points,
│                       #   project_atom_on_grid_*, poissonW,
│                       #   BsplineConv3D*, sample3D*, laplace, slabPotential
│                       #   (from GridFF.cl, cleaned up)
│
├── surface.cl          # Surface-specific interactions
│                       #   getSurfMorse, getSurfFolded*, getSurfFlat,
│                       #   getSurfaceIsoSurfMorse, getSurfaceIsoGridFF
│                       #   Ewald2D: compute_ewald_coefficients,
│                       #   eval_potential_vacuum/full/brute
│                       #   (from Surface.cl + duplicates in relax*.cl)
│
├── spff.cl     # SPFFsp3-specific bonding kernels
│                       #   getSPFFf4 (bonds, angles, torsions, pi-pi, H-bond)
│                       #   evalPiPi, updatePiPos0
│                       #   (from relax_multi.cl)
│                       #   updateAtomsSPFFf4 (with recoil, pi-orbital norm,
│                       #   bboxes, sysneighs/sysbonds)
│                       #   cleanForceSPFFf4, gatherForceAndMove
│                       #   groupForce, updateGroups
│
├── uff.cl      # UFF-specific bonding kernels
│                       #   evalBondsAndHNeigh_UFF, evalAngles_UFF,
│                       #   evalDihedrals_UFF, evalInversions_UFF,
│                       #   assembleForces_UFF, clear_fapos/fint_UFF
│                       #   updateAtomsSPFFf4 (simplified, no recoil,
│                       #   no pi-orbitals, no bboxes/sysneighs)
│
├── rigid.cl            # 6-DOF rigid body dynamics
│                       #   rigid_body_dynamics_kernel,
│                       #   rigid_body_gridff_kernel
│                       #   (uses pair_potentials.cl for forces,
│                       #    gridff.cl for B-spline sampling)
│
├── assembly.cl         # Rigid-body assembly + collision
│                       #   emit_configuration_xyz, evaluate_packing_3d
│
├── afm.cl              # AFM probe-particle relaxation + imaging
│                       #   getFEinPoints*, getFEinStrokes*,
│                       #   relaxPoints, relaxStrokes*,
│                       #   getZisoTilted*, convolveZ, izoZ,
│                       #   evalLJC_QZs*, evalMorseC_QZs_toImg,
│                       #   evalDispersion_toImg, gradient_central_diff
│                       #   (from relax.cl, AFM-only parts)
│
├── lcao_grid.cl        # LCAO density/orbital projection
│                       #   project_density_sparse*, project_orbital*,
│                       #   project_density_dense*, count/fill/compact tasks
│                       #   (from Grid_dftb.cl)
│
├── lcao_stm.cl         # STM/Dyson equation kernels
│                       #   mo_overlap_points_exp_sk*,
│                       #   response_amplitude_exp,
│                       #   solve_stm_dyson_wg, stm_gf_dyson_2mol_mo_scan
│                       #   (from Grid_dftb.cl)
```

### Composition examples (Python harness concatenates)

**UFF MD step**: `common.cl` + `Forces.cl` + `uff.cl` + `nonbonded.cl`

**SPFF MD step**: `common.cl` + `Forces.cl` + `spff.cl` + `nonbonded.cl` (+ optionally `gridff.cl` + `surface.cl` if GridFF/surface forces needed)

**Rigid body + GridFF**: `common.cl` + `Forces.cl` + `gridff.cl` + `rigid.cl`

**AFM scan**: `common.cl` + `Forces.cl` + `afm.cl` (+ `gridff.cl` if using precomputed field)

**Surface scan**: `common.cl` + `Forces.cl` + `surface.cl` (+ `gridff.cl`)

### What gets eliminated
- `relax.cl` → split into `afm.cl` + shared parts go to `nonbonded.cl`/`surface.cl`
- `relax_multi.cl` → split into `spff.cl` + `nonbonded.cl` + `surface.cl`
- `relax_multi_mini.cl` → merged into the above; `getSPFFf4_rot` and `runMD` go into `spff.cl`, `scanNonBond*` into respective forcefield files
- `UFF.cl` → becomes `uff.cl` (bonding + integrator); non-bonded/surface parts go to shared files
- Only `ex2` (2nd-neighbor exclusion) variant of `getNonBond` survives; old `neighs`/`neighCell` variant dropped

---

## Python Migration Plan

### Problem
All Python modules that use OpenCL still reference old kernel file paths (e.g. `../../cpp/common_resources/cl/relax_multi.cl`) that no longer exist. They need to be updated to load from the new `kernels/` directory using the concatenation model.

### Key infrastructure: `OpenCLBase.load_program()`
Located at `spamm/utils/OpenCLBase.py:204`. Currently loads a **single** `.cl` file. Needs a new method (or modification) to **concatenate multiple** `.cl` files in order and compile the result.

**Proposed new method**: `load_program_multi(kernel_paths, build_options=None, bMakeHeaders=True)`
- Reads each file in the list, concatenates source strings
- Compiles the combined source
- Extracts kernel headers from combined source

### Python modules to modify (8 modules)

#### 1. `spamm/forcefields/MolecularDynamics.py` (line 93)
- **Current**: loads `../../cpp/common_resources/cl/relax_multi.cl` (single file)
- **New**: concatenate `common.cl` + `Forces.cl` + `SPFF.cl` + `nonbonded.cl`
- **Also**: conditionally append `gridFF.cl` + `surface.cl` when GridFF/surface forces are enabled
- **Kernel names**: already renamed (SPFF) via sed — `getSPFFf4`, `updateAtomsSPFFf4`, `cleanForceSPFFf4` etc.
- **Key kernels used**: `getSPFFf4`, `updateAtomsSPFFf4`, `cleanForceSPFFf4`, `getNonBond_ex2`, `getNonBond_GridFF_Bspline_ex2`, `getNonBond_GridFF_Bspline_tex`, `getSurfMorse`, `getSurfFlat`, `getSurfFolded*`, `sampleGridFF_Bspline_points`, `getSurfaceIsoSurfMorse`, `getSurfaceIsoGridFF`, `getShortRangeBuckets*`, `sortAtomsToBucketOverlaps`
- **Effort**: HIGH — this is the workhorse module, ~2000 lines, many kernel calls

#### 2. `spamm/forcefields/UFF.py` (line 42)
- **Current**: loads `../../cpp/common_resources/cl/UFF.cl` (single file)
- **New**: concatenate `common.cl` + `Forces.cl` + `UFF.cl` + `nonbonded.cl`
- **Key kernels used**: `clear_fapos_UFF`, `clear_fint_UFF`, `evalBondsAndHNeigh_UFF`, `evalAngles_UFF`, `evalDihedrals_UFF`, `evalInversions_UFF`, `assembleForces_UFF`, `updateAtomsSPFFf4`, `getNonBond_ex2`, `getSurfMorse`
- **Effort**: MEDIUM

#### 3. `spamm/forcefields/RigidBodyDynamics.py` (line 220)
- **Current**: loads `../../cpp/common_resources/cl/Rigid.cl` (single file)
- **New**: concatenate `common.cl` + `Forces.cl` + `gridFF.cl` + `rigid.cl`
- **Key kernels used**: `rigid_body_dynamics_kernel`, `rigid_body_gridff_kernel`
- **Note**: `bMakeHeaders=False` currently — may need to enable if kernel args are auto-generated
- **Effort**: LOW

#### 4. `spamm/forcefields/Assembly.py` (line 28)
- **Current**: loads `cl/Assembly.cl` (relative to module dir)
- **New**: load `kernels/assembly.cl` (self-contained, no concatenation needed)
- **Key kernels used**: `emit_configuration_xyz`, `evaluate_packing_3d`
- **Effort**: LOW — just update path

#### 5. `spamm/spm/AFM.py` (line 100)
- **Current**: loads `relax.cl` from `cl_src_dir` (default: `../../cpp/common_resources/cl/`)
- **New**: concatenate `common.cl` + `Forces.cl` + `AFM.cl`
- **Key kernels used**: `getFEinPoints`, `getFEinStrokes`, `relaxPoints`, `relaxStrokes`, `relaxStrokesTilted`, `getZisoTilted`, `convolveZ`, `izoZ`, `evalLJC_QZs`, `evalMorseC_QZs_toImg`, `evalDispersion_toImg`, `gradient_central_diff`
- **Note**: AFM.cl now includes its own samplers (`sampler_1`, `sampler_2`, `sampler_nearest`)
- **Effort**: MEDIUM

#### 6. `spamm/spm/AFM_utils.py` (lines 832, 1116)
- **Current**: reads `relax.cl` source to get `interpFE` function, then appends custom kernels
- **New**: read `common.cl` + `Forces.cl` + `AFM.cl` to get `interpFE` and samplers, then append custom kernels
- **Effort**: LOW — just update path and concatenate

#### 7. `spamm/surfaces/SurfaceEwald.py` (line 83) and `spamm/surfaces/GridFF.py` (line 82)
- **Current**: load `Surface.cl` / `GridFF.cl` from `../../cpp/common_resources/cl/`
- **New**: `SurfaceEwald.py` → concatenate `common.cl` + `Forces.cl` + `surface.cl`; `GridFF.py` → concatenate `common.cl` + `Forces.cl` + `gridFF.cl`
- **Effort**: MEDIUM

#### 8. `spamm/quantum/dftb/Grid_dftb.py` (line 185)
- **Current**: loads `cl/Grid_dftb.cl` (self-generated in `cl/` subdir)
- **New**: concatenate `kernels/LCAO_grid.cl` + `kernels/LCAO_STM.cl`
- **Key kernels used**: `project_density_sparse*`, `project_orbital*`, `response_amplitude_exp`, `solve_stm_dyson_wg`, `stm_gf_dyson_2mol_mo_scan`
- **Effort**: MEDIUM

### Implementation order (by risk/effort)

1. **LOW effort first** (path-only changes, self-contained files):
   - `Assembly.py` → `assembly.cl`

2. **LOW-MEDIUM** (concatenation needed, few kernels):
   - `RigidBodyDynamics.py` → `common.cl` + `Forces.cl` + `gridFF.cl` + `rigid.cl`
   - `AFM_utils.py` → update source reading to concatenated `AFM.cl`

3. **MEDIUM** (multi-file concatenation, many kernels):
   - `AFM.py` → `common.cl` + `Forces.cl` + `AFM.cl`
   - `SurfaceEwald.py` → `common.cl` + `Forces.cl` + `surface.cl`
   - `GridFF.py` → `common.cl` + `Forces.cl` + `gridFF.cl`
   - `Grid_dftb.py` → `LCAO_grid.cl` + `LCAO_STM.cl`
   - `UFF.py` → `common.cl` + `Forces.cl` + `UFF.cl` + `nonbonded.cl`

4. **HIGH** (workhorse, test thoroughly):
   - `MolecularDynamics.py` → `common.cl` + `Forces.cl` + `SPFF.cl` + `nonbonded.cl` (+ optional `gridFF.cl` + `surface.cl`)

### Prerequisite: `OpenCLBase.load_program_multi()`

Before modifying any Python module, add a method to `OpenCLBase` that:
1. Takes a list of kernel file paths
2. Reads and concatenates them (in order)
3. Compiles the combined source
4. Extracts kernel headers from combined source
5. Optionally writes combined source to a temp file for debugging

```python
def load_program_multi(self, kernel_paths, build_options=None, bMakeHeaders=True, bPrint=False):
    """Load and compile multiple .cl files concatenated together."""
    source_parts = []
    for path in kernel_paths:
        if not os.path.exists(path):
            print(f"load_program_multi() ERROR: {path} not found")
            return False
        with open(path, 'r') as f:
            source_parts.append(f.read())
    kernel_source = '\n'.join(source_parts)
    # ... compile, extract headers, etc.
```

### Kernel directory resolution
All Python modules should resolve the kernel directory as:
```python
KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../kernels')
```
(adjusting `../../` depth based on module location in the package tree)

### Testing strategy
- After each module migration, run its corresponding test:
  - `MolecularDynamics.py` → `tests/` SPFF/UFF MD tests
  - `UFF.py` → `tests/` UFF tests
  - `AFM.py` → `tests/` AFM tests
  - `RigidBodyDynamics.py` → `tests/` rigid body tests
  - `Grid_dftb.py` → `tests/` DFTB grid tests
- Verify kernel compilation succeeds (no missing symbols, no duplicate definitions)
- Verify numerical results match pre-reorganization (use `kernels_bak/` as reference)
