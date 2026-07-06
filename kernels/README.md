# kernels/

OpenCL source for SPAMMM GPU compute. Python harnesses concatenate `.cl` snippets in order (`OpenCLBase.load_program_multi`) — there is no reliable `#include`.

## File index

| File | Role | Python consumer |
|------|------|-----------------|
| `common.cl` | Shared types, constants, math helpers | (always first) |
| `Forces.cl` | Inline pairwise potentials (LJ, Morse, Coulomb) | all force modules |
| `SPFF.cl` | SPFFsp3 bonding + MD integrator (π-orbitals) | `forcefields/SPFF_cl.py` |
| `UFF.cl` | UFF bonding + simplified integrator | `forcefields/UFF_cl.py` |
| `nonbonded.cl` | `getNonBond_ex2` — LJ/Coulomb with 2nd-neighbor exclusion | SPFF, UFF |
| `nonbonded_grid.cl` | GridFF-augmented nonbonded + spatial bucketing | SPFF (on demand) |
| `gridFF.cl` | 3D B-spline grid build, Poisson, sampling | SPFF, GridFF, rigid |
| `surface.cl` | Ewald2D, folded basis, brute Morse, isosurface | SPFF, SurfaceEwald |
| `contact_surface.cl` | Quasi-2D contact surface (separable + PIC) | `surfaces/ContactSurface.py` |
| `rigid.cl` | 6-DOF rigid body (GridFF + folded) | `forcefields/RigidBodyDynamics.py` |
| `assembly.cl` | Multi-molecule rigid transforms + clash | `forcefields/Assembly.py` |
| `AFM.cl` | Probe relaxation + AFM image generation | `SPM/AFM.py` |
| `LCAO_grid.cl` | LCAO density/orbital grid projection | `quantum/DFTB/Grid_dftb.py` |
| `LCAO_STM.cl` | STM / Dyson equation | `quantum/DFTB/Grid_dftb.py` |
| `lingebra.cl` | Batched small-matrix Jacobi eigh | `utils/Lingebra_ocl.py` |

## Composition rules

Concatenation order matters. Typical stacks:

| Use case | Files (in order) |
|----------|------------------|
| SPFF MD | `common` + `Forces` + `SPFF` + `gridFF` + `surface` + `nonbonded` |
| UFF MD | `common` + `Forces` + `UFF` + `nonbonded` |
| SPFF + GridFF NB | add `nonbonded_grid.cl` (separate program) |
| Rigid body + GridFF | `common` + `Forces` + `gridFF` + `rigid` |
| Rigid body + folded | `common` + `Forces` + `surface` + `rigid` |
| AFM scan | `common` + `Forces` + `AFM` |
| Surface Ewald | `common` + `Forces` + `surface` |
| GridFF only | `common` + `Forces` + `gridFF` |
| Contact surface | `common` + `Forces` + `contact_surface` |
| LCAO / STM | `LCAO_grid` + `LCAO_STM` |
| Assembly | `assembly` (standalone) |
| Lingebra | `lingebra` (standalone) |

---

## common.cl

Shared foundation — concatenated **first** for all multi-file builds.

- Types: `cl_Mat3`
- Constants: `COULOMB_CONST`, `R2SAFE`, `EXCL_MAX`
- Math: `modulo`, `udiv_cmplx`, `rotMat`, `mixREQ_arithmetic`, `clampForce`
- No `__kernel` functions

---

## Forces.cl

Inline pairwise functions (not standalone kernels). Returns `float4 (Fx, Fy, Fz, E)`.

- `getLJQH`, `getMorseQH`, `getMorsePLQH`, `getCoulomb`
- Energy/decomposition macros: `MODEL_LJQH2_PAIR`, `MODEL_MorseQ_PAIR`, `ENERGY_*`, `*_DECOMP`
- Used by nonbonded, surface, AFM, GridFF builders

---

## SPFF.cl

SPFFsp3 force field: bonds, angles, torsions, π–π alignment, H-bond; MD integrator with π-orbital recoil.

**MD step:** `getSPFFf4` → `getNonBond_ex2` → `cleanForceSPFFf4` → `updateAtomsSPFFf4`

| Kernel | Role |
|--------|------|
| `getSPFFf4` | Bonding forces (1 thread = 1 node atom) |
| `updateGroups`, `groupForce` | Rigid group kinematics |
| `updateAtomsSPFFf4` | Integrator: recoil gather, constraints, π normalization |
| `cleanForceSPFFf4` | Zero force buffers |

---

## UFF.cl

Universal Force Field without π DOFs. Per-interaction eval + scatter pattern.

| Kernel | Role |
|--------|------|
| `evalBondsAndHNeigh_UFF` | Harmonic bonds + H-neighbor vectors |
| `evalAngles_UFF`, `evalDihedrals_UFF`, `evalInversions_UFF` | Valence terms |
| `assembleForces_UFF` | Scatter `fint` → `fapos` |
| `clear_fapos_UFF`, `clear_fint_UFF` | Buffer reset |
| `updateAtomsSPFFf4` | Simplified integrator (no recoil) |

---

## nonbonded.cl

Molecule–molecule nonbonded with **2nd-neighbor exclusion** (`excl`, `EXCL_MAX`).

| Kernel | Role |
|--------|------|
| `getNonBond_ex2` | LJ/Morse/Coulomb + PBC; local-memory tiling |

GridFF-augmented variants live in `nonbonded_grid.cl`.

---

## nonbonded_grid.cl

| Kernel | Role |
|--------|------|
| `getNonBond_GridFF_Bspline_ex2` | Nonbonded + substrate GridFF (buffer) |
| `getNonBond_GridFF_Bspline_tex` | Same with texture sampling |
| `getShortRangeBuckets*` | Spatial bucketing |
| `sortAtomsToBucketOverlaps` | Atom sort into buckets |

Requires: `common` + `Forces` + `gridFF` + `surface` (for `getR4repulsion`, `fe3d_pbc_comb`).

---

## gridFF.cl

3D B-spline force-field grids: build, convolve, Poisson, sample.

Key groups:
- **Sampling:** `sample3D*`, `sample1D_pbc`, `sampleGridFF_Bspline_points`
- **Convolution:** `BsplineConv3D*`, `Convolution3D_General`
- **Build:** `make_MorseFF*`, `make_Coulomb_points`, `make_GridFF`
- **Projection:** `project_atom_on_grid_cubic_pbc`, `project_atoms_on_grid_quintic_pbc`
- **Poisson:** `poissonW*`, `laplace_real_pbc`, `slabPotential*`
- **Utilities:** `addMul`, `dot_wg`, `setLinear`, `move`, `setMul`

Inline: `make_inds_pbc`, `fe3d_pbc_comb` (B-spline PBC interpolation).

---

## surface.cl

Molecule–substrate interactions: brute, folded basis, Ewald2D, isosurface.

| Kernel | Role |
|--------|------|
| `getSurfMorse`, `getSurfFlat` | Brute pairwise Morse |
| `getSurfFolded*`, `getSurfFolded_harmonics` | Folded analytic basis |
| `compute_ewald_coefficients`, `eval_potential_*` | 2D Ewald electrostatics |
| `eval_potential_brute` | Validation |
| `getSurfaceIsoSurfMorse`, `getSurfaceIsoGridFF` | Isosurface forces |
| `addDipoleField` | Macro dipole sheet |

Helpers: `macro_phi_rect_*`, `folded_eval_basis/grad`, `getR4repulsion`.

---

## contact_surface.cl

Quasi-2D contact surface for static AFM. See `doc/Topics/AFM/ContactSurface_Static.md`.

| Kernel | Role |
|--------|------|
| `cs_brute_plqh_points` | Brute Morse+PLQH reference |
| `evalSeparableBsplinePoly` | Separable B-spline × poly eval |
| `cs_sep_Av`, `cs_sep_Atv*` | Separable fit operators |
| `cs_pic_*`, `evalRadialPIC` | PIC fit/eval |
| `dot_wg`, `addMul`, `cs_zero`, `cs_copy` | CG helpers |

**Driver:** `python tests/testplot_contact_surface.py` → `debug/testplot_contact_surface/`

---

## rigid.cl

6-DOF rigid-body MD with quaternion integration.

| Kernel | Role |
|--------|------|
| `rigid_body_dynamics_kernel` | Pairwise forces + quaternion step |
| `rigid_body_gridff_kernel` | GridFF B-spline substrate forces |
| `rigid_body_folded_kernel` | Folded-basis analytic substrate forces |

Helpers: `quat_mult`, `sinc_div_r2_taylor`, `quat_factors_taylor`.

---

## assembly.cl

Rigid-body multi-molecule packing. **Self-contained** (no `common.cl`).

| Kernel | Role |
|--------|------|
| `emit_configuration_xyz` | Apply transforms → assembled coords |
| `evaluate_packing_3d` | Steric clash scoring |

**Driver:** `python tests/testplot_assembly.py` → `debug/testplot_assembly/`

---

## AFM.cl

AFM probe-particle relaxation and image generation.

Flow: build Z-slices → sample field → `relaxPoints`/`relaxStrokes*` → isosurface → `convolveZ` → `izoZ`.

Key kernels: `getFEinPoints*`, `relaxPoints`, `relaxStrokesTilted*`, `getZisoTilted*`, `evalLJC_QZs_toImg`, `evalMorseC_QZs_toImg`, `evalDispersion_toImg`, `gradient_central_diff`.

Helpers: `interpFE`, `update_FIRE`, `tipForce`, trilinear `read_imagef` sampling.

---

## LCAO_grid.cl

Project LCAO density and orbitals onto 3D grids (DFTB, Fireball, etc.). **Self-contained** types (`GridSpec`, `AtomData`, `TaskData`).

| Kernel | Role |
|--------|------|
| `project_density_sparse*` | Sparse density → grid |
| `project_orbital*` | Orbital projection (sparse/dense, grid/points) |
| `mo_overlap_points_exp_sk*` | MO overlap at points (STM input) |
| `count_atoms_per_block`, `fill_task_atoms`, `compact_tasks` | Task scheduling |

---

## LCAO_STM.cl

STM tunneling via Dyson equation. Requires `LCAO_grid.cl` types.

| Kernel | Role |
|--------|------|
| `response_amplitude_exp` | STM response amplitude |
| `solve_stm_dyson_wg` | Workgroup Dyson solve |
| `stm_gf_dyson_2mol_mo_scan` | Two-molecule scan |

---

## lingebra.cl

Batched symmetric eigendecomposition (parallel Jacobi). **Self-contained.**

| Kernel | Role |
|--------|------|
| `local_jacobi_blocks_parallel` | (batch × m × m) → eigenvalues + eigenvectors |

**Tests:** `pytest tests/test_lingebra.py`

---

## Adding a new kernel file

1. Add a file-level header: purpose, kernel list, composition requirements, Python consumer.
2. Register in this README (table + section).
3. Wire via `load_program_multi` in the Python module.
4. Add pytest or `testplot_*` diagnostic.

## Historical note

Pre-2025 monolithic files (`relax_multi.cl`, `relax.cl`, `Grid_dftb.cl`) were split into the modular layout above. Old migration notes lived in `INVENTORY.md` (renamed to this file).
