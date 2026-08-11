# kernels/

OpenCL source for SPAMMM GPU compute. Python harnesses concatenate `.cl` snippets in order (`OpenCLBase.load_program_multi`) — there is no reliable `#include`.

## File index

| File | Role | Python consumer |
|------|------|-----------------|
| `common.cl` | Shared types, constants, math helpers | (always first) |
| `Forces.cl` | Inline pairwise potentials (LJ, Morse, Coulomb) | all force modules |
| `SPFF.cl` | SPFFsp3 bonding + MD integrator (π-orbitals) | `forcefields/SPFF_cl.py` |
| `UFF.cl` | UFF bonding + fused multi-step (+ optional FAF) | `forcefields/UFF_cl.py` |
| `LFF.cl` | Projective Jacobi on K₁₂/K₁₃/K₁₄ springs + FAF outer | `forcefields/LFFSolver.py` |
| `nonbonded.cl` | `getNonBond_ex2` — LJ/Coulomb with 2nd-neighbor exclusion | SPFF, UFF |
| `nonbonded_grid.cl` | GridFF-augmented nonbonded + spatial bucketing | SPFF (on demand) |
| `gridFF.cl` | 3D B-spline grid build, Poisson, sampling | SPFF, GridFF, rigid |
| `surface.cl` | Ewald2D, folded basis, brute Morse, isosurface | SPFF, SurfaceEwald |
| `contact_surface.cl` | Quasi-2D contact + **contact_pme** (local/bucket FIRE, `fillContactPMEMeshVL`) | `SPM/AFM.py`, `surfaces/ContactSurface.py` |
| `rigid.cl` | 6-DOF rigid body, PairFF+FAF replica energy, and ping-pong concurrent multi-molecule MD | `forcefields/RigidBodyDynamics.py` |
| `assembly.cl` | Multi-molecule rigid transforms + clash | `forcefields/Assembly.py` |
| `AFM.cl` | Probe relaxation + AFM image generation | `SPM/AFM.py` |
| `grids.cl` | Density project/downsample (dipole-preserving), Gaussian NA, axpy | `utils/GridsOCL.py` |
| `LCAO_grid.cl` | LCAO density/orbital grid projection + legacy `mo_overlap_points_exp_sk` | `quantum/DFTB/Grid_dftb.py` |
| `LCAO_STM_FGR.cl` | First-order FGR STM \(M=c^\dagger(H-ES)c\) with tabulated long-tail SK τ | `quantum/DFTB/Grid_dftb.py` |
| `LCAO_STM.cl` | STM / Dyson equation (GF-dressed; not FGR product path) | `quantum/DFTB/Grid_dftb.py` |
| `lingebra.cl` | Batched small-matrix Jacobi eigh | `utils/Lingebra_ocl.py` |

## Composition rules

Concatenation order matters. Typical stacks:

| Use case | Files (in order) |
|----------|------------------|
| SPFF MD | `common` + `Forces` + `SPFF` + `gridFF` + `surface` + `nonbonded` |
| UFF MD | `common` + `Forces` + `UFF` + `nonbonded` |
| LFF projective | `LFF.cl` alone (`LFFSolver`; FAF helpers inlined) |
| SPFF + GridFF NB | add `nonbonded_grid.cl` (separate program) |
| Rigid body + GridFF | `common` + `Forces` + `gridFF` + `rigid` |
| Rigid body + folded | `common` + `Forces` + `surface` + `rigid` |
| AFM scan | `common` + `Forces` + `AFM` |
| Surface Ewald | `common` + `Forces` + `surface` |
| GridFF only | `common` + `Forces` + `gridFF` |
| Contact surface | `common` + `Forces` + `contact_surface` |
| Grids (density project) | `grids.cl` (standalone) |
| LCAO / STM | `LCAO_grid` + `LCAO_STM` + `LCAO_STM_FGR` |
| Assembly | `assembly` (standalone) |
| Lingebra | `lingebra` (standalone) |

---

## common.cl

Shared foundation — concatenated **first** for all multi-file builds. No `__kernel` functions; only types, constants, and inline helpers.

- **Types:** `cl_Mat3` (3×3 row-major matrix)
- **Constants:** `COULOMB_CONST` (1/4πϵ₀ in MD units), `R2SAFE` (minimum r² to avoid singularity), `EXCL_MAX` (max exclusions per atom)
- **Mixing rules:** `mixREQ_arithmetic` — Lorentz-Berthelot: R_ij = R_i + R_j, E_ij = √(E_i·E_j), Q_ij = Q_i·Q_j
- **Math:** `modulo` (PBC wrap), `udiv_cmplx` (complex division), `rotMat` (rotation matrix from quaternion), `clampForce` (force capping for stability)

---

## Forces.cl

Inline pairwise potential functions (not standalone kernels). All return `float4 (Fx, Fy, Fz, E)`. Called from nonbonded, surface, AFM, and GridFF kernels.

| Function | Potential |
|----------|-----------|
| `getLJQH` | Lennard-Jones + Coulomb + H-bond: V = 4ε[(σ/r)¹²−(σ/r)⁶] + q_iq_j/r + H-bond term |
| `getMorseQH` | Morse + Coulomb + H-bond: V = D[e^{−2α(r−r₀)}−2e^{−α(r−r₀)}] + q/r + H |
| `getMorsePLQH` | Factorized Morse (Pauli/London/Coulomb/H-bond channels for GridFF) |
| `getCoulomb` | Bare Coulomb: V = q_iq_j/(4πϵ₀r) |

Macros: `MODEL_LJQH2_PAIR`, `MODEL_MorseQ_PAIR` (compile-time model selection); `ENERGY_*`, `*_DECOMP` (energy decomposition channels).

---

## SPFF.cl

SPFFsp3 force field for sp3 systems: harmonic bonds, cosine angle bending, torsions, π–π alignment, π–σ orthogonalization, H-bond. Includes MD integrator with π-orbital recoil.

**MD step:** `getSPFFf4` → `getNonBond_ex2` → `cleanForceSPFFf4` → `updateAtomsSPFFf4`

| Kernel | Role |
|--------|------|
| `getSPFFf4` | All bonding forces: bonds (E=kΔr²), angles (cosθ), torsions (cos3φ), π-alignment, H-bond. 1 thread = 1 node atom |
| `getSPFFf4_rot` | Same with torque-based group rotation variant |
| `updateGroups`, `groupForce` | Rigid group kinematics: gather atomic forces → group COM force + torque |
| `updateAtomsSPFFf4` | Integrator: velocity-Verlet + π-orbital recoil, bond constraints, π-vector normalization |
| `updateAtomsSPFFf4_rot` | Integrator with quaternion-based group rotation |
| `cleanForceSPFFf4` | Zero force buffers (run before force kernels) |
| `relax_nsteps_serial` | Serial FIRE relaxation (damping + line search) |

**Caveat:** π-orbital vectors are normalized each step; if constraints fight normalization, energy can drift. FIRE damping factor resets on direction change.

---

## UFF.cl

Universal Force Field (Rappé et al.): harmonic bonds, cosine angle bending, torsional dihedrals, inversion (improper) terms. No π-orbital DOFs. Uses eval-then-scatter pattern (compute per-interaction forces into `fint`, then scatter to `fapos`).

| Kernel | Role |
|--------|------|
| `evalBondsAndHNeigh_UFF` | Harmonic bonds (E=k(r−r₀)²) + H-neighbor direction vectors for angle terms |
| `evalAngles_UFF` | Cosine angle bending: E=k(cosθ−cosθ₀)². Uses UFF small-angle harmonic variant |
| `evalDihedrals_UFF` | Torsional: E=V_n[1+cos(nφ−φ₀)] with n=1,2,3 |
| `evalInversions_UFF` | Inversion (improper): Wilson–Morse form V=A(r_inv−r₀)² |
| `assembleForces_UFF` | Scatter per-interaction `fint` → per-atom `fapos` (Newton's 3rd law) |
| `clear_fapos_UFF`, `clear_fint_UFF` | Zero force buffers |
| `updateAtomsSPFFf4` | Simplified velocity-Verlet integrator (no π-recoil) |

**Caveat:** Force scattering uses atomic_add — race-free but non-deterministic accumulation order.

Fused multi-step (`relax_nsteps_{local,global}_UFF`): bonds+angles+dihedrals+inversions with tiled gather (no atomics); optional FAF each step. Angle force must match multi-kernel Fourier formula (historical shrink bug). Topic: `doc/Tasks/PerfBenchmark_Relaxation.md`.

---

## LFF.cl

**Linearized Force Field** — third relax path. Hard intramolecular geometry → distance springs; soft substrate (FAF) / E-field in a large outer predictor; inner diagonal projective Jacobi with \(M/dt^2\).

| Kernel | Role |
|--------|------|
| `lff_jacobi` | Outer soft force + predict; inner Jacobi on packed `neighs`/`KLs` springs; optional FAF |
| `lff_nb_jacobi` | Legacy NB variant (FireCore parity; adsorbates prefer FAF) |

**Caps:** `LFF_WG_SIZE=64` (one molecule ≤ 64 atoms / WG), `MAX_NEIGHBORS=24` (PAH K₁₂+K₁₃+K₁₄ packing).

**Caveats:** Not energy-parity with UFF/SPFF (surrogate for speed). K₁₄ must use capped \(K\) + geometry-based \(l_0\) — raw \(V/(dl/d\phi)^2\) blows up. Uniform `mass=1` for relax. Topic: `doc/Topics/ForceFields/LFF_ProjectiveRelax.md`.

**Driver:** `pytest tests/test_relax_ptcda_faf.py --develop -s` → `debug/test_relax_ptcda_faf/lff_*`

---

## nonbonded.cl

Molecule–molecule nonbonded (LJ/Coulomb/H-bond) with **2nd-neighbor exclusion** — skips 1-2 (bonded) and 1-3 (angle) pairs via packed sorted exclusion list (`excl` array, `EXCL_MAX` per atom). PBC: hardcoded 3×3 image sum in xy. Local-memory tiling (32 atoms per tile) for cache efficiency.

| Kernel | Role |
|--------|------|
| `getNonBond_ex2` | LJ/Coulomb/H-bond pairwise. 1 thread = 1 atom, 1 system per dim-1. Walks sorted `excl` list in parallel with j-loop. PBC: 3×3 images (ix=0..2, iy=0..2) |
| `getShortRangeBuckets` | Spatial bucketing: assigns atoms to grid buckets via R4 repulsion overlap. 1 thread = 1 bucket |
| `getShortRangeBuckets2` | Same with precomputed overlap lists + PBC cell encoding. **Known bug:** lvec.a used where lvec.b intended in PBC shift |
| `sortAtomsToBucketOverlaps` | Sort atoms into bucket overlap list (CSR format). Checks PBC image overlap |

**Caveat:** Exclusion list is sorted by j-index and walked in lockstep with the j-loop — O(EXCL_MAX) per atom. Inverted BB comparisons in bucket kernels are intentional (overlap test).

---

## nonbonded_grid.cl

Combines molecule–molecule nonbonded (LJ/Coulomb) with molecule–substrate forces from a **precomputed GridFF** (B-spline interpolated 3D potential). Replaces O(N_surface) loop with O(1) grid sampling. Two variants: buffer-based (`_ex2`, portable) and texture-based (`_tex`, hardware-cached `image3d_t` with repeat wrapping).

GridFF stores 4 channels: P=Pauli, L=London, Q=Coulomb, H=H-bond. Atom-specific combination via factorized Morse: ej=exp(α·RvdW), PLQH=(ej²·EvdW, ej·EvdW, Q, 0).

| Kernel | Role |
|--------|------|
| `getNonBond_GridFF_Bspline_ex2` | Nonbonded tiling + GridFF B-spline (buffer). Packed `excl` exclusion. 3×3 PBC. 1 thread = 1 atom |
| `getNonBond_GridFF_Bspline_tex` | Same with `image3d_t` texture + neighs/neighCell exclusion. Full (2·nPBC+1)³ PBC loop |
| `getShortRangeBuckets*` | Spatial bucketing (same as nonbonded.cl) |
| `sortAtomsToBucketOverlaps` | Atom sort into buckets |

**Requires:** `common` + `Forces` + `gridFF` + `surface` (for `getR4repulsion`, `fe3d_pbc_comb`).

**Caveat:** Force written with `=` (not `+=`) — must be first force kernel in pipeline. `if(iG>=natoms) return` is after non-bonded loop (all threads must reach barriers).

---

## gridFF.cl

3D B-spline force-field grids: build from substrate atoms, convolve, solve Poisson, sample at arbitrary points. The GridFF precomputes substrate potential on a grid so that MD kernels can sample O(1) instead of summing O(N_surface) atoms.

| Group | Key kernels |
|-------|-------------|
| **Sampling** | `sample3D*`, `sample1D_pbc`, `sampleGridFF_Bspline_points` — tricubic B-spline interpolation with PBC wrapping |
| **Convolution** | `BsplineConv3D*`, `Convolution3D_General` — separable 1D convolutions along x/y/z |
| **Build** | `make_MorseFF*`, `make_Coulomb_points`, `make_GridFF` — project pairwise potentials onto grid |
| **Projection** | `project_atom_on_grid_cubic_pbc`, `project_atoms_on_grid_quintic_pbc` — deposit atomic density onto grid |
| **Poisson** | `poissonW*`, `laplace_real_pbc`, `slabPotential*` — FFT-based Poisson solver with 2D PBC + slab correction |
| **Utilities** | `addMul`, `dot_wg`, `setLinear`, `move`, `setMul` — CG vector ops |

Inline helpers: `make_inds_pbc` (wrapped index pattern), `fe3d_pbc_comb` (B-spline gradient evaluation with PBC).

---

## surface.cl

Molecule–substrate interactions: brute pairwise, folded analytic basis (Fourier decomposition of periodic surface), 2D Ewald electrostatics, isosurface-based forces, macroscopic dipole sheet.

| Kernel | Role |
|--------|------|
| `getSurfMorse`, `getSurfFlat` | Brute-force pairwise Morse/Coulomb over all substrate atoms. O(N_surface) per atom |
| `getSurfFolded*`, `getSurfFolded_harmonics` | Folded analytic basis: surface potential expanded in Fourier components, evaluated via `folded_eval_basis/grad`. O(N_harmonics) per atom |
| `compute_ewald_coefficients`, `eval_potential_*` | 2D Ewald summation for electrostatics of periodic slab. Precomputes K-space coefficients, then evaluates at atom positions |
| `eval_potential_brute` | Direct real-space sum for validation against Ewald |
| `getSurfaceIsoSurfMorse`, `getSurfaceIsoGridFF` | Isosurface-based forces: probe feels force from nearest isosurface point |
| `addDipoleField` | Macroscopic dipole sheet correction (analytic φ for rectangular dipole sheet) |

Helpers: `macro_phi_rect_*` (analytic dipole potential), `folded_eval_basis/grad` (Fourier basis + gradient), `getR4repulsion` (R⁻4 dispersion).

**Caveat:** `folded_eval_grad` has swapped off-diagonal elements (known bug, same class as rigid.cl gyroscopic term).

---

## contact_surface.cl

Quasi-2D contact field for **aperiodic rigid PP-AFM** — replaces `interpFE(img_FF)` during
relaxation. Two representations share brute Morse reference and CG helpers.
Spec: `doc/Topics/AFM/ContactSurface_Static.md` · pitfalls: `doc/Takeways.md`

### Separable (B-spline × poly)

| Kernel | Role |
|--------|------|
| `cs_eval_separable_fe_at` | Inline E,F at probe position (h₀ chain rule) |
| `evalSeparableBsplinePoly` | Batch eval on query list |
| `cs_sep_Av`, `cs_sep_Atv` | Matrix-free fit operators |
| `cs_sep_Atv_w`, `cs_sep_Atv_f_w` | Weighted Atv (Boltzmann + force rows) |
| `cs_sep_Av_f` | Force stencil for fit (`fcomp` = Fx/Fy/Fz) |
| `cs_sep_stencil`, `cs_sep_stencil_f` | Basis stencil assembly |
| `relaxStrokesTiltedContact` | PP-AFM relaxation using separable field |

### PIC (radial atom-centric)

| Kernel | Role |
|--------|------|
| `cs_eval_pic_fe_at` | Inline E,F — shared by eval and PP relaxation |
| `evalRadialPIC` | Point-list PIC eval |
| `cs_pic_eval_tile16` | 16×16 tiled eval with cooperative atom preload |
| `cs_pic_Av`, `cs_pic_Atv`, `cs_pic_Atv_w` | Matrix-free PIC fit |
| `relaxStrokesTiltedPIC` | PP-AFM relaxation using PIC field |

### contact_pme (particle-mesh: coarse mesh + compact cores)

| Kernel | Role |
|--------|------|
| `evalContactPME` / `evalContactPMELocal` | Batch E,F (bucket vs WG+local atom preload) |
| `relaxStrokesTiltedContactPME` / `…Local` | Fused FIRE PP scan (CLI uses **Local**) |
| `fillContactPMEMeshVL` | FIT: raster PAW V_L on coarse mesh (WG+local) |

Report: `doc/Reports/ContactPME_PAW_AFM_MemSpeed_2026-08-11.md` · plan: `doc/Tasks/ContactSurface_PME_ParallelPlan.md`

### Reference & helpers

| Kernel | Role |
|--------|------|
| `cs_brute_plqh_points` | Brute Morse+PLQH at query points |
| `poly_z_doubling_modes` | Compact z/radial basis `t^(m·2^k)` |
| `dot_wg`, `addMul`, `cs_zero`, `cs_copy` | CG vector ops |

**Build:** `common` + `Forces` + `contact_surface` (standalone `ContactSurfaceCL`);
AFMulator adds `AFM.cl` for full scan stack.

**Drivers:**
- `pytest tests/SPM/test_afm_contact_surface.py` — L0 (+ `--develop` for CLI AFM strips)
- `run_spm.py afm --model contact_pme --xyz data/xyz/pyridine.xyz`
- `RUN_CONTACT_PP=1 python tests/testplot_contact_surface.py` → `debug/testplot_contact_surface/`

---

## rigid.cl

6-DOF rigid-body dynamics / relaxation: 3 translational + 3 rotational (quaternion) DOFs per body. Substrate forces from GridFF (B-spline) or folded analytic basis. Quaternion integration uses exact exponential map with Taylor-series `sinc`/`cos` for small-angle stability.

### Folded-substrate kernels (compare these four)

| Kernel | Optimizer | Parallelism | Hessian / solve |
|--------|-----------|-------------|-----------------|
| `rigid_body_folded_kernel` | MD / FIRE (velocities) | 1 WG / body, WG=32, atoms across threads | — |
| `rigid_body_folded_replicas_kernel` | MD / FIRE | 1 thread / replica, WG=128 | — |
| `rigid_body_folded_newton_kernel` | Pure Newton + LM trust | 1 WG / body, WG=32; H in `__local` | lid 0: GE 6×6 |
| `rigid_body_folded_newton_replicas_kernel` | Pure Newton + LM trust | 1 thread / replica, WG=128; H private | per-thread GE 6×6 |

**Newton does not diagonalize H** (no Jacobi). Algorithm: forward FD Hessian on `u=(Δx,Δθ_body)`, symmetrize, solve `(H+λI)Δ=G` with dense 6×6 Gaussian elimination + partial pivoting (`rigid_solve6_lm`), and trust-cap `‖Δ‖`. Accepted boundary steps grow trust and lower `λ`; rejection returns to at least `λ₀`. Replica float32 energy ties are accepted only when the force/torque residual decreases. Parallelism is spent on atoms/replicas — the 6×6 solve is serial (~216 flops).

### Other kernels + helpers

| Symbol | Role |
|--------|------|
| `rigid_body_dynamics_kernel` | Generic E-field + anchor springs (no substrate potential) |
| `rigid_body_gridff_kernel` | GridFF B-spline substrate + MD/FIRE |
| `folded_eval_basis_rigid` / `folded_eval_grad_rigid` | One basis value / world ∇B |
| `folded_FT_replica` | Full F, τ_body, E at one pose (replicas path) |
| `folded_FT_perturb` | Same after ±eps along one of 6 DOFs (FD Hessian column) |
| `rigid_solve6_lm` | `(A+λI)x=b` — GE + partial pivoting; return 0 if singular |
| `rigid_update_FIRE` | Bitzek-style velocity zeroing + dt/damp adapt |
| `rigid_body_pairff_energy_replica_kernel` | Replica×active PairFF+FAF/Kz/anchor energy plus fused real-atom/CoM clash flags; 64-thread NVIDIA tile path |

**Caveat:** Gyroscopic term ω×(I·ω) must use body-frame inertia. Folded setup scales both `I` and `I⁻¹` with the requested effective mass; changing translation alone is inconsistent. FIRE when `md_params.w < 0`. Python: `run_folded(..., fire=True)`, `run_folded_newton` / `run_folded_newton_replicas`; host FD Newton is `relax_newton_host` (debug only).

---

## assembly.cl

Rigid-body multi-molecule packing: apply transforms to molecular fragments and score steric clashes. **Self-contained** (no `common.cl` dependency).

| Kernel | Role |
|--------|------|
| `emit_configuration_xyz` | Apply rigid transforms (rotation + translation) to fragment atoms → assembled XYZ coordinates |
| `evaluate_packing_3d` | Steric clash scoring: counts atom pairs closer than vdW sum. Penalty ∝ overlap depth |

**Driver:** `python tests/testplot_assembly.py` → `debug/testplot_assembly/`

---

## AFM.cl

AFM probe-particle relaxation and image generation. Simulates the oscillating AFM tip by relaxing a probe particle in the sample force field, then computing frequency shift (Δf) images.

**Flow:** build Z-slices → sample field at probe positions → `relaxPoints`/`relaxStrokes*` (FIRE relaxation) → isosurface extraction → `convolveZ` (tip oscillation convolution) → `izoZ` (Δf image).

| Kernel | Role |
|--------|------|
| `getFEinPoints*` | Sample force field (LJ/Morse/Coulomb) at probe positions. Trilinear interpolation from grid |
| `relaxPoints`, `relaxStrokesTilted*` | FIRE relaxation of probe particle in force field. Tilted variant for non-vertical tip |
| `getZisoTilted*` | Isosurface extraction (z-height where force = threshold) |
| `evalLJC_QZs_toImg`, `evalMorseC_QZs_toImg` | Compute Δf image from relaxed probe positions. LJ or Morse potential |
| `evalDispersion_toImg` | Dispersion contribution to Δf |
| `gradient_central_diff` | Numerical gradient via central differences |
| `fdbm_pad_roll_f32` | Pad+roll tip density onto target grid (Round-2 FAST_S3) |
| `fdbm_flip3_f32` | Spatial reverse for ES convolution convention |
| `fdbm_xyz_to_fft_c64` / `fdbm_fft_real_to_xyz_f32` | Host-layout ↔ gpyFFT buffer transpose on device |
| `fdbm_scale_pauli_pow_f32` | `E = A·overlap^β` on GPU (no download) |
| `fdbm_compose_E_to_img` | Pauli + ES + vdW → energy image for gradient |
| `fdbm_mul_poisson_tip_c64` | Fused ES: `ρ_diff(k)·tip(k)/k²` in Fourier space |

Helpers: `interpFE` (trilinear field interpolation), `update_FIRE` (FIRE damping + line search), `tipForce` (tip-sample force model), `read_imagef` (hardware texture sampling).

**FDBM Stage-3 perf:** default path uses `fdbm_*` + gpyFFT (Python: `spammm/SPM/AFM.py`, switch `SPAMMM_AFM_FAST_S3`). Pauli overlap stays a separate FFT (no `1/k²`). See `doc/Tasks/PerfBenchmark_FDBM.md`.

---

## LCAO_grid.cl

Projects LCAO density matrices and molecular orbitals onto 3D grids or arbitrary point lists. Supports any LCAO basis (DFTB, Fireball, Siesta) with s, p, d orbitals. **Self-contained** types (`GridSpec`, `AtomData`, `TaskData`).

**Physics:** ρ(r) = Σ_{i,j} Σ_{μ,ν} D_{μ_i,ν_j} φ_μ(r−R_i) φ_ν(r−R_j), where φ = R_l(r)·Y_l^m(r̂). Radial: cubic B-spline (tabulated) or exp(−β(r−r₀)) (vacuum/STM).

**Orbital order:** Fortran [s,py,pz,px] vs OpenCL [px,py,pz,s] — swizzle `.wyzx` converts.

**Parallelization:** Grid divided into 8³-voxel task blocks. 3-pass scheduling: count overlapping atoms → fill atom lists → compact non-empty blocks. 1 workgroup = 1 task block.

| Kernel | Role |
|--------|------|
| `project_density_sparse` | Sparse density → grid. 4×4 (s+p) hardcoded blocks. Linear neighbor search O(neigh_max) |
| `project_density_sparse_tiled` | Same with __local atom/ρ caching (TILE_ATOMS=8). Uses atomic_add + `.wyzx` swizzle |
| `project_orbital` | Single MO → grid. s+p only (float4 coeffs) |
| `project_orbital_dense` | Dense MO → grid. Full s,p,d via `eval_angular_dense` |
| `project_orbital_points[_exp]` | MO at arbitrary points. Spline or exponential radial |
| `project_orbital_dense_points[_exp]` | Dense MO at points, full s/p/d. `_exp` has no cutoff (STM vacuum) |
| `project_density_dense_points` | Dense density at points. O(N²) per point, max 9 orbitals/atom |
| `mo_overlap_points_exp_sk[_2mol]` | MO overlap with Slater-Koster angular. 1 thread = 1 scan pixel. Quaternion tip rotation |
| `count_atoms_per_block` | Sphere-AABB test, atomic_inc per block. 1 thread = 1 atom |
| `fill_task_atoms` | Write atom indices into block lists. 1 thread = 1 atom |
| `compact_tasks` | Prefix-sum compaction of non-empty blocks |

Helpers: `evaluate_radial` (B-spline), `eval_angular_dense` (Y_lm l=0,1,2), `eval_atom_orbitals` (full φ at point), `sk_contract_sp` (SK s-p contraction), `quat_rotate3` (quaternion rotation).

**Caveats:** Sparse variant limited to s+p (4 orbitals). Dense points limited to 9 orbitals/atom (no f). Exponential variants have no distance cutoff.

---

## LCAO_STM_FGR.cl

First-order Fermi-golden-rule STM only: \(M = c_T^\dagger(H_{TS}-E S_{TS})c_S\). **No** Dyson, GF, SCC, or diagonalization. Radial tables are **custom long-tail** STOs (not mio/3ob). Concatenated after `LCAO_grid` + `LCAO_STM` in `Grid_dftb`.

| Kernel | Role |
|--------|------|
| `build_stm_transfer_sk_tables` | \(\tau = H - E S\) once per energy |
| `stm_fgr_sk_tau_scan_real` | **Production** real-MO scan → `(M, M², npair, 0)` |
| `stm_fgr_sk_tau_scan` | Complex coeffs (Bloch / SOC) |
| `stm_fgr_sk_hs_scan` | Debug: interpolate H and S separately |

**Docs:** `doc/Ideas/LCAO_STM_FGR_WIRING.md`, report `doc/Reports/STM_FGR_Transfer_H_ES_2026-07-29.md`, audit `doc/TopicalAudit/STM_FGR_Transfer.md`.  
**CLI:** `run_spm.py stm fgr`.

**Caveats:** Host must remap DFTB orbital order to `[px,py,pz,s]`. Level-B EH tables make \(H\propto S\) (Level A unfinished). Energy zero of \(H\) and \(E_\mathrm{tunnel}\) must match.

---

## LCAO_STM.cl

STM tunneling current via Dyson equation: G = (I − G₀·V_TS)⁻¹·G₀, where G₀ = diag(G_T, G_S) and V_TS is the tip-sample hopping (Slater-Koster with exponential radial decay). Requires `LCAO_grid.cl` types. **Do not use as the first-order Bardeen/FGR validation path** — see `LCAO_STM_FGR.cl`.

Three approaches (increasing physical accuracy, decreasing GPU cost):

| Kernel | Approach | Parallelization |
|--------|----------|-----------------|
| `response_amplitude_exp` | Scalar Dyson (Tersoff-Hamann s-tip): resp = \|v·a_st^H\|² / \|(E−E_tip) − a_st·G0·a_st^H\|². CPU precomputes G0, v. O(ns²) per point | 1 thread = 1 point |
| `solve_stm_dyson_wg` | Full matrix Dyson with **active subspace trick**: extract ≤32×32 sub-blocks of G_T/G_S into __local, build W=I−G_S·V^H·G_T·V, solve by parallel Gauss-Jordan. 8 KB/matrix | 1 workgroup = 1 pixel |
| `stm_gf_dyson_2mol_mo_scan` | Bardeen transfer Hamiltonian: amp = Σ u_T[it]·H_hop(it,is)·v_S[is]. CPU precomputes u_T=c_tip^H·G_T, v_S=G_S·c_smp. O(n_tip×n_smp) per pixel, no matrix inversion | 1 thread = 1 pixel |

Helpers: `c_add`, `c_sub`, `c_mul`, `c_div` (complex float2 arithmetic, 1e-30 regularization).

**Caveats:** `response_amplitude_exp` limited to 256 sample orbitals (private array). `solve_stm_dyson_wg` drops atoms beyond 8 per side (MAX_ACT_ORB=32). Gauss-Jordan has no partial pivoting (stable for N≤32). All kernels assume 4 orbitals/atom (s+p), zero-padded for H.

---

## lingebra.cl

Batched symmetric eigendecomposition via parallel cyclic Jacobi rotations. **Self-contained.** Each workgroup solves one m×m eigenproblem; threads collaborate on off-diagonal annihilation sweeps.

| Kernel | Role |
|--------|------|
| `local_jacobi_blocks_parallel` | (batch × m × m) → eigenvalues + eigenvectors. Converges in O(log²m) sweeps for typical matrices |

**Tests:** `pytest tests/test_lingebra.py`

---

## Adding a new kernel file

1. Add a file-level header: purpose, kernel list, composition requirements, Python consumer.
2. Register in this README (table + section).
3. Wire via `load_program_multi` in the Python module.
4. Add pytest or `testplot_*` diagnostic.

## Historical note

Pre-2025 monolithic files (`relax_multi.cl`, `relax.cl`, `Grid_dftb.cl`) were split into the modular layout above. Old migration notes lived in `INVENTORY.md` (renamed to this file).
