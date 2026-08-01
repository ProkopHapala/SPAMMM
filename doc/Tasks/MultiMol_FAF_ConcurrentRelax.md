---
type: Task
title: Concurrent multi-molecule relaxation on FAF substrate — fuse folded-basis substrate forces into the multimol kernel
status: design proposed — awaiting USER approval before implementation
tags: [OpenCL, MD, performance, rigid-body, flexible, PairFF, UFF, SPFF, FAF, GridFF, substrate, launch-overhead]
timestamp: 2026-08-01
related: [MultiMol_MD_LaunchOverhead.md, PairFF_FAF_Substrate.md, PairFF_MultiBody_Kernel.md, PairFF_RigidEnergy_MC_GA.md]
skills: [gpu-optimize, port-to-opencl, forcefield-validation]
---

# Concurrent multi-molecule relaxation on FAF substrate

## 1. Goal

We want to relax or run MD of **an assembly of molecules on a surface** — e.g. PTCDA molecules adsorbed on NaCl (or CaF2), forming a self-assembled monolayer. The molecules form **one coupled system**: they interact with each other via non-covalent forces (van der Waals / Pauli repulsion / electrostatics) and with the substrate below. All molecules move simultaneously and feel each other's forces at every step.

Each molecule is a **rigid body** (6-DOF: 3 translational + 3 rotational). The total force on each molecule has three contributions:

1. **Intermolecular PairFF** — non-covalent interactions between molecules (compact-exp Pauli+London + Coulomb). This is the coupling that makes it one system, not N independent molecules. Already in kernel 15.
2. **FAF substrate** — non-covalent interactions with the surface (folded-basis Pauli+London+Coulomb). The substrate is frozen; its effect is pre-fitted into a compact analytic basis. Currently only in kernel 13 (single active mol).
3. **Confinement springs** (optional) — k_z spring toward z_target, anchor springs on pinned atoms. Already in kernel 15.

All molecules are relaxed **concurrently**, each by its own workgroup, in a single kernel launch per timestep. This eliminates the Python launch overhead that dominates when cycling molecules one-by-one (the bottleneck documented in [MultiMol_MD_LaunchOverhead.md](MultiMol_MD_LaunchOverhead.md)).

**Without FAF, the concurrent kernels are useless for this use case** — molecules would float in vacuum with no substrate. This task fuses the FAF substrate forces into the concurrent multimol kernel, completing the original motivation for the entire launch-overhead effort.

---

## 2. COMPLETE KERNEL INVENTORY (rigid.cl — 17 kernels)

### Feature matrix

| # | Kernel name | Substrate | Body mode | Operation | PairFF | Anchors/k_z | FIRE |
|---|-------------|-----------|-----------|-----------|--------|-------------|------|
| 1 | `rigid_body_dynamics_kernel` | none | single | MD | no | anchors | yes |
| 2 | `rigid_body_gridff_kernel` | GridFF | single | MD | no | anchors | yes |
| 3 | `rigid_body_folded_kernel` | FAF (typed+factor) | single | MD | no | anchors | yes |
| 4 | `rigid_body_folded_replicas_kernel` | FAF (typed+factor) | replicas | MD | no | anchors | yes |
| 5 | `rigid_body_folded_newton_replicas_kernel` | FAF (typed only) | replicas | Newton | no | anchors | N/A |
| 6 | `rigid_body_folded_newton_kernel` | FAF (typed only) | single | Newton | no | anchors | N/A |
| 7 | `rigid_body_pairff_kernel` | none | single | MD | yes (legacy) | both | yes |
| 8 | `rigid_body_pairff_unified_kernel` | none | single | MD | yes (unified) | both | yes |
| 9 | `rigid_body_pairff_unified_env_kernel` | none | single+tiled env | MD | yes (unified) | both | yes |
| 10 | `rigid_body_pairff_unified_faf_kernel` | FAF (typed+factor) | single | MD | yes (unified) | both | yes |
| 11 | `rigid_body_pairff_unified_env_faf_kernel` | FAF (typed+factor) | single+tiled env | MD | yes (unified) | both | yes |
| 12 | `rigid_body_pairff_unified_allmol_kernel` | none | multi-body (1 active) | MD | yes (unified) | both | yes |
| 13 | `rigid_body_pairff_unified_allmol_faf_kernel` | FAF (typed+factor) | multi-body (1 active) | MD | yes (unified) | both | yes |
| 14 | `rigid_body_pairff_energy_replica_kernel` | FAF (typed+factor) | replicas | **energy-only** | yes (unified) | both | N/A |
| 15 | `rigid_body_pairff_multimol_kernel` | **none** | multi-body (all active) | MD | yes (unified) | both | yes |
| 16 | `rigid_body_pairff_multimol_persistent_kernel` | **none** | multi-body (all active) | MD | yes (unified) | both | yes |
| 17 | `rigid_body_pairff_multimol_single_wg_kernel` | **none** | multi-body (all active) | MD | yes (unified) | both | yes |

### Key observations

- **Kernels 15/16/17 (concurrent multimol) have NO FAF** — this is the gap this task fills
- **Kernel 14 (energy replica) has FAF** but is energy-only (no forces, no MD) — used for MC/GA scoring
- **Kernel 13 (allmol+FAF) has FAF** but is sequential (1 active, others frozen) — not concurrent
- **Newton kernels (5/6) are typed-only** — factorized mode not supported
- **GridFF (kernel 2) is single-body only** — no PairFF, no multi-body
- **Legacy kernel 7** is superseded by kernel 8 (unified) — kept for compat

### FAF evaluation functions (rigid.cl lines 822-859)

```c
float folded_eval_basis_rigid(u, v, z, prm);    // B = cos(2πk_u·u)·cos(2πk_v·v)·exp(-α·max(0,z-z₀))
float3 folded_eval_grad_rigid(u, v, z, prm, invLvec2d);  // ∇B with chain rule
float folded_coeff_rigid(LCOEFFS, ib, nbasis, ityp, plqh, factorized);  // mode dispatch
```

- **Typed mode** (`folded_meta.y > 0`): `c = coeffs[ityp*nbasis + ib]` — scalar lookup per (type, basis)
- **Factorized mode** (`folded_meta.y < 0`): `c = dot(vload4(ib, coeffs), plqh)` — float4 dot product, 4 components = (pauli, london, coulomb, hbond)

### Local memory budget per kernel (GTX 1650: 48 KB)

| Kernel | Local mem | Notes |
|--------|-----------|-------|
| 1-2 (no FAF) | ~40-56 floats | Minimal |
| 3-6 (FAF only) | ~1280-1500 floats | LBASIS[128] + LCOEFFS[8*128] |
| 7-9 (PairFF, no FAF) | ~72-80 floats | + Lstatic[128×3] |
| 10-13 (PairFF+FAF) | ~1360 floats | PairFF local + FAF local |
| 14 (energy replica) | ~320 floats | Compact, no integration |
| 15-17 (multimol, no FAF) | ~80-2400 floats | 17 is largest (all state in local) |

A direct flat-FAF port would add ~1280 floats (5 KiB) per workgroup and would fit within 48 KiB, but §7 rejects that design. The chosen tensor/materialized path keeps coefficients in a coalesced global read-only buffer and adds only small private recurrence arrays, preserving kernel-15 local-memory occupancy. Kernels 16/17 do not receive FAF in the first implementation.

---

## 3. PYTHON API INVENTORY

### Run methods (RigidBodyDynamics.py)

| Method | Lines | Kernel(s) | Used by |
|--------|-------|-----------|---------|
| `run()` | 740 | 1 | tests (generic) |
| `run_gridff()` | 758 | 2 | tests (GridFF) |
| `run_folded()` | 773 | 3 | FoldedRigidExtension, test_folded_relax |
| `run_folded_replicas()` | 788 | 4 | testplot_ptcda_nacl_replicas |
| `run_pairff()` | 2462 | 7/8/9/10/11/12/13 (auto-select) | RigidAssemblyExtension, RigidBodyVispy, demo_pairff |
| `run_multimol_md()` | 2400 | 15 | bench_multimol_md, test_forcefield |
| `run_multimol_single_wg()` | 2424 | 17 | bench_multimol_md |
| `run_multimol_persistent()` | 2442 | 16 | bench_multimol_md |

### Relaxation methods

| Method | Lines | Backend | Used by |
|--------|-------|---------|---------|
| `relax_fire()` | 1086 | run_folded/run_gridff/run | FoldedRigid.relax_folded |
| `relax_newton_host()` | 1122 | host-side FD + NumPy | debug only |
| `relax_pairff()` | 2551 | run_pairff(fire=True) | RigidBodyVispy, tip_pull_scan |
| `relax_folded()` (FoldedRigid.py) | 685 | run_folded(fire=True) | FoldedRigidExtension, test_folded_relax |

### Energy evaluation (MC/GA)

| Method | Lines | Kernel | Used by |
|--------|-------|--------|---------|
| `eval_energy_replicas()` | 2680 | 14 | testplot_pairff_energy_mc, test_forcefield, greedy_energy_step |
| `eval_energy_system()` | 2744 | 14 (wrapper) | testplot, RigidAssemblyExtension |
| `greedy_energy_step()` (RigidBodyUtils) | 165 | 14 (via eval_energy_replicas) | testplot, RigidAssemblyExtension |

### FAF initialization

| Method | Lines | Purpose |
|--------|-------|---------|
| `init_folded()` | 804 | Store folded_params, create kernels 3/6 |
| `init_replicas()` | 915 | Store replica params, create kernels 4/5 |
| `attach_pairff_faf()` | 2323 | Attach FAF to PairFF (calls init_folded) |
| `enable_pairff_faf()` | 2249 | Toggle faf_mode flag for run_pairff |

### `run_pairff()` kernel selection logic (line 2462)

```
allmol_mode + FAF  → kernel 13  (sequential: 1 active, rest frozen)
allmol_mode no FAF → kernel 12
env_mode + FAF     → kernel 11  (tiled environment)
env_mode no FAF    → kernel 9
unified + FAF      → kernel 10  (single dynamic + single static)
unified no FAF     → kernel 8
legacy             → kernel 7
```

**Gap:** `run_multimol_md()` always uses kernel 15 (no FAF). There is no `run_multimol_md(faf=True)` path.

---

## 4. FAF MODE INVENTORY

### Two fit modes

| Mode | `fit_mode` string | Coeffs shape | Atom_type | Coulomb | Use case |
|------|-------------------|--------------|-----------|---------|----------|
| **Typed** | `'typed_combined'` | `(ntypes, nbasis)` | int IDs | Baked at fit time (discretized Q) | Molecule known at fit time, max speed |
| **Factorized** | `'factorized_plqh'` | `(nbasis, 4)` float4 | float4 PLQH | Substrate-only phi, Q applied at eval | Substrate precomputed, exact per-atom Q |

### 4-component packing (factorized mode)

`coeffs4[:, :]` = (pauli, london, coulomb, hbond) per basis function.
- **Pauli**: `cP = exp(2*alpha*R) * E`
- **London**: `cL = exp(alpha*R) * E`
- **Coulomb**: substrate-only `phi_ewald`, multiplied by atom Q at eval time
- **Hbond**: `cH = exp(2*alpha*R) * H` — **currently not fitted** (set to 0)

### Mixing rules

- **Morse (Pauli+London)**: `R0_mix = R0_probe + R0_substrate` (arithmetic), `E0_mix = sqrt(E0_probe * E0_substrate)` (geometric)
- **Coulomb**: `E_coulomb = Q_atom * phi_substrate` (linear)
- **Hbond**: Lorentzian `min(0, Qi*Qj)` attraction — handled in PairFF via `gij` flag, NOT in FAF

### Substrate definitions

| Substrate | File | Lattice | Z_SURF_TOP | Status |
|-----------|------|---------|------------|--------|
| NaCl | `data/substrates/NaCl_1x1_L3.xyz` | a=(4,0,0), b=(0,4,0) | -3.25 | **Primary** — all fits use this |
| CaF2 | `data/substrates/CaF2_3x3_6L.xyz` | a=(11.59,0,0), b=(5.79,10.04,0) | not hardcoded | Available, less tested |
| Graphene | not found | — | — | **Not available** |

### FAF data files (data/fits/)

```
*_typed.npz       — typed combined fits (PTCDA, NTCDI, formic_acid, etc.)
*_factorized.npz  — factorized 4-component fits (same molecules)
*.npz (plain)     — legacy fits (h2o_nacl, hcooh_nacl, ptcda_nacl)
```

---

## 5. CONSUMER INVENTORY — who uses what

### By task

| Task | Script(s) | Kernel(s) | FAF | Substrate | Mode |
|------|-----------|-----------|-----|-----------|------|
| **MC assembly (greedy)** | testplot_pairff_energy_mc, RigidAssemblyExtension | 14 (energy) | typed | NaCl | batch + GUI |
| **Interactive dragging** | RigidAssemblyExtension, RigidBodyVispy | 12/13 (allmol) | optional | NaCl | GUI |
| **Single-mol relaxation** | FoldedRigidExtension, test_folded_relax | 3 (folded) | factorized | NaCl | GUI + test |
| **AFM imaging (replicas)** | testplot_ptcda_nacl_replicas | 4/5 (replicas) | factorized | NaCl | batch |
| **Lateral scan** | FoldedRigidExtension, test_folded_relax | 3 (via lateral_scan) | factorized | NaCl | GUI + test |
| **Relaxed scan (manipulation)** | FoldedRigidExtension, run_manipulation | 3 (via relaxed_scan) | factorized | NaCl | GUI + CLI |
| **Tip pull** | (nobody — unused) | 13 (via tip_pull_scan) | optional | NaCl | — |
| **Multimol MD benchmark** | bench_multimol_md | 15/16/17 | **none** | **none** | batch |
| **Manipulation path opt** | ManipulationPathOpt | SPFF_cl (NOT rigid) | GridFF | optional | batch |
| **Assembly search** | testplot_assembly | assembly.cl (NOT rigid) | none | none | batch |

### Key findings

1. **`tip_pull_scan` is unused** — implemented in RigidBodyUtils but no consumer calls it
2. **`ManipulationPathOpt` uses SPFF_cl, not RigidBodyDynamics** — completely separate codepath
3. **No swarm/particle swarm optimization exists** anywhere in the codebase
4. **Interactive dragging** uses `run_pairff` (sequential allmol, kernel 12/13) — not concurrent
5. **MC assembly** uses kernel 14 (energy-only) — no forces, no MD, just scoring
6. **Multimol concurrent (15/16/17)** is only used in benchmarks — never in production with FAF

---

## 6. TASKS TO SUPPORT AND THEIR REQUIREMENTS

### Task A: Concurrent multi-molecule relaxation on FAF substrate
**Status:** Not implemented (this task)
**Needs:** Optional optimized FAF in exact ping-pong kernel 15 only
**Challenge:** Preserve synchronous PairFF semantics while adding a tensor-product substrate evaluator without duplicating kernel 15 or reducing occupancy

### Task B: Global optimization of assembly (MC / SA / greedy / swarm)
**Status:** Greedy implemented (kernel 14, energy-only). SA and GA are design-only.
**Needs:**
- **Greedy**: kernel 14 (have it) — energy-only scoring, host-side accept/reject
- **Simulated annealing**: kernel 14 (same) — host changes acceptance to Metropolis
- **GA population**: kernel 14 (same) — host does selection/crossover/mutation
- **Swarm optimization**: kernel 14 (same) — host does particle swarm updates
- **All benefit from concurrent relaxation** (Task A) as a post-step: relax each population member's pose with kernel 15+FAF

**Key insight:** All global optimizers share the same GPU kernel (14). The difference is entirely host-side. The kernel is already algorithm-agnostic.

### Task C: Interactive dragging of molecules
**Status:** Implemented via `run_pairff` (sequential, kernel 12/13)
**Needs:** Real-time force feedback when user drags one molecule; others relax concurrently
**Challenge:** Current approach cycles molecules one-by-one (slow). Concurrent kernel 15+FAF would allow all molecules to relax simultaneously while one is pinned.
**Gap:** Need anchor spring support in concurrent kernel (already present in 15) + FAF (missing).

### Task D: Manipulation trajectory optimization
**Status:** `relaxed_scan` / `manipulation_trajectory` in FoldedRigid.py (single-mol, kernel 3). `ManipulationPathOpt` uses SPFF_cl (different system).
**Needs:** Optimize a path for moving one molecule across surface while others respond
**Challenge:** Current `relaxed_scan` is single-molecule only (kernel 3, no PairFF). To include intermolecular forces during manipulation, need kernel 13 (allmol+FAF, sequential) or kernel 15+FAF (concurrent).
**Gap:** No multi-molecule manipulation exists. `tip_pull_scan` was designed for this but is unused and also sequential.

---

## 7. DESIGN DECISION — ONE PRODUCTION PATH

### 7.1 Scope split

There are two related but different physical problems:

1. **Rigid assembly (this task, implement first):** every molecule keeps fixed internal geometry; one workgroup owns one molecule; PairFF couples molecules; FAF couples every molecule to the substrate.
2. **Flexible assembly (follow-up):** UFF/SPFF changes internal coordinates, intermolecular non-bonded forces couple molecules, and FAF/GridFF acts on every atom. This needs atom-state ping-pong and variable molecular topology; it cannot be added correctly by calling the rigid kernel.

The production target of this task is therefore **rigid PairFF + FAF, exact synchronous update, one workgroup per molecule**. Flexible UFF/SPFF is specified in §11 as a separate second-stage engine.

### 7.2 Choose FAF, not GridFF, for the first production substrate path

Use **FAF** first because the project already has fitted NaCl data, factorized Pauli/London/Coulomb coefficients, arbitrary 2D lattice support, and validated rigid/UFF/SPFF evaluators. FAF is compact and has no large 3D-grid transfer or texture setup.

This is a design choice, not an unsupported performance claim. Before production wiring, benchmark optimized FAF against existing GridFF on the same surface and positions. GridFF is reconsidered only if it is materially faster at equal force/energy accuracy and its memory footprint is acceptable.

### 7.3 Extend kernel 15; do not create a copied kernel 18

**Chosen kernel:** `rigid_body_pairff_multimol_kernel` (15), with optional FAF arguments and one uniform `do_faf` branch.

Do **not** copy kernel 15 into a second 250-line `multimol_faf` kernel. That would duplicate PairFF, FIRE, quaternion, reduction, prediction, and ping-pong logic and guarantee future parity drift. The no-FAF path passes `do_faf=0`; the whole workgroup takes the same branch, so there is no warp divergence. Small valid dummy buffers are bound when FAF is disabled.

Kernels 16 and 17 remain experimental/reference variants and do not receive FAF initially:
- Kernel 16 can deadlock because OpenCL does not guarantee simultaneous workgroup residency.
- Kernel 17 uses one CU and is slower beyond very small systems.
- Kernel 15 already provides the portable exact path and retained/eventless launches.

### 7.4 Exact timestep semantics

For exact mode (`batch=1`):

```text
state A (all old poses)
  ├─ WG 0: PairFF against all molecules in A + FAF at body 0 pose → body 0 state B
  ├─ WG 1: PairFF against all molecules in A + FAF at body 1 pose → body 1 state B
  └─ ...
kernel boundary
state B becomes the next common snapshot
```

Every workgroup reads only `poss_in/qrots_in/v*_in` and writes only its own entry in `*_out`. This is exact synchronous Jacobi dynamics: no workgroup can expose a partially updated pose to another molecule.

For `batch>1`, FAF remains exact for the molecule's evolving own pose, but partner molecule poses are stale or predicted. Therefore `batch>1` remains explicitly approximate and is never the production default.

### 7.5 Canonical FAF runtime representation

Do not carry typed-vs-factorized dispatch into the hot loop. At upload time, materialize **one scalar coefficient per site and basis function**:

```text
factorized input: c(site,b) = dot(coeffs4[b], PLQH[site])
typed input:      c(site,b) = coeffs[type(site), b]
dummy site:       c(site,b) = 0
```

Upload the result in **basis-major layout**:

```text
site_coeffs[b * total_sites + site]
```

At a fixed basis index, all work-items in a warp then read adjacent site coefficients. This gives:
- one branch-free runtime format for typed, factorized, multi-species, and per-atom QEq charges;
- no `FOLDED_TYPES_MAX` runtime limitation;
- no `dot(float4, PLQH)` in every MD step;
- no 4–6 KiB coefficient table duplicated in local memory per workgroup;
- exact preservation of the existing fit physics (materialization is algebra only).

The coefficient buffer is rebuilt only when the FAF fit or per-site REQ/charge changes. Rigid MD does not change it. Stage 0 must compare this coalesced materialized layout against the existing local factorized `float4`+dot evaluator; keep the materialized layout only if profiling confirms that reduced arithmetic outweighs added global reads.

### 7.6 Tensor-product FAF evaluator

The current flat rigid FAF loop calls two trigonometric functions and one exponential **for every basis function**. Current fits are a regular tensor product created by `_build_folded_basis_params()`:

```text
basis index = (iu * nv + iv) * nz + iz
ku = iu, kv = iv, alpha = alphas[iz], z0 = constant
```

Use the existing tensor-Fourier idea from `getSurfFolded_tensor_exp`, but preserve the current linear FAF physics:

1. Compute fractional `(u,v)` once per atom.
2. Evaluate one `sincos(2πu)` and one `sincos(2πv)`.
3. Generate all integer harmonics by complex recurrence (`cos(nφ), sin(nφ)`).
4. Evaluate the `nz` exponentials once per atom and keep `bz[iz]`, `dbz[iz]` in private storage.
5. Traverse `(iu,iv,iz)` in the fit's existing order; load coalesced scalar `site_coeffs[b,site]`; accumulate `E`, `dE/du`, `dE/dv`, `dE/dz`.
6. Transform the fractional gradient with the full inverse 2D lattice matrix, including off-diagonal terms for CaF2/sheared cells.

For the common `nu=4, nv=4, nz=7` fit this reduces special-function work per atom from roughly 224 trig + 112 exp calls to **2 sincos + 7 exp calls**. The remaining loop is multiply-add dominated.

Require regular tensor metadata `(nu,nv,nz)` and fail loud on unsupported legacy irregular bases. Legacy files can be converted or refitted; silently falling back to the slow evaluator would hide performance regressions.

### 7.7 Memory and occupancy

The optimized path does **not** add `LBASIS[128]` or `LCOEFFS[8*128]` to kernel 15. It adds only small private harmonic/z arrays and reads the immutable coefficient matrix from `__global const` memory in a coalesced pattern.

Typical 8×PTCDA coefficient storage (`400 sites × 112 basis × 4 bytes`) is about **175 KiB**, small relative to GPU memory and L2 cache. Existing kernel-15 local memory remains approximately 5–6 KiB/workgroup, preserving occupancy. This is preferable to increasing local memory and repeating a factorized dot product at every basis evaluation.

### 7.8 Host launch path

Extend `run_multimol_md(..., faf=None)` and `_multimol_launch_pair(...)`:

- `faf=None` follows `self.faf_mode`; explicit True/False overrides it.
- `faf=True` requires `init_folded()`/`attach_pairff_faf()` and tensor metadata; otherwise raise.
- Include FAF enable state and coefficient-buffer generation in the launcher cache key.
- Build A→B and B→A kernel objects once; bind all arguments once.
- Use retained normal PyOpenCL enqueue by default; allow `eventless=True` on an in-order queue.
- Never call `generate_kernel_args`, `set_args`, allocate buffers, or upload coefficients inside the timestep loop.
- Keep one `finish()` at the requested synchronization boundary, not per step.

---

## 8. IMPLEMENTATION STAGES

### Stage 0 — evaluator parity and microbenchmark

Before modifying dynamics, implement the tensor evaluator as one reusable inline helper in `rigid.cl` and test it at fixed atom positions against the existing flat FAF evaluator/CPU reference.

Required checks:
- energy and all three force components;
- random `(x,y,z)` samples and boundary-wrapped `(u,v)` samples;
- NaCl orthogonal and synthetic/CaF2 sheared cells;
- typed input materialized to scalar coefficients vs factorized input materialized to the same coefficients;
- coalesced materialized scalar coefficients vs the existing local factorized `float4`+dot path, using event profiling and the same evaluator;
- standard math first; benchmark `native_sincos/native_exp` only after parity.

Proceed only if worst error is within the agreed float32 tolerance and tensor evaluation is faster.

### Stage 1 — canonical coefficient upload

In existing FAF attachment/setup code:
- reuse `materialize_factorized_coeffs()` semantics;
- generate scalar coefficients for every PairFF site (zero for epair/σ-hole dummies);
- transpose to basis-major layout and upload once;
- store `(nu,nv,nz,total_sites)` and z parameters;
- invalidate cached launchers if the buffer or metadata changes.

### Stage 2 — optional FAF in kernel 15

Add the tensor FAF force/energy contribution immediately after intermolecular PairFF accumulation and before anchor/k_z forces. Accumulate into the existing per-atom `f` and `E`, so the existing body-force/body-torque reduction and FIRE/integration remain the single source of truth.

Do not change PairFF, quaternion, FIRE, prediction, ping-pong, or output semantics in this stage.

### Stage 3 — Python API and exact relaxation wrapper

- Extend `run_multimol_md()` instead of adding a parallel API.
- Add `relax_multimol(...)` only if an existing relaxation wrapper cannot be generalized; it should call `run_multimol_md(..., fire=True, batch=1)` in chunks and inspect convergence between chunks.
- GUI callbacks/readbacks occur every display chunk, not every MD step.
- Wire production GUI/rigid manipulation only after L0/L1 review.

### Stage 4 — benchmark and choose defaults

Benchmark `n_mol = 1,2,4,8,16` with realistic PTCDA/NaCl and at least one heterogeneous assembly. Report separately:
- kernel event profiling;
- total wall time including host orchestration;
- time per **global assembly timestep**;
- FAF overhead relative to PairFF-only kernel 15;
- exact concurrent vs existing sequential-active kernel 13;
- normal vs eventless enqueue;
- flat vs tensor FAF evaluator;
- FAF vs GridFF at matched numerical accuracy.

Do not use the cache-clearing `A_naive` benchmark as the scientific production baseline and do not promise a speedup before measurement.

### Stage 5 — production integration

After USER review of numerical trajectories and timing:
- RigidAssembly “Run/Relax”: exact kernel 15 + FAF, `batch=1`.
- Interactive drag: dragged body constrained/anchored; all other bodies relax concurrently.
- Multi-molecule manipulation path: same kernel, changing only anchor targets.
- MC/SA/GA scoring remains kernel 14; optionally relax accepted candidates with kernel 15+FAF. Do not relax every rejected trial.

---

## 9. VERIFICATION SPECIFICATION

### L0 — numerical assertions

1. **FAF evaluator parity:** tensor/materialized evaluator vs existing flat evaluator and CPU reference at fixed positions; compare per-atom E/F, including worst-difference index.
2. **No-FAF regression:** extended kernel 15 with `do_faf=0` must be bit-identical to current kernel 15.
3. **Force-component isolation:** PairFF only, FAF only, anchors/k_z only, then combined.
4. **Combined one-step parity:** all body forces/torques from one common pose snapshot must match a reference assembled from existing PairFF and FAF evaluators within float32 tolerance.
5. **Integrator parity:** one exact `batch=1` step from identical forces/state; compare poses, velocities, FIRE state, world sites, force and torque buffers.
6. **Multi-species and dummies:** verify real sites receive FAF, dummy epair/σ-hole sites receive zero FAF, and heterogeneous molecule offsets are correct.
7. **Finite-state checks:** no NaN/Inf in real poses, velocities, forces, energies, or coefficients.

Do not use a sequential Gauss-Seidel trajectory as the exact reference: updating molecule 0 before evaluating molecule 1 is a different algorithm. Compare forces from a common snapshot or a synchronous CPU reference.

### L1 — agent-reviewed outputs

- Energy, maximum body force/torque, and convergence history for PairFF-only, FAF-only, and combined relaxation.
- Exact concurrent vs approximate `batch>1` RMSD/ΔE table.
- Timing table described in Stage 4.

### L2 — USER visual review

- XY and XZ assembly geometry before/after relaxation on NaCl.
- Intermolecular contacts and adsorption heights labelled.
- Drag/manipulation trajectory showing neighboring molecules responding.

### Scientific checks

- PairFF-only internal forces obey Newton's third law (net assembly force/torque near zero without anchors/substrate).
- With FAF enabled, the assembly momentum change equals the summed external substrate/anchor force.
- FIRE relaxation should reduce force norm and reach a stable basin; energy need not be strictly monotonic on every inertial FIRE step.
- Sheared-cell forces must agree with finite differences in x and y.

---

## 10. SUCCESS CRITERIA

1. Exact `batch=1` concurrent PairFF+FAF passes all L0 component and one-step parity checks.
2. `do_faf=0` has no numerical regression and less than 2% kernel-time regression.
3. Tensor FAF is measurably faster than the existing flat evaluator at the same tolerance.
4. Concurrent execution is faster than sequential active-body kernel 13 for the target assembly sizes; report the crossover rather than assuming it.
5. Interactive relaxation remains responsive with readback/GUI refresh performed only per display chunk.
6. No persistent-global-barrier dependency is required for production correctness.

The task remains unverified until the USER reviews the L1 timing/trajectory data and L2 geometry.

---

## 11. FLEXIBLE UFF/SPFF ASSEMBLIES — SEPARATE FOLLOW-UP

The final project goal also includes molecules that deform internally. This is not the rigid kernel with another force term.

### 11.1 Immediate exact path (reuse existing kernels)

Represent all molecules as one disconnected atomistic assembly:
- covalent topology and UFF/SPFF terms exist only within each molecule;
- non-bonded exclusions remove 1–2/1–3/(configured 1–4) intramolecular pairs;
- all inter-molecule atom pairs remain in the non-bonded kernel;
- FAF or GridFF acts on every real atom.

Use retained/eventless launches for the existing ordered pipeline (clear → bonded → non-bonded → substrate → assemble → integrate), with one final synchronization per requested chunk. Kernel boundaries provide exact global synchronization. This is the lowest-risk first production implementation for flexible assemblies. For this flexible pipeline, benchmark FAF against `getNonBond_GridFF_Bspline_ex2`, which already fuses atom-atom non-bonded and GridFF substrate forces; GridFF may win here even though FAF is selected for the rigid path.

### 11.2 Why existing fused UFF/SPFF kernels are insufficient

`relax_nsteps_{serial,global}` and `relax_nsteps_{local,global}_UFF` already fuse intramolecular terms + FAF for **one molecule**, but they omit molecule–molecule non-bonded interactions and have local-size/topology caps. They cannot model a coupled flexible assembly as written.

### 11.3 Later maximum-performance path

Only after profiling the retained multi-kernel pipeline, consider a new ping-pong flexible-assembly kernel:
- one workgroup per molecule;
- read all atom positions from a common input snapshot;
- evaluate that molecule's bonded UFF/SPFF terms, tiled intermolecular non-bonded forces, and FAF;
- write only that molecule's atoms to the output snapshot;
- one kernel boundary per exact timestep.

This is substantially larger than the rigid task because it needs variable topology offsets, SPFF pi-node recoil, UFF angle/dihedral/inversion gathers, exclusions, and atom-level ping-pong. Implement SPFF and UFF as separate specialized kernels sharing small force helpers; do not build one branch-heavy universal kernel.

---

## 12. DISPOSITION OF OTHER QUESTIONS

- **Global optimization:** kernel 14 remains the energy-scoring engine. SA/GA/PSO are separate algorithm tasks; PSO host updates must be profiled and moved GPU-side if they become hot.
- **Interactive dragging/manipulation:** use exact kernel 15+FAF with anchors; no manipulation-specific kernel.
- **H-bond substrate channel:** leave zero for NaCl unless a physical substrate model/reference justifies it. Do not fit an unconstrained channel merely because storage exists.
- **CaF2:** the 2D inverse lattice supports sheared cells; require fit generation plus finite-difference force parity and a non-hardcoded surface-top reference.
- **Legacy kernel consolidation/deletion:** out of scope. Do not delete or merge old kernels during this task.

---

## 13. FILE MAP

| File | Role / planned change |
|------|-----------------------|
| `kernels/rigid.cl` | Add reusable tensor FAF evaluator; extend kernel 15 with optional FAF; do not duplicate kernel 15 |
| `kernels/surface.cl` | Reference tensor-Fourier recurrence (`getSurfFolded_tensor_exp`) |
| `spammm/forcefields/RigidBodyDynamics.py` | Canonical coefficient upload, cached launch binding, `run_multimol_md(faf=...)` |
| `spammm/surfaces/FoldedRigid.py` | Existing fit metadata and factorized-coefficient materialization semantics |
| `spammm/forcefields/SPFF_cl.py` | Existing flexible SPFF+FAF path; future flexible-assembly orchestration |
| `spammm/forcefields/UFF_cl.py` | Existing flexible UFF+FAF path; future flexible-assembly orchestration |
| `spammm/GUI/RigidAssemblyExtension.py` | Production rigid assembly relaxation/drag integration after review |
| `tests/test_forcefield.py` | L0 evaluator, component, and one-step parity |
| `tests/bench_multimol_md.py` | Exact production benchmark with FAF and event profiling |
| `tests/test_relax_ptcda_faf.py` | Existing flexible UFF/SPFF+FAF reference and future assembly coverage |
| `data/fits/*.npz` | FAF input fits; convert/materialize to canonical runtime coefficients |
| `data/substrates/NaCl_1x1_L3.xyz` | Primary substrate |
| `data/substrates/CaF2_3x3_6L.xyz` | Sheared-cell validation substrate |
