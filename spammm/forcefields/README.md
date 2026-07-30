# forcefields/

GPU-accelerated intramolecular FFs and MD. All OpenCL modules inherit from `utils/OpenCLBase.py`. Three relax paths: **SPFF** (π-orbitals), **UFF** (classic), **LFF** (projective Jacobi springs) — see `doc/Topics/ForceFields/LFF_ProjectiveRelax.md`.

- **FFController.py** — Pure-logic orchestrator: AtomicSystem → FF build → GPU relax → download (no Qt). `ff_type` in `{spff, uff, lff}`
- **FFEvaluator.py** — Single-point UFF/SPFF `eval_fn(pos)→(E,F)` for FD Hessians (`dynamics/Vibrations.py`)
- **UFF_cl.py** — UFF runtime + fused multi-step `relax_serial` / `relax_global` (bonds+angles+dihedrals+inversions, optional FAF)
- **UFFbuilder.py** — AtomicSystem → UFF topology arrays (types, bonds, angles, torsions, inversions, exclusions)
- **SPFF_cl.py** — SPFFsp3 + π-DOFs; fused `relax_serial` / `relax_global` (+ optional FAF); π–π/π–σ audit still open
- **SPFFbuilder.py** — AtomicSystem → SPFF topology (neighbors, params, π-orbitals)
- **LFFSolver.py** — Linearized projective Jacobi: UFF→K₁₂/K₁₃/K₁₄ springs, soft FAF outer, ~50 outer steps vs thousands of MD; kernel `kernels/LFF.cl`
- **RigidBodyDynamics.py** — 6-DOF rigid body GPU dynamics; FIRE quench in-kernel (`fire=True`); pure GPU Newton with boundary-aware trust/LM recovery (`run_folded_newton` / `run_folded_newton_replicas`). Host FD Newton (`relax_newton_host`) is debug-only. **PairFF:** `RigidBodyPairFF` — allmol shared multi-mol buffers, `attach_pairff_faf`, fused PairFF±FAF kernels. GPU `poss`/`qrots` and host `_mb_*` remain per-algorithm working storage.
- **RigidEnsemble.py** — Shared numpy rigid-pose rep (`pos`+`qrot` per molecule, stable ids, xyzw quats). Optional, dependency-free (numpy only), poses-only — templates stay in consuming modules. Rigid modules import it and read `get_poses()`; one-way `ensemble → AtomicGraph` on demand. NOT a global SSOT; `AtomicGraph`/GUI stay independent. See `doc/Tasks/RigidMoleculePose_SSOT.md`.
- **molecule_loaders.py** — Shared rigid-body molecule loaders (XYZ/mol2 + QEq + bonds). Returns `(apos, enames, REQs, bonds)` ready for `_prepare_molecule_pack` / `RigidBodyPairFF.from_molecules`. Reused by `tests/testplot_pairff_energy_mc.py` and `spammm/GUI/RigidAssemblyExtension.py`. Per-molecule loaders (`load_ptcda`, `load_ntcdi`, …) + `LOADERS` registry + `FAF_FITS` map + `remap_fit_for_molecule` + **`graph_to_rigid_fragments`** (splits `AtomicGraph` into connected components → rigid bodies with mass-weighted CoM).
- **RigidBodyAFM.py** — Molecule-on-tip AFM vs substrate GridFF (future: ContactSurface)
- **QEq.py** — Charge equilibration (Rappé–Goddard; Cholesky+Schur / LU)
- **Assembly.py** — Hexagonal SAM packing: GPU clash + orchestration; kernel `kernels/assembly.cl`
- **AssemblyPlot.py** — Top views, clash/strain maps, XYZ export (`tests/testplot_assembly.py`)

## Relax path cheat-sheet

| Path | Hard terms | Soft / substrate | Typical use |
|------|------------|------------------|-------------|
| SPFF fused | bonds, angles, π, … | FAF in fused loop | Accurate adsorbate MD |
| UFF fused | bonds, angles, dih, inv | FAF in fused loop | Universal typing, no π |
| LFF | distance springs from UFF | FAF outer predictor | Fast GUI / morphing |

Perf & PTCDA+FAF numbers: `doc/Tasks/PerfBenchmark_Relaxation.md`.

## Assembly (on-surface SAM)

**Model:** fixed experimental unit cell; **6 C6-related orientations per cell** (`n_sym=6`); collision radius **1.0 Å** (softened vs VdW for rigid search).

**Workflow:**
1. Sample rotations (tilt / inplane / full3d) × fractional translations × 6-fold sym × PBC replicas
2. GPU `evaluate_packing_3d` → clash sum + min inter-molecular distance per config
3. Rank: `clash + zpenalty×z_span + pack_weight×min_dist` (flat, steric, tight packing)
4. Export best configs via `AssemblyPlot` (PNG + XYZ + `.diag`)

**Run:**
```bash
python tests/testplot_assembly.py
python tests/testplot_assembly.py --preset tetraceno --plot_best_k 3
```

**Artifacts:** `debug/testplot_assembly/assembly_{preset}_rank{N}.{png,xyz,diag}` + `_*_{clash,strain,clearance}.png`

**FireCore reference:** `FireCore/tests/tMMFF/test_assembly.py` (driver); core ported from `pyBall/OCL/Assembly.py`
