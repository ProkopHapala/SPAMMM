---
type: Task
title: PairFF MC harness — Python bottleneck (clash detection) violates harness policy
status: implementation tested — awaiting USER confirmation
tags: [PairFF, rigid-body, MonteCarlo, performance, Python, OpenCL, vectorization]
timestamp: 2026-08-01
related: [PairFF_RigidEnergy_MC_GA.md, PairFF_MultiBody_Kernel.md]
skills: [python-perf, gpu-optimize, port-to-opencl]
---

# PairFF MC harness — Python bottleneck violates "Python is harness" policy

## 1. User goal & preferences (2026-08-01)

The user wants the greedy MC assembly harness (`RigidBodyPairFF.greedy_energy_step`) to run fast enough that 1000-step runs are interactive (seconds, not minutes). The molecules are small (10–50 atoms, hundreds–thousands of pairwise interactions) and the GPU is a GTX 1650 (~3 TFLOPS) — so the current 40–89 ms/step is unacceptable for a high-performance system.

**User's stated policies (already documented in `AGENTS.md` + skills, violated here):**
- **Python is the harness, not the engine** (skill:`python-perf`). NEVER write hot loops in Python — batch via NumPy or push to OpenCL.
- **Simulation code lives in pyOpenCL kernels** (skill:`gpu-optimize`, skill:`port-to-opencl`). Kernels must be well-parallelized: workgroups, local memory, minimize host-device transfers and kernel launch overhead.
- **Long-running scripts must print unbuffered progress** — never run silently for minutes.

The original implementation violated all three: (1) a 511-iteration Python clash-detection loop, (2) GPU idle while Python rebuilt and compared atom positions, and (3) buffered progress in a long run. The progress issue had already been corrected before this performance pass.

## 2. Baseline evidence (PTCDA, 4 mols, ntrial=512, rmin_atom=1.6)

The original detailed profile measured 88.71 ms/step on an NVIDIA GeForce GTX 1650:

| Component | Time (ms) | % of step | Policy violation |
|-----------|-----------|-----------|------------------|
| **GPU kernel** (`eval_energy_replicas`) | **0.66** | **0.7%** | (correct — this is where compute belongs) |
| Trial generation loop (Python) | 5.22 | 5.9% | skill:`python-perf` §1 "Batch Everything" |
| Packing energy loop (512× Python) | 2.00 | 2.3% | skill:`python-perf` §1 |
| **Clash detection (Python loop)** | **82.70** | **93.2%** | skill:`python-perf` §2 "Forbidden: nested loops" |
| **Total** | **88.71** | 100% | |

The GPU does 0.7% of the work. **99.3% is Python orchestration** — the exact anti-pattern `python-perf` forbids.

This pass independently reproduced **79.29 ms/step** over five steps with:

```bash
PYTHONUNBUFFERED=1 python3 tests/testplot_pairff_energy_mc.py --mol PTCDA --nmol 4 --steps 5 --ntrial 512 --no-faf --no-plot --timing -v 0
```

`cProfile` attributed 0.267 s of a three-step run to `greedy_energy_step`; 6170 calls to `_body_sites_world` consumed 0.073 s. This confirmed that the earlier profile remained representative.

## 3. Root cause: clash detection Python loop

The old `RigidBodyPairFF.greedy_energy_step` ran a triple-nested Python loop: 511 trials × 4 molecules × pairwise distance checks. Each trial called `_body_sites_world` (quaternion→matrix + matmul) four times, then `np.einsum` + `np.min` for six molecule pairs.

For PTCDA (38 real atoms/mol), that is about **4.4 million atom-pair distances per MC step**, repeatedly dispatched through Python/NumPy calls. The arithmetic volume is small for a GPU; the Python call/loop structure was the cost.

Two secondary costs amplified the problem:

- Trial proposal used one Python iteration per replica and per moved body.
- Packing energy used one Python call/reduction per replica.
- Both test and GUI consumers launched a second full-system energy evaluation after every greedy step even though the active-set energy difference already equals the full-system difference.

## 4. Architecture selected and implemented

### 4.1 Fuse clash detection into kernel 14

`rigid_body_pairff_energy_replica_kernel` already evaluates every active-site/partner-site distance needed for PairFF. It now compares the same `r2` against `rmin_atom²` when both sites are real atoms. CoM separation is checked once per active/partner molecule pair. No second kernel, atom transform, distance pass, global buffer, or host download was added.

The previous reserved `E_out.w` channel now reports a clash count. The kernel remains optimizer-agnostic: it reports geometry; `greedy_energy_step` decides that flagged trial replicas get `inf`, while replica 0 remains the current-state reference even if the initial geometry violates a requested cutoff. `energy_changed` intentionally continues to reduce only `x/y/z`.

Only active-vs-partner pairs are checked. Frozen-frozen geometry is invariant across a trial batch, so recomputing those pairs cannot affect move acceptance. This also avoids a global clash reduction.

Local reduction storage increased by 256 B (one scalar per thread), from about 2.9 to **3.1 KiB/workgroup**. This remains small relative to the GTX 1650's 48 KiB local-memory budget.

### 4.2 Batch host-side proposal math

`greedy_energy_step` now:

- generates all translation/rotation random values in one RNG call;
- applies the planar quaternion product to the whole `(replica, moved)` batch with NumPy broadcasting;
- computes all packing energies in one reduction;
- uploads one contiguous replica pose batch and downloads one `float4` energy batch.

There is no Python loop over trials, moved molecules, molecule pairs, or atom pairs.

### 4.3 Remove unnecessary synchronization and launches

- The explicit `queue.finish()` before a blocking device-to-host copy was redundant and was removed.
- `tests/testplot_pairff_energy_mc.py` and `RigidAssemblyExtension` now update full energy with `Ebest-E0` after an accepted move. Frozen-frozen terms cancel exactly, so a second replica-kernel launch per step was wasted work.
- The GUI recomputes its full baseline only when the packing constant changes.

## 5. Measured result

All figures below are from the NVIDIA GeForce GTX 1650 on 2026-08-01:

| Case | Before | After | Speedup |
|------|--------|-------|---------|
| PTCDA, 4 mols, 512 trials, PairFF, 5 steps | 79.29 ms/step | 0.69 ms/step | 115× |
| PTCDA, 4 mols, 512 trials, PairFF, sustained 100 steps | 79–89 ms/step baseline range | **0.66 ms/step** | **120–134×** |
| PTCDA, 4 mols, 512 trials, PairFF+FAF, 20 steps | not re-baselined | **0.93 ms/step** | — |

The sustained PairFF run therefore spends about **0.66 s in `greedy_energy_step` per 1000 steps**, versus roughly 79–89 s before. Plot/GIF generation is separate.

The post-change 50-step `cProfile` run shows 0.042 s total in 50 `greedy_energy_step` calls (~0.84 ms/call under profiler); `_body_sites_world` is absent from the MC hot path.

## 6. CLI improvements (done 2026-08-01)

The test script (`testplot_pairff_energy_mc.py`) now has:
- `--no-plot` / `--no-gif` — skip plotting for pure benchmarks
- `-v/--verbosity {0,1,2,3}` — 0=silent, 1=accepted-only, 2=every-10-steps, 3=debug+timing
- `--timing` — print avg ms/step
- `--profile` — cProfile the MC loop (declared, not yet wired)

## 7. Plotting bottleneck (fixed 2026-08-01)

`_draw_substrate_atoms` in `surface_plots.py` called `ax.scatter` once per ion (3281 calls = 20s/frame). Vectorized to one call per element type: **20s → 0.5s/frame (40×)**. Also reduced substrate replication extent from 160×160 Å to 32×32 Å.

## 8. Verification performed

- Kernel compile and inline energy parity, PairFF only: `|err| = 3.912e-08`.
- Kernel compile and inline energy parity, PairFF+FAF: `|err| = 3.912e-08`.
- New focused L0 GPU test compares kernel clash flags against CPU real-atom/CoM distances and proves that enabling clash reporting does not change energy channels: `1 passed`.
- Focused GUI MC parity test after energy-cache change: `1 passed`.
- Deterministic 100-step run preserved the same accepted count (`20/100`) and final energy (`2.681513 eV`) before and after removing the redundant full-energy launch.
- Standard `pytest -m "not slow"`: **384 passed**, including all PairFF/rigid-assembly tests; 12 failures and 2 errors are in unrelated AFM contact-surface, directory/export, Ewald, SMILES/unified-editing, and missing-PME-data paths.

Commands:

```bash
pytest tests/test_forcefield.py::test_pairff_replica_clash_channel_matches_cpu -s
pytest tests/GUI/test_rigid_assembly_extension.py::test_mc_step_parity_vs_testplot -s
python3 tests/testplot_pairff_energy_mc.py --mol PTCDA --nmol 4 --steps 100 --ntrial 512 --no-faf --no-plot --timing -v 0
python3 tests/testplot_pairff_energy_mc.py --mol PTCDA --nmol 4 --steps 20 --ntrial 512 --no-plot --timing -v 0
```

This task is deliberately **not marked fixed/done** until the user confirms the behavior and performance.

## 9. Recommendations

### Keep the present CPU/GPU boundary

At 0.66–0.93 ms/step, moving proposal RNG into OpenCL would add persistent RNG state, reproducibility work, and optimizer coupling for at most a sub-millisecond gain. The current boundary is appropriate: NumPy proposes one batch; OpenCL evaluates all physics and clashes in one launch.

### Profile before any further kernel rewrite

The next backend decision should use OpenCL event timing and an `ntrial × nmol × atoms` sweep. If large populations or multi-active GA moves make kernel 14 dominant, investigate:

1. persistent replica pose buffers with a small GPU proposal kernel to avoid full pose upload;
2. incremental upload of only moved-body poses;
3. specialization for `nactive=1`, avoiding general active-list classification;
4. partner world-position caching only when many active molecules share the same replica.

Do not implement these speculatively: they add state and kernel variants, and current measured throughput already exceeds the interactive target.

### Deferred low-priority cleanup

Suitable for a cheaper follow-up model:

- wire or remove the currently advertised `--profile` flag;
- make `--timing` wording explicitly “end-to-end greedy step” and remove stale `t_gpu_total`;
- clean unused driver locals/import shadowing;
- add benchmark usage to `tests/README.md`;
- extend L0 coverage to mixed molecule sizes and multi-active clash aggregation.
