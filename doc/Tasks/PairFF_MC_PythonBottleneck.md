---
type: Task
title: PairFF MC harness — Python bottleneck (clash detection) violates harness policy
status: analysis complete — fix pending USER approval
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

The agent violated all three: (1) wrote a 511-iteration Python clash-detection loop, (2) left the GPU idle 99% of the time instead of moving clash detection into a kernel, (3) ran the script with buffered output so the user saw nothing for minutes. The agent also restated these policies in AGENTS.md instead of referencing the existing skills — now corrected.

## 2. Profiling results (PTCDA, 4 mols, ntrial=512, rmin_atom=1.6)

Measured 2026-08-01 on NVIDIA GeForce GTX 1650, 50-run average:

| Component | Time (ms) | % of step | Policy violation |
|-----------|-----------|-----------|------------------|
| **GPU kernel** (`eval_energy_replicas`) | **0.66** | **0.7%** | (correct — this is where compute belongs) |
| Trial generation loop (Python) | 5.22 | 5.9% | skill:`python-perf` §1 "Batch Everything" |
| Packing energy loop (512× Python) | 2.00 | 2.3% | skill:`python-perf` §1 |
| **Clash detection (Python loop)** | **82.70** | **93.2%** | skill:`python-perf` §2 "Forbidden: nested loops" |
| **Total** | **88.71** | 100% | |

The GPU does 0.7% of the work. **99.3% is Python orchestration** — the exact anti-pattern `python-perf` forbids.

## 3. Root cause: clash detection Python loop

`greedy_energy_step` (`RigidBodyDynamics.py` lines 2636-2655) runs a triple-nested Python loop: 511 trials × 4 molecules × pairwise distance checks. Each trial calls `_body_sites_world` (quaternion→matrix + matmul) 4×, then `np.einsum` + `np.min` for 6 molecule pairs.

For PTCDA (37 real atoms/mol): **4.2M distance computations in Python** that the GPU could do in a single kernel pass.

## 4. Proposed fixes (priority order)

### Fix 1 (immediate): Vectorize clash detection in NumPy
Batch all 511 trials into one numpy operation. Only the moved molecule rotates per trial; all others are static (compute their world positions once). See skill:`python-perf` §1 "Batch Everything".
**Expected:** 82.7 ms → ~1 ms.

### Fix 2 (immediate): Vectorize trial generation + packing
Generate all random numbers at once, batch quaternion multiply, one `np.sum` for packing. skill:`python-perf` §2.
**Expected:** 7.2 ms → ~0.5 ms.

### Fix 3 (proper fix): Move clash detection to GPU kernel
Add a clash-detection pass to kernel 14 (or a separate kernel) that writes `inf` to `E_out` for clashing replicas. This eliminates the host-side loop entirely and follows skill:`gpu-optimize` / skill:`port-to-opencl` — simulation code belongs in pyOpenCL, not NumPy.
**Expected:** 1 ms → 0 ms (overlapped with energy kernel).

## 5. Expected performance

| Component | Before (ms) | After Fix 1-2 (ms) | After Fix 3 (ms) |
|-----------|-------------|---------------------|-------------------|
| GPU kernel | 0.66 | 0.66 | 0.66 (+ clash, overlapped) |
| Trial gen | 5.22 | 0.5 | 0.5 |
| Packing | 2.00 | 0.05 | 0.05 |
| Clash detect | 82.70 | 1.0 | ~0 |
| **Total** | **88.71** | **~2.2** | **~1.2** |
| **Speedup** | 1× | **~40×** | **~70×** |

At 1.2 ms/step: 1000 steps = 1.2 s. 10000 steps = 12 s.

## 6. CLI improvements (done 2026-08-01)

The test script (`testplot_pairff_energy_mc.py`) now has:
- `--no-plot` / `--no-gif` — skip plotting for pure benchmarks
- `-v/--verbosity {0,1,2,3}` — 0=silent, 1=accepted-only, 2=every-10-steps, 3=debug+timing
- `--timing` — print avg ms/step
- `--profile` — cProfile the MC loop (declared, not yet wired)

## 7. Plotting bottleneck (fixed 2026-08-01)

`_draw_substrate_atoms` in `surface_plots.py` called `ax.scatter` once per ion (3281 calls = 20s/frame). Vectorized to one call per element type: **20s → 0.5s/frame (40×)**. Also reduced substrate replication extent from 160×160 Å to 32×32 Å.
