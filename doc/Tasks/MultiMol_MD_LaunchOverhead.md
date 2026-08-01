---
type: Task
title: Multi-molecule MD launch-overhead experiment — eliminate Python harness bottleneck for concurrent rigid-body dynamics
status: verified — ping-pong fix correct, benchmarks reproduced, awaiting USER confirmation for production integration
tags: [OpenCL, MD, performance, benchmark, rigid-body, PairFF, UFF, launch-overhead]
timestamp: 2026-08-01
related: [PairFF_MC_PythonBottleneck.md, PairFF_MultiBody_Kernel.md, MultiMol_FAF_ConcurrentRelax.md]
skills: [gpu-optimize, port-to-opencl, python-perf]
---

# Multi-molecule MD launch-overhead experiment

## 1. Problem statement

We run molecular dynamics of **multiple rigid molecules on a substrate** (UFF/PairFF
non-bonded + FAF substrate). Each molecule is one workgroup in the PairFF kernel.
Molecules interact with each other AND with the substrate.

The current `run_pairff(N, ...)` runs N MD steps inside one kernel call. But
within a kernel, `barrier()` only synchronizes within one workgroup — there is
no cross-workgroup barrier. So during the in-kernel loop, each molecule sees
**stale positions** of all other molecules (read once at kernel start). This is
correct for sequential relaxation (one active molecule, others frozen) but
**wrong for concurrent multi-molecule MD** where all molecules move simultaneously.

For correct concurrent MD, every step requires a **device-wide synchronization
point** — i.e. a kernel boundary. This means N steps = N kernel launches, and
the Python harness overhead per launch becomes the bottleneck.

## 2. Hardware constraints (measured 2026-08-01)

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce GTX 1650 |
| Compute units | 14 |
| Max workgroup | 1024 |
| Local memory | 48 KB |
| OpenCL C version | 1.2 |
| `cl_khr_command_buffer` | **not supported** |
| `cl_khr_device_enqueue` | **not supported** |

**Eliminated strategies:**
- Command buffers (`cl_khr_command_buffer`) — driver lacks extension
- Device-side enqueue (`cl_khr_device_enqueue`) — driver lacks extension
- OpenCL 2.0+ features — driver is OpenCL C 1.2

## 3. Current overhead sources in `run_pairff`

Per `run_pairff(1, ...)` call (one MD step):

1. `generate_kernel_args(kname)` — rebuilds arg list from `kernel_params` dict
   (dict lookups + list construction + header parsing for every arg)
2. `krnl(self.queue, gs, ls, *args)` — PyOpenCL `Kernel.__call__` internally
   calls `set_args()` for ALL arguments, then `enqueue_nd_range_kernel()`,
   then creates an `Event` wrapper
3. `self.queue.finish()` — **blocks the host** until GPU completes

For a small system (4 molecules × 38 atoms), the GPU kernel itself runs in
<1 ms. The Python overhead per call is estimated at 0.1–0.5 ms, and
`queue.finish()` adds a host-device round-trip latency bubble.

## 4. Strategies to benchmark

Five strategies, ordered from safest to most aggressive. All are implementable
in pure Python + PyOpenCL (no C/C++ compilation).

### Strategy A — Optimized bare-enqueue loop (niter=1 per call)

**Concept:** Keep the current single-kernel-per-step approach but eliminate all
avoidable Python overhead.

**Changes:**
1. Retain `cl.Kernel` object once (already done: `self.krnl_pairff_unified`)
2. Call `krnl.set_args(...)` **once** before the loop — all buffer handles are
   persistent, contents change but handles don't
3. Put changing scalars (`dt`, `md_params`, `niter`) into a **device-side
   control buffer** (`sim_state` float4) so `set_args` is never repeated
4. Inner loop: only `cl.enqueue_nd_range_kernel(queue, krnl, gs, ls)` — no
   `finish()`, no `set_args()`, no `generate_kernel_args()`
5. Single `queue.finish()` at the end of the 100-step batch

**Correctness:** Fully correct. Each kernel launch is a device-wide sync point.
All workgroups see updated positions from the previous step.

**Expected overhead per step:** ~10–50 µs (bare `clEnqueueNDRangeKernel` +
Python call overhead + Event wrapper). vs. current ~100–500 µs.

**Risk:** None. This is the standard PyOpenCL fast path documented in
[PyOpenCL docs](https://documen.tician.de/pyopencl/runtime_program.html).

### Strategy B — Split force + integrate into 2 kernels

**Concept:** Separate the force computation from the integration step.
- `force_kernel`: global NDRange, each workgroup computes forces on its molecule
  from ALL other molecules' current positions. Writes force/torque to buffers.
- `integrate_kernel`: global NDRange, each workgroup reads its force/torque and
  updates position/velocity/quaternion. Writes new positions to buffers.

**Changes:** Same optimized enqueue as Strategy A, but 2 enqueues per step
(200 total for 100 steps).

**Correctness:** Fully correct. Force kernel completes before integrate kernel
starts (in-order queue). Integrate kernel completes before next force kernel.

**Why test it:** The current unified kernel does force+integrate in one launch.
Splitting allows the force kernel to be re-used for energy-only evaluation
(no integration), and may improve occupancy if the unified kernel has register
pressure. But 2× the launches may be slower.

**Expected:** Likely slower than A due to 2× launch overhead, but worth
measuring to quantify the cost of kernel splitting.

### Strategy C — Persistent kernel + software global barrier

**Concept:** One kernel launch with an internal 100-step loop. A software
global barrier synchronizes all workgroups between steps using atomic counters.

**Global barrier pattern:**
```c
// At end of each step:
barrier(CLK_LOCAL_MEM_FENCE);           // local sync
if (lid == 0) atomic_add(g_barrier_count, 1);
while (atomic_load(g_barrier_count) < n_workgroups * (step+1)) {
    // spin-wait
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
```

**Critical constraint — residency deadlock:**
If the GPU cannot make all workgroups simultaneously resident, the first
resident workgroups spin at the barrier waiting for non-resident workgroups
that can never be scheduled → **deadlock**.

**Mitigation:** Launch exactly `n_workgroups ≤ max_resident_workgroups`.
On GTX 1650 with 14 CUs, this means ≤14–28 workgroups (1–2 per CU). Each
workgroup handles one molecule, so this limits us to ≤14–28 molecules.

**Changes:**
1. New kernel variant with `niter` loop + atomic global barrier
2. Launch with `global_size = n_mols * nloc`, `local_size = nloc`
3. Must query/guess max resident workgroups (conservative: 14 = n_CUs)
4. Atomic counter buffer, reset before each launch

**Correctness:** Correct IF all workgroups are simultaneously resident.
No portable OpenCL guarantee of this — it's an architectural gamble.

**Expected overhead per step:** ~0 (no Python, no launch). Only GPU dispatch
+ spin-wait latency.

**Risk:** HIGH. Deadlock if residency assumption is wrong. Must be tested
carefully with varying molecule counts. May break on different GPUs/drivers.

### Strategy D — Stale-position multi-step (current approach, baseline)

**Concept:** Run K steps per kernel call with stale inter-molecule positions.
K=1 is fully correct. K>1 is an approximation — each molecule integrates
against a snapshot of other molecules' positions from step start.

**Changes:** None — this is `run_pairff(K, ...)` as currently implemented.

**Why test it:** Quantify the accuracy/speed tradeoff. If K=5–10 introduces
acceptable energy drift for short MD runs, this is the simplest speedup.

**Correctness:** Approximate for K>1. Energy conservation degrades with K.
Acceptable for relaxation/annealing, questionable for production MD.

**Expected:** K× fewer Python calls. For K=10, 10× reduction in overhead.

### Strategy E — Single-workgroup all-molecule kernel

**Concept:** If the total number of atoms across all molecules fits in one
workgroup's local memory, run ALL molecules in a single workgroup. Then the
in-kernel multi-step loop is fully correct — `barrier(CLK_LOCAL_MEM_FENCE)`
synchronizes all molecules.

**Constraint:** Local memory = 48 KB. Each atom needs ~32 bytes (pos + REQ +
force). So ~1500 atoms max. For 4 × 38-atom molecules = 152 atoms, this fits
easily.

**Changes:**
1. New kernel variant: one workgroup, all molecules in local memory
2. Each thread handles multiple atoms across molecules
3. In-kernel loop with `barrier(CLK_LOCAL_MEM_FENCE)` between steps
4. No global barrier needed — single workgroup

**Correctness:** Fully correct. All molecules in one workgroup, local barrier
synchronizes everything.

**Expected:** Fastest for small systems (1 Python call, no launch overhead
per step). Limited by single-workgroup occupancy (only 1 CU used).

**Risk:** Low. Standard OpenCL pattern. Only limitation is system size.

## 5. Experiment design

### 5.1 Standalone headless script

`tests/bench_multimol_md.py` — standalone, no GUI, no plotting (except optional
debug output). Pure benchmark + correctness check.

### 5.2 System configurations

| Config | Molecules | Atoms/mol | Total atoms | Purpose |
|--------|-----------|-----------|-------------|---------|
| Small  | 4 PTCDA   | 38        | 152         | Fits single WG (E), typical MC case |
| Medium | 8 PTCDA   | 38        | 304         | Exceeds single WG, tests A/C |
| Large  | 16 PTCDA  | 38        | 608         | Tests C residency limit (14 CUs) |

### 5.3 Benchmark protocol

For each strategy × config:

1. **Warm-up:** 10 steps (not timed) — ensures buffers allocated, kernel compiled
2. **Timed run:** 100, 500, 1000 steps
3. **Measure:**
   - Wall time (Python `time.perf_counter`)
   - GPU time (`clGetEventProfilingInfo` on first/last event)
   - Steps/second
   - µs/step
4. **Correctness:**
   - Energy drift over 100 steps vs Strategy A (reference)
   - Final positions vs Strategy A (RMSD)
   - Energy conservation (ΔE/E₀ for NVE, or monotonic decrease for FIRE)

### 5.4 Output format

```
=== Multi-mol MD launch-overhead benchmark ===
GPU: NVIDIA GeForce GTX 1650 (14 CUs, 48KB local)

Config: 4×PTCDA (152 atoms), 100 steps
  Strategy A (optimized enqueue):    12.3 ms  (8130 steps/s,  1.23 µs/step)
  Strategy B (split force+integrate): 23.1 ms  (4330 steps/s,  2.31 µs/step)
  Strategy C (persistent+barrier):     5.2 ms  (19230 steps/s, 0.52 µs/step)
  Strategy D (stale K=10):             3.1 ms  (32260 steps/s, 0.31 µs/step) [ΔE/E₀=0.03]
  Strategy E (single-WG):              4.8 ms  (20830 steps/s, 0.48 µs/step)

Config: 8×PTCDA (304 atoms), 100 steps
  ...
```

### 5.5 What we want to learn

1. **How much overhead is avoidable?** (A vs current `run_pairff(1, ...)` loop)
2. **Is kernel splitting worth it?** (B vs A)
3. **Can persistent kernel work safely on GTX 1650?** (C — does it deadlock?)
4. **How much error does stale-position introduce?** (D — ΔE/E₀ for K=1,5,10)
5. **What's the crossover where single-WG is too small?** (E vs A at 4, 8, 16 mols)
6. **Which strategy is best for production?** (Pareto: speed vs correctness vs generality)

## 6. Implementation plan

### Phase 1 — Baseline + Strategy A (optimized enqueue)

1. Create `tests/bench_multimol_md.py`
2. Build system using existing `RigidBodyPairFF.from_molecules` + `attach_pairff_faf`
3. Implement Strategy A:
   - Retain kernel object
   - `set_args()` once (move dt/md_params to device buffer)
   - Bare `enqueue_nd_range_kernel()` loop
   - Single `finish()` at end
4. Benchmark current `run_pairff(1, ...)` loop vs Strategy A
5. **Decision gate:** If A is already fast enough (<10 µs/step), stop here.

### Phase 2 — Strategy E (single-workgroup)

1. Write new kernel `rigid_body_pairff_single_wg_kernel`:
   - One workgroup, all molecules in local memory
   - In-kernel loop with `barrier(CLK_LOCAL_MEM_FENCE)`
   - Same PairFF physics as `rigid_body_pairff_unified_kernel`
2. Benchmark for 4-mol case
3. **Decision gate:** If E is fastest for small systems, use it for ≤14 mols.

### Phase 3 — Strategy D (stale-position sweep)

1. Run current `run_pairff(K, ...)` for K=1,2,5,10,20
2. Measure energy drift and position RMSD vs K=1
3. **Decision gate:** Identify max K with acceptable drift (<1% ΔE/E₀).

### Phase 4 — Strategy C (persistent kernel)

1. Write new kernel `rigid_body_pairff_persistent_kernel`:
   - Internal `niter` loop + atomic global barrier
   - Launch with `n_workgroups = min(n_mols, 14)` (1 per CU)
2. Test for deadlock with 4, 8, 14, 16, 28 molecules
3. **Decision gate:** If C deadlocks above 14 mols, limit to ≤14 or abandon.

### Phase 5 — Strategy B (split kernels)

1. Write `force_only_kernel` and `integrate_only_kernel`
2. Benchmark 2-enqueue-per-step vs A
3. **Decision gate:** Only keep if it enables something A can't (e.g. energy-only eval).

### Phase 6 — Cross-strategy comparison + recommendation

1. Generate summary table (all strategies × all configs)
2. Pareto plot: speed vs correctness
3. Recommend production strategy per system size

## 7. Key design decisions

### 7.1 Device-side control buffer

Instead of passing `dt`, `md_params`, `niter` as scalar kernel arguments (which
require `set_arg()` every call), store them in a persistent `float4 sim_state`
buffer:

```c
__global float4* sim_state;  // (dt, damp_lin, damp_ang, fire_flag)
```

The kernel reads from this buffer. The host writes once before the loop. During
the loop, no `set_args()` is needed.

For FIRE adaptive dt/damp, the kernel writes back to `fire_state` buffer
(already exists). No host intervention needed.

### 7.2 No `queue.finish()` in the loop

The in-order command queue guarantees that kernel N+1 starts only after kernel
N completes. `finish()` is only needed before a blocking `fromGPU()` readback.
For a pure 100-step MD run with no intermediate readback, one `finish()` at
the end suffices.

### 7.3 New kernels are variants, not replacements

New kernels (single-WG, persistent, split force/integrate) are **additional
variants** in `kernels/rigid.cl`. The existing kernels are not modified. The
experiment script selects which kernel to use.

### 7.4 UFF / PairFF parameters

The experiment uses the existing PairFF non-bonded model (compact exponential
with R, E, Q parameters — UFF-compatible). No new force field code needed.
The FAF substrate is optional (`--no-faf` flag for pure inter-molecular MD).

## 8. File layout

```
tests/bench_multimol_md.py          — experiment script (standalone, headless)
kernels/rigid.cl                    — new kernel variants added here
doc/Tasks/MultiMol_MD_LaunchOverhead.md — this spec
debug/bench_multimol_md/            — benchmark output (gitignored)
```

## 9. Success criteria

1. **Strategy A** achieves <50 µs/step for 4-mol system (10× improvement over
   naive `run_pairff(1,...)` loop with `finish()` each step)
2. **Strategy E** achieves <5 µs/step for 4-mol system (single-WG, no launch
   overhead per step)
3. **Strategy C** runs without deadlock for ≤14 molecules
4. **Strategy D** identifies max K with <1% energy drift over 100 steps
5. Clear recommendation for which strategy to use per system size

## 10. What this experiment does NOT do

- Does not modify existing `RigidBodyDynamics.py` production code
- Does not change the GUI or CLI
- Does not implement new force fields (uses existing PairFF/FAF)
- Does not add C/C++ dependencies
- Does not use command buffers or device-side enqueue (not supported)
- Migration to production modules is a separate decision after benchmarks

## 11. Implementation and verified results (2026-08-01, NVIDIA GeForce GTX 1650)

### Kernels added to `kernels/rigid.cl`

- **Kernel 15** `rigid_body_pairff_multimol_kernel` — N workgroups (1 per molecule), ping-pong I/O buffers (`poss_in`→`poss_out`), `niter` internal loop, optional `predict_partners` constant-velocity extrapolation. This is the production exact-path kernel.
- **Kernel 16** `rigid_body_pairff_multimol_persistent_kernel` — same physics + ping-pong state + atomic-counter global barrier. One launch = N steps. Experimental (requires all WGs simultaneously resident).
- **Kernel 17** `rigid_body_pairff_multimol_single_wg_kernel` — single workgroup, all molecules in local memory, Jacobi update (all forces from one snapshot → integrate all → barrier). One launch = N steps. Exact but uses only 1 CU.

### Production API (`RigidBodyPairFF`)

- **`run_multimol_md(n_steps, dt, ..., batch=1, predict_partners=False, eventless=False)`** — exact synchronous concurrent MD. `batch>1` runs K steps per launch with stale or predicted partner positions (approximate). `eventless=True` uses ctypes direct enqueue (no OpenCL event object).
- **`run_multimol_single_wg(n_steps, ...)`** — exact single-workgroup Jacobi. Limited to ≤32 molecules.
- **`run_multimol_persistent(n_steps, ...)`** — experimental persistent kernel. Requires `n_mols ≤ max_compute_units`.
- **`OpenCLBase.bind_ndrange(kernel, gs, ls, eventless=True)`** — returns a closure that calls `clEnqueueNDRangeKernel` via ctypes with `event=NULL`, bypassing PyOpenCL's event wrapper overhead (~2 µs/launch saved).

All three concurrent APIs **fail loud** if FAF is enabled — they do not silently omit the substrate.

### Correctness: ping-pong race fix

The original `multimol_kernel` read and wrote `poss/qrots` in the same multi-workgroup dispatch. A fast workgroup could overwrite its pose while a slower workgroup still read the old timestep — a **read/write race** within one dispatch (kernel boundaries synchronize, but there is no barrier *inside* a dispatch).

**Fix**: separate input/output buffers (`poss_in`/`poss_out`). Two retained kernel objects bind A→B and B→A. `run_multimol_md(batch=1)` alternates them. Odd launch counts copy the final state back to canonical buffers.

The persistent kernel (kernel 16) uses the same ping-pong + an atomic-counter global barrier (`atomic_add` + spin-wait on `g_barrier`). This is now **correct** — verified by L0 parity test (`test_pairff_multimol_launch_parity`, tolerance 2e-6).

The single-WG kernel (kernel 17) was changed from Gauss-Seidel (sequential molecule integration) to **Jacobi** (all forces from one local snapshot → integrate all → barrier). This is both faster and more correct (no ordering bias).

### Benchmark script

`tests/bench_multimol_md.py` — standalone, headless, reuses `from_molecules` setup. Tests all strategies with configurable molecule count, steps, K (batch size).

### L0 parity test

`tests/test_forcefield.py::test_pairff_multimol_launch_parity` — 2×HCOOH, 3 steps. Verifies:
- Eventless = bit-identical to ping-pong (atol=0)
- Single-WG = within 2e-6 of ping-pong
- Persistent = within 2e-6 of ping-pong
- K=1 predictor = bit-identical to K=1 (atol=0)

**Status: PASSED** (verified 2026-08-01).

### Strategies tested

| Key | Description | Correct? |
|-----|-------------|:--------:|
| `A_naive` | `run_multimol_md(1)` + launcher cache clear per step (anti-pattern baseline) | ✓ |
| `A_opt` | `run_multimol_md(N)` — cached ping-pong launchers, single `finish()` | ✓ |
| `A_eventless` | Same as A_opt but `eventless=True` (ctypes enqueue, no cl_event) | ✓ bit-identical |
| `A_niter_opt` | `run_multimol_md(N, batch=K)` — K steps/launch, stale partner positions | ~approx |
| `A_predict` | `run_multimol_md(N, batch=K, predict_partners=True)` — constant-velocity extrapolation | ~approx |
| `C_persist` | `run_multimol_persistent(N)` — 1 launch, atomic global barrier, ping-pong | ✓ (experimental) |
| `E_singlewg` | `run_multimol_single_wg(N)` — 1 workgroup, local Jacobi, 1 launch | ✓ |
| `D_stale` | Existing `run_pairff(K)` cycling active_mol (sequential, not concurrent) | ✗ inaccurate |

### Verified benchmark results (2026-08-01, 100 steps, K=10, dt=0.05)

All numbers below are from a fresh run of `tests/bench_multimol_md.py` on NVIDIA GeForce GTX 1650 (14 CUs). RMSD and ΔE are vs `A_opt` (exact ping-pong reference).

| Config | Strategy | µs/step | steps/s | vs A_naive | E_final (eV) | RMSD (Å) | ΔE (eV) |
|--------|----------|--------:|--------:|-----------:|-------------:|---------:|--------:|
| 1×PTCDA | A_naive | 511.7 | 1,954 | 1.0× | 0.0000 | — | — |
| 1×PTCDA | **A_opt** | **11.2** | **89,487** | **45.7×** | 0.0000 | — | — |
| 1×PTCDA | A_eventless | 9.4 | 106,889 | 54.4× | 0.0000 | 0.000 | 0.000 |
| 1×PTCDA | **A_niter_opt** | **3.8** | **263,190** | **134.7×** | 0.0000 | 0.000 | 0.000 |
| 1×PTCDA | A_predict | 3.8 | 263,457 | 134.7× | 0.0000 | 0.000 | 0.000 |
| 1×PTCDA | C_persist | 6.7 | 149,553 | 76.4× | 0.0000 | 0.000 | 0.000 |
| 1×PTCDA | E_singlewg | 6.0 | 165,549 | 85.3× | 0.0000 | 0.000 | 0.000 |
| 2×PTCDA | A_naive | 542.0 | 1,845 | 1.0× | -0.0029 | — | — |
| 2×PTCDA | **A_opt** | **35.4** | **28,258** | **15.3×** | -0.0029 | — | — |
| 2×PTCDA | A_eventless | 33.5 | 29,827 | 16.2× | -0.0029 | 0.000 | 0.000 |
| 2×PTCDA | A_niter_opt | 28.3 | 35,312 | 19.2× | -0.0030 | 0.009 | -0.00009 |
| 2×PTCDA | **A_predict** | **28.2** | **35,522** | **19.2×** | -0.0029 | **0.0002** | **+0.000003** |
| 2×PTCDA | C_persist | 31.0 | 32,251 | 17.5× | -0.0029 | 0.000 | 0.000 |
| 2×PTCDA | E_singlewg | 55.5 | 18,008 | 9.8× | -0.0029 | 0.000 | 0.000 |
| 4×PTCDA | A_naive | 592.2 | 1,689 | 1.0× | -0.1268 | — | — |
| 4×PTCDA | **A_opt** | **86.1** | **11,609** | **6.9×** | -0.1268 | — | — |
| 4×PTCDA | A_eventless | 84.2 | 11,871 | 7.0× | -0.1268 | 0.000 | 0.000 |
| 4×PTCDA | A_niter_opt | 79.7 | 12,547 | 7.4× | -0.1252 | 0.475 | +0.0017 |
| 4×PTCDA | A_predict | 79.1 | 12,635 | 7.5× | -0.1323 | 0.032 | -0.0055 |
| 4×PTCDA | C_persist | 81.9 | 12,206 | 7.2× | -0.1268 | 0.000 | 0.000 |
| 4×PTCDA | E_singlewg | 299.7 | 3,337 | 2.0× | -0.1268 | 0.000 | 0.000 |
| 8×PTCDA | A_naive | 655.6 | 1,525 | 1.0× | 0.1017 | — | — |
| 8×PTCDA | **A_opt** | **150.9** | **6,626** | **4.3×** | 0.1017 | — | — |
| 8×PTCDA | A_eventless | 149.6 | 6,683 | 4.4× | 0.1017 | 0.000 | 0.000 |
| 8×PTCDA | A_niter_opt | 145.9 | 6,854 | 4.5× | -0.4185 | 2.235 | -0.520 |
| 8×PTCDA | A_predict | 143.7 | 6,959 | 4.6× | 0.0460 | 0.853 | -0.056 |
| 8×PTCDA | C_persist | 147.9 | 6,760 | 4.4× | 0.1017 | 0.000 | 0.000 |
| 8×PTCDA | E_singlewg | 1,095.8 | 913 | 0.6× | 0.1017 | 0.000 | 0.000 |

### Where does the time go?

**Bare launch overhead** (niter=0, empty kernel): **6.5–7.1 µs/launch** (PyOpenCL), **4.8 µs** (eventless ctypes).

**Per-step time breakdown** (A_naive vs A_opt):

| Component | A_naive | A_opt |
|-----------|--------:|------:|
| Launcher cache miss (`generate_kernel_args` ×2 + `set_args` ×2) | ~500 µs | 0 (cached) |
| `finish()` (host block) | ~50 µs | 0 (only at end) |
| `enqueue_nd_range_kernel()` | ~7 µs | ~7 µs |
| **Kernel execution** | **varies** | **varies** |

**Kernel execution time per step** (derived from A_opt − bare overhead):

| Config | A_opt µs/step | Kernel µs/step | Overhead fraction |
|--------|--------------:|---------------:|------------------:|
| 1×PTCDA | 11.2 | 4.4 | 61% overhead |
| 2×PTCDA | 35.4 | 28.6 | 19% overhead |
| 4×PTCDA | 86.1 | 79.3 | 8% overhead |
| 8×PTCDA | 150.9 | 144.1 | 5% overhead |

**Key insight**: For 1 molecule, Python/launch overhead is 61% of total time. For 8 molecules, it's only 5%. This explains why A_opt gives 46× for 1 mol but only 4.3× for 8 mol.

### Key findings

1. **C_persist is now CORRECT** — the ping-pong fix eliminated the OpenCL C 1.2 memory visibility problem. RMSD=0, ΔE=0 for all tested configs. It is ~2% faster than A_opt (saves the kernel-boundary sync overhead). Still experimental because OpenCL does not formally guarantee all WGs are simultaneously resident.

2. **E_singlewg is now CORRECT** — changed from Gauss-Seidel to Jacobi (all forces from one snapshot → integrate all → barrier). RMSD=0, ΔE=0. But uses only 1 CU, so it's slower than A_opt for >2 molecules (1096 µs vs 151 µs for 8 mol).

3. **A_eventless saves ~2 µs/step**, bit-identical to A_opt. The ctypes direct enqueue bypasses PyOpenCL's cl_event wrapper. Useful for small systems where launch overhead dominates.

4. **A_niter_opt (K=10) gives 135× for 1 mol** (exact — no inter-mol forces), **19× for 2 mol** (RMSD=0.009 Å — excellent), but **diverges for 8 mol** (RMSD=2.2 Å, ΔE=-0.52 eV — unusable).

5. **A_predict dramatically improves accuracy over A_niter_opt**: 2 mol RMSD drops from 0.009 to 0.0002 Å; 4 mol from 0.475 to 0.032 Å. But 8 mol still has RMSD=0.85 Å — not acceptable. Prediction extrapolates partner poses using constant linear/angular velocity from the last step.

6. **A_opt (batch=1) gives 4.3–46× speedup** — the exact synchronous path. Speedup inversely correlates with kernel time (which grows as O(n_mol²)).

7. **D_stale (sequential) is catastrophically inaccurate** — only usable for sequential relaxation (one active molecule, others frozen).

### Why not 100× for large systems? (Root cause analysis)

We **do achieve 135×** for 1 molecule with `A_niter_opt(K=10)`. For multi-molecule systems, the speedup drops because:

1. **Kernel time grows as O(n_mol²)** — each molecule interacts with all others. For 8 mols: 144 µs/step kernel time vs 4.4 µs/step for 1 mol.
2. **Python overhead is constant** (~7 µs/launch) — becomes a smaller fraction of total time.
3. **batch>1 (stale/predicted) degrades accuracy** for many molecules — stale positions cause divergence.

The **only way to get 10×+ for 8 mol** is to reduce kernel time, not launch overhead:
- **Neighbor lists / cutoff** — skip far-away molecule pairs (currently all N² pairs computed)
- **Cell lists / spatial hashing** — O(N) instead of O(N²) pair finding
- **Larger timestep** — fewer steps needed (if stability allows)

### Recommendation

| System size | Best exact strategy | Speedup | Best approximate | Speedup | Approx accuracy |
|-------------|---------------------|--------:|------------------|--------:|-----------------|
| 1 mol | A_niter_opt (K=100) | **135×** | A_niter_opt (any K) | 135× | exact |
| 2 mol | A_opt or C_persist | 15–17× | A_predict (K=10) | 19× | RMSD=0.0002 Å |
| 4 mol | A_opt or C_persist | 6.9–7.2× | A_predict (K=10) | 7.5× | RMSD=0.032 Å |
| 8+ mol | A_opt or C_persist | 4.3–4.4× | — (not accurate enough) | — | — |

**For production**: use `run_multimol_md(batch=1)` (A_opt) for exact results. Use `run_multimol_md(batch=K, predict_partners=True)` for small systems (≤4 mol) where the approximation is acceptable. Use `run_multimol_persistent` only when you need maximum throughput on small systems and can accept the experimental residency assumption.

**For 10×+ on 8+ mol**: implement neighbor lists / cutoff in the kernel (separate task).

### Integration plan (next steps)

The ping-pong concurrent MD API is production-ready for PairFF (no FAF). To integrate into UFF/SPFF relaxation on surfaces:

1. **GridFF + UFF/SPFF**: The multimol kernel currently only supports PairFF (intermolecular compact-exp). For UFF/SPFF intramolecular relaxation on a substrate, the existing `run_pairff` path (single active molecule) remains correct. The multimol API is useful when multiple molecules move simultaneously on the substrate.

2. **FAF integration**: The concurrent APIs currently reject FAF. A fused multimol+FAF kernel would need to duplicate the FAF substrate evaluation per workgroup — this is a larger refactor, deliberately deferred.

3. **Rigid body manipulation**: The `RigidAssemblyExtension` GUI already uses `RigidBodyPairFF`. The new `run_multimol_md` API can be wired into the "Run" button for concurrent relaxation of all molecules, and `run_multimol_persistent` for maximum throughput on small ensembles.

4. **Neighbor lists**: The O(n_mol²) inter-molecule force loop is the bottleneck for 8+ molecules. A cell-list or Verlet-list approach in the kernel would reduce this to O(N), enabling 10×+ speedups for large ensembles.

### Files

| File | Purpose |
|------|---------|
| `kernels/rigid.cl` lines 4085–4912 | Three kernel variants (15: multimol, 16: persistent, 17: single-WG) |
| `spammm/forcefields/RigidBodyDynamics.py` lines 2370–2461 | Production API: `run_multimol_md`, `run_multimol_single_wg`, `run_multimol_persistent` |
| `spammm/utils/OpenCLBase.py` lines 219–246 | `bind_ndrange(eventless=True)` — ctypes direct enqueue |
| `tests/bench_multimol_md.py` | Standalone benchmark script (all strategies, K-sweep) |
| `tests/test_forcefield.py` lines 176–202 | L0 parity test (`test_pairff_multimol_launch_parity`) |
| `doc/Tasks/MultiMol_MD_LaunchOverhead.md` | This document |

### How to reproduce

```bash
# L0 parity test (fast — verifies correctness of all variants)
pytest tests/test_forcefield.py::test_pairff_multimol_launch_parity -x -s

# Full benchmark (all strategies, 1-8 mols, 100 steps, K=10)
python3 tests/bench_multimol_md.py --mol PTCDA --nmol 1 2 4 8 --steps 100 --strategies A_naive,A_opt,A_eventless,A_niter_opt,A_predict,C_persist,E_singlewg --K 10

# Small systems only (shows 15-135× speedup)
python3 tests/bench_multimol_md.py --mol PTCDA --nmol 1 2 --steps 100 --K 10

# K-sweep (accuracy vs speed tradeoff for approximate strategies)
python3 tests/bench_multimol_md.py --mol PTCDA --nmol 2 --steps 100 --strategies A_niter_opt,A_predict --K 100
```

### Remaining caveats

- **`run_multimol_persistent` is experimental**: the residency check (`n_mols ≤ max_compute_units`) is conservative but OpenCL does not guarantee simultaneous workgroup residency. Deadlock is possible if the scheduler doesn't place all WGs at once. Verified working on GTX 1650 (14 CUs) for ≤8 molecules.
- **`run_multimol_single_wg` is exact but slow** for >2 molecules — one-CU execution loses to multi-WG parallelism. Useful only for very small ensembles (1-2 molecules) where it avoids kernel-boundary overhead.
- **Concurrent all-mobile FAF is not implemented** — the three concurrent APIs fail loud when FAF is enabled. A fused multimol+FAF variant needs a dedicated kernel or larger refactor.
- **`A_predict` (constant-velocity extrapolation)** is experimental and useful only with an accuracy gate. K=10 is acceptable for ≤4 molecules but not for 8+.
- **`A_niter_opt` (stale positions)** diverges for 8+ molecules at K=10. Use `A_predict` instead, or stick with `batch=1`.
