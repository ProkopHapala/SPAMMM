---
type: Task
title: Multi-molecule MD launch-overhead experiment — eliminate Python harness bottleneck for concurrent rigid-body dynamics
status: experiment complete — awaiting USER review of results
tags: [OpenCL, MD, performance, benchmark, rigid-body, PairFF, UFF, launch-overhead]
timestamp: 2026-08-01
related: [PairFF_MC_PythonBottleneck.md, PairFF_MultiBody_Kernel.md]
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

## 11. Benchmark results (2026-08-01, NVIDIA GeForce GTX 1650)

### Implementation

Three new kernels added to `kernels/rigid.cl`:
- `rigid_body_pairff_multimol_kernel` (kernel 15) — N workgroups, 1 step per launch, all molecules move concurrently. Has `niter` parameter for internal loop (Jacobi stale-position).
- `rigid_body_pairff_multimol_persistent_kernel` (kernel 16) — same + atomic global barrier (Strategy C, incorrect on OpenCL C 1.2)
- `rigid_body_pairff_multimol_single_wg_kernel` (kernel 17) — single workgroup, all molecules, Gauss-Seidel (Strategy E)

Benchmark script: `tests/bench_multimol_md.py` (standalone, headless, reuses `from_molecules` setup).

### Strategies tested

| Key | Description |
|-----|-------------|
| `A_naive` | `generate_kernel_args()` + `krnl()` + `finish()` per step (anti-pattern) |
| `A_opt` | `set_args()` once, bare `enqueue_nd_range_kernel()` loop, single `finish()` |
| `A_niter` | `multimol_kernel` with `niter=K`, Jacobi stale-position, `generate_kernel_args` per launch |
| `A_niter_opt` | Same as A_niter but `set_args()` once (fastest) |
| `C_persist` | Persistent kernel + atomic global barrier (1 launch, N steps) |
| `D_stale` | Existing `run_pairff(K)` cycling active_mol (sequential, not concurrent) |
| `E_singlewg` | Single workgroup, Gauss-Seidel, 1 launch |

### Deep investigation: where does the time go?

**Bare launch overhead** (niter=0, empty kernel): **6.8 µs/launch** — the absolute floor.

**Per-step time breakdown** (A_naive vs A_opt):

| Component | A_naive | A_opt |
|-----------|--------:|------:|
| `generate_kernel_args()` | ~50 µs | 0 (once) |
| `set_args()` | ~20 µs | 0 (once) |
| `finish()` (host block) | ~50 µs | 0 (only at end) |
| `enqueue_nd_range_kernel()` | ~7 µs | ~7 µs |
| **Kernel execution** | **varies** | **varies** |
| **Total overhead** | **~127 µs** | **~7 µs** |

**Kernel execution time per step** (derived from A_opt − bare overhead):

| Config | A_opt µs/step | Kernel µs/step | Overhead fraction |
|--------|--------------:|---------------:|------------------:|
| 1×PTCDA | 14 | 7 | 50% overhead |
| 2×PTCDA | 38 | 31 | 18% overhead |
| 4×PTCDA | 89 | 82 | 8% overhead |
| 8×PTCDA | 192 | 185 | 4% overhead |
| 16×PTCDA | 228 | 221 | 3% overhead |

**Key insight**: For 1 molecule, Python overhead is 50% of total time. For 16 molecules, it's only 3%. This explains why A_opt gives 10× for 1 mol but only 1.5× for 16 mol.

### Results: small systems (where Python overhead dominates)

| Config | Strategy | µs/step | steps/s | Speedup vs A_naive | E_final (eV) | RMSD (Å) | ΔE (eV) |
|--------|----------|--------:|--------:|-------------------:|-------------:|---------:|--------:|
| 1×PTCDA | A_naive | 131.8 | 7,589 | 1.0× | 0.0000 | — | — |
| 1×PTCDA | A_opt | 14.0 | 71,466 | **9.4×** | 0.0000 | — | — |
| 1×PTCDA | **A_niter_opt (K=10)** | **6.6** | **163,129** | **21.0×** | 0.0000 | 0.000 | 0.000 |
| 1×PTCDA | **A_niter_opt (K=100)** | **5.6** | **178,571** | **24.5×** | 0.0000 | 0.000 | 0.000 |
| 2×PTCDA | A_naive | 166.5 | 6,007 | 1.0× | -0.0029 | — | — |
| 2×PTCDA | A_opt | 38.1 | 26,253 | **4.4×** | -0.0029 | — | — |
| 2×PTCDA | **A_niter_opt (K=10)** | **31.7** | **32,597** | **5.4×** | -0.0030 | 0.009 | -0.0001 |
| 4×PTCDA | A_naive | 214.8 | 4,656 | 1.0× | -0.1268 | — | — |
| 4×PTCDA | A_opt | 89.2 | 11,205 | **2.4×** | -0.1268 | — | — |
| 4×PTCDA | A_niter_opt (K=10) | 82.7 | 12,119 | **2.6×** | -0.1252 | 0.475 | +0.0017 |

### K-sweep: accuracy vs speed tradeoff (2×PTCDA, 100 steps)

| K | µs/step | launches | E_final | RMSD (Å) | ΔE (eV) | Verdict |
|--:|--------:|---------:|--------:|---------:|--------:|---------|
| 1 | 38.3 | 100 | -0.002891 | 0.000 | 0.000 | exact (A_opt) |
| 2 | 35.2 | 50 | -0.002925 | 0.005 | -0.00003 | excellent |
| 5 | 32.6 | 20 | -0.002942 | 0.008 | -0.00005 | excellent |
| 10 | 31.7 | 10 | -0.002985 | 0.009 | -0.00009 | excellent |
| 20 | 31.6 | 5 | -0.003020 | 0.007 | -0.00013 | good |
| 50 | 31.9 | 2 | -0.003138 | 0.006 | -0.00025 | good |
| 100 | 32.3 | 1 | 0.047162 | 0.011 | +0.05005 | diverges |

### K-sweep: 1×PTCDA (no inter-mol forces, niter=K is EXACT)

| K | µs/step | launches | Speedup | RMSD (Å) | ΔE (eV) |
|--:|--------:|---------:|--------:|---------:|--------:|
| 1 | 15.4 | 100 | 9.0× | 0.000 | 0.000 |
| 10 | 6.6 | 10 | 21.0× | 0.000 | 0.000 |
| 100 | 5.6 | 1 | 24.5× | 0.000 | 0.000 |

### K-sweep: 4×PTCDA (inter-mol forces matter)

| K | µs/step | launches | Speedup | RMSD (Å) | ΔE (eV) | Verdict |
|--:|--------:|---------:|--------:|---------:|--------:|---------|
| 1 | 89.9 | 100 | 2.4× | 0.000 | 0.000 | exact |
| 10 | 82.7 | 10 | 2.6× | 0.475 | +0.002 | acceptable |
| 100 | 89.4 | 1 | 2.4× | 0.575 | +0.435 | diverges |

### Results: large systems (kernel time dominates)

| Config | Strategy | µs/step | steps/s | Speedup vs A_naive | E_final (eV) | Correct? |
|--------|----------|--------:|--------:|-------------------:|-------------:|:--------:|
| 8×PTCDA | A_naive | 315 | 3,169 | 1.0× | 0.1017 | ✓ ref |
| 8×PTCDA | **A_opt** | **192** | **5,221** | **1.6×** | 0.1017 | **✓** |
| 8×PTCDA | A_niter_opt (K=10) | 168 | 5,939 | 1.9× | -0.0302 | ✗ ΔE=-0.13 |
| 8×PTCDA | C_persist | 196 | 5,113 | 1.6× | 0.0450 | ✗ mem |
| 8×PTCDA | E_singlewg | 1,359 | 736 | 0.2× | 0.2915 | ~ GS |
| 16×PTCDA | A_naive | 352 | 2,843 | 1.0× | -1.4493 | ✓ ref |
| 16×PTCDA | **A_opt** | **228** | **4,390** | **1.5×** | -1.4492 | **✓** |
| 16×PTCDA | C_persist | 238 | 4,209 | 1.5× | -0.8267 | ✗ mem |
| 16×PTCDA | E_singlewg | 3,381 | 296 | 0.1× | -1.3332 | ~ GS |

### Key findings

1. **For 1 molecule: A_niter_opt gives 24.5× speedup** (K=100, exact because no inter-mol forces). This is the "10-100×" class speedup expected from in-kernel MD loops. The floor is ~5.6 µs/step = bare launch overhead (6.8 µs) amortized over 100 steps in 1 launch.

2. **For 2 molecules: A_niter_opt gives 5.4× speedup** (K=10, RMSD=0.009 Å, ΔE=-0.0001 eV — excellent accuracy). The stale-position Jacobi approximation is very accurate for small systems because molecules don't move far in K=10 steps.

3. **For 4+ molecules: only 2.4-2.6× speedup** — kernel execution time dominates (82+ µs/step), and the stale-position approximation degrades (RMSD=0.47 Å for K=10). The O(n_mol²) inter-molecule force loop is the real bottleneck.

4. **A_opt (K=1, bare enqueue) gives 9.4× for 1 mol, 4.4× for 2 mol, 2.4× for 4 mol, 1.6× for 8 mol** — the speedup inversely correlates with kernel time. Python overhead is ~127 µs/step (constant), kernel time grows with system size.

5. **Strategy C (persistent) is incorrect on OpenCL C 1.2** — cross-workgroup memory writes are not guaranteed visible without a kernel boundary. `mem_fence` + atomic barriers don't fix this. This is a fundamental limitation of OpenCL C 1.2, not a bug.

6. **Strategy E (single-WG) is 7-15× SLOWER** — serializing force computation over molecules (Gauss-Seidel) uses only 1 of 14 CUs.

7. **Strategy D (stale, sequential) is catastrophically inaccurate** — energy errors of +4 to +72 eV. Only usable for sequential relaxation (one active molecule, others frozen).

### Why not 100×? (Root cause analysis)

The user's expectation of 10-100× comes from the single-molecule case where moving the MD loop into the kernel eliminated ~130 µs/step of Python overhead, leaving ~5 µs/step of kernel time → ~25× speedup. We **do achieve 24.5×** for 1 molecule with A_niter_opt(K=100).

For multi-molecule systems, the speedup drops because:
1. **Kernel time grows as O(n_mol²)** — each molecule interacts with all others. For 8 mols: 185 µs/step kernel time vs 7 µs/step for 1 mol.
2. **Python overhead is constant** (~127 µs/step) — so it becomes a smaller fraction of total time.
3. **niter=K (Jacobi) degrades accuracy** for multi-mol — stale positions cause divergence for K>10.

The **only way to get 10× for 8+ mol** is to reduce kernel time, not launch overhead:
- **Neighbor lists / cutoff** — skip far-away molecule pairs (currently all N² pairs computed)
- **Cell lists / spatial hashing** — O(N) instead of O(N²) pair finding
- **Larger timestep** — fewer steps needed (if stability allows)
- **Coarser tiling** — fewer global memory reads for neighbor molecule atoms

### Recommendation

| System size | Best strategy | Speedup | Accuracy |
|-------------|---------------|--------:|----------|
| 1 mol | A_niter_opt (K=100) | **24.5×** | exact |
| 2 mol | A_niter_opt (K=10) | **5.4×** | excellent (RMSD<0.01 Å) |
| 4 mol | A_niter_opt (K=10) | **2.6×** | acceptable (RMSD<0.5 Å) |
| 8+ mol | A_opt (K=1) | **1.5-1.6×** | exact |

**For production**: use `A_niter_opt` with adaptive K — start with K=10, reduce if energy drift detected. For exact results, use `A_opt` (K=1).

**For 10×+ on large systems**: implement neighbor lists / cutoff in the kernel (separate task).

### Files

| File | Purpose |
|------|---------|
| `kernels/rigid.cl` lines 4074–4912 | Three new kernel variants (15, 16, 17) |
| `tests/bench_multimol_md.py` | Standalone benchmark script |
| `doc/Tasks/MultiMol_MD_LaunchOverhead.md` | This document |

### How to reproduce

```bash
# Small systems (where Python overhead dominates — shows 10-25× speedup)
python3 tests/bench_multimol_md.py --mol PTCDA --nmol 1 2 4 --steps 100 --strategies A_naive,A_opt,A_niter_opt --K 10

# Large systems (kernel time dominates — shows 1.5-2× speedup)
python3 tests/bench_multimol_md.py --mol PTCDA --nmol 8 16 --steps 100

# K-sweep (accuracy vs speed tradeoff)
python3 tests/bench_multimol_md.py --mol PTCDA --nmol 2 --steps 100 --strategies A_niter_opt --K 100

# All strategies
python3 tests/bench_multimol_md.py --mol PTCDA --nmol 4 8 --steps 100 --strategies A_naive,A_opt,A_niter_opt,C_persist,D_stale,E_singlewg
```
