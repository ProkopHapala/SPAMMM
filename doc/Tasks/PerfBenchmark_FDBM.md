# Performance Benchmark: AFM FDBM Pipeline

**Goal:** End-to-end FDBM AFM image generation < 1 second for small molecules (benzene, H2O, coronene).

**Status:** Investigation done (2026-07-19). Instrumentation and `bench_fdbm.py` **not implemented yet**. Plan below remains the implementation target.

---

## GUI: which AFM engine?

**SPAMMM GUI uses FDBM only** — not Morse+Coulomb (point charges).

| Path | Engine | Used by GUI? |
|------|--------|--------------|
| `ModularAFMPipeline` → FDBM densities + Pauli/ES/vdW grids + PP relax | FDBM | **Yes** — `spammm/GUI/AFMExtension.py` |
| `AFMulator(use_morse=True/False)` + `make_forcefield()` / LJ|Morse+Q | Morse/LJ + point charges | **No** — tests/scripts only (`tests/SPM/test_afm_morse.py`, `testplot_afm_morse.py`) |

Evidence:
- `AFMExtension._ensure_pipeline()` constructs `ModularAFMPipeline(...)` (FDBM staged pipeline).
- Stage buttons / tooltips say “FDBM potentials”; no Morse/LJ toggle in the AFM panel.
- Inside the pipeline, `AFMulator(use_morse=False)` is only used as a **GPU helper** for `compute_gradient_cl` and FDBM scan/relax — not for building Morse/LJ forcefields.
- ExtensionManager: `'afm'` → `spammm.GUI.AFMExtension`, enabled by default.

`AFM.py` supports both engines; GUI wires only the FDBM modular path.

---

## Pipeline map (user steps ↔ code stages)

User’s 4 steps miss density projection and fold gradients into “build potential”. Actual GUI/pipeline partition:

| User step | Pipeline stage | What actually runs | Gradients? |
|-----------|----------------|--------------------|------------|
| 1) DFTB+ calculation | **S1** `stage1_scf()` | DFTBcore SCF → `dm_dense`, eigvecs/eigvals; `cache_stage1_scf.npz` | — |
| *(missing in user list)* | **S2** `stage2_project()` | GPU density grids: `rho_scf`, `rho_na`, `rho_diff`; `cache_stage2_grids.npz` | — |
| 2) building potential (Pauli, Coulomb, vdW) | **S3** first half | Poisson `V_ES`, CO tip load/pad, Pauli overlap+scale, ES convolution, dispersion → `E_*` fields | energy only (`return_grads=False`) |
| 3) computing gradients | **S3** second half | `AFMulator.compute_gradient_cl(E_total, step)` → `F_total`; then cache | **Yes — required** |
| 4) probe-particle relaxation | **S4** `stage4_relax()` | `compose_and_relax_total` + `scan_fdbm` → `df`, tip_disp, FEs; `cache_stage4_relax.npz` | uses `F_total` |
| (GUI extras) | **S5/S6** | STM / BR-STM from eigvecs + tip_disp | not in FDBM AFM timing target |

### Are gradients needed?

**Yes.** Stage 3 deliberately builds scalar energy fields with `return_grads=False`, then converts `E_total → F_total` once via GPU finite-difference/kernel gradient. Stage 4 PP MD needs forces on the tip, not energies alone. Timing must treat gradient as its own sub-segment inside S3 (or as a named S3b), not leave it unaccounted.

### Coverage rule for benchmarking

Every wall-clock second from “Run AFM” / CLI entry to `df` ready must sit inside a timed segment. Current untimed gaps to instrument:

1. Pipeline `__init__` / geometry+grid setup / DFTB projector init  
2. Cache load vs recompute branches  
3. CO tip load / pad / on-the-fly compute (often hidden inside S3)  
4. Host↔device copies and `np.savez_compressed` I/O  
5. AFMulator construction + OpenCL compile (can dominate cold runs)  
6. S5/S6 if measuring “full GUI Run”, else exclude explicitly from FDBM bench

---

## How speed benchmarking works **today** (investigation)

### Verdict

| Layer | Status |
|-------|--------|
| End-to-end FDBM stage timing | **None** — `ModularPipeline.py` has **zero** `time` / `perf_counter` / `[BENCH]` |
| Dedicated bench script | **Missing** — `tests/SPM/bench_fdbm.py` planned in this doc, does not exist |
| Python profiler (`cProfile` / line_profiler) on AFM path | **Not used** |
| OpenCL event profiling (`PROFILING_ENABLE`) on AFM | **Not used** |
| Ad-hoc stopwatches elsewhere | **Yes**, sparse — see below |
| Formal protocol | Documented in `doc/AGENTS/protocols/general/performance_optimization.md` (warmup, median, `queue.finish()`, event profiling) — **not applied to FDBM yet** |

**Gating today:** nothing gates FDBM perf. No CI timing assert, no bench harness. Perf work is “Not started” in `doc/TASKS.md` T01. Functional tests (`tests/SPM/test_afm_fdbm.py`) check correctness, not speed.

### What exists (manual stopwatches, not a unified system)

1. **`Grid_dftb.py`** — fine-grained GPU wall-clock around kernels:
   - `time.perf_counter_ns()` + `queue.finish()` then print `[TIME] … [ms]`
   - Also coarser `time.time()` around orbital loops in `project_dftb_density` / `project_neutral_density`
   - Useful for S2 GPU internals; not rolled up to ModularPipeline stages

2. **`AFM_utils.get_density_from_pyscf`** — coarse `time.time()` prints for SCF / density eval / total (pySCF backend only; GUI default is DFTB)

3. **`AFM_utils` fitting/compare helpers** — wall-clock around whole FDBM-generate + fit jobs (minutes-scale scripts), not per-stage S1–S4

4. **`FFExtension.py` (GUI FF relax, not AFM)** — `time.time()` around build/step/relax — pattern to copy, different subsystem

5. **Protocol doc** recommends:
   - Wall-clock with warmup + median/min/max  
   - Separate GPU kernel time via OpenCL events (not only wall-clock)  
   - `queue.finish()` before stopping the timer for GPU work  

So today = **scattered manual stopwatches inside lower-level kernels**, plus a **written plan** for stage wrappers. No profiler-driven AFM benchmark, no single table covering S1–S4.

### Recommended dual approach (to implement)

| Tool | Role |
|------|------|
| **Manual segmented stopwatch** (`time.perf_counter` + `queue.finish()`) | Primary SSOT for stage/sub-op table; always on in `bench_fdbm.py` and optional `SPAMMM_AFM_BENCH=1` in ModularPipeline |
| **`cProfile` / `pyinstrument` (optional flag)** | Find unexpected Python hotspots *outside* named segments; one cold + one warm run |
| **OpenCL event profiling** (optional, deeper) | Split GPU kernel vs host for S2/S3/S4 once wall-clock shows which stage dominates |

---

## Pipeline stages (ModularPipeline.py S1–S6)

| Stage | What | Key code | Suspected bottleneck |
|-------|------|----------|---------------------|
| S1: SCF | DFTB+ SCF or pySCF | `ModularPipeline.stage1_scf()` | DFTB+ itself is fast (~0.05s for coronene). pySCF unknown. Python overhead in setup/teardown? |
| S2: Density projection | GPU density grid projection | `stage2_project()` → `Grid_dftb` | GPU kernel itself may be fine; Python orchestration / projector setup / host copies suspected. Partial `[TIME]` prints already in Grid_dftb |
| S3a: Potentials | Pauli (FFT), Poisson (FFT), vdW | `stage3_potentials()` energy part | FFT convolution; CO tip I/O; Python-side array copies; `np.savez` |
| S3b: Gradients | `E_total → F_total` | `compute_gradient_cl()` | Must be timed separately; required for S4 |
| S4: PP relaxation | Probe-particle MD on GPU | `stage4_relax()` | GPU kernel should be fast. Per-step `queue.finish()`? Double work: `compose_and_relax_total` then another `scan_fdbm` |
| S5: df / STM | Frequency shift / LDOS | `stage5_*` | Negligible for FDBM AFM target; exclude from <1s criterion unless measuring full GUI |
| S6: BR-STM / viz | Bond-resolved STM / plots | `stage6_*` | Visual only |

## Suspected Python-side overheads

1. **Host-device transfers between stages**: Each stage may copy arrays GPU→CPU→GPU unnecessarily. Check if `rho_scf`, `V_ES`, `F_total` stay on GPU or round-trip through numpy.
2. **Cache I/O**: `np.savez` / `np.load` for each stage — disk writes add latency even when cache is hit.
3. **Redundant recomputation**: `force_recompute` flags may trigger unnecessary re-runs. Dirty flag system (S1–S6) may not track dependencies correctly.
4. **Grid setup per call**: projector / AFMulator may rebuild (compile kernels, allocate buffers) every call instead of caching across stages.
5. **OpenCL kernel compilation**: Kernels may recompile per session instead of being cached via `pyopencl.cache_dir`.
6. **S4 double scan**: `compose_and_relax_total` then a second `scan_fdbm` for `FEs_relax` — likely duplicate GPU work; confirm and time both.

---

## Benchmarking plan

### Step 1: Per-stage timing instrumentation

Add timing wrappers to each `stageN_*()` method in `ModularPipeline.py`, and **sub-timers** so nothing is outside a bucket:

```python
import time
t0 = time.perf_counter()
# ... stage / sub-op ...
# if GPU: queue.finish()
t1 = time.perf_counter()
print(f"[BENCH] Stage {N}/{label}: {t1-t0:.4f}s")
```

Required segments (cold and warm / cache-hit runs):

| Segment | Sub-ops to time |
|---------|-----------------|
| init | geometry, grids, DFTB projector / basis load |
| S1 | SCF only; separately cache write |
| S2 | projector setup, `project_density_dense`, `project_neutral_density`, rho_diff, cache write |
| S3a | Poisson, CO tip get, Pauli, ES conv, vdW, E_total sum |
| S3b | AFMulator ctor (if cold), `compute_gradient_cl`, cache write |
| S4 | upload/setup FDBM grid, `compose_and_relax_total`, second `scan_fdbm` if kept, cache write |
| I/O | each `np.load` / `np.savez_compressed` |

Gate prints with env `SPAMMM_AFM_BENCH=1` so GUI stays quiet by default.

### Step 2: End-to-end benchmark script

`tests/SPM/bench_fdbm.py` — CLI script, not pytest:

```bash
python tests/SPM/bench_fdbm.py --mol data/xyz/benzene.xyz --repeats 5
python tests/SPM/bench_fdbm.py --mol ... --profile   # optional cProfile dump
```

Output: per-stage timing table + total. Run with and without cache. Report median/min/max (see performance protocol).

### Step 3: Identify bottlenecks

For each stage, separate:
- **GPU kernel time**: `queue.finish()` + wall-clock around kernel calls; later OpenCL events
- **Python overhead**: total stage time − GPU kernel time
- **I/O time**: cache save/load time

### Step 4: Optimization targets

- Move host-device transfers out of hot loops
- Pre-allocate and reuse GPU buffers across stages
- Cache OpenCL kernel compilations / reuse one AFMulator across S3–S4
- Skip cache I/O when running end-to-end (in-memory pipeline)
- Eliminate duplicate S4 scan if confirmed redundant

## Test molecules

| Molecule | Atoms | Expected time | Notes |
|----------|-------|---------------|-------|
| H2O | 3 | < 0.1s | Minimal |
| Benzene | 12 | < 0.3s | Small aromatic |
| Coronene | 24 | < 0.5s | Medium PAH |
| Pentacene | 36 | < 1.0s | Target: < 1s |

## Success criteria

- End-to-end FDBM (S1→S5 df, or S1→S4) for benzene: < 0.3s
- End-to-end FDBM for pentacene: < 1.0s
- No stage exceeds 50% of total time (balanced pipeline)
- Python overhead < 30% of total time
- Timed segments sum to ≥ 95% of end-to-end wall time (coverage check)

## References

- `spammm/GUI/AFMExtension.py` — GUI wiring (FDBM ModularPipeline only)
- `spammm/SPM/ModularPipeline.py` — staged pipeline (no stage timers yet)
- `spammm/SPM/AFM.py` — AFMulator core (FDBM + Morse/LJ engines)
- `spammm/SPM/AFM_utils.py` — high-level orchestration; sparse pySCF timings
- `spammm/quantum/DFTB/Grid_dftb.py` — existing `[TIME]` kernel stopwatches (S2)
- `tests/SPM/test_afm_fdbm.py` — existing FDBM tests (functional, not perf)
- `tests/SPM/test_afm_morse.py` — Morse+point-charge path (not GUI)
- `doc/AGENTS/protocols/general/performance_optimization.md` — warmup/median/event profiling
- `doc/ARCHITECTURE_ROADMAP.md` §2 (pySCF backend)
