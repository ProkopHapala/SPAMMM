# Performance Benchmark: AFM FDBM Pipeline

**Goal:** End-to-end FDBM AFM image generation < 1 second for small molecules (benzene, H2O, coronene).

**Status:** Not started. This document defines the benchmarking plan and identifies suspected bottlenecks.

---

## Pipeline stages (ModularPipeline.py S1–S6)

| Stage | What | Key code | Suspected bottleneck |
|-------|------|----------|---------------------|
| S1: SCF | DFTB+ SCF or pySCF | `ModularPipeline.stage1_scf()` | DFTB+ itself is fast (~0.05s for coronene). pySCF unknown. Python overhead in setup/teardown? |
| S2: Density projection | GPU density grid projection | `stage2_project()` → `Grid_dftb` | **ModuleNotFoundError: `spammm.DFTB`** — import broken. GPU kernel itself may be fine, but Python orchestration (setup_gridprojector_from_dftb) suspected slow |
| S3: Potentials | Pauli (FFT), Poisson (FFT), vdW | `stage3_potentials()` | FFT convolution on GPU — but Python-side array copies? np.savez cache I/O? |
| S4: PP relaxation | Probe-particle MD on GPU | `stage4_relax()` | GPU kernel should be fast. Per-step queue.finish()? |
| S5: df conversion | Frequency shift from Fz | `stage5_df()` | Likely negligible (pure numpy) |
| S6: Visualization | 2D image slices | `stage6_visual()` | Only for visual mode, not perf-critical |

## Suspected Python-side overheads

1. **Host-device transfers between stages**: Each stage may copy arrays GPU→CPU→GPU unnecessarily. Check if `rho_scf`, `V_ES`, `F_total` stay on GPU or round-trip through numpy.
2. **Cache I/O**: `np.savez` / `np.load` for each stage — disk writes add latency even when cache is hit.
3. **Redundant recomputation**: `force_recompute` flags may trigger unnecessary re-runs. Dirty flag system (S1–S6) may not track dependencies correctly.
4. **Grid setup per call**: `setup_gridprojector_from_dftb()` may rebuild projector (compile kernels, allocate buffers) every call instead of caching.
5. **OpenCL kernel compilation**: Kernels may recompile per session instead of being cached via `pyopencl.cache_dir`.
6. **Import errors**: `spammm.DFTB` module not found (seen in traceback) — pipeline crashes at S2.

## Benchmarking plan

### Step 1: Per-stage timing instrumentation

Add timing wrappers to each `stageN_*()` method in `ModularPipeline.py`:

```python
import time
t0 = time.time()
# ... stage code ...
t1 = time.time()
print(f"[BENCH] Stage {N}: {t1-t0:.4f}s")
```

Also time sub-operations within each stage (kernel launch, host-device transfer, cache I/O).

### Step 2: End-to-end benchmark script

`tests/SPM/bench_fdbm.py` — CLI script, not pytest:

```bash
python tests/SPM/bench_fdbm.py --mol data/xyz/benzene.xyz --repeats 5
```

Output: per-stage timing table + total. Run with and without cache.

### Step 3: Identify bottlenecks

For each stage, separate:
- **GPU kernel time**: `cl_queue.finish()` + `time.time()` around kernel calls
- **Python overhead**: total stage time − GPU kernel time
- **I/O time**: cache save/load time

### Step 4: Optimization targets

- Move host-device transfers out of hot loops
- Pre-allocate and reuse GPU buffers across stages
- Cache OpenCL kernel compilations
- Skip cache I/O when running end-to-end (in-memory pipeline)
- Fix `spammm.DFTB` import error

## Test molecules

| Molecule | Atoms | Expected time | Notes |
|----------|-------|---------------|-------|
| H2O | 3 | < 0.1s | Minimal |
| Benzene | 12 | < 0.3s | Small aromatic |
| Coronene | 24 | < 0.5s | Medium PAH |
| Pentacene | 36 | < 1.0s | Target: < 1s |

## Success criteria

- End-to-end FDBM (S1→S5) for benzene: < 0.3s
- End-to-end FDBM (S1→S5) for pentacene: < 1.0s
- No stage exceeds 50% of total time (balanced pipeline)
- Python overhead < 30% of total time

## References

- `spammm/SPM/ModularPipeline.py` — staged pipeline
- `spammm/SPM/AFM.py` — AFMulator core
- `spammm/SPM/AFM_utils.py` — high-level orchestration
- `tests/SPM/test_afm_fdbm.py` — existing FDBM tests (functional, not perf)
- `doc/ARCHITECTURE_ROADMAP.md` §2 (pySCF backend)
