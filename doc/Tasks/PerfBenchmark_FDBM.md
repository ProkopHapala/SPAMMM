# Performance Benchmark: AFM FDBM Pipeline

**Goal:** End-to-end FDBM AFM image generation interactive (~0.1 s).

**Status (2026-07-19):** Round-1 + Round-2 **shipped and measured**. Benzene warm ~**0.18 s**; large-mol (flat_1) warm S3+S4 ~**1.4 s**. Fast S3 default (`SPAMMM_AFM_FAST_S3=1`); legacy via `=0`.  
`AFMBench` SSOT: `spammm/SPM/AFM.py` (not `globals.py`).

**How to re-run:**
```bash
# Quiet summary tables (default SPAMMM_AFM_BENCH=1); live >>/<< with =2
SPAMMM_AFM_BENCH=1 SPAMMM_AFM_BENCH_NO_IO=1 SPAMMM_VERBOSITY=0 AFM_DEBUG_PLOT_LEVEL=0 \
  python tests/SPM/bench_fdbm.py --mol data/xyz/benzene.xyz --tip co --step 0.1 --scan-step 0.1 --repeats 2
# Legacy S3 for comparison:
#   SPAMMM_AFM_FAST_S3=0 ...
```

---

## Executive summary — speedups (2026-07-19)

| Case | Before | After | Approx. factor |
|------|--------|-------|----------------|
| Benzene warm E2E (GUI-like, tip=co) | ~1.65 s | ~**0.18 s** (R2) | **~9×** |
| Benzene S3 field build | ~0.26 s | ~**0.07 s** | **~4×** |
| Flat_1 S2 NA density | 5.87 s | **0.03 s** | **~200×** |
| Flat_1 S3 cache I/O | ~10 s compressed | ~**0.4 s** uncompressed | **~25×** |
| Flat_1 warm S3+S4 (NO_IO) | many s / GUI stuck | **~1.4 s** | order-of-magnitude |

**Round 1:** GPU density tasks + gpyFFT + dense NA + shared AFMulator + drop duplicate scan + `np.savez`.  
**Round 2:** fused ES (`(ρ_diff·ρ_tip)/k²`) + GPU pad/roll + GPU Pauli scale + stay-on-GPU compose/grad (`fdbm_*` kernels). Pauli remains a separate overlap FFT (no `1/k²`).

---

## Speedup report — detail

### Baseline (before Round 1) → After

| Case | Before | After | How |
|------|--------|-------|-----|
| Benzene GUI-like warm (step=0.1 tip=co) | ~**1.65 s** | ~**0.55 s** (earlier GPU wiring) | GPU `build_tasks` + gpyFFT |
| Flat_1 (96 atoms) S2 `rho_na` | **5.87 s** | **0.03 s** | Diagonal NA DM → one `project_density_dense` (was per-AO loop) |
| Flat_1 Stage 3 wall (GUI, with cache) | ~**11–12 s** felt | ~**1.6–1.8 s** | Kill `savez_compressed` (was **~10 s** alone) + GPU k-mul + one AFMulator |
| Flat_1 warm S3+S4 NO_IO | (not measured cleanly) | **~1.4 s** | Same + no duplicate S4 scan |

**Flat_1 headless (RTX 3090, 336×256×144, tip=co, `SPAMMM_AFM_BENCH_NO_IO=1`):**

| Mode | Wall |
|------|------|
| COLD S1–S4 | **1.86 s** |
| WARM S2 | **0.08 s** |
| WARM S3+S4 | **1.41 s** |
| WARM S3 + uncompressed cache | **1.57 s** (cache write **0.41 s**) |

**Warm S3 remaining after Round 1 (NO_IO, pre–Round 2) — for history:**

| Segment | sec | Note |
|---------|-----|------|
| Pauli FFT | 0.22 | host↔GPU field copies |
| ES FFT | 0.22 | same |
| `compute_gradient_cl` | 0.20 | RGBA pack + upload/download |
| Poisson FFT | 0.14 | same |
| `pauli_scale` | 0.11 | **CPU** — eliminated in R2 |
| Tip pad/roll | 0.10 | **CPU** — eliminated in R2 |
| Dispersion | 0.09 | GPU |
| `E_total` sum | 0.06 | **CPU** — eliminated in R2 |
| S4 compose/relax | 0.24 | GPU |

After Round 2, benzene S3 collapses to a single `S3.fast_fields_gpu` ~**0.07 s** (see table above).

### Round-1 changes (already in tree — commit candidates)

1. GPU `build_tasks` default (`SPAMMM_AFM_CPU_TASKS=1` backup)
2. gpyFFT Poisson/Pauli/ES default; **k-space multiply on device** (no `.get()/.set()` mid-FFT)
3. `project_neutral_density` → dense diagonal NA DM (`SPAMMM_AFM_NA_ORBITAL_LOOP=1` backup)
4. One shared `AFMulator` S3–S4; drop duplicate `scan_fdbm`
5. Stage cache: `np.savez` not `savez_compressed` (huge GUI win)
6. `AFMBench` in `AFM.py`; summary-only default (`=2` for live lines)

---

## TODO next (partially done Round 2 — 2026-07-19)

### T-next-1 — Kill remaining CPU ops — **DONE (switchable)**

- [x] GPU `pauli_scale` (`fdbm_scale_pauli_pow_f32`) fused after Pauli IFFT
- [x] GPU tip pad/roll (`fdbm_pad_roll_f32`); raw tip via `get_tip_densities(..., pad_mode='none')`
- [x] `E_total` compose on GPU (`fdbm_compose_E_to_img`)
- Legacy path kept: `SPAMMM_AFM_FAST_S3=0`

### T-next-2 — Fuse Poisson + ES — **DONE (switchable)**

- [x] `es_fused_from_rho_cl`: \(E_\mathrm{ES}\propto\mathrm{IFFT}(\rho_\mathrm{diff}(k)\,\tilde\rho_\mathrm{tip}(k)/k^2)\)
- [x] Pauli remains separate (overlap, **no** \(1/k^2\))
- Parity: `tests/SPM/test_afm_fdbm.py::test_fdbm_fast_s3_parity_pauli_es`

### T-next-3 — Stay on GPU — **mostly done**

- [x] Shared AFMulator ctx with gpyFFT; device compose→gradient→`setup_fdbm_grid_from_img`
- [x] S4 `reuse_fdbm_grid` skips F re-upload
- [ ] Optional: skip F host download entirely when NO_IO (S4-only device)

### T-next-4 — Benchmarks (still open)

- [ ] Skip/async stage cache write for interactive GUI
- [ ] Coarser `step=0.15` interactive (not default)

### Round-2 measured (benzene, tip=co, 144×128×144, NO_IO)

| Path | S3 wall (incl. AFMulator ctor ~40 ms) | Field compute |
|------|----------------------------------------|---------------|
| Legacy `FAST_S3=0` | **0.256 s** | ~0.22 s (3 FFTs + CPU scale/pad + grad) |
| Fast `FAST_S3=1` | **0.109 s** | **0.068 s** (`S3.fast_fields_gpu`) |
| S4 after fast (reuse grid) | **0.002 s** | vs legacy S4 **0.047 s** |

Switch: `SPAMMM_AFM_FAST_S3=1` (default) / `=0` legacy. Diag downloads: `SPAMMM_AFM_DIAG_DOWNLOAD=1`.

---

## Historical notes (pre–Round 2 TODOs kept for context)

### T-next-1 (original) — Kill remaining CPU ops (pauli_scale, tip pad/roll, E_total)

See Round 2 section above — implemented behind `SPAMMM_AFM_FAST_S3`.

### T-next-2 (original) — Fuse Poisson + ES

See Round 2 — physics confirmed; fused kernel in use when FAST_S3=1.

---

## Historical baseline (pre–Round 1) — benzene 2026-07-19

Wall **1.65 s** warm GUI-like. Grid `136×128×144`. Dominant: Python `build_tasks`/AABB + NumPy FFT + slow NA orbital loop on larger mols.

### cProfile smoking guns (then)

| Hotspot | Note |
|---------|------|
| `build_tasks` / `check_overlap_sphere_aabb` | Pure Python — **fixed** → GPU default |
| `numpy.fft` | Poisson+Pauli+ES — **fixed** → gpyFFT |
| `project_neutral_density` AO loop | **fixed** → dense NA DM |

---

## CPU vs GPU inventory (current code, post Round 1)

| Operation | Implementation | Device |
|-----------|----------------|--------|
| S1 DFTB+ SCF | `DFTBcore` native lib | CPU |
| S2 `project_density_dense` | OpenCL kernels | GPU (+ heavy Python prep) |
| S2 `project_neutral_density` | OpenCL `project_density_dense` (diagonal NA DM); legacy AO loop via `SPAMMM_AFM_NA_ORBITAL_LOOP=1` | GPU (default) |
| S2 `build_tasks` / AABB | OpenCL `build_tasks_gpu` (default); CPU via `SPAMMM_AFM_CPU_TASKS=1` | GPU |
| S3 Poisson `fft_poisson` | gpyFFT; k-space multiply **on device** (default); NumPy via `SPAMMM_AFM_CPU_FFT=1` | GPU |
| S3 Pauli overlap | gpyFFT corr on device (default) | GPU |
| S3 Pauli scale `overlap**beta` | OpenCL `fdbm_scale_pauli_pow` (FAST_S3); numpy if `FAST_S3=0` | GPU / CPU |
| S3 ES | fused `ρ_diff·tip/k²` (FAST_S3); else Poisson+conv | GPU |
| S3 tip gaussian/CO pad | OpenCL `fdbm_pad_roll` (FAST_S3); numpy if `FAST_S3=0` | GPU / CPU |
| S3 `E_total` sum | OpenCL `fdbm_compose_E_to_img` (FAST_S3); numpy if legacy | GPU / CPU |
| S3 dispersion | `compute_dispersion_grid_cl` / `compute_dispersion_to_img_cl` | GPU |
| S3 `compute_gradient_cl` | OpenCL (shared AFMulator) | GPU |
| S4 `setup_fdbm_grid` + `scan_fdbm` | OpenCL (same AFMulator; no duplicate scan) | GPU |
| S4 `compute_df` | numpy | CPU (tiny) |
| Stage cache `np.savez` | disk | IO (gated by `SPAMMM_AFM_BENCH_NO_IO`) |
| Matplotlib plots | — | gated by `AFM_DEBUG_PLOT_LEVEL` |

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
- GUI defaults: `step=0.1`, `scan_step=0.1` (hardcoded), `tip_mode` pipeline default **`co`**.

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
4. Host↔device copies and `np.savez` I/O (uncompressed; still measurable on large grids)  
5. AFMulator construction + OpenCL compile (can dominate cold runs)  
6. S5/S6 if measuring “full GUI Run”, else exclude explicitly from FDBM bench

---

## How speed benchmarking works **today**

| Layer | Status |
|-------|--------|
| End-to-end FDBM stage timing | **`AFMBench`** in `spammm/SPM/AFM.py`; wired in `ModularPipeline` |
| Dedicated bench script | **`tests/SPM/bench_fdbm.py`** |
| Env gates | `SPAMMM_AFM_BENCH=0/1/2`, `SPAMMM_AFM_BENCH_NO_IO=1`, `SPAMMM_AFM_CPU_FFT`, `SPAMMM_AFM_CPU_TASKS`, `SPAMMM_AFM_NA_ORBITAL_LOOP` |
| OpenCL event profiling | Still optional / not default |
| Protocol | `doc/AGENTS/protocols/general/performance_optimization.md` |

---

## Pipeline stages (ModularPipeline.py S1–S6)

| Stage | What | Key code | Suspected bottleneck |
|-------|------|----------|---------------------|
| S1: SCF | DFTB+ SCF or pySCF | `ModularPipeline.stage1_scf()` | DFTB+ itself is fast (~0.05s for coronene). pySCF unknown. Python overhead in setup/teardown? |
| S2: Density projection | GPU density grid projection | `stage2_project()` → `Grid_dftb` | GPU kernel itself may be fine; Python orchestration / projector setup / host copies suspected. Partial `[TIME]` prints already in Grid_dftb |
| S3a: Potentials | Pauli (FFT), Poisson (FFT), vdW | `stage3_potentials()` energy part | FFT convolution; CO tip I/O; Python-side array copies; `np.savez` |
| S3b: Gradients | `E_total → F_total` | `compute_gradient_cl()` | Must be timed separately; required for S4 |
| S4: PP relaxation | Probe-particle MD on GPU | `stage4_relax()` | GPU; duplicate `scan_fdbm` removed (returns `FEs_relax` from compose) |
| S5: df / STM | Frequency shift / LDOS | `stage5_*` | Negligible for FDBM AFM target; exclude from <1s criterion unless measuring full GUI |
| S6: BR-STM / viz | Bond-resolved STM / plots | `stage6_*` | Visual only |

## Suspected Python-side overheads

1. **Host-device transfers between stages**: Each stage may copy arrays GPU→CPU→GPU unnecessarily. Check if `rho_scf`, `V_ES`, `F_total` stay on GPU or round-trip through numpy.
2. **Cache I/O**: `np.savez` / `np.load` for each stage — disk writes add latency even when cache is hit.
3. **Redundant recomputation**: `force_recompute` flags may trigger unnecessary re-runs. Dirty flag system (S1–S6) may not track dependencies correctly.
4. **Grid setup per call**: projector / AFMulator may rebuild (compile kernels, allocate buffers) every call instead of caching across stages.
5. **OpenCL kernel compilation**: Kernels may recompile per session instead of being cached via `pyopencl.cache_dir`.
6. ~~**S4 double scan**~~ — removed (Round 1).

---

## Benchmarking plan (done for Round 1; keep for Round 2)

- [x] Per-stage `AFMBench` + `tests/SPM/bench_fdbm.py`
- [x] GPU FFT / tasks / NA dense / shared AFMulator / uncompressed cache
- [ ] Round-2 TODOs above (T-next-1 … T-next-4) — **after commit**
- [ ] Optional OpenCL event profiling once host CPU ops are gone

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
- `spammm/SPM/ModularPipeline.py` — staged pipeline + `AFMBench` hooks
- `spammm/SPM/AFM.py` — AFMulator + `AFMBench` + FFT GPU path
- `spammm/quantum/DFTB/Grid_dftb.py` — density projection / GPU tasks
- `tests/SPM/bench_fdbm.py` — headless FDBM perf harness
- `tests/SPM/test_afm_fdbm.py` — functional FDBM tests
- `doc/Tasks/AFMTesting.md` — functional tests + Round-2 perf TODO pointer
- `doc/AGENTS/protocols/general/performance_optimization.md`
