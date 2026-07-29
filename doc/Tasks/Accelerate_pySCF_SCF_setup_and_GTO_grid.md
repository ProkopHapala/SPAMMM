# Accelerate pySCF for FDBM — SCF setup + GTO density-on-grid

**Goal:** Make DFT (pySCF) density generation for FDBM AFM competitive enough that GPU SCF wins are not eaten by setup + ρ(r) projection.

**Status (re-measured 2026-07-29):** full job setup+SCF+GPU ρ — pentacene **~26→4.4 s (~6×)**, PTCDA **~38→7.1 s (~5.4×)**. Setup **3.5–3.7×** (screen/rmax); ρ **~200×** vs numint. Compact DF setup ~2 s PTCDA (fork A2) but f32 GPU-DF SCF does not converge. **Not wired** into `get_density_from_pyscf` yet.

**SSOT (fork):** `doc/reports/2026-07-SCF_setup_breakdown.md`, `GTO_FDBM_grid_projection.md`, parent task under `/home/prokop/git/pyscf/doc/tasks/`.  
**SPAMMM bench:** `tests/SPM/run_bench_pyscf_full_job.py` → `debug/pyscf_full_job_bench/`.

---

## Timing vocabulary (do not confuse)

| Name | What | Bench column |
|------|------|----------------|
| **Outer geometry step** | MD / opt / CO z-point | once per geom |
| **SCF setup** | grids, XC plan, DF `_cderi`, OpenCL | **`setup`** (once per job; amortize across geom if possible) |
| **SCF total** | full `mf.kernel()` all cycles | **`SCF`** |
| **SCF cycle** | one DIIS/J+XC iteration | `SCF / n_cycles` |
| **ρ_proj** | DM → ρ on FDBM Cartesian grid | **`ρ_proj`** (once per geom after SCF) |
| **Disk I/O** | `.npy` / `.cube` | **exclude** from benches |

```
for geom in MD/opt/z-scan:          # OUTER
    setup_or_reuse_mf(geom)         # setup
    for cycle in SCF: ...           # INNER
    rho = project_DM_to_grid(...)   # ρ_proj
```

---

## Baseline snapshot (SPAMMM bench 2026-07-28, before fork ρ GPU)

Grid `0.1 Å`, margin 5, z_extra 6; PBE/def2-SVP vs DFTB 3ob; RTX 3090; no ρ disk I/O.

| mol | method | nao | cyc | setup | SCF tot | ≈/cyc | ρ_proj | total |
|-----|--------|-----|-----|-------|---------|-------|--------|-------|
| pentacene | DFTB 3ob | 102 | — | — | 0.16 | — | **0.02** | 0.18 |
| pentacene | pySCF GPU | 378 | 12 | 6.91 | **1.73** | 0.14 | **17.52** | 26 |
| pentacene | pySCF CPU | 378 | 13 | ~0 | **55.3** | 4.3 | **18.2** | 74 |
| PTCDA | DFTB 3ob | 128 | — | — | 0.15 | — | **0.02** | 0.17 |
| PTCDA | pySCF GPU | 460 | 19 | 10.1 | **3.88** | 0.20 | **24.4** | 38 |
| PTCDA | pySCF CPU | 460 | 17 | ~0 | **106** | 6.3 | **23.9** | 130 |

| Ratio | pentacene | PTCDA |
|-------|-----------|-------|
| SCF total GPU/CPU | **32×** | **27×** |
| ρ CPU numint / DFTB | ~1000× | ~1000× |

### After fork optimizations (2026-07-29 re-measure, AFM tols)

Profile `production_radial_screened_splitk` + GPU GTO ρ (not numint). Isolated subprocess.

| mol | setup | SCF (cyc) | ρ_proj | **job** | vs 2026-07-28 job |
|-----|------:|----------:|-------:|--------:|------------------:|
| pentacene | **1.89** (was 6.91) | 2.40 (15) | **0.09** | **4.4** | **~6×** |
| PTCDA | **2.90** (was 10.11) | 4.10 (19) | **0.10** | **7.1** | **~5.4×** |

Setup stages now: XC ~0.5 s (was ~6 s); DF ~0.9–1.8 s on `full` path. Compact DF setup PTCDA ~2.0 s but f32 GPU-DF SCF does not converge — keep production+CPU DF for energies.

---

## Improvement 1 — SCF setup (fork progress)

**Profile:** setup was **CPU**, not OpenCL — Python screen CSR (5–8 s) + O(G×N) rmax (2 s) + DF `_cderi` (3–6 s).

**Landed (2026-07-29):** screen/rmax → XC setup ~0.5 s; production E2E setup **~1.9–2.9 s** (was 7–10 s). Compact GPU DF further → **~2.0 s** PTCDA (report A2). Residual on production = DF `full` (~1–2 s). See fork `2026-07-SCF_setup_breakdown.md`.

**Hygiene:** always `DF.storage=incore` + `prepare_df_for_scf` before timing cycles (silent HDF5 outcore spoils J) — `df_storage_and_benchmark_hygiene.md`.

---

## Improvement 2 — GTO ρ on FDBM grid (fork progress)

**Done in fork:** 8³ tiled Hermite Cartesian projector (`GTOGridProjector`), multi-ζ shell list (not `ish=l`), GPU task boxes (CPU box build was the false 5 s “kernel”). Report: `GTO_FDBM_grid_projection.md`.

**Open for SPAMMM:**

- [ ] Wire `GTOGridProjector` into `get_density_from_pyscf` (GPU + `numint` fallback)
- [ ] PTCDA full-grid ρ_proj bench (same protocol)
- [ ] Re-run headline AFM table with GPU ρ column
- [ ] Optional: reuse Hermite tables from SCF setup; per-shell rcut

---

## Acceptance

Do not mark done until USER confirms: setup policy + AFM path using GPU ρ with parity vs `numint`, and quoted numbers use the vocabulary above.
