---
type: Report
title: FDBM→coremesh compression benchmark — dense GridFF vs compact sp-core + B-spline mesh
tags: [FDBM, contact-pme, compression, benchmark, gpu, coremesh]
timestamp: 2026-10-09
---

# FDBM→coremesh compression benchmark (RTX 3090)

**Setup.** Dense FDBM fields (DFTB+ backend, CO tip, float4 E+F, step 0.1 Å)
are compressed into the `poly8sp` coremesh representation —
`A:s3p2+B:s1` compact core (9 coefs/center) + cubic B-spline mesh h=1 Å —
via `fit_core_sp_gram` (GPU Gram, λ=1e-6) + omit-by-inpaint separable banded
mesh fit (design doc §14–18, HowTo `doc/HowTo/FDBM_Compression_CoreMesh.md`).
Scan = full PP-AFM relaxation + df through the ContactPME kernels.
Each molecule ran in its own process (device images accumulate otherwise).

## 1. Disk / memory — the headline

| molecule | atoms | dense GridFF (float4) | coremesh archive (.pkl) | GPU-resident (coefs+mesh) | ratio |
|---|---|---|---|---|---|
| azaindol | 15 | 650.3 MB | 156 KB | 75 KB | 8 691× |
| pentacene | 36 | 990.9 MB | 239 KB | 114 KB | 8 728× |
| PTCDA | 38 | 891.8 MB | 215 KB | 101 KB | 8 822× |
| PTCDI | 40 | 1 032.2 MB | 229 KB | 108 KB | 9 590× |
| circumcoronene | 72 | 2 202.0 MB | 505 KB | 240 KB | 9 167× |

- Dense GridFF = `(nx,ny,nz,4) float32`, e.g. circumcoronene 640×640×336.
- Archive = `ContactPMEParams` pickle: core coefs + mesh coefficients + grid scalars.
- GPU-resident is what the evaluator actually uploads (centers+coefs+mesh);
  ~10⁴× less than the dense image it replaces, and ~1000 molecules' worth of
  compressed fields fit in the memory of **one** dense field — the enabling
  property for ML-scale molecule sets.
- Archive grows ~linearly with centers (≈3.1 KB/center incl. mesh overlap).

## 2. Fit speed (one-time per molecule)

| molecule | ncoef | field gen `run_fields` | sample | **core (gpugram)** | mesh (inpaint) | **total fit** |
|---|---|---|---|---|---|---|
| azaindol | 279 | 0.93 s | 0.05 s | 0.34 s | 0.96 s | **1.36 s** |
| pentacene | 684 | 1.33 s | — | 0.78 s | 1.8 s | **2.70 s** |
| PTCDA | 738 | 1.44 s | — | 0.76 s | 1.4 s | **2.20 s** |
| PTCDI | 774 | 1.38 s | — | 0.80 s | 1.5 s | **2.38 s** |
| circumcoronene | 1458 | 8.69 s | 0.17 s | 1.87 s | 5.04 s | **7.09 s** |

Core fit scales ~linearly in **active coefficient count** (0.3→1.9 s for
279→738 active columns, ~1.5 ms/column of GPU operator calls) — i.e. ~linear
in atoms, and crucially **independent of the sample count** (no dense design
matrix anywhere). History for the same fits:

| stage | before | now | speedup |
|---|---|---|---|
| core fit (circumcoronene) | 230 s CPU-Gram / 7.9 s CGLS | 1.87 s | ~120× / 4× |
| mesh fit (circumcoronene) | 120 s weighted scipy CG | 5.04 s | ~24× |
| mesh fit (azaindol) | 8.7 s | 0.96 s | ~9× |

`run_fields` (generating the dense reference the fit consumes) is now the
dominant one-time cost — and it, not the compact model, is what fails on
circumcircumcoronene (120 atoms): 3.2 GB field image + resident FFT buffers
exceed allocatable device memory.

## 3. Scan / evaluation speed (per image, both on GPU)

| molecule | dense GridFF scan | coremesh scan | ratio |
|---|---|---|---|
| azaindol | 0.013 s | 0.04 s | ~3× |
| pentacene | 0.022 s | 0.08 s | ~4× |
| PTCDA | 0.021 s | 0.08 s | ~4× |
| PTCDI | 0.024 s | 0.08 s | ~3× |
| circumcoronene | 0.042 s | 0.21 s | ~5× |

Compressed eval is ~3–5× slower than a trilinear image read (9-mode core eval
+ mesh trilinear + bucket scatter), but absolute scan time stays ≤0.2 s —
irrelevant next to field generation or DFTB. Trade memory 10⁴×, pay ~4× eval.

## 4. Accuracy

| molecule | df corr min | df corr max | holdout Fz corr | CPU↔GPU max\|ΔF\| |
|---|---|---|---|---|
| azaindol | .984 (h_df 4.1) | .998 | .999 | 1.1e-6 |
| pentacene | .986 | .998 | .999 | — |
| PTCDA | .958 | .998 | .999 | — |
| PTCDI | .966 | .999 | 1.000 | — |
| circumcoronene | .982 (h_df 4.1) | .999 | 1.000 | 1.7e-6 |

Min sits at h_df≈4.1 Å where Fz crosses zero (contrast sign flips are most
sensitive there). Inpaint mesh fit is **equal-or-better than the old weighted
CG** for `A:s3p2+B:s1`; atoms-only `A:s3p3` regresses at mid heights
(.74–.92) — keep `--mesh-fit cg` for it.

## 5. Reproduce

```bash
python3 tests/SPM/testplot_fdbm_fields_compress.py <mol> \
    --method coremesh --core-basis poly8sp \
    --core-solver gpugram --core-ridge 1e-6 --skip-spline-only
# per-molecule dirs: debug/testplot_fdbm_fields_compress/<mol>_s0.1_coremesh_poly8sp-As3p2Bs11/
#   SUMMARY.out · compare_df_Fz.png · coremesh_1.pkl · comparison_arrays.npz
```

Figures: `debug/testplot_fdbm_fields_compress/*_s0.1_coremesh_poly8sp-As3p2Bs11/compare_df_Fz.png`.
