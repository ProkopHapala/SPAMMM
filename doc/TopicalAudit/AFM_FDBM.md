---
type: TopicalAudit
title: AFM FDBM Pipeline
tags: [AFM, FDBM, SPM, OpenCL, performance]
timestamp: 2026-07-19
---

# AFM FDBM Pipeline

## Summary

GUI AFM uses the **Full Density-Based Model** only: DFTB densities → Pauli (overlap) + electrostatics (Poisson / fused Fourier) + vdW → PP relax → df. Morse+point-charge AFMulator paths are tests/scripts only. Round-1+2 (2026-07) made Stage 2–3 GPU-first with a switchable fast Stage-3 (`SPAMMM_AFM_FAST_S3`).

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Python | `spammm/SPM/AFM.py` | active | AFMulator, `AFMBench`, gpyFFT, `stage3_fdbm_fields_fast` |
| Python | `spammm/SPM/ModularPipeline.py` | active | S1–S6; dual S3 (fast / `FAST_S3=0` legacy) |
| Python | `spammm/SPM/AFM_utils.py` | active | Tips (`pad_mode`), compose/relax |
| Python | `spammm/GUI/AFMExtension.py` | active | GUI wiring; K_LAT in N/m |
| Python | `spammm/quantum/DFTB/Grid_dftb.py` | active | Dense NA DM; GPU `build_tasks` |
| OpenCL | `kernels/AFM.cl` | active | PP relax + `fdbm_*` Stage-3 helpers |
| OpenCL | `kernels/LCAO_grid.cl` | active | Density projection (via Grid_dftb) |

## Parity Status

| Pair | Tolerance | Test |
|------|-----------|------|
| FAST_S3 Pauli+ES vs legacy FFT path | corr > 0.999, RMSE < 1e-5 | `tests/SPM/test_afm_fdbm.py::test_fdbm_fast_s3_parity_pauli_es` |
| GPU vs CPU tip pad/roll | exact float32 | ad-hoc (see perf session) |

## Measured speedups (RTX 3090, 2026-07-19)

| Case | Before | After |
|------|--------|-------|
| Benzene warm E2E | ~1.65 s | ~0.18 s |
| Benzene S3 fields | ~0.26 s | ~0.07 s |
| Flat_1 S2 `rho_na` | 5.87 s | 0.03 s |
| Flat_1 S3 cache | ~10 s compressed | ~0.4 s `savez` |
| Flat_1 warm S3+S4 NO_IO | many s | ~1.4 s |

Full report: [`doc/Tasks/PerfBenchmark_FDBM.md`](../Tasks/PerfBenchmark_FDBM.md).

## Design notes

- **ES fused:** \(E_\mathrm{ES}(k)\propto\rho_\mathrm{diff}(k)\,\tilde\rho_\mathrm{tip}(k)/k^2\) — not full `rho_scf`; tip = delta-density; match flip/conv convention.
- **Pauli separate:** density overlap FFT then `A·overlap^β` — never `1/k²`.
- **Legacy restore:** `SPAMMM_AFM_FAST_S3=0` (keeps old kernels/Python path).

## Open Issues

- [ ] Skip/async stage cache write for interactive GUI
- [ ] Optional interactive `step=0.15` (not quality default — hex symmetry / `q_diff`)
- [ ] Optional skip F host download when `NO_IO` (S4 device-only)
- [ ] Hex/rot60 image QA at coarse grids — `doc/Tasks/AFMTesting.md`
