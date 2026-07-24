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
| Python | `run_spm.py` | active | Thin CLI: `afm` / `opt` / `smiles-afm`; amp-align strips; see `user_guide/SPM_CLI.md` |
| Python | `spammm/SPM/AFM.py` | active | AFMulator, `AFMBench`, gpyFFT, `stage3_fdbm_fields_fast`, `compute_df_amp` |
| Python | `spammm/SPM/ModularPipeline.py` | active | S1–S6; dual S3 (fast / `FAST_S3=0` legacy) |
| Python | `spammm/SPM/AFM_utils.py` | active | Tips (`pad_mode`), compose/relax; `plot_afm_variant_height_strip(amp_align=…)` |
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
- [ ] Site-resolved Pauli \(A,\beta\) vs Kriging (transferability) — `doc/Tasks/Pauli_A_beta_KrigingTransferability.md`
- [ ] Prolonged / dual-basis Pauli (stock ES) — `doc/Tasks/ProlongedRadialBasis_DFTB.md`
- [ ] **Fukui molecule panel FDBM:** cube vs DFTB stock vs prolonged for pentacene, PTCDA, azaindol_(iso)dimer, benzoicacid/amid dimers — `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/*_PBE_def2-SVP/`
- [ ] Cube tip Δρ / ES morphology vs Kriging — `doc/Tasks/Import_KrigingGridFF.md`, reports `Kriging_*.md`
- [~] **All-electron Δρ / NA multipoles:** aniso `(sx,sy,sz)` + **element-invariant** clamp compensation + cube-node→project adapter implemented (`delta_rho_clamp_compact_na(q_na_mode='element_mean')`). Pentacene \|p_xy\| 0.28→0.018 (near NA_cube control). Report [`Cube_ES_DeltaRho_NA_dipole.md`](../Reports/Cube_ES_DeltaRho_NA_dipole.md); handoff [`Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md`](../Reports/Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md). **Awaiting USER visual confirmation** — do not mark fixed yet.
- [~] **df↔Fz amp presentation (CLI):** `run_spm.py afm` defaults amp-align + dense z (Fukui notes §2). Panel-fukui 3-row Fz_u/Fz_r/df still open.
