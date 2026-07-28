---
type: TopicalAudit
title: AFM FDBM Pipeline
tags: [AFM, FDBM, SPM, OpenCL, performance, GUI, CLI]
timestamp: 2026-07-28
---

# AFM FDBM Pipeline

## Summary

Product FDBM path is **ModularAFMPipeline** (FAST_S3 GPU Stage-3 + FIRE PP scan): DFTB densities → Pauli + ES + vdW → PP relax → df / BR-STM. GUI already uses this. CLI `afm` still forks through `tests/SPM/testplot_fdbm_relax._run_from_density` (NumPy FFT Stage-3 by default) — **parity vs FAST_S3 confirmed** (pentacene/PTCDA, 2026-07-28); **legacy CLI Stage-3 must be removed** and `cmd_afm` folded onto ModularPipeline. Morse+Coulomb is a separate shared runner (`AFM_utils.run_morse_coulomb_afm`), not FDBM.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| Python | `spammm/SPM/ModularPipeline.py` | **active (SSOT)** | S1–S6; FAST_S3 default; GUI + `stm br` |
| Python | `spammm/SPM/AFM.py` | active | AFMulator, `AFMBench`, gpyFFT, `stage3_fdbm_fields_fast`, `compute_df_amp` |
| Python | `spammm/SPM/AFM_utils.py` | active | Tips, `compose_and_relax_total`, `run_br_stm_afm_panel`, plot SSOT |
| Python | `spammm/GUI/AFMExtension.py` | active | GUI adapter; must use Pauli SSOT from `PAULI_FITTED_DEFAULTS` |
| Python | `run_spm.py` `afm` | **deprecated fork** | Still calls `_run_from_density`; replace after parity gate (done) |
| Python | `tests/SPM/testplot_fdbm_relax._run_from_density` | **deprecated** | Legacy CPU-FFT Stage-3; keep only as parity/diagnostic until CLI cutover |
| Python | `tests/SPM/testplot_cli_vs_modular_parity.py` | active | LEGACY vs FAST step parity + timing |
| OpenCL | `kernels/AFM.cl` | active | PP relax + `fdbm_*` Stage-3 helpers |

## Parity Status

| Pair | Result | Artifacts / test |
|------|--------|------------------|
| FAST_S3 Pauli+ES vs legacy FFT (synthetic) | corr > 0.999, RMSE < 1e-5 | `test_afm_fdbm.py::test_fdbm_fast_s3_parity_pauli_es` |
| **CLI legacy Stage3–4 vs Modular FAST_S3** (shared ρ/tip/scan/FIRE) | corr ≥ 0.9996 on df; fields ~1.000; **~5.5×** S3+S4 speedup | `testplot_cli_vs_modular_parity.py` → `debug/cli_vs_modular_parity/` — **USER confirmed plots 2026-07-28** |
| GUI defaults vs CLI SSOT | **not yet identical** — see Open Issues (wrong Pauli spins were primary) | — |

### CLI legacy vs FAST (RTX 3090, warm S3+S4 only)

| Mol | LEGACY | FAST | Speedup | df corr |
|-----|--------|------|---------|---------|
| pentacene | 1.23 s | 0.23 s | 5.45× | 0.9996 |
| PTCDA | 1.16 s | 0.21 s | 5.47× | 0.9996 |

## Measured speedups (ModularPipeline Round-1+2, RTX 3090)

| Case | Before | After |
|------|--------|-------|
| Benzene warm E2E | ~1.65 s | ~0.18 s |
| Benzene S3 fields | ~0.26 s | ~0.07 s |
| Flat_1 warm S3+S4 NO_IO | many s | ~1.4 s |

Full report: [`doc/Tasks/PerfBenchmark_FDBM.md`](../Tasks/PerfBenchmark_FDBM.md).

## Design notes

- **ES fused (FAST_S3):** \(E_\mathrm{ES}(k)\propto\rho_\mathrm{diff}(k)\,\tilde\rho_\mathrm{tip}(k)/k^2\).
- **Pauli separate:** overlap FFT then `A·overlap^β` — never `1/k²`.
- **Pauli A,β SSOT:** `AFM.PAULI_FITTED_DEFAULTS['3ob-3-1']` = **A=124.84, β=1.4330** (evaluation). Old single-atom fits (509.28 / 1.0586) are obsolete.
- **Height SSOT:** df window 3.7–4.7 Å, amp=1.0 → `afm_df_height_stacks` + `compute_df_amp`.
- **Legacy restore only for debug:** `SPAMMM_AFM_FAST_S3=0` or `SPAMMM_AFM_CPU_FFT=1` — not product path.

## Open Issues

- [ ] **Remove legacy CLI FDBM Stage-3** — fold `run_spm.py afm` (+ gallery/`panel-fukui` helpers) onto ModularPipeline / FAST_S3; demote `_run_from_density` to diagnostic or delete after cutover. Tracker: [`Consolidate_GUI_CLI_Backend_Input_Protocol.md`](../Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md).
- [~] **GUI parameter parity with CLI** — Pauli / FIRE / scan_margin defaults fixed 2026-07-28; residual: ModularPipeline `linspace` vs CLI `arange` lateral sampling. Await USER visual GUI↔CLI check.
- [ ] Skip/async stage cache write for interactive GUI
- [ ] Optional skip F host download when `NO_IO` (S4 device-only)
- [ ] Site-resolved Pauli \(A,\beta\) vs Kriging — `doc/Tasks/Pauli_A_beta_KrigingTransferability.md`
- [~] All-electron Δρ / NA multipoles — awaiting USER visual confirmation on cube ES reports
