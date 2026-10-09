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
| Python | `spammm/SPM/FDBMPipeline.py` | **active (CLI default)** | Device-resident path (`afm --pipeline gpu`, default): one ctx, one SCF, ρ never leaves GPU; parity df ≥0.9996 vs legacy; **batch contract: reuse ONE instance across molecules (different natoms/positions/DM) — same ngrid+step → warm ~0.1–0.2 s/mol**; `run_fields`/`fields_for_fit` for FDBM→2.5D-contact encode (`doc/export_invAFM/fdbm_compression.md`); `doc/Tasks/FDBM_EndToEnd_GPU_Pipeline.md` |
| Python | `spammm/SPM/AFM.py` | active | AFMulator, `AFMBench`, gpyFFT, `stage3_fdbm_fields_fast` (now also accepts `*_cl` device buffers), `compute_df_amp` |
| Python | `spammm/SPM/AFM_utils.py` | active | Tips, `compose_and_relax_total`, `run_br_stm_afm_panel`, plot SSOT |
| Python | `spammm/GUI/AFMExtension.py` | active | GUI adapter; must use Pauli SSOT from `PAULI_FITTED_DEFAULTS` |
| Python | `run_spm.py` `afm` | **deprecated fork** | Still calls `_run_from_density`; replace after parity gate (done) |
| Python | `tests/SPM/testplot_fdbm_fields_compress.py` | active | FDBM→coremesh compression harness (`--method coremesh --core-basis poly8sp --core-solver gpugram`): ~9000× disk vs dense GridFF (156–505 KB vs 0.65–2.2 GB), df corr .958–.999 on 5 PAHs — benchmark `doc/Reports/FDBM_CoreMesh_Compression_Benchmark_2026-10-09.md`, recipe `doc/HowTo/FDBM_Compression_CoreMesh.md` |
| Python | `tests/SPM/testplot_fdbm_relax._run_from_density` | **deprecated** | Legacy CPU-FFT Stage-3; keep only as parity/diagnostic until CLI cutover |
| Python | `tests/SPM/testplot_cli_vs_modular_parity.py` | active | LEGACY vs FAST step parity + timing |
| OpenCL | `kernels/AFM.cl` | active | PP relax + `fdbm_*` Stage-3 helpers |

## Parity Status

| Pair | Result | Artifacts / test |
|------|--------|------------------|
| FAST_S3 Pauli+ES vs legacy FFT (synthetic) | corr > 0.999, RMSE < 1e-5 | `test_afm_fdbm.py::test_fdbm_fast_s3_parity_pauli_es` |
| **CLI legacy Stage3–4 vs Modular FAST_S3** (shared ρ/tip/scan/FIRE) | corr ≥ 0.9996 on df; fields ~1.000; **~5.5×** S3+S4 speedup | `testplot_cli_vs_modular_parity.py` → `debug/cli_vs_modular_parity/` — **USER confirmed plots 2026-07-28** |
| **FDBMPipeline (device-resident) vs legacy CLI path** | ρ max\|Δ\| ≈1–3e-5; df corr 0.9996–0.9998 (azaindol/pentacene/PTCDA); ~2.2× end-to-end; host RSS ~3× lower | `test_fdbm_pipeline_v2.py` + `testplot_fdbm_pipeline_parity.py` → `debug/testplot_fdbm_pipeline_parity/` |
| GUI defaults vs CLI SSOT | **not yet identical** — see Open Issues (wrong Pauli spins were primary) | — |
| Directional df, vertical compatibility | `compute_df*_dir(n=z)` matches established vertical functions; analytical x derivative retains z slices | `test_afm_morse.py::test_df_direction_*` |
| Lateral scan geometry | `scan_fdbm(n=x)` and `scan_fdbm(n=z)` acquire identical `(x,y,z)` volumes on NVIDIA; only df projection changes | `test_afm_morse.py::test_scan_fdbm_oscillation_direction_does_not_replace_z_approach` |

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
- **Lateral/arbitrary AFM:** acquisition remains an `(x,y,z)` volume approached along z. For unit oscillation vector `n`, df is `-n·∇(F·n)`; finite amplitude samples along `n`. Only `amp*abs(n_z)` pads/aligns z, while x/y amplitude pads the lateral scan before cropping.
- **CPU/legacy FFT path DEPRECATED (2026-10-06):** `SPAMMM_AFM_CPU_FFT=1`, `--cpu-fft`, and the unfused `fft_poisson`/`compute_pauli_overlap`/`compute_es_conv_field` NumPy path emit `DeprecationWarning` — parity backup only. Several× slower (PTCDA 17.7→4.8 s) and multi-GB complex128 workspaces. Product path is `stage3_fdbm_fields_fast` (fused GPU FAST_S3) everywhere; `SPAMMM_AFM_FAST_S3=0` is debug-only. Never `os.environ.setdefault('SPAMMM_AFM_CPU_FFT','1')` at module level — that silent forcing caused a 4× speed/memory regression.
- **Residual waste (design doc):** densities still bounce GPU→CPU→GPU (projector context ≠ AFMulator context), two-basis calls share no SCF/projector state, tip+FFT plans rebuilt per molecule shape. Design + API sketch: [`doc/Tasks/FDBM_EndToEnd_GPU_Pipeline.md`](../Tasks/FDBM_EndToEnd_GPU_Pipeline.md). `get_density_from_dftb_dense` gained `dm_in`/`need_es`/`need_ves` (2026-10-07) — dual-basis caller now does ONE SCF + skips unused V_ES/ρ_na.

## Open Issues

- [ ] **Remove legacy CLI FDBM Stage-3** — fold `run_spm.py afm` (+ gallery/`panel-fukui` helpers) onto ModularPipeline / FAST_S3; demote `_run_from_density` to diagnostic or delete after cutover. Tracker: [`Consolidate_GUI_CLI_Backend_Input_Protocol.md`](../Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md).
- [~] **GUI parameter parity with CLI** — Pauli / FIRE / scan_margin defaults fixed 2026-07-28; residual: ModularPipeline `linspace` vs CLI `arange` lateral sampling. Await USER visual GUI↔CLI check.
- [ ] Skip/async stage cache write for interactive GUI
- [ ] Optional skip F host download when `NO_IO` (S4 device-only)
- [ ] Site-resolved Pauli \(A,\beta\) vs Kriging — `doc/Tasks/Pauli_A_beta_KrigingTransferability.md`
- [~] All-electron Δρ / NA multipoles — awaiting USER visual confirmation on cube ES reports
- [~] Lateral-AFM coordinate correction — analytical and NVIDIA tests pass; PTCDA x-oscillation review artifact at `debug/lateral_afm/lat_x_zslices_corrected/compare_per_image.png`; awaiting USER visual confirmation
