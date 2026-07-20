# SPM/

Scanning Probe Microscopy — AFM and STM. GPU-accelerated via PyOpenCL. FDBM path is the GUI engine (not Morse+point-charge).

- **AFM.py** — AFMulator (LJ/Morse + FDBM), PP relax, df; **stiffness SSOT** `stiffness_Nm_to_eVA2` / `K_LAT_HAPALA_*` (internal eV/Å², GUI N/m). **Perf:** `AFMBench`, gpyFFT helpers, Round-2 `stage3_fdbm_fields_fast` / `fdbm_*` via shared AFMulator ctx. Switches: `SPAMMM_AFM_FAST_S3` (default 1), `SPAMMM_AFM_CPU_FFT`, `SPAMMM_AFM_CPU_TASKS`, `SPAMMM_AFM_BENCH`, `SPAMMM_AFM_BENCH_NO_IO`.
- **AFM_utils.py** — Tip densities (`pad_mode='cpu'|'none'`), FDBM orchestration, `compose_and_relax_total(..., reuse_fdbm_grid=)` for device-side S4.
- **ModularPipeline.py** — Staged S1–S6 AFM/STM with disk cache; dual Stage-3 (fast vs `FAST_S3=0` legacy).
- **KrigingGridFF.py** — DFT z-scan → GridFF `(nx,ny,nz,4)=(Fx,Fy,Fz,E)` for `setup_fdbm_grid`; Mithun loaders; deps NumPy/SciPy only
- **InterpolatorKriging.py** / **InterpolatorRBF.py** / **interpy.py** — Wendland C2 Kriging/RBF (ported from ppafm)
- **ManipulationPathOpt.py** — AFM tip-path optimization
- **ScanUtils.py** — Trajectory generators (grid/line/rotational/tilted)

**DFT ↔ FDBM:** science schema + data symlinks → `doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md`. Linked data: `data/mithun_afm_scans`, `mithun_afm_scans_flat`, `mithun_afm_tip_fukui`.

**Perf (2026-07, RTX 3090):** benzene warm ~0.18 s (was ~1.65 s); flat_1 S2 NA ~0.03 s (was ~6 s); S3 cache ~0.4 s uncompressed (was ~10 s compressed); fused ES + GPU pad/scale in Stage 3. Report: `doc/Tasks/PerfBenchmark_FDBM.md`. Bench: `tests/SPM/bench_fdbm.py`.

**Caveats:** (1) PP `K_LAT` N/m vs eV/Å² — `doc/Tasks/AFMTesting.md`. (2) Prefer grid `step ≤ 0.1 Å` for ES hex symmetry. (3) Pauli stays a separate density-overlap FFT — never fold into `1/k²` ES path.
