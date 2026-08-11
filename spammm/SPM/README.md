# SPM/

Scanning Probe Microscopy — AFM and STM. GPU-accelerated via PyOpenCL. FDBM path is the GUI engine (not Morse+point-charge).

- **AFM.py** — AFMulator (LJ/Morse + FDBM), PP relax; **contact_pme** `fit_contact_pme` / `run_scan_contact_pme` (local WG default); stiffness SSOT; FAST_S3 switches.
- **AFM_utils.py** — Tip densities, FDBM; strip plots; **`run_contact_pme_pp_afm`** (CLI SSOT for `--model contact_pme`, forces `core_backend='local'`).
- **stm_compare.py** — DFTB vs pySCF frontier orbitals / STM current / vacuum panels; SSOT for `run_spm.py stm *`.
- **ModularPipeline.py** — Staged S1–S6 AFM/STM with disk cache; dual Stage-3 (fast vs `FAST_S3=0` legacy).
- **KrigingGridFF.py** — DFT z-scan → GridFF `(nx,ny,nz,4)=(Fx,Fy,Fz,E)` for `setup_fdbm_grid`; Mithun loaders; deps NumPy/SciPy only
- **InterpolatorKriging.py** / **InterpolatorRBF.py** / **interpy.py** — Wendland C2 Kriging/RBF (ported from ppafm)
- **ManipulationPathOpt.py** — AFM tip-path optimization
- **ScanUtils.py** — Trajectory generators (grid/line/rotational/tilted)

**CLI:** repo-root `run_spm.py` — see [`user_guide/SPM_CLI.md`](../../user_guide/SPM_CLI.md). Models: `fdbm`, `morse`, `contact`, **`contact_pme`**. Report: [`ContactPME_PAW_AFM_MemSpeed_2026-08-11.md`](../../doc/Reports/ContactPME_PAW_AFM_MemSpeed_2026-08-11.md).

**Campaigns / open tasks:** STM mio/3ob/prolonged vs pySCF cubes → `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md` (`tests/SPM/testplot_stm_basis_compare.py`, artifacts `debug/stm_orbital_compare/`); site Pauli \(A,\beta\) maps → `doc/Tasks/Pauli_A_beta_KrigingTransferability.md`; Kriging import → `doc/Tasks/Import_KrigingGridFF.md`. GPU pySCF notes: `doc/AGENTS/notes/pyscf-gpu-scf.md`.

**Fukui DFT density panel (FDBM refs):** `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/{pentacene,PTCDA,azaindol_dimer,azaindol_isodimer,benzoicacid_dimer,benzoicamid_dimer}_PBE_def2-SVP/` (`rho_N`/`esp_N`, PBE/def2-SVP) — compare cube FDBM vs DFTB stock vs prolonged; table in `doc/Tasks/ProlongedRadialBasis_DFTB.md`. XYZ in `data/xyz/`.

**DFT ↔ FDBM:** science schema + data symlinks → `doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md`. Linked data: `data/mithun_afm_scans`, `mithun_afm_scans_flat`, `mithun_afm_tip_fukui`.

**Perf (2026-07, RTX 3090):** benzene warm ~0.18 s (was ~1.65 s); flat_1 S2 NA ~0.03 s (was ~6 s); S3 cache ~0.4 s uncompressed (was ~10 s compressed); fused ES + GPU pad/scale in Stage 3. Report: `doc/Tasks/PerfBenchmark_FDBM.md`. Bench: `tests/SPM/bench_fdbm.py`.

**Caveats:** (1) PP `K_LAT` N/m vs eV/Å² — `doc/Tasks/AFMTesting.md`. (2) Prefer grid `step ≤ 0.1 Å` for ES hex symmetry. (3) Pauli stays a separate density-overlap FFT — never fold into `1/k²` ES path.
