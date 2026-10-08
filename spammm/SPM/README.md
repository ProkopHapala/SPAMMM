# SPM/

Scanning Probe Microscopy — AFM and STM. GPU-accelerated via PyOpenCL. FDBM path is the GUI engine (not Morse+point-charge).

- **AFM.py** — AFMulator (LJ/Morse + FDBM), PP relax; **contact_pme** `fit_contact_pme` / `run_scan_contact_pme` (local WG default; `relax_mode` = `fire` default | `qn` | `sph` ~4× faster, `qn_cap`/`qn_conv` knobs — report `doc/Reports/ContactPME_RelaxQuasiNewton_Sph_2026-10-03.md`); stiffness SSOT; FAST_S3 switches.
- **AFM_utils.py** — Tip densities, FDBM; strip plots; **`run_contact_pme_pp_afm`** (CLI SSOT for `--model contact_pme`, forces `core_backend='local'`).
- **stm_compare.py** — DFTB vs pySCF frontier orbitals / STM current / vacuum panels; SSOT for `run_spm.py stm *`.
- **ModularPipeline.py** — Staged S1–S6 AFM/STM with disk cache; dual Stage-3 (fast vs `FAST_S3=0` legacy).
- **KrigingGridFF.py** — DFT z-scan → GridFF `(nx,ny,nz,4)=(Fx,Fy,Fz,E)` for `setup_fdbm_grid`; Mithun loaders; deps NumPy/SciPy only
- **InterpolatorKriging.py** / **InterpolatorRBF.py** / **interpy.py** — Wendland C2 Kriging/RBF (ported from ppafm)
- **ManipulationPathOpt.py** — AFM tip-path optimization
- **ScanUtils.py** — Trajectory generators (grid/line/rotational/tilted)

**CLI:** repo-root `run_spm.py` — see [`user_guide/SPM_CLI.md`](../../user_guide/SPM_CLI.md). Models: `fdbm`, `morse`, `contact`, **`contact_pme`**. Report: [`ContactPME_PAW_AFM_MemSpeed_2026-08-11.md`](../../doc/Reports/ContactPME_PAW_AFM_MemSpeed_2026-08-11.md).

**FDBM → compact field (experimental, USER visual pending):** recommended producer `FDBMPipeline.run_fields(..., oracle=True)` — device-resident fields, `res[variant]['sample']` = `sample_fdbm` oracle (no 74 MB `F_total` host download); batch contract: one pipeline across molecules (same ngrid+step → warm ~0.1–0.2 s/mol). Two compression backends (see `doc/export_invAFM/fdbm_compression.md`):
  - **ContactPME dense mesh (RECOMMENDED — interim, 2026-10-08):** `sample_fdbm(mesh_nodes)` → `ContactSurface.cpm_params_from_samples(E_nodes, lo, h_mesh, halo, apos_heavy, cLJs)` — direct `_prefilter_3d` tridiagonal solves, **no CG, ~0.05 s/mol**, zero core → `a.cpm = cpm` → `run_scan_contact_pme` / `eval_contact_pme` (GPU kernels, ~0.02 s scan). h_mesh=**0.20 Å** for 0.1 Å FDBM grids (df_corr ≥0.99; h=0.15 fits source trilinear kinks — worse); ~2–3 MB/mol ≈ 110–120× vs dense. **The real PME split (PAW cores + coarse mesh) is the design goal — `--method pme` (no `--no-core`) wires `fit_cores_paw_field`, but measured 2026-10-08 it fails df parity (azaindol df_corr 0.11–0.83; taper-shell ringing) — zero-core dense mesh wins on accuracy-per-byte TODAY, not by principle.** Harness: `python3 -u tests/SPM/testplot_fdbm_fields_compress.py azaindol PTCDA --method pme --h-mesh 0.2`.
  - **Separable 2.5D (DEPRECATED — slow CG, kept for minimum-bytes/back-compat):** `AFMulator.fit_contact_field(sep, xyz, E_ref, F_ref)` CG ~2 min/mol → `SeparableParams.save_npz`/`load_npz` (~23 KB/mol) → `setup_contact_surface` → `run_scan_contact` (~0.5 s scan). Reproduce measurement harness — outdir MUST be new (frozen artifacts, never overwrite):
```bash
python3 -u tests/testplot_contact_surface.py --fdbm-cache debug/spm_brstm/pentacene --outdir debug/testplot_contact_surface/fdbm_pentacene_review --grid-step 0.1 --bspl-steps 0.8 --fit-iters 1200 --z-modes 8 --poly-z0 2.2 --poly-R 8
```
Focused tests: `python3 -m pytest tests/SPM/test_afm_contact_surface.py -k 'contact_field or sample_fdbm or separable_npz' -s`

> Experimental FDBM compression (2026-10-02): cached prolonged pentacene GridFF (74,317,824 bytes) fitted to a separable 2.5D field, saved/reloaded as a 23,064-byte NPZ (28,152 bytes for float32 coefficients + h0). PTCDA with the same recipe: 73,728,000 bytes → 22,945-byte NPZ (29,952 resident bytes). Across 11 physical probe heights 2.7–3.7 Å (df centers 3.7–4.7 Å, amp=1 Å), df correlations were 0.9868–0.9998 for pentacene and 0.9871–0.9998 for PTCDA. These are fixed-tip, planar two-molecule results; this is not the Morse-derived ContactPME core model, not lossless, and not yet USER-confirmed. The current contact evaluator is slower than GridFF; iteration-cap convergence, arbitrary trajectories, broader chemical coverage and tip transfer remain to be validated.

> `run_scan_contact` / `get_raw_FE_contact` / `run_scan_pic` perf fix (2026-10-02): kernels must NOT be launched `ls=(1,)` — 1-thread workgroups cost ~34x (1.19 s → 0.035 s on 38k×31 raster, RTX 3090). They have no bounds guard, so `points`/`FEs` are host-padded to the `ls=128` multiple and results sliced back. See `doc/Caveats.md` §20.

PTCDA variant: same recipe, just `--fdbm-cache debug/spm_brstm/PTCDA` and a fresh `--outdir` (e.g. `debug/testplot_contact_surface/fdbm_ptcda_review`).

**Campaigns / open tasks:** STM mio/3ob/prolonged vs pySCF cubes → `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md` (`tests/SPM/testplot_stm_basis_compare.py`, artifacts `debug/stm_orbital_compare/`); site Pauli \(A,\beta\) maps → `doc/Tasks/Pauli_A_beta_KrigingTransferability.md`; Kriging import → `doc/Tasks/Import_KrigingGridFF.md`. GPU pySCF notes: `doc/AGENTS/notes/pyscf-gpu-scf.md`.

**Fukui DFT density panel (FDBM refs):** `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/{pentacene,PTCDA,azaindol_dimer,azaindol_isodimer,benzoicacid_dimer,benzoicamid_dimer}_PBE_def2-SVP/` (`rho_N`/`esp_N`, PBE/def2-SVP) — compare cube FDBM vs DFTB stock vs prolonged; table in `doc/Tasks/ProlongedRadialBasis_DFTB.md`. XYZ in `data/xyz/`.

**DFT ↔ FDBM:** science schema + data symlinks → `doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md`. Linked data: `data/mithun_afm_scans`, `mithun_afm_scans_flat`, `mithun_afm_tip_fukui`.

**Perf (2026-07, RTX 3090):** benzene warm ~0.18 s (was ~1.65 s); flat_1 S2 NA ~0.03 s (was ~6 s); S3 cache ~0.4 s uncompressed (was ~10 s compressed); fused ES + GPU pad/scale in Stage 3. Report: `doc/Tasks/PerfBenchmark_FDBM.md`. Bench: `tests/SPM/bench_fdbm.py`.

**Caveats:** (1) PP `K_LAT` N/m vs eV/Å² — `doc/Tasks/AFMTesting.md`. (2) Prefer grid `step ≤ 0.1 Å` for ES hex symmetry. (3) Pauli stays a separate density-overlap FFT — never fold into `1/k²` ES path.
