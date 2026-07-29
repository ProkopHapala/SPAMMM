# tests/SPM/

AFM (Atomic Force Microscopy) and scanning probe simulation tests and visual demos.

See `doc/TEST_DESIGN.md` for L0/L1/L2 review levels.

## Files

| Script | Class | Purpose |
|--------|-------|---------|
| `test_afm_morse.py` | pytest | Morse/LJ + Coulomb AFM: force field grid, scans, frequency shift |
| `test_afm_fdbm.py` | pytest | FDBM pipeline: DFTB SCF, density projection, relaxed scan |
| `testplot_density_projection.py` | visual | DFTB+ density projection, 2D slices + `.cub` |
| `testplot_lcao_tile_partition.py` | visual | LCAO 8³ block tiling vs atom Rcut (denmap pairs) → `debug/lcao_tile_partition/` |
| `testplot_fdbm_potentials.py` | visual | FDBM potential plots (Pauli, ES, dispersion) |
| `testplot_fdbm_relax.py` | visual | FDBM + PP relaxation; **legacy** `_run_from_density` (deprecated for product CLI — see parity script) |
| `testplot_cli_vs_modular_parity.py` | visual | **LEGACY CLI Stage3–4 vs Modular FAST_S3** parity + timing → `debug/cli_vs_modular_parity/` |
| `bench_fdbm.py` | bench | ModularPipeline `AFMBench` segment timings |
| `run_afm_cli_fdbm_gallery.py` | CLI | Fukui gallery via `panel-fukui` (still on legacy until cutover) |
| `testplot_afm_morse.py` | visual | Morse AFM energy/Fz/df maps |
| `test_afm_contact_surface.py` | pytest | Contact-surface AFM: separable + PIC parity |
| `testplot_afm_contact_surface.py` | visual | Contact-surface fit + PP relaxed scan |
| `testplot_folded_rigid_diag.py` | visual | Folded basis rigid body relaxation diagnostics |
| `testplot_zscan_reference.py` | visual | Z-scan reference curves |
| `extract_pauli_zscan.py` | visual | FDBM Pauli vs Ez (log); CO or Gaussian tip |
| `compute_densities.py` | CLI | Cache ρ for z-scan / Pauli / SA fit |
| `testplot_3ob_basis_tails.py` | visual | 3ob basis tail diagnostics |
| `testplot_dftb_vs_pyscf_basis.py` | visual | DFTB vs PySCF basis comparison |
| `testplot_kriging_vs_fdbm_cube.py` | visual | Kriging DFT vs FDBM; Pauli fit modes |
| `testplot_stm_basis_compare.py` | visual | STM HOMO/LUMO: mio/3ob stock vs prolonged vs pySCF → `debug/stm_orbital_compare/` |
| `test_stm_basis_compare.py` | pytest | L0: prolonged ≠ stock at vacuum height (benzene, `@gpu @slow`) |

**Open task scripts:** Pauli site maps → `doc/Tasks/Pauli_A_beta_KrigingTransferability.md`; contact-surface finish → `doc/Tasks/Fast_2p5D_AFM_ContactSurface.md`.

**Fukui DFT densities (FDBM refs):** `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/{pentacene,PTCDA,azaindol_dimer,azaindol_isodimer,benzoicacid_dimer,benzoicamid_dimer}_PBE_def2-SVP/` — cube vs DFTB stock vs prolonged; see `doc/Tasks/ProlongedRadialBasis_DFTB.md`.

Run visual demos: `python tests/SPM/testplot_fdbm_relax.py`  
PTCDA stock vs SA: `SPAMMM_AFM_CPU_FFT=1 python tests/SPM/testplot_fdbm_relax.py --ptcda-stock-vs-sa`  
Presentation artifacts: `FOR_PRESENTATION.md`, `debug/presentation.html`
