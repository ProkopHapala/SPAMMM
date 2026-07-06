# tests/SPM/

AFM (Atomic Force Microscopy) and scanning probe simulation tests and visual demos.

See `doc/TEST_DESIGN.md` for L0/L1/L2 review levels.

## Files

| Script | Class | Purpose |
|--------|-------|---------|
| `test_afm_morse.py` | pytest | Morse/LJ + Coulomb AFM: force field grid, scans, frequency shift |
| `test_afm_fdbm.py` | pytest | FDBM pipeline: DFTB SCF, density projection, relaxed scan |
| `testplot_density_projection.py` | visual | DFTB+ density projection, 2D slices + `.cub` |
| `testplot_fdbm_potentials.py` | visual | FDBM potential plots (Pauli, ES, dispersion) |
| `testplot_fdbm_relax.py` | visual | FDBM + PP relaxation: Fz, df, tip displacement |
| `testplot_afm_morse.py` | visual | Morse AFM energy/Fz/df maps |
| `testplot_zscan_reference.py` | visual | Z-scan reference curves |
| `testplot_3ob_basis_tails.py` | visual | 3ob basis tail diagnostics |
| `testplot_dftb_vs_pyscf_basis.py` | visual | DFTB vs PySCF basis comparison |

Run visual demos: `python tests/SPM/testplot_fdbm_relax.py`
