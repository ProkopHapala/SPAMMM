# tests/SPM/

AFM (Atomic Force Microscopy) and scanning probe simulation tests and diagnostic plots.

## Files

| Script | Class | Purpose |
|--------|-------|---------|
| `test_afm_morse.py` | 1 | AFM imaging with Morse/LJ + Coulomb: force field grid, raw/relaxed scans, frequency shift |
| `test_afm_fdbm.py` | 1 | Full Density-Based Model pipeline: DFTB SCF, density projection, Pauli/ES/dispersion, relaxed scan |
| `plot_density_projection.py` | 2 | Project DFTB+ electron density, save 2D slices + Gaussian `.cub` files |
| `plot_fdbm_potentials.py` | 2 | Plot FDBM potentials (Pauli, electrostatic, dispersion, total) — XY slices, XZ cross-sections, 1D curves |
| `plot_fdbm_relax.py` | 2 | Full FDBM pipeline with PP relaxation: Fz forces, frequency shift, tip displacement |
| `run_afm_morse_visual.py` | 2 | Morse potential AFM visualization: energy slices, Fz maps, df maps for pentacene/PTCDA |
