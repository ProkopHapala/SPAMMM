# quantum/

Quantum chemistry integration for electron density computation. Densities feed into the FDBM AFM method in `SPM/`.

- **DFTB_utils.py** — DFTB+ integration: input generation, output parsing, SK parameter management (subprocess and C-API)
- **pySCF_utils.py** — pySCF RHF/DFT calculations for electron densities on 3D grids (alternative to DFTB for higher accuracy)
- **DFTB/** — DFTB+ ctypes wrapper, basis parser, GPU density projection, basis optimizer
