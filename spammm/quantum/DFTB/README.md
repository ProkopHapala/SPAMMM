# quantum/DFTB/

DFTB+ integration: ctypes C-API wrapper, basis set parsing, GPU density projection, and basis optimization.

- **DFTBcore.py** — ctypes interface to DFTB+ C-API: SCF calculations, density/Hamiltonian/overlap matrix export
- **DFTBplusParser.py** — Parse DFTB+ HSD input files (wfc.*.hsd) to extract Slater-type orbital basis parameters
- **Grid_dftb.py** — GPU projection of DFTB wavefunctions and density matrices onto 3D real-space grids (OpenCL, sparse and dense modes)
- **basis_optimizer.py** — Fit single-exponential Slater-tail basis (N, zeta) to reference density via simulated annealing on z-profile points
