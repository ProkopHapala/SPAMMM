# quantum/DFTB/

DFTB+ integration: ctypes C-API wrapper, basis set parsing, GPU density projection, and basis optimization.

- **DFTBcore.py** — ctypes interface to DFTB+ C-API: SCF calculations, density/Hamiltonian/overlap matrix export
- **DFTBplusParser.py** — Parse DFTB+ HSD input files (wfc.*.hsd) to extract Slater-type orbital basis parameters
- **Grid_dftb.py** — GPU projection of DFTB densities onto 3D grids (OpenCL). Default: dense `project_density_dense` + GPU `build_tasks`; NA path uses diagonal NA DM (one launch). Backups: `SPAMMM_AFM_CPU_TASKS=1`, `SPAMMM_AFM_NA_ORBITAL_LOOP=1` (legacy per-AO — ~200× slower on large mols).
- **basis_optimizer.py** — Fit single-exponential Slater-tail basis (N, zeta) to reference density via simulated annealing on z-profile points
