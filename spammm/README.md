# spammm/

Core Python package for SPAMMM — Scanning Probe Accelerated Modeling of Microscopy and Manipulation. Contains molecular topology, force fields, SPM simulation, quantum chemistry backends, surface interaction models, and GUI.

- **AtomicSystem.py** — Array-based molecular representation for force field evaluation (flat NumPy arrays: positions, types, bonds)
- **atomicUtils.py** — Molecular I/O (XYZ, MOL, MOL2), bond finding, geometry utilities
- **elements.py** — Periodic table data (Z, mass, radii, colors, electronegativity) for all 118 elements
- **globals.py** — Centralized debug/verbosity controls, dependency-free
- **config_utils.py** — JSON config loading, path resolution for DFTB basis sets and SK parameters
- **plotUtils.py** — Pure-matplotlib 1D/2D plotting functions (energies, forces, orbitals, densities, ESP)
- **topology/** — Editable molecular topology: AtomicGraph (SSOT), Kekule backend, heterocycle generation
- **forcefields/** — UFF, SPFFsp3, QEq, rigid body dynamics, FFController orchestrator
- **SPM/** — AFM/STM simulation: AFMulator, modular pipeline, manipulation path optimization
- **quantum/** — DFTB+ and pySCF integration for electron density computation
- **surfaces/** — GridFF, 2D Ewald, substrate builder, relaxed PES scanning
- **GUI/** — PyQt5 application: molecular editor, 3D viewers, extension system
- **utils/** — OpenCL base class, device selection, linear algebra, test utilities
