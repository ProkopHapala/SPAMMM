# surfaces/

Substrate interaction modeling for molecule-on-surface simulations. GridFF precomputes 3D potential grids; Ewald handles electrostatics.

- **GridFF.py** — PyOpenCL B-spline grid force field: precompute Pauli/London/Coulomb potentials for a substrate, interpolate at arbitrary points
- **SurfaceEwald.py** — GPU 2D Ewald summation for electrostatic potentials/fields above periodic surfaces (production path)
- **Ewald2D.py** — Pure NumPy 2D Ewald reference implementation (plane-wave formulation, for parity checks against GPU)
- **Surface_utils.py** — GridFF metadata, loading precomputed grids, visualization with atom overlay, sampling at atom positions
- **GridFFRelaxedScan.py** — Relaxed PES scanning: interaction energy vs position/orientation with full geometry relaxation at each point
- **FoldedRigid.py** — Folded-basis rigid-body simulation: fitting folded potentials, relaxation, lateral scans, manipulation trajectories
- **SubstrateBuilder.py** — Crystal slab generation for ionic crystals (NaCl, CaF2): flat slabs and step edges
- **surface_plots.py** — Matplotlib visualization for relaxation trajectories, lateral scans, manipulation trajectories, relaxed scans
