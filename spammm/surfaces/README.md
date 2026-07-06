# surfaces/

Substrate and sample interaction modeling for molecule-on-surface / AFM simulations. GridFF precomputes dense 3D B-spline grids; **ContactSurface** is the compact quasi-2D alternative for aperiodic rigid samples.

- **GridFF.py** — PyOpenCL B-spline grid force field: precompute Pauli/London/Coulomb potentials for a substrate, interpolate at arbitrary points
- **ContactSurface.py** — GPU quasi-2D contact surface prototype (static AFM): separable B-spline(xy)×poly(dz) + radial PIC; brute Morse reference; matrix-free CG fit. Design: `doc/Topics/AFM/ContactSurface_Static.md`
- **SurfaceEwald.py** — GPU 2D Ewald summation for electrostatic potentials/fields above periodic surfaces (production path)
- **Ewald2D.py** — Pure NumPy 2D Ewald reference implementation (plane-wave formulation, for parity checks against GPU)
- **Surface_utils.py** — GridFF metadata, loading precomputed grids, visualization with atom overlay, sampling at atom positions
- **GridFFRelaxedScan.py** — Relaxed PES scanning: interaction energy vs position/orientation with full geometry relaxation at each point
- **FoldedRigid.py** — Folded-basis rigid-body simulation: fitting folded potentials, relaxation, lateral scans, manipulation trajectories
- **SubstrateBuilder.py** — Crystal slab generation for ionic crystals (NaCl, CaF2): flat slabs and step edges
- **surface_plots.py** — Matplotlib visualization for relaxation trajectories, lateral scans, manipulation trajectories, relaxed scans

## ContactSurface (static AFM prototype)

**Model:** `V(x,y,z) = Σ c_{ijk} B_i(x) B_j(y) φ_k(max(0, z - h(x,y)))` with doubling poly modes `t^(m_start·2^k)`; optional per-atom radial PIC sum.

**Key pieces:**
- `build_contact_height_map()` — B-spline field `h₀` on same xy grid as coeffs
- `ContactSurfaceCL` — GPU brute (`cs_brute_plqh_points`), separable CG fit (`fit_separable_cg`), PIC fit/eval (`fit_pic_cg`, `eval_pic_grid`)
- Kernel: `kernels/contact_surface.cl` (see `kernels/README.md`)

**Run diagnostic:**
```bash
python tests/testplot_contact_surface.py
```

**Artifacts:** `debug/testplot_contact_surface/contact_surface_comparison.png`, `contact_surface_summary.out`

**Caveats:** Fit must sample **multiple z planes** near `z_scan` (not a single slice) to constrain `Fz`; force convention `F = ∇E` (matches `getMorsePLQH`). Not yet wired into `AFMulator` / `RigidBodyAFM`.
