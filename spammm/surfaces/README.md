# surfaces/

Substrate and sample interaction modeling for molecule-on-surface / AFM simulations.
**GridFF** precomputes dense 3D B-spline grids for periodic substrates.
**ContactSurface** is the compact quasi-2D replacement for `img_FF` during PP-AFM on
aperiodic rigid molecules — same relaxation loop, far fewer degrees of freedom.

Design spec: [doc/Topics/AFM/ContactSurface_Static.md](../../doc/Topics/AFM/ContactSurface_Static.md)  
Task SSOT: [doc/Tasks/Fast_2p5D_AFM_ContactSurface.md](../../doc/Tasks/Fast_2p5D_AFM_ContactSurface.md)  
Debugging pitfalls: [doc/Takeways.md](../../doc/Takeways.md) (z alignment, F_ref layout, reg, weighting)

## File index

- **ContactSurface.py** — GPU contact field: `ContactSurfaceCL` (brute reference, separable CG fit, PIC CG fit, eval); `SeparableParams`, `PICParams`, `select_contact_atoms`, `build_pic_buckets`, `build_contact_height_map`
- **GridFF.py** — PyOpenCL B-spline grid force field for periodic substrates (Pauli/London/Coulomb channels)
- **SurfaceEwald.py** — GPU 2D Ewald summation for electrostatic potentials/fields above periodic surfaces
- **Ewald2D.py** — NumPy 2D Ewald reference (plane-wave formulation, parity vs GPU)
- **Surface_utils.py** — GridFF metadata, load precomputed grids, visualization, atom-position sampling
- **GridFFRelaxedScan.py** — Relaxed PES scanning with full geometry relaxation at each grid point
- **FoldedRigid.py** — Folded-basis rigid-body simulation (fit potentials, relaxation, manipulation, `setup_rigid_folded`)
- **SubstrateBuilder.py** — Crystal slab generation (NaCl, CaF₂): flat slabs and step edges
- **surface_plots.py** — Matplotlib plots for relaxation trajectories, lateral scans, manipulation

## Contact surface — variants (2.5D)

| | (i) Separable + \(h_0\) | (ii) PIC radial | (iii) Hybrid / other |
|--|------------------------|-----------------|----------------------|
| **Form** | `Σ c_ijk B_i(x) B_j(y) φ_k(dz−h₀)` | `Σ_i Σ_m c_im φ_m(\|r−r_i\|)` | Coarse A + PIC residual, or folded z-basis family — **not one API yet** |
| **Storage** | `ncx×ncy×nz_modes` (~10⁴–10⁵) | `nat×nmodes` (~10²–10³) | TBD |
| **Best for** | Moderate scan patches, smooth corrugation | Many surface atoms, large xy extent | Large systems / residual correction |
| **Fit** | `fit_separable_cg` — Boltzmann + force rows OK | `fit_pic_cg` — unweighted, `reg≈1e-2` | — |
| **PP scan** | `run_scan_contact` → `relaxStrokesTiltedContact` | `run_scan_pic` → `relaxStrokesTiltedPIC` | — |
| **Kernel** | `evalSeparableBsplinePoly`, `cs_sep_Av/Atv*` | `evalRadialPIC`, `cs_pic_Av/Atv*`, `cs_pic_eval_tile16` | — |

Shared: brute Morse reference (`cs_brute_afm_morse_c_points` via AFMulator), probe-z
convention, `F = −∇E`, particle-in-cell only on PIC path. See task file for finish criteria.

### Library usage (AFMulator)

```python
from spammm.SPM.AFM import AFMulator

afm = AFMulator(use_morse=True, use_fire=False)
afm.load_molecule('data/xyz/PTCDA.xyz')
afm.assign_params(params_path='data/ElementTypes.dat', tip_R=0.0, tip_E=1.0)

# --- Separable B-spline × poly ---
sep = afm.fit_contact_surface(
    margin=4.0, bspl_dx=0.2, poly_R=5.0, poly_z0=1.0, m_start=4, nz=6,
    fit_z_adaptive=(1.0, 6.0, 0.1, 1.0),   # z offsets [Å] above zmax, adaptive dz
    fit_dx=0.2, fit_force_weight=1.0,       # E + Fx,Fy,Fz rows, RMS-equalized
)
FEs, pts = afm.run_scan_contact(nxy=(99, 75), nz=25, dtip=-0.15, ...)

# --- PIC radial (contact atoms) ---
pic = afm.fit_pic_contact_surface(
    margin=4.0, poly_R=5.0, m_start=4, nz=5, cell_size=10.0,
    z_local=1.2, xy_radius=14.0, reg=1e-2,
    fit_z_adaptive=(1.0, 6.0, 0.1, 1.0), fit_dx=0.2,
)
FEs_pic, pts = afm.run_scan_pic(nxy=(99, 75), nz=25, dtip=-0.15, ...)
```

Legacy 3D reference: `setup_grid()` → `make_forcefield()` → `run_scan()`.

### Tests & review artifacts

| Script | Level | Output |
|--------|-------|--------|
| `tests/testplot_contact_surface.py` | L1+L2 | `debug/testplot_contact_surface/` — fit, z-alignment, separable + PIC parity, PP relaxed |
| `tests/SPM/test_afm_contact_surface.py` | L0 | Force-stencil parity, separable `run_scan_contact` smoke |
| `tests/SPM/testplot_afm_contact_surface.py` | L2 | `debug/testplot_afm_contact_surface/` — PP Fz/df vs 3D |

Run full visual stack: `RUN_CONTACT_PP=1 python tests/testplot_contact_surface.py`

### Key fit knobs

| Parameter | Separable | PIC | Notes |
|-----------|-----------|-----|-------|
| Lateral sample/grid step | `bspl_dx`, `fit_dx` | `fit_dx` | 0.2 Å typical for PTCDA |
| z fit range | `fit_z_adaptive` | same | Probe z = `zmax + offset`, not tip height |
| Poly cutoff | `poly_R`, `poly_z0` | `poly_R` | `t = 1 − clamp((dz−z_poly0)/Rc,0,1)` |
| Modes | `m_start`, `nz` | `m_start`, `nz` | Doubling powers `t^(m·2^k)` |
| Regularization | global `0`, tiles `1e-2` | **`1e-2`** | PIC diverges at `1e-4` |
| Sample weights | Boltzmann on | **off** (logged only) | See Takeways |
| Force loss | `fit_force_weight` | not yet | Planar `F_ref` upload critical |
