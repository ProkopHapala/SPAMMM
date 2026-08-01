# surfaces/

Substrate and sample interaction modeling for molecule-on-surface / AFM simulations.
**GridFF** precomputes dense 3D B-spline grids for periodic substrates.
**ContactSurface** is the compact quasi-2D replacement for `img_FF` during PP-AFM on
aperiodic rigid molecules — same relaxation loop, far fewer degrees of freedom.

Design spec: [doc/Topics/AFM/ContactSurface_Static.md](../../doc/Topics/AFM/ContactSurface_Static.md)  
Task SSOT: [doc/Tasks/Fast_2p5D_AFM_ContactSurface.md](../../doc/Tasks/Fast_2p5D_AFM_ContactSurface.md)  
Parity vs GridFF: [doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md](../../doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md)  
Caveats: [doc/Caveats.md](../../doc/Caveats.md) §6 · Debugging: [doc/Takeways.md](../../doc/Takeways.md)

## File index

- **ContactSurface.py** — GPU contact field: sphere-envelope `h₀` (`build_contact_height_map` / `eval_sphere_contact_height`); `ContactSurfaceCL` (brute, separable CG, PIC); `SeparableParams`, `PICParams`
- **GridFF.py** — PyOpenCL B-spline grid force field for periodic substrates (Pauli/London/Coulomb channels)
- **SurfaceEwald.py** — GPU 2D Ewald summation for electrostatic potentials/fields above periodic surfaces
- **Ewald2D.py** — NumPy 2D Ewald reference (plane-wave formulation, parity vs GPU)
- **Surface_utils.py** — GridFF metadata, load precomputed grids, visualization, atom-position sampling
- **GridFFRelaxedScan.py** — Relaxed PES scanning with full geometry relaxation at each grid point
- **FoldedRigid.py** — Folded-basis rigid-body simulation; versioned `typed_combined` and substrate-only `factorized_plqh` fits, constrained charge discretization, fit comparison harness, relaxation/manipulation, and CPU map helpers. Architecture/verification: [`FAF_Fit_Architecture.md`](../../doc/Tasks/FAF_Fit_Architecture.md)
- **surface_plots.py** — Matplotlib: FoldedRigid traj/scans + PairFF tip-pull movies; **map display must reuse Vispy `potential_to_rgba`** (`doc/Tasks/PairFF_MapDisplay_SSOT.md`)
- **SubstrateBuilder.py** — Crystal slab generation (NaCl, CaF₂): flat slabs and step edges

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

# --- Separable B-spline × poly (atom-scale nodes; sphere h₀) ---
sep = afm.fit_contact_surface(
    margin=4.0, bspl_dx=1.0, poly_R=4.0, poly_z0=0.0, m_start=4, nz=6,
    fit_z_adaptive=(0.05, 4.0, 0.1, 0.8),
    fit_dx=1.0, fit_force_weight=1.0,
    h0_mode='spheres', h0_R_scale=0.75,   # clamp in hard repulsion, not at well
)
FEs, pts = afm.run_scan_contact(nxy=(…), nz=25, dtip=-0.15, ...)
```

Assembly screening defaults: `--bspl-dx 1.0 --scan-dx 0.5 --h0-R-scale 0.75` (`run_assembly_afm.py`).

Legacy 3D reference: `setup_grid()` → `make_forcefield()` → `run_scan()`.

### Tests & review artifacts

| Script | Level | Output |
|--------|-------|--------|
| `tests/testplot_contact_surface.py` | L1+L2 | `debug/testplot_contact_surface/` — fit, parity, `--toys` |
| `tests/SPM/test_afm_contact_surface.py` | L0 | Force-stencil parity |
| `tests/SPM/testplot_afm_contact_surface.py` | L2 | PP Fz/df vs 3D |
| `run_assembly_afm.py --compare-dir` | L2 | helicene contact vs GridFF maps + E/Fz profiles |

### Key fit knobs

| Parameter | Separable | PIC | Notes |
|-----------|-----------|-----|-------|
| Lateral nodes | `bspl_dx` | `fit_dx` | **~1.0 Å** atom-scale (was 0.2 — too fine) |
| Image pixels | `scan_dx` (CLI) | same | **~0.5 Å** default in assembly |
| `h₀` | `h0_mode='spheres'`, `h0_R_scale=0.75` | — | Not `atom_z`; scale&lt;1 so clamp is repulsive |
| z fit range | `fit_z_adaptive` | same | Offsets **above contact** `h₀`, not bare zmax |
| Poly cutoff | `poly_R`, `poly_z0` | `poly_R` | |
| Modes | `m_start`, `nz` | same | |
| Regularization | global `0`, tiles `1e-2` | **`1e-2`** | PIC diverges at `1e-4` |
| Sample weights | Boltzmann on | **off** | See Takeways |
| Force loss | `fit_force_weight` | not yet | Planar `F_ref` upload critical |
