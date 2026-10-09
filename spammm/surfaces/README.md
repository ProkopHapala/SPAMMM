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

- **ContactSurface.py** — GPU contact field: sphere-envelope `h₀`; separable/PIC; **`ContactPMEParams`** for particle-mesh backend
- **PMESplit.py** — atomwise PAW soft-replacement long/short split (default `paw`); `precompute_split_cache`, closed-form a0. **Pairwise Morse+Q oracle only** — sampled fields (FDBM) have no per-atom `v_i(r)`; use `fit_morse_atom_params` refit or a field-space split (`doc/Caveats.md` §22)
- **CoarseMesh.py** — coarse 3D B-spline of V_L; batched prefilter; CPU raster (GPU fill lives in AFMulator / `fillContactPMEMeshVL`). **`fit_mesh_lsq_3d(R_ss, s, lam)`** — separable penalized LSQ on an s× supersampled node grid (per-axis `Bᵀ` projection + batched banded solves; s=1,λ=0 ≡ `_prefilter_3d`). Production-style coremesh mesh stage: omit too-close points (`r_min<2.5`, `z<z_min`) via `PICCore.inpaint_residual` harmonic fill (optional `seed=` from a node-level fit) + `fit_mesh_lsq_3d` — ~30× faster than the weighted CG and equal/better df for `A:s3p2+B:s1` (design doc §17; `fit_coremesh_lsq` CG remains for debugging)
- **Sampled-field fit (`CoarseMesh.fit_coremesh_lsq`)** — cores first, then one weighted joint solve over overlapping cores and mesh; C2 radial joins, raw five-power coefficients, sparse axis products and analytic E/F evaluation. Diagnostic: `python3 tests/SPM/testplot_fdbm_pme_debug.py cc --samples debug/testplot_fdbm_pme_debug/cc_h0.5/coremesh_samples.npz`. Cached C–C E improvement measured; real-field force/df and molecular generalization remain unverified. Design: [ContactPME_CoreMesh_Fit_Design.md](../../doc/Tasks/ContactPME_CoreMesh_Fit_Design.md).
- **PICCore.py** — compact residual core fit (`fit_core_1d`, powers 2…32); non-pairwise field variants: `fit_cores_paw_field` (damped-Jacobi radial oracle), `fit_cores_from_samples` (joint LSQ, incl. energy-space `E_hard` targets); core+mesh bases: `poly8` (`fit_core_poly_shell`/`eval_core_poly`, kernel `CS_PME_POLY_CORE`, 5 scalar slots) and **`poly8sp`** (`fit_core_sp_shell`/`eval_core_sp`, kernel `CS_PME_SP_CORE`, 3 s + 3·NP p slots/center — spec `A:s3p2+B:s1` → 9, `A:s3p3` → 12; `r_lo` = per-center radius scale in the atom w-channel). `fit_core_sp_shell(solver='gpu', afm=...)` runs the matrix-free GPU CGLS (`ContactSurfaceCL.fit_core_sp_cg`: `cs_sp_Av`/`cs_sp_Atv` owner-computes gather, no atomics, Jacobi column scaling; `eval_core_sp_gpu`) — O(ns+ncoef) memory, flat in molecule size. **`solver='gpugram'`** = direct normal-equation solve: GPU Gram assembly `G[:,j]=Atv(Av e_j)` over active columns + Jacobi-scaled Tikhonov (`ridge`≈1e-6 — matches the CGLS early-stop regularization; exact λ→0 hurts df) + host Cholesky — ~4× faster than CGLS, deterministic, equal/better df (design doc §18). `solver='gram'`/`'lstsq'` = CPU references. `min_dist_to_atoms` (cKDTree) for shell masks — never broadcast `P[:,None]-apos`. Design: `doc/Tasks/ContactPME_CoreMesh_Fit_Design.md` §16.
- **GridFF.py** — PyOpenCL B-spline grid force field for periodic substrates (Pauli/London/Coulomb channels)
- **SurfaceEwald.py** — GPU 2D Ewald summation for electrostatic potentials/fields above periodic surfaces
- **Ewald2D.py** — NumPy 2D Ewald reference (plane-wave formulation, parity vs GPU)
- **Surface_utils.py** — GridFF metadata, load precomputed grids, visualization, atom-position sampling
- **GridFFRelaxedScan.py** — Relaxed PES scanning with full geometry relaxation at each grid point
- **FoldedRigid.py** — Folded-basis rigid-body simulation; versioned `typed_combined` and substrate-only `factorized_plqh` fits, constrained charge discretization, fit comparison harness, relaxation/manipulation, and CPU map helpers. Architecture/verification: [`FAF_Fit_Architecture.md`](../../doc/Tasks/FAF_Fit_Architecture.md)
- **CoreBasisStudy.py** — experimental angular core-basis machinery for the core+mesh FDBM fit: spec grammar `A:s3p2+B:s1` (atoms/bond centers × s/p/d radials `(1-r/R)₊⁸`), design matrix, weighted LSQ with force rows, **core-only residual study** (`core_residual_study`, high-pass smoothness metric). Survey driver: `tests/SPM/testplot_coremesh_basis_survey.py` (`--rcuts/--rins/--wFs`).
- **surface_plots.py** — Matplotlib: FoldedRigid traj/scans + PairFF tip-pull movies; **canonical core-residual figure `plot_core_residual_study`** (rows=fit variants, cols=E xz + Fz slices, shared clim; NaN=outside fit domain); map display must reuse Vispy `potential_to_rgba` (`doc/Tasks/PairFF_MapDisplay_SSOT.md`)
- **SubstrateBuilder.py** — Crystal slab generation (NaCl, CaF₂): flat slabs and step edges

## contact_pme (particle-mesh)

`V ≈ V_mesh + Σ V_core`. CLI: `run_spm.py afm --model contact_pme`. Plan: [`ContactSurface_PME_ParallelPlan.md`](../../doc/Tasks/ContactSurface_PME_ParallelPlan.md). Reports: [`ContactPME_PAW_AFM_MemSpeed_2026-08-11.md`](../../doc/Reports/ContactPME_PAW_AFM_MemSpeed_2026-08-11.md), [`ContactPME_Supersample_Scale_2026-10-04.md`](../../doc/Reports/ContactPME_Supersample_Scale_2026-10-04.md) (1×1×1 spline is enough). Audit: [`AFM_ContactSurface.md`](../../doc/TopicalAudit/AFM_ContactSurface.md).

| Stage | Device | Entry |
|-------|--------|-------|
| FIT mesh V_L | GPU | `fillContactPMEMeshVL` via `AFMulator.fit_contact_pme` |
| FIT core LS | GPU for paw, host otherwise | `cs_fit_core_paw` / `PICCore.fit_core_1d` |
| SCAN FIRE | GPU | `relaxStrokesTiltedContactPMELocal` (`core_backend='local'`) |

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
