---
type: Caveats
title: SPAMMM scientific / numerical caveats
tags: [caveats, FDBM, ES, grids, NA, multipoles]
timestamp: 2026-07-24
---

# Caveats (global)

Recurring traps that look like “bugs in Poisson / tip / plot” but are usually **Δρ construction** or **grid convention**. Read this before “fixing” AFM electrostatic asymmetry.

---

## 1. All-electron ρ − crude NA → fake multipoles → L–R V_ES

**Symptom:** Cube-row AFM / V_ES looks strongly asymmetric (horizontal or diagonal background slope); DFTB row with the same tip and Poisson is nearly symmetric.

**Cause:** `V_ES = fft_poisson(Δρ)` with `Δρ = ρ_N − ρ_NA`. If `ρ_NA` is a synthetic Gaussian or clamp→compact core, its multipoles **do not** match all-electron `ρ_N`. Leftover dipole (even with ∫Δρ≈0) dominates far-field under PBC Poisson. AFM tip convolution makes this look like a huge L–R force asymmetry.

**Evidence (pentacene, 2026-07-24, native cube, no dipole strip):**

| Δρ recipe | \|p_xy\| (e·Å) | V mX @ z_mol+1 |
|-----------|----------------|----------------|
| **ρ_N − ρ_NA.cube** (pySCF) | **~0.01** | **~0.035** (clean) |
| ρ_N − Gauss CORNER | ~1.9 | ~0.80 (strong x-slope) |
| ρ_N − Gauss CENTER | ~10 | ~0.99 (diagonal slope) |
| clamp → compact | ~1.5 | ~0.60 (better, still bad) |

Decomposition: `p(ρ_N) ≈ p(ρ_NA.cube)` → cancel. Our Gauss/compact NA multipoles ≠ `ρ_N` → leftover dipole.

**Do:** Prefer `Δρ = ρ_N − ρ_NA.cube` (DFT-like / DFTB-like) when the NA cube exists.  
**Do not:** Treat `strip_monopole_dipole` as a fix (DEFAULT OFF — masks the symptom).  
**Do not:** Blame `fft_poisson` alone when DFTB V_ES is fine with the same solver.

**Canonical plot (preserve / regenerate):**  
`AFM_utils.plot_cube_delta_rho_na_origin_diag` →  
`debug/fdbm_fukui_panel_flat/<mol>/es_diag/dipole_origin_bisect.png`  
CLI: `python run_spm.py es-diag --molecule pentacene`

**Reports:** [Reports/Fukui_FDBM_panel_notes_2026-07-23.md](Reports/Fukui_FDBM_panel_notes_2026-07-23.md) §1b · pyridine: [Reports/Kriging_DFT_vs_DFTB_FDBM_pyridine.md](Reports/Kriging_DFT_vs_DFTB_FDBM_pyridine.md)

---

## 2. Corner vs center voxel sampling

| Path | Sample positions |
|------|------------------|
| `make_gaussian_rho_na`, OpenCL `add_gaussian`, `grid_moments` | **corners** `origin + i·h` |
| `GridsOCL.project_density` / `grid_moments_centers` | **centers** `origin + (i+½)·h` |

Mixing conventions (e.g. corner-sampled ρ_N minus center-sampled Gauss) changes core alignment by ~½ voxel and can **qualitatively** change V_ES slope direction (CORNER → x-slope; CENTER → diagonal). Charge-neutral Δρ moments are less convention-dependent, but **ρ − wrong-NA** is not.

Always state which convention a moment or NA builder uses.

---

## 3. Scipy *sample* vs GridsOCL *project* for density transfer

| Method | Code | Conserves ∫ρ, p? | Use for Δρ / charge? |
|--------|------|------------------|----------------------|
| scipy sample | `resample_field_to_grid` / `map_coordinates` | **No** | **Forbidden** |
| trilinear scatter | `project_density_to_grid` → `grids.cl` | **Yes** (in-bounds) | **Required** |

Resampling onto a taller FDBM box must **project**, not sample. Sample can inject extra multipole error on top of §1.

---

## 4. Related but secondary

- **Tip innocent** when tip XY mirror metrics ~1e−7 and peak at (0,0,0) after roll — check tip first, then sample Δρ.
- **Manual monopole/dipole strip** about COM: diagnostic only; never default “production fix”.
- **z-box asymmetry** (extra vacuum on +z only): can bias vertical fields; less often the main XY L–R dipole vs §1.
- **OpenCL device:** NVIDIA first; never report PoCL timings as GPU — `.cursor/rules/opencl-nvidia-gpu.mdc`.

---

## Index of code entry points

| Topic | Location |
|-------|----------|
| NA-origin diagnostic plot | `spammm/SPM/AFM_utils.py` → `plot_cube_delta_rho_na_origin_diag` |
| ES chain plots / mirror metrics | `plot_cube_es_chain_diag`, `mirror_asymmetry_2d` |
| Cube load + synthetic NA | `get_density_from_cube`, `make_gaussian_rho_na` |
| Clamp→compact | `delta_rho_clamp_compact_na` |
| Cube → FDBM | `allelectron_cube_to_fdbm_grid` |
| Project / moments | `spammm/utils/GridsOCL.py` |
| CLI | `run_spm.py es-diag` → `tests/SPM/testplot_fdbm_relax.py` |
