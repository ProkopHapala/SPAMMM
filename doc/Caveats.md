---
type: Caveats
title: SPAMMM scientific / numerical caveats
tags: [caveats, FDBM, ES, grids, NA, multipoles, anisotropic, contact-surface, AFM]
timestamp: 2026-07-24
---

# Caveats (global)

Recurring traps that look like “bugs in Poisson / tip / plot” but are usually **Δρ construction** or **grid convention**. Read this before “fixing” AFM electrostatic asymmetry.

---

## 1. Anisotropic DFT cubes — never collapse `(sx,sy,sz)` to a mean

**Primary cause of the huge pentacene V(N−Gauss) dipole (2026-07-24).**

pySCF cubes often have **~0.1–0.2%** axis anisotropy from cell rounding. That looks “isotropic enough” but **is not**:

- Density samples live at `origin + (i·sx, j·sy, k·sz)`.
- Building Gauss/compact NA with `step = mean(sx,sy,sz)` places nuclei on a **warped** lattice relative to ρ.
- Error grows with grid index (≲0.2 voxel across pentacene) → fake Δρ dipole ~**1.9 e·Å**.

| Δρ (pentacene native) | \|p_xy\| | V mX @ z+1 |
|-----------------------|----------|------------|
| N − NA_cube | ~0.01 | ~0.035 |
| N − Gauss **mean step** (old bug) | ~1.9 | ~0.80 |
| N − Gauss **true (sx,sy,sz)** | ~0.28 | ~0.42 |
| N − clamp **element-invariant** compact | ~0.018 | ~0.033 |

**Fix:** `get_density_from_cube` keeps `step` as `(3,)`; `make_gaussian_rho_na` / `fft_poisson_cpu` / GridsOCL already accept `(3,)`.

**FDBM pipeline readiness:**

| Stage | Anisotropic? | Notes |
|-------|--------------|-------|
| Cube load / native NA / native Poisson | **Must keep (3,)** | Fixed 2026-07-24 |
| `GridsOCL.project_density` | **Yes** — `(3,)` `step_s` / `step_d` | Already |
| FDBM dest (`make_fdbm_grid_com_zsym`, DFTB) | **Isotropic by design** | OK — project cube→FDBM |
| GPU `AFMulator` / fused ES | Scalar `step` | Fine on isotropic dest |

Do **not** accept mild aniso and then silently use the mean (old bug).

---

## 2. All-electron cores on coarse cubes → fake Δρ dipole

After §1, raw `ρ_N − Gauss` still leaves \|p\|~0.28 on pentacene. That is **not** “wrong σ”. At ~0.15 bohr, C/N/O 1s cusps are under-resolved; equivalent atoms get different sampled peaks / `Q_rem_i` purely from sub-voxel phase.

**Production recipe (SSOT):** `delta_rho_clamp_compact_na(..., q_na_mode='element_mean')`

```text
ρ_ps   = soft_clamp(ρ_SCF)                 # kill aliased cusps
Qrem_Z = mean_i(Q_rem_i for atoms of Z)    # never use noisy Q_rem_i as q_i
q_Z    = Z − Qrem_Z  (+ charge close)      # equal elements → equal q
Δρ_ES  = ρ_ps − Σ_i compact(q_{Z_i}, R_i)
```

- Legacy `q_na_mode='per_atom'` (`q_i=Z_i−Q_rem_i`) **recreates** the alias dipole — do not use for production.
- `rho_NA.cube` is an **optional diagnostic** (cancels the same alias by construction), **not** a required input.
- Manual `strip_monopole_dipole` DEFAULT **OFF** (symptom mask).

Pentacene check: element-mean \|p_xy\| ~0.018 ≈ NA_cube control ~0.014 (vs ~0.28 legacy).

**Canonical plot:** `plot_cube_delta_rho_na_origin_diag` →  
`debug/fdbm_fukui_panel_flat/<mol>/es_diag/dipole_origin_bisect.png`  
CLI: `python run_spm.py es-diag --molecule pentacene`

**Full write-up:** [Reports/Cube_ES_DeltaRho_NA_dipole.md](Reports/Cube_ES_DeltaRho_NA_dipole.md) · handoff: [Reports/Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md](Reports/Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md)

---

## 3. Corner/node vs center voxel sampling

| Path | Sample positions |
|------|------------------|
| Cubes, `make_gaussian_rho_na`, `grid_moments` | **nodes** `origin + i·h` |
| Low-level `GridsOCL.project_density` kernel | **centers** `origin + (i+½)·h` |

`project_density_to_grid(..., src_convention='nodes')` (default) shifts `origin_src − ½ step` before calling the kernel. Use `'centers'` only for true cell-centered fields. Mixing conventions shifts moments by `q·step/2` and can change V slope direction.

ρ_NA **must** be atom-centered (same nuclei as the cube header). Mirror metrics must use **fractional** physical centers (`mirror_asymmetry_2d`), not integer pixel flips.

---

## 4. Scipy *sample* vs GridsOCL *project* for density transfer

| Method | Code | Conserves ∫ρ, p? | Use for Δρ / charge? |
|--------|------|------------------|----------------------|
| scipy sample | `resample_field_to_grid` / `map_coordinates` | **No** | **Forbidden** |
| trilinear scatter | `project_density_to_grid` → `grids.cl` | **Yes** (in-bounds) | **Required** |

---

## 6. Contact-surface 2.5D vs GridFF (classical Morse AFM)

Quasi-2D contact-sep is meant to **approximate** Morse(+Coulomb), not invent sharper physics. Recurring traps:

| Trap | Symptom | Fix / rule |
|------|---------|------------|
| `h₀ = max atom z` | Soft / long-ranged E(z); “too close” images | `h0_mode='spheres'` — ray vs `R=scale×R0` spheres |
| `h0_R_scale=1.0` | Repulsion plateaus at well | Default **0.75** — clamp in hard wall |
| `bspl_dx` / `scan_dx` ≪ 1 Å | Razor atom pins; against coarse-grain design | Defaults **1.0 / 0.5 Å** (atom-scale nodes/pixels) |
| Trusting contact XY vs GridFF without profiles | Looks “better resolved” | Always check E/Fz(z) vs **brute**; GridFF ≈ brute for classical FF |

**Report:** [Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md](Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md) · helicene session [Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md](Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md)

---

## 7. Related but secondary

- Tip innocent when tip XY mirror ~1e−7 and peak at (0,0,0) after roll.
- z-box asymmetry (extra vacuum on +z only) can bias vertical fields.
- OpenCL: NVIDIA first — `.cursor/rules/opencl-nvidia-gpu.mdc`.

---

## Index of code entry points

| Topic | Location |
|-------|----------|
| NA-origin diagnostic plot | `spammm/SPM/AFM_utils.py` → `plot_cube_delta_rho_na_origin_diag` |
| ES chain plots / mirror metrics | `plot_cube_es_chain_diag`, `mirror_asymmetry_2d` |
| Cube load + synthetic NA | `get_density_from_cube`, `make_gaussian_rho_na` |
| Clamp→compact (element-mean) | `delta_rho_clamp_compact_na` (`q_na_mode='element_mean'`) |
| Cube → FDBM | `allelectron_cube_to_fdbm_grid` |
| Project / moments | `spammm/utils/GridsOCL.py` |
| CLI | `run_spm.py es-diag` → `tests/SPM/testplot_fdbm_relax.py` |
| Contact-surface fit / scan | `spammm/SPM/AFM.py` → `fit_contact_surface`, `run_scan_contact` |
| Assembly AFM compare | `run_assembly_afm.py --compare-dir` |
