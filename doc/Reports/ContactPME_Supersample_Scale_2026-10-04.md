---
type: Report
title: contact_pme fit/scan scale and V_L supersampling
tags: [contact-pme, afm, speed, gridff, supersample, b-spline]
timestamp: 2026-10-04
---

# Contact-PME: whole-simulation scale, GridFF, and 1×1×1 vs supersampling

## Summary

On a square bilayer graphene sheet the contact-PME cost is the PP scan, not the fit. A denser sample of the smooth part `V_L`, projected back onto the same 1 Å cubic B-spline, does not reduce the corrugation error and does not change a relaxed df image. **1×1×1 (nodal collocation, no supersampling) is sufficient.** USER confirmed the relaxed df strip (2026-10-04).

The comparison is a normal PP-AFM scan with the SPM_CLI contract: `K_LAT = 0.5 N/m` (0.0312 eV/Å²), `K_RAD = 20 eV/Å²`, CO length `L = 3 Å`, FIRE, df columns 3.7–4.7 Å, amplitude 1 Å, gray, per-panel range, repulsive (positive df) bright. Tip charges are off on every row because `fit_contact_pme` rejects the CLI two-site tip. Artifact: `debug/testplot_pme_tiles/heights_df_strip.png`. Harness: `tests/SPM/testplot_pme_tiles.py --heights`.

## What was measured

Device: NVIDIA GTX 1650. Default `Δ_b = 2.0` (not changed). Pixel pitch 0.1 Å. Bottom graphene sheet is not contact core; the high PP window drops it.

### Fit vs scan (second run, 16×16 tiles, nz=21)

| side (Å) | atoms | pixels | mesh ms | core ms | fit ms | scan ms | total ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 194 | 176² | 1.3 | 1.3 | 2.6 | 60 | 63 |
| 24 | 418 | 256² | 1.5 | 1.1 | 2.6 | 88 | 91 |
| 32 | 810 | 336² | 2.1 | 1.6 | 3.7 | 156 | 160 |
| 48 | 1756 | 496² | 5.4 | 3.2 | 8.6 | 361 | 370 |
| 64 | 3232 | 656² | 12 | 5.7 | 18 | 729 | 747 |
| 80 | 4876 | 816² | 26 | 8.6 | 35 | 1223 | 1257 |

Scan tracks pixel count. Fit stays under 35 ms out to ~5000 atoms. Plot: `debug/testplot_pme_tiles/scale_graphene.png`.

### Why GridFF fit looks slow and PME fit does not

GridFF (`make_forcefield` → `evalMorseC_QZs_toImg`) evaluates the full Morse field on a 0.1 Å voxel image. The scan only interpolates that image. PME never builds that image: `fillContactPMEMeshVL` samples `V_L` once per 1 Å node, the GPU Thomas prefilter turns those samples into cardinal cubic coefficients, and the scan re-evaluates the cheap model (64-tap spline + compact cores) during relaxation.

| side (Å) | PME mesh ms | GridFF voxels | GridFF build ms | GridFF scan ms | PME scan ms |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.0 | 6.3e6 | 100 | 78 | 61 |
| 24 | 1.4 | 1.1e7 | 391 | 163 | 85 |
| 32 | 2.2 | 2.0e7 | 1338 | 277 | 161 |
| 48 | 5.0 | 4.0e7 | 5911 | 624 | 377 |

At 48 Å the GridFF build is ~9× the GridFF scan. Plot: `debug/testplot_pme_tiles/gridff_vs_pme.png`.

### Supersampling `V_L` onto the same 1 Å coefficients

Least-squares projection of fine samples (`h = 1, 1/2, 1/3`) onto the production cubic basis. `s = 1` matches the nodal prefilter to `max|Δc| = 2.5e-8`. Off-node force error vs direct `V_L` got worse, not better:

| s | samples (32 Å sheet) | max\|ΔE\| eV | max\|ΔF\| eV/Å |
|---:|---:|---:|---:|
| 1 | 4.2e4 | 5.3e-5 | 9.1e-5 |
| 2 | 3.2e5 | 1.6e-4 | 3.1e-4 |
| 3 | 1.1e6 | 1.4e-4 | 2.7e-4 |

Atom cores were **not** refit on the 3D samples. They stay the 32-point radial PAW fit (`cs_fit_core_paw`). Only the smooth-part spline changes. The relaxed df rows for 1×1×1, 2×2×2 and 3×3×3 match each other and match GridFF. Denser sampling does not remove the bumps.

### PAW core fit on the GPU

`cs_fit_core_paw`: one work-item per atom, 32 Chebyshev nodes, 5×5 Cholesky. At 1520 atoms the host loop was 861 ms and the GPU fit 1.2 ms, `max|Δc| = 6e-6` vs the float64 grid oracle. Non-paw modes still use host `fit_core_1d`.

## Open

- `Δ_b` default is still 2.0. Recommended locality/accuracy values (0.6 / 1.2) are not applied.
- Tiled-kernel escapes (~7%, suspected `tile_kz` off-by-one) are unchanged.
- Next eval speed lever is the tiled sphere solver, not a denser spline. On this machine the local kernel stops near 1365 atoms; tile sphere stays under 1 s out to 3232 atoms while bucket FIRE is 66 s. USER confirmed `debug/testplot_pme_tiles/solvers_scale.png`. See `doc/Reports/ContactPME_RelaxQuasiNewton_Sph_2026-10-03.md`.
