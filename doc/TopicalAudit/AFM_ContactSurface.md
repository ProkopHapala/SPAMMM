---
type: TopicalAudit
title: AFM Contact Surface + contact_pme
tags: [afm, contact-surface, contact-pme, paw, morse, opencl, parity]
timestamp: 2026-10-04
---

# Topical Audit: AFM Contact Surface + contact_pme

## Summary

Two compact replacements for dense 3D `img_FF` in classical Morse(+Q) PP-AFM:

1. **Quasi-2D contact surface** — separable B-spline(xy)×z-modes or radial PIC on a height map `h₀` (2026-07 path).
2. **contact_pme** — particle-mesh analogy: coarse 3D B-spline of soft long-range `V_L` + compact PIC cores of residual `v_S`, after atomwise **PAW** soft-replacement split (`Δ_in=1.0`). Target: ~10³× field memory for ML-scale molecule sets; CLI `run_spm.py afm --model contact_pme`.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| OpenCL | `kernels/contact_surface.cl` | active | Separable/PIC + **PME**: `evalContactPME`/`Local`, `relaxStrokesTiltedContactPME`/`Local`, `fillContactPMEMeshVL`, **fast relaxers** `…LocalQN` (secant→FD-Newton→FIRE) and `…LocalSph` (sphere-constrained, 2 soft DOF) |
| Python | `spammm/surfaces/PMESplit.py` | active | PAW/hermite/plateau/rho split; `precompute_split_cache`; closed-form a0 |
| Python | `spammm/surfaces/CoarseMesh.py` | active | CPU V_L raster + batched prefilter (oracle / fallback) |
| Python | `spammm/surfaces/PICCore.py` | active | `fit_core_1d` (host, non-paw); paw oracle `fit_core_paw_grid`; production paw fit is GPU `cs_fit_core_paw` |
| Python | `spammm/surfaces/ContactSurface.py` | active | Quasi-2D + `ContactPMEParams` |
| Python | `spammm/SPM/AFM.py` | active | `fit_contact_pme` (GPU mesh), `run_scan_contact_pme` (`core_backend` local/bucket, `relax_mode` fire/qn/sph + `qn_cap`/`qn_conv`) |
| Python | `spammm/SPM/AFM_utils.py` | active | `run_contact_pme_pp_afm` — CLI SSOT, forces `local` |
| CLI | `run_spm.py afm --model contact_pme` | active | Same ScanSpec / strip plots as Morse/FDBM |
| Design | `doc/Tasks/ContactSurface_PME_ParallelPlan.md` | active | Parallel plan + harness packet |
| Report | `doc/Reports/ContactPME_PAW_AFM_MemSpeed_2026-08-11.md` | active | Memory/speed SSOT |
| Report | `doc/Reports/ContactPME_Split_Rcut_Locality_2026-10-04.md` | active | Split mechanics, Rcut(Δ_b) sweep, layer pruning, mesh-first-residual experiment |
| Report | `doc/Reports/ContactPME_RelaxQuasiNewton_Sph_2026-10-03.md` | active | QN/Sph relaxers, bench vs FIRE |
| Report | `doc/Reports/ContactPME_Supersample_Scale_2026-10-04.md` | active | Fit vs scan scale, GridFF, 1×1×1 sufficient (USER 2026-10-04) |
| Design | `doc/Topics/AFM/ContactSurface_Static.md` | active | Quasi-2D physics + API |
| Report | `doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md` | active | Quasi-2D vs GridFF parity |

## Parity Status

| Pair | Tolerance / metric | Test / artifact | Status |
|------|--------------------|-----------------|--------|
| Separable eval vs force stencil | RMSE < 1e-4 | `tests/SPM/test_afm_contact_surface.py` | verified (L0) |
| PAW molecule vs Morse+Q (PLQH) | relE ≲1%, relF ≲7% @ z+4.5 | `wave2_paw_mol/*_paw_vs_plqh_h4.5.png` | L0+L2 regenerated |
| local vs bucket batch eval | max\|ΔE\|,‖ΔF‖ ≲ 2e-6 | harness | exact / ~1e-9 |
| local vs bucket FIRE FE | gate ≤2e-5 | pyridine exact; PTCDA sparse ~2e-3 float32 drift | reported |
| GPU vs CPU mesh coeffs | ~1e-8 | `fillContactPMEMeshVL` vs `build_coarse_mesh` | verified |
| GPU paw core vs float64 Chebyshev | max\|Δc\| 6e-6 (na=1520) | `tests/SPM/test_afm_contact_surface.py` `test_contact_pme_core_fit_gpu_parity` | measured |
| 2×/3× V_L supersample vs 1×1×1 | off-node \|ΔF\| worse (9e-5 → 3e-4 eV/Å); relaxed df rows match | `tests/SPM/testplot_pme_tiles.py --heights` | **USER: 1×1×1 sufficient (2026-10-04)** |
| CLI AFM strips | visual | `wave2_afm_cli/*/compare_per_image.png` | **USER confirmed OK (2026-08-11)** |
| `sph` relax vs FIRE | NCC(Fx) 0.998–0.9999, max\|ΔF\|≈0.003–0.014 eV/Å | `invPPAFM testplot_artifact_atlas.py --bench-qn` | measured, USER review pending |
| `qn` relax vs FIRE | NCC(Fx) 0.998, max\|ΔF\|≈0.024 eV/Å | same | measured, superseded by sph |

## Performance (RTX 3090, 2026-08-11)

| Cost | pyridine | PTCDA |
|------|----------|-------|
| Field resident | ~33 KB (~924× vs Morse@0.1) | ~53 KB (~1200×) |
| SCAN kernel-only (local) | 1.87 ms | 15.65 ms |
| SCAN wall (after pts-loop fix) | ~10 ms | ~20 ms |
| FIT (GPU mesh + host core LS) | ~11 ms | ~36 ms |

### SCAN relax solvers (GTX 1650, PTCDA 240²×31, 2026-10-03)

| `relax_mode` | wall | kernel | NCC(Fx) vs FIRE | note |
|---|---|---|---|---|
| `fire` (default) | ~95 ms | ~87 ms | — | mean ~7 evals/cell |
| `sph` | **~25 ms** | ~15 ms | 0.998–0.9999 | K_RAD→∞ caveat; ≤20 iters/slice |
| `qn` | ~33 ms | ~24 ms | 0.998 | superseded by sph |

Wall = ~1.2 evals/cell floor + ~10 ms host (FEs D2H 29MB, telemetry, postflight).

### Solver scaling (square bilayer, GTX 1650, CLI scan, 2026-10-04)

Local kernel = 36 B/atom in local memory, mesh left global. Device local is 48 KB, so it stops near 1365 atoms (1756 atoms needs 63 KB). Tile sphere preloads the workgroup atoms plus a short mesh window (24 Å sheet: 4.1 KB, 10×10×5 nodes).

| atoms | local FIRE | local QN | local sphere | tile sphere | bucket FIRE |
|---:|---:|---:|---:|---:|---:|
| 194 | 373 ms | 80 ms | 78 ms | 40 ms | |
| 418 | 2.22 s | 476 ms | 512 ms | 86 ms | |
| 810 | 12.6 s | 2.92 s | 3.31 s | 182 ms | |
| 1756 | does not fit | | | 467 ms | 37.4 s |
| 3232 | does not fit | | | 881 ms | 66.0 s |

max|Δdf| vs FIRE: sphere 1.4–1.6×10⁻², quasi-Newton 2.1–3.2×10⁻². **USER confirmed** `debug/testplot_pme_tiles/solvers_scale.png`. Harness: `tests/SPM/testplot_pme_tiles.py --solvers`. Full table in `doc/Reports/ContactPME_RelaxQuasiNewton_Sph_2026-10-03.md`.

### Split / Rcut mechanics (graphene 2L+N, 468 atoms, 2026-10-04)

- `r_b = R0 + Δ_b` is Rcut: join point of the PAW even-poly `v_L` AND the core
  support end (basis `t^(2^n)` → 0 exactly there). `Δ_b` exposed as
  `fit_contact_pme(delta_b=...)`; default still 2.0.
- Residual `v_S` is **invariant to Rcut** — dead by r≈3.8–4.0 — so r_b is a pure
  locality knob; smallest-safe ≈4.0 Å (Δ_b≈0.6, halo −45%, max|ΔE| 2.6e-4).
  Accuracy optimum Δ_b≈1.2 (r_b=4.58, 1.0e-4). h_mesh 0.5 vs 1.0 → no change.
- Field split: V_mesh = flat background; V_core = ALL atomic corrugation.
- Mesh-first residual targeting (fit core to `v − V_mesh`) **fails** — mesh error is
  anisotropic + neighbor-contaminated → radial basis injects error (300× worse).
- Bottom layer unreachable at proper PP heights → z-screenable from core lists
  (tile z-box bug fixed: was tip-top, now PP reach).
- df plotting: `compute_df_amp_z` needs ascending-z stacks; df wants gray + per-panel
  autoscale (symmetric norm and/or flipped z hides contrast).
- WG tiles on the same flake (8×4 px, ρ=1 Å): in-box and ρ-margin are independent of
  Rcut (mean 3.4 and 10.1). At Δ_b=1.2 the worst tile preloads 128 atoms
  (0 in-box + 20 margin + 108 halo) = 7.55 KB local, 16% of 48 KB. Mesh slab
  13×10×6 is 3.05 KB and does not shrink with Rcut. 54/128 are the bottom layer
  because the full-stroke z-box reaches z=2.25 Å.
- Full report: `doc/Reports/ContactPME_Split_Rcut_Locality_2026-10-04.md`.
- The WG-tile atom counts in the bullet above used a stretched scan grid and counted the bottom sheet. They are not the AFM partition. Production core lookup is PIC (`cell = r_b`, query reads 3×3). A 0.1 Å pixel makes a 16×16 workgroup 1.6×1.6 Å. With the PP z-window above the top sheet the tile list contains no bottom-layer atoms.

### Speed and supersampling (square bilayer, GTX 1650, 2026-10-04)

- Whole simulation: scan is the cost. At 4876 atoms the fit is 35 ms and the scan is 1.2 s. GridFF's expensive step is the 0.1 Å voxel build, which PME never does.
- Projecting 2×2×2 or 3×3×3 samples of `V_L` onto the same 1 Å cubic coefficients increases off-node force error. Cores stay the radial fit. Relaxed df (CLI springs, FIRE, gray) is unchanged. **1×1×1 is sufficient** — USER confirmed `heights_df_strip.png`.
- Report: `doc/Reports/ContactPME_Supersample_Scale_2026-10-04.md`.

## Open Issues

- [x] USER confirm regenerated CLI AFM strips (`wave2_afm_cli`) — OK 2026-08-11
- [ ] Non-paw core fit is still host `fit_core_1d`. Paw production path is GPU `cs_fit_core_paw` (measured, not a separate USER sign-off)
- [ ] Δ_b default still 2.0 — user decision pending: 0.6 (r_b≈4.0, halo −45%) vs 1.2 (best accuracy) — see 2026-10-04 report
- [ ] tile scan escapes at mesh-node-boundary z slices (~7% → global fallback; correct but wasteful) — suspected tile_kz slab off-by-one
- [ ] PTCDA FIRE local-vs-bucket sparse float32 drift (p99 fine; max ~2e-3)
- [~] Quasi-2D XY sharpness vs GridFF (older path; separate from contact_pme)
- [~] `sph` relaxer = `K_RAD→∞` (no radial compression, ~0.1 Å deep contact) — USER confirm df parity before pipeline use; bucket backend has no qn/sph variant
- ND `--contact-surface` flag still open for quasi-2D
