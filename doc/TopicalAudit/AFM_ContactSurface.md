---
type: TopicalAudit
title: AFM Contact Surface + contact_pme
tags: [afm, contact-surface, contact-pme, paw, morse, opencl, parity]
timestamp: 2026-08-11
---

# Topical Audit: AFM Contact Surface + contact_pme

## Summary

Two compact replacements for dense 3D `img_FF` in classical Morse(+Q) PP-AFM:

1. **Quasi-2D contact surface** — separable B-spline(xy)×z-modes or radial PIC on a height map `h₀` (2026-07 path).
2. **contact_pme** — particle-mesh analogy: coarse 3D B-spline of soft long-range `V_L` + compact PIC cores of residual `v_S`, after atomwise **PAW** soft-replacement split (`Δ_in=1.0`). Target: ~10³× field memory for ML-scale molecule sets; CLI `run_spm.py afm --model contact_pme`.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| OpenCL | `kernels/contact_surface.cl` | active | Separable/PIC + **PME**: `evalContactPME`/`Local`, `relaxStrokesTiltedContactPME`/`Local`, `fillContactPMEMeshVL` |
| Python | `spammm/surfaces/PMESplit.py` | active | PAW/hermite/plateau/rho split; `precompute_split_cache`; closed-form a0 |
| Python | `spammm/surfaces/CoarseMesh.py` | active | CPU V_L raster + batched prefilter (oracle / fallback) |
| Python | `spammm/surfaces/PICCore.py` | active | `fit_core_1d` doubling-power residual (host LS) |
| Python | `spammm/surfaces/ContactSurface.py` | active | Quasi-2D + `ContactPMEParams` |
| Python | `spammm/SPM/AFM.py` | active | `fit_contact_pme` (GPU mesh), `run_scan_contact_pme` (`core_backend` local/bucket) |
| Python | `spammm/SPM/AFM_utils.py` | active | `run_contact_pme_pp_afm` — CLI SSOT, forces `local` |
| CLI | `run_spm.py afm --model contact_pme` | active | Same ScanSpec / strip plots as Morse/FDBM |
| Design | `doc/Tasks/ContactSurface_PME_ParallelPlan.md` | active | Parallel plan + harness packet |
| Report | `doc/Reports/ContactPME_PAW_AFM_MemSpeed_2026-08-11.md` | active | Memory/speed SSOT |
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
| CLI AFM strips | visual | `wave2_afm_cli/*/compare_per_image.png` | **USER confirmed OK (2026-08-11)** |

## Performance (RTX 3090, 2026-08-11)

| Cost | pyridine | PTCDA |
|------|----------|-------|
| Field resident | ~33 KB (~924× vs Morse@0.1) | ~53 KB (~1200×) |
| SCAN kernel-only (local) | 1.87 ms | 15.65 ms |
| SCAN wall (after pts-loop fix) | ~10 ms | ~20 ms |
| FIT (GPU mesh + host core LS) | ~11 ms | ~36 ms |

## Open Issues

- [x] USER confirm regenerated CLI AFM strips (`wave2_afm_cli`) — OK 2026-08-11
- [ ] Core LS still host — optional GPU residual sampling / lstsq
- [ ] PTCDA FIRE local-vs-bucket sparse float32 drift (p99 fine; max ~2e-3)
- [~] Quasi-2D XY sharpness vs GridFF (older path; separate from contact_pme)
- ND `--contact-surface` flag still open for quasi-2D
