---
type: TopicalAudit
title: AFM Contact Surface (quasi-2D / 2.5D)
tags: [afm, contact-surface, morse, opencl, parity]
timestamp: 2026-07-24
---

# Topical Audit: AFM Contact Surface

## Summary

Quasi-2D **contact-surface** replaces dense 3D `img_FF` for classical Morse/LJ (+ Coulomb) PP-AFM: fit a lateral field (separable B-spline × z-modes, or radial PIC) once, then evaluate during tip relaxation. **Design intent:** atom-scale B-spline nodes (`bspl_dx~1 Å`) with high-order modes — not sub-atomic voxels. Parity target is Morse+Coulomb **onset / E(z) shape** (brute or GridFF). 2026-07-24: sphere `h₀` + `h0_R_scale=0.75` + coarse dx fixed the soft/long-range bug; XY still sharper than GridFF — **USER visual pending**.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| OpenCL | `kernels/contact_surface.cl` | active | Brute Morse ref, separable Av/Atv/eval, PIC, PP relax |
| Python | `spammm/surfaces/ContactSurface.py` | active | Sphere `h₀`, `SeparableParams`, `PICParams` |
| Python | `spammm/SPM/AFM.py` | active | `fit_contact_surface(h0_mode='spheres')`, `run_scan_contact` |
| Python | `run_assembly_afm.py` | experimental | Defaults: `bspl_dx=1.0`, `scan_dx=0.5`, `h0_R_scale=0.75` |
| Design | `doc/Topics/AFM/ContactSurface_Static.md` | active | SSOT physics + API |
| Design | `doc/Topics/AFM/ContactSurface_Elastic.md` | unfinished | Phase 2 Winkler |
| Report | `doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md` | active | **Parity SSOT** vs GridFF |

## Parity Status

| Pair | Tolerance / metric | Test / artifact | Status |
|------|--------------------|-----------------|--------|
| Separable eval vs force stencil | RMSE < 1e-4 | `tests/SPM/test_afm_contact_surface.py` | verified (L0) |
| Toys: spheres vs atom_z vs brute | well/Fz shift ~0 | `testplot_contact_surface.py --toys` | spheres+0.75 OK |
| PTCDA fit / close parity | RMSE_E ~0.01–0.03 eV | `debug/testplot_contact_surface/` | regenerated scale=0.75 |
| Helicene E/Fz(z) vs brute | qualitative track | `rank09…/compare_*_profiles.png` | improved |
| Helicene XY vs GridFF | morphology | `rank09…/compare_*_maps.png` | still sharper; **USER review** |
| Separable PP Fz vs 3D `img_FF` (PTCDA) | ~14 meV/Å (old knobs) | `testplot_afm_contact_surface.py` | re-check with new defaults |

## Open Issues

- [~] USER confirm PTCDA + helicene maps/profiles (`ContactSurface_2p5D_vs_GridFF_2026-07-24.md`)
- Residual XY sharpness vs GridFF at close approach
- Re-run PP-relaxed PTCDA parity with `bspl_dx=1.0` / sphere `h₀`
- PIC + sphere `h₀` / coarse dx not re-validated
- ND `--contact-surface {separable,pic,grid3d}` flag still open
- Assembly GPU atom-cloud dedup deferred
