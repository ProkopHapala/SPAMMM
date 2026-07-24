---
type: TopicalAudit
title: AFM Contact Surface (quasi-2D / 2.5D)
tags: [afm, contact-surface, morse, opencl, parity]
---

# Topical Audit: AFM Contact Surface

## Summary

Quasi-2D **contact-surface** replaces dense 3D `img_FF` for classical Morse/LJ (+ Coulomb) PP-AFM: fit a lateral field (separable B-spline × z-modes, or radial PIC) once, then evaluate during tip relaxation. Designed to reproduce Morse+Coulomb **qualitatively**; PTCDA harnesses exist. Helicene assembly screening (2026-07-24) found separable contact-sep **too long-ranged / too sharp** vs GridFF+brute — treat as open parity regression until minimal-atom bisect closes it.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| OpenCL | `kernels/contact_surface.cl` | active | Brute Morse ref, separable Av/Atv/eval, PIC, PP relax |
| Python | `spammm/surfaces/ContactSurface.py` | active | `SeparableParams`, `PICParams`, fit helpers |
| Python | `spammm/SPM/AFM.py` | active | `fit_contact_surface`, `run_scan_contact`, GridFF `run_scan` |
| Python | `run_assembly_afm.py` | experimental | Helicene SAM pipeline + `--compare-dir` |
| Design | `doc/Topics/AFM/ContactSurface_Static.md` | active | SSOT physics + API |
| Design | `doc/Topics/AFM/ContactSurface_Elastic.md` | unfinished | Phase 2 Winkler — do not start unasked |

## Parity Status

| Pair | Tolerance / metric | Test / artifact | Status |
|------|--------------------|-----------------|--------|
| Separable eval vs force stencil | RMSE < 1e-4 | `tests/SPM/test_afm_contact_surface.py` | verified (L0) |
| Separable / PIC fit vs brute Morse+Coulomb (PTCDA) | E fit ~7 meV sep; close E ~8 meV | `tests/testplot_contact_surface.py` → `debug/testplot_contact_surface/` | prototype OK |
| Separable PP Fz vs 3D `img_FF` (PTCDA) | ~14 meV/Å mean | `tests/SPM/testplot_afm_contact_surface.py` | prototype OK |
| contact-sep vs GridFF+brute (helicene assembly) | qualitative fail: long-range / deep well | `rank09…/compare_contact_vs_gridff_*.png` | **open** |
| 1-atom / 2-atom toy | TBD | planned | not started |

## Open Issues

- Helicene: contact-sep E(z) well shifted out vs brute (~4 Å vs ~2.8 Å) — why? Boltzmann weights, fit-z window, basis, or Coulomb channel?
- Re-validate PTCDA with **current** default knobs before rewriting fit code.
- Minimal 1-atom (q=0) and 2-atom (charged) bisect — **next**; do not invent a second parity stack.
- Assembly GPU atom-cloud dedup deferred (`Assembly_AFM_Pipeline.md` Phase 2).
- ND `--contact-surface {separable,pic,grid3d}` flag still open.
