---
type: Task
title: Fast 2.5D AFM — contact-surface variants
tags: [afm, contact-surface, morse, opencl]
timestamp: 2026-07-24
---

# Task: Fast 2.5D AFM — contact-surface variants (inventory + finish)

**Status:** investigating (**h₀ spheres + h0_R_scale=0.75 + atom-scale bspl/scan dx implemented**; XY vs GridFF still sharper — USER visual pending)  
**Priority:** P1 (blocks Assembly AFM screening quality) / was P2  
**Design:** `doc/Topics/AFM/ContactSurface_Static.md`, `ContactSurface_Elastic.md`  
**Audit:** `doc/TopicalAudit/AFM_ContactSurface.md`  
**Parity report (SSOT):** `doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md`  
**Report (helicene pipeline):** `doc/Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md`  
**Caveats:** `doc/Caveats.md` §6  
**Pitfalls:** `doc/Takeways.md` (z alignment, F_ref layout, reg, weighting)  
**Was:** scattered Later items in `ToDo.agents.md` — this file is the SSOT task

## Objective

Finish and document the **memory-efficient 2.5D AFM** approach that avoids a full 3D `img_FF` GridFF: store a **coarse 2D lateral field** (atom-scale nodes) plus a **few z-basis functions**, fit once, evaluate during PP relax.

**Hard requirement:** separable (and PIC) must reproduce Morse+Coulomb **qualitatively** (E(z)/Fz(z) shape and onset height). Soft/`h₀=atom_z` and sub-atomic sampling bugs are fixed; remaining: USER accept XY morphology vs GridFF.

## Variants in the repo

| # | Idea | Implementation | Status |
|---|------|----------------|--------|
| **(i)** | Contact height \(h_0(x,y)\) + z-modes | Separable B-spline × poly-of-exp; **sphere h₀**, scale 0.75 | Prototype — profiles OK; XY review open |
| **(ii)** | Atom-bounded radial modes + PIC cells | Radial PIC | Prototype ~20 meV/Å; `reg≈1e-2` |
| **(iii)** | Hybrid / folded | Not shipped as one API | Investigate / defer |

**Code SSOT:** `kernels/contact_surface.cl`, `spammm/surfaces/ContactSurface.py`, `spammm/SPM/AFM.py`  
**Defaults (assembly):** `--bspl-dx 1.0 --scan-dx 0.5 --h0-R-scale 0.75`  
**L0:** `tests/SPM/test_afm_contact_surface.py`  
**L2 PTCDA:** `tests/testplot_contact_surface.py` → `debug/testplot_contact_surface/`  
**L2 helicene:** `run_assembly_afm.py --compare-dir …/rank09_idx2767`

## Work plan

1. Inventory — done.  
2. **Parity hardening**  
   - [x] Sphere `h₀` + `h0_R_scale=0.75`  
   - [x] Atom-scale `bspl_dx` / `scan_dx` defaults  
   - [x] Toys + PTCDA + helicene regenerate  
   - [ ] USER visual confirmation  
   - [ ] Re-check PP-relaxed PTCDA with new knobs  
3. Pipeline flag `{separable, pic, grid3d}`.  
4. Hybrid (iii) — decide implement vs defer.  
5. GUI — optional later.  
6. Elastic Phase 2 — design-only unless USER asks.

## Deliverables

- [x] Topical audit + surfaces README clear on variants / sampling intent  
- [x] Parity report vs GridFF  
- [x] 1-atom / 2-atom toy harness (`--toys`)  
- [ ] L0 parity asserts with published RMSE targets (refresh)  
- [ ] Decision on hybrid (iii)  
- [ ] Optional `field_mode=` switch  
- [ ] USER confirms qualitative Morse+Coulomb match restored  

## Acceptance

- USER confirms demo-ready vs experimental.  
- Do not mark Done without confirmation.  
- Elastic stays Later unless promoted.

## Out of scope

- FDBM 3D density grids (different physics).  
- Full per-pixel GridFFRelaxedScan MD.
