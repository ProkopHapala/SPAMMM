---
type: Task
title: Fast 2.5D AFM — contact-surface variants
tags: [afm, contact-surface, morse, opencl]
timestamp: 2026-07-24
---

# Task: Fast 2.5D AFM — contact-surface variants (inventory + finish)

**Status:** investigating (**priority bump:** helicene assembly exposed long-range / “too close” bias vs Morse+Coulomb GridFF)  
**Priority:** P1 (blocks Assembly AFM screening quality) / was P2  
**Design:** `doc/Topics/AFM/ContactSurface_Static.md`, `ContactSurface_Elastic.md`  
**Audit:** `doc/TopicalAudit/AFM_ContactSurface.md`  
**Report (helicene):** `doc/Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md`  
**Pitfalls:** `doc/Takeways.md` (z alignment, F_ref layout, reg, weighting)  
**Was:** scattered Later items in `ToDo.agents.md` — this file is the SSOT task

## Objective

Finish and document the **memory-efficient 2.5D AFM** approach that avoids a full 3D `img_FF` GridFF: store a **low-res 2D lateral field** plus a **few z-basis functions**, fit once, evaluate during PP relax.

**Hard requirement:** separable (and PIC) must reproduce Morse+Coulomb **qualitatively** (E(z)/Fz(z) shape and onset height). Helicene assembly compare failed that bar — fix before trusting screening images.

## Variants in the repo

| # | Idea | Implementation | Status |
|---|------|----------------|--------|
| **(i)** | Contact height \(h_0(x,y)\) + z-modes | Separable B-spline × poly-of-exp | Prototype — PTCDA PP Fz ~14 meV/Å; **helicene bias open** |
| **(ii)** | Atom-bounded radial modes + PIC cells | Radial PIC | Prototype ~20 meV/Å; `reg≈1e-2` |
| **(iii)** | Hybrid / folded | Not shipped as one API | Investigate / defer |

**Code SSOT:** `kernels/contact_surface.cl`, `spammm/surfaces/ContactSurface.py`, `spammm/SPM/AFM.py`  
**L0:** `tests/SPM/test_afm_contact_surface.py`  
**L2 PTCDA:** `tests/testplot_contact_surface.py`, `tests/SPM/testplot_afm_contact_surface.py` → `debug/testplot_contact_surface/`  
**L2 helicene:** `run_assembly_afm.py --compare-dir …/rank09_idx2767`

## Work plan

1. Inventory — done.  
2. **Parity hardening (NOW — do not reinvent)**  
   - Re-run PTCDA L2 with **current** knobs; compare to documented ~14 meV/Å.  
   - **New:** 1-atom (q=0) and 2-atom (charged) E/Fz vs brute + GridFF (same spirit as helicene `--compare-dir`).  
   - Bisect Boltzmann / fit-z / force weight / poly basis if toys fail.  
3. Pipeline flag `{separable, pic, grid3d}`.  
4. Hybrid (iii) — decide implement vs defer.  
5. GUI — optional later.  
6. Elastic Phase 2 — design-only unless USER asks.

## Deliverables

- [ ] Topical audit + surfaces README clear on variants  
- [ ] L0 parity asserts with published RMSE targets  
- [ ] 1-atom / 2-atom toy parity plots + `.out`  
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
