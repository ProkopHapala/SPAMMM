# Task: Fast 2.5D AFM — contact-surface variants (inventory + finish)

**Status:** investigating  
**Priority:** P2 / P3 (speed story; prototype largely in-repo)  
**Design:** `doc/Topics/AFM/ContactSurface_Static.md`, `ContactSurface_Elastic.md`  
**Pitfalls:** `doc/Takeways.md` (z alignment, F_ref layout, reg, weighting)  
**Was:** scattered Later items in `ToDo.agents.md` — this file is the SSOT task

## Objective

Finish and document the **memory-efficient 2.5D AFM** approach that avoids a full 3D `img_FF` GridFF: store a **low-res 2D lateral field** plus a **few z-basis functions**, fit once, evaluate during PP relax.

Inventory and harden the variants already attempted; wire a clear AFMulator / GUI path; keep elastic Phase 2 design-only unless prioritized.

## Variants in the repo (current understanding)

| # | Idea (USER) | Implementation | Status |
|---|-------------|----------------|--------|
| **(i)** | Smooth 2D **contact height** \(h_0(x,y)\) + z-curves from that height (B-spline xy × exponential / poly-of-exp in \(z-h_0\)) | **Separable** `SeparableParams` + `build_contact_height_map`; kernels `evalSeparableBsplinePoly`, `relaxStrokesTiltedContact` | **Prototype working** — PTCDA Morse parity ~14 meV/Å |
| **(ii)** | Hybrid / short-range **atom-bounded** modes + grid cells | **Radial PIC** `PICParams` + PIC buckets; `evalRadialPIC`, `relaxStrokesTiltedPIC`, `cs_pic_eval_tile16` | **Prototype working** — parity ~20 meV/Å; needs `reg≈1e-2` |
| **(iii)** | Other / hybrid | Related patterns: **folded surface** tensor×exp (`surface.cl` / FoldedRigid FAF-like z-basis); optional **A+B hybrid** (coarse separable + PIC residual) mentioned in design but **not shipped** as one API | **Investigate** — document what exists vs vaporware |

**Code SSOT**

| Path | Role |
|------|------|
| `kernels/contact_surface.cl` | Brute Morse ref, separable Av/Atv, PIC fit/eval, PP relax |
| `spammm/surfaces/ContactSurface.py` | `ContactSurfaceCL`, fit helpers, params |
| `spammm/SPM/AFM.py` | `fit_contact_surface`, `run_scan_contact`, `fit_pic_contact_surface`, `run_scan_pic` |
| `spammm/surfaces/README.md` | Module index + knobs |
| L0 | `tests/SPM/test_afm_contact_surface.py` |
| L2 | `tests/testplot_contact_surface.py`, `tests/SPM/testplot_afm_contact_surface.py` |

## Why

3D GridFF for large aperiodic molecules is memory- and bandwidth-heavy. Contact surface keeps PP-AFM geometry but compresses \(z\) into 4–8 modes above \(h_0(x,y)\). Separable suits moderate windows; PIC scales with contact atoms.

## Work plan

1. **Inventory note (this file)** — done above; keep updated if a third API appears.  
2. **Parity hardening**  
   - Expand L0: separable + PIC force stencil vs brute; optional vs 3D `run_scan` RMSE bounds.  
   - Document default knobs (Boltzmann on separable only; PIC `reg=1e-2`).  
3. **Pipeline flag**  
   - Explicit mode `{separable, pic, grid3d}` in AFMulator / ModularPipeline (design already lists this).  
4. **Hybrid (iii) — decide**  
   - Either implement coarse separable + PIC residual correction, **or** formally defer and point to FoldedRigid / FAF as the “other” 2.5D family for **substrates** (periodic / folded), not aperiodic molecules.  
5. **GUI** — optional later; not conference-blocking.  
6. **Elastic Phase 2** — remains design-only (`ContactSurface_Elastic.md`); do not start unless USER asks.

## Deliverables

- [ ] Updated topical audit row + `spammm/surfaces/README.md` listing all variants clearly  
- [ ] L0 parity asserts with published RMSE targets  
- [ ] Decision record on hybrid (iii): implement vs defer  
- [ ] Optional: AFMulator `field_mode=` switch

## Acceptance

- USER confirms which variants are “demo-ready” vs experimental.  
- Do not mark Done without confirmation.  
- Elastic AFM stays Later unless promoted.

## Out of scope

- Replacing FDBM 3D density grids (different physics — Pauli overlap needs 3D ρ). Contact surface is for **classical Morse/LJ (and similar) PP fields**.  
- Full GridFFRelaxedScan per-pixel MD.
