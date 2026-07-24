---
type: Task
title: Assembly → multi-rank AFM pipeline
tags: [assembly, afm, contact-surface, helicene]
timestamp: 2026-07-24
---

# Task: Assembly → multi-rank AFM pipeline (plots + dedup)

**Status:** investigating (Phase 1 plots/PBC OK per USER; contact-sep physics bias open — see report)  
**Priority:** P1 screening blocked on contact-surface parity until qualitative Morse+Coulomb match returns  
**Report:** [`doc/Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md`](../Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md)  
**Related:** [`Fast_2p5D_AFM_ContactSurface.md`](Fast_2p5D_AFM_ContactSurface.md), [`../TopicalAudit/AFM_ContactSurface.md`](../TopicalAudit/AFM_ContactSurface.md)

## Objective

Rigid-body **hexagonal SAM assembly** → wrap/PBC → multi-rank **Morse PP-AFM** with trustworthy geometry overlays and (later) rigorous export dedup. Fast path uses **separable contact-surface**; must not silently disagree with Morse+Coulomb GridFF.

## Done (Phase 1 — awaiting formal “done” only after USER OK on physics)

- [x] Combined `geometry_xy_xz.png` (light `.` atoms, thin guides)  
- [x] `afm_df_Fz_heights.png` (df\|Fz × heights; thin cell + top dots)  
- [x] Multi-rank `rankXX_idxYY/` + score-twin **annotation** (keep twins)  
- [x] PBC xyz fail-loud (enames length); AFM uses full 3×3 layer (4536 atoms)  
- [x] Heights SSOT: above **zmax**; `--z-clearance` default 8 Å  
- [x] `--compare-dir`: contact-sep vs GridFF maps + E/Fz at 2 tops  

## Critical open finding

On `rank09_idx2767`, **contact-sep is too sharp / long-ranged** vs GridFF Morse+Coulomb + brute (E well ~4 Å vs ~2.8 Å). Screening images can look “too close” even at `h_probe=8 Å`. **Do not expand assembly AFM campaigns until this is understood.**

## Next (ordered) — do not reinvent PTCDA stack

1. **Recall / re-run PTCDA parity** (already in repo):  
   - `tests/testplot_contact_surface.py` — fit vs brute  
   - `tests/SPM/testplot_afm_contact_surface.py` — PP maps vs 3D `run_scan`  
   - Metrics in `ContactSurface_Static.md` (~14 meV/Å PP Fz). Confirm with **current** knobs.  
2. **Minimal toy parity** (new, small): 1 atom q=0; 2 atoms with charge — same panel style as `--compare-dir` (brute / contact-sep / GridFF E+Fz).  
3. Knob bisect (Boltzmann, fit-z, force weight, poly basis) guided by toys + PTCDA.  
4. Resume helicene multi-rank once qualitative match returns.  
5. Phase 2: GPU atom-cloud dedup (optional `--collapse-atom-dups`).

## Acceptance

- USER confirms contact-sep vs Morse+Coulomb qualitative agreement on toys (+ PTCDA).  
- Then USER confirms helicene maps at onset heights.  
- Never mark Done without confirmation.
