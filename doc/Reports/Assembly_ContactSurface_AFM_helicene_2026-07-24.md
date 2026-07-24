---
type: Report
title: Rigid Assembly SAM → 2.5D Contact-Surface AFM (helicene)
description: Session handoff for DiTetraceno helicene rigid-body assembly plus Morse PP-AFM via separable contact surface; includes GridFF Morse+Coulomb compare that exposed long-range / too-close bias.
tags: [assembly, afm, contact-surface, morse, helicene, parity]
timestamp: 2026-07-24
---

# Report: Rigid Assembly → Contact-Surface AFM (helicene SAM)

**Status:** investigating — PBC/overlays OK; `h₀` spheres + coarse dx fixed soft bias; see **parity SSOT** [`ContactSurface_2p5D_vs_GridFF_2026-07-24.md`](ContactSurface_2p5D_vs_GridFF_2026-07-24.md) (XY vs GridFF still sharper — USER review)  
**Task SSOT:** [`doc/Tasks/Assembly_AFM_Pipeline.md`](../Tasks/Assembly_AFM_Pipeline.md), [`doc/Tasks/Fast_2p5D_AFM_ContactSurface.md`](../Tasks/Fast_2p5D_AFM_ContactSurface.md)  
**Design:** [`doc/Topics/AFM/ContactSurface_Static.md`](../Topics/AFM/ContactSurface_Static.md)  
**Audit:** [`doc/TopicalAudit/AFM_ContactSurface.md`](../TopicalAudit/AFM_ContactSurface.md)

## Essence

Screen **hexagonal SAM assemblies** (6 C6 copies / cell) with a fast **separable 2.5D contact-surface** Morse+Coulomb PP-AFM path, then sanity-check against **3D GridFF Morse+Coulomb** on one rank. Geometry overlays and PBC atom loading are trustworthy. Soft/long-range bias traced to wrong `h₀` and sub-atomic sampling — **fixed in code**; remaining acceptance = USER visual on maps/profiles (see parity report).

## What was built (pipeline)

| Piece | Role |
|-------|------|
| `run_assembly_afm.py` | Assembly search → wrap → optional 3×3 FF PBC → multi-rank AFM; `--compare-dir` for contact vs GridFF |
| `spammm/forcefields/Assembly.py` | GPU packing; `annotate_score_twins` (keep twins, annotate) |
| `AssemblyPlot.plot_assembly_xy_xz_panel` | XY over XZ, light skeleton / `.` atoms, thin cut/probe lines |
| `AFM_utils.plot_afm_df_Fz_height_strip` | df\|Fz × heights; thin cell + top-atom dots |
| Heights | `h_probe`, `h_tip` measured **above molecular zmax** (not substrate); `L≈4 Å` lever |

**Per-rank artifacts:** `debug/helicene_afm_pipeline/rankXX_idxYY/{geometry_xy_xz.png, afm_df_Fz_heights.png, assembly*.xyz}`  
**Index:** `debug/helicene_afm_pipeline/SUMMARY.out`

### Reproduction

```bash
# Multi-rank screening (contact-sep)
python run_assembly_afm.py --preset tetraceno --n-afm 10 --z-clearance 8 --nz-scan 40 --ff-pbc 1

# One-rank: contact-sep vs GridFF Morse+Coulomb + E/Fz profiles at 2 tops
python run_assembly_afm.py --compare-dir debug/helicene_afm_pipeline/rank09_idx2767 \
  --h0-R-scale 0.75 --bspl-dx 1.0 --scan-dx 0.5 --z-clearance 8 --nz-scan 40 --grid-dx 0.25
```

Parity details / caveats: [`ContactSurface_2p5D_vs_GridFF_2026-07-24.md`](ContactSurface_2p5D_vs_GridFF_2026-07-24.md).
## Bugs fixed this session (do not re-hunt)

1. **Truncated PBC xyz** — `enames` was 1-molecule (84) while positions were 504×9; `zip` wrote 756 atoms with header 4536 → AFMulator imaged a broken layer while overlays used full cell. **Fix:** expand enames; fail-loud length checks; verify header==body; assert load count.  
2. **Top atoms vs Fz** — after PBC fix, local Fz peaks track tops (~1 Å median); **not** an `[ix,iy]` transpose bug (mean systematic shift ≈0).  
3. **Plot style / multi-rank** — combined XY∥XZ; df/Fz height strips; `rankXX_idxYY/` dirs; score-twin annotation only (no silent collapse).

## Height / contact scale (context)

- `R0 ≈ tip_R + R_vdW(C) ≈ 1.45 + 1.93 ≈ 3.38 Å` → hard contact when `h_probe ≲ R0` above a top atom.  
- Default scan raised to `--z-clearance 8` → `h_probe ∈ [8, 4.1] Å` (onset → light contact).  
- Still looked “deep” with contact-sep → **method bias**, not only z label.

## Important finding: contact-sep ≠ Morse+Coulomb (helicene rank09)

**Artifact:** `debug/helicene_afm_pipeline/rank09_idx2767/compare_contact_vs_gridff_*.png`  
**Setup:** same 4536-atom PBC sample; same XY scan; GridFF box = **cell AABB only** (~0.041 GB), not full 3×3 bbox.

| Observation | Implication |
|-------------|-------------|
| contact-sep df/Fz much sharper / stronger at large `h_probe` than GridFF | Images look “too close” |
| E(z) at tops: contact-sep well deeper and shifted out (~4 Å vs brute/GridFF ~2.8 Å) | Fit / basis / Boltzmann weighting may bias long-range |
| GridFF raw ≈ brute Morse+Coulomb (except very close) | 3D path is the reference for classical Morse+Coulomb |

**Do not treat contact-sep as “demo-ready” for quantitative height until this is resolved.**

### Toy rigid-FF bisect (2026-07-24) — root cause: missing spherical contact h₀

Harness: `python tests/testplot_contact_surface.py --toys` → `debug/testplot_contact_surface/toys/`  
**Rigid only** — no PP. tip_R=0, tip_E=1.

**Diagnosis (USER):** softness is not mainly a poly_R knob issue — legacy `h₀ = max atom-center z` is **not** a contact surface. Intended: ray vs nearby **spheres** (Morse R0) so `dz` is height above the tip–atom contact envelope.

| h₀ mode | 1-atom Δh_well | best Fz shift | rmse_E_fit | rmse_Fz_fit |
|---------|----------------|---------------|------------|-------------|
| `atom_z` (legacy PTCDA knobs) | +0.20 Å | +0.21 Å | 7.9e-2 | 4.3e-1 |
| **`spheres` (Morse R0)** | **+0.00 Å** | **+0.02 Å** | **2.0e-3** | **2.1e-3** |

Implemented: `build_contact_height_map(..., Rs=R0)`, `fit_contact_surface(h0_mode='spheres')`, fit offsets above `h0_max`; `run_assembly_afm._fit` updated. Status stays **investigating** until USER confirms on toys + helicene.

**REVIEW (open these):**
- `…/toys/C1_q0_atomz_ptcda/rigid_EFz_profiles.png` — soft / shifted
- `…/toys/C1_q0_spheres/rigid_EFz_profiles.png` — tracks brute
- `…/toys/INDEX.out`

## Do not reinvent — existing PTCDA parity (reuse first)

Contact surface was **already** validated on **PTCDA** against the same Morse+Coulomb reference:

| Harness | What it compares | Artifacts / notes |
|---------|------------------|-------------------|
| `tests/testplot_contact_surface.py` | Separable (+ PIC) **fit** vs `_brute_afm_morse_c_queries`; close / z-stack / z-profile | `debug/testplot_contact_surface/contact_surface_*parity*.png`, `*_z_profile.*` |
| `tests/SPM/testplot_afm_contact_surface.py` | **PP-relaxed** images: contact path vs **3D `img_FF` `run_scan`** | `pp_afm_parity_Fz_*_relaxed.png` |
| L0 `tests/SPM/test_afm_contact_surface.py` | Force-stencil vs eval | RMSE < 1e-4 |
| Topic table | Separable PP Fz RMSE ~**14 meV/Å** (PTCDA prototype) | [`ContactSurface_Static.md`](../Topics/AFM/ContactSurface_Static.md) §Parity |

**Next bisect** should: (1) re-run / re-read PTCDA harness with **current** fit knobs (`bspl_dx`, Boltzmann T, `fit_z_adaptive`, `fit_force_weight`); (2) add **1-atom (q=0)** and **2-atom (with q)** toys — same brute / contact / GridFF profile panel as helicene `--compare-dir`. Helicene assembly is a **stress test** (corrugated, dense, PBC), not the first place to invent a new parity stack.

## Next work (ordered)

1. **Minimal systems** — 1 atom Morse-only; 2 atoms Morse+Coulomb; E(z)/Fz(z) brute vs contact-sep vs GridFF (extend or fork `testplot_contact_surface` / `--compare-dir`).  
2. **Knob bisect on PTCDA** — Boltzmann on/off, fit-z window, force weight, `poly_R` / `m_start`; check whether long-range bias already appears on PTCDA.  
3. **Then** return to helicene assembly screening (or PIC) once contact-sep recovers qualitative Morse+Coulomb shape.  
4. Phase 2 assembly: GPU **atom-cloud** dedup (optional; score-twins stay annotated).

## Open issues / caveats

- Score-twin **geometry** dedup (GPU NN under PBC) not implemented — annotate only.  
- COM wrap leaves atoms sticking outside the green parallelogram (expected).  
- GridFF compare uses cell AABB; edges need PBC atom images (we load 3×3) but grid does not span full supercell.  
- Status fields stay **investigating** until USER confirms contact-sep fix.
