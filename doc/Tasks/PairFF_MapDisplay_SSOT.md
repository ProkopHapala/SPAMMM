---
type: Task
title: PairFF+FAF map display SSOT (reuse Vispy; tip-pull movies)
status: open
priority: tomorrow
tags: [PairFF, FAF, Vispy, visualization, tip-pull, reuse]
timestamp: 2026-07-28
---

# Task: PairFF + FAF map display — reuse Vispy SSOT (do tomorrow)

**Status:** open — **do not invent a new scaler tonight.**  
**Why now:** Tip-pull / report movies used ad-hoc `softclip(PairFF)+FAF` + fixed symmetric `imshow` limits. USER: substrate and PairFF live on **different scales**; the **GUI/demo map already looked right**. We keep re-deriving what was established.

**Report context:** [`../Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md`](../Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md)

---

## Established SSOT (build on this — do not rewrite)

| Piece | Location | Behavior |
|-------|----------|----------|
| Compose | `RigidBodyVispy._recompute_map` | `Emap = PairFF(static, probe) + FAF(probe)` at **active CoM z** |
| Probe | ControlPanel H+ / O− | Same R0,E0,Q pick FAF type via `faf_type_idx_for_probe` |
| **Display** | `potential_to_rgba(Emap)` | `vmax = max(\|Emin\|, 0.01)`, `vmin = -vmax`; **clip Pauli cores**; show attractive basins + FAF corrugation |
| Demo | `demos/demo_pairff.py --faf` | USER confirmed map look (“PERFECT”, 2026-07-24) |
| Fit helpers | `FoldedRigid.eval_folded_potential_grid` | Already used by Vispy |

Physics fuse on GPU (`*_faf_kernel`) is fine. This task is **diagnostic visualization** for matplotlib / GIF / stills so they match Vispy.

---

## What went wrong (tip-pull session)

1. Offline path in `surface_plots.plot_pairff_faf_background` took a pre-baked `Esum` + caller-chosen `vmax`.  
2. Session script softclipped PairFF then added FAF — **new recipe**, not Vispy.  
3. Symmetric fixed `±0.3 eV` either washes out FAF or floods with cores → “I do not see surface / interaction”.

---

## Tomorrow implementation plan (KISS / reuse)

1. **Inventory first:** read `_recompute_map` + `potential_to_rgba`; optionally export a thin shared helper (prefer **move/reuse** from Vispy into a place both GUI and `surface_plots` can import — e.g. keep function in `RigidBodyVispy` and import, or lift to `spammm/plotUtils` / `surface_plots` only if Vispy import is heavy). **Ask before new module.**  
2. **Headless compose API** (one function, call Vispy logic):
   - inputs: static hosts + probe + optional `faf_fit` + `z_height` + grid
   - output: `Emap` (raw sum) + display-normalized field **or** RGBA using `potential_to_rgba`
3. **Wire tip-pull movie** (`render_pairff_tip_pull_movie` / `plot_pairff_faf_background`) to that helper — **delete softclip path**.  
4. **Fixed extent** (already required): keep explicit `xlim`/`ylim` from extent.  
5. **Parity check:** same scene as `demo_pairff.py --faf` (e.g. 2–4 HCOOH) → matplotlib still vs Vispy screenshot / USER eye; then re-render PTCDI tip-pull GIF.  
6. If single-scale sum still insufficient for movies: **dual panel** (FAF-only \| PairFF-only \| Vispy-scaled sum) using the **same** Emaps Vispy would compute — not a third softclip formula.

**Out of scope tomorrow unless USER asks:** changing fused GPU energy; main SPAMMM_GUI PairFF panel.

---

## Acceptance (USER confirms)

- [ ] Tip-pull / report maps **look like** `--faf` Vispy (attractive scale, FAF lattice visible, PairFF basins visible without inventing softclip).  
- [ ] No duplicate scaling recipe outside the shared helper.  
- [ ] PTCDI (or HCOOH) GIF regenerated; USER says OK before marking Done.

---

## Do not

- Softclip PairFF then add FAF as a “clever” display fix  
- Hardcode `vmax=0.35` / `tanh` without going through SSOT  
- Re-fit FAF or re-derive QEq “from scratch” while fixing display  
- Mark Done without USER visual confirmation
