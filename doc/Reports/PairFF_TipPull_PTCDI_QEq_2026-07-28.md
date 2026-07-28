---
type: Report
title: PairFF tip-pull on NaCl — PTCDI + QEq (session 2026-07-28)
status: investigating
tags: [PairFF, FAF, tip-pull, PTCDI, QEq, visualization]
timestamp: 2026-07-28
---

# PairFF tip-pull on NaCl — PTCDI + QEq

**Status:** investigating (physics/API usable; **map display not SSOT** — redo tomorrow).  
**Artifacts:** `debug/pairff_ptcdi_tip_pull/` · earlier PTCDA draft `debug/pairff_ptcda_tip_pull/`  
**Fit cache:** `data/fits/ptcdi_nacl.npz` (element-mean QEq charges in unique types)  
**Related:** [`PairFF_MapDisplay_SSOT.md`](../Tasks/PairFF_MapDisplay_SSOT.md) (tomorrow), [`PairFF_FAF_Substrate.md`](../Tasks/PairFF_FAF_Substrate.md), [`TopicalAudit/PairFF_RigidBody.md`](../TopicalAudit/PairFF_RigidBody.md), Vispy SSOT `spammm/GUI/RigidBodyVispy.py` → `potential_to_rgba` / `_recompute_map`.

---

## Essence

GPU **allmol PairFF + fused FAF** can drive a stepwise tip-pull of a second adsorbate past a static neighbor on NaCl. The session established that **XYZ default charges are zero**, so Coulomb and directional H-bond terms were silent until **physical QEq** was applied (PTCDI). Offline matplotlib movies are **not** yet faithful to the Vispy/demo map look that USER already confirmed.

---

## What shipped (API / physics)

| Piece | Location | Notes |
|-------|----------|--------|
| Shared multi-mol + FAF | `RigidBodyPairFF.from_molecules`, `attach_pairff_faf` | Already Done earlier; reused |
| Tip spring scan | `RigidBodyPairFF.tip_pull_scan` | Pin local atom → path; `run_pairff` per step |
| World sites dump | `world_sites_all_bodies` | Frames / XYZ traj |
| Plot helpers | `spammm/surfaces/surface_plots.py` | `plot_pairff_faf_background`, `render_pairff_tip_pull_movie`, `save_pairff_xyz_trajectory` |
| PTCDI FAF fit | `data/fits/ptcdi_nacl.npz` | `fit_folded_for_molecule` + element-mean QEq `q_override` (per-atom Q → 40 types exceeds `FOLDED_TYPES_MAX=8`) |

**Protocol used (stepwise):**

1. Relax **one** PTCDI on FAF alone  
2. Add second PTCDI; keep mol0 static; relax mol1  
3. Tip-pull mol1 **along long axis** (+x), pin imide H; fixed plot `extent`

**Electrostatics (root cause of “no attraction”):**

- `load_xyz_with_REQs` → `REQs[:,2] = 0` for PTCDA/PTCDI XYZ (no 4th column).  
- Unified PairFF: attraction needs `Qi·Qj` and/or epair `min(0, Q_atom·He)`. With `Q=0`, only Morse cores remain.  
- **Physical QEq:** `q = -solve_from_elements(...)` — same sign convention as `examples/density_comparison/HBondFF/ff_map.py` (GUI `QEqExtension` currently does **not** flip; landmine).  
- Typical PTCDI: O≈−0.39, H≈+0.13, N≈−0.31 → Coulomb + N–H···O H-bond path open.

**Map probe:** Vispy **O−** preset (`R0`/`E0` from AtomTypes O, `q=−0.4`); FAF type via `faf_type_idx_for_probe`.

---

## What looked wrong (USER feedback → tomorrow)

Offline tip-pull frames **ad-hoc softclipped** PairFF then **added** FAF with a fixed symmetric `vmin/vmax`. That is **not** the established Vispy diagnostic:

```text
# SSOT (demo / RigidBodyVispy._recompute_map) — DO NOT reinvent
Emap = E_PairFF(static, probe) + E_FAF(probe)   # same z = active CoM
rgba = potential_to_rgba(Emap)                  # vmax = max(|Emin|, 0.01)
# → scales to attractive well; Pauli cores oversaturated on purpose
```

PairFF Pauli walls (~tens of eV) and FAF corrugation (~0.1 eV) live on **different scales**. Softclip+add reinvented poorly and forced another “re-discover what we already shipped” loop. **Do not implement the fix in this session** — see task [`PairFF_MapDisplay_SSOT.md`](../Tasks/PairFF_MapDisplay_SSOT.md).

Dynamics energy itself **does** fuse PairFF+FAF on GPU (correct). The failure mode is **headless/movie visualization**, not the fused kernel.

---

## Observed numbers (unverified vs Vispy look)

| Quantity | Approx. |
|----------|---------|
| Step2 E (active) | ≈ −0.80 eV (converged) |
| min O···H (mol0–mol1) | ≈ 3.3 Å |
| Tip-pull E(x) | Oscillates ~4 Å period; troughs to ≈ −1.18 eV |
| CoM Δx along pull | ≈ 10 Å |

Energy oscillations are consistent with **NaCl registry** under FAF, not with a featureless Morse wall.

---

## Artifacts to review

```text
debug/pairff_ptcdi_tip_pull/
  map_3panel_Oprobe.png      # FAF | softclip PairFF | sum  (display NOT SSOT)
  faf_only.png
  step2_adsorbed.png
  ptcdi_tip_pull_anim.gif
  ptcdi_tip_pull_trajectory.{png,svg}
  ptcdi_tip_pull_energy.png
  ptcdi_tip_pull.xyz
  traj.npz
data/fits/ptcdi_nacl.npz
```

---

## Reproduction sketch (physics only; viz TBD after MapDisplay task)

```bash
# Fit once (element-mean QEq) — already cached
# Then: stepwise relax + tip_pull_scan via RigidBodyPairFF (inline session script;
# promote to demos/ only after map display reuses Vispy SSOT)
```

No dedicated `demos/demo_pairff_tip_pull.py` yet — intentional until display composition is fixed on top of tested Vispy helpers.

---

## Open / next

1. **Tomorrow:** [`PairFF_MapDisplay_SSOT.md`](../Tasks/PairFF_MapDisplay_SSOT.md) — reuse Vispy compose+`potential_to_rgba`; stop softclip reinvent.  
2. Stronger N–H···O seed geometry before pull (optional).  
3. Align GUI QEq sign with HBondFF / this report (or document single SSOT).  
4. L0 pytest for tip_pull / QEq nonzero charges (thin).
