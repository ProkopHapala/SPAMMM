---
type: Task
title: PairFF + FAF substrate (compose, optional demo path)
status: done
tags: [PairFF, FAF, folded, demo, reuse]
timestamp: 2026-07-24
---

# Task: PairFF molecules + FAF substrate (optional demo path)

**Status:** Done (USER confirmed Vispy map = PairFF + FAF at molecule height, 2026-07-24).  
**Goal:** Optional mode: active molecule feels **PairFF from other molecules** + **FAF from substrate**; background map from both; keep **Kz-without-FAF** switchable.

**Related:** [`PairFF_MultiBody_Kernel.md`](PairFF_MultiBody_Kernel.md) (allmol shared buffers), [`PairFF_GUI_Integration.md`](PairFF_GUI_Integration.md), [`TopicalAudit/PairFF_RigidBody.md`](../TopicalAudit/PairFF_RigidBody.md), [`Topics/ForceFields/PairFF.md`](../Topics/ForceFields/PairFF.md), [`demos/PairFF_manual.md`](../../demos/PairFF_manual.md).

---

## Delivered

| Piece | Where |
|-------|--------|
| Fused GPU kernels | `rigid_body_pairff_unified_{,env_,allmol_}faf_kernel` in `kernels/rigid.cl` |
| Shared multi-mol layout | `from_molecules` → `allmol_mode`; `set_active_body` = index only (dynamics persist; FAF survives switch) |
| Attach API | `RigidBodyPairFF.attach_pairff_faf(fit, z_init=…, k_z=0)` |
| Map helpers | `FoldedRigid.eval_folded_potential_grid`, `faf_type_idx_for_probe` |
| Vispy map | `E_map = E_PairFF(env) + E_FAF(probe)` at **active CoM z** |
| Demo CLI | `--faf` / `--faf-fit` / `--z-init`; default fit cache `data/fits/hcooh_nacl.npz` |
| Kz path | Default (no `--faf`): unchanged vacuum + `k_z` |

```bash
python3 demos/demo_pairff.py --bodies 4 --faf
python3 demos/demo_pairff.py --bodies 4 --faf --faf-fit data/fits/hcooh_nacl.npz --z-init 3.5
```

Physics intent (unchanged):

1. One mobile molecule (integrator)  
2. Other molecules = PairFF (not FAF)  
3. Crystalline substrate (e.g. NaCl) = FAF  

Do **not** put substrate atoms into PairFF env when FAF models the surface (double-count).

---

## Inventory (SSOT after ship)

| Piece | Location |
|-------|----------|
| Fit / load / CPU FAF eval | `spammm/surfaces/FoldedRigid.py` |
| PairFF + FAF bind / run | `RigidBodyPairFF.attach_pairff_faf`, `enable_pairff_faf`, `run_pairff` |
| Vispy compose | `spammm/GUI/RigidBodyVispy.py` → `_recompute_map` |
| Demo | `demos/demo_pairff.py` |

`Z_SURF_TOP = -3.25`; with `--z-init 3.5` CoM sits at `0.25` Å.

---

## Follow-ups (not blocking Done)

- **[Tomorrow — P0]** Offline tip-pull / report maps must reuse Vispy display SSOT (`potential_to_rgba`), not softclip+add — [`PairFF_MapDisplay_SSOT.md`](PairFF_MapDisplay_SSOT.md); report [`../Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md`](../Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md)
- Main `SPAMMM_GUI` PairFF panel still design-only (`PairFF_GUI_Integration.md`)
- Mixed `--mols` + single HCOOH fit: `atom_type_ids` must match each pack’s real-atom count
- Optional: substrate atom markers in Vispy; coulomb-channel-only map overlay
- Formal L0 pytest for FAF+PairFF parity still thin (demo / headless smoke used)
- PTCDI FAF cache `data/fits/ptcdi_nacl.npz` (element-mean QEq); tip-pull API `tip_pull_scan` — viz pending MapDisplay task
