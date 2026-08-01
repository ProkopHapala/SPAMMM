---
type: TopicalAudit
title: PairFF rigid-body nonbonded (H-bond / σ-hole)
tags: [PairFF, rigid-body, OpenCL, Vispy, FIRE, FAF, tip-pull, QEq]
timestamp: 2026-07-28
---

# PairFF rigid-body nonbonded

## Summary

**PairFF** is SPAMMM’s rigid-molecule intermolecular force field: pairwise compact-exp (unified) or Morse+Lorentzian (legacy), with electron-pair and optional σ-hole dummy sites for directional H-bonds. Dynamics are **6-DOF** (CoM + quaternion) on the GPU.

**Multi-body (current SSOT):** all molecules live in **one shared GPU layout** (`allmol_mode`). Exactly one `active_mol` integrates; neighbors contribute via tiled PairFF (`j != active`). Switching active is an **index write** — poses, velocities, and FIRE state stay persistent (ready for future all-mobile MD). Legacy `*_env_*` kernels (rebuild env on switch) remain in `rigid.cl` for compat but are not the demo path.

**Optional substrate:** fused PairFF+FAF kernels + Vispy map `E_PairFF + E_FAF` at active CoM height (`--faf`). **Display SSOT:** `potential_to_rgba` scales to attractive `|Emin|` (Pauli cores clipped). Kz-without-FAF remains the default.

**Tip-pull (2026-07-28):** `tip_pull_scan` + PTCDI/QEq stepwise adsorbate pull on NaCl — physics/API in place; **offline matplotlib movies not yet Vispy-faithful** → [`Tasks/PairFF_MapDisplay_SSOT.md`](../Tasks/PairFF_MapDisplay_SSOT.md). Report: [`Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md`](../Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md).

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| OpenCL | `kernels/Forces.cl` (`compact_exp_pair_EF`) | active | Shared radial primitive |
| OpenCL | `kernels/rigid.cl` 7–9 | active | Legacy, unified 1+1, unified env (legacy multi) |
| OpenCL | `kernels/rigid.cl` 10–13 | active | Unified±FAF; **allmol** ± FAF (preferred multi) |
| OpenCL | `kernels/rigid.cl` 14 | active | Replica×active energy channels plus fused real-atom/CoM clash flags for MC/GA; GTX 1650 PairFF/FAF parity and runtime tested |
| Python | `RigidBodyDynamics.py` → `RigidBodyPairFF` | active | `from_molecules`, `set_active_body`, `attach_pairff_faf`, batched `greedy_energy_step`, `tip_pull_scan`, `world_sites_all_bodies` |
| Python | `spammm/surfaces/FoldedRigid.py` | active | Fit/load; `eval_folded_potential_grid`; probe type pick |
| Python | `spammm/surfaces/surface_plots.py` | experimental | Tip-pull movie helpers; **display scale not yet Vispy SSOT** |
| Python/GUI | `spammm/GUI/RigidBodyVispy.py` | active | FIRE default ON; click→active; map PairFF[+FAF]; **`potential_to_rgba` = display SSOT** |
| Demo | `demos/demo_pairff.py` | active | `--bodies`, `--mols`, `--faf`, `--faf-fit`, `--z-init` |
| Fit cache | `data/fits/hcooh_nacl.npz` | active | HCOOH@NaCl folded total coeffs |
| Fit cache | `data/fits/ptcdi_nacl.npz` | active | PTCDI@NaCl with element-mean QEq charges |
| User guide | `demos/PairFF_manual.md` | active | End-user manual |
| CPU ref | `examples/density_comparison/HBondFF/` | active | Radial fit / map; **physical QEq = negate solver** |
| Main GUI | FoldedRigid / PairFF panel | unfinished | `doc/Tasks/PairFF_GUI_Integration.md` |

## Parity Status

- Headless multi-body FIRE on NVIDIA: only active CoM moves; inactive pose/vel persist across `set_active_body` (USER-confirmed).
- `--faf` Vispy map = PairFF(env) + FAF(probe) at molecule z (USER-confirmed “PERFECT”, 2026-07-24).
- FAF buffers survive active switch (no realloc).
- CPU map vs GPU: same compact-exp family as `fit_radial.py`; map omits Coulomb (diagnostic).
- Pure `run_folded` path remains a **different** interaction model (substrate-only).
- Tip-pull PTCDI+QEq: energy/registry oscillations seen; **map GIFs unverified vs Vispy** (softclip path).
- Kernel 14 PairFF and PairFF+FAF channel parity on GTX 1650: `|err|=3.912e-08`; fused clash flags match CPU real-atom/CoM distances in `test_pairff_replica_clash_channel_matches_cpu`.
- PTCDA 4-molecule/512-trial greedy step: 79–89 ms → 0.66 ms sustained after GPU clash fusion and vectorized proposal/packing ([task report](../Tasks/PairFF_MC_PythonBottleneck.md)); awaiting USER confirmation.

## Open Issues

- **[Tomorrow]** Tip-pull / `surface_plots` map display must **reuse** Vispy compose + `potential_to_rgba` — no softclip reinvent ([`PairFF_MapDisplay_SSOT.md`](../Tasks/PairFF_MapDisplay_SSOT.md)).
- XYZ PTCDA/PTCDI ship with `Q=0` → Coulomb/Hbond silent until QEq (physical sign: negate `solve_from_elements`; GUI QEq does not flip — SSOT TBD).
- Per-atom QEq in FAF fit → too many types (`FOLDED_TYPES_MAX=8`); use element-mean `q_override`.
- No `data/xyz/NTCDA.xyz`; demo uses PTCDA as stand-in.
- Main `SPAMMM_GUI` integration still design-only.
- Shared rigid-molecule pose SSOT (`pos`+`qrot`) across PairFF / Assembly / FoldedRigid / PME — see [`RigidBody.md`](RigidBody.md) and [`Tasks/RigidMoleculePose_SSOT.md`](../Tasks/RigidMoleculePose_SSOT.md).
- Strategy C (flat site chunks) not implemented; allmol/M first.
- Formal pytest L0 for PairFF multi-body / FAF compose / tip_pull still thin.
- Mixed-species `--mols` + FAF requires fit `atom_type_ids` length match per pack.
