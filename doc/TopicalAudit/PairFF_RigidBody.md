---
type: TopicalAudit
title: PairFF rigid-body nonbonded (H-bond / σ-hole)
tags: [PairFF, rigid-body, OpenCL, Vispy, FIRE, FAF]
timestamp: 2026-07-24
---

# PairFF rigid-body nonbonded

## Summary

**PairFF** is SPAMMM’s rigid-molecule intermolecular force field: pairwise compact-exp (unified) or Morse+Lorentzian (legacy), with electron-pair and optional σ-hole dummy sites for directional H-bonds. Dynamics are **6-DOF** (CoM + quaternion) on the GPU.

**Multi-body (current SSOT):** all molecules live in **one shared GPU layout** (`allmol_mode`). Exactly one `active_mol` integrates; neighbors contribute via tiled PairFF (`j != active`). Switching active is an **index write** — poses, velocities, and FIRE state stay persistent (ready for future all-mobile MD). Legacy `*_env_*` kernels (rebuild env on switch) remain in `rigid.cl` for compat but are not the demo path.

**Optional substrate:** fused PairFF+FAF kernels + Vispy map `E_PairFF + E_FAF` at active CoM height (`--faf`). Kz-without-FAF remains the default.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| OpenCL | `kernels/Forces.cl` (`compact_exp_pair_EF`) | active | Shared radial primitive |
| OpenCL | `kernels/rigid.cl` 7–9 | active | Legacy, unified 1+1, unified env (legacy multi) |
| OpenCL | `kernels/rigid.cl` 10–13 | active | Unified±FAF; **allmol** ± FAF (preferred multi) |
| Python | `RigidBodyDynamics.py` → `RigidBodyPairFF` | active | `from_molecules`, `set_active_body`, `attach_pairff_faf` |
| Python | `spammm/surfaces/FoldedRigid.py` | active | Fit/load; `eval_folded_potential_grid`; probe type pick |
| Python/GUI | `spammm/GUI/RigidBodyVispy.py` | active | FIRE default ON; click→active; map PairFF[+FAF] |
| Demo | `demos/demo_pairff.py` | active | `--bodies`, `--mols`, `--faf`, `--faf-fit`, `--z-init` |
| Fit cache | `data/fits/hcooh_nacl.npz` | active | HCOOH@NaCl folded total coeffs |
| User guide | `demos/PairFF_manual.md` | active | End-user manual |
| CPU ref | `examples/density_comparison/HBondFF/` | active | Radial fit / map |
| Main GUI | FoldedRigid / PairFF panel | unfinished | `doc/Tasks/PairFF_GUI_Integration.md` |

## Parity Status

- Headless multi-body FIRE on NVIDIA: only active CoM moves; inactive pose/vel persist across `set_active_body` (USER-confirmed).
- `--faf` Vispy map = PairFF(env) + FAF(probe) at molecule z (USER-confirmed “PERFECT”, 2026-07-24).
- FAF buffers survive active switch (no realloc).
- CPU map vs GPU: same compact-exp family as `fit_radial.py`; map omits Coulomb (diagnostic).
- Pure `run_folded` path remains a **different** interaction model (substrate-only).

## Open Issues

- No `data/xyz/NTCDA.xyz`; demo uses PTCDA as stand-in.
- Main `SPAMMM_GUI` integration still design-only.
- Strategy C (flat site chunks) not implemented; allmol/M first.
- Formal pytest L0 for PairFF multi-body / FAF compose still thin.
- Mixed-species `--mols` + FAF requires fit `atom_type_ids` length match per pack.
