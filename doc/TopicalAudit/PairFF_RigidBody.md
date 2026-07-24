---
type: TopicalAudit
title: PairFF rigid-body nonbonded (H-bond / σ-hole)
tags: [PairFF, rigid-body, OpenCL, Vispy, FIRE]
timestamp: 2026-07-24
---

# PairFF rigid-body nonbonded

## Summary

**PairFF** is SPAMMM’s rigid-molecule intermolecular force field: pairwise compact-exp (unified) or Morse+Lorentzian (legacy), with electron-pair and optional σ-hole dummy sites for directional H-bonds. Dynamics are **6-DOF** (CoM + quaternion) on the GPU. Multi-body mode integrates **one active** body against a tiled **frozen environment** (Strategy M). Interactive Vispy demo supports click-to-select active molecule and live potential maps.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| OpenCL | `kernels/Forces.cl` (`compact_exp_pair_EF`) | active | Shared radial primitive |
| OpenCL | `kernels/rigid.cl` kernels 7–9 | active | Legacy, unified 1+1, unified env (Strategy M) |
| Python | `spammm/forcefields/RigidBodyDynamics.py` → `RigidBodyPairFF` | active | `from_two_molecules`, `from_molecules`, `set_active_body` |
| Python/GUI | `spammm/GUI/RigidBodyVispy.py` | active | Vispy+Qt panel; FIRE default ON; click→active |
| Demo | `demos/demo_pairff.py` | active | `--bodies`, `--mols`, `--no-vis` |
| User guide | `demos/PairFF_manual.md` | active | End-user manual |
| CPU ref | `examples/density_comparison/HBondFF/` | active | Radial fit / map (`fit_radial.py`, `ff_map.py`) |
| Main GUI | FoldedRigid / PairFF panel | unfinished | See `doc/Tasks/PairFF_GUI_Integration.md` |

## Parity Status

- Headless multi-body FIRE on NVIDIA: active CoM moves; `set_active_body` rebuilds env (USER-confirmed interactive pick).
- CPU map vs GPU forces: same compact-exp family as `fit_radial.py`; not a formal L0 regression suite yet.
- Folded-basis rigid path (`run_folded`) is a **different** interaction model — do not confuse with PairFF.

## Open Issues

- No `data/xyz/NTCDA.xyz` (ASCII template only); demo uses PTCDA as stand-in.
- Main `SPAMMM_GUI` integration still design-only.
- Strategy C (flat site chunks) not implemented; M first.
- Formal pytest L0 for PairFF multi-body still thin (demo/headless verification).
