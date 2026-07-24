---
type: TopicReport
title: PairFF — design report
tags: [PairFF, rigid-body, OpenCL, FIRE]
timestamp: 2026-07-24
---

# PairFF — design report

## Background

Scanning-probe workflows need a fast intermolecular layer between atomistic UFF/SPFF and heavy AFM/QM. Rigid molecules with directional nonbonded sites (lone pairs, σ-holes) match how experimentalists think about H-bond docking and 2D assemblies. PairFF ports that idea onto SPAMMM’s existing **rigid-body OpenCL** stack (`rigid.cl`), reusing FIRE/Verlet integration already used for folded-basis adsorbates — but with **pairwise** forces instead of a folded substrate field.

Motivation vs alternatives:

- **Flexible FF** — too many degrees of freedom for interactive assembly scoring.
- **Folded basis / GridFF** — excellent for lattice substrates; weak for molecule–molecule H-bond geometry.
- **PairFF** — explicit partner sites, one active 6-DOF body, frozen neighbors as environment.

## Architecture

```
XYZ (+ REQs) → epair/σ dummies → body packs (rel, REQ_ext, types)
     ↓
one active → GPU buffers (pos, quat, dyn_*)
env rest  → flat env_* tiled per molecule (Strategy M)
     ↓
rigid_body_pairff_unified[_env]_kernel → F, τ → FIRE/Verlet
     ↓
Vispy download / potential map (CPU, static sites only)
```

**SSOT for multi-body poses on host:** `_mb_pos`, `_mb_quat`, `_mb_packs`. GPU holds only the active body; `set_active_body` syncs pose from GPU, swaps packs, rebuilds env, re-inits kernel args.

## Kernels

| Kernel | Role |
|--------|------|
| `compact_exp_pair_EF` | Shared unified radial (Forces.cl) |
| `rigid_body_pairff_kernel` | Legacy Morse + Lorentzian (1 static partner) |
| `rigid_body_pairff_unified_kernel` | Unified 1+1 |
| `rigid_body_pairff_unified_env_kernel` | Unified multi-env; one local tile per env molecule |

Env constraint: each molecule tile ≤ `MAX_STATIC_ATOMS` (128).

## Python API (essence)

- `RigidBodyPairFF.from_two_molecules(...)` — classic static + dynamic.
- `RigidBodyPairFF.from_molecules(molecules, body_positions, active_body=...)` — mixed or identical species.
- `set_active_body(k)` — switch integrator; rebuild env.
- `run_pairff(..., fire=True|False)` / `relax_pairff(...)` — FIRE quench when `md_params.w < 0`.

## Interactive demo

Standalone Vispy (`RigidBodyVispy`): FIRE **on by default**; LMB selects active molecule in multi-body mode and recomputes the CPU potential map over the new static set. User manual: [`demos/PairFF_manual.md`](../../demos/PairFF_manual.md).

## Status & next steps

- Demo + Strategy M: working (USER confirmed click-to-select).
- Main GUI: not wired — [`PairFF_GUI_Integration.md`](../Tasks/PairFF_GUI_Integration.md).
- Optional: check in `NTCDA.xyz`; Strategy C tiling; L0 pytest parity vs CPU map.
