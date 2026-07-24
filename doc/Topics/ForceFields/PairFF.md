---
type: TopicReport
title: PairFF — design report
tags: [PairFF, rigid-body, OpenCL, FIRE, FAF]
timestamp: 2026-07-24
---

# PairFF — design report

## Background

Scanning-probe workflows need a fast intermolecular layer between atomistic UFF/SPFF and heavy AFM/QM. Rigid molecules with directional nonbonded sites (lone pairs, σ-holes) match how experimentalists think about H-bond docking and 2D assemblies. PairFF ports that idea onto SPAMMM’s existing **rigid-body OpenCL** stack (`rigid.cl`), reusing FIRE/Verlet already used for folded-basis adsorbates — but with **pairwise** forces instead of (or in addition to) a folded substrate field.

Motivation vs alternatives:

- **Flexible FF** — too many DOF for interactive assembly scoring.
- **Folded basis / GridFF alone** — excellent for lattice substrates; weak for molecule–molecule H-bond geometry.
- **PairFF** — explicit partner sites; one active 6-DOF body; neighbors as PairFF environment; optional **FAF** for the crystal.

## Architecture (current SSOT)

```
XYZ (+ REQs) → epair/σ dummies → packs (rel, REQ_ext, types)
     ↓
from_molecules → flat apos_body/dyn_*/mols[] + poss/qrots[n_mols]   (allmol_mode)
     ↓
active_mol index → rigid_body_pairff_unified_allmol[_faf]_kernel
     (tile j≠active; integrate only active; optional FAF on real atoms)
     ↓
Vispy: download active sites; map = PairFF(static view) [+ FAF(probe)] @ CoM z
```

**SSOT poses:** GPU holds **all** molecules. Host `_mb_pos` / `_mb_quat` / `_mb_packs` mirror for picking/map. `set_active_body(k)` writes `active_mol` only — **no** realloc, **no** velocity zeroing, FAF stays bound.

Legacy path (compat): `*_env_*` kernels with world-frame `env_*` rebuilt on switch — superseded for the demo by allmol.

## Kernels

| Kernel | Role |
|--------|------|
| `compact_exp_pair_EF` | Shared unified radial (`Forces.cl`) |
| `rigid_body_pairff_kernel` | Legacy Morse + Lorentzian (1 static partner) |
| `rigid_body_pairff_unified_kernel` | Unified 1+1 |
| `rigid_body_pairff_unified_env_kernel` | Legacy multi-env (rebuild env) |
| `rigid_body_pairff_unified_{,env_}faf_kernel` | Fused PairFF+FAF (1+1 / env) |
| `rigid_body_pairff_unified_allmol[_faf]_kernel` | **Preferred multi** ± FAF |

Constraint: each molecule tile ≤ `MAX_STATIC_ATOMS` (128).

## Python API (essence)

- `from_two_molecules(...)` — classic static + dynamic (no allmol).
- `from_molecules(...)` — shared buffers, `allmol_mode=True`.
- `set_active_body(k)` — index only; persistent dynamics.
- `attach_pairff_faf(fit, z_init=…, k_z=0)` — raise to `Z_SURF_TOP+z_init`, `init_folded`, enable fused kernel, store `faf_fit` for map.
- `run_pairff(..., fire=…, faf=…)` / `relax_pairff(...)` — monitor `active_body` by default.

## Interactive demo

Standalone Vispy (`RigidBodyVispy`): FIRE **on by default**; LMB selects active; map plane at active CoM z. With `--faf`, map shows **PairFF(env) + FAF(probe)**. Manual: [`demos/PairFF_manual.md`](../../demos/PairFF_manual.md).

```bash
python3 demos/demo_pairff.py --bodies 4 --active 0
python3 demos/demo_pairff.py --bodies 4 --faf
```

## Status & next steps

- Multi-body allmol + click-to-select: USER confirmed.
- `--faf` map compose: USER confirmed.
- Main GUI: not wired — [`PairFF_GUI_Integration.md`](../Tasks/PairFF_GUI_Integration.md).
- Optional later: all-mobile MD (drop active gate); Strategy C tiling; L0 pytest.
