---
type: UserGuide
title: PairFF rigid-body demo — user manual
tags: [PairFF, rigid-body, Vispy, FIRE, H-bond, demo]
timestamp: 2026-07-24
---

# PairFF demo — user manual

**Essence.** GPU rigid-body PairFF lets you dock and rearrange molecules that interact only through **nonbonded** sites (Morse/compact-exp + directional H-bonds via electron-pair dummies + optional σ-holes), while each molecule stays **internally rigid**. The Vispy demo is deliberately didactic: one active body integrates; the rest are a frozen environment; click another molecule to swap who moves.

**Entry point:** [`demo_pairff.py`](demo_pairff.py) · viewer [`spammm/GUI/RigidBodyVispy.py`](../spammm/GUI/RigidBodyVispy.py) · physics [`RigidBodyPairFF`](../spammm/forcefields/RigidBodyDynamics.py).

---

## Why this exists

Flexible force fields (UFF/SPFF) move every atom. For assemblies and H-bond docking you often want **6-DOF rigid molecules** with cheap pairwise nonbonded forces that still feel directional (lone pairs, σ-holes). PairFF is that middle layer: fast enough for interactive FIRE, explicit enough to teach / score docking geometries before heavier AFM or QM stages.

---

## Quick start

Needs an NVIDIA OpenCL device (PoCL/CPU is wrong for this demo).

```bash
# Classic: uracil (static) + HCOOH (mobile)
python3 demos/demo_pairff.py

# Four identical HCOOH — click any molecule to select the mobile one
python3 demos/demo_pairff.py --bodies 4 --active 0

# Mixed species (formamide.xyz = HCONH2; PTCDA stands in until NTCDA.xyz exists)
python3 demos/demo_pairff.py --mols PTCDA.xyz HCOOH.xyz formamide.xyz --spacing 12

# Headless FIRE (no window)
python3 demos/demo_pairff.py --bodies 4 --active 1 --no-vis --steps 300
```

Basenames resolve under `data/xyz/`. Absolute/relative paths also work.

---

## Concepts (SSOT)

| Term | Meaning |
|------|---------|
| **Body** | One rigid molecule (pose = CoM + quaternion) |
| **Active body** | Only this body integrates translation/rotation |
| **Environment (env)** | All other bodies, frozen in world frame, still force the active one |
| **Sites** | Real atoms + optional epair (`E`) / σ-hole (`Sh`) dummies |
| **Unified kernel** | Compact-exp PairFF (`rigid_body_pairff_unified_*`); required for multi-body |
| **Legacy kernel** | Morse + Lorentzian H-bond; classic 1+1 only |
| **Potential map** | CPU 2D scan of a probe atom over **static** sites (background under markers) |

**Multi-body rule:** exactly one active integrator; Strategy M tiles one env molecule into local memory per pass (`rigid_body_pairff_unified_env_kernel`). Each env molecule ≤ **128** sites (atoms + dummies).

---

## Interactive controls

### Mouse / keys

| Input | Action |
|-------|--------|
| **LMB** on atom (classic) | Anchor-spring drag on the mobile molecule |
| **LMB** on any molecule (multi-body) | That molecule becomes **active**; others freeze; map rebuilds; drag starts |
| Wheel | Zoom |
| Arrows | Pan |
| **SPACE** | Run / pause |
| **R** | Reset velocities |
| **F** | Toggle FIRE (default **ON**) |
| **ESC** | Quit |

### Side panel

- **Run / Reset V / FIRE** — integration mode (see below)
- **Active: k/N** — which body is mobile (multi-body)
- **Kernel** — Legacy vs Unified (multi-body needs Unified)
- **FF params** — He, Hs, w, β, k_z, … (live map recompute)
- **Probe** — H+ / O− presets for the background map
- **Show map** — toggle potential overlay

---

## FIRE vs damped MD (what changes)

Both modes call the same force kernel each step. The difference is **velocity handling** on the GPU (`md_params.w`):

| Mode | Behavior | Feel |
|------|----------|------|
| **FIRE ON** (default) | Quench when \(v\cdot F < 0\) (and analog for torque); adaptive damping — standard FIRE minimization | Snappy; settles to local minima; best for docking / dragging |
| **FIRE OFF** | Velocity Verlet + fixed linear/angular damping | More “physics MD”; can oscillate or drift; useful to feel inertia |

Headless `--no-vis` always uses FIRE via `relax_pairff`.

---

## CLI reference

| Flag | Role |
|------|------|
| `--no-vis` | Headless FIRE relax |
| `--steps`, `--dt` | Headless budget / timestep |
| `--pairff-mode {unified,legacy}` | Force kernel (default unified) |
| `--bodies N` | N copies of HCOOH on a grid |
| `--mols A.xyz B.xyz …` | Mixed molecules (overrides `--bodies`) |
| `--active k` | Initial active body index |
| `--spacing Å` | Grid CoM spacing |
| `--he`, `--hs`, `--w`, `--beta`, `--kz` | PairFF parameters |
| `--epair-dist`, `--sigma-dist` | Dummy geometry |

Library API for scripts:

```python
from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF
rbd = RigidBodyPairFF.from_molecules(molecules, body_pos, active_body=0, ...)
rbd.set_active_body(2)          # swap who moves; rebuilds env
rbd.relax_pairff(max_steps=300) # FIRE
```

---

## Didactic tour (suggested)

1. Run `--bodies 4 --active 0` with map on — see env potential under three frozen molecules.
2. Drag an atom with FIRE ON — watch the body snap into a basin.
3. Click another molecule — active label updates; map changes (old active becomes part of the env).
4. Toggle FIRE off briefly — feel softer, more inertial motion; toggle back on.
5. Run `--mols PTCDA.xyz HCOOH.xyz formamide.xyz --spacing 12` — mixed site counts still work.

---

## Limits & pitfalls

- Multi-body **requires** unified mode and an NVIDIA GPU.
- No `data/xyz/NTCDA.xyz` yet — use `PTCDA.xyz` or export your own; `formamide.xyz` = HCONH2.
- Potential map is CPU and uses **static** sites only (not the active molecule).
- Not yet a panel inside main `SPAMMM_GUI` (standalone Vispy); GUI wiring is tracked in [`doc/Tasks/PairFF_GUI_Integration.md`](../doc/Tasks/PairFF_GUI_Integration.md).

---

## Related docs

| Doc | Role |
|-----|------|
| [`doc/TopicalAudit/PairFF_RigidBody.md`](../doc/TopicalAudit/PairFF_RigidBody.md) | Cross-module inventory |
| [`doc/Topics/ForceFields/PairFF.md`](../doc/Topics/ForceFields/PairFF.md) | Design report (kernels, API) |
| [`doc/Tasks/PairFF_MultiBody_Kernel.md`](../doc/Tasks/PairFF_MultiBody_Kernel.md) | Strategy M design notes |
| [`examples/density_comparison/HBondFF/`](../examples/density_comparison/HBondFF/) | CPU radial / map reference |
