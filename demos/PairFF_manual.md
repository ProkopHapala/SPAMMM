---
type: UserGuide
title: PairFF rigid-body demo — user manual
tags: [PairFF, rigid-body, Vispy, FIRE, H-bond, FAF, demo]
timestamp: 2026-07-28
---

# PairFF demo — user manual

**Essence.** GPU rigid-body PairFF docks and rearranges molecules that interact through **nonbonded** sites (compact-exp / Morse + directional H-bonds via epair + optional σ-hole dummies), while each molecule stays **internally rigid**. Exactly one body integrates; the rest still force it via PairFF. Optional **FAF** adds a crystalline substrate (e.g. NaCl); the background map then shows **PairFF + FAF** at the molecule height — the diagnostic view for on-surface docking. **Display** scales to the attractive well (`potential_to_rgba`: `vmax=|Emin|`) so FAF corrugation is visible under Pauli cores — do not replace this with ad-hoc softclip in offline plots.

**Entry point:** [`demo_pairff.py`](demo_pairff.py) · viewer [`spammm/GUI/RigidBodyVispy.py`](../spammm/GUI/RigidBodyVispy.py) · physics [`RigidBodyPairFF`](../spammm/forcefields/RigidBodyDynamics.py).

**Charges:** many `data/xyz/*.xyz` ship with `Q=0`. Coulomb and H-bond terms need nonzero `REQs[:,2]` — use physical QEq (negate `solve_from_elements`, same as HBondFF). Tip-pull / PTCDI session: [`doc/Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md`](../doc/Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md). Offline map fix (tomorrow): [`doc/Tasks/PairFF_MapDisplay_SSOT.md`](../doc/Tasks/PairFF_MapDisplay_SSOT.md).

---

## Why this exists

Flexible FFs (UFF/SPFF) move every atom. For assemblies and H-bond docking you often want **6-DOF rigid molecules** with cheap pairwise nonbonded forces that still feel directional. PairFF is that middle layer. FAF (folded atomic functions) models the **lattice** so you can see molecule–molecule and molecule–substrate energies together on one map.

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

# On NaCl: PairFF between molecules + FAF substrate; map = sum at CoM z
python3 demos/demo_pairff.py --bodies 4 --faf

# Headless FIRE (no window)
python3 demos/demo_pairff.py --bodies 4 --active 1 --no-vis --steps 300
python3 demos/demo_pairff.py --bodies 4 --faf --no-vis --steps 80
```

Basenames resolve under `data/xyz/`. Absolute/relative paths also work.

---

## Concepts (SSOT)

| Term | Meaning |
|------|---------|
| **Body** | One rigid molecule (pose = CoM + quaternion) |
| **Active body** | Only this body integrates translation/rotation |
| **Environment** | Other molecules — still force the active one (PairFF), but do not integrate |
| **allmol** | Shared GPU buffers for **all** molecules; switch active = index only |
| **Sites** | Real atoms + optional epair (`E`) / σ-hole (`Sh`) dummies |
| **Unified kernel** | Compact-exp PairFF; required for multi-body / FAF |
| **FAF** | Folded substrate potential (NaCl); fused into dynamics when `--faf` |
| **Potential map** | CPU 2D probe scan: PairFF over **static** sites; **+ FAF** when `--faf`, plane at **active CoM z**; colored via `potential_to_rgba` (`vmax=\|Emin\|`) |

**Multi-body rule:** one active integrator; neighbors tiled from shared body-frame sites (`rigid_body_pairff_unified_allmol[_faf]_kernel`). Each molecule ≤ **128** sites (atoms + dummies). Poses/velocities of inactive bodies **persist** across switches.

---

## Interactive controls

### Mouse / keys

| Input | Action |
|-------|--------|
| **LMB** on atom (classic) | Anchor-spring drag on the mobile molecule |
| **LMB** on any molecule (multi-body) | That molecule becomes **active**; others stay put; map rebuilds; drag starts |
| Wheel | Zoom |
| Arrows | Pan |
| **SPACE** | Run / pause |
| **R** | Reset velocities |
| **F** | Toggle FIRE (default **ON**) |
| **ESC** | Quit |

### Side panel

- **Run / Reset V / FIRE** — integration mode
- **Active: k/N** — which body is mobile (multi-body)
- **Kernel** — Legacy vs Unified (multi-body / FAF need Unified)
- **FF params** — He, Hs, w, β, k_z, … (live map recompute; `--faf` starts with `k_z=0`)
- **Probe** — H+ / O− presets for the background map (also picks nearest FAF type)
- **Show map** — toggle potential overlay

---

## FIRE vs damped MD

| Mode | Behavior | Feel |
|------|----------|------|
| **FIRE ON** (default) | Quench when \(v\cdot F < 0\) (and torque analog); adaptive damping | Snappy; docking / dragging |
| **FIRE OFF** | Velocity Verlet + fixed damping | More inertial |

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
| `--faf` | Enable FAF NaCl substrate + fused dynamics + map compose |
| `--faf-fit PATH` | Folded `.npz` (default `data/fits/hcooh_nacl.npz`; fit+save if missing) |
| `--z-init Å` | Height above `Z_SURF_TOP` (−3.25) when `--faf` (default 3.5 → CoM z≈0.25) |
| `--he`, `--hs`, `--w`, `--beta`, `--kz` | PairFF parameters (`--faf` forces `k_z=0` on attach) |
| `--epair-dist`, `--sigma-dist` | Dummy geometry |

Library API:

```python
from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF
from spammm.surfaces.FoldedRigid import load_fit

rbd = RigidBodyPairFF.from_molecules(molecules, body_pos, active_body=0, ...)
rbd.set_active_body(2)   # index only; dynamics persist
fit = load_fit('data/fits/hcooh_nacl.npz')
rbd.attach_pairff_faf(fit, z_init=3.5, k_z=0.0)
rbd.relax_pairff(max_steps=300)
```

---

## Didactic tour

1. `--bodies 4 --active 0` with map on — PairFF basins from three frozen neighbors.
2. Drag with FIRE ON — body snaps into a basin.
3. Click another molecule — active label updates; inactive poses unchanged.
4. `--bodies 4 --faf` — substrate corrugation appears on the map under the molecules; dynamics feel the surface.
5. Toggle probe H+ ↔ O− — FAF channel follows nearest unique_REQ type.
6. `--mols PTCDA.xyz HCOOH.xyz formamide.xyz --spacing 12` — mixed site counts (FAF fit must match real-atom count if `--faf`).

---

## Limits & pitfalls

- Multi-body / FAF require **unified** mode and an **NVIDIA** GPU.
- No `data/xyz/NTCDA.xyz` yet — use `PTCDA.xyz`; `formamide.xyz` = HCONH2.
- Map is CPU; PairFF layer uses **static** sites only (not the active molecule); Vispy display uses `potential_to_rgba` (`vmax=|Emin|`).
- Do not put substrate atoms into PairFF env when FAF is on (double-count).
- XYZ often has `Q=0` — assign physical QEq before expecting Coulomb/H-bonds.
- Not yet inside main `SPAMMM_GUI` — see [`doc/Tasks/PairFF_GUI_Integration.md`](../doc/Tasks/PairFF_GUI_Integration.md).

---

## Related docs

| Doc | Role |
|-----|------|
| [`doc/TopicalAudit/PairFF_RigidBody.md`](../doc/TopicalAudit/PairFF_RigidBody.md) | Cross-module inventory |
| [`doc/Topics/ForceFields/PairFF.md`](../doc/Topics/ForceFields/PairFF.md) | Design report |
| [`doc/Tasks/PairFF_FAF_Substrate.md`](../doc/Tasks/PairFF_FAF_Substrate.md) | FAF+PairFF task (Done) |
| [`doc/Tasks/PairFF_MapDisplay_SSOT.md`](../doc/Tasks/PairFF_MapDisplay_SSOT.md) | **Tomorrow:** offline maps reuse Vispy display |
| [`doc/Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md`](../doc/Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md) | Tip-pull PTCDI+QEq session |
| [`doc/Tasks/PairFF_MultiBody_Kernel.md`](../doc/Tasks/PairFF_MultiBody_Kernel.md) | Multi-body / allmol design notes |
| [`examples/density_comparison/HBondFF/`](../examples/density_comparison/HBondFF/) | CPU radial / map reference |
