# Task: Stable Cosserat / cassette rods for coarse-grid PTCDA

**Status:** investigating  
**Priority:** Later (research import)  
**Human ToDo:** item 1  
**Parent:** `doc/Tasks/RepoConsolidation.md`

## Objective

Port (or reimplement) **stable Cosserat / cassette rod** dynamics for coarse-grained PTCDA (and similar flat oligomers) on a substrate grid — interactive morphing / assembly without full atomistic FF at every step.

**External reference:** [Stable Cosserat Rods (Utah)](https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/)

## Why SPAMMM

PTCDA / NTCDA brickwalls and cassette-like adsorbates need fast coarse dynamics for GUI editing and assembly scoring. Atomistic UFF/SPFF/LFF already exist; rods fill the **mesoscale** gap between rigid-body (`RigidBodyDynamics`) and full MD.

## Inventory (do before coding)

| Location | What |
|----------|------|
| Utah project page / papers / code release | Discrete Cosserat rod, energy, constraints, time integration |
| SPAMMM `spammm/forcefields/RigidBodyDynamics.py`, `kernels/rigid.cl` | Existing 6-DOF rigid bodies — possible building blocks |
| SPAMMM `spammm/forcefields/Assembly.py`, `kernels/assembly.cl` | Clash / packing on surfaces |
| SPAMMM LFF | `kernels/LFF.cl` — projective springs; may share constraint ideas |
| FireCore XPBD / RigidAtom | `doc/FireCore_migration_codemap.md` — do not duplicate without decision |

**Ask USER** if Utah code may be vendored vs clean-room reimplementation from papers.

## Deliverables (phases)

1. **Design note** — degrees of freedom (centerline + material frame), coupling to substrate GridFF/FAF, PTCDA segment length / bending stiffness mapping.
2. **OpenCL kernel + Python wrapper** under `spammm/dynamics/` or `spammm/forcefields/` (reuse `OpenCLBase`).
3. **L0 test** — energy decrease / constraint residual on a short rod; PTCDA coarse chain smoke test.
4. **GUI** — optional plugin: drag coarse PTCDA cassette in mol browser / editor (after compute is solid).

## Out of scope

- Replacing atomistic FF for chemistry.
- Full XPBD molecular port (separate Later item in `ToDo.agents.md`).

## Acceptance

- [ ] Design approved by USER (Utah license / API sketch)
- [ ] GPU rod step runs on NVIDIA; PoCL not accepted as “done”
- [ ] One PTCDA coarse demo plot in `debug/`
- [ ] Linked from `ToDo.agents.md`; USER confirms before Done
