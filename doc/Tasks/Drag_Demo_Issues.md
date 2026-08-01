---
type: Task
title: Drag Demo Issues — Analysis
status: resolved — USER confirmed GIF quality (2026-08-01)
tags: [drag, stick-slip, FAF, PTCDI, NaCl, GUI-script, visualization]
timestamp: 2026-08-01
related:
  - doc/Reports/PTCDA_DragDemo_StickSlip_2026-08-01.md
  - doc/Tasks/GUI_Scripting_DemoRunner.md
---

# Drag Demo Issues — Analysis

## Status: RESOLVED — USER confirmed the final GIF (2026-08-01)

See [PTCDA_DragDemo_StickSlip_2026-08-01.md](../Reports/PTCDA_DragDemo_StickSlip_2026-08-01.md) for the full report.

## User Observations

1. **Stick-slip looks artificial** — appears induced by the anchor spring driver, not by molecule-surface interaction
2. **Other molecules move smoothly** — no visible surface interaction
3. **No surface visible** under the molecules in the animation
4. **Should be a GUI script** (`ptcda_interactive_drag.py`), not a standalone offline demo (`tests/run_drag_demo.py`)

## Physics Analysis

### FAF Surface Forces (measured at z_init=3.0 Å, mol 0 at origin)

From direct GPU force download at z=3.0 Å above NaCl:

| Component | Per-atom range | Total (38 atoms) |
|-----------|---------------|-----------------|
| Fx | [-0.079, +0.079] eV/Å | -0.001 (cancels) |
| Fy | [-0.065, +0.065] eV/Å | +0.000 (cancels) |
| Fz | [-0.004, +0.345] eV/Å | +4.679 (strong repulsion) |

**Key finding:** The lateral FAF forces (Fx, Fy) are only ~0.08 eV/Å per atom and **cancel to ~0** when summed over the molecule at a symmetric position. The FAF corrugation barrier is the **variation** of these forces as the molecule translates across the surface — not the absolute force at one point.

### Why the stick-slip looks artificial

The anchor spring has K=0.2 (PairFF internal units). The drag moves the target by `DRAG_X_END/N_STEPS = 40/16000 = 0.0025 Å/step`. With `STEPS_PER_CHUNK=5`, the spring stretches by ~0.0125 Å per chunk, building up force until it exceeds the FAF lateral barrier, then the molecule jumps forward.

**This IS stick-slip physics** — but the user is right that it looks artificial because:
- The spring is the **only** lateral force driver (no thermal noise, no other perturbation)
- The jumps are synchronized with the spring force buildup, not with surface lattice periodicity
- The molecule doesn't show **lattice-periodic** sticking — it just follows the spring

### Why other molecules don't show surface interaction

The obstacle molecule (mol 1) at x=24 Å has:
- K_Z=5.0 (strong z-confinement) — so it stays at z_init
- No anchor spring — so no lateral driving force
- FAF lateral forces ~0 at equilibrium → **no visible motion**

The obstacle only moves when mol 0 collides with it (PairFF Pauli repulsion). Between collisions, it sits still because the FAF lateral forces are tiny at equilibrium. This is physically correct but **visually uninteresting** — the user expects to see the obstacle also "feel" the surface.

### Why no surface is visible

The `make_gif()` function in `run_drag_demo.py` only draws:
- Atoms (colored circles per molecule)
- COG trails (dashed lines)
- Anchor path (red line)
- A dashed line at z=0 (labeled "surface top reference")

It does **not** render the NaCl lattice (Na+ and Cl- ions). The surface is invisible.

### Z_SURF_TOP = -3.25 Å

The NaCl surface top is at z=-3.25 Å. Molecules are placed at z_init=3.0 Å **above** the surface, so their absolute z = -3.25 + 3.0 = -0.25 Å. The animation's xz panel shows z near 0, which is correct but confusing without the lattice.

## What Needs to Change

### 1. Render the NaCl surface in the animation

Draw Na+ and Cl- ions at their lattice positions (z = Z_SURF_TOP = -3.25 Å) as a background layer. The NaCl(100) lattice has:
- Lattice constant a = 5.64 Å (but FAF uses a folded 2×2 cell)
- Need to check `fit['folded_lvec2d']` for the actual periodicity

### 2. Make stick-slip more visible and natural

Options:
- **Slower drag** — reduce drag speed so the molecule has time to settle into each lattice site
- **Weaker spring** — so the molecule sticks longer before slipping
- **Add thermal noise** — so the molecule explores the surface even without the spring
- **Show FAF lateral force** — overlay arrows or a force trace showing the surface corrugation

### 3. Make it a GUI script

The `ptcda_interactive_drag.py` script already exists and sets up the system for **interactive** dragging (user picks an O atom with the mouse). But it doesn't produce a GIF — it just prepares the system and waits for the user to drag.

For a **non-interactive** GUI-script demo with automatic dragging and GIF output, we would need to:
- Use the GUI's drag mode (`ra_drag`) programmatically
- Or use `run_multimol_md` directly (like `run_drag_demo.py` does) but called from within the GUI script context
- Capture frames from the VisPy canvas at intervals
- Render the GIF with the surface visible

### 4. Show surface interaction for all molecules

All molecules have FAF enabled (K_Z=5.0), so they all feel the surface. But without lateral driving, they don't move. To show surface interaction:
- Add small random lateral perturbations to all molecules
- Or drag all molecules slowly in different directions
- Or show the FAF potential landscape as a background heatmap

## Relevant Files

| File | Role |
|------|------|
| `tests/run_drag_demo.py` | Standalone offline drag demo (produces drag_demo.gif) |
| `spammm/GUI/gui_scripts/ptcda_interactive_drag.py` | GUI script for interactive drag setup |
| `spammm/GUI/RigidAssemblyExtension.py` | `_make_ramdrag_mode` (line ~447) — interactive drag handler |
| `spammm/forcefields/RigidBodyDynamics.py` | `run_multimol_md`, `attach_pairff_faf`, `reset_dynamics_state` |
| `spammm/surfaces/FoldedRigid.py` | `Z_SURF_TOP = -3.25`, `load_fit`, FAF substrate potential |
| `data/fits/ptcda_nacl_factorized.npz` | FAF fit for PTCDA on NaCl |

## Recommendation

The most impactful fix is to **render the NaCl lattice** in the animation and **slow down the drag** so the stick-slip is clearly lattice-periodic. The GUI script version should use the same `run_multimol_md` physics but capture frames from the VisPy canvas (which already shows atoms) and composite them with a surface rendering.
