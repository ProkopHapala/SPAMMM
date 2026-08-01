# FAF Multi-Molecule MD — Diagnostic Report

**Status:** UNRESOLVED — surface interaction (stick-slip) not visible in demo
**Date:** 2025-08-01
**Context:** Fusing FAF substrate forces into concurrent multi-molecule rigid-body MD (kernel 15)

---

## 2026-08-01 implementation under USER review

The original report mixed valid GUI integration faults with two invalid physics
comparisons. Candidate corrections are implemented and numerically tested, but
visible stick-slip remains **UNRESOLVED** pending USER interaction/review.

- The GUI build had a 3.25 Å pose split: ensemble/display at `+3.0 Å`, but
  FAF GPU/host poses at `Z_SURF_TOP+3.0=-0.25 Å`. All pose stores now start at
  the same absolute FAF height and are checked after build.
- Scene picking returned a dense **real-atom** index, but the anchor buffer is
  indexed over real + epair/σ-hole sites. The drag path now maps
  `display index → body → flat GPU site`.
- `from_molecules()` uploaded anchors but did not initialize `rbd.anchors`;
  first drag therefore raised `AttributeError`. Host and GPU anchors now start
  together with `w=-1` (disabled).
- GUI dragging previously ran the active-only `run_pairff()` path, so partner
  molecules were frozen. It now uses FAF-capable concurrent kernel 15 through
  `run_multimol_md(..., faf=True/None)`, downloads all poses, and commits them
  to `RigidEnsemble` and `AtomicGraph`.
- Momentum/FIRE/Newton state is reset at pose and drag discontinuities; spring
  release cannot leave hidden momentum for the next manipulation.
- PTCDA is loaded with QEq in the interactive GUI script; measured charge range
  is `[-0.368,+0.363] e`.

The earlier “40% tensor Fx mismatch” is not reproduced by the current
tensor-vs-flat parity test: relative errors are approximately
`E=6e-8`, `F=4e-8`, `torque=1.8e-7` on NVIDIA. The comparison of a single
bare O–Na pair with a neutral periodic NaCl lattice is also not a valid FAF
amplitude reference because the other lattice ions screen/cancel the pair.
A direct periodic/Ewald physical parity test is still needed before making a
claim about absolute lateral corrugation.

Current evidence:

```text
all-mobile anchored drag step: dragged body moves, partner body moves,
ensemble == GPU == _mb_* (atol=1e-6)
multimol FAF relaxation: E 522.902649 → -1.193753 eV
real GUI setup: GTX 1650, 4×PTCDA, FAF=True, max |Q|=0.368 e
```

Run the interactive review:

```bash
./run_gui.sh --script demos/gui_scripts/ptcda_interactive_drag.py
```

Pick an O atom and pull it toward the center. Whether the motion shows the
expected lattice-scale stick-slip remains a USER visual/physical verdict.

---

## Symptom

Dragging a PTCDA molecule across NaCl surface with FAF should show **discrete
stick-slip steps** (O atoms snapping to Na+ sites, ~4 Å lattice). Instead the
molecule moves smoothly with no visible surface corrugation. The FAF force
appears too weak compared to inter-molecular PairFF and z-confinement forces.

---

## Root Cause Candidates

### 1. FAF charges were zero (FIXED, verified)

`attach_pairff_faf()` in `RigidBodyDynamics.py` was re-deriving PLQH from
runtime REQs. When molecules loaded with `qeq=False` (Q=0 for all atoms), the
FAF Coulomb component vanished entirely (E_coul ~0.0005 eV instead of ~1.76 eV).

**Fix applied:** Added `plqh_override` parameter to `_folded_plqh_all_sites()`.
When the fit file contains `atom_plqh` (with correct charges from fitting), it
is used directly instead of re-deriving from runtime REQs.

**Verification:** After fix, O[25] PLQH Q=-0.3675 (was 0.0). FAF Coulomb energy
at z=1.0 is -0.013 eV (still weak — see issue #2).

**Files changed:**
- `spammm/forcefields/RigidBodyDynamics.py` lines ~2302-2334 (`_folded_plqh_all_sites`)
- `spammm/forcefields/RigidBodyDynamics.py` line ~2433 (`attach_pairff_faf`)

### 2. FAF force magnitude is fundamentally weak (OPEN)

Even with correct charges, the FAF potential is **orders of magnitude weaker**
than expected for O-Na+ Coulomb interaction:

| z (Å above surface) | E_coul (FAF) | E_coul (expected O-Na+) | Ratio |
|---------------------|--------------|--------------------------|-------|
| 1.0                 | -0.013 eV    | -1.76 eV                 | 0.007 |
| 2.0                 | -0.094 eV    | -0.88 eV                 | 0.11  |
| 3.0                 | -0.0005 eV   | -0.59 eV                 | 0.001 |

**The FAF fit coefficients appear to under-represent the Coulomb interaction
by 10-1000×.** The fit's `coeffs4[:, 2]` (Coulomb column) has max abs 1.54, but
when multiplied by PLQH Q=-0.37 and summed over the basis, the result is tiny.

**Possible causes:**
- The fit was trained on a different charge state (maybe neutral QEq charges
  were used during fitting, but the Coulomb basis functions absorb the charge
  scaling differently)
- The Coulomb basis expansion converges poorly (needs more harmonics)
- The `eval_folded_potential` reference and the kernel tensor evaluator
  disagree (see issue #4)

### 3. z-confinement (k_z) fights FAF (OPEN)

The FAF equilibrium height is z_init ≈ 3.6 Å (where Fz=0). But at that height
the lateral corrugation is only 0.24 eV (Fx_max=0.19 eV/Å) — too weak for
visible stick-slip.

At z_init=3.0 (closer, corrugation 0.75 eV, Fx_max=0.59 eV/Å), the molecule
is in the Pauli repulsive wall (Fz=4.6 eV/Å pushing it away). To keep it there,
k_z must be strong, but then the z-confinement energy dominates:
- k_z=5.0 → E_z_conf = 51 eV (overwhelms everything)
- k_z=0.5 → molecule lifts off (z goes from 0.35 to 3.24)
- k_z=0.0 → molecule flies away from surface

**The z-confinement is applied per-atom** (`f.z += -k_z * (p_world.z - z_target)`),
which is correct, but the energy `0.5 * k_z * (z - z_target)^2` per atom × 38
atoms = large total even for small k_z.

### 4. Kernel tensor evaluator ≠ reference evaluator (OPEN, CRITICAL)

The kernel's `folded_eval_tensor_rigid` gives different forces than the Python
`eval_folded_potential` reference:

| Quantity | Kernel | Reference | Discrepancy |
|----------|--------|-----------|-------------|
| Fx       | -0.031 | -0.052    | 40%         |
| Fz       | 167.7  | 164.9     | 1.7%        |
| E        | 48.74  | 48.72     | 0.05%       |

The energy matches well but **Fx is 40% off**. This suggests the tensor
evaluator's lateral force computation has a bug — possibly in the chain rule
for u/v → x/y conversion, or in the harmonic recurrence.

**Location:** `kernels/rigid.cl` lines 878-929 (`folded_eval_tensor_rigid`)

**Key lines:**
```c
// Line 925-927: force = -dE/dr = -(dE/du * du/dx + dE/dv * dv/dx)
f.x = -(dEdu * invLvec2d.x + dEdv * invLvec2d.z);
f.y = -(dEdu * invLvec2d.y + dEdv * invLvec2d.w);
f.z = -dEdz;
```

The `invLvec2d` is computed at line 4241-4244:
```c
float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
float det = ax*by - bx*ay;
invLvec2d_faf = (float4)( by/det, -bx/det, -ay/det, ax/det );
```

For the NaCl fit, `folded_lvec2d = [4, 0, 0, 4]` (diagonal), so
`invLvec2d = [1/4, 0, 0, 1/4]`. This looks correct.

**Possible bug:** The `dEdu` and `dEdv` derivatives might be missing a factor
of 2π. Check lines 908-911:
```c
float by  = cv_arr[iv].x;
float dby = -2.0f*M_PI_F*(float)iv * cv_arr[iv].y;  // d/dv of cos(2π*kv*v)
float bx  = cu_arr[iu].x;
float dbx = -2.0f*M_PI_F*(float)iu * cu_arr[iu].y;  // d/du of cos(2π*ku*u)
```
These look correct. The issue may be in how `u = x/Lx` vs `u = x * invLvec.x`
interacts with the derivative chain rule.

### 5. apos_world.w (energy) not populated before first MD step (MINOR)

`apos_world[ia].w` is only written inside the force kernel. Before the first
MD step, it contains zeros. The demo script worked around this by computing
anchor base position from `apos_body` + CoM manually.

**Location:** `kernels/rigid.cl` line 591 (initialization writes `.w=0.f`),
lines 704, 1082, etc. (force kernels write `.w=E`)

---

## Key Numbers

### FAF potential landscape (single O[25] atom, from Python reference)

| z_init (Å) | world_z | E_min (eV) | E_max (eV) | corrugation | Fx_max (eV/Å) |
|------------|---------|------------|------------|-------------|---------------|
| 3.6        | 0.35    | -0.298     | -0.054     | 0.244       | 0.189         |
| 3.0        | -0.25   | -0.154     | +0.601     | 0.755       | 0.594         |
| 2.5        | -0.75   | -0.352     | +0.145     | 0.497       | 0.393         |
| 2.0        | -1.25   | -0.352     | +0.145     | 0.497       | 0.393         |

### FAF potential (whole PTCDA, 38 atoms, from Python reference)

| z_init (Å) | E_total (eV) | Fz_total (eV/Å) | Fx_total (eV/Å) |
|------------|--------------|-----------------|-----------------|
| 2.0        | 48.74        | 165.0           | -0.052          |
| 3.0        | 0.60         | 4.58            | 0.003           |
| 3.5        | -0.29        | 0.21            | 0.002           |
| 3.6 (eq)   | -0.30        | ~0              | ~0              |
| 4.0        | -0.23        | -0.25           | 0.001           |

### PairFF inter-molecular energy (4 PTCDA, 16 Å spacing)

| Molecule | E (eV) | Cause |
|----------|--------|-------|
| mol 0    | -0.006 | No neighbors in cutoff |
| mol 1    | 34.75  | Pauli overlap with mol 0 |
| mol 2    | 69.51  | Pauli overlap with mol 1 and mol 3 |
| mol 3    | 34.76  | Pauli overlap with mol 2 |

**PairFF cutoff ~9 Å** (beta=1.7, R0~3.85, rho_c~8.3). PTCDA is 11.7 Å long,
so 16 Å spacing → 4.3 Å gap between ends → deep Pauli overlap.

---

## Fit File Details

```
File: data/fits/ptcda_nacl_factorized.npz
Mode: factorized (FAF_MODE_FACTOR)
z_range: (-1.75, 4.75)  # relative to Z_SURF_TOP = -3.25
folded_lvec2d: [4, 0, 0, 4]  # 4×4 Å NaCl unit cell
alpha_morse: 1.8
basis_params shape: (112, 4)  # (ku, kv, alpha, z0) per basis function
coeffs4 shape: (112, 4)  # (pauli, london, coulomb, h_bond) per basis
atom_plqh shape: (38, 4)  # (P, L, Q, H) per real atom

O[25] PLQH: P=27.78, L=1.1903, Q=-0.3675, H=0.0
C[0]  PLQH: P=69.12, L=2.1596, Q=0.0137, H=0.0
H[30] PLQH: P=7.877, L=0.5866, Q=0.0919, H=0.0

Coulomb coeffs: max abs 1.54, sum 2.84, all 112 nonzero
Pauli coeffs:  max abs 0.173
London coeffs: max abs 0.369
```

---

## What to Investigate Next

1. **Verify the FAF fit itself** — is the Coulomb interaction supposed to be
   this weak? Compare with the original LAMMPS simulation that the fit was
   trained on. The O-Na+ Coulomb at 3 Å should be ~-1.76 eV, but FAF gives
   -0.0005 eV. Either:
   - The fit was trained with screened/neutral charges (not bare Coulomb)
   - The Coulomb basis expansion is fundamentally inadequate
   - There's a unit/scale error in the fit or evaluator

2. **Fix the kernel tensor evaluator Fx discrepancy** (40% off from reference).
   This is a real bug that affects all FAF MD simulations, not just the demo.

3. **Check if `eval_folded_potential` (Python reference) is itself correct.**
   It may have the same bug as the kernel, or a different one. Compare both
   against a direct sum of O-Na+ pair interactions.

4. **The z-confinement approach is wrong for FAF.** FAF should hold the
   molecule on the surface by itself (Fz attractive at equilibrium). If it
   can't, the fit is too weak. Don't use k_z as a workaround — fix the FAF.

---

## Files Modified (Not Yet Verified Working)

- `spammm/forcefields/RigidBodyDynamics.py` — `_folded_plqh_all_sites()` now
  accepts `plqh_override` parameter; `attach_pairff_faf()` passes fit's
  `atom_plqh` when available
- `tests/run_drag_demo.py` — demo script (not working correctly, many
  parameter iterations attempted)

## Files NOT Modified (Potential Bug Locations)

- `kernels/rigid.cl` lines 878-929 — `folded_eval_tensor_rigid` (Fx 40% off)
- `spammm/surfaces/FoldedRigid.py` — `eval_folded_potential` (reference)
- `data/fits/ptcda_nacl_factorized.npz` — the fit file itself (Coulomb too weak?)
