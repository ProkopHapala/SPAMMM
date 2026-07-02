# GUI Forcefield Relaxation — Implementation Report

## Overview

This document describes the integration of GPU-accelerated molecular relaxation into the SPAMMM GUI, covering the serial OpenCL kernel, bidirectional geometry synchronization, topology change handling, atom type assignment, debugging infrastructure, and remaining issues.

---

## 1. The Serial Local-Memory Kernel (`relax_nsteps_serial`)

### Problem

The original relaxation pipeline used three separate OpenCL kernel calls per MD step, dispatched from Python:

1. `cleanForceSPFFf4` — zero force buffers
2. `getSPFFf4` — compute bonded forces (bonds, angles, pi-alignment)
3. `updateAtomsSPFFf4` — gather recoil forces, integrate velocity Verlet

For a molecule like benzene (12 atoms, 6 pi-nodes, 18 vectors), each step required 3 kernel launches with `queue.finish()` synchronization. The Python dispatch overhead dominated: ~1000 steps took several seconds, making interactive relaxation impractical.

### Solution

A single kernel `relax_nsteps_serial` in `kernels/SPFF.cl` (lines 1024–1250) runs **N steps entirely on GPU** in one kernel call, with all data in `__local` memory.

**Key design:**
- **Single workgroup** of 128 threads (`WG_SIZE=128`)
- All molecular data (positions, velocities, forces, neighbors, bond params, constraints) is loaded cooperatively into `__local` memory at kernel start
- The main loop runs `nsteps` iterations with `barrier(CLK_LOCAL_MEM_FENCE)` between phases — no global memory round-trips
- Results are written back to global memory only once at the end

**Local memory layout (total ~20 KB, fits in 48 KB GPU limit):**
```
s_apos     [128]  float4   — positions (atoms + pi)
s_avel     [128]  float4   — velocities
s_aforce   [128]  float4   — forces
s_fneigh   [512]  float4   — recoil forces (nnode*4*2 max)
s_neighs   [128]  int4     — neighbor indices
s_bkNeighs [128]  int4     — back-neighbor indices
s_apars    [128]  float4   — FF angle params
s_bLs      [128]  float4   — bond lengths
s_bKs      [128]  float4   — bond stiffness
s_Ksp      [128]  float4   — sigma-pi stiffness
s_Kpp      [128]  float4   — pi-pi stiffness
s_constr   [128]  float4   — constraints
s_constrK  [128]  float4   — constraint stiffness
```

**Per-step data flow:**
1. **Phase 1** (all threads): Zero `s_aforce` and `s_fneigh`
2. **Phase 2** (threads 0..nnode-1): Compute SPFF bonded forces — bonds, angles, pi-alignment — store recoil forces in `s_fneigh`
3. **Phase 3** (threads 0..nvec-1): Gather recoil via back-neighbors, apply force limiting, apply constraints, damped velocity Verlet integration, normalize pi vectors

**Constraints:**
- `nvec = natoms + nnode <= 128` (WG_SIZE)
- `nnode <= 64` (MAX_NNODE)
- `nSystems == 1` (single molecule)
- No non-bonded interactions (slot reserved in kernel signature for future)

### Performance

Measured on NVIDIA RTX 3090 with benzene (12 atoms, 18 nvec):

| Mode | Steps | Time | Speedup |
|------|-------|------|---------|
| Batch (3 kernels/step from Python) | 1000 | ~5s | 1x |
| Serial (1 kernel, all local) | 1000 | 0.05s | **~100x** |
| Serial | 10000 | 0.5s | ~100x |
| Serial | 67000 | 2.2s | ~100x |

The speedup is consistent across step counts because the per-step overhead (kernel launch + Python dispatch) is eliminated. The kernel itself runs in ~50µs/step for benzene.

### Python Dispatch (`SPFF_cl.relax_serial`)

Located in `spammm/forcefields/SPFF_cl.py` (lines 1724–1757). Validates constraints, sets MD params, packs kernel arguments, and launches with `(WG_SIZE,)` global and local work size.

### Auto-Fallback (`FFController`)

`FFController._can_use_serial(do_nb)` checks:
- `md.nSystems == 1`
- `md.nvecs <= 128`
- `md.nnode <= 64`
- `not do_nb` (non-bonded not yet supported in serial)

If any condition fails, `relax_n` and `relax_until_converged` automatically fall back to `relax_batch` (the original 3-kernel-per-step approach).

---

## 2. Bidirectional Geometry Synchronization

### Problem

The GUI has two representations of molecular geometry:
- **`AtomicGraph`** — authoritative, uses `Atom` objects with stable IDs, managed by `KekuleBackend`
- **`AtomicSystem`** (`sys`) — NumPy arrays (`apos`, `enames`, `bonds`), used by forcefield and Vispy rendering

The forcefield operates on GPU buffers. Three synchronization directions are needed:

1. **AtomGraph → GPU** (before relaxation): user drags atoms in GUI, positions change in `AtomicGraph`, must be uploaded to GPU
2. **GPU → AtomGraph** (after relaxation): relaxed positions must be written back to authoritative `AtomicGraph`
3. **AtomGraph → Vispy** (for display): `refresh_view()` reads from `sys.apos` which is rebuilt from `AtomicGraph`

### Implementation

**Before relaxation** (`FFExtension._ensure_built`):
```python
if ctrl.is_built:
    ctrl.update_positions(sys.apos[:, :3])  # upload current positions to GPU
    return True
```
`_get_sys()` calls `backend._sync_sys()` which rebuilds `sys.apos` from `AtomicGraph.to_arrays()`, ensuring the latest dragged positions are used.

**After relaxation** (`FFExtension._sync_positions_to_graph`):
```python
backend.graph.update_positions_from_array(positions)
backend._sync_sys()  # rebuild sys from graph
```
This writes relaxed positions back to `Atom` objects in the `AtomicGraph`, then rebuilds `sys` for rendering.

**During interactive relaxation** (`_interactive_tick`):
Positions are synced every frame for live visual feedback.

### The "First Relax Slow" Issue

Initial observation: first "Relax" click was slow, second was fast. Timing prints revealed:
- `[FF] Build took 0.12s` — FF build (including OpenCL kernel compilation cache lookup) is fast
- The slowness was from the first relaxation doing more steps (molecule starts far from equilibrium)
- Second click starts from already-relaxed geometry, converges in fewer steps

The OpenCL kernel compilation happens once (cached by pyopencl), so it's not the bottleneck.

---

## 3. Topology Change Handling

### Problem

When the user edits topology (add/delete/merge atoms) after the forcefield is built, the GPU buffers have the wrong size. Relaxation would crash with `ValueError: could not broadcast input array from shape (18,3) into shape (16,3)`.

### Solution

`_ensure_built` detects atom count mismatch and tears down the stale FF before rebuilding:
```python
if ctrl.is_built and hasattr(ctrl, 'natoms') and ctrl.natoms != len(sys.apos):
    print(f"[FF] Topology changed: natoms {ctrl.natoms} → {len(sys.apos)}, rebuilding FF")
    ctrl.teardown()
```

This triggers a full rebuild on the next relax/step, ensuring GPU buffers match the current topology.

---

## 4. Atom Type Assignment (sp2 vs sp3)

### Problem

Aromatic carbons were being treated as sp3 (`C_3`) instead of sp2/aromatic (`C_R`), leading to wrong bond angles (109.5° instead of 120°) and incorrect forcefield behavior.

### Solution

Atom types are now assigned based on `n_pi` (pi-electron count) from the `KekuleBackend`, which tracks hybridization state:

**Mapping** (`FFExtension._NPI_TO_TYPE`):
```python
_NPI_TO_TYPE = {
    ('C', 0): 'C_3',   # sp3
    ('C', 1): 'C_R',   # aromatic
    ('C', 2): 'C_1',   # sp (alkyne)
    ('N', 0): 'N_3',   # sp3
    ('N', 1): 'N_R',   # aromatic
    ('N', 2): 'N_1',   # sp
    ('O', 0): 'O_3',   # sp3
    ('O', 1): 'O_R',   # aromatic
    ('O', 2): 'O_1',   # sp
}
```

**Flow:**
1. `_get_sys()` reads `backend.atom_npi` (set by Kekule solver)
2. Maps each atom's `(element, n_pi)` to SPFF type name (e.g., `C_R` for aromatic C)
3. Sets `sys.atom_types_spff` list
4. `FFController.build_ff()` prefers `atom_types_spff` over generic element names
5. `SPFFparams.getAtomType()` maps type name → integer index for GPU

**Verification** (benzene debug output):
```
atom_types_spff: ['C_R', 'C_R', 'C_R', 'C_R', 'C_R', 'C_R', 'H', 'H', 'H', 'H', 'H', 'H']
spff.npi_list: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
sys.atypes: [34 34 34 34 34 34 11 11 11 11 11 11]
```
All 6 carbons correctly typed as `C_R` (aromatic, type index 34).

---

## 5. Debug Infrastructure

### "Debug FF" Checkbox

Added to the FF panel next to "Serial" checkbox. When checked, `_debug_print_ff()` is called after every FF build and prints:

- **Dimensions**: `natoms`, `nnode`, `ncap`, `nvecs`, `ntors`
- **Serial eligibility**: `_can_use_serial` result, `enable_nonbond` flag
- **Atom types**: `sys.enames`, `sys.atypes`, `atom_types_spff`
- **Hybridization**: `spff.npi_list`, `spff.nep_list`
- **Full arrays** via `SPFF.printArrays()`: `apos`, `REQs`, `neighs`, `bLs`, `bKs`, `apars`, `Ksp`, `Kpp`
- **Per-atom detail** (all atoms): type, npi, position, neighbors, and for node atoms: angle params (`apars`), bond lengths (`bLs`), bond stiffness (`bKs`), sigma-pi stiffness (`Ksp`), pi-pi stiffness (`Kpp`), pi orbital direction (`pipos`)
- **Pi orbital vectors** in `apos[natoms:]`
- **Back neighbors** (all entries with valid indices)

### Timing Prints

All FF operations now print timing:
- `[FF] Build took X.XXs | natoms=... nnode=... nvecs=... | serial=...`
- `[FF] Relax took X.XXs (serial/batch, N steps)`
- `[FF] Step took X.XXXs (N steps)`
- `[FF] Topology changed: natoms X → Y, rebuilding FF`
- `[FF] Serial checkbox checked but _can_use_serial=False (...)`

---

## 6. GUI Panel Layout

The FFExtension panel (`spammm/GUI/FFExtension.py`) provides:

```
Row 1: [FF: SPFF▼] [Build]
Row 2: nSteps: [100] [Step]
Row 3: dt: [0.010] damp: [0.95]
Row 4: [Relax] [Interactive] [✓Serial] [☐Debug FF]
       E: --- | Fmax: ---
       ─────────────────
       [Pin Sel] [Unpin All]
       Status: Not built
```

- **Build**: Explicitly builds FF from current molecule
- **Step**: Runs N steps, updates view once
- **Relax**: Runs until convergence (fmax < 0.05) or max_steps, updates view every batch
- **Interactive**: QTimer-driven continuous relaxation with live Vispy update
- **Serial**: Toggle serial kernel (auto-unchecks if molecule too large)
- **Debug FF**: Toggle verbose array printing after build

---

## 7. Remaining Issues

### 7.1 Hydrogen Tilt in Aromatic Systems

Some hydrogen atoms in naphthalene relax to a tilted position instead of staying in-plane. The debug output shows all pi vectors are `(0, 0, 1)` (correct for planar aromatic) and all atom types are `C_R` (correct). The issue may be in:
- Angle parameters for `C_R`–`C_R`–`H` triplets (the `apars` array stores `cos(θ/2)` and stiffness per node, but H atoms are capping atoms with no angle terms of their own)
- The `Ksp` (sigma-pi) interaction that keeps the H in the plane of the pi orbital
- Missing or wrong neighbor assignment for capping H atoms

**Investigation needed**: Compare the `neighs`, `bLs`, `bKs`, `Ksp` arrays for a correctly-relaxing H vs a tilted one.

### 7.2 Atom Pinning (Untested)

The pinning infrastructure exists in `FFController`:
- `set_pinned(mask, positions)` — pin atoms by boolean mask
- `clear_pins()` — remove all pins
- `_apply_pinned()` — upload constraint buffers to GPU
- `update_positions()` re-applies pins after position sync

The GUI has "Pin Sel" and "Unpin All" buttons, but these are **not yet tested**. The serial kernel supports constraints (reads `s_constr` and `s_constrK` in Phase 3), but the full flow (select atoms → pin → relax with pins) needs validation.

### 7.3 Interactive MD Authority Conflict

Interactive relaxation uses a `QTimer` that calls `relax_n` every frame and updates positions directly on the Vispy scene. However, in the current edit modes ("Atom", "Bond", "Hex", etc.), mouse dragging also moves atoms. This creates a conflict:
- User drags an atom → `AtomicGraph` position updates
- Timer fires → GPU relaxation moves atoms → overwrites the drag
- Result: atoms jump between dragged and relaxed positions

**Proposed solution**: Add a dedicated **"InteractiveMD"** edit mode (alongside "Atom", "Bond", "Hex", etc.) that:
- Disables atom dragging (like Bond/Hex modes do via `scene.lock_drag = True`)
- Enables the relaxation timer
- Mouse interaction is limited to camera control (zoom, pan, rotate)
- Atoms move only from relaxation forces
- Exiting InteractiveMD mode stops the timer and syncs final positions back to `AtomicGraph`

This avoids mixed dragging and relaxation, giving the user a clear mental model: "in InteractiveMD mode, the forcefield drives the atoms; in Atom mode, I drive them."

### 7.4 Non-Bonded Interactions in Serial Kernel

The serial kernel has reserved argument slots for `g_REQs` and `g_excl` (non-bonded parameters and exclusion lists) but does not use them. Adding non-bonded support would require:
- Loading exclusion lists into local memory
- Computing pairwise non-bonded forces within the workgroup
- This is feasible for small molecules (≤80 atoms) since all positions are already in `s_apos`

### 7.5 UFF Forcefield Path

`FFController.build_ff()` raises `NotImplementedError` for `ff_type='uff'`. The UFF path needs integration with the MD engine (`SPFF_cl`), which currently expects SPFF-format arrays.

### 7.6 Multi-Molecule Batch Relaxation

The serial kernel handles only `nSystems=1`. For batch relaxation of multiple molecules (e.g., conformational screening), the batch path (`relax_batch`) is used. A future optimization could launch multiple serial workgroups (one per molecule) for intermediate-sized batches.

---

## 8. File Map

| File | Role |
|------|------|
| `kernels/SPFF.cl` (lines 1024–1250) | `relax_nsteps_serial` OpenCL kernel |
| `spammm/forcefields/SPFF_cl.py` (lines 1724–1757) | `relax_serial()` Python dispatch |
| `spammm/forcefields/SPFF_cl.py` (lines 1703–1722) | `relax_batch()` original 3-kernel approach |
| `spammm/forcefields/FFController.py` | `FFController`: build, relax_n, relax_until_converged, update_positions, pinning |
| `spammm/forcefields/SPFFbuilder.py` | `SPFF`: topology building, array packing, `printArrays()` |
| `spammm/GUI/FFExtension.py` | GUI panel: Build/Step/Relax/Interactive buttons, Debug FF, sync logic |
| `spammm/topology/KekuleBackend.py` | `AtomicGraph` ↔ `AtomicSystem` sync, `atom_npi` |
| `spammm/topology/AtomicGraph.py` | `to_arrays()`, `update_positions_from_array()` |
| `data/AtomTypes.dat` | Atom type definitions (C_3, C_R, C_2, C_1, N_3, N_R, etc.) |
| `tests/test_relax_serial.py` | Parity test: serial vs batch for multiple molecules |

---

## 9. Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     GUI (PyQt5 + Vispy)                  │
│                                                          │
│  FFExtension.py          KekuleExtension.py              │
│  ┌─────────────┐         ┌──────────────┐               │
│  │ Build/Step/ │         │ Kekule Solver │               │
│  │ Relax/Inter │         │ → atom_npi    │               │
│  └──────┬──────┘         └──────┬───────┘               │
│         │                       │                        │
│         ▼                       ▼                        │
│  ┌──────────────────────────────────────┐               │
│  │         KekuleBackend                │               │
│  │  ┌────────────┐  ┌────────────────┐ │               │
│  │  │ AtomicGraph │  │ AtomicSystem   │ │               │
│  │  │ (authority) │←→│ (sys, arrays)  │ │               │
│  │  │  Atom objs  │  │  apos, enames  │ │               │
│  │  │  pos, npi   │  │  bonds, atypes │ │               │
│  │  └────────────┘  └────────────────┘ │               │
│  └──────────────────────────────────────┘               │
│         │                       │                        │
│         ▼                       ▼                        │
│  ┌──────────────────────────────────────┐               │
│  │          FFController                │               │
│  │  build_ff(sys) → SPFF + SPFF_cl      │               │
│  │  relax_n()     → serial or batch     │               │
│  │  update_positions() → GPU upload     │               │
│  │  get_state()   → GPU download        │               │
│  └──────────────────────────────────────┘               │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────────────────────────┐               │
│  │          SPFF_cl (OpenCL)            │               │
│  │  relax_serial()  → 1 kernel, local   │               │
│  │  relax_batch()   → 3 kernels/step    │               │
│  └──────────────────────────────────────┘               │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────────────────────────┐               │
│  │       GPU (RTX 3090)                 │               │
│  │  relax_nsteps_serial kernel          │               │
│  │  128 threads, all __local memory     │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Key Design Decisions

1. **Single workgroup, not multi-workgroup**: The serial kernel uses one workgroup of 128 threads. This keeps all data in local memory with no cross-workgroup synchronization needed. Trade-off: limits molecule size to ~80 atoms (nvec ≤ 128).

2. **Velocity Verlet with damping**: The integration scheme is damped MD (velocity *= damp, then v += f*dt/m, then x += v*dt). This is simple and robust for geometry optimization. FIRE (fast inertial relaxation engine) or BFGS could be added later for faster convergence.

3. **Pi vectors as unit vectors**: Pi orbital directions are stored in `apos[natoms:]` and normalized after each step (`normalize(pe.xyz)`). The pi-pi alignment force (`evalPiAling`) keeps neighboring pi vectors parallel, maintaining planarity.

4. **Back-neighbor mapping**: The `back_neighs` array maps each atom/pi-vector to its entries in the `fneigh` (recoil force) array. This allows O(1) force gathering in Phase 3 without searching.

5. **Auto-fallback, not manual**: The serial/batch decision is automatic based on molecule size. The user sees a checkbox that auto-unchecks when the molecule is too large, with a status message explaining why.

6. **Bidirectional sync, not one-way**: Positions flow AtomGraph → GPU before relaxation, and GPU → AtomGraph after. This ensures the forcefield always operates on the latest geometry, and the GUI always displays the latest relaxed geometry.
