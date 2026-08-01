# USER

OK, I thins we shouldsplit @RigidBodyDynamics.py right now, there are many other high level fatures which shouldnot be part of the code like 

everything which is calling
rbd.****

from_two_molecules
from_molecules

maybe some other, analyse it deeply and consider how to split it. Where we have actually implemented our monte-carlo/greedy/GA bakcend, or manipulation of molecule on FAF surface?

Maybe our test scripts and demos hae some function which can be refactored in new shared module

do not code yet, make deep inventory and analysis or fexiting code related to RigidBody 

---

# GLM 5.2

## Current state of `RigidBodyDynamics.py` (3066 lines)

### What's in there now (by concern):

**1. GPU core engine (~900 lines, 29%)** — belongs here
- `RigidBodyDynamics` class: buffer alloc, kernel launches, upload/download, FIRE/Newton relaxation
- `RigidBodyPairFF` class: PairFF kernel binding, env buffers, FAF fusion, energy replica kernel
- Low-level utilities: `_pack_float3`, `_ensure_float4`, `_reqs_to_plq`, `_quat_to_matrix_np`, `compute_mass_properties`, etc.

**2. Molecule loading + graph splitting (~100 lines)** — doesn't belong here
- `load_molecule` (line 228) — we just moved it here, but it's not GPU code
- `graph_to_rigid_fragments` (line 258) — AtomicGraph → rigid bodies, not GPU code
- `_bonds_from_geom` (line 218) — geometry inference, not GPU code

**3. High-level workflow methods (~1100 lines, 36%)** — mixed concerns
- `from_molecules` (line 2940) — builds multi-body PairFF from molecule list. Calls `realloc_molecules`, `upload_dyn_types_req`, `init_pairff`. Used by GUI, demo, tests.
- `from_two_molecules` (line 2862) — builds two-body PairFF. **Not used anywhere** (legacy/unused).
- `greedy_energy_step` (line 2751) — greedy MC step. Calls `eval_energy_replicas`. Used by testplot, GUI.
- `tip_pull_scan` (line 2802) — AFM manipulation. Calls `relax_pairff`, `world_sites_all_bodies`. **Not used anywhere**.
- `eval_energy_replicas` (line 2676) — GPU energy evaluation for MC/GA. This IS GPU core.
- `attach_pairff_faf` (line 2411) — attaches FAF substrate. This is GPU configuration.
- `relax_pairff` (line 2547) — FIRE relaxation. This is GPU workflow but tightly coupled.

**4. PairFF pack utilities (~200 lines)** — tightly coupled to GPU buffer layout
- `_prepare_molecule_pack` (line 1638) — adds electron pairs, centers CoM. Needed by `from_molecules`.
- `add_electron_pairs_via_atomic_system` (line 1563) — lone pair placement.
- `_extend_reqs_with_epairs` (line 3041) — extends REQ array.
- `_body_sites_world` (line 1631) — body→world transform.

## Where MC/GA/manipulation code lives

| Code | Location | Used by | Status |
|------|----------|---------|--------|
| `greedy_energy_step` | `RigidBodyPairFF` method | testplot, GUI | Implemented |
| `eval_energy_replicas` | `RigidBodyPairFF` method | testplot, GUI | Implemented (GPU core) |
| MC assembly loop | **Inline in testplot** (lines 390-423) | testplot only | Not shared |
| MC assembly loop | **Inline in GUI** (lines 291-338) | GUI only | Duplicated |
| Metropolis/SA | Not implemented | — | Design only |
| GA | Not implemented | — | Design only |
| `tip_pull_scan` | `RigidBodyPairFF` method | **Unused** | Implemented |
| `lateral_scan` | `FoldedRigid.py` function | GUI, tests | Good pattern |
| `relaxed_scan` | `FoldedRigid.py` function | GUI, tests | Good pattern |
| `manipulation_trajectory` | `FoldedRigid.py` function | tests | Good pattern |

## Duplicated code across tests/demos/GUI

| Function | Copies | Locations |
|----------|--------|-----------|
| `grid_pos` | 3 | testplot (line 72), demo (line 68), GUI (line 241) |
| `load_molecule` | 2 | `RigidBodyDynamics.py` (line 228), demo (line 48) |
| MC assembly loop | 2 | testplot (lines 390-423), GUI (lines 291-338) |
| `assembly_real_atoms` | 1 | testplot (line 83) — should be shared |
| `write_xyz` | 1 | testplot (line 106) — should be shared |

## The right pattern (already exists in FoldedRigid.py)

`FoldedRigid.py` is the model to follow:
- **Standalone functions** that take an `rbd` object and orchestrate workflows
- `setup_rigid_folded(rbd, ...)`, `relax_folded(rbd, ...)`, `lateral_scan(rbd, ...)`, `relaxed_scan(rbd, ...)`
- Called by GUI, tests, demos — all import from the same module
- The `rbd` object is pure GPU engine — no workflow logic in its methods

## Proposed split

### `RigidBodyDynamics.py` (core, ~1800 lines) — GPU engine only

Keep:
- `RigidBodyDynamics` class — all GPU buffer management, kernel launches, relaxation
- `RigidBodyPairFF` class — PairFF kernel binding, env buffers, FAF fusion, `eval_energy_replicas`, `attach_pairff_faf`, `relax_pairff`, `run_pairff`, `world_sites_all_bodies`
- GPU-coupled utilities: `_pack_float3`, `_ensure_float4`, `_reqs_to_plq`, `_quat_to_matrix_np`, `compute_mass_properties`, `_prepare_molecule_pack`, `add_electron_pairs_via_atomic_system`, `_body_sites_world`, `_extend_reqs_with_epairs`, `_load_npy_legacy`

Remove (move to utils):
- `load_molecule`, `graph_to_rigid_fragments`, `_bonds_from_geom`
- `from_molecules`, `from_two_molecules` (become functions in utils)
- `greedy_energy_step` (becomes function in utils)
- `tip_pull_scan` (becomes function in utils)

### `RigidBodyUtils.py` (new, ~500 lines) — workflow orchestration

Following the `FoldedRigid.py` pattern — standalone functions that take `rbd`:

**Molecule loading:**
- `load_molecule(path, qeq, planarize, name)` — moved from RBD
- `graph_to_rigid_fragments(graph, qeq, planarize)` — moved from RBD
- `_bonds_from_geom(apos, enames)` — moved from RBD

**Assembly builders:**
- `build_pairff_from_molecules(molecules, body_positions, ...)` — was `RigidBodyPairFF.from_molecules`
- `build_pairff_from_two_molecules(...)` — was `RigidBodyPairFF.from_two_molecules` (or delete if truly unused)

**MC/GA harness:**
- `greedy_energy_step(rbd, pos, quat, moved, ...)` — was method
- `run_greedy_mc_assembly(rbd, ensemble, n_steps, ...)` — extracted from testplot/GUI inline loops

**Manipulation:**
- `tip_pull_scan(rbd, pin_local_idx, path, ...)` — was method

**Grid utilities:**
- `grid_pos(n, spacing, z)` — consolidated from 3 copies
- `assembly_real_atoms(packs, pos, quat, bonds0)` — from testplot
- `write_xyz(packs, pos, quat, enames, path)` — from testplot

### Key question: `from_molecules` as classmethod vs function

`from_molecules` is currently `RigidBodyPairFF.from_molecules(...)` — a classmethod constructor. Two options:

**Option A:** Move to `RigidBodyUtils.build_pairff_from_molecules(...)` — creates and returns an `RigidBodyPairFF` instance. Callers change from `RigidBodyPairFF.from_molecules(...)` to `build_pairff_from_molecules(...)`. Cleaner separation, but breaks all call sites.

**Option B:** Keep `from_molecules` as classmethod on `RigidBodyPairFF` — it's a constructor pattern, and constructors are naturally class methods. Move only `greedy_energy_step`, `tip_pull_scan`, and the loading utilities out.

I'd recommend **Option B** for `from_molecules` (it's a constructor) but **Option A** for `greedy_energy_step` and `tip_pull_scan` (they're workflow orchestration, not construction). The loading utilities (`load_molecule`, `graph_to_rigid_fragments`) also move out.

What do you think? Should I proceed with this split, or do you want to adjust the boundary?