# Task: PairFF multi-body kernel — one active relaxer, many fixed neighbors

**Status:** Done (USER confirmed click-to-select + mixed `--mols` + allmol persistent dynamics; FAF path in `PairFF_FAF_Substrate.md`).  
**Implemented (current SSOT = shared allmol buffers):**
- Kernels 12–13 in `kernels/rigid.cl` — `rigid_body_pairff_unified_allmol[_faf]_kernel` (preferred)
- Kernel 9 — `*_env_*` kept for compat (rebuild env on switch; not demo default)
- `from_molecules` → flat packs + `realloc_molecules`; `set_active_body` = **index only** (no realloc / no vel zero)
- Vispy: LMB → active; map rebuild; **FIRE default ON**; optional FAF compose
- Mixed species via `--mols`; identical N via `--bodies`
- Docs: `demos/PairFF_manual.md`, topical audit, design report, FAF task
- Demo:
  - `python3 demos/demo_pairff.py --bodies 4 --active 0`
  - `python3 demos/demo_pairff.py --mols PTCDA.xyz HCOOH.xyz formamide.xyz --spacing 12`
  - `python3 demos/demo_pairff.py --bodies 4 --faf`
**Priority:** P0 before GUI integration (`PairFF_GUI_Integration.md`)  
**Depends on:** `RigidBodyPairFF`, `demos/demo_pairff.py`  
**Related:** `kernels/rigid.cl`, `doc/Tasks/PairFF_FAF_Substrate.md`, `doc/Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md`

---

## Objective

Evolve the demo from **1 dynamic + 1 static** molecule to **N rigid molecules** (1 workgroup each), where:

- **Exactly one** molecule is **active** (integrates translation + rotation; FIRE / MD).
- All others are **fixed** (frozen pose) but still exert **PairFF forces** on the active body.
- User picks which molecule is active (Vispy click; CLI `--active`).

This matches the ultimate assembly application: many adsorbates, relax one at a time while others provide a frozen environment.

### Mixed molecules (already supported)

`from_molecules` takes a **list of independent** `(apos, enames, REQs)` packs — species need not match. Constraints:

- Unified kernel only (`--pairff-mode unified`)
- Each molecule tile ≤ **128** sites (atoms + epairs + sigma holes)
- Switching active body is an **index write** (allmol); irregular site counts via `mols[]` offsets — **no** realloc

There is **no** `data/xyz/NTCDA.xyz` yet (ASCII template exists in `ascii_art_heterocycle.py`); use `PTCDA.xyz` or export an XYZ. `formamide.xyz` = HCONH2.

---

## Current limitation (SSOT)

### Kernel (`rigid_body_pairff_unified_kernel`)

- One **dynamic** rigid body per workgroup (`gid` = body index).
- One **static** partner baked into `static_apos[]`, `static_REQ[]`, `static_type[]` (world frame, fixed).
- Static sites cached in **`__local`** arrays sized by `MAX_STATIC_ATOMS` (128):

```opencl
__local float4 Lstatic_pos[MAX_STATIC_ATOMS];
__local float4 Lstatic_REQ[MAX_STATIC_ATOMS];
__local float  Lstatic_g[MAX_STATIC_ATOMS];
```

- Inner loop: `for (j = 0; j < n_static; j++)` — assumes **all** interaction partners fit in local memory at once.

### Python (`RigidBodyPairFF`)

- `from_two_molecules()`: 1 dynamic body + 1 static list.
- `realloc(n_bodies, num_atoms)`: assumes **uniform** `num_atoms` per body (`total_atoms = n_bodies * num_atoms`).
- `mols[]` offsets exist in kernel but host API is not yet general multi-molecule assembly.

**Cannot scale** to “10 molecules × ~25 sites = 250 partners” without kernel redesign.

---

## Target semantics

| Concept | Behavior |
|---------|----------|
| `n_bodies` | Number of rigid molecules in scene |
| `active_body` | Index `a` — only this workgroup runs pose integration |
| Fixed body `j ≠ a` | `pos/quat` constant; atoms contribute to forces on `a` |
| Self interaction | Skip pairs within same body |
| Dummy atoms | Epairs (`type=1`), sigma holes (`type=2`) on every body; REQ packing unchanged |
| Kernel default | **Unified** compact-exp (`rigid_body_pairff_unified_kernel`) |

**Demo CLI:**

```bash
python3 demos/demo_pairff.py --bodies 4 --active 0          # identical HCOOH; click to switch
python3 demos/demo_pairff.py --mols PTCDA.xyz HCOOH.xyz formamide.xyz --spacing 12
python3 demos/demo_pairff.py --bodies 4 --active 2 --no-vis
```

Load same or different XYZs; place at grid offsets; relax / pick active against frozen neighbors.

---

## Kernel design: tiled environment loop (local-memory reuse)

### Core idea (shared by both tiling strategies)

Replace “load entire static partner into `__local` once” with an **outer loop over environment tiles**. Each tile is cooperatively loaded into `__local`, then every owned atom of the active body interacts with that tile; then the next tile is loaded. Local footprint stays **O(tile size)**, not O(total env sites).

```
for each MD step:
  build rotation R, position pos for active body
  clear per-atom forces
  for each env tile T:                    // see Strategy M vs C below
      cooperative load T into __local L_pos[], L_REQ[], L_g[]
      barrier
      for each owned active atom i:
          for j in 0 .. |T|-1:
              unified_pair_force(i, L[j])
  reduce force/torque; integrate active body
```

**Constraint:** each env molecule (atoms + epairs + sigma holes) must fit in one tile if using Strategy M — i.e. `n_sites_mol ≤ MAX_SITES_PER_BODY` (today 128). Typical HCOOH/uracil with dummies ≪ 128, so this holds for intended use cases.

**Environment sites** = all atoms of all **other** bodies in **world frame** (host pre-transforms or kernel reads `poss/qrots` per env body — see “Env geometry source”).

---

### Two tiling strategies (undecided — pick later by profiling)

USER’s original idea was **Strategy M** (one molecule at a time). Strategy C (fixed site chunks) may be more bandwidth-efficient but crosses molecule boundaries. **Do not decide yet** — both stay group-local; compare below.

#### Strategy M — one molecule (`mol_j`) per tile

Outer loop iterates over **environment body indices** `j ≠ active`:

```
for j in env_bodies:
    n_j = mols[j+1] - mols[j]
    cooperative load sites of body j into __local (size n_j ≤ 128)
    barrier
    interact all active atoms × n_j sites
```

| Pros | Cons |
|------|------|
| Natural SoC: molecule = rigid body = tile | Variable tile length → last threads idle when `n_j` small / uneven |
| Self-skip trivial: never load `active` body | More barriers if many small molecules (e.g. 20× HCOOH → 20 tiles) |
| Easy to skip far bodies with CoM/AABB cut before load | Underfills local mem when `n_j ≪ MAX` (wasted capacity, not wrong) |
| Matches host data layout (`mols[]` offsets) | Transform-on-load (Variant B) is natural per body |
| No cross-molecule bookkeeping in the loader | |

**Efficiency note:** Barrier + load cost is **per molecule**, not per site. For many tiny molecules this can dominate compute. For few large molecules (PTCDA-scale) Strategy M is nearly optimal.

#### Strategy C — fixed-size site chunks (`MAX_ENV_CHUNK`)

Outer loop iterates over a **flat concatenated env site list** in fixed windows of size `C` (e.g. 64–128), independent of molecule boundaries:

```
for chunk in 0 .. ceil(n_env_sites / C) - 1:
    cooperative load env_sites[chunk*C : (chunk+1)*C] into __local
    barrier
    interact all active atoms × (up to C) sites
```

| Pros | Cons |
|------|------|
| Steady local fill → better amortize barrier vs compute | Chunk may **span molecule boundaries** (end of mol_j + start of mol_{j+1}) |
| Fewer tiles when many small molecules (`n_tiles ≈ n_env/C`) | Need flat concat buffer + offsets, or careful loader across `mols[]` |
| Same pattern as `assembly.cl` / `nonbonded_grid.cl` atom tiles | Self-interaction skip needs site→body id if active sites ever appear in flat list (avoid by excluding active from env) |
| Full use of chosen `__local` capacity | Slightly more complex host/kernel indexing |

**Boundary complexity (the main objection to C):**  
If the loader walks a flat array, a chunk can contain sites from two molecules. That is **fine for PairFF physics** (pair loop is site–site; no per-molecule state inside the force). Complexity appears only in:

1. Building/maintaining the flat `env_*` array (or a streaming loader that crosses `mols[j]` boundaries).
2. Optional far-field culls that want **per-molecule** AABB/CoM — harder if a chunk mixes two molecules (cull before concatenating, or cull per site).

So molecule-boundary spanning is a **bookkeeping** issue, not a physics issue — but it does make “skip whole far molecule before loading” less clean than Strategy M.

#### Hybrid (optional later)

- Cull / decide at **molecule** granularity (Strategy M outer list).
- Inside large molecules only, sub-chunk if somehow `n_j > MAX` (should not happen with current 128 limit).
- Or: Strategy C with env list sorted by body, and optional early exit when next site’s body is far — still undecided.

#### Efficiency intuition (qualitative)

| Scene | Likely better |
|-------|----------------|
| Few env molecules, each ~20–80 sites | **M** — one load/barrier per neighbor; simple |
| Many small env molecules (tens of HCOOH) | **C** — fewer barriers; fuller tiles |
| Mixed sizes + CoM culling important | **M** or hybrid — cull before load |
| Uniform replicas, flat buffer already built | **C** — simplest GPU loop |

**Decision deferred** until a multi-body demo exists and we can time M vs C on NVIDIA (same physics, swap outer loop).

---

### Env geometry source (orthogonal to M vs C)

| Variant | Env geometry source | Pros | Cons |
|---------|---------------------|------|------|
| **A. Host world cache** | Global `env_apos_world[]`, `env_REQ[]` rebuilt each frame | Simple kernel; easy picking | Host transform cost; large buffer |
| **B. Kernel on-the-fly** | Global `body_pos`, `body_quat`, `apos_body` per env body; transform in tile loader | Fresher poses if multi-active later | More registers; loader complexity |

**Recommendation for v1:** **Variant A** — matches current `static_*` upload. Works with both M (slice by `mols[]`) and C (flat concat).

**Recommendation for v2:** **Variant B** — especially natural with Strategy M (load one body’s body-frame atoms + rotate by that body’s quat).

---

## Local memory budget (RTX 3090 class)

Device local mem per workgroup: **48 KiB** typical (see OpenCL device query in demos).

### Current unified pairff local usage (approx.)

| Array | Size (bytes) |
|-------|----------------|
| `pos, qrot, vpos, vrot` | 4×16 = 64 |
| `inv_mass` | 4 |
| `R, Iinv_body, Ibody` (cl_Mat3) | 3×48 ≈ 144 |
| `Ltorq[32], Lforce[32]` | 2×32×16 = 1024 |
| `Lstatic_pos[128]` | 2048 |
| `Lstatic_REQ[128]` | 2048 |
| `Lstatic_g[128]` | 512 |
| Barriers / compiler temps | ~1–4 KiB |
| **Total** | **~6–8 KiB** |

**Headroom:** ~40 KiB free — enough for one full molecule tile (≤128 sites) or one `MAX_ENV_CHUNK` tile; **not** enough for an entire multi-molecule scene.

### Tile / chunk size

| Strategy | Tile size | Local env bytes (pos+REQ+g ≈ 36 B/site) |
|----------|-----------|----------------------------------------|
| **M** | `n_j` (variable, ≤128) | up to ~4.6 KiB per molecule |
| **C** | fixed `C` ∈ {64, 96, 128} | ~2.3–4.6 KiB constant |

Outer loop cost: **O(n_env_sites)** pair evaluations + **O(n_tiles)** barriers/loads.  
Strategy M: `n_tiles = n_env_molecules`. Strategy C: `n_tiles = ceil(n_env_sites / C)`.

### Occupancy note

- 1 workgroup = 1 body, `WORKGROUP_SIZE = 32` (fixed in `rigid.cl`).
- `n_bodies` workgroups launch in parallel — **bodies are embarrassingly parallel** across SMs.
- Only **one** body integrates per user action → other workgroups can **early-exit** after force eval or skip integration via `active_body` uniform (see below).

---

## Active vs fixed bodies in one launch

### Option 1 — Single kernel, guard integration (recommended)

All `n_bodies` workgroups compute forces on their own atoms from environment (for debugging / future multi-active). **Only** `gid == active_body`:

- updates `pos`, `quat`, `vposs`, `vrots`, `fire_state`
- applies FIRE

Fixed bodies: force evaluation optional (waste) OR skip force eval entirely on non-active groups.

**Optimization:** Non-active workgroups return immediately after barrier (no integration). They do **not** need force eval unless we want forces on fixed bodies for display — usually **skip** for speed.

```opencl
if (gid != active_body) return;  // at start, after loading own pose for env contribution elsewhere
```

Wait — env forces on active body need **other bodies' atoms**, not other workgroups' force eval. Non-active WGs don't need to run at all if env is in flat buffer.

**Simpler launch:** Only **1 workgroup** (active body) runs pairff MD kernel; env buffer contains all other molecules. Other bodies updated only on host when user switches active index.

This matches demo physics exactly and is **Phase 1**.

### Option 2 — All workgroups run (future)

Needed if multiple bodies relax simultaneously. Out of scope for first demo.

**Phased plan:**

| Phase | Launch | Env representation |
|-------|--------|-------------------|
| **1** | 1 WG, `active_body` | Flat or per-body `env_*` ≤128 total (extend current 1+1 demo to few small neighbors without tiling) |
| **2** | 1 WG | **Tiled env loop** — Strategy **M** (per `mol_j`) and/or Strategy **C** (`MAX_ENV_CHUNK`); decide after profiling |
| **3** | N WG | Tiled + per-body poses in global memory (multi-active optional) |

---

## Python API changes (outline)

### New / extended types

```python
class RigidBodyPairFF(RigidBodyDynamics):
    def alloc_pairff_env(self, max_env_sites: int): ...
    def upload_env(self, apos_world, REQ, types, body_id_per_site=None): ...
    def set_active_body(self, body_id: int): ...
    def init_pairff(..., mode='unified', ...): ...  # existing

    @classmethod
    def from_molecules(cls, molecules, poses, active_body=0, ...): ...
```

### `realloc` generalization (required for variable sizes)

Current: `num_atoms` uniform per body.

Target:

- `num_atoms_per_body[i]` arbitrary ≤ 128
- `mols[i]` cumulative offsets in `apos_body` (kernel already uses `mols[gid]`, `na = mols[gid+1]-mols[gid]`)
- `total_atoms = sum(num_atoms_per_body)`
- `anchors` length = `total_atoms`

**Risk:** Touch `realloc()` carefully — folded / gridff paths use same buffers (`doc/Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md`). Add `realloc_variable()` or optional `atom_counts[]` without breaking single-body callers.

### Building environment for active body `a`

```python
env_apos, env_REQ, env_types = [], [], []
for j, mol in enumerate(molecules):
    if j == a: continue
    world = transform_body_to_world(poses[j], mol.apos_body)
    env_apos.append(world)
    env_REQ.append(mol.REQ_ext)
    env_types.append(mol.types)
# concat, upload, set n_env_sites
```

---

## Performance considerations

| Topic | Note |
|-------|------|
| **Tile loop overhead** | Extra barrier + cooperative load **per tile**; amortize with `niter` MD steps per launch |
| **M vs C barrier count** | M: `n_tiles = n_env_mols`; C: `n_tiles = ceil(n_env_sites/C)` — many small mols favor C |
| **Molecule-boundary span (C)** | Physics OK (site–site); cost is indexing + harder per-mol CoM cull before load |
| **Underfilled tiles (M)** | Small `n_j` wastes local capacity and leaves idle lanes in the load — not incorrect |
| **Cutoff** | Unified kernel already has `r2 <= rc2` on compact channel; far sites skipped cheaply after load |
| **Coulomb** | Only real–real (`gij==1`); dummy–dummy skipped via `both_dummy` — keep |
| **Host upload** | Rebuild env when any **non-active** body moves; active-only relax → update once per step if only active moves |
| **vs brute CPU** | GPU wins when `n_env_sites` > ~50 and many MD steps; profile with 5–20 molecules |
| **Warp divergence** | Unified kernel avoids per-pair type branching — keep single loop in tile inner |
| **Register pressure** | `compact_exp_pair_EF` + tile loader — profile M vs C and `C` ∈ {64,128} |

**Reference patterns in repo:**

- `kernels/assembly.cl` — tiled neighbor loops in local memory (closer to Strategy C)
- `kernels/nonbonded_grid.cl` — atom tiles + barriers
- `rigid_body_folded_replicas_kernel` — 1 thread = 1 replica (different parallelism model)
- Current `rigid_body_pairff_*` — whole partner molecule in `__local` (degenerate case of Strategy M with `n_env_mols = 1`)

---

## Kernel naming (proposed)

| Kernel | Purpose |
|--------|---------|
| `rigid_body_pairff_unified_kernel` | Keep for 1 static list ≤128 (backward compat / tests) |
| `rigid_body_pairff_unified_env_kernel` | Active body + tiled env (M and/or C; compile-time or runtime switch TBD) |
| (later) `rigid_body_pairff_unified_multibody_kernel` | N bodies, tiled cross-body |

Avoid breaking `demos/demo_pairff.py` legacy path — add new kernel entry points.

---

## Demo milestones (before GUI)

1. **Multi-molecule scene** — 3–5 copies of HCOOH / uracil at XY offsets; headless FIRE.
2. **`--active k`** — only body `k` moves; verify energy decreases vs frozen neighbors.
3. **Switch active** — relax body 0, then body 1 against updated poses (manual CLI).
4. **Tiling A/B** — implement Strategy M first (matches mental model); optional Strategy C for same scene; force parity identical.
5. **Stress + profile** — 10× ~20 sites; NVIDIA-only ms/step vs `n_env_mols` / `n_env_sites` for M vs C.

---

## Verification

| Check | Method |
|-------|--------|
| Force parity | CPU double loop unified formula vs GPU for 1 active + 2 env molecules |
| M ≡ C | Same scene, both tiling strategies → max \|ΔF\|, \|ΔE\| within float32 noise |
| Momentum | Active body only integrates; fixed bodies unchanged |
| Regression | `test_folded_relax.py` still passes (unchanged folded kernels) |
| Existing demo | `demo_pairff.py` 1+1 case unchanged with `--pairff-mode unified` |

---

## Open questions for USER

1. **Tiling Strategy M vs C** — prefer implement M first (one `mol_j` per tile), keep C as alternate, or implement both and pick by bench? *(default lean: M first, C optional)*
2. **Variable atoms per body** in v1, or require same topology (replicas) for simpler `realloc`?
3. **Maximum assembly size** target (molecules count, sites count) for tile sizing / culling?
4. **Legacy kernel** — keep for multi-body or unified-only in new kernel?
5. When switching active molecule, **rebuild epairs** from graph or keep cached GPU bodies?

---

## Acceptance (do not mark Done without USER confirmation)

- [ ] Tiling strategy (M / C / both) reviewed; USER confirms which to ship as default
- [ ] Demo: N molecules, 1 active, headless FIRE converges on test scene
- [ ] Local memory / tile size documented with measured occupancy if possible
- [ ] CPU parity test for multi-body configuration; M≡C if both exist
- [ ] No regression on folded rigid / gridff kernels

## Out of scope

- Intramolecular SPFF/UFF relax inside each rigid body
- Folded-basis substrate + PairFF in one kernel (compose at Python level first)
- Full GUI picking (see `PairFF_GUI_Integration.md`)