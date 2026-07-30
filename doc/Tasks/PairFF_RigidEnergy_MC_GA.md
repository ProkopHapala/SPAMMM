---
type: Task
title: PairFF rigid energy kernel — MC / GA / population harness
status: K1+P1+H1+F implemented; H2 (SA) and H3 (GA) pending
tags: [PairFF, rigid-body, OpenCL, MonteCarlo, GeneticAlgorithm, FAF, performance]
timestamp: 2026-07-30
---

# PairFF rigid energy kernel — algorithm-agnostic replica evaluator

**Status:** K1 kernel + P1 host API + H1 greedy harness + F real FAF fits implemented and verified on NVIDIA RTX 3090 across 4 molecules (PTCDA, formic acid, terephthalic acid, NTCDI). Numerical parity confirmed (err < 1e-7). H2 (simulated annealing) and H3 (GA population) remain pending — greedy MC stalls in deep minima (21–75/1000 accepted), motivating SA.  
**Goal:** One GPU kernel that evaluates **PairFF + FAF (+ optional Kz/anchors) energy** for rigid molecular assemblies across many **replicas**, for an arbitrary **active molecule set**. Host harness chooses greedy / Metropolis / SA / genetic algorithm — **kernel never knows the optimizer**.

**Related:** [`PairFF_MultiBody_Kernel.md`](PairFF_MultiBody_Kernel.md), [`PairFF_FAF_Substrate.md`](PairFF_FAF_Substrate.md), [`TopicalAudit/PairFF_RigidBody.md`](../TopicalAudit/PairFF_RigidBody.md), [`TopicalAudit/RigidBody.md`](../TopicalAudit/RigidBody.md) (pose inventory), [`RigidMoleculePose_SSOT.md`](RigidMoleculePose_SSOT.md) (host pose authority for replicas when implemented), `kernels/rigid.cl` (allmol FAF MD kernels 12–13; replica energy kernel 14), `spammm/forcefields/RigidBodyDynamics.py` (`RigidBodyPairFF`).

**Not this task:** `AssemblyOCL` / `assembly.cl` (clash packing only). Do not mix clash scores with PairFF energy.

---

## 1. Problem statement

We want to optimize poses of \(N_\mathrm{mol}\) rigid adsorbates on a surface (NTCDI / PTCDA / …):

\[
E = E_\mathrm{PairFF}(\{\mathbf{r}_i,\mathbf{q}_i\}) + E_\mathrm{FAF} + E_\mathrm{Kz/anchors}.
\]

Optimizers (MC annealing, greedy batch, GA population) differ only in **how poses are proposed and accepted**. The GPU must only answer:

> Given \(R\) replica pose sets and a list of \(N_\mathrm{active}\) molecule indices, return per-(replica, active) energy channels so the host can form \(E\) or \(\Delta E\).

Constraints from product use:

| Requirement | Implication |
|-------------|-------------|
| New kernel, energy-only | No forces, torques, MD, FIRE, velocities |
| Algorithm-agnostic | No Metropolis / temperature / mutation inside `.cl` |
| Local *and* global moves | `nactive ∈ [1, nmol]`; `nactive=1` is a special case |
| Population of systems | Buffer layout = `n_replica × nmol` poses from day one |
| Planar harness first | Trials may fix \(z=0\), in-plane \(\phi\); kernel accepts full SE(3) pose |
| FAF in kernel from start | Flat/zero coeffs ⇒ vacuum; real fit later without kernel change |
| NVIDIA local memory | Budget LM carefully; avoid register spill |

---

## 2. Separation of concerns (SSOT)

```
┌─────────────────────────────────────────────────────────────┐
│  Harness (Python)                                           │
│  - propose trials / crossover / mutate                      │
│  - fill pose buffers for R replicas                         │
│  - fill active_mols[0..nactive)                             │
│  - launch energy kernel                                     │
│  - reduce channels → E or ΔE                                │
│  - accept/reject / replace population (greedy, Metropolis…) │
└──────────────────────────▲──────────────────────────────────┘
                           │ E channels only
┌──────────────────────────┴──────────────────────────────────┐
│  Kernel (OpenCL)                                            │
│  - 1 workgroup = 1 (replica, active_slot)                   │
│  - gather poses, tile PairFF, add FAF/Kz/anchors            │
│  - write float4 energy channels (no optimizer logic)        │
└─────────────────────────────────────────────────────────────┘
```

**Kernel name:** `rigid_body_pairff_energy_replica_kernel`  
**Location:** `kernels/rigid.cl` (reuse `compact_exp_pair_EF` energy half, `folded_eval_basis_rigid` — **not** `folded_eval_grad_rigid`).

---

## 3. Indexing model

### 3.1 Dimensions

| Symbol | Meaning |
|--------|---------|
| `nmol` | Molecules per replica (system size) |
| `n_replica` | Independent pose sets (MC trials **or** GA population **or** both) |
| `nactive` | Length of active list this launch (`1 … nmol`) |
| `active_mols[nactive]` | Molecule indices whose energy channels are requested |

### 3.2 Workgroup map

Preliminary 1D map (not selected for K1):

```
gid   = get_group_id(0)          // 0 .. n_replica * nactive - 1
irepl = gid / nactive
islot = gid % nactive
ia    = active_mols[islot]       // molecule index in [0, nmol)
```

Launch: `global_size = (n_replica * nactive * WG_SIZE,)`, `local_size = (WG_SIZE,)`.  
Saturation: e.g. `n_replica=1024`, `nactive=1` → 1024 groups; `n_replica=64`, `nactive=nmol=8` → 512 groups. Host batches if needed.

**Alternative (optional later):** 2D NDRange `(nactive, n_replica)` — same math, clearer profiling.

K1 selects the 2D map to remove integer division/modulo from every work-item:

```
islot = get_group_id(0)
irepl = get_group_id(1)
ia    = active_mols[islot]
```

Launch: `global_size=(nactive*64, n_replica)`, `local_size=(64, 1)`.

### 3.3 Pose SoA (persistent, algorithm-agnostic)

```
poss [n_replica * nmol] : float4   // xyz = CoM, w unused or mass
qrots[n_replica * nmol] : float4   // body→world quaternion
```

Index molecule \(j\) in replica \(r\):

```
idx = r * nmol + j
```

Body-frame chemistry is **shared across replicas** (v1 assumption: same packs / same `mols[]`):

```
mols[nmol+1]           // site offsets
apos_body[nsites]      // float4 body-frame sites
dyn_REQ[nsites]        // float4
dyn_type[nsites]       // int  (0=real, 1=epair, 2=σ-hole)
folded_*               // FAF tables (shared)
folded_atom_type[]     // per site (or per real atom; same as allmol)
anchors[nsites]        // optional
```

If a future GA needs different species per replica → new pack buffers; **out of scope for v1**.

### 3.4 Active list

```
__global const int* active_mols;  // length nactive
const int nactive;
const int nmol;
const int n_replica;              // or derived: get_num_groups(0)/nactive
```

Optional host-side bitmask `mol_is_active[nmol]` uploaded once per launch (or rebuilt in `__local` from `active_mols`) so the kernel can classify partner \(j\) as **active-partner** vs **frozen** without \(O(nactive)\) scans on every \(j\) when `nactive` is large. For `nactive ≲ 16`, scanning the small list is fine.

---

## 4. Energy channels (correctness SSOT)

### 4.1 Why a single scalar `E_mol` is not enough

For molecule \(i\), a naive

\[
E_i = \sum_{j\neq i}\mathrm{Pair}(i,j) + \mathrm{One}(i)
\]

satisfies

\[
\sum_i E_i = 2\,E_\mathrm{PairFF} + E_\mathrm{One}.
\]

For a **partial** move of set \(A\) (frozen complement \(F\)):

\[
\sum_{i\in A} E_i = 2\,E_{AA} + E_{AF} + E_{\mathrm{One},A},
\]

while the physically changed energy is

\[
E_\Delta = E_{AA} + E_{AF} + E_{\mathrm{One},A}.
\]

So \(\sum_{i\in A} E_i\) **overcounts** \(E_{AA}\). A single float cannot support both `nactive=1` and `nactive>1` without host knowledge of \(E_{AA}\).

### 4.2 Per-workgroup output: `float4` channels

Each workgroup for \((r, i)\) with \(i\in A\) writes:

| Component | Content |
|-----------|---------|
| `E.x` | \(E_{i,A\setminus\{i\}}\) — PairFF of \(i\) vs **other active** molecules in same replica |
| `E.y` | \(E_{i,F}\) — PairFF of \(i\) vs **frozen** molecules |
| `E.z` | \(E_{\mathrm{one},i}\) — FAF + Kz + anchors (one-body) |
| `E.w` | reserved (`0`) or debug (`E.x+E.y+E.z`) |

Buffer:

```
E_out[n_replica * nactive] : float4
```

### 4.3 Host reductions (harness utilities, not kernel)

**Changed energy of active set** (exact for any \(A\)):

```python
# E_out shape (n_replica, nactive, 4)
Ex, Ey, Ez = E_out[...,0], E_out[...,1], E_out[...,2]
E_changed = 0.5 * Ex.sum(axis=-1) + Ey.sum(axis=-1) + Ez.sum(axis=-1)
```

Special cases:

| Move | \(A\) | Formula simplifies to |
|------|-------|------------------------|
| Single-molecule | \(nactive=1\) | `E_changed = Ey + Ez` (`Ex≡0`) |
| Global / all-mobile | \(nactive=nmol\) | `E_changed = E_tot = 0.5*Ex.sum + Ez.sum` (`Ey≡0`) |
| Multi-molecule subset | \(1 < nactive < nmol\) | full formula above |

**ΔE between two pose buffers** (old vs new), same `active_mols`:

```python
dE = E_changed(new) - E_changed(old)
```

Frozen–frozen pairs cancel and are never computed.

**Full system energy** when needed (logging / GA fitness): set `active_mols = range(nmol)`, `nactive=nmol`, use global formula.

This is the contract that keeps MC annealing, greedy-best-of-batch, and GA interchangeable at harness level.

---

## 5. Physics inside the kernel (energy-only)

Reuse unified PairFF + FAF from allmol kernels; **delete** force/torque/MD paths.

### 5.1 PairFF (per owned site of molecule `ia`)

Same compact-exp + Coulomb as `rigid_body_pairff_unified_allmol_faf_kernel`, but accumulate **energy only**:

- Load partner molecule \(j\neq ia\) tile into `__local` (world positions + REQ + `g` flag).
- Classify \(j\) once: active vs frozen → add pair energy into `E.x` or `E.y`.
- Skip self (`j == ia`).
- Dummy rules (`gij`, `both_dummy`, `attr` vs `REQ.y`) unchanged vs MD kernel (parity requirement).

### 5.2 One-body

For each **real** site (`folded_atom_type >= 0`):

- FAF: \(\sum_b c_{t,b}\, B_b(u,v,z)\) via `folded_eval_basis_rigid` only (**no gradient**).
- Optional `k_z` harmonic vs `z_target` (harness may set `k_z=0` for planar \(z=0\) tests).
- Optional anchors (`anchor.w > 0`).

Dummy sites: PairFF only; no FAF (same as current allmol FAF path).

### 5.3 Flat / disabled FAF

Initial tests: `nbasis=0` **or** all coeffs `0` **or** single constant basis → homogeneous substrate. Kernel always has the FAF loop; empty/zero is free branching if `nbasis==0` early-out after loading meta.

---

## 6. Kernel algorithm (pseudocode)

```
__kernel void rigid_body_pairff_energy_replica_kernel(
    mols, poss, qrots,
    apos_body, dyn_REQ, dyn_type,
    active_mols, nactive, nmol, n_replica,
    pairff_params, beta, z_target,
    folded_coeffs, folded_kxyz, folded_atom_type, folded_meta, folded_lvec2d,
    anchors,
    E_out   // float4[n_replica * nactive]
){
    lid = get_local_id(0);  lsize = get_local_size(0);
    islot = get_group_id(0);
    irepl = get_group_id(1);
    ia    = active_mols[islot];

    // --- pose of active molecule (local) ---
    if (lid==0) { pos = poss[irepl*nmol+ia]; qrot = normalize(qrots[...]); build R; }
    barrier;

    // --- optional: load active flags to local ---
    // Lactive[j] = 1 if j in active_mols

    // --- FAF tables once per WG (or skip if nbasis==0) ---
    coop-load LBASIS, LCOEFFS; barrier;

    // --- own sites: world pos + REQ in registers (energy only) ---
    for own atoms owned by lid:
        p_world, REQ, g, ityp  // no f_acc, no r_store

    float Ex=0, Ey=0, Ez=0;

    // --- PairFF tiles over j ≠ ia ---
    for j in 0..nmol-1:
        if (j==ia) continue;
        coop-load tile of mol j from poss/qrots[irepl*nmol+j] → Lenv_*;
        barrier;
        for own atoms:
            for t in tile:
                Epair = compact_exp + coulomb;  // energy scalar only
                if (Lactive[j]) Ex += Epair; else Ey += Epair;
        barrier;

    // --- One-body (scoped block; releases pair temporaries) ---
    {
        for own atoms with ityp>=0:
            Ez += FAF(u,v,z) + Kz + anchor;
    }

    // --- reduce Ex,Ey,Ez across WG ---
    LEx[lid]=Ex; LEy[lid]=Ey; LEz[lid]=Ez;
    tree_reduce;
    if (lid==0) E_out[gid] = (float4)(LEx[0], LEy[0], LEz[0], 0);
}
```

No writes to `poss` / `qrots` / velocities. Pure function of inputs.

---

## 7. Performance design (NVIDIA)

### 7.1 Lessons from current allmol FAF MD kernel

Current `*_allmol_faf_kernel` LM footprint (~11–12 KB/WG) is acceptable but **register-heavy**:

- `f_acc[ATOMS_PER_THREAD]` as `float3`
- `r_store`, torque path, FIRE/dt state
- `folded_eval_grad_rigid` (sins + chain rule) live with PairFF temps

Energy-only kernel must cut that pressure so many replica WGs hide latency.

### 7.2 Local memory budget (compile-time caps)

**Implemented K1 (supersedes the preliminary estimate below):** one 64-site scratch tile is reused in two phases. During PairFF it stores `(world_xyz,g)` plus REQ; during FAF it stores 32 basis records plus up to `8×32` type-major coefficients. The same allocation therefore serves both computations instead of making environment and full FAF tables resident together.

| K1 buffer | Bytes |
|-----------|------:|
| `Lsite[64]` | 1024 |
| `Lparam[256]` | 1024 |
| three-channel reduction, WG=64 | 768 |
| pose/rotation/indices + alignment | 96 |
| **NVIDIA compiler total** | **2912** |

On the RTX 3090, `2912×16 = 46592 < 49152` bytes, so local memory permits 16 resident two-warp workgroups. Partner molecules and active molecules larger than 64 sites stream in chunks; there is no silent 128-site truncation.

Preliminary full-residency estimate (not selected):

Keep existing caps unless profiling says otherwise:

| Buffer | Size | Bytes |
|--------|------|------:|
| `Lenv_pos[TILE]` | `TILE×float4` | \(16\,\mathrm{TILE}\) |
| `Lenv_REQ[TILE]` | `TILE×float4` | \(16\,\mathrm{TILE}\) |
| `Lenv_g[TILE]` | `TILE×float` | \(4\,\mathrm{TILE}\) |
| `LE[3][WG]` or 3×`float[WG]` | reduce channels | \(3\times4\times32 = 384\) |
| `cl_Mat3 R, Rj` | 2× | \(\sim 96\) |
| `LBASIS[nbasis≤128]` | float4 | \(\le 2048\) |
| `LCOEFFS[ntypes×nbasis≤8×128]` | float | \(\le 4096\) |
| `Lactive[nmol]` optional | uchar/int | \(\sim 4\,\mathrm{nmol}\) |

With `TILE = MAX_STATIC_ATOMS = 128` (one partner molecule ≤ 128 sites, already the allmol invariant):

\[
\mathrm{LM} \approx 4.6\,\mathrm{KB\,(env)} + 6\,\mathrm{KB\,(FAF)} + 0.5\,\mathrm{KB} \approx \mathbf{11\,\mathrm{KB}}
\]

NVIDIA ~48 KB LM/SM ⇒ **~4 WGs/SM** LM-limited if FAF fully resident. Acceptable for v1 (matches today’s FAF MD kernels).

**If occupancy becomes the bottleneck** (profile with `n_replica` large, FAF on):

1. **Stream FAF basis in tiles** (e.g. 32 basis funcs at a time) — drop `LBASIS`/`LCOEFFS` peak by ~4×; extra barriers, more global reads.
2. **Shrink env TILE** only if partner molecules can exceed a smaller cap — today one mol ≤ 128 sites, so TILE=128 is one mol = one tile (good: one barrier per partner mol).
3. Do **not** put all frozen atoms in LM at once (Strategy C from multi-body doc) in v1 — Strategy M (per-molecule tiles) matches mental model and LM.

**Preliminary recommendation not selected:** TILE=128 with full FAF residency costs too much occupancy on the 48 KiB RTX 3090 SM. K1 uses `SITE_TILE=64`, `FAF_TILE=32`, and scratch reuse.

### 7.3 Register pressure (highest priority)

| Do | Don’t |
|----|--------|
| Accumulate `float Ex,Ey,Ez` only | `float3 f_acc[]`, torque, `r_store` |
| Call only `folded_eval_basis_rigid` | Call `folded_eval_grad_rigid` |
| Scope PairFF block vs FAF block `{ }` so temps die | Keep PairFF and FAF locals live together |
| One owned site/thread with WG=64; stream larger bodies | Two owned sites/thread: same 64 registers but small spills on RTX 3090 |
| `float` energy from `compact_exp_pair_EF` (use `.x` only) | Keep force `.y` live across loops |
| Coalesce `poss`/`qrots` as `float4` SoA | AoS structs |

NVIDIA `ptxas` for K1 reports **64 registers, 2912 bytes shared memory, zero stack and zero spills**. Runtime throughput and numerical parity still require the deferred harness.

### 7.4 Memory traffic

- **Amortize:** each partner tile loaded once per WG, reused by all owned atoms → factor ~`na_owned`.
- **Replica poses:** threads cooperatively read one `float4` pose for `j`; avoid every thread hitting the same address without broadcast (lid==0 load + local is fine for one pose).
- **No atom world buffer required** for energy path (optional debug write disabled by default) — saves bandwidth.
- **Host transfers:** upload only changed poses between MC steps; keep `apos_body` / FAF / REQ persistent. Download only `E_out` (`n_replica * nactive * 16` bytes) — tiny.

### 7.5 Launch overhead

PyOpenCL launch cost is large vs a small kernel. Prefer:

- One launch with large `n_replica` (hundreds–thousands of trials/population members)
- Not one launch per trial

Harness batches proposals → one energy eval → vectorized accept on NumPy.

### 7.6 Workgroup size

K1 requires **`WORKGROUP_SIZE = 64`**. The RTX 3090 can host at most 16 blocks/SM, so a 32-thread group caps residency at 16 warps; 64 threads provide 32 resident warps at the measured 64-register/2912-byte footprint. A 128-thread group gives more idle lanes for the 26–40-real-atom target molecules and was not selected without a runtime benchmark.

---

## 8. Harness algorithms (kernel unchanged)

All use the same kernel + `E_changed` helper.

### 8.1 Greedy best-of-batch (start here)

1. Current replica `r=0` is the champion; `E0 = E_changed` with `A` = moved set (often 1 mol).
2. Fill `poss/qrots` for `r=1..R-1` with proposed poses (only active molecules differ; frozen copied).
3. Launch kernel; pick `r* = argmin E_changed[r]`.
4. If `E_changed[r*] < E0` (or always take best), copy pose into champion.

### 8.2 Metropolis / simulated annealing (one-liner change)

Same batch; accept `r*` (or first improving / random candidate) with

\[
P = \min\bigl(1, e^{-(E'-E)/T}\bigr).
\]

Cool \(T\) on host schedule.

### 8.3 Genetic algorithm / population

- `n_replica = population_size`
- Each replica is a full genotype (all `nmol` poses)
- Fitness = `E_tot` via `nactive=nmol`
- Selection / crossover / mutation on host; kernel only scores

Trials-from-one-champion and true populations are the **same buffer layout**; only who writes poses differs.

### 8.4 Move classes (harness only)

| Move | `nactive` | Pose edits |
|------|-----------|------------|
| Local translate/rotate one mol | 1 | that mol’s `poss/qrot` |
| Coupled dimer move | 2 | two molecules |
| Global rigid shift of island | \(k\) | \(k\) molecules |
| Full reshuffle / GA mutate all | `nmol` | all |

Kernel always: evaluate listed actives vs current replica poses (which already contain the trial positions of all moved molecules).

**Planar v1 harness:** \(\Delta x,\Delta y,\Delta\phi_z\); fix \(z\) and tilt in pose buffers. Kernel still reads full quaternion + \(z\) (future 6-DOF free).

---

## 9. API sketch (Python, existing module)

Extend `RigidBodyPairFF` (no new file unless USER asks):

```python
def ensure_replica_buffers(self, n_replica): ...
def upload_replica_poses(self, poss, qrots):     # (R, nmol, 4)
def set_active_mols(self, active_idx):           # 1d int array
def eval_energy_replicas(self) -> np.ndarray:    # (R, nactive, 4) float32
@staticmethod
def energy_changed(E_chan) -> np.ndarray:        # (R,) from §4.3
```

Demo path later: NTCDI/PTCDA grid, greedy planar MC, optional FAF fit.

---

## 10. Parity & test plan

| Level | Check |
|-------|--------|
| L0 | Two-mol dimer: `nactive=1` `Ey+Ez` equals half of full `E_tot` pair share + onebody of that mol |
| L0 | `nactive=nmol` → `0.5*Ex.sum+Ez.sum` matches CPU reference PairFF+FAF |
| L0 | Move one mol: `ΔE` from channels equals `E_tot_new - E_tot_old` |
| L0 | Move two mols: same ΔE identity (guards half-counting) |
| L0 | `nbasis=0` / zero coeffs ⇒ FAF channel 0 |
| L1 | NVIDIA timing: energy kernel vs allmol MD one-step; spill/occupancy notes in `.out` |
| L2 | Greedy planar NTCDI/PTCDA assembly snapshot |

Reference: reuse compact-exp CPU path from HBondFF / existing PairFF map helpers where possible.

---

## 11. Implementation phases

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **D** (this doc) | Design locked | ✅ done |
| **K1** | Energy kernel + channel reduce; FAF optional via `nbasis` | ✅ done — `kernels/rigid.cl` kernel 14, WG=64, 2912 B LM, 64 regs, 0 spills |
| **P1** | `eval_energy_replicas` + `energy_changed` on `RigidBodyPairFF` | ✅ done — `RigidBodyDynamics.py` L2428–2648 |
| **H1** | Greedy planar harness, `nactive=1`, no FAF | ✅ done — `greedy_energy_step` + `tests/testplot_pairff_energy_mc.py` |
| **H2** | Metropolis/SA one-liner; multi-`nactive` moves | ⏳ pending — greedy MC stalls (21–75/1000 acc); SA is next |
| **H3** | Population / GA fitness path (`nactive=nmol`) | ⏳ pending |
| **F** | Real FAF fit (NTCDI/PTCDA@NaCl) | ✅ done — 4 molecules, remapped fits, FAF+PairFF verified |

Do **not** implement H2/H3 until USER confirms priorities.

---

## 12. Explicit non-goals (v1)

- Forces / FIRE / MD inside energy kernel
- Mixing `AssemblyOCL` clash score into `E`
- Per-replica different molecular topologies
- Strategy C flat site chunks
- In-kernel RNG or Metropolis
- Writing world atom positions every eval (debug flag only)

---

## 13. Open questions for USER

1. **Replica capacity default:** hard cap (e.g. `n_replica ≤ 4096`) for buffer sizing, or fully dynamic `try_make_buffers`?
2. **Active flags:** rebuild `mol_is_active[nmol]` on host each launch vs kernel scans `active_mols`?
3. **Kz in planar tests:** keep channel with `k_z=0`, or omit Kz from kernel until needed? (Recommendation: keep, `k_z=0`.)
4. **NTCDI input:** mol2 loader now vs temporary XYZ export?

---

## 13. Results (H1+F, 2026-07-30)

**Driver:** `tests/testplot_pairff_energy_mc.py` → `debug/testplot_pairff_energy_mc/<mol>/`

**Single-species runs (1000 steps, 256 trials):**

| Molecule | nmol | E_initial [eV] | E_final [eV] | ΔE [eV] | Accepted | Parity err |
|----------|------|----------------|-------------|---------|----------|------------|
| PTCDA | 4 | 6.89 | 2.90 | -3.98 | 21/1000 | 9.9e-08 |
| Formic acid | 6 | 6.70 | -0.23 | -6.93 | 75/1000 | 5.8e-08 |
| Terephthalic acid | 4 | 3.44 | 0.31 | -3.13 | 40/1000 | 7.8e-08 |
| NTCDI | 4 | 4.95 | 0.58 | -4.37 | 29/1000 | 7.5e-09 |
| TBTAP | 4 | 9.42 | 1.95 | -7.48 | 18/1000 | 2.2e-07 |
| Azaindol | 4 | 3.37 | 0.77 | -2.61 | 38/1000 | 4.5e-08 |
| Uracil | 4 | 1.93 | -0.40 | -2.33 | 36/1000 | 1.1e-07 |
| Adenine | 4 | 2.30 | 0.44 | -1.87 | 32/1000 | 2.1e-08 |

**Multi-species runs (1000 steps, 256 trials):**

| Molecules | nmol/species | E_initial [eV] | E_final [eV] | ΔE [eV] | Accepted | Parity err |
|-----------|-------------|----------------|-------------|---------|----------|------------|
| adenine + uracil | 3 | 10.17 | 1.68 | -8.49 | 54/1000 | 8.9e-08 |

**Usage:**
```bash
# Single molecule
python3 tests/testplot_pairff_energy_mc.py --mol PTCDA --steps 1000
# Multi-species (comma-separated)
python3 tests/testplot_pairff_energy_mc.py --mol adenine,uracil --nmol 3 --spacing 10
```

**Artifacts per run** (`debug/testplot_pairff_energy_mc/<mol>/` or `.../<mol1+mol2>/`):
- `summary.out` — energies, charges, inter-mol distances, final positions
- `energy_history.png` — E vs MC step
- `assembly_before_after.png` — FAF substrate + bonds + charge-colored atoms (before/after)
- `trajectory.gif` — animated MC trajectory (FAF + bonds + charges, fixed axes)
- `before.xyz`, `after.xyz`, `traj.xyz` — structures

**FAF fits used (remapped by REQ similarity):**
- PTCDA → `ptcda_nacl.npz` (C, O, H); Formic acid → `hcooh_nacl.npz` (H, C, O, H)
- NTCDI, TBTAP, azaindol, uracil, adenine → `ptcdi_nacl.npz` (C, N, O, H — broadest coverage)
- Terephthalic acid → `ptcda_nacl.npz` (C, O, H)
- Multi-species → `ptcdi_nacl.npz` (default — covers all elements)

**Multi-species support:**
- `--mol adenine,uracil` creates `nmol` copies of each species (total = nmol × n_species)
- `_folded_types_all_sites` in `RigidBodyDynamics.py` slices FAF atom_type_ids per pack
- `assembly_real_atoms` accepts per-pack bonds list for correct skeleton rendering
- Molecules interleave on the grid for better mixing

**Key observation:** greedy MC stalls in deep minima (18–75/1000 accepted). Bigger steps (dxy=1.5 Å, dphi=0.8 rad) help but don't solve it. Simulated annealing (H2) is the proper fix — the kernel already supports it; only harness acceptance logic needs `P=exp(-ΔE/T)` with a cooling schedule.

---

## 14. Summary

- **One kernel**, energy-only, FAF-capable, replica-major poses, `active_mols[nactive]` with \(nactive\in[1,nmol]\).
- **float4 channels** \((E_{iA}, E_{iF}, E_{\mathrm{one}}, 0)\) make single-molecule, multi-molecule, and full-system energies exact without kernel changes.
- **Harness** owns greedy / Metropolis / SA / GA; kernel is a pure evaluator.
- **Performance:** drop forces/grads; tile partners in LM; keep WG=32; watch FAF LM (~6 KB) and register spill; batch many replicas per launch.
