---
type: Task
title: Concurrent multi-molecule relaxation on FAF substrate — fuse folded-basis substrate forces into the multimol kernel
status: spec — awaiting USER approval before implementation
tags: [OpenCL, MD, performance, rigid-body, PairFF, FAF, substrate, launch-overhead]
timestamp: 2026-08-01
related: [MultiMol_MD_LaunchOverhead.md, PairFF_FAF_Substrate.md, PairFF_MultiBody_Kernel.md]
skills: [gpu-optimize, port-to-opencl, forcefield-validation]
---

# Concurrent multi-molecule relaxation on FAF substrate

## 1. Goal

We want to relax or run MD of **an assembly of molecules on a surface** — e.g. PTCDA molecules adsorbed on NaCl (or CaF2), forming a self-assembled monolayer. The molecules form **one coupled system**: they interact with each other via non-covalent forces (van der Waals / Pauli repulsion / electrostatics) and with the substrate below. All molecules move simultaneously and feel each other's forces at every step.

Each molecule is a **rigid body** (6-DOF: 3 translational + 3 rotational). The total force on each molecule has three contributions:

1. **Intermolecular PairFF** — non-covalent interactions between molecules (compact-exp Pauli+London + Coulomb). This is the coupling that makes it one system, not N independent molecules. Already in kernel 15.
2. **FAF substrate** — non-covalent interactions with the surface (folded-basis Pauli+London+Coulomb). The substrate is frozen; its effect is pre-fitted into a compact analytic basis. Currently only in kernel 13 (single active mol).
3. **Confinement springs** (optional) — k_z spring toward z_target, anchor springs on pinned atoms. Already in kernel 15.

All molecules are relaxed **concurrently**, each by its own workgroup, in a single kernel launch per timestep. This eliminates the Python launch overhead that dominates when cycling molecules one-by-one (the bottleneck documented in [MultiMol_MD_LaunchOverhead.md](MultiMol_MD_LaunchOverhead.md)).

**Without FAF, the concurrent kernels are useless for this use case** — molecules would float in vacuum with no substrate. This task fuses the FAF substrate forces into the concurrent multimol kernel, completing the original motivation for the entire launch-overhead effort.

## 2. Chosen method: fused multimol+FAF kernel

**One kernel, one launch per step.** Each workgroup evaluates the total non-covalent force on its molecule from two sources:

1. **Other molecules** (intermolecular PairFF) — compact-exp Pauli+London + Coulomb between atoms of different molecules. This is the **coupling** between molecules in the assembly. Already in kernel 15.
2. **Substrate** (FAF) — folded-basis Pauli+London+Coulomb from the frozen surface, evaluated at each atom's world position. Port from kernel 13.

Both accumulate into the same `f_acc`/`E_acc` arrays, then reduce to body force/torque and integrate the rigid-body equations of motion (Euler + gyro, FIRE or damped).

### Why fused (not two-kernel)?

| Approach | Launches/step | Sync needed | Complexity | Correctness |
|----------|:-------------:|:-----------:|:----------:|:-----------:|
| **Fused (chosen)** | 1 | none | port ~20 lines from kernel 13 | exact |
| Two-kernel (PairFF then FAF) | 2 | `finish()` between | new "allmol FAF" kernel + 2× launch overhead | exact |
| Sequential (existing `run_pairff`) | N (one per mol) | none | already works | exact but slow |

The fused approach is strictly better: no extra launch, no extra sync, and the FAF code is small and self-contained. The two-kernel approach would double the launch overhead we just spent weeks eliminating.

### Why FAF (not GridFF)?

| Substrate model | Memory | Eval cost | Already in `rigid.cl` | Suitable for multimol? |
|-----------------|--------|-----------|-----------------------|:----------------------:|
| **FAF (folded basis)** | ~KiB (nbasis × ntypes floats) | O(nbasis) per atom | Yes (kernel 3, 13) | ✓ compact, per-atom independent |
| GridFF (B-spline grid) | MiB (nx×ny×nz floats) | O(1) tricubic interp | Yes (kernel 2) | ✓ but larger memory, no per-type |
| Full atomistic substrate | N/A | O(n_substrate × n_atoms) | No | ✗ too expensive |

FAF is the right choice because:
- **Compact**: folded basis is ~128 floats vs MiB for GridFF → fits in local memory per workgroup
- **Per-atom independent**: each workgroup evaluates FAF for its own atoms → natural parallelism
- **Already implemented**: kernel 13 has the exact FAF evaluation loop, just needs porting
- **Per-type coefficients**: FAF supports different substrate interactions per atom type (C, N, O, H)
- GridFF can be added later as an alternative substrate model if needed (same fusion pattern)

## 3. Implementation plan

### Phase 1: New kernel `rigid_body_pairff_multimol_faf_kernel` (kernel 18)

Copy kernel 15 (`rigid_body_pairff_multimol_kernel`) and add:

**New arguments** (after `z_target`, before `dt`):
```c
__global const float*    folded_coeffs,      // [ntypes*nbasis] or [nbasis*4] if factorized
__global const float4*   folded_kxyz,        // [nbasis] basis parameters
__global const int*      folded_atom_type,   // [total_atoms] type index per atom (-1 = skip)
const int4               folded_meta,        // (nbasis, ntypes, 0, 0); ntypes<0 = factorized
const float4             folded_lvec2d,      // (ax, bx, ay, by) lattice vectors
```

**New local memory** (already declared in kernel 13):
```c
__local float4 LBASIS[FOLDED_BASIS_MAX_RIGID];              // nbasis basis params
__local float  LCOEFFS[FOLDED_TYPES_MAX_RIGID * FOLDED_BASIS_MAX_RIGID]; // coefficients
```

**Basis loading** (before the `niter` loop, after body state load — same as kernel 13 lines 3647–3650):
```c
for (int ib = lid; ib < nbasis; ib += lsize) LBASIS[ib] = folded_kxyz[ib];
const int ncoeff = factorized ? nbasis*4 : nbasis*ntypes;
for (int ib = lid; ib < ncoeff; ib += lsize) LCOEFFS[ib] = folded_coeffs[ib];
barrier(CLK_LOCAL_MEM_FENCE);
```

**FAF force accumulation** (in the per-atom force reduction loop, after inter-mol PairFF, before anchor springs — same as kernel 13 lines 3752–3764):
```c
const int ityp = factorized ? 0 : folded_atom_type[ia];
const float4 plqh = factorized ? ((__global const float4*)folded_atom_type)[ia] : (float4)(0.0f);
if ((factorized && dyn_type[ia] == 0) || (!factorized && ityp >= 0 && ityp < ntypes)) {
    float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
    float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
    u = u - floor(u); v = v - floor(v);
    for (int ib = 0; ib < nbasis; ib++) {
        float c = folded_coeff_rigid(LCOEFFS, ib, nbasis, ityp, plqh, factorized);
        float4 prm = LBASIS[ib];
        E += c * folded_eval_basis_rigid(u, v, p_world.z, prm);
        f -= c * folded_eval_grad_rigid(u, v, p_world.z, prm, invLvec2d);
    }
}
```

**Ping-pong**: same `poss_in`/`poss_out` layout as kernel 15. The FAF evaluation only reads `p_world` (computed from the molecule's own pose), so it doesn't interact with the ping-pong race fix.

**`invLvec2d` computation** (same as kernel 13 line 3654):
```c
float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
float det = ax*by - bx*ay;
float4 invLvec2d = (float4)( by/det, -bx/det, -ay/det, ax/det );
```

### Phase 2: Python API in `RigidBodyDynamics.py`

**New method** `run_multimol_md_faf(...)` — or extend `run_multimol_md` with `faf=True`:

```python
def run_multimol_md(self, n_steps, dt=0.05, ..., faf=None, batch=1, ...):
    use_faf = self.faf_mode if faf is None else bool(faf)
    if use_faf:
        if self.folded_params is None:
            raise RuntimeError("multimol+FAF requires init_folded(...) first")
        kname = "rigid_body_pairff_multimol_faf_kernel"
        # FAF args come from self.folded_params (already in kernel_params)
    else:
        kname = "rigid_body_pairff_multimol_kernel"
    # Rest is identical to existing run_multimol_md
```

The FAF arguments (`folded_coeffs`, `folded_kxyz`, `folded_atom_type`, `folded_meta`, `folded_lvec2d`) are already managed by `init_folded()` and stored in `self.kernel_params`. The `generate_kernel_args(kname)` call will pick them up automatically — no new buffer management needed.

**Launcher caching**: extend `_multimol_launch_pair` to include `use_faf` in the cache key, and select the kernel name accordingly.

### Phase 3: Persistent + single-WG variants (optional, later)

The same FAF fusion can be applied to:
- **Kernel 16** (persistent) — add FAF args + local memory loading + force accumulation
- **Kernel 17** (single-WG) — same

These are lower priority since the ping-pong kernel 15+FAF already gives 4–15× speedup. Persistent+FAF would give ~2% more (saves kernel-boundary sync). Single-WG+FAF is only useful for ≤2 molecules.

### Phase 4: Parity test

**L0 test** in `tests/test_forcefield.py`:

```python
@pytest.mark.gpu
def test_pairff_multimol_faf_parity():
    """Concurrent multimol+FAF must match sequential PairFF+FAF (one mol at a time)."""
    # Setup: 2-4 molecules on FAF substrate
    # Reference: run_pairff(N, faf=True) cycling active_mol (sequential, exact)
    # Test: run_multimol_md(N, faf=True) (concurrent, exact)
    # Assert: body state matches within 2e-6
```

The reference is the existing `run_pairff` with `faf_mode=True` — it evaluates the same PairFF+FAF physics but one molecule at a time. The concurrent kernel must produce identical forces (since FAF is per-molecule independent, there's no approximation).

### Phase 5: Benchmark

Extend `tests/bench_multimol_md.py` with `--faf` flag:
- Compare `A_opt` (multimol+FAF, concurrent) vs `A_naive` (sequential `run_pairff` with FAF, cycling active_mol)
- Measure speedup for 1, 2, 4, 8 molecules on a realistic substrate (e.g., PTCDA on graphene/NaCl)

Expected: same speedup ratios as the no-FAF case (4–135×), since FAF evaluation is O(nbasis × n_atoms) per molecule — small compared to O(n_mol²) inter-mol PairFF.

## 4. What can be reused

| Component | Source | Reuse |
|-----------|--------|-------|
| `folded_eval_basis_rigid` | `kernels/rigid.cl` line 822 | inline function, already global |
| `folded_eval_grad_rigid` | `kernels/rigid.cl` line 835 | inline function, already global |
| `folded_coeff_rigid` | `kernels/rigid.cl` line 857 | inline function, already global |
| FAF force loop | kernel 13 lines 3752–3764 | copy-paste (20 lines) |
| LBASIS/LCOEFFS loading | kernel 13 lines 3647–3650 | copy-paste (4 lines) |
| invLvec2d computation | kernel 13 line 3654 | copy-paste (3 lines) |
| `init_folded()` | `RigidBodyDynamics.py` line 804 | no change needed |
| `folded_params` in `kernel_params` | `RigidBodyDynamics.py` line 854 | no change needed |
| `generate_kernel_args()` | `RigidBodyDynamics.py` | auto-picks up new kernel header |
| Ping-pong launcher caching | `RigidBodyDynamics.py` line 2370 | extend with `use_faf` key |

**No new Python infrastructure needed.** The FAF buffer management is already implemented by `init_folded()`. The new kernel just needs to be registered and called.

## 5. Local memory budget

Kernel 15 currently uses:
- `Ltorq[WORKGROUP_SIZE]` + `Lforce[WORKGROUP_SIZE]` — 2 × 32 × 16 = 1 KiB
- `Lenv_pos[MAX_STATIC_ATOMS]` + `Lenv_REQ[MAX_STATIC_ATOMS]` + `Lenv_g[MAX_STATIC_ATOMS]` — 3 × 128 × 16 = 6 KiB
- Misc (`pos`, `qrot`, `R`, `Rj`, etc.) — ~1 KiB

Adding FAF:
- `LBASIS[FOLDED_BASIS_MAX_RIGID]` — 128 × 16 = 2 KiB
- `LCOEFFS[FOLDED_TYPES_MAX_RIGID * FOLDED_BASIS_MAX_RIGID]` — 8 × 128 × 4 = 4 KiB

**Total: ~14 KiB** — well within the 48 KiB local memory limit on GTX 1650. No budget issues.

## 6. Edge cases

- **Epairs/σ-hole dummies** (`dyn_type != 0`): FAF is skipped for these (check `dyn_type[ia] == 0` or `ityp >= 0`). Already handled in kernel 13.
- **Factorized mode** (`folded_meta.y < 0`): per-atom PLQH coefficients stored as `float4` in `folded_atom_type` buffer. Already handled by `folded_coeff_rigid`.
- **k_z spring + z_target**: when FAF is active, typically set `k_z=0` and `z_target=0` (the substrate provides the z-force). The kernel still applies the spring if `k_z > 0` — this is a user choice, not a bug.
- **Anchor springs**: orthogonal to FAF, both can be active simultaneously.

## 7. Future: GridFF variant

If a user has a GridFF substrate (B-spline 3D grid) instead of a folded basis, the same fusion pattern applies:
- Add `BsplinePLQ`, `grid_ns`, `grid_invStep`, `grid_p0` arguments
- Replace the FAF basis loop with tricubic B-spline interpolation (from kernel 2)
- The rest of the kernel is identical

This is deferred — FAF is the primary substrate model. GridFF fusion can be added when needed.

## 8. Success criteria

1. **L0 parity**: `run_multimol_md(N, faf=True)` matches `run_pairff(N, faf=True)` (sequential, cycling active_mol) within 2e-6 for 2-4 molecules
2. **Performance**: 4–135× speedup over sequential (same ratios as no-FAF case)
3. **No regression**: existing `run_pairff(faf=True)` still works unchanged
4. **Fail loud**: calling `run_multimol_md(faf=True)` without `init_folded()` raises RuntimeError

## 9. Files to modify

| File | Change |
|------|--------|
| `kernels/rigid.cl` | Add kernel 18 `rigid_body_pairff_multimol_faf_kernel` (~20 lines new + copy from kernel 13) |
| `spammm/forcefields/RigidBodyDynamics.py` | Extend `run_multimol_md` with `faf` parameter; extend `_multimol_launch_pair` cache key |
| `tests/test_forcefield.py` | Add `test_pairff_multimol_faf_parity` |
| `tests/bench_multimol_md.py` | Add `--faf` flag + `A_opt_faf` strategy |
| `doc/Tasks/MultiMol_FAF_ConcurrentRelax.md` | This document |

## 10. What this does NOT do

- Does not implement GridFF fusion (FAF only — GridFF is future work)
- Does not implement FAF for persistent (kernel 16) or single-WG (kernel 17) variants (Phase 3, optional)
- Does not change the existing `run_pairff(faf=True)` single-active-mol path
- Does not implement neighbor lists / cutoff (still O(n_mol²) inter-mol — separate task)
