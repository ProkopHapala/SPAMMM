# Performance Benchmark: UFF/SPFF Relaxation

**Goal:** GUI "Relax" button should feel instant (< 0.5s for typical molecules, < 2s for large PAHs).

**Status:** Benchmarks run (2026-07-19) on `flat_1.mol2` — **unverified** pending USER review of `debug/test_relax_flat1/` artifacts.

---

## Measured results — `flat_1.mol2` (96 atoms, 66 C + 30 H)

Input: `/home/prokop/svec_triptacene/flat_1.mol2`  
Harness: `tests/test_relax_flat1.py` (`pytest tests/test_relax_flat1.py --develop -s`)  
GPU: NVIDIA RTX 3090, `PYOPENCL_CTX=0`

| Case | Mode | Steps | Time | E | fmax | planarity_C | Notes |
|------|------|-------|------|---|------|-------------|-------|
| Vacuum UFF | multi-kernel/step | 2000 | **2.07 s** | 9.99 | 1.01 | 0.31 | mean C–C 1.41, C–H 1.08 |
| Vacuum SPFF | `relax_batch` | 2000 | **0.80 s** | −424.21 | 2.62 | 0.36 | damp=0.9 |
| Vacuum SPFF | `relax_serial` WG=192 | 2000 | **0.0049 s** | −424.21 | 2.62 | 0.36 | **~163× vs batch**, identical E/fmax |
| NaCl GridFF + SPFF | batch + GridFF ex2 | 1500 | **0.93 s** | −413.71 | 2.95 | — | z_rel(min−top)=3.55 Å (stays above surface) |

Artifacts (L2 review):

- `debug/test_relax_flat1/uff_vacuum_{geometry.png,init_final.xyz,out}`
- `debug/test_relax_flat1/spff_batch_vacuum_*`
- `debug/test_relax_flat1/spff_serial_vacuum_*`
- `debug/test_relax_flat1/spff_substrate_{geometry.png,init_final.xyz,out}`

### Findings

1. **Serial local-memory kernel works for flat_1** after memory-safe sizing (`WG_SIZE=192`, `MAX_NVEC=192`, `MAX_NATOM=128`, `MAX_NNODE=96` ≈ 38 KB local — fits NVIDIA 48 KB). Naive WG=256 with 12×WG float4 arrays would be ~64 KB and fail.
2. **Parity**: serial vs batch on flat_1 give identical energy and fmax (and `tests/test_relax_serial.py` still all-pass).
3. **GUI damp default was too low**: `DEFAULT_DAMP=0.1` left SPFF at fmax≈12 after 2000 steps; `damp=0.9` → fmax≈2.6. Default updated to **0.9**.
4. **UFF** is usable at 96 atoms (~2 s / 2000 steps) but has no serial fused kernel.
5. **UFF+GridFF** not wired; substrate case is SPFF+GridFF only. `nonbonded_grid.cl` now loads when `enable_nonbond=True`.

### Serial limits (updated)

| Cap | Value |
|-----|-------|
| `WG_SIZE` | 192 |
| `MAX_NVEC` | 192 |
| `MAX_NATOM` | 128 |
| `MAX_NNODE` | 96 |
| Requires | `nSystems==1`, no non-bonded / no GridFF in serial kernel |

---

## Current relaxation paths

### GUI path (`FFExtension._on_relax`)

`spammm/GUI/FFExtension.py`

1. `_ensure_built(window)` → `FFController.build_ff()` (if not built)
2. `ctrl.relax_until_converged(max_steps, dt, damp, callback, batch_size)`
3. Callback `_cb()` runs **every batch**: GPU→host sync, AtomicGraph update, Vispy refresh, `processEvents`

### GPU paths (`FFController.relax_n` / `relax_until_converged`)

- **`relax_serial`**: Single-kernel local-memory, WG=192. Runs N steps in one kernel call. ~160× faster than batch on flat_1. Caps above; no non-bonded.
- **`relax_batch`**: Per-step kernel launches; sync at end (or each GUI callback).

### Remaining suspected bottlenecks (GUI)

1. **GUI callback overhead** every batch (`get_state` + refresh + `processEvents`)
2. **Serial unavailable** when non-bonded / GridFF / nvecs>192
3. Kernel launch overhead in batch mode for large systems

## Optimization targets (still open)

1. Reduce GUI callback frequency (refresh every N batches or only at end)
2. Optional WG=256 with even tighter local packing if needed for larger PAHs
3. Serial / fused kernel + GridFF (not started)
4. Wire UFF into `FFController` GUI combo (still `NotImplementedError`)

## Future work — fused kernels completeness (do not start until scheduled)

Status: **planned / not started**. Notes only — no implementation yet.

### UFF fused (`kernels/UFF.cl` `relax_nsteps_local_UFF` / `relax_nsteps_global_UFF`)

- [ ] **Torsions / dihedrals** in the fused multi-step kernels (parallel eval → force slots → gather, same pattern as angles — never serial `iL==0`).
- [ ] **Inversions** in the same fused path (parity with multi-kernel UFF).
- [ ] Parity test vs multi-kernel UFF on flat_1 / benzene (E, fmax, geometry).
- Today fused UFF is **bonds + Fourier angles only**; that is why energies differ from full UFF multi-kernel.

### SPFF fused (`kernels/SPFF.cl` `relax_nsteps_serial` / `relax_nsteps_global`)

- [ ] Audit / complete **π–π** (`evalPiAling`) and **π–σ** (`evalAngCos` / Ksp) coverage in both local and global fused loops; keep parity with `getSPFFf4` / `relax_batch`.
- [ ] Document which terms are in serial vs global vs batch; add L0 parity asserts if any term is missing.

### Substrate + non-covalent (fused path)

- [ ] Wire **FAF** (`getSurfFolded` / `upload_folded_fit` + `relax_global(..., do_faf=True)`) into flat_1 / GUI substrate relax — analytic substrate, not GridFF-first.
- [ ] Wire **non-covalent** (LJ/Coulomb exclusions, and/or GridFF) into fused multi-step kernels without falling back to per-step host launches.
- [ ] Bench FAF fused vs GridFF batch on NaCl (speed + z_rel / energy).

See also: `doc/ToDo/ToDo.agents.md` (Soon items), `doc/ARCHITECTURE_ROADMAP.md` §5, `doc/GUI_FF_Relaxation.md` § non-bonded gaps.

## Success criteria

- Benzene (12 atoms): < 0.1s to convergence (fmax < 0.05)
- Coronene (24 atoms): < 0.3s
- Pentacene (36 atoms): < 0.5s
- Large PAH (~100 atoms): < 2s — **flat_1 serial vacuum: 0.005 s / 2000 steps (GPU only)**
- GUI remains responsive during relaxation
- Serial kernel used for molecules with nvecs≤192, nnode≤96

## References

- `spammm/forcefields/FFController.py` — relax_n, `_can_use_serial`
- `spammm/forcefields/SPFF_cl.py` — `relax_serial` / `relax_batch`, `SERIAL_*` caps
- `kernels/SPFF.cl` — `relax_nsteps_serial`
- `spammm/forcefields/UFF_cl.py` — UFF relax path
- `spammm/GUI/FFExtension.py` — GUI wiring
- `tests/test_relax_flat1.py` — flat_1 systematic benchmarks
- `tests/test_relax_serial.py` — serial vs batch parity
