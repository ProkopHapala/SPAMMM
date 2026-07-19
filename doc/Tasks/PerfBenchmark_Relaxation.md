# Performance Benchmark: UFF/SPFF Relaxation

**Goal:** GUI "Relax" button should feel instant (< 0.5s for typical molecules, < 2s for large PAHs).

**Status:** Not started. This document defines the benchmarking plan and identifies suspected bottlenecks.

---

## Current relaxation paths

### GUI path (`FFExtension._on_relax`)

`@/home/prokop/git/SPAMMM/spammm/GUI/FFExtension.py:382`

1. `_ensure_built(window)` → `FFController.build_ff()` (if not built)
2. `ctrl.relax_until_converged(max_steps, dt, damp, callback, batch_size)`
3. Callback `_cb()` runs **every batch**:
   - `ctrl.get_state()` — copies positions from GPU to host
   - `_sync_positions_to_graph()` — updates AtomicGraph
   - `window.refresh_view()` — redraws 3D scene
   - `QtWidgets.QApplication.processEvents()` — processes Qt events

### GPU paths (`FFController.relax_n` / `relax_until_converged`)

`@/home/prokop/git/SPAMMM/spammm/forcefields/FFController.py:164`

- **`relax_serial`**: Single-kernel local-memory, 128-thread workgroup. Runs N steps in one kernel call. ~150x faster than batch. **Only works when** `nSystems==1, nvecs<=128, nnode<=64, no non-bonded`.
- **`relax_batch`**: Per-step kernel launches with `queue.finish()` only at end. Faster than per-step sync but still has kernel launch overhead per step.

### Suspected bottlenecks

1. **GUI callback overhead**: `_cb()` calls `get_state()` + `_sync_positions_to_graph()` + `refresh_view()` + `processEvents()` **every batch**. If `batch_size` is small (e.g. 100 steps), this means frequent GPU→CPU transfers and Qt redraws during relaxation. This is likely the #1 bottleneck for perceived speed.

2. **Serial kernel not used**: `_can_use_serial()` requires `nvecs <= 128, nnode <= 64`. For molecules with > 64 atoms or > 128 bond vectors, falls back to `relax_batch`. Check if threshold is too conservative or if large molecules always hit batch path.

3. **Non-bonded enabled**: If `enable_nonbond=True` (default?), serial path is disabled. Non-bonded kernel may be slow or unnecessary for planar PAHs.

4. **Kernel launch overhead in batch mode**: `relax_batch` launches 4+ kernels per step (cleanForce, evalBond, evalAngle, evalDihedral, evalInversion, assembleForces, updateAtoms). For 5000 steps = 20,000+ kernel launches. Each launch has ~10μs overhead → 0.2s just in launches.

5. **Queue synchronization**: `relax_batch` only syncs at end, but `get_state()` in callback forces implicit sync via `cl.enqueue_copy` → blocks CPU until GPU finishes.

6. **FFController.build_ff() rebuild**: If GUI doesn't cache the controller between relax calls, every click rebuilds kernels + reallocates buffers.

## Benchmarking plan

### Step 1: Isolate GPU time from GUI overhead

```python
# In _on_relax, time the components:
t0 = time.time()
result = ctrl.relax_until_converged(...)  # pure GPU
t_gpu = time.time() - t0

# vs with callback:
t0 = time.time()
result = ctrl.relax_until_converged(..., callback=_cb, batch_size=nsteps)
t_total = time.time() - t0

print(f"GPU: {t_gpu:.3f}s  Total: {t_total:.3f}s  GUI overhead: {t_total-t_gpu:.3f}s")
```

### Step 2: Benchmark serial vs batch

```python
# For same molecule, same nsteps:
ctrl.md.relax_serial(nsteps=1000)  # time this
ctrl.md.relax_batch(nsteps=1000)   # time this
```

### Step 3: Profile kernel times

Use `cl.command_queue` profiling:

```python
import pyopencl as cl
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
# ... launch kernels ...
# Get event profiling info: event.get_profiling_info(cl.profiling_info.START)
```

### Step 4: Test with increasing molecule size

| Molecule | Atoms | nvecs | nnode | Serial? | Expected |
|----------|-------|-------|-------|---------|----------|
| Benzene | 12 | ~18 | 12 | Yes | < 0.1s |
| Coronene | 24 | ~42 | 24 | Yes | < 0.2s |
| Pentacene | 36 | ~66 | 36 | Yes | < 0.3s |
| Triacene | 60 | ~100+ | 60 | Maybe | < 1s |
| Large PAH (100+ atoms) | 100+ | >128 | >64 | No (batch) | < 2s |

### Step 5: Test molecules

User specified `flat_1.mol2` and `flat_1_relaxed.mol2` — need to create or locate these. Likely a flat PAH for testing relaxation convergence. Use coronene or pentacene from `data/xyz/` as fallback.

## Optimization targets

1. **Reduce callback frequency**: Only refresh GUI every N batches (e.g. every 1000 steps), not every batch. Or: run relaxation fully on GPU, only sync at end.
2. **Increase serial threshold**: If `relax_serial` kernel can handle nvecs=256, raise the limit. Check local memory size.
3. **Disable non-bonded for planar molecules**: Non-bonded (vdW) may not be needed for in-plane relaxation of PAHs.
4. **Cache FFController**: Don't rebuild on every relax click if molecule hasn't changed topology.
5. **Fire-and-forget relaxation**: Run relaxation in background thread, update GUI only at end or on timer.

## Success criteria

- Benzene (12 atoms): < 0.1s to convergence (fmax < 0.05)
- Coronene (24 atoms): < 0.3s
- Pentacene (36 atoms): < 0.5s
- Large PAH (100+ atoms): < 2s
- GUI remains responsive during relaxation (no freezing)
- Serial kernel used for molecules ≤ 64 atoms

## References

- `spammm/forcefields/FFController.py` — relax_n, relax_until_converged, _can_use_serial
- `spammm/forcefields/SPFF_cl.py` — relax_serial (line 1724), relax_batch (line 1703), run_step_basic (line 1637)
- `spammm/forcefields/UFF_cl.py` — UFF relax path
- `spammm/GUI/FFExtension.py` — _on_relax (line 382), _cb callback
- `tests/test_forcefield.py` — existing FF tests (functional, not perf)
- `tests/test_relax_serial.py` — serial vs batch parity test
