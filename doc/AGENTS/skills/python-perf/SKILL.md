---
name: python-perf
description: Performant Python for scientific computing — vectorization, NumPy anti-patterns, preallocation
trigger:
  glob:
    - "**/*.py"
    - "**/tests/**/*"
    - "**/*bench*.py"
    - "**/*perf*.py"
---

## Core Principle

**Python is the harness, not the engine.**  Call NumPy or OpenCL kernels on as many items at once as possible to minimize harness overhead. Use advanced array slicing. NEVER write low-level hot loops in Python.

## Rules

### 1. Batch Everything
- Minimize number of loop iterations in python.
- Process entire arrays in single NumPy/OpenCL calls
- Minimize Python function call overhead
- One kernel call for 1M points > 1M kernel calls for 1 point each

### 2. Vectorized Operations Only
- **Forbidden:** `for i in range(len(array)): result[i] = array[i] * 2`
- **Required:** `result = array * 2`
- **Forbidden:** Nested loops over 2D/3D grids
- **Required:** `np.meshgrid`, slicing, broadcasting, or OpenCL kernels

### 3. Minimal Allocation
- Preallocate buffers, reuse in hot paths
- Use in-place operations (`array *= 2` vs `array = array * 2`)
- Avoid intermediate arrays where possible

### 4. Advanced Slicing
- `array[mask]` for conditional selection
- `array[:, :, 0]` for channel extraction
- `array.reshape(-1)` for flattening

### 5. Batch All Trial/Replica Math
When generating trial poses (MC/GA/annealing), batch ALL trials into single NumPy operations — never loop over trials in Python.

**Forbidden:** `for r in range(n_trial): poss[r, i, 0] += float(rng.normal(0, dxy))`
**Required:** `rnd = rng.normal(size=(n_trial, n_moved, 3)); poss[1:, moved, 0] += rnd[..., 0] * dxy`

Quaternion updates: use NumPy broadcasting for the full `(trial, moved)` batch, not `_quat_mul` in a loop. Planar rotation about z-axis is a 4-component product that broadcasts directly:
```python
s = np.sin(0.5 * rnd[..., 2] * dphi); c = np.cos(0.5 * rnd[..., 2] * dphi)
qr[..., 0] = q0[..., 0] * c + q0[..., 1] * s  # x
qr[..., 1] = q0[..., 1] * c - q0[..., 0] * s  # y
qr[..., 2] = q0[..., 3] * s + q0[..., 2] * c  # z
qr[..., 3] = q0[..., 3] * c - q0[..., 2] * s  # w
qr /= np.linalg.norm(qr, axis=-1, keepdims=True)
```

Packing/reduction energy: one `np.sum(d * d, axis=(1, 2))` for all trials, not `for r: E[r] += packing_energy(poss[r])`.

### 6. Active-Set Incremental Updates
When only a subset of objects move (active set in MC/GA), the interactions among frozen objects are invariant. Use `E_new = E_old + (E_active_new - E_active_old)` instead of recomputing the full system energy. This eliminates a redundant full evaluation (and often a redundant GPU kernel launch) per step.

**Condition:** the energy must decompose as `E = E_frozen_frozen + E_active_frozen + E_active_active + E_one`. Since `E_frozen_frozen` is constant across a trial batch, `ΔE_full = ΔE_active`. Verified numerically (PairFF parity err = 3.9e-08).

## Common Anti-Patterns

### ❌ Grid Iteration
```python
# WRONG: 1M Python iterations
for i in range(nx):
    for j in range(ny):
        result[i,j] = grid[i,j] * weight
```

### ✅ Vectorized
```python
# CORRECT: Single NumPy operation
result = grid * weight
```

### ❌ Position Generation
```python
# WRONG: Nested loops + stack overhead
positions = []
for ix in range(nx):
    for iy in range(ny):
        positions.append([x[ix], y[iy], z])
positions = np.array(positions)
```

### ✅ Meshgrid (Simple)
```python
# CORRECT: 3D grid directly, minimum allocation
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
# Use xx, yy, zz directly as 3D arrays
```

### ❌ Conditional Loop
```python
# WRONG: Python loop with condition
for i in range(len(data)):
    if data[i] > threshold:
        result[i] = data[i] * scale
```

### ✅ Boolean Masking
```python
# CORRECT: Vectorized
mask = data > threshold
result[mask] = data[mask] * scale
```

## When Python Loops ARE Acceptable

- Orchestration: file I/O, argument parsing
- Small constants: <100 iterations, not in hot path
- Debugging: temporary prints (remove before commit)

