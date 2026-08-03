---
name: centralized-plotting
description: 2D scalar field plotting — use shared plotUtils functions, avoid transpose/aspect-ratio/alignment bugs
trigger:
  glob:
    - "**/*.py"
    - "**/GUI/**"
---

## Core Rule: Use Shared Plotting Functions, Never Roll Your Own

Before writing ANY matplotlib/VisPy plotting code, check `spammm/plotUtils.py` and `spammm/GUI/plotutils.py` for existing functions. This is a blocking step — do not skip it. See also `code-reuse` skill.

**AFM E / Fz / df maps and z-profiles** are NOT covered here — use skill:`afm-plotting` and `spammm.SPM.AFM_utils` (`imshow_afm`, `plot_afm_height_panel`, `plot_afm_z_profiles`, `plot_fdbm_vs_kriging_zlayout`, …). Do not reinvent those with raw `imshow`/`plot`.

## Shared Functions (SSOT)

### Pure matplotlib — `spammm/plotUtils.py`
- `compute_grid_extent(apos)` — non-square grid from atomic positions, preserves aspect ratio
- `make_2d_grid(grid_origin, size_xy, center_z, z_height, n=200)` — generates grid points, returns `(points, extent, nx, ny)`
- `plot_2d_scalar(data_2d, extent, title, z_label, cmap, symmetric, apos, enames)` — complete heatmap with colorbar + atom overlay
- `overlay_atoms(ax, apos, enames, xs, ys, label_heavy)` — atom scatter with element colors
- `plot_field_slice(ax, field, origin, step, z, cmap, title, sym)` — 2D slice of 3D field
- `plot_1d`, `plot_funcs`, `plot_compare_1d`, `plotGeometry`, etc. — 1D and geometry plotting

### Qt-specific — `spammm/GUI/plotutils.py`
- `show_in_plot_window(window, fig, title, attr)` — embed matplotlib Figure in reusable QDialog
- Re-exports all pure functions from `plotUtils.py`

## Recurring Bugs to Avoid

### 1. Not Using Existing Functions
**Symptom**: Writing inline `imshow` + `colorbar` + `scatter` code instead of calling `plot_2d_scalar`.
**Fix**: Always import from `plotUtils` / `GUI.plotutils`. If you need a variant, extend the shared function, don't copy-paste.
**Check**: `grep_search` for `imshow` in your file — if it's not inside `plotUtils.py`, you're probably duplicating.

### 2. Wrong `.T` Transpose
**Symptom**: Potential spots don't align with atoms; x/y axes appear swapped.
**Root cause**: `np.meshgrid(xs, ys)` with default `indexing='xy'` produces `X.shape = (ny, nx)`. After `.ravel()` and `.reshape(ny, nx)`, `data[i,j]` already maps to `(xs[j], ys[i])` — which is exactly what `imshow(origin='lower', extent=[xmin,xmax,ymin,ymax])` expects **without** `.T`.
**Rule**: 
- With `make_2d_grid` (uses `indexing='xy'`): `imshow(data_2d, origin='lower', extent=extent)` — **NO `.T`**
- With `indexing='ij'` (some AFM code): `data[i,j]` = `(xs[i], ys[j])` → needs `.T` for imshow
- **Always check which `meshgrid` indexing was used before deciding on `.T`**

### 3. Wrong Aspect Ratio / Square Grid for Non-Square Molecule
**Symptom**: Plot doesn't fit molecule; stretched or padded with empty space.
**Root cause**: Using `size = max(width, height)` and `nx = ny = n` forces a square grid even when the molecule is elongated.
**Fix**: Use `compute_grid_extent` (returns per-axis `size_xy`) + `make_2d_grid` (computes `nx ≠ ny` to preserve aspect ratio). Never hardcode `nx = ny`.
**Check**: If `nx == ny` but `x_span != y_span`, you have a bug.

### 4. Wrong Atom Alignment
**Symptom**: Atom overlay dots don't sit on top of potential/density features.
**Root causes**:
- Transpose bug (see #2) — x and y swapped in the image
- `extent` doesn't match the grid that generated `data_2d`
- Atom positions not projected to 2D correctly (forgetting to use same x,y as grid)
**Fix**: Use `overlay_atoms` from `plotUtils` — it derives `xs, ys` from `extent` and `data.shape` to ensure consistency. Or use `plot_2d_scalar` which handles everything end-to-end.
**Check**: Print `extent` and `apos[:, :2].min/max` — they must overlap.

## Decision Tree: Which Function to Use

```
Need a 2D heatmap of scalar data?
├── Have atomic positions + want atom overlay?
│   └── plot_2d_scalar(data, extent, ..., apos=pos, enames=enames)  ← does everything
├── Just the heatmap, no atoms?
│   └── plot_2d_scalar(data, extent, ..., apos=None)
├── Need it in a Qt window?
│   └── fig = plot_2d_scalar(...); show_in_plot_window(window, fig)
├── Slicing a 3D field?
│   └── plot_field_slice(ax, field, origin, step, z, ...)
└── Something custom?
    ├── Check plotUtils.py first for existing helpers
    └── If truly new: add it to plotUtils.py, don't inline it
```

## Code Pattern: Standard 2D ESP/Density Plot

```python
from spammm.plotUtils import compute_grid_extent, make_2d_grid, plot_2d_scalar
from spammm.GUI.plotutils import show_in_plot_window  # if in GUI context

pos = np.asarray(sys.apos, dtype=np.float64)
grid_origin, size_xy, center_z = compute_grid_extent(pos)
points, extent, nx, ny = make_2d_grid(grid_origin, size_xy, center_z, z_height)

# ... compute scalar values at grid points ...
data_2d = values.reshape(ny, nx)  # NOTE: ny rows, nx cols — matches meshgrid('xy')

fig = plot_2d_scalar(data_2d, extent, title=f"ESP z={z_height:.1f}Å",
                     z_label='eV', cmap='seismic', symmetric=True,
                     apos=pos, enames=sys.enames)
show_in_plot_window(window, fig, title="ESP")
```

## STOP Triggers

Before committing plotting code, verify:
- [ ] No `imshow` call outside `plotUtils.py` (unless genuinely novel use case)
- [ ] No hardcoded `nx = ny` when molecule aspect ratio is non-square
- [ ] No `.T` on `data_2d` unless you used `indexing='ij'` in meshgrid
- [ ] Atom overlay uses `overlay_atoms` or `plot_2d_scalar(..., apos=...)`
- [ ] `extent` passed to `imshow` matches the grid that produced `data_2d`

## Color Scale Rule (USER MANDATED — DO NOT DEVIATE)

For 2D potential/energy maps (PairFF, FAF, combined), the color scale MUST be:
- `vmin = Emin` (the actual data minimum, which is negative)
- `vmax = -vmin = |Emin|` (symmetric around 0)
- Floor: `vmax = max(|Emin|, 0.01)` to avoid zero-range

This is `vmax = |Emin|`, `vmin = -|Emin|`. Repulsive Pauli cores (positive, much larger
than `|Emin|`) are intentionally clipped — this shows the attractive basins and FAF
corrugation clearly. **Never** use `max(|Emin|, |Emax|)`, percentile-based limits, or
asymmetric `[Emin, Emax]` ranges for these maps. The SSOT function is
`spammm.GUI.RigidBodyVispy.potential_to_rgba` — always delegate to it.

For GUI map overlays that cache and recolor: reset the color-limit spin to 0 (Auto)
on every recompute so `|Emin|` is freshly evaluated from the new data. A stale positive
spin value will use the old limit and oversaturate the new map.

**Nuclear exclusion:** The probe-map kernels (GPU `rigid_body_pairff_probe_grid` and
CPU `_compute_unified_probe_pair_map`) compute the physical energy at every pixel —
no NaN holes in the display. A separate boolean `exclude_mask` (True within 1 Å of any
real atom) is returned by `compute_combined_probe_map` and
`RigidBodyUtils.nuclear_exclusion_mask`. The mask is used only to exclude nuclear
singularities from `vmin`/`vmax` estimation (`E_for_lim = E[~exclude_mask]`), so the
color scale reflects the chemically meaningful attractive basins and FAF corrugation
rather than the infinitely deep wells at nuclei. The map itself remains fully finite
and is displayed everywhere.

## Related Skills
- `code-reuse` — general inventory-first rule for all code
- `visual-debugging` — diagnostic plots for debugging
- `doc-read-navigate` — where to search for existing implementations
