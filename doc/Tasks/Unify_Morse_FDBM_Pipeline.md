# Task: Unify Morse+Q and FDBM CLI Pipeline

## Problem

`afm-morse` and `afm` (FDBM) use **completely separate pipelines** from grid setup through plotting. They should share the same pipeline from the PP-relaxation stage onward, differing only in how the forcefield grid is generated.

## Current State — Two Divergent Pipelines

### FDBM path (`afm` → `cmd_afm` → `run_fdbm_pp_from_density`)

| Stage | What happens | Code |
|-------|-------------|------|
| Geometry | Planarize z=0, PCA orient long axis → x | `run_spm.py:200-211` |
| Grid | `make_fdbm_grid_com_zsym(atomPos, step=0.1, margin=4.0, z_extra=6.0)` → origin, ngrid, step. Z-symmetric, COM-centered, 0.1 Å spacing | `run_spm.py:214-215` |
| Forcefield | `stage3_fdbm_fields_fast` → E_pauli + E_ES + E_vdw → F_total (gradient) | `AFM_utils.py:3353-3357` |
| Grid upload | `setup_fdbm_grid(F_total, origin, step)` → OpenCL image | `AFM_utils.py:3420` |
| PP scan | `scan_fdbm(scan_xs, scan_ys, h_scan, K_LAT, K_RAD, bond_length, use_fire=True, ppm_mode=True)` | `AFM_utils.py:3421-3426` |
| Scan geometry | `scan_xs = arange(atomPos.x_min - scan_margin, atomPos.x_max + scan_margin, step)`, same for y. `h_scan = afm_df_height_stacks(h_min=3.7, h_max=4.7, h_step=0.1, amp=1.0, amp_align=True)` | `AFM_utils.py:3388-3398` |
| df | `compute_df_amp_dir(FEs, spacing, osc_dir, amp=1.0)` — proper amplitude convolution | `AFM_utils.py:3438` |
| Fz extraction | Fz at `h_Fz = h_df - amp` (amp-aligned) | `AFM_utils.py:3440-3445` |
| Plotting | `plot_afm_variant_height_strip(variants, row_specs, heights, scale='per_image', long_axis_vertical=True, tight=True, extent=scan_extent(...))` | `run_spm.py:322-325` |
| CLI args | `--step --margin --h-min --h-max --h-step --amp --K-LAT --K-RAD --bond-length --scan-margin --osc-dir --base-pos --scale --df-cmap --cmap --plots --show-atoms` | `run_spm.py:62-123` |

### Morse+Q path (`afm-morse` → `cmd_afm_morse` → `run_morse_coulomb_afm`)

| Stage | What happens | Code |
|-------|-------------|------|
| Geometry | **No planarization, no PCA orientation** | — |
| Grid | `setup_grid(n=(60,60,40), margin=4.0, z_top=14.0)` — coarse (~0.18 Å), not z-symmetric, not COM-centered | `AFM_utils.py:3797` |
| Forcefield | `make_forcefield()` — Morse + point-charge Coulomb directly on grid | `AFM_utils.py:3798` |
| Grid upload | Inside `setup_grid` (uses `dinvA/B/C` for OpenCL image sampling) | `AFM.py:setup_grid` |
| PP scan | `run_scan(nxy=(40,40), nz=25, dtip=-0.15)` — `relaxStrokesTilted` (no FIRE, no ppm_mode) | `AFM_utils.py:3802` |
| Scan geometry | 90% of molecule bounding box, 40×40 pixels (not step-based), nz=25 with dtip=-0.15 (not h_min/h_max/h_step) | `AFM.py:920-930` |
| df | `compute_df(Fz, abs(dtip))` — simple derivative, **no amplitude convolution** | `AFM_utils.py:3804` |
| Fz extraction | All nz slices (no amp alignment) | — |
| Plotting | `plot_afm_height_panel(data, heights, iz=sel, extent=..., ...)` — basic panel, no variant strip, no per_image scale, no long_axis_vertical | `AFM_utils.py:3834-3837` |
| CLI args | `--nx --ny --nz --margin --z-top --scan-nx --scan-ny --nz-scan --dtip --slice-indices` | `run_spm.py:718-733` |

## Key Differences (what must be unified)

1. **Grid setup**: Morse uses `setup_grid(n, margin, z_top)` → should use `make_fdbm_grid_com_zsym(atomPos, step, margin, z_extra)` for same grid quality (0.1 Å, z-symmetric, COM-centered)
2. **Forcefield generation**: Morse uses `make_forcefield()` on the `setup_grid` grid → should generate Morse+Coulomb `F_total` on the FDBM-style grid, then `setup_fdbm_grid(F_total, origin, step)` to upload
3. **PP relaxation**: Morse uses `relaxStrokesTilted` (no FIRE) → should use `scan_fdbm(scan_xs, scan_ys, h_scan, K_LAT, K_RAD, bond_length, use_fire=True, ppm_mode=True)` — same as FDBM
4. **Scan geometry**: Morse uses nxy/nz/dtip → should use `scan_xs/scan_ys` from `atomPos ± scan_margin` at `step`, and `h_scan` from `afm_df_height_stacks(h_min, h_max, h_step, amp, amp_align)`
5. **df computation**: Morse uses `compute_df(Fz, dtip)` → should use `compute_df_amp_dir(FEs, spacing, osc_dir, amp)` for proper amplitude convolution
6. **Fz extraction**: Morse returns all slices → should extract Fz at amp-aligned heights `h_Fz = h_df - amp`
7. **Plotting**: Morse uses `plot_afm_height_panel` → should use `plot_afm_variant_height_strip` with `scale='per_image'`, `long_axis_vertical=True`, `tight=True`
8. **Geometry prep**: Morse skips planarization and PCA orientation → should do the same as FDBM
9. **CLI args**: Morse has completely different args → should share the same scan/plot args as FDBM (`--step --h-min --h-max --h-step --amp --K-LAT --K-RAD --bond-length --scan-margin --osc-dir --scale --df-cmap --cmap --plots --show-atoms`), plus Morse-specific `--params --lj`
10. **Return dict**: Morse returns `{Fz, df, heights, ...}` → should return same dict shape as FDBM (`{df, Fz, heights, heights_Fz, scan_xs, scan_ys, amp_align, ...}`) so `plot_afm_variant_height_strip` works identically

## What Should Stay Different (forcefield generation only)

| | Morse+Q | FDBM |
|--|---------|------|
| Pauli + vdw | Morse potential (pairwise atom-tip) | `compute_pauli_overlap(rho_scf, tip_tot)` + `compute_dispersion_grid` |
| Coulomb | Point-charge Coulomb (atom charges) | `fft_poisson(rho_diff)` → `compute_es_conv_field(V_ES, tip_del)` |
| Result | F_total gradient on grid | F_total gradient on grid |
| **After this point** | **SAME pipeline** | **SAME pipeline** |

## Proposed Implementation

### Option A: Extend `run_fdbm_pp_from_density` with a Morse mode

Add a `forcefield_mode='fdbm'|'morse'` parameter to `run_fdbm_pp_from_density`. When `'morse'`:
- Skip density/tip density computation
- Call `afmulator.make_forcefield()` (Morse+Coulomb) instead of `stage3_fdbm_fields_fast`
- Upload via `setup_fdbm_grid` (same as FDBM)
- Everything after (scan, df, plot) is identical

### Option B: New `run_morse_pp_afm` function mirroring `run_fdbm_pp_from_density`

Create `run_morse_pp_afm(tag, atomPos, atomTypes, origin, step, ngrid, ...)` that:
- Sets up `AFMulator(use_morse=True)`
- Calls `make_forcefield()` to get F_total
- Calls `setup_fdbm_grid(F_total, origin, step)`
- Then calls `scan_fdbm` / `compute_df_amp_dir` / returns same dict — copy-paste from `run_fdbm_pp_from_density` lines 3388-3478

### CLI: Unify `afm-morse` args with `afm` args

```python
p_morse = sub.add_parser('afm-morse', help='Morse/LJ + Coulomb AFM (same pipeline as FDBM)')
_add_common_afm_args(p_morse)  # reuse ALL common args
p_morse.add_argument('--params', default=None, help='ElementTypes.dat path')
p_morse.add_argument('--lj', action='store_true', help='Use LJ instead of Morse')
# Remove: --nx --ny --nz --z-top --scan-nx --scan-ny --nz-scan --dtip --slice-indices
```

`cmd_afm_morse` then:
1. Resolves geometry (same as `cmd_afm`: planarize, orient)
2. Builds grid with `make_fdbm_grid_com_zsym`
3. Calls `run_morse_pp_afm(...)` or `run_fdbm_pp_from_density(forcefield_mode='morse', ...)`
4. Plots with `plot_afm_variant_height_strip` (same as `cmd_afm`)

### Expected result

```
python run_spm.py afm-morse --xyz data/xyz/pentacene.xyz
python run_spm.py afm         --xyz data/xyz/pentacene.xyz
```

Both produce `compare_per_image.png` with identical layout, same height range, same colorscale policy, same aspect ratio — only the physics content differs (Morse vs FDBM Pauli+ES+vdW).

## Contact-Surface 2.5D Mode — Third First-Class Citizen

### Current state

The contact-surface 2.5D AFM mode (`fit_contact_surface` → `run_scan_contact` / `cs_eval_separable`) is **not in the CLI at all**. It is only accessible via test scripts (`tests/testplot_contact_surface.py`) and debug diagnostics. This makes it a second-class experimental feature instead of a production method.

### What must be done

Add `afm-contact` as a third CLI subcommand that shares the same unified pipeline:

```
python run_spm.py afm-contact --xyz data/xyz/pentacene.xyz
```

| Stage | Contact-surface 2.5D specifics | Shared with FDBM/Morse |
|-------|-------------------------------|----------------------|
| Geometry | Same planarize + PCA orient | ✅ Same |
| Grid | Same `make_fdbm_grid_com_zsym` (FFT-friendly, 0.1 Å, z-symmetric) | ✅ Same |
| Forcefield | `fit_contact_surface` → `cs_eval_separable` (2.5D polynomial basis on h0 surface) | ❌ Different (this is the point) |
| Grid upload | `setup_fdbm_grid` or equivalent | ✅ Same interface |
| PP relaxation | `scan_fdbm` with FIRE, ppm_mode | ✅ Same |
| Scan geometry | `scan_xs/scan_ys` from atomPos ± scan_margin, `h_scan` from `afm_df_height_stacks` | ✅ Same |
| df | `compute_df_amp_dir` with amplitude convolution | ✅ Same |
| Fz extraction | Amp-aligned `h_Fz = h_df - amp` | ✅ Same |
| Plotting | `plot_afm_variant_height_strip` with `scale='per_image'`, `long_axis_vertical=True`, `tight=True` | ✅ Same |
| CLI args | Same common args (`--step --h-min --h-max --h-step --amp --K-LAT --K-RAD --bond-length --scan-margin --osc-dir --scale --df-cmap --cmap --plots --show-atoms`) | ✅ Same |

### Why this matters

For meaningful comparison between Morse+Q, FDBM, and contact-surface 2.5D, **all three must use identical harness**:
- **Input**: same molecule loading, SMILES support, z-planarization, PCA orientation
- **Output**: same figure alignment, colorscale policy, z-height sequence, Fz→df Giessibl conversion, lateral AFM support

Only then can differences in the images be attributed to physics (forcefield model), not to pipeline artifacts (different grid, scan, df, or plotting).

### CLI design

```python
p_contact = sub.add_parser('afm-contact', help='Contact-surface 2.5D AFM (same pipeline as FDBM/Morse)')
_add_common_afm_args(p_contact)  # reuse ALL common args
p_contact.add_argument('--params', default=None, help='ElementTypes.dat path')
# Contact-surface-specific args (basis, s_min/s_max, etc.) as needed
```

`cmd_afm_contact` then:
1. Resolves geometry (same as `cmd_afm`: planarize, orient)
2. Builds grid with `make_fdbm_grid_com_zsym`
3. Calls `run_contact_pp_afm(...)` (new function, mirrors `run_fdbm_pp_from_density` but uses `fit_contact_surface` for forcefield)
4. Plots with `plot_afm_variant_height_strip` (same as `cmd_afm`)

### Expected result

```
python run_spm.py afm-morse   --xyz data/xyz/pentacene.xyz
python run_spm.py afm         --xyz data/xyz/pentacene.xyz
python run_spm.py afm-contact --xyz data/xyz/pentacene.xyz
```

All three produce `compare_per_image.png` with identical layout — only physics content differs. Contact-surface 2.5D becomes a first-class production method.

---

## CRITICAL PERFORMANCE: GPU FFT-Friendly Grids — No Silent CPU Fallback

### Problem

The FDBM pipeline uses `fft_poisson` which calls gpyFFT (GPU clFFT) by default. clFFT requires grid dimensions to have **only prime factors 2, 3, 5, 7**. When a grid dimension has other prime factors (e.g., 11, 13), clFFT fails.

**Current behavior**: The code falls back to CPU FFT (`SPAMMM_AFM_CPU_FFT=1` or `fft_poisson_cpu`). This is **unacceptable for production** — CPU FFT is orders of magnitude slower than GPU FFT. This was observed during testing where `SPAMMM_AFM_CPU_FFT=1` was used as a workaround for benzene (nx=88, which has prime factor 11).

### Root cause

`make_fdbm_grid_com_zsym` already calls `round_fft_friendly` (`AFM.py:2139`) which rounds up to multiples of 8 with only prime factors 2,3,5,7. However:
1. **The Morse path** (`run_morse_coulomb_afm`) uses `setup_grid` which does NOT call `round_fft_friendly` — it uses arbitrary `n=(60,60,40)`.
2. **Some code paths** still hit non-friendly sizes (e.g., cube resampling, PTCDA `ny=176` with factor 11) and silently fall back to CPU.
3. **The `--cpu-fft` CLI flag** exists as an explicit escape hatch, but the real fix is to **always pad the grid** to FFT-friendly sizes.

### Required fix

1. **ALL grid construction must use `round_fft_friendly`** — no exceptions. This includes:
   - `setup_grid` (Morse path) — must round n to FFT-friendly
   - `make_fdbm_grid_com_zsym` (already does — ✅)
   - `setup_density_grid` / `setup_grid_world` / `setup_grid_lvec` — must round
   - Cube resampling — must round target grid

2. **NEVER silently fall back to CPU FFT**. If `fft_poisson` (GPU) fails due to grid shape:
   - **CRASH with a clear error message** stating the grid dimensions and their prime factors
   - Do NOT automatically switch to `fft_poisson_cpu`
   - The user must either fix the grid or explicitly pass `--cpu-fft`

3. **`round_fft_friendly` must be applied at grid construction time**, not at FFT call time. The grid size is chosen once; all downstream code (density projection, Poisson, Pauli overlap, ES convolution) uses that same grid.

4. **Performance impact**: CPU FFT for a 128×128×128 grid takes ~500ms vs ~5ms on GPU (100× slower). For interactive throughput (multiple molecules per second), this is fatal. Every millisecond counts — this is the project's main selling point.

### Code locations to fix

| Location | Issue | Fix |
|----------|-------|-----|
| `AFM.py:setup_grid` | Uses arbitrary `n` without `round_fft_friendly` | Round n to FFT-friendly |
| `AFM.py:setup_grid_world` | Same | Same |
| `AFM.py:setup_grid_lvec` | Same | Same |
| `AFM_utils.py:fft_poisson` | Falls back to CPU when GPU fails | Crash with clear error instead |
| `AFM.py:fft_poisson` | Same | Same |
| `run_spm.py:cmd_afm_morse` | Passes `n=(60,60,40)` to `run_morse_coulomb_afm` | Use `make_fdbm_grid_com_zsym` instead (already FFT-friendly) |
| `examples/density_comparison/optimize_basis.py:133,193` | Sets `SPAMMM_AFM_CPU_FFT=1` as workaround | Remove after grid fix; use `round_fft_friendly` |
| `run_spm.py:cmd_es_diag:524` | Sets `SPAMMM_AFM_CPU_FFT=1` | Remove after grid fix |

### Existing infrastructure

`_FDBMGpyFFT.round_fft_friendly` (`AFM.py:2139`) already exists and works:
```python
@staticmethod
def round_fft_friendly(n, multiple=8):
    """Round up to multiple of `multiple` with only prime factors 2,3,5,7 (clFFT)."""
    n = int(((int(n) + multiple - 1) // multiple) * multiple)
    while not _FDBMGpyFFT.is_fft_friendly(n):
        n += multiple
    return n
```

`is_fft_friendly` checks that all prime factors are in {2, 3, 5, 7}. This is already used in `make_fdbm_grid_com_zsym` and `ModularPipeline`. It just needs to be applied **everywhere**.

---

## Priority

**CRITICAL** — performance regression (CPU FFT fallback) and usability/correctness issue (divergent pipelines). The Morse path produces wrong-looking plots because it bypasses the standard pipeline. Users cannot compare Morse vs FDBM vs contact-surface results side-by-side because they use different grids, scan geometries, df computation, and plotting. The CPU FFT fallback silently destroys throughput — the project's main selling point.
