---
type: Reference
title: Takeways — lessons from debugging
tags: [debugging, gui, matplotlib, dftb, documentation, afm, contact-surface]
---

# Takeways — lessons from debugging SPAMMM

Short notes for future developers: non-obvious bugs we hit, how we fixed them, and
patterns worth reusing. Not a design spec — see topical docs (`doc/Topics/`) and
`doc/topical_audit.md` for architecture.

**Related topical docs:**
- [ReactionCoordinateScan.md](Topics/ReactionCoordinateScan.md)
- [AFM Contact Surface (static)](Topics/AFM/ContactSurface_Static.md)

**Blit helper:** `spammm/GUI/mpl_blit.py`

---

## Matplotlib blit with Qt (embedded GUI)

**Context:** Reaction-coordinate ESP animation (`spammm/GUI/rc_esp_view.py`) —
slider-driven heatmap + atom overlay at many frames per second.

**Reusable helper:** `spammm/GUI/mpl_blit.py` → `MplBlitManager`

### Symptoms we saw

- Heatmap and atom markers slightly **offset** from axes / colorbar
- **Double title** and faint leftover text from previous frames
- **Smears** at plot edges after scrubbing the slider

### Root causes

| Issue | Cause |
|-------|--------|
| Offset / ghost image | Background captured **before** Qt finished layout or HiDPI scaling |
| Colorbar artifacts | Used `fig.bbox` for blit while animated artists live on **main ax only** |
| Double title | Updated `ax.set_title()` every frame **and** redrew title via `draw_artist` — title already in static background |
| Stale pixels after resize | Background snapshot not refreshed when window size changed |

### Working pattern

1. Mark **only** changing artists `animated=True` (`imshow`, scatter lines, etc.).
2. Snapshot in **`draw_event`** (or after `show()` + `canvas.draw()`), using **`ax.bbox`** when colorbar is on another axes.
3. Update loop: `restore_region(bg)` → `ax.draw_artist(each animated)` → `blit(ax.bbox)` → `flush_events()`.
4. **Static** matplotlib text (title, axis labels) — set once. Put **frame index** in Qt (`QLabel`, dialog title).
5. On **`resize_event`**: full `draw()` + new snapshot (`capture_background()`).

### When not to blit

- Headless tests / `savefig` → normal `Agg` + full `draw()`.
- Whole figure changes every frame → `draw_idle()` is simpler and fast enough.
- Autoscale or limit changes every frame → blit needs a new background each time (often not worth it).

### References

- [Matplotlib blitting tutorial](https://matplotlib.org/stable/users/explain/animations/blitting.html)
- matplotlib issue #5135 — PyQt background captured before resize
- First working implementation: `RCESPBlitView` + `MplBlitManager` (2026-07)

---

## GUI vs test geometry — ASCII art must match the library path

**Context:** DFTB endpoint relax **failed in GUI** but **passed in pytest** for the same
molecule (`2Quinolone`).

### Symptom

```
SKIP: DFTB+ relax failed … SCC is NOT converged
```

GUI still showed a trajectory (rigid H-only morph); only H atoms moved on the slider.

### Root cause

Two different build paths produced **different atom counts** (26 vs 24):

| Path | How | Atoms |
|------|-----|-------|
| Tests / offline | `build_ascii_hbond_system('2Quinolone')` | 24 |
| GUI (broken) | `ASCII_EXAMPLES[name].strip()` + `generate_ascii_molecule` | 26 |

Stripping the leading newline in dimer ASCII art changed parser layout → extra H atoms →
bad DFTB SCC on u=1 endpoint.

### Fix

- GUI control script uses **`build_ascii_hbond_system`** (SSOT, same as tests).
- `load_ascii_example`: load raw art from dict, **do not** `.strip()` the whole block.
- `generate_ascii_molecule`: same pipeline as library (`hbond_length=3.0`, Jacobi before
  `resolve_hbond_pairs`).

### Takeaway

**Always align GUI molecule construction with the library helper used in tests.**
When GUI and pytest disagree, compare `natoms`, not just the molecule name.

---

## DFTB relax failures — fail loud, diagnose, clean workdirs

**Context:** `run_dftb_relax` returned `SKIP` with one line; trajectory still labeled
`pm_neb_relaxed`.

### Issues

1. Stale `charges.bin`, `OUT`, etc. in reused `debug/rc_scan/endpoint_u*` poisoned retries.
2. Non-zero exit only reported as `SKIP: DFTB+ relax failed` — no SCC tail from `OUT`.
3. `endpoints_relaxed=True` even when relax was skipped → misleading metadata.

### Fixes (`spammm/quantum/DFTB_utils.py`, `coordinate_scan.py`)

- `clean_dftb_workdir()` before each run
- `dftb_failure_summary()` — print SCC error line + tail of `OUT`/`ERR`
- Optional SP warmup before geometry opt
- `endpoints_relaxed` only if both endpoint energies are finite; else `scan_type=pm_neb_preview`

### Takeaway

For external calculators (DFTB+, LAMMPS, …): **clean work dir**, **surface log tails on
failure**, and **do not claim success in dataset metadata** when a step was skipped.

---

## Reaction-coordinate scan data model

**Context:** H-bond scan GUI + `ScanDataset` npz.

Useful conventions established in this work:

| Field | Shape | Notes |
|-------|-------|-------|
| `apos` | `[nframes, natoms, 3]` | Full-atom pm-NEB path when endpoints relaxed |
| `controls` | `[nframes, m]` | Stored; fractions derived via `mapping` in meta |
| `charges` | `[nframes, natoms]` | Mulliken from DFTB SP per frame |
| `esp_xy` | `[nframes, ny, nx]` | Optional precomputed Coulomb ESP (same KE/r as QEq) |

Mulliken: parse last block in `detailed.out` (`parse_mulliken_charges`). pm-NEB relaxed
runs an SP charge pass on each interpolated frame even when energies are not needed.

ESP animation: precompute stack once (`spammm/quantum/esp_grid.py`), blit in GUI, drive
from the **same** RC slider as the 3D view.

---

## GUI control scripts (`./run_gui.sh --script …`)

**Pattern:** `spammm/GUI/gui_scripts/*.py` with `run(window, argv)` — setup after
`window.show()`, argv after `--` (leading `--` stripped in `SPAMMM_GUI.py`).

- **`--preview`**: load cached npz from `debug/testplot_rc_scan_gui/` if present.
- **Full run**: DFTB + save cache; use for first run or after geometry/scan logic changes.
- **Offline mirror:** `spammm/GUI/gui_scripts/rc_scan_offline.py` (no Qt), same
  `build_ascii_hbond_system` path.

---

## AFM contact surface — z alignment, signs, GPU buffers, fit stability

**Context:** Quasi-2D contact surface replacing 3D `img_FF` for PP-AFM on rigid molecules
(PTCDA prototype). Separable B-spline×poly path + radial PIC (atom-centric basis).
`kernels/contact_surface.cl`, `spammm/surfaces/ContactSurface.py`, `spammm/SPM/AFM.py`.

**Visual harness:** `tests/testplot_contact_surface.py` → `debug/testplot_contact_surface/`  
**GPU L0:** `tests/SPM/test_afm_contact_surface.py` (force-stencil parity + separable scan)

### 1. Tip height vs probe height (z alignment)

**Symptom:** PP-relaxed Fz parity looked wrong even when unrelaxed E slices were OK; confusion
about which z to use in fit samples vs scan planes.

**Convention (must stay consistent everywhere):**

| Quantity | Meaning |
|----------|---------|
| `z` in fit queries / `cs_eval_*` | **Probe particle** position (same frame as `atoms_arr`) |
| `h₀(x,y)` | Local contact height (max atom z in xy neighborhood on B-spline grid) |
| `dz` (separable z-basis) | `z − h₀(x,y) − poly_z0` — **not** tip z |
| `h_tip` | Tip base height along scan (from `scan_p0[2]` + `iz·dtip`, relative to `zmax`) |
| `h_probe` | `h_tip + dpos0_z` — probe offset from tip (`afm.dpos0`) |

Fit samples must be built at **probe** z (absolute coords above molecule), matching what
`relaxStrokesTiltedContact` / `relaxStrokesTiltedPIC` evaluate during PP relaxation.

**Flat molecules (PTCDA):** all atoms at same z → `h₀` is constant → `dz ≈ z − zmax − poly_z0`.
Diagnostic: `contact_surface_scan_z_alignment.png` (raw + PP Fz vs `h_probe` at scan center).

**Takeaway:** When parity fails, plot **probe height** not tip height. Verify
`dpos0`, `scan_p0[2]`, and fit z-planes use the same absolute frame as `atoms_arr`.

### 2. Force sign and stencil adjoint (`F = −∇E`)

**Symptom:** Adding force rows to the fit made B-splines worse — looked like a sign error.

**Root cause:** Two separate issues were conflated:

1. **Correct physics:** eval kernels return `(Fx, Fy, Fz, E)` with `F = −∇E`. Force-fit
   stencils (`cs_sep_stencil_f`) implement `−∂E/∂x` etc., including chain rule through
   `h₀(x,y)` when `dz` depends on xy.
2. **Actual bug:** `F_ref` GPU upload layout (below) — not a sign mistake.

**Verification:** `test_contact_surface_force_stencil_parity` — `cs_sep_Av_f` vs analytic
`eval_separable` on random coeffs.

**Takeaway:** Run stencil parity **before** tuning force weights. Do not flip signs in the
loss until buffer layout and `h₀` chain rule are verified.

### 3. `F_ref` GPU buffer layout (horrific B-spline oscillation)

**Symptom:** Separable fit with force loss produced wild spatial oscillations; E-only fit
looked fine.

**Root cause:** `cs_Fref_buff` stores **planar** blocks `[Fx₀…Fxₙ, Fy₀…Fyₙ, Fz₀…Fzₙ]` for
`get_sub_region` per component. Upload used **interleaved** `[Fx₀,Fy₀,Fz₀, …]`.

**Fix** (`ContactSurface.py` `upload_samples`):

```python
np.concatenate([F_ref[:, 0], F_ref[:, 1], F_ref[:, 2]])
```

**Takeaway:** Any time host `(N,3)` forces go to OpenCL, confirm kernel read pattern
(component-major vs interleaved). Add a parity test when adding new force channels.

### 4. Regularization — separable vs PIC need different defaults

**Symptom:** PIC CG diverged (`|F|` growing, fit RMSE ~10² eV) while separable looked fine.

| Fit | Default `reg` | Notes |
|-----|---------------|-------|
| Separable global CG | `0` | Works for PTCDA with force loss + Boltzmann |
| Separable tiled | `1e-2` | Local patches need Tikhonov |
| PIC radial CG | **`1e-2`** | `1e-4` diverges — overlapping atom spheres + doubling modes are ill-conditioned |

PIC also has rank-deficient modes at `r→0` (all `φ_k → 1` for doubling exponents).

**Takeaway:** If matrix-free CG diverges, raise `reg` and check conditioning before blaming
the basis. PIC is **not** separable — copy hyperparams only after testing.

### 5. Sample weighting — Boltzmann and force RMS equalization

**Separable fit (works well):**

- **Boltzmann weights** `w = exp(−(E−E_min)/T)` on sample rows — emphasizes attractive well;
  auto `T` from data spread.
- **Force loss** (`fit_force_weight=1`): extra rows for Fx, Fy, Fz with **RMS equalization**
  so eV and eV/Å contribute comparably (`_loss_row_weights` in `ContactSurface.py`).
- Weighted normal equations: only `Atv` multiplies by `w`; `Av` stays unweighted (same
  pattern as `cs_sep_Atv_w`).

**PIC fit (different):**

- Boltzmann weights **hurt** close-contact parity (fit chases deep well, misses Pauli wall).
- Current policy: log Boltzmann stats but fit **unweighted** CG (`fit_pic_contact_surface`).
- `cs_pic_Atv_w` exists for future tuned weighting; not used in production fit yet.

**Takeaway:** Boltzmann is not universal — good for separable global fit, bad for PIC
atom-sum basis near contact. Re-check weighting whenever the basis or fit region changes.

### 6. Stale GPU state when reusing `ContactSurfaceCL`

**Symptom:** PIC fit OK in isolation, diverged when run **after** separable fit in the same
`testplot_contact_surface.py` session (`|F| ~ 10²` at iter 0).

**Root cause:** Shared lazy helper `AFMulator._cs_fit_helper()` keeps large buffers
(`n_coeff` up to ~48k). PIC uses `nc ≈ nat×nmodes` (~190). Initial PIC CG called `Atv` into
`cs_AtAp_buff` **without** `cs_zero` first — leftover separable `AtAp` garbage in the first
190 slots poisoned the residual.

**Fix:** `fit_pic_cg` now zeros `cs_AtAp_buff` before the initial `Atv` (mirrors
`fit_separable_cg`).

**Takeaway:** Any `atomic_add` accumulation buffer must be zeroed before each use. When
reusing a GPU helper across representations, assume buffers are **dirty** unless zeroed.

### 7. Other OpenCL / kernel landmines

| Issue | Symptom | Fix |
|-------|---------|-----|
| Brute reference `barrier()` | Wrong / nondeterministic brute E on GPU | Padded work-items must not `return` before group `barrier()` (`contact_surface.cl`) |
| Stencil scratch size | Wrong fit / garbage stencils for `nz=6` | `ic[]` / `w[]` local arrays ≥ 128 entries (`cs_sep_stencil*`) |
| `poly_z0` vs fit z range | Cutoff kink at wrong height | Use `poly_z0=1 Å` with adaptive z sampling `z∈[1,6] Å` above `zmax` |
| B-spline bbox edge | Clipped samples at margin | Fit margin vs scan margin are separate knobs (`MARGIN`, `SCAN_MARGIN`) |

### 8. PIC atom selection and PP relaxation

**Diagnostics:** `plot_pic_atom_selection` → `contact_surface_pic_atoms.png` — which atoms
carry radial modes, bucket grid, Rc circles, index list (`select_contact_atoms`).

**PP path:** `run_scan_pic` + `relaxStrokesTiltedPIC` — same scan geometry as
`run_scan_contact`; shared `cs_eval_pic_fe_at` for eval and relaxation.

**Current PTCDA quality (order of magnitude, not tuned):**

| Path | Close E @ z+1.2 Å | PP relaxed mean Fz RMSE |
|------|-------------------|-------------------------|
| Separable + force loss | ~8 meV | ~14 meV/Å |
| PIC (`reg=1e-2`) | ~32 meV | ~20 meV/Å |

**Open work:** optimize basis (`m_start`, `nz`, `Rc`), atom pruning, PIC force loss,
fitting region — “not great, not terrible” until then.

**Takeaway:** Trust PP parity only after unrelaxed E **and** z-alignment plots look sane.
PIC needs its own fit knobs; do not assume separable recipes transfer.

### References

- Design spec: `doc/Topics/AFM/ContactSurface_Static.md`
- Module index: `spammm/surfaces/README.md`
- Phase plots: `debug/testplot_contact_surface/`, `debug/testplot_afm_contact_surface/`

---

## Adding to this document

When you spend time on a non-obvious bug or pattern:

1. One short **symptom** paragraph
2. **Root cause** (table or bullet)
3. **Fix / pattern** (file pointers)
4. **Takeaway** one-liner for grep-ability

Keep entries self-contained so they help someone who hits the same class of problem in
a different module.
