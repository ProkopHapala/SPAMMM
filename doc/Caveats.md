---
type: Caveats
title: SPAMMM scientific / numerical caveats
tags: [caveats, FDBM, ES, grids, NA, multipoles, anisotropic, contact-surface, AFM]
timestamp: 2026-07-24
---

# Caveats (global)

Recurring traps that look like “bugs in Poisson / tip / plot” but are usually **Δρ construction** or **grid convention**. Read this before “fixing” AFM electrostatic asymmetry.

---

## 1. Anisotropic DFT cubes — never collapse `(sx,sy,sz)` to a mean

**Primary cause of the huge pentacene V(N−Gauss) dipole (2026-07-24).**

pySCF cubes often have **~0.1–0.2%** axis anisotropy from cell rounding. That looks “isotropic enough” but **is not**:

- Density samples live at `origin + (i·sx, j·sy, k·sz)`.
- Building Gauss/compact NA with `step = mean(sx,sy,sz)` places nuclei on a **warped** lattice relative to ρ.
- Error grows with grid index (≲0.2 voxel across pentacene) → fake Δρ dipole ~**1.9 e·Å**.

| Δρ (pentacene native) | \|p_xy\| | V mX @ z+1 |
|-----------------------|----------|------------|
| N − NA_cube | ~0.01 | ~0.035 |
| N − Gauss **mean step** (old bug) | ~1.9 | ~0.80 |
| N − Gauss **true (sx,sy,sz)** | ~0.28 | ~0.42 |
| N − clamp **element-invariant** compact | ~0.018 | ~0.033 |

**Fix:** `get_density_from_cube` keeps `step` as `(3,)`; `make_gaussian_rho_na` / `fft_poisson_cpu` / GridsOCL already accept `(3,)`.

**FDBM pipeline readiness:**

| Stage | Anisotropic? | Notes |
|-------|--------------|-------|
| Cube load / native NA / native Poisson | **Must keep (3,)** | Fixed 2026-07-24 |
| `GridsOCL.project_density` | **Yes** — `(3,)` `step_s` / `step_d` | Already |
| FDBM dest (`make_fdbm_grid_com_zsym`, DFTB) | **Isotropic by design** | OK — project cube→FDBM |
| GPU `AFMulator` / fused ES | Scalar `step` | Fine on isotropic dest |

Do **not** accept mild aniso and then silently use the mean (old bug).

---

## 2. All-electron cores on coarse cubes → fake Δρ dipole

After §1, raw `ρ_N − Gauss` still leaves \|p\|~0.28 on pentacene. That is **not** “wrong σ”. At ~0.15 bohr, C/N/O 1s cusps are under-resolved; equivalent atoms get different sampled peaks / `Q_rem_i` purely from sub-voxel phase.

**Production recipe (SSOT):** `delta_rho_clamp_compact_na(..., q_na_mode='element_mean')`

```text
ρ_ps   = soft_clamp(ρ_SCF)                 # kill aliased cusps
Qrem_Z = mean_i(Q_rem_i for atoms of Z)    # never use noisy Q_rem_i as q_i
q_Z    = Z − Qrem_Z  (+ charge close)      # equal elements → equal q
Δρ_ES  = ρ_ps − Σ_i compact(q_{Z_i}, R_i)
```

- Legacy `q_na_mode='per_atom'` (`q_i=Z_i−Q_rem_i`) **recreates** the alias dipole — do not use for production.
- `rho_NA.cube` is an **optional diagnostic** (cancels the same alias by construction), **not** a required input.
- Manual `strip_monopole_dipole` DEFAULT **OFF** (symptom mask).

Pentacene check: element-mean \|p_xy\| ~0.018 ≈ NA_cube control ~0.014 (vs ~0.28 legacy).

**Canonical plot:** `plot_cube_delta_rho_na_origin_diag` →  
`debug/fdbm_fukui_panel_flat/<mol>/es_diag/dipole_origin_bisect.png`  
CLI: `python run_spm.py es-diag --molecule pentacene`

**Full write-up:** [Reports/Cube_ES_DeltaRho_NA_dipole.md](Reports/Cube_ES_DeltaRho_NA_dipole.md) · handoff: [Reports/Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md](Reports/Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md)

---

## 3. Corner/node vs center voxel sampling

| Path | Sample positions |
|------|------------------|
| Cubes, `make_gaussian_rho_na`, `grid_moments` | **nodes** `origin + i·h` |
| Low-level `GridsOCL.project_density` kernel | **centers** `origin + (i+½)·h` |

`project_density_to_grid(..., src_convention='nodes')` (default) shifts `origin_src − ½ step` before calling the kernel. Use `'centers'` only for true cell-centered fields. Mixing conventions shifts moments by `q·step/2` and can change V slope direction.

ρ_NA **must** be atom-centered (same nuclei as the cube header). Mirror metrics must use **fractional** physical centers (`mirror_asymmetry_2d`), not integer pixel flips.

---

## 4. Scipy *sample* vs GridsOCL *project* for density transfer

| Method | Code | Conserves ∫ρ, p? | Use for Δρ / charge? |
|--------|------|------------------|----------------------|
| scipy sample | `resample_field_to_grid` / `map_coordinates` | **No** | **Forbidden** |
| trilinear scatter | `project_density_to_grid` → `grids.cl` | **Yes** (in-bounds) | **Required** |

---

## 6. Contact-surface 2.5D vs GridFF (classical Morse AFM)

Quasi-2D contact-sep is meant to **approximate** Morse(+Coulomb), not invent sharper physics. Recurring traps:

| Trap | Symptom | Fix / rule |
|------|---------|------------|
| `h₀ = max atom z` | Soft / long-ranged E(z); “too close” images | `h0_mode='spheres'` — ray vs `R=scale×R0` spheres |
| `h0_R_scale=1.0` | Repulsion plateaus at well | Default **0.75** — clamp in hard wall |
| `bspl_dx` / `scan_dx` ≪ 1 Å | Razor atom pins; against coarse-gain design | Defaults **1.0 / 0.5 Å** (atom-scale nodes/pixels) |
| Trusting contact XY vs GridFF without profiles | Looks “better resolved” | Always check E/Fz(z) vs **brute**; GridFF ≈ brute for classical FF |
| **`tip_R=0.0, tip_E=1.0`** (non-physical point tip) | Morse minimum at sample atom vdW radius (~1.9 Å for C) instead of correct tip-atom contact (~3.4 Å); fit RMSE 24× worse; E(z) curves show attraction at unphysical distances | **ALWAYS use real tip params**: `assign_params(params_path=PARAMS)` → default `tip_R=1.452, tip_E=6.8e-4` (CO tip). R0 = tip_R + RvdW ≈ 3.2–3.4 Å. Never override with `tip_R=0` unless explicitly testing point-tip physics. |
| **Fit z-range too narrow** | Contact surface = 0 above fit range → hard cutoff in E(z)/Fz(z) | Fit z-range must cover full scan: `fit_z_adaptive=(0.05, 8.0, 0.1, 1.0)` for real tip |
| **E(z) ylim not normalized** | Can't see well + repulsion onset together | `ylim=(E_min*1.2, -2*E_min)` — see skill `afm-plotting-alignment` |
| **`tip_R=0.0, tip_E=1.0`** (non-physical point tip) | Morse minimum at sample atom vdW radius (~1.9 Å for C) instead of correct tip-atom contact (~3.4 Å); fit RMSE 24× worse; E(z) curves show attraction at unphysical distances | **ALWAYS use real tip params**: `assign_params(params_path=PARAMS)` → default `tip_R=1.452, tip_E=6.8e-4` (CO tip). R0 = tip_R + RvdW ≈ 3.2–3.4 Å. Never override with `tip_R=0` unless explicitly testing point-tip physics. |
| **Fit z-range too narrow** | Contact surface = 0 above fit range → hard cutoff in E(z)/Fz(z) | Fit z-range must cover full scan: `fit_z_adaptive=(0.05, 8.0, 0.1, 1.0)` for real tip |
| **E(z) ylim not normalized** | Can't see well + repulsion onset together | `ylim=(E_min*1.2, -2*E_min)` — see skill `afm-plotting-alignment` |

**Report:** [Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md](Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md) · helicene session [Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md](Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md)

---

## 7. DFTB+ Slater-Koster (SK) path configuration — the #1 clone-to-new-machine bug

**Symptom:** Main molecule SCF works (e.g. `3ob-3-1`), but CO tip computation crashes with `SK file ... not found` for `O-O.skf` in `mio-1-1`. Or: everything worked on machine A, broken after `git clone` on machine B.

**Root cause:** DFTB+ needs two SK parameter sets — `3ob-3-1` for the sample molecule and `mio-1-1` for the CO tip (`compute_co_tip.py:157`, hardcoded). These sets live in different subdirectory layouts depending on how they were downloaded:

```
~/SIMULATIONS/dftbplus/slakos/
├── 3ob-3-1/              ← direct child (some installs)
├── library/
│   └── 3ob-3-1/          ← nested under library/ (other installs)
├── mio/
│   └── mio-1-1/          ← nested under mio/ (always, from dftb.org)
└── ...
```

`DFTB_SK_PATH` must point to the **parent** (`.../slakos/`), not a subdirectory like `.../slakos/library/`. If it points to `library/`, only `3ob-3-1` is found; `mio-1-1` (under `mio/`) is invisible.

**Resolution priority** (highest → lowest):
1. **`firecore_config.json`** (repo root, gitignored) — explicit `sk_path` per basis set; machine-specific
2. **`DFTB_SK_PATH` env var** — parent dir; code searches `sk_path/{basis}`, `sk_path/mio/{basis}`, `sk_path/library/{basis}`, and also the **parent** of `sk_path` (handles wrong `library/` setting)
3. **Bundled WFC files** — `spammm/quantum/DFTB/data/wfc.{basis}.hsd` (always in repo, no SK files)

**Setup checklist for a new machine:**
1. Download SK sets: `3ob-3-1` (has O, Br, P, S…) and `mio-1-1` (has O, N, C, H…)
   - From https://dftb.org/parameters/download.html
   - `mio-1-1` typically extracts as `mio/mio-1-1/` (nested)
   - `3ob-3-1` may extract as `3ob-3-1/` or `library/3ob-3-1/` depending on source
2. Set `DFTB_SK_PATH` to the **parent** containing both:
   ```bash
   export DFTB_SK_PATH="$HOME/SIMULATIONS/dftbplus/slakos/"
   ```
3. **Verify** both sets resolve:
   ```bash
   python3 -c "from spammm.quantum.DFTB_utils import SK_PATHS; print(SK_PATHS)"
   # Should show both mio-1-1 and 3ob-3-1 with real paths
   ```
4. If paths still don't resolve, create `firecore_config.json` (see template below)
5. **Verify O-O.skf exists** in the `mio-1-1` set (CO tip needs it)

**`firecore_config.json` template** (repo root, gitignored):
```json
{
  "paths": {
    "dftb_sk_path": "/home/USER/SIMULATIONS/dftbplus/slakos/"
  },
  "dftb": {
    "basis_sets": {
      "mio-1-1": {
        "sk_path": "/home/USER/SIMULATIONS/dftbplus/slakos/mio/mio-1-1/",
        "wfc_path": "@REPO_ROOT/spammm/quantum/DFTB/data/wfc.mio-1-1.hsd"
      },
      "3ob-3-1": {
        "sk_path": "/home/USER/SIMULATIONS/dftbplus/slakos/library/3ob-3-1/",
        "wfc_path": "@REPO_ROOT/spammm/quantum/DFTB/data/wfc.3ob-3-1.hsd"
      }
    }
  }
}
```

`@REPO_ROOT` is resolved to the repo directory at load time.

**Code entry points:**

| Function | File | Role |
|----------|------|------|
| `get_dftb_sk_path()` | `spammm/config_utils.py` | Smart search: config → env var + subdir patterns → parent dir fallback |
| `_get_sk_paths()` | `spammm/quantum/DFTB_utils.py` | Builds `SK_PATHS` dict at import time; uses `get_dftb_sk_path()` |
| `_check_sk_path()` | `spammm/quantum/DFTB_utils.py` | Validates `DFTB_SK_PATH` at import; searches 2 levels deep |
| `get_density_from_dftb_dense()` | `spammm/SPM/AFM_utils.py` | SK fallback for sample molecule SCF |
| `_stage1_scf_dftb()` | `spammm/SPM/ModularPipeline.py` | SK fallback for pipeline Stage 1 |
| `compute_co_tip.py:157` | `spammm/SPM/` | **Hardcoded** `basis_name='mio-1-1'` for CO tip |

**Why `mio-1-1` for the tip?** The CO tip (C+O, 10 electrons) is a small diatomic — `mio-1-1` is the standard set for organic molecules with C, H, N, O. `3ob-3-1` would also work (has O), but `mio-1-1` is the conventional choice for tip density.

---

## 8. Related but secondary

- Tip innocent when tip XY mirror ~1e−7 and peak at (0,0,0) after roll.
- z-box asymmetry (extra vacuum on +z only) can bias vertical fields.
- OpenCL: NVIDIA first — `.cursor/rules/opencl-nvidia-gpu.mdc`.

---

## Index of code entry points

| Topic | Location |
|-------|----------|
| NA-origin diagnostic plot | `spammm/SPM/AFM_utils.py` → `plot_cube_delta_rho_na_origin_diag` |
| ES chain plots / mirror metrics | `plot_cube_es_chain_diag`, `mirror_asymmetry_2d` |
| Cube load + synthetic NA | `get_density_from_cube`, `make_gaussian_rho_na` |
| Clamp→compact (element-mean) | `delta_rho_clamp_compact_na` (`q_na_mode='element_mean'`) |
| Cube → FDBM | `allelectron_cube_to_fdbm_grid` |
| Project / moments | `spammm/utils/GridsOCL.py` |
| CLI | `run_spm.py es-diag` → `tests/SPM/testplot_fdbm_relax.py` |
| Contact-surface fit / scan | `spammm/SPM/AFM.py` → `fit_contact_surface`, `run_scan_contact` |
| Assembly AFM compare | `run_assembly_afm.py --compare-dir` |

---

## 9. Rigid Assembly — graph↔assembly synchronization

**Context:** `RigidAssemblyExtension._ensure_backend_matched` rebuilds the display
graph from the assembly's world atoms when the atom count differs from the editor
graph. This is needed for "From file" builds (e.g. 4×PTCDA=104 atoms vs 0 in editor).

**Trap (2026-08-03):** When loading a dimer from the editor and splitting via
connected components (`graph_to_rigid_fragments`), the fragments have atoms in
**BFS order** within each component, while the editor graph has atoms in **XYZ file
order**. Both have the same count (e.g. 30), so the old count-only check passed and
`update_positions_from_array` assigned assembly-ordered positions to graph-ordered
atoms — **scrambling bonds and atom colors**.

**Fix:** `_ensure_backend_matched` now also checks that the **enames sequences match**.
If they differ (same count, different order), the graph is rebuilt from the assembly's
atom order with correct bonds from `ra_bonds0`.

**Rule:** Atom count equality is necessary but **not sufficient** for graph-assembly
synchronization. The enames sequence (or a stable atom-ID mapping) must also match.

See [Takeways.md](Takeways.md) → "Graph rebuild enames check".

---

## 10. Rigid Assembly — FAF for editor builds

**Context:** The "From editor" source in the RA panel builds rigid bodies from
`AtomicGraph.find_connected_components`. Previously, FAF was hardcoded off for
editor builds (`faf_enabled = source != 'From editor'`).

**Fix (2026-08-03):** FAF is now enabled for editor builds. The fit is computed on
the first fragment's atoms with `mol_name='editor_frag0'` for cache filename.

**Caveat:** `load_or_fit_faf(mol, mol_name=None)` crashes with
`AttributeError: 'NoneType' object has no attribute 'lower'` because the cache
path uses `mol_name.lower()`. Always pass a valid string `mol_name`.

---

## 11. Rigid Assembly — processEvents re-entrancy in vispy mouse callbacks

**Context:** `SPAMMM_GUI.refresh_view()` and `RigidAssemblyExtension._status()` end with
`QtWidgets.QApplication.processEvents()`. The RA drag handler calls `on_move` →
`_sync_display` → `refresh_view()` during mouse move events.

**Trap (2026-08-03):** `processEvents()` processes pending Qt events synchronously. When
called from within a vispy `mouse_move` callback, it re-enters `mouse_move` on the same
vispy EventEmitter → `_emitting > 1` → `RuntimeError: EventEmitter loop detected!`.

This is a **pre-existing latent bug** — `processEvents()` was in `refresh_view()` before
the RA drag feature. The drag mode's `on_move` → `refresh_view` path simply exposed it.

**Fix:** `_in_mouse_callback` flag on the window, set `True` during
`on_mouse_press`/`on_mouse_move`/`on_mouse_release` (via try/finally).
`refresh_view()` and `_status()` skip `processEvents()` when flag is set.

**Rule:** `processEvents()` must never be called from within a Qt/vispy event callback.
Use a re-entrancy guard flag instead. `canvas.update()` alone schedules the repaint.

See [Takeways.md](Takeways.md) → "processEvents re-entrancy in vispy mouse callbacks".

---

## 12. Rigid Assembly — probe map parameters (SUPERSEDED — see §12b)

> **Superseded 2026-08-03** by [RigidAssembly_Demo_MapMode_Consolidation.md](Tasks/RigidAssembly_Demo_MapMode_Consolidation.md)
> §0: the map now uses the exact PairFF parameters from `rbd.pairff_params_host` — no
> display-only He/Hs substitution. The contrast issue is solved by the nuclear exclusion
> mask (§12b) and the symmetric `|Emin|` color scale rule. This entry is retained for
> provenance.

**Original context (F2, pre-consolidation):** The combined PairFF+FAF probe map in
`RigidAssemblyExtension` used `He=-1.0, Hs=0.0, w=0.7` for visualization, while the
assembly's PairFF dynamics used `He=-0.1, Hs=1.0`. "Fixing" the map's He/Hs to match the
assembly's PairFF values (review finding F2) made the H+ probe map lose its attractive
blue minima at electron pairs. The temporary fix was to revert to display-only
`He=-1.0, Hs=0.0`. The permanent fix (consolidation) uses honest parameters + nuclear
exclusion mask + symmetric `|Emin|` color scale.

See [Takeways.md](Takeways.md) → "Diagnostic visualization parameters ≠ simulation
parameters (SUPERSEDED)".

---

## 12b. Rigid Assembly — probe map color scale and nuclear exclusion (USER MANDATED)

**Context:** The combined PairFF+FAF probe map has infinitely deep attractive wells at
real-atom nuclei (compact-exp + damped Coulomb). These dominate `|Emin|` and wash out
the chemically meaningful attractive basins and FAF corrugation.

**Rule 1 — Color scale:** `vmin = Emin`, `vmax = |Emin|` (symmetric), where
`Emin = min(E_for_lim)` and `E_for_lim = E[~exclude_mask]`. The color-limit spin resets
to 0 (Auto) on every recompute so the scale is always freshly derived from the current
data. Never use `np.percentile` — use the actual minimum. See
`.devin/skills/centralized-plotting/SKILL.md` §"Color Scale Rule".

**Rule 2 — Nuclear exclusion:** `nuclear_exclusion_mask(xs, ys, z_probe, static_apos,
static_types, r=1.0)` returns a boolean mask True within 1 Å of any real atom. The mask
is used **only** for `vmin/vmax` estimation. The map itself is fully finite and displayed
everywhere — **no NaN holes**. NaN holes were tried and rejected ("disturbing white
areas"). See `.devin/skills/centralized-plotting/SKILL.md` §"Nuclear exclusion".

**Trap:** Do not set pixels near nuclei to NaN in the GPU kernel or CPU reference. The
map must be physically complete everywhere. Only the color scale derivation excludes the
singularities.

See [Takeways.md](Takeways.md) → "Nuclear exclusion: mask for color scale, not NaN holes".

---

## 13. Rigid Assembly — VisPy scene detection in dual GUI/test code

**Context:** RA overlay functions (`_recompute_ra_combined_map`, `_update_anchor_visuals`,
etc.) need to detect whether `window.scene` is a real VisPy scene (create visuals) or a
test mock (skip).

**Trap (2026-08-03):** `isinstance(window.scene, vispy.scene.Node)` fails for real scenes
because `AtomScene` doesn't subclass `Node` in the expected way. Using it as a guard
caused the combined map overlay to disappear in the real GUI.

**Rule:** Use `hasattr(window.scene, 'view')` — real VisPy scenes have a `view`
attribute (the `SceneCanvas.view`), test mocks don't. Duck-typing over `isinstance`
for vispy class hierarchy checks.

See [Takeways.md](Takeways.md) → "VisPy scene detection — use `hasattr(view)`, not
`isinstance(Node)`".

---

## 14. Rigid Assembly — per-pack PLQH by molecular identity, not atom count

**Context:** Factorized FAF uses per-atom PLQH (Pauli/London/Charge/Q) arrays from cached
fits. For mixed-species assemblies, each pack needs its own PLQH.

**Trap (2026-08-03):** `_folded_plqh_all_sites` decided whether to reuse the first fit's
`atom_plqh` solely by atom count. Two chemically different molecules with equal
real-atom counts received the same PLQH array — wrong FAF forces. The `editor_frag0`
cache name also collided across different editor fragments.

**Rule:** Atom count is not molecular identity. Build PLQH per-pack from runtime
`REQ_base`. Verify molecular identity for cache loads (not just cache name).

See [Takeways.md](Takeways.md) → "Per-pack PLQH by molecular identity, not atom count".

---

## 15. Gaussian `.cube` format — units, strides, and mol_shift z-component

**Recurring trap (invPPAFM cube export, 2026-08-14).** Writing `.cube` files for
V_PP / rho_atom / Δρ and finding the data is scrambled, atoms are misaligned, or
volumetric features float above the molecule. There are **three** independent
gotchas that each produce subtly wrong visualizations.

### 15a. Units — Bohr, not Angstrom

The Gaussian cube spec requires **all lengths in Bohr** (1 Bohr = 0.529177 Å).
Writing Angstrom values makes every standard reader (VMD, Avogadro, ASE, Ovito)
interpret the grid as ~1.89× too small — bonds look compressed, density appears
contracted.

```
BOHR_PER_ANG = 1.0 / 0.529177210903
origin_bohr  = origin_ang  * BOHR_PER_ANG
step_bohr    = step_ang    * BOHR_PER_ANG
atoms_bohr   = atoms_ang   * BOHR_PER_ANG
```

**Do not** rely on ASE's `write_cube` to handle this — its unit handling has
changed across versions and can silently write Angstroms. The custom `save_cube`
in `invPPAFM/tests/testplot_ml_maps.py` converts explicitly.

### 15b. Data ordering — z-fastest (C-order of `(nx, ny, nz)`)

The cube spec stores data with **z varying fastest, then y, then x**. This is
exactly the C-order flattening of a `(nx, ny, nz)` array:

```python
data_xyz = data_zyx.transpose(2, 1, 0)   # (Nz,Ny,Nx) -> (Nx,Ny,Nz)
data_flat = data_xyz.reshape(-1)          # C-order: z varies fastest
```

**Common mistakes:**
- `data_zyx.reshape(-1)` directly → x varies fastest → **scrambled** data.
- `np.flatten(order='F')` → x varies fastest → **scrambled** data.
- Transposing to `(nx,ny,nz)` but then using `order='F'` → **scrambled** data.

The correct chain is: **transpose to `(nx,ny,nz)`, then C-order flatten**.

### 15c. mol_shift — all 3 components, including z

**This was the root cause of V_PP appearing ~2 Å above the molecule.**

`AFMulator.setup_grid(shift_atoms=True)` shifts atoms to kernel-space:
```python
p0_raw    = [mn_x - margin, mn_y - margin, mn_z - margin/2]
mol_shift = -p0_raw = [margin - mn_x, margin - mn_y, margin/2 - mn_z]
atoms_kernel = atoms_world + mol_shift   # ALL 3 components
```

After the shift, the grid origin `p0 = (0,0,0)` and atoms sit inside the grid.
The FF data (V_PP, F_total) lives on this kernel-space grid.

**The trap:** when writing atoms to the cube, it is tempting to apply only the
`(x, y)` shift (because the scan grid is described in xy terms) and forget the
z-component. For a flat molecule (mn_z ≈ 0), `mol_shift[2] = margin/2 ≈ 2.0 Å`.
Omitting it places atoms **2 Å too low** relative to the grid, so V_PP peaks
(which are correctly placed on the kernel-space grid) appear to float ~2 Å
above the atoms.

**Fix:** Always apply all 3 components:
```python
atoms_kernel = atoms_world.copy()
atoms_kernel[:, 0] += mol_shift[0]   # x
atoms_kernel[:, 1] += mol_shift[1]   # y
atoms_kernel[:, 2] += mol_shift[2]   # z  ← THIS WAS MISSING
```

`generate_from_xyz` now returns `mol_shift` (3-component) and `mol_z` in the
sample dict so callers don't have to recompute it.

### 15d. Anisotropic step — keep `(dx, dy, dz)` separate

rho_atom grids use a different z-step than xy (e.g. `dx=dy=0.1 Å, dz=0.2 Å`).
The cube header supports anisotropic strides via the diagonal vectors:
```
nx  dx  0   0
ny  0   dy  0
nz  0   0   dz
```
**Do not** collapse to a scalar `step = mean(dx,dy,dz)` — this warps the z-axis
and misplaces features (same class of bug as §1 for DFT cubes).

### 15e. Verification recipe

After writing a cube, verify alignment with an independent reader:
```python
# 1. Bond lengths should be ~1.4 Å for C-C aromatic, ~1.1 Å for C-H
# 2. V_PP max should be <0.3 Å from nearest atom (was 1.8 Å before z-fix)
# 3. rho_atom should be positive at every atom position
# 4. Atom z-range should match the grid z-range (not offset by ~2 Å)
```

**Code:** `save_cube` in `invPPAFM/tests/testplot_ml_maps.py`  
**Verification:** independent `read_cube` reader (same file, inline in `main()`)

---

## 16. OpenCL `read_imagef` normalized coordinates — half-texel offset

**Root cause of GridFF lateral registration error (R2.1, 2026-08-09).**

OpenCL `read_imagef` with `CLK_NORMALIZED_COORDS_TRUE` maps normalized coordinate `u∈[0,1]` to texel **edges**, not centers. For an image of size `n`, texel center `i` is at `u = (i+0.5)/n`, not `u = i/n`.

The `dinvA/B/C` inverse transform matrices in `AFM.py` compute `coord = i/n` at voxel `i`, which is the **edge** between texels `i-1` and `i`. This causes a systematic **half-voxel shift** in every GridFF interpolation, scaling with grid spacing (`0.5·dx_grid ≈ 0.1 Å` for benzene).

| Variant | Synthetic RMSE | Real benzene Fz RMSE | Registration shift |
|---------|---------------|---------------------|--------------------|
| `current` (`i/n`) | 0.5 per axis | 1.17e-01 | ~0.5·dx_grid |
| `plus_05` (`(i+0.5)/n`) | **0** | **6.9e-03** (17× better) | **0** |

**Fix (NOT yet applied to production):** Add `+0.5/n` to each diagonal element of `dinvA/B/C`. Five constructors in `AFM.py` are affected. Contact-surface kernels do **not** use `interpFE` and are unaffected.

**Rule:** When converting voxel index to OpenCL normalized coordinate, use `(i + 0.5) / n`, never `i / n`. This is the same class of bug as §3 (corner/node vs center voxel sampling) but for the GPU sampler path.

See [ContactSurface_Parity_InvPPAFM_Benzene.md](Tasks/ContactSurface_Parity_InvPPAFM_Benzene.md) §R2.1.

---

## 17. Contact-surface clamp at s=0 — constant E/Fz below contact height

**Primary source of contact model error (R2.2, 2026-08-09).**

The separable contact-surface model clamps the local height coordinate `s = z − h₀(x,y) − poly_z0` to `s ≥ 0` before evaluating the polynomial basis. When `s < 0` (probe below contact height), the basis evaluates at `s = 0`, producing **one constant E and Fz per (x,y) site** — regardless of how far below contact the probe is.

Brute-force Morse continues to vary strongly in this region (Fz ranges 0.07–2.92 eV/Å for benzene). The clamp therefore produces:

| Metric | s<0 (clamp) | s∈[0.05,0.25) (fit region) |
|--------|-------------|---------------------------|
| Fz RMSE | 0.54 eV/Å | 0.025 eV/Å |
| Fz Pearson | 0.18 | 0.73 |
| Sign disagreements | 32 | 0 |
| % of total Fz² error | **99.15%** | 0.23% |

**Secondary effect — h0 interpolation error:** B-spline h₀ with 1.0 Å spacing has RMSE=0.117 Å vs analytic sphere formula, with **systematic negative bias** (interp lower than analytic). This pushes 12.5% of points that should be at s≥0 into false s<0 clamp, where they receive constant (wrong) E/Fz. Error is concentrated at sphere seams where curvature changes sharply.

| h0 error source | RMSE | Max | Effect |
|----------------|------|-----|--------|
| B-spline vs analytic (all points) | 0.117 Å | 0.316 Å | 278/2223 clamp crossings |
| At seam dist [0.1,0.2) | 0.135 Å | 0.316 Å | 15% false clamp rate |
| At seam dist [0.2,0.5) | 0.165 Å | 0.305 Å | 13% false clamp rate |

**Rules:**
1. The clamp at s=0 is a **design limitation**, not a bug — the model has no basis for sub-contact physics. But its impact on PP relaxation must be understood: any probe trajectory that dips below h₀ receives a constant force plateau, which can trap or deflect the probe.
2. h₀ B-spline spacing (`bspl_dx`) must be fine enough to resolve sphere seam curvature. At 1.0 Å, seam errors dominate. Consider finer spacing (0.5 Å) or analytic h₀ evaluation for seam-adjacent points.
3. Always stratify contact-vs-brute error by local s before declaring model quality. Aggregate RMSE hides the fact that 99% of error is in the clamp region.

See [ContactSurface_Parity_InvPPAFM_Benzene.md](Tasks/ContactSurface_Parity_InvPPAFM_Benzene.md) §R2.2.

---

## 15. Gaussian `.cube` format — units, strides, and mol_shift z-component

**Recurring trap (invPPAFM cube export, 2026-08-14).** Writing `.cube` files for
V_PP / rho_atom / Δρ and finding the data is scrambled, atoms are misaligned, or
volumetric features float above the molecule. There are **three** independent
gotchas that each produce subtly wrong visualizations.

### 15a. Units — Bohr, not Angstrom

The Gaussian cube spec requires **all lengths in Bohr** (1 Bohr = 0.529177 Å).
Writing Angstrom values makes every standard reader (VMD, Avogadro, ASE, Ovito)
interpret the grid as ~1.89× too small — bonds look compressed, density appears
contracted.

```
BOHR_PER_ANG = 1.0 / 0.529177210903
origin_bohr  = origin_ang  * BOHR_PER_ANG
step_bohr    = step_ang    * BOHR_PER_ANG
atoms_bohr   = atoms_ang   * BOHR_PER_ANG
```

**Do not** rely on ASE's `write_cube` to handle this — its unit handling has
changed across versions and can silently write Angstroms. The custom `save_cube`
in `invPPAFM/tests/testplot_ml_maps.py` converts explicitly.

### 15b. Data ordering — z-fastest (C-order of `(nx, ny, nz)`)

The cube spec stores data with **z varying fastest, then y, then x**. This is
exactly the C-order flattening of a `(nx, ny, nz)` array:

```python
data_xyz = data_zyx.transpose(2, 1, 0)   # (Nz,Ny,Nx) -> (Nx,Ny,Nz)
data_flat = data_xyz.reshape(-1)          # C-order: z varies fastest
```

**Common mistakes:**
- `data_zyx.reshape(-1)` directly → x varies fastest → **scrambled** data.
- `np.flatten(order='F')` → x varies fastest → **scrambled** data.
- Transposing to `(nx,ny,nz)` but then using `order='F'` → **scrambled** data.

The correct chain is: **transpose to `(nx,ny,nz)`, then C-order flatten**.

### 15c. mol_shift — all 3 components, including z

**This was the root cause of V_PP appearing ~2 Å above the molecule.**

`AFMulator.setup_grid(shift_atoms=True)` shifts atoms to kernel-space:
```python
p0_raw    = [mn_x - margin, mn_y - margin, mn_z - margin/2]
mol_shift = -p0_raw = [margin - mn_x, margin - mn_y, margin/2 - mn_z]
atoms_kernel = atoms_world + mol_shift   # ALL 3 components
```

After the shift, the grid origin `p0 = (0,0,0)` and atoms sit inside the grid.
The FF data (V_PP, F_total) lives on this kernel-space grid.

**The trap:** when writing atoms to the cube, it is tempting to apply only the
`(x, y)` shift (because the scan grid is described in xy terms) and forget the
z-component. For a flat molecule (mn_z ≈ 0), `mol_shift[2] = margin/2 ≈ 2.0 Å`.
Omitting it places atoms **2 Å too low** relative to the grid, so V_PP peaks
(which are correctly placed on the kernel-space grid) appear to float ~2 Å
above the atoms.

**Fix:** Always apply all 3 components:
```python
atoms_kernel = atoms_world.copy()
atoms_kernel[:, 0] += mol_shift[0]   # x
atoms_kernel[:, 1] += mol_shift[1]   # y
atoms_kernel[:, 2] += mol_shift[2]   # z  ← THIS WAS MISSING
```

`generate_from_xyz` now returns `mol_shift` (3-component) and `mol_z` in the
sample dict so callers don't have to recompute it.

### 15d. Anisotropic step — keep `(dx, dy, dz)` separate

rho_atom grids use a different z-step than xy (e.g. `dx=dy=0.1 Å, dz=0.2 Å`).
The cube header supports anisotropic strides via the diagonal vectors:
```
nx  dx  0   0
ny  0   dy  0
nz  0   0   dz
```
**Do not** collapse to a scalar `step = mean(dx,dy,dz)` — this warps the z-axis
and misplaces features (same class of bug as §1 for DFT cubes).

### 15e. Verification recipe

After writing a cube, verify alignment with an independent reader:
```python
# 1. Bond lengths should be ~1.4 Å for C-C aromatic, ~1.1 Å for C-H
# 2. V_PP max should be <0.3 Å from nearest atom (was 1.8 Å before z-fix)
# 3. rho_atom should be positive at every atom position
# 4. Atom z-range should match the grid z-range (not offset by ~2 Å)
```

**Code:** `save_cube` in `invPPAFM/tests/testplot_ml_maps.py`  
**Verification:** independent `read_cube` reader (same file, inline in `main()`)

---

## 16. OpenCL `read_imagef` normalized coordinates — half-texel offset

**Root cause of GridFF lateral registration error (R2.1, 2026-08-09).**

OpenCL `read_imagef` with `CLK_NORMALIZED_COORDS_TRUE` maps normalized coordinate `u∈[0,1]` to texel **edges**, not centers. For an image of size `n`, texel center `i` is at `u = (i+0.5)/n`, not `u = i/n`.

The `dinvA/B/C` inverse transform matrices in `AFM.py` compute `coord = i/n` at voxel `i`, which is the **edge** between texels `i-1` and `i`. This causes a systematic **half-voxel shift** in every GridFF interpolation, scaling with grid spacing (`0.5·dx_grid ≈ 0.1 Å` for benzene).

| Variant | Synthetic RMSE | Real benzene Fz RMSE | Registration shift |
|---------|---------------|---------------------|--------------------|
| `current` (`i/n`) | 0.5 per axis | 1.17e-01 | ~0.5·dx_grid |
| `plus_05` (`(i+0.5)/n`) | **0** | **6.9e-03** (17× better) | **0** |

**Fix (applied 2026-08-09):** All 5 `dinvA/B/C` constructors in `AFM.py` now use centralized `_make_dinv_axis_aligned` / `_make_dinv_lvec` helpers with `+0.5/n` offset. Contact-surface kernels do **not** use `interpFE` and are unaffected.

**Rule:** When converting voxel index to OpenCL normalized coordinate, use `(i + 0.5) / n`, never `i / n`. This is the same class of bug as §3 (corner/node vs center voxel sampling) but for the GPU sampler path.

See [ContactSurface_Parity_InvPPAFM_Benzene.md](Tasks/ContactSurface_Parity_InvPPAFM_Benzene.md) §R2.1.

---

## 17. Contact-surface clamp at s=0 — nonzero Fz with constant E (FIXED)

**Primary source of contact model error (R2.2, 2026-08-09).**

The separable contact-surface model clamped the local height coordinate `s = z − h₀(x,y) − poly_z0` via `dz = fmax(s, 0.0f)` before evaluating `poly_z_doubling_modes`. When `s < 0` (probe below contact height), this froze E to its s=0 value **but still evaluated nonzero `dphi` at `dz=0`**, producing nonzero Fz for a z-constant E — **violating F=−∇E**. This is an implementation bug, not merely a representation limitation.

Brute-force Morse continues to vary strongly in this region (Fz ranges 0.07–2.92 eV/Å for benzene). The clamp bug produced:

| Metric | s<0 (clamp) | s∈[0.05,0.25) (fit region) |
|--------|-------------|---------------------------|
| Fz RMSE | 0.54 eV/Å | 0.025 eV/Å |
| Fz Pearson | 0.18 | 0.73 |
| Sign disagreements | 32 | 0 |
| % of total Fz² error | **99.15%** | 0.23% |

**Fix (2026-08-09):** Removed `fmax()` in all 3 call sites in `contact_surface.cl`. Now raw `s` is passed to `poly_z_doubling_modes`, which sets `active=false` and `dphi=0` when `s<0`, giving F=0 for constant E — conservative force semantics restored.

**Secondary effect — h0 interpolation error (FIXED):** B-spline h₀ with 1.0 Å spacing had RMSE=0.117 Å vs analytic sphere formula, with **systematic negative bias** (interp lower than analytic). Since `s = z − h₀`, lower h₀_interp means higher s_interp, so 255 points with s_analytic<0 escaped to s_interp≥0 (**missed clamp**), while 23 points entered false clamp. Error concentrated at sphere seams.

**Fix (2026-08-09):** Added cubic B-spline prefilter (`_bspline_prefilter_2d`) to `build_contact_height_map` so control coefficients exactly reproduce nodal values (Unser et al. causal/anti-causal recursive filter).

| h0 error source | RMSE | Max | Effect |
|----------------|------|-----|--------|
| B-spline vs analytic (all points) | 0.117 Å | 0.316 Å | 278/2223 clamp crossings |
| At seam dist [0.1,0.2) | 0.135 Å | 0.316 Å | 15% clamp crossing rate |
| At seam dist [0.2,0.5) | 0.165 Å | 0.305 Å | 13% clamp crossing rate |

**Rules:**
1. Below-support E must be constant and F must be zero — never return nonzero force for a z-constant energy. The `poly_z_doubling_modes` `active` flag handles this correctly when raw `s` is passed.
2. h₀ B-spline control coefficients must be prefiltered (not raw nodal values) for exact interpolation. Cardinal cubic B-spline does not interpolate its control points without prefiltering.
3. Always stratify contact-vs-brute error by local s before declaring model quality. Aggregate RMSE hides the fact that 99% of error is in the clamp region.
4. "99% of RMSE" depends on the validation set's below-contact sampling density. R2.4 must determine real PP trajectory clamp occupancy.

See [ContactSurface_Parity_InvPPAFM_Benzene.md](Tasks/ContactSurface_Parity_InvPPAFM_Benzene.md) §R2.2.

