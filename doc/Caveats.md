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
| `bspl_dx` / `scan_dx` ≪ 1 Å | Razor atom pins; against coarse-grain design | Defaults **1.0 / 0.5 Å** (atom-scale nodes/pixels) |
| Trusting contact XY vs GridFF without profiles | Looks “better resolved” | Always check E/Fz(z) vs **brute**; GridFF ≈ brute for classical FF |

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

