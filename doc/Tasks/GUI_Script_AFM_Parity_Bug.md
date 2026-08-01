# BUG: GUI Script AFM Images Don't Match Manual GUI Clicking

## Status: UNRESOLVED — needs investigation by a more capable agent

## 2026-08-01 implementation under USER review

The investigation found two independent state/path mismatches and implemented
candidate corrections. This status remains **UNRESOLVED** until the USER reviews
the resulting windmill, AFM, and BR-STM images.

1. **Rigid pose height was split by 3.25 Å.** `_on_build()` created
   `RigidEnsemble` at `z=+3.0`, while `attach_pairff_faf()` moved only GPU
   `poss` and `_mb_pos` to `Z_SURF_TOP + 3.0 = -0.25`. Display/AFM read the
   ensemble; accepted MC moves uploaded it back to the GPU. The build now
   resolves the absolute FAF height before creating any pose store and fails
   loud if ensemble/GPU/`_mb_*` differ.
2. **The script was not using the AFM product-button path.** Its manual S1–S4
   sequence called `run_afm_stage3()`, which reads cached stock `rho_scf`;
   the default prolonged product path `_run_afm_s1_to_s4()` explicitly computes
   the prolonged Pauli density. `conference_demo.py` now calls
   `run_afm_full_pipeline()`, the same function as the **AFM** button, then
   `run_br_stm()`, the same function as the **BR-STM** button.
3. `_sync_display()` now explicitly marks the AFM geometry dirty in addition
   to the existing strong geometry hash.
4. The script no longer writes AFM backend, projection, MO, or height widgets.
   It validates the untouched defaults and fails loud if they are not
   `3ob-3-1`, `DFTB FDBM (prolonged)`, `prolonged`, and `z=3.0 Å`.

NVIDIA end-to-end evidence from the real GUI:

```text
4×PTCDA MC: accepted=26/1000, E_final=-0.274403 eV
AFM defaults: 3ob-3-1, DFTB FDBM (prolonged), z=3.0 Å
AFM |dxy|_max=1.1573 Å
BR-STM grid finite, nonzero, max=4.0908e-07
```

The numerical path and postconditions now pass; visual parity still requires
USER confirmation and must not be marked resolved before that review.

## The Core Requirement

**GUI scripts must use the EXACT same code path as manual user clicking.** This is the entire point of the GUI scripting system — it exists so that demonstrations are reproducible and debuggable. If a script calls different functions, uses different parameters, or takes a different code path than manual clicking, it defeats the purpose. The script should be a thin orchestrator that calls the same `run_afm_stage1/2/3/4`, `run_br_stm`, etc. functions that the GUI buttons call, with the same default parameters.

## The Problem

When the user manually:
1. Builds 4×PTCDA via RigidAssembly panel (Build button)
2. Clicks AFM stages 1-4
3. Clicks BR-STM

...the AFM images show **high-resolution contrast corresponding to probe particle (PP) distortions** — sharp features, bond-like structures in BR-STM.

When the `conference_demo.py` script does the same sequence programmatically (calling the same `run_afm_stage1/2/3/4` and `run_br_stm` functions), the AFM images look **completely different** — no PP distortion contrast, no bond resolution, flat/blurry images.

The numerical output is nearly identical regardless of `z_mol` setting:
- z_mol=0.0: AFM plot range=[-0.077, 0.000], mean=-0.029, |dxy|_max=0.95
- z_mol=3.0: AFM plot range=[-0.076, 0.000], mean=-0.032, |dxy|_max=0.95

This should be physically impossible if the scan heights were absolute z, but they're not (see below).

## What I Found (Architecture Analysis)

### 1. AFM Scan Heights Are RELATIVE to mol_z, Not Absolute

**File:** `spammm/SPM/AFM_utils.py`, function `compose_and_relax_total` (line ~3089)

```python
mol_z = float(atomPos[:,2].max())          # line 3116
...
afmulator.scan_fdbm(scan_xs, scan_ys, heights, mol_z=mol_z, ...)  # line 3131
```

The `heights` parameter [3.7, 4.7] is **heights above mol_z**, not absolute z. So:
- z_mol=0.0 → actual scan z = 0 + [3.7, 4.7] = [3.7, 4.7] Å
- z_mol=3.0 → actual scan z = 3.0 + [3.7, 4.7] = [6.7, 7.7] Å

**This means z_mol does NOT affect the AFM scan distance** — the probe always scans 3.7-4.7 Å above the molecule. Changing z_mol only affects MC substrate physics, not AFM. This is why my z_mol=0.0 fix had no effect on the AFM images.

### 2. The Grid Origin Also Shifts With the Molecule

**File:** `spammm/SPM/ModularPipeline.py`, function `stage2_project` (line ~496)

```python
z_min = self.atomPos[:,2].min() - self.margin   # line 496
origin = np.array([x_min, y_min, z_min], ...)   # line 498
```

The DFTB density grid origin z depends on the molecule's z position. So when z_mol changes, both the grid AND the scan heights shift together — the relative geometry is preserved.

### 3. The Script Uses the Same Functions (Confirmed)

The `conference_demo.py` script calls:
- `RA._on_build(window)` (via `GSU.click_button(window.ra_build_btn)`)
- `AFM.run_afm_stage1(window)` through `AFM.run_afm_stage4(window)`
- `AFM.run_br_stm(window)`

These are the SAME functions the GUI buttons call. The script does NOT call different functions. So the discrepancy is NOT in which functions are called, but in the **state/parameters** when they're called.

### 4. Suspected Discrepancy: State Not Properly Synced

The most likely root cause is that the GUI script does not properly replicate the full state that manual clicking would establish. Possible issues:

#### a) Geometry sync: RigidAssembly → backend.sys → AFM

**File:** `spammm/GUI/RigidAssemblyExtension.py`

- `_on_build` (line 189): builds ensemble, calls `_sync_display` (line 278)
- `_sync_display` (line 167): calls `_assembly_world_atoms` → `_update_graph`
- `_update_graph` (line 153): writes to `backend.graph`, calls `backend._sync_sys()`
- AFM reads from `window.backend.sys.apos` via `_get_afm_geometry` (line 83)

**Question:** Does `backend._sync_sys()` properly update `sys.apos` from `graph`? If the graph has the right atoms but `sys.apos` is stale or has different z-coordinates, the AFM would see wrong geometry.

#### b) AFM pipeline caching

**File:** `spammm/GUI/AFMExtension.py`, `_ensure_pipeline` (line 370)

The pipeline uses a geometry hash (MD5 of atomPos + enames) to decide whether to reinit. If the hash matches a cached pipeline from a previous run (same atom count, same elements), it might reuse a stale pipeline with wrong geometry.

The `_afm_output_dir` is set once and persists. Cache files (`cache_stage1_scf.npz`, etc.) in this directory survive across script runs. If the script runs after a manual session with the same molecule, it might load cached results from the manual run.

**Key question:** Is `_afm_output_dir` cleared between manual and script runs? It's initialized to `None` in `init_afm` (line 1388) but only set to a tempdir on first use. If the GUI is restarted, it gets a new tempdir. But if the script runs in the same session as a manual run, it reuses the same dir.

#### c) AFM parameter widgets not fully synced

The script sets some AFM parameters via `GSU.set_spin_value` but may miss others. The GUI has many interconnected widgets — `afm_basis_combo`, `afm_step_spin`, `afm_margin_spin`, `afm_pauli_a_spin`, `afm_pauli_beta_spin`, `afm_vdw_c6_spin`, `afm_klat_spin`, `afm_bond_length_spin`, `afm_projection_combo`, etc. If any of these differ from what manual clicking would use, the AFM results differ.

**The script should NOT override any parameters** — it should use GUI defaults. But it also needs to ensure the defaults are actually loaded (widgets initialized to correct values).

#### d) Dirty flags not properly set

The AFM pipeline uses `AFMDirtyFlags` (line 32) to track which stages need recomputation. If the script doesn't properly mark stages dirty after geometry changes, the pipeline might skip recomputation and return stale cached results.

When `_on_build` is called, does it mark the AFM geometry as dirty? Let me check:

**File:** `spammm/GUI/RigidAssemblyExtension.py` — `_on_build` does NOT call `window._afm_dirty.mark_geometry_changed()`. The AFM dirty flags are only set by AFM widget signal handlers. If the script builds molecules without triggering AFM widget signals, the AFM might not know geometry changed.

**This is likely a key issue.** When the user manually builds molecules and then clicks AFM, the AFM pipeline sees a new geometry hash and reinitializes. But if dirty flags aren't set, the pipeline might reuse cached results.

#### e) The `_afm_pipeline` might not be None'd after build

After `_on_build` creates new geometry, the existing `_afm_pipeline` (if any) is stale. The geometry hash check in `_ensure_pipeline` should catch this, but only if the hash actually changes. If the previous pipeline was for a different geometry but same atom count + elements, the hash would differ and reinit would happen. But if there's any hash collision or if the hash isn't computed correctly, the stale pipeline persists.

## Relevant Files

| File | Key Functions | Role |
|------|--------------|------|
| `spammm/GUI/gui_scripts/conference_demo.py` | `run(window, argv, ctx)` | The demo script — orchestrates build → MC → AFM → BR-STM → PME |
| `spammm/GUI/RigidAssemblyExtension.py` | `_on_build` (189), `_sync_display` (167), `_assembly_world_atoms` (88), `_update_graph` (153) | Builds rigid assembly, syncs to AtomicGraph → backend.sys |
| `spammm/GUI/AFMExtension.py` | `_get_afm_geometry` (83), `_ensure_pipeline` (370), `run_afm_stage1` (894), `run_afm_stage2` (914), `run_afm_stage3` (937), `run_afm_stage4` (964), `run_br_stm`, `plot_afm_slice` (1110), `AFMDirtyFlags` (32), `init_afm` (1385) | AFM pipeline orchestration, reads geometry from backend.sys |
| `spammm/SPM/ModularPipeline.py` | `ModularAFMPipeline.__init__` (69), `_init_geometry_and_grids` (185), `stage2_project` (415), `stage4_relax` (702) | AFM compute pipeline — grid setup, density, relaxation |
| `spammm/SPM/AFM_utils.py` | `compose_and_relax_total` (3089) | Key: `mol_z = atomPos[:,2].max()`, heights are relative to mol_z |
| `spammm/GUI/gui_script_utils.py` | `set_spin_value` (57), `click_button` (41), `set_combo_text` (47) | Script helpers — thin wrappers around Qt widget operations |
| `spammm/GUI/VispyUtils.py` | `fit_to_atoms` (706) | Camera viewport fit |
| `spammm/topology/MoleculeEditorBackend.py` | `_sync_sys`, `sys.apos` | Backend that holds atom positions; AFM reads from here |

## What the Next Agent Should Do

1. **Start by reproducing both paths** — run the GUI manually (build 4×PTCDA, click AFM stages, click BR-STM) and run the script, then compare the AFM plot output numbers (range, mean, |dxy|_max). If they differ, the discrepancy is real.

2. **Check if `backend.sys.apos` matches `_assembly_world_atoms` output** after `_on_build` — print both in the script and verify they're identical (same z, same xy).

3. **Check if the AFM pipeline is reusing cached results** — add a print in `_ensure_pipeline` showing whether `needs_reinit` is True or False, and what the geometry hash is. If it's reusing a cache from a previous run, that's the bug.

4. **Check if AFM dirty flags are properly set** after `_on_build` — the build changes geometry but may not mark AFM stages dirty. Add `window._afm_dirty.mark_geometry_changed()` after build and see if that fixes it.

5. **Compare ALL AFM widget values** between manual and script — print every `afm_*_spin.value()` and `afm_*_combo.currentText()` in both paths and diff them.

6. **Check if `run_br_stm` reads MO selection from the right widget** — the script sets `afm_stm_mo_list` to "1" and `afm_stm_relative_mo` to True, but `run_br_stm` might read from a different source.

7. **The key principle: the script must be a thin orchestrator** — it should only call the same functions the GUI buttons call, with the same default parameters. No parameter overrides. No shortcuts. If the GUI sets up state via widget signals, the script must trigger those same signals (via `GSU.set_spin_value` etc.) or explicitly set the same state.

## Current State of conference_demo.py

The script currently:
- Sets `ra_z_spin` to GUI default (3.0) — correct, no override
- Sets AFM params to GUI defaults — correct, no overrides (removed all overrides)
- Sets `afm_stm_mo_list` to "1" with `afm_stm_relative_mo` checked — for LUMO selection
- Calls `fit_to_atoms(margin=3.0)` after build and MC — for viewport
- Calls the same `run_afm_stage1/2/3/4` and `run_br_stm` functions as GUI buttons

The script does NOT:
- Call `window._afm_dirty.mark_geometry_changed()` after build (SUSPECTED BUG)
- Clear AFM cache directories before running
- Print/verify that `backend.sys.apos` matches the assembly's world atoms
- Verify all AFM widget values match GUI defaults
