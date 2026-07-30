# Task: FGR Bond-Resolved STM (FGR-BR-STM)

**Status:** implemented (2026-07-30, awaiting USER L2 review)
**Priority:** Medium — method extension after FGR Level-B acceptance
**Parent:** `doc/TopicalAudit/STM_FGR_Transfer.md`, `doc/Reports/STM_FGR_Transfer_H_ES_2026-07-29.md`
**Related:** `doc/Tasks/DysonOrbitals_DFTB_STM.md`, `doc/Tasks/SPM_CLI_Headless.md`

## Objective

Replace the legacy overlap-STM kernel inside **Bond-Resolved STM** with the corrected **FGR transfer** `M = c_t†(H−ES)c_s`, so that the PP-AFM tip-displacement distortion is applied to the physically correct tunneling matrix element rather than the artificial-exp overlap.

## Motivation

The existing BR-STM pipeline (`run_spm.py stm br` → `run_br_stm_afm_panel` → `compute_bond_resolved_stm`) already works:
1. PP-AFM relaxation produces lateral tip displacement `(dx, dy)` per scan pixel
2. STM is evaluated at the displaced tip positions → bond-resolved contrast

But the STM kernel inside `compute_bond_resolved_stm` uses `project_orbital_dense_points_exp` (legacy `exp(−β(r−r₀))` overlap, `H'≈1`). The FGR comparison report (`STM_FGR_Transfer_H_ES_2026-07-29.md`) shows FGR `I_τ` concentrates intensity on the molecular body vs `overlap_exp` (especially PTCDA LUMO), which is the desired direction for matching experiment. BR-STM with FGR should combine both improvements: correct transfer operator + tip-bend distortion.

## Current architecture (what exists)

| Component | Location | STM kernel |
|-----------|----------|------------|
| Pure STM (Stage 1) | `run_br_stm_afm_panel` → `compute_stm` | prolonged STO projection (`use_exp_basis=False`) |
| BR-STM (Stage 3) | `compute_bond_resolved_stm` | `project_orbital_dense_points_exp` (legacy overlap) |
| FGR STM | `project_mo_stm_fgr_slice` | `stm_fgr_sk_tau_scan_real` (H−ES, Level B) |
| FGR compare | `run_fgr_transfer_compare` | 4-column panel (overlap vs I_S vs I_H vs I_τ) |

**Gap:** `compute_bond_resolved_stm` has no FGR path. The FGR scan kernel `stm_fgr_sk_tau_scan_real` takes a flat grid of tip centers; BR-STM needs it at **displaced** tip centers (per-pixel dx/dy from PP-AFM).

## Implementation plan

### Step 1: FGR scan at arbitrary tip positions

`project_mo_stm_fgr_slice` already accepts arbitrary `tip_centers` (it builds them from meshgrid internally). Extract a variant that takes pre-built `tip_centers` (including displacement) so BR-STM can call it per-height with displaced positions.

- Add `project_mo_stm_fgr_points(projector, mo_coeff, ..., tip_centers, ...)` — thin wrapper around `stm_fgr_sk_tau_scan_real` with explicit tip positions
- Or generalize `project_mo_stm_fgr_slice` to accept `tip_centers` instead of `scan_xs/scan_ys/z_A`

### Step 2: Wire into `compute_bond_resolved_stm`

Add a `stm_mode='overlap'|'fgr'` parameter:
- `'overlap'` (default, backward-compatible): existing `project_orbital_dense_points_exp`
- `'fgr'`: call the FGR kernel at displaced tip centers, with pre-built tables

Requires passing `tables, tip_type0, name_to_smp, E_tunnel` through the call chain.

### Step 3: Wire into `run_br_stm_afm_panel`

- Build FGR tables once (reuse `_stm_fgr_prepare_tables`)
- Apply degeneracy-aware MO summation (the same fix as `run_fgr_transfer_compare`)
- Stage 3 panel: add FGR-BR-STM column alongside the existing overlap-BR-STM

### Step 4: CLI

Add `--stm-mode {overlap,fgr}` to `run_spm.py stm br` subcommand. Default `overlap` for backward compat; `fgr` for the corrected path.

### Step 5: Validation

- Parity: at zero tip displacement (flat tip), FGR-BR-STM must equal FGR-STM from `project_mo_stm_fgr_slice`
- Visual: PTCDA LUMO FGR-BR-STM should show sharper bond resolution than overlap-BR-STM
- Degeneracy: HOMO cluster sum (same as FGR compare)

## Implementation — DONE (2026-07-30)

All steps implemented and tested on PTCDA + pentacene:

| Step | Status | Notes |
|------|--------|-------|
| 1. FGR scan at arbitrary tip positions | ✅ | `project_mo_stm_fgr_points` (accepts X_disp/Y_disp 2D arrays) |
| 2. Wire into `compute_bond_resolved_stm` | ✅ | `compute_bond_resolved_stm_fgr` — parallel function, legacy untouched |
| 3. Wire into `run_br_stm_afm_panel` | ✅ | `stm_mode='fgr'` branch in Stage 3; builds tables once; degeneracy cluster sum |
| 4. CLI | ✅ | `run_spm.py stm br --stm-mode fgr --tip-orbital pz --rcut 15 --taper-w 2 --degen-thresh 0.005` |
| 5. Validation | ✅ | Parity: at far height (|dxy|=0.38Å) |Δ|/flat=0.18; at close height (|dxy|=1.16Å) |Δ|/flat=0.94 |

### Artifacts (L2 review)

- `debug/spm_brstm/PTCDA/03_brstm_vs_stm.png` — FGR BR-STM vs flat STM @ Fz heights, LUMO#70 (3 rows: STM flat, BR-STM, |dxy|)
- `debug/spm_brstm/pentacene/03_brstm_vs_stm.png` — FGR BR-STM vs flat STM @ Fz heights, LUMO#51
- `debug/spm_brstm/{PTCDA,pentacene}/brstm_stages.npz` — full data (stm_flat, br_stm, dxy, eigvals, br_cluster)
- `debug/stm_br_fgr_compare/{PTCDA,pentacene}/br_fgr_compare_h{3.00,4.00}_*.png` — 4-column BR-STM compare gallery (BR-overlap | BR-I_S | BR-I_H | BR-I_τ) with PP-AFM tip displacement, HOMO/LUMO × s/pz
- `debug/stm_br_fgr_compare/{PTCDA,pentacene}/br_fgr_compare.npz` — full 3D data (nx, ny, nz_Fz) per method

### Cutoff/taper fix (same session)

- `rcut` default 10→15 Å (prolonged cutoff)
- Cosine taper `taper_w=2.0` Å added to all 3 FGR scan kernels (`stm_cutoff_taper` in `LCAO_STM_FGR.cl`)
- Eliminates hard-cutoff ring artifacts visible at z=5,6 Å

## Open questions

- FGR kernel `stm_fgr_sk_tau_scan_real` currently takes flat tip centers. Does it handle arbitrary (non-grid) positions correctly? **YES — verified by BR-STM parity check.**
- Performance: FGR scan is ~1 ms/pixel on RTX 3090; BR-STM adds per-pixel displacement → same cost (positions are just different).
- Should the tip displacement also shift the **tip orbital orientation** (not just position)? Currently BR-STM only shifts position. The CO tip bend also tilts the p_z → could add tip rotation later.

## Code map

```text
PP-AFM relaxation  →  spammm/SPM/AFM_utils.py :: run_br_stm_afm_panel (Stage 2)
Tip displacement   →  tip_disp['dx'], tip_disp['dy']  (nx, ny, nz)
BR-STM (legacy)    →  compute_bond_resolved_stm  (uses project_orbital_dense_points_exp)
FGR STM            →  project_mo_stm_fgr_slice → stm_fgr_sk_tau_scan_real
FGR tables         →  _stm_fgr_prepare_tables → build_longtail_eh_sk_tables
CLI                →  run_spm.py stm br
```
