---
type: Report
title: Cube FDBM ES — Codex handoff (element-invariant Δρ / NA)
tags: [FDBM, ES, cube, Δρ, NA, pentacene, handoff]
timestamp: 2026-07-24
status: investigating
---

# Cube FDBM ES — Codex handoff (pentacene)

**Status:** implemented + panels regenerated; **awaiting USER visual confirmation** — do not mark fixed yet.

**Full investigation:** [`Cube_ES_DeltaRho_NA_dipole.md`](Cube_ES_DeltaRho_NA_dipole.md) · **Caveats:** [`../Caveats.md`](../Caveats.md)

---

## Verdict

The asymmetric DFT-cube AFM / V_ES row was **not** tip or Poisson. It was a fake far-field dipole in `Δρ = ρ_SCF − ρ_NA` from three stacking bugs. With all three fixed, pentacene native `|p_xy|` falls **0.28 → 0.018 e·Å** (near the `ρ_N−ρ_NA.cube` control ~0.014).

Production no longer needs `rho_NA.cube`. Keep that cube only as a diagnostic.

---

## Root causes (priority)

1. **Anisotropic cube step collapsed to `mean(sx,sy,sz)`**  
   Mild pySCF aniso (~0.2%) warped nuclei vs density → `|p|~1.9 e·Å`.  
   **Fix:** keep `step` as `(3,)` on the native cube.

2. **Main residual ~0.28 e·Å — unresolved AE cores + per-atom `Q_rem` feedback**  
   ~0.15 bohr cubes cannot resolve C 1s cusps. Equivalent carbons land at different sub-voxel phases → different sampled peaks and `Q_rem_i`. Soft-clamp removes the alias from `ρ_electronic`, but the old recipe `q_i = Z_i − Q_rem_i` fed that noise back into compact nuclear charges and recreated the fake dipole.  
   **Fix:** `q_na_mode='element_mean'` (default):
   ```text
   Q_comp(Z) = mean_i(Q_rem_i for Z_i=Z)
   q_i = Z_i − Q_comp(Z_i)   # then charge-close so ∫ρ_NA = ∫ρ_clamped
   ```

3. **Cube nodes passed into a center-source projector**  
   `GridsOCL.project_density` places sources at `origin+(i+½)·step`; cubes are node-sampled (`origin+i·step`). Old adapter shifted every charge by `+step/2` and biased moments.  
   **Fix:** `project_density_to_grid(..., src_convention='nodes')` passes `origin − 0.5·step`. Use `'centers'` only for true cell centers.

Also: `mirror_asymmetry_2d` now reflects about a **fractional** physical center (linear interp). Integer-pixel flip lied on half-pixel COM cases.

---

## Key code

| Location | Role |
|----------|------|
| `spammm/SPM/AFM_utils.py` → `delta_rho_clamp_compact_na(..., q_na_mode='element_mean')` | SSOT Δρ |
| `project_density_to_grid(..., src_convention='nodes')` | Cube→grid adapter |
| `allelectron_cube_to_fdbm_grid` | Clamp + project pipeline |
| `get_density_from_cube` | Keep `(3,)` step |
| `mirror_difference_2d` / `mirror_asymmetry_2d` | Fractional-center metrics |

**Tests:** `test_clamp_compensation_is_element_invariant`, `test_mirror_asymmetry_fractional_center`, `test_cube_node_adapter_preserves_charge_and_dipole`.

---

## Pentacene numbers (native cube, RTX 3090)

| Recipe | \|p_xy\| | V_mX (z+1) |
|--------|----------|------------|
| ρ_N − Gauss (aniso OK) | ≈ 0.284 | ≈ 0.42 |
| clamp per-atom (legacy) | ≈ 0.284 | ≈ 0.27 |
| clamp **element_mean** (now) | ≈ **0.018** | ≈ **0.033** |
| ρ_N − ρ_NA.cube (control) | ≈ 0.014 | ≈ 0.035 |

Projected Δρ after node-correct project: `|q| < 2e−7 e`; small residual `p`.

---

## REVIEW plots

Under `debug/fdbm_fukui_panel_flat/pentacene/`:

| File | Meaning |
|------|---------|
| `compare_cube_stock_prolonged.png` | Full AFM panel |
| `es_diag/dipole_origin_bisect.png` | Smoking-gun 3×4 Δρ/V bisect |
| `es_diag/es_chain_native_clamp_compact.png` | Native clamp chain |
| `es_diag/es_chain_fdbm_grid.png` | After project to FDBM grid |

```bash
python run_spm.py es-diag --molecule pentacene --outdir debug/fdbm_fukui_panel_flat
python run_spm.py panel-fukui --molecule pentacene --outdir debug/fdbm_fukui_panel_flat
```

---

## Still open

- Confirm AFM cube row by eye (USER).
- Deliberately polar molecule: physical valence dipole must survive grid shifts.
- Clamp `y1/y2/rc` still heuristic; long-term = element-specific analytic cores.
- Pentacene XYZ is not exact D₂h (mirror residuals ~0.03–0.10 Å) → leftover mY.
- Do **not** re-enable `strip_monopole_dipole` as a “fix”.
