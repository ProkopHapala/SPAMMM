# Report: Pyridine Kriging vs FDBM — NA recipes, tip swap, Pauli \(A,\beta\) fits (2026-07-21)

**Status:** investigating (not Done — do not mark closed without USER confirmation)  
**Date:** 2026-07-21  
**System:** Mithun `N-h` (pyridine) + tip `CO_O`  
**Artifact root:** [`debug/afm_fdbm_diag_pyridine_gui_match/`](../../debug/afm_fdbm_diag_pyridine_gui_match/)  
**HTML index (clickable gallery):** [`debug/afm_fdbm_diag_pyridine_gui_match/index.html`](../../debug/afm_fdbm_diag_pyridine_gui_match/index.html)  
**Parent campaign report:** [`Kriging_DFT_vs_DFTB_FDBM_pyridine.md`](Kriging_DFT_vs_DFTB_FDBM_pyridine.md)  
**Task:** [`doc/Tasks/Import_KrigingGridFF.md`](../Tasks/Import_KrigingGridFF.md)

This document is the **day log + conclusions + warnings** for the 2026-07-21 workstream: sample NA diagnostics, tip×sample energy profiles, two Pauli fit criteria, AFM rebuilds, and result packaging.

---

## 1. Goals for the day

1. Understand how **diag Denmat0 ρ_NA** vs **compact spherical NA** changes \(V_\mathrm{ES}\) / \(E\) for DFTB samples (and whether tip choice matters).  
2. Produce readable **ρ + E** z-profiles (N and H; later N/C/H) with fixed energy windows (±0.2 eV) and smooth sampling.  
3. **Fit Pauli \(A,\beta\)** so FDBM reproduces Kriging DFT for three setups:
   - DFT sample × DFT tip  
   - DFTB sample × DFT tip  
   - DFTB prolonged-Slater Pauli × DFTB tip  
4. Clarify / split **two fit criteria** (contact wall vs AFM residual) without destroying the original method.  
5. Rebuild **AFM 3-row** images (Fz unrelax / Fz relax / df) with fitted params.  
6. Repeat residual fit with **N+C only** (ignore H).  
7. **Organize** artifacts + write an HTML summary.

---

## 2. What we did (chronological)

### 2.1 Plotting hygiene (diag vs compact)

- Fixed `COMPARE_diagNA_vs_compactNA_zprofiles.png`:  
  - \(V_\mathrm{ES}\) and \(E\) panels **forced to ±0.2 eV** (USER SSOT; had been auto-scaled again).  
  - ρ left auto-scaled.  
  - Replaced nearest-voxel z-sampling stairs with **trilinear** `sample_field_z_profile(..., order=1)` default in `AFM_utils.py`.

### 2.2 Tip dependence of diag vs compact

- Confirmed earlier compare used **DFTB tip**.  
- Rebuilt E channels with **Mithun DFT `CO_O` tip** (clamp→compact NA tip Δρ), same sample \(V_\mathrm{ES}\).  
- Result: ρ/\(V_\mathrm{ES}\) unchanged (sample-only); Pauli identical between NA recipes (same ρ_scf); **only \(E_\mathrm{es}\) changes**.  
- At AFM heights \(z\gtrsim 2.5\) Å, diag↔compact ΔE stays ~meV; near contact \(z\sim 2\) DFT tip **amplifies** the difference.

### 2.3 Three ρ+E figures (both tips)

Helper: `plot_fdbm_rho_E_sites` — columns N & para-H; rows ρ (z∈[−2,+2]) and E (z∈[1,6], ±0.2 eV).

| Fig | Content | File |
|-----|---------|------|
| 1 | DFT sample (N-h cube), DFT tip vs DFTB tip | `COMPARE_1_DFTsample_bothTips_NH.png` |
| 2 | DFTB sample, **diag** ρ_NA, both tips | `COMPARE_2_DFTBsample_diagNA_bothTips_NH.png` |
| 3 | DFTB sample, **compact** NA, both tips | `COMPARE_3_DFTBsample_compactNA_bothTips_NH.png` |

Blue = DFT tip; orange = DFTB tip (`E_tot` / `E_es` / `E_pauli`).

### 2.4 Pauli \(A,\beta\) vs Kriging — first pass (contact-style)

Three FDBM field caches:

| Tag | Fields |
|-----|--------|
| `1_DFTxDFT` | `FDBM_sampDFT_tipDFT_fields.npz` |
| `2_DFTBsamp_DFTtip` | `FDBM_sampDFTB_tipDFT_fields.npz` |
| `3_DFTB_prolongedPauli_DFTBtip` | `FDBM_DFTB_prolongedPauli_fields.npz` |

Overlap recovered from stored \(E_\mathrm{pauli}\) via \(S=(E/A)^{1/\beta}\).  
Sites: chemically matched N / para-C / para-H (DFTB geometry ≠ Kriging XY — match by site name).

**Original contact criterion** (USER-insisted historically): fit \(A S^\beta \approx E_\mathrm{Kriging}\) on the **repulsive wall** where \(E>0\) (typically \(z\sim 1.5\)–\(2.0\) or \(2\)–\(3\)), assuming ES/vdW negligible.

### 2.5 Clarified objective — why “fit can look worse”

`_fit_pauli_powerlaw` minimizes

\[
\sum_i \big(E_\mathrm{ref}(z_i) - A\, S(z_i)^\beta\big)^2
\]

only on points with \(S>0\) and \(E_\mathrm{ref}>0\) in the fit window. It does **not** minimize \(\|E_\mathrm{tot}^\mathrm{FDBM}-E^\mathrm{Kriging}\|\) over the attractive well. So after a “good” wall fit, green \(E_\mathrm{tot}\) can look worse than an older hand-tuned \(A,\beta\) if ES is wrong or the well dominates visual judgment.

### 2.6 New residual fit (AFM heights) — **separate function**

**Do not override contact.** Added:

| Function | Role |
|----------|------|
| `_fit_pauli_powerlaw` | Contact wall (original; \(E_\mathrm{ref}>0\)) |
| `_fit_pauli_powerlaw_residual` | AFM residual; **signed** \(E_\mathrm{ref}\) |

Residual objective (USER request):

\[
\min_{A,\beta}\sum_{\text{sites}}\sum_{z\in[2.5,5]}
\big(E_\mathrm{Kriging}-E_\mathrm{es}-E_\mathrm{vdW}-A\,S^\beta\big)^2
\]

Magenta line on plots = **Kriging−ES−vdW** (“Kriging-Pauli”).  
CLI switch: `tests/SPM/testplot_kriging_vs_fdbm_cube.py --fit_mode contact|residual` (or `--fit_residual`).

### 2.7 AFM rebuilds with residual params

Standard SSOT `plot_afm_Fz_df_threerow`: tip 5.5→8 step 0.5, probe = tip−3, amp=1.0 Å, \(K_\mathrm{LAT}=0.5\) N/m.

### 2.8 Residual fit N+C only (ignore H)

Same residual method; pool **N + para-C** only (H plotted for validation, not in loss). Rationale: H contrast is weak; H wants systematically larger \(A\) and pulls pooled fits.

### 2.9 Packaging

```
debug/afm_fdbm_diag_pyridine_gui_match/
  index.html                          ← gallery + descriptions
  pauli_fit_kriging_3ways/
    fit_contact_NCH/                  ← contact / Kriging-E series
    fit_resid_NCH/                    ← residual N+C+H
    fit_resid_NC/                     ← residual N+C only
    N-h-CO_O_kriging_*                ← Kriging cache
```

---

## 3. Numerical results (residual fits)

### 3.1 Pooled N+C+H — `_fit_pauli_powerlaw_residual`, \(z\in[2.5,5]\)

| Case | A | β | R² (pooled) |
|------|---|---|-------------|
| DFT×DFT | 18.07 | 0.844 | 0.939 |
| DFTB sample × DFT tip | 17.74 | 0.695 | 0.939 |
| DFTB prolonged Pauli × DFTB tip | 15.86 | 0.837 | 0.950 |

Artifacts: `pauli_fit_kriging_3ways/fit_resid_NCH/`

### 3.2 Pooled N+C only (H ignored)

| Case | A | β | R² (pooled) |
|------|---|---|-------------|
| DFT×DFT | 22.06 | 0.888 | 0.934 |
| DFTB sample × DFT tip | 17.70 | 0.695 | 0.930 |
| DFTB prolonged Pauli × DFTB tip | 19.74 | 0.886 | 0.948 |

Artifacts: `pauli_fit_kriging_3ways/fit_resid_NC/`

**Per-site pattern (all residual runs):** H always wants **larger \(A\)** than N/C (~20–33 vs ~9–16). Pooling H therefore softens N/C compromise; dropping H raises pooled \(A\) for DFT×DFT and prolonged cases.

### 3.3 Qualitative AFM

- **Kriging** remains the morphological reference (attractive halo / site contrast).  
- **Prolonged Pauli × DFTB tip** after residual fit: clearest improvement of \(E_\mathrm{tot}(z)\) vs Kriging among the three (especially vs old stock \(A,\beta=13.42/0.706\)).  
- **DFT×DFT**: residual fit improves C/H \(E_\mathrm{tot}\) RMSE vs Kriging on [2.5,5] in several metrics, but **N can worsen vs old** if cube tip \(E_\mathrm{es}\) is still pathological — Pauli cannot invent the Kriging well alone when ES is wrong.  
- **DFTB×DFT tip**: mild ES; residual fit helps C/H; N sometimes closer with old params depending on metric.

---

## 4. Physics conclusions

1. **Pauli dominates AFM contrast** at usual heights; sample NA recipe (diag vs compact) mainly shifts \(V_\mathrm{ES}\) by a near-constant far-field offset (~+0.12 eV) → little df/Fz change once Pauli is fixed.  
2. **Tip Δρ** (DFT cube clamp vs soft DFTB) matters more for \(E_\mathrm{es}\) magnitude than sample NA choice at \(z\gtrsim 2.5\).  
3. **Dual basis (prolonged Slater)** is for **Pauli density only**; never charge-normalize; ES stays stock Δρ → \(V_\mathrm{ES}\). Prolonged sample Pauli needs **re-fitted** \(A,\beta\) (stock params under-repel).  
4. **Contact wall fit** and **AFM residual fit** answer different questions:
   - Contact: “match short-range Pauli physics where ES≈0.”  
   - Residual: “make \(E_\mathrm{tot}\) track Kriging on AFM heights given current ES+vdW.”  
   Residual fit **can absorb ES errors into Pauli** — useful for imaging parity, dangerous as a claim of “correct Pauli.”  
5. **One global \(A,\beta\)** cannot optimally serve N, C, and H simultaneously; prefer N+C pool for imaging contrast, or per-type params later.

---

## 5. Problems, caveats, how we addressed them

| Problem | Symptom | Fix / mitigation |
|---------|---------|------------------|
| Auto y-limits on V/E | ± huge scales; “ignored ±0.2 again” | Hardcode ±0.2 eV on V/E panels; ρ auto |
| Stepwise z-curves | Nearest voxel | `sample_field_z_profile` default `order=1` (trilinear) |
| Fit looks worse after fit | Optimized Pauli≠visual \(E_\mathrm{tot}\) | Document criterion; add residual fitter for AFM-range \(E_\mathrm{tot}\) goal |
| Overrode contact fitter | USER rejected flag-on-same-fn | Restore `_fit_pauli_powerlaw`; new `_fit_pauli_powerlaw_residual` |
| Geometry mismatch DFTB vs Kriging | Different COM / XY | Chemical site matching (N, para-C, para-H), not raw XY copy |
| Cube tip ES bend | Attractive \(E_\mathrm{es}\) too strong / wrong morphology | Clamp tip Δρ; still open — do not trust residual Pauli as physics fix |
| H in pooled fit | Inflates \(A\) | Optional N+C-only pool |
| Artifact sprawl | 100+ files in one folder | Subfolders + `index.html` |
| OpenCL sandbox | PoCL instead of NVIDIA | Shell with unrestricted ICD (`all`); `preferred_vendor='nvidia'` |

### Remaining open (not solved)

- Cube **tip** valence-like Δρ still imperfect vs DFTB morphology (apex charge sign / cusp residuals).  
- No USER-approved lock of a single \(A,\beta\) for production GUI.  
- Prolonged **tip** Slater SA not done (sample-only prolonged run exists).  
- WideXY / GridFF margin policy documented but not fully standardized in all scripts.  
- Status must stay **investigating** until USER confirms.

---

## 6. Takeaways for future developers / users

### Must label

Always state **sample source** and **tip source** separately (DFT-cube vs DFTB). Cross combos are not interchangeable.

### Must choose fit mode explicitly

```text
contact  → AFM_utils._fit_pauli_powerlaw          # wall, E_ref>0
residual → AFM_utils._fit_pauli_powerlaw_residual # Kriging−ES−vdW, signed
```

CLI: `--fit_mode contact|residual` in `testplot_kriging_vs_fdbm_cube.py`.

### Plot SSOT

Use `spammm/SPM/AFM_utils.py` helpers (`plot_afm_Fz_df_threerow`, `plot_fdbm_rho_E_sites`, `fdbm_probe_sites_nch`, …). Skill: `doc/AGENTS/skills/afm-plotting/SKILL.md`.

Conventions that burned us when ignored:

- Probe height ≠ tip height (\(L\approx 3\) Å).  
- df amp = **peak** 1.0 Å; dense z for `compute_df_amp`.  
- \(E-E(6)\), \(V-V(8)\); yellow AFM band \(z\ge 2.5\); **aspect='equal'**.  
- Overlay **atoms only**, never Kriging `points_clean` as `apos`.

### Dual basis

Prolonged / SA Slater ρ → **Pauli only**. Never normalize ∫ρ. ES = stock Δρ.

### GPU

Prefer NVIDIA via `OpenCLBase.select_device(preferred_vendor='nvidia')`. Agent shells that run OpenCL need unrestricted permissions so the NVIDIA ICD is visible (PoCL-only = wrong benches).

### Reporting paths

Prefer **absolute paths** (or `file://` links) in reviews so humans can click; relative globs are easy to miss.

### Do not mark Done

Code + plots ≠ confirmation. Wait for USER before changing task status.

---

## 7. Recommended next steps

1. USER review of HTML gallery: residual N+C+H vs N+C AFM morphology vs Kriging.  
2. If imaging parity is the goal: adopt **residual N+C** \(A,\beta\) per tip×sample combo as provisional GUI presets (still label as “fit to Kriging residual,” not ab initio Pauli).  
3. If physics Pauli is the goal: keep **contact** fits; fix cube tip Δρ / ES before re-fitting.  
4. Tip prolonged Slater SA + re-fit.  
7. Optional: atom-type-resolved \(A_Z,\beta_Z\) (N vs C vs H) instead of one global pair — **elevated to task** `doc/Tasks/Pauli_A_beta_KrigingTransferability.md` (site maps + cross-molecule transferability).

---

## 8. Key file map

| Path | Role |
|------|------|
| `spammm/SPM/AFM_utils.py` | `_fit_pauli_powerlaw`, `_fit_pauli_powerlaw_residual`, plot helpers, cube/DFTB FDBM builders |
| `tests/SPM/testplot_kriging_vs_fdbm_cube.py` | `--fit_mode contact\|residual` |
| `debug/.../index.html` | Human gallery |
| `debug/.../pauli_fit_kriging_3ways/fit_resid_NCH/` | Residual N+C+H FIT + AFM |
| `debug/.../pauli_fit_kriging_3ways/fit_resid_NC/` | Residual N+C FIT + AFM |
| `debug/.../pauli_fit_kriging_3ways/fit_contact_NCH/` | Contact / Kriging-E series |
| `FDBM_*_fields.npz` (root of debug dir) | Cached E / V channels for rebuilds |

---

*End of 2026-07-21 report. Companion narrative of the broader campaign: `Kriging_DFT_vs_DFTB_FDBM_pyridine.md`.*
