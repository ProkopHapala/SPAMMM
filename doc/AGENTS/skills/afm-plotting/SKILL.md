---
name: afm-plotting
description: AFM E/Fz/df plotting SSOT — 2D XY maps and 1D z-profiles. Use whenever plotting AFM energy, force, frequency shift (imshow panels or E(z)/Fz(z)/df(z) curves). Forbids ad-hoc matplotlib for these plots.
trigger:
  glob:
    - "**/tests/SPM/**"
    - "**/spammm/SPM/**"
    - "**/*afm*"
    - "**/*fdbm*"
    - "**/*zscan*"
---

## HARD RULE

**Never** write ad-hoc `imshow` / `plt.plot` for AFM **E / Fz / df** maps or z-profiles in tests or new code.

Import from `spammm.SPM.AFM_utils` (SSOT). If a helper is missing a knob, **extend that helper** — do not copy-paste a new plot loop into the script.

Generic ESP/density heatmaps → skill:`centralized-plotting` (`spammm.plotUtils.plot_2d_scalar`).

---

## Conventions (USER-approved — do not rediscover)

### Tip vs probe (lever length L)

Physical PPM: probe (CO O-apex) hangs **L** below tip apex (metal / C).

| Quantity | Meaning |
|----------|---------|
| `probe_z` | O / interaction height above molecule (GridFF / FDBM sample plane) |
| `tip_z` | Tip apex = `probe_z + L` |
| `bond_length` / L | Typical **3.0 Å** (GUI sometimes 4.0 — always print L) |

**Rules:**

1. Label **both** tip and probe on every column (`afm_tip_probe_heights`, `plot_afm_Fz_df_threerow`).
2. Sample **Fz unrelax** and report **df** at the **probe** plane — not at tip_z. Tip_z ≳ 5–7 Å is vacuum → blank maps.
3. `scan_fdbm(probe_heights=…)` already places tip apex at probe+L internally. Do **not** add L a second time.
4. K_LAT: human/GUI = **N/m**; internal = **eV/Å²** (`stiffness_Nm_to_eVA2`). Hapala 0.5 N/m ≈ 0.031 eV/Å².

### Oscillation amplitude vs Fz / df

`compute_df_amp(Fz, dz, amp)` uses **peak** amplitude `amp` (half of peak-to-peak).

```
df(z) averages Fz over [z − amp, z + amp]
closest approach ≈ probe_z − amp
```

| Default | Value |
|---------|-------|
| `amp` | **1.0 Å** (peak) |

**Pitfalls (already hit this session):**

- Computing df on a **short coarse** height stack (`mode='nearest'`) makes high-z columns inherit contact from low-z edges → df looks “way too close”.
- **Fix:** dense PP scan with `dz≈0.1` over `[probe_min−amp, probe_max+amp]`, then extract df at display probe heights.
- Expect df contrast to look ~`amp` closer than Fz in the same column — physical, not a lever bug.
- For fair Fz↔df visual match, use smaller `amp` (e.g. 0.5) or label columns by closest approach.

### Recommended z ladders (pyridine / AFM)

| Use | Tip_z | Probe_z (= tip−L, L=3) |
|-----|-------|-------------------------|
| Channel z-profiles | — | 1.6…6.0 (E), 1.5…8 (V) |
| AFM 3-row images | **5.5…8.0** step 0.5 | **2.5…5.0** |
| Zero refs | — | E−E(6), V−V(8); yellow band z≥2.5 |

Helper: `afm_tip_probe_heights(5.5, 8.0, 0.5, bond_length=3.0)`.

### Probe sites (any molecule)

- **Do not hardcode 3 sites.** Pass any list to `normalize_probe_sites` / `plot_fdbm_vs_kriging_zlayout`.
- Pyridine helper: `fdbm_probe_sites_nch` → N, farthest **C** (not H), farthest H.
- General: `fdbm_probe_sites_from_indices(apos, indices, names=…)`.
- Always print `xy=` when diagnosing V/E site cuts. “Opposite C” = farthest carbon from heteroatom.

### Spatial maps

- **`aspect='equal'`** always (1 Å x = 1 Å y/z). Never `aspect='auto'`.
- Tip density diagnostics: molecular frame **before** pad+roll; mark 1D cut on 2D; draw cube box if cube source.
- **Same XY window size** when comparing methods: same Δx×Δy. Center on **that simulation’s molecule COM**.
- For fair Kriging↔FDBM AFM: FDBM **GridFF** `margin_xy` ≥ **5.5–6 Å** (current default 4 Å → XY≈12.7 < Kriging≈13.7×14.7). PP scan should use **full GridFF XY** (Kriging does); the old ~8×8 Å imaging window was only ~2 Å past atoms.
- Overlay **`apos` = molecule atoms only** (cyan dots). Never pass Kriging `points_clean` sample sites as `apos`.

### Tip × sample labeling

Always state **sample** and **tip** separately (DFT-cube vs DFTB). Matched vs cross combos — see `doc/Reports/Kriging_DFT_vs_DFTB_FDBM_pyridine.md`.

---

## SSOT API (`spammm/SPM/AFM_utils.py`)

| Need | Call |
|------|------|
| One XY map on an Axes | `imshow_afm(ax, arr_nxny, extent=..., cmap='bwr')` |
| Row of maps at selected heights | `plot_afm_height_panel(data, heights, iz=..., extent=..., label='Fz', fname=...)` |
| **3-row Fz_u / Fz_r / df × nz** | `plot_afm_Fz_df_threerow(..., scan_xs=, scan_ys=, view_extent=, apos=atoms)` |
| Crop scan to shared XY | `crop_afm_xy(data, scan_xs, scan_ys, view_extent)` |
| Tip/probe height ladders | `afm_tip_probe_heights(tip_min, tip_max, tip_step, bond_length)` |
| Full height dump (all z) | `save_afm_images(df, scan_xs, scan_ys, heights, out_dir, prefix='df')` |
| Dense multi-row height grid | `plot_grid_Fz(Fz, heights, label, fname, x_ext=..., y_ext=...)` |
| 1D E(z) / Fz(z) / df(z) | `plot_afm_z_profiles(z, profiles, ylabel=..., title=..., fname=...)` |
| **FDBM vs Kriging z-layout (n sites)** | `plot_fdbm_vs_kriging_zlayout(sites, origin, step, V_ES, E_pauli, E_es, E_vdw, E_tot, …)` |
| **Multi-method 4-panel overlay** | `plot_fdbm_methods_zcompare_4panel(methods, sites_per_method, …)` |
| Normalize / build sites | `normalize_probe_sites`, `fdbm_probe_sites_from_indices`, `fdbm_probe_sites_nch` |
| Sample field along z at XY | `sample_field_z_profile(F, origin, step, xy, zs, zref=6)` |
| Extent from scan axes | `scan_extent(scan_xs, scan_ys)` |

### Canonical FDBM z-layout

User-approved: **normalize** E←E(6 Å), V←V(8 Å); yellow band z≥2.5 Å. Bottom row has **one panel per site** (any count).

| Row | Panels | Lines |
|-----|--------|-------|
| Top | V_ES (±0.2) · E_es (±0.05) · E_Pauli(dashed)+E_tot(solid) (±0.1, lw=0.5) | sites as colors |
| Bottom | **one Axes per site** (±0.1) | Kriging, E_tot, E_Pauli, E_es, E_vdW |

```python
from spammm.SPM.AFM_utils import (
    fdbm_probe_sites_nch, fdbm_probe_sites_from_indices, normalize_probe_sites,
    plot_fdbm_vs_kriging_zlayout, plot_afm_Fz_df_threerow, afm_tip_probe_heights,
)

sites = fdbm_probe_sites_nch(apos, Zs)  # or fdbm_probe_sites_from_indices(...)
plot_fdbm_vs_kriging_zlayout(
    sites, origin, step, V_ES, E_pauli, E_es, E_vdw, E_tot,
    kriging_E=..., A_pauli=A, beta_pauli=beta, fname='layout.png', save_dir=out)

tip, probe = afm_tip_probe_heights(5.5, 8.0, 0.5, bond_length=3.0)
plot_afm_Fz_df_threerow(Fz_u, Fz_r, df, tip, probe, amp=1.0, bond_length=3.0, ...)
```

Array convention: AFM volumes are **`(nx, ny, nz)`**. Helpers transpose for `imshow` unless `transpose=False`.

## STOP Triggers (blocking)

- About to type `ax.imshow(` or `plt.imshow(` for Fz/df/E → **STOP** → `imshow_afm` / `plot_afm_height_panel` / `plot_grid_Fz`
- About to type `ax.plot(z, Fz[` or `df[` for a height curve → **STOP** → `plot_afm_z_profiles`
- About to paste a **FDBM vs Kriging / site-decomposition** z-layout → **STOP** → `plot_fdbm_vs_kriging_zlayout`
- About to hardcode N/C/H-only bottom row for a new molecule → **STOP** → pass arbitrary `sites`
- About to sample Fz/df at tip_z for AFM images → **STOP** → use probe_z; label tip=probe+L
- About to call `compute_df_amp` on ≤5 coarse slices → **STOP** → dense z stack covering ±amp
- About to pass Kriging `points_clean` as `apos` overlay → **STOP** → molecule atoms only
- About to compare Kriging vs FDBM AFM without shared `view_extent` → **STOP** → crop to FDBM (or common) XY
- About to paste a height-panel loop from another test → **STOP** → call SSOT, or generalize SSOT

## Pre-commit self-check

```bash
rg 'imshow\(' tests/SPM/ --glob '*.py'   # new hits outside AFM_utils = bug
```

## Related

- skill:`code-reuse` — inventory-first, no new plot helpers in scripts
- skill:`centralized-plotting` — generic 2D scalar / ESP (`plotUtils`)
- skill:`visual-debugging` — L2 PNG paths under `debug/<script>/`
- `doc/Tasks/Import_KrigingGridFF.md` — tip×sample matrix + clamp Δρ recipe
- `doc/Reports/Kriging_DFT_vs_DFTB_FDBM_pyridine.md` — physics session report
