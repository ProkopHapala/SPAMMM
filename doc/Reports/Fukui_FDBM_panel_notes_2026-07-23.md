# Notes: Fukui FDBM panel — cube ES + df↔Fz height shift (2026-07-23)

**Status:** investigating — USER review. **No fix yet** (commit CLI / panel first).  
**Gallery:** `debug/fdbm_fukui_panel/` · CLI: `run_spm.py panel-fukui` · task: `doc/Tasks/ProlongedRadialBasis_DFTB.md`

---

## 1. PTCDA (and panel): cube row looks bad; DFTB prolonged looks reasonable

**USER observation:** On PTCDA especially, **DFTB stock / prolonged** AFM contrast is plausible; the **DFT-cube** row is not — electrostatics look **too strong** and **asymmetric**.

### What the panel actually did (cube path)

| Piece | Implementation in `run_fukui_one` |
|-------|-----------------------------------|
| Sample ρ | `rho_N.cube` via `get_density_from_cube` |
| ρ_NA / Δρ | **Gaussian** NA (`sigma_na=0.3`), rescale to ∫ρ — **not** the pyridine clamp→compact-NA recipe |
| V_ES | `fft_poisson_cpu(ρ_diff)` after resample to FDBM grid |
| Tip | **DFTB CO** tip (same for all three rows) |
| Pauli | full cube ρ_scf ⊗ tip; A,β = `pyscf_6-31g*` defaults |

So the cube row is a **cross combo**: cube sample × DFTB tip. Asymmetry / huge ES is **not** surprising given prior pyridine work.

### Likely causes (hypothesis list — do not “fix” without diagnostics)

1. **All-electron ρ − crude Gaussian ρ_NA** on ~0.15 Å grids → cuspy Δρ, wrong multipoles, strong far-field V (same class as `Import_KrigingGridFF` / pyridine Phase B–F).
2. **No soft-clamp / compact-NA rematch** on Fukui cubes (pyridine SSOT: clamp cores → compact \(f=(1-(r/r_c)^2)^2\) NA → ∫Δρ≈0).
3. **Tip×sample mismatch:** DFTB tip Δρ ⊗ cube V can amplify sample multipole errors (pyridine: tip clamp / apex δ+ issues).
4. **Monopole strip / z-box** after resample — check \(q_{\Delta\rho}\), \(p_z\), far-field V→0 by z∼8–12.
5. **Less likely as sole cause:** Pauli A,β scale (would change magnitude more uniformly; USER points at ES morphology / asymmetry).

### Next diagnostics (after commit — not done now)

- Stage plots: E_ES / V_ES XY+XZ for cube vs stock on PTCDA.
- Multipole print: ∫Δρ, \(p\), apex tip Δρ.
- Re-run cube with pyridine Δρ recipe (`delta_rho_clamp_compact_na`) + optional ESP cube cross-check (`esp_N`).
- Same tip×sample labels on every figure.

**Related:** `doc/Reports/Kriging_DFT_vs_DFTB_FDBM_pyridine.md` § cube ES / tip×sample; `doc/Tasks/Import_KrigingGridFF.md`.

---

## 2. df vs Fz “height shift” (~1.4 Å) — chemical contrast window

**USER observation (PTCDA / panel strips):**

| Channel | Heights where bond / atom / group contrast evolves (experimentally relevant) |
|---------|-------------------------------------------------------------------------------|
| **df** | roughly **h = 4.3 → 5.3 Å** (bond sharpening clear ~**4.5 Å**) |
| **Fz** (same strip) | similar evolution roughly **h = 2.9 → 3.9 Å** (sharpening mainly **below ~3.0 Å**) |

Apparent shift ≈ **1.4 Å**. Misleading if columns are compared as “same height” without stating what df samples.

### Oscillation amplitude used in this panel

| Parameter | Value |
|-----------|--------|
| `amp` | **1.0 Å** (peak; half of peak-to-peak) |
| `compute_df_amp` | Giessibl-style weighted average of ∂Fz/∂z over **[z − amp, z + amp]** |
| Closest approach during df(z) | ≈ **probe_z − amp** |
| Height step in panel | **0.4 Å** (coarse for amp conversion; pyridine fix used dense **dz≈0.1**) |

Code: `spammm/SPM/AFM.py` → `compute_df_amp`; panel via `_run_from_density` → `amp=args.amp` (default 1.0).

**Easiest explanation (primary):** df at labeled height \(h\) is **not** Fz(\(h\)); it mixes Fz over \(h\pm 1\) Å with weight that emphasizes the **lower turning point**. So chemical contrast that lives in Fz near ~3.0–3.5 appears in df near ~4.0–4.5 (+ ~amp). A reported **~1.4 Å** shift vs amp=1.0 is in the right ballpark; leftover ~0.4 Å can come from (2)–(4) below.

### Other contributions (not mutually exclusive)

1. **Amplitude window (primary)** — as above; label columns by **closest approach** \(h_\mathrm{ca}=h-\mathrm{amp}\) when comparing Fz↔df, or plot with `amp→0` / small amp for fair morphology match.
2. **Coarse z for df** — panel `h_step=0.4` while amp=1.0; `map_coordinates(..., mode='nearest')` at stack edges. Pyridine already hit “df looks too close” with coarse stacks — prefer dense Fz then extract df (skill:`afm-plotting`).
3. **Only relaxed Fz in the strip** — `_run_from_density` plots **PP-relaxed** Fz from `scan_fdbm` + df from that volume. **No Fz_unrelaxed row** (unlike pyridine `plot_afm_Fz_df_threerow`: Fz_u · Fz_r · df). Lateral relaxation sharpens bonds; can shift *apparent* contrast in z relative to unrelaxed Pauli wall.
4. **Per-image color scale** — relative contrast can make high-z df look “feature-rich” while absolute Fz at the same \(h\) looks empty; does not create the ~1 Å physics shift but confuses visual matching.
5. **Not tip vs probe mislabel here** — both df and Fz use the same `heights` probe ladder; L=3 is inside `scan_fdbm`, not double-counted. Still always print amp and L on figures.

### Presentation TODO (after commit — do not implement now)

- Resample chemical window **h = 4.3 → 5.3** (df) with **Δz = 0.1 Å**; for Fz morphology show **2.9 → 3.9** and/or annotate \(h_\mathrm{ca}=h-\mathrm{amp}\).
- Add **3-row** panels: **Fz_unrelax · Fz_relax · df** (pyridine SSOT) for prolonged (and cube after ES fix).
- Put **amp=… Å** and “df mixes Fz over ±amp” in every compare-strip title (`plot_afm_variant_height_strip`).

---

## 3. What is *not* claimed

- Cube ES bug is **not** marked fixed.
- Prolonged DFTB looking good does **not** validate the cube row.
- Panel prolonged = default `SLATER_TAIL_ZETA`, not per-molecule SA (except PTCDA A,β from earlier Ez fit).
