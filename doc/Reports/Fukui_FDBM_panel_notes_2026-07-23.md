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

## 1b. ES asymmetry diagnostics (2026-07-24) — **investigating, not fixed**

**CLI:** `python run_spm.py es-diag --molecule PTCDA pentacene phtalo_1-dftb-relax phtalo_2-dftb-relax`  
**Outputs:** `debug/fdbm_fukui_panel_flat/<mol>/es_diag/`  
(`es_chain_native_cube.png`, `es_chain_fdbm_grid.png`, `tip_co.png`, `ES_ASYM.out`)

### Confirmed path
`V_ES = fft_poisson(Δρ)` with `Δρ = ρ_scf − ρ_NA(Gaussian)`; **not** `esp_N` / locpot cube.  
`E_ES = tip_Δρ ⊗ V_ES` (DFTB CO tip).

### Verdict (mirror metrics about mol COM)

| Suspect | Result |
|---------|--------|
| **CO tip** | **Innocent.** peak=(0,0,0), XY mX/mY ~ 1e−7, upright along z |
| **ρ_scf / Δρ morphology** | Nearly symmetric (mX ~ 0.02–0.1) |
| **V_ES = Poisson(Δρ)** | **Guilty.** Native cube already: mX ~ 0.6–1.1 (pentacene, phtalo); FDBM grid worse at z+5 |
| **DFTB V_ES control** | Symmetric (mX ~ 0.03–0.05 at z+5) → same tip/grid; sample Δρ multipoles differ |

### Root-cause evidence
1. **Same `fft_poisson_cpu` for cube and DFTB** — Poisson is not the differentiator.
2. **Differentiator = Δρ construction + grid transfer:**
   - **LEGACY (wrong, was used in Fukui panel):** Gaussian NA + `resample_field_to_grid` (scipy *sample*)
   - **SSOT (pyridine / now wired):** `delta_rho_clamp_compact_na` + `GridsOCL.project_density` (trilinear *scatter*, `kernels/grids.cl` atomic_add)
3. Helper: `AFM_utils.allelectron_cube_to_fdbm_grid` — used by `run_fukui_one`, `run_spm.py afm --cube`, `es-diag`.
4. **2026-07-24 deeper bisect (pentacene) — root cause of dipole (NO manual strip):**
   - Manual `strip_monopole_dipole` **DEFAULT OFF** again (symptom mask only).
   - **NOT the pySCF cube pair:** `Δρ = ρ_N − ρ_NA.cube` has `|p_xy|≈0.010`, native V mX@1≈**0.035** (clean).
   - **IS our NA subtraction:** `ρ_N − Gauss(σ=0.3)` `|p_xy|≈1.90`, V mX≈**0.80**; clamp→compact `|p_xy|≈1.47`, V mX≈**0.60**.
   - Decomposition: `p(ρ_N)≈p(ρ_NA_cube)` so they cancel; our Gauss/compact NA has wrong multipoles vs all-e ρ_N → leftover dipole.
   - Grid centering: our NA uses **corner** samples `origin+i·h`; GridsOCL project uses **centers** `origin+(i+½)·h`. Native Δρ dipole of GaussCORNER is convention-independent (q≈0). GaussCENTER vs corner-sampled ρ_N is worse (half-voxel core miss).
   - Nuclei↔sample: both conventions ~0.03–0.06 Å; pad Δx≈0.1 Å; face density nearly symmetric — not the main XY dipole.
   - Artifacts: `…/es_diag/dipole_origin_bisect.png`, `DIPOLE_ORIGIN.out`.
   - **USER (2026-07-24):** plot is the smoking gun — V(N−NA_cube) symmetric; V(N−Gauss CORNER) x-slope; V(N−Gauss CENTER) smoother but x+y slope; V(clamp) improved but not enough. Preserve in code + `doc/Caveats.md`.
   - Status: **investigating** — next should use cube `ρ_NA` (or match it), not dipole strip.

### Resample vs project (USER clarification)
| Method | Code | Conserves ∫ρ, p? | Use for Δρ? |
|--------|------|------------------|-------------|
| scipy sample | `resample_field_to_grid` / `map_coordinates` | **No** | **Forbidden** |
| trilinear scatter | `project_density_to_grid` → `GridsOCL` / `grids.cl` | **Yes** (in-bounds) | **Required** |

Still need a taller FDBM dest grid for AFM (native cube too short in z) — that transfer must be **project**, not sample.

### Fix candidates remaining (status: **investigating** — not fixed)
- **Preferred:** wire cube path to `Δρ = ρ_N − ρ_NA.cube` (when present), same spirit as DFTB orbital NA — **not** dipole strip.
- Clamp→compact + project remains better than Gauss+scipy-sample but **insufficient** (see §1b table).
- Cross-check `esp_N.cube` only after Δρ multipoles are clean.
- Global write-up: [`doc/Caveats.md`](../Caveats.md).

**Preserve:** regenerate with `plot_cube_delta_rho_na_origin_diag` / `es-diag` — do not delete `dipole_origin_bisect.png` workflow.

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

### Presentation (CLI default 2026-07-24 — panel may differ)

**`run_spm.py afm` / `smiles-afm` now:**

- Dense chem window for **df**: default **h = 3.7 → 4.7** Å, **Δz = 0.1**
- **Amp-align:** same columns show **Fz at h − amp** (default amp=1 → Fz **2.7 → 3.7**)
- Column titles: `df h=…` / `Fz@…`; `--no-amp-align` restores same-h (misleading) layout
- Title note via `plot_afm_variant_height_strip(..., amp_align=True)`
- Default plots: `compare_per_column.png` + `stage_*.png` (`--plots compare,stage`)

**Still open for Fukui `panel-fukui` / 3-row SSOT:**

- Add **3-row** panels: **Fz_unrelax · Fz_relax · df** (pyridine SSOT) for prolonged (and cube after ES fix)
- Bring panel-fukui onto the same amp-align + dense-z defaults (or document if kept coarser)

---

## 3. What is *not* claimed

- Cube ES bug is **not** marked fixed.
- Prolonged DFTB looking good does **not** validate the cube row.
- Panel prolonged = default `SLATER_TAIL_ZETA`, not per-molecule SA (except PTCDA A,β from earlier Ez fit).
