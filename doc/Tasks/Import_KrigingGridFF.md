# Task: Import PPAFM Kriging / RBF z-scan → GridFF

**Status:** investigating  
**Priority:** P1 (nc-AFM — `doc/ARCHITECTURE_ROADMAP.md` §TOC)  

**Human ToDo:** item 2  
**Parent:** `doc/Tasks/RepoConsolidation.md`  
**Export SSOT:** `/home/prokop/git/ppafm/docs/export/interpolation.export.md`  
**Science / data map:** `doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md`

## Objective

Bring **Kriging / RBF interpolation of DFT AFM z-scans** into SPAMMM so sparse tip–sample samples become a regular 3D GridFF (`E, Fx, Fy, Fz`) usable by `AFMulator` / PP relax — without depending on ppafm’s scattered `tests/Interpolation/` CLIs.

Longer goal (same task family): **compare** Kriging DFT GridFF to **FDBM from Psi4/pySCF cubes** (then fit \(A,\beta\)). That comparison is meaningless unless grids are aligned — see **CRITICAL** below.

---

## Session report — 2026-07-20 (detailed)

> Do **not** mark Done. USER confirmed pyridine DFTB AFM looks more sane when flat + correct K units path exists, but **cube-FDBM ES vs Kriging remains broken**, and **PP stiffness may now be too soft** after the N/m fix. Status stays **investigating**.

### What we implemented today

| Area | Code / artifacts | Purpose |
|------|------------------|---------|
| Kriging port (earlier + today) | `spammm/SPM/{interpy,InterpolatorKriging,InterpolatorRBF,KrigingGridFF}.py` | DFT z-scan → GridFF |
| Cube → FDBM | `AFM_utils.get_density_from_cube`, `build_fdbm_grid_from_cubes` | Psi4 `Dt.cube` sample+tip → Pauli/ES/vdW |
| Compare CLI | `tests/SPM/testplot_kriging_vs_fdbm_cube.py` | N-h + CO_O profiles, XY, Pauli fit |
| GPU density project | `kernels/grids.cl`, `spammm/utils/GridsOCL.py`, `tests/utils/test_grids_ocl.py` | Charge+dipole-preserving resample (vs scipy sample that broke ∫Δρ) |
| Soft core clamp | `prepare_delta_rho_clamped` / `soft_clamp_density` | Kill nuclear cusps on Δρ for ES (no double-count recipe) |
| DFTB control | `testplot_fdbm_relax.py` + ad-hoc runs | GUI-default FDBM for flat pyridine vs benzene |
| PP K units in test script | `testplot_fdbm_relax.py --K_LAT` now **N/m** → `stiffness_Nm_to_eVA2` | Match GUI; Hapala 0.5 N/m ≈ 0.031 eV/Å² |

**Preferred compare pair:** sample `N-h` (pyridine) + tip `CO_O` (O-down). Probes: **N** and **para-C**.  
Artifacts: `debug/testplot_kriging_vs_fdbm_cube/N-h-CO_O/`, `debug/afm_fdbm_diag_pyridine_vs_benzene/`.

```bash
# Cube path (still scientifically broken on ES — see Open)
python tests/SPM/testplot_kriging_vs_fdbm_cube.py --endgroup N-h --tip CO_O

# DFTB GUI-like FDBM (sane ES order of magnitude when geometry flat)
SPAMMM_AFM_CPU_FFT=1 python tests/SPM/testplot_fdbm_relax.py \
  --xyz data/xyz/pyridine.xyz --basis mio-1-1 --step 0.1 --margin 4.0 \
  --tip-mode co --K_LAT 0.5 --h_min 2.0 --h_max 5.5 --h_step 0.25 \
  --outdir debug/afm_fdbm_diag_pyridine
# --K_LAT is N/m (converted internally)
```

---

### Challenges / bugs found (and status)

#### 1. FFT z-wrap on short boxes — **solved**
Short \(L_z\sim8\) Å → tip density wraps into sample → fake Pauli/ES at high \(z\).  
**Fix:** tall boxes (\(L_z\gtrsim18\)–32 Å); `build_fdbm_grid_from_cubes` rejects \(L_z<12\).

#### 2. Fake sample \(p_z\) / ES parabola — **solved (mechanism)**
Uniform monopole strip of leftover \(q\) on a **z-asymmetric** cell injects huge fake \(p_z\) (e.g. −4.5) → parabolic ES floor.  
Scipy `map_coordinates` resample also broke ∫Δρ (~0 → ~0.4 e).  
**Fix:** GPU trilinear *project* (preserves \(q,p\)); **z-symmetric** box about molecular plane; strip monopole only if \(|q|\) large.

Far-field ES after this: decays → 0 by \(z\sim8\)–12 (`z_profiles_ES_far.png`) — USER OK on that check.

#### 3. Pauli fit absorbing bad ES — **solved (procedure)**
Fitting Pauli to `Kriging − ES − vdW` let Pauli cancel wrong ES.  
**Default restored:** fit Pauli to **Kriging \(E\) only**; `--fit_residual` for old mode. Rebuild channels after fit.  
Pooled N-h/CO_O contact fit ≈ \(A\sim10.3\), \(\beta\sim0.82\) (unverified vs USER; defaults in `PAULI_FITTED_DEFAULTS['pyscf_6-31g*']` remain **Gaussian-tip** A≈39.5, β≈1.15 — **wrong for real CO**).

#### 4. Tip core-clamp destroying CO multipoles — **diagnosed / partially mitigated**
CO tip raw Δρ: **monopole ≈ 0**, **tiny dipole** \(p_z\approx +0.02\,e\cdot\mathrm{Å}\).  
After soft-clamp + NA rematch: \(p_z\to +0.50\) (**~25×**); \(|\Delta\rho|\) peak moves ~1.4 Å off O → independent `_pad_and_roll_co_tip` on tip_del misaligns tip ES vs tip_tot.  
**Effect @ \(z=3\):** \(E_\mathrm{ES}\) center **+0.55 eV** (clamp+indep roll) vs **−0.03 eV** (raw tip Δρ).  
**Lesson:** core-clamp is for **sample** nuclear cusps on all-electron cubes — **do not clamp tip the same way**; roll tip_del with **shared** tip_tot (O) peak.

#### 5. “Strong ES repulsion in FDBM but not DFT” — **OPEN (main science bug)**
Even after far-field \(p_z\) fix and tip-clamp diagnosis, **cube-FDBM** still does not reproduce Kriging’s attractive halo around pyridine at AFM heights. Control experiment:

| Pipeline | Density | NA | \(E_\mathrm{ES}@+3\) Å (COM) |
|----------|---------|-----|------------------------------|
| Kriging DFT z-scan | — | — | attractive halo (reference) |
| FDBM **Psi4 cubes** + Gaussian NA (± clamp) | all-electron cusps | spherical Gaussians σ≈0.5 | **large / wrong morphology** |
| FDBM **DFTB+** `get_density_from_dftb_dense` | valence AO projection | **DFTB ρ_NA** (orbital) | **~meV** (benzene & flat pyridine) |

**Conclusion:** problem is **not** “pyridine physics” and **not** the PP relax path. It is **cube Δρ construction** (all-electron SCF − crude Gaussian NA on a ~0.1 a₀ / AFM grid), i.e. **core / cusp treatment**. DFTB never sees all-electron cores → ES stays small and sane.

Hypotheses still open (ordered by likelihood):
1. **Gaussian ρ_NA** cannot cancel Psi4 nuclear cusps → residual core multipoles / near-field junk on coarse grids. Soft clamp helps far-field but **reshapes valence multipoles** (sample \(p\) changes; tip \(p\) exploded when clamped).
2. Wrong / inconsistent **charge convention** tip_del ↔ \(V_\mathrm{ES}\) (ESP.cube sign vs Poisson(Δρ) still muddy; valence corr(Poisson, ESP)≈−0.87 beyond 1.5 Å historically).
3. Tip Δρ still imperfect after project+roll even with raw tip.
4. Missing physics vs DFT reference (BSSE, dispersion quality) — secondary; would not create +0.5 eV ES walls alone.

#### 6. Tilted `data/xyz/pyridine.xyz` — **solved (geometry)**
Repo file was **~21° tilted** FireCore/`lvs` junk (initial commit). Made DFTB FDBM look “broken” (out-of-plane dipole, sheared maps).  
USER flattened+symmetrized → tilt **0°**. Mithun cube `N-h` atoms already flat.  
**Caveat:** always check Δz / plane tilt before blaming FDBM. Prefer Mithun `N-h` or flat `pyridine.xyz` for AFM.

#### 7. PP lateral stiffness units — **partially solved; recalibration open**
Internal PP springs are **eV/Å²**. Literature / GUI: **0.5 N/m** Hapala ≈ **0.031 eV/Å²** (`AFM.stiffness_Nm_to_eVA2`, `K_LAT_HAPALA_*`).  
`1 eV/Å² = 16.02 N/m`. Passing bare `0.5` as eV/Å² → **~8 N/m** (~16× too stiff) → no visible bend.  
Today: `testplot_fdbm_relax` CLI now takes **N/m** like the GUI. With 0.5 N/m + \(L=3\) Å: `|dr|_max` ≈ 1.5 Å @ 2.0, 0.86 Å @ 2.5 — **bending visible**.  
**USER (evening):** stiffness now **looks too soft**. Open: pick working \(K_\mathrm{lat}\) (maybe 1–2 N/m), verify bond length / FIRE, and ensure no **double conversion** anywhere. Do not silently change GUI SSOT without check.

#### 8. Wrong plot path confusion — **process caveat**
`debug/testplot_kriging_vs_fdbm_cube/GridFF_E_Fz_sameXY.png` (parent) was **old HHO+H2O_O**. Pyridine lives under **`N-h-CO_O/`**. Always open the pair subfolder.

---

### What works / what does not (snapshot)

| Check | Result |
|-------|--------|
| Kriging GridFF + PP relax (HHO demo) | USER: looks fine (z-range uncertain) |
| Cube FDBM far-field ES (\(p_z\), no parabola) | OK after z-sym + GPU project |
| Cube FDBM vs Kriging AFM-height morphology | **FAIL** — ES / total still wrong |
| DFTB FDBM pyridine (flat) vs benzene | ES ~meV; AFM df/Fz progression looks normal |
| Pauli \(A,\beta\) for real CO tip | Fit machinery exists; values **not USER-approved** |
| PP K_LAT units in GUI | OK (N/m) |
| PP K_LAT in `testplot_fdbm_relax` | Fixed to N/m; **magnitude may need retune** |

---

### Caveats for agents (do not forget)

1. **OpenCL:** always NVIDIA (`preferred_vendor='nvidia'`); Shell needs `required_permissions: ["all"]` or ICD shows PoCL only. Never report PoCL as GPU.
2. **All-electron cubes ≠ DFTB densities.** Do not treat Gaussian-NA Δρ as equivalent to DFTB ρ_scf−ρ_NA.
3. **Clamp only with multipole diagnostics** (\(q\), \(\mathbf{p}\) before/after). Tip clamp is dangerous.
4. **Roll tip_tot and tip_del with the same peak** (O / apex of ρ_scf), never independent \(|\Delta\rho|\) peak.
5. **K_LAT:** API internal = eV/Å²; human/GUI/test CLI = N/m. Print both.
6. **Geometry:** verify molecule is flat in XY before AFM diagnostics.
7. **Never mark Done** without USER confirmation + shown verification.

---

### Open issues (priority)

1. **P0 — Cube / pySCF(all-electron) Δρ for ES**  
   Replace or fix NA: better NA (basis-projected), ESP.cube path, Voronoi/core mask, or don’t use Gaussian NA on cuspy grids. Prove ES matches DFTB order of magnitude on same geometry (flat pyridine) before trusting Kriging compare.

2. **P0 — Tip ES recipe**  
   Raw tip Δρ + shared roll; confirm CO monopole~0, small dipole preserved through project.

3. **P1 — Retune PP \(K_\mathrm{lat}\)** after N/m fix (USER: too soft at 0.5 N/m in latest images).

4. **P1 — Pauli \(A,\beta\) for CO_O** vs Kriging contact; store separately from Gaussian-tip defaults.

5. **P2 — L0 gradient parity** for Kriging GridFF; Fukui 4th channel deferred.

6. **P2 — Attractive halo** at \(z\gtrsim3\): once ES sane, revisit vdW / BSSE vs DFT.

---

## Source (ppafm) — do not reinvent

| Piece | Path |
|-------|------|
| Wendland C2 + distances | `ppafm/pyProbeParticle/interpy.py` |
| Ordinary Kriging | `ppafm/pyProbeParticle/InterpolatorKriging.py` |
| RBF | `ppafm/pyProbeParticle/InterpolatorRBF.py` |
| z-scan → volume | `ppafm/tests/Interpolation/interp_zscan_to_grid.py` |
| volume + forces → npy GridFF | `ppafm/tests/Interpolation/interp_zscan_to_grid_and_ff.py` |
| Point augmentation | `ppafm/tests/Interpolation/augment_points_with_grid_8.py` |
| E2E relax | `ppafm/tests/Interpolation/test_relax_kriging.py` |
| Guides | `BetterInterpolation_*.md`, `PPAFM_KiringInterpolation_Integration.md` |

Physics: compact Wendland C2 covariance; per-point adaptive support radii; analytic Fx,Fy; Fz via Δz; output axis order **[z,y,x]** (ppafm convention) — map carefully to SPAMMM GridFF.

## SPAMMM targets

| Role | Where |
|------|-------|
| Library | `spammm/SPM/{interpy,InterpolatorKriging,InterpolatorRBF,KrigingGridFF}.py` |
| Cube → FDBM | `AFM_utils.get_density_from_cube`, `build_fdbm_grid_from_cubes` |
| GPU project | `spammm/utils/GridsOCL.py` + `kernels/grids.cl` |
| PP relax | `AFMulator.setup_fdbm_grid` — `F (nx,ny,nz,4)=(Fx,Fy,Fz,E)` |
| Tests | `tests/SPM/testplot_kriging_relax.py`; `testplot_kriging_vs_fdbm_cube.py`; `testplot_fdbm_relax.py` |
| GUI | Later: AFM extension “Load z-scan → GridFF” |

## Work plan

1. ~~Port `interpy` + Kriging/RBF into SPAMMM~~  
2. ~~Port z-scan → volume + forces; kcal/mol → eV~~  
3. ~~Wire to AFMulator; Mithun symlink data~~  
4. ~~Cube density + Gaussian ρ_NA; FDBM-from-cubes path~~  
5. **In progress:** alignment OK-ish; **cube ES vs Kriging still wrong**; DFTB control proves NA/core issue; fix Δρ/ESP; then \(A,\beta\); retune \(K_\mathrm{lat}\)  
6. Fukui 4th channel — deferred  

### Preferred pair for Pauli \(A,\beta\) fit

| Role | Mithun name | Why |
|------|-------------|-----|
| Sample | **`N-h`** (= pyridine, C₅H₅N) | Symmetric; N vs para-C probes |
| Tip | **`CO_O`** (O-down) | Small tip multipole vs H₂O; contact zoom ≈ Pauli |

Data: `points_clean/N-h_points_clean.txt`, `results/N-h-CO_O.dat`, cubes `mithun_afm_tip_fukui/neutral/{N-h,CO_O}/`.  
Flat geometry for DFTB tests: `data/xyz/pyridine.xyz` (USER-flattened 2026-07-20) or Mithun `endgroups/N-h.xyz`.

**Doubt / SSOT on Pauli defaults:** `AFM.PAULI_FITTED_DEFAULTS['pyscf_6-31g*']` = **A=39.53, β=1.1544** fitted with **Gaussian tip σ=0.7 Å**, not real CO. Re-fit per tip.

```bash
python tests/SPM/testplot_kriging_vs_fdbm_cube.py --endgroup N-h --tip CO_O
# default: --fit_pauli --fit_zmin 1.5 --fit_zmax 2.0
```

## Acceptance

- [x] Module in `spammm/`, not only under `tests/`
- [ ] Gradient parity test passes (L0 still open)
- [x] One relax→OutFz plot for HHO+H2O_O (USER: looks fine; z-range uncertain) — `debug/testplot_kriging_relax/`
- [ ] USER confirmation that Kriging vs **cube** FDBM comparison is scientifically usable (**blocked on ES/core**)
- [ ] DFTB FDBM flat pyridine progression reviewed (artifacts under `debug/afm_fdbm_diag_pyridine_vs_benzene/`) — awaiting USER on \(K_\mathrm{lat}\) feel
- [ ] USER review before marking Done

---

## CRITICAL: Grid alignment (Kriging ↔ FDBM)

**Notorious failure mode:** comparing XY maps when XY extents differ or numeric \(z\) labels disagree.

### Conventions (SSOT for this pair)

| Grid | XY | Z |
|------|----|---|
| **Kriging DFT** | From `points_clean` (+ pad); same lab frame as cube atoms (~0.01 Å) | DFT scan: \(z=1.6+i\cdot0.1\) Å → **1.6–6.0 Å** |
| **FDBM cubes** | Sample bbox + ≥4 Å margin; tip rolled apex at `(0,0,0)` | Sample atoms \(z\approx0\); **tall z-symmetric** box; tip peak at grid origin |

### Rules (always)

1. **Z-profiles first** — full \(E(z)\), \(F_z(z)\) before XY.  
2. **Same XY window** — intersection bbox only.  
3. Optional `--z_offset` — visual only; auto-Δz unreliable when amplitudes differ 10–100×.  
4. Large-\(z\) zero reference.  
5. **Never** compare PP OutFz until E/Fz sane.  
6. Prefer pair subdir artifacts: `debug/testplot_kriging_vs_fdbm_cube/N-h-CO_O/`.

### Diagnostic script

```bash
python tests/SPM/testplot_kriging_vs_fdbm_cube.py --endgroup N-h --tip CO_O
```

| File | Purpose |
|------|---------|
| `z_profiles_*.png` | E/Fz / ES far-field |
| `GridFF_E_Fz_sameXY.png` | E/Fz on intersection |
| `FDBM_channels_sameXY.png` | Pauli / ES / vdW |
| `tip_clamp_vs_raw_ES.png` | Tip clamp multipole smoking gun |
| `SUMMARY.out` | Numbers |

### Earlier findings (HHO-h-p_1 + H2O_O) — still relevant

| Issue | Result |
|-------|--------|
| First maps horrendous | FFT z-wrap — fixed tall box |
| XY lab frame | Cube ↔ points ~0.01 Å |
| AFM zoom | Kriging attractive; FDBM repulsive / ES-dominated — **same family as pyridine cube bug** |

---

## Notes

- Human ToDo “kiring” = **Kriging**.  
- Data: symlink only — `data/mithun_afm_scans`, `mithun_afm_scans_flat`, `mithun_afm_tip_fukui`.  
- Sample = endgroup; tip = `{endgroup}-{tip}.dat`.  
- **Do not mark Done without explicit USER confirmation.**
