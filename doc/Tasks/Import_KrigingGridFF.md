# Task: Import PPAFM Kriging / RBF z-scan → GridFF

**Status:** investigating  
**Priority:** P1 (nc-AFM — `doc/ARCHITECTURE_ROADMAP.md` §TOC)  

**Human ToDo:** item 2  
**Parent:** `doc/Tasks/RepoConsolidation.md`  
**Export SSOT:** `/home/prokop/git/ppafm/docs/export/interpolation.export.md`  
**Science / data map:** `doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md`  
**Campaign report (2026-07-20…21):** `doc/Reports/Kriging_DFT_vs_DFTB_FDBM_pyridine.md` — physics findings, tip×sample matrix, dual-basis rules, next experiments.  
**Pauli day report:** `doc/Reports/Kriging_FDBM_PauliFit_pyridine_2026-07-21.md`.  
**Follow-on (site \(A,\beta\) maps + transferability):** `doc/Tasks/Pauli_A_beta_KrigingTransferability.md`.  
**Plot SSOT:** skill:`afm-plotting` (`spammm/SPM/AFM_utils.py` — arbitrary probe sites, tip/probe heights, FDBM z-layout, 3-row AFM).

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
| Soft core clamp | `delta_rho_clamp_compact_na` / `soft_clamp_rational` | All-electron Δρ SSOT (clamp → rebuild compact NA). Do **not** use removed `prepare_delta_rho_clamped` (rescale cube NA — inverted V_ES). |
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
| FDBM **Psi4 cubes** + Gaussian NA (± clamp) | all-electron cusps | spherical Gaussians σ=0.3 (default; σ≳0.6 worsens V_ES) | **large / wrong morphology** |
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
8. **Prolonged Slater = Pauli only.** Never normalize prolonged ρ; never use it for ES. Dual basis is intentional.
9. **Tip project margin:** crop CO tip with margin≲1 Å → fake \(q_{\Delta\rho}\sim-0.7\). Use ≥3–4 Å (or full AFM cell).
10. **\(V_\mathrm{ES}\) ≠ \(E_\mathrm{es}\):** \(E_\mathrm{es}=\mathrm{tip}_{\Delta\rho}\otimes V\); similar \(V(z)\) can still give different \(E_\mathrm{es}\).
11. **Probe sites:** “opposite C” = farthest **carbon** from N, never farthest of all atoms (that is para-H). Always print `xy=` on plot titles.

### Fukui pySCF density panel (USER 2026-07-23) — expand cube-FDBM tests

Beyond pyridine Mithun cubes, use **PBE/def2-SVP** sample densities at:

`/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/{mol}_PBE_def2-SVP/`  
with `rho_N.{cube,npy}`, `esp_N.{cube,npy}` (and A/C for pentacene/PTCDA).

| mol tag | XYZ in repo |
|---------|-------------|
| `pentacene`, `PTCDA` | `data/xyz/pentacene.xyz`, `PTCDA.xyz` |
| `azaindol_dimer`, `azaindol_isodimer` | `data/xyz/azaindol_*.xyz` |
| `benzoicacid_dimer`, `benzoicamid_dimer` | `data/xyz/benzoic*_dimer.xyz` |

**Workflow:** cube FDBM (reference) ↔ DFTB stock ↔ DFTB prolonged — see `ProlongedRadialBasis_DFTB.md` § molecule panel and `Pauli_A_beta_KrigingTransferability.md`. Paths are **top-level** under `pyscf_fukui_cluster/`, not only legacy `jobs/results/`.

**USER review of first panel run (2026-07-23) — notes only:** PTCDA **cube** row ES looks too strong / asymmetric (DFTB prolonged OK). Panel cube path used **Gaussian NA**, not clamp→compact-NA. Also df↔Fz apparent height shift ~1.4 Å with **amp=1.0 Å**. Full notes: `doc/Reports/Fukui_FDBM_panel_notes_2026-07-23.md`. Do not mark cube-FDBM Done.
12. **CRITICAL — `aspect='equal'`** on every spatial density/potential `imshow` (1 Å x = 1 Å y/z). Never `aspect='auto'`. Tip maps: molecular frame before pad+roll; mark 1D cut on 2D. See `NOTE_plot_axis_equal.out`.

---

### 2026-07-21 morning — GUI vs agent DFTB images (resolved mismatch)

USER: GUI pyridine AFM looks reasonable; agent strips looked wrong / too soft.

**Cause (not density, not K_LAT):** agent scripts used `PAULI_FITTED_DEFAULTS['mio-1-1']` = **A=155.33, β=1.5507**, while GUI spinboxes still ship **old** pentacene defaults **A=787.22, β=1.2371** (`AFMExtension.py`). Same ModularPipeline + flat pyridine:

| Setting | df range @ GUI heights 2.8–3.6 |
|---------|--------------------------------|
| GUI A=787 | `[−0.10, +1.02]` — atomic PP contrast |
| Fitted A=155 | `[−0.22, ~0]` — mostly attractive blob |

Also: GUI heights **2.8–3.6** (not 2.0–5.5); `compose_and_relax_total` **L=4.0 Å**; GUI `compute_df` (not `compute_df_amp`). With GUI match: `|dr|_max`≈0.5–0.9 Å at 0.5 N/m — bending present, not “too soft”.

Artifacts: `debug/afm_fdbm_diag_pyridine_gui_match/`  
**Action:** sync GUI spinbox ↔ `PAULI_FITTED_DEFAULTS` (USER pick which is SSOT); diagnostics must call **ModularPipeline with GUI params** when comparing to GUI.

### 2026-07-21 — Consistent same-tip compare + prolonged dual basis + CO tip / ES bend

USER liked `zprofiles_CONSISTENT_DFTB_pySCF_Kriging.png` (same Mithun `CO_O` tip; Pauli \(A,\beta\) fit to Kriging). Status stays **investigating**.

#### Dual basis / prolonged Slater (USER correction — SSOT)

**Wrong agent assumption (do not repeat):** rejecting prolonged ρ because \(q\approx322\neq N_e\), or trying to charge-normalize it, or using prolonged ρ for ES.

**Correct design (practical DFTB AFM trick):**

| Channel | Density | Rule |
|---------|---------|------|
| **ES** | stock short-basis Δρ → \(V_\mathrm{ES}\) | multipoles / neutrality must stay physical |
| **Pauli only** | prolonged / SA Slater projection of **same** SCF DM | long vacuum tails; **never** rescale ∫ρ |

- Prolonged ∫ρ is **not** meant to equal \(N_e\). Pauli \(A(\int\rho_s\rho_t)^\beta\) cares about **local** overlap; \(A,\beta\) absorb scale.
- Dual basis looks inconsistent but is intentional: improve short-basis DFTB AFM tails **without** re-SCF and **without** corrupting ES.
- Code comments: `make_slater_tail_species_list`, `get_density_from_dftb_dense(... projection_basis_ang=...)`, `testplot_fdbm_relax --ptcda-stock-vs-sa`, `doc/DFTB_basis_fit.md`, `doc/Tasks/ProlongedRadialBasis_DFTB.md`.

#### CO tip: DFTB pipeline vs Mithun `CO_O` (deep compare)

Artifacts: `debug/afm_fdbm_diag_pyridine_gui_match/COtip_DFTB_vs_Mithun_{XY_cuts,zprofiles,XZ}.png`, `SUMMARY_COtip_compare.out`.

| Quantity | DFTB pipeline tip | Mithun `CO_O` (compact NA, margin≥3 Å project) |
|----------|-------------------|--------------------------------------------------|
| \(q_\mathrm{tot}\) | ≈ **10** (valence; expected) | ≈ **14** (all-electron) |
| \(q_{\Delta\rho}\) | ≈ −0.015 | ≈ 0 (native cube) |
| \(p_z\) (rolled) | ≈ +0.15 | ≈ +0.022 |
| \(\|\Delta\rho\|\) peak | mild (~1) | huge nuclear cusps (~10²–10³ on native; ~15 on 0.1 Å grid) |
| \(\rho_\mathrm{tot}\) support | soft valence on ~8 Å tip box | compact cube ~4.2×4.2×5.4 Å — **OK**, density ≠ box edge |

**CORRECTION (USER right):** Mithun CO cubes are **not** “cut mid-tail”. Agent plotted a window past the cube (`z_O+4` > cube top ≈ `z_O+3.3`) and called exterior zeros “truncation” — that was **agent display fault**, not Mithun. Native check: density does not touch any face; ~1 Å vacuum margin. See `COtip_Mithun_NATIVE_cubes_CO_O_CO_C.png`, `NOTE_Mithun_cube_NOT_truncated.out`.

**Cropping trap (separate, real):** projecting tip onto a **tight** bbox (`margin=1` Å) dropped \(q_{\Delta\rho}\) to **≈ −0.71** (fake monopole). **`margin≥3` Å** or full AFM cell restores neutrality. pad+roll itself does not cut charge if dest ≥ tip support.

Always plot tip diagnostics in **native / molecular frame** with cube boundary drawn; never claim truncation without checking density-vs-edge.

#### Remaining ES problem (same tip, still serious)

With **same** Mithun `CO_O` tip, Pauli can look similar after \(A,\beta\) fit, but **\(E_\mathrm{es}\)** still differs: pySCF bends attractive much sooner than DFTB while **\(V_\mathrm{ES}(z)\)** along the axis looks only moderately different (both repulsive over N).

Tip↔sample swap on one cell (`Ees_tip_sample_swap.png`, tip project margin=4, \(q_{\Delta\rho}\approx0\)):

| \(E_\mathrm{es}\) @ N, \(z=2\) Å | Mithun tip Δρ | DFTB tip Δρ |
|----------------------------------|---------------|-------------|
| **pySCF** sample \(V\) | **≈ −0.48 eV** (bend) | ≈ −0.06 eV |
| **DFTB** sample \(V\) | ≈ +0.01 eV | ≈ −0.06 eV |

**Verdict:** the early attractive bend is **not** explained by tip monopole alone (that was the margin=1 artifact). It is the **coupling** of cuspy all-electron tip Δρ with pySCF/cube sample \(V\) near field. Soft DFTB tip × either \(V\), or Mithun tip × DFTB \(V\), stay mild. So fixing ES requires tip and/or sample Δρ/NA treatment on the cube path (core clamp / better NA / ESP.cube), not only “use same tip”.

#### Probe-site bug (V_ES “sign flip” over C — 2026-07-21)

USER correctly flagged: `Ees_tip_sample_swap.png` bottom-right \(V\) over “opposite C” was **negative**, while `samecell_V_ES_zprofiles_rc_compact_sweep.png` over opposite C was **positive** (DFTB). **This was NOT a Poisson sign change.**

| Plot | Site actually sampled | \(V_\mathrm{DFTB}@1.7\) (zero@8 Å) |
|------|----------------------|-------------------------------------|
| samecell | para-**C** `xy≈(0.00, 1.09)` atom i=3 | **+0.10** |
| Ees_swap (buggy) | para-**H** `xy≈(0.00, 0.00)` atom i=8 | **−0.15** |

Cause: `argmax` distance from N among **all** atoms picks the para-H, not the ring carbon. SSOT: farthest **carbon** (`Z==6`) from N — same as `_probe_sites` in `testplot_kriging_vs_fdbm_cube.py`.

Fixed `Ees_tip_sample_swap.png` now labels XY and includes a diagnosis panel (para-C vs para-H). Tip density maps: `COtip_DFTB_vs_Mithun_rho_drho.png` (ρ + Δρ z-profiles and XZ for both tips).

#### Same-cell / NA notes (earlier same day, still valid)

- Far-field \(V_\mathrm{ES}\) zero at \(z\sim8\) Å; Gaussian \(\sigma_\mathrm{NA}\gtrsim0.6\) puts NA into tip region; default \(\sigma=0.3\); compact core \(r_c\le1\) flat for tip \(z\ge2\).
- Do not confuse \(V_\mathrm{ES}\) plots with FDBM \(E_\mathrm{es}=\mathrm{tip}_{\Delta\rho}\otimes V\).


### 2026-07-21 evening — Pauli fit criteria split + residual N+C + HTML

Full write-up: `doc/Reports/Kriging_FDBM_PauliFit_pyridine_2026-07-21.md`.  
Gallery: `debug/afm_fdbm_diag_pyridine_gui_match/index.html`.

- Kept `_fit_pauli_powerlaw` (contact wall); added `_fit_pauli_powerlaw_residual` (Kriging−ES−vdW, z∈[2.5,5]).
- CLI `--fit_mode contact|residual`.
- Residual pools: N+C+H and N+C-only; AFM 3-row rebuilt; artifacts under `pauli_fit_kriging_3ways/fit_resid_{NCH,NC}/`.
- Status remains **investigating**.


### CO guinea-pig (2026-07-21) — Δρ recipe + tip Slater

**Why CO:** smallest tip with real multipoles; native cubes OK (`CO_O`/`CO_C`); DFTB CO valence tip in cache. Debug algorithms here **before** pyridine/sample.

**Distinguish densities:**

| Kind | Source | ∫ρ |
|------|--------|-----|
| **All-electron** | Psi4/pySCF Mithun cubes | ≈ ∑Z (CO: 14) |
| **DFTB valence** | `get_tip_densities` / Grid_dftb | ≈ ∑Z_val (CO: 10) |

#### (1) All-electron Δρ — soft-clamp then compact NA (in progress)

Code: `AFM_utils.soft_clamp_rational`, `delta_rho_clamp_compact_na`, compact profile `f=(1-(r/r_c)^2)^2` (`profile='r2'`).

1. Soft-clamp nuclear spikes: USER rational clamp (`y1`, `y2`) — same math as `test_utils.soft_clamp`.
2. Per nucleus, \(Q_{\mathrm{rem},i}=\int(\rho-\rho_c)\,dV\) inside sphere \(R\sim0.5\)–\(0.7\) Å.
3. Build ρ_NA with \(f=(1-(r/r_c)^2)^2\), charges so ∫ρ_NA = ∫ρ_c (“NA charge − clamped charge”); Δρ = ρ_c − ρ_NA.
4. **Neutrality:** |∫Δρ| must be ~0 (CO tip first run: \(|Q_{\mathrm{diff}}|\sim2\times10^{-7}\)).
5. Plot vs DFTB valence Δρ on a **common valence axis** (ignore core spikes in the view).

Artifact: `debug/afm_fdbm_diag_pyridine_gui_match/COtip_delta_rho_clamp_compact_vs_DFTB.png`.  

#### Sample pyridine / N-h — same recipe → V_ES, E_ES (2026-07-21)

Applied `delta_rho_clamp_compact_na` to Mithun `N-h` (∑Z=42 → Q_clamped≈30.6, |∫Δρ|≈1×10⁻⁷). Then Poisson → V_ES; E_es = tip_Δρ ⊗ V.

Artifacts: `sample_clamp_NA_Ves_Ees.png`, `sample_delta_rho_clamp_vs_DFTB.png`, `SUMMARY_sample_clamp_Ves_Ees.out`.

| Combo @ N, z=2 Å | E_es |
|------------------|------|
| DFTB_V ⊗ DFTB_tip | ≈ +0.02 eV |
| **clamp_V ⊗ DFTB_tip** | ≈ −0.03 eV (mild — close to DFTB) |
| clamp_V ⊗ clamp_tip | ≈ −0.47 eV (still bends) |
| raw_V ⊗ raw_tip | ≈ −0.48 eV (old bend) |

**Reading:** sample clamp makes V usable with a soft (DFTB) tip. The remaining E_es bend is still dominated by **cuspy all-e tip Δρ**, not sample V alone. Tip clamp params need more work (or use DFTB/prolonged tip for ES).

#### FDBM recompute from cubes (Pauli=full ρ, ES=Δρ) — 2026-07-21

Pipeline on `N-h` + `CO_O`:
- **Pauli:** full `ρ_scf` sample ⊗ tip (no NA subtraction)
- **ES:** `delta_rho_clamp_compact_na` on **both** sample and tip → \(V=\mathrm{Poisson}(\Delta\rho_s)\), \(E_\mathrm{es}=\Delta\rho_t\otimes V\)
- Pauli \(A=11.05\), \(\beta=0.85\) (prior contact fit; not re-tuned here)

Artifacts: `FDBM_cube_clamp_VES_maps_zprofiles.png`, `FDBM_cube_clamp_VES_and_channels.png`, `FDBM_cube_clamp_fields.npz`.

@ z=2 Å over N: \(V\approx+0.56\), \(E_\mathrm{Pauli}\approx+1.56\), \(E_\mathrm{es}\approx-0.47\) (tip clamp still drives ES bend), \(E_\mathrm{tot}\approx+0.83\).

#### Tip apex charge / V sign (USER 2026-07-21) — investigating

**Poisson convention:** `fft_poisson` does \(V_k=+4\pi C\,\rho_e/k^2\) — treats **electron** Δρ as if it were **positive charge**. So:
- \(V_\mathrm{code}>0\) over electronegative sites (N: electron excess) 
- True electric potential \(\phi\) (where **+ attracts e−**) is \(\phi\approx -V_\mathrm{code}\)

Site ordering @ z=2 (code): \(V_N=+0.56 > V_C=+0.15 > 0 > V_H=-0.11\) — electronegativity N>C, H δ+ is correct in this convention.

**CO tip problem:** vacuum side of O (toward sample) has \(\langle\Delta\rho_e\rangle<0\) for both raw and clamp tip → electron **depletion** → apex acts as **δ+**, not δ−. Hence \(E_\mathrm{es}<0\) (attractive) over N — opposite of expected δ− tip repulsion from electronegative N. Classic PP CO tip uses tipQs apex ≈ −0.1.

Artifacts: `FDBM_vs_Kriging_tip_charge_diagnosis.png`, `FDBM_vs_Kriging_tip_sign_flip.png` (E_es with \(-\Delta\rho_\mathrm{tip}\)), includes **para-H** probe.

#### Tip × sample matrix + canonical z-layout (2026-07-21)

**Always label tip and sample separately.** Previous “DFT” / “DFTB” figures were **matched** pairs:

| Label | Sample | Tip |
|-------|--------|-----|
| DFT×DFT | Mithun `N-h` cube (clamp Δρ → \(V_\mathrm{ES}\); full ρ Pauli) | Mithun `CO_O` cube (clamp Δρ ES; full ρ Pauli) |
| DFTB×DFTB | DFTB pyridine stock Δρ | DFTB CO (`get_tip_densities(..., backend='dftb')`) |

**Cross** (same Pauli split; A,β still sample-matched — may need re-fit after tip swap):

| Cross | Layout artifact |
|-------|-----------------|
| DFT-sample × DFTB-tip | `FDBM_sampDFT_tipDFTB_vs_Kriging_layout.png` |
| DFTB-sample × DFT-tip | `FDBM_sampDFTB_tipDFT_vs_Kriging_layout.png` |

Tip-swap overlays (fixed sample): `FDBM_cross_DFTsample_tipswap_4panel.png`, `FDBM_cross_DFTBsample_tipswap_4panel.png`. Note: `NOTE_tip_sample_combos.out`.

**Plot SSOT (do not reinvent):** `spammm.SPM.AFM_utils.plot_fdbm_vs_kriging_zlayout` / `plot_fdbm_methods_zcompare_4panel` / `fdbm_probe_sites_nch`. Skill:`afm-plotting`. Normalization E−E(6), V−V(8); top=sites overlapped, bottom=per-site channels ±0.1 eV.

#### (2) Prolonged Slater for Pauli — tip first

Prolonged STOs today mostly on **sample**. USER: tip is **more** important for Pauli overlap. Systematic SA on CO tip (tip / sample / both); tip-only precomputed may be enough. Dual basis unchanged — see `ProlongedRadialBasis_DFTB.md`, `DFTB_basis_fit.md`.

### Open issues (priority)

1. **P0 — All-electron Δρ clamp→compact NA (CO tip)**  
   Tune clamp/NA params until Δρ matches DFTB valence morphology on common axis; then sample.
   `build_fdbm_grid_from_cubes` now calls `delta_rho_clamp_compact_na` (not the removed rescale recipe).

2. **P0 — Tip prolonged Slater SA**  
   Fit prolonged tip ρ for Pauli; compare tip-only vs tip+sample vs sample-only.

3. **P0 — Cube / pySCF \(E_\mathrm{es}\) bend** (same tip)  
   Revisit after (1); was cuspy tip Δρ ⊗ pySCF \(V\).

4. **P1 — Prolonged Pauli properly** (sample path)  
   Stock \(V_\mathrm{ES}\) + prolonged ρ; fit \(A,\beta\); **no** ∫ρ rescale.

5. **P1 — Sync Pauli defaults** GUI vs `PAULI_FITTED_DEFAULTS`.

6. **P2 — L0 gradient parity** / Fukui 4th channel / attractive halo.

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
