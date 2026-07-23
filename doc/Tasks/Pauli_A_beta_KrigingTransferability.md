# Task: Site-resolved Pauli \(A,\beta\) — Kriging of parameters + transferability

**Status:** investigating  
**Priority:** P1 (builds on pyridine Kriging/FDBM campaign)  
**Depends on:** `doc/Tasks/Import_KrigingGridFF.md`, `doc/Tasks/PauliFitting_TestDesign.md`  
**Reports:** `doc/Reports/Kriging_FDBM_PauliFit_pyridine_2026-07-21.md`, `Kriging_DFT_vs_DFTB_FDBM_pyridine.md`  
**Science schema:** `doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md`

## Objective

Treat Kriging-interpolated DFT approach curves as the **reference** for the FDBM Pauli term

\[
E_\mathrm{pauli}(\mathbf{r}_\mathrm{tip}) = A\,\Bigg(\int \rho_\mathrm{sample}(\mathbf{r})\,\rho_\mathrm{tip}(\mathbf{r}+\mathbf{r}_\mathrm{tip})\,d\mathbf{r}\Bigg)^{\beta}
\]

and answer two related questions when a **single global** \((A,\beta)\) cannot fit all sites:

1. **Parameter fields:** at chemically interesting lateral sites (atoms, bonds, ring centers), fit local \((A,\beta)\) (or \(A\) at fixed \(\beta\)) vs Kriging, then **Kriging / RBF-interpolate those parameters** in the plane — instead of (or in addition to) Kriging of raw \(E,F\).
2. **Transferability:** quantify spread of \((A,\beta)\) within atom / site classes (e.g. N vs C vs H; ring-center vs bond midpoint) and across molecules (pyridine → PTCDA / pentacene / …).

## Why

Pyridine day report already shows: H wants systematically larger \(A\) than N/C; one pooled \((A,\beta)\) is a compromise; residual fit can absorb bad ES into Pauli. Next step is **spatial / chemical structure of Pauli parameters**, not another global scalar.

## Current inventory (do not reinvent)

| Piece | Path | Notes |
|-------|------|-------|
| Kriging GridFF | `spammm/SPM/KrigingGridFF.py`, `InterpolatorKriging.py`, `InterpolatorRBF.py` | DFT z-scan → \((E,F)\) GridFF |
| Fitters | `AFM_utils._fit_pauli_powerlaw`, `_fit_pauli_powerlaw_residual` | contact wall vs AFM residual |
| Compare CLI | `tests/SPM/testplot_kriging_vs_fdbm_cube.py` | `--fit_mode contact\|residual` |
| Site probes | `fdbm_probe_sites_nch`, `plot_fdbm_rho_E_sites` | N/C/H patterns |
| Defaults | `AFM.PAULI_FITTED_DEFAULTS` | global per basis — not site maps |
| Dual basis | stock Δρ→ES; prolonged ρ→Pauli only | never charge-normalize prolonged ρ |

## Two workstreams

### (i) Kriging of \(A\) (and optionally \(\beta\)) instead of only \(E,F\)

1. Pick a dense-enough set of lateral probe sites on one molecule (atoms + selected bond midpoints + ring COGs).  
2. At each site, extract Kriging \(E(z)\) and FDBM overlap \(S(z)\) (or \(E_\mathrm{pauli}\) → \(S\)).  
3. Fit per site: prefer **fixed \(\beta\)** (from pooled N+C or literature) and map \(A(x,y)\); optionally free \(\beta(x,y)\) if stable.  
4. Fit modes stay explicit: **contact** (\(E_\mathrm{ref}>0\)) vs **residual** (\(E_\mathrm{Kriging}-E_\mathrm{es}-E_\mathrm{vdW}\)) — never mix criteria silently.  
5. Build a 2D field \(A(x,y)\) [and \(\beta(x,y)\)] via existing Wendland/Kriging interpolators; evaluate FDBM as \(A(x,y)\,S^\beta\).  
6. L2: map of \(A\), histogram by site class, z-curves at held-out sites.

### (ii) Transferability across sites and molecules

1. Same protocol on the **Fukui pySCF panel** (PBE/def2-SVP densities under `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/`):
   - older: `pentacene_PBE_def2-SVP`, `PTCDA_PBE_def2-SVP`
   - new: `azaindol_dimer_PBE_def2-SVP`, `azaindol_isodimer_PBE_def2-SVP`, `benzoicacid_dimer_PBE_def2-SVP`, `benzoicamid_dimer_PBE_def2-SVP`
   - local XYZ: `data/xyz/{pentacene,PTCDA,azaindol_dimer,azaindol_isodimer,benzoicacid_dimer,benzoicamid_dimer}.xyz`
2. Compare FDBM channels: **DFT-cube reference** vs **DFTB stock** vs **DFTB prolonged** (same campaign as `ProlongedRadialBasis_DFTB.md` § molecule panel).  
3. Report per-class mean / std / range of \(A\) (and \(\beta\) if free) — N/C/O/H and H-bonded dimer sites.  
4. Cross-apply: take \(A\) stats from molecule A, evaluate FDBM on molecule B vs that molecule’s DFT-cube (or Kriging if available).  
5. Decide whether production defaults should be **per element**, **per site class**, or stay global with documented RMSE.

**Note:** New dimer folders currently ship **neutral only** (`rho_N` / `esp_N`); pentacene/PTCDA also have A/C. For Pauli/FDBM use `rho_N` as sample density.

## Deliverables

- [ ] Script / library API: `fit_pauli_sites(...)` → table of sites + \((A,\beta)\) + fit metrics  
- [ ] Optional: `A_map` / `beta_map` GridFF-like 2D field + eval path  
- [ ] Transferability table + L2 figures (histograms, maps)  
- [ ] Short report `doc/Reports/` linking pyridine campaign  
- [ ] Recommendation for GUI defaults (global vs per-type) — USER must confirm

## Acceptance

- Site maps and cross-molecule table reviewed by USER.  
- Fit mode (contact vs residual) labeled on every figure.  
- Status stays investigating until USER confirmation — do **not** mark Done from code alone.

## Out of scope

- Fixing cube tip Δρ / ES morphology (tracked in `Import_KrigingGridFF.md`) — parameter maps may still absorb residual ES; label as such.  
- Replacing full 3D Kriging \(E,F\) GridFF for PP relax (this task is about Pauli params).  
- Elastic contact-surface (separate task).
