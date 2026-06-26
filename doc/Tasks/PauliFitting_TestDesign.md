# Pauli Potential Fitting: Test Design Notes

## Current State Assessment

### SPAMMM: What Exists

**Test files:**
- `tests/SPM/test_afm_fdbm.py` — pytest FDBM pipeline tests (11 tests). Runs DFTB SCF → density → Pauli → ES → vdw → total force → PP relaxation → df. Tests check finiteness/positivity/integrals but **no comparison against any quantum reference energy**. The pipeline was reported as broken (degenerate df, constant Fz_relax) in TEST_RESULTS.md, with root causes identified (FIRE integrator commented out, r=0 division).
- `tests/SPM/plot_fdbm_potentials.py` — standalone diagnostic script. Plots XY slices, XZ cross-sections, and 1D E(z) curves above a carbon atom for Pauli/ES/vdW/Total. **No reference comparison** — just visualizes the FDBM fields.
- `tests/SPM/plot_fdbm_relax.py` — standalone diagnostic script. Full pipeline with DFTB-computed CO tip density (not Gaussian). Plots Fz_relax, df, tip displacement. **No reference comparison**.

**Pauli parameters in SPAMMM** (`spammm/SPM/AFM.py:1577`):
```python
PAULI_FITTED_DEFAULTS = {
    'mio-1-1':     {'A': 787.22, 'beta': 1.2371},   # fitted for pentacene atom 0
    '3ob-3-1':     {'A': 509.28, 'beta': 1.0586},   # fitted for pentacene atom 0
    'pyscf_sto-3g': {'A': 1.15, 'beta': 0.36},       # fitted for pentacene atom 0
}
```
These were **fitted for pentacene atom 0 only** (one carbon atom). They were never validated against other atoms, other molecules, or other quantum methods. The pySCF values look suspicious (A=1.15, beta=0.36 — very different from DFTB), suggesting possible unit or normalization issues.

### Key Problems Identified

1. **No reference energy comparison in any SPAMMM test.** The FDBM tests check "is it finite?" but never "does it match DFT?" The Pauli parameters are used blindly.

2. **Pauli parameters may be wrong.** They were fitted for one atom of pentacene against DFTB only. No cross-validation, no multi-method comparison. The pySCF params especially look suspicious.

3. **The FDBM pipeline was broken** (degenerate df, constant Fz_relax). Root causes identified but may not be fully fixed. The visual diagnostic scripts (plot_fdbm_*.py) may work for potential visualization even if the relaxation scan is broken.

4. **No C2H4.xyz file exists** in `data/xyz/`. Need to create one.

5. **pySCF integration is minimal** (`spammm/quantum/pySCF_utils.py`). Only has `evalHf()` (UHF energy) and `preparemol()` (geometry optimization). No density-on-grid evaluation, no DFT (RKS) support beyond what ModularPipeline attempts. Need to add proper pySCF density evaluation.

### FireCore: What Was Done (Reference Implementation)

The FireCore repo (`tests/tAFM/pyocl_fdbm/`) has a **rigorous Pauli fitting pipeline**:

1. **`run_dftb_zscan.py`** — Rigid z-scan: CO tip approaches molecule at specific atom positions. For each z-distance, runs DFTB+ single-point SCF on combined (molecule + CO) system. Saves `zscan_z.npy` and `zscan_energy_eV.npy`. Uses caching to avoid recomputation. Key details:
   - CO orientation: O at bottom (apex), C above at 1.13 Å bond length
   - z-distance = distance from O atom to target atom
   - z-range: 2.0–10.0 Å, step 0.15 Å
   - Energy shifted: `e_rel = e - e[-1]` (relative to farthest point)

2. **`fit_fdbm_pauli.py`** — Fits `E_pauli = A * overlap^beta` against DFTB z-scan reference. Key methodology:
   - Loads pre-computed FDBM grids (Pauli overlap field, ES, vdw)
   - Extracts z-profile at target atom position using `atom_to_grid_idx()`
   - Fit range: z ∈ [2.0, 3.0] Å (close-range where Pauli dominates)
   - **Two-stage fit**: (a) log-linear initial guess via `np.polyfit(log(overlap), log(E_ref), 1)`, (b) nonlinear `scipy.optimize.curve_fit` with bounds A∈[0,1e6], beta∈[0,5]
   - Reports A, beta, R², RMSE(fit range), RMSE(all z)
   - Per-atom plots (linear + log scale) and multi-atom summary

3. **`test_fit_pauli_pyscf.py`** — Same fitting but using pySCF as reference instead of DFTB. Calls `afm_utils.fit_pauli_parameters_pyscf()` which generates pySCF z-scan reference and FDBM grids on-the-fly.

4. **`compare_fireball_dftb.py`** — Compares Fireball vs DFTB densities and Pauli fields. Uses log-slope ratio method: `beta = slope(ln E_ref) / slope(ln overlap)`, `A = exp(intercept_ref - beta * intercept_overlap)`.

5. **`plot_transition.py`** — Visualizes attractive→repulsive transition at multiple z-heights. Shows Pauli/vdW/ES/Total energy components vs z for multiple target atoms.

### FDBM_fit.md (FireCore doc)

The `FDBM_Fit/FDBM_fit.md` document describes a **linear least-squares approach** for fitting multiple atom-type coefficients simultaneously:
- E = Σ_i P_i·V_Pauli(r_i) + L_i·V_London(r_i) + Q_i·V_Coulomb(r_i)
- Build feature matrix A (M configs × 2·N_types), solve via `scipy.optimize.lsq_linear` with bounds P>0, L>0
- Coulomb fixed by RESP charges (known)
- Can tune Pauli exponent α via outer 1D optimization
- Weighting scheme to exclude close-contact blowup

This is more general than the single-atom z-scan fit — it fits all atom types simultaneously from multiple molecular configurations on a substrate.

---

## Proposed Test Design: Rigorous Pauli Potential Fitting for C2H4

### Objective

Fit and validate the Pauli potential parameters (A, beta) for the FDBM model by comparing against **multiple quantum chemistry methods**. Use ethylene (C2H4) as a small, fast test system.

### Why C2H4 (Ethylene)?

- **Small**: 6 atoms, fast SCF for all methods
- **Two atom types**: C and H — tests per-type Pauli parameters
- **Planar molecule**: simple geometry, clear approach directions
- **No heteroatoms**: isolates Pauli repulsion (no strong electrostatics)
- **Available in both DFTB basis sets**: C and H are in mio-1-1 and 3ob-3-1

### Test System Geometry

```
C2H4 (ethylene), planar, D2h symmetry:
  C at (±0.667, 0, 0)
  H at (±1.234, ±0.937, 0)
```

Need to create `data/xyz/C2H4.xyz`.

### Reference Energy Curves: E(z) for CO tip approach

**8 curves** = 2 approach sites × 4 quantum methods:

**Approach sites:**
1. **Above Carbon** — CO tip descends along z-axis above a C atom
2. **Above Hydrogen** — CO tip descends along z-axis above an H atom

**Quantum methods:**
1. **DFTB mio-1-1** — tight-binding, fastest, existing DFTB+ integration
2. **DFTB 3ob-3-1** — tighter-binding with more elements, existing DFTB+ integration
3. **pySCF PBE/6-31G*** — DFT with GGA functional, moderate cost
4. **pySCF B3LYP/6-31G*** — DFT with hybrid functional, higher cost but gold standard

### Z-Scan Protocol (per method, per site)

For each (method, site) combination:

1. **Build combined system**: C2H4 + CO (O at apex, C above at 1.13 Å)
2. **Rigid scan**: translate CO along z, no relaxation
   - z-range: 2.0 to 8.0 Å (z = distance from O to target atom)
   - z-step: 0.15 Å (matching FDBM grid step)
   - ~40 points per curve
3. **Single-point SCF** at each z position
4. **Extract total energy**, compute relative energy: `E_rel(z) = E(z) - E(z_max)`
5. **Save**: `z.npy`, `E_rel.npy` per (method, site) combination

**Key detail**: The reference energy is the **total interaction energy** of the combined system (C2H4 + CO), not just Pauli. To isolate Pauli, we need to subtract the other components:
- `E_Pauli_ref(z) ≈ E_total(z) - E_ES(z) - E_vdW(z)`
- Or more simply: fit the **total FDBM energy** (Pauli + ES + vdw) against the **total DFT energy**, and extract Pauli parameters from the fit

### FDBM Side: Computing the Overlap and Energy Components

For each (method, site):

1. **Run FDBM pipeline** on isolated C2H4 with the given method:
   - DFTB mio/3ob: existing `get_density_from_dftb_dense()`
   - pySCF PBE/B3LYP: need `get_density_from_pyscf()` (may need to implement)
2. **Compute overlap**: `overlap_raw(x,y,z) = FFT_convolution(rho_sample, rho_tip)`
3. **Extract z-profile** at target atom grid position: `overlap(z) = overlap_raw[ix, iy, :]`
4. **Compute ES and vdw** components at same position
5. **Pauli model**: `E_pauli(z) = A * overlap(z)^beta`

### Fitting Strategy

#### Option A: Pauli-only fit (simplest, matches FireCore approach)

**Assumption**: At close range (z < 3 Å), Pauli dominates. ES and vdw are small for C2H4 (no heteroatoms, no charges).

```
E_ref(z) ≈ A * overlap(z)^beta    (in fit range z ∈ [2.0, 3.5] Å)
```

- Fit (A, beta) via log-linear initial guess + nonlinear `curve_fit`
- Fit range: z ∈ [2.0, 3.5] Å — where Pauli is the dominant repulsive component
- Validate: plot fitted E_pauli(z) vs E_ref(z) over full z-range

#### Option B: Total energy fit (more rigorous)

**Assumption**: E_ref = E_pauli + E_ES + E_vdw, where E_ES and E_vdw are computed from FDBM and only (A, beta) are free.

```
E_ref(z) = A * overlap(z)^beta + E_ES(z) + E_vdw(z)
```

- Subtract known ES + vdw: `ΔE(z) = E_ref(z) - E_ES(z) - E_vdw(z)`
- Fit: `ΔE(z) = A * overlap(z)^beta`
- This accounts for electrostatic and dispersion contributions

#### Option C: Multi-method simultaneous fit (most rigorous)

Fit a **single** (A, beta) pair that best matches all 4 methods simultaneously:
- Minimize Σ_methods Σ_sites |E_ref - E_fitted|²
- Tests whether a single Pauli parameter set works across quantum methods
- If it doesn't, it reveals method-dependent systematic differences

### What to Plot

**Main figure: 8-curve overlay**
- 2 subplots (above C, above H) or 1 subplot with 8 curves
- x-axis: z (Å), y-axis: E (eV)
- 4 methods × 2 sites = 8 curves
- Each curve: E_ref(z) from quantum method + E_fitted(z) from FDBM Pauli
- Log-scale inset for the exponential decay region

**Per-method diagnostic plots:**
- E_ref(z) vs E_pauli_fitted(z) — linear and log scale
- Residual: E_ref - E_fitted vs z
- E_ES(z) and E_vdw(z) shown separately to justify fit range

**Parameter summary table:**
| Method | Site | A | beta | R² | RMSE_fit | RMSE_all |
|--------|------|---|------|----|----------|----------|
| DFTB mio | C | ... | ... | ... | ... | ... |
| DFTB mio | H | ... | ... | ... | ... | ... |
| DFTB 3ob | C | ... | ... | ... | ... | ... |
| DFTB 3ob | H | ... | ... | ... | ... | ... |
| pySCF PBE | C | ... | ... | ... | ... | ... |
| pySCF PBE | H | ... | ... | ... | ... | ... |
| pySCF B3LYP | C | ... | ... | ... | ... | ... |
| pySCF B3LYP | H | ... | ... | ... | ... | ... |

**Cross-method comparison:**
- A and beta bar charts per method
- Overlay all 8 fitted curves to see if they converge
- If A/beta differ significantly between methods, the Pauli model is method-dependent

### Implementation Plan (NOT yet — notes only)

#### Step 1: Create C2H4.xyz
Standard ethylene geometry, planar, in xy-plane.

#### Step 2: Z-scan reference curves (4 methods × 2 sites = 8 scans)

**DFTB z-scan** (reuse FireCore's `run_dftb_zscan.py` pattern):
- For each basis (mio-1-1, 3ob-3-1):
  - For each target atom (one C, one H):
    - Loop z from 2.0 to 8.0 Å, step 0.15 Å
    - Build combined system (C2H4 + CO), run DFTB+ single-point
    - Save E(z)

**pySCF z-scan** (new, need to implement):
- For each functional (PBE, B3LYP) with basis 6-31G*:
  - For each target atom (one C, one H):
    - Loop z from 2.0 to 8.0 Å, step 0.15 Å
    - Build combined system (C2H4 + CO), run pySCF single-point
    - Save E(z)
- pySCF is slower than DFTB but C2H4+CO = 8 atoms, should be feasible

#### Step 3: FDBM overlap computation (4 methods × 1 molecule)

For each method:
- Run SCF on isolated C2H4 → get electron density on 3D grid
- Build CO tip density (Gaussian or DFTB-computed)
- FFT convolution → overlap_raw grid
- Extract z-profile at C and H atom positions

#### Step 4: Fit and plot

- Fit (A, beta) for each (method, site) pair
- Generate 8-curve overlay plot
- Generate parameter summary table
- Generate cross-method comparison plots

### Key Technical Concerns

1. **pySCF density on grid**: `pySCF_utils.py` currently only has energy evaluation. Need `dft.numint.eval_ao()` + `dft.numint.get_rho()` to compute density on a 3D grid. Or use `pyscf.dft.gen_grid.Grids` with a custom grid matching SPAMMM's grid spec.

2. **CO tip density**: The FireCore code supports both Gaussian tip (σ=0.7 Å) and DFTB-computed CO density. For the fitting comparison, we should use the **same tip model** across all methods. Gaussian tip is simplest and most consistent. The DFTB-computed CO tip may not be compatible with pySCF densities.

3. **Grid alignment**: The z-scan reference positions must align with FDBM grid points. Use `atom_to_grid_idx()` to find the nearest grid point, or interpolate the FDBM field at the exact atom position.

4. **Energy zero**: All curves should be shifted to `E(z_max) = 0` (relative energy). The absolute SCF energy includes monomer self-energies which cancel in the relative energy.

5. **Basis set consistency**: The Pauli overlap depends on the electron density, which depends on the quantum method. So the overlap itself is method-dependent. The fit absorbs this into (A, beta), but the physical interpretation is that different methods produce different densities and thus different overlap functions.

6. **Fit range sensitivity**: The fit range [2.0, 3.5] Å should be tested for sensitivity. Too close → Pauli blowup dominates (but also ES/vdW may become inaccurate). Too far → Pauli is small, noise dominates. The FireCore code used [2.0, 3.0] Å for pentacene.

7. **BSSE (Basis Set Superposition Error)**: For DFTB this is less of an issue, but for pySCF with localized basis sets, BSSE can affect the interaction energy. Counterpoise correction may be needed for rigorous comparison. However, since we're fitting Pauli parameters (not computing binding energies), BSSE mainly affects the reference curve shape, which propagates into (A, beta).

### What This Test Will Reveal

1. **Are the current PAULI_FITTED_DEFAULTS correct?** The current values were fitted for pentacene atom 0 only. C2H4 will test whether they generalize.

2. **Is the Pauli model (A * overlap^beta) adequate?** If the fit quality (R²) is poor for some methods, the functional form may be wrong. The overlap^beta model assumes a power-law relationship between density overlap and Pauli energy, which is an approximation.

3. **Are A and beta transferable across molecules?** If the fitted A and beta for C2H4 differ significantly from pentacene values, the parameters are molecule-dependent, which is a fundamental problem for the FDBM model.

4. **How do DFTB and pySCF compare?** If DFTB mio and 3ob give similar Pauli parameters, the tight-binding basis is not critical. If pySCF PBE and B3LYP differ significantly, the XC functional matters for the Pauli fit.

5. **Are C-site and H-site parameters different?** If yes, per-atom-type parameters are needed (which the FDBM model already supports via P_i coefficients). If no, a single (A, beta) pair suffices.

### Relationship to FDBM_fit.md Linear Approach

The FDBM_fit.md document proposes fitting per-atom-type coefficients P_i via linear least squares across many configurations. This is complementary:
- **Our z-scan approach**: Fits 2 global parameters (A, beta) for the Pauli functional form. Simple, transparent, good for validation.
- **FDBM_fit.md approach**: Fits per-type linear coefficients P_i using pre-computed density^alpha grids. More general, can handle multiple atom types and configurations simultaneously.

**Recommendation**: Start with the z-scan approach (simpler, directly comparable to FireCore). If parameters are not transferable, move to the FDBM_fit.md linear approach with multiple molecules/configurations.

### File Organization (proposed)

```
tests/SPM/
  test_pauli_fitting.py          — pytest: run z-scan, fit, assert R² > 0.95, RMSE < threshold
  plot_pauli_fitting.py          — standalone: generate 8-curve overlay + parameter table
  run_zscan_dftb.py              — DFTB z-scan for C2H4 (reusable for other molecules)
  run_zscan_pyscf.py             — pySCF z-scan for C2H4

data/xyz/
  C2H4.xyz                       — ethylene geometry

debug/pauli_fitting/             — all outputs (gitignored)
  zscan_dftb_mio/                — DFTB mio z-scan results
  zscan_dftb_3ob/                — DFTB 3ob z-scan results
  zscan_pyscf_pbe/               — pySCF PBE z-scan results
  zscan_pyscf_b3lyp/             — pySCF B3LYP z-scan results
  fdbm_grids_dftb_mio/           — FDBM overlap grids for DFTB mio
  fdbm_grids_dftb_3ob/           — FDBM overlap grids for DFTB 3ob
  fdbm_grids_pyscf_pbe/          — FDBM overlap grids for pySCF PBE
  fdbm_grids_pyscf_b3lyp/        — FDBM overlap grids for pySCF B3LYP
  fit_results.json               — all fitted parameters
  plot_8curves.png               — main 8-curve overlay
  plot_parameters.png            — A/beta bar charts
  plot_residuals.png             — residual plots per method
```

### pytest Assertions (for test_pauli_fitting.py)

```python
# Per (method, site) fit:
assert r2 > 0.95, f"{method}/{site}: R²={r2:.4f} < 0.95"
assert rmse_fit < 0.05, f"{method}/{site}: RMSE_fit={rmse:.4f} > 0.05 eV"
assert A > 0, f"{method}/{site}: A={A} must be positive (repulsion)"
assert 0.5 < beta < 3.0, f"{method}/{site}: beta={beta} outside physical range"

# Cross-method consistency:
# A and beta should be within factor ~3 across methods for same site
# (DFTB vs DFT vs hybrid will differ, but not by orders of magnitude)
```

### Open Questions

1. **Should we use Gaussian tip or DFTB-computed CO tip?** Gaussian is simpler and method-independent. DFTB CO tip is more physical but introduces a method dependence. **Recommendation**: Use Gaussian (σ=0.7 Å) for the fitting comparison to keep the tip model constant across methods.

2. **Should we counterpoise-correct pySCF energies?** BSSE affects interaction energies with localized basis sets. For 6-31G* on C2H4+CO (8 atoms), BSSE could be ~0.1–0.3 eV at close range. **Recommendation**: Start without CP correction. If pySCF curves look systematically shifted vs DFTB, add CP correction.

3. **What basis set for pySCF?** 6-31G* is a reasonable compromise. Could also test cc-pVTZ for convergence check, but it's much slower for the z-scan (40 SCF calculations). **Recommendation**: 6-31G* for initial tests, optionally cc-pVDZ as a second point.

4. **Should we also compute forces?** The z-scan gives E(z). Forces F(z) = -dE/dz would provide an additional validation channel (compare FDBM force gradient vs numerical derivative of DFT energy). **Recommendation**: Optional — add if E(z) fit is good but AFM images are still wrong.

5. **Should we test more molecules?** C2H4 is the starting point. If parameters are not transferable from pentacene to C2H4, we should test a series: CH4, C2H4, benzene, H2O, NH3. **Recommendation**: Start with C2H4, expand if needed.
