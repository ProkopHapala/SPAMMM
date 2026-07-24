---
type: Report
title: Cube FDBM electrostatics — Δρ / NA dipole asymmetry
tags: [FDBM, ES, cube, Δρ, NA, anisotropic, multipoles, investigating]
timestamp: 2026-07-24
status: investigating
---

# Cube FDBM ES — Δρ / NA dipole asymmetry

**Status:** investigating — **code + panels regenerated; awaiting USER visual confirmation; not marked fixed.**

**Handoff (1 page):** [`Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md`](Cube_ES_DeltaRho_NA_Codex_handoff_2026-07-24.md)

**Solution in production (default):**
1. Keep cube `(sx,sy,sz)` — never `mean` step.
2. Soft-clamp AE cores → compact NA with **`q_na_mode='element_mean'`** (not per-atom `Q_rem_i`).
3. Cube→FDBM via `project_density_to_grid(..., src_convention='nodes')`.
4. No dipole strip. `rho_NA.cube` optional diagnostic only.

Pentacene: native `|p_xy|` **0.284 → 0.018** e·Å (NA_cube control ≈ 0.014).

**Canonical plot:** [`debug/fdbm_fukui_panel_flat/pentacene/es_diag/dipole_origin_bisect.png`](../../debug/fdbm_fukui_panel_flat/pentacene/es_diag/dipole_origin_bisect.png)

**Reproduce:**
```bash
python run_spm.py es-diag --molecule pentacene \
  --outdir debug/fdbm_fukui_panel_flat
# or regenerate only the smoking-gun panel via AFM_utils.plot_cube_delta_rho_na_origin_diag
```

**Related:** [`doc/Caveats.md`](../Caveats.md) · [`Fukui_FDBM_panel_notes_2026-07-23.md`](Fukui_FDBM_panel_notes_2026-07-23.md) · [`Import_KrigingGridFF.md`](../Tasks/Import_KrigingGridFF.md)

---

## 1. What is the problem?

### Symptom (USER)
On the Fukui FDBM panel, the **DFT-cube** AFM / electrostatic row looks **strongly asymmetric** (L–R or diagonal background slope in V_ES / forces). The **DFTB** rows (same tip, same Poisson-style ES path) look nearly symmetric. Worst on large π-systems (pentacene, phthalocyanines, PTCDA).

### Physical path (confirmed)
```
V_ES = fft_poisson(Δρ),   Δρ = ρ_scf − ρ_NA
E_ES = tip_Δρ ⊗ V_ES
```
- We do **not** use `esp_N` / locpot as the production V for this bisect.
- Tip (DFTB CO) is **innocent** (mirror metrics ~1e−7, peak at origin after roll).
- Same `fft_poisson` for cube and DFTB → **Poisson solver is not the differentiator**.
- Differentiator = **how Δρ is built** on the cube (and how it is transferred to the FDBM grid).

### Required physical target (USER clarification)

The intended total charge density is

```text
Δρ_phys(r) = ρ_electron(r) − Σ_i Z_i g_i(r−R_i)
```

where every `g_i` is spherical, normalized, and centered on its nucleus. Its width changes only the near-core regularization; in a complete domain its monopole and first moment are exactly `Z_i` and `Z_i R_i`. Therefore:

- `rho_NA.cube` is useful only as an independent alias/alignment control.
- A residual far-field dipole in `ρ_SCF−Gaussian_nuclei` must come from the sampled SCF density, discrete core representation, atom/grid coordinates, or a non-symmetric input geometry.
- The objective is **not** to replace nuclear smearing by `rho_NA`; it is to make the analytical nuclear-smearing path charge-, moment-, and symmetry-correct.

### Why AFM is hypersensitive
PBC Poisson turns a small leftover **dipole** in charge-neutral Δρ into a far-field linear slope in V. Tip convolution makes that look like a catastrophic L–R force asymmetry even when |p| is only ~0.3 e·Å.

---

## 2. What we tried / resolved

| Attempt | Result | Keep? |
|---------|--------|-------|
| Blame tip | Tip XY symmetric | Tip OK |
| Blame Poisson | Same solver for DFTB (clean) vs cube (bad) | Poisson OK |
| Scipy `map_coordinates` resample cube→FDBM | Breaks ∫Δρ / multipoles | **Forbidden** for densities |
| Clamp cores → per-atom compact NA + GridsOCL **project** (legacy) | Better than Gauss+sample; still asymmetric | Hygiene, **not sufficient** |
| Manual monopole+dipole strip of Δρ | Improves far-field AFM; **masks** root cause | DEFAULT **OFF** (diagnostic only) |
| Corner vs center Gauss diagnostic | Both bad; different slope directions | Useful caveat; not root of 1.9 e·Å |
| “ρ doesn’t peak on nuclei” framing | **Rejected by USER** — correct instinct | Atoms must center NA |
| Atoms from external XYZ? | **No** — cube header only; N/NA headers identical | Ruled out |
| Collapse cube `(sx,sy,sz)` → `mean` | **Main quantitative bug** for \|p\|~1.9 | **Fixed in code** |
| Keep full anisotropic step for NA + Poisson | \|p\| 1.9 → **0.28**; V mX 0.80 → **0.42** | **Necessary, not sufficient** |
| Diagnostic control: `Δρ = ρ_N − ρ_NA.cube` | \|p\|~0.01; V mX~0.035 — cusp/grid errors cancel | Control only, **not required input** |
| Globally rescaling Gauss 146 → ∫ρ_N=145.453 e | Changes p by only ~0.0014 e·Å | Ruled out as cause of the 0.28 e·Å |
| Exact-symmetry C₂H₄ through same parser + Gauss subtraction | \|p_xy\|=`1.79e−4 e·Å` | Analytical subtraction itself passes |
| Carbon cusp peak vs sub-voxel distance | correlation `−0.99685`; peak range `201–344 e/Å³` | **Smoking gun: core aliasing** |
| Clamp SCF cores | SCF p_x `0.295 → 0.0098 e·Å` | Clamp removes the alias |
| Reuse each noisy sampled `Q_rem_i` as compact charge | noise moment \|p_xy\|=`0.28401 e·Å` | **Main reconstruction bug** |
| Use element-mean `Q_rem(Z)` for compensation | \|p_xy\| `0.28365 → 0.01823`; physical-center V mX `0.2756 → 0.0334` | **Implemented; L0 + panel rerun pass** |
| Project cube nodes as source voxel centers | Adds `q·step_src/2` after native construction | **Repaired in cube adapter; low-level center API unchanged** |

### Anisotropy fix (resolved in code, 2026-07-24)

pySCF cubes are often **~0.1–0.2%** axis-anisotropic. Old `get_density_from_cube` did:

```text
step = mean(sx, sy, sz)   # if aniso < 1%
Gauss NA on origin + i·step_mean
```

while ρ samples live at `origin + (i·sx, j·sy, k·sz)`. That **warps nuclei vs density** (up to ~0.2 voxel across pentacene) and injects a fake Δρ dipole.

**Now:** `step` is kept as `(3,)`; `make_gaussian_rho_na`, `fft_poisson_cpu`, GridsOCL project already accept `(3,)`.

**FDBM dest grids** (DFTB / `make_fdbm_grid_com_zsym`) remain **isotropic by design** — OK. Cube→FDBM must **project** with aniso `step_s` → iso `step_d`.

---

## 3. What we found (numbers — pentacene)

Native cube `229×117×54`,  
`step ≈ (0.07962, 0.07962, 0.07988) Å`, aniso ≈ 2.2e−3.

| Δρ recipe | \|p_xy\| (e·Å) | V mirror-X @ z_mol+1 |
|-----------|----------------|----------------------|
| **ρ_N − ρ_NA.cube** | **0.01422** | **0.03155** (diagnostic control) |
| ρ_N − Gauss, **mean step (old bug)** | **1.89** | **0.80** (horrible) |
| ρ_N − Gauss, **true (sx,sy,sz)** | **0.28** | **0.42** (still bad) |
| ρ_N − clamp→element-invariant compact (aniso) | **0.01823** | **0.03342** |
| ρ_N − Gauss CENTER (diag only) | ~8.4 | ~0.88 (worse) |

### USER visual judgment
After the aniso fix, V(N−Gauss) / V(clamp) **still do not look much better** than the original smoking-gun panel vs V(N−NA_cube). Metrics agree: V mX improved ~2× (0.80→0.42) but remains **~12× worse** than the NA_cube control (0.035). **Aniso fix alone does not solve the cube ES row.**

### Leftover \|p\|≈0.28 after aniso — corrected interpretation

The earlier “promolecule shape mismatch” conclusion missed the intended physics. A normalized spherical nuclear smear has the correct first moment independently of σ, so σ-insensitivity is expected. The decisive question is why the **sampled all-electron SCF cube** does not have the correct low-order moment.

The cube spacing is ~`0.15 bohr`, too coarse for first-row all-electron 1s cusps. Equivalent carbon atoms land at different sub-voxel phases. Their sampled peak heights and integrated removed-core charges therefore vary even when their physical core charge should be element-invariant.

Pentacene smoking-gun numbers:

| Quantity | Result |
|----------|--------|
| Carbon local peak range | `201.24–344.04 e/Å³` |
| corr(carbon peak, distance to nearest node) | **−0.99685** |
| `Q_rem(C)` from current clamp | mean `1.75194 e`, std `0.01458 e`, range `1.72805–1.78449 e` |
| corr(`Q_rem(C)`, distance to nearest node) | **−0.95969** |
| First moment of `Q_rem_i−mean_Z(Q_rem)` | `(0.27901, 0.05302, 0) e·Å`; \|p_xy\|=**0.28401** |

This equals the bad residual within numerical noise. The legacy per-atom clamp reconstruction did two opposing things:

1. `soft_clamp_rational` removes the aliased cusp: the SCF electronic `p_x` falls from `0.29518` to `0.00977 e·Å`.
2. `delta_rho_clamp_compact_na` then sets each compact charge to `q_i=Z_i−Q_rem_i`. Because `Q_rem_i` contains the sub-voxel quadrature noise, the compact NA obtains `p_x≈−0.2717 e·Å` and recreates `p_x≈+0.2815 e·Å` in Δρ.

Therefore the main bug is **feeding a sampled per-atom core quadrature error back into the analytical compensation charge**. Core compensation must be element-invariant (or come from an analytic element-specific core model), while the clamped SCF density retains the chemically meaningful valence redistribution.

### Parser / offset / axis bisect

The suspected gross coordinate errors were tested independently:

- `read_cube(rho_N.cube)` equals the independently written `rho_N.npy` bit-for-bit; reshape order is not the differentiator.
- The cube and PySCF convention is node-sampled `origin+i·step`. Carbon core maxima have node-coordinate RMS distance `0.0519 Å` (dominated by atoms halfway between the two z planes). Reinterpreting values as cell centers creates a systematic `(+0.0429,+0.0430) Å` XY displacement and larger RMS `0.0690 Å`.
- `(sx,sy,sz)` are now kept independently. Swapping x/y is incompatible with the grid extents and nuclear peak layout.
- The unscaled σ=`0.3 Å` Gaussian integrates to `146.000000 e` and reproduces the analytic nuclear first moment to numerical precision. The nuclear builder's node placement is correct.
- Per-atom stamp parity is far below the observed error: Gaussian max charge error `1.53e−7 e`, max local moment `8.47e−7 e·Å`; compact (`rc=0.6 Å`) max charge error `8.13e−5 e`, max local moment `1.53e−5 e·Å`. Stamp discretization cannot explain `0.284 e·Å`.
- Rectangle versus trapezoid integration changes pentacene \|p_xy\| only `0.283652→0.283693 e·Å`; endpoint weighting is not the cause.
- An almost exactly symmetric C₂H₄ cube passed the same reader, units, anisotropic step, and Gaussian subtraction with \|p_xy\|=`1.79e−4 e·Å`. This rules out a systematic parser, axis, unit, or Gaussian-centering failure.
- `mirror_asymmetry_2d` rounds the physical symmetry center to an integer pixel. On C₂H₄ it falsely reports V mirror `(X,Y)=(0.188,0.203)`, while interpolation about the physical center with the same max-error normalization gives `(1.38e−4,3.97e−4)`. Absolute mirror metrics must use physical-coordinate reflection; the moment bisect above is unaffected.

### Input geometry is not exactly symmetric

The pentacene coordinates in the cube header are only approximately D₂h:

| Atomic symmetry fit | RMS position residual | Max residual |
|---------------------|----------------------:|-------------:|
| mirror X | 0.0174 Å | 0.0286 Å |
| mirror Y | 0.0461 Å | 0.0990 Å |
| inversion | 0.0427 Å | 0.0712 Å |

Continuous physical-coordinate reflection of the z-integrated density (not integer-pixel flipping) gives:

| Field | mirror-X asym | mirror-Y asym |
|-------|--------------:|--------------:|
| raw `ρ_SCF` | 0.0263 | 0.0610 |
| clamped `ρ_SCF` | 0.00760 | 0.0197 |
| atom-centered nuclear Gaussians | 0.0159 | 0.0407 |

Thus two effects coexist: core sampling alias dominates the fake far-field moment, while the input geometry itself explains a smaller real loss of exact mirror symmetry. Exact density symmetry requires a symmetry-enforced geometry before SCF; grid code cannot manufacture a point-group symmetry absent from the nuclei.

### Focused moment decomposition (2026-07-24 re-test)

All moments below use the native cube **node** coordinates `origin + i·step` and the atom-position mean as the reporting origin.

| Field | q (e) | p_COM,xy (e·Å) | \|p_xy\| (e·Å) |
|-------|------:|-----------------|----------------:|
| `ρ_N` | 145.452722 | `(0.295184, −0.329281)` | 0.442222 |
| `ρ_NA.cube` | 145.391860 | `(0.304059, −0.318166)` | 0.440093 |
| Gauss, unscaled | 146.000000 | `(0.013996, −0.366589)` | 0.366856 |
| `ρ_N − ρ_NA.cube` | +0.060862 | `(−0.008875, −0.011116)` | **0.014224** |
| `ρ_N − Gauss`, unscaled | −0.547278 | `(0.281188, 0.037307)` | **0.283652** |
| `ρ_N − Gauss`, rescaled | ~0 | `(0.281241, 0.035933)` | **0.283527** |

The Gaussian rescale changes the analytic first moment by only
`(1−145.452722/146) Σ_i Z_i(R_i−COM) = (0.000052, −0.001374, 0)` e·Å.
It cannot explain the ~0.28 e·Å residual. The mismatch is already present in the under-resolved sampled `ρ_N`; `rho_NA.cube` merely carries nearly the same nucleus-phase alias and cancels it on subtraction.

The paired-cube result is not exactly charge-neutral (`q=+0.060862 e` for pentacene) because finite-grid cusp quadrature and tails do not cancel perfectly. `fft_poisson_cpu` explicitly sets `V_k[0,0,0]=0`, so this monopole does not explain the observed linear slope. A manual dipole strip remains unjustified.

### Clamp reconstruction A/B

For the existing automatic clamp (`y1=1.734`, `y2=3.468 e/Å³`, `R=rc=0.6 Å`):

| Reconstruction | \|p_xy\| (e·Å) | V mirror-X @ z+1 |
|----------------|----------------:|------------------:|
| Current per-atom `q_i=Z_i−Q_rem_i` | 0.283647 | 0.274714 |
| Element-invariant `q_Z=Z−mean_{i:Z_i=Z}(Q_rem_i)`, charge closed | **0.018228** | **0.039865** |
| Fixed valence pattern (H=1, C=4), charge closed | 0.024662 | 0.034396 |
| `ρ_N−rho_NA.cube` diagnostic | 0.014224 | 0.03492 |

The element-invariant result is not sensitive to one fine-tuned compact radius: at the automatic clamp, `rc=0.4,0.5,0.6,0.8 Å` gives \|p_xy\|=`0.0220,0.0169,0.0182,0.0179 e·Å`. Across `y1=0.5–20 e/Å³` and those radii it remains `0.006–0.033 e·Å`, versus `0.284 e·Å` currently.

The table's V column is the existing integer-pixel diagnostic for continuity with prior artifacts. A physical-coordinate reflection with identical max-error normalization gives pentacene `(mX,mY)=(0.2756,0.1662)` current versus **`(0.0334,0.1008)` element-invariant**. The remaining Y asymmetry is consistent with the much larger Y-geometry residual.

### Cube→FDBM projection convention bug (secondary)

The repository has two lattice conventions:

- Cube, DFTB density, Gaussian NA, and Poisson arrays are sampled at **nodes**: `origin + i·step`.
- `GridsOCL.project_density` is explicitly tested as **source centers → destination nodes**. Its kernel constructs `p_s = origin_s + (i+½)·step_s`.

The old `project_density_to_grid` passed cube `origin_src` unchanged. It therefore translated every cube source sample by `+step_src/2` before scatter. For a nonzero projected charge, the node first moment changed by the exact amount

```text
Δp = q · step_src/2
   = 0.0608616 · (0.079615, 0.079619, 0.079876)/2
   = (0.002423, 0.002423, 0.002431) e·Å       # pentacene paired Δρ
```

NVIDIA RTX 3090 A/B result on the `160×96×96`, `0.15 Å` FDBM grid:

| Path | q (e) | \|p_xy\| node convention (e·Å) | V mirror-X @ z+1 |
|------|------:|--------------------------------:|------------------:|
| Native `N−NA_cube` | 0.0608616 | 0.014224 | 0.0349 |
| Current project call | 0.0608616 | 0.010826 | 0.06815 |
| Node-correct emulation (`origin_src−step_src/2`) | 0.0608616 | **0.014224** | **0.02229** |
| Current clamp project | ~0 | 0.283645 | 0.26396 |

The apparently smaller old projected dipole was accidental cancellation, not correctness. The implemented adapter defaults to `src_convention='nodes'` and passes `origin_src−step_src/2` to the center-source kernel; explicit `src_convention='centers'` preserves the original API. The node-correct call preserves the native moment to ~`1.3e−7 e·Å`.

### Cross-molecule clamp reconstruction check

All eight paired cubes have exactly matching headers, so `rho_NA` remains a clean independent diagnostic. The important comparison is current per-atom clamp compensation versus the same clamped SCF density with element-invariant compensation:

| Molecule | current per-atom \|p_xy\| | element-invariant \|p_xy\| | `N−NA_cube` diagnostic | removed-core noise moment |
|----------|---------------------------:|-----------------------------:|------------------------:|--------------------------:|
| azaindol_dimer | 0.007864 | 0.007937 | 0.005828 | 0.015779 |
| azaindol_isodimer | 0.014244 | **0.002582** | 0.002138 | 0.016209 |
| benzoicacid_dimer | 0.008571 | **0.000958** | 0.000507 | 0.007683 |
| benzoicamid_dimer | 0.012147 | **0.002810** | 0.001019 | 0.015058 |
| pentacene | 0.283647 | **0.018228** | 0.014224 | 0.284007 |
| PTCDA | 0.008326 | 0.009922 | 0.002984 | 0.016599 |
| phtalo_1-dftb-relax | 0.017385 | **0.002108** | 0.000113 | 0.015383 |
| phtalo_2-dftb-relax | 0.033821 | **0.001586** | 0.000975 | 0.035547 |

The correction strongly improves six systems and keeps the other two small. This is evidence for the core-alias mechanism, not yet a production acceptance test; AFM images and physical-dipole preservation on intentionally polar molecules still require review.

---

## 4. Remaining problems (open)

1. **Element-mean compensation is implemented and removes the dominant alias**, but it still needs USER visual acceptance and a deliberately polar-molecule preservation test.
2. **Compact/Gaussian stamp error is not the present root cause**, but production should make the already-small per-atom charge/center invariants exact rather than rely on global rescaling.
3. **Clamp parameters are heuristic and global.** A production pseudization needs an element-specific core criterion that removes cusp alias without erasing valence/bond density.
4. **Charge closure is underdefined for a coarse all-electron cube.** Pentacene integrates to `145.4527` rather than 146 e. The correction must be distributed by an element/core rule, never by noisy per-atom phase or an unlabeled dipole strip.
5. **Pentacene input coordinates are not exactly point-group symmetric.** The residual physical geometry asymmetry must not be confused with grid alias.
6. **Cube node projection is repaired locally**; broader callers with genuinely cell-centered input must opt into `src_convention='centers'`.
7. **Pentacene ES and AFM panels were regenerated end-to-end**; USER visual review remains the acceptance gate.
8. **GPU Poisson / AFMulator** assume scalar step on the isotropic FDBM destination; native cube operations must retain `(sx,sy,sz)` until that projection.
9. **The mirror diagnostic now reflects by linear interpolation about the fractional physical grid index.** Remaining nonzero Y asymmetry includes the non-symmetric input geometry.

**Do not claim fixed** until USER reviews a cube row produced from `ρ_SCF` plus analytical/pseudized nuclear compensation **without requiring `rho_NA.cube`**.

---

## 5. Suggestions (priority order)

### Minimal proper cube-only correction

Keep the existing separation: full `ρ_SCF` remains the Pauli density; only the ES density is pseudized.

```text
ρ_ps       = soft_clamp_core(ρ_SCF)
Qrem_i     = ∫sphere_i (ρ_SCF − ρ_ps)                         # diagnostic sampling
Qrem_Z     = mean(Qrem_i for atoms with element Z)            # core is element-local
q_Z^0      = Z − Qrem_Z                                       # identical for equal elements
δQ         = ∫ρ_ps − Σ_i q^0_{Z_i}
q_Z        = charge-close(q_Z^0, δQ, element/core weights)     # equal atoms stay equal
ρ_comp     = Σ_i moment_exact_spherical_stamp(q_{Z_i}, R_i)
Δρ_ES      = ρ_ps − ρ_comp
```

Key rules:

1. **Never use `Qrem_i` as the individual compact charge.** Its variation is primarily voxel-phase error.
2. Equal elements receive equal compensation parameters. Distribute `δQ` only through a declared element/core weighting rule (for first-row systems, core-bearing atoms rather than H is the natural default).
3. Construct every compensation stamp with exact discrete monopole and center. A robust implementation is trilinear charge deposition at `R_i` followed by convolution with a normalized symmetric Gaussian/compact kernel; this preserves charge and first moment independently of sub-voxel phase.
4. Verify `∫Δρ_ES≈0` and compare its dipole to the clamped electronic valence moment. Do not force the dipole to zero.
5. Keep native `(sx,sy,sz)` throughout.

### More rigorous long-term solution

Replace the density-value clamp by element-specific analytic core pseudization, analogous to a compensation-charge/pseudopotential construction:

- obtain a normalized atomic core model `c_Z(r)` from the basis or a tabulated isolated atom;
- remove the sampled cusp contribution locally while preserving the non-spherical valence residual;
- use a fixed element compensation charge whose discrete monopole and first moment are exact;
- if the AO density matrix is available, evaluate the pseudized/valence density directly instead of exporting an under-resolved all-electron cube.

A globally much finer cube is mathematically simpler but inefficient: `0.15 bohr` does not resolve C/N/O 1s cusps, while the valence/AFM region does not need that resolution. Larger vacuum alone does not cure core aliasing.

### Transfer and verification

1. The center-source projector is now adapted locally for cube nodes with `origin_src_project=origin_src−0.5*step_src` and explicit `src_convention`.
2. Add L0 tests:
   - exactly symmetric C₂H₄: analytical path has near-zero dipole;
   - random sub-voxel translations of identical analytic cusps: q/p and V are invariant;
   - compensation stamps satisfy per-atom charge and first moment;
   - a deliberately polar molecule retains its physical valence dipole under grid translations;
   - anisotropic native node grid → isotropic FDBM node grid preserves q/p;
   - existing center-source `GridsOCL` tests continue to pass.
3. Extend `es-diag` with per-element `Qrem` mean/std/range, correlation with sub-voxel phase, raw→clamped→compensated moment decomposition, and physical-coordinate reflection metrics.
4. Use `rho_NA.cube` only when available as an independent diagnostic—not as a required production input.
5. Re-run the cube panel row and present images to the USER. Only USER visual confirmation may change status.

### Verification performed in this investigation

```text
python -m pytest tests/SPM/test_afm_fdbm.py::test_mirror_asymmetry_fractional_center \
  tests/SPM/test_afm_fdbm.py::test_clamp_compensation_is_element_invariant -s
2 passed

python -m pytest tests/utils/test_grids_ocl.py -s
5 passed, 1 warning; NVIDIA GeForce RTX 3090

python run_spm.py es-diag --molecule pentacene --outdir debug/fdbm_fukui_panel_flat
python run_spm.py panel-fukui --molecule pentacene --outdir debug/fdbm_fukui_panel_flat
```

Regenerated result: native `|p_xy|=0.01823 e·Å`, native `V_mX(z+1)=0.03342`; projected `p=(0.00394,−0.01780,−0.00059) e·Å`, with `|q|<2e−7 e`. The full cube/stock/prolonged AFM panel completed on the RTX 3090. A broad non-slow run reached `302 passed, 2 skipped`; its 11 failures were in unrelated pre-existing export/topology/surface/contact-surface areas of the dirty worktree. The complete affected modules passed `20/20`. These are verification results, not a USER-confirmed resolution.

---

## 6. Relevant files

### Code (SPAMMM)

| Path | Role |
|------|------|
| `spammm/SPM/AFM_utils.py` | Element-invariant clamp compensation, analytical NA builders, cube-node projection adapter, and fractional-center diagnostics |
| `spammm/SPM/AFM.py` | `fft_poisson_cpu` (aniso kx/ky/kz), GPU Poisson / FDBM (iso dest) |
| `spammm/utils/GridsOCL.py` | Center-source → node-destination project API; `(3,)` steps; separate node/center moment helpers |
| `kernels/grids.cl` | Trilinear scatter kernel; source coordinate is explicitly `origin+(i+½)step` |
| `spammm/quantum/DFTB/DFTBplusParser.py` | `read_cube` |
| `tests/SPM/testplot_fdbm_relax.py` | `run_fukui_es_diag*`, Fukui panel runners |
| `run_spm.py` | CLI `es-diag`, `panel-fukui`, `afm --cube` |

### Documentation

| Path | Role |
|------|------|
| **This file** | Separated investigation SSOT for cube Δρ/NA dipole |
| `doc/Caveats.md` | Global traps (aniso, NA multipoles, corner/center, sample vs project) |
| `doc/Reports/Fukui_FDBM_panel_notes_2026-07-23.md` | Panel + ES + df↔Fz notes |
| `doc/Tasks/Import_KrigingGridFF.md` | Pyridine / cube ES history |
| `doc/Tasks/ProlongedRadialBasis_DFTB.md` | Fukui panel task context |
| `doc/TopicalAudit/AFM_FDBM.md` | Open issue pointer |
| `debug/README.md` | Canonical artifact index |

### Input data (external cubes)

Root: `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/`

Per molecule (example pentacene):
```text
…/pentacene_PBE_def2-SVP/
  rho_N.cube      # all-electron SCF density (required)
  rho_NA.cube     # optional independent diagnostic; not required by intended analytical path
  esp_N.cube      # optional ESP cross-check (not used in main bisect)
  …               # A/C charge variants may exist
```

Same layout for: PTCDA, phtalo_*, azaindol_*, benzoic* (see Fukui panel list in `testplot_fdbm_relax.py`).

### Output / diagnostic artifacts (gitignored under `debug/`)

```text
debug/fdbm_fukui_panel_flat/
  SUMMARY_es_diag.out
  <mol>/es_diag/
    dipole_origin_bisect.png     # PRESERVE — smoking-gun 3×4 panel
    DIPOLE_ORIGIN.out            # |pxy| / V_mX lines
    es_chain_native_gaussNA.png
    es_chain_native_clamp_compact.png
    es_chain_fdbm_grid.png
    ES_ASYM.out
    tip_co.png
  <mol>/compare_cube_stock_prolonged.png   # full AFM panel (when run)
```

**Primary molecule for this report:** `pentacene`.

---

## 7. Short verdict

| Item | Verdict |
|------|---------|
| Tip / Poisson as root cause | **Ruled out** |
| Mean-step on anisotropic cube | **Real bug; fixed; necessary** |
| Parser axis/order, half-voxel native interpretation, Gaussian centering | **Ruled out by npy parity, core peaks, analytic moments, and C₂H₄ control** |
| Main native asymmetry | **0.15-bohr all-electron cusp alias + feeding noisy per-atom `Q_rem_i` into compensation charges** |
| Pentacene geometry exactly symmetric? | **No** — mirror/inversion residuals up to 0.07–0.10 Å |
| Preliminary cube-only correction | **Element-invariant clamp compensation: \|p_xy\| 0.28365→0.01823; V mX 0.2747→0.0399** |
| `rho_NA.cube` role | Diagnostic control only; **not the required solution** |
| Cube nodes passed as projector centers | Secondary transfer bug; **localized adapter repair tested on NVIDIA** |
| Next code step | USER review of regenerated plots; then add polar-molecule/grid-translation coverage and, if needed, element-specific analytic core models |

**Status remains: investigating.**
