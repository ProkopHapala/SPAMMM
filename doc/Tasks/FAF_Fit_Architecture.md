---
type: Task
title: FAF fit architecture — Coulomb baked with molecule Q, components summed, type explosion with QEq
status: implementation under verification — awaiting USER physical review
tags: [FAF, FoldedRigid, Coulomb, Ewald, substrate, architecture]
timestamp: 2026-08-01
related: [PairFF_MC_PythonBottleneck.md]
skills: [python-perf, gpu-optimize]
---

# FAF fit architecture — Coulomb baked with Q, components summed, type explosion

## 1. User's expected architecture

FAF (Folded Approximate Field) should be a **substrate-only** representation:
1. Compute electrostatics (2D Ewald) + Morse (Pauli + London) for the **substrate alone**
2. Fit x,y Fourier + poly_exp z-basis (tensor product) to these substrate potentials
3. Fit is **molecule-independent** — precomputable and storable before any molecule is known
4. Once a molecule is defined, **mixing rules** (same as GridFF) combine the probe atom's (R0, E0, Q) with the pre-fitted substrate potential at evaluation time

## 2. What the legacy typed path does

### 2.1 The fit is molecule-dependent

`fit_folded_for_molecule(mol_file, ...)` (`FoldedRigid.py:159`):
- Loads the **molecule** XYZ → `reqs` (R, sqrt(E), Q, H) for each molecule atom
- `_folded_unique_types(self.rigid_REQs0)` — finds unique (R,E,Q,H) combos **from the molecule's atoms** (`SPFF_cl.py:1034`)
- For each unique type: place a single-atom probe at every (u,v,z) grid point, evaluate Morse + Coulomb, fit basis coefficients

### 2.2 Coulomb Q is baked at fit time

`SPFF_cl.py:1346-1351`:
```python
q_probe = float(uniq_REQs[it, 2])   # molecule atom's charge
phi = ew.eval_full(X, Y, Z)          # substrate Ewald potential (molecule-independent!)
y = phi.reshape(-1) * q_probe        # Q baked into fit target
```

At eval (`FoldedRigid.py:644`): `return (basis * c).sum()` — no Q multiplication, it's already in the coefficients.

### 2.3 Components are SUMMED, not separated

`FoldedRigid.py:198`:
```python
total_coeffs = coeff_sets['pauli'] + coeff_sets['london'] + coeff_sets['coulomb']
```

Pauli + London + Coulomb are **added into one coefficient array** before saving. Individual components are NOT stored. You cannot extract Coulomb at eval time to multiply by Q.

### 2.4 Consequence: Q=0 → no Coulomb checkerboard

PTCDA.xyz has no charges → `qs=0` → `uniq_REQs[:,2]=0` → `y = phi_ewald * 0 = 0` → Coulomb coefficients all zero. FAF heatmap shows only Morse repulsion (slightly stronger over Cl = bigger atoms), NOT the Na/Cl electrostatic checkerboard. This is what the user observed.

### 2.5 The type explosion problem

The "per-type" approach groups atoms by unique (R,E,Q,H). This works when Q=0 (3 types for PTCDA: C, O, H), but **breaks with QEq charges**:

| | No QEq (current) | With QEq |
|---|---|---|
| Unique types | **3** (C, O, H, all Q=0) | **38** (every atom has different Q) |
| Fit cost | 3 SPFF evaluations | 38 SPFF evaluations |
| Storage | 3×112 = 336 floats | 38×112 = 4256 floats |

With QEq, every C atom has a slightly different Q (from +0.363 to −0.047). The per-type approach treats each distinct Q as a new type → **38 types for 38 atoms** — one type per atom, the optimization is completely defeated.

For Coulomb this is especially absurd: `phi_ewald` is the **same substrate potential** for all 38 types — only the scalar `Q_probe` differs. We'd fit the same 112 basis functions 38 times, scaled by different constants. 37 redundant fits.

### 2.6 Morse is legitimately per-type (but per-element, not per-atom)

Morse mixing rules depend on the probe's R0, E0:
- `R0_mix = R0_probe + R0_substrate` (arithmetic)
- `E0_mix = sqrt(E0_probe * E0_substrate)` (geometric)

R0 and E0 vary by **element** (C, O, H), not per-atom. So Morse per-type is legitimate with ~3-5 types. The problem is only Coulomb, where Q varies per-atom.

## 3. The two strategies (both kept)

### Strategy 1: Per-type combined potential

**Idea:** Fit one combined potential per atom-type. All four components (Pauli, London, Coulomb, Hbond) are summed into a single coefficient array. At eval, just look up the type and evaluate one basis expansion.

**Types:** Atoms are grouped into a small number of types by (R, E, Q, H). Since Q varies per-atom with QEq, raw per-atom QEq would cause type explosion (38 types for PTCDA). **Charge discretization** (§3.3) solves this: bin per-atom charges into ~3-5 types while preserving molecular multipole moments (neutrality, dipole, quadrupole).

**Storage:** `coeffs[ntypes, nbasis]` — one row per discretized type. For PTCDA with element+sign discretization: 4 types × 112 basis = 448 floats = 1.8 KB.

**Eval:** `E = (basis * coeffs[type_id]).sum()` — single dot product, no mixing rules at eval time. All mixing (R0+E0 for Morse, Q×phi for Coulomb) is baked into the coefficients at fit time.

**Pros:**
- Simplest eval — one basis expansion, one lookup
- Fastest GPU kernel (single channel, no per-atom mixing)
- Compact storage

**Cons:**
- Fit is molecule-dependent (must know QEq charges to discretize types)
- Coulomb is approximate (discretized charges, not exact per-atom Q)
- Cannot reuse a fit for a molecule with different charges without refitting
- Requires charge discretization step before fitting

**When to use:** When the molecule is known at fit time, charges are available (QEq), and approximate Coulomb is acceptable. Good for MC/GA assembly where the molecule is fixed and we want maximum eval speed.

### Strategy 2: Per-atom with mixing rules, 4 separate components

**Idea:** Store the four interaction components (Pauli, London, Coulomb, Hbond) as **separate coefficient arrays**. At eval, apply mixing rules per-atom using the atom's own (R, E, Q, H) parameters. This mirrors GridFF's approach — the substrate potential is precomputed, the molecule enters at eval time via mixing rules.

**Components:**
- **Coulomb:** Fit `phi_ewald(x,y,z)` directly — the substrate electrostatic potential, **no molecule Q**. Stored as `coeffs_coulomb[1, nbasis]` (substrate-only, 1 set). At eval: `E_coulomb = Q_atom * (basis * coeffs_coulomb).sum()`. Q_atom comes from the molecule at runtime. **Molecule-independent, precomputable.**
- **Pauli:** Fit per-element-type (C, O, H = 3 types). Mixing: `R0_mix = R0_atom + R0_substrate`, `E0_mix = sqrt(E0_atom * E0_substrate)`. At eval: `E_pauli = f(coeffs_pauli[type], R0_atom, E0_atom)`.
- **London:** Same per-element-type fitting as Pauli. At eval: `E_london = g(coeffs_london[type], R0_atom, E0_atom)`.
- **Hbond:** Fit per-type. At eval: `E_hbond = h(coeffs_hbond[type], H_atom, Q_atom)`.

**Storage:** `coeffs_coulomb[1, nbasis]` + `coeffs_pauli[ntypes_elem, nbasis]` + `coeffs_london[ntypes_elem, nbasis]` + `coeffs_hbond[ntypes_elem, nbasis]`. For PTCDA (3 element types): (1+3+3+3) × 112 = 1120 floats = 4.5 KB. Still small.

**GPU layout — float4:** Two levels of float4 packing fit the GPU vector architecture:

1. **Per-atom parameters:** `REQH_atom = float4(R0, sqrt(E0), Q, H)` — one float4 per atom, read once.
2. **Per-type coefficients:** interleave the 4 component arrays as `float4 coeffs4[ntypes, nbasis]` where `.xyzw` = (pauli, london, coulomb, hbond). The GPU kernel reads one `float4` per basis function per type, multiplies by the scalar basis value, and accumulates into a `float4` partial sum. After the basis loop, the kernel applies per-atom mixing rules to combine the 4 channels into the final energy. This maps cleanly to the GPU's float4 vector operations — one load, one multiply-add per basis function, no wasted channels.

**Eval:** `E = mix_pauli(basis·c_pauli[type], R0_atom, E0_atom) + mix_london(basis·c_london[type], R0_atom, E0_atom) + Q_atom * (basis·c_coulomb).sum() + mix_hbond(basis·c_hbond[type], H_atom, Q_atom)`

**Pros:**
- Coulomb is **exact** (per-atom Q, not discretized)
- Coulomb fit is **substrate-only** — precompute once, reuse for any molecule
- Morse fit is per-element-type (3-5 types, no explosion)
- Natural float4 layout for GPU (REQH per-atom, 4 component channels)
- Matches GridFF mixing-rule philosophy

**Cons:**
- More complex eval (4 channels + mixing rules vs 1 lookup)
- Slightly larger storage (4 arrays vs 1)
- GPU kernel needs per-atom REQH + mixing-rule logic

**When to use:** When the substrate fit should be precomputed and reused across molecules, when exact per-atom Coulomb is needed, or when integrating with GridFF-style mixing. Preferred for the factorized architecture.

### Comparison

| | Strategy 1: Per-type combined | Strategy 2: Per-atom 4-component |
|---|---|---|
| **Components** | Pauli+London+Coulomb+Hbond summed into 1 array | 4 separate arrays (float4 channels) |
| **Coulomb** | Baked with discretized Q (approximate) | Substrate-only, Q×phi at eval (exact) |
| **Morse** | Baked with probe R0,E0 per type | Per-element-type, mixing rules at eval |
| **Types** | ~3-5 discretized (element+sign, charge-binned) | ~3-5 element types (C,O,H) for Morse; 1 for Coulomb |
| **Per-atom params** | Not needed at eval (baked) | REQH float4 per atom, mixing rules at eval |
| **Storage** | `coeffs[ntypes, nbasis]` | `coeffs4[ntypes, nbasis]` (float4) + `REQH_atom[N]` (float4) |
| **Eval** | `(basis * c[type]).sum()` | 4-channel basis eval + per-atom mixing |
| **GPU kernel** | Single channel, no mixing | float4 channels + REQH mixing (fits GPU vector arch) |
| **Precomputable** | No (needs molecule QEq for discretization) | Coulomb yes (substrate-only); Morse per-element (reusable) |
| **Coulomb accuracy** | Approximate (multipole-preserving discretization) | Exact (per-atom Q) |
| **Fit cost** | 3-5 SPFF evals + Ewald + discretization | 3-5 SPFF evals + 1 Ewald (no discretization) |

### Charge discretization (for Strategy 1)

The per-type approach can represent electrostatics **approximately** if we discretize per-atom QEq charges into a small number of types while preserving the molecule's multipole moments:

1. Compute QEq charges per-atom
2. Assign atoms to types by element (C, O, H → 3 types) or element + charge-sign (C+, C−, O, H → 4-5 types)
3. Solve constrained least-squares for per-type charge Q_t:
   - **Hard constraint:** charge neutrality: `sum_t n_t * Q_t = 0`
   - **Soft constraints (least-squares):** preserve dipole `sum_t Q_t * R_t = M1` and quadrupole `sum_t Q_t * S_t = M2`
   - For planar molecules (PTCDA): dipole = 2 components (x,y), quadrupole = 3 components (xx,xy,yy) → 1+2+3 = 6 constraints, 3 unknowns → over-determined, least-squares fit
4. The discretized charges approximate the molecule's electrostatic field while keeping the type count small (~3-5)

**Quality:** for PTCDA (planar, 3 element types), this should reproduce the dipole well and quadrupole approximately. The Na/Cl checkerboard will be visible because Q ≠ 0. For non-planar molecules, add z-dipole and zz/yz/xz quadrupole components.

**Limitation:** the discretized field is an approximation — it cannot reproduce the full per-atom Coulomb field. But for the FAF substrate interaction (which is itself a smooth Fourier fit), the approximation is adequate. Strategy 2 is exact; this is a pragmatic middle ground for the per-type path.

**Tested:** `tests/test_faftype_discretize.py` now exercises the shared production solver. On PTCDA/QEq, four element+sign types preserve total charge exactly, dipole within `2.69e-4 eÅ`, and the traceless-quadrupole tensor within `2.77e-6 eÅ²`; per-atom charge RMS error remains `0.235 e`, so this is a controlled approximation, not an exact charge representation.

## 4. Historical implementation draft (superseded by §6–§10)

### Strategy 1 fixes (per-type combined, keep current architecture)

#### Phase 1a: Charge discretization before fit
1. Compute QEq charges per-atom
2. Run `discretize_charges()` (tested in `test_faftype_discretize.py`) to get ~3-5 types with multipole-preserving charges
3. Pass discretized `q_override` to `fit_folded_for_molecule` so the fit uses non-zero Q per type
4. This fixes the immediate visualization bug (checkerboard appears) without changing the architecture

### Strategy 2 implementation (per-atom 4-component, new)

#### Phase 2a: Separate components in the fit
1. **`SPFF_cl.py:fit_folded_surface_basis`**: return `coeff_sets` dict with separate `pauli`, `london`, `coulomb`, `h_bond` arrays (already computed, just don't sum them).
2. **`FoldedRigid.py:fit_folded_for_molecule`**: store all 4 component arrays separately in the fit dict.
3. **`FoldedRigid.py:save_fit` / `load_fit`**: save/load 4 separate arrays. Pack as `float4 coeffs4[ntypes, nbasis]` where `.xyzw` = (pauli, london, coulomb, hbond).

#### Phase 2b: Fit Coulomb substrate-only (no Q)
1. **`SPFF_cl.py:1346-1351`**: change Coulomb fit target from `phi_ewald * q_probe` to `phi_ewald` (drop `* q_probe`). Coulomb coefficients are now substrate-only.
2. Coulomb needs only **1 type** (not per-molecule-type) → fit once, store as `coeffs_coulomb[1, nbasis]`.

#### Phase 2c: Per-atom mixing rules at eval time
1. **`FoldedRigid.py:eval_folded_potential`**: split eval into 4 channels + per-atom mixing:
   ```
   E_pauli  = mix_pauli(basis · coeffs_pauli[type], R0_atom, E0_atom)
   E_london = mix_london(basis · coeffs_london[type], R0_atom, E0_atom)
   E_coulomb = Q_atom * (basis · coeffs_coulomb).sum()
   E_hbond  = mix_hbond(basis · coeffs_hbond[type], H_atom, Q_atom)
   E = E_pauli + E_london + E_coulomb + E_hbond
   ```
2. **`RigidBodyDynamics.py:attach_pairff_faf`**: pass per-atom `REQH_float4` array to GPU (one float4 per atom: R, sqrt(E), Q, H).
3. **`kernels/rigid.cl:rigid_body_folded_kernel`**: read `coeffs4[type]` as float4 per basis function, accumulate 4-channel partial sums, apply per-atom REQH mixing at the end. Fits GPU float4 vector architecture.
4. **`surface_plots.py:plot_assembly_on_substrate`**: use `coeffs_coulomb` for the heatmap (multiply by probe Q for display).

### Backward compatibility (both strategies)

1. Old fits (summed coeffs) still load — detect format by checking for `coeffs_coulomb` key.
2. If old format: fall back to current behavior (summed, Q baked).
3. If new format: use factorized eval.

### Phase 5: Immediate workaround (before refactor)

Pass QEq charges via `q_override` to `fit_folded_for_molecule` so the current per-type fit at least has non-zero Q. This fixes the visualization (checkerboard appears) but does NOT fix the architecture (still molecule-dependent, type explosion with QEq).

## 5. Files involved

- `spammm/forcefields/SPFF_cl.py:1258` — `fit_folded_surface_basis` (fit, line 1346 = Q baking, line 1366 = component separation already exists)
- `spammm/surfaces/FoldedRigid.py:159` — `fit_folded_for_molecule` (wrapper, line 198 = summing)
- `spammm/surfaces/FoldedRigid.py:614` — `eval_folded_potential` (eval, no Q multiplication)
- `spammm/surfaces/FoldedRigid.py:219` — `save_fit` / `load_fit` (storage format)
- `spammm/forcefields/molecule_loaders.py:203` — `remap_fit_for_molecule` (remap)
- `spammm/forcefields/RigidBodyDynamics.py:2261` — `attach_pairff_faf` (GPU binding)
- `spammm/surfaces/surface_plots.py:458` — `_faf_probe_type` (heatmap probe selection)
- `kernels/rigid.cl` — `rigid_body_folded_kernel` (GPU eval)
- `tests/testplot_pairff_energy_mc.py:313` — fit call without q_override

## 6. Critical architecture correction found during implementation

The Phase 2 draft above says to fit Pauli/London per probe element and then apply
mixing rules again. That would double-apply the probe parameters. The actual
`getMorsePLQH()` law is exactly separable, so the factorized representation can
be both simpler and more general.

For `alpha = -K`, substrate atom `j`, and probe atom `i`:

```text
E_P = [sqrt(E_i) exp(2 alpha R_i)] [sqrt(E_j) exp(2 alpha R_j) exp(-2 alpha r)]
E_L = [sqrt(E_i) exp(  alpha R_i)] [-2 sqrt(E_j) exp(alpha R_j) exp(-alpha r)]
E_Q = Q_i [COULOMB_CONST Q_j/r]
```

The bracket on the left is exactly the existing `_reqs_to_plq()` runtime atom
coefficient `(cP,cL,Q,cH)`. Therefore Strategy 2 uses one universal fit probe
`REQH=(0,1,1,0)` and stores one substrate `coeffs4[nbasis]`, not one float4 row
per element. This has three important consequences:

- Morse types disappear as well as charge types; fit cost is one SPFF surface
  evaluation plus one Ewald field.
- Any molecule with compatible force-field conventions can reuse the fit.
- Runtime mixing is mathematically identical to the original pair law before
  basis-fit error. “Exact Coulomb” means exact runtime `Q_i` scaling of the
  fitted substrate potential; the finite FAF basis still has approximation
  error relative to direct Ewald.

H-bond is reserved as `.w`, but `getMorsePLQH()` currently has no H-bond term.
The generated `.w` field is consequently zero. It must not be advertised as
physically implemented until a substrate H-bond law and parity reference exist.

## 7. Implemented backend (2026-08-01, awaiting USER confirmation)

### 7.1 Fit formats and harness

`spammm/surfaces/FoldedRigid.py` now defines two explicit version-2 modes:

- `typed_combined`: legacy scalar `coeffs[ntypes,nbasis]`, optionally using
  robust charge discretization. Existing unversioned NPZ fits load as this mode.
- `factorized_plqh`: substrate-only `coeffs4[nbasis,4]` plus per-atom runtime
  PLQH. `fit_folded_for_molecule(..., fit_mode='factorized_plqh')` still accepts
  a molecule for immediate binding, but the fitted coefficients do not depend
  on that molecule.

`save_fit()` / `load_fit()` preserve both formats without pickle.
`materialize_factorized_coeffs()` provides an analytical host reference.
`compare_faf_fit_modes()` fits both modes with identical basis settings and
reports fit time plus coefficient parity.

The fitter now reuses one `SPFF_cl` probe engine across typed rows and computes
the substrate Ewald field once. Previously it rebuilt the whole OpenCL engine
and reevaluated the identical Ewald potential for every type. The cached
`getSurfMorse` kernel object also removes repeated PyOpenCL kernel retrieval.

### 7.2 OpenCL backend

The existing kernel ABI is retained:

- `folded_meta.y > 0`: typed scalar coefficients (unchanged).
- `folded_meta.y < 0`: factorized float4 coefficients; the existing
  `folded_atom_type` buffer carries aligned runtime PLQH float4 values.

Factorized evaluation is implemented in:

- rigid folded MD/FIRE kernel;
- many-replica folded MD/FIRE kernel;
- fused PairFF+FAF single/static, environment, and all-molecule kernels;
- PairFF replica×active MC/GA energy kernel.

Dummy PairFF sites receive zero PLQH, so FAF remains restricted to real atoms.
GPU Newton and Newton-replica kernels remain typed-only and now fail loudly from
the Python harness if called with a factorized fit.

### 7.3 Charge discretization

The reusable implementation moved to `FoldedRigid.discretize_charges()`; the
old test-local prototype is retained only as historical context. Improvements:

- exact total-charge equality via a null-space constrained solve, not a `1e6`
  penalty;
- preserves the input net charge rather than assuming every molecule neutral;
- center-of-geometry origin and RMS-radius scaling for conditioning;
- conventional traceless quadrupole with five independent components;
- stable `element_sign` split with a configurable near-zero threshold;
- input/finite checks, rank/singular-value diagnostics, charge RMS reporting;
- translation- and atom-permutation-invariance tests.

For PTCDA/QEq the fitted four charges are `C+=+0.17573`, `C-=-0.44101`,
`H+=+0.29733`, `O-=-0.27704 e`. The excellent low multipole error coexists with
large per-atom RMS error (`0.235 e`). For adsorption accuracy, a better future
typed objective is to fit the molecule’s electrostatic energy over representative
surface poses (or weight multipoles by the substrate reciprocal harmonics),
rather than assuming low free-space multipoles alone determine the error.

## 8. Verification evidence shown to USER

All OpenCL results below used the NVIDIA GeForce GTX 1650, not PoCL:

| Check | Result |
|---|---:|
| Charge/storage L0 tests | `5 passed` |
| Existing H2O/NaCl folded relaxation regression | `1 passed`; final `|F|=3.9e-5`, `|tau|=6e-6` |
| Factorized rigid CPU↔GPU energy | absolute error `2.89e-6 eV` |
| Factorized replica CPU↔GPU energy | max absolute error `1.39e-5 eV` over 4 replicas |
| Factorized vs equivalent typed PairFF/MC energy kernel | absolute error `0.0 eV` |
| NaCl/PTCDA fitted-coefficient parity | RMS `1.13e-6`, max `4.41e-6`, reference RMS `1.607` |
| PTCDA reduced fit time (4 typed rows vs 1 factor row) | `0.513 s` vs `0.339 s` end-to-end (`1.51x`; compilation dominates this small case) |
| Steady folded kernel, 20k in-kernel steps | factorized/typed median time ratio `1.020` (about 2% overhead) |

The coefficient-parity result is the strongest physics check: the independently
fitted universal fields, mixed through PLQ, reproduce direct probe-specific
typed fits to about one part per million for the tested reduced basis.

These results verify implementation parity, not adsorption physics. Per project
policy this task remains under verification until the USER reviews/accepts the
physical behavior.

## 9. Explicitly delegated lower-priority Python work

The following work was inventoried by a cheaper LLM and intentionally not mixed
into the performance-critical backend change:

1. Add thin `describe_fit`, validation, capability, and cache-key helpers. Cache
   keys must include format version, substrate, basis, alpha, and mode; legacy
   fits must never be silently reused as factorized fits.
2. Update `molecule_loaders.remap_fit_for_molecule`: typed mode keeps type
   remapping; factorized mode preserves runtime REQH and must not map by Q.
3. Update GUI/demo/MC cache and CLI UX with
   `--faf-mode {auto,factorized,legacy}`, explicit selected mode, exact-Q status,
   and rejection of incompatible fits.
4. Replace “display atom type row” with physical quantities:
   `coulomb_phi`, `coulomb_energy(probe_q)`, `pauli`, `london`, and
   `total(probe_REQH)`. Labels must distinguish potential from probe energy.
5. Update FoldedRigid and Rigid Assembly GUI controls to expose fit mode,
   component, probe source, and an explicit
   `factorized required / permit legacy / FAF off` policy.
6. Add cache migration and mixed-species tests, GUI smoke tests, and legacy NPZ
   parity. Core GPU-vs-CPU component parity remains owned by backend tests.
7. After USER physical confirmation, update demos/manuals, CODEMAP, remaining
   README entries, and remove stale advice to use element-mean `q_override`.

Important constraints for that follow-up: Python must call the shared workhorse
functions rather than inspect NPZ keys independently; factorized plots must not
call `faf_type_idx_for_probe`; no GUI/CLI code may implement a second fit or
mixing path.

## 10. Recommended next physics/performance work

1. USER review: compare direct Ewald+Morse, typed-discretized FAF, and factorized
   FAF on adsorption-height z profiles and Na/Cl lateral registries for at least
   PTCDA and one polar non-planar molecule.
2. If typed electrostatics is retained for production, add optional bounds or a
   substrate-field/pose-weighted charge objective; multipole parity alone hides
   the large per-atom error.
3. Profile the production 112-basis MC/GA kernel with realistic molecule counts.
   The measured factorized arithmetic overhead is small, but occupancy/register
   behavior must be checked at actual launch sizes.
4. Implement factorized Newton only if it is needed; do not duplicate the
   evaluation formula—route all Newton energy/gradient reads through
   `folded_coeff_rigid`.
5. Define and validate a real substrate H-bond channel before enabling `.w`.
