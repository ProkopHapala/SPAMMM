---
type: Task / Design
title: Contact-PME fit of a real (FDBM/DFT) field — radial cores + coarse supersampled B-spline mesh
tags: [contact-pme, pic, paw, b-spline, fdbm, fitting, design]
timestamp: 2026-10-05
status: PARTIALLY IMPLEMENTED — FAILING. See §8 failure record. The implementation as built does NOT satisfy the design intent (smooth complementary decomposition); do not continue tuning it without reading §8.
---

# Contact-PME fit of a real field: radial cores + coarse supersampled mesh

## 1. Goal

Represent a **real** sampled potential `E(x)` (FDBM: Pauli + electrostatics + dispersion; later DFT) for PP-AFM with the existing cheap evaluator

```
E_model(x) = Σ_i Σ_k c_ik φ_k(|x − x_i|)  +  Σ_j m_j B_j(x)
              ───── radial cores ─────       ─ cubic B-spline mesh ─
```

- **Few coefficients per atom** (5 `t^p` modes, `PICCore.core_basis`) — dual-purpose: short-range Pauli **and** short-range electrostatics (signs free).
- **Coarsest possible mesh** (`h = 0.5 … 1.0 Å`) — memory and evaluation speed.
- Accuracy is required **only where the probe goes** (low-energy region, `E ≲ E_cut`); the deep wall only has to be *bounded, monotone-ish and smooth* so it never leaks into the reachable region.
- The evaluator (GPU kernels: 64-tap spline + compact cores) **does not change**. Only the fitter changes.

Non-goals: exact representation of the keV interior; per-atom physical meaning of the core coefficients.

## 2. What was wrong in the previous attempts (2026-10-04/05 debug session)

Artifacts: `debug/testplot_fdbm_pme_debug/{azaindol,cc}_h0.5/` (`slice_xz_*.png`, `split_1d_clamp.png`). Harness: `tests/SPM/testplot_fdbm_pme_debug.py`.

| # | mistake | symptom |
|---|---|---|
| M1 | Cores were asked to fit an **unbounded** target (`E − f(E)` contains the full keV wall) with 5 polynomial modes, plain Boltzmann weights. | `rmse_E ≈ 6.3` vs `wrmse ≈ 0.034`; LSQ trades keV interior against a **negative core tail ≈ −0.017 eV at r ≈ 3.0 Å** → basin overshoot ×2 in the total. |
| M2 | Mesh coefficients were obtained by **nodal interpolation** (`CoarseMesh._prefilter_3d` on node values). Any residual feature sharper than `h` → Gibbs ringing; nothing between the nodes is ever seen by the fit. | ±30 eV oscillations, mottled df, crest error bands, lateral blue pockets. |
| M3 | Split defined in **E-space by a pointwise clamp** with thresholds near/below 0 (`y1 = E_min`). | Clamp acted on the attractive tail → offset between soft and total at large r (fixed: `y1 > 0`). |
| M4 | (earlier) per-atom independent radial profiles; inpainting / mask dilation / caps to hide residual spikes. | Venn-diagram shells, seams. **Forbidden** — hides root cause. |

Note on the earlier supersampling report (`doc/Reports/ContactPME_Supersample_Scale_2026-10-04.md`): there the mesh target `V_L` was **analytically smooth** (Morse model), so nodal collocation was already exact and supersampled L2 projection could not help. For a real field the mesh target (residual after cores) is *not* guaranteed smooth below `h`, so that conclusion does **not** transfer.

## 3. Design

Separate (block) fitting is kept for practicality — the mesh solve stays **separable and banded** (same structure as the existing Thomas prefilter, GPU-friendly). Joint optimality is approached by a few alternation sweeps (block Gauss–Seidel), not by a monolithic sparse solve.

### 3.1 Samples (shared by both stages)

- **Supersampled regular grid** with factor `s` (default `s = 2`: nodes + mid-points along each axis, i.e. `h/s` spacing) over the mesh box incl. halo. Evaluated once by the GPU FDBM oracle (`res['prolonged']['sample']`).
- Regular grid is required for the separable mesh solve (§3.3). Cores may additionally use the dense `sample_core_shells` points in their own stage.
- Every sample carries `E`, optionally `F` (force rows later).

### 3.2 Bounded target

```
T = soft_clamp_rational(E, y1 = E_cut, y2 = E_cap)     E_cut ≈ probe-reachable energy (start 0.3 eV), E_cap ≈ 2–4·E_cut
```

`T = E` exactly wherever the probe goes; inside the wall it is a smooth plateau `< E_cap`. **All stages fit `T`, never raw `E`.** The keV dynamic range disappears from the problem (fixes M1 at the source).

### 3.3 Stage A — core prefit (weighted, small sparse LSQ)

Target: `T` (bounded). Unknowns: `c_ik`, 5 per atom. Existing solver `PICCore.fit_cores_from_samples` (joint over all atoms — required, cores overlap).

- **Weights** focus on the low-energy region but must keep some repulsion in the fit (user requirement):
  - default: relative-error weight `w = 1 / (E_w + max(T,0))²` (minimises relative error on the wall; `E_w ≈ |E_min|`) — keeps the wall foot in the fit without letting the plateau dominate;
  - alternative: Boltzmann `w = exp(−max(T,0)/k_T)`, `k_T` tuned (current default 0.5 eV) — compare both on the dimer.
- **Core domain per element**: `r_lo` set where the *isolated-atom* profile reaches `≈ E_cap` (so the flat-clamped region `r < r_lo` is the unreachable interior), `r_b = r_lo + Δ` with `Δ` from the locality report (`ContactPME_Split_Rcut_Locality_2026-10-04.md`, Δ_b ≈ 0.6–1.2 Å). The basis then spans only ~2 decades of energy — feasible for 5 modes.
- Samples restricted to `r_i ∈ [r_lo, r_b]` of at least one atom (others contribute nothing to the core columns anyway).

Stage A on its own target leaves the long-range part to the mesh automatically because cores are compact (`φ = 0` beyond `r_b`).

### 3.4 Stage B — mesh fit: supersampled, penalised, separable

Target: `R = T − Σ cores` on the supersampled grid. Unknowns: mesh coefficients `m` (`n_x × n_y × n_z`).

Least squares with curvature penalty (P-spline), **in tensor-product form so it stays separable**:

```
per axis a:   A_a = B_aᵀ B_a + λ D_aᵀ D_a      (B_a: (s·n_a) × n_a cubic B-spline collocation at sample points,
                                                D_a: 2nd difference on coefficients)
solve        (A_x ⊗ A_y ⊗ A_z) m = (B_x ⊗ B_y ⊗ B_z)ᵀ R
          →  three batched banded solves (bandwidth 3, i.e. solve_banded((3,3))) after the separable projection Bᵀ R
```

- `s = 1, λ = 0` must reproduce `_prefilter_3d` exactly (parity test, §5).
- `s = 2` gives the fit information *between* nodes (fixes M2); `λ > 0` forbids ringing when `R` still has sub-`h` content — such content stays in the residual instead of being aliased into oscillations.
- Uniform weights here (the target is already bounded by §3.2), which is what keeps the Kronecker structure. Non-uniform weights would need PCG with this separable solve as preconditioner — **deferred**, only if needed.
- Cost: `O(s³ N_nodes)` projection + banded solves; same order as the current prefilter. GPU port later = Thomas → 7-band solver per line.

### 3.5 Stage C — alternation (optional, 1–3 sweeps)

```
repeat:  cores ← fit_cores(T − mesh)      (Stage A, same weights)
         mesh  ← fit_mesh (T − cores)     (Stage B)
```

Block Gauss–Seidel on the joint (penalised, hence positive-definite) normal equations → converges towards the joint optimum without a joint solver. Stop when the reachable-region error stops decreasing. Answers M3/sequential-split concerns: smooth content migrates to the (penalty-cheap) mesh, sharp per-atom content to the cores.

## 4. Implementation plan (minimal, reuse first)

1. `spammm/surfaces/CoarseMesh.py`: add `fit_mesh_lsq_3d(R_ss, s, lam)` — separable `Bᵀ`-projection + per-axis banded solve of `A_a`. Reuse `_basis`; keep `_prefilter_3d` untouched.
2. `spammm/surfaces/PICCore.py`: `fit_cores_from_samples` gains `weight_mode` (`'boltzmann'` default = current behaviour, `'relative'`), no other change.
3. `tests/SPM/testplot_fdbm_pme_debug.py`: new branch `--fit coremesh` driving A → B → C with CLI `--e-cut --e-cap --ss --lam --sweeps --weight-mode`. Old branches stay as they are (debug/back-compat).
4. No change to `ContactSurface` / GPU kernels until the dimer + azaindol pass §5 and the USER confirms.

## 5. Verification (in order, stop at first failure)

| level | test | pass criterion |
|---|---|---|
| L0 | `fit_mesh_lsq_3d(s=1, λ=0)` vs `_prefilter_3d` on random data | `max|Δc| < 1e-8` |
| L0 | smooth analytic field (Gaussian sum), `s=2, λ=0` | off-node error ≤ nodal-interp error |
| L0 | step / sharp-wall 1D field, `s=2`, sweep `λ` | overshoot decreases monotonically with `λ` (ringing control works) |
| L2 | C–C dimer, `h = 0.5, 0.75, 1.0`: 1D 4-panel (split / mesh fit ×100 / core fit ×100 / total) + 2D xz (same layout as `slice_xz_*.png`), window `±2|E_min|` | no negative core tail in the basin, no rings/pockets; visually smooth error |
| L1 | metrics **only in reachable region** (`E < E_cut`, above molecule), by `r_min` band 2.3–3.2 / 3.2–4.5 / 4.5–8 Å | E rms ≪ `|E_min|` (target: ≤ 5 % of `|E_min|`), F rms reported |
| L2 | azaindol, same plots | same criteria |
| — | **df maps only after USER confirms E and F** | — |

## 6. Open parameters (to be settled on the dimer)

- `E_cut`, `E_cap` for the CO probe (start 0.3 / 1.0 eV).
- weight mode and `E_w` / `k_T`.
- `s ∈ {2, 3}`, `λ` (scan, log-spaced), number of sweeps.
- whether electrostatic short-range needs an extra `exp(−a r)·cutoff` mode per atom (only if the 5 `t^p` modes leave structured residual near charged atoms).

## 7. Implementation attempt — FAILURE RECORD (2026-10-05)

**Verdict: the hybrid fit is no better than spline-only. The decomposition as implemented is broken, not the basis.** Written so the next implementer does not repeat this.

### 7.1 What was actually built

- `CoarseMesh.fit_mesh_lsq_3d(R, s, lam)` — supersampled separable penalised mesh solve. Parity vs `_prefilter_3d` at `s=1, λ=0`: `max|Δc| = 1.6e-14` ✓ (this part is verified good).
- `--fit-mode coremesh` in `tests/SPM/testplot_fdbm_pme_debug.py`: bounded target `T = soft_clamp(E, e_cut=0.3, e_top=1.0)`, alternating `cores ← fit(T − mesh)`, `mesh ← fit(T − cores)`.
- Generalized centers: atoms (Bspl ×7 or t^p ×5) + bond midpoints (t^p ×3 × angular 1,cos²,cos⁴) → 23 unknowns on C–C.
- Core fit restricted to probe region (`z > mol+1.5`) — a **hard mask diverged** (J grew ~10×/sweep: unconstrained cores explode outside the fit region, mesh absorbs the garbage, ping-pong). Patched with weak out-of-region weight `--w-out 0.02`.

### 7.2 Numbers (C–C dimer, h=0.5, ss=2, λ=1e-2, sweeps=6)

| model | joint J (probe region) | reach rms (E<0.3) | reach max |
|---|---|---|---|
| spline-only LSQ | 1.47e-5 | 0.0178 | 0.0836 |
| spline-only interp | — | 0.0131 | 0.1022 |
| **cores+mesh (23 unknowns)** | 6.75e-6 | **0.0138** | **0.1164** |

J converged monotonically (8.1e-6→6.8e-6) yet reachable-region error is **unchanged vs spline-only and the max is worse**. Error artifacts are an error rim at the wall crest (z≈2.7–3.0, E≈0.05–0.3 band) and flanking pockets — spatial scale comparable to/larger than h=0.5 — impossible to excuse as spline representation error.

Cores-only capacity test (same T, same samples, no mesh): atom Bspl×7 + bond ×3 ang reaches **rms 0.0016 eV**. The information IS representable by the compact basis — the decomposition fails to deliver it.

### 7.3 Diagnosis — how the design degraded

1. **`T − cores` is not smooth.** The rational clamp bounds amplitude but not spatial sharpness: where the wall is steep, the clamp's rollover compresses into ~sub-Å spatial width → the mesh target contains a sharp arch/step (visible as the pointed arch in the `mesh target` panel). Handing a sub-h feature to a penalised spline guarantees ringing — exactly the artifact pattern seen.
2. **The two stages never share one objective.** Cores are weighted (`w_f`), the mesh is unweighted uniform over the whole grid; alternation then does not minimise a single functional — it is two different problems ping-ponging. J happened to converge on the probe mask, but there is no mechanism forcing `T − cores` to be mesh-representable.
3. **Misallocation at it=0 poisons the split.** Cores first fit `T` alone → they absorb smooth basin content they should never own; every later sweep inherits the misallocation (cores still carry a −0.017 negative dip at the basin edge).
4. **Added basis functions do nothing.** Bond centers demonstrably can represent the field (§7.4) but inside this split they are wasted — "a million spurious basis functions doing nothing to help the mesh remove its residual" (user). The failure is in how the residual is produced, not in basis capacity.

### 7.4 Established facts (usable by next attempt)

- Single atom is radial to ~3% on the wall; the **dimer's non-radial excess is bond-centered** — atom-only bases plateau at rms 0.05–0.10 eV no matter how many modes; +1 bond midpoint center → 0.011–0.028; + angular cos²/cos⁴ → 0.004–0.011; with Bspl atoms ×7 → **0.0016**. (`/tmp/radial_capacity.py`, `coremesh_basis_cmp_{xz,1d}.png`)
- `t^p` basis wastes resolution near `r_lo` (flat region) — radial B-splines ×5–7 per atom are ~5× better in the basin at same count.
- Core domain must come from the **measured wall** (E=1 eV → r_lo, E=0 → r_b), not generic Morse radii.
- Hard restriction of the core fit region → divergence; keep weak weights outside.
- Supersampled samples + weights: `debug/testplot_fdbm_pme_debug/cc_h0.5/coremesh_samples.npz` (pts_ss, E, T_ss, w_ss, rays, dense xz).

### 7.5 Requirements for the next attempt (the original intent, restated)

The decomposition must be **smooth by construction** — the bases share the same space and complement each other:

- Whatever the mesh receives must contain **no spatial feature sharper than ~2h**. Either: (a) mesh target = low-pass/projection of T defined first, cores fit the complement; (b) true joint weighted LSQ over {c, m} with the mesh curvature penalty — one functional, one weight — so smoothness of `T−cores` is *enforced*, not hoped for; or (c) split in a smoother variable (e.g., E^(1/β) gains only 2–4×, insufficient alone).
- The clamp must bound the target **and** its spatial derivatives where it engages — a plateau bounded in value but approached over sub-Å distance is still sharp.
- Same weighting/same objective for both stages; verify `J_mesh_solvable = mean((Proj_mesh(T) − cores − mesh)²)` decreases, not just raw J.
- Success criterion (unchanged): reach rms must be **strictly below** spline-only on the same mask, and no error structure wider than h may remain.

## 8. Projected solver (`fit_coremesh_projected`) — measured, still failing (2026-10-07)

New Codex implementation: mesh eliminated algebraically via the separable
projection (`mesh = P(T − Ac)`, batched banded solves), cores solved on the
small reduced system in complement space `Q = (I−BP)A`. Fit time: ~0.3 s (cc),
~7 s (azaindol) vs ~20 min for the weighted CG. `--bond-centers` adds explicit
bond-midpoint centers.

### Results (dense xz, reachable E<0.3, z>mol+1.5)

| case | model | rms | max | crest 2.3–3.2 |
|---|---|---|---|---|
| cc | spline-only proj. | 16.3 meV | 118 | 35.7 |
| cc | atoms only | 16.2 | 136 | 35.2 |
| cc | atoms + 1 bond | **7.5** | **78** | **16.3** |
| azaindol | spline-only proj. | 15.0 | 147 | 34.1 |
| azaindol | atoms only | 15.5 | 154 | 35.4 |
| azaindol | atoms + 16 bonds | **8.0** | **94** | **18.2** |
| azaindol | *(CG joint, atoms only)* | *3.5* | *73* | *8.7* |

Bond centers confirm their value (≈2× on both systems). But the projected fit
is **worse than the weighted CG joint** and shows a residual **checkerboard at
mesh scale** in both error panels.

### The pathology (diagnosed 2026-10-07): gauge freedom + near-singular Q

The reduced solve only constrains `Q c` — the part of each core mode the mesh
CANNOT represent. With `core_span = 3 Å`, `r_b ≈ 4.7 Å`, the C2 smooth core
modes are almost entirely representable by the h=0.5 mesh → `Q` nearly
singular → the 1e-8-scaled ridge fails to pin the near-null directions →
core coefficients explode. The mesh then cancels exactly what it projected
out: `joint_ray_*.png` shows **core ≈ +0.8 eV spike, mesh ≈ −0.8 eV spike,
sum correct**. Mathematically consistent, physically useless: the "cores"
no longer represent the compact short-range wall, and the force components
oscillate ±0.8 eV/Å while cancelling.

The joint objective only constrains the SUM; nothing separates "sharp content
→ cores" from "smooth content → either". Both solvers share this issue; CG
survives it better only because the weighted mesh operator damps the gauge.

### Fixes to try next (ordered)

1. **SVD cutoff on Q**: project `c` onto the non-null complement directions
   only — content the mesh can hold must live in the mesh, coefficient
   directions with `‖Qc‖ ≈ 0` must be zeroed, not ridge-floated.
2. **Shrink `core_span`** (3 Å → ~1–1.5 Å): make cores genuinely sharper than
   the mesh resolution so the complement is non-degenerate *and* the physical
   split matches (cores = short-range wall only).
3. **Penalize `‖c‖`** in the same units as the data term, not scaled by
   `core_ridge = 1e-8` — smooth content should prefer the mesh.
4. Weighted mesh projection (or accept CG for final fits; projected = preview).

## 9. Related

- `doc/Tasks/PME_ContactSurface_Opt.md`, `doc/Tasks/ContactSurface_PME_ParallelPlan.md`
- `doc/Reports/ContactPME_Split_Rcut_Locality_2026-10-04.md` (core span Δ_in, Δ_b)
- `doc/Reports/ContactPME_Supersample_Scale_2026-10-04.md` (why that result does not transfer, §2)
- `spammm/surfaces/PICCore.py` (`core_basis`, `fit_cores_from_samples`, `eval_core_fast`), `spammm/surfaces/CoarseMesh.py` (`_basis`, `_prefilter_3d`), `spammm/SPM/AFM_utils.py` (`soft_clamp_rational`)

## 10. Overlapping cores-first implementation — under verification (2026-10-07)

USER clarification: cores and spline occupy the same space and augment each
other. There is no geometric surface at which the physical potential switches
representations. Prefit cores first so the initial mesh target is their residual.

`CoarseMesh.fit_coremesh_lsq` now implements this directly. It first solves the
weighted core prefit, then uses that as the initial guess for a matrix-free CG
solve of **one** objective: weighted total-field error plus additive per-axis
mesh second-difference penalties and a small core ridge. Both bases see exactly
the same samples and relevance weights. There is no alternating pair of
incompatible weighted/unweighted objectives, residual cap, inpainting, or mask
dilation. The mesh remains free to contribute inside and outside every core.

The existing five powers and evaluator are retained. A null-space transform
imposes zero slope/curvature at the flat inner extension and zero curvature at
the outer support; value/slope already vanish there. This gives C2 joins without
asking the mesh to fit any boundary values. It leaves **two independent core
combinations per center**, an explicit capacity limitation of keeping the old
five-power evaluator. Small per-center whitening reduces their conditioning
problem; no global dense core normal matrix is formed. Sparse axis matrices
avoid a 64-tap 3D fitting matrix. Sparse neighbor pairs avoid a samples×atoms
distance array. `eval_mesh` now has bounded vectorized batches and preserves its
scalar path as an evaluation oracle.

The diagnostic's new default is `--fit-mode joint`; old `clamp` and `coremesh`
branches remain available for reproducing the failure. Its target uses the
optional C2 continuation of `soft_clamp_rational`, which is unchanged below
E_cut. Relevance weights use ORIGINAL energies (Boltzmann T=0.1 eV by default),
with weak continuation weights and a smooth z preference shared by both bases.
The density-clamp default remains unchanged. Numerical core radius is measured
at E=20 eV along each atom ray; outer support is a numerical span (3 Å), not an
assertion that a non-pairwise field vanishes there. Cached wall-ray interpolation
is used only to initialize this support; independent oracle rays/xz samples
provide the accuracy measurement.

Quick reproduction:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 python3 tests/SPM/testplot_fdbm_pme_debug.py cc --samples debug/testplot_fdbm_pme_debug/cc_h0.5/coremesh_samples.npz
pytest tests/SPM/test_bspline_boundary.py --develop -s
```

Measured cached C–C result, h=0.5 Å, ss=2, λ=1e-4, same reachable xz mask:
hybrid E rms **0.00294898 eV**, maximum **0.0460023 eV**; weighted spline-only
rms **0.00405824 eV**, maximum **0.0598506 eV**. The same penalized objective
decreases from 0.0782469 to 0.0379712. Float32 coefficients/geometry occupy
103568 bytes. On 5547 supersampled points with original E>=1 eV, minimum model
E is 0.340283 eV: no sampled deep-wall point becomes accessible below E_cut=0.3.
This is a sampled guard, not a proof over the continuous interior.

Artifacts: `debug/testplot_fdbm_pme_debug/cc_h0.5/joint/joint.out`,
`joint_xz.png`, `joint_ray_0.png`, `joint_ray_1.png`, `joint_model.npz`.
Synthetic regression independently checks off-grid E and F improvement,
continuity, sparse/dense basis parity, scalar/vectorized evaluator parity,
analytic F=-grad(E), and fail-loud inputs/nonconvergence.

**Still unverified:** cached FDBM data contain no force oracle (ray Fz is clearly
labelled as finite differences of cached E); real-field force/df parity,
azaindol, coarser meshes and scan integration have not been established. Current
unweighted reachable rms also exceeds the aspirational 5% of basin-depth target.
No status is promoted to fixed/resolved; USER review remains required.

## 11. Stop — fix the radial basis FIRST, before any mesh coupling (2026-10-07)

USER direction (verbatim requirements, restated):

1. The core-fit domain is a **geometric shell in space**, not an energy-weighted
   box: a sample point is IN if any atom is closer than `Rcut`, OUT if any atom
   is closer than `Rmin` (i.e. `Rmin < min_atom_distance < Rcut`). Boltzmann
   weights may apply *inside* the shell, or not — secondary detail. Verified
   visually: `debug/testplot_fdbm_pme_debug/{cc,azaindol}_h0.5/coremask_xz.png`.
2. **Forget the mesh until the atomic basis demonstrably fits the core region.**
   If a compact radial basis cannot reproduce a wall that is smooth and only
   mildly curved in 2.4–4.0 Å, no mesh will rescue it. Basis capacity is the
   blocker, not solver mechanics.
3. Only after the radial basis alone fits the shell region to a small residual
   does the mesh have a sensible job: a smooth remainder.

### 11.1 Why the current basis is horrendously bad — measured, not guessed

`core_basis(r, r_lo, r_b, powers, smooth)` defines `t = (r_b−r)/(r_b−r_lo)`,
optional smootherstep `s(t)`, modes `s(t)^p, p ∈ {2,4,8,16,32}` — flat (=1)
below `r_lo`, zero beyond `r_b`. In the shell experiment `r_lo=Rmin`, `r_b=Rcut`
per center, so the basis ramp is locked to the shell. Its defect:

- **All sharpness is at the WRONG edge.** `s(t)^p` with large p stays ≈0 through
  most of the shell and rises steeply only as `r→r_lo` (t→1) — i.e. the steep
  resolution sits at the INNER edge, against the flat-plateau clamp where the
  target is already constant. The physically relevant steep part of the wall
  (E ≈ 0.05–0.3 eV, the wall foot) lives in the OUTER half of the shell where
  only the broad shallow modes (p=2,4) are active. The basis cannot put a
  transition where the transition is.
- Shell-restricted fit confirmed the symptom: core rms inside the shell stuck
  ~0.014 eV, residual keeps a systematic −0.02 eV dip exactly at the wall foot,
  alternation does not converge it away
  (`debug/testplot_fdbm_pme_debug/cc_h0.5/shellfit_1d.png`).
- Earlier capacity probe already showed t^p ×5 vs radial Bspl ×7 on [r_lo,r_b]:
  0.0092 → 0.0016 eV rms. Knots placed where the wall is beat a power ladder
  that bunches at one endpoint. This was ignored when 'smooth' modes were
  adopted — the C2-join convenience cost the resolution placement.

### 11.2 Required basis properties (for the next implementation)

- Resolution distributed where the wall is: radial **B-splines** (or equivalent
  local radial modes) over [Rmin, Rcut], ~5–7 knots, C2 joins at both ends.
- No reliance on the flat inner clamp to carry anything — inside Rmin the target
  is excluded from the fit anyway; the basis need only be bounded there.
- Bond midpoints as extra centers (validated: cc rms halves, azaindol halves);
  optional cos^2θ/cos^4θ factors along the bond axis.
- Fit domain = geometric shell §11.1; report core-only shell residual BEFORE
  any mesh stage is switched on.

### 11.3 Acceptance gate (do not skip)

Fit the C–C dimer shell with the candidate basis alone and plot the fitted
radial profile against the reference along wall rays. Proceed to mesh coupling
only if the core-only residual in 2.4–4.0 Å is small (~few meV) and visibly
smooth. Any new basis must also come with a parity check between the fitting
basis and the evaluator (`eval_core_fast`) — basis/evaluator mismatches have
burned this project before.

## 12. Radial-basis survey, no mesh (C–C dimer, 2026-10-08)

Script: `/tmp/radial_survey.py` (temporary). Data: cached
`cc_h0.5/coremesh_samples.npz`. Fit region: shell `2.25 < r_min < 4.0 Å`,
`z > 0.5`, 7011 3D samples (0.25 Å); test on dense xz cut (0.05 Å, 3782 pts).
Uniform LSQ for coefficients; exponents (+ optional global center z-shift)
by Nelder-Mead. Plot: `cc_h0.5/radial_survey/survey_xz_rays.png`.

| config (per atom / bond) | ncoef | test rms | test E<0.3 rms |
|---|---|---|---|
| any family, atoms only, n=2..5 (Slater, Gauss, r^-k) | 4–10 | 59 meV | 26–27 |
| compact B-spline [Rmin,Rcut] atoms ×5 | 10 | 65 | 33 |
| current `tp` (s(t)^p) atoms ×5 | 10 | 69 | 35 |
| Slater atoms ×3 + bond ×2 | 8 | 9.4 | 5.9 |
| same + global z-shift | 8 | 7.6 | 5.1 |
| Slater atoms ×2 × (s,px,py,pz) | 16 | 10.2 | 6.0 |
| Slater atoms ×2 × (s,pz) | 8 | 52 | 36 |

Findings:
- **Atom-only radial fits hit a hard floor (~52–59 meV) independent of family and
  of n=2..5.** The limit is geometric (the field is not a sum of atom-radial
  functions — excess between atoms), not radial-shape capacity.
- **One bond center with 2 exponentials removes 85% of it** (59 → 9 meV). Adding
  radial functions beyond (3 atom, 2 bond) changes nothing.
- Bond center (2 coefs) ≈ p-angular along the bond (6 extra coefs/atom) →
  bond centers are the efficient choice, as the USER suspected.
- z-angular (`spz`) and center z-shift (tip asymmetry) are minor in the z>0.5
  region (shift → −0.017 Å once a bond center exists).
- Non-compact exponentials beat compact bases on fixed [Rmin,Rcut]; the
  current `tp` basis is the worst family tested.
- Below the molecule the field is strongly asymmetric (tip body overlap: at
  r=3.0 Å E≈−0.02 above vs up to 12 eV below) → fit only the probe side.
- Remaining error with bond center: smooth ±10 meV pattern at the wall foot
  over the bond; basin depth ≈27 meV, so still ~20% of basin. Not yet solved.

### 12.1 Fit window and weights (Rmin sweep)

Rmin=2.5 Å, uniform weights is best (shell then holds only E≤0.35 eV):
Slater at3+bd2 in-shell rms 2.4 meV, max 27 meV; E<0.3 on fixed test
region 4.3–5.0 meV. Boltzmann weighting only helps when Rmin is too small
(acts as a soft inner cutoff); T=0.1 always hurts (kills the 0.1–0.3 eV foot,
which the probe does visit). Plots: `radial_survey/fitregion_r*_T*.png`.

### 12.2 Why `tp` (t^p) fails although it was meant as a polynomial Slater

`t^p = ((r_b−r)/D)^p ∝ (1 − r/r_b)^p ≈ exp(−p r/r_b)` only for r ≪ r_b. In the
shell r/r_b ∈ [0.62, 1] → invalid. Local decay rate p/(r_b−r) diverges at r_b;
mid-shell it is 2.7/5.3/11/21/43 Å⁻¹ for p=2/4/8/16/32 vs fitted Slater
a≈1–3 Å⁻¹ → p≥8 modes are ≈0 across the shell. Smootherstep additionally
gives dφ/dr = 0 at r_lo = Rmin, exactly where the target slope is largest,
and all modes are forced to 0 at r_b = outer fit edge. Plot:
`radial_survey/modes_r2.5-4.0_T0.0.png`.

Correct polynomial analog of exp(−a r): `(1 − a r/N)₊^N`, fixed N, per-mode
cutoff R_k = N/a_k, even-tempered a_k (`poly_aN`). C–C, Rmin=2.5, at3+bd2,
in-shell rms: Slater 2.48, **poly_a8 2.55**, poly_a4 2.55, poly_R (shared free
cutoff) 5.4, bspl 10.8, **tp 13.3 meV**. poly_a8 matches Slater and is compact;
poly_a16 fits in-shell but extrapolates wildly (ill-conditioned).

### 12.3 Exact basis definitions (r = distance to center, k = 0..n−1)

| name | φ_k(r) | parameters |
|---|---|---|
| `slater` | `exp(−a_k r)`, `a_k = a0·β^k` | a0, β fitted (non-compact) |
| `poly_a8` | `(1 − a_k r/8)₊⁸`, `a_k = a0·β^k`; cutoff `R_k = 8/a_k` | a0, β fitted; → exp(−a_k r) as N→∞ |
| `poly_a8` constrained (proposed) | same, with `R_k` geometric from R_max (~9 Å) down to Rcut | guarantees every mode is non-zero in the shell |
| `poly_R` | `(1 − r/R)₊^{p_k}`, `p_k = 2,3,4,…` | one shared cutoff R fitted |
| `tp` (PICCore `core_basis(smooth=True)`) | `t=(Rcut−r)/(Rcut−Rmin)` clipped to [0,1], `s=6t⁵−15t⁴+10t³`, `φ_k = s^{p_k}`, p=2,4,8,16,32 | none; =1 for r<Rmin, 0 for r>Rcut |
| `bspl` | cubic cardinal B-spline, Δ=(Rcut−Rmin)/(n+1), centers Rmin+kΔ, r→max(r,Rmin) | none |

Why poly_a8 works and poly_R / tp do not: in poly_a8 each mode has its **own
cutoff** R_k = N/a_k far enough out that its log-slope `a_k/(1 − a_k r/N)` is
roughly constant across the shell — a genuine Slater-like decay with a
spread of decay rates. poly_R and tp share **one** zero (R or Rcut) for all
modes, so every mode's log-slope `p/(R − r)` diverges at the same point: the
family is a set of increasingly sharp "knees" at one radius, not a set of
different decay lengths. tp additionally flattens (zero slope) at Rmin.

Caveat seen in the fitted poly_a8 (a0=0.894, β=2.571): modes k≥2 have
R_k = 1.35, 0.53, 0.20 Å < Rmin → identically zero in the shell (dead). The
3-mode fit effectively used 2 atom modes. Constrain R_k ≥ Rcut (row 3).
Plot of all sequences: `radial_survey/basis_sequences.png`
(script `/tmp/plot_basis_seq.py`).

## 13. Sequential core → mesh on azaindol — first working result (2026-10-08)

Script `/tmp/core_then_mesh.py` (temporary), cached samples, no new FDBM run.

1. **Core**: `φ_k = (1 − r/R_k)₊⁸`, fixed cutoff ladder R_k geometric 9 → 3.5 Å
   (atoms 3 modes: 9, 5.6, 3.5; bonds 2 modes: 9, 3.5), 15 atoms + 16 bond
   centers (d<1.6 Å) = 77 coefs. Plain LSQ, uniform weights, on samples with
   `2.5 < r_min < 4.0` and `z > mol_z + 0.5`. No nonlinear parameters.
2. **Mesh**: residual `ref − core` on the full sample grid → weighted P-spline
   (`fit_coremesh_lsq` with no atoms = mesh-only CG, λ=1e-3), weight 1 where
   `r_min > 2.5` and `z > mol_z + 0.5`, else 0. **No outer cut** — the mesh shares
   the 2.5–4.0 shell with the core; fit is sequential (core first, fixed).
3. Baseline: same mesh fit on `ref` directly (spline-only).

Dense xz cut (0.05 Å, y=0.76, r_min>2.5), rms / max in meV:

| mesh h | band | core only | **core+mesh** | spline only |
|---|---|---|---|---|
| 0.5 Å | E<0.3 | 4.0 / 52 | **1.36 / 32** | 4.05 / 72 |
| 0.5 Å | r 3–4 | 3.0 / 13 | **0.65 / 4.8** | 2.6 / 22 |
| 0.5 Å | r>4 | 3.9 / 6.6 | **0.09 / 0.8** | 0.33 / 2.6 |
| 1.0 Å | E<0.3 | 4.0 / 52 | **1.84 / 47** | 9.5 / 126 |
| 1.0 Å | r 2.5–3 | 7.2 / 56 | **6.5 / 53** | 41 / 408 |
| 1.0 Å | r 3–4 | 3.0 / 13 | **1.43 / 8.3** | 8.6 / 33 |
| 1.0 Å | r>4 | 3.9 / 6.6 | **0.40 / 2.0** | 3.4 / 14 |

- The residual `ref − core` is smooth and long-ranged (visible in
  `azaindol_h0.5/core_then_mesh/xz_*.png`); a **1 Å mesh** then fits it, giving
  5× lower rms than spline-only at 1 Å, and better than spline-only at 0.5 Å.
- Core alone has a smooth ~3–6 meV error at r>4 (finite cutoff) — exactly what
  the mesh removes (0.4 meV at h=1).
- Remaining error is concentrated in the thin band r_min 2.5–3.0 (wall foot,
  ~6 meV rms, max ~50 meV at the 2.5 Å edge). The ref rays show fine ripples
  there (likely FDBM grid interpolation), which no smooth basis follows.
- 4 atom + 3 bond modes (108 coefs) improves E<0.3 only slightly (1.58 vs 1.84
  at h=1). Cutoff ladder matters: Rlow=2.8–3.0 with 3+2 modes is worse
  (ill-conditioned); 9 → 3.5 was best on C–C.
- Unverified: forces, other molecules, z<mol_z+0.5 behaviour (excluded),
  GPU evaluator. Status: experimental, USER review required.

## 14. Core-only residual study — canonical figure + basis/cutoff survey (2026-10-08)

The mesh and core are judged separately: first the core must leave a smooth,
slowly-varying residual `R = FDBM − E_core`; only then does a coarse mesh take it.

### 14.1 Canonical visualization (SSOT)

- Compute: `spammm/surfaces/CoreBasisStudy.py` — spec grammar, design matrix,
  weighted LSQ (`fit_core_lsq`, force rows via `wF`), `core_residual_study`
  (band/slice/high-pass metrics on the oracle grid).
- Figure: `spammm/surfaces/surface_plots.py::plot_core_residual_study` — one row
  per fit variant; columns = E residual on an xz cut + Fz residual on xy slices
  at h = 2.5/3.0/3.5/4.0 Å. Shared symmetric clim per column; NaN outside the
  display domain (r_min < rin or z below mol+0.5). Contours: black r_min=2.5,
  green r_min=rcut. Row labels carry the high-pass rms.
- Driver: `tests/SPM/testplot_coremesh_basis_survey.py --rcuts … --rins … --wFs …`
  → `debug/testplot_coremesh_basis_survey/<mol>_corestudy_*.png`.

### 14.2 Basis definition (selected Pareto optimum `A:s3p2+B:s1`)

Per **atom** (9 coefs): s-radials R = {9, 5.61, 3.5} Å + p-radials R = {7, 3.5} Å
(3 components each). Per **bond midpoint** (1 coef): s-radial R = 5 Å.
All `φ = (1 − r/R)₊⁸`; p-term = `φ(R)·n_α`, n = (x−C)/|x−C|.
151 coefs on 31 centers for azaindol. Fit: LSQ on shell rin<r_min<rcut,
z > mol+0.5; force-aware variant appends wF·∂E rows (target −wF·F).

### 14.3 Results (azaindol; consistent on PTCDA, pentacene)

- **Force fitting is the main lever for smoothness**: wF=0.3 cuts the high-pass
  Fz residual ~2.5× (3.4→1.4 meV/Å azaindol; 5.5→2.4 PTCDA; 4.4→1.6 pentacene)
  and Fz rms at h=2.5 from 16–26 to 4–6 meV/Å, at a small cost in near-wall E rms.
- **Widening the fit shell (rcut 4→5–6 Å) does NOT smooth the residual** — it
  trades near-wall accuracy for a small gain in the 4–5 Å band (2.7→2.1 meV).
  That outer residual is a smooth vdW/ES tail ≈ mesh job anyway. rcut=4 kept.
- **Shortest radial R_LO = 3.5 Å is optimal** (3.0/4.0/5.0 all worse, 5.0
  ill-conditioned). The basis cutoff = PIC bounding-box cutoff — keep shared.
- **Lowering rin to 2.3 Å hurts** (band 2.5–3: 2.34→3.26 meV, HP Fz 1.4→4.6):
  the 2.3–2.5 band is too steep for the basis and corrupts the whole fit.
- **A short p-shell (R=3.0, `p[7,3.5,3.0]`) adds ~nothing** (HP Fz 1.42→1.40):
  the p-channel is saturated; R=2.5 is identically zero in the domain.
- **One d-shell** (`A:s3p2d1+B:s1`, R=4, +5/atom) gives a real but modest gain:
  Fz@2.5 4.1→2.9, HP Fz 1.42→1.22 — candidate if the extra cost is acceptable.

Residual after core (wF=0.3, rcut=4): ±20 meV/Å atom-localized at h=2.5 (wall
foot — only sharp feature left); ≤1.5 meV/Å smooth blobs at h≥3.0.

## 15. poly8sp GPU kernel path (angular s/p core in ContactPME)

The angular basis now runs in the existing contact_pme kernels — no new kernel
family; `cs_pme_sp_accum` is called from the two core-eval funnels
(`cs_pme_core_eval_at` bucket loop, `cs_pme_core_eval_local_at` local-mem loop)
so all eval/relax/tile paths get it for free.

- **Build flags** (`AFMulator(pme_core_basis='poly8sp', pme_poly_R=Rlad)` or
  `set_pme_core_basis('poly8sp', ...)`): `CS_PME_SP_CORE=1` + `CS_PME_POLY_CORE=1`
  + `CS_PME_NMODES=3+3·NP` + `CS_PME_POLY_R0..R{2+NP}` ladder.
- **Slot ABI** (per center, `CS_PME_NMODES` floats): 0–2 s radials at
  R0..R2·w; then NP p-shells, slots 3+3k..5+3k = `(c·nx, c·ny, c·nz)` at
  R_{3+k}·w. `w` = **per-center radius scale** in the atom w-channel: atoms 1.0,
  bond midpoints `Rb/R0` (5/9 → bond s is R=5 Å = 'B:s1'). Effective per-center
  support `r_b_i = w·d_span` (`d_span` = `core_span_override` = R0·max w = 9 Å).
- **p-term gradient**: `∇(n·P) = n·(n·P') + (P − n(n·P))/r` — tangential term,
  not purely radial like the scalar modes. `cs_pme_sp_accum` returns the
  GRADIENT; callers subtract it for force.
- **Host**: `PICCore.fit_core_sp_shell` (spec param, e.g. `'A:s3p2+B:s1'` or
  atom-only `'A:s3p3'`) packs `CoreBasisStudy` mode-major coefs into the slot
  layout (asserts the order matches); `PICCore.eval_core_sp` is the CPU
  reference; `PICCore.sp_ladder(spec)` → kernel `poly_R` ladder;
  `cpm_params_from_coremesh` accepts `basis='poly8sp'` (nmod from `poly_R`
  length; PIC buckets built with cs = max r_b). `AFM._pme_n_modes()` =
  `3·(len(poly_R)−2)` drives all local-mem coefficient allocations
  (eval/relax/tile) — the old `5*4` literals are gone from the shared paths.
- **Harness**: `testplot_fdbm_fields_compress.py --method coremesh
  --core-basis poly8sp [--core-spec 'A:s3p3']`; output dir is tagged
  `coremesh_poly8sp-<spec>`. `poly8` path untouched (`--core-basis poly8`).
- **Parity** (`test_bspline_boundary.py::test_coremesh_poly8sp_eval_parity`,
  plus ad-hoc 12-mode check): GPU local+bucket vs `eval_core_sp` max|dE|≈1.5e-7,
  max|dF|≈7e-7; host F vs FD(E) ≈4e-6.

### End-to-end relaxed PP-AFM (h=1 Å mesh, wF=0.3, shell 2.5–4.0 Å)

| h_df [Å] | 3.7 | 3.9 | 4.0 | 4.1 | 4.2 | 4.3 | 4.4 | 4.5 | 4.6 | 4.7 |
|---|---|---|---|---|---|---|---|---|---|---|
| df corr, **poly8** (5-slot s-only) | .93 | .95 | .92 | .86 | .82 | .89 | .95 | .98 | .995 | .997 |
| df corr, **s3p2+B:s1** | .988 | .991 | .980 | .960 | .983 | .991 | .993 | .994 | .997 | .999 |
| df corr, **s3p3** (atoms only) | .985 | .988 | .980 | .975 | .989 | .994 | .996 | .997 | .998 | .999 |

PTCDA / pentacene land at worst ~0.963/0.936 (s3p2+B:s1, h_df 4.1) and
~0.966/0.948 (s3p3). The old two-lobe artifact at h_df 4.0–4.3 is gone.
s3p3 (atom-only, 12 slots) is slightly *smoother* in the mid-range than
s3p2+B:s1 — consistent with the core-residual survey (its atom p-shells absorb
the directional residual the bond-center s's miss at ~3 Å). Cost is ~1.3× the
poly8 eval; scan still ≪0.1 s on RTX 3090. cond ~3–6e6, max|coef| ~1e4 —
comfortable in float32 but keep an eye on it if more shells are added.

## 16. GPU matrix-free core fit (CGLS, no atomics) — design (2026-10-09)

### Why
Scaling the core fit to larger PAHs with the CPU `np.linalg.lstsq` path is
impossible: `fit_core_lsq` materializes the dense design `(n_rows × ncoef)`
with `n_rows = 4·n_shell` (E + 3 force rows) and `ncoef = 9·n_atoms + n_bonds`,
plus `(npts × ncenters)` distance broadcasts for the shell mask. On
circumcoronene (72 atoms, 1.6 M-pt sample grid) this reached >15 GB RSS and the
process was OOM-killed (PTCDI, 40 atoms, still fit in 12 s). The problem is
structurally a *small-unknown / huge-sample* linear LSQ: ncoef ≤ ~1500 even for
120-atom flakes, n_rows ~1e5–1e6. The right tool is an iterative solver whose
only primitives are `A·v` and `Aᵀ·v`, both O(ns + nc) in memory — and both are
exactly the operations the PP-AFM kernels already perform.

### Operators (kernel side, `kernels/contact_surface.cl`, under `CS_PME_SP_CORE`)
- **Forward `cs_sp_Av`**: per sample `i`, `(wF·∂E/∂x, wF·∂E/∂y, wF·∂E/∂z, E)`
  with the coefficient buffer holding the trial vector `v`. The field is linear
  in the coefficients, so this is *verbatim* `cs_pme_core_eval_local_at` (the
  relaxation-kernel funnel, `cs_pme_sp_accum` inside) with F→−∇E; one float4
  per sample. Local backend only (all centers + coefficients preloaded in
  `__local`; fail loud if `nat·(16 + 4·NMODES)` exceeds device local memory).
- **Adjoint `cs_sp_Atv`** (the "accumulate dE/dc back onto the DOFs" step):
  `out[c,m] = Σ_i φ_m(x_i−C_c)·v_i.w + wF·∇φ_m(x_i−C_c)·v_i.xyz`. Needs the
  *uncontracted* basis `cs_pme_sp_basis(dx,dy,dz,r,w → φ[NMODES], ∇φ[NMODES][3])`
  (s slot: `φ=u⁸`, `∇φ=−(N/R)u⁷·n`; p slot (k,a): `φ=ph·n_a`,
  `∇φ = n_a·dph·n + ph·(e_a − n_a n)/r`). `cs_pme_sp_accum` stays as the fast
  contracted evaluator; the two are tied by the adjoint dot-test below.
- **No atomics — owner-computes gather + two-pass reduce.** Work-group `g`
  owns a contiguous block of samples (`SB` per group, e.g. 1024). Inside the
  group the roles are *transposed*: samples are streamed through `__local` in
  tiles of `lsize` (pos + residual float4, ~7 floats each), and each work-item
  owns a disjoint set of centers (`at = lid, lid+lsize, …`). For every owned
  center it loops over the tile, evaluates `cs_pme_sp_basis` once per
  (sample, center) pair inside the cutoff and accumulates all `NMODES` slots in
  private registers, then adds them to its exclusively-owned entries of a
  per-group `__local` copy of the coefficient vector (`nc ≤ ~1500 floats = 6 KB`).
  No two work-items ever touch the same accumulator → plain `+=`, no local or
  global atomics. At the end each group writes its copy to
  `partial[g·nc … (g+1)·nc)`; a trivial second kernel `cs_reduce_groups` sums
  the `ngroups` copies per coefficient (`ngroups ≈ ns/SB` ~ 1e2–1e3 →
  partial buffer a few MB). Cost per Atv = one basis pass over all pairs, same
  as the forward; deterministic (no atomic ordering noise).
- **Column norms** (Jacobi preconditioner): same gather kernel with `sq=1`,
  accumulating `φ² + wF²|∇φ|²` instead of `φ·v` (no residual input).

### Solver (host, `ContactSurfaceCL.fit_core_sp_cg`)
CGLS (CG on the normal equations without ever forming them — same
Av/Atv-per-iteration cost as `fit_pic_cg`, numerically conditioned by `κ(A)`
rather than `κ(A)²`) on the **column-scaled** problem `Â D z = b̂`,
`D = diag(1/‖â_j‖)`, `c = D z`. Column scaling is essential here: the s/p
modes at R = 9…3.5 Å differ in column norm by orders of magnitude (that is
the cond ~6e6 and the ~1e4 coefficients); without it CG would crawl.
`Â = [A_E; wF·A_∇]`, `b̂ = [E_ref; −wF·F_ref]` as one float4-per-sample buffer,
so all vector ops are the existing `addMul`/`setLinear`/`dot_wg` on `4·ns`
floats. Optional Tikhonov `λ` folded in (`s −= λ²x`, `‖q‖² += λ²‖p‖²`).
Stop on `‖Âᵀr‖/‖Âᵀr₀‖ < tol` or `n_iter`. Returns `coeffs (nc, NMODES)` in the
kernel slot layout (no repacking — the GPU already uses this layout) and a
history of the residual norms. The fitted coefficient buffer stays resident so
`eval_core_sp_gpu(pts)` (= `cs_sp_Av` with `wF=1`, F=−∇) evaluates the core on
the full mesh-sample grid without the CPU `eval_core_sp` pair gather.

### Sample selection without broadcasts
Shell mask `rin < r_min < rcut` via `cKDTree(apos).query(pts)` →
`PICCore.min_dist_to_atoms(P, apos)` replaces every `P[:,None,:]−apos[None]`
broadcast (PICCore ×2, CoreBasisStudy, compress harness ×2). `eval_core_sp`
(CPU reference) keeps its pair gather but with a bounded chunk.

### What stays
`CoreBasisStudy.fit_core_lsq` / `design` remain the **CPU reference** for the
survey scripts (small molecules only — documented as such); `fit_core_sp_shell`
gets `solver='lstsq'|'gpu'` (lstsq default for back-compat; harness uses gpu).
Parity: (1) adjoint dot-test `⟨Âz, v⟩ = ⟨z, Âᵀv⟩` on random z, v (ties
`cs_pme_sp_basis`/`∇` to `cs_pme_sp_accum`); (2) CGLS vs `lstsq` on a synthetic
`Â c_true + noise` target and on azaindol — compare *predicted E/F on the
shell* (coefficients may differ along near-null directions at cond ~1e6), and
the end-to-end df correlations must reproduce §15.

### Implemented + measured (2026-10-09, RTX 3090, h=1 Å mesh, wF=0.3, shell 2.5–4 Å)
Kernels `cs_pme_sp_basis` / `cs_sp_Av` / `cs_sp_Atv` / `cs_reduce_groups`
(+ `cs_mul` Hadamard helper), host `ContactSurfaceCL.fit_core_sp_cg` +
`eval_core_sp_gpu`, `PICCore.fit_core_sp_shell(solver='gpu', afm=...)`,
`PICCore.min_dist_to_atoms`. Deviations from the plan above: bond centers get an
`active` column mask (slot 0 only, = 'B:s1') — unmasked the GPU layout gives
bonds extra DOFs lstsq never had and a different minimum (df_corr 0.9385 vs
0.9598); `n_iter` default 4000 (500 was not converged: df_corr 0.935);
zero-norm columns (bond slots that see no shell sample) are simply frozen at 0.
Tests: `test_sp_cg_adjoint_dot_test` (⟨Âz,v⟩−⟨z,Âᵀv⟩ ≈ 2e-6, NMODES 9 and 12),
`test_fit_core_sp_cg_vs_lstsq` (predicted E/F rel RMS 4e-7).

Size scaling, `A:s3p2+B:s1` (`--skip-spline-only`; mesh = scipy CG on CPU):

| molecule | atoms | centers | ncoef | sample pts | mesh | core fit [s] (CGLS it) | mesh fit [s] | scan [s] | peak RSS | df corr min/max |
|---|---|---|---|---|---|---|---|---|---|---|
| azaindol | 15 | 31 | 279 | 1.07 M | 27×27×25 | 3.2 (1624) | 8.7 | 0.04 | 2.2 GB | .960/.999 |
| pentacene | 36 | 76 | 684 | 1.61 M | 33×33×25 | 6.0 (3024) | 14.5 | 0.08 | 3.2 GB | .936/.999 |
| PTCDA | 38 | 82 | 738 | 1.42 M | 31×31×25 | 7.7 (4000*) | 13.3 | 0.08 | 2.9 GB | .963/.999 |
| PTCDI | 40 | 86 | 774 | 1.52 M | 32×32×25 | 7.7 (4000*) | 15.5 | 0.09 | 3.3 GB | .961/.999 |
| circumcoronene | 72 | 162 | 1458 | 3.46 M | 48×48×25 | 9.5 (4000*) | 120 | 0.20 | 6.7 GB | .933/.999 |

`A:s3p3` (atoms only, 12 slots): same picture; core fit 3.9–17.4 s, mesh
8–104 s, df corr min .975/.948/.964/.968/.926. (*) = hit the 4000-iteration
cap with ‖Âᵀs‖/‖Âᵀs₀‖ stalled at 1e-5…1e-4 — float32 operator floor, not a
convergence failure (df_corr unaffected; the converged cases land at the same
values). Archives 180–560 KB vs 0.65–2.2 GB dense (~9000×).

Reading: the **core fit is now flat in system size** (per-sample cost bounded
by the 9 Å cutoff; the iteration count, not ncoef, sets the time — a better
preconditioner than Jacobi or float64 accumulation in the reduce would cut the
~4000 iterations). **Evaluation/scan is ≤0.2 s everywhere**. The **mesh fit
is the only thing that scales badly**: it is the CPU scipy CG over the full
supersampled grid, 120 s on the 72-atom flake (3.5 M samples, 57 k nodes).
circumcircumcoronene (120 atoms) fails earlier still — the dense FDBM field
image (784×768×336 float4 ≈ 3.2 GB) does not fit on the 3090 next to the
resident FFT buffers (`MEM_OBJECT_ALLOCATION_FAILURE` in
`setup_fdbm_grid_from_img`), i.e. the *reference* field generation at step
0.1 Å is the limit there, not the compact representation.

### Out of scope here
The mesh fit (`fit_coremesh_lsq`, scipy CG on the full supersampled grid,
~14 s each for PTCDI, ×2 with the spline-only control) is the next bottleneck
and scales with volume; it can take the same Av/Atv treatment with the
B-spline operators (`cs_bspline_prefilter_lines` exists) — separate step.

## 17. Mesh fit without CG — omit-by-inpaint + separable banded solve (2026-10-09)

The weighted supersampled CG was only needed because the data-weight mask
(`r_min<2.5 Å`, `z<z_min`) breaks separability of `BᵀWB`. The production PAW
path solved the same problem long ago: **omit the too-close points** by
inpainting them with a smooth continuation, then the separable banded solve
applies directly (`cpm_params_from_samples` recipe). Reused verbatim:

1. residual at mesh nodes (`resid_ss[::s,::s,::s]`);
2. mask = `r_min<2.5 | z<z_min` on nodes, `binary_dilation(it=1)`;
3. `inpaint_residual` (harmonic Laplace fill, PICCore.py) → `fit_mesh_lsq_3d(s=1, lam)`;
4. if `s>1`: mask on the ss grid → fill those cells with the node-fit mesh
   evaluated there + `inpaint_residual(..., seed=seed, n_iter=150)` →
   `fit_mesh_lsq_3d(rf_ss, s=4, lam)` — still all banded solves.
   (`inpaint_residual` gained an optional `seed` parameter.)

`--mesh-fit {inpaint,cg}` in `testplot_fdbm_fields_compress.py`, default
`inpaint`; the CG path stays for debugging.

**Results** (df_corr min over h_df 3.7–4.7; mesh time; total fit):

| molecule | CG: df min / mesh s | inpaint: df min / mesh s / total fit s |
|---|---|---|
| azaindol | .960 / 8.7 | **.984 / 0.8 / 3.5** |
| PTCDA | .963 / 13.3 | .957 / 1.2 / 7.8 |
| pentacene | .936 / 14.5 | **.985 / 1.4 / 7.9** |
| PTCDI | .961 / 15.5 | **.967 / 1.2 / 8.1** |
| circumcoronene | .933 / 120 | **.981 / 4.2 / 12.2** |

→ 20–30× faster mesh fit and equal-or-better df correlation for the selected
`A:s3p2+B:s1` basis. Caveat: atoms-only `A:s3p3` degrades in the mid-height
dip (min .74–.92 vs .93–.97 CG) — its residual is sharper right at the mask
edge, where fill-vs-weight-zero biases shared boundary coefficients. The
`A:s3p3`+CG combination remains available (`--mesh-fit cg`).

Remaining scaling limit: circumcircumcoronene (120 atoms) still fails in
`run_fields` — the dense reference field (784×768×336 float4 ≈ 3.2 GB) plus
resident buffers exceeds the 3090's allocatable memory; also running several
molecules in one process accumulates device images (batch OOM at
`fdbm_compose_E_to_img`); running each molecule in its own process avoids it.

## 18. Direct Gram solve instead of CGLS (2026-10-09)

ncoef ≤ ~1500 → the normal matrix ÂᵀÂ is only ~17 MB. Assembly on the GPU via
the existing operators: `G[:,j] = Atv(Av e_j)` — one host loop over the active
columns (inactive bond slots skipped, they are exactly 0), ~4 kernel calls per
column (~1.5 ms) → **0.1–1.1 s** assembly even for circumcoronene; Jacobi
column scaling (the sq=1 colnorm pass = diag G), Tikhonov λ on the scaled
unit-diagonal Gram, host Cholesky (LAPACK <0.1 s). Implemented as
`ContactSurfaceCL.fit_core_sp_gram` (`--core-solver gpugram`); CPU reference
`CoreBasisStudy.fit_core_gram` (`--core-solver gram`, chunked `AᵀA` — fine for
small molecules, ~230 s for the 72-atom flake because `design()` evaluates
all centers at every sample).

**Critical finding — ridge is required, and it explains the CGLS "stall":**
the exact minimizer (λ→0) has WORSE df than the early-stopped CGLS
(azaindol dip .853 vs .984). CGLS stopping at the float32 residual floor
≈ Tikhonov with λ ~ 1e-6 (scaled space). Ridge sweep on azaindol (min df):
1e-8 → .853, **1e-6 → .984**, 1e-4 → .970, 1e-2 → .886, 1e-1 → .813.
So the "convergence problem" was never a problem — it was implicit
regularization. Gram solve with `--core-ridge 1e-6` reproduces it exactly,
deterministically, in one shot.

Results, `A:s3p2+B:s1` + inpaint mesh (df min / core-fit s / total-fit s):

| molecule | CGLS | gpugram λ=1e-6 |
|---|---|---|
| azaindol | .984 / 2.6 / 3.5 | .984 / **0.34** / 1.4 |
| PTCDA | .957 / 6.6 / 7.8 | .958 / **0.76** / 2.2 |
| pentacene | .985 / 6.4 / 7.9 | .986 / **0.78** / 2.7 |
| PTCDI | .967 / 6.7 / 8.1 | .966 / **0.80** / 2.4 |
| circumcoronene | .981 / 7.8 / 12.2 | .982 / **1.87** / 7.1 |

→ core fit ~4× faster, flat in size (Gram cost ∝ active columns × operator
pass); equal or better df everywhere. Total fit ≈ field sampling + core +
mesh now ~1.4–7 s. The CGLS path remains (`--core-solver gpu`) for debugging.
