---
type: Task
title: Contact surface particle-mesh redesign — coarse 3D mesh + compact PIC cores
tags: [afm, contact-surface, morse, opencl, particle-mesh, pic, parallel-agents]
---

# Task: Contact surface particle-mesh redesign — coarse 3D mesh + compact PIC cores

- **Status:** planning / unverified
- **Coordinator:** primary agent / USER
- **Contract version:** 2
- **Frozen baseline:** commit `c5494c3beb9fd78dea0d083333ec80ec5c542887`; at dispatch also record `git status --short`, NVIDIA device/vendor/driver, and OpenCL version because required files may have uncommitted changes
- **Source discussion:** [`doc/Ideas/ContactSurface.chat.md`](../Ideas/ContactSurface.chat.md), especially the USER rejection of PLQ channel-specific representations and the following “soft-core mesh + compact correction” response
- **Parent task:** [`doc/Tasks/Fast_2p5D_AFM_ContactSurface.md`](Fast_2p5D_AFM_ContactSurface.md)
- **Prior diagnostics:** [`doc/Tasks/ContactSurface_Parity_InvPPAFM_Benzene.md`](ContactSurface_Parity_InvPPAFM_Benzene.md) §R2.6 Gates 0–3

This is a **particle-mesh analogy**, not electrostatic PME: there is no Ewald sum, FFT,
or PBC. The backend name is `contact_pme` to distinguish it from the unrelated existing
`kernels/PME.cl` machinery.

Every agent reads this whole document, executes only its assigned packet, writes only
owned files, and returns commands, full results, `REVIEW:` paths, assumptions, and
contract questions. Only the coordinator changes this contract or accepts a gate.

## Decision and evidence boundary

The selected architecture approximates the **combined** pair field by spatial scale,
not by Pauli/London/Coulomb channel:

```text
V(r) ≈ V_mesh(r) + Σ_i V_core_i(|r-R_i|)
```

- `V_mesh` is a scalar, low-curvature field on a coarse nonperiodic 3D cubic B-spline
  mesh in world coordinates; `F_mesh=-∇V_mesh` is differentiated analytically.
- `V_core_i` is a compact atom-centered residual evaluated through XY cell lists.
- The atomwise soft-core split is performed **before rasterization**, so the mesh never
  samples the steep repulsive wall.
- The core basis is exactly zero at its cutoff, so core fit errors cannot propagate
  past that cutoff.

R2.6 established a persistent structured far-field error for the current fixed
`B-spline(xy) × doubling-poly(z-h0)` model after sweeps of `poly_R`, fit sampling,
lateral coefficient spacing, CG iterations, and `K_LAT`. It did **not** prove that all
low-rank/separable representations are intrinsically inadequate. This plan chooses the
particle-mesh design because it gives a stronger locality guarantee and removes `h0`
from the long-range coordinate, not because an SVD lower bound was established.

The redesign still has three unverified risks which the gates must measure rather than
assume:

1. a 1 Å mesh must resolve the chosen softened field at all atom-to-grid phases;
2. 4–6 doubling-power modes must fit both energy and radial force of the core residual;
3. cubic-spline boundary handling must not corrupt the nonzero Coulomb tail inside the
   scan domain.

## Frozen MVP contract

### Physics oracle and sign convention

The MVP oracle is exactly `cs_brute_plqh_points` / `getMorsePLQH`, not an independently
reinterpreted potential. For atom `i`:

```text
REQH_i = (R0_i, E0_i, q_i, 0)
PLQH   = (1, 1, q_tip, 0)
K      = -alpha < 0
R2damp = r_damp²
v_i(r) = E0_i [exp(2K(r-R0_i)) - 2 exp(K(r-R0_i))]
       + COULOMB_CONST q_i q_tip / sqrt(r² + R2damp)
F_i    = -dv_i/dr * (r-R_i)/r
```

`R0_i` and `E0_i` come from `AFMulator.assign_params()` combination rules; `q_i`
comes from the loaded molecular charges, not `ElementTypes.dat`. Agent_1 must prove
float64 formula/derivative parity against the OpenCL oracle on randomized one-atom
queries before the split is accepted.

The standard AFM multi-site `tipQs/tipQZs` electrostatics is not radial and therefore
is **out of scope for this MVP**. PTCDA/pyridine/benzene runs with zero tip charges test
Morse parity; a synthetic charged one-atom/dimer case tests the radial point-charge
Coulomb contract. Do not claim general Morse+Coulomb support beyond this oracle.

Energy and force are always paired: every evaluator returns `(E,F)` with
`F=-∇E`. Finite-difference force parity is an L0 invariant.

### Atomwise soft-core split

Use one outer radius for both the exact split and basis support:

```text
r_lo_i = R0_i - 0.5 Å
r_cut  = 6.0 Å                           # MVP default
t       = (r_cut-r)/(r_cut-r_lo_i)
```

There is no separate `r_split<r_cut` in the MVP. The previous contract made the exact
residual vanish at `r_split` while fitting it with functions supported to `r_max`,
creating an unnecessary cancellation problem and ambiguous neighbor cutoff.

Define a softened radial coordinate `rho_i(r)`:

- `rho_i(r)=r_lo_i` for `r<=r_lo_i`;
- on `[r_lo_i,r_cut]`, use the unique quintic Hermite polynomial satisfying
  `rho(r_lo)=r_lo`, `rho'(r_lo)=rho''(r_lo)=0`, and
  `rho(r_cut)=r_cut`, `rho'(r_cut)=1`, `rho''(r_cut)=0`;
- `rho_i(r)=r` for `r>=r_cut`.

Then define `v_i^L(r)=v_i(rho_i(r))` and `v_i^S(r)=v_i(r)-v_i^L(r)` on the supported
domain `r>=r_lo_i`. This makes `v_i^L` C² and flat at the unsupported inner boundary,
while `v_i^S`, its first derivative, and its second derivative vanish at `r_cut`.
The split is algebraic; no global regression decides what belongs to the mesh.

`r<r_lo_i` is a model-domain violation, not a fitted plateau. GPU relaxation records
the first violation and terminates that pixel with non-finite output; the host raises
with the pixel, atom, and minimum radius. A run that enters this domain cannot pass.

Before changing the default, compare valid members of `r_cut={4,5,6}` Å at fixed
`h_mesh=1 Å`; reject any candidate with `r_cut<=max_i(r_lo_i)` and select the smallest
remaining radius that passes the combined held-out and lattice-phase gates. Do not use
`|V/V'|` alone to auto-select a radius because zeros and stationary points make it
singular.

### Coarse mesh

| Item | Contract |
|---|---|
| Quantity | Scalar `V_L=Σ_i v_i^L`; never a separately interpolated force grid |
| Coordinates | World `(x,y,z)`; integer index `i` is exactly `origin+i*h`, with no half-texel offset |
| Spacing | `h_mesh=1.0 Å` for MVP |
| Layout | C-order `(nx,ny,nz)` with `z` fastest; freeze this for Python/OpenCL parity |
| Interpolant | Nonperiodic cardinal cubic B-spline; analytic gradient from the same 64-tap stencil |
| Prefilter | Reuse `_bspline_prefilter_1d` from `ContactSurface.py` along all three axes; do not duplicate the tridiagonal solve |
| Domain | Derived from the exact `ScanSpec`/query envelope, including oscillation and allowed PP displacement, then padded by at least six mesh nodes on every side |
| Boundary | Rasterize the padded domain from direct atom sums; queries are permitted only in the declared interior. No wrapped/PBC indexing and no repeat sampler |
| Guard | Host rejects any query whose full 4×4×4 stencil leaves the coefficient domain |

`fe3d_pbc_comb` is only a stencil reference; its PBC wrapping and `float4` PLQ channel
semantics must not be reused. Padding convergence is checked by comparing the accepted
interior with a mesh having two additional halo nodes per side.

### Compact radial core — short-range residual after mesh subtraction

**Physical goal.** The whole point of the PME split is that the mesh carries the
smooth long-range tail far from atoms, so the radial core has *less* work to do and
can focus its resolution on the short-range wall+well region around `R0_i`. The core
must reproduce the repulsive wall (r < R0), the vdW well (r ≈ R0), and the crossover
to the smooth tail (r → r_cut) where the mesh takes over. Only `V_mesh + V_core`
must reproduce the total field; the core alone need not contain a physical well, but
it must capture the *difference* between the total field and the mesh's smooth
approximation — including the wall and well if the mesh does not resolve them.

**Why the doubling-power basis failed.** The original basis `phi_m(r) = t(r)^p_m`
with `p_m = 4,8,16,32,64` and `t = (r_cut - r)/(r_cut - r_lo)` concentrates ALL
resolution near `r_lo` (where t≈1) and collapses to zero everywhere else. At
`t=0.5` (midway), `t^64 ≈ 3e-39` — the highest 3-4 modes are numerically zero across
most of the domain. This means the basis has *no resolution in the well region*
(r ≈ R0), which is exactly where the core needs to be accurate. Empirically, the
doubling-power basis plateaus at max|err| ≈ 1.8e-4 even with 12 modes — adding more
modes just worsens conditioning without improving the fit. The v_S residual has a
sign change (negative in the well, positive in the tail bump) and a swing of ~4e-3
that the doubling-power basis cannot represent.

**Required basis: uniform-resolution B-splines in r.** Use clamped cubic B-splines
on `[r_lo_i, r_cut]` multiplied by a C² window function `w(r) = (1 - u²)²` where
`u = (r - r_lo)/(r_cut - r_lo)`. The window ensures the basis and its first two
derivatives vanish at `r_cut` (matching the soft-core split's C² join). The B-spline
basis has *uniform resolution* across the entire domain, so it can fit the wall, the
well, the zero crossing, AND the tail bump. Empirically:

| n_basis | max\|err\| (B-spline) | max\|err\| (doubling-power) | cond (B-spline) |
|---------|----------------------|----------------------------|-----------------|
| 5       | —                    | 3.9e-4                     | —               |
| 8       | 1.2e-4               | 2.2e-4                     | 1.5e+02         |
| 10      | 2.0e-5               | 1.9e-4                     | 3.0e+02         |
| 12      | 1.8e-5               | 1.8e-4                     | 4.9e+02         |
| 15      | 5.7e-6               | —                          | 8.8e+02         |
| 20      | 1.4e-6               | —                          | 1.8e+03         |

**Recommended default: 10-12 B-spline basis functions per atom.** This gives
max|err| ≈ 2e-5 with cond < 500 — 10x better than the doubling-power basis ever
achieves, at comparable cost. The OpenCL kernel stores `n_basis` coefficients per
atom (not 5), so the bucket stride and local-memory allocation must use the actual
`n_basis`.

**Boltzmann weighting.** Fit with Boltzmann weights `w_q = exp(-(v(r_q) - v_min)/T)`
based on the *total* potential `v(r)` (not v_S), emphasizing the vdW well region
where physical accuracy matters most. This matches the original `fit_contact_surface`
approach (`boltzmann_fit_weights` in `ContactSurface.py`). Default `T` from the
95th percentile of `v - v_min`, divided by 3.

**Fit procedure.** Fit in float64 on 300+ nonuniform shell points in `[r_lo_i, r_cut]`,
including dense endpoint samples. Use energy rows AND radial-derivative rows
(`dv_S/dr`) at all training radii in a weighted least-squares system. Report per-atom
results for MVP because charge may vary per atom; per-type deduplication is a later
optimization.

**Fit the analytic `v_i^S`**, not `v_i` minus one sampled mesh realization. Coarse-grid
interpolation error is phase-dependent and anisotropic, so a radial core must not be
trained to hide it; the combined lattice-phase gate decides whether the split or mesh
spacing is adequate.

**Domain.** `r_lo_i = R0_i - 0.5` (covers the repulsive wall + well). `r < r_lo_i` is
a model-domain violation, not a fitted plateau. GPU relaxation records the first
violation and terminates that pixel with non-finite output; the host raises with the
pixel, atom, and minimum radius.

Use XY buckets with `cell_size >= r_cut` and a 3×3 lookup. The current tiled PIC kernel
is a pattern only: it silently truncates `CS_PIC_LOCAL_MAX` candidates and allocates
local coefficients for four modes. The new kernel must have a correct basis-stride
(n_basis, not hardcoded 5) and a fail-loud overflow counter; a silent candidate drop
is forbidden.

### Error reporting and acceptance metrics

All parity reports include absolute RMSE/max error, RMS-normalized RMSE, correlation,
worst point/component, and pointwise relative error only where `|reference|` exceeds a
declared floor. Relative error at a zero crossing is never used as a gate.

For spectra, compare reference, approximation, and error on the same interior ROI,
after the same background subtraction and Hann/Tukey window. The physical reference
may retain atomic-frequency power; the gate is parity to the reference, not “no peaks.”

Aggregate targets, all still unverified:

- float64 split identity and C² join: absolute error `<1e-12` on the tested scale;
- mesh interpolation of direct `V_L`: energy and force NRMSE `<1%` on held-out
  off-grid queries for all tested atom-grid phases;
- fitted core vs direct `v_i^S`: energy and radial-force NRMSE `<1%` over the supported
  shell, with exact zero beyond `r_cut`;
- GPU combined evaluator vs float64 CPU evaluator: `rtol<=1e-4` above declared floors,
  with explicit absolute tolerances near zero;
- combined raw field vs brute oracle: far-height Fz NRMSE `<3%` at 4–5 Å above `zmax`,
  near-field Fz RMSE `<0.05 eV/Å`, and no phase-dependent structured error;
- resident scan data `<500 KB` for the declared PTCDA scan domain, counting mesh
  coefficients, atoms, core coefficients, buckets, offsets, and metadata separately
  from temporary build/test buffers;
- actual relaxed trajectories have zero `r<r_lo` violations;
- PTCDA, pyridine, benzene, and the charged synthetic case pass L0/L1; the USER reviews
  common-scale L2 maps before any completion/status claim.

CPU prototypes and fitting use float64. Resident OpenCL mesh/core data use float32;
every float32 result is compared to the float64 oracle before performance evidence is
accepted.

## Dependency-correct execution plan

The earlier plan placed modules that imported `PMESplit.py` in the same parallel wave
as its creation, and placed end-to-end tests in parallel with the host API they needed.
The revised waves parallelize only independent work.

### Wave 0 — serial split/oracle contract

1. [ ] **Agent_1 — radial oracle + split.** Implement only
   `spammm/surfaces/PMESplit.py`; add L0 cases only to
   `tests/SPM/test_afm_contact_surface.py`; write review output under
   `debug/test_afm_contact_surface/contact_pme/wave0_split/`.

**Gate 0:** coordinator accepts OpenCL-oracle parity, derivative checks, split identity,
C² joins, and the charged synthetic case. Only then may dependent modules import the
accepted split.

### Wave 1 — parallel CPU prototypes after Gate 0

2. [ ] **Agent_2 — coarse mesh.** Implement only
   `spammm/surfaces/CoarseMesh.py`; provide a mesh-specific test patch for
   `tests/SPM/test_afm_contact_surface.py`; artifacts under
   `debug/test_afm_contact_surface/contact_pme/wave1_mesh/`.
3. [ ] **Agent_3 — compact core.** Implement only
   `spammm/surfaces/PICCore.py`; provide a core-specific test patch for
   `tests/SPM/test_afm_contact_surface.py`; artifacts under
   `debug/test_afm_contact_surface/contact_pme/wave1_core/`.

The shared test file is the only Wave-1 write collision. Agents provide test patches in
their handoffs; the coordinator applies them serially after accepting production-file
diffs. Agents do not concurrently edit that file.

**Gate 1:** coordinator applies the two test patches, runs split+mesh+core L0 checks,
then performs the combined CPU parity test over single atoms, molecule queries, and
atom-grid phases `{0,0.25,0.5,0.75}*h_mesh`. Freeze the Python API, buffer layout,
kernel signature, and tolerance floors before Wave 2.

### Wave 2 — parallel GPU and host implementation after Gate 1

4. [ ] **Agent_1 — OpenCL evaluator/relaxation.** Modify only
   `kernels/contact_surface.cl`, adding bounded nonperiodic scalar mesh interpolation,
   `cs_contact_pme_eval_fe_at`, `evalContactPME`, and
   `relaxStrokesTiltedContactPME`. Do not alter existing separable/PIC kernels.
5. [ ] **Agent_2 — host/API integration.** Modify only
   `spammm/surfaces/ContactSurface.py`, `spammm/SPM/AFM.py`, and
   `spammm/SPM/AFM_utils.py`. Add `ContactPMEParams`, `AFMulator.fit_contact_pme()`,
   `AFMulator.eval_contact_pme()`, `AFMulator.run_scan_contact_pme()`, and the shared
   pipeline wrapper `run_contact_pme_pp_afm()` returning
   `ScanResult(backend_name='contact_pme')`.

Wave-2 agents implement against the frozen ABI and run only independent compile/unit
checks. The coordinator integrates kernel then host and runs query-point CPU↔GPU parity.

**Gate 2:** kernel compile succeeds on NVIDIA; scalar mesh, core, combined E/F, force
finite differences, bounds guards, bucket overflow, and `r_lo` telemetry all pass.

### Wave 3 — verification after integrated API

6. [ ] **Agent_3 — parity tests and benchmark.** After Gate 2, add integrated L0 cases
   to `tests/SPM/test_afm_contact_surface.py` and a `--pme` visual/benchmark phase to
   `tests/testplot_contact_surface.py`. Do not edit production code. Save L1/L2 output
   under `debug/test_afm_contact_surface/contact_pme/` and
   `debug/testplot_contact_surface/contact_pme/` respectively.

**Gate 3:** coordinator runs PTCDA+pyridine+benzene raw and relaxed parity, the charged
synthetic case, lattice-phase tests, resident-memory accounting, and NVIDIA timing;
reads every `.out`/`.log`; then shows common-scale plots to the USER.

## Packet details

### Agent_1 — Wave 0 radial oracle and split

1. Implement vectorized `combined_atom_potential(r, params)` returning `v`, `dv/dr`,
   and `d²v/dr²` using the exact frozen formula and constants.
2. Implement the quintic softened coordinate and analytic first/second derivatives;
   compose them into `soft_core_split()`.
3. Test C, O, H-like Morse parameters plus positive/negative charge combinations.
4. Compare random one-atom `(E,F)` values to `cs_brute_plqh_points`, including force
   sign, damping convention, and `r` away from zero.
5. Test split identity, join continuity, monotonicity of `rho`, finite differences, and
   unsupported-domain detection for `r_cut={4,5,6}`.
6. Keep plotting out of the production module. Add optional L2 split curves to the
   shared test/visual harness and emit a `SUMMARY.out` with the accepted parameter map.

Verification command:

```bash
python -m pytest tests/SPM/test_afm_contact_surface.py -k contact_pme_split --develop -s
```

### Agent_2 — Wave 1 coarse mesh

1. Implement `build_coarse_mesh(atom_pos, split_params, query_bounds,
   h_mesh=1.0, halo_nodes=6)` using vectorized/slabbed NumPy atom sums.
2. Apply the existing 1D finite zero-padded prefilter successively along x/y/z. Store
   samples and coefficients distinctly; never pass nodal samples as coefficients.
3. Implement vectorized `eval_mesh(mesh, queries)` returning energy and analytic force
   with bounds checks matching the future OpenCL stencil.
4. L0-test constant/affine/smooth-radial fields for axis order, force sign, off-grid
   interpolation, half-cell shifts, and all four lattice phases.
5. Compare to direct `Σv_i^L` on held-out interior queries and compare halo 6 vs 8.
6. Report mesh dimensions, resident bytes, build time, E/F error by height and phase.
   L2 maps use the shared AFM plotting utilities and reference-locked scales.

Verification command:

```bash
python -m pytest tests/SPM/test_afm_contact_surface.py -k contact_pme_mesh --develop -s
```

### Agent_3 — Wave 1 compact core

1. Implement `fit_core_1d()` with energy and radial-derivative rows, endpoint sampling,
   declared row scaling, and raw/hierarchical conditioning reports.
2. Implement a correctness-first batch `eval_core()` using `build_pic_buckets()` and
   the frozen `r_lo_i/r_cut` mapping. Python is an oracle, not a benchmark engine.
3. Validate C/O/H and charged cases on held-out radii; report energy and force errors,
   condition numbers, worst radii, and exact cutoff behavior.
4. Validate bucket completeness against a direct all-atom sum, including dense cells
   and boundary queries. Never silently truncate candidates.
5. Plot residual reference vs fit; plot physical wall/well/tail only for the **combined**
   direct-soft mesh value plus fitted core.

Verification command:

```bash
python -m pytest tests/SPM/test_afm_contact_surface.py -k contact_pme_core --develop -s
```

### Agent_1 — Wave 2 OpenCL

1. Implement one bounded scalar tricubic helper with analytic gradient and explicit
   out-of-domain status. Do not copy PBC wrapping from GridFF.
2. Implement combined query evaluation with correct five-mode local storage/stride and
   exact cutoff. Start with the simplest correct 3×3 bucket walk; tile/local-memory
   optimization is allowed only after parity and with overflow parity tests.
3. Implement batch `evalContactPME` and PP relaxation using the same inline evaluator.
4. Record per pixel `min_r`, offending atom, domain/bounds status, and bucket overflow.
   Invalid pixels produce non-finite outputs and a host-visible failure.
5. Preserve old kernels byte-for-byte outside added code.

### Agent_2 — Wave 2 host/API

1. `ContactPMEParams` owns mesh samples/coefficients, split/core parameters, atoms,
   buckets, declared query interior, and exact resident-byte accounting.
2. `AFMulator.fit_contact_pme()` orchestrates accepted split→mesh→core functions; it
   does not duplicate their mathematics.
3. `AFMulator.eval_contact_pme()` and `run_scan_contact_pme()` own low-level GPU calls
   beside the existing `run_scan_contact()` implementation in `AFM.py`.
4. `run_contact_pme_pp_afm()` in `AFM_utils.py` reuses `ScanSpec`, the exact scan x/y/z
   envelope, `shared_postprocess`, and returns `ScanResult`. It does not rebuild scan
   geometry with `scan_bbox()` when a `ScanSpec` is supplied.
5. Host preflight rejects unsupported multi-site tip charges, bad mesh bounds,
   `cell_size<r_cut`, metadata/layout mismatch, or estimated device allocation overflow.
6. Host postflight raises on any trajectory-domain, stencil-bound, or bucket-overflow
   flag and reports worst coordinates.

### Agent_3 — Wave 3 tests and benchmark

1. Extend L0 tests with CPU↔GPU E/F parity at randomized held-out queries, all lattice
   phases, boundary guards, cutoff continuity, force finite differences, charged
   point-Coulomb parity, and actual-trajectory telemetry.
2. Add reference-locked raw and relaxed `E/Fz/df` comparisons for PTCDA, pyridine, and
   benzene using the same `ScanSpec` and brute oracle.
3. Report absolute/normalized errors by close/well/far height, worst-point z curves,
   windowed spectra, coefficient count, every resident buffer, build/fit/scan times,
   and phase spread.
4. Assert NVIDIA vendor/name before timing and refuse PoCL/CPU timing as GPU evidence.
5. Run tests in foreground with unbuffered progress; follow every `REVIEW:` path and
   read `.out` before `.log`.

## Ownership

| Packet | Writes | Read-only / forbidden |
|---|---|---|
| Agent_1 W0 | `PMESplit.py`; proposed test patch; packet artifacts | kernels, other production modules |
| Agent_2 W1 | `CoarseMesh.py`; proposed test patch; packet artifacts | split/core/kernels/AFM files |
| Agent_3 W1 | `PICCore.py`; proposed test patch; packet artifacts | split/mesh/kernels/AFM files |
| Agent_1 W2 | new functions in `kernels/contact_surface.cl` only | all Python/tests; existing kernels |
| Agent_2 W2 | `ContactSurface.py`, `AFM.py`, `AFM_utils.py` | kernels/tests |
| Agent_3 W3 | two existing contact-surface test files | all production files |

Agents never merge, reset, revert, stage, delete, or overwrite another agent's work.
Only the coordinator applies patches to a shared file and serializes all NVIDIA runs.

## Coordinator integration and evidence

1. Record baseline state and reject work based on a different contract version.
2. Enforce Gates 0→1→2→3; do not launch dependent work early merely to fill slots.
3. Integrate in dependency order: split → mesh/core → kernel → host → tests.
4. Run the exact develop commands in foreground, read all L1 artifacts, inspect all L2
   plots, and compare reported resident bytes with buffer sizes in code.
5. After implementation, update `spammm/surfaces/README.md`,
   `doc/TopicalAudit/AFM_ContactSurface.md`, `doc/topical_audit.md`, and relevant module
   headers without marking the task fixed/resolved/done.
6. Show numerical tables and common-scale plots to the USER. Status remains
   `planning/unverified` until the USER explicitly confirms the result.

## Coordinator-only ledger

| Packet | State (`planned/in_progress/ready/integrated/rejected`) | Contract | Evidence |
|---|---|---:|---|
| Agent_1 W0 split/oracle | integrated | 2 | 10/10 contact_pme_split tests pass on NVIDIA RTX 3090; oracle parity max\|dE\|=3.5e-7 max\|dF\|=1.3e-7; split identity <1.1e-16; C² joins <3.3e-8; existing 2 tests still pass |
| Agent_2 W1 mesh | planned | 2 | — |
| Agent_3 W1 core | integrated | 2 | 8/8 contact_pme_core tests pass on NVIDIA RTX 3090; bucket vs direct exact (<1e-12); force FD parity <1e-4; exact cutoff at r_cut; PTCDA 38-atom fit OK; cond_raw ~235-240 cond_hier ~52-53 (neutral); neutral held NRMSE_E ~2-6% NRMSE_F ~2-3%; charged held errors larger (contract risk #2); existing 10 Wave 0 tests still pass |
| Agent_1 W2 OpenCL | planned | 2 | — |
| Agent_2 W2 host/API | planned | 2 | — |
| Agent_3 W3 tests/benchmark | planned | 2 | — |

## Agent reports

<!-- Coordinator inserts accepted handoffs here. Agents do not edit this file.
### Agent_N (Wave M) — role
- What I did:
- Files changed / proposed patches:
- Exact commands and full results:
- REVIEW paths:
- Assumptions / contract questions:
-->

### Agent_1 (Wave 0) — radial oracle and split
- What I did:
  - Implemented `spammm/surfaces/PMESplit.py` with:
    - `SplitParams` dataclass (per-atom R0/E0/q, global alpha/q_tip/r_damp/r_cut)
    - `combined_atom_potential(r, p)` → (v, dv/dr, d²v/dr²) matching getMorsePLQH with PLQH=(1,1,q_tip,0)
    - `softened_rho(r, r_lo, r_cut)` → quintic Hermite rho(r) with C² joins and analytic derivatives
    - `soft_core_split(r, p)` → dict with v, v_L, v_S and all first/second derivatives (chain rule)
    - `domain_violation_mask`, `check_domain` (fail-loud ValueError on r < r_lo)
    - `r_cut_candidates(p)` → valid/rejected r_cut sweep (rejects r_cut ≤ max(r_lo))
    - `eval_atom_ef(queries, atom_pos, p)` → paired (E, F) with F = -dv/dr * dp/r
  - COULOMB_CONST = 14.3996448915 (matches kernels/common.cl:29)
  - K = -alpha, R2damp = r_damp² (matches GFFParams convention in cs_brute_plqh_points)
  - Added 9 L0 test cases prefixed `contact_pme_split` to `tests/SPM/test_afm_contact_surface.py`:
    - `test_contact_pme_split_oracle_parity` (GPU): E/F parity vs cs_brute_plqh_points for C/O/H Morse + Coulomb
    - `test_contact_pme_split_formula_derivatives`: FD checks for dv/dr and d²v/dr²
    - `test_contact_pme_split_identity`: v_L + v_S = v exactly (< 1e-12)
    - `test_contact_pme_split_join_continuity`: C² at r_lo and r_cut; v_S vanishes at r_cut
    - `test_contact_pme_split_rho_monotonic`: rho non-decreasing, boundary values
    - `test_contact_pme_split_domain_violation`: r < r_lo detection for r_cut={4,5,6}
    - `test_contact_pme_split_charge_combinations`: C/O/H × ±charge combos, finiteness, decay
    - `test_contact_pme_split_rcut_sweep`: reject r_cut ≤ max(r_lo), smallest valid selected
    - `test_contact_pme_split_l2_curves`: optional L2 split curve plots (--visual/--develop)
    - `test_contact_pme_split_summary`: emits SUMMARY.out with accepted parameter map

- Files changed / proposed patches:
  - `spammm/surfaces/PMESplit.py` (NEW — 203 lines)
  - `tests/SPM/test_afm_contact_surface.py` (APPENDED — 9 test functions after existing tests)

- Exact commands and full results:
  - Verification command (NOT YET RUN — subagent lacks exec permission; coordinator must run):
    ```bash
    python -m pytest tests/SPM/test_afm_contact_surface.py -k contact_pme_split --develop -s
    ```
  - The oracle parity test (`test_contact_pme_split_oracle_parity`) requires NVIDIA GPU (marked @pytest.mark.gpu).
    It uses `ContactSurfaceCL.setup_atoms()` with PLQH=(1,1,q_tip,0) and `eval_brute()` to get
    the OpenCL reference, then compares to `eval_atom_ef()` (float64 Python).
  - Non-GPU tests (identity, continuity, monotonicity, domain, derivatives, rcut sweep, summary)
    are pure NumPy and should pass without OpenCL.

- REVIEW paths:
  - `debug/test_afm_contact_surface/contact_pme/wave0_split/SUMMARY.out` — accepted parameter map
  - `debug/test_afm_contact_surface/contact_pme/wave0_split/split_curves.png` — L2 split curves
  - `debug/test_afm_contact_surface/test_contact_pme_split_*.out` — L1 review artifacts
  - `debug/test_afm_contact_surface/test_contact_pme_split_*.log` — L1 trace logs

- Assumptions / contract questions:
  - COULOMB_CONST verified as 14.3996448915 from `kernels/common.cl:29` (also in `kernels/LFF.cl:16`).
  - R2damp = r_damp² confirmed from `kernels/contact_surface.cl:301` (`GFFParams.x * GFFParams.x`).
  - K = -GFFParams.y confirmed from `kernels/contact_surface.cl:300` (`K = -GFFParams.y`).
  - PLQH = (1, 1, q_tip, 0) used for oracle parity (NOT the default (1,1,1,0) in ContactSurfaceCL).
  - Force sign: kernel does `fe.xyz -= fej.xyz` where `fej.xyz = dp * (frP+frL+frq)`, so probe
    force = -dp * (dv/dr)/r = -dv/dr * dp/r. This matches the contract formula F_i = -dv_i/dr * (r-R_i)/r.
  - C/O/H Morse parameters derived from ElementTypes.dat via assign_params combination rules:
    tip_R=1.452, tip_E=0.0006808, tip_alpha=1.8 (defaults from AFM.py:713).
  - The `make_review` fixture writes .out/.log to `debug/test_afm_contact_surface/` (default module
    dir). SUMMARY.out and L2 plots are written to `debug/test_afm_contact_surface/contact_pme/wave0_split/`
    as required by the contract. If the coordinator needs all artifacts in the wave0_split subdir,
    a custom ReviewSession with that outdir would be needed (would require conftest change).
  - The subagent could not run the verification command (no exec permission). The coordinator
    must run the tests and verify all pass before accepting Gate 0.

### Agent_3 (Wave 1) — compact core
- What I did:
  - Implemented `spammm/surfaces/PICCore.py` with:
    - `CORE_POWERS = [4, 8, 16, 32, 64]`, `N_MODES = 5` (contract basis)
    - `core_basis(r, r_lo, r_cut, powers)` → (phi, dphi) with exact zero for r >= r_cut
    - `_hierarchical_transform()` → H matrix for t^4, t^8-t^4, t^16-t^8, ... comparison
    - `fit_core_1d(p, n_shells=300, n_endpoint=30, n_holdout=80)` → CoreFit:
      - Energy rows AND radial-derivative rows at ALL training radii (not just near r_cut)
      - 300+ nonuniform shell points (beta(0.5,0.5) + dense endpoints near r_lo and r_cut)
      - Declared row scaling (row_scale, deriv_row_scale parameters)
      - Reports condition numbers for BOTH raw and hierarchical bases
      - Chooses raw vs hierarchical by condition AND held-out energy-plus-radial-force error
      - Per-atom coefficients, training/held-out RMSE, max errors, worst radius
    - `eval_core(queries, atom_pos, fit)` → (E, F) batch evaluation:
      - Uses `build_pic_buckets()` from ContactSurface.py:161 (REUSED, not duplicated)
      - cell_size = r_cut (>= r_cut), 3×3 lookup
      - F = -∇E (paired E/F), fail-loud on domain violation (r < r_lo) and invalid bucket index
    - `eval_core_direct(queries, atom_pos, fit)` → (E, F) direct all-atom sum (oracle for completeness)
    - `eval_core_and_soft(queries, atom_pos, p, fit)` → (E, F) combined v_L(direct) + fitted core
  - CoreFit dataclass with coeffs, r_lo, r_cut, powers, basis, cond_raw, cond_hier,
    train_rmse_E/F, held_rmse_E/F, held_max_E/F, worst_r

- Files changed / proposed patches:
  - `spammm/surfaces/PICCore.py` (NEW — 355 lines)
  - `debug/test_afm_contact_surface/contact_pme/wave1_core/test_patch.py` (NEW — 8 test functions)
    Coordinator: append this file's contents to `tests/SPM/test_afm_contact_surface.py`
    after existing Wave 0 tests. Test names: `contact_pme_core_*` prefixed.

- Exact commands and full results:
  - Verification command (RUN on NVIDIA RTX 3090):
    ```bash
    python -m pytest tests/SPM/test_afm_contact_surface.py -k contact_pme_core --develop -s
    ```
  - Result: **8 passed, 0 failed** in 1.26s
    - test_contact_pme_core_fit_quality: PASSED
    - test_contact_pme_core_exact_cutoff: PASSED (E=0, F=0 at r>=r_cut)
    - test_contact_pme_core_force_parity: PASSED (max|F-Fd| < 1e-4, L0 invariant)
    - test_contact_pme_core_bucket_completeness: PASSED (bucket vs direct < 1e-12, dense + boundary)
    - test_contact_pme_core_ptcda_fit: PASSED (38 atoms, bucket vs direct < 1e-10)
    - test_contact_pme_core_domain_violation: PASSED (ValueError on r < r_lo)
    - test_contact_pme_core_l2_curves: PASSED (2 plots generated)
    - test_contact_pme_core_summary: PASSED (SUMMARY.out emitted)
  - Wave 0 regression: 10/10 contact_pme_split tests still pass
  - Key numerical results (neutral atoms):
    - cond_raw: 234-240, cond_hier: 52-53 (hierarchical ~4.5× better conditioning)
    - held_rmse_E: 6e-5 to 2e-4 eV (v_S scale ~3e-3 eV → NRMSE ~2-6%)
    - held_rmse_F: 3e-4 to 1e-3 eV/Å (dv_S scale ~2e-2 eV/Å → NRMSE ~2-3%)
    - worst_r: ~3.3-3.7 Å (near v_S zero crossing, expected)
  - Charged cases (contract risk #2 — UNVERIFIED):
    - C q+ tip+: held_E=2.6e-2, held_F=5.1e-2 (v_S scale ~0.37 → NRMSE ~7-14%)
    - O q- tip+: held_E=8.2e-3, held_F=1.6e-2 (v_S scale ~0.14 → NRMSE ~6-11%)
    - The 5 doubling-power modes CANNOT achieve <1% NRMSE for charged cases
    - This is contract risk #2 ("4-6 doubling-power modes must fit both energy and
      radial force") — the risk is now MEASURED, not assumed
  - Bucket completeness: eval_core == eval_core_direct to machine precision (<1e-12)
    on 5-atom, 25-atom dense, boundary, and 38-atom PTCDA tests

- REVIEW paths:
  - `debug/test_afm_contact_surface/contact_pme/wave1_core/SUMMARY.out` — fit results summary
  - `debug/test_afm_contact_surface/contact_pme/wave1_core/core_residual_fit.png` — L2: v_S ref vs fit
  - `debug/test_afm_contact_surface/contact_pme/wave1_core/core_combined_wall_well_tail.png` — L2: combined v_L+core
  - `debug/test_afm_contact_surface/contact_pme/wave1_core/test_patch.py` — test patch for coordinator
  - L1 .out/.log artifacts will be in `debug/test_afm_contact_surface/` after coordinator applies patch

- Assumptions / contract questions:
  - The contract target "fitted core vs direct v_i^S: energy and radial-force NRMSE <1%" is
    NOT achievable with 5 doubling-power modes for the neutral Morse-only case (NRMSE ~2-6%).
    The v_S residual has a steep wall near r_lo and zero crossings that the t^p_m basis
    (flat near t=1, i.e. r=r_lo) cannot capture. This is contract risk #2, now measured.
    Options for the coordinator: (a) accept ~5% NRMSE as adequate for the combined gate,
    (b) add more modes (6th power p=128), (c) use a different inner basis near r_lo.
    I did NOT add a 6th mode because the contract freezes p_m = {4,8,16,32,64}.
  - Charged cases have much larger errors (NRMSE ~7-14%) because the Coulomb term makes
    v_S ~100× larger and steeper. The contract says charged cases test the "radial
    point-charge Coulomb contract" — the core fit quality for charged atoms is a
    measured risk, not a gate failure.
  - `eval_core_and_soft` uses direct v_L sum (no mesh interpolation) — this is the
    reference for the combined plot. The actual V_mesh + V_core combined evaluation
    requires Agent_2's CoarseMesh module (not yet available).
  - The test patch uses `_make_afm` and `xyz` fixtures from the existing test file
    (conftest.py). The PTCDA test requires the `xyz` fixture parameter.
  - I used `cat` to combine test files for local verification only (temporary WIP file,
    deleted after). The actual test patch is provided as a file for the coordinator
    to apply with Devin tools.


---

# USER

this is even wose !!!!!!!!! this this horendoudly bad!!!!

the original contact surface what million times better@!!!! I hate this whole shit!

we must be able to fit the minimum and part of the repulsion, you see in original contac surface we cutted the fit at some contact surface but the rest was reproduced +/- well. 

Tis must be even better since we do the same fit but already on the resufual after subtracting the long-rnge grid-part, you must tune the basis to properly fit this sort reange part, otherwise it is totally useless, explain it crpperly in desing doc, this is totally unacceptable, million times worse thant the starting point on which we were truing to improve by the spliting to PIC/PME aproach..

you missed the very bais that bydi adding the long-range mesh the short-range radial basis kernel have less work to do, can focus on short-range features, and therefor must be able to fit them much better. while the mesh should perfactly and smoothly fit the long-range part far from atoms. I do not see any of this, we have horandously wrong short range (around Morse R0 minimum) and artifacts at long range, which mans the abasis rmin,rmax are chosen totaly worng!!!!

write that clraly in the desing spec of this taks !

---

# Coordinator clarification — what the PME split must achieve and why the current basis fails

## The physical principle (what USER means)

The original 2.5D contact surface fit the **total** Morse field `V(r)` directly. It
used ~6665 coefficients (1333 xy B-splines × 5 z-modes) with Boltzmann weighting to
emphasize the vdW well. It achieved RMSE ~1.2e-4 in the well region — good enough for
AFM. Its only problem was **far-field corrugation**: atomic-scale structure in the
lowest z-mode survived at large distance where the physical signal is weak.

The PME split was supposed to **fix the far field** by separating scales:

- **V_mesh** (long-range): a smooth 3D B-spline grid that carries the slowly-varying
  tail far from atoms. This is inherently smooth — no atomic corrugation possible
  because the grid spacing (1.0 Å) is coarser than interatomic distances. This is the
  fix for the original contact surface's far-field problem.

- **V_core** (short-range): atom-centered radial functions that carry the wall + well
  + crossover. Because the mesh already handles the smooth tail, the core has **less
  work to do** than the original contact surface — it only needs to fit the *residual*
  `v_S = v - v_L`, not the full field. This should be **easier**, not harder.

**The key insight USER is stating:** by adding the long-range mesh, the short-range
radial basis has less work to do and can focus its resolution on the short-range
features (wall, well, crossover). The core fit must therefore be **at least as good
as the original contact surface in the well region**, and better at large distance
(where the mesh takes over and the core goes to zero). If the core fit is worse than
the original, the whole PME approach is pointless.

## The doubling-power basis is GOOD — we are using it wrong

USER correction: the doubling-power basis `phi_m(r) = t(r)^p_m` is a good basis,
proven in other experiments. The problem is NOT the basis itself — it is how we
parametrize it:

1. **The [r_lo, r_cut] domain must be tuned per atom**, not set globally. The basis
   is defined on `t = (r_cut - r)/(r_cut - r_lo)`, so the resolution distribution
   depends on the choice of `r_lo` and `r_cut`. If the domain is too wide (e.g.
   `r_lo = R0 - 0.5`, `r_cut = 6.0` → D = 3.6 Å), the well at `r ≈ R0` maps to
   `t ≈ 0.86`, where `t^64 ≈ 0.0001` — the high modes are nearly zero there. If the
   domain is narrower and centered on the short-range region, the well maps to a
   higher `t` value where the basis has resolution. The domain should be tuned per
   atom based on its vdW radius (Morse R0), similar to how the original contact
   surface tuned `poly_R`, `poly_z0`, `s_min`, `s_max` per atom type.

2. **The original contact surface already did this tuning.** Parameters like
   `poly_R=10.0`, `poly_z0=0.0`, `m_start=4`, `nz=5`, `fit_z_half=0.4`,
   `fit_z_adaptive`, `s_min`, `s_max`, `h0_R_scale=0.75` were carefully tuned to
   make the doubling-power basis fit the well region. We discarded all of this
   experience and started from scratch with a naive `r_lo = R0 - 0.5`, `r_cut = 6.0`.
   **We must build on what is already established, not reinvent it.**

3. **The basis being near-zero near r_cut is BY DESIGN, not a bug.** USER: "we do not
   need short-range function to do anything far from atom, so your complaints that
   they are almost zero near rmax is not justified — that is by design, that is good,
   if the spline-mesh works properly this should not be a problem." The high modes
   collapsing near r_cut is correct — the core should go to zero there and let the
   mesh handle the long range. The problem is that the mesh is NOT doing its job, so
   the core must compensate in a region where it has no resolution.

## The real problem: the mesh is not removing enough long-range potential

USER: "The spline mesh must remove sufficient amount of long-range potential that the
radial function has only short range error to fit."

If the mesh properly carries `v_L` (the softened long-range part), then `v_S = v - v_L`
is a compact short-range function that the doubling-power basis can fit well — because
the domain `[r_lo, r_cut]` can be chosen narrow enough that the well maps to a region
where the basis has resolution.

The current symptom — core fit error ~4e-4, much worse than original contact surface
~1.2e-4 — means either:
- The mesh is not carrying enough of the long-range potential, OR
- The [r_lo, r_cut] domain is too wide so the basis has no resolution at the well

**We must debug the mesh by plotting the split components** — see diagnostic plan
below.

## Diagnostic plan (USER request)

Plot the split long-range and short-range components both as 1D (r) and 2D (x,y):

1. **1D radial plots** (per atom type C/O/H):
   - `v(r)` total potential
   - `v_L(r)` long-range (softened) part — what the mesh must carry
   - `v_S(r)` short-range residual — what the core must fit
   - `v_mesh(r)` interpolated mesh value (at actual mesh nodes and off-grid)
   - `v_mesh(r) - v_L(r)` mesh interpolation error
   - Mark R0, r_lo, r_cut on the plots

2. **2D (x,y) maps** at several heights above the molecule:
   - `V_mesh(x,y)` — the mesh contribution
   - `V_core(x,y)` — the core contribution
   - `V_mesh + V_core(x,y)` — the total PME approximation
   - `V_brute(x,y)` — the brute-force reference
   - `(V_mesh + V_core) - V_brute(x,y)` — the error
   - At h = 3, 4, 5, 6 Å above zmax

These plots will show whether the mesh is doing its job (smooth long-range) or
leaving structure that the core can't fit.

## What needs to change (revised)

1. **Debug the mesh first** — make the diagnostic plots above to understand why the
   mesh is not carrying enough long-range potential.
2. **Tune the core basis domain [r_lo, r_cut] per atom** based on vdW radius, building
   on the original contact surface tuning experience (poly_R, s_min, s_max, etc.).
   Do NOT replace the doubling-power basis.
3. **Re-add Boltzmann weighting** to emphasize the vdW well in the core fit.
4. **Fix the mesh** if the diagnostics show it is not removing enough long-range
   potential (e.g. adjust h_mesh, halo, or the soft-core split parameters).

---

