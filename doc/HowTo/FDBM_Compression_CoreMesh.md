---
type: HowTo
title: Compressed FDBM force field — compact sp cores + coarse B-spline mesh (coremesh)
tags: [contact-pme, fdbm, compression, b-spline, gpu, fitting]
timestamp: 2026-10-09
---

# Compressed FDBM force field (coremesh, `poly8sp`)

Goal: replace the ~1 GB dense FDBM field image by a **compact core + coarse
mesh** representation (~200–500 KB, ~9000×) that evaluates through the
**existing ContactPME kernels** — no new evaluator, PP-AFM relaxation and df
come for free.

```
E_model(x) = Σ_centers Σ_modes c_m φ_m(x − C, w)   +   Σ_nodes m_j B_j(x)
             ── compact core (9 Å support) ──         ── cubic B-spline, h=1 Å ──
```

## 1. The pieces

### Core basis `poly8sp` (`CS_PME_SP_CORE`, `CS_PME_NMODES=3+3·NP`)

Basis functions `(1 − r/R)₊⁸` on **atom centers + bond midpoints**
(spec grammar `A:s3p2+B:s1`):

| center | functions | radii R [Å] | coefs |
|---|---|---|---|
| atom | 3 × s | 9, 5.61, 3.5 | 3 |
| atom | NP × p = φ·(nx,ny,nz) | 7, 3.5 | 6 |
| bond midpoint | 1 × s | 5 | 1 |

Kernel ABI: per-center slots — 0-2 s at `R[0..2]·w`, then p-shells (px,py,pz)
at `R[3+k]·w`; the **w channel is a per-center radius scale** (atoms 1.0,
bonds 5/9 → effective R=5 Å). Inactive bond slots are exactly 0. Same basis
survey rationale: doc `ContactPME_CoreMesh_Fit_Design.md` §14 — s/p on atoms +
one bond-s was the Pareto optimum; atom-only `A:s3p3` (12 slots) also works.

### Why a split at all

The core eats the sharp short-range wall (the thing a mesh cannot hold at
h=1 Å); the mesh only carries the smooth residual — electrostatics, vdW tail,
the part PAW would call V_L. Mesh h=1 Å is therefore sufficient and could
probably be coarser.

## 2. The fit (all on GPU or banded — no dense matrices, no CG in production)

```
dense FDBM field (GPU image, never on host)
   └─ supersampled grid (s=4 × h=1 Å nodes) → E, F  (sample)
        ├─ core fit  : shell 2.5 < r_min < 4.0 Å, z > mol+0.5 Å,
        │             weighted E + wF·F rows → Gram solve (gpugram)
        └─ mesh fit  : residual = E − E_core at nodes → omit too-close
                      cells via harmonic inpaint → separable banded solve
```

### Core fit — `fit_core_sp_gram` (design doc §18)

- `G[:,j] = Atv(Av e_j)` — Gram columns assembled on GPU with the *same*
  operator kernels as evaluation (`cs_sp_Av`/`cs_sp_Atv`), ~4 kernel calls per
  active column → 0.1–1.1 s for up to ~1500 coefficients.
- Jacobi column scaling (`sq=1` pass = diag G) + **Tikhonov ridge λ≈1e-6**
  (scaled space) + host Cholesky.
- **λ is essential**: the exact minimizer (λ→0) is worse than early-stopped
  CGLS (azaindol dip .853 vs .984). CGLS stalling at the float32 floor was
  implicit regularization; the Gram solve makes it explicit via `ridge`.
- Solvers: `--core-solver {gpugram (default-fast), gpu (CGLS), gram, lstsq}`.
  CGLS kept for debugging; CPU `gram`/`lstsq` = references.

### Mesh fit — omit-by-inpaint + banded solve (design doc §17)

The too-repulsive region (`r_min < 2.5 Å`, `z < mol+0.5 Å`, ~41 % of nodes) is
**omitted** — cells get a smooth continuation instead of data:

1. node-grid residual → mask+dilate → `inpaint_residual` (harmonic fill);
2. `fit_mesh_lsq_3d(s=1)` — separable banded solve (tridiagonal per axis);
3. supersampled polish: masked ss cells seeded with the node-fit mesh +
   `inpaint_residual(seed=…)` → `fit_mesh_lsq_3d(s=4)` — still all banded.

Same recipe as the production PAW `cpm_params_from_samples` path.
`--mesh-fit {inpaint (default), cg}`; the weighted scipy CG stays for debug.

## 3. Usage

```bash
# fit + evaluate + PP-AFM scan on one molecule (each in its own process —
# device images accumulate across molecules in one process!)
python3 tests/SPM/testplot_fdbm_fields_compress.py circumcoronene \
    --method coremesh --core-basis poly8sp \
    --core-solver gpugram --core-ridge 1e-6 \
    --skip-spline-only
```

Outputs per molecule in `debug/testplot_fdbm_fields_compress/<mol>_s<step>_coremesh_poly8sp-<spec>1/`:
`SUMMARY.out` (df/Fz correlations, timings, holdout RMSE, CPU↔GPU parity),
`compare_df_Fz.png` (df & Fz panels vs FDBM reference), `coremesh_*.pkl`
(ContactPMEParams archive — what a production run would load), `core_*.npz`.

Flags: `--core-spec 'A:s3p2+B:s1'|'A:s3p3'`, `--core-wF 0.3` (force-row
weight), `--lam 0.001` (mesh curvature penalty), `--h-mesh 1.0`.

## 4. Results (FDBM, DFTB+ backend, CO tip, RTX 3090, h=1 Å, s=4)

`A:s3p2+B:s1` + gpugram(λ=1e-6) + inpaint mesh:

| molecule | atoms | centers | ncoef | core fit | mesh fit | scan | df min/max | archive |
|---|---|---|---|---|---|---|---|---|
| azaindol | 15 | 31 | 279 | 0.34 s | 1.0 s | 0.04 s | .984/.998 | 152 KB |
| pentacene | 36 | 76 | 684 | 0.78 s | 1.8 s | 0.08 s | .986/.998 | 234 KB |
| PTCDA | 38 | 82 | 738 | 0.76 s | 1.4 s | 0.08 s | .958/.998 | 210 KB |
| PTCDI | 40 | 86 | 774 | 0.80 s | 1.5 s | 0.08 s | .966/.999 | 224 KB |
| circumcoronene | 72 | 162 | 1458 | 1.87 s | 5.0 s | 0.21 s | .982/.999 | 493 KB |

Dense reference fields are 0.65–2.2 GB → ~9000× compression; scan ≤0.2 s;
holdout CPU↔GPU parity ≤2e-6.

Accuracy notes: min df correlation sits in the h_df≈4.0–4.2 Å band where the
Fz signal crosses zero (most sensitive height); at all other heights .98–.999.

## 5. Pitfalls / open issues

- **Device memory, not host, is the limit**: circumcircumcoronene (120 atoms)
  fails in `run_fields` — dense field image (784×768×336 float4 ≈ 3.2 GB) +
  resident FFT/AFM buffers > allocatable. Remedy is tiled/streamed field
  generation, not the compact representation.
- **Don't batch molecules in one process** — GPU images accumulate
  (`fdbm_compose_E_to_img` OOM); per-molecule processes are clean.
- **`A:s3p3` (atoms-only) + inpaint mesh degrades** at mid heights (.74–.92);
  its residual is sharper at the mask edge. Use `--mesh-fit cg` for it, or
  stick with `A:s3p2+B:s1`.
- **Zero-norm columns** (p/d modes on bond centers): frozen at 0 via the
  active mask — do not remove the mask, unrestricted bond slots poison the fit.
- Ridge λ≠0 intentionally — see §18; do not "fix" the underconverged-looking
  residual.

## 6. Code map

- `spammm/surfaces/PICCore.py` — `fit_core_sp_shell` (solver dispatch),
  `eval_core_sp` (CPU ref), `sp_ladder`, `inpaint_residual`, `min_dist_to_atoms`
- `spammm/surfaces/ContactSurface.py` — `ContactSurfaceCL.fit_core_sp_gram`,
  `.fit_core_sp_cg`, `.eval_core_sp_gpu`; `cpm_params_from_coremesh`
- `spammm/surfaces/CoarseMesh.py` — `fit_mesh_lsq_3d` (banded LSQ),
  `_prefilter_3d`, `CoarseMesh`
- `spammm/surfaces/CoreBasisStudy.py` — spec grammar `build_terms`, `design`,
  `fit_core_gram` (CPU Gram ref), `core_residual_study`
- `kernels/contact_surface.cl` — `cs_pme_sp_accum` (eval funnel),
  `cs_sp_Av`/`cs_sp_Atv`/`cs_reduce_groups` (fit operators),
  `cs_bspline_prefilter_lines`
- `tests/SPM/testplot_fdbm_fields_compress.py` — the harness above
- Design/design-history: `doc/Tasks/ContactPME_CoreMesh_Fit_Design.md` §13–18
