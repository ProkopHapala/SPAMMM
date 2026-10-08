---
type: Task
title: Contact-PME AFM optimization — tiled local-memory evaluator + GPU-resident fitting
tags: [afm, contact-surface, contact_pme, opencl, local-memory, b-spline, prefilter, pic]
---

# Task: Contact-PME AFM optimization — tiled evaluator + GPU fitting

- **Status:** in progress (started 2026-10-04). Step gates: none confirmed yet.
- **Source discussion:** [`doc/chats/AFM_PME_opt.chat.md`](../chats/AFM_PME_opt.chat.md) — USER/ChatGPT design dialogue; this doc is the agreed plan distilled from it.
- **Parent task:** [`doc/Tasks/ContactSurface_PME_ParallelPlan.md`](ContactSurface_PME_ParallelPlan.md) (contract v2, frozen baseline `c5494c3`)
- **Related analysis:** [`doc/Reports/ContactPME_Split_Rcut_Locality_2026-10-04.md`](../Reports/ContactPME_Split_Rcut_Locality_2026-10-04.md) — split mechanics, Rcut=Δ_b sweep (residual dead by ~4 Å; r_b is locality knob → Δ_b≈0.6), layer pruning, mesh-first-residual experiment (fails — anisotropic error)

Two independent work streams, verified against the same golden references:

- **Track A — evaluation kernel:** local-memory layout + preload strategy.
  The current `Local` kernels copy **all** atoms into local memory and leave the
  mesh in global memory — the spatial-culling point of the PME split is unused.
- **Track B — fitting:** move remaining CPU/NumPy/SciPy numerics to OpenCL
  (mesh prefilter, per-atom split-cache loop, per-atom `fit_core_1d`).

**Scope note (2026-10-07):** all of this is the **Morse(+Q) oracle** track —
the PAW split needs an analytic per-atom `v_i(r)`. FDBM fields are
non-pairwise and need a field-space split instead (see `doc/Caveats.md`
§22); the tile/locality numbers here do not transfer verbatim to FDBM.

## Hard design rules (USER — non-negotiable, do not regress)

1. **No heavy numerics in Python.** Grid-sized and atom-sized numerical work
   (eval, raster, prefilter, core-fit sampling/solves) belongs in OpenCL kernels.
   Python orchestrates only: launches, buffer setup, small constants/metadata.
   Tight Python loops over atoms or grid elements are **prohibited** — even
   "cheap" ones grow with system size. See `skill:perf`, `skill:port-to-opencl`.
2. **No dedup-by-parameter-tuple for core fits.** Atoms that share `(R0,E0,q)`
   do **not** share core coefficients conceptually: the compact core is the
   residual of *that atom's* potential after the split, and evaluation uses
   per-atom coefficients — so fitting must be per-atom too. The scaling
   mechanism is **locality** (each core is short-range, `r_b ≈ R0+Δ_b`), not
   deduplication. Speed comes from a GPU kernel fitting all atoms in parallel,
   not from fitting fewer atoms.
3. **PP never reads global memory outside its tile.** The workgroup tile +
   halo must cover everywhere the PP can go. Keep the lateral bound tight
   (`rho_cap ~ 0.5–1.0 Å`) so the halo stays ~1 cell — never enlarge the tile
   to accommodate unphysical PP excursions.
4. **Clamped slices are invalid, not facilitated.** Probe-particle lateral
   deflection beyond `rho_cap` only occurs (a) when the tip has pushed the PP
   to/below the molecule — deeper than any real experiment — or (b) on a
   numerical blow-up. In both cases the result is meaningless: mark the slice
   invalid (`status` bit 8), abort the stroke early (deeper slices are also
   invalid), and let consumers cut the invalid region. The sphere constraint
   `|dpos|=L` stays analytic (`z=-sqrt(L²-x²-y²)`, no trig — `sph` mode); the
   lateral cap is an additional hard bound `|dxy| ≤ rho_cap`.
5. **Reference paths stay as oracles.** `evalContactPME{,Local}` and the
   `relaxStrokesTiltedContactPME{,Local*}` kernels remain for parity; new paths
   must match them on the valid (unclamped) region before perf claims.

## Current state (verified 2026-10-04)

### Evaluation (`kernels/contact_surface.cl`)

- `cs_pme_tricubic_eval` (L1062): 4×4×4=64-tap cubic B-spline, z-fastest `vload4`,
  **global** memory read on every force eval.
- `cs_pme_core_eval_at` (L1141): XY 3×3 bucket lookup per query (global).
- `cs_pme_core_eval_local_at` (L1202): loops over **all** `na` atoms in
  `LATOMS`/`LCOEFFS` — 36·na bytes/WG, no culling.
- `relaxStrokesTiltedContactPMELocal{,QN,Sph}` (L1655/1854/2008): preload all atoms,
  `if(gid>=n_scan) return` after preload barrier, divergent relaxation — no
  barriers inside FIRE/QN/sph loop.
- `cs_pic_eval_tile16` (L646): existing 2D-tiled kernel with cooperative
  bucket-atom preload — **the pattern to copy**.
- `fillContactPMEMeshVL` (L1738): dense all-atom→all-node raster, tiles atoms
  through local mem (already GPU).

### Fitting (CPU pieces to remove)

- `_pme_build_coarse_mesh_gpu` (AFM.py L1699): GPU raster → **GPU→CPU copy →
  scipy `solve_banded` ×3 (`_prefilter_3d`) → back to GPU** on next upload.
- `fit_core_1d` remains the oracle for non-paw modes and for the random-shell fit.
  PAW production path is `cs_fit_core_paw`: one thread per atom, 32 Chebyshev
  nodes, 5×5 Cholesky. Measured on GTX 1650, na=1520: Python loop 861 ms, GPU
  1.2 ms, max |Δc| vs the float64 grid 6e-6. Not user-confirmed.
- `_pme_build_coarse_mesh_gpu` (L1717): `for ia in range(na):
  precompute_split_cache(...)` Python loop.
- Prefilter matrix: `[1,4,1]/6` tridiagonal, zero-padded boundary, κ<3 —
  exact Thomas solve, **no CG/QN/Cholesky of a big matrix needed**
  (`ContactSurface._bspline_tridiag_ab`, CoarseMesh._prefilter_3d L57).

## Phase 0 — Baseline + verification (gates everything)

- **0.1 Evaluator microbenchmark:** fixed ~10⁶ query set inside the query
  interior; ns/eval for `evalContactPME` (bucket) vs `evalContactPMELocal`,
  NVIDIA GPU only (device selection per `doc/AGENTS/notes/opencl-nvidia-device.md`).
- **0.2 Golden reference:** one representative `run_scan_contact_pme` scan
  (`relax_mode='sph'` and `'fire'`); save FEs + `out_pp` + iters under `debug/`.
  Parity tol ~1e-4–1e-5 rel on F (summation order changes → not bit-exact).
- **0.3 Regression gate:** `pytest tests/SPM/test_afm_contact_surface.py -m "not slow"`
  after every step (`mesh_force_parity`, `core_force_parity`,
  `bucket_completeness`, `paw_molecule_parity`, `afm_cli_ssot`).

## Track A — evaluation kernel: local-memory layout + preload

### A1. Per-WG atom lists (restores the PME point; biggest win for large systems)

- Host (`_pme_scan_gpu`): split scan grid into compact 2D pixel tiles (e.g. 8×4);
  per tile, atom list = atoms whose `r_b`-sphere intersects swept box
  `(tile XY footprint + ρ_max) × [z_min, z_max]` — vectorized numpy, once/scan.
- Buffers `wg_atom_ids` + `wg_atom_offsets` (CSR) uploaded per scan.
- Kernel: cooperative preload of listed atoms+coeffs only; inner loop `nlocal`.
- Bin WG launches by `nlocal` (≤32/64/128) so dense corners don't kill occupancy.

### A2. Mesh slab cache: `NxL × NyL × NzL` local tile, streamed in z

- Lateral window = union of lanes' stencil cells: `N = 4 + (i_max − i_min)`
  per axis (typ. 8–14 nodes at h=0.5–1.0 Å, ρ_max≈0.5–1 Å → ~2–4 kB).
- **z minimal:** `NzL = 4 + (jz_max − jz_min)`; `jz_{min,max}` from sphere
  constraint `Δz ≈ ρ²/(2L)` ≈ 0.04–0.17 Å ⇒ almost always `NzL=4`, sometimes 5.
- Host precomputes `kz_lo[nWG × nz]` (tip path + ρ_max known).
- Kernel: at each `iz`, all lanes (incl. inactive — replace `return` with
  `active` flag) read `kz_lo`; if changed → cooperative reload of `NzL` planes
  → barrier → divergent relaxation (ring buffer only if reload shows hot).
- Local tile layout: z-fastest `(ix,iy,iz_local)` → eval keeps contiguous `vload4`.
- **Escape path:** PP cell outside cached window → fall back to global
  `cs_pme_tricubic_eval`/`cs_pme_core_eval_at` for that eval + atomic escape
  counter (target <1–2%; absolute correctness preserved).

### A3. New kernel `relaxStrokesTiltedContactPMETileSph`

- sph variant first (z provably bounded). Existing kernels untouched =
  fallback + parity reference. First pass only (`iz0=0`); pass-2 stragglers
  rerun with existing kernel. 2D launch `(nx_s,ny_s)` × `(Tx,Ty)`,
  mirrors `cs_pic_eval_tile16`. Keep bucket/`Local`/`QN` paths working.

### A4. Measure & tune

- Sweep `(Tx,Ty)`∈{4×8,8×4,8×8}, `h_mesh`∈{0.5,1.0}, `ρ_max`; report ns/eval,
  scan time, escape rate. Only then consider Q2×Q2×Q3 (36 taps) — new
  basis+prefilter; cubic stays default.

## Track B — fitting: CPU/NumPy → OpenCL

### B1. GPU tridiagonal prefilter (isolated; do first)

- One kernel `cs_bspline_prefilter_lines(buf, n_lines, line_len, n_inner,
  invden, cprime)`: one work-item per line, Thomas algorithm.
  `n_inner=1` → z-lines; `n_inner=nz` → y-lines; `n_inner=ny*nz` → x-lines.
- `invden`/`cprime` (len n, tens of floats) host-precomputed once — scalar
  setup, not grid numerics. Must reproduce `_bspline_tridiag_ab` **exactly**
  (diag 4/6, off 1/6, zero-pad; skip axis when n<3).
- `_pme_build_coarse_mesh_gpu`: raster → 3 prefilter launches → download coeffs
  once for `ContactPMEParams` (host copy stays SSOT for CPU fallback).
  Same helper reused by `pme_set_field`.
- **Verify:** L0 parity vs `solve_banded` on random arrays + real mesh.

### B2. Kill per-atom Python loops in mesh-build prep

- `for ia: precompute_split_cache` — the paw coefficient construction
  (`_paw_even_coeffs` 3×3 solves + polynomial consts) is per-atom metadata; move
  it into an OpenCL kernel (one lane per atom, emits `paw`/`reqs`/`r_b` rows
  directly into the fill buffers) or batch via stacked `np.linalg.solve` as a
  temporary measure only. Applies to `build_coarse_mesh` too.

### B3. GPU per-atom 5×5 core fit (replaces the old dedup idea)

- `fit_core_1d`'s Python loop is the dominant fit cost at scale
  (measured 1.0 s of 1.2 s total at na=1748). Port to OpenCL: one WG (or lane)
  per atom — sample `soft_core_split` + `core_basis` at the radial grid,
  accumulate symmetric G (15 floats) + b (5) in local-mem reduction, one lane
  Cholesky 5×5; broadcast the result. Thousands of independent tiny solves —
  no global matrix. Per-atom fits stay per-atom (see rule 2).
- **Verify:** coefficients match `fit_core_1d` per atom (tolerance ~1e-3 abs —
  radial sampling can share a deterministic grid so results are near-identical).
- Optional further: `c_i = E0_i·c_M + q_i·c_C` requires dropping Boltzmann
  weights — **accuracy experiment first** (`test_contact_pme_core_fit_quality`).

### B4. (only if B3 insufficient) further batched solve options

- Same WG-per-atom structure can host both the raw and hierarchical
  (`A @ Hᵀ`) solves and pick per `cond`/`held_rmse` like the CPU path.

### B5. Profile-gated follow-up

- `fillContactPMEMeshVL` is O(ntot·na) dense — fine to ~10⁴ atoms; beyond that
  needs design attention (no trivial culling for the long-range part).
  Do not pre-optimize.

## Recommended order

1. Phase 0 (baseline + golden refs)
2. **B1** — isolated, small, immediate fit win
3. **A1** — biggest eval win for large systems, prerequisite for A3
4. **A2+A3** — tiled kernel (+ `rho_cap` hard bound, invalid-slice marking,
   early stroke abort — rules 3–4)
5. **B3** — GPU per-atom core fit (biggest fit win at scale; dedup rejected)
6. A4 tuning; B2/B4/B5 only if profiling still shows pain
