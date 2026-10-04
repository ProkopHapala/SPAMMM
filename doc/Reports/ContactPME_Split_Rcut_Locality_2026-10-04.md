---
type: Report
title: Contact-PME split mechanics, Rcut sweep, and locality analysis
tags: [contact-pme, paw, split, rcut, locality, pic, afm]
timestamp: 2026-10-04
---

# Contact-PME: split mechanics, Rcut sweep, locality

Analysis session on `graphene_10x10x2L_N.xyz` (468 atoms, bilayer, N defect centered,
C–C ≈1.42 Å, layer gap ≈3.4 Å). Oracle = GPU brute Morse (`cs_brute_afm_morse_c_points`).
Queries restricted to PP-relevant heights h ≥ 2.6 Å above the top atom (below that the
r<r_lo close-approach cap dominates any comparison — by design).

## User goals (driving this analysis)

- **Minimize the compact-core radius** — locality is the scaling mechanism: fewer halo
  atoms per tile → smaller local-memory load per workgroup. ~1e-4 eV field error is fine.
- Let the **mesh absorb as much of the long-range part as possible**, leaving only a
  very localized core.
- **Never** deduplicate per-atom fits by element type: the core is a residual *after*
  the smooth mesh, so it depends on each atom's position relative to the mesh.
- Real PIC partition = uniform **square** bucket cells; show actual per-atom `r_b`
  cutoff spheres; distinguish atom→mesh projection (dense) from PIC partition (local).

## How the split actually works (SSOT check)

`soft_core_split` (PMESplit.py, `split_mode='paw'` default):

- `v_L` = even polynomial inside `r_b` (C²-matched to v, v′, v″ at `r_b`, finite at
  r=0) and `= v` outside `r_b`. So the mesh never sees the diverging Morse core
  (that is the anti-ringing damping the user asked about — **the split IS the damping**;
  `Δ_b` sets how much of the well is smoothed for the mesh).
- `v_S = v − v_L` is the compact residual, fit per atom by `fit_core_1d` on
  `[r_lo, r_b]` with basis `φ_m = t^(2^n)`, `t = (r_b−r)/(r_b−r_lo)`, powers
  (2,4,8,16,32) — **exactly zero at r_b** (this IS the core cutoff; `Rcut = r_b`).
- `r_b = R0 + Δ_b`, `r_lo = R0 − Δ_in` (`Δ_in=1.0`, `Δ_b=2.0` defaults; R0 = tip-atom
  Morse minimum ≈3.38 Å for C-tip → r_b=5.38 Å). `delta_b` is a free parameter,
  now exposed as `AFMulator.fit_contact_pme(delta_b=...)`.
- The core is fitted against the **analytic** `v_S`, not against `v − V_mesh`
  (the two coincide up to mesh interpolation error — see failed experiment below).

## Key findings

### 1. The residual is essentially INVARIANT to Rcut

v_S(r) curves for Δ_b = 0.4…2.5 almost perfectly overlap: the residual is the
repulsive core + well edge, dead (~0) by r ≈ 3.8–4.0 Å regardless of where the join
is placed. Because v_L is C²-matched at r_b, wherever v is smooth, v−v_L ≈ 0.
**Consequence:** `r_b` is a pure locality knob — the only constraint is "smallest r_b
where the residual is already dead" ≈ 4.0–4.2 Å. Artifact:
`debug/testplot_pme_tiles/residual_vs_Rcut.png`.

### 2. What the parts carry (field decomposition, real fitted system)

xz cut + +z profile: `V_mesh` = smooth basin ≈ −0.011 eV, NO atomic contrast;
`V_core` carries **all** the atomic corrugation (±0.005 eV at lattice period) — the
AFM image lives entirely in the compact cores. At the well depth V_mesh ≈ −0.015 eV
— the mesh carries most of the binding already. Bottom layer contributes nothing at
PP heights. Artifacts: `debug/testplot_pme_tiles/field_decomp_graphene_10x10x2L_N.png`,
`debug/testplot_pme_tiles/parts_1d_zoom.png`,
`debug/testplot_pme_tiles/residual_2d_xz.png`.

### 3. Rcut sweep on the real system (PME vs brute, h=2.6–6.0 Å)

| Δ_b | r_b | max\|ΔE\| | max\|ΔF\| | halo scale (≈r_b²) |
|-----|-----|-----------|-----------|--------------------|
| 2.5 | 5.88 | 5.0e-4 | 5.7e-4 | 1.19× |
| 2.0 (old default) | 5.38 | 4.2e-4 | 6.6e-4 | 1.00× |
| 1.5 | 4.88 | 1.3e-4 | 3.8e-4 | 0.82× |
| **1.2 (accuracy optimum)** | 4.58 | **1.0e-4** | **3.7e-4** | 0.72× |
| **0.6 (locality pick)** | 3.98 | 2.6e-4 | 8.1e-4 | **0.55×** |
| 0.4 | 3.78 | 1.7e-4 | 7.8e-4 | 0.49× |
| 0.25 | 3.63 | 1.6e-4 | 9.2e-4 | 0.45× |
| 0.05 | 3.43 | 3.7e-4 | 1.3e-3 | 0.41× |

- Optimum ≈ Δ_b 1.2: r_b lands just past where v turns slowly-varying — v_S small
  AND v_L mesh-smooth. Smaller still → mesh has to fake live residual tail → |ΔF|
  climbs (1.3e-3 at Δ_b=0.05).
- **h_mesh barely matters** (0.5/0.75/1.0 nearly identical) — the PAW poly is already
  mesh-smooth at h=1 Å. The floor is the residual's tail, not mesh resolution.
- Recommendation per goals: **Δ_b=0.6 (r_b≈4.0)** aggressive, Δ_b=1.2 conservative.
  Artifact: `debug/tile_scan/db_sweep_err.png` (data `db_sweep.npy`).

### 4. "Mesh-first, core-absorbs-residual" experiment — fails, instructively

User proposal: core should fit `v − V_mesh` (absorb mesh interpolation error too,
not just the analytic residual). Measured per-atom shell-averaged mesh error
δ = E_brute − E_mesh − Σv_S: **−3e-3 eV mean, up to 6e-3** near atoms — real.
But fitting radial cores to v_S+δ → total field error **1e-4 → 3.5e-2** (300× worse).
Why: mesh error is **anisotropic** (sub-grid phase) and, on 1.42 Å-dense graphene,
**cross-contaminated** between neighbors — a radial per-atom basis cannot represent
it; it injects systematic error in directions where the true error differs.
Conclusion: keep the analytic split; to absorb mesh error you would need angular
core modes or per-atom mesh_i attribution (heavy machinery for ~1e-4 eV gain).

### 5. Layer pruning — confirmed

At proper PP heights (PP z ∈ [6.1, 9.1] for h_df 3.7–4.7 over ztop=3.4),
bottom-layer atoms (z=0) are ≥6.1 Å from any query > r_b=5.38 → their cores are
**dead** in the whole stroke — excludable from tile atom lists (they stay in the
mesh, long-range unaffected). In the cell detail: 10 top-layer atoms inside, 43
halo, 63 z-screened (mostly bottom layer), ~300 excluded. Artifact:
`debug/testplot_pme_tiles/pic_cell_detail_graphene_10x10x2L_N.png`.
Also fixed: `_pme_build_wg_tiles` swept-box z-top was the *tip* top — replaced by
PP reach `[tipZ−L, tipZ−L+ρ²/2L]` so z-culling actually prunes.

### 6. AFM image verification (same system, CLI convention)

df = `-dFz/dz` amplitude-convolved (`compute_df_amp_z`, **needs ascending-z input** —
descending stacks silently flip the sign; plot df with `cmap='gray'` + per-panel
autoscale, never symmetric norm). 4×4 Å patch over the N defect @0.1 Å/px,
3 rows GridFF/PME-local/PME-tile: same honeycomb contrast at h_df=3.7–4.7.
Artifact: `debug/tile_scan/df_3way_zoom_Ndefect.png`.

## Parameters

| name | where | meaning | default | recommended |
|------|-------|---------|---------|-------------|
| `r_b` | SplitParams.r_b = R0+Δ_b | **Rcut**: v_L join + core support end | R0+2.0 | R0+0.6…1.2 (≈4.0–4.6 Å) |
| `r_lo` | R0−Δ_in | inner cap; basis flat below (close-approach clamp) | R0−1.0 | keep |
| `h_mesh` | fit_contact_pme | mesh spacing | 1.0 | keep (no gain at 0.5) |
| `delta_b` | fit_contact_pme kwarg | **new** exposed knob | 2.0 | 0.6–1.2 |

## Diagnostics now available

`tests/SPM/testplot_pme_tiles.py --mol <xyz> --rho 1.0` produces, in
`debug/testplot_pme_tiles/`:
- `pic_cell_detail_*` — central PIC cell: inside/halo/z-screened atoms + r_b spheres (XY+XZ)
- `projection_*` — atom→cell scatter vs atom→mesh (dense V_mesh plane)
- `core_basis_1d_*` — split decomposition, basis modes, fitted core, fit residual
- `field_decomp_*` — xz maps V_mesh / ΣV_core / total + h=2.7 profile
- `residual_vs_Rcut.png`, `split_actual_residual.png`, `parts_1d_zoom.png`, `residual_2d_xz.png`

## Tile boxes and local memory (same flake, 48×48 scan, tile 8×4, ρ=1 Å, nzL=6)

The kernel does not allocate the mean. Every workgroup gets `nloc_max` atoms
(`LATOMS` = nloc×16 B, `LCOEFFS` = nloc×5×4 B) plus one mesh slab
`LMESH` = nxL_max×nyL_max×nzL×4 B. On this scan the pixel box is 4.54×1.27 Å
and the swept box (pixels + ρ) is 6.54×3.27 Å. In-box and margin do **not**
depend on Rcut; only the halo does. Device: GTX 1650, 48 KB local.

| Δ_b | r_b | in mean/max | margin mean/max | halo mean/max | nloc max | lower layer in worst | local | % of 48 KB |
|-----|-----|-------------|-----------------|---------------|----------|----------------------|-------|------------|
| 2.0 (default) | 5.38 | 3.4 / 8 | 10.1 / 22 | 96 / 145 | 159 | 74 | 8.64 KB | 18% |
| **1.2** | 4.58 | 3.4 / 8 | 10.1 / 22 | 76 / 111 | **128** | 54 | **7.55 KB** | 16% |
| 0.6 | 3.98 | 3.4 / 8 | 10.1 / 22 | 61 / 90 | 105 | 49 | 6.74 KB | 14% |

Worst tile at Δ_b=1.2 (the one that sizes the launch): **in-box 0 + margin 20 + halo 108 = 128**.
The 1.27 Å-tall pixel strip falls between graphene rows; the ρ margin catches the
nearest row; the halo is everything else inside r_b of the swept box. Of those 128,
54 are the bottom layer, because the stroke z-box goes down to 2.25 Å and
r_b=4.58 still reaches z=0. Mesh slab is 13×10×6 floats = 3.05 KB and does not
shrink with Rcut (40% of the 7.55 KB). All-atom `Local` kernel would preload
468×36 B = 16.5 KB and still leave the mesh in global memory.

PIC buckets are a different partition (cell = r_b, one owner per atom). At
Δ_b=1.2: max 20 atoms/cell, 144 atoms in the 3×3 around the origin. The tile
kernel does not walk that 3×3; it loops the preloaded 128.

Headroom: 48 KB holds about 1280 atoms/tile after the 3 KB mesh. Graphene at
this tile size is far under that. The cost of Rcut is the inner-loop length
(128 vs 159), not the local-memory ceiling.

## Open items

- `relaxStrokesTiltedContactPMETileSph` escapes cluster at mesh-node-boundary z
  slices (~7% of evals → exact global fallback; results correct, perf cost small) —
  suspected slab off-by-one in `tile_kz` indexing, deprioritized.
- `fit_core_1d` still a Python per-atom loop (~80% of fit time, ~1 s @ 1748 atoms) —
  slated for an OpenCL per-atom fit kernel (B3 in `doc/Tasks/PME_ContactSurface_Opt.md`);
  per-atom coefficients MUST be kept (no type dedup).
- Δ_b default still 2.0 — changing the default pending user decision
  (0.6 for locality vs 1.2 for accuracy).
