---
type: Report
title: Contact-PME PAW split — working AFM path, memory win, scan-speed analysis
description: Soft-replacement PAW long/short split for contact_pme; CLI-SSOT AFM on pyridine/PTCDA; ~900–1200× field-memory reduction vs dense Morse@0.1Å; PP-scan still ~12–18× slower than texture Morse; kernel/workgroup/local-memory notes and speed-up ideas.
tags: [afm, contact-pme, paw, pic, particle-mesh, memory, benchmark, opencl]
timestamp: 2026-08-11
status: memory-goal-met; AFM-path-USER-confirmed; scan-speed-open
---

# Contact-PME PAW — smooth split, real AFM, memory vs speed

**Task / plan:** [`../Tasks/ContactSurface_PME_ParallelPlan.md`](../Tasks/ContactSurface_PME_ParallelPlan.md)  
**CLI:** `run_spm.py afm --model contact_pme` (same ScanSpec / `AFM_utils` path as Morse/FDBM)  
**Bench artifacts:** `debug/test_afm_contact_surface/contact_pme/wave2_diag/mem_speed/`  
**AFM strip review:** `debug/test_afm_contact_surface/contact_pme/wave2_afm_cli/{pyridine,PTCDA}/compare_per_image.png`  
**Device:** NVIDIA GeForce RTX 3090 (agents must run OpenCL with unrestricted Shell so NVIDIA ICD is visible)

**Status (2026-08-11):** USER confirmed pyridine/PTCDA CLI-SSOT AFM strips look correct after the PAW split + scan-domain fixes. **Memory goal achieved (~10³×).** **Scan-speed / cache hypothesis not confirmed** — dense Morse PP remains faster on these molecule sizes.

---

## 1. What we set out to do

Classical PP-AFM stores a dense 3D force-field image (`img_FF`, float4 voxels at ~0.1–0.2 Å). That is fine for a few molecules, but for **ML datasets of hundreds–thousands of molecules** the voxel volumes dominate storage and residency.

`contact_pme` is a **particle-mesh analogy** (not Ewald/FFT PME):

```text
V(r) ≈ V_mesh(r) + Σ_i V_core_i(|r − R_i|)
```

- **`V_mesh`:** coarse nonperiodic 3D cubic B-spline (`h_mesh ≈ 1 Å`) carrying a **smooth long-range** field.
- **`V_core`:** compact atom-centered residual (PIC buckets + few radial modes per atom).
- Split is done **atomwise / radially before** mesh rasterization so the mesh never samples the steep repulsive wall.

Goal: same Morse+Q oracle AFM as the dense path, with **tiny resident field memory**, without inventing mesh/core artifacts.

---

## 2. The split: what was wrong, what works now

### 2.1 Dead-ends (errors to avoid)

| Dead-end | Why it failed | Lesson |
|----------|---------------|--------|
| **`rho(r)` soft-core** (`v_L = v(rho(r))`) | Coordinate map can have `ρ' > 1` → **steepening** and **displaced force bumps** in the mesh | Do not “distort the true radial function until smooth.” Distortion invents physics. |
| **`plateau` W-blend** `v_L = C + W(v−C)` | Force gets `W'(v−C)` in the switch → **fake force shells** even when energy looks smooth | Soft **replacement** of the interior, not multiplicative blend of `(v−C)`. |
| **Doubling-power core basis** `t^{4,8,16,…}` | Resolution collapses away from `r_lo`; cannot fit well/wall simultaneously | Basis must resolve the **residual** where it lives (uniform B-spline / modest mode set), not only the wall. |
| **`r_lo = R0 − 0.5` too high** | CLI amp-aligned Fz goes to ~2.7 Å → tip enters `r < r_lo` | Domain (`r_lo`) must sit **below** closest approach; raise `Δ_in` (now default **1.0**). |
| **Raise/NaN on `r < r_lo` in GPU relax** | One deep PP step kills the whole pixel | Clamp core basis for `r < r_lo` (t=1, dφ=0); treat as model edge, not silent abort mid-FIRE. |
| **Query envelope `z_above_lo=3` fixed** | Missed amp-aligned closest approach → mesh OOB / NaNs | Derive `query_bounds` from **ScanSpec** (+ PP pad). |
| **Telemetry buffers sized `n_scan` not `n_scan·nz`** | Kernel writes `gid*nz+iz` → **GPU overrun → garbage strips** in AFM maps | Always size telemetry to the kernel index formula. |
| **`core_d_span` only for `plateau`** | `paw`/`hermite` wrongly used legacy `r_cut` envelope | Compact modes share one span path (`_COMPACT_SPLIT_MODES`). |

### 2.2 Working split: `paw` (default)

Soft **replacement** of the interior by an even polynomial (smooth at the origin as a 3D radial field), C²-matched to the true radial potential at the outer join:

```text
P(r) = a₀ + a₂ r² + a₄ r⁴ + a₆ r⁶
r_b  = R0 + Δ_b
```

- Match `P, P', P''` to `v, v', v''` at `r_b`.
- Free `a₀` chosen to minimize `∫(P'')²` on `[0, r_b]` (softest curvature).
- Outside `r_b`: `v_L = v` (exact tail on the mesh); inside: `v_L = P`, `v_S = v − v_L` (compact residual for PIC).
- Geometry: `r_lo = R0 − Δ_in`, `r_a = R0 + Δ_a`, `r_b = R0 + Δ_b` with defaults **`Δ_in=1.0`, `Δ_a=0.5`, `Δ_b=2.0`**.
- Code SSOT: `spammm/surfaces/PMESplit.py` (`split_mode='paw'`).

Conceptual picture (realized):

```text
V_exact  = steep wall + well + smooth tail
V_mesh   ≈ soft even poly → C² join → exact tail   (no displaced minimum)
V_core   = compact difference (wall/well correction), → 0 at cutoff
```

Also retained for experiments: `hermite`, `softcore`, `plateau` (warned), `rho` (legacy).

### 2.3 AFM path (CLI SSOT)

- Fit: `AFMulator.fit_contact_pme(...)`
- Scan: `run_scan_contact_pme` → GPU `relaxStrokesTiltedContactPME`
- Harness: `AFM_utils.run_contact_pme_pp_afm` + `plot_afm_variant_height_strip`
- Heights: CLI `h_df=3.7…4.7`, `amp=1`, amp-align Fz; `K_LAT=0.5 N/m`, `K_RAD=20`, `L=3`
- MVP: `tipQs=0` (radial oracle only); optional `--pme-q-tip`

**USER review (2026-08-11):** strips look reasonable; sharp rings at very close approach are known `r_lo`/`r_b` join features (may lower `rmin` later). Lack of dramatic “sharp bonds” matches **Morse with the same params** (soft corrugation, `|dxy|≲0.4 Å` at Fz@2.7) — not a missing PP stiffness wiring.

---

## 3. Memory vs dense Morse — goal achieved

Device-resident **field** only (float4 dense image vs PME mesh+core+buckets). Same ScanSpec envelope. RTX 3090.

| Molecule | Dense Morse @0.1 Å | contact_pme `h_mesh=1.0` | Reduction |
|----------|--------------------|--------------------------|-----------|
| pyridine (11 at) | **30.0 MB** (128×120×128 ≈ 2.0M voxels) | **33.3 KB** (mesh 21×21×19 + 55 core coefs) | **~924×** |
| PTCDA (38 at) | **62.5 MB** (200×160×128 ≈ 4.1M voxels) | **53.4 KB** (mesh 29×24×19 + 190 coefs) | **~1200×** |

Coarser mesh (still usable for screening):

| | `h=1.0` | `h=1.5` | `h=2.0` |
|--|---------|---------|---------|
| pyridine | 33.3 KB | 23.2 KB | 18.6 KB |
| PTCDA | 53.4 KB | 32.3 KB | 25.4 KB |

**ML dataset projection (field store only):**

| N × molecule | Dense | PME `h=1` |
|--------------|-------|-----------|
| 1 000 × PTCDA | ~61 GB | ~0.05 GB |
| 10 000 × PTCDA | ~610 GB | ~0.5 GB |

Plan target “\<500 KB resident for PTCDA scan domain” is met with large margin (~53 KB).

---

## 4. Simulation speed — cache hypothesis not confirmed (yet)

Fair PP-scan: same `scan_xs/ys`, `h_scan`, `K_LAT/K_RAD/L`, FIRE. Median of 3 reps after warmup.

| Molecule | Morse build | Morse PP-scan | PME fit (host) | PME PP-scan | scan / Morse |
|----------|-------------|---------------|----------------|-------------|--------------|
| pyridine | ~2 ms | **~7 ms** | ~0.43 s | **~84 ms** | **~11–12× slower** |
| PTCDA | ~3 ms | **~17 ms** | ~1.8 s | **~303 ms** | **~16–18× slower** |

Per-state: Morse ~0.03 µs/state; PME ~0.4–0.55 µs/state.

Force-eval only (batched `evalContactPME`): ~0.06–0.12 µs/pt — eval itself is not terrible; **full PP-scan pays many evals inside FIRE** and loses badly to **3D texture `interpFE`**.

Coarser `h_mesh` barely changes scan time → **PIC/core + tricubic arithmetic dominate**, not mesh footprint.

**Interpretation:** at 30–62 MB the dense FF still rides the GPU texture/cache path well. PME is more arithmetic-heavy and (today) poorly parallelized inside the pixel (see §5). Memory win is real; speed win needs kernel work, not coarser mesh alone.

Fit cost (0.4–1.8 s) is mostly **host** mesh/core construction — fine if amortized once per molecule for a dataset; painful for interactive one-offs vs Morse’s ~ms build.

---

## 5. Kernel architecture notes (why speed is still open)

### 5.1 Is PPM relaxation in a single kernel?

**Yes.** `relaxStrokesTiltedContactPME` (`kernels/contact_surface.cl`) does, per scan pixel:

1. Outer loop over `nz` tip heights (stroke),
2. Inner FIRE / damped loop (`N_RELAX_STEP_MAX`, typically 128) calling `cs_eval_contact_pme_at`,
3. `tipForce` + optional `OPT_FIRE`,
4. Write `FEs[gid*nz+iz]` + telemetry.

Same pattern as `relaxStrokes` / `relaxStrokesTiltedContact` for GridFF / 2.5D. Host does **not** run FIRE in Python for the GPU path.

### 5.2 Workgroups and local memory?

**Mostly no — this is a primary performance gap.**

| Piece | Launch / memory | Notes |
|-------|-----------------|-------|
| `relaxStrokesTiltedContactPME` | Host: `gs=(roundup(n_scan),)`, **`ls=(1,)`** | One work-item per XY pixel; **no workgroup sharing**. |
| `cs_eval_contact_pme_at` | Inline in that WI | Mesh tricubic (64-tap globals) + PIC 3×3 buckets — **all `__global`**, no `__local` tile. |
| `cs_pme_core_eval_at` | Same | Per-atom global loads; no local atom/coeff cache. |
| Contrast: `cs_brute_afm_morse_c_points`, older PIC tiles | `__local` atom tiles + barriers | Pattern exists in the same `.cl` file but **not wired into contact_pme relax**. |
| Dense Morse PP | `relaxStrokes` + `image3d_t` | Hardware texture path; hard to beat at small volumes. |

So the user’s suspicion is right: **PP is fused in one kernel, but without workgroups/local memory the “cache-efficient compact representation” advantage is not realized in the hot path.** Compact data helps residency; it does not automatically help latency when every FIRE step re-fetches globals with `local_size=1`.

### 5.3 Python vs GPU overhead

| Stage | Where | Comment |
|-------|-------|---------|
| Split / mesh raster / core fit | **Python / NumPy (host)** | Dominates “fit” time (0.4–1.8 s). Candidate for GPU later if dataset throughput matters. |
| PP stroke + FIRE | **Single OpenCL kernel** | Not Python-bound; limited by eval cost × FIRE iters × WI scheduling. |
| Scan FE download / df postprocess | Host | Shared with Morse; small vs PTCDA PME scan (~0.3 s). |

---

## 6. Ideas to improve performance (no code in this report)

Priority order suggested by the measurements:

1. **Workgroup + local memory for PIC**  
   Load 3×3 bucket atoms/coeffs into `__local` once per tile of pixels (or per workgroup of nearby scan points). Reuse pattern from existing `CS_PIC_LOCAL_MAX` kernels.

2. **Do not launch with `local_size=(1,)`**  
   Even without changing math: group WIs that share buckets / mesh cells; coalesce mesh stencil loads.

3. **Mesh stencil caching**  
   Tricubic 4×4×4 is 64 scattered reads per eval. Prefetch cell neighborhood into local/private; or texture/`image3d` for the coarse mesh (tiny — fits easily).

4. **Fewer FIRE force evals**  
   Early exit / looser `F2CONV` for screening; warm-start from previous `iz`; compare iteration histograms vs Morse (same tip params).

5. **GPU-ize fit path** (secondary for interactive AFM; primary for huge datasets)  
   Mesh fill + core LS are currently host-bound.

6. **When dense may lose**  
   Re-bench at finer step (0.05 Å), larger vacuum, or FDBM multi-field volumes where dense ≫ L2/texture — PME may then win on **both** memory and scan. Current 30–62 MB Morse boxes are a best case for texture.

7. **Keep `h_mesh` tuning for memory, not speed**  
   `h=1→2` saves KB but barely moves PP-scan time.

---

## 7. Takeaways (short)

1. **Split math first, optimize later** — `rho` / W-blend “smooth energy, wrong force” wasted cycles; PAW soft replacement fixed the representation.
2. **Memory goal is real** — ~10³× field compression; ML-scale storage becomes practical.
3. **Speed goal is open** — fused PP kernel exists, but **no WG/local-mem** and arithmetic-heavy eval lose to texture Morse on small grids.
4. **Index/buffer contracts matter** — telemetry `n_scan` vs `n_scan·nz` looked like “physics garbage” until sized correctly.
5. **Compare like-with-like** — “no sharp bonds” was Morse physics at CLI heights, not missing `K_LAT`.

---

## 8. Reproduction

```bash
# AFM strips (NVIDIA ICD visible — unrestricted env)
pytest tests/SPM/test_afm_contact_surface.py::test_contact_pme_afm_cli_ssot --develop -s

# CLI
python run_spm.py afm --model contact_pme --xyz data/xyz/pyridine.xyz --show-atoms
python run_spm.py afm --model contact_pme --xyz data/xyz/PTCDA.xyz --pme-q-tip -0.1 --show-atoms

# Bench tables
# debug/test_afm_contact_surface/contact_pme/wave2_diag/mem_speed/SUMMARY.out
# debug/test_afm_contact_surface/contact_pme/wave2_diag/mem_speed/results.json
```

**REVIEW paths:**

- `debug/test_afm_contact_surface/contact_pme/wave2_afm_cli/pyridine/compare_per_image.png`
- `debug/test_afm_contact_surface/contact_pme/wave2_afm_cli/PTCDA/compare_per_image.png`
- `debug/test_afm_contact_surface/contact_pme/wave2_diag/morse_vs_pme/compare_morse_vs_pme.png`
- `debug/test_afm_contact_surface/contact_pme/wave2_diag/mem_speed/SUMMARY.out`
