---
type: Report
title: Contact-PME PAW — AFM path, memory win, local-kernel SCAN, GPU mesh FIT
description: Soft-replacement PAW split; CLI-SSOT AFM; ~10³× field memory; Codex local WG SCAN; Python pts-loop wall tax fixed; GPU fillContactPMEMeshVL FIT mesh; core LS still host.
tags: [afm, contact-pme, paw, pic, particle-mesh, memory, benchmark, opencl]
timestamp: 2026-08-11
status: memory-goal-met; AFM-CLI-visual-regenerated; scan-local-default; fit-mesh-gpu; core-ls-host
---

# Contact-PME PAW — smooth split, real AFM, memory vs speed

**Task / plan:** [`../Tasks/ContactSurface_PME_ParallelPlan.md`](../Tasks/ContactSurface_PME_ParallelPlan.md)  
**CLI:** `run_spm.py afm --model contact_pme` → `AFM_utils.run_contact_pme_pp_afm` with **`core_backend='local'`** (Codex WG+local)  
**Bench:** `debug/test_afm_contact_surface/contact_pme/kernel_local/` · `wave2_diag/mem_speed/`  
**AFM strips (regenerated 2026-08-11 evening, local kernels + GPU mesh fill):**  
`debug/test_afm_contact_surface/contact_pme/wave2_afm_cli/{pyridine,PTCDA}/compare_per_image.png`  
**PAW vs Morse+Q maps:** `debug/.../wave2_paw_mol/{pyridine,PTCDA}_paw_vs_plqh_h4.5.png`  
**Device:** NVIDIA GeForce RTX 3090 (OpenCL Shell must use unrestricted `all`)

**Status:** Memory ~10³×. CLI SCAN uses **local** kernel. FIT mesh via **`fillContactPMEMeshVL`**. SCAN wall Python pts-loop removed. **USER confirmed regenerated AFM strips look good (2026-08-11).** Core LS still host.

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

## 4b. Naming + timings (avoid mixing apples/oranges)

Three **different** clocks:

| Name | What is timed | Where |
|------|---------------|-------|
| **FIT** | PAW split + mesh raster + core LS (once/molecule) | mesh→GPU; core LS still host |
| **SCAN kernel-only** | OpenCL event of fused FIRE PP kernel | GPU |
| **SCAN wall** | `run_scan_contact_pme` return | GPU + unavoidable FE download |

| SCAN kernel | Name |
|-------------|------|
| `relaxStrokesTiltedContactPME` | **bucket** (old) |
| `relaxStrokesTiltedContactPMELocal` | **local** (Codex optimized) — CLI/`auto` default |

### SCAN kernel-only (Codex events)

| Case | bucket | local | Speedup |
|------|--------|-------|---------|
| pyridine | 10.93 ms | **1.87 ms** WG64 | 5.85× |
| PTCDA | 207.60 ms | **15.65 ms** WG32 | 13.26× |

### SCAN wall — Python overhead was the real bug

Before harness fix, pyridine wall ~32 ms vs kernel 1.87 ms (**~17× host tax**). Root cause: **Python `for ix: for iy:` building 200k scan points (~20 ms alone)** + downloading 4 full telemetry volumes every scan.

After fix (vectorized pts + status-only telemetry + quiet prints):

| Molecule | SCAN wall (local) | vs Codex kernel-only |
|----------|-------------------|----------------------|
| pyridine | **~9.9 ms** | ~5× (was ~17×); FE download ~1.7 ms unavoidable |
| PTCDA | **~19.9 ms** | ~1.3× (nearly kernel-bound) |

### FIT

| Molecule | FIT before | FIT now | Mesh raster |
|----------|------------|---------|-------------|
| pyridine | ~0.43 s | **~11 ms** | GPU `fillContactPMEMeshVL` (parity vs CPU ~1e-8) |
| PTCDA | ~1.8 s | **~36 ms** | GPU (~3.6 ms) vs CPU (~20 ms) |

Core least-squares still host (small `na×5`); PAW a0 cache + closed-form a0 on host (tiny). **Mesh fill is OpenCL WG+local** (Codex-style).

**REVIEW:** `debug/.../kernel_local/`

---

## 4c. Host/GPU FIT details

1. **PAW a0** — closed-form quadratic min + `precompute_split_cache` (host, once/atom).
2. **Mesh V_L raster** — OpenCL `fillContactPMEMeshVL` (WG+local atom/PAW preload); host `_prefilter_3d` on tiny coarse grid.
3. **Core LS** — still `fit_core_1d` on host (na×5 lstsq); next GPU target if FIT must go further.

CPU `build_coarse_mesh` remains as fallback / parity oracle.

---

## 5. Kernel architecture notes

### 5.1 Is PPM relaxation in a single kernel?

**Yes.** `relaxStrokesTiltedContactPME` / `…Local` do, per scan pixel:

1. Outer loop over `nz` tip heights (stroke),
2. Inner FIRE / damped loop (`N_RELAX_STEP_MAX`, typically 128) calling `cs_eval_contact_pme_at` or local equiv,
3. `tipForce` + optional `OPT_FIRE`,
4. Write `FEs[gid*nz+iz]` + telemetry.

Same pattern as `relaxStrokes` / `relaxStrokesTiltedContact` for GridFF / 2.5D. Host does **not** run FIRE in Python for the GPU path.

### 5.2 Workgroups and local memory

| Piece | Launch / memory | Notes |
|-------|-----------------|-------|
| `relaxStrokesTiltedContactPME` | Host: `gs=(roundup(n_scan),)`, **`ls=(1,)`** | Bucket fallback; no WG sharing. |
| `relaxStrokesTiltedContactPMELocal` | `gs=(roundup(n,WG),)`, **`ls=(WG,)`** + `LATOMS`/`LCOEFFS` | Cooperative preload once; no barrier inside FIRE. |
| Dense Morse PP | `relaxStrokes` + `image3d_t` | Hardware texture path; still slightly ahead on wall (§4b). |

With local kernels, pyridine/PTCDA PP wall is within ~1.2–1.3× of Morse on this harness; remaining gap is FIRE arithmetic vs texture, not missing WG cache.

### 5.3 Python vs GPU overhead

| Stage | Where | Comment |
|-------|-------|---------|
| Split / mesh raster / core fit | **Python / NumPy (host)** | After §4c: tens of ms, no longer dominant for interactive AFM. |
| PP stroke + FIRE | **Single OpenCL kernel** | Local path ~2–4× wall vs bucket; kernel-only up to ~13×. |
| Scan FE download / df postprocess | Host | Shared with Morse. |

---

## 6. Ideas to improve performance (remaining)

1. **USER already requested fastest default** → `auto`/`CLI` use `local` when local mem fits; `bucket` only as overflow fallback.
2. **OpenCL event profiling** in harness (kernel-only vs wall) as in Codex packet.
3. **Optional `image3d` / texture for tiny coarse mesh.**
4. **FIRE warm-start / fewer evals** for screening.
5. **GPU-ize FIT** (mesh raster + core LS) — still CPU; policy gap vs “hot path on GPU”.
6. **Re-bench** when dense volumes grow (finer step / FDBM).
7. **Deduplicate PAW cache by element type** if FIT must go sub-10 ms on huge mols.

---

## 7. Takeaways (short)

1. **Split math first, optimize later** — `rho` / W-blend “smooth energy, wrong force” wasted cycles; PAW soft replacement fixed the representation.
2. **Memory goal is real** — ~10³× field compression; ML-scale storage becomes practical.
3. **Scan speed largely closed for pyridine/PTCDA class** — local WG kernels ≈ Morse wall (~1.2–1.3×); bucket was the 12–18× gap.
4. **Fit host path was the other bottleneck** — cache + closed-form a0 → ~30×; tens of ms now.
5. **Index/buffer contracts matter** — telemetry `n_scan` vs `n_scan·nz` looked like “physics garbage” until sized correctly.
6. **Compare like-with-like** — “no sharp bonds” was Morse physics at CLI heights, not missing `K_LAT`.
7. **FIRE float32 drift** on PTCDA local vs bucket is sparse and must stay visible until USER accepts maps / gate change.

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
# debug/test_afm_contact_surface/contact_pme/kernel_local/SUMMARY.out
```

**REVIEW paths:**

- `debug/test_afm_contact_surface/contact_pme/wave2_afm_cli/pyridine/compare_per_image.png`
- `debug/test_afm_contact_surface/contact_pme/wave2_afm_cli/PTCDA/compare_per_image.png`
- `debug/test_afm_contact_surface/contact_pme/wave2_paw_mol/pyridine_paw_vs_plqh_h4.5.png`
- `debug/test_afm_contact_surface/contact_pme/wave2_paw_mol/PTCDA_paw_vs_plqh_h4.5.png`
- `debug/test_afm_contact_surface/contact_pme/wave2_diag/morse_vs_pme/compare_morse_vs_pme.png`
- `debug/test_afm_contact_surface/contact_pme/wave2_diag/mem_speed/SUMMARY.out`
- `debug/test_afm_contact_surface/contact_pme/kernel_local/SUMMARY.out`
- `debug/test_afm_contact_surface/contact_pme/kernel_local/pyridine_Fz_local_vs_bucket.png`
- `debug/test_afm_contact_surface/contact_pme/kernel_local/PTCDA_Fz_local_vs_bucket.png`
