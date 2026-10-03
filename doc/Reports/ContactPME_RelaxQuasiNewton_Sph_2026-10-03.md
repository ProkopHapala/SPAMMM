---
type: Report
title: contact_pme PP relaxation — quasi-Newton & sphere-constrained solvers
tags: [contact-pme, afm, relaxation, quasi-newton, sphere-constraint, opencl, perf]
timestamp: 2026-10-03
---

# Contact-PME PP Relaxation: Quasi-Newton & Sphere-Constrained Solvers

## Summary

Two **new** relaxation kernels for the `contact_pme` Local-backend scan, alongside the
untouched FIRE reference (`relaxStrokesTiltedContactPME`, `…Local`):

- `relaxStrokesTiltedContactPMELocalQN` — 3-phase quasi-Newton on all 3 Cartesian PP DOFs.
- `relaxStrokesTiltedContactPMELocalSph` — **sphere-constrained** Newton: `|dpos|=L` is
  enforced analytically; only the 2 *soft lateral* DOFs are iterated.

Selected via `run_scan_contact_pme(..., relax_mode='fire'|'qn'|'sph', qn_cap=0, qn_conv=1.0)`
(local backend only). The FIRE path is the default and unchanged.

**Result (PTCDA, 240² px × 31 z-slices, GTX 1650):** wall 95–100 ms → **~23–27 ms (~4×)**
with `relax_mode='sph'`, NCC(Fx) vs FIRE ≈ 0.998–0.9999, max|ΔF| ≈ 0.003–0.014 eV/Å.

## Motivation & root cause of FIRE slowness

PP equilibrium solves `G(d) = f_sample(d) + f_tip(d) + f_surf = 0`. FIRE is damped-MD:
its global dt is limited by the **stiffest** DOF — the radial tip spring
(`K_RAD≈20 N/m` vs lateral `K_LAT≈0.5 N/m`, ~40:1 conditioning) — so the soft lateral
DOFs (the physically interesting ones) converge slowly. Measured FIRE mean ≈ **6–7
evals/cell** (iteration cap 128 is hit only by a thin deep-contact tail; warp-divergent
lanes still dominate wall time).

| relax_mode | mean evals/cell (implied) | kernel ms | wall ms | NCC(Fx) | max|ΔF| eV/Å |
|-----------|---------------------------|-----------|---------|---------|------|
| `fire`    | ~7 (+final)                 | ~86–94    | ~95     | —       | —    |
| `qn`      | ~4                          | ~24       | ~33–35  | 0.998   | 0.024|
| `sph`     | ~1.2                        | ~15–17    | ~23–27  | 0.998–0.9999 | 0.003–0.014 |

**Cap ≠ mean.** Wall time scales with *mean evals/cell* (~8 ns/eval on GTX 1650),
not with the iteration cap. The tail (deep contact slices) is what hits the cap.

## Algorithms

Both new kernels share the FIRE kernel's column loop (tip descends `nz` slices,
`pos += dTip` warm start) and identical telemetry (`out_status/min_r/offender/overflow`,
plus `out_iters`).

### `…LocalQN` — diagonal-secant quasi-Newton (3 DOF)

1. Phase 1 (≤ `QN_BUDGET=8` iters): `dx = −f/Jd`, `Jd` init = `stiffness.xyz`
   (Hooke `dx=F/k`, linear-response), refined per-axis by secant `Δf/Δx`;
   trust cap |dx|≤0.1 Å, reject-and-shrink on force growth, `Jd` clamped negative
   and **carried across z-slices**.
2. Phase 2 (≤ `NR_BUDGET=16`): full 3×3 FD-Newton of total `G` (3 extra evals/iter),
   symmetrized, `solve3x3_general`.
3. Phase 3: plain FIRE (last-resort tail).

### `…LocalSph` — sphere-constrained Newton (2 DOF)

PP constrained analytically to `|dpos|=L=dpos0.w` in tip coords:
`dpos_tip = (x, y, −√(L²−x²−y²))`. Residual = total force projected on the two
tangent directions `e₁ = a + (x/|z|)c`, `e₂ = b + (y/|z|)c` — the stiff radial DOF
becomes a Lagrange multiplier (`K_RAD→∞` limit). Same 3-phase structure, now
diagonal-secant → 2×2 FD-Newton → damped-MD on `(x,y)`. Converges in ≤8–20
iters/slice everywhere at strict tolerance (p50=1).

> **Caveat:** `sph` omits radial compression (~0.1–0.3 Å in hard repulsive contact).
> At imaging heights (h≈3.7–4.7 Å) measured df diff ≈ 0.002–0.013 — systematic
> sharpening of feature outlines. Fine for training-data generation; use `fire`
> for quantitative work.

## Host API (`spammm/SPM/AFM.py`)

```
run_scan_contact_pme(..., relax_mode='fire'|'qn'|'sph', qn_cap=0, qn_conv=1.0)
```

- `relax_mode` — `fire` (default, untouched kernels) | `qn` | `sph`; non-`fire`
  requires `core_backend='local'` (raises otherwise; missing kernel raises RuntimeError).
- `qn_cap` — optional two-pass: pass-1 iter cap; flagged (non-converged) pixel columns
  re-scanned in pass 2 with full budget, resuming at first flagged slice `iz0`
  (per-slice PP pos stored in `scan_disps`/`out_pp`). **Measured NOT faster**
  (packing + IO overhead ≥ tail saving); default `0` disables it. With `qn_cap=0`
  `iz_start`/`pix_map`/`out_pp` are passed as NULL — no extra traffic.
- `qn_conv` — multiplies `F2CONV`. `1e4` (|f|<0.01 eV/Å) collapses the residual
  tail — **but leaves visible per-pixel grain in df** (tolerance-floor noise from
  the C1-discontinuous tricubic field). Keep `1.0` unless grain is acceptable.
- Iteration diagnostics in `self.last_relax_iters` (per px×slice; negative = flagged),
  `self.last_relax_hard` (pixel mask), `self.last_relax_flagz` (per-slice counts).
- Per-section wall profile under `verbosity>1` (`scan_prof[mode]` print).

## Numerical findings (why the speedup is "only" ~4×)

1. Iterations were already ~1 eval/cell for most pixels (warm start + Hooke).
   The solver win came from cutting the *tail*, and the remaining kernel time is
   the **eval floor**: ~1.2 evals/cell × 1.8M cells ≈ 15 ms (tricubic mesh gather
   + per-atom core loop, memory-latency bound).
2. Converged slices reuse the loop's force eval for `FEs` (skips the redundant
   final eval — this alone halved kernel time, 28→15 ms).
3. The convergence tolerance `F2CONV=1e-8` is near the spline-interpolation error
   floor (C1 curvature discontinuities at 1 Å cell boundaries): below |f|~0.01 eV/Å
   Newton just jitters between cell-adjacent minima — FIRE burned 50–100 iters
   on that same unreachable tolerance.
4. Host side ≈ 10 ms: FEs D2H 29 MB ≈ 2–6 ms PCIe + telemetry ~1.5 ms + postflight ~3 ms.

Remaining levers (not done): fewer z-slices per column (df needs only h±amp band),
cheaper eval (mesh/stencil tradeoff).

## Parity status

Measured vs FIRE (same fit, same geometry, NVIDIA GTX 1650, GPU-synced):

| mode | NCC(Fx) | max|ΔF| | max|ΔE| | iters p99/max |
|------|---------|----------|----------|--------------|
| `qn` strict | 0.998 | 0.013–0.024 eV/Å | 0.003–0.006 eV | 11–12 / 128 |
| `sph` strict | **0.998–0.9999** | **0.003–0.014** | 0.0007–0.003 | 8–20 / 128 |
| `*_conv1e4` | 0.99 (grain) | 0.018–0.024 | 0.004–0.005 | 1–2 / ~29 |

df-image diff at h=3.7: `sph` max|d|≈0.002–0.013 df-units, faint outlines only.
Bench harness + figures: `invPPAFM/export_invAFM/scripts/testplot_artifact_atlas.py
--bench-qn` → `invPPAFM/debug/artifact_atlas/bench_qn.png`, `bench_qn_map.png`.

## Files touched

- `kernels/contact_surface.cl` — added: `cs_pme_total_force_local`,
  `solve3x3_general`, `relaxStrokesTiltedContactPMELocalQN`,
  `relaxStrokesTiltedContactPMELocalSph`, `QN_BUDGET`/`NR_BUDGET` defines.
  FIRE kernels unchanged.
- `spammm/SPM/AFM.py` — `relax_mode`/`qn_cap`/`qn_conv` on
  `run_scan_contact_pme` + `_pme_scan_gpu`; two-pass buffers (`cpm_scan_iters`,
  `cpm_izstart`, `cpm_pixmap`); `scan_disps` reused as `out_pp`; verbosity>1
  `scan_prof` section timing.

## Open issues

- `sph` = `K_RAD→∞` limit — radial compression omitted (small deep-contact diff).
- Two-pass (`qn_cap>0`) is a net loss on GTX 1650 (kept for experimentation).
- `qn_conv>1` produces per-pixel grain — training-data volume only, if at all.
- Bucket backend has no QN/Sph variant (raises explicitly).
