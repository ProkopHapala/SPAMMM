# Lateral AFM — Arbitrary Oscillation Direction

## Goal

Support lateral AFM where the tip oscillates not only vertically (z) but also
horizontally (x, y) or in an arbitrary direction (nx, ny, nz). The scan grid
stays (x, y). The stiffness model (3D anisotropic + radial) stays as-is. The
force field (gridFF / FDBM) stays as-is. Only the **oscillation response direction** and
**df computation** change; the z-approach direction does not.

## Correct Coordinate Model (2026-08-06, awaiting USER confirmation)

- The acquired relaxed force volume is always `(x,y,z,4)` and the probe approaches along z.
- `--h-min`, `--h-max`, every plotted column, and array axis 2 remain z heights.
- Oscillation is an independent unit vector `n`; `F_n = F·n` and `df = -n·∇F_n`.
- Finite-amplitude df samples the xyz volume along `n`. Pure lateral amplitude pads x/y,
  not z; only `amp*abs(n_z)` affects z padding and closest-approach alignment.
- `dTip` in `scan_fdbm` remains `(0,0,-dh)`. Replacing it with `-dh*n` destroys the
  z stack and was the cause of the repeated “height” contrast.

Automated and NVIDIA verification passes, and the corrected PTCDA artifact is
`debug/lateral_afm/lat_x_zslices_corrected/compare_per_image.png`. Per repository
policy, this task remains **unverified** until the USER reviews and confirms it.

## Design Decisions (confirmed with USER)

1. **Scan grid and approach**: Keep the `(x,y)` lateral grid and z height stack.
   Oscillation direction is independent of both.
2. **df computation**: Project force onto oscillation direction
   `F_n = F · n`, compute `df = -d(F_n)/ds` where `s` is along `n`.
   Use amplitude averaging (generalized `compute_df_amp`).
3. **Stiffness**: Stay as-is — `(kx, ky, kz, k_rad)` independent of oscillation
   direction. No rotation of stiffness tensor.
4. **Modularity**: Oscillation direction is a parameter. Swapping between
   vertical (0,0,1) and lateral (1,0,0) / (0,1,0) / arbitrary should be trivial.

## Initial Architecture Analysis (partly superseded)

### Already 3D / direction-agnostic (NO changes needed)

- **Force field grid** (`gridFF` / FDBM): 3D `(Fx, Fy, Fz, E)` — fully general
- **`tipForce()` kernel** (`AFM.cl:53`): 3D harmonic + radial spring
- **`stiffness` parameter**: `float4 (kx, ky, kz, k_rad)` — already 3D anisotropic
- **FIRE integrator** (`AFM.cl:136`): fully 3D
- **`interpFE()`** (`AFM.cl:100`): 3D trilinear — direction-agnostic
- **`dTip` parameter in kernels**: controls acquisition progression; in `scan_fdbm`
  it must remain z-directed because output axis 2 is the z stack.
- **`relaxStrokes` kernel** (`AFM.cl:345`): accepts z-directed `dTip` and relaxes
  in 3D at each z point. **No kernel changes needed.**
- **`relaxStrokesTilted` kernel** (`AFM.cl:478`): same — `dTip = tipC.xyz * tipC.w`

### Historical hardcoded-z inventory (df needed generalization; acquisition did not)

| # | Location | What | Current | Target |
|---|----------|------|---------|--------|
| 1 | `AFM.py:504` | `dTip` in `scan_fdbm` | `(0, 0, -dh)` | **Keep:** z approach is independent of oscillation |
| 2 | `AFM.py:488` | `z_start` in `scan_fdbm` | `max(probe_heights) + mol_z + bond_length` | **Keep:** heights are z |
| 3 | `AFM.py:512` | `pts[:, 2] = z_start` | z-coordinate fixed | **Keep:** each scan begins at the highest z |
| 4 | `AFM.py:898` | `tipC = [0,0,1,dtip]` in `run_scan` | z-only | `tipC = n * dtip` |
| 5 | `AFM.py:1235` | `dTip = (0,0,dtip)` in `get_raw_FE` | z-only | `n * dtip` |
| 6 | `AFM.py:1824-1865` | `compute_df` / `compute_df_amp` | `Fz`, `dFz/dz` | `F·n`, `d(F·n)/ds` |
| 7 | `AFM_utils.py:3174` | `compose_and_relax_total` | hardcodes `FEs[:,:,:,2]` | project onto `n` |
| 8 | `AFM_utils.py:644-664` | `afm_df_height_stacks` | full amp shifts z | shift z only by `amp*abs(n_z)` |
| 9 | `ModularPipeline.py:204-216` | `_init_geometry_and_grids` | x/y scan + z heights | **Keep:** authoritative acquisition coordinates |
| 10 | `AFM.py:540` | output reshape | `[:, :, ::-1, :]` | **Keep:** flip descending approach to ascending z |

## Historical Implementation Plan (superseded by the coordinate model above)

### Phase 1: Core — `scan_fdbm` generalization (small, high impact)

**Goal**: Make `scan_fdbm()` accept an `osc_dir` parameter.

- Add `osc_dir=(0,0,1)` parameter to `scan_fdbm()` (default = vertical = backward compat)
- Generalize `dTip`: `dTip = np.array([-dh*nx, -dh*ny, -dh*nz, 0.])`
- Generalize starting position: instead of `z_start` only, compute start
  position along `osc_dir` (analogous to current z_start but projected)
- Generalize `pts` construction: set the coordinate along `osc_dir` to the
  start position, keep other two coords from scan grid
- Output reshape stays `(nx_s, ny_s, n_stroke, 4)` — the stroke dimension is
  just no longer necessarily z

**Files**: `AFM.py` (`scan_fdbm` method)

### Phase 2: df computation generalization (small, high impact)

**Goal**: Generalize `compute_df_amp` and `compute_df` for arbitrary direction.

- Add `compute_df_amp_dir(FEs, dpos, osc_dir, amp)`:
  - Project force: `F_n = FEs[...,0]*nx + FEs[...,1]*ny + FEs[...,2]*nz`
  - Gradient along oscillation direction: `d(F_n)/ds` where `ds = |dTip|`
  - Amplitude averaging via Gauss-Chebyshev (same algorithm, different axis)
- Or generalize existing `compute_df_amp` with optional `axis` / `osc_dir` param
- Keep `compute_df_amp(Fz, dz, amp)` as backward-compat wrapper

**Files**: `AFM.py` (`compute_df_amp`, `compute_df`)

### Phase 3: Pipeline integration (moderate)

**Goal**: Thread `osc_dir` through `compose_and_relax_total` and `ModularPipeline`.

- Add `osc_dir` parameter to `compose_and_relax_total()`
- Pass to `scan_fdbm()` and to generalized `compute_df_amp`
- Add `osc_dir` to `ModularAFMPipeline.__init__` / `stage4_relax()`
- Generalize `afm_df_height_stacks` to produce stroke positions along `osc_dir`

**Files**: `AFM_utils.py`, `ModularPipeline.py`

### Phase 4: `run_scan` / `get_raw_FE` generalization (moderate)

**Goal**: Generalize the LJ/Morse scan path (`run_scan`, `get_raw_FE`) too.

- `tipC` currently `(0, 0, 1, dtip)` → generalize to `(nx, ny, nz, dtip)`
- `dTip = tipC.xyz * tipC.w` already works if `tipC.xyz` is the oscillation dir
- Scan grid auto-computation may need adjustment for non-z stroke

**Files**: `AFM.py` (`run_scan`, `get_raw_FE`, `_scan_grid_auto`)

### Phase 5: GUI / CLI integration (moderate, can be deferred)

**Goal**: Expose oscillation direction in GUI and CLI.

- Add oscillation direction selector to `AFMExtension.py`
- Add CLI flag for oscillation direction
- Update plotting to label the correct axis

**Files**: `AFMExtension.py`, CLI scripts, `AFM_utils.py` (plotting)

### Phase 6: Testing / validation

- Unit test: `scan_fdbm(osc_dir=(0,0,1))` matches current behavior exactly
- Lateral test: `scan_fdbm(osc_dir=(1,0,0))` produces sensible Fx/dFx results
- Parity: `compute_df_amp_dir` with `osc_dir=(0,0,1)` matches `compute_df_amp`
- Visual: lateral AFM images for a known molecule

## Difficulty Summary

| Phase | Difficulty | Lines changed (est.) | Kernel changes? |
|-------|-----------|---------------------|-----------------|
| 1: scan_fdbm | Easy-Moderate | ~20-30 | No |
| 2: compute_df_amp | Easy | ~15-25 | No |
| 3: Pipeline | Moderate | ~20-30 | No |
| 4: run_scan etc. | Moderate | ~15-20 | No |
| 5: GUI/CLI | Moderate | ~30-50 | No |
| 6: Testing | Easy-Moderate | ~50-80 (new) | No |

**Overall: Moderate. No OpenCL kernel changes needed.** The architecture is
already 3D — `dTip` and `tipForce` are direction-agnostic. The work is
purely Python-side parameterization and df computation generalization.

## Historical Key Insight (incorrect)

The `relaxStrokes` kernel already does 3D relaxation at each stroke point.
The `dTip` float4 already controls the step direction. The only reason the
code is "vertical-only" is that the Python side always sets `dTip = (0,0,-dh)`
and `compute_df_amp` always uses `Fz/dz`. Generalizing these two Python-side
hardcodings is the entire core of the work.

This confused acquisition progression with cantilever oscillation. `dTip` must
stay vertical so output axis 2 remains z; only df projection/differentiation
uses the oscillation vector.

## Status (automated verification only; USER confirmation pending)

- [~] Phase 1: `scan_fdbm` keeps z-directed `dTip`; `osc_dir` no longer changes acquisition geometry
- [~] Phase 2: `compute_df_dir` / `compute_df_amp_dir` use `n·∇(F·n)` on `(x,y,z)` volumes
- [~] Phase 3: pipeline threads `osc_dir` only into df and preserves z heights
- [~] Phase 4: `run_scan` / `get_raw_FE` keep the physical tip frame and z approach
- [~] Phase 5: CLI exposes `--osc-dir`; `--base-pos` is only a constant scan offset
- [ ] Phase 5b: GUI integration — NOT done (AFMExtension.py does not yet expose osc_dir selector)
- [~] Phase 6: analytical, vertical-parity, NVIDIA acquisition, and PTCDA end-to-end checks pass
- [ ] **BUG: Lateral images all look identical across heights** — see Known Bugs below

---

## Historical Failed Implementation Notes (superseded; retained for audit trail)

### What was implemented (all Python-side, no kernel changes)

#### Phase 1: `scan_fdbm()` — `AFM.py:~496`
- Added `osc_dir=(0,0,1)` and `base_pos=(0,0,0)` parameters
- `dTip` generalized from `(0,0,-dh,0)` to `(-dh*osc_n[0], -dh*osc_n[1], -dh*osc_n[2], 0)`
- `start_along` computation:
  - z-oscillation: `max(probe_heights) + mol_z + bond_length` (original behavior)
  - non-z: `max(probe_heights)` only (bond_length offset is in z via dpos0, not along osc_dir)
- `pts` construction: `pts[:,k] = scan_grid_coord + start_along * osc_n[k] + base[k]`
- Output reshape unchanged: `(nx_s, ny_s, n_stroke, 4)` with `[:, :, ::-1, :]` flip

#### Phase 2: df computation — `AFM.py:~1886`
- `compute_df_dir(FEs, ds, osc_dir=(0,0,1))`: projects `F_n = F·n`, computes `-d(F_n)/ds` via `np.gradient` along axis 2
- `compute_df_amp_dir(FEs, ds, osc_dir, amp)`: same + Gauss-Chebyshev amplitude averaging
- Original `compute_df` / `compute_df_amp` preserved for backward compat
- **Verified**: `compute_df_dir` with `(0,0,1)` matches `compute_df` exactly (diff=0)
- **Verified**: `compute_df_amp_dir` with `(0,0,1)` matches `compute_df_amp` exactly (diff=0)

#### Phase 3: Pipeline threading
- `compose_and_relax_total()` (`AFM_utils.py:~3089`): accepts `osc_dir`, `base_pos`; passes to `scan_fdbm`; uses `compute_df_dir` for non-z
- `ModularPipeline.stage4_relax()` (`ModularPipeline.py:~702`): accepts `osc_dir`, `base_pos`; passes through

#### Phase 4: `run_scan` / `get_raw_FE` — `AFM.py`
- `run_scan(~881)`: `tipC[:3] = osc_n`, `tipC[3] = dtip`
- `get_raw_FE(~1328)`: `dTip = osc_n * dtip`

#### Phase 5a: CLI — `run_spm.py`
- Added `_parse_vec3()` helper (~line 52)
- Added `--osc-dir "x,y,z"` (default `"0,0,1"`) and `--base-pos "x,y,z"` (default `"0,0,0"`) to `_add_common_afm_args`
- `cmd_afm()` (~line 229): parses `osc_dir`/`base_pos`, prints if non-vertical, passes to `run_fdbm_pp_from_density`
- `run_fdbm_pp_from_density()` (`AFM_utils.py:~3284`): accepts `osc_dir`, `base_pos`; passes to `scan_fdbm`; uses `compute_df_amp_dir` for non-z
- Scan grid shift: `AFM_utils.py:~3370` shifts scan grid by `-midpoint(h_scan)` along osc_dir to center probe on molecule

### Known Bugs / Issues

#### BUG 1 (CRITICAL): All height slices look identical in lateral AFM images

**Symptom**: When running lateral x-oscillation on PTCDA, the df/Fz images at
different heights (h=2.5 to 3.5 Å) all look the same. The force values barely
change across the stroke dimension.

**Likely cause**: The `relaxStrokes` kernel steps the tip by `dTip.xyz` each
iteration (line 391: `tipPos += dTip.xyz; pos += dTip.xyz`). For lateral
oscillation with `dTip = (-dh, 0, 0)`, the tip moves in x but the probe stays
at constant z (from `dpos0`). The force field at z=3.0 (probe height) may be
nearly uniform in x at the scan positions, producing identical forces.

**But more likely**: The probe is sampling the force field at z=3.0 Å, which is
far from the molecule (mol at z=0). The force field at z=3.0 is weak and smooth,
so moving in x by ±1 Å barely changes the force. In vertical AFM, moving from
z=2.5 to z=3.5 crosses the steep repulsive wall — that's why heights differ.
In lateral AFM at fixed z=3.0, the x-gradient is much weaker.

**To investigate**:
1. Check if `FEs[:,:,0,0]` (Fx at first stroke point) differs from `FEs[:,:,18,0]` (Fx at last)
2. Try lower z (e.g. `base_pos=(0,0,4.0)` → probe at z=1.0) where forces are stronger
3. Try wider oscillation range (e.g. `--h-min 0.5 --h-max 5.5`) to see if extremes differ
4. Print per-slice Fx statistics to verify the kernel actually steps in x

**Important**: The `start_along` fix (removing `mol_z + bond_length` for non-z
oscillation) was applied in `AFM.py:~508`. Before this fix, the probe was ~3 Å
too far in x (into vacuum). After the fix, geometry is:
- `start_along = max(probe_heights) = 3.9` (for h_scan=[2.1...3.9])
- Tip apex: `x = scan_x + 3.9, z = base_pos[2] = 6.0`
- Probe: `x = scan_x + 3.9, z = 6.0 - 3.0 = 3.0` (bond_length below tip in z)
- Steps: x goes from `scan_x + 3.9` to `scan_x + 2.1` (dh=0.1, 19 steps)

#### BUG 2 (MINOR): Scan grid may go out of bounds for large scan_margin

**Symptom**: With `scan_margin > 3.0` and lateral oscillation, the probe can
sample outside the force field grid. The `interpFE` kernel uses
`read_imagef` with `CLK_ADDRESS_CLAMP_TO_EDGE`, so out-of-bounds returns the
edge value — producing flat/constant regions.

**Fix applied**: Scan grid is shifted by `-midpoint(h_scan)` along osc_dir
(`AFM_utils.py:~3374`). With default `scan_margin=2.0`, probe stays in bounds.
With `scan_margin >= 4.0`, may overflow. Use `scan_margin=2.0` (default) for now.

### Commands to Reproduce

#### Vertical AFM (backward compat, works correctly)
```bash
cd /home/prokophapala/git/SPAMMM
python3 run_spm.py afm \
    --xyz data/xyz/PTCDA.xyz --basis 3ob-3-1 --projection prolonged \
    --outdir debug/lateral_afm/vertical \
    --plots compare,stage \
    --h-min 3.7 --h-max 4.7 --h-step 0.1 --amp 1.0 \
    --K-LAT 0.5 --K-RAD 20.0 --bond-length 3.0
```

#### Lateral x-oscillation (BUG: images all look same across heights)
```bash
cd /home/prokophapala/git/SPAMMM
python3 run_spm.py afm \
    --xyz data/xyz/PTCDA.xyz --basis 3ob-3-1 --projection prolonged \
    --outdir debug/lateral_afm/lat_x_d3_v3 \
    --plots compare,stage \
    --h-min 2.5 --h-max 3.5 --h-step 0.1 --amp 0.4 \
    --K-LAT 0.5 --K-RAD 20.0 --bond-length 3.0 \
    --osc-dir 1,0,0 --base-pos 0,0,6.0
```
- `base_pos=(0,0,6.0)` → probe at z = 6.0 - 3.0 = 3.0 Å above molecule
- `h_min/h_max` = oscillation range along x: [2.5, 3.5] (probe x = scan_x + [2.1...3.9])
- `amp=0.4` = small oscillation amplitude

#### Lateral y-oscillation
```bash
python3 run_spm.py afm \
    --xyz data/xyz/PTCDA.xyz --basis 3ob-3-1 --projection prolonged \
    --outdir debug/lateral_afm/lat_y_d3 \
    --plots compare --h-min 2.5 --h-max 3.5 --h-step 0.1 --amp 0.4 \
    --K-LAT 0.5 --K-RAD 20.0 --bond-length 3.0 \
    --osc-dir 0,1,0 --base-pos 0,0,6.0
```

#### Unit tests (all pass)
```bash
cd /home/prokophapala/git/SPAMMM
python3 -c "
import numpy as np
from spammm.SPM import AFM as afm

# Test 1: compute_df_dir with osc_dir=(0,0,1) matches compute_df
np.random.seed(42)
Fz = np.random.randn(4, 5, 6).astype(np.float32)
FEs = np.zeros((4, 5, 6, 4), dtype=np.float32)
FEs[..., 2] = Fz
ds = 0.1
df_old = afm.compute_df(Fz, ds)
df_new = afm.compute_df_dir(FEs, ds, osc_dir=(0., 0., 1.))
assert np.allclose(df_old, df_new, atol=1e-6), 'MISMATCH!'

# Test 2: compute_df_amp_dir with osc_dir=(0,0,1) matches compute_df_amp
df_amp_old = afm.compute_df_amp(Fz, ds, amp=1.0)
df_amp_new = afm.compute_df_amp_dir(FEs, ds, osc_dir=(0., 0., 1.), amp=1.0)
assert np.allclose(df_amp_old, df_amp_new, atol=1e-6), 'MISMATCH!'

# Test 3: compute_df_dir with osc_dir=(1,0,0) uses Fx
FEs2 = np.zeros((4, 5, 6, 4), dtype=np.float32)
FEs2[..., 0] = Fz
df_x = afm.compute_df_dir(FEs2, ds, osc_dir=(1., 0., 0.))
df_x_ref = -np.gradient(Fz, abs(ds), axis=2)
assert np.allclose(df_x, df_x_ref, atol=1e-6), 'MISMATCH!'

print('All df computation tests PASSED')
"
```

### Output Artifacts

- `debug/lateral_afm/vertical/compare_per_image.png` — vertical AFM (correct, varies with height)
- `debug/lateral_afm/lat_x_d3_v3/compare_per_image.png` — lateral x (BUG: all heights look same)
- `debug/lateral_afm/lat_y_d3/compare_per_image.png` — lateral y (same BUG)
- `debug/lateral_afm/lat_x_d3_v3/stage_prolonged.png` — FDBM stage diagnostic

### Next Steps for Debugging BUG 1

1. **Verify the kernel actually steps in x**: Add a debug print in
   `scan_fdbm` to dump `dTip` and the first/last `pts` positions. Confirm
   `dTip = (-0.1, 0, 0, 0)` and pts x-coords differ by 1.8 Å across the stroke.

2. **Check FEs variation across stroke**: After `scan_fdbm`, print
   `FEs[center_x, center_y, :, 0]` (Fx across stroke at molecule center).
   If all values are the same, the kernel is not stepping or the force field
   is uniform at that z.

3. **Try lower z**: Run with `--base-pos 0,0,4.0` (probe at z=1.0) where the
   force field has stronger x-gradients. If images vary, the issue is just
   that z=3.0 is too far for lateral contrast.

4. **Try wider oscillation range**: `--h-min 0.0 --h-max 6.0` to see if
   extremes of the stroke produce different forces.

5. **Check if the issue is in df computation**: Print `FEs[..., 0]` (Fx) at
   two different stroke indices and compare. If Fx varies but df doesn't,
   the bug is in `compute_df_amp_dir`.

6. **Consider physics**: At z=3.0 Å above a flat molecule, the force field
   is mostly vertical (Fz dominates). Fx is weak and slowly varying. The
   x-gradient of Fx at z=3.0 may be nearly zero, producing flat df. This
   would be correct physics, not a bug. To get strong lateral contrast,
   the probe needs to be closer (z~1.5-2.0 Å) or the molecule needs 3D
   structure (non-planar).

### Files Modified

| File | Lines (approx) | Changes |
|------|-----------------|---------|
| `spammm/SPM/AFM.py` | ~496-512 | `scan_fdbm`: `osc_dir`, `base_pos`, `dTip`, `start_along` |
| `spammm/SPM/AFM.py` | ~881-921 | `run_scan`: `osc_dir`, `tipC` direction |
| `spammm/SPM/AFM.py` | ~1328-1367 | `get_raw_FE`: `osc_dir`, `dTip` direction |
| `spammm/SPM/AFM.py` | ~1886-1933 | `compute_df_dir`, `compute_df_amp_dir` (new functions) |
| `spammm/SPM/AFM_utils.py` | ~3089-3115 | `compose_and_relax_total`: `osc_dir`, `base_pos` |
| `spammm/SPM/AFM_utils.py` | ~3284-3292 | `run_fdbm_pp_from_density`: `osc_dir`, `base_pos` params |
| `spammm/SPM/AFM_utils.py` | ~3370-3382 | Scan grid shift for lateral oscillation |
| `spammm/SPM/AFM_utils.py` | ~3392-3396 | df computation: conditional `compute_df_amp_dir` |
| `spammm/SPM/ModularPipeline.py` | ~702-739 | `stage4_relax`: `osc_dir`, `base_pos` |
| `run_spm.py` | ~52-59 | `_parse_vec3` helper |
| `run_spm.py` | ~96-101 | `--osc-dir`, `--base-pos` CLI args |
| `run_spm.py` | ~229-249 | `cmd_afm`: parse + pass `osc_dir`, `base_pos` |

---

## Current Implementation and Verification (2026-08-06)

### Data flow

1. `scan_fdbm` acquires relaxed `(Fx,Fy,Fz,E)` on `(scan_x,scan_y,z_height)`.
2. `compute_df_dir` projects `F_n=F·n`, computes the xyz gradient, then contracts `n·∇F_n`.
3. `compute_df_amp_dir` samples along the finite-amplitude vector in xyz index space.
4. `run_fdbm_pp_from_density` pads x/y by `ceil(amp*abs(n_xy)/step)`, pads z through
   `afm_df_height_stacks(..., osc_dir=n)`, computes df, and crops x/y back to the requested image.
5. Plot columns remain z. Pure lateral oscillation compares Fz at the same z; arbitrary
   oscillation uses a z alignment of `amp*abs(n_z)`.

### Automatic checks

| Check | Result |
|---|---|
| Analytical `Fx=x*z`, `n=x` | `df=-z` on retained z slices |
| `Fx=z²`, `n=x` | lateral df exactly zero |
| `n=z` backward parity | `compute_df_dir` and amplitude version match established vertical functions |
| `scan_fdbm(n=x)` vs `scan_fdbm(n=z)` | byte-identical xyz volumes on NVIDIA GTX 1650 |
| AFM regression module | 16 passed |
| PTCDA `n=x`, z=3.7…4.7 Å | first↔last df RMS delta `1.634e-3`, relative `0.595`, correlation `0.9611` |

Review artifact: `debug/lateral_afm/lat_x_zslices_corrected/compare_per_image.png`.
Status remains **unverified pending USER confirmation**.
