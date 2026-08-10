---
name: afm-plotting-alignment
description: invPPAFM AFM data plotting and z-height alignment — coordinate conventions, Vpp z-selection, E(z) curves, atom overlays. Use whenever plotting Vpp/V_raw/df/rho_atom maps or z-profiles from invPPAFM samples. Prevents the recurring mistakes: wrong z-plane, wrong coordinate origin, missing mol_shift, wrong cmap/clim.
trigger:
  glob:
    - "**/testplot_*.py"
    - "**/validate.py"
    - "**/generate.py"
    - "**/tests/test_*.py"
---

## Why this skill exists

Every agent that touches invPPAFM plotting makes the same mistakes:
1. Picks a Vpp z-plane with zero contrast (middle of the field grid)
2. Forgets `mol_shift` when overlaying atoms
3. Measures z from the grid bottom instead of the molecule center
4. Uses the wrong cmap/clim for diverging data (DoG rho, Morse V)
5. Draws huge atom markers that cover the contrast

This skill documents the correct conventions ONCE. Follow it or the USER will be angry.

## Coordinate conventions (invPPAFM-specific)

### Storage order: `[z, y, x]`

All 3D arrays in `.npz` files and downstream code are stored as `[Nz, Ny, Nx]`
(Contract C1). SPAMMM internally uses `[x, y, z]`. The transpose happens in
`generate.py` at save time:

```python
# SPAMMM output (nx, ny, nz) → stored (nz, ny, nx)
df_zyx = np.ascontiguousarray(df.transpose(2, 1, 0))
V_zyx  = np.ascontiguousarray(V_raw.transpose(2, 1, 0))
FEpp   = np.ascontiguousarray(F_total.transpose(2, 1, 0, 3))  # [nz,ny,nx,4]
```

**When plotting with `imshow`:** `imshow(arr_2d, origin='lower', extent=[x0, x1, y0, y1])`
expects `arr_2d` in `[ny, nx]` order. For a z-slice `V_raw[iz]` which is `[ny, nx]`,
this works directly. For `extent`, use the **scan-grid** xs/ys, NOT the field-grid
fx/fy (they differ for real molecules).

### Two coordinate systems for real molecules

SPAMMM shifts the molecule into a "kernel-space" coordinate system via `mol_shift`.
The sample dict contains both:

| Key | System | Description |
|-----|--------|-------------|
| `atoms` | world | Original .xyz coordinates |
| `mol_shift` | — | SPAMMM's shift vector (added to world → kernel) |
| `xs`, `ys` | kernel | Scan-grid coordinates (post mol_shift) |
| `field_origin`, `field_step`, `field_shape` | kernel | Force-field grid in kernel space |
| `z_v` | kernel | Field-grid z coordinates (kernel space) |
| `mol_z` | world | Molecule top z in world coords (= `max(atoms[:,2])`) |

**To overlay atoms on a scan-grid image:**
```python
atoms_xy = atoms[:, :2].copy()
atoms_xy[:, 0] += mol_shift[0]  # world → kernel x
atoms_xy[:, 1] += mol_shift[1]  # world → kernel y
ax.scatter(atoms_xy[:, 0], atoms_xy[:, 1], ...)
```

**To overlay atoms on a field-grid image:**
```python
fx = field_origin[0] + np.arange(field_shape[0]) * field_step
fy = field_origin[1] + np.arange(field_shape[1]) * field_step
atoms_kx = atoms[:, 0] + mol_shift[0]
atoms_ky = atoms[:, 1] + mol_shift[1]
ix = ((atoms_kx - field_origin[0]) / field_step).astype(int)
iy = ((atoms_ky - field_origin[1]) / field_step).astype(int)
```

### z measured from molecule center

The molecule center in kernel space is at `z = mol_z + mol_shift[2]` (typically
`mol_shift[2] = 2.0`). The field grid starts at `z_v[0] = field_origin[2]` (typically 0).

**z above molecule center:**
```python
mol_center_z = float(s["mol_z"]) + float(s["mol_shift"][2])
z_above = z_v - mol_center_z  # z relative to molecule center [Å]
```

**NEVER** use raw `z_v` as "height above molecule" — it's kernel-space z from the
grid bottom. The molecule sits at `z_v ≈ 2.0`, not at `z_v = 0`.

## Vpp z-plane selection (the #1 mistake)

### The problem

Agents pick `iz_mid = len(z_v) // 2` or some arbitrary z. For pentacene with
180 z-slices (z_v = 0..17.5 Å), the middle is z=8.75 Å — far above the molecule,
zero contrast. The interesting physics is at z_above = 1.5–3.0 Å (Morse repulsive
wall and attractive well).

### Correct procedure

1. **Compute z_above** (z from molecule center, see above)
2. **Find the Morse minimum** on the "over atom" E(z) curve:
   ```python
   ez_atom = V_raw[:, iy_atom, ix_atom]
   attractive = z_above > 0
   iz_min = np.argmin(np.where(attractive, ez_atom, np.inf))
   E_min = ez_atom[iz_min]
   ```
3. **Find the zero crossing** (repulsive onset) — where E crosses 0 going from
   attractive (negative) to repulsive (positive) as z decreases:
   ```python
   for iz in range(len(z_v)-1, -1, -1):
       if z_above[iz] < 0: break
       if ez_atom[iz] < 0 and iz > 0 and ez_atom[iz-1] >= 0:
           iz_zero = iz  # just past zero crossing, entering repulsion
           break
   ```
4. **Plot the 2D Vpp at `iz_zero`** — this is where atoms first appear as
   repulsive features, with the best contrast between atom positions and
   inter-atomic space.

### Reference values (pentacene/PTCDA, Morse)

| Feature | z_above [Å] | E [eV] |
|---------|-------------|--------|
| Morse minimum | ~2.5 | ~-0.013 |
| Zero crossing (repulsive onset) | ~2.1 | 0 |
| Strong repulsion | ~1.5 | ~1.5 |
| Saturated (clipped at 100 eV) | ~0.5 | 100 |

## E(z) curve plotting

### Normalization (USER-mandated)

For E(z) Morse curves:
- `vmin = E_min` (the Morse minimum, a small negative number)
- `vmax = -2 * E_min` (twice the well depth, positive)
- This shows both the attractive well and the onset of repulsion without
  the curve running off to 100 eV (saturated/clipped)

For Fz(z) curves:
- `vmin = Fz_min * 1.2` (most negative attractive force, with 20% margin)
- `vmax = -2 * Fz_min` (twice the attractive depth, positive = repulsive onset)
- Same logic as E(z): show the well + onset of repulsion, not the full blowup

### Sample points

Always pick 3 points for E(z) curves:
1. **Over atom** — atom with max V at z ≈ Morse minimum (red cross)
2. **Inside ring** — centroid of all atoms (lime cross)
3. **Outside molecule** — corner of grid far from all atoms (cyan cross)

### Marking points on the 2D image (USER-mandated)

Two types of markers, never confuse them:

**Molecule atoms** — single-pixel dots, small, semi-transparent:
```python
ax.plot(fx_a[ix_atoms], fy_a[iy_atoms], '.', color='cyan', markersize=1,
        markeredgecolor='none', alpha=0.5)
```

**E(z) curve sample points** — small crosses with single-pixel line width:
```python
ax.plot(x, y, '+', color=color, markersize=5, markeredgewidth=0.5,
        markeredgecolor=color, markerfacecolor='none')
```

**NEVER** use large circles (`s=20`, `marker='o'`) — they cover the AFM contrast.
The USER has complained about this repeatedly. Crosses mark the 3 diagnostic
points; dots show all molecule atoms. Both must be tiny.

## Canonical plot: Vpp_curves (USER-approved 2026-08-08)

The plot `pentacene_Vpp_curves.png` / `PTCDA_Vpp_curves.png` in
`debug/testplot_t1_real_molecules/` is the **canonical style** for Vpp
z-profile analysis. Always plot Vpp this way. The USER explicitly approved
this layout, color scale, E(z) curves, and vlim/ylim.

### Layout: 2-panel side-by-side

```
+----------------------------+------------------------+
|  Left (1.2× width):        |  Right:                |
|  V_raw 2D map at z_zero    |  E(z) curves at 3 pts  |
|  RdBu_r, symmetric clim    |  vmin=Emin, vmax=-2Emin|
|  molecule atoms = dots     |  z from mol center      |
|  3 curve pts = crosses     |  axvline z_zero, z_min  |
+----------------------------+------------------------+
```

### Left panel — 2D V_raw at zero crossing

- `cmap='RdBu_r'` (diverging: blue=attractive, red=repulsive)
- `vmin=-vabs, vmax=vabs` where `vabs = min(max(|min|,|max|), -2*Emin*2)`
  (clip saturated pixels)
- `aspect='equal'`, `origin='lower'`
- `extent` = field-grid extent `[fx[0], fx[-1], fy[0], fy[-1]]`
- Title: `V_raw at z_above={z_zero:.2f}Å (zero crossing)\nrange [{min:.3f}, {max:.3f}] eV`
- Colorbar label: `E [eV]`
- Molecule atoms: `'.'` markers, `markersize=1`, `color='cyan'`, `alpha=0.5`
- 3 curve points: `'+'` markers, `markersize=5`, `markeredgewidth=0.5`

### Right panel — E(z) Morse curves

- 3 curves (over atom=red, inside ring=lime, outside mol=cyan), `linewidth=1.5`
- `axvline(z_zero, color='gray', linestyle='--')` — Vpp image z
- `axvline(z_min, color='orange', linestyle=':')` — Morse minimum z
- `axhline(0, color='black', linewidth=0.5, alpha=0.5)`
- `xlim = (-0.5, 6)` — focus on attractive + onset of repulsion
- `ylim = (E_min * 1.2, -2 * E_min)` — USER-mandated normalization
- xlabel: `z above molecule center [Å]`
- ylabel: `E [eV]`
- Title: `E(z) Morse curves — vmin={Emin:.4f}, vmax={-2*Emin:.4f} eV`
- Legend: `fontsize=8, loc='upper right'`
- Grid: `alpha=0.3`

### Suptitle

`{name}: Vpp z-profile (z from mol center, mol at z_kernel={mol_center_z:.1f}Å)`

### Reference implementation

`tests/testplot_t1_real_molecules.py` → `plot_Vpp_curves(name, s)`.
This is the SSOT — copy its style, do not reinvent.

## Colormap and clim rules

### V_raw / Vpp (Morse energy)

- **2D map at zero crossing**: `cmap='RdBu_r'`, symmetric `vmin=-vabs, vmax=vabs`
  where `vabs = max(|min|, |max|)` (clipped to avoid saturation dominance)
- **2D map in repulsive region**: `cmap='hot'`, `vmin=0, vmax=percentile(99.5)`
- **1D E(z) curves**: `vmin=E_min, vmax=-2*E_min` (see above)

### rho_atom (DoG Mexican hat)

- `cmap='RdBu_r'` (diverging: blue=negative exclusion, red=positive core)
- `vmin=-vabs, vmax=vabs` where `vabs = max(|min|, |max|)`

### df (frequency shift)

- `cmap='gray'` or `cmap='RdBu'` (if showing both attractive and repulsive)
- Per-image `vmin=percentile(1), vmax=percentile(99)` for gray
- Symmetric `vmin=-vabs, vmax=vabs` for RdBu

### NEVER

- `cmap='jet'` — rainbow is forbidden for scientific data
- `aspect='auto'` — always `aspect='equal'` (1 Å x = 1 Å y)
- Shared clim across different z-planes — each panel gets its own clim

## SSOT plotting helpers

### invPPAFM (this repo)

The invPPAFM repo uses SPAMMM's plotting SSOT helpers per AGENTS.md:
- `spammm.SPM.AFM_utils.imshow_afm` — single AFM map on an Axes
- `spammm.SPM.AFM_utils.plot_afm_height_panel` — row of maps at selected heights
- `spammm.plotUtils.plot_2d_scalar` — generic 2D scalar with atom overlay

Import lazily via `validate._ensure_spammm()` to avoid GPU context contention.

**However**, for `testplot_*.py` visual test scripts, direct `imshow` is acceptable
when the plot is a custom diagnostic (E(z) curves + Vpp image side-by-side) that
doesn't match any SSOT helper signature. In that case, follow the conventions in
this skill (extent, cmap, clim, atom markers).

### SPAMMM (sibling repo)

See skill:`afm-plotting` (SPAMMM) for the full AFM plotting API. The SPAMMM skills
cover `imshow_afm`, `plot_afm_height_panel`, `plot_afm_z_profiles`, etc. Those are
for SPAMMM-internal code. invPPAFM test scripts may use them when the array shapes
match (they expect `(nx, ny)` — transpose from `[ny, nx]`).

## Scan height convention (h above molecule top)

See skill:`afm-z-heights` for the full convention. Summary:

- `h` = probe-particle z **above molecule top atom** (`mol_top_z = max(atoms[:,2])`)
- `tip_z = mol_top_z + h + bond_length`
- **Bond-resolved AFM contrast at h ≈ 3.0 Å** for real molecules (pentacene, PTCDA)
  with Morse VdW params (R0 ≈ 3-4 Å). This is the standard scan height.
- Morse minimum at z ≈ 1.9 Å for pentacene C → scan h=[2.5, 4.0] covers the well
  and the repulsive onset. NEVER scan at h < 2.0 for real molecules — too close,
  the PP relaxes into the molecule and df saturates.
- Soft sphere (pure repulsive) → can scan far → `config.Z_DF = [3.0, 5.0]` is fine
- **ALWAYS verify** `df.max() > 0` (repulsive features exist) before showing images

## Force-field and scan-grid margins (CRITICAL)

**The #2 mistake (after z-height): insufficient margin around the molecule.**

The force-field grid and scan grid MUST extend ≥4 Å beyond the molecule bbox on
all sides. This is not optional — it is physics:

1. **Probe-particle radius** ≈ 1.5 Å (CO tip). The PP samples the force field at
   its center, which is offset from the scan pixel by the lateral relaxation.
2. **Lateral PP relaxation** can be up to ~1 Å near atoms (K_LAT=0.0312 eV/Å²).
3. **Force-field edge artifacts**: if the PP samples near the FF grid edge, the
   interpolated force is wrong (zero outside grid → spurious attractive pull).
4. **Scan-grid edge artifacts**: if the scan grid extends to the FF grid edge,
   the PP at scan-edge pixels samples outside the valid FF region.

### Required margins

| Parameter | Min value | Reason |
|-----------|-----------|--------|
| `margin` (FF grid) | 4.0 Å | FF must cover molecule + scan + PP radius + relaxation |
| `scan_margin` (scan grid) | 4.0 Å | Scan pixels must be ≥4 Å from molecule edge |
| `z_top` (FF grid z above mol) | 16.0 Å | Cover scan range + PP relaxation + Coulomb tail |

### Verification

After generating a sample, verify:
```python
# FF grid covers scan grid + margin
ff_x_lo = field_origin[0]
ff_x_hi = field_origin[0] + field_shape[0] * field_step
scan_x_lo, scan_x_hi = xs[0], xs[-1]
assert ff_x_lo <= scan_x_lo - 2.0, f"FF grid too close to scan edge in x_lo"
assert ff_x_hi >= scan_x_hi + 2.0, f"FF grid too close to scan edge in x_hi"
# Same for y
```

### Boundary effect symptoms

If you see in df images:
- Bright/dark stripes at the image edges → scan grid too close to FF edge
- Asymmetric contrast near borders → FF grid not centered on molecule
- Sudden force jumps at edges → PP sampling outside FF grid (zero force)

→ Increase `margin` and `scan_margin` to 4.0 Å minimum.

## Curve style: reference vs model (MANDATORY)

When plotting E(z), Fz(z), df(z) or any 1D curve comparing a **reference**
(full 3D GridFF / brute) against a **model** (2.5D contact surface / kriging /
fitted approximation), use this style so the USER can read it at a glance:

| Role | Line style | Line width | Color |
|------|-----------|------------|-------|
| **Reference** (3D GridFF, brute, truth) | thick dotted `ls=':'` | `lw=1.5` | one color per sample point |
| **Model** (2.5D contact, approximation) | thin full `ls='-'` | `lw=0.5` | **same color** as its reference |

```python
# CORRECT — same color per point, ref dotted thick, model thin full
ax.plot(h, E_ref[ix, iy, :], ls=':', lw=1.5, color=col, label=f'{label} 3D ref')
ax.plot(h, E_mod[ix, iy, :], ls='-', lw=0.5, color=col, label=f'{label} 2.5D')
```

**Why:** The USER identifies the sample point by color (matches the marker on
the 2D map), then sees ref vs model overlap. Thick dotted = "ground truth",
thin full = "approximation". If they coincide, the thin line sits on top of
the dotted one → visually clear that parity is good.

### Combined 2D map + 1D curves (bijection plot)

When comparing a force field / potential representation, show **both** the 2D
spatial map AND the 1D z-curves in one figure, with explicit bijection:

1. **Left panel:** 2D map (Fz or E at a chosen z-slice) with sample points
   marked as colored crosses (`'+'`, `ms=12`, `mew=2`) + text labels.
2. **Right panels:** E(z) and Fz(z) curves at those exact points, same colors.
3. **Vertical dashed line** on the 1D curves at the z-height of the 2D map
   (`axvline(h_map, c='gray', ls='--')`) — shows which slice the map is from.
4. **Fit z-range** shaded (`axvspan(FIT_Z_LO, FIT_Z_HI, color='0.9')`).

```python
# Layout: [2D map | E(z) | Fz(z)]
fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1.2, 1.2]})
ax_map, axE, axFz = axes
# 2D map with marked points
for ix, iy, label, col in sample_pts:
    ax_map.plot(scan_xs[ix], scan_ys[iy], '+', color=col, ms=12, mew=2, zorder=10)
# 1D curves — ref dotted thick, model thin full, same color
for ix, iy, label, col in sample_pts:
    axE.plot(h, E_ref[ix, iy, :], ls=':', lw=1.5, color=col)
    axE.plot(h, E_mod[ix, iy, :], ls='-', lw=0.5, color=col)
axE.axvline(h_map, c='gray', ls='--')  # bijection: which slice the map shows
```

**Never** show 1D curves without the 2D map showing WHERE the curves are
sampled. The USER needs the bijection to interpret the curves.

## Checklist before showing any AFM plot

- [ ] z-plane selected at zero crossing or Morse minimum, NOT grid middle
- [ ] z labeled as "z above molecule center [Å]", NOT raw kernel z
- [ ] Atoms shifted by `mol_shift` before overlay
- [ ] Molecule atoms: `'.'` markers, `markersize=1`, `color='cyan'`, `alpha=0.5`
- [ ] E(z) curve points: `'+'` markers, `markersize=5`, `markeredgewidth=0.5`
- [ ] Vpp_curves plot follows canonical style (2-panel, RdBu_r, Emin/-2Emin ylim)
- [ ] `aspect='equal'` on all spatial maps
- [ ] Diverging data (rho_atom, V at zero crossing) uses `RdBu_r` with symmetric clim
- [ ] E(z) curves normalized: `vmin=E_min, vmax=-2*E_min`
- [ ] Fz(z) curves normalized: `vmin=Fz_min*1.2, vmax=-2*Fz_min`
- [ ] `extent` matches the grid that produced the data (scan-grid xs/ys for df, field-grid fx/fy for V_raw)
- [ ] `df.max() > 0` verified (repulsive features exist)
- [ ] 1D curves: ref=thick dotted `ls=':' lw=1.5`, model=thin full `ls='-' lw=0.5`, same color per point
- [ ] 2D map + 1D curves shown together with bijection (colored crosses on map, same colors on curves, `axvline` for map z-height)
- [ ] Scan height h ≈ 3.0 Å for real molecules (bond-resolved contrast)
- [ ] FF margin ≥ 4.0 Å, scan margin ≥ 4.0 Å (no boundary effects)
- [ ] Molecule name + atom count in plot title
- [ ] Plot title includes scan height and margin values

## Related

- skill:`afm-z-heights` (`.devin/skills/afm-z-heights/`) — scan height convention
- skill:`afm-plotting` (SPAMMM `.devin/skills/afm-plotting/`) — SPAMMM AFM plotting SSOT
- skill:`centralized-plotting` (SPAMMM) — generic 2D scalar plotting
- `AGENTS.md` § "AFM E/Fz/df plots (SSOT)" — repo-level plotting rules
- `config.py` — grid constants (FIELD_ORIGIN, Z_DF, Z_V, Z_A)
- `physics_utils.py` — `encode_V` / `decode_V` (V normalization gauge)
