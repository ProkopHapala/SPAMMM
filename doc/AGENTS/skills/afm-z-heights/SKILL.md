# AFM Scan Z-Height Convention

## Problem
Every time an agent sets up an AFM scan, it gets the z-heights wrong and produces images with only dark (attractive) features and no bright (repulsive) protrusions. This skill documents the correct convention ONCE so it never happens again.

## The Convention (SSOT)

AFM scan height `h` is **always** measured as the probe-particle (PP) z-position **above the molecule top atom**, NOT in world coordinates.

```
h = PP_z - mol_top_z     (mol_top_z = max(atoms[:, 2]))
PP_z = mol_top_z + h
tip_z = PP_z + bond_length    (PP hangs below tip by bond_length)
```

### Scan setup procedure

1. **Find molecule top**: `mol_top_z = float(atoms[:, 2].max())`
2. **Choose h range** (see table below): `h_Fz = np.linspace(h_min, h_max, n_h)`
3. **Tip start z**: `z_tip_start = mol_top_z + h_max + bond_length`
4. **Step downward**: `dtip = -abs(h_step)` (negative = approaching)
5. **After scan**: PP z goes from `mol_top_z + h_max` (first step) down to `mol_top_z + h_min` (last step)
6. **Reverse Fz** to get low→high order: `Fz = Fz[:, :, ::-1]`
7. **df**: `compute_df(Fz, dz=h_step)`

### Reference: `generate_from_xyz` in `generate.py`

```python
mol_z = float(afm.mol_z)                          # SPAMMM shifts mol top to z≈0
h_Fz = (h_min + np.arange(n_h) * h_step)          # probe above mol
z_tip_start = mol_z + float(h_Fz[-1]) + bond_length
dtip = -abs(h_step)
scan_p0 = np.array([scan_x0, scan_y0, z_tip_start], dtype=np.float32)
FEs, _ = afm.run_scan(nxy=nxy, nz=n_h, dtip=dtip, scan_p0=scan_p0, ...)
Fz = FEs[:, :, ::-1, 2]   # reverse high→low to low→high
df = compute_df(Fz, dz=h_step)
```

## Height Table by Potential Type

| Potential | R0 (Å) | Repulsive wall (r < R0) | h_min (Å) | h_max (Å) | Source |
|-----------|--------|-------------------------|-----------|-----------|--------|
| Soft sphere (pure repulsive) | R=1.5 | all r (exponential) | 2.0 | 4.0 | config Z_DF=[3.0,5.0] with mol_top≈1 |
| Morse (VdW, real molecules) | 3.0–4.0 | r < R0 | 2.7 | 3.7 | `generate_from_xyz` defaults, SPAMMM CLI |
| Morse (soft-sphere mapped, R0=1.5) | 1.5 | r < 1.5 | 0.5 | 2.0 | MUST be closer than soft sphere! |
| LJ (VdW) | 3.0–4.0 | r < R_min | 2.7 | 3.7 | SPAMMM CLI defaults |

## Critical Rule

**The scan height range MUST match the potential's repulsive wall distance.**

- Pure repulsive (soft sphere): V = A·exp(-β(r-R)) → repulsion at ALL r → can scan far (h=2-4 Å)
- Morse: E = E0·(exp(2K(r-R0)) - 2·exp(K(r-R0))) → repulsion ONLY at r < R0 → MUST scan at h < R0
  - At h > R0, Morse is **attractive** → df is all negative → no bright features
  - At h < R0, Morse is **repulsive** → df has bright protrusions at atom positions

## SPAMMM CLI Defaults (reference)

```
--h-min 3.7 --h-max 4.7 --h-step 0.1 --amp 1.0 --bond-length 3.0
```
These are for **real molecules** with VdW Morse params (R0 ≈ 3-4 Å). For soft-sphere-mapped Morse (R0=1.5), use h_min=0.5, h_max=2.0 instead.

## Common Mistakes

1. **Using world z directly** instead of h above molecule top → probe ends up far above or below molecule
2. **Using soft-sphere Z_DF for Morse** → Morse attractive at those distances → all-dark images
3. **Forgetting bond_length offset** → tip_z ≠ PP_z + bond_length → PP at wrong height
4. **Not reversing Fz** → df computed with wrong z-order → inverted contrast
5. **Field grid doesn't cover scan range** → PP samples zero-force region → flat df

## Verification

After generating df, ALWAYS check:
- `df.max() > 0` (some repulsive features exist)
- `df.min() < 0` (some attractive features exist)
- `np.std(df) > 1e-4` (non-degenerate signal)
- Plot df at closest probe height → should show bright protrusions at atom positions
