---
type: Report
title: Coarse-grained 2.5D contact-surface vs GridFF AFM
description: Physics, sampling intent, fixed h₀ bugs, and remaining contact-sep vs Morse+Coulomb GridFF differences on PTCDA and helicene assemblies.
tags: [afm, contact-surface, 2.5D, GridFF, Morse, helicene, PTCDA, parity]
timestamp: 2026-07-24
status: investigating
---

# Coarse-grained 2.5D contact-surface vs GridFF AFM

**Status:** investigating — major `h₀` / sampling bugs fixed and images regenerated; **awaiting USER visual confirmation** before marking fixed.  
**Design SSOT:** [`../Topics/AFM/ContactSurface_Static.md`](../Topics/AFM/ContactSurface_Static.md)  
**Task:** [`../Tasks/Fast_2p5D_AFM_ContactSurface.md`](../Tasks/Fast_2p5D_AFM_ContactSurface.md)  
**Assembly session:** [`Assembly_ContactSurface_AFM_helicene_2026-07-24.md`](Assembly_ContactSurface_AFM_helicene_2026-07-24.md)  
**Audit:** [`../TopicalAudit/AFM_ContactSurface.md`](../TopicalAudit/AFM_ContactSurface.md)  
**Caveats:** [`../Caveats.md`](../Caveats.md) §contact-surface

---

## 1. Essence

Classical PP-AFM usually stores a dense 3D Morse(+Coulomb) GridFF (`img_FF`) and interpolates it every tip step. For large rigid adsorbates / SAM assemblies that volume is wasteful: the tip only feels a **thin contact layer**.

The **2.5D contact surface** replaces `img_FF` with:

```text
E(x,y,z) ≈ Σ_{ijk} c_ijk  B_i(x) B_j(y)  φ_k( z − h₀(x,y) )
```

Fit once from brute Morse(+Coulomb) samples; evaluate during PP relax. Intent: **atom-scale lateral nodes** (high-order B-splines carry the smooth in-between), not sub-atomic voxels.

**Reference for classical Morse+Coulomb:** 3D GridFF / brute atom sum. Contact-sep is an approximation that must match **onset height and E(z)/Fz(z) shape**, not invent sharper physics.

---

## 2. Methods compared

| Path | Representation | Typical cost | Role |
|------|----------------|--------------|------|
| **Brute** | Σ Morse(+Coulomb) per query | Slow | Fit target / profile truth |
| **GridFF 3D** | Dense `(nx,ny,nz)` + trilinear/`interpFE` | Fit heavy; scan OK | Classical reference images |
| **contact-sep** | `h₀(xy)` + B-spline×z-modes | Fit ~seconds; scan ≪1 s | Screening / assemblies |
| **contact-PIC** | Per-atom radial modes + PIC | Compact | Alternative (ii); not this report’s focus |

Assembly CLI: `run_assembly_afm.py` (`--method contact-sep`, `--compare-dir` for contact vs GridFF).

---

## 3. Bugs found and fixed (2026-07-24)

### 3.1 Wrong `h₀` — atom centers vs contact spheres

**Legacy:** `h₀ = max_i z_i` under the tip (atom **centers**). That is not a contact surface → soft / long-ranged fits (`dz` measures height above nuclei, not above contact).

**Correct:** ray vs nearby spheres of radius `R = h0_R_scale × Morse_R0`:

```text
h₀(x,y) = max_i  [ z_i + sqrt(R_i² − ρ_i²) ]   (ρ < R_i)
```

Code: `build_contact_height_map(..., Rs=…)`, `fit_contact_surface(h0_mode='spheres')`.

| Mode | 1-atom Δh_well vs brute | Notes |
|------|-------------------------|-------|
| `atom_z` (legacy) | ~+0.2 Å soft | Wrong physics |
| `spheres`, scale=1.0 | Well at R0 but clamp kills repulsion | Clamp sits at Morse minimum |
| **`spheres`, scale=0.75** | Tracks brute into hard wall | Default — clamp in **repulsion** |

**Rule:** `h0_R_scale < 1` so the hard clamp is inside the repulsive wall, not at the well.

### 3.2 Sub-atomic sampling (against design intent)

Defaults were accidentally **~10× finer than an atom**:

| Knob | Old default | **Corrected default** | Meaning |
|------|-------------|----------------------|---------|
| `--bspl-dx` | 0.2 Å | **1.0 Å** | B-spline / `h₀` nodes |
| `--scan-dx` | 0.15 Å | **0.5 Å** | Image pixels |

Rationale: high-order B-splines exist so the **coefficient grid can be atom-coarse**; dense scan only makes razor peaks look “physical.” Helicene rank09 after correction: scan **112×70** @ 0.5 Å; setup prints `step=1.00Å`.

### 3.3 Profiles after fix (helicene rank09)

With `h0_mode=spheres`, `h0_R_scale=0.75`, `bspl_dx=1.0`, `scan_dx=0.5`:

- **E(z) / Fz(z)** at tops T0/T1: contact-sep tracks **brute**; GridFF can deviate near hard wall (voxel / margin).
- **XY maps:** contact still **sharper** than GridFF at close approach — residual model sharpness, not 0.15 Å oversampling alone.
- Fit RMSE (separable) ~5×10⁻⁴ on rank09 with coarse nodes.

**Do not call demo-ready** until USER accepts the XY morphology.

---

## 4. Reproduction

```bash
# PTCDA separable fit + close / z-stack parity
python tests/testplot_contact_surface.py
# REVIEW: debug/testplot_contact_surface/contact_surface_{comparison,z_profile,close_parity}.png

# Helicene re-AFM all ranks (coarse defaults)
python run_assembly_afm.py --rerun-ranks --outdir debug/helicene_afm_pipeline \
  --h0-R-scale 0.75 --bspl-dx 1.0 --scan-dx 0.5 --z-clearance 8 --nz-scan 40

# Contact-sep vs GridFF maps + E/Fz profiles (rank09)
python run_assembly_afm.py --compare-dir debug/helicene_afm_pipeline/rank09_idx2767 \
  --h0-R-scale 0.75 --bspl-dx 1.0 --scan-dx 0.5 --grid-dx 0.25 --z-clearance 8 --nz-scan 40
```

**REVIEW (helicene):**

- `debug/helicene_afm_pipeline/rank09_idx2767/compare_contact_vs_gridff_maps.png`
- `debug/helicene_afm_pipeline/rank09_idx2767/compare_contact_vs_gridff_profiles.png`
- `debug/helicene_afm_pipeline/rank*/afm_df_Fz_heights.png`

**Toys:** `python tests/testplot_contact_surface.py --toys` → `debug/testplot_contact_surface/toys/`

---

## 5. Current defaults (`run_assembly_afm.py`)

| Flag | Default | Notes |
|------|---------|-------|
| `--method` | `contact-sep` | PIC / morse-3d available |
| `--h0-R-scale` | **0.75** | Sphere radius = scale×Morse_R0 |
| `--bspl-dx` | **1.0 Å** | Atom-scale nodes |
| `--scan-dx` | **0.5 Å** | Atom-scale pixels |
| `--z-clearance` | 8.0 Å | Initial `h_probe` above mol zmax |
| `--grid-dx` | 0.25 Å | GridFF voxels (compare only) |

Heights in plots are **Å above molecular zmax**, not substrate. Lever `L≈|dpos0_z|≈4 Å` → tip = probe + L.

---

## 6. Open issues

1. **XY sharpness** — contact-sep still more atom-pinpoint than GridFF at close `h`; decide if basis / force weighting / Boltzmann needs further softening, or if GridFF blur is the fair classical look.
2. **USER confirmation** of PTCDA + helicene maps/profiles before status → fixed.
3. **PIC path** not re-validated with sphere `h₀` / coarse dx.
4. **GridFF compare box** = cell AABB; FF atoms from 3×3 PBC — edge effects at supercell rim.
5. Pipeline flag `--contact-surface {separable,pic,grid3d}` still open.
6. Elastic Phase 2 (`ContactSurface_Elastic.md`) out of scope unless asked.

---

## 7. Code map

| Path | Role |
|------|------|
| `spammm/surfaces/ContactSurface.py` | `h₀` spheres, separable/PIC fit, GPU eval |
| `spammm/SPM/AFM.py` | `fit_contact_surface`, `run_scan_contact`, GridFF `run_scan` |
| `kernels/contact_surface.cl` | Brute Morse, Av/Atv, PP contact relax |
| `run_assembly_afm.py` | Assembly → AFM; `--compare-dir` |
| `tests/testplot_contact_surface.py` | PTCDA + toys |
| `tests/SPM/test_afm_contact_surface.py` | L0 stencil |

---

## 8. Short verdict

| Item | Verdict |
|------|---------|
| Tip / plot labels as root of “too close” | Ruled out (PBC xyz + height SSOT fixed earlier) |
| `h₀ = max z_atom` | **Wrong** — use sphere envelope |
| `h0_R_scale = 1.0` | **Wrong** — clamp at well; use **0.75** |
| Sub-atomic `bspl/scan` dx | **Against design** — defaults **1.0 / 0.5 Å** |
| E/Fz(z) vs brute (tops) | Much improved |
| XY vs GridFF | Still sharper; **USER review** |
| Mark fixed? | **No** until USER confirms |
