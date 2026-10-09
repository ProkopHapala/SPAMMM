# AFM "Ghost/Shade" Model — Asymmetric-Tip Contrast: Physics, Image Model, De-Shading & Augmentation

Status: implemented + validated (benzene parity, real-molecule demo, emboss fit on bonus set).
Purpose: document the directional-shade ("emboss"/"ghost") artifact seen in
`Images_Bonus/set_2_shade_AFM/` — its physical origin, the two models we built
for it, and how to (a) remove it from experimental data to help ring
recognition and (b) fake it for ML data augmentation.

## 1. The artifact

The bonus set (`set_2_shade_AFM/1.png … 15.png`, 128×128 px, bond scale
b ≈ 7.9 px ≈ 0.18 Å/px) shows strong directional shading: every ring/protrusion
is bright on one side, dark on the opposite side, plus a smooth bright/dark
"halo" around the molecular footprint. Per-image band-pass decomposition
(`gauss(0.9·b) − gauss(3.5·b)`) isolates it cleanly — see figures below.

## 2. Physical model — displaced lateral-spring rest point (`dpos0.xy`)

The probe-particle (PP) spring model in `kernels/AFM.cl` (`tipForce`, ~l.53):

```c
return (dpos - dpos0.xyz) * stiffness.xyz        // harmonic 3D spring → rest at dpos0.xyz
     + dpos * (stiffness.w * (r - dpos0.w) / r); // radial spring → |dpos| → sphere r=dpos0.w
```

- **Symmetric CO tip**: `dpos0 = (0, 0, −L, L)` — lateral-spring minimum directly
  below the anchor; stiff radial spring keeps the PP on the sphere |dpos|=L.
- **Asymmetric (bent CO) tip**: `dpos0 = (Δx, Δy, −L, L)` — the harmonic minimum
  is displaced; the PP bends preferentially toward (+Δx,+Δy). Stroke and
  cantilever axis stay vertical, so the fast z-only df pipeline remains valid.

Typical parameters (user-specified): `K_RAD = 20–30 N/m` (≫ lateral),
`K_LAT = 0.5 N/m` (→ ~0.031 eV/Å² via `stiffness_Nm_to_eVA2` — do **not** pass
0.5 as eV/Å², that is ~17× too stiff), bond/sphere length `L ≈ 3 Å`,
`AMP = 1.0 Å`. Realistic asymmetry: `|Δ| ≈ 0.3–0.6 Å` (≈6–12° lean); ≥1 Å is
exaggerated and suppresses df amplitude (bent CO slides off atoms instead of
pressing — `max df` 0.108→0.053→0.031 for |Δ|=0,0.5,1.0 on benzene).

### Implementation

| piece | where |
|---|---|
| spring force `tipForce` (dpos0 = rest point + sphere radius) | `kernels/AFM.cl` ~l.53 |
| `AFMulator.set_tip_asymmetry(dx, dy)` — shifts `dpos0.xy` | `spammm/SPM/AFM.py` ~l.1392 |
| `AFMulator.set_tip_tilt(theta, phi)` — tilts whole tip frame + stroke axis (different, weaker effect; NOT the right mechanism for this shade) | `spammm/SPM/AFM.py` ~l.1402 |
| `dpos0` override in established GridFF scan | `AFMulator.scan_fdbm` `spammm/SPM/AFM.py` ~l.493 |
| `PmeForward.eval(apos, Zs, dxy=(dx,dy))` — ContactPME fast evaluator w/ asymmetry | `doc/export_invAFM/scripts/testplot_img2mol_correct.py` ~l.118 |
| `asym_demo()` — φ×|Δ| grid figure | `doc/export_invAFM/scripts/testplot_img2mol_correct.py` ~l.328 |

Relaxation is mandatory — the asymmetry only appears in **relaxed** PP-AFM
(`relaxStrokesTilted*` kernels iterate FIRE/damped relaxation; `tip_disp` output
reaches ~1.8–3.5 Å at close approach). Evaluating the bare potential/force at
unrelaxed positions shows almost nothing.

Backend parity: GridFF-Morse (`scan_fdbm`, CLI `run_spm.py` path) and
ContactPME agree to ~4 decimals on benzene (df over C atom 0.02478 both) and
produce the same asymmetry dipole — ContactPME is safe as the fast forward
model (~0.13–0.15 s/image once the field fit is cached).

### What it produces

At close approach (h_df = 3.7 Å → closest approach ~2.7 Å), one rim of each
ring brightens and sharpens while the opposite side dims — a dipole signature
per feature that mirrors under Δ → −Δ. Verified on benzene
(`debug/testplot_img2mol_correct/benzene_asym/benzene_asym_close.png`) and the
main-set molecule (`debug/testplot_img2mol_correct/2/2_asym_demo.png`).

## 3. Image-space model — blurred blob + emboss (directional derivative)

The relaxed-asymmetry physics is equivalent, to first order, to the displaced
probe sampling `df(x+Δ) ≈ df(x) + Δ·∇df`. In image space that is literally
"blurred density + emboss filter" — the model the user proposed.

**Decomposition fit** (per bonus image): isolate the shade by band-passing
`s = gauss(g, 0.9b) − gauss(g, 3.5b)`, then least-squares fit

```
s(x,y) ≈ c0 + c·Ĩ + a·∂xĨ + b·∂yĨ        (+ optional 2nd derivatives)
```

with `Ĩ = gauss(g, σ≈1.0·b)` the blurred image. Results (residual RMS /
RMS(s)):

| image | resid (blob+emboss) | emboss φ |
|---|---|---|
| 1.png  | 0.38 | 103° |
| 2.png  | 0.41 |   5° |
| 3.png  | 0.37 |  38° |
| 4.png  | 0.37 | 340° |
| 5.png  | 0.35 | 126° |
| 10.png | 0.41 |  28° |
| 11.png | 0.40 |   0° |
| 15.png | 0.44 | 353° |

Key findings:

- **Pure emboss alone** (`s ≈ a·∂xĨ + b·∂yĨ`): resid ≈ **1.0** — no fit. The
  shade is not just a derivative of the blob.
- **blob + emboss**: resid ≈ **0.35–0.44** — dominant term is `c·Ĩ` (the
  displaced/smeared molecular blob itself = dark lobes on ring clusters,
  bright surround), emboss adds the directional dipoles per feature.
- Adding **2nd derivatives** of Ĩ (or of blurred atom density ρ̃) pushes resid
  to ~0.42–0.46 from either basis — marginal gain, more params.
- emboss direction φ is per-image consistent (clusters ~0–40° and ~103–126°) —
  i.e. a per-scan tip bend direction.

Interpretation: the experimental shade = the *effectively displaced* blurred
molecular contrast. Both terms (`c·Ĩ` and `Δ·∇Ĩ`) come from one lateral
displacement Δ — consistent with the bent-CO model at a blurred band.

## 4. Use A — de-shading experimental images (enhance bond contrast)

Goal: remove the low-frequency ghost so ring/center recognition
(`ring_centers_graph` / U-Net) sees sharp bond features.

Procedure (per image, `b` = `img2mol.estimate_bond_scale`):

1. `g` = raw grayscale; band-pass `s = gauss(g, 0.9·b) − gauss(g, 3.5·b)`.
2. `Ĩ = gauss(g, σ)` with `σ ≈ 1.0·b` (fit gave σ ∈ [5.5, 7.5] px ≈ 0.7–1.0 b;
   sweep σ over e.g. [0.5, 1.5]·b and keep the argmin residual).
3. Least-squares `s ≈ c0 + c·Ĩ + a·Gx + b·Gy` (Gx,Gy = `np.gradient(Ĩ)`) over
   interior pixels → emboss magnitude `α = hypot(a,b)`, direction
   `φ = atan2(b,a)` (image coords; watch the y-flip vs Å coords).
4. De-shaded image: `g' = g − (a·Gx + b·Gy)` — optionally also `− c·Ĩ` if the
   goal is a flat background (for bond contrast keep only the emboss removal).
5. Then run the usual pipeline (`prep_image 'flat6n'` already high-passes
   σbg=0.75b + local-RMS — it removes most of the ghost; explicit emboss
   subtraction targets exactly the directional dipole the high-pass leaves).

Note the correction loop already uses this robust scoring: `Scorer` in
`doc/export_invAFM/scripts/testplot_img2mol_correct.py` (~l.148) preps both experiment and sim
with `flat6n` and computes NCC inside the `flat16` molecule mask, so the score
tracks bond-scale features, not the shade.

## 5. Use B — augmentation generator (fake the ghost for ML training)

Cheap recipe (one PME sim, arbitrary many shaded variants):

```python
bl  = gaussian_filter(df, sigma)                      # σ ≈ 2–3 px (≈1.5–2 Å)
Gx, Gy = np.gradient(bl)
emb = (cosφ·Gx − sinφ·Gy) / percentile(hypot(Gx,Gy), 98)   # normalized emboss
df_shaded = df + α · (p99(df) − p1(df)) · emb              # α ∈ [0.3, 1.0], φ ∈ [0, 2π)
```

Validated: at σ≈3 px, α≈0.6–1.0 the composites are visually
indistinguishable from the bonus set's shaded look
(`debug/testplot_img2mol_correct/shade_model/shade_generator.png`).

### 5a. Synthetic dataset generator — `doc/export_invAFM/scripts/testplot_shade_aug.py`

Builds a shaded-image training set from `debug/pah_afm_db/pah_db.pkl` (100
relaxed PAH flakes, ContactPME df stacks, 11 slices h_df=3.7–4.7 Å @ 0.1 Å/px):

- per molecule × `--aug` variants: random df slice (near slices 0–4 — sharp,
  exp-like), emboss(σ∈[1.5,6]px, φ∈[0,2π), α∈[0,1]), 50% global tilt ramp,
  pixel noise, normalize [0,1];
- `fit_canvas` center-crop/edge-pads every image to **160×160** (fixed size so
  `np.stack` works; 160 covers the largest flake — 128 was cutting molecules);
- **targets**: ring centers from the true carbon skeleton via
  `nx.minimum_cycle_basis` (faces of the fused graph — correctly resolves
  pentagon cuts and Stone–Wales 5-7-7-5), rendered as σ=2 px Gaussian heatmap
  (same target convention as `testplot_img2mol_unet.build_sample`);
- output `debug/shade_aug/shade_aug.npz`: `imgs` (N,160,160) f32,
  `Y` heatmaps, `centers` (object array of (k,2) px coords, col/row order),
  `params` (mol, iz, h, σ, φ, α, tilt); review grid `aug_montage.png`.

Run: `python3 -u doc/export_invAFM/scripts/testplot_shade_aug.py [--max N] [--aug K] [--seed S]`
(400 samples in ~1 min; no GPU needed — pure image-space postprocess of
stored df stacks).

Physical alternative: `PmeForward.eval(apos, Zs, dxy=(Δx,Δy))` with random
|Δ| ∈ [0,0.6 Å], φ ∈ [0,2π) — ~0.15 s/image, gives the real relaxed asymmetry
(weaker at same |Δ|; the image-space emboss mimics the stronger experimental
bends cheaply). Optionally add a global linear ramp (sample tilt), low-freq
noise and scan-line stripes — `augment()` in `doc/export_invAFM/scripts/testplot_img2mol_unet.py`
(~l.51) already does affine + coarse-shading + stripes + pixel noise; plug the
emboss term in there for U-Net training-time augmentation.

### 5b. Background-field augmentation (local magnification/shrinkage)

Second observed artifact class: some image areas are enlarged, others shrunk —
typically from a low-frequency force background (electrostatic patches, vdW
background). Modeled by superimposing a slowly-varying potential into the
relaxed scan.

**Implementation**: `AFMulator.pme_set_field(V_fn)` (`spammm/SPM/AFM.py` ~l.1668).
V_fn(X,Y,Z) [eV] is sampled on the coarse PME mesh nodes, converted to B-spline
control coefficients by the same `_prefilter_3d` used at fit time, and ADDED to
`cpm.mesh_coeffs` (pristine coeffs cached; V_fn=None restores). The force
contribution then enters automatically through the analytic tricubic gradient
inside `cs_eval_contact_pme_at/_local_at` — i.e. it perturbs the PP *during
relaxation* and lands in the stored relaxed FEs. No kernel changes; works on
both `relaxStrokesTiltedContactPME` and `…Local`. Mesh h=1.0 Å tricubic
→ fields with σ ≳ 2–3 Å are well resolved; the field must be significant only
inside the scan volume (mesh = query_bounds + halo).

**Sweep** (`doc/export_invAFM/scripts/testplot_pah_afm_db.py --field [--mol path]`): `field_combos`
builds **atom-centred Gaussian-smeared Coulomb charges**
(`charges_field(apos, idx, qs, sigma)`: `V(r) = Σ q_i·K·erf(r_i/√2σ)/r_i`,
K=14.3996 eV·Å/e² — smooth 1/r tail + regularized core). Charge sites are the
heteroatoms (N/O) when present — the physical "charged atom" model of charge
transfer — or the two most exposed rim atoms for pure-C flakes.

**The honest energy scale — E_pp at the relaxed PP position.** Do NOT judge
V_ext on a fixed z-plane: the PP rides a 3 Å sphere below the apex and relaxes
*sideways out of the repulsive wall*, so what it actually samples is
`E_pp = res.FEs[...,3]` — the field energy evaluated at each stroke's relaxed
PP position (ascending-z after `shared_postprocess`). `field_fig` therefore
shows `E_pp(base) | ΔE_pp | E_pp(with field)` at apex h≈3 Å + df columns, and
prints `|dE_pp|max` vs `|E_pp|max` per row.

Measured on pyrene_2N (ContactPME): **|E_pp|max ≈ 16 meV** at closest approach
— the PP-sampled landscape is tiny. Atom-centred bare-Coulomb charges give:
q=−0.01e → 54 meV (3.4× landscape), q=−0.03e → 163 meV, q=−0.10e → 545 meV,
q=−0.30e → 1.6 eV (crater). So even "small" partial charges injected raw are
10–100× too strong: bare Coulomb at Å distances is enormous.

**Realistic distortion criterion**: |ΔE_pp| ≲ 5–80 meV at closest approach —
a fraction of the sampled landscape, not a multiple. Options that achieve it:

- bare smeared Coulomb with **q ≲ 0.005–0.01 e/site** (well below chemical
  partial charges — effectively a *screened* charge, matching that on-surface
  ES is damped);
- **screened Coulomb** `q·K·exp(−r/λ)/r`, λ ~ 2–4 Å (not yet implemented);
- **meV-scale atom-centred Gaussians**: `gauss_field` at atom sites,
  amp 5–50 meV, σ 2–3 Å — amplitude controlled directly.

Qualitative response (field_pyrene_2N.png): attractive charge → local
darkening at the site (PP pulled down → feature expands); repulsive → dimming
(feature shrinks); site dipole → one-sided asymmetry. df itself is robust —
even ΔE ≈ 5× landscape gives a believable image, so the image-space
"too strong" threshold is higher than the energy numbers suggest.

**Substrate model** (`--substrate`, user-added): `substrate_field` = LJ(9-3)
half-space `E_s[(d_eq/d)⁹/3 − (d_eq/d)³]` at z_sub=−3.3 Å (d clamped ≥2 Å —
unclamped d⁻⁹ poles explode the spline mesh); `mol_footprint` = soft heavy-atom
xy mask → footprint-screened variant ("molecule occludes substrate") produces
the **rim-following halo**; `footprint_attract` = footprint-localized
image-charge-like attraction `−A·mask/(z−z_img)²`. Uniform half-space terms
only shift the PP baseline — lateral structure needs footprint modulation.
The experiment-like look (bright wireframe on dark surround at h≈4.2 Å) came
from `sub E_s=0.4, screened` + `escale≈4` (stronger tip Morse tail).

### 5c. Real electrostatics: QEq charges × tip quadrupole (GridFF path)

Physical alternative to injected fields: give the **molecule** realistic
charges and the **tip** a quadrupole, let the Coulomb term do the distortion.

- `spammm/forcefields/QEq.py` — Rappe–Goddard charge equilibration
  (`compute_qeq_reqs(apos, enames)` → REQs with Q in col 2). For pyrene_2N:
  q(N) = **−0.243 e** on each pyridinic N, compensating + on adjacent C —
  correct electronegative-N physics.
- `doc/export_invAFM/scripts/testplot_pah_afm_db.py --qeq [--mol xyz]` → `qeq_sweep_one`:
  `a.atoms_arr[:,3] = qs` (QEq charges), `a.tipQs = [0,−sQ,+sQ,0]` at
  `tipQZs=[0,1.8,3.6]` Å (charges ±sQ above the PP — axial quadrupole
  **Qzz = 9.72·sQ eÅ²**); `make_forcefield` bakes Morse + tipQs·q_atom
  Coulomb into the GridFF grid per sQ (`evalMorseC_QZs_toImg`), then
  `scan_fdbm` relaxation + `shared_postprocess`. Multi-site tipQs is
  **GridFF-only**: ContactPME rejects non-radial tip electrostatics.
- Measured (pyrene_2N, h≈3 Å): |ΔE_pp| = 1.3 meV (sQ=0.025) → 10 meV
  (sQ=0.2, Qzz=1.94 eÅ²). The ES distortion is genuinely *weak* — meV-scale
  on the ~16 meV PP-sampled landscape — and spatially a rim/edge banding
  (quadrupole × molecular dipole-field gradient), not atom-localized.
- For the user-requested "sQ=0.05" regime (Qzz≈0.49 eÅ²): ΔE ~2.5 meV,
  df barely modulated — realistic CO-tip quadrupole ES contrast on a
  molecule this size is *subtle*; strong ES asymmetry in experiment needs
  either charged adsorbates (the §5b site charges, screened) or a charged/
  strongly polar tip (monopole q_tip — PLQH radial channel in the PME path).
- **Big N-doped flake (`--mol debug/mol_flakes/zz8x4_N.xyz`, C66H4N22,
  `--sq` now a CLI list):** 22 pyridinic-edge N at q≈−0.20 e each → strong
  molecular dipole field; the quadrupole couples to it much more visibly:
  |ΔE_pp|max = 12 meV (sQ=0.1) → 74 meV (sQ=0.6, Qzz=5.8 eÅ²) vs 19 meV
  baseline. The Δdf-vs-base columns in `field_zz8x4_N.png` show the
  distortion directly: at h=3.0 Å a **rim-following atom-resolved
  modulation** (|Δdf|max≈0.45–0.49, already saturated at sQ=0.1), at
  h=4.2 Å a **smooth corner-to-corner dipolar shading** (|Δdf|max≈0.003–
  0.009) — the same one-side-bright/one-side-dark character as the
  experimental ghost. So on large polar flakes even a modest quadrupole
  produces visible shading; on small molecules it stays meV-subtle.
- **Charged-defect augmentation (`--defect --dq D`):** `qeq_defect_combos`
  adds δq on top of QEq at a site — net charge = ionized dopant /
  charge-transfer defect; site+neighbor ±δq = neutral polarized-bond dipole.
  Sites: most interior + most exposed heteroatom (or rim C on pure-C flakes),
  circled in `field_fig`. **Control verified**: all defect rows at sQ=0 give
  exactly zero distortion — sample–sample Coulomb never acts on the PP;
  contrast only appears via the tip multipole × site-field coupling.
  Measured at sQ=0.3: dq=+0.3e → |ΔE_pp|max = 80–96 meV (vs 37 meV
  quadrupole-only, 19 meV baseline) with a ~3–5 Å localized blob at the site;
  neutral dipole defects ≈2× weaker/more compact. On the random PAH flakes
  an interior −0.3e nearly collapses local contrast (df range −0.08..0.62 →
  −0.004..0.075) — so keep |dq| ≲ 0.3 e and prefer 0.1–0.2 e for subtle
  defects. Verified on rand8p1_s102r/s105r (pure-C pentagon-defect flakes),
  rc4x6_N (12-N heterocycle ribbon), zz8x4_N. See
  `doc/export_invAFM/augmentation_mechanisms.md` for the full mechanism table
  and scale cheat-sheet.

Caveat: `run_scan_morse_direct` (exact uncapped Morse+Coulomb) also supports
tipQs but did not converge at closest approach within N_RELAX_STEP_MAX=128
FIRE steps — the interpolated GridFF path is the robust route for scans.

## 6. GPU df pipeline + sign convention (why earlier maps looked inverted)

- `compute_df_amp_z` (`AFM.py` ~l.3110): vectorized lerp + `np.gradient`
  replacement for the old `map_coordinates` path — bitwise-identical, ~10×
  faster. `compute_df_amp_dir` (~l.3259) auto fast-paths `osc_dir=(0,0,1)`;
  `compute_df_amp` (~l.3129) delegates.
- `dfAmpZ` kernel (`kernels/AFM.cl` ~l.717): 9-node Gauss–Chebyshev + central
  diff per (scan lane, iz), runs on the still-resident scan buffer;
  `AFMulator.compute_df_amp_z_gpu` (~l.1417) downloads only the df volume.
  ~0.4–1.5 ms vs ~215 ms scipy postprocess (144×109×31 volume).
- **Sign convention (fixed bug)**: scan buffers are kernel-order
  (iz=0 = highest tip z = descending z), so raw kernel output is `+dFz/dz =
  −df`. `compute_df_amp_z_gpu` now negates AND flips z → physical df in
  ascending-z, matching `shared_postprocess` (`AFM_utils.py` ~l.3363).
  Verified corr = +1.0000, max|Δ| ≈ 1e-8. Convention: repulsive bonds bright
  (df > 0 over atoms), attractive vdW halo darker. All asym/correction maps
  pre-fix were displayed inverted.

## 7. Correction loop (context)

`doc/export_invAFM/scripts/testplot_img2mol_correct.py` — greedy fused-ring graph moves (add 5/6-gon
on rim edges both sides with vertex snapping, remove ring+leaf verts), each
candidate scored by ContactPME relaxed scan → GPU df → best-|NCC| over df
slices vs flat6n-prepped experiment inside the flat16 mask. Bonus-set images
build graphs via cached U-Net centers + `constrained_ring_graph`
(`extract_graph_unet`, ~l.50). On bonus 5.png: 0.52→0.76 NCC in 6 moves
(4 adds, 1 rem, 1 pentagon) — but that was pre-de-shading scoring; with flat6n
scoring reached 0.717 @ 19 rings at the iteration cap.

## 8. Files & artifacts

Source:
- `spammm/SPM/AFM.py` — `set_tip_asymmetry` (1392), `set_tip_tilt` (1402),
  `compute_df_amp_z_gpu` (1417), `compute_df_amp_z` (3110), `compute_df_amp`
  (3129), `compute_df_amp_dir` (3259), `fit_contact_pme` (1513),
  `pme_set_field` (~1668), `run_scan_contact_pme` (~1950), `scan_fdbm` (493)
- `kernels/AFM.cl` — `tipForce` (~53), `relaxStrokesTilted*` (~478+),
  `dfAmpZ` (~717)
- `spammm/SPM/AFM_utils.py` — `shared_postprocess` (3363),
  `afm_df_height_stacks` (653), `scan_extent` (65), `imshow_afm` (362),
  `run_morse_pp_afm` (4025), `run_contact_pme_pp_afm` (4336)
- `doc/export_invAFM/scripts/testplot_img2mol_correct.py` — `PmeForward` (70), `Scorer` (148),
  graph moves (176–264), `asym_demo` (328), `correct` (368)
- `doc/export_invAFM/scripts/testplot_img2mol_afm.py` — `extract_graph` (57), `scan_spec_for`
  (80), `sim_to_px` (95), `ncc_masked` (107), `_make_afmulator` (164)
- `doc/export_invAFM/scripts/testplot_pah_afm_db.py` — PAH flake df database builder; `--sweep`
  scan-param sweep; `--field` external-field sweep (`gauss_field`,
  `field_combos`, `field_sweep_one`, `field_fig`)
- `doc/export_invAFM/scripts/testplot_img2mol_unet.py` — U-Net trainer, `augment` (51),
  `build_sample` (24)
- `doc/export_invAFM/scripts/testplot_shade_aug.py` — synthetic shaded training-set generator
  (from `pah_db.pkl` df stacks; emboss augmentation + ring-center targets
  via `nx.minimum_cycle_basis` faces; `fit_canvas` → 160×160)
- `spammm/img2mol.py` — `prep_image` (67; flat6n/flat16), `molecule_mask`
  (98), `estimate_bond_scale` (476), `constrained_ring_graph`

Artifacts (review):
- `debug/testplot_img2mol_correct/shade_model/shade_bases.png` — basis fits
- `debug/testplot_img2mol_correct/shade_model/shade_model_v3.png` — blob+emboss decomposition, all steps
- `debug/testplot_img2mol_correct/shade_model/shade_generator.png` — augmentation sweep (σ blur steps, emboss φ, composites α×φ)
- `debug/testplot_img2mol_correct/benzene_asym/benzene_asym_close.png` — GridFF vs PME asymmetry validation
- `debug/testplot_img2mol_correct/2/2_asym_demo.png` — real-molecule φ×|Δ| demo (ContactPME, corrected sign)
- `debug/testplot_img2mol_correct/unet_5/` — correction run on shaded bonus image
- `debug/pah_afm_db/field_rand8p1_s102r.png`, `field_rand8p1_s105r.png` — V_ext sweep figures (§5b)
- `debug/pah_afm_db/pah_db.pkl` — 100-flake ContactPME df database (input of the generator)
- `debug/shade_aug/shade_aug.npz` — 400-sample synthetic dataset (imgs/Y/centers/params, 160×160)
- `debug/shade_aug/aug_montage.png` — review grid of augmented samples

Exploratory scripts (to be folded into `testplot_shade_aug.py`):
`/tmp/shade_model.py`, `/tmp/shade_model2.py`, `/tmp/shade_model3.py`,
`/tmp/shade_gen.py`, `/tmp/test_df_gpu.py`, `/tmp/bench_df*.py`.

## 9. Open questions / next steps

- The experimental ghost may additionally involve an asymmetric tip *charge*
  (bent CO apex charge) — if needed, `tipQs` (multi-site tip charges) is the
  next dial; currently `tipQs[:]=0` and `q_tip=0` in the PME fit.
- De-shading parameter auto-fit (σ, φ, α) per image is implemented in
  `/tmp/shade_model3.py` — needs folding into `img2mol` or the test script.
- For the correction loop: also consider scanning *two* df slices and scoring
  on both, since shade strength is height-dependent.
