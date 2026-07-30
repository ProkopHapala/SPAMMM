---
type: Report
title: FGR STM transfer M=H−ES vs overlap — pentacene / PTCDA (2026-07-29)
status: awaiting_user_review
tags: [STM, FGR, Bardeen, DFTB, OpenCL, overlap, prolonged-basis]
timestamp: 2026-07-29
---

# FGR STM transfer \(M = H - ES\) vs orbital overlap

**Status:** awaiting USER review (not Done).  
**Artifacts:** `debug/stm_fgr_compare/{pentacene,PTCDA}/`  
**CLI:** `python run_spm.py stm fgr --molecule pentacene,PTCDA`  
**Design notes:** [`Ideas/LCAO_STM_FGR_WIRING.md`](../Ideas/LCAO_STM_FGR_WIRING.md), chat [`Ideas/STM_perturbation_H.chat.md`](../Ideas/STM_perturbation_H.chat.md)  
**Topical audit:** [`TopicalAudit/STM_FGR_Transfer.md`](../TopicalAudit/STM_FGR_Transfer.md)  
**Related:** prolonged tails [`Reports/STM_ExtendedBasis_OrbitalCompare.md`](STM_ExtendedBasis_OrbitalCompare.md); Dyson later [`Tasks/DysonOrbitals_DFTB_STM.md`](../Tasks/DysonOrbitals_DFTB_STM.md)

---

## Essence

Constant-height STM current between one tip orbital and one sample MO is first-order Fermi golden rule with a **nonorthogonal** tip–sample block:

\[
M_{ts}(E) = c_t^\dagger \bigl(H_{TS} - E\, S_{TS}\bigr)\, c_s .
\]

The legacy GPU path approximates this by an artificial exponential Slater–Koster “overlap” (`mo_overlap_points_exp_sk`), i.e. \(H'\approx 1\). That systematically **over-weights the vacuum halo** relative to the molecular core. This session wired the existing `LCAO_STM_FGR.cl` kernels to **prolonged STO** two-centre tables (Level B: exact \(S\) + extended-Hückel \(H\)) and compared four intensity maps on DFTB+ 3ob pentacene and PTCDA.

---

## Physics (what \(H'\) is)

Partition \(H = H_S + H'\). For a sample eigenstate \(H_S|\psi_s\rangle=\varepsilon_s|\psi_s\rangle\),

\[
M_{ts} = \langle\phi_t|H - H_S|\psi_s\rangle = H_{ts} - \varepsilon_s S_{ts}.
\]

Elastic tunnelling \(\varepsilon_t\simeq\varepsilon_s=E\) → \(M = H_{TS}-E S_{TS}\). Gauge: \(H\to H+CS\), \(E\to E+C\) leaves \(H-ES\) invariant; bare \(H\) or bare \(S\) do not.

**Not** a local nuclear-potential mask on the overlap volume. Volume \(H-ES\) cancels the nonorthogonal “looks alike” part of the tails; Bardeen’s surface flux is the partition-independent content.

**Deliberately omitted here:** SCC charge response, tip-induced polarization, Dyson/GF dressing, bias-window / DOS factors, diagonalization in the scan.

---

## What was implemented (Level B)

| Layer | Location | Role |
|-------|----------|------|
| OpenCL | `kernels/LCAO_STM_FGR.cl` | `build_stm_transfer_sk_tables`, `stm_fgr_sk_tau_scan_real` (production), complex + HS debug kernels |
| Host scan | `spammm/quantum/DFTB/Grid_dftb.py` | Loads FGR kernels; `stm_fgr_sk_tau_scan_real`, `build_stm_transfer_sk_tables_gpu` |
| Tables | `DFTBplusParser.build_longtail_eh_sk_tables` | Prolonged STO \(S_\gamma(R)\) (cylindrical quadrature) + \(H_\gamma = K\cdot\tfrac12(\varepsilon_A+\varepsilon_B)\,S_\gamma\) |
| API / panel | `AFM_utils.project_mo_stm_fgr_slice`, `stm_compare.run_fgr_transfer_compare` | Modes `tau` / `S` / `H` vs legacy `overlap_exp` |
| CLI | `run_spm.py stm fgr` | Default outdir `debug/stm_fgr_compare/` |

**Table generation levels** (from wiring note):

| Level | Content | Status |
|-------|---------|--------|
| **B** (this report) | Exact long-tail \(S\); EH \(H\propto S\) | **active prototype** |
| A | Frozen neutral-atom \(H^0\) integrals + exact \(S\) | not coded |
| C | Fit \(\tau(R)\) directly | not coded |

**Dirty but intentional basis split:** DFTB MO coefficients \(c\) from short mio/3ob; radial transfer from prolonged \(\tilde\chi \propto r^{n-1}e^{-\zeta r}\). Nodes/phases from \(c\); vacuum decay from tables. No rediagonalization.

**Orbital order:** OpenCL `[px,py,pz,s]` (same as `mo_overlap_points_exp_sk`). Host remaps DFTB `[s,p…]` via `evec_to_kernel_coeffs`. Directed axis \(u=(R_\mathrm{sample}-R_\mathrm{tip})/R\); `sp` and `ps` stored separately (typically \(S_{ps}\approx -S_{sp}\) for identical orbitals).

**Energy zero:** \(H\) and \(E_\mathrm{tunnel}\) in Hartree from DFTB onsite (SKF line 2) and sample eigenvalue; first tests use \(E=\varepsilon_s\).

---

## Reproduce

```bash
# NVIDIA ICD must be visible (agents: Shell permissions all)
python -c "import pyopencl as cl; print([(p.name,[d.name for d in p.get_devices()]) for p in cl.get_platforms()])"

python run_spm.py stm fgr --molecule pentacene,PTCDA \
  --stm-z-above 3.0 --stm-tips s,pz --scan-step 0.3 --margin 3.0 --bases 3ob
```

Useful knobs: `--eh-K` (default 1.75), `--tip-elem C`, `--rcut 10`.

---

## Protocol (this run)

| Item | Value |
|------|--------|
| Basis | DFTB+ **3ob-3-1** |
| Heights | \(z = z_\mathrm{mol}+3.0\) Å |
| Scan | step 0.3 Å, margin 3 Å |
| Tips | point tip \(s\), \(p_z\) (phantom C for SK types) |
| EH \(K\) | 1.75 |
| Prolonged \(\zeta\) | `SLATER_TAIL_ZETA` (+ PTCDA SA override if present) |
| Device | RTX 3090 (NVIDIA first; not PoCL) |

**Panel columns:** `overlap_exp` | `I_S=\|c^\dagger S c\|^2` | `I_H=\|c^\dagger H c\|^2` | `I_\tau=\|c^\dagger(H-ES)c\|^2`

---

## Results (agent reading)

### Artifacts

| Path | Content |
|------|---------|
| `debug/stm_fgr_compare/pentacene/fgr_compare_z3.0_pentacene.png` | 4×4 panel |
| `debug/stm_fgr_compare/PTCDA/fgr_compare_z3.0_PTCDA.png` | 4×4 panel |
| `…/SUMMARY.out` | scalars + crude core/border contrast |
| `…/fgr_compare_z3.0.npz` | raw maps |

### Crude contrast (= mean(centre)/mean(frame); >1 ⇒ brighter core)

| Molecule / row | overlap_exp | I_S | I_H | I_τ |
|----------------|-------------|-----|-----|-----|
| pentacene HOMO \(s\) | 19.7 | 188 | 188 | 188 |
| pentacene LUMO \(s\) | 9.9 | 60.9 | 60.9 | 60.9 |
| pentacene HOMO \(p_z\) | 190 | 205 | 205 | 205 |
| pentacene LUMO \(p_z\) | 68.6 | 65.5 | 65.5 | 65.5 |
| PTCDA HOMO \(s\) | 8.0 | 1.5 | 0.9 | **0.7** |
| PTCDA LUMO \(s\) | 4.1 | 111 | 82 | **72** |
| PTCDA HOMO \(p_z\) | 6.4 | 1.2 | 0.6 | **0.5** |
| PTCDA LUMO \(p_z\) | 26.5 | 83 | 44 | **24** |

### Interpretation (not USER-confirmed)

1. **Long-tail \(S\) vs `overlap_exp`** already changes lateral weight a lot (pentacene HOMO \(s\): core/frame ~20 → ~190). Replacing the artificial \(\exp(-\beta(r-r_0))\) SK radial is the first-order fix for vacuum halos.
2. **Level B limitation:** \(H_\gamma\propto S_\gamma\) → for C/H-dominated tip–sample pairs, \(I_S\), \(I_H\), \(I_\tau\) share essentially the **same shape** (identical contrast ratios on pentacene). Channel-dependent \((K\bar\varepsilon-E)\) only reshapes when onsite mix matters (O in PTCDA).
3. **PTCDA LUMO:** FGR long-tail maps concentrate intensity on the molecular body vs `overlap_exp` (desired direction for the original “halo too bright” complaint).
4. **PTCDA HOMO asymmetry — root cause found (2026-07-30):** HOMO#69 (E=−6.4310 eV) and HOMO-1 #68 (E=−6.4328 eV) are split by only **1.8 meV** — a near-degenerate pair. DFTB's eigensolver returns an arbitrary rotation within this 2D subspace, breaking the molecular inversion symmetry (PTCDA has D₂h with inversion, not independent x/y mirrors). Single-MO maps have inversion asymmetry ~1.60; **summing I over the degenerate pair restores symmetry** (asymmetry → 0.04, comparable to LUMO's 0.02 baseline). LUMO#70 is 1598 meV above LUMO+1 → well isolated → symmetric. Fix implemented: `--degen-thresh 0.005` (eV) in `run_fgr_transfer_compare` sums I over the degenerate cluster. This is the physically correct STM current at finite bias/T (all degenerate states contribute). The same fix is needed in all STM paths — see `Tasks/STM_FGR_CLI_GUI_Integration.md`.
5. Absolute intensities differ by orders of magnitude across columns (expected: different operators / units); panels use **per-image** clim.

### Height ladder (2026-07-30, z = 3, 4, 5, 6 Å, degen fix applied)

Contrast = mean(centre)/mean(border)  (>1 ⇒ brighter molecular core)

**Pentacene** (no degeneracy, HOMO#50/LUMO#51 well separated):

| z Å | HOMO s overlap | I_τ | LUMO s overlap | I_τ | HOMO pz overlap | I_τ | LUMO pz overlap | I_τ |
|-----|----------------|-----|----------------|-----|-----------------|-----|-----------------|-----|
| 3 | 19.7 | 188 | 9.9 | 60.9 | 190 | 205 | 68.6 | 65.5 |
| 4 | 5.6 | 82.6 | 3.4 | 30.3 | 33.2 | 86.6 | 16.3 | 31.5 |
| 5 | 2.0 | 39.5 | 1.8 | 16.5 | 7.8 | 40.7 | 4.6 | 16.8 |
| 6 | 0.7 | 20.3 | 0.9 | 9.5 | 2.4 | 20.7 | 1.7 | 9.6 |

**PTCDA** (HOMO×2 degenerate cluster [68,69], LUMO#70 isolated):

| z Å | HOMO×2 s overlap | I_τ | LUMO s overlap | I_τ | HOMO×2 pz overlap | I_τ | LUMO pz overlap | I_τ |
|-----|------------------|-----|----------------|-----|-------------------|-----|-----------------|-----|
| 3 | 8.3 | 0.83 | 4.1 | 71.7 | 6.8 | 0.59 | 26.5 | 23.8 |
| 4 | 7.1 | 1.16 | 2.5 | 45.6 | 6.8 | 0.52 | 7.4 | 18.3 |
| 5 | 6.0 | 1.70 | 2.2 | 24.7 | 6.1 | 0.50 | 3.0 | 13.6 |
| 6 | 6.1 | 2.56 | 2.5 | 13.3 | 5.9 | 0.52 | 2.3 | 9.7 |

Observations: (a) At all heights, `overlap_exp` decays much slower than FGR (overlap_exp contrast drops below 1 at z=6 for pentacene, while I_τ stays >9 — FGR concentrates on the molecular core). (b) PTCDA LUMO I_τ contrast decreases with height but stays >1 (core-bright) at all heights. (c) PTCDA HOMO×2 I_τ stays <1 at all heights (heteroatom weighting suppresses core). Artifacts: `debug/stm_fgr_compare/{pentacene,PTCDA}/fgr_compare_z{3,4,5,6}.0_*.png`.

### Timing (order of magnitude, this machine)

DFTB SCF ~0.15–0.2 s; table build ~0.07–0.08 s (few element pairs × ~100 \(R\)); scan pixel ~1 ms class on RTX 3090 for these grids.

---

## Open issues / next steps

- [ ] **USER L2 review** of height-ladder PNGs (z=3,4,5,6 Å, degen fix applied) — morphology vs experiment / known BR-STM.
- [x] **Height ladder** z=3,4,5,6 Å run for pentacene + PTCDA (2026-07-30). See height-ladder table above.
- [x] **PTCDA HOMO asymmetry** root-caused (near-degenerate pair 1.8 meV) and fixed (sum I over cluster). See interpretation §4 above.
- [ ] **Level A** frozen \(H^0\) (kinetic + \(v_A^0+v_B^0\)) so \(H\) and \(S\) differ radially — expected to unlock true \(H-ES\) cancellation beyond EH.
- [x] **BR-STM with FGR** — wired FGR transfer into `compute_bond_resolved_stm` (new `compute_bond_resolved_stm_fgr`) and `run_br_stm_afm_panel` Stage 3 via `stm_mode='fgr'`. CLI: `run_spm.py stm br --stm-mode fgr`. Parity verified (far height |Δ|/flat=0.18, close height=0.94). See `Tasks/STM_FGR_BondResolved.md`.
- [x] **BR-STM FGR compare gallery** — `run_spm.py stm br-fgr` produces 4-column panels (BR-overlap | BR-I_S | BR-I_H | BR-I_τ) with PP-AFM tip displacement at Fz heights. Artifacts: `debug/stm_br_fgr_compare/{PTCDA,pentacene}/`.
- [x] **Cutoff/taper fix** (2026-07-30): `rcut` default 10→15 Å + cosine taper `taper_w=2.0` Å in all 3 FGR scan kernels (`stm_cutoff_taper` in `LCAO_STM_FGR.cl`). Eliminates hard-cutoff ring artifacts visible at z=5,6 Å.
- [ ] **FGR into CLI/GUI** — promote from compare gallery to first-class STM mode. See `Tasks/STM_FGR_CLI_GUI_Integration.md`.
- [ ] Optional Bardeen-plane FFT reference for selected geometries.
- [x] **L0 pytest** (`tests/SPM/test_stm_fgr_compare.py`): table Sps≈−Ssp smoke + benzene scan I_τ≠overlap_exp + NVIDIA device marker.
- [ ] Do **not** promote Level B into BR-STM product path until USER OK.

---

## Code map (SSOT pointers)

```text
Ideas / derivation  →  doc/Ideas/STM_perturbation_H.chat.md
Wiring / table API  →  doc/Ideas/LCAO_STM_FGR_WIRING.md
Kernel              →  kernels/LCAO_STM_FGR.cl
Legacy overlap      →  kernels/LCAO_grid.cl :: mo_overlap_points_exp_sk
Host                →  spammm/quantum/DFTB/Grid_dftb.py
Tables              →  spammm/quantum/DFTB/DFTBplusParser.py
Compare driver      →  spammm/SPM/stm_compare.py :: run_fgr_transfer_compare
CLI                 →  run_spm.py stm fgr
```
