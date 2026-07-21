# Report: Kriging DFT vs FDBM (DFT-cube vs DFTB) — pyridine / CO tip

**Status:** investigating (not Done — awaiting USER confirmation)  
**Date span:** 2026-07-20 … 2026-07-21  
**Preferred system:** Mithun endgroup `N-h` (pyridine) + tip `CO_O`  
**Primary artifacts:** `debug/afm_fdbm_diag_pyridine_gui_match/`  

**Living task / design docs (read these first):**

| Doc | Role |
|-----|------|
| [`doc/Tasks/Import_KrigingGridFF.md`](../Tasks/Import_KrigingGridFF.md) | Session log, open bugs, tip×sample matrix, Δρ clamp recipe |
| [`doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md`](../Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md) | Science schema: Kriging GridFF ↔ FDBM channels |
| [`doc/Tasks/ProlongedRadialBasis_DFTB.md`](../Tasks/ProlongedRadialBasis_DFTB.md) | Dual basis: prolonged Slater = **Pauli only**, never normalize |
| [`doc/Tasks/PauliFitting_TestDesign.md`](../Tasks/PauliFitting_TestDesign.md) | How Pauli \(A,\beta\) should be fitted vs QM z-scans |
| [`doc/DFTB_basis_fit.md`](../DFTB_basis_fit.md) | Prolonged / SA Slater API |
| skill:`afm-plotting` | Plot SSOT: tip vs probe, amp, z ladders, site lists |

This report consolidates **what we implemented**, **what physics we found**, and **what to test next**. It does **not** close the task.

---

## 1. Goal

Bring **Kriging / RBF interpolation of DFT AFM z-scans** into SPAMMM as a regular GridFF, then **compare** that DFT reference to **FDBM** built from:

1. **All-electron cubes** (Psi4/pySCF Mithun `Dt.cube` sample + tip), and  
2. **DFTB+** valence densities (`get_density_from_dftb_dense` / ModularPipeline).

Comparison is meaningless unless grids, tip×sample labels, probe sites, and PP conventions (lever \(L\), \(K_\mathrm{lat}\) units, df amplitude) are consistent.

**FDBM channels used here (PLQ):** Pauli + London(vdW) + Coulomb/Hartree. Fukui as 4th GridFF channel is noted for later (`KrigingGridFF_DFT_vs_FDBM.md`).

---

## 2. What we implemented (engineering)

### 2.1 Kriging → GridFF

- Modules: `spammm/SPM/{interpy,InterpolatorKriging,InterpolatorRBF,KrigingGridFF}.py`
- Mithun `N-h` + `CO_O` → `F_total(nx,ny,nz,4)` for `AFMulator.setup_fdbm_grid`
- Demo / compare: `tests/SPM/testplot_kriging_relax.py`, `testplot_kriging_vs_fdbm_cube.py`

### 2.2 Cube → FDBM path

- `get_density_from_cube`, Gaussian / compact ρ_NA, `soft_clamp_*`, `delta_rho_clamp_compact_na`
- GPU charge+dipole-preserving resample (`GridsOCL` / `kernels/grids.cl`) — scipy sample broke ∫Δρ
- Tall z-symmetric boxes (\(L_z\gtrsim18\) Å) to kill FFT tip wrap
- Pipeline split: **Pauli = full ρ**; **ES = Poisson(Δρ_sample) ⊗ tip_Δρ**

### 2.3 DFTB FDBM path

- ModularPipeline / `testplot_fdbm_relax.py` with GUI-matched params when comparing to GUI
- Dual-basis hooks for prolonged Slater on Pauli only (see §5)

### 2.4 Tip × sample matrix (must label both)

| Label | Sample | Tip |
|-------|--------|-----|
| DFT×DFT | Mithun `N-h` cube (clamp Δρ → \(V\); full ρ Pauli) | Mithun `CO_O` |
| DFTB×DFTB | DFTB pyridine stock Δρ | DFTB CO |
| DFT×DFTB | cube sample | DFTB tip |
| DFTB×DFT | DFTB sample | cube tip |

Layouts: `FDBM_*_vs_Kriging_layout.png`, tip-swap 4-panels, `NOTE_tip_sample_combos.out`.  
Cached fields: `FDBM_sampDFT_tipDFT_fields.npz`, `…tipDFTB…`, `FDBM_DFTB_fields.npz`, `FDBM_sampDFTB_tipDFT_fields.npz`.

### 2.5 Plotting SSOT (refactored for reuse)

Canonical helpers in `spammm/SPM/AFM_utils.py` (skill:`afm-plotting`):

| API | Purpose |
|-----|---------|
| `normalize_probe_sites` / `fdbm_probe_sites_from_indices` | **Any** number of sites (not hardcoded 3) |
| `fdbm_probe_sites_nch` | Pyridine: N, farthest **C**, farthest H |
| `plot_fdbm_vs_kriging_zlayout` | Top: V / E_es / Pauli+tot; bottom: **one panel per site** |
| `plot_fdbm_methods_zcompare_4panel` | Overlay methods (DFT solid / DFTB dashed) |
| `afm_tip_probe_heights` | Tip ladder ↔ probe = tip − \(L\) |
| `plot_afm_Fz_df_threerow` | 3×nz: Fz_unrelax · Fz_relax · df |

**Plot conventions (USER-approved):** E−E(6), V−V(8); yellow band \(z\ge2.5\); `aspect='equal'`; AFM tip 5.5→8.0 with \(L=3\) ⇒ probe 2.5→5.0; df **peak** amp = 1.0 Å on a **dense** z stack.

AFM images: `AFM_sampDFT_tipDFT_df_Fz.png`, `…DFTB…`, `AFM_Kriging_df_Fz.png`, etc.

---

## 3. Session history (physics timeline)

### Phase A — Infrastructure & false leads (2026-07-20)

1. **FFT z-wrap** on short boxes → fake high-\(z\) Pauli/ES → tall boxes.  
2. **Fake sample \(p_z\)** from monopole strip on z-asymmetric cell + scipy resample → GPU project + z-symmetric box; far-field ES → 0 by \(z\sim8\)–12.  
3. **Pauli fit absorbing bad ES** → fit Pauli to Kriging \(E\) only (not residual).  
4. **Tilted `pyridine.xyz`** (~21°) made DFTB look broken → flatten; prefer Mithun `N-h`.  
5. **\(K_\mathrm{lat}\):** GUI/human = N/m; internal = eV/Å². Bare `0.5` as eV/Å² ≈ 8 N/m (too stiff). After fix, 0.5 N/m may look soft — retune still open.

### Phase B — “Cube ES vs Kriging / DFTB” (main science)

Control: **DFTB FDBM** ES ~meV and AFM morphology sane; **cube FDBM** still wrong attractive halo / ES walls vs Kriging.

Early hypothesis: all-electron cores − crude Gaussian ρ_NA on ~0.1 Å AFM grids. Soft-clamp helps far field but can reshape multipoles; **tip clamp** exploded CO \(p_z\) (~25×) when applied like sample.

### Phase C — Same tip, tip×sample swap (2026-07-21)

With **same** Mithun `CO_O` tip and Pauli \(A,\beta\) fit to Kriging, Pauli can look similar, but **\(E_\mathrm{es}\)** still differs: cube tip ⊗ cube \(V\) bends attractive early (~−0.5 eV @ N, \(z=2\)); soft DFTB tip stays mild.

**Verdict:** not tip monopole alone (that was a tight `margin=1` crop artifact, \(q_{\Delta\rho}\sim-0.7\)). Remaining bend is **cuspy all-e tip Δρ ⊗ cube sample \(V\)**. Sample clamp + soft tip ≈ DFTB-mild ES; clamp tip still bends.

### Phase D — Probe-site bug

“Opposite C” with `argmax` over **all** atoms picks **para-H**, not carbon → fake \(V\) sign flip. SSOT: farthest **carbon** (`Z==6`). Always print `xy=` on titles.

### Phase E — GUI vs agent AFM mismatch

Not density / not \(K_\mathrm{lat}\): agent used fitted Pauli \(A=155\), GUI still **A=787** (old pentacene). Heights / \(L\) also differed. Diagnostics must match GUI params when comparing to GUI.

### Phase F — Δρ clamp → compact NA + AFM images

Recipe (CO guinea-pig first): soft-clamp spikes → compact \(f=(1-(r/r_c)^2)^2\) NA rematch → ∫Δρ≈0. Applied to sample `N-h` and tip `CO_O`. Tip apex Δρ still electron-depleted on vacuum side (δ+ vs classic tipQs≈−0.1) — open.

AFM 3-row panels for all tip×sample combos + Kriging. **df amp bug:** `compute_df_amp` on 5 coarse slices + nearest clamp made df look too close; fix = dense PP scan covering ±amp.

### Phase G — Plotting refactor (this handoff)

Generalize site count; document tip/probe/amp/z in skill:`afm-plotting`; this report.

---

## 4. What we found: two DFTB ↔ DFT distinctions

Comparing Kriging (DFT interaction) to FDBM is **not** one “DFTB vs DFT” knob. Two independent differences dominate morphology and ES:

### Distinction 1 — How Δρ (neutral-atom subtraction) is built

| Path | Total density | Neutral atom / core treatment | Typical ES at AFM heights |
|------|---------------|-------------------------------|---------------------------|
| **DFTB** | Valence AO projection | Diagonal **ρ_NA** from Denmat0 / DFTB NA | ~meV; sane |
| **Cube (early)** | All-electron SCF | Spherical Gaussian σ≈0.3 (or worse σ≳0.6 leak) | Large / wrong morphology |
| **Cube (current recipe)** | All-electron | Soft-clamp + compact \(f=(1-(r/r_c)^2)^2\) NA | Sample \(V\) usable with soft tip; tip Δρ still problematic |

**Charge bookkeeping:** cubes ∫ρ ≈ ∑Z (CO: 14); DFTB ∫ρ ≈ ∑Z_val (CO: 10). Do not compare core spikes on the same color scale — use a common **valence** axis for Δρ morphology.

**Future test A:** run **DFTB** with **spherical / compact NA subtract** instead of diagonal ρ_NA — isolate whether “orbital NA vs spherical NA” matters when the basis is the same.

### Distinction 2 — Basis range (vacuum tails)

| Path | Radial support | Effect on Pauli |
|------|----------------|-----------------|
| **DFTB stock** (mio/3ob) | Short STOs, hard cutoff ~6 Bohr | Underestimates density 1–4 Å above molecule → weak / wrong Pauli tails |
| **DFT / cube** | Longer Gaussian / AO basis | Stronger vacuum density → different overlap at AFM heights |
| **Prolonged / SA Slater** | Extended tails on **same** SCF DM | Intended correction for Pauli only |

**Future test B:** **extended Slater orbitals** for Pauli projection (tip first — overlap lives in vacuum), stock short basis kept for Δρ → \(V_\mathrm{ES}\).

---

## 5. Dual basis / extended Slater — critical rules (do not “fix”)

USER clarification (2026-07-21) — agents repeatedly got this wrong:

| Channel | Density | Normalize ∫ρ = \(N_e\)? |
|---------|---------|-------------------------|
| **ES** | Stock short-basis **Δρ** → Poisson | Yes — neutrality / multipoles must be physical |
| **Pauli** | Prolonged / SA / extended Slater projection of **same** DM | **Never** — ∫ρ ≉ \(N_e\) **by design** |

- Extended / prolonged Slater is **ONLY for Pauli**. It is **not** a proper charge density for electrostatics.
- Pauli \(A(\int\rho_s\rho_t)^\beta\) cares about **local** overlap in the tip region; \(A,\beta\) absorb overall scale.
- Dual basis looks inconsistent; it is an intentional practical trick so short-basis DFTB AFM improves **without** re-SCF and **without** corrupting ES.
- Code: `make_slater_tail_species_list`, `get_density_from_dftb_dense(..., projection_basis_ang=...)`, `ProlongedRadialBasis_DFTB.md`, `DFTB_basis_fit.md`. Tip prolonged SA is higher priority than sample-only.

---

## 6. Snapshot: what works vs open

| Check | Result |
|-------|--------|
| Kriging GridFF + PP path | Works (reference for pyridine+CO) |
| Cube FDBM far-field ES (\(p_z\), no parabola) | OK after z-sym + GPU project |
| Cube FDBM vs Kriging AFM-height morphology | **Still FAIL** on ES / total (tip Δρ / apex charge open) |
| DFTB FDBM flat pyridine | ES ~meV; df/Fz progression looks normal (GUI A/β) |
| Sample clamp Δρ + soft tip \(E_\mathrm{es}\) | Mild — close to DFTB |
| Clamp tip ⊗ cube \(V\) | Still early attractive bend |
| Opposite-C probe site | Fixed (carbon, not H) |
| Plot helpers (n sites, tip/probe, 3-row AFM) | In `AFM_utils` + skill |
| Prolonged tip Slater SA | **Not done** (planned) |
| DFTB + spherical NA (vs diagonal ρ_NA) | **Not done** (planned) |
| Pauli \(A,\beta\) USER-approved for real CO | Fit exists; values not locked |
| \(K_\mathrm{lat}\) magnitude | Units path OK; working N/m may need retune |

---

## 7. Recommended next experiments (systematic)

Do these **separately** so effects do not confound:

1. **DFTB Δρ recipe swap** — same stock DFTB ρ_scf; replace diagonal ρ_NA with compact/Gaussian spherical NA; compare \(V_\mathrm{ES}\), \(E_\mathrm{es}\), AFM to Kriging.  
2. **Extended Slater Pauli** — dual basis: stock Δρ→ES; prolonged tip (then sample) ρ→Pauli; **never normalize**; re-fit \(A,\beta\) to Kriging contact.  
3. **Cube tip Δρ** — finish CO guinea-pig clamp/NA until valence Δρ matches DFTB morphology; check apex δ− vs δ+; only then re-run DFT×DFT AFM.  
4. **Cross matrix after each change** — always report sample×tip labels; keep plot SSOT.

---

## 8. Reproduction pointers

```bash
# Cube vs Kriging compare CLI
python tests/SPM/testplot_kriging_vs_fdbm_cube.py --endgroup N-h --tip CO_O

# DFTB GUI-like (K_LAT in N/m)
SPAMMM_AFM_CPU_FFT=1 python tests/SPM/testplot_fdbm_relax.py \
  --xyz data/xyz/pyridine.xyz --basis mio-1-1 --tip-mode co \
  --K_LAT 0.5 --outdir debug/afm_fdbm_diag_pyridine

# OpenCL: unrestricted shell so NVIDIA ICD is visible (not PoCL)
```

Canonical z-layout / AFM plotting: import from `spammm.SPM.AFM_utils` — do not reinvent (skill:`afm-plotting`).

---

## 9. Agent caveats (from this campaign)

1. Prefer NVIDIA OpenCL; never report PoCL timings as GPU.  
2. All-electron ≠ DFTB valence.  
3. Clamp tip only with multipole diagnostics; roll tip_tot and tip_del with the **same** O peak.  
4. Tip project margin ≥3–4 Å (or full cell).  
5. \(V_\mathrm{ES} \neq E_\mathrm{es}\); similar \(V(z)\) can still give different convolution.  
6. Probe = tip − \(L\); sample Fz/df at **probe** plane; df amp needs dense \(z\).  
7. Never mark this task Done without USER confirmation + shown verification.
