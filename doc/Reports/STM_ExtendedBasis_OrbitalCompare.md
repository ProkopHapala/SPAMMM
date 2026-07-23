# Report: STM orbital panel — stock vs prolonged DFTB vs pySCF

**Date:** 2026-07-23  
**Status:** awaiting USER review (not Done)  
**Task:** `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md`  
**Artifacts:** `debug/stm_orbital_compare/`  
**Driver:** `run_spm.py stm *` / `tests/SPM/testplot_stm_basis_compare.py`  
**GPU pySCF notes:** `doc/AGENTS/notes/pyscf-gpu-scf.md`

## Verdict (agent)

1. **HOMO-index bug (fixed):** `eigvals < 0` labeled near-zero **virtuals** as HOMO (pentacene #56 @ −0.18 eV). Correct valence count → HOMO#50 @ −4.79 eV. That largely explains the earlier “DFTB ≠ pySCF orbital shape” at STM heights.
2. **Frontier at z=0.5 Å:** pentacene HOMO−4…LUMO+1 morphologies match well; farther virtuals diverge. PTCDA **LUMO…LUMO+5** match; **occupied** HOMO−5…HOMO still differ (possible near-degeneracy / DFTB π ordering).
3. **Vacuum tails:** stock STO still dies by ~3.5 Å; prolonged restores LDOS (separate from indexing).

## Caveat (SSOT): DFTB HOMO ≠ `eigvals < 0`

DFTB MO energies sit around E_F (~−4 eV for aromatics). Empty states between E_F and 0 eV are still **negative**, so `np.where(eigvals < 0)[-1]` returns a near-gap **virtual**, not HOMO.

| Molecule | Wrong (`E<0`) | Correct (valence `n_elec//2`) |
|----------|---------------|-------------------------------|
| pentacene | #56 @ −0.18 eV | #50 @ −4.79 eV |
| PTCDA | #78 @ −0.64 eV | #69 @ −6.43 eV |

**Code:** `spammm.SPM.AFM_utils.dftb_frontier_mo_indices` (and all STM paths that call it). Do not reintroduce `eigvals < 0` as occupation.

## Projection kernels (DFTB vs pySCF)

| Channel | What evaluates ψ(r) | Basis completeness |
|---------|---------------------|--------------------|
| **DFTB** | OpenCL `project_orbital_dense_points[_exp]` in `kernels/LCAO_grid.cl` via `Grid_dftb` | Multi-ζ **STO** tables (stock / prolonged) — not GTO |
| **pySCF** | CPU `pyscf.dft.numint.eval_ao` × `mo_coeff` in `eval_mo_on_xy_slice` / `eval_mo_on_grid` | **Full** contracted GTO of the SCF basis (e.g. def2-SVP double-ζ). **No** AO dropping or density truncation |

There is no OpenCL GTO STM path yet; pySCF orbital maps are rigorous LCAO on the host.

## Orbital vs STM maps (plot convention)

| Kind | Physics | Plot `field` | Colormap |
|------|---------|--------------|----------|
| **Orbital** | ψ with **phase** (signed) | `psi` | RdBu_r |
| **STM current** | I ≥ 0, no phase; tip picks φ_t, I ∝ \|⟨φ_s\|H'\|φ_t⟩\|² | `psi2` / `stm` | viridis |

Frontier diagnostic @ z=0.5 Å → orbitals (`--frontier-diag`). MO-resolved STM @ z≈3 Å → current (`--frontier-stm-diag`).

## Frontier diagnostic (z=0.5 Å, ±5 MOs)

```bash
python run_spm.py stm orbitals --molecule pentacene,PTCDA --n-near 5
# MO-resolved STM current @ z=3Å, separate plot per tip (s / pz / py)
python run_spm.py stm current --molecule pentacene,PTCDA --n-near 5 --stm-tips s,pz,py
```

Legacy testplot flags (`--frontier-diag`, `--frontier-stm-diag`) call the same `spammm.SPM.stm_compare` backend.

Energy convention on plots: **E grows up** (unoccupied above occupied).

| File | Content |
|------|---------|
| `…/frontier_diag/spectrum_*.png` | eigspectrum with arrows to ±5 MOs |
| `…/frontier_diag/orbitals_z0.5_*.png` | ψ maps DFTB \| pySCF (virtuals on top) |
| `…/frontier_diag/spectrum_orbitals_vertical_z0.5_*.png` | portrait: ψ \| E↑ \| ψ + sloped connectors |
| `…/frontier_diag/spectrum_orbitals_horizontal_z0.5_*.png` | widescreen: DFTB / E→ / pySCF |
| `…/frontier_stm_diag/tip_{s,pz,py}/spectrum_stm_{vertical,horizontal}_z3.0_*.png` | MO-resolved STM current @ z=3Å |
| `…/TIMING_frontier_diag.out` | Orbital ψ: SCF + projection |
| `…/TIMING_frontier_stm_diag.out` | STM current: SCF + SK/GTO projection |

### Energies (correct HOMO)

| | DFTB HOMO | pySCF HOMO | gap DFTB | gap pySCF |
|--|-----------|------------|----------|-----------|
| pentacene | −4.79 eV (#50) | −4.17 eV (#72) | 1.12 eV | 0.96 eV |
| PTCDA | −6.43 eV (#69) | −6.15 eV (#99) | 1.33 eV | 1.27 eV |

Legacy `eigvals<0` would have picked pentacene #56 (−0.18 eV) / PTCDA #78 (−0.64 eV).

### Timing benchmark (2026-07-23, RTX 3090, 3ob-3-1, PBE/def2-SVP)

**Protocol:** `n_near=±5` → **12 MOs** per molecule; scan step **0.25 Å**; pySCF backend `auto` (GPU OpenCL when available). Wall time from `time.perf_counter()` in driver (includes Python orchestration; DFTB SCF is DFTB+ wrapper wall).

#### A — Frontier orbitals @ z = 0.5 Å (signed ψ, OpenCL STO vs `eval_ao`)

| Molecule | scan | DFTB SCF | DFTB ψ proj (12 MO) | ms/MO | pySCF SCF | pySCF ψ proj (12 MO) | ms/MO | SCF ratio py/DFTB |
|----------|------|----------|----------------------|-------|-----------|----------------------|-------|-------------------|
| pentacene | 80×45 | **0.22 s** | **0.007 s** | 0.6 | 9.6 s | 0.038 s | 3.2 | **44×** |
| PTCDA | 71×52 | **0.14 s** | **0.006 s** | 0.5 | 14.0 s | 0.054 s | 4.5 | **103×** |

**Takeaway:** after SCF, **orbital maps are cheap on both sides** (≪1 s for 12 MOs). pySCF projection is ~5–7× slower per MO than DFTB OpenCL STO, but still only tens of ms/MO. **SCF dominates pySCF wall time** (~10–14 s vs ~0.2 s DFTB).

#### B — MO-resolved STM current @ z = 3.0 Å (`mo_overlap_points_exp_sk` vs `eval_ao`+deriv²)

Per-tip projection over the same 12 MOs; tips `s`, `p_z`, `p_y` (3×12 = 36 maps/molecule).

| Molecule | DFTB SCF | DFTB STM proj (3 tips) | ms/MO (s-tip) | pySCF SCF | pySCF STM proj (3 tips) | ms/MO (s-tip) | ms/MO (p_z tip) |
|----------|----------|------------------------|---------------|-----------|-------------------------|---------------|-----------------|
| pentacene | 0.17 s | **0.019 s** | 0.5 | 9.5 s | **0.22 s** | 3.2 | 7.4 |
| PTCDA | 0.13 s | **0.018 s** | 0.5 | 13.0 s | **0.28 s** | 4.0 | 9.8 |

p_y tip ≈ p_z (pySCF ∂/∂y vs ∂/∂z on host). DFTB SK overlap is **~0.5 ms/MO** regardless of tip; pySCF p-tip costs **~2× s-tip** (deriv evaluation).

**Takeaway:** STM current maps remain **interactive after SCF** on DFTB (<20 ms for 12 MOs×3 tips). pySCF STM projection adds **~0.2–0.3 s** per molecule (still small vs SCF).

#### C — End-to-end cost model (one molecule, both diagnostics)

| Stage | pentacene | PTCDA |
|-------|-----------|-------|
| DFTB SCF (once, shared) | ~0.2 s | ~0.1 s |
| pySCF SCF (once, shared) | ~9.6 s | ~14 s |
| + orbitals ψ (12 MO) | +0.01 s | +0.01 s |
| + STM current (3 tips × 12 MO) | +0.02 s | +0.02 s |
| **Total DFTB** (ψ + STM) | **~0.2 s** | **~0.1 s** |
| **Total pySCF** (ψ + STM) | **~9.8 s** | **~14.3 s** |

FDBM AFM imaging is a **separate** density-based path (`run_spm.py afm`); timings live in `doc/Tasks/PerfBenchmark_FDBM.md`, not this STM report.

Artifacts: `debug/stm_orbital_compare/TIMING_frontier_diag.out`, `TIMING_frontier_stm_diag.out`.

## Method

| Channel | SCF | Projection |
|---------|-----|------------|
| DFTB mio / 3ob **stock** | stock WFC SCF | stock multi-ζ STO (`use_exp_basis=False`) |
| DFTB **prolonged** | same SCF MOs | `make_slater_tail_species_list` (PTCDA: SA ζ from `debug/dftb_basis_sa_ptcda/`) |
| **pySCF** | GPU OpenCL RKS PBE/def2-SVP | Full GTO `numint.eval_ao`×`mo_coeff` on same xy slices + HOMO/LUMO `.cube` (no AO truncation) |

Important: STM uses the **STO table**, not generic `exp(β,r0)` — otherwise prolonged WFC would not matter.

## Review files

```
debug/stm_orbital_compare/pentacene/panel_pentacene_vs_pyscf_z{2.5,3.0,3.5}.png
debug/stm_orbital_compare/PTCDA/panel_PTCDA_vs_pyscf_z{2.5,3.0,3.5}.png
debug/stm_orbital_compare/{mol}/panel_{mol}_all_bases_z2.5.png   # mio+3ob
debug/stm_orbital_compare/{mol}/pyscf/*_HOMO.cube, *_LUMO.cube
debug/stm_orbital_compare/{mol}/SUMMARY.out
```

## Intensity snapshot (3ob, |ψ|² max)

| Molecule | z / Å | stock HOMO | prolonged HOMO |
|----------|-------|------------|----------------|
| pentacene | 2.5 | 1.6e-7 | 2.8e-6 |
| pentacene | 3.0 | 1.1e-9 | 1.9e-7 |
| pentacene | 3.5 | **0** | 1.2e-8 |
| PTCDA | 2.5 | 1.8e-7 | 8.1e-6 |
| PTCDA | 3.0 | 1.0e-9 | 8.7e-7 |
| PTCDA | 3.5 | **0** | 8.9e-8 |

pySCF SCF (RTX 3090): pentacene ~2.5 s (16 cyc), PTCDA ~3.9 s (19 cyc); cubes+slices ~14–16 s/mol total.

## Reproduce

```bash
# Timing + plots (orbitals)
python run_spm.py stm orbitals --molecule pentacene,PTCDA --n-near 5 --bases 3ob
# Timing + plots (STM current)
python run_spm.py stm current --molecule pentacene,PTCDA --n-near 5 --stm-tips s,pz,py
# DFTB vacuum HOMO/LUMO panel only
python run_spm.py stm panel --molecule pentacene,PTCDA --skip-pyscf
# L0 regression
pytest tests/SPM/test_stm_basis_compare.py -m gpu -s
```

## Open / not Done

- USER visual sign-off on nodal morphology vs pySCF.
- Optional: sign-correlation / RMSE metrics vs resampled cubes.
- Do not mark task Done without USER confirmation.
