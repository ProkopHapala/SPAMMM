# Task: Prolonged / corrected DFTB radial basis (STM + AFM)

**Status:** investigating  
**Priority:** Soon  
**Human ToDo:** item 4 (“find where we have these tests”)  
**Parent:** `doc/Tasks/RepoConsolidation.md`

## Objective

Make DFTB+ **density / orbital projection** use radial bases with correct **vacuum tails** (prolonged / Slater-corrected), for both **AFM (FDBM density)** and **STM (orbital maps)** — and wire the existing diagnostic scripts into the formal test system.

## Why

Standard 3ob/mio STOs are confined (hard cutoff ~6 Bohr). AFM/STM probe density/orbitals **1–4 Å above** the molecule; short tails underestimate Pauli / LDOS.

Design write-up already exists: `doc/DFTB_basis_fit.md`.

## Where the tests / tools already are

| Artifact | Path | Role |
|----------|------|------|
| Tail fit + corrected `wfc` writer | `tests/SPM/testplot_3ob_basis_tails.py` | Visual: log\|R(r)\|, blend to exp tail, writes `debug/plot_3ob_basis_tails/wfc.3ob-3-1.corrected.hsd` |
| DFTB vs pySCF radial compare | `tests/SPM/testplot_dftb_vs_pyscf_basis.py` | Visual basis shapes |
| README index | `tests/SPM/README.md` | Lists `testplot_3ob_basis_tails.py` |
| SA optimizer | `spammm/quantum/DFTB/basis_optimizer.py` | Fit N,ζ vs reference z-profile |
| SA + local pySCF ρ CLI | `examples/density_comparison/optimize_basis.py` | `--ref-rho` / `--project-density` / `--compare-pauli` |
| FDBM stock vs SA AFM | `tests/SPM/testplot_fdbm_relax.py --ptcda-stock-vs-sa` | 8-height df/Fz strip |
| Parser + Slater-tail helpers | `spammm/quantum/DFTB/DFTBplusParser.py` (`make_slater_tail_species_list`, …) |
| GPU projector | `spammm/quantum/DFTB/Grid_dftb.py` | `update_basis_sto()` without recompile |
| Design | `doc/DFTB_basis_fit.md` | Motivation + API |
| Density compare examples | `examples/density_comparison/` | `optimize_basis.py`, `compare_densities.py` |
| Topical audit | `doc/topical_audit.md` § QM Integration | Points at the plot scripts |
| Related Soon item | Folded **poly** power-sequence (`FeatureChecklist.md`) | Different topic (surface folded basis), do not confuse |
| Campaign report | `doc/Reports/PTCDA_FDBM_prolonged_basis.md` | PTCDA Ez → SA → AFM |
| Presentation | `FOR_PRESENTATION.md`, `debug/presentation.html` | Artifact links / gallery |

**Gap:** tools are mostly `testplot_*` / examples — **few L0 asserts** that prolonged basis improves tail vs pySCF/GPAW or that STM/AFM pipelines pick the corrected WFC.

**DFT reference progress (2026-07-20):** site-correct **pySCF GPU** CO z-scans for PTCDA (PBE/def2-SVP, z→15 Å) are archived in `tests/ref_data/CO_scan_pyscf_gpu/` with timing — see `doc/Reports/PySCF_GPU_CO_zscan_PTCDA.md`. Use these (not Fukui `jobs_CO_scan_pyscf_short`) when fitting prolonged-basis / FDBM Pauli vs DFT for large aromatics.

## Progress (2026-07-20) — PTCDA end-to-end (USER review)

**Not Done** — awaiting USER confirmation. Full write-up: `doc/Reports/PTCDA_FDBM_prolonged_basis.md` · slides: `FOR_PRESENTATION.md` · gallery: `debug/presentation.html`.

| Done | Item |
|------|------|
| ✓ | Site-correct pySCF Ez + sample ρ for PTCDA |
| ✓ | SA Slater-tail fit vs pySCF ρ (`optimize_basis.py --ref-rho`) → `debug/dftb_basis_sa_ptcda/` |
| ✓ | Pauli A,β CO-tip refit vs Ez (stock 3ob & SA-prolonged) |
| ✓ | L2 AFM strip: stock **3ob** vs **SA-prolonged** over 8 heights → `debug/fdbm_ptcda_stock_vs_sa/compare_stock3ob_vs_SAprolonged_heights.png` |
| ✗ | L0 pytest for tails / WFC selection |
| ✗ | GUI / ModularPipeline switch to prolonged WFC as default projection |

**Naming:** stock = multi-ζ **3ob-3-1** (not mio). **SA** = Simulated Annealing of single-exp Slater `(N,ζ)` for projection only.

**CLI:** `SPAMMM_AFM_CPU_FFT=1 python tests/SPM/testplot_fdbm_relax.py --ptcda-stock-vs-sa`

## Deliverables

1. **Inventory note in this file** (done above) — answer to “find where we have these tests?”.
2. **L0 pytest** — e.g. `tests/SPM/test_dftb_basis_tails.py`:
   - parse 3ob; assert corrected ζ in range; assert log-density at z=2–3 Å higher than raw 3ob cutoff (vs synthetic or cached ref).
3. **Pipeline switch** — AFM FDBM + STM orbital projection honor `basis` / `wfc` path for prolonged set (GUI combo already has mio/3ob in `AFMExtension`).
4. **L2** — keep `testplot_3ob_basis_tails` + one FDBM/STM slice with prolonged vs stock.
5. Document default: stock 3ob for energy; prolonged for projection-only (SSOT in `DFTB_basis_fit.md` + module header).

## Acceptance

- [ ] L0 test in `pytest -m "not slow"` (or marked `dftb` if exe required)
- [ ] Prolonged WFC selectable in AFM/STM path
- [ ] USER reviews before/after density or orbital plot — **PTCDA AFM strip ready** (`debug/fdbm_ptcda_stock_vs_sa/`)
- [ ] Not marked Done without USER confirmation

## Out of scope

- Folded poly surface basis power sequence (separate Soon item).
- Full Dyson coefficients (see `DysonOrbitals_DFTB_STM.md`) — prolonged basis is a **prerequisite** for vacuum LDOS quality.
