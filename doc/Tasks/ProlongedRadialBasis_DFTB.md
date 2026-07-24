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

### Dual basis (agents: do not “normalize” prolonged ρ)

USER clarification (2026-07-21) — agents kept getting this wrong:

- **ES:** always stock short-basis Δρ → Poisson (neutrality / multipoles).
- **Pauli only:** prolonged / SA Slater projection of the **same** SCF DM — vacuum tails.
- Prolonged ∫ρ is **not** supposed to equal \(N_e\); **never** charge-normalize it. Pauli is sensitive to **local** density in the tip region, not total charge; \(A,\beta\) absorb scale.
- Dual basis may look inconsistent; it is an intentional practical correction so short-basis DFTB+ AFM improves **without** re-SCF and without compromising ES.
- Code SSOT: `make_slater_tail_species_list` in `DFTBplusParser.py`; also `get_density_from_dftb_dense(..., projection_basis_ang=...)`, `ModularPipeline` header, `doc/DFTB_basis_fit.md`, session notes in `doc/Tasks/Import_KrigingGridFF.md`.

### Tip-first prolonged Slater (USER 2026-07-21)

We currently fit / apply prolonged Slater mainly for the **sample**. For AFM Pauli it is **even more important for the tip** (overlap is tip×sample in the vacuum region).

**Guinea-pig:** Mithun `CO_O` / DFTB CO tip — small, well-characterized.

**Plan:** systematic SA fit of prolonged STOs for tip (and optionally sample / both). End goal may be **tip-only** prolonged ρ (precomputed once) — simpler and cheaper. Still dual basis: stock Δρ→V_ES; prolonged ρ→Pauli only; never charge-normalize.

See also: all-electron Δρ clamp recipe on the same CO tip (`Import_KrigingGridFF.md` §CO guinea-pig).

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

**STM follow-on:** systematic HOMO/LUMO mio/3ob/prolonged vs pySCF cubes — `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md`  
**STM L2 gallery (2026-07-23, awaiting USER review):** `debug/stm_orbital_compare/` + report `doc/Reports/STM_ExtendedBasis_OrbitalCompare.md` (`tests/SPM/testplot_stm_basis_compare.py`).

**DFT reference progress (2026-07-20):** site-correct **pySCF GPU** CO z-scans for PTCDA (PBE/def2-SVP, z→15 Å) are archived in `tests/ref_data/CO_scan_pyscf_gpu/` with timing — see `doc/Reports/PySCF_GPU_CO_zscan_PTCDA.md`. Use these (not Fukui `jobs_CO_scan_pyscf_short`) when fitting prolonged-basis / FDBM Pauli vs DFT for large aromatics.

## Reference molecule panel — pySCF Fukui densities (USER 2026-07-23)

**Root:** `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/`  
**Level:** PBE / **def2-SVP**. Neutral sample density+ESP for FDBM: `rho_N.{cube,npy}`, `esp_N.{cube,npy}` (pentacene/PTCDA also have anion/cation A/C for Fukui).

| Dir (under root) | Local geometry | Notes |
|------------------|----------------|-------|
| `pentacene_PBE_def2-SVP/` | `data/xyz/pentacene.xyz` | flat z=0; N+A+C |
| `PTCDA_PBE_def2-SVP/` | `data/xyz/PTCDA.xyz` | flat z=0; N+A+C |
| `azaindol_dimer_PBE_def2-SVP/` | `data/xyz/azaindol_dimer.xyz` | flat z=0 (recomputed 2026-07-24) |
| `azaindol_isodimer_PBE_def2-SVP/` | `data/xyz/azaindol_isodimer.xyz` | flat z=0 |
| `benzoicacid_dimer_PBE_def2-SVP/` | `data/xyz/benzoicacid_dimer.xyz` | flat z=0 |
| `benzoicamid_dimer_PBE_def2-SVP/` | `data/xyz/benzoicamid_dimer.xyz` | flat z=0 |
| `phtalo_1-dftb-relax_PBE_def2-SVP/` | `data/xyz/phtalo_1.xyz` | **new**; flat; 50 atoms |
| `phtalo_2-dftb-relax_PBE_def2-SVP/` | `data/xyz/phtalo_2.xyz` | **new**; flat; 50 atoms |

**Flat recompute (2026-07-24):** USER removed z-corrugation (all atoms z=0). Gallery via CLI:

`python run_spm.py panel-fukui --outdir debug/fdbm_fukui_panel_flat --h-min 2.5 --h-max 5.5 --h-step 0.2`

→ `debug/fdbm_fukui_panel_flat/<mol>/compare_cube_stock_prolonged.png` (3×df + 3×Fz). Status investigating.

**Campaign (same pattern as PTCDA stock vs SA):** for each molecule

1. **DFT-cube FDBM reference** — `get_density_from_cube` / `build_fdbm_grid_from_cubes` on `rho_N` (+ tip as usual); optional ESP cross-check via `esp_N`.  
2. **DFTB+ FDBM stock** — 3ob (or mio) projection.  
3. **DFTB+ FDBM extended / SA Slater** — dual basis: stock Δρ→ES, prolonged ρ→Pauli only.  
4. L2: ρ / \(E_\mathrm{pauli}\) / Fz / df panels; refit \(A,\beta\) per molecule or site class when needed.

Do **not** assume cubes live under legacy `jobs/results/` — these folders sit **directly** under `pyscf_fukui_cluster/`. Older GPAW/pySCF compare scripts still point at `jobs/results/`; update paths when wiring this panel.

**Related:** transferability / site \(A,\beta\) → `Pauli_A_beta_KrigingTransferability.md`; cube↔DFTB ES caveats → `Import_KrigingGridFF.md`.

### Open issues from USER review of `debug/fdbm_fukui_panel/` (2026-07-23) — notes only, no fix yet

Full write-up: **`doc/Reports/Fukui_FDBM_panel_notes_2026-07-23.md`**.

1. **Cube FDBM ES (esp. PTCDA):** DFTB prolonged looks reasonable; **cube row** looks overly strong / asymmetric ES. Suspect all-electron ρ − Gaussian ρ_NA (panel did **not** use pyridine clamp→compact-NA). Tip is DFTB CO × cube sample. Investigate after commit — do not mark Done.
2. **df vs Fz height shift ~1.4 Å:** Chemical contrast / bond sharpening in **df** ~4.3–5.3 Å (sharpening ~4.5) vs **Fz** ~2.9–3.9 (sharpening ≲3.0). Panel used **`amp=1.0` Å** peak → df(\(h\)) mixes Fz over \([h-\mathrm{amp},\,h+\mathrm{amp}]\); closest approach ≈ \(h-\mathrm{amp}\). Primary explanation = oscillation amplitude; also coarse `h_step=0.4`, relaxed-only Fz (no Fz_unrelax row), per-image clim. Next: dense Δz=0.1 in 4.3–5.3 window; 3-row Fz_u/Fz_r/df like pyridine.
3. **CLI / per-image strips:** `run_spm.py`, `user_guide/SPM_CLI.md`, `*/per_image/` — commit before further physics fixes.

**CLI:** `python run_spm.py panel-fukui` · replot: `python run_spm.py replot-panel`

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
| ~ | Extend stock-vs-SA FDBM panel to Fukui set: azaindol_(iso)dimer, benzoicacid/amid dimers (+ pentacene) — **ran** cube vs stock vs default Slater-tail prolonged → `debug/fdbm_fukui_panel/` (investigating; not SA-refit yet; not Done) |

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
