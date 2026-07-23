# Task: Orbital-resolved STM — stock vs extended Slater (+ DFT cubes)

**Status:** investigating  
**Priority:** P1 (conference-relevant STM demos)  
**Depends on:** `doc/Tasks/ProlongedRadialBasis_DFTB.md` (prolonged WFC for vacuum LDOS)  
**Related:** `doc/Tasks/DysonOrbitals_DFTB_STM.md` (optional Dyson ladder — **not** required for this campaign)  
**Parent:** `doc/Tasks/RepoConsolidation.md`

## Objective

Systematic **orbital-resolved STM** comparison for several molecules across:

**Primary AFM/FDBM density panel (already on disk):** see `ProlongedRadialBasis_DFTB.md` § *Reference molecule panel* — pySCF PBE/def2-SVP under `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/` (`pentacene`, `PTCDA`, `azaindol_dimer`, `azaindol_isodimer`, `benzoicacid_dimer`, `benzoicamid_dimer`). Those are **total ρ / ESP**, not frontier MO cubes — reuse geometries; still need HOMO/LUMO cubes for STM.

**STM molecule set (at least):** **pentacene**, **PTCDA**; optionally azaindol dimers / benzoic* dimers / pyridine.

| Channel | Source | Role |
|---------|--------|------|
| DFTB **mio-1-1** | stock radial WFC | baseline confined STO |
| DFTB **3ob-3-1** | stock radial WFC | baseline (often production AFM) |
| DFTB **extended / SA Slater** | prolonged projection basis | vacuum tails for HOMO/LUMO maps |
| **DFT reference** | pySCF frontier MO **cube** files | morphological / nodal truth |

Optional stretch (conference-optional): project pySCF MOs from LCAO coefficient matrix \(C\) onto a grid (parity vs cubes). Nice-to-have, not blocking.

## Why

Stock 3ob/mio cutoffs kill vacuum LDOS 1–4 Å above the plane — exactly where STM tunnels. Prolonged / SA Slater already helps FDBM Pauli (`ProlongedRadialBasis_DFTB.md`, PTCDA AFM strip). Same projection bug hurts STM HOMO/LUMO contrast. Need a **molecule panel + basis panel** with DFT cubes as reference, not anecdotal single maps.

## Current repo inventory

| Piece | Path | Status |
|-------|------|--------|
| GPU orbital / STM kernels | `kernels/LCAO_grid.cl`, `kernels/LCAO_STM.cl` | active, little L0 coverage |
| Density / orbital projector | `spammm/quantum/DFTB/Grid_dftb.py` | `project_orbital_*`, dense DM |
| STM helpers | `spammm/SPM/AFM_utils.py` — `compute_stm`, `compute_bond_resolved_stm` | active |
| ModularPipeline S6 | `spammm/SPM/ModularPipeline.py` | STM stage exists |
| GUI STM/Orbitals pane | `spammm/GUI/AFMExtension.py` | widgets often empty / broken (human ToDo) |
| DFTB waveplot → cubes | `spammm/quantum/DFTB_utils.py` — `run_waveplot`, `read_cube*` | exists |
| Prolonged / SA Slater | `basis_optimizer.py`, `DFTBplusParser.make_slater_tail_species_list`, `doc/DFTB_basis_fit.md` | AFM-proven; STM wiring incomplete |
| pySCF utils | `spammm/quantum/pySCF_utils.py` | SCF/opt; **no frontier-cube export yet** |
| Dyson (later) | `DysonOrbitals_DFTB_STM.md` | do **not** block this task |

**Gap:** no campaign script that (i) runs HOMO/LUMO maps for mio / 3ob / prolonged on the same grid, (ii) dumps pySCF HOMO/LUMO cubes, (iii) produces L2 side-by-side panels for pentacene + PTCDA.

## Work plan

1. **pySCF frontier cubes (DFT ref)**  
   - RKS/PBE (or USER-chosen functional/basis) on flat geometries.  
   - Write Gaussian cubes for HOMO and LUMO (optionally HOMO−1 / LUMO+1).  
   - Archive under `tests/ref_data/` or `debug/stm_orbital_compare/<mol>/pyscf/` with grid metadata (origin, step, shape).  
   - Prefer reuse of existing cube I/O (`DFTB_utils.read_cube_with_grid`, `DFTBplusParser.read_cube`).

2. **DFTB STM panel (mio / 3ob / prolonged)**  
   - Same molecule geometries as DFT.  
   - Project selected MOs with stock WFC vs prolonged / SA Slater (`projection_basis_ang` / corrected WFC).  
   - Dual-basis rule for **density** AFM does **not** apply to STM ψ — prolonged radial is the imaging object.  
   - Heights: fixed vacuum slice(s) matching typical BR-STM / constant-height (~2–4 Å above plane).

3. **Compare harness (L2 primary)**  
   - `tests/SPM/testplot_stm_basis_compare.py` (name flexible): rows = molecules; columns = mio | 3ob | prolonged | pySCF cube.  
   - Metrics (L0 where cheap): nodal plane count / sign correlation vs DFT cube resampled to same grid; max-|ψ| location; optional RMSE on normalized |ψ|².

4. **Optional: pySCF \(C\) → grid**  
   - If time: evaluate MOs from pySCF MO coeffs on the same AO basis as cubes → parity check cube vs LCAO eval.  
   - Defer without guilt if conference deadline bites.

5. **GUI**  
   - Only if panes already work: expose basis combo for STM. Else leave to human ToDo (empty STM widgets).

## Deliverables

- [x] pySCF HOMO/LUMO cubes for pentacene + PTCDA (+ note functional/basis) — `debug/stm_orbital_compare/<mol>/pyscf/` (PBE/def2-SVP GPU)
- [x] DFTB STM maps: mio, 3ob, prolonged on identical scan grids — `debug/stm_orbital_compare/`
- [x] L2 gallery + short report under `doc/Reports/` — `doc/Reports/STM_ExtendedBasis_OrbitalCompare.md`
- [x] L0 smoke: at least one molecule, finite non-zero HOMO map, prolonged ≠ stock at vacuum height — `tests/SPM/test_stm_basis_compare.py`
- [x] Cross-links from `ProlongedRadialBasis_DFTB.md` and `spammm/SPM/README.md`

**Status:** investigating — awaiting USER review of L2 panels. Do **not** mark Done without confirmation.

## Acceptance

- USER reviews side-by-side panels (nodal structure / vacuum intensity).  
- Do **not** mark Done without USER confirmation.  
- Dyson Levels 2–3 explicitly out of scope.

## Out of scope

- Full NEGF / GF-STM transport.  
- Charge-state Dyson orbitals (`DysonOrbitals_DFTB_STM.md`).  
- Fitting prolonged basis from scratch (reuse existing SA / Slater-tail tools).
