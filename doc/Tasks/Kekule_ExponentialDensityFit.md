# Task: Kekulé π density → exponential RI / density fitting

**Status:** investigating  
**Priority:** P2 (method / conference stretch)  
**Related:** `spammm/topology/KekulePure.py`, `doc/Tasks/ProlongedRadialBasis_DFTB.md`, FDBM density path in `Grid_dftb.py`  
**DFT refs:** pySCF / DFTB total density (existing cube & projector tools)

## Objective

Build a **semiclassical density model** for π-systems:

1. Run the **Kekulé solver** to obtain on-site π occupations and bond π-density matrix elements (bond orders).  
2. Place **exponential (Slater-like) radial functions** at **atom centers** and **bond midpoints** — resolution-of-identity / density-fitting style:

\[
\rho_\pi(\mathbf{r}) \approx \sum_i q_i\,f_i(|\mathbf{r}-\mathbf{R}_i|) + \sum_{\langle ij\rangle} q_{ij}\,f_{ij}(|\mathbf{r}-\mathbf{R}_{ij}|)
\]

with \(f\sim e^{-\zeta r}\) (or short multi-ζ).  
3. Compare to **DFT / DFTB** total (or π-projected) density on the same molecules — and optionally feed the fitted \(\rho\) into **FDBM Pauli** as a cheap tip/sample density surrogate.

## Why

Full DFTB/pySCF SCF + LCAO projection is correct but heavy for interactive / many-molecule AFM. Kekulé already gives chemically meaningful π bond orders on `AtomicGraph`. Fitting a few exponentials at atoms+bonds is the classical RI idea: cheap \(\rho\) with correct nodal / bond topology for PAHs.

## Current inventory

| Piece | Path | Role |
|-------|------|------|
| Kekulé π bond orders | `spammm/topology/KekulePure.py`, GUI `KekuleExtension.py` | on-site / bond π structure |
| Graph SSOT | `AtomicGraph` | atom/bond geometry for centers |
| STO / Slater projectors | `Grid_dftb.py`, `basis_optimizer.py`, `DFTBplusParser` | radial primitives + SA fit vs DFT ρ |
| Cube I/O | `DFTB_utils.read_cube*`, `AFM_utils.get_density_from_cube` | DFT reference ρ |
| Density compare examples | `examples/density_comparison/` | optimize / compare Pauli patterns |

**Gap:** no pipeline that maps Kekulé \(q_i, q_{ij}\) → atom+bond exponential RI density and scores vs DFT/DFTB grids.

## Work plan

1. **Extract π charges from Kekulé**  
   - Define SSOT: on-site \(q_i\) from `n_pi` / solver occupations; bond \(q_{ij}\) from π bond orders (normalize so \(\sum q = N_\pi\)).  
   - Document convention in module header (fail loud if graph not π-ready).

2. **Radial primitives**  
   - Start with single-ζ exponentials per element (C, N, O, …) at atoms; one ζ_bond class at midpoints.  
   - Optional: SA / least-squares fit of \(\zeta\) (and amplitudes) vs DFTB or pySCF ρ on a training molecule (reuse `basis_optimizer` patterns).

3. **Project to grid**  
   - CPU NumPy first; GPU later only if needed (`LCAO_grid` / `grids.cl` gather).  
   - Same box / step as FDBM density for fair Pauli overlap tests.

4. **Parity vs DFT / DFTB**  
   - Prefer the **Fukui pySCF panel** (PBE/def2-SVP `rho_N`):  
     `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/{pentacene,PTCDA,azaindol_dimer,azaindol_isodimer,benzoicacid_dimer,benzoicamid_dimer}_PBE_def2-SVP/`  
     with matching `data/xyz/*.xyz` (H-bonded dimers stress bond centers + heteroatoms).  
   - Metrics: ∫ρ, XY slice RMSE / correlation above plane, vacuum tail at \(z\sim 2\)–3 Å.  
   - L2 panels: Kekulé-RI | DFTB stock | DFTB prolonged | pySCF cube.

5. **Optional FDBM hook**  
   - Swap sample (or tip) ρ for RI ρ in Pauli-only channel; compare Fz/df to DFTB FDBM (expect qualitative PAH contrast, not quantitative ES).

## Deliverables

- [ ] Library function: `rho_pi_from_kekule(graph, …) → grid` (or atom/bond coeff table)  
- [ ] Fit / compare script under `tests/SPM/` or `tests/topology/`  
- [ ] L2 gallery + brief note in `doc/Reports/` or topical audit  
- [ ] L0: electron count / finite density / Kekulé-RI ≠ zero on benzene

## Acceptance

- USER reviews density slices vs DFT/DFTB.  
- Do not mark Done without confirmation.  
- Clarify whether RI ρ is for **visualization / Pauli only** vs full SCF replacement (default: Pauli / cheap AFM only).

## Out of scope

- Replacing DFTB SCC energy.  
- σ-framework density (core + σ bonds) unless trivial H-caps added later.  
- Prolonged-basis SCF (separate P0 task) — may share ζ-fitting tools only.
