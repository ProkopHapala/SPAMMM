# Porting guide: FGR STM transfer `M = c_t†(H − E·S)c_s` (pyOpenCL)

How the Fermi-golden-rule STM path works in SPAMMM and what to copy to
reimplement it in another pyOpenCL codebase.

## Physics

Elastic weak-coupling tunneling between a tip MO and a sample MO, in a
nonorthogonal LCAO basis:

    M_ts(E) = c_t† [ H_TS − E·S_TS ] c_s ,   I ∝ |M|²

- `E` = tunnelling energy = eigenvalue of the selected sample MO (elastic, same
  energy zero and unit as `H`, Hartree here).
- `H−E·S` is invariant under a common energy-zero shift; `H` alone is not.
- Legacy baseline in this repo is `overlap_exp`: artificial `exp(-β(r−r0))`
  Slater–Koster "overlap" (i.e. `H'≈1`). It over-weights the vacuum halo —
  that is the thing FGR fixes.

## Pipeline

    DFTB eigvecs ──► per-atom sp coeffs [px,py,pz,s] ──┐
                                                      ├──► scan kernel ──► I(x,y)
    prolonged STOs ──► S(R),H(R) SK tables ──► τ=H−E·S ┘    (1 work-item / pixel)

## File map

| Piece | File | Key symbols |
|---|---|---|
| OpenCL kernels | `kernels/LCAO_STM_FGR.cl` | `build_stm_transfer_sk_tables`, `stm_fgr_sk_tau_scan_real`, `stm_fgr_sk_tau_scan` (complex), `stm_fgr_sk_hs_scan` (debug) |
| Legacy overlap kernel | `kernels/LCAO_grid.cl` | `mo_overlap_points_exp_sk` |
| Host bindings (pyOpenCL) | `spammm/quantum/DFTB/Grid_dftb.py` | `stm_fgr_sk_tau_scan_real` (~L1004), `build_stm_transfer_sk_tables_gpu` (~L1126), kernel load at `_load_kernels` (~L220) |
| SK table builder (CPU, NumPy) | `spammm/quantum/DFTB/DFTBplusParser.py` | `build_longtail_eh_sk_tables` (~L1462), `sto_two_center_sk_channels` (~L1378), `prolonged_sto_params`, `read_skf_onsite_sp`, `evec_to_kernel_coeffs` (~L1144) |
| Per-slice wrappers | `spammm/SPM/AFM_utils.py` | `project_mo_stm_fgr_slice` (~L6290), `project_mo_stm_fgr_points` (~L5666), `_stm_fgr_prepare_tables` (~L6244), `project_mo_stm_sk_slice` (overlap, ~L6212), `STM_TIP_ORBITALS` (~L6204) |
| Compare driver | `spammm/SPM/stm_compare.py` | `run_fgr_transfer_compare` (~L551), `run_br_stm_fgr_compare` (~L723) |
| CLI | `run_spm.py` | `stm fgr`, `stm br-fgr`, `stm br --stm-mode fgr` |
| Test | `tests/SPM/test_stm_fgr_compare.py` | table `Sps≈−Ssp` smoke + GPU scan `I_τ ≠ overlap_exp` |
| Design docs | `doc/Ideas/LCAO_STM_FGR_WIRING.md` (buffer layout, Levels A/B/C), `doc/TopicalAudit/STM_FGR_Transfer.md`, `doc/Reports/STM_FGR_Transfer_H_ES_2026-07-29.md`, `doc/Ideas/STM_perturbation_H.chat.md` (derivation) | |

## Kernel contract (`LCAO_STM_FGR.cl`, self-contained — copy it verbatim)

### Coefficient packing

Per atom, 4 floats: `[px, py, pz, s]` — **not** the DFTB `[s,px,py,pz]` order.
H and other s-only atoms are zero-padded in the p slots. Host-side remap:
`evec_to_kernel_coeffs` (DFTBplusParser.py). Tip orbitals are unit vectors from
`STM_TIP_ORBITALS` (`'s'`, `'px'`, `'py'`, `'pz'`).

### Radial tables (the SK contract)

For every **ordered** (tip type, sample type) pair and each radial grid point,
five signed channels:

    tau4   = ( τ_ss, τ_spσ, τ_psσ, τ_ppσ )   float4
    tau_pi =   τ_ppπ                          float

with `u = (R_sample − R_tip)/|…|` and **directed** definitions
`X_sp = ⟨s_T|X|p_{S,u}⟩`, `X_ps = ⟨p_{T,u}|X|s_S⟩`. Signs live in the tables —
do **not** add hard-coded minus signs in the kernel (invariant tested by
`Sps ≈ −Ssp` for identical STOs). `pair_map[tip_type*n_sample_types + smp_type]`
→ compact pair index (`−1` = no interaction). All pairs share one uniform grid
`r_grid0 + i·dr`, `n_r` points; linear interpolation in the kernel.

### Scan kernel `stm_fgr_sk_tau_scan_real`

One work-item per tip position. Loops tip atoms × sample atoms; per pair:
`pair_map` lookup → `r` → lerp τ → cosine `taper` → `stm_contract_sp_pair_real`
(the 4×4 SK block contraction reduced to 5 products). Output per pixel:
`float4 = (M, M², npair_used, 0)`. No DOS, occupation, or `2π/ħ` — host's job.

### Table build kernel `build_stm_transfer_sk_tables`

Trivially elementwise: `τ = H − E·S` for the whole `[n_pair·n_r]` table. Rebuild
whenever `E_tunnel` changes; a CPU one-liner is equally fine.

## Table generation (Level B — what's implemented)

`build_longtail_eh_sk_tables` in DFTBplusParser.py:

- `S_γ(R)` from numerical 2-centre integrals of **prolonged** (Slater-tail) STOs
  (`sto_two_center_sk_channels`, cylindrical grid, axisymmetric → 2π ρ dρ dz).
- `H_γ(R) = K · ½(ε_A,l + ε_B,l′) · S_γ(R)` — extended Hückel /
  Wolfsberg–Helmholtz, `K=1.75`, on-site `ε` from homoatomic `.skf` files
  (`read_skf_onsite_sp`).
- Prolonged STO params `(N, ζ)` per element: `make_slater_tail_species_list` +
  `prolonged_sto_params` (longer tails than stock mio/3ob AOs control vacuum
  decay; MO coeffs keep only nodal/phase pattern — OK for image shape, see
  kernel header caveats).

Level A (frozen `H⁰` from the real Hamiltonian) and Level C (empirical tables)
are designed but **not** implemented — see the wiring doc.

## Running it here

```bash
python run_spm.py stm fgr --molecule pentacene,PTCDA      # → debug/stm_fgr_compare/<mol>/
python run_spm.py stm br-fgr --molecule pentacene        # → debug/stm_br_fgr_compare/ (with PP-AFM tip displacement)
pytest tests/SPM/test_stm_fgr_compare.py                 # L0 smoke (GPU test is -m gpu --slow)
```

Example artifacts: `debug/stm_fgr_compare/pentacene/fgr_compare_z3.0_pentacene.png`
+ `SUMMARY.out` (4 columns: `overlap_exp | I_S | I_H | I_τ`).

## Porting recipe (minimal)

1. Copy `kernels/LCAO_STM_FGR.cl` (self-contained: taper, lerp, complex+real
   contractions, all 4 kernels).
2. Provide `(npts,3)` tip positions, `(ntip,3)` tip atom offsets (0 for a point
   tip), `(nsmp,3)` sample positions, per-atom `[px,py,pz,s]` coeffs, int32
   atom-type arrays, `pair_map`, and the two τ tables.
3. Build `H,S` tables however you like — the kernel doesn't care where τ comes
   from. If you already have tip–sample `H_TS`,`S_TS` blocks in your own code,
   just tabulate the 5 SK channels or, simpler still, keep full per-pair
   matrices and simplify the kernel.
4. Call `build_stm_transfer_sk_tables(E_tunnel = ε_sample_MO)` →
   `stm_fgr_sk_tau_scan_real` → reshape `M²` to `(nx, ny)`.
5. Validation: `Sps ≈ −Ssp` for identical STOs; `I_τ ≥ 0`, finite, and
   morphologically different from pure `I_S` (unless `H ∝ S`, the Level-B EH
   limit where they're similar — expected, not a bug).

## Caveats

- Orbital order `[px,py,pz,s]` everywhere on the GPU side; remap on host.
- `tip_pos_rel` lets the tip be a multi-atom cluster; a point tip is one atom at
  the origin with a unit-orbital coeff vector.
- `rcut` + `taper_w` (cosine taper) remove hard-cutoff ring artifacts — keep them.
- Real kernel assumes real MO coeffs (Γ-point non-magnetic); use
  `stm_fgr_sk_tau_scan` for complex coeffs (e.g. k-sampled, transport codes).
- Degenerate MOs: sum `I` over the cluster on the host (`degen_cluster` in
  `run_fgr_transfer_compare`) — single-MO maps of a degenerate subspace are
  arbitrary-rotation artifacts.
