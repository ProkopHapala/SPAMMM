# Wiring `LCAO_STM_FGR.cl`

## Recommended execution path

1. Pack one tip MO and one sample MO atom-major as `[px, py, pz, s]`.
2. Upload custom long-tail transfer tables `H(R)` and `S(R)` once.
3. For a selected elastic tunnelling energy `E_tunnel`, run
   `build_stm_transfer_sk_tables` once to form `tau(R,E)=H(R)-E*S(R)`.
4. For ordinary real molecular orbitals, run `stm_fgr_sk_tau_scan_real`.
5. Read `out_M_M2[ip].y = M(ip)^2` as the unbroadened state-pair spatial weight.

The complex kernel `stm_fgr_sk_tau_scan` is needed only for genuinely complex
orbitals, for example Bloch states away from Gamma, spin-orbit wavefunctions,
or externally phase-dressed states. `stm_fgr_sk_hs_scan` is a debug/reference
kernel and should not be the production path.

## Required buffers

### Geometry

- `tip_centers[n_points] : float4`
- `tip_pos_rel[ntip_atoms] : float4`
- `smp_pos[nsmp_atoms] : float4`

### Atom types and pair mapping

- `tip_atom_type[ntip_atoms] : int`
- `smp_atom_type[nsmp_atoms] : int`
- `pair_map[n_tip_types*n_sample_types] : int`

`pair_map[type_t*n_sample_types + type_s]` gives the ordered tip-sample table
index. Use `-1` for a missing pair.

### Molecular-orbital coefficients

Fast real kernel:

- `c_tip[4*ntip_atoms] : float`
- `c_smp[4*nsmp_atoms] : float`

Complex kernel:

- `c_tip[4*ntip_atoms] : float2`
- `c_smp[4*nsmp_atoms] : float2`

Every atom block is `[px,py,pz,s]`. Zero-pad absent orbitals.

### Radial SK tables

All ordered pair tables share one uniform radial grid

`r_i = r_grid0 + i/inv_dr`, `i=0..n_r-1`.

For each pair and radial point store

- `H4 = (Hss_sigma, Hsp_sigma, Hps_sigma, Hpp_sigma)`
- `Hpp_pi`
- `S4 = (Sss_sigma, Ssp_sigma, Sps_sigma, Spp_sigma)`
- `Spp_pi`

Flattening:

`index = pair*n_r + i`.

The directed axis is `u=(R_sample-R_tip)/R`. Therefore `sp` means
`<s_tip|X|p_sample,u>` and `ps` means `<p_tip,u|X|s_sample>`. Store their
actual signed values; do not take absolute values and do not force them equal.

## Long-tail tunnelling basis

Use the original DFTB MO coefficients only as the nodal/phase weights, but
define a separate transfer basis

`chi_tilde_A,nlm(r) = N r^(n-1) exp(-zeta_A,l r) Y_lm(rhat)`.

Generate the cross-electrode tables from these functions, not from the original
short-ranged mio/3ob orbitals.

The consistent frozen, non-SCC definitions are

`S_AB(R) = <chi_tilde_A|chi_tilde_B>`

and

`H0_AB(R) = <chi_tilde_A| -1/2 nabla^2 + v_A^0 + v_B^0 |chi_tilde_B>`,

where `v_A^0` and `v_B^0` are fixed neutral-atom reference potentials. No
charge iteration and no SCC correction is added.

The scan uses only

`tau_AB(R,E) = H0_AB(R) - E*S_AB(R)`.

The original isolated-molecule DFTB Hamiltonian is not passed into the scan
kernel. It was needed only to obtain the MO coefficients and state energies.

## Practical table-generation levels

### Level A: recommended physical version

Numerically integrate the five two-centre channels for the chosen long-tail
STOs using fixed neutral-atom effective potentials. Do this offline once per
ordered element/type pair. This is the direct analogue of DFTB0 table
generation, but with your tunnelling orbitals.

### Level B: very fast prototype

Generate the STO overlap tables exactly, then use a frozen extended-Hueckel
closure

`H_gamma(R) = K_gamma * 0.5*(epsilon_A,l + epsilon_B,l') * S_gamma(R)`.

The `K_gamma` factors may be fitted per channel or pair. This is internally
consistent enough for an initial BR-STM contrast test, but it makes H and S
share the same radial shape.

### Level C: direct empirical transfer tables

Fit `tau_gamma(R,E_ref)` itself to reference Bardeen/DFT data or to a smooth
long-range form and use the production tau kernel. This is fastest and avoids
an ambiguous energy zero, but it no longer separately identifies H and S.

## Energy zero

`H` and `E_tunnel` must share an energy zero. If custom H tables are generated
with vacuum at zero, use the tunnelling electron energy relative to vacuum. If
an atomic/DFTB-like zero is used, align the state energy to that same zero.
A constant shift applied consistently to H and E cancels in `H-E*S`.

For a state pair with nearly equal energies, use `E_tunnel=E_sample` for the
sample-partition derivation, or `(E_tip+E_sample)/2` as a symmetric numerical
choice.

## What is deliberately omitted

- SCC charge-response term
- electrostatic polarization induced by the approaching tip
- Green functions and Dyson resummation
- diagonalization in the scan
- density-of-states factors, occupations, and bias integration

The kernel returns the state-pair quantity `|M|^2`; the host may later multiply
and sum it over states and energies.
