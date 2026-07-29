// LCAO_STM_FGR.cl
// ============================================================================
// First-order Fermi-golden-rule STM with nonorthogonal LCAO orbitals
// and tabulated Slater-Koster transfer integrals.
//
// This file intentionally contains NO Dyson equation, NO Green's function,
// NO diagonalization, and NO SCC/self-consistent polarization.  It evaluates
// only the weak-coupling transfer matrix element between already known tip and
// sample molecular orbitals.
//
// OpenCL orbital order used here is the SAME as in stm_gf_dyson_2mol_mo_scan:
//
//                       [ px, py, pz, s ]
//
// IMPORTANT HOST-SIDE CAVEAT:
// Some other kernels / DFTB exports use [s, px, py, pz].  The host currently
// remaps coefficients.  Keep doing that.  This kernel deliberately does not
// attempt to detect or repair ordering mismatches.  Hydrogen and other atoms
// without p orbitals must be represented by zero-padded p coefficients.
//
// Units:
//   H and E_tunnel must use the same energy unit and the same energy zero.
//   S is dimensionless.  Positions, r_grid0, dr and rcut use the same length
//   unit (normally Angstrom in this project).
// ============================================================================

#ifndef LCAO_STM_FGR_CL
#define LCAO_STM_FGR_CL

// ----------------------------------------------------------------------------
// Complex helpers: float2 = (real, imaginary)
// ----------------------------------------------------------------------------
inline float2 stm_cadd(const float2 a, const float2 b) {
    return (float2)(a.x + b.x, a.y + b.y);
}

inline float2 stm_cmul(const float2 a, const float2 b) {
    return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline float2 stm_cscale(const float2 a, const float x) {
    return (float2)(a.x*x, a.y*x);
}

inline float2 stm_cconj(const float2 a) {
    return (float2)(a.x, -a.y);
}

inline float stm_cnorm2(const float2 a) {
    return a.x*a.x + a.y*a.y;
}

// Linear interpolation of a uniformly sampled radial table.
// Returns 0 outside [r_grid0, r_grid0 + dr*(n_r-1)].
inline int stm_sk_grid_coord(
    const float r,
    const float r_grid0,
    const float inv_dr,
    const int n_r,
    int* i0,
    float* t
) {
    if (n_r < 2 || inv_dr <= 0.0f) return 0;

    const float x = (r - r_grid0) * inv_dr;
    const float xmax = (float)(n_r - 1);
    if (x < 0.0f || x > xmax) return 0;

    int i = (int)floor(x);
    float f = x - (float)i;
    if (i >= n_r - 1) {
        i = n_r - 2;
        f = 1.0f;
    }
    *i0 = i;
    *t  = f;
    return 1;
}

inline float4 stm_lerp4(const float4 a, const float4 b, const float t) {
    return a + t*(b - a);
}

inline float stm_lerp1(const float a, const float b, const float t) {
    return a + t*(b - a);
}

// ----------------------------------------------------------------------------
// SK table convention
// ----------------------------------------------------------------------------
// For every ORDERED pair (tip atom type A, sample atom type B), and every
// radial grid point, store five signed two-centre channels:
//
//   sk4 = ( X_ss_sigma, X_sp_sigma, X_ps_sigma, X_pp_sigma )
//   skpi = X_pp_pi
//
// where X is H, S, or tau = H - E*S.
//
// Let u = (R_sample - R_tip)/|R_sample - R_tip|.  The stored axial channels
// are DEFINED operationally by
//
//   X_ss       = < s_T       | X | s_S       >
//   X_sp_sigma = < s_T       | X | p_{S,u}   >
//   X_ps_sigma = < p_{T,u}   | X | s_S       >
//   X_pp_sigma = < p_{T,u}   | X | p_{S,u}   >
//   X_pp_pi    = < p_{T,v}   | X | p_{S,v}   >,  v perpendicular to u.
//
// The signs of sp and ps are therefore part of the tables.  Do NOT apply an
// additional hard-coded sign in the scan kernel.  For identical real atomic
// orbitals one normally obtains X_ps_sigma = -X_sp_sigma with this directed
// axis convention, but heteronuclear / differently confined orbitals should
// still be tabulated explicitly.
//
// pair_map is indexed as
//
//   pair_map[ tip_type * n_sample_types + sample_type ]
//
// and returns a compact pair-table index, or -1 if the interaction is absent.
// Every pair table uses the same uniform radial grid and occupies n_r entries.
// ----------------------------------------------------------------------------

// Contract one real SK transfer block with complex MO coefficients.
//
// For one tip atom T and one sample atom S, define
//
//   |psi_T> = c_Ts |s_T> + sum_i c_Ti |p_Ti>
//   |psi_S> = c_Ss |s_S> + sum_j c_Sj |p_Sj>.
//
// With u pointing from tip to sample, the SK block is
//
//   tau_ss = tau_ss_sigma
//   tau_s,pj = u_j tau_sp_sigma
//   tau_pi,s = u_i tau_ps_sigma
//   tau_pi,pj = tau_pp_pi delta_ij
//             + (tau_pp_sigma - tau_pp_pi) u_i u_j.
//
// Hence the complete 4x4 contraction can be evaluated without constructing
// the matrix explicitly:
//
// M_TS = conj(c_Ts) tau_ss c_Ss
//      + conj(c_Ts) tau_sp (u . c_Sp)
//      + (conj(c_Tp) . u) tau_ps c_Ss
//      + tau_pp_pi (conj(c_Tp) . c_Sp)
//      + (tau_pp_sigma-tau_pp_pi)
//                    (conj(c_Tp) . u)(u . c_Sp).
//
// This reduces each atom pair to five complex products after one radial lookup.
inline float2 stm_contract_sp_pair(
    const float3 u,
    const float4 tau4,       // (ss, sp, ps, pp_sigma)
    const float tau_pp_pi,
    const float2 ct_px,
    const float2 ct_py,
    const float2 ct_pz,
    const float2 ct_s,
    const float2 cs_px,
    const float2 cs_py,
    const float2 cs_pz,
    const float2 cs_s
) {
    // u . c_Sp
    const float2 us = (float2)(
        u.x*cs_px.x + u.y*cs_py.x + u.z*cs_pz.x,
        u.x*cs_px.y + u.y*cs_py.y + u.z*cs_pz.y
    );

    // conj(c_Tp) . u
    const float2 tu = (float2)(
         u.x*ct_px.x + u.y*ct_py.x + u.z*ct_pz.x,
        -u.x*ct_px.y - u.y*ct_py.y - u.z*ct_pz.y
    );

    // conj(c_Tp) . c_Sp
    float2 pp = stm_cmul(stm_cconj(ct_px), cs_px);
    pp = stm_cadd(pp, stm_cmul(stm_cconj(ct_py), cs_py));
    pp = stm_cadd(pp, stm_cmul(stm_cconj(ct_pz), cs_pz));

    float2 out = stm_cscale(stm_cmul(stm_cconj(ct_s), cs_s), tau4.x);
    out = stm_cadd(out, stm_cscale(stm_cmul(stm_cconj(ct_s), us), tau4.y));
    out = stm_cadd(out, stm_cscale(stm_cmul(tu, cs_s), tau4.z));
    out = stm_cadd(out, stm_cscale(pp, tau_pp_pi));
    out = stm_cadd(out, stm_cscale(stm_cmul(tu, us), tau4.w - tau_pp_pi));
    return out;
}


// Real-valued specialization of the same 4x4 contraction.  This is the
// preferred path for ordinary finite, non-magnetic DFTB molecules whose MO
// coefficients can be chosen real.
inline float stm_contract_sp_pair_real(
    const float3 u,
    const float4 tau4,
    const float tau_pp_pi,
    const float ct_px,
    const float ct_py,
    const float ct_pz,
    const float ct_s,
    const float cs_px,
    const float cs_py,
    const float cs_pz,
    const float cs_s
) {
    const float us = u.x*cs_px + u.y*cs_py + u.z*cs_pz;
    const float tu = u.x*ct_px + u.y*ct_py + u.z*ct_pz;
    const float pp = ct_px*cs_px + ct_py*cs_py + ct_pz*cs_pz;

    return tau4.x*ct_s*cs_s
         + tau4.y*ct_s*us
         + tau4.z*tu*cs_s
         + tau_pp_pi*pp
         + (tau4.w - tau_pp_pi)*tu*us;
}

// ============================================================================
// build_stm_transfer_sk_tables()
// ============================================================================
//
// Build the energy-dependent transfer table
//
//                         tau(E) = H - E*S
//
// once for a selected tunnelling energy E_tunnel.  The scan kernel then reads
// only tau, which halves radial-table traffic and avoids repeatedly evaluating
// H-E*S for every atom pair and every pixel.
//
// Physical derivation:
//
// Let |psi_S> be an eigenstate of an isolated sample Hamiltonian H_S,
//
//                         H_S |psi_S> = E |psi_S>.
//
// Partition the coupled system as H = H_S + H', so the first-order transfer
// matrix element is
//
//   M_TS = <psi_T|H'|psi_S>
//        = <psi_T|H-H_S|psi_S>
//        = <psi_T|H|psi_S> - E <psi_T|psi_S>.
//
// In a nonorthogonal LCAO basis this becomes
//
//   M_TS(E) = c_T^dagger [ H_TS - E S_TS ] c_S
//           = c_T^dagger tau_TS(E) c_S.
//
// The combination H-E*S is invariant under a common shift of energy zero:
// H -> H + C*S and E -> E+C.  H alone is not.
//
// This kernel is tiny compared with an STM scan.  It may be run whenever the
// selected energy changes, or the same operation may be performed on the CPU.
// ============================================================================
__kernel void build_stm_transfer_sk_tables(
    const int n_values,                  // n_pair_tables * n_r
    __global const float4* H4,           // (Hss,Hsp,Hps,Hpp_sigma)
    __global const float*  Hpp_pi,
    __global const float4* S4,           // (Sss,Ssp,Sps,Spp_sigma)
    __global const float*  Spp_pi,
    const float E_tunnel,
    __global float4* tau4,
    __global float*  tau_pp_pi
) {
    const int i = get_global_id(0);
    if (i >= n_values) return;
    tau4[i]      = H4[i] - E_tunnel*S4[i];
    tau_pp_pi[i] = Hpp_pi[i] - E_tunnel*Spp_pi[i];
}


// ============================================================================
// stm_fgr_sk_tau_scan_real()
// ============================================================================
//
// Fastest production variant for real molecular-orbital coefficients.
// It evaluates exactly the same first-order matrix element as the complex
// kernel below,
//
//              M(E) = c_T^T [H_TS - E S_TS] c_S,
//
// but stores one float per coefficient and performs only real arithmetic.
// For isolated non-magnetic molecules at Gamma, DFTB H and S are real and the
// generalized eigenvectors can be chosen real, so this should normally be the
// default BR-STM kernel.
//
// Output:
//   out_M_M2[ip] = (M, M*M, number_of_used_atom_pairs, 0).
// ============================================================================
__kernel void stm_fgr_sk_tau_scan_real(
    const int n_points,
    __global const float4* tip_centers,
    __global const float4* tip_pos_rel,
    __global const float4* smp_pos,
    __global const int* tip_atom_type,
    __global const int* smp_atom_type,
    const int n_sample_types,
    __global const int* pair_map,
    __global const float* c_tip,          // [4*ntip_atoms], [px,py,pz,s]
    __global const float* c_smp,          // [4*nsmp_atoms], [px,py,pz,s]
    const int ntip_atoms,
    const int nsmp_atoms,
    __global const float4* tau4_table,
    __global const float* tau_pp_pi_table,
    const int n_r,
    const float r_grid0,
    const float inv_dr,
    const float rcut,
    const float amplitude_scale,
    __global float4* out_M_M2
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 tip_center = tip_centers[ip].xyz;
    const float rcut2 = rcut*rcut;
    float M = 0.0f;
    int npair_used = 0;

    for (int ia = 0; ia < ntip_atoms; ia++) {
        const int type_t = tip_atom_type[ia];
        if (type_t < 0) continue;

        const int it0 = 4*ia;
        const float ct_px = c_tip[it0    ];
        const float ct_py = c_tip[it0 + 1];
        const float ct_pz = c_tip[it0 + 2];
        const float ct_s  = c_tip[it0 + 3];
        const float wt = ct_px*ct_px + ct_py*ct_py + ct_pz*ct_pz + ct_s*ct_s;
        if (wt == 0.0f) continue;

        const float3 rt = tip_center + tip_pos_rel[ia].xyz;

        for (int ja = 0; ja < nsmp_atoms; ja++) {
            const int type_s = smp_atom_type[ja];
            if (type_s < 0) continue;

            const int pair = pair_map[type_t*n_sample_types + type_s];
            if (pair < 0) continue;

            const float3 d = smp_pos[ja].xyz - rt;
            const float r2 = dot(d, d);
            if (r2 > rcut2 || r2 < 1.0e-16f) continue;

            const float r = sqrt(r2);
            int ir;
            float fr;
            if (!stm_sk_grid_coord(r, r_grid0, inv_dr, n_r, &ir, &fr)) continue;

            const int k0 = pair*n_r + ir;
            const float4 tau4 = stm_lerp4(tau4_table[k0],
                                           tau4_table[k0 + 1], fr);
            const float tau_pi = stm_lerp1(tau_pp_pi_table[k0],
                                            tau_pp_pi_table[k0 + 1], fr);

            const int is0 = 4*ja;
            const float cs_px = c_smp[is0    ];
            const float cs_py = c_smp[is0 + 1];
            const float cs_pz = c_smp[is0 + 2];
            const float cs_s  = c_smp[is0 + 3];
            const float ws = cs_px*cs_px + cs_py*cs_py + cs_pz*cs_pz + cs_s*cs_s;
            if (ws == 0.0f) continue;

            const float3 u = d * (1.0f/r);
            M += stm_contract_sp_pair_real(
                u, tau4, tau_pi,
                ct_px, ct_py, ct_pz, ct_s,
                cs_px, cs_py, cs_pz, cs_s
            );
            npair_used++;
        }
    }

    M *= amplitude_scale;
    out_M_M2[ip] = (float4)(M, M*M, (float)npair_used, 0.0f);
}

// ============================================================================
// stm_fgr_sk_tau_scan()
// ============================================================================
//
// Production first-order Fermi-golden-rule STM kernel.
// One OpenCL work item computes one tip position.
//
// INPUT STATES
// ------------
// The host supplies already known molecular-orbital coefficient vectors:
//
//   |psi_T> = sum_mu c_T,mu |chi_T,mu>
//   |psi_S> = sum_nu c_S,nu |chi_S,nu>.
//
// No eigenproblem is solved here.  Coefficients may come directly from a
// previous DFTB calculation and are packed atom-major as
//
//   c[4*iatom + 0..3] = [c_px, c_py, c_pz, c_s].
//
// TRANSFER MATRIX ELEMENT
// -----------------------
// For weak tip-sample coupling, the transition rate between one tip state and
// one sample state follows Fermi's golden rule,
//
//   Gamma_TS(E) = (2*pi/hbar) |M_TS(E)|^2 delta(E_T-E_S),
//
// with
//
//   M_TS(E) = sum_{mu in T,nu in S}
//             conj(c_T,mu) [H_mu,nu - E S_mu,nu] c_S,nu.
//
// The kernel returns
//
//   out_M_M2[ip] = ( Re M, Im M, |M|^2, number_of_used_atom_pairs ).
//
// It does NOT multiply by density of states, occupation factors, 2*pi/hbar,
// or integrate over a bias window.  Those factors belong on the host and do
// not affect the constant-height spatial contrast for one selected state pair.
//
// DELIBERATELY EXTENDED TUNNELLING BASIS
// --------------------------------------
// c_T and c_S may be DFTB MO coefficients obtained with short-ranged mio/3ob
// pseudoatomic orbitals, while tau is generated for deliberately longer
// Slater-type tails.  This is not a variationally self-consistent basis change:
// the coefficients retain mainly the molecular nodal / phase pattern, whereas
// the custom transfer tables control vacuum decay.  For a fixed state pair,
// any missing re-normalization is only an overall intensity factor; it does
// not change the lateral image shape.  If intensities of different orbitals
// are compared quantitatively, normalize the reconstructed long-tail states
// once on the host.
//
// PERFORMANCE
// -----------
// Geometry and SK interpolation are performed once per ATOM pair, not once per
// orbital pair.  The complete 4x4 s,p block is contracted analytically.
// ============================================================================
__kernel void stm_fgr_sk_tau_scan(
    const int n_points,
    __global const float4* tip_centers,   // [n_points], absolute scan positions
    __global const float4* tip_pos_rel,   // [ntip_atoms], relative to tip center
    __global const float4* smp_pos,       // [nsmp_atoms], absolute positions
    __global const int* tip_atom_type,    // [ntip_atoms], compact type index
    __global const int* smp_atom_type,    // [nsmp_atoms]
    const int n_sample_types,
    __global const int* pair_map,         // [n_tip_types*n_sample_types]
    __global const float2* c_tip,         // [4*ntip_atoms], [px,py,pz,s]
    __global const float2* c_smp,         // [4*nsmp_atoms], [px,py,pz,s]
    const int ntip_atoms,
    const int nsmp_atoms,
    __global const float4* tau4_table,    // [n_pair_tables*n_r]
    __global const float* tau_pp_pi_table,
    const int n_r,
    const float r_grid0,
    const float inv_dr,
    const float rcut,
    const float amplitude_scale,
    __global float4* out_M_M2
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 tip_center = tip_centers[ip].xyz;
    const float rcut2 = rcut*rcut;
    float2 M = (float2)(0.0f, 0.0f);
    int npair_used = 0;

    for (int ia = 0; ia < ntip_atoms; ia++) {
        const int type_t = tip_atom_type[ia];
        if (type_t < 0) continue;

        const int it0 = 4*ia;
        const float2 ct_px = c_tip[it0    ];
        const float2 ct_py = c_tip[it0 + 1];
        const float2 ct_pz = c_tip[it0 + 2];
        const float2 ct_s  = c_tip[it0 + 3];

        // Skip an entirely zero-padded atom block.
        const float wt = stm_cnorm2(ct_px) + stm_cnorm2(ct_py)
                       + stm_cnorm2(ct_pz) + stm_cnorm2(ct_s);
        if (wt == 0.0f) continue;

        const float3 rt = tip_center + tip_pos_rel[ia].xyz;

        for (int ja = 0; ja < nsmp_atoms; ja++) {
            const int type_s = smp_atom_type[ja];
            if (type_s < 0) continue;

            const int pair = pair_map[type_t*n_sample_types + type_s];
            if (pair < 0) continue;

            // Directed SK axis: from tip atom to sample atom.
            const float3 d = smp_pos[ja].xyz - rt;
            const float r2 = dot(d, d);
            if (r2 > rcut2 || r2 < 1.0e-16f) continue;

            const float r = sqrt(r2);
            int ir;
            float fr;
            if (!stm_sk_grid_coord(r, r_grid0, inv_dr, n_r, &ir, &fr)) continue;

            const int k0 = pair*n_r + ir;
            const float4 tau4 = stm_lerp4(tau4_table[k0],
                                           tau4_table[k0 + 1], fr);
            const float tau_pi = stm_lerp1(tau_pp_pi_table[k0],
                                            tau_pp_pi_table[k0 + 1], fr);

            const int is0 = 4*ja;
            const float2 cs_px = c_smp[is0    ];
            const float2 cs_py = c_smp[is0 + 1];
            const float2 cs_pz = c_smp[is0 + 2];
            const float2 cs_s  = c_smp[is0 + 3];

            const float ws = stm_cnorm2(cs_px) + stm_cnorm2(cs_py)
                           + stm_cnorm2(cs_pz) + stm_cnorm2(cs_s);
            if (ws == 0.0f) continue;

            const float3 u = d * (1.0f/r);
            M = stm_cadd(M, stm_contract_sp_pair(
                u, tau4, tau_pi,
                ct_px, ct_py, ct_pz, ct_s,
                cs_px, cs_py, cs_pz, cs_s
            ));
            npair_used++;
        }
    }

    M = stm_cscale(M, amplitude_scale);
    out_M_M2[ip] = (float4)(M.x, M.y, stm_cnorm2(M), (float)npair_used);
}

// ============================================================================
// stm_fgr_sk_hs_scan()
// ============================================================================
//
// Reference/debug version of stm_fgr_sk_tau_scan.  It interpolates H and S
// separately and forms tau=H-E*S inside the scan loop.  It is useful for
// validating signs, energy-zero alignment, and cancellation between H and ES.
// For production at a fixed energy, prebuild tau and use stm_fgr_sk_tau_scan.
//
// The physics and state conventions are identical to the production kernel.
// This version is expected to read roughly twice as much SK-table data.
// ============================================================================
__kernel void stm_fgr_sk_hs_scan(
    const int n_points,
    __global const float4* tip_centers,
    __global const float4* tip_pos_rel,
    __global const float4* smp_pos,
    __global const int* tip_atom_type,
    __global const int* smp_atom_type,
    const int n_sample_types,
    __global const int* pair_map,
    __global const float2* c_tip,
    __global const float2* c_smp,
    const int ntip_atoms,
    const int nsmp_atoms,
    __global const float4* H4_table,
    __global const float* Hpp_pi_table,
    __global const float4* S4_table,
    __global const float* Spp_pi_table,
    const int n_r,
    const float r_grid0,
    const float inv_dr,
    const float rcut,
    const float E_tunnel,
    const float amplitude_scale,
    __global float4* out_M_M2
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 tip_center = tip_centers[ip].xyz;
    const float rcut2 = rcut*rcut;
    float2 M = (float2)(0.0f, 0.0f);
    int npair_used = 0;

    for (int ia = 0; ia < ntip_atoms; ia++) {
        const int type_t = tip_atom_type[ia];
        if (type_t < 0) continue;

        const int it0 = 4*ia;
        const float2 ct_px = c_tip[it0    ];
        const float2 ct_py = c_tip[it0 + 1];
        const float2 ct_pz = c_tip[it0 + 2];
        const float2 ct_s  = c_tip[it0 + 3];

        const float wt = stm_cnorm2(ct_px) + stm_cnorm2(ct_py)
                       + stm_cnorm2(ct_pz) + stm_cnorm2(ct_s);
        if (wt == 0.0f) continue;

        const float3 rt = tip_center + tip_pos_rel[ia].xyz;

        for (int ja = 0; ja < nsmp_atoms; ja++) {
            const int type_s = smp_atom_type[ja];
            if (type_s < 0) continue;

            const int pair = pair_map[type_t*n_sample_types + type_s];
            if (pair < 0) continue;

            const float3 d = smp_pos[ja].xyz - rt;
            const float r2 = dot(d, d);
            if (r2 > rcut2 || r2 < 1.0e-16f) continue;

            const float r = sqrt(r2);
            int ir;
            float fr;
            if (!stm_sk_grid_coord(r, r_grid0, inv_dr, n_r, &ir, &fr)) continue;

            const int k0 = pair*n_r + ir;
            const float4 H4 = stm_lerp4(H4_table[k0], H4_table[k0 + 1], fr);
            const float4 S4 = stm_lerp4(S4_table[k0], S4_table[k0 + 1], fr);
            const float Hpi = stm_lerp1(Hpp_pi_table[k0],
                                        Hpp_pi_table[k0 + 1], fr);
            const float Spi = stm_lerp1(Spp_pi_table[k0],
                                        Spp_pi_table[k0 + 1], fr);

            const float4 tau4 = H4 - E_tunnel*S4;
            const float tau_pi = Hpi - E_tunnel*Spi;

            const int is0 = 4*ja;
            const float2 cs_px = c_smp[is0    ];
            const float2 cs_py = c_smp[is0 + 1];
            const float2 cs_pz = c_smp[is0 + 2];
            const float2 cs_s  = c_smp[is0 + 3];

            const float ws = stm_cnorm2(cs_px) + stm_cnorm2(cs_py)
                           + stm_cnorm2(cs_pz) + stm_cnorm2(cs_s);
            if (ws == 0.0f) continue;

            const float3 u = d * (1.0f/r);
            M = stm_cadd(M, stm_contract_sp_pair(
                u, tau4, tau_pi,
                ct_px, ct_py, ct_pz, ct_s,
                cs_px, cs_py, cs_pz, cs_s
            ));
            npair_used++;
        }
    }

    M = stm_cscale(M, amplitude_scale);
    out_M_M2[ip] = (float4)(M.x, M.y, stm_cnorm2(M), (float)npair_used);
}

#endif // LCAO_STM_FGR_CL
