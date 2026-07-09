// lcao_stm.cl - STM (Scanning Tunneling Microscopy) and Dyson equation kernels
// ====================================================================
//
// STM AND ELECTRON TRANSPORT VIA DYSON EQUATION ON GPU
// =====================================================
//
// Computes STM images and electron transport using the Dyson equation approach
// with LCAO basis sets. The Dyson equation relates the Green's function of the
// combined tip+sample system to the isolated tip and sample Green's functions
// via their coupling (overlap integrals from lcao_grid.cl).
//
// --- Physics Overview ---
//
// STM measures the tunneling current between a conductive tip and a sample
// surface as a function of tip position (x, y) and bias voltage (V).
// The current is proportional to the transmission probability, which can be
// computed via the Green's function formalism:
//
//   I(V) ∝ ∫ Tr[Γ_T(E) G^R(E) Γ_S(E) G^A(E)] dE
//
// where G^R is the retarded Green's function of the combined system,
// Γ_T/S are the coupling matrices to tip/sample leads, and the integral
// is over the bias window around the Fermi energy.
//
// The Dyson equation gives the combined Green's function:
//   G = G_0 + G_0 · V_TS · G
//   => G = (I - G_0 · V_TS)^{-1} · G_0
// where G_0 = diag(G_T, G_S) is the block-diagonal Green's function of
// isolated tip and sample, and V_TS is the tip-sample hopping matrix.
//
// Three approaches are implemented:
//
//   1. response_amplitude_exp: Simplified single-orbital tip model.
//      Computes the response amplitude using a scalar Dyson equation:
//        resp = |v · a_st^H|^2 / |(E - E_tip) - a_st · G0 · a_st^H|^2
//      where a_st is the tip-sample coupling vector and v = C^T · G0.
//      Fast (O(ns^2) per point) but limited to s-orbital tip.
//
//   2. solve_stm_dyson_wg: Full matrix Dyson equation with workgroup solver.
//      Uses the "active subspace" trick: only atoms within rcut of the tip
//      contribute to V_TS, so the Dyson matrix W = I - G_S·V^H·G_T·V is small
//      (≤ 32×32). Solved by in-place parallel Gauss-Jordan elimination in
//      __local memory. 1 workgroup = 1 pixel.
//
//   3. stm_gf_dyson_2mol_mo_scan: Precomputed GF contraction.
//      CPU precomputes u_T = c_tip^H · G_T and v_S = G_S · c_smp.
//      GPU computes amp = Σ_{it,is} u_T[it] · H_hop(it,is) · v_S[is].
//      Simplest and fastest: O(n_tip × n_smp) per pixel, no matrix inversion.
//
// --- Slater-Koster Hopping ---
//
// The tip-sample coupling V_TS uses the Slater-Koster two-center approximation:
//   f(r) = exp(-beta * (r - r0))    [exponential radial decay]
//   V_ss = A_ss * f,  V_sp = A_sp * f
//   V_pp = V_pp_sig * (p·u)(p·u) + V_pp_pi * [p·p - (p·u)(p·u)]
// where (l,m,n) = unit vector along the interatomic axis.
//
// Orbital order in STM kernels: [px, py, pz, s] (OpenCL convention, index 0..3).
//
// --- Key Caveats ---
//
//   CAVEAT 1: The response_amplitude_exp kernel uses a fixed-size private array
//   a_st[256]. Systems with more than 256 sample orbitals will overflow.
//
//   CAVEAT 2: The solve_stm_dyson_wg kernel uses MAX_ACT_ORB=32 (8 atoms × 4
//   orbitals). If more than 8 atoms are within rcut, the excess is dropped.
//   The 32×32 complex matrix uses 8 KB of __local memory per array.
//
//   CAVEAT 3: The Gauss-Jordan solver in solve_stm_dyson_wg uses thread-0-only
//   pivot normalization (sequential), then parallel row elimination. This is
//   efficient for N ≤ 32 but would need partial pivoting for stability at
//   larger sizes.
//
//   CAVEAT 4: The stm_gf_dyson_2mol_mo_scan kernel assumes all atoms have
//   exactly 4 orbitals (px, py, pz, s). Atoms with only s (e.g. H) have
//   zero-padded px, py, pz coefficients. The kernel skips zero u_T/v_S entries.
//
// Kernels:
//   - response_amplitude_exp: Scalar Dyson response, s-tip. 1 thread = 1 point.
//   - solve_stm_dyson_wg: Full matrix Dyson with Gauss-Jordan. 1 workgroup = 1 pixel.
//   - stm_gf_dyson_2mol_mo_scan: Precomputed GF contraction. 1 thread = 1 pixel.
//
// Helper functions: c_add, c_sub, c_mul, c_div (complex arithmetic for
// Green's function operations in frequency domain).
// Requires: lcao_grid.cl to be concatenated before this file (for types/helpers).

// ======================================================================
//                          response_amplitude_exp()
// ======================================================================
//
//  Simplified STM response amplitude with single s-orbital tip model.
//  1 thread = 1 scan point. Uses exponential radial basis (no spline tables).
//
//  Physics:
//    The tip is modeled as a single s-orbital at position p. The coupling
//    to the sample is a_st = -H_TS (since S=0 for tunneling):
//      a_st[i0]   = -Vss = -A_ss * exp(-beta*(r-r0))
//      a_st[i0+k] = -Vsp * dir_k  (for p orbitals on sample atom)
//
//    The response amplitude is a scalar Dyson equation:
//      s1 = v · a_st^H      (overlap of incoming wave with coupling)
//      s2 = a_st · G0 · a_st^H  (self-energy correction)
//      d  = (E - E_tip) - s2    (denominator)
//      resp = |s1|^2 / |d|^2
//
//    This is the Tersoff-Hamann approximation extended to p-orbitals.
//
//  Precomputed on CPU:
//    G0 = inv((E+iη)S - H)  — sample Green's function (ns × ns complex)
//    v  = C^T · G0          — incoming wave vector (ns complex)
//
//  CAVEAT: Private array a_st[256] limits to 256 sample orbitals.
//  CAVEAT: The O(ns^2) loop for s2 (G0 · a_st) is the bottleneck for large ns.
//
__kernel void response_amplitude_exp(
    const int n_points,
    __global const float4* points,
    const int natoms_s,
    __global const AtomData* atoms_s,
    __global const int* starts_s,
    const int ns,
    __global const float* v_re,
    __global const float* v_im,
    __global const float* G0_re,
    __global const float* G0_im,
    const float E_re,
    const float E_im,
    const float E_tip,
    const float beta,
    const float r0,
    const float A_ss,
    const float A_sp,
    const float rcut,
    __global float* out_resp
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 p = points[ip].xyz;
    const float rcut2 = rcut * rcut;

    // Build a_st in private memory (max 256 orbitals)
    float2 a_st[256];
    for (int i = 0; i < ns; i++) { a_st[i] = (float2)(0.0f, 0.0f); }

    for (int ia = 0; ia < natoms_s; ia++) {
        const AtomData ad = atoms_s[ia];
        float3 d = p - ad.pos_rcut.xyz;
        const float r2 = dot(d, d);
        if (r2 > rcut2 || r2 < 1e-16f) continue;
        const float r = sqrt(r2);
        const float invr = 1.0f / r;
        const float3 rhat = d * invr;
        const float l = rhat.x, m = rhat.y, n = rhat.z;
        const float f = exp(-beta * (r - r0));

        const float Vss = A_ss * f;
        const float Vsp = A_sp * f;

        const int i0 = starts_s[ia];
        const int nj = starts_s[ia+1] - i0;

        // a = z*S - H; with S=0 for tunneling => a = -H
        // s-tip row: [Vss, l*Vsp, m*Vsp, n*Vsp]
        a_st[i0] = (float2)(-Vss, 0.0f);
        if (nj > 1) {
            a_st[i0+1] = (float2)(-l * Vsp, 0.0f);
            a_st[i0+2] = (float2)(-m * Vsp, 0.0f);
            a_st[i0+3] = (float2)(-n * Vsp, 0.0f);
        }
    }

    // s1 = sum_i v_i * conj(a_i) = sum_i (v_re_i * a_re_i) + i * sum_i (-v_im_i * a_re_i)
    float s1_re = 0.0f, s1_im = 0.0f;
    for (int i = 0; i < ns; i++) {
        s1_re += v_re[i] * a_st[i].x;
        s1_im += -v_im[i] * a_st[i].x;
    }

    // s2 = sum_i a_i * sum_j G0_ij * conj(a_j)
    // conj(a_j) = a_j since a_im = 0
    // b_i = sum_j (G0_re_ij + i*G0_im_ij) * a_j
    float s2_re = 0.0f, s2_im = 0.0f;
    for (int i = 0; i < ns; i++) {
        float b_re = 0.0f, b_im = 0.0f;
        for (int j = 0; j < ns; j++) {
            const int ij = i * ns + j;
            b_re += G0_re[ij] * a_st[j].x;
            b_im += G0_im[ij] * a_st[j].x;
        }
        s2_re += a_st[i].x * b_re;
        s2_im += a_st[i].x * b_im;
    }

    const float d_re = (E_re - E_tip) - s2_re;
    const float d_im = E_im - s2_im;
    const float d_norm2 = d_re * d_re + d_im * d_im;
    const float s1_norm2 = s1_re * s1_re + s1_im * s1_im;

    out_resp[ip] = (d_norm2 > 1e-30f) ? (s1_norm2 / d_norm2) : 0.0f;
}







// ====================


// ======================================================================
//  Complex Arithmetic Helpers
// ======================================================================
//
//  Standard complex (float2) operations for Green's function calculations.
//  float2 = (real, imag). c_div includes 1e-30 regularization to avoid
//  division by zero when the denominator vanishes (e.g., at resonance).
//
//  These are used in solve_stm_dyson_wg for the Dyson matrix construction
//  and Gauss-Jordan elimination.
//
inline float2 c_add(float2 a, float2 b) { return (float2)(a.x+b.x, a.y+b.y); }
inline float2 c_sub(float2 a, float2 b) { return (float2)(a.x-b.x, a.y-b.y); }
inline float2 c_mul(float2 a, float2 b) { return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
inline float2 c_div(float2 a, float2 b) {
    float den = b.x*b.x + b.y*b.y + 1e-30f;
    return (float2)((a.x*b.x + a.y*b.y)/den, (a.y*b.x - a.x*b.y)/den);
}

// ======================================================================
//                          solve_stm_dyson_wg()
// ======================================================================
//
//  Full Dyson equation solver with workgroup-based parallel linear algebra.
//  1 workgroup = 1 pixel (tip position). Threads collaborate on matrix ops.
//
//  --- The "Active Subspace" Trick ---
//
//  The full Green's functions G_T and G_S can be huge (e.g. 400×400), far
//  too large for __local memory. But the hopping V_TS is sparse — only
//  atoms within rcut of the tip contribute. So we extract only the active
//  sub-blocks of G_T and G_S into __local memory (≤ 32×32 = 8 KB each).
//
//  Algorithm (per workgroup/pixel):
//    1. Distance filter: Find active tip atoms (first 8) and sample atoms
//       within rcut of tip center. Build active_T_atoms, active_S_atoms.
//    2. Preload: Copy active sub-blocks of G_S, G_T from global to __local.
//    3. Build V_TS: Compute SK hopping matrix for active tip-sample pairs.
//    4. Matrix multiply: M1 = G_T · V_TS  (N_T × N_S)
//    5. Dyson matrix: W = I - G_S · (V_TS^H · M1)  (N_S × N_S)
//    6. RHS: b_S = G_S · (V_TS^H · (G_T · u_T))
//    7. Solve: W · x_S = b_S via in-place parallel Gauss-Jordan elimination
//    8. Current: I = |x_S|^2 (assuming Γ_R = I for simplicity)
//
//  The Gauss-Jordan solver:
//    - Thread 0 normalizes pivot row (sequential, N_S steps)
//    - All threads eliminate other rows in parallel
//    - barrier between each pivot step
//    - No partial pivoting (stable for N ≤ 32 with well-conditioned W)
//
//  CAVEAT: MAX_ACT_ORB = 32 limits to 8 active atoms per side (4 orbitals each).
//  If more atoms are within rcut, they are silently dropped.
//
//  CAVEAT: The current formula I = |x_S|^2 assumes Γ_R = I (identity).
//  For proper NEGF, Γ_R should be the imaginary part of the self-energy
//  from the drain lead. This simplification may overestimate current.
//
#define MAX_ACT_ORB 32

__kernel void solve_stm_dyson_wg(
    const int n_pixels,
    __global const float4* tip_centers,
    __global const float4* tip_pos_rel,
    __global const float4* smp_pos,
    const int ntip_atoms,
    const int nsmp_atoms,
    // Precalculated full Green's functions from Host (Global Memory)
    __global const float2* GT_global, // [4*ntip_atoms * 4*ntip_atoms]
    __global const float2* GS_global, // [4*nsmp_atoms * 4*nsmp_atoms]
    // Incident wave vector injected from the source lead into the tip
    __global const float2* uT_source, // [4*ntip_atoms]
    // Hopping parameters
    const float beta, const float r0, const float rcut,
    // Output
    __global float* out_current
) {
    // 1 Workgroup = 1 Pixel
    const int pixel_id = get_group_id(0);
    const int t_idx    = get_local_id(0); // Thread ID (e.g., 0 to 63)
    const int threads  = get_local_size(0);
    
    if (pixel_id >= n_pixels) return;

    // --- SHARED MEMORY ALLOCATIONS ---
    __local int active_T_atoms[8]; // Max 8 active tip atoms
    __local int active_S_atoms[8]; // Max 8 active sample atoms
    __local int num_act_T, num_act_S;

    // Local Matrices (32x32 floats2 = 8KB each)
    __local float2 GS_loc[MAX_ACT_ORB][MAX_ACT_ORB];
    __local float2 GT_loc[MAX_ACT_ORB][MAX_ACT_ORB];
    __local float2 V_ts[MAX_ACT_ORB][MAX_ACT_ORB]; // Hopping
    __local float2 W[MAX_ACT_ORB][MAX_ACT_ORB]; // The Dyson Matrix to invert
    
    // Local Vectors
    __local float2 uT_loc[MAX_ACT_ORB];
    __local float2 bS_loc[MAX_ACT_ORB]; // Right-hand side

    if (t_idx == 0) { num_act_T = 0; num_act_S = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 1. DYNAMICALLY IDENTIFY ACTIVE ATOMS (Distance < rcut)
    const float3 cen   = tip_centers[pixel_id].xyz;
    const float rcut2  = rcut * rcut;

    // Tip: take first up to 8 atoms (this is a simple, deterministic choice)
    if (t_idx == 0) {
        const int nt = min(ntip_atoms, 8);
        num_act_T = nt;
        for (int i = 0; i < nt; i++) active_T_atoms[i] = i;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sample: collect up to 8 atoms within rcut from the tip center
    for (int ja = t_idx; ja < nsmp_atoms; ja += threads) {
        const float3 ps = smp_pos[ja].xyz;
        const float3 d  = ps - cen;
        const float  r2 = dot(d, d);
        if (r2 < rcut2) {
            const int slot = atomic_inc((volatile __local int*)&num_act_S);
            if (slot < 8) active_S_atoms[slot] = ja;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t_idx == 0) {
        if (num_act_S > 8) num_act_S = 8;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int N_T = min(num_act_T * 4, MAX_ACT_ORB);
    const int N_S = min(num_act_S * 4, MAX_ACT_ORB);
    if ((N_T <= 0) || (N_S <= 0)) {
        if (t_idx == 0) out_current[pixel_id] = 0.0f;
        return;
    }

    // 2. PRELOAD ACTIVE BLOCKS FROM GLOBAL TO SHARED MEMORY
    // Threads loop through the 2D local arrays and fetch from global memory
    for(int i = t_idx; i < N_S * N_S; i += threads) {
        int r = i / N_S; int c = i % N_S;
        int glob_r = active_S_atoms[r/4]*4 + (r%4);
        int glob_c = active_S_atoms[c/4]*4 + (c%4);
        GS_loc[r][c] = GS_global[glob_r * (4*nsmp_atoms) + glob_c];
    }
    for(int i = t_idx; i < N_T * N_T; i += threads) {
        int r = i / N_T; int c = i % N_T;
        int glob_r = active_T_atoms[r/4]*4 + (r%4);
        int glob_c = active_T_atoms[c/4]*4 + (c%4);
        GT_loc[r][c] = GT_global[glob_r * (4*ntip_atoms) + glob_c];
    }
    for(int i = t_idx; i < N_T; i += threads) {
        int glob_i = active_T_atoms[i/4]*4 + (i%4);
        uT_loc[i] = uT_source[glob_i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. COMPUTE SLATER-KOSTER HOPPING MATRIX V_ts (real, stored as complex with imag=0)
    for (int i = t_idx; i < N_T * N_S; i += threads) {
        const int it = i / N_S;
        const int is = i - it * N_S;
        const int ia = active_T_atoms[it / 4];
        const int ja = active_S_atoms[is / 4];
        const int ot = it & 3;
        const int os = is & 3;
        const float3 pt = cen + tip_pos_rel[ia].xyz;
        const float3 ps = smp_pos[ja].xyz;
        float3 d = ps - pt;
        const float r2 = dot(d, d);
        if (r2 > rcut2 || r2 < 1e-16f) {
            V_ts[it][is] = (float2)(0.0f, 0.0f);
            continue;
        }
        const float r = sqrt(r2);
        const float invr = 1.0f / r;
        const float l = d.x * invr;
        const float m = d.y * invr;
        const float n = d.z * invr;
        const float f = exp(-beta * (r - r0));

        // Use simple isotropic exponential SK parameters (units are arbitrary scaling for now)
        const float Vss     = 1.0f * f;
        const float Vsp     = 1.0f * f;
        const float Vps     = 1.0f * f;
        const float Vpp_sig = 1.0f * f;
        const float Vpp_pi  = 0.2f * f;

        // Orbital order in this kernel: (px,py,pz,s) index 0..3
        float val = 0.0f;
        if (ot == 3 && os == 3) {
            val = Vss;
        } else if (ot == 3 && os != 3) {
            val = Vsp * ((os == 0) ? l : (os == 1) ? m : n);
        } else if (ot != 3 && os == 3) {
            val = Vps * ((ot == 0) ? l : (ot == 1) ? m : n);
        } else {
            const float ut = (ot == 0) ? l : (ot == 1) ? m : n;
            const float us = (os == 0) ? l : (os == 1) ? m : n;
            const float dV = Vpp_sig - Vpp_pi;
            const float delta = (ot == os) ? 1.0f : 0.0f;
            val = Vpp_pi * delta + dV * ut * us;
        }
        V_ts[it][is] = (float2)(val, 0.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3b) M1 = GT * V_ts   (N_T x N_S)
    __local float2 M1[MAX_ACT_ORB][MAX_ACT_ORB];
    for (int i = t_idx; i < N_T * N_S; i += threads) {
        const int it = i / N_S;
        const int is = i - it * N_S;
        float2 acc = (float2)(0.0f, 0.0f);
        for (int kt = 0; kt < N_T; kt++) {
            acc = c_add(acc, c_mul(GT_loc[it][kt], V_ts[kt][is]));
        }
        M1[it][is] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4. Construct W = I - GS * (V^H * (GT*V)) = I - GS * M2
    for (int i = t_idx; i < N_S * N_S; i += threads) {
        const int is = i / N_S;
        const int js = i - is * N_S;

        // M2[ks,js] = sum_t conj(V[t,ks]) * M1[t,js]
        float2 m3 = (float2)(0.0f, 0.0f);
        for (int ks = 0; ks < N_S; ks++) {
            float2 m2 = (float2)(0.0f, 0.0f);
            for (int it = 0; it < N_T; it++) {
                const float2 v = V_ts[it][ks];
                const float2 vH = (float2)(v.x, -v.y);
                m2 = c_add(m2, c_mul(vH, M1[it][js]));
            }
            m3 = c_add(m3, c_mul(GS_loc[is][ks], m2));
        }

        const float2 Iij = (is == js) ? (float2)(1.0f, 0.0f) : (float2)(0.0f, 0.0f);
        W[is][js] = c_sub(Iij, m3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // RHS: bS = GS * (V^H * (GT * uT))
    __local float2 tvec[MAX_ACT_ORB];
    __local float2 svec[MAX_ACT_ORB];
    for (int it = t_idx; it < N_T; it += threads) {
        float2 acc = (float2)(0.0f, 0.0f);
        for (int kt = 0; kt < N_T; kt++) {
            acc = c_add(acc, c_mul(GT_loc[it][kt], uT_loc[kt]));
        }
        tvec[it] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int is = t_idx; is < N_S; is += threads) {
        float2 acc = (float2)(0.0f, 0.0f);
        for (int it = 0; it < N_T; it++) {
            const float2 v = V_ts[it][is];
            const float2 vH = (float2)(v.x, -v.y);
            acc = c_add(acc, c_mul(vH, tvec[it]));
        }
        svec[is] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int is = t_idx; is < N_S; is += threads) {
        float2 acc = (float2)(0.0f, 0.0f);
        for (int ks = 0; ks < N_S; ks++) {
            acc = c_add(acc, c_mul(GS_loc[is][ks], svec[ks]));
        }
        bS_loc[is] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 5. IN-PLACE PARALLEL GAUSS-JORDAN SOLVER (W * xS = bS)
    // We solve the linear system for the active sample subspace
    for (int k = 0; k < N_S; k++) {
        // Thread 0 finds pivot (partial pivoting optional for N<32, but good for stability)
        if (t_idx == 0) {
            float2 pivot = W[k][k];
            // Normalize pivot row
            for(int j=k; j < N_S; j++) W[k][j] = c_div(W[k][j], pivot);
            bS_loc[k] = c_div(bS_loc[k], pivot);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // All threads eliminate the other rows
        for (int i = t_idx; i < N_S; i += threads) {
            if (i != k) {
                float2 factor = W[i][k];
                for (int j = k; j < N_S; j++) {
                    W[i][j] = c_sub(W[i][j], c_mul(factor, W[k][j]));
                }
                bS_loc[i] = c_sub(bS_loc[i], c_mul(factor, bS_loc[k]));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Now bS_loc contains the exact response wave x_S.

    // 6. CALCULATE FINAL TRANSMISSION CURRENT
    // I = x_S^H * Gamma_R * x_S
    float current = 0.0f;
    if (t_idx == 0) {
        for(int i=0; i<N_S; i++) {
            // Assuming Gamma_R is a simple density-of-states weighting for the drain
            current += (bS_loc[i].x*bS_loc[i].x + bS_loc[i].y*bS_loc[i].y);
        }
        out_current[pixel_id] = current;
    }
}

// ======================================================================
//                          stm_gf_dyson_2mol_mo_scan()
// ======================================================================
//
//  Precomputed Green's function contraction for STM. 1 thread = 1 pixel.
//  The simplest and fastest STM kernel — no matrix inversion on GPU.
//
//  Physics:
//    amp = c_tip^H · G_T · H_TS · G_S · c_smp
//
//  CPU precomputes the GF contractions:
//    v_S = G_S · c_smp    (sample Green's function applied to sample MO)
//    u_T = c_tip^H · G_T  (tip MO conjugated with tip Green's function)
//
//  GPU computes only the hopping contraction per pixel:
//    amp = Σ_{it∈tip} Σ_{is∈smp} u_T[it] · H_hop(it, is) · v_S[is]
//    I   = |amp|^2
//
//  This is the Bardeen transfer Hamiltonian approximation: the transmission
//  is the square of the hopping matrix element between tip and sample
//  scattering states. Valid in the weak-coupling limit (small V_TS).
//
//  Hopping H_hop uses simplified exponential SK:
//    f = exp(-beta*(r - r0))
//    Vss = f, Vsp = f, Vps = f, Vpp_sig = f, Vpp_pi = 0.2*f
//
//  Orbital convention: [px, py, pz, s] per atom (OpenCL order, index 0..3).
//  All vectors (u_T, v_S) and orb2atom arrays are pre-remapped on CPU.
//
//  CAVEAT: Assumes 4 orbitals per atom (s+p). The orb2atom mapping uses
//  it & 3 and is & 3 to extract orbital type. For d-orbital systems,
//  this kernel would need modification.
//
//  CAVEAT: Zero-padded orbitals (e.g. H with only s) are skipped via
//  if(uTit.x==0 && uTit.y==0) continue. This assumes the CPU has set
//  px,py,pz coefficients to zero for atoms without p orbitals.
//
__kernel void stm_gf_dyson_2mol_mo_scan(
    const int n_points,
    __global const float4* tip_centers,
    __global const float4* tip_pos_rel,
    __global const float4* smp_pos,
    __global const int* tip_orb2atom,     // [tip_norb_ocl] 0-based atom index per orbital
    __global const int* smp_orb2atom,     // [smp_norb_ocl]
    __global const float2* u_T,            // [tip_norb_ocl] c_tip^H @ GT  in OCL order
    __global const float2* v_S,            // [smp_norb_ocl] GS @ c_smp   in OCL order
    const int ntip_atoms,
    const int nsmp_atoms,
    const int tip_norb_ocl,                // = ntip_atoms * 4
    const int smp_norb_ocl,                // = nsmp_atoms * 4
    const float beta,
    const float r0,
    const float rcut,
    __global float* out_current
) {
    const int ip = get_global_id(0);
    if (ip >= n_points) return;

    const float3 cen = tip_centers[ip].xyz;
    const float rcut2 = rcut * rcut;
    float2 amp = (float2)(0.0f, 0.0f);

    for (int it = 0; it < tip_norb_ocl; it++) {
        const int ia = tip_orb2atom[it];
        if (ia < 0 || ia >= ntip_atoms) continue;
        const float2 uTit = u_T[it];
        if (uTit.x == 0.0f && uTit.y == 0.0f) continue;  // skip zero-padded orbitals (e.g. H px,py,pz)

        const float3 pt = cen + tip_pos_rel[ia].xyz;
        const int ot = it & 3;  // 0=px, 1=py, 2=pz, 3=s in OCL convention

        for (int is = 0; is < smp_norb_ocl; is++) {
            const int ja = smp_orb2atom[is];
            if (ja < 0 || ja >= nsmp_atoms) continue;
            const float2 vSis = v_S[is];
            if (vSis.x == 0.0f && vSis.y == 0.0f) continue;

            const float3 ps = smp_pos[ja].xyz;
            const float3 d = pt - ps;
            const float r2 = dot(d, d);
            if (r2 > rcut2 || r2 < 1e-16f) continue;

            const float r = sqrt(r2);
            const float invr = 1.0f / r;
            const float l = d.x * invr;
            const float m = d.y * invr;
            const float n = d.z * invr;
            const float f = exp(-beta * (r - r0));
            const int os = is & 3;

            // Simplified exponential SK hopping (real, symmetric)
            float V = 0.0f;
            if (ot == 3 && os == 3) {
                V = f;                                   // Vss
            } else if (ot == 3 && os != 3) {
                V = f * ((os == 0) ? l : (os == 1) ? m : n);  // Vsp * dir
            } else if (ot != 3 && os == 3) {
                V = f * ((ot == 0) ? l : (ot == 1) ? m : n);  // Vps * dir
            } else {
                const float ut = (ot == 0) ? l : (ot == 1) ? m : n;
                const float us = (os == 0) ? l : (os == 1) ? m : n;
                const float Vpp_pi = 0.2f * f;
                const float dV = f - Vpp_pi;              // Vpp_sig - Vpp_pi
                const float delta = (ot == os) ? 1.0f : 0.0f;
                V = Vpp_pi * delta + dV * ut * us;
            }

            // amp += u_T[it] * V * v_S[is]   (V is real, stored as scalar)
            amp.x += uTit.x * V * vSis.x - uTit.y * V * vSis.y;
            amp.y += uTit.x * V * vSis.y + uTit.y * V * vSis.x;
        }
    }
    out_current[ip] = amp.x * amp.x + amp.y * amp.y;
}