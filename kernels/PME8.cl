// PME8.cl
// Pauli Master Equation kernel for up to 8 sites (256 states).
// Optimised: 1 workgroup = 1 pixel, 256 threads = 256 states.
// Sparse iterative solver (Jacobi + normalisation) — fits in ~13 KB local memory.
//
// === AUTO-DOC BEGIN ===
// @file PME8.cl
// @brief 8-site / 256-state PME STM kernel. Sparse rate matrix + Jacobi solver.
// Each state has ≤8 single-hop neighbours (one per site flip). Rates stored
// in local memory as Rates[256][8] + DiagLoss[256]. Convergence by max|ΔP|<tol.
// Current via parallel reduction across 256 threads.
// === AUTO-DOC END ===

#define N_SITES 8
#define N_STATES 256
#define PI 3.14159265359f

// ======================================================================
// Helper Functions  (identical to PME.cl)
// ======================================================================

inline float fermi(float E, float mu, float T) {
    return 1.0f / (1.0f + exp((E - mu) / T));
}

inline float eval_multipole(float3 d, int order, __global const float* cs) {
    float r2 = dot(d, d);
    float ir2 = 1.0f / r2;
    float E = cs[0]; // Monopole

    if (order > 0) { // Dipole
        E += ir2 * (cs[1]*d.x + cs[2]*d.y + cs[3]*d.z);
    }
    if (order > 1) { // Quadrupole
        float ir4 = ir2 * ir2;
        E += ir4 * ( (cs[4]*d.x + cs[9]*d.y)*d.x +
                     (cs[5]*d.y + cs[7]*d.z)*d.y +
                     (cs[6]*d.z + cs[8]*d.x)*d.z );
    }
    return sqrt(ir2) * E;
}

// ======================================================================
// Kernel 1: Tip Field Calculation  (same as PME.cl, n_sites passed as arg)
// ======================================================================

__kernel void compute_tip_interaction(
    int n_pixels,
    int n_sites,
    __global const float4* restrict p_tips,
    __global const float4* restrict p_sites,
    __global const float*  restrict rots,
    __global const float*  restrict v_tips,
    __global const float*  restrict multipole_cs,
    __global const float* restrict params,
    int order,
    __global float* restrict out_H_shifts,
    __global float* restrict out_T_factors
) {
    int gid = get_global_id(0);
    if (gid >= n_pixels) return;

    float4 tip_data = p_tips[gid];
    float3 tip_pos = tip_data.xyz;
    float v_bias = v_tips[gid];

    float Rtip    = params[0];
    float zV0     = params[1];
    float zVd     = params[2];
    float beta    = params[4];
    float bMirror = params[7];
    float bRamp   = params[8];

    float zV1 = tip_pos.z + zVd;

    for (int i = 0; i < n_sites; i++) {
        float4 site_data = p_sites[i];
        float3 site_pos = site_data.xyz;
        float E_base = site_data.w;

        const float* R = rots + i * 9;

        float3 d = tip_pos - site_pos;
        float3 tip_mirror = (float3)(tip_pos.x, tip_pos.y, 2.0f*zV0 - tip_pos.z);
        float3 d_mir = tip_mirror - site_pos;

        float3 d_rot;
        d_rot.x = R[0]*d.x + R[1]*d.y + R[2]*d.z;
        d_rot.y = R[3]*d.x + R[4]*d.y + R[5]*d.z;
        d_rot.z = R[6]*d.x + R[7]*d.y + R[8]*d.z;

        float3 d_mir_rot;
        d_mir_rot.x = R[0]*d_mir.x + R[1]*d_mir.y + R[2]*d_mir.z;
        d_mir_rot.y = R[3]*d_mir.x + R[4]*d_mir.y + R[5]*d_mir.z;
        d_mir_rot.z = R[6]*d_mir.x + R[7]*d_mir.y + R[8]*d_mir.z;

        float E_val = eval_multipole(d_rot, order, multipole_cs);

        if (bMirror > 0.5f) {
            E_val -= eval_multipole(d_mir_rot, order, multipole_cs);
        }

        E_val *= (v_bias * Rtip);

        if (bRamp > 0.5f) {
            float ramp = (site_pos.z - zV0) / (zV1 - zV0);
            ramp = clamp(ramp, 0.0f, 1.0f);
            E_val += multipole_cs[0] * v_bias * ramp;
        }

        float r = length(d);
        float t_fac = native_exp(-beta * r);

        out_H_shifts[gid * n_sites + i] = E_base + E_val;
        out_T_factors[gid * n_sites + i] = t_fac;
    }
}

// ======================================================================
// Kernel 2: PME Solver — 8 sites / 256 states (sparse iterative)
// ======================================================================
// 1 workgroup per pixel, 256 threads.
// Local memory ~13 KB:
//   Energies[256]          1 KB
//   Rates[256][8]          8 KB
//   DiagLoss[256]          1 KB
//   P_old[256], P_new[256] 2 KB
//   reduce_buf[256]        1 KB
//
// Algorithm:
//   Phase 1: many-body energies (loop over 8 sites + Wij pairs)
//   Phase 2: sparse rate matrix (8 neighbours per state)
//   Phase 3: Jacobi iteration with normalisation constraint
//   Phase 4: current via parallel reduction

__kernel void solve_pme8(
    int n_pixels,
    int n_sites,     // ≤8
    int n_states,    // 256 (or 1<<n_sites)

    __global const float* restrict H_shifts,    // [n_pixels * n_sites]
    __global const float* restrict T_factors,   // [n_pixels * n_sites]
    __global const float* restrict v_tips,      // [n_pixels]

    __global const float* restrict lead_params, // [mu0, T0, mu1, T1]
    __global const float* restrict H_single_base,// [n_sites * n_sites]
    __global const float* restrict Wij,          // [n_sites * n_sites] or NULL
    float W_scalar,
    float Gamma0,   // Substrate coupling (already squared: (Gamma/pi)^2)
    float Gamma1,   // Tip coupling

    __global float* restrict out_current,       // [n_pixels]
    __global float* restrict out_probs,         // [n_pixels * n_states] or NULL
    __global float* restrict out_stateEs,       // [n_pixels * n_states] or NULL

    int   max_iter,
    float tol
) {
    int pix_id = get_group_id(0);
    int tid    = get_local_id(0);   // 0..255

    if (pix_id >= n_pixels) return;

    // --- Local memory ---
    __local float Energies[N_STATES];           // 1 KB
    __local float Rates  [N_STATES][N_SITES];   // 8 KB
    __local float DiagLoss[N_STATES];           // 1 KB
    __local float P_old  [N_STATES];            // 1 KB
    __local float P_new  [N_STATES];            // 1 KB
    __local float reduce_buf[N_STATES];         // 1 KB
    __local float s_max_diff;                   // convergence flag
    __local float s_max_rate;                   // max rate for dt
    __local float s_dt;                         // timestep
    __local float s_psum;                       // probability sum

    // ------------------------------------------------------------------
    // Phase 1: Many-Body Energies
    // ------------------------------------------------------------------
    int mask = tid;   // state bitmask = thread id directly
    float my_energy = 0.0f;

    for (int i = 0; i < n_sites; i++) {
        if ((mask >> i) & 1) {
            my_energy += H_single_base[i * n_sites + i]
                       + H_shifts[pix_id * n_sites + i];
        }
    }
    // Coulomb (Wij or W_scalar)
    if (Wij) {
        for (int i = 0; i < n_sites; i++) {
            if ((mask >> i) & 1) {
                for (int j = i + 1; j < n_sites; j++) {
                    if ((mask >> j) & 1) {
                        my_energy += Wij[i * n_sites + j];
                    }
                }
            }
        }
    } else {
        int nocc = 0;
        for (int i = 0; i < n_sites; i++) {
            if ((mask >> i) & 1) nocc++;
        }
        my_energy += W_scalar * nocc * (nocc - 1) / 2;
    }
    Energies[tid]  = my_energy;
    DiagLoss[tid]  = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    // ------------------------------------------------------------------
    // Phase 2: Build Sparse Rate Matrix
    // ------------------------------------------------------------------
    // Neighbour of state b along site k = b ^ (1<<k).
    // No need to store neighbour indices — compute on the fly.
    float mu0 = lead_params[0]; float T0 = lead_params[1];
    float mu1 = v_tips[pix_id]; float T1 = lead_params[3];

    float diag = 0.0f;
    for (int k = 0; k < n_sites; k++) {
        int mask_c = mask ^ (1 << k);
        int c = mask_c;

        bool bit_b = ((mask >> k) & 1) != 0;
        bool bit_c = ((mask_c >> k) & 1) != 0;
        bool added = (bit_c && !bit_b);  // b→c adds electron (b lower)

        float t_val = T_factors[pix_id * n_sites + k];
        float coup0 = Gamma0;
        float coup1 = Gamma1 * t_val * t_val;

        float rate_in, rate_out;
        if (added) {
            float E_diff = Energies[c] - Energies[tid];
            float f0 = fermi(E_diff, mu0, T0);
            float f1 = fermi(E_diff, mu1, T1);
            rate_out = (coup0 * f0 + coup1 * f1) * 2.0f * PI;          // b→c
            rate_in  = (coup0 * (1.0f-f0) + coup1 * (1.0f-f1)) * 2.0f * PI; // c→b
        } else {
            float E_diff = Energies[tid] - Energies[c];
            float f0 = fermi(E_diff, mu0, T0);
            float f1 = fermi(E_diff, mu1, T1);
            rate_in  = (coup0 * f0 + coup1 * f1) * 2.0f * PI;          // c→b
            rate_out = (coup0 * (1.0f-f0) + coup1 * (1.0f-f1)) * 2.0f * PI; // b→c
        }

        diag -= rate_out;
        Rates[tid][k] = rate_in;   // inflow to b from c
    }
    DiagLoss[tid] = diag;

    barrier(CLK_LOCAL_MEM_FENCE);

    // ------------------------------------------------------------------
    // Phase 3: Explicit Euler Time-Stepping to Steady State
    // ------------------------------------------------------------------
    // dP/dt = K·P, where K[b][b]=DiagLoss[b], K[b][c]=Rates[b][k].
    // Columns of K sum to 0 (probability conserved).
    // Explicit Euler: P_new = P_old + dt * (K · P_old)
    //   P_new[b] = P_old[b] + dt * (DiagLoss[b]*P_old[b] + sum_k Rates[b][k]*P_old[c_k])
    // Then normalise: P /= sum(P).
    // Converges to steady state for dt < 1/max|DiagLoss|.

    // Find max|DiagLoss| for adaptive dt (parallel max reduction)
    reduce_buf[tid] = fabs(DiagLoss[tid]);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = reduce_buf[tid];
            float b = reduce_buf[tid + stride];
            reduce_buf[tid] = (a > b) ? a : b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        s_max_rate = reduce_buf[0];
        s_dt = 0.5f / (s_max_rate + 1e-30f);  // stable step
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float dt = s_dt;

    P_old[tid] = 1.0f / (float)n_states;   // uniform start
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int iter = 0; iter < max_iter; iter++) {
        // --- Euler step ---
        float inflow = 0.0f;
        for (int k = 0; k < n_sites; k++) {
            int c = mask ^ (1 << k);
            inflow += Rates[tid][k] * P_old[c];
        }
        P_new[tid] = P_old[tid] + dt * (DiagLoss[tid] * P_old[tid] + inflow);
        if (P_new[tid] < 0.0f) P_new[tid] = 0.0f;  // clamp negatives
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Normalise: P /= sum(P) ---
        reduce_buf[tid] = P_new[tid];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] += reduce_buf[tid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) s_psum = reduce_buf[0];
        barrier(CLK_LOCAL_MEM_FENCE);
        P_new[tid] /= (s_psum + 1e-30f);
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Convergence check: max|P_new - P_old| ---
        reduce_buf[tid] = fabs(P_new[tid] - P_old[tid]);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float a = reduce_buf[tid];
                float b = reduce_buf[tid + stride];
                reduce_buf[tid] = (a > b) ? a : b;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) s_max_diff = reduce_buf[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Swap P_old and P_new ---
        float tmp = P_old[tid];
        P_old[tid] = P_new[tid];
        P_new[tid] = tmp;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (s_max_diff < tol) break;
    }

    // ------------------------------------------------------------------
    // Output probabilities and energies
    // ------------------------------------------------------------------
    if (out_probs) {
        out_probs[pix_id * n_states + tid] = P_old[tid];
    }
    if (out_stateEs) {
        out_stateEs[pix_id * n_states + tid] = Energies[tid];
    }

    // ------------------------------------------------------------------
    // Phase 4: Current via Parallel Reduction
    // ------------------------------------------------------------------
    // I = sum over transitions b(lower)→c(higher) of [P_b * rate_enter_tip - P_c * rate_leave_tip]
    // Each thread computes its own state's contribution (only added-electron transitions).
    float my_current = 0.0f;
    for (int k = 0; k < n_sites; k++) {
        int mask_c = mask ^ (1 << k);
        int c = mask_c;
        bool bit_c = ((mask_c >> k) & 1) != 0;
        bool bit_b = ((mask >> k) & 1) != 0;
        bool added = (bit_c && !bit_b);
        if (!added) continue;

        float P_b = P_old[tid];
        float P_c = P_old[c];
        float E_diff = Energies[c] - Energies[tid];
        float t_val = T_factors[pix_id * n_sites + k];
        float coupling1 = Gamma1 * t_val * t_val;
        float f1 = fermi(E_diff, mu1, T1);
        float rate_enter = coupling1 * f1 * 2.0f * PI;
        float rate_leave = coupling1 * (1.0f - f1) * 2.0f * PI;
        my_current += P_b * rate_enter - P_c * rate_leave;
    }
    reduce_buf[tid] = my_current;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel sum reduction
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        out_current[pix_id] = reduce_buf[0];
    }
}
