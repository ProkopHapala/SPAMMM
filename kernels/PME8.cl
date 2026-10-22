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
#define PME_GMRES_M 128

// Optional stationary arithmetic precision. Public field/current buffers retain
// their original float ABI; the historical build remains entirely float.
#ifdef PME8_STATIONARY_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double pme_real;
#else
typedef float pme_real;
#endif

inline pme_real pme_dot(int tid, __local const pme_real* a, __local const pme_real* b, __local pme_real* tmp) {
    tmp[tid]=a[tid]*b[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=128;stride>0;stride>>=1){if(tid<stride)tmp[tid]+=tmp[tid+stride];barrier(CLK_LOCAL_MEM_FENCE);}
    pme_real value=tmp[0];barrier(CLK_LOCAL_MEM_FENCE);return value;
}

inline void pme_stationary_matvec(int tid,int ns,__local const pme_real* x,__local pme_real* y,__local const pme_real* diag,__local const pme_real* rates,__local pme_real* tmp){
    pme_real inflow=0;
    for(int k=0;k<ns;k++)inflow+=rates[tid*N_SITES+k]*x[tid^(1<<k)];
    tmp[tid]=x[tid];barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=128;stride>0;stride>>=1){if(tid<stride)tmp[tid]+=tmp[tid+stride];barrier(CLK_LOCAL_MEM_FENCE);}
    // Replace one redundant balance equation with sum(P)=1; row-scale the others.
    y[tid]=(tid==255)?tmp[0]/256:(diag[tid]<0?x[tid]+inflow/diag[tid]:0);
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline pme_real pme_dot_global_local(int tid,__global const pme_real* a,__local const pme_real* b,__local pme_real* tmp){
    tmp[tid]=a[tid]*b[tid];barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=128;stride>0;stride>>=1){if(tid<stride)tmp[tid]+=tmp[tid+stride];barrier(CLK_LOCAL_MEM_FENCE);}
    pme_real value=tmp[0];barrier(CLK_LOCAL_MEM_FENCE);return value;
}

inline void pme_stationary_matvec_global(int tid,int ns,__global const pme_real* x,__local pme_real* y,__local const pme_real* diag,__local const pme_real* rates,__local pme_real* tmp){
    pme_real inflow=0;for(int k=0;k<ns;k++)inflow+=rates[tid*N_SITES+k]*x[tid^(1<<k)];
    tmp[tid]=x[tid];barrier(CLK_LOCAL_MEM_FENCE);
    for(int stride=128;stride>0;stride>>=1){if(tid<stride)tmp[tid]+=tmp[tid+stride];barrier(CLK_LOCAL_MEM_FENCE);}
    y[tid]=(tid==255)?tmp[0]/256:(diag[tid]<0?x[tid]+inflow/diag[tid]:0);barrier(CLK_LOCAL_MEM_FENCE);
}

// ======================================================================
// Helper Functions  (identical to PME.cl)
// ======================================================================

inline pme_real fermi(float E, float mu, float T) {
    return (pme_real)1 / ((pme_real)1 + exp(((pme_real)E - mu) / T));
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

inline void compute_tip_interaction_impl(
    int n_pixels,
    int n_sites,
    __global const float4* restrict p_tips,
    __global const float4* restrict p_sites,
    __global const float*  restrict rots,
    __global const float*  restrict v_tips,
    __global const float*  restrict multipole_cs,
    __global const float* restrict params,
    int order,
    __global const float* cell,
    __global const int* periodic_sites,
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
        if (cell && periodic_sites[i]) {
            float2 dxy = tip_pos.xy - site_pos.xy;
            float2 a = (float2)(cell[0], cell[1]);
            float2 b = (float2)(cell[2], cell[3]);
            float2 fractional = (float2)(dxy.x*cell[4]+dxy.y*cell[6],
                                         dxy.x*cell[5]+dxy.y*cell[7]);
            float2 center = rint(fractional);
            float best = INFINITY;
            float2 shift = (float2)(0.0f);
            // Host Gauss-reduces the lattice; nine candidates contain the
            // closest image, including oblique cells. Never sum replicas.
            for (int u=-1; u<=1; ++u) for (int v=-1; v<=1; ++v) {
                float2 candidate = (center.x+u)*a + (center.y+v)*b;
                float2 delta = dxy-candidate;
                float distance = dot(delta,delta);
                if (distance < best) { best=distance; shift=candidate; }
            }
            site_pos.xy += shift;
        }
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
    compute_tip_interaction_impl(n_pixels, n_sites, p_tips, p_sites, rots, v_tips, multipole_cs, params, order, 0, 0, out_H_shifts, out_T_factors);
}

__kernel void compute_tip_interaction_nearest(
    int n_pixels,
    int n_sites,
    __global const float4* restrict p_tips,
    __global const float4* restrict p_sites,
    __global const float*  restrict rots,
    __global const float*  restrict v_tips,
    __global const float*  restrict multipole_cs,
    __global const float* restrict params,
    int order,
    __global const float* cell,
    __global const int* periodic_sites,
    __global float* restrict out_H_shifts,
    __global float* restrict out_T_factors
) {
    compute_tip_interaction_impl(n_pixels, n_sites, p_tips, p_sites, rots, v_tips, multipole_cs, params, order, cell, periodic_sites, out_H_shifts, out_T_factors);
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
    float tol,
    int stationary_method, // 0: historical Euler; 1: Jacobi; 2: BiCGSTAB; 3: GMRES(32)
    __global pme_real* krylov_workspace
) {
    int pix_id = get_group_id(0);
    int tid    = get_local_id(0);   // 0..255

    if (pix_id >= n_pixels) return;

    // --- Local memory ---
    __local float Energies[N_STATES];           // 1 KB
    __local pme_real Rates  [N_STATES][N_SITES];
    __local pme_real DiagLoss[N_STATES];
    __local pme_real P_old  [N_STATES];
    __local pme_real P_new  [N_STATES];
    __local pme_real reduce_buf[N_STATES];
    __local pme_real s_max_diff;                // convergence flag
    __local pme_real s_max_rate;                // max rate for dt
    __local pme_real s_dt;                      // timestep
    __local pme_real s_psum;                    // probability sum
    __local pme_real krylov_vectors[6][N_STATES];
    #define krylov_r krylov_vectors[0]
    #define krylov_rhat krylov_vectors[1]
    #define krylov_p krylov_vectors[2]
    #define krylov_v krylov_vectors[3]
    #define krylov_s krylov_vectors[4]
    #define krylov_t krylov_vectors[5]
    __local pme_real gmres_cs[PME_GMRES_M],gmres_sn[PME_GMRES_M],gmres_g[PME_GMRES_M+1],gmres_y[PME_GMRES_M];

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

    pme_real diag = 0.0f;
    for (int k = 0; k < n_sites; k++) {
        int mask_c = mask ^ (1 << k);
        int c = mask_c;

        bool bit_b = ((mask >> k) & 1) != 0;
        bool bit_c = ((mask_c >> k) & 1) != 0;
        bool added = (bit_c && !bit_b);  // b→c adds electron (b lower)

        float t_val = T_factors[pix_id * n_sites + k];
        pme_real coup0 = Gamma0;
        pme_real coup1 = (pme_real)Gamma1 * t_val * t_val;

        pme_real rate_in, rate_out;
        if (added) {
            float E_diff = Energies[c] - Energies[tid];
            pme_real f0 = fermi(E_diff, mu0, T0);
            pme_real f1 = fermi(E_diff, mu1, T1);
            rate_out = (coup0 * f0 + coup1 * f1) * 2.0f * PI;          // b→c
            rate_in  = (coup0 * (1.0f-f0) + coup1 * (1.0f-f1)) * 2.0f * PI; // c→b
        } else {
            float E_diff = Energies[tid] - Energies[c];
            pme_real f0 = fermi(E_diff, mu0, T0);
            pme_real f1 = fermi(E_diff, mu1, T1);
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
            pme_real a = reduce_buf[tid];
            pme_real b = reduce_buf[tid + stride];
            reduce_buf[tid] = (a > b) ? a : b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0) {
        s_max_rate = reduce_buf[0];
        s_dt = 0.5f / (s_max_rate + 1e-30f);  // stable step
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    pme_real dt = s_dt;

    P_old[tid] = 1.0f / (float)n_states;   // uniform start
    barrier(CLK_LOCAL_MEM_FENCE);

    if(stationary_method==3){
        // Restarted GMRES(128), with the actual stationary residual checked after
        // every restart. The sparse rates and normalized system are unchanged.
        __global pme_real* basis=krylov_workspace+(size_t)pix_id*(PME_GMRES_M+1)*(256+PME_GMRES_M);
        __global pme_real* gmres_H=basis+(PME_GMRES_M+1)*256;
        for(int cycle=0;cycle<max_iter/PME_GMRES_M;cycle++){
            pme_stationary_matvec(tid,n_sites,P_old,P_new,DiagLoss,&Rates[0][0],reduce_buf);
            krylov_vectors[0][tid]=(tid==255?((pme_real)1/256):0)-P_new[tid];barrier(CLK_LOCAL_MEM_FENCE);
            pme_real beta=sqrt(pme_dot(tid,krylov_vectors[0],krylov_vectors[0],reduce_buf));
            if(beta<tol)break;
            basis[tid]=krylov_vectors[0][tid]/beta;
            for(int k=tid;k<(PME_GMRES_M+1)*PME_GMRES_M;k+=256)gmres_H[k]=0;
            if(tid<PME_GMRES_M+1)gmres_g[tid]=tid==0?beta:0;
            barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
            int used=0;
            for(int j=0;j<PME_GMRES_M;j++){
                pme_stationary_matvec_global(tid,n_sites,basis+j*256,P_new,DiagLoss,&Rates[0][0],reduce_buf);
                for(int i=0;i<=j;i++){
                    pme_real hij=pme_dot_global_local(tid,basis+i*256,P_new,reduce_buf);
                    if(tid==0)gmres_H[i*PME_GMRES_M+j]=hij;
                    P_new[tid]-=hij*basis[i*256+tid];barrier(CLK_LOCAL_MEM_FENCE);
                }
                pme_real hn=sqrt(pme_dot(tid,P_new,P_new,reduce_buf));
                basis[(j+1)*256+tid]=hn>(pme_real)1e-30?P_new[tid]/hn:0;
                if(tid==0){
                    gmres_H[(j+1)*PME_GMRES_M+j]=hn;
                    for(int i=0;i<j;i++){
                        pme_real hi=gmres_H[i*PME_GMRES_M+j],hn1=gmres_H[(i+1)*PME_GMRES_M+j];
                        gmres_H[i*PME_GMRES_M+j]=gmres_cs[i]*hi+gmres_sn[i]*hn1;
                        gmres_H[(i+1)*PME_GMRES_M+j]=-gmres_sn[i]*hi+gmres_cs[i]*hn1;
                    }
                    pme_real aa=gmres_H[j*PME_GMRES_M+j],bb=gmres_H[(j+1)*PME_GMRES_M+j],rr=hypot(aa,bb);
                    gmres_cs[j]=rr>0?aa/rr:1;gmres_sn[j]=rr>0?bb/rr:0;
                    gmres_H[j*PME_GMRES_M+j]=rr;gmres_H[(j+1)*PME_GMRES_M+j]=0;
                    gmres_g[j+1]=-gmres_sn[j]*gmres_g[j];gmres_g[j]*=gmres_cs[j];
                }
                barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);used=j+1;
                if(fabs(gmres_g[j+1])<tol || hn<(pme_real)1e-30)break;
            }
            if(tid==0){
                for(int i=used-1;i>=0;i--){
                    pme_real rhs=gmres_g[i];
                    for(int j=i+1;j<used;j++)rhs-=gmres_H[i*PME_GMRES_M+j]*gmres_y[j];
                    gmres_y[i]=fabs(gmres_H[i*PME_GMRES_M+i])>(pme_real)1e-30?rhs/gmres_H[i*PME_GMRES_M+i]:0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for(int i=0;i<used;i++)P_old[tid]+=gmres_y[i]*basis[i*256+tid];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        pme_stationary_matvec(tid,n_sites,P_old,P_new,DiagLoss,&Rates[0][0],reduce_buf);
        krylov_r[tid]=(tid==255?((pme_real)1/256):0)-P_new[tid];barrier(CLK_LOCAL_MEM_FENCE);
        pme_real residual=pme_dot(tid,krylov_r,krylov_r,reduce_buf);
        // Squared-residual floor: FP64 can reach 1e-20; FP32 converged residuals
        // are ~1e-12, so the same gate would flag every good solution as NaN.
#ifdef PME8_STATIONARY_FP64
        const pme_real residual_gate=(pme_real)1e-20;
#else
        const pme_real residual_gate=(pme_real)1e-8;
#endif
        if(residual>residual_gate)P_old[tid]=NAN;
        if(P_old[tid]<0 && P_old[tid]>-(pme_real)1e-10)P_old[tid]=0;
        barrier(CLK_LOCAL_MEM_FENCE);
    } else if(stationary_method==2){
        // Residual-controlled BiCGSTAB for the same normalized stationary system.
        // It resolves weakly connected configurations without physical rate changes.
        pme_stationary_matvec(tid,n_sites,P_old,krylov_v,DiagLoss,&Rates[0][0],reduce_buf);
        krylov_r[tid]=(tid==255?((pme_real)1/256):0)-krylov_v[tid];
        krylov_rhat[tid]=krylov_r[tid];krylov_p[tid]=0;krylov_v[tid]=0;
        barrier(CLK_LOCAL_MEM_FENCE);
        pme_real rho_old=1,alpha=1,omega=1;
        for(int iter=0;iter<max_iter;iter++){
            pme_real rho=pme_dot(tid,krylov_rhat,krylov_r,reduce_buf);
            if(fabs(rho)<(pme_real)1e-30)break;
            pme_real beta=(rho/rho_old)*(alpha/omega);
            krylov_p[tid]=krylov_r[tid]+beta*(krylov_p[tid]-omega*krylov_v[tid]);barrier(CLK_LOCAL_MEM_FENCE);
            pme_stationary_matvec(tid,n_sites,krylov_p,krylov_v,DiagLoss,&Rates[0][0],reduce_buf);
            pme_real denominator=pme_dot(tid,krylov_rhat,krylov_v,reduce_buf);
            if(fabs(denominator)<(pme_real)1e-30)break;
            alpha=rho/denominator;
            krylov_s[tid]=krylov_r[tid]-alpha*krylov_v[tid];barrier(CLK_LOCAL_MEM_FENCE);
            pme_real snorm=pme_dot(tid,krylov_s,krylov_s,reduce_buf);
            if(snorm<(pme_real)tol*tol){P_old[tid]+=alpha*krylov_p[tid];barrier(CLK_LOCAL_MEM_FENCE);break;}
            pme_stationary_matvec(tid,n_sites,krylov_s,krylov_t,DiagLoss,&Rates[0][0],reduce_buf);
            pme_real ts=pme_dot(tid,krylov_t,krylov_s,reduce_buf);
            pme_real tt=pme_dot(tid,krylov_t,krylov_t,reduce_buf);
            if(tt<(pme_real)1e-30)break;
            omega=ts/tt;
            P_old[tid]+=alpha*krylov_p[tid]+omega*krylov_s[tid];
            krylov_r[tid]=krylov_s[tid]-omega*krylov_t[tid];barrier(CLK_LOCAL_MEM_FENCE);
            pme_real rnorm=pme_dot(tid,krylov_r,krylov_r,reduce_buf);
            if(rnorm<(pme_real)tol*tol)break;
            if(fabs(omega)<(pme_real)1e-30)break;
            rho_old=rho;
        }
        pme_stationary_matvec(tid,n_sites,P_old,krylov_v,DiagLoss,&Rates[0][0],reduce_buf);
        krylov_r[tid]=(tid==255?((pme_real)1/256):0)-krylov_v[tid];barrier(CLK_LOCAL_MEM_FENCE);
        pme_real residual=pme_dot(tid,krylov_r,krylov_r,reduce_buf);
        // Fail loudly on a nonconverged linear solve, instead of returning a plausible map.
        if(residual>(pme_real)1e-16)P_old[tid]=NAN;
        if(P_old[tid]<0 && P_old[tid]>-(pme_real)1e-10)P_old[tid]=0;
        barrier(CLK_LOCAL_MEM_FENCE);
    } else {
    for (int iter = 0; iter < max_iter; iter++) {
        // --- Euler step ---
        pme_real inflow = 0.0f;
        for (int k = 0; k < n_sites; k++) {
            int c = mask ^ (1 << k);
            inflow += Rates[tid][k] * P_old[c];
        }
        if (stationary_method == 1) {
            // Solve D*P + R*P = 0 with damped Jacobi. Dividing each row by
            // its own exit rate avoids the global Euler timestep freezing
            // weak-tip transitions when substrate and tip rates differ greatly.
            // Normalization below fixes the null-vector scale; the stationary
            // equation is identical to Euler's. Preserve zero-exit states.
            pme_real exit_rate = -DiagLoss[tid];
            P_new[tid] = exit_rate > 0.0f ? 0.5f*(P_old[tid] + inflow/exit_rate) : P_old[tid];
        } else {
            P_new[tid] = P_old[tid] + dt * (DiagLoss[tid] * P_old[tid] + inflow);
        }
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
                pme_real a = reduce_buf[tid];
                pme_real b = reduce_buf[tid + stride];
                reduce_buf[tid] = (a > b) ? a : b;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid == 0) s_max_diff = reduce_buf[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Swap P_old and P_new ---
        pme_real tmp = P_old[tid];
        P_old[tid] = P_new[tid];
        P_new[tid] = tmp;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (s_max_diff < tol) break;
    }
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
    pme_real my_current = 0.0f;
    for (int k = 0; k < n_sites; k++) {
        int mask_c = mask ^ (1 << k);
        int c = mask_c;
        bool bit_c = ((mask_c >> k) & 1) != 0;
        bool bit_b = ((mask >> k) & 1) != 0;
        bool added = (bit_c && !bit_b);
        if (!added) continue;

        pme_real P_b = P_old[tid];
        pme_real P_c = P_old[c];
        float E_diff = Energies[c] - Energies[tid];
        float t_val = T_factors[pix_id * n_sites + k];
        pme_real coupling1 = (pme_real)Gamma1 * t_val * t_val;
        pme_real f1 = fermi(E_diff, mu1, T1);
        pme_real rate_enter = coupling1 * f1 * 2.0f * PI;
        pme_real rate_leave = coupling1 * (1.0f - f1) * 2.0f * PI;
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
