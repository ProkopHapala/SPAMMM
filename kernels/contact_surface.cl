// contact_surface.cl — Quasi-2D contact surface for static AFM (aperiodic rigid sample)
//
// Compact alternative to full 3D GridFF: separable B-spline(xy) × doubling-poly(dz)
// with optional per-atom radial PIC correction. Used for fitting and evaluating
// molecule–sample interaction above a height map h₀(x,y).
//
// Kernels:
//   - cs_brute_plqh_points: Brute Morse+PLQH reference at query points (validation)
//   - evalSeparableBsplinePoly: Separable field eval; F = -∇E (AFM force convention)
//   - cs_sep_Av / cs_sep_Atv / cs_sep_Atv_masked: Matrix-free separable fit operators
//   - cs_pic_Av / cs_pic_Atv / cs_pic_eval_tile16 / evalRadialPIC: PIC fit and eval
//   - relaxStrokesTiltedContactPMELocalQN:   opt-in quasi-Newton PP relax (secant->FD-Newton->FIRE)
//   - relaxStrokesTiltedContactPMELocalSph:  opt-in sphere-constrained relax (|dpos|=L, 2 soft DOF)
//   - CG helpers: dot_wg, addMul, setLinear, cs_zero, cs_copy
//
// Python: spammm/surfaces/ContactSurface.py
// Design: doc/Topics/AFM/ContactSurface_Static.md
// Requires: common.cl + Forces.cl (getMorsePLQH) concatenated before this file.

#define CS_TILE 16
#define CS_ATOM_TILE 32
#define CS_PIC_LOCAL_MAX 384

inline void atomic_add_f(__global float* addr, float val) {
    union { uint u; float f; } old, neu;
    old.u = as_uint(*addr);
    while (1) {
        neu.f = old.f + val;
        uint prev = atomic_cmpxchg((__global uint*)addr, old.u, neu.u);
        if (prev == old.u) break;
        old.u = prev;
        old.f = as_float(prev);
    }
}

// ===================== math utilities (GridFF pattern) =====================

__kernel void cs_zero(const int n, __global float* a) {
    int i = get_global_id(0);
    if (i >= n) return;
    a[i] = 0.0f;
}

__kernel void cs_zero_outside(const int n, __global float* a, const int i0, const int i1) {
    int i = get_global_id(0);
    if (i >= n) return;
    if (i < i0 || i >= i1) a[i] = 0.0f;
}

__kernel void cs_copy(const int n, __global const float* src, __global float* dst) {
    int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = src[i];
}

__kernel void addMul(const int ntot, __global float* a, __global const float* b, const float c) {
    int i = get_global_id(0);
    if (i >= ntot) return;
    a[i] += b[i] * c;
}

__kernel void setLinear(const int ntot, __global float* out, const float c1, __global const float* a1, const float c2, __global const float* a2) {
    int i = get_global_id(0);
    if (i >= ntot) return;
    out[i] = c1 * a1[i] + c2 * a2[i];
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void dot_wg(const int ntot, __global const float* a, __global const float* b, __global float* partial) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    float acc = 0.0f;
    for (int i = gid; i < ntot; i += get_global_size(0)) {
        acc += a[i] * b[i];
    }
    __local float s[64];
    s[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = lsz >> 1; step > 0; step >>= 1) {
        if (lid < step) { s[lid] += s[lid + step]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) { partial[get_group_id(0)] = s[0]; }
}

// ===================== basis helpers =====================

// GridFF-compatible B-spline cell index: knots (ix-1, ix, ix+1, ix+2), local param tx in [0,1)
inline int cs_bspline_cell(float u, float* t_local) {
    int i = (int)u;
    if (u < 0.0f) i--;
    *t_local = u - (float)i;
    return i;
}

inline void bspline4(float t, float4* B, float4* dB) {
    float t2 = t * t;
    float t3 = t2 * t;
    float om = 1.0f - t;
    float om2 = om * om;
    float om3 = om2 * om;
    B->x = om3 * (1.0f / 6.0f);
    B->y = (3.0f * t3 - 6.0f * t2 + 4.0f) * (1.0f / 6.0f);
    B->z = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) * (1.0f / 6.0f);
    B->w = t3 * (1.0f / 6.0f);
    dB->x = -0.5f * om2;
    dB->y = (1.5f * t2 - 2.0f * t);
    dB->z = (-1.5f * t2 + t + 0.5f);
    dB->w = 0.5f * t2;
}

inline float poly_z_basis(float dz, float invRc, float power, float* dphi_dz) {
    float x = fmin(fmax(dz, 0.0f) * invRc, 1.0f);
    float t = 1.0f - x;
    float phi = pow(t, power);
    if (dphi_dz != 0) {
        if (dz <= 0.0f || x >= 1.0f) *dphi_dz = 0.0f;
        else *dphi_dz = -power * pow(t, power - 1.0f) * invRc;
    }
    return phi;
}

// Doubling z/radial modes: t^ms, t^(2*ms), t^(4*ms), ...  (ms = m_start)
inline void poly_z_doubling_modes(float dist, float invRc, int m_start, int nz, float* phi, float* dphi) {
    float x = fmin(fmax(dist, 0.0f) * invRc, 1.0f);
    float t = 1.0f - x;
    bool active = (dist >= 0.0f) && (x < 1.0f);
    float tpow = 1.0f;
    for (int i = 0; i < m_start; i++) { tpow *= t; }
    float dtpow = 0.0f;
    if (active && m_start > 0) {
        float tprev = 1.0f;
        for (int i = 0; i < m_start - 1; i++) { tprev *= t; }
        dtpow = -(float)m_start * invRc * tprev;
    } else if (m_start <= 0) {
        tpow = 1.0f;
        dtpow = 0.0f;
    }
    for (int k = 0; k < nz; k++) {
        phi[k] = tpow;
        dphi[k] = active ? dtpow : 0.0f;
        if (k + 1 < nz) {
            float tp_n = tpow;
            tpow = tpow * tpow;
            dtpow = 2.0f * tp_n * dtpow;
        }
    }
}

// B-spline interpolate contact height h0(x,y) on same xy grid as coeffs
inline float cs_interp_h0(float4 Bx, float4 By, int ix, int iy, int ncx, int ncy, __global const float* h0) {
    float z0 = 0.0f;
    float bx[4] = {Bx.x, Bx.y, Bx.z, Bx.w};
    float by[4] = {By.x, By.y, By.z, By.w};
    for (int j = 0; j < 4; j++) {
        int jy = iy - 1 + j;
        if (jy < 0 || jy >= ncy) continue;
        for (int ii = 0; ii < 4; ii++) {
            int ixk = ix - 1 + ii;
            if (ixk < 0 || ixk >= ncx) continue;
            z0 += bx[ii] * by[j] * h0[jy * ncx + ixk];
        }
    }
    return z0;
}

inline void cs_interp_h0_grad(float4 Bx, float4 dBx, float4 By, float4 dBy, int ix, int iy,
    int ncx, int ncy, float dx, float dy, __global const float* h0, float* dz0_dx, float* dz0_dy) {
    float gx = 0.0f;
    float gy = 0.0f;
    float bx[4] = {Bx.x, Bx.y, Bx.z, Bx.w};
    float by[4] = {By.x, By.y, By.z, By.w};
    float dbx[4] = {dBx.x, dBx.y, dBx.z, dBx.w};
    float dby[4] = {dBy.x, dBy.y, dBy.z, dBy.w};
    for (int j = 0; j < 4; j++) {
        int jy = iy - 1 + j;
        if (jy < 0 || jy >= ncy) continue;
        for (int ii = 0; ii < 4; ii++) {
            int ixk = ix - 1 + ii;
            if (ixk < 0 || ixk >= ncx) continue;
            float h = h0[jy * ncx + ixk];
            gx += dbx[ii] * by[j] * h / dx;
            gy += bx[ii] * dby[j] * h / dy;
        }
    }
    *dz0_dx = gx;
    *dz0_dy = gy;
}

inline void cs_sep_stencil(float x, float y, float z,
    int ncx, int ncy, int nz, float x0, float y0, float z_start, float dx, float dy, float invRc, int m_start,
    __global const float* h0, int* out_n, int* out_ic, float* out_w) {
    float ux = (x - x0) / dx;
    float uy = (y - y0) / dy;
    float tx, ty;
    int ix = cs_bspline_cell(ux, &tx);
    int iy = cs_bspline_cell(uy, &ty);
    float4 Bx, dBx, By, dBy;
    bspline4(tx, &Bx, &dBx);
    bspline4(ty, &By, &dBy);
    float z0b = cs_interp_h0(Bx, By, ix, iy, ncx, ncy, h0);
    float dz = z - z0b - z_start;  // raw s; poly_z_doubling_modes handles s<0 (active=false → dphi=0)
    float bx[4] = {Bx.x, Bx.y, Bx.z, Bx.w};
    float by[4] = {By.x, By.y, By.z, By.w};
    float phi[8];
    float dphi[8];
    poly_z_doubling_modes(dz, invRc, m_start, nz, phi, dphi);
    int n = 0;
    for (int kz = 0; kz < nz; kz++) {
        for (int j = 0; j < 4; j++) {
            int jy = iy - 1 + j;
            if (jy < 0 || jy >= ncy) continue;
            for (int ii = 0; ii < 4; ii++) {
                int ixk = ix - 1 + ii;
                if (ixk < 0 || ixk >= ncx) continue;
                int ic = ixk + ncx * (jy + ncy * kz);
                float w = bx[ii] * by[j] * phi[kz];
                if (fabs(w) < 1e-20f) continue;
                out_ic[n] = ic;
                out_w[n] = w;
                n++;
            }
        }
    }
    *out_n = n;
}

// fcomp: 0=Fx, 1=Fy, 2=Fz  (F = -∇E, same as cs_eval_separable_fe_at)
inline void cs_sep_stencil_f(float x, float y, float z,
    int ncx, int ncy, int nz, float x0, float y0, float z_start, float dx, float dy, float invRc, int m_start,
    __global const float* h0, int fcomp, int* out_n, int* out_ic, float* out_w) {
    float ux = (x - x0) / dx;
    float uy = (y - y0) / dy;
    float tx, ty;
    int ix = cs_bspline_cell(ux, &tx);
    int iy = cs_bspline_cell(uy, &ty);
    float4 Bx, dBx, By, dBy;
    bspline4(tx, &Bx, &dBx);
    bspline4(ty, &By, &dBy);
    float z0b = cs_interp_h0(Bx, By, ix, iy, ncx, ncy, h0);
    float dz0_dx = 0.0f;
    float dz0_dy = 0.0f;
    cs_interp_h0_grad(Bx, dBx, By, dBy, ix, iy, ncx, ncy, dx, dy, h0, &dz0_dx, &dz0_dy);
    float dz = z - z0b - z_start;  // raw s; poly_z_doubling_modes handles s<0 (active=false → dphi=0)
    float bx[4] = {Bx.x, Bx.y, Bx.z, Bx.w};
    float by[4] = {By.x, By.y, By.z, By.w};
    float dbx[4] = {dBx.x, dBx.y, dBx.z, dBx.w};
    float dby[4] = {dBy.x, dBy.y, dBy.z, dBy.w};
    float phi[8];
    float dphi[8];
    poly_z_doubling_modes(dz, invRc, m_start, nz, phi, dphi);
    int n = 0;
    for (int kz = 0; kz < nz; kz++) {
        for (int j = 0; j < 4; j++) {
            int jy = iy - 1 + j;
            if (jy < 0 || jy >= ncy) continue;
            for (int ii = 0; ii < 4; ii++) {
                int ixk = ix - 1 + ii;
                if (ixk < 0 || ixk >= ncx) continue;
                int ic = ixk + ncx * (jy + ncy * kz);
                float bxy = bx[ii] * by[j];
                float dbxy_dx = dbx[ii] * by[j] / dx;
                float dbxy_dy = bx[ii] * dby[j] / dy;
                float w;
                if (fcomp == 0) {
                    w = -(dbxy_dx * phi[kz] - bxy * dphi[kz] * dz0_dx);
                } else if (fcomp == 1) {
                    w = -(dbxy_dy * phi[kz] - bxy * dphi[kz] * dz0_dy);
                } else {
                    w = -bxy * dphi[kz];
                }
                if (fabs(w) < 1e-20f) continue;
                out_ic[n] = ic;
                out_w[n] = w;
                n++;
            }
        }
    }
    *out_n = n;
}

// ===================== brute Morse+PLQH reference at query points =====================

__attribute__((reqd_work_group_size(CS_ATOM_TILE, 1, 1)))
__kernel void cs_brute_plqh_points(
    const int natoms,
    __global const float4* atoms,
    __global const float4* reqs,
    __global const float* queries,
    __global float4* out_fe,
    const int nq,
    const float4 GFFParams,
    const float4 PLQH
) {
    __local float4 LATOMS[CS_ATOM_TILE];
    __local float4 LREQS[CS_ATOM_TILE];
    const int iq = get_global_id(0);
    const int iL = get_local_id(0);
    const int nL = get_local_size(0);
    const bool active_q = (iq < nq);
    const float K = -GFFParams.y;
    const float R2damp = GFFParams.x * GFFParams.x;
    float3 pos = (float3)(0.0f, 0.0f, 0.0f);
    if (active_q) {
        pos = (float3)(queries[iq * 3 + 0], queries[iq * 3 + 1], queries[iq * 3 + 2]);
    }
    float4 fe = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j0 = 0; j0 < natoms; j0 += nL) {
        int j = j0 + iL;
        float4 ap = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 rq = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        if (j < natoms) {
            ap = atoms[j];
            rq = reqs[j];
        }
        LATOMS[iL] = ap;
        LREQS[iL] = rq;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int jl = 0; jl < nL; jl++) {
            int ja = jl + j0;
            if (active_q && ja < natoms) {
                float3 dp = pos - LATOMS[jl].xyz;
                float4 fej = getMorsePLQH(dp, LREQS[jl], PLQH, K, R2damp);
                fe.xyz -= fej.xyz;
                fe.w += fej.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active_q) out_fe[iq] = fe;
}

// ===================== separable eval =====================

inline float4 cs_eval_separable_fe_at(
    float x, float y, float z,
    __global const float* coeffs,
    __global const float* h0,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart
) {
    float x0 = origin_step.x;
    float y0 = origin_step.y;
    float z_start = origin_step.z;
    float dx = origin_step.w;
    float dy = dy_rc.x;
    float invRc = invRc_mstart.x;
    int m_start = (int)invRc_mstart.y;
    int ncx = meta.x;
    int ncy = meta.y;
    int nz = meta.z;
    float ux = (x - x0) / dx;
    float uy = (y - y0) / dy;
    float tx, ty;
    int ix = cs_bspline_cell(ux, &tx);
    int iy = cs_bspline_cell(uy, &ty);
    float4 Bx, dBx, By, dBy;
    bspline4(tx, &Bx, &dBx);
    bspline4(ty, &By, &dBy);
    float z0b = cs_interp_h0(Bx, By, ix, iy, ncx, ncy, h0);
    float dz0_dx = 0.0f;
    float dz0_dy = 0.0f;
    cs_interp_h0_grad(Bx, dBx, By, dBy, ix, iy, ncx, ncy, dx, dy, h0, &dz0_dx, &dz0_dy);
    float dz = z - z0b - z_start;  // raw s; poly_z_doubling_modes handles s<0 (active=false → dphi=0)
    float phi[8];
    float dphi[8];
    poly_z_doubling_modes(dz, invRc, m_start, nz, phi, dphi);
    float E = 0.0f;
    float dEdx = 0.0f;
    float dEdy = 0.0f;
    float dEdz = 0.0f;
    float bx[4] = {Bx.x, Bx.y, Bx.z, Bx.w};
    float by[4] = {By.x, By.y, By.z, By.w};
    float dbx[4] = {dBx.x, dBx.y, dBx.z, dBx.w};
    float dby[4] = {dBy.x, dBy.y, dBy.z, dBy.w};
    for (int kz = 0; kz < nz; kz++) {
        float ph = phi[kz];
        float dph = dphi[kz];
        for (int j = 0; j < 4; j++) {
            int jy = iy - 1 + j;
            if (jy < 0 || jy >= ncy) continue;
            for (int ii = 0; ii < 4; ii++) {
                int ixk = ix - 1 + ii;
                if (ixk < 0 || ixk >= ncx) continue;
                int ic = ixk + ncx * (jy + ncy * kz);
                float c = coeffs[ic];
                float bxy = bx[ii] * by[j];
                float dbxy_dx = dbx[ii] * by[j] / dx;
                float dbxy_dy = bx[ii] * dby[j] / dy;
                E += c * bxy * ph;
                dEdx += c * (dbxy_dx * ph - bxy * dph * dz0_dx);
                dEdy += c * (dbxy_dy * ph - bxy * dph * dz0_dy);
                dEdz += c * bxy * dph;
            }
        }
    }
    return (float4)(-dEdx, -dEdy, -dEdz, E);
}

__kernel void evalSeparableBsplinePoly(
    __global const float* queries,
    __global float4* out_fe,
    __global const float* coeffs,
    __global const float* h0,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart,
    const int nq
) {
    int i = get_global_id(0);
    if (i >= nq) return;
    float x = queries[i * 3 + 0];
    float y = queries[i * 3 + 1];
    float z = queries[i * 3 + 2];
    out_fe[i] = cs_eval_separable_fe_at(x, y, z, coeffs, h0, meta, origin_step, dy_rc, invRc_mstart);
}

__kernel void cs_sep_Av(
    __global const float* queries,
    __global const float* coeffs,
    __global float* out_y,
    __global const float* h0,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart,
    const int ns
) {
    int is = get_global_id(0);
    if (is >= ns) return;
    float x = queries[is * 3 + 0];
    float y = queries[is * 3 + 1];
    float z = queries[is * 3 + 2];
    int ncx = meta.x;
    int ncy = meta.y;
    int nz = meta.z;
    float x0 = origin_step.x;
    float y0 = origin_step.y;
    float z_start = origin_step.z;
    float dx = origin_step.w;
    float dy = dy_rc.x;
    float invRc = invRc_mstart.x;
    int m_start = (int)invRc_mstart.y;
    int ic[128];
    float w[128];
    int nn = 0;
    cs_sep_stencil(x, y, z, ncx, ncy, nz, x0, y0, z_start, dx, dy, invRc, m_start, h0, &nn, ic, w);
    float acc = 0.0f;
    for (int k = 0; k < nn; k++) { acc += coeffs[ic[k]] * w[k]; }
    out_y[is] = acc;
}

__kernel void cs_sep_Av_f(
    __global const float* queries,
    __global const float* coeffs,
    __global float* out_y,
    __global const float* h0,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart,
    const int ns,
    const int fcomp
) {
    int is = get_global_id(0);
    if (is >= ns) return;
    float x = queries[is * 3 + 0];
    float y = queries[is * 3 + 1];
    float z = queries[is * 3 + 2];
    int ncx = meta.x;
    int ncy = meta.y;
    int nz = meta.z;
    float x0 = origin_step.x;
    float y0 = origin_step.y;
    float z_start = origin_step.z;
    float dx = origin_step.w;
    float dy = dy_rc.x;
    float invRc = invRc_mstart.x;
    int m_start = (int)invRc_mstart.y;
    int ic[128];
    float w[128];
    int nn = 0;
    cs_sep_stencil_f(x, y, z, ncx, ncy, nz, x0, y0, z_start, dx, dy, invRc, m_start, h0, fcomp, &nn, ic, w);
    float acc = 0.0f;
    for (int k = 0; k < nn; k++) { acc += coeffs[ic[k]] * w[k]; }
    out_y[is] = acc;
}

__kernel void cs_sep_Atv(
    __global const float* queries,
    __global const float* vec_y,
    __global float* out_x,
    __global const float* h0,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart,
    const int ns
) {
    int is = get_global_id(0);
    if (is >= ns) return;
    float x = queries[is * 3 + 0];
    float y = queries[is * 3 + 1];
    float z = queries[is * 3 + 2];
    int ncx = meta.x;
    int ncy = meta.y;
    int nz = meta.z;
    float x0 = origin_step.x;
    float y0 = origin_step.y;
    float z_start = origin_step.z;
    float dx = origin_step.w;
    float dy = dy_rc.x;
    float invRc = invRc_mstart.x;
    int m_start = (int)invRc_mstart.y;
    int ic[128];
    float w[128];
    int nn = 0;
    cs_sep_stencil(x, y, z, ncx, ncy, nz, x0, y0, z_start, dx, dy, invRc, m_start, h0, &nn, ic, w);
    float vy = vec_y[is];
    for (int k = 0; k < nn; k++) {
        atomic_add_f(&out_x[ic[k]], w[k] * vy);
    }
}

__kernel void cs_sep_Atv_w(
    __global const float* queries,
    __global const float* vec_y,
    __global float* out_x,
    __global const float* h0,
    __global const float* sample_w,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart,
    const int ns,
    const float row_scale
) {
    int is = get_global_id(0);
    if (is >= ns) return;
    float x = queries[is * 3 + 0];
    float y = queries[is * 3 + 1];
    float z = queries[is * 3 + 2];
    int ncx = meta.x;
    int ncy = meta.y;
    int nz = meta.z;
    float x0 = origin_step.x;
    float y0 = origin_step.y;
    float z_start = origin_step.z;
    float dx = origin_step.w;
    float dy = dy_rc.x;
    float invRc = invRc_mstart.x;
    int m_start = (int)invRc_mstart.y;
    int ic[128];
    float w[128];
    int nn = 0;
    cs_sep_stencil(x, y, z, ncx, ncy, nz, x0, y0, z_start, dx, dy, invRc, m_start, h0, &nn, ic, w);
    float vy = vec_y[is] * sample_w[is] * row_scale;
    for (int k = 0; k < nn; k++) {
        atomic_add_f(&out_x[ic[k]], w[k] * vy);
    }
}

__kernel void cs_sep_Atv_f_w(
    __global const float* queries,
    __global const float* vec_y,
    __global float* out_x,
    __global const float* h0,
    __global const float* sample_w,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart,
    const int ns,
    const int fcomp,
    const float force_weight
) {
    int is = get_global_id(0);
    if (is >= ns) return;
    float x = queries[is * 3 + 0];
    float y = queries[is * 3 + 1];
    float z = queries[is * 3 + 2];
    int ncx = meta.x;
    int ncy = meta.y;
    int nz = meta.z;
    float x0 = origin_step.x;
    float y0 = origin_step.y;
    float z_start = origin_step.z;
    float dx = origin_step.w;
    float dy = dy_rc.x;
    float invRc = invRc_mstart.x;
    int m_start = (int)invRc_mstart.y;
    int ic[128];
    float w[128];
    int nn = 0;
    cs_sep_stencil_f(x, y, z, ncx, ncy, nz, x0, y0, z_start, dx, dy, invRc, m_start, h0, fcomp, &nn, ic, w);
    float vy = vec_y[is] * sample_w[is] * force_weight;
    for (int k = 0; k < nn; k++) {
        atomic_add_f(&out_x[ic[k]], w[k] * vy);
    }
}

__kernel void cs_sep_Atv_masked(
    __global const float* queries,
    __global const float* vec_y,
    __global float* out_x,
    __global const float* h0,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart,
    const int ns,
    const int coeff_i0,
    const int coeff_i1
) {
    int is = get_global_id(0);
    if (is >= ns) return;
    float x = queries[is * 3 + 0];
    float y = queries[is * 3 + 1];
    float z = queries[is * 3 + 2];
    int ncx = meta.x;
    int ncy = meta.y;
    int nz = meta.z;
    float x0 = origin_step.x;
    float y0 = origin_step.y;
    float z_start = origin_step.z;
    float dx = origin_step.w;
    float dy = dy_rc.x;
    float invRc = invRc_mstart.x;
    int m_start = (int)invRc_mstart.y;
    int ic[128];
    float w[128];
    int nn = 0;
    cs_sep_stencil(x, y, z, ncx, ncy, nz, x0, y0, z_start, dx, dy, invRc, m_start, h0, &nn, ic, w);
    float vy = vec_y[is];
    for (int k = 0; k < nn; k++) {
        if (ic[k] >= coeff_i0 && ic[k] < coeff_i1) {
            atomic_add_f(&out_x[ic[k]], w[k] * vy);
        }
    }
}

// ===================== PIC eval — 16×16 tiled, cooperative atom preload =====================

__attribute__((reqd_work_group_size(CS_TILE, CS_TILE, 1)))
__kernel void cs_pic_eval_tile16(
    __global const float* queries,
    __global float4* out_fe,
    __global const float4* atoms,
    __global const float* atom_coeffs,
    __global const int* bucket_atoms,
    __global const int* bucket_offsets,
    const int4 meta,
    const float4 bucket_meta,
    const float m_start,
    const int4 grid_meta,
    const float4 query_origin_step
) {
    __local float4 LATOMS[CS_PIC_LOCAL_MAX];
    __local float LC[CS_PIC_LOCAL_MAX * 4];
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tile_x = get_group_id(0);
    const int tile_y = get_group_id(1);
    const int ngx = grid_meta.x;
    const int ngy = grid_meta.y;
    const int nq = grid_meta.z;
    const int ix = tile_x * CS_TILE + tx;
    const int iy = tile_y * CS_TILE + ty;
    const int iq = iy * ngx + ix;
    const int nat = meta.x;
    const int nmodes = meta.y;
    const int nbuckets = meta.z;
    const int nbx = meta.w;
    const int nby = (nbuckets + nbx - 1) / nbx;
    float x0 = bucket_meta.x;
    float y0 = bucket_meta.y;
    float cell = bucket_meta.z;
    float invRc = bucket_meta.w;
    float qx0 = query_origin_step.x;
    float qy0 = query_origin_step.y;
    float qdx = query_origin_step.z;
    float qdy = query_origin_step.w;
    float xt0 = qx0 + (float)(tile_x * CS_TILE) * qdx;
    float yt0 = qy0 + (float)(tile_y * CS_TILE) * qdy;
    float xt1 = xt0 + (float)(CS_TILE - 1) * qdx;
    float yt1 = yt0 + (float)(CS_TILE - 1) * qdy;
    float Rc = 1.0f / invRc;
    float xmin = fmin(xt0, xt1) - Rc;
    float xmax = fmax(xt0, xt1) + Rc;
    float ymin = fmin(yt0, yt1) - Rc;
    float ymax = fmax(yt0, yt1) + Rc;
    float x = qx0 + (float)ix * qdx;
    float y = qy0 + (float)iy * qdy;
    float z = (iq < nq) ? queries[iq * 3 + 2] : 0.0f;
    int bx0 = (int)floor((xmin - x0) / cell);
    int by0 = (int)floor((ymin - y0) / cell);
    int bx1 = (int)floor((xmax - x0) / cell);
    int by1 = (int)floor((ymax - y0) / cell);
    if (bx0 < 0) bx0 = 0;
    if (by0 < 0) by0 = 0;
    if (bx1 >= nbx) bx1 = nbx - 1;
    if (by1 >= nby) by1 = nby - 1;
    const int tid = ty * CS_TILE + tx;
    const int nthreads = CS_TILE * CS_TILE;
    __local int nload_l;
    if (tid == 0) { nload_l = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int by = by0; by <= by1; by++) {
        for (int bx = bx0; bx <= bx1; bx++) {
            int bid = by * nbx + bx;
            if (bid < 0 || bid >= nbuckets) continue;
            int i0 = bucket_offsets[bid];
            int i1 = bucket_offsets[bid + 1];
            for (int ia = i0 + tid; ia < i1; ia += nthreads) {
                int at = bucket_atoms[ia];
                if (at < 0 || at >= nat) continue;
                int slot = atomic_inc(&nload_l);
                if (slot < CS_PIC_LOCAL_MAX) {
                    LATOMS[slot] = atoms[at];
                    for (int m = 0; m < nmodes; m++) { LC[slot * nmodes + m] = atom_coeffs[at * nmodes + m]; }
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int nload = nload_l;
    if (nload > CS_PIC_LOCAL_MAX) { nload = CS_PIC_LOCAL_MAX; }
    if (iq >= nq) return;
    z = queries[iq * 3 + 2];
    float E = 0.0f;
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;
    for (int ia = 0; ia < nload; ia++) {
        float4 ap = LATOMS[ia];
        float dxp = x - ap.x;
        float dyp = y - ap.y;
        float dzp = z - ap.z;
        float r2 = dxp * dxp + dyp * dyp + dzp * dzp;
        float r = sqrt(r2 + 1e-20f);
        if (r >= 1.0f / invRc) continue;
        float phi[8];
        float dphi_dr[8];
        int ms = (int)m_start;
        poly_z_doubling_modes(r, invRc, ms, nmodes, phi, dphi_dr);
        for (int m = 0; m < nmodes; m++) {
            float c = LC[ia * nmodes + m];
            E += c * phi[m];
            if (r > 1e-8f) {
                float dE_dr = c * dphi_dr[m];
                Fx -= dE_dr * (dxp / r);
                Fy -= dE_dr * (dyp / r);
                Fz -= dE_dr * (dzp / r);
            }
        }
    }
    out_fe[iq] = (float4)(Fx, Fy, Fz, E);
}

// PIC matrix-free Av / Atv for CG fit
__kernel void cs_pic_Av(
    __global const float* queries,
    __global const float* atom_coeffs,
    __global float* out_y,
    __global const float4* atoms,
    __global const int* bucket_atoms,
    __global const int* bucket_offsets,
    const int4 meta,
    const float4 bucket_meta,
    const float m_start,
    const int nq
) {
    int i = get_global_id(0);
    if (i >= nq) return;
    float x = queries[i * 3 + 0];
    float y = queries[i * 3 + 1];
    float z = queries[i * 3 + 2];
    int nat = meta.x;
    int nmodes = meta.y;
    int nbuckets = meta.z;
    int nbx = meta.w;
    int nby = (nbuckets + nbx - 1) / nbx;
    float x0 = bucket_meta.x;
    float y0 = bucket_meta.y;
    float cell = bucket_meta.z;
    float invRc = bucket_meta.w;
    int bx = (int)floor((x - x0) / cell);
    int by = (int)floor((y - y0) / cell);
    float E = 0.0f;
    for (int dyb = -1; dyb <= 1; dyb++) {
        for (int dxb = -1; dxb <= 1; dxb++) {
            int bix = bx + dxb;
            int biy = by + dyb;
            if (bix < 0 || biy < 0 || bix >= nbx || biy >= nby) continue;
            int bid = biy * nbx + bix;
            if (bid < 0 || bid >= nbuckets) continue;
            int i0 = bucket_offsets[bid];
            int i1 = bucket_offsets[bid + 1];
            for (int ia = i0; ia < i1; ia++) {
                int at = bucket_atoms[ia];
                if (at < 0 || at >= nat) continue;
                float4 ap = atoms[at];
                float dxp = x - ap.x;
                float dyp = y - ap.y;
                float dzp = z - ap.z;
                float r = sqrt(dxp * dxp + dyp * dyp + dzp * dzp + 1e-20f);
                if (r >= 1.0f / invRc) continue;
                float phi[8];
                float dphi[8];
                int ms = (int)m_start;
                poly_z_doubling_modes(r, invRc, ms, nmodes, phi, dphi);
                for (int m = 0; m < nmodes; m++) {
                    E += atom_coeffs[at * nmodes + m] * phi[m];
                }
            }
        }
    }
    out_y[i] = E;
}

__kernel void cs_pic_Atv(
    __global const float* queries,
    __global const float* vec_y,
    __global float* out_c,
    __global const float4* atoms,
    __global const int* bucket_atoms,
    __global const int* bucket_offsets,
    const int4 meta,
    const float4 bucket_meta,
    const float m_start,
    const int nq
) {
    int i = get_global_id(0);
    if (i >= nq) return;
    float x = queries[i * 3 + 0];
    float y = queries[i * 3 + 1];
    float z = queries[i * 3 + 2];
    float vy = vec_y[i];
    int nat = meta.x;
    int nmodes = meta.y;
    int nbuckets = meta.z;
    int nbx = meta.w;
    int nby = (nbuckets + nbx - 1) / nbx;
    float x0 = bucket_meta.x;
    float y0 = bucket_meta.y;
    float cell = bucket_meta.z;
    float invRc = bucket_meta.w;
    int bx = (int)floor((x - x0) / cell);
    int by = (int)floor((y - y0) / cell);
    for (int dyb = -1; dyb <= 1; dyb++) {
        for (int dxb = -1; dxb <= 1; dxb++) {
            int bix = bx + dxb;
            int biy = by + dyb;
            if (bix < 0 || biy < 0 || bix >= nbx || biy >= nby) continue;
            int bid = biy * nbx + bix;
            if (bid < 0 || bid >= nbuckets) continue;
            int i0 = bucket_offsets[bid];
            int i1 = bucket_offsets[bid + 1];
            for (int ia = i0; ia < i1; ia++) {
                int at = bucket_atoms[ia];
                if (at < 0 || at >= nat) continue;
                float4 ap = atoms[at];
                float dxp = x - ap.x;
                float dyp = y - ap.y;
                float dzp = z - ap.z;
                float r = sqrt(dxp * dxp + dyp * dyp + dzp * dzp + 1e-20f);
                if (r >= 1.0f / invRc) continue;
                float phi[8];
                float dphi[8];
                int ms = (int)m_start;
                poly_z_doubling_modes(r, invRc, ms, nmodes, phi, dphi);
                for (int m = 0; m < nmodes; m++) {
                    atomic_add_f(&out_c[at * nmodes + m], phi[m] * vy);
                }
            }
        }
    }
}

__kernel void cs_pic_Atv_w(
    __global const float* queries,
    __global const float* vec_y,
    __global float* out_c,
    __global const float* sample_w,
    __global const float4* atoms,
    __global const int* bucket_atoms,
    __global const int* bucket_offsets,
    const int4 meta,
    const float4 bucket_meta,
    const float m_start,
    const int nq
) {
    int i = get_global_id(0);
    if (i >= nq) return;
    float x = queries[i * 3 + 0];
    float y = queries[i * 3 + 1];
    float z = queries[i * 3 + 2];
    float vy = vec_y[i] * sample_w[i];
    int nat = meta.x;
    int nmodes = meta.y;
    int nbuckets = meta.z;
    int nbx = meta.w;
    int nby = (nbuckets + nbx - 1) / nbx;
    float x0 = bucket_meta.x;
    float y0 = bucket_meta.y;
    float cell = bucket_meta.z;
    float invRc = bucket_meta.w;
    int bx = (int)floor((x - x0) / cell);
    int by = (int)floor((y - y0) / cell);
    for (int dyb = -1; dyb <= 1; dyb++) {
        for (int dxb = -1; dxb <= 1; dxb++) {
            int bix = bx + dxb;
            int biy = by + dyb;
            if (bix < 0 || biy < 0 || bix >= nbx || biy >= nby) continue;
            int bid = biy * nbx + bix;
            if (bid < 0 || bid >= nbuckets) continue;
            int i0 = bucket_offsets[bid];
            int i1 = bucket_offsets[bid + 1];
            for (int ia = i0; ia < i1; ia++) {
                int at = bucket_atoms[ia];
                if (at < 0 || at >= nat) continue;
                float4 ap = atoms[at];
                float dxp = x - ap.x;
                float dyp = y - ap.y;
                float dzp = z - ap.z;
                float r = sqrt(dxp * dxp + dyp * dyp + dzp * dzp + 1e-20f);
                if (r >= 1.0f / invRc) continue;
                float phi[8];
                float dphi[8];
                int ms = (int)m_start;
                poly_z_doubling_modes(r, invRc, ms, nmodes, phi, dphi);
                for (int m = 0; m < nmodes; m++) {
                    atomic_add_f(&out_c[at * nmodes + m], phi[m] * vy);
                }
            }
        }
    }
}

// PIC field at one point (shared by evalRadialPIC and PP-AFM relaxation)
inline float4 cs_eval_pic_fe_at(
    float x, float y, float z,
    __global const float4* atoms,
    __global const float* atom_coeffs,
    __global const int* bucket_atoms,
    __global const int* bucket_offsets,
    const int4 meta,
    const float4 bucket_meta,
    const float m_start
) {
    float x0 = bucket_meta.x;
    float y0 = bucket_meta.y;
    float cell = bucket_meta.z;
    float invRc = bucket_meta.w;
    int nat = meta.x;
    int nmodes = meta.y;
    int nbuckets = meta.z;
    int nbx_host = meta.w;
    int nby_host = (nbuckets + nbx_host - 1) / nbx_host;
    int bx = (int)floor((x - x0) / cell);
    int by = (int)floor((y - y0) / cell);
    float E = 0.0f;
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;
    for (int dyb = -1; dyb <= 1; dyb++) {
        for (int dxb = -1; dxb <= 1; dxb++) {
            int bix = bx + dxb;
            int biy = by + dyb;
            if (bix < 0 || biy < 0 || bix >= nbx_host || biy >= nby_host) continue;
            int bid = biy * nbx_host + bix;
            if (bid < 0 || bid >= nbuckets) continue;
            int i0 = bucket_offsets[bid];
            int i1 = bucket_offsets[bid + 1];
            for (int ia = i0; ia < i1; ia++) {
                int at = bucket_atoms[ia];
                if (at < 0 || at >= nat) continue;
                float4 ap = atoms[at];
                float dxp = x - ap.x;
                float dyp = y - ap.y;
                float dzp = z - ap.z;
                float r2 = dxp * dxp + dyp * dyp + dzp * dzp;
                float r = sqrt(r2 + 1e-20f);
                if (r >= 1.0f / invRc) continue;
                float phi[8];
                float dphi_dr[8];
                int ms = (int)m_start;
                poly_z_doubling_modes(r, invRc, ms, nmodes, phi, dphi_dr);
                for (int m = 0; m < nmodes; m++) {
                    float c = atom_coeffs[at * nmodes + m];
                    E += c * phi[m];
                    if (r > 1e-8f) {
                        float dE_dr = c * dphi_dr[m];
                        Fx -= dE_dr * (dxp / r);
                        Fy -= dE_dr * (dyp / r);
                        Fz -= dE_dr * (dzp / r);
                    }
                }
            }
        }
    }
    return (float4)(Fx, Fy, Fz, E);
}

// legacy 1D PIC eval (fallback for irregular query lists)
__kernel void evalRadialPIC(
    __global const float* queries,
    __global float4* out_fe,
    __global const float4* atoms,
    __global const float* atom_coeffs,
    __global const int* bucket_atoms,
    __global const int* bucket_offsets,
    const int4 meta,
    const float4 bucket_meta,
    const float m_start,
    const int nq
) {
    int i = get_global_id(0);
    if (i >= nq) return;
    float x = queries[i * 3 + 0];
    float y = queries[i * 3 + 1];
    float z = queries[i * 3 + 2];
    out_fe[i] = cs_eval_pic_fe_at(x, y, z, atoms, atom_coeffs, bucket_atoms, bucket_offsets, meta, bucket_meta, m_start);
}

// ===================== contact_pme: particle-mesh contact surface (contract v2) =====================
// Nonperiodic, zero-padded boundary. NO PBC wrapping. V(r) ≈ V_mesh + Σ_i V_core_i; F = -∇E.
// Python prototypes: spammm/surfaces/CoarseMesh.py (mesh), PICCore.py (core), PMESplit.py (split).
// Stencil reference: gridFF.cl:72-93 (basis/dbasis) — copied formulas only, NOT PBC wrapping.

#define CS_PME_NMODES 5           // doubling-power core modes: p_m = 2,4,8,16,32
#define CS_PME_CORE_MAX_CAND 512  // safety cap on core candidates per query (fail-loud overflow)

// ---- cardinal cubic B-spline basis (matches gridFF.cl:72-93 and CoarseMesh._basis) ----
inline float4 cs_pme_basis(float u) {
    const float inv6 = 1.0f / 6.0f;
    const float u2 = u * u;
    const float t = 1.0f - u;
    return (float4)(inv6 * t * t * t,
                   inv6 * (3.0f * u2 * (u - 2.0f) + 4.0f),
                   inv6 * (3.0f * u * (1.0f + u - u2) + 1.0f),
                   inv6 * u2 * u);
}

inline float4 cs_pme_dbasis(float u) {
    const float u2 = u * u;
    const float t = 1.0f - u;
    return (float4)(-0.5f * t * t,
                    0.5f * (3.0f * u2 - 4.0f * u),
                    0.5f * (-3.0f * u2 + 2.0f * u + 1.0f),
                    0.5f * u2);
}

// Bounded scalar tricubic B-spline interpolation with analytic gradient.
// Zero-padding boundary (NO PBC): full 4×4×4 stencil must fit in [0,n-1].
// Layout: C-order (nx,ny,nz), z fastest → index = (ix*ny + iy)*nz + iz.
// Returns float4 (Fx,Fy,Fz,E). *out_status: 0=OK, 1=stencil out of bounds (→ NAN).
inline float4 cs_pme_tricubic_eval(
    float x, float y, float z,
    __global const float* coeffs, int nx, int ny, int nz,
    float ox, float oy, float oz, float h,
    int* out_status)
{
    float inv_h = 1.0f / h;
    float fx = (x - ox) * inv_h;
    float fy = (y - oy) * inv_h;
    float fz = (z - oz) * inv_h;
    int ix = (int)floor(fx); int iy = (int)floor(fy); int iz = (int)floor(fz);
    float ux = fx - (float)ix; float uy = fy - (float)iy; float uz = fz - (float)iz;
    int i0x = ix - 1, i0y = iy - 1, i0z = iz - 1;  // stencil base ix-1..ix+2
    if (i0x < 0 || i0y < 0 || i0z < 0 || i0x + 3 >= nx || i0y + 3 >= ny || i0z + 3 >= nz) {
        *out_status = 1;
        return (float4)(NAN, NAN, NAN, NAN);
    }
    float4 bx = cs_pme_basis(ux), by = cs_pme_basis(uy), bz = cs_pme_basis(uz);
    float4 dbx = cs_pme_dbasis(ux) * inv_h, dby = cs_pme_dbasis(uy) * inv_h, dbz = cs_pme_dbasis(uz) * inv_h;
    float bxa[4] = {bx.x, bx.y, bx.z, bx.w};
    float bya[4] = {by.x, by.y, by.z, by.w};
    float dbxa[4] = {dbx.x, dbx.y, dbx.z, dbx.w};
    float dbya[4] = {dby.x, dby.y, dby.z, dby.w};
    float e = 0.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f;
    for (int a = 0; a < 4; a++) {
        int ia = i0x + a; float cx = bxa[a], dx = dbxa[a];
        for (int b = 0; b < 4; b++) {
            int ib = i0y + b; float cy = bya[b], dy = dbya[b];
            float cxy = cx * cy;
            int ixyz = (ia * ny + ib) * nz + i0z;
            float4 v = vload4(0, coeffs + ixyz);  // z-fastest: one contiguous 4-tap load
            float ez = dot(v, bz);
            float dez = dot(v, dbz);
            e  += cxy * ez;
            gx += dx * cy * ez;
            gy += cx * dy * ez;
            gz += cxy * dez;
        }
    }
    *out_status = 0;
    return (float4)(-gx, -gy, -gz, e);  // F = -∇E
}

// 5-mode doubling-power core basis: phi_m(r) = t^p_m, p_m = 2,4,8,16,32.
// t = (r_b - r)/(r_b - r_lo_i). Exactly zero for r >= r_b. Matches PICCore.core_basis.
// r_b is per-atom (plateau: r_lo + Δ_in + Δ_b). Derivatives via chain rule on repeated squaring.
inline void cs_pme_core_basis(float r, float r_lo_i, float r_b, float* phi, float* dphi) {
    float D = r_b - r_lo_i;
    float t = (r_b - r) / D;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    // Active for all r < r_b. Below r_lo, t clips to 1 → constant residual (dphi=0).
    // Needed for PP-AFM close approach; mesh soft field (PAW) remains valid at r→0.
    bool active = (r < r_b);
    float dt = (r > r_lo_i && r < r_b) ? (-1.0f / D) : 0.0f;  // flat for r<=r_lo
    // powers 2,4,8,16,32 via successive squaring from t^2
    float t2 = t * t;         float dt2 = 2.0f * t * dt;
    float t4 = t2 * t2;       float dt4 = 2.0f * t2 * dt2;
    float t8 = t4 * t4;       float dt8 = 2.0f * t4 * dt4;
    float t16 = t8 * t8;      float dt16 = 2.0f * t8 * dt8;
    float t32 = t16 * t16;    float dt32 = 2.0f * t16 * dt16;
    phi[0] = active ? t2  : 0.0f;  dphi[0] = active ? dt2  : 0.0f;
    phi[1] = active ? t4  : 0.0f;  dphi[1] = active ? dt4  : 0.0f;
    phi[2] = active ? t8  : 0.0f;  dphi[2] = active ? dt8  : 0.0f;
    phi[3] = active ? t16 : 0.0f;  dphi[3] = active ? dt16 : 0.0f;
    phi[4] = active ? t32 : 0.0f;  dphi[4] = active ? dt32 : 0.0f;
}

inline float2 cs_pme_core_reduce(const float* phi, const float* dphi, float c0, float c1, float c2, float c3, float c4) {
    return (float2)(c0 * phi[0] + c1 * phi[1] + c2 * phi[2] + c3 * phi[3] + c4 * phi[4],
                    c0 * dphi[0] + c1 * dphi[1] + c2 * dphi[2] + c3 * dphi[3] + c4 * dphi[4]);
}

// Core field V_core at one point via XY buckets (3×3 lookup, cell_size >= r_core_max).
// atoms[i] = (x, y, z, r_lo_i). atom_coeffs[i*NMODES + m]. Per-atom r_lo_i (not global).
// d_span = r_b - r_lo (= Δ_in+Δ_b for plateau); r_b_i = r_lo_i + d_span.
// Returns float4 (Fx,Fy,Fz,E). Telemetry via pointers.
// *out_status: 0=OK, bit 2 (4)=bucket overflow (candidates > CS_PME_CORE_MAX_CAND, fail-loud).
// r < r_lo is NOT an error — basis clamps to t=1 (AFM close-approach).
inline float4 cs_pme_core_eval_at(
    float x, float y, float z,
    __global const float4* atoms, __global const float* atom_coeffs,
    __global const int* bucket_atoms, __global const int* bucket_offsets,
    int nat, int nbx, int nby, int nbuckets,
    float x0, float y0, float cell, float d_span,
    int* out_status, float* out_min_r, int* out_offender, int* out_overflow)
{
    float E = 0.0f, Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    int status = 0;
    float min_r2 = 1e30f;
    int offender = -1;
    int n_cand = 0;
    int bx = (int)floor((x - x0) / cell);
    int by = (int)floor((y - y0) / cell);
    for (int dyb = -1; dyb <= 1; dyb++) {
        int biy = by + dyb;
        if (biy < 0 || biy >= nby) continue;
        for (int dxb = -1; dxb <= 1; dxb++) {
            int bix = bx + dxb;
            if (bix < 0 || bix >= nbx) continue;
            int bid = biy * nbx + bix;
            if (bid < 0 || bid >= nbuckets) continue;
            int i0 = bucket_offsets[bid];
            int i1 = bucket_offsets[bid + 1];
            for (int ia = i0; ia < i1; ia++) {
                int at = bucket_atoms[ia];
                if (at < 0 || at >= nat) continue;
                n_cand++;
                float4 ap = atoms[at];
                float r_lo_i = ap.w;
                float r_b_i = r_lo_i + d_span;
                float dxp = x - ap.x, dyp = y - ap.y, dzp = z - ap.z;
                float r2 = dxp * dxp + dyp * dyp + dzp * dzp + 1e-20f;
                if (r2 < min_r2) { min_r2 = r2; offender = at; }
                if (r2 >= r_b_i * r_b_i) continue;
                float r = sqrt(r2);
                float phi[CS_PME_NMODES], dphi[CS_PME_NMODES];
                cs_pme_core_basis(r, r_lo_i, r_b_i, phi, dphi);
                int ic = at * CS_PME_NMODES;
                float4 c03 = vload4(0, atom_coeffs + ic);
                float2 ed = cs_pme_core_reduce(phi, dphi, c03.x, c03.y, c03.z, c03.w, atom_coeffs[ic + 4]);
                E += ed.x;
                if (r > 1e-8f) {
                    float fr = -ed.y / r;
                    Fx += fr * dxp;
                    Fy += fr * dyp;
                    Fz += fr * dzp;
                }
            }
        }
    }
    int overflow = n_cand > CS_PME_CORE_MAX_CAND ? (n_cand - CS_PME_CORE_MAX_CAND) : 0;
    if (overflow > 0) status |= 4;
    *out_status = status; *out_min_r = offender >= 0 ? sqrt(min_r2) : 1e30f; *out_offender = offender; *out_overflow = overflow;
    return (float4)(Fx, Fy, Fz, E);
}

// Direct compact-core sum over atoms/coefficients cooperatively cached once per workgroup.
// Intended for small/medium molecules where bucket indirection costs more than the skipped atoms.
// The caller must preload LATOMS/LCOEFFS and synchronize before invoking this function.
inline float4 cs_pme_core_eval_local_at(
    float x, float y, float z,
    __local const float4* LATOMS, __local const float* LCOEFFS,
    int nat, float d_span,
    int* out_status, float* out_min_r, int* out_offender, int* out_overflow)
{
    float E = 0.0f, Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    float min_r2 = 1e30f;
    int offender = -1;
    for (int at = 0; at < nat; at++) {
        float4 ap = LATOMS[at];
        float r_b_i = ap.w + d_span;
        float dxp = x - ap.x, dyp = y - ap.y, dzp = z - ap.z;
        float r2 = dxp * dxp + dyp * dyp + dzp * dzp + 1e-20f;
        if (r2 < min_r2) { min_r2 = r2; offender = at; }
        if (r2 >= r_b_i * r_b_i) continue;
        float r = sqrt(r2);
        float phi[CS_PME_NMODES], dphi[CS_PME_NMODES];
        cs_pme_core_basis(r, ap.w, r_b_i, phi, dphi);
        int ic = at * CS_PME_NMODES;
        float4 c03 = vload4(0, LCOEFFS + ic);
        float2 ed = cs_pme_core_reduce(phi, dphi, c03.x, c03.y, c03.z, c03.w, LCOEFFS[ic + 4]);
        E += ed.x;
        if (r > 1e-8f) {
            float fr = -ed.y / r;
            Fx += fr * dxp;
            Fy += fr * dyp;
            Fz += fr * dzp;
        }
    }
    int overflow = 0;  // Direct all-atom loop has no bucket candidate cap or truncation.
    *out_status = 0;
    *out_min_r = offender >= 0 ? sqrt(min_r2) : 1e30f;
    *out_offender = offender;
    *out_overflow = overflow;
    return (float4)(Fx, Fy, Fz, E);
}

// Combined PME eval at one point: V_mesh + V_core. F = -∇E. Shared by evalContactPME and relaxation.
// Returns float4 (Fx,Fy,Fz,E). Telemetry via pointers.
// *out_status: 0=OK, bit 0 (1)=mesh stencil OOB, bit 1 (2)=core domain violation, bit 2 (4)=overflow.
// Invalid (mesh OOB or domain violation) → non-finite (NAN) outputs.
inline float4 cs_eval_contact_pme_at(
    float x, float y, float z,
    __global const float* mesh_coeffs, int nx, int ny, int nz,
    float ox, float oy, float oz, float h,
    __global const float4* atoms, __global const float* atom_coeffs,
    __global const int* bucket_atoms, __global const int* bucket_offsets,
    int nat, int nbx, int nby, int nbuckets,
    float bx0, float by0, float cell, float r_cut,
    int* out_status, float* out_min_r, int* out_offender, int* out_overflow)
{
    int mesh_status = 0;
    float4 fm = cs_pme_tricubic_eval(x, y, z, mesh_coeffs, nx, ny, nz, ox, oy, oz, h, &mesh_status);
    int core_status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
    float4 fc = cs_pme_core_eval_at(x, y, z, atoms, atom_coeffs, bucket_atoms, bucket_offsets,
                                    nat, nbx, nby, nbuckets, bx0, by0, cell, r_cut,
                                    &core_status, &min_r, &offender, &overflow);
    int status = mesh_status | core_status;
    *out_status = status; *out_min_r = min_r; *out_offender = offender; *out_overflow = overflow;
    if (status & 1 || status & 2) return (float4)(NAN, NAN, NAN, NAN);
    float E = fm.w + fc.w;
    float3 F = fm.xyz + fc.xyz;
    return (float4)(F.x, F.y, F.z, E);
}

inline float4 cs_eval_contact_pme_local_at(
    float x, float y, float z,
    __global const float* mesh_coeffs, int nx, int ny, int nz,
    float ox, float oy, float oz, float h,
    __local const float4* LATOMS, __local const float* LCOEFFS,
    int nat, float d_span,
    int* out_status, float* out_min_r, int* out_offender, int* out_overflow)
{
    int mesh_status = 0;
    float4 fm = cs_pme_tricubic_eval(x, y, z, mesh_coeffs, nx, ny, nz, ox, oy, oz, h, &mesh_status);
    int core_status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
    float4 fc = cs_pme_core_eval_local_at(x, y, z, LATOMS, LCOEFFS, nat, d_span, &core_status, &min_r, &offender, &overflow);
    int status = mesh_status | core_status;
    *out_status = status; *out_min_r = min_r; *out_offender = offender; *out_overflow = overflow;
    if (status & 1 || status & 2) return (float4)(NAN, NAN, NAN, NAN);
    return (float4)(fm.xyz + fc.xyz, fm.w + fc.w);
}

// ===================== contact_pme tiled evaluator (WG-local mesh slab) =====================

// Tricubic eval from a WG-cached coefficient slab (local memory, z-fastest):
// LMESH[(ix_l*nyL + iy_l)*nzL + iz_l] holds global coeff node (ix0+ix_l, iy0+iy_l, kz0+iz_l).
// Caller must guarantee the 4x4x4 stencil base lies inside the slab —
// cs_eval_contact_pme_tile_at checks this and falls back to global eval otherwise.
inline float4 cs_pme_tricubic_eval_tile(
    float x, float y, float z,
    __local const float* coeffs,
    int ix0, int iy0, int kz0, int nxL, int nyL, int nzL,
    float ox, float oy, float oz, float h)
{
    float inv_h = 1.0f / h;
    float fx = (x - ox) * inv_h;
    float fy = (y - oy) * inv_h;
    float fz = (z - oz) * inv_h;
    int ix = (int)floor(fx); int iy = (int)floor(fy); int iz = (int)floor(fz);
    float ux = fx - (float)ix; float uy = fy - (float)iy; float uz = fz - (float)iz;
    int i0x = ix - 1 - ix0, i0y = iy - 1 - iy0, i0z = iz - 1 - kz0;  // local stencil base
    float4 bx = cs_pme_basis(ux), by = cs_pme_basis(uy), bz = cs_pme_basis(uz);
    float4 dbx = cs_pme_dbasis(ux) * inv_h, dby = cs_pme_dbasis(uy) * inv_h, dbz = cs_pme_dbasis(uz) * inv_h;
    float bxa[4] = {bx.x, bx.y, bx.z, bx.w};
    float bya[4] = {by.x, by.y, by.z, by.w};
    float dbxa[4] = {dbx.x, dbx.y, dbx.z, dbx.w};
    float dbya[4] = {dby.x, dby.y, dby.z, dby.w};
    float e = 0.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f;
    for (int a = 0; a < 4; a++) {
        int ia = i0x + a; float cx = bxa[a], dx = dbxa[a];
        for (int b = 0; b < 4; b++) {
            int ib = i0y + b; float cy = bya[b], dy = dbya[b];
            float cxy = cx * cy;
            int ixyz = (ia * nyL + ib) * nzL + i0z;
            float4 v = vload4(0, coeffs + ixyz);
            float ez = dot(v, bz);
            float dez = dot(v, dbz);
            e  += cxy * ez;
            gx += dx * cy * ez;
            gy += cx * dy * ez;
            gz += cxy * dez;
        }
    }
    return (float4)(-gx, -gy, -gz, e);
}

// Tiled PME eval: if the query's 4x4x4 stencil fits the cached slab window AND
// the tile atom list, evaluate entirely from local memory; else fall back to the
// global mesh + bucket core eval (exact) and increment *inout_esc.
// Telemetry identical to cs_eval_contact_pme_at (offender translated to global id).
inline float4 cs_eval_contact_pme_tile_at(
    float x, float y, float z,
    __global const float* mesh_coeffs, int nx, int ny, int nz,
    float ox, float oy, float oz, float h,
    __global const float4* atoms, __global const float* atom_coeffs,
    __global const int* bucket_atoms, __global const int* bucket_offsets,
    int nat, int nbx, int nby, int nbuckets,
    float bx0, float by0, float cell,
    __local const float* LMESH, int ix0, int iy0, int nxL, int nyL, int kz0, int nzL,
    __local const float4* LATOMS, __local const float* LCOEFFS, int nloc, float d_span,
    __global const int* wg_ids,
    int* out_status, float* out_min_r, int* out_offender, int* out_overflow,
    int3* inout_esc)
{
    float inv_h = 1.0f / h;
    int i0x = (int)floor((x - ox) * inv_h) - 1;
    int i0y = (int)floor((y - oy) * inv_h) - 1;
    int i0z = (int)floor((z - oz) * inv_h) - 1;
    bool inside = (i0x >= ix0) && (i0x + 3 < ix0 + nxL)
               && (i0y >= iy0) && (i0y + 3 < iy0 + nyL)
               && (i0z >= kz0) && (i0z + 3 < kz0 + nzL);
    int mesh_status = 0, core_status = 0;
    float min_r = 1e30f; int offender = -1; int overflow = 0;
    float4 fm, fc;
    if (inside) {
        fm = cs_pme_tricubic_eval_tile(x, y, z, LMESH, ix0, iy0, kz0, nxL, nyL, nzL, ox, oy, oz, h);
        fc = cs_pme_core_eval_local_at(x, y, z, LATOMS, LCOEFFS, nloc, d_span, &core_status, &min_r, &offender, &overflow);
        if (offender >= 0) offender = wg_ids[offender];   // local index -> global atom id
    } else {
        if (inout_esc) {
            inout_esc->x += (i0x < ix0 || i0x + 3 >= ix0 + nxL) ? 1 : 0;
            inout_esc->y += (i0y < iy0 || i0y + 3 >= iy0 + nyL) ? 1 : 0;
            inout_esc->z += (i0z < kz0 || i0z + 3 >= kz0 + nzL) ? 1 : 0;
        }

        fm = cs_pme_tricubic_eval(x, y, z, mesh_coeffs, nx, ny, nz, ox, oy, oz, h, &mesh_status);
        if (nloc == nat) {
            // local list holds every atom — exact everywhere, reuse it
            fc = cs_pme_core_eval_local_at(x, y, z, LATOMS, LCOEFFS, nloc, d_span, &core_status, &min_r, &offender, &overflow);
            if (offender >= 0) offender = wg_ids[offender];
        } else {
            fc = cs_pme_core_eval_at(x, y, z, atoms, atom_coeffs, bucket_atoms, bucket_offsets,
                                     nat, nbx, nby, nbuckets, bx0, by0, cell, d_span,
                                     &core_status, &min_r, &offender, &overflow);
        }
    }
    int status = mesh_status | core_status;
    *out_status = status; *out_min_r = min_r; *out_offender = offender; *out_overflow = overflow;
    if (status & 1 || status & 2) return (float4)(NAN, NAN, NAN, NAN);
    return (float4)(fm.xyz + fc.xyz, fm.w + fc.w);
}

// Batch evaluation of V_mesh + V_core at query points. Returns (E,F) per query + telemetry.
// Invalid queries (mesh OOB or domain violation) produce non-finite outputs.
__kernel void evalContactPME(
    __global const float* queries, __global float4* out_fe,
    __global int* out_status, __global float* out_min_r, __global int* out_offender, __global int* out_overflow,
    __global const float* mesh_coeffs, const int4 mesh_meta, const float4 mesh_origin_h,
    __global const float4* atoms, __global const float* atom_coeffs,
    __global const int* bucket_atoms, __global const int* bucket_offsets,
    const int4 core_meta, const float4 core_bucket_meta,
    const int nq)
{
    int i = get_global_id(0);
    if (i >= nq) return;
    float x = queries[i * 3 + 0], y = queries[i * 3 + 1], z = queries[i * 3 + 2];
    int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
    float4 fe = cs_eval_contact_pme_at(x, y, z,
        mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
        mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
        atoms, atom_coeffs, bucket_atoms, bucket_offsets,
        core_meta.x, core_meta.y, core_meta.z, core_meta.w,
        core_bucket_meta.x, core_bucket_meta.y, core_bucket_meta.z, core_bucket_meta.w,
        &status, &min_r, &offender, &overflow);
    out_fe[i] = fe; out_status[i] = status; out_min_r[i] = min_r; out_offender[i] = offender; out_overflow[i] = overflow;
}

// Workgroup/local-memory batch evaluator for small/medium molecules.
// All work-items, including padded ones, preload and reach the barrier before returning.
__kernel void evalContactPMELocal(
    __global const float* queries, __global float4* out_fe,
    __global int* out_status, __global float* out_min_r, __global int* out_offender, __global int* out_overflow,
    __global const float* mesh_coeffs, const int4 mesh_meta, const float4 mesh_origin_h,
    __global const float4* atoms, __global const float* atom_coeffs,
    const int4 core_meta, const float4 core_bucket_meta,
    const int nq, __local float4* LATOMS, __local float* LCOEFFS)
{
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int nat = core_meta.x;
    for (int i = lid; i < nat; i += lsize) LATOMS[i] = atoms[i];
    for (int i = lid; i < nat * CS_PME_NMODES; i += lsize) LCOEFFS[i] = atom_coeffs[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    int iq = get_global_id(0);
    if (iq >= nq) return;
    float x = queries[iq * 3 + 0], y = queries[iq * 3 + 1], z = queries[iq * 3 + 2];
    int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
    float4 fe = cs_eval_contact_pme_local_at(x, y, z,
        mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
        mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
        LATOMS, LCOEFFS, nat, core_bucket_meta.w,
        &status, &min_r, &offender, &overflow);
    out_fe[iq] = fe; out_status[iq] = status; out_min_r[iq] = min_r; out_offender[iq] = offender; out_overflow[iq] = overflow;
}

// Separable cubic B-spline prefilter — Thomas solve of tridiag(1,4,1) c = 6 d
// along one axis. Zero-padded boundary; reproduces scipy solve_banded of
// ContactSurface._bspline_tridiag_ab (diag 4/6, off 1/6) used by
// CoarseMesh._prefilter_3d. One work-item per line.
// Line layout: base = (gid/n_inner)*(line_len*n_inner) + (gid%n_inner), stride = n_inner.
//   z-axis: n_inner = 1       y-axis: n_inner = nz       x-axis: n_inner = ny*nz
// invden[i] = 1/(4 - c'[i-1]) and cprime[i] = invden[i] (c_i = 1), host-precomputed
// by CoarseMesh._bspline_thomas_factors.
__kernel void cs_bspline_prefilter_lines(
    __global float* a,
    const int n_lines, const int line_len, const int n_inner,
    __global const float* invden, __global const float* cprime)
{
    const int gid = get_global_id(0);
    if (gid >= n_lines || line_len < 2) return;
    const int base = (gid / n_inner) * (line_len * n_inner) + (gid % n_inner);
    // forward elimination: d'[i] = (6 d[i] - d'[i-1]) * invden[i]  (a_i = 1)
    float yp = 6.0f * a[base] * invden[0];
    a[base] = yp;
    for (int i = 1; i < line_len; i++) {
        const int p = base + i * n_inner;
        const float yi = (6.0f * a[p] - yp) * invden[i];
        a[p] = yi;
        yp = yi;
    }
    // backward substitution: x[i] = d'[i] - cprime[i] * x[i+1]
    float xp = a[base + (line_len - 1) * n_inner];
    for (int i = line_len - 2; i >= 0; i--) {
        const int p = base + i * n_inner;
        const float xi = a[p] - cprime[i] * xp;
        a[p] = xi;
        xp = xi;
    }
}

// Per-atom PAW core fit. One work-item per atom, 32 Chebyshev nodes on [r_lo, r_b].
// Normal equations of the weighted energy+force rows, then a 5×5 Cholesky.
// Matches PICCore.fit_core_paw_grid (float64 oracle). Powers 2,4,8,16,32.
#define CS_CORE_FIT_NSAMP 32

inline void cs_chol_solve_5(float* G, float* rhs, float* c) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j <= i; j++) {
            float s = G[i * 5 + j];
            for (int k = 0; k < j; k++) s -= G[i * 5 + k] * G[j * 5 + k];
            if (i == j) {
                G[i * 5 + i] = (s > 1e-20f) ? sqrt(s) : 0.0f;
            } else {
                G[i * 5 + j] = s / G[j * 5 + j];
            }
        }
    }
    float y[5];
    for (int i = 0; i < 5; i++) {
        float s = rhs[i];
        for (int k = 0; k < i; k++) s -= G[i * 5 + k] * y[k];
        y[i] = (G[i * 5 + i] > 0.0f) ? s / G[i * 5 + i] : 0.0f;
    }
    for (int i = 4; i >= 0; i--) {
        float s = y[i];
        for (int k = i + 1; k < 5; k++) s -= G[k * 5 + i] * c[k];
        c[i] = (G[i * 5 + i] > 0.0f) ? s / G[i * 5 + i] : NAN;
    }
}

__kernel void cs_fit_core_paw(
    __global const float4* reqs,     // (R0, E0, q, r_lo)
    __global const float4* paw,      // (a0, a2, a4, a6)
    __global const float* r_b,
    const float alpha, const float r_damp, const float q_tip,
    const int na, const int n_samp,
    __global float* coeffs)
{
    const int ia = get_global_id(0);
    if (ia >= na || n_samp != CS_CORE_FIT_NSAMP) return;
    const float4 rq = reqs[ia];
    const float4 pw = paw[ia];
    const float R0 = rq.x, E0 = rq.y, q = rq.z, r_lo = rq.w;
    const float rb = r_b[ia];
    const float D = rb - r_lo;
    const float K = -alpha;
    const float R2 = r_damp * r_damp;
    const float cc = 14.3996448915f * q * q_tip;
    float vt[CS_CORE_FIT_NSAMP], vs[CS_CORE_FIT_NSAMP], dvs[CS_CORE_FIT_NSAMP];
    float phi[5], dphi[5];
    for (int k = 0; k < CS_CORE_FIT_NSAMP; k++) {
        float u = 0.5f * (1.0f - cos(M_PI_F * ((float)k + 0.5f) / (float)CS_CORE_FIT_NSAMP));
        float r = r_lo + D * u;
        float e = exp(K * (r - R0));
        float e2 = e * e;
        float v = E0 * (e2 - 2.0f * e);
        float dv = 2.0f * K * E0 * (e2 - e);
        float s2 = r * r + R2;
        float inv_s = 1.0f / sqrt(s2);
        float inv_s3 = inv_s * inv_s * inv_s;
        v += cc * inv_s;
        dv -= cc * r * inv_s3;
        float r2 = r * r, r4 = r2 * r2, r6 = r4 * r2;
        float P = pw.x + pw.y * r2 + pw.z * r4 + pw.w * r6;
        float dP = 2.0f * pw.y * r + 4.0f * pw.z * r2 * r + 6.0f * pw.w * r4 * r;
        vt[k] = v;
        vs[k] = v - P;
        dvs[k] = dv - dP;
    }
    float vmin = vt[0];
    for (int k = 1; k < CS_CORE_FIT_NSAMP; k++) vmin = fmin(vmin, vt[k]);
    float ord[CS_CORE_FIT_NSAMP];
    for (int k = 0; k < CS_CORE_FIT_NSAMP; k++) ord[k] = vt[k];
    for (int i = 1; i < CS_CORE_FIT_NSAMP; i++) {
        float key = ord[i];
        int j = i - 1;
        while (j >= 0 && ord[j] > key) { ord[j + 1] = ord[j]; j--; }
        ord[j + 1] = key;
    }
    float pos = 0.95f * (float)(CS_CORE_FIT_NSAMP - 1);
    int lo = (int)pos;
    int hi = lo + 1 < CS_CORE_FIT_NSAMP ? lo + 1 : CS_CORE_FIT_NSAMP - 1;
    float p95 = (1.0f - (pos - (float)lo)) * ord[lo] + (pos - (float)lo) * ord[hi];
    float T = fmax((p95 - vmin) / 3.0f, 0.05f);
    float meanS = 0.0f, meanF = 0.0f, wmax = 0.0f;
    float w[CS_CORE_FIT_NSAMP];
    for (int k = 0; k < CS_CORE_FIT_NSAMP; k++) {
        w[k] = exp(-(vt[k] - vmin) / T);
        wmax = fmax(wmax, w[k]);
        meanS += vs[k];
        meanF += dvs[k];
    }
    wmax = fmax(wmax, 1e-16f);
    meanS /= (float)CS_CORE_FIT_NSAMP;
    meanF /= (float)CS_CORE_FIT_NSAMP;
    float varS = 0.0f, varF = 0.0f;
    for (int k = 0; k < CS_CORE_FIT_NSAMP; k++) {
        w[k] /= wmax;
        float ds = vs[k] - meanS, df = dvs[k] - meanF;
        varS += ds * ds;
        varF += df * df;
    }
    varS /= (float)CS_CORE_FIT_NSAMP;
    varF /= (float)CS_CORE_FIT_NSAMP;
    float E_scale = fmax(sqrt(varS), 1e-12f);
    float F_scale = fmax(sqrt(varF), 1e-12f);
    float lamE2 = 1.0f / (E_scale * E_scale);
    float lamF2 = 1.0f / (F_scale * F_scale);
    float G[25], rhs[5];
    for (int m = 0; m < 25; m++) G[m] = 0.0f;
    for (int m = 0; m < 5; m++) rhs[m] = 0.0f;
    for (int k = 0; k < CS_CORE_FIT_NSAMP; k++) {
        float u = 0.5f * (1.0f - cos(M_PI_F * ((float)k + 0.5f) / (float)CS_CORE_FIT_NSAMP));
        float r = r_lo + D * u;
        cs_pme_core_basis(r, r_lo, rb, phi, dphi);
        float wE = w[k] * lamE2, wF = w[k] * lamF2;
        for (int m = 0; m < 5; m++) {
            rhs[m] += wE * phi[m] * vs[k] + wF * dphi[m] * dvs[k];
            for (int n = 0; n < 5; n++)
                G[m * 5 + n] += wE * phi[m] * phi[n] + wF * dphi[m] * dphi[n];
        }
    }
    float col[5];
    for (int m = 0; m < 5; m++) col[m] = fmax(sqrt(fmax(G[m * 5 + m], 0.0f)), 1e-12f);
    for (int m = 0; m < 5; m++) {
        rhs[m] /= col[m];
        for (int n = 0; n < 5; n++) G[m * 5 + n] /= (col[m] * col[n]);
    }
    float c[5];
    cs_chol_solve_5(G, rhs, c);
    for (int m = 0; m < 5; m++) coeffs[ia * 5 + m] = c[m] / col[m];
}

// ===================== AFMulator integration (requires AFM.cl before this file) =====================
#ifndef AFM_STANDALONE

__attribute__((reqd_work_group_size(CS_ATOM_TILE, 1, 1)))
__kernel void cs_brute_afm_morse_c_points(
    const int natoms,
    __global const float4* atoms,
    __global const float4* cMs,
    __global const float* queries,
    __global float4* out_fe,
    const int nq,
    float4 Qs,
    float4 QZs
) {
    __local float4 LATOMS[CS_ATOM_TILE];
    __local float4 LCMS[CS_ATOM_TILE];
    const int iq = get_global_id(0);
    const int iL = get_local_id(0);
    const int nL = get_local_size(0);
    const bool active_q = (iq < nq);
    float3 pos = (float3)(0.0f, 0.0f, 0.0f);
    if (active_q) {
        pos = (float3)(queries[iq * 3 + 0], queries[iq * 3 + 1], queries[iq * 3 + 2]);
    }
    float4 fe = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    Qs *= COULOMB_CONST;
    for (int j0 = 0; j0 < natoms; j0 += nL) {
        int j = j0 + iL;
        float4 ap = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 cm = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        if (j < natoms) {
            ap = atoms[j];
            cm = cMs[j];
        }
        LATOMS[iL] = ap;
        LCMS[iL] = cm;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int jl = 0; jl < nL; jl++) {
            int ja = jl + j0;
            if (active_q && ja < natoms) {
                float4 xyzq = LATOMS[jl];
                float3 dp = pos - xyzq.xyz;
                fe += getMorse(dp, LCMS[jl].xyz);
                fe += getCoulombAFM(xyzq, pos + (float3)(0.0f, 0.0f, QZs.x)) * Qs.x;
                fe += getCoulombAFM(xyzq, pos + (float3)(0.0f, 0.0f, QZs.y)) * Qs.y;
                fe += getCoulombAFM(xyzq, pos + (float3)(0.0f, 0.0f, QZs.z)) * Qs.z;
                fe += getCoulombAFM(xyzq, pos + (float3)(0.0f, 0.0f, QZs.w)) * Qs.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active_q) out_fe[iq] = fe;
}

__kernel void getFEinStrokesTiltedContact(
    __global const float* coeffs,
    __global const float* h0,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart,
    __global float4* points,
    __global float4* FEs,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 dTip,
    float4 dpos0,
    int nz
) {
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT(dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos = tipPos.xyz + dpos0_.xyz;
    for (int iz = 0; iz < nz; iz++) {
        float4 fe = cs_eval_separable_fe_at(pos.x, pos.y, pos.z, coeffs, h0, meta, origin_step, dy_rc, invRc_mstart);
        float4 fe_;
        fe_.xyz = rotMat(fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
        fe_.w = fe.w;
        FEs[get_global_id(0) * nz + iz] = fe_;
        tipPos += dTip.xyz;
        pos += dTip.xyz;
    }
}

__kernel void relaxStrokesTiltedContact(
    __global const float* coeffs,
    __global const float* h0,
    const int4 meta,
    const float4 origin_step,
    const float2 dy_rc,
    const float2 invRc_mstart,
    __global float4* points,
    __global float4* FEs,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params,
    float4 surfFF,
    int nz
) {
    const float3 dTip = tipC.xyz * tipC.w;
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT(dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos = tipPos.xyz + dpos0_.xyz;
    float dt = relax_params.x;
    float damp = relax_params.y;
    float dtmax = dt;
    float dtmin = dtmax * 0.1f;
    float damp0 = damp;
    for (int iz = 0; iz < nz; iz++) {
        float4 fe;
        float3 v = (float3)(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < N_RELAX_STEP_MAX; i++) {
            fe = cs_eval_separable_fe_at(pos.x, pos.y, pos.z, coeffs, h0, meta, origin_step, dy_rc, invRc_mstart);
            float3 f = fe.xyz;
            float3 dpos = pos - tipPos;
            float3 dpos_ = rotMat(dpos, tipA.xyz, tipB.xyz, tipC.xyz);
            float3 ftip = tipForce(dpos_, stiffness, dpos0);
            f += rotMatT(ftip, tipA.xyz, tipB.xyz, tipC.xyz);
            f += tipC.xyz * surfFF.x;
            #if OPT_FIRE
            v = update_FIRE(f, v, &dt, &damp, dtmin, dtmax, damp0);
            #else
            v *= (1.0f - damp);
            #endif
            v += f * dt;
            pos.xyz += v * dt;
            if (dot(f, f) < F2CONV) break;
        }
        fe = cs_eval_separable_fe_at(pos.x, pos.y, pos.z, coeffs, h0, meta, origin_step, dy_rc, invRc_mstart);
        float4 fe_;
        fe_.xyz = rotMat(fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
        fe_.w = fe.w;
        FEs[get_global_id(0) * nz + iz] = fe_;
        tipPos += dTip.xyz;
        pos += dTip.xyz;
    }
}

__kernel void getFEinStrokesTiltedPIC(
    __global const float4* pic_atoms,
    __global const float* pic_coeffs,
    __global const int* bucket_atoms,
    __global const int* bucket_offsets,
    const int4 pic_meta,
    const float4 pic_bucket_meta,
    const float m_start,
    __global float4* points,
    __global float4* FEs,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 dTip,
    float4 dpos0,
    int nz
) {
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT(dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos = tipPos.xyz + dpos0_.xyz;
    for (int iz = 0; iz < nz; iz++) {
        float4 fe = cs_eval_pic_fe_at(pos.x, pos.y, pos.z, pic_atoms, pic_coeffs, bucket_atoms, bucket_offsets, pic_meta, pic_bucket_meta, m_start);
        float4 fe_;
        fe_.xyz = rotMat(fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
        fe_.w = fe.w;
        FEs[get_global_id(0) * nz + iz] = fe_;
        tipPos += dTip.xyz;
        pos += dTip.xyz;
    }
}

__kernel void relaxStrokesTiltedPIC(
    __global const float4* pic_atoms,
    __global const float* pic_coeffs,
    __global const int* bucket_atoms,
    __global const int* bucket_offsets,
    const int4 pic_meta,
    const float4 pic_bucket_meta,
    const float m_start,
    __global float4* points,
    __global float4* FEs,
    float4 tipA,
    float4 tipB,
    float4 tipC,
    float4 stiffness,
    float4 dpos0,
    float4 relax_params,
    float4 surfFF,
    int nz
) {
    const float3 dTip = tipC.xyz * tipC.w;
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT(dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos = tipPos.xyz + dpos0_.xyz;
    float dt = relax_params.x;
    float damp = relax_params.y;
    float dtmax = dt;
    float dtmin = dtmax * 0.1f;
    float damp0 = damp;
    for (int iz = 0; iz < nz; iz++) {
        float4 fe;
        float3 v = (float3)(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < N_RELAX_STEP_MAX; i++) {
            fe = cs_eval_pic_fe_at(pos.x, pos.y, pos.z, pic_atoms, pic_coeffs, bucket_atoms, bucket_offsets, pic_meta, pic_bucket_meta, m_start);
            float3 f = fe.xyz;
            float3 dpos = pos - tipPos;
            float3 dpos_ = rotMat(dpos, tipA.xyz, tipB.xyz, tipC.xyz);
            float3 ftip = tipForce(dpos_, stiffness, dpos0);
            f += rotMatT(ftip, tipA.xyz, tipB.xyz, tipC.xyz);
            f += tipC.xyz * surfFF.x;
            #if OPT_FIRE
            v = update_FIRE(f, v, &dt, &damp, dtmin, dtmax, damp0);
            #else
            v *= (1.0f - damp);
            #endif
            v += f * dt;
            pos.xyz += v * dt;
            if (dot(f, f) < F2CONV) break;
        }
        fe = cs_eval_pic_fe_at(pos.x, pos.y, pos.z, pic_atoms, pic_coeffs, bucket_atoms, bucket_offsets, pic_meta, pic_bucket_meta, m_start);
        float4 fe_;
        fe_.xyz = rotMat(fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
        fe_.w = fe.w;
        FEs[get_global_id(0) * nz + iz] = fe_;
        tipPos += dTip.xyz;
        pos += dTip.xyz;
    }
}

// PP relaxation using the contact_pme inline evaluator (mesh + core). Same pattern as
// relaxStrokesTiltedContact but with cs_eval_contact_pme_at instead of the separable surface.
// Telemetry (status/min_r/offender/overflow) accumulated per (pixel,z) across all relax steps.
__kernel void relaxStrokesTiltedContactPME(
    __global const float* mesh_coeffs, const int4 mesh_meta, const float4 mesh_origin_h,
    __global const float4* atoms, __global const float* atom_coeffs,
    __global const int* bucket_atoms, __global const int* bucket_offsets,
    const int4 core_meta, const float4 core_bucket_meta,
    __global int* out_status, __global float* out_min_r, __global int* out_offender, __global int* out_overflow,
    __global float4* points, __global float4* FEs,
    float4 tipA, float4 tipB, float4 tipC,
    float4 stiffness, float4 dpos0, float4 relax_params, float4 surfFF,
    int nz)
{
    const float3 dTip = tipC.xyz * tipC.w;
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT(dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
    float3 tipPos = points[get_global_id(0)].xyz;
    float3 pos = tipPos.xyz + dpos0_.xyz;
    float dt = relax_params.x;
    float damp = relax_params.y;
    float dtmax = dt;
    float dtmin = dtmax * 0.1f;
    float damp0 = damp;
    for (int iz = 0; iz < nz; iz++) {
        float4 fe;
        float3 v = (float3)(0.0f, 0.0f, 0.0f);
        int status_acc = 0; float min_r_acc = 1e30f; int offender_acc = -1; int overflow_acc = 0;
        for (int i = 0; i < N_RELAX_STEP_MAX; i++) {
            int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
            fe = cs_eval_contact_pme_at(pos.x, pos.y, pos.z,
                mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                atoms, atom_coeffs, bucket_atoms, bucket_offsets,
                core_meta.x, core_meta.y, core_meta.z, core_meta.w,
                core_bucket_meta.x, core_bucket_meta.y, core_bucket_meta.z, core_bucket_meta.w,
                &status, &min_r, &offender, &overflow);
            status_acc |= status;
            if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
            overflow_acc += overflow;
            if (status & 1 || status & 2) break;  // invalid → cannot relax with NaN force
            float3 f = fe.xyz;
            float3 dpos = pos - tipPos;
            float3 dpos_ = rotMat(dpos, tipA.xyz, tipB.xyz, tipC.xyz);
            float3 ftip = tipForce(dpos_, stiffness, dpos0);
            f += rotMatT(ftip, tipA.xyz, tipB.xyz, tipC.xyz);
            f += tipC.xyz * surfFF.x;
            #if OPT_FIRE
            v = update_FIRE(f, v, &dt, &damp, dtmin, dtmax, damp0);
            #else
            v *= (1.0f - damp);
            #endif
            v += f * dt;
            pos.xyz += v * dt;
            if (dot(f, f) < F2CONV) break;
        }
        int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
        fe = cs_eval_contact_pme_at(pos.x, pos.y, pos.z,
            mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
            mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
            atoms, atom_coeffs, bucket_atoms, bucket_offsets,
            core_meta.x, core_meta.y, core_meta.z, core_meta.w,
            core_bucket_meta.x, core_bucket_meta.y, core_bucket_meta.z, core_bucket_meta.w,
            &status, &min_r, &offender, &overflow);
        status_acc |= status;
        if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
        overflow_acc += overflow;
        float4 fe_;
        fe_.xyz = rotMat(fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
        fe_.w = fe.w;
        int idx = get_global_id(0) * nz + iz;
        FEs[idx] = fe_;
        out_status[idx] = status_acc; out_min_r[idx] = min_r_acc; out_offender[idx] = offender_acc; out_overflow[idx] = overflow_acc;
        tipPos += dTip.xyz;
        pos += dTip.xyz;
    }
}

// Workgroup/local-memory contact-PME relaxation for small/medium molecules.
// Host contract: global size padded to local size; n_scan is the unpadded lane count;
// local allocations are nat*sizeof(float4) and nat*CS_PME_NMODES*sizeof(float).
// No barrier occurs inside FIRE: divergent convergence cannot deadlock the workgroup.
__kernel void relaxStrokesTiltedContactPMELocal(
    __global const float* mesh_coeffs, const int4 mesh_meta, const float4 mesh_origin_h,
    __global const float4* atoms, __global const float* atom_coeffs,
    const int4 core_meta, const float4 core_bucket_meta,
    __global int* out_status, __global float* out_min_r, __global int* out_offender, __global int* out_overflow,
    __global float4* points, __global float4* FEs, __global float4* out_pp,
    float4 tipA, float4 tipB, float4 tipC,
    float4 stiffness, float4 dpos0, float4 relax_params, float4 surfFF,
    int n_scan, int nz, __local float4* LATOMS, __local float* LCOEFFS)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int nat = core_meta.x;
    for (int i = lid; i < nat; i += lsize) LATOMS[i] = atoms[i];
    for (int i = lid; i < nat * CS_PME_NMODES; i += lsize) LCOEFFS[i] = atom_coeffs[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (gid >= n_scan) return;

    const float3 dTip = tipC.xyz * tipC.w;
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT(dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
    float3 tipPos = points[gid].xyz;
    float3 pos = tipPos.xyz + dpos0_.xyz;
    float dt = relax_params.x;
    float damp = relax_params.y;
    float dtmax = dt;
    float dtmin = dtmax * 0.1f;
    float damp0 = damp;
    for (int iz = 0; iz < nz; iz++) {
        float4 fe;
        float3 v = (float3)(0.0f, 0.0f, 0.0f);
        int status_acc = 0; float min_r_acc = 1e30f; int offender_acc = -1; int overflow_acc = 0;
        for (int i = 0; i < N_RELAX_STEP_MAX; i++) {
            int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
            fe = cs_eval_contact_pme_local_at(pos.x, pos.y, pos.z,
                mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                LATOMS, LCOEFFS, nat, core_bucket_meta.w,
                &status, &min_r, &offender, &overflow);
            status_acc |= status;
            if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
            overflow_acc += overflow;
            if (status & 1 || status & 2) break;
            float3 f = fe.xyz;
            float3 dpos = pos - tipPos;
            float3 dpos_ = rotMat(dpos, tipA.xyz, tipB.xyz, tipC.xyz);
            float3 ftip = tipForce(dpos_, stiffness, dpos0);
            f += rotMatT(ftip, tipA.xyz, tipB.xyz, tipC.xyz);
            f += tipC.xyz * surfFF.x;
            #if OPT_FIRE
            v = update_FIRE(f, v, &dt, &damp, dtmin, dtmax, damp0);
            #else
            v *= (1.0f - damp);
            #endif
            v += f * dt;
            pos.xyz += v * dt;
            if (dot(f, f) < F2CONV) break;
        }
        int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
        fe = cs_eval_contact_pme_local_at(pos.x, pos.y, pos.z,
            mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
            mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
            LATOMS, LCOEFFS, nat, core_bucket_meta.w,
            &status, &min_r, &offender, &overflow);
        status_acc |= status;
        if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
        overflow_acc += overflow;
        float4 fe_;
        fe_.xyz = rotMat(fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
        fe_.w = fe.w;
        int idx = gid * nz + iz;
        FEs[idx] = fe_;
        out_pp[idx] = (float4)(pos.x, pos.y, pos.z, 0.0f);
        out_status[idx] = status_acc; out_min_r[idx] = min_r_acc; out_offender[idx] = offender_acc; out_overflow[idx] = overflow_acc;
        tipPos += dTip.xyz;
        pos += dTip.xyz;
    }
}

// One work-item per atom. Reads the relaxed PP at the pixel above that atom and
// returns the Morse force of that atom alone. Fr > 0 is repulsive (force along
// +dp, away from the nucleus): the Pauli wall, r < R0.
__kernel void pauliForceAtRelaxedPP(
    __global const float4* atoms, __global const float4* rea, __global const float4* pp,
    int nx, int ny, int nz, int iz, float x0, float y0, float dx, float dy,
    __global float4* out)
{
    int ia = get_global_id(0);
    float4 a = atoms[ia];
    float4 m = rea[ia];
    int ix = (int)floor((a.x - x0) / dx + 0.5f);
    int iy = (int)floor((a.y - y0) / dy + 0.5f);
    ix = clamp(ix, 0, nx - 1);
    iy = clamp(iy, 0, ny - 1);
    float4 p = pp[(ix * ny + iy) * nz + iz];
    float3 dp = p.xyz - a.xyz;
    float4 fe = getMorse(dp, m.xyz);
    float r = sqrt(dot(dp, dp) + R2SAFE);
    float Fr = dot(fe.xyz, dp) / r;
    out[ia] = (float4)(r, m.x, Fr, fe.z);
}

// ===================== contact_pme FIT — mesh V_L raster (PAW), WG+local atoms =====================
// One WI per mesh node. Cooperative preload of atoms / REQH / PAW coeffs (Codex-style).
// paw[i] = (a0, a2, a4, a6); paw_rb[i].x = r_b. Outside r_b: Morse+PLQH energy via getMorsePLQH.

__kernel void fillContactPMEMeshVL(
    __global float* samples,
    const int4 mesh_n,
    const float4 origin_h,
    __global const float4* atoms,
    __global const float4* reqs,
    __global const float4* paw,
    __global const float4* paw_rb,
    const int na,
    const float4 GFFParams,
    const float4 PLQH,
    __local float4* LATOMS,
    __local float4* LREQS,
    __local float4* LPAW,
    __local float4* LRB
) {
    const int nx = mesh_n.x, ny = mesh_n.y, nz = mesh_n.z;
    const int ntot = nx * ny * nz;
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const bool active = (gid < ntot);
    float3 pos = (float3)(0.0f, 0.0f, 0.0f);
    if (active) {
        int ix = gid / (ny * nz);
        int rem = gid - ix * ny * nz;
        int iy = rem / nz;
        int iz = rem - iy * nz;
        pos = (float3)(origin_h.x + ix * origin_h.w,
                       origin_h.y + iy * origin_h.w,
                       origin_h.z + iz * origin_h.w);
    }
    const float K = -GFFParams.y;
    const float R2damp = GFFParams.x * GFFParams.x;
    float vL = 0.0f;
    for (int j0 = 0; j0 < na; j0 += lsz) {
        int j = j0 + lid;
        float4 ap = (float4)(0.0f); float4 rq = (float4)(0.0f);
        float4 pw = (float4)(0.0f); float4 rb = (float4)(0.0f);
        if (j < na) { ap = atoms[j]; rq = reqs[j]; pw = paw[j]; rb = paw_rb[j]; }
        LATOMS[lid] = ap; LREQS[lid] = rq; LPAW[lid] = pw; LRB[lid] = rb;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (active) {
            int ntile = min(lsz, na - j0);
            for (int jl = 0; jl < ntile; jl++) {
                float3 dp = pos - LATOMS[jl].xyz;
                float r2 = dot(dp, dp);
                float r = sqrt(r2);
                float r_b = LRB[jl].x;
                if (r < r_b) {
                    float r4 = r2 * r2;
                    float4 c = LPAW[jl];
                    vL += c.x + c.y * r2 + c.z * r4 + c.w * (r4 * r2);
                } else {
                    float4 fej = getMorsePLQH(dp, LREQS[jl], PLQH, K, R2damp);
                    vL += fej.w;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (active) samples[gid] = vL;
}

// ===================== contact_pme SCAN — quasi-Newton relaxation =====================
#ifndef QN_BUDGET
#define QN_BUDGET 8
#endif
#ifndef NR_BUDGET
#define NR_BUDGET 16
#endif

// Total residual force on PP at pos: f_sample + tip-spring + surf bias (Local backend).
// Same telemetry pointers as cs_eval_contact_pme_local_at.
inline float3 cs_pme_total_force_local(
    float3 pos, float3 tipPos,
    float3 tipA, float3 tipB, float3 tipC,
    __global const float* mesh_coeffs, const int4 mesh_meta, const float4 mesh_origin_h,
    __local const float4* LATOMS, __local const float* LCOEFFS, int nat, float d_span,
    float4 stiffness, float4 dpos0, float4 surfFF,
    int* out_status, float* out_min_r, int* out_offender, int* out_overflow,
    float4* out_fe)
{
    float4 fe = cs_eval_contact_pme_local_at(pos.x, pos.y, pos.z,
        mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
        mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
        LATOMS, LCOEFFS, nat, d_span, out_status, out_min_r, out_offender, out_overflow);
    *out_fe = fe;
    float3 f = fe.xyz;
    float3 dpos_ = rotMat(pos - tipPos, tipA, tipB, tipC);
    f += rotMatT(tipForce(dpos_, stiffness, dpos0), tipA, tipB, tipC);
    f += tipC * surfFF.x;
    return f;
}

// Solve general 3x3 A x = b by Cramer/cofactors. Returns 0 ok, 4 singular.
inline int solve3x3_general(float3 a0, float3 a1, float3 a2, float3 b, float3* x){
    // a0,a1,a2 are ROWS of A
    float det = a0.x*(a1.y*a2.z - a1.z*a2.y) - a0.y*(a1.x*a2.z - a1.z*a2.x) + a0.z*(a1.x*a2.y - a1.y*a2.x);
    float scale = fmax(fabs(a0.x), fmax(fabs(a1.y), fmax(fabs(a2.z), fmax(fabs(a0.y), fmax(fabs(a1.x), fabs(a0.z))))));
    if (fabs(det) < 1e-12f * fmax(scale*scale*scale, 1e-12f)) return 4;
    float inv = 1.0f / det;
    x->x = (b.x*(a1.y*a2.z - a1.z*a2.y) - a0.y*(b.y*a2.z - a1.z*b.z) + a0.z*(b.y*a2.y - a1.y*b.z)) * inv;
    x->y = (a0.x*(b.y*a2.z - a1.z*b.z) - b.x*(a1.x*a2.z - a1.z*a2.x) + a0.z*(a1.x*b.z - b.y*a2.x)) * inv;
    x->z = (a0.x*(a1.y*b.z - b.y*a2.y) - a0.y*(a1.x*b.z - b.y*a2.x) + b.x*(a1.x*a2.y - a1.y*a2.x)) * inv;
    return 0;
}
// relaxStrokesTiltedContactPMELocalQN — same scan loop/telemetry as
// relaxStrokesTiltedContactPMELocal, but the inner loop is a diagonal-secant
// quasi-Newton solver instead of FIRE/damped-MD:
//   G(x) = f_s(x) + f_tip(x - tipPos) + f_surf = 0  (3x1 residual)
//   dx = -G / Jd , Jd ~ diag(dG/dx) refined by secant df/dx per axis.
//   Init Jd = stiffness.xyz (Hook's law: dx = F / k_tip in linear response).
// Safeguards: trust cap |dx|<=0.1A, reject-and-damp on force growth, clamp
// Jd negative (stable directions only). QN budget = QN_BUDGET iters; harder
// slices then finish under plain FIRE (nonlinear contact regime).
__kernel void relaxStrokesTiltedContactPMELocalQN(
    __global const float* mesh_coeffs, const int4 mesh_meta, const float4 mesh_origin_h,
    __global const float4* atoms, __global const float* atom_coeffs,
    const int4 core_meta, const float4 core_bucket_meta,
    __global int* out_status, __global float* out_min_r, __global int* out_offender, __global int* out_overflow,
    __global int* out_iters, __global float4* out_pp,
    __global float4* points, __global float4* FEs,
    float4 tipA, float4 tipB, float4 tipC,
    float4 stiffness, float4 dpos0, float4 relax_params, float4 surfFF,
    int n_scan, int nz,
    __global const int* iz_start_buf, __global const int* pix_map,
    __local float4* LATOMS, __local float* LCOEFFS)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int nat = core_meta.x;
    for (int i = lid; i < nat; i += lsize) LATOMS[i] = atoms[i];
    for (int i = lid; i < nat * CS_PME_NMODES; i += lsize) LCOEFFS[i] = atom_coeffs[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (gid >= n_scan) return;

    const float3 dTip = tipC.xyz * tipC.w;
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT(dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
    // two-pass resume: iz0 = first flagged slice of this pixel (0 = fresh column).
    // out_pp[ipx*nz+iz] stores the START pos of slice iz+1 (written after pos+=dTip),
    // so pass 2 resumes from out_pp[iz0-1] — warm start preserved.
    const int ipx = (pix_map) ? pix_map[gid] : gid;      // original pixel slot for out_pp
    const int iz0 = (iz_start_buf) ? iz_start_buf[gid] : 0;
    float3 tipPos = points[gid].xyz + dTip * (float)iz0;
    float3 pos = (iz0 > 0) ? out_pp[ipx * nz + iz0 - 1].xyz : (tipPos.xyz + dpos0_.xyz);
    const float dt_md = relax_params.x;      // FIRE dt (fallback phase)
    const float damp0 = relax_params.y;      // FIRE damp
    const int imax = (relax_params.z > 0.5f) ? (int)relax_params.z : N_RELAX_STEP_MAX;  // two-pass cap
    const float f2conv = F2CONV * fmax(relax_params.w, 1e-12f);   // qn_conv: tol scale (spline-smooth field)
    const float3 a3 = tipA.xyz, b3 = tipB.xyz, c3 = tipC.xyz;
    float3 Jd = stiffness.xyz;               // Hooke init, then CARRIED across slices (warm Jacobian)

    for (int iz = iz0; iz < nz; iz++) {
        float4 fe;
        float3 v = (float3)(0.0f, 0.0f, 0.0f);
        int status_acc = 0; float min_r_acc = 1e30f; int offender_acc = -1; int overflow_acc = 0;
        float3 fp = (float3)(0.0f, 0.0f, 0.0f);
        float3 xp = pos;
        float f2p = 1e30f;
        float trust = 1.0f;
        float dt = dt_md, damp = damp0;
        const float dtmin = dt_md * 0.1f, dtmax = dt_md;
        int niter = 0, conv = 0;
        float f2l = 1e30f;                     // last residual — flag only genuinely stuck slices
        for (int i = 0; i < imax; i++) {
            niter = i + 1;
            int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
            float3 f = cs_pme_total_force_local(pos, tipPos, a3, b3, c3,
                mesh_coeffs, mesh_meta, mesh_origin_h, LATOMS, LCOEFFS, nat, core_bucket_meta.w,
                stiffness, dpos0, surfFF, &status, &min_r, &offender, &overflow, &fe);
            status_acc |= status;
            if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
            overflow_acc += overflow;
            if (status & 1 || status & 2) { conv = -1; break; }   // invalid — more iters cannot help
            float f2 = dot(f, f);
            f2l = f2;
            if (f2 < f2conv) { conv = 1; break; }

            if (i < QN_BUDGET) {
                // --- phase 1: diagonal-secant quasi-Newton (cheap, 1 eval/iter) ---
                if (i > 0) {
                    float3 dd = pos - xp;
                    float3 df = f - fp;
                    if (fabs(dd.x) > 1e-4f) Jd.x = df.x / dd.x;
                    if (fabs(dd.y) > 1e-4f) Jd.y = df.y / dd.y;
                    if (fabs(dd.z) > 1e-4f) Jd.z = df.z / dd.z;
                    Jd = clamp(Jd, -1e6f, -0.1f);    // keep negative (stable)
                    if (f2 > f2p * 3.0f) {           // overshot -> reject, shrink trust
                        pos = xp;
                        trust = fmax(trust * 0.3f, 0.05f);
                    }
                }
                float3 dx = -f / Jd;                    // Jd<0 -> along +f
                dx *= 0.7f * trust;
                dx = clamp(dx, -0.1f, 0.1f);
                xp = pos; fp = f; f2p = f2;
                pos += dx;
                trust = fmin(trust * 1.4f, 1.0f);
            } else if (i < QN_BUDGET + NR_BUDGET) {
                // --- phase 2: full Newton, FD Jacobian of total G (4 evals/iter) ---
                const float h = 1e-3f;
                int s1 = 0, s2_ = 0, s3 = 0; float mr; int of_, ov; float4 fet;
                float3 gx = cs_pme_total_force_local(pos + (float3)(h, 0.f, 0.f), tipPos, a3, b3, c3, mesh_coeffs, mesh_meta, mesh_origin_h, LATOMS, LCOEFFS, nat, core_bucket_meta.w, stiffness, dpos0, surfFF, &s1, &mr, &of_, &ov, &fet);
                float3 gy = cs_pme_total_force_local(pos + (float3)(0.f, h, 0.f), tipPos, a3, b3, c3, mesh_coeffs, mesh_meta, mesh_origin_h, LATOMS, LCOEFFS, nat, core_bucket_meta.w, stiffness, dpos0, surfFF, &s2_, &mr, &of_, &ov, &fet);
                float3 gz = cs_pme_total_force_local(pos + (float3)(0.f, 0.f, h), tipPos, a3, b3, c3, mesh_coeffs, mesh_meta, mesh_origin_h, LATOMS, LCOEFFS, nat, core_bucket_meta.w, stiffness, dpos0, surfFF, &s3, &mr, &of_, &ov, &fet);
                status_acc |= s1 | s2_ | s3;
                if ((s1 | s2_ | s3) & 3) break;         // perturbed point invalid -> stay
                float3 j0 = (gx - f) / h;              // columns dG/dx_j
                float3 j1 = (gy - f) / h;
                float3 j2 = (gz - f) / h;
                // rows of J = cols transposed; symmetrize for stability
                float3 r0 = (float3)(j0.x, (j1.x + j0.y) * 0.5f, (j2.x + j0.z) * 0.5f);
                float3 r1 = (float3)(r0.y, j1.y, (j2.y + j1.z) * 0.5f);
                float3 r2 = (float3)(r0.z, r1.z, j2.z);
                float3 dx;
                if (solve3x3_general(r0, r1, r2, -f, &dx) != 0 || !all(isfinite(dx)))
                    dx = -f / Jd;                       // singular -> Hooke/secant fallback
                dx *= 0.8f;
                dx = clamp(dx, -0.15f, 0.15f);
                pos += dx;
            } else {
                // --- phase 3: plain FIRE (last resort, same as reference kernel) ---
                #if OPT_FIRE
                v = update_FIRE(f, v, &dt, &damp, dtmin, dtmax, damp0);
                #else
                v *= (1.0f - damp);
                #endif
                v += f * dt;
                pos += v * dt;
            }
        }
        if (!conv) {
            // cap exit -> re-eval at moved pos (breaks already have fe at pos)
            int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
            fe = cs_eval_contact_pme_local_at(pos.x, pos.y, pos.z,
                mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                LATOMS, LCOEFFS, nat, core_bucket_meta.w,
                &status, &min_r, &offender, &overflow);
            status_acc |= status;
            if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
            overflow_acc += overflow;
        }
        float4 fe_;
        fe_.xyz = rotMat(fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
        fe_.w = fe.w;
        int idx = gid * nz + iz;
        FEs[idx] = fe_;
        out_status[idx] = status_acc; out_min_r[idx] = min_r_acc; out_offender[idx] = offender_acc; out_overflow[idx] = overflow_acc;
        // negative iters = hit cap AND residual >> tol (spline-noise floor) -> pass-2 candidate.
        out_iters[idx] = (conv != 0 || f2l < f2conv * 1e4f) ? niter : -niter;
        tipPos += dTip.xyz;
        pos += dTip.xyz;
        if (iz_start_buf) out_pp[ipx * nz + iz] = (float4)(pos, 0.0f);   // warm-start pos for slice iz+1 (two-pass only)
    }
}

// ===================== contact_pme SCAN — sphere-constrained relaxation =====================
// relaxStrokesTiltedContactPMELocalSph — same scan/telemetry as the LocalQN kernel,
// but the PP is CONSTRAINED to the sphere |dpos|=L=dpos0.w analytically:
// the stiff radial DOF (K_RAD >> K_LAT) is eliminated; we solve only the two
// soft lateral DOFs in tip coords  dpos_tip = (x, y, -sqrt(L^2-x^2-y^2)).
// Residual: G_i = f_total . e_i  with tangent dirs e1=a+x/|z|c, e2=b+y/|z|c.
// Physics note: this is the K_RAD -> inf limit — the PP cannot compress
// radially (~0.1-0.3 A in hard contact), so expect small systematic diffs
// vs FIRE in deepest contact. For training-data generation likely fine.
// 3 phases: diag-secant QN (soft DOFs) -> full 2x2 FD-Newton -> damped-MD.
__kernel void relaxStrokesTiltedContactPMELocalSph(
    __global const float* mesh_coeffs, const int4 mesh_meta, const float4 mesh_origin_h,
    __global const float4* atoms, __global const float* atom_coeffs,
    const int4 core_meta, const float4 core_bucket_meta,
    __global int* out_status, __global float* out_min_r, __global int* out_offender, __global int* out_overflow,
    __global int* out_iters, __global float4* out_pp,
    __global float4* points, __global float4* FEs,
    float4 tipA, float4 tipB, float4 tipC,
    float4 stiffness, float4 dpos0, float4 relax_params, float4 surfFF,
    int n_scan, int nz,
    __global const int* iz_start_buf, __global const int* pix_map,
    __local float4* LATOMS, __local float* LCOEFFS)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int nat = core_meta.x;
    for (int i = lid; i < nat; i += lsize) LATOMS[i] = atoms[i];
    for (int i = lid; i < nat * CS_PME_NMODES; i += lsize) LCOEFFS[i] = atom_coeffs[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (gid >= n_scan) return;

    const float3 dTip = tipC.xyz * tipC.w;
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT(dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
    const int ipx = (pix_map) ? pix_map[gid] : gid;
    const int iz0 = (iz_start_buf) ? iz_start_buf[gid] : 0;
    float3 tipPos = points[gid].xyz + dTip * (float)iz0;
    float3 pos;
    const float L = dpos0.w;                                // sphere radius [A]
    const float L2 = L * L;
    const float3 a3 = tipA.xyz, b3 = tipB.xyz, c3 = tipC.xyz;
    float2 dxy;
    if (iz0 > 0) {                                          // resume from stored pos
        pos = out_pp[ipx * nz + iz0 - 1].xyz;
        float3 dv = rotMat(pos - tipPos, a3, b3, c3);       // world dpos -> tip coords
        dxy = dv.xy;
    } else {
        dxy = dpos0_.xy;
        pos = tipPos + rotMatT((float3)(dxy.x, dxy.y, -sqrt(fmax(L2 - dxy.x*dxy.x - dxy.y*dxy.y, 1e-4f))), a3, b3, c3);
    }
    const float dt_md = relax_params.x;
    const float damp0 = relax_params.y;
    const int imax = (relax_params.z > 0.5f) ? (int)relax_params.z : N_RELAX_STEP_MAX;
    const float f2conv = F2CONV * fmax(relax_params.w, 1e-12f);
    float2 Jd = stiffness.xy;                               // Hooke init: lateral tip stiffness

    for (int iz = iz0; iz < nz; iz++) {
        float4 fe;
        float2 v = (float2)(0.0f, 0.0f);
        int status_acc = 0; float min_r_acc = 1e30f; int offender_acc = -1; int overflow_acc = 0;
        float2 fp = (float2)(0.0f); float2 xp = dxy;
        float f2p = 1e30f;
        float trust = 1.0f;
        float dt = dt_md, damp = damp0;
        const float dtmin = dt_md * 0.1f, dtmax = dt_md;
        int niter = 0, conv = 0;
        float f2l = 1e30f;
        int ball = 0;
        for (int i = 0; i < imax; i++) {
            niter = i + 1;
            // place PP on sphere (clamp inside 0.9L if overbent)
            float rr = dxy.x*dxy.x + dxy.y*dxy.y;
            float Lz2 = L2 - rr;
            if (Lz2 < 1e-4f) {                              // >~90 deg bend — off-model, cap
                float sc = sqrt(fmax(L2 * 0.81f, 1e-8f) / fmax(rr, 1e-8f));
                dxy *= sc; Lz2 = L2 - dxy.x*dxy.x - dxy.y*dxy.y;
            }
            float zz = -sqrt(Lz2);
            float3 d_ = (float3)(dxy.x, dxy.y, zz);
            pos = tipPos + rotMatT(d_, a3, b3, c3);
            int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
            fe = cs_eval_contact_pme_local_at(pos.x, pos.y, pos.z,
                mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                LATOMS, LCOEFFS, nat, core_bucket_meta.w,
                &status, &min_r, &offender, &overflow);
            status_acc |= status;
            if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
            overflow_acc += overflow;
            if (status & 1 || status & 2) { conv = -1; break; }
            float3 f = fe.xyz + rotMatT(tipForce(d_, stiffness, dpos0), a3, b3, c3) + c3 * surfFF.x;
            // tangent residual: G = f . e_i, e1=a+x/|z|c, e2=b+y/|z|c
            float iz_ = 1.0f / (-zz);
            float fc_ = dot(f, c3);
            float2 G = (float2)(dot(f, a3) + dxy.x * iz_ * fc_,
                                dot(f, b3) + dxy.y * iz_ * fc_);
            float f2 = dot(G, G);
            f2l = f2;
            if (f2 < f2conv) { conv = 1; break; }

            if (i < QN_BUDGET) {
                // --- phase 1: diagonal secant on 2 soft DOFs ---
                if (i > 0) {
                    float2 dd = dxy - xp;
                    float2 df = G - fp;
                    if (fabs(dd.x) > 1e-4f) Jd.x = df.x / dd.x;
                    if (fabs(dd.y) > 1e-4f) Jd.y = df.y / dd.y;
                    Jd = clamp(Jd, -1e6f, -0.1f);
                    if (f2 > f2p * 3.0f) { dxy = xp; trust = fmax(trust * 0.3f, 0.05f); }
                }
                float2 dx = -G / Jd;
                dx *= 0.7f * trust;
                dx = clamp(dx, -0.1f, 0.1f);
                xp = dxy; fp = G; f2p = f2;
                dxy += dx;
                trust = fmin(trust * 1.4f, 1.0f);
            } else if (i < QN_BUDGET + NR_BUDGET) {
                // --- phase 2: full 2x2 FD-Newton on (x,y) ---
                const float h = 1e-3f;
                float2 Gx_, Gy_; int s1 = 0, s2_ = 0; float mr; int of_, ov;
                // probe +x
                {
                    float2 dp = dxy + (float2)(h, 0.f);
                    float z2 = L2 - dp.x*dp.x - dp.y*dp.y; if (z2 < 1e-4f) z2 = 1e-4f;
                    float z3 = -sqrt(z2);
                    float3 dd_ = (float3)(dp.x, dp.y, z3);
                    float3 pp = tipPos + rotMatT(dd_, a3, b3, c3);
                    float4 fej = cs_eval_contact_pme_local_at(pp.x, pp.y, pp.z,
                        mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                        mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                        LATOMS, LCOEFFS, nat, core_bucket_meta.w, &s1, &mr, &of_, &ov);
                    float3 fj = fej.xyz + rotMatT(tipForce(dd_, stiffness, dpos0), a3, b3, c3) + c3 * surfFF.x;
                    float izj = 1.0f / (-z3); float fcj = dot(fj, c3);
                    Gx_ = (float2)(dot(fj, a3) + dp.x * izj * fcj, dot(fj, b3) + dp.y * izj * fcj);
                }
                {
                    float2 dp = dxy + (float2)(0.f, h);
                    float z2 = L2 - dp.x*dp.x - dp.y*dp.y; if (z2 < 1e-4f) z2 = 1e-4f;
                    float z3 = -sqrt(z2);
                    float3 dd_ = (float3)(dp.x, dp.y, z3);
                    float3 pp = tipPos + rotMatT(dd_, a3, b3, c3);
                    float4 fej = cs_eval_contact_pme_local_at(pp.x, pp.y, pp.z,
                        mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                        mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                        LATOMS, LCOEFFS, nat, core_bucket_meta.w, &s2_, &mr, &of_, &ov);
                    float3 fj = fej.xyz + rotMatT(tipForce(dd_, stiffness, dpos0), a3, b3, c3) + c3 * surfFF.x;
                    float izj = 1.0f / (-z3); float fcj = dot(fj, c3);
                    Gy_ = (float2)(dot(fj, a3) + dp.x * izj * fcj, dot(fj, b3) + dp.y * izj * fcj);
                }
                status_acc |= s1 | s2_;
                if ((s1 | s2_) & 3) break;
                // J cols: dG/dx, dG/dy
                float j00 = (Gx_.x - G.x) / h, j10 = (Gx_.y - G.x) / h;
                float j01 = (Gy_.x - G.y) / h, j11 = (Gy_.y - G.y) / h;
                float det = j00 * j11 - j01 * j10;
                float2 dx;
                if (fabs(det) > 1e-12f && isfinite(det)) {
                    dx = (float2)((-j11 * G.x + j01 * G.y) / det, (j10 * G.x - j00 * G.y) / det);  // -J^-1 G
                } else {
                    dx = -G / Jd;
                }
                dx *= 0.8f;
                dx = clamp(dx, -0.15f, 0.15f);
                dxy += dx;
            } else {
                // --- phase 3: damped-MD on the 2 soft DOFs ---
                v *= (1.0f - damp);
                v += G * dt;
                dxy += v * dt;
            }
        }
        // final eval at converged dxy -> pos. Skip when the loop already evaluated
        // fe at exactly this pos (converged break) — saves ~1 eval/slice.
        float rr = dxy.x*dxy.x + dxy.y*dxy.y;
        float Lz2 = L2 - rr;
        if (Lz2 < 1e-4f) { float sc = sqrt(fmax(L2*0.81f,1e-8f)/fmax(rr,1e-8f)); dxy *= sc; Lz2 = L2 - dxy.x*dxy.x - dxy.y*dxy.y; }
        float zz = -sqrt(Lz2);
        float3 d_ = (float3)(dxy.x, dxy.y, zz);
        pos = tipPos + rotMatT(d_, a3, b3, c3);
        if (!conv) {
            int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
            fe = cs_eval_contact_pme_local_at(pos.x, pos.y, pos.z,
                mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                LATOMS, LCOEFFS, nat, core_bucket_meta.w,
                &status, &min_r, &offender, &overflow);
            status_acc |= status;
            if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
            overflow_acc += overflow;
        }
        float4 fe_;
        fe_.xyz = rotMat(fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
        fe_.w = fe.w;
        int idx = gid * nz + iz;
        FEs[idx] = fe_;
        out_status[idx] = status_acc; out_min_r[idx] = min_r_acc; out_offender[idx] = offender_acc; out_overflow[idx] = overflow_acc;
        out_iters[idx] = (conv != 0 || f2l < f2conv * 1e4f) ? niter : -niter;
        tipPos += dTip.xyz;
        pos += dTip.xyz;
        if (iz_start_buf) out_pp[ipx * nz + iz] = (float4)(pos, 0.0f);   // two-pass only
        // warm-start dxy for next slice from new pos
        {
            float3 dv = rotMat(pos - tipPos, a3, b3, c3);
            dxy = dv.xy;
        }
    }
}

// Batch tile-eval for verification/diagnostics: WG t preloads tile t's atoms +
// mesh slab (kz = tile_kz[t*nz + iz_tile]), each lane evaluates one query via
// cs_eval_contact_pme_tile_at. Queries grouped by tile on the host:
// queries[t*lsize + lid].
__kernel void evalContactPMETile(
    __global const float* mesh_coeffs, const int4 mesh_meta, const float4 mesh_origin_h,
    __global const float4* atoms, __global const float* atom_coeffs,
    __global const int* bucket_atoms, __global const int* bucket_offsets,
    const int4 core_meta, const float4 core_bucket_meta,
    __global const float4* queries, __global float4* out_fe, __global int* out_status,
    __global const int4* tile_desc, __global const int* tile_kz,
    __global const int* wg_atom_offsets, __global const int* wg_atom_ids,
    const int4 tile_meta, const int iz_tile,
    __local float4* LATOMS, __local float* LCOEFFS, __local float* LMESH)
{
    const int iwg = get_group_id(0);
    const int lid = get_local_id(0), lsz = get_local_size(0);
    const int nzL = tile_meta.y;
    const int4 td = tile_desc[iwg];
    const int ix0 = td.x, iy0 = td.y, nxL = td.z, nyL = td.w;
    const int a0 = wg_atom_offsets[iwg];
    const int nloc = wg_atom_offsets[iwg + 1] - a0;
    const int nmesh = nxL * nyL * nzL;
    for (int j = lid; j < nloc; j += lsz) {
        const int ia = wg_atom_ids[a0 + j];
        LATOMS[j] = atoms[ia];
        const int ig = ia * CS_PME_NMODES, il = j * CS_PME_NMODES;
        const float4 c03 = vload4(0, atom_coeffs + ig);
        vstore4(c03, 0, LCOEFFS + il);
        LCOEFFS[il + 4] = atom_coeffs[ig + 4];
    }
    const int kz0 = tile_kz[iwg * (tile_meta.z ? tile_meta.z : 1) + iz_tile];
    for (int e = lid; e < nmesh; e += lsz) {
        const int izl = e % nzL, t2 = e / nzL, iy = t2 % nyL, ix = t2 / nyL;
        LMESH[e] = mesh_coeffs[((ix0 + ix) * mesh_meta.y + (iy0 + iy)) * mesh_meta.z + kz0 + izl];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int gid = get_global_id(0);
    const float4 q = queries[gid];
    int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
    float4 fe = cs_eval_contact_pme_tile_at(q.x, q.y, q.z,
        mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
        mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
        atoms, atom_coeffs, bucket_atoms, bucket_offsets,
        core_meta.x, core_meta.y, core_meta.z, core_meta.w,
        core_bucket_meta.x, core_bucket_meta.y, core_bucket_meta.z,
        LMESH, ix0, iy0, nxL, nyL, kz0, nzL,
        LATOMS, LCOEFFS, nloc, core_bucket_meta.w,
        wg_atom_ids + a0,
        &status, &min_r, &offender, &overflow, NULL);
    out_fe[gid] = fe;
    out_status[gid] = status;
}

// Tiled contact-PME relaxation (sph) — doc/Tasks/PME_ContactSurface_Opt.md A1–A3.
// WG = compact 2D pixel tile (Tx,Ty). Local memory holds ONLY the tile's nearby
// atoms + a thin mesh slab (nxL×nyL×nzL, streamed in z via tile_kz schedule).
// Host contract: 2D launch (nx_s-pad, ny_s-pad) × (Tx,Ty); lane (gx,gy) → pixel
// ipx = gx*ny_s + gy (ix-major pts). tile_desc[iwg] = (ix0,iy0,nxL,nyL);
// tile_kz[iwg*nz+iz] = base z-node of the cached slab at step iz;
// wg_atom_offsets/ids = per-tile atom CSR list. Padded lanes stay alive for barriers.
// PP escaping the tile → exact global eval, counted in out_esc[ipx].
// Single-pass only (no iz_start/out_pp resume); pass-2 uses LocalSph.
__kernel void relaxStrokesTiltedContactPMETileSph(
    __global const float* mesh_coeffs, const int4 mesh_meta, const float4 mesh_origin_h,
    __global const float4* atoms, __global const float* atom_coeffs,
    __global const int* bucket_atoms, __global const int* bucket_offsets,
    const int4 core_meta, const float4 core_bucket_meta,
    __global int* out_status, __global float* out_min_r, __global int* out_offender, __global int* out_overflow,
    __global int* out_iters, __global int* out_esc, __global int* out_nclamp,
    __global float4* points, __global float4* FEs,
    float4 tipA, float4 tipB, float4 tipC,
    float4 stiffness, float4 dpos0, float4 relax_params, float4 surfFF,
    int nx_s, int ny_s, int nz,
    __global const int4* tile_desc, __global const int* tile_kz,
    __global const int* wg_atom_offsets, __global const int* wg_atom_ids,
    const int4 tile_meta, const float rho_cap,
    __local float4* LATOMS, __local float* LCOEFFS, __local float* LMESH)
{
    const int gx = get_global_id(0), gy = get_global_id(1);
    const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int lsz = get_local_size(0) * get_local_size(1);
    const int ntx = tile_meta.x, nzL = tile_meta.y;
    const int iwg = get_group_id(1) * ntx + get_group_id(0);
    const bool active = (gx < nx_s) && (gy < ny_s);
    const int ipx = gx * ny_s + gy;
    const int4 td = tile_desc[iwg];
    const int ix0 = td.x, iy0 = td.y, nxL = td.z, nyL = td.w;
    const int a0 = wg_atom_offsets[iwg];
    const int nloc = wg_atom_offsets[iwg + 1] - a0;
    const int nat = core_meta.x;
    const int nmesh = nxL * nyL * nzL;
    // --- one-time cooperative preload: tile atoms + 5 coeffs each ---
    for (int j = lid; j < nloc; j += lsz) {
        const int ia = wg_atom_ids[a0 + j];
        LATOMS[j] = atoms[ia];
        const int ig = ia * CS_PME_NMODES, il = j * CS_PME_NMODES;
        const float4 c03 = vload4(0, atom_coeffs + ig);
        vstore4(c03, 0, LCOEFFS + il);
        LCOEFFS[il + 4] = atom_coeffs[ig + 4];
    }
    // --- initial mesh slab (iz = 0) ---
    int cache_kz = tile_kz[iwg * nz];
    for (int e = lid; e < nmesh; e += lsz) {
        const int izl = e % nzL, t2 = e / nzL, iy = t2 % nyL, ix = t2 / nyL;
        LMESH[e] = mesh_coeffs[((ix0 + ix) * mesh_meta.y + (iy0 + iy)) * mesh_meta.z + cache_kz + izl];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float3 dTip = tipC.xyz * tipC.w;
    float4 dpos0_ = dpos0;
    dpos0_.xyz = rotMatT(dpos0_.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
    const float L = dpos0.w;
    const float L2 = L * L;
    const float3 a3 = tipA.xyz, b3 = tipB.xyz, c3 = tipC.xyz;
    float3 tipPos = (float3)(0.0f, 0.0f, 0.0f);
    float3 pos = tipPos;
    float2 dxy = dpos0_.xy;
    if (active) {
        tipPos = points[ipx].xyz;
        pos = tipPos + rotMatT((float3)(dxy.x, dxy.y, -sqrt(fmax(L2 - dxy.x*dxy.x - dxy.y*dxy.y, 1e-4f))), a3, b3, c3);
    }
    const float dt_md = relax_params.x;
    const float damp0 = relax_params.y;
    const int imax = (relax_params.z > 0.5f) ? (int)relax_params.z : N_RELAX_STEP_MAX;
    const float f2conv = F2CONV * fmax(relax_params.w, 1e-12f);
    float2 Jd = stiffness.xy;                               // Hooke init: lateral tip stiffness
    const float rho_cap2 = rho_cap * rho_cap;               // hard lateral bound |dxy| <= rho_cap
    bool done = false;                                      // stroke aborted (PP at cap — deeper slices invalid)

    for (int iz = 0; iz < nz; iz++) {
        const int kz_new = tile_kz[iwg * nz + iz];
        if (kz_new != cache_kz) {                           // slab window moved — stream new planes
            barrier(CLK_LOCAL_MEM_FENCE);                   // all lanes done with old slab
            for (int e = lid; e < nmesh; e += lsz) {
                const int izl = e % nzL, t2 = e / nzL, iy = t2 % nyL, ix = t2 / nyL;
                LMESH[e] = mesh_coeffs[((ix0 + ix) * mesh_meta.y + (iy0 + iy)) * mesh_meta.z + kz_new + izl];
            }
            cache_kz = kz_new;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (!active || done) continue;                      // padded lanes / aborted strokes: barriers only

        int3 esc_iz = (int3)(0, 0, 0);
        int nclamp_iz = 0;
        float4 fe;
        float2 v = (float2)(0.0f, 0.0f);
        int status_acc = 0; float min_r_acc = 1e30f; int offender_acc = -1; int overflow_acc = 0;
        float2 fp = (float2)(0.0f); float2 xp = dxy;
        float f2p = 1e30f;
        float trust = 1.0f;
        float dt = dt_md, damp = damp0;
        const float dtmin = dt_md * 0.1f, dtmax = dt_md;
        int niter = 0, conv = 0;
        float f2l = 1e30f;
        for (int i = 0; i < imax; i++) {
            niter = i + 1;
            // place PP on sphere: hard lateral cap |dxy| <= rho_cap, then |dpos|=L
            float rr = dxy.x*dxy.x + dxy.y*dxy.y;
            if (rr > rho_cap2) { dxy *= rho_cap * rsqrt(rr); rr = rho_cap2; nclamp_iz++; }
            float Lz2 = L2 - rr;
            if (Lz2 < 1e-4f) {                              // >~90 deg bend — off-model, cap
                float sc = sqrt(fmax(L2 * 0.81f, 1e-8f) / fmax(rr, 1e-8f));
                dxy *= sc; Lz2 = L2 - dxy.x*dxy.x - dxy.y*dxy.y;
            }
            float zz = -sqrt(Lz2);
            float3 d_ = (float3)(dxy.x, dxy.y, zz);
            pos = tipPos + rotMatT(d_, a3, b3, c3);
            int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
            fe = cs_eval_contact_pme_tile_at(pos.x, pos.y, pos.z,
                mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                atoms, atom_coeffs, bucket_atoms, bucket_offsets,
                nat, core_meta.y, core_meta.z, core_meta.w,
                core_bucket_meta.x, core_bucket_meta.y, core_bucket_meta.z,
                LMESH, ix0, iy0, nxL, nyL, cache_kz, nzL,
                LATOMS, LCOEFFS, nloc, core_bucket_meta.w,
                wg_atom_ids + a0,
                &status, &min_r, &offender, &overflow, &esc_iz);
            status_acc |= status;
            if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
            overflow_acc += overflow;
            if (status & 1 || status & 2) { conv = -1; break; }
            float3 f = fe.xyz + rotMatT(tipForce(d_, stiffness, dpos0), a3, b3, c3) + c3 * surfFF.x;
            // tangent residual: G = f . e_i, e1=a+x/|z|c, e2=b+y/|z|c
            float iz_ = 1.0f / (-zz);
            float fc_ = dot(f, c3);
            float2 G = (float2)(dot(f, a3) + dxy.x * iz_ * fc_,
                                dot(f, b3) + dxy.y * iz_ * fc_);
            float f2 = dot(G, G);
            f2l = f2;
            if (f2 < f2conv) { conv = 1; break; }

            if (i < QN_BUDGET) {
                // --- phase 1: diagonal secant on 2 soft DOFs ---
                if (i > 0) {
                    float2 dd = dxy - xp;
                    float2 df = G - fp;
                    if (fabs(dd.x) > 1e-4f) Jd.x = df.x / dd.x;
                    if (fabs(dd.y) > 1e-4f) Jd.y = df.y / dd.y;
                    Jd = clamp(Jd, -1e6f, -0.1f);
                    if (f2 > f2p * 3.0f) { dxy = xp; trust = fmax(trust * 0.3f, 0.05f); }
                }
                float2 dx = -G / Jd;
                dx *= 0.7f * trust;
                dx = clamp(dx, -0.1f, 0.1f);
                xp = dxy; fp = G; f2p = f2;
                dxy += dx;
                trust = fmin(trust * 1.4f, 1.0f);
            } else if (i < QN_BUDGET + NR_BUDGET) {
                // --- phase 2: full 2x2 FD-Newton on (x,y) ---
                const float h = 1e-3f;
                float2 Gx_, Gy_; int s1 = 0, s2_ = 0; float mr; int of_, ov;
                // probe +x
                {
                    float2 dp = dxy + (float2)(h, 0.f);
                    float z2 = L2 - dp.x*dp.x - dp.y*dp.y; if (z2 < 1e-4f) z2 = 1e-4f;
                    float z3 = -sqrt(z2);
                    float3 dd_ = (float3)(dp.x, dp.y, z3);
                    float3 pp = tipPos + rotMatT(dd_, a3, b3, c3);
                    float4 fej = cs_eval_contact_pme_tile_at(pp.x, pp.y, pp.z,
                        mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                        mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                        atoms, atom_coeffs, bucket_atoms, bucket_offsets,
                        nat, core_meta.y, core_meta.z, core_meta.w,
                        core_bucket_meta.x, core_bucket_meta.y, core_bucket_meta.z,
                        LMESH, ix0, iy0, nxL, nyL, cache_kz, nzL,
                        LATOMS, LCOEFFS, nloc, core_bucket_meta.w,
                        wg_atom_ids + a0,
                        &s1, &mr, &of_, &ov, &esc_iz);
                    float3 fj = fej.xyz + rotMatT(tipForce(dd_, stiffness, dpos0), a3, b3, c3) + c3 * surfFF.x;
                    float izj = 1.0f / (-z3); float fcj = dot(fj, c3);
                    Gx_ = (float2)(dot(fj, a3) + dp.x * izj * fcj, dot(fj, b3) + dp.y * izj * fcj);
                }
                {
                    float2 dp = dxy + (float2)(0.f, h);
                    float z2 = L2 - dp.x*dp.x - dp.y*dp.y; if (z2 < 1e-4f) z2 = 1e-4f;
                    float z3 = -sqrt(z2);
                    float3 dd_ = (float3)(dp.x, dp.y, z3);
                    float3 pp = tipPos + rotMatT(dd_, a3, b3, c3);
                    float4 fej = cs_eval_contact_pme_tile_at(pp.x, pp.y, pp.z,
                        mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                        mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                        atoms, atom_coeffs, bucket_atoms, bucket_offsets,
                        nat, core_meta.y, core_meta.z, core_meta.w,
                        core_bucket_meta.x, core_bucket_meta.y, core_bucket_meta.z,
                        LMESH, ix0, iy0, nxL, nyL, cache_kz, nzL,
                        LATOMS, LCOEFFS, nloc, core_bucket_meta.w,
                        wg_atom_ids + a0,
                        &s2_, &mr, &of_, &ov, &esc_iz);
                    float3 fj = fej.xyz + rotMatT(tipForce(dd_, stiffness, dpos0), a3, b3, c3) + c3 * surfFF.x;
                    float izj = 1.0f / (-z3); float fcj = dot(fj, c3);
                    Gy_ = (float2)(dot(fj, a3) + dp.x * izj * fcj, dot(fj, b3) + dp.y * izj * fcj);
                }
                status_acc |= s1 | s2_;
                if ((s1 | s2_) & 3) break;
                // J cols: dG/dx, dG/dy
                float j00 = (Gx_.x - G.x) / h, j10 = (Gx_.y - G.x) / h;
                float j01 = (Gy_.x - G.y) / h, j11 = (Gy_.y - G.y) / h;
                float det = j00 * j11 - j01 * j10;
                float2 dx;
                if (fabs(det) > 1e-12f && isfinite(det)) {
                    dx = (float2)((-j11 * G.x + j01 * G.y) / det, (j10 * G.x - j00 * G.y) / det);  // -J^-1 G
                } else {
                    dx = -G / Jd;
                }
                dx *= 0.8f;
                dx = clamp(dx, -0.15f, 0.15f);
                dxy += dx;
            } else {
                // --- phase 3: damped-MD on the 2 soft DOFs ---
                v *= (1.0f - damp);
                v += G * dt;
                dxy += v * dt;
            }
        }
        // final eval at converged dxy -> pos. Skip when the loop already evaluated
        // fe at exactly this pos (converged break) — saves ~1 eval/slice.
        float rr = dxy.x*dxy.x + dxy.y*dxy.y;
        if (rr > rho_cap2) { dxy *= rho_cap * rsqrt(rr); rr = rho_cap2; nclamp_iz++; }
        float Lz2 = L2 - rr;
        if (Lz2 < 1e-4f) { float sc = sqrt(fmax(L2*0.81f,1e-8f)/fmax(rr,1e-8f)); dxy *= sc; Lz2 = L2 - dxy.x*dxy.x - dxy.y*dxy.y; }
        float zz = -sqrt(Lz2);
        float3 d_ = (float3)(dxy.x, dxy.y, zz);
        pos = tipPos + rotMatT(d_, a3, b3, c3);
        if (!conv) {
            int status = 0; float min_r = 1e30f; int offender = -1; int overflow = 0;
            fe = cs_eval_contact_pme_tile_at(pos.x, pos.y, pos.z,
                mesh_coeffs, mesh_meta.x, mesh_meta.y, mesh_meta.z,
                mesh_origin_h.x, mesh_origin_h.y, mesh_origin_h.z, mesh_origin_h.w,
                atoms, atom_coeffs, bucket_atoms, bucket_offsets,
                nat, core_meta.y, core_meta.z, core_meta.w,
                core_bucket_meta.x, core_bucket_meta.y, core_bucket_meta.z,
                LMESH, ix0, iy0, nxL, nyL, cache_kz, nzL,
                LATOMS, LCOEFFS, nloc, core_bucket_meta.w,
                wg_atom_ids + a0,
                &status, &min_r, &offender, &overflow, &esc_iz);
            status_acc |= status;
            if (min_r < min_r_acc) { min_r_acc = min_r; offender_acc = offender; }
            overflow_acc += overflow;
        }
        float4 fe_;
        fe_.xyz = rotMat(fe.xyz, tipA.xyz, tipB.xyz, tipC.xyz);
        fe_.w = fe.w;
        int idx = ipx * nz + iz;
        FEs[idx] = fe_;
        const bool at_cap = (rr >= rho_cap2 * 0.99f);       // converged state sits on the cap → invalid regime
        if (at_cap) status_acc |= 8;
        out_status[idx] = status_acc; out_min_r[idx] = min_r_acc; out_offender[idx] = offender_acc; out_overflow[idx] = overflow_acc;
        out_iters[idx] = (conv != 0 || f2l < f2conv * 1e4f) ? niter : -niter;
        out_esc[ipx * nz + iz] = esc_iz.x + esc_iz.y + esc_iz.z;
        out_nclamp[ipx * nz + iz] = nclamp_iz;
        if (at_cap) {
            // PP snapped to the lateral cap — deeper approach stays invalid.
            // Mark remaining slices (status bit 8, iters=-1) and stop this stroke.
            for (int iz2 = iz + 1; iz2 < nz; iz2++) {
                int idx2 = ipx * nz + iz2;
                FEs[idx2] = fe_;
                out_status[idx2] = 8; out_min_r[idx2] = min_r_acc; out_offender[idx2] = offender_acc; out_overflow[idx2] = 0;
                out_iters[idx2] = -1;
                out_esc[idx2] = 0; out_nclamp[idx2] = nclamp_iz;
            }
            done = true;
            continue;                                       // skip tipPos advance — stroke over
        }
        tipPos += dTip.xyz;
        pos += dTip.xyz;
        // warm-start dxy for next slice from new pos
        {
            float3 dv = rotMat(pos - tipPos, a3, b3, c3);
            dxy = dv.xy;
        }
    }
}

#endif // AFM_STANDALONE
