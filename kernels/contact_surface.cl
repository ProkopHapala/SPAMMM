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
    float bza[4] = {bz.x, bz.y, bz.z, bz.w};
    float dbxa[4] = {dbx.x, dbx.y, dbx.z, dbx.w};
    float dbya[4] = {dby.x, dby.y, dby.z, dby.w};
    float dbza[4] = {dbz.x, dbz.y, dbz.z, dbz.w};
    float e = 0.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f;
    for (int a = 0; a < 4; a++) {
        int ia = i0x + a; float cx = bxa[a], dx = dbxa[a];
        for (int b = 0; b < 4; b++) {
            int ib = i0y + b; float cy = bya[b], dy = dbya[b];
            float cxy = cx * cy;
            for (int c = 0; c < 4; c++) {
                int ic = i0z + c; float cz = bza[c], dz = dbza[c];
                float v = coeffs[(ia * ny + ib) * nz + ic];
                float w = cxy * cz;
                e  += w * v;
                gx += dx * cy * cz * v;
                gy += cx * dy * cz * v;
                gz += cxy * dz * v;
            }
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
    float min_r = 1e30f;
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
                float r = sqrt(dxp * dxp + dyp * dyp + dzp * dzp + 1e-20f);
                if (r < min_r) { min_r = r; offender = at; }
                if (r >= r_b_i) continue;
                float phi[CS_PME_NMODES], dphi[CS_PME_NMODES];
                cs_pme_core_basis(r, r_lo_i, r_b_i, phi, dphi);
                for (int m = 0; m < CS_PME_NMODES; m++) {
                    float c = atom_coeffs[at * CS_PME_NMODES + m];
                    E += c * phi[m];
                    if (r > 1e-8f) {
                        float dE_dr = c * dphi[m];
                        float inv_r = 1.0f / r;
                        Fx -= dE_dr * dxp * inv_r;
                        Fy -= dE_dr * dyp * inv_r;
                        Fz -= dE_dr * dzp * inv_r;
                    }
                }
            }
        }
    }
    int overflow = n_cand > CS_PME_CORE_MAX_CAND ? (n_cand - CS_PME_CORE_MAX_CAND) : 0;
    if (overflow > 0) status |= 4;
    *out_status = status; *out_min_r = min_r; *out_offender = offender; *out_overflow = overflow;
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

#endif // AFM_STANDALONE
