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

#endif // AFM_STANDALONE
