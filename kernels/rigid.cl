// rigid.cl - 6-DOF rigid body dynamics with quaternion integration
// ====================================================================
//
// RIGID BODY MOLECULAR DYNAMICS ON GPU
// ====================================
//
// This file implements rigid-body molecular dynamics (RBMD) for simulating
// molecules adsorbed on surfaces (e.g. AFM tip manipulation). Each rigid
// body has 6 degrees of freedom: 3 translational (CoM position) + 3
// rotational (quaternion orientation). Forces and torques are accumulated
// from per-atom interactions and integrated via symplectic Euler.
//
// --- Physics Summary ---
//
// State variables (per rigid body, stored in body frame):
//   pos    = (x, y, z, mass)          — CoM position and total mass
//   qrot   = (qx, qy, qz, qw)         — unit quaternion, body→world rotation
//   vpos   = (vx, vy, vz, 0)          — linear velocity (NOT momentum)
//   vrot   = (ωx, ωy, ωz, 0)         — angular velocity in BODY frame
//   I_body = 3×3 inertia tensor (body frame, constant)
//   Iinv   = inverse of I_body
//
// Equations of motion (symplectic Euler, body-frame angular velocity):
//
//   Translation (Newton):
//     v += (F / m) * dt
//     x += v * dt
//
//   Rotation (Euler's rigid body equation):
//     I·ω̇ + ω × (I·ω) = τ
//   => ω̇ = I⁻¹·(τ_body - ω × I·ω)
//   => ω += α * dt       where α = I⁻¹·(τ_body - ω × L_body)
//                                  L_body = I·ω
//
//   The gyroscopic term ω × (I·ω) is the Coriolis-like coupling that arises
//   from expressing angular velocity in the rotating body frame. For a
//   symmetric top (I = diag(I,I,I₃)) this produces the familiar precession.
//   OMITTING this term causes energy drift and incorrect tumbling for
//   asymmetric molecules (e.g. H₂O). See: Goldstein, Classical Mechanics,
//   Ch. 4; or Landau & Lifshitz, Mechanics, §33–35.
//
//   Quaternion update (right-multiply = body-frame increment):
//     q' = q ⊗ δq(ω·dt)
//   where δq(θ) = (θ·sinc(|θ|/2), cos(|θ|/2))  is the quaternion exponential.
//   We use a Taylor-series approximation for sinc and cos (see below).
//   Right multiplication ensures δq is expressed in the body frame.
//
// --- Integration Scheme ---
//
//   Symplectic (semi-implicit) Euler with per-step damping OR FIRE:
//     1. Compute F, τ from current positions/orientations
//     2a. Damped MD:  v *= damp;  ω *= damp
//     2b. FIRE (md_params.w < 0): if v·F < 0 → v=0; if ω·τ < 0 → ω=0;
//         else mix velocity toward drive and adapt dt/damp (Bitzek 2006)
//     3. Kick:  v += (F/m)·dt;  ω += α·dt
//     4. Drift: x += v·dt;     q = normalize(q ⊗ δq(ω·dt))
//
//   Damping: v *= damp is equivalent to viscous drag with γ = -ln(damp)/dt.
//   For physical (conservative) simulations set damp = 1.0.
//   For relaxation prefer FIRE (fast) or damp < 1 (e.g. 0.95).
//
//   CAVEAT: Per-step damping is NOT time-step invariant. Changing dt changes
//   the effective friction. For reproducible physics use damp = exp(-γ·dt)
//   and control γ, not damp directly. FIRE adapts dt locally.
//
// --- Validation Invariants ---
//
//   With F=0, τ=0, damp=1.0 (free rigid body):
//     • |q| = 1            (quaternion norm)
//     • R^T·R = I           (rotation orthogonality)
//     • |v| = const         (linear momentum conservation)
//     • L_world = R·(I·ω) = const  (angular momentum in world frame)
//     • T_rot = ½·ω^T·I·ω = const  (rotational kinetic energy)
//
//   Any drift in these signals a physics bug (e.g. missing gyro term,
//   wrong frame for torque, inconsistent I/I⁻¹).
//
// --- Kernel catalog (folded substrate family) ---
//
//   MD / FIRE (velocity integrator):
//     rigid_body_folded_kernel
//       1 workgroup = 1 rigid body. WORKGROUP_SIZE=32 threads split atoms
//       (ATOMS_PER_THREAD=4 → max 128 atoms). Force/torque tree-reduced in
//       __local; lid==0 integrates (symplectic Euler ± FIRE).
//
//     rigid_body_folded_replicas_kernel
//       1 thread = 1 replica (scan pixel / molecule copy). REPLICAS_WG=128.
//       Same MD/FIRE physics; molecule+basis cooperatively loaded to __local
//       once; no barriers in the step loop (imaging throughput path).
//
//   Pure Newton / trust-region (pose optimizer, NO velocities):
//     rigid_body_folded_newton_kernel
//       1 workgroup = 1 body (WG=32). Atoms parallel for each force eval;
//       Hessian 6×6 in __local; lid==0 builds FD columns + solves.
//
//     rigid_body_folded_newton_replicas_kernel
//       1 thread = 1 replica (WG=128). Hessian 6×6 in private memory;
//       each thread independently FD + LM-solve (imaging Newton path).
//
// --- How the 6×6 Newton step is solved (IMPORTANT) ---
//
//   We do NOT diagonalize H and we do NOT use Jacobi rotations.
//   Pseudocode per Newton iteration (coordinates u = (Δx, Δθ_body) ∈ R⁶):
//
//     G ← [F_world, τ_body]           // G = −∇E
//     for j = 0..5:                   // finite-difference Hessian
//         replica imaging: H[:,j] ← −(G(u+eps_j e_j)−G(u−eps_j e_j))/(2 eps_j)
//         single-body WG:  H[:,j] ← −(G(u+eps_j e_j)−G(u))/eps_j
//     H ← ½(H + Hᵀ)                   // symmetrize
//     loop trust / λ (Levenberg–Marquardt):
//         solve (H + λ I) Δ = G        // dense 6×6 Gaussian elimination
//                                     // with partial pivoting (rigid_solve6_lm)
//         if ||Δ|| > trust: Δ ← Δ · trust/||Δ||
//         try pose ← pose ⊕ Δ
//         if E_new < E: accept; maybe ↑trust, ↓λ
//         else: reject; ↑λ, ↓trust
//
//   Why serial GE on 6×6?
//     A 6×6 system is ~O(6³)=216 flops. Spreading it across 32 threads would
//     cost more in barriers/shared traffic than it saves. Parallelism is spent
//     where it matters: atoms (force eval) or replicas (many pixels).
//
//   Workgroup sizes:
//     WORKGROUP_SIZE = 32   — single-body kernels (atom parallelism)
//     REPLICAS_WG    = 128  — multi-replica kernels (1 thread / molecule)
//
// --- Three Kernel Variants (legacy list, see catalog above for Newton) ---
//
//   1. rigid_body_dynamics_kernel:
//      Generic forces from E-field + anchor springs. No substrate potential.
//      Used for testing and E-field-driven dynamics.
//
//   2. rigid_body_gridff_kernel:
//      Forces from precomputed B-spline GridFF (3D grid of PLQ coefficients).
//      Samples the grid at each atom's world position via tricubic B-spline
//      interpolation with periodic boundary conditions in x,y.
//      Grid must be pre-generated by SurfaceFF.
//
//   3. rigid_body_folded_kernel:          — MD/FIRE, 1 WG / body
//   4. rigid_body_folded_replicas_kernel: — MD/FIRE, 1 thread / replica
//   5. rigid_body_folded_newton_kernel:   — Newton, 1 WG / body
//   6. rigid_body_folded_newton_replicas_kernel: — Newton, 1 thread / replica
//
//   Folded force model (kernels 3–6):
//        E(x,y,z) = Σ_{i,b} c_{i,b} · cos(2π·k_u·u) · cos(2π·k_v·v) · exp(-α·(z-z₀))
//      where (u,v) are fractional surface-lattice coordinates.
//      Coefficients c_{i,b} are pre-fitted per atom type by
//      fit_folded_surface_basis() to encode Pauli+London+Coulomb(Ewald).
//      No grid needed — fully analytic, differentiable, and fast.
//
// --- GPU Parallelization ---
//
//   Single-body kernels (dynamics, gridff, folded, folded_newton):
//     1 body → 1 workgroup of WORKGROUP_SIZE=32 threads.
//     Atoms distributed round-robin (ATOMS_PER_THREAD=4 → max 128 atoms).
//     Force/torque tree-reduced in __local; lid==0 integrates or Newton-steps.
//
//   Multi-replica kernels (*_replicas_*):
//     1 replica → 1 thread; workgroup REPLICAS_WG=128.
//     Shared molecule/basis loaded cooperatively to __local once.
//     No barriers inside the MD/Newton iteration loop.
//
// --- Anchor Springs (Mouse Picking) ---
//
//   When anchors[ia].w > 0, atom ia is pulled toward anchors[ia].xyz by a
//   harmonic spring:  F = -k·(p_atom - p_anchor),  E = ½·k·|p_atom - p_anchor|²
//   This models mouse-drag manipulation in the GUI. The spring contributes
//   to both force, torque, and energy (included in atom_force.w).
//
// Helper functions: quat_mult, make_qrot, qrot_omega, quat_to_a/b/c,
// sinc_div_r2_taylor, quat_factors_taylor, folded_eval_*, folded_FT_*,
// rigid_solve6_lm, rigid_update_FIRE.
// Requires: common.cl + Forces.cl to be concatenated before this file.

#ifndef RIGID_DBG
#define RIGID_DBG 0
#endif

// ==================================================================
//  Helper Functions
// ==================================================================

// Quaternion multiplication: q_out = q1 ⊗ q2 (Hamilton product).
// Convention: (x, y, z, w) where w is the scalar part.
// This represents the composition of rotations: applying q1 then q2
// corresponds to q_out = q2 ⊗ q1 (note the order).
// In our kernels we use right-multiplication q' = q ⊗ δq to apply
// a body-frame rotation increment.
inline float4 quat_mult(float4 q1, float4 q2) {
    return (float4)(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

// Taylor series for sin(r)/r and (1-cos(r))/r², accurate to r^6.
// Used by the axis-angle quaternion exponential (make_qrot).
// CAVEAT: For |r| > ~0.5 rad the O(r^8) truncation error becomes
// non-negligible. In practice ω·dt is small (< 0.1 rad/step) so
// this is safe. For large rotations use make_qrot (trigonometric).
float2 sinc_div_r2_taylor(float r2){
    // s = sin(r)/r      = 1   - r^2/6  + r^4/120 - r^6/5040
    // c = (1-cos r)/r^2 = 1/2 - r^2/24 + r^4/720 - r^6/40320
    const float s = 1.0f + r2 * ( (-1.0f/6.0f)  + r2 * ( (1.0f/120.0f) + r2 * (-1.0f/5040.0f  ) ) );
    const float c = 0.5f + r2 * ( (-1.0f/24.0f) + r2 * ( (1.0f/720.0f) + r2 * (-1.0f/40320.0f ) ) );
    return (float2){s, c};
}

// Taylor series for sin(r/2)/r and cos(r/2), accurate to r^6.
// These are the scalar/vector factors of the quaternion exponential:
//   δq(θ) = (θ · sin(|θ|/2)/|θ|,  cos(|θ|/2))
// where θ = ω·dt is the body-frame rotation vector.
// Input r2 = |θ|² = dot(θ,θ).
// CAVEAT: Same accuracy limit as sinc_div_r2_taylor — keep |ω·dt| < 0.5.
inline float2 quat_factors_taylor(float r2){
    // s = sin(r/2)/r = 1/2 - r2/48 + r2^2/3840 - r2^3/645120
    // c = cos(r/2)   = 1   - r2/8  + r2^2/384  - r2^3/46080
    const float s = 0.5f + r2 * ((-1.0f/48.0f)  + r2 * ((1.0f/3840.0f) + r2 * (-1.0f/645120.0f)));
    const float c = 1.0f + r2 * ((-1.0f/8.0f)   + r2 * ((1.0f/384.0f)  + r2 * (-1.0f/46080.0f)));
    return (float2)(s, c);
}

// Update quaternion by body-frame angular velocity ω over time dt:
//   q' = q ⊗ δq(ω·dt)
// Uses Taylor-series quaternion exponential (fast, accurate for small ω·dt).
float4 qrot_omega_taylor( float4 qrot, float3 omega){
    const float r2 = dot(omega,omega);
    const float2 sc = quat_factors_taylor(r2);
    return quat_mult(qrot, (float4)(omega * sc.x, sc.y) );
}

// Build quaternion exponential δq from rotation vector θ = ω·dt:
//   δq = (θ · sin(|θ|/2)/|θ|,  cos(|θ|/2))
// Taylor-series version (fast path, used in all kernels).
float4 make_qrot_taylor(  float3 omega){
    const float r2 = dot(omega,omega);
    const float2 sc = quat_factors_taylor(r2);
    return (float4)(omega * sc.x, sc.y);
}

// Build quaternion exponential δq from rotation vector θ = ω·dt.
// Trigonometric version (exact, used for validation/large angles).
inline float4 make_qrot(float3 omega){
    const float angle = length(omega);
    if(angle < 1e-12f) return (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    const float3 axis = omega / angle;
    const float s = sin(0.5f * angle);
    float c = cos(0.5f * angle);
    return (float4)(axis * s, c);
}

// Update quaternion by body-frame angular velocity (trigonometric version).
float4 qrot_omega( float4 qrot, float3 omega){
    const float4 dq  = make_qrot(omega);
    return quat_mult(qrot, dq);
}

// FIRE velocity update (Bitzek et al., PRL 2006). Essential quench:
//   if v·f < 0 → v = 0  (velocity against force / torque)
// else mix v toward f and adapt dt/damp. Used for both linear (f=F, v=v)
// and angular (f=τ_body, v=ω_body) DOFs.
#ifndef RIGID_FIRE_FTDEC
#define RIGID_FIRE_FTDEC 0.5f
#endif
#ifndef RIGID_FIRE_FTINC
#define RIGID_FIRE_FTINC 1.1f
#endif
#ifndef RIGID_FIRE_FDAMP
#define RIGID_FIRE_FDAMP 0.99f
#endif
#ifndef RIGID_FIRE_F2SAFE
#define RIGID_FIRE_F2SAFE 1e-8f
#endif
// FIRE quench helper (Bitzek 2006 style, AFM-like).
// If v·f < 0 → zero velocity, shrink dt, reset damp; else mix v toward f and grow dt.
// Applied separately to linear (vpos,F) and angular (ω,τ_body) channels in MD kernels.
inline float3 rigid_update_FIRE(float3 f, float3 v, float* dt, float* damp, float dtmin, float dtmax, float damp0){
    const float ff = dot(f, f);
    const float vv = dot(v, v);
    const float vf = dot(v, f);
    if (vf < 0.0f) {
        v *= 0.0f;
        *dt   = fmax(dtmin, (*dt) * RIGID_FIRE_FTDEC);
        *damp = damp0;
    } else {
        v *= (1.0f - (*damp));
        v += f * ((*damp) * sqrt(vv / (ff + RIGID_FIRE_F2SAFE)));
        *dt    = fmin(dtmax, (*dt) * RIGID_FIRE_FTINC);
        *damp *= RIGID_FIRE_FDAMP;
    }
    return v;
}

// Rotate vector v by rotation matrix R: v' = R·v  (body→world transform).
// R is stored as cl_Mat3 with rows (a, b, c) = (x-axis, y-axis, z-axis)
// of the body frame expressed in world coordinates.
inline float3 rotate_vec_by_matrix(const float3 v, __local const cl_Mat3* R) {
    return (float3)(
        dot(R->a.xyz, v.xyz),
        dot(R->b.xyz, v.xyz),
        dot(R->c.xyz, v.xyz)
    );
}

// Rotate vector v by R^T: v' = R^T·v  (world→body transform).
// Used to convert world-frame torque to body-frame torque.
inline float3 rotate_vec_by_matrix_T(const float3 v, __local const cl_Mat3* R) {  return R->a.xyz*v.x + R->b.xyz*v.y + R->c.xyz*v.z;}

// Extract body-frame axis vectors (columns of R) from quaternion q.
// These are the rows of the rotation matrix R that maps body→world:
//   R = [ a | b | c ]   (a = x-axis, b = y-axis, c = z-axis)
// Standard quaternion→rotation-matrix formula (Goldstein Eq. 4.46).
inline float3 quat_to_a( float4 q ){  return(float3)(1.0f-2.0f*(q.y*q.y + q.z*q.z),      2.0f*(q.x*q.y - q.z*q.w),      2.0f*(q.x*q.z + q.y*q.w));}
inline float3 quat_to_b( float4 q ){  return(float3)(     2.0f*(q.x*q.y + q.z*q.w), 1.0f-2.0f*(q.x*q.x + q.z*q.z),      2.0f*(q.y*q.z - q.x*q.w));}
inline float3 quat_to_c( float4 q ){  return(float3)(     2.0f*(q.x*q.z - q.y*q.w),      2.0f*(q.y*q.z + q.x*q.w), 1.0f-2.0f*(q.x*q.x + q.y*q.y));}

// Matrix-vector products for cl_Mat3.
//   mat3_dot(M, v)   = M·v   (rows of M dotted with v)
//   mat3_dot_T(M, v) = M^T·v (columns of M dotted with v)
// Used for: I·ω (inertia×angular velocity), I⁻¹·τ (inverse inertia×torque),
//           R^T·τ_world (world→body torque transform).
inline float3 mat3_dot(const cl_Mat3 M, const float3 v){
    return (float3)( dot(M.a.xyz, v), dot(M.b.xyz, v), dot(M.c.xyz, v) );
}

inline float3 mat3_dot_T(const cl_Mat3 M, const float3 v){
    return M.a.xyz*v.x + M.b.xyz*v.y + M.c.xyz*v.z;
}

inline int4 make_inds_pbc(const int n, const int iG) {
    switch( iG ){
        case 0: { return (int4)(0, 1,   2,   3  ); }
        case 1: { return (int4)(0, 1,   2,   3-n); }
        case 2: { return (int4)(0, 1,   2-n, 3-n); }
        case 3: { return (int4)(0, 1-n, 2-n, 3-n); }
    }
    return (int4)(-100, -100, -100, -100);
}

inline int4 choose_inds_pbc(const int i, const int n, const __local int4* iqs) {
    if (i >= (n-3)) {
        const int ii = i + 4 - n;
        return iqs[ii];
    }
    return (int4)(0, +1, +2, +3);
}

inline float4 basis(float u) {
    const float inv6 = 1.0f / 6.0f;
    const float u2 = u * u;
    const float t = 1.0f - u;
    return (float4)(
        inv6 * t * t * t,
        inv6 * (3.0f * u2 * (u - 2.0f) + 4.0f),
        inv6 * (3.0f * u * (1.0f + u - u2) + 1.0f),
        inv6 * u2 * u
    );
}

inline float4 dbasis(float u) {
    const float u2 = u * u;
    const float t = 1.0f - u;
    return (float4)(
        -0.5f * t * t,
        0.5f * (3.0f * u2 - 4.0f * u),
        0.5f * (-3.0f * u2 + 2.0f * u + 1.0f),
        0.5f * u2
    );
}

inline float2 fe1Dcomb(__global const float4* E, const float4 C, const float4 p, const float4 d) {
    const float4 cs = (float4)(dot(C, E[0]), dot(C, E[1]), dot(C, E[2]), dot(C, E[3]));
    return (float2)(dot(p, cs), dot(d, cs));
}

inline float3 fe2d_comb(int nz, __global const float4* E, int4 di, const float4 C, const float4 pz, const float4 dz, const float4 by, const float4 dy) {
    const float2 fe0 = fe1Dcomb(E + di.x, C, pz, dz);
    const float2 fe1 = fe1Dcomb(E + di.y, C, pz, dz);
    const float2 fe2 = fe1Dcomb(E + di.z, C, pz, dz);
    const float2 fe3 = fe1Dcomb(E + di.w, C, pz, dz);
    return (float3)(
        fe0.x * dy.x + fe1.x * dy.y + fe2.x * dy.z + fe3.x * dy.w,
        fe0.y * by.x + fe1.y * by.y + fe2.y * by.z + fe3.y * by.w,
        fe0.x * by.x + fe1.x * by.y + fe2.x * by.z + fe3.x * by.w
    );
}

inline float4 fe3d_pbc_comb(const float3 u, const int3 n, __global const float4* Es, const float4 PLQH, __local const int4* xqis, __local int4* yqis) {
    int ix = (int)u.x;
    int iy = (int)u.y;
    int iz = (int)u.z;
    if (u.x < 0) ix--;
    if (u.y < 0) iy--;
    const float tx = u.x - ix;
    const float ty = u.y - iy;
    const float tz = u.z - iz;
    if ((iz < 1) || (iz >= n.z - 2)) return (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    ix = modulo(ix-1, n.x);
    iy = modulo(iy-1, n.y);
    const int nyz = n.z * n.y;
    int4 qx = choose_inds_pbc(ix, n.x, xqis);
    const int4 qy = choose_inds_pbc(iy, n.y, yqis) * n.z;
    const float4 bz = basis(tz);
    const float4 dz = dbasis(tz);
    const float4 by = basis(ty);
    const float4 dy = dbasis(ty);
    const int i0 = (iz - 1) + n.z * (iy + n.y * ix);
    qx *= nyz;
    float3 E1 = fe2d_comb(n.z, Es + (i0 + qx.x), qy, PLQH, bz, dz, by, dy);
    float3 E2 = fe2d_comb(n.z, Es + (i0 + qx.y), qy, PLQH, bz, dz, by, dy);
    float3 E3 = fe2d_comb(n.z, Es + (i0 + qx.z), qy, PLQH, bz, dz, by, dy);
    float3 E4 = fe2d_comb(n.z, Es + (i0 + qx.w), qy, PLQH, bz, dz, by, dy);
    const float4 bx = basis(tx);
    const float4 dx = dbasis(tx);
    return (float4)(
        dot(dx, (float4)(E1.z, E2.z, E3.z, E4.z)),
        dot(bx, (float4)(E1.x, E2.x, E3.x, E4.x)),
        dot(bx, (float4)(E1.y, E2.y, E3.y, E4.y)),
        dot(bx, (float4)(E1.z, E2.z, E3.z, E4.z))
    );
}

// ==================================================================
//  Constants
// ==================================================================
// WORKGROUP_SIZE: compile-time max for local-memory arrays. The actual
//   workgroup size (lsize) is set at kernel launch and may differ.
//   Atom loops use lsize (runtime), NOT WORKGROUP_SIZE, for correctness.
// ATOMS_PER_THREAD: each thread processes up to this many atoms per step.
//   With WORKGROUP_SIZE=32 and ATOMS_PER_THREAD=4, max atoms = 128.
// CAVEAT: If na > WORKGROUP_SIZE*ATOMS_PER_THREAD, excess atoms are
//   silently skipped. Ensure na ≤ 128 or increase these constants.
#define WORKGROUP_SIZE     32
#define MAX_ATOMS_PER_BODY 128
#define ATOMS_PER_THREAD   4

// ==================================================================
//  Kernel 1: rigid_body_dynamics_kernel (generic E-field + anchors)
// ==================================================================
//
//  Simplest kernel: forces come from a uniform E-field (acting on atom
//  charges stored in apos_body.w) plus optional anchor springs.
//  No substrate potential. Used for testing and E-field-driven dynamics.
//
//  Arguments:
//    mols[gid+1]-mols[gid] = number of atoms in body gid
//    poss[gid]  = (x, y, z, mass)
//    qrots[gid] = (qx, qy, qz, qw)  unit quaternion
//    vposs[gid] = (vx, vy, vz, 0)   linear velocity
//    vrots[gid] = (ωx, ωy, ωz, 0)  angular velocity (BODY frame)
//    I_body_inv[gid] = I⁻¹ (body frame, 3×3)
//    I_body[gid]     = I   (body frame, 3×3, needed for gyroscopic term)
//    apos_body[ia]   = (x, y, z, q)  atom position in body frame, q=charge
//    anchors[ia]     = (ax, ay, az, k)  anchor target + spring constant
//                     k > 0: active spring; k < 0: no anchor
//    md_params = (lin_damp, ang_damp, force_scale, torque_scale)
//    Efield = (Ex, Ey, Ez, 0)
//
//  Physics: F_atom = q·E + F_anchor;  τ = Σ r_i × F_i
//           Euler:  α = I⁻¹·(τ_body - ω × I·ω)
//
__kernel
void rigid_body_dynamics_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   anchors,
    const int   natoms,
    const int   niter,
    const float dt,
    const float4             md_params,
    const float3  Efield
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    const int ia0 = mols[gid];
    const int na  = mols[gid+1]-ia0;
    // FIRE locals (adapted per step when md_params.w < 0)
    float dt_lin = dt;
    float dt_ang = dt;
    float damp_lin = md_params.x;
    float damp_ang = md_params.y;
    const float damp0_lin = md_params.x;
    const float damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f;
    const float dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    if (lid == 0) {
        pos      = poss   [gid];
        qrot     = qrots  [gid];  qrot=normalize(qrot);
        vpos     = vposs  [gid];
        vrot     = vrots  [gid];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[gid].a;
        Iinv_body.b = I_body_inv[gid].b;
        Iinv_body.c = I_body_inv[gid].c;
        Ibody.a     = I_body[gid].a;
        Ibody.b     = I_body[gid].b;
        Ibody.c     = I_body[gid].c;
    }
    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i=0; i<ATOMS_PER_THREAD; i++) {
            int atom_idx = lid+i*lsize;
            if(atom_idx >= na){ break; }
            float4 p_body  = apos_body[ia0+atom_idx];
            float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            float3 p_world = pos.xyz + r_world;
            float4 f = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            f.xyz += Efield.xyz*p_body.w;
            float4 anchor   = anchors[ia0+atom_idx];
            if(anchor.w > 0.0f){
                float3 d  = p_world.xyz - anchor.xyz;
                float3 fa = d * -anchor.w;
                f.xyz    += fa;
            }
            float3 tq = cross(r_world, f.xyz);
            total_torque.xyz += tq;
            total_force .xyz += f.xyz;
        }
        Ltorq [lid] = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if ( lid == 0 ){
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            float3 tq_body  = mat3_dot_T(R, tq_world);
            float3 L_body    = mat3_dot(Ibody, vrot.xyz);
            float3 gyro      = cross(vrot.xyz, L_body);
            float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                // FIRE: zero v if v·F<0 (and ω if ω·τ<0); else mix toward drive
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            float4 dq = make_qrot_taylor(vrot.xyz * dta);
            qrot = normalize(quat_mult(qrot, dq));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // --- Write back final state and world positions ---
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i=0; i<ATOMS_PER_THREAD; i++) {
        int atom_idx = lid+i*lsize;
        if(atom_idx >= na){ break; }
        int ia = ia0+atom_idx;
        float4 p_body  = apos_body[ia];
        float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia] = (float4){p_world, 0.f};
    }
    if (lid == 0) {
        poss   [gid] = pos;
        qrots  [gid] = qrot;
        vposs  [gid] = vpos;
        vrots  [gid] = vrot;
    }
}

// ==================================================================
//  Kernel 2: rigid_body_gridff_kernel (B-spline GridFF substrate)
// ==================================================================
//
//  Forces from a precomputed 3D B-spline grid of PLQ coefficients.
//  The grid stores (dE/dx, dE/dy, dE/dz, E) at each grid point.
//  Tricubic B-spline interpolation with PBC in x,y is used to sample
//  at each atom's world position. Force = -∇E = -fe.xyz * inv_dg.
//
//  Same integration physics as Kernel 1 (Euler's equation with gyro).
//  Additionally outputs per-atom force and energy for diagnostics.
//
//  Grid layout: Es[nx*ny*nz] with index = iz + nz*(iy + ny*ix)
//  PLQ per atom: atom_PLQ[ia] = (cP, cL, Q, cH) from REQ conversion
//
__kernel
void rigid_body_gridff_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   atom_PLQ,
    __global const float4*   BsplinePLQ,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    const int4               grid_ns,
    const float4             grid_invStep,
    const float4             grid_p0,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local int4 xqs[4];
    __local int4 yqs[4];
    const int ia0 = mols[gid];
    const int na  = mols[gid+1]-ia0;
    float dt_lin = dt, dt_ang = dt;
    float damp_lin = md_params.x, damp_ang = md_params.y;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    if      (lid < 4){ xqs[lid]   = make_inds_pbc(grid_ns.x, lid); }
    else if (lid < 8){ yqs[lid-4] = make_inds_pbc(grid_ns.y, lid-4); }
    if (lid == 0) {
        pos      = poss[gid];
        qrot     = normalize(qrots[gid]);
        vpos     = vposs[gid];
        vrot     = vrots[gid];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[gid].a;
        Iinv_body.b = I_body_inv[gid].b;
        Iinv_body.c = I_body_inv[gid].c;
        Ibody.a     = I_body[gid].a;
        Ibody.b     = I_body[gid].b;
        Ibody.c     = I_body[gid].c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float3 inv_dg = grid_invStep.xyz;
    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i=0; i<ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if(atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            const float3 p_world = pos.xyz + r_world;
            const float4 fe = fe3d_pbc_comb((p_world - grid_p0.xyz) * inv_dg, grid_ns.xyz, BsplinePLQ, atom_PLQ[ia], xqs, yqs);
            float3 f = fe.xyz * (-inv_dg);
            float E_atom = fe.w;
            float4 anchor = anchors[ia];
            if(anchor.w > 0.0f){
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E_atom += 0.5f * anchor.w * dot(d, d);
            }
            total_force.xyz += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E_atom);
            atom_force[ia] = (float4)(f, E_atom);
#if RIGID_DBG
            if((gid==0)&&(step==0)&&(atom_idx<4)){
                printf("RIGID_DBG atom %i p(%g,%g,%g) PLQ(%g,%g,%g,%g) fe(%g,%g,%g,%g)\n", atom_idx, p_world.x,p_world.y,p_world.z, atom_PLQ[ia].x,atom_PLQ[ia].y,atom_PLQ[ia].z,atom_PLQ[ia].w, f.x,f.y,f.z,fe.w);
            }
#endif
        }
        Ltorq[lid] = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            const float force_scale = use_fire ? 1.0f : md_params.z;
            const float torque_scale = use_fire ? 1.0f : md_params.w;
            const float3 f = Lforce[0].xyz * force_scale;
            const float3 tq_world = Ltorq[0].xyz * torque_scale;
            body_force [gid] = (float4)(f, 0.0f);
            body_torque[gid] = (float4)(tq_world, 0.0f);
#if RIGID_DBG
            if(gid==0){
                printf("RIGID_DBG body f(%g,%g,%g) tq(%g,%g,%g) pos(%g,%g,%g)\n", f.x,f.y,f.z, tq_world.x,tq_world.y,tq_world.z, pos.x,pos.y,pos.z);
            }
#endif
            const float3 tq_body  = mat3_dot_T(R, tq_world);
            const float3 L_body    = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro      = cross(vrot.xyz, L_body);
            const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // --- Write back final state and world positions ---
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i=0; i<ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if(atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
    if (lid == 0) {
        poss [gid] = pos;
        qrots[gid] = qrot;
        vposs[gid] = vpos;
        vrots[gid] = vrot;
    }
}

// ==================================================================
//  Kernel 3: rigid_body_folded_kernel (analytic folded basis)
// ==================================================================
//
//  Surface forces from an analytic folded basis expansion:
//    E(x,y,z) = Σ_{i,b} c_{i,b} · cos(2π·k_u·u) · cos(2π·k_v·v) · exp(-α·(z-z₀))
//
//  where (u,v) are fractional coordinates w.r.t. the 2D surface lattice:
//    u = (b_y·x - b_x·y) / det    (fractional coordinate along lattice vector a)
//    v = (-a_y·x + a_x·y) / det   (fractional coordinate along lattice vector b)
//  with det = a_x·b_y - b_x·a_y.
//
//  The inverse lattice matrix is:
//    [ du/dx  du/dy ]   [  b_y/det  -b_x/det ]
//    [ dv/dx  dv/dy ] = [ -a_y/det   a_x/det ]
//  stored as invLvec2d = (du/dx, du/dy, dv/dx, dv/dy).
//
//  CAVEAT: For non-orthogonal (sheared) lattices, the off-diagonal terms
//  (du/dy, dv/dx) are non-zero and MUST be correctly assigned. Swapping
//  them produces wrong forces for sheared cells (invisible for diagonal
//  lattices like NaCl where off-diagonal = 0).
//
//  Gradient (chain rule through fractional coordinates):
//    dE/dx = dE/du · du/dx + dE/dv · dv/dx
//    dE/dy = dE/du · du/dy + dE/dv · dv/dy
//    dE/dz = -α · E_basis   (for z > z₀)
//
//  Force on atom: F = -∇E = -Σ_b c_b · ∇basis_b
//
//  Same integration physics as Kernels 1 and 2 (Euler's equation with gyro).
//
//  Coefficients c_{i,b} are pre-fitted per atom type by
//  fit_folded_surface_basis() to encode Pauli+London+Coulomb(Ewald).
//  No grid precomputation needed — fully analytic and differentiable.
//
//  Requires: common.cl + Forces.cl + rigid.cl (self-contained)

#ifndef FOLDED_BASIS_MAX_RIGID
#define FOLDED_BASIS_MAX_RIGID 128
#endif
#ifndef FOLDED_TYPES_MAX_RIGID
#define FOLDED_TYPES_MAX_RIGID 8
#endif

// Evaluate one folded basis function at fractional (u,v) and height z.
//   B = cos(2π k_u u) · cos(2π k_v v) · exp(−α · max(0, z−z₀))
// prm = (k_u, k_v, α, z₀).  Used by energy accumulation and diagnostics.
inline float folded_eval_basis_rigid(float u, float v, float z, float4 prm){
    float bx = cos( (2.0f*M_PI_F) * prm.x * u );
    float by = cos( (2.0f*M_PI_F) * prm.y * v );
    float dz = fmax(0.0f, z - prm.w);
    float bz = exp( -prm.z * dz );
    return bx * by * bz;
}

// Analytic ∇B in world (x,y,z) for one basis function.
// Chain rule through fractional coords:
//   ∂B/∂x = (∂B/∂u)(du/dx) + (∂B/∂v)(dv/dx),  same for y;  ∂B/∂z = −α B (z>z₀).
// invLvec2d = (du/dx, du/dy, dv/dx, dv/dy). Force contribution is F = −c · ∇B.
// CAVEAT: off-diagonal du/dy, dv/dx matter for sheared cells (zero on NaCl).
inline float3 folded_eval_grad_rigid(float u, float v, float z, float4 prm, float4 invLvec2d){
    float phix = (2.0f*M_PI_F) * prm.x;
    float phiy = (2.0f*M_PI_F) * prm.y;
    float cu = cos(phix*u);
    float su = sin(phix*u);
    float cv = cos(phiy*v);
    float sv = sin(phiy*v);
    float dz = fmax(0.0f, z - prm.w);
    float bz = exp(-prm.z * dz);
    float dEdu = -phix * su * cv * bz;
    float dEdv = -phiy * cu * sv * bz;
    float dEdz = (z > prm.w) ? (-prm.z * cu * cv * bz) : 0.0f;
    float dudx = invLvec2d.x;
    float dudy = invLvec2d.y;
    float dvdx = invLvec2d.z;
    float dvdy = invLvec2d.w;
    return (float3)( dEdu*dudx + dEdv*dvdx, dEdu*dudy + dEdv*dvdy, dEdz );
}

// folded_meta.y > 0: legacy typed scalar coefficients.
// folded_meta.y < 0: one substrate-only float4 coefficient per basis and
// runtime atom PLQH stored in the folded_atom_type buffer.
inline float folded_coeff_rigid(__local const float* coeffs, int ib, int nbasis, int ityp, float4 plqh, int factorized){
    return factorized ? dot(vload4(ib, coeffs), plqh) : coeffs[ityp*nbasis + ib];
}

// ---- Tensor-product FAF evaluator (materialized scalar coefficients) --------
//
// Same linear physics as folded_eval_basis_rigid/grad_rigid but uses Fourier
// recurrence (one sincos per axis + cmul chain) instead of calling cos/sin
// for every basis function.  For nu=nv=4, nz=7 this cuts special-function work
// from ~224 trig + 112 exp to 2 sincos + 7 exp per atom.
//
// Coefficients are pre-materialized on the host to one scalar per (site,basis)
// (typed OR factorized → same runtime format) and stored basis-major:
//     site_coeffs[b * total_sites + site]
// so all work-items in a warp read adjacent site values for a fixed basis b.
//
// The basis MUST be a regular tensor product with index (iu*nv+iv)*nz+iz,
// ku=iu, kv=iv, and per-iz (alpha,z0).  Irregular legacy bases are rejected
// on the host (caller checks before launch).
//
// Returns float4 (.xyz = force F = -∇E, .w = energy E).
inline float4 folded_eval_tensor_rigid(
    float3 p_world, int site, int total_sites,
    __global const float*  site_coeffs,   // [nbasis * total_sites] basis-major
    __global const float4* z_params,       // [nz] (alpha, z0, 0, 0)
    int nu, int nv, int nz,
    float4 invLvec2d                       // (du/dx, du/dy, dv/dx, dv/dy)
){
    float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
    float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
    u = u - floor(u); v = v - floor(v);
    float cu, su = sincos(2.0f*M_PI_F*u, &cu);
    float cv, sv = sincos(2.0f*M_PI_F*v, &cv);
    float2 z1_u = (float2)(cu, su);
    float2 z1_v = (float2)(cv, sv);
    // Private arrays: harmonic recurrences + z exponentials
    float2 cu_arr[16]; float2 cv_arr[16];   // up to 16 harmonics per axis
    float  bz_arr[16];  float  dbz_arr[16]; // up to 16 z exponentials
    cu_arr[0] = (float2)(1.0f, 0.0f);
    cv_arr[0] = (float2)(1.0f, 0.0f);
    for (int i = 1; i < nu; i++) cu_arr[i] = (float2)(cu_arr[i-1].x*z1_u.x - cu_arr[i-1].y*z1_u.y, cu_arr[i-1].x*z1_u.y + cu_arr[i-1].y*z1_u.x);
    for (int i = 1; i < nv; i++) cv_arr[i] = (float2)(cv_arr[i-1].x*z1_v.x - cv_arr[i-1].y*z1_v.y, cv_arr[i-1].x*z1_v.y + cv_arr[i-1].y*z1_v.x);
    for (int iz = 0; iz < nz; iz++) {
        float alpha = z_params[iz].x, z0 = z_params[iz].y;
        float dz = fmax(0.0f, p_world.z - z0);
        bz_arr[iz]   = exp(-alpha * dz);
        dbz_arr[iz]  = (p_world.z > z0) ? (-alpha * bz_arr[iz]) : 0.0f;
    }
    float E = 0.0f, dEdu = 0.0f, dEdv = 0.0f, dEdz = 0.0f;
    for (int iv = 0; iv < nv; iv++) {
        float by  = cv_arr[iv].x;
        float dby = -2.0f*M_PI_F*(float)iv * cv_arr[iv].y;
        for (int iu = 0; iu < nu; iu++) {
            float bx  = cu_arr[iu].x;
            float dbx = -2.0f*M_PI_F*(float)iu * cu_arr[iu].y;
            for (int iz = 0; iz < nz; iz++) {
                float bz = bz_arr[iz], dbz = dbz_arr[iz];
                float B  = bx * by * bz;
                int b = (iu*nv + iv)*nz + iz;
                float c = site_coeffs[b * total_sites + site];
                E    += c * B;
                dEdu += c * (dbx * by  * bz);
                dEdv += c * (bx  * dby * bz);
                dEdz += c * (bx  * by  * dbz);
            }
        }
    }
    float3 f;
    f.x = -(dEdu * invLvec2d.x + dEdv * invLvec2d.z);
    f.y = -(dEdu * invLvec2d.y + dEdv * invLvec2d.w);
    f.z = -dEdz;
    return (float4)(f, E);
}

// ==================================================================
//  Kernel: rigid_body_folded_kernel  — MD / FIRE, one body per workgroup
// ==================================================================
//
// ROLE: Relax or integrate ONE rigid molecule on a folded-basis substrate
//       using symplectic Euler (optional FIRE quench when md_params.w < 0).
//
// PARALLELISM:
//   get_group_id(0) = body index
//   get_local_size  = WORKGROUP_SIZE (32)
//   Each lid owns atoms lid, lid+32, lid+64, … (ATOMS_PER_THREAD slots)
//   Tree-reduce F,τ in __local Lforce/Ltorq; lid==0 integrates pose.
//
// NOT a Newton optimizer — see rigid_body_folded_newton_kernel for that.
//
__kernel
void rigid_body_folded_kernel(
    __global const int*      mols,             // 1
    __global       float4*   poss,             // 2
    __global       float4*   qrots,            // 3
    __global       float4*   vposs,            // 4
    __global       float4*   vrots,            // 5
    __global       float4*   fire_state,       // persistent (dt_lin, dt_ang, damp_lin, damp_ang)
    __global const cl_Mat3*  I_body_inv,       // 6
    __global const cl_Mat3*  I_body,           // 6b
    __global const float4*   apos_body,        // 7
    __global       float4*   apos_world,       // 8
    __global       float4*   atom_force,       // 9
    __global       float4*   body_force,       // 10
    __global       float4*   body_torque,      // 11
    __global const float4*   anchors,          // 12
    __global const float*    folded_coeffs,    // 13  [ntypes*nbasis]
    __global const float4*   folded_kxyz,      // 14  [nbasis]
    __global const int*      folded_atom_type, // 15  [natoms]
    const int4               folded_meta,      // 16  (nbasis, ntypes, 0, 0)
    const float4             folded_lvec2d,    // 17  (ax, bx, ay, by)
    const float              dt,               // 18
    const float4             md_params,        // 19  (lin_damp, ang_damp, force_scale, torque_scale)
    const int                niter             // 20
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 LBASIS[FOLDED_BASIS_MAX_RIGID];
    __local float  LCOEFFS[FOLDED_TYPES_MAX_RIGID * FOLDED_BASIS_MAX_RIGID];
    const int ia0 = mols[gid];
    const int na  = mols[gid+1]-ia0;
    const int nbasis = folded_meta.x;
    const int factorized = (folded_meta.y < 0);
    const int ntypes = factorized ? 1 : folded_meta.y;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[gid];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;

    if (lid == 0) {
        pos      = poss[gid];
        qrot     = resume_fire ? qrots[gid] : normalize(qrots[gid]);
        vpos     = vposs[gid];
        vrot     = vrots[gid];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[gid].a;
        Iinv_body.b = I_body_inv[gid].b;
        Iinv_body.c = I_body_inv[gid].c;
        Ibody.a     = I_body[gid].a;
        Ibody.b     = I_body[gid].b;
        Ibody.c     = I_body[gid].c;
    }
    // Cooperative load of basis params and coefficients into local memory
    for (int j = lid; j < nbasis; j += lsize) {
        LBASIS[j] = folded_kxyz[j];
    }
    const int ncoeff = factorized ? nbasis*4 : nbasis*ntypes;
    for (int j = lid; j < ncoeff; j += lsize) {
        LCOEFFS[j] = folded_coeffs[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Precompute inverse 2D lattice matrix.
    // Surface lattice: a = (ax, ay), b = (bx, by)  [stored as (ax, bx, ay, by)]
    // Inverse:  [du/dx  du/dy] = [ by/det  -bx/det]
    //           [dv/dx  dv/dy]   [-ay/det   ax/det]
    // invLvec2d = (du/dx, du/dy, dv/dx, dv/dy)
    float ax = folded_lvec2d.x;
    float bx = folded_lvec2d.y;
    float ay = folded_lvec2d.z;
    float by = folded_lvec2d.w;
    float det = ax*by - bx*ay;
    float4 invLvec2d = (float4)( by/det, -bx/det, -ay/det, ax/det );

    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            const float3 p_world = pos.xyz + r_world;
            // Compute fractional coordinates (u,v) from world (x,y):
            //   u = (du/dx)·x + (du/dy)·y
            //   v = (dv/dx)·x + (dv/dy)·y
            // Then wrap to [0,1) for periodicity.
            float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
            float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
            u = u - floor(u);
            v = v - floor(v);
            // Evaluate folded basis potential and gradient for this atom's type
            float E = 0.0f;
            float3 f = (float3)(0.0f, 0.0f, 0.0f);
            const int ityp = factorized ? 0 : folded_atom_type[atom_idx];
            const float4 plqh = factorized ? ((__global const float4*)folded_atom_type)[atom_idx] : (float4)(0.0f);
            if (factorized || (ityp >= 0 && ityp < ntypes)) {
                for (int ib = 0; ib < nbasis; ib++) {
                    float c = folded_coeff_rigid(LCOEFFS, ib, nbasis, ityp, plqh, factorized);
                    float4 prm = LBASIS[ib];
                    float  b = folded_eval_basis_rigid(u, v, p_world.z, prm);
                    float3 g = folded_eval_grad_rigid (u, v, p_world.z, prm, invLvec2d);
                    E += c * b;
                    f -= c * g;
                }
            }
            // Anchor spring (mouse picking): F = -k·(p - anchor), E = ½·k·|p-anchor|²
            // k = anchors[ia].w; k > 0 means active, k < 0 means no anchor.
            // Spring energy is included in E for diagnostics/conservation checks.
            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            const float force_scale  = use_fire ? 1.0f : md_params.z;
            const float torque_scale = use_fire ? 1.0f : md_params.w;
            const float3 f       = Lforce[0].xyz * force_scale;
            const float3 tq_world= Ltorq[0].xyz  * torque_scale;
            body_force [gid] = (float4)(f, 0.0f);
            body_torque[gid] = (float4)(tq_world, 0.0f);
            const float3 tq_body    = mat3_dot_T(R, tq_world);
            const float3 L_body      = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro        = cross(vrot.xyz, L_body);
            const float3 alpha_body  = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // Write back final state and world positions
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
    if (lid == 0) {
        if (use_fire) fire_state[gid] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
        poss [gid] = pos;
        qrots[gid] = qrot;
        vposs[gid] = vpos;
        vrots[gid] = vrot;
    }
}

// ======================================================================
//  Kernel: rigid_body_folded_replicas_kernel — MD / FIRE, many replicas
// ======================================================================
//
// ROLE: Same physics as rigid_body_folded_kernel (folded forces + Euler/FIRE)
//       but for N copies of the SAME molecule (e.g. AFM scan pixels).
//
// PARALLELISM:
//   1 thread = 1 replica (global_id)
//   workgroup size REPLICAS_WG = 128
//   Cooperatively load shared apos_body / atom_type / basis / I into __local
//   once, then each thread runs its own MD loop with NO barriers.
//
// Use this for high-throughput damped/FIRE dynamics. For pose optimization
// prefer rigid_body_folded_newton_replicas_kernel.
//
// ======================================================================
#ifndef REPLICAS_WG
#define REPLICAS_WG 128
#endif

__kernel void rigid_body_folded_replicas_kernel(
    __global       float4*   poss,             // 1  [n_replicas]
    __global       float4*   qrots,            // 2  [n_replicas]
    __global       float4*   vposs,            // 3  [n_replicas]
    __global       float4*   vrots,            // 4  [n_replicas]
    __global       float4*   fire_state,       // persistent (dt_lin, dt_ang, damp_lin, damp_ang)
    __global const cl_Mat3*  I_body_inv,       // 5  [1] shared
    __global const cl_Mat3*  I_body,           // 6  [1] shared
    __global const float4*   apos_body,        // 7  [na] shared
    __global       float4*   apos_world,       // 8  [n_replicas * na]
    __global       float4*   atom_force,       // 9  [n_replicas * na]
    __global       float4*   body_force,       // 10 [n_replicas]
    __global       float4*   body_torque,      // 11 [n_replicas]
    __global const float4*   anchors,          // 12 [n_replicas * na]
    __global const float*    folded_coeffs,    // 13 [ntypes * nbasis]
    __global const float4*   folded_kxyz,      // 14 [nbasis]
    __global const int*      folded_atom_type, // 15 [na] shared
    const int4               folded_meta,      // 16 (nbasis, ntypes, na, n_replicas)
    const float4             folded_lvec2d,    // 17 (ax, bx, ay, by)
    const float              dt,               // 18
    const float4             md_params,        // 19 (lin_damp, ang_damp, force_scale, torque_scale)
    const int                niter             // 20
) {
    const int tid = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int nbasis = folded_meta.x;
    const int factorized = (folded_meta.y < 0);
    const int ntypes = factorized ? 1 : folded_meta.y;
    const int na     = folded_meta.z;
    const int n_rep  = folded_meta.w;

    // ---- Local memory: shared molecule data (loaded once) ----
    __local float4 s_apos_body[MAX_ATOMS_PER_BODY];
    __local int    s_atom_type[MAX_ATOMS_PER_BODY];
    __local float4 s_atom_plqh[MAX_ATOMS_PER_BODY];
    __local cl_Mat3 s_Iinv;
    __local cl_Mat3 s_Ibody;
    __local float4 LBASIS[FOLDED_BASIS_MAX_RIGID];
    __local float  LCOEFFS[FOLDED_TYPES_MAX_RIGID * FOLDED_BASIS_MAX_RIGID];

    // ---- Cooperative load of shared data ----
    for (int i = lid; i < na;     i += lsize) s_apos_body[i]  = apos_body[i];
    for (int i = lid; i < na;     i += lsize) {
        if (factorized) s_atom_plqh[i] = ((__global const float4*)folded_atom_type)[i];
        else            s_atom_type[i] = folded_atom_type[i];
    }
    for (int j = lid; j < nbasis; j += lsize) LBASIS[j]       = folded_kxyz[j];
    const int ncoeff = factorized ? nbasis*4 : nbasis*ntypes;
    for (int j = lid; j < ncoeff; j += lsize) LCOEFFS[j] = folded_coeffs[j];
    if (lid == 0) {
        s_Iinv.a  = I_body_inv[0].a;  s_Iinv.b  = I_body_inv[0].b;  s_Iinv.c  = I_body_inv[0].c;
        s_Ibody.a = I_body[0].a;      s_Ibody.b = I_body[0].b;      s_Ibody.c = I_body[0].c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Extra threads (n_rep not multiple of REPLICAS_WG) exit after load
    if (tid >= n_rep) return;

    // ---- Per-thread state (registers) ----
    float4 pos  = poss[tid];
    const float4 fstate = fire_state[tid];
    const int resume_fire = (md_params.w < 0.0f) && (fstate.x > 0.0f);
    float4 qrot = resume_fire ? qrots[tid] : normalize(qrots[tid]);
    float4 vpos = vposs[tid];
    float4 vrot = vrots[tid];
    float  inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;

    // Cache inertia tensors in registers (read from local once)
    float3 I_a  = s_Ibody.a.xyz,  I_b  = s_Ibody.b.xyz,  I_c  = s_Ibody.c.xyz;
    float3 Ii_a = s_Iinv.a.xyz,   Ii_b = s_Iinv.b.xyz,   Ii_c = s_Iinv.c.xyz;

    // Precompute inverse 2D lattice
    float ax = folded_lvec2d.x, bx = folded_lvec2d.y;
    float ay = folded_lvec2d.z, by = folded_lvec2d.w;
    float det = ax*by - bx*ay;
    float4 invLvec2d = (float4)( by/det, -bx/det, -ay/det, ax/det );

    const float lin_damp0   = md_params.x;
    const float ang_damp0   = md_params.y;
    const int   use_fire    = (md_params.w < 0.0f);
    const float force_scale = use_fire ? 1.0f : md_params.z;
    const float torque_scale= use_fire ? 1.0f : md_params.w;
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : lin_damp0, damp_ang = resume_fire ? fstate.w : ang_damp0;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;

    // ---- Main iteration loop (NO barriers!) ----
    // body_torque.w ← steps actually used (FIRE early-exit when both |F|² and |τ|² are below RIGID_FIRE_F2SAFE)
    // vposs.w ← max |F| seen over the trajectory (crash / contact diagnostic)
    int steps_used = niter;
    float Fmax = use_fire ? vpos.w : 0.0f;
    for (int step = 0; step < niter; ++step) {
        float3 Ra = quat_to_a(qrot);
        float3 Rb = quat_to_b(qrot);
        float3 Rc = quat_to_c(qrot);

        float3 total_force  = (float3)(0.0f);
        float3 total_torque = (float3)(0.0f);

        for (int ia = 0; ia < na; ia++) {
            float4 p_body = s_apos_body[ia];
            float3 r_world = (float3)(dot(Ra, p_body.xyz), dot(Rb, p_body.xyz), dot(Rc, p_body.xyz));
            float3 p_world = pos.xyz + r_world;

            float u = invLvec2d.x * p_world.x + invLvec2d.y * p_world.y;
            float v = invLvec2d.z * p_world.x + invLvec2d.w * p_world.y;
            u = u - floor(u);
            v = v - floor(v);

            float3 f = (float3)(0.0f);
            int ityp = factorized ? 0 : s_atom_type[ia];
            float4 plqh = factorized ? s_atom_plqh[ia] : (float4)(0.0f);
            if (factorized || (ityp >= 0 && ityp < ntypes)) {
                for (int ib = 0; ib < nbasis; ib++) {
                    float c = folded_coeff_rigid(LCOEFFS, ib, nbasis, ityp, plqh, factorized);
                    float4 prm = LBASIS[ib];
                    float3 g = folded_eval_grad_rigid(u, v, p_world.z, prm, invLvec2d);
                    f -= c * g;
                }
            }

            float4 anchor = anchors[tid * na + ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
            }

            total_force  += f;
            total_torque += cross(r_world, f);
        }

        // Integration (symplectic Euler / FIRE, all in registers)
        float3 f        = total_force * force_scale;
        float3 tq_world = total_torque * torque_scale;
        float3 tq_body  = Ra * tq_world.x + Rb * tq_world.y + Rc * tq_world.z;
        Fmax = fmax(Fmax, length(total_force));
        if (use_fire && (dot(f, f) < RIGID_FIRE_F2SAFE) && (dot(tq_body, tq_body) < RIGID_FIRE_F2SAFE)) {
            steps_used = step;
            break;
        }
        float3 L_body   = (float3)(dot(I_a, vrot.xyz), dot(I_b, vrot.xyz), dot(I_c, vrot.xyz));
        float3 gyro     = cross(vrot.xyz, L_body);
        float3 rhs      = tq_body - gyro;
        float3 alpha    = (float3)(dot(Ii_a, rhs), dot(Ii_b, rhs), dot(Ii_c, rhs));

        if (use_fire) {
            vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, lin_damp0);
            vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, ang_damp0);
        } else {
            vpos.xyz *= damp_lin;
            vrot.xyz *= damp_ang;
        }
        const float dtl = use_fire ? dt_lin : dt;
        const float dta = use_fire ? dt_ang : dt;
        vpos.xyz += f * (dtl * inv_mass);
        vrot.xyz += alpha * dta;
        pos.xyz  += vpos.xyz * dtl;
        qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
    }

    // ---- Write final state ----
    if (use_fire) fire_state[tid] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
    poss [tid] = pos;
    qrots[tid] = qrot;
    vposs[tid] = (float4)(vpos.xyz, Fmax);  // .w = peak |F| over trajectory
    vrots[tid] = vrot;

    // ---- Recompute forces at final position for diagnostics ----
    float3 Ra = quat_to_a(qrot);
    float3 Rb = quat_to_b(qrot);
    float3 Rc = quat_to_c(qrot);
    float3 final_force  = (float3)(0.0f);
    float3 final_torque = (float3)(0.0f);
    float  E_tot = 0.0f;

    for (int ia = 0; ia < na; ia++) {
        float4 p_body = s_apos_body[ia];
        float3 r_world = (float3)(dot(Ra, p_body.xyz), dot(Rb, p_body.xyz), dot(Rc, p_body.xyz));
        float3 p_world = pos.xyz + r_world;
        float u = invLvec2d.x * p_world.x + invLvec2d.y * p_world.y;
        float v = invLvec2d.z * p_world.x + invLvec2d.w * p_world.y;
        u = u - floor(u);
        v = v - floor(v);

        float E = 0.0f;
        float3 f = (float3)(0.0f);
        int ityp = factorized ? 0 : s_atom_type[ia];
        float4 plqh = factorized ? s_atom_plqh[ia] : (float4)(0.0f);
        if (factorized || (ityp >= 0 && ityp < ntypes)) {
            for (int ib = 0; ib < nbasis; ib++) {
                float c = folded_coeff_rigid(LCOEFFS, ib, nbasis, ityp, plqh, factorized);
                float4 prm = LBASIS[ib];
                float  b = folded_eval_basis_rigid(u, v, p_world.z, prm);
                float3 g = folded_eval_grad_rigid (u, v, p_world.z, prm, invLvec2d);
                E += c * b;
                f -= c * g;
            }
        }
        float4 anchor = anchors[tid * na + ia];
        if (anchor.w > 0.0f) {
            float3 d = p_world - anchor.xyz;
            f += d * -anchor.w;
            E += 0.5f * anchor.w * dot(d, d);
        }
        final_force  += f;
        final_torque += cross(r_world, f);
        E_tot += E;
        apos_world[tid * na + ia] = (float4)(p_world, E);
        atom_force [tid * na + ia] = (float4)(f, E);
    }
    body_force [tid] = (float4)(final_force * force_scale, E_tot);
    body_torque[tid] = (float4)(final_torque * torque_scale, (float)steps_used);
}
// ======================================================================
//  Pure Newton / trust-region helpers + kernels (GPU, no Python round-trips)
// ======================================================================
//
// NOT eigendecomposition / Jacobi. Dense 6×6 Gaussian elimination + LM trust.
// See file-top section "How the 6×6 Newton step is solved".
//
// Helpers used by both Newton kernels:
//   rigid_quat_apply_body_dtheta — right-multiply body-frame Δθ onto quaternion
//   rigid_solve6_lm             — (H+λI)Δ = G via GE + partial pivoting
//   folded_FT_replica           — full force/torque/energy at one pose (replicas)
//   folded_FT_perturb           — same after ±eps along one of 6 coords (FD column)

// Apply body-frame angular increment Δθ to unit quaternion q (right multiply).
//   q ← normalize( q ⊗ exp(Δθ) )   with Taylor make_qrot_taylor
inline void rigid_quat_apply_body_dtheta(float4* q, float3 dth){
    float4 dq = make_qrot_taylor(dth);
    *q = normalize(quat_mult(*q, dq));
}

// Solve (A + λ I) x = b for dense row-major 6×6 A.
// Algorithm: copy A→M, add λ on diagonal, Gaussian elimination with partial
// pivoting, back-substitution. Returns 1 on success, 0 if pivot < 1e-20
// (singular / near-singular → caller increases λ).
//
// This is NOT Jacobi diagonalization and is NOT parallelized across the WG:
// 6³ flops is cheaper than a barrier. Parallelism lives in force eval / replicas.
inline int rigid_solve6_lm(float A[36], const float b[6], float lam, float x[6]){
    float M[36];
    float rhs[6];
    for (int i = 0; i < 36; i++) M[i] = A[i];
    for (int i = 0; i < 6; i++) { M[i*6+i] += lam; rhs[i] = b[i]; }
    for (int k = 0; k < 6; k++) {
        int piv = k;
        float best = fabs(M[k*6+k]);
        for (int i = k+1; i < 6; i++) {
            float v = fabs(M[i*6+k]);
            if (v > best) { best = v; piv = i; }
        }
        if (best < 1e-20f) return 0;
        if (piv != k) {
            for (int j = k; j < 6; j++) {
                float tmp = M[k*6+j]; M[k*6+j] = M[piv*6+j]; M[piv*6+j] = tmp;
            }
            float tb = rhs[k]; rhs[k] = rhs[piv]; rhs[piv] = tb;
        }
        float diag = M[k*6+k];
        for (int i = k+1; i < 6; i++) {
            float f = M[i*6+k] / diag;
            M[i*6+k] = 0.0f;
            for (int j = k+1; j < 6; j++) M[i*6+j] -= f * M[k*6+j];
            rhs[i] -= f * rhs[k];
        }
    }
    for (int i = 5; i >= 0; i--) {
        float s = rhs[i];
        for (int j = i+1; j < 6; j++) s -= M[i*6+j] * x[j];
        x[i] = s / M[i*6+i];
    }
    return 1;
}

// Force + torque + energy for ONE replica pose on the folded substrate.
//
// WHAT IT DOES:
//   For each atom: world pos = CoM + R(q)·r_body; map (x,y)→fractional (u,v);
//   sum over basis: E += c·B,  F_atom += −c·∇B; optional harmonic anchor.
//   Accumulates F_world and τ_world = Σ r×F, then returns τ_BODY = Rᵀ τ_world.
//
// WHY: Newton / FD Hessian need many F,τ evaluations at nearby poses. This is
// the shared SSOT force eval for the replicas Newton path (serial over atoms
// inside one thread — molecules are small, e.g. 3–5 atoms).
//
// Inputs from __local: s_apos_body, s_atom_type, LBASIS, LCOEFFS (shared mol).
// anchors indexed as anchors[tid*na + ia] (per-replica picking springs).
// Outputs: *F_out (world), *Tb_out (body), *E_out (scalar energy).
inline void folded_FT_replica(
    float3 pos, float4 qrot,
    int na, int nbasis, int ntypes, int tid,
    __local const float4* s_apos_body,
    __local const int*    s_atom_type,
    __local const float4* LBASIS,
    __local const float*  LCOEFFS,
    __global const float4* anchors,
    float4 invLvec2d,
    float3* F_out, float3* Tb_out, float* E_out
){
    float3 Ra = quat_to_a(qrot);
    float3 Rb = quat_to_b(qrot);
    float3 Rc = quat_to_c(qrot);
    float3 F = (float3)(0.0f);
    float3 Tw = (float3)(0.0f);
    float E = 0.0f;
    for (int ia = 0; ia < na; ia++) {
        float4 p_body = s_apos_body[ia];
        float3 r_world = (float3)(dot(Ra, p_body.xyz), dot(Rb, p_body.xyz), dot(Rc, p_body.xyz));
        float3 p_world = pos + r_world;
        float u = invLvec2d.x * p_world.x + invLvec2d.y * p_world.y;
        float v = invLvec2d.z * p_world.x + invLvec2d.w * p_world.y;
        u = u - floor(u); v = v - floor(v);
        float3 f = (float3)(0.0f);
        float Ea = 0.0f;
        int ityp = s_atom_type[ia];
        if (ityp >= 0 && ityp < ntypes) {
            int ioff = ityp * nbasis;
            for (int ib = 0; ib < nbasis; ib++) {
                float c = LCOEFFS[ioff + ib];
                float4 prm = LBASIS[ib];
                float  b = folded_eval_basis_rigid(u, v, p_world.z, prm);
                float3 g = folded_eval_grad_rigid(u, v, p_world.z, prm, invLvec2d);
                Ea += c * b;
                f  -= c * g;
            }
        }
        float4 anchor = anchors[tid * na + ia];
        if (anchor.w > 0.0f) {
            float3 d = p_world - anchor.xyz;
            f += d * -anchor.w;
            Ea += 0.5f * anchor.w * dot(d, d);
        }
        F  += f;
        Tw += cross(r_world, f);
        E  += Ea;
    }
    *F_out  = F;
    *Tb_out = Ra * Tw.x + Rb * Tw.y + Rc * Tw.z; // R^T @ Tw
    *E_out  = E;
}

// Finite-difference probe: evaluate F,τ,E at pose perturbed along one of 6 DOFs.
//
// Coordinates u = (Δx, Δy, Δz, Δθx, Δθy, Δθz)_body:
//   ipert 0..2 → translate by eps_t along x/y/z
//   ipert 3..5 → rotate by eps_r about body x/y/z (via rigid_quat_apply_body_dtheta)
// Then calls folded_FT_replica. The replicas imaging kernel calls this at ±eps
// to build centered-difference Hessian columns.
inline void folded_FT_perturb(
    float3 pos0, float4 q0, int ipert, float eps_t, float eps_r,
    int na, int nbasis, int ntypes, int tid,
    __local const float4* s_apos_body, __local const int* s_atom_type,
    __local const float4* LBASIS, __local const float* LCOEFFS,
    __global const float4* anchors, float4 invLvec2d,
    float3* F, float3* Tb, float* E
){
    float3 pos = pos0;
    float4 q = q0;
    float3 dth = (float3)(0.0f);
    if (ipert < 3) {
        if (ipert == 0) pos.x += eps_t;
        else if (ipert == 1) pos.y += eps_t;
        else pos.z += eps_t;
    } else {
        if (ipert == 3) dth.x = eps_r;
        else if (ipert == 4) dth.y = eps_r;
        else dth.z = eps_r;
        rigid_quat_apply_body_dtheta(&q, dth);
    }
    folded_FT_replica(pos, q, na, nbasis, ntypes, tid, s_apos_body, s_atom_type,
                      LBASIS, LCOEFFS, anchors, invLvec2d, F, Tb, E);
}

// ======================================================================
//  Kernel: rigid_body_folded_newton_replicas_kernel
// ======================================================================
//
// ROLE: Pure Newton / Levenberg–Marquardt pose optimizer for MANY replicas
//       of the same molecule (AFM scan pixels). NO velocities / FIRE / MD.
//
// PARALLELISM:
//   1 thread = 1 replica; workgroup REPLICAS_WG=128
//   Hessian 6×6 in PRIVATE memory; each thread serial FD + rigid_solve6_lm
//   Shared molecule/basis loaded once to __local
//
// vs rigid_body_folded_replicas_kernel: that one does MD/FIRE with velocities;
// this one only updates (pos, qrot) when a Newton step lowers energy.
//
// vs rigid_body_folded_newton_kernel: same Newton math, but one body uses a
// 32-thread WG with atom-parallel force eval + H in __local.
//
__kernel
void rigid_body_folded_newton_replicas_kernel(
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   newton_state,     // persistent (trust, lambda, recovery_used, Fmax)
    __global const cl_Mat3*  I_body_inv,   // unused (pose optimizer), kept for ABI symmetry
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,    // (nbasis, ntypes, na, n_replicas)
    const float4             folded_lvec2d,
    const float4             newton_params,  // (eps_t, eps_r, trust0, lambda_min=lambda_initial)
    const float              f2tol,          // separate |F|² and |τ|² stop threshold
    const int                niter
) {
    const int tid = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int nbasis = folded_meta.x;
    const int ntypes = folded_meta.y;
    const int na     = folded_meta.z;
    const int n_rep  = folded_meta.w;

    __local float4 s_apos_body[MAX_ATOMS_PER_BODY];
    __local int    s_atom_type[MAX_ATOMS_PER_BODY];
    __local float4 LBASIS[FOLDED_BASIS_MAX_RIGID];
    __local float  LCOEFFS[FOLDED_TYPES_MAX_RIGID * FOLDED_BASIS_MAX_RIGID];

    for (int i = lid; i < na; i += lsize) s_apos_body[i] = apos_body[i];
    for (int i = lid; i < na; i += lsize) s_atom_type[i] = folded_atom_type[i];
    for (int j = lid; j < nbasis; j += lsize) LBASIS[j] = folded_kxyz[j];
    for (int j = lid; j < nbasis * ntypes; j += lsize) LCOEFFS[j] = folded_coeffs[j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid >= n_rep) return;

    const float4 nstate = newton_state[tid];
    const int resume_newton = (nstate.x > 0.0f);
    float3 pos = poss[tid].xyz;
    float  mass_w = poss[tid].w;
    float4 qrot = resume_newton ? qrots[tid] : normalize(qrots[tid]);
    float4 invLvec2d;
    {
        float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
        float det = ax*by - bx*ay;
        invLvec2d = (float4)(by/det, -bx/det, -ay/det, ax/det);
    }
    const float eps_t = newton_params.x;
    const float eps_r = newton_params.y;
    float trust = resume_newton ? nstate.x : newton_params.z;
    float lam   = resume_newton ? nstate.y : newton_params.w;
    const float eps6[6] = {eps_t, eps_t, eps_t, eps_r, eps_r, eps_r};

    float3 F, Tb; float E;
    folded_FT_replica(pos, qrot, na, nbasis, ntypes, tid, s_apos_body, s_atom_type,
                      LBASIS, LCOEFFS, anchors, invLvec2d, &F, &Tb, &E);

    int iters_used = 0;
    int recovery_used = resume_newton ? (nstate.z > 0.5f) : 0;
    float Fmax = resume_newton ? fmax(nstate.w, length(F)) : length(F);
    for (int it = 0; it < niter; ++it) {
        if ((dot(F,F) < f2tol) && (dot(Tb,Tb) < f2tol)) break;
        iters_used = it + 1;
        Fmax = fmax(Fmax, length(F));

        // Forward-FD Hessian in private memory (36 floats)
        float H[36];
        float G0[6] = {F.x, F.y, F.z, Tb.x, Tb.y, Tb.z};
        const float g2_old = dot(F,F) + dot(Tb,Tb);
        for (int j = 0; j < 6; j++) {
            float3 Fp, Tbp, Fm, Tbm; float Ep, Em;
            folded_FT_perturb(pos, qrot, j, eps_t, eps_r, na, nbasis, ntypes, tid,
                              s_apos_body, s_atom_type, LBASIS, LCOEFFS, anchors, invLvec2d,
                              &Fp, &Tbp, &Ep);
            folded_FT_perturb(pos, qrot, j, -eps_t, -eps_r, na, nbasis, ntypes, tid,
                              s_apos_body, s_atom_type, LBASIS, LCOEFFS, anchors, invLvec2d,
                              &Fm, &Tbm, &Em);
            float Gp[6] = {Fp.x, Fp.y, Fp.z, Tbp.x, Tbp.y, Tbp.z};
            float Gm[6] = {Fm.x, Fm.y, Fm.z, Tbm.x, Tbm.y, Tbm.z};
            float inv_2e = 0.5f / eps6[j];
            for (int i = 0; i < 6; i++) {
                H[i*6+j] = -(Gp[i] - Gm[i]) * inv_2e; // H=-dG/du, centered FD
            }
        }
        // Symmetrize
        for (int i = 0; i < 6; i++) {
            for (int j = i+1; j < 6; j++) {
                float s = 0.5f * (H[i*6+j] + H[j*6+i]);
                H[i*6+j] = s; H[j*6+i] = s;
            }
        }

        int accepted = 0;
        float3 pos_try = pos;
        float4 q_try = qrot;
        float3 F2 = F; float3 Tb2 = Tb; float E2 = E;
        for (int itry = 0; itry < 8; itry++) {
            float delta[6];
            if (!rigid_solve6_lm(H, G0, lam, delta)) { lam = fmax(lam * 10.0f, newton_params.w); continue; }
            float Gd = 0.0f;
            for (int i = 0; i < 6; i++) Gd += G0[i] * delta[i];
            if (Gd <= 0.0f) { lam = fmax(lam * 10.0f, newton_params.w); continue; }
            float nrm2 = 0.0f;
            for (int i = 0; i < 6; i++) nrm2 += delta[i]*delta[i];
            float nrm = sqrt(nrm2);
            if (nrm > trust && nrm > 1e-30f) {
                float s = trust / nrm;
                for (int i = 0; i < 6; i++) delta[i] *= s;
                nrm = trust;
            }
            pos_try = pos + (float3)(delta[0], delta[1], delta[2]);
            q_try = qrot;
            rigid_quat_apply_body_dtheta(&q_try, (float3)(delta[3], delta[4], delta[5]));
            folded_FT_replica(pos_try, q_try, na, nbasis, ntypes, tid, s_apos_body, s_atom_type,
                              LBASIS, LCOEFFS, anchors, invLvec2d, &F2, &Tb2, &E2);
            const float g2_try = dot(F2,F2) + dot(Tb2,Tb2);
            const float E_tol = 2e-7f * (1.0f + fabs(E));
            if ((E2 < E) || ((E2 <= E + E_tol) && (g2_try < 0.99f * g2_old))) {
                pos = pos_try; qrot = q_try; F = F2; Tb = Tb2; E = E2;
                accepted = 1;
                recovery_used = 0;
                Fmax = fmax(Fmax, length(F));
                // A successful boundary-limited step is evidence that the trust
                // radius is too small.  Always relax LM damping after acceptance;
                // otherwise one rejected trial can leave all later steps overdamped.
                if (nrm > 0.8f * trust) {
                    trust = fmin(trust * 2.0f, newton_params.z);
                }
                lam = fmax(lam * 0.3f, newton_params.w);
                break;
            }
            trust = fmax(trust * 0.5f, 1e-4f);
            lam = fmin(fmax(lam * 5.0f, newton_params.w), 1e4f);
        }
        if (!accepted) {
            if (recovery_used) break;
            trust = newton_params.z;
            lam = newton_params.w;
            recovery_used = 1;
        }
    }

    // Write state + diagnostic forces (velocities zeroed — optimizer, not MD)
    // vposs.w = peak |F| over Newton trajectory (contact / crash diagnostic)
    newton_state[tid] = (float4)(trust, lam, (float)recovery_used, Fmax);
    poss[tid]  = (float4)(pos, mass_w);
    qrots[tid] = qrot;
    vposs[tid] = (float4)(0.0f, 0.0f, 0.0f, Fmax);
    vrots[tid] = (float4)(0.0f);

    float3 Ra = quat_to_a(qrot), Rb = quat_to_b(qrot), Rc = quat_to_c(qrot);
    float3 Tw = Ra * Tb.x + Rb * Tb.y + Rc * Tb.z; // body→world for output convention
    // rebuild world atoms / per-atom forces
    float E_tot = 0.0f;
    float3 Fsum = (float3)(0.0f);
    float3 Twsum = (float3)(0.0f);
    for (int ia = 0; ia < na; ia++) {
        float4 p_body = s_apos_body[ia];
        float3 r_world = (float3)(dot(Ra, p_body.xyz), dot(Rb, p_body.xyz), dot(Rc, p_body.xyz));
        float3 p_world = pos + r_world;
        float u = invLvec2d.x * p_world.x + invLvec2d.y * p_world.y;
        float v = invLvec2d.z * p_world.x + invLvec2d.w * p_world.y;
        u = u - floor(u); v = v - floor(v);
        float Ea = 0.0f; float3 f = (float3)(0.0f);
        int ityp = s_atom_type[ia];
        if (ityp >= 0 && ityp < ntypes) {
            int ioff = ityp * nbasis;
            for (int ib = 0; ib < nbasis; ib++) {
                float c = LCOEFFS[ioff + ib];
                float4 prm = LBASIS[ib];
                Ea += c * folded_eval_basis_rigid(u, v, p_world.z, prm);
                f  -= c * folded_eval_grad_rigid(u, v, p_world.z, prm, invLvec2d);
            }
        }
        float4 anchor = anchors[tid * na + ia];
        if (anchor.w > 0.0f) {
            float3 d = p_world - anchor.xyz;
            f += d * -anchor.w;
            Ea += 0.5f * anchor.w * dot(d, d);
        }
        Fsum += f; Twsum += cross(r_world, f); E_tot += Ea;
        apos_world[tid*na+ia] = (float4)(p_world, Ea);
        atom_force [tid*na+ia] = (float4)(f, Ea);
    }
    body_force [tid] = (float4)(Fsum, E_tot);
    body_torque[tid] = (float4)(Twsum, (float)iters_used);  // .w = Newton iters used
}

// ======================================================================
//  Kernel: rigid_body_folded_newton_kernel
// ======================================================================
//
// ROLE: Pure Newton / LM trust-region for ONE rigid body on folded substrate.
//       Same algorithm as newton_replicas, different parallelism layout.
//
// PARALLELISM:
//   1 workgroup = 1 body; WORKGROUP_SIZE=32
//   Atoms split across lids for each F/τ eval (tree reduce → __local)
//   Hessian 6×6 in __local LH[36]; only lid==0 builds columns + rigid_solve6_lm
//   Other lids idle during the tiny 6×6 solve (intentional — see file header)
//
// vs rigid_body_folded_kernel: MD/FIRE integrator vs pose Newton optimizer.
//
__kernel
void rigid_body_folded_newton_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   newton_state,     // persistent (trust, lambda, recovery_used, Fmax)
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float4             newton_params,  // (eps_t, eps_r, trust0, lambda_min=lambda_initial)
    const float              f2tol,
    const int                niter
) {
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    const int ia0 = mols[gid];
    const int na  = mols[gid+1] - ia0;
    const int nbasis = folded_meta.x;
    const int ntypes = folded_meta.y;

    __local float4 pos;
    __local float4 qrot;
    __local float4 LBASIS[FOLDED_BASIS_MAX_RIGID];
    __local float  LCOEFFS[FOLDED_TYPES_MAX_RIGID * FOLDED_BASIS_MAX_RIGID];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Ltorq[WORKGROUP_SIZE];
    __local float  LE[WORKGROUP_SIZE];
    __local float  LH[36];
    __local float  LG0[6];
    __local float  Ldelta[6];
    __local int    Laccept, Lrecovery;
    __local float  Ltrust, Llam, LE0;
    __local float  LFmax;
    __local float3 LF, LTb;
    __local float4 invLvec2d;
    __local float4 pos0, q0;

    if (lid == 0) {
        float4 nstate = newton_state[gid];
        pos  = poss[gid];
        qrot = (nstate.x > 0.0f) ? qrots[gid] : normalize(qrots[gid]);
        float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
        float det = ax*by - bx*ay;
        invLvec2d = (float4)(by/det, -bx/det, -ay/det, ax/det);
        Ltrust = (nstate.x > 0.0f) ? nstate.x : newton_params.z;
        Llam   = (nstate.x > 0.0f) ? nstate.y : newton_params.w;
        Lrecovery = (nstate.x > 0.0f) ? (nstate.z > 0.5f) : 0;
        LFmax = (nstate.x > 0.0f) ? nstate.w : 0.0f;
    }
    for (int j = lid; j < nbasis; j += lsize) LBASIS[j] = folded_kxyz[j];
    for (int j = lid; j < nbasis*ntypes; j += lsize) LCOEFFS[j] = folded_coeffs[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    const float eps_t = newton_params.x;
    const float eps_r = newton_params.y;

    for (int it = 0; it < niter; ++it) {
        // --- Force reduction at current pose ---
        float3 Ra = quat_to_a(qrot), Rb = quat_to_b(qrot), Rc = quat_to_c(qrot);
        float4 tf = (float4)(0.0f), tt = (float4)(0.0f); float eacc = 0.0f;
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            int ia = ia0 + atom_idx;
            float4 p_body = apos_body[ia];
            float3 r_world = (float3)(dot(Ra,p_body.xyz), dot(Rb,p_body.xyz), dot(Rc,p_body.xyz));
            float3 p_world = pos.xyz + r_world;
            float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
            float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
            u -= floor(u); v -= floor(v);
            float3 f = (float3)(0.0f); float Ea = 0.0f;
            int ityp = folded_atom_type[atom_idx];
            if (ityp >= 0 && ityp < ntypes) {
                int ioff = ityp * nbasis;
                for (int ib = 0; ib < nbasis; ib++) {
                    float c = LCOEFFS[ioff+ib]; float4 prm = LBASIS[ib];
                    Ea += c * folded_eval_basis_rigid(u,v,p_world.z,prm);
                    f  -= c * folded_eval_grad_rigid(u,v,p_world.z,prm,invLvec2d);
                }
            }
            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w; Ea += 0.5f * anchor.w * dot(d,d);
            }
            tf.xyz += f; tt.xyz += cross(r_world, f); eacc += Ea;
            apos_world[ia] = (float4)(p_world, Ea);
            atom_force[ia] = (float4)(f, Ea);
        }
        Lforce[lid] = tf; Ltorq[lid] = tt; LE[lid] = eacc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE>>1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Lforce[lid] += Lforce[lid+stride];
                Ltorq[lid]  += Ltorq[lid+stride];
                LE[lid]     += LE[lid+stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            LF = Lforce[0].xyz;
            float3 Tw = Ltorq[0].xyz;
            cl_Mat3 R; R.a = (float4)(Ra,0.f); R.b = (float4)(Rb,0.f); R.c = (float4)(Rc,0.f);
            LFmax = fmax(LFmax, length(LF));
            LTb = mat3_dot_T(R, Tw);
            LE0 = LE[0];
            LG0[0]=LF.x; LG0[1]=LF.y; LG0[2]=LF.z; LG0[3]=LTb.x; LG0[4]=LTb.y; LG0[5]=LTb.z;
            body_force[gid]  = (float4)(LF, LE0);
            body_torque[gid] = (float4)(Tw, 0.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((dot(LF,LF) < f2tol) && (dot(LTb,LTb) < f2tol)) break;

        // --- FD columns: each column is a full WG force eval at perturbed pose ---
        // Store base pose
        if (lid == 0) { pos0 = pos; q0 = qrot; }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < 6; j++) {
            if (lid == 0) {
                pos = pos0; qrot = q0;
                if (j == 0) pos.x += eps_t;
                else if (j == 1) pos.y += eps_t;
                else if (j == 2) pos.z += eps_t;
                else if (j == 3) rigid_quat_apply_body_dtheta(&qrot, (float3)(eps_r,0,0));
                else if (j == 4) rigid_quat_apply_body_dtheta(&qrot, (float3)(0,eps_r,0));
                else             rigid_quat_apply_body_dtheta(&qrot, (float3)(0,0,eps_r));
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            Ra = quat_to_a(qrot); Rb = quat_to_b(qrot); Rc = quat_to_c(qrot);
            tf = (float4)(0.0f); tt = (float4)(0.0f);
            for (int i = 0; i < ATOMS_PER_THREAD; i++) {
                int atom_idx = lid + i*lsize;
                if (atom_idx >= na) break;
                int ia = ia0 + atom_idx;
                float4 p_body = apos_body[ia];
                float3 r_world = (float3)(dot(Ra,p_body.xyz), dot(Rb,p_body.xyz), dot(Rc,p_body.xyz));
                float3 p_world = pos.xyz + r_world;
                float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
                float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
                u -= floor(u); v -= floor(v);
                float3 f = (float3)(0.0f);
                int ityp = folded_atom_type[atom_idx];
                if (ityp >= 0 && ityp < ntypes) {
                    int ioff = ityp * nbasis;
                    for (int ib = 0; ib < nbasis; ib++) {
                        float c = LCOEFFS[ioff+ib];
                        f -= c * folded_eval_grad_rigid(u,v,p_world.z,LBASIS[ib],invLvec2d);
                    }
                }
                float4 anchor = anchors[ia];
                if (anchor.w > 0.0f) f += (p_world - anchor.xyz) * -anchor.w;
                tf.xyz += f; tt.xyz += cross(r_world, f);
            }
            Lforce[lid] = tf; Ltorq[lid] = tt;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = WORKGROUP_SIZE>>1; stride > 0; stride >>= 1) {
                if (lid < stride) { Lforce[lid]+=Lforce[lid+stride]; Ltorq[lid]+=Ltorq[lid+stride]; }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                float3 Fp = Lforce[0].xyz;
                cl_Mat3 R; R.a=(float4)(quat_to_a(qrot),0); R.b=(float4)(quat_to_b(qrot),0); R.c=(float4)(quat_to_c(qrot),0);
                float3 Tbp = mat3_dot_T(R, Ltorq[0].xyz);
                float Gj[6] = {Fp.x,Fp.y,Fp.z, Tbp.x,Tbp.y,Tbp.z};
                float eps = (j < 3) ? eps_t : eps_r;
                float inv_e = 1.0f / eps;
                for (int i = 0; i < 6; i++) LH[i*6+j] = -(Gj[i] - LG0[i]) * inv_e;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // restore base pose before step
        if (lid == 0) { pos = pos0; qrot = q0; }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            for (int i = 0; i < 6; i++) for (int j = i+1; j < 6; j++) {
                float s = 0.5f*(LH[i*6+j]+LH[j*6+i]); LH[i*6+j]=s; LH[j*6+i]=s;
            }
            Laccept = 0;
            float Hpriv[36];
            float Gpriv[6];
            for (int i = 0; i < 36; i++) Hpriv[i] = LH[i];
            for (int i = 0; i < 6; i++) Gpriv[i] = LG0[i];
            for (int itry = 0; itry < 8; itry++) {
                float delta[6];
                float Htry[36];
                for (int i = 0; i < 36; i++) Htry[i] = Hpriv[i];
                if (!rigid_solve6_lm(Htry, Gpriv, Llam, delta)) { Llam = fmax(Llam*10.f, newton_params.w); continue; }
                float Gd=0.0f;
                for (int i=0;i<6;i++) Gd+=Gpriv[i]*delta[i];
                if (Gd <= 0.0f) { Llam=fmax(Llam*10.f,newton_params.w); continue; }
                float nrm2=0; for (int i=0;i<6;i++) nrm2+=delta[i]*delta[i];
                float nrm=sqrt(nrm2);
                if (nrm > Ltrust && nrm > 1e-30f) {
                    float s = Ltrust/nrm; for (int i=0;i<6;i++) delta[i]*=s; nrm=Ltrust;
                }
                for (int i=0;i<6;i++) Ldelta[i]=delta[i];
                float3 pos_try = pos0.xyz + (float3)(delta[0],delta[1],delta[2]);
                float4 q_try = q0;
                rigid_quat_apply_body_dtheta(&q_try, (float3)(delta[3],delta[4],delta[5]));
                // Evaluate E at trial with lid0-only serial atom loop (small na)
                float Etry = 0.0f;
                float3 Rat=quat_to_a(q_try), Rbt=quat_to_b(q_try), Rct=quat_to_c(q_try);
                for (int ia = 0; ia < na; ia++) {
                    float4 p_body = apos_body[ia0+ia];
                    float3 r_world = (float3)(dot(Rat,p_body.xyz),dot(Rbt,p_body.xyz),dot(Rct,p_body.xyz));
                    float3 p_world = pos_try + r_world;
                    float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
                    float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
                    u-=floor(u); v-=floor(v);
                    int ityp = folded_atom_type[ia];
                    if (ityp>=0 && ityp<ntypes) {
                        int ioff=ityp*nbasis;
                        for (int ib=0; ib<nbasis; ib++)
                            Etry += LCOEFFS[ioff+ib]*folded_eval_basis_rigid(u,v,p_world.z,LBASIS[ib]);
                    }
                    float4 anchor = anchors[ia0+ia];
                    if (anchor.w>0.0f) {
                        float3 d=p_world-anchor.xyz; Etry += 0.5f*anchor.w*dot(d,d);
                    }
                }
                if (Etry < LE0 - 1e-12f) {
                    pos.xyz = pos_try; qrot = q_try;
                    Laccept = 1;
                    Lrecovery = 0;
                    // Grow after a successful boundary step; small unconstrained
                    // steps do not need a larger trust radius.
                    if (nrm > 0.8f*Ltrust) Ltrust=fmin(Ltrust*2.f,newton_params.z);
                    Llam=fmax(Llam*0.3f,newton_params.w);
                    break;
                }
                Ltrust = fmax(Ltrust*0.5f, 1e-4f);
                Llam = fmin(fmax(Llam*5.f, newton_params.w), 1e4f);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (!Laccept) {
            if (lid == 0) {
                if (Lrecovery) Laccept = -1;
                else { Ltrust = newton_params.z; Llam = newton_params.w; Lrecovery = 1; }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (Laccept < 0) break;
        }
    }

    // Final writeback
    if (lid == 0) {
        newton_state[gid] = (float4)(Ltrust, Llam, (float)Lrecovery, LFmax);
        poss[gid] = pos;
        qrots[gid] = qrot;
        vposs[gid] = (float4)(0.0f, 0.0f, 0.0f, LFmax);
        vrots[gid] = (float4)(0.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float3 Ra = quat_to_a(qrot), Rb = quat_to_b(qrot), Rc = quat_to_c(qrot);
    float4 tf=(float4)(0.0f), tt=(float4)(0.0f); float eacc=0.0f;
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        int ia = ia0 + atom_idx;
        float4 p_body = apos_body[ia];
        float3 r_world = (float3)(dot(Ra,p_body.xyz),dot(Rb,p_body.xyz),dot(Rc,p_body.xyz));
        float3 p_world = pos.xyz + r_world;
        float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
        float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
        u-=floor(u); v-=floor(v);
        float3 f=(float3)(0.0f); float Ea=0.0f;
        int ityp = folded_atom_type[atom_idx];
        if (ityp>=0 && ityp<ntypes) {
            int ioff=ityp*nbasis;
            for (int ib=0; ib<nbasis; ib++) {
                float c=LCOEFFS[ioff+ib]; float4 prm=LBASIS[ib];
                Ea += c*folded_eval_basis_rigid(u,v,p_world.z,prm);
                f  -= c*folded_eval_grad_rigid(u,v,p_world.z,prm,invLvec2d);
            }
        }
        float4 anchor = anchors[ia];
        if (anchor.w>0.0f) { float3 d=p_world-anchor.xyz; f+=d*-anchor.w; Ea+=0.5f*anchor.w*dot(d,d); }
        tf.xyz+=f; tt.xyz+=cross(r_world,f); eacc+=Ea;
        apos_world[ia]=(float4)(p_world,Ea); atom_force[ia]=(float4)(f,Ea);
    }
    Lforce[lid]=tf; Ltorq[lid]=tt; LE[lid]=eacc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride=WORKGROUP_SIZE>>1; stride>0; stride>>=1) {
        if (lid<stride) { Lforce[lid]+=Lforce[lid+stride]; Ltorq[lid]+=Ltorq[lid+stride]; LE[lid]+=LE[lid+stride]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid==0) {
        body_force[gid]=(float4)(Lforce[0].xyz, LE[0]);
        body_torque[gid]=(float4)(Ltorq[0].xyz, 0.0f);
    }
}

// ==================================================================
//  Kernel 7: rigid_body_pairff_kernel (pairwise molecule-molecule)
// ==================================================================
//
//  Forces from pairwise interactions between a dynamic rigid body and
//  a static set of atoms (another molecule fixed in world space).
//
//  Data layout (CRITICAL for branch-free GPU execution):
//    Both dynamic and static atom arrays are sorted: real atoms first,
//    then electron pairs (type=1), then sigma holes (type=2).
//    n_static_atoms / n_dyn_atoms count only real atoms (type=0).
//    This lets each thread determine its role by index comparison
//    (atom_idx < n_dyn_atoms) rather than branching on type per-pair.
//
//  Pseudo-charge encoding:
//    REQ.z stores the real partial charge for atoms (type=0).
//    For epairs (type=1), REQ.z = He (Hbond pseudo-charge, negative).
//    For sigma holes (type=2), REQ.z = Hs (sigma-hole pseudo-charge, positive).
//    This avoids per-pair type checks — the coeff = min(0, Qi*Qj) formula
//    naturally produces attraction only when signs are opposite.
//
//  Interaction models:
//    atom-atom (Morse + damped Coulomb):
//      E_morse = E0 * (e^2 - 2e),  e = exp(-alpha*(r - R0))
//      E_coul  = COULOMB_CONST * Q / sqrt(r^2 + R2SAFE)
//      Mixing: R0 = R_i + R_j, E0 = E_i * E_j, Q = Q_i * Q_j
//
//    atom-epair / epair-atom (Lorentzian Hbond / sigma-hole):
//      E = coeff * fcut(r) * lorenc(r)
//      fcut  = smoothstep(1 - r/rc) = 3x^2 - 2x^3,  x = max(0, 1-r/rc)
//      lorenc = 1/(w^2 + r^2)
//      coeff  = min(0, Q_atom * Q_pseudo)
//      He < 0 → epairs attract positive probes (H-bond donors)
//      Hs > 0 → sigma holes attract negative probes (H-bond acceptors)
//      The min(0, ...) clips to attractive-only (coeff <= 0).
//
//    epair-epair / sigma-sigma / epair-sigma: skipped (no interaction)
//
//  Branch-free design:
//    Each thread processes atoms round-robin (atom_idx = lid + i*lsize).
//    If atom_idx < n_dyn_atoms → thread is a real atom: loops over
//      static atoms (Morse+Coulomb) then static epairs+sigma (Lorentzian).
//    Else → thread is an epair/sigma: loops over static atoms only
//      (Lorentzian). All threads in the warp take the same path because
//      atoms are sorted (real atoms occupy indices 0..n_dyn_atoms-1).
//
//  Z-harmonic constraint (per-atom, not per-CoM):
//    F_z += -k_z * (p_world.z - z_target)  applied to every atom.
//    This produces both net force AND torque on the rigid body,
//    constraining the molecule to remain planar (z≈z_target) rather
//    than just constraining the CoM.
//
//  Integration: Euler with gyroscopic torque + optional FIRE.
//  Anchor springs for mouse picking included.
//
//  Parallelism: 1 workgroup = 1 body, WORKGROUP_SIZE=32, atoms round-robin.
//  Local memory: Lstatic_pos/REQ cached for all static atoms (up to 128).
//
//  TODO (DONE): See rigid_body_pairff_unified_kernel below for the single-loop
//  compact exponential formulation. Keep this legacy kernel for A/B comparison.

#ifndef MAX_STATIC_ATOMS
#define MAX_STATIC_ATOMS 128
#endif

__kernel
void rigid_body_pairff_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   static_apos,
    __global const float4*   static_REQ,
    __global const int*      static_type,
    const int                n_static,
    const int                n_static_atoms,
    const int                n_dyn_atoms,
    const float4             pairff_params,
    const float              morse_alpha,
    const float              z_target,
    const float              Hs,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lstatic_pos[MAX_STATIC_ATOMS];
    __local float4 Lstatic_REQ[MAX_STATIC_ATOMS];
    const int ia0 = mols[gid];
    const int na  = mols[gid+1] - ia0;
    const float rc   = pairff_params.y;
    const float w_hb = pairff_params.z;
    const float k_z  = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[gid];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;

    if (lid == 0) {
        pos      = poss[gid];
        qrot     = resume_fire ? qrots[gid] : normalize(qrots[gid]);
        vpos     = vposs[gid];
        vrot     = vrots[gid];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[gid].a;
        Iinv_body.b = I_body_inv[gid].b;
        Iinv_body.c = I_body_inv[gid].c;
        Ibody.a     = I_body[gid].a;
        Ibody.b     = I_body[gid].b;
        Ibody.c     = I_body[gid].c;
    }
    for (int j = lid; j < n_static; j += lsize) {
        Lstatic_pos[j]  = static_apos[j];
        Lstatic_REQ[j]  = static_REQ[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            const float3 p_world = pos.xyz + r_world;
            const float4 REQ_i = dyn_REQ[ia];
            float3 f = (float3)(0.0f);
            float E = 0.0f;

            if (atom_idx < n_dyn_atoms) {
                // --- Dyn atom vs static atoms: Morse + Coulomb ---
                for (int j = 0; j < n_static_atoms; j++) {
                    float3 dp = p_world - Lstatic_pos[j].xyz;
                    float r2 = dot(dp, dp);
                    float r  = sqrt(r2 + 1e-12f);
                    float inv_r = 1.0f / r;
                    float R0 = REQ_i.x + Lstatic_REQ[j].x;
                    float E0 = REQ_i.y * Lstatic_REQ[j].y;
                    float Q  = REQ_i.z * Lstatic_REQ[j].z;
                    float e = exp(-morse_alpha * (r - R0));
                    float E_morse = E0 * (e*e - 2.0f*e);
                    float dE_morse_dr = -2.0f * morse_alpha * E0 * e * (e - 1.0f);
                    float r2d = r2 + R2SAFE;
                    float ir2d = 1.0f / r2d;
                    float sqr_ir2d = sqrt(ir2d);
                    float E_coul = COULOMB_CONST * Q * sqr_ir2d;
                    float dE_coul_dr = -COULOMB_CONST * Q * r * ir2d * sqr_ir2d;
                    E += E_morse + E_coul;
                    f += dp * (-(dE_morse_dr + dE_coul_dr) * inv_r);
                }
                // --- Dyn atom vs static epairs: fcut * lorentzian ---
                // REQ_i.z = atom charge, Lstatic_REQ[j].z = pseudo-charge (He or Hs)
                for (int j = n_static_atoms; j < n_static; j++) {
                    float3 dp = p_world - Lstatic_pos[j].xyz;
                    float r2 = dot(dp, dp);
                    float r  = sqrt(r2 + 1e-12f);
                    float inv_r = 1.0f / r;
                    float coeff = fmin(0.0f, REQ_i.z * Lstatic_REQ[j].z);
                    if (coeff == 0.0f) continue;
                    float x = 1.0f - r * (1.0f / rc);
                    if (x <= 0.0f) continue;
                    float fcut = 3.0f*x*x - 2.0f*x*x*x;
                    float w2   = w_hb * w_hb;
                    float lorenc = 1.0f / (w2 + r2);
                    float dfcut_dr = -6.0f * x * (1.0f - x) / rc;
                    float dlorenc_dr = -2.0f * r * lorenc * lorenc;
                    float dE_dr = coeff * (dfcut_dr * lorenc + fcut * dlorenc_dr);
                    E += coeff * fcut * lorenc;
                    f += dp * (-dE_dr * inv_r);
                }
            } else {
                // --- Dyn epair vs static atoms: fcut * lorentzian ---
                // REQ_i.z = pseudo-charge (He or Hs), Lstatic_REQ[j].z = atom charge
                for (int j = 0; j < n_static_atoms; j++) {
                    float3 dp = p_world - Lstatic_pos[j].xyz;
                    float r2 = dot(dp, dp);
                    float r  = sqrt(r2 + 1e-12f);
                    float inv_r = 1.0f / r;
                    float coeff = fmin(0.0f, Lstatic_REQ[j].z * REQ_i.z);
                    if (coeff == 0.0f) continue;
                    float x = 1.0f - r * (1.0f / rc);
                    if (x <= 0.0f) continue;
                    float fcut = 3.0f*x*x - 2.0f*x*x*x;
                    float w2   = w_hb * w_hb;
                    float lorenc = 1.0f / (w2 + r2);
                    float dfcut_dr = -6.0f * x * (1.0f - x) / rc;
                    float dlorenc_dr = -2.0f * r * lorenc * lorenc;
                    float dE_dr = coeff * (dfcut_dr * lorenc + fcut * dlorenc_dr);
                    E += coeff * fcut * lorenc;
                    f += dp * (-dE_dr * inv_r);
                }
            }
            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            // Z-harmonic constraint per atom (produces both force AND torque)
            f.z += -k_z * (p_world.z - z_target);
            E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            body_force [gid] = (float4)(f, 0.0f);
            body_torque[gid] = (float4)(tq_world, 0.0f);
            const float3 tq_body   = mat3_dot_T(R, tq_world);
            const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro       = cross(vrot.xyz, L_body);
            const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
    if (lid == 0) {
        if (use_fire) fire_state[gid] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
        poss [gid] = pos;
        qrots[gid] = qrot;
        vposs[gid] = vpos;
        vrots[gid] = vrot;
    }
}

// ==================================================================
//  Kernel 8: rigid_body_pairff_unified_kernel
// ==================================================================
//
//  Same rigid-body / FIRE / anchor / z-constraint scaffolding as
//  rigid_body_pairff_kernel, but forces use ONE branch-free loop over
//  all static sites (atoms + epairs + sigma holes).
//
//  Unified compact exponential (n=8), see Forces.cl::compact_exp_pair_EF
//  and examples/density_comparison/HBondFF/fit_radial.py.
//
//  Per-site REQ packing (host):
//    .x = R,  .y = e=sqrt(E),  .z = Q (charge / He / Hs),  .w = w_blunt
//  Core flag g = (type==0) ? 1 : 0 from dyn_type / static_type.
//
//  Mixing (all lanes, same ops):
//    gij = gi*gj
//    R0  = gij*(Ri+Rj),  alpha = gij,  w = wi+wj
//    E0  = mix(attr, ei*ej, gij) * (gi+gj > 0)   // attr = -min(0,Qi*Qj)
//    Coulomb only when gij==1 (real-real).
//
//  Soft radius uses one sqrt: rho = r2/(sqrt(r2+w*w)+w). No r=sqrt(r2)
//  for the compact channel (Coulomb still uses its own damped sqrt).

__kernel
void rigid_body_pairff_unified_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   static_apos,
    __global const float4*   static_REQ,
    __global const int*      static_type,
    const int                n_static,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lstatic_pos[MAX_STATIC_ATOMS];
    __local float4 Lstatic_REQ[MAX_STATIC_ATOMS];
    __local float  Lstatic_g[MAX_STATIC_ATOMS];
    const int ia0 = mols[gid];
    const int na  = mols[gid+1] - ia0;
    const float k_z = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[gid];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);

    if (lid == 0) {
        pos      = poss[gid];
        qrot     = resume_fire ? qrots[gid] : normalize(qrots[gid]);
        vpos     = vposs[gid];
        vrot     = vrots[gid];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[gid].a;
        Iinv_body.b = I_body_inv[gid].b;
        Iinv_body.c = I_body_inv[gid].c;
        Ibody.a     = I_body[gid].a;
        Ibody.b     = I_body[gid].b;
        Ibody.c     = I_body[gid].c;
    }
    for (int j = lid; j < n_static; j += lsize) {
        Lstatic_pos[j] = static_apos[j];
        Lstatic_REQ[j] = static_REQ[j];
        Lstatic_g[j]   = (static_type[j] == 0) ? 1.0f : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            const float3 p_world = pos.xyz + r_world;
            const float4 REQ_i = dyn_REQ[ia];
            const float gi = (dyn_type[ia] == 0) ? 1.0f : 0.0f;
            float3 f = (float3)(0.0f);
            float E = 0.0f;

            for (int j = 0; j < n_static; j++) {
                float3 dp = p_world - Lstatic_pos[j].xyz;
                float r2 = dot(dp, dp);
                float4 REQ_j = Lstatic_REQ[j];
                float gj = Lstatic_g[j];
                float gij = gi * gj;
                float R0 = gij * (REQ_i.x + REQ_j.x);
                float w  = REQ_i.w + REQ_j.w;
                float alpha = gij;
                float rho_c = R0 + inv_beta_n;
                float rc2 = rho_c * (rho_c + 2.0f * w);
                float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                if (E0 != 0.0f && r2 <= rc2) {
                    float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
                    E += ev.x;
                    f += dp * ev.y;
                }
                if (gij > 0.5f) {
                    float Q = REQ_i.z * REQ_j.z;
                    float r2d = r2 + R2SAFE;
                    float ir2d = 1.0f / r2d;
                    float sqr_ir2d = sqrt(ir2d);
                    float E_coul = COULOMB_CONST * Q * sqr_ir2d;
                    // F = +kQ/(r2+eps)^{1.5} * dp  (repulsive for Q>0)
                    float f_coul_over_r = COULOMB_CONST * Q * ir2d * sqr_ir2d;
                    E += E_coul;
                    f += dp * f_coul_over_r;
                }
            }

            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            f.z += -k_z * (p_world.z - z_target);
            E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            body_force [gid] = (float4)(f, 0.0f);
            body_torque[gid] = (float4)(tq_world, 0.0f);
            const float3 tq_body   = mat3_dot_T(R, tq_world);
            const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro       = cross(vrot.xyz, L_body);
            const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
    if (lid == 0) {
        if (use_fire) fire_state[gid] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
        poss [gid] = pos;
        qrots[gid] = qrot;
        vposs[gid] = vpos;
        vrots[gid] = vrot;
    }
}

// ==================================================================
//  Kernel 9: rigid_body_pairff_unified_env_kernel  (Strategy M)
// ==================================================================
//
//  One active rigid body (gid, typically launch n_bodies=1) interacts with
//  many FIXED environment molecules via unified compact-exp PairFF.
//
//  Tiling Strategy M: outer loop over env molecules; each mol_j is
//  cooperatively loaded into __local (≤ MAX_STATIC_ATOMS sites), then
//  all active atoms interact with that tile. Local footprint O(1 mol).
//
//  Env buffers (host world-frame, Variant A):
//    env_apos / env_REQ / env_type  — flat concat of env sites
//    env_mols[n_env_mols+1]         — offsets into flat arrays
//
//  Physics identical to rigid_body_pairff_unified_kernel; only the env
//  loading pattern differs (tiled vs single static partner).

__kernel
void rigid_body_pairff_unified_env_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   env_apos,
    __global const float4*   env_REQ,
    __global const int*      env_type,
    __global const int*      env_mols,
    const int                n_env_mols,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lenv_pos[MAX_STATIC_ATOMS];
    __local float4 Lenv_REQ[MAX_STATIC_ATOMS];
    __local float  Lenv_g[MAX_STATIC_ATOMS];
    __local int    Ln_tile;
    const int ia0 = mols[gid];
    const int na  = mols[gid+1] - ia0;
    const float k_z = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[gid];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);

    if (lid == 0) {
        pos      = poss[gid];
        qrot     = resume_fire ? qrots[gid] : normalize(qrots[gid]);
        vpos     = vposs[gid];
        vrot     = vrots[gid];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[gid].a;
        Iinv_body.b = I_body_inv[gid].b;
        Iinv_body.c = I_body_inv[gid].c;
        Ibody.a     = I_body[gid].a;
        Ibody.b     = I_body[gid].b;
        Ibody.c     = I_body[gid].c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);

        float3 f_acc[ATOMS_PER_THREAD];
        float  E_acc[ATOMS_PER_THREAD];
        float3 r_store[ATOMS_PER_THREAD];
        float3 p_store[ATOMS_PER_THREAD];
        float4 REQ_store[ATOMS_PER_THREAD];
        float  gi_store[ATOMS_PER_THREAD];
        int    ia_store[ATOMS_PER_THREAD];
        int    n_own = 0;
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            r_store[n_own] = r_world;
            p_store[n_own] = pos.xyz + r_world;
            REQ_store[n_own] = dyn_REQ[ia];
            gi_store[n_own] = (dyn_type[ia] == 0) ? 1.0f : 0.0f;
            ia_store[n_own] = ia;
            f_acc[n_own] = (float3)(0.0f);
            E_acc[n_own] = 0.0f;
            n_own++;
        }

        for (int em = 0; em < n_env_mols; em++) {
            const int j0 = env_mols[em];
            const int j1 = env_mols[em+1];
            const int n_tile = j1 - j0;
            if (lid == 0) Ln_tile = n_tile;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int t = lid; t < n_tile; t += lsize) {
                Lenv_pos[t] = env_apos[j0 + t];
                Lenv_REQ[t] = env_REQ[j0 + t];
                Lenv_g[t]   = (env_type[j0 + t] == 0) ? 1.0f : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int i = 0; i < n_own; i++) {
                const float3 p_world = p_store[i];
                const float4 REQ_i = REQ_store[i];
                const float gi = gi_store[i];
                float3 f = f_acc[i];
                float E = E_acc[i];
                for (int j = 0; j < Ln_tile; j++) {
                    float3 dp = p_world - Lenv_pos[j].xyz;
                    float r2 = dot(dp, dp);
                    float4 REQ_j = Lenv_REQ[j];
                    float gj = Lenv_g[j];
                    float gij = gi * gj;
                    float R0 = gij * (REQ_i.x + REQ_j.x);
                    float w  = REQ_i.w + REQ_j.w;
                    float alpha = gij;
                    float rho_c = R0 + inv_beta_n;
                    float rc2 = rho_c * (rho_c + 2.0f * w);
                    float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                    float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                    float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                    if (E0 != 0.0f && r2 <= rc2) {
                        float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
                        E += ev.x;
                        f += dp * ev.y;
                    }
                    if (gij > 0.5f) {
                        float Q = REQ_i.z * REQ_j.z;
                        float r2d = r2 + R2SAFE;
                        float ir2d = 1.0f / r2d;
                        float sqr_ir2d = sqrt(ir2d);
                        float E_coul = COULOMB_CONST * Q * sqr_ir2d;
                        float f_coul_over_r = COULOMB_CONST * Q * ir2d * sqr_ir2d;
                        E += E_coul;
                        f += dp * f_coul_over_r;
                    }
                }
                f_acc[i] = f;
                E_acc[i] = E;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < n_own; i++) {
            float3 f = f_acc[i];
            float E = E_acc[i];
            const float3 p_world = p_store[i];
            const float3 r_world = r_store[i];
            const int ia = ia_store[i];
            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            f.z += -k_z * (p_world.z - z_target);
            E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            body_force [gid] = (float4)(f, 0.0f);
            body_torque[gid] = (float4)(tq_world, 0.0f);
            const float3 tq_body   = mat3_dot_T(R, tq_world);
            const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro       = cross(vrot.xyz, L_body);
            const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
    if (lid == 0) {
        if (use_fire) fire_state[gid] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
        poss [gid] = pos;
        qrots[gid] = qrot;
        vposs[gid] = vpos;
        vrots[gid] = vrot;
    }
}


// ==================================================================
//  Kernel 10: rigid_body_pairff_unified_faf_kernel
// ==================================================================
//
//  Fused PairFF (unified compact-exp, single static partner) + FAF substrate.
//  Multi-step MD/FIRE inside one launch (same niter loop as kernels 8/9).
//
//  Per owned atom each step:
//    1) PairFF vs static_* (identical to rigid_body_pairff_unified_kernel)
//    2) FAF if folded_atom_type[ia] ∈ [0,ntypes)  (epair/σ → use type=-1)
//    3) anchors + optional Kz (set k_z=0 when FAF holds z)
//
//  FAF basis/coeffs loaded once into __local (same as rigid_body_folded_kernel).

__kernel
void rigid_body_pairff_unified_faf_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   static_apos,
    __global const float4*   static_REQ,
    __global const int*      static_type,
    const int                n_static,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lstatic_pos[MAX_STATIC_ATOMS];
    __local float4 Lstatic_REQ[MAX_STATIC_ATOMS];
    __local float  Lstatic_g[MAX_STATIC_ATOMS];
    __local float4 LBASIS[FOLDED_BASIS_MAX_RIGID];
    __local float  LCOEFFS[FOLDED_TYPES_MAX_RIGID * FOLDED_BASIS_MAX_RIGID];
    const int ia0 = mols[gid];
    const int na  = mols[gid+1] - ia0;
    const float k_z = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[gid];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);
    const int nbasis = folded_meta.x;
    const int factorized = (folded_meta.y < 0);
    const int ntypes = factorized ? 1 : folded_meta.y;

    if (lid == 0) {
        pos      = poss[gid];
        qrot     = resume_fire ? qrots[gid] : normalize(qrots[gid]);
        vpos     = vposs[gid];
        vrot     = vrots[gid];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[gid].a;
        Iinv_body.b = I_body_inv[gid].b;
        Iinv_body.c = I_body_inv[gid].c;
        Ibody.a     = I_body[gid].a;
        Ibody.b     = I_body[gid].b;
        Ibody.c     = I_body[gid].c;
    }
    for (int j = lid; j < n_static; j += lsize) {
        Lstatic_pos[j] = static_apos[j];
        Lstatic_REQ[j] = static_REQ[j];
        Lstatic_g[j]   = (static_type[j] == 0) ? 1.0f : 0.0f;
    }
    for (int j = lid; j < nbasis; j += lsize) LBASIS[j] = folded_kxyz[j];
    const int ncoeff = factorized ? nbasis*4 : nbasis*ntypes;
    for (int j = lid; j < ncoeff; j += lsize) LCOEFFS[j] = folded_coeffs[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
    float det = ax*by - bx*ay;
    float4 invLvec2d = (float4)( by/det, -bx/det, -ay/det, ax/det );

    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            const float3 p_world = pos.xyz + r_world;
            const float4 REQ_i = dyn_REQ[ia];
            const float gi = (dyn_type[ia] == 0) ? 1.0f : 0.0f;
            float3 f = (float3)(0.0f);
            float E = 0.0f;

            for (int j = 0; j < n_static; j++) {
                float3 dp = p_world - Lstatic_pos[j].xyz;
                float r2 = dot(dp, dp);
                float4 REQ_j = Lstatic_REQ[j];
                float gj = Lstatic_g[j];
                float gij = gi * gj;
                float R0 = gij * (REQ_i.x + REQ_j.x);
                float w  = REQ_i.w + REQ_j.w;
                float alpha = gij;
                float rho_c = R0 + inv_beta_n;
                float rc2 = rho_c * (rho_c + 2.0f * w);
                float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                if (E0 != 0.0f && r2 <= rc2) {
                    float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
                    E += ev.x;
                    f += dp * ev.y;
                }
                if (gij > 0.5f) {
                    float Q = REQ_i.z * REQ_j.z;
                    float r2d = r2 + R2SAFE;
                    float ir2d = 1.0f / r2d;
                    float sqr_ir2d = sqrt(ir2d);
                    E += COULOMB_CONST * Q * sqr_ir2d;
                    f += dp * (COULOMB_CONST * Q * ir2d * sqr_ir2d);
                }
            }

            const int ityp = factorized ? 0 : folded_atom_type[ia];
            const float4 plqh = factorized ? ((__global const float4*)folded_atom_type)[ia] : (float4)(0.0f);
            if ((factorized && dyn_type[ia] == 0) || (!factorized && ityp >= 0 && ityp < ntypes)) {
                float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
                float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
                u = u - floor(u);
                v = v - floor(v);
                for (int ib = 0; ib < nbasis; ib++) {
                    float c = folded_coeff_rigid(LCOEFFS, ib, nbasis, ityp, plqh, factorized);
                    float4 prm = LBASIS[ib];
                    float  b = folded_eval_basis_rigid(u, v, p_world.z, prm);
                    float3 g = folded_eval_grad_rigid (u, v, p_world.z, prm, invLvec2d);
                    E += c * b;
                    f -= c * g;
                }
            }

            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            f.z += -k_z * (p_world.z - z_target);
            E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            body_force [gid] = (float4)(f, 0.0f);
            body_torque[gid] = (float4)(tq_world, 0.0f);
            const float3 tq_body   = mat3_dot_T(R, tq_world);
            const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro       = cross(vrot.xyz, L_body);
            const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
    if (lid == 0) {
        if (use_fire) fire_state[gid] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
        poss [gid] = pos;
        qrots[gid] = qrot;
        vposs[gid] = vpos;
        vrots[gid] = vrot;
    }
}

// ==================================================================
//  Kernel 11: rigid_body_pairff_unified_env_faf_kernel  (Strategy M + FAF)
// ==================================================================
//
//  One active body, many fixed env molecules (tiled PairFF) + FAF substrate.
//  Intended for interactive drag: niter MD/FIRE steps per host launch.
//  Local: env tile (Strategy M) + FAF basis/coeffs (loaded once).

__kernel
void rigid_body_pairff_unified_env_faf_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   env_apos,
    __global const float4*   env_REQ,
    __global const int*      env_type,
    __global const int*      env_mols,
    const int                n_env_mols,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lenv_pos[MAX_STATIC_ATOMS];
    __local float4 Lenv_REQ[MAX_STATIC_ATOMS];
    __local float  Lenv_g[MAX_STATIC_ATOMS];
    __local int    Ln_tile;
    __local float4 LBASIS[FOLDED_BASIS_MAX_RIGID];
    __local float  LCOEFFS[FOLDED_TYPES_MAX_RIGID * FOLDED_BASIS_MAX_RIGID];
    const int ia0 = mols[gid];
    const int na  = mols[gid+1] - ia0;
    const float k_z = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[gid];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);
    const int nbasis = folded_meta.x;
    const int factorized = (folded_meta.y < 0);
    const int ntypes = factorized ? 1 : folded_meta.y;

    if (lid == 0) {
        pos      = poss[gid];
        qrot     = resume_fire ? qrots[gid] : normalize(qrots[gid]);
        vpos     = vposs[gid];
        vrot     = vrots[gid];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[gid].a;
        Iinv_body.b = I_body_inv[gid].b;
        Iinv_body.c = I_body_inv[gid].c;
        Ibody.a     = I_body[gid].a;
        Ibody.b     = I_body[gid].b;
        Ibody.c     = I_body[gid].c;
    }
    for (int j = lid; j < nbasis; j += lsize) LBASIS[j] = folded_kxyz[j];
    const int ncoeff = factorized ? nbasis*4 : nbasis*ntypes;
    for (int j = lid; j < ncoeff; j += lsize) LCOEFFS[j] = folded_coeffs[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
    float det = ax*by - bx*ay;
    float4 invLvec2d = (float4)( by/det, -bx/det, -ay/det, ax/det );

    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);

        float3 f_acc[ATOMS_PER_THREAD];
        float  E_acc[ATOMS_PER_THREAD];
        float3 r_store[ATOMS_PER_THREAD];
        float3 p_store[ATOMS_PER_THREAD];
        float4 REQ_store[ATOMS_PER_THREAD];
        float  gi_store[ATOMS_PER_THREAD];
        int    ia_store[ATOMS_PER_THREAD];
        int    n_own = 0;
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            r_store[n_own] = r_world;
            p_store[n_own] = pos.xyz + r_world;
            REQ_store[n_own] = dyn_REQ[ia];
            gi_store[n_own] = (dyn_type[ia] == 0) ? 1.0f : 0.0f;
            ia_store[n_own] = ia;
            f_acc[n_own] = (float3)(0.0f);
            E_acc[n_own] = 0.0f;
            n_own++;
        }

        for (int em = 0; em < n_env_mols; em++) {
            const int j0 = env_mols[em];
            const int j1 = env_mols[em+1];
            const int n_tile = j1 - j0;
            if (lid == 0) Ln_tile = n_tile;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int t = lid; t < n_tile; t += lsize) {
                Lenv_pos[t] = env_apos[j0 + t];
                Lenv_REQ[t] = env_REQ[j0 + t];
                Lenv_g[t]   = (env_type[j0 + t] == 0) ? 1.0f : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int i = 0; i < n_own; i++) {
                const float3 p_world = p_store[i];
                const float4 REQ_i = REQ_store[i];
                const float gi = gi_store[i];
                float3 f = f_acc[i];
                float E = E_acc[i];
                for (int j = 0; j < Ln_tile; j++) {
                    float3 dp = p_world - Lenv_pos[j].xyz;
                    float r2 = dot(dp, dp);
                    float4 REQ_j = Lenv_REQ[j];
                    float gj = Lenv_g[j];
                    float gij = gi * gj;
                    float R0 = gij * (REQ_i.x + REQ_j.x);
                    float w  = REQ_i.w + REQ_j.w;
                    float alpha = gij;
                    float rho_c = R0 + inv_beta_n;
                    float rc2 = rho_c * (rho_c + 2.0f * w);
                    float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                    float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                    float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                    if (E0 != 0.0f && r2 <= rc2) {
                        float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
                        E += ev.x;
                        f += dp * ev.y;
                    }
                    if (gij > 0.5f) {
                        float Q = REQ_i.z * REQ_j.z;
                        float r2d = r2 + R2SAFE;
                        float ir2d = 1.0f / r2d;
                        float sqr_ir2d = sqrt(ir2d);
                        E += COULOMB_CONST * Q * sqr_ir2d;
                        f += dp * (COULOMB_CONST * Q * ir2d * sqr_ir2d);
                    }
                }
                f_acc[i] = f;
                E_acc[i] = E;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < n_own; i++) {
            float3 f = f_acc[i];
            float E = E_acc[i];
            const float3 p_world = p_store[i];
            const float3 r_world = r_store[i];
            const int ia = ia_store[i];

            const int ityp = factorized ? 0 : folded_atom_type[ia];
            const float4 plqh = factorized ? ((__global const float4*)folded_atom_type)[ia] : (float4)(0.0f);
            if ((factorized && dyn_type[ia] == 0) || (!factorized && ityp >= 0 && ityp < ntypes)) {
                float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
                float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
                u = u - floor(u);
                v = v - floor(v);
                for (int ib = 0; ib < nbasis; ib++) {
                    float c = folded_coeff_rigid(LCOEFFS, ib, nbasis, ityp, plqh, factorized);
                    float4 prm = LBASIS[ib];
                    float  b = folded_eval_basis_rigid(u, v, p_world.z, prm);
                    float3 g = folded_eval_grad_rigid (u, v, p_world.z, prm, invLvec2d);
                    E += c * b;
                    f -= c * g;
                }
            }

            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            f.z += -k_z * (p_world.z - z_target);
            E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            body_force [gid] = (float4)(f, 0.0f);
            body_torque[gid] = (float4)(tq_world, 0.0f);
            const float3 tq_body   = mat3_dot_T(R, tq_world);
            const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro       = cross(vrot.xyz, L_body);
            const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
    if (lid == 0) {
        if (use_fire) fire_state[gid] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
        poss [gid] = pos;
        qrots[gid] = qrot;
        vposs[gid] = vpos;
        vrots[gid] = vrot;
    }
}


// ==================================================================
//  Kernel 12: rigid_body_pairff_unified_allmol_kernel
// ==================================================================
//
//  Shared all-molecule layout (no env rebuild / no realloc on active switch):
//    mols[n_mols+1]           cumulative site offsets into flat buffers
//    apos_body / dyn_REQ / dyn_type  — body-frame sites for ALL molecules
//    poss / qrots / …[n_mols] — persistent dynamics for every molecule
//    active_mol               — which body integrates this launch
//
//  Tiling: for j != active_mol, load mol j into __local (transform body→world
//  with poss[j]/qrots[j]), PairFF vs active sites. Self-interaction skipped.
//
//  Launch: 1 workgroup (interactive single-mobile). Later: N groups, each
//  gid integrates (drop active_mol gate) for fully dynamic multi-mol MD.
//
//  FAF-free variant; see *_allmol_faf_kernel for substrate.

__kernel
void rigid_body_pairff_unified_allmol_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    const int                n_mols,
    const int                active_mol,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    const int a = active_mol;
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Rj;
    __local float4 posj;
    __local float4 qrotj;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lenv_pos[MAX_STATIC_ATOMS];
    __local float4 Lenv_REQ[MAX_STATIC_ATOMS];
    __local float  Lenv_g[MAX_STATIC_ATOMS];
    __local int    Ln_tile;
    const int ia0 = mols[a];
    const int na  = mols[a+1] - ia0;
    const float k_z = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[a];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);

    if (lid == 0) {
        pos      = poss[a];
        qrot     = resume_fire ? qrots[a] : normalize(qrots[a]);
        vpos     = vposs[a];
        vrot     = vrots[a];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[a].a;
        Iinv_body.b = I_body_inv[a].b;
        Iinv_body.c = I_body_inv[a].c;
        Ibody.a     = I_body[a].a;
        Ibody.b     = I_body[a].b;
        Ibody.c     = I_body[a].c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);

        float3 f_acc[ATOMS_PER_THREAD];
        float  E_acc[ATOMS_PER_THREAD];
        float3 r_store[ATOMS_PER_THREAD];
        float3 p_store[ATOMS_PER_THREAD];
        float4 REQ_store[ATOMS_PER_THREAD];
        float  gi_store[ATOMS_PER_THREAD];
        int    ia_store[ATOMS_PER_THREAD];
        int    n_own = 0;
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            r_store[n_own] = r_world;
            p_store[n_own] = pos.xyz + r_world;
            REQ_store[n_own] = dyn_REQ[ia];
            gi_store[n_own] = (dyn_type[ia] == 0) ? 1.0f : 0.0f;
            ia_store[n_own] = ia;
            f_acc[n_own] = (float3)(0.0f);
            E_acc[n_own] = 0.0f;
            n_own++;
        }

        for (int j = 0; j < n_mols; j++) {
            if (j == a) continue;
            const int j0 = mols[j];
            const int n_tile = mols[j+1] - j0;
            if (lid == 0) {
                Ln_tile = n_tile;
                posj  = poss[j];
                qrotj = qrots[j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if      (lid == 0) Rj.a = (float4){ quat_to_a(qrotj), 0.f };
            else if (lid == 1) Rj.b = (float4){ quat_to_b(qrotj), 0.f };
            else if (lid == 2) Rj.c = (float4){ quat_to_c(qrotj), 0.f };
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int t = lid; t < n_tile; t += lsize) {
                const float4 pb = apos_body[j0 + t];
                const float3 pw = posj.xyz + rotate_vec_by_matrix(pb.xyz, &Rj);
                Lenv_pos[t] = (float4)(pw, 0.0f);
                Lenv_REQ[t] = dyn_REQ[j0 + t];
                Lenv_g[t]   = (dyn_type[j0 + t] == 0) ? 1.0f : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int i = 0; i < n_own; i++) {
                const float3 p_world = p_store[i];
                const float4 REQ_i = REQ_store[i];
                const float gi = gi_store[i];
                float3 f = f_acc[i];
                float E = E_acc[i];
                for (int t = 0; t < Ln_tile; t++) {
                    float3 dp = p_world - Lenv_pos[t].xyz;
                    float r2 = dot(dp, dp);
                    float4 REQ_j = Lenv_REQ[t];
                    float gj = Lenv_g[t];
                    float gij = gi * gj;
                    float R0 = gij * (REQ_i.x + REQ_j.x);
                    float w  = REQ_i.w + REQ_j.w;
                    float alpha = gij;
                    float rho_c = R0 + inv_beta_n;
                    float rc2 = rho_c * (rho_c + 2.0f * w);
                    float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                    float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                    float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                    if (E0 != 0.0f && r2 <= rc2) {
                        float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
                        E += ev.x;
                        f += dp * ev.y;
                    }
                    if (gij > 0.5f) {
                        float Q = REQ_i.z * REQ_j.z;
                        float r2d = r2 + R2SAFE;
                        float ir2d = 1.0f / r2d;
                        float sqr_ir2d = sqrt(ir2d);
                        E += COULOMB_CONST * Q * sqr_ir2d;
                        f += dp * (COULOMB_CONST * Q * ir2d * sqr_ir2d);
                    }
                }
                f_acc[i] = f;
                E_acc[i] = E;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < n_own; i++) {
            float3 f = f_acc[i];
            float E = E_acc[i];
            const float3 p_world = p_store[i];
            const float3 r_world = r_store[i];
            const int ia = ia_store[i];
            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            f.z += -k_z * (p_world.z - z_target);
            E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            body_force [a] = (float4)(f, 0.0f);
            body_torque[a] = (float4)(tq_world, 0.0f);
            const float3 tq_body   = mat3_dot_T(R, tq_world);
            const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro       = cross(vrot.xyz, L_body);
            const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
    if (lid == 0) {
        if (use_fire) fire_state[a] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
        poss [a] = pos;
        qrots[a] = qrot;
        vposs[a] = vpos;
        vrots[a] = vrot;
    }
}

// ==================================================================
//  Kernel 13: rigid_body_pairff_unified_allmol_faf_kernel
// ==================================================================
//  Same as allmol + FAF substrate on active real atoms (type>=0).

__kernel
void rigid_body_pairff_unified_allmol_faf_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    const int                n_mols,
    const int                active_mol,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    const int a = active_mol;
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Rj;
    __local float4 posj;
    __local float4 qrotj;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lenv_pos[MAX_STATIC_ATOMS];
    __local float4 Lenv_REQ[MAX_STATIC_ATOMS];
    __local float  Lenv_g[MAX_STATIC_ATOMS];
    __local int    Ln_tile;
    __local float4 LBASIS[FOLDED_BASIS_MAX_RIGID];
    __local float  LCOEFFS[FOLDED_TYPES_MAX_RIGID * FOLDED_BASIS_MAX_RIGID];
    const int ia0 = mols[a];
    const int na  = mols[a+1] - ia0;
    const float k_z = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[a];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);
    const int nbasis = folded_meta.x;
    const int factorized = (folded_meta.y < 0);
    const int ntypes = factorized ? 1 : folded_meta.y;

    if (lid == 0) {
        pos      = poss[a];
        qrot     = resume_fire ? qrots[a] : normalize(qrots[a]);
        vpos     = vposs[a];
        vrot     = vrots[a];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[a].a;
        Iinv_body.b = I_body_inv[a].b;
        Iinv_body.c = I_body_inv[a].c;
        Ibody.a     = I_body[a].a;
        Ibody.b     = I_body[a].b;
        Ibody.c     = I_body[a].c;
    }
    for (int ib = lid; ib < nbasis; ib += lsize) LBASIS[ib] = folded_kxyz[ib];
    const int ncoeff = factorized ? nbasis*4 : nbasis*ntypes;
    for (int ib = lid; ib < ncoeff; ib += lsize) LCOEFFS[ib] = folded_coeffs[ib];
    barrier(CLK_LOCAL_MEM_FENCE);

    float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
    float det = ax*by - bx*ay;
    float4 invLvec2d = (float4)( by/det, -bx/det, -ay/det, ax/det );

    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);

        float3 f_acc[ATOMS_PER_THREAD];
        float  E_acc[ATOMS_PER_THREAD];
        float3 r_store[ATOMS_PER_THREAD];
        float3 p_store[ATOMS_PER_THREAD];
        float4 REQ_store[ATOMS_PER_THREAD];
        float  gi_store[ATOMS_PER_THREAD];
        int    ia_store[ATOMS_PER_THREAD];
        int    n_own = 0;
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            r_store[n_own] = r_world;
            p_store[n_own] = pos.xyz + r_world;
            REQ_store[n_own] = dyn_REQ[ia];
            gi_store[n_own] = (dyn_type[ia] == 0) ? 1.0f : 0.0f;
            ia_store[n_own] = ia;
            f_acc[n_own] = (float3)(0.0f);
            E_acc[n_own] = 0.0f;
            n_own++;
        }

        for (int j = 0; j < n_mols; j++) {
            if (j == a) continue;
            const int j0 = mols[j];
            const int n_tile = mols[j+1] - j0;
            if (lid == 0) { Ln_tile = n_tile; posj = poss[j]; qrotj = qrots[j]; }
            barrier(CLK_LOCAL_MEM_FENCE);
            if      (lid == 0) Rj.a = (float4){ quat_to_a(qrotj), 0.f };
            else if (lid == 1) Rj.b = (float4){ quat_to_b(qrotj), 0.f };
            else if (lid == 2) Rj.c = (float4){ quat_to_c(qrotj), 0.f };
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int t = lid; t < n_tile; t += lsize) {
                const float4 pb = apos_body[j0 + t];
                const float3 pw = posj.xyz + rotate_vec_by_matrix(pb.xyz, &Rj);
                Lenv_pos[t] = (float4)(pw, 0.0f);
                Lenv_REQ[t] = dyn_REQ[j0 + t];
                Lenv_g[t]   = (dyn_type[j0 + t] == 0) ? 1.0f : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = 0; i < n_own; i++) {
                const float3 p_world = p_store[i];
                const float4 REQ_i = REQ_store[i];
                const float gi = gi_store[i];
                float3 f = f_acc[i];
                float E = E_acc[i];
                for (int t = 0; t < Ln_tile; t++) {
                    float3 dp = p_world - Lenv_pos[t].xyz;
                    float r2 = dot(dp, dp);
                    float4 REQ_j = Lenv_REQ[t];
                    float gj = Lenv_g[t];
                    float gij = gi * gj;
                    float R0 = gij * (REQ_i.x + REQ_j.x);
                    float w  = REQ_i.w + REQ_j.w;
                    float alpha = gij;
                    float rho_c = R0 + inv_beta_n;
                    float rc2 = rho_c * (rho_c + 2.0f * w);
                    float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                    float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                    float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                    if (E0 != 0.0f && r2 <= rc2) {
                        float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
                        E += ev.x;
                        f += dp * ev.y;
                    }
                    if (gij > 0.5f) {
                        float Q = REQ_i.z * REQ_j.z;
                        float r2d = r2 + R2SAFE;
                        float ir2d = 1.0f / r2d;
                        float sqr_ir2d = sqrt(ir2d);
                        E += COULOMB_CONST * Q * sqr_ir2d;
                        f += dp * (COULOMB_CONST * Q * ir2d * sqr_ir2d);
                    }
                }
                f_acc[i] = f;
                E_acc[i] = E;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < n_own; i++) {
            float3 f = f_acc[i];
            float E = E_acc[i];
            const float3 p_world = p_store[i];
            const float3 r_world = r_store[i];
            const int ia = ia_store[i];
            const int ityp = factorized ? 0 : folded_atom_type[ia];
            const float4 plqh = factorized ? ((__global const float4*)folded_atom_type)[ia] : (float4)(0.0f);
            if ((factorized && dyn_type[ia] == 0) || (!factorized && ityp >= 0 && ityp < ntypes)) {
                float u = invLvec2d.x*p_world.x + invLvec2d.y*p_world.y;
                float v = invLvec2d.z*p_world.x + invLvec2d.w*p_world.y;
                u = u - floor(u); v = v - floor(v);
                for (int ib = 0; ib < nbasis; ib++) {
                    float c = folded_coeff_rigid(LCOEFFS, ib, nbasis, ityp, plqh, factorized);
                    float4 prm = LBASIS[ib];
                    E += c * folded_eval_basis_rigid(u, v, p_world.z, prm);
                    f -= c * folded_eval_grad_rigid(u, v, p_world.z, prm, invLvec2d);
                }
            }
            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            f.z += -k_z * (p_world.z - z_target);
            E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            body_force [a] = (float4)(f, 0.0f);
            body_torque[a] = (float4)(tq_world, 0.0f);
            const float3 tq_body   = mat3_dot_T(R, tq_world);
            const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro       = cross(vrot.xyz, L_body);
            const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        apos_world[ia].xyz = pos.xyz + rotate_vec_by_matrix(apos_body[ia].xyz, &R);
    }
    if (lid == 0) {
        if (use_fire) fire_state[a] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
        poss [a] = pos; qrots[a] = qrot; vposs[a] = vpos; vrots[a] = vrot;
    }
}

// ==================================================================
//  Kernel 14: rigid_body_pairff_energy_replica_kernel
// ==================================================================
//  Pure PairFF + FAF/Kz/anchor energy evaluator for MC/GA populations.
//
//  2D launch:
//    global = (nactive*RIGID_ENERGY_WG, n_replica)
//    local  = (RIGID_ENERGY_WG, 1)
//    group  = (active slot, replica)
//
//  E_out = (PairFF vs active, PairFF vs frozen, one-body, clash count).
//  The host obtains the changed-set energy as 0.5*sum(E.x)+sum(E.y+E.z).
//  Clash checks reuse PairFF distances and cover each active molecule against
//  every partner; frozen-frozen pairs are invariant across a trial batch.
//
//  NVIDIA design:
//    - 64 threads = two warps; one private site/thread avoids register spills.
//    - One 64-site scratch tile is reused by PairFF and 32-basis FAF tiles.
//    - Local footprint is ~3.1 KiB/WG, allowing 15 resident WGs in 48 KiB.
//    - Larger molecules are streamed in chunks instead of being truncated.

#ifndef RIGID_ENERGY_WG
#define RIGID_ENERGY_WG 64
#endif
#ifndef RIGID_ENERGY_SITES_PER_THREAD
#define RIGID_ENERGY_SITES_PER_THREAD 1
#endif
#ifndef RIGID_ENERGY_SITE_TILE
#define RIGID_ENERGY_SITE_TILE 64
#endif
#define RIGID_ENERGY_FAF_TILE ((RIGID_ENERGY_SITE_TILE*4)/FOLDED_TYPES_MAX_RIGID)

__kernel __attribute__((reqd_work_group_size(RIGID_ENERGY_WG, 1, 1)))
void rigid_body_pairff_energy_replica_kernel(
    __global const int*     restrict mols,
    __global const float4*  restrict poss,
    __global const float4*  restrict qrots,
    __global const float4*  restrict apos_body,
    __global const float4*  restrict dyn_REQ,
    __global const int*     restrict dyn_type,
    __global const int*     restrict active_mols,
    const int                        nactive,
    const int                        nmol,
    const int                        n_replica,
    const float2                     clash_r2,
    const float4                     pairff_params,
    const float                      beta,
    const float                      z_target,
    __global const float*   restrict folded_coeffs,
    __global const float4*  restrict folded_kxyz,
    __global const int*     restrict folded_atom_type,
    const int4                       folded_meta,
    const float4                     folded_lvec2d,
    __global const float4*  restrict anchors,
    __global float4*        restrict E_out,
    __global const int*     restrict body_state
) {
    const int lid    = get_local_id(0);
    const int islot  = get_group_id(0);
    const int irepl  = get_group_id(1);
    if (islot >= nactive || irepl >= n_replica) return;

    const int ia     = active_mols[islot];
    // Skip deleted active bodies (state < 0): write zero energy.
    if (body_state && body_state[ia] < 0) {
        if (lid == 0) E_out[irepl * nactive + islot] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    const int ia0    = mols[ia];
    const int na     = mols[ia+1] - ia0;
    const int ipose0 = irepl * nmol;
    const int nbasis = folded_meta.x;
    const int factorized = (folded_meta.y < 0);
    const int ntypes = factorized ? 1 : folded_meta.y;
    const float k_z  = pairff_params.w;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);

    __local float4  Lsite[RIGID_ENERGY_SITE_TILE];       // PairFF: xyz,g; FAF: basis params
    __local float   Lparam[RIGID_ENERGY_SITE_TILE*4];    // PairFF: REQ;   FAF: type-major coeff tile
    __local float   Lsum[4*RIGID_ENERGY_WG];
    __local cl_Mat3 LR;
    __local float4  Lpos;
    __local int     Lj0, Lnj, Lpartner_active;

    float4 invLvec2d = (float4)(0.0f);
    if (nbasis > 0) {
        const float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
        const float invdet = 1.0f / (ax*by - bx*ay);
        invLvec2d = (float4)(by*invdet, -bx*invdet, -ay*invdet, ax*invdet);
    }

    float E_active = 0.0f;
    float E_frozen = 0.0f;
    float E_one    = 0.0f;
    float E_clash  = 0.0f;

    for (int abase = 0; abase < na; abase += RIGID_ENERGY_WG*RIGID_ENERGY_SITES_PER_THREAD) {
        if (lid == 0) {
            const float4 q = normalize(qrots[ipose0 + ia]);
            Lpos = poss[ipose0 + ia];
            LR.a = (float4)(quat_to_a(q), 0.0f);
            LR.b = (float4)(quat_to_b(q), 0.0f);
            LR.c = (float4)(quat_to_c(q), 0.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float4 own_pos[RIGID_ENERGY_SITES_PER_THREAD];
        float4 own_REQ[RIGID_ENERGY_SITES_PER_THREAD];
        float4 own_faf[RIGID_ENERGY_SITES_PER_THREAD];
        int    own_type[RIGID_ENERGY_SITES_PER_THREAD];
        int nown = 0;
        for (int i = 0; i < RIGID_ENERGY_SITES_PER_THREAD; i++) {
            const int atom_idx = abase + lid + i*RIGID_ENERGY_WG;
            if (atom_idx >= na) break;
            const int isite = ia0 + atom_idx;
            const float3 pw = Lpos.xyz + rotate_vec_by_matrix(apos_body[isite].xyz, &LR);
            own_pos[nown]  = (float4)(pw, (dyn_type[isite] == 0) ? 1.0f : 0.0f);
            own_REQ[nown]  = dyn_REQ[isite];
            own_type[nown] = (nbasis > 0 && !factorized) ? folded_atom_type[isite] : -1;
            own_faf[nown]  = (nbasis > 0 && factorized) ? ((__global const float4*)folded_atom_type)[isite] : (float4)(0.0f);
            nown++;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // PairFF partner tiles. Classification is once per partner, not per site pair.
        for (int j = 0; j < nmol; j++) {
            if (j == ia) continue;
            if (body_state && body_state[j] < 0) continue;  // skip deleted partners
            if (lid == 0) {
                Lj0 = mols[j];
                Lnj = mols[j+1] - Lj0;
                const float4 q = normalize(qrots[ipose0 + j]);
                Lpos = poss[ipose0 + j];
                const float2 dcom = poss[ipose0 + ia].xy - Lpos.xy;
                if (clash_r2.x > 0.0f && dot(dcom, dcom) < clash_r2.x) E_clash = 1.0f;
                LR.a = (float4)(quat_to_a(q), 0.0f);
                LR.b = (float4)(quat_to_b(q), 0.0f);
                LR.c = (float4)(quat_to_c(q), 0.0f);
                int is_active = (nactive == nmol);
                if (nactive == 1) {
                    is_active = 0;
                } else if (nactive != nmol) {
                    for (int k = 0; k < nactive; k++) {
                        if (active_mols[k] == j) { is_active = 1; break; }
                    }
                }
                Lpartner_active = is_active;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float Ej = 0.0f;
            for (int jbase = 0; jbase < Lnj; jbase += RIGID_ENERGY_SITE_TILE) {
                const int nt = min(RIGID_ENERGY_SITE_TILE, Lnj - jbase);
                for (int t = lid; t < nt; t += RIGID_ENERGY_WG) {
                    const int jsite = Lj0 + jbase + t;
                    const float3 pw = Lpos.xyz + rotate_vec_by_matrix(apos_body[jsite].xyz, &LR);
                    Lsite[t] = (float4)(pw, (dyn_type[jsite] == 0) ? 1.0f : 0.0f);
                    vstore4(dyn_REQ[jsite], t, Lparam);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                for (int i = 0; i < nown; i++) {
                    const float3 pi = own_pos[i].xyz;
                    const float4 REQ_i = own_REQ[i];
                    const float gi = own_pos[i].w;
                    for (int t = 0; t < nt; t++) {
                        const float3 dp = pi - Lsite[t].xyz;
                        const float r2 = dot(dp, dp);
                        const float4 REQ_j = vload4(t, Lparam);
                        const float gj = Lsite[t].w;
                        const float gij = gi * gj;
                        if (clash_r2.y > 0.0f && gij > 0.5f && r2 < clash_r2.y) E_clash = 1.0f;
                        const float R0 = gij * (REQ_i.x + REQ_j.x);
                        const float w  = REQ_i.w + REQ_j.w;
                        const float rho_c = R0 + inv_beta_n;
                        const float rc2 = rho_c * (rho_c + 2.0f*w);
                        const float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                        const float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                        const float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                        if (E0 != 0.0f && r2 <= rc2) Ej += compact_exp_pair_EF(dp, R0, E0, gij, w, beta).x;
                        if (gij > 0.5f) Ej += COULOMB_CONST * REQ_i.z * REQ_j.z * sqrt(1.0f / (r2 + R2SAFE));
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (Lpartner_active) E_active += Ej; else E_frozen += Ej;
        }

        // Reuse the PairFF scratch for small FAF basis/coeff tiles.
        if (nbasis > 0) {
            for (int ibase = 0; ibase < nbasis; ibase += RIGID_ENERGY_FAF_TILE) {
                const int nbt = min(RIGID_ENERGY_FAF_TILE, nbasis - ibase);
                for (int ib = lid; ib < nbt; ib += RIGID_ENERGY_WG) Lsite[ib] = folded_kxyz[ibase + ib];
                if (factorized) {
                    for (int ic = lid; ic < nbt*4; ic += RIGID_ENERGY_WG) Lparam[ic] = folded_coeffs[ibase*4 + ic];
                } else {
                    for (int it = 0; it < ntypes; it++) {
                        for (int ib = lid; ib < nbt; ib += RIGID_ENERGY_WG) Lparam[it*nbt + ib] = folded_coeffs[it*nbasis + ibase + ib];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                for (int i = 0; i < nown; i++) {
                    const int ityp = factorized ? 0 : own_type[i];
                    if ((factorized && own_pos[i].w > 0.5f) || (!factorized && ityp >= 0 && ityp < ntypes)) {
                        const float3 p = own_pos[i].xyz;
                        float u = invLvec2d.x*p.x + invLvec2d.y*p.y;
                        float v = invLvec2d.z*p.x + invLvec2d.w*p.y;
                        u -= floor(u);
                        v -= floor(v);
                        if (factorized) {
                            for (int ib = 0; ib < nbt; ib++) E_one += dot(vload4(ib, Lparam), own_faf[i]) * folded_eval_basis_rigid(u, v, p.z, Lsite[ib]);
                        } else {
                            const int ioff = ityp * nbt;
                            for (int ib = 0; ib < nbt; ib++) E_one += Lparam[ioff + ib] * folded_eval_basis_rigid(u, v, p.z, Lsite[ib]);
                        }
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        // Preserve allmol semantics: FAF only on real sites; Kz/anchors on all sites.
        for (int i = 0; i < nown; i++) {
            const int isite = ia0 + abase + lid + i*RIGID_ENERGY_WG;
            const float3 p = own_pos[i].xyz;
            const float4 anchor = anchors[isite];
            if (anchor.w > 0.0f) {
                const float3 d = p - anchor.xyz;
                E_one += 0.5f * anchor.w * dot(d, d);
            }
            const float dz = p.z - z_target;
            E_one += 0.5f * k_z * dz * dz;
        }
    }

    Lsum[lid]                     = E_active;
    Lsum[RIGID_ENERGY_WG + lid]   = E_frozen;
    Lsum[2*RIGID_ENERGY_WG + lid] = E_one;
    Lsum[3*RIGID_ENERGY_WG + lid] = E_clash;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = RIGID_ENERGY_WG >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            Lsum[lid]                     += Lsum[lid + stride];
            Lsum[RIGID_ENERGY_WG + lid]   += Lsum[RIGID_ENERGY_WG + lid + stride];
            Lsum[2*RIGID_ENERGY_WG + lid] += Lsum[2*RIGID_ENERGY_WG + lid + stride];
            Lsum[3*RIGID_ENERGY_WG + lid] += Lsum[3*RIGID_ENERGY_WG + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) E_out[irepl*nactive + islot] = (float4)(Lsum[0], Lsum[RIGID_ENERGY_WG], Lsum[2*RIGID_ENERGY_WG], Lsum[3*RIGID_ENERGY_WG]);
}

// ==================================================================
//  Kernel 15: rigid_body_pairff_multimol_kernel
// ==================================================================
//
//  Concurrent multi-molecule MD — all molecules move simultaneously.
//  Launch: N workgroups (gid = molecule index), one per molecule.
//  Each workgroup computes forces on its molecule from ALL others
//  (read from global poss/qrots), then integrates its own molecule.
//
//  One kernel launch = one MD step (all molecules updated).
//  For N steps, enqueue this kernel N times (Strategy A) or use
//  the persistent variant below (Strategy C).
//
//  Input and output body state MUST be distinct. This ping-pong layout prevents
//  a fast workgroup from overwriting a pose while another workgroup still reads
//  the old timestep. predict_partners is an optional constant-velocity/angular-
//  velocity extrapolation for approximate multi-step chunks; K=1 is exact.
//
//  Physics identical to rigid_body_pairff_unified_allmol_kernel,
//  but a = get_group_id(0) (every molecule is active) instead of
//  a = active_mol (only one active).

__kernel
void rigid_body_pairff_multimol_kernel(
    __global const int*      mols,
    __global const float4*   poss_in,
    __global const float4*   qrots_in,
    __global const float4*   vposs_in,
    __global const float4*   vrots_in,
    __global       float4*   poss_out,
    __global       float4*   qrots_out,
    __global       float4*   vposs_out,
    __global       float4*   vrots_out,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    const int                n_mols,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter,
    const int                predict_partners,
    const int                do_faf,
    __global const float*    folded_site_coeffs,
    __global const float4*   folded_z_params,
    const int4               folded_tensor_meta,
    const float4             folded_lvec2d,
    __global const int*      body_state
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    const int a     = gid;  // each workgroup integrates its own molecule
    // body_state: +1 dynamic, 0 static, -1 deleted. Default (no buffer) = dynamic.
    const int state_a = body_state ? body_state[a] : 1;
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Rj;
    __local float4 posj;
    __local float4 qrotj;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lenv_pos[MAX_STATIC_ATOMS];
    __local float4 Lenv_REQ[MAX_STATIC_ATOMS];
    __local float  Lenv_g[MAX_STATIC_ATOMS];
    __local int    Ln_tile;
    const int ia0 = mols[a];
    const int na  = mols[a+1] - ia0;
    const float k_z = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[a];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);
    // FAF tensor metadata (uniform across the workgroup → no divergence)
    const int nu_faf  = folded_tensor_meta.x;
    const int nv_faf  = folded_tensor_meta.y;
    const int nz_faf  = folded_tensor_meta.z;
    const int nsite_faf = folded_tensor_meta.w;
    float4 invLvec2d_faf;
    if (do_faf) {
        float ax = folded_lvec2d.x, bx = folded_lvec2d.y, ay = folded_lvec2d.z, by = folded_lvec2d.w;
        float det = ax*by - bx*ay;
        invLvec2d_faf = (float4)( by/det, -bx/det, -ay/det, ax/det );
    }

    if (lid == 0) {
        pos      = poss_in[a];
        qrot     = resume_fire ? qrots_in[a] : normalize(qrots_in[a]);
        vpos     = vposs_in[a];
        vrot     = vrots_in[a];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[a].a;
        Iinv_body.b = I_body_inv[a].b;
        Iinv_body.c = I_body_inv[a].c;
        Ibody.a     = I_body[a].a;
        Ibody.b     = I_body[a].b;
        Ibody.c     = I_body[a].c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Early exit for static/deleted bodies: skip all force calculation, just write pose back
    if (state_a <= 0) {
        if (lid == 0) {
            poss_out[a]  = pos;
            qrots_out[a] = qrot;
            vposs_out[a] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            vrots_out[a] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            body_force[a]  = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            body_torque[a] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            if (use_fire) fire_state[a] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }

    for (int step = 0; step < niter; ++step) {
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);

        float3 f_acc[ATOMS_PER_THREAD];
        float  E_acc[ATOMS_PER_THREAD];
        float3 r_store[ATOMS_PER_THREAD];
        float3 p_store[ATOMS_PER_THREAD];
        float4 REQ_store[ATOMS_PER_THREAD];
        float  gi_store[ATOMS_PER_THREAD];
        int    ia_store[ATOMS_PER_THREAD];
        int    n_own = 0;
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            r_store[n_own] = r_world;
            p_store[n_own] = pos.xyz + r_world;
            REQ_store[n_own] = dyn_REQ[ia];
            gi_store[n_own] = (dyn_type[ia] == 0) ? 1.0f : 0.0f;
            ia_store[n_own] = ia;
            f_acc[n_own] = (float3)(0.0f);
            E_acc[n_own] = 0.0f;
            n_own++;
        }

        for (int j = 0; j < n_mols; j++) {
            if (j == a) continue;
            if (body_state && body_state[j] < 0) continue;  // skip deleted partners
            const int j0 = mols[j];
            const int n_tile = mols[j+1] - j0;
            if (lid == 0) {
                Ln_tile = n_tile;
                posj  = poss_in[j];
                qrotj = qrots_in[j];
                if (predict_partners && step > 0) {
                    const float tau = dt * (float)step;
                    posj.xyz += vposs_in[j].xyz * tau;
                    qrotj = normalize(quat_mult(qrotj, make_qrot_taylor(vrots_in[j].xyz * tau)));
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if      (lid == 0) Rj.a = (float4){ quat_to_a(qrotj), 0.f };
            else if (lid == 1) Rj.b = (float4){ quat_to_b(qrotj), 0.f };
            else if (lid == 2) Rj.c = (float4){ quat_to_c(qrotj), 0.f };
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int t = lid; t < n_tile; t += lsize) {
                const float4 pb = apos_body[j0 + t];
                const float3 pw = posj.xyz + rotate_vec_by_matrix(pb.xyz, &Rj);
                Lenv_pos[t] = (float4)(pw, 0.0f);
                Lenv_REQ[t] = dyn_REQ[j0 + t];
                Lenv_g[t]   = (dyn_type[j0 + t] == 0) ? 1.0f : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int i = 0; i < n_own; i++) {
                const float3 p_world = p_store[i];
                const float4 REQ_i = REQ_store[i];
                const float gi = gi_store[i];
                float3 f = f_acc[i];
                float E = E_acc[i];
                for (int t = 0; t < Ln_tile; t++) {
                    float3 dp = p_world - Lenv_pos[t].xyz;
                    float r2 = dot(dp, dp);
                    float4 REQ_j = Lenv_REQ[t];
                    float gj = Lenv_g[t];
                    float gij = gi * gj;
                    float R0 = gij * (REQ_i.x + REQ_j.x);
                    float w  = REQ_i.w + REQ_j.w;
                    float alpha = gij;
                    float rho_c = R0 + inv_beta_n;
                    float rc2 = rho_c * (rho_c + 2.0f * w);
                    float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                    float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                    float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                    if (E0 != 0.0f && r2 <= rc2) {
                        float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
                        E += ev.x;
                        f += dp * ev.y;
                    }
                    if (gij > 0.5f) {
                        float Q = REQ_i.z * REQ_j.z;
                        float r2d = r2 + R2SAFE;
                        float ir2d = 1.0f / r2d;
                        float sqr_ir2d = sqrt(ir2d);
                        E += COULOMB_CONST * Q * sqr_ir2d;
                        f += dp * (COULOMB_CONST * Q * ir2d * sqr_ir2d);
                    }
                }
                f_acc[i] = f;
                E_acc[i] = E;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < n_own; i++) {
            float3 f = f_acc[i];
            float E = E_acc[i];
            const float3 p_world = p_store[i];
            const float3 r_world = r_store[i];
            const int ia = ia_store[i];
            // FAF substrate force (tensor evaluator, materialized scalar coeffs)
            if (do_faf) {
                float4 ef = folded_eval_tensor_rigid(p_world, ia, nsite_faf, folded_site_coeffs, folded_z_params, nu_faf, nv_faf, nz_faf, invLvec2d_faf);
                f += ef.xyz;
                E += ef.w;
            }
            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            f.z += -k_z * (p_world.z - z_target);
            E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            body_force [a] = (float4)(f, 0.0f);
            body_torque[a] = (float4)(tq_world, 0.0f);
            if (state_a > 0) {  // dynamic body: integrate
                const float3 tq_body   = mat3_dot_T(R, tq_world);
                const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
                const float3 gyro       = cross(vrot.xyz, L_body);
                const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
                if (use_fire) {
                    vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                    vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
                } else {
                    vpos.xyz *= damp_lin;
                    vrot.xyz *= damp_ang;
                }
                const float dtl = use_fire ? dt_lin : dt;
                const float dta = use_fire ? dt_ang : dt;
                vpos.xyz += f * (dtl * inv_mass);
                vrot.xyz += alpha_body * dta;
                pos.xyz  += vpos.xyz * dtl;
                qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
            } else {  // static or deleted: zero velocity, keep pose
                vpos = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                vrot = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
    if (lid == 0) {
        if (use_fire && state_a > 0) fire_state[a] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
        else if (use_fire)           fire_state[a] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        poss_out [a] = pos;
        qrots_out[a] = qrot;
        vposs_out[a] = vpos;
        vrots_out[a] = vrot;
    }
}

// ==================================================================
//  Kernel 16: rigid_body_pairff_multimol_persistent_kernel
// ==================================================================
//
//  Same physics as multimol_kernel but with an internal niter loop, ping-pong
//  body state, and a software global barrier. One kernel launch = N steps.
//
//  WARNING: Requires ALL workgroups to be simultaneously resident.
//  If the GPU cannot schedule all n_mols workgroups at once, the
//  atomic barrier deadlocks. Launch with n_mols ≤ max_resident_WG
//  (conservative: n_mols ≤ 14 on GTX 1650 with 14 CUs).

__kernel
void rigid_body_pairff_multimol_persistent_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   poss_alt,
    __global       float4*   qrots_alt,
    __global       float4*   vposs_alt,
    __global       float4*   vrots_alt,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global       int*      g_barrier,     // atomic counter, reset to 0 before launch
    const int                n_mols,
    const int                n_wg,          // total workgroups launched (= n_mols)
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int gid   = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    const int a     = gid;
    __local float4 pos;
    __local float4 qrot;
    __local float4 vpos;
    __local float4 vrot;
    __local float  inv_mass;
    __local cl_Mat3 R;
    __local cl_Mat3 Rj;
    __local float4 posj;
    __local float4 qrotj;
    __local cl_Mat3 Iinv_body;
    __local cl_Mat3 Ibody;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lenv_pos[MAX_STATIC_ATOMS];
    __local float4 Lenv_REQ[MAX_STATIC_ATOMS];
    __local float  Lenv_g[MAX_STATIC_ATOMS];
    __local int    Ln_tile;
    const int ia0 = mols[a];
    const int na  = mols[a+1] - ia0;
    const float k_z = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float4 fstate = fire_state[a];
    const int resume_fire = use_fire && (fstate.x > 0.0f);
    float dt_lin = resume_fire ? fstate.x : dt, dt_ang = resume_fire ? fstate.y : dt;
    float damp_lin = resume_fire ? fstate.z : md_params.x, damp_ang = resume_fire ? fstate.w : md_params.y;
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);

    if (lid == 0) {
        pos      = poss[a];
        qrot     = resume_fire ? qrots[a] : normalize(qrots[a]);
        vpos     = vposs[a];
        vrot     = vrots[a];
        inv_mass = (pos.w > 1e-8f) ? (1.0f / pos.w) : 1.0f;
        Iinv_body.a = I_body_inv[a].a;
        Iinv_body.b = I_body_inv[a].b;
        Iinv_body.c = I_body_inv[a].c;
        Ibody.a     = I_body[a].a;
        Ibody.b     = I_body[a].b;
        Ibody.c     = I_body[a].c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 0; step < niter; ++step) {
        __global volatile float4* poss_src = (step & 1) ? poss_alt : poss;
        __global volatile float4* qrots_src = (step & 1) ? qrots_alt : qrots;
        __global volatile float4* poss_dst = (step & 1) ? poss : poss_alt;
        __global volatile float4* qrots_dst = (step & 1) ? qrots : qrots_alt;
        __global volatile float4* vposs_dst = (step & 1) ? vposs : vposs_alt;
        __global volatile float4* vrots_dst = (step & 1) ? vrots : vrots_alt;

        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);

        float3 f_acc[ATOMS_PER_THREAD];
        float  E_acc[ATOMS_PER_THREAD];
        float3 r_store[ATOMS_PER_THREAD];
        float3 p_store[ATOMS_PER_THREAD];
        float4 REQ_store[ATOMS_PER_THREAD];
        float  gi_store[ATOMS_PER_THREAD];
        int    ia_store[ATOMS_PER_THREAD];
        int    n_own = 0;
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
            r_store[n_own] = r_world;
            p_store[n_own] = pos.xyz + r_world;
            REQ_store[n_own] = dyn_REQ[ia];
            gi_store[n_own] = (dyn_type[ia] == 0) ? 1.0f : 0.0f;
            ia_store[n_own] = ia;
            f_acc[n_own] = (float3)(0.0f);
            E_acc[n_own] = 0.0f;
            n_own++;
        }

        for (int j = 0; j < n_mols; j++) {
            if (j == a) continue;
            const int j0 = mols[j];
            const int n_tile = mols[j+1] - j0;
            if (lid == 0) {
                Ln_tile = n_tile;
                posj  = poss_src[j];
                qrotj = qrots_src[j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if      (lid == 0) Rj.a = (float4){ quat_to_a(qrotj), 0.f };
            else if (lid == 1) Rj.b = (float4){ quat_to_b(qrotj), 0.f };
            else if (lid == 2) Rj.c = (float4){ quat_to_c(qrotj), 0.f };
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int t = lid; t < n_tile; t += lsize) {
                const float4 pb = apos_body[j0 + t];
                const float3 pw = posj.xyz + rotate_vec_by_matrix(pb.xyz, &Rj);
                Lenv_pos[t] = (float4)(pw, 0.0f);
                Lenv_REQ[t] = dyn_REQ[j0 + t];
                Lenv_g[t]   = (dyn_type[j0 + t] == 0) ? 1.0f : 0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int i = 0; i < n_own; i++) {
                const float3 p_world = p_store[i];
                const float4 REQ_i = REQ_store[i];
                const float gi = gi_store[i];
                float3 f = f_acc[i];
                float E = E_acc[i];
                for (int t = 0; t < Ln_tile; t++) {
                    float3 dp = p_world - Lenv_pos[t].xyz;
                    float r2 = dot(dp, dp);
                    float4 REQ_j = Lenv_REQ[t];
                    float gj = Lenv_g[t];
                    float gij = gi * gj;
                    float R0 = gij * (REQ_i.x + REQ_j.x);
                    float w  = REQ_i.w + REQ_j.w;
                    float alpha = gij;
                    float rho_c = R0 + inv_beta_n;
                    float rc2 = rho_c * (rho_c + 2.0f * w);
                    float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                    float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                    float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                    if (E0 != 0.0f && r2 <= rc2) {
                        float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
                        E += ev.x;
                        f += dp * ev.y;
                    }
                    if (gij > 0.5f) {
                        float Q = REQ_i.z * REQ_j.z;
                        float r2d = r2 + R2SAFE;
                        float ir2d = 1.0f / r2d;
                        float sqr_ir2d = sqrt(ir2d);
                        E += COULOMB_CONST * Q * sqr_ir2d;
                        f += dp * (COULOMB_CONST * Q * ir2d * sqr_ir2d);
                    }
                }
                f_acc[i] = f;
                E_acc[i] = E;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float4 total_torque = (float4)(0.0f);
        float4 total_force  = (float4)(0.0f);
        for (int i = 0; i < n_own; i++) {
            float3 f = f_acc[i];
            float E = E_acc[i];
            const float3 p_world = p_store[i];
            const float3 r_world = r_store[i];
            const int ia = ia_store[i];
            float4 anchor = anchors[ia];
            if (anchor.w > 0.0f) {
                float3 d = p_world - anchor.xyz;
                f += d * -anchor.w;
                E += 0.5f * anchor.w * dot(d, d);
            }
            f.z += -k_z * (p_world.z - z_target);
            E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
            total_force.xyz  += f;
            total_torque.xyz += cross(r_world, f);
            apos_world[ia] = (float4)(p_world, E);
            atom_force[ia] = (float4)(f, E);
        }
        Ltorq[lid]  = total_torque;
        Lforce[lid] = total_force;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                Ltorq[lid]  += Ltorq [lid + stride];
                Lforce[lid] += Lforce[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            float3 f = Lforce[0].xyz;
            float3 tq_world = Ltorq[0].xyz;
            body_force [a] = (float4)(f, 0.0f);
            body_torque[a] = (float4)(tq_world, 0.0f);
            const float3 tq_body   = mat3_dot_T(R, tq_world);
            const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
            const float3 gyro       = cross(vrot.xyz, L_body);
            const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
            if (use_fire) {
                vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
            } else {
                vpos.xyz *= damp_lin;
                vrot.xyz *= damp_ang;
            }
            const float dtl = use_fire ? dt_lin : dt;
            const float dta = use_fire ? dt_ang : dt;
            vpos.xyz += f * (dtl * inv_mass);
            vrot.xyz += alpha_body * dta;
            pos.xyz  += vpos.xyz * dtl;
            qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
            // Ping-pong writes cannot race with current-step reads.
            poss_dst [a] = pos;
            qrots_dst[a] = qrot;
            vposs_dst[a] = vpos;
            vrots_dst[a] = vrot;
            if (use_fire) fire_state[a] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        if (lid == 0) {
            atomic_add(g_barrier, 1);
            const int target = n_wg * (step + 1);
            while (atomic_add(g_barrier, 0) < target) {}
            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    }

    if ((niter & 1) && lid == 0) {
        poss[a] = pos;
        qrots[a] = qrot;
        vposs[a] = vpos;
        vrots[a] = vrot;
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // Final world positions
    if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
    else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
    else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < ATOMS_PER_THREAD; i++) {
        const int atom_idx = lid + i*lsize;
        if (atom_idx >= na) break;
        const int ia = ia0 + atom_idx;
        const float4 p_body = apos_body[ia];
        const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
        apos_world[ia].xyz = p_world;
    }
}

// ==================================================================
//  Kernel 17: rigid_body_pairff_multimol_single_wg_kernel
// ==================================================================
//
//  Single-workgroup concurrent multi-molecule MD.
//  All molecules in ONE workgroup — local barrier synchronizes everything.
//  Internal niter loop is fully correct (no global barrier needed).
//
//  Limitation: each molecule must fit WORKGROUP_SIZE*ATOMS_PER_THREAD and
//  MAX_STATIC_ATOMS. With WG=32, APT=4 → max 128 atoms per molecule.
//  For larger systems, use multimol_kernel (Strategy A) or
//  multimol_persistent_kernel (Strategy C).
//
//  Jacobi update: all molecule forces are evaluated from one local snapshot,
//  then all poses are integrated together after a workgroup barrier.
//  Threads parallelize over atoms of the current molecule.

#ifndef MAX_MULTIMOL_WG
#define MAX_MULTIMOL_WG 32
#endif

__kernel
void rigid_body_pairff_multimol_single_wg_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    const int                n_mols,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
) {
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);
    const float k_z = pairff_params.w;
    const float damp0_lin = md_params.x, damp0_ang = md_params.y;
    const float dtmin = dt * 0.1f, dtmax = dt * 10.0f;
    const int use_fire = (md_params.w < 0.0f);
    const float inv_beta_n = 8.0f / fmax(beta, 1e-6f);

    // Per-molecule local state (all molecules in one workgroup)
    __local float4 Lpos[MAX_MULTIMOL_WG];
    __local float4 Lqrot[MAX_MULTIMOL_WG];
    __local float4 Lvpos[MAX_MULTIMOL_WG];
    __local float4 Lvrot[MAX_MULTIMOL_WG];
    __local float  Linv_mass[MAX_MULTIMOL_WG];
    __local cl_Mat3 LIinv[MAX_MULTIMOL_WG];
    __local cl_Mat3 LIbody[MAX_MULTIMOL_WG];
    __local float4 Lfstate[MAX_MULTIMOL_WG];

    // Working state for current molecule
    __local cl_Mat3 R;
    __local cl_Mat3 Rj;
    __local float4 posj;
    __local float4 qrotj;
    __local float4 Ltorq [WORKGROUP_SIZE];
    __local float4 Lforce[WORKGROUP_SIZE];
    __local float4 Lenv_pos[MAX_STATIC_ATOMS];
    __local float4 Lenv_REQ[MAX_STATIC_ATOMS];
    __local float  Lenv_g[MAX_STATIC_ATOMS];
    __local int    Ln_tile;

    // Load all molecules' state into local memory
    for (int a = 0; a < n_mols; a++) {
        if (lid == 0) {
            Lpos[a]    = poss[a];
            Lqrot[a]   = (use_fire && fire_state[a].x > 0.0f) ? qrots[a] : normalize(qrots[a]);
            Lvpos[a]   = vposs[a];
            Lvrot[a]   = vrots[a];
            Linv_mass[a] = (Lpos[a].w > 1e-8f) ? (1.0f / Lpos[a].w) : 1.0f;
            LIinv[a].a = I_body_inv[a].a; LIinv[a].b = I_body_inv[a].b; LIinv[a].c = I_body_inv[a].c;
            LIbody[a].a = I_body[a].a;    LIbody[a].b = I_body[a].b;    LIbody[a].c = I_body[a].c;
            Lfstate[a] = fire_state[a];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = 0; step < niter; ++step) {
        // Gauss-Seidel: integrate each molecule sequentially
        for (int a = 0; a < n_mols; a++) {
            float4 pos  = Lpos[a];
            float4 qrot = Lqrot[a];
            float4 vpos = Lvpos[a];
            float4 vrot = Lvrot[a];
            float  inv_mass = Linv_mass[a];
            cl_Mat3 Iinv_body = LIinv[a];
            cl_Mat3 Ibody     = LIbody[a];
            float dt_lin, dt_ang, damp_lin, damp_ang;
            const int resume_fire = use_fire && (Lfstate[a].x > 0.0f);
            dt_lin = resume_fire ? Lfstate[a].x : dt;
            dt_ang = resume_fire ? Lfstate[a].y : dt;
            damp_lin = resume_fire ? Lfstate[a].z : md_params.x;
            damp_ang = resume_fire ? Lfstate[a].w : md_params.y;

            const int ia0 = mols[a];
            const int na  = mols[a+1] - ia0;

            if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
            else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
            else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
            barrier(CLK_LOCAL_MEM_FENCE);

            float3 f_acc[ATOMS_PER_THREAD];
            float  E_acc[ATOMS_PER_THREAD];
            float3 r_store[ATOMS_PER_THREAD];
            float3 p_store[ATOMS_PER_THREAD];
            float4 REQ_store[ATOMS_PER_THREAD];
            float  gi_store[ATOMS_PER_THREAD];
            int    ia_store[ATOMS_PER_THREAD];
            int    n_own = 0;
            for (int i = 0; i < ATOMS_PER_THREAD; i++) {
                const int atom_idx = lid + i*lsize;
                if (atom_idx >= na) break;
                const int ia = ia0 + atom_idx;
                const float4 p_body = apos_body[ia];
                const float3 r_world = rotate_vec_by_matrix(p_body.xyz, &R);
                r_store[n_own] = r_world;
                p_store[n_own] = pos.xyz + r_world;
                REQ_store[n_own] = dyn_REQ[ia];
                gi_store[n_own] = (dyn_type[ia] == 0) ? 1.0f : 0.0f;
                ia_store[n_own] = ia;
                f_acc[n_own] = (float3)(0.0f);
                E_acc[n_own] = 0.0f;
                n_own++;
            }

            for (int j = 0; j < n_mols; j++) {
                if (j == a) continue;
                const int j0 = mols[j];
                const int n_tile = mols[j+1] - j0;
                if (lid == 0) {
                    Ln_tile = n_tile;
                    posj  = Lpos[j];    // read from local (updated within this step if j < a)
                    qrotj = Lqrot[j];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if      (lid == 0) Rj.a = (float4){ quat_to_a(qrotj), 0.f };
                else if (lid == 1) Rj.b = (float4){ quat_to_b(qrotj), 0.f };
                else if (lid == 2) Rj.c = (float4){ quat_to_c(qrotj), 0.f };
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int t = lid; t < n_tile; t += lsize) {
                    const float4 pb = apos_body[j0 + t];
                    const float3 pw = posj.xyz + rotate_vec_by_matrix(pb.xyz, &Rj);
                    Lenv_pos[t] = (float4)(pw, 0.0f);
                    Lenv_REQ[t] = dyn_REQ[j0 + t];
                    Lenv_g[t]   = (dyn_type[j0 + t] == 0) ? 1.0f : 0.0f;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                for (int i = 0; i < n_own; i++) {
                    const float3 p_world = p_store[i];
                    const float4 REQ_i = REQ_store[i];
                    const float gi = gi_store[i];
                    float3 f = f_acc[i];
                    float E = E_acc[i];
                    for (int t = 0; t < Ln_tile; t++) {
                        float3 dp = p_world - Lenv_pos[t].xyz;
                        float r2 = dot(dp, dp);
                        float4 REQ_j = Lenv_REQ[t];
                        float gj = Lenv_g[t];
                        float gij = gi * gj;
                        float R0 = gij * (REQ_i.x + REQ_j.x);
                        float w  = REQ_i.w + REQ_j.w;
                        float alpha = gij;
                        float rho_c = R0 + inv_beta_n;
                        float rc2 = rho_c * (rho_c + 2.0f * w);
                        float attr = -fmin(0.0f, REQ_i.z * REQ_j.z);
                        float both_dummy = 1.0f - fmin(gi + gj, 1.0f);
                        float E0 = mix(attr, REQ_i.y * REQ_j.y, gij) * (1.0f - both_dummy);
                        if (E0 != 0.0f && r2 <= rc2) {
                            float2 ev = compact_exp_pair_EF(dp, R0, E0, alpha, w, beta);
                            E += ev.x;
                            f += dp * ev.y;
                        }
                        if (gij > 0.5f) {
                            float Q = REQ_i.z * REQ_j.z;
                            float r2d = r2 + R2SAFE;
                            float ir2d = 1.0f / r2d;
                            float sqr_ir2d = sqrt(ir2d);
                            E += COULOMB_CONST * Q * sqr_ir2d;
                            f += dp * (COULOMB_CONST * Q * ir2d * sqr_ir2d);
                        }
                    }
                    f_acc[i] = f;
                    E_acc[i] = E;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            float4 total_torque = (float4)(0.0f);
            float4 total_force  = (float4)(0.0f);
            for (int i = 0; i < n_own; i++) {
                float3 f = f_acc[i];
                float E = E_acc[i];
                const float3 p_world = p_store[i];
                const float3 r_world = r_store[i];
                const int ia = ia_store[i];
                float4 anchor = anchors[ia];
                if (anchor.w > 0.0f) {
                    float3 d = p_world - anchor.xyz;
                    f += d * -anchor.w;
                    E += 0.5f * anchor.w * dot(d, d);
                }
                f.z += -k_z * (p_world.z - z_target);
                E += 0.5f * k_z * (p_world.z - z_target) * (p_world.z - z_target);
                total_force.xyz  += f;
                total_torque.xyz += cross(r_world, f);
                apos_world[ia] = (float4)(p_world, E);
                atom_force[ia] = (float4)(f, E);
            }
            Ltorq[lid]  = total_torque;
            Lforce[lid] = total_force;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int stride = WORKGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    Ltorq[lid]  += Ltorq [lid + stride];
                    Lforce[lid] += Lforce[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                float3 f = Lforce[0].xyz;
                float3 tq_world = Ltorq[0].xyz;
                body_force [a] = (float4)(f, 0.0f);
                body_torque[a] = (float4)(tq_world, 0.0f);
#if 0
                // Historical Gauss-Seidel integration updated Lpos/Lqrot here,
                // allowing later molecules in the same step to see new state.
                const float3 tq_body   = mat3_dot_T(R, tq_world);
                const float3 L_body     = mat3_dot(Ibody, vrot.xyz);
                const float3 gyro       = cross(vrot.xyz, L_body);
                const float3 alpha_body = mat3_dot(Iinv_body, tq_body - gyro);
                if (use_fire) {
                    vpos.xyz = rigid_update_FIRE(f,       vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                    vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
                } else {
                    vpos.xyz *= damp_lin;
                    vrot.xyz *= damp_ang;
                }
                const float dtl = use_fire ? dt_lin : dt;
                const float dta = use_fire ? dt_ang : dt;
                vpos.xyz += f * (dtl * inv_mass);
                vrot.xyz += alpha_body * dta;
                pos.xyz  += vpos.xyz * dtl;
                qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
                // Write back to local (Gauss-Seidel: immediately visible to next molecule)
                Lpos[a]    = pos;
                Lqrot[a]   = qrot;
                Lvpos[a]   = vpos;
                Lvrot[a]   = vrot;
                if (use_fire) Lfstate[a] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
#endif
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        } // end force loop over molecules

        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            for (int a = 0; a < n_mols; a++) {
                float4 pos = Lpos[a], qrot = Lqrot[a], vpos = Lvpos[a], vrot = Lvrot[a];
                cl_Mat3 Ra;
                Ra.a = (float4)(quat_to_a(qrot), 0.0f);
                Ra.b = (float4)(quat_to_b(qrot), 0.0f);
                Ra.c = (float4)(quat_to_c(qrot), 0.0f);
                const float3 f = body_force[a].xyz;
                const float3 tq_body = mat3_dot_T(Ra, body_torque[a].xyz);
                const float3 L_body = mat3_dot(LIbody[a], vrot.xyz);
                const float3 alpha_body = mat3_dot(LIinv[a], tq_body - cross(vrot.xyz, L_body));
                float dt_lin, dt_ang, damp_lin, damp_ang;
                const int resume_fire = use_fire && (Lfstate[a].x > 0.0f);
                dt_lin = resume_fire ? Lfstate[a].x : dt;
                dt_ang = resume_fire ? Lfstate[a].y : dt;
                damp_lin = resume_fire ? Lfstate[a].z : md_params.x;
                damp_ang = resume_fire ? Lfstate[a].w : md_params.y;
                if (use_fire) {
                    vpos.xyz = rigid_update_FIRE(f, vpos.xyz, &dt_lin, &damp_lin, dtmin, dtmax, damp0_lin);
                    vrot.xyz = rigid_update_FIRE(tq_body, vrot.xyz, &dt_ang, &damp_ang, dtmin, dtmax, damp0_ang);
                } else {
                    vpos.xyz *= damp_lin;
                    vrot.xyz *= damp_ang;
                }
                const float dtl = use_fire ? dt_lin : dt;
                const float dta = use_fire ? dt_ang : dt;
                vpos.xyz += f * (dtl * Linv_mass[a]);
                vrot.xyz += alpha_body * dta;
                pos.xyz += vpos.xyz * dtl;
                qrot = normalize(quat_mult(qrot, make_qrot_taylor(vrot.xyz * dta)));
                Lpos[a] = pos; Lqrot[a] = qrot; Lvpos[a] = vpos; Lvrot[a] = vrot;
                if (use_fire) Lfstate[a] = (float4)(dt_lin, dt_ang, damp_lin, damp_ang);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    } // end step loop

    // Final: write all molecules' state back to global memory
    for (int a = 0; a < n_mols; a++) {
        if (lid == 0) {
            poss[a]    = Lpos[a];
            qrots[a]   = Lqrot[a];
            vposs[a]   = Lvpos[a];
            vrots[a]   = Lvrot[a];
            if (use_fire) fire_state[a] = Lfstate[a];
        }
    }
    // Final world positions — reuse __local R
    for (int a = 0; a < n_mols; a++) {
        const int ia0 = mols[a];
        const int na  = mols[a+1] - ia0;
        float4 qrot = Lqrot[a];
        float4 pos  = Lpos[a];
        if      (lid == 0) R.a = (float4){ quat_to_a(qrot), 0.f };
        else if (lid == 1) R.b = (float4){ quat_to_b(qrot), 0.f };
        else if (lid == 2) R.c = (float4){ quat_to_c(qrot), 0.f };
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < ATOMS_PER_THREAD; i++) {
            const int atom_idx = lid + i*lsize;
            if (atom_idx >= na) break;
            const int ia = ia0 + atom_idx;
            const float4 p_body = apos_body[ia];
            const float3 p_world = pos.xyz + rotate_vec_by_matrix(p_body.xyz, &R);
            apos_world[ia].xyz = p_world;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
