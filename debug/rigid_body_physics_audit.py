#!/usr/bin/env python3
"""Physics audit for RigidBodyDynamics / rigid.cl.

Checks (unambiguous):
  A) Free-body conservation: |v|, L_world, T_rot with F=0, damp=1
  B) Finite-difference force/torque vs GPU (folded)
  C) Residual F/T after long damped relax + force recompute at final pose
  D) Whether GPU F/T vanish at a scipy energy-minimized pose

Run: python debug/rigid_body_physics_audit.py
"""
from __future__ import annotations
import os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from spammm.forcefields.RigidBodyDynamics import (
    RigidBodyDynamics, compute_mass_properties, _quat_to_matrix_np, _ensure_float4,
)
from spammm.surfaces.FoldedRigid import load_fit, setup_rigid_folded, eval_folded_potential

FIT = os.path.join(REPO, 'data', 'fits', 'h2o_nacl.npz')
MOL = os.path.join(REPO, 'data', 'xyz', 'H2O.xyz')
OUT = os.path.join(REPO, 'debug', 'rigid_body_physics_audit')
os.makedirs(OUT, exist_ok=True)


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-30)


def energy_cpu(fit, atom_body, atype, pos, quat):
    """Total folded energy for rigid pose (CPU reference)."""
    R = _quat_to_matrix_np(np.asarray(quat, dtype=np.float32))
    # body vectors row-wise: world_i = R @ body_i
    world = pos[None, :3] + (R @ atom_body[:, :3].T).T
    E = 0.0
    for ia, it in enumerate(atype):
        E += float(eval_folded_potential(fit, int(it), world[ia:ia+1])[0])
    return E


def fd_force_torque(fit, atom_body, atype, pos, quat, eps_t=1e-4, eps_r=1e-4):
    """Central FD of energy w.r.t. COM and body-frame infinitesimal rotation."""
    pos = np.asarray(pos, dtype=np.float64).copy()
    quat = quat_normalize(quat)
    F = np.zeros(3, dtype=np.float64)
    for i in range(3):
        dp = np.zeros(3); dp[i] = eps_t
        Ep = energy_cpu(fit, atom_body, atype, pos + dp, quat)
        Em = energy_cpu(fit, atom_body, atype, pos - dp, quat)
        F[i] = -(Ep - Em) / (2 * eps_t)

    # Body-frame torque via δq = (θ/2, 1) for small θ, right-multiply: q' = q ⊗ δq
    # ΔE ≈ -τ_body · θ  => τ = -dE/dθ
    T = np.zeros(3, dtype=np.float64)
    for i in range(3):
        th = np.zeros(3); th[i] = eps_r
        # δq ≈ (th/2, 1) to first order (small-angle quaternion)
        dq = np.array([0.5 * th[0], 0.5 * th[1], 0.5 * th[2], 1.0], dtype=np.float64)
        dq = quat_normalize(dq)
        # Hamilton product q ⊗ dq
        x1,y1,z1,w1 = quat
        x2,y2,z2,w2 = dq
        qp = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])
        qm_th = -th
        dqm = np.array([0.5*qm_th[0], 0.5*qm_th[1], 0.5*qm_th[2], 1.0])
        dqm = quat_normalize(dqm)
        x2,y2,z2,w2 = dqm
        qm = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])
        Ep = energy_cpu(fit, atom_body, atype, pos, qp)
        Em = energy_cpu(fit, atom_body, atype, pos, qm)
        T[i] = -(Ep - Em) / (2 * eps_r)
    return F, T


def gpu_force_at_pose(rbd, pos, quat, zero_vel=True):
    """Upload pose, run 1 folded step with dt=0 damp=0 (no motion), read body F/T.

    Trick: lin_damp=0, ang_damp=0, dt=0 => no state change; force still evaluated.
    Actually dt=0 keeps pose; damp irrelevant. Use niter=1.
    """
    n = rbd.n_bodies
    pos4 = np.zeros((n, 4), np.float32)
    quat4 = np.zeros((n, 4), np.float32)
    pos4[0, :3] = pos
    pos4[0, 3] = rbd.mass_total if rbd.mass_total else 1.0
    quat4[0] = quat_normalize(quat).astype(np.float32)
    zero = np.zeros((n, 4), np.float32)
    # rewrite only state buffers
    rbd.toGPU('poss', pos4)
    rbd.toGPU('qrots', quat4)
    rbd.toGPU('vposs', zero)
    rbd.toGPU('vrots', zero)
    rbd.run_folded(1, dt=0.0, lin_damp=1.0, ang_damp=1.0)
    out = rbd.download_outputs()
    return out['body_force'][0, :3].astype(np.float64), out['body_torque'][0, :3].astype(np.float64), out


def section_A_free_body():
    print('\n=== A) Free rigid-body conservation (no forces) ===')
    # Water geometry, unit masses, random ω
    from spammm.topology.FFparams import load_xyz_with_REQs
    apos, reqs, enames, _, _ = load_xyz_with_REQs(MOL)
    masses = np.ones(len(enames), np.float32)
    com = (apos * masses[:, None]).sum(0) / masses.sum()
    rel = apos - com
    mtot, I, Iinv = compute_mass_properties(rel, masses)

    rbd = RigidBodyDynamics(debug=False)
    rbd.realloc(1, len(enames))
    pos4 = np.zeros((1, 4), np.float32); pos4[0, 3] = mtot
    quat4 = np.zeros((1, 4), np.float32); quat4[0, 3] = 1.0
    vpos = np.zeros((1, 4), np.float32); vpos[0, :3] = [0.1, -0.05, 0.02]
    vrot = np.zeros((1, 4), np.float32); vrot[0, :3] = [0.3, -0.2, 0.5]  # body-frame ω
    atom_body = rel[None, :, :].astype(np.float32)
    rbd.upload_state(pos4, quat4, vpos, vrot, mtot, 1.0/mtot,
                     Iinv[None], atom_body, inertia=I[None])
    # Generic kernel with Efield=0, no anchors, damp=1
    dt = 0.001
    nstep = 2000
    # Record L_world and T
    def L_world(out):
        q = out['quats'][0]
        w = out['ang_mom'][0, :3]  # actually ω body
        R = _quat_to_matrix_np(q)
        L_body = I @ w
        return R @ L_body

    def T_rot(out):
        w = out['ang_mom'][0, :3]
        return 0.5 * float(w @ (I @ w))

    out0 = rbd.download_outputs()
    L0 = L_world(out0)
    T0 = T_rot(out0)
    v0 = out0['lin_mom'][0, :3].copy()
    qn0 = np.linalg.norm(out0['quats'][0])

    rbd.run(nstep, dt, efield=[0, 0, 0], lin_damp=1.0, ang_damp=1.0)
    out1 = rbd.download_outputs()
    L1 = L_world(out1)
    T1 = T_rot(out1)
    v1 = out1['lin_mom'][0, :3]
    qn1 = np.linalg.norm(out1['quats'][0])

    print(f'  |v| drift:     {np.linalg.norm(v1-v0):.3e}  (expect ~0)')
    print(f'  |L_world|0:    {np.linalg.norm(L0):.6e}  final {np.linalg.norm(L1):.6e}  rel_drift={(np.linalg.norm(L1-L0)/max(np.linalg.norm(L0),1e-30)):.3e}')
    print(f'  L_world vec0:  {L0}')
    print(f'  L_world vec1:  {L1}')
    print(f'  T_rot0/T_rot1: {T0:.6e} / {T1:.6e}  rel_drift={(T1-T0)/max(abs(T0),1e-30):.3e}')
    print(f'  |q|0/|q|1:     {qn0:.8f} / {qn1:.8f}')
    return {'L_rel': np.linalg.norm(L1-L0)/max(np.linalg.norm(L0),1e-30), 'T_rel': abs(T1-T0)/max(abs(T0),1e-30)}


def section_B_fd():
    print('\n=== B) Finite-difference F/T vs GPU (folded) ===')
    fit = load_fit(FIT)
    rbd = setup_rigid_folded(MOL, fit, z_init=2.5, xy_init=(0.0, 0.0))
    atom_body = rbd.atom_body_host.reshape(rbd.num_atoms, 4)
    atype = fit['atom_type_ids']
    pos = np.array([0.3, -0.2, 7.0], dtype=np.float64)  # Z_SURF_TOP~5. something + 2.5
    # use actual uploaded z
    out = rbd.download_outputs()
    pos = out['pos'][0, :3].astype(np.float64).copy()
    pos[0] += 0.35
    pos[1] -= 0.25
    # tilt a bit
    ang = 0.15
    quat = np.array([np.sin(ang/2), 0.0, 0.0, np.cos(ang/2)], dtype=np.float64)

    F_fd, T_fd = fd_force_torque(fit, atom_body, atype, pos, quat)
    F_gpu, T_gpu, _ = gpu_force_at_pose(rbd, pos, quat)

    # GPU body_torque is WORLD frame (kernel stores tq_world)
    R = _quat_to_matrix_np(quat.astype(np.float32))
    T_gpu_body = R.T @ T_gpu
    T_fd_world = R @ T_fd

    print(f'  pos={pos}')
    print(f'  F_fd   = {F_fd}')
    print(f'  F_gpu  = {F_gpu}')
    print(f'  |F_fd - F_gpu| = {np.linalg.norm(F_fd - F_gpu):.3e}  rel={np.linalg.norm(F_fd-F_gpu)/max(np.linalg.norm(F_fd),1e-30):.3e}')
    print(f'  T_fd_body  = {T_fd}')
    print(f'  T_gpu_body = {T_gpu_body}  (R^T @ T_gpu_world)')
    print(f'  T_gpu_world= {T_gpu}')
    print(f'  T_fd_world = {T_fd_world}')
    print(f'  |T_body_fd - T_body_from_gpu| = {np.linalg.norm(T_fd - T_gpu_body):.3e}')
    print(f'  |T_world_fd - T_gpu_world|    = {np.linalg.norm(T_fd_world - T_gpu):.3e}')
    return {
        'F_err': np.linalg.norm(F_fd - F_gpu),
        'T_body_err': np.linalg.norm(T_fd - T_gpu_body),
        'T_world_err': np.linalg.norm(T_fd_world - T_gpu),
    }


def section_C_relax():
    print('\n=== C) Damped relax residuals (reported vs recomputed) ===')
    fit = load_fit(FIT)
    rbd = setup_rigid_folded(MOL, fit, z_init=2.5, xy_init=(0.0, 0.0))
    rbd.run_folded(8000, dt=0.01, lin_damp=0.95, ang_damp=0.90)
    out = rbd.download_outputs()
    F_rep = out['body_force'][0, :3].astype(np.float64)
    T_rep = out['body_torque'][0, :3].astype(np.float64)
    pos = out['pos'][0, :3].astype(np.float64)
    quat = out['quats'][0].astype(np.float64)
    vpos = out['lin_mom'][0, :3]
    vrot = out['ang_mom'][0, :3]
    F_rec, T_rec, out2 = gpu_force_at_pose(rbd, pos, quat)
    print(f'  final pos={pos} quat={quat}')
    print(f'  |vpos|={np.linalg.norm(vpos):.3e} |vrot|={np.linalg.norm(vrot):.3e}')
    print(f'  reported  |F|={np.linalg.norm(F_rep):.3e} |T|={np.linalg.norm(T_rep):.3e}')
    print(f'  recomputed|F|={np.linalg.norm(F_rec):.3e} |T|={np.linalg.norm(T_rec):.3e}')
    atom_body = rbd.atom_body_host.reshape(rbd.num_atoms, 4)
    F_fd, T_fd = fd_force_torque(fit, atom_body, fit['atom_type_ids'], pos, quat)
    R = _quat_to_matrix_np(quat.astype(np.float32))
    print(f'  FD at final |F|={np.linalg.norm(F_fd):.3e} |T_body|={np.linalg.norm(T_fd):.3e}')
    print(f'  |F_rec - F_fd|={np.linalg.norm(F_rec-F_fd):.3e}')
    print(f'  |T_rec_world - R@T_fd|={np.linalg.norm(T_rec - R@T_fd):.3e}')
    return {
        'F_rep': np.linalg.norm(F_rep), 'T_rep': np.linalg.norm(T_rep),
        'F_rec': np.linalg.norm(F_rec), 'T_rec': np.linalg.norm(T_rec),
        'F_fd': np.linalg.norm(F_fd), 'T_fd': np.linalg.norm(T_fd),
    }


def section_D_scipy_min():
    print('\n=== D) Scipy energy min → check if GPU F/T → 0 ===')
    from scipy.optimize import minimize
    fit = load_fit(FIT)
    rbd = setup_rigid_folded(MOL, fit, z_init=2.5, xy_init=(0.0, 0.0))
    atom_body = rbd.atom_body_host.reshape(rbd.num_atoms, 4)
    atype = fit['atom_type_ids']
    out = rbd.download_outputs()
    pos0 = out['pos'][0, :3].astype(np.float64)
    quat0 = out['quats'][0].astype(np.float64)

    # Parameterize: COM (3) + rotation vector θ (3) applied as right-multiply from identity start
    # Start from current quat: q = q0 ⊗ exp(θ)
    def pose_from_x(x):
        pos = x[:3]
        th = x[3:6]
        ang = np.linalg.norm(th)
        if ang < 1e-14:
            dq = np.array([0., 0., 0., 1.])
        else:
            axis = th / ang
            dq = np.array([*(axis * np.sin(ang/2)), np.cos(ang/2)])
        # q = quat0 ⊗ dq
        x1,y1,z1,w1 = quat0
        x2,y2,z2,w2 = dq
        q = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ])
        return pos, quat_normalize(q)

    def obj(x):
        pos, q = pose_from_x(x)
        return energy_cpu(fit, atom_body, atype, pos, q)

    x0 = np.zeros(6)
    x0[:3] = pos0
    res = minimize(obj, x0, method='L-BFGS-B', options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 500})
    pos, quat = pose_from_x(res.x)
    F_gpu, T_gpu, _ = gpu_force_at_pose(rbd, pos, quat)
    F_fd, T_fd = fd_force_torque(fit, atom_body, atype, pos, quat)
    R = _quat_to_matrix_np(quat.astype(np.float32))
    print(f'  scipy success={res.success} nfev={res.nfev} E={res.fun:.8f}')
    print(f'  min pos={pos} quat={quat}')
    print(f'  at E-min: |F_gpu|={np.linalg.norm(F_gpu):.3e} |T_gpu|={np.linalg.norm(T_gpu):.3e}')
    print(f'  at E-min: |F_fd |={np.linalg.norm(F_fd):.3e} |T_fd |={np.linalg.norm(T_fd):.3e}')
    print(f'  |F_gpu-F_fd|={np.linalg.norm(F_gpu-F_fd):.3e} |T_gpu - R@T_fd|={np.linalg.norm(T_gpu - R@T_fd):.3e}')
    return {
        'F_gpu': np.linalg.norm(F_gpu), 'T_gpu': np.linalg.norm(T_gpu),
        'F_fd': np.linalg.norm(F_fd), 'T_fd': np.linalg.norm(T_fd),
        'E': res.fun,
    }


def main():
    A = section_A_free_body()
    B = section_B_fd()
    C = section_C_relax()
    D = section_D_scipy_min()
    print('\n=== SUMMARY ===')
    print(f'A L_rel={A["L_rel"]:.3e} T_rel={A["T_rel"]:.3e}')
    print(f'B F_err={B["F_err"]:.3e} T_body_err={B["T_body_err"]:.3e} T_world_err={B["T_world_err"]:.3e}')
    print(f'C F_rep={C["F_rep"]:.3e} F_rec={C["F_rec"]:.3e} F_fd={C["F_fd"]:.3e}')
    print(f'C T_rep={C["T_rep"]:.3e} T_rec={C["T_rec"]:.3e} T_fd={C["T_fd"]:.3e}')
    print(f'D at E-min F_gpu={D["F_gpu"]:.3e} T_gpu={D["T_gpu"]:.3e} F_fd={D["F_fd"]:.3e} T_fd={D["T_fd"]:.3e}')
    # Heuristic verdicts
    if B['F_err'] > 1e-3:
        print('VERDICT: Force FD mismatch — likely force/energy inconsistency or sign/frame bug')
    if B['T_world_err'] > 1e-3 and B['T_body_err'] < 1e-3:
        print('VERDICT: Torque stored as body but compared as world (or vice versa) — frame labeling bug')
    if B['T_body_err'] > 1e-3 and B['T_world_err'] > 1e-3:
        print('VERDICT: Torque FD mismatch in both frames — torque physics bug')
    if A['L_rel'] > 1e-2:
        print('VERDICT: Angular momentum not conserved — gyro/frame/quat bug')
    if D['F_fd'] < 1e-4 and D['F_gpu'] > 1e-2:
        print('VERDICT: GPU force wrong at true energy minimum')
    if C['F_rec'] > 1e-2:
        print('VERDICT: Damped MD does not reach force-zero (optimizer / damping / residual dynamics)')


if __name__ == '__main__':
    main()
