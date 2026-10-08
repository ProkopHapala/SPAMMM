"""Deep component debug of the ContactPME-PAW core split on a REAL FDBM field.

Per-atom decomposition, NO full df scan — the goal is to localize WHERE the
split fails, not to look at final images:
  (a) per-atom radial oracle: cone samples (scatter), Jacobi spline v_i(r),
      PAW poly P_i, residual v_S,i = v_i − P_i and the fitted 5-mode core;
  (b) vertical rays over chosen atoms: E_true vs mesh-only vs core-only vs
      total (mesh+core) — 1D error localization in the AFM domain;
  (c) xz slice maps: E_true / mesh / core / total / |err|;
  (d) mesh-residual map: is E − Σv_S^oracle actually smooth on the mesh?

Usage: python3 tests/SPM/testplot_fdbm_pme_debug.py azaindol [--h-mesh 0.5]
Output: debug/testplot_fdbm_pme_debug/<mol>/*.png + DEBUG.out  (REVIEW)
New default: --fit-mode joint. Cores are prefitted before a shared weighted
solve; they overlap the entire mesh, with no physical split surface. Legacy
clamp/coremesh branches are retained for reproducing the failure record.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

OUTROOT = 'debug/testplot_fdbm_pme_debug'
MESH_HALO = 6


def softcap(E, E_cap, k=4):
    """Smooth clamp: f(E)~E for |E|<<E_cap, ->E_cap for E>>E_cap.
    f(E) = E*(1+(E/cap)^k)^(-1/k) — the soft part; E-f(E) is the hard part,
    automatically compact (nonzero only where E exceeds the cap)."""
    E = np.asarray(E, np.float64)
    return E * (1.0 + (E / E_cap) ** k) ** (-1.0 / k)


def review_joint_fit(args, apos=None, sample=None, qb=None):
    """Diagnostic glue for the shared fitter; cached rays are independent oracle samples."""
    from scipy.ndimage import map_coordinates
    from spammm.surfaces.CoarseMesh import CoarseMesh, fit_coremesh_lsq, fit_coremesh_projected, _prefilter_3d, eval_mesh
    from spammm.surfaces.PICCore import eval_core_fast
    from spammm.SPM.AFM_utils import soft_clamp_rational, plot_afm_z_profiles, imshow_afm
    h, ss = args.h_mesh, args.ss
    if args.samples:
        d = np.load(args.samples)
        sample_step = float(d['h'])/int(d['ss'])
        ss = int(round(h/sample_step))
        if ss<2 or not np.isclose(h, ss*sample_step):
            raise ValueError('--h-mesh must be an integer >=2 multiple of cached sample spacing')
        apos, lo, ns = d['apos'], d['lo'], d['ns'].astype(int)
        E = d['E_ss'].reshape(d['ns_s'])
        shape = tuple((n-1)//ss*ss+1 for n in E.shape)
        cut = tuple(slice(0, n) for n in shape)
        pts = d['pts_ss'].reshape(*E.shape, 3)[cut].reshape(-1, 3).astype(float)
        E = E[cut]; ns = (np.asarray(shape)-1)//ss+1
        rays, zr, E_ray = d['ray_pts'], d['ray_z'], d['E_ray']
        xd, zd, E_dense = d['dense_x'], d['dense_z'], d['E_dense']
        F_dense = F_ray = None
        print(f'[joint review] cached ORIGINAL field; h={h:g}, ss={ss}; no cached force oracle', flush=True)
    else:
        lo = qb[:, 0]-MESH_HALO*h
        ns = np.round((qb[:, 1]+MESH_HALO*h-lo)/h).astype(int)+1
        shape = (ns-1)*ss+1
        xyz = [lo[a]+np.arange(shape[a])*h/ss for a in range(3)]
        pts = np.stack(np.meshgrid(*xyz, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
        print(f'[joint review] sampling {len(pts)} original field points', flush=True)
        E = sample(pts)[0].reshape(shape)
        zr = apos[:, 2].max()+np.arange(1.0, 7.0, 0.02)
        sites = np.stack([apos[0], apos.mean(0)])
        rays = np.broadcast_to(sites[:, None, :], (2, len(zr), 3)).copy(); rays[:, :, 2] = zr
        E_ray, F_ray = sample(rays.reshape(-1, 3).astype(np.float32))
        E_ray, F_ray = E_ray.reshape(2, -1), F_ray.reshape(2, -1, 3)
        xd = np.arange(qb[0, 0]+1, qb[0, 1]-1, 0.05)
        zd = apos[:, 2].max()+np.arange(2.0, 6.5, 0.05)
        XD, ZD = np.meshgrid(xd, zd, indexing='ij')
        qd = np.stack([XD.ravel(), np.full(XD.size, np.median(apos[:, 1])), ZD.ravel()], 1)
        E_dense, F_dense = sample(qd.astype(np.float32))
        E_dense, F_dense = E_dense.reshape(XD.shape), F_dense.reshape(*XD.shape, 3)
    outdir = os.path.join(OUTROOT, f'{args.mol}_h{h:g}', 'projected' if args.solver == 'projected' else 'joint')
    os.makedirs(outdir, exist_ok=True)
    if not args.samples:                                  # cache for cheap refits via --samples
        np.savez(os.path.join(outdir, 'joint_samples.npz'), h=h, ss=ss, apos=apos, lo=lo, ns=ns, ns_s=E.shape,
                 E_ss=E, pts_ss=pts, ray_pts=rays, ray_z=zr, E_ray=E_ray, dense_x=xd, dense_z=zd, E_dense=E_dense)
        print(f'[joint review] cached samples -> {outdir}/joint_samples.npz (rerun with --samples)', flush=True)
    T = soft_clamp_rational(E, args.e_cut, args.e_top, smoothness=2)[0]
    # Original energy sets accessibility. The z preference is a smooth sampling
    # weight, never a hard fit mask; exactly the same weights feed both blocks.
    w = np.exp(-np.maximum(E, 0)/args.boltzmann_T) if args.wmode == 'boltz' else (1/(max(abs(E.min()), 1e-3)+np.maximum(E, 0))**2 if args.wmode == 'rel' else np.ones_like(E))
    w /= w.max()
    w += args.weight_floor
    # Keep the physical weighting identical when changing mesh spacing.
    w *= (args.w_out+(1-args.w_out)/(1+np.exp(-(pts[:, 2]-apos[:, 2].max()-args.fit_z0)/0.5))).reshape(E.shape)
    # Inner flattening is deep in the wall; outer support is a numerical radius,
    # NOT the point where the field vanishes. The mesh exists on both sides.
    rr = np.linspace(0.5, 5.0, 901)
    qwall = np.broadcast_to(apos[:, None, :], (len(apos), len(rr), 3)).copy(); qwall[:, :, 2] += rr
    if sample is None:
        Ew = map_coordinates(E, ((qwall.reshape(-1, 3)-lo)*ss/h).T, order=1, mode='nearest').reshape(len(apos), -1)
    else:
        Ew = sample(qwall.reshape(-1, 3).astype(np.float32))[0].reshape(len(apos), -1)
    crossing = Ew < args.core_elo
    if not crossing.any(axis=1).all():
        raise ValueError('core wall ray never drops below --core-elo within 5 A')
    rlo = rr[np.argmax(crossing, axis=1)]; rb = rlo+args.core_span
    centers = apos
    if args.bond_centers:
        pairs = np.array([[int(i) for i in p.split('-')] for p in args.bond_centers.split(',')])
        if pairs.ndim!=2 or pairs.shape[1]!=2 or np.any(pairs<0) or np.any(pairs>=len(apos)) or np.any(pairs[:, 0]==pairs[:, 1]):
            raise ValueError('--bond-centers requires explicit distinct atom pairs, e.g. 0-1,2-3')
        centers = np.r_[apos, apos[pairs].mean(axis=1)]
        rlo = np.r_[rlo, rlo[pairs].mean(axis=1)]; rb = rlo+args.core_span
    print(f'[joint review] numerical core supports r_lo={rlo}, r_b={rb}; overlap with mesh everywhere', flush=True)
    fitter = fit_coremesh_projected if args.solver=='projected' else fit_coremesh_lsq
    extra = dict(max_workspace_bytes=int(args.max_workspace*1024**2)) if args.solver=='projected' else dict(tol=args.solver_tol, maxiter=args.maxiter)
    mesh, core, diag = fitter(E, centers, lo, h, rlo, rb, s=ss, weights=w, target=T, lam=args.lam, bPrint=True, **extra)
    base, _, base_diag = fitter(E, np.empty((0, 3)), lo, h, [], [], s=ss, weights=w, target=T, lam=args.lam, bPrint=True, **extra)
    base_name = 'spline-only projection' if args.solver=='projected' else 'spline-only weighted'
    interp = CoarseMesh(_prefilter_3d(T[::ss, ::ss, ::ss]), lo, h, 2, mesh.query_interior)
    lines = [f'Source: {args.samples or "fresh NVIDIA FDBM oracle"}', f'h={h} ss={ss} E_cut={args.e_cut} E_top={args.e_top} boltzmann_T={args.boltzmann_T} lam={args.lam}', f'core supports: {rlo} .. {rb}; both bases overlap', f'{args.solver} objective {diag["objective"]:.9g}; {base_name} objective {base_diag["objective"]:.9g}']
    if args.solver=='projected': lines += [f'Projected reduced objective; NOT the fully weighted mesh optimum. Atom centers={len(apos)} bond centers={len(centers)-len(apos)}', f'Fit CPU preparation={diag["preparation_ms"]:.3f} ms solve={diag["solve_ms"]:.3f} ms total={diag["total_ms"]:.3f} ms; spline-only projection={base_diag["total_ms"]:.3f} ms']
    assert diag['objective'] <= base_diag['objective']*(1+1e-5), 'joint fit must improve the SAME objective'
    g = (pts-lo)/h
    deep = (E.ravel()>=args.e_top) & np.all((g>=1) & (g+2<ns), axis=1)
    if deep.any():
        deep_E = eval_mesh(mesh, pts[deep])[0]+eval_core_fast(pts[deep], centers, core)[0]
        i_min = int(np.argmin(deep_E)); p_min = pts[deep][i_min]
        msg = f'deep-wall all-domain (E>={args.e_top}): min model E={deep_E.min():.9g} at {np.round(p_min,2)} E_orig={E.ravel()[np.flatnonzero(deep)[i_min]]:.2f} n={deep.sum()}'
        lines.append(msg); print(msg, flush=True)
        # the probe only approaches from +z above the molecule; pockets below it are unreachable
        dp = pts[deep, 2] > apos[:, 2].max() + args.fit_z0
        if dp.any():
            dE = deep_E[dp]; i2 = int(np.argmin(dE)); p2 = pts[deep][dp][i2]
            msg = f'deep-wall guard probe-side (z>mol+{args.fit_z0}): min model E={dE.min():.9g} at {np.round(p2,2)} n={dp.sum()}'
            lines.append(msg); print(msg, flush=True)
            assert dE.min() > args.e_cut, f'bounded continuation created an accessible pocket in the probe-side deep wall: {msg}'
    XD, ZD = np.meshgrid(xd, zd, indexing='ij')
    qd = np.stack([XD.ravel(), np.full(XD.size, np.median(apos[:, 1])), ZD.ravel()], 1)
    Ec, Fc = eval_core_fast(qd, centers, core)
    Em, Fm = eval_mesh(mesh, qd)
    models = {base_name: eval_mesh(base, qd), 'spline-only interp': eval_mesh(interp, qd), 'core+mesh': (Em+Ec, Fm+Fc)}
    reach = (E_dense.ravel()<args.e_cut) & (qd[:, 2]>apos[:, 2].max()+args.fit_z0)
    rmin = np.min(np.linalg.norm(qd[:, None, :]-apos[None], axis=-1), axis=1)
    for name, (Ep, Fp) in models.items():
        err = Ep-E_dense.ravel()
        lines.append(f'{name}: held xz reachable E rms={np.sqrt(np.mean(err[reach]**2)):.9g} max={np.max(np.abs(err[reach])):.9g} eV n={reach.sum()}')
        if F_dense is not None:
            ferr = (Fp-F_dense.reshape(-1, 3))[reach]
            lines.append(f'{name}: held F vector rms={np.sqrt(np.mean(np.sum(ferr**2, axis=1))):.9g} max={np.max(np.linalg.norm(ferr, axis=1)):.9g} eV/A')
        for a, b in ((2.3, 3.2), (3.2, 4.5), (4.5, 8.0)):
            mask = reach & (rmin>=a) & (rmin<b)
            if mask.any(): lines.append(f'  r_min [{a},{b}): E rms={np.sqrt(np.mean(err[mask]**2)):.9g} eV n={mask.sum()}')
    # Separate the actual mesh projection error from the complementary residual.
    T_dense = soft_clamp_rational(E_dense.ravel(), args.e_cut, args.e_top, smoothness=2)[0]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    energy_lim = max(2*abs(float(E.min())), 0.03)
    fields = [E_dense, Ec.reshape(XD.shape), (T_dense-Ec).reshape(XD.shape), Em.reshape(XD.shape), (Em+Ec-E_dense.ravel()).reshape(XD.shape), (models[base_name][0]-E_dense.ravel()).reshape(XD.shape)]
    titles = ['Original E', 'Cores (C2 joins)', 'Mesh target T - cores', 'Mesh fit', 'Hybrid error E_fit - E', base_name+' error']
    for i, (arr, title) in enumerate(zip(fields, titles)):
        lim = energy_lim if i<4 else 0.02
        imshow_afm(ax.flat[i], np.clip(arr, -lim, lim), extent=[xd[0], xd[-1], zd[0], zd[-1]], title=title, pct=100)
        ax.flat[i].set_xlabel('x [A]'); ax.flat[i].set_ylabel('z [A]')
    fig.suptitle(f'{args.mol}: overlapping cores-first {args.solver} fit, h={h:g} A; inaccessible wall errors are excluded from metrics')
    fig.tight_layout(); path = os.path.join(outdir, 'joint_xz.png'); fig.savefig(path, dpi=150); plt.close(fig); print(f'REVIEW: {path}', flush=True)
    for i, ray in enumerate(rays):
        ec, fc = eval_core_fast(ray, centers, core); em, fm = eval_mesh(mesh, ray)
        eb, fb = eval_mesh(base, ray)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        plot_afm_z_profiles(zr, {'reference': E_ray[i], 'core': ec, 'mesh': em, 'core+mesh': ec+em, 'spline-only': eb}, ylabel='E [eV]', title='Overlapping decomposition', ax=ax[0])
        ax[0].set_ylim(-energy_lim, energy_lim)
        plot_afm_z_profiles(zr, {'hybrid': ec+em-E_ray[i], 'spline-only': eb-E_ray[i]}, ylabel='E error [eV]', title='Independent oracle ray', ax=ax[1]); ax[1].set_ylim(-0.02, 0.02)
        fref = F_ray[i, :, 2] if F_ray is not None else -np.gradient(E_ray[i].astype(float), zr)
        plot_afm_z_profiles(zr, {'reference': fref, 'core+mesh': fc[:, 2]+fm[:, 2], 'spline-only': fb[:, 2]}, ylabel='Fz [eV/A]', title='Oracle force' if F_ray is not None else 'Reference: FD of cached E ray', ax=ax[2]); ax[2].set_ylim(-0.1, 0.8)
        fig.tight_layout(); path = os.path.join(outdir, f'joint_ray_{i}.png'); fig.savefig(path, dpi=150); plt.close(fig); print(f'REVIEW: {path}', flush=True)
    np.savez(os.path.join(outdir, 'joint_model.npz'), mesh_coeffs=mesh.coeffs.astype(np.float32), core_coeffs=core.coeffs.astype(np.float32), origin=lo, h=h, apos=centers, molecule_apos=apos, r_lo=rlo, r_b=rb, powers=core.powers, basis=core.basis)
    lines.append(f'Resident coefficients+geometry float32 bytes={mesh.coeffs.size*4+core.coeffs.size*4+centers.size*4+rlo.size*4+rb.size*4}')
    path = os.path.join(outdir, 'joint.out')
    with open(path, 'w') as f: f.write('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True); print(f'REVIEW: {path}', flush=True)
    return mesh, core, models


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('mol', nargs='?', default='azaindol')
    ap.add_argument('--h-mesh', type=float, default=0.5)
    ap.add_argument('--din', type=float, default=1.0)
    ap.add_argument('--db', type=float, default=1.0)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--z-fit', type=float, default=1.2)
    ap.add_argument('--cap', type=float, default=10.0)
    ap.add_argument('--dilate', type=int, default=0)
    ap.add_argument('--cone-min', type=float, default=0.3)
    ap.add_argument('--e-cap', type=str, default='auto')   # y1: 'auto' = |E_min| (repulsive side only)
    ap.add_argument('--y2', type=str, default='auto')      # y2: 'auto' = 2|E_min| asymptote
    ap.add_argument('--ry1', type=str, default='auto')     # resid-clamp start: 'auto' = 2|E_min|
    ap.add_argument('--ry2', type=str, default='auto')     # resid-clamp asymptote: 'auto' = 8|E_min|
    ap.add_argument('--cap-k', type=float, default=4.0)
    ap.add_argument('--fit-mode', choices=['joint', 'clamp', 'coremesh'], default='joint')
    ap.add_argument('--samples', help='reuse original field and independent oracle rays from coremesh_samples.npz (joint only)')
    ap.add_argument('--solver', choices=['projected', 'cg'], default='projected', help='direct banded mesh elimination; cg retained as an accuracy reference')
    ap.add_argument('--bond-centers', default='', help='explicit topology pairs for extra radial centers, e.g. 0-1; no distance-based bond inference')
    ap.add_argument('--boltzmann-T', type=float, default=0.1)
    ap.add_argument('--weight-floor', type=float, default=1e-6)
    ap.add_argument('--core-span', type=float, default=3.0, help='numerical radial support width; no physical split surface')
    ap.add_argument('--solver-tol', type=float, default=1e-8)
    ap.add_argument('--maxiter', type=int, default=3000)
    ap.add_argument('--max-workspace', type=float, default=256, help='projected-solver workspace bound, MiB')
    ap.add_argument('--e-cut', type=float, default=0.3)    # T=clamp(E) onset  (coremesh)
    ap.add_argument('--e-top', type=float, default=1.0)    # T=clamp(E) asymptote (coremesh)
    ap.add_argument('--ss', type=int, default=2)           # mesh supersampling (coremesh)
    ap.add_argument('--lam', type=float, default=1e-4)     # mesh curvature penalty (legacy failure used 1e-2)
    ap.add_argument('--sweeps', type=int, default=1)       # core<->mesh alternations (coremesh)
    ap.add_argument('--wmode', choices=['boltz', 'rel', 'uniform'], default='boltz')
    ap.add_argument('--core-dom', choices=['clj', 'wall'], default='wall')  # coremesh: core [r_lo,r_b] from cLJ or measured wall
    ap.add_argument('--core-elo', type=float, default=20.0)   # wall: r_lo where E = core_elo (T ~ plateau)
    ap.add_argument('--core-rb-off', type=float, default=0.5) # wall: r_b = r(E=0) + off
    ap.add_argument('--bonds', choices=['none', 'rad', 'ang'], default='ang')  # coremesh: bond centers
    ap.add_argument('--bond-rmax', type=float, default=1.9)   # bond detection: pairs closer than this
    ap.add_argument('--atom-basis', choices=['tp', 'bspl'], default='bspl')  # coremesh atom radial basis
    ap.add_argument('--atom-nb', type=int, default=7)         # bspl modes per atom
    ap.add_argument('--fit-z0', type=float, default=1.5)      # coremesh core-fit region: z > mol_z + this
    ap.add_argument('--w-out', type=float, default=0.02)      # weak weight outside fit region (keeps cores bounded)
    args = ap.parse_args()
    if args.samples and args.fit_mode != 'joint':
        ap.error('--samples is supported only by --fit-mode joint')
    if args.boltzmann_T <= 0 or args.weight_floor <= 0 or not 0<args.w_out<=1 or args.core_span<=0:
        ap.error('need boltzmann-T>0, weight-floor>0, 0<w-out<=1, core-span>0')
    if args.samples:
        review_joint_fit(args)
        return
    mol = args.mol
    h = float(args.h_mesh)

    from spammm import atomicUtils as au
    from spammm.SPM import AFM as afm, AFM_utils as afm_utils
    from spammm.SPM.FDBMPipeline import FDBMPipeline
    from spammm.config_utils import get_dftb_basis_path
    from spammm.forcefields.FFController import make_planar_xy, orient_long_axis_x
    from spammm.surfaces.PMESplit import SplitParams
    from spammm.surfaces.PICCore import (sample_core_shells, fit_cores_paw_field,
                                         eval_vs_oracle, eval_core_fast, core_basis,
                                         inpaint_residual, fit_cores_from_samples)
    from spammm.surfaces.ContactSurface import cpm_params_from_samples
    from spammm.surfaces.CoarseMesh import _prefilter_3d, fit_mesh_lsq_3d
    from scipy.ndimage import map_coordinates

    MOLS = {'azaindol': 'data/xyz/azaindol.xyz', 'PTCDA': 'data/xyz/PTCDA.xyz'}
    # synthetic 2-atom cases for split debugging: name = (el1, el2, bond)
    DIMERS = {'cc': (['C', 'C'], 1.40), 'cn': (['C', 'N'], 1.30), 'nn': (['N', 'N'], 1.10)}
    outdir = os.path.join(OUTROOT, f'{mol}_h{h:g}')
    os.makedirs(outdir, exist_ok=True)
    lines = []

    MONO = {'c1': 'C', 'n1': 'N', 'h1': 'H'}   # DEBUG: isolated atom — radiality reference
    if mol in DIMERS:
        els, bond = DIMERS[mol]
        enames = els
        atomPos = np.array([[-bond / 2, 0., 0.], [bond / 2, 0., 0.]])
    elif mol in MONO:
        enames = [MONO[mol]]
        atomPos = np.zeros((1, 3))
    else:
        atomPos, _, enames, _, _ = au.load_xyz(MOLS[mol])
    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16}
    atomTypes = np.array([ELEM_Z[e] for e in enames], dtype=np.int32)
    atomPos = np.asarray(atomPos, dtype=np.float64)
    if len(atomPos) > 1:
        atomPos[:] = make_planar_xy(atomPos)
        orient_long_axis_x(atomPos)
    atomPos[:, 2] = 0.0
    mol_z = float(atomPos[:, 2].max())

    # query bounds = same formula as compress test (probe domain)
    R = float(max(abs(atomPos[:, 0].min()), abs(atomPos[:, 0].max()),
                  abs(atomPos[:, 1].min()), abs(atomPos[:, 1].max()))) + 3.0
    qb = np.array([[-R, R], [-R, R], [mol_z - 2.0, mol_z + 8.0]])
    m_xy = R + MESH_HALO * h + 0.5
    z_vac = float(max(qb[2, 1] - mol_z, mol_z - qb[2, 0])) + MESH_HALO * h + 0.5
    grid_spec, origin, ngrid, step = afm_utils.make_fdbm_grid_com_zsym(atomPos, args.step, m_xy, z_vac=z_vac)
    print(f'{mol} nat={len(enames)} grid={ngrid} step={step} qb={qb.tolist()}')

    basis_hsd = get_dftb_basis_path('3ob-3-1')
    pa = afm.PAULI_FITTED_DEFAULTS['3ob-3-1']
    pipe = FDBMPipeline()
    res = pipe.run_fields(atomPos, atomTypes, basis_hsd, os.path.join(outdir, 'gpu_work'),
                          grid_spec, origin, step, ngrid, float(pa['A']), float(pa['beta']), 'co', oracle=True)
    a = pipe.ensure_afm(); a.queue.finish()
    sample = res['prolonged']['sample']
    if args.fit_mode == 'joint':
        review_joint_fit(args, atomPos, sample, qb)
        return

    # ── fit cores via field oracle ──
    apos = atomPos
    _, cLJs = afm_utils._morse_atoms_from_Z(apos, atomTypes)
    alpha = float(abs(cLJs[0, 2]))
    sp = SplitParams(R0=cLJs[:, 0].astype(np.float64), E0=cLJs[:, 1].astype(np.float64),
                     q=np.zeros(len(apos)), alpha=alpha, q_tip=0.0, r_damp=0.1,
                     r_cut=6.0, split_mode='paw', delta_in=args.din, delta_b=args.db)
    z_fit = mol_z + args.z_fit
    rng = np.random.default_rng(20261002)
    shell = sample_core_shells(apos, sp.r_lo, sp.r_b, n_per_atom=6000, seed=7)
    shell = shell[shell[:, 2] > z_fit]
    cloud = rng.uniform([qb[0, 0], qb[1, 0], z_fit], qb[:, 1], (32768, 3))
    pts = np.concatenate([shell, cloud]).astype(np.float32)
    E_pts, _ = sample(pts)
    fit, orc = fit_cores_paw_field(apos, pts, E_pts, sp, bPrint=True,
                                   cone_min=args.cone_min)

    # ── mesh residual (no taper for now — see what raw residual looks like) ──
    lo = qb[:, 0] - MESH_HALO * h; hi = qb[:, 1] + MESH_HALO * h
    ns = np.round((hi - lo) / h).astype(int) + 1
    X, Y, Z = np.meshgrid(lo[0] + np.arange(ns[0]) * h,
                          lo[1] + np.arange(ns[1]) * h,
                          lo[2] + np.arange(ns[2]) * h, indexing='ij')
    nodes = np.ascontiguousarray(np.stack([X, Y, Z], axis=-1).reshape(-1, 3), dtype=np.float32)
    E_nodes, _ = sample(nodes)
    # ── SOFT-CLAMP SPLIT: clamp ONLY the repulsive side (E>0); attraction untouched ──
    # y1 = +|E_min| (start clamping at the wall foot), y2 = 2|E_min| asymptote
    E_min = float(min(E_pts.min(), E_nodes.min()))
    y1_c = abs(E_min) if args.e_cap == 'auto' else float(args.e_cap.split(',')[0])
    y2_c = 2.0 * abs(E_min) if args.y2 == 'auto' else float(args.y2.split(',')[0])
    f_soft = lambda E: afm_utils.soft_clamp_rational(np.asarray(E, np.float64), y1_c, y2_c)[0]
    print(f'E_min={E_min:.4f}  y1={y1_c:.4f} y2={y2_c:.4f}')
    # ── TWO FIT PATHS ──
    # 'clamp':    cores fit the repulsive wall E-f(E); mesh holds CLAMPED resid g(E-cores)
    # 'coremesh': doc/Tasks/ContactPME_CoreMesh_Fit_Design.md — bounded target
    #             T=clamp(E,e_cut,e_top); cores fit T (low-E weighted); mesh =
    #             supersampled penalised LSQ of T-cores; optional alternation sweeps
    if args.fit_mode == 'clamp':
        E_hard_pts = E_pts - f_soft(E_pts)
        fit_hard = fit_cores_from_samples(apos, pts, E_hard_pts, sp.r_lo, sp.r_b, bPrint=True)
        core_nodes_h, _ = eval_core_fast(nodes, apos, fit_hard)
        R_nodes = E_nodes - core_nodes_h              # resid: attraction + wall misfit
        ry1 = 2.0 * abs(E_min) if args.ry1 == 'auto' else float(args.ry1.split(',')[0])
        ry2 = 8.0 * abs(E_min) if args.ry2 == 'auto' else float(args.ry2.split(',')[0])
        g_resid = lambda R: afm_utils.soft_clamp_rational(np.asarray(R, np.float64), ry1, ry2)[0]
        cs_c = _prefilter_3d(g_resid(R_nodes).reshape(ns)).astype(np.float64)
        def mesh_eval(qq):
            return map_coordinates(cs_c, ((qq - lo) / h).T, order=3, mode='nearest', prefilter=False)
        eval_cores = lambda qq: eval_core_fast(qq, apos, fit_hard)[0]
    else:
        e_cut, e_top = args.e_cut, args.e_top
        T_of = lambda E: afm_utils.soft_clamp_rational(np.asarray(E, np.float64), e_cut, e_top)[0]
        ssm = args.ss
        ns_s = (ns - 1) * ssm + 1
        Xi = [lo[a] + np.arange(ns_s[a]) * h / ssm for a in range(3)]
        XS, YS, ZS = np.meshgrid(*Xi, indexing='ij')
        pts_ss = np.ascontiguousarray(np.stack([XS, YS, ZS], -1).reshape(-1, 3), dtype=np.float32)
        E_ss, _ = sample(pts_ss)
        T_ss = T_of(E_ss)
        E_w = abs(E_min)
        # DEBUG: core domain vs where the wall actually is (ray straight above atom 0)
        zr = np.linspace(apos[0, 2] + 0.5, apos[0, 2] + 6.0, 2000)
        qr = np.stack([np.full_like(zr, apos[0, 0]), np.full_like(zr, apos[0, 1]), zr], 1).astype(np.float32)
        Er, _ = sample(qr); rr = zr - apos[0, 2]
        r_at = lambda Ev: float(rr[np.argmax(Er < Ev)])
        print(f'[coremesh] cLJ core domain r_lo={np.atleast_1d(sp.r_lo)[0]:.3f} r_b={np.atleast_1d(sp.r_b)[0]:.3f}  '
              f'wall above atom0: r(E={args.core_elo:g})={r_at(args.core_elo):.3f} r(E=e_top)={r_at(e_top):.3f} r(E=e_cut)={r_at(e_cut):.3f} r(E=0)={r_at(0.0):.3f} r(E_min)={rr[np.argmin(Er)]:.3f}', flush=True)
        if args.core_dom == 'wall':   # DEBUG/EXPERIMENTAL: one wall ray (atom 0) for all atoms — OK for the C-C dimer only
            c_rlo, c_rb = r_at(args.core_elo), r_at(0.0) + args.core_rb_off
        else:
            c_rlo, c_rb = sp.r_lo, sp.r_b
        print(f'[coremesh] USING core domain ({args.core_dom}) r_lo={np.atleast_1d(c_rlo)[0]:.3f} r_b={np.atleast_1d(c_rb)[0]:.3f}', flush=True)
        # ── generalized centers: atoms + optional bond midpoints ──
        # radial basis per center: ('tp',powers) production t^p or ('bspl',nb) cubic B-splines
        # over [r_lo,r_b]. bond adds optional cos^l(theta_axis) factors.
        from scipy.sparse import coo_matrix
        from scipy.interpolate import BSpline
        from spammm.surfaces.PICCore import core_basis, CORE_POWERS
        BOND_SPEC = ('tp', np.array([2, 4, 8]))   # bond centers stay t^p x3 (equal to bspl in tests, kernel-compatible)
        ATOM_SPEC = ('bspl', args.atom_nb) if args.atom_basis == 'bspl' else ('tp', CORE_POWERS)
        na_at = len(apos)
        CEN = [p.copy() for p in apos]; DOM = [(c_rlo, c_rb)] * na_at
        PWR = [ATOM_SPEC] * na_at; AXX = [None] * na_at; LAG = [[] for _ in range(na_at)]
        if args.bonds != 'none':
            from scipy.spatial import cKDTree
            for i, j in sorted(cKDTree(apos).query_pairs(args.bond_rmax)):
                bm = 0.5 * (apos[i] + apos[j]); ub_ = (apos[j] - apos[i]) / np.linalg.norm(apos[j] - apos[i])
                zb = np.linspace(mol_z + 0.5, mol_z + 7.0, 1300)                       # wall ray over the midpoint
                qb_ = np.stack([np.full_like(zb, bm[0]), np.full_like(zb, bm[1]), zb], 1).astype(np.float32)
                Eb_, _ = sample(qb_); rbv = zb - bm[2]
                rlo_b = float(rbv[np.argmax(Eb_ < args.core_elo)]) if Eb_.max() > args.core_elo else c_rlo
                rb_b = float(rbv[np.argmax(Eb_ < 0.0)]) + args.core_rb_off if Eb_.min() < 0 else c_rb
                CEN.append(bm); DOM.append((rlo_b, rb_b)); PWR.append(BOND_SPEC)
                AXX.append(ub_); LAG.append([2, 4] if args.bonds == 'ang' else [])
        CEN = np.array(CEN); RL_ = np.array([d[0] for d in DOM]); RB_ = np.array([d[1] for d in DOM])
        def radial_block(r, j):
            kind, p = PWR[j]
            if kind == 'tp': return core_basis(r, RL_[j], RB_[j], p)[0]
            kn = np.linspace(RL_[j], RB_[j], p - 2); t = np.r_[[kn[0]] * 3, kn, [kn[-1]] * 3]
            B = np.zeros((len(r), p))
            for m in range(p):
                cm = np.zeros(p); cm[m] = 1
                B[:, m] = BSpline(t, cm, 3, extrapolate=False)(np.clip(r, t[3], t[-4] - 1e-9))
            B[r > RB_[j]] = 0.0
            return np.nan_to_num(B)
        nb_j = []
        for j in range(len(CEN)):
            nr = PWR[j][1] if PWR[j][0] == 'bspl' else len(PWR[j][1])
            nb_j.append(nr * (1 + len(LAG[j])))
        OFF = np.concatenate([[0], np.cumsum(nb_j)]); NC = int(OFF[-1])
        print(f'[coremesh] centers: {na_at} atoms x{nb_j[0]} + {len(CEN) - na_at} bonds x{nb_j[-1] if len(CEN) > na_at else 0} = {NC} unknowns', flush=True)
        def design_cores(qq):
            Rq = np.linalg.norm(qq[:, None, :] - CEN[None], axis=-1)
            rows, cols, vals = [], [], []
            for j in range(len(CEN)):
                si = np.nonzero(Rq[:, j] < RB_[j])[0]
                if not len(si): continue
                ph = radial_block(Rq[si, j], j)
                if LAG[j]:
                    cz = ((qq[si] - CEN[j]) @ AXX[j]) / np.maximum(Rq[si, j], 1e-9)
                    ph = np.concatenate([ph * cz[:, None] ** l for l in [0] + LAG[j]], 1)
                rows.append(np.repeat(si, ph.shape[1])); cols.append(np.tile(np.arange(OFF[j], OFF[j] + ph.shape[1]), len(si))); vals.append(ph.ravel())
            return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(qq), NC)).tocsr()
        cc_vec = np.zeros(NC)
        def eval_cores(qq): return design_cores(qq) @ cc_vec
        def fit_centers(qq, tgt, w):
            A = design_cores(qq); Aw = A.multiply(w[:, None])
            G = (A.T @ Aw).toarray(); rhs = A.T @ (w * tgt)
            G += 1e-6 * np.trace(G) / NC * np.eye(NC)
            return np.linalg.solve(G, rhs)
        # Block Gauss-Seidel only converges to the JOINT optimum if both blocks minimise the
        # SAME objective. Mesh LSQ is uniform-weighted (separable) -> cores must be too ('uniform'),
        # on the SAME sample set. 'rel'/'boltz' kept for comparison (inconsistent objective).
        w_ss = {'rel': lambda: 1.0 / (E_w + np.maximum(T_ss, 0.0))**2,
                'boltz': lambda: np.exp(-np.maximum(T_ss, 0.0) / 0.5),
                'uniform': lambda: np.ones_like(T_ss)}[args.wmode]()
        g_ss = ((pts_ss - lo) / h).T
        # core fit focuses on the probe region (z > mol_z + z0), but keeps a WEAK weight
        # everywhere else — hard restriction made cores extrapolate unboundedly outside
        # and the block alternation diverged (J grew 10x per sweep).
        mfit = pts_ss[:, 2] > mol_z + args.fit_z0
        w_f = np.where(mfit, w_ss, args.w_out * w_ss)
        J_of = lambda cc, core: float(np.mean((T_ss[mfit] - core[mfit] - map_coordinates(cc, g_ss[:, mfit], order=3, mode='nearest', prefilter=False))**2))
        cs_c = np.zeros(tuple(ns))
        for it in range(args.sweeps):
            tgt = T_ss - map_coordinates(cs_c, g_ss, order=3, mode='nearest', prefilter=False)
            cc_vec = fit_centers(pts_ss, tgt, w_f)
            core_ss = eval_cores(pts_ss)
            print(f'[coremesh it={it}] core LSQ rms(probe)={np.sqrt(np.mean((eval_cores(pts_ss[mfit]) - tgt[mfit])**2)):.4e} ({NC} cols)', flush=True)
            cs_c = fit_mesh_lsq_3d(T_ss.reshape(ns_s) - core_ss.reshape(ns_s), s=ssm, lam=args.lam)
            print(f'[coremesh it={it}] joint objective J=mean((T-cores-mesh)^2) = {J_of(cs_c, core_ss):.4e}', flush=True)
            # DEBUG: what does the mesh actually receive?
            Rm = T_ss - core_ss; reach = E_ss < e_cut
            print(f'[coremesh it={it}] cores range [{core_ss.min():.3f},{core_ss.max():.3f}]  '
                  f'mesh target T-cores range [{Rm.min():.3f},{Rm.max():.3f}]  '
                  f'unreach |T-cores| p99={np.percentile(np.abs(Rm[~reach]),99):.3f}  '
                  f'reach |T-cores| p99={np.percentile(np.abs(Rm[reach]),99):.4f}', flush=True)
        # dense reference queries for plotting at FDBM resolution (not the fit grid)
        z_ray = mol_z + np.arange(1.0, 7.0, 0.02)
        rays = np.stack([np.tile(apos[0], (len(z_ray), 1)) * np.array([1, 1, 0]) + z_ray[:, None] * np.array([0, 0, 1.0]),
                         np.tile(apos.mean(0), (len(z_ray), 1)) * np.array([1, 1, 0]) + z_ray[:, None] * np.array([0, 0, 1.0])], 0)  # over atom0, over mol center
        E_ray = np.stack([sample(r.astype(np.float32))[0] for r in rays])
        xds = np.arange(apos[:, 0].min() - 2.0, apos[:, 0].max() + 2.0, 0.05)
        zds = np.arange(mol_z + 2.0, mol_z + 6.5, 0.05)
        XD, ZD = np.meshgrid(xds, zds, indexing='ij')
        qd = np.stack([XD.ravel(), np.full(XD.size, np.median(apos[:, 1])), ZD.ravel()], 1).astype(np.float32)
        E_dense = sample(qd)[0].reshape(XD.shape)
        np.savez(os.path.join(outdir, 'coremesh_samples.npz'), pts_ss=pts_ss, E_ss=E_ss, T_ss=T_ss, apos=apos,
                 lo=lo, h=h, ns=ns, ns_s=ns_s, ss=ssm, e_cut=e_cut, e_top=e_top,
                 ray_z=z_ray, ray_pts=rays, E_ray=E_ray, dense_x=xds, dense_z=zds, E_dense=E_dense)   # DEBUG: offline basis experiments
        cs_mo = fit_mesh_lsq_3d(T_ss.reshape(ns_s), s=ssm, lam=args.lam)   # DEBUG baseline: spline only, no cores
        print(f'[coremesh] spline-only joint objective J(probe) = {J_of(cs_mo, np.zeros_like(T_ss)):.4e}  (cores+mesh must be <= this)', flush=True)
        cs_mo1 = _prefilter_3d(T_of(E_nodes).reshape(ns))                 # DEBUG baseline: spline only, nodal interp
        def mesh_eval(qq):
            return map_coordinates(cs_c, ((qq - lo) / h).T, order=3, mode='nearest', prefilter=False)
    vs_nodes, cov_nodes = eval_vs_oracle(nodes, apos, orc)
    # residual must use the FITTED core (what the GPU kernel evaluates),
    # not the oracle table -> total == E by construction where cores valid
    core_nodes, _ = eval_core_fast(nodes, apos, fit)
    resid_raw = E_nodes - core_nodes
    # mask = un-sampled deep ball OR residual too big for the mesh to hold
    # smoothly -> inpaint (the whole blob interior is then <= cap by the
    # maximum principle; the PP-stencil leak is ~cap/(6..3), not keV)
    cap = args.cap
    deep = (cov_nodes < 0.0) | (np.abs(resid_raw) > cap)
    # where are the biggest unmasked residuals? print top offenders
    i_ord = np.argsort(-np.abs(resid_raw * (~deep)))[:8]
    for i0 in i_ord:
        p = nodes[i0]; r_a = np.linalg.norm(apos - p, axis=1); ia = int(np.argmin(r_a))
        lines.append(f'  resid {resid_raw[i0]:9.1f} at {p} r_atom={r_a[ia]:.2f} '
                     f'dz/r={ (p[2]-apos[ia,2])/max(r_a[ia],1e-9):.2f} cov={cov_nodes[i0]:.2f}')
    from scipy.ndimage import binary_dilation
    if args.dilate > 0:
        deep = binary_dilation(deep.reshape(ns), iterations=args.dilate).ravel()
    resid_nodes = inpaint_residual(resid_raw, ns, deep, n_iter=400)
    # seam smoothness: z-gradient jump across the mask boundary (df-artifact proxy)
    R = resid_nodes.reshape(ns); dm = deep.reshape(ns)
    dRz = np.abs(np.diff(R, axis=2)); dz_edge = dm[:, :, :-1] ^ dm[:, :, 1:]
    if dz_edge.any():
        lines.append(f'seam |dresid/dz| p50={np.percentile(dRz[dz_edge],50):.3f} '
                     f'max={dRz[dz_edge].max():.3f}  (h={h})')
    # residual node values along the column over atom 0 — ringing vs node data?
    i0a = np.argmin(np.linalg.norm(apos - np.array([apos[0,0], apos[0,1], 0]), axis=1))
    ix0 = int(np.argmin(np.abs(lo[0] + np.arange(ns[0])*h - apos[0,0])))
    iy0 = int(np.argmin(np.abs(lo[1] + np.arange(ns[1])*h - apos[0,1])))
    col = resid_nodes.reshape(ns)[ix0, iy0, :]
    colE = E_nodes.reshape(ns)[ix0, iy0, :]
    lines.append('node column over atom0: z, E, resid')
    for iz in range(ns[2]):
        if colE[iz] > 1e-3 or abs(col[iz]) > 0.05:
            lines.append(f'  z={lo[2]+iz*h:6.2f} E={colE[iz]:10.2f} resid={col[iz]:10.3f}')
    lines.append(f'mesh E range [{E_nodes.min():.2f},{E_nodes.max():.2f}]  '
                 f'resid raw [{resid_raw.min():.2f},{resid_raw.max():.2f}] -> '
                 f'inpainted [{resid_nodes.min():.2f},{resid_nodes.max():.2f}]  '
                 f'deep frac={np.mean(deep):.4f}')

    # ═══ THE SPLIT VISUALIZED — same as the 1D illustration but on real FDBM ═══
    # vertical ray over chosen atoms: E_true, per-atom v_S contributions,
    # Σv_S (compact/core part), E−Σv_S (mesh part), mesh-node dots.
    pick_a = sorted(set(min(i, len(apos) - 1) for i in (0, 2, 4)))
    fig, ax = plt.subplots(1, len(pick_a), figsize=(5 * len(pick_a), 6), sharey=True)
    for j, i in enumerate(pick_a):
        axx = ax[j]
        z = np.linspace(0.15, 6.0, 600)
        q = np.stack([np.full_like(z, apos[i, 0]), np.full_like(z, apos[i, 1]), z], axis=1).astype(np.float32)
        E_t, _ = sample(q)
        vs_i = orc.vs_atom(i, np.abs(z - apos[i, 2])) if np.isfinite(orc.r_min[i]) else np.zeros_like(z)
        vs_all = np.zeros_like(z)
        for ia in range(len(apos)):
            if np.isfinite(orc.r_min[ia]):
                vs_all += orc.vs_atom(ia, np.linalg.norm(q - apos[ia], axis=1))
        resid = E_t - vs_all
        axx.set_yscale('symlog', linthresh=1e-2)
        axx.plot(z - apos[i, 2], E_t, 'k-', lw=2.2, label='E true')
        axx.plot(z - apos[i, 2], vs_i, 'r-', lw=1.8, label=f'v_S atom{i}')
        axx.plot(z - apos[i, 2], vs_all - vs_i, color='orange', lw=1.4, label='v_S neighbors')
        axx.plot(z - apos[i, 2], resid, 'b-', lw=1.8, label='residual (=mesh)')
        zn = lo[2] + np.arange(ns[2]) * h
        zn = zn[(zn > 0.15) & (zn < 6.0)]
        qn = np.stack([np.full_like(zn, apos[i, 0]), np.full_like(zn, apos[i, 1]), zn], axis=1).astype(np.float32)
        En, _ = sample(qn)
        vsn = np.zeros_like(zn)
        for ia in range(len(apos)):
            if np.isfinite(orc.r_min[ia]):
                vsn += orc.vs_atom(ia, np.linalg.norm(qn - apos[ia], axis=1))
        axx.plot(zn - apos[i, 2], En - vsn, 'b.', ms=7)
        axx.axhline(0, color='0.7', lw=0.5)
        for rr, lb in ((float(orc.r_lo[i]), 'r_lo'), (float(orc.r_b[i]), 'r_b'),
                       (float(orc.r_min[i]), 'r_min')):
            axx.axvline(rr, color='0.5', ls=':', lw=0.8)
            axx.text(rr, 40, lb, fontsize=7, rotation=90)
        axx.set_xlim(0, 5.5); axx.set_ylim(-0.3, 60)
        axx.set_title(f'ray over atom {i} ({enames[i]})', fontsize=10)
        axx.set_xlabel('z above atom [Å]')
        if j == 0:
            axx.legend(fontsize=8); axx.set_ylabel('E [eV]')
    # high-freq content: is the residual ripple in E_true or added by the oracle?
    from scipy.ndimage import gaussian_filter1d
    i = pick_a[0]
    z = np.linspace(1.5, 6.0, 1200)
    q = np.stack([np.full_like(z, apos[i, 0]), np.full_like(z, apos[i, 1]), z], axis=1).astype(np.float32)
    E_r, _ = sample(q)
    vs_r = np.zeros_like(z)
    for ia in range(len(apos)):
        if np.isfinite(orc.r_min[ia]):
            vs_r += orc.vs_atom(ia, np.linalg.norm(q - apos[ia], axis=1))
    res_r = E_r - vs_r
    sg = 4.0                                              # ~0.03 A smoothing
    hf_E = E_r - gaussian_filter1d(E_r, sg)
    hf_res = res_r - gaussian_filter1d(res_r, sg)
    hf_vs = vs_r - gaussian_filter1d(vs_r, sg)
    zm = z > 2.5
    lines.append(f'ray atom{i} z>2.5: hf rms  E={np.sqrt(np.mean(hf_E[zm]**2)):.4f}  '
                 f'resid={np.sqrt(np.mean(hf_res[zm]**2)):.4f}  '
                 f'v_S={np.sqrt(np.mean(hf_vs[zm]**2)):.4f}')
    fig.suptitle(f'{mol} THE SPLIT on a vertical ray — v_S must carry the wall, residual must be smooth')
    fp = os.path.join(outdir, 'split_ray.png')
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'REVIEW: {fp}')

    # same ray BETWEEN two bonded atoms — where the overlap problem lives
    i0, i1 = 0, 1
    mid = 0.5 * (apos[i0] + apos[i1])
    d01 = apos[i1] - apos[i0]
    s = np.linspace(-0.5, 1.5, 400)                    # param along bond direction
    qq = apos[i0][None, :] + s[:, None] * d01[None, :]
    qq[:, 2] = np.linspace(0.15, 3.0, 400)             # lift upward diagonally
    E_b, _ = sample(qq.astype(np.float32))
    vs_b = np.zeros(len(s))
    for ia in range(len(apos)):
        if np.isfinite(orc.r_min[ia]):
            vs_b += orc.vs_atom(ia, np.linalg.norm(qq - apos[ia], axis=1))
    fig, axx = plt.subplots(figsize=(8, 5))
    axx.plot(s, E_b, 'k-', lw=2, label='E true')
    axx.plot(s, vs_b, 'r-', lw=1.8, label='Σ v_S')
    axx.plot(s, E_b - vs_b, 'b-', lw=1.8, label='residual')
    for ia in (i0, i1):
        if np.isfinite(orc.r_min[ia]):
            axx.plot(s, orc.vs_atom(ia, np.linalg.norm(qq - apos[ia], axis=1)), '--',
                     lw=1.2, label=f'v_S atom{ia}')
    axx.axvline(0, color='0.5', ls=':'); axx.axvline(1, color='0.5', ls=':')
    axx.text(0, 0, f'atom{i0}', fontsize=8); axx.text(1, 0, f'atom{i1}', fontsize=8)
    axx.set_yscale('symlog', linthresh=1e-2); axx.legend(fontsize=8)
    axx.set_title(f'{mol} diagonal ray atom{i0}→atom{i1} lifted in z — overlap region')
    fp = os.path.join(outdir, 'split_ray_bond.png')
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'REVIEW: {fp}')

    # ═══ (a0) THE RAW SPLIT — is the field radial? is E−Σv_S smooth? ═══
    # radial fan: E vs r along rays from atom0 at several inclinations + oracle
    fan_a = pick_a[:3]
    fig, ax = plt.subplots(1, len(fan_a), figsize=(5 * len(fan_a), 5))
    ax = np.atleast_1d(ax)
    dirs = {'+z (0°)': (0, 0, 1), '30°': (0.5, 0, 0.866), '60°': (0.866, 0, 0.5),
            'lateral (90°)': (1, 0, 0), '135°': (0.707, 0, -0.707), '-z (180°)': (0, 0, -1)}
    for j, i in enumerate(fan_a):
        axx = ax[j]
        for nm, d in dirs.items():
            d = np.asarray(d, float); d /= np.linalg.norm(d)
            rr = np.linspace(0.4, 5.5, 300)
            qq = apos[i] + rr[:, None] * d[None, :]
            E_r, _ = sample(qq.astype(np.float32))
            axx.plot(rr, E_r, lw=1.1, label=nm)
        r_fine = np.linspace(0.2, float(orc.r_b[i]) + 0.6, 300)
        if np.isfinite(orc.r_min[i]):
            rp, vp = orc.r_prof[i], orc.v_prof[i]; ok = np.isfinite(vp)
            axx.plot(rp[ok], vp[ok], 'k--', lw=2, label='v_i oracle')
        axx.axvline(float(orc.r_lo[i]), color='0.5', ls=':')
        axx.axvline(float(orc.r_b[i]), color='0.5', ls=':')
        axx.set_yscale('symlog', linthresh=1e-2); axx.set_ylim(-0.3, 50)
        axx.set_title(f'atom {i} ({enames[i]}) — E along directions', fontsize=9)
        if j == 0: axx.legend(fontsize=7)
    fig.suptitle(f'{mol} RADIALITY CHECK — is the wall atom-radial? (oracle dashed)')
    fp = os.path.join(outdir, 'radiality_fan.png')
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'REVIEW: {fp}')

    # raw split on xz slice BEFORE any fitting: Σv_S oracle vs residual
    nx_s, nz_s = 160, 120
    xs = np.linspace(qb[0, 0] + 1, qb[0, 1] - 1, nx_s)
    zs = np.linspace(mol_z + 0.2, mol_z + 6.0, nz_s)
    XX, ZZ = np.meshgrid(xs, zs, indexing='ij')
    y_mid = float(np.median(apos[:, 1]))
    q = np.stack([XX.ravel(), np.full(XX.size, y_mid), ZZ.ravel()], axis=1).astype(np.float32)
    E_t, _ = sample(q)
    vs_s, cov_s = eval_vs_oracle(q, apos, orc)
    resid_s = E_t - vs_s
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    panels = [('E true', np.clip(E_t, -0.5, 2), 'viridis'),
              ('Σv_S oracle (core part)', np.clip(vs_s, -0.5, 2), 'viridis'),
              ('E−Σv_S (mesh part)', np.clip(resid_s, -0.5, 2), 'viridis'),
              ('mesh part zoom', np.clip(resid_s, -0.1, 0.1), 'bwr')]
    for k, (tt, F, cm) in enumerate(panels):
        im = ax[k].imshow(F.reshape(XX.shape).T, origin='lower', cmap=cm,
                          extent=[xs[0], xs[-1], zs[0], zs[-1]], aspect='auto')
        ax[k].plot(apos[:, 0], apos[:, 2], 'r.', ms=3)
        ax[k].set_title(tt, fontsize=10); fig.colorbar(im, ax=ax[k], shrink=0.8)
    fig.suptitle(f'{mol} RAW SPLIT (no fitting) — v_S must carry wall, residual must be smooth')
    fp = os.path.join(outdir, 'raw_split_xz.png')
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'REVIEW: {fp}')

    # ═══ (a) per-atom oracle panels ═══
    nat_show = min(6, len(apos))
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for k in range(6):
        axx = ax[k // 3, k % 3]
        if k >= nat_show:
            axx.axis('off'); continue
        i = k
        dp = pts - apos[i]; rr = np.linalg.norm(dp, axis=1)
        cone = (dp[:, 2] / np.maximum(rr, 1e-9) > 0.3) & (rr < float(orc.r_b[i]))
        axx.scatter(rr[cone], E_pts[cone], s=0.3, c='0.8', rasterized=True)
        r_fine = np.linspace(0.2, float(orc.r_b[i]) + 0.8, 400)
        if np.isfinite(orc.r_min[i]):
            rp, vp = orc.r_prof[i], orc.v_prof[i]
            ok = np.isfinite(vp)
            axx.plot(rp[ok], vp[ok], 'b-', lw=1.5, label='v_i oracle')
            a0, a2, a4, a6 = orc.paw[i]
            P = a0 + a2 * r_fine**2 + a4 * r_fine**4 + a6 * r_fine**6
            axx.plot(r_fine, P, 'c-', lw=1.2, label='P_i (mesh part)')
            axx.plot(r_fine, orc.vs_atom(i, r_fine), 'r-', lw=1.5, label='v_S,i')
            phi, _ = core_basis(r_fine, orc.r_lo[i], orc.r_b[i])
            axx.plot(r_fine, phi @ fit.coeffs[i], 'm--', lw=1.2, label='core fit')
            axx.axvline(orc.r_min[i], color='0.4', ls=':')
            axx.text(orc.r_min[i], 0, 'r_min', rotation=90, fontsize=7)
        axx.axvline(float(orc.r_lo[i]), color='0.4', ls='--', lw=0.7)
        axx.axvline(float(orc.r_b[i]), color='0.4', ls='--', lw=0.7)
        axx.set_yscale('symlog', linthresh=1e-2)
        axx.set_ylim(-0.3, 60)
        axx.set_title(f'atom {i} ({enames[i]}) r_min={orc.r_min[i]:.2f}', fontsize=9)
        if k == 0:
            axx.legend(fontsize=8)
    fig.suptitle(f'{mol} per-atom oracle decomposition (in-cone samples grey)')
    fp = os.path.join(outdir, 'atom_oracles.png')
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'REVIEW: {fp}')

    # ═══ (b) vertical rays over atoms: E_true vs mesh vs core vs total ═══
    from spammm.surfaces.CoarseMesh import _prefilter_3d
    coeffs_mesh = _prefilter_3d(resid_nodes.reshape(ns)).astype(np.float32)
    def mesh_eval(q):                                   # coeffs already prefiltered!
        g = (q - lo) / h
        return map_coordinates(coeffs_mesh.astype(np.float64), g.T, order=3,
                               mode='nearest', prefilter=False)
    # soft-clamp split handled in the slice_xz section (per-cap prefilter there)
    cap_k = args.cap_k
    pick = sorted(set(min(i, len(apos) - 1) for i in (0, 2, 4, len(apos) - 1)))
    fig, ax = plt.subplots(1, len(pick), figsize=(4 * len(pick), 5), sharey=True)
    for j, i in enumerate(pick):
        z = np.linspace(0.3, 6.5, 300)
        q = np.stack([np.full_like(z, apos[i, 0]), np.full_like(z, apos[i, 1]), z], axis=1).astype(np.float32)
        E_t, _ = sample(q)
        E_m = mesh_eval(q)
        E_c, _ = eval_core_fast(q, apos, fit)
        axx = ax[j]
        axx.plot(E_t, z, 'k-', lw=1.6, label='E true')
        axx.plot(E_m, z, 'b-', lw=1.4, label='mesh only')
        axx.plot(E_c, z, 'r-', lw=1.4, label='core only')
        axx.plot(E_m + E_c, z, 'm--', lw=1.4, label='total')
        axx.axvline(0, color='0.7', lw=0.5)
        axx.set_xscale('symlog', linthresh=1e-2)
        axx.set_xlim(-0.3, 50)
        axx.set_title(f'over atom {i} ({enames[i]})', fontsize=9)
        if j == 0:
            axx.set_ylabel('z above atom [Å]'); axx.legend(fontsize=8)
    fig.suptitle(f'{mol} vertical rays — mesh+core total vs FDBM')
    fp = os.path.join(outdir, 'rays_z.png')
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'REVIEW: {fp}')

    # ── same but lateral ray at fixed z (over the atom, along +x) ──
    fig, ax = plt.subplots(1, len(pick), figsize=(4 * len(pick), 5), sharey=True)
    z_fix = mol_z + 3.2
    for j, i in enumerate(pick):
        x = np.linspace(apos[i, 0] - 3.0, apos[i, 0] + 3.0, 240)
        q = np.stack([x, np.full_like(x, apos[i, 1]), np.full_like(x, z_fix)], axis=1).astype(np.float32)
        E_t, _ = sample(q)
        E_m = mesh_eval(q); E_c, _ = eval_core_fast(q, apos, fit)
        axx = ax[j]
        axx.plot(x - apos[i, 0], E_t, 'k-', lw=1.6, label='E true')
        axx.plot(x - apos[i, 0], E_m, 'b-', lw=1.4, label='mesh')
        axx.plot(x - apos[i, 0], E_c, 'r-', lw=1.4, label='core')
        axx.plot(x - apos[i, 0], E_m + E_c, 'm--', lw=1.4, label='total')
        axx.set_xlim(-3, 3); axx.set_ylim(-0.2, 1.5)
        axx.set_title(f'z={z_fix:.1f} x-ray atom {i}', fontsize=9)
        if j == 0:
            axx.legend(fontsize=8)
    fig.suptitle(f'{mol} lateral rays at z={z_fix:.1f} — total vs FDBM')
    fp = os.path.join(outdir, 'rays_x.png')
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'REVIEW: {fp}')

    # ═══ (c) xz slice maps — E_true | E_hard (cores) | E_soft (E−hard) | mask ═══
    nx_s, nz_s = 160, 120
    xs = np.linspace(qb[0, 0] + 1, qb[0, 1] - 1, nx_s)
    zs = np.linspace(mol_z + 0.2, mol_z + 6.0, nz_s)
    XX, ZZ = np.meshgrid(xs, zs, indexing='ij')
    y_mid = float(np.median(apos[:, 1]))
    q = np.stack([XX.ravel(), np.full(XX.size, y_mid), ZZ.ravel()], axis=1).astype(np.float32)
    E_t, _ = sample(q)
    r_min_q = np.linalg.norm(q[:, None, :] - apos[None, :, :], axis=-1).min(axis=1)
    # ═══ SOFT-CLAMP SPLIT: E_soft = soft_clamp_rational -> mesh; E_hard -> radial cores ═══
    # scale by E_min (basin floor): y1=E_min-ish, y2=-2*E_min; plot range +/-2|E_min|
    VS = 2.0 * abs(E_min)
    if args.fit_mode == 'coremesh' or (args.e_cap == 'auto' and args.y2 == 'auto'):
        split_list = [(y1_c, y2_c)]
    else:
        split_list = [(float(a), float(b)) for a in args.e_cap.split(',') for b in args.y2.split(',')]
    for e_cap_i, y2 in split_list:
        if args.fit_mode == 'clamp':
            f_i = lambda E, _y1=e_cap_i, _y2=y2: afm_utils.soft_clamp_rational(np.asarray(E, np.float64), _y1, _y2)[0]
            tag = f'y1{e_cap_i:g}_y2{y2:g}'
            if (e_cap_i, y2) != (y1_c, y2_c):           # refit cores for this config
                fh2 = fit_cores_from_samples(apos, pts, E_pts - f_i(E_pts), sp.r_lo, sp.r_b, bPrint=False)
                cn_h, _ = eval_core_fast(nodes, apos, fh2)
                E_hard = eval_core_fast(q, apos, fh2)[0]
            else:
                cn_h = core_nodes_h
                E_hard = eval_cores(q)
            cs = _prefilter_3d(g_resid(E_nodes - cn_h).reshape(ns)).astype(np.float64)
            E_resid_mesh = map_coordinates(cs, ((q - lo) / h).T, order=3, mode='nearest', prefilter=False)
            tgt_panel, tgt_title = E_t - f_i(E_t), 'E hard target = E - f(E)'
        else:                                          # coremesh — single config, no sweep
            tag = f'ecut{e_cut:g}_etop{e_top:g}_ss{ssm}_lam{args.lam:g}_sw{args.sweeps}'
            E_hard = eval_cores(q)
            E_resid_mesh = mesh_eval(q)
            tgt_panel, tgt_title = T_of(E_t) - E_hard, 'mesh target = T - cores'
            # DEBUG baselines on the same slice: reachable region only (E<e_cut)
            g_q = ((q - lo) / h).T
            reach_q = E_t < e_cut
            for nm_b, cc_b, add_core in (('spline-only LSQ', cs_mo, False), ('spline-only interp', cs_mo1, False), ('cores+mesh', cs_c, True)):
                Eb = map_coordinates(cc_b, g_q, order=3, mode='nearest', prefilter=False) + (E_hard if add_core else 0.0)
                eb = (Eb - E_t)[reach_q]
                lines.append(f'BASELINE {nm_b:20s} reach(E<{e_cut}) err rms={np.sqrt(np.mean(eb**2)):.5f} max={np.abs(eb).max():.5f}')
                print(lines[-1], flush=True)
                for e0, e1 in ((-1.0, 0.0), (0.0, 0.05), (0.05, 0.15), (0.15, e_cut)):
                    mb = (E_t >= e0) & (E_t < e1)
                    if mb.any():
                        eb2 = (Eb - E_t)[mb]
                        print(f'    {nm_b:20s} E in [{e0:5.2f},{e1:5.2f}) n={mb.sum():5d} rms={np.sqrt(np.mean(eb2**2)):.5f} max={np.abs(eb2).max():.5f}', flush=True)
        E_tot = E_resid_mesh + E_hard
        err = E_tot - E_t
        Fs = [np.clip(E_t, -VS, VS), np.clip(E_resid_mesh, -VS, VS), np.clip(E_hard, -VS, VS),
              np.clip(E_tot, -VS, VS), np.clip(err, -0.05, 0.05), np.clip(tgt_panel, -VS, VS)]
        titles = ['E true', 'mesh part', 'cores',
                  'E total = mesh+cores', 'err total-true +/-0.05', tgt_title]
        fig, ax = plt.subplots(2, 3, figsize=(18, 9))
        for k, (tt, F) in enumerate(zip(titles, Fs)):
            axx = ax[k // 3, k % 3]
            v0, v1 = ((-0.05, 0.05) if 'err' in tt else (-VS, VS))
            im = axx.imshow(F.reshape(XX.shape).T, origin='lower', cmap='bwr', vmin=v0, vmax=v1,
                            extent=[xs[0], xs[-1], zs[0], zs[-1]], aspect='auto')
            axx.plot(apos[:, 0], apos[:, 2], 'r.', ms=3)
            axx.set_title(tt, fontsize=10)
            fig.colorbar(im, ax=axx, shrink=0.8)
        fig.suptitle(f'{mol} xz y={y_mid:.2f} — cores-first ({tag}): mesh holds g(E-cores); range +/-2|E_min|={VS:.3f}')
        fp = os.path.join(outdir, f'slice_xz_{tag}.png')
        fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
        print(f'REVIEW: {fp}')
        for lo_r, hi_r in ((2.0, 3.0), (3.0, 4.0), (4.0, 6.0)):
            m = (r_min_q >= lo_r) & (r_min_q < hi_r)
            lines.append(f'{tag} r_min {lo_r}-{hi_r}: total err rms={np.sqrt(np.mean(err[m]**2)):.4f} max={np.abs(err[m]).max():.4f}')
        mesh_err_last = err
    mesh_err = mesh_err_last                            # last config's err for band metrics below

    # ═══ 1D — 4 panels over ONE atom: split | resid->mesh fit | core fit | total ═══
    E_lim = 2.0 * abs(E_min)                          # same window as the 2D panels
    y1_show, y2_show = split_list[0]
    f_show = lambda E: afm_utils.soft_clamp_rational(np.asarray(E, np.float64), y1_show, y2_show)[0]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    i = pick[0]
    z = np.linspace(1.0, 7.0, 800)
    q1 = np.stack([np.full_like(z, apos[i, 0]), np.full_like(z, apos[i, 1]), z], axis=1).astype(np.float32)
    E_r, _ = sample(q1)
    x = z - apos[i, 2]
    H_c = eval_cores(q1)
    R_r = E_r - H_c                                   # raw residual after cores
    M_r = mesh_eval(q1)
    T_fit = H_c + M_r
    if args.fit_mode == 'clamp':
        mesh_tgt = g_resid(R_r)                       # clamped residual -> mesh target
        t2_lbl = f'g(resid) ry1={ry1:.3f}'; ctgt = E_r - f_show(E_r); c3_lbl = 'core target = E - f(E)'
        r3 = (H_c - ctgt) * 100
    else:                                             # coremesh
        T_r = T_of(E_r)
        mesh_tgt = T_r - H_c                          # what the mesh was fitted to
        ctgt = T_r - M_r                              # what the cores must cover
        t2_lbl = 'mesh target = T - cores'; c3_lbl = 'core target = T - mesh'
        r3 = (H_c - ctgt) * 100
    i_min = int(np.argmin(E_r)); i_y1 = int(np.argmin(np.abs(E_r - y1_show)))
    for axx in ax:                                    # markers + window on all panels
        axx.axvline(x[i_min], color='0.4', ls='-', lw=0.8, label='attr. min')
        axx.axvline(x[i_y1], color='purple', ls='--', lw=0.8, label='E=y1 split')
        axx.axhline(0, color='0.7', lw=0.5)
        axx.set_xlim(1.5, 6.5); axx.set_ylim(-E_lim, E_lim)
        axx.set_xlabel('r above atom [Å]')
    ax[0].plot(x, E_r, 'k-', lw=1.2, label='E total')
    ax[0].plot(x, H_c, 'r-', lw=1.0, label='cores')
    ax[0].plot(x, R_r, 'b-', lw=1.0, label='resid = E - cores')
    ax[0].set_title('1) cores + resid split'); ax[0].set_ylabel('E [eV]')
    ax[1].plot(x, R_r, color='0.6', ls=':', lw=0.8, label='resid raw')
    ax[1].plot(x, mesh_tgt, 'b-', lw=1.0, label=t2_lbl)
    ax[1].plot(x, M_r, 'c-', lw=1.0, label='mesh eval')
    ax[1].plot(x, (M_r - mesh_tgt) * 100, color='0.3', lw=0.8, label='mesh resid x100')
    ax[1].set_title('2) mesh fit')
    ax[2].plot(x, ctgt, 'r-', lw=1.0, label=c3_lbl)
    ax[2].plot(x, H_c, 'm-', lw=1.0, label='cores')
    ax[2].plot(x, r3, color='0.3', lw=0.8, label='resid x100')
    ax[2].set_title('3) hard fit (cores)')
    ax[3].plot(x, E_r, 'k-', lw=1.2, label='total ref')
    ax[3].plot(x, T_fit, 'm--', lw=1.0, label='total fit')
    ax[3].plot(x, M_r, 'b-', lw=1.0, label='mesh part')
    ax[3].plot(x, H_c, 'r-', lw=1.0, label='cores')
    ax[3].plot(x, (T_fit - E_r) * 100, color='0.3', lw=0.8, label='err x100')
    ax[3].set_title('4) total')
    for axx in ax:
        axx.legend(fontsize=8, loc='upper right', framealpha=0.9)   # AFTER labeled lines
    stit = (f'{mol} 1D cores-first over atom {i} ({enames[i]}) — y1={y1_show:.3f} y2={y2_show:.3f} '
            f'ry1={ry1:.3f} ry2={ry2:.3f}' if args.fit_mode == 'clamp' else
            f'{mol} 1D coremesh over atom {i} ({enames[i]}) — e_cut={e_cut:.3f} e_top={e_top:.3f} '
            f'ss={ssm} lam={args.lam:g} sweeps={args.sweeps}')
    fig.suptitle(stit + f'; window +/-2|E_min|={E_lim:.3f}')
    fp = os.path.join(outdir, 'split_1d_clamp.png')
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'REVIEW: {fp}')

    # ═══ (d) the mesh residual itself — is E − Σv_S smooth? ═══
    vs_s, cov_s = eval_vs_oracle(q, apos, orc)
    resid_slice = (E_t - vs_s).reshape(XX.shape)
    E_t2 = E_t.reshape(XX.shape)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    im = ax[0].imshow(np.clip(E_t2, -0.5, 5).T, origin='lower', cmap='viridis',
                      extent=[xs[0], xs[-1], zs[0], zs[-1]], aspect='auto')
    ax[0].set_title('E true'); fig.colorbar(im, ax=ax[0], shrink=0.8)
    im = ax[1].imshow(np.clip(resid_slice, -0.5, 5).T, origin='lower', cmap='viridis',
                      extent=[xs[0], xs[-1], zs[0], zs[-1]], aspect='auto')
    ax[1].set_title('residual E − Σv_S (what mesh holds)'); fig.colorbar(im, ax=ax[1], shrink=0.8)
    im = ax[2].imshow(np.clip(resid_slice, -0.05, 0.05).T, origin='lower', cmap='bwr',
                      extent=[xs[0], xs[-1], zs[0], zs[-1]], aspect='auto')
    ax[2].set_title('residual zoom ±0.05'); fig.colorbar(im, ax=ax[2], shrink=0.8)
    for k in range(3):
        ax[k].plot(apos[:, 0], apos[:, 2], 'r.', ms=3)
    # overlay r_min coverage boundary of first atoms
    fig.suptitle(f'{mol} residual smoothness check (xz y={y_mid:.2f})')
    fp = os.path.join(outdir, 'residual_xz.png')
    fig.savefig(fp, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'REVIEW: {fp}')

    # metrics by radial band — pure mesh-interp error on the soft part
    r_min_all = r_min_q
    e1 = mesh_err.ravel()
    for lo_r, hi_r in ((0.5, 2.3), (2.3, 3.2), (3.2, 4.5), (4.5, 8.0)):
        m = (r_min_all >= lo_r) & (r_min_all < hi_r)
        lines.append(f'r_min {lo_r}-{hi_r}: total err rms={np.sqrt(np.mean(e1[m]**2)):.4f} '
                     f'max={np.abs(e1[m]).max():.4f} (n={m.sum()})')

    with open(os.path.join(outdir, 'DEBUG.out'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
