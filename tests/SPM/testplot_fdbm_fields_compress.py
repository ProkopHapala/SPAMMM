"""FDBM → compact-field compression driven by FDBMPipeline.run_fields.

Two compression methods on the same device-resident field (img_FF_fdbm +
sample_fdbm oracle — no F_total host download):

  ⚠ STATUS 2026-10-08: the PAW split (--method pme default) is BROKEN for
  FDBM fields — df_corr 0.1-0.8 at every h_mesh, worse than zero core. It
  was only ever validated on the analytic Morse oracle (a literal sum of
  radial per-atom potentials). This script is a DEBUGGING harness for the
  split, not a validated pipeline. Production dataset path: invPPAFM
  `pme_dataset.py --scan fdbm` (raw FDBM image, no compression).

  --method pme   ContactPME PAW split (DEFAULT, real PME): field oracle —
                 per-atom radial v_i(r) extracted by damped-Jacobi decomposition
                 of sampled field (fit_cores_paw_field), then the production
                 split verbatim: v_S,i = v_i − P_i (even PAW poly C²-matched at
                 r_b), per-atom 5×5 core fit (same math as cs_fit_core_paw GPU
                 kernel), mesh = _prefilter_3d of E − Σv_S on a COARSE mesh
                 (0.35-1.0 A); the un-sampled deep region + |resid|>cap nodes
                 get a smooth harmonic fill (PAW pseudoization analog).
                 -> ContactPMEParams -> run_scan_contact_pme.
                 --no-core: legacy zero-core dense mesh (needs h<=0.2 to avoid
                 sharp-dot artifacts over atoms — the mesh can't carry the wall).
  --method sep   separable 2.5D contact fit (SeparableParams, B-spline XY x
                 poly z modes) -> fit_contact_field CG (~2 min/mol — DEPRECATED
                 slow path, kept for size comparison).
  --method coremesh  poly8 core + coarse mesh: compact radial cores
                 phi=(1-r/R)_+^8 on atoms + bond midpoints (slots of the global
                 ladder R=[9,5.612,3.5,1,1], atoms {0,1,2} bonds {0,2}), plain
                 LSQ on the shell 2.5<r_min<4.0 (probe side z>mol_z+0.5), then a
                 weighted P-spline mesh (fit_coremesh_lsq) on E-core for
                 r_min>2.5 (no outer cut). Kernel-side -DCS_PME_POLY_CORE=1.
                 Also scans a 'spline_only' baseline (same weighted mesh on full
                 E, zero core) and writes compare_pp_dxy.png (relaxed-PP lateral
                 displacement of FDBM vs both fits).

Per molecule: azaindol + PTCDA; ONE FDBMPipeline shared across the batch.
Validation: holdout E/F vs oracle, serialization round-trip, rescan df/Fz corr.
Outputs per mol+method: <label>.npz/pkl, compare_df_Fz.png, SUMMARY.out,
comparison_arrays.npz under debug/testplot_fdbm_fields_compress/<mol>/.
"""

import os
import re
import sys
import time
import numpy as np

OUTROOT = 'debug/testplot_fdbm_fields_compress'
SCAN_MARGIN = 2.0
H_MESH = 0.35        # PME mesh spacing [A] (pme_dataset H_MESH_FDBM)
MESH_HALO = 6        # halo nodes per side (production convention)
BOND_L = 3.0         # probe bond length [A]


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    sa, sb = np.std(a), np.std(b)
    return float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else np.nan


def rmse(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _variant_figure(a_util, variants, row_specs, h_Fz, scan_xs, scan_ys, figure, title):
    row_specs_full = row_specs + [('Fz', k, f'{k} Fz', 'bwr') for k in variants]
    a_util.plot_afm_variant_height_strip(variants, row_specs_full, h_Fz, figure,
                                         title=title, extent=a_util.scan_extent(scan_xs, scan_ys),
                                         amp=None, amp_align=False, dpi=150)


def compress_mol(pipe, mol_name, xyz_file, step, method='pme', h_mesh=H_MESH,
                 bspl_steps=(0.8,), n_iter=1200, z_modes=8, poly_z0=2.2, poly_R=8.0,
                 no_core=False, n_shell=6000, n_cloud=32768,
                 delta_in=1.0, delta_b=1.0, z_fit_dz=1.2, resid_cap=10.0, lam=1e-3,
                 core_basis='poly8', core_wF=0.3, core_spec='A:s3p2+B:s1', core_solver='gpu', core_ridge=1e-8,
                 mesh_fit='inpaint', skip_spline_only=False):
    from spammm import atomicUtils as au
    from spammm.SPM import AFM as afm, AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    from spammm.forcefields.FFController import make_planar_xy, orient_long_axis_x

    tag = method
    if method == 'coremesh':
        spec_tag = '' if core_basis == 'poly8' else '-' + re.sub(r'[^A-Za-z0-9]+', '', core_spec)
        tag = f'coremesh_{core_basis}{spec_tag}' + ('' if mesh_fit == 'inpaint' else f'_{mesh_fit}')
    outdir = os.path.join(OUTROOT, f'{mol_name}_s{step:g}_{tag}{h_mesh if method != "sep" else bspl_steps[0]:g}')
    os.makedirs(outdir, exist_ok=True)
    atomPos, _, enames, _, _ = au.load_xyz(xyz_file)
    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16}
    atomTypes = np.array([ELEM_Z[e] for e in enames], dtype=np.int32)
    atomPos = np.asarray(atomPos, dtype=np.float64)
    atomPos[:] = make_planar_xy(atomPos)
    orient_long_axis_x(atomPos)
    atomPos[:, 2] = 0.0
    mol_z = float(atomPos[:, 2].max())

    scan_dx = 0.1          # scan raster stays fine regardless of field step
    scan_xs = np.arange(atomPos[:, 0].min() - SCAN_MARGIN, atomPos[:, 0].max() + SCAN_MARGIN, scan_dx, dtype=np.float32)
    scan_ys = np.arange(atomPos[:, 1].min() - SCAN_MARGIN, atomPos[:, 1].max() + SCAN_MARGIN, scan_dx, dtype=np.float32)
    h_df, h_Fz, h_scan = afm_utils.afm_df_height_stacks(3.7, 4.7, 0.1, amp=1.0, amp_align=True)

    # ── query bounds covering full PP trajectory (production qb formula) ──
    R = float(max(abs(scan_xs[0]), abs(scan_xs[-1]), abs(scan_ys[0]), abs(scan_ys[-1]))) + 1.0
    qb = np.array([[-R, R], [-R, R], [mol_z - 2.0, mol_z + float(h_scan.max()) + BOND_L + 1.5]])
    m_xy = R + MESH_HALO * h_mesh + 0.5
    z_vac = float(max(qb[2, 1] - mol_z, mol_z - qb[2, 0])) + MESH_HALO * h_mesh + 0.5
    grid_spec, origin, ngrid, step = afm_utils.make_fdbm_grid_com_zsym(atomPos, step, m_xy, z_vac=z_vac)
    print(f"\n{'='*90}\n{mol_name}  natoms={len(enames)}  grid={ngrid} ({np.prod(ngrid)/1e6:.2f}Mvox)  "
          f"step={step}  qb={qb.tolist()}\n{'='*90}")

    # ── 1. fields via shared pipeline (device-resident oracle) ──
    basis_hsd = get_dftb_basis_path('3ob-3-1')
    pa = afm.PAULI_FITTED_DEFAULTS['3ob-3-1']
    t0 = time.perf_counter()
    res = pipe.run_fields(atomPos, atomTypes, basis_hsd, os.path.join(outdir, 'gpu_work'),
                          grid_spec, origin, step, ngrid, float(pa['A']), float(pa['beta']), 'co', oracle=True)
    a = pipe.ensure_afm()
    a.queue.finish()
    t_fields = time.perf_counter() - t0
    sample = res['prolonged']['sample']
    dense_bytes = int(np.prod(ngrid)) * 16
    print(f'run_fields: {t_fields:.2f}s  (dense field would be {dense_bytes/1e6:.1f} MB — never on host)')

    # ── 2. holdout + reference dense scan ──
    xlo, xhi = qb[0]; ylo, yhi = qb[1]
    rng = np.random.default_rng(20261002)
    holdout = rng.uniform([xlo + 1.0, ylo + 1.0, mol_z + 2.5], [xhi - 1.0, yhi - 1.0, mol_z + 6.8], (8192, 3)).astype(np.float32)
    E_hold, F_hold = sample(holdout)
    K_lat, K_rad = afm.stiffness_Nm_to_eVA2(0.5), 20.0
    t0 = time.perf_counter()
    FEs_ref, disp_ref = a.scan_fdbm(scan_xs, scan_ys, h_scan, mol_z=mol_z,
                                    K_LAT=K_lat, K_RAD=K_rad, bond_length=BOND_L, use_fire=True)
    t_scan_ref = time.perf_counter() - t0
    idx_df = np.array([int(np.argmin(abs(h_scan - h))) for h in h_df])
    idx_Fz = np.array([int(np.argmin(abs(h_scan - h))) for h in h_Fz])
    df_ref = afm.compute_df_amp(FEs_ref[..., 2], 0.1, amp=1.0)[..., idx_df]
    Fz_ref = FEs_ref[..., 2][..., idx_Fz]
    print(f'reference scan {t_scan_ref:.2f}s')

    a.stiffness = np.array([-K_lat, -K_lat, -K_lat, -K_rad], dtype=np.float32)
    a.dpos0 = np.array([0., 0., -BOND_L, BOND_L], dtype=np.float32)
    a.surfFF[:] = 0.0
    scan_p0 = np.array([scan_xs[0], scan_ys[0], mol_z + h_scan[-1] + BOND_L], dtype=np.float32)
    scan_da = np.array([float(scan_xs[1] - scan_xs[0]), 0., 0.], dtype=np.float32)
    scan_db = np.array([0., float(scan_ys[1] - scan_ys[0]), 0.], dtype=np.float32)

    variants = {'FDBM': {'df': df_ref, 'Fz': Fz_ref}}
    row_specs = [('df', 'FDBM', 'FDBM df', 'gray')]
    lines = [f'FDBM→{method} compression via FDBMPipeline.run_fields (device-resident; no F_total host copy)',
             f'molecule={mol_name} natoms={len(enames)} grid={ngrid} step={step}',
             f'dense_float4_bytes={dense_bytes} qb={qb.tolist()}',
             f'run_fields_sec={t_fields:.4f} reference_scan_sec={t_scan_ref:.4f}',
             f'K_lat_Nm=0.5 K_rad=20 L=3 amp=1 h_Fz={h_Fz.tolist()} h_df={h_df.tolist()}',
             'Status: experimental; USER visual review pending']
    snapshots = dict(holdout=holdout, E_hold=E_hold, F_hold=F_hold,
                     scan_xs=scan_xs, scan_ys=scan_ys, h_df=h_df, h_Fz=h_Fz, h_scan=h_scan,
                     df_ref=df_ref, Fz_ref=Fz_ref, FEs_ref=FEs_ref,
                     atomPos=atomPos, origin=origin, grid_step=step, qb=qb)

    label = f'pme_{h_mesh:g}' if no_core else f'pme_core_{h_mesh:g}'
    t_fit = scan_sec = np.nan
    if method == 'pme':
        from spammm.surfaces.ContactSurface import cpm_params_from_samples
        from spammm.surfaces.PMESplit import SplitParams
        from spammm.surfaces.PICCore import sample_core_shells
        import pickle
        # ── 3a. PME mesh: sample E at nodes via GPU oracle + direct prefilter ──
        h = float(h_mesh); halo = MESH_HALO
        lo = qb[:, 0] - halo * h; hi = qb[:, 1] + halo * h
        ns = np.round((hi - lo) / h).astype(int) + 1
        xs_m = lo[0] + np.arange(ns[0]) * h
        ys_m = lo[1] + np.arange(ns[1]) * h
        zs_m = lo[2] + np.arange(ns[2]) * h
        X, Y, Z = np.meshgrid(xs_m, ys_m, zs_m, indexing='ij')
        nodes = np.ascontiguousarray(np.stack([X, Y, Z], axis=-1).reshape(-1, 3), dtype=np.float32)
        t0 = time.perf_counter()
        E_nodes, _ = sample(nodes)
        assert np.isfinite(E_nodes).all(), 'mesh node outside FDBM grid'
        heavy = atomTypes > 1                     # still used for labels
        apos_h = atomPos                           # PAW cores on ALL atoms (production convention)
        _, cLJs = afm_utils._morse_atoms_from_Z(apos_h, atomTypes)
        if no_core:
            cpm = cpm_params_from_samples(E_nodes.reshape(ns), lo, h, halo, apos_h, cLJs)
        else:
            # ── real PAW split, field oracle: per-atom radial v_i(r) extracted
            # by damped-Jacobi decomposition of the sampled field (all atoms
            # contribute — Σ_{j≠i} ṽ_j subtracted before profiling), then the
            # production construction verbatim: P_i = even PAW poly C²-matched
            # at r_b, v_S,i = v_i − P_i (per-atom 5×5 like cs_fit_core_paw),
            # mesh = E − Σv_S (smooth where sampled). Inside the un-sampled
            # deep region (and wherever |resid| > cap) the mesh holds a smooth
            # harmonic continuation — the FDBM analog of "mesh = Σv_L inside". ──
            from spammm.surfaces.PICCore import (fit_cores_paw_field, eval_vs_oracle,
                                                 inpaint_residual)
            alpha = float(abs(cLJs[0, 2]))
            z_fit = mol_z + z_fit_dz
            sp = SplitParams(R0=cLJs[:, 0].astype(np.float64), E0=cLJs[:, 1].astype(np.float64),
                             q=np.zeros(len(apos_h)), alpha=alpha, q_tip=0.0, r_damp=0.1,
                             r_cut=6.0, split_mode='paw', delta_in=delta_in, delta_b=delta_b)
            shell = sample_core_shells(apos_h, sp.r_lo, sp.r_b, n_per_atom=n_shell, seed=7)
            shell = shell[shell[:, 2] > z_fit]
            cloud = rng.uniform([qb[0, 0], qb[1, 0], z_fit], qb[:, 1], (n_cloud, 3))
            pts = np.concatenate([shell, cloud]).astype(np.float32)
            E_pts, _ = sample(pts)
            snapshots['pts'] = pts; snapshots['E_pts'] = E_pts
            fit, orc = fit_cores_paw_field(apos_h, pts, E_pts, sp, bPrint=True)
            vs_nodes, cov_nodes = eval_vs_oracle(nodes, apos_h, orc)
            resid_nodes = E_nodes - vs_nodes
            deep = (cov_nodes < 0.0) | (np.abs(resid_nodes) > resid_cap)
            from scipy.ndimage import binary_dilation
            deep = binary_dilation(deep.reshape(ns), iterations=1).ravel()
            resid_nodes = inpaint_residual(resid_nodes, ns, deep, n_iter=400)
            cpm = cpm_params_from_samples(resid_nodes.reshape(ns), lo, h, halo,
                                          apos_h, cLJs, core_fit=fit, split_params=sp)
            snapshots['core_coeffs'] = fit.coeffs
        a.cpm = cpm; a._cpm_upload_id = None; a._cpm_coeffs0 = None
        a.pme_set_field(None)
        t_fit = time.perf_counter() - t0
        blob = pickle.dumps(cpm)
        archive = os.path.join(outdir, f'{label}.pkl')
        with open(archive, 'wb') as f:
            f.write(blob)
        arch_bytes = len(blob)
        # serialization round-trip parity
        cpm2 = pickle.loads(blob)
        E_fit, F_fit = a.eval_contact_pme(holdout, params=cpm2)
        assert np.isfinite(E_fit).all() and np.isfinite(F_fit).all()
        # ── 4a. PME rescan ──
        t0 = time.perf_counter()
        FEs_fit, _ = a.run_scan_contact_pme(nxy=(len(scan_xs), len(scan_ys)), nz=len(h_scan),
                                            dtip=-0.1, scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db,
                                            core_backend='local')
        scan_sec = time.perf_counter() - t0
        FEs_fit = FEs_fit[:, :, ::-1, :]
        assert np.isfinite(FEs_fit).all() and float(np.std(FEs_fit[..., 2])) > 1e-8
        df_fit = afm.compute_df_amp(FEs_fit[..., 2], 0.1, amp=1.0)[..., idx_df]
        Fz_fit = FEs_fit[..., 2][..., idx_Fz]
        variants[label] = {'df': df_fit, 'Fz': Fz_fit}
        row_specs.append(('df', label, f'PME h={h:g} df', 'gray'))
        snapshots.update({f'{label}_E_hold': E_fit, f'{label}_F_hold': F_fit,
                          f'{label}_df': df_fit, f'{label}_Fz': Fz_fit, f'{label}_FEs': FEs_fit,
                          'mesh_nodes': nodes, 'mesh_E': E_nodes, 'mesh_ns': ns, 'mesh_lo': lo})
        lines.extend([f'\n[{label}] mesh={tuple(ns)} h={h} halo={halo} '
                      f'{"ZERO-CORE" if no_core else f"PAW cores na={len(apos_h)} r_lo={sp.r_lo.min():.2f}..{sp.r_lo.max():.2f} r_b={sp.r_b.min():.2f}..{sp.r_b.max():.2f} (parametric Morse fit)"}',
                      f'mesh_f32_bytes={cpm.mesh_coeffs.size * 4} resident_bytes={cpm.resident_bytes} '
                      f'archive_pickle_bytes={arch_bytes} dense_ratio={dense_bytes / cpm.resident_bytes:.0f}x',
                      f'mesh_sample+prefilter_sec={t_fit:.2f} pme_scan_sec={scan_sec:.2f}',
                      f'holdout_E_rmse={rmse(E_hold, E_fit):.4e} max={np.abs(E_hold - E_fit).max():.4e}'])
        for c, nm in enumerate(('Fx', 'Fy', 'Fz')):
            lines.append(f'holdout_{nm}_rmse={rmse(F_hold[:, c], F_fit[:, c]):.4e} corr={corr(F_hold[:, c], F_fit[:, c]):.6f}')
        for iz, (hf, hd) in enumerate(zip(h_Fz, h_df)):
            lines.append(f'h_Fz={hf:.2f} h_df={hd:.2f} df_corr={corr(df_ref[..., iz], df_fit[..., iz]):.6f} '
                         f'df_rmse={rmse(df_ref[..., iz], df_fit[..., iz]):.4e} '
                         f'Fz_corr={corr(Fz_ref[..., iz], Fz_fit[..., iz]):.6f} '
                         f'Fz_rmse={rmse(Fz_ref[..., iz], Fz_fit[..., iz]):.4e}')
        print('\n'.join(lines[-(4 + 3 + len(h_Fz)):]), flush=True)

    elif method == 'coremesh':
        # poly8 core (atoms + bond midpoints, phi=(1-r/R)_+^8) fitted on the shell
        # 2.5<r_min<4.0, then a weighted cubic B-spline mesh on E-core for r_min>2.5
        # (NO outer cut). Kernel-side: same contact_pme ABI with -DCS_PME_POLY_CORE=1.
        from spammm.surfaces.PICCore import (POLY_CORE_R, core_centers_atoms_bonds,
                                             fit_core_poly_shell, eval_core_poly)
        from spammm.surfaces.CoarseMesh import fit_coremesh_lsq
        from spammm.surfaces.ContactSurface import cpm_params_from_coremesh
        import pickle, dataclasses
        h = float(h_mesh); halo = MESH_HALO
        s = 4 if h >= 1.0 else 2                                  # supersampling -> sample step h/s
        dg = h / s
        lo = qb[:, 0] - halo * h
        n_mesh = np.round((qb[:, 1] + halo * h - lo) / h).astype(int) + 1
        ns_ss = (n_mesh - 1) * s + 1                              # node-aligned supersampled grid
        grids = np.meshgrid(*[lo[a_] + np.arange(ns_ss[a_]) * dg for a_ in range(3)], indexing='ij')
        pts = np.ascontiguousarray(np.stack(grids, -1).reshape(-1, 3), dtype=np.float32)
        print(f'[coremesh] h={h} s={s} sample grid {tuple(ns_ss)} ({pts.shape[0]} pts) -> mesh nodes {tuple(n_mesh)}', flush=True)
        t0 = time.perf_counter()
        E_ss, F_ss = sample(pts)
        assert np.isfinite(E_ss).all(), 'coremesh sample grid outside FDBM grid'
        t_samp = time.perf_counter() - t0
        z_min = mol_z + 0.5
        t1 = time.perf_counter()
        cg_info = None
        if core_basis == 'poly8sp':
            # angular sp core (default spec A:s3p2+B:s1 -> 9 slots/center, kernel CS_PME_SP_CORE)
            from spammm.surfaces.PICCore import fit_core_sp_shell, eval_core_sp, sp_ladder
            Rlab, _ = sp_ladder(core_spec)
            if core_solver in ('gpu', 'gpugram'):
                # the GPU solvers run in the kernel slot layout -> SP-compiled
                # program must exist BEFORE the fit (the later call is a no-op)
                a.set_pme_core_basis('poly8sp', poly_R=Rlab)
            fit, centers = fit_core_sp_shell(atomPos, pts=pts, E=E_ss, F=F_ss, wF=core_wF, rin=2.5, rcut=4.0, z_min=z_min, spec=core_spec, solver=core_solver, afm=a, ridge=core_ridge)
            cg_info = getattr(fit, 'cg_info', None)
            if core_solver in ('gpu', 'gpugram'):
                E_core, _ = a._cs_fit_helper().eval_core_sp_gpu(pts)
            else:
                E_core, _ = eval_core_sp(pts, centers, fit.coeffs, fit.r_lo, fit.poly_R)
            n_bond = len(centers) - len(atomPos)
            Rlab = fit.poly_R
        else:
            centers, is_bond = core_centers_atoms_bonds(atomPos, dbond=1.6)
            fit = fit_core_poly_shell(atomPos, centers, is_bond, pts=pts, E=E_ss, rin=2.5, rcut=4.0, z_min=z_min)
            E_core, _ = eval_core_poly(pts, centers, fit.coeffs)
            n_bond = int(is_bond.sum())
            Rlab = POLY_CORE_R
        resid = (E_ss.astype(np.float64) - E_core).reshape(ns_ss)
        from spammm.surfaces.PICCore import min_dist_to_atoms, inpaint_residual
        t_core = time.perf_counter() - t1

        def mesh_fit_lsq(target3):
            """Fit mesh on node grid. 'inpaint': omit too-close cells (rm<2.5 or below
            z_min) by harmonic fill (PAW recipe) + separable banded solve — no CG.
            'cg': legacy weighted supersampled CG (fit_coremesh_lsq)."""
            if mesh_fit == 'inpaint':
                from scipy.ndimage import binary_dilation, map_coordinates
                from spammm.surfaces.CoarseMesh import fit_mesh_lsq_3d, CoarseMesh
                rn = np.ascontiguousarray(target3[::s, ::s, ::s])          # node samples
                Pn = np.stack(np.meshgrid(*[lo[a_] + np.arange(n_mesh[a_]) * h for a_ in range(3)], indexing='ij'), -1).reshape(-1, 3)
                mask = (min_dist_to_atoms(Pn, atomPos) < 2.5) | (Pn[:, 2] < z_min)
                mask = binary_dilation(mask.reshape(n_mesh), iterations=1)
                rf = inpaint_residual(rn.ravel(), n_mesh, mask).reshape(n_mesh)
                cf = fit_mesh_lsq_3d(rf, s=1, lam=lam)
                ev = map_coordinates(cf, np.indices(n_mesh, dtype=np.float64).reshape(3, -1), order=3, mode='nearest', prefilter=False).reshape(n_mesh)
                wrms = np.sqrt(np.mean((rn - ev)[~mask] ** 2))
                print(f'[coremesh] mesh inpaint nodes: masked {int(mask.sum())}/{mask.size} ({100.0 * mask.mean():.1f}%), node wrms={wrms * 1e3:.3f} meV', flush=True)
                if s > 1:
                    # omitted ss cells: seed with node-fit mesh value, then harmonic
                    # fill -> smooth continuation; refit on the full ss grid (banded)
                    msk_ss = ((min_dist_to_atoms(pts, atomPos) < 2.5) | (pts[:, 2] < z_min)).reshape(ns_ss)
                    msk_ss = binary_dilation(msk_ss, iterations=s // 2)
                    g = ((pts - np.asarray(lo)) / h).T                       # node-fraction coords
                    seed = np.zeros(int(np.prod(ns_ss)))
                    seed[msk_ss.ravel()] = map_coordinates(cf, g[:, msk_ss.ravel()], order=3, mode='nearest', prefilter=False)
                    rf_ss = inpaint_residual(target3.ravel(), ns_ss, msk_ss, n_iter=150, seed=seed).reshape(ns_ss)
                    cf = fit_mesh_lsq_3d(rf_ss, s=s, lam=lam)
                return CoarseMesh(cf, np.asarray(lo, np.float64).copy(), float(h), 2,
                                  (np.full(3, halo, np.int64), np.asarray(n_mesh) - 1 - halo)), dict(iterations=0, wrmse=wrms)
            rm_pts = min_dist_to_atoms(pts, atomPos)
            w = ((rm_pts > 2.5) & (pts[:, 2] > z_min)).reshape(ns_ss).astype(np.float64)
            mesh, _, dg = fit_coremesh_lsq(target3, np.empty((0, 3)), lo, h, np.empty(0), np.empty(0),
                                           s=s, weights=w, lam=lam, tol=1e-7, maxiter=20000)
            return mesh, dg

        tm = time.perf_counter()
        mesh, dg_res = mesh_fit_lsq(resid)
        t_m1 = time.perf_counter() - tm
        print(f'[coremesh] mesh(E-core) {mesh_fit} iters {dg_res["iterations"]} wrms={dg_res["wrmse"]*1e3:.3f} meV ({t_m1:.1f}s)', flush=True)
        cpm = cpm_params_from_coremesh(mesh.coeffs, lo, h, halo, centers, fit)
        tm = time.perf_counter()
        if skip_spline_only:
            mesh_s = None; t_m2 = 0.0
            print('[coremesh] mesh_spl=skipped (--skip-spline-only)', flush=True)
        else:
            mesh_s, dg_spl = mesh_fit_lsq(E_ss.astype(np.float64).reshape(ns_ss))
            t_m2 = time.perf_counter() - tm
            print(f'[coremesh] mesh(spline-only) {mesh_fit} iters {dg_spl["iterations"]} wrms={dg_spl["wrmse"]*1e3:.3f} meV ({t_m2:.1f}s)', flush=True)
        print(f'[timing] sample={t_samp:.2f}s core_fit={t_core:.2f}s mesh1={t_m1:.2f}s mesh2={t_m2:.2f}s' + (f' cg_iters={cg_info["iters"]} cg_sec={cg_info["sec"]:.2f}' if (cg_info is not None and 'iters' in cg_info) else ''), flush=True)
        fit0 = dataclasses.replace(fit, coeffs=np.zeros_like(fit.coeffs))
        cpm0 = cpm_params_from_coremesh(mesh_s.coeffs, lo, h, halo, centers, fit0) if not skip_spline_only else None
        a.set_pme_core_basis(core_basis, poly_R=Rlab)
        t_fit = time.perf_counter() - t0
        blob = pickle.dumps(cpm)
        archive = os.path.join(outdir, f'coremesh_{h:g}.pkl')
        with open(archive, 'wb') as f:
            f.write(blob)
        arch_bytes = len(blob)
        snapshots.update({'coremesh_centers': centers, 'coremesh_coeffs': fit.coeffs,
                          'mesh_ns': n_mesh, 'mesh_lo': lo, 'mesh_coeffs': mesh.coeffs})
        lines.extend([f'\n[coremesh_{h:g}] {core_basis} core+mesh: {len(centers)} centers '
                      f'({n_bond} bonds), R={np.asarray(Rlab).tolist()}, s={s}, lam={lam:g}, wF={core_wF:g}',
                      f'fit breakdown: sample={t_samp:.2f}s core={t_core:.2f}s mesh_res={t_m1:.2f}s mesh_spl={t_m2:.2f}s total={t_fit:.2f}s',
                      f'mesh={tuple(n_mesh)} h={h} halo={halo} resident_bytes={cpm.resident_bytes} '
                      f'archive_pickle_bytes={arch_bytes} dense_ratio={dense_bytes / cpm.resident_bytes:.0f}x',
                      f'fit_sec={t_fit:.2f} (sample+core+2x mesh CG)'])
        nom_xy = np.meshgrid(scan_xs, scan_ys, indexing='ij')

        def scan_variant(label, cpm_v):
            a.cpm = cpm_v; a._cpm_upload_id = None; a._cpm_coeffs0 = None
            a.pme_set_field(None)
            E_g, F_g = a.eval_contact_pme(holdout, params=cpm_v, use_gpu=True, core_backend='local')
            t0 = time.perf_counter()                                  # eval scaling: warm GPU re-eval of the holdout batch
            a.eval_contact_pme(holdout, params=cpm_v, use_gpu=True, core_backend='local')
            a.queue.finish()
            t_eval = (time.perf_counter() - t0) * 1e3 / len(holdout)  # ms per 1k points
            E_py, F_py = a._pme_eval_python(cpm_v, holdout)
            mlow = E_hold < 0.3
            lines.append(f'[{label}] holdout_E_rmse={rmse(E_hold, E_g):.4e} max={np.abs(E_hold - E_g).max():.4e} '
                         f'E<0.3_rmse={rmse(E_hold[mlow], E_g[mlow]):.4e} (n={int(mlow.sum())})')
            for c, nm in enumerate(('Fx', 'Fy', 'Fz')):
                lines.append(f'[{label}] holdout_{nm}_rmse={rmse(F_hold[:, c], F_g[:, c]):.4e} corr={corr(F_hold[:, c], F_g[:, c]):.6f}')
            lines.append(f'[{label}] CPU-GPU parity max|dE|={np.abs(E_g - E_py).max():.3e} max|dF|={np.abs(F_g - F_py).max():.3e}')
            t0 = time.perf_counter()
            FEs_fit, _ = a.run_scan_contact_pme(nxy=(len(scan_xs), len(scan_ys)), nz=len(h_scan),
                                                dtip=-0.1, scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db,
                                                core_backend='local')
            sc = time.perf_counter() - t0
            pp = a.last_pp[:, :, ::-1, :3]                       # kernel iz=0=top -> ascending height
            rm_pp = min_dist_to_atoms(pp.reshape(-1, 3), atomPos)
            dz_pp = pp[..., 2].ravel() - mol_z
            lines.append(f'[{label}] hole guard: relaxed PP min r_min={rm_pp.min():.3f} A  min z-mol_z={dz_pp.min():.3f} A  '
                         f'n(r_min<2.0)={int((rm_pp < 2.0).sum())}  n(r_min<2.0 & z<mol_z+2.0)={int(((rm_pp < 2.0) & (dz_pp < 2.0)).sum())}')
            FEs_fit = FEs_fit[:, :, ::-1, :]
            assert np.isfinite(FEs_fit).all() and float(np.std(FEs_fit[..., 2])) > 1e-8
            df_fit = afm.compute_df_amp(FEs_fit[..., 2], 0.1, amp=1.0)[..., idx_df]
            Fz_fit = FEs_fit[..., 2][..., idx_Fz]
            dxy = np.hypot(pp[..., 0] - nom_xy[0][:, :, None], pp[..., 1] - nom_xy[1][:, :, None])[..., idx_Fz]
            for iz, (hf, hd) in enumerate(zip(h_Fz, h_df)):
                lines.append(f'[{label}] h_Fz={hf:.2f} h_df={hd:.2f} df_corr={corr(df_ref[..., iz], df_fit[..., iz]):.6f} '
                             f'df_rmse={rmse(df_ref[..., iz], df_fit[..., iz]):.4e} '
                             f'Fz_corr={corr(Fz_ref[..., iz], Fz_fit[..., iz]):.6f} '
                             f'Fz_rmse={rmse(Fz_ref[..., iz], Fz_fit[..., iz]):.4e}')
            lines.append(f'[{label}] scan_sec={sc:.2f} eval_ms_per_1k={t_eval:.3f} npts={len(holdout)}')
            print('\n'.join(lines[-(2 + 3 + len(h_Fz)):]), flush=True)
            return dict(df=df_fit, Fz=Fz_fit, dxy=dxy, E_hold=E_g, F_hold=F_g, FEs=FEs_fit, scan_sec=sc, eval_ms=t_eval)

        variants['FDBM']['dxy'] = np.hypot(disp_ref['dx'], disp_ref['dy'])[..., idx_Fz]
        pp_variants = {}; scan_secs = []
        eval_ms = None
        for label, cpm_v in ((f'coremesh_{h:g}', cpm),) + (() if skip_spline_only else ((f'spline_only_{h:g}', cpm0),)):
            r = scan_variant(label, cpm_v)
            if label.startswith('coremesh'):
                eval_ms = r['eval_ms']
            variants[label] = {'df': r['df'], 'Fz': r['Fz'], 'dxy': r['dxy']}
            row_specs.append(('df', label, f'{label} df', 'gray'))
            snapshots.update({f'{label}_E_hold': r['E_hold'], f'{label}_F_hold': r['F_hold'],
                              f'{label}_df': r['df'], f'{label}_Fz': r['Fz'], f'{label}_FEs': r['FEs'],
                              f'{label}_dxy': r['dxy']})
            pp_variants[label] = {'dxy': r['dxy']}
            scan_secs.append(r['scan_sec'])
        scan_sec = sum(scan_secs)
        pp_fig = os.path.join(outdir, 'compare_pp_dxy.png')
        pv = {'FDBM': {'dxy': variants['FDBM']['dxy']}}; pv.update(pp_variants)
        afm_utils.plot_afm_variant_height_strip(pv, [('dxy', k, f'{k} |dxy|', 'viridis') for k in pv],
                                                h_Fz, pp_fig, title=f'{mol_name} relaxed-PP lateral displacement |dxy| (FDBM vs fits)',
                                                extent=afm_utils.scan_extent(scan_xs, scan_ys), amp=None, amp_align=False, dpi=150)
        print(f'REVIEW: {pp_fig}', flush=True)

    else:  # separable 2.5D contact fit (CG — DEPRECATED slow path)
        from spammm.surfaces.ContactSurface import (
            SeparableParams, _bspline_prefilter_2d, bspline_n_intervals, make_fit_grid_zstack)
        xlo_f, xhi_f = float(scan_xs[0]) - 0.75, float(scan_xs[-1]) + 0.75
        ylo_f, yhi_f = float(scan_ys[0]) - 0.75, float(scan_ys[-1]) + 0.75
        fit_heights = np.arange(2.4, 7.0001, 0.15)
        xyz = make_fit_grid_zstack(xlo_f, xhi_f, ylo_f, yhi_f, mol_z + fit_heights, 0.25, 0.25)
        ns_ = len(xyz)
        pad = (-ns_) % 32
        xyz = np.concatenate([xyz, np.repeat(xyz[:1], pad, axis=0)])
        weights = np.concatenate([np.ones(ns_), np.zeros(pad)])
        E_ref, F_ref = sample(xyz)
        for dx in bspl_steps:
            label = f'contact_{dx:g}'
            x0, y0 = xlo_f - 2 * dx, ylo_f - 2 * dx
            ncx, ncy = bspline_n_intervals(xhi_f - x0 + 2 * dx, dx), bspline_n_intervals(yhi_f - y0 + 2 * dx, dx)
            h0 = _bspline_prefilter_2d(np.full((ncy, ncx), mol_z))
            sep = SeparableParams(x0, y0, dx, dx, ncx, ncy, poly_R=poly_R, poly_z0=poly_z0,
                                  m_start=1, nz=z_modes, h0_map=h0.ravel())
            print(f'Fitting {label}: {ncx}x{ncy}x{z_modes} n_iter={n_iter}', flush=True)
            t0 = time.perf_counter()
            a.fit_contact_field(sep, xyz, E_ref, F_ref, n_iter=n_iter, force_weight=1.0,
                                sample_weights=weights, bPrint=True)
            t_fit = time.perf_counter() - t0
            archive = os.path.join(outdir, f'{label}.npz')
            sep.save_npz(archive, metadata={'source': 'FDBMPipeline.run_fields', 'molecule': mol_name,
                                            'grid_step': step, 'atomPos': atomPos.tolist()})
            loaded = SeparableParams.load_npz(archive)
            E_before, F_before = a._cs_fit_helper().eval_separable(holdout, sep)
            E_fit, F_fit = a._cs_fit_helper().eval_separable(holdout, loaded)
            assert np.array_equal(E_before, E_fit) and np.array_equal(F_before, F_fit)
            a.setup_contact_surface(loaded)
            t0 = time.perf_counter()
            FEs_fit, _ = a.run_scan_contact(nxy=(len(scan_xs), len(scan_ys)), nz=len(h_scan),
                                            dtip=-0.1, scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db)
            scan_sec = time.perf_counter() - t0
            FEs_fit = FEs_fit[:, :, ::-1, :]
            df_fit = afm.compute_df_amp(FEs_fit[..., 2], 0.1, amp=1.0)[..., idx_df]
            Fz_fit = FEs_fit[..., 2][..., idx_Fz]
            variants[label] = {'df': df_fit, 'Fz': Fz_fit}
            row_specs.append(('df', label, f'{dx:g} A mesh df', 'gray'))
            arch_bytes = os.path.getsize(archive)
            lines.extend([f'\n[{label}] knots={ncx}x{ncy} nz={sep.nz} n_coeff={sep.n_coeff}',
                          f'resident_f32_bytes={loaded.resident_bytes} archive_bytes={arch_bytes} ratio={dense_bytes / loaded.resident_bytes:.0f}x',
                          f'fit_sec={t_fit:.2f} contact_scan_sec={scan_sec:.2f}',
                          f'holdout_E_rmse={rmse(E_hold, E_fit):.4e} max={np.abs(E_hold - E_fit).max():.4e}'])
            for c, nm in enumerate(('Fx', 'Fy', 'Fz')):
                lines.append(f'holdout_{nm}_rmse={rmse(F_hold[:, c], F_fit[:, c]):.4e} corr={corr(F_hold[:, c], F_fit[:, c]):.6f}')
            for iz, (hf, hd) in enumerate(zip(h_Fz, h_df)):
                lines.append(f'h_Fz={hf:.2f} h_df={hd:.2f} df_corr={corr(df_ref[..., iz], df_fit[..., iz]):.6f} '
                             f'df_rmse={rmse(df_ref[..., iz], df_fit[..., iz]):.4e} '
                             f'Fz_corr={corr(Fz_ref[..., iz], Fz_fit[..., iz]):.6f} '
                             f'Fz_rmse={rmse(Fz_ref[..., iz], Fz_fit[..., iz]):.4e}')
            print('\n'.join(lines[-(4 + 3 + len(h_Fz)):]), flush=True)
            snapshots.update({f'{label}_E_hold': E_fit, f'{label}_F_hold': F_fit,
                              f'{label}_df': df_fit, f'{label}_Fz': Fz_fit, f'{label}_FEs': FEs_fit})

    figure = os.path.join(outdir, 'compare_df_Fz.png')
    _variant_figure(afm_utils, variants, row_specs, h_Fz, scan_xs, scan_ys, figure,
                    f'{mol_name} FDBM vs {method} compressed field; amp=1 A')
    np.savez_compressed(os.path.join(outdir, 'comparison_arrays.npz'), **snapshots)
    summary = os.path.join(outdir, 'SUMMARY.out')
    with open(summary, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'REVIEW: {summary}\nREVIEW: {figure}', flush=True)
    row = {'mol': mol_name, 'method': method, 'tag': tag, 't_fields': t_fields, 'fit_sec': t_fit,
           'scan_sec': scan_sec, 'archive_kb': arch_bytes / 1024,
           'dense_mb': dense_bytes / 1e6,
           'ratio': dense_bytes / (loaded.resident_bytes if method == 'sep' else cpm.resident_bytes)}
    if method == 'coremesh':
        row.update(natoms=len(atomPos), ncenters=len(centers), ncoef=int(fit.coeffs.size),
                   npts=len(pts), mesh_nodes='x'.join(map(str, n_mesh)), core_fit_sec=t_core,
                   cg_iters=(cg_info['iters'] if (cg_info is not None and 'iters' in cg_info) else -1), cg_sec=(cg_info.get('sec', 0.0) if cg_info is not None else 0.0),
                   mesh_res_sec=t_m1, mesh_res_iters=dg_res['iterations'], eval_ms=eval_ms,
                   resident_bytes=cpm.resident_bytes)
    return row


def main():
    from spammm.SPM.FDBMPipeline import FDBMPipeline
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('mols', nargs='*', default=['azaindol', 'PTCDA'])
    p.add_argument('--method', choices=('pme', 'sep', 'coremesh'), default='pme')
    p.add_argument('--h-mesh', type=float, default=None, help=f'mesh spacing [A]; default {H_MESH} for pme, 1.0 for coremesh')
    p.add_argument('--lam', type=float, default=1e-3, help='mesh smoothness penalty (coremesh)')
    p.add_argument('--step', type=float, default=0.1, help='FDBM density/field grid step [A]')
    p.add_argument('--no-core', action='store_true', help='zero-core PME (old artifact-prone path)')
    p.add_argument('--din', type=float, default=1.0, help='SplitParams delta_in (r_lo=R0-din)')
    p.add_argument('--db', type=float, default=1.0, help='SplitParams delta_b (r_b=R0+db)')
    p.add_argument('--z-fit', type=float, default=1.2, help='min z of core-fit samples above mol_top [A]')
    p.add_argument('--cap', type=float, default=10.0, help='residual cap: nodes |resid|>cap are inpainted [eV]')
    p.add_argument('--core-basis', choices=('poly8', 'poly8sp'), default='poly8',
                   help='coremesh core basis: poly8 = 5 scalar slots; poly8sp = 9-slot angular sp (A:s3p2+B:s1)')
    p.add_argument('--core-wF', type=float, default=0.3, help='poly8sp force-row weight in core LSQ (0 = energy only)')
    p.add_argument('--core-spec', default='A:s3p2+B:s1', help="poly8sp basis spec, e.g. 'A:s3p2+B:s1' or atom-only 'A:s3p3'")
    p.add_argument('--core-solver', choices=('gpu', 'gpugram', 'lstsq', 'gram'), default='gpu',
                   help="poly8sp core fit solver: gpu = matrix-free CGLS on OpenCL (design doc §16); gpugram = GPU Gram columns + Cholesky (exact+ridge, no iterations); gram/lstsq = CPU references")
    p.add_argument('--core-ridge', type=float, default=1e-8, help='gram solver: Tikhonov ridge on the column-scaled Gram (fraction of trace/ncoef)')
    p.add_argument('--mesh-fit', choices=('inpaint', 'cg'), default='inpaint',
                   help="coremesh mesh solve: inpaint = omit rm<2.5/z<z_min cells via harmonic fill + separable banded fit (fast, PAW recipe); cg = legacy weighted supersampled CG")
    p.add_argument('--skip-spline-only', action='store_true', help='coremesh: skip the spline-only mesh fit + scan variant (halves mesh time)')
    args = p.parse_args()
    MOLS = {'azaindol': 'data/xyz/azaindol.xyz', 'PTCDA': 'data/xyz/PTCDA.xyz',
            'pentacene': 'data/xyz/pentacene.xyz', 'PTCDI': 'data/xyz/PTCDI.xyz',
            'porphirin': 'data/xyz/porphirin.xyz',
            'circumcoronene': 'data/xyz/circumcoronene.xyz',
            'circumcircumcoronene': 'data/xyz/circumcircumcoronene.xyz'}
    pipe = FDBMPipeline()          # ONE pipeline for the whole batch (batch contract)
    rows = []
    for m in args.mols:
        h_mesh = args.h_mesh if args.h_mesh is not None else (1.0 if args.method == 'coremesh' else H_MESH)
        rows.append(compress_mol(pipe, m, MOLS[m], args.step, method=args.method,
                                 h_mesh=h_mesh, no_core=args.no_core,
                                 delta_in=args.din, delta_b=args.db,
                                 z_fit_dz=args.z_fit, resid_cap=args.cap, lam=args.lam,
                                 core_basis=args.core_basis, core_wF=args.core_wF, core_spec=args.core_spec, core_solver=args.core_solver,
                                 skip_spline_only=args.skip_spline_only, mesh_fit=args.mesh_fit, core_ridge=args.core_ridge))
    print(f"\n\n{'='*80}")
    hdr = (f"{'mol':22s} {'method':8s} {'nat':>4s} {'ncntr':>6s} {'ncoef':>6s} {'npts':>9s} {'mesh':>10s} "
           f"{'core_s':>7s} {'cgit':>5s} {'mesh1_s':>7s} {'scan_s':>7s} {'ms/1k':>6s} {'res_MB':>7s} "
           f"{'fit_s':>7s} {'dense_MB':>8s} {'arch_KB':>8s} {'ratio':>7s}")

    def _c(v, fm, na='-'):
        return na if v is None else fm.format(v)

    def _row(r):
        return (f"{r['mol']:22s} {r['method']:8s} {_c(r.get('natoms'), '{:4d}')} {_c(r.get('ncenters'), '{:6d}')} "
                f"{_c(r.get('ncoef'), '{:6d}')} {_c(r.get('npts'), '{:9d}')} {_c(r.get('mesh_nodes'), '{:>10s}')} "
                f"{_c(r.get('core_fit_sec'), '{:7.2f}')} {_c(r.get('cg_iters'), '{:5d}')} {_c(r.get('mesh_res_sec'), '{:7.2f}')} "
                f"{r['scan_sec']:7.2f} {_c(r.get('eval_ms'), '{:6.3f}')} {_c(None if r.get('resident_bytes') is None else r['resident_bytes'] / 1e6, '{:7.2f}')} "
                f"{r['fit_sec']:7.2f} {r['dense_mb']:8.1f} {r['archive_kb']:8.1f} {r['ratio']:6.0f}x")

    print(hdr)
    for r in rows:
        line = _row(r)
        print(line)
        scal = os.path.join(OUTROOT, f"SCALING_{r.get('tag', r['method'])}.out")
        if not os.path.exists(scal) or os.path.getsize(scal) == 0:
            with open(scal, 'w') as f:
                f.write(hdr + '\n')
        with open(scal, 'a') as f:
            f.write(line + '\n')
        print(f'REVIEW: {scal}')


if __name__ == '__main__':
    main()
