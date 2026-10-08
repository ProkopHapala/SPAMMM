"""L1/L2 benchmark + parity: FDBMPipeline (device-resident) vs legacy orchestration.

Runs both end-to-end on several molecules, prints per-stage timing + host RSS /
GPU VRAM deltas, and writes side-by-side df/Fz compare strips (legacy | gpu | diff)
for USER review.

Usage:
    python tests/SPM/testplot_fdbm_pipeline_parity.py [mol ...]
Output: debug/testplot_fdbm_pipeline_parity/<mol>/compare.png + printed table.
"""
import os, sys, time, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OUTROOT = 'debug/testplot_fdbm_pipeline_parity'
MOLS = {
    'azaindol':  ('data/xyz/azaindol.xyz',  0.10),
    'pentacene': ('data/xyz/pentacene.xyz', 0.10),
    'PTCDA':     ('data/xyz/PTCDA.xyz',     0.10),
}
SCAN_KW = dict(h_min=3.7, h_max=4.7, h_step=0.1, amp=1.0, amp_align=True,
               K_LAT_Nm=0.5, K_RAD=20.0, bond_length=3.0, scan_margin=2.0)


def host_rss_mb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def gpu_mb():
    """GPU bytes used by THIS process (nvidia-smi per-app accounting)."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            pid, mb = [x.strip() for x in line.split(',')]
            if int(pid) == os.getpid():
                return float(mb)
    except Exception:
        pass
    return 0.0  # process not on GPU yet → 0 (deltas stay numeric)


def corr(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def bench_mol(mol_name, xyz, step):
    from spammm import atomicUtils as au
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.SPM.FDBMPipeline import FDBMPipeline
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import (
        parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list)
    from spammm.forcefields.FFController import make_planar_xy, orient_long_axis_x

    outdir = os.path.join(OUTROOT, mol_name)
    os.makedirs(outdir, exist_ok=True)
    atomPos, _, enames, _, _ = au.load_xyz(xyz)
    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16}
    atomTypes = np.array([ELEM_Z[e] for e in enames], dtype=np.int32)
    natoms = len(enames)
    atomPos[:] = make_planar_xy(atomPos)
    orient_long_axis_x(atomPos)
    atomPos[:, 2] = 0.0
    margin, z_vac = 4.0, 6.0
    grid_spec, origin, ngrid, step = afm_utils.make_fdbm_grid_com_zsym(atomPos, step, margin, z_vac=z_vac)
    nvox = int(np.prod(ngrid))
    basis_hsd = get_dftb_basis_path('3ob-3-1')
    pa = afm.PAULI_FITTED_DEFAULTS['3ob-3-1']
    A, beta = float(pa['A']), float(pa['beta'])
    print(f"\n{'='*90}\n{mol_name}  natoms={natoms}  grid={ngrid} ({nvox/1e6:.2f}Mvox)  step={step}\n{'='*90}")

    # ── LEGACY ──────────────────────────────────────────────────────────────
    rss0, gpu0 = host_rss_mb(), gpu_mb()
    t0 = time.time()
    res_s = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, os.path.join(outdir, 'leg_work_stock'),
        grid_spec=grid_spec, step=step, verbosity=0, need_ves=False)
    basis_data = parse_wfc_hsd(basis_hsd)
    basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
    prol = make_slater_tail_species_list(basis_ang)
    res_p = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, os.path.join(outdir, 'leg_work_prol'),
        grid_spec=grid_spec, step=step, verbosity=0, projection_basis_ang=prol,
        need_es=False, need_ves=False)
    t_leg_dens = time.time() - t0
    t0 = time.time()
    leg = afm_utils.run_fdbm_pp_from_density(
        'prolonged', res_p['rho_scf'], atomPos, atomTypes, origin, step, ngrid,
        A, beta, 'co', os.path.join(outdir, 'leg'),
        rho_diff=res_s['rho_diff'], basis='3ob-3-1', margin=margin, plots=None,
        use_fast_s3=True, **SCAN_KW)
    t_leg_pp = time.time() - t0
    rss_leg, gpu_leg = host_rss_mb() - rss0, gpu_mb() - gpu0

    # keep legacy density arrays alive for parity (real usage keeps them too)
    rho_diff_leg = res_s['rho_diff']; rho_prol_leg = res_p['rho_scf']

    # ── GPU PIPELINE (cold = first touch; warm = same mol again on same pipe →
    #     shows batch steady-state: FFT plan/tip/projector/img caches all hot) ──
    rss0, gpu0 = host_rss_mb(), gpu_mb()
    pipe = FDBMPipeline()
    def _run(tag):
        t0 = time.time()
        prep = pipe.dftb_prep(atomPos, atomTypes, basis_hsd, os.path.join(outdir, f'gpu_work{tag}'))
        t_prep = time.time() - t0
        t0 = time.time()
        bufs = pipe.project_densities(prep, grid_spec, step, variants=('diff', 'prol'))
        t_proj = time.time() - t0
        t0 = time.time()
        pipe.fields(bufs['prol'], bufs['diff'], atomPos, atomTypes, origin, step, ngrid,
                    A, beta, tip_mode='co', margin=margin, basis='3ob-3-1', output_dir=outdir)
        pipe.ensure_afm().queue.finish()
        t_fields = time.time() - t0
        t0 = time.time()
        new = pipe.scan_and_post('prolonged', atomPos, atomTypes, origin, step,
                                 A_pauli=A, beta_pauli=beta, outdir=None, plots=None, **SCAN_KW)
        t_scan = time.time() - t0
        return prep, bufs, new, t_prep, t_proj, t_fields, t_scan
    prep, bufs, new, t_prep, t_proj, t_fields, t_scan = _run('')
    rss_gpu, gpu_gpu = host_rss_mb() - rss0, gpu_mb() - gpu0
    t_gpu_tot = t_prep + t_proj + t_fields + t_scan
    _p, _b, _n, t_prep_w, t_proj_w, t_fields_w, t_scan_w = _run('_warm')
    t_gpu_warm = t_prep_w + t_proj_w + t_fields_w + t_scan_w
    print(f'  [warm] prep={t_prep_w:.2f} proj={t_proj_w:.2f} field={t_fields_w:.2f} scan={t_scan_w:.2f}  total={t_gpu_warm:.2f}s')

    # ── parity ──────────────────────────────────────────────────────────────
    rho_diff_dev = np.ascontiguousarray(bufs['diff'].get())
    rho_prol_dev = np.ascontiguousarray(bufs['prol'].get())
    d_rho = np.abs(rho_diff_dev - rho_diff_leg).max()
    d_prol = np.abs(rho_prol_dev - rho_prol_leg).max()
    c_df, c_fz, c_fe, c_ed = corr(leg.df, new.df), corr(leg.Fz, new.Fz), corr(leg.FEs, new.FEs), corr(leg.E_diss, new.E_diss)
    rms_df = float(np.sqrt(np.mean((leg.df - new.df) ** 2)))
    print(f"\n[{mol_name}] PARITY  rho_diff max|d|={d_rho:.2e}  rho_prol max|d|={d_prol:.2e}")
    print(f"  df corr={c_df:.6f} (rms={rms_df:.3e})  Fz corr={c_fz:.6f}  FEs corr={c_fe:.6f}  Ediss corr={c_ed:.6f}")

    # ── compare figure (legacy | gpu | diff) ────────────────────────────────
    diff_v = {'df': (new.df - leg.df).astype(np.float32), 'Fz': (new.Fz - leg.Fz).astype(np.float32)}
    row_specs = [
        ('df', 'legacy', f'df legacy', 'gray'),
        ('df', 'gpu', f'df gpu', 'gray'),
        ('df', 'diff', f'df Δ', 'seismic'),
        ('Fz', 'legacy', f'Fz legacy', 'seismic'),
        ('Fz', 'gpu', f'Fz gpu', 'seismic'),
    ]
    png = os.path.join(outdir, 'compare.png')
    afm_utils.plot_afm_variant_height_strip(
        {'legacy': leg, 'gpu': new, 'diff': diff_v}, row_specs, leg.heights, png,
        scale='per_image', title=f'{mol_name} FDBM legacy vs gpu-pipeline  df corr={c_df:.4f}',
        dpi=140, apos=atomPos, show_atoms=True,
        extent=afm_utils.scan_extent(leg.scan_xs, leg.scan_ys),
        amp=SCAN_KW['amp'], amp_align=True, amp_z=SCAN_KW['amp'], tight=True)
    print(f'REVIEW: {png}')

    return {
        'mol': mol_name, 'natoms': natoms, 'ngrid': ngrid, 'nvox': nvox,
        't_leg_dens': t_leg_dens, 't_leg_pp': t_leg_pp, 't_leg': t_leg_dens + t_leg_pp,
        't_prep': t_prep, 't_proj': t_proj, 't_fields': t_fields, 't_scan': t_scan, 't_gpu': t_gpu_tot,
        'rss_leg': rss_leg, 'rss_gpu': rss_gpu, 'gpu_leg': gpu_leg, 'gpu_gpu': gpu_gpu,
        't_gpu_warm': t_gpu_warm,
        'c_df': c_df, 'c_fz': c_fz, 'c_fe': c_fe, 'c_ed': c_ed,
    }


def main():
    mols = sys.argv[1:] or list(MOLS)
    rows = []
    for m in mols:
        xyz, step = MOLS[m]
        rows.append(bench_mol(m, xyz, step))

    print(f"\n\n{'='*100}")
    print(f"{'mol':10s} {'natoms':>6s} {'grid':>15s} | {'LEG dens':>8s} {'LEG pp':>7s} {'LEG tot':>8s} | "
          f"{'prep':>5s} {'proj':>5s} {'field':>5s} {'scan':>5s} {'GPU tot':>7s} | {'warm':>5s} | {'speedup':>7s}")
    print('-' * 110)
    for r in rows:
        print(f"{r['mol']:10s} {r['natoms']:6d} {str(r['ngrid']):>15s} | "
              f"{r['t_leg_dens']:8.2f} {r['t_leg_pp']:7.2f} {r['t_leg']:8.2f} | "
              f"{r['t_prep']:5.2f} {r['t_proj']:5.2f} {r['t_fields']:5.2f} {r['t_scan']:5.2f} {r['t_gpu']:7.2f} | "
              f"{r['t_gpu_warm']:5.2f} | {r['t_leg']/max(r['t_gpu_warm'],1e-9):6.2f}x")
    print(f"\n{'mol':10s} {'host RSS Δ LEG':>14s} {'host RSS Δ GPU':>14s} {'GPU VRAM Δ LEG':>14s} {'GPU VRAM Δ GPU':>14s}")
    print('-' * 70)
    for r in rows:
        mb_per_vox = 4.0 * r['nvox'] / 1e6
        print(f"{r['mol']:10s} {r['rss_leg']:13.0f}M {r['rss_gpu']:13.0f}M {r['gpu_leg']:13.0f}M {r['gpu_gpu']:13.0f}M"
              f"   (ngrid float32 = {mb_per_vox:.0f}MB/array)")
    print(f"\n{'mol':10s} {'df corr':>9s} {'Fz corr':>9s} {'FEs corr':>9s} {'Ediss':>9s}")
    for r in rows:
        print(f"{r['mol']:10s} {r['c_df']:9.5f} {r['c_fz']:9.5f} {r['c_fe']:9.5f} {r['c_ed']:9.5f}")
    print(f"\nNOTE host-RSS Δ includes result FEs (~scan grid) + interpreter noise; "
          f"the structural difference is ngrid host arrays: LEG holds rho_scf/rho_na/rho_diff(+rho_prol)/tips/F_total "
          f"≈10×ngrid×4B on host; GPU pipe keeps all of them device-resident.")


if __name__ == '__main__':
    main()
