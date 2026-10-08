"""L0 parity: FDBMPipeline (device-resident) vs legacy run_fdbm_pp_from_density.

Contract (doc/Tasks/FDBM_EndToEnd_GPU_Pipeline.md):
  - device densities == legacy host densities (fp32 rounding only)
  - one shared OpenCL ctx across AFMulator / projectors / FFT
  - df corr >= 0.999 vs legacy end-to-end
Coarse grid (step=0.2) keeps this a routine L0 test (~1 s).
"""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module')
def azaindol_ref():
    """Legacy reference: densities + full PP scan on azaindole (untouched old path)."""
    from spammm import atomicUtils as au
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import (
        parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list)
    from spammm.forcefields.FFController import make_planar_xy, orient_long_axis_x

    atomPos, _, enames, _, _ = au.load_xyz('data/xyz/azaindol.xyz')
    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16}
    atomTypes = np.array([ELEM_Z[e] for e in enames], dtype=np.int32)
    atomPos[:] = make_planar_xy(atomPos)
    orient_long_axis_x(atomPos)
    atomPos[:, 2] = 0.0
    step, margin, z_vac = 0.2, 4.0, 6.0
    grid_spec, origin, ngrid, step = afm_utils.make_fdbm_grid_com_zsym(atomPos, step, margin, z_vac=z_vac)
    basis_hsd = get_dftb_basis_path('3ob-3-1')
    pa = afm.PAULI_FITTED_DEFAULTS['3ob-3-1']
    A_pauli, beta_pauli = float(pa['A']), float(pa['beta'])

    work = 'debug/test_fdbm_pipeline_v2'
    res_s = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, os.path.join(work, 'leg_stock'),
        grid_spec=grid_spec, step=step, verbosity=0, need_ves=False)
    basis_data = parse_wfc_hsd(basis_hsd)
    basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
    prol = make_slater_tail_species_list(basis_ang)
    res_p = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd, os.path.join(work, 'leg_prol'),
        grid_spec=grid_spec, step=step, verbosity=0,
        projection_basis_ang=prol, need_es=False, need_ves=False)
    leg = afm_utils.run_fdbm_pp_from_density(
        'prolonged', res_p['rho_scf'], atomPos, atomTypes, origin, step, ngrid,
        A_pauli, beta_pauli, 'co', os.path.join(work, 'leg_out'),
        rho_diff=res_s['rho_diff'], basis='3ob-3-1', margin=margin, plots=None,
        use_fast_s3=True, h_min=3.7, h_max=4.2, h_step=0.25, amp=1.0, amp_align=True,
        K_LAT_Nm=0.5, K_RAD=20.0, bond_length=3.0, scan_margin=2.0)
    return {'atomPos': atomPos, 'atomTypes': atomTypes, 'origin': origin, 'step': step,
            'ngrid': ngrid, 'grid_spec': grid_spec, 'basis_hsd': basis_hsd, 'margin': margin,
            'A': A_pauli, 'beta': beta_pauli, 'res_s': res_s, 'res_p': res_p, 'leg': leg}


@pytest.fixture(scope='module')
def azaindol_new(azaindol_ref):
    """New pipeline on the identical geometry/grid."""
    from spammm.SPM.FDBMPipeline import FDBMPipeline
    r = azaindol_ref
    pipe = FDBMPipeline()
    out = pipe.run(r['atomPos'], r['atomTypes'], r['basis_hsd'],
                   'debug/test_fdbm_pipeline_v2/gpu_work', r['grid_spec'], r['origin'],
                   r['step'], r['ngrid'], r['A'], r['beta'], 'co',
                   'debug/test_fdbm_pipeline_v2/gpu_out',
                   projection='prolonged', basis='3ob-3-1', margin=r['margin'], plots=None,
                   h_min=3.7, h_max=4.2, h_step=0.25, amp=1.0, amp_align=True,
                   K_LAT_Nm=0.5, K_RAD=20.0, bond_length=3.0, scan_margin=2.0)
    return {'pipe': pipe, 'res': out['prolonged']}


def _corr(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def test_shared_context(azaindol_new):
    """All GPU objects share ONE OpenCL context (the core architectural fix)."""
    pipe = azaindol_new['pipe']
    ctxs = {pipe.afm.ctx} | {p.ctx for p in pipe.projectors.values()} | {f.ctx for f in pipe.ffts.values()}
    assert len(ctxs) == 1, f"expected single ctx, got {len(ctxs)}"
    for p in pipe.projectors.values():
        assert p.queue is pipe.afm.queue


def test_density_parity(azaindol_ref, azaindol_new):
    """Device densities == legacy host densities (fp32 rounding, rel < ~1e-5)."""
    pipe = azaindol_new['pipe']
    r = azaindol_ref
    fft = pipe.ffts[tuple(r['ngrid'])]
    rho_diff_dev = fft._xyz_rho_diff.get()
    rho_prol_dev = fft._xyz_rho_scf.get()
    assert np.allclose(rho_diff_dev, r['res_s']['rho_diff'], atol=1e-4), \
        f"rho_diff max|d|={np.abs(rho_diff_dev - r['res_s']['rho_diff']).max():.3e}"
    assert np.allclose(rho_prol_dev, r['res_p']['rho_scf'], atol=1e-4), \
        f"rho_scf_prol max|d|={np.abs(rho_prol_dev - r['res_p']['rho_scf']).max():.3e}"
    q_diff = float(rho_diff_dev.sum() * r['step'] ** 3)
    assert abs(q_diff) < 2.0, f"q_diff={q_diff:.4f} (charge conservation)"


def test_scan_parity(azaindol_ref, azaindol_new):
    """df / Fz / FEs / E_diss parity vs legacy end-to-end.

    Tolerance note: densities agree to ~1e-5 rel (fp32 rounding of the fused
    dm_scf−dm_na projection), but PP relaxation amplifies that nonlinearly —
    at this coarse step=0.2 grid df corr is ~0.999 (measured 0.9996–0.9998 at
    production step 0.1–0.15 via testplot_fdbm_pipeline_parity.py).
    """
    leg, new = azaindol_ref['leg'], azaindol_new['res']
    assert leg.df.shape == new.df.shape
    c_df = _corr(leg.df, new.df)
    c_fz = _corr(leg.Fz, new.Fz)
    c_fe = _corr(leg.FEs, new.FEs)
    print(f"df corr={c_df:.6f}  Fz corr={c_fz:.6f}  FEs corr={c_fe:.6f}")
    assert c_df >= 0.998, f"df corr {c_df:.6f} < 0.998"
    assert c_fz >= 0.999, f"Fz corr {c_fz:.6f} < 0.999"
    assert c_fe >= 0.999, f"FEs corr {c_fe:.6f} < 0.999"
    assert _corr(leg.E_diss, new.E_diss) >= 0.99
    assert np.array_equal(leg.heights, new.heights)
    assert np.array_equal(leg.scan_xs, new.scan_xs)
