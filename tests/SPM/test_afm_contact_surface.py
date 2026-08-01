"""
test_afm_contact_surface.py — Memory-efficient Morse PP-AFM via contact surface.

Pipeline: load molecule → assign_params → fit_contact_surface (no 3D grid)
→ run_scan_contact (relaxStrokesTiltedContact) → finite FEs.
"""
import os
import pytest
import numpy as np

os.environ.setdefault('PYOPENCL_CTX', '0')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
PARAMS_PATH = os.path.join(DATA_DIR, 'ElementTypes.dat')


def _make_afm(xyz_path):
    from spammm.SPM.AFM import AFMulator
    afm = AFMulator(use_morse=True, use_fire=False)
    afm.load_molecule(xyz_path)
    afm.assign_params(params_path=PARAMS_PATH)
    return afm


@pytest.mark.gpu
def test_contact_surface_force_stencil_parity(xyz):
    """Av_f stencil must match eval_separable F (sign + chain rule)."""
    afm = _make_afm(xyz('benzene.xyz'))
    apos = afm.atoms_arr[:, :3]
    margin = 2.0
    x0f, x1f = float(apos[:, 0].min()) - margin, float(apos[:, 0].max()) + margin
    y0f, y1f = float(apos[:, 1].min()) - margin, float(apos[:, 1].max()) + margin
    from spammm.surfaces.ContactSurface import SeparableParams, bspline_n_intervals
    sep = SeparableParams(x0f, y0f, 0.6, 0.6, bspline_n_intervals(x1f - x0f, 0.6), bspline_n_intervals(y1f - y0f, 0.6), poly_R=5.0, poly_z0=1.0, m_start=4, nz=4, apos=apos)
    cs = afm._cs_fit_helper()
    rng = np.random.default_rng(0)
    sep.coeffs = rng.standard_normal(sep.n_coeff)
    cs.setup_separable(sep, apos=apos)
    z = float(apos[:, 2].max()) + 2.0
    q = np.column_stack([apos[::3, 0], apos[::3, 1], np.full(len(apos[::3]), z)]).astype(np.float32)
    _, F_ev = cs.eval_separable(q, sep)
    E_ref, F_ref = afm._brute_afm_morse_c_queries(q)
    cs.upload_samples(q, E_ref, F_ref=F_ref)
    meta, origin_step, dy_rc, invRc_mstart = cs._sep_cl_args(sep)
    ns, nG = len(q), cs._roundup(len(q), cs.nloc)
    cs.toGPU_(cs.cs_coeffs_buff, sep.coeffs.astype(np.float32))
    import pyopencl as cl
    for fcomp in range(3):
        cs.prg.cs_sep_Av_f(cs.queue, (nG,), (cs.nloc,), cs.cs_samples_buff, cs.cs_coeffs_buff, cs.cs_AFp_buff, cs.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns), np.int32(fcomp))
        pred = np.zeros(ns, dtype=np.float32)
        cl.enqueue_copy(cs.queue, pred, cs.cs_AFp_buff)
        err = pred - F_ev[:, fcomp]
        assert float(np.sqrt(np.mean(err * err))) < 1e-4, f"Av_f fcomp={fcomp} != eval F"


@pytest.mark.gpu
def test_afm_contact_surface_scan(xyz):
    """Fit separable contact surface and run PP-relaxed scan without img_FF."""
    afm = _make_afm(xyz('benzene.xyz'))
    sep = afm.fit_contact_surface(margin=2.0, bspl_dx=0.6, fit_z_stack=(1.0, 1.2, 1.5), n_iter=40, bPrint=False)
    assert sep.coeffs is not None
    assert sep.n_coeff < 50000, f"unexpected coeff count {sep.n_coeff}"
    apos = afm.atoms_arr[:, :3]
    # Query ABOVE the contact surface (h0_max + offset), not above bare atom zmax.
    # The separable basis is zero for z < h0 (clamped by fmax(z-h0, 0) in the kernel),
    # so querying at zmax+1.2 (below h0 for benzene where h0~2.5 > zmax=0) gives a
    # degenerate fit evaluation that cannot match the brute force. The fit was done
    # at z = h0_max + [1.0, 1.2, 1.5], so query at h0_max + 1.2 (inside fit region).
    h0_max = float(np.max(sep.h0_map)) if sep.h0_map is not None else float(apos[:, 2].max())
    z_probe = h0_max + 1.2
    pts_ref = np.column_stack([apos[::2, 0], apos[::2, 1], np.full(len(apos[::2]), z_probe)]).astype(np.float32)
    _, F_ref = afm._brute_afm_morse_c_queries(pts_ref)
    _, F_fit = afm._cs_fit_helper().eval_separable(pts_ref, sep)
    active = np.abs(F_ref[:, 2]) > 1e-5
    assert np.all(np.sign(F_fit[active, 2]) == np.sign(F_ref[active, 2])), "contact surface Fz sign must match AFM brute force"
    FEs, pts = afm.run_scan_contact(nxy=(12, 12), nz=8, dtip=-0.15)
    assert FEs.shape == (12, 12, 8, 4)
    assert np.isfinite(FEs).all()
    Fz = FEs[:, :, :, 2]
    assert float(np.max(np.abs(Fz))) < 1e6
    assert float(np.std(Fz)) > 1e-8, "degenerate Fz map"
    raw, _ = afm.get_raw_FE_contact(nxy=(8, 8), nz=5, dtip=-0.15)
    assert np.isfinite(raw).all()
