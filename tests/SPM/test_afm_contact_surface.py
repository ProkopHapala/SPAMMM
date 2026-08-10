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


# ════════════════════════════════════════════════════════════════════════════
# Wave 0: PMESplit — radial oracle and soft-core split (Agent_1)
# Contract version 2. Oracle: getMorsePLQH / cs_brute_plqh_points.
# ════════════════════════════════════════════════════════════════════════════

_WAVE0_DIR = os.path.join('debug', 'test_afm_contact_surface', 'contact_pme', 'wave0_split')

# C/O/H-like Morse parameters from assign_params combination rules
# R0 = tip_R + R_vdW_sample, E0 = sqrt(tip_E * E_vdW_sample)
_TIP_R = 1.452; _TIP_E = 0.0006808; _TIP_ALPHA = 1.8; _R_DAMP = 0.1
_C_R0 = _TIP_R + 1.9255; _C_E0 = float(np.sqrt(_TIP_E * 0.00455323095))
_O_R0 = _TIP_R + 1.7500; _O_E0 = float(np.sqrt(_TIP_E * 0.00260184625))
_H_R0 = _TIP_R + 1.4430; _H_E0 = float(np.sqrt(_TIP_E * 0.00190802059))


def _split_params(R0, E0, q=0.0, q_tip=0.0, r_cut=6.0):
    from spammm.surfaces.PMESplit import SplitParams
    return SplitParams(R0=np.float64(R0), E0=np.float64(E0), q=np.float64(q),
                       alpha=_TIP_ALPHA, q_tip=q_tip, r_damp=_R_DAMP, r_cut=r_cut)


@pytest.mark.gpu
def test_contact_pme_split_oracle_parity(make_review):
    """Float64 combined_atom_potential (E,F) must match cs_brute_plqh_points OpenCL oracle."""
    from spammm.surfaces.ContactSurface import ContactSurfaceCL
    from spammm.surfaces.PMESplit import eval_atom_ef
    rv = make_review('test_contact_pme_split_oracle_parity')
    cs = ContactSurfaceCL(nloc=64)
    rng = np.random.default_rng(42)
    atom_pos = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    # (R0, E0, q_i, q_tip, label) — C/O/H Morse + positive/negative charge combos
    cases = [
        (_C_R0, _C_E0, 0.0, 0.0, 'C Morse-only'),
        (_C_R0, _C_E0, 0.5, 1.0, 'C q+ tip+'),
        (_C_R0, _C_E0, -0.5, 1.0, 'C q- tip+'),
        (_O_R0, _O_E0, 0.0, 0.0, 'O Morse-only'),
        (_O_R0, _O_E0, -0.3, 0.5, 'O q- tip+'),
        (_H_R0, _H_E0, 0.3, -0.5, 'H q+ tip-'),
    ]
    # Random queries at r in [1.5, 10] from atom (away from singularity, within scan range)
    queries = (atom_pos + rng.uniform(-9, 9, (300, 3)).astype(np.float32))
    dp0 = queries - atom_pos
    r0 = np.linalg.norm(dp0, axis=1)
    queries = queries[(r0 > 1.5) & (r0 < 10.0)]
    nq = len(queries)
    rv.out(f'Queries: {nq} points, r in [1.5, 10] from atom at {atom_pos}')
    rv.out(f'Constants: COULOMB_CONST=14.3996448915, r_damp={_R_DAMP}, alpha={_TIP_ALPHA}')
    rv.out(f'PLQH convention: (1, 1, q_tip, 0)')
    all_max_E = 0.0; all_max_F = 0.0
    for R0, E0, q_i, q_tip, label in cases:
        reqs = np.array([[R0, E0, q_i, 0.0]], dtype=np.float32)
        plqh = np.array([1.0, 1.0, q_tip, 0.0], dtype=np.float32)
        cs.setup_atoms(atom_pos.reshape(1, 3), reqs, alpha_morse=_TIP_ALPHA, r_damp=_R_DAMP, plqh=plqh)
        E_gpu, F_gpu = cs.eval_brute(queries)
        p = _split_params(R0, E0, q_i, q_tip=q_tip)
        E_py, F_py = eval_atom_ef(queries, atom_pos, p)
        err_E = np.abs(E_py - E_gpu.astype(np.float64))
        err_F = np.abs(F_py - F_gpu.astype(np.float64))
        max_E = float(np.max(err_E)); max_F = float(np.max(err_F))
        rms_E = float(np.sqrt(np.mean(err_E**2))); rms_F = float(np.sqrt(np.mean(err_F**2)))
        all_max_E = max(all_max_E, max_E); all_max_F = max(all_max_F, max_F)
        rv.out(f'  {label:20s}: max|dE|={max_E:.4e} rms|dE|={rms_E:.4e}  max|dF|={max_F:.4e} rms|dF|={rms_F:.4e}')
        rv.log(f'  {label}: E_gpu[:3]={E_gpu[:3]} E_py[:3]={E_py[:3]}')
        rv.log(f'  {label}: F_gpu[:2]={F_gpu[:2]} F_py[:2]={F_py[:2]}')
        # float32 GPU vs float64 CPU: rtol=1e-3, atol=1e-4 (generous for exp blowup)
        scale_E = max(1.0, float(np.max(np.abs(E_gpu))))
        scale_F = max(1.0, float(np.max(np.abs(F_gpu))))
        assert max_E < 1e-3 * scale_E + 1e-4, f'{label}: E parity failed max|dE|={max_E}'
        assert max_F < 1e-3 * scale_F + 1e-4, f'{label}: F parity failed max|dF|={max_F}'
    rv.out(f'\nOverall: max|dE|={all_max_E:.4e} max|dF|={all_max_F:.4e}')
    rv.checklist('Oracle parity vs cs_brute_plqh_points for C/O/H Morse + Coulomb',
                 'Force sign: F = -dv/dr * dp/r (probe force, matches kernel fe.xyz -= fej.xyz)',
                 'Damping: R2damp = r_damp^2 = 0.01, COULOMB_CONST = 14.3996448915',
                 'PLQH = (1,1,q_tip,0) convention used (not default (1,1,1,0))')
    rv.finish()


def test_contact_pme_split_formula_derivatives(make_review):
    """Analytical dv/dr and d²v/dr² must match finite differences (float64)."""
    from spammm.surfaces.PMESplit import combined_atom_potential
    rv = make_review('test_contact_pme_split_formula_derivatives')
    rng = np.random.default_rng(7)
    h = 1e-5  # FD step for float64
    cases = [(_C_R0, _C_E0, 0.0, 0.0, 'C Morse'), (_C_R0, _C_E0, 0.5, 1.0, 'C q+ tip+'),
             (_O_R0, _O_E0, -0.3, 0.5, 'O q- tip+'), (_H_R0, _H_E0, 0.0, 0.0, 'H Morse')]
    r_test = rng.uniform(1.0, 10.0, 100)
    all_err1 = 0.0; all_err2 = 0.0
    for R0, E0, q_i, q_tip, label in cases:
        p = _split_params(R0, E0, q_i, q_tip=q_tip)
        v, dv, d2v = combined_atom_potential(r_test, p)
        vp, _, _ = combined_atom_potential(r_test + h, p)
        vm, _, _ = combined_atom_potential(r_test - h, p)
        dv_fd = (vp - vm) / (2 * h)
        d2v_fd = (vp - 2 * v + vm) / (h * h)
        err1 = float(np.max(np.abs(dv - dv_fd)))
        err2 = float(np.max(np.abs(d2v - d2v_fd)))
        all_err1 = max(all_err1, err1); all_err2 = max(all_err2, err2)
        rv.out(f'  {label:20s}: max|dv-dv_fd|={err1:.4e} max|d2v-d2v_fd|={err2:.4e}')
        assert err1 < 1e-6, f'{label}: dv/dr FD mismatch {err1}'
        assert err2 < 1e-3, f'{label}: d²v/dr² FD mismatch {err2}'
    rv.out(f'\nOverall: max|dv-dv_fd|={all_err1:.4e} max|d2v-d2v_fd|={all_err2:.4e}')
    rv.checklist('Analytical first derivative matches FD (rtol<1e-6)',
                 'Analytical second derivative matches FD (rtol<1e-3)',
                 'Tested C/O/H Morse + charged Coulomb combinations')
    rv.finish()


def test_contact_pme_split_identity(make_review):
    """v_L + v_S = v exactly for r >= r_lo (split identity)."""
    from spammm.surfaces.PMESplit import soft_core_split
    rv = make_review('test_contact_pme_split_identity')
    rng = np.random.default_rng(11)
    cases = [(_C_R0, _C_E0, 0.0, 0.0, 'C'), (_O_R0, _O_E0, 0.5, 1.0, 'O q+'), (_H_R0, _H_E0, -0.3, 0.5, 'H q-')]
    all_err = 0.0
    for R0, E0, q_i, q_tip, label in cases:
        p = _split_params(R0, E0, q_i, q_tip=q_tip)
        r_lo = R0 - 0.5
        r = np.sort(rng.uniform(r_lo, r_lo + 8.0, 500))  # r >= r_lo
        s = soft_core_split(r, p)
        err_v = float(np.max(np.abs(s['v_L'] + s['v_S'] - s['v'])))
        err_dv = float(np.max(np.abs(s['dv_L_dr'] + s['dv_S_dr'] - s['dvdr'])))
        err_d2v = float(np.max(np.abs(s['d2v_L'] + s['d2v_S'] - s['d2v'])))
        all_err = max(all_err, err_v, err_dv, err_d2v)
        rv.out(f'  {label}: max|v_L+v_S-v|={err_v:.4e} max|dv_L+dv_S-dv|={err_dv:.4e} max|d2v_L+d2v_S-d2v|={err_d2v:.4e}')
        assert err_v < 1e-12, f'{label}: split identity v failed {err_v}'
        assert err_dv < 1e-12, f'{label}: split identity dv failed {err_dv}'
        assert err_d2v < 1e-12, f'{label}: split identity d2v failed {err_d2v}'
    rv.out(f'\nOverall max error: {all_err:.4e} (target < 1e-12)')
    rv.checklist('v_L + v_S = v exactly (< 1e-12)',
                 'dv_L + dv_S = dv exactly (< 1e-12)',
                 'd²v_L + d²v_S = d²v exactly (< 1e-12)')
    rv.finish()


def test_contact_pme_split_join_continuity(make_review):
    """C² continuity at r_lo and r_cut joins; v_S and derivatives vanish at r_cut."""
    from spammm.surfaces.PMESplit import soft_core_split, combined_atom_potential
    rv = make_review('test_contact_pme_split_join_continuity')
    eps = 1e-8  # small eps for one-sided limit comparison
    cases = [(_C_R0, _C_E0, 0.0, 0.0, 'C'), (_O_R0, _O_E0, 0.5, 1.0, 'O q+'), (_H_R0, _H_E0, -0.3, 0.5, 'H q-')]
    all_err = 0.0
    for R0, E0, q_i, q_tip, label in cases:
        p = _split_params(R0, E0, q_i, q_tip=q_tip)
        r_lo = R0 - 0.5; r_cut = 6.0
        # Test C² at r_lo: compare just below and just above
        r_below = np.array([r_lo - eps]); r_above = np.array([r_lo + eps])
        s_below = soft_core_split(r_below, p); s_above = soft_core_split(r_above, p)
        err_vL_lo = abs(s_below['v_L'][0] - s_above['v_L'][0])
        err_dvL_lo = abs(s_below['dv_L_dr'][0] - s_above['dv_L_dr'][0])
        err_d2vL_lo = abs(s_below['d2v_L'][0] - s_above['d2v_L'][0])
        # Test C² at r_cut: compare just below and just above
        r_below_rc = np.array([r_cut - eps]); r_above_rc = np.array([r_cut + eps])
        s_below_rc = soft_core_split(r_below_rc, p); s_above_rc = soft_core_split(r_above_rc, p)
        err_vL_rc = abs(s_below_rc['v_L'][0] - s_above_rc['v_L'][0])
        err_dvL_rc = abs(s_below_rc['dv_L_dr'][0] - s_above_rc['dv_L_dr'][0])
        err_d2vL_rc = abs(s_below_rc['d2v_L'][0] - s_above_rc['d2v_L'][0])
        # v_S and derivatives must vanish at r_cut (check from below, where split is defined)
        err_vS_rc = abs(s_below_rc['v_S'][0])
        err_dvS_rc = abs(s_below_rc['dv_S_dr'][0])
        err_d2vS_rc = abs(s_below_rc['d2v_S'][0])
        mx = max(err_vL_lo, err_dvL_lo, err_d2vL_lo, err_vL_rc, err_dvL_rc, err_d2vL_rc,
                 err_vS_rc, err_dvS_rc, err_d2vS_rc)
        all_err = max(all_err, mx)
        rv.out(f'  {label}: C²@r_lo: |d vL|={err_vL_lo:.4e} |d dvL|={err_dvL_lo:.4e} |d d2vL|={err_d2vL_lo:.4e}')
        rv.out(f'  {label}: C²@r_cut: |d vL|={err_vL_rc:.4e} |d dvL|={err_dvL_rc:.4e} |d d2vL|={err_d2vL_rc:.4e}')
        rv.out(f'  {label}: v_S@r_cut: |vS|={err_vS_rc:.4e} |dvS|={err_dvS_rc:.4e} |d2vS|={err_d2vS_rc:.4e}')
        # Tolerances account for O(eps) one-sided limit error with eps=1e-8
        assert err_vL_lo < 1e-10 and err_dvL_lo < 1e-10 and err_d2vL_lo < 1e-6, f'{label}: C² fail at r_lo'
        assert err_vL_rc < 1e-6 and err_dvL_rc < 1e-6 and err_d2vL_rc < 1e-4, f'{label}: C² fail at r_cut'
        assert err_vS_rc < 1e-10 and err_dvS_rc < 1e-6 and err_d2vS_rc < 1e-4, f'{label}: v_S not zero at r_cut'
    rv.out(f'\nOverall max join error: {all_err:.4e}')
    rv.checklist('v_L is C² at r_lo (flat inner boundary)',
                 'v_L is C² at r_cut (matches v exactly)',
                 'v_S, v_S\', v_S\'\' all vanish at r_cut')
    rv.finish()


def test_contact_pme_split_rho_monotonic(make_review):
    """rho(r) must be monotonically non-decreasing; drho/dr >= 0."""
    from spammm.surfaces.PMESplit import softened_rho
    rv = make_review('test_contact_pme_split_rho_monotonic')
    cases = [(_C_R0, 'C'), (_O_R0, 'O'), (_H_R0, 'H')]
    r_cut = 6.0
    all_err = 0.0
    for R0, label in cases:
        r_lo = R0 - 0.5
        r = np.linspace(r_lo - 1.0, r_cut + 1.0, 2000)
        rho, drho, d2rho = softened_rho(r, r_lo, r_cut)
        # Monotonicity: drho/dr >= 0 everywhere
        min_drho = float(np.min(drho))
        # rho non-decreasing
        drho_diff = np.diff(rho)
        min_drho_diff = float(np.min(drho_diff))
        all_err = max(all_err, -min_drho, -min_drho_diff)
        rv.out(f'  {label}: r_lo={r_lo:.4f} min(drho/dr)={min_drho:.4e} min(diff(rho))={min_drho_diff:.4e}')
        rv.log(f'  {label}: rho[0]={rho[0]:.6f} rho[-1]={rho[-1]:.6f} (expect {r_lo:.4f} and {r_cut + 1.0:.4f})')
        assert min_drho >= -1e-14, f'{label}: drho/dr < 0 ({min_drho})'
        assert min_drho_diff >= -1e-14, f'{label}: rho not monotonic ({min_drho_diff})'
        # Check boundary values
        assert abs(rho[0] - r_lo) < 1e-14, f'{label}: rho(r<r_lo) != r_lo'
        assert abs(rho[-1] - (r_cut + 1.0)) < 1e-14, f'{label}: rho(r>r_cut) != r'
    rv.out(f'\nOverall monotonicity violation: {all_err:.4e} (target < 1e-14)')
    rv.checklist('rho(r) is monotonically non-decreasing for C/O/H',
                 'rho(r<r_lo) = r_lo (constant), rho(r>r_cut) = r (identity)')
    rv.finish()


def test_contact_pme_split_domain_violation(make_review):
    """r < r_lo_i is a model-domain violation; detect and report for r_cut={4,5,6}."""
    from spammm.surfaces.PMESplit import domain_violation_mask, check_domain, soft_core_split
    rv = make_review('test_contact_pme_split_domain_violation')
    r_cuts = [4.0, 5.0, 6.0]
    cases = [(_C_R0, _C_E0, 'C'), (_O_R0, _O_E0, 'O'), (_H_R0, _H_E0, 'H')]
    for r_cut in r_cuts:
        rv.out(f'\n--- r_cut = {r_cut} Å ---')
        for R0, E0, label in cases:
            p = _split_params(R0, E0, r_cut=r_cut)
            r_lo = R0 - 0.5
            r_ok = np.array([r_lo + 0.1, r_lo + 1.0, r_cut - 0.5, r_cut + 1.0])
            r_bad = np.array([r_lo - 0.01, r_lo - 0.5, 0.1])
            mask_ok = domain_violation_mask(r_ok, p)
            mask_bad = domain_violation_mask(r_bad, p)
            rv.out(f'  {label} r_lo={r_lo:.4f}: ok_mask={mask_ok} bad_mask={mask_bad}')
            assert not np.any(mask_ok), f'{label}: false violation at r >= r_lo'
            assert np.all(mask_bad), f'{label}: missed violation at r < r_lo'
            # check_domain must raise on violations
            try:
                check_domain(r_bad, p)
                raise AssertionError(f'{label}: check_domain did not raise on r < r_lo')
            except ValueError as e:
                rv.log(f'  {label}: check_domain raised: {e}')
            # soft_core_split at r < r_lo: v_L = v(r_lo) constant, v_S undefined but computed
            s = soft_core_split(np.array([r_lo - 0.1]), p)
            rv.log(f'  {label}: v_L at r_lo-0.1 = {s["v_L"][0]:.6e} (expect v(r_lo)={s["v"][0]:.6e})')
    rv.checklist('domain_violation_mask correctly identifies r < r_lo',
                 'check_domain raises ValueError on violations (fail-loud)',
                 'Tested for r_cut = {4, 5, 6} Å with C/O/H atoms')
    rv.finish()


def test_contact_pme_split_charge_combinations(make_review):
    """C/O/H Morse parameters plus positive/negative charge combinations."""
    from spammm.surfaces.PMESplit import combined_atom_potential, soft_core_split
    rv = make_review('test_contact_pme_split_charge_combinations')
    # (R0, E0, q_i, q_tip, label)
    cases = [
        (_C_R0, _C_E0, 0.0, 0.0, 'C neutral'),
        (_C_R0, _C_E0, 0.5, 1.0, 'C+ tip+'),
        (_C_R0, _C_E0, -0.5, 1.0, 'C- tip+'),
        (_C_R0, _C_E0, 0.5, -1.0, 'C+ tip-'),
        (_O_R0, _O_E0, 0.3, 0.5, 'O+ tip+'),
        (_O_R0, _O_E0, -0.3, -0.5, 'O- tip-'),
        (_H_R0, _H_E0, 0.1, 0.0, 'H+ neutral-tip'),
        (_H_R0, _H_E0, 0.0, 0.5, 'H neutral tip+'),
    ]
    r = np.linspace(1.0, 10.0, 200)
    for R0, E0, q_i, q_tip, label in cases:
        p = _split_params(R0, E0, q_i, q_tip=q_tip)
        v, dv, d2v = combined_atom_potential(r, p)
        s = soft_core_split(r, p)
        # Sanity: v must be finite everywhere
        assert np.all(np.isfinite(v)), f'{label}: non-finite v'
        assert np.all(np.isfinite(s['v_L'])), f'{label}: non-finite v_L'
        assert np.all(np.isfinite(s['v_S'])), f'{label}: non-finite v_S'
        # At large r, Morse → 0, Coulomb → 0 (for q=0) or → 0 (1/r decay)
        assert abs(v[-1]) < 1.0, f'{label}: v not decaying at r=10'
        # Split identity on valid domain
        r_valid = r[r >= p.r_lo]
        s_valid = soft_core_split(r_valid, p)
        err = float(np.max(np.abs(s_valid['v_L'] + s_valid['v_S'] - s_valid['v'])))
        rv.out(f'  {label:25s}: v[0]={v[0]:.4e} v[-1]={v[-1]:.4e} split_err={err:.4e}')
        assert err < 1e-12
    rv.checklist('All C/O/H × charge combinations produce finite v, v_L, v_S',
                 'Split identity holds on valid domain for all combinations',
                 'Potential decays at large r')
    rv.finish()


def test_contact_pme_split_rcut_sweep(make_review):
    """r_cut sweep: reject r_cut <= max(r_lo), select smallest valid candidate."""
    from spammm.surfaces.PMESplit import r_cut_candidates, SplitParams
    rv = make_review('test_contact_pme_split_rcut_sweep')
    # Case 1: C/O/H atoms — all r_cut candidates valid
    p_normal = SplitParams(R0=np.array([_C_R0, _O_R0, _H_R0]), E0=np.array([_C_E0, _O_E0, _H_E0]),
                           q=np.array([0.0, 0.0, 0.0]), alpha=_TIP_ALPHA, q_tip=0.0, r_damp=_R_DAMP)
    valid, rejected, r_lo_max = r_cut_candidates(p_normal)
    rv.out(f'Normal atoms: r_lo_max={r_lo_max:.4f} valid={valid} rejected={rejected}')
    assert len(rejected) == 0, 'No rejections expected for C/O/H'
    assert valid == [4.0, 5.0, 6.0], 'All candidates valid for C/O/H'
    assert valid[0] == 4.0, 'Smallest valid candidate should be 4.0'
    # Case 2: large atom with R0=4.5 → r_lo=4.0 → r_cut=4 rejected (4 <= 4.0)
    p_large = SplitParams(R0=np.array([4.5]), E0=np.array([0.001]), q=np.array([0.0]),
                          alpha=_TIP_ALPHA, q_tip=0.0, r_damp=_R_DAMP)
    valid2, rejected2, r_lo_max2 = r_cut_candidates(p_large)
    rv.out(f'Large atom: r_lo_max={r_lo_max2:.4f} valid={valid2} rejected={rejected2}')
    assert 4.0 in rejected2, 'r_cut=4 should be rejected when r_lo_max=4.0'
    assert valid2 == [5.0, 6.0], 'Only 5 and 6 should be valid for large atom'
    assert valid2[0] == 5.0, 'Smallest valid candidate should be 5.0'
    # Case 3: very large atom with R0=5.6 → r_lo=5.1 → only r_cut=6 valid
    p_vlarge = SplitParams(R0=np.array([5.6]), E0=np.array([0.001]), q=np.array([0.0]),
                           alpha=_TIP_ALPHA, q_tip=0.0, r_damp=_R_DAMP)
    valid3, rejected3, r_lo_max3 = r_cut_candidates(p_vlarge)
    rv.out(f'Very large atom: r_lo_max={r_lo_max3:.4f} valid={valid3} rejected={rejected3}')
    assert valid3 == [6.0], 'Only r_cut=6 should be valid'
    assert set(rejected3) == {4.0, 5.0}, 'r_cut=4 and 5 should be rejected'
    rv.checklist('r_cut candidates rejected when r_cut <= max(r_lo_i)',
                 'Smallest valid candidate selected (ascending sort)',
                 'Tested normal C/O/H, large, and very-large atom cases',
                 'No |V/V\'| auto-selection used (singular at zeros)')
    rv.finish()


def test_contact_pme_split_l2_curves(visual_output_dir, make_review):
    """L2: optional split curves v, v_L, v_S vs r for C/O/H atoms (visual harness)."""
    from spammm.surfaces.PMESplit import soft_core_split
    rv = make_review('test_contact_pme_split_l2_curves')
    if visual_output_dir is None:
        rv.out('L2 split curves: skipped (no --visual/--develop)')
        rv.finish()
        return
    rv.out('L2 split curves: generating plots')
    outdir = _WAVE0_DIR
    os.makedirs(outdir, exist_ok=True)
    cases = [(_C_R0, _C_E0, 0.0, 0.0, 'C'), (_O_R0, _O_E0, 0.5, 1.0, 'O_q+'),
             (_H_R0, _H_E0, -0.3, 0.5, 'H_q-')]
    r = np.linspace(0.5, 10.0, 1000)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (R0, E0, q_i, q_tip, label) in zip(axes, cases):
        p = _split_params(R0, E0, q_i, q_tip=q_tip)
        r_lo = R0 - 0.5
        r_valid = r[r >= r_lo]
        s = soft_core_split(r_valid, p)
        ax.plot(r_valid, s['v'], 'k-', label='v (total)', lw=1.5)
        ax.plot(r_valid, s['v_L'], 'b--', label='v_L (soft/mesh)', lw=1.2)
        ax.plot(r_valid, s['v_S'], 'r:', label='v_S (core residual)', lw=1.2)
        ax.axvline(r_lo, color='gray', ls=':', alpha=0.5, label=f'r_lo={r_lo:.2f}')
        ax.axvline(6.0, color='gray', ls='--', alpha=0.5, label='r_cut=6.0')
        ax.set_title(f'{label} (R0={R0:.2f}, E0={E0:.4e})')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('V [eV]')
        ax.legend(fontsize=7); ax.set_ylim(bottom=min(s['v'].min(), s['v_S'].min()) * 1.1,
                                           top=max(s['v'].max(), 0.01))
    fig.suptitle('PMESplit: combined potential and soft-core split (Wave 0)', fontsize=11)
    plot_path = os.path.join(outdir, 'split_curves.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    rv.out(f'Saved L2 plot: {plot_path}')
    print(f'REVIEW: {plot_path}', flush=True)
    rv.finish()


def test_contact_pme_split_summary(make_review):
    """Emit SUMMARY.out with accepted parameter map for Wave 0."""
    from spammm.surfaces.PMESplit import r_cut_candidates, SplitParams, COULOMB_CONST
    rv = make_review('test_contact_pme_split_summary')
    outdir = _WAVE0_DIR
    os.makedirs(outdir, exist_ok=True)
    p = SplitParams(R0=np.array([_C_R0, _O_R0, _H_R0]), E0=np.array([_C_E0, _O_E0, _H_E0]),
                    q=np.array([0.0, 0.0, 0.0]), alpha=_TIP_ALPHA, q_tip=0.0, r_damp=_R_DAMP, r_cut=6.0)
    valid, rejected, r_lo_max = r_cut_candidates(p)
    lines = [
        '# PMESplit Wave 0 SUMMARY.out — accepted parameter map',
        f'# Contract version: 2',
        f'# COULOMB_CONST = {COULOMB_CONST}',
        f'#',
        f'# Global tip parameters:',
        f'alpha (tip stiffness) = {_TIP_ALPHA}',
        f'K = -alpha = {-_TIP_ALPHA}',
        f'r_damp = {_R_DAMP} Å',
        f'R2damp = {_R_DAMP**2} Å²',
        f'r_cut (MVP default) = 6.0 Å',
        f'PLQH convention = (1, 1, q_tip, 0)',
        f'',
        f'# Per-atom parameters (from assign_params combination rules):',
        f'# R0 = tip_R + R_vdW_sample, E0 = sqrt(tip_E * E_vdW_sample)',
        f'tip_R = {_TIP_R}, tip_E = {_TIP_E}',
        f'',
        f'# Atom  R0[Å]    E0[eV]       r_lo[Å]  q[e]',
        f'C      {_C_R0:.4f}  {_C_E0:.6e}  {_C_R0 - 0.5:.4f}  0.0',
        f'O      {_O_R0:.4f}  {_O_E0:.6e}  {_O_R0 - 0.5:.4f}  0.0',
        f'H      {_H_R0:.4f}  {_H_E0:.6e}  {_H_R0 - 0.5:.4f}  0.0',
        f'',
        f'# r_cut sweep (candidates = {{4, 5, 6}} Å, h_mesh = 1 Å):',
        f'r_lo_max = {r_lo_max:.4f} Å',
        f'valid candidates = {valid} (ascending, smallest first for gate selection)',
        f'rejected candidates = {rejected} (r_cut <= r_lo_max)',
        f'accepted r_cut = {valid[0]} Å (smallest valid; final selection by downstream gates)',
        f'',
        f'# Oracle: getMorsePLQH / cs_brute_plqh_points (kernels/Forces.cl:235-249)',
        f'# Split: quintic Hermite rho(r), v_L = v(rho), v_S = v - v_L',
        f'# Domain violation: r < r_lo_i → ValueError (fail-loud, no extrapolation)',
    ]
    summary_path = os.path.join(outdir, 'SUMMARY.out')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    rv.out(f'Summary written to {summary_path}')
    rv.out('\n'.join(lines))
    print(f'REVIEW: {summary_path}', flush=True)
    rv.checklist('SUMMARY.out emitted with accepted parameter map',
                 'r_cut sweep results recorded',
                 'Oracle and split conventions documented')
    rv.finish()


# ════════════════════════════════════════════════════════════════════════════
# Wave 1: PICCore — compact atom-centered radial core (Agent_3)
# Contract version 2. Fits analytic v_i^S with doubling-power basis.
# ════════════════════════════════════════════════════════════════════════════

_WAVE1_CORE_DIR = os.path.join('debug', 'test_afm_contact_surface', 'contact_pme', 'wave1_core')


def _core_split_params(R0, E0, q=0.0, q_tip=0.0, r_cut=6.0):
    from spammm.surfaces.PMESplit import SplitParams
    return SplitParams(R0=np.float64(R0), E0=np.float64(E0), q=np.float64(q),
                       alpha=_TIP_ALPHA, q_tip=q_tip, r_damp=_R_DAMP, r_cut=r_cut)


def test_contact_pme_core_fit_quality(make_review):
    """fit_core_1d must fit v_S with energy+force rows; report conditioning and held-out errors."""
    from spammm.surfaces.PICCore import fit_core_1d, N_MODES, CORE_POWERS
    rv = make_review('test_contact_pme_core_fit_quality')
    cases = [
        (_C_R0, _C_E0, 0.0, 0.0, 'C neutral'),
        (_O_R0, _O_E0, 0.0, 0.0, 'O neutral'),
        (_H_R0, _H_E0, 0.0, 0.0, 'H neutral'),
        (_C_R0, _C_E0, 0.5, 1.0, 'C q+ tip+'),
        (_O_R0, _O_E0, -0.3, 0.5, 'O q- tip+'),
    ]
    rv.out(f'N_MODES={N_MODES}, powers={CORE_POWERS.tolist()}')
    rv.out(f'{"label":20s} {"cond_raw":>10s} {"cond_hier":>10s} {"train_E":>12s} {"train_F":>12s} {"held_E":>12s} {"held_F":>12s} {"max_E":>12s} {"max_F":>12s} {"worst_r":>8s}')
    for R0, E0, q_i, q_tip, label in cases:
        p = _core_split_params(R0, E0, q_i, q_tip=q_tip)
        fit = fit_core_1d(p)
        rv.out(f'{label:20s} {fit.cond_raw[0]:10.1f} {fit.cond_hier[0]:10.1f} {fit.train_rmse_E[0]:12.4e} {fit.train_rmse_F[0]:12.4e} {fit.held_rmse_E[0]:12.4e} {fit.held_rmse_F[0]:12.4e} {fit.held_max_E[0]:12.4e} {fit.held_max_F[0]:12.4e} {fit.worst_r[0]:8.3f}')
        assert fit.coeffs.shape == (1, N_MODES), f'{label}: wrong coeff shape'
        assert np.all(np.isfinite(fit.coeffs)), f'{label}: non-finite coeffs'
        assert fit.cond_raw[0] > 1.0, f'{label}: cond_raw must be > 1'
        assert fit.cond_hier[0] > 1.0, f'{label}: cond_hier must be > 1'
        rv.log(f'{label}: cond_hier/cond_raw = {fit.cond_hier[0]/fit.cond_raw[0]:.3f}')
    rv.checklist('fit_core_1d produces N_MODES=5 coefficients per atom',
                 'Energy AND radial-derivative rows used in least-squares',
                 'Condition numbers reported for raw and hierarchical bases',
                 'Held-out energy and force errors reported (NOT training energy alone)',
                 'Worst held-out radius identified per atom',
                 'C/O/H neutral and charged cases tested')
    rv.finish()


def test_contact_pme_core_exact_cutoff(make_review):
    """Core must be exactly zero at and beyond r_cut (no tail leakage)."""
    from spammm.surfaces.PICCore import fit_core_1d, eval_core
    rv = make_review('test_contact_pme_core_exact_cutoff')
    p = _core_split_params(_C_R0, _C_E0)
    fit = fit_core_1d(p)
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    r_test = np.array([6.0, 6.001, 7.0, 10.0, 100.0])
    queries = np.column_stack([r_test, np.zeros(len(r_test)), np.zeros(len(r_test))])
    E, F = eval_core(queries, atom_pos, fit)
    rv.out(f'E at/above r_cut: {E}')
    rv.out(f'F at/above r_cut: {F}')
    assert np.allclose(E, 0.0, atol=0.0), f'Core E must be exactly 0 at r>=r_cut, got {E}'
    assert np.allclose(F, 0.0, atol=0.0), f'Core F must be exactly 0 at r>=r_cut, got {F}'
    rv.checklist('Core energy is exactly 0 at r >= r_cut',
                 'Core force is exactly 0 at r >= r_cut',
                 'No tail leakage beyond cutoff')
    rv.finish()


def test_contact_pme_core_force_parity(make_review):
    """F = -∇E: finite-difference force must match analytic force (L0 invariant)."""
    from spammm.surfaces.PICCore import fit_core_1d, eval_core
    rv = make_review('test_contact_pme_core_force_parity')
    p = _core_split_params(_C_R0, _C_E0)
    fit = fit_core_1d(p)
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    h = 1e-5
    rng = np.random.default_rng(99)
    r_vals = rng.uniform(3.5, 5.5, 20)
    theta = rng.uniform(0, np.pi, 20)
    phi = rng.uniform(0, 2 * np.pi, 20)
    queries = np.column_stack([r_vals * np.sin(theta) * np.cos(phi),
                               r_vals * np.sin(theta) * np.sin(phi),
                               r_vals * np.cos(theta)])
    E0, F0 = eval_core(queries, atom_pos, fit)
    max_err = 0.0
    for ic in range(3):
        q_p = queries.copy(); q_p[:, ic] += h
        q_m = queries.copy(); q_m[:, ic] -= h
        E_p, _ = eval_core(q_p, atom_pos, fit)
        E_m, _ = eval_core(q_m, atom_pos, fit)
        F_fd = -(E_p - E_m) / (2 * h)
        err = np.max(np.abs(F0[:, ic] - F_fd))
        max_err = max(max_err, float(err))
        rv.out(f'  F component {ic}: max|F_analytic - F_fd| = {float(err):.4e}')
    rv.out(f'\nOverall max force parity error: {max_err:.4e}')
    assert max_err < 1e-4, f'Force parity failed: max|F-Fd|={max_err}'
    rv.checklist('F = -∇E verified by finite differences (L0 invariant)',
                 'Tested at 20 random 3D query points at safe distance',
                 'FD step h=1e-5 for float64')
    rv.finish()


def test_contact_pme_core_bucket_completeness(make_review):
    """Bucket-based eval_core must match direct all-atom sum (no silent truncation)."""
    from spammm.surfaces.PICCore import fit_core_1d, eval_core, eval_core_direct
    rv = make_review('test_contact_pme_core_bucket_completeness')
    atom_pos = np.array([
        [0.0, 0.0, 0.0], [3.0, 1.0, 0.5], [-2.0, 2.0, 0.0],
        [1.5, -1.0, 0.3], [4.0, 3.0, 0.0],
    ])
    R0s = np.array([_C_R0, _O_R0, _C_R0, _H_R0, _O_R0])
    E0s = np.array([_C_E0, _O_E0, _C_E0, _H_E0, _O_E0])
    p = _core_split_params_multi(R0s, E0s)
    fit = fit_core_1d(p)
    z_probe = 4.0
    rng = np.random.default_rng(77)
    nq = 50
    queries = np.column_stack([rng.uniform(-3, 6, nq), rng.uniform(-2, 5, nq), np.full(nq, z_probe)])
    safe = np.ones(nq, dtype=bool)
    for iq in range(nq):
        for ia in range(len(atom_pos)):
            r = np.linalg.norm(queries[iq] - atom_pos[ia])
            if r < float(p.r_lo[ia]):
                safe[iq] = False
                break
    queries = queries[safe]
    rv.out(f'Queries: {len(queries)} safe points (filtered from {nq})')
    E_b, F_b = eval_core(queries, atom_pos, fit)
    E_d, F_d = eval_core_direct(queries, atom_pos, fit)
    err_E = float(np.max(np.abs(E_b - E_d)))
    err_F = float(np.max(np.abs(F_b - F_d)))
    rv.out(f'Bucket vs direct: max|dE|={err_E:.4e} max|dF|={err_F:.4e}')
    assert err_E < 1e-12, f'Bucket completeness E failed: {err_E}'
    assert err_F < 1e-12, f'Bucket completeness F failed: {err_F}'
    dense_pos = np.array([[i * 0.3, j * 0.3, 0.0] for i in range(5) for j in range(5)])
    R0s_d = np.full(25, _C_R0)
    E0s_d = np.full(25, _C_E0)
    p_d = _core_split_params_multi(R0s_d, E0s_d)
    fit_d = fit_core_1d(p_d)
    q_dense = np.array([[0.6, 0.6, 4.0], [1.2, 0.9, 4.0], [0.3, 0.3, 5.0]])
    E_b2, F_b2 = eval_core(q_dense, dense_pos, fit_d)
    E_d2, F_d2 = eval_core_direct(q_dense, dense_pos, fit_d)
    err_E2 = float(np.max(np.abs(E_b2 - E_d2)))
    err_F2 = float(np.max(np.abs(F_b2 - F_d2)))
    rv.out(f'Dense cell (25 atoms): max|dE|={err_E2:.4e} max|dF|={err_F2:.4e}')
    assert err_E2 < 1e-12, f'Dense cell E failed: {err_E2}'
    assert err_F2 < 1e-12, f'Dense cell F failed: {err_F2}'
    q_boundary = np.array([[0.0, 0.0, 4.0], [6.0, 0.0, 4.0]])
    E_b3, F_b3 = eval_core(q_boundary, dense_pos, fit_d)
    E_d3, F_d3 = eval_core_direct(q_boundary, dense_pos, fit_d)
    err_E3 = float(np.max(np.abs(E_b3 - E_d3)))
    err_F3 = float(np.max(np.abs(F_b3 - F_d3)))
    rv.out(f'Boundary query: max|dE|={err_E3:.4e} max|dF|={err_F3:.4e}')
    assert err_E3 < 1e-12, f'Boundary E failed: {err_E3}'
    assert err_F3 < 1e-12, f'Boundary F failed: {err_F3}'
    rv.checklist('Bucket-based eval_core matches direct all-atom sum exactly',
                 'Dense cell (25 atoms in small area) tested — no silent truncation',
                 'Bucket boundary queries tested',
                 'No silent candidate drop (fail-loud on invalid index)')
    rv.finish()


def test_contact_pme_core_ptcda_fit(xyz, make_review):
    """Fit core for PTCDA molecule; report per-atom errors and conditioning."""
    from spammm.surfaces.PICCore import fit_core_1d, eval_core, eval_core_direct, N_MODES
    from spammm.surfaces.PMESplit import SplitParams
    rv = make_review('test_contact_pme_core_ptcda_fit')
    afm = _make_afm(xyz('PTCDA.xyz'))
    apos = afm.atoms_arr[:, :3].astype(np.float64)
    cMs = afm.cLJs_arr
    qs = afm.atoms_arr[:, 3].astype(np.float64) if afm.atoms_arr.shape[1] > 3 else np.zeros(len(apos))
    na = len(apos)
    rv.out(f'PTCDA: {na} atoms, R0 range: {cMs[:,0].min():.4f} to {cMs[:,0].max():.4f}')
    rv.out(f'q range: {qs.min():.4f} to {qs.max():.4f}')
    p = SplitParams(R0=cMs[:,0].astype(np.float64), E0=cMs[:,1].astype(np.float64),
                    q=qs, alpha=_TIP_ALPHA, q_tip=0.0, r_damp=_R_DAMP, r_cut=6.0)
    fit = fit_core_1d(p)
    assert fit.coeffs.shape == (na, N_MODES), f'coeff shape mismatch: {fit.coeffs.shape}'
    rv.out(f'\nPer-atom summary (first 10 atoms):')
    rv.out(f'{"atom":>4s} {"R0":>8s} {"r_lo":>8s} {"cond_raw":>10s} {"cond_hier":>10s} {"held_E":>12s} {"held_F":>12s} {"max_E":>12s} {"max_F":>12s} {"worst_r":>8s}')
    for i in range(min(10, na)):
        rv.out(f'{i:4d} {cMs[i,0]:8.4f} {float(p.r_lo[i]):8.4f} {fit.cond_raw[i]:10.1f} {fit.cond_hier[i]:10.1f} {fit.held_rmse_E[i]:12.4e} {fit.held_rmse_F[i]:12.4e} {fit.held_max_E[i]:12.4e} {fit.held_max_F[i]:12.4e} {fit.worst_r[i]:8.3f}')
    rv.out(f'\nAggregate: cond_raw=[{fit.cond_raw.min():.1f}, {fit.cond_raw.max():.1f}]')
    rv.out(f'           cond_hier=[{fit.cond_hier.min():.1f}, {fit.cond_hier.max():.1f}]')
    rv.out(f'           held_rmse_E=[{fit.held_rmse_E.min():.4e}, {fit.held_rmse_E.max():.4e}]')
    rv.out(f'           held_rmse_F=[{fit.held_rmse_F.min():.4e}, {fit.held_rmse_F.max():.4e}]')
    z_probe = float(apos[:,2].max()) + 3.0
    rng = np.random.default_rng(42)
    queries = np.column_stack([apos[:,0] + rng.uniform(-2,2,na), apos[:,1] + rng.uniform(-2,2,na), np.full(na, z_probe)])
    safe = np.ones(na, dtype=bool)
    for iq in range(na):
        for ia in range(na):
            r = np.linalg.norm(queries[iq] - apos[ia])
            if r < float(p.r_lo[ia]):
                safe[iq] = False; break
    queries = queries[safe]
    E_b, F_b = eval_core(queries, apos, fit)
    E_d, F_d = eval_core_direct(queries, apos, fit)
    err_E = float(np.max(np.abs(E_b - E_d)))
    err_F = float(np.max(np.abs(F_b - F_d)))
    rv.out(f'\nBucket vs direct ({len(queries)} queries): max|dE|={err_E:.4e} max|dF|={err_F:.4e}')
    assert err_E < 1e-10, f'PTCDA bucket completeness E failed: {err_E}'
    assert err_F < 1e-10, f'PTCDA bucket completeness F failed: {err_F}'
    rv.checklist('PTCDA core fit produces correct coeff shape (38, 5)',
                 'Per-atom conditioning and held-out errors reported',
                 'Bucket vs direct sum exact on PTCDA queries')
    rv.finish()


def test_contact_pme_core_domain_violation(make_review):
    """eval_core must raise ValueError on r < r_lo (fail-loud, no silent extrapolation)."""
    from spammm.surfaces.PICCore import fit_core_1d, eval_core
    rv = make_review('test_contact_pme_core_domain_violation')
    p = _core_split_params(_C_R0, _C_E0)
    fit = fit_core_1d(p)
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    r_lo = _C_R0 - 0.5
    q_bad = np.array([[r_lo - 0.1, 0.0, 0.0]])
    raised = False
    try:
        eval_core(q_bad, atom_pos, fit)
    except ValueError as e:
        rv.out(f'Domain violation raised: {e}')
        raised = True
    assert raised, 'eval_core must raise ValueError on r < r_lo'
    q_ok = np.array([[r_lo + 0.01, 0.0, 0.0]])
    E_ok, F_ok = eval_core(q_ok, atom_pos, fit)
    rv.out(f'At r_lo+0.01: E={E_ok[0]:.6e} F={F_ok[0]}')
    assert np.all(np.isfinite(E_ok)), 'E must be finite at r >= r_lo'
    assert np.all(np.isfinite(F_ok)), 'F must be finite at r >= r_lo'
    rv.checklist('eval_core raises ValueError on r < r_lo (fail-loud)',
                 'eval_core works at r >= r_lo (boundary OK)',
                 'No silent extrapolation or fallback')
    rv.finish()


def test_contact_pme_core_l2_curves(visual_output_dir, make_review):
    """L2: residual reference vs fit; combined wall/well/tail plot."""
    from spammm.surfaces.PICCore import fit_core_1d, core_basis, eval_core_and_soft
    from spammm.surfaces.PMESplit import soft_core_split, combined_atom_potential, SplitParams
    rv = make_review('test_contact_pme_core_l2_curves')
    if visual_output_dir is None:
        rv.out('L2 core curves: skipped (no --visual/--develop)')
        rv.finish()
        return
    rv.out('L2 core curves: generating plots')
    outdir = _WAVE1_CORE_DIR
    os.makedirs(outdir, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = [(_C_R0, _C_E0, 0.0, 0.0, 'C'), (_O_R0, _O_E0, 0.0, 0.0, 'O'), (_H_R0, _H_E0, 0.0, 0.0, 'H')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (R0, E0, q_i, q_tip, label) in zip(axes, cases):
        p = _core_split_params(R0, E0, q_i, q_tip=q_tip)
        fit = fit_core_1d(p)
        r_lo = R0 - 0.5; r_cut = 6.0
        r = np.linspace(r_lo, r_cut, 500)
        s = soft_core_split(r, p)
        phi, dphi = core_basis(r, r_lo, r_cut)
        v_fit = phi @ fit.coeffs[0]
        ax.plot(r, s['v_S'], 'k-', label='v_S (ref)', lw=1.5)
        ax.plot(r, v_fit, 'r--', label='v_S (fit)', lw=1.2)
        ax.axhline(0, color='gray', ls=':', alpha=0.3)
        ax.axvline(r_lo, color='gray', ls=':', alpha=0.5, label=f'r_lo={r_lo:.2f}')
        ax.axvline(r_cut, color='gray', ls='--', alpha=0.5, label='r_cut=6.0')
        ax.set_title(f'{label} residual (R0={R0:.2f})')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('V_S [eV]')
        ax.legend(fontsize=7)
    fig.suptitle('PICCore: residual reference vs fit (Wave 1)', fontsize=11)
    plot_path = os.path.join(outdir, 'core_residual_fit.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    rv.out(f'Saved: {plot_path}')
    print(f'REVIEW: {plot_path}', flush=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (R0, E0, q_i, q_tip, label) in zip(axes2, cases):
        p = _core_split_params(R0, E0, q_i, q_tip=q_tip)
        fit = fit_core_1d(p)
        r_lo = R0 - 0.5; r_cut = 6.0
        atom_pos = np.array([[0.0, 0.0, 0.0]])
        r = np.linspace(r_lo, 10.0, 500)
        queries = np.column_stack([r, np.zeros(len(r)), np.zeros(len(r))])
        E_comb, F_comb = eval_core_and_soft(queries, atom_pos, p, fit)
        v_direct, _, _ = combined_atom_potential(r, p)
        ax.plot(r, v_direct, 'k-', label='v (direct total)', lw=1.5)
        ax.plot(r, E_comb, 'r--', label='v_L + core (combined)', lw=1.2)
        ax.axvline(r_lo, color='gray', ls=':', alpha=0.5, label=f'r_lo={r_lo:.2f}')
        ax.axvline(r_cut, color='gray', ls='--', alpha=0.5, label='r_cut=6.0')
        ax.set_title(f'{label} wall/well/tail (R0={R0:.2f})')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('V [eV]')
        ax.legend(fontsize=7)
    fig2.suptitle('PICCore: combined v_L(direct) + fitted core (Wave 1)', fontsize=11)
    plot_path2 = os.path.join(outdir, 'core_combined_wall_well_tail.png')
    fig2.savefig(plot_path2, dpi=150)
    plt.close(fig2)
    rv.out(f'Saved: {plot_path2}')
    print(f'REVIEW: {plot_path2}', flush=True)
    rv.checklist('L2 residual reference vs fit plot generated',
                 'L2 combined wall/well/tail plot generated (v_L direct + core)',
                 'Plots saved to wave1_core/ directory')
    rv.finish()


def test_contact_pme_core_summary(make_review):
    """Emit SUMMARY.out with core fit results for Wave 1."""
    from spammm.surfaces.PICCore import fit_core_1d, N_MODES, CORE_POWERS
    from spammm.surfaces.PMESplit import SplitParams
    rv = make_review('test_contact_pme_core_summary')
    outdir = _WAVE1_CORE_DIR
    os.makedirs(outdir, exist_ok=True)
    lines = [
        '# PICCore Wave 1 SUMMARY.out — compact radial core fit',
        f'# Contract version: 2',
        f'#',
        f'# Basis: phi_m(r) = t^p_m, t = (r_cut - r)/(r_cut - r_lo_i)',
        f'# Powers: {CORE_POWERS.tolist()}',
        f'# N_MODES = {N_MODES}',
        f'# Cutoff: every mode exactly zero for r >= r_cut',
        f'#',
        f'# Fit: energy + radial-derivative rows at ALL training radii',
        f'# Sampling: 300+ nonuniform shell points, dense endpoints',
        f'# Selection: raw vs hierarchical by condition + held-out E+F error',
        f'#',
        f'# Per-atom results (C/O/H neutral + charged):',
    ]
    cases = [
        (_C_R0, _C_E0, 0.0, 0.0, 'C'), (_O_R0, _O_E0, 0.0, 0.0, 'O'), (_H_R0, _H_E0, 0.0, 0.0, 'H'),
        (_C_R0, _C_E0, 0.5, 1.0, 'C_q+'), (_O_R0, _O_E0, -0.3, 0.5, 'O_q-'),
    ]
    for R0, E0, q_i, q_tip, label in cases:
        p = _core_split_params(R0, E0, q_i, q_tip=q_tip)
        fit = fit_core_1d(p)
        lines.append(f'{label:8s}: R0={R0:.4f} r_lo={R0-0.5:.4f} cond_raw={fit.cond_raw[0]:.1f} cond_hier={fit.cond_hier[0]:.1f} held_E={fit.held_rmse_E[0]:.4e} held_F={fit.held_rmse_F[0]:.4e}')
    lines.append(f'')
    lines.append(f'# Bucket: build_pic_buckets with cell_size >= r_cut, 3x3 lookup')
    lines.append(f'# Completeness: eval_core == eval_core_direct (exact, no truncation)')
    lines.append(f'# Domain: r < r_lo → ValueError (fail-loud)')
    lines.append(f'# Force: F = -∇E, finite-difference parity verified')
    summary_path = os.path.join(outdir, 'SUMMARY.out')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    rv.out(f'Summary written to {summary_path}')
    rv.out('\n'.join(lines))
    print(f'REVIEW: {summary_path}', flush=True)
    rv.checklist('SUMMARY.out emitted with core fit results',
                 'Per-atom conditioning and held-out errors recorded',
                 'Buckets, domain, and force conventions documented')
    rv.finish()


def _core_split_params_multi(R0s, E0s, qs=None, q_tip=0.0, r_cut=6.0):
    from spammm.surfaces.PMESplit import SplitParams
    if qs is None:
        qs = np.zeros(len(R0s))
    return SplitParams(R0=np.asarray(R0s, dtype=np.float64), E0=np.asarray(E0s, dtype=np.float64),
                       q=np.asarray(qs, dtype=np.float64), alpha=_TIP_ALPHA, q_tip=q_tip,
                       r_damp=_R_DAMP, r_cut=r_cut)


# ════════════════════════════════════════════════════════════════════════════
# Wave 1: CoarseMesh — coarse 3D B-spline mesh (Agent_2)
# Contract version 2. Stores V_L = Σ v_i^L on a coarse 3D grid.
# ════════════════════════════════════════════════════════════════════════════

_WAVE1_MESH_DIR = os.path.join('debug', 'test_afm_contact_surface', 'contact_pme', 'wave1_mesh')


def _mesh_split_params(R0, E0, q=0.0, q_tip=0.0, r_cut=6.0):
    from spammm.surfaces.PMESplit import SplitParams
    return SplitParams(R0=np.float64(R0), E0=np.float64(E0), q=np.float64(q),
                       alpha=_TIP_ALPHA, q_tip=q_tip, r_damp=_R_DAMP, r_cut=r_cut)


def _mesh_split_params_multi(R0s, E0s, qs=None, q_tip=0.0, r_cut=6.0):
    from spammm.surfaces.PMESplit import SplitParams
    if qs is None:
        qs = np.zeros(len(R0s))
    return SplitParams(R0=np.asarray(R0s, dtype=np.float64), E0=np.asarray(E0s, dtype=np.float64),
                       q=np.asarray(qs, dtype=np.float64), alpha=_TIP_ALPHA, q_tip=q_tip,
                       r_damp=_R_DAMP, r_cut=r_cut)


def test_contact_pme_mesh_constant_field(make_review):
    """A constant field must interpolate exactly (partition of unity)."""
    from spammm.surfaces.CoarseMesh import CoarseMesh, eval_mesh
    rv = make_review('test_contact_pme_mesh_constant_field')
    nx, ny, nz = 10, 10, 10
    h = 1.0; origin = np.array([0.0, 0.0, 0.0])
    const_val = 3.14
    coeffs = np.full((nx, ny, nz), const_val, dtype=np.float64)
    mesh = CoarseMesh(coeffs=coeffs, origin=origin, h=h, halo=2,
                      query_interior=(np.array([2, 2, 2]), np.array([7, 7, 7])))
    rng = np.random.default_rng(42)
    queries = np.column_stack([rng.uniform(2.5, 6.5, 30), rng.uniform(2.5, 6.5, 30), rng.uniform(2.5, 6.5, 30)])
    E, F = eval_mesh(mesh, queries)
    err_E = float(np.max(np.abs(E - const_val)))
    err_F = float(np.max(np.abs(F)))
    rv.out(f'Constant field: max|E - c| = {err_E:.4e}, max|F| = {err_F:.4e}')
    assert err_E < 1e-12, f'Constant field E failed: {err_E}'
    assert err_F < 1e-12, f'Constant field F failed: {err_F}'
    rv.checklist('Constant field interpolates exactly (partition of unity)',
                 'Force is exactly zero on a constant field')
    rv.finish()


def test_contact_pme_mesh_affine_field(make_review):
    """An affine field E = a*x + b*y + c*z + d must interpolate exactly in the interior; F = -(a,b,c)."""
    from spammm.surfaces.CoarseMesh import CoarseMesh, eval_mesh
    rv = make_review('test_contact_pme_mesh_affine_field')
    # Large grid so queries are far from zero-padding boundaries
    nx, ny, nz = 30, 30, 30
    h = 1.0; origin = np.array([0.0, 0.0, 0.0])
    a, b, c, d = 1.5, -2.0, 0.7, 10.0
    xs = origin[0] + np.arange(nx) * h
    ys = origin[1] + np.arange(ny) * h
    zs = origin[2] + np.arange(nz) * h
    samples = (a * xs[:, None, None] + b * ys[None, :, None] + c * zs[None, None, :] + d)
    from spammm.surfaces.ContactSurface import _bspline_prefilter_1d
    coeffs = samples.copy().astype(np.float64)
    for ix in range(nx):
        for iy in range(ny):
            coeffs[ix, iy, :] = _bspline_prefilter_1d(coeffs[ix, iy, :])
    for ix in range(nx):
        for iz in range(nz):
            coeffs[ix, :, iz] = _bspline_prefilter_1d(coeffs[ix, :, iz])
    for iy in range(ny):
        for iz in range(nz):
            coeffs[:, iy, iz] = _bspline_prefilter_1d(coeffs[:, iy, iz])
    mesh = CoarseMesh(coeffs=coeffs, origin=origin, h=h, halo=2,
                      query_interior=(np.array([5, 5, 5]), np.array([24, 24, 24])))
    rng = np.random.default_rng(99)
    # Query deep in the interior (zero-padding boundary effect decays as ~4^-distance)
    queries = np.column_stack([rng.uniform(10.0, 20.0, 30), rng.uniform(10.0, 20.0, 30), rng.uniform(10.0, 20.0, 30)])
    E, F = eval_mesh(mesh, queries)
    E_exact = a * queries[:, 0] + b * queries[:, 1] + c * queries[:, 2] + d
    F_exact = np.tile([-a, -b, -c], (len(queries), 1))
    err_E = float(np.max(np.abs(E - E_exact)))
    err_F = float(np.max(np.abs(F - F_exact)))
    rv.out(f'Affine field: max|E - exact| = {err_E:.4e}, max|F - exact| = {err_F:.4e}')
    # Zero-padding prefilter does NOT reproduce affine exactly — boundary effect
    # decays as ~4^-distance. At 10 nodes from edge, 3D compounded error ~1e-5.
    assert err_E < 1e-4, f'Affine field E failed: {err_E}'
    assert err_F < 1e-4, f'Affine field F failed: {err_F}'
    rv.checklist('Affine field interpolates accurately in the deep interior (cubic B-spline reproduces affine)',
                 'Analytic gradient F = -(a,b,c) matches accurately',
                 'Zero-padding boundary effect negligible at 10+ nodes from edge (~1e-5)')
    rv.finish()


def test_contact_pme_mesh_force_parity(make_review):
    """F = -∇E: finite-difference force must match analytic force (L0 invariant)."""
    from spammm.surfaces.CoarseMesh import build_coarse_mesh, eval_mesh
    rv = make_review('test_contact_pme_mesh_force_parity')
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    p = _mesh_split_params(_C_R0, _C_E0)
    qb = np.array([[-3.0, 3.0], [-3.0, 3.0], [3.0, 7.0]])
    mesh = build_coarse_mesh(atom_pos, p, qb, h_mesh=1.0, halo_nodes=6)
    h = 1e-5
    rng = np.random.default_rng(77)
    queries = np.column_stack([rng.uniform(-1.5, 1.5, 15), rng.uniform(-1.5, 1.5, 15), rng.uniform(4.0, 6.0, 15)])
    E0, F0 = eval_mesh(mesh, queries)
    max_err = 0.0
    for ic in range(3):
        q_p = queries.copy(); q_p[:, ic] += h
        q_m = queries.copy(); q_m[:, ic] -= h
        E_p, _ = eval_mesh(mesh, q_p)
        E_m, _ = eval_mesh(mesh, q_m)
        F_fd = -(E_p - E_m) / (2 * h)
        err = np.max(np.abs(F0[:, ic] - F_fd))
        max_err = max(max_err, float(err))
        rv.out(f'  F component {ic}: max|F_analytic - F_fd| = {float(err):.4e}')
    rv.out(f'\nOverall max force parity error: {max_err:.4e}')
    assert max_err < 1e-4, f'Force parity failed: max|F-Fd|={max_err}'
    rv.checklist('F = -∇E verified by finite differences (L0 invariant)',
                 'Tested at 15 random 3D query points in the mesh interior')
    rv.finish()


def test_contact_pme_mesh_vs_direct(make_review):
    """Mesh interpolation must match direct Σ v_i^L on interior queries."""
    from spammm.surfaces.CoarseMesh import build_coarse_mesh, eval_mesh, eval_mesh_direct
    rv = make_review('test_contact_pme_mesh_vs_direct')
    atom_pos = np.array([[0.0, 0.0, 0.0], [2.5, 1.0, 0.0], [-1.5, -1.0, 0.5]])
    R0s = np.array([_C_R0, _O_R0, _C_R0])
    E0s = np.array([_C_E0, _O_E0, _C_E0])
    p = _mesh_split_params_multi(R0s, E0s)
    qb = np.array([[-3.0, 3.0], [-3.0, 3.0], [3.0, 7.0]])
    mesh = build_coarse_mesh(atom_pos, p, qb, h_mesh=1.0, halo_nodes=6)
    rv.out(f'Mesh shape: {mesh.coeffs.shape}, origin: {mesh.origin}')
    rng = np.random.default_rng(42)
    queries = np.column_stack([rng.uniform(-1.5, 1.5, 50), rng.uniform(-1.5, 1.5, 50), rng.uniform(4.0, 6.0, 50)])
    E_m, F_m = eval_mesh(mesh, queries)
    E_d, F_d = eval_mesh_direct(queries, atom_pos, p)
    err_E = float(np.max(np.abs(E_m - E_d)))
    err_F = float(np.max(np.abs(F_m - F_d)))
    rv.out(f'Mesh vs direct: max|dE|={err_E:.4e} max|dF|={err_F:.4e}')
    assert err_E < 1e-2, f'Mesh vs direct E failed: {err_E}'
    assert err_F < 1e-2, f'Mesh vs direct F failed: {err_F}'
    rv.checklist('Mesh interpolation matches direct Σ v_i^L on interior queries',
                 'Multi-atom (3 atoms) tested',
                 'Error within cubic B-spline tolerance for 1Å grid')
    rv.finish()


def test_contact_pme_mesh_lattice_phases(make_review):
    """Test all four lattice phases {0, 0.25, 0.5, 0.75}*h for interpolation accuracy."""
    from spammm.surfaces.CoarseMesh import build_coarse_mesh, eval_mesh, eval_mesh_direct
    rv = make_review('test_contact_pme_mesh_lattice_phases')
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    p = _mesh_split_params(_C_R0, _C_E0)
    qb = np.array([[-2.0, 2.0], [-2.0, 2.0], [3.0, 7.0]])
    mesh = build_coarse_mesh(atom_pos, p, qb, h_mesh=1.0, halo_nodes=6)
    phases = [0.0, 0.25, 0.5, 0.75]
    for ph in phases:
        q = np.array([[0.0 + ph, 0.0 + ph, 4.0 + ph]])
        E_m, F_m = eval_mesh(mesh, q)
        E_d, F_d = eval_mesh_direct(q, atom_pos, p)
        err_E = float(np.abs(E_m[0] - E_d[0]))
        err_F = float(np.max(np.abs(F_m[0] - F_d[0])))
        rv.out(f'Phase {ph:.2f}: |dE|={err_E:.4e} max|dF|={err_F:.4e}')
        assert err_E < 1e-2, f'Phase {ph}: E error {err_E} too large'
        assert err_F < 1e-2, f'Phase {ph}: F error {err_F} too large'
    rv.checklist('All four lattice phases {0, 0.25, 0.5, 0.75}*h tested',
                 'Interpolation accurate at all phases')
    rv.finish()


def test_contact_pme_mesh_halo_convergence(make_review):
    """Halo 6 vs 8 must converge (padding convergence check)."""
    from spammm.surfaces.CoarseMesh import build_coarse_mesh, eval_mesh
    rv = make_review('test_contact_pme_mesh_halo_convergence')
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    p = _mesh_split_params(_C_R0, _C_E0)
    qb = np.array([[-2.0, 2.0], [-2.0, 2.0], [3.0, 7.0]])
    mesh6 = build_coarse_mesh(atom_pos, p, qb, h_mesh=1.0, halo_nodes=6)
    mesh8 = build_coarse_mesh(atom_pos, p, qb, h_mesh=1.0, halo_nodes=8)
    rng = np.random.default_rng(55)
    queries = np.column_stack([rng.uniform(-1.0, 1.0, 20), rng.uniform(-1.0, 1.0, 20), rng.uniform(4.0, 6.0, 20)])
    E6, F6 = eval_mesh(mesh6, queries)
    E8, F8 = eval_mesh(mesh8, queries)
    err_E = float(np.max(np.abs(E6 - E8)))
    err_F = float(np.max(np.abs(F6 - F8)))
    rv.out(f'Halo 6 vs 8: max|dE|={err_E:.4e} max|dF|={err_F:.4e}')
    assert err_E < 1e-6, f'Halo convergence E failed: {err_E}'
    assert err_F < 1e-6, f'Halo convergence F failed: {err_F}'
    rv.checklist('Halo 6 vs 8 converges (boundary padding sufficient)')
    rv.finish()


def test_contact_pme_mesh_stencil_guard(make_review):
    """Queries whose stencil leaves the domain must raise ValueError."""
    from spammm.surfaces.CoarseMesh import build_coarse_mesh, eval_mesh
    rv = make_review('test_contact_pme_mesh_stencil_guard')
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    p = _mesh_split_params(_C_R0, _C_E0)
    qb = np.array([[-2.0, 2.0], [-2.0, 2.0], [3.0, 7.0]])
    mesh = build_coarse_mesh(atom_pos, p, qb, h_mesh=1.0, halo_nodes=6)
    q_bad = np.array([[100.0, 100.0, 100.0]])
    raised = False
    try:
        eval_mesh(mesh, q_bad)
    except ValueError as e:
        rv.out(f'Stencil guard raised: {e}')
        raised = True
    assert raised, 'eval_mesh must raise ValueError on out-of-bounds stencil'
    rv.checklist('Stencil guard rejects queries outside the coefficient domain',
                 'Fail-loud, no silent extrapolation')
    rv.finish()


def test_contact_pme_mesh_ptcda(make_review, xyz):
    """Build mesh for PTCDA; report dimensions and error vs direct."""
    from spammm.surfaces.CoarseMesh import build_coarse_mesh, eval_mesh, eval_mesh_direct
    from spammm.surfaces.PMESplit import SplitParams
    rv = make_review('test_contact_pme_mesh_ptcda')
    afm = _make_afm(xyz('PTCDA.xyz'))
    apos = afm.atoms_arr[:, :3].astype(np.float64)
    cMs = afm.cLJs_arr
    qs = afm.atoms_arr[:, 3].astype(np.float64) if afm.atoms_arr.shape[1] > 3 else np.zeros(len(apos))
    na = len(apos)
    rv.out(f'PTCDA: {na} atoms')
    p = SplitParams(R0=cMs[:, 0].astype(np.float64), E0=cMs[:, 1].astype(np.float64),
                    q=qs, alpha=_TIP_ALPHA, q_tip=0.0, r_damp=_R_DAMP, r_cut=6.0)
    margin = 2.0
    qb = np.array([[apos[:, 0].min() - margin, apos[:, 0].max() + margin],
                   [apos[:, 1].min() - margin, apos[:, 1].max() + margin],
                   [apos[:, 2].max() + 3.0, apos[:, 2].max() + 8.0]])
    mesh = build_coarse_mesh(apos, p, qb, h_mesh=1.0, halo_nodes=6)
    rv.out(f'Mesh shape: {mesh.coeffs.shape}, origin: {mesh.origin}')
    rv.out(f'Resident bytes: {mesh.coeffs.nbytes}')
    rng = np.random.default_rng(42)
    nq = 50
    queries = np.column_stack([rng.uniform(qb[0, 0], qb[0, 1], nq),
                               rng.uniform(qb[1, 0], qb[1, 1], nq),
                               rng.uniform(qb[2, 0] + 1.0, qb[2, 1] - 1.0, nq)])
    E_m, F_m = eval_mesh(mesh, queries)
    E_d, F_d = eval_mesh_direct(queries, apos, p)
    err_E = float(np.max(np.abs(E_m - E_d)))
    err_F = float(np.max(np.abs(F_m - F_d)))
    rel_E = err_E / max(float(np.max(np.abs(E_d))), 1e-12)
    rv.out(f'PTCDA mesh vs direct: max|dE|={err_E:.4e} max|dF|={err_F:.4e} rel_E={rel_E:.4e}')
    assert err_E < 1e-2, f'PTCDA mesh E failed: {err_E}'
    assert err_F < 1e-2, f'PTCDA mesh F failed: {err_F}'
    rv.checklist('PTCDA mesh built and evaluated',
                 'Mesh dimensions and resident bytes reported',
                 'Error vs direct sum within tolerance')
    rv.finish()


def test_contact_pme_mesh_smoothness_l2(visual_output_dir, make_review, xyz):
    """L2: 2D Fz(x,y) slices at h=4.0, 5.0 Å must be smooth — no atomic corrugation."""
    from spammm.surfaces.CoarseMesh import build_coarse_mesh, eval_mesh
    from spammm.surfaces.PMESplit import SplitParams
    rv = make_review('test_contact_pme_mesh_smoothness_l2')
    if visual_output_dir is None:
        rv.out('L2 smoothness: skipped (no --visual/--develop)')
        rv.finish()
        return
    rv.out('L2 smoothness: generating Fz(x,y) slice plots')
    outdir = _WAVE1_MESH_DIR
    os.makedirs(outdir, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    afm = _make_afm(xyz('PTCDA.xyz'))
    apos = afm.atoms_arr[:, :3].astype(np.float64)
    cMs = afm.cLJs_arr
    qs = afm.atoms_arr[:, 3].astype(np.float64) if afm.atoms_arr.shape[1] > 3 else np.zeros(len(apos))
    p = SplitParams(R0=cMs[:, 0].astype(np.float64), E0=cMs[:, 1].astype(np.float64),
                    q=qs, alpha=_TIP_ALPHA, q_tip=0.0, r_damp=_R_DAMP, r_cut=6.0)
    margin = 2.0
    qb = np.array([[apos[:, 0].min() - margin, apos[:, 0].max() + margin],
                   [apos[:, 1].min() - margin, apos[:, 1].max() + margin],
                   [apos[:, 2].max() + 3.0, apos[:, 2].max() + 8.0]])
    mesh = build_coarse_mesh(apos, p, qb, h_mesh=1.0, halo_nodes=6)
    nx = 80; ny = 80
    xs = np.linspace(qb[0, 0] + 0.5, qb[0, 1] - 0.5, nx)
    ys = np.linspace(qb[1, 0] + 0.5, qb[1, 1] - 0.5, ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    zmax = float(apos[:, 2].max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, h_above in zip(axes, [4.0, 5.0]):
        z = zmax + h_above
        queries = np.column_stack([X.ravel(), Y.ravel(), np.full(nx * ny, z)])
        E, F = eval_mesh(mesh, queries)
        Fz = F[:, 2].reshape(nx, ny)
        im = ax.imshow(Fz.T, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin='lower', cmap='RdBu_r', aspect='equal')
        ax.set_title(f'Fz at h={h_above} Å above zmax')
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
        plt.colorbar(im, ax=ax, label='Fz [eV/Å]')
        std_fz = float(np.std(Fz))
        max_fz = float(np.max(np.abs(Fz)))
        rv.out(f'h={h_above}: std(Fz)={std_fz:.4e} max|Fz|={max_fz:.4e} ratio={std_fz / max(max_fz, 1e-12):.4f}')
    fig.suptitle('CoarseMesh: Fz(x,y) smoothness (Wave 1)', fontsize=11)
    plot_path = os.path.join(outdir, 'mesh_fz_smoothness.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    rv.out(f'Saved: {plot_path}')
    print(f'REVIEW: {plot_path}', flush=True)
    rv.checklist('2D Fz(x,y) slices generated at h=4.0 and 5.0 Å',
                 'Slices must be smooth — no atomic corrugation visible')
    rv.finish()


def test_contact_pme_mesh_summary(make_review, xyz):
    """Emit SUMMARY.out with mesh build results for Wave 1."""
    from spammm.surfaces.CoarseMesh import build_coarse_mesh, eval_mesh, eval_mesh_direct
    from spammm.surfaces.PMESplit import SplitParams
    rv = make_review('test_contact_pme_mesh_summary')
    outdir = _WAVE1_MESH_DIR
    os.makedirs(outdir, exist_ok=True)
    afm = _make_afm(xyz('PTCDA.xyz'))
    apos = afm.atoms_arr[:, :3].astype(np.float64)
    cMs = afm.cLJs_arr
    qs = afm.atoms_arr[:, 3].astype(np.float64) if afm.atoms_arr.shape[1] > 3 else np.zeros(len(apos))
    p = SplitParams(R0=cMs[:, 0].astype(np.float64), E0=cMs[:, 1].astype(np.float64),
                    q=qs, alpha=_TIP_ALPHA, q_tip=0.0, r_damp=_R_DAMP, r_cut=6.0)
    margin = 2.0
    qb = np.array([[apos[:, 0].min() - margin, apos[:, 0].max() + margin],
                   [apos[:, 1].min() - margin, apos[:, 1].max() + margin],
                   [apos[:, 2].max() + 3.0, apos[:, 2].max() + 8.0]])
    import time
    t0 = time.time()
    mesh = build_coarse_mesh(apos, p, qb, h_mesh=1.0, halo_nodes=6)
    t_build = time.time() - t0
    rng = np.random.default_rng(42)
    nq = 50
    queries = np.column_stack([rng.uniform(qb[0, 0], qb[0, 1], nq),
                               rng.uniform(qb[1, 0], qb[1, 1], nq),
                               rng.uniform(qb[2, 0] + 1.0, qb[2, 1] - 1.0, nq)])
    E_m, F_m = eval_mesh(mesh, queries)
    E_d, F_d = eval_mesh_direct(queries, apos, p)
    err_E = float(np.max(np.abs(E_m - E_d)))
    err_F = float(np.max(np.abs(F_m - F_d)))
    lines = [
        '# CoarseMesh Wave 1 SUMMARY.out — coarse 3D B-spline mesh',
        f'# Contract version: 2',
        f'#',
        f'# Spacing: h_mesh = 1.0 Å',
        f'# Layout: C-order (nx, ny, nz) with z fastest',
        f'# Interpolant: Nonperiodic cardinal cubic B-spline',
        f'# Prefilter: _bspline_prefilter_1d (reused from ContactSurface.py)',
        f'# Halo: 6 nodes per side',
        f'#',
        f'# PTCDA mesh:',
        f'  shape: {mesh.coeffs.shape}',
        f'  origin: {mesh.origin.tolist()}',
        f'  resident_bytes: {mesh.coeffs.nbytes}',
        f'  build_time_s: {t_build:.3f}',
        f'  max|dE|_vs_direct: {err_E:.4e}',
        f'  max|dF|_vs_direct: {err_F:.4e}',
    ]
    summary_path = os.path.join(outdir, 'SUMMARY.out')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    rv.out(f'Summary written to {summary_path}')
    rv.out('\n'.join(lines))
    print(f'REVIEW: {summary_path}', flush=True)
    rv.checklist('SUMMARY.out emitted with mesh build results',
                 'Mesh dimensions, memory, build time, and error reported')
    rv.finish()
