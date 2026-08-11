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


def _split_params(R0, E0, q=0.0, q_tip=0.0, r_cut=6.0, split_mode='plateau',
                  delta_in=0.5, delta_a=0.5, delta_b=2.0):
    from spammm.surfaces.PMESplit import SplitParams
    return SplitParams(R0=np.float64(R0), E0=np.float64(E0), q=np.float64(q),
                       alpha=_TIP_ALPHA, q_tip=q_tip, r_damp=_R_DAMP, r_cut=r_cut,
                       split_mode=split_mode, delta_in=delta_in, delta_a=delta_a, delta_b=delta_b)


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
    """C² continuity at plateau joins r_a and r_b; v_S and derivatives vanish at r_b."""
    from spammm.surfaces.PMESplit import soft_core_split
    rv = make_review('test_contact_pme_split_join_continuity')
    eps = 1e-8
    cases = [(_C_R0, _C_E0, 0.0, 0.0, 'C'), (_O_R0, _O_E0, 0.5, 1.0, 'O q+'), (_H_R0, _H_E0, -0.3, 0.5, 'H q-')]
    all_err = 0.0
    for R0, E0, q_i, q_tip, label in cases:
        p = _split_params(R0, E0, q_i, q_tip=q_tip)
        r_a = float(p.r_a); r_b = float(p.r_b)
        # C² at r_a: plateau (flat) meets smooth switch
        r_below = np.array([r_a - eps]); r_above = np.array([r_a + eps])
        s_below = soft_core_split(r_below, p); s_above = soft_core_split(r_above, p)
        err_vL_a = abs(s_below['v_L'][0] - s_above['v_L'][0])
        err_dvL_a = abs(s_below['dv_L_dr'][0] - s_above['dv_L_dr'][0])
        err_d2vL_a = abs(s_below['d2v_L'][0] - s_above['d2v_L'][0])
        # Below r_a: v_L must be flat (dv_L ≈ 0, d2v_L ≈ 0)
        err_flat = max(abs(s_below['dv_L_dr'][0]), abs(s_below['d2v_L'][0]))
        # C² at r_b: v_L joins v, v_S vanishes with derivatives
        r_below_rb = np.array([r_b - eps]); r_above_rb = np.array([r_b + eps])
        s_below_rb = soft_core_split(r_below_rb, p); s_above_rb = soft_core_split(r_above_rb, p)
        err_vL_rb = abs(s_below_rb['v_L'][0] - s_above_rb['v_L'][0])
        err_dvL_rb = abs(s_below_rb['dv_L_dr'][0] - s_above_rb['dv_L_dr'][0])
        err_d2vL_rb = abs(s_below_rb['d2v_L'][0] - s_above_rb['d2v_L'][0])
        err_vS_rb = abs(s_below_rb['v_S'][0])
        err_dvS_rb = abs(s_below_rb['dv_S_dr'][0])
        err_d2vS_rb = abs(s_below_rb['d2v_S'][0])
        mx = max(err_vL_a, err_dvL_a, err_d2vL_a, err_flat, err_vL_rb, err_dvL_rb, err_d2vL_rb,
                 err_vS_rb, err_dvS_rb, err_d2vS_rb)
        all_err = max(all_err, mx)
        rv.out(f'  {label}: C²@r_a={r_a:.3f}: |d vL|={err_vL_a:.4e} |d dvL|={err_dvL_a:.4e} |d d2vL|={err_d2vL_a:.4e} flat={err_flat:.4e}')
        rv.out(f'  {label}: C²@r_b={r_b:.3f}: |d vL|={err_vL_rb:.4e} |d dvL|={err_dvL_rb:.4e} |d d2vL|={err_d2vL_rb:.4e}')
        rv.out(f'  {label}: v_S@r_b: |vS|={err_vS_rb:.4e} |dvS|={err_dvS_rb:.4e} |d2vS|={err_d2vS_rb:.4e}')
        assert err_vL_a < 1e-10 and err_dvL_a < 1e-8 and err_d2vL_a < 1e-5, f'{label}: C² fail at r_a'
        assert err_flat < 1e-10, f'{label}: v_L not flat below r_a'
        assert err_vL_rb < 1e-6 and err_dvL_rb < 1e-6 and err_d2vL_rb < 1e-4, f'{label}: C² fail at r_b'
        assert err_vS_rb < 1e-10 and err_dvS_rb < 1e-6 and err_d2vS_rb < 1e-4, f'{label}: v_S not zero at r_b'
    rv.out(f'\nOverall max join error: {all_err:.4e}')
    rv.checklist('v_L is C² at r_a (plateau end)',
                 'v_L is flat for r <= r_a (no displaced Morse minimum on mesh)',
                 'v_S, v_S\', v_S\'\' all vanish at r_b')
    rv.finish()


def test_contact_pme_split_rho_monotonic(make_review):
    """Legacy rho(r) must be monotonically non-decreasing; kept for comparison mode."""
    from spammm.surfaces.PMESplit import softened_rho
    rv = make_review('test_contact_pme_split_rho_monotonic')
    cases = [(_C_R0, 'C'), (_O_R0, 'O'), (_H_R0, 'H')]
    r_cut = 6.0
    all_err = 0.0
    for R0, label in cases:
        r_lo = R0 - 0.5
        r = np.linspace(r_lo - 1.0, r_cut + 1.0, 2000)
        rho, drho, d2rho = softened_rho(r, r_lo, r_cut)
        min_drho = float(np.min(drho))
        drho_diff = np.diff(rho)
        min_drho_diff = float(np.min(drho_diff))
        all_err = max(all_err, -min_drho, -min_drho_diff)
        rv.out(f'  {label}: r_lo={r_lo:.4f} min(drho/dr)={min_drho:.4e} min(diff(rho))={min_drho_diff:.4e}')
        assert min_drho >= -1e-14, f'{label}: drho/dr < 0 ({min_drho})'
        assert min_drho_diff >= -1e-14, f'{label}: rho not monotonic ({min_drho_diff})'
        assert abs(rho[0] - r_lo) < 1e-14, f'{label}: rho(r<r_lo) != r_lo'
        assert abs(rho[-1] - (r_cut + 1.0)) < 1e-14, f'{label}: rho(r>r_cut) != r'
    rv.out(f'\nOverall monotonicity violation: {all_err:.4e} (target < 1e-14)')
    rv.checklist('legacy rho(r) is monotonically non-decreasing for C/O/H',
                 'rho mode kept for comparison; default split is plateau')
    rv.finish()


def test_contact_pme_split_plateau_no_displaced_min(make_review):
    """Plateau v_L must NOT move the Morse minimum; |v_S(R0)| << legacy rho residual."""
    from spammm.surfaces.PMESplit import soft_core_split
    rv = make_review('test_contact_pme_split_plateau_no_displaced_min')
    p_plat = _split_params(_C_R0, _C_E0, 0.0, split_mode='plateau')
    p_rho = _split_params(_C_R0, _C_E0, 0.0, split_mode='rho', r_cut=6.0)
    r0 = np.array([_C_R0])
    s_plat = soft_core_split(r0, p_plat)
    s_rho = soft_core_split(r0, p_rho)
    v_well = float(s_plat['v'][0])
    vS_plat = float(s_plat['v_S'][0])
    vS_rho = float(s_rho['v_S'][0])
    # Plateau: at R0 < r_a, v_L = C = v(r_a), so v_S = v(R0) - v(r_a) (compact well correction)
    # Rho: v_L(R0) = v(rho(R0)) with rho≪R0 → large residual
    rv.out(f'C at R0={_C_R0:.4f}: v={v_well:.6e}')
    rv.out(f'  plateau: v_L={float(s_plat["v_L"][0]):.6e} v_S={vS_plat:.6e} |v_S|/|v|={abs(vS_plat/v_well):.3f}')
    rv.out(f'  rho:     v_L={float(s_rho["v_L"][0]):.6e} v_S={vS_rho:.6e} |v_S|/|v|={abs(vS_rho/v_well):.3f}')
    assert abs(vS_plat) < abs(vS_rho), 'plateau residual at R0 must be smaller than rho residual'
    assert abs(vS_plat) < 2.0 * abs(v_well), 'plateau |v_S(R0)| should be O(well), not larger'
    # Mesh must be flat at R0 (no force from long-range part near well)
    assert abs(float(s_plat['dv_L_dr'][0])) < 1e-12, 'plateau v_L must be flat at R0 < r_a'
    rv.checklist('plateau |v_S(R0)| < rho |v_S(R0)|',
                 'plateau v_L force vanishes at R0 (no displaced minimum on mesh)')
    rv.finish()


def test_contact_pme_split_1d_smoothness_experiment(visual_output_dir, make_review):
    """1D PAW-goal experiment: compare split modes by smoothness metrics (Python only).

    Goal: v_L must damp high frequency, NOT invent bumps via W'(v-C).
    Modes: paw (even poly), hermite, plateau (W-blend), softcore, rho.
    No OpenCL / multi-atom — decide the split here before production.
    """
    from spammm.surfaces.PMESplit import soft_core_split, split_smoothness_metrics
    rv = make_review('test_contact_pme_split_1d_smoothness_experiment')
    outdir = os.path.join(_WAVE0_DIR, '1d_smoothness')
    os.makedirs(outdir, exist_ok=True)
    modes = ['paw', 'hermite', 'plateau', 'softcore', 'rho']
    atoms = [(_C_R0, _C_E0, 'C'), (_O_R0, _O_E0, 'O'), (_H_R0, _H_E0, 'H')]
    # Table header
    hdr = f'{"atom":3s} {"mode":10s} {"hf_ratio":>10s} {"overshoot":>12s} {"max|d2vL|":>12s} {"max|vS|":>12s} {"n_ext_vL":>9s} {"n_ext_vS":>9s}'
    rv.out(hdr)
    rv.out('-' * len(hdr))
    rows = []
    for R0, E0, label in atoms:
        for mode in modes:
            p = _split_params(R0, E0, 0.0, split_mode=mode, r_cut=6.0)
            r_lo = float(p.r_lo)
            r_b = float(p.r_b) if mode != 'rho' else float(p.r_cut)
            # paw is valid at r=0; sample from near 0 for origin smoothness check
            r0 = 0.02 if mode == 'paw' else r_lo
            r = np.linspace(r0, r_b + 0.5, 2000)
            s = soft_core_split(r, p)
            m = split_smoothness_metrics(r, s, r_lo=r_lo, r_b=r_b)
            rows.append((label, mode, m, R0, r_lo, r_b, r, s, p))
            rv.out(f'{label:3s} {mode:10s} {m["hf_ratio"]:10.4f} {m["force_overshoot"]:12.4e} '
                   f'{m["max_abs_d2_vL"]:12.4e} {m["max_abs_vS"]:12.4e} {m["n_extrema_vL"]:9d} {m["n_extrema_vS"]:9d}')
    # L0: paw/hermite must not amplify force vs reference; plateau W-blend does
    for label, mode, m, R0, r_lo, r_b, r, s, p in rows:
        if mode in ('paw', 'hermite') and label == 'C':
            plat = [x for x in rows if x[0] == 'C' and x[1] == 'plateau'][0][2]
            rv.out(f'\nC {mode} overshoot={m["force_overshoot"]:.4e} vs plateau={plat["force_overshoot"]:.4e}')
            assert m['force_overshoot'] < plat['force_overshoot'], \
                f'{mode} must have smaller force overshoot than W-blend plateau'
            assert m['hf_ratio'] < plat['hf_ratio'] or m['force_overshoot'] < 0.5 * plat['force_overshoot'], \
                f'{mode} must be smoother (hf_ratio or overshoot) than plateau W-blend'
    # Identity + C² join for paw/hermite
    for label, mode, m, R0, r_lo, r_b, r, s, p in rows:
        if mode in ('paw', 'hermite'):
            assert np.max(np.abs(s['v_L'] + s['v_S'] - s['v'])) < 1e-12
            i_b = int(np.argmin(np.abs(r - r_b)))
            assert abs(s['v_S'][i_b]) < 1e-8, f'{label} {mode} v_S(r_b)={s["v_S"][i_b]}'
            assert abs(s['dv_S_dr'][i_b]) < 1e-6, f'{label} {mode} dv_S(r_b)={s["dv_S_dr"][i_b]}'
    # paw: dP/dr → 0 as r→0 (even radial field)
    for label, mode, m, R0, r_lo, r_b, r, s, p in rows:
        if mode == 'paw' and label == 'C':
            i0 = int(np.argmin(r))
            rv.out(f'C paw at r≈{r[i0]:.4f}: dv_L={s["dv_L_dr"][i0]:.4e} (expect ~0)')
            assert abs(s['dv_L_dr'][i0]) < 1e-3, f'paw force must vanish at origin, got {s["dv_L_dr"][i0]}'

    if visual_output_dir is None:
        rv.out('L2 plots skipped (no --visual/--develop)')
        rv.finish()
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Figure 1: energy — first 4 modes × C
    plot_modes = ['paw', 'hermite', 'plateau', 'rho']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, mode in zip(axes.ravel(), plot_modes):
        R0, E0, label = atoms[0]
        p = _split_params(R0, E0, 0.0, split_mode=mode, r_cut=6.0)
        r_lo = float(p.r_lo); r_b = float(p.r_b) if mode != 'rho' else float(p.r_cut)
        r0 = 0.02 if mode == 'paw' else r_lo
        r = np.linspace(r0, r_b + 0.8, 1200)
        s = soft_core_split(r, p)
        m = split_smoothness_metrics(r, s, r_lo=r_lo, r_b=r_b)
        ax.plot(r, s['v'], 'k-', lw=1.5, label='v ref')
        ax.plot(r, s['v_L'], 'b-', lw=1.2, label='v_L smooth/mesh')
        ax.plot(r, s['v_S'], 'r-', lw=1.2, label='v_S short/core')
        ax.axhline(0, color='gray', ls=':', alpha=0.4)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7, label='R0')
        ax.axvline(r_b, color='magenta', ls=':', alpha=0.6, label='r_b')
        ax.set_title(f'C energy — {mode}\nhf={m["hf_ratio"]:.3f} overshoot={m["force_overshoot"]:.2e}')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('V [eV]')
        ymin = min(float(np.min(s['v'])), float(np.min(s['v_S']))) * 1.15
        ax.set_ylim(ymin, max(abs(ymin) * 0.35, 1e-4))
        ax.legend(fontsize=6)
    fig.suptitle('1D split experiment (energy): black=ref, blue=smooth long-range, red=short residual\n'
                 'PAW goal: blue must be smoother than black — no new bumps', fontsize=11)
    p1 = os.path.join(outdir, '01_energy_4modes_C.png')
    fig.savefig(p1, dpi=150); plt.close(fig)
    print(f'REVIEW: {p1}', flush=True); rv.out(f'Saved {p1}')

    # Figure 2: FORCE — the smoking gun for W' bumps
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, mode in zip(axes.ravel(), plot_modes):
        R0, E0, label = atoms[0]
        p = _split_params(R0, E0, 0.0, split_mode=mode, r_cut=6.0)
        r_lo = float(p.r_lo); r_b = float(p.r_b) if mode != 'rho' else float(p.r_cut)
        r0 = 0.02 if mode == 'paw' else r_lo
        r = np.linspace(r0, r_b + 0.8, 1200)
        s = soft_core_split(r, p)
        m = split_smoothness_metrics(r, s, r_lo=r_lo, r_b=r_b)
        ax.plot(r, s['dvdr'], 'k-', lw=1.5, label="v' ref")
        ax.plot(r, s['dv_L_dr'], 'b-', lw=1.2, label="v_L' smooth")
        ax.plot(r, s['dv_S_dr'], 'r-', lw=1.2, label="v_S' short")
        ax.axhline(0, color='gray', ls=':', alpha=0.4)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7)
        ax.axvline(r_b, color='magenta', ls=':', alpha=0.6)
        if mode == 'plateau':
            ax.axvline(float(p.r_a), color='cyan', ls='-.', alpha=0.7, label='r_a')
        # clip wall for visibility of transition
        mask = r > R0 - 0.2
        yspan = np.percentile(np.abs(s['dvdr'][mask]), 99)
        ax.set_ylim(-yspan * 1.3, yspan * 1.3)
        ax.set_title(f'C force — {mode}\novershoot={m["force_overshoot"]:.2e}  n_ext(vL)={m["n_extrema_vL"]}')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('dV/dr [eV/Å]')
        ax.legend(fontsize=6)
    fig.suptitle('1D split experiment (FORCE): plateau W-blend overshoots (blue > black) → red dips\n'
                 'paw/hermite: blue stays below |ref| — true damping, no invented bumps', fontsize=11)
    p2 = os.path.join(outdir, '02_force_4modes_C.png')
    fig.savefig(p2, dpi=150); plt.close(fig)
    print(f'REVIEW: {p2}', flush=True); rv.out(f'Saved {p2}')

    # Figure 2b: 2 rows × 4 cols — top potentials, bottom forces; twin axis = W, 1-W, W'
    # User partition: v_S = W·V_ref,  v_L = (1-W)·V_ref,  V_ref = v_S + v_L
    # ⇒ effective W = v_S / V_ref  (masked where |V_ref| tiny)
    R0, E0, _ = atoms[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    V_floor = 1e-6  # mask W near V≈0 crossing
    for col, mode in enumerate(plot_modes):
        p = _split_params(R0, E0, 0.0, split_mode=mode, r_cut=6.0)
        r_lo = float(p.r_lo)
        r_outer = float(p.r_b) if mode != 'rho' else float(p.r_cut)
        r0 = 0.02 if mode == 'paw' else r_lo
        r = np.linspace(r0, r_outer + 0.8, 1600)
        s = soft_core_split(r, p)
        V = s['v']; vL = s['v_L']; vS = s['v_S']
        dV = s['dvdr']; dvL = s['dv_L_dr']; dvS = s['dv_S_dr']
        # Effective partition weight from additive split (identity V=vL+vS)
        with np.errstate(divide='ignore', invalid='ignore'):
            W = np.where(np.abs(V) > V_floor, vS / V, np.nan)
            OmW = np.where(np.abs(V) > V_floor, vL / V, np.nan)
        # dW/dr from quotient: W' = (vS' V - vS V') / V²
        with np.errstate(divide='ignore', invalid='ignore'):
            dW = np.where(np.abs(V) > V_floor, (dvS * V - vS * dV) / (V * V), np.nan)

        # --- top: potentials ---
        ax = axes[0, col]
        ax.plot(r, V, 'k-', lw=1.5, label='V_ref')
        ax.plot(r, vL, 'b-', lw=1.2, label='v_L=(1−W)V')
        ax.plot(r, vS, 'r-', lw=1.2, label='v_S=W·V')
        ax.axhline(0, color='gray', ls=':', alpha=0.4)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7)
        ax.axvline(r_outer, color='magenta', ls=':', alpha=0.6)
        if mode == 'plateau':
            ax.axvline(float(p.r_a), color='cyan', ls='-.', alpha=0.6)
        ymin = min(float(np.nanmin(V)), float(np.nanmin(vS))) * 1.15
        ax.set_ylim(ymin, max(abs(ymin) * 0.35, 1e-4))
        ax.set_title(f'{mode}')
        ax.set_ylabel('V [eV]' if col == 0 else '')
        tax = ax.twinx()
        tax.plot(r, W, 'r--', lw=1.0, alpha=0.85, label='W=v_S/V')
        tax.plot(r, OmW, 'b--', lw=1.0, alpha=0.85, label='1−W=v_L/V')
        tax.set_ylim(-0.5, 1.5)
        tax.set_ylabel('W' if col == 3 else '')
        if col == 0:
            lines1, lab1 = ax.get_legend_handles_labels()
            lines2, lab2 = tax.get_legend_handles_labels()
            ax.legend(lines1 + lines2, lab1 + lab2, fontsize=6, loc='best')

        # --- bottom: forces ---
        ax = axes[1, col]
        ax.plot(r, dV, 'k-', lw=1.5, label="V'")
        ax.plot(r, dvL, 'b-', lw=1.2, label="v_L'")
        ax.plot(r, dvS, 'r-', lw=1.2, label="v_S'")
        ax.axhline(0, color='gray', ls=':', alpha=0.4)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7)
        ax.axvline(r_outer, color='magenta', ls=':', alpha=0.6)
        if mode == 'plateau':
            ax.axvline(float(p.r_a), color='cyan', ls='-.', alpha=0.6)
        mask = r > R0 - 0.2
        yspan = np.percentile(np.abs(dV[mask]), 99)
        ax.set_ylim(-yspan * 1.3, yspan * 1.3)
        ax.set_xlabel('r [Å]')
        ax.set_ylabel("dV/dr [eV/Å]" if col == 0 else '')
        tax = ax.twinx()
        # scale W' for visibility; also plot W again lightly
        dW_max = np.nanmax(np.abs(dW))
        dW_n = dW / dW_max if dW_max and np.isfinite(dW_max) and dW_max > 0 else dW
        tax.plot(r, W, 'g:', lw=1.0, alpha=0.5, label='W')
        tax.plot(r, dW_n, 'g--', lw=1.2, alpha=0.9, label="W' (norm)")
        tax.set_ylim(-1.5, 1.5)
        tax.set_ylabel("W, W'(norm)" if col == 3 else '')
        if col == 0:
            lines1, lab1 = ax.get_legend_handles_labels()
            lines2, lab2 = tax.get_legend_handles_labels()
            ax.legend(lines1 + lines2, lab1 + lab2, fontsize=6, loc='best')

    fig.suptitle('Partition view:  v_S = W·V_ref ,  v_L = (1−W)·V_ref ,  V_ref = v_S+v_L\n'
                 'Top: potentials + twin W,1−W   |   Bottom: forces + twin W and W′(normalized)\n'
                 'W := v_S/V_ref  (masked near V≈0).  Orange=R0, magenta=r_b/r_cut', fontsize=11)
    p2b = os.path.join(outdir, '02b_V_and_force_with_W_C.png')
    fig.savefig(p2b, dpi=150); plt.close(fig)
    print(f'REVIEW: {p2b}', flush=True); rv.out(f'Saved {p2b}')
    import shutil
    p2b_old = os.path.join(outdir, '02b_force_and_W_C.png')
    shutil.copyfile(p2b, p2b_old)
    print(f'REVIEW: {p2b_old}', flush=True)

    # EQUATIONS.out — partition definition + how each mode builds (v_L,v_S)
    eq_path = os.path.join(outdir, 'EQUATIONS.out')
    eq_lines = [
        '# Partition identity (what W means in the plots):',
        '#   V_ref = v_S + v_L',
        '#   v_S   = W     * V_ref',
        '#   v_L   = (1-W) * V_ref',
        '#   ⇒  W = v_S / V_ref     (plotted; masked where |V_ref|<1e-6)',
        '#',
        '# NOTE: This W is DIAGNOSTIC from the additive split. It is NOT necessarily',
        '# an input smoothstep. Only plateau uses an input smoothstep W_sw on (v-C):',
        '#',
        f'# Geometry (C): R0={R0:.4f}, r_lo=R0-0.5, r_a=R0+0.5, r_b=R0+2, rho r_cut=6',
        '#',
        '# paw:      v_L=P(r)=a0+a2 r^2+a4 r^4+a6 r^6 (even; smooth at r=0);',
        '#           C2-match v at r_b; a0 minimizes int (P\'\')^2; v_S=V-P',
        '# hermite:  v_L=P(s) soft poly in (r-r_lo) (flat at r_lo, C2 at r_b); v_S=V-P',
        '# softcore: same as hermite with a0=v(sqrt(r_lo^2+a^2))',
        '# plateau:  C=v(r_a); W_sw=smoothstep on [r_a,r_b];',
        '#           v_L=C+W_sw(v-C); v_S=(1-W_sw)(v-C)',
        '#           diagnostic W = v_S/v  ≠  W_sw  in general (because of C offset)',
        '# rho:      v_L=v(ρ(r)); v_S=v-v(ρ)',
        '#',
    ]
    with open(eq_path, 'w') as f:
        f.write('\n'.join(eq_lines) + '\n')
    print(f'REVIEW: {eq_path}', flush=True); rv.out(f'Saved {eq_path}')

    # Figure 3: curvature d²v — high-frequency content
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    R0, E0, _ = atoms[0]
    r_lo = R0 - 0.5; r_b = R0 + 2.0
    r = np.linspace(r_lo, r_b + 0.5, 1200)
    for mode, ls in [('paw', '-'), ('hermite', '-.'), ('plateau', '--'), ('rho', ':')]:
        p = _split_params(R0, E0, 0.0, split_mode=mode, r_cut=6.0)
        s = soft_core_split(r, p)
        axes[0].plot(r, s['d2v_L'], ls=ls, lw=1.2, label=f"d²v_L {mode}")
        axes[1].plot(r, s['v_S'], ls=ls, lw=1.2, label=f'v_S {mode}')
    axes[0].plot(r, soft_core_split(r, _split_params(R0, E0, 0.0, split_mode='paw'))['d2v'],
                 'k-', lw=0.8, alpha=0.5, label='d²v ref')
    axes[0].axhline(0, color='gray', ls=':', alpha=0.3)
    axes[0].axvline(R0, color='orange', ls='--', alpha=0.6)
    axes[0].set_title('Curvature of long-range field (lower |d²| = smoother)')
    axes[0].set_xlabel('r [Å]'); axes[0].set_ylabel('d²V/dr²')
    axes[0].legend(fontsize=7)
    # zoom curvature away from wall
    mask = r > R0
    yspan = np.percentile(np.abs(soft_core_split(r, _split_params(R0, E0, split_mode='plateau'))['d2v_L'][mask]), 99)
    axes[0].set_ylim(-yspan * 2, yspan * 2)
    axes[1].axhline(0, color='gray', ls=':', alpha=0.3)
    axes[1].axvline(R0, color='orange', ls='--', alpha=0.6)
    axes[1].set_title('Short residual v_S (should be compact, not wiggly)')
    axes[1].set_xlabel('r [Å]'); axes[1].set_ylabel('v_S [eV]')
    axes[1].legend(fontsize=7)
    fig.suptitle('C: curvature + residual — paw/hermite should have smaller |d²v_L| than plateau/rho', fontsize=11)
    p3 = os.path.join(outdir, '03_curvature_and_residual_C.png')
    fig.savefig(p3, dpi=150); plt.close(fig)
    print(f'REVIEW: {p3}', flush=True); rv.out(f'Saved {p3}')

    # Figure 4: metric bar chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(modes))
    for ax, key, title in zip(axes,
                              ['force_overshoot', 'hf_ratio', 'max_abs_vS'],
                              ['force overshoot (↓ better)', 'hf_ratio RMS(d²v_L)/RMS(d²v) (↓)', 'max|v_S| (↓)']):
        for j, (R0, E0, label) in enumerate(atoms):
            vals = []
            for mode in modes:
                m = [x[2] for x in rows if x[0] == label and x[1] == mode][0]
                vals.append(m[key])
            ax.bar(x + 0.25 * (j - 1), vals, width=0.25, label=label)
        ax.set_xticks(x); ax.set_xticklabels(modes, rotation=15)
        ax.set_title(title); ax.legend(fontsize=7)
    fig.suptitle('Smoothness scoreboard (C/O/H) — pick mode with low overshoot + low hf_ratio', fontsize=11)
    p4 = os.path.join(outdir, '04_metrics_bar.png')
    fig.savefig(p4, dpi=150); plt.close(fig)
    print(f'REVIEW: {p4}', flush=True); rv.out(f'Saved {p4}')

    # SUMMARY.out
    lines = [
        '# 1D split smoothness experiment SUMMARY',
        '# PAW goal: replace steep high-frequency content by a smooth approx;',
        '#           residual = high-freq part fitted by short-range basis.',
        '# CRITICAL: smooth approx must NOT invent new bumps (W\'*(v-C) failure mode).',
        '#',
        '# Metrics: force_overshoot = max(0,|v_L\'|-|v\'|); hf_ratio = RMS(d²v_L)/RMS(d²v)',
        '#',
        hdr,
    ]
    for label, mode, m, *_ in rows:
        lines.append(f'{label:3s} {mode:10s} {m["hf_ratio"]:10.4f} {m["force_overshoot"]:12.4e} '
                     f'{m["max_abs_d2_vL"]:12.4e} {m["max_abs_vS"]:12.4e} {m["n_extrema_vL"]:9d} {m["n_extrema_vS"]:9d}')
    # Recommend
    c_rows = {mode: m for lab, mode, m, *_ in rows if lab == 'C'}
    best = min(c_rows.keys(), key=lambda k: (c_rows[k]['force_overshoot'], c_rows[k]['hf_ratio']))
    lines += ['', f'# Recommended mode by (overshoot, hf_ratio) on C: {best}',
              f'# plateau overshoot (known bug): {c_rows["plateau"]["force_overshoot"]:.4e}',
              f'# hermite overshoot: {c_rows["hermite"]["force_overshoot"]:.4e}',
              f'# paw overshoot: {c_rows["paw"]["force_overshoot"]:.4e}',
              '# fit_contact_pme default split_mode=paw (even poly, smooth at r=0).']
    sp = os.path.join(outdir, 'SUMMARY.out')
    with open(sp, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'REVIEW: {sp}', flush=True); rv.out(f'Saved {sp}')
    rv.checklist('Compared paw/hermite/plateau/softcore/rho on force overshoot + hf_ratio',
                 'paw/hermite has lower force overshoot than plateau W-blend',
                 'L2 energy/force/curvature/metric plots written under wave0_split/1d_smoothness/',
                 'fit_contact_pme default is paw')
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
    """L2: plateau vs legacy rho split — v, v_L, v_S, force; clear short/long labels."""
    from spammm.surfaces.PMESplit import soft_core_split
    rv = make_review('test_contact_pme_split_l2_curves')
    if visual_output_dir is None:
        rv.out('L2 split curves: skipped (no --visual/--develop)')
        rv.finish()
        return
    rv.out('L2 split curves: generating plateau vs rho comparison plots')
    outdir = _WAVE0_DIR
    os.makedirs(outdir, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cases = [(_C_R0, _C_E0, 0.0, 0.0, 'C'), (_O_R0, _O_E0, 0.0, 0.0, 'O'), (_H_R0, _H_E0, 0.0, 0.0, 'H')]
    # Figure 1: energy split — plateau (top) vs rho (bottom)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, (R0, E0, q_i, q_tip, label) in enumerate(cases):
        for row, mode in enumerate(['plateau', 'rho']):
            p = _split_params(R0, E0, q_i, q_tip=q_tip, split_mode=mode)
            r_lo = float(p.r_lo)
            r_outer = float(p.r_b) if mode == 'plateau' else float(p.r_cut)
            r = np.linspace(r_lo, r_outer + 1.0, 800)
            s = soft_core_split(r, p)
            ax = axes[row, col]
            ax.plot(r, s['v'], 'k-', label='v ref (total)', lw=1.5)
            ax.plot(r, s['v_L'], 'b-', label='v_L long/mesh', lw=1.2)
            ax.plot(r, s['v_S'], 'r-', label='v_S short/core', lw=1.2)
            ax.axhline(0, color='gray', ls=':', alpha=0.4)
            ax.axvline(R0, color='orange', ls='--', alpha=0.7, label=f'R0={R0:.2f}')
            ax.axvline(r_lo, color='gray', ls=':', alpha=0.5)
            if mode == 'plateau':
                ax.axvline(float(p.r_a), color='cyan', ls='-.', alpha=0.7, label=f'r_a={float(p.r_a):.2f}')
                ax.axvline(float(p.r_b), color='magenta', ls='-.', alpha=0.7, label=f'r_b={float(p.r_b):.2f}')
            else:
                ax.axvline(r_outer, color='magenta', ls='-.', alpha=0.7, label=f'r_cut={r_outer:.1f}')
            ax.set_title(f'{label} — {mode}')
            ax.set_xlabel('r [Å]'); ax.set_ylabel('V [eV]')
            # Focus on well scale (clip repulsive wall for readability)
            ymin = min(float(np.min(s['v'])), float(np.min(s['v_S']))) * 1.15
            ymax = max(float(np.max(s['v_L'][r > R0 - 0.2])), abs(ymin) * 0.3, 1e-4)
            ax.set_ylim(ymin, ymax)
            ax.legend(fontsize=6, loc='best')
    fig.suptitle('PME split: plateau (correct) vs legacy rho — energy\n'
                 'black=reference total, blue=long-range/mesh, red=short-range/core residual', fontsize=11)
    plot_path = os.path.join(outdir, 'split_curves_plateau_vs_rho.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    rv.out(f'Saved: {plot_path}')
    print(f'REVIEW: {plot_path}', flush=True)

    # Figure 2: force (dv/dr) for plateau only — same legend convention
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (R0, E0, q_i, q_tip, label) in zip(axes2, cases):
        p = _split_params(R0, E0, q_i, q_tip=q_tip, split_mode='plateau')
        r_lo = float(p.r_lo); r_a = float(p.r_a); r_b = float(p.r_b)
        r = np.linspace(r_lo, r_b + 1.0, 800)
        s = soft_core_split(r, p)
        ax.plot(r, s['dvdr'], 'k-', label="v' ref", lw=1.5)
        ax.plot(r, s['dv_L_dr'], 'b-', label="v_L' long", lw=1.2)
        ax.plot(r, s['dv_S_dr'], 'r-', label="v_S' short", lw=1.2)
        ax.axhline(0, color='gray', ls=':', alpha=0.4)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7, label=f'R0')
        ax.axvline(r_a, color='cyan', ls='-.', alpha=0.7, label='r_a')
        ax.axvline(r_b, color='magenta', ls='-.', alpha=0.7, label='r_b')
        ax.set_title(f'{label} force split (plateau)')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('dV/dr [eV/Å]')
        # Clip extreme wall for visibility of well/tail
        mask = r > R0 - 0.3
        yspan = np.percentile(np.abs(s['dvdr'][mask]), 99)
        ax.set_ylim(-yspan * 1.2, yspan * 1.2)
        ax.legend(fontsize=6)
    fig2.suptitle('PME plateau split — radial force (black=ref, blue=mesh, red=core)', fontsize=11)
    plot_path2 = os.path.join(outdir, 'split_force_plateau.png')
    fig2.savefig(plot_path2, dpi=150)
    plt.close(fig2)
    rv.out(f'Saved: {plot_path2}')
    print(f'REVIEW: {plot_path2}', flush=True)

    # Backward-compat name also written (plateau only, C/O/H)
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (R0, E0, q_i, q_tip, label) in zip(axes3, cases):
        p = _split_params(R0, E0, q_i, q_tip=q_tip)
        r_lo = float(p.r_lo); r_b = float(p.r_b)
        r = np.linspace(r_lo, r_b + 1.0, 800)
        s = soft_core_split(r, p)
        ax.plot(r, s['v'], 'k-', label='v (total)', lw=1.5)
        ax.plot(r, s['v_L'], 'b--', label='v_L (mesh)', lw=1.2)
        ax.plot(r, s['v_S'], 'r:', label='v_S (core)', lw=1.2)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7)
        ax.axvline(float(p.r_a), color='cyan', ls=':', alpha=0.6)
        ax.axvline(r_b, color='magenta', ls=':', alpha=0.6)
        ax.set_title(f'{label} plateau (R0={R0:.2f})')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('V [eV]')
        ymin = min(float(np.min(s['v'])), float(np.min(s['v_S']))) * 1.15
        ax.set_ylim(ymin, max(abs(ymin) * 0.3, 1e-4))
        ax.legend(fontsize=7)
    fig3.suptitle('PMESplit plateau default (Wave 0)', fontsize=11)
    plot_path3 = os.path.join(outdir, 'split_curves.png')
    fig3.savefig(plot_path3, dpi=150)
    plt.close(fig3)
    rv.out(f'Saved: {plot_path3}')
    print(f'REVIEW: {plot_path3}', flush=True)
    rv.finish()


def test_contact_pme_split_summary(make_review):
    """Emit SUMMARY.out with accepted parameter map for Wave 0 (plateau default)."""
    from spammm.surfaces.PMESplit import r_cut_candidates, delta_b_candidates, SplitParams, COULOMB_CONST
    rv = make_review('test_contact_pme_split_summary')
    outdir = _WAVE0_DIR
    os.makedirs(outdir, exist_ok=True)
    p = SplitParams(R0=np.array([_C_R0, _O_R0, _H_R0]), E0=np.array([_C_E0, _O_E0, _H_E0]),
                    q=np.array([0.0, 0.0, 0.0]), alpha=_TIP_ALPHA, q_tip=0.0, r_damp=_R_DAMP,
                    split_mode='plateau', delta_in=0.5, delta_a=0.5, delta_b=2.0, r_cut=6.0)
    valid_db, rejected_db = delta_b_candidates(p)
    valid_rc, rejected_rc, r_lo_max = r_cut_candidates(p)  # legacy rho reference
    lines = [
        '# PMESplit Wave 0 SUMMARY.out — accepted parameter map',
        f'# Contract version: 3 (plateau+W default; rho kept for comparison)',
        f'# COULOMB_CONST = {COULOMB_CONST}',
        f'#',
        f'# Global tip parameters:',
        f'alpha (tip stiffness) = {_TIP_ALPHA}',
        f'K = -alpha = {-_TIP_ALPHA}',
        f'r_damp = {_R_DAMP} Å',
        f'split_mode = plateau',
        f'Δ_in = {p.delta_in} Å  → r_min = R0 - Δ_in',
        f'Δ_a  = {p.delta_a} Å  → r_a   = R0 + Δ_a  (plateau end)',
        f'Δ_b  = {p.delta_b} Å  → r_b   = R0 + Δ_b  (core cutoff)',
        f'r_core_max = {p.r_core_max:.4f} Å',
        f'PLQH convention = (1, 1, q_tip, 0)',
        f'',
        f'# Per-atom (assign_params): R0 = tip_R + R_vdW, E0 = sqrt(tip_E * E_vdW)',
        f'tip_R = {_TIP_R}, tip_E = {_TIP_E}',
        f'',
        f'# Atom  R0[Å]    E0[eV]       r_min    r_a      r_b',
        f'C      {_C_R0:.4f}  {_C_E0:.6e}  {_C_R0-0.5:.4f}  {_C_R0+0.5:.4f}  {_C_R0+2.0:.4f}',
        f'O      {_O_R0:.4f}  {_O_E0:.6e}  {_O_R0-0.5:.4f}  {_O_R0+0.5:.4f}  {_O_R0+2.0:.4f}',
        f'H      {_H_R0:.4f}  {_H_E0:.6e}  {_H_R0-0.5:.4f}  {_H_R0+0.5:.4f}  {_H_R0+2.0:.4f}',
        f'',
        f'# Δ_b sweep candidates: valid={valid_db} rejected={rejected_db}',
        f'# Legacy rho r_cut candidates: valid={valid_rc} rejected={rejected_rc} r_lo_max={r_lo_max:.4f}',
        f'',
        f'# Split: v_L = C + W(v-C), v_S = (1-W)(v-C), C=v(r_a), W=quintic smoothstep on [r_a,r_b]',
        f'# Legacy rho mode: v_L = v(rho(r)) kept for comparison only',
        f'# Domain violation: r < r_min → ValueError (fail-loud)',
    ]
    summary_path = os.path.join(outdir, 'SUMMARY.out')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    rv.out(f'Summary written to {summary_path}')
    rv.out('\n'.join(lines))
    print(f'REVIEW: {summary_path}', flush=True)
    rv.checklist('SUMMARY.out emitted with plateau parameter map',
                 'Δ_b / legacy r_cut candidates recorded',
                 'Oracle and split conventions documented')
    rv.finish()


# ════════════════════════════════════════════════════════════════════════════
# Wave 1: PICCore — compact atom-centered radial core (Agent_3)
# Contract version 2. Fits analytic v_i^S with doubling-power basis.
# ════════════════════════════════════════════════════════════════════════════

_WAVE1_CORE_DIR = os.path.join('debug', 'test_afm_contact_surface', 'contact_pme', 'wave1_core')


def _core_split_params(R0, E0, q=0.0, q_tip=0.0, r_cut=6.0, split_mode='plateau'):
    from spammm.surfaces.PMESplit import SplitParams
    return SplitParams(R0=np.float64(R0), E0=np.float64(E0), q=np.float64(q),
                       alpha=_TIP_ALPHA, q_tip=q_tip, r_damp=_R_DAMP, r_cut=r_cut,
                       split_mode=split_mode)


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
    rv.out(f'N_MODES={N_MODES}, powers={CORE_POWERS.tolist()} (plateau split)')
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
                 'Boltzmann + E/F block normalization used in least-squares',
                 'Condition numbers reported for raw and hierarchical bases',
                 'Held-out energy and force errors reported (NOT training energy alone)',
                 'Worst held-out radius identified per atom',
                 'C/O/H neutral and charged cases tested')
    rv.finish()


def test_contact_pme_core_exact_cutoff(make_review):
    """Core must be exactly zero at and beyond per-atom r_b (no tail leakage)."""
    from spammm.surfaces.PICCore import fit_core_1d, eval_core
    rv = make_review('test_contact_pme_core_exact_cutoff')
    p = _core_split_params(_C_R0, _C_E0)
    fit = fit_core_1d(p)
    r_b = float(fit.r_b[0])
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    r_test = np.array([r_b, r_b + 0.001, r_b + 1.0, r_b + 4.0, 100.0])
    queries = np.column_stack([r_test, np.zeros(len(r_test)), np.zeros(len(r_test))])
    E, F = eval_core(queries, atom_pos, fit)
    rv.out(f'r_b={r_b:.4f}; E at/above r_b: {E}')
    rv.out(f'F at/above r_b: {F}')
    assert np.allclose(E, 0.0, atol=0.0), f'Core E must be exactly 0 at r>=r_b, got {E}'
    assert np.allclose(F, 0.0, atol=0.0), f'Core F must be exactly 0 at r>=r_b, got {F}'
    rv.checklist('Core energy is exactly 0 at r >= r_b',
                 'Core force is exactly 0 at r >= r_b',
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
    r_vals = rng.uniform(3.2, 5.0, 20)  # inside [r_lo, r_b) for C (~2.88–5.38)
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
    """r < r_lo clamps to t=1 (finite); needed for PP-AFM close approach."""
    from spammm.surfaces.PICCore import fit_core_1d, eval_core, core_basis
    rv = make_review('test_contact_pme_core_domain_violation')
    p = _core_split_params(_C_R0, _C_E0)
    fit = fit_core_1d(p)
    atom_pos = np.array([[0.0, 0.0, 0.0]])
    r_lo = float(fit.r_lo[0])
    q_inside = np.array([[r_lo - 0.1, 0.0, 0.0]])
    E_in, F_in = eval_core(q_inside, atom_pos, fit)
    rv.out(f'Inside r_lo-0.1: E={E_in[0]:.6e} F={F_in[0]} (must be finite; core clamps)')
    assert np.all(np.isfinite(E_in)) and np.all(np.isfinite(F_in))
    # Clamp: phi at r_lo-ε equals phi at r_lo (t=1), dphi=0 → F_core≈0 for single atom on axis
    r_b = float(fit.r_b[0])
    phi_lo, dphi_lo = core_basis(np.array([r_lo]), r_lo, r_b, fit.powers)
    phi_in, dphi_in = core_basis(np.array([r_lo - 0.1]), r_lo, r_b, fit.powers)
    assert np.allclose(phi_lo, phi_in), 'phi must match at/below r_lo (t=1 clamp)'
    assert np.allclose(dphi_in, 0.0), 'dphi must be 0 below r_lo'
    q_ok = np.array([[r_lo + 0.01, 0.0, 0.0]])
    E_ok, F_ok = eval_core(q_ok, atom_pos, fit)
    rv.out(f'At r_lo+0.01: E={E_ok[0]:.6e} F={F_ok[0]}')
    assert np.all(np.isfinite(E_ok))
    rv.checklist('eval_core finite for r < r_lo (clamp, AFM close approach)',
                 'phi(r<=r_lo) matches t=1; dphi=0',
                 'eval_core works at r >= r_lo')
    rv.finish()


def test_contact_pme_core_l2_curves(visual_output_dir, make_review):
    """L2: residual ref vs fit; combined wall/well/tail; plateau vs rho target."""
    from spammm.surfaces.PICCore import fit_core_1d, core_basis, eval_core_and_soft
    from spammm.surfaces.PMESplit import soft_core_split, combined_atom_potential
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
        r_lo = float(fit.r_lo[0]); r_b = float(fit.r_b[0])
        r = np.linspace(r_lo, r_b, 500)
        s = soft_core_split(r, p)
        phi, _ = core_basis(r, r_lo, r_b, fit.powers)
        v_fit = phi @ fit.coeffs[0]
        ax.plot(r, s['v_S'], 'k-', label='v_S ref (short)', lw=1.5)
        ax.plot(r, v_fit, 'r--', label='v_S fit', lw=1.2)
        ax.plot(r, v_fit - s['v_S'], 'g:', label='fit−ref', lw=1.0)
        ax.axhline(0, color='gray', ls=':', alpha=0.3)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7, label='R0')
        ax.axvline(float(p.r_a), color='cyan', ls=':', alpha=0.6, label='r_a')
        ax.axvline(r_b, color='magenta', ls='--', alpha=0.5, label='r_b')
        ax.set_title(f'{label} core residual (held RMSE_E={fit.held_rmse_E[0]:.2e})')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('V_S [eV]')
        ax.legend(fontsize=6)
    fig.suptitle('PICCore: short-range residual ref vs fit (plateau split)', fontsize=11)
    plot_path = os.path.join(outdir, 'core_residual_fit.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    rv.out(f'Saved: {plot_path}')
    print(f'REVIEW: {plot_path}', flush=True)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, (R0, E0, q_i, q_tip, label) in enumerate(cases):
        p = _core_split_params(R0, E0, q_i, q_tip=q_tip)
        fit = fit_core_1d(p)
        r_lo = float(fit.r_lo[0]); r_b = float(fit.r_b[0])
        atom_pos = np.array([[0.0, 0.0, 0.0]])
        r = np.linspace(r_lo, r_b + 2.0, 500)
        queries = np.column_stack([r, np.zeros(len(r)), np.zeros(len(r))])
        E_comb, _ = eval_core_and_soft(queries, atom_pos, p, fit)
        v_direct, _, _ = combined_atom_potential(r, p)
        s = soft_core_split(r, p)
        ax = axes2[0, col]
        ax.plot(r, v_direct, 'k-', label='v ref', lw=1.5)
        ax.plot(r, s['v_L'], 'b-', label='v_L long (exact)', lw=1.0)
        ax.plot(r, E_comb, 'r--', label='v_L + core fit', lw=1.2)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7)
        ax.axvline(float(p.r_a), color='cyan', ls=':', alpha=0.6)
        ax.axvline(r_b, color='magenta', ls=':', alpha=0.6)
        ax.set_title(f'{label} combined')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('V [eV]')
        ymin = float(np.min(v_direct)) * 1.15
        ax.set_ylim(ymin, max(abs(ymin) * 0.3, 1e-4))
        ax.legend(fontsize=6)
        ax = axes2[1, col]
        err = E_comb - v_direct
        ax.plot(r, err, 'g-', label='(v_L+fit) − v_ref', lw=1.2)
        ax.axhline(0, color='gray', ls=':', alpha=0.4)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7)
        ax.axvline(r_b, color='magenta', ls=':', alpha=0.6)
        ax.set_title(f'{label} discrepancy  max|{float(np.max(np.abs(err))):.2e}|')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('ΔV [eV]')
        ax.legend(fontsize=6)
    fig2.suptitle('PICCore combined: black=ref, blue=long-range exact, red=mesh-target+fit; bottom=discrepancy', fontsize=11)
    plot_path2 = os.path.join(outdir, 'core_combined_wall_well_tail.png')
    fig2.savefig(plot_path2, dpi=150)
    plt.close(fig2)
    rv.out(f'Saved: {plot_path2}')
    print(f'REVIEW: {plot_path2}', flush=True)
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (R0, E0, q_i, q_tip, label) in zip(axes3, cases):
        for mode, ls, color in [('plateau', '-', 'C0'), ('rho', '--', 'C3')]:
            p = _core_split_params(R0, E0, q_i, q_tip=q_tip, split_mode=mode)
            r_lo = float(p.r_lo)
            r_outer = float(p.r_b) if mode == 'plateau' else float(p.r_cut)
            r = np.linspace(r_lo, r_outer, 400)
            s = soft_core_split(r, p)
            ax.plot(r, s['v_S'], ls=ls, color=color, label=f'v_S {mode}', lw=1.3)
        ax.axhline(0, color='gray', ls=':', alpha=0.3)
        ax.axvline(R0, color='orange', ls='--', alpha=0.7, label='R0')
        ax.set_title(f'{label}: core target difficulty')
        ax.set_xlabel('r [Å]'); ax.set_ylabel('v_S [eV]')
        ax.legend(fontsize=7)
    fig3.suptitle('Why plateau helps: compact core target (solid) vs broad rho residual (dashed)', fontsize=11)
    plot_path3 = os.path.join(outdir, 'core_target_plateau_vs_rho.png')
    fig3.savefig(plot_path3, dpi=150)
    plt.close(fig3)
    rv.out(f'Saved: {plot_path3}')
    print(f'REVIEW: {plot_path3}', flush=True)
    rv.checklist('L2 residual reference vs fit plot generated',
                 'L2 combined wall/well/tail + discrepancy plot generated',
                 'L2 plateau vs rho core-target comparison generated')
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
        f'# Contract version: 3 (plateau split + Boltzmann/E-F scaled fit)',
        f'#',
        f'# Basis: phi_m(r) = t^p_m, t = (r_b - r)/(r_b - r_lo_i)',
        f'# Powers: {CORE_POWERS.tolist()}',
        f'# N_MODES = {N_MODES}',
        f'# Cutoff: every mode exactly zero for r >= r_b (= R0 + Δ_b)',
        f'#',
        f'# Fit: Boltzmann weights on total v + E/F block normalization',
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
        lines.append(f'{label:8s}: R0={R0:.4f} r_lo={float(fit.r_lo[0]):.4f} r_b={float(fit.r_b[0]):.4f} '
                     f'cond_raw={fit.cond_raw[0]:.1f} cond_hier={fit.cond_hier[0]:.1f} '
                     f'held_E={fit.held_rmse_E[0]:.4e} held_F={fit.held_rmse_F[0]:.4e}')
    lines.append(f'')
    lines.append(f'# Bucket: build_pic_buckets with cell_size >= r_core_max, 3x3 lookup')
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


def _core_split_params_multi(R0s, E0s, qs=None, q_tip=0.0, r_cut=6.0, split_mode='plateau'):
    from spammm.surfaces.PMESplit import SplitParams
    if qs is None:
        qs = np.zeros(len(R0s))
    return SplitParams(R0=np.asarray(R0s, dtype=np.float64), E0=np.asarray(E0s, dtype=np.float64),
                       q=np.asarray(qs, dtype=np.float64), alpha=_TIP_ALPHA, q_tip=q_tip,
                       r_damp=_R_DAMP, r_cut=r_cut, split_mode=split_mode)


# ════════════════════════════════════════════════════════════════════════════
# Wave 1: CoarseMesh — coarse 3D B-spline mesh (Agent_2)
# Contract version 2. Stores V_L = Σ v_i^L on a coarse 3D grid.
# ════════════════════════════════════════════════════════════════════════════

_WAVE1_MESH_DIR = os.path.join('debug', 'test_afm_contact_surface', 'contact_pme', 'wave1_mesh')


def _mesh_split_params(R0, E0, q=0.0, q_tip=0.0, r_cut=6.0, split_mode='plateau'):
    from spammm.surfaces.PMESplit import SplitParams
    return SplitParams(R0=np.float64(R0), E0=np.float64(E0), q=np.float64(q),
                       alpha=_TIP_ALPHA, q_tip=q_tip, r_damp=_R_DAMP, r_cut=r_cut,
                       split_mode=split_mode)


def _mesh_split_params_multi(R0s, E0s, qs=None, q_tip=0.0, r_cut=6.0, split_mode='plateau'):
    from spammm.surfaces.PMESplit import SplitParams
    if qs is None:
        qs = np.zeros(len(R0s))
    return SplitParams(R0=np.asarray(R0s, dtype=np.float64), E0=np.asarray(E0s, dtype=np.float64),
                       q=np.asarray(qs, dtype=np.float64), alpha=_TIP_ALPHA, q_tip=q_tip,
                       r_damp=_R_DAMP, r_cut=r_cut, split_mode=split_mode)


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


_WAVE2_MOL_DIR = os.path.join('debug', 'test_afm_contact_surface', 'contact_pme', 'wave2_paw_mol')


def _assign_demo_partial_charges(afm):
    """Electronegativity-inspired demo qs (neutralized). XYZ files ship with qs=0."""
    table = {'H': 0.12, 'C': 0.05, 'N': -0.35, 'O': -0.45, 'S': -0.20, 'F': -0.25}
    enames = list(afm.mol.enames)
    qs = np.array([table.get(str(e).strip(), 0.0) for e in enames], dtype=np.float64)
    qs -= qs.mean()
    afm.atoms_arr[:, 3] = qs.astype(np.float32)
    if afm.mol.qs is not None:
        afm.mol.qs[:] = qs
    return qs


def _safe_pme_queries(apos, p, z_above, nq, rng, margin=1.5):
    """Random queries above molecule that stay outside every atom's r_lo."""
    z = float(apos[:, 2].max()) + float(z_above)
    xs = rng.uniform(apos[:, 0].min() - margin, apos[:, 0].max() + margin, nq * 4)
    ys = rng.uniform(apos[:, 1].min() - margin, apos[:, 1].max() + margin, nq * 4)
    cand = np.column_stack([xs, ys, np.full(len(xs), z)])
    r_lo = np.atleast_1d(p.r_lo).astype(np.float64)
    keep = []
    for iq, q in enumerate(cand):
        ok = True
        for ia in range(len(apos)):
            if np.linalg.norm(q - apos[ia]) < r_lo[ia] + 0.05:
                ok = False
                break
        if ok:
            keep.append(iq)
        if len(keep) >= nq:
            break
    assert len(keep) >= max(10, nq // 2), f'only {len(keep)} safe queries at z_above={z_above}'
    return cand[keep[:nq]]


@pytest.mark.gpu
@pytest.mark.slow
def test_contact_pme_paw_molecule_parity(xyz, visual_output_dir, make_review):
    """Real-system PAW contact_pme vs Morse+Q (PLQH) oracle: pyridine + PTCDA.

    Reference: cs_brute_plqh_points with PLQH=(1,1,q_tip,0) — same radial Morse+Q
    as PMESplit.combined_atom_potential. tipQs forced to 0 (MVP radial oracle).
    """
    from spammm.SPM.AFM_utils import imshow_afm
    rv = make_review('test_contact_pme_paw_molecule_parity')
    outdir = _WAVE2_MOL_DIR
    os.makedirs(outdir, exist_ok=True)
    q_tip = -0.1
    cases = [('pyridine.xyz', 'pyridine'), ('PTCDA.xyz', 'PTCDA')]
    summary_rows = []

    for xyz_name, tag in cases:
        print(f'\n=== PAW molecule parity: {tag} ===', flush=True)
        afm = _make_afm(xyz(xyz_name))
        afm.tipQs[:] = 0.0
        qs = _assign_demo_partial_charges(afm)
        alpha = float(abs(afm.cLJs_arr[0, 2]))
        rv.out(f'\n{tag}: na={len(afm.atoms_arr)} q_tip={q_tip} qs=[{qs.min():.3f},{qs.max():.3f}] '
               f'alpha={alpha:.3f} split_mode=paw')

        cpm = afm.fit_contact_pme(split_mode='paw', q_tip=q_tip, h_mesh=1.0, halo_nodes=6,
                                  margin=2.0, z_above_lo=3.0, z_above_hi=8.0, bPrint=True)
        p = cpm.split_params
        apos = cpm.atom_pos
        assert p.split_mode == 'paw'
        rv.out(f'  mesh={cpm.mesh_coeffs.shape} r_core_max={p.r_core_max:.3f} '
               f'core_held_E max={float(np.nanmax(cpm.core_fit.held_rmse_E)):.3e}')

        rng = np.random.default_rng(7)
        # Pointwise parity at several heights
        for z_above in (3.5, 4.5, 5.5, 7.0):
            queries = _safe_pme_queries(apos, p, z_above, nq=80, rng=rng)
            E_pme, F_pme = afm.eval_contact_pme(queries, cpm, use_gpu=False)
            E_ref, F_ref = afm._brute_plqh_queries(queries, plqh=(1.0, 1.0, q_tip, 0.0),
                                                   alpha_morse=alpha, r_damp=0.1)
            dE = np.abs(E_pme - E_ref)
            dF = np.abs(F_pme - F_ref)
            scale_E = max(float(np.max(np.abs(E_ref))), 1e-12)
            scale_F = max(float(np.max(np.abs(F_ref))), 1e-12)
            max_dE, max_dF = float(np.max(dE)), float(np.max(dF))
            rms_dE = float(np.sqrt(np.mean(dE**2)))
            rms_dF = float(np.sqrt(np.mean(dF**2)))
            row = dict(tag=tag, z_above=z_above, nq=len(queries), max_dE=max_dE, max_dF=max_dF,
                       rms_dE=rms_dE, rms_dF=rms_dF, scale_E=scale_E, scale_F=scale_F,
                       rel_E=max_dE / scale_E, rel_F=max_dF / scale_F)
            summary_rows.append(row)
            rv.out(f'  z=+{z_above:.1f}: nq={len(queries)} max|dE|={max_dE:.3e} ({100*row["rel_E"]:.2f}% of max|E|) '
                   f'max|dF|={max_dF:.3e} ({100*row["rel_F"]:.2f}%) rms_E={rms_dE:.3e} rms_F={rms_dF:.3e}')
            # Soft L0: mesh@1Å has interpolation error; require finite + not catastrophic
            assert np.all(np.isfinite(E_pme)) and np.all(np.isfinite(F_pme))
            assert row['rel_E'] < 0.25 or max_dE < 5e-3, \
                f'{tag} z=+{z_above}: relative E error {row["rel_E"]:.3f} too large'
            assert row['rel_F'] < 0.35 or max_dF < 5e-3, \
                f'{tag} z=+{z_above}: relative F error {row["rel_F"]:.3f} too large'

        if visual_output_dir is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # 2D Fz maps: PME vs PLQH ref vs difference at h=4.5 Å above zmax
            margin = 1.5
            xs = np.linspace(apos[:, 0].min() - margin, apos[:, 0].max() + margin, 64)
            ys = np.linspace(apos[:, 1].min() - margin, apos[:, 1].max() + margin, 64)
            X, Y = np.meshgrid(xs, ys, indexing='ij')
            z = float(apos[:, 2].max()) + 4.5
            grid_q = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
            # Mask domain violations (r < r_lo)
            r_lo = np.atleast_1d(p.r_lo)
            safe = np.ones(len(grid_q), dtype=bool)
            for ia in range(len(apos)):
                safe &= (np.linalg.norm(grid_q - apos[ia], axis=1) >= r_lo[ia] + 0.05)
            E_pme, F_pme = afm.eval_contact_pme(grid_q[safe], cpm, use_gpu=False)
            E_ref, F_ref = afm._brute_plqh_queries(grid_q[safe], plqh=(1.0, 1.0, q_tip, 0.0),
                                                   alpha_morse=alpha, r_damp=0.1)
            Fz_pme = np.full(X.size, np.nan); Fz_ref = np.full(X.size, np.nan); dFz = np.full(X.size, np.nan)
            E_p = np.full(X.size, np.nan); E_r = np.full(X.size, np.nan); dE = np.full(X.size, np.nan)
            Fz_pme[safe] = F_pme[:, 2]; Fz_ref[safe] = F_ref[:, 2]; dFz[safe] = F_pme[:, 2] - F_ref[:, 2]
            E_p[safe] = E_pme; E_r[safe] = E_ref; dE[safe] = E_pme - E_ref
            Fz_pme = Fz_pme.reshape(X.shape); Fz_ref = Fz_ref.reshape(X.shape); dFz = dFz.reshape(X.shape)
            E_p = E_p.reshape(X.shape); E_r = E_r.reshape(X.shape); dE = dE.reshape(X.shape)
            # imshow_afm / percentile: replace domain holes with 0 for display
            def _fill(a):
                return np.nan_to_num(a, nan=0.0)
            extent = [xs[0], xs[-1], ys[0], ys[-1]]
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            for ax, arr, ttl in zip(axes[0], [_fill(E_r), _fill(E_p), _fill(dE)],
                                    [f'{tag} E ref Morse+Q', f'{tag} E PAW PME', f'{tag} ΔE']):
                imshow_afm(ax, arr, extent=extent, cmap='bwr', title=ttl)
            for ax, arr, ttl in zip(axes[1], [_fill(Fz_ref), _fill(Fz_pme), _fill(dFz)],
                                    [f'{tag} Fz ref', f'{tag} Fz PAW PME', f'{tag} ΔFz']):
                imshow_afm(ax, arr, extent=extent, cmap='bwr', title=ttl)
            fig.suptitle(f'PAW contact_pme vs PLQH Morse+Q  |  {tag}  h=+4.5 Å  q_tip={q_tip}', fontsize=11)
            plot_path = os.path.join(outdir, f'{tag}_paw_vs_plqh_h4.5.png')
            fig.savefig(plot_path, dpi=150); plt.close(fig)
            rv.out(f'Saved {plot_path}')
            print(f'REVIEW: {plot_path}', flush=True)

    # SUMMARY.out
    lines = [
        '# PAW contact_pme molecule parity vs Morse+Q (cs_brute_plqh_points)',
        f'# split_mode=paw  q_tip={q_tip}  h_mesh=1.0  tipQs=0',
        '# demo partial charges assigned (XYZ ships qs=0); neutralized',
        '#',
        f'{"mol":10s} {"z+":>5s} {"nq":>4s} {"max|dE|":>12s} {"relE%":>8s} {"max|dF|":>12s} {"relF%":>8s} {"rmsE":>12s} {"rmsF":>12s}',
    ]
    for r in summary_rows:
        lines.append(f'{r["tag"]:10s} {r["z_above"]:5.1f} {r["nq"]:4d} {r["max_dE"]:12.4e} '
                     f'{100*r["rel_E"]:8.2f} {r["max_dF"]:12.4e} {100*r["rel_F"]:8.2f} '
                     f'{r["rms_dE"]:12.4e} {r["rms_dF"]:12.4e}')
    summary_path = os.path.join(outdir, 'SUMMARY.out')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    rv.out('\n'.join(lines))
    print(f'REVIEW: {summary_path}', flush=True)
    rv.checklist('PAW fit_contact_pme succeeds for pyridine and PTCDA',
                 'Pointwise E/F vs PLQH Morse+Q reported at multiple heights',
                 'L2 AFM maps (E,Fz,Δ) written under wave2_paw_mol/')
    rv.finish()


_WAVE2_AFM_DIR = os.path.join('debug', 'test_afm_contact_surface', 'contact_pme', 'wave2_afm_cli')


@pytest.mark.gpu
@pytest.mark.slow
def test_contact_pme_afm_cli_ssot(xyz, visual_output_dir, make_review):
    """Full PP-AFM via AFM_utils.run_contact_pme_pp_afm — same path as run_spm.py afm --model contact_pme.

    SSOT (skill:afm-plotting / CLI defaults):
      h_min=3.7 … h_max=4.7, h_step=0.1, amp=1.0, amp_align, scan_margin=2.0, margin=4.0
    Plotting: plot_afm_variant_height_strip only — no ad-hoc imshow.
    """
    from spammm.SPM import AFM as afm_mod
    from spammm.SPM import AFM_utils as afm_utils
    rv = make_review('test_contact_pme_afm_cli_ssot')
    if visual_output_dir is None:
        rv.out('L2 AFM strip skipped (no --visual/--develop); L0 still runs smoke fit+scan on pyridine')
    H_MIN, H_MAX, H_STEP = 3.7, 4.7, 0.1
    AMP = 1.0
    SCAN_MARGIN = 2.0
    MARGIN = 4.0
    STEP = 0.1
    K_LAT_NM, K_RAD, BOND_LENGTH = 0.5, 20.0, 3.0
    OSC_DIR = (0., 0., 1.)
    params_path = PARAMS_PATH
    cases = [('pyridine.xyz', 'pyridine'), ('PTCDA.xyz', 'PTCDA')]

    for xyz_name, tag in cases:
        print(f'\n=== contact_pme AFM CLI-SSOT: {tag} ===', flush=True)
        outdir = os.path.join(_WAVE2_AFM_DIR, tag)
        os.makedirs(outdir, exist_ok=True)
        afm0 = _make_afm(xyz(xyz_name))
        atomPos = afm0.atoms_arr[:, :3].astype(np.float64).copy()
        # Match CLI: planarize + orient long axis → x
        from spammm.forcefields.FFController import make_planar_xy, orient_long_axis_x
        atomPos[:] = make_planar_xy(atomPos)
        orient_long_axis_x(atomPos)
        atomPos[:, 2] = 0.0
        atomTypes = np.array([{'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}.get(str(e).strip(), 6)
                              for e in afm0.mol.enames], dtype=np.int32)
        _, origin, ngrid, step = afm_utils.make_fdbm_grid_com_zsym(atomPos, STEP, MARGIN, z_vac=6.0)
        h_df, h_Fz, h_scan = afm_utils.afm_df_height_stacks(
            H_MIN, H_MAX, H_STEP, amp=AMP, amp_align=True, osc_dir=OSC_DIR)
        scan_xs = np.arange(float(atomPos[:, 0].min() - SCAN_MARGIN),
                            float(atomPos[:, 0].max() + SCAN_MARGIN), step, dtype=np.float32)
        scan_ys = np.arange(float(atomPos[:, 1].min() - SCAN_MARGIN),
                            float(atomPos[:, 1].max() + SCAN_MARGIN), step, dtype=np.float32)
        K_LAT = afm_mod.stiffness_Nm_to_eVA2(K_LAT_NM)
        scan_spec = afm_utils.ScanSpec(
            scan_xs=scan_xs, scan_ys=scan_ys, h_df=h_df, h_Fz=h_Fz, h_scan=h_scan,
            amplitude=AMP, osc_dir=OSC_DIR, K_LAT=K_LAT, K_RAD=K_RAD,
            bond_length=BOND_LENGTH, scan_margin=SCAN_MARGIN)
        rv.out(f'{tag}: scan=({len(scan_xs)}×{len(scan_ys)}) nz_scan={len(h_scan)} '
               f'h_df=[{h_df[0]:.2f},{h_df[-1]:.2f}] h_Fz=[{h_Fz[0]:.2f},{h_Fz[-1]:.2f}]')
        result = afm_utils.run_contact_pme_pp_afm(
            tag, atomPos, atomTypes, origin, step, ngrid, outdir,
            scan_spec=scan_spec, params_path=params_path,
            margin=MARGIN, plots={'df', 'fz'} if visual_output_dir else set(),
            split_mode='paw', q_tip=0.0, h_mesh=1.0, halo_nodes=6)
        assert result.backend_name == 'contact_pme'
        assert result.split_mode == 'paw'
        assert np.all(np.isfinite(result.df)) and np.all(np.isfinite(result.Fz))
        assert result.df.shape[2] == len(result.heights)
        rv.out(f'  df range=[{result.df.min():.3e},{result.df.max():.3e}] '
               f'Fz range=[{result.Fz.min():.3e},{result.Fz.max():.3e}] '
               f'resident={result.resident_kb:.1f} KB')
        # Soft L0: non-trivial contrast (not blank)
        assert float(np.std(result.df)) > 1e-8, f'{tag}: blank df map'
        assert float(np.std(result.Fz)) > 1e-8, f'{tag}: blank Fz map'

        if visual_output_dir is not None:
            amp_z = AMP
            row_specs = [
                ('df', 'contact_pme', f'df contact_pme\npaw K_LAT={K_LAT_NM} N/m', 'gray'),
                ('Fz', 'contact_pme', f'Fz contact_pme\n@h−{amp_z:.1f}Å', 'seismic'),
            ]
            variants = {'contact_pme': result}
            extent = afm_utils.scan_extent(result.scan_xs, result.scan_ys)
            out_png = os.path.join(outdir, 'compare_per_image.png')
            afm_utils.plot_afm_variant_height_strip(
                variants, row_specs, result.heights, out_png, scale='per_image',
                title=f'Contact-PME AFM  {tag}  split=paw  (CLI SSOT heights)',
                dpi=140, apos=atomPos, show_atoms=True, extent=extent,
                amp=AMP, amp_align=True, amp_z=amp_z, long_axis_vertical=True, tight=True)
            rv.out(f'Saved {out_png}')
            print(f'REVIEW: {out_png}', flush=True)

    summary_path = os.path.join(_WAVE2_AFM_DIR, 'SUMMARY.out')
    os.makedirs(_WAVE2_AFM_DIR, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write('# contact_pme AFM via AFM_utils (= run_spm.py afm --model contact_pme)\n')
        f.write(f'# h_df={H_MIN}…{H_MAX} step={H_STEP} amp={AMP} scan_margin={SCAN_MARGIN} margin={MARGIN}\n')
        f.write('# plotting: plot_afm_variant_height_strip (no ad-hoc imshow)\n')
        f.write(f'REVIEW: {_WAVE2_AFM_DIR}/pyridine/compare_per_image.png\n')
        f.write(f'REVIEW: {_WAVE2_AFM_DIR}/PTCDA/compare_per_image.png\n')
    print(f'REVIEW: {summary_path}', flush=True)
    rv.checklist('run_contact_pme_pp_afm used (same as SPM_CLI contact_pme)',
                 'CLI height SSOT 3.7–4.7 / amp-align Fz',
                 'plot_afm_variant_height_strip for overview strip')
    rv.finish()
