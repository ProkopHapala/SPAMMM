"""
test_bspline_boundary.py — L0: cubic B-spline prefilter boundary correctness.

Verifies that _bspline_prefilter_1d / _bspline_prefilter_2d produce control
coefficients that exactly reproduce nodal values at EVERY node — including the
first and last (boundary) — when interpolated with the zero-padding convention
used by cs_interp_h0 (kernel) / interp_h0 (Python).

The previous causal/anti-causal IIR (Unser infinite-signal filter) gave ~1e-16
interior error but O(1e-2) error at the first node. The tridiagonal solve
(inverse of the finite zero-padded interpolation matrix) is exact everywhere.
"""
import numpy as np
import pytest
from scipy.linalg import solve_banded

from spammm.surfaces.ContactSurface import (
    _bspline_prefilter_1d, _bspline_prefilter_2d, interp_h0, build_contact_height_map,
)


def _eval_bspline_1d(i, coeffs):
    """Evaluate cubic B-spline at node i with zero-padding (matches cs_interp_h0 1D)."""
    s = 0.0
    weights = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0]
    for k, ii in enumerate([i - 1, i, i + 1, i + 2]):
        if 0 <= ii < len(coeffs):
            s += weights[k] * coeffs[ii]
    return s


@pytest.mark.parametrize("n", [5, 10, 20, 50])
def test_prefilter_1d_interior_exact(n):
    """Interior nodes reproduce nodal values to machine precision."""
    xs = np.arange(n) * 0.3
    data = np.sin(xs) + 0.5 * np.cos(xs * 0.7)
    coeffs = _bspline_prefilter_1d(data)
    recon = np.array([_eval_bspline_1d(i, coeffs) for i in range(n)])
    err = np.abs(recon - data)
    assert err[1:-1].max() < 1e-10, f"interior max err {err[1:-1].max():.3e}"


@pytest.mark.parametrize("n", [5, 10, 20, 50])
def test_prefilter_1d_boundary_exact(n):
    """Boundary nodes (first and last) reproduce nodal values to machine precision.

    This is the key fix: the old IIR prefilter had O(1e-2) error at the first node.
    """
    xs = np.arange(n) * 0.3
    data = np.sin(xs) + 0.5 * np.cos(xs * 0.7)
    coeffs = _bspline_prefilter_1d(data)
    recon = np.array([_eval_bspline_1d(i, coeffs) for i in range(n)])
    err = np.abs(recon - data)
    assert err[0] < 1e-10, f"first node err {err[0]:.3e}"
    assert err[-1] < 1e-10, f"last node err {err[-1]:.3e}"


@pytest.mark.parametrize("ncx,ncy", [(6, 7), (10, 10), (15, 8)])
def test_prefilter_2d_all_nodes_exact(ncx, ncy):
    """2D separable prefilter: all nodes (incl. boundaries) reproduce via interp_h0."""
    dx = 0.4
    x0, y0 = 0.0, 0.0
    xs = np.arange(ncx) * dx
    ys = np.arange(ncy) * dx
    # h0_2d stored as (ncy, ncx) C-order, flat index jy*ncx+ix (matches build_contact_height_map)
    h0_2d = np.sin(xs[None, :]) + 0.3 * np.cos(ys[:, None] * 0.8)
    coeffs_2d = _bspline_prefilter_2d(h0_2d)
    cflat = coeffs_2d.reshape(-1)
    maxerr = 0.0
    for ix in range(ncx):
        for iy in range(ncy):
            v = interp_h0(float(xs[ix]), float(ys[iy]), cflat, dx, x0, y0, ncx, ncy)
            maxerr = max(maxerr, abs(v - h0_2d[iy, ix]))
    assert maxerr < 1e-10, f"2D max node err {maxerr:.3e}"


def test_prefilter_matches_tridiag_solve():
    """Prefilter output must match direct tridiagonal solve (the exact inverse)."""
    n = 20
    xs = np.arange(n) * 0.3
    data = np.sin(xs) + 0.5 * np.cos(xs * 0.7)
    coeffs = _bspline_prefilter_1d(data)
    # Direct tridiagonal solve: A c = d, A=tridiag(1/6, 4/6, 1/6)
    ab = np.zeros((3, n))
    ab[0, 1:] = 1.0 / 6.0
    ab[1, :] = 4.0 / 6.0
    ab[2, :-1] = 1.0 / 6.0
    ref = solve_banded((1, 1), ab, data)
    assert np.allclose(coeffs, ref, atol=1e-12), f"max diff {np.abs(coeffs - ref).max():.3e}"


def test_build_contact_height_map_returns_dict():
    """build_contact_height_map returns dict with h0_samples and h0_coeffs."""
    apos = np.array([[0, 0, 1.0], [1, 0, 1.5], [0, 1, 1.2], [1, 1, 1.8]], dtype=np.float64)
    Rs = np.array([2.0, 2.0, 2.0, 2.0])
    result = build_contact_height_map(apos, -2, -2, 0.5, 0.5, 10, 10, r_xy=8.0, Rs=Rs)
    assert isinstance(result, dict)
    assert 'h0_samples' in result and 'h0_coeffs' in result
    assert result['h0_samples'].shape == (100,)
    assert result['h0_coeffs'].shape == (100,)


def test_h0_coeffs_reproduce_samples_at_nodes():
    """h0_coeffs from build_contact_height_map reproduce h0_samples at every node via interp_h0."""
    apos = np.array([[0, 0, 1.0], [1, 0, 1.5], [0, 1, 1.2], [1, 1, 1.8]], dtype=np.float64)
    Rs = np.array([2.0, 2.0, 2.0, 2.0])
    ncx, ncy, dx, x0, y0 = 10, 10, 0.5, -2.0, -2.0
    result = build_contact_height_map(apos, x0, y0, dx, dx, ncx, ncy, r_xy=8.0, Rs=Rs)
    samples = result['h0_samples'].reshape(ncy, ncx)
    cflat = result['h0_coeffs']
    maxerr = 0.0
    for ix in range(ncx):
        for iy in range(ncy):
            cx = x0 + ix * dx
            cy = y0 + iy * dx
            v = interp_h0(float(cx), float(cy), cflat, dx, x0, y0, ncx, ncy)
            maxerr = max(maxerr, abs(v - samples[iy, ix]))
    assert maxerr < 1e-6, f"contact height map node err {maxerr:.3e}"


def test_h0_samples_are_physical_values():
    """h0_samples contain the raw nodal heights (physical contact values), not coefficients.

    Coefficients can overshoot/undershoot the nodal range. Samples must stay within
    [zmin, zmax+Rmax] for the sphere-envelope mode.
    """
    apos = np.array([[0, 0, 1.0], [1, 0, 1.5], [0, 1, 1.2], [1, 1, 1.8]], dtype=np.float64)
    Rs = np.array([2.0, 2.0, 2.0, 2.0])
    result = build_contact_height_map(apos, -2, -2, 0.5, 0.5, 10, 10, r_xy=8.0, Rs=Rs)
    samples = result['h0_samples']
    coeffs = result['h0_coeffs']
    # Samples range: zmin to zmax + Rmax (sphere envelope)
    zmin = float(apos[:, 2].min())
    zmax_Rmax = float(apos[:, 2].max()) + float(Rs.max())
    assert samples.min() >= zmin - 1e-6
    assert samples.max() <= zmax_Rmax + 1e-6
    # Coefficients CAN exceed this range (B-spline control points overshoot)
    # — this is exactly why h0_min/h0_max must come from samples, not coeffs.
    coeff_range = float(coeffs.max()) - float(coeffs.min())
    sample_range = float(samples.max()) - float(samples.min())
    # Coefficients typically have a wider range than samples
    assert coeff_range >= sample_range - 1e-6


def test_coremesh_collocation_sparse_parity():
    from spammm.surfaces.CoarseMesh import _colloc_axis, fit_mesh_lsq_3d, _prefilter_3d
    for s in (1, 2, 3):
        B = _colloc_axis(9, s, sparse=True)
        assert B.nnz <= 4*B.shape[0]
        np.testing.assert_array_equal(B.toarray(), _colloc_axis(9, s))
    R = np.random.default_rng(11).normal(size=(7, 8, 9))
    np.testing.assert_allclose(fit_mesh_lsq_3d(R, s=1), _prefilter_3d(R), atol=1e-12)


def test_coremesh_vectorized_force_parity():
    from spammm.surfaces.CoarseMesh import CoarseMesh, eval_mesh
    mesh = CoarseMesh(np.random.default_rng(8).normal(size=(12, 13, 14)), np.zeros(3), 0.7, 2, (np.full(3, 2), np.full(3, 9)))
    q = np.random.default_rng(9).uniform(1.5, 6.0, (35, 3))
    E, F = eval_mesh(mesh, q)
    Er, Fr = eval_mesh(mesh, q, vectorized=False)
    np.testing.assert_allclose(E, Er, atol=1e-13)
    np.testing.assert_allclose(F, Fr, atol=1e-13)
    for a in range(3):
        dq = np.zeros(3); dq[a] = 1e-5
        np.testing.assert_allclose(F[:, a], -(eval_mesh(mesh, q+dq)[0]-eval_mesh(mesh, q-dq)[0])/2e-5, atol=1e-8)
    with pytest.raises(ValueError, match='stencil out of bounds'):
        eval_mesh(mesh, [[0, 0, 0]])


@pytest.mark.slow
def test_coremesh_joint_weighted_objective(make_review):
    """Overlapping bases recover a known smooth core; baseline uses identical weights."""
    from scipy.linalg import null_space
    from spammm.surfaces.CoarseMesh import _colloc_axis, _apply_along, fit_coremesh_lsq, eval_mesh
    from spammm.surfaces.PICCore import CORE_POWERS, fit_cores_from_samples, eval_core_fast
    rv = make_review('test_coremesh_joint_weighted_objective')
    origin = np.full(3, -4.2); h = 0.7; ns = (13,)*3; s = 2; shape = (25,)*3
    P = np.stack(np.meshgrid(*[origin[a]+np.arange(shape[a])*h/s for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
    apos = np.zeros((1, 3)); r_lo, r_b = 0.8, 3.5
    p = CORE_POWERS.astype(float)
    Z = null_space(np.stack([p, p*(p-1), (p == 2).astype(float)]))
    coeffs = Z @ [3.0, -2.0]
    A, _ = fit_cores_from_samples(apos, P, np.zeros(len(P)), r_lo, r_b, return_design=True)
    long = np.full(ns, -0.02)
    target = long.copy()
    for a in range(3):
        target = _apply_along(_colloc_axis(ns[a], s, sparse=True), target, a)
    target += (A @ coeffs).reshape(shape)
    w = np.exp(-np.maximum(target, 0)/0.1) + 1e-4
    mesh, core, diag = fit_coremesh_lsq(target, apos, origin, h, r_lo, r_b, s=s, weights=w, lam=1e-7, core_ridge=1e-10, tol=1e-10)
    baseline, _, base = fit_coremesh_lsq(target, np.empty((0, 3)), origin, h, [], [], s=s, weights=w, lam=1e-7, core_ridge=1e-10, tol=1e-10)
    assert diag['objective'] < base['objective']*0.05
    q = np.random.default_rng(31).uniform(-2.6, 2.6, (500, 3))
    ref_core = core.coeffs.copy(); core.coeffs[:] = coeffs
    E_ref, F_ref = eval_core_fast(q, apos, core)
    E_ref -= 0.02
    core.coeffs[:] = ref_core
    E_m, F_m = eval_mesh(mesh, q); E_c, F_c = eval_core_fast(q, apos, core)
    E_b, F_b = eval_mesh(baseline, q)
    err = np.sqrt(np.mean((E_m+E_c-E_ref)**2)); err_b = np.sqrt(np.mean((E_b-E_ref)**2))
    ferr = np.sqrt(np.mean((F_m+F_c-F_ref)**2)); ferr_b = np.sqrt(np.mean((F_b-F_ref)**2))
    assert err < err_b*0.1
    assert ferr < ferr_b*0.1
    np.testing.assert_allclose(core.coeffs @ p, 0, atol=1e-12)
    np.testing.assert_allclose(core.coeffs @ (p*(p-1)), 0, atol=1e-11)
    np.testing.assert_allclose(core.coeffs[:, 0], 0, atol=1e-12)
    for r in (r_lo, r_b):
        qq = np.array([[r-1e-7, 0, 0], [r+1e-7, 0, 0]])
        Ee, Fe = eval_core_fast(qq, apos, core)
        assert abs(Ee[1]-Ee[0]) < 1e-7
        assert np.max(np.abs(Fe[1]-Fe[0])) < 1e-6
    rv.out(f'joint J={diag["objective"]:.8g}, spline-only J={base["objective"]:.8g}; held E rms {err:.8g} vs {err_b:.8g}; F rms {ferr:.8g} vs {ferr_b:.8g}')
    rv.checklist('Same weighted objective for core and mesh', 'Core prefit initializes joint fit', 'C2 radial joins without a physical handoff surface', 'Independent off-grid E and F improve over spline-only')
    rv.finish()


def test_coremesh_c2_clamp():
    from spammm.SPM.AFM_utils import soft_clamp_rational
    y = np.array([-0.1, 0.1, 0.3, 0.301, 1.0, 1000.0])
    T, dT = soft_clamp_rational(y, 0.3, 1.0, dy=np.ones_like(y), smoothness=2)
    np.testing.assert_allclose(T[:3], y[:3], atol=1e-8)
    np.testing.assert_array_equal(dT[:3], 1)
    assert np.all(np.diff(T)>0) and T.max()<=1.0
    assert abs(dT[3]-1)<1e-5
    with pytest.raises(ValueError):
        soft_clamp_rational(y, 0.3, 1.0, smoothness=3)


def test_coremesh_invalid_inputs():
    from spammm.surfaces.CoarseMesh import fit_coremesh_lsq
    E = np.zeros((7, 7, 7)); apos = np.zeros((1, 3))
    for opts in (dict(s=0), dict(s=4), dict(weights=np.zeros_like(E)), dict(weights=-np.ones_like(E)), dict(lam=-1)):
        with pytest.raises(ValueError):
            fit_coremesh_lsq(E, apos, np.zeros(3), 0.5, 0.5, 3.5, **opts)
    with pytest.raises(ValueError):
        fit_coremesh_lsq(E, apos, np.zeros(3), 0.5, np.nan, 3.5)
    with pytest.raises(RuntimeError, match='did not converge'):
        fit_coremesh_lsq(np.random.default_rng(6).normal(size=E.shape), np.empty((0, 3)), np.zeros(3), 0.5, [], [], maxiter=1)


def test_coremesh_smooth_basis_joins():
    from spammm.surfaces.PICCore import core_basis
    r = np.array([0.8-1e-5, 0.8, 0.8+1e-5, 3.5-1e-5, 3.5, 3.5+1e-5])
    phi, dphi = core_basis(r, 0.8, 3.5, smooth=True)
    np.testing.assert_allclose(phi[:3], 1, atol=2e-12)
    assert np.abs(dphi[:3]).max()<1e-7
    assert np.abs(phi[3:]).max()<1e-25 and np.abs(dphi[3:]).max()<1e-18
    x = np.linspace(0.9, 3.4, 25); eps=1e-5
    f, df = core_basis(x, 0.8, 3.5, smooth=True)
    f1 = core_basis(x+eps, 0.8, 3.5, smooth=True)[0]; f0 = core_basis(x-eps, 0.8, 3.5, smooth=True)[0]
    np.testing.assert_allclose(df, (f1-f0)/(2*eps), atol=1e-8)
    assert np.linalg.matrix_rank(f)==5


def test_coremesh_projected_fit(make_review):
    from spammm.surfaces.CoarseMesh import _colloc_axis, _apply_along, fit_coremesh_projected, eval_mesh
    from spammm.surfaces.PICCore import fit_cores_from_samples, eval_core_fast
    rv = make_review('test_coremesh_projected_fit')
    origin=np.full(3, -4.2); h=0.7; s=2; ns=(13,)*3; shape=(25,)*3; apos=np.zeros((1, 3))
    pts=np.stack(np.meshgrid(*[origin[a]+np.arange(shape[a])*h/s for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
    A,_=fit_cores_from_samples(apos, pts, np.zeros(len(pts)), 0.8, 3.5, return_design=True, smooth=True)
    c=np.array([0.2, -0.1, 0.3, -0.2, 0.1])
    target=np.full(ns, -0.02)
    for a in range(3): target=_apply_along(_colloc_axis(ns[a], s, sparse=True), target, a)
    target += (A @ c).reshape(shape)
    w=np.exp(-np.maximum(target, 0)/0.1)+1e-4
    mesh,core,di=fit_coremesh_projected(target, apos, origin, h, 0.8, 3.5, weights=w, lam=0, core_ridge=1e-12)
    baseline,_,db=fit_coremesh_projected(target, np.empty((0,3)), origin, h, [], [], weights=w, lam=0, core_ridge=1e-12)
    assert di['objective']<db['objective']*1e-5 and di['iterations']==0
    np.testing.assert_allclose(core.coeffs[0], c, atol=2e-5)
    q=np.random.default_rng(42).uniform(-2.6, 2.6, (150,3)); E,F=eval_mesh(mesh,q); ec,fc=eval_core_fast(q,apos,core)
    old=core.coeffs.copy(); core.coeffs[:]=c; er,fr=eval_core_fast(q,apos,core); core.coeffs[:]=old
    np.testing.assert_allclose(E+ec, er-0.02, atol=1e-7)
    np.testing.assert_allclose(F+fc, fr, atol=2e-7)
    packed=np.stack([target, 2*target], -1)
    from spammm.surfaces.CoarseMesh import fit_mesh_lsq_3d
    fit=fit_mesh_lsq_3d(packed, s=s)
    np.testing.assert_allclose(fit[...,1], 2*fit[...,0], atol=1e-12)
    with pytest.raises(MemoryError): fit_coremesh_projected(target,apos,origin,h,0.8,3.5,max_workspace_bytes=1)
    with pytest.raises(ValueError): fit_coremesh_projected(target,apos,origin,h,0.8,3.5,s=1)
    rv.out(f'Projected J={di["objective"]:.9g}, baseline J={db["objective"]:.9g}; zero CG iterations; total {di["total_ms"]:.3f} ms')
    rv.checklist('Five independent C2 modes', 'Direct banded mesh elimination', 'Same weighted reduced objective', 'Analytic held-out E/F parity', 'Bounded workspace and no silent fallback')
    rv.finish()


@pytest.mark.gpu
def test_coremesh_smooth_gpu_basis(make_review):
    """Compile only the actual tiny radial evaluator; no field generation/scan."""
    from pathlib import Path
    import pyopencl as cl
    from spammm.utils.OpenCLBase import select_device
    from spammm.surfaces.PICCore import core_basis
    rv=make_review('test_coremesh_smooth_gpu_basis')
    ctx,queue=select_device(preferred_vendor='nvidia', return_queue=True)
    if 'nvidia' not in ctx.devices[0].vendor.lower(): pytest.skip('NVIDIA OpenCL device unavailable')
    source=(Path(__file__).resolve().parents[2]/'kernels/contact_surface.cl').read_text()
    fn='inline void cs_pme_core_basis'+source.split('inline void cs_pme_core_basis',1)[1].split('inline float2 cs_pme_core_reduce',1)[0]
    wrapper='\n__kernel void probe(__global const float* r, __global float* out){ int i=get_global_id(0); float p[5],d[5]; cs_pme_core_basis(r[i],0.8f,3.5f,p,d); for(int k=0;k<5;k++){out[10*i+k]=p[k];out[10*i+5+k]=d[k];}}'
    r=np.r_[0.0,0.8,np.linspace(0.81,3.49,100),3.5,4.0].astype(np.float32)
    rb=cl.Buffer(ctx,cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR,hostbuf=r)
    out=np.empty((len(r),10),np.float32); ob=cl.Buffer(ctx,cl.mem_flags.WRITE_ONLY,out.nbytes)
    for smooth in (0,1):
        program=cl.Program(ctx,fn+wrapper).build(options=[f'-DCS_PME_SMOOTH_CORE={smooth}'])
        program.probe(queue,(len(r),),None,rb,ob); cl.enqueue_copy(queue,out,ob).wait()
        # Input parity first: 0.8f lies just ABOVE float64 0.8. At a raw
        # clipping join that changes which one-sided derivative is selected.
        p,d=core_basis(r,float(np.float32(0.8)),float(np.float32(3.5)),smooth=bool(smooth))
        np.testing.assert_allclose(out[:,:5],p,atol=8e-6,rtol=3e-5)
        np.testing.assert_allclose(out[:,5:],d,atol=2e-5,rtol=3e-5)
        rv.out(f'{ctx.devices[0].name} smooth={smooth}: max E basis delta={np.max(abs(out[:,:5]-p)):.8g}, max derivative delta={np.max(abs(out[:,5:]-d)):.8g}')
    rv.finish()


def test_coremesh_gpu_basis_mismatch():
    from types import SimpleNamespace
    from spammm.SPM.AFM import AFMulator
    afm=AFMulator.__new__(AFMulator); afm.pme_core_basis='raw'
    params=SimpleNamespace(core_fit=SimpleNamespace(basis='smooth',coeffs=np.ones((1,5))))
    with pytest.raises(ValueError, match='mismatches compiled'):
        afm._pme_upload_resident(params)
