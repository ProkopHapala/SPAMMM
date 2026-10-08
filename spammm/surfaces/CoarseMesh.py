"""
CoarseMesh.py — Coarse 3D B-spline mesh for the contact_pme particle-mesh backend.

Stores the smooth long-range part V_L = Σ_i v_i^L(r) on a coarse 3D cubic B-spline
grid in world coordinates. Contract version 2.

Layout: C-order (nx, ny, nz) with z fastest; frozen for Python/OpenCL parity.
Spacing: h_mesh = 1.0 Å for MVP.
Interpolant: Nonperiodic cardinal cubic B-spline; analytic gradient from same
64-tap stencil. Prefilter REUSES _bspline_prefilter_1d from ContactSurface.py
(separable along x/y/z). No PBC wrapping.

Energy and force are always paired: every evaluator returns (E, F) with F = -∇E.
Sampled FDBM fields: fit_coremesh_lsq uses one weighted objective, C2-constrained
compact cores, and matrix-free mesh products. No pointwise residual caps or inpainting.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from spammm.surfaces.PMESplit import SplitParams, soft_core_split, precompute_split_cache
from spammm.surfaces.ContactSurface import _bspline_prefilter_1d, _bspline_tridiag_ab
from scipy.linalg import solve_banded


@dataclass
class CoarseMesh:
    """Coarse 3D B-spline mesh storing V_L control coefficients."""
    coeffs: np.ndarray      # (nx, ny, nz) float64 — B-spline control coefficients (NOT nodal samples)
    origin: np.ndarray      # (3,) world coords of node (0,0,0)
    h: float                # mesh spacing [Å]
    halo: int               # halo node count per side used at build time
    query_interior: tuple   # (lo[3], hi[3]) integer node indices defining the safe query interior


# ── cubic B-spline basis (matches gridFF.cl:72-93) ──────────────────────────

def _basis(u):
    """Cardinal cubic B-spline weights at fractional coord u in [0,1). Returns (4,)."""
    inv6 = 1.0 / 6.0
    u2 = u * u
    t = 1.0 - u
    return np.array([inv6 * t * t * t,
                     inv6 * (3.0 * u2 * (u - 2.0) + 4.0),
                     inv6 * (3.0 * u * (1.0 + u - u2) + 1.0),
                     inv6 * u2 * u], dtype=np.float64)


def _dbasis(u):
    """Derivative of cardinal cubic B-spline w.r.t. u. Returns (4,)."""
    u2 = u * u
    t = 1.0 - u
    return np.array([-0.5 * t * t,
                     0.5 * (3.0 * u2 - 4.0 * u),
                     0.5 * (-3.0 * u2 + 2.0 * u + 1.0),
                     0.5 * u2], dtype=np.float64)


def _bspline_thomas_factors(n):
    """Thomas LU factors for the zero-padded cubic B-spline tridiagonal [1,4,1].

    Solves c_{i-1}+4c_i+c_{i+1} = 6 d_i (equivalent to _bspline_tridiag_ab with
    RHS scaled by 6). Returns (invden, cprime) float32 arrays consumed by the
    cs_bspline_prefilter_lines kernel: invden[i] = 1/(4 - c'[i-1]).
    """
    invden = np.empty(n, np.float64)
    cprime = np.empty(n, np.float64)
    invden[0] = 0.25
    cprime[0] = 0.25 if n > 1 else 0.0
    for i in range(1, n):
        invden[i] = 1.0 / (4.0 - cprime[i - 1])
        cprime[i] = invden[i] if i < n - 1 else 0.0
    return invden.astype(np.float32), cprime.astype(np.float32)


def _prefilter_3d(samples):
    """Separable cubic B-spline prefilter with batched solve_banded (no Python line loops)."""
    coeffs = np.asarray(samples, dtype=np.float64).copy()
    nx, ny, nz = coeffs.shape
    # Along z: reshape (nx*ny, nz) → solve with nz rows, nx*ny RHS
    if nz >= 3:
        flat = coeffs.reshape(nx * ny, nz).T  # (nz, nxy)
        coeffs = solve_banded((1, 1), _bspline_tridiag_ab(nz), flat).T.reshape(nx, ny, nz)
    # Along y: for each x, (ny, nz) like 2D — batch as (nx*nz, ny)
    if ny >= 3:
        # coeffs[ix, :, iz] → move y to last: (nx, nz, ny)
        tmp = np.transpose(coeffs, (0, 2, 1)).reshape(nx * nz, ny).T  # (ny, nx*nz)
        tmp = solve_banded((1, 1), _bspline_tridiag_ab(ny), tmp).T.reshape(nx, nz, ny)
        coeffs = np.transpose(tmp, (0, 2, 1))
    # Along x: (nx, ny*nz)
    if nx >= 3:
        flat = coeffs.reshape(nx, ny * nz)  # (nx, nyz) — solve_banded wants (n, nrhs) leading = nx
        coeffs = solve_banded((1, 1), _bspline_tridiag_ab(nx), flat).reshape(nx, ny, nz)
    return coeffs


# ── supersampled penalised least-squares fit (separable) ────────────────────
# Design: doc/Tasks/ContactPME_CoreMesh_Fit_Design.md. The mesh is fit to a
# grid s-times denser than the nodes (interior samples included), with a
# curvature penalty λ·D2 on the coefficients. Keeps the solve separable:
# (Ax⊗Ay⊗Az) c = (Bx⊗By⊗Bz)ᵀ R — three batched banded solves after a
# separable projection. At s=1, λ=0 it reduces to _prefilter_3d exactly
# (B is then the invertible [1,4,1]/6 tridiagonal, Bc=d ⇔ Mc=6d).

def _colloc_axis(n, s, sparse=False):
    """Cubic B-spline collocation matrix B (n_samp, n) on the s×-supersampled
    node grid: samples at x=k/s, k=0..s*(n-1). Out-of-range basis indices are
    dropped (zero-pad boundary, matching the _bspline_tridiag_ab convention)."""
    ns_ = s * (n - 1) + 1
    k = np.arange(ns_)
    x = k / float(s)
    i0 = np.floor(x).astype(int)
    u = x - i0
    u2 = u * u; t = 1.0 - u
    w = np.stack([t**3 / 6.0, (3.0 * u2 * (u - 2.0) + 4.0) / 6.0,
                  (3.0 * u * (1.0 + u - u2) + 1.0) / 6.0, u**3 / 6.0], axis=1)  # (ns_,4)
    cols = i0[:, None] + np.array([-1, 0, 1, 2])
    m = (cols >= 0) & (cols < n)
    rr = np.broadcast_to(np.arange(ns_)[:, None], cols.shape)
    if sparse:
        from scipy.sparse import coo_matrix
        return coo_matrix((w[m], (rr[m], cols[m])), shape=(ns_, n)).tocsr()
    B = np.zeros((ns_, n))
    B[rr[m], cols[m]] = w[m]
    return B


def _ab_banded(A):
    """Dense symmetric matrix (bandwidth ≤3) → ab array for solve_banded((3,3),...)."""
    n = A.shape[0]
    ab = np.zeros((7, n))
    for d in range(-3, 4):
        j = np.arange(max(0, -d), n - max(0, d))
        ab[3 + d, j] = A[j + d, j]
    return ab


def _apply_along(F, X, axis):
    """Dense (m, n_ax) matrix applied along `axis` of a tensor X."""
    X = np.moveaxis(X, axis, 0)
    sh = X.shape
    Y = (F @ X.reshape(sh[0], -1)).reshape(F.shape[0], *sh[1:])
    return np.moveaxis(Y, 0, axis)


def _solve_along(ab, X, axis):
    """Batched solve_banded((3,3)) along `axis` of a tensor X."""
    X = np.moveaxis(X, axis, 0)
    sh = X.shape
    Y = solve_banded((3, 3), ab, X.reshape(sh[0], -1)).reshape(sh)
    return np.moveaxis(Y, 0, axis)


def fit_mesh_lsq_3d(R_ss, s=2, lam=0.0):
    """Penalised separable cubic B-spline LSQ on an s×-supersampled node grid.

    R_ss : (s*(nx-1)+1, s*(ny-1)+1, s*(nz-1)+1) field samples on the regular
           supersampled grid (spacing h/s, same origin as the node grid).
    s    : supersampling factor (1 = node samples only → exact interp at λ=0).
    lam  : curvature penalty weight on coefficient 2nd differences (P-spline).
           λ=0 → plain L2 projection; λ>0 suppresses ringing on targets with
           sub-h features (such content stays in the residual, not aliased).

    Trailing batch axes are allowed (e.g. projecting a few core modes together).
    Returns (nx, ny, nz, ...) float64 B-spline control coefficients, same contract
    as _prefilter_3d output (eval: map_coordinates(c, g.T, order=3,
    prefilter=False)).
    """
    R_ss = np.asarray(R_ss, dtype=np.float64)
    ns = [(R_ss.shape[a] - 1) // s + 1 for a in range(3)]
    P = R_ss
    abs_ = []
    for a, n in enumerate(ns):
        B = _colloc_axis(n, s, sparse=True)
        A = (B.T @ B).toarray()
        if lam > 0.0 and n > 2:
            D2 = np.zeros((n - 2, n))
            r = np.arange(n - 2)
            D2[r, r] = 1.0; D2[r, r + 1] = -2.0; D2[r, r + 2] = 1.0
            A = A + lam * (D2.T @ D2)
        abs_.append(_ab_banded(A))
        P = _apply_along(B.T, P, a)          # separable projection B_aᵀ R
    for a in range(3):
        P = _solve_along(abs_[a], P, a)
    return P


def fit_coremesh_projected(E_ss, apos, origin, h, r_lo, r_b, s=2, weights=None, target=None, lam=1e-4, core_ridge=1e-8, max_workspace_bytes=256*1024**2, bPrint=False):
    """Fast overlapping fit: eliminate the mesh with the existing banded projection.

    P maps sampled residuals to mesh coefficients; B evaluates those coefficients.
    With mesh=P(T-Ac), the total field is BP(T)+(I-BP)Ac. Prefit cores first,
    then solve only the SMALL weighted system for their correction. The reduced
    objective is ||sqrt(w)((I-BP)Ac-(I-BP)T)||² + ridge; no CG or mesh unknowns.
    This closure approximates the fully weighted joint fit: the mesh projection
    is separable/unweighted, while the remaining error is relevance-weighted.
    Supersampling (s>=2) is essential: at nodal collocation the complement is zero.
    Five smootherstep-power modes per center are C2 individually (no lost DOFs).
    Bounds transient projection storage explicitly; large systems must be tiled
    before using this reference fitter. Returns the same triple as the CG fitter.
    """
    import time
    from spammm.surfaces.PICCore import fit_cores_from_samples, CoreFit, CORE_POWERS
    t0 = time.perf_counter()
    E = np.asarray(E_ss, np.float64)
    if E.ndim != 3 or not isinstance(s, (int, np.integer)) or s<2 or any((n-1)%s for n in E.shape) or h<=0 or lam<0 or core_ridge<=0:
        raise ValueError('projected coremesh requires a node-aligned 3D grid, integer s>=2, h>0, lam>=0, core_ridge>0')
    ns = tuple((n-1)//s+1 for n in E.shape)
    if min(ns)<3: raise ValueError('mesh needs >=3 nodes per axis')
    T = E if target is None else np.asarray(target, np.float64).reshape(E.shape)
    w = np.ones_like(E) if weights is None else np.asarray(weights, np.float64).reshape(E.shape)
    if not np.isfinite(E).all() or not np.isfinite(T).all() or not np.isfinite(w).all() or np.any(w<0) or not np.any(w>0):
        raise ValueError('finite target and nonnegative finite weights with positive mass required')
    apos = np.asarray(apos, np.float64).reshape(-1, 3)
    r_lo = np.broadcast_to(np.asarray(r_lo, np.float64), (len(apos),)).copy()
    r_b = np.broadcast_to(np.asarray(r_b, np.float64), (len(apos),)).copy()
    if not np.isfinite(apos).all() or not np.isfinite(r_lo).all() or not np.isfinite(r_b).all() or np.any(r_lo<=0) or np.any(r_b<=r_lo):
        raise ValueError('finite centers and radii 0<r_lo<r_b required')
    nc = len(apos)*len(CORE_POWERS)
    required = 8*(4*E.size*(nc+1)+2*np.prod(ns)*(nc+1)+nc*nc)
    if required>max_workspace_bytes:
        raise MemoryError(f'projected coremesh workspace bound {required} B exceeds {max_workspace_bytes} B; use a smaller fitting box')
    if bPrint: print(f'[projected coremesh] {E.size} samples, {nc} core unknowns; direct banded mesh solves', flush=True)
    if nc:
        pts = np.stack(np.meshgrid(*[origin[a]+np.arange(E.shape[a])*h/s for a in range(3)], indexing='ij'), -1).reshape(-1, 3)
        core = fit_cores_from_samples(apos, pts, T.ravel(), r_lo, r_b, weights=w.ravel(), tikhonov=core_ridge, smooth=True)
        A, _ = fit_cores_from_samples(apos, pts, T.ravel(), r_lo, r_b, return_design=True, smooth=True)
        del pts
        Ac = A.toarray()
        prefit = core.coeffs.copy()
    else:
        Ac = np.empty((E.size, 0)); prefit = np.empty((0, len(CORE_POWERS)))
        z = np.empty(0)
        core = CoreFit(prefit.copy(), z.copy(), z.copy(), CORE_POWERS.copy(), 'smooth', z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy())
    preparation_ms = 1e3*(time.perf_counter()-t0)
    # Project target and all small core columns in one batch of banded solves.
    packed = np.concatenate([T[..., None], Ac.reshape(*E.shape, nc)], axis=-1)
    projected = fit_mesh_lsq_3d(packed, s=s, lam=lam)
    values = projected
    for a in range(3): values = _apply_along(_colloc_axis(ns[a], s, sparse=True), values, a)
    R = T.ravel()-values[..., 0].ravel()
    Q = Ac-values[..., 1:].reshape(E.size, nc)
    if nc:
        # Start from core prefit and solve its correction, retaining full overlap.
        scale = np.sqrt(np.maximum(np.sum(w.ravel()[:, None]*Q*Q, axis=0), 1e-30))
        Q /= scale
        G = Q.T @ (w.ravel()[:, None]*Q)
        ridge = core_ridge*np.trace(G)/nc
        G += ridge*np.eye(nc)
        c0 = prefit.ravel()*scale
        dc = np.linalg.solve(G, Q.T @ (w.ravel()*(R-Q @ c0))-ridge*c0)
        c = (c0+dc)/scale
        core.coeffs = c.reshape(len(apos), -1)
        core.train_rmse_E[:] = np.sqrt(np.mean((Ac @ c-T.ravel())**2))
        coeffs = projected[..., 0]-np.einsum('ijkm,m->ijk', projected[..., 1:], c)
        residual = Q @ (c*scale)-R
        penalty = ridge*np.sum((c*scale)**2)
    else:
        coeffs = projected[..., 0]; residual = -R; penalty = 0.0
    if not np.isfinite(coeffs).all() or not np.isfinite(core.coeffs).all(): raise RuntimeError('nonfinite projected fit')
    data_J = float(np.sum(w.ravel()*residual**2))
    diagnostics = dict(iterations=0, data_J=data_J, penalty=float(penalty), objective=data_J+float(penalty), wrmse=np.sqrt(data_J/w.sum()), core_prefit_coeffs=prefit, preparation_ms=preparation_ms, solve_ms=1e3*(time.perf_counter()-t0)-preparation_ms, total_ms=1e3*(time.perf_counter()-t0), workspace_bound_bytes=int(required), solver='projected')
    if bPrint: print(f'[projected coremesh] preparation={preparation_ms:.2f} ms solve={diagnostics["solve_ms"]:.2f} ms weighted rms={diagnostics["wrmse"]:.6g}', flush=True)
    mesh = CoarseMesh(coeffs, np.asarray(origin, np.float64).copy(), float(h), 2, (np.full(3, 2), np.asarray(ns)-3))
    return mesh, core, diagnostics


# ── one weighted objective for sampled-field cores + mesh ───────────────────

def fit_coremesh_lsq(E_ss, apos, origin, h, r_lo, r_b, s=2, weights=None, target=None, lam=1e-4, core_ridge=1e-8, tol=1e-8, maxiter=3000, bPrint=False):
    """Joint weighted LSQ: ||sqrt(w)(cores + mesh - target)||² + λ||D2 mesh||².

    Weights describe relevance of the ORIGINAL E_ss (caller-supplied), never a
    capped residual. target may be a bounded continuation; it must equal E_ss
    in the reachable region. λ penalizes actual axis-wise second differences,
    not a product of three penalized Gram matrices. No dense 3D design/Gram.
    Cores retain the production powers/ABI. Null-space constraints make value,
    slope and curvature continuous at BOTH radial joins (flat inner extension,
    zero outer extension). Returned coeffs are raw powers, ready for evaluation.
    Returns (CoarseMesh, CoreFit, diagnostics). Empty apos gives the same weighted
    spline-only baseline. Nonconvergence raises; it is never silently accepted.
    """
    from scipy.linalg import null_space
    from scipy.sparse import block_diag
    from scipy.sparse.linalg import LinearOperator, cg
    from spammm.surfaces.PICCore import CORE_POWERS, CoreFit, fit_cores_from_samples
    E = np.asarray(E_ss, np.float64)
    if E.ndim != 3 or not isinstance(s, (int, np.integer)) or s < 1 or any((n - 1) % s for n in E.shape) or h <= 0:
        raise ValueError('fit_coremesh_lsq requires a 3D node-aligned supersampled grid, integer s>=1 and h>0')
    ns = tuple((n - 1) // s + 1 for n in E.shape)
    if min(ns) < 3 or lam < 0 or core_ridge <= 0:
        raise ValueError('mesh axes need >=3 nodes; lam>=0 and core_ridge>0 are required')
    w = np.ones_like(E) if weights is None else np.asarray(weights, np.float64).reshape(E.shape)
    T = E if target is None else np.asarray(target, np.float64).reshape(E.shape)
    if not np.isfinite(E).all() or not np.isfinite(T).all() or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError('field/target/weights must be finite, weights nonnegative with positive mass')
    apos = np.asarray(apos, np.float64).reshape(-1, 3)
    na, nm = len(apos), len(CORE_POWERS)
    r_lo = np.broadcast_to(np.asarray(r_lo, np.float64), (na,)).copy()
    r_b = np.broadcast_to(np.asarray(r_b, np.float64), (na,)).copy()
    if not np.isfinite(apos).all() or not np.isfinite(r_lo).all() or not np.isfinite(r_b).all() or np.any(r_lo <= 0) or np.any(r_b <= r_lo):
        raise ValueError('core radii require 0<r_lo<r_b')
    B = [_colloc_axis(n, s, sparse=True) for n in ns]
    def mesh_apply(c, transpose=False, squared=False):
        out = c
        for a in range(3):
            b = B[a].multiply(B[a]) if squared else B[a]
            out = _apply_along(b.T if transpose else b, out, a)
        return out
    def roughness(c):
        out = np.zeros_like(c)
        for a in range(3):
            v = np.moveaxis(out, a, 0)
            d = np.diff(c, n=2, axis=a)
            d = np.moveaxis(d, a, 0)
            v[:-2] += d; v[1:-1] -= 2*d; v[2:] += d
        return out
    # Only C2 combinations of the existing t^p modes: p=2 coefficient is zero
    # at r_b; Σp*c = Σp(p-1)*c = 0 at r_lo. Orthonormal reduced columns.
    p = CORE_POWERS.astype(float)
    Z = null_space(np.stack([p, p*(p-1), (p == 2).astype(float)]))
    if na:
        grids = np.meshgrid(*[origin[a] + np.arange(E.shape[a])*h/s for a in range(3)], indexing='ij')
        pts = np.stack(grids, -1).reshape(-1, 3)
        A, _ = fit_cores_from_samples(apos, pts, T.ravel(), r_lo, r_b, return_design=True)
        del pts, grids
        transform = block_diag([Z]*na, format='csr')
        A = (A @ transform).tocsr()
        # Whiten each small radial block; no all-center dense normal matrix.
        transforms = []
        for i in range(na):
            ai = A[:, i*Z.shape[1]:(i+1)*Z.shape[1]]
            G = (ai.T @ ai.multiply(w.ravel()[:, None])).toarray()
            ev, U = np.linalg.eigh(G)
            transforms.append(U @ np.diag(1/np.sqrt(np.maximum(ev, max(ev.max(), 1)*1e-10))) @ U.T)
        white = block_diag(transforms, format='csr')
        transform = transform @ white
        A = (A @ white).tocsr()
    else:
        from scipy.sparse import csr_matrix
        A = csr_matrix((E.size, 0))
    nc = A.shape[1]; ng = int(np.prod(ns)); size = nc + ng
    def normal(v):
        c = v[nc:].reshape(ns)
        residual = w*(mesh_apply(c) + (A @ v[:nc]).reshape(E.shape))
        return np.r_[A.T @ residual.ravel() + core_ridge*v[:nc], (mesh_apply(residual, transpose=True) + lam*roughness(c)).ravel()]
    rhs = np.r_[A.T @ (w*T).ravel(), mesh_apply(w*T, transpose=True).ravel()]
    diag_core = np.asarray(A.multiply(A).T @ w.ravel()).ravel() + core_ridge
    diag_pen = np.zeros(ns)
    for a, n in enumerate(ns):
        d = np.zeros(n); d[:-2] += 1; d[1:-1] += 4; d[2:] += 1
        sh = [1, 1, 1]; sh[a] = n; diag_pen += d.reshape(sh)
    diag_mesh = mesh_apply(w, transpose=True, squared=True) + lam*diag_pen
    diag = np.r_[diag_core, diag_mesh.ravel()]
    if np.any(diag <= 0):
        raise ValueError('unconstrained mesh coefficients: supply positive continuation weights or lam>0')
    operator = LinearOperator((size, size), matvec=normal, dtype=np.float64)
    precond = LinearOperator((size, size), matvec=lambda v: v/diag, dtype=np.float64)
    initial = np.zeros(size)
    if nc:
        # Cores first: the mesh never sees the original sharp field as its first
        # target. This prefit is an INITIAL GUESS, not a spatial partition or a
        # frozen split; the joint solve then lets both overlapping bases adjust.
        core_operator = LinearOperator((nc, nc), matvec=lambda v: A.T @ (w.ravel()*(A @ v)) + core_ridge*v, dtype=np.float64)
        core_precond = LinearOperator((nc, nc), matvec=lambda v: v/diag_core, dtype=np.float64)
        initial[:nc], core_info = cg(core_operator, rhs[:nc], M=core_precond, rtol=tol, atol=0.0, maxiter=maxiter)
        if core_info != 0:
            raise RuntimeError(f'core prefit did not converge: info={core_info}')
        if bPrint:
            print(f'[joint coremesh] cores-first weighted E rms={np.sqrt(np.sum(w.ravel()*(A @ initial[:nc]-T.ravel())**2)/w.sum()):.6g}', flush=True)
    count = [0]
    def progress(v):
        count[0] += 1
        if bPrint and count[0] % 100 == 0:
            print(f'[joint coremesh] CG iteration {count[0]}', flush=True)
    if bPrint:
        print(f'[joint coremesh] {E.size} samples, {ng} mesh + {nc} core unknowns, lam={lam:g}', flush=True)
    sol, info = cg(operator, rhs, x0=initial, M=precond, rtol=tol, atol=0.0, maxiter=maxiter, callback=progress)
    if info != 0 or not np.isfinite(sol).all():
        raise RuntimeError(f'joint coremesh CG did not converge: info={info}, iterations={count[0]}')
    c = sol[nc:].reshape(ns)
    coeffs = np.asarray(transform @ sol[:nc]).reshape(na, nm) if na else np.empty((0, nm))
    residual = mesh_apply(c) + (A @ sol[:nc]).reshape(E.shape) - T
    data_J = float(np.sum(w*residual**2))
    penalty = float(lam*sum(np.sum(np.diff(c, n=2, axis=a)**2) for a in range(3)) + core_ridge*np.dot(sol[:nc], sol[:nc]))
    diagnostics = dict(iterations=count[0], data_J=data_J, penalty=penalty, objective=data_J+penalty, wrmse=np.sqrt(data_J/w.sum()), normal_residual=np.linalg.norm(normal(sol)-rhs)/max(np.linalg.norm(rhs), 1e-30), core_prefit_coeffs=np.asarray(transform @ initial[:nc]).reshape(na, nm) if na else np.empty((0, nm)))
    if bPrint:
        print(f'[joint coremesh] iterations={count[0]} J={data_J+penalty:.6g} weighted E rms={diagnostics["wrmse"]:.6g}', flush=True)
    z = np.full(na, np.nan)
    core = CoreFit(coeffs, r_lo, r_b, CORE_POWERS.copy(), 'raw', z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy())
    mesh = CoarseMesh(c, np.asarray(origin, np.float64).copy(), float(h), 2, (np.full(3, 2), np.asarray(ns)-3))
    return mesh, core, diagnostics


# ── build ───────────────────────────────────────────────────────────────────

def build_coarse_mesh(atom_pos, split_params, query_bounds, h_mesh=1.0, halo_nodes=6):
    """Build a coarse 3D mesh of V_L = Σ_i v_i^L(|r - R_i|).

    atom_pos: (na, 3) float64
    split_params: SplitParams (na atoms)
    query_bounds: (3, 2) array — [[xmin,xmax],[ymin,ymax],[zmin,zmax]] of the
                  query envelope (where the tip will probe). The mesh domain is
                  this envelope padded by `halo_nodes` mesh nodes on every side.
    h_mesh: mesh spacing [Å]
    halo_nodes: halo padding on every side (contract: >= 6)

    Returns CoarseMesh with control coefficients (prefiltered), origin, and
    the safe interior node-index range.
    """
    atom_pos = np.asarray(atom_pos, dtype=np.float64).reshape(-1, 3)
    qb = np.asarray(query_bounds, dtype=np.float64)
    h = float(h_mesh)
    halo = int(halo_nodes)

    # Domain: query envelope + halo on every side
    lo = qb[:, 0] - halo * h
    hi = qb[:, 1] + halo * h
    nx = int(np.round((hi[0] - lo[0]) / h)) + 1
    ny = int(np.round((hi[1] - lo[1]) / h)) + 1
    nz = int(np.round((hi[2] - lo[2]) / h)) + 1
    origin = lo.copy()

    xs = origin[0] + np.arange(nx) * h
    ys = origin[1] + np.arange(ny) * h
    zs = origin[2] + np.arange(nz) * h
    samples = np.zeros((nx, ny, nz), dtype=np.float64)
    na = len(atom_pos)

    # One soft-poly cache + one full-volume soft_core_split per atom (not nx×na).
    # Coarse meshes are tiny (~10⁴ nodes); full (nx,ny,nz) per atom is fine.
    for ia in range(na):
        pi = split_params.with_atom(ia)
        cache = precompute_split_cache(pi)
        dx = xs[:, None, None] - atom_pos[ia, 0]
        dy = ys[None, :, None] - atom_pos[ia, 1]
        dz = zs[None, None, :] - atom_pos[ia, 2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        samples += soft_core_split(r, pi, cache=cache)['v_L']

    coeffs = _prefilter_3d(samples)

    # Safe interior: queries whose full 4×4×4 stencil stays inside [0, n-1]
    interior_lo = np.array([halo, halo, halo], dtype=np.int64)
    interior_hi = np.array([nx - 1 - halo, ny - 1 - halo, nz - 1 - halo], dtype=np.int64)

    return CoarseMesh(coeffs=coeffs, origin=origin, h=h, halo=halo,
                      query_interior=(interior_lo, interior_hi))


# ── evaluation ──────────────────────────────────────────────────────────────

def eval_mesh(mesh: CoarseMesh, queries, vectorized=True):
    """Evaluate V_L and F = -∇V_L at query points via cubic B-spline interpolation.

    queries: (nq, 3) float64 world coordinates.
    Returns (E, F) with E shape (nq,) and F shape (nq, 3).

    Guard: rejects any query whose full 4×4×4 stencil leaves the coefficient domain.
    """
    q = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    nq = len(q)
    c = mesh.coeffs
    nx, ny, nz = c.shape
    h = mesh.h
    inv_h = 1.0 / h
    origin = mesh.origin

    if vectorized:
        # Bounded 64-tap batches; retain scalar path below as the parity oracle.
        E = np.empty(nq); F = np.empty((nq, 3))
        for start in range(0, nq, 8192):
            stop = min(start + 8192, nq)
            g = (q[start:stop] - origin)*inv_h
            i = np.floor(g).astype(np.int64); u = g-i
            bad = np.any((i < 1) | (i + 2 >= np.asarray(c.shape)), axis=1)
            if bad.any():
                iq = start + int(np.flatnonzero(bad)[0])
                raise ValueError(f'Query {iq} at {q[iq]} stencil out of bounds for mesh {c.shape}')
            ix = i[:, :, None] + np.arange(-1, 3)
            v = c[ix[:, 0, :, None, None], ix[:, 1, None, :, None], ix[:, 2, None, None, :]]
            b = [_basis(u[:, a]).T for a in range(3)]
            db = [_dbasis(u[:, a]).T*inv_h for a in range(3)]
            E[start:stop] = np.einsum('nijk,ni,nj,nk->n', v, *b, optimize=True)
            for a in range(3):
                bb = b.copy(); bb[a] = db[a]
                F[start:stop, a] = -np.einsum('nijk,ni,nj,nk->n', v, *bb, optimize=True)
        return E, F

    E = np.zeros(nq, dtype=np.float64)
    F = np.zeros((nq, 3), dtype=np.float64)

    for iq in range(nq):
        # Fractional grid coordinate
        fx = (q[iq, 0] - origin[0]) * inv_h
        fy = (q[iq, 1] - origin[1]) * inv_h
        fz = (q[iq, 2] - origin[2]) * inv_h
        ix = int(np.floor(fx)); iy = int(np.floor(fy)); iz = int(np.floor(fz))
        ux = fx - ix; uy = fy - iy; uz = fz - iz
        # Stencil base (matches gridFF.cl: ix-1 .. ix+2)
        i0x = ix - 1; i0y = iy - 1; i0z = iz - 1
        # Guard: full 4×4×4 stencil must stay in [0, n-1]
        if i0x < 0 or i0y < 0 or i0z < 0 or i0x + 3 >= nx or i0y + 3 >= ny or i0z + 3 >= nz:
            raise ValueError(f"Query {iq} at {q[iq]} stencil out of bounds: "
                             f"i0=({i0x},{i0y},{i0z}) n=({nx},{ny},{nz})")
        bx = _basis(ux); by = _basis(uy); bz = _basis(uz)
        dbx = _dbasis(ux) * inv_h; dby = _dbasis(uy) * inv_h; dbz = _dbasis(uz) * inv_h
        # 64-tap stencil: E = Σ bx*by*bz * c
        # F = -∇E: dE/dx = Σ dbx*by*bz * c, etc.
        e = 0.0; fx_ = 0.0; fy_ = 0.0; fz_ = 0.0
        for a in range(4):
            cx = bx[a]; dx_ = dbx[a]; ia = i0x + a
            for b in range(4):
                cy = by[b]; dy_ = dby[b]; ib = i0y + b
                cxy = cx * cy; dxy = dx_ * cy; dxy2 = cx * dy_
                for cc in range(4):
                    cz = bz[cc]; dz_ = dbz[cc]; ic = i0z + cc
                    v = c[ia, ib, ic]
                    w = cxy * cz
                    e += w * v
                    fx_ += dxy * cz * v
                    fy_ += dxy2 * cz * v
                    fz_ += cxy * dz_ * v
        E[iq] = e
        F[iq, 0] = -fx_
        F[iq, 1] = -fy_
        F[iq, 2] = -fz_
    return E, F


def eval_mesh_direct(queries, atom_pos, split_params):
    """Direct Σ v_i^L oracle for the mesh part (no interpolation). Returns (E, F)."""
    q = np.asarray(queries, dtype=np.float64).reshape(-1, 3)
    atom_pos = np.asarray(atom_pos, dtype=np.float64).reshape(-1, 3)
    nq = len(q); na = len(atom_pos)
    E = np.zeros(nq, dtype=np.float64)
    F = np.zeros((nq, 3), dtype=np.float64)
    for iq in range(nq):
        for ia in range(na):
            dp = q[iq] - atom_pos[ia]
            r = float(np.linalg.norm(dp))
            pi = split_params.with_atom(ia)
            s = soft_core_split(np.array([r]), pi)
            E[iq] += float(s['v_L'][0])
            if r > 1e-30:
                F[iq] -= float(s['dv_L_dr'][0]) * dp / r
    return E, F
