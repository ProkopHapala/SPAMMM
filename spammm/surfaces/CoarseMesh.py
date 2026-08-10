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
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from spammm.surfaces.PMESplit import SplitParams, soft_core_split
from spammm.surfaces.ContactSurface import _bspline_prefilter_1d


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

    # Rasterize V_L = Σ_i v_i^L(|r - R_i|) by direct atom sum (vectorized over grid)
    # Build coordinate vectors
    xs = origin[0] + np.arange(nx) * h
    ys = origin[1] + np.arange(ny) * h
    zs = origin[2] + np.arange(nz) * h
    samples = np.zeros((nx, ny, nz), dtype=np.float64)

    R0_arr = np.atleast_1d(split_params.R0).astype(np.float64)
    E0_arr = np.atleast_1d(split_params.E0).astype(np.float64)
    q_arr = np.atleast_1d(split_params.q).astype(np.float64)
    r_cut = float(split_params.r_cut)
    na = len(atom_pos)

    # Slabbed over x to bound memory: each x-slab is (ny, nz)
    for ix in range(nx):
        dx_ia = xs[ix] - atom_pos[:, 0]  # (na,)
        for ia in range(na):
            ry = ys - atom_pos[ia, 1]  # (ny,)
            rz = zs - atom_pos[ia, 2]  # (nz,)
            r2 = dx_ia[ia] ** 2 + ry[:, None] ** 2 + rz[None, :] ** 2  # (ny, nz)
            r = np.sqrt(r2)
            pi = SplitParams(R0=np.array([R0_arr[ia]]), E0=np.array([E0_arr[ia]]),
                             q=np.array([q_arr[ia]]), alpha=split_params.alpha,
                             q_tip=split_params.q_tip, r_damp=split_params.r_damp, r_cut=r_cut)
            s = soft_core_split(r, pi)
            samples[ix] += s['v_L']

    # Prefilter: separable along x, y, z (REUSE _bspline_prefilter_1d)
    coeffs = samples.copy()
    # Along z (axis 2): batched over (nx, ny)
    for ix in range(nx):
        for iy in range(ny):
            coeffs[ix, iy, :] = _bspline_prefilter_1d(coeffs[ix, iy, :])
    # Along y (axis 1): batched over (nx, nz)
    for ix in range(nx):
        for iz in range(nz):
            coeffs[ix, :, iz] = _bspline_prefilter_1d(coeffs[ix, :, iz])
    # Along x (axis 0): batched over (ny, nz)
    for iy in range(ny):
        for iz in range(nz):
            coeffs[:, iy, iz] = _bspline_prefilter_1d(coeffs[:, iy, iz])

    # Safe interior: queries whose full 4×4×4 stencil stays inside [0, n-1]
    interior_lo = np.array([halo, halo, halo], dtype=np.int64)
    interior_hi = np.array([nx - 1 - halo, ny - 1 - halo, nz - 1 - halo], dtype=np.int64)

    return CoarseMesh(coeffs=coeffs, origin=origin, h=h, halo=halo,
                      query_interior=(interior_lo, interior_hi))


# ── evaluation ──────────────────────────────────────────────────────────────

def eval_mesh(mesh: CoarseMesh, queries):
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
    r_cut = float(split_params.r_cut)
    R0_arr = np.atleast_1d(split_params.R0).astype(np.float64)
    E0_arr = np.atleast_1d(split_params.E0).astype(np.float64)
    q_arr = np.atleast_1d(split_params.q).astype(np.float64)
    E = np.zeros(nq, dtype=np.float64)
    F = np.zeros((nq, 3), dtype=np.float64)
    for iq in range(nq):
        for ia in range(na):
            dp = q[iq] - atom_pos[ia]
            r = float(np.linalg.norm(dp))
            pi = SplitParams(R0=np.array([R0_arr[ia]]), E0=np.array([E0_arr[ia]]),
                             q=np.array([q_arr[ia]]), alpha=split_params.alpha,
                             q_tip=split_params.q_tip, r_damp=split_params.r_damp, r_cut=r_cut)
            s = soft_core_split(np.array([r]), pi)
            E[iq] += float(s['v_L'][0])
            if r > 1e-30:
                F[iq] -= float(s['dv_L_dr'][0]) * dp / r
    return E, F
