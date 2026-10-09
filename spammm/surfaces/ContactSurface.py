"""
ContactSurface.py — GPU quasi-2D contact field replacing 3D img_FF for static PP-AFM.

Motivation: dense Morse/LJ voxel grids scale as nx×ny×nz; molecules need compact
aperiodic fields evaluated every PP relaxation step. Two representations share the
same brute reference and AFMulator scan API: separable B-spline(xy)×poly(dz) and
radial atom-centric PIC.

Design:
- **Separable:** coeffs on B-spline xy grid × doubling poly modes in dz = z−h₀−poly_z0;
  matrix-free CG (cs_sep_Av/Atv); optional Boltzmann weights + Fx,Fy,Fz force rows.
- **PIC:** per-atom radial modes + particle-in-cell buckets; CG with Tikhonov reg≈1e-2.
- **h₀(x,y):** spherical contact envelope (ray vs Morse-R0 spheres) on B-spline nodes;
  legacy atom-center max-z still available via `h0_mode='atom_z'`. Chain rule in forces.
- **F = −∇E** throughout; F_ref GPU upload is planar [Fx…][Fy…][Fz…] not interleaved.

Open issues:
- USER visual pending on contact-sep vs GridFF XY (profiles improved). See
  doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md; Caveats §6.
- PIC: no force-loss rows yet; Boltzmann weights hurt close contact (fit unweighted).
- Tile CG (`fit_separable_tiles`) experimental; global CG preferred.
- PIC tiled eval caps local preload (`CS_PIC_LOCAL_MAX`).
- Zero `cs_AtAp_buff` before each Atv when reusing ContactSurfaceCL across fits.

AFMulator: fit_contact_surface / run_scan_contact (separable);
fit_pic_contact_surface / run_scan_pic (PIC).
Design: doc/Topics/AFM/ContactSurface_Static.md · parity report:
doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md · pitfalls: doc/Takeways.md
"""

import os
import time
import json
import numpy as np
import pyopencl as cl
from scipy.linalg import solve_banded

from spammm.utils.OpenCLBase import OpenCLBase
from spammm.utils import clUtils as clu
from spammm.topology.FFparams import load_xyz_with_REQs

COULOMB_CONST = 14.3996448915
_KERNEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kernels')

# Cubic B-spline prefilter: solves for control coefficients that exactly reproduce
# nodal values when interpolated with the cardinal cubic B-spline stencil used by
# cs_interp_h0 (kernel) / interp_h0 (Python), which ZERO-PADS out-of-bounds indices.
#
# The interpolation at node i is  d[i] = (1/6)c[i-1] + (4/6)c[i] + (1/6)c[i+1]
# with c[-1] = c[n] = 0. This is a symmetric tridiagonal system  A c = d  with
# diag = 4/6 and off-diags = 1/6. Solving it directly (batched tridiagonal solve)
# reproduces nodal values EXACTLY at every node — including the boundaries — which
# the previous causal/anti-causal IIR (Unser infinite-signal filter) could not: it
# gave ~1e-16 interior error but O(1e-2) error at the first node because the IIR
# inverts the infinite Toeplitz convolution, not the finite zero-padded system that
# the kernel actually evaluates. The tridiagonal solve is the mathematically exact
# inverse for the kernel's zero-padding boundary convention.
# Ref: Unser et al., "B-Spline Signal Processing" (pole z1 = -2+sqrt(3) ≈ -0.2679).
_BSPLINE_Z1 = -2.0 + np.sqrt(3.0)      # kept for reference / downstream use
_BSPLINE_SCALE = -6.0 * _BSPLINE_Z1    # kept for reference (IIR scale, unused by tridiagonal path)

def _bspline_tridiag_ab(n):
    """Banded representation of the n×n zero-padded cubic B-spline interpolation matrix.

    A[i,i]=4/6, A[i,i-1]=A[i,i+1]=1/6, with no wrap-around (zero-padding boundary).
    Returns ab in scipy.linalg.solve_banded((1,1), ab, b) layout: row0=super, row1=diag, row2=sub.
    """
    ab = np.zeros((3, n), dtype=np.float64)
    ab[0, 1:] = 1.0 / 6.0
    ab[1, :] = 4.0 / 6.0
    ab[2, :-1] = 1.0 / 6.0
    return ab

def _bspline_prefilter_1d(data):
    """Apply cubic B-spline prefilter to a 1D array (zero-padding boundary convention).

    Converts sampled/nodal values → B-spline control coefficients so that the cubic
    B-spline (with out-of-bounds stencil terms dropped, matching cs_interp_h0) exactly
    reproduces the input at every node, including the first and last.
    """
    n = len(data)
    if n < 3:
        return np.asarray(data, dtype=np.float64).copy()
    d = np.asarray(data, dtype=np.float64)
    return solve_banded((1, 1), _bspline_tridiag_ab(n), d)

def _bspline_prefilter_2d(h0_2d):
    """Apply separable cubic B-spline prefilter to a 2D array (zero-padding boundary).

    h0_2d: (rows, cols) array of nodal values → control coefficients. Solves the
    tridiagonal system along each axis in a single batched call (all rows / all cols
    at once). Exact at every node including boundaries.
    """
    out = np.array(h0_2d, dtype=np.float64)
    nrows, ncols = out.shape
    # Along axis 1 (cols): transpose so leading dim = ncols (multiple RHS = nrows)
    if ncols >= 3:
        out = solve_banded((1, 1), _bspline_tridiag_ab(ncols), out.T).T
    # Along axis 0 (rows): leading dim = nrows (multiple RHS = ncols)
    if nrows >= 3:
        out = solve_banded((1, 1), _bspline_tridiag_ab(nrows), out)
    return out


def _bspline4(t):
    """Cubic B-spline basis and derivative at parameter t in [0,1)."""
    t2 = t*t; t3 = t2*t; om = 1-t; om2 = om*om; om3 = om2*om
    B = np.array([om3/6, (3*t3-6*t2+4)/6, (-3*t3+3*t2+3*t+1)/6, t3/6])
    dB = np.array([-0.5*om2, 1.5*t2-2*t, -1.5*t2+t+0.5, 0.5*t2])
    return B, dB


def interp_h0(x, y, h0_flat, dx, x0, y0, ncx, ncy):
    """Python B-spline interpolation of h0(x,y) — matches kernel cs_interp_h0.

    Args:
        x, y: query position [Ang]
        h0_flat: (ncx*ncy,) B-spline control coefficients (prefiltered)
        dx: B-spline grid spacing [Ang]
        x0, y0: grid origin [Ang]
        ncx, ncy: grid dimensions

    Returns:
        h0(x,y) interpolated value
    """
    ux = (x - x0) / dx; uy = (y - y0) / dx  # dx=dy for contact surface
    ix = int(np.floor(ux)); tx = ux - ix
    iy = int(np.floor(uy)); ty = uy - iy
    Bx, _ = _bspline4(tx); By, _ = _bspline4(ty)
    z0 = 0.0
    for j in range(4):
        jy = iy - 1 + j
        if jy < 0 or jy >= ncy: continue
        for ii in range(4):
            ixk = ix - 1 + ii
            if ixk < 0 or ixk >= ncx: continue
            z0 += Bx[ii] * By[j] * h0_flat[jy * ncx + ixk]
    return z0


def select_contact_atoms(atom_pos, z_slab=5.0, z_quantile=None, xy_radius=12.0, z_local=1.0):
    """Keep atoms near local contact height (top of each molecular patch)."""
    apos = np.asarray(atom_pos, dtype=np.float64)
    z = apos[:, 2]
    if z_quantile is not None:
        mask = z >= float(np.quantile(z, z_quantile))
        return np.where(mask)[0]
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(apos[:, :2])
        nbrs = tree.query_ball_point(apos[:, :2], r=xy_radius)
        zloc = np.array([apos[idx, 2].max() if idx else z[i] for i, idx in enumerate(nbrs)], dtype=np.float64)
        mask = z >= (zloc - z_local)
        if mask.sum() < max(20, len(apos) // 20):
            mask = z >= (float(np.max(z)) - z_slab)
        return np.where(mask)[0]
    except Exception:
        zmax = float(np.max(z))
        return np.where(z >= (zmax - z_slab))[0]


def build_pic_buckets(atom_pos, x0, y0, x1, y1, cell_size):
    """Particle-in-cell bucket lists. cell_size should be > 2*Rc."""
    nx = max(1, int(np.ceil((x1 - x0) / cell_size)))
    ny = max(1, int(np.ceil((y1 - y0) / cell_size)))
    nb = nx * ny
    buckets = [[] for _ in range(nb)]
    for i, (x, y, _) in enumerate(atom_pos):
        bx = int((x - x0) / cell_size)
        by = int((y - y0) / cell_size)
        bx = min(max(bx, 0), nx - 1)
        by = min(max(by, 0), ny - 1)
        buckets[by * nx + bx].append(i)
    flat = []
    offsets = [0]
    for b in buckets:
        flat.extend(b)
        offsets.append(len(flat))
    return np.array(flat, dtype=np.int32), np.array(offsets, dtype=np.int32), nx, ny


def eval_sphere_contact_height(apos, Rs, x, y, z_fallback=None):
    """h₀(x,y) = max over spheres: z_i + sqrt(R_i²−ρ²) if ρ<R_i; else z_fallback or nan."""
    apos = np.asarray(apos, dtype=np.float64)
    Rs = np.asarray(Rs, dtype=np.float64).reshape(-1)
    best = None
    for i in range(len(apos)):
        R = float(Rs[i])
        if R <= 0.0:
            continue
        rho2 = (float(apos[i, 0]) - float(x)) ** 2 + (float(apos[i, 1]) - float(y)) ** 2
        if rho2 < R * R:
            h = float(apos[i, 2]) + float(np.sqrt(R * R - rho2))
            best = h if best is None else max(best, h)
    if best is not None:
        return best
    if z_fallback is not None:
        return float(z_fallback)
    return float('nan')


def build_contact_height_map(apos, x0, y0, dx, dy, ncx, ncy, r_xy=8.0, Rs=None):
    """Contact height h₀ on B-spline nodes.

    Modes:
      Rs is None  — legacy: max atom-center z within r_xy (NOT a true contact surface).
      Rs given    — spherical envelope (the intended contact surface):
                    h = max_i [ z_i + sqrt(R_i² − ρ_i²) ] for ρ_i < R_i,
                    else fallback to max nearby atom z.
                    R_i should be Morse R0 (= tip_R + R_vdW) so the apex is tip–atom contact.

    Returns a dict with two keys (h0_samples / h0_coeffs separation):
      'h0_samples': (ncx*ncy,) float32 — raw nodal heights (physical/contact values).
                    Use for min/max/extent queries (z_ref, h0_min, h0_max).
      'h0_coeffs':  (ncx*ncy,) float32 — B-spline control coefficients (prefiltered
                    with the zero-padding-boundary tridiagonal solve so that
                    cs_interp_h0 / interp_h0 exactly reproduces h0_samples at nodes).
                    Upload this to the OpenCL cs_h0 buffer; pass to SeparableParams.h0_map.
    """
    apos = np.asarray(apos, dtype=np.float64)
    zmin = float(np.min(apos[:, 2]))
    h0 = np.full(ncx * ncy, zmin, dtype=np.float32)
    if Rs is None:
        r2 = float(r_xy) ** 2
        for iy in range(ncy):
            cy = y0 + iy * dy
            for ix in range(ncx):
                cx = x0 + ix * dx
                d2 = (apos[:, 0] - cx) ** 2 + (apos[:, 1] - cy) ** 2
                mask = d2 < r2
                if np.any(mask):
                    h0[iy * ncx + ix] = float(np.max(apos[mask, 2]))
    else:
        Rs = np.asarray(Rs, dtype=np.float64).reshape(-1)
        assert len(Rs) == len(apos), f'Rs length {len(Rs)} != natoms {len(apos)}'
        Rmax = float(np.max(Rs)) if len(Rs) else 0.0
        r_search = max(float(r_xy), Rmax + 0.5)
        r2_search = r_search ** 2
        for iy in range(ncy):
            cy = y0 + iy * dy
            for ix in range(ncx):
                cx = x0 + ix * dx
                d2 = (apos[:, 0] - cx) ** 2 + (apos[:, 1] - cy) ** 2
                near = d2 < r2_search
                if not np.any(near):
                    continue
                best = zmin
                hit = False
                for ia in np.where(near)[0]:
                    rho2 = float(d2[ia])
                    R = float(Rs[ia])
                    if R <= 0.0:
                        continue
                    if rho2 < R * R:
                        h = float(apos[ia, 2]) + float(np.sqrt(R * R - rho2))
                        if h > best:
                            best = h
                        hit = True
                if hit:
                    h0[iy * ncx + ix] = best
                else:
                    h0[iy * ncx + ix] = float(np.max(apos[near, 2]))
    # h0_samples = raw nodal heights (physical values); h0_coeffs = prefiltered B-spline
    # control coefficients so cs_interp_h0 exactly reproduces h0_samples at every node
    # (including boundaries, via the zero-padding tridiagonal solve).
    h0_samples = h0.copy()
    h0_2d = h0.reshape(ncy, ncx)
    h0_coeffs = _bspline_prefilter_2d(h0_2d).astype(np.float32).reshape(-1)
    return {'h0_samples': h0_samples, 'h0_coeffs': h0_coeffs}


def bspline_n_intervals(length_ang, dx):
    """Number of B-spline knot intervals for cubic open end (+3)."""
    return max(1, int(np.ceil(length_ang / dx))) + 3


class SeparableParams:
    """B-spline(xy) × poly(z - h0(x,y) - poly_z0) with doubling powers t^(m_start*2^k)."""

    def __init__(self, x0, y0, dx, dy, ncx, ncy, poly_R=10.0, m_start=4, nz=5, poly_z0=0.0, h0_map=None, apos=None, h0_r_xy=8.0, Rs=None):
        self.x0 = float(x0); self.y0 = float(y0)
        self.dx = float(dx); self.dy = float(dy)
        self.ncx = int(ncx); self.ncy = int(ncy)
        self.poly_R = float(poly_R)
        self.poly_z0 = float(poly_z0)
        self.m_start = int(m_start)
        self.nz = int(nz)
        self.poly_powers = np.array([m_start * (2 ** k) for k in range(nz)], dtype=np.float32)
        self.h0_Rs = None if Rs is None else np.asarray(Rs, dtype=np.float64).reshape(-1)
        # h0_map accepts: dict from build_contact_height_map ({'h0_samples','h0_coeffs'}),
        # a raw flat array (legacy: treated as h0_coeffs), or None (built from apos).
        self.h0_samples = None
        if h0_map is not None:
            if isinstance(h0_map, dict):
                self.h0_map = np.ascontiguousarray(h0_map['h0_coeffs'], dtype=np.float32).reshape(-1)
                self.h0_samples = np.ascontiguousarray(h0_map['h0_samples'], dtype=np.float32).reshape(-1)
            else:
                self.h0_map = np.ascontiguousarray(h0_map, dtype=np.float32).reshape(-1)
        elif apos is not None:
            h0d = build_contact_height_map(apos, self.x0, self.y0, self.dx, self.dy, self.ncx, self.ncy, r_xy=h0_r_xy, Rs=self.h0_Rs)
            self.h0_map = np.ascontiguousarray(h0d['h0_coeffs'], dtype=np.float32).reshape(-1)
            self.h0_samples = np.ascontiguousarray(h0d['h0_samples'], dtype=np.float32).reshape(-1)
        else:
            self.h0_map = None
        self.coeffs = None

    @property
    def n_coeff(self):
        return self.ncx * self.ncy * self.nz

    @property
    def resident_bytes(self):
        """GPU-resident bytes: float32 coeffs + h0_map only (no fit/sample arrays)."""
        if self.coeffs is None or self.h0_map is None:
            raise RuntimeError('SeparableParams not fitted (coeffs/h0_map missing)')
        return int(4 * (self.n_coeff + self.ncx * self.ncy))

    def _validate_npz_state(self):
        """Fail-loud consistency checks shared by save_npz (pre-write) and load_npz."""
        for name in ('x0', 'y0', 'dx', 'dy', 'poly_R', 'poly_z0'):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f'SeparableParams {name}={getattr(self, name)} not finite')
        if not (self.dx > 0.0 and self.dy > 0.0 and self.poly_R > 0.0):
            raise ValueError(f'dx/dy/poly_R must be positive, got {self.dx}/{self.dy}/{self.poly_R}')
        if self.ncx <= 0 or self.ncy <= 0:
            raise ValueError(f'ncx/ncy must be positive, got {self.ncx}x{self.ncy}')
        if not 1 <= self.nz <= 8:
            raise ValueError(f'nz must be in 1..8 (kernel phi[8]), got {self.nz}')
        if self.m_start <= 0:
            raise ValueError(f'm_start must be positive, got {self.m_start}')
        if self.h0_map is None or np.asarray(self.h0_map).size != self.ncx * self.ncy or not np.isfinite(self.h0_map).all():
            raise ValueError('h0_map must be finite with ncx*ncy entries')
        if self.h0_samples is not None and (np.asarray(self.h0_samples).size != self.ncx * self.ncy or not np.isfinite(self.h0_samples).all()):
            raise ValueError('h0_samples must be finite with ncx*ncy entries')
        if self.coeffs is None or np.asarray(self.coeffs).size != self.n_coeff or not np.isfinite(self.coeffs).all():
            raise ValueError(f'coeffs must be finite with n_coeff={self.n_coeff} entries')
        fb = getattr(self, 'fit_bounds', None)
        if fb is not None:
            fb = np.asarray(fb, dtype=np.float64)
            if fb.shape != (2, 3) or not np.isfinite(fb).all() or np.any(fb[0] > fb[1]):
                raise ValueError('fit_bounds must be finite ordered (2,3)')
        fr = getattr(self, 'fit_rmse', None)
        if fr is not None and (not np.isfinite(fr) or fr < 0.0):
            raise ValueError(f'fit_rmse must be finite >= 0, got {fr}')

    def save_npz(self, path, *, metadata=None):
        """Serialize reconstruction + provenance data (never fit samples / __dict__). 'xb' refuses overwrite."""
        if self.coeffs is None or self.h0_map is None:
            raise RuntimeError('SeparableParams.save_npz requires fitted coeffs and h0_map')
        self._validate_npz_state()
        md = metadata if metadata is not None else getattr(self, 'metadata', None)
        if md is not None and not isinstance(md, dict):
            raise ValueError(f'metadata must be a dict, got {type(md).__name__}')
        arrs = {
            'schema_version': np.array(1, dtype=np.int64),
            'x0': np.float64(self.x0), 'y0': np.float64(self.y0),
            'dx': np.float64(self.dx), 'dy': np.float64(self.dy),
            'ncx': np.int64(self.ncx), 'ncy': np.int64(self.ncy),
            'nz': np.int64(self.nz), 'm_start': np.int64(self.m_start),
            'poly_R': np.float64(self.poly_R), 'poly_z0': np.float64(self.poly_z0),
            'coeffs': np.ascontiguousarray(self.coeffs, dtype=np.float32).reshape(-1),
            'h0_map': np.ascontiguousarray(self.h0_map, dtype=np.float32).reshape(-1),
            'metadata_json': np.array(json.dumps(md or {})),
        }
        if self.h0_samples is not None:
            arrs['h0_samples'] = np.ascontiguousarray(self.h0_samples, dtype=np.float32).reshape(-1)
        if getattr(self, 'fit_bounds', None) is not None:
            arrs['fit_bounds'] = np.ascontiguousarray(self.fit_bounds, dtype=np.float64)
        if getattr(self, 'fit_rmse', None) is not None:
            arrs['fit_rmse'] = np.float64(self.fit_rmse)
        with open(path, 'xb') as f:
            np.savez_compressed(f, **arrs)

    @classmethod
    def load_npz(cls, path):
        """Inverse of save_npz; schema_version must be 1, all fields validated via _validate_npz_state."""
        with np.load(path, allow_pickle=False) as z:
            ver = int(z['schema_version'].item())
            if ver != 1:
                raise ValueError(f'unsupported SeparableParams schema_version={ver}')
            sep = cls(float(z['x0']), float(z['y0']), float(z['dx']), float(z['dy']), int(z['ncx']), int(z['ncy']), poly_R=float(z['poly_R']), poly_z0=float(z['poly_z0']), m_start=int(z['m_start']), nz=int(z['nz']), h0_map=np.asarray(z['h0_map'], dtype=np.float32).reshape(-1))
            sep.coeffs = np.asarray(z['coeffs'], dtype=np.float32).reshape(-1).astype(np.float64)
            if 'h0_samples' in z:
                sep.h0_samples = np.asarray(z['h0_samples'], dtype=np.float32).reshape(-1)
            if 'fit_bounds' in z:
                sep.fit_bounds = np.asarray(z['fit_bounds'], dtype=np.float64)
            if 'fit_rmse' in z:
                sep.fit_rmse = float(z['fit_rmse'])
            md = json.loads(z['metadata_json'].item()) if 'metadata_json' in z else {}
            if not isinstance(md, dict):
                raise ValueError(f'metadata_json must decode to a dict, got {type(md).__name__}')
            sep.metadata = md
        sep._validate_npz_state()
        return sep


def separable_cl_args(sep: SeparableParams):
    """OpenCL int4/float4 metadata for separable eval/fit kernels."""
    meta = np.array([sep.ncx, sep.ncy, sep.nz, 0], dtype=np.int32)
    origin_step = np.array([sep.x0, sep.y0, sep.poly_z0, sep.dx], dtype=np.float32)
    dy_rc = np.array([sep.dy, sep.poly_R], dtype=np.float32)
    invRc_mstart = np.array([1.0 / sep.poly_R, float(sep.m_start)], dtype=np.float32)
    return meta, origin_step, dy_rc, invRc_mstart


class PICParams:
    """Radial PIC metadata (coeffs + buckets on GPU)."""

    def __init__(self, atom_pos, atom_indices, poly_R=10.0, m_start=4, nz=4, cell_size=10.0, bounds=None):
        self.atom_pos_full = np.asarray(atom_pos, dtype=np.float64)
        self.indices = np.asarray(atom_indices, dtype=np.int32)
        self.atom_pos = self.atom_pos_full[self.indices]
        self.nat = len(self.indices)
        self.poly_R = float(poly_R)
        self.m_start = int(m_start)
        self.nmodes = int(nz)
        self.poly_powers = np.array([m_start * (2 ** k) for k in range(nz)], dtype=np.float32)
        if bounds is None:
            x0, y0 = self.atom_pos[:, 0].min() - poly_R, self.atom_pos[:, 1].min() - poly_R
            x1, y1 = self.atom_pos[:, 0].max() + poly_R, self.atom_pos[:, 1].max() + poly_R
        else:
            x0, y0, x1, y1 = bounds
        self.bounds = (x0, y0, x1, y1)
        self.cell_size = float(cell_size)
        flat, off, nbx, nby = build_pic_buckets(self.atom_pos, x0, y0, x1, y1, cell_size)
        self.bucket_atoms = flat
        self.bucket_offsets = off
        self.nbx, self.nby = nbx, nby
        self.coeffs = None


class ContactSurfaceCL(OpenCLBase):
    """GPU contact surface: brute reference, separable CG/tile fit, PIC (pure OpenCL)."""

    def __init__(self, nloc=64, ctx=None, queue=None, build_options=None, **kw):
        super().__init__(nloc=nloc, ctx=ctx, queue=queue, **kw)
        kernel_paths = [
            os.path.join(_KERNEL_DIR, 'common.cl'),
            os.path.join(_KERNEL_DIR, 'Forces.cl'),
            os.path.join(_KERNEL_DIR, 'contact_surface.cl'),
        ]
        opts = ['-DDBG_UFF=0', '-D', 'AFM_STANDALONE=1'] + list(build_options or [])
        self.load_program_multi(kernel_paths, bPrint=False, build_options=opts)
        self._natoms = 0
        self._nq_max = 0
        self._ns_max = 0
        self._n_coeff_max = 0
        self._pic_nat = 0
        self._dot_partial = None
        self.alpha_morse = 1.8
        self.r_damp = 0.1
        self.plqh = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        self.nloc_atom = 32
        self.sep = None
        self.pic = None

    @staticmethod
    def _roundup(n, loc):
        return int((int(n) + int(loc) - 1) // int(loc) * int(loc))

    def _sep_cl_args(self, sep: SeparableParams):
        return separable_cl_args(sep)

    def setup_separable(self, sep: SeparableParams, apos=None):
        self.sep = sep
        if sep.h0_map is None:
            assert apos is not None, 'SeparableParams needs h0_map or apos for build_contact_height_map'
            h0d = build_contact_height_map(apos, sep.x0, sep.y0, sep.dx, sep.dy, sep.ncx, sep.ncy)
            sep.h0_map = np.ascontiguousarray(h0d['h0_coeffs'], dtype=np.float32).reshape(-1)
            sep.h0_samples = np.ascontiguousarray(h0d['h0_samples'], dtype=np.float32).reshape(-1)
        n_h0 = sep.ncx * sep.ncy
        self.try_make_buffers({'cs_h0': n_h0 * 4}, suffix='_buff')
        self.toGPU_(self.cs_h0_buff, np.ascontiguousarray(sep.h0_map, dtype=np.float32))
        self._ensure_coeffs(sep.n_coeff)
        self.prg.cs_zero(self.queue, (self._roundup(sep.n_coeff, self.nloc),), (self.nloc,), np.int32(sep.n_coeff), self.cs_coeffs_buff)

    def _pic_cl_meta(self, pic: PICParams):
        x0, y0, _, _ = pic.bounds
        meta = np.array([pic.nat, pic.nmodes, pic.nbx * pic.nby, pic.nbx], dtype=np.int32)
        bmeta = np.array([x0, y0, pic.cell_size, 1.0 / pic.poly_R], dtype=np.float32)
        return meta, bmeta

    def setup_atoms(self, apos, reqs, alpha_morse=1.8, r_damp=0.1, plqh=(1.0, 1.0, 1.0, 0.0)):
        apos = np.ascontiguousarray(apos, dtype=np.float32)
        reqs = np.ascontiguousarray(reqs, dtype=np.float32)
        self._natoms = len(apos)
        atoms4 = np.zeros((self._natoms, 4), dtype=np.float32)
        atoms4[:, :3] = apos
        self.try_make_buffers({'cs_atoms': self._natoms * 16, 'cs_reqs': self._natoms * 16}, suffix='_buff')
        self.toGPU_(self.cs_atoms_buff, atoms4)
        self.toGPU_(self.cs_reqs_buff, reqs)
        self.alpha_morse = float(alpha_morse)
        self.r_damp = float(r_damp)
        self.plqh = np.array(plqh, dtype=np.float32)

    def _ensure_queries(self, nq):
        nq = int(nq)
        if nq > self._nq_max:
            self._nq_max = self._roundup(nq, self.nloc)
            szq = self._nq_max * 3 * 4
            szfe = self._nq_max * 16
            self.try_make_buffers({'cs_queries': szq, 'cs_out_fe': szfe, 'cs_y': self._nq_max * 4, 'cs_Ap': self._nq_max * 4}, suffix='_buff')

    def _ensure_samples(self, ns):
        ns = int(ns)
        if ns > self._ns_max:
            self._ns_max = self._roundup(ns, self.nloc)
            self.try_make_buffers({'cs_samples': self._ns_max * 3 * 4, 'cs_Eref': self._ns_max * 4, 'cs_Fref': self._ns_max * 3 * 4, 'cs_Ap': self._ns_max * 4, 'cs_AFp': self._ns_max * 4, 'cs_sample_w': self._ns_max * 4}, suffix='_buff')

    def _ensure_coeffs(self, nc):
        nc = int(nc)
        if nc > self._n_coeff_max:
            self._n_coeff_max = self._roundup(nc, self.nloc)
            sz = self._n_coeff_max * 4
            self.try_make_buffers({'cs_coeffs': sz, 'cs_r': sz, 'cs_p': sz, 'cs_AtAp': sz, 'cs_Atb': sz}, suffix='_buff')

    def dot_gpu(self, buf_a, buf_b, n):
        wg = 64
        nG = clu.roundup_global_size(n, wg)
        ngrp = nG // wg
        if self._dot_partial is None or self._dot_partial.size < ngrp * 4:
            self._dot_partial = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=ngrp * 4)
        self.prg.dot_wg(self.queue, (nG,), (wg,), np.int32(n), buf_a, buf_b, self._dot_partial)
        partial = np.empty(ngrp, dtype=np.float32)
        cl.enqueue_copy(self.queue, partial, self._dot_partial)
        return float(partial.sum())

    def eval_brute(self, queries, finish=True):
        """GPU Morse+PLQH reference at query points."""
        queries = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, 3)
        nq = len(queries)
        self._ensure_queries(nq)
        self.toGPU_(self.cs_queries_buff, queries.reshape(-1))
        gff = np.array([self.r_damp, self.alpha_morse, 0.0, 0.0], dtype=np.float32)
        gs = (self._roundup(nq, self.nloc_atom),)
        self.prg.cs_brute_plqh_points(self.queue, gs, (self.nloc_atom,), np.int32(self._natoms), self.cs_atoms_buff, self.cs_reqs_buff, self.cs_queries_buff, self.cs_out_fe_buff, np.int32(nq), gff, self.plqh)
        if finish:
            self.queue.finish()
            out = np.zeros((nq, 4), dtype=np.float32)
            cl.enqueue_copy(self.queue, out, self.cs_out_fe_buff)
            return out[:, 3], out[:, :3]
        return None

    def upload_samples(self, xyz, E_ref, F_ref=None, sample_weights=None):
        xyz = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
        E_ref = np.ascontiguousarray(E_ref, dtype=np.float32).reshape(-1)
        ns = len(xyz)
        self._ensure_samples(ns)
        self.toGPU_(self.cs_samples_buff, xyz.reshape(-1))
        self.toGPU_(self.cs_Eref_buff, E_ref)
        self._use_force_loss = F_ref is not None
        if self._use_force_loss:
            F_ref = np.ascontiguousarray(F_ref, dtype=np.float32).reshape(-1, 3)
            assert len(F_ref) == ns
            # GPU layout: [Fx0..Fx_{ns-1}, Fy0.., Fz0..] for get_sub_region per component
            self.toGPU_(self.cs_Fref_buff, np.concatenate([F_ref[:, c] for c in range(3)]))
        self._use_sample_weights = (sample_weights is not None) or self._use_force_loss
        if self._use_sample_weights:
            w = np.ones(ns, dtype=np.float32) if sample_weights is None else np.ascontiguousarray(sample_weights, dtype=np.float32).reshape(-1)
            assert len(w) == ns
            self.toGPU_(self.cs_sample_w_buff, w)

    def _loss_row_weights(self, E_ref, F_ref, force_weight, force_equalize=True):
        """Per-row-type weights so E and Fx,Fy,Fz contribute comparably (unit-aware equal weight)."""
        if force_weight <= 0.0:
            return 1.0, np.zeros(3, dtype=np.float32)
        E_ref = np.asarray(E_ref, dtype=np.float64)
        F_ref = np.asarray(F_ref, dtype=np.float64)
        if not force_equalize:
            return 1.0, np.full(3, force_weight, dtype=np.float32)
        sE = max(float(np.sqrt(np.mean(E_ref * E_ref))), 1e-6)
        sF = np.array([max(float(np.sqrt(np.mean(F_ref[:, c] * F_ref[:, c]))), 1e-6) for c in range(3)], dtype=np.float64)
        return float(1.0 / (sE * sE)), (force_weight / (sF * sF)).astype(np.float32)

    def _sep_atv(self, nG, ns, vec_y_buf, out_buf, meta, origin_step, dy_rc, invRc_mstart, row_scale=1.0):
        if getattr(self, '_use_sample_weights', False):
            self.prg.cs_sep_Atv_w(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, vec_y_buf, out_buf, self.cs_h0_buff, self.cs_sample_w_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns), np.float32(row_scale))
        else:
            self.prg.cs_sep_Atv(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, vec_y_buf, out_buf, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns))

    def _fref_buf(self, ns, fcomp):
        return self.cs_Fref_buff.get_sub_region(int(fcomp) * ns * 4, ns * 4)

    def _sep_atv_f(self, nG, ns, fcomp, vec_y_buf, out_buf, meta, origin_step, dy_rc, invRc_mstart, force_weight):
        self.prg.cs_sep_Atv_f_w(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, vec_y_buf, out_buf, self.cs_h0_buff, self.cs_sample_w_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns), np.int32(fcomp), np.float32(force_weight))

    def _add_force_normal(self, nG, ns, nc, nGc, vec_y_buf, out_buf, meta, origin_step, dy_rc, invRc_mstart, force_weight, fcomp, coeffs_buf=None):
        cbuf = self.cs_p_buff if coeffs_buf is None else coeffs_buf
        self.prg.cs_sep_Av_f(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, cbuf, self.cs_AFp_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns), np.int32(fcomp))
        self._sep_atv_f(nG, ns, fcomp, self.cs_AFp_buff, out_buf, meta, origin_step, dy_rc, invRc_mstart, force_weight)

    def _cg_step_sep(self, ns, nc, meta, origin_step, dy_rc, invRc_mstart, masked_range=None, reg=0.0, force_weight=0.0, wE=1.0, wF=None):
        """One CG iteration on normal equations (matrix-free Av/Atv)."""
        wF = np.zeros(3, dtype=np.float32) if wF is None else wF
        nG = self._roundup(ns, self.nloc)
        nGc = self._roundup(nc, self.nloc)
        self.prg.cs_zero(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_AtAp_buff)
        self.prg.cs_sep_Av(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_p_buff, self.cs_Ap_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns))
        if masked_range is None:
            self._sep_atv(nG, ns, self.cs_Ap_buff, self.cs_AtAp_buff, meta, origin_step, dy_rc, invRc_mstart, row_scale=wE)
            if getattr(self, '_use_force_loss', False) and force_weight > 0.0:
                for fcomp in range(3):
                    self._add_force_normal(nG, ns, nc, nGc, None, self.cs_AtAp_buff, meta, origin_step, dy_rc, invRc_mstart, float(wF[fcomp]), fcomp)
        else:
            i0, i1 = masked_range
            self.prg.cs_sep_Atv_masked(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_Ap_buff, self.cs_AtAp_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns), np.int32(i0), np.int32(i1))
            self.prg.cs_zero_outside(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_p_buff, np.int32(i0), np.int32(i1))
            self.prg.cs_zero_outside(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_AtAp_buff, np.int32(i0), np.int32(i1))
        if reg > 0.0:
            self.prg.addMul(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_AtAp_buff, self.cs_p_buff, np.float32(reg))
        pAp = self.dot_gpu(self.cs_p_buff, self.cs_AtAp_buff, nc)
        rsold = self.dot_gpu(self.cs_r_buff, self.cs_r_buff, nc)
        alpha = rsold / (pAp + 1e-16)
        self.prg.addMul(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_coeffs_buff, self.cs_p_buff, np.float32(alpha))
        self.prg.addMul(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_r_buff, self.cs_AtAp_buff, np.float32(-alpha))
        rsnew = self.dot_gpu(self.cs_r_buff, self.cs_r_buff, nc)
        beta = rsnew / (rsold + 1e-16)
        self.prg.setLinear(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_p_buff, np.float32(1.0), self.cs_r_buff, np.float32(beta), self.cs_p_buff)
        if masked_range is not None:
            i0, i1 = masked_range
            self.prg.cs_zero_outside(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_p_buff, np.int32(i0), np.int32(i1))
        return np.sqrt(rsnew / max(nc, 1))

    def fit_separable_cg(self, sep: SeparableParams, xyz, E_ref, F_ref=None, apos=None, n_iter=80, tol=1e-5, sample_weights=None, force_weight=0.0, force_equalize=True, bPrint=False):
        """Global GPU CG fit. Optional WLS plus Fx,Fy,Fz rows; force_weight=1 equalizes RMS(E) vs RMS(Fα).

        DEPRECATED for batch compression — iterative CG (~2 min/mol). Prefer the
        direct cpm_params_from_samples() dense-mesh prefilter (~0.05 s/mol).
        """
        self.setup_separable(sep, apos=apos)
        self.upload_samples(xyz, E_ref, F_ref=F_ref if force_weight > 0.0 else None, sample_weights=sample_weights)
        wE, wF = self._loss_row_weights(E_ref, F_ref, force_weight, force_equalize=force_equalize)
        if bPrint and force_weight > 0.0:
            print(f'  fit loss row weights: wE={wE:.3e}  wFx={wF[0]:.3e}  wFy={wF[1]:.3e}  wFz={wF[2]:.3e}  (equalize={force_equalize})')
        ns = len(xyz)
        nc = sep.n_coeff
        meta, origin_step, dy_rc, invRc_mstart = self._sep_cl_args(sep)
        nGc = self._roundup(nc, self.nloc)
        nG = self._roundup(ns, self.nloc)
        self.prg.cs_zero(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_Atb_buff)
        self._sep_atv(nG, ns, self.cs_Eref_buff, self.cs_Atb_buff, meta, origin_step, dy_rc, invRc_mstart, row_scale=wE)
        if getattr(self, '_use_force_loss', False):
            for fcomp in range(3):
                self._sep_atv_f(nG, ns, fcomp, self._fref_buf(ns, fcomp), self.cs_Atb_buff, meta, origin_step, dy_rc, invRc_mstart, float(wF[fcomp]))
        self.prg.cs_copy(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_Atb_buff, self.cs_r_buff)
        self.prg.cs_sep_Av(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_coeffs_buff, self.cs_Ap_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns))
        self.prg.cs_zero(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_AtAp_buff)
        self._sep_atv(nG, ns, self.cs_Ap_buff, self.cs_AtAp_buff, meta, origin_step, dy_rc, invRc_mstart, row_scale=wE)
        if getattr(self, '_use_force_loss', False):
            for fcomp in range(3):
                self._add_force_normal(nG, ns, nc, nGc, None, self.cs_AtAp_buff, meta, origin_step, dy_rc, invRc_mstart, float(wF[fcomp]), fcomp, coeffs_buf=self.cs_coeffs_buff)
        self.prg.addMul(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_r_buff, self.cs_AtAp_buff, np.float32(-1.0))
        self.prg.cs_copy(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_r_buff, self.cs_p_buff)
        t0 = time.perf_counter()
        for it in range(n_iter):
            Ftot = self._cg_step_sep(ns, nc, meta, origin_step, dy_rc, invRc_mstart, force_weight=force_weight, wE=wE, wF=wF)
            if bPrint and (it % 20 == 0):
                print(f'  sep_CG[{it}] |F|={Ftot:.3e}')
            if Ftot < tol:
                break
        self.queue.finish()
        coeffs = np.zeros(nc, dtype=np.float32)
        cl.enqueue_copy(self.queue, coeffs, self.cs_coeffs_buff)
        sep.coeffs = coeffs.astype(np.float64)
        self.prg.cs_sep_Av(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_coeffs_buff, self.cs_Ap_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns))
        pred = np.zeros(ns, dtype=np.float32)
        cl.enqueue_copy(self.queue, pred, self.cs_Ap_buff)
        E_ref = np.asarray(E_ref, dtype=np.float32)
        err = pred - E_ref
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float64)
            wsum = max(float(w.sum()), 1e-16)
            rmse = float(np.sqrt(np.sum(w * err.astype(np.float64) ** 2) / wsum))
        else:
            rmse = float(np.sqrt(np.mean(err ** 2)))
        if bPrint:
            print(f'  fit_separable_cg: {it+1} iters, {time.perf_counter()-t0:.2f}s, RMSE={rmse:.4e}')
            if getattr(self, '_use_force_loss', False):
                w = np.asarray(sample_weights, dtype=np.float64) if sample_weights is not None else np.ones(ns, dtype=np.float64)
                wsum = max(float(w.sum()), 1e-16)
                F_ref = np.asarray(F_ref, dtype=np.float64)
                for fcomp, label in enumerate(('Fx', 'Fy', 'Fz')):
                    self.prg.cs_sep_Av_f(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_coeffs_buff, self.cs_AFp_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns), np.int32(fcomp))
                    pred_f = np.zeros(ns, dtype=np.float32)
                    cl.enqueue_copy(self.queue, pred_f, self.cs_AFp_buff)
                    f_err = pred_f.astype(np.float64) - F_ref[:, fcomp]
                    f_rmse = float(np.sqrt(np.sum(w * f_err * f_err) / wsum))
                    print(f'  fit_separable_cg: weighted {label}_RMSE={f_rmse:.4e}  force_weight={force_weight:.3g}')
        return rmse

    def _tile_coeff_ranges(self, sep: SeparableParams, tile_ang, x0f, x1f, y0f, y1f):
        """Coeff index ranges per xy tile (owned interior + 1-cell halo for neighbor coupling)."""
        dx, dy = sep.dx, sep.dy
        ntx = max(1, int(np.ceil((x1f - x0f) / tile_ang)))
        nty = max(1, int(np.ceil((y1f - y0f) / tile_ang)))
        tiles = []
        for ity in range(nty):
            for itx in range(ntx):
                tx0 = x0f + itx * tile_ang
                ty0 = y0f + ity * tile_ang
                tx1 = min(x1f, tx0 + tile_ang)
                ty1 = min(y1f, ty0 + tile_ang)
                ix0 = max(0, int(np.floor((tx0 - sep.x0) / dx)) - 3)
                iy0 = max(0, int(np.floor((ty0 - sep.y0) / dy)) - 3)
                ix1 = min(sep.ncx, int(np.ceil((tx1 - sep.x0) / dx)) + 4)
                iy1 = min(sep.ncy, int(np.ceil((ty1 - sep.y0) / dy)) + 4)
                ic_list = []
                for kz in range(sep.nz):
                    for iy in range(iy0, iy1):
                        for ix in range(ix0, ix1):
                            ic_list.append(ix + sep.ncx * (iy + sep.ncy * kz))
                if not ic_list:
                    continue
                ic_arr = np.array(ic_list, dtype=np.int32)
                tiles.append({'itx': itx, 'ity': ity, 'bbox': (tx0, ty0, tx1, ty1), 'ic0': int(ic_arr.min()), 'ic1': int(ic_arr.max()) + 1})
        return tiles

    def fit_separable_tiles(self, sep: SeparableParams, xyz, E_ref, apos=None, tile_ang=32.0, x0f=0.0, x1f=200.0, y0f=0.0, y1f=200.0, n_iter_per_tile=40, tol=1e-5, bPrint=False):
        """Independent tile CG: each tile updates only its coeff range (8-neighbor halo via full Av)."""
        self.setup_separable(sep, apos=apos)
        self.upload_samples(xyz, E_ref)
        ns = len(xyz)
        nc = sep.n_coeff
        meta, origin_step, dy_rc, invRc_mstart = self._sep_cl_args(sep)
        tiles = self._tile_coeff_ranges(sep, tile_ang, x0f, x1f, y0f, y1f)
        nGc = self._roundup(nc, self.nloc)
        nG = self._roundup(ns, self.nloc)
        t0 = time.perf_counter()
        for tile in tiles:
            i0, i1 = tile['ic0'], tile['ic1']
            self.prg.cs_zero(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_Atb_buff)
            self.prg.cs_sep_Atv(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_Eref_buff, self.cs_Atb_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns))
            self.prg.cs_copy(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_Atb_buff, self.cs_r_buff)
            self.prg.cs_sep_Av(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_coeffs_buff, self.cs_Ap_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns))
            self.prg.cs_zero(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_AtAp_buff)
            self.prg.cs_sep_Atv(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_Ap_buff, self.cs_AtAp_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns))
            self.prg.addMul(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_r_buff, self.cs_AtAp_buff, np.float32(-1.0))
            self.prg.cs_copy(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_r_buff, self.cs_p_buff)
            for _ in range(n_iter_per_tile):
                Ftot = self._cg_step_sep(ns, nc, meta, origin_step, dy_rc, invRc_mstart, masked_range=(i0, i1), reg=1e-2)
                if Ftot < tol:
                    break
        self.queue.finish()
        coeffs = np.zeros(nc, dtype=np.float32)
        cl.enqueue_copy(self.queue, coeffs, self.cs_coeffs_buff)
        sep.coeffs = coeffs.astype(np.float64)
        self.prg.cs_sep_Av(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_coeffs_buff, self.cs_Ap_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(ns))
        pred = np.zeros(ns, dtype=np.float32)
        cl.enqueue_copy(self.queue, pred, self.cs_Ap_buff)
        rmse = float(np.sqrt(np.mean((pred - np.asarray(E_ref, dtype=np.float32)) ** 2)))
        if bPrint:
            print(f'  fit_separable_tiles: {len(tiles)} tiles, {time.perf_counter()-t0:.2f}s, RMSE={rmse:.4e}')
        return rmse

    def eval_separable(self, queries, sep: SeparableParams = None, finish=True):
        sep = sep or self.sep
        assert sep is not None and sep.coeffs is not None
        queries = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, 3)
        nq = len(queries)
        self._ensure_queries(nq)
        coeffs = np.ascontiguousarray(sep.coeffs, dtype=np.float32)
        self.toGPU_(self.cs_queries_buff, queries.reshape(-1))
        if coeffs.size <= self._n_coeff_max:
            cl.enqueue_copy(self.queue, self.cs_coeffs_buff, coeffs)
        meta, origin_step, dy_rc, invRc_mstart = self._sep_cl_args(sep)
        gs = (self._roundup(nq, self.nloc),)
        self.prg.evalSeparableBsplinePoly(self.queue, gs, (self.nloc,), self.cs_queries_buff, self.cs_out_fe_buff, self.cs_coeffs_buff, self.cs_h0_buff, meta, origin_step, dy_rc, invRc_mstart, np.int32(nq))
        if finish:
            self.queue.finish()
            out = np.zeros((nq, 4), dtype=np.float32)
            cl.enqueue_copy(self.queue, out, self.cs_out_fe_buff)
            return out[:, 3], out[:, :3]
        return None

    def setup_pic(self, pic: PICParams):
        self.pic = pic
        nat, nm = pic.nat, pic.nmodes
        self._pic_nat = nat
        nc = nat * nm
        self._ensure_coeffs(nc)
        atoms4 = np.zeros((nat, 4), dtype=np.float32)
        atoms4[:, :3] = pic.atom_pos.astype(np.float32)
        flat = np.ascontiguousarray(pic.bucket_atoms, dtype=np.int32)
        off = np.ascontiguousarray(pic.bucket_offsets, dtype=np.int32)
        self.try_make_buffers({'cs_pic_atoms': nat * 16, 'cs_pic_buckets': flat.nbytes, 'cs_pic_offsets': off.nbytes}, suffix='_buff')
        self.toGPU_(self.cs_pic_atoms_buff, atoms4)
        self.toGPU_(self.cs_pic_buckets_buff, flat)
        self.toGPU_(self.cs_pic_offsets_buff, off)
        self.prg.cs_zero(self.queue, (self._roundup(nc, self.nloc),), (self.nloc,), np.int32(nc), self.cs_coeffs_buff)

    def _pic_atv(self, nG, ns, vec_y_buf, out_buf, meta, bmeta, pic, m_start):
        if getattr(self, '_use_sample_weights', False):
            self.prg.cs_pic_Atv_w(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, vec_y_buf, out_buf, self.cs_sample_w_buff, self.cs_pic_atoms_buff, self.cs_pic_buckets_buff, self.cs_pic_offsets_buff, meta, bmeta, np.float32(m_start), np.int32(ns))
        else:
            self.prg.cs_pic_Atv(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, vec_y_buf, out_buf, self.cs_pic_atoms_buff, self.cs_pic_buckets_buff, self.cs_pic_offsets_buff, meta, bmeta, np.float32(m_start), np.int32(ns))

    def fit_pic_cg(self, pic: PICParams, xyz, E_ref, n_iter=60, tol=1e-5, reg=1e-2, sample_weights=None, bPrint=False):
        """GPU CG fit for radial PIC coefficients (Tikhonov reg on diagonal)."""
        self.setup_pic(pic)
        self.upload_samples(xyz, E_ref, sample_weights=sample_weights)
        ns = len(xyz)
        nc = pic.nat * pic.nmodes
        meta, bmeta = self._pic_cl_meta(pic)
        nGc = self._roundup(nc, self.nloc)
        nG = self._roundup(ns, self.nloc)
        self.prg.cs_zero(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_Atb_buff)
        self._pic_atv(nG, ns, self.cs_Eref_buff, self.cs_Atb_buff, meta, bmeta, pic, pic.m_start)
        self.prg.cs_copy(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_Atb_buff, self.cs_r_buff)
        self.prg.cs_pic_Av(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_coeffs_buff, self.cs_Ap_buff, self.cs_pic_atoms_buff, self.cs_pic_buckets_buff, self.cs_pic_offsets_buff, meta, bmeta, np.float32(pic.m_start), np.int32(ns))
        self.prg.cs_zero(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_AtAp_buff)
        self._pic_atv(nG, ns, self.cs_Ap_buff, self.cs_AtAp_buff, meta, bmeta, pic, pic.m_start)
        self.prg.addMul(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_r_buff, self.cs_AtAp_buff, np.float32(-1.0))
        self.prg.cs_copy(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_r_buff, self.cs_p_buff)
        t0 = time.perf_counter()
        for it in range(n_iter):
            self.prg.cs_zero(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_AtAp_buff)
            self.prg.cs_pic_Av(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_p_buff, self.cs_Ap_buff, self.cs_pic_atoms_buff, self.cs_pic_buckets_buff, self.cs_pic_offsets_buff, meta, bmeta, np.float32(pic.m_start), np.int32(ns))
            self._pic_atv(nG, ns, self.cs_Ap_buff, self.cs_AtAp_buff, meta, bmeta, pic, pic.m_start)
            if reg > 0.0:
                self.prg.addMul(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_AtAp_buff, self.cs_p_buff, np.float32(reg))
            pAp = self.dot_gpu(self.cs_p_buff, self.cs_AtAp_buff, nc)
            rsold = self.dot_gpu(self.cs_r_buff, self.cs_r_buff, nc)
            alpha = rsold / (pAp + 1e-16)
            self.prg.addMul(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_coeffs_buff, self.cs_p_buff, np.float32(alpha))
            self.prg.addMul(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_r_buff, self.cs_AtAp_buff, np.float32(-alpha))
            rsnew = self.dot_gpu(self.cs_r_buff, self.cs_r_buff, nc)
            beta = rsnew / (rsold + 1e-16)
            self.prg.setLinear(self.queue, (nGc,), (self.nloc,), np.int32(nc), self.cs_p_buff, np.float32(1.0), self.cs_r_buff, np.float32(beta), self.cs_p_buff)
            Ftot = np.sqrt(rsnew / max(nc, 1))
            if bPrint and (it % 20 == 0):
                print(f'  pic_CG[{it}] |F|={Ftot:.3e}')
            if Ftot < tol:
                break
        self.queue.finish()
        coeffs = np.zeros(nc, dtype=np.float32)
        cl.enqueue_copy(self.queue, coeffs, self.cs_coeffs_buff)
        pic.coeffs = coeffs.reshape(pic.nat, pic.nmodes).astype(np.float64)
        cl.enqueue_copy(self.queue, self.cs_coeffs_buff, coeffs)
        self.prg.cs_pic_Av(self.queue, (nG,), (self.nloc,), self.cs_samples_buff, self.cs_coeffs_buff, self.cs_Ap_buff, self.cs_pic_atoms_buff, self.cs_pic_buckets_buff, self.cs_pic_offsets_buff, meta, bmeta, np.float32(pic.m_start), np.int32(ns))
        pred = np.zeros(ns, dtype=np.float32)
        cl.enqueue_copy(self.queue, pred, self.cs_Ap_buff)
        rmse = float(np.sqrt(np.mean((pred - np.asarray(E_ref, dtype=np.float32)) ** 2)))
        if bPrint:
            print(f'  fit_pic_cg: {it+1} iters, {time.perf_counter()-t0:.2f}s, RMSE={rmse:.4e}')
        return rmse

    def eval_pic_grid(self, x0, y0, z, dx, dy, nx, ny, pic: PICParams = None, finish=True):
        """16×16 tiled PIC eval on regular xy grid (pure OpenCL)."""
        pic = pic or self.pic
        assert pic is not None and pic.coeffs is not None
        nq = nx * ny
        self._ensure_queries(nq)
        queries = np.zeros((nq, 3), dtype=np.float32)
        queries[:, 2] = float(z)
        self.toGPU_(self.cs_queries_buff, queries.reshape(-1))
        coeffs = np.ascontiguousarray(pic.coeffs.reshape(-1), dtype=np.float32)
        cl.enqueue_copy(self.queue, self.cs_coeffs_buff, coeffs)
        meta, bmeta = self._pic_cl_meta(pic)
        grid_meta = np.array([nx, ny, nq, 0], dtype=np.int32)
        q_origin = np.array([x0, y0, dx, dy], dtype=np.float32)
        ntx = (nx + 15) // 16
        nty = (ny + 15) // 16
        gs = (ntx * 16, nty * 16)
        ls = (16, 16)
        self.prg.cs_pic_eval_tile16(self.queue, gs, ls, self.cs_queries_buff, self.cs_out_fe_buff, self.cs_pic_atoms_buff, self.cs_coeffs_buff, self.cs_pic_buckets_buff, self.cs_pic_offsets_buff, meta, bmeta, np.float32(pic.m_start), grid_meta, q_origin)
        if finish:
            self.queue.finish()
            out = np.zeros((nq, 4), dtype=np.float32)
            cl.enqueue_copy(self.queue, out, self.cs_out_fe_buff)
            return out[:, 3], out[:, :3]
        return None

    def eval_pic(self, queries, pic: PICParams = None, finish=True):
        """PIC eval at arbitrary query points (1 thread/query)."""
        pic = pic or self.pic
        assert pic is not None and pic.coeffs is not None
        queries = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, 3)
        nq = len(queries)
        self._ensure_queries(nq)
        self.toGPU_(self.cs_queries_buff, queries.reshape(-1))
        coeffs = np.ascontiguousarray(pic.coeffs.reshape(-1), dtype=np.float32)
        cl.enqueue_copy(self.queue, self.cs_coeffs_buff, coeffs)
        meta, bmeta = self._pic_cl_meta(pic)
        gs = (self._roundup(nq, self.nloc),)
        self.prg.evalRadialPIC(self.queue, gs, (self.nloc,), self.cs_queries_buff, self.cs_out_fe_buff, self.cs_pic_atoms_buff, self.cs_coeffs_buff, self.cs_pic_buckets_buff, self.cs_pic_offsets_buff, meta, bmeta, np.float32(pic.m_start), np.int32(nq))
        if finish:
            self.queue.finish()
            out = np.zeros((nq, 4), dtype=np.float32)
            cl.enqueue_copy(self.queue, out, self.cs_out_fe_buff)
            return out[:, 3], out[:, :3]
        return None

    # ── poly8sp matrix-free core fit (CGLS; design doc §16) ──────────────────
    # Program must be built with CS_PME_SP_CORE=1 (AFMulator._pme_build_opts for
    # pme_core_basis='poly8sp'). Operators: cs_sp_Av = [E; wF*grad] per sample,
    # cs_sp_Atv (sq=0 adjoint / sq=1 column norms) + cs_reduce_groups.
    # Sample-space vectors are float4-per-sample handled as flat 4*ns floats.

    def _sp_ensure(self, ns, ncoef, ngroups):
        n = dict(ns=ns, ncoef=ncoef, ng=ngroups)
        if getattr(self, '_sp_dims', None) != n:
            self.try_make_buffers({'cs_sp_samp': ns * 12, 'cs_sp_b': ns * 16, 'cs_sp_s': ns * 16, 'cs_sp_q': ns * 16,
                                   'cs_sp_z': ncoef * 4, 'cs_sp_g': ncoef * 4, 'cs_sp_p': ncoef * 4, 'cs_sp_D': ncoef * 4,
                                   'cs_sp_v': ncoef * 4, 'cs_sp_partial': ngroups * ncoef * 4}, suffix='_buff')
            self._sp_dims = n

    def _sp_Av(self, nG, ns, ncoef, nat, d_span, wF, v_buf, y_buf):
        lsz = self.nloc
        self.prg.cs_sp_Av(self.queue, (nG,), (lsz,), self.cs_sp_samp_buff, v_buf, y_buf, self.cs_sp_atoms_buff,
                          np.int32(nat), np.float32(d_span), np.float32(wF), np.int32(ns),
                          cl.LocalMemory(nat * 16), cl.LocalMemory(ncoef * 4))

    def _sp_Atv(self, ns, ncoef, nat, ngroups, SB, d_span, wF, sq, out_buf):
        lsz = self.nloc
        self.prg.cs_sp_Atv(self.queue, (ngroups * lsz,), (lsz,), self.cs_sp_samp_buff, self.cs_sp_s_buff, self.cs_sp_partial_buff,
                           self.cs_sp_atoms_buff, np.int32(nat), np.float32(d_span), np.float32(wF),
                           np.int32(ns), np.int32(SB), np.int32(sq),
                           cl.LocalMemory(nat * 16), cl.LocalMemory(ncoef * 4), cl.LocalMemory(lsz * 16), cl.LocalMemory(lsz * 16))
        nGc = self._roundup(ncoef, lsz)
        self.prg.cs_reduce_groups(self.queue, (nGc,), (lsz,), np.int32(ncoef), np.int32(ngroups), self.cs_sp_partial_buff, out_buf)

    def fit_core_sp_cg(self, centers, w, Rlad, pts, E_ref, F_ref, wF=0.3, n_iter=4000, tol=1e-6, lam=0.0, SB=1024, bPrint=True, active=None):
        """GPU matrix-free CGLS fit of the poly8sp core (design doc §16).

        Solves  min_c ||Â c - b̂||²  with Â = [E_rows; wF*grad_rows],
        b̂ = (-wF*F_ref, E_ref) one float4/sample, on the column-scaled unknown
        z (c = D z, D = 1/||â_j|| via the sq=1 column-norm pass). Never forms Â:
        each iteration is one cs_sp_Av + one cs_sp_Atv/reduce. `active` (ncoef
        bool, optional) freezes inactive columns at 0 via D=0 (e.g. bond centers
        restricted to slot0). Returns (coeffs (nc,NMODES) float64 in kernel slot
        layout, info dict). The fitted coefficient buffer stays resident for
        eval_core_sp_gpu()."""
        assert hasattr(self.prg, 'cs_sp_Av'), 'ContactSurfaceCL program lacks cs_sp_* kernels — build with CS_PME_SP_CORE=1 (poly8sp options)'
        centers = np.ascontiguousarray(centers, np.float64).reshape(-1, 3)
        w = np.ascontiguousarray(w, np.float64).reshape(-1)
        assert len(w) == len(centers)
        Rlad = np.asarray(Rlad, np.float64)
        nmod = 3 + 3 * (len(Rlad) - 3)
        nc = len(centers); ncoef = nc * nmod
        pts = np.ascontiguousarray(pts, np.float32).reshape(-1, 3)
        E_ref = np.ascontiguousarray(E_ref, np.float64).reshape(-1)
        assert len(pts) == len(E_ref)
        ns = len(pts)
        wF_eff = float(wF) if F_ref is not None else 0.0
        if F_ref is None:
            F_ref = np.zeros((ns, 3), np.float64)
        lsz = self.nloc
        loc = self.ctx.devices[0].get_info(cl.device_info.LOCAL_MEM_SIZE)
        need = nc * 16 + ncoef * 4 + 2 * lsz * 16          # Atv: LATOMS+LACC+LPOS+LV (max of both kernels)
        assert need <= loc, f'local memory {need} B > device {loc} B for nat={nc} NMODES={nmod}'
        SB = max(int(SB), lsz)
        ngroups = (ns + SB - 1) // SB
        self._sp_ensure(ns, ncoef, ngroups)
        atoms4 = np.zeros((nc, 4), np.float32); atoms4[:, :3] = centers; atoms4[:, 3] = w
        if getattr(self, 'cs_sp_atoms_buff', None) is None or self.cs_sp_atoms_buff.size != nc * 16:
            self.cs_sp_atoms_buff = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nc * 16)
        self.toGPU_(self.cs_sp_atoms_buff, atoms4)
        b4 = np.zeros((ns, 4), np.float32)
        b4[:, :3] = -wF_eff * F_ref; b4[:, 3] = E_ref
        self.toGPU_(self.cs_sp_samp_buff, pts.reshape(-1))
        self.toGPU_(self.cs_sp_b_buff, b4)
        d_span = np.float32(Rlad[0])
        t0 = time.perf_counter()
        nGs = self._roundup(ns, lsz)
        nGc = self._roundup(ncoef, lsz)
        ns4 = 4 * ns
        # residual s = b̂ - Â z, z = 0 -> s = b̂ ; g = D (Âᵀ s) ; p = g
        self.prg.cs_copy(self.queue, (self._roundup(ns4, lsz),), (lsz,), np.int32(ns4), self.cs_sp_b_buff, self.cs_sp_s_buff)
        # Jacobi column norms: D = 1/sqrt(colnorm) (sq=1 pass ignores v)
        self._sp_Atv(ns, ncoef, nc, ngroups, SB, d_span, wF_eff, 1, self.cs_sp_D_buff)
        colnorm = self.fromGPU_(self.cs_sp_D_buff, shape=(ncoef,))
        assert np.isfinite(colnorm).all(), 'non-finite column norm in sp design'
        if bPrint and (colnorm <= 0).any():
            print(f'  fit_core_sp_cg: {(colnorm <= 0).sum()}/{ncoef} zero-norm columns (unconstrained DOFs stay 0)', flush=True)
        D = 1.0 / np.sqrt(np.maximum(colnorm, 1e-30)).astype(np.float32)
        if active is not None:
            active = np.asarray(active, bool)
            assert active.shape == (ncoef,)
            D[~active] = 0.0                                     # frozen columns: g,p,z,c all stay 0
        self.toGPU_(self.cs_sp_D_buff, D)
        self._sp_Atv(ns, ncoef, nc, ngroups, SB, d_span, wF_eff, 0, self.cs_sp_g_buff)
        self.prg.cs_mul(self.queue, (nGc,), (lsz,), np.int32(ncoef), self.cs_sp_g_buff, self.cs_sp_D_buff, self.cs_sp_g_buff)
        self.prg.cs_zero(self.queue, (nGc,), (lsz,), np.int32(ncoef), self.cs_sp_z_buff)
        self.prg.cs_copy(self.queue, (nGc,), (lsz,), np.int32(ncoef), self.cs_sp_g_buff, self.cs_sp_p_buff)
        gg = self.dot_gpu(self.cs_sp_g_buff, self.cs_sp_g_buff, ncoef)
        g0 = np.sqrt(gg)
        lam2 = np.float32(lam * lam)
        hist = []
        it = 0
        for it in range(1, n_iter + 1):
            hist.append(np.sqrt(gg) / g0)
            if hist[-1] < tol:
                break
            self.prg.cs_mul(self.queue, (nGc,), (lsz,), np.int32(ncoef), self.cs_sp_p_buff, self.cs_sp_D_buff, self.cs_sp_v_buff)   # v = D p
            self._sp_Av(nGs, ns, ncoef, nc, d_span, wF_eff, self.cs_sp_v_buff, self.cs_sp_q_buff)                                    # q = Â D p
            qq = self.dot_gpu(self.cs_sp_q_buff, self.cs_sp_q_buff, ns4)
            if lam > 0:
                qq += float(lam2) * self.dot_gpu(self.cs_sp_p_buff, self.cs_sp_p_buff, ncoef)
            alpha = gg / (qq + 1e-30)
            self.prg.addMul(self.queue, (nGc,), (lsz,), np.int32(ncoef), self.cs_sp_z_buff, self.cs_sp_p_buff, np.float32(alpha))
            self.prg.addMul(self.queue, (self._roundup(ns4, lsz),), (lsz,), np.int32(ns4), self.cs_sp_s_buff, self.cs_sp_q_buff, np.float32(-alpha))
            self._sp_Atv(ns, ncoef, nc, ngroups, SB, d_span, wF_eff, 0, self.cs_sp_g_buff)
            self.prg.cs_mul(self.queue, (nGc,), (lsz,), np.int32(ncoef), self.cs_sp_g_buff, self.cs_sp_D_buff, self.cs_sp_g_buff)      # g = D Âᵀ s
            if lam > 0:
                self.prg.addMul(self.queue, (nGc,), (lsz,), np.int32(ncoef), self.cs_sp_g_buff, self.cs_sp_z_buff, np.float32(-lam2))
            gg_new = self.dot_gpu(self.cs_sp_g_buff, self.cs_sp_g_buff, ncoef)
            beta = gg_new / (gg + 1e-30)
            self.prg.setLinear(self.queue, (nGc,), (lsz,), np.int32(ncoef), self.cs_sp_p_buff, np.float32(1.0), self.cs_sp_g_buff, np.float32(beta), self.cs_sp_p_buff)
            gg = gg_new
        # fitted coefficients c = D z -> resident cs_sp_v_buff, eval via cs_sp_Av
        self.prg.cs_mul(self.queue, (nGc,), (lsz,), np.int32(ncoef), self.cs_sp_z_buff, self.cs_sp_D_buff, self.cs_sp_v_buff)
        self.queue.finish()
        sec = time.perf_counter() - t0
        self._sp_fit = dict(nat=nc, nmod=nmod, d_span=float(d_span), wF=float(wF_eff))
        coeffs = self.fromGPU_(self.cs_sp_v_buff, shape=(ncoef,)).astype(np.float64).reshape(nc, nmod)
        info = dict(iters=it, sec=sec, hist=hist, colnorm=colnorm, rms_s=float(np.sqrt(self.dot_gpu(self.cs_sp_s_buff, self.cs_sp_s_buff, ns4) / ns4)))
        if bPrint:
            print(f'fit_core_sp_cg: ns={ns} ncoef={ncoef} ({nc}x{nmod}) iters={it} |Âᵀs|/|Âᵀs0|={hist[-1]:.2e} resid_rms={info["rms_s"]:.3e} ({sec:.1f}s)', flush=True)
        return coeffs, info

    def fit_core_sp_gram(self, centers, w, Rlad, pts, E_ref, F_ref, wF=0.3, ridge=1e-6, SB=1024, bPrint=True, active=None):
        """Direct normal-equation solve on the GPU: assemble G = ÂᵀÂ column by
        column (G[:,j] = Atv(Av e_j), ~3 kernel calls/column, no dense design
        anywhere), then Jacobi-scaled Tikhonov + Cholesky on the host.
        Deterministic — no CG iterations. Same contract as fit_core_sp_cg
        (returns per-center slot-layout coeffs; fitted buffer stays resident)."""
        assert hasattr(self.prg, 'cs_sp_Av'), 'ContactSurfaceCL program lacks cs_sp_* kernels — build with CS_PME_SP_CORE=1 (poly8sp options)'
        centers = np.ascontiguousarray(centers, np.float64).reshape(-1, 3)
        w = np.ascontiguousarray(w, np.float64).reshape(-1)
        Rlad = np.asarray(Rlad, np.float64)
        nmod = 3 + 3 * (len(Rlad) - 3)
        nc = len(centers); ncoef = nc * nmod
        pts = np.ascontiguousarray(pts, np.float32).reshape(-1, 3)
        E_ref = np.ascontiguousarray(E_ref, np.float64).reshape(-1)
        assert len(pts) == len(E_ref)
        ns = len(pts)
        wF_eff = float(wF) if F_ref is not None else 0.0
        if F_ref is None:
            F_ref = np.zeros((ns, 3), np.float64)
        lsz = self.nloc
        loc = self.ctx.devices[0].get_info(cl.device_info.LOCAL_MEM_SIZE)
        assert nc * 16 + ncoef * 4 + 2 * lsz * 16 <= loc, f'local memory insufficient for nat={nc} NMODES={nmod}'
        SB = max(int(SB), lsz)
        ngroups = (ns + SB - 1) // SB
        self._sp_ensure(ns, ncoef, ngroups)
        atoms4 = np.zeros((nc, 4), np.float32); atoms4[:, :3] = centers; atoms4[:, 3] = w
        if getattr(self, 'cs_sp_atoms_buff', None) is None or self.cs_sp_atoms_buff.size != nc * 16:
            self.cs_sp_atoms_buff = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, nc * 16)
        self.toGPU_(self.cs_sp_atoms_buff, atoms4)
        b4 = np.zeros((ns, 4), np.float32)
        b4[:, :3] = -wF_eff * F_ref; b4[:, 3] = E_ref
        self.toGPU_(self.cs_sp_samp_buff, pts.reshape(-1))
        self.toGPU_(self.cs_sp_b_buff, b4)
        d_span = np.float32(Rlad[0])
        t0 = time.perf_counter()
        nGs = self._roundup(ns, lsz)
        ns4 = 4 * ns
        act = np.ones(ncoef, bool) if active is None else np.asarray(active, bool)
        assert act.shape == (ncoef,)
        # column norms (diag of G) and rhs b = Âᵀ b̂
        self.prg.cs_copy(self.queue, (self._roundup(ns4, lsz),), (lsz,), np.int32(ns4), self.cs_sp_b_buff, self.cs_sp_s_buff)
        self._sp_Atv(ns, ncoef, nc, ngroups, SB, d_span, wF_eff, 1, self.cs_sp_D_buff)
        colnorm = self.fromGPU_(self.cs_sp_D_buff, shape=(ncoef,)).astype(np.float64)
        assert np.isfinite(colnorm).all(), 'non-finite column norm in sp design'
        self._sp_Atv(ns, ncoef, nc, ngroups, SB, d_span, wF_eff, 0, self.cs_sp_g_buff)
        b = self.fromGPU_(self.cs_sp_g_buff, shape=(ncoef,)).astype(np.float64)
        # Gram columns: only active ones (inactive slot columns are exactly 0)
        G = np.zeros((ncoef, ncoef), np.float64)
        jact = np.flatnonzero(act & (colnorm > 0))
        one = np.zeros(1, np.float32); one[0] = 1.0
        for j in jact:
            self.prg.cs_zero(self.queue, (self._roundup(ncoef, lsz),), (lsz,), np.int32(ncoef), self.cs_sp_v_buff)
            cl.enqueue_fill_buffer(self.queue, self.cs_sp_v_buff, one, 4 * int(j), 4)
            self._sp_Av(nGs, ns, ncoef, nc, d_span, wF_eff, self.cs_sp_v_buff, self.cs_sp_q_buff)
            self.prg.cs_copy(self.queue, (self._roundup(ns4, lsz),), (lsz,), np.int32(ns4), self.cs_sp_q_buff, self.cs_sp_s_buff)
            self._sp_Atv(ns, ncoef, nc, ngroups, SB, d_span, wF_eff, 0, self.cs_sp_g_buff)
            G[:, j] = self.fromGPU_(self.cs_sp_g_buff, shape=(ncoef,))
            if bPrint and j % 200 == 0:
                print(f'  fit_core_sp_gram: col {j}/{ncoef} ({time.perf_counter() - t0:.1f}s)', flush=True)
        G = 0.5 * (G + G.T)                                        # symmetrize f32 assembly noise
        d = 1.0 / np.sqrt(np.maximum(colnorm, 1e-30)); d[~(act & (colnorm > 0))] = 0.0
        Gs = G * d[:, None] * d[None, :] + float(ridge) * np.eye(ncoef)
        try:
            xs = np.linalg.solve(Gs, d * b)
        except np.linalg.LinAlgError:
            xs = np.linalg.lstsq(Gs, d * b, rcond=1e-12)[0]
        c = (xs * d).astype(np.float32)
        self.toGPU_(self.cs_sp_v_buff, c)
        # true operator residual: s = b̂ - Â c
        self._sp_Av(nGs, ns, ncoef, nc, d_span, wF_eff, self.cs_sp_v_buff, self.cs_sp_q_buff)
        self.prg.cs_copy(self.queue, (self._roundup(ns4, lsz),), (lsz,), np.int32(ns4), self.cs_sp_b_buff, self.cs_sp_s_buff)
        self.prg.addMul(self.queue, (self._roundup(ns4, lsz),), (lsz,), np.int32(ns4), self.cs_sp_s_buff, self.cs_sp_q_buff, np.float32(-1.0))
        rms_s = float(np.sqrt(self.dot_gpu(self.cs_sp_s_buff, self.cs_sp_s_buff, ns4) / ns4))
        self.queue.finish()
        sec = time.perf_counter() - t0
        self._sp_fit = dict(nat=nc, nmod=nmod, d_span=float(d_span), wF=float(wF_eff))
        coeffs = c.astype(np.float64).reshape(nc, nmod)
        resid = np.linalg.norm(G @ c - b) / max(np.linalg.norm(b), 1e-30)
        info = dict(iters=0, sec=sec, nactive=len(jact), colnorm=colnorm, resid=resid, rms_s=rms_s)
        if bPrint:
            print(f'fit_core_sp_gram: ns={ns} ncoef={ncoef} active={len(jact)} ridge={ridge:g} |Gc-b|/|b|={resid:.2e} ({sec:.1f}s)', flush=True)
        return coeffs, info

    def eval_core_sp_gpu(self, pts, chunk=1 << 20):
        """Evaluate the fitted poly8sp core (E, F=-grad E) via cs_sp_Av with wF=1.
        Requires a prior fit_core_sp_cg (resident centers + coefficients)."""
        fit = getattr(self, '_sp_fit', None)
        assert fit is not None, 'eval_core_sp_gpu requires a prior fit_core_sp_cg'
        pts = np.ascontiguousarray(pts, np.float32).reshape(-1, 3)
        n = len(pts)
        ncoef = fit['nat'] * fit['nmod']
        E = np.empty(n); F = np.empty((n, 3))
        for i0 in range(0, n, int(chunk)):
            nb = min(int(chunk), n - i0)
            self._sp_ensure_eval(nb)
            self.toGPU_(self.cs_sp_samp_buff, pts[i0:i0 + nb].reshape(-1))
            self._sp_Av(self._roundup(nb, self.nloc), nb, ncoef, fit['nat'], np.float32(fit['d_span']), 1.0, self.cs_sp_v_buff, self.cs_sp_q_buff)
            y = self.fromGPU_(self.cs_sp_q_buff, shape=(nb, 4))
            E[i0:i0 + nb] = y[:, 3]
            F[i0:i0 + nb] = -y[:, :3]
        return E, F

    def _sp_ensure_eval(self, ns):
        if getattr(self, 'cs_sp_samp_buff', None) is None or self.cs_sp_samp_buff.size < ns * 12:
            self.try_make_buffers({'cs_sp_samp': ns * 12, 'cs_sp_q': ns * 16}, suffix='_buff')


# backward-compatible aliases
SeparableBsplinePoly = SeparableParams
RadialPIC = PICParams


def load_atom_data(xyz_path, type_map=None):
    apos, reqs, enames, Zs, lvec = load_xyz_with_REQs(xyz_path, type_map=type_map)
    qs = reqs[:, 2].copy()
    return apos, reqs, enames, lvec, qs


def scatter_molecules(mol_path, n_mol, box_xy, box_z, max_tilt_deg=5.0, margin=15.0, seed=42, type_map=None):
    apos0, reqs0, enames0, _, qs0 = load_atom_data(mol_path, type_map=type_map)
    com0 = apos0.mean(axis=0)
    apos0 = apos0 - com0
    rng = np.random.default_rng(seed)
    all_pos, all_req, all_en = [], [], []
    for _ in range(n_mol):
        tilt = rng.uniform(-max_tilt_deg, max_tilt_deg, size=3) * np.pi / 180.0
        cx, sx = np.cos(tilt[0]), np.sin(tilt[0])
        cy, sy = np.cos(tilt[1]), np.sin(tilt[1])
        cz, sz = np.cos(tilt[2]), np.sin(tilt[2])
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        x = rng.uniform(margin, box_xy - margin)
        y = rng.uniform(margin, box_xy - margin)
        z = rng.uniform(0.0, max(1.0, box_z * 0.15))
        t = np.array([x, y, z])
        p = (apos0 @ R.T) + t
        all_pos.append(p)
        all_req.append(reqs0)
        all_en.extend(enames0)
    apos = np.vstack(all_pos)
    reqs = np.vstack(all_req)
    lvec = np.array([[box_xy, 0, 0], [0, box_xy, 0], [0, 0, box_z]], dtype=np.float64)
    return apos, reqs, all_en, lvec


def write_assembly_xyz(path, apos, enames, qs, lvec):
    with open(path, 'w') as f:
        f.write(f'{len(apos)}\n')
        f.write('lvec: ' + ' '.join(f'{v:.3f}' for row in lvec for v in row) + '\n')
        for i in range(len(apos)):
            q = qs[i] if qs is not None else 0.0
            f.write(f'{enames[i]:2s} {apos[i,0]:12.6f} {apos[i,1]:12.6f} {apos[i,2]:12.6f} {q: .4f}\n')


def make_fit_grid(x0, x1, y0, y1, z0, z1, dx, dy, dz):
    xs = np.arange(x0, x1 + 1e-9, dx)
    ys = np.arange(y0, y1 + 1e-9, dy)
    zs = np.arange(z0, z1 + 1e-9, dz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def make_fit_grid_zstack(x0, x1, y0, y1, z_planes, dx, dy):
    """Sample xy grid on multiple z planes (contact z-stack for Fz-constrained fit)."""
    chunks = [make_fit_grid(x0, x1, y0, y1, float(z), float(z), dx, dy, dx) for z in z_planes]
    return np.vstack(chunks)


def make_fit_z_planes_adaptive(z_lo, z_hi, dz_lo=0.1, dz_hi=1.0):
    """Non-uniform z sample offsets: dz grows linearly from dz_lo at z_lo to dz_hi at z_hi."""
    z = float(z_lo)
    z_hi = float(z_hi)
    span = max(z_hi - z_lo, 1e-12)
    planes = [z]
    while z < z_hi - 1e-9:
        frac = (z - z_lo) / span
        dz = float(dz_lo) + (float(dz_hi) - float(dz_lo)) * frac
        z = min(z + dz, z_hi)
        if z > planes[-1] + 1e-9:
            planes.append(z)
    if planes[-1] < z_hi - 1e-9:
        planes.append(z_hi)
    return np.asarray(planes, dtype=np.float64)


def make_fit_grid_surface_following(x0, x1, y0, y1, s_offsets, dx, dy, h0_map, bspl_dx, x0_h0, y0_h0, ncx, ncy):
    """Surface-following fit grid: z = h0(x,y) + s_k for each (x,y) and each s offset.

    This ensures every lateral site gets fit samples at the same local-s values,
    unlike global z-planes which only sample near-contact at the highest h0 sites.

    Args:
        x0, x1, y0, y1: fit grid bounds [Ang]
        s_offsets: (ns,) array of local-s values to sample at
        dx, dy: fit grid spacing [Ang]
        h0_map: (ncx*ncy,) B-spline control coefficients for h0 interpolation
        bspl_dx: B-spline grid spacing [Ang]
        x0_h0, y0_h0: h0 grid origin [Ang]
        ncx, ncy: h0 grid dimensions

    Returns:
        fit_pts: (nxy * ns, 3) array of (x, y, z) fit points
    """
    xs = np.arange(x0, x1 + 1e-9, dx)
    ys = np.arange(y0, y1 + 1e-9, dy)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    nxy = gx.size
    ns = len(s_offsets)
    pts = np.zeros((nxy * ns, 3), dtype=np.float32)
    # Interpolate h0 at each (x,y) fit point using the same B-spline as the kernel
    h0_xy = np.array([interp_h0(float(gx.flat[i]), float(gy.flat[i]), h0_map, bspl_dx, x0_h0, y0_h0, ncx, ncy) for i in range(nxy)])
    for k, s in enumerate(s_offsets):
        sl = slice(k * nxy, (k + 1) * nxy)
        pts[sl, 0] = gx.ravel()
        pts[sl, 1] = gy.ravel()
        pts[sl, 2] = h0_xy + float(s)
    return pts


def poly_z_doubling_modes(dz, poly_R=10.0, m_start=4, nz=6, poly_z0=0.0):
    """Z radial basis φ_k(dz) matching contact_surface.cl poly_z_doubling_modes.

    dz = z - h0 [Å].  Rc = poly_R.  x = clip((dz-poly_z0)/Rc, 0, 1), t = 1-x.
    φ_0 = t^m_start, φ_{k+1} = φ_k^2  (powers m_start·2^k).
    Returns phi (n, nz), dphi_dz (n, nz), t, x.
    """
    dz = np.asarray(dz, dtype=np.float64).reshape(-1)
    invRc = 1.0 / float(poly_R)
    dist = dz - float(poly_z0)
    x = np.clip(dist * invRc, 0.0, 1.0)
    t = 1.0 - x
    n = len(dz)
    phi = np.zeros((n, int(nz)), dtype=np.float64)
    dphi = np.zeros((n, int(nz)), dtype=np.float64)
    for i in range(n):
        di, xi, ti = float(dist[i]), float(x[i]), float(t[i])
        active = (di >= 0.0) and (xi < 1.0)
        tpow = ti ** int(m_start) if m_start > 0 else 1.0
        dtpow = (-float(m_start) * invRc * (ti ** (int(m_start) - 1))) if active and m_start > 0 else 0.0
        for k in range(int(nz)):
            phi[i, k] = tpow
            dphi[i, k] = dtpow if active else 0.0
            if k + 1 < int(nz):
                tp_n = tpow
                tpow = tpow * tpow
                dtpow = 2.0 * tp_n * dtpow
    return phi, dphi, t, x


def poly_z_mode_powers(m_start=4, nz=6):
    """Exponent of t for each z mode: m_start * 2^k."""
    return [int(m_start) * (2 ** k) for k in range(int(nz))]


def boltzmann_fit_weights(E_ref, T=None, E_shift=None):
    """Boltzmann weights w=exp(-(E-E_shift)/T), normalized to max 1. Emphasizes low-energy (vdW-well) samples."""
    E = np.asarray(E_ref, dtype=np.float64).reshape(-1)
    E_shift = float(E.min() if E_shift is None else E_shift)
    if T is None:
        spread = float(np.percentile(E, 95) - E_shift)
        T = max(spread / 3.0, 0.05)
    w = np.exp(-(E - E_shift) / float(T))
    w /= max(float(w.max()), 1e-16)
    return w.astype(np.float32), float(T), E_shift


def eval_slice_map_gpu(ocl, eval_fn, x0, x1, y0, y1, z_scan, dx, dy):
    """Single GPU launch for full xy slice; one readback at end. Returns Fz[ix,iy] at xs,ys."""
    xs = np.arange(x0, x1 + 1e-9, dx)
    ys = np.arange(y0, y1 + 1e-9, dy)
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_scan)], axis=1)
    E, F = eval_fn(pts)
    return xs, ys, F[:, 2].reshape(nx, ny)


def pic_grid_to_map(F, nx, ny):
    """PIC tiled kernel uses iq=iy*nx+ix; map to Fz[ix,iy] like eval_slice_map_gpu."""
    return F[:, 2].reshape(ny, nx).T


# ═══════════════════════════════════════════════════════════════════════════════
# ContactPMEParams — Wave 2 host/API container for the contact_pme backend
# (doc/Tasks/ContactSurface_PME_ParallelPlan.md §Agent_2 Wave 2). Owns the mesh
# coefficients, core fit, split params, atoms, PIC buckets, declared query
# interior, and exact resident-byte accounting. Mathematics lives in
# CoarseMesh.py / PICCore.py / PMESplit.py — this dataclass only aggregates.
# ═══════════════════════════════════════════════════════════════════════════════
from dataclasses import dataclass, field

@dataclass
class ContactPMEParams:
    """Aggregated state for the contact_pme particle-mesh backend.

    Owns mesh coefficients (CoarseMesh), core fit (CoreFit), split params
    (SplitParams), atom positions, PIC buckets, the declared safe query
    interior, and exact resident-byte accounting. Built by
    AFMulator.fit_contact_pme(); evaluated by AFMulator.eval_contact_pme().
    """
    # Mesh (V_L on coarse 3D B-spline grid); coeffs are prefiltered control coefficients
    mesh_coeffs: np.ndarray            # (nx, ny, nz) float64 B-spline control coefficients
    mesh_origin: np.ndarray            # (3,) world coords of node (0,0,0)
    mesh_h: float                      # mesh spacing [Å]
    mesh_halo: int                     # halo node count per side used at build time
    query_interior: tuple              # (lo[3], hi[3]) integer node indices defining safe interior
    # Core (compact atom-centered residual v_S)
    core_fit: object                   # PICCore.CoreFit (per-atom coeffs, r_lo, r_cut, metrics)
    # Split (atomwise soft-core split + combined radial oracle)
    split_params: object               # PMESplit.SplitParams (R0/E0/q/alpha/q_tip/r_damp/r_cut)
    # Atoms + PIC buckets (XY cell lists for core evaluation)
    atom_pos: np.ndarray               # (na, 3) float64 atom positions [Å, world coords]
    bucket_atoms: np.ndarray           # (n_list,) int32 atom indices in bucket order
    bucket_offsets: np.ndarray         # (nbx*nby+1,) int32 CSR-style offsets
    bucket_nbx: int                    # number of buckets along x
    bucket_nby: int                    # number of buckets along y
    bucket_cell_size: float            # bucket cell size [Å] (>= r_cut)
    bucket_bounds: tuple               # (x0, y0, x1, y1) bucket domain [Å]
    # Resident-byte accounting (sum of all device-resident buffers, excluding
    # temporary build/test buffers). Updated by resident_bytes property.
    _resident_bytes: int = 0
    core_span_override: float = None   # set for split_params=None params (poly8 core+mesh):
                                       # supplies r_cut and core_d_span (kernel ABI span)

    def __post_init__(self):
        self._resident_bytes = self._compute_resident_bytes()

    def _compute_resident_bytes(self) -> int:
        """Exact sum of device-resident buffer sizes [bytes].

        Counts mesh coefficients, atoms, core coefficients, buckets, offsets,
        and metadata separately from temporary build/test buffers. Mesh/core
        data are uploaded as float32 on device; atoms as float4; buckets as int32.
        """
        b = 0
        # Mesh coeffs: (nx, ny, nz) float32 on device
        b += int(self.mesh_coeffs.size * 4)
        # Atoms: (na, 4) float32 (pos + charge)
        b += int(self.atom_pos.shape[0] * 4 * 4)
        # Core coeffs: (na, N_MODES) float32
        b += int(self.core_fit.coeffs.size * 4)
        # Per-atom r_lo: (na,) float32
        b += int(self.core_fit.r_lo.size * 4)
        # Buckets: (n_list,) int32
        b += int(self.bucket_atoms.size * 4)
        # Offsets: (nbx*nby+1,) int32
        b += int(self.bucket_offsets.size * 4)
        # Metadata scalars (mesh origin/h/halo, split globals) — small fixed cost
        b += 3 * 4 + 4 + 4  # origin(3 float) + h + halo
        return b

    @property
    def resident_bytes(self) -> int:
        """Exact device-resident buffer total [bytes] (excludes temp build/test)."""
        return self._resident_bytes

    @property
    def resident_kb(self) -> float:
        return self._resident_bytes / 1024.0

    @property
    def na(self) -> int:
        return int(self.atom_pos.shape[0])

    @property
    def r_cut(self) -> float:
        """Neighbor / core outer cutoff: r_core_max for compact splits, else legacy r_cut."""
        if self.core_span_override is not None:
            return float(self.core_span_override)
        sp = self.split_params
        return float(getattr(sp, 'r_core_max', sp.r_cut))

    @property
    def core_d_span(self) -> float:
        """r_b - r_lo (= Δ_in + Δ_b). Passed to OpenCL as basis support span."""
        if self.core_span_override is not None:
            return float(self.core_span_override)
        from spammm.surfaces.PMESplit import _COMPACT_SPLIT_MODES
        sp = self.split_params
        if getattr(sp, 'split_mode', 'paw') in _COMPACT_SPLIT_MODES:
            return float(sp.delta_in + sp.delta_b)
        # legacy rho: not constant; host should not use GPU core with mixed spans
        return float(sp.r_cut - float(np.min(sp.r_lo)))

    @property
    def mesh_shape(self) -> tuple:
        return tuple(self.mesh_coeffs.shape)


def cpm_params_from_samples(samples, mesh_origin, mesh_h, halo, apos, cLJs,
                            split_mode='paw', r_cut=6.0, core_fit=None, split_params=None):
    """Assemble ContactPMEParams for an ARBITRARY sampled energy field.

    Fast PME compression of a grid field (e.g. FDBM E_total): the dense tricubic
    B-spline mesh IS the representation — prefiltered control coefficients from
    CoarseMesh._prefilter_3d (separable tridiagonal solves, O(n), no CG).

    Two modes:
      - `core_fit=None` (default): zero core — the whole field sits on the mesh;
        mesh_h must resolve the repulsive wall itself. Measured (FDBM field,
        0.1 Å source grid, 2026-10-07): h_mesh=0.4 Å → interpolating
        B-spline ringing on the ~1.5 Å atom peaks → moiré/checkerboard in
        relaxed df (NCC ~0.73–0.91); h_mesh=0.3 → smooth-undershoot regime,
        NCC ≥0.98, faint atom-localized dots; h_mesh≈0.2 Å removes them
        (NCC ≥0.989). The dots are a smoothed-wall artifact amplified by
        PP relaxation — NOT the (absent) core, NOT the eval kernel. Also
        measured: mesh/field grid ALIGNMENT is irrelevant — sampling nodes
        offset by half a field step is marginally BETTER than coincident
        nodes (trilinear pre-smoothing tames spline ringing); do not bother
        snapping mesh_origin to the field lattice.
      - `core_fit=None` deep-wall note: Fz is under-represented inside the
        repulsive core (z≲2.5 Å over atoms, ~50–150 eV/Å) at ALL mesh
        spacings — unreachable by the PP, cosmetic only.
      - `core_fit=CoreFit` (real PAW — the intended design): `samples` must
        be the SMOOTH RESIDUAL (E − core evaluated at nodes); the mesh can
        then be very coarse (0.35–1.0 Å) — cores carry the sharp short-range
        structure. Fit cores with PICCore.fit_cores_paw_field /
        fit_cores_from_samples against oracle samples. CAVEAT 2026-10-08:
        on FDBM fields this path currently FAILS df parity (azaindol
        df_corr 0.11–0.83; taper-shell ringing) — fix before relying on it;
        zero-core h≤0.2 is the measured-accurate interim.

    Args:
        samples: (nx, ny, nz) E-field (or residual) values at the mesh nodes.
        mesh_origin: (3,) world coords of node (0,0,0) — query_bounds lo - halo*h.
        mesh_h: mesh spacing [Å].
        halo: halo node count per side (>= ~4 for the tricubic stencil interior).
        apos: (na, 3) atom positions (heavy atoms only, production convention).
        cLJs: (na, 4) Morse params (R0, E0, alpha, 0) — feeds SplitParams unless
              `split_params` is given.
        core_fit: optional CoreFit from PICCore.fit_cores_from_samples — r_lo/r_b
              must match `split_params`/kernel ABI (span = Δ_in+Δ_b).
        split_params: optional SplitParams override (needed when core_fit given —
              its r_lo/r_b drive both the core eval and the bucket cell size).
    Returns:
        ContactPMEParams (mesh + CoreFit + 'paw' SplitParams + PIC buckets).
    """
    from spammm.surfaces.CoarseMesh import _prefilter_3d
    from spammm.surfaces.PICCore import CoreFit, CORE_POWERS
    from spammm.surfaces.PMESplit import SplitParams
    samples = np.asarray(samples, np.float64)
    ns = np.asarray(samples.shape, np.int64)
    coeffs = _prefilter_3d(samples).astype(np.float32)
    na = len(apos)
    cMs = np.asarray(cLJs, np.float64)
    sp = split_params if split_params is not None else SplitParams(
                     R0=cMs[:, 0], E0=cMs[:, 1], q=np.zeros(na), alpha=float(abs(cMs[0, 2])),
                     q_tip=0.0, r_damp=0.1, r_cut=r_cut, split_mode=split_mode)
    if core_fit is None:
        z_ = np.zeros(na, np.float64)
        core = CoreFit(coeffs=np.zeros((na, len(CORE_POWERS))), r_lo=np.full(na, 2.0),
                       r_b=np.full(na, 4.0), powers=np.asarray(CORE_POWERS, np.int64), basis='raw',
                       cond_raw=z_, cond_hier=z_, train_rmse_E=z_, train_rmse_F=z_,
                       held_rmse_E=z_, held_rmse_F=z_, held_max_E=z_, held_max_F=z_, worst_r=z_)
    else:
        core = core_fit
        assert core.coeffs.shape[0] == na, f'core_fit na={core.coeffs.shape[0]} vs {na}'
        sp_span = float(sp.delta_in + sp.delta_b)
        assert np.allclose(np.asarray(core.r_b) - np.asarray(core.r_lo), sp_span), \
            f'core span {core.r_b - core.r_lo} vs split_params Δ_in+Δ_b={sp_span} (kernel ABI)'
    cs = float(sp.r_core_max)
    x0 = float(apos[:, 0].min()) - cs; x1 = float(apos[:, 0].max()) + cs
    y0 = float(apos[:, 1].min()) - cs; y1 = float(apos[:, 1].max()) + cs
    batoms, boffs, nbx, nby = build_pic_buckets(np.asarray(apos, np.float64), x0, y0, x1, y1, cs)
    interior = (np.array([halo] * 3, np.int64),
                np.array([ns[0] - 1 - halo, ns[1] - 1 - halo, ns[2] - 1 - halo], np.int64))
    return ContactPMEParams(mesh_coeffs=coeffs, mesh_origin=np.asarray(mesh_origin, np.float64),
                            mesh_h=float(mesh_h), mesh_halo=int(halo),
                            query_interior=interior, core_fit=core, split_params=sp,
                            atom_pos=np.asarray(apos, np.float64), bucket_atoms=batoms,
                            bucket_offsets=boffs, bucket_nbx=nbx, bucket_nby=nby,
                            bucket_cell_size=cs, bucket_bounds=(x0, y0, x1, y1))


def cpm_params_from_coremesh(mesh_coeffs, mesh_origin, mesh_h, halo, centers, core_fit):
    """Assemble ContactPMEParams for the poly8 core+coarse-mesh representation.

    Unlike cpm_params_from_samples, `mesh_coeffs` are FITTED B-spline control
    coefficients (fit_coremesh_lsq output on E - core) — NOT node samples, so no
    prefilter is applied. `centers` are the poly8 core centers (atoms + bond
    midpoints), uploaded as the kernel atoms with w=r_lo (poly8: 0; poly8sp:
    radius scale) and span=max cutoff via core_span_override (split_params=None
    — nothing in the eval/scan path
    may dereference it; r_cut/core_d_span come from the override).

    Args:
        mesh_coeffs: (nx,ny,nz) fitted B-spline control coefficients.
        mesh_origin: (3,) world coords of node (0,0,0); mesh_h: node spacing [A].
        halo: halo node count per side used at fit time.
        centers: (nc,3) core centers; core_fit: poly8 (coeffs (nc,5)) or
            poly8sp (coeffs (nc,9), r_lo=radius scale) CoreFit.
    """
    centers = np.asarray(centers, np.float64).reshape(-1, 3)
    assert core_fit.basis in ('poly8', 'poly8sp'), f'cpm_params_from_coremesh expects basis poly8/poly8sp, got {core_fit.basis}'
    nmod = 3 * (len(core_fit.poly_R) - 2) if core_fit.basis == 'poly8sp' else 5   # 3 s + 3*NP p slots
    assert core_fit.coeffs.shape == (len(centers), nmod), f'coeffs {core_fit.coeffs.shape} vs ({len(centers)},{nmod})'
    if core_fit.basis == 'poly8sp':
        # kernel SP branch: w = radius scale, per-center cutoff = w * R0 (poly_R[0])
        assert np.all(np.asarray(core_fit.r_lo) > 0), 'poly8sp expects r_lo = radius scale > 0'
        assert np.allclose(np.asarray(core_fit.r_b), np.asarray(core_fit.r_lo) * core_fit.poly_R[0]), 'poly8sp r_b != r_lo * R0'
    ns = np.asarray(np.asarray(mesh_coeffs).shape, np.int64)
    cs = float(np.max(core_fit.r_b))            # max per-center evaluation cutoff (== max(poly_R) for poly8)
    x0 = float(centers[:, 0].min()) - cs; x1 = float(centers[:, 0].max()) + cs
    y0 = float(centers[:, 1].min()) - cs; y1 = float(centers[:, 1].max()) + cs
    batoms, boffs, nbx, nby = build_pic_buckets(centers, x0, y0, x1, y1, cs)
    interior = (np.array([halo] * 3, np.int64),
                np.array([ns[0] - 1 - halo, ns[1] - 1 - halo, ns[2] - 1 - halo], np.int64))
    return ContactPMEParams(mesh_coeffs=np.asarray(mesh_coeffs, np.float64),
                            mesh_origin=np.asarray(mesh_origin, np.float64),
                            mesh_h=float(mesh_h), mesh_halo=int(halo),
                            query_interior=interior, core_fit=core_fit, split_params=None,
                            atom_pos=centers, bucket_atoms=batoms, bucket_offsets=boffs,
                            bucket_nbx=nbx, bucket_nby=nby, bucket_cell_size=cs,
                            bucket_bounds=(x0, y0, x1, y1), core_span_override=cs)

