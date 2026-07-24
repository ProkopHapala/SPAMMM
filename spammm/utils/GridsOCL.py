"""GridsOCL — GPU 3D grid ops (dipole-preserving project/downsample, Gaussians).

Wraps ``kernels/grids.cl``. Primary use: down-sample cube densities onto a
coarser FDBM lattice by *projecting* each source voxel as a point charge onto
8 dest neighbors (trilinear weights) — preserves charge and dipole when the
dest box covers the support.

Caveat: project treats voxels as **centers** ``origin+(i+½)·h``; Gaussian NA
builders / ``grid_moments`` default use **corners** ``origin+i·h``. Mixing
conventions without care changes multipoles (see ``doc/Caveats.md``,
``AFM_utils.plot_cube_delta_rho_na_origin_diag``).

Also: Gaussian ρ_NA splat/add, axpy, subtract, fill.
"""
from __future__ import annotations

import os
import numpy as np
import pyopencl as cl

from .OpenCLBase import OpenCLBase


def grid_moments(rho, origin, step):
    """Return (q, dipole_xyz) with q=∫ρ dV, p=∫ρ(r−origin)dV? — p about geometric origin of coords.

    Positions at voxel *corners* matching OpenCL add_gaussian (ix*step), for
    project kernel centers use ``centers=True``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
    if np.ndim(step) == 0:
        sx = sy = sz = float(step)
    else:
        sx, sy, sz = map(float, step[:3])
    nx, ny, nz = rho.shape
    dV = sx * sy * sz
    xs = origin[0] + sx * np.arange(nx, dtype=np.float64)
    ys = origin[1] + sy * np.arange(ny, dtype=np.float64)
    zs = origin[2] + sz * np.arange(nz, dtype=np.float64)
    # corner sampling (same as Gaussian kernel grid points)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    q = float(rho.sum() * dV)
    if abs(q) < 1e-30:
        return q, np.zeros(3, dtype=np.float64)
    px = float((X * rho).sum() * dV)
    py = float((Y * rho).sum() * dV)
    pz = float((Z * rho).sum() * dV)
    return q, np.array([px, py, pz], dtype=np.float64)


def grid_moments_centers(rho, origin, step):
    """Moments with positions at voxel centers (matches project_density_trilinear)."""
    rho = np.asarray(rho, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
    if np.ndim(step) == 0:
        sx = sy = sz = float(step)
    else:
        sx, sy, sz = map(float, step[:3])
    nx, ny, nz = rho.shape
    dV = sx * sy * sz
    xs = origin[0] + sx * (np.arange(nx, dtype=np.float64) + 0.5)
    ys = origin[1] + sy * (np.arange(ny, dtype=np.float64) + 0.5)
    zs = origin[2] + sz * (np.arange(nz, dtype=np.float64) + 0.5)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    q = float(rho.sum() * dV)
    px = float((X * rho).sum() * dV)
    py = float((Y * rho).sum() * dV)
    pz = float((Z * rho).sum() * dV)
    return q, np.array([px, py, pz], dtype=np.float64)


class GridsOCL(OpenCLBase):
    """PyOpenCL driver for ``kernels/grids.cl``."""

    def __init__(self, nloc=64, preferred_vendor='nvidia', device_index=0, ctx=None, queue=None):
        super().__init__(nloc=nloc, preferred_vendor=preferred_vendor, device_index=device_index,
                         ctx=ctx, queue=queue)
        kdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kernels')
        kpath = os.path.abspath(os.path.join(kdir, 'grids.cl'))
        self.load_program(kernel_path=kpath)

    def _gs(self, n):
        n = int(n)
        nloc = min(int(self.nloc), n) if n > 0 else 1
        # round global up
        gs = ((n + nloc - 1) // nloc) * nloc
        return (gs,), (nloc,)

    def fill(self, buf, n, val=0.0):
        self.prg.grid_fill(self.queue, *self._gs(n), np.int32(n), buf, np.float32(val))

    def axpy(self, alpha, x_buf, beta, y_buf, n):
        self.prg.grid_axpy(self.queue, *self._gs(n),
                           np.int32(n), np.float32(alpha), x_buf, np.float32(beta), y_buf)

    def subtract(self, a_buf, b_buf, out_buf, n):
        self.prg.grid_subtract(self.queue, *self._gs(n), np.int32(n), a_buf, b_buf, out_buf)

    def add_const(self, buf, n, c):
        self.prg.grid_add_const(self.queue, *self._gs(n), np.int32(n), buf, np.float32(c))

    def project_density(self, src, origin_s, step_s, origin_d, step_d, ngrid_d,
                        R=None, t=None, dst=None, bTryAllocate=True):
        """Project ``src`` (nx,ny,nz) density onto dest grid by trilinear scatter.

        Parameters
        ----------
        src : (nx,ny,nz) float32 density [e/Å³]
        origin_s, step_s : source lattice (Å); step may be float or (3,)
        origin_d, step_d, ngrid_d : destination lattice
        R : optional (3,3) rotation (src world → dest world), default I
        t : optional (3,) translation in dest world Å, default 0
        dst : optional preallocated array; else zeros

        Returns
        -------
        dst : (nx_d,ny_d,nz_d) float32
        """
        src = np.asarray(src, dtype=np.float32)
        assert src.ndim == 3
        nx_s, ny_s, nz_s = src.shape
        origin_s = np.asarray(origin_s, dtype=np.float64).ravel()[:3]
        origin_d = np.asarray(origin_d, dtype=np.float64).ravel()[:3]
        if np.ndim(step_s) == 0:
            ss = np.array([step_s, step_s, step_s], dtype=np.float64)
        else:
            ss = np.asarray(step_s, dtype=np.float64).ravel()[:3]
        if np.ndim(step_d) == 0:
            sd = np.array([step_d, step_d, step_d], dtype=np.float64)
        else:
            sd = np.asarray(step_d, dtype=np.float64).ravel()[:3]
        nx_d, ny_d, nz_d = [int(x) for x in ngrid_d]
        if dst is None:
            dst = np.zeros((nx_d, ny_d, nz_d), dtype=np.float32)
        else:
            dst = np.asarray(dst, dtype=np.float32)
            assert dst.shape == (nx_d, ny_d, nz_d)

        if R is None:
            R = np.eye(3, dtype=np.float32)
        else:
            R = np.asarray(R, dtype=np.float32).reshape(3, 3)
        if t is None:
            t = np.zeros(3, dtype=np.float32)
        else:
            t = np.asarray(t, dtype=np.float32).ravel()[:3]

        dV_s = float(ss[0] * ss[1] * ss[2])
        dV_d = float(sd[0] * sd[1] * sd[2])
        vol_scale = np.float32(dV_s / dV_d)

        n_src = nx_s * ny_s * nz_s
        n_dst = nx_d * ny_d * nz_d
        if bTryAllocate:
            self.try_make_buffers({'src': n_src * 4, 'dst': n_dst * 4}, suffix='_g')

        self.toGPU_(self.src_g, np.ascontiguousarray(src.ravel()))
        self.fill(self.dst_g, n_dst, 0.0)

        R0 = cl.cltypes.make_float3(float(R[0, 0]), float(R[0, 1]), float(R[0, 2]))
        R1 = cl.cltypes.make_float3(float(R[1, 0]), float(R[1, 1]), float(R[1, 2]))
        R2 = cl.cltypes.make_float3(float(R[2, 0]), float(R[2, 1]), float(R[2, 2]))
        tt = cl.cltypes.make_float3(float(t[0]), float(t[1]), float(t[2]))

        self.prg.project_density_trilinear(
            self.queue, *self._gs(n_src),
            self.src_g,
            np.int32(nx_s), np.int32(ny_s), np.int32(nz_s),
            np.float32(origin_s[0]), np.float32(origin_s[1]), np.float32(origin_s[2]),
            np.float32(ss[0]), np.float32(ss[1]), np.float32(ss[2]),
            R0, R1, R2, tt,
            self.dst_g,
            np.int32(nx_d), np.int32(ny_d), np.int32(nz_d),
            np.float32(origin_d[0]), np.float32(origin_d[1]), np.float32(origin_d[2]),
            np.float32(sd[0]), np.float32(sd[1]), np.float32(sd[2]),
            vol_scale,
        )
        self.queue.finish()
        self.fromGPU_(self.dst_g, dst.ravel())
        return dst

    def splat_gaussians(self, grid, origin, step, atomPos, atomZ, sigma=0.3, sign=1.0,
                        nsig=5.0, bTryAllocate=True):
        """Add (sign=+1) or subtract (sign=-1) Σ Z_i N(r−R_i;σ) onto ``grid`` in-place (GPU)."""
        grid = np.asarray(grid, dtype=np.float32)
        nx, ny, nz = grid.shape
        origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
        if np.ndim(step) == 0:
            sx = sy = sz = float(step)
        else:
            sx, sy, sz = map(float, np.asarray(step).ravel()[:3])
        atomPos = np.asarray(atomPos, dtype=np.float64).reshape(-1, 3)
        atomZ = np.asarray(atomZ, dtype=np.float64).reshape(-1)
        nat = len(atomZ)
        atoms = np.zeros((nat, 4), dtype=np.float32)
        atoms[:, :3] = atomPos
        atoms[:, 3] = atomZ

        n = nx * ny * nz
        if bTryAllocate:
            self.try_make_buffers({'grid': n * 4, 'atoms': nat * 16}, suffix='_g')
        self.toGPU_(self.grid_g, np.ascontiguousarray(grid.ravel()))
        self.toGPU_(self.atoms_g, atoms.ravel())

        self.prg.splat_gaussian_atoms(
            self.queue, *self._gs(max(nat, 1)),
            self.grid_g,
            np.int32(nx), np.int32(ny), np.int32(nz),
            np.float32(origin[0]), np.float32(origin[1]), np.float32(origin[2]),
            np.float32(sx), np.float32(sy), np.float32(sz),
            self.atoms_g, np.int32(nat),
            np.float32(sigma), np.float32(sign), np.float32(nsig),
        )
        self.queue.finish()
        self.fromGPU_(self.grid_g, grid.ravel())
        return grid
