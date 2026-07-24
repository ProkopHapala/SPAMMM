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
import numpy as np
import pyopencl as cl

from spammm.utils.OpenCLBase import OpenCLBase
from spammm.utils import clUtils as clu
from spammm.topology.FFparams import load_xyz_with_REQs

COULOMB_CONST = 14.3996448915
_KERNEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kernels')


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
        return h0
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
    return h0


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
        if h0_map is not None:
            self.h0_map = np.ascontiguousarray(h0_map, dtype=np.float32).reshape(-1)
        elif apos is not None:
            self.h0_map = build_contact_height_map(apos, self.x0, self.y0, self.dx, self.dy, self.ncx, self.ncy, r_xy=h0_r_xy, Rs=self.h0_Rs)
        else:
            self.h0_map = None
        self.coeffs = None

    @property
    def n_coeff(self):
        return self.ncx * self.ncy * self.nz


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

    def __init__(self, nloc=64, ctx=None, queue=None, **kw):
        super().__init__(nloc=nloc, ctx=ctx, queue=queue, **kw)
        kernel_paths = [
            os.path.join(_KERNEL_DIR, 'common.cl'),
            os.path.join(_KERNEL_DIR, 'Forces.cl'),
            os.path.join(_KERNEL_DIR, 'contact_surface.cl'),
        ]
        self.load_program_multi(kernel_paths, bPrint=False, build_options=['-DDBG_UFF=0', '-D', 'AFM_STANDALONE=1'])
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
            sep.h0_map = build_contact_height_map(apos, sep.x0, sep.y0, sep.dx, sep.dy, sep.ncx, sep.ncy)
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
        """Global GPU CG fit. Optional WLS plus Fx,Fy,Fz rows; force_weight=1 equalizes RMS(E) vs RMS(Fα)."""
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
