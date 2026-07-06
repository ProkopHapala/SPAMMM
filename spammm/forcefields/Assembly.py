"""
Assembly.py — Hexagonal SAM rigid-body packing search (OpenCL).

Essence: GPU-batch evaluation of rigid placements on a fixed surface unit cell;
rank configs by clash, z-thickness, and inter-molecular contact.

Design:
  - AssemblyOCL wraps kernels/assembly.cl (emit_configuration_xyz, evaluate_packing_3d).
  - n_sym=6 (default): six C6-related orientations per cell (experimental hexagonal SAM).
  - align_flat: SO(3) pre-align to minimal z-span, then sample tilt/in-plane in that frame.
  - radius=1.0 Å default — softened vs element VdW to approximate residual flexibility.

Open issues / caveats:
  - Plotting and per-atom diagnostics live in AssemblyPlot.py (SoC).
  - prune_duplicate_rotations_sym is O(N²) in rotation count; off by default (dedup=False).
  - Rotation dedup is CPU-only; GPU evaluates full config grid (rot × shift × sym × PBC).
  - Legacy generate_transform_buffer() still available (54-replica 3×3 grid, full3d rotations).
"""

import os
import numpy as np
import pyopencl as cl
from ..utils.OpenCLBase import OpenCLBase
from scipy.spatial.transform import Rotation as R

class AssemblyOCL(OpenCLBase):
    def __init__(self, nloc=128, device_index=0):
        super().__init__(nloc=nloc, device_index=device_index)
        # Load kernel
        kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../kernels')
        self.load_program(kernel_path=os.path.join(kernel_dir, 'assembly.cl'))
        self.krn_emit_configuration_xyz = cl.Kernel(self.prg, "emit_configuration_xyz")
        self.krn_evaluate_packing_3d    = cl.Kernel(self.prg, "evaluate_packing_3d")
        self.d_base_atoms = None
        self.natoms = 0
        
    def upload_base_atoms(self, atoms):
        # atoms should be float32 (natoms, 4) where w is radius
        self.natoms = np.int32(len(atoms))
        atoms_f32 = self._flat32(atoms)
        self.d_base_atoms = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=atoms_f32)
        
    def evaluate_packing(self, transforms, nmols, max_clash_penalty=50.0):
        # transforms: (n_confs, nmols, 4, 4) float32
        n_confs = np.int32(transforms.shape[0])
        nmols_i32 = np.int32(nmols)
        max_penalty_f32 = np.float32(max_clash_penalty)
        
        transforms_f32 = self._flat32(transforms)
        d_transforms = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=transforms_f32)
        
        results_host = np.zeros(n_confs, dtype=np.float32)
        d_results = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, results_host.nbytes)
        
        results_min_host = np.zeros(n_confs, dtype=np.float32)
        d_results_min = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, results_min_host.nbytes)
        
        local_replica = cl.LocalMemory(self.nloc * 16) # float4
        local_scores = cl.LocalMemory(self.nloc * 4)   # float
        local_min_dist = cl.LocalMemory(self.nloc * 4) # float
        
        global_size = (int(n_confs * self.nloc),)
        local_size = (int(self.nloc),)
        
        self.krn_evaluate_packing_3d.set_args(
            self.d_base_atoms,
            self.natoms,
            d_transforms,
            nmols_i32,
            max_penalty_f32,
            local_replica,
            local_scores,
            local_min_dist,
            d_results,
            d_results_min
        )
        cl.enqueue_nd_range_kernel(self.queue, self.krn_evaluate_packing_3d, global_size, local_size).wait()
        
        cl.enqueue_copy(self.queue, results_host, d_results).wait()
        cl.enqueue_copy(self.queue, results_min_host, d_results_min).wait()
        return results_host, results_min_host
        
    def emit_configuration(self, transforms_single_conf, nmols):
        # transforms_single_conf: (nmols, 4, 4) float32
        nmols_i32 = np.int32(nmols)
        transforms_f32 = self._flat32(transforms_single_conf)
        d_transforms = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=transforms_f32)
        
        out_atoms_host = np.zeros((self.natoms * nmols, 4), dtype=np.float32)
        d_out_atoms = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, out_atoms_host.nbytes)
        
        global_size = (int(self.natoms * nmols),)
        local_size = None # let OpenCL decide
        
        self.krn_emit_configuration_xyz.set_args(
            self.d_base_atoms,
            self.natoms,
            d_transforms,
            nmols_i32,
            d_out_atoms,
        )
        cl.enqueue_nd_range_kernel(self.queue, self.krn_emit_configuration_xyz, global_size, local_size).wait()
        
        cl.enqueue_copy(self.queue, out_atoms_host, d_out_atoms).wait()
        return out_atoms_host

def pack_transforms(rotmats, shifts):
    # rotmats: (..., 3, 3), shifts: (..., 3)
    shape = rotmats.shape[:-2]
    out = np.zeros(shape + (4, 4), dtype=np.float32)
    out[..., 0, :3] = rotmats[..., 0, :]
    out[..., 1, :3] = rotmats[..., 1, :]
    out[..., 2, :3] = rotmats[..., 2, :]
    out[..., 3, :3] = shifts
    return out

def super_fibonacci_rotations(N):
    phi = np.sqrt(2)
    psi = (1 + np.sqrt(5)/2 + np.sqrt((5 + 2*np.sqrt(5) + np.sqrt(25 + 20*np.sqrt(5)))/4)) ** 0.25
    quats = np.zeros((N, 4))
    for i in range(N):
        s = i + 1
        t = s / N
        d = 2 * np.pi * s
        r = np.sqrt(t)
        R_val = np.sqrt(1 - t)
        alpha = d * phi
        beta = d * psi
        quats[i] = [r * np.sin(alpha), r * np.cos(alpha), R_val * np.sin(beta), R_val * np.cos(beta)]
    return R.from_quat(quats).as_matrix()

def generate_transform_buffer(lattice_a, lattice_b, n_rot=100, n_shift=6):
    # hexagonal symmetry (6 rotations around Z)
    angles_60 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    S_sym = R.from_euler('z', angles_60).as_matrix() # (6, 3, 3)
    
    # 3x3 grid shifts, ordered so [0,0] is index 0
    u = np.array([0, -1, 1])
    uu, vv = np.meshgrid(u, u)
    L_lat = np.zeros((9, 3))
    L_lat[:, 0] = uu.flatten() * lattice_a[0] + vv.flatten() * lattice_b[0]
    L_lat[:, 1] = uu.flatten() * lattice_a[1] + vv.flatten() * lattice_b[1]
    L_lat[:, 2] = uu.flatten() * lattice_a[2] + vv.flatten() * lattice_b[2]
    
    # Base search space using Super-Fibonacci Spirals
    R_base = super_fibonacci_rotations(n_rot) # (n_rot, 3, 3)
    
    fa = np.linspace(0, 1.0, n_shift, endpoint=False)
    fb = np.linspace(0, 1.0, n_shift, endpoint=False)
    FA, FB = np.meshgrid(fa, fb, indexing='ij')
    
    T_base = np.zeros((n_shift**2, 3))
    T_base[:, 0] = FA.flatten() * lattice_a[0] + FB.flatten() * lattice_b[0]
    T_base[:, 1] = FA.flatten() * lattice_a[1] + FB.flatten() * lattice_b[1]
    T_base[:, 2] = FA.flatten() * lattice_a[2] + FB.flatten() * lattice_b[2]
    
    N_rot = len(R_base)
    N_shift = len(T_base)
    N_confs = N_rot * N_shift
    
    R_conf = np.repeat(R_base, N_shift, axis=0) # (N_confs, 3, 3)
    T_conf = np.tile(T_base, (N_rot, 1))        # (N_confs, 3)
    
    # Apply 6-fold symmetry: orient and shift
    R_sym = np.einsum('kij,cjl->ckil', S_sym, R_conf) # (N_confs, 6, 3, 3)
    T_sym = np.einsum('kij,cj->cki', S_sym, T_conf)   # (N_confs, 6, 3)
    
    # Tile across 9 lattice cells
    R_all = np.tile(R_sym, (1, 9, 1, 1, 1))           # (N_confs, 9, 6, 3, 3)
    T_all = np.zeros((N_confs, 9, 6, 3))
    for l in range(9):
        T_all[:, l, :, :] = T_sym + L_lat[l].reshape(1, 1, 3)
        
    # Flatten to 54 replicas
    R_all = R_all.reshape(N_confs, 54, 3, 3)
    T_all = T_all.reshape(N_confs, 54, 3)
    
    cl_transforms = pack_transforms(R_all, T_all) # (N_confs, 54, 4, 4)
    return cl_transforms, N_confs, R_conf, T_conf

def generate_transform_buffer_simple(lattice_a, lattice_b, n_rot=4, n_shift=4):
    """Simple version with 1 replica for smoke testing"""
    R_base = np.eye(3)
    T_base = np.zeros(3)
    R_all = np.array([[[R_base]]])
    T_all = np.array([[[T_base]]])
    cl_transforms = pack_transforms(R_all, T_all)
    return cl_transforms, 1, np.array([R_base]), np.array([T_base])


# ---------------------------------------------------------------------------
# Orchestration — rotation/translation grids, filtering, GPU search
# ---------------------------------------------------------------------------

def parse_lattice_vectors(s):
    parts = s.replace('lvs', '').replace(':', '').split()
    return np.array([float(x) for x in parts], dtype=np.float64).reshape(3, 3)


def pack_atoms_with_radii(mol, radius_override=None):
    from .. import elements
    natoms = mol.natoms
    atoms = np.zeros((natoms, 4), dtype=np.float32)
    atoms[:, :3] = mol.apos
    if radius_override is not None:
        atoms[:, 3] = radius_override
    else:
        for i, ename in enumerate(mol.enames):
            elem = ename.split('_')[0] if '_' in ename else ename
            atoms[i, 3] = elements.ELEMENT_DICT[elem][7] if elem in elements.ELEMENT_DICT else 1.0
    return atoms


def symmetry_mats_6fold():
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    return R.from_euler('z', angles).as_matrix()


def lattice_offsets(n_pbc_test):
    if n_pbc_test == 0:
        u = np.array([0])
    elif n_pbc_test == 1:
        u = np.array([0, -1, 1])
    elif n_pbc_test == 2:
        u = np.array([0, -1, 1, -2, 2])
    elif n_pbc_test == 3:
        u = np.array([0, -1, 1, -2, 2, -3, 3])
    else:
        raise ValueError(f'n_pbc_test={n_pbc_test} not supported (use 0–3)')
    uu, vv = np.meshgrid(u, u)
    return uu.flatten(), vv.flatten()


def lattice_shifts(lattice_a, lattice_b, n_pbc_test):
    uu, vv = lattice_offsets(n_pbc_test)
    L_lat = np.zeros((len(uu), 3), dtype=np.float64)
    L_lat[:, 0] = uu * lattice_a[0] + vv * lattice_b[0]
    L_lat[:, 1] = uu * lattice_a[1] + vv * lattice_b[1]
    L_lat[:, 2] = uu * lattice_a[2] + vv * lattice_b[2]
    return L_lat


def export_replica_indices(n_pbc_test, n_pbc_xyz, n_sym=6):
    uu, vv = lattice_offsets(n_pbc_test)
    sel_lat = [i for i, (a, b) in enumerate(zip(uu, vv)) if max(abs(a), abs(b)) <= n_pbc_xyz]
    expected = (2 * n_pbc_xyz + 1) ** 2
    if len(sel_lat) != expected:
        raise RuntimeError(f'Expected {expected} lattice cells for n_pbc_xyz={n_pbc_xyz}, got {len(sel_lat)}')
    sel = []
    if n_sym == 6:
        for l in sel_lat:
            sel.extend([l * 6 + s for s in range(6)])
    else:
        sel.extend(sel_lat)
    return np.array(sel, dtype=np.int32), len(sel)


def prune_duplicate_rotations_sym(R_arr, sym_mats, tol_rad):
    """Drop rotations equivalent under sym_mats (O(N²) in #rotations; off by default via dedup=False)."""
    if len(R_arr) == 0:
        return R_arr
    sym_mats = np.asarray(sym_mats)
    keep = [R_arr[0]]
    for R_new in R_arr[1:]:
        dup = False
        for R_ref in keep:
            SR = sym_mats @ R_new  # (6,3,3)
            M = np.einsum('ij,sjl->sil', R_ref.T, SR)
            trace = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
            if np.any(np.arccos(np.clip(0.5 * (trace - 1.0), -1.0, 1.0)) < tol_rad):
                dup = True
                break
        if not dup:
            keep.append(R_new)
    return np.asarray(keep)


def generate_rotations(mode, nrot, tilt_range=0.1, n_tilt=3, rot_tol=1e-2, do_dedup=False):
    if mode == 'full3d':
        R_raw = super_fibonacci_rotations(nrot)
    elif mode == 'inplane':
        R_raw = R.from_euler('z', np.linspace(0, 2 * np.pi, nrot, endpoint=False)).as_matrix()
    elif mode == 'tilt':
        angles_z = np.linspace(0, 2 * np.pi, nrot, endpoint=False)
        tilt_x, tilt_y = np.linspace(-tilt_range, tilt_range, n_tilt), np.linspace(-tilt_range, tilt_range, n_tilt)
        X, Y, Z = np.meshgrid(tilt_x, tilt_y, angles_z, indexing='ij')
        R_raw = R.from_euler('xyz', np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))).as_matrix()
    else:
        raise ValueError(f'Unknown rotation mode: {mode}')
    if not do_dedup:
        return R_raw
    R_unique = prune_duplicate_rotations_sym(R_raw, symmetry_mats_6fold(), rot_tol)
    return R_unique


def generate_assembly_transforms(lattice_a, lattice_b, n_rot, n_shift, shift_range=1.0, rot_mode='full3d', tilt_range=0.1, n_tilt=3, shift_region='triangle', shift_sum_max=0.8001, n_pbc_test=2, do_dedup=False, rot_tol=1e-2, apos=None, align_flat=False, n_sym=6):
    """Full pose grid: rotations × translations × 6-fold sym × PBC lattice replicas.

    n_sym=6 (default): six C6-related orientations per unit cell (experimental hexagonal SAM).
    n_sym=1: single orientation per cell, clash vs PBC neighbors only.
    align_flat: pre-rotate to flat-adsorbed frame then sample small tilts/in-plane.
    """
    L_lat = lattice_shifts(lattice_a, lattice_b, n_pbc_test)
    n_lat = len(L_lat)
    if align_flat and apos is not None:
        R_base = generate_rotations_flat(apos, rot_mode, n_rot, tilt_range, n_tilt, do_dedup=do_dedup, rot_tol=rot_tol)
    else:
        R_base = generate_rotations(rot_mode, n_rot, tilt_range, n_tilt, rot_tol, do_dedup)
    if shift_region == 'triangle':
        fa, fb = np.linspace(0.0, shift_range, n_shift), np.linspace(0.0, shift_range, n_shift)
        FA, FB = np.meshgrid(fa, fb, indexing='ij')
        fa_flat, fb_flat = FA.flatten(), FB.flatten()
        mask = (fa_flat >= 0) & (fb_flat >= 0) & ((fa_flat + fb_flat) <= shift_sum_max)
        fa_flat, fb_flat = fa_flat[mask], fb_flat[mask]
        if len(fa_flat) == 0:
            raise ValueError('Shift region mask removed all translations; increase shift_range or shift_sum_max')
    else:
        fa, fb = np.linspace(-shift_range, shift_range, n_shift), np.linspace(-shift_range, shift_range, n_shift)
        FA, FB = np.meshgrid(fa, fb, indexing='ij')
        fa_flat, fb_flat = FA.flatten(), FB.flatten()
    T_base = np.zeros((len(fa_flat), 3), dtype=np.float64)
    T_base[:, 0] = fa_flat * lattice_a[0] + fb_flat * lattice_b[0]
    T_base[:, 1] = fa_flat * lattice_a[1] + fb_flat * lattice_b[1]
    T_base[:, 2] = fa_flat * lattice_a[2] + fb_flat * lattice_b[2]
    N_rot, N_shift = len(R_base), len(T_base)
    R_conf = np.repeat(R_base, N_shift, axis=0)
    T_conf = np.tile(T_base, (N_rot, 1))
    N_confs = N_rot * N_shift
    if n_sym == 6:
        S_sym = symmetry_mats_6fold()
        R_sym = np.einsum('kij,cjl->ckil', S_sym, R_conf)
        T_sym = np.einsum('kij,cj->cki', S_sym, T_conf)
        nmols_eval = 6 * n_lat
        R_all = np.tile(R_sym[:, None, :, :, :], (1, n_lat, 1, 1, 1)).reshape(N_confs, nmols_eval, 3, 3)
        T_all = np.zeros((N_confs, n_lat, 6, 3), dtype=np.float64)
        for l in range(n_lat):
            T_all[:, l, :, :] = T_sym + L_lat[l].reshape(1, 1, 3)
        T_all = T_all.reshape(N_confs, nmols_eval, 3)
    else:
        nmols_eval = n_lat
        R_all = np.tile(R_conf[:, None, :, :], (1, n_lat, 1, 1)).reshape(N_confs, nmols_eval, 3, 3)
        T_all = np.zeros((N_confs, n_lat, 3), dtype=np.float64)
        for l in range(n_lat):
            T_all[:, l, :] = T_conf + L_lat[l].reshape(1, 3)
    return pack_transforms(R_all, T_all), N_confs, R_conf, T_conf, nmols_eval


def suggest_cell_scale(apos, cell_lvs, fill=1.75):
    """Scale factor so flat molecule footprint ~fill× unit-cell area (tight SAM contact)."""
    Rf, _ = best_flat_rotation(apos)
    xy = (apos @ Rf.T)[:, :2]
    extent = xy.max(0) - xy.min(0)
    mol_area = float(extent[0] * extent[1])
    a, b = cell_lvs[0, :2], cell_lvs[1, :2]
    cell_area = abs(a[0] * b[1] - a[1] * b[0])
    return float(np.sqrt(mol_area * fill / max(cell_area, 1e-9)))


def scale_cell_xy(cell_lvs, scale):
    out = cell_lvs.copy()
    out[0] *= scale
    out[1] *= scale
    return out


def z_spans_for_configs(R_conf, apos):
    z_coords = np.einsum('cij,aj->cai', R_conf, apos)
    return z_coords[:, :, 2].max(axis=1) - z_coords[:, :, 2].min(axis=1)


def min_z_span_oriented(apos, n=200):
    """Minimum z-extent over SO(3) — flat-adsorbed layer thickness target."""
    spans = []
    for Rm in super_fibonacci_rotations(n):
        z = (apos @ Rm.T)[:, 2]
        spans.append(float(z.max() - z.min()))
    return float(min(spans))


def best_flat_rotation(apos, n=200):
    """Rotation R_flat such that R_flat @ apos has minimal z-span (molecule lying on surface)."""
    best_R, best_span = np.eye(3), 1e9
    for Rm in super_fibonacci_rotations(n):
        z = (apos @ Rm.T)[:, 2]
        s = float(z.max() - z.min())
        if s < best_span:
            best_span, best_R = s, Rm
    return best_R, best_span


def min_z_span_inplane(apos, n=48):
    """Smallest z-extent by rotation about z (legacy; use min_z_span_oriented for SAM)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    spans = []
    for a in angles:
        Rz = R.from_euler('z', a).as_matrix()
        z = (apos @ Rz.T)[:, 2]
        spans.append(float(z.max() - z.min()))
    return float(min(spans))


def suggest_zspan_max(apos, slack=1.15):
    """Max allowed layer thickness: flat-adsorbed height + small tilt slack."""
    return min_z_span_oriented(apos) * slack


def generate_rotations_flat(apos, mode, nrot, tilt_range=0.12, n_tilt=4, n_so3=200, do_dedup=False, rot_tol=1e-2):
    """Rotations about the flat-adsorbed frame: R_total = R_pert @ R_flat."""
    R_flat, _ = best_flat_rotation(apos, n_so3)
    R_pert = generate_rotations(mode, nrot, tilt_range, n_tilt, rot_tol, do_dedup)
    return np.einsum('cij,jk->cik', R_pert, R_flat)


def assembly_composite_score(clash, z_span, min_dist, w_clash=20.0, w_z=4.0, w_pack=2.0):
    """SAM objective: no clash, minimal z-thickness, tight xy packing.

    clash     — sum of soft-sphere overlaps (Å²); must be ≈0 for rigid layer
    z_span    — molecular height; surface attraction → minimize
    min_dist  — closest inter-molecular approach; minimize for dense SAM (subject to clash=0)
    """
    return w_clash * clash + w_z * z_span + w_pack * min_dist


def run_assembly_search(mol, cell_lvs, ocl=None, *, cell_scale=1.0, nrot=16, rot_mode='tilt', n_tilt=5, tilt_range=0.25, nshift=10, shift_range=0.4, shift_region='triangle', shift_sum_max=0.8, n_pbc_test=2, n_pbc_xyz=1, n_sym=6, zspan_max=None, zspan_slack=1.15, clash_max=5.0, dist_min=1.0, dist_max=None, zpenalty=2.0, pack_weight=1.0, clash_weight=1.0, penalty=50.0, radius=1.0, export_max=100, top_k=10, dedup=False, align_flat=True, wg=128, device=0, simple=False):
    """End-to-end assembly search on a fixed experimental unit cell.

  Model: n_sym=6 places six C6-related orientations in the cell (hexagonal SAM).
  radius=1.0 Å default — softer than VdW to approximate residual flexibility in rigid search.
  Score: clash + zpenalty*z_span + pack_weight*min_dist (minimize all three).
    """
    base_atoms = pack_atoms_with_radii(mol, radius_override=radius)
    cell_lvs = np.asarray(cell_lvs, dtype=np.float64)
    if cell_scale != 1.0:
        cell_lvs = scale_cell_xy(cell_lvs, cell_scale)
    z_flat = min_z_span_oriented(mol.apos)
    if zspan_max is None:
        zspan_max = suggest_zspan_max(mol.apos, slack=zspan_slack)
    if simple:
        transforms, n_confs, R_conf, T_conf = generate_transform_buffer_simple(cell_lvs[0], cell_lvs[1])
        nmols_eval, sel_indices, nmols_out = 1, np.array([0], dtype=np.int32), 1
    else:
        transforms, n_confs, R_conf, T_conf, nmols_eval = generate_assembly_transforms(cell_lvs[0], cell_lvs[1], nrot, nshift, shift_range, rot_mode, tilt_range, n_tilt, shift_region, shift_sum_max, n_pbc_test, dedup, apos=mol.apos, align_flat=align_flat, n_sym=n_sym)
        nmols_out_req = (6 if n_pbc_xyz == 0 else 6 * ((2 * n_pbc_xyz + 1) ** 2)) if n_sym == 6 else ((2 * n_pbc_xyz + 1) ** 2)
        if nmols_out_req > nmols_eval:
            raise RuntimeError(f'n_pbc_xyz={n_pbc_xyz} n_sym={n_sym} requires {nmols_out_req} replicas but n_pbc_test={n_pbc_test} generated {nmols_eval}')
        sel_indices, nmols_out = export_replica_indices(n_pbc_test, n_pbc_xyz, n_sym=n_sym)
    z_spans = z_spans_for_configs(R_conf, mol.apos)
    z_ok = z_spans <= zspan_max
    if not np.any(z_ok):
        raise ValueError(f'All configs filtered by zspan_max={zspan_max:.2f} (flat target={z_flat:.2f}); z-span range [{z_spans.min():.2f}, {z_spans.max():.2f}]')
    transforms, R_conf, T_conf, z_spans = transforms[z_ok], R_conf[z_ok], T_conf[z_ok], z_spans[z_ok]
    n_confs = transforms.shape[0]
    if ocl is None:
        ocl = AssemblyOCL(nloc=wg, device_index=device)
    ocl.upload_base_atoms(base_atoms)
    scores, min_dists = ocl.evaluate_packing(transforms, nmols=nmols_eval, max_clash_penalty=penalty)
    total_scores = assembly_composite_score(scores, z_spans, min_dists, w_clash=clash_weight, w_z=zpenalty, w_pack=pack_weight)
    export_mask = (z_spans <= zspan_max) & (scores <= clash_max) & (min_dists >= dist_min)
    if dist_max is not None:
        export_mask &= (min_dists <= dist_max)
    export_indices = np.where(export_mask)[0]
    if export_indices.size == 0:
        export_sorted = np.array([], dtype=np.int32)
        best_indices = np.array([], dtype=np.int32)
        best_idx = int(np.argmin(total_scores))
    else:
        export_sorted = export_indices[np.argsort(total_scores[export_indices])][:export_max]
        best_indices = export_sorted[:top_k]
        best_idx = int(best_indices[0])
    return {
        'ocl': ocl, 'mol': mol, 'cell_lvs': cell_lvs, 'base_atoms': base_atoms,
        'transforms': transforms, 'R_conf': R_conf, 'T_conf': T_conf,
        'scores': scores, 'min_dists': min_dists, 'z_spans': z_spans, 'total_scores': total_scores,
        'n_confs': n_confs, 'nmols_eval': nmols_eval, 'nmols_out': nmols_out,
        'sel_indices': sel_indices, 'export_sorted': export_sorted, 'best_indices': best_indices, 'best_idx': best_idx,
        'export_mask': export_mask, 'zspan_max': zspan_max, 'z_flat': z_flat, 'n_sym': n_sym, 'cell_scale': cell_scale,
    }
