"""
RigidBodyDynamics.py — 6-DOF rigid body dynamics on the GPU.

Purpose: Simulate rigid bodies (molecules / AFM tips) with quaternion-based
rotation and translational dynamics. Force backends: GridFF, folded FAF,
and **PairFF** (pairwise intermolecular + optional fused FAF).

Key functionality:
  - Quaternion-to-rotation-matrix conversion; per-atom body-frame buffers
  - GridFF / folded FAF sampling; FIRE / Verlet / Newton on 6-DOF
  - Anchor springs on specific atoms
  - **RigidBodyPairFF**: allmol shared multi-mol layout (`from_molecules`);
    `set_active_body` = index only (persistent dynamics); `attach_pairff_faf`
    for substrate; `tip_pull_scan` / `world_sites_all_bodies` for AFM-like pulls;
    Vispy map uses `faf_fit` + `potential_to_rgba` display SSOT

Role in SPAMMM: Rigid engine for AFM manipulation (`RigidBodyAFM.py`) and
interactive PairFF docking (`demos/demo_pairff.py`). Kernels in `rigid.cl`.
"""

import os
import ast
import numpy as np
import pyopencl as cl

from ..utils.OpenCLBase import OpenCLBase
from ..topology.FFparams import load_xyz_with_REQs
from ..AtomicSystem import AtomicSystem


DEFAULT_WORKGROUP_SIZE = 32
DEFAULT_MAX_ATOMS_PER_BODY = 128
DEFAULT_ALPHA_MORSE = 1.5


def _pack_float3(arr):
    vec = np.asarray(arr, dtype=np.float32)
    if vec.shape != (3,):
        raise ValueError(f"Expected array of shape (3,) for float3, got {vec.shape}")
    out = np.zeros(4, dtype=np.float32)
    out[:3] = vec
    return out


def _ensure_float4(arr, w_value=0.0):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for float4 conversion, got shape {arr.shape}")
    if arr.shape[1] == 3:
        w = np.full((arr.shape[0], 1), np.float32(w_value), dtype=np.float32)
        arr = np.hstack((arr, w))
    if arr.shape[1] != 4:
        raise ValueError(f"Expected array with 4 columns after padding, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _ensure_cl_mat3(mat, n_bodies):
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape[:2] != (n_bodies, 3):
        raise ValueError(f"Expected inertia tensor shape (n_bodies,3,3) or (n_bodies,3,4), got {mat.shape}")
    if mat.shape[2] == 3:
        out = np.zeros((n_bodies, 3, 4), dtype=np.float32)
        out[:, :, :3] = mat
        return out
    if mat.shape[2] == 4:
        return np.ascontiguousarray(mat, dtype=np.float32)
    raise ValueError(f"Unsupported inertia tensor trailing dimension {mat.shape[2]}")


def _reqs_to_plq(reqs, alpha=DEFAULT_ALPHA_MORSE):
    """Convert REQ parameters to PLQ coefficients for GridFF sampling.
    
    CRITICAL: This function expects REQ.y to be sqrt(EvdW), NOT raw EvdW.
    If reading from ElementTypes.dat, you MUST sqrt the E value before calling this.
    
    Formula (matching C++ REQ2PLQ in Forces.h):
        e  = exp(alpha * R)
        cL = e * E              # London coefficient
        cP = e * cL = e^2 * E   # Pauli coefficient
        cH = e^2 * H           # H-bond coefficient (usually 0)
    
    The sqrt(E) convention ensures proper mixed interaction:
        Eij = sqrt(Ei * Ej) when GridFF channels contain substrate sqrt(Ej)
    
    Args:
        reqs: (n, 4) array of (R, sqrt(EvdW), Q, H) - E MUST be sqrt(EvdW)
        alpha: alphaMorse parameter (default DEFAULT_ALPHA_MORSE, must match GridFF generation)
    
    Returns:
        (n, 4) array of PLQ coefficients (cP, cL, Q, cH)
    """
    reqs = np.asarray(reqs, dtype=np.float32)
    if reqs.ndim != 2 or reqs.shape[1] != 4:
        raise ValueError(f"Expected REQs shape (n,4), got {reqs.shape}")
    e = np.exp(alpha * reqs[:, 0]).astype(np.float32)
    cL = e * reqs[:, 1]
    cP = e * cL
    cH = e * e * reqs[:, 3]
    out = np.zeros_like(reqs, dtype=np.float32)
    out[:, 0] = cP
    out[:, 1] = cL
    out[:, 2] = reqs[:, 2]
    out[:, 3] = cH
    return out


def _plq_to_coeffs(plq):
    plq = np.asarray(plq, dtype=np.float32)
    if plq.ndim != 2 or plq.shape[1] != 4:
        raise ValueError(f"Expected PLQ shape (n,4), got {plq.shape}")
    return {
        'Pauli': plq[:, 0],
        'London': plq[:, 1],
        'Coulomb': plq[:, 2],
        'Hb': plq[:, 3],
    }


def _guess_mass(enames):
    mass_table = {
        'H': 1.0079, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Na': 22.990,
        'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974, 'S': 32.06, 'Cl': 35.45,
        'K': 39.098, 'Ca': 40.078, 'Br': 79.904, 'I': 126.904,
    }
    masses = np.zeros(len(enames), dtype=np.float32)
    for i, e in enumerate(enames):
        if e not in mass_table:
            raise KeyError(f"Missing atomic mass for element '{e}'")
        masses[i] = mass_table[e]
    return masses


def _load_npy_legacy(fname):
    with open(fname, 'rb') as f:
        magic = f.read(6)
        if magic != b'\x93NUMPY':
            raise ValueError(f"Unsupported grid file magic in {fname}")
        major = int.from_bytes(f.read(1), 'little')
        minor = int.from_bytes(f.read(1), 'little')
        if major == 1:
            hlen = int.from_bytes(f.read(2), 'little')
        elif major in (2, 3):
            hlen = int.from_bytes(f.read(4), 'little')
        else:
            raise ValueError(f"Unsupported npy version {(major, minor)} in {fname}")
        header = f.read(hlen).decode('latin1').strip()
        if not header:
            raise ValueError(f"Empty npy header in {fname}")
        meta = ast.literal_eval(header)
        descr = meta.get('descr', None)
        shape = meta.get('shape', None)
        fortran_order = bool(meta.get('fortran_order', False))
        if descr is None or shape is None:
            raise ValueError(f"Incomplete npy header in {fname}: {meta}")
        dtype = np.dtype(descr)
        count = int(np.prod(shape))
        data = np.fromfile(f, dtype=dtype, count=count)
        if data.size != count:
            raise ValueError(f"Unexpected data size in {fname}: expected {count}, got {data.size}")
        arr = data.reshape(shape, order='F' if fortran_order else 'C')
        return arr


def compute_mass_properties(rel_positions, masses):
    rel = np.asarray(rel_positions, dtype=np.float32)
    m = np.asarray(masses, dtype=np.float32)
    if rel.ndim != 2 or rel.shape[1] != 3:
        raise ValueError(f"Expected relative positions shape (n,3), got {rel.shape}")
    if m.shape != (rel.shape[0],):
        raise ValueError(f"Expected masses shape ({rel.shape[0]},), got {m.shape}")
    mtot = float(m.sum())
    if mtot <= 0.0:
        raise ValueError(f"Non-positive total mass {mtot}")
    I = np.zeros((3, 3), dtype=np.float32)
    for mi, r in zip(m, rel):
        rr = np.dot(r, r)
        I += mi * (rr * np.eye(3, dtype=np.float32) - np.outer(r, r).astype(np.float32))
    det = np.linalg.det(I)
    if abs(det) < 1e-10:
        raise ValueError(f"Singular inertia tensor det={det}")
    Iinv = np.linalg.inv(I).astype(np.float32)
    return np.float32(mtot), I.astype(np.float32), Iinv


def _quat_to_matrix_np(q):
    q = np.asarray(q, dtype=np.float32)
    if q.shape == (4,):  # Single quaternion
        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float32)
    elif q.ndim == 2 and q.shape[1] == 4:  # Multiple quaternions (N, 4)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        # Return (N, 3, 3) array of rotation matrices
        return np.stack([
            np.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=1),
            np.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=1),
            np.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=1),
        ], axis=2).astype(np.float32)
    else:
        raise ValueError(f"Quaternion must have shape (4,) or (N,4), got {q.shape}")


class RigidBodyDynamics(OpenCLBase):
    """
    Simple pyOpenCL wrapper around `rigid_body_dynamics_kernel`.
    Each rigid body is simulated within a single workgroup.
    """

    def __init__(self, nloc=DEFAULT_WORKGROUP_SIZE, max_atoms=DEFAULT_MAX_ATOMS_PER_BODY, debug=False):
        if nloc != DEFAULT_WORKGROUP_SIZE:
            raise ValueError(f"Kernel expects workgroup size {DEFAULT_WORKGROUP_SIZE}, got {nloc}")
        super().__init__(nloc=nloc, device_index=0)
        kernel_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../kernels')
        build_options = ['-D', f'RIGID_DBG={1 if debug else 0}']
        kernel_paths = [
            os.path.join(kernel_dir, 'common.cl'),
            os.path.join(kernel_dir, 'Forces.cl'),
            os.path.join(kernel_dir, 'rigid.cl'),
        ]
        if not self.load_program_multi(kernel_paths, build_options=build_options, bMakeHeaders=False):
            raise RuntimeError("Failed to load rigid body kernels")

        self.debug = bool(debug)
        self.max_atoms_per_body = max_atoms
        self.n_bodies = 0
        self.n_replicas = 0
        self.num_atoms = 0
        self.total_atoms = 0
        self.atom_counts = None
        self.mol_offsets = None
        self.max_atoms_body = 0

        self.kernelheaders = {
            "rigid_body_dynamics_kernel": """__kernel
void rigid_body_dynamics_kernel(
    __global const int*      mols,
    __global float4*         poss,
    __global float4*         qrots,
    __global float4*         vposs,
    __global float4*         vrots,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global float4*         apos_world,
    __global const float4*   anchors,
    const int   natoms,
    const int   niter,
    const float dt,
    const float4             md_params,
    const float3  Efield
)""",
            "rigid_body_gridff_kernel": """__kernel
void rigid_body_gridff_kernel(
    __global const int*      mols,
    __global float4*         poss,
    __global float4*         qrots,
    __global float4*         vposs,
    __global float4*         vrots,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global float4*         apos_world,
    __global const float4*   atom_PLQ,
    __global const float4*   BsplinePLQ,
    __global float4*         atom_force,
    __global float4*         body_force,
    __global float4*         body_torque,
    __global const float4*   anchors,
    const int4               grid_ns,
    const float4             grid_invStep,
    const float4             grid_p0,
    const float              dt,
    const float4             md_params,
    const int                niter
)""",
            "rigid_body_folded_kernel": """__kernel
void rigid_body_folded_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float              dt,
    const float4             md_params,
    const int                niter
)""",
            "rigid_body_folded_replicas_kernel": """__kernel
void rigid_body_folded_replicas_kernel(
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float              dt,
    const float4             md_params,
    const int                niter
)""",
            "rigid_body_folded_newton_kernel": """__kernel
void rigid_body_folded_newton_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   newton_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float4             newton_params,
    const float              f2tol,
    const int                niter
)""",
            "rigid_body_folded_newton_replicas_kernel": """__kernel
void rigid_body_folded_newton_replicas_kernel(
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   newton_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float4             newton_params,
    const float              f2tol,
    const int                niter
)"""
        }

        self.kernel_params = {}
        self.kernel_args = None
        self.gridff_args = None
        self.folded_args = None
        self.replicas_args = None
        self.newton_args = None
        self.newton_replicas_args = None
        self.krnl_folded_replicas = None
        self.krnl_newton = None
        self.krnl_newton_replicas = None
        self.grid_shape = None
        self.grid_data = None
        self.grid_p0 = None
        self.grid_step = None
        self.atom_PLQ = None
        self.atom_REQ = None
        self.folded_params = None
        self.folded_atom_type_ids = None
        self.enames = None
        self.atom_masses = None
        self.atom_types_assigned = None
        self.last_atom_force = None
        self.last_body_force = None
        self.last_body_torque = None
        self.atom_body_host = None
        self.mass_total = None
        self.inertia_inv_host = None
        self.inertia_host = None

    def realloc(self, n_bodies, num_atoms):
        if num_atoms > self.max_atoms_per_body:
            raise ValueError(f"num_atoms={num_atoms} exceeds max_atoms_per_body={self.max_atoms_per_body}")
        self.n_bodies = int(n_bodies)
        self.num_atoms = int(num_atoms)
        self.total_atoms = self.n_bodies * self.num_atoms

        float_size = np.float32().itemsize
        int_size = np.int32().itemsize
        mat3_size = 3 * 4 * float_size
        atom_block_size = self.max_atoms_per_body * 4 * float_size  # kept for compatibility, actual total handled per-body
        mf = cl.mem_flags
        bytes_per_body = 4 * float_size

        self.create_buffer('mols', (self.n_bodies + 1) * int_size, mf.READ_ONLY)
        self.create_buffer('poss',   self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('qrots',  self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('vposs',  self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('vrots',  self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('fire_state', self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('newton_state', self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('I_body_inv', self.n_bodies * mat3_size,      mf.READ_ONLY)
        self.create_buffer('I_body',     self.n_bodies * mat3_size,      mf.READ_ONLY)
        self.create_buffer('anchors', self.total_atoms * 4 * float_size, mf.READ_ONLY)

        total_atom_bytes = self.total_atoms * 4 * float_size
        self.create_buffer('apos_body',  total_atom_bytes, mf.READ_ONLY)
        self.create_buffer('apos_world', total_atom_bytes, mf.READ_WRITE)
        self.create_buffer('atom_PLQ',   total_atom_bytes, mf.READ_ONLY)
        self.create_buffer('atom_force', total_atom_bytes, mf.READ_WRITE)
        self.create_buffer('body_force', self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('body_torque', self.n_bodies * bytes_per_body, mf.READ_WRITE)

        # Folded basis buffers (allocated on demand by init_folded)
        FOLDED_BASIS_MAX = 128
        FOLDED_TYPES_MAX = 8
        self.create_buffer('folded_coeffs',  FOLDED_TYPES_MAX * FOLDED_BASIS_MAX * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('folded_kxyz',    FOLDED_BASIS_MAX * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('folded_atom_type', self.total_atoms * int_size, mf.READ_ONLY)

        self.kernel_params = {
            'natoms': np.int32(self.total_atoms),
            'niter': np.int32(1),
            'dt': np.float32(0.01),
            'Efield': np.zeros(4, dtype=np.float32),
            'md_params': np.array([0.92, 0.88, 1.0, 1.0], dtype=np.float32),
        }
        self.kernel_args = self.generate_kernel_args("rigid_body_dynamics_kernel")
        self.gridff_args = None

    def realloc_molecules(self, atom_counts):
        """Allocate multi-molecule buffers with irregular per-mol site counts (flat packing).

        mols[i0:i1] offsets into apos_body / anchors / dyn_*; n_bodies = n_mols poses.
        total_atoms = sum(atom_counts). Unlike realloc(), does not assume uniform num_atoms.
        """
        counts = np.asarray(atom_counts, dtype=np.int32).ravel()
        if counts.ndim != 1 or counts.size < 1:
            raise ValueError("atom_counts must be a non-empty 1D array")
        if np.any(counts <= 0):
            raise ValueError(f"atom_counts must be positive, got {counts}")
        if int(counts.max()) > self.max_atoms_per_body:
            raise ValueError(f"max(atom_counts)={int(counts.max())} exceeds max_atoms_per_body={self.max_atoms_per_body}")
        n_mols = int(counts.shape[0])
        total = int(counts.sum())
        self.n_bodies = n_mols
        self.num_atoms = int(counts.max())  # legacy field: max sites/mol
        self.total_atoms = total
        self.atom_counts = counts.copy()
        offsets = np.zeros(n_mols + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts)
        self.mol_offsets = offsets

        float_size = np.float32().itemsize
        int_size = np.int32().itemsize
        mat3_size = 3 * 4 * float_size
        mf = cl.mem_flags
        bytes_per_body = 4 * float_size

        self.create_buffer('mols', (self.n_bodies + 1) * int_size, mf.READ_ONLY)
        self.create_buffer('poss',   self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('qrots',  self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('vposs',  self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('vrots',  self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('fire_state', self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('newton_state', self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('I_body_inv', self.n_bodies * mat3_size, mf.READ_ONLY)
        self.create_buffer('I_body',     self.n_bodies * mat3_size, mf.READ_ONLY)
        self.create_buffer('anchors', self.total_atoms * 4 * float_size, mf.READ_ONLY)
        total_atom_bytes = self.total_atoms * 4 * float_size
        self.create_buffer('apos_body',  total_atom_bytes, mf.READ_ONLY)
        self.create_buffer('apos_world', total_atom_bytes, mf.READ_WRITE)
        self.create_buffer('atom_PLQ',   total_atom_bytes, mf.READ_ONLY)
        self.create_buffer('atom_force', total_atom_bytes, mf.READ_WRITE)
        self.create_buffer('body_force', self.n_bodies * bytes_per_body, mf.READ_WRITE)
        self.create_buffer('body_torque', self.n_bodies * bytes_per_body, mf.READ_WRITE)
        FOLDED_BASIS_MAX = 128
        FOLDED_TYPES_MAX = 8
        self.create_buffer('folded_coeffs',  FOLDED_TYPES_MAX * FOLDED_BASIS_MAX * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('folded_kxyz',    FOLDED_BASIS_MAX * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('folded_atom_type', self.total_atoms * int_size, mf.READ_ONLY)
        self.create_buffer('dyn_REQ',  self.total_atoms * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('dyn_type', self.total_atoms * int_size, mf.READ_ONLY)

        self.kernel_params = {
            'natoms': np.int32(self.total_atoms),
            'niter': np.int32(1),
            'dt': np.float32(0.01),
            'Efield': np.zeros(4, dtype=np.float32),
            'md_params': np.array([0.92, 0.88, 1.0, 1.0], dtype=np.float32),
            'n_mols': np.int32(n_mols),
            'active_mol': np.int32(0),
        }
        self.toGPU('mols', offsets)
        self.kernel_args = None
        self.gridff_args = None

    def realloc_replicas(self, n_replicas, num_atoms):
        """Allocate buffers for the replicas kernel (many copies of same molecule).

        Reuses the standard buffer names but with different semantics:
        - poss/qrots/vposs/vrots: [n_replicas] per-replica state
        - apos_body: [num_atoms] shared body-frame positions (not [n_replicas * na])
        - I_body_inv/I_body: [1] shared inertia (not [n_replicas])
        - folded_atom_type: [num_atoms] shared (not [n_replicas * na])
        - anchors: [n_replicas * num_atoms] per-replica spring targets
        """
        # Reuse standard realloc with n_bodies=n_replicas, then override sizes
        # that differ. The buffers are oversized for shared data but that's harmless.
        self.realloc(n_bodies=n_replicas, num_atoms=num_atoms)
        self.n_replicas = n_replicas

    def upload_replicas_state(self, pos, quats, lin_mom, ang_mom, mass, inertia_inv, atom_pos_body, anchors=None, inertia=None):
        """Upload per-replica state + shared molecule data for the replicas kernel.

        Args:
            pos:  (n_replicas, 4) float32 — CoM position + mass
            quats: (n_replicas, 4) float32
            lin_mom: (n_replicas, 4) float32
            ang_mom: (n_replicas, 4) float32
            mass: float (used for pos.w, same for all replicas)
            inertia_inv: (3, 3) float32 — single shared inverse inertia
            atom_pos_body: (num_atoms, 3/4) float32 — shared body-frame positions
            anchors: (n_replicas, num_atoms, 4) or None
            inertia: (3, 3) float32 — single shared inertia (optional)
        """
        pos_in   = _ensure_float4(pos)
        quats_in = _ensure_float4(quats)
        lin_in   = _ensure_float4(lin_mom)
        ang_in   = _ensure_float4(ang_mom)
        n_rep = self.n_replicas
        na = self.num_atoms

        # Per-replica state
        self.toGPU('poss',  pos_in)
        self.toGPU('qrots', quats_in)
        self.toGPU('vposs', lin_in)
        self.toGPU('vrots', ang_in)

        # Shared inertia (upload as [1] cl_Mat3, buffer is [n_rep] but kernel reads [0])
        Iinv_cl = _ensure_cl_mat3(inertia_inv[None, :, :], 1)
        self.toGPU('I_body_inv', Iinv_cl)
        if inertia is not None:
            I_cl = _ensure_cl_mat3(inertia[None, :, :], 1)
            self.toGPU('I_body', I_cl)

        # Shared body-frame atom positions (upload [na], buffer is [n_rep*na] but kernel reads [na])
        atoms = np.asarray(atom_pos_body, dtype=np.float32)
        if atoms.ndim == 3:
            atoms = atoms[0]  # take first molecule if (1, na, 3/4)
        if atoms.shape[1] == 3:
            atoms = np.concatenate([atoms, np.zeros((atoms.shape[0], 1), dtype=np.float32)], axis=1)
        atoms_flat = atoms[:na]  # just the first na atoms
        self.atom_body_host = atoms_flat.copy()
        self.toGPU('apos_body', atoms_flat)

        # Anchors (per-replica)
        if anchors is not None:
            anc = np.asarray(anchors, dtype=np.float32).reshape(n_rep * na, 4)
        else:
            anc = np.zeros((n_rep * na, 4), dtype=np.float32)
            anc[:, 3] = -1.0
        self.toGPU('anchors', anc)

        # Zero output buffers
        self.toGPU('apos_world',   np.zeros((n_rep * na, 4), dtype=np.float32))
        self.toGPU('atom_force',   np.zeros((n_rep * na, 4), dtype=np.float32))
        self.toGPU('body_force',   np.zeros((n_rep, 4), dtype=np.float32))
        self.toGPU('body_torque',  np.zeros((n_rep, 4), dtype=np.float32))
        self.reset_optimizer_state(finish=False)
        self.queue.finish()

    def upload_state(self, pos, quats, lin_mom, ang_mom, mass, inv_mass, inertia_inv, atom_pos_body, anchors=None, atom_PLQ=None, inertia=None):
        if self.n_bodies == 0:
            raise RuntimeError("Call realloc() before uploading state")

        pos_in   = _ensure_float4(pos)
        quats_in = _ensure_float4(quats)
        lin_in   = _ensure_float4(lin_mom)
        ang_in   = _ensure_float4(ang_mom)

        inertia_inv = _ensure_cl_mat3(inertia_inv, self.n_bodies)

        atoms = np.asarray(atom_pos_body, dtype=np.float32)
        if atoms.shape != (self.n_bodies, self.num_atoms, 3) and atoms.shape != (self.n_bodies, self.num_atoms, 4):
            raise ValueError(f"Expected body atom positions shape ({self.n_bodies},{self.num_atoms},3/4), got {atoms.shape}")

        if atoms.shape[2] == 3:
            pad = np.zeros((self.n_bodies, self.num_atoms, 1), dtype=np.float32)
            atoms = np.concatenate((atoms, pad), axis=2)

        atoms_body = atoms.reshape(self.total_atoms, 4)
        self.atom_body_host = atoms_body.copy()
        self.mass_total = float(pos_in[0, 3]) if len(pos_in) else None
        self.inertia_inv_host = inertia_inv.copy()
        if inertia is not None:
            inertia_cl = _ensure_cl_mat3(inertia, self.n_bodies)
            self.inertia_host = inertia_cl.copy()
            for b in range(self.n_bodies):
                Iinv_b = inertia_inv[b, :, :3]
                I_b = inertia_cl[b, :, :3]
                prod = Iinv_b @ I_b
                err = float(np.max(np.abs(prod - np.eye(3, dtype=np.float32))))
                if err > 1e-4:
                    raise ValueError(f"I_body_inv @ I_body != I for body {b}: max error {err:.2e}. Ensure I_body and I_body_inv are consistent.")
        else:
            self.inertia_host = None

        mols = np.arange(0, self.total_atoms + 1, self.num_atoms, dtype=np.int32)

        self.toGPU('mols', mols)
        self.toGPU('poss', pos_in)
        self.toGPU('qrots', quats_in)
        self.toGPU('vposs', lin_in)
        self.toGPU('vrots', ang_in)
        self.toGPU('I_body_inv', inertia_inv)
        if self.inertia_host is not None:
            self.toGPU('I_body', self.inertia_host)
        self.toGPU('apos_body', atoms_body)
        if atom_PLQ is not None:
            plq = _ensure_float4(atom_PLQ)
            if plq.shape[0] != self.total_atoms:
                raise ValueError(f"atom_PLQ length {plq.shape[0]} does not match total atoms {self.total_atoms}")
            self.atom_PLQ = plq.copy()
            self.toGPU('atom_PLQ', self.atom_PLQ)

        # GPU already recomputes apos_world from apos_body+qrots in every kernel step.
        # No need to precompute on CPU - just upload zeros; kernel overwrites on first step.
        world_atoms_flat = np.zeros((self.total_atoms, 4), dtype=np.float32)
        world_atoms_flat[:, 3] = atoms_body[:, 3]  # preserve w (charge/mass)
        # NOTE: CPU backup below (kept for reference/debugging)
        # atoms  = atoms_body.reshape(self.n_bodies, self.num_atoms, 4)
        # rot_mats = _quat_to_matrix_np(quats_in)              # (n_bodies, 3, 3)
        # rotated = np.einsum('bij,bkj->bik', atoms[:, :, :3], rot_mats)
        # world_atoms_flat[:, :3] = (rotated + pos_in[:, :3][:, None, :]).reshape(self.total_atoms, 3)

        if anchors is None:
            anchors = np.zeros_like(world_atoms_flat)
            anchors[:, 3] = -1.0
        else:
            anchors = _ensure_float4(anchors, w_value=-1.0)
            if anchors.shape[0] != self.total_atoms:
                raise ValueError(f"anchors array length {anchors.shape[0]} does not match total atoms {self.total_atoms}")
            anchors = anchors.copy()
        
        self.anchors = anchors
        self.upload_anchors()

        self.toGPU('apos_world', world_atoms_flat)
        self.toGPU('atom_force', np.zeros_like(world_atoms_flat))
        self.toGPU('body_force', np.zeros((self.n_bodies, 4), dtype=np.float32))
        self.toGPU('body_torque', np.zeros((self.n_bodies, 4), dtype=np.float32))
        self.reset_optimizer_state(finish=False)
        self.queue.finish()

    def reset_optimizer_state(self, finish=True):
        """Reset persistent FIRE and Newton adaptation after changing the physical problem."""
        z = np.zeros((self.n_bodies, 4), dtype=np.float32)
        self.toGPU('fire_state', z)
        self.toGPU('newton_state', z)
        if finish:
            self.queue.finish()

    def upload_anchors(self):
        self.toGPU('anchors', self.anchors)

    def init_gridff(self, bspline_data, grid_p0, grid_step):
        arr = np.asarray(bspline_data)
        if arr.ndim != 4 or arr.shape[3] not in (3, 4):
            raise ValueError(f"Expected Bspline grid shape (nx,ny,nz,3/4), got {arr.shape}")
        if arr.shape[3] == 3:
            tmp = np.zeros(arr.shape[:3] + (4,), dtype=np.float32)
            tmp[..., :3] = arr.astype(np.float32)
            arr = tmp
        else:
            arr = arr.astype(np.float32, copy=False)
        self.grid_shape = tuple(int(v) for v in arr.shape[:3])
        self.grid_data = np.ascontiguousarray(arr, dtype=np.float32)
        self.grid_p0 = np.array([*grid_p0, 0.0], dtype=np.float32)
        self.grid_step = np.array([*grid_step, 0.0], dtype=np.float32)
        self.buffer_dict['BsplinePLQ'] = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.grid_data)
        self.kernel_params['grid_ns'] = np.array([*self.grid_shape, 0], dtype=np.int32)
        self.kernel_params['grid_invStep'] = np.array([1.0 / v for v in grid_step] + [0.0], dtype=np.float32)
        self.kernel_params['grid_p0'] = self.grid_p0
        self.kernel_params['md_params'] = np.array([0.92, 0.88, 1.0, 1.0], dtype=np.float32)
        self.gridff_args = self.generate_kernel_args("rigid_body_gridff_kernel")
        self.krnl_gridff = cl.Kernel(self.prg, "rigid_body_gridff_kernel")

    def run(self, num_steps, dt, efield=None, lin_damp=0.92, ang_damp=0.88, fire=False):
        if self.kernel_args is None:
            raise RuntimeError("Kernel arguments not initialized; call realloc() first")

        self.kernel_params['niter'] = np.int32(num_steps)
        self.kernel_params['dt'] = np.float32(dt)
        # md_params.w < 0 enables FIRE (v·F / ω·τ quench) in rigid.cl
        self.kernel_params['md_params'] = np.array([lin_damp, ang_damp, 1.0, -1.0 if fire else 1.0], dtype=np.float32)
        if efield is not None:
            self.kernel_params['Efield'] = _pack_float3(efield)
        self.kernel_args = self.generate_kernel_args("rigid_body_dynamics_kernel")

        global_size = (self.roundUpGlobalSize(self.n_bodies * self.nloc),)
        local_size = (self.nloc,)

        self.prg.rigid_body_dynamics_kernel(self.queue, global_size, local_size, *self.kernel_args)
        self.queue.finish()

    def run_gridff(self, num_steps, dt, lin_damp=0.92, ang_damp=0.88, force_scale=1.0, torque_scale=1.0, fire=False):
        if self.gridff_args is None:
            raise RuntimeError("GridFF kernel arguments not initialized; call init_gridff(...) first")
        self.kernel_params['dt'] = np.float32(dt)
        self.kernel_params['niter'] = np.int32(num_steps)
        if fire:
            self.kernel_params['md_params'] = np.array([lin_damp, ang_damp, 1.0, -1.0], dtype=np.float32)
        else:
            self.kernel_params['md_params'] = np.array([lin_damp, ang_damp, force_scale, torque_scale], dtype=np.float32)
        self.gridff_args = self.generate_kernel_args("rigid_body_gridff_kernel")
        global_size = (self.roundUpGlobalSize(self.n_bodies * self.nloc),)
        local_size = (self.nloc,)
        self.krnl_gridff(self.queue, global_size, local_size, *self.gridff_args)
        self.queue.finish()

    def run_folded(self, num_steps, dt, lin_damp=0.92, ang_damp=0.88, force_scale=1.0, torque_scale=1.0, fire=False):
        if self.folded_args is None:
            raise RuntimeError("Folded kernel arguments not initialized; call init_folded(...) first")
        self.kernel_params['dt'] = np.float32(dt)
        self.kernel_params['niter'] = np.int32(num_steps)
        if fire:
            self.kernel_params['md_params'] = np.array([lin_damp, ang_damp, 1.0, -1.0], dtype=np.float32)
        else:
            self.kernel_params['md_params'] = np.array([lin_damp, ang_damp, force_scale, torque_scale], dtype=np.float32)
        self.folded_args = self.generate_kernel_args("rigid_body_folded_kernel")
        global_size = (self.roundUpGlobalSize(self.n_bodies * self.nloc),)
        local_size = (self.nloc,)
        self.krnl_folded(self.queue, global_size, local_size, *self.folded_args)
        self.queue.finish()

    def run_folded_replicas(self, num_steps, dt, lin_damp=0.92, ang_damp=0.88, force_scale=1.0, torque_scale=1.0, fire=False):
        if self.replicas_args is None:
            raise RuntimeError("Replicas kernel not initialized; call init_replicas(...) first")
        self.kernel_params['dt'] = np.float32(dt)
        self.kernel_params['niter'] = np.int32(num_steps)
        if fire:
            self.kernel_params['md_params'] = np.array([lin_damp, ang_damp, 1.0, -1.0], dtype=np.float32)
        else:
            self.kernel_params['md_params'] = np.array([lin_damp, ang_damp, force_scale, torque_scale], dtype=np.float32)
        self.replicas_args = self.generate_kernel_args("rigid_body_folded_replicas_kernel")
        REPLICAS_WG = 128
        global_size = (int(np.ceil(self.n_replicas / REPLICAS_WG) * REPLICAS_WG),)
        local_size = (REPLICAS_WG,)
        self.krnl_folded_replicas(self.queue, global_size, local_size, *self.replicas_args)
        self.queue.finish()

    def init_folded(self, folded_coeffs, folded_kxyz, folded_atom_type, folded_lvec2d, folded_meta=None):
        """Initialize folded basis surface interaction for rigid body dynamics.

        Args:
            folded_coeffs: (ntypes, nbasis) float32 — fitted coefficients per atom type
            folded_kxyz:   (nbasis, 4) float32 — basis params (ku, kv, alpha, z0) per basis function
            folded_atom_type: (natoms,) int32 — type index per atom in the rigid body
            folded_lvec2d: (4,) float32 — 2D lattice vectors as (ax, bx, ay, by)
            folded_meta: (4,) int32 — (nbasis, ntypes, 0, 0). If None, inferred from shapes.
        """
        coeffs = np.asarray(folded_coeffs, dtype=np.float32)
        kxyz   = np.asarray(folded_kxyz,   dtype=np.float32)
        atype  = np.asarray(folded_atom_type, dtype=np.int32)
        lvec2d = np.asarray(folded_lvec2d, dtype=np.float32)
        ntypes, nbasis = coeffs.shape
        if kxyz.shape[0] < nbasis:
            raise ValueError(f"folded_kxyz has {kxyz.shape[0]} basis params but coeffs expects {nbasis}")
        if atype.shape[0] != self.total_atoms:
            raise ValueError(f"folded_atom_type length {atype.shape[0]} != total_atoms {self.total_atoms}")
        if folded_meta is None:
            folded_meta = np.array([nbasis, ntypes, 0, 0], dtype=np.int32)
        else:
            folded_meta = np.asarray(folded_meta, dtype=np.int32)
        # Pad to GPU buffer sizes
        FOLDED_BASIS_MAX = 128
        FOLDED_TYPES_MAX = 8
        if nbasis > FOLDED_BASIS_MAX:
            raise ValueError(f"nbasis={nbasis} exceeds FOLDED_BASIS_MAX={FOLDED_BASIS_MAX}")
        if ntypes > FOLDED_TYPES_MAX:
            raise ValueError(f"ntypes={ntypes} exceeds FOLDED_TYPES_MAX={FOLDED_TYPES_MAX}")
        coeff_pad = np.zeros(FOLDED_TYPES_MAX * FOLDED_BASIS_MAX * 4, dtype=np.float32)
        coeff_flat = np.asarray(coeffs, dtype=np.float32).reshape(ntypes, -1)[:, :nbasis]
        coeff_pad[:ntypes * nbasis] = coeff_flat.flatten()
        kxyz_pad = np.zeros((FOLDED_BASIS_MAX, 4), dtype=np.float32)
        kxyz_pad[:nbasis, :] = kxyz[:nbasis]
        self.toGPU('folded_coeffs',    coeff_pad)
        self.toGPU('folded_kxyz',      kxyz_pad)
        self.toGPU('folded_atom_type', atype)
        self.kernel_params['folded_meta']   = folded_meta
        self.kernel_params['folded_lvec2d'] = lvec2d
        self.kernel_params['md_params']     = np.array([0.92, 0.88, 1.0, 1.0], dtype=np.float32)
        self.folded_params = {
            'coeffs': coeffs.copy(), 'kxyz': kxyz[:nbasis].copy(),
            'atom_type': atype.copy(), 'lvec2d': lvec2d.copy(), 'meta': folded_meta.copy(),
        }
        self.folded_atom_type_ids = atype.copy()
        self.folded_args = self.generate_kernel_args("rigid_body_folded_kernel")
        self.krnl_folded = cl.Kernel(self.prg, "rigid_body_folded_kernel")
        self.kernel_params.setdefault('newton_params', np.array([0.1, 0.1, 0.5, 1e-2], dtype=np.float32))
        self.kernel_params.setdefault('f2tol', np.float32(1e-10))
        self.newton_args = self.generate_kernel_args("rigid_body_folded_newton_kernel")
        self.krnl_newton = cl.Kernel(self.prg, "rigid_body_folded_newton_kernel")

    def run_folded_newton(self, niter=80, eps_t=0.1, eps_r=0.1, trust0=0.5, lambda0=1e-2, f_tol=1e-5, t_tol=1e-5):
        """Pure GPU trust-region Newton (one kernel launch, no host FD).

        All Hessian FD + 6×6 LM solve + trust steps run inside
        ``rigid_body_folded_newton_kernel``. Hessian lives in __local.
        ``lambda0`` is both the initial LM damping and its lower bound.
        """
        if self.newton_args is None:
            raise RuntimeError("Newton kernel not initialized; call init_folded(...) first")
        f2tol = float(min(f_tol * f_tol, t_tol * t_tol))
        self.kernel_params['newton_params'] = np.array([eps_t, eps_r, trust0, lambda0], dtype=np.float32)
        self.kernel_params['f2tol'] = np.float32(f2tol)
        self.kernel_params['niter'] = np.int32(niter)
        self.newton_args = self.generate_kernel_args("rigid_body_folded_newton_kernel")
        global_size = (self.roundUpGlobalSize(self.n_bodies * self.nloc),)
        local_size = (self.nloc,)
        self.krnl_newton(self.queue, global_size, local_size, *self.newton_args)
        self.queue.finish()
        out = self.download_outputs()
        F = out['body_force'][0, :3]
        T = out['body_torque'][0, :3]
        E = float(out['atom_positions'][0][:, 3].sum())
        return {
            'iters': int(niter),
            'converged': float(np.linalg.norm(F)) < f_tol and float(np.linalg.norm(T)) < t_tol,
            'F': float(np.linalg.norm(F)), 'T': float(np.linalg.norm(T)), 'E': E,
            'out': out,
        }

    def run_folded_newton_replicas(self, niter=80, eps_t=0.1, eps_r=0.1, trust0=0.5, lambda0=1e-2, f_tol=1e-5, t_tol=1e-5):
        """Centered-FD GPU Newton for many replicas; lambda0 is the persistent LM floor."""
        if self.newton_replicas_args is None:
            raise RuntimeError("Newton replicas kernel not initialized; call init_replicas(...) first")
        f2tol = float(min(f_tol * f_tol, t_tol * t_tol))
        self.kernel_params['newton_params'] = np.array([eps_t, eps_r, trust0, lambda0], dtype=np.float32)
        self.kernel_params['f2tol'] = np.float32(f2tol)
        self.kernel_params['niter'] = np.int32(niter)
        self.newton_replicas_args = self.generate_kernel_args("rigid_body_folded_newton_replicas_kernel")
        REPLICAS_WG = 128
        global_size = (int(np.ceil(self.n_replicas / REPLICAS_WG) * REPLICAS_WG),)
        local_size = (REPLICAS_WG,)
        self.krnl_newton_replicas(self.queue, global_size, local_size, *self.newton_replicas_args)
        self.queue.finish()
        return self.download_outputs()

    def init_replicas(self, n_replicas, folded_coeffs, folded_kxyz, folded_atom_type, folded_lvec2d, folded_meta=None):
        """Initialize the replicas kernel for many copies of the same molecule.

        Unlike init_folded(), this kernel uses 1 thread per replica (no workgroup
        per body). Shared molecule data (apos_body, inertia, basis) is loaded to
        local memory once per workgroup. Per-replica state stays in registers.

        Args:
            n_replicas: number of rigid body replicas
            folded_coeffs: (ntypes, nbasis) float32
            folded_kxyz: (nbasis, 4) float32
            folded_atom_type: (natoms,) int32 — type index per atom (shared)
            folded_lvec2d: (4,) float32 — (ax, bx, ay, by)
            folded_meta: (4,) int32 — (nbasis, ntypes, na, n_replicas). If None, inferred.
        """
        if self.n_bodies == 0:
            raise RuntimeError("Call realloc() before init_replicas()")
        coeffs = np.asarray(folded_coeffs, dtype=np.float32)
        kxyz   = np.asarray(folded_kxyz,   dtype=np.float32)
        atype  = np.asarray(folded_atom_type, dtype=np.int32)
        lvec2d = np.asarray(folded_lvec2d, dtype=np.float32)
        ntypes, nbasis = coeffs.shape
        na = atype.shape[0]
        if kxyz.shape[0] < nbasis:
            raise ValueError(f"folded_kxyz has {kxyz.shape[0]} basis params but coeffs expects {nbasis}")
        if folded_meta is None:
            folded_meta = np.array([nbasis, ntypes, na, n_replicas], dtype=np.int32)
        else:
            folded_meta = np.asarray(folded_meta, dtype=np.int32)
        FOLDED_BASIS_MAX = 128
        FOLDED_TYPES_MAX = 8
        if nbasis > FOLDED_BASIS_MAX:
            raise ValueError(f"nbasis={nbasis} exceeds FOLDED_BASIS_MAX={FOLDED_BASIS_MAX}")
        if ntypes > FOLDED_TYPES_MAX:
            raise ValueError(f"ntypes={ntypes} exceeds FOLDED_TYPES_MAX={FOLDED_TYPES_MAX}")
        coeff_pad = np.zeros(FOLDED_TYPES_MAX * FOLDED_BASIS_MAX * 4, dtype=np.float32)
        coeff_flat = np.asarray(coeffs, dtype=np.float32).reshape(ntypes, -1)[:, :nbasis]
        coeff_pad[:ntypes * nbasis] = coeff_flat.flatten()
        kxyz_pad = np.zeros((FOLDED_BASIS_MAX, 4), dtype=np.float32)
        kxyz_pad[:nbasis, :] = kxyz[:nbasis]
        self.toGPU('folded_coeffs',    coeff_pad)
        self.toGPU('folded_kxyz',      kxyz_pad)
        self.toGPU('folded_atom_type', atype)
        self.kernel_params['folded_meta']   = folded_meta
        self.kernel_params['folded_lvec2d'] = lvec2d
        self.kernel_params['md_params']     = np.array([0.92, 0.88, 1.0, 1.0], dtype=np.float32)
        self.kernel_params.setdefault('newton_params', np.array([0.1, 0.1, 0.5, 1e-2], dtype=np.float32))
        self.kernel_params.setdefault('f2tol', np.float32(1e-10))
        self.replicas_args = self.generate_kernel_args("rigid_body_folded_replicas_kernel")
        self.krnl_folded_replicas = cl.Kernel(self.prg, "rigid_body_folded_replicas_kernel")
        self.newton_replicas_args = self.generate_kernel_args("rigid_body_folded_newton_replicas_kernel")
        self.krnl_newton_replicas = cl.Kernel(self.prg, "rigid_body_folded_newton_replicas_kernel")
        self.n_replicas = n_replicas

    # ------------------------------------------------------------------
    #  Relaxation: FIRE quench + trust-region Newton on 6×6 Hessian
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_normalize(q):
        q = np.asarray(q, dtype=np.float64)
        return (q / max(np.linalg.norm(q), 1e-30)).astype(np.float32)

    @staticmethod
    def _quat_mul(q1, q2):
        x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64)
        x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64)
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=np.float64)

    @staticmethod
    def _quat_from_rotvec(th):
        th = np.asarray(th, dtype=np.float64)
        ang = float(np.linalg.norm(th))
        if ang < 1e-14:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        axis = th / ang
        s = np.sin(0.5 * ang)
        return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(0.5 * ang)], dtype=np.float64)

    def _backend_run(self, backend, nstep, dt, lin_damp, ang_damp, fire):
        if backend == 'folded':
            self.run_folded(nstep, dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire)
        elif backend == 'gridff':
            self.run_gridff(nstep, dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire)
        elif backend == 'generic':
            self.run(nstep, dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire)
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    def eval_force_torque(self, backend='folded', body_id=0):
        """Force/torque at current pose without moving (dt=0). Returns F_world, T_world, T_body, E."""
        self._backend_run(backend, 1, 0.0, 1.0, 1.0, fire=False)
        out = self.download_outputs()
        F = out['body_force'][body_id, :3].astype(np.float64)
        Tw = out['body_torque'][body_id, :3].astype(np.float64)
        R = _quat_to_matrix_np(out['quats'][body_id]).astype(np.float64)
        Tb = R.T @ Tw
        E = float(out['atom_positions'][body_id][:, 3].sum())
        return F, Tw, Tb, E, out

    def _set_pose(self, pos, quat, body_id=0, zero_vel=True):
        pos4 = np.empty((self.n_bodies, 4), dtype=np.float32)
        quat4 = np.empty((self.n_bodies, 4), dtype=np.float32)
        self.fromGPU('poss', pos4)
        self.fromGPU('qrots', quat4)
        pos4[body_id, :3] = np.asarray(pos, dtype=np.float32)
        quat4[body_id] = self._quat_normalize(quat)
        self.toGPU('poss', pos4)
        self.toGPU('qrots', quat4)
        if zero_vel:
            z = np.zeros((self.n_bodies, 4), dtype=np.float32)
            self.toGPU('vposs', z)
            self.toGPU('vrots', z)
        self.queue.finish()

    def _apply_delta(self, dx, dtheta, body_id=0):
        """pos += dx; q ← q ⊗ exp(dθ) with dθ in body frame."""
        pos4 = np.empty((self.n_bodies, 4), dtype=np.float32)
        quat4 = np.empty((self.n_bodies, 4), dtype=np.float32)
        self.fromGPU('poss', pos4)
        self.fromGPU('qrots', quat4)
        pos4[body_id, :3] = pos4[body_id, :3] + np.asarray(dx, dtype=np.float32)
        q = self._quat_mul(quat4[body_id], self._quat_from_rotvec(dtheta))
        quat4[body_id] = self._quat_normalize(q)
        self.toGPU('poss', pos4)
        self.toGPU('qrots', quat4)
        z = np.zeros((self.n_bodies, 4), dtype=np.float32)
        self.toGPU('vposs', z)
        self.toGPU('vrots', z)
        self.queue.finish()

    def hessian_fd(self, backend='folded', body_id=0, eps_t=0.1, eps_r=0.1):
        """Central-difference 6×6 Hessian of E in (Δx, Δθ_body).

        Uses G=[F, τ_body]=−∇E so H = −∂G/∂u. Symmetrized.
        """
        pos0 = np.empty((self.n_bodies, 4), dtype=np.float32)
        quat0 = np.empty((self.n_bodies, 4), dtype=np.float32)
        self.fromGPU('poss', pos0)
        self.fromGPU('qrots', quat0)
        p0 = pos0[body_id, :3].astype(np.float64).copy()
        q0 = quat0[body_id].astype(np.float64).copy()
        eps = np.array([eps_t, eps_t, eps_t, eps_r, eps_r, eps_r], dtype=np.float64)
        H = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            d = np.zeros(6); d[i] = eps[i]
            self._set_pose(p0 + d[:3], self._quat_mul(q0, self._quat_from_rotvec(d[3:])), body_id)
            Fp, _, Tp, _, _ = self.eval_force_torque(backend, body_id)
            Gp = np.concatenate([Fp, Tp])
            self._set_pose(p0 - d[:3], self._quat_mul(q0, self._quat_from_rotvec(-d[3:])), body_id)
            Fm, _, Tm, _, _ = self.eval_force_torque(backend, body_id)
            Gm = np.concatenate([Fm, Tm])
            # G = −∇E ⇒ H_E = −∂G/∂u
            H[:, i] = -(Gp - Gm) / (2.0 * eps[i])
        self._set_pose(p0, q0, body_id)
        return 0.5 * (H + H.T)

    def relax_fire(self, max_steps=500, dt=0.05, damp0=0.1, backend='folded',
                   f_tol=1e-4, t_tol=1e-4, batch=32, body_id=0, record=False):
        """FIRE relaxation with host-side early exit on |F|,|τ|.

        Kernel does v·F / ω·τ quench + velocity mixing (md_params.w < 0).
        Velocities are preserved across host batches (required for FIRE inertia).
        """
        hist = [] if record else None
        n_done = 0
        Fmag = Tmag = E = np.inf
        # Keep GPU-side FIRE state (adaptive dt/damp) by using large batches
        batch = max(int(batch), 64)
        while n_done < max_steps:
            n = min(batch, max_steps - n_done)
            self._backend_run(backend, n, dt, damp0, damp0, fire=True)
            n_done += n
            out = self.download_outputs()
            F = out['body_force'][body_id, :3].astype(np.float64)
            Tw = out['body_torque'][body_id, :3].astype(np.float64)
            E = float(out['atom_positions'][body_id][:, 3].sum())
            Fmag = float(np.linalg.norm(F))
            Tmag = float(np.linalg.norm(Tw))
            if record:
                hist.append(dict(step=n_done, E=E, F=Fmag, T=Tmag,
                                 pos=out['pos'][body_id].copy(), quat=out['quats'][body_id].copy()))
            if Fmag < f_tol and Tmag < t_tol:
                break
        F, Tw, _, E, _ = self.eval_force_torque(backend, body_id)
        z = np.zeros((self.n_bodies, 4), dtype=np.float32)
        self.toGPU('vposs', z); self.toGPU('vrots', z); self.queue.finish()
        Fmag = float(np.linalg.norm(F)); Tmag = float(np.linalg.norm(Tw))
        return {
            'steps': n_done, 'converged': Fmag < f_tol and Tmag < t_tol,
            'F': Fmag, 'T': Tmag, 'E': E, 'history': hist,
        }

    def relax_newton_host(self, max_iter=40, trust0=0.5, backend='folded', body_id=0,
                     f_tol=1e-5, t_tol=1e-5, eps_t=0.1, eps_r=0.1,
                     lambda0=1e-2, rot_scale=1.0, fire_warmstart=0, record=False):
        """HOST-SIDE Newton (debug only). Prefer ``run_folded_newton`` for production.

        Each FD column is a separate OpenCL launch + NumPy 6×6 solve — far too
        slow for imaging. Kept for parity checks against the GPU Newton kernel.
        Default fire_warmstart=0 (pure Newton).
        """
        if fire_warmstart and fire_warmstart > 0:
            self.relax_fire(max_steps=int(fire_warmstart), dt=0.05, damp0=0.1,
                            backend=backend, body_id=body_id, f_tol=f_tol, t_tol=t_tol, batch=16)

        sigma = float(rot_scale)
        S = np.array([1.0, 1.0, 1.0, sigma, sigma, sigma], dtype=np.float64)
        trust = float(trust0)
        lam = float(lambda0)
        recovery_used = False
        hist = [] if record else None
        F, Tw, Tb, E, out = self.eval_force_torque(backend, body_id)
        for it in range(max_iter):
            Fmag = float(np.linalg.norm(F))
            Tmag = float(np.linalg.norm(Tw))
            if record:
                hist.append(dict(iter=it, E=E, F=Fmag, T=Tmag, trust=trust, lam=lam))
            if Fmag < f_tol and Tmag < t_tol:
                return {'iters': it, 'converged': True, 'F': Fmag, 'T': Tmag, 'E': E, 'history': hist}

            H = self.hessian_fd(backend, body_id, eps_t=eps_t, eps_r=eps_r)
            # Scale: H_y = S H S, rhs_y = S [F, τ]
            Hs = (S[:, None] * H) * S[None, :]
            rhs = S * np.concatenate([F, Tb])
            accepted = False
            for _try in range(10):
                A = Hs + lam * np.eye(6)
                try:
                    dy = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    lam = max(lam * 10.0, lambda0)
                    continue
                # Δu = S Δy; limit ||Δy|| (scaled trust)
                nrm = float(np.linalg.norm(dy))
                if nrm > trust and nrm > 1e-30:
                    dy = dy * (trust / nrm)
                    nrm = trust
                delta = S * dy
                pos4 = np.empty((self.n_bodies, 4), dtype=np.float32)
                quat4 = np.empty((self.n_bodies, 4), dtype=np.float32)
                self.fromGPU('poss', pos4); self.fromGPU('qrots', quat4)
                p_save = pos4[body_id].copy(); q_save = quat4[body_id].copy()
                self._apply_delta(delta[:3], delta[3:], body_id)
                F2, Tw2, Tb2, E2, _ = self.eval_force_torque(backend, body_id)
                if E2 < E - 1e-14:
                    E, F, Tw, Tb = E2, F2, Tw2, Tb2
                    accepted = True
                    recovery_used = False
                    if nrm > 0.8 * trust:
                        trust = min(trust * 2.0, trust0)
                    lam = max(lam * 0.3, 1e-8)
                    break
                # Reject energy-increasing steps (avoids other local basins)
                self.toGPU('poss', pos4); self.toGPU('qrots', quat4)
                self.queue.finish()
                trust = max(trust * 0.5, 1e-4)
                lam = min(max(lam * 5.0, lambda0), 1e4)
            if not accepted:
                if recovery_used:
                    return {'iters': it + 1, 'converged': False, 'F': Fmag, 'T': Tmag, 'E': E, 'history': hist}
                trust = float(trust0)
                lam = float(lambda0)
                recovery_used = True

        Fmag = float(np.linalg.norm(F)); Tmag = float(np.linalg.norm(Tw))
        return {'iters': max_iter, 'converged': Fmag < f_tol and Tmag < t_tol,
                'F': Fmag, 'T': Tmag, 'E': E, 'history': hist}

    def download_outputs(self):
        pos         = np.empty((self.n_bodies, 4), dtype=np.float32)
        quats       = np.empty((self.n_bodies, 4), dtype=np.float32)
        lin_mom     = np.empty((self.n_bodies, 4), dtype=np.float32)
        ang_mom     = np.empty((self.n_bodies, 4), dtype=np.float32)
        atoms_world = np.empty((self.total_atoms, 4), dtype=np.float32)
        atom_force  = np.empty((self.total_atoms, 4), dtype=np.float32)
        body_force  = np.empty((self.n_bodies, 4), dtype=np.float32)
        body_torque = np.empty((self.n_bodies, 4), dtype=np.float32)

        self.fromGPU('poss', pos)
        self.fromGPU('qrots', quats)
        self.fromGPU('vposs', lin_mom)
        self.fromGPU('vrots', ang_mom)
        self.fromGPU('apos_world', atoms_world)
        self.fromGPU('atom_force', atom_force)
        self.fromGPU('body_force', body_force)
        self.fromGPU('body_torque', body_torque)
        self.queue.finish()

        if getattr(self, 'allmol_mode', False) and getattr(self, 'mol_offsets', None) is not None:
            a = int(self.active_body)
            i0, i1 = int(self.mol_offsets[a]), int(self.mol_offsets[a + 1])
            atoms_world = atoms_world[i0:i1][None, ...]
            atom_force = atom_force[i0:i1][None, ...]
        else:
            atoms_world = atoms_world.reshape(self.n_bodies, self.num_atoms, 4)
            atom_force = atom_force.reshape(self.n_bodies, self.num_atoms, 4)
        self.last_atom_force = atom_force
        self.last_body_force = body_force
        self.last_body_torque = body_torque

        return {
            'pos': pos,
            'quats': quats,
            'lin_mom': lin_mom,
            'ang_mom': ang_mom,
            'atom_positions': atoms_world,
            'atom_force': atom_force,
            'body_force': body_force,
            'body_torque': body_torque,
        }

    def download_selected(self, fields):
        req = tuple(fields)
        out = {}
        if 'pos' in req:
            buf = np.empty((self.n_bodies, 4), dtype=np.float32)
            self.fromGPU('poss', buf)
            out['pos'] = buf
        if 'quats' in req:
            buf = np.empty((self.n_bodies, 4), dtype=np.float32)
            self.fromGPU('qrots', buf)
            out['quats'] = buf
        if 'lin_mom' in req:
            buf = np.empty((self.n_bodies, 4), dtype=np.float32)
            self.fromGPU('vposs', buf)
            out['lin_mom'] = buf
        if 'ang_mom' in req:
            buf = np.empty((self.n_bodies, 4), dtype=np.float32)
            self.fromGPU('vrots', buf)
            out['ang_mom'] = buf
        if 'atom_positions' in req:
            buf = np.empty((self.total_atoms, 4), dtype=np.float32)
            self.fromGPU('apos_world', buf)
            out['atom_positions'] = buf.reshape(self.n_bodies, self.num_atoms, 4)
        if 'atom_force' in req:
            buf = np.empty((self.total_atoms, 4), dtype=np.float32)
            self.fromGPU('atom_force', buf)
            out['atom_force'] = buf.reshape(self.n_bodies, self.num_atoms, 4)
        if 'body_force' in req:
            buf = np.empty((self.n_bodies, 4), dtype=np.float32)
            self.fromGPU('body_force', buf)
            self.last_body_force = buf
            out['body_force'] = buf
        if 'body_torque' in req:
            buf = np.empty((self.n_bodies, 4), dtype=np.float32)
            self.fromGPU('body_torque', buf)
            self.last_body_torque = buf
            out['body_torque'] = buf
        if 'fire_state' in req:
            buf = np.empty((self.n_bodies, 4), dtype=np.float32)
            self.fromGPU('fire_state', buf)
            out['fire_state'] = buf
        if 'newton_state' in req:
            buf = np.empty((self.n_bodies, 4), dtype=np.float32)
            self.fromGPU('newton_state', buf)
            out['newton_state'] = buf
        self.queue.finish()
        return out

    def sync_outputs_to_inputs(self):
        self.queue.finish()

    def get_debug_dict(self):
        out = {
            'enames': list(self.enames) if self.enames is not None else None,
            'atom_types': list(self.atom_types_assigned) if self.atom_types_assigned is not None else None,
            'REQ': None if self.atom_REQ is None else np.array(self.atom_REQ, copy=True),
            'PLQ': None if self.atom_PLQ is None else np.array(self.atom_PLQ, copy=True),
            'PLQ_coeffs': None if self.atom_PLQ is None else _plq_to_coeffs(self.atom_PLQ),
            'masses': None if self.atom_masses is None else np.array(self.atom_masses, copy=True),
            'grid_shape': self.grid_shape,
            'grid_p0': None if self.grid_p0 is None else np.array(self.grid_p0, copy=True),
            'grid_step': None if self.grid_step is None else np.array(self.grid_step, copy=True),
            'last_atom_force': None if self.last_atom_force is None else np.array(self.last_atom_force, copy=True),
            'last_body_force': None if self.last_body_force is None else np.array(self.last_body_force, copy=True),
            'last_body_torque': None if self.last_body_torque is None else np.array(self.last_body_torque, copy=True),
            'folded_params': None if self.folded_params is None else {k: v.copy() for k, v in self.folded_params.items()},
        }
        return out

    @classmethod
    def from_xyz_and_grid(cls, mol_file, grid_file, substrate_xyz, n_bodies=1, body_positions=None, quats=None, alpha_morse=DEFAULT_ALPHA_MORSE, debug=False, type_map=None, mass_trans=1.0, mass_rot=None):
        apos, reqs, enames, _, _ = load_xyz_with_REQs(mol_file, type_map=type_map)
        masses = _guess_mass(enames)
        apos = np.asarray(apos, dtype=np.float32)
        com0 = (apos * masses[:, None]).sum(axis=0) / masses.sum()
        rel = apos - com0[None, :]
        mtot, I, Iinv = compute_mass_properties(rel, masses)
        mass_trans = float(mass_trans)
        if mass_trans <= 0.0:
            raise ValueError(f"mass_trans must be > 0, got {mass_trans}")
        if mass_rot is None:
            mass_rot = mass_trans
        mass_rot = float(mass_rot)
        if mass_rot <= 0.0:
            raise ValueError(f"mass_rot must be > 0, got {mass_rot}")
        Iinv_relax = Iinv * (mtot / mass_rot)
        I_relax = I * (mass_rot / mtot)
        if body_positions is None:
            body_positions = np.repeat(com0[None, :], n_bodies, axis=0).astype(np.float32)
        else:
            body_positions = np.asarray(body_positions, dtype=np.float32)
            if body_positions.shape != (n_bodies, 3):
                raise ValueError(f"Expected body_positions shape ({n_bodies},3), got {body_positions.shape}")
        pos4 = np.zeros((n_bodies, 4), dtype=np.float32)
        pos4[:, :3] = body_positions
        pos4[:, 3] = mass_trans
        quat4 = np.zeros((n_bodies, 4), dtype=np.float32)
        quat4[:, 3] = 1.0
        if quats is not None:
            q = _ensure_float4(quats)
            if q.shape[0] != n_bodies:
                raise ValueError(f"Expected quats shape ({n_bodies},4), got {q.shape}")
            quat4[:] = q
        zero4 = np.zeros((n_bodies, 4), dtype=np.float32)
        atom_body = np.repeat(rel[None, :, :], n_bodies, axis=0).astype(np.float32)
        atom_plq_single = _reqs_to_plq(reqs, alpha=alpha_morse)
        atom_plq = np.repeat(atom_plq_single[None, :, :], n_bodies, axis=0).reshape(n_bodies * len(enames), 4)
        try:
            grid = np.load(grid_file)
        except Exception:
            grid = _load_npy_legacy(grid_file)
        
        # Read lattice vectors from comment line manually
        with open(substrate_xyz, 'r') as f:
            lines = f.readlines()
            comment = lines[1].strip()
            lvec = None
            if "lvec:" in comment:
                idx = comment.find("lvec:") + 5
                parts = comment[idx:].split()
            elif "lvs" in comment:
                idx = comment.find("lvs") + 3
                parts = comment[idx:].split()
            else:
                parts = []
            
            try:
                vals = [float(v) for v in parts if v.strip()]
                if len(vals) >= 9:
                    lvec = np.array(vals[:9]).reshape(3,3).astype(np.float32)
            except ValueError:
                pass
            
            if lvec is None:
                raise ValueError(f"Substrate lattice vectors missing in {substrate_xyz}")
                
        ax = float(np.linalg.norm(lvec[0]))
        ay = float(np.linalg.norm(lvec[1]))
        az = float(np.linalg.norm(lvec[2]))
        if abs(lvec[0][1]) > 1e-6 or abs(lvec[1][0]) > 1e-6 or abs(lvec[0][2]) > 1e-6 or abs(lvec[1][2]) > 1e-6:
            raise ValueError(f"Only orthorhombic xy substrate cells supported for now, got lvec={lvec}")
        grid_step = (ax / grid.shape[0], ay / grid.shape[1], az / grid.shape[2])
        grid_p0 = (0.0, 0.0, 0.0)
        rbd = cls(debug=debug)
        rbd.realloc(n_bodies=n_bodies, num_atoms=len(enames))
        rbd.enames = list(enames)
        rbd.atom_types_assigned = [type_map.get(e, e) if type_map is not None else e for e in enames]
        rbd.atom_REQ = reqs.copy()
        rbd.atom_masses = masses.copy()
        rbd.mass_physical = float(mtot)
        rbd.mass_trans = mass_trans
        rbd.mass_rot = mass_rot
        rbd.atom_PLQ = atom_plq.copy()
        rbd.upload_state(pos4, quat4, zero4, zero4, mass_trans, 1.0 / mass_trans, np.repeat(Iinv_relax[None, :, :], n_bodies, axis=0), atom_body, atom_PLQ=atom_plq, inertia=np.repeat(I_relax[None, :, :], n_bodies, axis=0))
        rbd.init_gridff(grid, grid_p0=grid_p0, grid_step=grid_step)
        return rbd

    def reset_pose(self, pos, quats, lin_mom=None, ang_mom=None):
        if self.atom_body_host is None or self.inertia_inv_host is None or self.mass_total is None:
            raise RuntimeError("RigidBodyDynamics.reset_pose() requires prior upload_state() initialization")
        pos_in = _ensure_float4(pos)
        quats_in = _ensure_float4(quats)
        if lin_mom is None:
            lin_mom = np.zeros((self.n_bodies, 4), dtype=np.float32)
        if ang_mom is None:
            ang_mom = np.zeros((self.n_bodies, 4), dtype=np.float32)
        self.upload_state(
            pos_in,
            quats_in,
            lin_mom,
            ang_mom,
            self.mass_total,
            1.0 / self.mass_total,
            self.inertia_inv_host[:, :, :3],
            self.atom_body_host.reshape(self.n_bodies, self.num_atoms, 4)[:, :, :3],
            atom_PLQ=self.atom_PLQ,
            inertia=self.inertia_host[:, :, :3] if self.inertia_host is not None else None,
        )

    def update_anchors(self, anchors_world):
        self.anchors = _ensure_float4(anchors_world, w_value=-1.0)
        self.upload_anchors()
        self.reset_optimizer_state()

    def upload_anchors(self):
        self.toGPU('anchors', self.anchors)
        self.queue.finish()


# ==================================================================
#  RigidBodyPairFF — pairwise molecule-molecule rigid body dynamics
# ==================================================================
#
#  Force model (GPU kernel: rigid_body_pairff_kernel in rigid.cl):
#
#    atom-atom (type=0 ↔ type=0):
#      Morse:  E = E0 * [exp(2α(r-R0)) - 2*exp(α(r-R0))]
#      Coulomb: E = kQ / sqrt(r^2 + R2SAFE)   (damped to avoid singularity)
#      Mixing: R0 = Ri+Rj, E0 = sqrtEi*sqrtEj, Q = Qi*Qj
#
#    epair-atom (type=1 ↔ type=0, or type=0 ↔ type=1):
#      Lorentzian Hbond: E = coeff * fcut(r/rc) * 1/(w^2 + r^2)
#      fcut = smoothstep(1 - r/rc) = 3x^2 - 2x^3
#      coeff = min(0, Q_atom * Q_epair)  [attractive only]
#      Q_epair = He (pseudo-charge stored in REQ.z)
#
#    sigma-hole-atom (type=2 ↔ type=0):
#      Same Lorentzian form, Q_sigmahole = Hs (pseudo-charge in REQ.z)
#
#    Z-harmonic constraint: F_z = -k_z * (z - z_target)
#      Applied per-atom (produces both force AND torque on the body).
#
#  Design decisions:
#    - Epairs and sigma-holes are treated as pseudo-atoms with R=0, E=0.
#      They participate ONLY in Hbond/sigma-hole interactions, not in Morse/Coulomb.
#      This avoids double-counting: the Morse potential between real atoms
#      already captures Pauli+London; epairs add the directional Hbond correction.
#    - The pseudo-charge (He, Hs) is stored in REQ.z (the charge slot) so the
#      kernel can use the same REQ array for all interaction types.
#    - n_static_atoms and n_dyn_atoms are passed separately to the kernel so
#      it can skip epair-epair interactions (no Morse/Coulomb between epairs).
#
#  Open issues:
#    - Epair/sigma-hole positions are fixed in the body frame at construction
#      time. Changing epair_dist/sigma_dist via the GUI does NOT reposition
#      them on the GPU — requires rebuilding the body. A future fix would
#      recompute body-frame positions and re-upload.
#    - No epair-epair or sigma-sigma interactions (intentional — these are
#      directional corrections, not independent interaction sites).
#    - MAX_STATIC_ATOMS=128 is a compile-time limit in the kernel.
#

def add_electron_pairs_via_atomic_system(apos, enames, qs=None, epair_dist=1.4, sigma_dist=1.0):
    """Add electron pairs and sigma holes using AtomicSystem.

    Electron pairs (lone pairs on O/N) are placed by AtomicSystem.add_electron_pairs()
    at 0.5 Å from the host, then rescaled to epair_dist (default 1.4 Å for Hbond mode).
    Sigma holes are placed on H atoms bonded to O or N, at sigma_dist along the
    O-H / N-H bond direction (pointing away from the heavy atom).

    The resulting array is sorted: real atoms first, then epairs, then sigma holes.
    This ordering is CRITICAL for the GPU kernel's branch-free design — threads
    determine their role by index comparison (atom_idx < n_atoms) rather than
    per-pair type branching, which would cause warp divergence.

    Args:
        apos: (N,3) atom positions
        enames: list of element names
        qs: optional charges
        epair_dist: distance of epair from host atom [Å] (default 1.4 for Hbond mode)
        sigma_dist: distance of sigma hole from H atom [Å] (0 = disabled)

    Returns (extended_apos, extended_enames, type_flags) where:
        type_flags[i]=0 for atoms, 1 for epairs, 2 for sigma holes.
    """
    from .. import elements
    atypes = [elements.ELEMENT_DICT[e][0] if e in elements.ELEMENT_DICT else 200 for e in enames]
    mol = AtomicSystem(apos=np.asarray(apos, dtype=np.float32).copy(),  atypes=atypes, enames=list(enames), qs=qs)
    mol.neighs(bBond=True)
    mol.add_electron_pairs()

    # Rescale epair distances
    if epair_dist != 0.5:
        ep_mask = np.array([e == 'E' for e in mol.enames])
        ep_indices = np.where(ep_mask)[0]
        for i in ep_indices:
            host = next(j for j in mol.ngs[i] if mol.enames[j] != 'E')
            direction = mol.apos[i] - mol.apos[host]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                mol.apos[i] = mol.apos[host] + direction / norm * epair_dist

    # Add sigma holes on H bonded to O/N (type 'Sh')
    if sigma_dist > 0:
        n_before = len(mol.enames)
        for i, e in enumerate(mol.enames):
            if e != 'H': continue
            for j in mol.ngs[i]:
                if mol.enames[j] in ('O', 'N'):
                    direction = mol.apos[i] - mol.apos[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        mol.place_electron_pair(i, direction / norm, distance=sigma_dist, ename='Sh')
                    break

    n_total = len(mol.enames)
    types = np.zeros(n_total, dtype=np.int32)
    for i, e in enumerate(mol.enames):
        if e == 'E': types[i] = 1
        elif e == 'Sh': types[i] = 2
    return np.asarray(mol.apos, dtype=np.float32), list(mol.enames), types


PAIRFF_MODE_LEGACY = 'legacy'
PAIRFF_MODE_UNIFIED = 'unified'

MAX_ENV_SITES = 4096
MAX_ENV_MOLS = 64


def _body_sites_world(rel, pos, quat):
    """Body-frame sites (N,3/4) → world (N,3) given CoM pos and quaternion."""
    rel = np.asarray(rel, dtype=np.float32)
    R = _quat_to_matrix_np(np.asarray(quat, dtype=np.float32))
    return np.asarray(pos, dtype=np.float32)[:3] + (rel[:, :3] @ R.T)


def _prepare_molecule_pack(apos, enames, REQs, epair_dist=1.4, sigma_dist=1.0, He=-0.1, Hs=1.0, w=0.7):
    """Add epairs/sigma, center at CoM, return pack dict for multi-body PairFF."""
    REQs_base = np.asarray(REQs, dtype=np.float32).copy()
    apos_ext, enames_ext, types = add_electron_pairs_via_atomic_system(
        apos, enames, epair_dist=epair_dist, sigma_dist=sigma_dist)
    REQs_ext = _extend_reqs_with_epairs(REQs_base, enames_ext, types, He=He, Hs=Hs, w=w)
    masses = _guess_mass([e for e, t in zip(enames_ext, types) if t == 0])
    apos_ext = np.asarray(apos_ext, dtype=np.float32)
    com0 = (apos_ext[:len(masses)] * masses[:, None]).sum(axis=0) / masses.sum()
    rel = apos_ext - com0[None, :]
    mtot, I, Iinv = compute_mass_properties(rel[:len(masses)], masses)
    return dict(
        rel=rel.astype(np.float32),
        enames=list(enames_ext),
        types=np.asarray(types, dtype=np.int32),
        REQ_base=REQs_base,
        REQ_ext=REQs_ext,
        masses=masses,
        mtot=float(mtot),
        I=I,
        Iinv=Iinv,
        com0=com0.astype(np.float32),
    )


class RigidBodyPairFF(RigidBodyDynamics):
    """Rigid body dynamics with pairwise molecule-molecule interactions.

    Extends RigidBodyDynamics with a static molecule and pairwise forces.
    Two GPU force models (switch via pairff_mode):

      legacy  — rigid_body_pairff_kernel: 4-loop Morse+Coulomb / Lorentzian
      unified — rigid_body_pairff_unified_kernel: single compact-exp loop

    Design decisions:
        - Atoms and dummy atoms (epairs, sigma holes) are stored in a single
          sorted array: [real_atoms, epairs, sigma_holes] (legacy kernel needs
          this; unified does not, but layout is kept for rendering).
        - Pseudo-charges (He, Hs) are stored in REQ.z for dummy atoms.
          REQ.w stores blunt width w for unified soft-radius (0 for real atoms).
        - Z-harmonic constraint is applied per-atom (not per-CoM).
        - FIRE relaxation is supported with host-side early exit.
    """

    def __init__(self, debug=False):
        super().__init__(debug=debug)
        self.kernelheaders["rigid_body_pairff_kernel"] = """__kernel
void rigid_body_pairff_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   static_apos,
    __global const float4*   static_REQ,
    __global const int*      static_type,
    const int                n_static,
    const int                n_static_atoms,
    const int                n_dyn_atoms,
    const float4             pairff_params,
    const float              morse_alpha,
    const float              z_target,
    const float              Hs,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""
        self.kernelheaders["rigid_body_pairff_unified_kernel"] = """__kernel
void rigid_body_pairff_unified_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   static_apos,
    __global const float4*   static_REQ,
    __global const int*      static_type,
    const int                n_static,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""
        self.kernelheaders["rigid_body_pairff_unified_env_kernel"] = """__kernel
void rigid_body_pairff_unified_env_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   env_apos,
    __global const float4*   env_REQ,
    __global const int*      env_type,
    __global const int*      env_mols,
    const int                n_env_mols,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""
        self.kernelheaders["rigid_body_pairff_unified_faf_kernel"] = """__kernel
void rigid_body_pairff_unified_faf_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   static_apos,
    __global const float4*   static_REQ,
    __global const int*      static_type,
    const int                n_static,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""
        self.kernelheaders["rigid_body_pairff_unified_env_faf_kernel"] = """__kernel
void rigid_body_pairff_unified_env_faf_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    __global const float4*   env_apos,
    __global const float4*   env_REQ,
    __global const int*      env_type,
    __global const int*      env_mols,
    const int                n_env_mols,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""
        self.kernelheaders["rigid_body_pairff_unified_allmol_kernel"] = """__kernel
void rigid_body_pairff_unified_allmol_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    const int                n_mols,
    const int                active_mol,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""
        self.kernelheaders["rigid_body_pairff_unified_allmol_faf_kernel"] = """__kernel
void rigid_body_pairff_unified_allmol_faf_kernel(
    __global const int*      mols,
    __global       float4*   poss,
    __global       float4*   qrots,
    __global       float4*   vposs,
    __global       float4*   vrots,
    __global       float4*   fire_state,
    __global const cl_Mat3*  I_body_inv,
    __global const cl_Mat3*  I_body,
    __global const float4*   apos_body,
    __global       float4*   apos_world,
    __global const float4*   dyn_REQ,
    __global const int*      dyn_type,
    __global       float4*   atom_force,
    __global       float4*   body_force,
    __global       float4*   body_torque,
    __global const float4*   anchors,
    const int                n_mols,
    const int                active_mol,
    const float4             pairff_params,
    const float              beta,
    const float              z_target,
    __global const float*    folded_coeffs,
    __global const float4*   folded_kxyz,
    __global const int*      folded_atom_type,
    const int4               folded_meta,
    const float4             folded_lvec2d,
    const float              dt,
    const float4             md_params,
    const int                niter
)"""
        self.pairff_args = None
        self.krnl_pairff = None
        self.pairff_unified_args = None
        self.krnl_pairff_unified = None
        self.pairff_env_args = None
        self.krnl_pairff_env = None
        self.pairff_unified_faf_args = None
        self.krnl_pairff_unified_faf = None
        self.pairff_env_faf_args = None
        self.krnl_pairff_env_faf = None
        self.pairff_allmol_args = None
        self.krnl_pairff_allmol = None
        self.pairff_allmol_faf_args = None
        self.krnl_pairff_allmol_faf = None
        self.pairff_mode = PAIRFF_MODE_LEGACY
        self.env_mode = False
        self.allmol_mode = False  # shared all-molecule buffers; active_mol index only
        self.faf_mode = False  # when True, run_pairff uses fused PairFF+FAF kernels
        self.n_env_mols = 0
        self.n_env_sites = 0
        self.env_mols_host = None
        self.env_apos_host = None
        self.env_REQ_host = None
        self.env_type_host = None
        self._mb_packs = None
        self._mb_pos = None
        self._mb_quat = None
        self.active_body = 0
        self.static_n = 0
        self.static_apos_host = None
        self.static_REQ_host = None
        self.static_type_host = None
        self.dyn_type_host = None
        self.dyn_REQ_host = None
        self.pairff_params_host = None
        # Base REQs before He/Hs/w overlay (real-atom rows only, pre-extend)
        self._dyn_REQ_base = None
        self._static_REQ_base = None
        self._dyn_enames = None
        self._static_enames = None

    def alloc_pairff(self, n_static):
        """Allocate buffers for static atoms and dynamic atom types/REQ."""
        MAX_STATIC = 128
        if n_static > MAX_STATIC:
            raise ValueError(f"n_static={n_static} exceeds MAX_STATIC_ATOMS={MAX_STATIC}")
        self.static_n = int(n_static)
        float_size = np.float32().itemsize
        int_size = np.int32().itemsize
        mf = cl.mem_flags
        self.create_buffer('static_apos',  MAX_STATIC * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('static_REQ',  MAX_STATIC * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('static_type', MAX_STATIC * int_size,        mf.READ_ONLY)
        self.create_buffer('dyn_REQ',     self.total_atoms * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('dyn_type',    self.total_atoms * int_size,        mf.READ_ONLY)

    def upload_static(self, static_apos, static_REQ, static_type):
        """Upload static molecule data (world positions, REQ, type flags) to GPU."""
        n = len(static_apos)
        MAX_STATIC = 128
        if n > MAX_STATIC:
            raise ValueError(f"n_static={n} exceeds MAX_STATIC_ATOMS={MAX_STATIC}")
        apos_pad = np.zeros((MAX_STATIC, 4), dtype=np.float32)
        req_pad  = np.zeros((MAX_STATIC, 4), dtype=np.float32)
        type_pad = np.zeros(MAX_STATIC, dtype=np.int32)
        apos_pad[:n] = _ensure_float4(static_apos)
        req_pad[:n]  = _ensure_float4(static_REQ)
        type_pad[:n] = np.asarray(static_type, dtype=np.int32)
        self.toGPU('static_apos',  apos_pad)
        self.toGPU('static_REQ',   req_pad)
        self.toGPU('static_type',  type_pad)
        self.static_n = n
        self.static_apos_host = apos_pad[:n].copy()
        self.static_REQ_host  = req_pad[:n].copy()
        self.static_type_host = type_pad[:n].copy()

    def upload_dyn_types_req(self, dyn_type, dyn_REQ):
        """Upload per-atom type flags (0=atom, 1=epair) and REQ for dynamic body."""
        dtype_arr = np.asarray(dyn_type, dtype=np.int32)
        req_arr = _ensure_float4(dyn_REQ)
        if dtype_arr.shape[0] != self.total_atoms:
            raise ValueError(f"dyn_type length {dtype_arr.shape[0]} != total_atoms {self.total_atoms}")
        if req_arr.shape[0] != self.total_atoms:
            raise ValueError(f"dyn_REQ length {req_arr.shape[0]} != total_atoms {self.total_atoms}")
        self.toGPU('dyn_type', dtype_arr)
        self.toGPU('dyn_REQ',  req_arr)
        self.dyn_type_host = dtype_arr.copy()
        self.dyn_REQ_host  = req_arr.copy()

    def alloc_pairff_env(self, max_env_sites=MAX_ENV_SITES, max_env_mols=MAX_ENV_MOLS):
        """Allocate GPU buffers for multi-molecule environment (Strategy M tiling)."""
        max_env_sites = int(max_env_sites)
        max_env_mols = int(max_env_mols)
        if max_env_sites > MAX_ENV_SITES:
            raise ValueError(f"max_env_sites={max_env_sites} exceeds MAX_ENV_SITES={MAX_ENV_SITES}")
        if max_env_mols > MAX_ENV_MOLS:
            raise ValueError(f"max_env_mols={max_env_mols} exceeds MAX_ENV_MOLS={MAX_ENV_MOLS}")
        float_size = np.float32().itemsize
        int_size = np.int32().itemsize
        mf = cl.mem_flags
        self.create_buffer('env_apos',  MAX_ENV_SITES * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('env_REQ',   MAX_ENV_SITES * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('env_type',  MAX_ENV_SITES * int_size,        mf.READ_ONLY)
        self.create_buffer('env_mols',  (MAX_ENV_MOLS + 1) * int_size,   mf.READ_ONLY)
        self.create_buffer('dyn_REQ',   self.total_atoms * 4 * float_size, mf.READ_ONLY)
        self.create_buffer('dyn_type',  self.total_atoms * int_size,        mf.READ_ONLY)
        self.env_mode = True

    def upload_env(self, env_apos, env_REQ, env_type, env_mols):
        """Upload flat env sites + molecule offsets (Strategy M).

        env_mols: length n_env_mols+1 cumulative offsets into the flat arrays.
        Each molecule tile must have ≤ MAX_STATIC_ATOMS (128) sites.
        """
        env_mols = np.asarray(env_mols, dtype=np.int32).ravel()
        if env_mols.ndim != 1 or len(env_mols) < 1:
            raise ValueError("env_mols must be 1D offsets of length n_env_mols+1")
        n_env_mols = len(env_mols) - 1
        if n_env_mols < 0:
            raise ValueError("env_mols must have length >= 1")
        n_sites = int(env_mols[-1]) if n_env_mols >= 0 else 0
        if n_env_mols == 0:
            n_sites = 0
            env_mols = np.array([0], dtype=np.int32)
        if n_env_mols > MAX_ENV_MOLS:
            raise ValueError(f"n_env_mols={n_env_mols} exceeds MAX_ENV_MOLS={MAX_ENV_MOLS}")
        if n_sites > MAX_ENV_SITES:
            raise ValueError(f"n_env_sites={n_sites} exceeds MAX_ENV_SITES={MAX_ENV_SITES}")
        for em in range(n_env_mols):
            nj = int(env_mols[em + 1] - env_mols[em])
            if nj > 128:
                raise ValueError(f"env molecule {em} has {nj} sites > MAX_STATIC_ATOMS=128")
            if nj <= 0:
                raise ValueError(f"env molecule {em} has empty site range")
        apos_pad = np.zeros((MAX_ENV_SITES, 4), dtype=np.float32)
        req_pad  = np.zeros((MAX_ENV_SITES, 4), dtype=np.float32)
        type_pad = np.zeros(MAX_ENV_SITES, dtype=np.int32)
        mols_pad = np.zeros(MAX_ENV_MOLS + 1, dtype=np.int32)
        if n_sites > 0:
            apos_pad[:n_sites] = _ensure_float4(env_apos)[:n_sites]
            req_pad[:n_sites]  = _ensure_float4(env_REQ)[:n_sites]
            type_pad[:n_sites] = np.asarray(env_type, dtype=np.int32)[:n_sites]
        mols_pad[:len(env_mols)] = env_mols
        self.toGPU('env_apos', apos_pad)
        self.toGPU('env_REQ',  req_pad)
        self.toGPU('env_type', type_pad)
        self.toGPU('env_mols', mols_pad)
        self.n_env_mols = n_env_mols
        self.n_env_sites = n_sites
        self.env_mols_host = env_mols.copy()
        self.env_apos_host = apos_pad[:n_sites].copy()
        self.env_REQ_host  = req_pad[:n_sites].copy()
        self.env_type_host = type_pad[:n_sites].copy()
        # Vispy / debug: expose env as "static" for rendering
        self.static_n = n_sites
        self.static_apos_host = self.env_apos_host
        self.static_REQ_host = self.env_REQ_host
        self.static_type_host = self.env_type_host
        self.static_enames = ['E' if t == 1 else ('Sh' if t == 2 else '?') for t in self.env_type_host]

    def _rebuild_env_from_mb(self):
        """Rebuild env buffers from host multi-body packs excluding active_body."""
        if self._mb_packs is None:
            raise RuntimeError("No multi-body packs; call from_molecules() first")
        a_list, r_list, t_list, offsets = [], [], [], [0]
        enames_all = []
        for j, pack in enumerate(self._mb_packs):
            if j == self.active_body:
                continue
            world = _body_sites_world(pack['rel'], self._mb_pos[j], self._mb_quat[j])
            a_list.append(world)
            r_list.append(pack['REQ_ext'])
            t_list.append(pack['types'])
            enames_all.extend(pack['enames'])
            offsets.append(offsets[-1] + len(pack['types']))
        if len(a_list) == 0:
            self.upload_env(np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32),
                            np.zeros(0, dtype=np.int32), np.array([0], dtype=np.int32))
            self.static_enames = []
            return
        env_apos = np.vstack(a_list)
        env_REQ = np.vstack(r_list)
        env_type = np.concatenate(t_list)
        self.upload_env(env_apos, env_REQ, env_type, np.asarray(offsets, dtype=np.int32))
        self.static_enames = list(enames_all)

    def _upload_active_body_state(self, mass_trans=1.0, mass_rot=None):
        """Upload active molecule as the single GPU rigid body."""
        pack = self._mb_packs[self.active_body]
        if mass_rot is None:
            mass_rot = mass_trans
        Iinv_relax = pack['Iinv'] * (pack['mtot'] / float(mass_rot))
        I_relax = pack['I'] * (float(mass_rot) / pack['mtot'])
        n_dyn = len(pack['enames'])
        pos4 = np.zeros((1, 4), dtype=np.float32)
        pos4[0, :3] = self._mb_pos[self.active_body]
        pos4[0, 3] = float(mass_trans)
        quat4 = np.zeros((1, 4), dtype=np.float32)
        quat4[0] = self._mb_quat[self.active_body]
        zero4 = np.zeros((1, 4), dtype=np.float32)
        atom_body = pack['rel'][None, :, :].astype(np.float32)
        self.realloc(n_bodies=1, num_atoms=n_dyn)
        self.enames = list(pack['enames'])
        self.atom_REQ = pack['REQ_ext'].copy()
        self.atom_masses = pack['masses'].copy()
        self.mass_physical = float(pack['mtot'])
        self.mass_trans = float(mass_trans)
        self.mass_rot = float(mass_rot)
        self.upload_state(pos4, quat4, zero4, zero4, float(mass_trans), 1.0 / float(mass_trans),
                          Iinv_relax[None, :, :], atom_body, inertia=I_relax[None, :, :])
        self.alloc_pairff_env()
        self.upload_dyn_types_req(pack['types'], pack['REQ_ext'])
        self._dyn_REQ_base = pack['REQ_base']
        self._dyn_enames = list(pack['enames'])

    def sync_active_pose_from_gpu(self):
        """Copy all molecule poses from GPU into host multi-body state."""
        if self._mb_packs is None:
            return
        out = self.download_outputs()
        self._mb_pos = out['pos'][:, :3].copy()
        self._mb_quat = out['quats'].copy()

    def _apply_active_view_hosts(self):
        """Expose active molecule slices for Vispy (enames / dyn_type_host).

        GPU keeps flat all-mol dyn_type/REQ; host view is active pack only.
        """
        pack = self._mb_packs[self.active_body]
        self.enames = list(pack['enames'])
        self.dyn_type_host = pack['types'].copy()
        self._dyn_enames = list(pack['enames'])
        self._dyn_REQ_base = pack['REQ_base']

    def _refresh_static_view_hosts(self):
        """Host world sites of all non-active molecules for potential map / markers."""
        if self._mb_packs is None:
            return
        self.sync_active_pose_from_gpu()
        a_list, r_list, t_list, enames_all = [], [], [], []
        for j, pack in enumerate(self._mb_packs):
            if j == self.active_body:
                continue
            world = _body_sites_world(pack['rel'], self._mb_pos[j], self._mb_quat[j])
            a_list.append(world)
            r_list.append(pack['REQ_ext'])
            t_list.append(pack['types'])
            enames_all.extend(pack['enames'])
        if not a_list:
            self.static_n = 0
            self.static_apos_host = np.zeros((0, 4), dtype=np.float32)
            self.static_REQ_host = np.zeros((0, 4), dtype=np.float32)
            self.static_type_host = np.zeros(0, dtype=np.int32)
            self.static_enames = []
            return
        self.static_apos_host = np.vstack(a_list).astype(np.float32)
        self.static_REQ_host = np.vstack(r_list).astype(np.float32)
        self.static_type_host = np.concatenate(t_list).astype(np.int32)
        self.static_n = int(self.static_type_host.shape[0])
        self.static_enames = list(enames_all)

    def set_active_body(self, body_id, mass_trans=None, mass_rot=None):
        """Switch which molecule integrates — index only (no realloc / no env rebuild).

        All poses, velocities, and FIRE state stay persistent on GPU; non-active
        bodies are simply not integrated. Clears mouse anchors. FAF unchanged.
        """
        if self._mb_packs is None:
            raise RuntimeError("set_active_body requires from_molecules()")
        body_id = int(body_id)
        if body_id < 0 or body_id >= len(self._mb_packs):
            raise ValueError(f"active_body={body_id} out of range [0,{len(self._mb_packs)})")
        self.active_body = body_id
        self.kernel_params['active_mol'] = np.int32(body_id)
        # Clear anchors only (mouse springs are per-active-site); dynamics stay
        anc = np.zeros((self.total_atoms, 4), dtype=np.float32)
        anc[:, 3] = -1.0
        self.anchors = anc
        self.toGPU('anchors', anc)
        self._apply_active_view_hosts()
        self._refresh_static_view_hosts()
        if self.pairff_params_host is not None:
            self.pairff_params_host['active_body'] = body_id
            self.pairff_params_host['allmol_mode'] = self.allmol_mode


    def init_pairff(self, He=-0.1, rc=3.0, w=0.7, k_z=0.0, morse_alpha=1.8, z_target=0.0, Hs=1.0, epair_dist=1.4, sigma_dist=1.0, mode=None, beta=None):
        """Initialize pairff kernel parameters and generate argument list.

        Args:
            He:   Epair pseudo-charge (negative). Drives Hbond attraction with positive probes.
            rc:   Hbond/sigma-hole cutoff radius [Å] (legacy Lorentzian only).
            w:    Soft-radius / Lorentzian width [Å]. Unified: blunt width on dummy sites.
            k_z:  Z-harmonic constraint strength per atom (0 = no constraint).
            morse_alpha: Legacy Morse alpha [1/Å]. Also default beta for unified if beta=None.
            z_target:    Target z for harmonic constraint.
            Hs:   Sigma-hole pseudo-charge (positive).
            epair_dist, sigma_dist: stored for GUI / rebuild.
            mode: 'legacy' or 'unified' (default: keep current pairff_mode).
            beta: Compact-exp beta for unified mode (default: morse_alpha, or 1.7).
        """
        if mode is not None:
            mode = str(mode).lower()
            if mode not in (PAIRFF_MODE_LEGACY, PAIRFF_MODE_UNIFIED):
                raise ValueError(f"Unknown pairff mode {mode!r}; use 'legacy' or 'unified'")
            prev = self.pairff_mode
            self.pairff_mode = mode
            if prev != mode:
                # Mode discontinuity: clear velocities / FIRE state
                z = np.zeros((self.n_bodies, 4), dtype=np.float32)
                self.toGPU('vposs', z)
                self.toGPU('vrots', z)
                if 'fire_state' in self.buffer_dict:
                    self.toGPU('fire_state', z)
                if hasattr(self, 'reset_optimizer_state'):
                    self.reset_optimizer_state()

        if beta is None:
            beta = float(morse_alpha) if self.pairff_mode == PAIRFF_MODE_LEGACY else 1.7

        # Refresh REQ.z (He/Hs) and REQ.w (blunt width) then re-upload
        if self.allmol_mode and self._mb_packs is not None:
            for pack in self._mb_packs:
                pack['REQ_ext'] = _extend_reqs_with_epairs(
                    pack['REQ_base'], pack['enames'], pack['types'], He=He, Hs=Hs, w=w)
            req_flat = np.vstack([p['REQ_ext'] for p in self._mb_packs]).astype(np.float32)
            type_flat = np.concatenate([p['types'] for p in self._mb_packs]).astype(np.int32)
            self.upload_dyn_types_req(type_flat, req_flat)
            self._apply_active_view_hosts()
            self._refresh_static_view_hosts()
        elif self.env_mode and self._mb_packs is not None:
            for pack in self._mb_packs:
                pack['REQ_ext'] = _extend_reqs_with_epairs(
                    pack['REQ_base'], pack['enames'], pack['types'], He=He, Hs=Hs, w=w)
            self._rebuild_env_from_mb()
            pack_a = self._mb_packs[self.active_body]
            self._dyn_REQ_base = pack_a['REQ_base']
            self._dyn_enames = list(pack_a['enames'])
            self.upload_dyn_types_req(pack_a['types'], pack_a['REQ_ext'])
        else:
            if self.dyn_type_host is not None and self._dyn_REQ_base is not None:
                dyn_REQ = _extend_reqs_with_epairs(
                    self._dyn_REQ_base, self._dyn_enames, self.dyn_type_host, He=He, Hs=Hs, w=w)
                self.upload_dyn_types_req(self.dyn_type_host, dyn_REQ)
            if self.static_type_host is not None and self._static_REQ_base is not None:
                static_REQ = _extend_reqs_with_epairs(
                    self._static_REQ_base, self._static_enames, self.static_type_host, He=He, Hs=Hs, w=w)
                self.upload_static(self.static_apos_host, static_REQ, self.static_type_host)

        self.kernel_params['n_static'] = np.int32(self.static_n)
        n_static_atoms = int((self.static_type_host == 0).sum()) if self.static_type_host is not None else 0
        self.kernel_params['n_static_atoms'] = np.int32(n_static_atoms)
        n_dyn_atoms = int((self.dyn_type_host == 0).sum()) if self.dyn_type_host is not None else 0
        self.kernel_params['n_dyn_atoms'] = np.int32(n_dyn_atoms)
        self.kernel_params['n_env_mols'] = np.int32(self.n_env_mols)
        self.kernel_params['n_mols'] = np.int32(self.n_bodies)
        self.kernel_params['active_mol'] = np.int32(self.active_body)
        self.kernel_params['pairff_params'] = np.array([He, rc, w, k_z], dtype=np.float32)
        self.kernel_params['morse_alpha'] = np.float32(morse_alpha)
        self.kernel_params['beta'] = np.float32(beta)
        self.kernel_params['z_target'] = np.float32(z_target)
        self.kernel_params['Hs'] = np.float32(Hs)
        self.kernel_params['md_params'] = np.array([0.92, 0.88, 1.0, 1.0], dtype=np.float32)
        self.pairff_params_host = {
            'He': He, 'rc': rc, 'w': w, 'k_z': k_z, 'morse_alpha': morse_alpha,
            'beta': beta, 'z_target': z_target, 'Hs': Hs,
            'epair_dist': epair_dist, 'sigma_dist': sigma_dist,
            'mode': self.pairff_mode,
            'env_mode': self.env_mode,
            'allmol_mode': self.allmol_mode,
            'active_body': self.active_body,
        }
        if self.allmol_mode:
            self.pairff_allmol_args = self.generate_kernel_args("rigid_body_pairff_unified_allmol_kernel")
            self.krnl_pairff_allmol = cl.Kernel(self.prg, "rigid_body_pairff_unified_allmol_kernel")
            self.pairff_allmol_faf_args = None
            self.krnl_pairff_allmol_faf = None
        elif self.env_mode:
            self.pairff_env_args = self.generate_kernel_args("rigid_body_pairff_unified_env_kernel")
            self.krnl_pairff_env = cl.Kernel(self.prg, "rigid_body_pairff_unified_env_kernel")
            self.pairff_env_faf_args = None
            self.krnl_pairff_env_faf = None
        else:
            self.pairff_args = self.generate_kernel_args("rigid_body_pairff_kernel")
            self.krnl_pairff = cl.Kernel(self.prg, "rigid_body_pairff_kernel")
            self.pairff_unified_args = self.generate_kernel_args("rigid_body_pairff_unified_kernel")
            self.krnl_pairff_unified = cl.Kernel(self.prg, "rigid_body_pairff_unified_kernel")
            self.pairff_unified_faf_args = None
            self.krnl_pairff_unified_faf = None
        # Keep FAF bound across param refreshes (GUI spinboxes call init_pairff)
        if self.faf_mode and getattr(self, 'folded_params', None) is not None:
            self._bind_pairff_faf_kernels()

    def _bind_pairff_faf_kernels(self):
        """Generate fused PairFF+FAF kernel args (needs folded_meta / folded_lvec2d in kernel_params)."""
        if 'folded_meta' not in self.kernel_params or 'folded_lvec2d' not in self.kernel_params:
            raise RuntimeError("_bind_pairff_faf_kernels requires init_folded(...) first")
        if self.allmol_mode:
            self.pairff_allmol_faf_args = self.generate_kernel_args("rigid_body_pairff_unified_allmol_faf_kernel")
            self.krnl_pairff_allmol_faf = cl.Kernel(self.prg, "rigid_body_pairff_unified_allmol_faf_kernel")
        elif self.env_mode:
            self.pairff_env_faf_args = self.generate_kernel_args("rigid_body_pairff_unified_env_faf_kernel")
            self.krnl_pairff_env_faf = cl.Kernel(self.prg, "rigid_body_pairff_unified_env_faf_kernel")
        else:
            self.pairff_unified_faf_args = self.generate_kernel_args("rigid_body_pairff_unified_faf_kernel")
            self.krnl_pairff_unified_faf = cl.Kernel(self.prg, "rigid_body_pairff_unified_faf_kernel")

    def enable_pairff_faf(self, enabled=True):
        """Switch run_pairff between PairFF-only and fused PairFF+FAF kernels.

        Requires prior init_folded(...) with folded_atom_type length == total_atoms
        (use -1 for epair/σ sites so FAF is skipped). Prefer k_z=0 when FAF is on.
        """
        self.faf_mode = bool(enabled)
        if self.faf_mode:
            if getattr(self, 'folded_params', None) is None:
                raise RuntimeError("enable_pairff_faf(True) requires init_folded(...) first")
            self._bind_pairff_faf_kernels()

    @staticmethod
    def folded_types_for_pairff_sites(dyn_type, real_folded_types):
        """Map PairFF sites → FAF type indices: real atoms from real_folded_types, dummies → -1."""
        dyn_type = np.asarray(dyn_type, dtype=np.int32).ravel()
        real_folded_types = np.asarray(real_folded_types, dtype=np.int32).ravel()
        n_real = int((dyn_type == 0).sum())
        if real_folded_types.shape[0] != n_real:
            raise ValueError(f"real_folded_types length {real_folded_types.shape[0]} != n_real_atoms {n_real}")
        out = np.full(dyn_type.shape[0], -1, dtype=np.int32)
        out[dyn_type == 0] = real_folded_types
        return out

    def _folded_types_all_sites(self, fit_result):
        """Build folded_atom_type length == total_atoms (dummies -1) from fit atom_type_ids."""
        at_fit = np.asarray(fit_result['atom_type_ids'], dtype=np.int32).ravel()
        if self.allmol_mode and self._mb_packs is not None:
            parts = []
            for pack in self._mb_packs:
                parts.append(self.folded_types_for_pairff_sites(pack['types'], at_fit))
            out = np.concatenate(parts).astype(np.int32)
        else:
            out = self.folded_types_for_pairff_sites(self.dyn_type_host, at_fit)
        if out.shape[0] != self.total_atoms:
            raise ValueError(f"folded types length {out.shape[0]} != total_atoms {self.total_atoms}")
        return out

    def attach_pairff_faf(self, fit_result, z_init=3.5, k_z=0.0, enable=True):
        """Bind FAF substrate to this PairFF instance (dynamics + Vispy map).

        Raises all body CoMs to Z_SURF_TOP + z_init, sets k_z (default 0),
        init_folded with dummy sites skipped, stores ``faf_fit`` for map compose.
        Fit ``atom_type_ids`` must match real-atom count of each molecule pack
        (identical copies OK).
        """
        from spammm.surfaces.FoldedRigid import Z_SURF_TOP
        fit_result = dict(fit_result)
        z = float(Z_SURF_TOP + z_init)
        out = self.download_outputs()
        pos = out['pos'].copy()
        pos[:, 2] = z
        self.toGPU('poss', pos)
        if self._mb_pos is not None:
            self._mb_pos[:, 2] = z
        # Refresh PairFF params (k_z / z_target) without wiping FAF later
        ph = self.pairff_params_host or {}
        self.init_pairff(
            He=ph.get('He', -1.0), rc=ph.get('rc', 3.0), w=ph.get('w', 0.7),
            k_z=float(k_z), morse_alpha=ph.get('morse_alpha', 1.8),
            z_target=z, Hs=ph.get('Hs', 1.0),
            epair_dist=ph.get('epair_dist', 1.4), sigma_dist=ph.get('sigma_dist', 1.0),
            mode=self.pairff_mode, beta=ph.get('beta', 1.7),
        )
        atype = self._folded_types_all_sites(fit_result)
        coeffs = np.asarray(fit_result['coeffs'], dtype=np.float32)
        kxyz = np.asarray(fit_result['basis_params'], dtype=np.float32)
        lvec2d = np.asarray(fit_result['folded_lvec2d'], dtype=np.float32)
        ntypes, nbasis = coeffs.shape
        self.init_folded(coeffs, kxyz, atype, lvec2d,
                         folded_meta=np.array([nbasis, ntypes, 0, 0], dtype=np.int32))
        self.faf_fit = fit_result
        self.map_z = z
        if enable:
            self.enable_pairff_faf(True)
        if self.allmol_mode:
            self._refresh_static_view_hosts()
        return self

    def run_pairff(self, num_steps, dt, lin_damp=0.92, ang_damp=0.88, fire=False, faf=None):
        """Run PairFF (optional fused FAF) for num_steps inside one GPU launch.

        faf: None → use self.faf_mode; True/False overrides for this call.
        allmol_mode: 1 workgroup, active_mol index; all poses stay on GPU.
        """
        use_faf = self.faf_mode if faf is None else bool(faf)
        if use_faf and getattr(self, 'folded_params', None) is None:
            raise RuntimeError("FAF PairFF run requires init_folded(...) first")
        if self.allmol_mode:
            if self.pairff_mode != PAIRFF_MODE_UNIFIED:
                raise RuntimeError("allmol kernel requires pairff_mode='unified'")
            if use_faf:
                if self.pairff_allmol_faf_args is None:
                    raise RuntimeError("allmol+FAF kernel not bound; call enable_pairff_faf(True)")
                kname = "rigid_body_pairff_unified_allmol_faf_kernel"
                krnl = self.krnl_pairff_allmol_faf
            else:
                if self.pairff_allmol_args is None:
                    raise RuntimeError("allmol PairFF kernel not initialized; call init_pairff(...) first")
                kname = "rigid_body_pairff_unified_allmol_kernel"
                krnl = self.krnl_pairff_allmol
        elif self.env_mode:
            if self.pairff_mode != PAIRFF_MODE_UNIFIED:
                raise RuntimeError("Multi-body env kernel requires pairff_mode='unified'")
            if use_faf:
                if self.pairff_env_faf_args is None:
                    raise RuntimeError("Env+FAF PairFF kernel not initialized; call init_pairff(...) first")
                kname = "rigid_body_pairff_unified_env_faf_kernel"
                krnl = self.krnl_pairff_env_faf
            else:
                if self.pairff_env_args is None:
                    raise RuntimeError("Env PairFF kernel not initialized; call init_pairff(...) first")
                kname = "rigid_body_pairff_unified_env_kernel"
                krnl = self.krnl_pairff_env
        elif self.pairff_mode == PAIRFF_MODE_UNIFIED:
            if use_faf:
                if self.pairff_unified_faf_args is None:
                    raise RuntimeError("Unified+FAF PairFF kernel not initialized; call init_pairff(...) first")
                kname = "rigid_body_pairff_unified_faf_kernel"
                krnl = self.krnl_pairff_unified_faf
            else:
                if self.pairff_unified_args is None:
                    raise RuntimeError("Unified PairFF kernel not initialized; call init_pairff(...) first")
                kname = "rigid_body_pairff_unified_kernel"
                krnl = self.krnl_pairff_unified
        else:
            if use_faf:
                raise RuntimeError("FAF fusion requires pairff_mode='unified' (legacy not supported)")
            if self.pairff_args is None:
                raise RuntimeError("PairFF kernel not initialized; call init_pairff(...) first")
            kname = "rigid_body_pairff_kernel"
            krnl = self.krnl_pairff
        self.kernel_params['dt'] = np.float32(dt)
        self.kernel_params['niter'] = np.int32(num_steps)
        self.kernel_params['n_env_mols'] = np.int32(self.n_env_mols)
        self.kernel_params['n_mols'] = np.int32(self.n_bodies)
        self.kernel_params['active_mol'] = np.int32(self.active_body)
        if fire:
            self.kernel_params['md_params'] = np.array([lin_damp, ang_damp, 1.0, -1.0], dtype=np.float32)
        else:
            self.kernel_params['md_params'] = np.array([lin_damp, ang_damp, 1.0, 1.0], dtype=np.float32)
        args = self.generate_kernel_args(kname)
        if self.allmol_mode:
            if use_faf:
                self.pairff_allmol_faf_args = args
            else:
                self.pairff_allmol_args = args
            # One workgroup for the active molecule (generalizable to N later)
            global_size = (self.roundUpGlobalSize(self.nloc),)
        elif self.env_mode:
            if use_faf:
                self.pairff_env_faf_args = args
            else:
                self.pairff_env_args = args
            global_size = (self.roundUpGlobalSize(self.n_bodies * self.nloc),)
        elif self.pairff_mode == PAIRFF_MODE_UNIFIED:
            if use_faf:
                self.pairff_unified_faf_args = args
            else:
                self.pairff_unified_args = args
            global_size = (self.roundUpGlobalSize(self.n_bodies * self.nloc),)
        else:
            self.pairff_args = args
            global_size = (self.roundUpGlobalSize(self.n_bodies * self.nloc),)
        local_size = (self.nloc,)
        krnl(self.queue, global_size, local_size, *args)
        self.queue.finish()

    def relax_pairff(self, max_steps=500, dt=0.05, damp0=0.1, f_tol=1e-4, t_tol=1e-4, batch=64, body_id=None, record=False):
        """FIRE relaxation using the pairff kernel with host-side early exit.

        body_id: which pose to monitor (default: active_body). In allmol_mode,
        atom_positions download is already the active slice at index 0.
        """
        if body_id is None:
            body_id = int(getattr(self, 'active_body', 0))
        else:
            body_id = int(body_id)
        hist = [] if record else None
        n_done = 0
        Fmag = Tmag = E = np.inf
        batch = max(int(batch), 64)
        while n_done < max_steps:
            n = min(batch, max_steps - n_done)
            self.run_pairff(n, dt, lin_damp=damp0, ang_damp=damp0, fire=True)
            n_done += n
            out = self.download_outputs()
            F = out['body_force'][body_id, :3].astype(np.float64)
            Tw = out['body_torque'][body_id, :3].astype(np.float64)
            # allmol download_outputs returns active sites as atom_positions[0]
            atom_E = out['atom_positions'][0] if getattr(self, 'allmol_mode', False) else out['atom_positions'][body_id]
            E = float(atom_E[:, 3].sum())
            Fmag = float(np.linalg.norm(F))
            Tmag = float(np.linalg.norm(Tw))
            if record:
                hist.append(dict(step=n_done, E=E, F=Fmag, T=Tmag,
                                 pos=out['pos'][body_id].copy(), quat=out['quats'][body_id].copy()))
            if Fmag < f_tol and Tmag < t_tol:
                break
        z = np.zeros((self.n_bodies, 4), dtype=np.float32)
        self.toGPU('vposs', z); self.toGPU('vrots', z); self.queue.finish()
        return {'steps': n_done, 'converged': Fmag < f_tol and Tmag < t_tol,
                'F': Fmag, 'T': Tmag, 'E': E, 'history': hist}

    def world_sites_all_bodies(self, real_only=False):
        """Host world-frame sites for every multi-body molecule (syncs active pose)."""
        if self._mb_packs is None:
            raise RuntimeError("world_sites_all_bodies requires from_molecules()")
        self.sync_active_pose_from_gpu()
        out = []
        for j, pack in enumerate(self._mb_packs):
            world = _body_sites_world(pack['rel'], self._mb_pos[j], self._mb_quat[j])
            types = pack['types']
            enames = pack['enames']
            if real_only:
                m = types == 0
                world = world[m]
                enames = [e for e, t in zip(enames, types) if t == 0]
                types = types[m]
            out.append({'world': world, 'enames': list(enames), 'types': np.asarray(types), 'pos': self._mb_pos[j].copy(), 'quat': self._mb_quat[j].copy()})
        return out

    def tip_pull_scan(self, pin_local_idx, path, k_spring=20.0, n_relax=100, dt=0.02,
                      fire=True, record_every=1):
        """AFM-like tip pull: spring on one active-molecule atom, move target along path.

        ``pin_local_idx`` is local to the active molecule (0..na-1). In allmol_mode the
        GPU anchor is written at ``mol_offsets[active] + pin_local_idx``.
        Uses ``run_pairff`` (optional FAF). Inactive bodies stay frozen but force the active one.

        Returns dict with CoM/quat/pin trails, per-frame world sites (real atoms), tip path.
        """
        path = np.asarray(path, dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 3:
            raise ValueError(f"path must be (N,3), got {path.shape}")
        a = int(self.active_body)
        i0 = int(self.mol_offsets[a]) if getattr(self, 'allmol_mode', False) and self.mol_offsets is not None else 0
        gi = i0 + int(pin_local_idx)
        if gi < 0 or gi >= self.total_atoms:
            raise ValueError(f"pin global index {gi} out of range [0,{self.total_atoms})")

        positions, quats, pin_xyz, E_list, frames = [], [], [], [], []
        for ip, target in enumerate(path):
            anchors = np.zeros((self.total_atoms, 4), dtype=np.float32)
            anchors[:, 3] = -1.0
            anchors[gi, :3] = target
            anchors[gi, 3] = float(k_spring)
            self.anchors = anchors
            self.upload_anchors()
            self.run_pairff(int(n_relax), float(dt), lin_damp=0.9, ang_damp=0.88, fire=fire)
            out = self.download_outputs()
            sites = self.world_sites_all_bodies(real_only=True)
            pin_w = sites[a]['world'][int(pin_local_idx), :3].copy()
            # Energy of active sites (PairFF+FAF stored in apos_world.w)
            atom_E = out['atom_positions'][0]
            E = float(atom_E[:, 3].sum())
            if (ip % max(int(record_every), 1)) == 0:
                positions.append(out['pos'][a, :3].copy())
                quats.append(out['quats'][a].copy())
                pin_xyz.append(pin_w)
                E_list.append(E)
                frames.append(sites)
            print(f"  tip_pull {ip+1}/{len(path)}  CoM={out['pos'][a,:3]}  pin={pin_w}  E={E:.4f}")

        # clear tip spring
        anc = np.zeros((self.total_atoms, 4), dtype=np.float32)
        anc[:, 3] = -1.0
        self.anchors = anc
        self.upload_anchors()
        return {
            'path': path,
            'pos': np.asarray(positions, dtype=np.float32),
            'quat': np.asarray(quats, dtype=np.float32),
            'pin': np.asarray(pin_xyz, dtype=np.float32),
            'E': np.asarray(E_list, dtype=np.float64),
            'frames': frames,
            'active_body': a,
            'pin_local_idx': int(pin_local_idx),
            'k_spring': float(k_spring),
        }

    @classmethod
    def from_two_molecules(cls, dyn_apos, dyn_enames, dyn_REQs, static_apos, static_enames, static_REQs,
                           n_bodies=1, body_pos=None, quat=None, mass_trans=1.0, mass_rot=None,
                           He=-0.1, rc=3.0, w=0.7, k_z=0.0, morse_alpha=1.8, z_target=None,
                           epair_dist=1.4, sigma_dist=1.0, Hs=1.0,
                           mode=PAIRFF_MODE_LEGACY, beta=None,
                           debug=False, type_map=None):
        """Build RigidBodyPairFF from two molecules (positions + names + REQs).

        Electron pairs are automatically added to both molecules.
        The dynamic molecule is centered at body_pos (default: origin).
        The static molecule stays at its given positions.

        Args:
            dyn_apos:    (N_dyn, 3) positions of dynamic molecule atoms
            dyn_enames:  list of element names for dynamic molecule
            dyn_REQs:    (N_dyn, 4) REQ parameters (R, sqrt(E), Q, H)
            static_apos: (N_stat, 3) positions of static molecule atoms
            static_enames: list of element names for static molecule
            static_REQs: (N_stat, 4) REQ parameters
            body_pos:    (3,) initial CoM position for dynamic body
            quat:        (4,) initial quaternion (default: identity)
            mass_trans:  translational mass scaling
            mass_rot:    rotational mass scaling (default: same as mass_trans)
            He, rc, w, k_z, morse_alpha, z_target: pairff parameters
            mode:        'legacy' or 'unified' force kernel
            beta:        compact-exp beta for unified mode (default 1.7)
        """
        dyn_REQs_base = np.asarray(dyn_REQs, dtype=np.float32).copy()
        static_REQs_base = np.asarray(static_REQs, dtype=np.float32).copy()
        dyn_apos, dyn_enames, dyn_types = add_electron_pairs_via_atomic_system(dyn_apos, dyn_enames, epair_dist=epair_dist, sigma_dist=sigma_dist)
        static_apos, static_enames, static_types = add_electron_pairs_via_atomic_system(static_apos, static_enames, epair_dist=epair_dist, sigma_dist=sigma_dist)
        dyn_REQs_ext = _extend_reqs_with_epairs(dyn_REQs_base, dyn_enames, dyn_types, He=He, Hs=Hs, w=w)
        static_REQs_ext = _extend_reqs_with_epairs(static_REQs_base, static_enames, static_types, He=He, Hs=Hs, w=w)
        masses = _guess_mass([e for e, t in zip(dyn_enames, dyn_types) if t == 0])
        dyn_apos = np.asarray(dyn_apos, dtype=np.float32)
        com0 = (dyn_apos[:len(masses)] * masses[:, None]).sum(axis=0) / masses.sum()
        rel = dyn_apos - com0[None, :]
        mtot, I, Iinv = compute_mass_properties(rel[:len(masses)], masses)
        if mass_rot is None: mass_rot = mass_trans
        Iinv_relax = Iinv * (mtot / float(mass_rot))
        I_relax = I * (float(mass_rot) / mtot)
        if body_pos is None: body_pos = np.zeros(3, dtype=np.float32)
        if quat is None: quat = np.array([0, 0, 0, 1], dtype=np.float32)
        if z_target is None: z_target = float(body_pos[2]) if len(body_pos) >= 3 else 0.0
        n_dyn = len(dyn_enames)
        pos4 = np.zeros((n_bodies, 4), dtype=np.float32)
        pos4[:, :3] = np.asarray(body_pos, dtype=np.float32)
        pos4[:, 3] = float(mass_trans)
        quat4 = np.zeros((n_bodies, 4), dtype=np.float32)
        quat4[:] = np.asarray(quat, dtype=np.float32)
        if quat4[0, 3] == 0 and np.linalg.norm(quat4[0]) < 1e-10: quat4[:, 3] = 1.0
        zero4 = np.zeros((n_bodies, 4), dtype=np.float32)
        atom_body = np.repeat(rel[None, :, :], n_bodies, axis=0).astype(np.float32)
        rbd = cls(debug=debug)
        rbd.pairff_mode = mode
        rbd.realloc(n_bodies=n_bodies, num_atoms=n_dyn)
        rbd.enames = list(dyn_enames)
        rbd.atom_REQ = dyn_REQs_ext.copy()
        rbd.atom_masses = masses.copy()
        rbd.mass_physical = float(mtot)
        rbd.mass_trans = float(mass_trans)
        rbd.mass_rot = float(mass_rot)
        rbd.upload_state(pos4, quat4, zero4, zero4, float(mass_trans), 1.0/float(mass_trans),
                         np.repeat(Iinv_relax[None,:,:], n_bodies, axis=0), atom_body,
                         inertia=np.repeat(I_relax[None,:,:], n_bodies, axis=0))
        rbd.alloc_pairff(n_static=len(static_enames))
        rbd._dyn_REQ_base = dyn_REQs_base
        rbd._static_REQ_base = static_REQs_base
        rbd._dyn_enames = list(dyn_enames)
        rbd._static_enames = list(static_enames)
        rbd.upload_static(static_apos, static_REQs_ext, static_types)
        rbd.upload_dyn_types_req(dyn_types, dyn_REQs_ext)
        rbd.init_pairff(He=He, rc=rc, w=w, k_z=k_z, morse_alpha=morse_alpha, z_target=z_target, Hs=Hs,
                        epair_dist=epair_dist, sigma_dist=sigma_dist, mode=mode, beta=beta)
        rbd.static_enames = list(static_enames)
        return rbd

    @classmethod
    def from_molecules(cls, molecules, body_positions, quats=None, active_body=0,
                       mass_trans=1.0, mass_rot=None,
                       He=-0.1, rc=3.0, w=0.7, k_z=0.0, morse_alpha=1.8, z_target=None,
                       epair_dist=1.4, sigma_dist=1.0, Hs=1.0, beta=None,
                       debug=False):
        """Build multi-body PairFF with shared all-molecule GPU buffers.

        All molecules stay resident (poses + body-frame sites). Only ``active_body``
        integrates; neighbors contribute via tiled PairFF (j != active). Switching
        active is an index write — no realloc / env rebuild. Dynamics for all
        molecules remain persistent (ready for future all-mobile MD).
        """
        n = len(molecules)
        if n < 1:
            raise ValueError("from_molecules requires at least one molecule")
        body_positions = np.asarray(body_positions, dtype=np.float32)
        if body_positions.shape != (n, 3):
            raise ValueError(f"body_positions shape {body_positions.shape} != ({n},3)")
        if quats is None:
            quats = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n, 1))
        else:
            quats = np.asarray(quats, dtype=np.float32)
            if quats.shape != (n, 4):
                raise ValueError(f"quats shape {quats.shape} != ({n},4)")
        active_body = int(active_body)
        if active_body < 0 or active_body >= n:
            raise ValueError(f"active_body={active_body} out of range [0,{n})")
        if z_target is None:
            z_target = float(body_positions[active_body, 2])
        if mass_rot is None:
            mass_rot = mass_trans
        if beta is None:
            beta = 1.7

        packs = []
        for mol in molecules:
            if isinstance(mol, dict):
                apos, enames, REQs = mol['apos'], mol['enames'], mol['REQs']
            else:
                apos, enames, REQs = mol
            packs.append(_prepare_molecule_pack(
                apos, enames, REQs, epair_dist=epair_dist, sigma_dist=sigma_dist,
                He=He, Hs=Hs, w=w))

        counts = np.array([len(p['types']) for p in packs], dtype=np.int32)
        rbd = cls(debug=debug)
        rbd.pairff_mode = PAIRFF_MODE_UNIFIED
        rbd.allmol_mode = True
        rbd.env_mode = False
        rbd._mb_packs = packs
        rbd._mb_pos = body_positions.copy()
        rbd._mb_quat = quats.copy()
        rbd.active_body = active_body
        rbd.realloc_molecules(counts)

        # Flat body-frame sites + REQ/type for all molecules
        rel_flat = np.vstack([p['rel'] for p in packs]).astype(np.float32)
        req_flat = np.vstack([p['REQ_ext'] for p in packs]).astype(np.float32)
        type_flat = np.concatenate([p['types'] for p in packs]).astype(np.int32)
        apos4 = np.zeros((rbd.total_atoms, 4), dtype=np.float32)
        apos4[:, :3] = rel_flat[:, :3]
        rbd.toGPU('apos_body', apos4)
        rbd.toGPU('apos_world', np.zeros((rbd.total_atoms, 4), dtype=np.float32))
        rbd.toGPU('atom_force', np.zeros((rbd.total_atoms, 4), dtype=np.float32))
        rbd.toGPU('anchors', np.zeros((rbd.total_atoms, 4), dtype=np.float32))
        rbd.upload_dyn_types_req(type_flat, req_flat)

        pos4 = np.zeros((n, 4), dtype=np.float32)
        pos4[:, :3] = body_positions
        pos4[:, 3] = float(mass_trans)
        quat4 = quats.astype(np.float32).copy()
        zero4 = np.zeros((n, 4), dtype=np.float32)
        Iinv = np.zeros((n, 3, 3), dtype=np.float32)
        I = np.zeros((n, 3, 3), dtype=np.float32)
        for i, p in enumerate(packs):
            Iinv[i] = p['Iinv'] * (p['mtot'] / float(mass_rot))
            I[i] = p['I'] * (float(mass_rot) / p['mtot'])
        rbd.toGPU('poss', pos4)
        rbd.toGPU('qrots', quat4)
        rbd.toGPU('vposs', zero4)
        rbd.toGPU('vrots', zero4)
        rbd.toGPU('fire_state', zero4)
        rbd.toGPU('body_force', zero4)
        rbd.toGPU('body_torque', zero4)
        rbd.toGPU('I_body_inv', _ensure_cl_mat3(Iinv, n))
        rbd.toGPU('I_body', _ensure_cl_mat3(I, n))
        rbd.mass_trans = float(mass_trans)
        rbd.mass_rot = float(mass_rot)
        rbd.kernel_params['n_mols'] = np.int32(n)
        rbd.kernel_params['active_mol'] = np.int32(active_body)
        rbd._apply_active_view_hosts()
        rbd._refresh_static_view_hosts()
        rbd.init_pairff(He=He, rc=rc, w=w, k_z=k_z, morse_alpha=morse_alpha, z_target=z_target,
                        Hs=Hs, epair_dist=epair_dist, sigma_dist=sigma_dist,
                        mode=PAIRFF_MODE_UNIFIED, beta=beta)
        # n_env_mols used by Vispy label / debug: neighbors count
        rbd.n_env_mols = n - 1
        rbd.n_env_sites = int(rbd.static_n)
        return rbd


def _extend_reqs_with_epairs(reqs, enames, types, He=0.0, Hs=0.0, w=0.0):
    """Extend REQ array with dummy entries for electron pair / sigma hole atoms.

    Real atoms (type=0) keep their original [RvdW, sqrt(EvdW), Q, Hb] values
    with .w forced to 0 (no soft-radius blunt).
    Epairs (type=1) get R=0, E=0, Q=He, w=w — the pseudo-charge in Q (REQ.z)
    drives Hbond / unified attraction; REQ.w is the soft-radius blunt width.
    Sigma holes (type=2) get R=0, E=0, Q=Hs, w=w — same mechanism with Hs.

    Storing the pseudo-charge in REQ.z enables branch-free GPU execution:
    coeff = min(0, Qi*Qj) / attr = -min(0,Qi*Qj) works uniformly.
    """
    reqs = np.asarray(reqs, dtype=np.float32)
    n_total = len(types)
    out = np.zeros((n_total, 4), dtype=np.float32)
    ia = 0
    for i in range(n_total):
        if types[i] == 0:
            out[i] = reqs[ia]
            out[i, 3] = 0.0  # blunt width 0 for real atoms
            ia += 1
        elif types[i] == 1:
            out[i] = np.array([0.0, 0.0, He, w], dtype=np.float32)
        elif types[i] == 2:
            out[i] = np.array([0.0, 0.0, Hs, w], dtype=np.float32)
    return out
