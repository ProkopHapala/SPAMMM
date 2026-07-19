"""Projective Jacobi relax — UFF→K12/K13/K14 springs + soft FAF outer (3rd path beside UFF/SPFF).

# === AUTO-DOC BEGIN ===
Essence: hard intramolecular geometry as distance springs; soft substrate drives a large outer
predictor; inner diagonal Jacobi with M/dt². Surrogate for fast GUI/adsorbate morphing — not
energy-parity with UFF/SPFF.

Why not mass scaling: soft FAF and stiff bonds share the same atoms; inflating mass slows both.

Key APIs: build_linearized_from_uff, LFFSolver.from_uff / upload_folded_fit / relax.
Landmine: K14 must use capped K + geometry-based l0 (raw V/(dl/dφ)² crumples PAHs).
Topic: doc/Topics/ForceFields/LFF_ProjectiveRelax.md · Kernel: kernels/LFF.cl
# === AUTO-DOC END ===
"""
import os
from dataclasses import dataclass

import numpy as np
import pyopencl as cl
from pyopencl import cltypes

from ..utils.OpenCLBase import OpenCLBase

DEFAULT_WORKGROUP_SIZE = 64
MAX_NEIGHBORS = 24
LFF_WG_SIZE = 64
FAF_BASIS_MAX = 128
FAF_TYPES_MAX = 8


def _ensure_float4(arr, *, w_default=0.0):
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.shape}")
    if data.shape[1] == 3:
        data = np.hstack((data, np.full((data.shape[0], 1), np.float32(w_default), dtype=np.float32)))
    if data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns, got {data.shape}")
    return np.ascontiguousarray(data, dtype=np.float32)


def _pack_float3(vec):
    data = np.asarray(vec, dtype=np.float32).reshape(3)
    out = np.zeros(4, dtype=np.float32)
    out[:3] = data
    return out


def _law_of_cosines(a, b, cos_t):
    return float(np.sqrt(max(a * a + b * b - 2.0 * a * b * cos_t, 1e-12)))


def _dihedral_end_l0(rab, rbc, rcd, theta_abc, theta_bcd, phi0):
    """Equilibrium |a-d| for dihedral a-b-c-d (radians)."""
    # place b at origin, c on +x, a in xy, d with dihedral phi0
    ca, sa = np.cos(theta_abc), np.sin(theta_abc)
    cb, sb = np.cos(theta_bcd), np.sin(theta_bcd)
    cp, sp = np.cos(phi0), np.sin(phi0)
    # a in plane: from b toward -x tilted by (pi-theta)
    ax = -rab * ca
    ay = rab * sa
    az = 0.0
    cx, cy, cz = rbc, 0.0, 0.0
    # d from c
    dx = cx + rcd * cb  # along -bc direction? b->c is +x; angle at c between b-c-d
    # vector c->b = (-rbc,0,0); angle at c: place d in plane then rotate by phi around bc
    # local: in b-c-d plane before twist: d_local = (rcd*cos(pi-theta_bcd), rcd*sin(...)) = (-rcd*cb, rcd*sb)
    d_rel = np.array([-rcd * cb, rcd * sb * cp, rcd * sb * sp], dtype=np.float64)
    d = np.array([cx, cy, cz]) + d_rel
    a = np.array([ax, ay, az])
    return float(np.linalg.norm(d - a))


def _add_spring(bond_map, i, j, l0, k, tag):
    if k is None or not np.isfinite(k) or k <= 0.0:
        return
    if l0 is None or not np.isfinite(l0) or l0 <= 0.0:
        return
    key = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
    e = bond_map.get(key)
    if e is None:
        bond_map[key] = {'l0': float(l0), 'k': float(k), 'tags': [tag]}
        return
    pk, nk = e['k'], float(k)
    new_k = pk + nk
    e['l0'] = (e['l0'] * pk + float(l0) * nk) / new_k
    e['k'] = new_k
    e['tags'].append(tag)


def build_linearized_from_uff(uff_data, *, include_angles=True, include_dihedrals=True, k14_scale=1.0, max_neighbors=MAX_NEIGHBORS):
    """Build per-atom neighs/KLs and stick list from UFF arrays (K12/K13/K14)."""
    bonAtoms = np.asarray(uff_data['bonAtoms'], dtype=np.int32)
    bonParams = np.asarray(uff_data['bonParams'], dtype=np.float64)
    natoms = int(len(uff_data['neighs']))
    bond_map = {}
    sticks = []  # (i,j,l0,k,tag) for plotting

    # bond length lookup
    bl = {}
    for ib in range(len(bonAtoms)):
        i, j = int(bonAtoms[ib, 0]), int(bonAtoms[ib, 1])
        k, l0 = float(bonParams[ib, 0]), float(bonParams[ib, 1])
        bl[(min(i, j), max(i, j))] = l0
        _add_spring(bond_map, i, j, l0, k, 'bond')

    if include_angles and uff_data.get('angAtoms') is not None and len(uff_data['angAtoms']) > 0:
        angAtoms = np.asarray(uff_data['angAtoms'], dtype=np.int32)
        angParams = np.asarray(uff_data['angParams'], dtype=np.float64)
        for ia in range(len(angAtoms)):
            i, j, k = int(angAtoms[ia, 0]), int(angAtoms[ia, 1]), int(angAtoms[ia, 2])
            # endpoints i–k; central j. Prefer Ass from Fourier: C3=-1,C0=1 => 180°; else use current geometry via bonds
            rij = bl.get((min(i, j), max(i, j)))
            rjk = bl.get((min(j, k), max(j, k)))
            if rij is None or rjk is None:
                continue
            # UFF Ass encoded via C coeffs; for sp2 C3=-1 → theta=180° cos=-1; for tetrahedral use cos from C
            C0, C1, C2, C3 = angParams[ia, 1:5]
            Kang = float(angParams[ia, 0])
            # equilibrium cos from dE/dc=0 of K(C0+C1 c+C2 c2+C3 c3) ≈ prefer Ass of type:
            # use law of cosines with theta from bond vectors of rest: if C3~-1 and C0~1 → planar 120° for aromatic
            if abs(C3 + 1.0) < 1e-3 and abs(C0 - 1.0) < 1e-3:
                cos_t = -0.5  # 120°
            elif abs(C1 - 1.0) < 1e-3 and abs(C0 - 1.0) < 1e-3:
                cos_t = -1.0  # 180° linear
            else:
                # tetrahedral-ish from C1,C2
                cos_t = -C1 / (4.0 * C2) if abs(C2) > 1e-8 else -1.0 / 3.0
                cos_t = float(np.clip(cos_t, -1.0, 1.0))
            l0 = _law_of_cosines(rij, rjk, cos_t)
            # map Fourier K to distance spring (order-1 scale)
            Ks = max(Kang, 1e-3)
            _add_spring(bond_map, i, k, l0, Ks, 'angle')

    if include_dihedrals and uff_data.get('dihAtoms') is not None and len(uff_data['dihAtoms']) > 0:
        dihAtoms = np.asarray(uff_data['dihAtoms'], dtype=np.int32)
        dihParams = np.asarray(uff_data['dihParams'], dtype=np.float64)
        apos = uff_data.get('apos')
        if apos is None:
            apos = None
        for id_ in range(len(dihAtoms)):
            a, b, c, d = (int(dihAtoms[id_, x]) for x in range(4))
            V = float(dihParams[id_, 0])
            if not np.isfinite(V) or V <= 0.0:
                continue
            # Rest length from current geometry (preserves planar PAH shape); mild K from V
            if apos is not None:
                l0 = float(np.linalg.norm(np.asarray(apos[a, :3], dtype=np.float64) - np.asarray(apos[d, :3], dtype=np.float64)))
            else:
                rab = bl.get((min(a, b), max(a, b)))
                rbc = bl.get((min(b, c), max(b, c)))
                rcd = bl.get((min(c, d), max(c, d)))
                if rab is None or rbc is None or rcd is None:
                    continue
                l0 = _dihedral_end_l0(rab, rbc, rcd, np.deg2rad(120.0), np.deg2rad(120.0), np.pi)
            if not np.isfinite(l0) or l0 < 0.5:
                continue
            Kr = float(np.clip(V * 40.0 * float(k14_scale), 5.0, 80.0))
            _add_spring(bond_map, a, d, l0, Kr, 'dihedral')

    # Strengthen weak UFF Fourier→distance angle springs toward bond scale
    for key, info in list(bond_map.items()):
        if 'angle' in info['tags'] and 'bond' not in info['tags']:
            info['k'] = float(np.clip(info['k'] * 8.0, 2.0, 40.0))

    for (i, j), info in sorted(bond_map.items()):
        tag0 = info['tags'][0]
        # prefer most specific label if mixed
        if 'dihedral' in info['tags']:
            tag0 = 'dihedral'
        elif 'angle' in info['tags']:
            tag0 = 'angle'
        sticks.append((i, j, info['l0'], info['k'], tag0))

    neighs = np.full((natoms, max_neighbors), -1, dtype=np.int32)
    KLs = np.zeros((natoms, max_neighbors, 2), dtype=np.float32)
    counts = np.zeros(natoms, dtype=np.int32)
    for i, j, l0, k, _tag in sticks:
        for u, v in ((i, j), (j, i)):
            s = int(counts[u])
            if s >= max_neighbors:
                raise ValueError(f"Atom {u} has >{max_neighbors} LFF springs; raise MAX_NEIGHBORS")
            neighs[u, s] = v
            KLs[u, s, 0] = np.float32(k)
            KLs[u, s, 1] = np.float32(l0)
            counts[u] = s + 1
    return neighs, KLs, sticks, counts


@dataclass
class LFFParams:
    dt: float = 0.05
    n_outer: int = 1
    n_inner: int = 8
    efield: tuple = (0.0, 0.0, 0.0)
    bmix: float = 0.0
    damp: float = 0.9


class LFFSolver(OpenCLBase):
    """GPU LFF projective Jacobi (+ optional FAF)."""

    def __init__(self, *, workgroup_size=DEFAULT_WORKGROUP_SIZE, max_neighbors=MAX_NEIGHBORS, bPrint=False):
        if workgroup_size <= 0 or (workgroup_size & (workgroup_size - 1)) != 0:
            raise ValueError("workgroup_size must be a positive power of two")
        if workgroup_size < LFF_WG_SIZE:
            # kernel local arrays sized LFF_WG_SIZE; launch with that local size
            pass
        super().__init__(nloc=workgroup_size, device_index=0)
        kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kernels')
        if not self.load_program(kernel_path=os.path.join(kernel_dir, 'LFF.cl'), bPrint=bPrint, bMakeHeaders=True,
                                 build_options=[f'-DLFF_WG_SIZE={LFF_WG_SIZE}', f'-DMAX_NEIGHBORS={max_neighbors}']):
            raise RuntimeError("Failed to load LFF.cl")
        self.max_neighbors = int(max_neighbors)
        self.wg = LFF_WG_SIZE
        self.n_mols = 0
        self.nAtomTot = 0
        self.mol_offsets = None
        self.params = LFFParams()
        self.sticks = []
        self.has_reqhs = False
        self._faf_on = False
        self.folded_meta = np.zeros(4, dtype=np.int32)
        self.folded_lvec2d = np.array([1, 0, 0, 1], dtype=np.float32)
        self.kernel_params = {
            'Efield': _pack_float3(self.params.efield),
            'dt': np.float32(self.params.dt),
            'nOuter': np.int32(self.params.n_outer),
            'nInner': np.int32(self.params.n_inner),
            'bMix': np.float32(self.params.bmix),
            'damp': np.float32(self.params.damp),
            'do_faf': np.int32(0),
            'folded_meta': cltypes.make_int4(0, 0, 0, 0),
            'folded_lvec2d': cltypes.make_float4(1, 0, 0, 1),
        }
        self.kernel_args = {}

    def realloc(self, atom_counts):
        mol_sizes = np.asarray(atom_counts, dtype=np.int32).ravel()
        if mol_sizes.size == 0:
            raise ValueError("atom_counts empty")
        if int(mol_sizes.max()) > LFF_WG_SIZE:
            raise ValueError(f"molecule size {mol_sizes.max()} > LFF_WG_SIZE={LFF_WG_SIZE}")
        self.n_mols = mol_sizes.size
        self.mol_offsets = np.zeros(self.n_mols + 1, dtype=np.int32)
        np.cumsum(mol_sizes, out=self.mol_offsets[1:])
        self.nAtomTot = int(self.mol_offsets[-1])
        f32, i32 = 4, 4
        mf = cl.mem_flags
        self.create_buffer('mols', (self.n_mols + 1) * i32, mf.READ_ONLY)
        self.create_buffer('apos', self.nAtomTot * 4 * f32, mf.READ_WRITE)
        self.create_buffer('avel', self.nAtomTot * 4 * f32, mf.READ_WRITE)
        self.create_buffer('REQHs', self.nAtomTot * 4 * f32, mf.READ_ONLY)
        self.create_buffer('neighs', self.nAtomTot * self.max_neighbors * i32, mf.READ_ONLY)
        self.create_buffer('KLs', self.nAtomTot * self.max_neighbors * 2 * f32, mf.READ_ONLY)
        self.create_buffer('fixed_mask', self.nAtomTot * i32, mf.READ_ONLY)
        self.create_buffer('folded_coeffs', FAF_TYPES_MAX * FAF_BASIS_MAX * f32, mf.READ_ONLY)
        self.create_buffer('folded_kxyz', FAF_BASIS_MAX * 4 * f32, mf.READ_ONLY)
        self.create_buffer('folded_atom_type', self.nAtomTot * i32, mf.READ_ONLY)
        self.toGPU('mols', self.mol_offsets)
        self.has_reqhs = False
        self._faf_on = False

    def from_uff(self, uff_cl_or_data, mol=None, *, include_angles=True, include_dihedrals=True, mass=1.0, qs=None):
        """Build springs from UFF_cl instance or uff_data dict; upload positions from mol or uff buffers."""
        if hasattr(uff_cl_or_data, 'toUFF'):
            raise ValueError("pass uff_data dict from toUFF(), not UFF_cl")
        uff_data = dict(uff_cl_or_data)
        natoms = int(len(uff_data['neighs']))
        if mol is not None and 'apos' not in uff_data:
            uff_data['apos'] = np.asarray(mol.apos, dtype=np.float64)
        self.realloc([natoms])
        neighs, KLs, sticks, _ = build_linearized_from_uff(
            uff_data, include_angles=include_angles, include_dihedrals=include_dihedrals,
            max_neighbors=self.max_neighbors)
        self.sticks = sticks
        if mol is not None:
            pos = np.asarray(mol.apos, dtype=np.float32)
            q = np.asarray(mol.qs if qs is None and mol.qs is not None else (qs if qs is not None else np.zeros(natoms)), dtype=np.float32)
            if q.shape[0] != natoms:
                q = np.zeros(natoms, dtype=np.float32)
            pos4 = _ensure_float4(pos, w_default=0.0)
            pos4[:, 3] = q
        else:
            pos4 = np.zeros((natoms, 4), dtype=np.float32)
        vel4 = np.zeros((natoms, 4), dtype=np.float32)
        vel4[:, 3] = np.float32(mass)
        self.upload_state(pos=pos4, vel=vel4, neighs=neighs, KLs=KLs)
        return sticks

    def upload_state(self, *, pos, vel, neighs, KLs, fixed_mask=None, REQHs=None):
        if self.nAtomTot == 0:
            raise RuntimeError("realloc first")
        pos_dev = _ensure_float4(pos)
        vel_dev = _ensure_float4(vel)
        if pos_dev.shape[0] != self.nAtomTot or vel_dev.shape[0] != self.nAtomTot:
            raise ValueError("pos/vel length mismatch")
        neigh_arr = np.ascontiguousarray(neighs, dtype=np.int32)
        KLs_arr = np.ascontiguousarray(KLs, dtype=np.float32)
        if neigh_arr.shape != (self.nAtomTot, self.max_neighbors):
            raise ValueError(f"neighs shape {neigh_arr.shape}")
        if KLs_arr.shape != (self.nAtomTot, self.max_neighbors, 2):
            raise ValueError(f"KLs shape {KLs_arr.shape}")
        fixed_arr = np.zeros(self.nAtomTot, dtype=np.int32) if fixed_mask is None else np.ascontiguousarray(fixed_mask, dtype=np.int32)
        self.toGPU('apos', pos_dev)
        self.toGPU('avel', vel_dev)
        self.toGPU('neighs', neigh_arr)
        self.toGPU('KLs', KLs_arr)
        self.toGPU('fixed_mask', fixed_arr)
        if REQHs is not None:
            self.toGPU('REQHs', _ensure_float4(REQHs))
            self.has_reqhs = True
        self.queue.finish()

    def set_params(self, *, dt=None, n_outer=None, n_inner=None, efield=None, bmix=None, damp=None):
        if dt is not None: self.params.dt = float(dt)
        if n_outer is not None: self.params.n_outer = int(n_outer)
        if n_inner is not None: self.params.n_inner = int(n_inner)
        if efield is not None: self.params.efield = tuple(float(x) for x in efield)
        if bmix is not None: self.params.bmix = float(bmix)
        if damp is not None: self.params.damp = float(damp)
        self.kernel_params['Efield'] = _pack_float3(self.params.efield)
        self.kernel_params['dt'] = np.float32(self.params.dt)
        self.kernel_params['nOuter'] = np.int32(self.params.n_outer)
        self.kernel_params['nInner'] = np.int32(self.params.n_inner)
        self.kernel_params['bMix'] = np.float32(self.params.bmix)
        self.kernel_params['damp'] = np.float32(self.params.damp)

    def upload_folded_fit(self, fit):
        coeffs = np.asarray(fit['coeffs'], dtype=np.float32)
        kxyz = np.asarray(fit['basis_params'], dtype=np.float32)
        atype = np.asarray(fit['atom_type_ids'], dtype=np.int32)
        lvec2d = np.asarray(fit['folded_lvec2d'], dtype=np.float32).reshape(4)
        ntypes, nbasis = int(coeffs.shape[0]), int(coeffs.shape[1])
        if len(atype) != self.nAtomTot:
            raise ValueError(f"atom_type_ids len {len(atype)} != natoms {self.nAtomTot}")
        if nbasis > FAF_BASIS_MAX or ntypes > FAF_TYPES_MAX:
            raise ValueError(f"FAF caps exceeded ntypes={ntypes} nbasis={nbasis}")
        coeff_pad = np.zeros((FAF_TYPES_MAX, FAF_BASIS_MAX), dtype=np.float32)
        coeff_pad[:ntypes, :nbasis] = coeffs
        kxyz_pad = np.zeros((FAF_BASIS_MAX, 4), dtype=np.float32)
        kxyz_pad[:nbasis] = kxyz[:, :4] if kxyz.shape[1] >= 4 else np.hstack([kxyz, np.zeros((nbasis, 4 - kxyz.shape[1]))])
        self.toGPU('folded_coeffs', coeff_pad.ravel())
        self.toGPU('folded_kxyz', kxyz_pad)
        self.toGPU('folded_atom_type', atype)
        self.folded_meta = np.array([nbasis, ntypes, 0, 0], dtype=np.int32)
        self.folded_lvec2d = lvec2d
        self._faf_on = True
        self.queue.finish()

    def run_jacobi(self, *, do_faf=None):
        if self.n_mols == 0:
            raise RuntimeError("realloc first")
        use_faf = self._faf_on if do_faf is None else bool(do_faf)
        if use_faf and not self._faf_on:
            raise ValueError("upload_folded_fit() first")
        self.kernel_params['do_faf'] = np.int32(1 if use_faf else 0)
        self.kernel_params['folded_meta'] = cltypes.make_int4(int(self.folded_meta[0]), int(self.folded_meta[1]), 0, 0)
        self.kernel_params['folded_lvec2d'] = cltypes.make_float4(
            float(self.folded_lvec2d[0]), float(self.folded_lvec2d[1]),
            float(self.folded_lvec2d[2]), float(self.folded_lvec2d[3]))
        overrides = {
            'Efield': _pack_float3(self.params.efield),
            'dt': np.float32(self.params.dt),
            'nOuter': np.int32(self.params.n_outer),
            'nInner': np.int32(self.params.n_inner),
            'bMix': np.float32(self.params.bmix),
            'damp': np.float32(self.params.damp),
            'do_faf': np.int32(1 if use_faf else 0),
            'folded_meta': self.kernel_params['folded_meta'],
            'folded_lvec2d': self.kernel_params['folded_lvec2d'],
        }
        args = self.generate_kernel_args('lff_jacobi', overrides=overrides)
        # one workgroup per molecule
        global_size = (self.n_mols * self.wg,)
        local_size = (self.wg,)
        self.prg.lff_jacobi(self.queue, global_size, local_size, *args)
        self.queue.finish()

    def relax(self, n_outer=100, n_inner=8, dt=0.05, damp=0.9, bmix=0.0, do_faf=False):
        self.set_params(dt=dt, n_outer=n_outer, n_inner=n_inner, damp=damp, bmix=bmix)
        # zero velocities for clean relax
        vel = np.zeros((self.nAtomTot, 4), dtype=np.float32)
        vel[:, 3] = 1.0
        pos = np.empty((self.nAtomTot, 4), dtype=np.float32)
        self.fromGPU('apos', pos)
        self.toGPU('avel', vel)
        self.run_jacobi(do_faf=do_faf)
        return self.download_state()

    def download_state(self):
        pos = np.empty((self.nAtomTot, 4), dtype=np.float32)
        vel = np.empty((self.nAtomTot, 4), dtype=np.float32)
        self.fromGPU('apos', pos)
        self.fromGPU('avel', vel)
        self.queue.finish()
        return {'pos': pos, 'vel': vel}

    def get_positions(self):
        return self.download_state()['pos'][:, :3]
