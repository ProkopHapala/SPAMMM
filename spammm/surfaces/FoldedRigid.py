"""
FoldedRigid.py — Core functions for folded-basis rigid-body molecular simulation on surfaces.

Purpose: Workflow orchestration for fitting folded basis potentials, setting up
rigid-body dynamics, running relaxations, lateral scans, and manipulation
trajectories on periodic substrates.

Key functionality:
  - fit_folded_for_molecule() — fit folded basis coefficients for molecule+substrate
  - setup_rigid_folded() — create RigidBodyDynamics from fit result
  - relax_folded() — run relaxation recording trajectory
  - lateral_scan() — scan molecule across substrate at fixed z
  - relaxed_scan() — pinned-atom manipulation scan
  - manipulation_trajectory() — lateral manipulation simulation
  - replicate_substrate() — periodic replication of substrate atoms
  - load_substrate() — load substrate from XYZ
  - nearest_substrate_distance() — distance to nearest substrate atom of given element
  - find_bonds() — find bonds by distance cutoff
  - save_xyz_trajectory() — save multi-frame XYZ with substrate context
  - random_quaternion() — random rotation quaternion

Role in SPAMMM: Core simulation workflow module. Used by tests and user-facing
scripts. Plotting functions are in surface_plots.py.
"""

import os
import numpy as np

from spammm.forcefields.SPFF_cl import SPFF_cl as MolecularDynamics
from spammm.forcefields.RigidBodyDynamics import RigidBodyDynamics, _guess_mass, compute_mass_properties
from spammm.topology.FFparams import load_xyz_with_REQs
from spammm.AtomicSystem import AtomicSystem

# =============================================================================
# Constants
# =============================================================================

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NACL_SUBSTRATE = os.path.join(_proj_root, 'data', 'substrates', 'NaCl_1x1_L3.xyz')
Z_SURF_TOP = -3.25
LATTICE_A = 4.0

MORSE_ALPHAS = np.array([1.0, 1.8, 2.7, 3.6, 5.0], dtype=np.float32)
COULOMB_ALPHAS = np.array([0.0, 0.3, 0.6, 1.0, 1.5], dtype=np.float32)
COMBINED_ALPHAS = np.array([0.3, 0.6, 1.0, 1.8, 2.7, 3.6, 5.0], dtype=np.float32)


# =============================================================================
# Geometry & math utilities
# =============================================================================

def random_quaternion(max_angle=np.pi):
    """Generate a random rotation quaternion for rotation up to max_angle radians."""
    axis = np.random.randn(3)
    axis /= max(np.linalg.norm(axis), 1e-30)
    angle = np.random.uniform(0, max_angle)
    s = np.sin(angle * 0.5)
    c = np.cos(angle * 0.5)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, c], dtype=np.float32)


def nearest_substrate_distance(atom_pos, substrate_apos, substrate_enames, target_element):
    """Compute distance from a single atom position to nearest substrate atom of target_element.

    Args:
        atom_pos: (3,) array
        substrate_apos: (N, 3) or (N, 4) array
        substrate_enames: list of N element names
        target_element: e.g. 'Na' or 'Cl'
    Returns: (distance, substrate_atom_index)
    """
    p = np.asarray(atom_pos[:3], dtype=np.float64)
    apos = np.asarray(substrate_apos[:, :3], dtype=np.float64)
    enames = np.asarray(substrate_enames)
    mask = enames == target_element
    if not np.any(mask):
        return float('inf'), -1
    d = np.linalg.norm(apos[mask] - p[None, :], axis=1)
    idx = int(np.argmin(d))
    return float(d[idx]), int(np.where(mask)[0][idx])


def find_bonds(apos, enames, Rcut=1.8):
    """Find bonds by distance cutoff."""
    na = len(apos)
    bonds = []
    for i in range(na):
        for j in range(i + 1, na):
            r = np.linalg.norm(apos[i] - apos[j])
            if r < Rcut:
                bonds.append((i, j))
    return bonds


# =============================================================================
# Substrate utilities
# =============================================================================

def replicate_substrate(sub_apos, sub_enames, lvec, x_range, y_range, z_min=-10.0):
    """Periodically replicate substrate atoms within a given XY area.

    Args:
        sub_apos: (N, 3) or (N, 4) substrate positions in unit cell
        sub_enames: list of element names
        lvec: (3, 3) or (4, 3) lattice vectors (rows = a, b, c)
        x_range: (xmin, xmax)
        y_range: (ymin, ymax)
        z_min: only include atoms with z >= z_min (filter deep layers)

    Returns: (apos_rep, enames_rep) — replicated positions and names.
    """
    a = np.asarray(lvec[0, :3], dtype=np.float64)
    b = np.asarray(lvec[1, :3], dtype=np.float64)
    apos = np.asarray(sub_apos[:, :3], dtype=np.float64)
    na = len(sub_enames)

    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cell_extent = max(abs(ax) + abs(bx), abs(ay) + abs(by))
    n_rep = int(np.ceil(max(x_range[1] - x_range[0], y_range[1] - y_range[0]) / max(cell_extent, 1e-6))) + 2

    rep_pos = []
    rep_names = []
    for ix in range(-n_rep, n_rep + 1):
        for iy in range(-n_rep, n_rep + 1):
            shift = ix * a + iy * b
            for i in range(na):
                p = apos[i] + shift
                if p[0] < x_range[0] - 1 or p[0] > x_range[1] + 1:
                    continue
                if p[1] < y_range[0] - 1 or p[1] > y_range[1] + 1:
                    continue
                if p[2] < z_min:
                    continue
                rep_pos.append(p)
                rep_names.append(sub_enames[i])
    return np.array(rep_pos, dtype=np.float32), rep_names


def load_substrate(substrate_file=NACL_SUBSTRATE):
    mol = AtomicSystem(fname=substrate_file)
    apos = np.asarray(mol.apos, dtype=np.float32)
    enames = list(mol.enames) if hasattr(mol, 'enames') else []
    if not enames:
        from spammm import atomicUtils as au
        _, _, enames, _, _ = au.load_xyz(fname=substrate_file, bReadN=True)
    qs = np.asarray(mol.qs, dtype=np.float32) if hasattr(mol, 'qs') else None
    lvec = np.asarray(mol.lvec, dtype=np.float32) if hasattr(mol, 'lvec') else None
    return apos, enames, qs, lvec


# =============================================================================
# Folded basis workflow
# =============================================================================

def fit_folded_for_molecule(mol_file, substrate_file=NACL_SUBSTRATE, z_range_rel=(1.5, 8.0), nu=4, nv=4, nPBC=(4, 4, 0), alpha_morse=1.8, custom_alphas=None):
    """Fit folded basis coefficients for a molecule on a substrate.

    Morse (pauli+london) and Coulomb are fitted independently via
    fit_folded_surface_basis with coulomb_solver='ewald2d'.
    Returns dict with total_coeffs, basis_params, atom_type_ids, basis_lvec2d.
    """
    apos_mol, reqs, enames, _, _ = load_xyz_with_REQs(mol_file)
    z_range_abs = (Z_SURF_TOP + z_range_rel[0], Z_SURF_TOP + z_range_rel[1])
    if custom_alphas is None:
        custom_alphas = COMBINED_ALPHAS
    nz = len(custom_alphas)

    md = MolecularDynamics(nloc=32, debug_build_options='-DDBG_UFF=0')
    md.init_rigid_molecule_batch(np.zeros((len(enames), 3), dtype=np.float32), reqs, nSystems=8192)
    md.set_surface(substrate_file, nPBC=nPBC, alpha_morse=alpha_morse, bMacro=True)

    params = md.fit_folded_surface_basis(
        surf_xyz=substrate_file, components=('pauli', 'london', 'coulomb'),
        coulomb_solver='ewald2d', z_range=z_range_abs,
        nu=nu, nv=nv, nz=nz, custom_alphas=custom_alphas,
        nPBC=nPBC, alpha_morse=alpha_morse, nxy=32, nz_samp=60, ewald_n_harm=6,
    )

    coeff_sets = params['coeff_sets']
    total_coeffs = coeff_sets['pauli'] + coeff_sets['london'] + coeff_sets['coulomb']

    lvec2d = params['basis_lvec2d']
    a = np.array(lvec2d[0, :3], dtype=np.float32)
    b = np.array(lvec2d[1, :3], dtype=np.float32)
    folded_lvec2d = np.array([a[0], b[0], a[1], b[1]], dtype=np.float32)

    return {
        'coeffs': total_coeffs.astype(np.float32),
        'basis_params': params['basis_params'].astype(np.float32),
        'atom_type_ids': params['atom_type_ids'].astype(np.int32),
        'folded_lvec2d': folded_lvec2d,
        'unique_REQs': params['unique_REQs'],
        'z_range': params['z_range'],
        'enames': enames,
        'reqs': reqs,
        'apos_mol': apos_mol,
    }


def save_fit(fit, fname):
    """Save a fit result dict to a small .npz file."""
    np.savez(fname,
        coeffs=fit['coeffs'],
        basis_params=fit['basis_params'],
        atom_type_ids=fit['atom_type_ids'],
        folded_lvec2d=fit['folded_lvec2d'],
        unique_REQs=fit['unique_REQs'],
        z_range=np.array(fit['z_range'], dtype=np.float32),
        enames_str=','.join(fit['enames']),
        reqs=fit['reqs'],
        apos_mol=fit['apos_mol'],
    )


def load_fit(fname):
    """Load a fit result dict from .npz file written by save_fit."""
    d = np.load(fname, allow_pickle=False)
    return {
        'coeffs': d['coeffs'],
        'basis_params': d['basis_params'],
        'atom_type_ids': d['atom_type_ids'],
        'folded_lvec2d': d['folded_lvec2d'],
        'unique_REQs': d['unique_REQs'],
        'z_range': tuple(d['z_range']),
        'enames': list(str(d['enames_str'].item()).split(',')),
        'reqs': d['reqs'],
        'apos_mol': d['apos_mol'],
    }


def setup_rigid_folded(mol_file, fit_result, z_init=3.0, xy_init=(0.0, 0.0), quats=None, mass_trans=1.0, debug=False):
    """Create RigidBodyDynamics with folded basis from fit result.

    Args:
        mol_file: path to molecule XYZ, or None to use fit_result['apos_mol'] etc.
        fit_result: dict from fit_folded_for_molecule
        z_init: initial height above surface top in Angstrom
        xy_init: initial (x, y) position
        quats: (4,) initial quaternion, or None for identity
        mass_trans: translational mass parameter
    """
    if mol_file is None:
        apos_mol = fit_result['apos_mol']
        reqs = fit_result['reqs']
        enames = fit_result['enames']
    else:
        apos_mol, reqs, enames, _, _ = load_xyz_with_REQs(mol_file)
    apos_mol = np.asarray(apos_mol, dtype=np.float32)
    masses = np.ones(len(enames), dtype=np.float32)
    com0 = (apos_mol * masses[:, None]).sum(axis=0) / masses.sum()
    rel = apos_mol - com0[None, :]
    mtot, I, Iinv = compute_mass_properties(rel, masses)
    I_mean = float(np.mean(np.diag(I)))
    mass_trans = mtot

    n_bodies = 1
    n_atoms = len(enames)
    pos4 = np.zeros((n_bodies, 4), dtype=np.float32)
    pos4[0, :3] = [xy_init[0], xy_init[1], Z_SURF_TOP + z_init]
    pos4[0, 3] = mass_trans
    quat4 = np.zeros((n_bodies, 4), dtype=np.float32)
    quat4[0, 3] = 1.0
    if quats is not None:
        quat4[0, :] = np.asarray(quats, dtype=np.float32)
        quat4[0] /= max(np.linalg.norm(quat4[0]), 1e-30)
    zero4 = np.zeros((n_bodies, 4), dtype=np.float32)
    atom_body = rel[None, :, :].astype(np.float32)

    rbd = RigidBodyDynamics(debug=debug)
    rbd.realloc(n_bodies=n_bodies, num_atoms=n_atoms)
    rbd.enames = list(enames)
    rbd.atom_REQ = reqs.copy()
    rbd.atom_masses = masses.copy()
    rbd.mass_physical = float(mtot)
    rbd.mass_trans = mass_trans
    rbd.mass_rot = mass_trans
    rbd.upload_state(
        pos4, quat4, zero4, zero4,
        mass_trans, 1.0 / mass_trans,
        np.repeat(Iinv[None, :, :], n_bodies, axis=0),
        atom_body,
        inertia=np.repeat(I[None, :, :], n_bodies, axis=0),
    )

    coeffs = fit_result['coeffs']
    kxyz = fit_result['basis_params']
    atype = fit_result['atom_type_ids']
    lvec2d = fit_result['folded_lvec2d']
    ntypes, nbasis = coeffs.shape
    folded_meta = np.array([nbasis, ntypes, 0, 0], dtype=np.int32)
    rbd.init_folded(coeffs, kxyz, atype, lvec2d, folded_meta=folded_meta)
    return rbd


def relax_folded(rbd, n_steps=2000, dt=0.01, lin_damp=0.95, ang_damp=0.90, record_interval=100):
    """Run relaxation, recording trajectory.

    Returns dict with 'energies', 'forces', 'torques', 'positions', 'quaternions', 'atom_positions' lists.
    """
    energies = []
    forces = []
    torques = []
    positions = []
    quaternions = []
    atom_positions_list = []

    n_record = max(1, n_steps // record_interval) if record_interval > 0 else 0
    for i in range(0, n_steps, record_interval):
        steps = min(record_interval, n_steps - i)
        rbd.run_folded(steps, dt, lin_damp=lin_damp, ang_damp=ang_damp)
        out = rbd.download_outputs()
        atom_pos = out['atom_positions'][0]  # (natoms, 4)
        E = float(atom_pos[:, 3].sum())
        f = out['body_force'][0]
        tq = out['body_torque'][0]
        energies.append(E)
        forces.append(float(np.linalg.norm(f[:3])))
        torques.append(float(np.linalg.norm(tq[:3])))
        positions.append(out['pos'][0].copy())
        quaternions.append(out['quats'][0].copy())
        atom_positions_list.append(atom_pos[:, :3].copy())

    return {
        'energies': np.array(energies),
        'forces': np.array(forces),
        'torques': np.array(torques),
        'positions': np.array(positions),
        'quaternions': np.array(quaternions),
        'atom_positions': atom_positions_list,
    }


def relax_folded_diag(rbd, n_steps=5000, dt=0.02, lin_damp=0.95, ang_damp=0.90, record_interval=10):
    """Run relaxation with comprehensive diagnostics for debugging rotation dynamics.

    Records thinned trajectory of: energy, |force|, |torque|, COM, quaternion,
    angular velocity (vrot), linear velocity (vpos), per-atom forces, tilt angle,
    rotation angle per step, and atom positions.

    Returns dict with arrays of shape (n_record, ...).
    """
    from spammm.forcefields.RigidBodyDynamics import _quat_to_matrix_np
    n_record = max(1, n_steps // record_interval)
    recs = {
        'step': np.zeros(n_record, dtype=np.int32),
        'E': np.zeros(n_record, dtype=np.float64),
        'Fmag': np.zeros(n_record, dtype=np.float64),
        'Tmag': np.zeros(n_record, dtype=np.float64),
        'com': np.zeros((n_record, 3), dtype=np.float64),
        'quat': np.zeros((n_record, 4), dtype=np.float64),
        'vpos': np.zeros((n_record, 3), dtype=np.float64),
        'vrot': np.zeros((n_record, 3), dtype=np.float64),
        'vrot_mag': np.zeros(n_record, dtype=np.float64),
        'rot_per_step': np.zeros(n_record, dtype=np.float64),
        'tilt': np.zeros(n_record, dtype=np.float64),
        'atom_pos': np.zeros((n_record, rbd.num_atoms, 3), dtype=np.float64),
        'atom_force': np.zeros((n_record, rbd.num_atoms, 4), dtype=np.float64),
        'body_force': np.zeros((n_record, 4), dtype=np.float64),
        'body_torque': np.zeros((n_record, 4), dtype=np.float64),
    }
    body_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    surf_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    irec = 0
    for i in range(0, n_steps, record_interval):
        steps = min(record_interval, n_steps - i)
        rbd.run_folded(steps, dt, lin_damp=lin_damp, ang_damp=ang_damp)
        out = rbd.download_outputs()
        atom_pos = out['atom_positions'][0]
        atom_f = out['atom_force'][0]
        E = float(atom_pos[:, 3].sum())
        f = out['body_force'][0][:3]
        tq = out['body_torque'][0][:3]
        pos = out['pos'][0][:3]
        quat = out['quats'][0]
        vpos = out['lin_mom'][0][:3]
        vrot = out['ang_mom'][0][:3]
        R = _quat_to_matrix_np(quat)
        body_z_world = R @ body_axis
        tilt = float(np.arccos(np.clip(np.dot(body_z_world, surf_normal), -1, 1)))
        rot_angle = float(np.linalg.norm(vrot) * dt * steps)
        recs['step'][irec] = i + steps
        recs['E'][irec] = E
        recs['Fmag'][irec] = float(np.linalg.norm(f))
        recs['Tmag'][irec] = float(np.linalg.norm(tq))
        recs['com'][irec] = pos
        recs['quat'][irec] = quat
        recs['vpos'][irec] = vpos
        recs['vrot'][irec] = vrot
        recs['vrot_mag'][irec] = float(np.linalg.norm(vrot))
        recs['rot_per_step'][irec] = rot_angle
        recs['tilt'][irec] = tilt
        recs['atom_pos'][irec] = atom_pos[:, :3]
        recs['atom_force'][irec] = atom_f
        recs['body_force'][irec] = out['body_force'][0]
        recs['body_torque'][irec] = out['body_torque'][0]
        irec += 1
    return recs


def plot_relax_diag(recs, title="Relaxation Diagnostics", save_path=None):
    """Plot diagnostic relaxation data as a multi-panel summary.

    Panels: Energy, |F| & |T|, COM z, tilt angle, |vrot| & rot/step, quaternion components.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps = recs['step']
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14)

    ax = axes[0, 0]
    ax.plot(steps, recs['E'], 'b-', lw=0.8)
    ax.set_ylabel('Energy [eV]')
    ax.set_title('Total Energy')

    ax = axes[0, 1]
    ax.plot(steps, recs['Fmag'], 'r-', lw=0.8, label='|F|')
    ax.plot(steps, recs['Tmag'], 'g-', lw=0.8, label='|T|')
    ax.set_ylabel('Force / Torque')
    ax.set_title('|Force| & |Torque|')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[1, 0]
    ax.plot(steps, recs['com'][:, 2], 'b-', lw=0.8, label='z')
    ax.plot(steps, recs['com'][:, 0], 'r-', lw=0.5, label='x')
    ax.plot(steps, recs['com'][:, 1], 'g-', lw=0.5, label='y')
    ax.set_ylabel('Position [Å]')
    ax.set_title('COM Position')
    ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, np.degrees(recs['tilt']), 'm-', lw=0.8)
    ax.set_ylabel('Tilt [deg]')
    ax.set_title('Tilt angle (body z vs surface normal)')

    ax = axes[2, 0]
    ax.plot(steps, recs['vrot_mag'], 'r-', lw=0.8, label='|vrot|')
    ax.plot(steps, recs['rot_per_step'], 'b-', lw=0.5, label='rot/record')
    ax.set_ylabel('Angular velocity / rotation')
    ax.set_title('Angular velocity magnitude')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[2, 1]
    for j, c in enumerate('xyzw'):
        ax.plot(steps, recs['quat'][:, j], lw=0.5, label=f'q{j}({c})')
    ax.set_ylabel('Quaternion')
    ax.set_title('Quaternion components')
    ax.legend()
    ax.set_xlabel('Step')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_relax_diag] Saved to {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# Folded potential evaluation (for debugging / visualization)
# =============================================================================

def eval_folded_potential(fit_result, atom_type_idx, xyz):
    """Evaluate folded basis potential for a single atom type at arbitrary positions.

    Args:
        fit_result: dict from fit_folded_for_molecule / load_fit
        atom_type_idx: index into coeffs rows (0 = first unique REQ type)
        xyz: (N, 3) array of world coordinates

    Returns: (N,) array of potential energy values
    """
    coeffs = np.asarray(fit_result['coeffs'], dtype=np.float64)
    kxyz = np.asarray(fit_result['basis_params'], dtype=np.float64)
    lvec2d = np.asarray(fit_result['folded_lvec2d'], dtype=np.float64)
    ax, bx, ay, by = lvec2d
    det = ax * by - bx * ay
    invLvec = np.array([by / det, -bx / det, -ay / det, ax / det])
    xyz = np.asarray(xyz, dtype=np.float64)
    u = invLvec[0] * xyz[:, 0] + invLvec[1] * xyz[:, 1]
    v = invLvec[2] * xyz[:, 0] + invLvec[3] * xyz[:, 1]
    u = u - np.floor(u)
    v = v - np.floor(v)
    z = xyz[:, 2]
    ku = kxyz[None, :, 0]
    kv = kxyz[None, :, 1]
    az = kxyz[None, :, 2]
    z0 = kxyz[None, :, 3]
    bx_ = np.cos(2.0 * np.pi * ku * u[:, None])
    by_ = np.cos(2.0 * np.pi * kv * v[:, None])
    bz_ = np.exp(-az * np.maximum(0.0, z[:, None] - z0))
    basis = bx_ * by_ * bz_
    c = coeffs[atom_type_idx, :basis.shape[1]]
    return (basis * c[None, :]).sum(axis=1)


def eval_folded_potential_slice(fit_result, atom_type_idx, plane='xy', fixed_val=0.0, extent=(-8, 8, -8, 8), n=64):
    """Evaluate folded basis potential for a single atom type on a 2D grid.

    Args:
        fit_result: dict from fit_folded_for_molecule / load_fit
        atom_type_idx: index into coeffs rows
        plane: 'xy' (top view), 'xz' or 'yz' (side views)
        fixed_val: value of the fixed coordinate (z for 'xy', y for 'xz', x for 'yz')
        extent: (min1, max1, min2, max2) for the two varying coordinates
        n: grid resolution (n x n)

    Returns: (c1, c2, E) where c1, c2 are 1D coordinate arrays and E is (n, n) potential matrix
    """
    a = np.linspace(extent[0], extent[1], n)
    b = np.linspace(extent[2], extent[3], n)
    A, B = np.meshgrid(a, b, indexing='ij')
    if plane == 'xy':
        xyz = np.stack([A.ravel(), B.ravel(), np.full(A.size, fixed_val)], axis=1)
    elif plane == 'xz':
        xyz = np.stack([A.ravel(), np.full(A.size, fixed_val), B.ravel()], axis=1)
    elif plane == 'yz':
        xyz = np.stack([np.full(A.size, fixed_val), A.ravel(), B.ravel()], axis=1)
    else:
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")
    E = eval_folded_potential(fit_result, atom_type_idx, xyz).reshape(n, n)
    return a, b, E


# =============================================================================
# Scanning & manipulation
# =============================================================================

def lateral_scan(rbd, xs, ys, z, n_relax=50, dt=0.01):
    """Scan molecule across substrate at fixed z, measuring force at each (x,y).

    At each position, runs a short relaxation (n_relax steps) with strong damping
    to let forces settle, then records force and energy.

    Returns dict with 'X', 'Y', 'Fz', 'Fx', 'Fy', 'E', 'atom_positions' arrays.
    """
    from spammm.forcefields.RigidBodyDynamics import _ensure_float4
    nx, ny = len(xs), len(ys)
    Fz = np.zeros((nx, ny), dtype=np.float32)
    Fx = np.zeros((nx, ny), dtype=np.float32)
    Fy = np.zeros((nx, ny), dtype=np.float32)
    E_grid = np.zeros((nx, ny), dtype=np.float32)
    atom_pos_grid = []

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            pos = np.array([[x, y, z, rbd.mass_trans]], dtype=np.float32)
            quat = np.array([[0, 0, 0, 1]], dtype=np.float32)
            rbd.reset_pose(pos, quat)
            rbd.run_folded(n_relax, dt, lin_damp=0.98, ang_damp=0.95)
            out = rbd.download_outputs()
            f = out['body_force'][0]
            atoms = out['atom_positions'][0]
            E = float(atoms[:, 3].sum())
            Fx[ix, iy] = f[0]
            Fy[ix, iy] = f[1]
            Fz[ix, iy] = f[2]
            E_grid[ix, iy] = E
            atom_pos_grid.append(atoms[:, :3].copy())

    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return {
        'X': X, 'Y': Y,
        'Fx': Fx, 'Fy': Fy, 'Fz': Fz, 'E': E_grid,
        'atom_positions': atom_pos_grid,
    }


def relaxed_scan(rbd, pin_atom_idx, path, k_spring=5.0, n_relax=200, dt=0.005,
                 lin_damp=0.99, ang_damp=0.95, record_interval=10):
    """Relaxed scan: pin one atom with a spring and move it along a path.

    At each path point, the pinned atom is held by a harmonic spring to the target
    position. The rest of the molecule relaxes on the surface potential. This simulates
    AFM manipulation: dragging a molecule by one atom across the substrate.

    Args:
        rbd: RigidBodyDynamics instance (with folded basis initialized)
        pin_atom_idx: index of the atom to pin with spring
        path: (N, 3) array of target (x,y,z) positions for the pinned atom
        k_spring: spring constant in eV/Å² (higher = stiffer constraint)
        n_relax: relaxation steps per path point
        dt: timestep for relaxation
        lin_damp: linear damping factor per step
        ang_damp: angular damping factor per step
        record_interval: record state every N relaxation steps within each path point

    Returns:
        dict with 'positions', 'quaternions', 'forces', 'torques', 'atom_positions',
        'path', 'pin_forces' (spring force on pinned atom at each path point)
    """
    n_path = len(path)
    positions = []
    quaternions = []
    forces = []
    torques = []
    atom_positions_list = []
    pin_forces = []

    for i in range(n_path):
        target = np.asarray(path[i], dtype=np.float32)
        anchors = np.zeros((rbd.total_atoms, 4), dtype=np.float32)
        anchors[:, 3] = -1.0
        anchors[pin_atom_idx, :3] = target
        anchors[pin_atom_idx, 3] = k_spring
        rbd.update_anchors(anchors)

        for j in range(0, n_relax, record_interval):
            nrun = min(record_interval, n_relax - j)
            rbd.run_folded(nrun, dt, lin_damp=lin_damp, ang_damp=ang_damp)
            out = rbd.download_outputs()
            positions.append(out['pos'][0].copy())
            quaternions.append(out['quats'][0].copy())
            forces.append(out['body_force'][0][:3].copy())
            torques.append(out['body_torque'][0][:3].copy())
            atom_positions_list.append(out['atom_positions'][0][:, :3].copy())

        out = rbd.download_outputs()
        atom_f = out['atom_force'][0]
        pin_forces.append(atom_f[pin_atom_idx][:3].copy())

    n_rec = len(positions)
    return {
        'positions': np.array(positions),
        'quaternions': np.array(quaternions),
        'forces': np.array(forces),
        'torques': np.array(torques),
        'atom_positions': atom_positions_list,
        'path': np.asarray(path, dtype=np.float32),
        'pin_forces': np.array(pin_forces),
        'n_path': n_path,
        'n_relax': n_relax,
        'record_interval': record_interval,
    }


def manipulation_trajectory(rbd, x0, y0, z, dx, n_steps=50, dt=0.02, n_relax_per_step=20):
    """Simulate manipulation: move molecule laterally in small increments.

    At each step, shift target position by dx/n_steps, relax briefly, record state.
    Returns dict with 'positions', 'forces', 'atom_positions', 'path'.
    """
    positions = []
    forces = []
    atom_positions_list = []
    path = np.zeros((n_steps, 3), dtype=np.float32)

    for i in range(n_steps):
        frac = (i + 1) / n_steps
        x = x0 + dx[0] * frac
        y = y0 + dx[1] * frac
        pos = np.array([[x, y, z, rbd.mass_trans]], dtype=np.float32)
        quat = np.array([[0, 0, 0, 1]], dtype=np.float32)
        rbd.reset_pose(pos, quat)
        rbd.run_folded(n_relax_per_step, dt, lin_damp=0.98, ang_damp=0.95)
        out = rbd.download_outputs()
        f = out['body_force'][0]
        atoms = out['atom_positions'][0]
        positions.append(out['pos'][0].copy())
        forces.append(f[:3].copy())
        atom_positions_list.append(atoms[:, :3].copy())
        path[i] = [x, y, z]

    return {
        'positions': np.array(positions),
        'forces': np.array(forces),
        'atom_positions': atom_positions_list,
        'path': path,
    }


# =============================================================================
# I/O
# =============================================================================

def save_xyz_trajectory(filename, mol_enames, mol_positions_list, sub_apos=None, sub_enames=None, comments=None):
    """Save trajectory as multi-frame XYZ including substrate atoms for context."""
    na_mol = len(mol_enames)
    na_sub = len(sub_enames) if sub_apos is not None else 0
    na_total = na_mol + na_sub
    with open(filename, 'w') as f:
        for idx, mol_pos in enumerate(mol_positions_list):
            comment = comments[idx] if comments else f'frame {idx}'
            f.write(f'{na_total}\n{comment}\n')
            if sub_apos is not None:
                for e, p in zip(sub_enames, sub_apos):
                    f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
            for e, p in zip(mol_enames, mol_pos):
                f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
