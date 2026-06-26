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

from spammm.forcefields.MolecularDynamics import MolecularDynamics
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


def setup_rigid_folded(mol_file, fit_result, z_init=3.0, xy_init=(0.0, 0.0), quats=None, mass_trans=1.0, debug=False):
    """Create RigidBodyDynamics with folded basis from fit result.

    Args:
        mol_file: path to molecule XYZ
        fit_result: dict from fit_folded_for_molecule
        z_init: initial height above surface top in Angstrom
        xy_init: initial (x, y) position
        quats: (4,) initial quaternion, or None for identity
        mass_trans: translational mass parameter
    """
    apos_mol, reqs, enames, _, _ = load_xyz_with_REQs(mol_file)
    apos_mol = np.asarray(apos_mol, dtype=np.float32)
    masses = _guess_mass(enames)
    com0 = (apos_mol * masses[:, None]).sum(axis=0) / masses.sum()
    rel = apos_mol - com0[None, :]
    mtot, I, Iinv = compute_mass_properties(rel, masses)
    I_mean = float(np.mean(np.diag(I)))
    Iinv = Iinv * I_mean

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
