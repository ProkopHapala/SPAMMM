import os, sys, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from spammm.forcefields.MolecularDynamics import MolecularDynamics
from spammm.forcefields.RigidBodyDynamics import RigidBodyDynamics, _guess_mass, compute_mass_properties
from spammm.topology.FFparams import load_xyz_with_REQs
from spammm.AtomicSystem import AtomicSystem

NACL_SUBSTRATE = os.path.join(_proj_root, 'data', 'substrates', 'NaCl_1x1_L3.xyz')
Z_SURF_TOP = -3.25
LATTICE_A = 4.0

MORSE_ALPHAS = np.array([1.0, 1.8, 2.7, 3.6, 5.0], dtype=np.float32)
COULOMB_ALPHAS = np.array([0.0, 0.3, 0.6, 1.0, 1.5], dtype=np.float32)
COMBINED_ALPHAS = np.array([0.3, 0.6, 1.0, 1.8, 2.7, 3.6, 5.0], dtype=np.float32)

ELEMENT_COLORS = {'H': 'lightgray', 'C': 'black', 'O': 'red', 'N': 'blue', 'Na': 'goldenrod', 'Cl': 'green', 'S': 'yellow'}
ELEMENT_SIZES = {'H': 50, 'C': 120, 'O': 140, 'N': 130, 'Na': 180, 'Cl': 180, 'S': 150}


def random_quaternion(max_angle=np.pi):
    """Generate a random rotation quaternion for rotation up to max_angle radians."""
    axis = np.random.randn(3)
    axis /= max(np.linalg.norm(axis), 1e-30)
    angle = np.random.uniform(0, max_angle)
    s = np.sin(angle * 0.5)
    c = np.cos(angle * 0.5)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, c], dtype=np.float32)


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

    # Determine replication range
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
    # Normalize inertia to match mass_trans so angular and translational timescales are comparable
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


def plot_relaxation(traj, save_dir, name='relax'):
    """Plot energy, force, torque vs step."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    steps = np.arange(len(traj['energies']))
    axes[0].plot(steps, traj['energies'], 'b-')
    axes[0].set_ylabel('Energy [eV]')
    axes[0].set_title(f'{name}: Relaxation')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, traj['forces'], 'r-')
    axes[1].set_ylabel('|Force| [eV/Å]')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, traj['torques'], 'g-')
    axes[2].set_ylabel('|Torque| [eV]')
    axes[2].set_xlabel('Step')
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name}_relaxation.png'), dpi=150)
    plt.close(fig)


def plot_molecule_substrate_xy(ax, mol_apos, mol_enames, sub_apos, sub_enames, title='', bonds=None, highlight_element=None, highlight_color='magenta'):
    """Top-view (XY) plot of molecule on substrate.

    Args:
        ax: matplotlib Axes
        mol_apos: (natoms, 3) molecule positions
        mol_enames: list of element names
        sub_apos: (N, 3) or (N, 4) substrate positions
        sub_enames: list of substrate element names
        title: plot title
        bonds: list of (i,j) tuples for molecule bonds
        highlight_element: element name to highlight in molecule (e.g. 'O')
    """
    sub_apos = np.asarray(sub_apos[:, :3], dtype=np.float64)
    for i, (e, p) in enumerate(zip(sub_enames, sub_apos)):
        c = ELEMENT_COLORS.get(e, 'gray')
        s = ELEMENT_SIZES.get(e, 100)
        ax.scatter(p[0], p[1], c=c, s=s, zorder=2, edgecolors='black', linewidths=0.5)
        ax.text(p[0], p[1], e, fontsize=6, ha='center', va='center', zorder=4)
    if bonds:
        for i, j in bonds:
            ax.plot([mol_apos[i, 0], mol_apos[j, 0]], [mol_apos[i, 1], mol_apos[j, 1]], 'k-', linewidth=1.5, zorder=3)
    for i, (e, p) in enumerate(zip(mol_enames, mol_apos)):
        c = ELEMENT_COLORS.get(e, 'purple')
        s = ELEMENT_SIZES.get(e, 100) * 0.7
        edge = highlight_color if (highlight_element and e == highlight_element) else 'black'
        lw = 2.0 if (highlight_element and e == highlight_element) else 0.5
        ax.scatter(p[0], p[1], c=c, s=s, zorder=5, edgecolors=edge, linewidths=lw)
    ax.set_aspect('equal')
    ax.set_xlabel('X [Å]')
    ax.set_ylabel('Y [Å]')
    ax.set_title(title)


def plot_molecule_substrate_xz(ax, mol_apos, mol_enames, sub_apos, sub_enames, title='', bonds=None, highlight_element=None, highlight_color='magenta'):
    """Side-view (XZ) plot of molecule on substrate."""
    sub_apos = np.asarray(sub_apos[:, :3], dtype=np.float64)
    for i, (e, p) in enumerate(zip(sub_enames, sub_apos)):
        c = ELEMENT_COLORS.get(e, 'gray')
        s = ELEMENT_SIZES.get(e, 100)
        ax.scatter(p[0], p[2], c=c, s=s, zorder=2, edgecolors='black', linewidths=0.5)
        ax.text(p[0], p[2], e, fontsize=6, ha='center', va='center', zorder=4)
    if bonds:
        for i, j in bonds:
            ax.plot([mol_apos[i, 0], mol_apos[j, 0]], [mol_apos[i, 2], mol_apos[j, 2]], 'k-', linewidth=1.5, zorder=3)
    for i, (e, p) in enumerate(zip(mol_enames, mol_apos)):
        c = ELEMENT_COLORS.get(e, 'purple')
        s = ELEMENT_SIZES.get(e, 100) * 0.7
        edge = highlight_color if (highlight_element and e == highlight_element) else 'black'
        lw = 2.0 if (highlight_element and e == highlight_element) else 0.5
        ax.scatter(p[0], p[2], c=c, s=s, zorder=5, edgecolors=edge, linewidths=lw)
    ax.set_aspect('equal')
    ax.set_xlabel('X [Å]')
    ax.set_ylabel('Z [Å]')
    ax.set_title(title)


def plot_relax_overview(traj, mol_enames, sub_apos, sub_enames, save_dir, name, bonds=None, highlight_element='O', target_element='Na'):
    """Create comprehensive visualization: relaxation curves + XY/XZ views (initial & final).

    Also annotates distances from highlight_element atoms to nearest target_element substrate atom.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 14))

    # Row 1: Relaxation curves
    ax_e = fig.add_subplot(3, 3, 1)
    ax_f = fig.add_subplot(3, 3, 4)
    ax_t = fig.add_subplot(3, 3, 7)
    steps = np.arange(len(traj['energies']))
    ax_e.plot(steps, traj['energies'], 'b-')
    ax_e.set_ylabel('Energy [eV]'); ax_e.set_title(f'{name}: Energy'); ax_e.grid(True, alpha=0.3)
    ax_f.plot(steps, traj['forces'], 'r-')
    ax_f.set_ylabel('|Force| [eV/Å]'); ax_f.set_title('Force'); ax_f.grid(True, alpha=0.3)
    ax_t.plot(steps, traj['torques'], 'g-')
    ax_t.set_ylabel('|Torque| [eV]'); ax_t.set_xlabel('Step'); ax_t.set_title('Torque'); ax_t.grid(True, alpha=0.3)

    # Row 2: XY views (initial & final)
    ax_xy0 = fig.add_subplot(3, 3, 2)
    ax_xy1 = fig.add_subplot(3, 3, 3)
    mol0 = traj['atom_positions'][0]
    mol1 = traj['atom_positions'][-1]
    plot_molecule_substrate_xy(ax_xy0, mol0, mol_enames, sub_apos, sub_enames, title=f'{name} XY (initial)', bonds=bonds, highlight_element=highlight_element)
    plot_molecule_substrate_xy(ax_xy1, mol1, mol_enames, sub_apos, sub_enames, title=f'{name} XY (final)', bonds=bonds, highlight_element=highlight_element)

    # Row 3: XZ views (initial & final)
    ax_xz0 = fig.add_subplot(3, 3, 5)
    ax_xz1 = fig.add_subplot(3, 3, 6)
    plot_molecule_substrate_xz(ax_xz0, mol0, mol_enames, sub_apos, sub_enames, title=f'{name} XZ (initial)', bonds=bonds, highlight_element=highlight_element)
    plot_molecule_substrate_xz(ax_xz1, mol1, mol_enames, sub_apos, sub_enames, title=f'{name} XZ (final)', bonds=bonds, highlight_element=highlight_element)

    # Distance annotations
    if highlight_element and target_element:
        for label, mol_pos, ax in [('initial', mol0, ax_xy0), ('final', mol1, ax_xy1)]:
            for ia, e in enumerate(mol_enames):
                if e == highlight_element:
                    d, idx = nearest_substrate_distance(mol_pos[ia], sub_apos, sub_enames, target_element)
                    ax.annotate(f'd({e}-{target_element})={d:.2f}Å', xy=(mol_pos[ia, 0], mol_pos[ia, 1]),
                                fontsize=7, color='red', ha='center', va='bottom',
                                arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name}_overview.png'), dpi=150)
    plt.close(fig)

    # Also save separate high-res XY and XZ final views
    for proj, plot_fn, suffix in [('xy', plot_molecule_substrate_xy, 'XY'), ('xz', plot_molecule_substrate_xz, 'XZ')]:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        plot_fn(ax2, mol1, mol_enames, sub_apos, sub_enames, title=f'{name} {suffix} (final)', bonds=bonds, highlight_element=highlight_element)
        if highlight_element and target_element:
            for ia, e in enumerate(mol_enames):
                if e == highlight_element:
                    d, idx = nearest_substrate_distance(mol1[ia], sub_apos, sub_enames, target_element)
                    ax2.annotate(f'd({e}-{target_element})={d:.2f}Å', xy=(mol1[ia, 0] if proj == 'xy' else mol1[ia, 0], mol1[ia, 1] if proj == 'xy' else mol1[ia, 2]),
                                 fontsize=8, color='red', ha='center', va='bottom',
                                 arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_dir, f'{name}_{suffix}_final.png'), dpi=150)
        plt.close(fig2)


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


def plot_force_map(scan_result, save_dir, name='scan'):
    """Plot Fz and |F| lateral force maps."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    X, Y = scan_result['X'], scan_result['Y']
    Fz = scan_result['Fz']
    Fmag = np.sqrt(scan_result['Fx']**2 + scan_result['Fy']**2 + Fz**2)
    E = scan_result['E']

    for ax, data, title, cmap in [
        (axes[0], Fz, 'Fz [eV/Å]', 'RdBu_r'),
        (axes[1], Fmag, '|F| [eV/Å]', 'hot'),
        (axes[2], E, 'Energy [eV]', 'viridis'),
    ]:
        im = ax.pcolormesh(X, Y, data, shading='auto', cmap=cmap)
        ax.set_aspect('equal')
        ax.set_xlabel('X [Å]')
        ax.set_ylabel('Y [Å]')
        ax.set_title(f'{name}: {title}')
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name}_force_map.png'), dpi=150)
    plt.close(fig)


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
        # Set anchor on pinned atom
        anchors = np.zeros((rbd.total_atoms, 4), dtype=np.float32)
        anchors[:, 3] = -1.0
        anchors[pin_atom_idx, :3] = target
        anchors[pin_atom_idx, 3] = k_spring
        rbd.update_anchors(anchors)

        # Relax at this anchor position
        for j in range(0, n_relax, record_interval):
            nrun = min(record_interval, n_relax - j)
            rbd.run_folded(nrun, dt, lin_damp=lin_damp, ang_damp=ang_damp)
            out = rbd.download_outputs()
            positions.append(out['pos'][0].copy())
            quaternions.append(out['quats'][0].copy())
            forces.append(out['body_force'][0][:3].copy())
            torques.append(out['body_torque'][0][:3].copy())
            atom_positions_list.append(out['atom_positions'][0][:, :3].copy())

        # Record spring force on pinned atom (last state)
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


def plot_manipulation(traj, mol_enames, sub_apos, sub_enames, save_dir, name='manip', bonds=None, highlight_element='O', target_element='Na'):
    """Visualize manipulation trajectory with multi-snapshot XY and XZ views."""
    os.makedirs(save_dir, exist_ok=True)
    n_snap = len(traj['atom_positions'])
    n_show = min(6, n_snap)
    indices = np.linspace(0, n_snap - 1, n_show, dtype=int)

    # XY multi-snapshot
    fig_xy, axes_xy = plt.subplots(2, 3, figsize=(18, 12))
    for k, idx in enumerate(indices):
        ax = axes_xy[k // 3][k % 3]
        mol_pos = traj['atom_positions'][idx]
        plot_molecule_substrate_xy(ax, mol_pos, mol_enames, sub_apos, sub_enames,
                                   title=f'{name} XY step {idx}', bonds=bonds,
                                   highlight_element=highlight_element)
        if highlight_element and target_element:
            for ia, e in enumerate(mol_enames):
                if e == highlight_element:
                    d, _ = nearest_substrate_distance(mol_pos[ia], sub_apos, sub_enames, target_element)
                    ax.annotate(f'd={d:.2f}', xy=(mol_pos[ia, 0], mol_pos[ia, 1]),
                                fontsize=7, color='red', ha='center', va='bottom')
    fig_xy.tight_layout()
    fig_xy.savefig(os.path.join(save_dir, f'{name}_xy_snapshots.png'), dpi=150)
    plt.close(fig_xy)

    # XZ multi-snapshot
    fig_xz, axes_xz = plt.subplots(2, 3, figsize=(18, 12))
    for k, idx in enumerate(indices):
        ax = axes_xz[k // 3][k % 3]
        mol_pos = traj['atom_positions'][idx]
        plot_molecule_substrate_xz(ax, mol_pos, mol_enames, sub_apos, sub_enames,
                                   title=f'{name} XZ step {idx}', bonds=bonds,
                                   highlight_element=highlight_element)
    fig_xz.tight_layout()
    fig_xz.savefig(os.path.join(save_dir, f'{name}_xz_snapshots.png'), dpi=150)
    plt.close(fig_xz)

    # Force along path
    fig_f, ax_f = plt.subplots(figsize=(10, 5))
    steps = np.arange(n_snap)
    Fmag = np.linalg.norm(traj['forces'], axis=1)
    ax_f.plot(steps, Fmag, 'b-', label='|F|')
    ax_f.set_xlabel('Step')
    ax_f.set_ylabel('|Force| [eV/Å]')
    ax_f.set_title(f'{name}: Force along manipulation path')
    ax_f.grid(True, alpha=0.3)
    ax_f.legend()
    fig_f.tight_layout()
    fig_f.savefig(os.path.join(save_dir, f'{name}_force_path.png'), dpi=150)
    plt.close(fig_f)


def plot_relaxed_scan(traj, mol_enames, sub_apos, sub_enames, save_dir, name='rscan',
                      bonds=None, pin_atom_idx=None, highlight_element='O', target_element='Na'):
    """Visualize relaxed scan: snapshots, force/torque curves, pin force."""
    os.makedirs(save_dir, exist_ok=True)
    n_rec = len(traj['atom_positions'])
    n_path = traj['n_path']
    rec_per_path = n_rec // n_path

    # Select snapshots at end of each path point relaxation
    snap_indices = [min((i + 1) * rec_per_path - 1, n_rec - 1) for i in range(n_path)]
    n_show = min(6, n_path)
    show_indices = np.linspace(0, n_path - 1, n_show, dtype=int)

    # XY snapshots
    fig_xy, axes_xy = plt.subplots(2, 3, figsize=(18, 12))
    for k, pi in enumerate(show_indices):
        ax = axes_xy[k // 3][k % 3]
        si = snap_indices[pi]
        mol_pos = traj['atom_positions'][si]
        path_pt = traj['path'][pi]
        plot_molecule_substrate_xy(ax, mol_pos, mol_enames, sub_apos, sub_enames,
                                   title=f'{name} XY path {pi}', bonds=bonds,
                                   highlight_element=highlight_element)
        # Mark pinned atom target
        if pin_atom_idx is not None:
            ax.plot(path_pt[0], path_pt[1], 'rx', markersize=12, markeredgewidth=2)
            ax.annotate(f'pin', xy=(path_pt[0], path_pt[1]), fontsize=7, color='red', ha='left')
    fig_xy.tight_layout()
    fig_xy.savefig(os.path.join(save_dir, f'{name}_xy_snapshots.png'), dpi=150)
    plt.close(fig_xy)

    # XZ snapshots
    fig_xz, axes_xz = plt.subplots(2, 3, figsize=(18, 12))
    for k, pi in enumerate(show_indices):
        ax = axes_xz[k // 3][k % 3]
        si = snap_indices[pi]
        mol_pos = traj['atom_positions'][si]
        path_pt = traj['path'][pi]
        plot_molecule_substrate_xz(ax, mol_pos, mol_enames, sub_apos, sub_enames,
                                   title=f'{name} XZ path {pi}', bonds=bonds,
                                   highlight_element=highlight_element)
        if pin_atom_idx is not None:
            ax.plot(path_pt[0], path_pt[2], 'rx', markersize=12, markeredgewidth=2)
    fig_xz.tight_layout()
    fig_xz.savefig(os.path.join(save_dir, f'{name}_xz_snapshots.png'), dpi=150)
    plt.close(fig_xz)

    # Force and torque along path (at end of each path point relaxation)
    fig_f, (ax_f, ax_t, ax_pf) = plt.subplots(3, 1, figsize=(10, 12))
    path_steps = np.arange(n_path)
    Fmag = np.linalg.norm(traj['forces'][snap_indices], axis=1)
    Tmag = np.linalg.norm(traj['torques'][snap_indices], axis=1)
    Pmag = np.linalg.norm(traj['pin_forces'], axis=1)
    ax_f.plot(path_steps, Fmag, 'b-o', label='|F|')
    ax_f.set_ylabel('|Force| [eV/Å]')
    ax_f.set_title(f'{name}: Force/torque along relaxed scan')
    ax_f.grid(True, alpha=0.3)
    ax_f.legend()
    ax_t.plot(path_steps, Tmag, 'r-o', label='|τ|')
    ax_t.set_ylabel('|Torque| [eV]')
    ax_t.grid(True, alpha=0.3)
    ax_t.legend()
    ax_pf.plot(path_steps, Pmag, 'g-o', label='|F_pin|')
    ax_pf.set_ylabel('|Pin force| [eV/Å]')
    ax_pf.set_xlabel('Path step')
    ax_pf.grid(True, alpha=0.3)
    ax_pf.legend()
    fig_f.tight_layout()
    fig_f.savefig(os.path.join(save_dir, f'{name}_forces.png'), dpi=150)
    plt.close(fig_f)


def plot_manipulation_trail(traj, mol_enames, sub_apos, sub_enames, save_dir, name='rscan',
                            pin_atom_idx=None, opp_atom_idx=None,
                            highlight_element='O', target_element='Na'):
    """Plot manipulation trail: pin atom and opposite atom connected by thin lines.

    Instead of full molecule snapshots, shows the trail of two key atoms:
    - Pinned atom (red dots) — the one being dragged
    - Opposite atom (blue dots) — the far end of the molecule
    - Thin alpha-blended line connecting them for each snapshot
    This reveals how the molecule tilts and follows the tip.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_rec = len(traj['atom_positions'])
    n_path = traj['n_path']
    rec_per_path = n_rec // n_path
    snap_indices = [min((i + 1) * rec_per_path - 1, n_rec - 1) for i in range(n_path)]

    # Extract pin and opposite atom positions at each path point
    pin_pos = np.array([traj['atom_positions'][si][pin_atom_idx] for si in snap_indices])
    opp_pos = np.array([traj['atom_positions'][si][opp_atom_idx] for si in snap_indices])
    path_pts = traj['path']

    # Color gradient along path (dark→bright)
    colors = plt.cm.viridis(np.linspace(0, 1, n_path))

    # --- XY trail ---
    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 7))

    # Substrate atoms (1st layer only)
    sub = np.asarray(sub_apos[:, :3], dtype=np.float64)
    sub_z = sub[:, 2]
    z_top = sub_z.mean()
    layer_mask = np.abs(sub_z - z_top) < 0.5
    sub1 = sub[layer_mask]
    sub1_names = [sub_enames[i] for i in range(len(sub_enames)) if layer_mask[i]]

    for ax, proj, xlabel, ylabel, title_proj in [
        (ax_xy, [0, 1], 'x (Å)', 'y (Å)', 'XY'),
        (ax_xz, [0, 2], 'x (Å)', 'z (Å)', 'XZ'),
    ]:
        # Substrate
        for e, p in zip(sub1_names, sub1):
            c = 'blue' if e in ['Na', 'K', 'Ca', 'Mg'] else 'green'
            ax.plot(p[proj[0]], p[proj[1]], 'o', color=c, markersize=4, alpha=0.3)

        # Pin target path (dashed line)
        ax.plot(path_pts[:, proj[0]], path_pts[:, proj[1]], 'k--', alpha=0.3, linewidth=1)

        # Trail: thin alpha-blended lines connecting pin→opp for each snapshot
        for i in range(n_path):
            x = [pin_pos[i, proj[0]], opp_pos[i, proj[0]]]
            y = [pin_pos[i, proj[1]], opp_pos[i, proj[1]]]
            ax.plot(x, y, '-', color=colors[i], alpha=0.3, linewidth=0.8)
            ax.plot(pin_pos[i, proj[0]], pin_pos[i, proj[1]], '.', color=colors[i], markersize=3, alpha=0.6)
            ax.plot(opp_pos[i, proj[0]], opp_pos[i, proj[1]], '.', color=colors[i], markersize=3, alpha=0.6)

        # Mark start and end
        ax.plot(pin_pos[0, proj[0]], pin_pos[0, proj[1]], 'r^', markersize=8, label='pin start')
        ax.plot(pin_pos[-1, proj[0]], pin_pos[-1, proj[1]], 'rv', markersize=8, label='pin end')
        ax.plot(opp_pos[0, proj[0]], opp_pos[0, proj[1]], 'b^', markersize=8, label='opp start')
        ax.plot(opp_pos[-1, proj[0]], opp_pos[-1, proj[1]], 'bv', markersize=8, label='opp end')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} {title_proj} trail')
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'{name}: pin atom {pin_atom_idx} ({mol_enames[pin_atom_idx]}) → '
                 f'opp atom {opp_atom_idx} ({mol_enames[opp_atom_idx]})', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name}_trail.png'), dpi=150)
    plt.close(fig)

    # --- Pin-opp distance and tilt along path ---
    fig_d, (ax_d, ax_tilt) = plt.subplots(2, 1, figsize=(10, 8))
    dists = np.linalg.norm(pin_pos - opp_pos, axis=1)
    # Tilt = angle of pin→opp vector from horizontal (XY plane)
    dz = opp_pos[:, 2] - pin_pos[:, 2]
    dx = np.linalg.norm(opp_pos[:, :2] - pin_pos[:, :2], axis=1)
    tilts = np.degrees(np.arctan2(dz, dx))
    path_x = path_pts[:, 0]

    ax_d.plot(path_x, dists, 'k-o', markersize=3)
    ax_d.set_ylabel('pin–opp distance (Å)')
    ax_d.set_title(f'{name}: pin–opp distance and tilt along path')
    ax_d.grid(True, alpha=0.3)

    ax_tilt.plot(path_x, tilts, 'r-o', markersize=3)
    ax_tilt.set_ylabel('tilt angle (°)')
    ax_tilt.set_xlabel('pin x position (Å)')
    ax_tilt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_tilt.grid(True, alpha=0.3)

    fig_d.tight_layout()
    fig_d.savefig(os.path.join(save_dir, f'{name}_dist_tilt.png'), dpi=150)
    plt.close(fig_d)


# ---------------------------------------------------------------------------
# Reference data system for regression testing
# ---------------------------------------------------------------------------
# Stores physically meaningful properties (distances, heights, convergence)
# as human-readable JSON + XYZ text files. Comparison uses physical tolerances
# so small forcefield parameter changes don't break tests.
#
# File layout per test:
#   tests/ref_data/{ref_name}.ref.json   — physical properties (includes test_func field)
#   tests/ref_data/{ref_name}.ref.xyz    — final geometry (molecule + nearby substrate)

import json

REF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ref_data')


def _xyz_to_string(enames, apos):
    """Format atoms as XYZ string (no count/comment line)."""
    lines = []
    for e, p in zip(enames, apos):
        lines.append(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}')
    return '\n'.join(lines)


def _parse_xyz_string(s):
    """Parse XYZ body (no count line) into (enames, apos)."""
    enames, apos = [], []
    for line in s.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        enames.append(parts[0])
        apos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return enames, np.array(apos, dtype=np.float64)


def _first_layer_substrate(sub_apos, sub_enames, z_top=Z_SURF_TOP, z_tol=0.5):
    """Filter substrate to only 1st layer atoms (within z_tol of z_top)."""
    apos = np.asarray(sub_apos[:, :3], dtype=np.float64)
    mask = np.abs(apos[:, 2] - z_top) < z_tol
    return apos[mask], [sub_enames[i] for i in range(len(sub_enames)) if mask[i]]


def _substrate_near_molecule(sub_apos, sub_enames, mol_apos, margin=3.0, z_top=Z_SURF_TOP, z_tol=0.5):
    """1st-layer substrate atoms within bbox around molecule (XY + margin)."""
    apos1, enames1 = _first_layer_substrate(sub_apos, sub_enames, z_top, z_tol)
    mol = np.asarray(mol_apos[:, :3], dtype=np.float64)
    xmin, ymin = mol[:, 0].min() - margin, mol[:, 1].min() - margin
    xmax, ymax = mol[:, 0].max() + margin, mol[:, 1].max() + margin
    mask = (apos1[:, 0] >= xmin) & (apos1[:, 0] <= xmax) & (apos1[:, 1] >= ymin) & (apos1[:, 1] <= ymax)
    return apos1[mask], [enames1[i] for i in range(len(enames1)) if mask[i]]


def save_reference(ref_name, mol_enames, final_atoms, final_pos, final_force, final_torque,
                   sub_apos, sub_enames, z_rel, test_func=None, extra_distances=None):
    """Save reference data as JSON + XYZ files.

    Args:
        ref_name: e.g. 'ptcda_nacl' → writes tests/ref_data/ptcda_nacl.ref.json
        mol_enames: list of molecule element names
        final_atoms: (natoms, 3) final molecule atom positions
        final_pos: (3,) final COM position
        final_force: float |F|
        final_torque: float |τ|
        sub_apos: (N, 3) replicated substrate positions
        sub_enames: list of substrate element names
        z_rel: float, adsorption height above surface
        test_func: name of the test function this reference belongs to (e.g. 'test_relax_ptcda_nacl')
        extra_distances: list of dicts with {element, d_Na, d_Cl} per atom
    """
    os.makedirs(REF_DIR, exist_ok=True)
    json_path = os.path.join(REF_DIR, f'{ref_name}.ref.json')
    xyz_path = os.path.join(REF_DIR, f'{ref_name}.ref.xyz')

    # Collect per-atom distances to nearest Na and Cl
    atom_distances = []
    for ia, e in enumerate(mol_enames):
        d_Na, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Na')
        d_Cl, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Cl')
        atom_distances.append({'element': e, 'd_Na': round(d_Na, 3), 'd_Cl': round(d_Cl, 3)})

    # 1st-layer substrate atoms near molecule
    sub_near_pos, sub_near_names = _substrate_near_molecule(sub_apos, sub_enames, final_atoms)

    ref = {
        'description': f'Reference for {ref_name}: relaxed molecule on NaCl(100) folded basis',
        'test_func': test_func or '',
        'test_module': 'tests.test_folded_relax',
        'ref_name': ref_name,
        'z_rel': round(z_rel, 4),
        'force': round(final_force, 6),
        'torque': round(final_torque, 6),
        'atom_distances': atom_distances,
        'n_atoms': len(mol_enames),
    }

    with open(json_path, 'w') as f:
        json.dump(ref, f, indent=2)
        f.write('\n')

    # Save combined XYZ: substrate (1st layer near molecule) + molecule
    with open(xyz_path, 'w') as f:
        total = len(sub_near_names) + len(mol_enames)
        f.write(f'{total}\n')
        f.write(f'{ref_name} reference: {len(sub_near_names)} substrate + {len(mol_enames)} molecule atoms\n')
        for e, p in zip(sub_near_names, sub_near_pos):
            f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
        for e, p in zip(mol_enames, final_atoms):
            f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')

    print(f'[REF] Saved {json_path}')
    print(f'[REF] Saved {xyz_path}')


def load_reference(ref_name):
    """Load reference data. Returns dict or None if not found."""
    json_path = os.path.join(REF_DIR, f'{ref_name}.ref.json')
    xyz_path = os.path.join(REF_DIR, f'{ref_name}.ref.xyz')
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        ref = json.load(f)
    if os.path.exists(xyz_path):
        with open(xyz_path) as f:
            lines = f.readlines()
        # Skip count + comment, parse rest
        xyz_body = ''.join(lines[2:])
        ref['xyz_enames'], ref['xyz_apos'] = _parse_xyz_string(xyz_body)
    return ref


def compare_to_reference(ref_name, mol_enames, final_atoms, final_pos, final_force, final_torque,
                         sub_apos, sub_enames, z_rel,
                         tol_z=0.5, tol_dist=0.3, force_thresh=0.05, torque_thresh=0.05):
    """Compare current results to saved reference.

    Uses physical tolerances:
    - z_rel within ±tol_z Å
    - per-atom d_Na, d_Cl within ±tol_dist Å
    - "O closer to Na than Cl" boolean must match
    - force < force_thresh, torque < torque_thresh

    Returns (passed: bool, messages: list[str]).
    """
    ref = load_reference(ref_name)
    if ref is None:
        return False, [f'Reference {ref_name} not found in {REF_DIR}']

    msgs = []
    passed = True

    # z_rel comparison
    ref_z = ref['z_rel']
    dz = abs(z_rel - ref_z)
    if dz > tol_z:
        passed = False
        msgs.append(f'FAIL z_rel: {z_rel:.3f} vs ref {ref_z:.3f} (Δ={dz:.3f} > {tol_z})')
    else:
        msgs.append(f'OK   z_rel: {z_rel:.3f} vs ref {ref_z:.3f} (Δ={dz:.3f})')

    # Force convergence
    if final_force > force_thresh:
        passed = False
        msgs.append(f'FAIL |F|={final_force:.6f} > {force_thresh}')
    else:
        msgs.append(f'OK   |F|={final_force:.6f} < {force_thresh}')

    # Torque convergence
    if final_torque > torque_thresh:
        passed = False
        msgs.append(f'FAIL |τ|={final_torque:.6f} > {torque_thresh}')
    else:
        msgs.append(f'OK   |τ|={final_torque:.6f} < {torque_thresh}')

    # Per-atom distances
    ref_dists = ref['atom_distances']
    if len(ref_dists) != len(mol_enames):
        passed = False
        msgs.append(f'FAIL atom count: {len(mol_enames)} vs ref {len(ref_dists)}')
    else:
        for ia, (e, ref_d) in enumerate(zip(mol_enames, ref_dists)):
            d_Na, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Na')
            d_Cl, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Cl')
            ref_Na, ref_Cl = ref_d['d_Na'], ref_d['d_Cl']
            dNa = abs(d_Na - ref_Na)
            dCl = abs(d_Cl - ref_Cl)
            # Boolean: is O closer to Na than Cl?
            ref_bool = ref_Na < ref_Cl
            cur_bool = d_Na < d_Cl
            bool_match = ref_bool == cur_bool
            dist_ok = dNa < tol_dist and dCl < tol_dist
            if not bool_match:
                passed = False
                msgs.append(f'FAIL atom {ia} ({e}): Na<Cl changed (ref={ref_bool}, cur={cur_bool})')
            elif not dist_ok:
                passed = False
                msgs.append(f'FAIL atom {ia} ({e}): d(Na) Δ={dNa:.3f} d(Cl) Δ={dCl:.3f} > {tol_dist}')
            else:
                msgs.append(f'OK   atom {ia} ({e}): d(Na)={d_Na:.3f} (ref {ref_Na:.3f}) d(Cl)={d_Cl:.3f} (ref {ref_Cl:.3f})')

    return passed, msgs
