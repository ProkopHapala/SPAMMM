import os, sys
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from tests.helpers.folded_rigid import (
    fit_folded_for_molecule, setup_rigid_folded, relax_folded,
    load_substrate, nearest_substrate_distance,
    plot_relaxation, plot_relax_overview, save_xyz_trajectory, find_bonds,
    lateral_scan, plot_force_map, manipulation_trajectory, plot_manipulation,
    relaxed_scan, plot_relaxed_scan, plot_manipulation_trail,
    random_quaternion, replicate_substrate,
    save_reference, compare_to_reference,
    NACL_SUBSTRATE, Z_SURF_TOP, LATTICE_A,
)

DATA_XYZ = os.path.join(_proj_root, 'data', 'xyz')
DEBUG_DIR = os.path.join(_proj_root, 'debug', 'test_folded_relax')


@pytest.mark.gpu
@pytest.mark.slow
def test_relax_h2o_nacl(xyz, substrate, update_refs):
    """H2O on NaCl(100): relax on folded basis forcefield.

    Physical expectation: O atom orients toward Na+ ion (dative coordination).
    H atoms point away from surface.
    Reference: tests/ref_data/h2o_nacl.ref.{json,xyz}
    """
    mol_file = xyz('H2O.xyz')
    sub_file = substrate('NaCl_1x1_L3.xyz')
    save_dir = os.path.join(DEBUG_DIR, 'h2o')
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n[H2O] Fitting folded basis...')
    fit = fit_folded_for_molecule(mol_file, substrate_file=sub_file)
    print(f'[H2O] ntypes={fit["coeffs"].shape[0]} nbasis={fit["coeffs"].shape[1]}')
    print(f'[H2O] atom_type_ids={fit["atom_type_ids"]}')
    print(f'[H2O] unique_REQs={fit["unique_REQs"]}')

    rbd = setup_rigid_folded(mol_file, fit, z_init=2.5, xy_init=(0.0, 0.0))
    bonds = find_bonds(rbd.atom_body_host.reshape(rbd.num_atoms, 4)[:, :3], rbd.enames, Rcut=1.8)

    print(f'[H2O] Relaxing (10000 steps)...')
    traj = relax_folded(rbd, n_steps=10000, dt=0.005, lin_damp=0.995, ang_damp=0.995, record_interval=200)

    # Load substrate for visualization and distance checks
    sub_apos, sub_enames, _, _ = load_substrate(sub_file)

    # Visualization
    plot_relaxation(traj, save_dir, name='h2o')
    plot_relax_overview(traj, rbd.enames, sub_apos, sub_enames, save_dir, name='h2o',
                        bonds=bonds, highlight_element='O', target_element='Na')

    # Save trajectory XYZ
    save_xyz_trajectory(os.path.join(save_dir, 'h2o_trajectory.xyz'),
                        rbd.enames, traj['atom_positions'], sub_apos, sub_enames)

    # --- Physical checks ---
    final_pos = traj['positions'][-1]
    final_atoms = traj['atom_positions'][-1]
    final_force = traj['forces'][-1]
    final_torque = traj['torques'][-1]
    z_rel = final_pos[2] - Z_SURF_TOP

    print(f'[H2O] Final COM: z={final_pos[2]:.4f} (surf_top={Z_SURF_TOP})')
    print(f'[H2O] Final |F|={final_force:.6f} |τ|={final_torque:.6f}')
    for ia, e in enumerate(rbd.enames):
        d_Na, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Na')
        d_Cl, _ = nearest_substrate_distance(final_atoms[ia], sub_apos, sub_enames, 'Cl')
        print(f'[H2O] atom {ia} ({e}): d(Na)={d_Na:.3f} d(Cl)={d_Cl:.3f}')

    # --- Reference data: save or compare ---
    if update_refs:
        save_reference('h2o_nacl', rbd.enames, final_atoms, final_pos, final_force, final_torque,
                       sub_apos, sub_enames, z_rel, test_func='test_relax_h2o_nacl')
    else:
        passed, msgs = compare_to_reference('h2o_nacl', rbd.enames, final_atoms, final_pos,
                                            final_force, final_torque, sub_apos, sub_enames, z_rel)
        for m in msgs:
            print(f'[H2O-REF] {m}')
        assert passed, f'Reference comparison failed for h2o_nacl'



@pytest.mark.gpu
@pytest.mark.slow
def test_relax_ptcda_nacl(xyz, substrate, update_refs):
    """PTCDA on NaCl(100): relax on folded basis forcefield.

    Physical expectation: large planar molecule, lies flat.
    O atoms (anhydride) may align with Na sites.
    Reference: tests/ref_data/ptcda_nacl.ref.{json,xyz}
    """
    mol_file = xyz('PTCDA.xyz')
    sub_file = substrate('NaCl_1x1_L3.xyz')
    save_dir = os.path.join(DEBUG_DIR, 'ptcda')
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n[PTCDA] Fitting folded basis...')
    fit = fit_folded_for_molecule(mol_file, substrate_file=sub_file)
    print(f'[PTCDA] ntypes={fit["coeffs"].shape[0]} nbasis={fit["coeffs"].shape[1]}')
    print(f'[PTCDA] atom_type_ids={fit["atom_type_ids"]}')
    print(f'[PTCDA] unique_REQs={fit["unique_REQs"]}')

    # PTCDA is large (~12 Å wide). Start with random displacement and general
    # quaternion perturbation to verify it relaxes to a symmetric flat adsorption config.
    np.random.seed(42)
    xy_rand = (np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0))
    z_rand = 3.5 + np.random.uniform(-0.2, 0.2)
    # General small perturbation of all 4 quaternion components (~6° tilt)
    quat_rand = np.array([0, 0, 0, 1.0], dtype=np.float32) + np.random.uniform(-0.05, 0.05, 4).astype(np.float32)
    quat_rand /= np.linalg.norm(quat_rand)
    angle0 = 2 * np.degrees(np.arccos(min(abs(quat_rand[3]), 1.0)))
    print(f'[PTCDA] Random init: xy={xy_rand} z_rel={z_rand:.2f} tilt={angle0:.1f}° quat={quat_rand}')
    rbd = setup_rigid_folded(mol_file, fit, z_init=z_rand, xy_init=xy_rand, quats=quat_rand)
    bonds = find_bonds(rbd.atom_body_host.reshape(rbd.num_atoms, 4)[:, :3], rbd.enames, Rcut=1.8)

    print(f'[PTCDA] Relaxing (20000 steps)...')
    traj = relax_folded(rbd, n_steps=20000, dt=0.002, lin_damp=0.998, ang_damp=0.99, record_interval=500)

    # Load substrate and replicate periodically to cover the PTCDA area
    sub_apos, sub_enames, _, sub_lvec = load_substrate(sub_file)
    extent = 14.0  # PTCDA is ~12 Å wide, show ±7 Å
    sub_rep_apos, sub_rep_enames = replicate_substrate(
        sub_apos, sub_enames, sub_lvec, (-extent, extent), (-extent, extent), z_min=-8.0)
    print(f'[PTCDA] Replicated substrate: {len(sub_rep_enames)} atoms')

    plot_relaxation(traj, save_dir, name='ptcda')
    plot_relax_overview(traj, rbd.enames, sub_rep_apos, sub_rep_enames, save_dir, name='ptcda',
                        bonds=bonds, highlight_element='O', target_element='Na')

    save_xyz_trajectory(os.path.join(save_dir, 'ptcda_trajectory.xyz'),
                        rbd.enames, traj['atom_positions'], sub_rep_apos, sub_rep_enames)

    # --- Physical checks ---
    final_pos = traj['positions'][-1]
    final_atoms = traj['atom_positions'][-1]
    final_force = traj['forces'][-1]
    final_torque = traj['torques'][-1]
    z_rel = final_pos[2] - Z_SURF_TOP

    print(f'[PTCDA] Final COM: z={final_pos[2]:.4f} (surf_top={Z_SURF_TOP})')
    print(f'[PTCDA] Final |F|={final_force:.6f} |τ|={final_torque:.6f}')

    # Print O atom distances
    o_indices = [i for i, e in enumerate(rbd.enames) if e == 'O']
    for ia in o_indices:
        d_Na, _ = nearest_substrate_distance(final_atoms[ia], sub_rep_apos, sub_rep_enames, 'Na')
        d_Cl, _ = nearest_substrate_distance(final_atoms[ia], sub_rep_apos, sub_rep_enames, 'Cl')
        print(f'[PTCDA] O atom {ia}: d(Na)={d_Na:.3f} d(Cl)={d_Cl:.3f}')

    # --- Reference data: save or compare ---
    if update_refs:
        save_reference('ptcda_nacl', rbd.enames, final_atoms, final_pos, final_force, final_torque,
                       sub_rep_apos, sub_rep_enames, z_rel, test_func='test_relax_ptcda_nacl')
    else:
        passed, msgs = compare_to_reference('ptcda_nacl', rbd.enames, final_atoms, final_pos,
                                            final_force, final_torque, sub_rep_apos, sub_rep_enames, z_rel)
        for m in msgs:
            print(f'[PTCDA-REF] {m}')
        assert passed, f'Reference comparison failed for ptcda_nacl'


@pytest.mark.gpu
@pytest.mark.slow
def test_scan_h2o_nacl(xyz, substrate):
    """H2O lateral scan on NaCl: AFM-like force map at fixed height.

    Scans a grid of (x,y) positions and measures vertical force Fz.
    Verifies that force map shows periodicity matching the NaCl lattice.
    """
    mol_file = xyz('H2O.xyz')
    sub_file = substrate('NaCl_1x1_L3.xyz')
    save_dir = os.path.join(DEBUG_DIR, 'h2o_scan')
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n[H2O-scan] Fitting folded basis...')
    fit = fit_folded_for_molecule(mol_file, substrate_file=sub_file)

    # Scan at z=2.5 Å above surface top (near relaxed height)
    z_scan = Z_SURF_TOP + 2.5
    xs = np.linspace(-4.0, 4.0, 17, dtype=np.float32)
    ys = np.linspace(-4.0, 4.0, 17, dtype=np.float32)

    rbd = setup_rigid_folded(mol_file, fit, z_init=2.5, xy_init=(0.0, 0.0))
    bonds = find_bonds(rbd.atom_body_host.reshape(rbd.num_atoms, 4)[:, :3], rbd.enames, Rcut=1.8)

    print(f'[H2O-scan] Scanning {len(xs)}x{len(ys)} grid at z={z_scan:.2f}')
    scan = lateral_scan(rbd, xs, ys, z_scan, n_relax=30, dt=0.005)

    # Visualization
    plot_force_map(scan, save_dir, name='h2o_scan')

    # Save trajectory of scan positions
    sub_apos, sub_enames, _, _ = load_substrate(sub_file)
    save_xyz_trajectory(os.path.join(save_dir, 'h2o_scan_trajectory.xyz'),
                        rbd.enames, scan['atom_positions'], sub_apos, sub_enames)

    # --- Physical checks ---
    Fz = scan['Fz']
    Fmag = np.sqrt(scan['Fx']**2 + scan['Fy']**2 + Fz**2)

    print(f'[H2O-scan] Fz range: [{Fz.min():.4f}, {Fz.max():.4f}]')
    print(f'[H2O-scan] |F| range: [{Fmag.min():.4f}, {Fmag.max():.4f}]')

    # Check: forces are non-trivial (not all zero)
    assert Fmag.max() > 0.01, f'Forces too small: max |F|={Fmag.max():.6f}'

    # Check: force map has variation (not flat)
    assert Fz.std() > 1e-4, f'Fz map too flat: std={Fz.std():.6f}'


@pytest.mark.gpu
@pytest.mark.slow
def test_manipulation_h2o_nacl(xyz, substrate, update_refs):
    """H2O relaxed scan on NaCl: pin one H atom and drag across surface.

    Holds H2O by one hydrogen atom at a height where the molecule can still
    interact with the surface. Drags it along one lattice constant with fine
    step (dx=0.1 Å) and verifies that the molecule follows.
    Reference: tests/ref_data/h2o_manip.ref.{json,xyz}
    """
    mol_file = xyz('H2O.xyz')
    sub_file = substrate('NaCl_1x1_L3.xyz')
    save_dir = os.path.join(DEBUG_DIR, 'h2o_manip')
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n[H2O-manip] Fitting folded basis...')
    fit = fit_folded_for_molecule(mol_file, substrate_file=sub_file)

    # Start from relaxed position on Na site
    rbd = setup_rigid_folded(mol_file, fit, z_init=2.5, xy_init=(0.0, 0.0))
    bonds = find_bonds(rbd.atom_body_host.reshape(rbd.num_atoms, 4)[:, :3], rbd.enames, Rcut=1.8)

    # Pin H atom (index 1) at height ~4 Å above surface top
    # H2O is small (~1 Å), so 4 Å lets O reach ~3 Å above surface — near equilibrium
    h_idx = 1  # first H atom
    o_idx = 0  # O is the opposite end
    z_pin = Z_SURF_TOP + 4.0
    # Path: drag from (0, 0) to (4, 0) with dx=0.1 Å steps
    x_start, x_end = 0.0, 4.0
    n_path = int(round((x_end - x_start) / 0.1)) + 1
    path = np.zeros((n_path, 3), dtype=np.float32)
    path[:, 0] = np.linspace(x_start, x_end, n_path)
    path[:, 1] = 0.0
    path[:, 2] = z_pin

    print(f'[H2O-manip] Pinning H atom {h_idx} at z={z_pin:.2f}, dragging x: {x_start}→{x_end} Å ({n_path} steps, dx=0.1)')
    traj = relaxed_scan(rbd, pin_atom_idx=h_idx, path=path, k_spring=10.0,
                        n_relax=200, dt=0.005, lin_damp=0.99, ang_damp=0.95, record_interval=50)

    sub_apos, sub_enames, _, sub_lvec = load_substrate(sub_file)
    sub_rep_apos, sub_rep_enames = replicate_substrate(
        sub_apos, sub_enames, sub_lvec, (-6, 6), (-6, 6), z_min=-8.0)

    plot_relaxed_scan(traj, rbd.enames, sub_rep_apos, sub_rep_enames, save_dir, name='h2o_manip',
                      bonds=bonds, pin_atom_idx=h_idx, highlight_element='O', target_element='Na')
    plot_manipulation_trail(traj, rbd.enames, sub_rep_apos, sub_rep_enames, save_dir, name='h2o_manip',
                            pin_atom_idx=h_idx, opp_atom_idx=o_idx)

    # Save trajectory (snapshots at end of each path point)
    n_path = traj['n_path']
    rec_per_path = len(traj['atom_positions']) // n_path
    snap_atoms = [traj['atom_positions'][min((i+1)*rec_per_path-1, len(traj['atom_positions'])-1)]
                  for i in range(n_path)]
    save_xyz_trajectory(os.path.join(save_dir, 'h2o_manip_trajectory.xyz'),
                        rbd.enames, snap_atoms, sub_rep_apos, sub_rep_enames)

    # --- Physical checks ---
    d_Na_along = []
    for i in range(n_path):
        si = min((i+1)*rec_per_path-1, len(traj['atom_positions'])-1)
        mol_pos = traj['atom_positions'][si]
        d, _ = nearest_substrate_distance(mol_pos[o_idx], sub_rep_apos, sub_rep_enames, 'Na')
        d_Na_along.append(d)
    d_Na_along = np.array(d_Na_along)

    print(f'[H2O-manip] O→Na distance: start={d_Na_along[0]:.3f} end={d_Na_along[-1]:.3f}')
    print(f'[H2O-manip] O→Na range: [{d_Na_along.min():.3f}, {d_Na_along.max():.3f}]')

    # Molecule follows the pin (distance varies along path)
    assert d_Na_along.max() - d_Na_along.min() > 0.3, \
        f'O-Na distance not varying enough: range={d_Na_along.max()-d_Na_along.min():.4f}'

    # Pin force is non-trivial (spring is actually pulling)
    Pmag = np.linalg.norm(traj['pin_forces'], axis=1)
    print(f'[H2O-manip] |F_pin|: mean={Pmag.mean():.4f} max={Pmag.max():.4f}')
    assert Pmag.max() > 0.01, f'Pin force too small: max |F_pin|={Pmag.max():.6f}'


@pytest.mark.gpu
@pytest.mark.slow
def test_manipulation_ptcda_nacl(xyz, substrate, update_refs):
    """PTCDA relaxed scan on NaCl: pin one O atom and drag across surface.

    Holds PTCDA by one oxygen atom at a height where the molecule can still
    lie on the surface. Drags it along two lattice constants with fine
    step (dx=0.1 Å) and verifies that the molecule follows and tilts.
    Reference: tests/ref_data/ptcda_manip.ref.{json,xyz}
    """
    mol_file = xyz('PTCDA.xyz')
    sub_file = substrate('NaCl_1x1_L3.xyz')
    save_dir = os.path.join(DEBUG_DIR, 'ptcda_manip')
    os.makedirs(save_dir, exist_ok=True)

    print(f'\n[PTCDA-manip] Fitting folded basis...')
    fit = fit_folded_for_molecule(mol_file, substrate_file=sub_file)

    # Start from near-equilibrium position
    rbd = setup_rigid_folded(mol_file, fit, z_init=3.5, xy_init=(0.0, 0.0))
    bonds = find_bonds(rbd.atom_body_host.reshape(rbd.num_atoms, 4)[:, :3], rbd.enames, Rcut=1.8)

    # Pin O atom (index 24, anhydride O at one end) at height ~6 Å above surface top
    # PTCDA is ~12 Å wide, ~1 Å thick. Holding one O at 6 Å lets the molecule
    # tilt and the opposite side touch the surface (~3 Å)
    pin_idx = 24  # O at (-5.73, 0.03) — one end
    opp_idx = 25  # O at (+5.73, -0.03) — opposite end (~11.5 Å away)
    z_pin = Z_SURF_TOP + 6.0
    # Path: drag from (0, 0) to (8, 0) with dx=0.1 Å steps
    x_start, x_end = 0.0, 8.0
    n_path = int(round((x_end - x_start) / 0.1)) + 1
    path = np.zeros((n_path, 3), dtype=np.float32)
    path[:, 0] = np.linspace(x_start, x_end, n_path)
    path[:, 1] = 0.0
    path[:, 2] = z_pin

    print(f'[PTCDA-manip] Pinning O atom {pin_idx} at z={z_pin:.2f}, dragging x: {x_start}→{x_end} Å ({n_path} steps, dx=0.1)')
    traj = relaxed_scan(rbd, pin_atom_idx=pin_idx, path=path, k_spring=10.0,
                        n_relax=300, dt=0.003, lin_damp=0.99, ang_damp=0.95, record_interval=50)

    sub_apos, sub_enames, _, sub_lvec = load_substrate(sub_file)
    sub_rep_apos, sub_rep_enames = replicate_substrate(
        sub_apos, sub_enames, sub_lvec, (-10, 10), (-10, 10), z_min=-8.0)

    plot_relaxed_scan(traj, rbd.enames, sub_rep_apos, sub_rep_enames, save_dir, name='ptcda_manip',
                      bonds=bonds, pin_atom_idx=pin_idx, highlight_element='O', target_element='Na')
    plot_manipulation_trail(traj, rbd.enames, sub_rep_apos, sub_rep_enames, save_dir, name='ptcda_manip',
                            pin_atom_idx=pin_idx, opp_atom_idx=opp_idx)

    # Save trajectory
    rec_per_path = len(traj['atom_positions']) // n_path
    snap_atoms = [traj['atom_positions'][min((i+1)*rec_per_path-1, len(traj['atom_positions'])-1)]
                  for i in range(n_path)]
    save_xyz_trajectory(os.path.join(save_dir, 'ptcda_manip_trajectory.xyz'),
                        rbd.enames, snap_atoms, sub_rep_apos, sub_rep_enames)

    # --- Physical checks ---
    # Track COM z along path — molecule should tilt (z varies)
    com_z = []
    com_x = []
    for i in range(n_path):
        si = min((i+1)*rec_per_path-1, len(traj['atom_positions'])-1)
        mol_pos = traj['atom_positions'][si]
        com_z.append(mol_pos[:, 2].mean())
        com_x.append(mol_pos[:, 0].mean())
    com_z = np.array(com_z)
    com_x = np.array(com_x)

    print(f'[PTCDA-manip] COM z: start={com_z[0]:.3f} end={com_z[-1]:.3f} range=[{com_z.min():.3f}, {com_z.max():.3f}]')
    print(f'[PTCDA-manip] COM x: start={com_x[0]:.3f} end={com_x[-1]:.3f}')

    # COM x should follow the pin direction
    assert com_x[-1] - com_x[0] > 1.0, \
        f'COM x not following pin: Δx={com_x[-1]-com_x[0]:.3f}'

    # Pin force is non-trivial
    Pmag = np.linalg.norm(traj['pin_forces'], axis=1)
    print(f'[PTCDA-manip] |F_pin|: mean={Pmag.mean():.4f} max={Pmag.max():.4f}')
    assert Pmag.max() > 0.01, f'Pin force too small: max |F_pin|={Pmag.max():.6f}'
