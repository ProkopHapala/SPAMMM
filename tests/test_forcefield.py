import pytest, numpy as np, os, datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spammm.AtomicSystem import AtomicSystem
from spammm.forcefields.UFF import UFF_CL
from tests.helpers.geometry import distort, bond_lengths, bond_angle, assert_geometry, find_bonds, save_xyz_frames, plot_geometry
from tests.helpers.parity import plot_curves, overlay_plot, rmse, correlation

os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

def _debug_dir(name='forcefield'):
    d = os.path.join('debug', f'{datetime.date.today()}_{name}')
    os.makedirs(d, exist_ok=True)
    return d

# (molecule, forcefield, nsteps, expected_bonds: {(i,j): (r0, tol_frac)}, bNonBond)
RELAX_CASES = [
    ('H2O.xyz',     'UFF',  200, {(0,1): (0.96, 0.15), (0,2): (0.96, 0.15)}, False),
    ('CH4.xyz',     'UFF',  500, {(0,1): (1.09, 0.15), (0,2): (1.09, 0.15), (0,3): (1.09, 0.15)}, False),
    ('benzene.xyz', 'UFF',  2000, {(0,1): (1.40, 0.10)}, False),
    ('HCOOH.xyz',   'UFF',  500, {(0,1): (1.10, 0.15), (1,2): (1.21, 0.15), (1,3): (1.30, 0.15), (3,4): (0.98, 0.15)}, False),
    ('uracil.xyz',  'UFF',  1000, {(0,1): (1.34, 0.15), (1,2): (1.37, 0.15), (3,6): (1.23, 0.15), (5,7): (1.23, 0.15)}, False),
    ('PTCDA.xyz',   'UFF',  3000, {(0,2): (1.40, 0.15), (9,27): (1.21, 0.15), (12,31): (1.10, 0.15)}, False),
    ('H2O.xyz',     'SPFF', 300,  {(0,1): (0.96, 0.15), (0,2): (0.96, 0.15)}, False),
    ('CH4.xyz',     'SPFF', 500,  {(0,1): (1.09, 0.15), (0,2): (1.09, 0.15), (0,3): (1.09, 0.15)}, False),
    ('benzene.xyz', 'SPFF', 500,  {(0,1): (1.40, 0.10)}, False),
    ('HCOOH.xyz',   'SPFF', 500,  {(0,1): (1.10, 0.15), (1,2): (1.21, 0.15), (1,3): (1.30, 0.15), (3,4): (0.98, 0.15)}, False),
    ('uracil.xyz',  'SPFF', 1000, {(0,1): (1.34, 0.15), (1,2): (1.37, 0.15), (3,6): (1.23, 0.15), (5,7): (1.23, 0.15)}, False),
    ('PTCDA.xyz',   'SPFF', 3000, {(0,2): (1.40, 0.15), (9,27): (1.21, 0.15), (12,31): (1.10, 0.15)}, False),
    # SPFF with non-bonded interactions (1-2 and 1-3 exclusions)
    ('H2O.xyz',     'SPFF', 300,  {(0,1): (0.96, 0.15), (0,2): (0.96, 0.15)}, True),
    ('CH4.xyz',     'SPFF', 500,  {(0,1): (1.09, 0.15), (0,2): (1.09, 0.15), (0,3): (1.09, 0.15)}, True),
    ('benzene.xyz', 'SPFF', 1000, {(0,1): (1.40, 0.10)}, True),
    ('HCOOH.xyz',   'SPFF', 1000, {(0,1): (1.10, 0.15), (1,2): (1.21, 0.15), (1,3): (1.30, 0.15), (3,4): (0.98, 0.15)}, True),
    ('uracil.xyz',  'SPFF', 2000, {(0,1): (1.34, 0.15), (1,2): (1.37, 0.15), (3,6): (1.23, 0.15), (5,7): (1.23, 0.15)}, True),
    ('PTCDA.xyz',   'SPFF', 5000, {(0,2): (1.40, 0.15), (9,27): (1.21, 0.15), (12,31): (1.10, 0.15)}, True),
]

@pytest.mark.gpu
@pytest.mark.parametrize('mol_file,ff,nsteps,expected,bNonBond', RELAX_CASES)
def test_relax(xyz, mol_file, ff, nsteps, expected, bNonBond):
    """Distort molecule, relax with UFF or SPFF, check geometry converges."""
    mol = AtomicSystem(fname=xyz(mol_file))
    apos_init = mol.apos.copy()
    mol.apos = distort(mol.apos, 0.2)
    apos_distort = mol.apos.copy()
    save_dir = _debug_dir('relax')
    nb_tag = '_nb' if bNonBond else ''
    tag = f'{mol_file.replace(".xyz","")}_{ff}{nb_tag}'
    if ff == 'UFF':
        from spammm.forcefields.UFF import UFF_CL
        uff = UFF_CL()
        uff.toUFF(mol)
        # Approximate masses: C=12, H=1, O=16, N=14
        mass_map = {'H': 1.0, 'C': 12.0, 'O': 16.0, 'N': 14.0}
        masses = np.array([mass_map.get(e, 12.0) for e in mol.enames], dtype=np.float32)
        uff.upload_positions(mol.apos, masses=masses)
        uff.relax(nsteps=nsteps, dt=0.01, damp=0.95)
        mol.apos = uff.get_positions()
        forces = uff.get_forces()[0]
        fmax = np.max(np.linalg.norm(forces, axis=1))
        assert fmax < 1.0, f'{mol_file} {ff}: force too large after relax: {fmax:.2f}'
    elif ff == 'SPFF':
        from spammm.forcefields.MolecularDynamics import MolecularDynamics
        from spammm.forcefields.SPFF import SPFF
        from spammm.topology.FFparams import SPFFparams
        import pyopencl as cl
        # 1. Assign atom types
        params = SPFFparams('data/')
        mol.atypes = np.array([params.getAtomType(e, bErr=False) for e in mol.enames], dtype=np.int32)
        # 2. Build SPFF topology (positions, neighbors, bonds, pi-orbitals)
        spff = SPFF()
        spff.toSPFFsp3_loc(mol, params.atom_types_map)
        # 3. Store mass in apos.w for atom dynamics
        mass_map = {'H': 1.0, 'C': 12.0, 'O': 16.0, 'N': 14.0}
        for ia in range(spff.natoms):
            e = mol.enames[ia]
            spff.apos[ia, 3] = mass_map.get(e, 12.0)
        # 4. Initialize MD engine, upload to GPU
        md = MolecularDynamics(enable_nonbond=bNonBond)
        md.realloc(spff, nSystems=1)
        md.upload_all_systems()
        md.setup_kernels()
        # 5. Zero velocities
        cl.enqueue_fill_buffer(md.queue, md.buffer_dict['avel'], np.float32(0), 0, md.buffer_dict['avel'].size)
        md.queue.finish()
        # 6. Run relaxation with rotational pi-orbital dynamics
        md.relax(nsteps=nsteps, dt=0.01, damp=0.95, Flimit=100.0, use_rot=True, do_nb=bNonBond)
        # 7. Retrieve results (SPFF reorders atoms nodes-first; map back to original order)
        pos_spff = md.get_positions()
        perm = getattr(mol, 'perm_nodes_first', list(range(len(pos_spff))))
        inv_perm = getattr(mol, 'perm_inverse', list(range(len(pos_spff))))
        mol.apos = pos_spff[inv_perm]
        forces_full = md.get_forces()
        forces = forces_full[:len(pos_spff)][inv_perm]
        fmax = np.max(np.linalg.norm(forces[:, :3], axis=1))
        assert fmax < 1.0, f'{mol_file} {ff}{nb_tag}: force too large after relax: {fmax:.2f}'
        # Restore enames to original order for debug output
        mol.enames = [mol.enames[i] for i in inv_perm]
    apos_final = mol.apos.copy()
    # Save initial+final XYZ (distorted initial, final relaxed) in one file
    save_xyz_frames(os.path.join(save_dir, f'{tag}_init_final.xyz'),
                    mol.enames, [apos_distort, apos_final],
                    comments=[f'{tag} distorted (initial)', f'{tag} relaxed (final) fmax={fmax:.3f}'])
    # Plot initial vs final geometry with bond lengths
    bonds = find_bonds(apos_final, mol.enames)
    # Pick best projection plane (most spread)
    spread = np.std(apos_final, axis=0)
    proj = 'xy' if spread[2] <= spread[1] else ('xz' if spread[2] <= spread[0] else 'yz')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_geometry(ax1, apos_distort, mol.enames, bonds, title=f'{tag} initial (distorted)', proj=proj)
    plot_geometry(ax2, apos_final, mol.enames, bonds, title=f'{tag} final (relaxed)', proj=proj)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{tag}_geometry.png'), dpi=150)
    plt.close(fig)
    assert_geometry(mol.apos, expected, name=f'{mol_file}_{ff}')

@pytest.mark.gpu
def test_uff_energy_finite(xyz):
    """UFF single-point energy must be finite and non-zero for stable molecule.

    With bDoNonBonded=False (default), only bond+angle terms are active.
    These are harmonic potentials (E = k*dl^2 >= 0), so energy is always >= 0.
    A non-zero positive energy means the molecule is slightly strained relative
    to UFF equilibrium parameters (which differ from the input geometry).
    """
    from spammm.forcefields.UFF import UFF_CL
    mol = AtomicSystem(fname=xyz('H2O.xyz'))
    uff = UFF_CL()
    uff.toUFF(mol)
    uff.upload_positions(mol.apos)
    uff.run_eval_step()
    E = uff.get_total_energy()
    assert np.all(np.isfinite(E)), 'UFF energy is NaN/Inf'
    assert E[0] > 0, f'UFF energy should be positive (harmonic bonds+angles): {E[0]}'

@pytest.mark.gpu
def test_uff_force_newton3(xyz):
    """Net force on isolated molecule should be ~0 (Newton's 3rd law)."""
    from spammm.forcefields.UFF import UFF_CL
    mol = AtomicSystem(fname=xyz('CH4.xyz'))
    uff = UFF_CL()
    uff.toUFF(mol)
    uff.upload_positions(mol.apos)
    uff.run_eval_step()
    forces = uff.get_forces()
    net_force = np.sum(forces[0], axis=0)
    assert np.linalg.norm(net_force) < 0.1, f'Net force not zero: {net_force}'

INVARIANT_CASES = [
    # (mol_file, mass_map, nsteps, dt, distort_amp, dE_abs_tol, dL_tol)
    ('CH4.xyz',   {'H': 1.0, 'C': 12.0},             1000, 0.002, 0.05, 1e-3, 1e-6),
    ('CH2NH.xyz', {'H': 1.0, 'C': 12.0, 'N': 14.0},  2000, 0.001, 0.03, 1e-3, 5e-6),
]

@pytest.mark.visual
@pytest.mark.gpu
@pytest.mark.parametrize('mol_file,mass_map,nsteps,dt,distort_amp,dE_abs_tol,dL_tol', INVARIANT_CASES)
def test_invariants(xyz, mol_file, mass_map, nsteps, dt, distort_amp, dE_abs_tol, dL_tol):
    """MD invariants: energy, linear momentum, angular momentum conservation.

    Uses GPU-side symplectic Euler integrator (updateAtomsSPFFf4) with damp=1.0.
    Velocity is at half-steps; KE computed by averaging v(n) and v(n+1).
    Tests both symmetric (CH4) and asymmetric (CH2NH) molecules.
    Starts from strained (distorted) configuration to test invariants under oscillation.
    """
    from spammm.forcefields.UFF import UFF_CL
    import pyopencl as cl
    mol = AtomicSystem(fname=xyz(mol_file))
    uff = UFF_CL()
    uff.toUFF(mol)
    masses = np.array([mass_map[e] for e in mol.enames], dtype=np.float32)
    # Start from strained (distorted) configuration — no relaxation.
    # This ensures oscillating dynamics to test invariants conservation.
    strained_pos = distort(mol.apos, distort_amp)
    uff.upload_positions(strained_pos, masses=masses)
    uff.set_md_params(dt=dt, damp=1.0, Flimit=1e10)
    cl.enqueue_fill_buffer(uff.queue, uff.buffer_dict['avel'], np.float32(0), 0, uff.buffer_dict['avel'].size)
    uff.queue.finish()
    save_dir = _debug_dir('invariants')
    tag = mol_file.replace('.xyz', '')

    # Collect per-step data — preallocate download buffers to avoid per-step allocation
    natoms = len(masses)
    apos_buf = np.zeros(natoms * 4, dtype=np.float32)
    vel_buf  = np.zeros(natoms * 4, dtype=np.float32)
    fapos_buf = np.zeros(natoms * 4, dtype=np.float32)
    E_pes, E_kes, E_tots, ps, Ls, traj = [], [], [], [], [], []
    for step in range(nsteps):
        uff.run_eval_step()
        E_pe = uff.get_total_energy(buf=fapos_buf)[0]
        apos = uff.get_positions(buf=apos_buf)
        vel  = uff.get_velocities(buf=vel_buf)
        E_ke = 0.5 * np.sum(masses[:, None] * vel**2)
        E_pes.append(E_pe)
        E_kes.append(E_ke)
        E_tots.append(E_pe + E_ke)
        p = np.sum(masses[:, None] * vel, axis=0)
        ps.append(np.linalg.norm(p))
        com = np.average(apos, axis=0, weights=masses)
        L = np.sum(np.cross(apos - com, masses[:, None] * vel), axis=0)
        Ls.append(np.linalg.norm(L))
        traj.append(apos.copy())
        uff._run_integrator()

    E_pes  = np.array(E_pes)
    E_kes  = np.array(E_kes)
    E_tots = np.array(E_tots)
    ps     = np.array(ps)
    Ls     = np.array(Ls)
    traj   = np.array(traj)
    steps  = np.arange(nsteps)

    # Plot energy components (independent curves, no reference comparison)
    plot_curves(steps, [E_tots, E_pes, E_kes], ['E_total', 'E_pe', 'E_ke'],
                f'{tag} energy invariants', 'step', ylabel='Energy [eV]',
                savepath=os.path.join(save_dir, f'{tag}_energy.png'), pairs=None)
    # Plot linear momentum
    plot_curves(steps, [ps], ['|p|'],
                f'{tag} linear momentum', 'step', ylabel='|p| [amu*A/fs]',
                savepath=os.path.join(save_dir, f'{tag}_p.png'), pairs=None)
    # Plot angular momentum
    plot_curves(steps, [Ls], ['|L|'],
                f'{tag} angular momentum', 'step', ylabel='|L| [amu*A^2/fs]',
                savepath=os.path.join(save_dir, f'{tag}_L.png'), pairs=None)
    # Trajectory plot
    natoms = traj.shape[1]
    fig, axes = plt.subplots(natoms, 1, figsize=(8, 2*natoms), sharex=True)
    if natoms == 1: axes = [axes]
    for ia in range(natoms):
        for d, label in enumerate('xyz'):
            axes[ia].plot(steps, traj[:, ia, d], label=label)
        axes[ia].set_ylabel(f'atom {ia}')
        axes[ia].legend(fontsize=8)
    axes[-1].set_xlabel('step')
    fig.suptitle(f'{tag} atom trajectories')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{tag}_trajectory.png'), dpi=150)
    plt.close(fig)

    dE = E_tots[-1] - E_tots[0]
    dE_rel = dE / abs(E_tots[0]) if abs(E_tots[0]) > 1e-12 else 0.0
    dL = Ls.max()  # max angular momentum magnitude during run
    print(f'{tag}: E0={E_tots[0]:.6e}, E_final={E_tots[-1]:.6e}, dE={dE:.2e} (rel={dE_rel:.2e})')
    print(f'{tag}: |p|_max={ps.max():.2e}, |L|_max={dL:.2e}')
    # Save initial+final XYZ
    save_xyz_frames(os.path.join(save_dir, f'{tag}_init_final.xyz'),
                    mol.enames, [traj[0], traj[-1]],
                    comments=[f'{tag} initial', f'{tag} final dE={dE:.2e}'])
    # Plot initial vs final geometry
    bonds = find_bonds(traj[0], mol.enames)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_geometry(ax1, traj[0], mol.enames, bonds, title=f'{tag} initial', proj='xy')
    plot_geometry(ax2, traj[-1], mol.enames, bonds, title=f'{tag} final', proj='xy')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{tag}_geometry.png'), dpi=150)
    plt.close(fig)
    assert abs(dE) < dE_abs_tol, f'{tag}: energy drift too large: |dE|={abs(dE):.2e}'
    assert dL < dL_tol, f'{tag}: angular momentum not conserved: |L|_max={dL:.2e}'

@pytest.mark.visual
@pytest.mark.gpu
def test_visual_relax_energy(xyz):
    """Plot energy vs step for water relaxation (GPU-side integrator)."""
    from spammm.forcefields.UFF import UFF_CL
    mol = AtomicSystem(fname=xyz('H2O.xyz'))
    apos_init = mol.apos.copy()
    mol.apos = distort(mol.apos, 0.2)
    apos_distort = mol.apos.copy()
    uff = UFF_CL()
    uff.toUFF(mol)
    mass_map = {'H': 1.0, 'C': 12.0, 'O': 16.0, 'N': 14.0}
    masses = np.array([mass_map.get(e, 12.0) for e in mol.enames], dtype=np.float32)
    uff.upload_positions(mol.apos, masses=masses)
    uff.set_md_params(dt=0.01, damp=0.9, Flimit=100.0)
    import pyopencl as cl
    cl.enqueue_fill_buffer(uff.queue, uff.buffer_dict['avel'], np.float32(0), 0, uff.buffer_dict['avel'].size)
    uff.queue.finish()
    save_dir = _debug_dir()
    energies, fmax_arr, traj = [], [], []
    for step in range(100):
        uff.run_eval_step()
        E = uff.get_total_energy()[0]
        forces = uff.get_forces()[0]
        energies.append(E)
        fmax_arr.append(np.max(np.linalg.norm(forces, axis=1)))
        traj.append(uff.get_positions())
        uff._run_integrator()
    apos_final = traj[-1]
    plot_curves(np.arange(100), [energies], ['E_total'],
                'UFF relaxation H2O', 'step', ylabel='Energy [eV]',
                savepath=os.path.join(save_dir, 'relax_energy.png'), pairs=None)
    plot_curves(np.arange(100), [fmax_arr], ['F_max'],
                'UFF relaxation H2O', 'step', ylabel='Max force [eV/A]',
                savepath=os.path.join(save_dir, 'relax_force.png'), pairs=None)
    # Save initial+final XYZ
    save_xyz_frames(os.path.join(save_dir, 'H2O_UFF_init_final.xyz'),
                    mol.enames, [apos_distort, apos_final],
                    comments=['H2O distorted (initial)', 'H2O relaxed (final)'])
    # Plot initial vs final geometry with bond lengths
    bonds = find_bonds(apos_final, mol.enames)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_geometry(ax1, apos_distort, mol.enames, bonds, title='H2O initial (distorted)', proj='xy')
    plot_geometry(ax2, apos_final, mol.enames, bonds, title='H2O final (relaxed)', proj='xy')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'H2O_UFF_geometry.png'), dpi=150)
    plt.close(fig)


# ======================================================================
# Energy-Force Correspondence Tests
# F = -dE/dx : analytic forces must match numerical derivative of energy
# Supports both UFF and SPFF force fields (covalent-only, no non-bonded).
# ======================================================================

def _setup_uff(mol_file):
    """Create UFF eval_fn for a molecule. Returns (eval_fn, pos0, natoms, perm).
    perm=None (UFF does not reorder atoms)."""
    mol = AtomicSystem(fname=f'data/xyz/{mol_file}')
    uff = UFF_CL()
    uff.toUFF(mol)
    uff.bDoNonBonded = False
    uff.args_setup = False
    masses = np.ones(len(mol.apos), dtype=np.float32)
    uff.upload_positions(mol.apos.astype(np.float32), masses=masses)
    pos0 = mol.apos.astype(np.float32).copy()
    def eval_fn(pos):
        uff.upload_positions(pos.astype(np.float32), masses=masses)
        uff.run_eval_step()
        E = uff.get_total_energy()[0]
        F = uff.get_forces()[0].copy()
        return E, F
    return eval_fn, pos0, len(mol.apos), None

def _setup_spff(mol_file):
    """Create SPFF eval_fn for a molecule. Returns (eval_fn, pos0, natoms, perm).
    perm maps original atom indices → SPFF reordered indices."""
    from spammm.forcefields.MolecularDynamics import MolecularDynamics
    from spammm.forcefields.SPFF import SPFF
    from spammm.topology.FFparams import SPFFparams
    import pyopencl as cl
    mol = AtomicSystem(fname=f'data/xyz/{mol_file}')
    params = SPFFparams('data/')
    mol.atypes = np.array([params.getAtomType(e, bErr=False) for e in mol.enames], dtype=np.int32)
    spff = SPFF()
    spff.toSPFFsp3_loc(mol, params.atom_types_map)
    mass_map = {'H': 1.0, 'C': 12.0, 'O': 16.0, 'N': 14.0}
    for ia in range(spff.natoms):
        e = mol.enames[ia]
        spff.apos[ia, 3] = mass_map.get(e, 12.0)
    md = MolecularDynamics(enable_nonbond=False)
    md.realloc(spff, nSystems=1)
    md.upload_all_systems()
    md.setup_kernels()
    cl.enqueue_fill_buffer(md.queue, md.buffer_dict['avel'], np.float32(0), 0, md.buffer_dict['avel'].size)
    md.queue.finish()
    # dt=0 so updateAtomsSPFFf4 assembles recoil forces from fneigh onto fapos without moving atoms
    md.set_md_params(dt=0.0, damp=1.0, Flimit=0.0)
    natoms = spff.natoms
    pos0 = spff.apos[:natoms, :3].copy().astype(np.float32)
    perm = getattr(mol, 'perm_nodes_first', list(range(natoms)))
    def eval_fn(pos):
        spff.apos[:natoms, :3] = pos.astype(np.float32)
        md.toGPU('apos', md._flat32(spff.apos), byte_offset=0)
        md.run_cleanForceSPFFf4()
        md.run_getSPFFf4()
        md.run_updateAtomsSPFFf4()  # assemble recoil forces (dt=0 → no movement)
        E = md.get_total_energy()
        F = md.get_forces()[:, :3].copy()
        return E, F
    return eval_fn, pos0, natoms, perm

def _plot_ef_profile(scan_vals, E, F_analytic, F_numeric, xlabel, title, savepath, F_label='Force [eV/A]'):
    """Plot energy (top) and analytic vs numerical force (bottom)."""
    fig, (axE, axF) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    axE.plot(scan_vals, E, 'b-', lw=1.0)
    axE.set_ylabel('Energy [eV]')
    axE.set_title(title)
    axF.plot(scan_vals, F_analytic, '-', lw=0.5, label='analytic')
    axF.plot(scan_vals, F_numeric, ':', lw=1.5, label='numeric (-dE/dx)')
    axF.set_ylabel(F_label)
    axF.set_xlabel(xlabel)
    axF.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)

def _rodrigues(v, axis, angle):
    """Rotate vector v around unit axis by angle (radians). Preserves |v|."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)

def scan_bond(eval_fn, pos0, i, j, drs, eps=1e-4):
    """Scan bond (i,j): move atom j along bond direction, keeping all else fixed.
    Returns (rs, Es, F_analytic, F_numeric) where F is force on j along bond direction.
    F_analytic = F_j . r_hat,  F_numeric = -dE/dr (central difference).
    """
    rij = pos0[j] - pos0[i]
    r0 = np.linalg.norm(rij)
    r_hat = rij / r0
    Es, Fa, Fn = [], [], []
    for dr in drs:
        pos = pos0.copy(); pos[j] = pos0[j] + r_hat * dr
        E, F = eval_fn(pos)
        Es.append(E); Fa.append(np.dot(F[j], r_hat))
        pos_p = pos0.copy(); pos_p[j] = pos0[j] + r_hat * (dr + eps)
        pos_m = pos0.copy(); pos_m[j] = pos0[j] + r_hat * (dr - eps)
        E_p, _ = eval_fn(pos_p)
        E_m, _ = eval_fn(pos_m)
        Fn.append(-(E_p - E_m) / (2 * eps))
    return r0 + drs, np.array(Es), np.array(Fa), np.array(Fn)

def scan_angle(eval_fn, pos0, i, j, k, dthetas, eps=1e-4):
    """Scan angle (i,j,k): rotate atom k around central atom j in the (i,j,k) plane.
    Bond lengths r_jk and r_ji are preserved. Returns (thetas, Es, F_analytic, F_numeric).
    F_analytic = torque on k = F_k . tang * r_jk,  F_numeric = -dE/dtheta.
    """
    v1 = pos0[i] - pos0[j]  # j->i
    v2 = pos0[k] - pos0[j]  # j->k
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-30)
    theta0 = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30), -1, 1))
    Es, Fa, Fn = [], [], []
    for dth in dthetas:
        v2_new = _rodrigues(v2, normal, dth)
        pos = pos0.copy(); pos[k] = pos0[j] + v2_new
        E, F = eval_fn(pos)
        Es.append(E)
        v2n = v2_new / (np.linalg.norm(v2_new) + 1e-30)
        tang = np.cross(normal, v2n)  # direction of increasing theta
        Fa.append(np.dot(F[k], tang) * np.linalg.norm(v2_new))  # torque = F . r_perp
        v2_p = _rodrigues(v2, normal, dth + eps)
        v2_m = _rodrigues(v2, normal, dth - eps)
        pos_p = pos0.copy(); pos_p[k] = pos0[j] + v2_p
        pos_m = pos0.copy(); pos_m[k] = pos0[j] + v2_m
        E_p, _ = eval_fn(pos_p)
        E_m, _ = eval_fn(pos_m)
        Fn.append(-(E_p - E_m) / (2 * eps))
    return np.degrees(theta0 + dthetas), np.array(Es), np.array(Fa), np.array(Fn)

def scan_dihedral(eval_fn, pos0, i, j, k, l, dphis, eps=1e-4):
    """Scan dihedral (i,j,k,l): rotate atom l around bond (j,k).
    Bond length r_kl is preserved. Returns (phis, Es, F_analytic, F_numeric).
    F_analytic = torque on l = F_l . tang * r_perp,  F_numeric = -dE/dphi.
    """
    axis = pos0[k] - pos0[j]
    axis = axis / np.linalg.norm(axis)
    pivot = pos0[k].copy()
    rl = pos0[l] - pivot
    rl_perp = rl - np.dot(rl, axis) * axis
    rl_perp_len = np.linalg.norm(rl_perp)
    rl_perp_hat = rl_perp / (rl_perp_len + 1e-30)
    e2 = np.cross(axis, rl_perp_hat)
    Es, Fa, Fn = [], [], []
    for dphi in dphis:
        c, s = np.cos(dphi), np.sin(dphi)
        rl_new = rl_perp_len * (rl_perp_hat * c + e2 * s) + np.dot(rl, axis) * axis
        pos = pos0.copy(); pos[l] = pivot + rl_new
        E, F = eval_fn(pos)
        Es.append(E)
        tang = -rl_perp_hat * s + e2 * c
        Fa.append(np.dot(F[l], tang) * rl_perp_len)
        c_p, s_p = np.cos(dphi + eps), np.sin(dphi + eps)
        rl_p = rl_perp_len * (rl_perp_hat * c_p + e2 * s_p) + np.dot(rl, axis) * axis
        c_m, s_m = np.cos(dphi - eps), np.sin(dphi - eps)
        rl_m = rl_perp_len * (rl_perp_hat * c_m + e2 * s_m) + np.dot(rl, axis) * axis
        pos_p = pos0.copy(); pos_p[l] = pivot + rl_p
        pos_m = pos0.copy(); pos_m[l] = pivot + rl_m
        E_p, _ = eval_fn(pos_p)
        E_m, _ = eval_fn(pos_m)
        Fn.append(-(E_p - E_m) / (2 * eps))
    return np.degrees(dphis), np.array(Es), np.array(Fa), np.array(Fn)

def scan_full_distortion(eval_fn, pos0, amplitude=0.05, eps=1e-4, seed=123):
    """Random distortion: check all 3N force components against finite differences.
    Returns (F_analytic[natoms,3], F_numeric[natoms,3], max_err).
    """
    rng = np.random.default_rng(seed)
    pos_dist = pos0 + rng.normal(0, amplitude, pos0.shape).astype(np.float32)
    _, F0 = eval_fn(pos_dist)
    natoms = len(pos0)
    F_num = np.zeros_like(F0)
    for ia in range(natoms):
        for d in range(3):
            pos_p = pos_dist.copy(); pos_p[ia, d] += eps
            pos_m = pos_dist.copy(); pos_m[ia, d] -= eps
            E_p, _ = eval_fn(pos_p)
            E_m, _ = eval_fn(pos_m)
            F_num[ia, d] = -(E_p - E_m) / (2 * eps)
    return F0, F_num, np.max(np.abs(F0 - F_num))

# (ff, mol_file, bond_pair, angle_triple, dihedral_quad)
# Atom indices are in the original XYZ file ordering; mapped to SPFF ordering internally.
# Covalent only (no non-bonded) for ablation — isolates force-energy consistency.
EF_CORR_CASES = [
    ('UFF',  'CH4.xyz',   (0, 1), (1, 0, 2), None),
    ('UFF',  'CH2NH.xyz', (0, 1), (1, 0, 2), (2, 0, 1, 4)),
    ('SPFF', 'CH4.xyz',   (0, 1), (1, 0, 2), None),
    ('SPFF', 'CH2NH.xyz', (0, 1), (1, 0, 2), None),
]

@pytest.mark.visual
@pytest.mark.gpu
@pytest.mark.parametrize('ff, mol_file, bond_ij, angle_ijk, dih_ijkl', EF_CORR_CASES)
def test_ef_correspondence(xyz, ff, mol_file, bond_ij, angle_ijk, dih_ijkl):
    """Energy-force correspondence: F = -dE/dx along internal coordinates.

    Tests both UFF and SPFF force fields with general-purpose scan functions:
      - scan_bond(i, j): stretch bond by moving j along bond axis
      - scan_angle(i, j, k): rotate k around central atom j in (i,j,k) plane
      - scan_dihedral(i, j, k, l): rotate l around bond (j,k) [UFF only]
      - scan_full_distortion: all 3N components vs finite difference

    Covalent-only (no non-bonded) to isolate force-energy consistency.
    For SPFF, pi-orbital directions are held fixed (partial derivative w.r.t. atoms only).
    Atom indices in test cases refer to the XYZ file ordering; SPFF reordering is handled internally.

    Plots saved to debug/<date>_force_energy/: top = energy, bottom = analytic vs numeric force.
    Tolerance: |F_analytic - F_numeric| < 3e-2 eV/A (float32 GPU + FD truncation).
    """
    eps = 1e-4
    tol = 3e-2
    save_dir = _debug_dir('force_energy')
    tag = f"{mol_file.replace('.xyz', '')}_{ff}"
    if ff == 'UFF':
        eval_fn, pos0, natoms, perm = _setup_uff(mol_file)
    elif ff == 'SPFF':
        eval_fn, pos0, natoms, perm = _setup_spff(mol_file)
        # Map atom indices from original XYZ ordering to SPFF reordered ordering
        bond_ij = (perm[bond_ij[0]], perm[bond_ij[1]])
        angle_ijk = (perm[angle_ijk[0]], perm[angle_ijk[1]], perm[angle_ijk[2]])
        if dih_ijkl is not None:
            dih_ijkl = tuple(perm[x] for x in dih_ijkl)
    else:
        raise ValueError(f"Unknown force field: {ff}")
    max_errs = {}

    # 1. Bond stretch
    i, j = bond_ij
    r0 = np.linalg.norm(pos0[j] - pos0[i])
    rs, Es, Fa, Fn = scan_bond(eval_fn, pos0, i, j, np.linspace(-0.3*r0, 0.3*r0, 50), eps)
    max_errs['bond'] = np.max(np.abs(Fa - Fn))
    _plot_ef_profile(rs, Es, Fa, Fn, f'r_{{{i}{j}}} [A]', f'{tag} bond stretch ({i}-{j})',
                     os.path.join(save_dir, f'{tag}_bond_{i}{j}.png'))

    # 2. Angle bend
    i, j, k = angle_ijk
    v1 = pos0[i] - pos0[j]; v2 = pos0[k] - pos0[j]
    theta0 = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1))
    thetas, Es, Fa, Fn = scan_angle(eval_fn, pos0, i, j, k, np.linspace(-0.4, 0.4, 50), eps)
    max_errs['angle'] = np.max(np.abs(Fa - Fn))
    _plot_ef_profile(thetas, Es, Fa, Fn, f'theta_{{{i}{j}{k}}} [deg]', f'{tag} angle bend ({i}-{j}-{k})',
                     os.path.join(save_dir, f'{tag}_angle_{i}{j}{k}.png'), F_label='Torque [eV/rad]')

    # 3. Dihedral rotation (if specified)
    if dih_ijkl is not None:
        i, j, k, l = dih_ijkl
        phis, Es, Fa, Fn = scan_dihedral(eval_fn, pos0, i, j, k, l, np.linspace(-np.pi, np.pi, 100), eps)
        max_errs['dihedral'] = np.max(np.abs(Fa - Fn))
        _plot_ef_profile(phis, Es, Fa, Fn, f'phi_{{{i}{j}{k}{l}}} [deg]', f'{tag} dihedral ({i}-{j}-{k}-{l})',
                         os.path.join(save_dir, f'{tag}_dihedral_{i}{j}{k}{l}.png'), F_label='Torque [eV/rad]')

    # 4. Full random distortion (all 3N components)
    F0, F_num, err_full = scan_full_distortion(eval_fn, pos0, amplitude=0.05, eps=eps)
    max_errs['full_distortion'] = err_full
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(F_num.flatten(), F0.flatten(), s=10, alpha=0.7)
    lim = max(np.abs(F0).max(), np.abs(F_num).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.5)
    ax.set_xlabel('Numeric force -dE/dx [eV/A]')
    ax.set_ylabel('Analytic force [eV/A]')
    ax.set_title(f'{tag} full distortion (max|err|={err_full:.2e})')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{tag}_full_distortion.png'), dpi=150)
    plt.close(fig)

    # Assertions
    print(f'\n{tag} energy-force correspondence (covalent only):')
    for name, err in max_errs.items():
        print(f'  {name}: max|F_analytic - F_numeric| = {err:.2e} eV/A')
    for name, err in max_errs.items():
        assert err < tol, f'{tag} {name}: force-energy mismatch max|err|={err:.2e} > {tol:.0e}'
