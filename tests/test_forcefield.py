import pytest, numpy as np, os, datetime
from spammm.AtomicSystem import AtomicSystem
from tests.helpers.geometry import distort, bond_lengths, bond_angle, assert_geometry
from tests.helpers.parity import overlay_plot, rmse, correlation

def _debug_dir(name='forcefield'):
    d = os.path.join('debug', f'{datetime.date.today()}_{name}')
    os.makedirs(d, exist_ok=True)
    return d

# (molecule, forcefield, nsteps, expected_bonds: {(i,j): (r0, tol_frac)})
RELAX_CASES = [
    ('H2O.xyz',     'UFF',  100, {(0,1): (0.96, 0.15), (0,2): (0.96, 0.15)}),
    ('CH4.xyz',     'UFF',  100, {(0,1): (1.09, 0.15), (0,2): (1.09, 0.15), (0,3): (1.09, 0.15)}),
    ('benzene.xyz', 'UFF',  200, {(0,1): (1.40, 0.10)}),
    ('H2O.xyz',     'SPFF', 100, {(0,1): (0.96, 0.15), (0,2): (0.96, 0.15)}),
    ('benzene.xyz', 'SPFF', 200, {(0,1): (1.40, 0.10)}),
]

@pytest.mark.gpu
@pytest.mark.parametrize('mol_file,ff,nsteps,expected', RELAX_CASES)
def test_relax(xyz, mol_file, ff, nsteps, expected):
    """Distort molecule, relax with UFF or SPFF, check geometry converges."""
    mol = AtomicSystem(fname=xyz(mol_file))
    mol.apos = distort(mol.apos, 0.2)
    if ff == 'UFF':
        from spammm.forcefields.UFF import UFF_CL
        uff = UFF_CL()
        uff.toUFF(mol)
        uff.upload_positions(mol.apos)
        for _ in range(nsteps):
            uff.run_eval_step()
        forces = uff.get_forces()
        fmax = np.max(np.linalg.norm(forces, axis=1))
        assert fmax < 1.0, f'{mol_file} {ff}: force too large after relax: {fmax:.2f}'
    elif ff == 'SPFF':
        from spammm.forcefields.MolecularDynamics import MolecularDynamics
        from spammm.forcefields.SPFF import SPFF
        # TODO: setup SPFF topology + MD, run relax
        pass
    assert_geometry(mol.apos, expected, name=f'{mol_file}_{ff}')

@pytest.mark.gpu
def test_uff_energy_finite(xyz):
    """UFF single-point energy must be finite and negative for stable molecule."""
    from spammm.forcefields.UFF import UFF_CL
    mol = AtomicSystem(fname=xyz('H2O.xyz'))
    uff = UFF_CL()
    uff.toUFF(mol)
    uff.upload_positions(mol.apos)
    uff.run_eval_step()
    E = uff.get_total_energy()
    assert np.all(np.isfinite(E)), 'UFF energy is NaN/Inf'
    assert E[0] < 0, f'UFF energy should be negative for stable molecule: {E[0]}'

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
    net_force = np.sum(forces, axis=0)
    assert np.linalg.norm(net_force) < 0.1, f'Net force not zero: {net_force}'

@pytest.mark.gpu
def test_nve_conservation(xyz):
    """Energy conservation: methane NVE 500 steps, dE/E < 1e-3."""
    from spammm.forcefields.UFF import UFF_CL
    mol = AtomicSystem(fname=xyz('CH4.xyz'))
    uff = UFF_CL()
    uff.toUFF(mol)
    apos = mol.apos.copy().astype(np.float32)
    vel = np.zeros_like(apos, dtype=np.float32)
    dt = 0.5  # fs
    mass = 12.0  # approximate
    uff.upload_positions(apos)
    energies = []
    for step in range(500):
        uff.run_eval_step()
        forces = uff.get_forces()
        E = uff.get_total_energy()[0]
        KE = 0.5 * mass * np.sum(vel**2)
        energies.append(E + KE)
        vel += forces * dt / mass
        apos += vel * dt
        uff.upload_positions(apos)
    energies = np.array(energies)
    dE = (energies[-1] - energies[0]) / abs(energies[0])
    assert abs(dE) < 1e-3, f'Energy drift too large: dE/E={dE:.2e}'

@pytest.mark.visual
@pytest.mark.gpu
def test_visual_relax_energy(xyz):
    """Plot energy vs step for water relaxation."""
    from spammm.forcefields.UFF import UFF_CL
    mol = AtomicSystem(fname=xyz('H2O.xyz'))
    mol.apos = distort(mol.apos, 0.2)
    uff = UFF_CL()
    uff.toUFF(mol)
    uff.upload_positions(mol.apos)
    save_dir = _debug_dir()
    energies, fmax_arr = [], []
    for step in range(100):
        uff.run_eval_step()
        E = uff.get_total_energy()[0]
        forces = uff.get_forces()
        energies.append(E)
        fmax_arr.append(np.max(np.linalg.norm(forces, axis=1)))
    overlay_plot(np.arange(100), [energies], ['E_total'],
                 'UFF relaxation H2O', 'step', ylabel='Energy [eV]',
                 savepath=os.path.join(save_dir, 'relax_energy.png'), show_rmse=False)
    overlay_plot(np.arange(100), [fmax_arr], ['F_max'],
                 'UFF relaxation H2O', 'step', ylabel='Max force [eV/A]',
                 savepath=os.path.join(save_dir, 'relax_force.png'), show_rmse=False)
