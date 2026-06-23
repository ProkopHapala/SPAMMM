import pytest, numpy as np
from spammm.AtomicSystem import AtomicSystem
from spammm.topology.FFparams import read_element_types, read_atom_types, make_REQs_from_enames
from tests.helpers.geometry import bond_lengths, bond_angle, planarity
from tests.helpers.parity import assert_parity

# (filename, expected_nbonds, expected_types: {element: atom_type_name})
CASES = [
    ('H2O.xyz',     2, {'O': 'O_3', 'H': 'H'}),
    ('CH4.xyz',     4, {'C': 'C_3', 'H': 'H'}),
    ('benzene.xyz', 12, {'C': 'C_R', 'H': 'H'}),
    ('HCOOH.xyz',   4, {'C': 'C_2', 'O': 'O_2'}),
    ('CO.xyz',      1, {'C': 'C_1', 'O': 'O_1'}),
]

@pytest.mark.parametrize('fname,nbonds_exp,types_exp', CASES)
def test_bond_detection(xyz, fname, nbonds_exp, types_exp):
    mol = AtomicSystem(fname=xyz(fname))
    mol.findBonds()
    assert mol.natoms > 0
    assert len(mol.bonds) == nbonds_exp, f'{fname}: {len(mol.bonds)} != {nbonds_exp}'

@pytest.mark.parametrize('fname,nbonds_exp,types_exp', CASES)
def test_atom_type_assignment(xyz, dat, fname, nbonds_exp, types_exp):
    etypes = read_element_types(dat('ElementTypes.dat'))
    at = read_atom_types(dat('AtomTypes.dat'), etypes)
    mol = AtomicSystem(fname=xyz(fname))
    REQs = make_REQs_from_enames(mol.enames, mol.qs, at)
    assert np.all(np.isfinite(REQs)), f'{fname}: NaN in REQs'
    for ename, expected_type in types_exp.items():
        assert expected_type in at, f'{fname}: type {expected_type} not in AtomTypes.dat'

@pytest.mark.parametrize('fname,nbonds_exp,types_exp', CASES)
def test_neighbor_consistency(xyz, fname, nbonds_exp, types_exp):
    mol = AtomicSystem(fname=xyz(fname))
    mol.findBonds()
    ngs = mol.neighs()
    assert len(ngs) == mol.natoms
    total_neighbors = sum(len(ng) for ng in ngs)
    assert total_neighbors == 2 * len(mol.bonds), f'{fname}: neighbor count != 2*nbonds'

def test_water_geometry(xyz):
    mol = AtomicSystem(fname=xyz('H2O.xyz'))
    mol.findBonds()
    bls = bond_lengths(mol.apos, mol.bonds)
    assert all(0.85 < r < 1.10 for r in bls), f'O-H bonds out of range: {bls}'
    ang = bond_angle(mol.apos, 1, 0, 2)
    assert 95 < ang < 115, f'H-O-H angle out of range: {ang}'

def test_benzene_geometry(xyz):
    mol = AtomicSystem(fname=xyz('benzene.xyz'))
    mol.findBonds()
    bls = bond_lengths(mol.apos, mol.bonds)
    cc_bonds = [r for r, (a, b) in zip(bls, mol.bonds) if mol.enames[a] == 'C' and mol.enames[b] == 'C']
    ch_bonds = [r for r, (a, b) in zip(bls, mol.bonds) if mol.enames[a] == 'H' or mol.enames[b] == 'H']
    assert all(1.25 < r < 1.55 for r in cc_bonds), f'C-C bonds out of range: {cc_bonds}'
    assert all(0.95 < r < 1.15 for r in ch_bonds), f'C-H bonds out of range: {ch_bonds}'
    c_idx = [i for i, e in enumerate(mol.enames) if e == 'C']
    plan = planarity(mol.apos, c_idx)
    assert plan < 0.1, f'Benzene not planar: {plan}'

def test_methane_geometry(xyz):
    mol = AtomicSystem(fname=xyz('CH4.xyz'))
    mol.findBonds()
    bls = bond_lengths(mol.apos, mol.bonds)
    assert all(0.95 < r < 1.20 for r in bls), f'C-H bonds out of range: {bls}'
    for i in range(1, 4):
        ang = bond_angle(mol.apos, i, 0, 1 if i != 1 else 2)
        assert 100 < ang < 120, f'H-C-H angle out of range: {ang}'
