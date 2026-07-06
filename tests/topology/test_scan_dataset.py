"""L0 tests for ScanDataset, coordinate_scan frame builder, Kekulé C–C analysis."""
import os
import numpy as np
import pytest

from spammm.quantum.hbond_scan import build_ascii_hbond_system, identify_hbond_from_ascii
from spammm.topology.hbond_utils import find_hbonds_sys, default_mapping, controls_to_fractions, HbondRecord
from spammm.topology.scan_dataset import ScanDataset, bond_lengths, cc_bond_indices
from spammm.quantum.coordinate_scan import build_control_grid, build_frame, interpolate_all_atoms, dataset_from_frames
from spammm.topology.scan_kekule import cc_length_vs_control


def test_build_control_grid_1d():
    g = build_control_grid([(0.0, 1.0)], dx=0.25)
    assert g.shape[1] == 1
    assert abs(g[0, 0]) < 1e-8
    assert abs(g[-1, 0] - 1.0) < 1e-8


def test_controls_to_fractions_shared():
    mapping = default_mapping(2, m=1)
    f = controls_to_fractions([0.4], mapping)
    assert np.allclose(f, [0.4, 0.4])


def test_single_hbond_scan_mapping_length():
    """One selected H-bond uses mapping length 1 (not padded to all detected)."""
    mapping = default_mapping(1, m=1)
    assert mapping == [0]
    hb = HbondRecord(0, 1, 2)
    f = controls_to_fractions([0.5], mapping)
    assert len(f) == 1
    apo0 = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    apo = build_frame(apo0, [hb], [0.5], mapping)
    assert apo.shape == apo0.shape


def test_build_frame_moves_h_only():
    atoms = build_ascii_hbond_system('2Quinolone')
    atoms.neighs()
    hbonds = find_hbonds_sys(atoms, bPrint=False)
    assert len(hbonds) >= 1
    mapping = default_mapping(len(hbonds), m=1)
    apo0 = atoms.apos.copy()
    apo1 = build_frame(apo0, [hbonds[0]], [0.0], mapping)
    apo2 = build_frame(apo0, [hbonds[0]], [1.0], mapping)
    ih = hbonds[0].h_idx
    assert not np.allclose(apo1[ih], apo2[ih])
    heavy = [i for i in range(atoms.natoms) if i != ih]
    assert np.allclose(apo1[heavy], apo0[heavy])


def test_interpolate_all_atoms():
    a0 = np.array([[0., 0., 0.], [1., 0., 0.]])
    a1 = np.array([[0., 1., 0.], [1., 1., 0.]])
    path = interpolate_all_atoms(a0, a1, [0.0, 0.5, 1.0])
    assert path.shape == (3, 2, 3)
    assert np.allclose(path[0], a0)
    assert np.allclose(path[-1], a1)
    assert np.allclose(path[1], 0.5 * (a0 + a1))


def test_scan_dataset_roundtrip(tmp_path):
    atoms = build_ascii_hbond_system('2Quinolone')
    bonds = np.asarray(atoms.bonds, dtype=np.int32)
    etype = np.array([__import__('spammm.elements', fromlist=['ELEMENT_DICT']).ELEMENT_DICT[e][0] for e in atoms.enames], dtype=np.int32)
    apos0 = atoms.apos
    hb = find_hbonds_sys(atoms, bPrint=False)[0]
    mapping = [0]
    frames = np.array([build_frame(apos0, [hb], [u], mapping) for u in [0.0, 0.5, 1.0]])
    controls = np.array([[0.0], [0.5], [1.0]])
    ds = dataset_from_frames(etype, bonds, np.arange(len(etype)), frames, controls, np.array([0.0, 0.1, 0.2]), meta={'scan_type': 'preview'})
    p = tmp_path / 'scan.npz'
    ds.save_npz(str(p))
    ds2 = ScanDataset.load_npz(str(p))
    assert ds2.nframes == 3
    assert np.allclose(ds2.bond_len, bond_lengths(ds2.apos, ds2.bonds))
    cc = cc_bond_indices(ds2.etype, ds2.bonds)
    assert len(cc) > 0


def test_scan_dataset_charges_roundtrip(tmp_path):
    atoms = build_ascii_hbond_system('2Quinolone')
    bonds = np.asarray(atoms.bonds, dtype=np.int32)
    etype = np.array([__import__('spammm.elements', fromlist=['ELEMENT_DICT']).ELEMENT_DICT[e][0] for e in atoms.enames], dtype=np.int32)
    nframes, natoms = 3, len(etype)
    frames = np.tile(atoms.apos, (nframes, 1, 1))
    controls = np.array([[0.0], [0.5], [1.0]])
    charges = np.random.randn(nframes, natoms) * 0.1
    charges -= charges.sum(axis=1, keepdims=True)
    ds = dataset_from_frames(etype, bonds, np.arange(natoms), frames, controls, np.full(nframes, np.nan), meta={'charge_type': 'mulliken'}, charges=charges)
    p = tmp_path / 'scan_q.npz'
    ds.save_npz(str(p))
    ds2 = ScanDataset.load_npz(str(p))
    assert ds2.charges is not None
    assert np.allclose(ds2.charges, charges)


def test_esp_stack_shape():
    from spammm.quantum.esp_grid import compute_esp_stack
    apos = np.array([[[0., 0., 0.], [1., 0., 0.]], [[0., 0.1, 0.], [1.1, 0., 0.]]])
    q = np.array([[0.2, -0.2], [0.3, -0.3]])
    stack, extent, nx, ny, z_abs = compute_esp_stack(apos, q, z_height=1.0, n=32)
    assert stack.shape == (2, ny, nx)
    assert len(extent) == 4
    assert z_abs > 0


def test_parse_mulliken_charges():
    from spammm.quantum.DFTB_utils import parse_mulliken_charges
    p = '/home/prokophapala/git/SPAMMM/debug/rc_scan/endpoint_u0/detailed.out'
    import os
    if not os.path.isfile(p):
        pytest.skip('no DFTB detailed.out reference')
    q = parse_mulliken_charges(p, natoms=24)
    assert q.shape == (24,)
    assert abs(q.sum()) < 0.01


def test_cc_length_vs_control():
    atoms = build_ascii_hbond_system('2Quinolone')
    bonds = np.asarray(atoms.bonds, dtype=np.int32)
    hb = find_hbonds_sys(atoms, bPrint=False)[0]
    frames = np.array([build_frame(atoms.apos, [hb], [u], [0]) for u in np.linspace(0, 1, 5)])
    controls = np.linspace(0, 1, 5)[:, np.newaxis]
    etype = np.array([__import__('spammm.elements', fromlist=['ELEMENT_DICT']).ELEMENT_DICT[e][0] for e in atoms.enames], dtype=np.int32)
    ds = dataset_from_frames(etype, bonds, np.arange(atoms.natoms), frames, controls, np.full(5, np.nan), meta={})
    u, bl, cc_idx = cc_length_vs_control(ds)
    assert len(u) == 5
    assert bl.shape[1] == len(cc_idx)


def test_symmetric_dual_h_endpoints():
    """Both H-bonds move together at u=0 vs u=1 (2Quinolone dimer)."""
    from spammm.quantum.hbond_scan import build_ascii_hbond_system
    from spammm.topology.hbond_utils import find_hbonds_sys, default_mapping
    from spammm.quantum.coordinate_scan import build_pm_neb_endpoints
    atoms = build_ascii_hbond_system('2Quinolone')
    atoms.neighs()
    hbonds = find_hbonds_sys(atoms, bPrint=False)
    assert len(hbonds) >= 2
    mapping = default_mapping(len(hbonds), m=1)
    a0, a1 = build_pm_neb_endpoints(atoms.apos, hbonds, mapping)
    h_idx = [hb.h_idx for hb in hbonds]
    heavy = [i for i in range(atoms.natoms) if i not in h_idx]
    assert not np.allclose(a0[h_idx], a1[h_idx])
    assert np.allclose(a0[heavy], a1[heavy])


@pytest.mark.slow
def test_pm_neb_relaxed_dftb(tmp_path):
    """DFTB relax both isomers + interpolate (2Quinolone, all H-bonds)."""
    from spammm.quantum.hbond_scan import build_ascii_hbond_system
    from spammm.topology.hbond_utils import find_hbonds_sys, default_mapping
    from spammm.quantum.coordinate_scan import run_pm_neb
    from spammm import elements as el
    atoms = build_ascii_hbond_system('2Quinolone')
    atoms.neighs()
    hbonds = find_hbonds_sys(atoms, bPrint=False)
    mapping = default_mapping(len(hbonds), m=1)
    bonds = np.asarray(atoms.bonds, dtype=np.int32)
    etype = np.array([el.ELEMENT_DICT[e][0] for e in atoms.enames], dtype=np.int32)
    ds = run_pm_neb(atoms.enames, atoms.apos, hbonds, mapping, dx=0.5, relax_endpoints=True, run_sp=False, etype=etype, bonds=bonds, work_dir=str(tmp_path / 'pm_relax'), verbose=True, on_fail='skip')
    assert ds.meta.get('endpoints_relaxed')
    assert ds.nframes >= 3
    assert ds.meta.get('scan_type') == 'pm_neb_relaxed'
    assert ds.charges is not None and ds.charges.shape == (ds.nframes, ds.natoms)
    assert np.allclose(ds.charges.sum(axis=1), 0.0, atol=0.05)
    bl0 = ds.bond_len[0]
    blmid = ds.bond_len[ds.nframes // 2]
    assert not np.allclose(bl0, blmid)

    """Kekulé pi BO on interpolated frames (no DFTB) — smoke for scan_kekule."""
    from spammm.topology.scan_kekule import analyze_kekule_cc
    atoms = build_ascii_hbond_system('2Quinolone')
    bonds = np.asarray(atoms.bonds, dtype=np.int32)
    ih, ido, iac = identify_hbond_from_ascii(atoms, 0)
    hb = HbondRecord(ido, ih, iac)
    a0 = build_frame(atoms.apos, [hb], [0.0], [0])
    a1 = build_frame(atoms.apos, [hb], [1.0], [0])
    stack = interpolate_all_atoms(a0, a1, [0.0, 1.0])
    controls = np.array([[0.0], [1.0]])
    from spammm import elements as el
    etype = np.array([el.ELEMENT_DICT[e][0] for e in atoms.enames], dtype=np.int32)
    ds = dataset_from_frames(etype, bonds, np.arange(atoms.natoms), stack, controls, np.full(2, np.nan), meta={'scan_type': 'pm_neb_preview'})
    cc_idx, pi_bo = analyze_kekule_cc(ds, frame_stride=1, kval=50.0, kloc=5.0)
    assert pi_bo is not None
    assert pi_bo.shape == (2, len(cc_idx))
    assert np.any(np.isfinite(pi_bo))
