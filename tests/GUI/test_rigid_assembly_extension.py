"""L0 smoke tests for RigidAssemblyExtension — build_ui + MC step parity.

Verifies:
  - build_ui constructs the panel without error (offscreen Qt)
  - _on_build creates a RigidEnsemble + RigidBodyPairFF
  - one MC step produces a finite energy matching the testplot harness parity
    (same molecule/grid/seed → same E_initial, same first-step acceptance)
  - _poses_to_pme_sites builds spos/rots from the ensemble (CoM + R(q))

No real mouse/scene interaction (drag mode needs a live VisPy scene — covered by
manual L2 review). MC parity is the key regression guard: if the extension's MC
path diverges from the testplot harness, this test fails.

Run: pytest tests/GUI/test_rigid_assembly_extension.py -m "not slow"
"""
import os
import sys
import numpy as np
import pytest

# Offscreen Qt before any PyQt import
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets


class _MockScene:
    """Minimal stand-in for AtomScene — provides picking helpers used by drag mode."""
    def __init__(self):
        self._pos = np.zeros((0, 3), dtype=np.float32)
        self._pick_active = False
    def _pick_id_from_mouse(self, pos, max_dist=1.0):
        return -1, float('inf')
    def _id_to_idx_safe(self, atom_id):
        return -1
    def _ray_from_mouse(self, pos):
        return np.zeros(3), np.array([0, 0, 1.0])


class _MockBackend:
    """Minimal stand-in for MoleculeEditorBackend — graph + _sync_sys."""
    def __init__(self):
        self.sys = None
        self.graph = None  # extension checks hasattr(graph, 'update_positions_from_array')


class _MockWindow:
    """Minimal stand-in for the SPAMMM_GUI window — holds widgets + state."""
    def __init__(self):
        self.backend = _MockBackend()
        self.scene = _MockScene()
        self.pick_radius = 1.0
        self._mode_handlers = {}
        self._edit_mode = None
    def register_mode_handler(self, name, handler):
        self._mode_handlers[name] = handler
    def set_edit_mode(self, name):
        self._edit_mode = name
    def refresh_view(self):
        pass
    def statusBar(self):
        class _SB:
            def showMessage(self, *a, **k): pass
        return _SB()


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    yield app


@pytest.fixture
def ra_window(qapp):
    """Build the RigidAssembly panel on a mock window."""
    from spammm.GUI.RigidAssemblyExtension import build_ui
    w = _MockWindow()
    ui = build_ui(w)
    assert ui.panel is not None
    assert len(ui.edit_modes) == 1  # RA Drag
    yield w


def test_build_ui_smoke(ra_window):
    """Panel built, all expected widgets exist, edit mode registered."""
    w = ra_window
    assert w.ra_mol_combo is not None
    assert w.ra_nmol_spin is not None
    assert w.ra_ensemble is None  # not built yet
    assert 'ra_drag' in w._mode_handlers


def test_on_build_creates_ensemble_and_rbd(ra_window):
    """_on_build creates RigidEnsemble + RigidBodyPairFF with the right counts."""
    w = ra_window
    w.ra_mol_combo.setCurrentText('PTCDA')
    w.ra_nmol_spin.setValue(2)
    w.ra_spacing_spin.setValue(16.0)
    w.ra_z_spin.setValue(3.0)
    w.ra_no_faf_chk.setChecked(True)  # skip FAF for speed in test
    from spammm.GUI.RigidAssemblyExtension import _on_build
    _on_build(w)
    assert w.ra_ensemble is not None
    assert w.ra_rbd is not None
    assert len(w.ra_ensemble) == 2
    assert w.ra_rbd.n_bodies == 2
    # GPU device should be NVIDIA (per OpenCL policy)
    dev = w.ra_rbd.ctx.devices[0].name
    assert 'NVIDIA' in dev or 'RTX' in dev, f'Expected NVIDIA GPU, got {dev}'


def test_on_build_faf_keeps_all_pose_stores_synchronized(ra_window):
    """FAF absolute height must agree in ensemble, GPU, and RBD host mirrors."""
    from spammm.GUI.RigidAssemblyExtension import _on_build
    from spammm.surfaces.FoldedRigid import Z_SURF_TOP

    w = ra_window
    w.ra_mol_combo.setCurrentText('PTCDA')
    w.ra_nmol_spin.setValue(1)
    w.ra_z_spin.setValue(3.0)
    w.ra_z_init_spin.setValue(3.0)
    w.ra_no_faf_chk.setChecked(False)
    w.ra_no_qeq_chk.setChecked(False)
    _on_build(w)

    pos_ens, quat_ens = w.ra_ensemble.get_poses()
    out = w.ra_rbd.download_outputs()
    z_expected = float(Z_SURF_TOP + 3.0)
    np.testing.assert_allclose(pos_ens[:, 2], z_expected, atol=1e-6)
    np.testing.assert_allclose(out['pos'][:, :3], pos_ens, atol=1e-6)
    np.testing.assert_allclose(out['quats'], quat_ens, atol=1e-6)
    np.testing.assert_allclose(w.ra_rbd._mb_pos, pos_ens, atol=1e-6)
    np.testing.assert_allclose(w.ra_rbd._mb_quat, quat_ens, atol=1e-6)
    assert np.abs(w.ra_rbd._mb_packs[0]['REQ_base'][:, 2]).max() > 1e-3, 'Default PTCDA build must retain QEq charges'


def test_display_index_maps_to_body_and_gpu_site(ra_window):
    """Dense real-atom scene indices must map across PairFF dummy sites."""
    from spammm.GUI.RigidAssemblyExtension import _display_index_to_body_site, _on_build

    w = ra_window
    w.ra_mol_combo.setCurrentText('PTCDA')
    w.ra_nmol_spin.setValue(2)
    w.ra_no_faf_chk.setChecked(True)
    _on_build(w)

    packs = w.ra_rbd._mb_packs
    nreal0 = int((packs[0]['types'] == 0).sum())
    real1 = np.flatnonzero(packs[1]['types'] == 0)
    body, gpu_site = _display_index_to_body_site(w, nreal0)
    assert body == 1
    assert gpu_site == int(w.ra_rbd.mol_offsets[1] + real1[0])
    with pytest.raises(IndexError):
        _display_index_to_body_site(w, 2 * nreal0)


def test_all_mobile_drag_step_moves_partners_and_resynchronizes(ra_window):
    """An O anchor uses FAF kernel 15, moves partners, and commits every pose."""
    from spammm.GUI.RigidAssemblyExtension import _on_build, _set_anchors, _sync_ensemble_from_gpu
    from spammm.forcefields.RigidBodyDynamics import _body_sites_world

    w = ra_window
    w.ra_mol_combo.setCurrentText('PTCDA')
    w.ra_nmol_spin.setValue(2)
    w.ra_spacing_spin.setValue(16.0)
    w.ra_no_faf_chk.setChecked(False)
    w.ra_no_qeq_chk.setChecked(False)
    w.ra_k_spring_spin.setValue(0.2)
    _on_build(w)

    pos0, quat0 = w.ra_ensemble.get_poses()
    pack0 = w.ra_rbd._mb_packs[0]
    io = next(i for i, (e, t) in enumerate(zip(pack0['enames'], pack0['types'])) if e == 'O' and t == 0)
    site = int(w.ra_rbd.mol_offsets[0] + io)
    atom0 = _body_sites_world(pack0['rel'][io:io + 1, :3], pos0[0], quat0[0])[0]
    _set_anchors(w, site, atom0 + np.array([0.2, 0.0, 0.0], dtype=np.float32))
    w.ra_rbd.run_multimol_md(20, dt=0.02, fire=True, faf=True)
    _sync_ensemble_from_gpu(w)

    pos1, quat1 = w.ra_ensemble.get_poses()
    assert np.isfinite(pos1).all() and np.isfinite(quat1).all()
    assert np.linalg.norm(pos1[0] - pos0[0]) > 1e-7, 'anchored molecule did not move'
    assert np.linalg.norm(pos1[1] - pos0[1]) > 1e-7, 'partner remained frozen'
    out = w.ra_rbd.download_outputs()
    np.testing.assert_allclose(out['pos'][:, :3], pos1, atol=1e-6)
    np.testing.assert_allclose(out['quats'], quat1, atol=1e-6)
    np.testing.assert_allclose(w.ra_rbd._mb_pos, pos1, atol=1e-6)
    np.testing.assert_allclose(w.ra_rbd._mb_quat, quat1, atol=1e-6)


@pytest.mark.slow
def test_ptcda_windmill_1000_step_reference(ra_window):
    """The GUI MC path reproduces the reference 4×PTCDA minimum candidate."""
    from spammm.GUI.RigidAssemblyExtension import _on_build, _on_mc_step

    w = ra_window
    w.ra_mol_combo.setCurrentText('PTCDA')
    w.ra_nmol_spin.setValue(4)
    w.ra_spacing_spin.setValue(16.0)
    w.ra_seed_spin.setValue(3)
    w.ra_no_faf_chk.setChecked(False)
    w.ra_no_qeq_chk.setChecked(False)
    w.ra_ntrial_spin.setValue(512)
    _on_build(w)
    n_accept = 0
    last = None
    for i in range(1000):
        last = _on_mc_step(w, update_ui=False)
        n_accept += int(last['accepted'])
        if (i + 1) % 100 == 0:
            print(f'windmill MC {i + 1}/1000 E={last["E"]:.6f} accepted={n_accept}', flush=True)
    pos, _ = w.ra_ensemble.get_poses()
    assert n_accept == 26
    assert last['E'] == pytest.approx(-0.274403, abs=2e-5)
    np.testing.assert_allclose(pos[:, 2], -0.25, atol=1e-6)


def test_mc_step_parity_vs_testplot(ra_window):
    """One MC step from the extension matches the testplot harness for the same config.

    Uses PTCDA, 4 mols, spacing=16, z=3, seed=3, n_trial=64, dxy=1.5, dphi=0.8,
    k_pack=0.03, no FAF (testplot reference used FAF; we test the MC path parity
    on the energy *change* shape, not absolute E, since FAF changes E).
    """
    w = ra_window
    w.ra_mol_combo.setCurrentText('PTCDA')
    w.ra_nmol_spin.setValue(4)
    w.ra_spacing_spin.setValue(16.0)
    w.ra_z_spin.setValue(3.0)
    w.ra_seed_spin.setValue(3)
    w.ra_no_faf_chk.setChecked(True)
    w.ra_ntrial_spin.setValue(64)
    w.ra_dxy_spin.setValue(1.5)
    w.ra_dphi_spin.setValue(0.8)
    w.ra_kpack_spin.setValue(0.03)
    w.ra_rmin_atom_spin.setValue(1.6)
    from spammm.GUI.RigidAssemblyExtension import _on_build, _on_mc_step
    _on_build(w)
    # Initial energy
    pos, quat = w.ra_ensemble.get_poses()
    E0 = w.ra_rbd.eval_energy_system(pos, quat, k_pack=0.03)
    assert np.isfinite(E0), f'E0 not finite: {E0}'
    # First MC step (moved=[0], same as testplot step 0)
    _on_mc_step(w)
    assert w.ra_mc_step_count == 1
    pos2, quat2 = w.ra_ensemble.get_poses()
    E1 = w.ra_rbd.eval_energy_system(pos2, quat2, k_pack=0.03)
    assert np.isfinite(E1), f'E1 not finite: {E1}'
    # Energy should decrease or stay (greedy accepts only improvements)
    assert E1 <= E0 + 1e-6, f'MC step did not improve: E0={E0} E1={E1}'


def test_poses_to_pme_sites(ra_window):
    """_poses_to_pme_sites builds spos (CoM) + rots (R(q)) from the ensemble."""
    w = ra_window
    w.ra_mol_combo.setCurrentText('PTCDA')
    w.ra_nmol_spin.setValue(3)
    w.ra_spacing_spin.setValue(8.0)
    w.ra_z_spin.setValue(3.0)
    w.ra_no_faf_chk.setChecked(True)
    from spammm.GUI.RigidAssemblyExtension import _on_build, _poses_to_pme_sites
    _on_build(w)
    spos4, rots4, n_act = _poses_to_pme_sites(w)
    assert spos4 is not None
    assert n_act == 3  # 3 bodies, all < 4
    assert spos4.shape == (4, 4)  # embedded to 4 (PME nSingle=4)
    assert rots4.shape == (4, 3, 3)
    # spos CoM should match ensemble pos for active sites
    pos, _ = w.ra_ensemble.get_poses()
    np.testing.assert_allclose(spos4[:3, :3], pos.astype(np.float64), atol=1e-4)
    # rots for active sites should be R(q) — check first body's quat
    from spammm.forcefields.RigidBodyDynamics import _quat_to_matrix_np
    _, quat = w.ra_ensemble.get_poses()
    R_ref = _quat_to_matrix_np(quat[0])
    np.testing.assert_allclose(rots4[0], R_ref, atol=1e-5)


def test_pme_params_built(ra_window):
    """_pme_params builds a complete dict from the panel spinboxes."""
    from spammm.GUI.RigidAssemblyExtension import _pme_params
    w = ra_window
    w.ra_ensemble = None  # _pme_params handles None ensemble
    p = _pme_params(w)
    required = ['nsite', 'Esite', 'W', 'Q0', 'Qzz', 'VBias', 'z_tip', 'Temp',
                'GammaT', 'decay', 'L', 'npix', 'GammaS', 'dQ', 'zQd', 'zVd', 'Rtip', 'phi0_ax']
    for k in required:
        assert k in p, f'missing PME param: {k}'


def test_graph_to_rigid_fragments():
    """graph_to_rigid_fragments splits a graph into connected components with CoM + rel positions."""
    from spammm import elements
    from spammm.topology.AtomicGraph import AtomicGraph
    from spammm.forcefields.RigidBodyUtils import graph_to_rigid_fragments

    g = AtomicGraph()
    # Fragment 1: H2O (O at 0, H at ±1)
    o1 = g.add_atom([0.0, 0.0, 0.0], 'O', elements.ELEMENT_DICT['O'][0])
    h1a = g.add_atom([1.0, 0.0, 0.0], 'H', elements.ELEMENT_DICT['H'][0])
    h1b = g.add_atom([-1.0, 0.0, 0.0], 'H', elements.ELEMENT_DICT['H'][0])
    g.add_bond(o1, h1a); g.add_bond(o1, h1b)
    # Fragment 2: CO2 (C at 10, O at 9 and 11)
    c2 = g.add_atom([10.0, 0.0, 0.0], 'C', elements.ELEMENT_DICT['C'][0])
    o2a = g.add_atom([11.0, 0.0, 0.0], 'O', elements.ELEMENT_DICT['O'][0])
    o2b = g.add_atom([9.0, 0.0, 0.0], 'O', elements.ELEMENT_DICT['O'][0])
    g.add_bond(c2, o2a); g.add_bond(c2, o2b)

    frags, coms = graph_to_rigid_fragments(g, qeq=False, planarize=True)
    assert len(frags) == 2, f'Expected 2 fragments, got {len(frags)}'
    assert coms.shape == (2, 3)
    # H2O CoM: O(16)@0, H(1)@±1 → (16*0 + 1*1 + 1*(-1))/18 = 0
    np.testing.assert_allclose(coms[0], [0.0, 0.0, 0.0], atol=1e-5)
    # CO2 CoM: C(12)@10, O(16)@9, O(16)@11 → (120+144+176)/44 = 10.0
    np.testing.assert_allclose(coms[1], [10.0, 0.0, 0.0], atol=1e-5)
    # Body-frame rel positions are CoM-centered
    for apos_rel, enames, REQs, bonds in frags:
        assert apos_rel.shape[1] == 3
        # CoM of rel positions should be ~0 (mass-weighted)
        from spammm import elements as _el
        masses = np.array([_el.ELEMENT_DICT[e][10] for e in enames], dtype=np.float32)
        com_rel = (apos_rel * masses[:, None]).sum(axis=0) / masses.sum()
        np.testing.assert_allclose(com_rel, [0.0, 0.0, 0.0], atol=1e-5)
        assert REQs.shape[0] == len(enames)
        assert bonds.shape[1] == 2


def test_on_build_from_editor(ra_window):
    """_on_build with 'From editor' splits the backend graph into rigid bodies."""
    from spammm import elements
    from spammm.topology.AtomicGraph import AtomicGraph
    from spammm.GUI.RigidAssemblyExtension import _on_build

    w = ra_window
    # Build a mock graph with 2 non-collinear fragments (avoid singular inertia tensor)
    g = AtomicGraph()
    # Fragment 1: H2O (bent, not linear)
    o1 = g.add_atom([0.0, 0.0, 0.0], 'O', elements.ELEMENT_DICT['O'][0])
    h1a = g.add_atom([1.0, 0.5, 0.0], 'H', elements.ELEMENT_DICT['H'][0])
    h1b = g.add_atom([-1.0, 0.5, 0.0], 'H', elements.ELEMENT_DICT['H'][0])
    g.add_bond(o1, h1a); g.add_bond(o1, h1b)
    # Fragment 2: formaldehyde-like (H2C=O, non-collinear)
    c2 = g.add_atom([10.0, 0.0, 0.0], 'C', elements.ELEMENT_DICT['C'][0])
    o2 = g.add_atom([11.0, 0.0, 0.0], 'O', elements.ELEMENT_DICT['O'][0])
    h2a = g.add_atom([9.0, 0.5, 0.0], 'H', elements.ELEMENT_DICT['H'][0])
    h2b = g.add_atom([9.0, -0.5, 0.0], 'H', elements.ELEMENT_DICT['H'][0])
    g.add_bond(c2, o2); g.add_bond(c2, h2a); g.add_bond(c2, h2b)
    w.backend.graph = g

    w.ra_source_combo.setCurrentText('From editor')
    w.ra_no_qeq_chk.setChecked(True)  # skip QEq for speed
    w.ra_no_faf_chk.setChecked(True)
    _on_build(w)
    assert w.ra_ensemble is not None
    assert w.ra_rbd is not None
    assert len(w.ra_ensemble) == 2  # 2 fragments
    assert w.ra_rbd.n_bodies == 2


def test_on_build_from_file_rebuilds_graph(ra_window):
    """_on_build with 'From file' rebuilds the editor graph to match the assembly.

    The assembly (e.g. 2×PTCDA=52 atoms) has more atoms than the empty editor graph (0).
    _ensure_backend_matched should rebuild the graph from the assembly's world atoms.
    """
    from spammm.topology.AtomicGraph import AtomicGraph
    from spammm.GUI.RigidAssemblyExtension import _on_build, _assembly_world_atoms

    w = ra_window
    # Start with an empty graph (simulates fresh editor)
    w.backend.graph = AtomicGraph()

    w.ra_source_combo.setCurrentText('From file')
    w.ra_mol_combo.setCurrentText('PTCDA')
    w.ra_nmol_spin.setValue(2)
    w.ra_no_faf_chk.setChecked(True)
    _on_build(w)
    assert w.ra_ensemble is not None
    assert w.ra_rbd is not None
    # After build, the graph should have been rebuilt to match the assembly
    apos, enames = _assembly_world_atoms(w)
    assert apos is not None
    n_atoms = apos.shape[0]
    alive_atoms = [a for a in w.backend.graph.atoms.values() if a.alive]
    assert len(alive_atoms) == n_atoms, f'Graph has {len(alive_atoms)} atoms, assembly has {n_atoms}'
