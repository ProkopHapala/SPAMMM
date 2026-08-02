"""L0 tests for per-body state semantics (dynamic/static/deleted) in kernel 15 + 14.

Verifies (per doc/Tasks/RigidAssembly_StaticMols_PotentialMap.md §6 L0):
  1. All-dynamic regression: body_state=[1,1,1] matches default (no state buffer set).
  2. Frozen invariant: static body pos/qrot unchanged, velocity/FIRE exactly zero.
  3. Static interaction: dynamic body's force differs with a static partner vs deleted.
  4. Deletion parity: 3-body with body 1 deleted matches 2-body reference for live bodies.
  5. Kernel 14 deletion parity: MC energy with deleted body matches rebuilt reference.
  6. Mixed-species build + FAF step: 4 species, factorized PLQH, finite step.
  7. Combined map decomposition: E_sum == E_pairff + E_faf, invalidation on state/probe change.

Uses HCOOH (small, fast) for kernel tests. Mixed-species test uses NTCDI+TBTAP+uracil+benzoic_acid.
No FAF for kernel 15 tests — pure PairFF. FAF included for mixed-species and map tests.

See: doc/Reports/StaticObstacle_DragDemo_2026-08-03.md
"""
import os
import pytest
import numpy as np

from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF
from spammm.forcefields.RigidBodyUtils import load_molecule

_HCOOH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'xyz', 'HCOOH.xyz')


def _build_hcooh(n, spacing=5.0):
    """Build n HCOOH molecules on a grid, all dynamic."""
    apos, enames, REQs, _ = load_molecule(_HCOOH, qeq=False, name='formic_acid')
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        pos[i] = [i * spacing, 0.0, 0.0]
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n, 1))
    rbd = RigidBodyPairFF.from_molecules([(apos, enames, REQs)] * n, pos, quats=quat)
    return rbd


def _reset(rbd, pos, quat):
    zero = np.zeros((rbd.n_bodies, 4), dtype=np.float32)
    pos4 = np.zeros((rbd.n_bodies, 4), dtype=np.float32)
    pos4[:, :3] = pos
    pos4[:, 3] = 1.0
    rbd.toGPU('poss', pos4)
    rbd.toGPU('qrots', quat)
    rbd.toGPU('vposs', zero)
    rbd.toGPU('vrots', zero)
    rbd.toGPU('fire_state', zero)
    rbd.queue.finish()


def _state(rbd):
    return rbd.download_selected(('pos', 'quats', 'lin_mom', 'ang_mom', 'body_force', 'body_torque'))


# ─── L0-1: All-dynamic regression ──────────────────────────────────────────
@pytest.mark.gpu
def test_all_dynamic_matches_default():
    """body_state=[1,1,1] must match default (no set_body_states call)."""
    n = 3
    pos = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n, 1))

    rbd_ref = _build_hcooh(n)
    _reset(rbd_ref, pos, quat)
    rbd_ref.run_multimol_md(5, dt=0.01, lin_damp=1.0, ang_damp=1.0)
    ref = _state(rbd_ref)

    rbd_st = _build_hcooh(n)
    _reset(rbd_st, pos, quat)
    rbd_st.set_body_states(np.ones(n, dtype=np.int32))
    rbd_st.run_multimol_md(5, dt=0.01, lin_damp=1.0, ang_damp=1.0)
    st = _state(rbd_st)

    for name in ref:
        np.testing.assert_allclose(st[name], ref[name], rtol=2e-6, atol=2e-6,
                                   err_msg=f"all-dynamic {name} regression")


# ─── L0-2: Frozen invariant ────────────────────────────────────────────────
@pytest.mark.gpu
def test_frozen_body_does_not_move():
    """Static body (state=0) must retain pose exactly; velocity/FIRE = 0."""
    n = 3
    pos = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n, 1))

    rbd = _build_hcooh(n)
    _reset(rbd, pos, quat)
    rbd.set_body_state(1, 0)  # freeze body 1
    rbd.run_multimol_md(50, dt=0.02, fire=True)

    out = _state(rbd)
    # Body 1 pose unchanged
    np.testing.assert_allclose(out['pos'][1, :3], pos[1], atol=1e-7,
                               err_msg="frozen body pos moved")
    np.testing.assert_allclose(out['quats'][1], quat[1], atol=1e-7,
                               err_msg="frozen body quat changed")
    # Body 1 velocity exactly zero
    np.testing.assert_allclose(out['lin_mom'][1, :3], 0.0, atol=0.0,
                               err_msg="frozen body has nonzero velocity")
    np.testing.assert_allclose(out['ang_mom'][1, :3], 0.0, atol=0.0,
                               err_msg="frozen body has nonzero angular velocity")
    # At least one dynamic body moved
    moved = any(np.linalg.norm(out['pos'][i, :3] - pos[i]) > 1e-7 for i in (0, 2))
    assert moved, "no dynamic body moved"


# ─── L0-3: Static interaction ──────────────────────────────────────────────
@pytest.mark.gpu
def test_static_partner_still_interacts():
    """Dynamic body's force must differ with a static partner vs deleted partner."""
    n = 3
    pos = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n, 1))

    # Body 1 static
    rbd_static = _build_hcooh(n)
    _reset(rbd_static, pos, quat)
    rbd_static.set_body_state(1, 0)
    rbd_static.run_multimol_md(1, dt=0.0)  # zero dt = force eval only
    out_static = _state(rbd_static)

    # Body 1 deleted
    rbd_deleted = _build_hcooh(n)
    _reset(rbd_deleted, pos, quat)
    rbd_deleted.set_body_state(1, -1)
    rbd_deleted.run_multimol_md(1, dt=0.0)
    out_deleted = _state(rbd_deleted)

    # Body 0's force must differ — static partner contributes, deleted doesn't
    f_static = out_static['body_force'][0, :3]
    f_deleted = out_deleted['body_force'][0, :3]
    diff = np.linalg.norm(f_static - f_deleted)
    assert diff > 1e-6, f"static vs deleted force on body 0 identical (diff={diff})"


# ─── L0-4: Deletion parity ─────────────────────────────────────────────────
@pytest.mark.gpu
def test_deletion_matches_rebuilt_reference():
    """3-body with body 1 deleted matches 2-body reference (bodies 0 and 2)."""
    pos3 = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float32)
    quat3 = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (3, 1))

    # 2-body reference: bodies at positions 0 and 10 (skip body 1 at position 5)
    pos2 = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    quat2 = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (2, 1))

    rbd_ref = _build_hcooh(2, spacing=10.0)
    _reset(rbd_ref, pos2, quat2)
    rbd_ref.run_multimol_md(10, dt=0.01, lin_damp=1.0, ang_damp=1.0)
    ref = _state(rbd_ref)

    rbd_del = _build_hcooh(3, spacing=5.0)
    _reset(rbd_del, pos3, quat3)
    rbd_del.set_body_state(1, -1)  # delete body 1
    rbd_del.run_multimol_md(10, dt=0.01, lin_damp=1.0, ang_damp=1.0)
    del_out = _state(rbd_del)

    # Body 0 in 3-body = body 0 in 2-body
    np.testing.assert_allclose(del_out['pos'][0, :3], ref['pos'][0, :3], rtol=2e-6, atol=2e-6,
                               err_msg="deletion parity: body 0 pos mismatch")
    # Body 2 in 3-body = body 1 in 2-body
    np.testing.assert_allclose(del_out['pos'][2, :3], ref['pos'][1, :3], rtol=2e-6, atol=2e-6,
                               err_msg="deletion parity: body 2 pos mismatch")


# ─── L0-5: Kernel 14 (MC energy) deletion parity ───────────────────────────
@pytest.mark.gpu
def test_kernel14_deletion_matches_rebuilt_reference():
    """Kernel 14 energy with body 1 deleted matches 2-body reference for live bodies."""
    pos3 = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float32)
    quat3 = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (3, 1))
    pos2 = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    quat2 = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (2, 1))

    # 2-body reference
    rbd_ref = _build_hcooh(2, spacing=10.0)
    poss2 = np.zeros((1, 2, 4), dtype=np.float32)
    poss2[0, :, :3] = pos2
    qrots2 = np.tile(quat2[None, :, :], (1, 1, 1))
    E_ref = rbd_ref.eval_energy_replicas(poss2, qrots2, active_mols=[0, 1])

    # 3-body with body 1 deleted
    rbd_del = _build_hcooh(3, spacing=5.0)
    rbd_del.set_body_state(1, -1)
    poss3 = np.zeros((1, 3, 4), dtype=np.float32)
    poss3[0, :, :3] = pos3
    qrots3 = np.tile(quat3[None, :, :], (1, 1, 1))
    E_del = rbd_del.eval_energy_replicas(poss3, qrots3, active_mols=[0, 2])

    # E_del has 2 active slots (bodies 0 and 2); E_ref has 2 active slots (bodies 0 and 1)
    # Body 0 energy must match
    np.testing.assert_allclose(E_del[0, 0, :3], E_ref[0, 0, :3], rtol=2e-6, atol=2e-6,
                               err_msg="kernel14 deletion: body 0 energy mismatch")
    # Body 2 (deleted scene) = Body 1 (reference scene)
    np.testing.assert_allclose(E_del[0, 1, :3], E_ref[0, 1, :3], rtol=2e-6, atol=2e-6,
                               err_msg="kernel14 deletion: body 2 energy mismatch")


# ─── L0-6: Kernel 14 all-dynamic regression ────────────────────────────────
@pytest.mark.gpu
def test_kernel14_all_dynamic_matches_default():
    """Kernel 14 energy with body_state=[1,1,1] matches default (no set_body_states)."""
    pos = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (3, 1))
    poss = np.zeros((1, 3, 4), dtype=np.float32)
    poss[0, :, :3] = pos
    qrots = np.tile(quat[None, :, :], (1, 1, 1))

    rbd_ref = _build_hcooh(3)
    E_ref = rbd_ref.eval_energy_replicas(poss, qrots, active_mols=[0, 1, 2])

    rbd_st = _build_hcooh(3)
    rbd_st.set_body_states(np.ones(3, dtype=np.int32))
    E_st = rbd_st.eval_energy_replicas(poss, qrots, active_mols=[0, 1, 2])

    np.testing.assert_allclose(E_st, E_ref, rtol=0.0, atol=0.0,
                               err_msg="kernel14 all-dynamic regression")


# ─── L0-7: Mixed-species build + FAF step ─────────────────────────────────
@pytest.mark.gpu
def test_mixed_species_build_and_faf_step():
    """Build NTCDI+TBTAP+uracil+benzoic_acid on NaCl, run one kernel-15+FAF step."""
    from spammm.forcefields.RigidBodyUtils import build_mixed_species_assembly
    from spammm.surfaces.FoldedRigid import load_or_fit_faf, faf_fit_mode, FAF_MODE_TYPED, FAF_MODE_FACTOR, Z_SURF_TOP, remap_fit_for_molecule

    # Mirror the RA panel's MOL_PATHS
    _MOL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'mol')
    _XYZ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'xyz')
    mol_paths = {
        'NTCDI':        os.path.join(_MOL_DIR, 'NTCDI.mol2'),
        'TBTAP':        os.path.join(_MOL_DIR, 'TBTAP.mol2'),
        'uracil':       os.path.join(_XYZ_DIR, 'uracil.xyz'),
        'benzoic_acid': os.path.join(_XYZ_DIR, 'benzoic_acid.xyz'),
    }
    no_qeq = {'NTCDI'}
    mol_names = ['NTCDI', 'TBTAP', 'uracil', 'benzoic_acid']
    z_init = 3.0
    z_body = float(Z_SURF_TOP + z_init)

    molecules, tids, bonds_list, pos, quat, species_data = build_mixed_species_assembly(
        mol_names, 1, mol_paths, no_qeq, 16.0, z_body, seed=3, qeq=True)

    assert len(molecules) == 4
    assert tids == mol_names  # round-robin order for nmol=1
    assert len(bonds_list) == 4

    rbd = RigidBodyPairFF.from_molecules(
        molecules, pos, quats=quat, active_body=0,
        He=-0.1, rc=3.0, w=0.7, k_z=0.0, z_target=z_body, Hs=1.0, beta=1.7)

    # Verify per-pack data
    assert rbd.n_bodies == 4
    for j, sd in enumerate(species_data):
        sp_apos, sp_enames, sp_REQs, _ = sd
        n_real_sp = len(sp_enames)
        n_real_pack = int((rbd._mb_packs[j]['types'] == 0).sum())
        assert n_real_pack == n_real_sp, f"pack {j} ({tids[j]}): {n_real_pack} real != {n_real_sp}"

    # Attach factorized FAF (shared NaCl substrate)
    fit = load_or_fit_faf((species_data[0][0], species_data[0][1], species_data[0][2]), mol_name=mol_names[0])
    mode = faf_fit_mode(fit)
    n_real_total = sum(len(sd[2]) for sd in species_data)
    if mode == FAF_MODE_TYPED:
        at_ids_parts = []
        for sp_idx, (sp_apos, sp_enames, sp_REQs, _) in enumerate(species_data):
            sp_fit = remap_fit_for_molecule(fit, sp_REQs)
            at_ids_parts.append(sp_fit['atom_type_ids'])
        fit['atom_type_ids'] = np.concatenate(at_ids_parts).astype(np.int32)
    else:
        fit['atom_type_ids'] = np.zeros(n_real_total, dtype=np.int32)
    rbd.attach_pairff_faf(fit, z_init=z_init, k_z=0.0, enable=True)

    # Run one kernel-15+FAF step — must be finite
    rbd.run_multimol_md(1, dt=0.01, faf=True)
    out = rbd.download_outputs()
    assert np.isfinite(out['pos']).all(), "mixed-species FAF step produced non-finite positions"
    assert np.isfinite(out['quats']).all(), "mixed-species FAF step produced non-finite quats"
    print(f"[mixed-species] tids={tids}  n_bodies={rbd.n_bodies}  "
          f"total_atoms={rbd.total_atoms}  faf_mode={'typed' if mode == FAF_MODE_TYPED else 'factorized'}")


# ─── L0-8: Map decomposition + invalidation ────────────────────────────────
@pytest.mark.gpu
def test_combined_map_decomposition_and_invalidation():
    """E_sum == E_pairff_static + E_faf; dynamic-only pose changes leave it unchanged."""
    from spammm.forcefields.RigidBodyUtils import build_mixed_species_assembly, compute_combined_probe_map
    from spammm.surfaces.FoldedRigid import load_or_fit_faf, faf_fit_mode, FAF_MODE_TYPED, FAF_MODE_FACTOR, Z_SURF_TOP, remap_fit_for_molecule

    _MOL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'mol')
    _XYZ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'xyz')
    mol_paths = {
        'NTCDI':  os.path.join(_MOL_DIR, 'NTCDI.mol2'),
        'uracil': os.path.join(_XYZ_DIR, 'uracil.xyz'),
    }
    no_qeq = {'NTCDI'}
    mol_names = ['NTCDI', 'uracil']
    z_init = 3.0
    z_body = float(Z_SURF_TOP + z_init)

    molecules, tids, bonds_list, pos, quat, species_data = build_mixed_species_assembly(
        mol_names, 1, mol_paths, no_qeq, 16.0, z_body, seed=3, qeq=True)

    rbd = RigidBodyPairFF.from_molecules(
        molecules, pos, quats=quat, active_body=0,
        He=-0.1, rc=3.0, w=0.7, k_z=0.0, z_target=z_body, Hs=1.0, beta=1.7)

    fit = load_or_fit_faf((species_data[0][0], species_data[0][1], species_data[0][2]), mol_name=mol_names[0])
    mode = faf_fit_mode(fit)
    n_real_total = sum(len(sd[2]) for sd in species_data)
    if mode == FAF_MODE_TYPED:
        at_ids_parts = []
        for sp_idx, (sp_apos, sp_enames, sp_REQs, _) in enumerate(species_data):
            sp_fit = remap_fit_for_molecule(fit, sp_REQs)
            at_ids_parts.append(sp_fit['atom_type_ids'])
        fit['atom_type_ids'] = np.concatenate(at_ids_parts).astype(np.int32)
    else:
        fit['atom_type_ids'] = np.zeros(n_real_total, dtype=np.int32)
    rbd.attach_pairff_faf(fit, z_init=z_init, k_z=0.0, enable=True)

    # Freeze body 0 (NTCDI), body 1 (uracil) dynamic
    rbd.set_body_state(0, 0)
    frozen_mask = np.array([True, False], dtype=bool)
    z_probe = float(Z_SURF_TOP + 3.0)

    # O− probe parameters (default per spec)
    from spammm.forcefields.QEq import get_atom_types
    _, atom_types = get_atom_types()
    probe_R0, probe_E0, probe_q = float(atom_types['O'].RvdW), float(atom_types['O'].EvdW), -0.4

    E_sum, E_pairff, E_faf, xs, ys, extent = compute_combined_probe_map(
        rbd, fit, frozen_mask, probe_R0, probe_E0, probe_q, z_probe, beta=1.7)

    # Decomposition: E_sum ≈ E_pairff + E_faf
    if E_faf is not None:
        np.testing.assert_allclose(E_sum, E_pairff + E_faf, rtol=1e-10, atol=1e-10,
                                   err_msg="map decomposition: E_sum != E_pairff + E_faf")
    assert np.isfinite(E_sum).all(), "map has non-finite values"

    # Invalidation: dynamic-only pose change (body 1 moves) must NOT change the map
    # (body 1 is dynamic, excluded from map; only frozen body 0 contributes)
    pos_moved = pos.copy()
    pos_moved[1, 0] += 2.0  # move dynamic body 1
    rbd.toGPU('poss', np.concatenate([pos_moved, np.ones((2, 1), dtype=np.float32)], axis=1).astype(np.float32))
    rbd._mb_pos = pos_moved.copy()
    E_sum2, _, _, _, _, _ = compute_combined_probe_map(
        rbd, fit, frozen_mask, probe_R0, probe_E0, probe_q, z_probe, beta=1.7)
    np.testing.assert_allclose(E_sum2, E_sum, rtol=0.0, atol=0.0,
                               err_msg="map changed when only dynamic body moved (should be invariant)")

    print(f"[map] E_sum range=[{E_sum.min():.3f},{E_sum.max():.3f}]  "
          f"E_pairff range=[{E_pairff.min():.3f},{E_pairff.max():.3f}]  "
          f"E_faf range=[{E_faf.min():.3f},{E_faf.max():.3f}]  "
          f"grid={len(xs)}x{len(ys)}")
