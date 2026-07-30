"""L0 tests for RigidEnsemble — shared rigid-pose numpy store.

Verifies the locked conventions (doc/Tasks/RigidMoleculePose_SSOT.md):
  - qrot xyzw, identity [0,0,0,1]
  - round-trip set_poses -> get_poses unchanged
  - quat normalization preserves identity, fixes non-unit
  - R = _quat_to_matrix_np(qrot) matches kernel convention (parity with RigidBodyDynamics)
  - stable ids (never reused, id_to_idx bridge)
  - append-only dense indices stable
  - subset reads/writes (PME active subset pattern)
  - copy vs view semantics

No GPU, no OpenCL — pure numpy. Runs under `pytest -m "not slow"`.
"""
import numpy as np
import pytest
from spammm.forcefields.RigidEnsemble import RigidEnsemble, RigidBody, IDENTITY_QUAT


def _R_ref(q):
    """Reference R from RigidBodyDynamics._quat_to_matrix_np (the SSOT quat math)."""
    from spammm.forcefields.RigidBodyDynamics import _quat_to_matrix_np
    return _quat_to_matrix_np(np.asarray(q, dtype=np.float32))


# ─── construction & defaults ──────────────────────────────────────────────────
def test_empty_ensemble():
    ens = RigidEnsemble()
    assert len(ens) == 0
    assert ens.n_bodies == 0
    pos, qrot = ens.get_poses()
    assert pos.shape == (0, 3) and qrot.shape == (0, 4)


def test_add_body_defaults_identity_quat():
    ens = RigidEnsemble()
    i = ens.add_body('PTCDA', pos=[1.0, 2.0, 3.0])
    assert i == 0 and len(ens) == 1
    pos, qrot = ens.get_poses()
    assert pos.shape == (1, 3) and qrot.shape == (1, 4)
    np.testing.assert_allclose(pos[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(qrot[0], IDENTITY_QUAT)  # xyzw identity


def test_add_bodies_batch():
    ens = RigidEnsemble()
    tids = ['A', 'B', 'C']
    pos = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
    q = np.array([[0, 0, 0, 1], [0, 0, np.sin(0.5), np.cos(0.5)], [0, 0, 0, 1]], dtype=np.float32)
    idxs = ens.add_bodies(tids, pos, q)
    assert idxs == [0, 1, 2] and len(ens) == 3
    p, qr = ens.get_poses()
    np.testing.assert_allclose(p, pos)
    np.testing.assert_allclose(qr, q)


def test_from_poses_classmethod():
    tids = ['X', 'Y']
    pos = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    ens = RigidEnsemble.from_poses(tids, pos)
    assert len(ens) == 2
    p, _ = ens.get_poses()
    np.testing.assert_allclose(p, pos)


# ─── round-trip ───────────────────────────────────────────────────────────────
def test_set_poses_roundtrip():
    ens = RigidEnsemble.from_poses(['A', 'B', 'C'], np.zeros((3, 3), np.float32))
    pos = np.random.randn(3, 3).astype(np.float32)
    q = np.random.randn(3, 4).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ens.set_poses(pos, q)
    p, qr = ens.get_poses()
    np.testing.assert_allclose(p, pos)
    np.testing.assert_allclose(qr, q)


def test_set_pose_single():
    ens = RigidEnsemble.from_poses(['A', 'B'], np.zeros((2, 3), np.float32))
    ens.set_pose(1, [5, 6, 7], [0, 0, 0, 1])
    p, q = ens.get_pose(1)
    np.testing.assert_allclose(p, [5, 6, 7])
    np.testing.assert_allclose(q, [0, 0, 0, 1])


def test_set_poses_qrot_none_keeps_quats():
    ens = RigidEnsemble.from_poses(['A', 'B'], np.zeros((2, 3), np.float32))
    q0 = np.array([[0, 0, np.sin(0.3), np.cos(0.3)], [0, 0, 0, 1]], np.float32)
    ens.set_poses(np.zeros((2, 3), np.float32), q0)
    ens.set_poses(np.array([[9, 9, 9], [8, 8, 8]], np.float32))  # qrot=None
    p, q = ens.get_poses()
    np.testing.assert_allclose(q, q0)  # quats preserved
    np.testing.assert_allclose(p[0], [9, 9, 9])


# ─── quaternion normalization ────────────────────────────────────────────────
def test_normalize_quats_preserves_unit():
    ens = RigidEnsemble.from_poses(['A'], np.zeros((1, 3), np.float32),
                                   qrot=np.array([[0, 0, 0, 1]], np.float32))
    ens.normalize_quats()
    _, q = ens.get_poses()
    np.testing.assert_allclose(q[0], [0, 0, 0, 1])


def test_normalize_quats_fixes_nonunit():
    q = np.array([[0, 0, 0, 2.0], [0.3, 0.4, 0.5, 0.6]], np.float32)  # non-unit
    ens = RigidEnsemble.from_poses(['A', 'B'], np.zeros((2, 3), np.float32), qrot=q)
    ens.normalize_quats()
    _, qr = ens.get_poses()
    n = np.linalg.norm(qr, axis=1)
    np.testing.assert_allclose(n, [1.0, 1.0], atol=1e-6)


def test_normalize_quats_zero_guard():
    ens = RigidEnsemble.from_poses(['A'], np.zeros((1, 3), np.float32),
                                   qrot=np.array([[0, 0, 0, 0]], np.float32))
    ens.normalize_quats()  # must not NaN
    _, q = ens.get_poses()
    assert np.isfinite(q).all()


# ─── parity with RigidBodyDynamics._quat_to_matrix_np ─────────────────────────
def test_R_matches_quat_to_matrix_np_single():
    """R(q) from ensemble qrot must match the SSOT _quat_to_matrix_np (kernel convention)."""
    from spammm.forcefields.RigidBodyDynamics import _quat_to_matrix_np
    q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.01 - 0.04 - 0.09)], np.float32)
    ens = RigidEnsemble.from_poses(['A'], np.zeros((1, 3), np.float32), qrot=q[None])
    _, qr = ens.get_poses()
    R_ens = _quat_to_matrix_np(qr[0])
    R_ref = _R_ref(q)
    np.testing.assert_allclose(R_ens, R_ref, atol=1e-6)


def test_R_matches_quat_to_matrix_np_batch():
    """Batch path of _quat_to_matrix_np on ensemble qrot matches batch path on original q.

    Note: the batch path returns R transposed vs the single path (pre-existing convention
    in _quat_to_matrix_np: single -> R used as `rel @ R.T`; batch -> R.T used as `rel @ R`).
    This test verifies the ensemble does not corrupt qrot, comparing batch-vs-batch.
    """
    from spammm.forcefields.RigidBodyDynamics import _quat_to_matrix_np
    rng = np.random.default_rng(42)
    q = rng.standard_normal((5, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    ens = RigidEnsemble.from_poses(['A']*5, np.zeros((5, 3), np.float32), qrot=q)
    _, qr = ens.get_poses()
    R_from_ens = _quat_to_matrix_np(qr)        # (N,3,3) batch path on ensemble qrot
    R_from_orig = _quat_to_matrix_np(q)        # (N,3,3) batch path on original qrot
    np.testing.assert_allclose(R_from_ens, R_from_orig, atol=1e-6)


def test_world_atoms_formula_via_single_path():
    """apos_world = pos + rel @ R.T (single-path R) — the SE(3) convention consumers use."""
    from spammm.forcefields.RigidBodyDynamics import _quat_to_matrix_np, _body_sites_world
    q = np.array([0.1, 0.2, 0.3, np.sqrt(1 - 0.01 - 0.04 - 0.09)], np.float32)
    pos = np.array([1.0, 2.0, 3.0], np.float32)
    rel = np.random.randn(4, 3).astype(np.float32)
    ens = RigidEnsemble.from_poses(['A'], pos[None], qrot=q[None])
    p, qr = ens.get_poses()
    # Consumer does its own conversion using the existing helper:
    world = _body_sites_world(rel, p[0], qr[0])
    R = _quat_to_matrix_np(qr[0])
    world_ref = pos + rel @ R.T
    np.testing.assert_allclose(world, world_ref, atol=1e-5)


def test_identity_quat_gives_identity_R():
    from spammm.forcefields.RigidBodyDynamics import _quat_to_matrix_np
    ens = RigidEnsemble.from_poses(['A'], np.zeros((1, 3), np.float32))  # identity default
    _, q = ens.get_poses()
    R = _quat_to_matrix_np(q[0])
    np.testing.assert_allclose(R, np.eye(3, dtype=np.float32), atol=1e-6)


# ─── stable ids ───────────────────────────────────────────────────────────────
def test_stable_ids_never_reused():
    RigidBody._counter = 0  # reset for deterministic test
    ens = RigidEnsemble()
    i0 = ens.add_body('A', [0, 0, 0])
    i1 = ens.add_body('B', [1, 1, 1])
    id0, id1 = ens.get_ids()
    assert id0 == 1 and id1 == 2  # monotonic
    assert id0 != id1


def test_id_to_idx_bridge():
    ens = RigidEnsemble()
    ens.add_body('A', [0, 0, 0])
    i1 = ens.add_body('B', [1, 1, 1])
    id1 = ens.get_ids()[i1]
    assert ens.id_to_idx(id1) == i1
    p, _ = ens.get_pose(id1, by_id=True)
    np.testing.assert_allclose(p, [1, 1, 1])
    ens.set_pose(id1, [9, 9, 9], by_id=True)
    p, _ = ens.get_pose(i1)
    np.testing.assert_allclose(p, [9, 9, 9])


def test_dense_indices_stable_after_append():
    """Append-only: existing dense indices must not shift."""
    ens = RigidEnsemble()
    i0 = ens.add_body('A', [1, 2, 3])
    ens.add_body('B', [4, 5, 6])
    ens.add_body('C', [7, 8, 9])
    p, _ = ens.get_pose(i0)
    np.testing.assert_allclose(p, [1, 2, 3])  # index 0 unchanged


# ─── subset (PME active subset pattern) ───────────────────────────────────────
def test_subset_read():
    ens = RigidEnsemble.from_poses(['A', 'B', 'C', 'D'],
                                   np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], np.float32))
    sub = [0, 2]
    p, q = ens.get_poses(subset=sub)
    assert p.shape == (2, 3) and q.shape == (2, 4)
    np.testing.assert_allclose(p, [[0, 0, 0], [2, 2, 2]])


def test_subset_write():
    ens = RigidEnsemble.from_poses(['A', 'B', 'C'], np.zeros((3, 3), np.float32))
    ens.set_poses(np.array([[9, 9, 9], [8, 8, 8]], np.float32), subset=[0, 2])
    p, _ = ens.get_poses()
    np.testing.assert_allclose(p[0], [9, 9, 9])
    np.testing.assert_allclose(p[1], [0, 0, 0])  # untouched
    np.testing.assert_allclose(p[2], [8, 8, 8])


# ─── copy vs view ─────────────────────────────────────────────────────────────
def test_copy_default_does_not_mutate_ensemble():
    ens = RigidEnsemble.from_poses(['A'], np.zeros((1, 3), np.float32))
    p, _ = ens.get_poses()  # copy=True default
    p[0, 0] = 999.0
    p2, _ = ens.get_poses()
    assert p2[0, 0] == 0.0  # ensemble untouched


def test_view_is_fast_and_shares_storage():
    ens = RigidEnsemble.from_poses(['A'], np.array([[1, 2, 3]], np.float32))
    p, _ = ens.get_poses(copy=False)
    assert p.base is not None or p.flags.owndata is False  # view
    np.testing.assert_allclose(p[0], [1, 2, 3])


# ─── growth ───────────────────────────────────────────────────────────────────
def test_capacity_grows_on_add():
    ens = RigidEnsemble(capacity=2)
    for k in range(10):
        ens.add_body('A', [k, k, k])
    assert len(ens) == 10
    p, _ = ens.get_poses()
    np.testing.assert_allclose(p[9], [9, 9, 9])
    np.testing.assert_allclose(p[0], [0, 0, 0])  # preserved across growth


# ─── metadata ─────────────────────────────────────────────────────────────────
def test_get_tids_and_body():
    ens = RigidEnsemble.from_poses(['PTCDA', 'NTCDI'], np.zeros((2, 3), np.float32))
    assert ens.get_tids() == ['PTCDA', 'NTCDI']
    b = ens.get_body(0)
    assert b.tid == 'PTCDA' and b.active is True and b.in_pme_subset is True


def test_summary_finite():
    ens = RigidEnsemble.from_poses(['A', 'B'], np.random.randn(2, 3).astype(np.float32))
    s = ens.summary()
    assert 'n=2' in s and 'RigidEnsemble' in s
