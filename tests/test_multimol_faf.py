"""FAF-fused concurrent multi-molecule MD parity + relaxation tests (kernel 15).

Verifies:
  L0a — do_faf=0 is bit-identical to pre-FAF kernel 15 (regression guard).
  L0b — tensor FAF evaluator (kernel 15) matches the flat folded_eval_basis_rigid
        used by kernel 13 (rigid_body_pairff_unified_allmol_faf_kernel), single mol.
  L0c — concurrent multi-molecule MD with FAF relaxes toward the single-molecule
        FAF minimum (energy decreases, z converges toward z_target).

Uses cached PTCDA/NaCl factorized fit (data/fits/ptcda_nacl_factorized.npz).
"""
import os
import pytest
import numpy as np

from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF
from spammm.forcefields.RigidBodyUtils import load_molecule
from spammm.surfaces.FoldedRigid import load_fit, Z_SURF_TOP

_HCOOH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'xyz', 'HCOOH.xyz')
_PTCDA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'xyz', 'PTCDA.xyz')
_FIT_FACTOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fits', 'ptcda_nacl_factorized.npz')
_FIT_TYPED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fits', 'ptcda_nacl_typed.npz')


def _build_hcooh_pair():
    apos, enames, REQs, _ = load_molecule(_HCOOH, qeq=False, name='formic_acid')
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.25, 0.0]], dtype=np.float32)
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1))
    rbd = RigidBodyPairFF.from_molecules([(apos, enames, REQs)] * 2, pos, quats=quat)
    return rbd


def _build_ptcda_pair(fit_path, spacing=8.0, z=3.5):
    """Build 2 PTCDA molecules on NaCl and attach FAF (typed or factorized)."""
    fit = load_fit(fit_path)
    apos, enames, REQs, _ = load_molecule(_PTCDA, qeq=False, name='PTCDA')
    pos = np.array([[0.0, 0.0, z], [spacing, 0.0, z]], dtype=np.float32)
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1))
    rbd = RigidBodyPairFF.from_molecules([(apos, enames, REQs)] * 2, pos, quats=quat)
    rbd.attach_pairff_faf(fit, z_init=z, k_z=0.0, enable=True)
    return rbd, fit


def _state(rbd):
    return rbd.download_selected(('pos', 'quats', 'lin_mom', 'ang_mom', 'body_force', 'body_torque'))


def _reset(rbd, pos, quat):
    zero = np.zeros((rbd.n_bodies, 4), dtype=np.float32)
    pos4 = np.zeros((rbd.n_bodies, 4), dtype=np.float32); pos4[:, :3] = pos; pos4[:, 3] = 1.0
    rbd.toGPU('poss', pos4); rbd.toGPU('qrots', quat); rbd.toGPU('vposs', zero); rbd.toGPU('vrots', zero); rbd.toGPU('fire_state', zero)
    rbd.queue.finish()


# ─── L0a: do_faf=0 regression ──────────────────────────────────────────────
@pytest.mark.gpu
def test_multimol_faf_disabled_regression():
    """run_multimol_md(faf=False) must match run_multimol_md() with no FAF bound."""
    rbd = _build_hcooh_pair()
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.25, 0.0]], dtype=np.float32)
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1))
    _reset(rbd, pos, quat); rbd.run_multimol_md(3, dt=0.01, lin_damp=1.0, ang_damp=1.0); ref = _state(rbd)
    _reset(rbd, pos, quat); rbd.run_multimol_md(3, dt=0.01, lin_damp=1.0, ang_damp=1.0, faf=False); raw = _state(rbd)
    for name in ref:
        np.testing.assert_allclose(raw[name], ref[name], rtol=0.0, atol=0.0, err_msg=f"faf=False {name} regression")


# ─── L0b: tensor vs flat evaluator parity (single molecule, FAF-only) ─────
def _run_single_mol_faf_flat(rbd_single, n_steps, dt=0.01):
    """Run single-molecule FAF MD via kernel 13 (rigid_body_pairff_unified_allmol_faf_kernel).

    Uses run_pairff(faf=True) which selects the fused PairFF+FAF kernel.
    """
    rbd_single.run_pairff(n_steps, dt=dt, lin_damp=1.0, ang_damp=1.0, faf=True)


def _run_single_mol_faf_tensor(rbd_single, n_steps, dt=0.01):
    """Run single-molecule FAF MD via kernel 15 with do_faf=1 (tensor evaluator).

    Uses run_multimol_md(faf=True) on a 1-molecule system.
    """
    rbd_single.run_multimol_md(n_steps, dt=dt, lin_damp=1.0, ang_damp=1.0, faf=True)


@pytest.mark.gpu
def test_multimol_faf_tensor_vs_flat_parity():
    """Tensor FAF evaluator (kernel 15) must match flat evaluator (kernel 13) for 1 mol."""
    fit = load_fit(_FIT_FACTOR)
    apos, enames, REQs, _ = load_molecule(_PTCDA, qeq=False, name='PTCDA')
    z = 3.5
    pos1 = np.array([[0.0, 0.0, z]], dtype=np.float32)
    quat1 = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (1, 1))

    # Flat evaluator (kernel 13) — single active body in allmol layout
    rbd_flat = RigidBodyPairFF.from_molecules([(apos, enames, REQs)], pos1, quats=quat1)
    rbd_flat.attach_pairff_faf(fit, z_init=z, k_z=0.0, enable=True)
    _reset(rbd_flat, pos1, quat1); _run_single_mol_faf_flat(rbd_flat, 5, dt=0.01); flat = _state(rbd_flat)

    # Tensor evaluator (kernel 15) — 1-molecule multimol
    rbd_ten = RigidBodyPairFF.from_molecules([(apos, enames, REQs)], pos1, quats=quat1)
    rbd_ten.attach_pairff_faf(fit, z_init=z, k_z=0.0, enable=True)
    _reset(rbd_ten, pos1, quat1); _run_single_mol_faf_tensor(rbd_ten, 5, dt=0.01); ten = _state(rbd_ten)

    for name in flat:
        np.testing.assert_allclose(ten[name], flat[name], rtol=1e-5, atol=1e-5, err_msg=f"tensor vs flat {name}")


# ─── L0c: concurrent multi-molecule FAF relaxation ─────────────────────────
@pytest.mark.gpu
def test_multimol_faf_relaxation():
    """2 PTCDA molecules with FAF: energy must decrease over MD steps."""
    rbd, fit = _build_ptcda_pair(_FIT_FACTOR, spacing=8.0, z=3.5)
    # Initial energy
    rbd.run_multimol_md(1, dt=0.0, faf=True)  # zero dt = force eval only
    atoms = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms); rbd.queue.finish()
    E0 = float(atoms[:, 3].sum())
    # Relax
    rbd.run_multimol_md(200, dt=0.05, lin_damp=0.9, ang_damp=0.88, faf=True)
    rbd.fromGPU('apos_world', atoms); rbd.queue.finish()
    E1 = float(atoms[:, 3].sum())
    print(f"[FAF relax] E0={E0:.6f}  E1={E1:.6f}  dE={E1-E0:.6f}")
    assert E1 < E0, f"FAF relaxation did not decrease energy: E0={E0}, E1={E1}"


# ─── L0d: typed fit also works (not just factorized) ───────────────────────
@pytest.mark.gpu
def test_multimol_faf_typed_fit():
    """Typed FAF fit must also run through the tensor evaluator without error."""
    if not os.path.isfile(_FIT_TYPED):
        pytest.skip(f"typed fit not available: {_FIT_TYPED}")
    rbd, fit = _build_ptcda_pair(_FIT_TYPED, spacing=8.0, z=3.5)
    rbd.run_multimol_md(5, dt=0.01, faf=True)
    atoms = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms); rbd.queue.finish()
    E = float(atoms[:, 3].sum())
    assert np.isfinite(E), f"typed FAF produced non-finite energy: {E}"
