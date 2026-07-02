"""Parity test: relax_serial (single-kernel local-memory) vs relax_batch (3 kernel calls/step).

Verifies that the new relax_nsteps_serial kernel produces identical results
to the existing multi-kernel approach for several molecules.
"""
import pytest, numpy as np, os

os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

# Molecules small enough for relax_serial (nvec <= 128, nnode <= 64)
PARITY_CASES = [
    ('H2O.xyz',     200,  0.05, 0.9),
    ('CH4.xyz',     500,  0.05, 0.9),
    ('benzene.xyz', 500,  0.05, 0.9),
    ('HCOOH.xyz',   500,  0.05, 0.9),
    ('uracil.xyz',  500,  0.05, 0.9),
    ('PTCDA.xyz',   500,  0.05, 0.9),
]


@pytest.mark.gpu
@pytest.mark.parametrize('mol_file,nsteps,dt,damp', PARITY_CASES)
def test_relax_serial_parity(xyz, mol_file, nsteps, dt, damp):
    """Run same molecule with relax_batch and relax_serial, compare positions and energy."""
    from spammm.AtomicSystem import AtomicSystem
    from spammm.forcefields.FFController import FFController

    # --- Run A: relax_batch ---
    mol_a = AtomicSystem(fname=xyz(mol_file))
    # Distort both copies identically using fixed seed
    rng = np.random.RandomState(42)
    mol_a.apos = mol_a.apos + rng.randn(*mol_a.apos.shape) * 0.15
    ctrl_a = FFController()
    ctrl_a.build_ff(mol_a, ff_type='spff')
    ctrl_a.md.set_md_params(dt=dt, damp=damp, Flimit=100.0)
    ctrl_a.md.relax_batch(nsteps=nsteps, do_nb=False)
    ctrl_a.md.queue.finish()
    E_a = ctrl_a.md.get_total_energy()
    pos_a = ctrl_a.md.get_positions()
    ctrl_a.teardown()

    # --- Run B: relax_serial ---
    mol_b = AtomicSystem(fname=xyz(mol_file))
    rng = np.random.RandomState(42)
    mol_b.apos = mol_b.apos + rng.randn(*mol_b.apos.shape) * 0.15
    ctrl_b = FFController()
    ctrl_b.build_ff(mol_b, ff_type='spff')
    ctrl_b.md.relax_serial(nsteps=nsteps, dt=dt, damp=damp, Flimit=100.0)
    E_b = ctrl_b.md.get_total_energy()
    pos_b = ctrl_b.md.get_positions()
    ctrl_b.teardown()

    # --- Compare ---
    pos_tol = 1e-3  # 0.001 Å — allows for minor FP ordering differences
    E_tol = 1e-4    # 0.0001 eV

    max_pos_diff = np.max(np.abs(pos_a - pos_b))
    E_diff = abs(E_a - E_b)

    print(f"\n  {mol_file}: max_pos_diff={max_pos_diff:.6e}  E_diff={E_diff:.6e}  E_a={E_a:.6f}  E_b={E_b:.6f}")

    assert max_pos_diff < pos_tol, f"Position mismatch: max diff {max_pos_diff} > tol {pos_tol}"
    assert E_diff < E_tol, f"Energy mismatch: diff {E_diff} > tol {E_tol}"
