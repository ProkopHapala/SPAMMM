"""
FFEvaluator.py — GPU single-point UFF/SPFF evaluation for finite-difference Hessians.

Extracted from the `test_forcefield` eval_fn pattern so vibrations and EF parity tests
share the same force pipeline. SPFF path uses dt=0 `updateAtomsSPFFf4` to assemble
recoil forces without moving atoms.

Open issues: UFF relaxation in `FFController` is separate and still unintegrated.
"""

import numpy as np
import pyopencl as cl

from ..AtomicSystem import AtomicSystem
from .UFF_cl import UFF_cl
from .SPFF_cl import SPFF_cl
from .SPFFbuilder import SPFF
from ..topology.FFparams import SPFFparams

_MASS_MAP = {'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.07, 'F': 19.00, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90, 'Si': 28.09, 'P': 30.97}


def make_uff_eval_fn(mol, do_nonbond=False):
    """Build eval_fn(pos) -> (E, F) for UFF. mol: AtomicSystem or xyz path."""
    if isinstance(mol, str):
        mol = AtomicSystem(fname=mol)
    uff = UFF_cl()
    uff.toUFF(mol)
    uff.bDoNonBonded = do_nonbond
    uff.args_setup = False
    masses = np.ones(len(mol.apos), dtype=np.float32)
    uff.upload_positions(mol.apos.astype(np.float32), masses=masses)
    pos0 = mol.apos.astype(np.float32).copy()

    def eval_fn(pos):
        uff.upload_positions(pos.astype(np.float32), masses=masses)
        uff.run_eval_step()
        E = float(uff.get_total_energy()[0])
        F = uff.get_forces()[0].copy()
        return E, F

    return eval_fn, pos0, len(mol.apos), None, list(mol.enames)


def make_spff_eval_fn(mol, do_nonbond=False):
    """Build eval_fn(pos) -> (E, F) for SPFF (pi-orbitals frozen)."""
    if isinstance(mol, str):
        mol = AtomicSystem(fname=mol)
    params = SPFFparams('data/')
    mol.atypes = np.array([params.getAtomType(e, bErr=False) for e in mol.enames], dtype=np.int32)
    spff = SPFF()
    spff.toSPFFsp3_loc(mol, params.atom_types_map)
    for ia in range(spff.natoms):
        e = mol.enames[ia]
        spff.apos[ia, 3] = _MASS_MAP.get(e, 12.01)
    md = SPFF_cl(enable_nonbond=do_nonbond)
    md.realloc(spff, nSystems=1)
    md.upload_all_systems()
    md.setup_kernels()
    cl.enqueue_fill_buffer(md.queue, md.buffer_dict['avel'], np.float32(0), 0, md.buffer_dict['avel'].size)
    md.queue.finish()
    md.set_md_params(dt=0.0, damp=1.0, Flimit=0.0)
    natoms = spff.natoms
    pos0 = spff.apos[:natoms, :3].copy().astype(np.float32)
    perm = getattr(mol, 'perm_nodes_first', list(range(natoms)))

    def eval_fn(pos):
        spff.apos[:natoms, :3] = pos.astype(np.float32)
        md.toGPU('apos', md._flat32(spff.apos), byte_offset=0)
        md.run_cleanForceSPFFf4()
        md.run_getSPFFf4()
        md.run_updateAtomsSPFFf4()
        E = float(md.get_total_energy())
        F = md.get_forces()[:, :3].copy()
        return E, F

    return eval_fn, pos0, natoms, perm, list(mol.enames)


def make_ff_eval_fn(mol, ff='uff', do_nonbond=False):
    """Unified factory. ff in ('uff', 'spff'). Returns (eval_fn, pos0, natoms, perm, enames)."""
    if ff == 'uff':
        return make_uff_eval_fn(mol, do_nonbond=do_nonbond)
    if ff == 'spff':
        return make_spff_eval_fn(mol, do_nonbond=do_nonbond)
    raise ValueError(f"Unknown ff={ff!r}, expected 'uff' or 'spff'")
