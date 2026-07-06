"""Kekulé / C–C bond analysis on ScanDataset trajectories (poor-man's NEB, rigid scans)."""
import numpy as np

from spammm.topology.scan_dataset import cc_bond_indices
from spammm.topology.KekulePure import run_kekule_solver, make_n_pi
from spammm.AtomicSystem import AtomicSystem


def analyze_kekule_cc(dataset, localize=True, frame_stride=1, kval=50.0, kloc=5.0, karo=0.5, aromatic=True, verbose=False):
    """Run KekulePure on selected frames; return pi_bo_cc [nframes, n_cc] and cc bond index list."""
    cc_idx = cc_bond_indices(dataset.etype, dataset.bonds)
    if len(cc_idx) == 0:
        return cc_idx, None
    pi_bo = np.full((dataset.nframes, len(cc_idx)), np.nan, dtype=np.float64)
    enames = dataset.enames()
    for fi in range(0, dataset.nframes, frame_stride):
        apos = dataset.apos[fi]
        atoms = AtomicSystem(apos=apos, atypes=dataset.etype.copy(), enames=enames.copy())
        atoms.neighs()
        n_pi = make_n_pi(atoms)
        result = run_kekule_solver(atoms, n_pi=n_pi, Kval=kval, Kloc=kloc if localize else 0.0, Karo=karo, allow_aromatic=aromatic, localize=localize, sym_break=0.0)
        bo = np.asarray(result['bo_snap'], dtype=float)
        for j, bi in enumerate(cc_idx):
            pi_bo[fi, j] = bo[int(bi)]
        if verbose:
            print(f"  kekule frame {fi}: cc pi_bo mean={np.nanmean(pi_bo[fi]):.3f}")
    dataset.pi_bo_cc = pi_bo
    dataset.meta['cc_bond_idx'] = cc_idx.tolist()
    dataset.meta['kekule_analyzed'] = True
    return cc_idx, pi_bo


def cc_length_vs_control(dataset, cc_bond_idx=None):
    """Return (controls_1d, bond_len_cc [nframes, n_cc]) for plotting."""
    cc_bond_idx = np.asarray(cc_bond_idx if cc_bond_idx is not None else cc_bond_indices(dataset.etype, dataset.bonds), dtype=np.int32)
    u = dataset.controls[:, 0] if dataset.m >= 1 else np.arange(dataset.nframes, dtype=float)
    bl = dataset.bond_len[:, cc_bond_idx] if len(cc_bond_idx) else np.zeros((dataset.nframes, 0))
    return u, bl, cc_bond_idx
