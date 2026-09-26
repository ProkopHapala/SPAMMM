#!/usr/bin/env python3
"""Bond-order maps from exported PySCF matrices (mats/<key>.npz).

  total (default):  W_AB = sum_{i in A, j in B} P~_ij^2,  P~ = S^1/2 (Pa+Pb) S^1/2
                    (Lowdin Wiberg, sigma+pi; benzene C-C ~1.4, C=C ~2.0)
  --pi:             pi-only: contract each heavy-atom p-shell triplet with the
                    molecular plane normal n -> effective pi AOs; Löwdin-
                    orthonormalize (Ppi,Spi) in that subspace; W_pi[A,B] =
                    sum_{i in piAO(A), j in piAO(B)} P~_pi[i,j]^2
                    (benzene C-C pi ~0.3-0.4, ethylene pi ~1.0).
Open-shell jobs also annotate per-atom Mulliken spin Tr[(Pa-Pb)S]_AA.

Usage:
    python3 tests/topology/plot_pyscf_bo.py <jobdir> [--states a,b] [--pi]
                                                    [--out bo_pyscf]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm.AtomicSystem import AtomicSystem
from spammm.quantum.pi_bond_order import plot_bond_scalar_map, mol_plane_normal


def _rebuild(d, basis_fallback='def2-svp'):
    """npz -> (mol, enames, apos, S, dm)."""
    en = [str(e) for e in d['enames']]
    basis = str(d['basis']) if 'basis' in d.files else basis_fallback
    from pyscf import gto
    mol = gto.M(atom=[[e, tuple(p)] for e, p in zip(en, d['apos'])],
                basis=basis, unit='Ang', verbose=0,
                charge=int(d['charge']), spin=int(d['spin']))
    assert mol.nao == d['S'].shape[0], \
        f'nao mismatch: basis {basis} rebuilt {mol.nao} != saved {d["S"].shape[0]}'
    return mol, en, d['apos'], d['S'], d['dm']


def _lowdin(P, S):
    ew, ev = np.linalg.eigh(S)
    assert ew.min() > 1e-10, 'overlap not positive definite'
    S12 = (ev * np.sqrt(ew)) @ ev.T
    return S12 @ P @ S12


def wiberg_bo(npz, basis_fallback='def2-svp', pi=False):
    """npz -> (enames, apos_A, W[nat,nat], spin_pop[nat], meta).
    pi=True -> pi-subspace bond order (heavy atoms only; non-pi bonds = nan)."""
    d = np.load(npz)
    mol, en, apos, S, dm = _rebuild(d, basis_fallback)
    PaPb = dm.sum(0) if dm.ndim == 3 else dm
    sl = mol.aoslice_by_atom()[:, 2:4]
    nat = mol.natm
    if not pi:
        Pl = _lowdin(PaPb, S)
        W = np.zeros((nat, nat))
        for a in range(nat):
            for b in range(a + 1, nat):
                W[a, b] = W[b, a] = (Pl[sl[a, 0]:sl[a, 1], sl[b, 0]:sl[b, 1]] ** 2).sum()
    else:
        # effective pi AOs: p-triplets on heavy atoms contracted with plane normal
        trips = {}                                       # atom -> [triplet AO idx]
        for i, (ia, _s, shl, ml) in enumerate(mol.ao_labels(fmt=False)):
            if shl.endswith('p') and ml == 'x' and en[ia] != 'H':
                trips.setdefault(ia, []).append((i, i + 1, i + 2))
        n_pi = sum(len(t) for t in trips.values())
        W = np.full((nat, nat), np.nan)
        if n_pi == 0:                                # no pi centres (H-only etc.)
            spin = np.zeros(nat)
            if dm.ndim == 3:
                PS = (dm[0] - dm[1]) @ S
                spin = np.array([np.trace(PS[s0:s1, s0:s1]) for s0, s1 in sl])
            meta = dict(E_eV=float(d['E_eV']), S2=float(d['S2']),
                        basis=str(d['basis']) if 'basis' in d.files else basis_fallback)
            return en, apos, W, spin, meta
        normal = mol_plane_normal(apos, np.array([e != 'H' for e in en]))
        V = np.zeros((mol.nao, n_pi))
        owner = np.zeros(n_pi, dtype=int)
        k = 0
        for ia in sorted(trips):
            for (ix, iy, iz) in trips[ia]:
                V[ix, k], V[iy, k], V[iz, k] = normal     # n.(px,py,pz)
                owner[k] = ia
                k += 1
        Ppi = _lowdin(V.T @ PaPb @ V, V.T @ S @ V)
        for a in range(nat):
            ia = np.where(owner == a)[0]
            for b in range(a + 1, nat):
                ib = np.where(owner == b)[0]
                if len(ia) and len(ib):
                    W[a, b] = W[b, a] = (Ppi[np.ix_(ia, ib)] ** 2).sum()
    spin = np.zeros(nat)
    if dm.ndim == 3:                                     # UKS -> Mulliken spin pop
        PS = (dm[0] - dm[1]) @ S
        spin = np.array([np.trace(PS[s0:s1, s0:s1]) for s0, s1 in sl])
    meta = dict(E_eV=float(d['E_eV']), S2=float(d['S2']),
                basis=str(d['basis']) if 'basis' in d.files else basis_fallback)
    return en, apos, W, spin, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jobdir')
    ap.add_argument('--states', default=None, help='comma list of keys (default: all mats/)')
    ap.add_argument('--pi', action='store_true', help='pi-only bond order (p-perp projection)')
    ap.add_argument('--out', default='bo_pyscf', help='output png prefix inside jobdir')
    ap.add_argument('--vmin', type=float, default=None)
    ap.add_argument('--vmax', type=float, default=None)
    args = ap.parse_args()

    mdir = os.path.join(args.jobdir, 'mats')
    keys = args.states.split(',') if args.states else \
        sorted(f[:-4] for f in os.listdir(mdir) if f.endswith('.npz'))
    keys = [k for k in keys if os.path.isfile(os.path.join(mdir, k + '.npz'))]
    print(f'{len(keys)} mats: {keys}')

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    vmin, vmax = (args.vmin, args.vmax) if args.vmin is not None else \
        ((0.0, 2.0) if not args.pi else (0.0, 1.0))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    fig, axs = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6), squeeze=False)
    sm = None
    for ax, key in zip(axs[0], keys):
        en, apos, W, spin, meta = wiberg_bo(os.path.join(mdir, key + '.npz'), pi=args.pi)
        atoms = AtomicSystem(apos=apos, enames=en)
        atoms.findBonds(Rcut=3.0)
        bvals = np.array([W[i, j] for i, j in atoms.bonds]) if len(atoms.bonds) else np.array([])
        if len(bvals):
            _, sm = plot_bond_scalar_map(ax, atoms, apos, bvals, cmap='viridis',
                                         norm=norm, bAtoms=True, bLabels=True, sz=55., lws=5.)
        else:
            ax.scatter(apos[:, 0], apos[:, 1])
            ax.set_aspect('equal'); ax.axis('off')
        if np.abs(spin).max() > 0.02:                    # annotate Mulliken spin
            for i, (p, s) in enumerate(zip(apos, spin)):
                if abs(s) > 0.05:
                    ax.annotate(f'{s:+.2f}', (p[0], p[1]), fontsize=7,
                                color='k', ha='center', va='bottom')
        ax.set_title(f'{key}\nE={meta["E_eV"]:.2f} eV  S2={meta["S2"]:.2f}  [{meta["basis"]}]',
                     fontsize=8)
    if sm is not None:
        fig.colorbar(sm, ax=axs[-1, -1], fraction=0.046,
                     label=('pi bond order' if args.pi else 'Wiberg bond order'))
    fig.tight_layout()
    out = os.path.join(args.jobdir, args.out + ('_pi' if args.pi else '') + '.png')
    fig.savefig(out, dpi=160)
    print('REVIEW: ' + out)


if __name__ == '__main__':
    main()
