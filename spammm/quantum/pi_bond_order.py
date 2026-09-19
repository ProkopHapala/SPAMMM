"""pi_bond_order.py — pi bond orders from DFTB+ density matrices via DFTBcore.

Essence: DFTBcore SCF (minimal HSD, no Driver) -> dense density matrix P and overlap S ->
project onto the per-atom pi AO (p-triplet contracted with molecular plane normal n) ->
Lowdin-orthogonalize Ppi' = Spi^1/2 Ppi Spi^1/2 -> per-bond pi bond order Ppi'[i,j].

AO layout (DFTB+ tesseral m=-l..+l, see dftbp/type/orbitals.F90):
  sp-basis heavy atom (C,N,O): [s, py, pz, px] -> p triplet at offset+1..3 (x=+3,y=+1,z=+2)
  H: [s].
pi AO per atom = n . (px,py,pz); reduces to pz for flat xy molecules.

Caveats:
- DFTBcore reads `dftb_in.hsd` from CWD -> each call chdirs to work_dir (like AFM_utils).
- Raw P_ij is in the non-orthogonal AO basis (not a transferable bond order) -> always
  use lowdin_pi_dm(); parity check via energy against the dftb+ OUT that made the geometry.
- Match the Fermi `filling_temp` of the data-generating run (corner scan used 600 K).
"""
import os
import numpy as np

HAU2EV = 27.211386245988

# tesseral p-triplet local indices in (x,y,z) order for [s,py,pz,px] layout
_P_XYZ_LOCAL = (3, 1, 2)


def mol_plane_normal(apos, mask=None):
    """Unit normal of the best-fit plane through atoms (SVD of centered positions)."""
    p = np.asarray(apos, dtype=float)
    if mask is not None:
        p = p[mask]
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c)
    return vt[2]


def ao_layout(enames, max_am=None):
    """AO offsets per atom and p-triplet AO indices (x,y,z order) per atom.

    max_am: {element: l_max}; default s for H, p otherwise (3ob sp basis).
    Returns dict(offsets=[nat+1], norb=[nat], pxyz=(nat,3) int or -1, pi_atoms=[npi]).
    """
    if max_am is None:
        max_am = {}
    offs = [0]
    norb = []
    pxyz = -np.ones((len(enames), 3), dtype=int)
    pi_atoms = []
    for i, e in enumerate(enames):
        lmax = max_am.get(e, 0 if e == 'H' else 1)
        n = sum(2 * l + 1 for l in range(lmax + 1))   # shells s,(p),(d) in order
        norb.append(n)
        offs.append(offs[-1] + n)
        if lmax >= 1:
            o = offs[-2]                              # p shell: [py,pz,px] right after s
            pxyz[i] = [o + _P_XYZ_LOCAL[0], o + _P_XYZ_LOCAL[1], o + _P_XYZ_LOCAL[2]]
            pi_atoms.append(i)
    return {'offsets': np.array(offs, dtype=int), 'norb': np.array(norb, dtype=int),
            'pxyz': pxyz, 'pi_atoms': np.array(pi_atoms, dtype=int)}


def pi_project(M, pxyz, normal, pi_atoms):
    """Contract AO matrix onto per-atom pi AOs:  Mpi[i,j] = sum_ab n_a M[px_i[a], px_j[b]] n_b."""
    n = np.asarray(normal, dtype=float)
    idx = pxyz[pi_atoms]                             # (npi,3) AO idx in x,y,z order
    sub = M[np.ix_(idx.ravel(), idx.ravel())].reshape(len(pi_atoms), 3, len(pi_atoms), 3)
    return np.einsum('a,iajb,b->ij', n, sub, n)


def lowdin_orthogonalize(Ppi, Spi, eps=1e-8):
    """P~ = Spi^{1/2} Ppi Spi^{1/2}  (Lowdin)."""
    ew, ev = np.linalg.eigh(Spi)
    assert ew.min() > eps, f"pi overlap not positive definite: min eig {ew.min()}"
    S12 = (ev * np.sqrt(ew)) @ ev.T
    return S12 @ Ppi @ S12


def pi_bond_order_matrix(P, S, layout, normal=None, apos=None, enames=None):
    """Full pipeline: AO-basis P,S -> Lowdin pi bond-order matrix over pi_atoms.

    normal=None -> auto from molecular plane (needs apos; heavy-atom mask from layout).
    Returns (Ppi_ld [npi,npi], pi_atoms).
    """
    if normal is None:
        assert apos is not None, "normal or apos required"
        heavy = np.array([e != 'H' for e in enames]) if enames is not None else layout['norb'] > 1
        normal = mol_plane_normal(apos, heavy)
    Ppi = pi_project(P, layout['pxyz'], normal, layout['pi_atoms'])
    Spi = pi_project(S, layout['pxyz'], normal, layout['pi_atoms'])
    return lowdin_orthogonalize(Ppi, Spi), layout['pi_atoms']


def bond_orders_from_pmat(Ppi_ld, pi_atoms, bonds):
    """pi bond order per bond (i,j): Ppi_ld[i,j]; nan if either atom lacks a pi AO."""
    pos = {a: k for k, a in enumerate(pi_atoms)}
    bo = np.full(len(bonds), np.nan)
    for b, (i, j) in enumerate(bonds):
        if i in pos and j in pos:
            bo[b] = Ppi_ld[pos[i], pos[j]]
    return bo


def bond_lengths(apos, bonds):
    """Bond length [A] per bond."""
    d = apos[np.asarray(bonds)[:, 0]] - apos[np.asarray(bonds)[:, 1]]
    return np.linalg.norm(d, axis=1)


def run_dftbcore_sp(enames, apos, work_dir, sk_prefix=None, sk_set=None, filling_temp=None,
                    scctol=1e-7, maxscc=400, verbose=True):
    """DFTBcore single-point -> dict(E_ha, dm, S, eigvals, norb). Work dir keeps geom.xyz/hsd."""
    from spammm.quantum.DFTB.DFTBcore import DFTBcore
    from spammm.quantum.DFTB_utils import get_sk_path, write_dftb_input_sp
    from spammm import atomicUtils as au
    if sk_prefix is None:
        sk_prefix = get_sk_path(sk_set)
    os.makedirs(work_dir, exist_ok=True)
    xyz_path = os.path.join(work_dir, 'geom.xyz')
    au.save_xyz(xyz_path, list(enames), np.asarray(apos, dtype=float))
    write_dftb_input_sp(list(enames), xyz_path, os.path.join(work_dir, 'dftb_in.hsd'),
                        sk_prefix, scctol=scctol, maxscc=maxscc, filling_temp=filling_temp)
    cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        with DFTBcore() as dftb:
            dftb.init('dftb_in.hsd')
            dftb.enable_matrix_collection(dm=True, h=False, s=True)
            e_ha = dftb.run_scf()
            out = {'E_ha': e_ha, 'norb': dftb.get_basis_size(),
                   'dm': dftb.get_dm_dense(), 'S': dftb.get_s_dense()}
            C, ev = dftb.get_eigvecs_dense()
            out['eigvecs'], out['eigvals'] = C, ev
    finally:
        os.chdir(cwd)
    if verbose:
        print(f"    DFTBcore {os.path.basename(work_dir)}: E={e_ha * HAU2EV:.4f} eV  norb={out['norb']}")
    return out


def plot_bond_scalar_map(ax, atoms, apos, bvals, bonds=None, cmap='viridis', norm=None,
                         mask=None, sz=60., bLabels=False, bAtoms=True, lws=3.0):
    """Draw molecule on ax: atoms + grey skeleton, selected bonds colored by scalar.

    bvals: scalar per bond (len==len(bonds)); mask selects which bonds are colored
    (rest drawn thin grey). Returns the LineCollection of colored bonds (for colorbar).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from spammm import plotUtils as pu, elements
    apos = np.asarray(apos, dtype=float)
    bonds = np.asarray(atoms.bonds if bonds is None else bonds)
    if mask is None:
        mask = np.isfinite(bvals)
    enames = [e.split('_')[0] for e in atoms.enames]
    colors_a = [elements.ELEMENT_DICT[e][8] for e in enames]
    sizes_a = [elements.ELEMENT_DICT[e][6] * sz for e in enames]
    plt.sca(ax)
    if bAtoms:
        pu.plotAtoms(apos=apos, es=enames, sizes=sizes_a, colors=colors_a, marker='o', axes=(0, 1))
    if (~mask).any():
        pu.plotBonds(links=bonds[~mask], ps=apos, colors=(0.7, 0.7, 0.7, 0.6), lws=1.2, axes=(0, 1))
    norm = norm or mcolors.Normalize(vmin=np.nanmin(bvals[mask]), vmax=np.nanmax(bvals[mask]))
    rgba = plt.get_cmap(cmap)(norm(np.asarray(bvals)[mask]))
    lc = pu.plotBonds(links=bonds[mask], ps=apos, colors=rgba, lws=lws, axes=(0, 1))
    if bLabels:
        labs = [f"{v:.2f}" for v in np.asarray(bvals)[mask]]
        pu.plotBonds(links=bonds[mask], ps=apos, colors='none', lws=0, labels=labs, fnsz=7, axes=(0, 1))
    ax.set_aspect('equal'); ax.axis('off'); ax.margins(0.15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    return lc, sm
