"""gpaw_hs.py — analysis helpers for hs.npz exports (see gpaw_hs_export.export_hs).

hs.npz contents: pos_ac, cell_cv, Z_a, kpts_kc (fractional), w_k,
eps_kn/occ_kn [eV, occ 0..2], C_knM, S_kMM, H_kMM [eV], rho_kMM (per-k DM,
unweighted: rho(R) = sum_k w_k e^{ikR} rho_k), ao_atom_M/ao_l_M/ao_m_M,
E_total, E_fermi, n_electrons.

pi bond orders: same per-k Lowdin scheme as pi_bond_order.pi_bond_orders_pbc,
but the pi subspace is MULTI-CHANNEL — GPAW dzp gives each heavy atom TWO l=1
shells (2 pi channels) + an l=2 polarization shell (ignored); H gets none
(mio parity).  Real-harmonic order inside an l=1 shell is m=-1,0,+1 = (y,z,x)
— identical to DFTB+ tesseral ordering.  The diffuse 2nd shell makes the pi
overlap nearly singular, so pi_bond_orders_hs defaults to single=True
(dominant channel per atom = DFTB single-zeta analog).  The cell image of
each stored pair is chosen by dominant Fourier component, not by geometry —
GPAW wraps atom positions mod cell like DFTB's overreal ICELL.
"""
import numpy as np

from spammm.quantum.pi_bond_order import mol_plane_normal

Z2E = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}


def load_hs(path):
    """hs.npz -> plain dict of arrays."""
    d = np.load(path)
    return {k: d[k] for k in d.files}


def pi_channels(hs):
    """Pi-channel AO triplets in (x,y,z) order + channel->atom map.

    One channel per l=1 shell on each non-H atom (dzp: 2 per heavy atom).
    Returns (tris[nch,3] AO indices, chat[nch] atom index).
    """
    ao_a, ao_l, ao_m, Z = hs['ao_atom_M'], hs['ao_l_M'], hs['ao_m_M'], hs['Z_a']
    tris, chat = [], []
    for a in np.unique(ao_a):
        if Z[a] == 1:
            continue                                          # H: no pi channel
        idx = np.where((ao_a == a) & (ao_l == 1))[0]
        assert len(idx) % 3 == 0, f'atom {a}: {len(idx)} p AOs not a multiple of 3'
        for s in range(0, len(idx), 3):
            sh = idx[s:s + 3]
            assert (ao_m[sh] == [-1, 0, 1]).all(), f'atom {a}: unexpected m order {ao_m[sh]}'
            tris.append(sh[[2, 0, 1]])                        # (y,z,x) -> (x,y,z)
            chat.append(a)
    return np.asarray(tris), np.asarray(chat)


def pi_project_ch(M, tris, n):
    """Multi-channel pi projection: Mpi[c,d] = n . M[tris[c], tris[d]] . n."""
    nch = len(tris)
    sub = M[np.ix_(tris.ravel(), tris.ravel())].reshape(nch, 3, nch, 3)
    return np.einsum('a,iajb,b->ij', n, sub, n)


def band_energy(hs):
    """Eigenvalue-sum (band) energy E_band = sum_k w_k Tr(H_k rho_k) [eV]."""
    return np.einsum('k,kij,kji->', hs['w_k'], hs['H_kMM'], hs['rho_kMM']).real


def pi_bond_orders_hs(hs, bonds, apos=None, lvs=None, normal=None, single=True):
    """Periodic pi bond orders from an hs.npz dict -> (bpi[len(bonds)], info).

    Per k: B(k) = Spi^{1/2} Ppi Spi^{1/2}  (Lowdin on the pi subspace).
    Per bond (i,j): BO = Re sum_k w_k e^{+i 2pi kf.R*} sum_{ci~i,cj~j} B(k)[ci,cj].

    R* (cell image of the stored pair) is NOT predictable from the geometry:
    GPAW wraps atom positions mod the cell, so bonds touching atoms at x~0
    land on R=+-1 while interior bonds sit at R=0 (same issue as the DFTB
    overreal.dat ICELL).  Since the DM element is nonzero only for the bonded
    image, we scan R in {-1,0,+1} (x is the only periodic axis) and take the
    dominant Fourier component per bond.

    single=True: keep only the dominant pi channel per atom (valence pz,
    Spi~1.0) -- the DFTB single-zeta analog.  All-channel dzp makes Spi nearly
    singular (diffuse 2nd shell, min eig ~0.002) and Löwdin amplifies it.
    """
    apos = np.asarray(hs['pos_ac'] if apos is None else apos, float)
    lvs = np.asarray(hs['cell_cv'] if lvs is None else lvs, float)
    bonds = np.asarray(bonds)
    tris, chat = pi_channels(hs)
    n = normal if normal is not None else mol_plane_normal(apos, hs['Z_a'] != 1)
    kf, wk = np.asarray(hs['kpts_kc']), np.asarray(hs['w_k'])
    nk = len(wk)
    if single:                                                   # dominant pi channel per atom
        Ppi_av = sum(wk[k] * pi_project_ch(hs['rho_kMM'][k], tris, n) for k in range(nk)).real
        sel = np.array([np.where(chat == a)[0][np.argmax(np.diag(Ppi_av)[np.where(chat == a)[0]])] for a in np.unique(chat)])
        tris, chat = tris[sel], chat[sel]
    nch = len(tris)
    Bk = np.empty((nk, nch, nch), complex)
    for k in range(nk):
        Spi = pi_project_ch(hs['S_kMM'][k], tris, n)
        Ppi = pi_project_ch(hs['rho_kMM'][k], tris, n)
        ew, ev = np.linalg.eigh(Spi)
        assert ew.min() > 1e-8, f'pi overlap not positive definite at k={k}: min eig {ew.min()}'
        S12 = (ev * np.sqrt(ew)) @ ev.conj().T
        Bk[k] = S12 @ Ppi @ S12
    # real-space bond matrix at the 3 x-images: B(0,R) = sum_k w_k e^{+i2pi kf.R} B(k)
    ph = np.exp(1j * 2 * np.pi * np.outer(kf[:, 0], [-1, 0, 1]))  # (nk,3)
    Bs = np.einsum('k,ks,kij->sij', wk, ph, Bk)                  # (3,nch,nch)
    chof = {a: np.where(chat == a)[0] for a in np.unique(chat)}
    bpi = np.full(len(bonds), np.nan)
    for b, (i, j) in enumerate(bonds):
        if i not in chof or j not in chof:
            continue
        blk = Bs[:, chof[i], :][:, :, chof[j]].reshape(3, -1).sum(1)  # channel-block sum per R
        bpi[b] = blk[np.argmax(np.abs(blk))].real              # dominant image = bonded pair
    return bpi, {'E_band': band_energy(hs), 'Bk': Bk, 'tris': tris, 'chat': chat,
                 'nch_per_atom': np.bincount(chat)}
