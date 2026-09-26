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
                    scctol=1e-7, maxscc=400, verbose=True, lvs=None, nk=(1, 1, 1),
                    k_shift=(0., 0., 0.), mixer=None):
    """DFTBcore single-point -> dict(E_ha, dm, S, eigvals, norb). Work dir keeps geom.xyz/hsd.

    lvs=None  -> finite cluster input (open boundaries: edge artifacts in BO maps!).
    lvs=(3,3) -> periodic GenFormat-S input with nk k-points (same cell as the relax):
                 the DM intra-cell block then carries true periodic coupling, so bond
                 orders are uniform along periodic directions (no x-edge artifact).
    """
    from spammm.quantum.DFTB.DFTBcore import DFTBcore
    from spammm.quantum.DFTB_utils import get_sk_path, write_dftb_input_sp, makeDFTBjob_pbc
    from spammm import atomicUtils as au
    if sk_prefix is None:
        sk_prefix = get_sk_path(sk_set)
    os.makedirs(work_dir, exist_ok=True)
    xyz_path = os.path.join(work_dir, 'geom.xyz')
    au.save_xyz(xyz_path, list(enames), np.asarray(apos, dtype=float))
    if lvs is None:
        write_dftb_input_sp(list(enames), xyz_path, os.path.join(work_dir, 'dftb_in.hsd'),
                            sk_prefix, scctol=scctol, maxscc=maxscc, filling_temp=filling_temp)
    else:
        makeDFTBjob_pbc(list(enames), np.asarray(apos, dtype=float), np.asarray(lvs, dtype=float),
                        fname=os.path.join(work_dir, 'dftb_in.hsd'), sk_set=sk_set, nk=nk,
                        k_shift=k_shift, opt=False, SCCTolerance=scctol, MaxScc=maxscc,
                        Temperature=(300 if filling_temp is None else filling_temp), Mixer=mixer)
    cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        with DFTBcore() as dftb:
            dftb.init('dftb_in.hsd')
            dftb.enable_matrix_collection(dm=True, h=False, s=True)
            e_ha = dftb.run_scf()
            out = {'E_ha': e_ha, 'norb': dftb.get_basis_size(), 'dm': dftb.get_dm_dense()}
            if lvs is None:
                out['S'] = dftb.get_s_dense()
                C, ev = dftb.get_eigvecs_dense()
                out['eigvecs'], out['eigvals'] = C, ev
    finally:
        os.chdir(cwd)
    if lvs is not None:
        # periodic get_s_dense returns zeros -> fetch intra-cell S(0,0) from a cheap
        # finite-cluster SP on the same geometry (overlap is pairwise/geometry-only,
        # the (0,0) block is identical).  NOTE: get_dm_dense is ALSO all-zero for
        # periodic runs -> for real periodic bond orders use pi_bond_orders_pbc()
        # (supercell cluster) instead of this path.
        cl = run_dftbcore_sp(enames, apos, os.path.join(work_dir, 'S_cluster'),
                             sk_prefix=sk_prefix, filling_temp=filling_temp,
                             scctol=scctol, maxscc=maxscc, verbose=False)
        out['S'] = cl['S']
    if verbose:
        print(f"    DFTBcore {os.path.basename(work_dir)}: E={e_ha * HAU2EV:.4f} eV  norb={out['norb']}")
    return out


def _tile_pbc_cluster(enames, apos, bonds, lvs, nrep=(3, 3), center=None):
    """Tile a periodic cell into an nrep[0] x nrep[1] cluster of copies along a1, a2.

    Bonds crossing a cell edge connect the corresponding neighbour copies, so
    seam bonds become real in-cluster bonds (last-copy wraps are dropped).
    Returns (en2, ap2, bo2, idx_c) where idx_c[k] is the cluster-bond index of
    original bond k realized in the CENTER copy.
    """
    nx, ny = nrep
    cx, cy = center if center is not None else (nx // 2, ny // 2)
    n = len(enames)
    en2 = list(enames) * (nx * ny)
    ap2 = np.concatenate([apos + kx * lvs[0] + ky * lvs[1] for ky in range(ny) for kx in range(nx)])
    off = lambda kx, ky: (ky * nx + kx) * n
    finv = np.linalg.inv(np.asarray(lvs).T)                  # cart -> fractional
    sh = np.round((finv @ (apos[bonds[:, 1]] - apos[bonds[:, 0]]).T).T).astype(int)
    bo2, idx_c = [], np.full(len(bonds), -1, dtype=int)
    for k, (i, j) in enumerate(bonds):
        sx, sy = sh[k, 0], sh[k, 1]
        assert sh[k, 2] == 0, 'bond displacement along a3 not supported in 2D tiling'
        for ky in range(ny):
            jy = ky + sy
            if not 0 <= jy < ny:
                continue
            for kx in range(nx):
                jx = kx + sx
                if not 0 <= jx < nx:
                    continue
                bo2.append((i + off(kx, ky), j + off(jx, jy)))
                if kx == cx and ky == cy:
                    idx_c[k] = len(bo2) - 1
    assert (idx_c >= 0).all(), 'center-copy realization missing for some bond'
    return en2, ap2, np.asarray(bo2), idx_c


# ---------------------------------------------------------------------------
# Periodic pi bond orders via the dftb+ BINARY (DFTBcore's get_dm_dense /
# get_s_dense return zeros for periodic inputs in this build — the matrices
# are only reachable through the files DFTB+ itself writes):
#   run 'hs' : Options{WriteHS, WriteRealHS} -> oversqr.dat = S(k) (instant, no SCF)
#              + overreal.dat = sparse S(0,R) blocks with per-pair ICELL cell offsets
#   run 'scf': Analysis{WriteEigenvectors}  -> eigenvec.bin = C(k) per k (stream:
#              int32 runId, then per (k,spin) norb columns of norb complex128)
#              + band.out = eps(k), f(k), k-weights
# P(k) = C_k f_k C_k^dagger ; B(k) = Spi(k)^1/2 Ppi(k) Spi(k)^1/2  (Lowdin per k)
# DFTB+ stores k-space matrices as square[j,i](k) = sum_R e^{+ik.R} sparse(i@0, j@R)
# where R is the image cell assigned by its (folded-coordinate) neighbour list — NOT
# predictable from the input geometry (same bond lands on different R per system).
# overreal.dat's ICELL column is the authoritative per-pair offset:
#   BO(i,j) = Re sum_k w_k e^{-i2pi kf.ICELL} B(k)_{iAt1-1, iAt2f-1}
# eigenvec.bin stream is read as conj().T — verified by C^dagger S C = I at all k.
# ---------------------------------------------------------------------------
def read_band_out(fname):
    """band.out -> (kweights[nk], eigs[nk,norb], fills[nk,norb]); spin-unpolarized."""
    kw, eigs, fills, ce, cf = [], [], [], None, None
    for line in open(fname):
        if 'KPT' in line:
            if ce is not None:
                eigs.append(ce); fills.append(cf)
            kw.append(float(line.split('KWEIGHT')[1])); ce, cf = [], []
        elif ce is not None and line.split():
            p = line.split(); ce.append(float(p[1])); cf.append(float(p[2]))
    eigs.append(ce); fills.append(cf)
    return np.array(kw), np.array(eigs), np.array(fills)


def read_eigenvec_bin(fname, nk, norb):
    """eigenvec.bin -> C[nk, norb, norb], C[k, orb, i] = i-th eigenvector (AO basis)."""
    with open(fname, 'rb') as f:
        runid = np.fromfile(f, np.int32, 1)
        C = np.fromfile(f, np.complex128)
    assert C.size == nk * norb * norb, f'{fname}: expected {nk*norb*norb} cplx, got {C.size}'
    return C.reshape(nk, norb, norb).conj().transpose(0, 2, 1)   # conj().T: verified vs C†SC=I


def subcell_group_indices(apos_ideal, lvs, enames, ncells, nao=None, strict=True):
    """Atom->(equivalence group g, subcell index m') for x-unfolding.

    Groups atoms identical under translations R_m' = m'*lvs[0]/ncells using the
    IDEAL geometry (relaxed coords keep the same atom indexing, so the grouping
    is exact).  Returns IDX[(g,m',a)] = flat AO index (atom-major orbitals),
    with shape (ngroups, ncells, max_nao) and a mask; groups may have differing
    nao only if elements differ (they don't within a group by construction).
    """
    nao = nao or {'H': 1, 'C': 4, 'N': 4, 'O': 4}
    apos = np.asarray(apos_ideal, float)
    nat = len(apos)
    pitch = lvs[0, 0] / ncells
    assert strict or nat >= ncells
    if strict:
        assert nat % ncells == 0
    # equivalence: same (y,z,element), x differs by integer*pitch (mod wrap)
    used = np.zeros(nat, bool)
    glist = []
    for i in range(nat):
        if used[i]:
            continue
        dx = apos[:, 0] - apos[i, 0]
        f = (dx / pitch + 0.5) % 1.0 - 0.5                            # frac offset in [-.5,.5)
        sel = (np.abs(f) < 0.05) & (np.abs(apos[:, 1] - apos[i, 1]) < 0.05) & \
              (np.abs(apos[:, 2] - apos[i, 2]) < 0.05) & (np.array(enames) == enames[i])
        grp = np.where(sel)[0]
        if len(grp) != ncells:
            # atoms unique to the supercell (defect adsorbates etc.) — excluded
            # from the primitive-channel projection (strict=False)
            assert not strict, f'equivalence group of atom {i} has {len(grp)} members, expected {ncells}'
            used[grp] = True
            continue
        mm = np.floor(apos[grp, 0] / pitch + 1e-6).astype(int) % ncells  # absolute subcell index
        assert len(set(mm)) == ncells
        glist.append({int(mp): int(j) for mp, j in zip(mm, grp)})
        used[grp] = True
    glist.sort(key=lambda d: min(d.values()))
    for d in glist:
        assert len(d) == ncells, 'incomplete equivalence group (not an ncells supercell?)'
    off = np.zeros(nat + 1, int)
    for i, e in enumerate(enames):
        off[i + 1] = off[i] + nao[e]
    naomax = max(nao[enames[d[0]]] for d in glist)
    IDX = np.full((len(glist), ncells, naomax), -1, int)
    for g, d in enumerate(glist):
        for mp, i in d.items():
            for a in range(nao[enames[i]]):
                IDX[g, mp, a] = off[i] + a
    return IDX


def unfold_spectral_weights_DEPRECATED(C, kf_x, IDX, ncells):
    """DEPRECATED — wrong metric for nonorthogonal bases; use unfold_T_weights.

    This plain subcell-Fourier weight assumes Euclidean-orthonormal eigenvectors.
    DFTB (mio) eigenvectors are S-orthonormal with |S_offdiag| up to ~0.44 and
    GPAW dzp up to ~0.5, so the formula leaks ~20-30% of each state's weight
    into adjacent primitive channels even for exactly periodic systems, with a
    strong artificial k-dependence (DFTB xscanv 0H: Wmax med ~1.0 near the BZ
    centre collapsing to ~0.45 at the zone edge — pure artifact, the geometry
    is exactly periodic).  With unfold_T_weights the same states give
    Wmax med 1.000, 100% sharp.

    Kept (not deleted) for regression reference and for the legacy drivers
    testplot_unfold.py / testplot_ribbon.py --unfold, which still call it.

    C[nk,norb,nband]: eigenvectors from eigenvec.bin (C[k,orb,n]).
    kf_x[nk]: fractional supercell kx in [0,1).
    IDX from subcell_group_indices.  Primitive k-channels: q_m = (kf+m)/ncells,
    weight w_m = (1/ncells) * sum_mu |sum_m' c[mu+m'] e^{-i2pi m' q_m}|^2
    (sum_m w_m = |c|^2 per eigenstate — Parseval of the subcell DFT).

    Returns q[nk,ncells] (primitive fractional kx in [0,1)) and
    W[nk,ncells,nband] (spectral weight, each eigenstate total = 1).
    """
    nk, norb, nband = C.shape
    ng, nc, naomax = IDX.shape
    assert nc == ncells
    valid = IDX >= 0
    IDXc = np.where(valid, IDX, 0)
    kf = np.asarray(kf_x, float)
    q = (kf[:, None] + np.arange(ncells)[None, :]) / ncells            # [nk,ncells] in [0,1)
    W = np.empty((nk, ncells, nband))
    mrange = np.arange(ncells)
    for ik in range(nk):
        A = C[ik][IDXc]                                              # [ng,nc,naomax,nband]
        A = np.where(valid[..., None], A, 0.0)
        phase = np.exp(-2j * np.pi * mrange[None, :] * q[ik][:, None])       # [q,m']
        v = np.einsum('gcan,qc->gaqn', A, phase)                     # sum over m' (=c)
        W[ik] = (np.abs(v) ** 2).sum(axis=(0, 1)) / ncells           # [q,nband]
    W /= W.sum(axis=1, keepdims=True)                                # normalize per eigenstate (C is S-orthonormal, not Euclidean)
    return q, W


def unfold_T_weights(C, S_k, kf_x, IDX, ncells):
    """Channel weights from the spectral projectors of the subcell-translation
    operator T (shift by one primitive pitch along x) — the correct weight for
    NONORTHOGONAL bases (GPAW dzp, |S_offdiag|~0.5): the plain subcell-DFT
    formula is a Euclidean-space projector and leaks ~30% of the weight into
    adjacent channels.

        w_j(n) = (1/ncells) * sum_m exp(-i2pi q_j m) <psi_n|T^m|psi_n>
                 with <psi|T^m|psi> = c^+ S Top^m c          (m = 0..ncells-1)

    Top maps AO (g,m,a) -> (g,m+1,a) as a permutation; the wrap column
    m=ncells-1 -> 0 carries the Bloch factor exp(+i2pi*kf) (empirical for the
    hs.npz convention).  For an exactly periodic cell every eigenstate is a T
    eigenstate -> w ~1 in a single channel (measured 0.88..1.0 median on dzp
    vs ~0.7 for the DFT formula); defect-localized states dephase -> their
    weight spreads over channels, as it should.  sum_j w_j = <psi|S|psi> = 1
    exactly (Parseval over the spectral measure).

    C[nk,norb,nband] (kpt.C_nM layout transposed), S_k[nk,norb,norb].
    Returns q[nk,ncells], W[nk,ncells,nband] (per-state sums to 1).
    """
    nk, norb, nband = C.shape
    ng, nc, naomax = IDX.shape
    assert nc == ncells
    valid = IDX >= 0
    assert (valid == valid[:, :1, :]).all(), 'equivalence groups must be complete across subcells'
    kf = np.asarray(kf_x, float)
    q = (kf[:, None] + np.arange(ncells)[None, :]) / ncells           # [nk,ncells]
    W = np.zeros((nk, ncells, nband))
    for ik in range(nk):
        Top = np.zeros((norb, norb), complex)
        for m in range(ncells):
            dst = IDX[:, (m + 1) % ncells, :][valid[:, (m + 1) % ncells, :]]
            src = IDX[:, m, :][valid[:, m, :]]
            Top[dst, src] = np.exp(2j * np.pi * kf[ik]) if m == ncells - 1 else 1.0
        SCt = np.asarray(S_k[ik]) @ C[ik]                             # S c_n, [norb,nband]
        Tm = np.eye(norb, dtype=complex)
        for m in range(ncells):
            d = np.einsum('Mn,Mn->n', C[ik].conj(), Tm @ SCt)         # <n|T^m|n>, [nband]
            W[ik] += np.real(np.exp(-2j * np.pi * q[ik][:, None] * m) * d[None, :]) / ncells
            Tm = Tm @ Top
    return q, W


def unfold_projector_weights(C, S_k, kf_x, IDX, ncells):
    """SUPERSEDED by unfold_T_weights — kept for reference.

    The oblique channel subspaces overlap near the zone edges (the folded
    isometry F_j is only approximately S-orthonormal there), so the per-state
    normalization below divides away part of the true channel purity
    (measured Wmax med ~0.53 vs 0.88..1.0 for the T-spectral projector).
    unfold_T_weights is both sharper and cheaper — use it instead.

    Channel weights via the S-metric projector onto primitive-Bloch subspaces.

    For a NONORTHOGONAL basis (GPAW dzp, |S_offdiag| ~0.5 on diffuse shells) the
    plain subcell-DFT weight of unfold_spectral_weights_DEPRECATED is not a projector —
    it leaks weight into adjacent channels even for exactly periodic systems.
    Here each primitive channel q_j = (k_s + j)/ncells owns the function-space
    subspace spanned by the folded isometry F_j (columns = primitive orbitals),
    and the weight is the oblique projector expectation

        w_j(n) = c_n^+ S F_j (F_j^+ S F_j)^{-1} F_j^+ S c_n

    For an exactly periodic cell every eigenstate IS a primitive Bloch state,
    so w ~1 in a single channel (verified ~0.96 median on dzp vs ~0.7 for the
    DFT formula).  Defect orbitals absent from IDX are simply outside the
    projected subspace -> their states spread over channels (localized in k).

    C[nk,norb,nband], S_k[nk,norb,norb] (same k, same AO gauge as C).
    Returns q[nk,ncells], W[nk,ncells,nband] (per-state normalized).
    """
    nk, norb, nband = C.shape
    ng, nc, naomax = IDX.shape
    assert nc == ncells
    valid = IDX >= 0
    assert (valid == valid[:, :1, :]).all(), 'equivalence groups must be complete across subcells'
    kf = np.asarray(kf_x, float)
    q = (kf[:, None] + np.arange(ncells)[None, :]) / ncells           # [nk,ncells]
    npos = int(valid[:, 0, :].sum())                                  # primitive orbitals
    prow = np.arange(npos)
    W = np.empty((nk, ncells, nband))
    for ik in range(nk):
        SC = np.asarray(S_k[ik]) @ C[ik]                              # S c_n, [norb,nband]
        for j in range(ncells):
            F = np.zeros((norb, npos), complex)
            for m in range(ncells):
                F[IDX[:, m, :][valid[:, m, :]], prow] = np.exp(2j * np.pi * q[ik, j] * m) / np.sqrt(ncells)
            SF = np.asarray(S_k[ik]) @ F
            A = F.conj().T @ SC                                       # F^+ S c_n -> [npos,nband]
            Gi = np.linalg.pinv(F.conj().T @ SF)                      # (F^+ S F)^{-1}
            W[ik, j] = np.einsum('pn,pq,qn->n', A.conj(), Gi, A).real
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    return q, W


def read_sqr_dat(fname):
    """oversqr.dat / hamsqrN.dat -> complex M[nk, norb, norb] (rows of re,im pairs)."""
    toks = open(fname).read().split()
    i = toks.index('NKPOINT') + 1
    isreal, norb, nk = toks[i] == 'T', int(toks[i + 1]), int(toks[i + 2])
    M = np.empty((nk, norb, norb), dtype=float if isreal else complex)
    p = i + 3
    for k in range(nk):
        assert toks[p] == '#' and toks[p + 1] == 'IKPOINT', f'{fname}: bad k-block header at token {p}'
        assert toks[p + 3] == '#' and toks[p + 4] == 'MATRIX', f'{fname}: missing MATRIX at token {p}'
        p += 5
        v = np.array([float(x) for x in toks[p:p + norb * norb * (1 if isreal else 2)]])
        M[k] = v.reshape(norb, norb) if isreal else (v[0::2] + 1j * v[1::2]).reshape(norb, norb)
        p += norb * norb * (1 if isreal else 2)
    return M


def read_overreal_pairs(fname):
    """overreal.dat -> dict {(iAt1-1, iAt2f-1): [ICELL, ...]} mapping each stored
    neighbour pair (stored once per unordered pair, iAt1<iAt2f order) to its cell
    offset(s). One pair may appear at several ICELL (images in neighbour cells)."""
    import re
    blocks = re.findall(r'#\s+IATOM1\s+INEIGH\s+IATOM2F\s+ICELL\(1\)\s+ICELL\(2\)\s+ICELL\(3\)\s*\n'
                        r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)', open(fname).read())
    pairs = {}
    for i1, _n, i2, c1, c2, c3 in blocks:
        pairs.setdefault((int(i1) - 1, int(i2) - 1), []).append(np.array([int(c1), int(c2), int(c3)]))
    return pairs


def _run_dftbplus_binary(enames, apos, lvs, work_dir, extra_hsd, nk=(1, 1, 1), k_shift=(0., 0., 0.),
                         sk_set=None, filling_temp=300., scctol=1e-7, maxscc=400, mixer=None):
    """Write PBC input + run the dftb+ binary in work_dir; returns (E_ha, stdout)."""
    import subprocess
    from spammm.quantum.DFTB_utils import makeDFTBjob_pbc
    os.makedirs(work_dir, exist_ok=True)
    makeDFTBjob_pbc(list(enames), np.asarray(apos, float), np.asarray(lvs, float),
                    fname=os.path.join(work_dir, 'dftb_in.hsd'), sk_set=sk_set, nk=nk,
                    k_shift=k_shift, opt=False, SCCTolerance=scctol, MaxScc=maxscc,
                    Temperature=filling_temp, Mixer=mixer, extra_hsd=extra_hsd)
    r = subprocess.run(['dftb+'], cwd=work_dir, capture_output=True, text=True)
    open(os.path.join(work_dir, 'OUT'), 'w').write(r.stdout)
    hs_exit = 'Hamilton/Overlap written, exiting program' in r.stdout   # WriteHS stops before SCF by design
    assert (r.returncode == 0 or hs_exit) and 'ERROR' not in r.stdout, f'dftb+ failed in {work_dir}:\n{r.stdout[-2000:]}'
    assert 'NOT converged' not in r.stdout, f'SCC not converged in {work_dir}:\n{r.stdout[-2000:]}'
    m = [l for l in r.stdout.split('\n') if 'Total Energy:' in l]
    return (float(m[-1].split()[2]) if m else np.nan), r.stdout


def pi_bond_orders_pbc(enames, apos, bonds, lvs, work_dir, nk=(1, 1, 1), k_shift=(0., 0., 0.),
                       sk_set=None, filling_temp=300., scctol=1e-7, maxscc=400, mixer=None, verbose=True):
    """Periodic pi bond orders: dftb+ binary SP -> eigenvec.bin/oversqr.dat -> per-k Lowdin.

    Two dftb+ runs (same cell/k-mesh/protocol as the relax; ~seconds total):
      work_dir/hs  -> oversqr.dat = S(k)   (no SCF needed)
      work_dir/scf -> eigenvec.bin = C(k), band.out = eps,f,weights  (one SCF)
    Returns (bpi[len(bonds)] incl. real seam-bond values, info dict with E_ha, P00, B00).
    """
    import re
    bonds = np.asarray(bonds); apos = np.asarray(apos, float); lvs = np.asarray(lvs, float)
    _run_dftbplus_binary(enames, apos, lvs, os.path.join(work_dir, 'hs'), 'Options {\n  WriteHS = Yes\n  WriteRealHS = Yes\n}\n',
                         nk=nk, k_shift=k_shift, sk_set=sk_set, filling_temp=filling_temp, scctol=scctol, maxscc=maxscc, mixer=mixer)
    E_ha, out = _run_dftbplus_binary(enames, apos, lvs, os.path.join(work_dir, 'scf'), 'Analysis {\n  WriteEigenvectors = Yes\n}\n',
                                     nk=nk, k_shift=k_shift, sk_set=sk_set, filling_temp=filling_temp, scctol=scctol, maxscc=maxscc, mixer=mixer)
    wd = os.path.join(work_dir, 'scf')
    kw, eigs, fills = read_band_out(os.path.join(wd, 'band.out'))
    n_k, norb = fills.shape
    C = read_eigenvec_bin(os.path.join(wd, 'eigenvec.bin'), n_k, norb)
    Sk = read_sqr_dat(os.path.join(work_dir, 'hs', 'oversqr.dat'))
    assert Sk.shape[0] == n_k and Sk.shape[1] == norb, f'S(k) shape {Sk.shape} vs ({n_k},{norb})'
    n_up = float(re.search(r'Nr\. of up electrons:\s*(\S+)', out).group(1))
    ne = (kw[:, None] * fills).sum()
    assert abs(ne - 2 * n_up) < 1e-3, f'electron count {ne:.4f} != 2x{n_up}'
    Pk = np.einsum('kui,ki,kvi->kuv', C, fills, C.conj())
    layout = ao_layout(enames)
    n = mol_plane_normal(apos, np.array([e != 'H' for e in enames]))
    pa = layout['pi_atoms']
    Spi = np.stack([pi_project(Sk[k], layout['pxyz'], n, pa) for k in range(n_k)])
    Ppi = np.stack([pi_project(Pk[k], layout['pxyz'], n, pa) for k in range(n_k)])
    Bk = np.empty_like(Ppi)
    for k in range(n_k):
        ew, ev = np.linalg.eigh(Spi[k])
        assert ew.min() > 1e-8, f'pi overlap not positive definite at k={k}: min eig {ew.min()}'
        Bk[k] = (ev * np.sqrt(ew)) @ ev.conj().T @ Ppi[k] @ (ev * np.sqrt(ew)) @ ev.conj().T
    # per-pair ICELL from overreal.dat: the image cell where DFTB+ stored the pair
    pairs = read_overreal_pairs(os.path.join(work_dir, 'hs', 'overreal.dat'))
    lines = out.split('\n')
    i0 = next(i for i, l in enumerate(lines) if 'K-points and weights' in l)
    kf = np.array([list(map(float, l.split(':')[-1].split()[:3])) for l in lines[i0:i0 + n_k]])
    bpi = np.empty(len(bonds))
    pos = {a: t for t, a in enumerate(pa)}
    for b, (i, j) in enumerate(bonds):
        if i not in pos or j not in pos:
            bpi[b] = np.nan; continue
        key = (i, j) if (i, j) in pairs else (j, i)
        assert key in pairs, f'bond {(i, j)} absent from {work_dir}/hs/overreal.dat neighbour list'
        i1, i2 = key
        # pick the image actually forming the bond = min distance i@0 -- j@ICELL
        dmin, icell = min(((np.linalg.norm(apos[j] + R @ lvs - apos[i]), R) for R in pairs[key]), key=lambda t: t[0])
        ph = np.exp(-1j * 2 * np.pi * kf @ icell)
        bpi[b] = np.einsum('k,k,k->', kw, ph, Bk[:, pos[i1], pos[i2]]).real
    if verbose:
        print(f"    PBC pi-BO {os.path.basename(work_dir)}: E={E_ha * HAU2EV:.4f} eV  Ne={ne:.1f}  nk={n_k}")
    return bpi, {'E_ha': E_ha, 'P00': np.einsum('k,kuv->uv', kw, Pk).real, 'Bk': Bk, 'Ne': ne}


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
