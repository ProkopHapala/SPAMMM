"""Export LCAO Hamiltonian / overlap / density matrix / orbitals from GPAW.

Self-contained (numpy+gpaw+ase only — must run on Metacentrum without spammm).
Used by baked job.py files; also importable for local tests.

Export contents (hs.npz):
  pos_ac, cell_cv, Z_a                 geometry
  kpts_kc, w_k                         IBZ k-points (reduced) + weights
  eps_kn, occ_kn                       eigenvalues [eV], occupations (0..2)
  C_knM                                LCAO coefficients  psi_n(k) = sum_M C_nM phi_M
  S_kMM, H_kMM                         overlap + Hamiltonian [eV] per IBZ k
  rho_kMM                              per-k density matrix  rho = C^+ diag(occ) C
                                       (unweighted; real-space rho(R) = sum_k w_k rho_k e^{ikR})
  ao_atom_M, ao_l_M, ao_m_M            AO -> (atom, l, m) table; m=-l..l real-harmonic order
  E_total, E_fermi                     [eV]

Consistency contract (checked by test_gpaw_hs_export):
  C S C^+ = I,  H C^+ = S C^+ diag(eps),  Tr(rho S) * w_k = N_e.
"""


def _ao_table(calc):
    """AO index -> (atom, l, m).  l/m ordering follows GPAW real spherical
    harmonics (m = -l..+l)."""
    import numpy as np
    bf = calc.wfs.basis_functions
    M_a = np.asarray(bf.M_a, dtype=int)                      # first AO index per atom
    na = len(M_a); nao = int(calc.wfs.kpt_u[0].S_MM.shape[0])
    ao_atom = np.empty(nao, int); ao_l = np.empty(nao, int); ao_m = np.empty(nao, int)
    for a in range(na):
        M1 = M_a[a]; M2 = M_a[a + 1] if a + 1 < na else nao
        ls = [s.get_angular_momentum_number() for s in bf.sphere_a[a].spline_j]
        labs = [(l, m) for l in ls for m in range(-l, l + 1)]
        assert len(labs) == M2 - M1, f'AO count mismatch on atom {a}: {len(labs)} vs {M2 - M1}'
        for M, (l, m) in zip(range(M1, M2), labs):
            ao_atom[M], ao_l[M], ao_m[M] = a, l, m
    return ao_atom, ao_l, ao_m


def export_hs(calc, atoms, fname='hs.npz'):
    """Write hs.npz on master rank.  Requires an already-converged SCF.
    Asserts serial k-point/band distribution (GPAW default on single node)."""
    import numpy as np
    from ase.units import Ha
    from gpaw.mpi import world
    from gpaw.lcao.tools import get_lcao_hamiltonian
    H_skMM, S_kMM = get_lcao_hamiltonian(calc)               # eV; None on slaves
    if world.rank != 0:
        return
    wfs = calc.wfs
    kpts = calc.get_ibz_k_points()
    wk = calc.get_k_point_weights()
    assert len(wfs.kpt_u) == len(kpts) * wfs.nspins, 'k-points distributed over MPI — rerun with parallel={"kpt":1,"band":1}'
    # GPAW conventions (verified numerically, gpaw>=25):
    #   kpt.eps_n in HARTREE; kpt.S_MM = conj(raw S_kMM from get_lcao_hamiltonian)
    #   orthonormality: C* S C^T = I   (S = raw S_kMM; right factor NOT conjugated)
    #   eigenequation:  H C^T = S C^T diag(eps)   (H,S in eV once eps is xHa)
    #   rho_MN = sum_n occ_n C_nM C*_nN   ->  Tr(rho S) = N_e per k
    nk, nspins = len(kpts), wfs.nspins
    nb = len(wfs.kpt_u[0].eps_n); nao = wfs.kpt_u[0].S_MM.shape[0]
    eps_kn  = np.zeros((nk, nb)); occ_kn = np.zeros((nk, nb))
    C_knM   = np.zeros((nk, nb, nao), complex); rho_kMM = np.zeros((nk, nao, nao), complex)
    assert nspins == 1, 'use export_hs_spin for nspins=2'
    for kpt in wfs.kpt_u:                                  # index by kpt.k — kpt_u order is NOT guaranteed = IBZ order under MPI
        k = kpt.k
        occ = np.asarray(kpt.f_n) / kpt.weightk            # f_n = myocc * spin_degen * w_k -> occ 0..2
        C = np.asarray(kpt.C_nM)
        eps_kn[k], occ_kn[k], C_knM[k] = np.asarray(kpt.eps_n) * Ha, occ, C
        rho_kMM[k] = np.einsum('nM,n,nN->MN', C, occ, C.conj())
        assert np.allclose(C.conj() @ kpt.S_MM.conj() @ C.T, np.eye(nb), atol=1e-8), 'C* S C^T != I'
    S_kMM = np.asarray(S_kMM); H_kMM = np.asarray(H_skMM[0])
    # convention differs across GPAW versions: 24.1 returns conj(H) vs 25.x returns H directly
    # -> test the eigenequation on both and pick whichever satisfies it
    def _resid(H_k, S_k, k=0):
        return np.abs(H_k[k] @ C_knM[k].T - (S_k[k] @ C_knM[k].T) * eps_kn[k][None, :]).max()
    if _resid(H_kMM, S_kMM) > _resid(H_kMM.conj(), S_kMM.conj()):
        H_kMM, S_kMM = H_kMM.conj(), S_kMM.conj()
    for k in range(len(kpts)):
        H, S, C, eps = H_kMM[k], S_kMM[k], C_knM[k], eps_kn[k]
        assert np.allclose(H @ C.T, (S @ C.T) * eps[None, :], atol=1e-5), f'H C^T != S C^T eps at k={k}'
    ne = float(np.round(sum(wk[k] * np.trace(rho_kMM[k] @ S_kMM[k]).real for k in range(len(kpts))), 6))
    ao_atom, ao_l, ao_m = _ao_table(calc)
    np.savez_compressed(fname, pos_ac=atoms.get_positions(), cell_cv=np.asarray(atoms.cell),
                        Z_a=atoms.get_atomic_numbers(), kpts_kc=kpts, w_k=wk,
                        eps_kn=eps_kn, occ_kn=occ_kn, C_knM=C_knM,
                        S_kMM=S_kMM, H_kMM=H_kMM, rho_kMM=rho_kMM,
                        ao_atom_M=ao_atom, ao_l_M=ao_l, ao_m_M=ao_m,
                        E_total=float(atoms.get_potential_energy()),
                        E_fermi=float(calc.get_fermi_level()), n_electrons=ne)
    print(f'export_hs: wrote {fname}  (nao={S_kMM.shape[1]}, nk={len(kpts)}, nb={C_knM.shape[1]}, Ne={ne})')


def export_hs_spin(calc, atoms, fname='hs.npz'):
    """Spin-polarized (colinear nspins=2) variant of export_hs -> hs.npz.

    Same geometry/k-point/AO keys as export_hs, but band/H/rho arrays carry a
    leading spin index:  eps_skn, occ_skn, C_sknM, H_skMM, rho_skMM
    (s = 0 up, 1 down; occupations 0..1 per channel).  E_fermi_s = per-channel
    Fermi levels (identical unless fixmagmom); magmom = total spin moment
    [Bohr magneton], magmom_a = per-atom moments.
    Contract per (s,k):  C* S C^+ = I,  H_s C^T = S C^T diag(eps_s),
    sum_k w_k Tr(rho_sk S) = N_e per channel.
    """
    import numpy as np
    from ase.units import Ha
    from gpaw.mpi import world
    from gpaw.lcao.tools import get_lcao_hamiltonian
    H_skMM, S_kMM = get_lcao_hamiltonian(calc)               # eV; None on slaves
    if world.rank != 0:
        return
    wfs = calc.wfs
    kpts = calc.get_ibz_k_points()
    wk = calc.get_k_point_weights()
    nspins, nk = wfs.nspins, len(kpts)
    assert nspins == 2, f'export_hs_spin requires nspins=2, got {nspins}'
    assert len(wfs.kpt_u) == nk * nspins, 'k-points distributed over MPI — rerun with parallel={"kpt":1,"band":1}'
    nb = len(wfs.kpt_u[0].eps_n)
    nao = S_kMM.shape[1]
    eps_skn = np.zeros((nspins, nk, nb)); occ_skn = np.zeros((nspins, nk, nb))
    C_sknM = np.zeros((nspins, nk, nb, nao), complex); rho_skMM = np.zeros((nspins, nk, nao, nao), complex)
    for kpt in wfs.kpt_u:                                  # kpt.s/kpt.k = spin/k index
        s, k = kpt.s, kpt.k
        occ = np.asarray(kpt.f_n) / kpt.weightk            # spinpol: f_n = occ * w_k -> occ 0..1
        C = np.asarray(kpt.C_nM)
        eps_skn[s, k], occ_skn[s, k], C_sknM[s, k] = np.asarray(kpt.eps_n) * Ha, occ, C
        rho_skMM[s, k] = np.einsum('nM,n,nN->MN', C, occ, C.conj())
        n = len(kpt.eps_n)
        assert np.allclose(C.conj() @ kpt.S_MM.conj() @ C.T, np.eye(n), atol=1e-8), 'C* S C^T != I'
    H_skMM = np.asarray(H_skMM); S_kMM = np.asarray(S_kMM)
    def _resid_s(H_s, S_k):                      # same convention branch as export_hs
        return np.abs(H_s[0, 0] @ C_sknM[0, 0].T - (S_k[0] @ C_sknM[0, 0].T) * eps_skn[0, 0][None, :]).max()
    if _resid_s(H_skMM, S_kMM) > _resid_s(H_skMM.conj(), S_kMM.conj()):
        H_skMM, S_kMM = H_skMM.conj(), S_kMM.conj()
    for s in range(nspins):
        for k in range(nk):
            H, S, C, eps = H_skMM[s, k], S_kMM[k], C_sknM[s, k], eps_skn[s, k]
            assert np.allclose(H @ C.T, (S @ C.T) * eps[None, :], atol=1e-5), f'H C^T != S C^T eps at s={s} k={k}'
    ne_s = np.array([sum(wk[k] * np.trace(rho_skMM[s, k] @ S_kMM[k]).real for k in range(nk)) for s in range(nspins)])
    ao_atom, ao_l, ao_m = _ao_table(calc)
    np.savez_compressed(fname, pos_ac=atoms.get_positions(), cell_cv=np.asarray(atoms.cell),
                        Z_a=atoms.get_atomic_numbers(), kpts_kc=kpts, w_k=wk, nspins=nspins,
                        eps_skn=eps_skn, occ_skn=occ_skn, C_sknM=C_sknM,
                        S_kMM=S_kMM, H_skMM=H_skMM, rho_skMM=rho_skMM,
                        ao_atom_M=ao_atom, ao_l_M=ao_l, ao_m_M=ao_m,
                        E_total=float(atoms.get_potential_energy()),
                        E_fermi_s=np.asarray(calc.get_fermi_levels()),
                        magmom=float(calc.get_magnetic_moment()),
                        magmom_a=np.asarray(calc.get_magnetic_moments()),
                        n_electrons_s=ne_s)
    print(f'export_hs_spin: wrote {fname}  (nao={nao}, nk={nk}, nb={nb}, Ne_up={ne_s[0]:.3f} Ne_dn={ne_s[1]:.3f}, magmom={calc.get_magnetic_moment():.3f})')
