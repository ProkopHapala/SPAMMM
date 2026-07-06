"""
QEq.py — Charge Equilibration via direct matrix solve.

Implements the Rappe & Goddard QEq method (J. Phys. Chem. 1991, 95, 3358)
using direct linear algebra instead of iterative MD relaxation.

Two solvers:
  - solve_cholesky (default): Cholesky factorization + Schur complement, O(N^3/3)
  - solve_lu (backup):        Full (N+1) KKT system via LU, O(N^3)

Both run entirely in LAPACK native code — no Python-level iteration.

Energy model:
  E(q) = chi^T q + 1/2 q^T H q
  subject to  sum(q) = Q_target

KKT conditions:
  H q + 1*lambda = -chi
  1^T q           = Q_target

H is the damped Coulomb matrix (SPD when hardness > 0):
  H[i,i] = hardness[i]   (chemical hardness = 2*eta in some conventions)
  H[i,j] = e^2 / (1 + r_ij)   for i != j   (damped Coulomb, same as FireCore QEq.h)
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# Coulomb constant in eV*Angstrom/e^2
KE = 14.3996448915


def build_coulomb_matrix(pos, hardness, damping='linear'):
    """Build the QEq H matrix (damped Coulomb + hardness diagonal).

    Parameters:
      pos      : (N,3) atomic positions in Angstrom
      hardness : (N,)  chemical hardness (J_ii) in eV
      damping  : 'linear' => 1/(1+r)  (matches FireCore QEq.h)
                 'gaussian' => erf(eta*r)/r  (requires eta array)

    Returns:
      H : (N,N) symmetric float64
    """
    N = len(hardness)
    r = np.linalg.norm(pos[:, None] - pos[None, :], axis=2)
    if damping == 'linear':
        H = KE / (1.0 + r)
    elif damping == 'gaussian':
        raise NotImplementedError("gaussian damping not yet implemented")
    else:
        raise ValueError(f"unknown damping: {damping}")
    np.fill_diagonal(H, hardness)
    return H


def solve_cholesky(pos, chi, hardness, Q_target=0.0, damping='linear'):
    """Solve QEq via Cholesky factorization + Schur complement.

    H is SPD => factor once (O(N^3/3)), two back-substitutions (O(N^2) each).

    Returns: q (N,) charges in electrons, sum(q) == Q_target to machine precision.
    """
    N = len(chi)
    H = build_coulomb_matrix(pos, hardness, damping=damping)
    cf = cho_factor(H)
    u = cho_solve(cf, np.ones(N))
    v = cho_solve(cf, chi)
    lam = -(Q_target + v.sum()) / u.sum()
    q = -v - lam * u
    return q


def solve_lu(pos, chi, hardness, Q_target=0.0, damping='linear'):
    """Solve QEq via the full (N+1)x(N+1) KKT system using LU.

    Backup method — works even if H is not SPD (e.g. zero hardness).
    """
    N = len(chi)
    H = build_coulomb_matrix(pos, hardness, damping=damping)
    KKT = np.zeros((N + 1, N + 1))
    KKT[:N, :N] = H
    KKT[:N, N] = 1.0
    KKT[N, :N] = 1.0
    rhs = np.zeros(N + 1)
    rhs[:N] = -chi
    rhs[N] = Q_target
    sol = np.linalg.solve(KKT, rhs)
    return sol[:N]


def solve(pos, chi, hardness, Q_target=0.0, method='cholesky', damping='linear'):
    """Dispatch to the requested solver.

    Parameters:
      pos      : (N,3) positions in Angstrom
      chi      : (N,)  electronegativity (Eaff) in eV
      hardness : (N,)  chemical hardness (Ehard) in eV
      Q_target : total charge constraint
      method   : 'cholesky' (default) or 'lu'
      damping  : passed to build_coulomb_matrix

    Returns: q (N,) charges in electrons
    """
    if method == 'cholesky':
        return solve_cholesky(pos, chi, hardness, Q_target, damping)
    elif method == 'lu':
        return solve_lu(pos, chi, hardness, Q_target, damping)
    else:
        raise ValueError(f"unknown method: {method}")


def solve_from_elements(pos, enames, element_types, Q_target=0.0, method='cholesky'):
    """Convenience wrapper: extract chi/hardness from ElementType objects.

    Parameters:
      pos           : (N,3) positions
      enames        : list of N element name strings (e.g. ['C','H','H','O'])
      element_types : dict name -> ElementType (from FFparams.read_element_types)
      Q_target      : total charge
      method        : 'cholesky' or 'lu'

    Returns: q (N,) charges
    """
    chi = np.array([element_types[name].Eaff for name in enames], dtype=np.float64)
    hardness = np.array([element_types[name].Ehard for name in enames], dtype=np.float64)
    return solve(pos, chi, hardness, Q_target=Q_target, method=method)
