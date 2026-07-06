
#!/usr/bin/env python3
"""
KekulePure.py — Pure-Python NumPy Kekule bond-order optimizer.

Optimizes pi-bond orders for sp2 atoms using a flat-array, vectorized
relaxation.  The sigma skeleton is taken from AtomicSystem.bonds; only
the extra pi contribution is optimized.

Algorithm:
    1. Atom valence: parabolic penalty around atom-specific n_pi
       (sum of pi orders at each atom).  sp2 atoms have n_pi=1, sp3 have 0.
    2. Aromatic bond energy: parabola centered at pi=0.5.
    3. Localization snap: three piecewise parabolas around 0, 0.5, and 1.
    4. Gradient-descent relaxation.
"""

import os
import numpy as np


def _to_int_array(a):
    a = np.asarray(a)
    if a.dtype != np.int32:
        a = a.astype(np.int32)
    return a


class KekulePure:
    """
    Kekule pi-bond order optimizer.

    Parameters
    ----------
    system : AtomicSystem
        Must have `bonds` set.  Each bond is assumed to carry one sigma bond.
    n_pi : array-like of int, optional
        Target number of pi electrons per atom (e.g. 1 for sp2, 0 for sp3).
        If None, it is inferred from `pi_atoms`.
    pi_atoms : array-like of bool, optional
        Mask of atoms that participate in the pi system.  Only used when
        `n_pi` is not given.  If both are None, all atoms are treated as sp2
        (n_pi = 1).
    bonds : array-like of (int, int), optional
        Subset of bonds to optimize.  If None, all `system.bonds` are used.
    Kval : float
        Stiffness of the atom valence (pi-electron count) penalty.
    Kloc : float
        Stiffness of the three-piece localization snap potential (minima at
        pi=0, 0.5, 1).  Usually turned on after an aromatic pre-relaxation.
    Karo : float
        Aromatic stabilization energy.  A positive value makes pi=0.5 (total
        bond order 1.5) the minimum of the bond energy.
    Kbound : float
        Stiffness of the hard [0,1] bounds.
    """

    def __init__(self, system, n_pi=None, pi_atoms=None, bonds=None,
                 Kval=1.0, Kloc=0.0, Karo=0.3, Kbound=1.0, bRandomStart=False,
                 allow_aromatic=False):
        self.system = system
        self.natom = system.natoms
        self.allow_aromatic = allow_aromatic

        # bonds to optimize: (nbond, 2) int array
        if bonds is None:
            bonds = system.bonds
        self.bonds = _to_int_array(bonds)
        self.nbond = len(self.bonds)

        # target number of pi electrons per atom
        if n_pi is None:
            if pi_atoms is None:
                pi_atoms = np.ones(self.natom, dtype=bool)
            pi_atoms = np.asarray(pi_atoms, dtype=bool)
            n_pi = np.where(pi_atoms, 1.0, 0.0)
        self.n_pi = np.asarray(n_pi, dtype=float)

        self.Kval = Kval
        self.Kloc = Kloc
        self.Karo = Karo
        self.Kbound = Kbound

        # current pi bond orders
        if bRandomStart:
            self.bo = np.random.rand(self.nbond)
        else:
            self.bo = np.full(self.nbond, 0.5)
            # Random start is strongly recommended for discrete Kekule patterns
            # because the uniform 0.5 start rounds to 1 and all bonds get
            # pushed toward double bonds.
            self.bo = np.random.rand(self.nbond)

        # split atom indices for projection
        self.i0 = self.bonds[:, 0]
        self.i1 = self.bonds[:, 1]

        # incidence (atom x bond) for quadratic solves
        self._A = np.zeros((self.natom, self.nbond), dtype=float)
        ib = np.arange(self.nbond, dtype=np.int32)
        np.add.at(self._A, (self.i0, ib), 1.0)
        np.add.at(self._A, (self.i1, ib), 1.0)
        self._AT = self._A.T
        self._ATA = self._AT @ self._A

    def project_valence(self):
        """Return atom valences = sum of connected pi bond orders."""
        val = np.zeros(self.natom, dtype=float)
        np.add.at(val, self.i0, self.bo)
        np.add.at(val, self.i1, self.bo)
        return val

    def _localization_target(self, bo):
        """Return target bond order for the three-piece snap parabolas.

        Switching thresholds are 0.25 and 0.75.  If allow_aromatic is False,
        the middle basin is suppressed and the target is the nearest integer.
        """
        target = np.empty_like(bo)
        mask_0 = bo < 0.25
        mask_1 = bo > 0.75
        mask_a = ~(mask_0 | mask_1)
        target[mask_0] = 0.0
        target[mask_1] = 1.0
        if self.allow_aromatic:
            target[mask_a] = 0.5
        else:
            target[mask_a] = np.where(bo[mask_a] > 0.5, 1.0, 0.0)
        return target

    def set_random_bonds(self):
        self.bo = np.random.rand(self.nbond)

    def eval(self):
        """Evaluate forces and energy.  Returns total energy."""
        bo = self.bo

        # atom valence: E = Kval * (n_pi - sum_j pi_ij)^2
        val = self.project_valence()
        d_val = val - self.n_pi
        f_atom = -2.0 * self.Kval * d_val
        E_val = self.Kval * np.sum(d_val * d_val)

        # aromatic parabola centered at 0.5: E = Karo * (pi - 0.5)^2
        d_aro = bo - 0.5
        f_bond = -2.0 * self.Karo * d_aro
        E_aro = self.Karo * np.sum(d_aro * d_aro)

        # localization snap parabolas around 0 / 0.5 / 1
        target = self._localization_target(bo)
        d_loc = bo - target
        f_bond += -2.0 * self.Kloc * d_loc
        E_loc = self.Kloc * np.sum(d_loc * d_loc)

        # hard bounds [0,1]
        d_min = np.minimum(bo - 0.0, 0.0)
        d_max = np.maximum(bo - 1.0, 0.0)
        f_bond -= self.Kbound * (d_min + d_max)
        E_bound = 0.5 * self.Kbound * np.sum(d_min * d_min + d_max * d_max)

        # project atom forces back onto the bonds
        f_bond += f_atom[self.i0] + f_atom[self.i1]

        self._f_bond = f_bond
        self._f_atom = f_atom
        return E_val + E_aro + E_loc + E_bound

    def step(self, dt):
        """Gradient-descent step.  Returns squared force norm."""
        self.eval()
        self.bo += dt * self._f_bond
        self.bo = np.clip(self.bo, 0.0, 1.0)
        return float(np.sum(self._f_bond * self._f_bond))

    def relax(self, dt=0.1, nmax=2000, tol=1e-6, verbose=False):
        """Relax bond orders.  Returns F2 convergence metric."""
        F2 = 1.0
        for it in range(nmax):
            F2 = self.step(dt)
            if verbose and (it % 100 == 0 or it == nmax - 1):
                E = self.eval()
                print(f"iter {it:4d}  E={E:.6f}  F2={F2:.6e}")
            if F2 < tol:
                break
        return F2

    def relax_multistart(self, ntrials=20, dt=0.1, nmax=2000, tol=1e-6,
                         Kloc_final=2.0, aromatic_penalty=0.5):
        """
        Run multiple random starts and return the best discrete Kekule pattern.

        The schedule first enforces atom valence (Kval=10, Kloc=0), then
        localizes bonds to 0/1 (Kloc=Kloc_final, Kval=1).  A small penalty for
        aromatic bonds prevents spurious all-aromatic solutions for non-aromatic
        systems.
        """
        Kval_orig = self.Kval
        Kloc_orig = self.Kloc
        best = None
        best_score = 1e300
        for t in range(ntrials):
            self.set_random_bonds()
            # stage 1: satisfy valence
            self.Kval = 10.0
            self.Kloc = 0.0
            self.relax(dt=dt, nmax=nmax, tol=tol)
            # stage 2: localize
            self.Kval = Kval_orig
            self.Kloc = Kloc_final
            self.relax(dt=dt * 0.2, nmax=nmax * 2, tol=tol)
            E = self.eval()
            n_aromatic = np.sum(self.classify() == 1)
            score = E + aromatic_penalty * n_aromatic
            if score < best_score:
                best_score = score
                best = self.bo.copy()
        self.bo = best
        self.Kval = Kval_orig
        self.Kloc = Kloc_orig
        return best_score

    def pi_bond_orders(self):
        """Optimized pi bond orders, shape (nbond,)."""
        return self.bo.copy()

    def total_bond_orders(self):
        """Total bond orders = sigma (1) + pi."""
        return 1.0 + self.bo

    def snap(self, tol=0.15):
        """Round pi bond orders to the nearest discrete value (0, 0.5, 1).

        If allow_aromatic is False, only 0 or 1 are produced.
        """
        if self.allow_aromatic:
            bo = self.bo.copy()
            bo[bo < 0.25 - tol] = 0.0
            bo[bo > 0.75 + tol] = 1.0
            mask = (bo >= 0.25 - tol) & (bo <= 0.75 + tol)
            bo[mask] = 0.5
            self.bo = bo
        else:
            self.bo = np.round(self.bo)
        return self.bo

    def classify(self, tol=0.05):
        """Classify each pi bond as integer codes: 0 single, 1 aromatic, 2 double."""
        bo = self.bo
        out = np.empty(self.nbond, dtype=np.int8)
        if self.allow_aromatic:
            out[:] = 1
            out[bo < 0.25 - tol] = 0
            out[bo > 0.75 + tol] = 2
        else:
            out[:] = 0
            out[bo > 0.5] = 2
        return out

    def make_bond_style(self, tol=0.05):
        """
        Return line widths and colors for plotting bonds.

        Returns
        -------
        lws : (nbond,) array
        colors : (nbond,) array of color strings
        """
        cls = self.classify(tol=tol)
        lws = np.ones(self.nbond, dtype=float)
        colors = np.empty(self.nbond, dtype=object)
        colors[:] = 'k'
        lws[cls == 1] = 2.0
        colors[cls == 1] = 'green'
        lws[cls == 2] = 3.0
        colors[cls == 2] = 'k'
        return lws, colors

    def _solve_constrained_kkt(self, free, target=None, Karo=None, Kloc=None):
        """Solve the KKT system for free bonds with A x = n_pi enforced exactly.

        Returns updated full bond-order vector and the atom Lagrange multipliers.
        """
        if Karo is None: Karo = self.Karo
        if Kloc is None: Kloc = self.Kloc
        if target is None:
            target = np.full(self.nbond, 0.5)
        target = np.asarray(target, dtype=float)
        # fixed bonds are already stored in self.bo
        fixed = ~free
        # residual right-hand side for the atom constraints after fixed bonds
        rhs_atoms = self.n_pi.copy()
        if np.any(fixed):
            rhs_atoms -= self._A[:, fixed] @ self.bo[fixed]
        nfree = int(np.sum(free))
        if nfree == 0:
            return self.bo.copy(), np.zeros(self.natom)
        A_free = self._A[:, free]
        AT_free = A_free.T
        H = 2.0 * (Karo + Kloc) * np.eye(nfree)
        # [ H  A^T ] [ x ] = [ 2*(Karo*0.5 + Kloc*target) ]
        # [ A  0   ] [ l ]   [ rhs_atoms                ]
        top = np.block([[H, AT_free], [A_free, np.zeros((self.natom, self.natom))]])
        rhs_x = 2.0 * (Karo * 0.5 + Kloc * target[free])
        rhs = np.concatenate([rhs_x, rhs_atoms])
        try:
            sol = np.linalg.solve(top, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(top, rhs, rcond=None)
        x_new = self.bo.copy()
        x_new[free] = sol[:nfree]
        lam = sol[nfree:]
        return x_new, lam

    def solve_constrained(self, target=None, Karo=None, Kloc=None, max_iter=20,
                          tol=1e-9, clip=True):
        """Solve constrained QP: min energy s.t. A*bo = n_pi and 0 <= bo <= 1.

        Uses an active-set method: solve the unconstrained KKT, clip violating
        bonds to [0,1], fix them, and re-solve the reduced KKT until the active
        set stops changing.
        """
        free = np.ones(self.nbond, dtype=bool)
        for _ in range(max_iter):
            x_new, _ = self._solve_constrained_kkt(free, target=target, Karo=Karo, Kloc=Kloc)
            if clip:
                viol_lo = x_new < 0.0
                viol_hi = x_new > 1.0
                viol = viol_lo | viol_hi
                if np.any(viol):
                    x_new[viol_lo] = 0.0
                    x_new[viol_hi] = 1.0
                    free[viol] = False
                    self.bo = x_new
                    continue
            self.bo = x_new
            break
        # enforce constraint check — only check pi-active atoms (n_pi > 0)
        # non-pi atoms (H caps n_pi=-1, sp3 n_pi=0) have no bonds in solver, A@bo=0 for them
        is_pi = self.n_pi > 0
        err = (self._A @ self.bo - self.n_pi)[is_pi]
        max_err = float(np.max(np.abs(err))) if err.size else 0.0
        if max_err > tol:
            print(f"[KekulePure] WARNING: atom-sum constraint not satisfied, max|A@bo-n_pi|={max_err:.3e} (returning best-effort solution)")
        return self.bo

    def solve_snap(self, niter=25, Karo=None, Kloc=None, noise=0.0, seed=0):
        """Alternating constrained solve: update snap targets, then solve QP.

        When noise > 0, random noise is added to bo for target computation
        only (not to bo itself).  Noise is annealed: full strength in early
        iterations to break symmetry, then faded to zero for clean convergence.
        """
        if Karo is None: Karo = self.Karo
        if Kloc is None: Kloc = self.Kloc
        if noise:
            if seed:
                np.random.seed(seed)
            noise_arr = noise * (2.0 * np.random.rand(self.nbond) - 1.0)
        else:
            noise_arr = np.zeros(self.nbond)
        n_noise = max(1, niter // 3)  # apply noise for first 1/3 of iterations
        for it in range(niter):
            if noise and it < n_noise:
                scale = 1.0 - float(it) / n_noise
                target = self._localization_target(self.bo + scale * noise_arr)
            else:
                target = self._localization_target(self.bo)
            self.solve_constrained(target=target, Karo=Karo, Kloc=Kloc)
        return self.bo

    def solve_quadratic(self, Kval=None, Karo=None, Kloc=None, target=None, clip=True):
        """Backward-compatible wrapper: use the constrained KKT solver."""
        return self.solve_constrained(target=target, Karo=Karo, Kloc=Kloc)

    def kkt_matrix(self, target=None, Karo=None, Kloc=None):
        """Build the full KKT matrix for plotting/diagnostics.

        Blocks:
            [ 2(Karo+Kloc)I   A^T ]  <- top-left diagonal (bonds), top-right coupling
            [ A               0  ]  <- bottom-left coupling, bottom-right zeros
        """
        if Karo is None: Karo = self.Karo
        if Kloc is None: Kloc = self.Kloc
        H = 2.0 * (Karo + Kloc) * np.eye(self.nbond)
        return np.block([[H, self._AT], [self._A, np.zeros((self.natom, self.natom))]])

    def plot_kkt_matrix(self, target=None, Karo=None, Kloc=None, ax=None,
                        mode='signed', cmap='seismic', fname=None, show=None):
        """Visualize the KKT matrix with labelled blocks.

        Parameters
        ----------
        mode : 'signed' or 'logabs'
            'signed' shows matrix values with a diverging colormap and a
            symmetric color scale (vmin=-vmax).
            'logabs' shows log10(|M_ij|) with a sequential colormap; the colormap
            is reversed so the largest magnitudes are dark.
        cmap : str or Colormap
            'seismic' for signed mode, 'magma' or 'inferno' for logabs mode.
        """
        import matplotlib.pyplot as plt
        M = self.kkt_matrix(target=target, Karo=Karo, Kloc=Kloc)
        if show is None:
            show = (ax is None)
        if (ax is None):
            fig, ax = plt.subplots(figsize=(7, 7))
        if mode == 'signed':
            vmax = np.max(np.abs(M))
            im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, origin='lower')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='value')
        elif mode == 'logabs':
            Mlog = np.log10(np.abs(M) + 1e-30)
            im = ax.imshow(Mlog, cmap=cmap, origin='lower')
            im.set_cmap(plt.cm.get_cmap(cmap).reversed())
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=r'$\log_{10}(|M_{ij}|)$')
        else:
            raise ValueError("mode must be 'signed' or 'logabs'")
        ax.axvline(self.nbond - 0.5, color='red', lw=1)
        ax.axhline(self.nbond - 0.5, color='red', lw=1)
        ax.text(self.nbond * 0.35, self.nbond * 0.35, r'$2(K_{aro}+K_{loc})I$',
                color='red', ha='center', va='center')
        ax.text(self.nbond + self.natom * 0.5, self.nbond * 0.35, r'$A^T$',
                color='red', ha='center', va='center')
        ax.text(self.nbond * 0.35, self.nbond + self.natom * 0.5, r'$A$',
                color='red', ha='center', va='center')
        ax.text(self.nbond + self.natom * 0.5, self.nbond + self.natom * 0.5,
                '0', color='red', ha='center', va='center')
        ax.set_xlabel('variables (bonds | lambdas)')
        ax.set_ylabel('variables (bonds | lambdas)')
        ax.set_title(f'KKT matrix ({mode})')
        if fname is not None:
            plt.tight_layout()
            plt.savefig(fname)
        if show:
            plt.tight_layout()
            plt.show()
        return ax


def make_pi_mask(system, elements=None):
    """
    Return boolean mask of atoms considered sp2/pi atoms.

    Parameters
    ----------
    elements : set of str, optional
        Element symbols treated as pi atoms.  Default is uppercase C/N/O.
    """
    if elements is None:
        elements = {'C', 'N', 'O'}
    return np.array([e in elements for e in system.enames], dtype=bool)


def make_n_pi(system, sp2=None, sp3=None):
    """
    Return n_pi target per atom from element case.

    Uppercase element symbols (C, N, O, ...) are treated as sp2 with one
    pi electron.  Lowercase symbols (c, n, o, ...) are treated as sp3 with
    zero pi electrons.  If the system has an `_enames_original` attribute
    (e.g. from the ASCII parser) it is used, otherwise the current `enames`
    are used.  If the system has no case information, all atoms are assumed
    to be sp2.
    """
    if sp2 is None:
        sp2 = {'C', 'N', 'O'}
    if sp3 is None:
        sp3 = {'c', 'n', 'o'}
    names = np.asarray(getattr(system, '_enames_original', system.enames), dtype=str)
    namesU = np.char.upper(names)
    namesL = np.char.lower(names)
    is_lower = (names == namesL)
    is_sp2 = (~is_lower) & np.isin(namesU, list(sp2))
    is_sp3 = is_lower & np.isin(namesL, list(sp3))
    n_pi = np.where(is_sp2, 1.0, 0.0)
    n_pi = np.where(is_sp3, 0.0, n_pi)
    # Default rule for ambiguous cases (e.g. elements outside {C,N,O}):
    n_pi = np.where((~is_sp2) & (~is_sp3) & np.isin(namesU, list(sp2)) & (~is_lower), 1.0, n_pi)
    return n_pi


def optimize_pi_bonds(system, n_pi=None, pi_atoms=None, bonds=None, dt=0.1,
                      nmax=2000, tol=1e-6, Kval=1.0, Kloc=0.0, Karo=0.3,
                      Kbound=1.0):
    """
    Convenience function: create and relax a KekulePure instance.

    Returns the optimizer instance.
    """
    k = KekulePure(system, n_pi=n_pi, pi_atoms=pi_atoms, bonds=bonds,
                   Kval=Kval, Kloc=Kloc, Karo=Karo, Kbound=Kbound)
    k.relax(dt=dt, nmax=nmax, tol=tol)
    return k


# ---------------------------------------------------------------------------
# Solver orchestration: two-phase solve, localization, export helpers
# ---------------------------------------------------------------------------

def run_kekule_solver(atoms, Kval=50.0, Kloc=5.0, Karo=0.5, Kbound=1.0, allow_aromatic=True, solver='linsolve', sym_break=0.0, seed=0, n_pi=None, localize=True):
    """Run two-phase Kekulé solver on heavy-atom bonds of an AtomicSystem.

    Phase 1: quadratic solve with ``Kloc=0`` (delocalized / aromatic).
    Phase 2: localization with ``Kloc>0`` and optional symmetry-breaking noise.
    When ``localize=False``, only phase 1 runs and ``bo_snap`` equals ``bo_raw``.

    Args:
        atoms: AtomicSystem with bonds, enames, apos
        Kval: atom-sum stiffness
        Kloc: snap / localization stiffness (used only in phase 2)
        Karo: aromatization stiffness
        Kbound: hard [0,1] bound stiffness
        allow_aromatic: whether aromatic (0.5) bond orders are permitted
        solver: ``'linsolve'`` (quadratic) or ``'gd'`` (gradient descent)
        sym_break: symmetry-breaking noise amplitude (0 = off)
        seed: random seed for sym_break (0 = do not set)
        n_pi: explicit (natoms,) array of pi electron counts. If None, auto-derive from element case.
        localize: if True, run phase 2 (localization). If False, only delocalized phase 1.

    Returns:
        dict with keys ``bo_raw``, ``bo_snap``, ``n_pi``, ``k``, ``err``,
        and ``report`` (diagnostic sub-dict).
    """
    bonds_all = np.asarray(atoms.bonds, dtype=np.int32) if (atoms.bonds is not None) else np.zeros((0, 2), dtype=np.int32)
    if n_pi is None:
        n_pi = make_n_pi(atoms)
    else:
        n_pi = np.asarray(n_pi, dtype=float)
    # Only include bonds where both atoms have pi electrons (n_pi > 0)
    is_pi = n_pi > 0
    pi_mask = is_pi[bonds_all[:, 0]] & is_pi[bonds_all[:, 1]] if len(bonds_all) else np.zeros(0, dtype=bool)
    bonds_pi = bonds_all[pi_mask]
    idx_pi = np.nonzero(pi_mask)[0]
    bo_raw = np.zeros(len(bonds_all), dtype=float)
    bo_snap = np.zeros(len(bonds_all), dtype=float)
    k = KekulePure(atoms, n_pi=n_pi, bonds=bonds_pi, Kval=Kval, Kloc=0.0, Karo=Karo,
                   Kbound=Kbound, allow_aromatic=allow_aromatic)
    print(f"[KEKULE] Solving: natoms={atoms.natoms} nbonds_all={len(bonds_all)} nbonds_pi={len(bonds_pi)}")
    print(f"[KEKULE] n_pi={list(n_pi)}  enames={list(atoms.enames)}")
    if len(bonds_pi):
        print(f"[KEKULE] pi-bonds={bonds_pi.tolist()}")
    err = None
    report = {}
    try:
        if solver == 'linsolve':
            k.solve_quadratic(Kloc=0.0)
            F2 = 0.0
        else:
            F2 = k.relax(dt=0.05, nmax=5000, tol=1e-6)
        bo_raw_h = k.pi_bond_orders()
        bo_raw[idx_pi] = bo_raw_h
        report['phase1_F2'] = F2
        report['phase1_bo'] = bo_raw_h
        report['n_pi'] = n_pi
        if not localize:
            bo_snap[:] = bo_raw
            cls = k.classify()
            report['single'] = int(np.sum(cls == 0))
            report['aromatic'] = int(np.sum(cls == 1))
            report['double'] = int(np.sum(cls == 2))
        else:
            _localize_phase2(k, Kloc, sym_break, seed, solver, n_pi, report)
            bo_snap_h = k.pi_bond_orders().copy()
            bo_snap[idx_pi] = bo_snap_h
    except Exception as e:
        err = e
        import traceback
        report['err'] = str(err)
        report['traceback'] = traceback.format_exc()
    return {'bo_raw': bo_raw, 'bo_snap': bo_snap, 'n_pi': n_pi, 'k': k, 'err': err, 'report': report}


def localize_kekule(k, Kloc=5.0, sym_break=0.0, seed=0, solver='linsolve'):
    """Run phase 2 (localization) on an existing KekulePure solver instance.

    Forces allow_aromatic=False to snap bond orders to integer (0/1).
    When sym_break > 0, persistent noise is added to snap targets (not to bo)
    to break symmetry in degenerate systems.

    Args:
        k: KekulePure instance (already solved phase 1, k.bo holds delocalized result)
        Kloc: localization stiffness
        sym_break: noise amplitude (0 = off)
        seed: random seed (0 = do not set)
        solver: 'linsolve' or 'gd'
    """
    n_pi = k.n_pi
    report = {}
    k.Kloc = Kloc
    k.allow_aromatic = False  # force integer bond orders during localization
    if solver == 'linsolve':
        k.solve_snap(niter=50, noise=sym_break, seed=seed)
        F2 = 0.0
    else:
        if sym_break:
            if seed:
                np.random.seed(seed)
            k.bo = np.clip(k.bo + sym_break * (2.0 * np.random.rand(k.nbond) - 1.0), 0.0, 1.0)
        F2 = k.relax(dt=0.05, nmax=5000, tol=1e-6)
    cls = k.classify()
    report['phase2_F2'] = F2
    report['phase2_bo'] = k.pi_bond_orders()
    report['single'] = int(np.sum(cls == 0))
    report['aromatic'] = int(np.sum(cls == 1))
    report['double'] = int(np.sum(cls == 2))
    err2 = (k._A @ k.bo - n_pi)[n_pi > 0]
    report['max_err'] = float(np.max(np.abs(err2))) if err2.size else 0.0
    return report


def _localize_phase2(k, Kloc, sym_break, seed, solver, n_pi, report):
    """Internal: run phase 2 and fill report dict in-place."""
    k.Kloc = Kloc
    k.allow_aromatic = False
    if solver == 'linsolve':
        k.solve_snap(niter=50, noise=sym_break, seed=seed)
        F2 = 0.0
    else:
        if sym_break:
            if seed:
                np.random.seed(seed)
            k.bo = np.clip(k.bo + sym_break * (2.0 * np.random.rand(k.nbond) - 1.0), 0.0, 1.0)
        F2 = k.relax(dt=0.05, nmax=5000, tol=1e-6)
    cls = k.classify()
    report['phase2_F2'] = F2
    report['phase2_bo'] = k.pi_bond_orders()
    report['single'] = int(np.sum(cls == 0))
    report['aromatic'] = int(np.sum(cls == 1))
    report['double'] = int(np.sum(cls == 2))
    err2 = (k._A @ k.bo - n_pi)[n_pi > 0]
    report['max_err'] = float(np.max(np.abs(err2))) if err2.size else 0.0


def mol_bond_types(atoms, bo_snap=None, allow_aromatic=True, kekule=False):
    """Map Kekulé pi bond orders to MOL V2000 bond-type integers.

    Args:
        atoms: AtomicSystem with bonds and enames
        bo_snap: np.ndarray of pi bond orders (length = len(atoms.bonds))
        allow_aromatic: if True, classify ~0.5 as aromatic (type 4)
        kekule: if False, all bonds are type 1 (single)

    Returns:
        np.ndarray of int or None: bond type per bond (1=single, 2=double,
        4=aromatic), or *None* if the system has no bonds.
    """
    if atoms.bonds is None:
        return None
    bonds = np.asarray(atoms.bonds, dtype=int)
    bond_types = np.ones(len(bonds), dtype=int)
    if not kekule or bo_snap is None:
        return bond_types
    bo_pi = np.asarray(bo_snap, dtype=float)
    for ib, (i, j) in enumerate(bonds):
        if (atoms.enames[i] == 'H') or (atoms.enames[j] == 'H'):
            bond_types[ib] = 1
            continue
        x = float(bo_pi[ib]) if ib < len(bo_pi) else 0.0
        if allow_aromatic and (abs(x - 0.5) < 0.2):
            bond_types[ib] = 4
        elif x > 0.75:
            bond_types[ib] = 2
        else:
            bond_types[ib] = 1
    return bond_types


def export_mol(atoms, mol_opt='auto', out_path='/tmp/kekule/heterocycle.svg',
               title='molecule', bond_types=None):
    """Export an AtomicSystem to a MOL file unless disabled.

    Args:
        atoms: AtomicSystem
        mol_opt: ``'auto'``, ``'off'``, or an explicit file path
        out_path: base output path (used when ``mol_opt='auto'`` to derive
                  the ``.mol`` filename)
        title: MOL title line
        bond_types: np.ndarray of int per bond, or *None* for all-single

    Returns:
        str or None: path to the saved MOL file, or *None* if export is off
    """
    mol_opt = str(mol_opt).strip().lower() if (mol_opt is not None) else 'auto'
    if mol_opt in ('off', '0', 'none', ''):
        return None
    if mol_opt == 'auto':
        root, _ = os.path.splitext(os.path.abspath(out_path))
        mol_fname = root + '.mol'
    else:
        mol_fname = mol_opt
    atoms.save_mol(mol_fname, title=title, bond_types=bond_types)
    return mol_fname
