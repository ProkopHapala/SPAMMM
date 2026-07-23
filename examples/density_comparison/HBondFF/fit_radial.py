#!/usr/bin/python
"""Fit unified radial potentials to Morse+Coulomb 1D reference.

Two basis options (selectable via --basis):
  lorenz:  fc(r,rc) * ( a*lorenz(r,w) + c )
  fc22:    fc22^2 * ( a*fc22^2 + c ),  where fc22 = (1-(r/rc)^2)^2
  fc22r:   fc22 * ( a*r + c ),         where fc22 = (1-(r/rc)^2)^2
  fc22r2:  fc22^2 * ( a*r*fc22 + c ),    where fc22 = (1-(r/rc)^2)^2
  fc22r3:  fc22 * ( a*r + c*fc22 ),      where fc22 = (1-(r/rc)^2)^2
  fc22r4:  fc22^2 * ( a*r + c*fc22 ),     where fc22 = (1-(r/rc)^2)^2
  fc22r5:  fc22^2 * ( a*r + c*fc22^2 ),    where fc22 = (1-(r/rc)^2)^2
  morse1:  fc(r,rc) * a*(r - r_node),       r_node = R0 - ln(2)/beta (analytical Morse zero crossing)
  morse1b: fc^4 * a*(r - r_node),          r_node = R0 - ln(2)/beta, fc = smoothstep cutoff
  morse1c: fc22^4 * a*(r - r_node),         r_node = R0 - ln(2)/beta, fc22 = (1-(r/rc)^2)^2
  morse1cmp: compare morse1b vs morse1c on same plot
  morse1pw: compare fc^n and fc22^n for n=1,2,3,4 on same plot
  morse2:  fc*( a*fc^3*(r-r_node) + c ),  r_node = R0 - ln(2)/beta, fc = smoothstep
  morse2b: fc^2*( a*fc^2*(r-r_node) + c ), r_node = R0 - ln(2)/beta, fc = smoothstep
  morse2c: fc^2*( (r-r_node)*(a*fc^2 + c) ), r_node = R0 - ln(2)/beta, fc = smoothstep
  morse2d: fc*( (r-r_node)*(a*fc^3 + c) ),  r_node = R0 - ln(2)/beta, fc = smoothstep

Compact exponential family (use --compact-exp-demo):
  rho(r,w) = sqrt(r^2+w^2)-w   [default, fully smooth]
  y        = max(0,1-beta*(rho-R0)/n)^n
  V        = E0*y*(alpha*y-(1+alpha))

  alpha=1,w=0 : compact Morse atom-atom interaction
  alpha<1,w>0 : blunt atom-electron-pair interaction, same instructions
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from spammm.topology.FFparams import read_element_types, read_atom_types

COULOMB_CONST = 14.3996448915
MORSE_BETA = 1.7
RC = 6.0
W = 0.0

data_path = os.path.join(REPO_ROOT, 'data')
etypes = read_element_types(os.path.join(data_path, 'ElementTypes.dat'))
atom_types = read_atom_types(os.path.join(data_path, 'AtomTypes.dat'), etypes)

def get_REQ(ename):
    at = atom_types[ename]
    return at.RvdW, np.sqrt(at.EvdW)

def morse_energy(r, R0, E0):
    e = np.exp(-MORSE_BETA * (r - R0))
    return E0 * (e*e - 2.0*e)

def coulomb_energy(r, q_prod):
    return COULOMB_CONST * q_prod / r

def fcut(r, rc):
    x = np.clip(1.0 - r/rc, 0, 1)
    return 3*x**2 - 2*x**3

def lorenz(r, w):
    return 1.0 / (w*w + r*r)

def fc22(r, rc):
    x = np.clip(1.0 - (r/rc)**2, 0, 1)
    return x*x


# ==================================================================
# Compact exponential Morse + branch-free electron-pair smoothing
# ==================================================================
#
# Compact exponential overlap:
#
#   y_n(rho) = max(0, 1 - beta*(rho-R0)/n)^n
#
# For x=rho-R0,
#
#   ln(y_n) = -beta*x - beta^2*x^2/(2*n) + O(n^-2),
#
# hence y_n -> exp[-beta*(rho-R0)] as n->infinity.  The transformed cutoff
# is rho_c=R0+n/beta.  Powers n=2,4,8 are especially cheap by repeated
# squaring.
#
# A single energy formula covers real atoms and electron-pair dummy sites:
#
#   V = E0*y*(alpha*y-(1+alpha)) = A*y^2-B*y,
#   A=E0*alpha, B=E0*(1+alpha).
#
# alpha=1 gives compact Morse: E0*(y^2-2y).
# alpha=0 gives a purely attractive overlap: -E0*y.
# For every alpha, V(y=1)=-E0 and the tail approaches zero from below.
#
# To blunt only electron-pair interactions without a pair-kind branch, replace
# r by a numerical-parameter-dependent soft radius.  The recommended form is
#
#   rho_sqrt = sqrt(r^2+w^2)-w
#            = r^2/[sqrt(r^2+w^2)+w]       (stable evaluation).
#
# It satisfies rho=r for w=0 and rho=r^2/(2w)+O(r^4) for w>0.  Therefore the
# exact same instructions give a sharp atom core for w=0 and a smooth central
# blob for w>0.  A cheaper C2 alternative is rho_rat=r^2/(r+w).
#
# For atom-atom:
#   R0=Ri+Rj, E0=ei*ej, alpha=1, w=0.
#   Then V(R0)=-E0, V'(R0)=0 and V''(R0)=2*E0*beta^2 exactly for every n.
#
# For atom-epair:
#   R0=0, alpha<1, w>0.
#   With rho_sqrt, the central curvature is
#       k0 = E0*(1-alpha)*beta/w,
#   so w=E0*(1-alpha)*beta/k0 gives a requested harmonic stiffness.
#
# Suggested per-type mixing parameters:
#   R_i : atomic radius
#   e_i : sqrt(self well depth)
#   g_i : core flag, 1 for a real atom and 0 for an epair
#   w_i : blunt width, 0 for a real atom and >0 for an epair
#
#   g_ij     = g_i*g_j
#   R0_ij    = g_ij*(Ri+Rj)
#   E0_ij    = e_i*e_j
#   alpha_ij = g_ij
#   w_ij     = w_i+w_j
#
# Thus real-real is compact Morse and real-epair is a blunt attractive well.
# All quantities can be precomputed in a type-pair table; runtime lanes execute
# the same arithmetic.  max() is a scalar select, not a divergent pair-kind
# branch.


def soft_radius_sqrt(r, w):
    """Fully smooth rho=sqrt(r^2+w^2)-w and d(rho)/dr.

    The quotient form avoids catastrophic cancellation near r=0.
    """
    if w < 0.0:
        raise ValueError(f"w must be non-negative, got {w}")
    r = np.asarray(r, dtype=float)
    rw = np.sqrt(r*r+w*w)
    den = rw+w
    rho = np.divide(r*r, den, out=np.zeros_like(r), where=den > 0.0)
    drho = np.divide(r, rw, out=np.zeros_like(r), where=rw > 0.0)
    return rho, drho


def soft_radius_rational(r, w):
    """Cheaper rho=r^2/(r+w) and d(rho)/dr.

    It has the desired parabolic center and exact w=0 limit, but is only C2 in
    Cartesian space at the origin rather than analytic to all orders.
    """
    if w < 0.0:
        raise ValueError(f"w must be non-negative, got {w}")
    r = np.asarray(r, dtype=float)
    den = r+w
    rho = np.divide(r*r, den, out=np.zeros_like(r), where=den > 0.0)
    drho = np.divide(
        r*(r+2.0*w), den*den,
        out=np.zeros_like(r), where=den > 0.0,
    )
    return rho, drho


def soft_radius(r, w, kind='sqrt'):
    if kind == 'sqrt':
        return soft_radius_sqrt(r, w)
    if kind == 'rational':
        return soft_radius_rational(r, w)
    raise ValueError(f"Unknown soft-radius kind: {kind}")


def compact_exp_overlap(r, R0, beta, power=4, w=0.0,
                        soft_kind='sqrt'):
    """Return compact exponential overlap y and dy/dr."""
    if beta <= 0.0 or power <= 0:
        raise ValueError("beta and power must be positive")
    rho, drho = soft_radius(r, w, kind=soft_kind)
    u = 1.0-beta*(rho-R0)/float(power)
    u = np.maximum(u, 0.0)
    y = u**power
    dy = -beta*u**(power-1.0)*drho
    return y, dy


def compact_exp_unified_energy_force(r, R0, E0, beta, power=4,
                                     alpha=1.0, w=0.0,
                                     soft_kind='sqrt'):
    """Return energy V and scalar radial force F_r=-dV/dr."""
    if E0 < 0.0:
        raise ValueError(f"E0 must be non-negative, got {E0}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    y, dy = compact_exp_overlap(
        r, R0, beta, power=power, w=w, soft_kind=soft_kind)
    dVdy = E0*(2.0*alpha*y-(1.0+alpha))
    V = E0*y*(alpha*y-(1.0+alpha))
    Fr = -dVdy*dy
    return V, Fr


def compact_exp_force_over_r(r, R0, E0, beta, power=4,
                             alpha=1.0, w=0.0, soft_kind='sqrt',
                             eps=1e-12):
    """Return V and f such that vector force is F_vec=f*dr_vec.

    This mirrors a branch-free GPU kernel and avoids normalizing dr separately.
    """
    if beta <= 0.0 or power <= 0:
        raise ValueError("beta and power must be positive")
    r = np.asarray(r, dtype=float)
    r2 = r*r
    if soft_kind == 'sqrt':
        rw = np.sqrt(r2+w*w)
        rho = r2/np.maximum(rw+w, eps)
        grad_over_r = 1.0/np.maximum(rw, eps)
    elif soft_kind == 'rational':
        inv = 1.0/np.maximum(r+w, eps)
        rho = r2*inv
        grad_over_r = (r+2.0*w)*inv*inv
    else:
        raise ValueError(f"Unknown soft-radius kind: {soft_kind}")
    u = np.maximum(1.0-beta*(rho-R0)/float(power), 0.0)
    y = u**power
    un1 = u**(power-1.0)
    V = E0*y*(alpha*y-(1.0+alpha))
    f_over_r = (
        E0*beta*(2.0*alpha*y-(1.0+alpha))*un1*grad_over_r
    )
    return V, f_over_r


def compact_exp_inner_node(R0, beta, power=4, alpha=1.0):
    """Inner zero in transformed radius rho; None for pure attraction."""
    if alpha <= 0.0:
        return None
    y_zero = (1.0+alpha)/alpha
    return R0+(power/beta)*(1.0-y_zero**(1.0/power))


def compact_exp_physical_cutoff(R0, beta, power=4, w=0.0,
                                soft_kind='sqrt'):
    """Physical cutoff r_c and squared cutoff r_c^2."""
    rho_c = R0+power/beta
    if soft_kind == 'sqrt':
        rc2 = rho_c*(rho_c+2.0*w)
        return np.sqrt(rc2), rc2
    if soft_kind == 'rational':
        rc = 0.5*(rho_c+np.sqrt(rho_c*rho_c+4.0*rho_c*w))
        return rc, rc*rc
    raise ValueError(f"Unknown soft-radius kind: {soft_kind}")


def mix_compact_exp_pair(Ri, ei, gi, wi, Rj, ej, gj, wj,
                         beta, power=4, soft_kind='sqrt'):
    """Analytical atom/epair mixing with one branch-free runtime kernel."""
    if not (0.0 <= gi <= 1.0 and 0.0 <= gj <= 1.0):
        raise ValueError("gi and gj must lie in [0,1]")
    gij = gi*gj
    R0 = gij*(Ri+Rj)
    E0 = ei*ej
    alpha = gij
    w = wi+wj
    rc, rc2 = compact_exp_physical_cutoff(
        R0, beta, power=power, w=w, soft_kind=soft_kind)
    return {
        'R0': R0, 'E0': E0, 'alpha': alpha, 'w': w,
        'beta': beta, 'power': power, 'rc': rc, 'rc2': rc2,
        'A': E0*alpha, 'B': E0*(1.0+alpha),
        'rho_node': compact_exp_inner_node(
            R0, beta, power=power, alpha=alpha),
        'soft_kind': soft_kind,
    }


def compact_exp_diagnostics(R0, E0, beta, power=4,
                            alpha=1.0, w=0.0, soft_kind='sqrt'):
    """Return analytical/numerical shape diagnostics."""
    rc, rc2 = compact_exp_physical_cutoff(
        R0, beta, power=power, w=w, soft_kind=soft_kind)
    if alpha == 1.0 and w == 0.0:
        curvature = 2.0*E0*beta*beta
    elif R0 == 0.0 and w > 0.0:
        fac = 1.0 if soft_kind == 'sqrt' else 2.0
        curvature = fac*E0*(1.0-alpha)*beta/w
    else:
        curvature = np.nan
    return {
        'rc': rc, 'rc2': rc2,
        'rho_node': compact_exp_inner_node(
            R0, beta, power=power, alpha=alpha),
        'depth': -E0,
        'curvature': curvature,
        'morse_curvature': 2.0*E0*beta*beta,
        'soft_kind': soft_kind,
    }


def run_compact_exp_demo(powers=(2, 4, 8), epair_width=0.6,
                         epair_alpha=0.0, soft_kind='sqrt'):
    """Compare compact exponentials to Morse and show the epair limit."""
    powers = tuple(int(n) for n in powers)
    if not powers or any(n <= 0 for n in powers):
        raise ValueError("powers must contain positive integers")
    if epair_width <= 0.0:
        raise ValueError("epair_width must be positive")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (label, ei_name, ej_name, _qi, _qj) in enumerate(pairs):
        Ri, ei = get_REQ(ei_name)
        Rj, ej = get_REQ(ej_name)
        R0 = Ri+Rj
        E0 = ei*ej
        max_rc = max(
            compact_exp_physical_cutoff(
                R0, MORSE_BETA, n, 0.0, soft_kind=soft_kind)[0]
            for n in powers
        )
        r = np.linspace(max(0.02, 0.35*R0), max_rc*1.03, 900)
        ax = axes[idx]
        ax.plot(r, morse_energy(r, R0, E0), 'k:', linewidth=2.0,
                label='Morse')
        for n in powers:
            V, _ = compact_exp_unified_energy_force(
                r, R0, E0, MORSE_BETA, power=n, alpha=1.0, w=0.0,
                soft_kind=soft_kind)
            rc_n, _ = compact_exp_physical_cutoff(
                R0, MORSE_BETA, power=n, w=0.0,
                soft_kind=soft_kind)
            ax.plot(r, V, linewidth=1.1,
                    label=f'compact exp n={n}, rc={rc_n:.2f}')
            ax.axvline(rc_n, linestyle='--', linewidth=1.0, alpha=0.6)
        ax.axhline(0.0, linewidth=0.5)
        ax.set_ylim(-1.15*E0, 3.0*E0)
        ax.set_title(f'{label}: R0={R0:.3f}, E0={E0:.4g}')
        ax.set_xlabel('r [Angstrom]')
        ax.set_ylabel('Energy [eV]')
        ax.legend()

    ax = axes[3]
    E0 = 1.0
    R0 = 0.0
    max_rc = max(
        compact_exp_physical_cutoff(
            R0, MORSE_BETA, n, epair_width,
            soft_kind=soft_kind)[0]
        for n in powers
    )
    r = np.linspace(0.0, max_rc*1.03, 900)
    for n in powers:
        V, _ = compact_exp_unified_energy_force(
            r, R0, E0, MORSE_BETA, power=n,
            alpha=epair_alpha, w=epair_width, soft_kind=soft_kind)
        fac = 1.0 if soft_kind == 'sqrt' else 2.0
        k0 = fac*E0*(1.0-epair_alpha)*MORSE_BETA/epair_width
        ax.plot(r, V, linewidth=1.1,
                label=f'n={n}, alpha={epair_alpha:g}, k0={k0:.3g}')
        rc_n, _ = compact_exp_physical_cutoff(
            R0, MORSE_BETA, power=n, w=epair_width, soft_kind=soft_kind)
        ax.axvline(rc_n, linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axhline(0.0, linewidth=0.5)
    ax.set_title(
        f'Blunt atom-epair well: w={epair_width:g}, {soft_kind}')
    ax.set_xlabel('r [Angstrom]')
    ax.set_ylabel('Energy / E0')
    ax.legend()

    fig.suptitle(
        'Compact exponential Morse and branch-free smooth epair limit',
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ==================================================================
#  Analytical coefficient derivation for morse2c / morse2d
# ==================================================================
#
# Morse reference and pair mixing:
#
#   M(r)   = E0 [exp(-2*beta*(r-R0)) - 2*exp(-beta*(r-R0))]
#   R0_ij  = Ri + Rj
#   E0_ij  = sqrt(Eii*Ejj) = ei*ej, where ei=sqrt(Eii)
#   Delta  = ln(2)/beta
#   r_node = R0 - Delta                         (exact Morse zero)
#
# For the compact family
#
#   V(r) = f(r)^p * (r-r_node) * (a*f(r)^q + c)
#        = (r-r_node) [a*f(r)^(p+q) + c*f(r)^p],
#
# use p=1,q=3 for morse2d and p=2,q=2 for morse2c.  Since r_node is
# already fixed analytically, the two coefficients a,c can impose
#
#   V(R0)  = -E0       exact well depth
#   V'(R0) =  0        exact equilibrium position.
#
# Define f0=f(R0), h0=f'(R0)/f0, n=p+q, B=-E0/Delta, and
#
#   s = -1/(Delta*h0).
#
# Solving the two linear equations gives
#
#   a = B*(s-p)/(q*f0^n)
#   c = B*(n-s)/(q*f0^p).
#
# This is an analytical mixing rule, not a radial fit.  All pair dependence
# enters through R0=Ri+Rj and E0=ei*ej.  For a fixed beta and rc, a and c are
# deterministic functions of the additive pair radius and scale linearly with
# the geometrically mixed energy E0.
#
# Tail caveat:
#   near rc, V ~ c*(r-r_node)*f^p.  Thus c>0 means the potential approaches
#   zero from above and develops an outer repulsive bump.  c=0 is the clean
#   boundary where the tail approaches zero from below without a bump.
#
# For c=0, exact stationarity requires h0=-1/(n*Delta).  This determines an
# analytical pair cutoff.  In particular, for
#
#   fc22=(1-(r/rc)^2)^2,
#
#   rc^2 = R0^2 + 4*n*Delta*R0.
#
# For our n=p+q=4 basis, rc^2=R0^2+16*Delta*R0.  More generally, any
# positive envelope power n is possible:
#
#   V_n(r) = a_n*(r-r_node)*f(r)^n
#   rc^2   = R0^2 + 4*n*Delta*R0                 [fc22 cutoff]
#   a_n    = -E0/(Delta*f(R0)^n).
#
# Thus n=2 is just as analytical as n=4.  It has a slower algebraic decay near
# the cutoff (fc22^2 ~ (rc-r)^4 rather than fc22^4 ~ (rc-r)^8), although its
# exact-stationarity cutoff is shorter.
#
# Even better, one can interpolate continuously between powers n_lo and n_hi
# while retaining one common cutoff and exact Morse geometry:
#
#   V(r) = -(E0/Delta)*(r-r_node) *
#          [(1-lambda)*(f/f0)^n_lo + lambda*(f/f0)^n_hi]
#
#   n_eff = (1-lambda)*n_lo + lambda*n_hi
#
# and choose rc from the same pure-tail condition using n_eff.  At R0 the
# normalized bracket is one, while its logarithmic slope is n_eff*f'/f; hence
# V(R0)=-E0 and V'(R0)=0 exactly for every lambda.  For n_lo=2,n_hi=4:
#
#   rc^2 = R0^2 + 8*(1+lambda)*Delta*R0         [fc22 cutoff]
#   c2    = -(E0/Delta)*(1-lambda)/f0^2
#   a4    = -(E0/Delta)*lambda/f0^4
#   V     = (r-r_node)*(c2*f^2 + a4*f^4).
#
# Both coefficients are non-positive for 0<=lambda<=1, so there is no outer
# repulsive bump and no additional zero.  This is a one-cutoff, two-power
# homotopy, cheaper than literally evaluating and adding two endpoint potentials
# with two different cutoffs.
#
# The optional ``node-slope`` mode instead matches M'(r_node)=-4*beta*E0 and
# M(R0)=-E0.  It is retained for comparison, but it generally shifts the
# minimum away from R0.


def cutoff_value_and_derivative(r, rc, kind='smoothstep'):
    """Return scalar cutoff f(r) and df/dr."""
    if rc <= 0.0:
        raise ValueError(f"rc must be positive, got {rc}")
    if r <= 0.0:
        return 1.0, 0.0
    if r >= rc:
        return 0.0, 0.0
    if kind == 'smoothstep':
        t = r/rc
        return 1.0 - 3.0*t*t + 2.0*t*t*t, 6.0*t*(t-1.0)/rc
    if kind == 'fc22':
        t = r/rc
        u = 1.0 - t*t
        return u*u, -4.0*r*u/(rc*rc)
    raise ValueError(f"Unknown cutoff kind: {kind}")


def cutoff_array(r, rc, kind='smoothstep'):
    if kind == 'smoothstep':
        return fcut(r, rc)
    if kind == 'fc22':
        return fc22(r, rc)
    raise ValueError(f"Unknown cutoff kind: {kind}")


def morse_basis_pq(basis):
    if basis == 'morse2d':
        return 1, 3
    if basis == 'morse2c':
        return 2, 2
    return None


def critical_cutoff_for_pure_tail(R0, beta, cutoff_kind='fc22',
                                  envelope_power=4):
    """Return rc for which exact depth+minimum gives c=0."""
    if R0 <= 0.0 or beta <= 0.0 or envelope_power <= 0:
        raise ValueError("R0, beta, and envelope_power must be positive")
    delta = np.log(2.0)/beta
    n = float(envelope_power)
    if cutoff_kind == 'fc22':
        return np.sqrt(R0*R0 + 4.0*n*delta*R0)
    if cutoff_kind == 'smoothstep':
        # Let t=R0/rc.  Removing the degenerate root t=1 leaves
        # (6*n*Delta+2*R0)t^2 - R0*t - R0 = 0.
        A = 6.0*n*delta + 2.0*R0
        t = (R0 + np.sqrt(R0*R0 + 4.0*A*R0))/(2.0*A)
        if not (0.0 < t < 1.0):
            raise RuntimeError(f"Invalid smoothstep root t={t}")
        return R0/t
    raise ValueError(f"Unknown cutoff kind: {cutoff_kind}")


def mix_morse_pair_pure_tail(Ri, ei, Rj, ej, beta,
                             cutoff_kind='fc22', envelope_power=4):
    """Analytical one-term pair rule with exact node, minimum and depth."""
    R0 = Ri + Rj
    E0 = ei * ej
    delta = np.log(2.0)/beta
    r_node = R0 - delta
    rc = critical_cutoff_for_pure_tail(
        R0, beta, cutoff_kind=cutoff_kind,
        envelope_power=envelope_power,
    )
    f0, _ = cutoff_value_and_derivative(R0, rc, cutoff_kind)
    a = -E0/(delta*f0**envelope_power)
    return {
        'R0': R0, 'E0': E0, 'r_node': r_node,
        'rc': rc, 'rc2': rc*rc, 'a': a, 'c': 0.0,
        'cutoff': cutoff_kind, 'envelope_power': envelope_power,
    }


def mix_morse_pair_tail_blend(Ri, ei, Rj, ej, beta, blend=0.5,
                              cutoff_kind='fc22', power_lo=2, power_hi=4):
    """Blend two pure attractive envelope powers using one analytical cutoff.

    The returned potential is

        V(r) = (r-r_node) * (c_lo*f(r)^power_lo
                             + c_hi*f(r)^power_hi).

    ``blend=0`` is the pure lower-power solution and ``blend=1`` is the pure
    higher-power solution.  Every intermediate value exactly preserves the
    Morse zero crossing, equilibrium distance, and well depth.  Since both
    coefficients are <=0, the outer tail approaches zero from below.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if not (0.0 <= blend <= 1.0):
        raise ValueError(f"blend must be in [0,1], got {blend}")
    if power_lo <= 0 or power_hi <= power_lo:
        raise ValueError(
            f"Need 0 < power_lo < power_hi; got {power_lo}, {power_hi}"
        )

    R0 = Ri + Rj
    E0 = ei * ej
    delta = np.log(2.0)/beta
    r_node = R0 - delta

    # The weighted exponent is exactly the logarithmic derivative of the
    # normalized blend at R0.  It may be non-integer; it is used only to set
    # rc, while runtime evaluation still uses integer powers power_lo/power_hi.
    n_eff = (1.0-blend)*power_lo + blend*power_hi
    rc = critical_cutoff_for_pure_tail(
        R0, beta, cutoff_kind=cutoff_kind, envelope_power=n_eff,
    )
    f0, _ = cutoff_value_and_derivative(R0, rc, cutoff_kind)
    scale = -E0/delta
    c_lo = scale*(1.0-blend)/(f0**power_lo)
    c_hi = scale*blend/(f0**power_hi)

    return {
        'R0': R0, 'E0': E0, 'r_node': r_node,
        'rc': rc, 'rc2': rc*rc,
        'c_lo': c_lo, 'c_hi': c_hi,
        # Aliases matching V=(r-r_node)*(c*f^2+a*f^4), the common fast case.
        'c': c_lo if power_lo == 2 else None,
        'a': c_hi if power_hi == 4 else None,
        'blend': blend, 'power_lo': power_lo, 'power_hi': power_hi,
        'effective_power': n_eff, 'cutoff': cutoff_kind,
    }


def compact_tail_blend_value_and_derivative(
        r, rc, c_lo, c_hi, r_node, cutoff_kind='fc22',
        power_lo=2, power_hi=4):
    """Scalar V,dV/dr for the common-cutoff two-power pure-tail blend."""
    f, df = cutoff_value_and_derivative(r, rc, cutoff_kind)
    x = r-r_node
    A = c_lo*f**power_lo + c_hi*f**power_hi
    dA = df*(
        c_lo*power_lo*f**(power_lo-1)
        + c_hi*power_hi*f**(power_hi-1)
    )
    return x*A, A+x*dA


def tail_blend_residuals(params, beta):
    """Constraint and curvature diagnostics for a mixed-tail parameter set."""
    R0 = params['R0']
    E0 = params['E0']
    rn = params['r_node']
    args = (
        params['rc'], params['c_lo'], params['c_hi'], rn,
        params['cutoff'], params['power_lo'], params['power_hi'],
    )
    Vn, dVn = compact_tail_blend_value_and_derivative(rn, *args)
    V0, dV0 = compact_tail_blend_value_and_derivative(R0, *args)
    h = max(1e-5, 1e-4*R0)
    Vm, _ = compact_tail_blend_value_and_derivative(R0-h, *args)
    Vp, _ = compact_tail_blend_value_and_derivative(R0+h, *args)
    curvature = (Vp-2.0*V0+Vm)/(h*h)
    return {
        'node_value_error': Vn,
        'depth_error': V0+E0,
        'minimum_slope_error': dV0,
        'node_slope_error': dVn+4.0*beta*E0,
        'curvature': curvature,
        'morse_curvature': 2.0*beta*beta*E0,
        'tail_side': 'below/attractive',
    }



def mix_morse_pair(Ri, ei, Rj, ej, beta, rc, basis='morse2d',
                   cutoff_kind='smoothstep', mode='minimum'):
    """Apply Ri+Rj and ei*ej mixing and return pair-table parameters."""
    R0 = Ri + Rj
    E0 = ei * ej
    a, c, r_node = analytical_coeffs(
        R0, E0, beta, rc, basis=basis,
        cutoff_kind=cutoff_kind, mode=mode,
    )
    return {
        'R0': R0, 'E0': E0, 'r_node': r_node,
        'rc': rc, 'rc2': rc*rc, 'a': a, 'c': c,
        'basis': basis, 'cutoff': cutoff_kind, 'mode': mode,
    }


def analytical_coeffs(R0, E0, beta, rc, basis='morse2d',
                      cutoff_kind='smoothstep', mode='minimum'):
    """Return analytical ``a,c,r_node`` for the compact Morse basis."""
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if E0 < 0.0:
        raise ValueError(f"E0 must be non-negative, got {E0}")
    if not (0.0 < R0 < rc):
        raise ValueError(f"Need 0 < R0 < rc; got R0={R0}, rc={rc}")

    delta = np.log(2.0)/beta
    r_node = R0 - delta
    pq = morse_basis_pq(basis)
    if pq is None:
        return None, None, r_node
    p, q = pq
    n = p + q
    f0, df0 = cutoff_value_and_derivative(R0, rc, cutoff_kind)
    B = -E0/delta

    if mode == 'minimum':
        h0 = df0/f0
        if abs(h0) < 1e-14:
            raise ValueError("Cutoff derivative at R0 is too small")
        s = -1.0/(delta*h0)
        a = B*(s-p)/(q*f0**n)
        c = B*(n-s)/(q*f0**p)
        return a, c, r_node

    if mode == 'node-slope':
        fn, _ = cutoff_value_and_derivative(r_node, rc, cutoff_kind)
        Fn, Gn = fn**n, fn**p
        F0, G0 = f0**n, f0**p
        slope = -4.0*beta*E0
        det = Fn*G0 - F0*Gn
        if abs(det) < 1e-14*max(abs(Fn*G0), abs(F0*Gn), 1.0):
            raise ValueError("node-slope coefficient system is nearly singular")
        a = (slope*G0 - B*Gn)/det
        c = (Fn*B - F0*slope)/det
        return a, c, r_node

    raise ValueError(f"Unknown analytical mode: {mode}")


def compact_morse_value_and_derivative(r, rc, a, c, r_node,
                                       basis='morse2d',
                                       cutoff_kind='smoothstep'):
    """Scalar V(r), dV/dr for morse2c/morse2d."""
    p, q = morse_basis_pq(basis)
    n = p + q
    f, df = cutoff_value_and_derivative(r, rc, cutoff_kind)
    x = r-r_node
    A = a*f**n + c*f**p
    dA = df*(a*n*f**(n-1) + c*p*f**(p-1))
    return x*A, A + x*dA


def analytical_residuals(R0, E0, beta, rc, a, c, r_node,
                         basis='morse2d', cutoff_kind='smoothstep'):
    Vn, dVn = compact_morse_value_and_derivative(
        r_node, rc, a, c, r_node, basis, cutoff_kind)
    V0, dV0 = compact_morse_value_and_derivative(
        R0, rc, a, c, r_node, basis, cutoff_kind)
    h = max(1e-5, 1e-4*R0)
    Vm, _ = compact_morse_value_and_derivative(
        R0-h, rc, a, c, r_node, basis, cutoff_kind)
    Vp, _ = compact_morse_value_and_derivative(
        R0+h, rc, a, c, r_node, basis, cutoff_kind)
    curvature = (Vp - 2.0*V0 + Vm)/(h*h)
    return {
        'node_value_error': Vn,
        'depth_error': V0 + E0,
        'minimum_slope_error': dV0,
        'node_slope_error': dVn + 4.0*beta*E0,
        'curvature': curvature,
        'morse_curvature': 2.0*beta*beta*E0,
        'tail_side': 'above/repulsive' if c > 0.0 else 'below/attractive',
    }

# (name, ename_i, ename_j, q_i, q_j)
pairs = [
    ('O-O', 'O', 'O', 0.0, 0.0),
    ('O-H', 'O', 'H', 0.0, 0.0),
    ('H-H', 'H', 'H', 0.0, 0.0),
]

def run_fit(basis='lorenz', rc=RC, w=W, analytical=False,
            morse_cutoff='smoothstep', analytical_mode='minimum',
            tail_blend=0.5, pure_tail_power=4):
    r = np.linspace(0.5, 10.0, 500)
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    axes = axes.flatten()

    for idx, (label, ei, ej, qi, qj) in enumerate(pairs):
        Ri, Ei = get_REQ(ei)
        Rj, Ej = get_REQ(ej)
        R0 = Ri + Rj
        E0 = Ei * Ej
        E_morse = morse_energy(r, R0, E0)
        E_coul = coulomb_energy(r, qi * qj) if (qi != 0 or qj != 0) else np.zeros_like(r)
        E_ref = E_morse + E_coul
        Emin = E_ref.min()
        kT = abs(Emin) if abs(Emin) > 1e-6 else 0.5

        if basis == 'lorenz':
            # E = fc * (a*lorenz + c)
            # phi1 = fc*lorenz, phi2 = fc
            fc = fcut(r, rc)
            phi1 = fc * lorenz(r, w)
            phi2 = fc
            def E_fit_func(a, c): return fc * (a * lorenz(r, w) + c)
            f1_label = 'c*fc'
            f2_label = 'a*fc*lorenz'
            def f1_func(a, c): return c * fc
            def f2_func(a, c): return a * fc * lorenz(r, w)
            title_formula = f'fc(r,{rc})*(a*lorenz(r,{w})+c)'
        elif basis == 'fc22':
            # E = fc22^2 * (a*fc22^2 + c)
            # phi1 = fc22^4, phi2 = fc22^2
            f = fc22(r, rc)  # fc22 = (1-(r/rc)^2)^2
            f2v = f * f      # fc22^2
            f4v = f2v * f2v   # fc22^4
            phi1 = f4v        # a * fc22^4
            phi2 = f2v        # c * fc22^2
            def E_fit_func(a, c): return f2v * (a * f2v + c)
            f1_label = 'c*fc22^2'
            f2_label = 'a*fc22^4'
            def f1_func(a, c): return c * f2v
            def f2_func(a, c): return a * f4v
            title_formula = f'fc22^2*(a*fc22^2+c), fc22=(1-(r/{rc})^2)^2'
        elif basis == 'fc22r':
            # E = fc22 * (a*r + c)
            # phi1 = fc22*r, phi2 = fc22
            f = fc22(r, rc)
            phi1 = f * r     # a * fc22 * r
            phi2 = f          # c * fc22
            def E_fit_func(a, c): return f * (a * r + c)
            f1_label = 'c*fc22'
            f2_label = 'a*r*fc22'
            def f1_func(a, c): return c * f
            def f2_func(a, c): return a * f * r
            title_formula = f'fc22*(a*r+c), fc22=(1-(r/{rc})^2)^2'
        elif basis == 'fc22r2':
            # E = fc22^2 * (a*r*fc22 + c)
            # phi1 = fc22^3 * r, phi2 = fc22^2
            f = fc22(r, rc)
            f2v = f * f      # fc22^2
            f3v = f2v * f    # fc22^3
            phi1 = f3v * r   # a * r * fc22^3
            phi2 = f2v       # c * fc22^2
            def E_fit_func(a, c): return f2v * (a * r * f + c)
            f1_label = 'c*fc22^2'
            f2_label = 'a*r*fc22^3'
            def f1_func(a, c): return c * f2v
            def f2_func(a, c): return a * r * f3v
            title_formula = f'fc22^2*(a*r*fc22+c), fc22=(1-(r/{rc})^2)^2'
        elif basis == 'fc22r3':
            # E = fc22 * (a*r + c*fc22)
            # phi1 = fc22*r, phi2 = fc22^2
            f = fc22(r, rc)
            f2v = f * f      # fc22^2
            phi1 = f * r     # a * fc22 * r
            phi2 = f2v       # c * fc22^2
            def E_fit_func(a, c): return f * (a * r + c * f)
            f1_label = 'c*fc22^2'
            f2_label = 'a*r*fc22'
            def f1_func(a, c): return c * f2v
            def f2_func(a, c): return a * r * f
            title_formula = f'fc22*(a*r+c*fc22), fc22=(1-(r/{rc})^2)^2'
        elif basis == 'fc22r4':
            # E = fc22^2 * (a*r + c*fc22)
            # phi1 = fc22^2 * r, phi2 = fc22^3
            f = fc22(r, rc)
            f2v = f * f      # fc22^2
            f3v = f2v * f    # fc22^3
            phi1 = f2v * r   # a * r * fc22^2
            phi2 = f3v       # c * fc22^3
            def E_fit_func(a, c): return f2v * (a * r + c * f)
            f1_label = 'c*fc22^3'
            f2_label = 'a*r*fc22^2'
            def f1_func(a, c): return c * f3v
            def f2_func(a, c): return a * r * f2v
            title_formula = f'fc22^2*(a*r+c*fc22), fc22=(1-(r/{rc})^2)^2'
        elif basis == 'fc22r5':
            # E = fc22^2 * (a*r + c*fc22^2)
            # phi1 = fc22^2 * r, phi2 = fc22^4
            f = fc22(r, rc)
            f2v = f * f      # fc22^2
            f4v = f2v * f2v   # fc22^4
            phi1 = f2v * r   # a * r * fc22^2
            phi2 = f4v       # c * fc22^4
            def E_fit_func(a, c): return f2v * (a * r + c * f2v)
            f1_label = 'c*fc22^4'
            f2_label = 'a*r*fc22^2'
            def f1_func(a, c): return c * f4v
            def f2_func(a, c): return a * r * f2v
            title_formula = f'fc22^2*(a*r+c*fc22^2), fc22=(1-(r/{rc})^2)^2'
        elif basis == 'morse1':
            # E = fc * a * (r - r_node)
            # r_node = R0 - ln(2)/beta  (Morse zero crossing: exp(-beta*(r-R0))=2)
            r_node = R0 - np.log(2.0) / MORSE_BETA
            fc = fcut(r, rc)
            phi1 = fc * (r - r_node)  # only 1 parameter: a
            phi2 = np.zeros_like(r)   # dummy, not used
            def E_fit_func(a, c): return fc * a * (r - r_node)
            f1_label = f'r-r_node (r_node={r_node:.3f})'
            f2_label = 'a*(r-r_node)*fc'
            def f1_func(a, c): return fc * (r - r_node)  # shape (before scaling by a)
            def f2_func(a, c): return a * fc * (r - r_node)
            title_formula = f'fc*(a*(r-r_node)), r_node=R0-ln2/beta={r_node:.3f}'
        elif basis == 'morse1b':
            # E = fc^4 * a * (r - r_node)
            r_node = R0 - np.log(2.0) / MORSE_BETA
            fc = fcut(r, rc)
            fc4 = fc**4
            phi1 = fc4 * (r - r_node)
            phi2 = np.zeros_like(r)
            def E_fit_func(a, c): return fc4 * a * (r - r_node)
            f1_label = f'r-r_node (r_node={r_node:.3f})'
            f2_label = 'a*(r-r_node)*fc^4'
            def f1_func(a, c): return fc4 * (r - r_node)
            def f2_func(a, c): return a * fc4 * (r - r_node)
            title_formula = f'fc^4*(a*(r-r_node)), r_node=R0-ln2/beta={r_node:.3f}'
        elif basis in ('morse1c', 'morse1cmp', 'morse1pw'):
            # E = fc22^4 * a * (r - r_node), fc22 = (1-(r/rc)^2)^2
            r_node = R0 - np.log(2.0) / MORSE_BETA
            f = fc22(r, rc)
            fc4 = f**4  # fc22^4 = (1-(r/rc)^2)^8
            phi1 = fc4 * (r - r_node)
            phi2 = np.zeros_like(r)
            def E_fit_func(a, c): return fc4 * a * (r - r_node)
            f1_label = f'r-r_node (r_node={r_node:.3f})'
            f2_label = 'a*(r-r_node)*fc22^4'
            def f1_func(a, c): return fc4 * (r - r_node)
            def f2_func(a, c): return a * fc4 * (r - r_node)
            title_formula = f'fc22^4*(a*(r-r_node)), r_node=R0-ln2/beta={r_node:.3f}'
        elif basis == 'morse2':
            # E = fc * ( a*fc^3*(r-r_node) + c )
            # phi1 = fc^4*(r-r_node), phi2 = fc
            r_node = R0 - np.log(2.0) / MORSE_BETA
            fc = fcut(r, rc)
            fc4 = fc**4
            phi1 = fc4 * (r - r_node)  # a * fc^4 * (r-r_node)
            phi2 = fc                   # c * fc
            def E_fit_func(a, c): return fc * (a * fc**3 * (r - r_node) + c)
            f1_label = 'c*fc'
            f2_label = 'a*fc^4*(r-r_node)'
            def f1_func(a, c): return c * fc
            def f2_func(a, c): return a * fc4 * (r - r_node)
            title_formula = f'fc*(a*fc^3*(r-r_node)+c), r_node={r_node:.3f}'
        elif basis == 'morse2b':
            # E = fc^2 * ( a*fc^2*(r-r_node) + c )
            # phi1 = fc^4*(r-r_node), phi2 = fc^2
            r_node = R0 - np.log(2.0) / MORSE_BETA
            fc = fcut(r, rc)
            fc2 = fc**2
            fc4 = fc**4
            phi1 = fc4 * (r - r_node)  # a * fc^4 * (r-r_node)
            phi2 = fc2                  # c * fc^2
            def E_fit_func(a, c): return fc2 * (a * fc2 * (r - r_node) + c)
            f1_label = 'c*fc^2'
            f2_label = 'a*fc^4*(r-r_node)'
            def f1_func(a, c): return c * fc2
            def f2_func(a, c): return a * fc4 * (r - r_node)
            title_formula = f'fc^2*(a*fc^2*(r-r_node)+c), r_node={r_node:.3f}'
        elif basis == 'morse2c':
            # V = fc^2*(r-r_node)*(a*fc^2+c), p=2,q=2
            r_node = R0 - np.log(2.0) / MORSE_BETA
            fc = cutoff_array(r, rc, morse_cutoff)
            fc2 = fc*fc
            fc4 = fc2*fc2
            dr = r-r_node
            phi1 = fc4*dr
            phi2 = fc2*dr
            def E_fit_func(a, c): return fc2*dr*(a*fc2+c)
            f1_label = f'c*{morse_cutoff}^2*(r-r_node)'
            f2_label = f'a*{morse_cutoff}^4*(r-r_node)'
            def f1_func(a, c): return c*fc2*dr
            def f2_func(a, c): return a*fc4*dr
            title_formula = f'fc^2*(r-r_node)*(a*fc^2+c), fc={morse_cutoff}, r_node={r_node:.3f}'
        elif basis == 'morse2d':
            # V = fc*(r-r_node)*(a*fc^3+c), p=1,q=3
            r_node = R0 - np.log(2.0) / MORSE_BETA
            fc = cutoff_array(r, rc, morse_cutoff)
            fc2 = fc*fc
            fc3 = fc2*fc
            fc4 = fc2*fc2
            dr = r-r_node
            phi1 = fc4*dr
            phi2 = fc*dr
            def E_fit_func(a, c): return fc*dr*(a*fc3+c)
            f1_label = f'c*{morse_cutoff}*(r-r_node)'
            f2_label = f'a*{morse_cutoff}^4*(r-r_node)'
            def f1_func(a, c): return c*fc*dr
            def f2_func(a, c): return a*fc4*dr
            title_formula = f'fc*(r-r_node)*(a*fc^3+c), fc={morse_cutoff}, r_node={r_node:.3f}'
        else:
            raise ValueError(f"Unknown basis: {basis}")

        weights = np.exp(-E_ref / kT)
        Wdiag = weights
        if basis in ('morse1', 'morse1b', 'morse1c', 'morse1cmp', 'morse1pw'):
            # 1-parameter fit: a = (phi1^T W y) / (phi1^T W phi1)
            phi1W = phi1 * Wdiag
            a_fit = float(phi1W @ E_ref / (phi1W @ phi1))
            c_fit = 0.0
        else:
            A = np.column_stack([phi1, phi2])
            ATWA = A.T @ np.diag(Wdiag) @ A
            ATWy = A.T @ (Wdiag * E_ref)
            a_fit, c_fit = np.linalg.solve(ATWA, ATWy)
        E_fit = E_fit_func(a_fit, c_fit)

        ax = axes[idx]
        ax.plot(r, E_ref, 'k:', linewidth=2.0, label='Ref (Morse)')
        ax.plot(r, E_fit, 'k-', linewidth=0.5, label=f'Fit: a={a_fit:.3f}')
        if basis == 'morse1cmp':
            # Also fit with smoothstep^4 and overlay
            r_node_s = R0 - np.log(2.0) / MORSE_BETA
            fc_s = fcut(r, rc)
            fc4_s = fc_s**4
            phi1_s = fc4_s * (r - r_node_s)
            phi1W_s = phi1_s * Wdiag
            a_s = float(phi1W_s @ E_ref / (phi1W_s @ phi1_s))
            E_fit_s = fc4_s * a_s * (r - r_node_s)
            ax.plot(r, E_fit_s, 'r-', linewidth=1.0, alpha=0.7, label=f'smoothstep^4: a={a_s:.3f}')
            ax.plot(r, E_fit, 'b-', linewidth=1.0, alpha=0.7, label=f'fc22^4: a={a_fit:.3f}')
        elif basis == 'morse1pw':
            # Compare fc^n and fc22^n for n=1,2,3,4
            r_node_p = R0 - np.log(2.0) / MORSE_BETA
            fc_s = fcut(r, rc)
            fc_f = fc22(r, rc)
            colors_fc = ['r', 'g', 'b', 'm']
            colors_fc22 = ['r', 'g', 'b', 'm']
            linestyles = ['-', '--']  # solid=smoothstep, dashed=fc22
            for n in range(1, 5):
                # smoothstep^n
                env_s = fc_s**n
                phi1_s = env_s * (r - r_node_p)
                phi1W_s = phi1_s * Wdiag
                a_s = float(phi1W_s @ E_ref / (phi1W_s @ phi1_s))
                E_s = env_s * a_s * (r - r_node_p)
                ax.plot(r, E_s, color=colors_fc[n-1], linestyle='-', linewidth=1.0, alpha=0.8, label=f'fc^{n}: a={a_s:.4f}')
                # fc22^n
                env_f = fc_f**n
                phi1_f = env_f * (r - r_node_p)
                phi1W_f = phi1_f * Wdiag
                a_f = float(phi1W_f @ E_ref / (phi1W_f @ phi1_f))
                E_f = env_f * a_f * (r - r_node_p)
                ax.plot(r, E_f, color=colors_fc22[n-1], linestyle='--', linewidth=1.0, alpha=0.8, label=f'fc22^{n}: a={a_f:.4f}')
        if basis not in ('morse1', 'morse1b', 'morse1c', 'morse1cmp', 'morse1pw'):
            f1 = f1_func(a_fit, c_fit)
            f2 = f2_func(a_fit, c_fit)
            ax.plot(r, f1, 'm-', linewidth=1.0, alpha=0.8, label=f'{f1_label} ({c_fit:.3f})')
            ax.plot(r, f2, 'c-', linewidth=1.0, alpha=0.8, label=f'{f2_label} ({a_fit:.3f})')
        ax2 = ax.twinx()
        ax2.plot(r, weights, 'g-', linewidth=0.5, alpha=0.7, label='weights')
        ax2.set_ylabel('Boltzmann weight', color='green')
        ax2.set_ylim(0, max(weights)*1.1)
        vmin = E_ref.min()
        vmax = -2.0 * vmin if vmin < 0 else E_ref.max() * 0.5
        ax.set_ylim(vmin * 1.15, vmax)
        ax.set_title(f'{label}  (R0={R0:.2f}, E0={E0:.4f}, q={qi}*{qj}, kT={kT:.3f})')
        ax.set_xlabel('r [Å]')
        ax.set_ylabel('Energy [eV]')
        ax.legend(loc='upper right')
        ax.axhline(0, color='gray', linewidth=0.5)
        print(f"{label}: a={a_fit:.6f}, c={c_fit:.6f}, Emin={Emin:.4f}, kT={kT:.4f}")
        if analytical:
            blend_params = None
            if analytical_mode == 'pure-tail':
                pure = mix_morse_pair_pure_tail(
                    Ri, Ei, Rj, Ej, MORSE_BETA,
                    cutoff_kind=morse_cutoff,
                    envelope_power=pure_tail_power,
                )
                rc_an = pure['rc']
                rn_an = pure['r_node']
                a_an = pure['a']
                c_an = 0.0
                fc_an = cutoff_array(r, rc_an, morse_cutoff)
                E_an = a_an*(r-rn_an)*fc_an**pure_tail_power
                label_an = (f'Analytical pure-tail f^{pure_tail_power:g}: '
                            f'a={a_an:.3f}, rc={rc_an:.3f}')
            elif analytical_mode == 'tail-blend':
                blend_params = mix_morse_pair_tail_blend(
                    Ri, Ei, Rj, Ej, MORSE_BETA, blend=tail_blend,
                    cutoff_kind=morse_cutoff, power_lo=2, power_hi=4,
                )
                rc_an = blend_params['rc']
                rn_an = blend_params['r_node']
                c_an = blend_params['c_lo']
                a_an = blend_params['c_hi']
                fc_an = cutoff_array(r, rc_an, morse_cutoff)
                E_an = (r-rn_an)*(c_an*fc_an**2 + a_an*fc_an**4)
                label_an = (f'Analytical blend={tail_blend:.2f}: '
                            f'c2={c_an:.3f}, a4={a_an:.3f}, '
                            f'rc={rc_an:.3f}')
            else:
                rc_an = rc
                a_an, c_an, rn_an = analytical_coeffs(
                    R0, E0, MORSE_BETA, rc_an, basis,
                    cutoff_kind=morse_cutoff, mode=analytical_mode,
                )
                E_an = None if a_an is None else E_fit_func(a_an, c_an)
                label_an = (f'Analytical {analytical_mode}: a={a_an:.3f}, '
                            f'c={c_an:.3f}, rc={rc_an:.3f}') \
                    if a_an is not None else ''

            if a_an is not None:
                ax.plot(
                    r, E_an, 'r--', linewidth=1.0, alpha=0.8,
                    label=label_an,
                )
                if blend_params is not None:
                    residuals = tail_blend_residuals(
                        blend_params, MORSE_BETA)
                    rc_critical = rc_an
                else:
                    residuals = analytical_residuals(
                        R0, E0, MORSE_BETA, rc_an, a_an, c_an, rn_an,
                        basis=basis, cutoff_kind=morse_cutoff,
                    )
                    rc_critical = critical_cutoff_for_pure_tail(
                        R0, MORSE_BETA, cutoff_kind=morse_cutoff,
                        envelope_power=sum(morse_basis_pq(basis)),
                    )
                print(
                    f"  analytical[{analytical_mode},{morse_cutoff}]: "
                    f"a={a_an:.6f}, c={c_an:.6f}, r_node={rn_an:.6f}, "
                    f"rc={rc_an:.6f}"
                )
                if blend_params is not None:
                    print(
                        f"    blend={tail_blend:.6f}, "
                        f"n_eff={blend_params['effective_power']:.6f}, "
                        f"powers={blend_params['power_lo']},"
                        f"{blend_params['power_hi']}"
                    )
                print(
                    "    residuals: "
                    f"V(node)={residuals['node_value_error']:+.3e}, "
                    f"V(R0)+E0={residuals['depth_error']:+.3e}, "
                    f"V'(R0)={residuals['minimum_slope_error']:+.3e}, "
                    f"curvature={residuals['curvature']:.6g} "
                    f"(Morse {residuals['morse_curvature']:.6g})"
                )
                print(
                    f"    tail={residuals['tail_side']}; "
                    f"critical rc={rc_critical:.6f}"
                )
            else:
                print(f"  analytical: not available for basis={basis}")

    fig.suptitle(f'Unified radial fit: {title_formula}  |  kT=|Emin|  |  basis={basis}', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Fit unified radial potentials to Morse+Coulomb')
    parser.add_argument('--basis', choices=['lorenz', 'fc22', 'fc22r', 'fc22r2', 'fc22r3', 'fc22r4', 'fc22r5', 'morse1', 'morse1b', 'morse1c', 'morse1cmp', 'morse1pw', 'morse2', 'morse2b', 'morse2c', 'morse2d'], default='lorenz', help='Basis function set')
    parser.add_argument('--rc', type=float, default=RC, help='Cutoff radius [Å]')
    parser.add_argument('--w', type=float, default=W, help='Lorenzian width [Å] (lorenz basis only)')
    parser.add_argument('--analytical', action='store_true', help='Overlay analytical coefficient prediction')
    parser.add_argument('--morse-cutoff', choices=['smoothstep', 'fc22'], default='smoothstep', help='Cutoff used by morse2c/morse2d')
    parser.add_argument('--analytical-mode', choices=['minimum', 'node-slope', 'pure-tail', 'tail-blend'], default='minimum', help='pure-tail: analytical f^4; tail-blend: analytical common-cutoff mixture of f^2 and f^4')
    parser.add_argument('--tail-blend', type=float, default=0.5, help='f^2/f^4 blend in [0,1]: 0=pure f^2, 1=pure f^4')
    parser.add_argument('--pure-tail-power', type=float, default=4.0, help='Envelope power n for --analytical-mode pure-tail (e.g. 2 or 4)')
    parser.add_argument('--compact-exp-demo', action='store_true', help='Compare compact exponential Morse powers and show the smooth epair limit')
    parser.add_argument('--exp-powers', type=str, default='2,4,8', help='Comma-separated compact-exponential powers')
    parser.add_argument('--epair-width', type=float, default=0.6, help='Epair soft-radius width w [Angstrom]')
    parser.add_argument('--epair-alpha', type=float, default=0.0, help='Epair repulsive fraction alpha in [0,1]')
    parser.add_argument('--soft-radius', choices=['sqrt', 'rational'], default='sqrt', help='Soft-radius map for compact exponential')
    args = parser.parse_args()
    if args.compact_exp_demo:
        powers = tuple(int(x.strip()) for x in args.exp_powers.split(',') if x.strip())
        run_compact_exp_demo(
            powers=powers, epair_width=args.epair_width,
            epair_alpha=args.epair_alpha, soft_kind=args.soft_radius,
        )
        return
    run_fit(
        basis=args.basis, rc=args.rc, w=args.w, analytical=args.analytical,
        morse_cutoff=args.morse_cutoff, analytical_mode=args.analytical_mode,
        tail_blend=args.tail_blend,
        pure_tail_power=args.pure_tail_power,
    )

if __name__ == '__main__':
    main()
