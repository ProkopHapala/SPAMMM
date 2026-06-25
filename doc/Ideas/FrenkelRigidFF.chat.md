# USER

I have idea for computational project to simulate tip enhanced photoluminescence measurements where molecule is on the tip (like PTCDA hanging attached via coordination bond to metal apex atom)

Now I have also molecules of the substrate let's say all are PTCDA for simplicity although it is not important. We consider each molecule have some relevant optical transition (exciton) with different energies which couple by Frenkel Hamiltonian. In simplest case it is dipole dipole interaction, in General matrix element of coulomb operator between transition densities of each molecule (computed for isolate molecules). 

Now consider we have cluster of molecules on substrate and one molecule hanging on the tip. We have m-states on each molecule and N molecules in cluster so we have (N+1)*m dimension of Frenkel Hamiltonian. We have also Stark Shift on diagonal for each state.


As we move the tip measuring image (drag molecule on tip and let it relax by forcefield - in simple case just right translation and rotation as the molecule of PTCDA is pretty stiff and rigid)

So in the Frenkel Hamiltonian only coupling between this tip molecule and surface change. We want to diagonalize this Frankel Hamiltonian for every pixel of the image on GPU. This should not be too costly in this reduced basis.

If we neglect these Stark Shift we can solve this using shur complement (we precompute the cluster on surface and then we only solve the coupling with the tip which change), that will make it even faster but we can afford do also full diagonalization with stark shifts everywhere.

The interactions can be computed either by multipole expansion or by projection of transition density of each molecule to small number of point charges. When the molecules are close in self-assembled cluster the multipole expansion doesn't converge very well then the charge projection is better.

Think it through write exact equations and pseudo code as rigorous base for numerical implementation. Focus on physics and math first not specific programing language. Also give me feedback what can be chaveates or if you have better ideas how to do this.

---

# ChatGPT 5.5 medium

This is a very good project idea because it sits in the “sweet spot”: the electronic Hilbert space is small enough for brute-force diagonalization per pixel, but the geometry-dependent physics is rich enough to produce nontrivial contrast. The key is to separate **static surface aggregate physics** from **moving tip-molecule physics**.

I will denote the tip molecule as molecule `0`, and the surface molecules as `i = 1 ... N`. Each molecule has `m` local excited states, e.g. bright π–π* transitions, low dark states, CT-like states if needed.

The total Frenkel basis is

[
|i a\rangle
]

where `i` is the molecule and `a = 1...m` is the local excitation on that molecule.

The exciton wavefunction is

[
|\Psi_\lambda\rangle
====================

\sum_{i=0}^{N}
\sum_{a=1}^{m}
C^{(\lambda)}_{ia}
|i a\rangle .
]

The total Hamiltonian is

[
H_{ia,jb}
=========

\delta_{ij}\delta_{ab}
\left[
E_{ia}^{0}
+
\Delta E_{ia}^{\rm Stark}
+
\Delta E_{ia}^{\rm env}
\right]
+
(1-\delta_{ij}) J_{ia,jb}.
]

Here:

[
E_{ia}^{0}
]

is the isolated-molecule transition energy,

[
\Delta E_{ia}^{\rm Stark}
]

is the local field shift,

[
\Delta E_{ia}^{\rm env}
]

is any static substrate/screening/image-charge correction, and

[
J_{ia,jb}
]

is the excitonic coupling between transition densities.

For a rigid molecule, all quantities associated with its local transition are transformed by the rigid pose:

[
\mathbf r_{i\alpha}
===================

\mathbf R_i \mathbf r^0_{a\alpha}
+
\mathbf T_i ,
]

where (\mathbf R_i) is the molecular rotation and (\mathbf T_i) the translation.

---

## 1. Transition-density coupling

The most physical expression is the Coulomb matrix element between transition densities:

[
J_{ia,jb}
=========

\iint
\rho^{\rm tr}*{ia}(\mathbf r)
,
W(\mathbf r,\mathbf r')
,
\rho^{\rm tr}*{jb}(\mathbf r')
,
d\mathbf r d\mathbf r' .
]

In vacuum,

[
W(\mathbf r,\mathbf r')
=======================

\frac{1}{4\pi\varepsilon_0}
\frac{1}{|\mathbf r-\mathbf r'|}.
]

In molecular-simulation units,

[
J_{ia,jb}
=========

k_e
\iint
\frac{
\rho^{\rm tr}*{ia}(\mathbf r)
\rho^{\rm tr}*{jb}(\mathbf r')
}{
|\mathbf r-\mathbf r'|
}
d\mathbf r d\mathbf r',
]

with

[
k_e = 14.3996448915 \ {\rm eV,Å}.
]

If each transition density is approximated by transition charges,

[
\rho^{\rm tr}*{ia}(\mathbf r)
\approx
\sum*{\alpha \in ia}
q^{\rm tr}*{a\alpha}
\delta(\mathbf r-\mathbf r*{i\alpha}),
]

then

[
J_{ia,jb}
=========

k_e
\sum_{\alpha \in a}
\sum_{\beta \in b}
\frac{
q^{\rm tr}*{a\alpha}
q^{\rm tr}*{b\beta}
}{
|\mathbf r_{i\alpha}-\mathbf r_{j\beta}|
}.
]

For close molecules, this is much better than dipole-dipole because it preserves the spatial structure of the transition density. The transition charges should satisfy

[
\sum_\alpha q^{\rm tr}_{a\alpha}=0
]

for a neutral optical transition. Their dipole moment is

[
\boldsymbol\mu_{ia}
===================

\sum_\alpha q^{\rm tr}*{a\alpha}\mathbf r*{i\alpha}.
]

For distant molecules, one can switch to the dipole approximation:

[
J_{ia,jb}^{\rm dip}
===================

k_e
\left[
\frac{\boldsymbol\mu_{ia}\cdot\boldsymbol\mu_{jb}}{R^3}
-------------------------------------------------------

3
\frac{
(\boldsymbol\mu_{ia}\cdot\mathbf R)
(\boldsymbol\mu_{jb}\cdot\mathbf R)
}{R^5}
\right],
]

where

[
\mathbf R = \mathbf T_i-\mathbf T_j.
]

But for PTCDA-like close-packed systems, transition charges are probably the safer default.

A useful hybrid scheme is:

[
J_{ia,jb}
=========

\begin{cases}
J^{\rm charges}*{ia,jb}, & R*{ij}<R_{\rm switch},\
J^{\rm multipole}*{ia,jb}, & R*{ij}>R_{\rm switch}.
\end{cases}
]

With a smooth switching function (s(R)),

[
J =
s(R)J^{\rm charges}
+
[1-s(R)]J^{\rm multipole}.
]

This avoids discontinuities during tip scanning.

---

## 2. Stark shift

For each local transition (a) on molecule (i), the electric field changes the transition energy approximately as

[
\Delta E^{\rm Stark}_{ia}
=========================

## -\Delta\boldsymbol\mu_{ia}\cdot \mathbf E_i

\frac{1}{2}
\mathbf E_i^T
\Delta\boldsymbol\alpha_{ia}
\mathbf E_i
+\cdots
]

where

[
\Delta\boldsymbol\mu_{ia}
=========================

## \boldsymbol\mu^{\rm exc}_{ia}

\boldsymbol\mu^{\rm gs}_{i},
]

and

[
\Delta\boldsymbol\alpha_{ia}
============================

## \boldsymbol\alpha^{\rm exc}_{ia}

\boldsymbol\alpha^{\rm gs}_{i}.
]

The simplest model keeps only the linear term:

[
\Delta E^{\rm Stark}_{ia}
=========================

-\Delta\boldsymbol\mu_{ia}\cdot \mathbf E(\mathbf R_i).
]

But for centrosymmetric or nearly symmetric transitions, the quadratic term can matter more than the linear term.

If the field varies across a large molecule, one can do a more microscopic version using difference charges between excited and ground state:

[
\Delta E^{\rm Stark}_{ia}
=========================

\sum_\alpha
\Delta q_{a\alpha}
\phi(\mathbf r_{i\alpha}),
]

where

[
\Delta q_{a\alpha}
==================

## q^{\rm exc}_{a\alpha}

q^{\rm gs}_{\alpha}.
]

This is often better than evaluating the field only at the molecular center, especially near a metallic tip.

Important distinction:

* **transition density** (\rho^{\rm tr}) controls excitonic coupling and oscillator strength;
* **difference density** (\Delta\rho = \rho^{\rm exc}-\rho^{\rm gs}) controls Stark shift and reorganization.

They are different objects.

---

## 3. Hamiltonian block structure

Let the surface aggregate block be

[
H_S \in \mathbb R^{Nm \times Nm}.
]

Let the tip block be

[
H_T \in \mathbb R^{m \times m}.
]

Usually (H_T) is nearly diagonal in the isolated-molecule excitation basis:

[
(H_T)_{ab}
==========

\delta_{ab}
\left(
E^0_{0a}
+
\Delta E^{\rm Stark}*{0a}
\right)
+
\Delta*{ab}^{\rm local},
]

where (\Delta_{ab}^{\rm local}) may include local field mixing between excited states if needed.

The coupling block between tip and surface is

[
V_{a,Ib}
========

J_{0a,Ib},
]

where I use (I=1...N) for surface molecules.

The total Hamiltonian is then

[
H =
\begin{pmatrix}
H_T & V \
V^\dagger & H_S
\end{pmatrix}.
]

During scanning:

* (H_S) is constant if surface molecules are fixed and their Stark shifts are fixed;
* (H_T) changes mildly because the tip molecule samples different local field;
* (V) changes strongly because the tip molecule moves relative to the surface aggregate.

If the tip also perturbs the electric field on surface molecules, then (H_S) gains a pixel-dependent diagonal correction:

[
H_S(\mathbf R_{\rm tip})
========================

H_S^0
+
D_S(\mathbf R_{\rm tip}),
]

where (D_S) is diagonal or block-diagonal. This breaks the clean Schur-complement acceleration but still keeps the problem small.

---

## 4. Pre-diagonalization of the static surface cluster

If Stark shifts on the surface are fixed, prediagonalize

[
H_S U = U \varepsilon,
]

with

[
U^\dagger H_S U = \operatorname{diag}(\varepsilon_\nu).
]

The surface exciton states are

[
|\nu\rangle
===========

\sum_{Ib}
U_{Ib,\nu}
|Ib\rangle .
]

The tip-surface coupling in the surface-exciton basis is

[
\tilde V_{a\nu}
===============

\sum_{Ib}
V_{a,Ib} U_{Ib,\nu}.
]

Then the Hamiltonian becomes

[
\tilde H =
\begin{pmatrix}
H_T & \tilde V \
\tilde V^\dagger & \varepsilon
\end{pmatrix}.
]

This is already simpler because the surface block is diagonal. You still diagonalize a full ((N+1)m) matrix per pixel, but construction is cheaper and the static part is trivial.

Even better, you can truncate the surface exciton basis to states near the relevant optical window:

[
|\varepsilon_\nu - E_{\rm laser}| < \Delta E_{\rm window}.
]

Then the dimension may drop from (Nm) to (n_{\rm active}), often much smaller than (Nm).

---

## 5. Schur-complement formulation

The eigenvalue problem is

[
\begin{pmatrix}
H_T & \tilde V \
\tilde V^\dagger & \varepsilon
\end{pmatrix}
\begin{pmatrix}
c_T \
c_S
\end{pmatrix}
=============

E
\begin{pmatrix}
c_T \
c_S
\end{pmatrix}.
]

The surface component satisfies

[
(\varepsilon - E)c_S
====================

-\tilde V^\dagger c_T,
]

so

[
c_S
===

(E-\varepsilon)^{-1}
\tilde V^\dagger c_T.
]

Substituting into the tip equation gives an energy-dependent effective Hamiltonian:

[
\left[
H_T
+
\Sigma(E)
\right]
c_T
===

E c_T,
]

where

[
\Sigma(E)
=========

\tilde V
(E-\varepsilon)^{-1}
\tilde V^\dagger .
]

In components:

[
\Sigma_{ab}(E)
==============

\sum_\nu
\frac{
\tilde V_{a\nu}\tilde V^**{b\nu}
}{
E-\varepsilon*\nu
}.
]

This is very useful physically: the surface aggregate acts as a structured self-energy for the tip excitation.

If you care about spectra rather than individual eigenstates, use the Green’s function directly:

[
G_T(E)
======

\left[
E+i\eta
-------

## H_T

\Sigma(E+i\eta)
\right]^{-1}.
]

with

[
\Sigma_{ab}(E+i\eta)
====================

\sum_\nu
\frac{
\tilde V_{a\nu}\tilde V^**{b\nu}
}{
E+i\eta-\varepsilon*\nu
}.
]

This is probably better than diagonalizing if the experiment measures an energy-resolved PL spectrum at every pixel.

The local spectral function on the tip molecule is

[
A_T(E)
======

-\frac{1}{\pi}
\operatorname{Im}
\operatorname{Tr} G_T(E).
]

If the excitation is localized by the optical near-field mostly on the tip molecule, then the absorption-like signal is

[
I_{\rm abs}(E)
\propto
-\frac{1}{\pi}
\operatorname{Im}
\left[
\boldsymbol\mu_T^\dagger
G_T(E)
\boldsymbol\mu_T
\right].
]

This avoids full diagonalization and reduces the per-energy cost to inversion of an (m \times m) matrix. If (m=1,2,3), this is extremely cheap.

For (m=1), the expression becomes scalar:

[
G_T(E)
======

\frac{1}{
E+i\eta
-------

## E_T

\sum_\nu
\frac{|\tilde V_{\nu}|^2}{E+i\eta-\varepsilon_\nu}
}.
]

This gives Fano-like line shifts and broadenings automatically.

---

## 6. Optical excitation and PL intensity

Diagonalizing the Frenkel Hamiltonian gives exciton states (\lambda) with coefficients (C^{(\lambda)}_{ia}).

The transition dipole of exciton (\lambda) is

[
\boldsymbol\mu_\lambda
======================

\sum_{ia}
C^{(\lambda)}*{ia}
\boldsymbol\mu*{ia}.
]

More generally, if the excitation field is spatially inhomogeneous near the tip, the excitation amplitude is

[
M_\lambda^{\rm exc}
===================

\sum_{ia}
C^{(\lambda)}*{ia}
,
\boldsymbol\mu*{ia}\cdot
\mathbf E_{\rm exc}(\mathbf R_i).
]

The radiative emission strength into a far-field polarization (\hat{\mathbf e}) is

[
M_\lambda^{\rm em}
==================

\hat{\mathbf e}\cdot
\boldsymbol\mu_\lambda.
]

For orientation-averaged detection,

[
f_\lambda
\propto
|\boldsymbol\mu_\lambda|^2.
]

A simple PL image at fixed detection window could be

[
I(\mathbf R_{\rm tip})
======================

\sum_\lambda
P_\lambda
,
\Gamma_\lambda^{\rm rad}
,
L(E_{\rm det}-E_\lambda,\gamma_\lambda),
]

where

[
L(E-E_\lambda,\gamma)
=====================

\frac{\gamma^2}{(E-E_\lambda)^2+\gamma^2}.
]

The population (P_\lambda) is the difficult part. In the simplest coherent/linear model,

[
P_\lambda
\propto
|M_\lambda^{\rm exc}|^2
L(E_{\rm laser}-E_\lambda,\gamma_{\rm abs}).
]

Then

[
I(\mathbf R_{\rm tip})
======================

\sum_\lambda
|M_\lambda^{\rm exc}|^2
\Gamma_\lambda^{\rm rad}
L(E_{\rm laser}-E_\lambda,\gamma_{\rm abs})
L(E_{\rm det}-E_\lambda,\gamma_{\rm em}).
]

This is the “fast Frenkel image” model.

If relaxation between excitons is important, then after optical excitation the population should be propagated by a rate equation:

[
\frac{dP_\lambda}{dt}
=====================

G_\lambda
+
\sum_{\lambda'}
k_{\lambda'\to\lambda}P_{\lambda'}
----------------------------------

P_\lambda
\sum_{\lambda'}
k_{\lambda\to\lambda'}
----------------------

(\Gamma_\lambda^{\rm rad}+\Gamma_\lambda^{\rm nr})P_\lambda.
]

At steady state,

[
0 =
G + K P.
]

Then the PL signal is

[
I =
\sum_\lambda
\Gamma_\lambda^{\rm rad}P_\lambda.
]

This is more physical but more expensive and introduces poorly known phonon relaxation rates. I would start with the linear spectral model first.

---

## 7. Special case: one transition per molecule

For (m=1), the basis is just one excitation per molecule. The Hamiltonian is

[
H_{ij}
======

\delta_{ij}
\left[
E_i^0+\Delta E_i^{\rm Stark}
\right]
+
(1-\delta_{ij})J_{ij}.
]

The tip-surface block is the vector

[
v_i = J_{0i}.
]

The full Hamiltonian is

[
H =
\begin{pmatrix}
E_0 & \mathbf v^T \
\mathbf v & H_S
\end{pmatrix}.
]

After prediagonalizing (H_S),

[
\tilde v_\nu
============

\sum_i v_i U_{i\nu}.
]

The tip Green’s function is

[
G_0(E)
======

\frac{1}{
E+i\eta
-------

## E_0

\sum_\nu
\frac{|\tilde v_\nu|^2}{E+i\eta-\varepsilon_\nu}
}.
]

This is a beautiful minimal model. It will already show:

* resonance shifts of the tip molecule;
* hybridization with aggregate excitons;
* avoided crossings as tip moves;
* bright/dark state mixing;
* spatial maps reflecting nodal structure of surface excitons;
* quenching if nonradiative channels are assigned to surface or metal states.

---

## 8. Pseudocode: physical workflow

### Precomputation

```text
Input:
    molecular geometry of isolated PTCDA
    local excited states a = 1...m
    transition charges q_tr[a,alpha]
    transition dipoles mu_tr[a]
    difference charges dq[a,alpha] or Delta_mu[a], Delta_alpha[a]
    surface molecule poses {R_i, T_i}, i=1...N
    static electric potential/field from substrate and electrodes
    optional dielectric/screening model

For each surface molecule i:
    transform transition-charge sites:
        r[i,a,alpha] = R_i * r0[a,alpha] + T_i
    compute Stark shifts:
        dE_stark[i,a] = sum_alpha dq[a,alpha] * phi(r[i,a,alpha])
        or dE_stark[i,a] = -Delta_mu[i,a] dot E(T_i)

Build surface Hamiltonian H_S:
    for i=1...N:
        for a=1...m:
            H_S[ia,ia] = E0[a] + dE_stark[i,a]
    for i<j:
        for a=1...m:
            for b=1...m:
                J = CoulombTransitionDensityCoupling(i,a,j,b)
                H_S[ia,jb] = J
                H_S[jb,ia] = J

Diagonalize H_S:
    H_S U = U eps

Optionally select active surface excitons:
    keep nu where abs(eps[nu]-E_window_center) < E_window_width
```

### Per-pixel calculation

```text
For each image pixel p in parallel:

    Given tip base position X_p

    Relax rigid tip molecule:
        pose_tip = RelaxRigidTipMolecule(X_p, surface_forcefield)
        R_0, T_0 = pose_tip

    Transform tip transition-charge sites:
        r[0,a,alpha] = R_0 * r0[a,alpha] + T_0

    Compute tip Stark shifts:
        dE_stark[0,a] = sum_alpha dq[a,alpha] * phi(r[0,a,alpha])

    Build tip block H_T:
        for a=1...m:
            H_T[a,a] = E0_tip[a] + dE_stark[0,a]
        optionally include local field mixing H_T[a,b]

    Build tip-surface coupling V:
        for a=1...m:
            for i=1...N:
                for b=1...m:
                    V[a, i*b] = CouplingTransitionCharges(tip,a, surface i,b)

    Transform to surface exciton basis:
        for a=1...m:
            for nu in active_surface_states:
                Vtilde[a,nu] = sum_{i,b} V[a,i*b] * U[i*b,nu]

    Option A: full reduced diagonalization:
        Assemble H_red:
            H_red[tip,tip] = H_T
            H_red[surface,surface] = diag(eps)
            H_red[tip,surface] = Vtilde
        Diagonalize H_red
        Compute optical intensities
        Store selected observables

    Option B: Green function / Schur complement:
        for each energy E_k:
            Sigma[a,b] = sum_nu Vtilde[a,nu] Vtilde[b,nu]^* / (E_k+i*eta - eps[nu])
            G = inverse(E_k+i*eta - H_T - Sigma)
            I[p,E_k] = -Im( mu_tip^dagger G mu_tip ) / pi

    Output:
        spectral map I[p,E]
        or integrated image I[p]
        or exciton energy shifts E_lambda[p]
```

---

## 9. GPU-oriented interpretation

The natural parallelization is:

```text
one pixel = one workgroup
```

Inside one workgroup:

* one thread or subgroup computes one tip-surface molecule pair;
* transition-charge summations are local loops;
* the Hamiltonian dimension is small;
* diagonalization or Green’s-function evaluation happens inside the workgroup.

For example, if (N=50), (m=1), full dimension is (51). That is already manageable. If (m=2), dimension is (102), still not crazy, but full diagonalization per pixel becomes more expensive.

The Schur/Green approach is more GPU-friendly if (m) is small because instead of diagonalizing a (50)- to (200)-dimensional matrix, you evaluate sums over precomputed surface modes:

[
\Sigma_{ab}(E)
==============

\sum_\nu
\frac{
\tilde V_{a\nu}\tilde V^**{b\nu}
}{
E+i\eta-\varepsilon*\nu
}.
]

That is a reduction over (\nu), followed by inversion of a tiny (m\times m) matrix.

For (m=1), no matrix inversion is needed.

---

## 10. Important caveats

### 1. Frenkel basis may fail for strong charge transfer

PTCDA aggregates can have intermolecular CT character, especially close to metal surfaces. Frenkel excitons assume local neutral excitations:

[
|i^*j\rangle
]

but not charge-transfer states:

[
|i^+j^-\rangle.
]

If the tip molecule comes close to the surface molecule, CT states may mix strongly. Then you may need an extended basis:

[
{
|i,a\rangle_{\rm Frenkel},
|i^+j^-\rangle_{\rm CT}
}.
]

This is especially important for quenching, nonradiative decay, and metal-mediated charge transfer.

### 2. Transition charges are not unique

Fitted transition charges reproduce some moments of the transition density, but different charge models can give different near-field couplings. For close molecules, one should validate against direct transition-density Coulomb integrals for several representative dimers.

A good compromise:

* use atom-centered transition charges initially;
* later improve with bond-centered charges or distributed multipoles;
* benchmark against TDDFT/TDA transition-density cube integrals for selected geometries.

### 3. Screening is not just a scalar dielectric constant

Near a metal tip and substrate, Coulomb interaction is screened strongly and anisotropically. The real coupling should be

[
J_{ia,jb}
=========

\iint
\rho^{\rm tr}*{ia}(\mathbf r)
W*{\rm screened}(\mathbf r,\mathbf r')
\rho^{\rm tr}_{jb}(\mathbf r')
d\mathbf r d\mathbf r'.
]

A crude correction is

[
J \rightarrow \frac{J}{\varepsilon_{\rm eff}}.
]

A better correction is image-charge screening. For a planar metal surface at (z=0), one can approximate the screened interaction by adding image transition charges:

[
W(\mathbf r,\mathbf r')
\approx
k_e
\left[
\frac{1}{|\mathbf r-\mathbf r'|}
--------------------------------

\frac{1}{|\mathbf r-\mathbf r'_{\rm image}|}
\right].
]

This can strongly suppress transition dipoles perpendicular or parallel to the surface differently. For a metallic tip, the geometry is harder, but one can use precomputed Green’s functions or boundary-element approximations.

### 4. The tip enhances both excitation and decay

TEPL is not only a Frenkel Hamiltonian problem. The metal tip changes:

[
E_\lambda
]

but also:

[
\Gamma_\lambda^{\rm rad},
\qquad
\Gamma_\lambda^{\rm nr},
\qquad
\mathbf E_{\rm exc}(\mathbf r).
]

The measured PL contrast may be dominated by:

* near-field excitation enhancement;
* Purcell enhancement;
* metal quenching;
* exciton transfer to dark states;
* Stark shift;
* mechanical geometry change;
* hybridization with surface excitons.

So the Frenkel Hamiltonian gives the excitonic eigenstates, but the observable requires a model for optical pumping and decay.

A minimal practical signal model could be:

[
I(\mathbf R_{\rm tip})
======================

\sum_\lambda
|\sum_{ia} C_{ia}^{(\lambda)}
\boldsymbol\mu_{ia}\cdot \mathbf E_{\rm tip}(\mathbf R_i)|^2
,
\Gamma^{\rm rad}*\lambda
,
L(E*{\rm laser}-E_\lambda,\gamma)
,
Q_\lambda,
]

where (Q_\lambda) is a phenomenological quantum yield:

[
Q_\lambda
=========

\frac{\Gamma_\lambda^{\rm rad}}
{\Gamma_\lambda^{\rm rad}+\Gamma_\lambda^{\rm nr}}.
]

You can model metal quenching by assigning a distance-dependent nonradiative rate:

[
\Gamma_{\lambda}^{\rm nr}
=========================

\sum_{ia}
|C_{ia}^{(\lambda)}|^2
\Gamma^{\rm nr}_{ia}(z_i),
]

for example

[
\Gamma^{\rm nr}_{ia}(z)
=======================

\Gamma_0^{\rm nr}
+
A e^{-z/z_0}.
]

### 5. Full diagonalization may be unnecessary

If the output is an energy-resolved spectrum, the Green’s-function method is more natural than diagonalization.

Diagonalization gives eigenvalues and eigenvectors. But spectroscopy usually needs

[
\operatorname{Im}
\left[
\mu^\dagger
(E+i\eta-H)^{-1}
\mu
\right].
]

So the Green’s-function/Schur method may be both faster and more directly connected to the measurement.

### 6. Tip molecule relaxation can dominate contrast

Even if the electronic model is elegant, the image may mostly reflect changes in tip-molecule pose:

[
\mathbf R_0(\mathbf x,\mathbf y,z),
\qquad
\mathbf T_0(\mathbf x,\mathbf y,z).
]

For PTCDA hanging on a metal apex, small rotations can change the transition dipole orientation strongly. Therefore, the force-field relaxation should output not only position but also orientation, and the optical transition charges/dipoles must rotate with the molecule.

---

## 11. Recommended model hierarchy

I would implement the project in layers.

### Level 1: Minimal Frenkel-dipole model

One transition per molecule.

[
H_{ij}
======

E_i\delta_{ij}
+
J^{\rm dip}_{ij}.
]

Tip molecule moves rigidly. Surface block static. Use Schur complement.

This will test the whole pipeline.

### Level 2: Transition-charge Frenkel model

Replace dipole coupling by

[
J_{ij}
======

k_e
\sum_{\alpha\beta}
\frac{q_{i\alpha}^{\rm tr}q_{j\beta}^{\rm tr}}
{r_{\alpha\beta}}.
]

Still one state per molecule.

This should already be realistic for PTCDA aggregates.

### Level 3: Multiple local excitations per molecule

Use basis (|ia\rangle), (a=1...m). Include dark states and different polarization directions.

This allows the tip to mix bright/dark excitons through anisotropic coupling and Stark shifts.

### Level 4: Stark shifts from difference charges

Use

[
\Delta E_{ia}^{\rm Stark}
=========================

\sum_\alpha
\Delta q_{a\alpha}
\phi(\mathbf r_{i\alpha}).
]

This is better near metal tips than a uniform-field approximation.

### Level 5: Screened Coulomb interaction

Add image charges or a precomputed screened Green’s function.

[
J =
\sum_{\alpha\beta}
q_\alpha^{\rm tr}
W(\mathbf r_\alpha,\mathbf r_\beta)
q_\beta^{\rm tr}.
]

### Level 6: Rate-equation PL

After obtaining exciton states, solve population dynamics with pumping, relaxation, radiative decay, and quenching.

---

## 12. Core equations to implement first

I would start with this compact model:

[
H =
\begin{pmatrix}
E_T(\mathbf R) & \mathbf v(\mathbf R)^T \
\mathbf v(\mathbf R) & H_S
\end{pmatrix},
]

where

[
v_i(\mathbf R)
==============

k_e
\sum_{\alpha\in T}
\sum_{\beta\in i}
\frac{
q_\alpha^{\rm tr}
q_\beta^{\rm tr}
}{
|\mathbf r_{T\alpha}(\mathbf R)-\mathbf r_{i\beta}|
}.
]

Prediagonalize

[
H_S U = U\varepsilon.
]

Then

[
\tilde v_\nu(\mathbf R)
=======================

\sum_i v_i(\mathbf R)U_{i\nu}.
]

The tip spectral function is

[
A_T(E,\mathbf R)
================

-\frac{1}{\pi}
\operatorname{Im}
\frac{1}{
E+i\eta
-------

## E_T(\mathbf R)

\sum_\nu
\frac{
|\tilde v_\nu(\mathbf R)|^2
}{
E+i\eta-\varepsilon_\nu
}
}.
]

This one equation is probably the cleanest first numerical target.

It contains the essential physics:

* moving tip molecule;
* coupling to aggregate excitons;
* avoided crossings;
* spectral shifts;
* broadening through coupling to dense surface exciton manifold;
* spatial imaging contrast.

Then the predicted TEPL image can be

[
I(\mathbf R)
============

\int dE,
A_T(E,\mathbf R)
W_{\rm det}(E),
]

or at a fixed detection energy,

[
I(\mathbf R)
============

A_T(E_{\rm det},\mathbf R).
]

If you include excitation at laser energy,

[
I(\mathbf R)
\sim
A_T(E_{\rm laser},\mathbf R)
A_T(E_{\rm det},\mathbf R),
]

although this is only a crude phenomenological model.

---

## 13. My main recommendation

Do **not** start with full TDDFT-quality realism. Start with the minimal Schur-complement transition-charge model. It is simple, fast, and physically interpretable.

The most useful first implementation should output several maps:

[
E_{\rm tip}^{\rm eff}(\mathbf x,\mathbf y),
]

[
A_T(E_{\rm laser},\mathbf x,\mathbf y),
]

[
A_T(E_{\rm det},\mathbf x,\mathbf y),
]

[
\sum_\nu |\tilde v_\nu(\mathbf x,\mathbf y)|^2,
]

and maybe the dominant coupled surface exciton index

[
\nu_{\rm max}(\mathbf x,\mathbf y)
==================================

\arg\max_\nu
\frac{
|\tilde v_\nu|^2
}{
(E_T-\varepsilon_\nu)^2+\eta^2
}.
]

This will tell you whether the contrast comes from geometry, resonance, or coupling to specific aggregate exciton modes.

The most dangerous approximation is probably **screening by the metal tip/substrate**, not the Frenkel model itself. The second most dangerous is ignoring **charge-transfer states** when the tip molecule comes very close to the surface aggregate.

---

# USER

OK, now maybe you can suggest how to implement it on gpu in OpenCL kernels where we want to use efficiently workgroups. We want to preload usefull qunatiitis to local memory. Not sure if we should useone thread ot one worgroup for one pixel? (maybe for the diagonalization will be better to use whole workgroup, but for computing of dipole-dipole and shour component just one thread?) The size of the problem will be perhaps small, max 4 states per molecules, max 15 molecules per cluster, so max 64 states in total. 

For the moment forget about the screening, it complicate things, or introduce it only as some efective liferime and energy shift as scalar without strucutrre, we can think about porper screenin later. How many kernsl should we have? perhaps just one? (compute the matrix elements for tip_molecule vs surface_cluster ( max 64 matrix elements ), and diagonalize it (by shur component or full diagonalization by Jacobi rotation - do not program full jocobi rotation we have it somewhere already implemented). For stark shifts we need also compute max 64 diagonal shifts. The kernels should be parameterized by workgroup size which is size of the hamiltonian. We should perhaps make two kernesl, one using schur component without stark shifts in substrate, the other does full diagonalization with stark shifts in substrate.

What is also super importaint is how to compute the projection of transition desnity to charges. This should not be atomic charges. Ideally these should be charges placed into each + or - lobes of the transition density. question is how to find them? That is somethin about topology of 3D function described on grid. Are there fast algorithms for that? I think there are. My naive idea is to use laplacian resp laplacian of gaussian (density on grid may be not smooth we can smoothen it). Most molecule we want to deal with are planar aromatic system, there one can simplify the problem to 2D as molecule is anty symmetric along z-axis (mirror image with sign flip) but transtion density is ofren symmetric because it is product of two antysymetric functions. But even in 2D it is question how to do this search for extrema of function in robust and fast way. Sure we can always proejct the density on atomic charges, but I feel like it is not optimal. What we can also do is some hybrid approach - use molecular dipoles as 1st order fast representation, but augument it by fwe charges at the periphery of the molecule just as short range correction (this is usefull in the assembly where the molecules are in close contact). We can tst quality of our model by basically computing hatry potential form that density and convolving it with the other density - basically just FFT product in foruer space E = rho1(k)*rho2(k)/|k^2|which is very fast. This can be our accurate ful gird reference versus which we can compare our apporc model using point charges. Or instead of point charges we can use erf(r)/r (Boys function) which converge to point charge interaction very fast.

---

# Chat GPT 5.5 medium

For your size scale — at most roughly

[
N_{\rm mol}\le 16,\qquad m\le 4,\qquad n=(N_{\rm mol})m\le 64
]

— I would design the GPU code around:

[
\boxed{\text{one workgroup = one image pixel / one tip pose}}
]

not one thread per pixel.

One thread per pixel is tempting for the Schur formula, but it becomes bad as soon as you compute realistic transition-charge couplings, Stark shifts, or full diagonalization. The work per pixel is small but structured, so a whole workgroup lets you use local memory and parallel reductions efficiently.

The workgroup does several stages:

[
\text{tip pose}
\rightarrow
\text{tip-surface couplings}
\rightarrow
\text{Stark shifts}
\rightarrow
\text{Hamiltonian or self-energy}
\rightarrow
\text{spectrum / eigenvalues / image signal}.
]

The important point is that not all stages need the full workgroup equally. Some are embarrassingly parallel; others are tiny dense-matrix operations.

---

# 1. Data layout

Let the total number of local exciton states be

[
n = n_{\rm surf} + n_{\rm tip}
]

where

[
n_{\rm surf}=N_{\rm surf}m,\qquad n_{\rm tip}=m.
]

For the full model with max 15 surface molecules and 1 tip molecule:

[
n_{\rm surf}=15\times 4=60,\qquad n_{\rm tip}=4,\qquad n=64.
]

For each molecule/state pair ((i,a)), store:

[
E^0_{ia}
]

transition dipole

[
\boldsymbol\mu^{\rm tr}_{ia}
]

difference dipole or difference charges for Stark shifts

[
\Delta\boldsymbol\mu_{ia}
\quad \text{or} \quad
\Delta q_{ia,\alpha}
]

and transition-charge/Gaussian sites

[
\left{
\mathbf r^0_{a\alpha}, q^{\rm tr}*{a\alpha}, \sigma*{a\alpha}
\right}.
]

For PTCDA, you may have maybe 10–40 transition-density sites per transition if you want a compact lobe model. Let

[
n_q \le 32
]

per transition as a practical target.

For a given rigid molecule pose,

[
\mathbf r_{i a \alpha}
======================

\mathbf R_i \mathbf r^0_{a\alpha}+\mathbf T_i.
]

For surface molecules, these transformed positions can be precomputed once.

For the tip molecule, they are recomputed per pixel.

---

# 2. Coupling formula for GPU

The raw point-charge transition-density coupling is

[
J_{ia,jb}
=========

k_e
\sum_{\alpha=1}^{n_q}
\sum_{\beta=1}^{n_q}
\frac{
q_{a\alpha}q_{b\beta}
}{
|\mathbf r_{ia\alpha}-\mathbf r_{jb\beta}|
}.
]

But I would strongly prefer Gaussian-smeared charges:

[
\rho_\alpha(\mathbf r)
======================

q_\alpha
\frac{
e^{-|\mathbf r-\mathbf R_\alpha|^2/(2\sigma_\alpha^2)}
}{
(2\pi\sigma_\alpha^2)^{3/2}
}.
]

Then the Coulomb interaction between two Gaussian charges is

[
V_{\alpha\beta}(r)
==================

k_e q_\alpha q_\beta
\frac{
\operatorname{erf}
\left(
\frac{r}{\sqrt{2(\sigma_\alpha^2+\sigma_\beta^2)}}
\right)
}{r}.
]

So

[
J_{ia,jb}
=========

k_e
\sum_{\alpha\beta}
q_{a\alpha}q_{b\beta}
\frac{
\operatorname{erf}
\left(
\frac{r_{\alpha\beta}}
{\sqrt{2(\sigma_{a\alpha}^2+\sigma_{b\beta}^2)}}
\right)
}{
r_{\alpha\beta}
}.
]

This is much better numerically than point charges because the interaction does not explode when two fitted sites get close.

At large (r),

[
\frac{\operatorname{erf}(r/s)}{r}
\rightarrow
\frac{1}{r},
]

so it becomes a point-charge interaction automatically.

At small (r),

[
\frac{\operatorname{erf}(r/s)}{r}
\rightarrow
\frac{2}{\sqrt{\pi}s}.
]

So it is finite.

For GPU, `erf()` may be expensive. You can use three levels:

### Level A: point charges

Fastest, but dangerous at close contact.

[
V(r)=\frac{1}{r}.
]

### Level B: softened Coulomb

Approximate Gaussian smearing by

[
V(r)=\frac{1}{\sqrt{r^2+s^2}}.
]

Not exact, but fast and stable.

### Level C: true Gaussian Coulomb

[
V(r)=\frac{\operatorname{erf}(r/s)}{r}.
]

Best reference-quality compact representation.

For production imaging I would probably use softened Coulomb first:

[
J_{ia,jb}
=========

k_e
\sum_{\alpha\beta}
\frac{
q_{a\alpha}q_{b\beta}
}{
\sqrt{
|\mathbf r_{ia\alpha}-\mathbf r_{jb\beta}|^2
+s_{\alpha\beta}^2
}
}.
]

This is fast and robust.

---

# 3. Two main kernels

I agree with your instinct: there should be two main kernels.

## Kernel 1: Schur/Green kernel

Use when:

[
H_S
]

is static, prediagonalized, and substrate Stark shifts are ignored or constant.

This kernel computes:

[
\tilde V_{a\nu}(\mathbf R)
==========================

\sum_{Ib}
J_{0a,Ib}(\mathbf R)U_{Ib,\nu}.
]

Then either diagonalizes the reduced matrix

[
\begin{pmatrix}
H_T & \tilde V\
\tilde V^\dagger & \varepsilon
\end{pmatrix},
]

or more cheaply computes the tip Green’s function:

[
G_T(E)
======

\left[
E+i\eta
-------

## H_T

\Sigma(E)
\right]^{-1}
]

with

[
\Sigma_{ab}(E)
==============

\sum_\nu
\frac{
\tilde V_{a\nu}\tilde V_{b\nu}^*
}{
E+i\eta-\varepsilon_\nu
}.
]

This kernel is extremely suitable for one workgroup per pixel.

## Kernel 2: full-Hamiltonian kernel

Use when:

* surface Stark shifts change with tip position;
* local fields from the tip shift surface states;
* you want full eigenvectors;
* Schur approximation is not enough;
* you include additional geometry-dependent interactions inside the surface cluster.

This kernel builds the full dense matrix:

[
H_{pq},\qquad p,q=1...n,\quad n\le 64
]

in local memory and calls your Jacobi diagonalizer.

This is heavier, but still realistic for (n\le 64).

---

# 4. Workgroup size

For (n\le 64), I would use:

[
\boxed{N_{\rm WG}=64}
]

or

[
\boxed{N_{\rm WG}=128}
]

depending on the GPU.

The clean design is:

```c
#define MAX_STATES 64
#define WG_SIZE    64
```

Then each local thread id corresponds naturally to one Hamiltonian row/state:

[
\text{lid} \leftrightarrow p.
]

For matrix construction, each thread can compute one row or several matrix elements.

For diagonalization, your Jacobi routine probably benefits from 64 threads.

For Schur Green’s function, 64 threads may be more than needed, but this is fine because each pixel still has enough arithmetic:

[
m \times n_{\rm surf}
]

couplings, and each coupling involves (n_q^2) charge-pair interactions.

If

[
m=4,\quad n_{\rm surf}=60,\quad n_q=16,
]

then tip-surface coupling cost per pixel is roughly

[
4 \times 60 \times 16^2
=======================

61440
]

pair interactions.

That is enough work to justify a workgroup.

---

# 5. Local memory strategy

The useful quantities to load into local memory per workgroup are:

### Tip transformed charge sites

For each tip state (a) and site (\alpha):

[
\mathbf r_{0a\alpha}, q_{a\alpha}, \sigma_{a\alpha}.
]

This is small.

### Surface charge sites for currently processed surface molecule

Do not necessarily preload all surface transition charges for all molecules/states at once. That may be too much local memory.

Suppose:

[
N_{\rm surf}=15,\quad m=4,\quad n_q=32.
]

Then total surface charge sites are

[
15\times 4\times 32=1920.
]

Each site needs maybe:

```c
float4 pos_q;   // x,y,z,q
float  sigma;
```

That is roughly 20 bytes/site, or 38 kB. This is borderline but possible on some GPUs, bad on others.

Better approach:

* keep static surface sites in global/constant memory;
* process one surface molecule or one surface state block at a time;
* load a tile of surface charge sites into local memory;
* compute all tip-state couplings to that tile.

Since the surface cluster is small, global memory traffic is not terrible, especially if many pixels reuse the same surface cluster.

If many pixels scan the same cluster, surface data should be in constant memory or read-only cache.

---

# 6. Schur kernel pseudocode

Assume:

[
m\le 4,\quad n_{\rm surf}\le 60,\quad n_{\nu}\le 60.
]

Precomputed globally:

[
\varepsilon_\nu,
\qquad
U_{Ib,\nu}.
]

Per pixel:

```text
kernel_schur(pixel_id):

    lid = local_id
    group = pixel_id

    load tip pose for this pixel:
        T_tip[pixel_id], R_tip[pixel_id]

    transform tip transition-charge sites:
        for k assigned to lid:
            r_tip[a,alpha] = R_tip * r0[a,alpha] + T_tip

    barrier

    compute tip Stark block H_T:
        for a,b maybe assigned to lid:
            H_T[a,b] = ...
        usually diagonal:
            H_T[a,a] = E0_tip[a] + dE_stark_tip[a]

    barrier

    initialize VtipSurf[a, s] where s=(I,b)
        dimensions m x n_surf

    compute J between tip states and surface states:
        total elements = m * n_surf
        each lid handles several elements

        for elem = lid; elem < m*n_surf; elem += WG_SIZE:
            a = elem / n_surf
            s = elem % n_surf
            I,b = decode(s)
            J = transition_charge_coupling(tip state a, surface state I,b)
            V[a,s] = J

    barrier

    transform to surface-exciton basis:
        Vtilde[a,nu] = sum_s V[a,s] * U[s,nu]

        parallelize over a,nu:
            each thread handles one (a,nu)
            reduction over s can be serial because n_surf <= 60
            or parallel reduction if needed

    barrier

    Option Green:
        for each energy E_k:
            compute Sigma[a,b](E_k)
                Sigma[a,b] = sum_nu Vtilde[a,nu]*Vtilde[b,nu]/(E+i eta - eps[nu])
            compute G = inverse(E+i eta - H_T - Sigma)
            compute signal:
                I(E_k) = -Im( mu^dagger G mu ) / pi

    Option reduced diagonalization:
        assemble Hred:
            size = m + n_active
        diagonalize Hred
        compute optical observables
```

For (m=1), the Green path is trivial:

```text
Sigma = sum_nu |Vtilde[nu]|^2 / (E+i eta - eps[nu])
G     = 1 / (E+i eta - E_tip - Sigma)
I     = -imag(G)/pi
```

This is probably the first production-quality version I would build.

---

# 7. Full Hamiltonian kernel pseudocode

This is for variable surface Stark shifts.

```text
kernel_full(pixel_id):

    lid = local_id

    load / compute tip pose

    transform tip transition-charge sites

    compute Stark shifts for all states:
        for p = lid; p < n; p += WG_SIZE:
            if p belongs to tip:
                dE[p] = StarkFromTipPoseOrField(...)
            else:
                dE[p] = SurfaceStarkFromTipField(...)

    barrier

    build full Hamiltonian H[n,n] in local memory:
        for idx = lid; idx < n*n; idx += WG_SIZE:
            p = idx / n
            q = idx % n

            if p == q:
                H[p,q] = E0[p] + dE[p]

            else if both p,q are surface:
                if surface couplings static:
                    H[p,q] = Hsurf_static[p,q]
                else:
                    H[p,q] = compute_coupling(p,q)

            else if one is tip and one is surface:
                H[p,q] = transition_charge_coupling(p,q)

            else if both are tip:
                H[p,q] = Htip_local[p,q]

    barrier

    diagonalize H using local-memory Jacobi

    compute observables:
        oscillator strengths, PL proxy, selected eigenvalues

    write output[pixel_id]
```

Memory requirement:

[
64\times64=4096
]

floats.

That is

[
4096\times4=16,{\rm kB}
]

for real single precision.

If complex Hermitian, maybe 32 kB.

This is still acceptable per workgroup on many GPUs, but occupancy may be reduced. Since this is not enormous, full diagonalization is feasible.

For real Frenkel Hamiltonian without magnetic/chiral effects, use real symmetric matrices. Transition densities and Coulomb couplings are real. That halves memory and simplifies Jacobi.

---

# 8. How many kernels?

I would use three kernels in practice, although two are enough conceptually.

## Kernel A: geometry / tip relaxation

This computes relaxed tip pose per pixel:

[
\mathbf R_0(x,y,z),\mathbf T_0(x,y,z).
]

This may be iterative and force-field heavy. It should probably be separate from exciton calculation.

Reason: if tip relaxation requires many steps, mixing it with diagonalization makes the kernel long, branchy, and hard to debug.

Output:

```c
float4 tip_pos[pixel];
float4 tip_quat[pixel];
```

## Kernel B: Schur spectral kernel

Fast excitonic image when surface block is static.

Output:

```c
I[pixel]
```

or

```c
I[pixel, energy]
```

or selected features:

```c
E_peak[pixel], I_peak[pixel], Gamma_eff[pixel]
```

## Kernel C: full Hamiltonian kernel

Slower but more general. Use for validation and for cases with surface Stark shifts.

So I would not force everything into one kernel. The clean pipeline is:

```text
kernel_relax_tip_pose
kernel_schur_TEPL
kernel_full_TEPL       // optional / validation / strong Stark mode
```

For early development you can skip kernel A and supply rigid translated/rotated poses from CPU or a simple analytic mapping.

---

# 9. Schur vs full diagonalization

For your problem size, full diagonalization is feasible, but Schur is physically and computationally cleaner.

The Schur method becomes especially nice if you want spectra over many energies.

Full diagonalization gives:

[
E_\lambda,\quad C^\lambda
]

then you broaden:

[
I(E)=\sum_\lambda f_\lambda L(E-E_\lambda,\gamma).
]

Schur Green’s function directly gives:

[
I(E)
====

-\frac{1}{\pi}
\operatorname{Im}
\mu_T^\dagger
\left[
E+i\eta-H_T-\Sigma(E)
\right]^{-1}
\mu_T.
]

If the experiment mainly excites/detects the tip-localized molecule, Schur is ideal.

If you need full exciton eigenvectors, full diagonalization is better.

A good strategy:

* use Schur for production image generation;
* use full diagonalization for debugging selected pixels;
* compare both for cases without surface Stark shifts.

---

# 10. Efficient computation of (V_{a,s})

The expensive part is

[
V_{a,s}
=======

k_e
\sum_{\alpha\beta}
q_{a\alpha}q_{s\beta}
K(r_{\alpha\beta}).
]

where

[
K(r)=\frac{1}{r}
]

or

[
K(r)=\frac{1}{\sqrt{r^2+s^2}}.
]

A good workgroup mapping:

```text
one workgroup = one pixel
one thread = one coupling element or one partial charge-pair sum
```

There are two possible granularities.

## Mapping 1: one thread computes one state-state coupling

Each thread computes one (V_{a,s}) by looping over all charge pairs.

For max:

[
m n_{\rm surf}=4\times 60=240
]

coupling elements.

With 64 threads, each thread computes about 4 elements.

Each element costs:

[
n_q^2
]

charge-pair interactions.

If (n_q=16), that is 256 inner operations. Good.

This is simple and probably fast enough.

## Mapping 2: many threads cooperate on one coupling

Useful if (n_q) is large, e.g. 64 or 128.

Then one workgroup computes one or several (V_{a,s}) values using parallel reduction over charge pairs.

But for your small model, this may be overengineering.

I would start with Mapping 1.

---

# 11. Stark shifts on GPU

For the tip molecule:

[
\Delta E_{0a}
=============

\sum_\alpha
\Delta q_{a\alpha}
\phi(\mathbf r_{0a\alpha}).
]

If (\phi) is from a grid:

[
\phi(\mathbf r)
===============

\text{trilinear/tricubic interpolation}
]

from a precomputed electrostatic potential.

If the field/potential is analytic, e.g. point apex or simple capacitor field, compute directly.

For the surface molecules, if you include tip-induced Stark shift:

[
\Delta E_{ia}^{\rm tip}
=======================

\sum_\alpha
\Delta q_{a\alpha}
\phi_{\rm tip}(\mathbf r_{ia\alpha};\mathbf R_{\rm tip}).
]

This is diagonal and cheap.

But the full surface block then becomes pixel-dependent:

[
H_S(\text{pixel})=H_S^0+D(\text{pixel}).
]

Once you add this, the prediagonalized (H_S) no longer applies exactly. You have three options.

### Option 1: ignore surface Stark

Use Schur exactly.

### Option 2: include surface Stark perturbatively

Use precomputed surface eigenstates and compute

[
\Delta\varepsilon_\nu
=====================

\sum_s |U_{s\nu}|^2 \Delta E_s.
]

Then use

[
\varepsilon_\nu(\text{pixel})
=============================

\varepsilon_\nu^0+\Delta\varepsilon_\nu.
]

This keeps the Schur kernel and includes first-order diagonal energy shifts.

You may also compute off-diagonal Stark mixing:

[
D_{\nu\mu}
==========

\sum_s
U_{s\nu}\Delta E_s U_{s\mu}.
]

But then the surface block is no longer diagonal. If the perturbation is small, keep only the diagonal part.

### Option 3: full diagonalization

Build full (H) and diagonalize.

My recommendation:

[
\boxed{\text{Use first-order shifted surface exciton energies before going full diagonalization.}}
]

This gives a middle model:

[
\Sigma_{ab}(E)
==============

\sum_\nu
\frac{
\tilde V_{a\nu}\tilde V^**{b\nu}
}{
E+i\eta-\varepsilon*\nu-\Delta\varepsilon_\nu
}.
]

This is cheap and probably captures most Stark physics.

---

# 12. Transition-density reduction to compact charges

This is the other central problem. You are right: atomic transition charges are convenient but not necessarily optimal.

The goal is to approximate the Coulomb potential generated by the transition density:

[
\phi^{\rm tr}(\mathbf r)
========================

k_e
\int
\frac{\rho^{\rm tr}(\mathbf r')}{|\mathbf r-\mathbf r'|}
d\mathbf r'
]

by compact Gaussian charges:

[
\phi^{\rm fit}(\mathbf r)
=========================

k_e
\sum_p
q_p
\frac{
\operatorname{erf}
\left(
|\mathbf r-\mathbf R_p|/(\sqrt{2}\sigma_p)
\right)
}{
|\mathbf r-\mathbf R_p|}
.
]

Then fit

[
\rho^{\rm tr}(\mathbf r)
\approx
\sum_p q_p g_\sigma(\mathbf r-\mathbf R_p)
]

or, better, fit the potential.

The physically meaningful objective is not density RMS, but Coulomb-metric error:

[
\chi^2
======

\iint
\left[
\rho(\mathbf r)-\rho_{\rm fit}(\mathbf r)
\right]
\frac{1}{|\mathbf r-\mathbf r'|}
\left[
\rho(\mathbf r')-\rho_{\rm fit}(\mathbf r')
\right]
d\mathbf r d\mathbf r'.
]

In Fourier space:

[
\chi^2
======

4\pi
\int
\frac{
|\rho(\mathbf k)-\rho_{\rm fit}(\mathbf k)|^2
}{
k^2
}
d\mathbf k.
]

This is exactly what matters for exciton couplings.

So yes: your FFT reference idea is very good.

The exact reference coupling between two transition densities is

[
J_{12}
======

4\pi k_e
\int
\frac{
\rho_1(\mathbf k)^*
\rho_2(\mathbf k)
}{
k^2
}
\frac{d\mathbf k}{(2\pi)^3}.
]

On a grid, this becomes:

[
J_{12}
\approx
\sum_{\mathbf k\ne 0}
\frac{
4\pi k_e
}{
k^2
}
\rho_1(\mathbf k)^*
\rho_2(\mathbf k)
\Delta k^3.
]

The (k=0) term should vanish for transition densities because

[
\int\rho^{\rm tr}(\mathbf r)d\mathbf r=0.
]

This FFT Coulomb metric can be used to validate any point-charge or Gaussian-charge representation.

---

# 13. How to find lobes of transition density

You want charges placed in positive and negative lobes, not necessarily on atoms.

Let the transition density on a molecular grid be

[
\rho(\mathbf r).
]

A robust algorithm is:

1. smooth the density;
2. split positive and negative regions;
3. find connected components;
4. for each component, integrate charge and centroid;
5. optionally split large components into multiple Gaussian sites;
6. fit charges/widths/positions by Coulomb-metric optimization.

This is more robust than simply finding local extrema.

---

## 13.1 Smoothing

Use Gaussian smoothing:

[
\rho_\sigma = G_\sigma * \rho.
]

This suppresses grid noise and tiny oscillatory artifacts.

If the density is on a grid, smoothing can be done by FFT:

[
\rho_\sigma(\mathbf k)
======================

\rho(\mathbf k)e^{-\sigma^2 k^2/2}.
]

or by separable real-space convolution.

The smoothing length should be comparable to the grid spacing or slightly larger:

[
\sigma_{\rm smooth}\sim 0.3-0.7\ {\rm Å}.
]

Too much smoothing destroys nodal structure.

---

## 13.2 Connected components of sign regions

Define positive and negative masks:

[
M_+(\mathbf r)=
\left[
\rho_\sigma(\mathbf r)>\rho_{\rm cut}
\right],
]

[
M_-(\mathbf r)=
\left[
\rho_\sigma(\mathbf r)<-\rho_{\rm cut}
\right].
]

Use a threshold based on max density or integrated charge:

[
\rho_{\rm cut}
==============

\epsilon \max|\rho_\sigma|,
]

with (\epsilon\sim 0.02-0.05).

Then find connected components in 3D or 2D.

For each component (C_p), compute:

[
q_p
===

\int_{C_p}
\rho(\mathbf r)d\mathbf r,
]

[
\mathbf R_p
===========

\frac{1}{q_p}
\int_{C_p}
\mathbf r,\rho(\mathbf r)d\mathbf r.
]

For the width:

[
\sigma_p^2
==========

\frac{1}{3q_p}
\int_{C_p}
|\mathbf r-\mathbf R_p|^2\rho(\mathbf r)d\mathbf r.
]

For negative lobes, use signed (q_p), or use (|\rho|) for centroid/width and assign sign separately:

[
\mathbf R_p
===========

\frac{
\int_{C_p}
\mathbf r,|\rho(\mathbf r)|d\mathbf r
}{
\int_{C_p}
|\rho(\mathbf r)|d\mathbf r
}.
]

This is usually numerically safer.

This already gives a compact lobe-charge representation.

---

## 13.3 Local extrema / watershed refinement

Connected components may merge several lobes connected by a weak bridge. Then use watershed segmentation.

For positive density, find maxima of (\rho_\sigma). For negative density, find maxima of (-\rho_\sigma).

A local maximum satisfies:

[
\rho(\mathbf r_i)>\rho(\mathbf r_j)
]

for all neighboring grid points (j).

Then use watershed or steepest-ascent assignment:

```text
for every grid point above threshold:
    follow gradient ascent until reaching a local maximum
    assign grid point to that maximum basin
```

For negative lobes, do the same on (-\rho).

Then each basin gives one candidate charge site.

This is probably the “topological” algorithm you were thinking about. It is robust, fast, and simple.

The Laplacian-of-Gaussian idea is also related, but I would use it mainly for scale selection, not as the primary segmentation method.

---

## 13.4 Laplacian of Gaussian

The LoG operator is

[
\nabla^2(G_\sigma * \rho).
]

It detects blob-like structures at scale (\sigma). For each scale (\sigma), extrema of

[
\sigma^2 \nabla^2 G_\sigma * \rho
]

indicate features of size (\sigma).

This is useful if you want automatic multiscale placement of Gaussian sites.

But for transition densities, sign matters. The simpler signed watershed method may be easier and more physically transparent.

---

# 14. Planar aromatic simplification

For planar π systems, the transition density for a π–π* excitation is often approximately symmetric with respect to the molecular plane because it is a product of two π orbitals:

[
\rho^{\rm tr}(\mathbf r)
\sim
\psi_{\rm occ}(\mathbf r)\psi_{\rm virt}(\mathbf r).
]

Each π orbital is antisymmetric under reflection through the plane:

[
\psi(x,y,-z)=-\psi(x,y,z).
]

Their product is symmetric:

[
\psi_{\rm occ}(x,y,-z)\psi_{\rm virt}(x,y,-z)
=============================================

\psi_{\rm occ}(x,y,z)\psi_{\rm virt}(x,y,z).
]

So the transition density is mostly even in (z). Therefore you can project to 2D:

[
\rho_{2D}(x,y)
==============

\int \rho(x,y,z),dz.
]

Then find lobes in 2D. This is much easier and likely sufficient for flat PTCDA-like molecules.

A 2D lobe charge model gives sites:

[
(x_p,y_p,z_0),\quad q_p,\quad \sigma_p.
]

The width (\sigma_p) can include an effective out-of-plane spread.

This may be the best practical path:

[
\boxed{
\text{TDDFT transition density grid}
\rightarrow
\text{project to molecular plane}
\rightarrow
\text{signed watershed lobes}
\rightarrow
\text{Gaussian charges}
\rightarrow
\text{Coulomb-metric fit}
}
]

---

# 15. Fitting compact Gaussian charges

After initial lobe detection, optimize the charges (q_p), maybe positions (\mathbf R_p), and widths (\sigma_p).

Keep constraints:

### Neutral transition density

[
\sum_p q_p=0.
]

### Correct transition dipole

[
\sum_p q_p \mathbf R_p
======================

\boldsymbol\mu^{\rm tr}.
]

Optionally quadrupole:

[
\sum_p q_p
\left(
3R_{p\alpha}R_{p\beta}
----------------------

R_p^2\delta_{\alpha\beta}
\right)
=======

Q^{\rm tr}_{\alpha\beta}.
]

The fitting objective can be the potential error sampled outside the molecule:

[
\chi^2_\phi
===========

\sum_g
w_g
\left[
\phi_{\rm ref}(\mathbf r_g)
---------------------------

\phi_{\rm fit}(\mathbf r_g)
\right]^2.
]

or the Fourier Coulomb metric:

[
\chi^2_C
========

\sum_{\mathbf k\ne 0}
\frac{4\pi}{k^2}
\left|
\rho_{\rm ref}(\mathbf k)
-------------------------

\rho_{\rm fit}(\mathbf k)
\right|^2.
]

The Coulomb metric is more physically correct, but potential fitting on a shell outside the molecule is very intuitive and easier to debug visually.

---

# 16. Hybrid dipole plus short-range correction

Your hybrid idea is very good.

Use the exact transition dipole as the long-range representation:

[
\boldsymbol\mu
==============

\int \mathbf r\rho^{\rm tr}(\mathbf r)d\mathbf r.
]

Then add local correction charges that have zero net charge and zero dipole:

[
\sum_p q_p^{\rm corr}=0,
]

[
\sum_p q_p^{\rm corr}\mathbf R_p=0.
]

Then the full fitted density is

[
\rho_{\rm fit}
==============

\rho_{\rm dipole}
+
\rho_{\rm corr}.
]

The correction affects only short and intermediate distances. This is attractive because it guarantees correct far-field behavior.

Numerically, however, representing a pure dipole as two nearby charges can be unstable. Instead, you can compute dipole-dipole analytically and add short-range correction charges:

[
J_{ij}
======

J^{\rm dip}*{ij}
+
J^{\rm corr}*{ij}.
]

where (J^{\rm corr}) is constructed to vanish rapidly at long range.

One way:

[
J^{\rm corr}_{ij}
=================

## J^{\rm charges}_{ij}

J^{\rm dip}_{ij}
]

but only for (R<R_{\rm switch}), with smooth damping:

[
J_{ij}
======

J^{\rm dip}*{ij}
+
s(R)
\left[
J^{\rm charges}*{ij}
--------------------

J^{\rm dip}_{ij}
\right].
]

For large (R), (s(R)\to 0), so you recover dipole coupling.

For short (R), (s(R)\to 1), so you use charge-density coupling.

This is excellent for GPU because you can avoid charge-charge sums for distant pairs.

But in your small cluster, there are only 15 molecules, so you may not even need the dipole shortcut. Still, for generalization it is good.

---

# 17. Accurate grid reference with FFT

Your reference idea is exactly right.

Given transition densities (\rho_A) and (\rho_B) on the same grid, shifted/rotated relative to each other, the Coulomb energy is

[
J
=

k_e
\int
\rho_A(\mathbf r)
\phi_B(\mathbf r)d\mathbf r.
]

where

[
\nabla^2\phi_B=-4\pi\rho_B.
]

In Fourier space:

[
\phi_B(\mathbf k)
=================

\frac{4\pi}{k^2}\rho_B(\mathbf k).
]

Thus

[
J
=

k_e
\sum_{\mathbf k\ne0}
\rho_A^*(\mathbf k)
\frac{4\pi}{k^2}
\rho_B(\mathbf k).
]

For benchmarking, you can compute this on CPU/Python first. For many relative translations, FFT convolution gives the coupling map:

[
J(\mathbf R)
============

\int
\rho_A(\mathbf r)
\phi_B(\mathbf r-\mathbf R)d\mathbf r.
]

This is a cross-correlation:

[
J(\mathbf R)
============

\mathcal F^{-1}
\left[
\rho_A^*(\mathbf k)
\phi_B(\mathbf k)
\right].
]

With

[
\phi_B(\mathbf k)=\frac{4\pi}{k^2}\rho_B(\mathbf k).
]

So

[
J(\mathbf R)
============

\mathcal F^{-1}
\left[
\rho_A^*(\mathbf k)
\frac{4\pi}{k^2}
\rho_B(\mathbf k)
\right].
]

This gives a very accurate reference for all translations on a grid. Rotations are harder, but for planar molecules you can benchmark selected rotations.

Use this to fit/validate your compact charge model.

---

# 18. Suggested offline fitting pipeline

```text
For each molecule and transition state a:

    1. Load transition density rho_tr(x,y,z) from TDDFT/TDA/DFTB/whatever.

    2. Ensure normalization:
        integral rho_tr dV should be approximately 0.

    3. Compute reference transition dipole:
        mu = integral r rho_tr(r) dV.

    4. If molecule is planar:
        project:
            rho2D(x,y) = integral rho3D(x,y,z) dz

    5. Smooth:
        rho_smooth = GaussianSmooth(rho2D or rho3D)

    6. Threshold:
        keep points where abs(rho_smooth) > eps * max(abs(rho_smooth))

    7. Segment positive and negative lobes:
        signed connected components or watershed.

    8. For each lobe:
        q_p = integral_lobe rho dV
        R_p = integral_lobe r |rho| dV / integral_lobe |rho| dV
        sigma_p = second moment of lobe

    9. Merge tiny lobes:
        if abs(q_p) < q_min, merge into nearest same-sign lobe or discard then renormalize.

    10. Enforce constraints:
        sum q_p = 0
        sum q_p R_p = mu

    11. Optional fit:
        optimize q_p, maybe sigma_p, maybe R_p
        minimize Coulomb-metric or potential error.

    12. Validate:
        compare J_fit(R,theta) vs J_grid_reference(R,theta)
        for representative dimer geometries.

    13. Export:
        number of sites
        q_p
        R_p in molecule-local frame
        sigma_p
```

This should be done offline, not on GPU during imaging.

---

# 19. Constraints and correction of fitted charges

Suppose you get initial charges (q_p^0). You want corrected charges (q_p) satisfying:

[
Aq=b
]

where constraints are:

[
\sum_p q_p=0,
]

[
\sum_p q_p x_p=\mu_x,
]

[
\sum_p q_p y_p=\mu_y,
]

[
\sum_p q_p z_p=\mu_z.
]

Then find the nearest charge vector to the initial one:

[
\min_q ||q-q^0||^2
\quad \text{subject to} \quad
Aq=b.
]

The solution is:

[
q
=

q^0
+
A^T(AA^T)^{-1}
(b-Aq^0).
]

This is a nice simple postprocessing correction.

If you want weighted correction:

[
\min_q
(q-q^0)^T W (q-q^0),
]

then

[
q
=

q^0
+
W^{-1}A^T
(AW^{-1}A^T)^{-1}
(b-Aq^0).
]

This lets you preserve large important lobes and adjust small ones more.

---

# 20. What to put into local memory

For the Schur kernel, local memory could be:

```c
local float4 tipSites[MAX_TIP_STATES * MAX_Q];      // xyz q
local float  tipSig [MAX_TIP_STATES * MAX_Q];

local float  V[MAX_TIP_STATES * MAX_SURF_STATES];  // m x nSurf
local float  Vt[MAX_TIP_STATES * MAX_ACTIVE];      // m x nActive

local float  HT[MAX_TIP_STATES * MAX_TIP_STATES];

local float2 Sigma[MAX_TIP_STATES * MAX_TIP_STATES]; // complex
```

For (m=4), (n_{\rm surf}=60), (n_{\rm active}=60):

[
V:4\times60=240 \text{ floats}
]

[
Vt:4\times60=240 \text{ floats}
]

tiny.

For the full kernel:

```c
local float H[64*64];
local float eigVec[64*64]; // only if needed
local float eigVal[64];
```

If you need eigenvectors, memory doubles. If you only need eigenvalues, Jacobi may still maintain eigenvectors depending on your implementation. For oscillator strengths you need eigenvectors.

---

# 21. Full diagonalization observables

After diagonalization,

[
H C_\lambda = E_\lambda C_\lambda.
]

Oscillator strength / transition dipole:

[
\boldsymbol\mu_\lambda
======================

\sum_p C_{p\lambda}\boldsymbol\mu_p.
]

Near-field excitation amplitude:

[
M_\lambda^{\rm exc}
===================

\sum_p C_{p\lambda}
\boldsymbol\mu_p\cdot\mathbf E_{\rm loc}(\mathbf R_p).
]

PL proxy:

[
I
=

\sum_\lambda
|M_\lambda^{\rm exc}|^2
|\boldsymbol\mu_\lambda|^2
L(E_{\rm laser}-E_\lambda,\gamma_{\rm abs})
L(E_{\rm det}-E_\lambda,\gamma_{\rm em})
Q_\lambda.
]

With simple quenching:

[
Q_\lambda
=========

\frac{\Gamma^{\rm rad}*\lambda}
{\Gamma^{\rm rad}*\lambda+\Gamma^{\rm nr}_\lambda}
]

and

[
\Gamma^{\rm nr}_\lambda
=======================

\sum_p |C_{p\lambda}|^2 \Gamma_p^{\rm nr}.
]

For now, screening can enter as scalar shifts/lifetimes:

[
E_p \rightarrow E_p+\Delta E_p^{\rm metal}(z_p),
]

[
\gamma_p \rightarrow \gamma_p+\gamma_p^{\rm metal}(z_p).
]

For an eigenstate:

[
\gamma_\lambda
==============

\sum_p |C_{p\lambda}|^2\gamma_p.
]

This gives you effective quenching without a full electromagnetic solver.

---

# 22. Numerical precision

Single precision should be fine for imaging and qualitative TEPL maps, but watch these problems:

### Near degeneracy

If two exciton states are nearly degenerate, single-precision eigenvectors can rotate arbitrarily inside the degenerate subspace. Energies remain okay, but state identity becomes unstable.

Use observables summed over energy windows, not raw eigenvector labels.

### Coupling singularity

Avoid bare point charges without softening. Use at least:

[
1/\sqrt{r^2+s^2}.
]

### Energy denominators in Schur formula

When

[
E\approx \varepsilon_\nu
]

the denominator is small. Always use finite broadening:

[
E+i\eta-\varepsilon_\nu.
]

Choose (\eta) physically, e.g. 10–100 meV depending on linewidth.

### Jacobi convergence

For (64\times64), fixed-sweep Jacobi is GPU-friendly. Do not use convergence-based early exits unless carefully synchronized. Fixed number of sweeps gives deterministic runtime.

---

# 23. Preferred first implementation

I would implement this sequence:

## First version

* one transition per molecule;
* transition Gaussian charges;
* static surface block;
* Schur scalar Green’s function;
* one workgroup per pixel;
* no full diagonalization.

Main equation:

[
A_T(E,\mathbf R)
================

-\frac{1}{\pi}
\operatorname{Im}
\frac{1}{
E+i\eta
-------

## E_T(\mathbf R)

\sum_\nu
\frac{
|\tilde v_\nu(\mathbf R)|^2
}{
E+i\eta-\varepsilon_\nu
}
}.
]

This is simple, fast, and interpretable.

## Second version

* (m\le4);
* (m\times m) Schur self-energy;
* tiny matrix inversion per energy.

[
G_T(E)
======

\left[
E+i\eta
-------

## H_T

\Sigma(E)
\right]^{-1}.
]

## Third version

* full (64\times64) local-memory Jacobi kernel;
* include surface Stark shifts;
* compute oscillator strengths from eigenvectors.

This is the validation/generalization kernel.

---

# 24. Practical OpenCL kernel architecture

A reasonable compile-time parameter set:

```c
#define WG_SIZE        64
#define MAX_M          4
#define MAX_SURF_MOL   15
#define MAX_SURF_STATE (MAX_M*MAX_SURF_MOL)  // 60
#define MAX_STATE      (MAX_SURF_STATE+MAX_M) // 64
#define MAX_Q          32
#define MAX_ACTIVE     60
#define COULOMB_CONST  14.3996448915f
```

Use fixed-size local arrays. Avoid dynamic indexing complexity where possible.

Main kernel modes can be compile-time:

```c
#define MODE_SCHUR 1
#define MODE_FULL  2
```

but separate kernels may produce cleaner code.

---

# 25. High-level OpenCL pseudocode: Schur kernel

```c
__kernel void TEPL_schur(
    __global const float4* tipPose,        // position + maybe quaternion elsewhere
    __global const float4* tipQuat,
    __global const float4* molSites0,      // local transition sites: xyz,q
    __global const float*  molSigma,
    __global const float4* surfSites,      // transformed surface sites
    __global const float*  surfSigma,
    __global const float*  U,              // surface eigenvectors [s,nu]
    __global const float*  eps,            // surface eigenvalues
    __global const float*  Egrid,
    __global float*        outSpectrum
){
    int pix = get_group_id(0);
    int lid = get_local_id(0);

    __local float4 tipSite[MAX_M*MAX_Q];
    __local float  tipSgm [MAX_M*MAX_Q];
    __local float  V[MAX_M*MAX_SURF_STATE];
    __local float  Vt[MAX_M*MAX_ACTIVE];

    // 1. Transform tip sites
    for(int k=lid; k<MAX_M*MAX_Q; k+=WG_SIZE){
        float4 r0q = molSites0[k];
        float3 r = rotate(tipQuat[pix], r0q.xyz) + tipPose[pix].xyz;
        tipSite[k] = (float4)(r.x,r.y,r.z,r0q.w);
        tipSgm[k]  = molSigma[k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2. Compute V[a,s]
    for(int elem=lid; elem<MAX_M*MAX_SURF_STATE; elem+=WG_SIZE){
        int a = elem / MAX_SURF_STATE;
        int s = elem - a*MAX_SURF_STATE;
        float J = coupling_state_state(a, s, tipSite, tipSgm, surfSites, surfSigma);
        V[elem] = J;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3. Transform V -> Vt
    for(int elem=lid; elem<MAX_M*MAX_ACTIVE; elem+=WG_SIZE){
        int a  = elem / MAX_ACTIVE;
        int nu = elem - a*MAX_ACTIVE;
        float sum = 0.0f;
        for(int s=0; s<MAX_SURF_STATE; s++){
            sum += V[a*MAX_SURF_STATE+s] * U[s*MAX_ACTIVE+nu];
        }
        Vt[elem] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4. Compute spectrum
    for(int ie=lid; ie<NE; ie+=WG_SIZE){
        float E = Egrid[ie];

        // m=1 scalar version shown:
        float2 Sigma = (float2)(0.0f,0.0f);
        for(int nu=0; nu<MAX_ACTIVE; nu++){
            float v = Vt[nu];
            float de = E - eps[nu];
            float denom = de*de + ETA*ETA;
            Sigma.x += v*v*de/denom;
            Sigma.y += -v*v*ETA/denom;
        }

        float2 z;
        z.x = E - E_tip - Sigma.x;
        z.y = ETA - Sigma.y;

        float denom = z.x*z.x + z.y*z.y;
        float2 G = (float2)( z.x/denom, -z.y/denom );

        float A = -G.y * INV_PI;
        outSpectrum[pix*NE+ie] = A;
    }
}
```

For (m>1), replace the scalar Green’s function by a (4\times4) complex matrix inversion.

---

# 26. Why one workgroup per pixel is better

Because a pixel is not just one scalar evaluation. It includes:

[
m n_{\rm surf}
]

state couplings, each with

[
n_q^2
]

charge-pair interactions.

For realistic (m=4), (n_{\rm surf}=60), (n_q=16):

[
4\times60\times256
\approx 6\times 10^4
]

pair operations per pixel.

That is too much for one thread and nicely sized for one workgroup.

For the pure dipole approximation, one thread per pixel could be enough, but that is the least realistic mode. I would not optimize the architecture around that.

---

# 27. Main caveat about transition-density lobes

The lobe decomposition must preserve the Coulomb potential, not the visual shape of the density.

A beautiful-looking lobe model can still give poor couplings if it does not reproduce low-order multipoles and near-field potential.

Therefore the validation metric should be:

[
\Delta J(R,\theta)
==================

## J_{\rm fit}(R,\theta)

J_{\rm grid}(R,\theta)
]

for physically relevant dimer configurations.

For a self-assembled PTCDA cluster, test:

* cofacial displacement;
* side-by-side contact;
* T-like contact if relevant;
* tip-above-surface geometries;
* rotated tip molecule;
* distances from close contact to ~20 Å.

Then plot relative error:

[
\epsilon_J
==========

\frac{|J_{\rm fit}-J_{\rm grid}|}
{|J_{\rm grid}|+J_{\rm floor}}.
]

Use a floor like (J_{\rm floor}=1) meV to avoid meaningless large relative errors near zeros.

---

# 28. My strongest recommendation

The best architecture is:

[
\boxed{
\text{offline TD density}
\rightarrow
\text{Gaussian lobe charges}
\rightarrow
\text{FFT Coulomb validation}
\rightarrow
\text{OpenCL one-workgroup-per-pixel Schur kernel}
\rightarrow
\text{optional full }64\times64\text{ Jacobi kernel}
}
]

Start with the scalar (m=1) Schur model. It will already let you debug the whole physics pipeline. Then extend to (m=4). Only after that add full diagonalization with surface Stark shifts.

The most important implementation choice is to avoid representing the transition density by atomic charges only. Use signed lobes, preferably Gaussian-smeared charges constrained to reproduce:

[
\sum q_p=0,
\qquad
\sum q_p\mathbf R_p=\boldsymbol\mu^{\rm tr},
]

and validated against the FFT Coulomb reference.

---

# USER

OK, now I'm thinking that best representation of coarse grain density of any molecule which wokrs both in farfield and near field is simply coarse grained grid whwre charge in each grid point is fited to reproduce the electrostatic poential of the full transition density grid. We can put there the damping wusing error function or lorenzian to prevent singularity at zero, but in general the molecule will be never so close. This grid can be very sparse e.g. just 3x3x1 for phtalocyanine or 3x2x1 for PTCDA, then we ommit the problem with searching the centers of lobes. In this case we can forget the dipole.

One more thing - when you were estimating the memory requirement you were considering that each of 64 molecules is different molecule and have different charges. But this is rarely the case in experiment. Typically all molecules in the sample are the same. Sometimes the molecule on the tip can be different sometimes also the same. This means we can load only two molecule instances each with its set of effectve charges or multipole coefs and then only appoly rigid body rotations to each. This makes extremely compact representation which fully fits in local memory without any problems.

---

# Chat GPT 5.5 medium

Yes — this is probably the cleanest representation for this project.

Instead of trying to identify physical “lobes” of the transition density, use a **fixed molecule-local coarse grid** and fit effective smeared charges on that grid to reproduce the electrostatic potential of the full transition density. This is much more robust, more regular for GPU, and easier to validate.

The important conceptual shift is:

[
\text{not: find chemically meaningful charge centers}
]

but

[
\text{fit a compact numerical quadrature representation of the transition-density potential}.
]

That is exactly what you need for Frenkel couplings.

---

# 1. Coarse-grid transition-density representation

For each molecular transition (a), represent the transition density by charges on fixed molecule-local grid sites:

[
\rho_a^{\rm tr}(\mathbf r)
\approx
\sum_{p=1}^{n_g}
q_{ap}
g_\sigma(\mathbf r-\mathbf R_p),
]

where (\mathbf R_p) are fixed grid points in the molecule-local frame.

For PTCDA, for example:

[
n_g = 3\times 2\times 1 = 6
]

or perhaps

[
n_g = 4\times 2\times 1 = 8.
]

For phthalocyanine:

[
n_g = 3\times 3\times 1 = 9
]

or maybe (5\times5\times1) if you want more resolution.

The charge sites are not atoms and not “lobes”. They are just basis functions for fitting the potential.

Each site has:

[
\mathbf R_p,\quad q_{ap},\quad \sigma_p.
]

The transition-density coupling between state (a) on molecule (i) and state (b) on molecule (j) becomes

[
J_{ia,jb}
=========

k_e
\sum_{p=1}^{n_g}
\sum_{q=1}^{n_g}
q_{ap}q_{bq}
K_{\sigma_{pq}}
\left(
\left|
\mathbf r_{ip}-\mathbf r_{jq}
\right|
\right),
]

where

[
\mathbf r_{ip}
==============

\mathbf R_i\mathbf R_p+\mathbf T_i.
]

For Gaussian charges:

[
K_{\sigma_{pq}}(r)
==================

\frac{
\operatorname{erf}(r/\sigma_{pq})
}{r}.
]

For a fast softened approximation:

[
K_{\sigma_{pq}}(r)
==================

\frac{1}{\sqrt{r^2+\sigma_{pq}^2}}.
]

This is regular at short range and becomes (1/r) at long range.

This representation is excellent for GPU because all transitions of the same molecule type share the same grid topology. Only the charge coefficients differ between transitions.

---

# 2. Fitting to electrostatic potential, not density

Given a full transition-density grid (\rho_a^{\rm tr}(\mathbf r)), compute its reference electrostatic potential:

[
\phi_a^{\rm ref}(\mathbf r)
===========================

k_e
\int
\frac{
\rho_a^{\rm tr}(\mathbf r')
}{
|\mathbf r-\mathbf r'|
}
d\mathbf r'.
]

Then fit coarse charges (q_{ap}) so that

[
\phi_a^{\rm fit}(\mathbf r_s)
=============================

k_e
\sum_p q_{ap}
K_\sigma(|\mathbf r_s-\mathbf R_p|)
]

matches (\phi_a^{\rm ref}(\mathbf r_s)) on a set of sample points (\mathbf r_s) around the molecule.

So the least-squares problem is

[
\min_{{q_p}}
\sum_s
w_s
\left[
\phi^{\rm ref}(\mathbf r_s)
---------------------------

k_e
\sum_p q_p K_\sigma(|\mathbf r_s-\mathbf R_p|)
\right]^2.
]

This is linear in the unknown charges (q_p).

Define

[
A_{sp}
======

k_e K_\sigma(|\mathbf r_s-\mathbf R_p|),
]

[
b_s
===

\phi^{\rm ref}(\mathbf r_s).
]

Then solve

[
\min_q
|W^{1/2}(Aq-b)|^2.
]

With constraints:

[
\sum_p q_p = 0
]

and optionally

[
\sum_p q_p \mathbf R_p
======================

\boldsymbol\mu^{\rm tr}.
]

This is much better than atomic transition charges because the fitting objective is directly the quantity entering the excitonic coupling.

---

# 3. Why this works in both far field and near field

The far field of any neutral transition density is controlled by its multipoles:

[
Q = \int \rho(\mathbf r)d\mathbf r = 0,
]

[
\boldsymbol\mu = \int \mathbf r \rho(\mathbf r)d\mathbf r,
]

[
Q_{\alpha\beta} = \int \rho(\mathbf r)
(3r_\alpha r_\beta-r^2\delta_{\alpha\beta})d\mathbf r,
]

etc.

If your coarse-grid charges are constrained to reproduce at least:

[
\sum_p q_p=0,
]

[
\sum_p q_p\mathbf R_p=\boldsymbol\mu,
]

then the leading far-field behavior is correct.

Near field is handled because the charge sites are distributed across the molecular area, not collapsed into one dipole at the center.

So compared with a pure dipole:

[
J^{\rm dip}\sim R^{-3},
]

the coarse grid captures the lateral shape of the transition density at contact distances.

This is exactly the compromise you want.

---

# 4. But a warning: 3×2×1 may be too small for some transitions

A (3\times2\times1) PTCDA grid gives only 6 charges. With the neutrality constraint, you have 5 independent parameters. If you also constrain the full dipole vector, you have only 2 remaining degrees of freedom.

That may be enough for the lowest bright transition, but not necessarily for higher transitions with nodal structure.

So I would make the grid size adjustable:

[
3\times2,\quad 4\times2,\quad 5\times2,\quad 5\times3.
]

Then benchmark the coupling error against full grid convolution.

A good strategy is:

[
\boxed{
\text{use the smallest grid that reproduces }J(R,\theta)\text{ within target error}.
}
]

For example:

* (3\times2): fastest/debug/default;
* (4\times2): likely good for PTCDA;
* (5\times3): more robust near contact;
* atom-centered or dense grid: reference only.

The representation should be selected by error, not aesthetics.

---

# 5. Fitting with constraints

For one transition, solve:

[
\min_q
|Aq-b|^2
+
\lambda |q|^2
]

subject to:

[
Cq=d.
]

Here (C) contains the constraints.

For neutrality:

[
C_0 = (1,1,\dots,1),
\qquad
d_0=0.
]

For dipole:

[
C_x = (X_1,X_2,\dots,X_{n_g}),
]

[
C_y = (Y_1,Y_2,\dots,Y_{n_g}),
]

[
C_z = (Z_1,Z_2,\dots,Z_{n_g}),
]

and

[
d=(0,\mu_x,\mu_y,\mu_z).
]

The constrained normal equations are:

[
\begin{pmatrix}
A^TWA+\lambda I & C^T\
C & 0
\end{pmatrix}
\begin{pmatrix}
q\
\eta
\end{pmatrix}
=============

\begin{pmatrix}
A^TWb\
d
\end{pmatrix}.
]

This is an offline CPU operation. The result exported to GPU is just:

[
q_{ap}.
]

If the transition density has very weak dipole but strong quadrupole character, also constrain quadrupole moments. But with only 6–9 charges, do not overconstrain.

---

# 6. Choice of fitting sample points

You should not fit the potential inside the molecular density too strongly because the coarse model cannot reproduce all short-wavelength details. Fit mostly in the region where other molecules will actually sample the potential.

For a planar molecule, sample points can be:

[
z = 2.5, 3.0, 4.0, 5.0\ {\rm Å}
]

above and below the molecular plane, plus lateral shell points around the molecular perimeter.

For exciton coupling in aggregates, the relevant distances are intermolecular separations, so fit the potential outside the van der Waals envelope.

Use weights such as:

[
w_s =
\frac{1}{
|\phi^{\rm ref}(\mathbf r_s)|^2+\phi_0^2
}
]

if you want relative accuracy, or uniform weights if absolute meV error matters.

Another good objective is to sample physically relevant dimer configurations directly. For many relative poses (g), fit charges to reproduce:

[
J_g^{\rm ref}
=============

\iint
\rho_A(\mathbf r)
\frac{1}{|\mathbf r-\mathbf r'|}
\rho_B(g^{-1}\mathbf r')
d\mathbf r d\mathbf r'.
]

But potential fitting is more general and simpler.

---

# 7. The homogeneous molecule case changes everything

You are absolutely right: I overestimated memory if I implicitly treated every molecule as having independent charge data.

In the realistic experiment:

[
\text{many molecular instances}
\quad\neq\quad
\text{many molecular types}.
]

Usually:

[
N_{\rm type}=1
]

or

[
N_{\rm type}=2
]

if the tip molecule differs from the surface molecule.

Then we store only:

[
{\mathbf R_p^{\rm type}, q_{ap}^{\rm type}, \sigma_p^{\rm type}}
]

for each molecule type.

Each molecule instance stores only its pose:

[
\mathbf T_i,\quad \mathbf Q_i
]

and maybe type id:

[
t_i.
]

So the per-instance data is tiny.

For identical surface molecules:

[
\mathbf r_{iap}
===============

\mathbf R_i\mathbf R_p+\mathbf T_i
]

and (q_{ap}) is shared.

This is much better for GPU.

---

# 8. Revised memory estimate

Suppose:

[
N_{\rm type}=2,\quad m=4,\quad n_g=9.
]

For each type, store local grid positions:

[
n_g \times \text{float4}
]

where float4 = ((x,y,z,\sigma)).

Then transition charges:

[
m\times n_g
]

floats.

Total per type:

[
9\times4 + 4\times9 = 72
]

floats.

For two types:

[
144 \text{ floats} \approx 576\text{ bytes}.
]

This is nothing.

Surface molecule poses:

[
15\times(\text{pos}+\text{quat})
================================

# 15\times(4+4)

120
]

floats.

Also tiny.

The actual large object is not the transition-charge representation. It is the Hamiltonian/eigenvectors if full diagonalization is used:

[
64\times64 = 4096
]

floats (\approx 16) kB.

So the dominant local memory is still the dense matrix, not molecular data.

This supports a very clean design:

[
\boxed{
\text{load molecule type data once into local memory; use poses to generate all instances.}
}
]

---

# 9. New GPU data model

Use global constant/read-only arrays:

```c
typeGridPos[type][gridPoint]       // float4: x,y,z,sigma
typeQ[type][state][gridPoint]      // float
typeE0[type][state]                // float
typeMu[type][state]                // float4
typeDQ[type][state][gridPoint]     // optional for Stark
```

Surface instances:

```c
surfType[i]                        // usually all 0
surfPos[i]                         // float4
surfQuat[i]                        // float4
```

Tip per pixel:

```c
tipPos[pixel]
tipQuat[pixel]
tipType                           // usually fixed
```

Static precomputed surface Hamiltonian/eigenbasis:

```c
Hsurf[s][s]                        // for full mode or validation
eps[nu]
U[s][nu]
```

Inside one workgroup, load:

```c
local gridPosTipType[n_g]
local qTip[m][n_g]
local gridPosSurfType[n_g]
local qSurf[m][n_g]
local surfPose[Nsurf]
local tipPose
```

Then generate transformed coordinates as needed.

For the surface molecules, because their poses are static, you have two choices.

### Option A: precompute transformed surface grid coordinates

Store:

[
\mathbf r_{ip}
]

for every molecule instance and grid point.

Memory:

[
15\times9\times\text{float4}
============================

135\times16
\approx 2.2\text{ kB}.
]

This is tiny. Do it.

Then the kernel does not rotate surface grid points repeatedly.

### Option B: compute surface grid coordinates on the fly

Saves a tiny amount of memory but costs repeated quaternion rotations.

Not worth it.

So I recommend:

[
\boxed{
\text{precompute transformed surface grid coordinates globally.}
}
]

The tip grid coordinates are computed per pixel and stored in local memory.

---

# 10. Coupling with molecule types

For state (a) on tip type (T) and state (b) on surface type (S):

[
J_{a,Ib}
========

k_e
\sum_{p=1}^{n_g^T}
\sum_{q=1}^{n_g^S}
q^T_{ap} q^S_{bq}
K(|\mathbf r^T_p-\mathbf r^S_{Iq}|).
]

Since (n_g) is very small, e.g. 6–16, this is cheap.

For (m=4), (N_{\rm surf}=15), (n_g=9):

[
4\times 60\times 9^2
====================

19440
]

kernel-pair evaluations per pixel.

This is easily workgroup-friendly.

---

# 11. Revised Schur kernel with molecule-type data

The Schur kernel can now be cleaner:

```text
one workgroup = one pixel

1. Load tip pose.
2. Transform tip coarse-grid sites into local memory.
3. Surface grid sites are already transformed and read from global memory.
4. For each tip state a and surface state s=(I,b):
       compute J[a,s] from coarse-grid charge interaction.
5. Transform J[a,s] to surface exciton basis:
       Vtilde[a,nu] = sum_s J[a,s] U[s,nu].
6. Compute Schur self-energy and spectral signal.
```

This is very compact.

The local memory could be:

```c
local float4 tipGrid[MAX_GRID];
local float V[MAX_M*MAX_SURF_STATE];
local float Vt[MAX_M*MAX_ACTIVE];
local float2 Sigma[MAX_M*MAX_M];
```

No need to store per-molecule charges in local memory if they are in constant memory. But copying them into local memory is cheap and may be faster.

---

# 12. Should the coarse grid replace dipoles completely?

For this small system, yes, probably.

The coarse grid automatically reproduces dipole behavior if constrained correctly. The dipole formula is only useful as an asymptotic acceleration for large systems.

Since here:

[
N_{\rm surf}\le15
]

and

[
n_g\le 16,
]

there is no need to use dipole approximation for speed.

So the coupling should simply be:

[
J_{ia,jb}^{\rm coarse}
======================

k_e
\sum_{pq}
q_{ap}q_{bq}
K(r_{pq}).
]

No switching, no dipole branch, no multipoles initially.

That is good because branching hurts clarity and sometimes GPU performance.

---

# 13. What about multiple transitions?

For each molecule type, grid positions are shared:

[
\mathbf R_p
]

but charges are transition-dependent:

[
q_{ap}.
]

For (m=4), store:

[
q_{ap},\quad a=1...4,\ p=1...n_g.
]

If you include local transition mixing by fields, you might need transition transition-dipole couplings between excited states, but ignore this first.

The diagonal state energies are:

[
H_{ia,ia}=E_a^0+\Delta E_{ia}^{\rm Stark}.
]

The off-diagonal local couplings on the same molecule are usually zero if the local excited states are already isolated-molecule eigenstates:

[
H_{ia,ib}=0,\quad a\ne b.
]

Unless the local electric field mixes them.

---

# 14. Stark shift using the same coarse grid

A nice consequence: use a different coarse grid charge set for **difference density**:

[
\Delta\rho_a(\mathbf r)
=======================

\rho_a^{\rm exc}(\mathbf r)-\rho^{\rm gs}(\mathbf r).
]

Fit it on the same grid:

[
\Delta\rho_a(\mathbf r)
\approx
\sum_p
\Delta q_{ap}
g_\sigma(\mathbf r-\mathbf R_p).
]

Then the Stark shift is:

[
\Delta E_{ia}
=============

\sum_p
\Delta q_{ap}\phi_{\rm ext}(\mathbf r_{ip}).
]

This is very GPU-friendly and consistent with your transition-density representation.

So for each state (a), you store two charge vectors:

[
q^{\rm tr}_{ap}
]

for excitonic coupling, and

[
\Delta q_{ap}
]

for Stark shift.

They are not the same.

---

# 15. Full Hamiltonian with identical molecule type

The full Hamiltonian build becomes very regular.

State index:

[
s = i m + a.
]

Molecule index:

[
i = s/m.
]

Local transition index:

[
a = s \bmod m.
]

Then:

[
H_{s,t}
=======

\begin{cases}
E_a^0+\Delta E_{ia}, & s=t,\
J_{ia,jb}, & i\ne j,\
0, & i=j,\ a\ne b\ \text{initially}.
\end{cases}
]

For identical surface molecules, (E_a^0), (q_{ap}), (\Delta q_{ap}) are shared.

Only poses differ.

The tip molecule may have different type and different base energies.

---

# 16. Full Hamiltonian kernel: matrix construction becomes cheap

For (n=64), each matrix element can be assigned to one thread loop:

```c
for(idx=lid; idx<n*n; idx+=WG_SIZE){
    p = idx / n;
    q = idx % n;

    i = p / m;
    a = p % m;
    j = q / m;
    b = q % m;

    if(p==q){
        H[idx] = E0[type_i][a] + Stark(i,a);
    }else if(i==j){
        H[idx] = 0.0f; // unless local state mixing
    }else{
        H[idx] = coarse_grid_coupling(type_i,a,pose_i,type_j,b,pose_j);
    }
}
```

But note: if surface-surface couplings are static, do **not** recompute them per pixel. Store (H_S). Only compute tip-surface rows/columns and tip diagonal.

For full mode with surface Stark shifts, still use static off-diagonal surface couplings:

[
H_{S,ij}^{\rm offdiag} = \text{precomputed}.
]

Only diagonal surface Stark shifts change.

That saves time and keeps the full kernel efficient.

---

# 17. Better “full” kernel structure

Instead of computing all (n^2) couplings every pixel, build the matrix by block:

[
H =
\begin{pmatrix}
H_T & V \
V^T & H_S^0 + D_S
\end{pmatrix}.
]

Where:

* (H_S^0) is static and precomputed;
* (D_S) is diagonal Stark correction;
* (V) is tip-surface coupling;
* (H_T) is tip local block.

So per pixel:

```text
copy H_S_static into local H
compute D_S diagonal additions if enabled
compute H_T
compute V tip-surface couplings
run Jacobi
```

Copying (60\times60) floats from global memory is not too bad. But if you use Schur, you avoid this entirely.

---

# 18. The coarse grid gives regular GPU loops

The coupling function becomes something like:

```c
float coupling(int typeA, int stateA, float4* gridA,
               int typeB, int stateB, float4* gridB){
    float sum = 0.0f;
    for(int p=0; p<nGridA; p++){
        float qp = q[typeA][stateA][p];
        float4 rp = gridA[p];
        for(int q=0; q<nGridB; q++){
            float qq = q[typeB][stateB][q];
            float4 rq = gridB[q];
            float3 d = rp.xyz - rq.xyz;
            float r2 = dot(d,d);
            float s2 = sigma2[typeA][p] + sigma2[typeB][q];
            sum += qp*qq*rsqrt(r2+s2);
        }
    }
    return COULOMB_CONST*sum;
}
```

If all grid points share one (\sigma), even simpler:

[
K(r)=\frac{1}{\sqrt{r^2+\sigma^2}}.
]

Then (\sigma) is a type-level or transition-level scalar.

This regularity matters more than tiny theoretical optimality.

---

# 19. Potential issue: coarse grid charges can become large and oscillatory

A coarse potential fit can produce alternating large positive/negative charges that cancel in the far field but behave badly at short range.

Use regularization:

[
\min_q
|Aq-b|^2+\lambda |q|^2.
]

Or better:

[
\lambda \sum_p \frac{q_p^2}{w_p}
]

where (w_p) corresponds to the expected magnitude/support of that grid basis function.

Also enforce not only neutrality/dipole, but avoid pathological charges.

A practical rule:

[
|q_p| < q_{\rm max}
]

or include a soft penalty:

[
\lambda_4 \sum_p q_p^4.
]

But initially, linear Tikhonov regularization is enough.

---

# 20. Grid placement

For PTCDA, I would not necessarily use a perfectly uniform rectangular grid without considering molecular shape. Better use a molecule-local rectangular grid clipped or weighted by molecular footprint.

Example for PTCDA:

[
x\in{-L_x/3,0,+L_x/3},
]

[
y\in{-L_y/4,+L_y/4}
]

for (3\times2).

Or (4\times2):

[
x\in{-3L_x/8,-L_x/8,+L_x/8,+3L_x/8}.
]

For phthalocyanine:

[
3\times3
]

or

[
5\times5
]

with corner points possibly removed if outside the molecular disk.

The grid should cover the transition-density support, not necessarily the nuclear framework exactly.

For planar molecules, all sites can be at (z=0), but you may give them an out-of-plane width (\sigma_z) implicitly through softening.

---

# 21. Reference quality test

For each candidate grid size, compute the error matrix for a set of relative dimer poses:

[
\epsilon_g =
J_g^{\rm fit}-J_g^{\rm ref}.
]

Use several metrics:

[
{\rm RMSE}
==========

\sqrt{
\frac{1}{N_g}
\sum_g
\epsilon_g^2
},
]

[
{\rm MAE}
=========

\frac{1}{N_g}\sum_g|\epsilon_g|,
]

and relative error with floor:

[
\epsilon_g^{\rm rel}
====================

\frac{
|J_g^{\rm fit}-J_g^{\rm ref}|
}{
|J_g^{\rm ref}|+J_{\rm floor}
}.
]

Use

[
J_{\rm floor}\sim 1\ {\rm meV}.
]

The grid is good enough if the error is below the physical uncertainty from TDDFT/exciton linewidths, perhaps:

[
{\rm RMSE}<5\text{--}10\ {\rm meV}
]

depending on your target.

---

# 22. Revised implementation recommendation

Your revised idea suggests an even simpler staged implementation.

## Offline

For each molecule type and each local transition:

1. compute full transition-density grid;
2. compute reference electrostatic potential;
3. choose coarse molecule-local grid;
4. fit smeared charges (q^{\rm tr}_{ap});
5. fit difference-density charges (\Delta q_{ap}) for Stark shifts;
6. validate against FFT Coulomb couplings for dimers;
7. export compact type table.

## GPU runtime

Per pixel:

1. load tip pose;
2. transform only the tip coarse grid;
3. use precomputed transformed surface grids;
4. compute tip-surface Frenkel couplings;
5. use Schur Green function or full diagonalization;
6. output spectral/image observables.

This is a much cleaner architecture than arbitrary lobe detection.

---

# 23. Final architecture

The core runtime representation is:

[
\boxed{
\text{molecule type}
====================

\left[
\mathbf R_p,\sigma_p,
q^{\rm tr}*{ap},
\Delta q*{ap},
E_a^0,
\boldsymbol\mu_a
\right]
}
]

and

[
\boxed{
\text{molecule instance}
========================

\left[
\text{type id},
\mathbf T_i,
\mathbf Q_i
\right].
}
]

Then everything follows by rigid transformation.

This is exactly what you want for OpenCL:

* compact type data;
* small fixed loops;
* no topology search at runtime;
* no atom-specific branching;
* no per-instance charge storage;
* no singularities if softened/Gaussian charges are used;
* full compatibility with Schur and full (64\times64) diagonalization.

I agree with your correction: with identical molecules, the memory pressure from charge representation is essentially negligible. The real design choice becomes whether the per-pixel kernel computes a Schur spectral function or performs local-memory diagonalization. My recommendation remains:

[
\boxed{
\text{Schur/Green kernel for production, full Jacobi kernel for validation and surface-Stark mode.}
}
]

But now the coupling representation itself becomes very clean: **coarse fixed molecule-local potential-fitted grids**, not dipoles, not atom charges, not detected lobes.



