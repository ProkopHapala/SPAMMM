https://chatgpt.com/share/6a5e1609-e2b4-83eb-887d-6ad0a786f010

---

USER

Why Dyson orbitals are usefull and important for STM? I recently saw people using it? Can we efficiently compute Dyson orbitals using DFTB+ ?

---

# Chat GPT 5.6 sol hi

## The central idea

A conventional molecular orbital describes an electron in an **effective one-electron Hamiltonian of one fixed charge state**. STM, however, physically transfers an electron:

[
\Psi_i^{N}
\longleftrightarrow
\Psi_f^{N\pm1}.
]

The appropriate one-electron object is therefore not generally a Kohn–Sham orbital of the neutral molecule, but the transition amplitude between the two many-electron states—the **Dyson orbital**:

[
d_p^-=\left\langle\Psi_f^{N-1}\middle|\hat a_p\middle|\Psi_i^N\right\rangle ,
\qquad
d_p^+=\left\langle\Psi_f^{N+1}\middle|\hat a_p^\dagger\middle|\Psi_i^N\right\rangle ,
]

or, in real space,

[
\psi_{fi}^{D,\pm}(\mathbf r)
============================

\sum_p d_p^\pm \phi_p(\mathbf r).
]

Thus a Dyson orbital answers:

> At which position, and with which phase and amplitude, can one electron be added or removed so that the molecule changes from the initial many-electron state (i) to the particular charged state (f)?

This is exactly the operation performed by STM/STS. Dyson orbitals generalize the molecular-orbital picture to correlated many-electron states, while their norms give the corresponding pole or spectral strengths. ([AIP Publishing][1])

---

# Why they enter STM naturally

Write the weak tip–sample coupling schematically as

[
\hat H_T =
\sum_\sigma\int d\mathbf r,
t_T(\mathbf r-\mathbf R)
\hat c_{T\sigma}^{\dagger}
\hat\psi_\sigma(\mathbf r)
+\text{h.c.}
]

For tunnelling from an (N)-electron molecular state into a particular (N-1) state, the many-body matrix element becomes

[
M_{fi}(\mathbf R)
=================

\int d\mathbf r,
t_T(\mathbf r-\mathbf R)
\underbrace{
\left\langle\Psi_f^{N-1}
\middle|
\hat\psi(\mathbf r)
\middle|
\Psi_i^N\right\rangle
}*{\psi*{fi}^{D,-}(\mathbf r)} .
]

The entire complicated (3N)-dimensional many-electron overlap has therefore collapsed into a one-electron Dyson orbital.

For an ideal (s)-wave point tip, the Tersoff–Hamann approximation gives approximately

[
\frac{dI}{dV}(\mathbf R,V_f)
\propto
\left|
\psi_{fi}^{D}(\mathbf R)
\right|^2
]

for an isolated resonance. For a non-(s) tip, the appropriate Chen derivative operator acts on the **Dyson orbital**, just as it normally acts on a Kohn–Sham orbital. The recent multireference STM formulation makes this connection explicitly and shows examples where canonical orbitals fail but Dyson-orbital maps reproduce the measured contrast. ([arXiv][2])

More generally, STM probes the interacting local spectral function,

[
A(\mathbf r,\omega)
===================

\sum_f
\left|\psi^{D,+}*{fi}(\mathbf r)\right|^2
\delta!\left[
\omega-
(E_f^{N+1}-E_i^N)
\right]
+
\sum_f
\left|\psi^{D,-}*{fi}(\mathbf r)\right|^2
\delta!\left[
\omega+
(E_f^{N-1}-E_i^N)
\right].
]

So the interacting analogue of LDOS is literally a sum of Dyson-orbital densities, each located at a genuine electron-addition or electron-removal energy.

## Why ordinary HOMO/LUMO images often work

Suppose both states are frozen-orbital single determinants. Removing one occupied orbital gives

[
\Psi_f^{N-1}
============

\hat a_i\Psi_0^N .
]

Then

[
\psi_{fi}^{D,-}(\mathbf r)=\phi_i(\mathbf r),
\qquad
|\psi_D|^2=1.
]

Therefore, for a closed-shell, weakly correlated molecule, the first negative-ion and positive-ion resonances often look nearly identical to the neutral LUMO and HOMO. This explains the considerable empirical success of conventional orbital imaging; pentacene is one example discussed in the recent work. ([arXiv][2])

But the equivalence breaks down through two distinct mechanisms.

### Orbital relaxation

After charging, all orbitals respond to the added or removed electron. Even if both charge states remain single determinants, the Dyson orbital becomes a linear combination of several neutral orbitals.

### Electronic correlation

A correlated neutral state may connect to several ionic states:

[
\Psi_i^N
\overset{\hat\psi}{\longrightarrow}
\Psi_0^{N-1},\Psi_1^{N-1},\Psi_2^{N-1},\ldots
]

The one-electron spectral weight becomes fragmented. One apparent “HOMO” may therefore produce several ionization resonances, each having a different energy, strength and possibly spatial shape. Shake-up configurations such as (2h1p) contribute, and the norm

[
Z_f=\int|\psi_{fi}^D(\mathbf r)|^2,d\mathbf r
]

is typically below one for an individual resonance.

This is particularly important for:

* diradicals and polyradicals;
* nearly degenerate Kekulé or valence-bond structures;
* open-shell singlets;
* molecules where several charged multiplets are close in energy;
* Coulomb-blockade STM on thin insulating layers;
* claims about assigning separate SOMO, HOMO and LUMO levels from STS.

The 2025 multireference analysis specifically argues that canonical-orbital assignments become unreliable for strongly correlated polyradicals and that different charged many-body states must be considered. ([arXiv][2])

A nuance: the Dyson orbital has an arbitrary global phase, so I would not call the complex orbital itself a directly measurable observable without qualification. What is physically measurable is the resulting tunnelling matrix element, spatial intensity under specified tip assumptions, resonance energy and spectral strength.

---

# Can DFTB+ calculate them?

## Directly: apparently not at present

I could not find a documented direct Dyson-orbital calculation in the current stable DFTB+ release, which is release 25.1. DFTB+ does provide:

* charged and spin-polarized SCC-DFTB calculations;
* molecular-orbital eigenvectors;
* TD-DFTB neutral excitations;
* natural transition orbitals;
* waveplot visualization;
* REKS methods for some static-correlation problems.

But the documented natural transition orbitals are obtained from the (N\rightarrow N) **neutral excitation transition density**. They are not electron-attachment or electron-removal Dyson orbitals connecting (N) and (N\pm1) states. ([DFTB+][3])

Nevertheless, a useful approximate Dyson calculation can be added to DFTB+ rather easily.

# Level 1: frozen-orbital DFTB approximation

Use the neutral SCC-DFTB HOMO or LUMO directly:

[
\psi_D^-\approx \phi_{\mathrm{HOMO}},
\qquad
\psi_D^+\approx \phi_{\mathrm{LUMO}}.
]

This is the Koopmans-like approximation. It costs essentially nothing and is often sufficient for ordinary closed-shell molecules.

For the resonance energies, however, use vertical total-energy differences when possible:

[
I_f=E_f^{N-1}(\mathbf R_N)-E_0^N(\mathbf R_N),
]

[
A_f=E_0^N(\mathbf R_N)-E_f^{N+1}(\mathbf R_N),
]

rather than raw DFTB orbital eigenvalues. DFTB+ supports charged calculations using the `Charge` option and spin-polarized calculations using `SpinPolarisation`. ([DFTB+ Recipes][4])

Long-range-corrected DFTB is preferable when electron attachment, charge transfer or diffuse polarization is important; the LC-DFTB formulation was developed partly to improve ionization potentials, electron affinities and long-range response. ([arXiv][5])

---

# Level 2: relaxed single-determinant Dyson orbitals

This would be a particularly attractive extension for DFTB+ because it requires only:

1. one SCC-DFTB calculation for (N);
2. one SCC-DFTB calculation for the relevant (N-1) or (N+1) state;
3. a small overlap-matrix calculation.

Suppose the neutral and cation are represented by different Slater determinants. Let

[
C^N_{\mathrm{occ}}
]

contain the occupied neutral spin-orbital coefficients and

[
C^{N-1}_{\mathrm{occ}}
]

the occupied cation coefficients in the common DFTB atomic-orbital basis. Because DFTB uses a nonorthogonal basis, form the cross-overlap

[
B=
\left(C^{N-1}*{\mathrm{occ}}\right)^\dagger
S*{\mathrm{AO}}
C^N_{\mathrm{occ}} .
]

For electron removal, (B) has dimensions ((N-1)\times N). The exact Dyson orbital between these two determinants is

[
\psi_D^-(\mathbf r)
===================

\sum_{i=1}^{N} d_i,\phi_i^N(\mathbf r),
]

where

[
d_i=(-1)^{i+1}\det B_{\bar i},
]

and (B_{\bar i}) is the square matrix obtained by deleting column (i).

So even a mean-field Dyson orbital is not necessarily just one orbital: orbital relaxation mixes all occupied orbitals having compatible symmetry.

## Efficient implementation

One should not calculate every minor separately. The cofactor vector satisfies

[
B,\mathbf d=0.
]

Therefore:

1. compute the SVD or rank-revealing QR of (B);
2. take its one-dimensional right null vector (\mathbf v_0);
3. scale it by the product of the nonzero singular values:

[
|\mathbf d|
===========

# \sqrt{\det(BB^\dagger)}

\prod_{k=1}^{N-1}\sigma_k .
]

Thus

[
\mathbf d
=========

e^{i\theta}
\left(\prod_k\sigma_k\right)\mathbf v_0.
]

The global phase (e^{i\theta}) is irrelevant for an isolated STM intensity map.

The postprocessing cost is (O(N_{\rm occ}^3)), comparable to one small dense factorization and normally negligible relative to the SCC-DFTB calculation itself. For large systems, the almost unchanged core-orbital blocks can be factored separately, or one can perform the calculation in a localized valence subspace.

DFTB+ already writes the molecular-orbital eigenvectors required for this through `WriteEigenvectors`, and waveplot reconstructs real-space wavefunctions from these coefficients and the corresponding DFTB atomic basis functions. ([DFTB+ Recipes][6])

A practical implementation would need access to (S_{\rm AO}). This could be obtained through the DFTB+ library interface or by adding a small matrix-dump option to DFTB+. Once (\mathbf d) is known, it can be written in the same format as an ordinary DFTB eigenvector and passed to waveplot or PP-STM.

This level includes:

* self-consistent charge relaxation;
* spin polarization;
* nonorthogonality of the neutral and ionic determinants;
* a nontrivial pole strength caused by determinant mismatch.

It does **not** include true multiconfigurational correlation or fragmentation into multiple ionic final states.

---

# Level 3: an active-space DFTB–CI Dyson method

For the systems most interesting to you—bond-order switching, diradicals, competing Kekulé states and molecular QCA—I think this is the genuinely valuable route.

Use DFTB+ to generate a small set of frontier orbitals, then construct an effective correlated Hamiltonian:

[
\hat H_{\mathrm{act}}
=====================

\sum_{pq} h_{pq}\hat a_p^\dagger\hat a_q
+
\frac12
\sum_{pqrs}
V_{pqrs}
\hat a_p^\dagger
\hat a_q^\dagger
\hat a_s
\hat a_r .
]

The interaction integrals could use the same Mulliken transition-charge approximation underlying SCC-/TD-DFTB:

[
V_{pqrs}
\approx
\sum_{AB}
q_A^{pr},
\gamma_{AB},
q_B^{qs},
]

possibly including explicit exchange or a long-range-corrected kernel.

You then diagonalize the same effective Hamiltonian in the three particle-number sectors:

[
N-1,\qquad N,\qquad N+1.
]

Given CI expansions

[
|\Psi_i^N\rangle
================

\sum_I C_I^N|I\rangle ,
\qquad
|\Psi_f^{N-1}\rangle
====================

\sum_J C_J^{N-1}|J\rangle ,
]

the Dyson coefficients are simply

[
d_p^-=
\sum_{IJ}
(C_J^{N-1})^*
C_I^N
\langle J|\hat a_p|I\rangle .
]

Because (\hat a_p) only connects determinants differing by one occupation, this contraction is sparse and costs approximately

[
O(mN_{\rm det}),
]

which is negligible compared with obtaining the CI eigenstates.

For an active space such as CAS(8,8), this is extremely cheap. CAS(12,12) is larger but still quite realistic with bit-string determinants and Davidson diagonalization, especially because only a handful of low-energy states are required. The recent STM work used CASCI(12,12) for representative molecules, illustrating that relatively small (\pi)-active spaces can already expose the relevant difference between canonical and Dyson-orbital maps. ([arXiv][2])

This would give you:

* several positive- and negative-ion resonances;
* correct relative spectral weights (Z_f);
* spatial maps for every charged multiplet;
* genuine mixing of Kekulé and radical configurations;
* shake-up satellites within the chosen active space;
* spin-resolved tunnelling channels.

Tight-binding plus configuration interaction is an established general strategy, although such a Dyson-orbital active-space implementation does not appear to be a standard DFTB+ feature. ([ACS Publications][7])

---

# A serious STM-specific limitation of ordinary DFTB orbitals

Even when the coefficients (d_p) are accurate, STM probes the wavefunction several ångströms into vacuum. Standard DFTB uses a minimal, confined atomic-orbital basis. This can describe the nodal pattern around the molecule reasonably well, but its asymptotic vacuum decay may be poor, especially for:

* weakly bound anions;
* diffuse attachment states;
* Rydberg-like states;
* states close to the vacuum level.

Therefore, I would not directly compare a raw waveplot isosurface at large (z) with STM current.

A better workflow is:

[
\text{DFTB Dyson coefficients}
\rightarrow
\psi_D(\mathbf r,z_0)
\rightarrow
\text{vacuum propagation}
\rightarrow
\text{Bardeen/PP-STM}.
]

Above a matching plane (z_0), propagate each lateral Fourier component according to the vacuum barrier:

[
\tilde\psi_D(\mathbf k_\parallel,z)
===================================

\tilde\psi_D(\mathbf k_\parallel,z_0)
\exp\left[
-\sqrt{
\kappa_0^2+k_\parallel^2
},(z-z_0)
\right],
]

with approximately

[
\kappa_0
========

\sqrt{\frac{2m}{\hbar^2}
\left(\Phi-E\right)}.
]

This separates the chemically important Dyson coefficients from the universal vacuum filtering. It should fit very naturally into PP-STM: replace the sample MO by the propagated Dyson orbital and retain the usual tip-orbital matrix elements.

---

# What I would implement

For a useful first version, I would implement **relaxed determinant Dyson orbitals**:

[
\boxed{
\text{neutral SCC-DFTB}
+
\text{charged SCC-DFTB}
+
\text{cross-overlap SVD}
}
]

This is simple, fast and substantially more physical than plotting the neutral HOMO/LUMO. It would also provide a useful diagnostic:

[
Z_f=|\psi_D|^2.
]

When (Z_f\approx1) and the Dyson orbital overlaps almost completely with one neutral MO, the ordinary orbital interpretation is justified. When several neutral orbitals contribute strongly or (Z_f) decreases, the simple picture is breaking down.

For your strongly correlated molecular-automaton and polyradical problems, the next step should be a compact **DFTB-frontier-orbital CI in the (N,N\pm1) sectors**. The extraction of Dyson orbitals itself is then trivial; constructing a physically balanced effective interaction Hamiltonian is the main methodological research problem.

## Bottom line

**Dyson orbitals are important for STM because STM does not probe an electron sitting in a neutral-molecule orbital. It probes the amplitude for changing the complete molecule from an (N)-electron state to a particular (N\pm1)-electron state.**

DFTB+ cannot, as far as its current documented capabilities show, directly calculate correlated Dyson orbitals. But:

[
\boxed{\text{A relaxed single-determinant Dyson orbital can be added very cheaply.}}
]

It requires two SCC-DFTB charge-state calculations and one overlap-matrix SVD. For genuinely correlated radicals, a small active-space DFTB–CI extension would be necessary—and could still be orders of magnitude cheaper than ab initio CASSCF/EOM calculations while being much more physically appropriate than neutral HOMO/LUMO maps.

[1]: https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0016472/20014170/070902_1_5.0016472.pdf?utm_source=chatgpt.com "Dyson-orbital concepts for description of electrons in ..."
[2]: https://arxiv.org/html/2504.05896v1 "Theory of scanning tunneling spectroscopy beyond one-electron molecular orbitals: can we image molecular orbitals?"
[3]: https://dftbplus.org/download/stable.html "Stable release — DFTB+"
[4]: https://dftbplus-recipes.readthedocs.io/en/stable/basics/firstcalc.html?utm_source=chatgpt.com "First calculation with DFTB+"
[5]: https://arxiv.org/abs/1504.00243?utm_source=chatgpt.com "Implementation and benchmark of a long-range corrected functional in the density functional based tight-binding method"
[6]: https://dftbplus-recipes.readthedocs.io/en/stable/basics/waveplot.html "First steps with Waveplot — DFTB+ Recipes"
[7]: https://pubs.acs.org/doi/10.1021/ct700343t?utm_source=chatgpt.com "Tight-Binding Configuration Interaction (TBCI): A Noniterative ..."

---

USER

It seems to me closely related to Fukui function (diference of density when adding or removing electron). Can it be said that Fukui function is probability density of Dyson orbital? (Square of Dyson orbital amplitude in real space?) Or what is exactly difference? Seems to me that both are after full electronic relaxation (not instantaneous), right?

In one of my projects I want to compute chemical reaction (namely proton exchange, inelastic excitation, Frank-condon blokade, Huangh-rhys factor) when injecting electron or hole to polyaromatic heterocyclic molecule. Is Dyson orbital concept relevant and usefull for this topic? How can I use it in particular for this?

---

# Chat GPT 5.6 sol hi

Yes, they are closely related, but **the Fukui function is generally not the density of a Dyson orbital**.

A useful distinction is:

[
\boxed{\text{Dyson orbital: how the electron tunnels}}
]

[
\boxed{\text{Fukui/difference density: how the molecular density changes after charging}}
]

[
\boxed{\text{charged-state PES: how the nuclei subsequently move and react}}
]

These become identical only in the frozen-orbital, single-determinant limit.

---

# 1. Mathematical difference

Let (x=(\mathbf r,\sigma)) include position and spin.

For electron attachment from an (N)-electron state (i) to a particular ((N+1))-electron state (f), the Dyson orbital is

[
d_{fi}^{+}(x)
=============

\left\langle
\Psi_i^N
\middle|
\hat\psi(x)
\middle|
\Psi_f^{N+1}
\right\rangle .
]

For electron removal,

[
d_{fi}^{-}(x)
=============

\left\langle
\Psi_f^{N-1}
\middle|
\hat\psi(x)
\middle|
\Psi_i^N
\right\rangle .
]

This is a **transition amplitude between two many-electron states**. There is a separate Dyson orbital for every final cationic or anionic state (f). Its norm,

[
Z_{fi}=\int |d_{fi}^{\pm}(x)|^2,dx,
]

is a channel-dependent spectral or pole strength and need not equal one. Dyson orbitals and their norms generalize molecular orbitals and orbital occupations to correlated many-electron attachment and ionization processes. ([AIP Publishing][1])

By contrast, the electron-addition Fukui function is

[
f^+(\mathbf r)
==============

\left(
\frac{\partial\rho(\mathbf r)}{\partial N}
\right)*{v*{\rm ext}}^+
\approx
\rho_{N+1}(\mathbf r)-\rho_N(\mathbf r),
]

and the electron-removal Fukui function is conventionally

[
f^-(\mathbf r)
\approx
\rho_N(\mathbf r)-\rho_{N-1}(\mathbf r).
]

The derivative is taken at fixed external potential, meaning fixed nuclear positions and external fields. The finite-difference densities are normally the ground-state densities of the respective charge sectors.

Therefore,

[
\int f^\pm(\mathbf r),d\mathbf r=1.
]

But (f^\pm(\mathbf r)) can be locally negative because adding an electron can make some regions lose density through orbital relaxation. That fact alone proves that a general Fukui function cannot be (|d(\mathbf r)|^2), which is nonnegative. ([ACS Publications][2])

## The most important conceptual distinction

The Dyson orbital asks:

> What is the amplitude for adding or removing one electron at (x), while changing the molecule from state (i) to the specific final state (f)?

The Fukui function asks:

> How does the total one-electron density change when the equilibrium electron number changes from (N) to (N\pm1)?

The former is an **off-diagonal transition quantity in Fock space**. The latter is a **difference between diagonal one-particle densities**.

---

# 2. When are they equal?

Consider an (N)-electron neutral determinant with occupied orbitals

[
{\phi_1,\ldots,\phi_N},
]

and suppose an electron is added without changing any of these orbitals:

[
\Psi^{N+1}
==========

\hat a_a^\dagger\Psi^N.
]

Then

[
d^+(\mathbf r)=\phi_a(\mathbf r),
]

and

[
\rho_{N+1}(\mathbf r)
=====================

\rho_N(\mathbf r)+|\phi_a(\mathbf r)|^2.
]

Consequently,

[
\boxed{
f^+(\mathbf r)=|d^+(\mathbf r)|^2=|\phi_a(\mathbf r)|^2
}
]

in this frozen-orbital approximation. Likewise,

[
f^-(\mathbf r)=|d^-(\mathbf r)|^2
]

for frozen removal from one occupied orbital.

This is why frontier-orbital, Fukui-function and Dyson-orbital pictures often look similar for ordinary closed-shell molecules.

## Once orbitals relax

Let the neutral occupied orbitals be (\phi_i), while the relaxed anion orbitals are (\chi_j). Then

[
f^+(\mathbf r)
==============

## \sum_{j=1}^{N+1}|\chi_j(\mathbf r)|^2

\sum_{i=1}^{N}|\phi_i(\mathbf r)|^2.
]

This includes:

1. density from the newly occupied orbital;
2. polarization of all originally occupied orbitals;
3. charge transfer between molecular fragments;
4. spin polarization;
5. correlation-induced changes of orbital occupations.

The corresponding single-determinant Dyson orbital is instead a cofactor-weighted linear combination of the relaxed anion orbitals:

[
d^+(\mathbf r)
==============

\sum_j c_j\chi_j(\mathbf r),
]

where the (c_j) depend on overlaps between all neutral and anionic occupied orbitals.

Thus even at the relaxed Hartree–Fock or (\Delta)SCF level,

[
\boxed{f^+(\mathbf r)\neq |d^+(\mathbf r)|^2}
]

in general.

With correlation, the difference is greater because one initial state can connect to several ionic states (f), each with a different Dyson orbital and spectral weight.

---

# 3. An exact relation involving all Dyson orbitals

There is nevertheless a beautiful exact connection.

Insert a complete set of ((N-1))-electron states:

[
\rho_N(x)
=========

\langle\Psi_i^N|
\hat\psi^\dagger(x)\hat\psi(x)
|\Psi_i^N\rangle
================

\sum_f
\left|
\langle\Psi_f^{N-1}|
\hat\psi(x)
|\Psi_i^N\rangle
\right|^2.
]

Therefore,

[
\boxed{
\rho_N(x)=\sum_f |d_{fi}^-(x)|^2
}
]

when the sum includes the complete bound and continuum ion spectrum.

So the total density is a sum over **all removal Dyson-orbital densities**. Similarly, the density of an ((N+1))-electron state is a sum over all ways of removing one electron from it into every possible (N)-electron state.

This means that a density is related to a **complete family** of Dyson orbitals, not normally to one selected Dyson orbital.

---

# 4. Attachment/detachment densities are the closest bridge

For your purpose, a more useful object than the raw Fukui function may be the state-specific difference-density matrix

[
\Delta\gamma_f^+
================

\gamma_f^{N+1}-\gamma_i^N.
]

Diagonalizing it gives natural difference orbitals:

[
\Delta\gamma
============

\sum_p \lambda_p
|u_p\rangle\langle u_p|.
]

The positive and negative eigenvalues define attachment and detachment densities,

[
\rho_{\rm attach}(\mathbf r)
============================

\sum_{\lambda_p>0}
\lambda_p |u_p(\mathbf r)|^2,
]

[
\rho_{\rm detach}(\mathbf r)
============================

\sum_{\lambda_p<0}
|\lambda_p| |u_p(\mathbf r)|^2,
]

such that

[
\Delta\rho(\mathbf r)
=====================

## \rho_{\rm attach}(\mathbf r)

\rho_{\rm detach}(\mathbf r).
]

For electron addition,

[
\int\rho_{\rm attach},d\mathbf r
--------------------------------

\int\rho_{\rm detach},d\mathbf r
=1.
]

If strong electronic relaxation occurs, both integrals can be substantially larger than zero: more than one electron-equivalent may be rearranged even though the net charge change is exactly one. This attachment/detachment decomposition was introduced precisely to separate the positive and negative parts of many-electron density rearrangements. ([OSTI.gov][3])

Thus:

| Quantity                      | Physical interpretation                                |     |                                                                         |
| ----------------------------- | ------------------------------------------------------ | --- | ----------------------------------------------------------------------- |
| (                             | d_f^\pm(\mathbf r)                                     | ^2) | Channel-specific probability weight for tunnelling into ionic state (f) |
| (\Delta\rho_f^\pm(\mathbf r)) | Actual total density change after occupying state (f)  |     |                                                                         |
| (\rho_{\rm attach})           | Where density accumulates                              |     |                                                                         |
| (\rho_{\rm detach})           | Where density is expelled by relaxation                |     |                                                                         |
| (f^\pm)                       | Ground-state (N\leftrightarrow N\pm1) density response |     |                                                                         |

For an excited anionic or cationic state populated by STM, I would speak of a **state-specific attachment/detachment density**, not the ordinary ground-state Fukui function.

---

# 5. Are both “after full electronic relaxation”?

Not automatically. There are three separate meanings of relaxation.

## Electronic-state relaxation

A Dyson orbital can be constructed between:

* frozen Koopmans determinants;
* separately self-consistent neutral and ionic determinants;
* correlated state-specific wavefunctions;
* EOM attachment/ionization states.

Thus the Dyson definition itself does not require either frozen or relaxed orbitals. It inherits whatever relaxation and correlation are included in (\Psi_i^N) and (\Psi_f^{N\pm1}).

Similarly, a Fukui function may be approximated either by frontier orbitals,

[
f^+\approx|\phi_{\rm LUMO}|^2,
]

or by separately converged (N) and (N+1) densities. The latter includes self-consistent electronic relaxation; the former does not.

## Nuclear relaxation

The formal Fukui derivative is at fixed external potential, hence at fixed nuclear geometry:

[
f^+(\mathbf r;\mathbf R_0)
==========================

## \rho_{N+1}(\mathbf r;\mathbf R_0)

\rho_N(\mathbf r;\mathbf R_0).
]

Likewise, tunnelling is normally treated vertically at the instantaneous nuclear geometry (\mathbf R_0).

If you instead compare

[
\rho_{N+1}(\mathbf r;\mathbf R_{N+1}^{\rm min})
-----------------------------------------------

\rho_N(\mathbf r;\mathbf R_N^{\rm min}),
]

that includes geometry relaxation and is not strictly the conceptual-DFT Fukui function.

## Dynamical relaxation after injection

Even when (\Psi_f^{N+1}) is a fully relaxed electronic eigenstate, the tunnelling event is still sudden on the nuclear timescale. The electron-transfer operator projects the initial system onto one or more ionic eigenstates. Nuclear motion, vibrational relaxation, proton transfer and possible electronic decay occur afterward.

The Dyson orbital is therefore not “the density of the electron after it has relaxed.” After successful population of state (f), the density is

[
\rho_f^{N+1}(\mathbf r),
]

not

[
\rho_N(\mathbf r)+|d_f^+(\mathbf r)|^2.
]

---

# 6. Relevance to your electron-triggered proton-transfer project

The Dyson orbital is highly relevant, but it describes only one part of the process.

For a vibronic state,

[
|\Phi_{i\nu}^N\rangle
=====================

|\Psi_i^N(\mathbf R)\rangle
\chi_{i\nu}^N(\mathbf R),
]

the tunnelling matrix element from electrode or tip (\alpha) is schematically

[
M_{f\mu,i\nu}^{\alpha}
======================

\int d\mathbf R,
\chi_{f\mu}^{N\pm1*}(\mathbf R)
T_{fi}^{\alpha}(\mathbf R)
\chi_{i\nu}^{N}(\mathbf R),
]

where the electronic tunnelling amplitude is

[
T_{fi}^{\alpha}(\mathbf R)
==========================

\int dx,
t_\alpha(x;\mathbf R)
d_{fi}^{\pm}(x;\mathbf R).
]

This gives a clean division:

[
\boxed{
\text{Dyson orbital}
\rightarrow
T_{fi}^{\alpha}
\rightarrow
\text{which ionic state is populated and how rapidly}
}
]

[
\boxed{
\text{neutral/ionic PES}
\rightarrow
\chi_{i\nu},\chi_{f\mu}
\rightarrow
\text{vibrational excitation and reaction}
}
]

In the Condon approximation,

[
T_{fi}^{\alpha}(\mathbf R)
\approx
T_{fi}^{\alpha}(\mathbf R_0),
]

so the matrix element factorizes:

[
M_{f\mu,i\nu}^{\alpha}
\approx
T_{fi}^{\alpha}(\mathbf R_0)
\left\langle
\chi_{f\mu}^{N\pm1}
\middle|
\chi_{i\nu}^{N}
\right\rangle .
]

Consequently,

[
\Gamma_{i\nu\rightarrow f\mu}^{\alpha}
\propto
\underbrace{|T_{fi}^{\alpha}|^2}*{\text{Dyson/electronic factor}}
\underbrace{
\left|
\langle\chi*{f\mu}|\chi_{i\nu}\rangle
\right|^2
}_{\text{Franck--Condon factor}}.
]

Calculations of vibronic ionization processes use exactly this separation, with geometry-dependent Dyson orbitals becoming necessary when the Condon approximation breaks down. ([RSC Publishing][4])

---

# 7. Franck–Condon blockade and Huang–Rhys factors

Suppose charging displaces the equilibrium geometry along normal mode (k).

For neutral normal mode (k),

[
\Delta Q_k
==========

\sum_A
\sqrt{M_A},
\mathbf e_{Ak}\cdot
\left(
\mathbf R_A^{N\pm1}
-------------------

\mathbf R_A^{N}
\right).
]

In the displaced harmonic approximation,

[
S_k
===

\frac{\omega_k}{2\hbar}
(\Delta Q_k)^2
]

is the Huang–Rhys factor, and

[
\lambda_k
=========

# S_k\hbar\omega_k

\frac12\omega_k^2(\Delta Q_k)^2
]

is that mode’s reorganization energy.

For one mode, the zero-phonon overlap is

[
P_{0\rightarrow0}=e^{-S}.
]

For independent modes,

[
P_{0\rightarrow0}
=================

\exp\left(-\sum_k S_k\right).
]

Therefore, when

[
S_{\rm tot}=\sum_kS_k\gg1,
]

low-lying vibrational tunnelling channels become exponentially suppressed. This is the Franck–Condon blockade mechanism in sequential molecular transport. ([APS Links][5])

The Dyson orbital contributes the prefactor and state selection:

[
\Gamma_{0\to0}^{f}
\propto
|T_f[d_f]|^2 e^{-S_f}.
]

Different ionic states may have:

* different Dyson-orbital coupling to the tip;
* different equilibrium geometries;
* different (S_k);
* different proton-transfer barriers.

Therefore, the most strongly coupled ionic state need not be the state with the largest reaction yield.

---

# 8. What controls proton transfer?

Suppose the molecule initially has proton tautomer (A):

[
A^N
\xrightarrow{\Gamma_{\rm in}}
A^{N+1}
\xrightarrow{k_{\rm PT}^{N+1}}
B^{N+1}
\xrightarrow{\Gamma_{\rm out}}
B^N.
]

The roles are:

### Injection

[
\Gamma_{\rm in}
\propto
\left|
\int t_{\rm tip}(\mathbf r)
d_{A,f}^+(\mathbf r),d\mathbf r
\right|^2.
]

The attachment Dyson orbital tells you whether a particular anionic state can be reached efficiently from the chosen tip position and tip orbital.

### Driving force and barrier

Compute the charged-state proton-transfer surface

[
E_f^{N+1}(q_{\rm H}),
]

and compare it with

[
E_i^N(q_{\rm H}),
]

where (q_{\rm H}) is the proton coordinate.

Electron attachment may:

* change the relative energy of the two proton sites;
* lower or eliminate the proton-transfer barrier;
* change aromaticity or bond order;
* create a dissociative or strongly anharmonic proton potential.

Electron-induced intramolecular proton transfer is therefore fundamentally a coupled problem involving electron attachment and motion on an ionic potential-energy surface. ([ACS Publications][6])

### Competition with discharge

In a simple kinetic limit, the probability that proton transfer occurs during one charged-state residence is approximately

[
P_{\rm PT}
\approx
\frac{k_{\rm PT}}
{k_{\rm PT}+\Gamma_{\rm out}/\hbar+k_{\rm relax}+k_{\rm other}}.
]

Thus even a barrierless charged-state proton transfer may be inefficient if the anion lifetime is too short.

The Dyson orbital matters again because

[
\Gamma_{\rm out}(q_{\rm H})
\propto
|T_f[d_f(q_{\rm H})]|^2.
]

This suggests a particularly interesting mechanism for your system:

1. The electron couples strongly to the initial tautomer.
2. Proton transfer changes conjugation and the Dyson orbital.
3. The transferred-proton structure couples weakly back to the tip or substrate.
4. The ionic lifetime increases, or discharge preferentially leaves the product tautomer.

So you should consider calculating the Dyson orbital **along the proton coordinate**, not only at the initial geometry.

---

# 9. Difference density is more directly related to nuclear forces

For understanding why the proton starts moving, the state-specific difference density is generally more informative than the Dyson density:

[
\Delta\rho_f^+(\mathbf r;\mathbf R_0)
=====================================

## \rho_f^{N+1}(\mathbf r;\mathbf R_0)

\rho_i^N(\mathbf r;\mathbf R_0).
]

The vertical force change is

[
\Delta\mathbf F_A
=================

-\nabla_A
\left[
E_f^{N+1}(\mathbf R)
--------------------

E_i^N(\mathbf R)
\right]_{\mathbf R_0}.
]

In an exact fixed-basis Hellmann–Feynman picture, its electronic contribution is controlled by

[
\Delta\mathbf F_A
\sim
-\int
\Delta\rho_f^+(\mathbf r)
\nabla_A v_{\rm ext}(\mathbf r;\mathbf R)
,d\mathbf r.
]

Thus:

* the **Dyson orbital** predicts population of state (f);
* the **difference density** explains the initial force generated by occupying (f);
* the **ionic PES** determines the subsequent proton trajectory.

For chemical interpretation, I would calculate not only atomic Fukui charges but also:

[
\Delta P_{\mu\nu}
=================

## P_{\mu\nu}^{N\pm1,f}

P_{\mu\nu}^{N,i},
]

and from this obtain changes of:

* atomic charges;
* bond orders;
* local aromaticity;
* proton donor/acceptor polarization;
* electrostatic potential at the proton.

---

# 10. Resonant excitation versus ordinary IETS

These should be distinguished.

## Resonant transient charging

The molecule occupies a real (N\pm1) state for a finite time. Franck–Condon factors, Huang–Rhys factors, charged-state geometry and proton-transfer dynamics are central.

Here Dyson orbitals are directly useful:

[
\text{tip}
\rightarrow
\text{Dyson coupling}
\rightarrow
N\pm1\text{ state}
\rightarrow
\text{nuclear dynamics}.
]

## Off-resonant IETS

The ionic states are only virtual intermediates. The amplitude contains a sum over virtual (N\pm1) states:

[
\mathcal A_k
\sim
\sum_f
\frac{
T_{{\rm out},f},
g_{f,k},
T_{f,{\rm in}}
}
{\text{energy denominators}},
]

where (T_{\rm in/out}) are Dyson-orbital tunnelling amplitudes and (g_{f,k}) describes electron–vibration coupling.

Here the Dyson orbital remains relevant, but it is not sufficient. You also need:

* virtual-state energies;
* electronic broadening;
* electron–vibration matrix elements;
* interference among ionic states.

---

# 11. When Huang–Rhys factors are not enough

For ordinary skeletal modes of a rigid polyaromatic molecule, the harmonic and Duschinsky treatment is reasonable:

[
\mathbf Q_f=\mathbf J\mathbf Q_i+\mathbf K.
]

Here (\mathbf J) rotates the normal-mode basis and (\mathbf K) is the equilibrium displacement.

But proton transfer is usually a large-amplitude, anharmonic coordinate. A local harmonic Hessian may produce a large formal Huang–Rhys factor without correctly representing:

* a double-well potential;
* barrier crossing;
* proton tunnelling;
* tautomerization;
* coupling to donor–acceptor distance;
* coupling to substrate polarization.

For the proton coordinate, I would instead construct at least a two-dimensional surface,

[
E_f^{N\pm1}(q_{\rm H},R_{\rm DA}),
]

where (R_{\rm DA}) is the donor–acceptor distance, and solve the proton vibrational problem explicitly. The remaining approximately harmonic skeletal modes can still be treated by Huang–Rhys/Duschinsky theory.

---

# 12. A practical DFTB+ workflow

## Stage A: electronic channels at the vertical geometry

At the neutral equilibrium geometry (\mathbf R_N), calculate:

[
E_i^N(\mathbf R_N),
\qquad
E_f^{N+1}(\mathbf R_N),
\qquad
E_g^{N-1}(\mathbf R_N)
]

for relevant spin and electronic states.

For each channel calculate:

* vertical attachment/ionization energy;
* approximate Dyson orbital;
* Dyson norm;
* state-specific difference density;
* atomic charge and bond-order differences;
* initial ionic force.

For several nearly degenerate (\pi) states, ground-state SCC-DFTB alone is not sufficient to distinguish all channels. You would need constrained occupations, TD-DFTB within the charged sector, REKS-type treatment, or preferably your proposed small DFTB active-space CI.

DFTB+ supports open-shell excited-state calculations and state-specific excited-state gradients, although its minimal basis restricts it mainly to valence excitations. ([DFTB+ Recipes][7])

## Stage B: approximate Dyson orbitals

Three accuracy levels are natural:

1. **Frozen approximation**

   [
   d^+\approx\phi_{\rm LUMO},
   \qquad
   d^-\approx\phi_{\rm HOMO}.
   ]

2. **Relaxed determinant Dyson**

   Run separate SCC calculations for (N) and (N\pm1), then use the AO overlap matrix and occupied-orbital cross-overlap SVD.

3. **DFTB active-space CI Dyson**

   Obtain several correlated (N), (N+1) and (N-1) states and contract

   [
   d_{fi,p}^+
   ==========

   \langle\Psi_i^N|
   \hat a_p
   |\Psi_f^{N+1}\rangle.
   ]

For polyaromatic heterocycles with competing valence-bond structures, I expect level 3 eventually to be the scientifically most interesting.

## Stage C: ionic PES and proton transfer

For each significantly populated ionic state (f), calculate

[
E_f^{N\pm1}(q_{\rm H},R_{\rm DA},Q_{\rm ring},\ldots).
]

At minimum, use constrained optimizations along (q_{\rm H}). Better, include donor–acceptor compression and one or two important ring-deformation modes.

## Stage D: vibrational reorganization

Optimize the neutral and charged minima and calculate both Hessians. DFTB+ can generate finite-difference Hessians and mass-weighted normal modes. ([DFTB+ Recipes][8])

Then compute:

* Duschinsky matrix (\mathbf J);
* displacement (\mathbf K);
* mode-resolved (S_k);
* total reorganization energy;
* multidimensional Franck–Condon factors.

## Stage E: geometry-dependent tunnel coupling

Sample

[
d_f^\pm(\mathbf r;q_{\rm H})
]

at several points along the proton coordinate and evaluate

[
\Gamma_f(q_{\rm H})
\propto
\left|
\int t_{\rm tip}(\mathbf r)
d_f^\pm(\mathbf r;q_{\rm H}),d\mathbf r
\right|^2.
]

This gives a coordinate-dependent ionic lifetime, which may be crucial for predicting whether the proton transfers before the electron leaves.

## Stage F: kinetic network

Construct a master equation over states such as

[
|A,N,\nu\rangle,\quad
|A,N+1,\mu\rangle,\quad
|B,N+1,\mu\rangle,\quad
|B,N,\nu\rangle.
]

Include:

* Dyson-weighted injection and discharge;
* Franck–Condon vibronic branching;
* proton-transfer rates;
* vibrational relaxation;
* substrate-induced electronic relaxation;
* backward proton transfer.

This will let you predict not only spectra but also reaction yield per injected electron.

---

# My recommendation for your project

I would calculate and keep three separate real-space quantities for every relevant ionic state (f):

[
\boxed{d_f^\pm(\mathbf r)}
]

to determine the STM tunnelling amplitude and spatial selectivity;

[
\boxed{
\Delta\rho_f^\pm(\mathbf r)
===========================

\rho_f^{N\pm1}(\mathbf r)-\rho_i^N(\mathbf r)
}
]

to understand charge redistribution, bond-order changes and proton forces;

[
\boxed{
\Delta\mathbf F_f^\pm
=====================

-\nabla_{\mathbf R}
(E_f^{N\pm1}-E_i^N)
}
]

to identify the vibrational and reaction coordinates activated immediately after charging.

For a single weakly correlated attachment state, all three may correlate visually with the LUMO. For a heterocyclic polyaromatic molecule with proton transfer and several close ionic states, they may differ strongly—and that difference is likely where the interesting STM-induced chemistry resides.

[1]: https://pubs.aip.org/aip/jcp/article/153/7/070902/1064699/Dyson-orbital-concepts-for-description-of?utm_source=chatgpt.com "Dyson-orbital concepts for description of electrons in molecules"
[2]: https://pubs.acs.org/doi/10.1021/ja9924039?utm_source=chatgpt.com "The Fukui Function and Chemical Hardness Revisited"
[3]: https://www.osti.gov/biblio/121617?utm_source=chatgpt.com "Analysis of electronic transitions as the difference ..."
[4]: https://pubs.rsc.org/en/content/articlepdf/2024/cp/d3cp05559k?utm_source=chatgpt.com "On the multiphoton ionisation photoelectron spectra of ..."
[5]: https://link.aps.org/doi/10.1103/PhysRevB.74.205438?utm_source=chatgpt.com "Theory of the Franck-Condon blockade regime | Phys. Rev. B"
[6]: https://pubs.acs.org/doi/10.1021/acs.jpcb.1c06949?utm_source=chatgpt.com "Electron-Induced Proton Transfer - ACS Publications"
[7]: https://dftbplus-recipes.readthedocs.io/en/latest/linresp/introduction.html?utm_source=chatgpt.com "Introduction — DFTB+ Recipes"
[8]: https://dftbplus-recipes.readthedocs.io/en/latest/moleculardynamics/startinggeometry.html?utm_source=chatgpt.com "Preparing for an MD calculation — DFTB+ Recipes"
