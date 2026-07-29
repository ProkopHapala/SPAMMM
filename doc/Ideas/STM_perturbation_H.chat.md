# USER

I'm thinking to improve BR-STM functionality 

The true tunneling matrix element is 
M =  <phi_tip|H'|psi_sample>

question is what exacly is H' ? 
Mostly it is potntial of atoms of sample acting on tip wavefunction and vice versa (the oribtals are already solution of Laplace operator with original potential so that is already in H0)

I compute the fermi golden rule matrix element just as overlap M = <phi_tip|psi_sample>
resuting STM images have to high instentiy ourside molecule and low over the molecule coe, which is major discrepancy with experiment, this I can understand as I approximate H' =1 (constant in space), but in reality perhaps the H' is higher over the molecule as there is strongler potential of the molecule nuclei acting on tip wavefunction. 

Another view of this is thinking about breakdonw  of tunneling barries. Now question how to easily include this into my picture and into my kernel where I basically compute just overlap. I have coefficients of moleculer orbitals and from DFTB and I have also the hamiltonian. But I do not have common hamiltonian for tip and sample. I can compute it form slater coster on the fly in my kernel perhaps. 

---

# ChatGPT 5.6 sol

## What exactly is (H')?

For an electron initially in a sample eigenstate,

[
\hat H_S|\psi_s\rangle=\varepsilon_s|\psi_s\rangle ,
]

the transfer-Hamiltonian partition is

[
\hat H=\hat H_S+\hat H',
\qquad
\hat H'=\hat H-\hat H_S .
]

Therefore,

[
M_{ts}
======

# \langle \phi_t|\hat H-\hat H_S|\psi_s\rangle

## \langle \phi_t|\hat H|\psi_s\rangle

\varepsilon_s\langle\phi_t|\psi_s\rangle .
]

Defining

[
H_{ts}=\langle\phi_t|\hat H|\psi_s\rangle,
\qquad
S_{ts}=\langle\phi_t|\psi_s\rangle,
]

gives the crucial result

[
\boxed{M_{ts}=H_{ts}-\varepsilon_s S_{ts}.}
]

Equivalently, starting from the tip Hamiltonian gives

[
M_{ts}=H_{ts}-\varepsilon_tS_{ts}.
]

For elastic tunneling, (\varepsilon_t\simeq\varepsilon_s=E), so

[
\boxed{M_{ts}(E)=H_{ts}-E S_{ts}.}
]

This is the nonorthogonal-basis transfer coupling. In Green-function notation the off-diagonal block of the resolvent is

[
A_{TS}(z)=zS_{TS}-H_{TS},
]

or, with the opposite sign convention,

[
\tau_{TS}(z)=H_{TS}-zS_{TS}.
]

Bardeen’s derivation starts from precisely (\langle t|H-H_S|s\rangle), then transforms it into the familiar surface-flux integral in the vacuum. 

This also gives an important gauge test. Under a shift of energy origin,

[
H\rightarrow H+C S,\qquad E\rightarrow E+C,
]

the combination (H-ES) remains invariant. (H) alone does not, and overlap alone clearly does not have the dimensions of a tunneling matrix element.

---

## Is (H') the potential of the sample nuclei?

Only in one particular, asymmetric partition.

If

[
H=T+V_S+V_T
]

and the initial unperturbed Hamiltonian is (H_S=T+V_S), then

[
H'=V_T,
]

so it is actually the **tip potential acting on the sample state**. Starting from the tip side instead gives (H-H_T=V_S), the sample potential acting on the tip state.

These two forms differ by

[
(\varepsilon_t-\varepsilon_s)S_{ts},
]

and therefore become equivalent for energy-conserving tunneling. Their symmetric, partition-independent content is the Bardeen surface expression

[
M_{ts}
======

-\frac{\hbar^2}{2m}
\int_{\Sigma}
\left[
\phi_t^* \nabla\psi_s
---------------------

\psi_s\nabla\phi_t^*
\right]\cdot d\mathbf S .
]

Thus your intuition that a molecular nuclear potential could increase coupling near the molecular core is not entirely meaningless in a volume-integral representation, but it is **not correct to introduce the nuclear potential as a local multiplicative mask**. An equivalent derivation can move that potential to the opposite electrode, and both reduce to a flux-like expression in the vacuum. 

The physical matrix element is closer to a Wronskian or probability-current flux than to a volume overlap.

---

# What you should implement with DFTB

For tip and sample MOs,

[
|\phi_t\rangle=\sum_{\mu\in T}c^t_\mu|\chi_\mu\rangle,
\qquad
|\psi_s\rangle=\sum_{\nu\in S}c^s_\nu|\chi_\nu\rangle,
]

calculate cross-interface AO matrices

[
H^{TS}_{\mu\nu}
===============

\langle\chi_\mu^T|\hat H|\chi_\nu^S\rangle,
]

[
S^{TS}_{\mu\nu}
===============

\langle\chi_\mu^T|\chi_\nu^S\rangle.
]

Then

[
\boxed{
M_{ts}(E)
=========

\sum_{\mu\nu}
(c^t_\mu)^*
\left(
H^{TS}*{\mu\nu}-E S^{TS}*{\mu\nu}
\right)
c^s_\nu .
}
]

You do **not** need to solve the combined tip–sample system. You only need the cross blocks (H_{TS}) and (S_{TS}).

DFTB Slater–Koster files already contain radial tables for both Hamiltonian and overlap channels, including (ss\sigma), (sp\sigma), (pp\sigma), (pp\pi), and higher-angular-momentum channels where available. 

For an (s,p) basis, form the energy-dependent radial channels

[
\begin{aligned}
\tau_{ss\sigma}(r,E)&=H_{ss\sigma}(r)-E S_{ss\sigma}(r),\
\tau_{sp\sigma}(r,E)&=H_{sp\sigma}(r)-E S_{sp\sigma}(r),\
\tau_{pp\sigma}(r,E)&=H_{pp\sigma}(r)-E S_{pp\sigma}(r),\
\tau_{pp\pi}(r,E)&=H_{pp\pi}(r)-E S_{pp\pi}(r).
\end{aligned}
]

Then use the normal Slater–Koster angular transformation.

For example,

[
\tau_{p_i p_j}
==============

\tau_{pp\pi}\delta_{ij}
+
(\tau_{pp\sigma}-\tau_{pp\pi})
\hat r_i\hat r_j .
]

This is almost exactly the structure already present in your kernel.

---

## Minimal change to your OpenCL kernel

Your current `stm_gf_dyson_2mol_mo_scan()` does not actually compute a plain overlap. It computes an ad hoc hopping with

```c
f          = exp(-beta*(r-r0));
Vss        = f;
Vsp        = f;
Vpp_sigma  = f;
Vpp_pi     = 0.2f*f;
```

Replace these invented radial functions with DFTB (H) and (S) tables.

For direct, real-energy Fermi-golden-rule evaluation:

```c
float4 H4 = interpolate_H_SK(pair_type, r);
// H4 = (Hss_sigma, Hsp_sigma, Hpp_sigma, Hpp_pi)

float4 S4 = interpolate_S_SK(pair_type, r);

float4 T4 = H4 - E * S4;
// T4 contains tau = H - E*S
```

Then apply the same angular transformation you currently use for `V`.

For a Green function evaluated at

[
z=E+i\eta,
]

the coupling becomes complex:

```c
inline float2 make_tau(float H, float S, float2 z)
{
    // tau = H - z*S
    return (float2)(H - z.x*S, -z.y*S);
}
```

Because the Slater–Koster transformation is linear, it is cheapest to form the four (\tau) channels first and then perform the angular transformation.

For a single energy per kernel launch, I would precompute

[
\tau_\alpha(r,E)=H_\alpha(r)-E S_\alpha(r)
]

on the CPU in double precision and upload only the resulting four-channel table. That gives you:

* one radial interpolation rather than two;
* no (H-ES) cancellation in GPU single precision;
* exactly the same GPU cost as your present artificial exponential model.

For multiple energies in one launch, load both `float4 H4` and `float4 S4` and combine them on the GPU.

---

# How this applies to each kernel in your file

### `response_amplitude_exp`

Your own comment at lines 163–164 already contains the correct structure:

```c
// a = z*S - H; with S=0 for tunneling => a = -H
```

The problematic assumption is precisely “(S=0) for tunneling.” Spatially separated tip and sample subspaces have small overlap, but it is not identically zero, and (ES) can be comparable to (H) in the asymptotic region.

Use

[
a_{TS}=zS_{TS}-H_{TS}
]

instead.

### `solve_stm_dyson_wg`

The off-diagonal block should be

[
A_{TS}=zS_{TS}-H_{TS},
]

or equivalently use

[
\tau_{TS}=H_{TS}-zS_{TS}
]

consistently throughout the self-energy construction. Since the product enters quadratically in the Schur complement, the overall sign often cancels, but the convention must be consistent when constructing source amplitudes and interference between paths.

### `stm_gf_dyson_2mol_mo_scan`

Replace

[
u_T H_{\mathrm{hop}}v_S
]

by

[
u_T\left(H_{TS}-zS_{TS}\right)v_S.
]

However, note the conceptual distinction:

[
c_t^\dagger(H_{TS}-ES_{TS})c_s
]

is the bare Fermi-golden-rule/Bardeen matrix element between two isolated states.

Your expression

[
c_t^\dagger G_T
(H_{TS}-zS_{TS})
G_S c_s
]

is a Green-function-dressed response amplitude. It may be useful, but it is not simply the bare Bardeen matrix element. I would first validate the direct MO–MO expression before adding the two Green functions.

---

# Why this might fix the excessive intensity outside the molecule

Overlap has two undesirable properties here.

First, it integrates over an entire volume:

[
S_{ts}=\int \phi_t^*(\mathbf r)\psi_s(\mathbf r),d^3r .
]

A large volume containing two weak tails can contribute substantially. That can generate a broad halo outside the molecule.

Second, it gives every part of the overlap the same weight. In contrast,

[
H_{ts}-ES_{ts}
]

contains kinetic and potential contributions with significant cancellation. This cancellation is not an accidental numerical nuisance—it removes the part of one state that merely resembles the other because the basis sets are nonorthogonal.

Therefore, (H-ES) can substantially suppress diffuse, slowly varying overlap while retaining regions carrying actual flux between the two subsystems. Whether it increases the molecular centre specifically is system-dependent, but it is the first correction I would test.

I would render three diagnostic maps:

[
\begin{aligned}
I_S(\mathbf R)&=\left|c_t^\dagger S_{TS}(\mathbf R)c_s\right|^2,\
I_H(\mathbf R)&=\left|c_t^\dagger H_{TS}(\mathbf R)c_s\right|^2,\
I_\tau(\mathbf R)&=\left|c_t^\dagger[H_{TS}(\mathbf R)-ES_{TS}(\mathbf R)]c_s\right|^2.
\end{aligned}
]

Seeing the separate (H) and (ES) contributions will tell you immediately whether the corrected central contrast comes from physically meaningful destructive interference.

---

# Important problems already visible in the kernel

Before attributing the discrepancy entirely to (H'), I would correct several issues.

## 1. Inconsistent orbital ordering

`response_amplitude_exp()` uses

```c
[s, px, py, pz]
```

at lines 163–169, while the later kernels explicitly use

```c
[px, py, pz, s]
```

at lines 399 and 590.

Unless the host performs different remapping for the first kernel, this is a serious inconsistency and can completely change spatial contrast.

## 2. The (sp) and (ps) signs need checking

Your code uses the same directional sign for

```c
tip s  - sample p
tip p  - sample s
```

but an (sp\sigma) matrix element changes sign when the direction or the order of the (s) and (p) centres is reversed, because the (p) orbital is odd.

The exact branch sign depends on whether your direction vector points tip-to-sample or sample-to-tip and on the DFTB SK convention. Do not simply use identical signs for `Vsp` and `Vps`. This can produce very strong erroneous constructive or destructive interference over the molecular centre.

## 3. All radial channels are currently nearly identical

You currently impose

[
V_{ss}=V_{sp}=V_{pp\sigma}=f,\qquad V_{pp\pi}=0.2f.
]

Real (ss), (sp), (pp\sigma), and (pp\pi) matrix elements have different magnitudes, signs, nodes, and decay rates. For bond-resolved contrast, those differences are not a minor correction; they determine the interference pattern.

## 4. The Dyson kernel drops atoms exactly where the molecule is densest

`solve_stm_dyson_wg()` keeps only eight active sample atoms. More importantly, the eight atoms are selected by parallel `atomic_inc`, so when more than eight atoms satisfy the cutoff the selected subset is not geometrically well defined.

Over the molecular centre there will usually be more atoms within the cutoff than outside the molecule. Silently dropping the remainder can therefore produce exactly the qualitative error you describe: relative suppression over the centre and exaggerated intensity around the perimeter.

At minimum, select the nearest eight deterministically. Better, remove the truncation or process overflow separately.

## 5. `I=|x|^2` is not a complete NEGF current

In the Dyson solver you use

[
I=\sum_i|x_i|^2,
]

equivalent to setting the drain broadening matrix to identity. A physical transmission is

[
T(E)=\operatorname{Tr}
\left[
\Gamma_TG^r\Gamma_SG^a
\right].
]

This may not be responsible for the lateral halo in the weak-coupling kernel, but it makes quantitative interpretation of the Dyson result difficult.

---

# The better long-term reference: direct Bardeen surface integral

You do not actually need a common Hamiltonian at all if you calculate the surface expression.

Choose a plane (z=z_0) in the vacuum and evaluate

[
\psi_s(x,y,z_0),\qquad
\partial_z\psi_s(x,y,z_0),
]

and similarly for the tip. For a lateral tip displacement (\mathbf R),

[
M(\mathbf R)=
-\frac{\hbar^2}{2m}
\int d^2\rho
\left[
\phi_t^*(\boldsymbol\rho-\mathbf R)
\partial_z\psi_s(\boldsymbol\rho)
---------------------------------

\psi_s(\boldsymbol\rho)
\partial_z\phi_t^*(\boldsymbol\rho-\mathbf R)
\right].
]

Both terms are two-dimensional cross-correlations. Therefore, an entire STM image can be computed with two-dimensional FFTs:

[
M(\mathbf R)
\sim
\mathcal F^{-1}
\left[
\Phi_t^*(\mathbf k),
\mathcal F(\partial_z\psi_s)
----------------------------

\mathcal F(\partial_z\phi_t)^*,
\Psi_s(\mathbf k)
\right].
]

This gives you:

* the proper Bardeen matrix element;
* all scan positions simultaneously;
* no combined tip–sample SCF calculation;
* a clean reference against which to validate the fast Slater–Koster kernel.

Chen’s derivative rule is a local approximation to this surface integral: an (s)-type apex probes the sample wavefunction itself, while (p_x,p_y,p_z) apex components probe corresponding spatial derivatives. Interference between tip (s) and (p_z) channels alone can substantially change STM contrast. 

For a CO-functionalized or otherwise non-(s)-like tip, this tip-orbital interference may be as important as replacing overlap by (H-ES).

---

# About a local barrier correction

After implementing (H-ES), you can include the local barrier approximately through

[
M(\mathbf R)
\rightarrow
M(\mathbf R)
\exp\left[
-\int_{\text{path}}
\kappa(\mathbf r,E),ds
\right],
]

with

[
\kappa(\mathbf r,E)
===================

\frac{\sqrt{2m[U_{\mathrm{vac}}(\mathbf r)-E]}}{\hbar}.
]

But (U_{\mathrm{vac}}) should be a smooth effective vacuum barrier: electrostatic potential, work-function variation, applied bias and possibly image-potential effects. It should **not** be the bare sum of nuclear Coulomb potentials. The latter would introduce singular, strongly partition-dependent weighting and would double-count much of the chemistry already present in the orbitals and Hamiltonian.

This WKB factor is useful for modelling local work-function and molecular-dipole effects, but it should be a second correction, not the replacement for the transfer matrix element.

---

## Recommended implementation sequence

1. Fix the orbital ordering and (sp/ps) sign conventions.
2. Compute real DFTB (H_{TS}) and (S_{TS}) channels.
3. Replace the artificial hopping by
   [
   \tau(E)=H_{TS}-ES_{TS}.
   ]
4. Initially calculate the direct MO–MO matrix element without (G_T) and (G_S).
5. Remove or diagnose the eight-atom truncation.
6. Implement a small Bardeen-plane reference calculation for selected geometries.
7. Only then add local WKB barrier modulation or the full Dyson/NEGF dressing.

My strongest expectation is that the combination of **proper (H-ES), correct (sp) parity, and removing central-region truncation** will change the image much more than multiplying your current overlap by an approximate molecular potential.


---

# USER

OK, please write the new kernels direcly including detailed physical derivtion and exmaplantion with equations in the doc-strings, then tell me what exactly I need to wire here from DFTB (I guess I need radial SK tables for both S and H right?), but only non-SCF (SCC) I do not want to consider charge polarization self-cosistent response. This should be super fost, Also I do not want to diagonalize anything. Now focus just on fist order Fermi-golden rule STM, ignoring the Dyson or Green's function version.

ad inconsistent ordering [s,x,y,z] vs [x,y,z,s] yes, I know about this, yes I solve this on the host reordering, do not bother with that, use same ordering as is already used in the existing kernel. But note this isue clearly so we do not forget (comment it in the code).

However what is very important - the original DFTB+ basis function mio-1-1 and 3ob are very short rnage and are useless for STM, therefore we replac them by longer exponential slater orbitals exp(-b|r|). This means that SK tables adial part will change both Hmunu and Smunu

how to properly deal with that? 

 Note - even though this is inconsisten wrt exsting moleculer orbitals coefficients  (as modified basis would produce different orbitals coefs),  I do not care about that now, the exitsting coefs encode primarly nodal structure which is preserved, and the radial decay must change otherwise we would not see anything in STM. I know this is dirty trick but I'm OK with that, so you should be to, deal with that, nd do not overcomplicate it.  

---

# Chat GPT 5.6 sol

I created a clean first-order Fermi-golden-rule implementation, separate from the Dyson/Green-function code:

* [OpenCL FGR STM kernels](sandbox:/mnt/data/LCAO_STM_FGR.cl)
* [Host wiring and table-generation notes](sandbox:/mnt/data/LCAO_STM_FGR_WIRING.md)

The file was syntax-checked as OpenCL C 1.2. The atom-pair contraction was also checked against explicit (4\times4) Slater–Koster matrices.

## Kernels included

### `build_stm_transfer_sk_tables`

Precomputes

[
\tau_\gamma(R,E)=H_\gamma(R)-E,S_\gamma(R)
]

for all radial channels and pair types.

Run this only when the selected tunnelling energy changes. It is a tiny operation compared with producing an STM image.

### `stm_fgr_sk_tau_scan_real`

This is the recommended production kernel for ordinary real molecular orbitals:

[
M_{ts}(\mathbf R,E)
===================

c_t^{T},
\tau_{TS}(\mathbf R,E),
c_s.
]

It outputs

```text
out_M_M2[ip].x = M
out_M_M2[ip].y = M*M
out_M_M2[ip].z = number of atom pairs used
```

This should be substantially faster than the old kernel because:

* coefficients are `float`, not `float2`;
* geometry and SK interpolation are performed once per atom pair;
* the entire (4\times4) (s,p) block is contracted analytically;
* only the precombined (\tau=H-ES) table is read.

### `stm_fgr_sk_tau_scan`

The same calculation for complex coefficients. You only need this for genuinely complex states, such as non-(\Gamma) Bloch states, spin-orbit states, or some externally phase-dressed representation.

### `stm_fgr_sk_hs_scan`

A slower diagnostic version that reads (H) and (S) separately and constructs (H-ES) inside the scan loop. This is useful for validating energy alignment and observing cancellation between (H) and (ES), but it should not be the production path.

---

# Exactly what must come from DFTB

For one tip state and one sample state you need:

1. The tip MO coefficients (c_t).
2. The sample MO coefficients (c_s).
3. Their energies (\varepsilon_t,\varepsilon_s).
4. Atom positions and atom types.

Pack coefficients atom-major as your existing kernel already expects:

[
[;p_x,;p_y,;p_z,;s;].
]

Hydrogen is therefore

[
[0,0,0,c_s].
]

The ordering inconsistency with other code using ([s,p_x,p_y,p_z]) is explicitly documented near the top of the new file, but the kernel assumes that your existing host-side remapping remains in place.

You do **not** pass:

* the isolated molecular Hamiltonian matrices;
* the isolated molecular overlap matrices;
* Green functions;
* density matrices;
* SCC charges;
* Hubbard (U) values;
* repulsive DFTB potentials;
* any newly diagonalized states.

The existing DFTB Hamiltonian and overlap were used upstream to produce the MO coefficients. They are not involved in the scan.

---

# Yes: you need radial (H) and (S) SK tables—but not the mio/3ob tables

For every ordered tip-type/sample-type pair, supply five signed radial channels:

[
\begin{aligned}
H_{ss\sigma}(R),\quad
H_{sp\sigma}(R),\quad
H_{ps\sigma}(R),\quad
H_{pp\sigma}(R),\quad
H_{pp\pi}(R),\
S_{ss\sigma}(R),\quad
S_{sp\sigma}(R),\quad
S_{ps\sigma}(R),\quad
S_{pp\sigma}(R),\quad
S_{pp\pi}(R).
\end{aligned}
]

The official DFTB SK format likewise stores the two-centre Hamiltonian and overlap channels separately as radial tables. In ordinary DFTB0, the Hamiltonian integrals are formed from the kinetic operator and frozen neutral-atom effective potentials, while the overlap is computed directly from the atomic orbitals. ([Density Functional Tight Binding][1])

In the kernel, they are packed as:

```c
float4 H4 = (Hss, Hsp, Hps, Hpp_sigma);
float  Hpp_pi;

float4 S4 = (Sss, Ssp, Sps, Spp_sigma);
float  Spp_pi;
```

All pair tables currently share one uniform radial grid:

[
R_i=R_{\min}+i,\Delta R.
]

The flattened index is

```c
index = pair_index*n_r + radial_index;
```

---

# The important point: create a separate tunnelling basis

You should not try to use the original mio-1-1 or 3ob radial SK tables. Their (H) and (S) integrals were generated using their original strongly confined pseudoatomic orbitals. Extending only the displayed wavefunction while retaining those tables would make the three objects

[
c,\qquad H,\qquad S
]

refer to incompatible bases.

Instead define a deliberately extended **tunnelling basis**

[
\widetilde\chi_{A,nlm}(\mathbf r)
=================================

N_{A,nl},
r^{n-1}e^{-\zeta_{A,l}r}
Y_{lm}(\widehat{\mathbf r}).
]

Then retain the original DFTB coefficients but reinterpret the reconstructed vacuum state as

[
\widetilde\psi_s(\mathbf r)
===========================

\sum_\nu c_{s,\nu}\widetilde\chi_\nu(\mathbf r).
]

This is exactly your intended dirty approximation:

* (c_\nu) preserves molecular phases, signs and nodes;
* (\widetilde\chi_\nu) supplies a physically useful vacuum tail;
* you do not rediagonalize;
* you do not modify the molecular electronic structure internally.

The cross-electrode matrices must then be generated using the modified functions:

[
\widetilde S_{\mu\nu}^{AB}(R)
=============================

\left\langle
\widetilde\chi_{A\mu}
\middle|
\widetilde\chi_{B\nu}
\right\rangle,
]

and, in the frozen non-SCC approximation,

[
\widetilde H_{\mu\nu}^{0,AB}(R)
===============================

\left\langle
\widetilde\chi_{A\mu}
\middle|
-\frac{1}{2}\nabla^2
+v_A^0+v_B^0
\middle|
\widetilde\chi_{B\nu}
\right\rangle.
]

Here (v_A^0,v_B^0) are fixed neutral-atom reference potentials. No charges are iterated and no tip-induced response is computed.

The actual scan table is

[
\boxed{
\tau_{\mu\nu}(R,E)
==================

## \widetilde H_{\mu\nu}^{0}(R)

E,\widetilde S_{\mu\nu}(R)
}.
]

This is conceptually a DFTB0-like transfer block generated with your custom tunnelling orbitals, not a new DFTB calculation of the molecules.

---

# Why (sp) and (ps) are stored separately

The kernel defines the directed interatomic axis as

[
\mathbf u
=========

\frac{\mathbf R_{\mathrm{sample}}-\mathbf R_{\mathrm{tip}}}{R}.
]

It then defines

[
X_{sp\sigma}
============

\langle s_T|X|p_{S,\mathbf u}\rangle,
]

and

[
X_{ps\sigma}
============

\langle p_{T,\mathbf u}|X|s_S\rangle.
]

Their signs are stored directly in the tables.

For identical real orbitals with this convention, one normally gets approximately

[
X_{ps\sigma}=-X_{sp\sigma},
]

because the two (p)-orbital tails face the opposite way. The kernel therefore does **not** insert another hard-coded minus sign. This removes the ambiguity present in the previous implementation.

The ordered nature also matters for heteroatomic pairs. A C-tip/O-sample table is not necessarily interchangeable with O-tip/C-sample.

---

# The atom-pair contraction used by the kernel

For one atom pair, write

[
c_{Tp}=(c_{Tx},c_{Ty},c_{Tz}),
\qquad
c_{Sp}=(c_{Sx},c_{Sy},c_{Sz}).
]

The complete (4\times4) contraction is

[
\begin{aligned}
M_{AB}={}&
c_{Ts}^{*}\tau_{ss}c_{Ss}\
&+
c_{Ts}^{*}\tau_{sp}
(\mathbf u\cdot c_{Sp})\
&+
(c_{Tp}^{*}\cdot\mathbf u)
\tau_{ps}c_{Ss}\
&+
\tau_{pp\pi}
(c_{Tp}^{*}\cdot c_{Sp})\
&+
(\tau_{pp\sigma}-\tau_{pp\pi})
(c_{Tp}^{*}\cdot\mathbf u)
(\mathbf u\cdot c_{Sp}).
\end{aligned}
]

Thus the kernel does not loop through 16 orbital combinations. It performs one interpolated five-channel lookup and evaluates this expression.

---

# Three practical ways to produce your long-tail tables

## 1. Recommended: frozen pair Hamiltonian

Generate the five (S) channels from your long STOs and the five (H^0) channels using fixed neutral-atom effective potentials.

This is the cleanest version of your approximation. It is fully non-SCC and all generation happens offline. Runtime remains only table interpolation.

The official `skprogs` project provides programs for SK generation, although adapting it to externally imposed pure exponential tails may be less convenient than using your own two-centre numerical integrator. ([GitHub][2])

You can put the atoms on the (z)-axis and numerically evaluate the five canonical orientations:

[
ss,\quad s,p_z,\quad p_z,s,\quad p_zp_z,\quad p_xp_x.
]

Because the integrals are axially symmetric, they can be reduced to cylindrical ((\rho,z)) quadrature rather than a general 3D grid.

## 2. Very simple first implementation: extended Hückel closure

Generate only the overlap tables accurately from your extended STOs and approximate

[
H_{\mu\nu}(R)
=============

K_{\mu\nu}
\frac{\epsilon_\mu+\epsilon_\nu}{2}
S_{\mu\nu}(R).
]

Then

[
\tau_{\mu\nu}(R,E)
==================

\left[
K_{\mu\nu}
\frac{\epsilon_\mu+\epsilon_\nu}{2}
-E
\right]
S_{\mu\nu}(R).
]

Use the DFTB atomic (s,p) onsite energies as (\epsilon_\mu), and initially use perhaps one (K) per channel or pair.

This is compatible with the level of approximation you described. Its weakness is that (H) and (S) have identical radial shapes, so it may not capture all of the desired central enhancement.

## 3. Fit (\tau) directly

For STM imaging, only

[
\tau=H-ES
]

enters. You can therefore directly define or fit

[
\tau_\gamma(R)
==============

A_\gamma P_\gamma(R)e^{-\kappa_\gamma R}
]

for the five channels.

Then upload those tables directly and skip both (H) and (S) during scanning.

This is the fastest and avoids uncertainty about energy-zero alignment. However, it prevents you from separately diagnosing (H) and (ES). I recommend first implementing separate (H,S), looking at their cancellation, and later deciding whether direct (\tau) parametrization is sufficient.

---

# Energy alignment is important

The combination

[
H-ES
]

is invariant under

[
H\rightarrow H+C S,
\qquad
E\rightarrow E+C.
]

But this only works when (H) and (E) use the same energy zero.

Therefore:

* If your custom Hamiltonian uses vacuum as zero, use the electron energy relative to vacuum.
* If it uses a DFTB/atomic energy convention, align the MO energy to that convention.
* Do not blindly combine a vacuum-referenced custom (H) with an unaligned DFTB eigenvalue.

For a nearly elastic pair, you can use

[
E_{\mathrm{tunnel}}=\varepsilon_s
]

following the sample-side partition, or symmetrically

[
E_{\mathrm{tunnel}}
===================

\frac{\varepsilon_t+\varepsilon_s}{2}.
]

For your first contrast tests, (E) can also be treated as a controlled physical calibration parameter, provided the same convention is used for all pixels.

---

# Exact host call sequence

```text
1. Read c_tip, c_smp and their energies from DFTB.
2. Reorder each atom block to [px,py,pz,s].
3. Upload geometry, atom types and coefficient vectors.
4. Upload custom long-tail H and S SK tables.
5. Choose E_tunnel.
6. Run build_stm_transfer_sk_tables.
7. Run stm_fgr_sk_tau_scan_real.
8. Plot out_M_M2[:,1].
```

For a single state pair,

[
I(\mathbf R)\propto |M(\mathbf R)|^2.
]

For a bias-window image, the host can later sum

[
I(\mathbf R,V)
\propto
\sum_{t,s}
|M_{ts}(\mathbf R)|^2
,W_{ts}(V),
]

where (W_{ts}) contains occupation, energy broadening and density-of-states factors. No additional diagonalization is needed during scanning.

[1]: https://www.dftb.org/_downloads/85b02a0893bd3402438aec77de5bc1df/slakoformat.pdf?utm_source=chatgpt.com "Format of the v1.0 Slater-Koster files"
[2]: https://github.com/dftbplus/skprogs?utm_source=chatgpt.com "dftbplus/skprogs: Basic programs for generating Slater- ..."
