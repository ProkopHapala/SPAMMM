# Tau (τ) = H − E·S: The transfer matrix element in non-orthogonal basis

**Status:** Explanation of the FGR BR-STM method. See also [LCAO_STM_FGR_WIRING.md](LCAO_STM_FGR_WIRING.md) and [STM_perturbation_H.chat.md](STM_perturbation_H.chat.md).

## 1. The physical setup

We have two separated subsystems:
- **Tip** with Hamiltonian H_T, eigenstate |φ_t⟩, energy ε_t
- **Sample** with Hamiltonian H_S, eigenstate |ψ_s⟩, energy ε_s

The full Hamiltonian is H = H_T + H_S + H_int (or equivalently H = H_S + H' where H' = H − H_S).

## 2. Derivation of τ = H − E·S from Fermi Golden Rule

### 2.1 The transfer-Hamiltonian partition

The Fermi Golden Rule tunneling rate is:

```
W_{t←s} = (2π/ℏ) |M_{ts}|² δ(ε_t − ε_s)
```

The matrix element M_{ts} is defined as:

```
M_{ts} = ⟨φ_t| H' |ψ_s⟩ = ⟨φ_t| H − H_S |ψ_s⟩
```

### 2.2 Expanding in the non-orthogonal basis

Since |φ_t⟩ is an eigenstate of H_T and |ψ_s⟩ is an eigenstate of H_S, but these
eigenstates are **not orthogonal** to each other (they live on different atoms),
we have:

```
M_{ts} = ⟨φ_t| H |ψ_s⟩ − ⟨φ_t| H_S |ψ_s⟩
       = ⟨φ_t| H |ψ_s⟩ − ε_s ⟨φ_t| ψ_s⟩
       = H_{ts} − ε_s · S_{ts}
```

where:
- **H_{ts} = ⟨φ_t| H |ψ_s⟩** — the cross-electrode Hamiltonian matrix element
- **S_{ts} = ⟨φ_t| ψ_s⟩** — the cross-electrode overlap (NOT zero for non-orthogonal basis)
- **ε_s** — the sample eigenstate energy

### 2.3 Elastic tunneling simplification

For elastic tunneling (energy-conserving), ε_t ≈ ε_s = E, so:

```
M_{ts}(E) = H_{ts} − E · S_{ts} ≡ τ_{ts}(E)
```

This is the **τ matrix element**. It is the proper Fermi Golden Rule transfer
matrix element in a non-orthogonal basis.

### 2.4 Symmetry: tip-side derivation gives the same result

Starting from the tip side instead:

```
M_{ts} = ⟨φ_t| H − H_T |ψ_s⟩ = H_{ts} − ε_t · S_{ts}
```

For elastic tunneling (ε_t = ε_s = E), both derivations agree:

```
M_{ts}(E) = H_{ts} − E · S_{ts} = τ_{ts}(E)
```

This gauge invariance (independence of which electrode we partition from) is a
key physical consistency check. Under an energy shift H → H + C·S, E → E + C,
the combination H − E·S is invariant — **τ is gauge-invariant**, while H alone
or S alone is not.

## 3. Connection to Bardeen's surface integral

Bardeen's original derivation starts from the same ⟨t|H−H_S|s⟩ and transforms
it (via integration by parts / Green's identity) into a **surface flux integral**
in the vacuum barrier:

```
M_{ts} = −(ℏ²/2m) ∫_Σ [φ_t* ∇ψ_s − ψ_s ∇φ_t*] · dS
```

This is a Wronskian / probability-current expression evaluated on a surface Σ
in the vacuum between tip and sample. It is **equivalent** to τ = H − E·S but
expressed as a 2D flux rather than a 3D volume integral.

**Key insight:** τ is closer to a probability-current flux than to a volume
overlap. This is why it suppresses the diffuse halo that pure overlap S produces.

## 4. Why τ fixes the "excessive intensity outside molecule" problem

Pure overlap S_{ts} = ∫ φ_t*(r) ψ_s(r) d³r has two problems:

1. **Volume integration:** A large volume of two weak tails (far from both
   nuclei) contributes substantially → broad halo outside the molecule.

2. **Uniform weighting:** Every part of the overlap gets the same weight,
   including the diffuse, slowly-varying parts that carry no actual tunneling
   flux.

In contrast, τ = H − E·S contains kinetic and potential contributions with
**significant cancellation**. This cancellation is not a numerical nuisance —
it removes the part of one state that merely resembles the other because the
basis sets are non-orthogonal. The result:

- **Suppresses** diffuse, slowly-varying overlap (the halo)
- **Retains** regions carrying actual flux between the two subsystems
- Produces contrast that is more localized to the molecular bonds

## 5. Implementation in LCAO/STO basis

### 5.1 MO expansion

Tip and sample MOs are expanded in atomic orbitals:

```
|φ_t⟩ = Σ_{μ∈T} c^t_μ |χ_μ⟩
|ψ_s⟩ = Σ_{ν∈S} c^s_ν |χ_ν⟩
```

### 5.2 Cross-electrode AO matrices

We compute the cross-interface atomic-orbital matrix elements:

```
H^{TS}_{μν} = ⟨χ^T_μ| H |χ^S_ν⟩
S^{TS}_{μν} = ⟨χ^T_μ| χ^S_ν⟩
```

These are obtained from Slater-Koster two-center tables (same as DFTB), but
using **long-tail STO basis functions** (not the short-ranged mio/3ob orbitals).

### 5.3 The τ matrix element

```
M_{ts}(E) = Σ_{μν} (c^t_μ)* [H^{TS}_{μν} − E · S^{TS}_{μν}] c^s_ν
          = Σ_{μν} (c^t_μ)* τ^{TS}_{μν}(E) c^s_ν
```

### 5.4 Slater-Koster decomposition

For an {s,p} basis, form energy-dependent radial channels:

```
τ_{ssσ}(r,E) = H_{ssσ}(r) − E · S_{ssσ}(r)
τ_{spσ}(r,E) = H_{spσ}(r) − E · S_{spσ}(r)
τ_{ppσ}(r,E) = H_{ppσ}(r) − E · S_{ppσ}(r)
τ_{ppπ}(r,E) = H_{ppπ}(r) − E · S_{ppπ}(r)
```

Then apply the standard Slater-Koster angular transformation. For example:

```
τ_{p_i p_j} = τ_{ppπ} δ_{ij} + (τ_{ppσ} − τ_{ppπ}) r̂_i r̂_j
```

### 5.5 The three diagnostic quantities

```
I_S(R)  = |c_t† S_{TS}(R) c_s|²        — pure overlap (old BR-STM)
I_H(R)  = |c_t† H_{TS}(R) c_s|²        — Hamiltonian only
I_τ(R)  = |c_t† [H_{TS}(R) − E·S_{TS}(R)] c_s|²  — proper FGR transfer (τ)
```

I_τ is the physically correct Fermi Golden Rule matrix element. I_S is the
approximation used in the old BR-STM (overlap-only). I_H is an intermediate
diagnostic. Comparing all three reveals where the correction matters.

## 6. What is deliberately omitted (for now)

- **Lorentzian energy weighting:** The full tunneling current integrates
  |M|² over a Lorentzian window around E_F. We currently use a single E_tunnel
  per state (the state's own energy). This is the "no broadening" limit.
- **Bias integration:** No voltage-dependent DOS or occupation factors.
- **SCC charge response:** No self-consistent charge redistribution from the tip.
- **Green function dressing:** The bare M_{ts} is used, not c_t† G_T (H−ES) G_S c_s.
- **Bardeen surface integral:** The volume-integral form (τ = H − E·S) is used
  instead of the equivalent 2D surface flux form. Both are exact; the volume
  form is more convenient for the LCAO/SK framework.

## 7. Energy zero convention

H and E_tunnel must share the same energy zero. A constant shift applied
consistently to both H and E cancels in H − E·S (gauge invariance). In practice:

- Level B (current implementation): Extended Hückel closure
  H_γ(R) = K_γ · ½(ε_A + ε_B) · S_γ(R), with K=1.75 default.
  This makes H and S share the same radial shape, so I_S and I_τ look similar
  up to a global scale on C/H systems. The difference becomes significant when
  H and S have different radial shapes (Level A: numerical integration).

- For a state pair with nearly equal energies, use E_tunnel = E_sample
  (sample-partition derivation) or (E_tip + E_sample)/2 (symmetric choice).
