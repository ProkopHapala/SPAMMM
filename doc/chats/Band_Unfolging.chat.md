https://chatgpt.com/share/6ab0cd34-460c-83ed-a84f-98ee99d3cc83

---

# USER

I'm calculating some 2D band structures of graphene with DFTB. Do you think we can do some efficient unflving resp projection it to unicall to avoid having so much bands? 

look here we were doing this with Fereball now I want to do it with DFTB+ which should be conceptually the same
https://journals.aps.org/prb/supplemental/10.1103/PhysRevB.89.201412/supp_v8.pdf


I think in the end it was very easy we should do fourier trasnform, but the coefficients of molecular orbitals for each k-point already have the poper complex conjugate of bloch wave e(-ik) inside so we can just sum them up and we see cleared bands removing the folded replicas...


plese check the math and derivation and eventually write pythn script using numpy how to do that ...


---

# Chat GPT 5.6 sol

Yes. For what you want, **band unfolding is basically a Fourier transform of the supercell eigenvector over the primitive-cell replicas**. Your memory from the Fireball calculation is conceptually right. The one thing I would change is: with standard DFTB+ eigenvectors I would **not assume that the required \(e^{-i\mathbf k\cdot\mathbf R}\) phase has already been absorbed into the coefficients**. Put that phase in explicitly. Then there is a second, smaller issue: DFTB uses a non-orthogonal AO basis, so the rigorously normalized spectral weight contains the overlap matrix.

The nice part is that there is actually a paper doing essentially exactly this with **DFTB+ on graphene**. Farjam calculates the supercell bands along the path of the primitive graphene BZ, regarded as an extended-zone path of the supercell, and then applies the unfolding weight.  This is almost exactly the implementation I would use.

Your old paper is indeed the Si-\((\sqrt{19}\times\sqrt{19})R23.4^\circ/\mathrm{Pt}(111)\) Fireball work. ([APS Journals][1]) APS unfortunately blocks direct automated access to that supplemental PDF at the moment, so I could not inspect the exact notation you used there. But the DFTB derivation below independently gives the same Fourier-projection idea.

## 1. The simple derivation

Suppose the supercell contains \(N_c\) copies of the primitive cell, at primitive translations

$$
\mathbf R_j,\qquad j=0,\ldots,N_c-1.
$$

Let \(\mu\) identify an orbital **inside the primitive cell**. For graphene, for example,

$$
\mu =
(A,s),(A,p_x),(A,p_y),(A,p_z),
(B,s),(B,p_x),(B,p_y),(B,p_z).
$$

A supercell eigenstate at supercell momentum \(\mathbf K\) has coefficients

$$
C^{n\mathbf K}_{j\mu}.
$$

For a perfect crystal, a state which actually belongs to primitive-cell momentum

$$
\mathbf k=\mathbf K+\mathbf G
$$

has, apart from normalization,

$$
C^{n\mathbf K}_{j\mu}
\sim
c_\mu(\mathbf k)
e^{+i\mathbf k\cdot\mathbf R_j}.
$$

Therefore simply Fourier analyze the coefficients over \(j\):

$$
a_\mu^{n\mathbf K}(\mathbf k)
=
\frac{1}{\sqrt{N_c}}
\sum_j
e^{-i\mathbf k\cdot\mathbf R_j}
C^{n\mathbf K}_{j\mu}.
$$

For an orthogonal basis, the unfolded weight is

$$
\boxed{
W_{n\mathbf K}(\mathbf k)
=
\sum_\mu
\left|
a_\mu^{n\mathbf K}(\mathbf k)
\right|^2
}
$$

or explicitly,

$$
\boxed{
W_{n\mathbf K}(\mathbf k)
=
\frac{1}{N_c}
\sum_\mu
\left|
\sum_j
e^{-i\mathbf k\cdot\mathbf R_j}
C^{n\mathbf K}_{j\mu}
\right|^2 .
}
$$

That is the whole essential idea.

For a perfect supercell,

$$
W=1
$$

for the one primitive \(\mathbf k\) to which the state belongs and

$$
W=0
$$

for all its folded replicas.

So instead of plotting every eigenvalue with equal visibility, plot

$$
E_{n\mathbf K}
$$

with opacity/point size proportional to \(W_{n\mathbf K}(\mathbf k)\). The forest of folded bands disappears.

For a perturbed system, \(W\) becomes fractional, which is actually useful: it tells you how much primitive graphene Bloch character survives.

---

## 2. About your recollection that “we can just sum the coefficients”

This is the part I would be careful with.

If your coefficients had already been transformed as

$$
\widetilde C_{j\mu}
=
e^{-i\mathbf k\cdot\mathbf R_j}
C_{j\mu},
$$

then yes:

$$
a_\mu=\frac1{\sqrt{N_c}}\sum_j\widetilde C_{j\mu},
$$

so it literally becomes a sum.

That may well have been the convention/gauge of the Fireball output you used.

But for **DFTB+**, the published DFTB unfolding derivation explicitly contains the Fourier phase. In the orthogonal limit their Eq. (10) gives

$$
U_{NM}(\mathbf k)
=
e^{i\mathbf k\cdot(\mathbf r_N-\mathbf r_M)}
\delta_{n_N,m_M},
$$

which is precisely equivalent to the Fourier-sum expression above. 

So I would explicitly put the phase in. It costs essentially nothing.

There is a very easy test: run an **undistorted \(n\times n\) graphene supercell**. The resulting unfolded weights should be essentially \(0\) or \(1\). If plain summation works, then your particular eigenvector convention has already included the phase. If not, the formula above will.

---

# 3. An especially convenient trick: calculate directly along the primitive BZ path

You do **not** need to calculate only inside the small supercell BZ and then explicitly generate all the unfolded \(\mathbf k=\mathbf K+\mathbf G\).

You can give DFTB+ the path

$$
K_{\rm graphene}\rightarrow\Gamma\rightarrow M\rightarrow K_{\rm graphene}
$$

of the **primitive graphene BZ**, interpreted as an extended-zone path of the supercell.

That is exactly the shortcut used in the DFTB graphene unfolding paper: they note that the primitive-cell path can be treated as an extended-zone supercell path, so at each requested \(\mathbf k\) you only calculate one unfolding weight. 

DFTB+ specifies k-points in fractional coordinates of the reciprocal lattice of the periodic structure. ([DFTB+ Recipes][2])

For a simple \(n_1\times n_2\) graphene supercell,

$$
\mathbf A_1=n_1\mathbf a_1,\qquad
\mathbf A_2=n_2\mathbf a_2,
$$

primitive fractional coordinates

$$
(k_1,k_2)
$$

become supercell fractional coordinates

$$
(K_1,K_2)
=
(n_1 k_1,n_2 k_2).
$$

Values greater than 1 are fine conceptually: they just represent the extended zone.

So for a \(5\times5\) graphene supercell, primitive

$$
K=(1/3,2/3)
$$

corresponds to the supercell-coordinate point

$$
(5/3,10/3).
$$

The Hamiltonian spectrum repeats, of course; it is the **projection weight that selects the appropriate unfolded branch**.

---

# 4. Minimal NumPy implementation

Assume you have rearranged DFTB+ coefficients into

```python
C.shape == (nk, nband, ncell, norb_pc)
```

where `ncell = n1*n2`, and `norb_pc=8` for graphene with the usual \(s,p_x,p_y,p_z\) basis on each of the two carbons.

`cell_R` contains the primitive lattice translations of each replica, expressed in **primitive-cell lattice coordinates**:

```python
cell_R = [
    [0,0],
    [0,1],
    ...
]
```

Then:

```python
import numpy as np


def unfold_weights(C, kfrac, cell_R):
    """
    Primitive-cell band-unfolding weights.

    Parameters
    ----------
    C : complex ndarray, shape (nk, nband, ncell, norb_pc)
        Supercell eigenvector coefficients.

    kfrac : float ndarray, shape (nk, 2)
        Target primitive-cell k-points in fractional coordinates
        of primitive reciprocal vectors:
            k = k1*b1 + k2*b2

    cell_R : int/float ndarray, shape (ncell, 2)
        Primitive-cell translation of every replica:
            R_j = m1*a1 + m2*a2

    Returns
    -------
    W : float ndarray, shape (nk, nband)
        Unfolding spectral weight.
    """

    C = np.asarray(C)
    kfrac = np.asarray(kfrac)
    cell_R = np.asarray(cell_R)

    nk, nband, ncell, norb_pc = C.shape

    if cell_R.shape != (ncell, 2):
        raise ValueError("cell_R must have shape (ncell,2)")

    if kfrac.shape != (nk, 2):
        raise ValueError("kfrac must have shape (nk,2)")

    # k.R = 2*pi * (kfrac . cell_R)
    phase = np.exp(
        -2j * np.pi * np.einsum(
            'kd,cd->kc',
            kfrac,
            cell_R
        )
    )

    # Fourier projection over primitive-cell replicas
    #
    # A[k,b,mu] =
    #    1/sqrt(Ncell) sum_c phase[k,c] C[k,b,c,mu]
    A = np.einsum(
        'kc,kbcm->kbm',
        phase,
        C,
        optimize=True
    ) / np.sqrt(ncell)

    # Sum over primitive-cell orbitals
    W = np.sum(np.abs(A)**2, axis=-1)

    return W
```

For an ordinary \(n_1\times n_2\) supercell:

```python
def rectangular_cell_translations(n1, n2):
    return np.array(
        [(i, j)
         for i in range(n1)
         for j in range(n2)],
        dtype=float
    )


cell_R = rectangular_cell_translations(5, 5)
```

Then simply:

```python
W = unfold_weights(C, kfrac, cell_R)
```

and `W[ik, iband]` is how strongly that supercell eigenstate should be shown at that primitive-cell k-point.

The computation is tiny compared with DFTB itself.

---

# 5. Plotting the clean band structure

Given

```python
E.shape == (nk, nband)
W.shape == (nk, nband)
kdist.shape == (nk,)
```

something as simple as:

```python
import matplotlib.pyplot as plt

x = np.broadcast_to(kdist[:, None], E.shape)

plt.scatter(
    x.ravel(),
    E.ravel(),
    s=12.0 * W.ravel(),
    alpha=np.clip(W.ravel(), 0.0, 1.0)
)

plt.xlabel("k")
plt.ylabel("E (eV)")
plt.show()
```

will already produce the effective primitive-cell bands.

I would perhaps use a nonlinear visual mapping such as

```python
visibility = np.sqrt(np.clip(W, 0.0, 1.0))
```

only for plotting, because weak hybridized bands are otherwise difficult to see.

Do **not** alter the physical \(W\); only use the transformed value for graphical visibility.

---

# 6. But DFTB orbitals are non-orthogonal

This is the caveat.

DFTB+ solves

$$
H(\mathbf K)C
=
E\,S(\mathbf K)C,
$$

and the eigenvectors satisfy

$$
C^\dagger S C=1,
$$

not

$$
C^\dagger C=1.
$$

The DFTB+ graphene-unfolding paper explicitly points this out and uses both the eigenvectors and overlap integrals. 

The rigorous formula is

$$
\boxed{
W^{n\mathbf K}(\mathbf k)
=
\frac1{N_c}
C^\dagger
U(\mathbf k)
C
}
$$

with

$$
U_{NM}(\mathbf k)
=
\sum_{\mathbf r}
e^{i\mathbf k\cdot(\mathbf r-\mathbf r_M)}
S_{0N,\mathbf r m_M}.
$$

Those are essentially Eqs. (7–8) of the DFTB unfolding derivation. 

They also give the excellent diagnostic

$$
\boxed{
\sum_{\mathbf G}
W^{n\mathbf K}(\mathbf K+\mathbf G)=1.
}
$$



### Do you actually need this complication?

I would **not start with it**.

For your immediate purpose — removing the forest of folded graphene bands — I would first implement the simple Fourier projection above.

On ideal \(n\times n\) graphene it should be extremely obvious whether it works:

$$
W\approx1
$$

on the true bands and

$$
W\approx0
$$

on the unwanted replicas.

If you obtain something like

```text
0.98, 0.01, 0.00, ...
```

rather than mathematically perfect `1,0,0,...`, that is already perfectly adequate for visualization.

If instead weights are broadly distributed, say `0.6, 0.2, 0.1...` even for pristine graphene, then nonorthogonality/gauge conventions matter enough that we should implement the proper overlap-corrected version.

---

## 7. There is an even nicer simplification for pristine/repeated graphene

If all primitive-cell replicas are genuinely equivalent, the overlap correction can itself be reduced to a **small primitive-cell matrix**.

After the Fourier transform define

$$
\mathbf a(\mathbf k)
=
\frac1{\sqrt{N_c}}
\sum_j
e^{-i\mathbf k\mathbf R_j}
\mathbf C_j.
$$

Then instead of

$$
W=\mathbf a^\dagger\mathbf a,
$$

use approximately/exactly under primitive translational symmetry

$$
\boxed{
W=
\mathbf a^\dagger
S_{\rm pc}(\mathbf k)
\mathbf a .
}
$$

For graphene \(S_{\rm pc}\) is only \(8\times8\) if you have four valence orbitals per C.

That preserves essentially the same ultra-cheap algorithm and handles AO overlap properly for a pristine repeated lattice.

For strongly distorted/defective cells, I would go back to the general \(U(\mathbf k)\) expression.

---

# 8. One important implementation detail: group by *equivalent primitive orbital*, not merely by atom

For graphene you want, schematically,

$$
\begin{aligned}
&A_0\,p_z,\ A_1\,p_z,\ldots,A_{N_c-1}\,p_z \\
&B_0\,p_z,\ B_1\,p_z,\ldots,B_{N_c-1}\,p_z
\end{aligned}
$$

to be Fourier summed separately.

Do not sum A and B together before taking the modulus. Their relative phase is real physical information — it generates the graphene pseudospin structure.

Similarly,

```text
A:s
A:px
A:py
A:pz
B:s
B:px
B:py
B:pz
```

are eight independent primitive-cell channels.

Only after Fourier projection do you sum

$$
W=\sum_\mu |a_\mu|^2.
$$

That point is important.

---

## What I would implement first

The minimal pipeline I would use is:

```text
DFTB+ eigenvectors
        |
        v
reshape AO coefficients as
[k, band, primitive-cell-copy, primitive-orbital]
        |
        v
multiply cell copies by exp(-i k.R)
        |
        v
sum over cell copies
        |
        v
|...|^2 and sum over primitive orbitals
        |
        v
plot eigenvalue with intensity W
```

That is probably **tens of lines of Python**, not an unfolding package.

And in fact your intuition can be stated very compactly:

$$
\boxed{
\text{unfolding = Fourier transform of the eigenvector over the supercell translations.}
}
$$

The only amendment is: **for DFTB+, explicitly include the Fourier phase; don't rely on it already being hidden in the coefficients, and remember the \(S\)-matrix if you want quantitatively exact spectral weights.**

DFTB+ can output the eigenvectors explicitly; its documentation describes `eigenvec.out`/`eigenvec.bin` as containing the basis-function components for each eigenvector and k-point. ([Xi'an Jiaotong University][3])

If you show me one small example of your actual DFTB+ `eigenvec.out` plus the graphene `.gen` geometry and tell me the supercell dimensions, I can write the **actual parser + unfolding script matched to DFTB+'s orbital ordering**, rather than leaving `C` as an abstract NumPy array.

[1]: https://journals.aps.org/prb/recent?ei=IykdU7y2O4SrhQfhwoDoDw&page=3210&sa=U&usg=AFQjCNHVdztX5tDcC9giBYc3LkOj-eME3w&ved=0CDYQFjAGOAE&utm_source=chatgpt.com "Physical Review B - Recent Articles"
[2]: https://dftbplus-recipes.readthedocs.io/en/latest/basics/bandstruct.html?utm_source=chatgpt.com "Band structure, DOS and PDOS — DFTB+ Recipes"
[3]: https://gr.xjtu.edu.cn/c/document_library/get_file?folderId=2704407&name=DLFE-129827.pdf&utm_source=chatgpt.com "DFTB+
Version 1.3
U S E R
M A N U A L"
