# USER

in https\://github.com/ProkopHapala/SPAMMM ( [https://deepwiki.com/ProkopHapala/SPAMMM](https://deepwiki.com/ProkopHapala/SPAMMM)) I have 3 ways to do AFM in SPM\_CLI FDBM, Morse+Q (both of which work with 3D grid with step like 0.1-0.2A or Contact surface 2.5D method) main modication for contact surface is memory efficieny allwoed by small grid and small number of basis functiion. But current implementation is bumpy aspecially at large distance which is unphysical and make it unusable we are thinking about options how to improve it.

More details are in attached documents, plese review it and suggest your ideas, cricial analysis, etc. yourse your deep knowledge of numerical math and function aproximation and experience from related areas.

---

# Chat GPT 5.6 sol

I reviewed the current `main` code together with both attached task documents. My main conclusion is that the **2.5D contact-surface idea is fundamentally good**, and I would not abandon it in favor of a dense 3D grid. But I think the present problem is deeper than “B-splines are too coarse.” The current representation mixes spatial scales that physically should separate with height.

The attached diagnostic work has already ruled out several tempting explanations, but I would slightly revise its final diagnosis and architecture.

## 1. What I think is already solid

The unification of FDBM, Morse+Q, and Contact around the same scan geometry, PP parameters, amplitude-convolved `df`, Fz extraction, and plotting is the correct architecture. In particular, keeping Contact as a compressed representation rather than expanding it into a 3D `F_total` is exactly right. 

Several nasty implementation issues have also already been fixed correctly: the B-spline prefilter now solves the finite zero-padded tridiagonal interpolation problem rather than using an infinite-domain IIR inverse; `h0_samples` and B-spline control coefficients are distinguished; and the GridFF half-texel convention has regression tests. The reported B-spline boundary-node error went from about (2.7\times10^{-2}) to machine precision. 

The later contact diagnostics are particularly valuable because they corrected the earlier, too-simple “coarse grid + unconstrained coefficients” story. Refining `bspl_dx`, oversampling the fit points, varying CG iterations, changing (K_\mathrm{LAT}), and increasing `poly_R` did **not** remove the far-field structured error. At representative far heights the absolute errors are small but relative errors are already of order 10–20%, which is enough to create conspicuous corrugation when the physical signal itself is weak. 

So I would stop spending time on simply making the uniform B-spline grid finer.

# 2. The central mathematical problem

Your current representation is essentially

[
E(x,y,z)
========

\sum_k C_k(x,y),\phi_k(s),
\qquad
s=z-h_0(x,y)-z_0,
]

where each (C_k(x,y)) is a bicubic B-spline field and

[
\phi_k(s)
=========

\left(1-\frac{s}{R}\right)^{m,2^k},
]

inside the support. The actual kernel evaluates precisely this tensor-product structure and differentiates it consistently, including the (h_0) chain-rule term.

There are **three distinct kinds of scale mixing** here.

### Vertical modal mixing

For your typical (m=4,n_z=6),

[
p_k = 4,8,16,32,64,128.
]

Close to contact,

[
\phi_k(s)
\simeq 1-\frac{p_k}{R}s+\cdots,
]

so all the energy basis functions start close to the same constant. Farther away, however, the high powers collapse extremely rapidly. For example at (s=R/2),

[
\phi_k = 2^{-p_k},
]

so the (p=128) mode is already about (3\times10^{-39}). Essentially **one mode controls the far field**.

That fact by itself is not bad. The problem is that the coefficient map (C_0(x,y)) is obtained simultaneously from near-, medium-, and far-field data. Therefore the optimizer can use atomic-scale structure in (C_0) to reduce near-field residuals, and that same atomic structure survives as the only remaining spatial pattern at large (s).

This is the “spectral mixing” identified by your latest diagnostics. The evidence for it is strong: only mode 0 remains significant at large (s), while finer lateral grids and more CG iterations do not remove the far-field error. 

But I would make one important logical correction:

> **You have not demonstrated an intrinsic limitation of separability itself. You have demonstrated a limitation of this particular fixed global z basis combined with one unrestricted xy bandwidth per mode.**

That distinction matters enormously.

An SVD/POD basis with rank six might conceivably represent the same field much more accurately. Or a local z basis might. So I would not yet conclude “2.5D separability has a 10% floor.”

# 3. There is an even deeper issue: (h_0(x,y)) should probably be a *near-field coordinate only*

Because

[
s=z-h_0(x,y),
]

the lateral force contains

[
F_x =
-\sum_k
\left[
\frac{\partial C_k}{\partial x}\phi_k
-------------------------------------

C_k\phi'_k\frac{\partial h_0}{\partial x}
\right].
]

The kernel implements exactly this chain rule.

Near contact, that is beautiful: the potential follows the molecular contact surface, exactly what you want.

But far from the molecule it becomes conceptually dubious.

At fixed world (z), every variation of (h_0(x,y)) changes the argument of the long-range basis. Consequently even a perfectly smooth (C_0(x,y)) can acquire atom-scale modulation through

[
\phi_0!\left(z-h_0(x,y)\right).
]

And (\nabla h_0) directly enters (F_x,F_y).

This is the numerical analogue of using a highly corrugated curvilinear coordinate system in a region where the underlying field is becoming smooth.

Your sphere-based (h_0) is much better than the old `atom_z` version—the latter fails catastrophically in the ablation—but that does **not** prove that a local contact coordinate is optimal for the far field. 

I therefore think the ultimate model should have two coordinate systems:

[
\boxed{
E(\mathbf r)=E_\mathrm{long}(x,y,z)
+
E_\mathrm{short}(x,y,s)
}
]

with (s=z-h_0(x,y)) used only for the short-range contact physics.

This is probably the single most important architectural change I would make.

# 4. The physics actually gives you an extremely natural decomposition

For Morse+Q this is almost handed to you:

[
E_M(r)=
D e^{-2\alpha(r-R_0)}
---------------------

2D e^{-\alpha(r-R_0)}
]

plus electrostatics.

So there are intrinsically three different spatial scales:

[
E = E_\mathrm{P}+E_\mathrm{L}+E_\mathrm{Q},
]

where

[
E_\mathrm{P}\sim e^{-2\alpha r},
\qquad
E_\mathrm{L}\sim-e^{-\alpha r},
\qquad
E_\mathrm{Q}\sim 1/r.
]

And SPAMMM already has this decomposition explicitly in `getMorsePLQH`: `EP`, `EL`, and `EQ` are computed separately.

That is much better than asking a generic regression to discover the decomposition from the *sum*.

I would exploit it.

For FDBM you already have almost exactly the analogous channels:

[
E_\mathrm{FDBM}
===============

E_\mathrm{Pauli}
+
E_\mathrm{ES}
+
E_\mathrm{vdW}.
]

So this could become a **common compressed-field architecture for Morse and FDBM**, rather than just a fix for Contact:

[
\begin{array}{c|c|c}
\text{channel} & \text{natural coordinate} & \text{lateral resolution}\
\hline
\text{Pauli / Morse repulsion} & s=z-h_0 & \text{fine}\
\text{London / vdW} & z\ \text{or weakly surface-following} & \text{medium}\
\text{electrostatics} & x,y,z & \text{very coarse / analytic}
\end{array}
]

This prevents the present pathological situation in which the slowest basis function has to help fit the repulsive wall.

There is also a strong mathematical reason. Above a source plane, an electrostatic Fourier component obeys

[
\frac{\partial^2 \hat V(\mathbf k,z)}{\partial z^2}
-k^2\hat V(\mathbf k,z)=0
]

and therefore

[
\hat V(\mathbf k,z)
===================

\hat V(\mathbf k,0)e^{-|\mathbf k|z}.
]

**High lateral spatial frequencies necessarily disappear faster with height.**

Your current tensor-product model permits the opposite: an arbitrarily high-(k) xy structure in (C_0) can multiply the slowest-decaying z mode. That is the basic architectural mismatch.

The Morse terms have a similar qualitative behavior. Expanding

[
r=\sqrt{z^2+\rho^2}
\simeq z+\frac{\rho^2}{2z}
]

gives approximately

[
e^{-\alpha r}
\simeq
e^{-\alpha z}
\exp\left(-\frac{\alpha\rho^2}{2z}\right),
]

so lateral atomic contrast broadens with height. Again: **fine xy scales should be coupled to fast z decay, and coarse xy scales to slow z decay.**

# 5. What I would try, in order

1. **Prove the spectral-mixing hypothesis before redesigning anything.** Take the fitted coefficient maps (C_k(x,y)), compute their 2D spectra, and plot each mode's actual contribution (C_k\phi_k(s)) at several heights. Then do one deliberately artificial experiment: low-pass filter **only (C_0)** and reevaluate the field. Do not call this a solution; use it as a causal diagnostic. If the far bumpiness disappears while near-field parity hardly changes, you have directly proved the mechanism. Your existing evidence is strongly suggestive but this experiment would close the logical gap. 

   At the same time, build an **SVD oracle**. Sample the brute field on a surface-following tensor grid,

   [
   M_{qj}=E(x_q,y_q,h_0(x_q,y_q)+s_j),
   ]

   reshape (q=(x,y)), and compute the singular values. Reconstruct it with ranks (1\dots 8). This tells you the *best possible* global separable error at a given rank. If rank six SVD gives 1% while the current six powers give 15%, the separable concept is fine and your (\phi_k) are poor. If even optimal rank six stalls near 10%, then you have evidence for a genuine low-rank/coordinate limitation.

2. **As the quickest real experiment, implement the two-pass fit—but constrain the far mode spatially.** Your task document proposes fitting mode 0 on far samples, locking it, then fitting modes (1\dots5) to the residual.  I agree with the experiment, but I disagree with one phrase in the document: far-only fitting does **not by itself guarantee smoothness**. The far signal is weak, so a fine unregularized (C_0) can still wiggle.

   I would fit (C_0) either on a coarse 2–3 Å B-spline lattice or with

   [
   \min_{C_0}
   |A_0C_0-b_\mathrm{far}|_W^2
   +
   \lambda_0|\nabla^2 C_0|^2.
   ]

   Then fit higher modes to the residual with much weaker or zero smoothing. You can even prototype the coarse version **without changing the production evaluator**: fit coarse (C_0), prolong/interpolate it onto the existing fine coefficient grid, lock that coefficient block, and fit the rest.

3. **Replace the doubling powers by local B-splines in (s).** This is probably my favorite medium-term fix.

   Instead of six global functions

   [
   t^4,t^8,t^{16},\ldots,
   ]

   use a nonuniform one-dimensional cubic B-spline basis

   [
   E(x,y,s)
   ========

   \sum_{ijk}
   c_{ijk}B_i(x)B_j(y)B_k(s).
   ]

   The crucial difference is **local support**. A coefficient describing (s=4) Å cannot help reduce an error at (s=0.3) Å. Vertical spectral pollution disappears almost by construction.

   And this is surprisingly attractive computationally: a cubic spline uses only **four active z coefficients** at a point. Your present kernel evaluates 16 xy coefficients for *every one* of six z modes, about 96 coefficient contributions. A tricubic (xy\times s) spline needs (4^3=64), regardless of whether you have 10, 20, or 30 total z knots. Thus you can use many more nonuniform z knots while potentially making evaluation **faster**.

   Dense knots can cover the repulsive wall and well, with progressively wider spacing in the tail. This is exactly the kind of local multiresolution spline construction for which hierarchical B-spline methods are useful; THB-splines were developed specifically to retain locality and partition-of-unity properties under hierarchical refinement. ([ScienceDirect][1])

4. **For production, split long and short range explicitly.** I would aim for something like

   [
   E =
   E_\mathrm{long}(x,y,z)
   +
   \chi(s)E_\mathrm{contact}(x,y,s),
   ]

   where (\chi) is a smooth (C^2) or (C^3) switching function.

   `E_long` can be astonishingly cheap because it is smooth: a 0.5–1 Å 3D grid, a very coarse 2D spline plus a few z functions, a low-rank SVD representation, direct/multipole point-charge electrostatics, or some combination. Going from 0.1 Å to 0.5 Å reduces a uniform 3D voxel count by (5^3=125), so a **coarse long-range 3D field + fine 2.5D contact field** may actually be the best engineering compromise.

   The contact residual then has compact support and is allowed to become exactly zero because the physical tail has already been removed.

5. **Make lateral resolution depend on vertical scale.** This is the more elegant version of your R2.7 hierarchical proposal:

   [
   E =
   \sum_\ell
   C_\ell^{(h_\ell)}(x,y),
   \psi_\ell(s),
   \qquad
   h_0>h_1>h_2>\cdots
   ]

   with slow modes living only on coarse grids and fast modes on progressively finer grids.

   I would think of this as **multigrid rather than just “different grids per mode.”** The coarse grid represents the far-field component; finer grids represent corrections/residuals. A Laplacian/Gaussian pyramid or nested B-splines could make the implementation quite clean. THB-splines are one formal route to this kind of nested representation. ([ScienceDirect][1])

6. **Only after the representation is fixed, improve the least-squares solver.** Current `fit_separable_cg()` applies CG to matrix-free normal equations (A^TWA,c=A^TWb) in fp32, with no regularization in the global production fit.  That squares the effective condition number of (A), which is especially undesirable for strongly correlated powers (t^4,t^8,\ldots).

   Since you already have efficient (Av) and (A^Tu), LSQR/LSMR/CGLS is an extremely natural fit. LSQR operates directly through those two operators and supports damped least squares without explicitly forming normal equations. ([Stanford University][2])

   An even cheaper experiment is to build the tiny (6\times6) weighted Gram matrix of the z basis, including both energy and derivative rows, orthogonalize/whiten the z modes, fit in that basis, and transform the coefficients back to the current powers. That requires essentially no scan-kernel change.

# 6. I would seriously consider **interpolation instead of global fitting**

There is another option that I think deserves more attention.

You already have a very fast brute Morse reference. Build a small curvilinear tensor grid

[
(x_i,y_j,s_k),
\qquad
z_{ijk}=h_0(x_i,y_j)+s_k,
]

evaluate the exact energy there once, and compute tensor-product cubic B-spline coefficients.

Then you have a genuine narrow-band 3D spline in curvilinear coordinates.

No CG.

No global z-mode fitting.

No vertical spectral mixing.

No force-row weighting.

No arbitrary `m_start`.

And (F=-\nabla E) follows analytically from the spline.

With perhaps (20\times20\times12) coefficients you have only 4800 floats—roughly 19 kB. That is still utterly tiny compared with a dense 3D force field.

I would benchmark this surprisingly simple option against the current 1440-coefficient model before investing in complicated regularization. You may discover that spending 3–4× more **tiny** 2.5D memory buys dramatically better robustness while retaining essentially negligible memory consumption.

If energy-only cubic interpolation does not give sufficient force accuracy, then use derivative-constrained/Hermite fitting locally. But I would test plain interpolation first.

# 7. The current lower clamp is still a real production problem

The kernel deliberately makes the energy constant and the force zero for (s<0). That makes (F=-\nabla E) mathematically consistent, but physically it means the repulsive wall disappears if the PP enters that region. The attached task correctly recognizes this as dangerous. 

More importantly, I found that the current unified `clamp_occupancy` does **not actually test relaxed trajectories**.

It computes (h_0) by nearest sampling of `sep.h0_samples`, evaluates

[
s = z_\mathrm{nominal}-h_0-\mathrm{poly_z0},
]

and counts negative values. It does not evaluate interpolated (h_0) at the actual laterally relaxed PP position, and it does not use the PP's actual relaxed (z). `shared_postprocess` is in fact called with `tip_disp=None`.

So `clamp_occupancy=0.0000` is useful as a nominal-grid check but **is not the R2.4 trajectory safety condition** described in the task. The integration notes themselves acknowledge that R2.4 was not formally run. 

I would put very cheap telemetry directly into `relaxStrokesTiltedContact`: per pixel retain `min_s`, `n_s_lt_0`, `min(poly_R-s)`, convergence residual, and final PP position. A production scan should fail or loudly flag any trajectory for which

[
s_\mathrm{min}<s_\mathrm{fit,min}+\Delta s_\mathrm{guard}.
]

Better still, extend the representation into negative (s), perhaps to (-0.5) or (-1) Å depending on the actual trajectory envelope, so the fitted basis includes the repulsive wall.

And I would add an analytic repulsive continuation below the lower training boundary as a final safety mechanism rather than a zero-force plateau.

# 8. One contradiction in the current design notes

The R2.7 document says the “PIC backend already uses” a coarse long-range B-spline plus atom-centered short-range functions. 

The current kernel does **not** appear to do that. `cs_pic_eval_tile16` loads nearby atoms from buckets and sums atom-centered compact radial modes directly; I don't see a coarse B-spline long-range component in that evaluator.

So I would describe the proposed hybrid as **an extension inspired by PIC**, not the architecture PIC already implements.

That said, atom-centered residuals remain an excellent idea for the shortest-range part because they automatically move with the atoms and naturally represent atomic corrugation.

# 9. My preferred final architecture

If I were choosing where SPAMMM should converge, I would aim for:

[
\boxed{
E =
E_\mathrm{ES/Q}^{\mathrm{long}}
+
E_\mathrm{vdW/L}^{\mathrm{medium}}
+
E_\mathrm{Pauli/P}^{\mathrm{contact}}
}
]

with three very different numerical representations.

The **long-range channel** should use world coordinates, extremely low lateral bandwidth, and infinite/physical tail behavior. The **medium-range attractive channel** can use a coarse multiresolution spline or low-rank representation. The **repulsive channel** should use (s=z-h_0(x,y)), fine xy resolution, local-support z splines, and a finite compact support.

For Morse this decomposition is already exact at the pair-potential level. For FDBM it maps naturally onto Pauli / electrostatics / dispersion. That means the same compressed-field machinery could eventually serve both backends.

I think this is substantially more promising than trying to find the perfect set of six global powers.

The immediate experiment I would run is therefore **not** another `bspl_dx` or `poly_R` sweep. I would do the SVD oracle + (C_0)-spectrum diagnostic, then a coarse/smoothed far-mode two-pass fit, and in parallel a small prototype replacing the doubling powers by 1D local cubic B-splines in (s). Those three tests should tell you very quickly whether you can retain essentially the same 2.5D architecture or whether the explicit world-z long-range split is necessary.

[1]: https://www.sciencedirect.com/science/article/pii/S0167839612000519?utm_source=chatgpt.com "THB-splines: The truncated basis for hierarchical splines - ScienceDirect"
[2]: https://web.stanford.edu/group/SOL/software/lsqr/index.html?utm_source=chatgpt.com "LSQR: Sparse Equations and Least Squares"

---

# USER

I kinda do not like the idea to use different representation for the 3 physical componentas since all of them are basically radial basis function. This radial function for all 3 compoent s (Pauli,London,Coulomb) is very steep close to atom and very smooth far from atom, so it does not really make sense to make fundamentally different representation for the 3 interactions. We should split the representation based on other consideration (smooth, steep, short-range vs long-range )

But I want model which scales efficiently (short cuoffs particle-in-cell PIC architecture but at the same time smooth. 

I think better is to combine the components into one filed and than approximate that filed using 2 representations
1) smooth long range model (like B-spline), thsoi should be fitted first to ensure smoothness
2) as a corection short range atomic cores.

Reformulate the desing fundamentaly around these ideas. Focus on how to ensure we will not see or produce artifacts in the smooth long grage fied de to errors of fit in atomic details in the short thange fit, I think the best strategy is fit coarse-grid-based long range first and then embed short range radial function into it. 

In essence this is like Particle-Mesh-Ewald method (where we split electrostatics to short range real-space core and long-range FFT Grid). Just the grid voxel size and core-function shape other things are different here. We are not aiming for maximum accuracy but for maximum speed. I would suggest the grid spacing like 1A (roughly) and the basis function should b our exponential-polynominal approx 
(1-r/rc)^(2^m)

the finite cutoff is really nor proble, it is feature not a bug, We do not represent the potential all the way to the center of atom, we only need to discribe some chemically relevant region on the surface of atoms 

so our polynominal fit should represrnt Morse+Coulomb potential of an atom only in some spherical shell from [rmin,rmax] where e.g. rmin=R0-0.5A, and rmax=6.0A, something like that. 


---

# Chart GPT 5.6 sol

Yes. I would reformulate the whole Contact/PIC idea around a **spatial-scale decomposition of the total interaction field**, completely independent of whether a contribution originally came from Pauli, London, or Coulomb.

The fundamental model becomes

[
\boxed{
V(\mathbf r)
\approx
V_{\rm mesh}(\mathbf r)
+
\sum_i V^{\rm core}_i(|\mathbf r-\mathbf R_i|)
}
]

where:

* (V_{\rm mesh}) is deliberately **smooth and low bandwidth**, represented on a coarse 3D grid, perhaps (h\sim0.7-1.2) Å.
* (V_i^{\rm core}) is a **compact atom-centered residual**, evaluated only for nearby atoms using PIC/cell lists.
* Both together approximate the **total combined Morse+Coulomb field**.
* The core is fitted only over the physically relevant shell, e.g.
  [
  r\in[r_{\min},r_{\max}]
  ]
  and we explicitly do **not care** what happens deep inside (r<r_{\min}), because a valid PP trajectory must not go there.

This is much closer in spirit to PME/P3M, except your split criterion is essentially **spatial frequency / curvature**, not electrostatic range.

The current contact representation instead forces all spatial scales into
[
V(x,y,z)=\sum_k C_k(x,y)\phi_k(z-h_0),
]
which is exactly why a slow (z)-mode can acquire atomic-scale lateral structure. The latest diagnostics in your document show that this mixing remains even after refining `bspl_dx`, oversampling the fitting grid, and converging CG. 

## 1. The critical rule: the mesh must never see the steep core

This is more important than the precise fitting algorithm.

If you simply sample the exact field on a 1 Å grid and interpolate it, an unlucky grid node can land close to an atom. A huge repulsive value then contaminates several cubic B-spline cells, giving exactly the broad/bumpy artifact we are trying to eliminate.

So the desired decomposition is **not**

[
\text{coarse interpolation of raw }V
+
\text{whatever residual remains}.
]

Instead we must deliberately define a low-pass version.

A very clean construction for Morse+Q is to work with the **combined radial atom potential**

[
v_i(r)
======

v^{\rm Morse}_i(r)+v^{Q}_i(r)
]

first, and split that whole radial function:

[
v_i(r)=v_i^{L}(r)+v_i^{S}(r).
]

No Pauli/London/Coulomb distinction remains after this point.

---

# 2. I think a “soft-core mesh + compact correction” is the cleanest PME analogue

Choose a splitting radius (r_s), perhaps somewhere around the vdW/contact region.

For the smooth component define

[
v_i^L(r)=v_i(r),\qquad r\ge r_s .
]

Inside (r_s), **do not continue the steep true potential**. Replace it by some smooth soft-core continuation:

[
v_i^L(r)=P_i(r),\qquad r<r_s ,
]

where (P_i) is chosen to match perhaps

[
P_i(r_s)=v_i(r_s),
]

[
P_i'(r_s)=v_i'(r_s),
]

[
P_i''(r_s)=v_i''(r_s).
]

Deep inside the atom (P_i) can simply flatten.

Then define

[
\boxed{
v_i^S(r)=v_i(r)-v_i^L(r)
}
]

so automatically

[
v_i^S(r)=0,\qquad r\ge r_s.
]

If value, slope and curvature are matched at (r_s), then

[
v_i^S(r_s)=
v_i^{S,\prime}(r_s)=
v_i^{S,\prime\prime}(r_s)=0.
]

That gives you a very smooth compact core.

This has a huge conceptual advantage over fitting the mesh and core simultaneously:

> **The long-range field cannot possibly acquire information about the steep atomic repulsion because that repulsion has already been removed before rasterization.**

And conversely the core cannot generate a long-range artifact because it is identically zero beyond (r_s).

That is a much stronger guarantee than regularization.

---

# 3. Then build only the smooth part on the coarse mesh

The mesh potential is

[
V_L(\mathbf r)
==============

\sum_i v_i^L(|\mathbf r-\mathbf R_i|).
]

Sample this onto a coarse lattice:

[
h \sim 1\ \text{Å}.
]

A 1 Å grid compared with your usual 0.1 Å GridFF reduces voxel count by roughly

[
10^3 = 1000\times.
]

Even compared with 0.2 Å it is

[
5^3=125\times
]

smaller.

This is already so cheap that I would strongly consider a **full coarse 3D grid**, rather than trying to squeeze the smooth component into another 2.5D representation.

For example, a (40\times40\times16) region has only 25,600 coefficients:

* scalar float32 potential: ~100 kB;
* even float4 field: ~400 kB.

That is negligible compared with your original fine GridFF.

And there is no need for `h0(x,y)` or unusual coordinates for this component. It simply lives in ordinary world coordinates.

---

# 4. The mesh should represent energy, with a smooth interpolant

I would preferably store one scalar mesh (V_L), then obtain

[
\mathbf F_L=-\nabla V_L
]

analytically from the interpolation basis.

Cubic B-spline interpolation is ideal here:

[
V_L(x,y,z)
==========

\sum_{abc}
C_{abc}
B_a(x)B_b(y)B_c(z).
]

Then force is exactly consistent.

Naively this means (4^3=64) active coefficients. But on GPUs cubic B-spline interpolation can be reformulated using hardware-linear interpolation into roughly eight trilinear fetches, so I would not regard 64 conceptual taps as necessarily the final execution cost.

The important point is that the mesh is smooth by construction.

Linear texture interpolation of a stored `float4(F,E)` would be even cheaper, but I would initially keep the conservative

[
F=-\nabla V
]

form because PP relaxation is sensitive to small force-field discontinuities.

---

# 5. Atomic cores then fit only the residual

Once (V_L) is frozen,

[
V_S(\mathbf r)
==============

V_{\rm reference}(\mathbf r)-V_L(\mathbf r).
]

Because we deliberately removed the steep part before making (V_L), this residual should be strongly localized around atoms.

Now fit

[
V_S(\mathbf r)
\approx
\sum_i
\sum_m
c_{im},
\phi_m(r_i),
\qquad
r_i=|\mathbf r-\mathbf R_i|.
]

This is your PIC part.

Evaluation becomes

[
\boxed{
V(\mathbf r)
============

V_L^{\rm grid}(\mathbf r)
+
\sum_{i\in {\cal N}(\mathbf r)}
\sum_m c_{im}\phi_m(r_i)
}
]

where ({\cal N}(\mathbf r)) is obtained from a short-range cell list.

That has excellent asymptotic behavior:

[
O(1)*{\rm grid}
+
O(N*{\rm local})_{\rm core}.
]

No dependence on total atom count during scanning.

---

# 6. Your polynomial basis fits this architecture much better than the present one

Now the compact cutoff becomes an unambiguous **feature**.

Suppose the physically relevant shell is

[
r_{\min}=R_0-0.5\ {\rm Å},
]

[
r_{\max}=6\ {\rm Å}.
]

Define

[
x=
\frac{r-r_{\min}}
{r_{\max}-r_{\min}},
]

[
t=1-x.
]

Then

[
\phi_m(r)
=========

\begin{cases}
(1-x)^{p_m}, & 0\le x<1,\
0, & x\ge1,
\end{cases}
]

with for example

[
p_m=2^{m+m_0}
=============

4,8,16,32,\ldots
]

or similar.

This is actually much more appropriate here than in the old global (z) representation.

Why?

Because now **all these modes are short-range corrections**.

If one mode gets contaminated by some fine structure, it cannot produce a 10 Å artifact. Its support ends at 6 Å.

And for

[
\phi(r)=(1-x)^p,
]

the cutoff is extremely smooth. At (r=r_{\max}),

[
\phi=0,
]

and for (p>1),

[
\phi'=0,
]

for (p>2),

[
\phi''=0,
]

etc.

So (p=4) already joins zero with several smooth derivatives.

This is where your powers-of-two basis really shines.

---

# 7. I would not clamp the core below (r_{\min})

This is an important distinction from the current `s<0` contact clamp.

The model is simply declared valid over

[
r_{\min}\le r.
]

If a PP trajectory reaches

[
r<r_{\min},
]

that is a model-domain failure.

There is no need to invent a physically meaningful atomic center.

So conceptually:

```text
r < rmin:
    invalid / trajectory too close
rmin <= r < rmax:
    evaluate compact core
r >= rmax:
    core = 0
```

For production safety you might extrapolate the value/force at (r_{\min}) rather than generating NaNs, but telemetry should mark it as unsupported.

That is much cleaner than pretending the zero-force plateau is physical. Your current contact task already recognizes the danger of silently entering an unsupported region. 

---

# 8. There should be an **overlap region** where the mesh is authoritative

This will make the fitting much more robust.

For example:

[
r_{\min}\approx R_0-0.5,
]

[
r_{\rm split}\approx3.5-4.0,
]

[
r_{\max}\approx5.5-6.0\ {\rm Å}.
]

The coarse mesh should already be accurate in the outer part, say

[
r>r_{\rm split}.
]

The core correction exists until (r_{\max}), but is forced smoothly toward zero.

Thus there is an overlap annulus

[
r_{\rm split}<r<r_{\max}
]

where the two representations coexist but the correction gradually disappears.

This is exactly the place to test for seams.

The fitting objective for the core can weight that region strongly:

[
\min_c
\sum_q
w(r_q)
\left[
V_S^{\rm model}(q)
------------------

V_S^{\rm ref}(q)
\right]^2,
]

where (w(r)) ensures good matching near the cutoff.

The resulting core need not reproduce every tiny residual at (r_{\max}); it should preferably go to zero and leave the smooth mesh in control.

---

# 9. An even better basis transformation may be useful

The raw sequence

[
t^4,t^8,t^{16},t^{32},\ldots
]

is quite correlated near (t=1).

You can keep exactly the same span but use hierarchical difference functions:

[
\psi_0=t^4,
]

[
\psi_1=t^8-t^4,
]

[
\psi_2=t^{16}-t^8,
]

[
\psi_3=t^{32}-t^{16},
]

etc.

These behave rather like radial detail bands.

At the inner shell boundary (t=1),

[
\psi_{m>0}=0,
]

and at the cutoff (t=0),

[
\psi_m=0.
]

So the higher functions naturally describe intermediate/short radial scales instead of all competing to set the same constant value.

The spaces are identical because this is just an invertible linear transformation of the coefficients.

That could improve the conditioning essentially for free.

---

# 10. For Morse+Q, fitting the atomic core can actually become almost a 1D problem

This is another huge simplification.

For atom (i), the combined pair interaction is already radial:

[
v_i(r)
======

D_i
\left(
e^{-2\alpha(r-R_i)}
-------------------

2e^{-\alpha(r-R_i)}
\right)
+
\frac{Kq_iq_t}
{\sqrt{r^2+r_d^2}}.
]

Instead of fitting the 3D short-range residual globally, you can sample this 1D radial function densely:

[
r_j \in [r_{\min},r_{\max}]
]

and determine (c_{im}) from a tiny linear least-squares system.

If atom type determines the parameters, the coefficient vector can even be **per atom type**, not per atom:

[
c_{im}\rightarrow c_{\tau(i),m}.
]

Then for a molecule you store perhaps 4–6 floats per atom type plus whatever charge dependence is necessary.

If charges vary atom-by-atom, one possibility is still to fit the combined radial function separately per atom. Six coefficients per atom is tiny:

[
N\times6\times4\ \text{bytes}.
]

For (10^5) atoms that is only 2.4 MB.

So I would strongly exploit radiality here rather than solving a giant 3D CG problem unless necessary.

---

# 11. But we need to be careful about what “residual per atom” means

There are two possible implementations.

### A. General field-residual version

First build the complete coarse field (V_L).

Then fit atom cores to

[
V_{\rm exact}(\mathbf r)-V_L(\mathbf r)
]

at 3D sample points.

This works even for a reference that is not exactly pairwise/radial, potentially including FDBM later.

But neighboring atomic cores overlap, so coefficients should technically be determined by a sparse global least-squares problem.

### B. PME-like atomwise split

For each atom separately define

[
v_i=v_i^L+v_i^S.
]

Then

[
V_L=\sum_i v_i^L
]

is rasterized onto the mesh and

[
V_S=\sum_i v_i^S
]

is evaluated directly.

This is much cleaner for Morse+Q.

There is no fitting ambiguity whatsoever between neighboring atoms because linearity guarantees

[
\sum_i(v_i^L+v_i^S)
===================

\sum_i v_i.
]

I think **B should be your reference implementation** for Morse+Q.

Then A can later generalize the same architecture to fields that cannot be decomposed atomwise.

---

# 12. How to choose the split automatically

Rather than setting (r_s) arbitrarily, choose it based on what a 1 Å mesh can represent.

A grid with spacing (h) has Nyquist wave number approximately

[
k_N=\frac{\pi}{h}.
]

For (h=1) Å,

[
k_N\simeq3.14\ {\rm Å}^{-1}.
]

You don't want to use anything close to Nyquist in practice; perhaps require the smooth radial field to have characteristic variation length

[
\ell \gtrsim 2h.
]

A simple local diagnostic is

[
\ell_1(r)=
\left|
\frac{V(r)}{V'(r)}
\right|,
]

or

[
\ell_2(r)=
\sqrt{
\left|
\frac{V(r)}{V''(r)}
\right|
}.
]

Choose (r_s) roughly where both become comfortably larger than (h).

So the decomposition means literally:

> Everything whose curvature is too large for the mesh becomes a PIC correction.

That seems exactly aligned with what you want.

---

# 13. Runtime data structure becomes extremely simple

For the mesh:

```text
float mesh[nx*ny*nz]
```

or B-spline coefficients.

For the atoms:

```text
float4 atomPos
float coreCoeff[NATOM][NM]
```

plus a 2D or 3D cell list.

For a molecular surface I would probably use **2D XY buckets**, because the query is above the surface and your atoms occupy a comparatively thin slab.

Cell size could be approximately

[
L_{\rm cell}\sim r_{\max}.
]

A query inspects a (3\times3) neighborhood of cells.

For each candidate atom:

```text
dr = p - atomPos[i]
r2 = dot(dr,dr)

if r2 < rmax2:
    r = sqrt(r2)
    evaluate compact radial modes
    E += ...
    F += ...
```

And once per PP force evaluation:

```text
E,F = coarseGrid(p)
E,F += PICcores(p)
```

This is exactly the kind of work GPUs are good at.

---

# 14. Compared with the current ContactSurface

The conceptual change is quite radical.

Current:

[
\boxed{
\text{coarse-ish }xy
\times
\text{global }s\text{-modes}
}
]

with every coefficient field responsible for essentially the whole molecule.

New:

[
\boxed{
\text{smooth coarse 3D mesh}
+
\text{compact radial atomic details}
}
]

The new representation is more local in every sense:

* local support in real space;
* local atom lookup through PIC;
* no global fit coupling atomic detail into far-field modes;
* no (h_0(x,y)) dependence in the far field;
* memory scales as
  [
  O(N_{\rm mesh}+N_{\rm atom}N_m);
  ]
* scan cost scales as
  [
  O(N_{\rm pixels}N_z N_{\rm local}),
  ]
  independent of total molecule size.

And critically:

[
\boxed{
\text{short-range fitting errors cannot propagate to the far field}
}
]

because

[
V_i^{\rm core}(r\ge r_{\max})=0
]

identically.

That is the guarantee the present global separable fit fundamentally lacks.

---

## The version I would prototype first

I would make the first prototype intentionally simple:

[
h_{\rm mesh}=1.0\ {\rm Å},
\qquad
r_{\max}=6\ {\rm Å},
]

perhaps

[
r_{\min}=R_0-0.5\ {\rm Å},
]

and 4–6 compact radial functions

[
(1-x)^4,,
(1-x)^8,,
(1-x)^{16},,
(1-x)^{32},\ldots
]

per atomic radial potential.

Before rasterizing an atom onto the coarse mesh, replace its true combined Morse+Q potential inside (r_s) by a (C^2) smooth soft-core continuation. The difference between the true radial potential and that continuation becomes the compact PIC core.

That eliminates the need for the long-range fit to make any decision about atomic-scale structure at all.

I think that should be the new conceptual baseline. It is essentially a **particle–mesh + compact real-space residual method for arbitrary probe–sample potentials**, with PME as the closest computational analogy, but optimized for a static AFM force field rather than precision electrostatics.


---

# Devin review + definite plan (2026-08-10)

## Verification of ChatGPT claims against code

I verified every technical claim in the ChatGPT analysis against the actual codebase. All are correct:

| Claim | Code verified | Verdict |
|-------|---------------|---------|
| `getMorsePLQH` decomposes EP/EL/EQ separately | `kernels/Forces.cl:235-249` — EP=e^{2K(r-R0)}, EL=-2e^{K(r-R0)}, EQ=q/r, returned as separate terms in one float4 | ✅ Correct |
| Doubling powers t^4,t^8,...,t^128 | `kernels/contact_surface.cl:123-147` — `tpow = tpow * tpow` each iteration | ✅ Correct |
| `cs_eval_separable_fe_at` uses one xy grid for all z modes | `kernels/contact_surface.cl:334-399` — single `ncx/ncy/dx/dy` for all `kz` | ✅ Correct |
| `fit_separable_cg` uses CG on normal equations (A^TWA) | `spammm/surfaces/ContactSurface.py:537-598` — computes A^Tb and A^T(Ap), CG on squared system | ✅ Correct |
| `clamp_occupancy` does NOT test relaxed trajectories | `spammm/SPM/AFM_utils.py:4277-4287` — uses nearest-sampled h0 and nominal scan z, `tip_disp=None` | ✅ Correct |
| PIC is purely atom-centered, no coarse B-spline long-range | `kernels/contact_surface.cl:647-760` — `cs_pic_eval_tile16` loads atoms from buckets, sums `poly_z_doubling_modes(r,...)` per atom, no B-spline grid | ✅ Correct |
| R2.7 doc's claim "PIC already uses coarse long-range + atom short-range" is wrong | Confirmed — PIC has no long-range B-spline component | ✅ ChatGPT §8 is right |

## The problem identified

There are **three independent algorithmic defects**, each contributing to the far-field bumpiness. They compound each other.

### Defect 1: Spectral mixing via global doubling powers + unrestricted xy bandwidth

The z basis uses global compact-support polynomials with doubling powers: φ_k(s) = (1-s/R)^{m·2^k}. For m=4, nz=6 the powers are 4, 8, 16, 32, 64, 128.

At far z (s ≈ R/2), only mode 0 (t^4) has meaningful amplitude — all higher modes underflow to ~0. The global CG fit minimizes *total* error across all z, so it exploits the t^4 mode's full-z activity to reduce near-field residuals by injecting atomic-scale xy structure into C_0(x,y). That same atomic structure then survives as the *only* spatial pattern at far z, creating unphysical corrugation.

**Root cause confirmed by R2.6 gates:** poly_R sweep, bspl_dx sweep, CG iteration sweep, K_LAT sweep, and fit_dx sweep all failed to remove the ~10% far-z error. The error is structured (follows molecular features), not noise.

### Defect 2: h0(x,y) as a far-field coordinate is physically wrong

The contact height h0(x,y) is a sphere envelope: h0 = max_i [z_i + sqrt(R_i² - ρ_i²)]. This is the correct coordinate for the repulsive wall (near field). But the current representation uses s = z - h0(x,y) for *all* z modes, including the far field.

At fixed world z, every variation of h0(x,y) shifts the argument of the long-range basis. Even a perfectly smooth C_0(x,y) acquires atom-scale modulation through φ_0(z - h0(x,y)). And ∇h0 directly enters Fx, Fy via the chain rule. This is the numerical equivalent of using a corrugated curvilinear coordinate system in a region where the physical field is becoming smooth.

The physics confirms this: the far field of a sum of atom pair potentials is smooth in world coordinates. Lateral atomic contrast broadens with height (e^{-αr} ≈ e^{-αz}·exp(-αρ²/2z)). High lateral spatial frequencies necessarily disappear faster with height (electrostatics: V̂(k,z) = V̂(k,0)·e^{-|k|z}). The current tensor-product model permits the opposite — arbitrarily high-k xy structure in C_0 can multiply the slowest-decaying z mode.

### Defect 3: Fitting the sum instead of the physics channels

SPAMMM already computes the Pauli/London/Coulomb decomposition in `getMorsePLQH`:
- EP ~ e^{2αr} — fast decay, atomic contrast, near field only
- EL ~ -e^{αr} — medium decay, some atomic contrast
- EQ ~ 1/r — slow decay, smooth far field

The contact surface fits the *sum* E = EP + EL + EQ with one set of coefficients. The regression must rediscover the decomposition from the total, which is both harder and less constrained than fitting each channel separately. The three channels have fundamentally different spatial scales that the single separable basis cannot represent simultaneously without spectral mixing.

## The target architecture

```
E(x,y,z) = E_long(x,y,z) + E_short(x,y,s)
```

where s = z - h0(x,y) is used ONLY for the short-range contact physics.

Decomposed by physics channel (matching `getMorsePLQH`):

| Channel | Physics | Coordinate | xy resolution | z basis | Support |
|---------|---------|------------|---------------|---------|---------|
| E_short (EP) | Pauli repulsion e^{2αr} | s = z - h0(x,y) | fine (~1 Å) | local cubic B-spline in s | compact, s ∈ [-1, R] |
| E_medium (EL) | London attraction -e^{αr} | world z (or weakly surface-following) | coarse (~2-3 Å) | local cubic B-spline in z | compact or exponential tail |
| E_long (EQ) | Coulomb 1/r | world (x,y,z) | very coarse (~4-5 Å) or analytic | exponential decay e^{-kz} per Fourier mode | infinite tail |

**Memory budget** (for a 30×30×15 Å scan box):
- Current: 1440 coeffs × 8 B = 11.5 KB + h0 (400 × 4 B = 1.6 KB) ≈ 13 KB
- Proposed: EP (20×20 × 12 z-knots × 8 B = 38 KB) + EL (10×10 × 8 × 8 = 6.4 KB) + EQ (5×5 × 4 × 8 = 0.8 KB) + h0 (1.6 KB) ≈ 47 KB
- Full GridFF at 0.1 Å: 300×300×150 × 16 B = 216 MB; at 0.2 Å: 100×100×75 × 16 B = 12 MB

The proposed architecture is ~47 KB — still negligible vs GridFF, only 3.6× larger than current, but physically correct.

## Definite implementation plan

### Phase 0 — Diagnostics (prove the mechanism, no code changes to production)

**Goal:** Confirm spectral mixing is the root cause before any redesign. Cheap, fast, decisive.

**0a. SVD oracle** — Sample the brute Morse field on a surface-following tensor grid M[q,j] = E(x_q, y_q, h0(x_q,y_q) + s_j). Reshape (x,y)→q, compute SVD. Reconstruct with ranks 1..8. If rank-6 SVD gives <1% error while current 6 powers give ~10%, the separable concept is fine and the z basis is the problem. If even optimal rank-6 stalls near 10%, the coordinate system (h0) is the problem.

**0b. C_0 spectrum** — Extract the fitted coefficient map C_0(x,y) from an existing fit. Plot it as a 2D map + 2D power spectrum. If C_0 carries atomic-scale structure (high-k power), spectral mixing is confirmed directly.

**0c. Low-pass C_0 test** — Low-pass filter only C_0 (Gaussian σ=2 Å or truncate Fourier modes above k_cutoff). Reevaluate the field. If far-field bumpiness disappears while near-field parity barely changes, the mechanism is causally proven.

**0d. Per-channel SVD** — Repeat 0a but for EP, EL, EQ separately (using `getMorsePLQH` channel outputs). This tells us the intrinsic rank of each channel and confirms they should be fit separately.

**Artifacts:** `debug/contact_redesign/phase0/` — SVD spectra, C_0 maps, low-pass comparison plots.
**Verification:** RMSE vs rank plots, C_0 power spectra, before/after low-pass Fz maps.
**Time:** 1 diagnostic script, ~2 hours.

### Phase 1 — Two-pass fit with constrained far mode (quick fix, NO kernel changes)

**Goal:** Validate that separating far/near fitting improves far-field smoothness, using only Python orchestration.

**What:** In `fit_contact_surface`, call `fit_separable_cg` twice:
1. **Pass 1:** Fit only mode 0 (nz=1) using far-z samples only (s > s_split, e.g. s > 2 Å). Use a coarse xy B-spline grid (bspl_dx_far = 2.5 Å) OR add Tikhonov regularization λ₀·||∇²C_0||² to the CG loss. This produces a smooth C_0.
2. **Lock** C_0. Prolong/interpolate it onto the fine coefficient grid.
3. **Pass 2:** Fit modes 1..5 (nz=5) to the residual (E_ref - C_0·φ_0) using all z samples and the fine xy grid. These modes capture atomic contrast and are only active at near z.

**Why it works:** The far mode can no longer absorb near-field structure. The near modes have compact support in z (they're zero at far s), so their xy structure doesn't leak into the far field.

**Code changes:** `spammm/surfaces/ContactSurface.py` — add `fit_separable_cg_twopass()` method. No kernel changes. The existing `cs_eval_separable_fe_at` evaluates the sum of all modes regardless of how they were fitted.

**Verification:**
- Far-z RMSE should drop from ~10% to <3%
- Near-z parity should be preserved (RMSE Fz < 0.05)
- 2D Fz maps at h=4-5 Å should be smooth (no atomic corrugation)
- Compare against Phase 0 SVD oracle lower bound

**Artifacts:** `debug/contact_redesign/phase1/` — two-pass vs single-pass Fz maps, RMSE vs height.
**Time:** ~4 hours (Python only).

### Phase 2 — PLQ channel separation (physics fix, moderate kernel changes)

**Goal:** Fit EP, EL, EQ channels separately instead of the sum. This is the core physics fix.

**What:** Extend `SeparableParams` to hold three independent separable fields:
- `sep_EP`: fine xy grid (~1 Å), local z basis in s=z-h0, compact support
- `sep_EL`: coarse xy grid (~2.5 Å), z basis in world z, medium support
- `sep_EQ`: very coarse xy grid (~5 Å) or analytic point-charge Coulomb, z basis in world z, infinite tail

The brute reference `eval_brute` already computes EP, EL, EQ separately via `getMorsePLQH`. The fit samples each channel independently.

**Kernel changes:**
- `cs_eval_separable_fe_at` is already per-sep; just call it 3 times and sum the results. OR add a `cs_eval_plq_fe_at` that evaluates all 3 channels in one kernel call (avoids 3× kernel launch overhead).
- `relaxStrokesTiltedContact` sums forces from all 3 channels.

**Key design decisions:**
- EQ can use analytic point-charge Coulomb (sum over atom charges) instead of a grid — this is the most physically correct and uses zero stored coefficients. The atom positions and charges are already available. This eliminates the Coulomb representation error entirely.
- EL uses world-z coordinate (not s=z-h0) because the London attraction is not surface-following at medium range.
- EP uses s=z-h0 because the repulsive wall IS surface-following.

**Verification:**
- Per-channel parity: EP, EL, EQ each match brute reference independently
- Total E/Fz parity vs brute Morse+Coulomb
- Far-field smoothness: 2D Fz maps at h=4-5 Å must be smooth
- Memory: report coefficient count per channel, total vs current

**Artifacts:** `debug/contact_redesign/phase2/` — per-channel E(z)/Fz(z) curves, 2D maps per channel, total parity.
**Time:** ~1-2 days (kernel + Python).

### Phase 3 — Replace doubling powers with local B-spline in s (kernel changes)

**Goal:** Eliminate vertical spectral mixing by using local-support z basis.

**What:** Replace `poly_z_doubling_modes` (global t^4, t^8, ..., t^128) with a 1D nonuniform cubic B-spline basis in s:
```
E(x,y,s) = Σ_{i,j,k} c_{ijk} B_i(x) B_j(y) B_k(s)
```

The key difference is **local support**: a coefficient at s=4 Å cannot affect the field at s=0.3 Å. Vertical spectral pollution disappears by construction.

**Nonuniform z knots:** Dense near contact (s=0, 0.2, 0.5, 1.0, 1.5, 2.0 Å) and progressively wider in the tail (3.0, 4.0, 6.0, 8.0 Å). This gives multiresolution resolution: fine where the repulsive wall needs it, coarse where the field is smooth.

**Performance:** A cubic spline uses only 4 active z coefficients at any point. The current kernel evaluates 16 xy coefficients × 6 z modes = 96 contributions. A tricubic spline needs 4×4×4 = 64 contributions, regardless of how many z knots. So evaluation becomes **faster** while allowing more z knots.

**Kernel changes:**
- Replace `poly_z_doubling_modes` with `bspline3_s(s, knots, &B[4], &dB[4])` — 1D cubic B-spline evaluation on nonuniform knots.
- `cs_eval_separable_fe_at` inner loop changes from `for kz in range(nz): phi[kz]` to `for kz in range(4): B[kz]` — 4 active z basis functions instead of 6 global modes.
- Coefficient layout changes from `[ncx, ncy, nz]` to `[ncx, ncy, nknots]` — same structure, just different z dimension.

**This applies to EP channel only** (the short-range one). EL and EQ can keep simpler z bases (exponential or low-order polynomial in world z).

**Verification:**
- z-curve parity: E(z)/Fz(z) at atom centers, gaps, worst pixels
- No spectral mixing: C_k(s_knot) maps should NOT show atomic structure at far s_knots
- Performance: kernel eval time vs current (should be faster)
- Memory: coefficient count vs current

**Artifacts:** `debug/contact_redesign/phase3/` — z-curve parity, per-knot coefficient maps, timing comparison.
**Time:** ~2-3 days (kernel rewrite of z basis + Python fit adaptation).

### Phase 4 — Trajectory safety (fix clamp_occupancy)

**Goal:** Make the clamp check test actual relaxed trajectories, not nominal grid.

**What:**
1. Add per-pixel telemetry to `relaxStrokesTiltedContact` kernel: output `min_s`, `n_s_lt_0`, `min(poly_R - s)`, convergence residual, final PP position. Store in a side buffer.
2. After scan, compute `clamp_occupancy_real` = fraction of pixels where `min_s < 0` over the actual relaxation trajectory.
3. Fail loud (not silent): if `clamp_occupancy_real > 0`, print warning with the worst pixel coordinates and s values.
4. Extend EP basis to negative s (s ∈ [-1, R] instead of [0, R]) so the repulsive wall is represented, not clamped to zero force.
5. Add analytic repulsive continuation below the lower training boundary: for s < s_min_fit, extrapolate as E(s) = E(s_min_fit) + E'(s_min_fit)·(s - s_min_fit) + ½·k·(s - s_min_fit)² with k from the fit, rather than a zero-force plateau.

**Code changes:**
- `kernels/contact_surface.cl` — `relaxStrokesTiltedContact` outputs telemetry buffer
- `spammm/SPM/AFM_utils.py` — `run_contact_pp_afm` reads telemetry, computes real clamp occupancy
- `spammm/surfaces/ContactSurface.py` — extend fit z-range to include negative s

**Verification:**
- `clamp_occupancy_real` reported for PTCDA, pyridine, benzene scans
- If any pixel has s<0, the warning prints with coordinates
- Fz(z) curves in the clamped region show repulsive wall (not zero force)

**Artifacts:** `debug/contact_redesign/phase4/` — telemetry maps, clamp comparison.
**Time:** ~1 day.

### Phase 5 — Solver improvement (optional, after representation is fixed)

**Goal:** Replace normal-equations CG with LSQR to avoid squaring the condition number.

**What:** Replace `fit_separable_cg` (which does CG on A^TWA c = A^TWb) with LSQR/LSMR, which operates directly through the matrix-free Av and A^Tv operators. LSQR supports damped least squares (regularization) without explicitly forming normal equations.

**Cheaper alternative:** Build the 6×6 (or nknots×nknots) weighted Gram matrix of the z basis, orthogonalize/whiten the z modes, fit in the whitened basis, transform coefficients back. This requires essentially no scan-kernel change — only the fit routine changes the basis it works in.

**Verification:**
- Fit convergence: iterations to reach same RMSE
- Condition number: compare A^TWA vs A
- Fit RMSE: should be equal or better with fewer iterations

**Time:** ~0.5-1 day.

## Priority order and dependencies

```
Phase 0 (diagnostics) ──┐
                         ├─→ Phase 1 (two-pass, Python only) ──→ verify
                         │
                         └─→ Phase 2 (PLQ separation, kernel) ──→ Phase 3 (local B-spline, kernel)
                                                                    │
                                                                    └─→ Phase 4 (trajectory safety)
                                                                         │
                                                                         └─→ Phase 5 (solver, optional)
```

- **Phase 0 is mandatory first** — it proves the mechanism and tells us whether the fix is in the basis (Phase 3) or the coordinate system (Phase 2) or both.
- **Phase 1 is the quickest validation** — if two-pass fit fixes the bumpiness, we know the approach works before investing in kernel changes.
- **Phase 2 is the core physics fix** — separating PLQ channels is the most impactful change.
- **Phase 3 builds on Phase 2** — local B-spline in s applies to the EP channel specifically.
- **Phase 4 is independent** — can be done in parallel with Phase 2-3.
- **Phase 5 is optional** — only needed if the solver becomes the bottleneck after the representation is fixed.

## What we STOP doing

- ❌ Sweeping `bspl_dx`, `poly_R`, `fit_dx`, `n_iter`, `K_LAT` — R2.6 gates proved these don't help
- ❌ Making the uniform B-spline grid finer — the problem is not grid resolution
- ❌ Trying to find the "perfect set of six global powers" — the global doubling-power basis is the problem
- ❌ Post-fit smoothing of coefficients — treats the symptom, not the cause
- ❌ Claiming PIC already has a coarse long-range component — it doesn't (verified in code)

## Success criteria

The contact surface is a viable cheap alternative to full GridFF when:
1. **Far-field smoothness:** 2D Fz maps at h=4-5 Å show no atomic corrugation (relative error < 3%)
2. **Near-field parity:** E(z)/Fz(z) curves match brute Morse within RMSE Fz < 0.05 at atom centers
3. **Memory:** total coefficients < 100 KB for a typical molecule (vs 12-216 MB for GridFF)
4. **Trajectory safety:** `clamp_occupancy_real = 0` for standard scans, with loud failure if violated
5. **USER confirms** qualitative match vs GridFF for PTCDA, pyridine, benzene
