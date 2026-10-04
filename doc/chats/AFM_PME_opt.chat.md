# USER

We were experimenting with different ways how to further speed up probe-particle model on GPU (OpenCL), in fomulation of "contact surface", we tried different thgs, from low-resolution cubic splines, particle-in-cell, fited atomic potential etc.&#x20;

in the ned what works best is PME (from particle-mesh-ewald is the name but the algorithm is probably not nthe same as canonical PME for electrostatics)



Sortly the problem is we have large number of atoms and probe particle which interact with them localy. Normlally we use grid-forcefield to describe it, but one can use also grid-acceleration with cutoffs (the PIC - particle in cell method, where we map which atoms affect each cell). We must consider what is the fastest way to evaluate the grid-representation, and what is the fasters way to evaluate it. PME was optimize to provide small memory foodprint (like compresed grid with PME representation being like 100kb per cubic nanometer, while GridFF with 0.1A voxels being like 100MB). But fitting PME is rather slow.&#x20;



Now I was thinking we should make mofified kernel which is even faster to evaluate that it make sure each workgroup read read to local memory only relevant block of data, in particlar asume that the blocks of coefficients are larege cubes (voxels) or even better maybe triangular prisma ( interval in z, triangle in x,y ), and on it we do some polynominal interpolation, but for all samples within workgroup we do not interpolate independnetly accesing random grid position like GridFF (which is memory bottlenecks) but we have pre-loaded all necessary coefs in local memory. If the cell (prisma) is large enough we can guarnatee that all threads (like trajectrois of probe particle coresponting to each pixel assigned to this workgroup) are enclosed by given box, or by few boxes.&#x20;



I'mportant is also what is efficient interpolation algorithm. I can easily evaluate energy and derivatives (forces) from each atom in each corner of the cell. One can think about constructiong like Hermite spline (tricubic). But I remember there was problem with that it needed mixed derivatives like dddE/dx dx dz, which I do not have. Perhaps we can build some other interpolation which does not need them? Can be the triangular grid topology more efficient in that respect than rectangular? E.g. using besier triangle?&#x20;

[https://en.wikipedia.org/wiki/B%C3%A9zier_triangle](https://en.wikipedia.org/wiki/B%C3%A9zier_triangle)



A bit problem is that this triangle is less compact than square, so the pixels inside will espcape, so we will probably need still halo voxel stored in local memory. Critical for efficiency is to estimate how many sample points we nee to load to local memory, I guess if we use like 0.5A size of voels we realsitically need only nearest neibhor voxels, but this may still be too many in 3D.





---



if you need there are relevant references in our SPAMMM repo:



I'll search the codebase for PME (Particle Mesh Ewald) simulation code — OpenCL kernels and Python drivers.\
\
Let me check the kernel names in \`contact_surface.cl\` and see how the program is loaded, plus quickly verify what \`PME.cl\`/\`PME8.cl\` are for (they may be a different PME — the quantum Pauli solver).\
\
Found it. Heads-up on naming: "PME" appears twice in this repo with different meanings — \`kernels/PME.cl\`/\`PME8.cl\` are the \*\*Pauli Master Equation\*\* quantum STM solver (\`spammm/quantum/PauliSolverCL\*.py\`), not what you want. The \*\*contact-surface PME\*\* (split long/short-range potential field) lives in \`contact_surface.cl\` + \`spammm/surfaces/\`.\
\
\## OpenCL kernels\
\
\<ref_file file="/home/prokophapala/git/SPAMMM/kernels/contact_surface.cl" /> (2207 lines) — the PME-relevant kernels:\
\
\- \`fillContactPMEMeshVL\` — raster long-range part V_L onto coarse mesh (WG + local memory)\
\- \`evalContactPME\`, \`evalContactPMELocal\` — evaluate (E, F) at query points (bucket vs local-mem variants)\
\- \`relaxStrokesTiltedContactPME\`, \`relaxStrokesTiltedContactPMELocal\`, \`relaxStrokesTiltedContactPMELocalQN\`, \`relaxStrokesTiltedContactPMELocalSph\` — tip relaxation during scans\
\- Supporting: \`evalRadialPIC\`, \`cs_pic_eval_tile16\`/\`cs_pic_Av\`/\`cs_pic_Atv\*\`, \`evalSeparableBsplinePoly\`, \`cs_sep_A\*\` (separable B-spline operators), \`cs_brute_plqh_points\`, \`cs_brute_afm_morse_c_points\` (host-parallel references)\
\
\## Python modules (\`spammm/surfaces/\`)\
\
\- \<ref_file file="/home/prokophapala/git/SPAMMM/spammm/surfaces/PMESplit.py" /> — \`SplitParams\`, long/short-range split math (modes: \`paw\`, \`hermite\`, \`plateau\`, \`rho\`, \`softcore\`), \`precompute_split_cache\`\
\- \<ref_file file="/home/prokophapala/git/SPAMMM/spammm/surfaces/CoarseMesh.py" /> — \`build_coarse_mesh\`, B-spline prefilter (\`\_prefilter_3d\`)\
\- \<ref_file file="/home/prokophapala/git/SPAMMM/spammm/surfaces/PICCore.py" /> — \`fit_core_1d\`, compact radial core fit\
\- \<ref_file file="/home/prokophapala/git/SPAMMM/spammm/surfaces/ContactSurface.py" /> — \`ContactPMEParams\` (SSOT container), \`build_pic_buckets\`, host-side eval; also compiles \`contact_surface.cl\` itself\
\
\## Driver (orchestration + GPU launches)\
\
\<ref_file file="/home/prokophapala/git/SPAMMM/spammm/SPM/AFM.py" /> — in \`AFMulator\`:\
\
\- \`fit_contact_pme\` — orchestrates split → coarse mesh → core fit → PIC buckets (\~line 1608)\
\- \`\_pme_build_coarse_mesh_gpu\`, \`\_pme_upload_resident\`, \`\_pme_workgroup_size\`, \`\_pme_select_core_backend\`\
\- \`eval_contact_pme\` / \`\_pme_eval_gpu\` / \`\_pme_eval_python\` — CPU/GPU eval dispatch (\~line 1881)\
\- \`run_scan_contact_pme\` — PP-AFM scan on the PME backend (\~line 2010)\
\- Kernel compile: \`contact_surface.cl\` loaded at \~line 327\
\
\## Tests\
\
\- \<ref_file file="/home/prokophapala/git/SPAMMM/tests/SPM/test_afm_contact_surface.py" />\
\- \<ref_file file="/home/prokophapala/git/SPAMMM/tests/SPM/testplot_afm_contact_surface.py" />\
\- \<ref_file file="/home/prokophapala/git/SPAMMM/tests/testplot_contact_surface.py" />\
\- Debug harnesses under \`debug/testplot_contact_surface/\` (e.g. \`agent_4_relaxation/relax_diag.cl\` is a standalone diagnostic kernel copy)\
\
Want me to dig into any specific part (e.g. the mesh-eval data flow, the relax kernels, or the fit pipeline)?



[https://github.com/ProkopHapala/SPAMMM](https://github.com/ProkopHapala/SPAMMM)


---

# Chat GPT 5.6 sol

Yes. After looking at the current `contact_surface.cl` and the PME host code, I think this is a very promising next optimization. But I would **not switch to triangular prisms/Bézier triangles first**. I think the largest gain is available by changing the **workgroup spatial organization and caching**, while initially keeping essentially the existing cubic B-spline representation.

### What the present kernel is actually doing

The long-range `V_mesh` part is currently a scalar cubic cardinal B-spline. Every force evaluation touches a \(4\times4\times4=64\)-coefficient stencil, although the implementation is already reasonably clever: \(z\) is contiguous and loaded as 16 `float4`s. contact_surface

The `Local` PME variant, however, only puts the **short-range atomic part** into local memory:

\[
M_{\rm core}=N_{\rm atom}(16+5\times4)=36N_{\rm atom}\ {\rm bytes}.
\]

The mesh coefficients are still read from global memory on every PP force evaluation. More importantly, your FIRE loop deliberately has **no barriers after the initial preload**, because individual PP trajectories converge after different numbers of iterations. contact_surface

That immediately suggests the architecture I would use:

> **Preload one fixed spatial mesh brick covering the entire possible motion of the whole workgroup, once before FIRE starts. Never reload it inside FIRE.**

This preserves the very useful asynchronous/divergent relaxation you already have.

## The local mesh brick is surprisingly small

Suppose one workgroup corresponds to a compact \(T_x\times T_y\) pixel patch. Let the total envelope of all PP positions during the entire stroke be

\[
W_x,\quad W_y,\quad W_z.
\]

For a cubic B-spline with spacing \(h\), the union of all required stencils is bounded approximately by

\[
N_x \le \left\lceil\frac{W_x}{h}\right\rceil+4,
\qquad
N_y \le \left\lceil\frac{W_y}{h}\right\rceil+4,
\qquad
N_z \le \left\lceil\frac{W_z}{h}\right\rceil+4.
\]

So the local-memory cost is simply

\[
M_{\rm mesh}=4N_xN_yN_z\ {\rm bytes}.
\]

Take a fairly pessimistic example: an \(8\times8\) workgroup, scan pixel spacing \(0.1\) Å, PP allowed to deviate laterally by \(\pm0.5\) Å, and a 4 Å vertical stroke with another \(\pm0.25\) Å vertical relaxation. Then

\[
W_x=W_y=7(0.1)+1.0=1.7\ {\rm Å},
\qquad W_z=4.5\ {\rm Å}.
\]

At your **current \(h=1\) Å mesh**,

\[
N=(6,6,9), \qquad M_{\rm mesh}\approx1.3\ {\rm kB}.
\]

Even at **\(h=0.5\) Å**,

\[
N=(8,8,13), \qquad M_{\rm mesh}\approx3.3\ {\rm kB}.
\]

That is tiny. Even allowing \(\pm1\) Å lateral PP motion still leaves you at only a few kB.

So I think your intuition is correct: you do **not** need to worry much about the 3-D halo for the mesh. The halo sounds large as “nearest-neighbor voxels in 3D”, but because the grid is coarse and all 32/64 trajectories are spatially coherent, the *union* of their stencils is small.

And the gain can be enormous. Instead of, schematically,

\[
64\times N_{\rm thread}\times N_{\rm force-evals}
\]

global scalar coefficient reads, you do perhaps 500–1000 global reads **once per entire workgroup**, followed by thousands of local-memory accesses.

There will of course be some help from the hardware cache already, so this won't translate literally into that bandwidth ratio, but this changes the access pattern from “hope the cache recognizes the reuse” to explicit cooperative reuse.

## There is probably an even bigger optimization hiding in the core

Currently `relaxStrokesTiltedContactPMELocal` loads **all atoms and all five core coefficients for all atoms** into local memory and every thread loops over all atoms, rejecting most by \(r^2>r_b^2\). contact_surface

For 500 atoms that costs about 18 kB/workgroup; for 1000 atoms, 36 kB. That is much more important for occupancy than the 1–5 kB mesh brick.

I would therefore make the same spatial tiling apply to the compact atomic core:

\[
\text{WG spatial envelope}
\quad\longrightarrow\quad
\{\text{atoms whose }r_b\text{-sphere intersects envelope}\}.
\]

Build this list once on the host when the scan is configured. Then each workgroup cooperatively loads **only its relevant core atoms** and their five coefficients.

For a large molecule, this could change an inner loop over e.g. 500–2000 atoms into one over perhaps tens of nearby atoms. And it frees enough shared/local memory that you can keep 2–4 workgroups resident rather than one.

So conceptually I would turn the current two alternatives

\[
\text{all atoms in local memory}
\quad\text{vs}\quad
\text{global PIC buckets}
\]

into a third alternative:

\[
\boxed{\text{WG-neighbor list loaded once into local memory}}
\]

which is almost ideally matched to an AFM scan.

---

# I would keep the rectangular grid

Your recollection about tricubic Hermite is correct. A general tensor tricubic

\[
P(x,y,z)=\sum_{i,j,k=0}^3 a_{ijk}x^iy^jz^k
\]

has \(4^3=64\) coefficients. Eight corners give only

\[
8(1+3)=32
\]

numbers if you know \(E,E_x,E_y,E_z\). The conventional full tricubic Hermite specification needs

\[
E,\ E_x,E_y,E_z,\ E_{xy},E_{xz},E_{yz},E_{xyz}
\]

at every corner, giving \(8\times8=64\) constraints. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2772415822000281?utm_source=chatgpt.com)

You *can* set the mixed derivatives to zero, estimate them, or define some reduced 32-DOF Hermite polynomial. But then it is no longer general tricubic interpolation. For your particular field this is especially undesirable because a radial atomic potential has genuine cross derivatives such as \(E_{xy}\), \(E_{xz}\), etc. They are not small in general.

And storing `float4(E,Fx,Fy,Fz)` at every grid node already costs four times as much as your scalar B-spline coefficient lattice.

### A triangular Bézier grid doesn't really fix this

A degree-3 Bézier triangle has 10 coefficients. [Springer Link](https://link.springer.com/article/10.1007/s00366-012-0278-6?utm_source=chatgpt.com) Extruding it with a cubic \(z\) polynomial would therefore require

\[
10\times4=40
\]

coefficients for one triangular prism, versus 64 for a tensor cubic brick.

That initially sounds attractive. But there are **two triangles per square**, so if coefficients belong to individual patches you have effectively 80 coefficient slots per square-prism footprint rather than 64. More importantly, \(C^1\) continuity between neighboring Bézier patches imposes additional relations between control points across the common edge; matching the vertex values alone is not enough. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/016783969090028P?utm_source=chatgpt.com)

And your physical workload is rectangular: an \(8\times4\) or \(8\times8\) patch of AFM pixels. The union of all triangles touched by such a WG is basically the same rectangular bounding box anyway. So the supposed geometrical compactness largely disappears when you consider **WG-level caching rather than one query**.

I would only revisit triangular grids if isotropy of interpolation error becomes a demonstrated problem. If that happens, regular triangular **box splines** are mathematically more interesting than independent Bézier triangles.

## One interpolation change I *would* test: quadratic B-splines

This is much more attractive.

A quadratic tensor B-spline has support

\[
3\times3\times3=27
\]

rather than 64 points. A uniform quadratic B-spline is \(C^1\), while cubic is \(C^2\). [UW Computer Sciences](https://pages.cs.wisc.edu/~yw/CS559W24CW5.html?utm_source=chatgpt.com)

Therefore:

\[
E_{\rm quadratic}\in C^1
\quad\Rightarrow\quad
\mathbf F=-\nabla E \text{ is continuous},
\]

which may already be completely sufficient for PP relaxation.

The current cubic gives

\[
E_{\rm cubic}\in C^2
\quad\Rightarrow\quad
\mathbf F\in C^1.
\]

That is nicer, but possibly unnecessary.

An especially interesting compromise for AFM is

\[
\boxed{Q_2(x)\times Q_2(y)\times Q_3(z)}
\]

which needs only

\[
3\times3\times4=36
\]

coefficients per evaluation.

That preserves cubic smoothness in \(z\), where \(F_z\) and ultimately \(\Delta f\) are particularly sensitive, while reducing the stencil from 64 to 36 coefficients. In \(x,y\), energy is still \(C^1\), so lateral force itself remains continuous.

I think **36-tap Q2×Q2×Q3** is substantially more promising than Bézier triangles.

One nuisance is that even-degree cardinal B-splines have the usual half-cell centering convention, so the host prefilter/indexing needs a little care. But that is a one-time implementation detail, not a fundamental problem.

## The kernel structure I would implement

I would make this a new kernel rather than mutate the current one immediately:

```c
group = get_group_id(0);
lid   = get_local_id(0);

// Descriptor says which mesh brick belongs to this spatial WG
TileDesc td = tiles[group];

// cooperative mesh preload
for(int i=lid; i<td.nmesh; i+=lsize){
    Lmesh[i] = mesh_coeffs[ global_mesh_index(td,i) ];
}

// cooperative local-neighbour preload
for(int i=lid; i<td.natoms; i+=lsize){
    int ia = wg_atom_ids[td.atom0+i];
    LATOMS[i] = atoms[ia];
    LCOEFFS[...] = atom_coeffs[...];
}

barrier(CLK_LOCAL_MEM_FENCE);

// IMPORTANT: no more barriers from here onward
for(z slices){
    for(relaxation iterations){
        fe  = eval_mesh_local(pos,Lmesh,td);
        fe += eval_core_local(pos,LATOMS,LCOEFFS,td.natoms);
        ...
    }
}
```

The important implementation sequence would be:

1. **Reorder the scan into spatial workgroups**, probably \(8\times4=32\) first. Do not simply take 32 consecutive row-major pixels if that makes a long \(3.1\) Å strip. Your existing `pix_map` machinery in the QN path is already close to what is needed.

2. Give every WG a fixed `TileDesc` containing mesh origin/index bounds and a short compact-core atom-list offset. The mesh tile should encompass the complete nominal \(z\)-stroke plus a realistic lateral/vertical PP margin.

3. Cache the existing cubic mesh first. This isolates the effect of global→local memory without changing the mathematical representation.

4. Add the WG-local atom list. I actually expect this to be at least as important as the mesh caching for larger systems.

5. Then benchmark cubic-64 against Q2×Q2×Q3-36 and Q2³-27 using exactly the same reference trajectories.

6. Use a **tight tile with an escape path**, rather than making the tile enormous enough for a pathological 3 Å lateral PP excursion. If a PP leaves the cached brick, either fall back for that trajectory to the existing global mesh/bucket evaluator, or set an escape bit and rerun those exceptional pixels with the old kernel.

The last point is important because it allows something like a ±0.7–1 Å local envelope while retaining absolute correctness.

### What I would benchmark

Separate **evaluator speed** from **full scan speed**. Otherwise a slightly different interpolation changes FIRE/QN convergence count and you cannot tell whether the evaluator became faster.

For the evaluator microbenchmark, use exactly the same \(10^6\) or so positions and compare ns/evaluation. Then run complete PP scans and compare total iterations and final PP positions.

For interpolation quality I would watch not only RMS \(F\), but especially force continuity when crossing cells, maximum lateral-force error near bonds, equilibrium PP displacement error, and the final \(\Delta f(x,y,z)\). A 1–2% force error that is smooth is probably much less harmful here than a smaller RMS error with cell-boundary kinks.

### My expected ranking

If I had to bet before benchmarking:

\[
\boxed{
\text{cubic B-spline + WG local mesh + WG local core atoms}
}
\]

is the safest and probably already very fast.

Then I would try

\[
\boxed{
Q_2(x)Q_2(y)Q_3(z)
}
\]

as the likely best speed/accuracy compromise.

I would **not invest in triangular Bézier prisms yet**. They solve neither the mixed-derivative problem nor the WG-halo problem particularly cleanly, and they make addressing, continuity, fitting and memory layout more complicated precisely where your current tensor B-spline code is beautifully simple.

One further possibility is worth keeping in reserve: if **PME fitting time** remains troublesome after evaluation becomes fast, use an interpolating cubic-convolution/Catmull–Rom mesh from raw nodal \(E\) samples. It retains a 64-point cubic stencil but eliminates the B-spline prefilter. That is a fit-time optimization, whereas the local-brick scheme is the much more important scan-time optimization.

The main thing that surprised me after doing the numbers is that **the mesh tile is not the local-memory problem at all**. At \(h=0.5\) Å it is only a few kB/WG. The existing `36*Natom` local core representation is much more likely to constrain occupancy. So I would design the new kernel around *both* a local coefficient brick **and a spatially pruned local atom list**.


---

# USER

OK, these are very important findings!

1) tiled/PIC/block preload of atoms

You say we load ALL atoms to local memory? That is very bad. I think this is because we were simulating small molecules, where this does not matter but now we want to simulate large systems. In fact primary motivation to split the potential to long range and short range was to allow this optimization - that we need to read only nearby atoms form nearest boxes. That is the spirit of particle-mesh-ewald and why we call it like this. So we abandoned the main point it seems.

2) Spline-mesh block preload - I agree with your direction to keep 0.5-1.0A spline voxels as cubes. But I'm not sure you see it clearly. It seems to me you asume the proble particle sampling this potential moves only up and down on straight line, so you want to load one block with several z-levels? (4A range in z?). But in reality the problem is just the oposite. The movement along z-axis is kinda predicatable, vertical resp radial stiffnes of tip anchor is high, and we can even analytically constrain it on sphere and let it relax only angularly (like let relax lateraly in bylatral forces Fx,Fy and then reconstrain it to sphere which is faster than using angles atan2,sin,cos, we already do this optimization in some kernel), but the lateral bending stiffnes is small and the probe praticle often deflect ~0.5A laterally (sometimes even more but we can constrain that). So key is to pre-load given slice of spline grid with lateral nearest neighbors, while the vertical slices we can load on the fly as we go.

To me it seems reasonable to walk down in z direction as we do it anywas and load new z-slice when needed (we perhas need two z-slices if we are a the boundary...

so the block in working shared local memory should be more like 5x5x5 (nx,ny,nz) ?
I mean normal cubic b-spline need 4x4x4 blocks, and we should add one more to facilitate deflection?


---

# Chat GPT 5.6 sol

Yes. I think your interpretation is more accurate than my previous “cache the whole vertical trajectory” picture.

There are really **two different kinds of locality**:

- **short-range atomic core:** spatial locality in \(x,y\) → load only atoms from nearby PIC cells;
- **smooth mesh:** strong temporal coherence in \(z\) → keep a small lateral slab and stream new \(z\)-planes through it as the scan descends.

And yes: the current `Local` kernel really does preload **all atoms + all five coefficients for every atom**, then every PP loops over all of them and rejects distant ones. contact_surface That was sensible for the small-molecule benchmark but defeats one of the principal reasons for the PME split for large systems.

## 1. The atomic core should go back to genuine PIC

I would replace

\[
\texttt{LATOMS[0..Natom]}
\]

by approximately

\[
\texttt{LATOMS[atoms relevant to this WG tile]}.
\]

Because the substrate is static and the scan geometry is known, I wouldn't even do an expensive dynamic lookup on every PP evaluation. Determine the relevant boxes **once per workgroup**, cooperatively load their atoms into local memory, and use that fixed set throughout the complete vertical stroke.

Something like:

```text
                  r_core halo
             <---------------->

       +--------------------------+
       | nearby PIC boxes         |
       |                          |
       |      +------------+      |
       |      | WG x-y     |      |
       |      | footprint  |      |
       |      +------------+      |
       |                          |
       +--------------------------+
```

The fixed atom set must cover

\[
\text{WG nominal XY footprint}
+\text{maximum PP lateral deflection}
+r_{\rm core}.
\]

Then inside every force evaluation you still perform the actual 3-D distance test.

For a rigid sample this is extremely attractive because the lookup overhead is paid **once per whole stroke**, not once per FIRE iteration.

And I'd probably retain the current 2-D \(xy\) buckets rather than go to full 3-D buckets. The residual is compact and the experiment is a surface scan; full \(r^2\) rejection inside a reasonably small XY candidate set is cheap.

---

# 2. Mesh: yes, think of a moving slab, not a tall block

The ordinary cubic B-spline needs

\[
4\times4\times4
\]

coefficients around one query.

But the useful cached object should be

\[
N_x\times N_y\times N_z^{\rm cache},
\]

where \(N_z^{\rm cache}\) is only about **4–5 planes**, and the cache slides downward.

The reason this works especially well for PP-AFM is exactly what you say: \(z\) is very predictable. With the spherical bond constraint, lateral deflection actually changes \(z\) only quadratically:

\[
\delta z
= L-\sqrt{L^2-\rho^2}
\simeq \frac{\rho^2}{2L}.
\]

For \(L=3\) Å,

\[
\rho=0.5\ {\rm Å}
\Rightarrow \delta z\simeq0.042\ {\rm Å},
\]

and even

\[
\rho=1.0\ {\rm Å}
\Rightarrow \delta z\simeq0.17\ {\rm Å}.
\]

So at mesh spacings \(h=0.5-1\) Å, the PP practically stays in the expected \(z\) cell.

That is a beautiful property to exploit.

---

## Is `5×5×5` enough?

**For one trajectory whose grid-cell index can move by at most one cell in each direction, yes, roughly.**

A cubic B-spline query in cell \(i\) uses

\[
[i-1,i,i+1,i+2].
\]

If the PP may cross into adjacent cell \(i+1\), the union is

\[
[i-1,\ldots,i+3],
\]

i.e. **5 nodes**.

So the intuition

> cubic needs 4; add one extra node for wandering

is correct.

But there is one important qualification:

### The workgroup footprint also contributes

If all threads in a workgroup corresponded to the same nominal \(x,y\) cell, then `5×5` laterally is plausible.

But if, for example, a workgroup is an `8×4` patch of pixels at spacing \(0.1\) Å, its nominal lateral width is already

\[
0.7\times0.3\ {\rm Å}.
\]

Add possible \(\pm0.5\) Å PP bending:

\[
W_x \approx 1.7\ {\rm Å},\qquad
W_y \approx 1.3\ {\rm Å}.
\]

At \(h=0.5\) Å that spans several mesh cells.

The exact node count along an axis is

\[
\boxed{
N = 4 + (i_{\max}-i_{\min})
}
\]

where \(i_{\min},i_{\max}\) are the minimum/maximum B-spline **cell indices** reached by any PP in that WG.

So `5×5` means \(i_{\max}-i_{\min}=1\).

For a real WG we might instead get something like

\[
7\times6
\quad\text{or}\quad
8\times7.
\]

But this is actually **not a problem at all**.

An `8×8×5` scalar tile is only

\[
8\cdot8\cdot5\cdot4
=1280\ {\rm bytes}.
\]

Even `10×10×5` is only 2 kB.

So I would **not fight very hard to make it exactly `5×5×5`**. The atomic cache will cost much more local memory than this.

---

# 3. I think `Nx × Ny × 5` is the sweet spot

I would therefore organize it like this:

```text
z

       plane k+3       cached
    ----------------
       plane k+2       cached
    ----------------
       plane k+1       cached
    ----------------
       plane k         cached
    ----------------
       plane k-1       cached
    ----------------
          ↓ scan

          load one new plane
          discard oldest plane
```

Each plane is maybe `8×8`, `8×7`, etc.

A circular local-memory buffer:

```c
__local float Lmesh[5][NY][NX];
```

or flattened equivalent.

The initial load costs perhaps 300 floats.

Then when the nominal B-spline \(z\)-cell changes by one:

```text
old:

[k-1 k k+1 k+2 k+3]

downward →

[k-2 k-1 k k+1 k+2]
 ^ load only this plane
```

So you load only

\[
N_xN_y
\]

new floats per mesh-cell transition.

For `8×8`, that's merely **64 floats = 256 bytes**.

And note: if scan spacing is \(0.1\) Å and mesh spacing is \(0.5\) Å, you only need a new plane approximately once every **five scan-z samples**, not every z sample.

That is almost negligible global traffic.

---

# 4. There is a synchronization subtlety, but it is solvable

The current kernel deliberately has no barriers inside FIRE because threads converge at different iteration counts. contact_surface

But the outer loop

```c
for(iz=0; iz<nz; iz++)
```

is still common to all threads.

Therefore you **can synchronize between z slices**.

Essentially:

```c
for(iz=0; iz<nz; iz++){

    // everyone finished previous z
    barrier(CLK_LOCAL_MEM_FENCE);

    if( nominal_mesh_cell_changed ){
        // cooperative load of new z-plane
        for(int i=lid; i<NXY; i+=lsize)
            Lmesh[new_plane][i] = ...

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // completely divergent FIRE relaxation
    for(int iter=0; iter<NITER; iter++){
        ...
    }
}
```

There is one necessary change: padded/inactive lanes **must not return after the initial preload**, as the current kernel does. They must stay alive as dummy lanes so they continue to participate in subsequent barriers.

So instead of

```c
if(gid >= n_scan) return;
```

use something like

```c
bool active = gid < n_scan;
```

and let inactive lanes skip calculations while still reaching all workgroup barriers.

That seems entirely manageable.

---

# 5. Five z-planes are particularly natural with the sphere constraint

There is an additional nice asymmetry.

If the PP is constrained to a sphere of radius \(L\), lateral deflection makes the PP **rise** relative to the straight-down position:

\[
z=z_{\rm tip}-\sqrt{L^2-x^2-y^2}.
\]

It does not move equally in both \(+z\) and \(-z\).

So if the nominal vertical position corresponds to cell \(k\), you can bias the extra cached plane toward the physically possible side.

Instead of needing

```text
[k-2 ... k+3]     // 6 planes for ± one-cell uncertainty
```

you can likely safely use something like

```text
[k-1 ... k+3]     // 5 planes
```

depending on your scanning direction/index convention.

And retain a cheap safety check:

```c
if( actual_iz outside cached range ){
    escape_flag = 1;
    // optionally global-memory fallback
}
```

I strongly prefer this to making the cache excessively conservative.

---

# 6. I would couple workgroup geometry to the mesh

This now suggests something slightly different from conventional `32 consecutive pixels`.

Choose the workgroup scan patch so that its **physical XY extent is appropriate relative to \(h_{\rm mesh}\)**.

For example, with scan pixels \(d=0.1\) Å:

### \(h=1.0\) Å

An `8×4` WG spans

\[
0.7\times0.3\ {\rm Å}.
\]

Excellent. Even including PP deflection, the union stencil remains small.

### \(h=0.5\) Å

Perhaps `4×8`, `4×4`, or some mapping of 32 lanes to two nearby \(4\times4\) patches deserves benchmarking.

But I would first simply use `8×4`. The mesh cache remains tiny anyway.

The crucial thing is to make the WG **two-dimensional spatially compact**, not a row-major 32-pixel strip.

---

# What I would implement first

I think the new architecture is now quite clear:

\[
\boxed{
\begin{array}{l}
\textbf{WG fixed XY tile}\\
\\
\text{local atom list: nearby PIC cells only}\\
\text{local mesh: }N_x\times N_y\times5\\
\\
\text{for each z:}\\
\quad\text{stream new mesh plane when grid cell changes}\\
\quad\text{barrier}\\
\quad\text{run divergent PP relaxation}\\
\end{array}
}
\]

And **keep the existing cubic B-spline first**.

The mesh local-memory cost is so low that I no longer think reducing 64 → 36 interpolation taps is the first priority. Once those 64 coefficients are coming from shared/local memory, 64 arithmetic operations may be perfectly fine.

The much more severe waste is currently

\[
\boxed{\text{each PP examining every atom}}
\]

in the local PME path. For a large system, restoring the actual PIC idea should probably give the largest gain.

So yes: your proposed `~5×5×5` mental model is basically right, with one refinement: think **`Nx×Ny×5`, where Nx/Ny are the union of the WG pixel footprint + lateral PP bending**, rather than hard-coding 5. In practice I suspect `6–10 × 6–10 × 5`, still only ~1–2 kB, will be the comfortable regime.


---

# USER

1\) ad "Because the substrate is static and the scan geometry is known, I wouldn't even do an expensive dynamic lookup on every PP evaluation."

SURE! naturally! definitely we should not do this lookup every evaluation! that is the whole point! we determined which blocks of atoms short-range cores to read to local memory collectively at the very beginning and then work only with that local memory brok during the steping of PP-relaxation all threads   amortizing that read

2\) "where \\(N_z^{\rm cache}\\) is only about **4–5 planes**, and the cache slides downward."

NO! we you still optimize wrong thing. I said the z-slice is predictable and can be thin. We should read the smallest posible number of slice in z, perhaps only one slice (with necessary halo), we can analytically estimate comnpute the interval our PP can move if we constrain it to sphere and load only coefs relevant for this region.

the region will be flat - larger in x,y than in z.

every time we move iz we can check if we need to load new slice.&#x20;

But optimize the local memory so it accomoda sufficient lateral halo, do not waste the memory for vertial halo which can be loaded each z-tep in predicable way. That is what I wates stress already in my previous message , but you somehow ignored it and stioll want to cache more slices in z than in x,y

3\) you were taling about 3x3x3 blocks for quadratio B-spline, I was speeking about 4x4x4 for cubic B-spline, and you now speak also about 4x4x4 blocks, but what is acrually used in the code? We should preserve that s it seems to work more or less. Lower degree splines as also perhs faster to fit (like when we build te potential representation rather than evaluate it...)


---

# Chat GPT 5.6 sol

Yes — now I understand the geometry you mean. The right cached object is **very flat**: large enough in \(x,y\) to contain the whole WG footprint plus lateral PP bending, but *minimal* in \(z\). I was still over-caching \(z\).

For the current implementation, the crucial fact is:

\[
\boxed{\text{current mesh = cardinal cubic B-spline, }4\times4\times4=64\text{ coefficients/query}}
\]

The code explicitly evaluates four cubic basis functions in each dimension and performs the 64-tap interpolation. contact_surface

So if by one \(z\)-“slice” you mean **one spline-cell interval**, then the minimum cache is not one coefficient plane but:

\[
\boxed{N_z=4\text{ coefficient planes}}
\]

because a query in cell \(k\) requires

\[
k-1,\;k,\;k+1,\;k+2.
\]

That is the irreducible cubic halo.

But importantly, **there is no reason to cache a fifth plane unless the allowed PP \(z\)-interval can cross into the neighbouring spline cell during that relaxation step.**

So the design should be:

\[
\boxed{N_x\times N_y\times4}
\]

for the usual case, not \(N_x\times N_y\times5\).

Then when the scan moves sufficiently far in \(z\) that the allowed PP interval enters the next spline cell:

```text
old cubic stencil:
    k-1   k   k+1   k+2

next:
    k-2   k-1   k   k+1
```

we keep three planes and load **exactly one new \(N_x\times N_y\) plane**.

That is the streaming scheme you were describing.

## Using the spherical constraint makes this particularly clean

Suppose at outer scan step \(i_z\) we know the tip-anchor \(z_t\), sphere radius \(L\), and impose a maximum lateral PP deflection

\[
\rho\le\rho_{\max}.
\]

For the lower spherical branch,

\[
z_{\rm PP}=z_t-\sqrt{L^2-\rho^2}.
\]

Hence during all lateral relaxation at that fixed tip height,

\[
z_{\min}=z_t-L,
\]

and

\[
z_{\max}
=
z_t-\sqrt{L^2-\rho_{\max}^2}.
\]

The complete vertical uncertainty is therefore only

\[
\Delta z
=
L-\sqrt{L^2-\rho_{\max}^2}
\approx \frac{\rho_{\max}^2}{2L}.
\]

For \(L=3\) Å:

- \(\rho_{\max}=0.5\) Å → \(\Delta z=0.042\) Å
- \(\rho_{\max}=0.8\) Å → \(\Delta z=0.108\) Å
- \(\rho_{\max}=1.0\) Å → \(\Delta z=0.172\) Å

So even for \(h=0.5\) Å, the **entire physically allowed relaxation normally stays inside one spline \(z\)-cell**.

Therefore `Nz=4` should almost always suffice.

Before each outer \(i_z\) step, calculate

\[
j_{\min}
=
\left\lfloor\frac{z_{\min}-z_0}{h}\right\rfloor,
\qquad
j_{\max}
=
\left\lfloor\frac{z_{\max}-z_0}{h}\right\rfloor.
\]

Usually

\[
j_{\min}=j_{\max}.
\]

Then load exactly the union of the required cubic stencils:

\[
[j_{\min}-1,\;j_{\max}+2].
\]

Thus, generally,

\[
\boxed{N_z=4+(j_{\max}-j_{\min}).}
\]

That is probably the cleanest formulation. In the common case it gives 4. Only close to a mesh-cell boundary could it temporarily become 5.

And we can know this **analytically before doing the relaxation**, not discover it dynamically.

---

## Meanwhile \(x,y\) should get essentially all the spare local memory

Exactly the opposite asymmetry:

\[
N_x,N_y \gg N_z\simeq4.
\]

For each WG, define the complete possible lateral footprint from

1. nominal positions of all pixels assigned to the WG,
2. \(\pm\rho_{\max}\) PP bending,
3. cubic spline support.

If the accessible cell-index range is

\[
i_x^{\min}\ldots i_x^{\max},
\]

then

\[
N_x=(i_x^{\max}-i_x^{\min})+4,
\]

and analogously for \(y\).

So a quite reasonable cache could look like

\[
12\times10\times4
\]

rather than `5×5×5`.

And that still costs only

\[
12\cdot10\cdot4\cdot4
=1920\ {\rm bytes}.
\]

Even `16×16×4` is just 4 KB.

That is the direction I would optimize occupancy around.

---

# 1. Atomic core: agreed completely

I also understand your intended PME architecture more sharply now.

The current local kernel effectively does:

```text
global:
    all atoms
    all core coefficients

        ↓ one cooperative read

local:
    ALL atoms

        ↓

every PP evaluates/rejects all atoms
```

whereas the intended design is:

```text
global spatial PIC boxes
          ↓
find boxes intersecting WG footprint + r_core halo
          ↓
cooperative preload ONCE
          ↓
local:
    only nearby atoms + core coefficients
          ↓
all PP relaxation steps reuse them
```

No dynamic PIC lookup whatsoever inside relaxation.

That is indeed much closer to the actual PME philosophy: the smooth field carries the long-range contribution, precisely so that the particle part has **strict compact support** and can be spatially culled.

For a large surface, the work per PP then becomes independent of total system size:

\[
O(N_{\rm local})
\]

rather than

\[
O(N_{\rm total}).
\]

And since the whole workgroup shares basically the same neighborhood, the global atom/parameter read is amortized over perhaps

\[
32\times N_z^{scan}\times N_{\rm relax}
\]

force evaluations.

That should be a huge win.

---

# 3. What is actually used today?

It is definitely **cubic**, not quadratic.

The relevant code is:

```c
float4 bx = cs_pme_basis(ux);
float4 by = cs_pme_basis(uy);
float4 bz = cs_pme_basis(uz);
```

followed by

```c
for(a=0;a<4;a++)
    for(b=0;b<4;b++) {
        float4 v = vload4(...);
        ...
    }
```

so physically:

\[
4_x\times4_y\times4_z=64.
\]

And because \(z\) is fastest in memory, each \(x,y\) combination performs one contiguous `float4` load for the four \(z\) coefficients. contact_surface

The host representation also explicitly says:

> nonperiodic cardinal cubic B-spline

and constructs coefficients by a separable 3-D B-spline prefilter. The interpolation at a node satisfies

\[
d_i =
\frac16c_{i-1}
+\frac46c_i
+\frac16c_{i+1},
\]

so fitting the coefficients means solving this tridiagonal system independently along \(x,y,z\).

Therefore I agree with you:

\[
\boxed{\text{keep this cubic representation first.}}
\]

It is already tested and apparently gives acceptable accuracy. We should change only the data movement.

---

## Would quadratic be significantly faster to *fit*?

Probably **less than I suggested before**.

Quadratic B-spline evaluation certainly decreases the stencil:

\[
4^3=64
\quad\rightarrow\quad
3^3=27.
\]

But fitting quadratic B-spline coefficients is **not automatically free**.

For a centered cardinal quadratic spline, nodal samples are again related to neighbouring control coefficients by a short convolution, something roughly of the form

\[
d_i =
a\,c_{i-1}+b\,c_i+a\,c_{i+1},
\]

so one again needs a 1-D banded prefilter along every dimension.

Therefore both cubic and quadratic fitting are essentially

\[
O(N_{\rm grid})
\]

with a tridiagonal/short-band solve.

Quadratic would lower constants somewhat, but it would not fundamentally remove the fitting stage.

The representation for which fitting becomes really trivial is **linear/trilinear**:

\[
c_i=d_i,
\]

no prefilter at all.

But then the force is discontinuous across voxel boundaries, which I would be reluctant to introduce into PP relaxation.

So there is not a compelling fit-time reason to change away from cubic yet.

---

# I think the kernel should therefore have two independent caches

Conceptually:

```text
                   WORKGROUP XY PATCH
                          │
          ┌───────────────┴───────────────┐
          │                               │
     short-range core                 smooth mesh
          │                               │
 PIC boxes overlapping              calculate allowed
 WG footprint + Rcore               z interval analytically
          │                               │
 preload nearby atoms          preload Nx × Ny × 4 normally
 ONCE for entire stroke                  │
          │                     outer iz changes cell?
          │                               │
          │                           load ONE new
          │                           XY coefficient plane
          └───────────────┬───────────────┘
                          │
                  PP relaxation loop
               local memory only
```

That now seems to me like the clean architecture.

And it means local-memory budgeting should indeed prioritize, in order:

\[
\boxed{
\text{1. nearby atoms/core coefficients}
\gg
\text{2. wide lateral mesh tile}
\gg
\text{3. vertical mesh extent}
}
\]

with the last one normally fixed at the mathematically minimal **four cubic coefficient planes**.

That is substantially better than the tall-cache idea I initially had.


---

# USER

OK, can you write the kernels with comments?

Also plese look on the fitting procedure. I do not even know what we use for fitting now but I know it is quite costly. We were using some CG but that is quite inefficient. Can we use something better? Quasi newton? Or Cholesky, the main problem is - we cannot afford direct method inverting some large matrix, but we should exploit the fact that the lienar fittng problem is highly structured and homogenous (basically just laplace nearest neighbor), there should be highly efficient methods developed how to fit this. And we want to impleemnt such on GPU.

---

# Chat GPT 5.6 sol

Yes. I checked the fitting path as well, and there is a useful surprise: **the current contact-PME mesh itself is no longer fitted by CG**. The expensive CG machinery you remember belongs mostly to the older separable/contact-surface fit. The current PME path does something much more structured already — but unfortunately it brings the mesh back to the CPU for the final spline prefilter.

I wrote a prototype OpenCL file containing the tiled evaluator and exact GPU spline-prefilter kernels:

contact_pme_tiled_prefilter.cl[Download `contact_pme_tiled_prefilter.cl`](sandbox:/mnt/data/contact_pme_tiled_prefilter.cl)

I also syntax-checked it with Clang as OpenCL 1.2. It is intended to be concatenated after the existing `contact_surface.cl`, so it reuses the existing cubic basis, core basis, and global evaluator.

## 1. New evaluation architecture

The intended workgroup structure is now:

```text
                      ONE SPATIAL WORKGROUP
                              |
          +-------------------+-------------------+
          |                                       |
      short-range core                       smooth mesh
          |                                       |
 precomputed WG atom list                  wide XY tile
          |                               only 4 z planes
 cooperative preload ONCE                       |
          |                               cubic 4×4×4 eval
          |                                       |
 reused for entire z stroke          when nominal kz changes:
                                          load ONE XY plane
```

The atom list is fixed for the entire stroke. The host/PIC preprocessing finds every atom whose compact-core sphere can intersect the PP swept volume:

\[
V_{\rm sweep}
=
[\text{WG pixel footprint}+\rho_{\max}]
\times
[z_{\min},z_{\max}]
\]

expanded by the atom's \(r_b\).

So there is **no bucket lookup whatsoever inside PP relaxation**.

The kernel receives

```c
wg_atom_offsets[g]
wg_atom_ids[...]
```

and does one cooperative preload:

```c
inline void cs_pme_preload_wg_atoms(
    __global const float4* atoms,
    __global const float* atom_coeffs,
    __global const int* wg_atom_ids,
    const int atom0, const int nlocal,
    __local float4* LATOMS,
    __local float* LCOEFFS)
{
    const int lid=get_local_id(0);
    const int lsz=get_local_size(0);

    for(int j=lid; j<nlocal; j+=lsz){
        const int ia=wg_atom_ids[atom0+j];

        LATOMS[j]=atoms[ia];

        const int ig=ia*CS_PME_NMODES;
        const int il=j *CS_PME_NMODES;

        const float4 c03=vload4(0,atom_coeffs+ig);
        vstore4(c03,0,LCOEFFS+il);
        LCOEFFS[il+4]=atom_coeffs[ig+4];
    }
}
```

Then all relaxation iterations use only `LATOMS/LCOEFFS`.

That fixes the major architectural problem in the current `Local` kernel, where **all atoms** are copied to local memory and every PP tests every atom. contact_surface

For occupancy, I would not allocate local memory according to the worst WG in the whole scan. Bin workgroups, for example:

```text
0–32 local atoms
33–64
65–96
97–128
...
```

and launch each bin separately with the appropriate dynamic local-memory allocation. Then a dense corner of the molecule does not reduce occupancy everywhere else.

---

# 2. Mesh cache: exactly four z planes

The existing PME representation really is ordinary cardinal **cubic** B-spline:

\[
4_x\times4_y\times4_z=64
\]

coefficients. The current evaluator does 16 contiguous `float4` global loads because \(z\) is fastest. contact_surface

So I kept that unchanged.

The new local cache is

\[
\boxed{N_x\times N_y\times4}.
\]

Not 5 unless we later determine experimentally that a fifth plane is worth the occupancy cost.

The four local planes correspond to one particular spline cell \(k\):

\[
k-1,\quad k,\quad k+1,\quad k+2.
\]

I use a four-slot ring:

```c
inline int cs_pme_tile_zslot(const int gz){
    return gz & 3;
}
```

so moving one cell downward means loading **one new XY plane only**:

```c
inline void cs_pme_tile_update4(
    ...
    const int old_kz, const int new_kz,
    __local float* LMESH)
{
    const int dk = new_kz-old_kz;

    if(dk==0) return;

    if(dk==1){
        // [k-1,k,k+1,k+2] -> [k,k+1,k+2,k+3]
        cs_pme_tile_load_plane(...,new_kz+2,LMESH);

    }else if(dk==-1){
        // [k-1,k,k+1,k+2] -> [k-2,k-1,k,k+1]
        cs_pme_tile_load_plane(...,new_kz-1,LMESH);

    }else{
        // unusual jump/resume
        cs_pme_tile_load4(...,new_kz,LMESH);
    }
}
```

The `x,y` size is deliberately generous. For example

\[
12\times12\times4
\]

is only

\[
12\cdot12\cdot4\cdot4=2304\ {\rm bytes}.
\]

So local memory should be spent on **lateral halo and nearby atoms**, not vertical range.

---

## What if the PP crosses a z-cell boundary during lateral bending?

With sphere constraint,

\[
z_{\min}=z_t-L,
\]

\[
z_{\max}=z_t-\sqrt{L^2-\rho_{\max}^2}.
\]

Thus

\[
\Delta z
=
L-\sqrt{L^2-\rho_{\max}^2}.
\]

For \(L=3\) Å and \(\rho_{\max}=1\) Å,

\[
\Delta z\simeq0.172\ {\rm Å}.
\]

For a 0.5 Å spline grid, most scan steps therefore have

\[
\left\lfloor\frac{z_{\min}-z_0}{h}\right\rfloor
=
\left\lfloor\frac{z_{\max}-z_0}{h}\right\rfloor.
\]

Then the entire allowed PP motion is provably inside one cubic cell and four planes are sufficient.

Occasionally this small interval straddles a cell boundary. I don't think we should permanently spend a fifth plane for this.

The prototype does:

```c
if(actual_iz == cache_kz)
    evaluate_from_local_4_planes();
else
    evaluate_with_existing_global_mesh();
```

So it remains exactly correct. We should count these fallbacks. If they are, say, 0.5–2% of evaluations, excellent. If they turn out to be 20%, then `Nz=5` becomes worth reconsidering.

This is also useful for lateral tile escapes: choose a sensible \(\rho_{\max}\), constrain the spherical relaxation to it, and the XY cache becomes guaranteed.

---

# 3. How this plugs into `LocalSph`

The important synchronization structure should be:

```c
// -----------------------------------------------------------------
// ONE-TIME preload before entire z stroke
// -----------------------------------------------------------------

int g = get_group_id(0);

int a0 = wg_atom_offsets[g];
int a1 = wg_atom_offsets[g+1];
int nlocal = a1-a0;

cs_pme_preload_wg_atoms(
    atoms, atom_coeffs,
    wg_atom_ids,
    a0, nlocal,
    LATOMS, LCOEFFS
);

// choose cubic cell for first z
int cache_kz = ...;

// four planes only
cs_pme_tile_load4(
    mesh_coeffs,
    mesh_meta.y, mesh_meta.z,
    ix0, iy0, nxL, nyL,
    cache_kz,
    LMESH
);

barrier(CLK_LOCAL_MEM_FENCE);


// -----------------------------------------------------------------
// z stroke
// -----------------------------------------------------------------

for(int iz=0; iz<nz; iz++){

    int new_kz = analytically_determined_cell(iz);

    if(new_kz != cache_kz){

        // IMPORTANT: all threads must have finished using previous planes
        barrier(CLK_LOCAL_MEM_FENCE);

        // normally loads ONE plane
        cs_pme_tile_update4(
            mesh_coeffs,
            mesh_meta.y, mesh_meta.z,
            ix0, iy0, nxL, nyL,
            cache_kz, new_kz,
            LMESH
        );

        barrier(CLK_LOCAL_MEM_FENCE);

        cache_kz = new_kz;
    }

    // -------------------------------------------------------------
    // existing LocalSph relaxation; completely divergent here
    // NO BARRIERS
    // -------------------------------------------------------------

    for(int iter=0; iter<imax; iter++){

        ...

        fe = cs_eval_contact_pme_tiled_at(
            pos.x,pos.y,pos.z,

            mesh_coeffs,
            mesh_meta.x,mesh_meta.y,mesh_meta.z,
            mesh_origin_h.x,
            mesh_origin_h.y,
            mesh_origin_h.z,
            mesh_origin_h.w,

            LMESH,
            ix0,iy0,nxL,nyL,
            cache_kz,

            LATOMS,LCOEFFS,
            nlocal,d_span,

            &mesh_fallback,
            &status,
            &min_r
        );

        ...
    }
}
```

One consequence: padded lanes cannot do the current

```c
if(gid >= n_scan) return;
```

because they must participate in future cache-update barriers.

Instead:

```c
bool active = gid<n_scan;

for(iz...){
    // everybody participates in barriers/cache loads

    if(active){
        // PP solver
    }
}
```

The current kernel returns padded lanes immediately after the initial atom preload. contact_surface That has to change for streamed z caching.

Also, the current per-pixel `iz_start_buf` two-pass resume is incompatible with a WG-shared moving z cache **if lanes resume at different z indices**. I would initially use the tiled kernel for the normal first pass. The difficult few pixels from the second-pass recovery can simply use the existing global/local kernel. They are exceptional anyway.

That actually keeps the fast kernel cleaner.

---

# 4. The fitting situation is better than we thought

There are three distinct fitting problems in this codebase.

### A. Current contact-PME coarse mesh

This is the important one.

The current process is effectively

```text
GPU:
    fillContactPMEMeshVL
       ↓
    nodal V_L samples

GPU → CPU transfer

CPU:
    _prefilter_3d()
       ↓
    scipy.solve_banded in z
    scipy.solve_banded in y
    scipy.solve_banded in x

CPU → GPU later:
    spline coefficients
```

So this is **already not CG**.

The cubic B-spline nodal relationship in one dimension is simply

\[
d_i
=
\frac16c_{i-1}
+\frac46c_i
+\frac16c_{i+1}.
\]

Therefore

\[
c_{i-1}+4c_i+c_{i+1}=6d_i.
\]

That is a tridiagonal matrix

\[
A=
\begin{pmatrix}
4&1\\
1&4&1\\
&1&4&1\\
&&\ddots
\end{pmatrix}.
\]

Its eigenvalues are

\[
\lambda_k
=
4+2\cos\frac{k\pi}{n+1},
\]

so

\[
2<\lambda_k<6
\]

and hence

\[
\boxed{\kappa(A)<3.}
\]

This is exceptionally well-conditioned.

There is absolutely no reason for CG, quasi-Newton, global Cholesky, etc.

And because the 3-D cubic basis is separable,

\[
A_{3D}=A_x\otimes A_y\otimes A_z,
\]

we simply apply

\[
C=
A_x^{-1}A_y^{-1}A_z^{-1}D.
\]

That is exactly three batches of independent 1-D tridiagonal solves.

This is the classical spline-prefilter structure; efficient recursive spline filtering has been standard for decades. [EPFL](https://bigwww.epfl.ch/publications/unser9301.pdf?utm_source=chatgpt.com)

---

# 5. GPU direct solver: one thread per line

I put three kernels in the file:

```c
cs_bspline_prefilter_z
cs_bspline_prefilter_y
cs_bspline_prefilter_x
```

Each work-item solves one complete line using Thomas elimination.

For example the z kernel:

```c
__kernel void cs_bspline_prefilter_z(
    __global float* a,
    const int nx, const int ny, const int nz,
    __global const float* invden,
    __global const float* cprime)
{
    const int line=get_global_id(0);
    const int nline=nx*ny;

    if(line>=nline || nz<2) return;

    const int base=line*nz;

    // forward elimination
    float yp=(6.0f*a[base])*invden[0];
    a[base]=yp;

    for(int i=1;i<nz;i++){
        const int p=base+i;
        const float yi=(6.0f*a[p]-yp)*invden[i];
        a[p]=yi;
        yp=yi;
    }

    // backward substitution
    float xp=a[base+nz-1];

    for(int i=nz-2;i>=0;i--){
        const int p=base+i;
        const float xi=a[p]-cprime[i]*xp;
        a[p]=xi;
        xp=xi;
    }
}
```

The y and x versions are the same except for their stride.

The nice GPU property is that although **each individual line is sequential**, there are many independent lines.

For y solve, for example, thread IDs enumerate `(ix,iz)`, so at each common y iteration neighboring GPU lanes access neighboring `z` addresses:

```text
lane 0 -> a[ix, iy, iz=0]
lane 1 -> a[ix, iy, iz=1]
lane 2 -> a[ix, iy, iz=2]
...
```

so accesses are still nicely coalesced.

For meshes of order perhaps 20–100 nodes/axis, I strongly expect this simple batched Thomas solver to beat fancy PCR because we have many independent systems and each system is short. PCR/CR becomes interesting if there are very long lines and too few systems; GPU literature indeed finds different CR/PCR/hybrid tradeoffs depending on problem dimensions. [NVIDIA](https://research.nvidia.com/publication/2010-01_fast-tridiagonal-solvers-gpu?utm_source=chatgpt.com)

So I would start with Thomas, not PCR.

Three kernel launches total:

```text
fillContactPMEMeshVL
       ↓
prefilter_z
       ↓
prefilter_y
       ↓
prefilter_x
       ↓
DONE, coefficients already resident on GPU
```

Actually four including rasterization.

No iterations.

No reductions.

No atomics.

No CPU roundtrip.

No temporary large matrix.

That should be very hard for CG to compete with.

---

## Tiny factor arrays

The Thomas factors depend only on line length.

They can be computed on the host in microseconds:

```python
def cubic_bspline_thomas_factors(n):
    invden = np.empty(n, np.float32)
    cprime = np.empty(n, np.float32)

    invden[0] = 1.0 / 4.0
    cprime[0] = 1.0 / 4.0 if n > 1 else 0.0

    for i in range(1, n):
        den = 4.0 - cprime[i-1]
        invden[i] = 1.0 / den
        cprime[i] = invden[i] if i < n-1 else 0.0

    return invden, cprime
```

You need only one pair for each of `nx`, `ny`, `nz`.

These are tens of floats, not part of the expensive computation.

I tested the algebra against SciPy's current `solve_banded`; it reproduces the solution to floating-point precision.

---

# 6. So where was the CG?

The older `ContactSurface` code still has things like

```text
fit_separable_cg()
fit_pic_cg()
```

and does matrix-free iterations using

\[
A^T A\,c=A^Tb.
\]

That is useful when the basis is being fitted to arbitrary scattered energy/force samples with weights.

But that is **not what we need for the current PME mesh**.

For PME we control where we evaluate the reference potential. We evaluate it **at the spline nodes**. Therefore we can perform exact cardinal interpolation instead of a global least-squares fit.

That design decision is worth preserving.

Quasi-Newton would actually be a step backward here: the problem is linear and its matrix is analytically known.

---

# 7. There is still another expensive fit: the 5 core coefficients

`PICCore.py` currently does considerably more work than necessary for a large system.

For each atom it roughly does:

```text
~300 radial samples
soft_core_split()
Boltzmann weights
energy + force rows
condition number
lstsq raw basis
lstsq hierarchical basis
hold-out validation
choose best
```

for only

\[
5
\]

unknown coefficients.

For a few dozen atoms that's irrelevant. For tens of thousands of atoms, it is silly.

There are two possible optimizations.

### First: deduplicate identical parameter sets

If many atoms have the same

\[
R_0,E_0,q,\alpha,r_{\rm damp},...
\]

fit once and copy the five coefficients.

At least \(R_0,E_0\) are usually type-dependent rather than atom-dependent.

### Second: GPU 5×5 normal system

For each atom form

\[
G_{mn}
=
\sum_s
w_E\phi_m\phi_n
+
w_F\phi'_m\phi'_n,
\]

and

\[
b_m
=
\sum_s
w_E\phi_mV_S
+
w_F\phi'_mV'_S.
\]

Then solve

\[
Gc=b.
\]

`G` is only \(5\times5\).

One workgroup per atom can accumulate the 15 unique entries of symmetric \(G\) plus 5 RHS values, reduce them in local memory, and one lane performs a tiny Cholesky.

That is completely different from “Cholesky of a giant matrix.” It is thousands of independent **5×5** Choleskys.

I would do this after the mesh work, though.

The larger win is probably:

1. tiled local core evaluation,
2. GPU mesh prefilter,
3. only then optimize the 5-mode coefficient generation.

---

# 8. One potentially very strong simplification for the core fit

There may even be a route to almost eliminate per-atom fitting.

For fixed \(R_0,\alpha,r_{\rm damp},r_b\), the underlying potential is linear in

\[
E_0
\quad\text{and}\quad
q.
\]

The PAW smoothing construction is also a linear operation on the boundary values \(V,V',V''\).

Therefore the residual \(V_S\) is fundamentally linear in these amplitudes:

\[
V_S(r)
=
E_0\,f_M(r)
+
q\,f_C(r).
\]

If we replace the current atom-dependent Boltzmann weights with fixed radial weights, then the least-squares operator is also linear, and we can precompute

\[
c_M,\qquad c_C
\]

for each \(R_0\)/atom type and construct every atom's coefficients simply as

\[
\boxed{
c_i=E_{0,i}c_M+q_i c_C.
}
\]

That would reduce “fit thousands of atoms” to “fit perhaps 5–20 atom types.”

The current Boltzmann weighting breaks that exact linearity because the weights themselves depend on each atom's total potential. I would test whether those weights are really buying enough accuracy to justify destroying this very useful structure.

That might be a surprisingly large simplification.

---

## My implementation order

I would now do it in exactly this order:

1. **Validate `evalContactPMETiledWG`** against current `evalContactPMELocal`: same queries, ensure \(E,F\) agree and measure the cache-fallback rate.

2. Add the four-plane streaming cache to `LocalSph`, initially only for fresh scans — no two-pass resume complexity.

3. Replace all-atom local preload by precomputed **WG atom lists**.

4. Replace CPU `_prefilter_3d` with the three GPU Thomas kernels. This should be a very clean change.

5. Profile fit again. If fitting is still slow, determine whether the time is in `fillContactPMEMeshVL` or `fit_core_1d`.

6. Only then optimize the 5-mode core fit, probably by unique-type/template fitting rather than immediately building a complicated GPU least-squares solver.

The main conceptual simplification is that the PME fit should not be viewed as a generic optimization problem at all. For the mesh it is basically a **separable inverse filter**, and for the core it is a collection of independent five-parameter problems. Neither needs global CG.

---

# USER

wait you wrote some python code where you iterate over some arrays in python. I hope you are not meaning seriously that we should do any heavy numerics like this in pythin which is 100x slower than C and 10000x slower than OpenCL on GPU! That is just some illustrative pseudocode?
Hope that current implementation of the fitting does not doe anything in python or even numpy and does all serious heavy numerical work (like any iteration over our data-grids) in OpenCL ?

---

# Chat GPT 5.6 sol

Yes — the little Python loop I wrote for computing the Thomas factors was **only tiny host-side setup**, not intended for any grid-sized computation.

Those arrays have length only `nx`, `ny`, or `nz` (typically tens of elements), are computed once, and then reused. Whether that takes 2 µs or 200 µs is irrelevant. All operations over the actual 3-D mesh should be OpenCL.

But there is an important problem: **the current implementation is not yet fully GPU-resident.** Looking through the actual code, there are still substantial CPU/NumPy/SciPy pieces.

### Current contact-PME mesh fitting

For the preferred `paw` mode, the good part is already on GPU:

```text
fillContactPMEMeshVL
    atoms → nodal V_L mesh
```

So evaluating all atoms onto all coarse mesh points is done by OpenCL.

But immediately afterward `_pme_build_coarse_mesh_gpu()` currently does approximately:

```python
cl.enqueue_copy(... samples_f32 ...)      # GPU -> CPU

samples = samples_f32.astype(np.float64)
samples = samples.reshape(nx,ny,nz)

coeffs = _prefilter_3d(samples)            # CPU / SciPy

return CoarseMesh(coeffs=coeffs, ...)
```

And `_prefilter_3d()` calls SciPy's `solve_banded()` along z, y and x.

So right now the pipeline is unfortunately:

\[
\boxed{
GPU\ raster
\rightarrow CPU
\rightarrow SciPy\ spline\ prefilter
\rightarrow GPU
}
\]

That is exactly what the OpenCL Thomas kernels I proposed are meant to eliminate.

The desired version should be:

```text
GPU fillContactPMEMeshVL

        ↓ same GPU buffer

GPU prefilter_z
GPU prefilter_y
GPU prefilter_x

        ↓

resident spline coefficients
```

No mesh download at all.

---

### More seriously: the short-range core fit is currently CPU code

`PICCore.fit_core_1d()` currently contains a Python loop over atoms:

```python
for i in range(na):
    ...
```

Inside each iteration it generates hundreds of radial samples using NumPy, evaluates the split potential, constructs the design matrices, and calls things such as:

```python
np.linalg.cond(...)
np.linalg.lstsq(...)
```

twice, plus holdout tests.

So for a large system:

\[
\boxed{\text{this is absolutely not acceptable.}}
\]

It was evidently written for tens/small hundreds of atoms, where the convenience mattered more than scalability.

For thousands or millions of atoms we should definitely remove this.

---

## But we probably shouldn't GPU-fit every atom either

There is an even better optimization.

The core has only five coefficients:

\[
c_{i0},\ldots,c_{i4}.
\]

For chemically equivalent atoms, these coefficients are identical provided their force-field parameters are identical.

So if a surface contains, say,

```text
C type 1
C type 2
N
O
H
Cu
...
```

we should fit perhaps **10–100 unique parameter tuples**, not 100,000 atoms.

Create a key like

\[
(R_0,E_0,q,\alpha,r_{\rm damp},\Delta_{\rm in},\Delta_b,\ldots)
\]

and fit each unique tuple exactly once.

Then atom \(i\) merely stores

```c
type_id
```

or gets the corresponding five coefficients copied during preprocessing.

That may make GPU optimization of this fitting step almost unnecessary.

Even better, as I mentioned before, because Morse and Coulomb amplitudes enter linearly, we may be able to precompute basis fits and construct

\[
c_i =
E_{0,i} c^{M}(R_{0,i})
+
q_i c^{C}(R_{0,i}),
\]

provided we remove/modify the current potential-dependent Boltzmann weighting.

Then there is effectively **no per-atom fitting at all**.

---

## The older CG fitting is GPU arithmetic, but Python-driven

The older `fit_separable_cg()` is somewhat better in this regard.

The expensive operations

\[
Av,\quad A^Tv,\quad \mathbf a\cdot\mathbf b,\quad
\mathbf a+\lambda\mathbf b
\]

are OpenCL kernels.

But Python does:

```python
for it in range(n_iter):
    _cg_step_sep(...)
```

and each CG iteration launches several separate OpenCL kernels.

So the numerical arrays aren't being iterated element-by-element in Python, but you still pay **many Python→OpenCL kernel launches and synchronizations**.

Given your previous observations about PyOpenCL launch latency, this is also undesirable.

Fortunately, for the current cubic PME mesh we don't need CG at all.

---

# What I think the final fitting pipeline should be

### Smooth PME mesh

Entirely GPU:

```text
atoms
  │
  ▼
fillContactPMEMeshVL
  │
  │ nodal samples, GPU resident
  ▼
B-spline inverse filter Z
  │
  ▼
B-spline inverse filter Y
  │
  ▼
B-spline inverse filter X
  │
  ▼
mesh coefficients
```

Three deterministic \(O(N)\) passes.

No CG.

No SciPy.

No host transfer.

The only CPU calculations should be microscopic constants such as

```text
nx, ny, nz
origin
strides
Thomas factors (~20–100 floats)
PIC metadata
```

which don't scale with the volume in a problematic way.

### Compact atomic core

Prefer:

```text
unique FF parameter tuples
        │
        ▼
fit each unique type once
        │
        ▼
5 coefficients/type
        │
        ▼
atoms reference type
```

If there genuinely are many unique atoms, then move the 5×5 least-squares problem onto GPU as thousands of independent tiny fits.

### Scan/evaluation

Everything OpenCL, with:

```text
WG:
    preload relevant PIC atoms once
    preload Nx × Ny × 4 mesh slab

    for z:
        occasionally stream one new XY plane

        PP relaxation:
            zero global-memory field accesses
            except exceptional fallback
```

So yes: **your instinct is correct**. There should be no grid-sized Python numerics in the final implementation. And after inspecting the current code, there are still two CPU pieces worth removing: especially the CPU spline prefilter and the per-atom `fit_core_1d()`.