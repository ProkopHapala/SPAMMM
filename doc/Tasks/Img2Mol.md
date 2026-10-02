# Img2Mol — extract carbon skeleton graph from AFM/STM images of planar PAHs

Source data: `/home/prokop/Desktop/PROJECTs/Svec_Ditriptaceneo_Helicene/Images/`
(presentations Ditriptyceno[7]helicene; images `1.png`-`14.png` are noisy low-res
144-203 px crops, `big.png` 800x512 is the high-quality reference).

Code: workhorse `spammm/img2mol.py`, demo `doc/export_invAFM/scripts/testplot_img2mol.py`,
artifacts -> `debug/testplot_img2mol/<img>/stages.png`.

## Ground-truth structure of big.png (USER specification — reference solution)

Ditriptyceno[7]helicene. In big.png the molecule lies as a horizontal ribbon;
the bright fan/paddle lobes at the far left and right ends are the appendix
regions; a protruding blob sits at the top-middle of the core.

Ring inventory (USER-provided):

| part | rings | where in image |
|------|-------|----------------|
| coronene core | 1 central hexagon + 6 fused hexagons around = **7 hex** | center of image |
| left appendix | anthracene = **3 hex** in a row, attached to core through a **connector hexagon** | left fan/lobe |
| right appendix | same: **3 hex + 1 connector hex** | right fan/lobe |
| top substituent | **1 hexagon connected to the core via a pentagon** | top-middle blob |
| pentagons | **exactly 3 total** (1 under the top hexagon; other 2 near the appendix/helicene kink junctions — exact placement to be identified) | — |

Total expected: **16 hexagons + 3 pentagons = 19 rings**, ~57-63 C atoms
(coronene 24 C; each fused hex +4 C, fused pentagon +3 C along one edge).
V−E+F=1, all vertex degrees 2-3, every ring shares full edges with its neighbors.

**Use as reference:** big.png is the high-quality calibration image — the
algorithm's output on it should match this inventory (19 rings, 3 pentagons,
connected single graph). Since `1.png`-`14.png` show the SAME molecule at
lower resolution/noise, this structure can also serve as a **template prior**:
fit this fixed fused-ring topology (positions on a distorted triangular
lattice) to each image rather than discovering topology from scratch.

## Chemical graph prior (must drive the model, not just validate it)

## Chemical graph prior (must drive the model, not just validate it)

- sp2 carbon = vertex, connects **3 bonds**; interior fusion C shared by **3 rings**,
  rim C shared by 1-2 rings. **deg>3 vertex is unphysical — a hard error signal.**
- bond = shared edge of two adjacent rings (or rim edge of one ring)
- rings are only 5- or 6-membered; pentagons rare (~3 in big.png)
- **the molecule is ONE fused graph — n-gons share edges and vertices BY
  CONSTRUCTION, never independently fitted and merged afterward**
- the dual of a fused hexagonal PAH is a patch of the **triangular lattice**:
  fused ring centers are separated by apo_i + apo_j
  (hex-hex: sqrt(3)*b ~ 1.73b; hex-pent: ~1.55b; pent-pent ~1.38b; b = C-C bond)
- interior vertex = centroid of a triangle of 3 mutually fused ring centers
  (i.e. an occupied triangular face of the dual lattice)
- rim vertex = endpoint of a shared edge, or free polygon corner

## What was tried and how it failed (handoff notes)

All attempts were **bottom-up detection**: threshold image → extract features →
hope topology assembles. Every stage depended on a fragile local threshold that
ghost forces and noise violate; failures cascade and cannot be recovered
downstream. This whole paradigm was rejected by the user.

### Attempt log

1. **Skeletonize ridge mask → pixel graph → cycles.**
   Thick ridges (~4px) produce junction "ladders": deg>=3 pixels spread along
   walls instead of point junctions at atoms. Skeleton fragmented into a forest
   (E<V, 0 rings on first runs). Junction clustering/dilation over-merged chains.
   ABANDONED: pixel skeleton is the wrong representation at this resolution.

2. **Hole contours → RDP polygonization.**
   Ring interiors are dark holes in the ridge mask; contour corners = carbons.
   RDP corner count unstable on pixelated boundaries (hexagon gave 5-11 corners
   depending on epsilon). ABANDONED.

3. **Radial profile + harmonic fit of hole boundary.**
   r(theta) from hole centroid; fit cos(n·theta+phi) for n=5,6. Amplitudes weak
   (amp ~0.02-0.1, residuals ~0.97 for both n) — cannot discriminate 5 vs 6.
   Watershed regions merged into halo through wall gaps, contaminating profiles.
   ABANDONED as n-classifier (could still be a soft feature).

4. **Ring centers: LoG dark-blob peaks + enclosure test.**
   Centers = peaks of smoothed -LoG inside non-wall distance transform;
   accepted only if enclosed by walls (enclosure fraction > 0.55, bounded
   clearance 0.35b < D < 1.15b). **Result on 9.png: 9 centers / 8 rings vs ~20
   visible rings — ~50% missed.** When walls don't close (ghost forces, low
   contrast), the hole leaks into the halo and the center is rejected. This
   violates the first necessary condition: **reliable ring-center detection**.
   The user explicitly called this out.

5. **Adjacency by distance window + wall-brightness along midline.**
   Pair accepted if center distance in window and mean ridge response along the
   connecting midline > 0.3·Otsu(R). Missed real fused pairs where wall contrast
   is weak; accepted spurious ones in halo regions.

6. **Geometric vertex inference (current state).**
   Vertices = centroids of mutually-fused center triples + shared-edge endpoints
   + free-arc polygon fill; all proposals clustered within 0.35b so rings DO
   share vertex ids (topological sharing works). BUT produces **deg4 vertices**
   (unphysical for sp2 — clustering artifacts / wrong center sets), and the
   upstream center/adjacency errors make the output nonsensical: overlapping
   rings, disconnected fragments, wrong ring counts.
   Numbers: big.png V=52 E=69 15 rings (arms/fan rings missing); 7.png called
   9 pentagons (expected ~3); 9.png total failure.

### Root cause of failure

- **Threshold cascade**: Otsu(flat) → Otsu(R) → wall dilation → enclosure
  fraction → distance bounds → midline brightness → harmonic residual.
  ~8 hand-set thresholds, each a failure point on noisy 144-203 px images.
- **Centers were free-floating peaks** — the strongest unused prior is that
  fused-PAH ring centers form a subset of a TRIANGULAR LATTICE. Never used.
- **Ring existence was decided by hole enclosure** — the most fragile possible
  criterion, since the very defects being sought (broken walls) destroy it.
- **n=5/6 decided per-ring from weak local evidence** instead of globally under
  the rare-pentagon constraint.
- **Bond scale b was chicken-and-egg**: adjacency window needs b, b estimate
  needs clean adjacency; median-NN corrupted by false centers.
- Independent detection + post-hoc merging is structurally wrong — the user's
  core objection. Even the final "topological" version still decided each ring
  and each vertex locally.

## Recommended approach for the next attempt

**Generative / lattice-fit, not bottom-up detection.** The image should be
explained by a fused-graph hypothesis that is optimized as a whole.

1. **Estimate bond scale and lattice basis globally.** The ring-center dual is
   a triangular lattice; its spacing/orientation can be found from the image
   autocorrelation or FFT of the dark-interior response — robust to missing
   rings because it is a global periodicity estimate. Do NOT estimate b from
   nearest detected peaks.
2. **Propose candidate ring centers on the lattice**, not at blob peaks.
   Generate all lattice sites inside a loose molecule mask; each site gets a
   score: darkness of interior disk + wall brightness around its perimeter
   (ring integral at ~0.6-0.8·b in 6 directions). Lattice placement is what
   makes this robust — a merged/missing hole is still scored as a candidate
   site, and artifacts cannot create off-lattice sites.
3. **Select a subset of sites by global fusedness constraints**: accepted rings
   form a connected patch; each accepted ring should fuse to >=1-2 neighbors
   (isolated ring suspect); shared edges = adjacent accepted sites and must sit
   on a wall (score the midline, but as a soft energy term, not a veto).
   This is a small combinatorial/MRF problem — sites ~20, solve greedily or
   by local search with energy = -site_score - pair_wall_score + topology
   penalties (deg4 impossible, valence, planarity).
4. **Build the molecular graph purely from topology once the dual is known**:
   occupied triangular face of the dual -> interior atom; accepted adjacent
   pair -> shared edge with 2 endpoint atoms; remaining ring boundary -> rim
   atoms. Vertex positions from lattice geometry, optionally snapped to local
   ridge maxima for sub-pixel placement.
5. **Pentagons globally**: a pentagon is a lattice defect. First fit all-hexagon;
   only where the hexagon model systematically fails (site score insists a ring
   exists but 6 neighbors cannot tile — e.g. [7]helicene curvature, the known
   connector) test pentagon insertion. Hard cap on count (~3). Alternatively
   enumerate the handful of candidate fused-ring topologies of this molecule
   family (coronene core + appendices) and score each rendered overlay.
6. **Validate chemically, fail loud**: ring sizes in {5,6}, V-E+F=1, all
   degrees in {2,3}, ring incidence consistent; report violations, don't
   silently patch.

## What DOES work / reusable pieces

- Background flattening (divide by large-Gaussian smooth): `flatten_bg`
- LoG ridge response `-ndi.gaussian_laplace(flat, sigma~b/7)`: walls light up
  correctly even on noisy images — good SCORING input (but bad to threshold)
- Geometric vertex rules (triple-centroid, pair shared-edge endpoints) are
  mathematically correct WHEN the dual graph is right
- Stages-plot infrastructure: `doc/export_invAFM/scripts/testplot_img2mol.py`,
  `debug/testplot_img2mol/<img>/stages.png`
- `AtomicGraph` conversion + `au.saveXYZ` plumbing exists (untested end-to-end)

## Status: FAILED approach — handed off for redesign

- [x] ridge/mask response visualization on all images
- [x] shared-vertex graph construction (works mechanically)
- [ ] reliable ring-center detection on lattice — **the missing foundation**
- [ ] global fused-subset selection under chemical constraints
- [ ] hex/pent under global rare-pentagon constraint
- [ ] AtomicGraph + XYZ export; rim hydrogens
- [ ] robustness on all 14 low-res images

## Progress report, 2026-09-29 — awaiting visual verification on small scans

### What works

- On `big.png`, the revised soft ring proposal stage yields 19 centers. The user reviewed `debug/testplot_img2mol/big/provisional_centers.png` and accepted their placement, including the two weak orange sites.
- Delaunay neighbor proposals plus bright shared-wall scores give a 27-edge fused-ring dual. Neighbor angles identify 3 pentagons and 16 hexagons. Exact shared-corner merging yields 66 carbons and 84 bonds in one connected graph, with maximum carbon degree 3 and `V-E+F=1`. The user visually accepted `debug/testplot_img2mol/big/topology_candidate.png`.
- For repeated scans of this same molecular pattern, `align_ring_template` fits translation, scale, rotation, and reflection to soft ring proposals. It is a *diagnostic prior*: a transformed template site is not accepted as a ring merely because the template predicts it. Run `python3 -u doc/export_invAFM/scripts/testplot_img2mol.py --all --template-big`; inspect `debug/testplot_img2mol/template_contact_1-14.png` and per-image `template_centers.png` files. Green means a nearby image proposal inside the molecule mask; orange means a weak interior hypothesis; red X means outside the mask.

### What does not yet work

- Proposal-supported sites per small image, in image order 1–14: `18, 18, 14, 16, 18, 16, 13, 14, 13, 16, 15, 13, 14, 15` out of 19. These counts measure proximity to a soft image proposal, **not** proven ring accuracy. Images 1, 2, and 5 are promising; 7, 9, and 12 remain difficult. The severe scan artifact in 9 produces spurious proposal evidence.
- A similarity transform cannot explain all local distortions. Some orange sites appear to be real rings with weak contrast; some are misplaced. Mask rejection catches a few exterior predictions but does not establish that every interior site is enclosed by a bright rim.
- The accepted big-image graph is not yet automatically recovered from each low-resolution image. The original `img_to_graph` path remains unreliable, and template alignment is specific to repeated scans of one topology. Do not treat its output as a generic PAH reader or export a trusted XYZ from these small scans yet.

### Next direction

1. Refine each template-predicted center against *local enclosed rim* evidence, with one-to-one assignment to observed center proposals; reject exterior and scan-line artifacts explicitly. Keep weak interior sites as hypotheses rather than auto-filled rings.
2. Select centers and fused sides together using bright shared-wall evidence and carbon valence/Euler constraints; score local geometric deformation of the template. For unrelated molecules, infer the dual without a known 19-ring template.
3. Show per-image ring and bond overlays for human review, then validate chemistry and `AtomicGraph` export. Keep task status unverified until user confirms the low-resolution results.

## Progress report, session 3 — convolutional ring-center detector (`ring_centers_E`)

Status: **ring-center detection works well on most images (user-reviewed), NOT yet wired into graph building.** Nothing downstream (graph, XYZ) is verified.

### Why the previous pipeline was dropped as the main path

The evidence-fusion + simulated-annealing + lattice-snap pipeline in `img_to_graph` (`ring_hypotheses`, `select_rings`, `fused_topology`, `fit_center_lattice`) was too fragile. Evidence from this session:
- The weighted-sum score S (≈7 z-scored channels) let weak rim/vertex channels outvote a clean center dip, so 4+ real rings per image were lost at selection (panel D of `channels.png`).
- Hard lattice snapping destroyed whole images: on 12.png the lattice fit locked onto a sheared basis (|B| = 10/16.6 px, 65°), all centers came out off-lattice, and 12.png collapsed to 3 rings. Neither projecting onto an ideal 60° basis nor free affine refit helped (big.png then collapsed to a denser lattice, d0 = 53–58 px vs 65 expected). **The lattice should only be a soft consistency check, never a snap.** (`fit_center_lattice` now has an experimental regularized refit; treat it as unverified.)
- The b-rescale second pass could destroy a good first pass (9.png: 21 → 4 rings). `img_to_graph` now keeps the pass with the most rings.
- USER correction: the molecule is planar and is NOT a helicene. There is no non-planar distortion to model; the differences are noise, contrast and tip artefacts.

### How the new detector works (all convolution / batched sampling, ~0.02 s per small image, 0.45 s for big.png)

Code: `spammm/img2mol.py` → `center_response`, `sector_kernels`, `sector_enclosure`, `ray_enclosure`, `ring_centers_E`, `junction_kernels`, `junction_response`.
Diagnostics: `python3 doc/export_invAFM/scripts/testplot_img2mol.py --filters --img 12.png` (or with no `--img` for 1.png) → `debug/testplot_img2mol/<img>/filter_bank.png`, `filter_response.png`, `ring_centers_E.png`.

1. **Bond scale** `b` from ridge autocorrelation (`estimate_bond_scale`); about 8.1 px on the small scans, 37.5 px on big.png.
2. **Center response** `C = -LoG(flat, 0.45 b) / localRMS(2b)`. This is the smooth, well-denoised map; ring interiors are dark blobs. USER: "good base, lines a bit too wide".
3. **Filter bank** `sector_kernels(b, nsect=8, rads=(0.68, 0.90))`, a table of 2 radii × 8 angular sectors:
   - *Radial part*: a zero-mean ring wavelet: negative inside, positive bump at r0 (0.68 b ≈ pentagon apothem, 0.90 b ≈ hexagon), negative skirt outside, zero beyond. The weights are balanced so ∫K = 0 (DC-free, so insensitive to contrast offsets).
   - *Angular part*: periodic quadratic B-spline windows forming a partition of unity (Σ_j ang_j(θ) = 1, so the sectors sum to an isotropic annulus; checksum shown in `filter_bank.png`).
   - Each kernel is L2-normalized (ΣK² = 1).
   - FFT convolution (`signal.fftconvolve`); parity against `ndi.correlate` gives rel. error 2.5e-13.
4. **Fuzzy AND**: `M_j = max_i S_ij` (a rim at *some* radius in sector j), then `E = (Π_j clamp(M_j, 0))^(1/nsect)`. Peaks of E are the ring-center *candidates*.
5. **Filters on the candidates** (reason codes in `ring_centers_E.png`):
   - `ray`: **ray-cast enclosure** on a *sharp* ridge map `R = -LoG(flat, 0.2 b)/localRMS(3b)`. 36 rays, radii 0.35–1.3 b, take the radial max per ray; require the 10th percentile over rays `Rq10 > 0`. This is the hard "wall in every direction" test. On the hand-labelled cases, real rings had Rq10 of 0.15–1.08 and fakes had −0.68 to −0.31, with no overlap across 1/9/12/big.
   - `Ef`: the same filter bank applied to the raw flattened image instead of C. Background noise peaks have Ef ≈ 0–5 against 55–170 for real rings (C is RMS-normalized, so it amplifies background noise).
   - `close`: steric exclusion. Candidates closer than `dmin = 1.25 b` to a stronger one are removed. **Known to be wrong, see below.**
   - `comp`: isolated singletons with no candidate within 2.2 b are removed. Components of 2 or more are kept. An earlier rule that kept only the largest component deleted whole appendices on 2.png and 5.png.
   - The earlier gates `U = min_j M_j / mean_j M_j ≥ 0.55` and `C < 0` were removed from the decision. They are still stored as diagnostics. `C<0` wrongly rejected a real big.png ring (C = +0.39), and U rejected the weak connector ring of the big.png appendix.

### Why the fuzzy AND alone does not reach zero on false rings (answer to USER question)

- E is computed on C, which is itself a LoG output. **A LoG creates a positive side-lobe ring around every dark blob**, so a dark halo gap or dark bay gets a fabricated rim in every direction. Measured: the false peak in 9.png at (21,115) has M_j(C) ≥ 0.84 in all sectors, while on the raw image its sectors 4–7 drop to 20–35 against 150.
- The geometric mean ^(1/8) is soft: one sector at 1/5 of the others lowers E by only 5^(1/8) ≈ 1.22×.
- The B-spline sectors overlap (support 135°), so an opening narrower than about 45–90° is covered by the neighbouring sectors.
- This is why the hard test must use a *sharp* map along *narrow* rays (the `ray` check), not C.

### Results (current code state, including the `close` rule)

| img | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | big |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accepted | 19 | 18 | 19 | 18 | 17 | 17 | 17 | 17 | 15 | 18 | 20 | 20 | 20 | 20 | 19 |

USER review of the earlier U-based version: 12 and 3 perfect; 9 and 1 had one false positive each; big had 4 false negatives. The `ray` test fixed all of these: the 9.png (21,115) and 1.png (104,63) false positives are rejected, and the four missing big.png rings (164,308), (208,356), (250,405), (271,352) are recovered. The ray-based results on 2, 4–8, 10, 11, 13, 14 have **not yet been reviewed by the USER.**

### Known problems / where it fails

1. **The `close` steric rule is fragile and should be replaced.** It correctly removes the big.png false peak (169,264) at 1.18 b, but it also removes three probably-real connector rings at 1.24 b: 4.png (53,101), 5.png (54,79), 10.png (130,105). Distance does not separate these cases. These short-spaced connectors are probably pentagons (pent–hex spacing ≈ 1.56 b vs hex–hex 1.73 b; the median neighbour spacing measured here is ≈1.5 b).
   **Measured replacement, not yet implemented**: *raw ring depth* = (q10 over rays of the radial-max of `gaussian_filter(flat, 0.2b)` − value at the center) / std. All accepted rings on 4, 5, 10, 12 and big have depth ≥ −0.27 (median 0.6–1.2); the big.png false peak has −0.93 (its center is brighter than its rim); the three connectors have 0.10–0.53. Plan: add a `depth < -0.5` rejection, comment out the `close` rule, re-run all 15 images.
2. **Weak-contrast peripheral rings**, especially appendix connectors, are the main source of false negatives. Their outer wall is the soft molecule edge.
3. **Scan artefacts** (9.png bright blobs under the molecule) produce well-enclosed-looking candidates. Three were removed only as isolated singletons (`comp`).
4. **Remaining singleton `comp` rejects**: 13.png (49,72) has Rq10 = 1.19 (strongly enclosed) but no neighbour within 2.2 b. It could be a real ring whose neighbour was missed; review it.
5. **Thresholds** (Rq10 > 0, Ef_rel = 0.25, dmax = 2.2 b) come from a handful of labelled points on 1/9/12/big. They are not validated on a labelled set.
6. **`b` is almost constant (8.08 px) across the small images** because of autocorrelation binning. The measured median ring spacing is about 1.5 b, not √3 b ≈ 1.73 b, so b is probably overestimated by roughly 10–15%. The radii of the kernels and rays scale with b.
7. **V3 junction map** (`junction_kernels(narm=3)`): the kernels are correct, but the response on C looks ridge-textured rather than point-like. It is not used in any decision yet.

### Ideas / next steps

1. Replace `close` with the raw-depth test (item 1 above); USER should review all 15 `ring_centers_E.png`.
2. **Independent vertex / edge parity check** (USER request): V3 junction response on a sharper map (the sharp R, or `-LoG(0.15–0.2 b)`), evaluated only at predicted corners. For each accepted center and its fused neighbours, the shared-edge endpoints and triple-ring corners should light up in V3, and the shared wall should be bright in R along the center–center midline (measured: normal fused pairs have midline R ≥ 0.2, median 0.8). Use this to (a) confirm weak rings with low Rq10 and (b) link appendices across weak connectors.
3. Hysteresis growth: strict seeds (Rq10 > 0.3), then admit weaker candidates only if they are fused to an accepted ring (bright shared wall plus V3 at the shared corners).
4. Pentagon versus hexagon: use the local spacing to fused neighbours (pent–hex ≈ 0.90× hex–hex) and the angle between neighbour directions (72° vs 60°), together with a radius-resolved bank (keep M_ij per radius, not only its max: the argmax radius ≈ apothem ≈ ring size).
5. Better `b`: take it from the median nearest-neighbour distance of accepted centers (≈ √3 b for hex–hex), then iterate the bank once.
6. Then build the graph: accepted centers → fused pairs (bright midline wall, distance 1.3–2.1 b) → `graph_from_fused_rings` (shared vertices and edges by construction, degree ≤ 3, V−E+F = 1). Do **not** use annealing or hard lattice snapping; use the lattice only as a soft sanity check.
7. Plot hygiene: `channels.png` panel (E), "fitted rims", draws per-ring polygons with independently fitted phase. The USER called it useless: randomly oriented, disconnected hexagons. Drop it, or draw polygons derived from the fused-neighbour directions.

## Progress report, session 4 — angular filter bank to shared-corner skeleton (unverified)

The user's radial/angular fuzzy-AND detector remains the center proposal engine. The following changes are experimental and need visual review before this task can be called resolved.

### Changes

- `sector_kernels` now subtracts each discrete kernel's mean before L2 normalization. Measured maximum absolute kernel sum is `5.83e-16` at `b=8.08` and `2.55e-15` at `b=37.5`; L2 errors are below `4.45e-16`. FFT convolution versus direct correlation on 12.png has relative error `5.65e-14`.
- `ring_centers_E` uses an independent **raw-image rim depth**: 10th-percentile ray-wise radial maximum around a candidate minus its center intensity, normalized by image standard deviation. A depth below `-0.5` rejects a bright-center false positive. This replaces the nearest-center `close` veto, which rejected real short pentagon connectors.
- New `ring_centers_graph` takes accepted centers, tests Delaunay-neighbor sides against sharp ridge evidence, selects 5/6 sides from neighbor-angle consistency, and calls `graph_from_fused_rings`. The output rings share bond and vertex IDs by construction. It checks one connected dual, ring size 5/6, carbon degree at most 3, and `V-E+F=1`. Degree conflicts can remove a redundant weakest-wall side, but **no side was removed** in this review run; any such removal is printed by the diagnostic.
- `python3 -u doc/export_invAFM/scripts/testplot_img2mol.py --all --filters` regenerates detector plots. `python3 -u doc/export_invAFM/scripts/testplot_img2mol.py --all --filter-skeleton` regenerates skeleton overlays and exits with an error if any image cannot form a connected skeleton. Review `debug/testplot_img2mol/filter_skeleton_contact.png`, `debug/testplot_img2mol/big/filter_skeleton.png`, and per-image `ring_centers_E.png` / `filter_skeleton.png`.

### Measured results

| image | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | big |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| detected centers (after fractional scale calibration) | 19 | 19 | 19 | 19 | 19 | 16 | 17 | 17 | 15 | 19 | 20 | 20 | 20 | 20 | 19 |
| connected skeleton | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |

The accepted `big.png` result stays at 19 rings (3 pentagons, 16 hexagons), 27 fused sides, 66 carbons and 84 bonds. After fractional scale calibration, images 1–5 and 10 also give 19-center, 66-atom, 84-bond graphs. All 15 graph builds pass connectivity and topology invariants. This is a topology-invariant check, **not** proof that all detected rings are chemically correct: images 6–9 return only 15–17 centers, and images 11–14 return 20 despite the 19-ring reference.

At the earlier integer-valued scale, scans 2 and 5 failed connected-dual checks. The subpixel refinement shifts scale by up to 0.28 px and restores one center in each; both now give connected 19-ring graphs. This is evidence that the scale estimate matters near detector thresholds, not proof that the extra centers are chemically correct. Image 9 gives a coherent but incomplete 15-ring skeleton; 6–8 also miss rings. Images 11–14 have 20 accepted centers, so their extra site needs review against the known 19-ring reference. Ring count is not forced as an automatic rule.

### Outlook

1. Evaluate angular 3-arm vertex evidence **at corners of the connected topology** and contour-like bond evidence along each proposed shared side. Use these jointly with raw rim enclosure to reject false centers and recover weak peripheral centers; the existing V3 map alone does not yet distinguish missing connectors.
2. Add geometry-proposed missing centers only when their complete rim and adjoining side/vertex evidence supports them. Preserve the rejected status of 2 and 5 until a physical connector is supported.
3. Review the 15 new overlays with the user, then verify molecular export through `AtomicGraph`. The older `img_to_graph` path and XYZ output remain unverified; this detector has not established robustness for arbitrary planar PAHs.

### Geometry-proposed recovery check (before subpixel scale refinement)

The existing `propose_fused_ring_centers` was evaluated on scans 2, 5, and 9. Proposed sites were rescored with the raw fitted rim, all-direction sharp-ray enclosure, center darkness, and plausible shared-side evidence. **No candidate was promoted.** On 9, the strongest apparent rim proposal has ray q10 `-0.325`, so the enclosing wall is incomplete. On 2, the strongest proposal has rim q10 `-4.70` and ray q10 `-0.459`. On 5, one proposal has positive rim and ray evidence, but lies only `1.02b` from an accepted center and its shared-side / vertex evidence is weak; accepting it would create an implausible overlap. The diagnostic is `debug/testplot_img2mol/recovery_hypotheses.png`. This supports retaining the current fail-loud behavior until stronger image evidence is available.

## Progress report, session 5 — scale calibration and shaded bonus scans

### Goal and approach

The goal was to check whether the ring detector and shared-bond skeleton generalize beyond the original repeated scans, using the shaded, tip-asymmetric bonus set. The practical questions were whether a smooth, ring-scale high-pass can reduce the broad sharing artifact without amplifying noise, and whether each image should calibrate its ring scale before the angular convolutions. I kept the existing fuzzy-AND center detector as the baseline, measured its scale estimate on each image, compared several physically scaled background-removal and band-pass settings, then ran the same enclosure and graph checks. The aim was to identify reproducible improvements; neither a visually sharper image nor a plausible center count was treated as proof of a correct molecular graph.

### Code and reproducible diagnostics

- `estimate_bond_scale(flat, subpixel=True)` now fits a parabola to the radial autocorrelation peak. `subpixel=False` retains the former integer-bin result for comparison. This refines the scale estimate before setting convolution radii; it does not by itself establish the chemically correct ring size.
- `doc/export_invAFM/scripts/testplot_img2mol.py` accepts `--img-dir` for numeric PNG sets. Re-run the bonus set with `python3 -u doc/export_invAFM/scripts/testplot_img2mol.py --img-dir /home/prokop/Desktop/PROJECTs/Svec_Ditriptaceneo_Helicene/Images_Bonus/set_2_shade_AFM --all --filters`; outputs go under `debug/testplot_img2mol/set_2_shade_AFM/`. `--filter-skeleton` runs the graph step and fails with a list of rejected images. `--scale-calibration` compares integer and fractional autocorrelation estimates over the original and bonus sets.
- Across 29 images, interpolation shifts `b` by median absolute 0.130 px and maximum 0.279 px, within the parabolic bin's theoretical ±0.289 px limit. Bonus estimates change from 6.93–8.08 px (integer) to 7.03–7.90 px (fractional). The original set remains 8.05–8.79 px; `big.png` changes 37.528→37.695 px.

### Bonus-set result and shading experiment

The directory contains 14 images (1–12, 14, 15), each 128×128. The unmodified filter-bank path finds 15–25 accepted centers per image, but the graph builder rejects all 14: most fused duals are disconnected, with several ring-corner collapses or a degree-4 conflict. The asymmetric lobe adds plausible-looking texture and creates extra centers. No bonus-image molecular skeleton is reliable yet. Review `debug/testplot_img2mol/bonus_set2_raw_contact.png`, `debug/testplot_img2mol/bonus_set2_filter_centers.png`, and the individual filter-response panels under `debug/testplot_img2mol/set_2_shade_AFM/`.

I compared background-removal scales of 6, 8, 16 and 24 px, and band-pass inputs `G(0.18b)*I - G(sigma_bg)*I` at sigma values near 0.8b, 1.0b and 1.2b. Smaller background sigmas sharpen ring contrast, while counts and the ray-enclosure score change non-uniformly; the graph builder still fails on the full set. The default background sigma is 16 px, whose Gaussian high-pass half-amplitude wavelength is about `5.34*sigma ≈ 85 px`, roughly 11b. If the broad sharing spans 2–3 ring pitches (`pitch ≈ sqrt(3)b`), an initial cutoff wavelength is about 3.5–5.2b, corresponding to `sigma_bg ≈ wavelength/5.34 ≈ 0.65–0.98b` (about 5–8 px here). Use a light pre-smoothing scale around 0.15–0.2b to limit pixel noise. This is a physically scaled experiment range, not a selected default: it sharpens views but has not yet improved topology consistently. The image may contain a true tip-asymmetry ghost, which a scalar high-pass cannot remove safely.

### Calibration recommendation

1. Estimate a preliminary `b0` from a smoothly detrended ridge autocorrelation and subpixel peak interpolation before building the angular bank.
2. Evaluate a short per-image scale sweep around `b0` (for example 0.9, 1.0 and 1.1 times `b0`). Prefer ring centers that persist across adjacent scales and pass raw-depth / all-direction rim checks; do not choose the scale by maximizing the raw candidate count.
3. Once several strong centers and fused walls are independently supported, refine from the median center spacing of wall-confirmed hexagon–hexagon pairs divided by `sqrt(3)`. Do not use all nearest-neighbor detections: bonus nearest-neighbor ratios currently span 1.49–1.69b (median about 1.58b) and include false positives and missed sites.
4. Report the chosen scale, sweep scores, center persistence, and spread. If the scale objective is broad or competing peaks exist, retain the initial estimate with an uncertainty range instead of hiding ambiguity in one cutoff.

Next experiment: add the per-image scale sweep as a diagnostic and measure the shading cutoff on the same all-image figures. Keep the existing high-pass default until a ring-level or topology-level improvement repeats across scans and the bonus topology is reviewed.

## Progress report, session 6 — human-annotated reference + shading-removal sweep

### Reference data

`<imgdir>/solution/<n>.png` contains the raw scan with **pure-red dots** marking human-identified ring centers. `red_dot_centers(path)` extracts their centroids (red mask → connected components). IMPORTANT: reference ring counts are NOT uniform — main set is 15–20 per image (9.png→15, 6–8→17, 11–14→20, big→19), bonus set 16–21. Do not assume 19.

Evaluation: `python3 doc/export_invAFM/scripts/testplot_img2mol.py --eval-ref --all [--img-dir <set>] [--prep flat16,flat6,...]` prints per-image `ref/det/hit/FP/FN` (hit = detected within 0.6b of a ref dot).

### Result: main set is essentially solved at ring-center level

`ring_centers_E` on default `flat16` reproduces the human annotation **exactly on all 15 main images: FP=0, FN=0** (279 rings). Two fixes mattered:

- `depth_min` moved to **-0.65** (was -0.5): recovers the last missed ring on 6.png (depth=-0.53, all other evidence strong: Rq10=0.95, Ef=120) while still rejecting the big.png split peak (-0.93/-1.01).
- **NaN bug**: `sqrt(uniform_filter(G*G))` could produce NaN on tiny negative roundoff; one NaN then contaminates the whole map through FFT convolution → det=0 (this broke flat6 on main 8.png). Fixed by `sqrt(np.maximum(...,0))` at all 5 sites.

### Shading-removal comparison (USER review pending)

`prep_image(g, method)` variants + visual grid `debug/testplot_img2mol/<set>/<img>/preprocess_cmp.png` (rows=methods, cols: flat | R ridge | C center | ref+detected). CLI: `--preprocess [--img-dir ...]`.

| method | MAIN FP/FN | BONUS FP/FN | note |
|---|---|---|---|
| flat16 (σbg=16px) | **0 / 0** | 144 / 113 | default |
| flat6 (σbg=0.75b) | 2 / 0 | 96 / 71 | best shading removal |
| flat6n (flat6 + localRMS norm, stabilized) | 3 / 0 | **94 / 63** | best bonus FN; flat map looks cleanest |
| median (bg = median_filter 6b) | – | ~7-13 FP/img | washes ring contrast |
| poly2 (deg-2 surface) | – | ~14 FP/img | residual shading |
| rows (per-scanline median) | – | ~20 FP/img | scan-line artefacts |
| bandpass G(0.2b)-G(1.0b) | – | FP low, FN high | loses weak rings |
| lstd (dev/(locRMS+0.5·median)) | 179/29 | bad | RMS normalization amplifies background noise |

Conclusions so far:
1. σbg=0.75b (`flat6`) removes the broad halo best — matches session-5 estimate σbg≈0.65–0.98b for 2–3-pitch shading. On clean images flat6 is nearly as good as flat16 (2 extra FP total).
2. `flat6n` adds stabilized local-RMS normalization (floor = 0.5·median RMS — without the floor it amplifies flat-area noise catastrophically). Best on the hardest bonus images (8.png: 12 matched/3 FP vs flat6's 7/2 with FN=12).
3. Bonus set remains unsolved: even flat6n has ~94 FP + 63 FN total. FP are dark blob artefacts inside the big asymmetric halo (tip ghost); FN are real rings inside the dark-lobe region. **A scalar high-pass cannot fix a ring-scale ghost** — needs the dual-graph arbitration (fused-neighbor + shared-wall + V3-corner evidence) to demote ghost rings and rescue weak true rings.
4. `lstd`-style full local normalization amplifies background noise — wrong direction.

### Next steps

1. USER reviews `preprocess_cmp.png` grids → pick shading method (or a flat16/flat6 hybrid vote).
2. For ghost rejection: fused-pair support (bright midline wall on R) + V3 junction evidence at predicted shared corners; require ≥1–2 confirmed fused neighbors per ring.
3. For weak rings in dark lobes: contrast-adaptive depth/Rq10 (local, relative to neighboring accepted rings).
4. Only then rebuild the shared-corner graph on the rescued centers (see `ring_centers_graph`, already implemented).

### Session 6b — multi-method consensus + ghost analysis

`ring_centers_multi(g, methods=('flat6','bandpass'), base='flat6n')`: E = geometric mean of per-method fuzzy-AND maps (a real ring survives every preprocessing; artefacts appear in only one). Filtering channels run on `base`'s flat.

- MAIN: 278/279 hit, FP=1 (14.png), FN=1 (6.png — depth differs under flat6n). Essentially still solved.
- BONUS: 207 matched / 83 FP / 59 FN — best combination measured (single flat6n: 203/94/63; flat6+bandpass geo-mean beats it by ~10 FP).

**Why ghosts pass every filter (measured):** FP centers sit ~1.5–2.2b from the nearest real center — at shifted positions of the molecule pattern. The tip-ghost is a faint shifted *copy*, so its fake rings are (a) enclosed on all sides, (b) fused to neighbors with bright shared walls (FP wall-neighbor counts look identical to true rings: most have 2–4), (c) only ~20% separable by scalar features (Ef 1.28 vs 1.76 median; Rq10 0.20 vs 0.39; depth 0.51 vs 0.97 — overlapping distributions). **No single scalar gate will reject them** — they need either (a) a ghost-consistency argument (detect the copy's shift δ from autocorrelation/relative weakness of a whole lobe), or (b) dual-graph arbitration: a ghost lobe fused to the real molecule creates chemically impossible junctions (overcrowded rings, wrong angles, pentagon overcount) that the shared-corner graph can reject.

### Session 6c — U-Net experiment

`doc/export_invAFM/scripts/testplot_img2mol_unet.py` (run with `~/venvs/ML/bin/python` — torch 2.10+cu128 on RTX 3090; skimage was added to that venv). Tiny 3-level U-Net, input channels [flat6n, C, E_geo, R] (i.e. our filter bank is the front-end), target = σ=2px Gaussian heatmap at the human red dots, 8× dihedral augmentation, BCE pos_weight=30, Adam 1e-3, 3000 steps (~15 s). `big.png` excluded (different scale).

- **Trained on main only → memorizes main perfectly (0/0), fails on bonus** (hit~50-70%, FP~8-11/img, FN high): domain shift is real, clean-image features don't transfer to shaded ones.
- **Trained on both, mixed holdout (6 test imgs)**: held-out hit≈all rings (FN 0-2) but FP ~5-14/img — **the U-Net is a high-recall / low-precision detector**. Its heatmap lights up both true rings AND ghost rings (it learned "ring-looking" — ghosts are).
- **Hybrid (classical ∧ U-Net support, P>0.35 at candidate)**: bonus FP 83→68, hits unchanged; main unchanged. Modest but free improvement; `unet.pt` is saved in `debug/testplot_img2mol/unet/`.
- Verdict (pre-augmentation): ML helps as an extra evidence channel but does NOT solve ghost rejection.

### Session 6d — full augmentation, U-Net works

Replaced dihedral-only aug with GPU-side `augment()`: random rotation (any angle), scale 0.85–1.15, anisotropic warp, shift, mirror via `affine_grid`/`grid_sample` (geometry applied to X and Y); photometric on X only — low-frequency shading field, per-row scan-line stripes, pixel noise. 6000 steps.

- **Mixed holdout (6 test imgs spanning both sets): TEST totals 114 hits / 3 FP / 0 FN** — all 6 held-out images essentially perfect, including bonus/10 (21/21) and main/6 (recovers the ring the classical pipeline misses). Train images FP 0–4, FN 0–2.
- Training on main only + aug → bonus: recall much better (FN 1–5) but FP 14–23 — **the bonus labels themselves are what teach ghost rejection**; augmentation alone can't substitute for seeing the ghost domain.
- Conclusion flips: with heavy aug + both domains in training, the ~200k-param U-Net **outperforms the whole classical pipeline** (bonus: ~2 FP / ≤1 FN vs 83/59). Local ghost ambiguity is resolved by learned context, not by more filters.
- Caveat: thresholds are per-image best-F1 sweep (upper bound); a single global threshold will be slightly worse. Next: feed U-Net centers into `ring_centers_graph`/`graph_from_fused_rings`; also try U-Net to predict walls/vertices (multi-task) since centers alone worked this well.

### Session 7 — vertex/edge filters, the outer halo rim

New kernel banks in `img2mol.py` (same conventions: zero-mean, L2-normalized, sizes in units of b):

- `vertex_kernels` — 'Y' sp2 junction: `K = exp(-d_Y^2/2 sig^2) * decay(r) - lam * valley`. d_Y = distance to the Y skeleton (center + 3 arm rays, length ~1b) so center and arms share the same ridge value/width; `decay(r)=exp(-r^2/2(0.6b)^2)` makes the center the global max with arms fading out; `valley` = broad angular Gaussians (35 deg) on the three +60 deg bisectors times a radial window — the negative part fills the whole wedge space between arms like background (NOT narrow anti-ridges).
- `edge_kernels` — linear double-edge: along-axis profile `w(t)` = narrow bumps at t=+-d, negative center, negative skirt (NOT a radial annulus); across-axis Gaussian sigu=0.25b. `ds` tunable per ring size. Response logic per the user: `Resp(phi_k) = clip(max_d edge_{d,phi_k},0)` then `prod_k Resp_k` — strips intersect at ring positions (verified visually: the product map has compact blobs on ring centers). Do NOT max over orientations — that destroys the intersection localization.

**The outer rim (halo) problem.** Around the molecular boundary there is another ridge ~1-2 Angstrom outside the outermost bonds (tip convolution / molecule-substrate contrast step — it survives flat6n because it is ring-scale). The thin crescent band between the real rim and this halo passes every enclosure test: it has walls in all directions, just too close together. It produces the outer shell of false ring detections. Cannot be removed by kernel shape — it IS a real enclosed region. Two discriminator ideas (user):

1. **Peel the first onion layer** (non-convolution image operation): assume the outer rim always exists; identify the outermost ring/center layer (e.g. via distance-from-molecule-boundary or convex-layer ordering of detected centers) and drop it wholesale.
2. **Smoothness/slope asymmetry**: the halo ridge is much smoother (smaller slope and curvature) than a true bond rim, especially on its *outer* onset going inward — a real C-C bond wall is sharp on both flanks, the halo has a soft ramp from the substrate side. Measurable per-candidate: cast rays outward, compare outer-flank slope/curvature of the first wall vs the inner walls; or compare rim sharpness statistics for suspected outer-layer rings vs interior rings (interior gives the reference slope).

Related measured features (partial separators): nearest-wall distance `dmin` — real ring ~0.63b in all directions, halo crescent ~0.50b on the molecule side (main: kills ~45% FP at 0 real loss; bonus ~46% at ~10% real cost). Angular wall-distance spread is NOT usable (real rim rings have no wall on the outward side -> large spread legitimately).

**Minimum-distance prior (USER, essential).** Two different carbons can NEVER be closer than a bond length (~1.3-1.5 A). Consequences that MUST guide skeleton repair:
- Two ring centers at fused-pair distance (~1.3-1.9b) MUST share an edge — there is no other way to pack polygons that close. Never drop such a pair to fix a violation; the violation is instead a wrong side assignment, a wrong n, a missing third pair, or a false-positive ring.
- Conversely, if two rings' claimed corners land < ~b apart without being unioned, they are the SAME atom (or one ring is false/misplaced).
- Degree-4 carbons come from 'V' triple junctions: rings a,b,c meet at one vertex via fused pairs (a,b)+(b,c) with (a,c) missing. Physically impossible — the correct repair is to ADD the missing pair (a,c) when geometrically plausible (d < ~2.4b), giving the coronene-type triple junction. Removing a pair creates disconnected hexagons sitting closer than a bond length — equally impossible.

---

## Session 8 — constrained side assignment + shared-vertex refinement (best result so far)

### Detection checkpoint identification (regression resolved)

The earlier "much better" results were produced by checkpoint `unet_s3.pt`, NOT the default `unet.pt` which `--final` was silently loading. Measured with identical inputs/threshold 0.5, s3 reproduces the recorded best exactly: main 260 hits/11 FP/0 FN, bonus 265/24/1. Default unet.pt: main 98 FP, bonus 129 FP/5 FN. **Lesson: always pass `--ckpt` explicitly**; checkpoints are seeds, not quality labels.

The raw-Ef enclosure veto (`weak` in `final_grid`) is destructive on shaded images: with s3 it turns bonus 24 FP/1 FN into 22 FP/**23 FN** — losing 22 true rings to remove 2 FPs (e.g. -6 hits each on bonus/2.png and 12.png). Its `weak.mean()<0.35` auto-disable heuristic does not reliably detect shading. Do not use it on bonus; on main it removes most FPs at ~zero cost.

### Constrained reconstruction (new path, `constrained_ring_graph`)

Replaces the free per-ring Hungarian side assignment — the root cause of degree-4/corner-collapse failures — with a topology-constrained assignment:

- All Delaunay edges in the fused window 0.85b<d<2.2b are candidates (no wall gate; wall score kept for diagnostics). Candidate variants = full set + each single-edge deletion that keeps the dual connected.
- At each ring, neighbors are cyclically ordered; each angular gap gets an integer polygon side-step: **exactly 1 iff the gap closes a triangular face** (the two neighbors also share a pair AND gap<pi), else >=2 — two fused neighbors at adjacent sides MUST meet at a common atom, otherwise the corner is physically impossible.
- n in {5,6} per ring picked by layout cost (mean sq. angular residual + 0.02*(6-n), hexagon wins ties).
- Exact corner union (`graph_from_fused_rings(assignment=...)`) — atoms are shared by construction, never merged by proximity.
- **SLSQP vertex refinement** (positions normalized by b): min sum|v-v0|^2 + 4*sum|mean(v[ring])-center|^2 subject to EVERY distinct pair >=0.95b and every bonded pair <=1.05b — the user minimum-distance prior enforced as a hard constraint, not a check afterward. ~1.33 A at 1.4 A/b.
- Rejects: disconnected carbon graph, V-E+F != 1, edge shared by >2 rings, degree>3, disjoint-edge crossings, collinear overlaps, inverted/degenerate faces.
- **Never drops a ring.** Failure returns UNRESOLVED — displayed with orange detections, not a misleading partial skeleton.

Measured on frozen s3 raw detections (`doc/export_invAFM/scripts/testplot_img2mol_unet.py --reconstruct-audit`, artifacts in `debug/testplot_img2mol/unet/`):

| Set | Admissible | Unresolved | Notes |
|-----|-----------|------------|-------|
| main | 10/14 | 8,9,10,13 | All admissible: 0 dropped, min sep >=0.95b, bonds <=1.05b, connected |
| bonus | 11/14 | 9,10,12 | Same invariants; largest retained FP count still included |

Synthetic checks: fused pair 10V/11E/2F, coronene 24V/30E/7F, noisy (sigma=0.03b) passes all constraints, reflection parity, V-triple gives 13V/15E/3F without degree-4, default path regression unchanged.

**User verdict: `constrained_main.png` is almost perfect; `constrained_bonus.png` much worse (FPs on nearly every image) but still the best achieved.**

### Remaining FP taxonomy (USER observations on constrained grids)

1. **Rotated-image rim corners** — main/10.png, 8.png: FP sits at the image corner where rotation/scan artifacts create a partial rim; essentially one enclosed patch at the frame edge.
2. **Noise strips** — main/9.png: a scan-line/stripe artefact produces a fake enclosed region.
3. **Half-enclosed corner rings** — main/13.png, 14.png: FP at the image edge that is surrounded by a bright rim on only ~3/6 sides — should be easy to filter: real rings are enclosed from all (or all-but-outward) sides; a rim fragment covering only half the circumference is not a ring. Related to `dmin`/`maxgap` features already measured — check whether these FPs fail q10/maxgap at relaxed thresholds or need a "fraction of directions with wall in annulus" criterion.
4. **Bonus set**: FPs on nearly every image — halo crescents (session-7 problem) + ghost-lobe rings (contrast-repeat artefacts). These are genuinely ring-like in the feature maps; expect the multi-task net (below) to do the discrimination rather than more hand features.

### Planned work (from USER feedback — not yet implemented)

**A. Training-data augmentation with nonsense features.** The U-Net must learn to ignore non-molecular artefacts by seeing them: random noise strips/scan lines, image corners and fake edges from cropping, JPEG-like compression artefacts, bright bar/stripe patterns, partial rims at frame borders. Goal: artifacts produce no center/edge/vertex response anywhere, not just suppressed peaks at test time.

**B. Multi-task U-Net: predict ring-map + edge-map + vertex-map simultaneously (PLANNED, do not implement yet).** Same network, three output heads/channels trained on targets derived from the known molecular graph: center = Gaussian at ring centers (current), edge = ridge along bond lines, vertex = bump at atom positions. Rationale (user): a net that must explain all three levels simultaneously "understands the chemistry" — a blob that is only corner-surrounded cannot produce coherent edge+vertex responses, so FPs get rejected by cross-channel inconsistency. Downstream: graph reconstruction scores proposed rings/edges/vertices against all three maps — much better chance of assembling the correct skeleton, and gives parity-check channels for `repair`/`constrained_ring_graph` validation.

**C. Corner-FP filter (easy, pending):** require the candidate ring to be enclosed by bright rim on ~all sides (or all-but-outward for rim rings); a "3/6 sides only" patch is rejected. Combine with the corner/edge proximity flag from item 1.

### Artifacts this session

- `debug/testplot_img2mol/unet/constrained_main.png`, `constrained_bonus.png` — 3-col audit grids (original | s3 raw + ref | constrained skeleton or UNRESOLVED)
- `debug/testplot_img2mol/unet/constrained_audit.json` — per-image metrics (hits/FP/FN, atoms, bonds, pentagons, min_sep_b, max_center_shift_b)
- `debug/testplot_img2mol/unet/constrained_graphs.npz` — saved vertex/edge/ring arrays for all admissible cases
- `debug/testplot_img2mol/unet/checkpoint_comparison_{main,bonus}.png` — default vs s3 checkpoint, raw vs veto
- `debug/testplot_img2mol/unet/checkpoint_audit.npz` — frozen detections for both checkpoints
