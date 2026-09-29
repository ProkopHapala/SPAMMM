# Img2Mol — extract carbon skeleton graph from AFM/STM images of planar PAHs

Source data: `/home/prokop/Desktop/PROJECTs/Svec_Ditriptaceneo_Helicene/Images/`
(presentations Ditriptyceno[7]helicene; images `1.png`-`14.png` are noisy low-res
144-203 px crops, `big.png` 800x512 is the high-quality reference).

Code: workhorse `spammm/img2mol.py`, demo `tests/testplot_img2mol.py`,
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
- Stages-plot infrastructure: `tests/testplot_img2mol.py`,
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
- For repeated scans of this same molecular pattern, `align_ring_template` fits translation, scale, rotation, and reflection to soft ring proposals. It is a *diagnostic prior*: a transformed template site is not accepted as a ring merely because the template predicts it. Run `python3 -u tests/testplot_img2mol.py --all --template-big`; inspect `debug/testplot_img2mol/template_contact_1-14.png` and per-image `template_centers.png` files. Green means a nearby image proposal inside the molecule mask; orange means a weak interior hypothesis; red X means outside the mask.

### What does not yet work

- Proposal-supported sites per small image, in image order 1–14: `18, 18, 14, 16, 18, 16, 13, 14, 13, 16, 15, 13, 14, 15` out of 19. These counts measure proximity to a soft image proposal, **not** proven ring accuracy. Images 1, 2, and 5 are promising; 7, 9, and 12 remain difficult. The severe scan artifact in 9 produces spurious proposal evidence.
- A similarity transform cannot explain all local distortions. Some orange sites appear to be real rings with weak contrast; some are misplaced. Mask rejection catches a few exterior predictions but does not establish that every interior site is enclosed by a bright rim.
- The accepted big-image graph is not yet automatically recovered from each low-resolution image. The original `img_to_graph` path remains unreliable, and template alignment is specific to repeated scans of one topology. Do not treat its output as a generic PAH reader or export a trusted XYZ from these small scans yet.

### Next direction

1. Refine each template-predicted center against *local enclosed rim* evidence, with one-to-one assignment to observed center proposals; reject exterior and scan-line artifacts explicitly. Keep weak interior sites as hypotheses rather than auto-filled rings.
2. Select centers and fused sides together using bright shared-wall evidence and carbon valence/Euler constraints; score local geometric deformation of the template. For unrelated molecules, infer the dual without a known 19-ring template.
3. Show per-image ring and bond overlays for human review, then validate chemistry and `AtomicGraph` export. Keep task status unverified until user confirms the low-resolution results.
