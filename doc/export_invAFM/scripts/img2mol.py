"""img2mol.py — extract carbon skeleton graph from AFM/STM images of planar PAHs.

Rings-first pipeline (see doc/Tasks/Img2Mol.md for spec + chemical priors):
  load_gray, flatten_bg, molecule_mask   image prep (heavy smoothing, LoG only)
  ridge walls        -LoG response -> bright bond ridges
  ring centers       distance-transform peaks of non-wall interior (robust to
                     merged holes: fused double hole still gives 2 peaks)
  ring regions       watershed of -D -> per-ring interior region
  ring shape fit     harmonic fit r(th) ~ a0 + e*harm1 + A*harm_n, n in {5,6}
                     -> ring size (strong hex prior), polygon phase, mean radius
  ring adjacency     center pairs < ~1.9b AND bright wall along midline
  vertices           per-ring ideal polygon corners (phase-locked, R = b_loc)
                     clustered over all rings -> atoms (sp2 C = 2-3 rings)
  edges              consecutive ring corners; fused edges shared automatically

Bond scale b auto-estimated from fused center distances (d ~ 1.73b hex-hex)
and the whole pass re-runs at the corrected scale.

Step 2: graph_to_atomicgraph -> AtomicGraph -> XYZ (doc/export_invAFM/scripts/testplot_img2mol.py).
"""
import itertools, os
import numpy as np
import scipy.ndimage as ndi
from scipy import signal
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import skeletonize, binary_closing, binary_dilation, disk, remove_small_objects
from skimage.morphology import reconstruction
from skimage.feature import peak_local_max

K8 = np.ones((3, 3), int); K8[1, 1] = 0          # 8-neighbour kernel
NB = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
APO = {5: 0.6882, 6: 0.8660}                     # apothem / side for n-gon
CIRC = {5: 0.8507, 6: 1.0}                       # circumradius / side


def load_gray(path):
    a = np.asarray(Image.open(path)).astype(np.float64)
    g = a[..., :3].mean(-1) if a.ndim == 3 else a
    if a.ndim == 3 and a.shape[2] == 4:                 # RGBA: fill transparent with bg level
        alpha = a[..., 3] / 255.0
        g = np.where(alpha > 0.5, g, np.median(g[alpha > 0.5]))
    return g


def red_dot_centers(path):
    """Reference ring centers: centroids of pure-red marker dots drawn on the
    solution image (human annotation). Returns (N,2) xy."""
    if not os.path.isfile(path):
        return np.zeros((0, 2))
    a = np.asarray(Image.open(path)).astype(int)
    m = (a[..., 0] > 150) & (a[..., 0] - a[..., 1] > 60) & (a[..., 0] - a[..., 2] > 60)
    lab = label(m)
    if lab.max() == 0:
        return np.zeros((0, 2))
    return np.array([ndi.center_of_mass(m, lab, i)[::-1] for i in range(1, lab.max() + 1)])


def flatten_bg(g, sig_bg=None):
    sig_bg = sig_bg or min(g.shape) / 8.0
    flat = g - ndi.gaussian_filter(g, sig_bg)
    return flat - np.median(flat)


def prep_image(g, method, b=None):
    """Flattened image variants for shading/background removal. Returns flat."""
    b = b or estimate_bond_scale(flatten_bg(g))   # bootstrap scale from default flat
    if method == 'flat16':  return flatten_bg(g)
    if method == 'flat6':   return flatten_bg(g, 0.75 * b)
    if method == 'median':  # background = robust median filter r~3b
        flat = g - ndi.median_filter(g, size=int(6 * b) | 1)
        return flat - np.median(flat)
    if method == 'poly2':   # least-squares deg-2 polynomial surface
        yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]] / np.array(g.shape)[:, None, None]  # normalized coords
        A = np.stack([np.ones(xx.size), xx.ravel(), yy.ravel(), (xx * yy).ravel(), (xx * xx).ravel(), (yy * yy).ravel()], 1)
        c, *_ = np.linalg.lstsq(A, g.ravel(), rcond=None)
        flat = g - (A @ c).reshape(g.shape)
        return flat - np.median(flat)
    if method == 'rows':    # per-scanline median (AFM line shading) then column mean
        flat = g - np.median(g, axis=1, keepdims=True)
        return flat - np.median(flat)
    if method == 'bandpass':# G(0.2b) - G(1.0b) band-pass, noise-stabilized smooth
        return ndi.gaussian_filter(g, 0.2 * b) - ndi.gaussian_filter(g, 1.0 * b)
    if method == 'lstd':    # noise-stabilized high-pass: dev / (localRMS + floor); floor kills noise blow-up in flat areas
        flat = flatten_bg(g, 1.0 * b)
        dev = flat - ndi.gaussian_filter(flat, 0.2 * b)
        rms = np.sqrt(np.maximum(ndi.uniform_filter(dev * dev, int(3 * b)), 0))
        return dev / (rms + 0.5 * np.median(rms))
    if method == 'flat6n':  # flat6 + stabilized local-RMS normalization
        flat = flatten_bg(g, 0.75 * b)
        rms = np.sqrt(np.maximum(ndi.uniform_filter(flat * flat, int(3 * b)), 0))
        return flat / (rms + 0.5 * np.median(rms))
    raise ValueError(f'unknown prep method {method}')


def molecule_mask(flat, r_close):
    dev = np.abs(flat)
    mask = dev > threshold_otsu(dev)
    mask = binary_closing(mask, disk(max(1, int(r_close))))
    mask = ndi.binary_fill_holes(mask)
    mask = remove_small_objects(mask, 64)
    lab = label(mask)
    if lab.max() > 0:                                    # keep largest blob = molecule
        sizes = ndi.sum(mask, lab, range(1, lab.max() + 1))
        mask = lab == 1 + np.argmax(sizes)
    return mask


def ring_centers(flat, inner, D, b):
    """Ring centers = +LoG dark-blob peaks inside wall-bounded interior.
    Robust vs noise (heavy smoothing) and merged holes (blob max per ring)."""
    G = ndi.gaussian_laplace(flat, 0.45 * b)
    pk = peak_local_max(G, min_distance=max(2, int(0.55 * b)),
                        threshold_abs=np.percentile(G[inner], 70) if inner.any() else 0)
    keep = D[pk[:, 0], pk[:, 1]] > 0.25 * b if len(pk) else pk
    return pk[keep].astype(float)


def ring_candidates(flat, R, b):
    """Unvetoed dark-interior proposals with soft annular wall evidence.

    A broken wall may lower the score but cannot remove a candidate. The
    chemical graph, rather than a closed-hole threshold, selects rings.
    Returns xy positions and (center curvature, mean wall, lower-quartile wall).
    """
    G = ndi.gaussian_laplace(flat, 0.45 * b)
    pk = peak_local_max(G, min_distance=max(2, int(0.55 * b)), threshold_abs=np.percentile(G, 60), exclude_border=max(2, int(b)))
    bs = 0.65 * b
    Gs = ndi.gaussian_laplace(flat, 0.45 * bs)
    pk = np.vstack((pk, peak_local_max(Gs, min_distance=max(2, int(0.55 * bs)), threshold_abs=np.percentile(Gs, 60), exclude_border=max(2, int(bs)))))
    if not len(pk):
        return np.zeros((0, 2)), np.zeros((0, 4))
    th = 2 * np.pi * np.arange(36) / 36
    Rs = -ndi.gaussian_laplace(flat, max(0.8, bs / 7.0))
    scores = []
    for radius, ridge, center in ((b, R, G), (bs, Rs, Gs)):
        wall = np.maximum.reduce([ndi.map_coordinates(ridge, [pk[:, 0, None] + r * np.sin(th), pk[:, 1, None] + r * np.cos(th)], order=1, mode='nearest') for r in (0.7 * radius, 0.9 * radius, 1.1 * radius)])
        scale = np.percentile(ridge[ridge > 0], 90)
        scores.append(np.stack([center[pk[:, 0], pk[:, 1]], wall.mean(axis=1) / scale, np.quantile(wall, 0.25, axis=1) / scale], axis=1))
    # use_small = scores[1][:, 1] > scores[0][:, 1]
    # score = np.where(use_small[:, None], scores[1], scores[0])
    # score = np.column_stack((score, np.where(use_small, bs, b)))
    score = np.column_stack((scores[0], np.full(len(pk), b)))
    order = np.argsort(score[:, 1])[::-1]
    keep = []
    for i in order:
        if not keep or np.min(np.linalg.norm(pk[keep] - pk[i], axis=1)) > 0.55 * b:
            keep.append(i)
    return pk[keep, ::-1].astype(float), score[keep]


def center_response(flat, b):
    """Ring-center response: Mexican-hat/annulus-bump convolution.

    A ring interior is a dark disk enclosed by a bright rim at ~0.7b:
    annulus bump at r ~ apothem, valley at the center (a 'bump-at-radius'
    kernel). -LoG at sigma 0.45b is exactly that kernel; local-RMS
    normalization keeps it stable against shading/ghost contrast.
    Ring centers = LOCAL MINIMA."""
    G = -ndi.gaussian_laplace(flat, 0.45 * b)   # annulus bump at ~0.7b around center
    rms = np.sqrt(np.maximum(ndi.uniform_filter(G * G, max(3, int(2 * b))), 0)) + 1e-9
    return G / rms


def propose_centers(C, mask, b, zmin=0.6):
    """Candidate ring centers = local minima of C inside (dilated) mask.

    The minima test is just peak_local_max(-C); no enclosure/flood vetoes —
    scoring and global chemistry arbitrate later."""
    gate = binary_dilation(mask, disk(max(1, int(0.75 * b))))
    pk = peak_local_max(-C, min_distance=max(2, int(0.7 * b)),
                        threshold_abs=zmin, exclude_border=max(2, int(b)))
    if not len(pk):
        return np.zeros((0, 2))
    xy = pk[:, ::-1].astype(float)
    return xy[gate[pk[:, 0], pk[:, 1]]]


def sector_kernels(b, nsect=8, rads=(0.68, 0.90), sr=0.11):
    """Radial x angular ring-wavelet filter bank.

    Angular: periodic QUADRATIC B-SPLINE windows -> partition of unity,
    sum_j ang_j(theta) = 1 exactly, so sum_j K_ij = isotropic radial wavelet.

    Radial: zero-mean wavelet  w(r) = bump(r-r_i) - lam * neg(r)  where neg is
    the inner disk plus outer skirt (everything inside r<R3 minus the bump
    support) and lam enforces integral r*w dr = 0.  Shape: NEGATIVE at the
    ring interior, POSITIVE on the rim, NEGATIVE just outside, ZERO far out.
    DC-free -> insensitive to slow contrast/brightness variation.

    Each kernel is L2-normalized (sum K_ij^2 = 1) so responses are
    commensurate across sectors, radii and images.

    Enclosure logic (fuzzy AND over sectors, max over radii):
      S_ij(p) = corr(img, K_ij)(p)
      M_j(p)  = max_i S_ij(p)          -- rim present at SOME radius
      E(p)    = ( prod_j clamp(M_j,0) )^(1/nsect)
    A true ring center has a wall in EVERY direction at ~pent/hex apothem;
    one missing sector -> E=0. Each kernel averages 30+ px -> noise cancels."""
    h = int(np.ceil(1.45 * b))
    y, x = np.mgrid[-h:h + 1, -h:h + 1]
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    dth = 2 * np.pi / nsect
    ks = np.zeros((len(rads), nsect, 2 * h + 1, 2 * h + 1))
    for i, r0 in enumerate(rads):
        sig = sr * b
        pos = np.exp(-0.5 * ((r - r0 * b) / sig)**2) * (np.abs(r - r0 * b) < 2.5 * sig)
        neg = (r < r0 * b + 4 * sig) & (np.abs(r - r0 * b) >= 2.5 * sig)
        lam = (pos * r).sum() / max((neg.astype(float) * r).sum(), 1e-12)
        w = pos - lam * neg                                   # integral r*w dr = 0
        for k in range(nsect):
            u = np.abs((th - k * dth + np.pi) % (2 * np.pi) - np.pi) / dth
            ang = np.where(u < 0.5, 0.75 - u * u, np.where(u < 1.5, 0.5 * (1.5 - u)**2, 0.0))
            K = w * ang
            K -= K.mean()                                     # exact discrete DC rejection
            ks[i, k] = K / np.sqrt((K * K).sum())             # L2-normalized
    return ks


def sector_enclosure(img, ks):
    """E = fuzzy-AND over angular sectors of radially-maxed sector responses.
    Returns (E, M) with M[j] = max_i S_ij."""
    nrad, nsect = ks.shape[0], ks.shape[1]
    S = np.stack([signal.fftconvolve(img, ks[i, k][::-1, ::-1], mode='same')
                  for i in range(nrad) for k in range(nsect)])
    S = S.reshape(nrad, nsect, *img.shape)
    M = S.max(axis=0)
    E = np.prod(np.clip(M, 0, None), axis=0)**(1.0 / nsect)
    return E, M


def ray_enclosure(R, xy, b, nray=36, r12=(0.35, 1.3), nr=20):
    """Ray-cast enclosure: peak ridge response along nray rays from each center.
    Returns dict(q10, min, maxgap): 10th-percentile and min over rays of the
    radius-max ridge response pk_ray, plus the longest circular run of rays
    with pk_ray <= 0.8 in degrees. xy: (N,2); R: normalized sharp ridge map."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    th = 2 * np.pi * np.arange(nray) / nray
    rr = np.linspace(r12[0], r12[1], nr) * b
    ys = xy[:, 1, None, None] + rr[None, None, :] * np.sin(th)[None, :, None]
    xs = xy[:, 0, None, None] + rr[None, None, :] * np.cos(th)[None, :, None]
    S = ndi.map_coordinates(R, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(xy), nray, nr)
    pk = S.max(axis=2)                                   # (N, nray)
    maxgap = np.zeros(len(xy))
    for i, m in enumerate(np.concatenate([pk <= 0.8, pk <= 0.8], axis=1)):
        if m.all():
            maxgap[i] = nray
        else:
            z = np.concatenate([[-1], np.flatnonzero(~m), [2 * nray]])
            maxgap[i] = min(np.diff(z).max() - 1, nray)
    return dict(q10=np.quantile(pk, 0.1, axis=1), min=pk.min(axis=1), maxgap=maxgap * 360.0 / nray)


def ring_centers_multi(g, methods=('flat6', 'bandpass'), base='flat6n', **kw):
    """Consensus detection: E = geometric mean of per-method fuzzy-AND maps
    (artefacts appear in only one preprocessing, real rings in all).
    Filtering channels (ray/depth/Ef) run on the `base` method's flat."""
    flats = {m: prep_image(g, m) for m in methods}
    Es = []
    for m in methods:
        bm = estimate_bond_scale(flats[m])
        Cm = center_response(flats[m], bm)
        Es.append(sector_enclosure(Cm, sector_kernels(bm))[0])
    E_geo = np.prod(np.stack(Es), axis=0) ** (1.0 / len(Es))
    flat = prep_image(g, base)
    return ring_centers_E(flat, estimate_bond_scale(flat), E=E_geo, **kw)


def ring_centers_E(flat, b, Ef_rel=0.25, dmax=2.2, depth_min=-0.65, E=None):
    """Ring centers = peaks of fuzzy-AND enclosure E (on C), filtered by
    ray-cast ridge enclosure (Rq10>0), raw-image enclosure Ef, raw ring depth
    (reject a center brighter than most of its surrounding rim), and
    connectivity (isolated singletons out; appendices detached by a weak
    connector ring are kept — the graph builder handles that later).
    Returns xy (N,2) accepted and per-peak diagnostics including raw depth."""
    C = center_response(flat, b)
    ks = sector_kernels(b)
    if E is None:
        E, M = sector_enclosure(C, ks)
    else:
        M = sector_enclosure(C, ks)[1]
    Ef, _ = sector_enclosure(flat, ks)
    Rs = -ndi.gaussian_laplace(flat, 0.2 * b)
    R = Rs / (np.sqrt(np.maximum(ndi.uniform_filter(Rs * Rs, int(3 * b)), 0)) + 1e-9)
    pk = peak_local_max(E, min_distance=max(2, int(0.7 * b)), threshold_abs=0.05, exclude_border=max(1, int(b)))
    if not len(pk):
        z = np.zeros(0)
        return np.zeros((0, 2)), dict(xy=np.zeros((0, 2)), E=z, Ef=z, C=z, U=z, Rq10=z, Rmin=z, maxgap=z, depth=z, reason=np.zeros(0, dtype=object))
    xy = pk[:, ::-1].astype(float)
    rays = ray_enclosure(R, xy, b)
    Mv = M[:, pk[:, 0], pk[:, 1]]                        # (nsect, N)
    U = Mv.min(axis=0) / np.maximum(Mv.mean(axis=0), 1e-9)
    Cv = C[pk[:, 0], pk[:, 1]]
    Ev = E[pk[:, 0], pk[:, 1]]
    Efv = Ef[pk[:, 0], pk[:, 1]]
    sharp_flat = ndi.gaussian_filter(flat, 0.2 * b)
    rr = np.linspace(0.35, 1.3, 20) * b
    ys = xy[:, 1, None, None] + rr[None, None, :] * np.sin(np.arange(36) * (2*np.pi/36))[None, :, None]
    xs = xy[:, 0, None, None] + rr[None, None, :] * np.cos(np.arange(36) * (2*np.pi/36))[None, :, None]
    ray_flat = ndi.map_coordinates(sharp_flat, [ys.ravel(), xs.ravel()], order=1, mode='nearest').reshape(len(xy), 36, -1).max(axis=2)
    center_flat = ndi.map_coordinates(sharp_flat, [xy[:, 1], xy[:, 0]], order=1, mode='nearest')
    depth = (np.quantile(ray_flat, 0.1, axis=1) - center_flat) / (np.std(flat) + 1e-9)
    reason = np.full(len(pk), '', dtype=object)
    reason[rays['q10'] <= 0] = 'ray'
    passed = reason == ''
    reason[passed & (Efv < Ef_rel * np.quantile(Efv[passed], 0.9))] = 'Ef'
    reason[(reason == '') & (depth < depth_min)] = 'depth'
    cand = np.flatnonzero(reason == '')
    if len(cand):
        inset = np.zeros(len(pk), bool); inset[cand] = True
        linked = np.zeros(len(pk), bool)
        prs = np.array([(i, j) for i, j in cKDTree(pk).query_pairs(dmax * b) if inset[i] and inset[j]], dtype=int).reshape(-1, 2)
        if len(prs):
            linked[prs[:, 0]] = linked[prs[:, 1]] = True               # component size >= 2 survives
        reason[(reason == '') & ~linked] = 'comp'                      # isolated singletons only
    reason[reason == ''] = 'ok'
    return xy[reason == 'ok'], dict(xy=xy, E=Ev, Ef=Efv, C=Cv, U=U, Rq10=rays['q10'], Rmin=rays['min'], maxgap=rays['maxgap'], depth=depth, reason=reason)


def ring_centers_graph(flat, b=None, wall_min=0.0):
    """Detect enclosed centers, infer fused sides from wall evidence, build shared-corner skeleton."""
    b = estimate_bond_scale(flat) if b is None else float(b)
    centers, diag = ring_centers_E(flat, b)
    if len(centers) < 3:
        raise ValueError(f'ring_centers_graph: only {len(centers)} centers survived enclosure filters')
    R0 = -ndi.gaussian_laplace(flat, 0.2 * b)
    R = R0 / (np.sqrt(np.maximum(ndi.uniform_filter(R0 * R0, max(3, int(3 * b))), 0)) + 1e-9)
    tri = Delaunay(centers).simplices
    candidate = sorted({tuple(sorted((int(i), int(j)))) for t in tri for i, j in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))})
    pairs = [(i, j, float(np.linalg.norm(centers[i] - centers[j])), float(fused_wall_score(R, centers[i], centers[j], b))) for i, j in candidate]
    pairs = [p for p in pairs if 0.85 * b < p[2] < 2.2 * b and p[3] > wall_min]
    adj = [[] for _ in centers]
    for i, j, *_ in pairs:
        adj[i].append(j); adj[j].append(i)
    if any(not nbs for nbs in adj):
        raise ValueError(f'ring_centers_graph: {sum(not nbs for nbs in adj)} accepted centers have no bright fused side')
    reached = {0}; stack = [0]
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if j not in reached:
                reached.add(j); stack.append(j)
    if len(reached) != len(centers):
        raise ValueError(f'ring_centers_graph: fused ring dual is disconnected ({len(reached)}/{len(centers)} reached)')
    ns = {i: (5 if n_score_ang([np.arctan2(centers[j,1]-centers[i,1], centers[j,0]-centers[i,0]) for j in nbs], 5) > n_score_ang([np.arctan2(centers[j,1]-centers[i,1], centers[j,0]-centers[i,0]) for j in nbs], 6) else 6) for i, nbs in enumerate(adj)}
    removed_pairs = []
    while True:
        try:
            verts, edges, rings = graph_from_fused_rings(centers, pairs, ns, b)
            break
        except ValueError as exc:
            if 'degree-' not in str(exc) or len(pairs) <= len(centers) - 1:
                raise
            # Remove the weakest wall proposal only when it is redundant to dual connectivity.
            removed = False
            for k in np.argsort([p[3] for p in pairs]):
                trial = pairs[:k] + pairs[k + 1:]
                seen = {0}; stack = [0]
                while stack:
                    i = stack.pop()
                    for p in trial:
                        j = p[1] if p[0] == i else p[0] if p[1] == i else -1
                        if j >= 0 and j not in seen:
                            seen.add(j); stack.append(j)
                if len(seen) == len(centers):
                    removed_pairs.append(pairs[k]); pairs = trial; removed = True; break
            if not removed:
                raise ValueError(f'ring_centers_graph: cannot remove valence-conflicting side without disconnecting rings ({exc})') from exc
    if any(len(r) not in (5, 6) for r in rings):
        raise ValueError('ring_centers_graph: ring sizes outside 5/6')
    degree = np.bincount(np.asarray(edges).ravel(), minlength=len(verts))
    if degree.max() > 3 or len(verts) - len(edges) + len(rings) != 1:
        raise ValueError(f'ring_centers_graph: invalid skeleton invariants V={len(verts)} E={len(edges)} F={len(rings)} max_degree={degree.max()}')
    return dict(centers=centers, center_diagnostics=diag, pairs=pairs, removed_pairs=removed_pairs, ns=ns, verts=verts, edges=edges, rings=rings, R=R, bond_px=b)


def vertex_kernels(b, armL=1.0, sig=0.10, decay=0.6, nrot=6, rmax=1.35):
    """3-fold 'Y' vertex ridge filters for sp2 carbons (3 bonds, 120 deg).

    K(p) = Y(p)*decay(r) - lam * valley(p):
      Y(p)     = exp(-dY^2/(2 sig^2)), dY = distance to the Y skeleton
               (center + 3 arm rays at 120 deg, length armL*b) -> uniform-width
               ridge, same value center and arms;
      decay(r) = exp(-r^2/(2 (decay*b)^2)) -> MAXIMUM AT CENTER, arms fade out;
      valley   = broad background filling the wedge space BETWEEN the arms:
               angular Gaussians (35 deg) on the +60 deg bisectors times a
               smooth radial window [0.2b, 1.3b].
    Zero-mean, L2-normalized; nrot rotations cover the 120-deg period.
    Response: max over rotations of corr(img, K)."""
    h = int(np.ceil(rmax * b))
    y, x = np.mgrid[-h:h + 1, -h:h + 1]
    r = np.hypot(x, y); th = np.arctan2(y, x)
    ks = np.zeros((nrot, 2 * h + 1, 2 * h + 1))
    for k in range(nrot):
        ph = np.pi / 3.0 * k / nrot * 2.0                       # rotations within the 120-deg period
        dY = r.copy()
        wang = np.zeros_like(r)
        for m in range(3):
            a = ph + 2 * np.pi * m / 3
            ca, sa = np.cos(a), np.sin(a)
            t = x * ca + y * sa                               # along-arm coordinate
            u = -x * sa + y * ca                              # across-arm distance
            dY = np.minimum(dY, np.where((t >= 0) & (t <= armL * b), np.abs(u),
                                         np.hypot(np.minimum(np.abs(t - 0), np.abs(t - armL * b)), u)))
            dv = np.abs((th - (a + np.pi / 3) + np.pi) % (2 * np.pi) - np.pi)   # angular distance to bisector
            wang += np.exp(-0.5 * (dv / 0.6)**2)                # ~35 deg half-width valley wedges
        pos = np.exp(-0.5 * (dY / (sig * b))**2) * np.exp(-0.5 * (r / (decay * b))**2)
        neg = wang * np.clip((r - 0.15 * b) / (0.15 * b), 0, 1) * np.clip((1.30 * b - r) / (0.3 * b), 0, 1)
        lam = pos.sum() / max(neg.sum(), 1e-12)
        K = pos - min(lam, 0.9) * neg
        K -= K.mean()
        ks[k] = K / np.sqrt((K * K).sum())
    return ks


def edge_kernels(b, dirs=3, ds=(0.5, 0.75, 1.0), sigt=0.10, sigu=0.25, rmax=1.6):
    """Linear double-edge filters: along-axis profile with NARROW maxima at
    +-d and negative center + outer skirt; Gaussian ridge across the axis.

    K(t,u) = exp(-u^2/(2 sigu^2)) * w_d(t), w_d = bumps at t=+-d minus
    center lobe minus outer skirt, integral dt = 0 (DC-free), L2-normalized.
    dirs orientations x len(ds) distances -> strips response per orientation;
    product over orientations localizes edge segments / ring corners.
    ds in units of b (e.g. 0.5b hits the two vertices of one bond)."""
    h = int(np.ceil(rmax * b))
    y, x = np.mgrid[-h:h + 1, -h:h + 1]
    ks = np.zeros((len(ds), dirs, 2 * h + 1, 2 * h + 1))
    for i, d in enumerate(ds):
        for k in range(dirs):
            a = np.pi * k / dirs
            ca, sa = np.cos(a), np.sin(a)
            t = x * ca + y * sa                               # along-axis
            u = -x * sa + y * ca                              # across-axis
            pos = np.exp(-0.5 * ((np.abs(t) - d * b) / (sigt * b))**2)
            neg = np.exp(-0.5 * (t / (d * b + sigt * b))**2) + np.exp(-0.5 * ((np.abs(t) - (d + 4 * sigt) * b) / (sigt * b))**2) * 0.5
            lam = pos.sum() / max(neg.sum(), 1e-12)
            w = pos - lam * neg
            K = w * np.exp(-0.5 * (u / (sigu * b))**2)
            K -= K.mean()
            ks[i, k] = K / np.sqrt((K * K).sum())
    return ks


def junction_kernels(b, narm=3, r12=(0.10, 0.80), sig=0.10, nrot=12):
    """N-arm junction filters: ridges along narm rays separated by 2pi/narm,
    extending from r12[0]*b to r12[1]*b out of the vertex, width sig*b.

    narm=3 -> sp2 carbon (three bonds at 120 deg); narm=2 with 180 deg -> a
    straight bond continuation. Correlating and taking the MAX over nrot
    rotations gives the junction evidence map."""
    h = int(np.ceil(1.1 * b))
    y, x = np.mgrid[-h:h + 1, -h:h + 1]
    ks = []
    for irot in range(nrot):
        ph = 2 * np.pi * irot / nrot
        K = np.zeros_like(x, dtype=float)
        for m in range(narm):
            a = ph + 2 * np.pi * m / narm
            ca, sa = np.cos(a), np.sin(a)
            t = x * ca + y * sa                      # along-arm coordinate
            u = -x * sa + y * ca                     # across-arm distance
            K += np.exp(-0.5 * (u / (sig * b))**2) * ((t > r12[0] * b) & (t < r12[1] * b))
        ks.append(K / K.sum())
    return np.asarray(ks)


def junction_response(img, ks):
    """max_rot J_rot: best-aligned junction orientation wins."""
    J = np.stack([signal.fftconvolve(img, K[::-1, ::-1], mode='same') for K in ks])
    return J.max(axis=0)


def estimate_bond_scale(flat, subpixel=True):
    """Estimate C-C pixel scale from radial ridge autocorrelation.

    The ring-center dual has hexagonal pitch sqrt(3)*b. When subpixel=True,
    parabolic interpolation refines the integer-radius autocorrelation peak.
    """
    b0 = min(flat.shape) / 22.0
    R = -ndi.gaussian_laplace(flat, max(0.8, b0 / 7.0))
    ac = signal.fftconvolve(R, R[::-1, ::-1], mode='same')
    yy, xx = np.indices(ac.shape)
    cy, cx = np.array(ac.shape) // 2
    rr = np.hypot(yy - cy, xx - cx).astype(int)
    radial = np.bincount(rr.ravel(), weights=ac.ravel()) / np.bincount(rr.ravel())
    radial = ndi.gaussian_filter1d(radial, 1.0)
    lo, hi = int(1.5 * b0), int(3.5 * b0)
    peaks, _ = signal.find_peaks(radial[lo:hi])
    peaks += lo
    peaks = peaks[radial[peaks] > 0]
    if not len(peaks):
        raise ValueError('No positive ring-spacing peak in ridge autocorrelation')
    peak = peaks[np.argmax(radial[peaks])]
    if not subpixel:
        return peak / np.sqrt(3)
    ym, y0, yp = radial[peak - 1:peak + 2]
    curvature = ym - 2 * y0 + yp
    if not np.isfinite(curvature) or curvature >= 0:
        raise ValueError('Radial autocorrelation peak has no concave parabolic curvature')
    delta = 0.5 * (ym - yp) / curvature
    if not np.isfinite(delta) or abs(delta) > 0.5:
        raise ValueError(f'Invalid subpixel radial-peak offset: {delta}')
    return (peak + delta) / np.sqrt(3)


def fit_center_lattice(xy, scores, b, nseed=30):
    """Fit a triangular ring-center lattice globally from scored proposals.

    Returns origin, two basis vectors, integer site coordinates, and each
    proposal's distance from its nearest lattice site. Ring occupancy is a
    separate chemical selection problem.
    """
    if len(xy) < 3:
        raise ValueError('Need at least three ring-center proposals')
    ids = np.argsort(scores[:, 1])[::-1][:nseed]
    p = xy[ids]
    w = np.maximum(scores[ids, 1], 0)
    w /= max(np.percentile(w, 90), 1e-12)
    d0 = np.sqrt(3) * b
    best = (-np.inf, None, None)
    for i in range(len(p)):
        v = p - p[i]
        ds = np.linalg.norm(v, axis=1)
        near = np.flatnonzero((ds > 0.75 * d0) & (ds < 1.25 * d0))
        for j in near:
            for k in near:
                if j >= k:
                    continue
                ca = np.dot(v[j], v[k]) / (ds[j] * ds[k])
                if not (0.30 < abs(ca) < 0.70):      # triangular: ~+-60 deg (allow shear)
                    continue
                e1 = v[j] / ds[j]
                sgn = np.sign(e1[0] * v[k, 1] - e1[1] * v[k, 0]) or 1.0
                u2 = 0.5 * e1 + sgn * (np.sqrt(3) / 2.0) * np.array([-e1[1], e1[0]])
                djk = 0.5 * (ds[j] + ds[k])
                B = np.stack((djk * e1, djk * u2), axis=1)
                q = np.linalg.solve(B, (p - p[i]).T).T
                err = np.linalg.norm((q - np.rint(q)) @ B.T, axis=1)
                val = np.sum(w * np.exp(-0.5 * (err / (0.27 * d0))**2))
                if val > best[0]:
                    best = (val, p[i], B)
    if best[1] is None:
        raise ValueError('No triangular ring-center lattice among proposals')
    origin, B = best[1], best[2]
    # iteratively refit the (possibly sheared) triangular lattice to ALL
    # proposals: assign to nearest site, re-solve affine least squares.
    # This handles AFM scan distortion without leaving the lattice family.
    wfull = np.maximum(scores[:, 1], 0)
    wfull /= max(np.percentile(wfull, 90), 1e-12)
    for _ in range(8):
        q = np.linalg.solve(B, (xy - origin).T).T
        qi = np.rint(q)
        err = np.linalg.norm((q - qi) @ B.T, axis=1)
        wt = wfull * np.exp(-0.5 * (err / (0.3 * d0))**2)
        A = np.stack([np.ones(len(xy)), qi[:, 0], qi[:, 1]], axis=1)
        cx = np.linalg.lstsq(A * wt[:, None], xy[:, 0] * wt, rcond=None)[0]
        cy = np.linalg.lstsq(A * wt[:, None], xy[:, 1] * wt, rcond=None)[0]
        origin = np.array([cx[0], cy[0]])
        Bls = np.array([[cx[1], cy[1]], [cx[2], cy[2]]])
        # regularize toward ideal triangular (equal length, 60 deg apart):
        # real scan distortion keeps it near-ideal but not exactly
        l1, l2 = np.linalg.norm(Bls[:, 0]), np.linalg.norm(Bls[:, 1])
        phi = np.arctan2(Bls[0, 0] * Bls[1, 1] - Bls[1, 0] * Bls[0, 1], Bls[:, 0] @ Bls[:, 1])
        mid = np.arctan2(Bls[1, 0], Bls[0, 0]) + 0.5 * phi
        pm = 0.5 * np.sign(phi) * np.pi / 3.0
        dm = 0.5 * (l1 + l2)
        Bid = dm * np.array([[np.cos(mid - pm), np.sin(mid - pm)],
                             [np.cos(mid + pm), np.sin(mid + pm)]]).T
        B = 0.4 * Bls + 0.6 * Bid
    q = np.linalg.solve(B, (xy - origin).T).T
    sites = np.rint(q).astype(int)
    err = np.linalg.norm((q - sites) @ B.T, axis=1)
    return origin, B, sites, err


# ---------------------------------------------------------------------------
# Evidence-fusion channels (doc/chats/Image2mol.chat.md):
# independent feature maps; no single channel decides anything.
# ---------------------------------------------------------------------------

def ridge_map_ms(flat, b):
    """Multiscale locally-RMS-normalized ridge (bond wall) response.

    max_s[-LoG_s / localRMS_s] suppresses slow shading/ghost contrast while
    keeping weak walls that sit above their local noise level. Values are
    roughly z-scores: >~1-2 = brighter than local background fluctuations."""
    R = np.full_like(flat, -np.inf)
    for s in (b / 9.0, b / 7.0, b / 5.0):
        Rs = -ndi.gaussian_laplace(flat, max(0.8, s))
        rms = np.sqrt(np.maximum(ndi.uniform_filter(Rs * Rs, max(3, int(2 * b))), 0)) + 1e-9
        R = np.maximum(R, Rs / rms)
    return R


def vertex_response_at(R, pts, b, ndir=12):
    """Junction evidence at sparse points only — convolutional ray sampling.

    At each point integrate the ridge map R along ndir arms of length
    0.2-0.7b; a carbon atom is where bond ridges meet. V3 = best 3-arm at
    120 deg (trigonal sp2 fusion), V2 = best 2-arm at 90-150 deg (rim atom).
    min+mean mixing penalizes a missing arm without vetoing one blurred bond.
    pts: (N,2) xy. Returns V2, V3 per point."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    rr = np.linspace(0.2 * b, 0.7 * b, 4)
    th = 2 * np.pi * np.arange(ndir) / ndir
    A = np.empty((len(pts), ndir))
    for k, a in enumerate(th):
        acc = np.zeros(len(pts))
        for r in rr:
            acc += ndi.map_coordinates(R, [pts[:, 1] + r * np.sin(a), pts[:, 0] + r * np.cos(a)], order=1, mode='nearest')
        A[:, k] = acc / len(rr)
    step3 = ndir // 3
    V3 = np.full(len(pts), -np.inf)
    for p in range(ndir):
        tri = np.stack([A[:, p], A[:, (p + step3) % ndir], A[:, (p + 2 * step3) % ndir]], -1)
        V3 = np.maximum(V3, 0.4 * tri.min(-1) + 0.6 * tri.mean(-1))
    V2 = np.full(len(pts), -np.inf)
    for p in range(ndir):
        for ds in range(3, 6):                        # 90-150 deg separation (ndir=12)
            pr = np.stack([A[:, p], A[:, (p + ds) % ndir]], -1)
            V2 = np.maximum(V2, 0.4 * pr.min(-1) + 0.6 * pr.mean(-1))
    return V2, V3


def bond_evidence_lines(R, pa, pb, b, flank=0.22):
    """Bond-line scores = centerline ridge minus flanking interiors (batched).
    A real bond is a narrow bright line between two dark ring interiors;
    brightness without lateral contrast (wide blob) does not count.
    pa, pb: (M,2) xy endpoints. Returns (M,) scores."""
    pa = np.asarray(pa, float).reshape(-1, 2); pb = np.asarray(pb, float).reshape(-1, 2)
    d = pb - pa
    L = np.linalg.norm(d, axis=1)
    u = d / np.maximum(L, 1e-9)[:, None]; perp = np.stack([-u[:, 1], u[:, 0]], 1)
    t = np.linspace(0.15, 0.85, 7)[:, None, None]
    line = pa[None] + t * d[None]                          # (7, M, 2)
    off = flank * b
    def sm(pts):
        return ndi.map_coordinates(R, [pts[..., 1].ravel(), pts[..., 0].ravel()], order=1, mode='nearest').reshape(pts.shape[:2]).mean(0)
    return sm(line) - 0.5 * (sm(line + off * perp[None]) + sm(line - off * perp[None]))


def bond_evidence(R, pa, pb, b, flank=0.22):
    """Single-segment convenience wrapper around bond_evidence_lines."""
    return float(bond_evidence_lines(R, np.asarray(pa)[None], np.asarray(pb)[None], b, flank)[0])


def ray_profile(c, walls, b, nr=120, rmax_f=1.6):
    """March rays from center (yx) at nr angles; radius of first wall hit.
    Returns (th, rhit, enclosed_frac)."""
    th = np.linspace(0, 2 * np.pi, nr, endpoint=False)
    rr = np.arange(1, rmax_f * b)
    xs = np.clip((c[1] + rr[None, :] * np.cos(th)[:, None]).astype(int), 0, walls.shape[1] - 1)
    ys = np.clip((c[0] + rr[None, :] * np.sin(th)[:, None]).astype(int), 0, walls.shape[0] - 1)
    hit = walls[ys, xs]
    rhit = np.where(hit.any(1), rr[hit.argmax(1)], rmax_f * b)
    return th, rhit, (rhit < 0.95 * rmax_f * b).mean()


def ring_shape(th, rhit):
    """Harmonic fit of ray-hit radius profile -> {n:(resid, amp_rel, phase, rmean)}."""
    rs = ndi.gaussian_filter1d(rhit, 2.0, mode='wrap')
    out = {}
    for n in (5, 6):
        X = np.stack([np.ones(len(th)), np.cos(th), np.sin(th), np.cos(n * th), np.sin(n * th)], 1)
        coef, *_ = np.linalg.lstsq(X, rs, rcond=None)
        resid = np.linalg.norm(rs - X @ coef) / (np.linalg.norm(rs - rs.mean()) + 1e-9)
        out[n] = (resid, np.hypot(coef[3], coef[4]) / max(coef[0], 1e-9),
                  np.arctan2(coef[4], coef[3]), coef[0])
    return out


def ring_adjacency(centers, R, b0, wall_thr):
    """Fused-ring pairs. Two stages so b can't deadlock:
    (a) generous distance window <2.6*b0, compute wall score on midline segment
    (b) wall-bright pairs -> d_fused median -> b = d/1.73 -> rewindow [1.1,2.0]*b
    Returns (pairs[(i,j,d,w)], b_est)."""
    cand = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            if d > 2.6 * b0:
                continue
            t = np.linspace(0.30, 0.70, 9)[:, None]
            pts = centers[i] + t * (centers[j] - centers[i])
            ii = np.clip(pts[:, 0].astype(int), 0, R.shape[0] - 1)
            jj = np.clip(pts[:, 1].astype(int), 0, R.shape[1] - 1)
            cand.append((i, j, d, R[ii, jj].mean()))
            # cand.append((i, j, d, fused_wall_score(R, centers[i, ::-1], centers[j, ::-1], b0)))
    fused = [c for c in cand if c[3] > wall_thr]
    if not fused:
        return [], b0
    b = np.median([c[2] for c in fused]) / 1.73
    return [c for c in fused if 1.1 * b < c[2] < 2.0 * b], b


def cluster_points(pts, rjoin):
    """Cluster xy points (grid hash); returns (vert xy array, assign idx per pt)."""
    verts, vid = [], {}
    assign = np.zeros(len(pts), int)
    for p, xy in enumerate(pts):
        key = tuple(np.round(np.asarray(xy) / rjoin).astype(int))
        hit = -1
        for dk in [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            k = (key[0] + dk[0], key[1] + dk[1])
            if k in vid and np.linalg.norm(np.asarray(xy) - verts[vid[k]]) < rjoin:
                hit = vid[k]; break
        if hit < 0:
            hit = len(verts); verts.append(np.asarray(xy, float)); vid[key] = hit
        assign[p] = hit
    verts = np.asarray(verts).reshape(-1, 2)
    # cluster mean recompute
    acc = np.zeros_like(verts); cnt = np.zeros(len(verts))
    np.add.at(acc, assign, np.asarray(pts, float)); np.add.at(cnt, assign, 1)
    return acc / cnt[:, None], assign


def side_wall_score(R, pa, pb, t_r, lo=0.15, hi=0.85):
    """Mean ridge response along side midsection, normalized by ridge threshold.
    >1 ~ bright wall present; ~0 = dark interior (spurious side). pa/pb in xy."""
    t = np.linspace(lo, hi, 9)[:, None]
    pts = pa + t * (pb - pa)
    ii = np.clip(pts[:, 1].astype(int), 0, R.shape[0] - 1)
    jj = np.clip(pts[:, 0].astype(int), 0, R.shape[1] - 1)
    return R[ii, jj].mean() / t_r


def fused_wall_score(R, ci, cj, b):
    """Ridge score along the shared side perpendicular to two ring centers."""
    u = (cj - ci) / np.linalg.norm(cj - ci)
    perp = np.array([-u[1], u[0]])
    pts = 0.5 * (ci + cj) + np.linspace(-0.35, 0.35, 9)[:, None] * b * perp
    return ndi.map_coordinates(R, [pts[:, 1], pts[:, 0]], order=1, mode='nearest').mean()


def fit_ring_rims(flat, xy, b):
    """Fit a coherent bright 5/6-sided rim around each proposed dark center.

    All angular samples use one polygon and one radius; an unrelated bright
    feature at a different radius cannot supply a missing side. The 10th
    percentile contrast penalizes any long gap in the surrounding rim.
    Returns rim contrast, polygon size proposal, fitted side length and phase.
    The size proposal is image evidence only; chemical pentagon assignment is
    deferred until the connected fused graph is known.
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    if not len(xy):
        return np.zeros((0, 4))
    I = ndi.gaussian_filter(flat, max(0.8, 0.04 * b))
    center = ndi.map_coordinates(I, [xy[:, 1], xy[:, 0]], order=1, mode='nearest')
    th = 2 * np.pi * np.arange(48) / 48
    scales = b * np.linspace(0.7, 1.2, 11)
    out = np.column_stack((np.full(len(xy), -np.inf), np.zeros((len(xy), 3))))
    for n in (5, 6):
        phase = np.linspace(0, 2 * np.pi / n, 12, endpoint=False)
        delta = (th[None, :] - phase[:, None] + np.pi / n) % (2 * np.pi / n) - np.pi / n
        radius = APO[n] * scales[:, None, None] / np.cos(delta[None, :, :])
        yy = xy[:, 1, None, None, None] + radius[None, :, :, :] * np.sin(th)[None, None, None, :]
        xx = xy[:, 0, None, None, None] + radius[None, :, :, :] * np.cos(th)[None, None, None, :]
        contrast = ndi.map_coordinates(I, [yy, xx], order=1, mode='nearest') - center[:, None, None, None]
        q10 = np.quantile(contrast, 0.1, axis=-1)
        best = np.argmax(q10.reshape(len(xy), -1), axis=1)
        ib, ip = np.divmod(best, len(phase))
        value = q10[np.arange(len(xy)), ib, ip]
        use = value > out[:, 0]
        out[use] = np.stack((value[use], np.full(use.sum(), n), scales[ib[use]], phase[ip[use]]), axis=1)
    return out


def ring_enclosure(flat, xy, b):
    """Measure the bright spill barrier enclosing each dark ring candidate.

    Grayscale reconstruction fills dark basins to the lowest escape saddle.
    An exterior bay escapes through dark background and has a low spill level,
    even if unrelated bright segments surround part of its circular annulus.
    Returns spill level, basin depth, spill minus local-boundary median.
    """
    I = ndi.gaussian_filter(flat, max(0.8, 0.04 * b))
    rad = max(3, int(1.6 * b))
    out = np.zeros((len(xy), 3))
    for i, (x, y) in enumerate(xy):
        if not (0 <= x < I.shape[1] and 0 <= y < I.shape[0]):
            continue                                     # hypothesis outside image
        x0, x1 = max(0, int(x - rad)), min(I.shape[1], int(x + rad) + 1)
        y0, y1 = max(0, int(y - rad)), min(I.shape[0], int(y + rad) + 1)
        a = I[y0:y1, x0:x1]
        seed = np.full_like(a, a.max())
        seed[0, :] = a[0, :]; seed[-1, :] = a[-1, :]; seed[:, 0] = a[:, 0]; seed[:, -1] = a[:, -1]
        spill = reconstruction(seed, a, method='erosion')[int(y) - y0, int(x) - x0]
        border = np.r_[a[0, :], a[-1, :], a[:, 0], a[:, -1]]
        out[i] = spill, spill - I[int(y), int(x)], spill - np.median(border)
    return out


def propose_fused_ring_centers(centers, R, b):
    """Propose missing third rings from pairs already sharing a bright side.

    These locations come from fused-ring geometry, not local image extrema.
    They are hypotheses for subsequent full-rim evaluation, never accepted
    merely because a pair predicts them.
    """
    centers = np.asarray(centers, float).reshape(-1, 2)
    proposals = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            v = centers[j] - centers[i]
            d = np.linalg.norm(v)
            if not (0.95 * b < d < 2.2 * b) or fused_wall_score(R, centers[i], centers[j], b) <= 0:
                continue
            perp = np.array([-v[1], v[0]])
            for sign in (-1, 1):
                p = 0.5 * (centers[i] + centers[j]) + sign * np.sqrt(3) / 2 * perp
                if not (b < p[0] < R.shape[1] - b and b < p[1] < R.shape[0] - b):
                    continue
                if np.min(np.linalg.norm(centers - p, axis=1)) < 0.55 * b:
                    continue
                if not proposals or np.min(np.linalg.norm(np.asarray(proposals) - p, axis=1)) > 0.5 * b:
                    proposals.append(p)
    return np.asarray(proposals, float).reshape(-1, 2)


def n_score_ang(alphas, n, npsi=72, tol=0.17):
    """Can n-gon side normals (psi + k*2pi/n) match all neighbor dirs alphas?
    Returns matched_fraction. Hex normals at 60deg, pent at 72deg."""
    if not alphas:
        return 1.0
    best = 0
    for k in range(npsi):
        psi = np.pi * k / npsi                        # side normals mod pi/n enough
        na = psi + 2 * np.pi * np.arange(n) / n
        m = sum(1 for a in alphas if np.min(np.abs(np.angle(np.exp(1j * (a - na))))) < tol)
        best = max(best, m)
    return best / len(alphas)


def infer_graph(centers, fits, pairs, R, t_r, b):
    """Pure-geometry vertex inference from the ring-center dual graph.
    interior atom = centroid of 3 mutually fused centers
    shared edge   = mid +/- (s/2)*perp  (endpoint -> triple centroid if 3-ring atom)
    rim atom      = evenly spaced free corners in angular gaps of each ring
    n_i           = 6 unless only 5 satisfies the neighbor side-normal angles
    Returns verts(xy), edges, rings(vert ids per ring), ring_ids, ns."""
    adj = {i: set() for i in range(len(centers))}
    fused = set()
    for i, j, d, w in pairs:
        adj[i].add(j); adj[j].add(i); fused.add((i, j))
    # ---- n: angular consistency; hex prior; pent needs clear evidence
    ns, nsc = {}, {}
    for i in range(len(centers)):
        al = [np.arctan2(centers[j][0] - centers[i][0], centers[j][1] - centers[i][1]) for j in adj[i]]
        s5, s6 = n_score_ang(al, 5), n_score_ang(al, 6)
        f = fits.get(i); amp5 = f[5][1] if f else 0; amp6 = f[6][1] if f else 0
        r0 = f[5][3] if f else b
        n = 5 if (s5 > s6 or (s5 == s6 and amp5 > 1.5 * amp6 and r0 < 0.75 * b)) else 6
        ns[i] = n; nsc[i] = (s5, s6)
    s_pairs = {(i, j): d / (APO[ns[i]] + APO[ns[j]]) for i, j, d, w in pairs}
    b_med = np.median(list(s_pairs.values())) if s_pairs else b
    s_ring = {i: np.median([s_pairs[tuple(sorted((i, j)))] for j in adj[i]
                           if tuple(sorted((i, j))) in s_pairs])
              if adj[i] else b_med for i in range(len(centers))}
    s_ring = {i: (v if np.isfinite(v) else b_med) for i, v in s_ring.items()}
    # ---- proposals: (xy, {ring angles registered})
    props = []                                          # xy
    ring_vs = {i: [] for i in range(len(centers))}      # (angle, prop_idx)
    def add_prop(p_yx, ring_ids_affected):
        pi = len(props); props.append(np.array([p_yx[1], p_yx[0]]))
        for ri in ring_ids_affected:
            a = np.arctan2(p_yx[0] - centers[ri][0], p_yx[1] - centers[ri][1])
            ring_vs[ri].append((a, pi))
        return pi
    # triples -> interior atoms
    import itertools
    tri_map = {}
    for t in itertools.combinations(range(len(centers)), 3):
        if all(tuple(sorted(p)) in fused for p in itertools.combinations(t, 2)):
            v = (centers[t[0]] + centers[t[1]] + centers[t[2]]) / 3.0
            tri_map[t] = add_prop(v, t)
    # fused pair shared-edge endpoints
    for i, j, d, w in pairs:
        mid = 0.5 * (centers[i] + centers[j])
        u = centers[j] - centers[i]; u /= np.linalg.norm(u)
        perp = np.array([-u[1], u[0]])
        s_ij = s_pairs[(i, j)]
        for sgn in (-1, 1):
            p_yx = mid + sgn * 0.5 * s_ij * perp
            cov = None
            for t, vi in tri_map.items():
                if i in t and j in t:
                    tc = (centers[t[0]] + centers[t[1]] + centers[t[2]]) / 3.0
                    if np.linalg.norm(p_yx - tc) < 0.45 * s_ij:
                        cov = vi; break
            if cov is None:
                add_prop(p_yx, (i, j))
    # rim corners: fill angular gaps in each ring
    for i in range(len(centers)):
        n = ns[i]
        anch = sorted(ring_vs[i])
        if not anch:
            continue                                    # isolated/unanchored ring -> skip
        R_i = s_ring[i] * CIRC[n]
        m = len(anch)
        for k in range(m):
            a1 = anch[k][0]; a2 = anch[(k + 1) % m][0]
            span = (a2 - a1) % (2 * np.pi)
            need = int(round(span / (2 * np.pi / n))) - 1
            for q in range(1, need + 1):
                a = a1 + span * q / (need + 1)
                add_prop(np.array([centers[i][0] + R_i * np.sin(a),
                                   centers[i][1] + R_i * np.cos(a)]), (i,))
    if not props:
        return np.zeros((0, 2)), [], [], [], ns
    verts, assign = cluster_points(np.asarray(props), rjoin=0.35 * b_med)
    # ---- ring vertex cycles -> edges
    rings, ring_ids, edges, eset = [], [], [], set()
    ring_vids = {i: sorted(set(assign[p] for _, p in ring_vs[i])) for i in range(len(centers)) if ring_vs[i]}
    for i, vids in ring_vids.items():
        ang = {vid: np.arctan2(verts[vid][1] - centers[i][0], verts[vid][0] - centers[i][1]) for vid in vids}
        order = sorted(vids, key=lambda vid: ang[vid])
        rings.append(order); ring_ids.append(i)
        for k in range(len(order)):
            e = (min(order[k], order[(k + 1) % len(order)]), max(order[k], order[(k + 1) % len(order)]))
            if e[0] != e[1] and e not in eset:
                eset.add(e); edges.append(e)
    return verts, edges, rings, ring_ids, ns


def graph_from_fused_rings(centers, pairs, ns, b, *, assignment=None):
    """Build one atom per shared topological corner of fused ring polygons.

    The ring-side identifications are exact: no spatial vertex clustering can
    accidentally merge unrelated atoms. Raises when the proposed dual graph
    requires a carbon with degree greater than three.
    assignment: optional (phase, side) — phase[i] polygon rotation, side[(i,j)]
    the side index of ring i fused to ring j. When supplied the free
    phase/Hungarian fit below is skipped and the given layout is used verbatim.
    """
    centers = np.asarray(centers, float)
    nring = len(centers)
    if nring == 0:
        raise ValueError('No rings to build')
    offsets = np.r_[0, np.cumsum([ns[i] for i in range(nring)])]
    parent = np.arange(offsets[-1])
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, c):
        parent[root(c)] = root(a)
    adj = [[] for _ in range(nring)]
    for i, j, *_ in pairs:
        adj[i].append(j); adj[j].append(i)
    if assignment is not None:
        phase, side = assignment
    else:
        phase, side = {}, {}
        for i in range(nring):
            n = ns[i]
            if len(adj[i]) > n:
                raise ValueError(f'Ring {i} has {len(adj[i])} fused neighbors but only {n} sides')
            if not adj[i]:
                phase[i] = 0.0
                continue
            alpha = np.arctan2(centers[adj[i], 1] - centers[i, 1], centers[adj[i], 0] - centers[i, 0])
            best = (np.inf, None, None)
            for psi in np.linspace(0, 2 * np.pi / n, 181, endpoint=False):
                normals = psi + 2 * np.pi * np.arange(n) / n
                err = np.abs(np.angle(np.exp(1j * (alpha[:, None] - normals[None, :]))))
                rows, cols = linear_sum_assignment(err)
                value = np.square(err[rows, cols]).sum()
                if value < best[0]:
                    best = (value, psi, cols)
            phase[i] = best[1]
            for j, k in zip(adj[i], best[2]):
                side[i, j] = int(k)
    for i, j, *_ in pairs:
        ki, kj = side[i, j], side[j, i]
        union(offsets[i] + ki, offsets[j] + (kj + 1) % ns[j])
        union(offsets[i] + (ki + 1) % ns[i], offsets[j] + kj)
    proposals = []
    for i in range(nring):
        theta = phase[i] - np.pi / ns[i] + 2 * np.pi * np.arange(ns[i]) / ns[i]
        radius = b * CIRC[ns[i]]
        proposals.extend(centers[i] + radius * np.stack((np.cos(theta), np.sin(theta)), axis=1))
    roots = np.array([root(i) for i in range(offsets[-1])])
    _, ids = np.unique(roots, return_inverse=True)
    verts = np.zeros((ids.max() + 1, 2))
    np.add.at(verts, ids, proposals)
    verts /= np.bincount(ids)[:, None]
    rings = [ids[offsets[i]:offsets[i + 1]].tolist() for i in range(nring)]
    if any(len(set(ring)) != len(ring) for ring in rings):
        raise ValueError('Fused-side assignments collapse a ring corner')
    edges = sorted({tuple(sorted((ring[k], ring[(k + 1) % len(ring)]))) for ring in rings for k in range(len(ring))})
    degree = np.bincount(np.asarray(edges).ravel(), minlength=len(verts))
    if degree.max() > 3:
        raise ValueError(f'Fused-ring dual creates degree-{degree.max()} carbon')
    return verts, edges, rings


def _ring_layout(centers, i, nb, pset):
    """Best (n, phase, ksteps, cost) for ring i given its angle-sorted fused
    neighbors nb. Each cyclic gap takes an integer side-step: exactly 1 for a
    triangular gap (the two neighbors share a fused pair AND the gap < pi),
    >= 2 otherwise — prevents open V-junction corners and keeps
    triple-junction corners coherent. Steps must sum to n in {5,6}; cost =
    mean squared wrapped residual + 0.02*(6-n) (hexagon wins ties). None when
    neither n admits a legal assignment."""
    m = len(nb)
    alpha = np.arctan2(centers[np.asarray(nb), 1] - centers[i, 1], centers[np.asarray(nb), 0] - centers[i, 0])
    gap = np.roll(alpha, -1) - alpha; gap[-1] += 2 * np.pi   # cyclic gaps, sum 2pi
    tri = np.array([gap[g] < np.pi and tuple(sorted((nb[g], nb[(g + 1) % m]))) in pset for g in range(m)])
    best = None
    for n in (5, 6):
        if m > n:
            continue
        lo = np.where(tri, 1, 2)
        if lo.sum() > n:
            continue
        hi = np.where(tri, 1, n - lo.sum() + lo)           # triangular gap: step exactly 1
        for steps in itertools.product(*[range(lo[g], hi[g] + 1) for g in range(m)]):
            if sum(steps) != n:
                continue
            k = np.cumsum([0] + list(steps[:-1])) % n
            ph = np.angle(np.exp(1j * (alpha - 2 * np.pi * k / n)).mean())
            resid = np.abs(np.angle(np.exp(1j * (alpha - (ph + 2 * np.pi * k / n)))))
            c = float((resid**2).mean()) + 0.02 * (6 - n)
            if best is None or (c, 6 - n) < best[:2]:        # tie -> hexagon
                best = (c, 6 - n, ph, k, n)
    if best is None:
        return None
    c, _, ph, k, n = best
    return n, ph, k, c


def _constrained_layout_info(centers, pset, nring):
    """_constrained_layout that also reports the blocking ring.
    Returns ((ns, phase, side, cost), -1) or (None, ring_index)."""
    adj = [[] for _ in range(nring)]
    for i, j in pset:
        adj[i].append(j); adj[j].append(i)
    ns, phase, side, cost = {}, {}, {}, {}
    for i in range(nring):
        if not adj[i]:
            if nring == 1:
                ns[i], phase[i], cost[i] = 6, 0.0, 0.0    # isolated ring: hexagon, phase 0
                continue
            return None, i                                # isolated ring in multi-ring input: invalid
        nb = sorted(adj[i], key=lambda j: np.arctan2(centers[j, 1] - centers[i, 1], centers[j, 0] - centers[i, 0]))
        r = _ring_layout(centers, i, nb, pset)
        if r is None:
            return None, i
        n, ph, k, c = r
        ns[i], phase[i], cost[i] = n, ph, c
        for jj, j in enumerate(nb):
            side[i, j] = int(k[jj])
    return (ns, phase, side, cost), -1


def _constrained_layout(centers, pset, nring):
    """Per-ring constrained side assignment for fused pair set pset {(i,j)}.

    Neighbors are cyclically ordered by angle; each cyclic gap gets an integer
    side-step: exactly 1 when the gap is a triangular face (the two neighbors
    share a fused pair AND the angular gap < pi), >= 2 otherwise — this
    prevents open V-junction corners and keeps triple-junction corners
    coherent. Steps sum to n for n in {5,6}. Side index of the first neighbor
    is 0; subsequent indices accumulate the preceding gap steps mod n. Phase
    is the circular mean of alpha - 2pi*k/n; cost = mean squared wrapped
    residual + 0.02*(6-n), hexagon wins ties. Returns (ns, phase, side, cost)
    or None when a ring fits neither n."""
    lay, _ = _constrained_layout_info(centers, pset, nring)
    return lay


def _dual_connected(pset, nring):
    if nring == 1:
        return True
    adj = [[] for _ in range(nring)]
    for i, j in pset:
        adj[i].append(j); adj[j].append(i)
    seen, stack = {0}, [0]
    while stack:
        for j in adj[stack.pop()]:
            if j not in seen:
                seen.add(j); stack.append(j)
    return len(seen) == nring


def _topo_ok(verts, edges, rings, nring):
    """Static chemical invariants: connected carbon graph, Euler V-E+F==1,
    every edge in at most 2 rings."""
    if not edges:
        return False
    adj = [[] for _ in range(len(verts))]
    for a, c in edges:
        adj[a].append(c); adj[c].append(a)
    seen, stack = {0}, [0]
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if j not in seen:
                seen.add(j); stack.append(j)
    if len(seen) != len(verts):
        return False
    if len(verts) - len(edges) + nring != 1:
        return False
    inc = {}
    for r in rings:
        for k in range(len(r)):
            e = tuple(sorted((r[k], r[(k + 1) % len(r)])))
            inc[e] = inc.get(e, 0) + 1
            if inc[e] > 2:
                return False                            # an edge shared by >2 rings
    return True


def _seg_disjoint_bad(p, q, r, s):
    """Proper crossing or collinear interior overlap of segments pq and rs.
    Used only for vertex-disjoint edges (shared endpoints never reach here)."""
    d1 = np.cross(q - p, r - p); d2 = np.cross(q - p, s - p)
    d3 = np.cross(s - r, p - r); d4 = np.cross(s - r, q - r)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True                                           # proper crossing
    if abs(d1) < 1e-9 and abs(d2) < 1e-9 and abs(d3) < 1e-9 and abs(d4) < 1e-9:
        t = np.dot(np.stack([r - p, s - p]), q - p) / np.dot(q - p, q - p)
        lo, hi = min(t), max(t)
        if lo < 1.0 - 1e-9 and hi > 1e-9:                     # interior overlap on the pq line
            return True
    return False


def _geom_badness(verts, edges, rings):
    """Count of geometric violations: inverted/degenerate rings + crossing or
    overlapping disjoint edge pairs (incl. ring self-intersections). Also
    returns the ring indices involved, for targeted repair."""
    vs = np.asarray(verts, float)
    nbad = 0
    bad_rings = set()
    v2r = {}
    for ri, ring in enumerate(rings):
        for vi in ring:
            v2r.setdefault(vi, []).append(ri)
        p = vs[ring]
        if np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) <= 0:
            nbad += 1; bad_rings.add(ri)
    for (a, b_), (c, d) in itertools.combinations(edges, 2):
        if len({a, b_, c, d}) < 4:
            continue
        if _seg_disjoint_bad(vs[a], vs[b_], vs[c], vs[d]):
            nbad += 1
            bad_rings |= set(v2r.get(a, ())) | set(v2r.get(b_, ())) | set(v2r.get(c, ())) | set(v2r.get(d, ()))
    for ri, ring in enumerate(rings):
        n = len(ring)
        segs = [(ring[k], ring[(k + 1) % n]) for k in range(n)]
        for k1 in range(n):
            for k2 in range(k1 + 2, n):
                if k1 == 0 and k2 == n - 1:
                    continue                                  # adjacent (cyclic) edges share a vertex
                if _seg_disjoint_bad(vs[segs[k1][0]], vs[segs[k1][1]], vs[segs[k2][0]], vs[segs[k2][1]]):
                    nbad += 1; bad_rings.add(ri)
    return nbad, bad_rings


def _geom_bad(verts, edges, rings):
    """Disjoint-edge crossings/overlaps, ring self-intersection, and
    nonpositive (inverted) ring orientation."""
    return _geom_badness(verts, edges, rings)[0] > 0


def _refine_verts(verts, edges, rings, centers, b, lo=0.95, hi=1.05):
    """SLSQP refine of shared vertex positions (normalized by b).

    Objective: sum ||v-v0||^2 + 4*sum ||mean(v[ring])-center||^2. Hard
    constraints for every distinct vertex pair |dv|^2 >= lo^2 and for bonded
    pairs |dv|^2 <= hi^2. Returns refined verts (pixel units) or None."""
    from scipy.optimize import minimize
    v0 = np.asarray(verts, float) / b
    ctr = np.asarray(centers, float) / b
    rid = [i for i, r in enumerate(rings) for _ in r]
    flat_idx = [v for r in rings for v in r]
    ea = np.asarray(edges, int)
    ii, jj = np.triu_indices(len(v0), 1)

    def obj(v):
        v = v.reshape(-1, 2)
        acc = np.zeros((len(rings), 2)); cnt = np.zeros(len(rings))
        np.add.at(acc, rid, v[flat_idx]); np.add.at(cnt, rid, 1)
        return float(((v - v0)**2).sum() + 4.0 * ((acc / cnt[:, None] - ctr)**2).sum())

    def cons(v):
        v = v.reshape(-1, 2)
        d2 = ((v[ii] - v[jj])**2).sum(1)
        de = ((v[ea[:, 0]] - v[ea[:, 1]])**2).sum(1)
        return np.concatenate([d2 - lo * lo, hi * hi - de])

    res = minimize(obj, v0.ravel(), method='SLSQP', constraints=[{'type': 'ineq', 'fun': cons}],
                   options=dict(maxiter=200, ftol=1e-8))
    if not res.success or cons(res.x).min() < -1e-6:
        return None
    return res.x.reshape(-1, 2) * b


def constrained_ring_graph(centers, R, b):
    """Constrained fused-ring reconstruction: candidate duals = all Delaunay
    edges in the fused window (0.85b<d<2.2b, wall score kept in the pair
    record for diagnostics only) plus each single-edge deletion that keeps
    the dual connected. Per-ring side assignment is constrained by cyclic
    neighbor gaps (triangular gap -> step 1, else >=2). Candidates are
    ranked by summed per-ring layout cost; the first <=8 ranked candidates
    are tried for SLSQP geometric feasibility (all distinct pairs >=0.95b,
    bonded <=1.05b, no edge crossings, valid invariants). Raises ValueError
    when no physically admissible layout exists — never drops a ring.
    Returns (verts, edges, rings, ns, pairs)."""
    centers = np.asarray(centers, float)
    nring = len(centers)
    if nring == 0:
        raise ValueError('constrained_ring_graph: no ring centers')
    if nring < 3:
        cand = [(i, j) for i, j in itertools.combinations(range(nring), 2)
                if 0.85 * b < np.linalg.norm(centers[i] - centers[j]) < 2.2 * b]
    else:
        cand = sorted({tuple(sorted((int(a), int(c)))) for t in Delaunay(centers).simplices for a, c in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))})
        cand = [(i, j) for i, j in cand if 0.85 * b < np.linalg.norm(centers[i] - centers[j]) < 2.2 * b]
    rec = {(i, j): (i, j, float(np.linalg.norm(centers[i] - centers[j])), float(fused_wall_score(R, centers[i], centers[j], b))) if nring > 1 else None for i, j in cand}
    pset0 = set(cand)

    variants = [(None, pset0)]
    for e in sorted(pset0):
        trial = pset0 - {e}
        if _dual_connected(trial, nring):
            variants.append((e, trial))
    scored = []
    for deleted, pset in variants:
        lay = _constrained_layout(centers, pset, nring)
        if lay is None:
            continue
        ns, phase, side, cost = lay
        try:
            verts, edges, rings = graph_from_fused_rings(centers, [rec[e] for e in sorted(pset)], ns, b, assignment=(phase, side))
        except ValueError:
            continue
        if not _topo_ok(verts, edges, rings, nring):
            continue
        scored.append((sum(cost.values()), deleted if deleted is not None else (-1, -1),
                       verts, edges, rings, ns, pset))
    scored.sort(key=lambda s: (s[0], s[1]))
    for c_, deleted, verts, edges, rings, ns, pset in scored[:8]:
        vr = _refine_verts(verts, edges, rings, centers, b)
        if vr is None or _geom_bad(vr, edges, rings) or not _topo_ok(vr, edges, rings, nring):
            continue
        return vr, edges, rings, ns, [rec[e] for e in sorted(pset)]
    raise ValueError(f'constrained_ring_graph: no physically admissible layout for {nring} centers ({len(scored)} ranked candidates)')


# Fused-ring center-center distances in units of b (side length): the two
# apothems of the joined n-gons. 5-5 = 1.376b, 5-6 = 1.554b, 6-6 = 1.732b.
FUSED_D = np.array([2.0 * APO[5], APO[5] + APO[6], 2.0 * APO[6]])


def _ideal_fused_centers(centers, pairs, ns, phase, side, b, anchor=0.3):
    """Least-squares ideal ring centers for a fused-polygon embedding.

    Each fused pair (i,j) pins its two shared-side corners: corner ka of ring
    i coincides with corner kb of ring j -> linear equations c_i - c_j = rhs
    in the unknown centers (regular polygons of circumradius CIRC[n]*b with
    the fitted phases). Weak anchors (weight `anchor`) to the observed
    centers fix the otherwise-free global translation/rotation. Use when the
    observed positions are tip-distorted and untrustworthy."""
    centers = np.asarray(centers, float)
    nring = len(centers)
    rho = np.array([CIRC[ns[i]] for i in range(nring)]) * b
    def u(i, k):
        th = phase[i] - np.pi / ns[i] + 2 * np.pi * k / ns[i]
        return np.array([np.cos(th), np.sin(th)])
    A, y = [], []
    for i, j, *_ in pairs:
        ki, kj = side[i, j], side[j, i]
        for ka, kb in ((ki, (kj + 1) % ns[j]), ((ki + 1) % ns[i], kj)):
            rhs = rho[j] * u(j, kb) - rho[i] * u(i, ka)
            for d in (0, 1):
                row = np.zeros(2 * nring); row[2 * i + d] = 1.0; row[2 * j + d] = -1.0
                A.append(row); y.append(rhs[d])
    if anchor > 0:
        w = np.sqrt(anchor)
        for i in range(nring):
            for d in (0, 1):
                row = np.zeros(2 * nring); row[2 * i + d] = w
                A.append(row); y.append(w * centers[i, d])
    c = np.linalg.lstsq(np.asarray(A), np.asarray(y), rcond=None)[0]
    return c.reshape(-1, 2)


def _layout_fails(centers, pset, nring):
    """Ring indices with no legal 5/6 side-step assignment under fused pair
    set pset (isolated rings count as failures for nring > 1)."""
    adj = [[] for _ in range(nring)]
    for i, j in pset:
        adj[i].append(j); adj[j].append(i)
    fails = []
    for i in range(nring):
        if not adj[i]:
            if nring > 1:
                fails.append(i)
            continue
        nb = sorted(adj[i], key=lambda j: np.arctan2(centers[j, 1] - centers[i, 1], centers[j, 0] - centers[i, 0]))
        if _ring_layout(centers, i, nb, pset) is None:
            fails.append(i)
    return fails


def constrained_ring_graph_distorted(centers, R, b, dwin=(0.60, 2.60), anchor=0.3, max_iter=250, max_kick=20):
    """Distortion-tolerant fused-ring reconstruction for AFM images where tip
    deflection (electrostatics, vdW) shifts apparent ring positions off their
    true sites — the dots fix WHICH rings exist, not exact positions.

    Differences vs constrained_ring_graph:
      * candidates = ALL center pairs in dwin*b (not just Delaunay, wide
        window), each scored by misfit to the fused distances in FUSED_D;
      * repair = hill-climb over single-edge toggles (delete or re-add) on the
        candidate dual, scored by (#rings with no legal side assignment,
        topology/build failure, total distance misfit); connectivity is never
        broken and no ring is ever dropped; local minima escape via forced
        deletion of the worst-misfit edge at a failing ring ('kick');
      * final geometry = IDEAL fused-polygon embedding (_ideal_fused_centers),
        not the distorted dot positions.

    Returns (verts, edges, rings, ns, pairs, toggled) — same convention as
    constrained_ring_graph plus the list of toggled candidate pairs. Raises
    ValueError when no admissible dual is found."""
    centers = np.asarray(centers, float)
    nring = len(centers)
    if nring == 0:
        raise ValueError('constrained_ring_graph_distorted: no ring centers')
    if nring < 3:
        v, e, rings, ns, pairs = constrained_ring_graph(centers, R, b)
        return v, e, rings, ns, pairs, []
    dm = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1) / b
    cand = [(i, j) for i, j in itertools.combinations(range(nring), 2)
            if dwin[0] < dm[i, j] < dwin[1]]
    rec = {e: (e[0], e[1], float(np.linalg.norm(centers[e[0]] - centers[e[1]])),
               float(fused_wall_score(R, centers[e[0]], centers[e[1]], b))) for e in cand}
    ecost = {e: float(np.abs(dm[e] - FUSED_D).min()) for e in cand}
    pset = set(cand)
    visited, kicks = set(), 0

    def state_score(ps):
        """(n_layout_fails, topo/build fail, n_geom_crossings, total misfit).
        Geometry is scored on the IDEAL embedding — distorted positions only
        steer topology, never decide it."""
        nf = len(_layout_fails(centers, ps, nring))
        if nf:
            return (nf, 1, 999, 0.0), None
        lay, _ = _constrained_layout_info(centers, ps, nring)
        ns, phase, side, cost = lay
        pairs = [rec[e] for e in sorted(ps)]
        try:
            v, ed, rg = graph_from_fused_rings(centers, pairs, ns, b, assignment=(phase, side))
            ok = _topo_ok(v, ed, rg, nring)
        except ValueError:
            ok = False
        if not ok:
            return (0, 1, 999, sum(ecost[e] for e in ps)), None
        cI = _ideal_fused_centers(centers, pairs, ns, phase, side, b, anchor)
        try:
            v, ed, rg = graph_from_fused_rings(cI, pairs, ns, b, assignment=(phase, side))
            nbad, bad_rings = _geom_badness(v, ed, rg)
        except ValueError:
            nbad, bad_rings = 999, set()
        return (0, 0, nbad, sum(ecost[e] for e in ps)), (lay, pairs, bad_rings)

    for _ in range(max_iter):
        sc, res = state_score(pset)
        if sc[:3] == (0, 0, 0):
            lay, pairs, _ = res
            ns, phase, side, cost = lay
            cI = _ideal_fused_centers(centers, pairs, ns, phase, side, b, anchor)
            verts, edges, rings = graph_from_fused_rings(cI, pairs, ns, b, assignment=(phase, side))
            return verts, edges, rings, ns, pairs, sorted(set(cand) ^ pset)
        visited.add(frozenset(pset))
        best_e, best_sc = None, sc
        for e in cand:
            t = pset ^ {e}
            if frozenset(t) in visited:
                continue
            if e in pset and not _dual_connected(t, nring):
                continue                                  # never disconnect the dual / isolate a ring
            sc2, _ = state_score(t)
            if sc2 < best_sc:
                best_sc, best_e = sc2, e
        if best_e is None:                                # local minimum -> forced 'kick'
            fails = _layout_fails(centers, pset, nring)
            if fails:
                pool = [e for e in pset if any(f in e for f in fails)]
            elif res is not None and res[2]:              # geom-bad terminal: toggle edges at the crossing rings
                pool = [e for e in pset if any(f in e for f in res[2])]
            else:
                pool = list(pset)
            for e in sorted(pool, key=lambda e: -ecost[e]):
                t = pset - {e}
                if frozenset(t) not in visited and _dual_connected(t, nring):
                    best_e = e
                    break
            if best_e is None or (kicks := kicks + 1) > max_kick:
                break
        pset ^= {best_e}
    raise ValueError(f'constrained_ring_graph_distorted: no admissible layout for {nring} centers ({len(visited)} states tried)')


# ---------------------------------------------------------------------------
# Staggered-grid molecular channels ("pixelization") for image->graph nets.
# Atom channel: <=1 atom per ~1 A cell (atoms never closer than ~1 A), storing
# occupancy + exact sub-cell offset + element. Bond channels: staggered ports —
# each atom cell stores up to 3 links to the CELLS it is bonded to (a 1.4 A bond
# on a 1 A grid can reach cells 2 apart, so links encode cell offset, not just
# the 4 face-neighbor edges). Ring channel: same occ+offset as atoms.
# ---------------------------------------------------------------------------

BOND_DIRS = [(i, j) for j in range(-2, 3) for i in range(-2, 3)
             if (i, j) != (0, 0) and (i < 0 or (i == 0 and j < 0))]   # canonical half of neighbor offsets


def voxelize_mol(apos, bonds, cell=1.0, enames=None, ring_centers=None, origin=None, max_links=3):
    """Molecule -> staggered-grid channels. See header above.

    Returns dict of arrays on grid shape (H,W):
      atom_occ (H,W) 0/1, atom_pos (H,W,2) absolute xy of the atom in that cell,
      atom_elem (H,W) atomic number if enames given, bond_port (H,W,max_links,2)
      cell offsets (di,dj) of each bonded neighbor (BOTH endpoints store the link),
      ring_occ/ring_pos same scheme for ring centers, plus 'origin','cell'."""
    xy = np.asarray(apos, float)[:, :2]
    origin = (xy.min(axis=0) - cell) if origin is None else np.asarray(origin, float)
    qi = np.floor((xy - origin) / cell).astype(int)
    assert len(np.unique(qi, axis=0)) == len(qi), f'voxelize_mol: two atoms share a cell (cell={cell} too big)'
    W, H = qi.max(axis=0) + 2, qi[:, 1].max() + 2
    W = qi[:, 0].max() + 2
    atom_occ = np.zeros((H, W)); atom_pos = np.zeros((H, W, 2))
    atom_elem = np.zeros((H, W))
    ports = np.zeros((H, W, max_links, 2), dtype=np.int8)
    zof = {}
    if enames is not None:
        from spammm.elements import ELEMENTS
        zof = {e[1]: e[0] for e in ELEMENTS}
    for k, (ix, iy) in enumerate(qi):
        atom_occ[iy, ix] = 1; atom_pos[iy, ix] = xy[k]
        if enames is not None:
            atom_elem[iy, ix] = zof[str(enames[k]).split('_')[0]]
    for i, j in bonds:
        d = qi[j] - qi[i]
        assert abs(d[0]) <= 2 and abs(d[1]) <= 2, f'voxelize_mol: bond {i}-{j} spans >2 cells — increase cell'
        ci, cj = qi[i], qi[j]
        for (ix, iy), dd in ((ci, d), (cj, -d)):
            free = np.flatnonzero((ports[iy, ix] == 0).all(axis=1))
            assert len(free), f'voxelize_mol: atom cell has >{max_links} bonds'
            ports[iy, ix, free[0]] = dd
    out = dict(atom_occ=atom_occ, atom_pos=atom_pos, atom_elem=atom_elem, bond_port=ports, origin=origin, cell=cell)
    if ring_centers is not None:
        rc = np.asarray(ring_centers, float).reshape(-1, 2)
        rq = np.floor((rc - origin) / cell).astype(int)
        ring_occ = np.zeros((H, W)); ring_pos = np.zeros((H, W, 2))
        for p, (ix, iy) in zip(rc, rq):
            if 0 <= iy < H and 0 <= ix < W:
                assert ring_occ[iy, ix] == 0, 'voxelize_mol: two ring centers share a cell'
                ring_occ[iy, ix] = 1; ring_pos[iy, ix] = p
        out.update(ring_occ=ring_occ, ring_pos=ring_pos)
    return out


def unvoxelize_mol(g):
    """Staggered-grid channels -> (apos, bonds, ring_centers). Round-trip check."""
    iy, ix = np.nonzero(g['atom_occ'])
    apos = g['atom_pos'][iy, ix]
    idx = {(x, y): k for k, (x, y) in enumerate(zip(ix, iy))}
    bonds = set()
    for x, y in zip(ix, iy):
        for d in g['bond_port'][y, x]:
            if not d.any():
                continue
            jx, jy = x + d[0], y + d[1]
            assert (jx, jy) in idx, 'unvoxelize_mol: port points at empty cell'
            bonds.add(tuple(sorted((idx[(x, y)], idx[(jx, jy)]))))
    rc = g['ring_pos'][np.nonzero(g['ring_occ'])] if 'ring_occ' in g else np.zeros((0, 2))
    return apos, sorted(bonds), rc


def voxelize_mol_smooth(apos, bonds, cell=0.5, enames=None, ring_centers=None, origin=None, shape=None):
    """Smooth (bilinear-splatted) molecular channels — translation-equivariant
    version of voxelize_mol. Each feature's weight is distributed over its 4
    surrounding cell centers; each cell stores the weighted residual offset to
    the feature's true position (not absolute coordinates).

    Returns dict on grid (H,W):
      atom_w, atom_off(H,W,2)  — atom weight splat + residual to atom pos
      atom_elem (H,W)          — weighted mean Z (integer where single atom)
      bond_w, bond_off(H,W,2)  — bond-MIDPOINT splat + residual to midpoint
      bond_vec (H,W,2)         — weighted mean HALF-vector (pB-pA)/2; endpoints = mid +- vec
      ring_w, ring_off         — ring-center splat + residual
    With cell <= 0.5 the splat footprints of features >= ~1 A apart never share a
    cell, so every occupied cell encodes exactly one feature (no mixing)."""
    xy = np.asarray(apos, float)[:, :2]
    origin = (xy.min(axis=0) - 2 * cell) if origin is None else np.asarray(origin, float)
    if shape is None:
        span = np.ceil((xy.max(axis=0) - origin) / cell) + 3
        W, H = int(span[0]), int(span[1])
    else:
        H, W = shape

    def splat(pts):
        """pts (M,2) -> weight (H,W), residual (H,W,2).
        Cell (i,j) covers [origin+i*cell, origin+(i+1)*cell) — its CENTER is
        origin+(i+0.5)*cell; weights/offsets are relative to centers."""
        w = np.zeros((H, W)); res = np.zeros((H, W, 2))
        u = (pts - origin) / cell - 0.5                # position in cell-center coords
        i0, j0 = np.floor(u[:, 0]).astype(int), np.floor(u[:, 1]).astype(int)
        fx, fy = u[:, 0], u[:, 1]
        for di in (0, 1):
            for dj in (0, 1):
                i, j = i0 + di, j0 + dj
                ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
                wt = (1 - np.abs(fx - i)) * (1 - np.abs(fy - j))
                ii, jj, ww = i[ok], j[ok], wt[ok]
                cen = origin + cell * (np.stack([ii, jj], axis=1) + 0.5)
                np.add.at(w, (jj, ii), ww)
                np.add.at(res[..., 0], (jj, ii), ww * (pts[ok, 0] - cen[:, 0]))
                np.add.at(res[..., 1], (jj, ii), ww * (pts[ok, 1] - cen[:, 1]))
        m = w > 0
        res[m] /= w[m, None]
        return w, res

    def splat_val(pts, val):
        """Bilinear splat of per-point scalar val -> (H,W) weighted sum."""
        wv = np.zeros((H, W))
        u = (pts - origin) / cell - 0.5
        i0, j0 = np.floor(u[:, 0]).astype(int), np.floor(u[:, 1]).astype(int)
        for di in (0, 1):
            for dj in (0, 1):
                i, j = i0 + di, j0 + dj
                ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
                wt = (1 - np.abs(u[ok, 0] - i[ok])) * (1 - np.abs(u[ok, 1] - j[ok]))
                np.add.at(wv, (j[ok], i[ok]), wt * val[ok])
        return wv

    atom_w, atom_off = splat(xy)
    out = dict(atom_w=atom_w, atom_off=atom_off, origin=origin, cell=cell)
    if enames is not None:
        from spammm.elements import ELEMENTS
        zof = {e[1]: e[0] for e in ELEMENTS}
        Zs = np.array([zof[str(e).split('_')[0]] for e in enames], float)
        zw = splat_val(xy, Zs)
        m = atom_w > 0
        out['atom_elem'] = np.where(m, zw / np.maximum(atom_w, 1e-9), 0)
    if bonds is not None and len(bonds):
        bnd = np.asarray(bonds)
        mid = 0.5 * (xy[bnd[:, 0]] + xy[bnd[:, 1]])
        hv = 0.5 * (xy[bnd[:, 1]] - xy[bnd[:, 0]])
        bond_w, bond_off = splat(mid)
        # splat the half-vectors with the same bilinear weights, normalized by w
        bvec = np.zeros((H, W, 2))
        bvec[..., 0] = splat_val(mid, hv[:, 0]); bvec[..., 1] = splat_val(mid, hv[:, 1])
        m = bond_w > 0
        bvec[m] /= bond_w[m, None]
        out.update(bond_w=bond_w, bond_off=bond_off, bond_vec=bvec)
    if ring_centers is not None:
        ring_w, ring_off = splat(np.asarray(ring_centers, float).reshape(-1, 2))
        out.update(ring_w=ring_w, ring_off=ring_off)
    return out


def unvoxelize_mol_smooth(g, thr=0.15, merge=0.4):
    """Decode smooth channels back to (apos, bonds, ring_centers).
    Every cell with weight > thr proposes feature pos = cell_center + offset;
    proposals of the same feature land < merge*cell apart and are weight-averaged."""
    cell, origin = g['cell'], g['origin']

    def collect(w, off):
        pts, wts = [], []
        for j, i in zip(*np.nonzero(w > thr)):
            pts.append(origin + cell * (np.array([i, j]) + 0.5) + off[j, i]); wts.append(w[j, i])
        if not pts:
            return np.zeros((0, 2)), np.zeros(0)
        pts, wts = np.asarray(pts), np.asarray(wts)
        order = np.argsort(-wts); taken = np.zeros(len(pts), bool)
        outp, outw = [], []
        for k in order:
            if taken[k]:
                continue
            grp = np.linalg.norm(pts - pts[k], axis=1) < merge * cell * 2 + 0.01 * cell
            grp &= ~taken
            taken |= grp
            outp.append((pts[grp] * wts[grp, None]).sum(0) / wts[grp].sum()); outw.append(wts[grp].sum())
        return np.asarray(outp), np.asarray(outw)

    apos, _ = collect(g['atom_w'], g['atom_off'])
    iy, ix = np.nonzero(g['bond_w'] > thr)
    mids, vecs, bwts = [], [], []
    for j, i in zip(iy, ix):
        mids.append(origin + cell * (np.array([i, j]) + 0.5) + g['bond_off'][j, i])
        vecs.append(g['bond_vec'][j, i]); bwts.append(g['bond_w'][j, i])
    bonds = []
    if mids:
        mids, vecs, bwts = np.asarray(mids), np.asarray(vecs), np.asarray(bwts)
        order = np.argsort(-bwts); taken = np.zeros(len(mids), bool)
        for k in order:
            if taken[k]:
                continue
            grp = (np.linalg.norm(mids - mids[k], axis=1) < merge * cell * 2 + 0.01 * cell) & ~taken
            taken |= grp
            mid = (mids[grp] * bwts[grp, None]).sum(0) / bwts[grp].sum()
            vec = (vecs[grp] * bwts[grp, None]).sum(0) / bwts[grp].sum()
            bonds.append((mid - vec, mid + vec))               # endpoint pair
    rc, _ = collect(g['ring_w'], g['ring_off']) if 'ring_w' in g else (np.zeros((0, 2)), None)
    return apos, bonds, rc


def voxelize_mol_hard(apos, bonds, cell=0.5, enames=None, ring_centers=None, origin=None, shape=None):
    """HARD single-cell occupancy — alternative to voxelize_mol_smooth.
    Each feature occupies exactly the ONE cell containing it (floor indexing):
    w=1 there, 0 elsewhere (no bilinear smear). The cell still stores the
    residual offset center->feature, so precision is retained. Sparse by
    construction: activation maps are Dirac combs, never continuous.
    Same dict layout as voxelize_mol_smooth (w, off, bond_vec in UNITS of
    the caller's coordinate system; offsets in [-0.5, 0.5]*cell)."""
    xy = np.asarray(apos, float)[:, :2]
    origin = (xy.min(axis=0) - 2 * cell) if origin is None else np.asarray(origin, float)
    if shape is None:
        span = np.ceil((xy.max(axis=0) - origin) / cell) + 3
        W, H = int(span[0]), int(span[1])
    else:
        H, W = shape

    def assign(pts, tag='feature'):
        """pts (M,2) -> w (H,W) 0/1, res (H,W,2) offset center->point.
        Containing cell = floor((pts-origin)/cell), half-open bounds; two
        features in one cell are unrepresentable -> raise, never merge."""
        w = np.zeros((H, W)); res = np.zeros((H, W, 2))
        u = np.floor((pts - origin) / cell).astype(int)          # containing cell
        owner = {}
        for f, ((i, j), p) in enumerate(zip(u, pts)):
            if 0 <= i < W and 0 <= j < H:
                if w[j, i]:
                    raise ValueError(f'voxelize_mol_hard: {tag} cell collision at ({i},{j}) between features {owner[i, j]} and {f}')
                owner[i, j] = f
                w[j, i] = 1
                res[j, i] = p - (origin + cell * (np.array([i, j]) + 0.5))
        return w, res

    atom_w, atom_off = assign(xy, 'atom')
    out = dict(atom_w=atom_w, atom_off=atom_off, origin=origin, cell=cell)
    if enames is not None:
        from spammm.elements import ELEMENTS
        zof = {e[1]: e[0] for e in ELEMENTS}
        ae = np.zeros((H, W))
        u = np.floor((xy - origin) / cell).astype(int)
        Zs = np.array([zof[str(e).split('_')[0]] for e in enames], float)
        for (i, j), z in zip(u, Zs):
            if 0 <= i < W and 0 <= j < H: ae[j, i] = z
        out['atom_elem'] = ae
    if bonds is not None and len(bonds):
        bnd = np.asarray(bonds)
        mid = 0.5 * (xy[bnd[:, 0]] + xy[bnd[:, 1]])
        hv = 0.5 * (xy[bnd[:, 1]] - xy[bnd[:, 0]])
        bond_w, bond_off = assign(mid, 'bond')
        bvec = np.zeros((H, W, 2))
        u = np.floor((mid - origin) / cell).astype(int)
        for (i, j), v in zip(u, hv):
            if 0 <= i < W and 0 <= j < H: bvec[j, i] = v
        out.update(bond_w=bond_w, bond_off=bond_off, bond_vec=bvec)
    if ring_centers is not None:
        ring_w, ring_off = assign(np.asarray(ring_centers, float).reshape(-1, 2), 'ring')
        out.update(ring_w=ring_w, ring_off=ring_off)
    return out


def unvoxelize_mol_hard(g, thr=0.5):
    """Decode hard-occupancy channels -> (apos, bonds, ring_centers).
    Prediction maps may still be soft/delocalized, so apply non-max
    suppression: a cell counts only if w>thr AND it is the 3x3 local max.
    Then pos = cell_center + offset (no weight averaging — one cell/feature)."""
    cell, origin = g['cell'], g['origin']
    from scipy.ndimage import maximum_filter

    def collect(w, off):
        mx = maximum_filter(w, size=3, mode='constant')
        jy, ix = np.nonzero((w > thr) & (w >= mx))
        if len(jy) == 0:
            return np.zeros((0, 2))
        pts = origin + cell * (np.stack([ix, jy], 1) + 0.5) + off[jy, ix]
        return pts

    apos = collect(g['atom_w'], g['atom_off'])
    bonds = []
    if 'bond_w' in g:
        mx = maximum_filter(g['bond_w'], size=3, mode='constant')
        jy, ix = np.nonzero((g['bond_w'] > thr) & (g['bond_w'] >= mx))
        for j, i in zip(jy, ix):
            mid = origin + cell * (np.array([i, j]) + 0.5) + g['bond_off'][j, i]
            bonds.append((mid - g['bond_vec'][j, i], mid + g['bond_vec'][j, i]))
    rc = collect(g['ring_w'], g['ring_off']) if 'ring_w' in g else np.zeros((0, 2))
    return apos, bonds, rc


def ring_faces(apos, bonds, enames=None, max_ring=8):
    """Planar faces of a fused graph: shortest cycle through each bond with that
    bond removed (BFS), deduplicated. Returns list of atom-index lists."""
    nbr = {}
    heavy = range(len(apos)) if enames is None else [i for i, e in enumerate(enames) if str(e).split('_')[0] != 'H']
    hset = set(heavy)
    for i, j in bonds:
        if i in hset and j in hset:
            nbr.setdefault(i, []).append(j); nbr.setdefault(j, []).append(i)
    faces, seen = [], set()
    from collections import deque
    for a, c in bonds:
        if a not in hset or c not in hset:
            continue
        dist = {a: 0}; par = {a: None}; q = deque([a])
        while q:
            u = q.popleft()
            for v in nbr.get(u, []):
                if (u, v) == (a, c) or (v, u) == (a, c):
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1; par[v] = u; q.append(v)
        if c not in dist or dist[c] > max_ring - 1:
            continue
        path = [c]
        while path[-1] != a:
            path.append(par[path[-1]])
        key = tuple(sorted(path))
        if key not in seen:
            seen.add(key); faces.append(path)
    return faces


def ring_dual_proposals(centers, R, b, dwin=(0.85, 2.2)):
    """Fused-side candidates (pairs with distance + wall evidence) and
    per-ring n in {5,6} from neighbor-angle consistency. Used by
    graph_from_ring_centers and by repair loops needing _deg_violation_rings."""
    centers = np.asarray(centers, float)
    if len(centers) < 3:
        raise ValueError('Need at least three ring centers for fused topology')
    triangles = Delaunay(centers).simplices
    candidates = sorted({tuple(sorted((int(a), int(c)))) for t in triangles for a, c in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))})
    pairs = [(i, j, float(np.linalg.norm(centers[i] - centers[j])), float(fused_wall_score(R, centers[i], centers[j], b))) for i, j in candidates]
    pairs = [p for p in pairs if dwin[0] * b < p[2] < dwin[1] * b and p[3] > 0]   # fused side = 1.3-1.8b; gap-spanning links are not sides
    adj = [[] for _ in centers]
    for i, j, *_ in pairs:
        adj[i].append(j); adj[j].append(i)
    ns = {}
    for i, js in enumerate(adj):
        angles = [np.arctan2(centers[j, 1] - centers[i, 1], centers[j, 0] - centers[i, 0]) for j in js]
        ns[i] = 5 if n_score_ang(angles, 5) > n_score_ang(angles, 6) else 6
    return pairs, ns


def consistent_pairs(centers, pairs, ns, b, tol=0.45):
    """Drop fused-side proposals whose claimed shared EDGE does not
    geometrically coincide: after per-ring phase assignment, the side of ring i
    facing j and the side of j facing i must cover the same two corner
    positions (within tol*b), and no ring corner may be claimed by two
    different fused sides. Returns the filtered pair list."""
    centers = np.asarray(centers, float)
    adj, phase, side = _assign_sides(centers, pairs, ns)
    # corner positions per ring: vertex k sits at phase - pi/n + 2pi k/n
    def corners(i):
        n = ns[i]
        th = phase[i] - np.pi / n + 2 * np.pi * np.arange(n) / n
        return centers[i] + b * CIRC[n] * np.stack((np.cos(th), np.sin(th)), axis=1)
    keep = []
    for pi, (i, j, d, w) in enumerate(pairs):
        if (i, j) not in side or (j, i) not in side: continue
        ki, kj = side[i, j], side[j, i]
        ci, cj = corners(i), corners(j)
        seg_i = ci[[ki, (ki + 1) % ns[i]]]; seg_j = cj[[kj, (kj + 1) % ns[j]]]
        err = min(np.linalg.norm(seg_i - seg_j, axis=1).mean(),
                  np.linalg.norm(seg_i - seg_j[::-1], axis=1).mean()) / b
        if err < tol:
            keep.append((i, j, d, w))
    return keep


def _assign_sides(centers, pairs, ns):
    """Per-ring phase fit + Hungarian side assignment (as graph_from_fused_rings).
    Returns (adj, side dict (i,j)->side index of i facing j)."""
    nring = len(centers)
    adj = [[] for _ in range(nring)]
    for i, j, *_ in pairs:
        adj[i].append(j); adj[j].append(i)
    phase, side = {}, {}
    for i in range(nring):
        n = ns[i]
        if not adj[i]:
            phase[i] = 0.0; continue
        alpha = np.arctan2(centers[np.asarray(adj[i]), 1] - centers[i, 1], centers[np.asarray(adj[i]), 0] - centers[i, 0])
        best = (np.inf, None, None)
        for psi in np.linspace(0, 2 * np.pi / n, 181, endpoint=False):
            normals = psi + 2 * np.pi * np.arange(n) / n
            err = np.abs(np.angle(np.exp(1j * (alpha[:, None] - normals[None, :]))))
            rows, cols = linear_sum_assignment(err)
            v = np.square(err[rows, cols]).sum()
            if v < best[0]: best = (v, psi, cols)
        phase[i] = best[1]
        for j, k in zip(adj[i], best[2]):
            side[i, j] = int(k)
    return adj, phase, side


def _legal(pairs, ns, centers, b):
    try:
        graph_from_fused_rings(centers, pairs, ns, b); return True
    except ValueError:
        return False


def vertex_collisions(verts, edges, rings, tol):
    """Hard chemical check: two DISTINCT atoms can never be closer than a bond.
    Returns {ring_pair: n_collisions} for every pair of rings whose drawn
    corners lie < tol apart without being the same vertex. Same-ring corner
    collisions count under (i,i)."""
    eset = {tuple(sorted(e)) for e in edges}
    owner = {}
    for ri, r in enumerate(rings):
        for v in r:
            owner.setdefault(v, []).append(ri)
    bad = {}
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if (i, j) in eset: continue                       # bonded atoms ARE a bond apart... but tol<b so bonded never triggers
            if np.linalg.norm(verts[i] - verts[j]) < tol:
                for ri in owner.get(i, []):
                    for rj in owner.get(j, []):
                        key = (ri, rj) if ri <= rj else (rj, ri)
                        bad[key] = bad.get(key, 0) + 1
    return bad


def _violation_vertices(centers, pairs, ns, b):
    """Replay the fused-side union. Returns dict {vertex_id: [rings]} for every
    degree->3 carbon plus list of rings whose corners collapse."""
    centers = np.asarray(centers, float)
    nring = len(centers)
    offsets = np.r_[0, np.cumsum([ns[i] for i in range(nring)])]
    parent = np.arange(offsets[-1])
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, c):
        parent[root(c)] = root(a)
    adj, phase, side = _assign_sides(centers, pairs, ns)
    overflow = [i for i in range(nring) if len(adj[i]) > ns[i]]
    try:
        for i, j, *_ in pairs:
            ki, kj = side[i, j], side[j, i]
            union(offsets[i] + ki, offsets[j] + (kj + 1) % ns[j]); union(offsets[i] + (ki + 1) % ns[i], offsets[j] + kj)
    except KeyError:
        return {}, overflow if overflow else list(range(nring))
    roots = np.array([root(q) for q in range(offsets[-1])])
    _, ids = np.unique(roots, return_inverse=True)
    rings = [ids[offsets[i]:offsets[i + 1]].tolist() for i in range(nring)]
    collapsed = [i for i, r in enumerate(rings) if len(set(r)) != len(r)]
    edges = sorted({tuple(sorted((r[k], r[(k + 1) % len(r)]))) for r in rings for k in range(len(r))})
    deg = np.bincount(np.asarray(edges).ravel(), minlength=ids.max() + 1) if edges else np.zeros(1, int)
    badv = {v: [i for i, r in enumerate(rings) if v in r] for v in np.where(deg > 3)[0]}
    return badv, collapsed


def repair_fused_graph(centers, pairs, ns, b, R=None, max_iter=60):
    """Repair an invalid fused dual WITHOUT losing rings. Degree-4 carbons come
    from 'V' triple junctions — three rings claiming one vertex via only two
    fused pairs. Physical fixes tried in order:
      1. ADD the missing third pair when the two centers are at fused distance
         (d < ~1.95b) — coronene-type triple junction.
      2. FLIP n of a ring at the vertex (5<->6 changes side spacing 60<->72 deg).
      3. Only then is one claimed fusion false: report the violating rings as
         suspects for the caller (a ring center must be a false positive —
         min-distance prior forbids close non-fused rings).
    Returns (pairs, ns, suspects) — suspects empty on success."""
    centers = np.asarray(centers, float)
    pairs = list(pairs); ns = dict(ns)
    def refit_ns():
        adj = [[] for _ in range(len(centers))]
        for i, j, *_ in pairs:
            adj[i].append(j); adj[j].append(i)
        for i, js in enumerate(adj):
            if js:
                a = [np.arctan2(centers[j, 1] - centers[i, 1], centers[j, 0] - centers[i, 0]) for j in js]
                ns[i] = 5 if n_score_ang(a, 5) > n_score_ang(a, 6) else 6
    suspects = []
    for _ in range(max_iter):
        if _legal(pairs, ns, centers, b):
            refit_ns()
            if not _legal(pairs, ns, centers, b):
                continue
            verts, edges, rings = graph_from_fused_rings(centers, pairs, ns, b)
            coll = vertex_collisions(verts, edges, rings, 0.6 * b)
            if not coll:
                return pairs, ns, []
            # unbonded atoms closer than a bond — impossible. If the two rings
            # are at fused distance with free sides they MUST fuse; else FP.
            adj0 = [[] for _ in range(len(centers))]
            for i, j, *_ in pairs:
                adj0[i].append(j); adj0[j].append(i)
            pset0 = {tuple(sorted((p[0], p[1]))) for p in pairs}
            progressed = False
            for (i, j), cnt in sorted(coll.items(), key=lambda kv: -kv[1]):
                if i == j:
                    suspects = [i]; return pairs, ns, suspects
                if tuple(sorted((i, j))) in pset0: continue
                d = float(np.linalg.norm(centers[i] - centers[j]))
                if d < 1.95 * b and len(adj0[i]) < ns[i] and len(adj0[j]) < ns[j]:
                    w = float(fused_wall_score(R, centers[i], centers[j], b)) if R is not None else 0.0
                    pairs.append((i, j, d, w)); progressed = True; break
            if not progressed:
                (i, j), _ = max(coll.items(), key=lambda kv: kv[1])
                suspects = [i, j]
                return pairs, ns, suspects
            continue
        badv, collapsed = _violation_vertices(centers, pairs, ns, b)
        adj = [[] for _ in range(len(centers))]
        for i, j, *_ in pairs:
            adj[i].append(j); adj[j].append(i)
        pset = {tuple(sorted((p[0], p[1]))): p for p in pairs}
        if collapsed:
            cand = [p for p in pairs if collapsed[0] in (p[0], p[1])]
            if not cand:
                suspects = [collapsed[0]]; return pairs, ns, suspects
            pairs.remove(min(cand, key=lambda p: p[3])); continue
        if not badv:
            return pairs, ns, []
        rh = badv[sorted(badv)[0]]
        # 1) add a missing pair inside the junction when fused distance allows
        missing = [(i, j) for i, j in itertools.combinations(rh, 2) if tuple(sorted((i, j))) not in pset]
        added = False
        for i, j in sorted(missing, key=lambda p: np.linalg.norm(centers[p[0]] - centers[p[1]])):
            d = float(np.linalg.norm(centers[i] - centers[j]))
            if d > 1.95 * b: continue
            if len(adj[i]) >= ns[i] or len(adj[j]) >= ns[j]: continue
            w = float(fused_wall_score(R, centers[i], centers[j], b)) if R is not None else 0.0
            trial = pairs + [(i, j, d, w)]
            if _legal(trial, ns, centers, b) or len(_violation_vertices(centers, trial, ns, b)[0]) < len(badv):
                pairs = trial; added = True; break
        if added: continue
        # 2) flip n of a ring at the vertex if it reduces violations
        flipped = False
        for w in rh:
            ns2 = dict(ns); ns2[w] = 11 - ns[w]
            if len(adj[w]) > ns2[w]: continue
            nb2, _ = _violation_vertices(centers, pairs, ns2, b)
            if len(nb2) < len(badv):
                ns[w] = ns2[w]; flipped = True; break
        if flipped: continue
        # 3) unresolvable by side-level edits — these rings contain a false center
        suspects = rh
        return pairs, ns, suspects
    return pairs, ns, suspects if suspects else list(range(len(centers)))


def graph_from_ring_centers(centers, R, b):
    """Infer a planar fused dual and exact carbon graph from reviewed centers.

    Candidate shared sides come from Delaunay adjacency and bright wall
    evidence, geometrically filtered by consistent_pairs, then repaired by
    repair_fused_graph (drops weakest false-fusion pair at degree-4 vertices).
    Rare pentagons follow fivefold neighbor-angle consistency.
    """
    centers = np.asarray(centers, float)
    pairs, ns = ring_dual_proposals(centers, R, b)
    pairs = consistent_pairs(centers, pairs, ns, b)
    pairs, ns, suspects = repair_fused_graph(centers, pairs, ns, b, R=R)
    if suspects:
        raise ValueError(f'Unrepairable fused dual; suspect rings: {suspects}')
    verts, edges, rings = graph_from_fused_rings(centers, pairs, ns, b)
    return verts, edges, rings, ns, pairs


# ---------------------------------------------------------------------------
# Ring-hypothesis scoring and global subset selection.
# Each candidate predicts polygon corners (atoms) and sides (bonds); the
# independent channels score those predictions. No channel vetoes alone.
# ---------------------------------------------------------------------------

def _z(a):
    """Robust z-score (median/MAD-ish via std), used to fuse channels."""
    a = np.asarray(a, float)
    return (a - np.median(a)) / (a.std() + 1e-9)


def annulus_score(flat, R, xy, b):
    """Center darkness + annular wall evidence at arbitrary positions (xy).
    Same quantities ring_candidates computes at its peaks; usable for
    geometrically proposed (non-peak) centers."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if not len(xy):
        return np.zeros((0, 3))
    G = ndi.gaussian_laplace(flat, 0.45 * b)
    th = 2 * np.pi * np.arange(36) / 36
    wall = np.maximum.reduce([ndi.map_coordinates(R, [xy[:, 1, None] + r * np.sin(th), xy[:, 0, None] + r * np.cos(th)],
                                                  order=1, mode='nearest') for r in (0.7 * b, 0.9 * b, 1.1 * b)])
    scale = np.percentile(R[R > 0], 90) + 1e-9
    c = ndi.map_coordinates(G, [xy[:, 1], xy[:, 0]], order=1, mode='nearest')
    return np.stack([c / scale, wall.mean(axis=1) / scale, np.quantile(wall, 0.25, axis=1) / scale], axis=1)


def ring_hypotheses(flat, R, xy, b):
    """Independent evidence channels per candidate ring center (xy).

    rim_q10 : coherent polygon rim contrast (fit_ring_rims), best n in {5,6}
    vertex  : mean junction evidence at predicted polygon corners (V2/V3)
    bond    : per-side line-vs-flank evidence, mean + low quantile
    gap     : count-weighted penalty for consecutive missing sides (open bay)
    Returns dict of per-candidate arrays; rim_n/side_len/phase give geometry.
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    N = len(xy)
    out = dict(rim_q10=np.full(N, -np.inf), rim_n=np.zeros(N, int),
               side_len=np.full(N, b), phase=np.zeros(N),
               vertex=np.zeros(N), bond=np.zeros(N), gap=np.zeros(N))
    if not N:
        return out
    rim = fit_ring_rims(flat, xy, b)
    out['rim_q10'] = rim[:, 0]; out['rim_n'] = rim[:, 1].astype(int)
    out['side_len'] = rim[:, 2]; out['phase'] = rim[:, 3]
    scale = np.percentile(R[R > 0], 90) + 1e-9
    corners, ring_of, k_of = [], [], []                  # all predicted atoms
    for i in range(N):
        n = out['rim_n'][i]
        if n not in (5, 6):
            continue
        th = out['phase'][i] - np.pi / n + 2 * np.pi * np.arange(n) / n
        cs = xy[i][None, :] + out['side_len'][i] * CIRC[n] * np.stack([np.cos(th), np.sin(th)], 1)
        corners.append(cs); ring_of += [i] * n; k_of += list(range(n))
    if not corners:
        return out
    corners = np.vstack(corners); ring_of = np.asarray(ring_of); k_of = np.asarray(k_of)
    V2, V3 = vertex_response_at(R, corners, b)
    vv = np.maximum(V2, V3) / scale
    pa, pb = corners, corners[np.arange(len(corners)) - 0]  # side endpoints
    nxt = np.arange(len(corners))
    # next corner within same ring: k_of+1 mod n
    for i in range(N):
        m = np.flatnonzero(ring_of == i)
        n = len(m)
        pa[m] = corners[m]; pb[m] = corners[m[(np.arange(n) + 1) % n]]
    w = bond_evidence_lines(R, pa, pb, b) / scale
    miss = 1.0 / (1.0 + np.exp(2.0 * w))                # ~1 when side ridge dark
    for i in range(N):
        m = np.flatnonzero(ring_of == i)
        if not len(m):
            continue
        out['vertex'][i] = vv[m].mean()
        wi = w[m]
        out['bond'][i] = wi.mean() + 0.5 * np.quantile(wi, 0.25)
        mi = miss[m]
        out['gap'][i] = float(np.sum(mi * np.roll(mi, -1)))
    return out


def candidate_pair_factors(xy, R, b):
    """Candidate fused-ring pairs (Delaunay + all too-close conflicts).

    F_ij = shared-wall ridge score + endpoint junction evidence + distance
    prior. Pairs closer than a fused side-length that are NOT wall-bright are
    kept as conflicts (two occupied centers there physically overlap)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    N = len(xy)
    rows = []
    if N >= 3:
        tris = Delaunay(xy).simplices
        cand = sorted({tuple(sorted((int(a), int(c)))) for t in tris for a, c in itertools.combinations(t, 2)})
    else:
        cand = [(i, j) for i in range(N) for j in range(i + 1, N)]
    near = cKDTree(xy).query_pairs(1.15 * b)
    cand = sorted(set(cand) | {tuple(sorted(p)) for p in near})
    ep_list = []
    for i, j in cand:
        d = np.linalg.norm(xy[i] - xy[j])
        if d > 2.4 * b:
            continue
        u = (xy[j] - xy[i]) / max(d, 1e-9); perp = np.array([-u[1], u[0]])
        mid = 0.5 * (xy[i] + xy[j]); s = d / 1.732
        ep_list.append(np.stack([mid + 0.5 * s * perp, mid - 0.5 * s * perp]))
        rows.append([i, j, float(d), 0.0, 0.0])
    if not rows:
        return rows
    ep = np.concatenate(ep_list)
    V2e, V3e = vertex_response_at(R, ep, b)
    ve = np.maximum(V3e, V2e).reshape(-1, 2).mean(axis=1)
    scale = np.percentile(R[R > 0], 90) + 1e-9
    for k, (i, j, d, w, _v) in enumerate(rows):
        rows[k][3] = float(fused_wall_score(R, xy[i], xy[j], b))
        rows[k][4] = float(ve[k] / scale)
    return [tuple(r) for r in rows]


def select_rings(xy, S, pairs, b, niter=None, seed=0, verbose=False):
    """Anneal ring occupancy: maximize sum S_i + sum fused-pair factors under
    chemical constraints (connected fused patch, no non-fused overlaps,
    isolated rings penalized). Centers fused iff close AND wall-bright."""
    N = len(xy)
    if not N:
        return np.zeros(0, bool)
    pd = [[] for _ in range(N)]
    for i, j, d, w, ve in pairs:
        pd[i].append((j, d, w, ve)); pd[j].append((i, d, w, ve))

    def energy(occ):
        ids = np.flatnonzero(occ)
        if not len(ids):
            return 0.0
        E = -float(S[ids].sum()) + 0.8 * len(ids)
        adj = {i: [] for i in ids}
        for i in ids:
            for j, d, w, ve in pd[i]:
                if j <= i or not occ[j]:
                    continue
                if d < 1.15 * b:
                    E += 12.0                              # centers too close to coexist
                elif w > 0:
                    E -= 0.30 * min(w, 3.0) + 0.15 * min(ve, 3.0)   # fused pair: bounded reward
                    adj[i].append(j); adj[j].append(i)
                elif d < 2.1 * b:
                    E += 8.0                               # occupied, not fused, overlapping rims
        seen = set(); ncomp = 0; niso = 0
        for i in ids:
            if i in seen:
                continue
            ncomp += 1; niso += (not adj[i])
            stack = [i]; seen.add(i)
            while stack:
                for j in adj[stack.pop()]:
                    if j not in seen:
                        seen.add(j); stack.append(j)
        return E + 10.0 * (ncomp - 1) + 4.0 * niso

    niter = niter or max(8000, 150 * N)
    rng = np.random.default_rng(seed)
    occ = S > 0.0
    E = energy(occ); best = (E, occ.copy())
    for it in range(niter):
        T = 2.0 * (0.01 / 2.0) ** (it / niter)
        k = int(rng.integers(N))
        occ[k] = ~occ[k]
        E2 = energy(occ)
        if E2 <= E or rng.random() < np.exp(-(E2 - E) / T):
            E = E2
            if E2 < best[0]:
                best = (E2, occ.copy())
        else:
            occ[k] = ~occ[k]
        if verbose and it % 5000 == 0:
            print(f'    anneal it={it} T={T:.3f} E={E:.1f} occ={occ.sum()}')
    return best[1]


LATTICE_NN = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


def _deg_violation_rings(cxy, sub_pairs, ns):
    """Replay the shared-corner union of graph_from_fused_rings; return the set
    of ring indices contributing to any vertex with >3 distinct rings, any
    vertex with >3 incident EDGES (degree-4 carbon — 3 rings at a vertex with
    inconsistent corner alignment also produces this), or any ring whose own
    corners collapsed. Diagnostic for repair, not geometry."""
    from collections import defaultdict
    nring = len(cxy)
    offsets = np.r_[0, np.cumsum([ns[i] for i in range(nring)])]
    parent = np.arange(offsets[-1])
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, c):
        parent[root(c)] = root(a)
    adj = [[] for _ in range(nring)]
    for i, j, *_ in sub_pairs:
        adj[i].append(j); adj[j].append(i)
    phase, side = {}, {}
    for i in range(nring):
        n = ns[i]
        if not adj[i]:
            phase[i] = 0.0; continue
        alpha = np.array([np.arctan2(cxy[j, 1] - cxy[i, 1], cxy[j, 0] - cxy[i, 0]) for j in adj[i]])
        best = (np.inf, None, None)
        for psi in np.linspace(0, 2 * np.pi / n, 181, endpoint=False):
            normals = psi + 2 * np.pi * np.arange(n) / n
            err = np.abs(np.angle(np.exp(1j * (alpha[:, None] - normals[None, :]))))
            rows, cols = linear_sum_assignment(err)
            v = np.square(err[rows, cols]).sum()
            if v < best[0]:
                best = (v, psi, cols)
        phase[i] = best[1]
        for j, k in zip(adj[i], best[2]):
            side[i, j] = int(k)
    for i, j, *_ in sub_pairs:
        if (i, j) not in side or (j, i) not in side:
            continue
        ki, kj = side[i, j], side[j, i]
        union(offsets[i] + ki, offsets[j] + (kj + 1) % ns[j])
        union(offsets[i] + (ki + 1) % ns[i], offsets[j] + kj)
    rings_at = defaultdict(set)
    for i in range(nring):
        for k in range(ns[i]):
            rings_at[root(offsets[i] + k)].add(i)
    bad = set()
    for rs in rings_at.values():
        if len(rs) > 3:
            bad |= rs
    for i in range(nring):
        if len({root(offsets[i] + k) for k in range(ns[i])}) != ns[i]:
            bad.add(i)
        if len(adj[i]) > ns[i]:                             # more fused sides than the polygon has
            bad.add(i)
    # degree-4 carbons: count edges incident on each merged vertex
    roots = np.array([root(a) for a in range(offsets[-1])])
    _, ids = np.unique(roots, return_inverse=True)
    vedges = defaultdict(set)
    for i in range(nring):
        for k in range(ns[i]):
            vedges[tuple(sorted((ids[offsets[i] + k], ids[offsets[i] + (k + 1) % ns[i]])))].add(i)
    vdeg = defaultdict(set)
    for (a, c), rs in vedges.items():
        vdeg[a] |= rs; vdeg[c] |= rs
    deg = defaultdict(int)
    for a, c in vedges:
        deg[a] += 1; deg[c] += 1
    for v, d in deg.items():
        if d > 3:
            bad |= vdeg[v]
    return bad


def bridge_components(xy, occ, lattice):
    """Connect occupied components through the cheapest lattice paths.

    'One molecule' as a constructive constraint: multi-source Dijkstra over
    lattice sites (occupied cost 0, empty candidate site cost 1) finds the
    shortest path linking two components; its empty sites get occupied.
    Iterates until connected or no bridge exists."""
    import heapq
    if lattice is None:
        return occ
    origin, B = lattice
    occ = occ.copy()
    qi_all = np.rint(np.linalg.solve(B, (xy - origin).T).T).astype(int)
    cand_at = {}
    for i in range(len(xy)):
        cand_at.setdefault(tuple(qi_all[i]), i)
    for _ in range(16):
        site_of = {}
        for i in np.flatnonzero(occ):
            site_of.setdefault(tuple(qi_all[i]), i)
        if len(site_of) < 2:
            return occ
        comp = {}; nc = 0
        for s in site_of:
            if s in comp:
                continue
            comp[s] = nc; stack = [s]
            while stack:
                a = stack.pop()
                for dx, dy in LATTICE_NN:
                    ns_ = (a[0] + dx, a[1] + dy)
                    if ns_ in site_of and ns_ not in comp:
                        comp[ns_] = nc; stack.append(ns_)
            nc += 1
        if nc <= 1:
            return occ
        dist = {}; src = {}; parent = {}; pq = []
        for s, c in comp.items():
            dist[s] = 0; src[s] = c; parent[s] = None
            heapq.heappush(pq, (0, s, c))
        merged = False
        while pq and not merged:
            cost, s, k = heapq.heappop(pq)
            if dist.get(s) != cost or src.get(s) != k:
                continue
            for dx, dy in LATTICE_NN:
                nb = (s[0] + dx, s[1] + dy)
                step = 0 if nb in site_of else (1 if nb in cand_at else 4)  # non-candidate site: costly
                nd = cost + step
                if nb not in src:
                    dist[nb] = nd; src[nb] = k; parent[nb] = s
                    heapq.heappush(pq, (nd, nb, k))
                elif src[nb] != k:
                    for t in (s, nb):                        # walk both fronts back, occupy empties
                        while parent.get(t) is not None:
                            if t in cand_at:
                                occ[cand_at[t]] = True
                            t = parent[t]
                    merged = True
                    break
        if not merged:
            return occ
    return occ


def fused_topology(xy, occ, pairs, hyp, b, S=None, lattice=None):
    """Selected centers -> snapped lattice dual -> n in {5,6} -> shared graph.

    With `lattice=(origin, B)` the occupied centers are snapped to triangular
    ring-center sites; fused pairs are then exactly the occupied lattice
    nearest-neighbors, so interior atoms are shared by <=3 rings by
    construction and degree>3 carbon cannot arise. Pentagon identity is
    decided on the measured (unsnapped) neighbor directions; rim n_fit breaks
    ties. Sites >0.4 lattice spacings off-grid are treated as spurious and
    dropped. Without a lattice, falls back to wall-bright Delaunay pairs."""
    occ = occ.copy()
    force6 = set()                                       # rings retried as hexagon
    for _attempt in range(60):
        ids = np.flatnonzero(occ)
        cxy_meas = xy[ids]
        if lattice is not None:
            origin, B = lattice
            d0 = np.linalg.norm(B[:, 0])
            q = np.linalg.solve(B, (cxy_meas - origin).T).T
            qi = np.rint(q)
            off = np.linalg.norm((q - qi) @ B.T, axis=1)
            ok = off < 0.4 * d0
            if not ok.all():                             # off-lattice -> spurious, rescore set
                occ[ids[~ok]] = False
                continue
            order = np.argsort(-S[ids]) if S is not None else np.arange(len(ids))
            seen = set(); keep = []
            for k in order:
                tq = tuple(qi[k])
                if tq not in seen:
                    seen.add(tq); keep.append(k)
            if len(keep) < len(ids):                     # two candidates snapped to one site
                occ[ids[np.setdiff1d(np.arange(len(ids)), keep)]] = False
                continue
            cxy = origin + qi @ B.T                      # snapped geometry for topology
            orig_w = {(i, j): w for i, j, d, w, ve in pairs}
            nbs = cKDTree(cxy).query_pairs(1.15 * d0)
            adj = {k: set() for k in range(len(ids))}
            pw = {}
            for a, c in nbs:
                w = orig_w.get(tuple(sorted((int(ids[a]), int(ids[c])))), 0.0)
                pw[(a, c)] = (float(np.linalg.norm(cxy[a] - cxy[c])), w)
                adj[a].add(c); adj[c].add(a)
        else:
            cxy = cxy_meas
            adj = {k: set() for k in range(len(ids))}
            pw = {}
            remap = {c: k for k, c in enumerate(ids)}
            for i, j, d, w, ve in pairs:
                if occ[i] and occ[j] and w > 0 and d > 1.15 * b:
                    a, c = remap[i], remap[j]
                    pw[(a, c)] = (d, w)
                    adj[a].add(c); adj[c].add(a)
        if any(not adj[k] for k in adj):                 # lone rings can't fuse -> drop
            for k in [k for k in adj if not adj[k]]:
                occ[ids[k]] = False
            continue
        ns = {}
        for k in range(len(cxy)):
            al = [np.arctan2(cxy_meas[j, 1] - cxy_meas[k, 1], cxy_meas[j, 0] - cxy_meas[k, 0]) for j in adj[k]]
            s5, s6 = n_score_ang(al, 5), n_score_ang(al, 6)
            nfit = hyp['rim_n'][ids[k]]
            # strong hex prior: pentagon needs strictly better angular match
            # (>=2 neighbors) or exact tie with >=3 neighbors AND rim fit 5
            ns[k] = 6 if ids[k] in force6 else \
                    (5 if (len(al) >= 2 and s5 > s6) or
                          (len(al) >= 3 and len(al) <= 5 and s5 == s6 and s5 >= 0.75 and nfit == 5) else 6)
        crowding = [k for k in adj if len(adj[k]) > ns[k]]
        if crowding:                                     # fused degree exceeds ring sides
            k = crowding[0]
            weakest = min(adj[k], key=lambda j: pw[tuple(sorted((k, j)))][1])
            del pw[tuple(sorted((k, weakest)))]
            adj[k].discard(weakest); adj[weakest].discard(k)
            continue
        sub_pairs = [(a, c, d, w) for (a, c), (d, w) in pw.items()]
        comp = {}; nc = 0                                 # components of the fused dual
        for k in adj:
            if k in comp:
                continue
            comp[k] = nc; stack = [k]
            while stack:
                for j in adj[stack.pop()]:
                    if j not in comp:
                        comp[j] = nc; stack.append(j)
            nc += 1
        if nc > 1:                                       # one molecule: keep largest patch
            sizes = np.bincount([comp[k] for k in adj])
            main = int(np.argmax(sizes))
            for k, c in comp.items():
                if c != main:
                    occ[ids[k]] = False
            continue
        try:
            verts, edges, rings = graph_from_fused_rings(cxy, sub_pairs, ns, b)
            return verts, edges, rings, cxy, sub_pairs, ns, ids
        except ValueError as ex:
            bad = _deg_violation_rings(cxy, sub_pairs, ns)
            if not bad:
                bad = set(adj)
            pent = [k for k in bad if ns[k] == 5 and ids[k] not in force6]
            if pent:                                     # a misassigned pentagon is the usual culprit
                victim = min(pent, key=lambda k: S[ids[k]] if S is not None else 0)
                print(f'    fused_topology: {ex}; retrying ring {ids[victim]} as hexagon')
                force6.add(ids[victim])
                continue
            victim = min(bad, key=lambda k: S[ids[k]] if S is not None else 0)
            print(f'    fused_topology: {ex}; dropping ring {ids[victim]} (S={S[ids[victim]]:.2f})')
            occ[ids[victim]] = False                     # weakest offending ring out
    raise ValueError('fused_topology: could not repair ring selection (60 attempts)')


def align_ring_template(reference, proposals, scores, b, reference_b, flat=None):
    """Fit a reviewed ring-center pattern to noisy proposals by similarity voting.

    This is for repeat images of the *same* molecular topology. It does not
    determine the topology of an unrelated molecule. Returns transformed ring
    centers, the number of supported centers, and fit parameters (scale, angle,
    translation, reflection). No ring count is inferred from the target image here.
    """
    reference = np.asarray(reference, float)
    proposals = np.asarray(proposals, float)
    if len(reference) < 3 or len(proposals) < 3:
        raise ValueError('Template alignment needs at least three rings and proposals')
    origin = reference.mean(axis=0)
    src = reference - origin
    order = np.argsort(scores[:, 1])[::-1][:min(45, len(proposals))]
    dst = proposals[order]
    weight = np.clip(scores[order, 1], 0, 2)
    spacing = max(1.0, 0.30 * b)
    vote_range = int(np.ceil((np.max(proposals) - np.min(proposals) + 2 * b) / spacing)) + 1
    offset = np.min(proposals, axis=0) - b
    best = (-np.inf, None)
    tree = cKDTree(proposals)
    mask = molecule_mask(flat, 0.4 * b) if flat is not None else None
    for mirror in (1, -1):
      reflected = src * np.array([mirror, 1])
      for sf in np.linspace(0.80, 1.20, 9):
        scale = sf * b / reference_b
        for angle in np.deg2rad(np.arange(0, 360, 3)):
            ca, sa = np.cos(angle), np.sin(angle)
            rot = np.array([[ca, -sa], [sa, ca]])
            points = scale * (reflected @ rot.T)
            shifts = dst[:, None, :] - points[None, :, :]
            bins = np.rint((shifts - offset) / spacing).astype(int)
            valid = np.all((bins >= 0) & (bins < vote_range), axis=-1)
            flat = bins[..., 1] * vote_range + bins[..., 0]
            votes = np.bincount(flat[valid], weights=np.broadcast_to(weight[:, None], valid.shape)[valid], minlength=vote_range**2)
            top = np.argpartition(votes, -3)[-3:]
            for k in top:
                translation = offset + spacing * np.array([k % vote_range, k // vote_range])
                predicted = points + translation
                distance, near = tree.query(predicted)
                support = np.exp(-0.5 * (distance / (0.35 * b))**2)
                value = np.sum(support * np.clip(scores[near, 1], 0, 2))
                if mask is not None:
                    ix = np.rint(predicted[:, 0]).astype(int)
                    iy = np.rint(predicted[:, 1]).astype(int)
                    inside = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
                    value += 2.0 * np.count_nonzero(inside & mask[np.clip(iy, 0, mask.shape[0] - 1), np.clip(ix, 0, mask.shape[1] - 1)])
                if value > best[0]:
                    best = (value, (scale, angle, translation, mirror))
    scale, angle, translation, mirror = best[1]
    ca, sa = np.cos(angle), np.sin(angle)
    rot = np.array([[ca, -sa], [sa, ca]])
    predicted = scale * ((src * np.array([mirror, 1])) @ rot.T) + translation
    distance, near = tree.query(predicted)
    supported = int(np.count_nonzero((distance < 0.45 * b) & (scores[near, 1] > 0.3)))
    return predicted, supported, (scale, angle, translation, mirror)


def refine_verts(verts, ridges, r):
    """Snap each vertex to the centroid of ridge pixels within radius r (xy coords)."""
    out = verts.copy()
    for k, v in enumerate(verts):
        if not (0 <= v[0] < ridges.shape[1] and 0 <= v[1] < ridges.shape[0]):
            continue                                     # vertex outside image
        y0, y1 = max(0, int(v[1] - r)), min(ridges.shape[0], int(v[1] + r) + 1)
        x0, x1 = max(0, int(v[0] - r)), min(ridges.shape[1], int(v[0] + r) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        sel = ridges[yy, xx] & ((yy - v[1])**2 + (xx - v[0])**2 < r * r)
        if sel.sum() > 3:
            out[k] = [xx[sel].mean(), yy[sel].mean()]
    return out


def _merge_new_centers(xy, extra, rmin):
    """Append proposed positions farther than rmin from any existing center."""
    if not len(extra):
        return xy
    keep = []
    for p in np.asarray(extra, float).reshape(-1, 2):
        if not len(xy) or np.min(np.linalg.norm(xy - p, axis=1)) > rmin:
            keep.append(p)
    return np.vstack([xy, np.asarray(keep)]) if keep else xy


def img_to_graph(path, bond_px0=None, verbose=True, seed=0):
    """Evidence-fusion pipeline (doc/chats/Image2mol.chat.md).

    Parallel channels: multiscale ridge map R, V2/V3 junction maps, escape-
    barrier basin map, annular ring response. Overcomplete center proposals
    (peaks + lattice sites + fused-geometry predictions) are scored by all
    channels; occupancy + fused topology are selected globally under chemical
    constraints (connected patch, no overlapping rings, deg<=3 enforced by
    graph_from_fused_rings). No single feature can veto or approve a ring."""
    g = load_gray(path)
    flat = flatten_bg(g)
    b = bond_px0 or estimate_bond_scale(flat)
    best = None                                          # keep the pass with most fused rings
    for it in range(2):                                  # outer loop re-scales b once
        R = ridge_map_ms(flat, b)
        mask = molecule_mask(flat, 0.4 * b)
        C = center_response(flat, b)
        xy = propose_centers(C, mask, b)
        # triangular-lattice sites inside the molecule as additional hypotheses:
        # a real ring missing from peak proposals still gets scored on its site
        lattice = None
        if len(xy) >= 6:
            try:
                origin, B, _, _ = fit_center_lattice(xy, annulus_score(flat, R, xy, b), b)
                lattice = (origin, B)
            except ValueError:
                pass
            if lattice is not None:
                mbig = binary_dilation(mask, disk(max(1, int(0.5 * b))))
                ys, xs = np.nonzero(mbig)
                q = np.linalg.solve(B, (np.stack([xs, ys], 1) - origin).T).T
                qi = np.rint(q)
                sel = np.abs(q - qi).max(axis=1) < 0.4     # mask pixels near lattice sites
                sites = np.unique(qi[sel], axis=0)
                latpts = origin + sites @ B.T
                xy = _merge_new_centers(xy, latpts, 0.55 * b)
        hyp = ring_hypotheses(flat, R, xy, b)
        sc3 = annulus_score(flat, R, xy, b)
        cen = -ndi.map_coordinates(C, [xy[:, 1], xy[:, 0]], order=1, mode='nearest')
        S = (0.8 * _z(cen) + 0.6 * _z(sc3[:, 1]) + 0.5 * _z(sc3[:, 2])
             + 0.8 * _z(hyp['rim_q10']) + 0.8 * _z(hyp['vertex'])
             + 1.0 * _z(hyp['bond']) - 1.5 * hyp['gap'])
        pairs = candidate_pair_factors(xy, R, b)
        occ = select_rings(xy, S, pairs, b, seed=seed, verbose=verbose)
        # rescue round: geometrically proposed centers from accepted pairs
        extra = propose_fused_ring_centers(xy[occ], R, b)
        extra = np.asarray(extra).reshape(-1, 2)
        if len(extra):
            dmin = np.min(np.linalg.norm(xy[None, :, :] - extra[:, None, :], axis=2), axis=1)
            extra = extra[dmin > 0.55 * b]
        if len(extra):
            xy = np.vstack([xy, extra])
            hyp = ring_hypotheses(flat, R, xy, b)
            sc3 = annulus_score(flat, R, xy, b)
            cen = -ndi.map_coordinates(C, [xy[:, 1], xy[:, 0]], order=1, mode='nearest')
            S = (0.8 * _z(cen) + 0.6 * _z(sc3[:, 1]) + 0.5 * _z(sc3[:, 2])
                 + 0.8 * _z(hyp['rim_q10']) + 0.8 * _z(hyp['vertex'])
                 + 1.0 * _z(hyp['bond']) - 1.5 * hyp['gap'])
            pairs = candidate_pair_factors(xy, R, b)
            occ = select_rings(xy, S, pairs, b, seed=seed + 1, verbose=verbose)
        occ = bridge_components(xy, occ, lattice)
        try:
            verts, edges, rings, cxy, sub_pairs, ns, sel_ids = fused_topology(xy, occ, pairs, hyp, b, S=S, lattice=lattice)
        except ValueError:
            if verbose:
                print(f'  pass{it}: fused_topology failed', flush=True)
            continue
        if best is None or len(cxy) > len(best[3]):
            best = (verts, edges, rings, cxy, sub_pairs, ns, sel_ids, xy, sc3, S, occ, hyp, R, C, mask, b)
        b2 = np.median([d / (APO[ns[a]] + APO[ns[c]]) for a, c, d, w in sub_pairs]) if sub_pairs else b
        if verbose:
            n5 = sum(1 for v in ns.values() if v == 5)
            print(f'  pass{it}: b={b:.1f} cand={len(xy)} occ={occ.sum()} rings={len(cxy)} '
                  f'fused={len(sub_pairs)} V={len(verts)} E={len(edges)} pent={n5} b_est={b2:.1f}', flush=True)
        if abs(b2 / b - 1) < 0.12:
            break
        b = b2
    if best is None:
        raise ValueError('img_to_graph: no valid fused graph at any scale')
    verts, edges, rings, cxy, sub_pairs, ns, sel_ids, xy, sc3, S, occ, hyp, R, C, mask, b = best
    ridges = (R > np.quantile(R[mask], 0.75)) & mask
    walls = binary_dilation(ridges, disk(max(1, int(0.08 * b))))
    inner = mask & ~walls
    D = ndi.distance_transform_edt(inner)
    verts = refine_verts(verts, ridges, r=0.30 * b)
    return dict(gray=g, flat=flat, mask=mask, R=R, C=C, hyp=hyp,
                ridges=ridges, walls=walls, inner=inner, D=D,
                verts=verts, edges=edges, rings=rings, ring_ids=list(range(len(rings))),
                centers=cxy, pairs=sub_pairs, ns=ns,
                candidate_xy=xy, candidate_score=sc3, candidate_S=S,
                candidate_occ=np.isin(np.arange(len(xy)), sel_ids),
                candidate_bond_px=b, bond_px=b)


def graph_to_atomicgraph(verts, edges, bond_px, cc_len=1.42):
    """Step 2: vertex graph -> AtomicGraph of carbons (xy plane, Å)."""
    from spammm.topology.AtomicGraph import AtomicGraph
    s = cc_len / bond_px
    ag = AtomicGraph()
    atoms = [ag.add_atom([v[0] * s, -v[1] * s, 0.0], 'C', 6, npi=1) for v in verts]
    for i, j in edges:
        ag.add_bond(atoms[i], atoms[j])
    ag.detect_rings(max_ring_size=8)
    return ag


def add_hydrogens(ag, ch_len=1.09):
    """Attach H to every degree-2 carbon along the external bisector."""
    hs = []
    for a in ag.heavy_atoms():
        nbs = ag.neighbors(a)
        if len(nbs) == 2:
            d = a.pos - 0.5 * (nbs[0].pos + nbs[1].pos)
            d /= np.linalg.norm(d) + 1e-9
            hs.append(ag.add_atom(a.pos + d * ch_len, 'H', 1, npi=-1, parent=a))
            ag.add_bond(a, hs[-1])
    return hs
