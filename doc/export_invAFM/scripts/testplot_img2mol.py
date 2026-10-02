#!/usr/bin/env python3
"""testplot_img2mol.py — visual demo: AFM/STM image -> carbon skeleton graph -> molecule.

Step 1 (spammm/img2mol.img_to_graph), rings-first with chemical priors:
  heavy smoothing -> -LoG ridge walls -> molecule mask -> distance-transform
  ring centers -> watershed ring regions -> harmonic r(th) fit (n in {5,6} + phase)
  -> fused-pair adjacency validated by wall brightness -> per-ring polygon
  proposals -> clustered vertices (=sp2 C) + ring edges.

Step 2: graph_to_atomicgraph -> AtomicGraph (all C, C-C=1.42 A) -> XYZ, optional rim H.

Usage:
  python3 doc/export_invAFM/scripts/testplot_img2mol.py --img 1.png
  python3 doc/export_invAFM/scripts/testplot_img2mol.py --all            # images 1-14 + big.png
  python3 doc/export_invAFM/scripts/testplot_img2mol.py --img-dir /path/to/images --all --filters
  python3 doc/export_invAFM/scripts/testplot_img2mol.py --img 1.png --bpx 8 --H

Artifacts -> debug/testplot_img2mol/<imgname>/
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_repo = _here
while _repo != os.path.dirname(_repo) and not os.path.isdir(os.path.join(_repo, 'spammm')):
    _repo = os.path.dirname(_repo)      # repo root = dir containing spammm/ (works from any depth)
for p in (_repo, _here):
    if p not in sys.path:
        sys.path.insert(0, p)
import img2mol

IMGDIR = '/home/prokop/Desktop/PROJECTs/Svec_Ditriptaceneo_Helicene/Images'
BONUSDIR = '/home/prokop/Desktop/PROJECTs/Svec_Ditriptaceneo_Helicene/Images_Bonus/set_2_shade_AFM'
OUTDIR = os.path.join(_repo, 'debug', 'testplot_img2mol')


def plot_result(res, out, name):
    g, v, e = res['gray'], res['verts'], res['edges']
    ce = res['centers']; ns = res['ns']; pairs = res['pairs']
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs[0, 0].imshow(g, cmap='gray'); axs[0, 0].set_title(f'{name} gray')
    axs[0, 1].imshow(res['R'], cmap='viridis'); axs[0, 1].contour(res['mask'], colors='r', linewidths=0.5)
    axs[0, 1].set_title('ridge resp -LoG + mask')
    axs[0, 2].imshow(res['D'], cmap='viridis')
    axs[0, 2].plot(ce[:, 0], ce[:, 1], 'r+', ms=9)
    for i, j, d, w in pairs:
        axs[0, 2].plot([ce[i, 0], ce[j, 0]], [ce[i, 1], ce[j, 1]], 'w-', lw=1)
    axs[0, 2].set_title(f'dist transf: {len(ce)} centers, {len(pairs)} fused pairs')
    axs[1, 0].imshow(g, cmap='gray'); axs[1, 0].imshow(np.where(res['walls'], 1, np.nan), cmap='autumn')
    axs[1, 0].set_title('walls (dilated ridges)')
    # --- final graph overlay
    axs[1, 1].imshow(g, cmap='gray')
    for i, j in e:
        axs[1, 1].plot([v[i, 0], v[j, 0]], [v[i, 1], v[j, 1]], 'r-', lw=1)
    import networkx as nx
    G = nx.Graph(); G.add_edges_from(e)
    deg = np.array([G.degree(i) if i in G else 0 for i in range(len(v))])
    for d, c, s in [(0, 'gray', 20), (1, 'cyan', 30), (2, 'orange', 40), (3, 'red', 60), (4, 'magenta', 80)]:
        m = deg == d
        axs[1, 1].plot(v[m, 0], v[m, 1], 'o', c=c, ms=s / 12, label=f'deg{d} n={m.sum()}')
    for i, ids in zip(res['ring_ids'], res['rings']):
        cxy = ce[i]
        axs[1, 1].text(cxy[0], cxy[1], str(ns[i]), color='lime' if ns[i] == 6 else 'yellow',
                       fontsize=10, ha='center', va='center', weight='bold',
                       bbox=dict(fc='k', alpha=0.4, pad=0.5))
    axs[1, 1].legend(fontsize=7); axs[1, 1].set_title(f'graph: {len(v)} V, {len(e)} E, {len(res["rings"])} rings')
    # --- molecule (step 2)
    axs[1, 2].set_aspect('equal')
    for i, j in e:
        axs[1, 2].plot([v[i, 0], v[j, 0]], [-v[i, 1], -v[j, 1]], 'k-', lw=2)
    axs[1, 2].plot(v[:, 0], -v[:, 1], 'o', c='0.3', ms=5)
    axs[1, 2].set_title('molecule (C skeleton)')
    for ax in axs.flat:
        ax.set_axis_off()
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()


def plot_candidates(res, out, name):
    xy, score = res['candidate_xy'], res['candidate_score']
    rank = np.argsort(score[:, 1])[::-1][:35]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(res['gray'], cmap='gray')
    ax.plot(xy[:, 0], xy[:, 1], 'c.', ms=3)
    for k, i in enumerate(rank):
        ax.plot(xy[i, 0], xy[i, 1], 'ro', ms=5, mfc='none')
        ax.text(xy[i, 0] + 2, xy[i, 1], str(k), color='yellow', fontsize=7)
    ax.set_title(f'{name}: {len(xy)} proposals; red = strongest 35 annular scores')
    ax.set_axis_off(); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_provisional_centers(res, out, name):
    """Show the current high-coverage core separately from the legacy skeleton."""
    import networkx as nx
    from scipy.spatial.distance import cdist
    xy, score = res['candidate_xy'], res['candidate_score']
    b = res['candidate_bond_px']; R = -img2mol.ndi.gaussian_laplace(res['flat'], max(0.8, b / 7.0))
    ids = np.flatnonzero((score[:, 1] > 0.7) & (score[:, 2] > 0.45))
    d = cdist(xy[ids], xy[ids])
    graph = nx.Graph(); graph.add_nodes_from(ids.tolist())
    for a, i in enumerate(ids):
        for j in ids[a + 1:]:
            if 0.95 * b < d[a, np.where(ids == j)[0][0]] < 2.2 * b and img2mol.fused_wall_score(R, xy[i], xy[j], b) > 0:
                graph.add_edge(int(i), int(j))
    core = max(nx.connected_components(graph), key=len) if len(graph) else set()
    enclosure = img2mol.ring_enclosure(res['flat'], xy, b)
    rim = img2mol.fit_ring_rims(res['flat'], xy, b)
    core = {i for i in core if enclosure[i, 2] > 0 and rim[i, 0] > 0}
    fig, ax = plt.subplots(figsize=(10, 7)); ax.imshow(res['gray'], cmap='gray')
    for i in ids:
        ax.plot(xy[i, 0], xy[i, 1], 'o', ms=9, mfc='none', mec='lime' if i in core else 'orange', mew=1.5)
    ax.set_title(f'{name}: {len(ids)} ring proposals; green = strong enclosure, orange = weaker enclosure')
    ax.set_axis_off(); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_channels(res, out, name):
    """Evidence-fusion channels: ridge map, blob-center response, candidate
    centers colored by fused score, fitted rim polygons, final graph."""
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs[0, 0].imshow(res['gray'], cmap='gray'); axs[0, 0].set_title(f'(A) {name} gray')
    axs[0, 1].imshow(res['R'], cmap='viridis'); axs[0, 1].set_title('(B) ridge map R (multiscale, local-norm)')
    axs[0, 2].imshow(res['C'], cmap='viridis'); axs[0, 2].set_title('(C) ring-center response C (minima = centers)')
    axs[1, 0].imshow(res['gray'], cmap='gray')
    xy, S, occ = res['candidate_xy'], res['candidate_S'], res['candidate_occ']
    if len(xy):
        axs[1, 0].scatter(xy[~occ, 0], xy[~occ, 1], c=S[~occ], cmap='coolwarm', s=18, marker='x')
        axs[1, 0].plot(xy[occ, 0], xy[occ, 1], 'o', ms=10, mfc='none', mec='lime', mew=1.5)
    axs[1, 0].set_title(f'(D) {len(xy)} candidates: x color=S, green=selected')
    axs[1, 1].imshow(res['gray'], cmap='gray')
    hyp = res['hyp']
    for i in np.flatnonzero(occ):
        n = hyp['rim_n'][i]
        if n not in (5, 6):
            continue
        th = hyp['phase'][i] - np.pi / n + 2 * np.pi * np.arange(n + 1) / n
        c = img2mol.CIRC[n] * hyp['side_len'][i]
        axs[1, 1].plot(xy[i, 0] + c * np.cos(th), xy[i, 1] + c * np.sin(th),
                       'y-' if n == 5 else 'w-', lw=0.9)
    for i2, j, d, w in res['pairs']:
        axs[1, 1].plot([res['centers'][i2, 0], res['centers'][j, 0]],
                       [res['centers'][i2, 1], res['centers'][j, 1]], 'c-', lw=0.8)
    axs[1, 1].set_title('(E) fitted rims (yellow=5) + fused pairs')
    axs[1, 2].imshow(res['gray'], cmap='gray')
    v, e = res['verts'], res['edges']
    for i, j in e:
        axs[1, 2].plot([v[i, 0], v[j, 0]], [v[i, 1], v[j, 1]], 'r-', lw=1)
    axs[1, 2].plot(v[:, 0], v[:, 1], 'o', c='orange', ms=3)
    axs[1, 2].set_title(f'(F) selected graph: {len(v)} V {len(e)} E {len(res["rings"])} rings')
    for ax in axs.flat:
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def run_template(path, outdir, reference, reference_b):
    """Diagnostic for repeat scans: fit the reviewed big-image ring pattern."""
    from scipy.spatial import cKDTree
    os.makedirs(outdir, exist_ok=True)
    gray = img2mol.load_gray(path)
    flat = img2mol.flatten_bg(gray)
    b = img2mol.estimate_bond_scale(flat)
    R = -img2mol.ndi.gaussian_laplace(flat, max(0.8, b / 7.0))
    xy, score = img2mol.ring_candidates(flat, R, b)
    centers, support, params = img2mol.align_ring_template(reference, xy, score, b, reference_b, flat=flat)
    distance, near = cKDTree(xy).query(centers)
    good = (distance < 0.45 * b) & (score[near, 1] > 0.3)
    mask = img2mol.molecule_mask(flat, 0.4 * b)
    ix = np.rint(centers[:, 0]).astype(int)
    iy = np.rint(centers[:, 1]).astype(int)
    inside = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
    inside &= mask[np.clip(iy, 0, mask.shape[0] - 1), np.clip(ix, 0, mask.shape[1] - 1)]
    good &= inside
    fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(gray, cmap='gray')
    for i, (x, y) in enumerate(centers):
        ax.plot(x, y, 'o' if inside[i] else 'x', ms=8, mfc='none', mec='lime' if good[i] else 'orange' if inside[i] else 'red', mew=1.3)
        ax.text(x + 1, y - 1, str(i), color='yellow', fontsize=5)
    ax.set_title(f'{os.path.basename(path)}: {good.sum()}/{len(reference)} supported; orange=weak, red=outside')
    ax.set_axis_off(); fig.tight_layout()
    out = os.path.join(outdir, 'template_centers.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f'{os.path.basename(path)}: supported={good.sum()}/{len(reference)} weak-interior={(inside & ~good).sum()} outside={(~inside).sum()} scale={params[0]:.3f} angle={np.rad2deg(params[1]):.0f} mirror={params[3]}', flush=True)
    print(f'REVIEW: {out}', flush=True)
    return support


def plot_filters(path, outdir):
    """Radial x angular convolution filter bank and its response maps."""
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(path)
    flat = img2mol.flatten_bg(img2mol.load_gray(path))
    b = img2mol.estimate_bond_scale(flat)
    C = img2mol.center_response(flat, b)
    rads = (0.68, 0.90)
    ks = img2mol.sector_kernels(b, nsect=8, rads=rads)
    j3 = img2mol.junction_kernels(b, narm=3, nrot=12)
    E, M = img2mol.sector_enclosure(C, ks)
    V3 = img2mol.junction_response(C, j3)
    if name == '12.png':  # fftconvolve vs ndi.correlate parity (interior only)
        S_ref = np.stack([img2mol.ndi.correlate(C, ks[i, k], mode='reflect') for i in range(ks.shape[0]) for k in range(ks.shape[1])]).reshape(ks.shape[0], ks.shape[1], *C.shape)
        M_ref = S_ref.max(axis=0)
        E_ref = np.prod(np.clip(M_ref, 0, None), axis=0)**(1.0 / ks.shape[1])
        h = ks.shape[-1] // 2
        dif = np.abs(E[h:-h, h:-h] - E_ref[h:-h, h:-h])
        rel = dif.max() / max(np.abs(E_ref).max(), 1e-12)
        print(f'{name}: fftconvolve-vs-correlate interior parity max|dE|={dif.max():.3e} rel={rel:.3e}', flush=True)
    # ---- figure 1: the filter bank table + checksums ----
    nrad, nsect = ks.shape[0], ks.shape[1]
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, nsect + 2, width_ratios=[1] * nsect + [1.6, 1.6])
    for i in range(nrad):
        for k in range(nsect):
            ax = fig.add_subplot(gs[i, k])
            ax.imshow(ks[i, k], cmap='RdBu_r', vmin=-np.abs(ks).max(), vmax=np.abs(ks).max())
            ax.set_title(f'r={rads[i]}b {k * 360 // nsect}d', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, nsect])
    ths = np.linspace(-np.pi, np.pi, 720)
    dth = 2 * np.pi / nsect
    for k in range(nsect):
        u = np.abs((ths - k * dth + np.pi) % (2 * np.pi) - np.pi) / dth
        ax.plot(np.degrees(ths), np.where(u < 0.5, 0.75 - u**2, np.where(u < 1.5, 0.5 * (1.5 - u)**2, 0)), lw=1)
    tot = np.zeros(720)
    for k in range(nsect):
        u = np.abs((ths - k * dth + np.pi) % (2 * np.pi) - np.pi) / dth
        tot += np.where(u < 0.5, 0.75 - u**2, np.where(u < 1.5, 0.5 * (1.5 - u)**2, 0))
    ax.plot(np.degrees(ths), tot, 'k-', lw=2)
    ax.set_title('checksum: sum_j ang_j = 1', fontsize=9)
    ax = fig.add_subplot(gs[0, nsect + 1])
    ax.imshow(ks[0].sum(0), cmap='RdBu_r', vmin=-np.abs(ks).max(), vmax=np.abs(ks).max())
    ax.set_title('sum_j K_ij = isotropic wavelet', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, nsect])
    rr = np.linspace(0, 1.6, 400)
    hh = ks.shape[-1] // 2
    yy, xx = np.mgrid[-hh:hh + 1, -hh:hh + 1]
    for i, r0 in enumerate(rads):
        sig = 0.11 * b
        rr_ = np.hypot(xx, yy)
        pos = np.exp(-0.5 * ((rr_ - r0 * b) / sig)**2) * (np.abs(rr_ - r0 * b) < 2.5 * sig)
        neg = (rr_ < r0 * b + 4 * sig) & (np.abs(rr_ - r0 * b) >= 2.5 * sig)
        lam = (pos * rr_).sum() / max((neg.astype(float) * rr_).sum(), 1e-12)
        w = (pos - lam * neg)[:, hh]
        ax.plot(rr_[:, hh] / b, w, label=f'r0={r0}b')
    ax.axhline(0, c='k', lw=0.5); ax.legend(fontsize=8)
    ax.set_title('radial wavelet w(r)', fontsize=9); ax.set_xlabel('r/b')
    ax = fig.add_subplot(gs[1, nsect + 1])
    ax.imshow(ks[1].sum(0), cmap='RdBu_r', vmin=-np.abs(ks).max(), vmax=np.abs(ks).max())
    ax.set_title('sum_j K_1j', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{name}: ring-enclosure wavelet bank (b={b:.1f}px), L2-norm, zero-mean')
    out = os.path.join(outdir, 'filter_bank.png')
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f'REVIEW: {out}', flush=True)


    # ---- figure 2: response maps on the real image ----
    xy = img2mol.propose_centers(C, img2mol.molecule_mask(flat, 0.4 * b), b)
    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    axs[0, 0].imshow(flat, cmap='gray'); axs[0, 0].set_title(f'(A) {name} flattened')
    axs[0, 1].imshow(C, cmap='viridis'); axs[0, 1].set_title('(B) center response C')
    axs[0, 2].imshow(M[0], cmap='viridis'); axs[0, 2].set_title('(C) M[0deg]: radial-max sector')
    axs[0, 3].imshow(M[4], cmap='viridis'); axs[0, 3].set_title('(D) M[180deg]')
    axs[1, 0].imshow(M.min(0), cmap='viridis'); axs[1, 0].set_title('(E) min_j M_j (weakest sector)')
    axs[1, 1].imshow(E, cmap='viridis'); axs[1, 1].set_title('(F) E = fuzzy-AND sectors')
    axs[1, 2].imshow(V3, cmap='viridis'); axs[1, 2].set_title('(G) V3 junction (sp2 C)')
    axs[1, 3].imshow(E, cmap='viridis'); axs[1, 3].plot(xy[:, 0], xy[:, 1], 'rx', ms=5)
    axs[1, 3].set_title('(H) E + C-minima overlay')
    for ax in axs.flat:
        ax.set_axis_off()
    out = os.path.join(outdir, 'filter_response.png')
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f'REVIEW: {out}', flush=True)
    # ---- figure 3: ring centers from E peaks, with per-peak rejection reason ----
    import time
    t0 = time.time()
    acc, dg = img2mol.ring_centers_E(flat, b)
    dt = time.time() - t0
    import collections
    cnt = collections.Counter(dg['reason'])
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    axs[0].imshow(img2mol.load_gray(path), cmap='gray'); axs[0].set_title(f'(A) {name} gray')
    axs[1].imshow(E, cmap='viridis'); axs[1].set_title('(B) E map + ring centers')
    for ax in axs:
        for i in range(len(dg['xy'])):
            x, y = dg['xy'][i]
            if dg['reason'][i] == 'ok':
                ax.plot(x, y, 'o', ms=10, mfc='none', mec='lime', mew=1.5)
                ax.text(x + 1, y + 1, f"{dg['Rq10'][i]:.2f}", color='lime', fontsize=5)
            else:
                ax.plot(x, y, 'rx', ms=7)
                ax.text(x + 1, y + 1, f"{dg['reason'][i]}{dg['Rq10'][i]:.2f}", color='red', fontsize=6)
        ax.set_axis_off()
    fig.suptitle(f"{name}: {len(dg['xy'])} peaks, {dict(cnt)}, ring_centers_E {dt:.2f}s")
    out = os.path.join(outdir, 'ring_centers_E.png')
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f'{name}: ring_centers_E peaks={len(dg["xy"])} reasons={dict(cnt)} accepted={len(acc)} time={dt:.2f}s', flush=True)
    print(f'REVIEW: {out}', flush=True)


def plot_ve_kernels(path, outdir):
    """Vertex 'Y' and linear double-edge kernel banks, per-orientation
    responses and their products; kernels drawn in units of b with a molecule
    crop at the same pixel scale."""
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(path)
    flat = img2mol.flatten_bg(img2mol.load_gray(path))
    b = img2mol.estimate_bond_scale(flat)
    Rs = -img2mol.ndi.gaussian_laplace(flat, 0.2 * b)
    R = Rs / (np.sqrt(np.maximum(img2mol.ndi.uniform_filter(Rs * Rs, int(3 * b)), 0)) + 1e-9)
    f6n = img2mol.prep_image(img2mol.load_gray(path), 'flat6n')   # noise-stabilized high-pass
    kv = img2mol.vertex_kernels(b)
    ke = img2mol.edge_kernels(b)
    nrot, nd, dirs = kv.shape[0], ke.shape[0], ke.shape[1]
    Sv = np.stack([img2mol.signal.fftconvolve(R, k[::-1, ::-1], mode='same') for k in kv])   # (nrot,H,W) on LoG ridge
    Vr = Sv.max(axis=0)
    Svn = np.stack([img2mol.signal.fftconvolve(f6n, k[::-1, ::-1], mode='same') for k in kv])
    Vrn = Svn.max(axis=0)                                                                 # same on flat6n
    Se = np.stack([img2mol.signal.fftconvolve(R, ke[i, k][::-1, ::-1], mode='same')
                   for i in range(nd) for k in range(dirs)]).reshape(nd, dirs, *R.shape)   # edges on LoG ridge R
    Resp = np.clip(Se.max(axis=0), 0, None)                  # (dirs,H,W): per-orientation max over ds, positive part
    Estr = np.prod(Resp, axis=0)**(1.0 / dirs)               # product over orientations
    Sen = np.stack([img2mol.signal.fftconvolve(f6n, ke[i, k][::-1, ::-1], mode='same')
                    for i in range(nd) for k in range(dirs)]).reshape(nd, dirs, *f6n.shape)
    Estrn = np.prod(np.clip(Sen.max(axis=0), 0, None), axis=0)**(1.0 / dirs)
    fig = plt.figure(figsize=(19, 3.2 * (2 + nrot // 3)))
    ncol = max(kv.shape[0], dirs + 1) + 2
    gs = fig.add_gridspec(2 + nrot // 3, ncol, width_ratios=[1] * (ncol - 2) + [1.8, 1.8])
    # --- row 0: vertex kernels + same-scale molecule crop + V3 response ---
    for k in range(nrot):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(kv[k], cmap='RdBu_r', vmin=-np.abs(kv).max(), vmax=np.abs(kv).max(),
                  extent=np.array([-kv.shape[-1], kv.shape[-1], kv.shape[-1], -kv.shape[-1]]) / 2 / b)
        for rr in (0.5, 1.0):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, ec='k', lw=0.6, ls='--'))
        ax.set_title(f'V3-Y rot{k} (units of b)', fontsize=8); ax.set_aspect('equal')
    cy, cx = np.unravel_index(np.argmax(img2mol.center_response(flat, b) * -1), flat.shape)
    hw = int(1.5 * b)
    ax = fig.add_subplot(gs[0, -2])
    ax.imshow(flat[max(0, cy - hw):cy + hw, max(0, cx - hw):cx + hw], cmap='gray', extent=[-hw, hw, hw, -hw] / b)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec='r', lw=0.8, ls='--'))
    ax.set_title('molecule crop SAME SCALE', fontsize=8)
    ax = fig.add_subplot(gs[0, -1]); ax.imshow(Vr, cmap='viridis'); ax.set_title('V3-Y response (max rot, on R)', fontsize=8); ax.set_axis_off()
    # --- row 1: edge kernels (dirs cols, largest d) + strips product ---
    ke1 = ke[-1]
    for k in range(dirs):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(ke1[k], cmap='RdBu_r', vmin=-np.abs(ke1).max(), vmax=np.abs(ke1).max(),
                  extent=np.array([-ke1.shape[-1], ke1.shape[-1], ke1.shape[-1], -ke1.shape[-1]]) / 2 / b)
        for rr in (0.5, 1.0):
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, ec='k', lw=0.6, ls='--'))
        ax.set_title(f'edge d=1.0b dir{k} (units of b)', fontsize=8); ax.set_aspect('equal')
    ax = fig.add_subplot(gs[1, dirs]); ax.imshow(Estr, cmap='viridis'); ax.set_title('prod_k Resp_k on R (LoG ridge)', fontsize=8); ax.set_axis_off()
    ax = fig.add_subplot(gs[1, -2]); ax.imshow(Estrn, cmap='viridis'); ax.set_title('prod_k Resp_k on flat6n', fontsize=8); ax.set_axis_off()
    ax = fig.add_subplot(gs[1, -1]); ax.imshow(R, cmap='viridis'); ax.set_title('input R (LoG ridge, RMS-norm)', fontsize=8); ax.set_axis_off()
    # --- rows 2+: per-orientation edge responses + vertex responses on R vs flat6n ---
    for k in range(dirs):
        ax = fig.add_subplot(gs[2, k]); ax.imshow(Resp[k], cmap='viridis'); ax.set_title(f'Resp dir{k} = max_d edge^+ on R', fontsize=8); ax.set_axis_off()
    ax = fig.add_subplot(gs[2, -2]); ax.imshow(Vrn, cmap='viridis'); ax.set_title('V3-Y response on flat6n', fontsize=8); ax.set_axis_off()
    ax = fig.add_subplot(gs[2, -1]); ax.imshow(f6n, cmap='gray'); ax.set_title('input flat6n', fontsize=8); ax.set_axis_off()
    fig.suptitle(f'{name}: vertex-Y + linear double-edge filter banks, b={b:.1f}px')
    out = os.path.join(outdir, 'vertex_edge_kernels.png')
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f'REVIEW: {out}', flush=True)


def plot_filter_skeleton(path, outdir):
    """Review accepted ring centers and the connected shared-corner graph."""
    os.makedirs(outdir, exist_ok=True)
    gray = img2mol.load_gray(path)
    flat = img2mol.flatten_bg(gray)
    res = img2mol.ring_centers_graph(flat)
    fig, ax = plt.subplots(figsize=(8, 8)); ax.imshow(gray, cmap='gray')
    for ring in res['rings']:
        q = res['verts'][ring + [ring[0]]]
        ax.plot(q[:, 0], q[:, 1], color='lime', lw=1.0)
    ax.plot(res['centers'][:, 0], res['centers'][:, 1], 'o', mfc='none', mec='cyan', ms=5)
    n5 = sum(n == 5 for n in res['ns'].values()); n6 = sum(n == 6 for n in res['ns'].values())
    ax.set_title(f'{os.path.basename(path)} centers={len(res["centers"])} fused={len(res["pairs"])} rings(5:{n5},6:{n6}) V={len(res["verts"])} E={len(res["edges"])}')
    ax.set_axis_off(); fig.tight_layout()
    out = os.path.join(outdir, 'filter_skeleton.png'); fig.savefig(out, dpi=130); plt.close(fig)
    print(f'{os.path.basename(path)}: filter skeleton centers={len(res["centers"])} fused={len(res["pairs"])} V={len(res["verts"])} E={len(res["edges"])}', flush=True)
    if res['removed_pairs']:
        print(f'{os.path.basename(path)}: removed weak sides for valence={[(p[0], p[1], round(p[3], 3)) for p in res["removed_pairs"]]}', flush=True)
    print(f'REVIEW: {out}', flush=True)


def plot_preprocess_cmp(path, ref_path, outdir):
    """Shading-removal comparison: per method show flat / ridge R / center C /
    detections (lime circles) vs reference red dots, with match statistics."""
    from scipy.spatial import cKDTree
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(path)
    g = img2mol.load_gray(path)
    ref = img2mol.red_dot_centers(ref_path) if ref_path and os.path.isfile(ref_path) else None
    METHODS = ['flat16', 'flat6', 'flat6n', 'median', 'poly2', 'rows', 'bandpass', 'lstd']
    fig, axs = plt.subplots(len(METHODS), 4, figsize=(16, 2.6 * len(METHODS)))
    for i, method in enumerate(METHODS):
        flat = img2mol.prep_image(g, method)
        b = img2mol.estimate_bond_scale(flat)
        Rs = -img2mol.ndi.gaussian_laplace(flat, 0.2 * b)                      # same R as ring_centers_E
        R = Rs / (np.sqrt(img2mol.ndi.uniform_filter(Rs * Rs, int(3 * b))) + 1e-9)
        C = img2mol.center_response(flat, b)
        xy, _ = img2mol.ring_centers_E(flat, b)
        matched = FP = FN = 0
        if ref is not None and len(ref):
            dref = cKDTree(xy).query(ref)[0] if len(xy) else np.full(len(ref), np.inf)
            ddet = cKDTree(ref).query(xy)[0] if len(xy) else np.zeros(0)
            matched = int((dref < 0.6 * b).sum()); FP = int((ddet >= 0.6 * b).sum()); FN = int(len(ref) - matched)
            stat = f'matched={matched} FP={FP} FN={FN}'
        else:
            stat = 'no ref'
        axs[i, 0].imshow(flat, cmap='gray')
        axs[i, 1].imshow(R, cmap='gray')
        axs[i, 2].imshow(C, cmap='gray')
        axs[i, 3].imshow(g, cmap='gray')
        if ref is not None and len(ref):
            axs[i, 3].plot(ref[:, 0], ref[:, 1], 'r+', ms=8)
        if len(xy):
            axs[i, 3].plot(xy[:, 0], xy[:, 1], 'o', ms=10, mfc='none', mec='lime', mew=1.2)
        for j, t in enumerate(['flat', 'R ridge', 'C center', 'ref + detected']):
            axs[i, j].set_axis_off()
            if i == 0:
                axs[i, j].set_title(t, fontsize=9)
        axs[i, 0].set_ylabel(f'{method}\n{stat}', fontsize=9)
        print(f'{name} {method}: b={b:.2f} det={len(xy)} matched={matched} FP={FP} FN={FN}', flush=True)
    fig.suptitle(f'{name}: shading-removal comparison')
    fig.tight_layout()
    out = os.path.join(outdir, 'preprocess_cmp.png')
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f'REVIEW: {out}', flush=True)


def eval_vs_reference(names, img_dir, methods=('flat16',)):
    """Score ring_centers_E against solution/ red-dot reference centers."""
    from scipy.spatial import cKDTree
    soldir = os.path.join(img_dir, 'solution')
    for method in methods:
        TF = TN = TD = 0
        for n in names:
            ref = img2mol.red_dot_centers(os.path.join(soldir, n))
            flat = img2mol.prep_image(img2mol.load_gray(os.path.join(img_dir, n)), method)
            b = img2mol.estimate_bond_scale(flat)
            xy, _ = img2mol.ring_centers_E(flat, b)
            dref = cKDTree(xy).query(ref)[0] if len(xy) else np.full(len(ref), np.inf)
            ddet = cKDTree(ref).query(xy)[0] if len(ref) else np.ones(len(xy)) * np.inf
            hit = int((dref < 0.6 * b).sum()); fp = int((ddet >= 0.6 * b).sum()); fn = len(ref) - hit
            TF += fp; TN += fn; TD += len(xy)
            print(f'{n:8s} {method:8s} ref={len(ref):2d} det={len(xy):2d} hit={hit:2d} FP={fp} FN={fn} b={b:.2f}', flush=True)
        print(f'   {method:8s} TOTAL det={TD} FP={TF} FN={TN}', flush=True)


def report_scale_calibration():
    """Compare integer and parabolically interpolated autocorrelation scales."""
    sets = [('original', [os.path.join(IMGDIR, f'{i}.png') for i in range(1, 15)] + [os.path.join(IMGDIR, 'big.png')]),
            ('bonus', [os.path.join(BONUSDIR, f) for f in sorted(os.listdir(BONUSDIR), key=lambda f: int(os.path.splitext(f)[0])) if f.endswith('.png') and os.path.splitext(f)[0].isdigit()])]
    rows = []
    print('set/image           b_integer_px  b_subpixel_px  offset_px  change_pct', flush=True)
    for group, paths in sets:
        for path in paths:
            flat = img2mol.flatten_bg(img2mol.load_gray(path))
            b_int = img2mol.estimate_bond_scale(flat, subpixel=False)
            b_sub = img2mol.estimate_bond_scale(flat)
            rows.append((b_int, b_sub))
            print(f'{group}/{os.path.basename(path):14} {b_int:12.3f} {b_sub:14.3f} {b_sub-b_int:10.3f} {100*(b_sub/b_int-1):10.2f}', flush=True)
    offsets = np.asarray([b_sub - b_int for b_int, b_sub in rows])
    print(f'images={len(rows)} median|offset|={np.median(np.abs(offsets)):.3f}px max|offset|={np.max(np.abs(offsets)):.3f}px', flush=True)


def run(path, outdir, bpx=None, addH=False):
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(path)
    res = img2mol.img_to_graph(path, bond_px0=bpx)
    plot_result(res, os.path.join(outdir, 'stages.png'), name)
    plot_candidates(res, os.path.join(outdir, 'candidates.png'), name)
    plot_provisional_centers(res, os.path.join(outdir, 'provisional_centers.png'), name)
    plot_channels(res, os.path.join(outdir, 'channels.png'), name)
    ag = img2mol.graph_to_atomicgraph(res['verts'], res['edges'], res['bond_px'])
    nh = len(img2mol.add_hydrogens(ag)) if addH else 0
    import spammm.atomicUtils as au
    atoms, enames_l, apos, atypes, bonds, bl, rl = ag.to_arrays()
    au.saveXYZ(enames_l, apos, os.path.join(outdir, name.replace('.png', '.xyz')))
    n5 = sum(1 for x in res['ns'].values() if x == 5); n6 = sum(1 for x in res['ns'].values() if x == 6)
    print(f'{name}: V={len(res["verts"])} E={len(res["edges"])} rings={len(res["rings"])} (5:{n5} 6:{n6}) '
          f'pairs={len(res["pairs"])} bond_px={res["bond_px"]:.1f} H={nh}')
    print(f'REVIEW: {outdir}/stages.png')
    print(f'REVIEW: {outdir}/candidates.png')
    print(f'REVIEW: {outdir}/provisional_centers.png')
    print(f'REVIEW: {outdir}/channels.png')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default=None)
    ap.add_argument('--img-dir', default=IMGDIR, help='image directory for --all, --filters, and --filter-skeleton')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--bpx', type=float, default=None, help='initial bond length guess [px]')
    ap.add_argument('--H', action='store_true', help='add rim hydrogens')
    ap.add_argument('--template-big', action='store_true', help='diagnose alignment to the reviewed big.png ring pattern')
    ap.add_argument('--filters', action='store_true', help='plot the radial x angular filter bank and response maps')
    ap.add_argument('--ve-kernels', action='store_true', help='plot vertex (3-fold) and double-edge (paired-sector) kernel banks + responses')
    ap.add_argument('--filter-skeleton', action='store_true', help='plot accepted filter-bank centers as a shared-corner fused graph')
    ap.add_argument('--scale-calibration', action='store_true', help='compare integer and fractional autocorrelation scale estimates on reference image sets')
    ap.add_argument('--preprocess', action='store_true', help='compare shading-removal methods vs solution red-dot centers')
    ap.add_argument('--eval-ref', action='store_true', help='score ring_centers_E vs solution/ red-dot centers')
    ap.add_argument('--prep', default='flat16', help='prep_image method for --eval-ref (comma-separated)')
    args = ap.parse_args()
    names = [args.img] if args.img else \
        (sorted([f for f in os.listdir(args.img_dir) if f.endswith('.png') and os.path.splitext(f)[0].isdigit()], key=lambda f: int(os.path.splitext(f)[0]))
         if args.all and args.img_dir != IMGDIR else [f'{i}.png' for i in range(1, 15)] + ['big.png'] if args.all else ['1.png'])
    outroot = OUTDIR if args.img_dir == IMGDIR else os.path.join(OUTDIR, os.path.basename(os.path.normpath(args.img_dir)))
    if args.scale_calibration:
        report_scale_calibration()
        sys.exit(0)
    if args.eval_ref:
        eval_vs_reference(names, args.img_dir, args.prep.split(','))
        sys.exit(0)
    if args.preprocess:
        soldir = os.path.join(args.img_dir, 'solution')
        soldir = soldir if os.path.isdir(soldir) else None
        for n in names:
            ref_path = os.path.join(soldir, n) if soldir else None
            plot_preprocess_cmp(os.path.join(args.img_dir, n), ref_path, os.path.join(outroot, n.replace('.png', '')))
        sys.exit(0)
    if args.filters:
        for n in names:
            plot_filters(os.path.join(args.img_dir, n), os.path.join(outroot, n.replace('.png', '')))
        sys.exit(0)
    if args.ve_kernels:
        for n in names:
            plot_ve_kernels(os.path.join(args.img_dir, n), os.path.join(outroot, n.replace('.png', '')))
        sys.exit(0)
    if args.filter_skeleton:
        failed = []
        for n in names:
            try:
                plot_filter_skeleton(os.path.join(args.img_dir, n), os.path.join(outroot, n.replace('.png', '')))
            except ValueError as exc:
                print(f'{n}: filter skeleton rejected: {exc}', flush=True)
                failed.append(n)
        if failed:
            raise ValueError(f'filter skeleton rejected {len(failed)}/{len(names)} images: {failed}')
        sys.exit(0)
    if args.template_big:
        flat = img2mol.flatten_bg(img2mol.load_gray(os.path.join(args.img_dir, 'big.png')))
        reference_b = img2mol.estimate_bond_scale(flat)
        R = -img2mol.ndi.gaussian_laplace(flat, max(0.8, reference_b / 7.0))
        reference, score = img2mol.ring_candidates(flat, R, reference_b)
        reference = reference[(score[:, 1] > 0.7) & (score[:, 2] > 0.45)]
        if len(reference) != 19:
            raise ValueError(f'Expected 19 reviewed big.png ring centers, found {len(reference)}')
        for n in names:
            run_template(os.path.join(args.img_dir, n), os.path.join(outroot, n.replace('.png', '')), reference, reference_b)
        sys.exit(0)
    for n in names:
        run(os.path.join(args.img_dir, n), os.path.join(outroot, n.replace('.png', '')), bpx=args.bpx, addH=args.H)
