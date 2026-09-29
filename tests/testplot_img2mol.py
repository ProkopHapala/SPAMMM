#!/usr/bin/env python3
"""testplot_img2mol.py — visual demo: AFM/STM image -> carbon skeleton graph -> molecule.

Step 1 (spammm/img2mol.img_to_graph), rings-first with chemical priors:
  heavy smoothing -> -LoG ridge walls -> molecule mask -> distance-transform
  ring centers -> watershed ring regions -> harmonic r(th) fit (n in {5,6} + phase)
  -> fused-pair adjacency validated by wall brightness -> per-ring polygon
  proposals -> clustered vertices (=sp2 C) + ring edges.

Step 2: graph_to_atomicgraph -> AtomicGraph (all C, C-C=1.42 A) -> XYZ, optional rim H.

Usage:
  python3 tests/testplot_img2mol.py --img 1.png
  python3 tests/testplot_img2mol.py --all            # images 1-14 + big.png
  python3 tests/testplot_img2mol.py --img 1.png --bpx 8 --H

Artifacts -> debug/testplot_img2mol/<imgname>/
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from spammm import img2mol

IMGDIR = '/home/prokop/Desktop/PROJECTs/Svec_Ditriptaceneo_Helicene/Images'
OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'debug', 'testplot_img2mol')


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
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--bpx', type=float, default=None, help='initial bond length guess [px]')
    ap.add_argument('--H', action='store_true', help='add rim hydrogens')
    ap.add_argument('--template-big', action='store_true', help='diagnose alignment to the reviewed big.png ring pattern')
    ap.add_argument('--filters', action='store_true', help='plot the radial x angular filter bank and response maps')
    args = ap.parse_args()
    names = [args.img] if args.img else \
        ([f'{i}.png' for i in range(1, 15)] + ['big.png'] if args.all else ['1.png'])
    if args.filters:
        for n in names:
            plot_filters(os.path.join(IMGDIR, n), os.path.join(OUTDIR, n.replace('.png', '')))
        sys.exit(0)
    if args.template_big:
        flat = img2mol.flatten_bg(img2mol.load_gray(os.path.join(IMGDIR, 'big.png')))
        reference_b = img2mol.estimate_bond_scale(flat)
        R = -img2mol.ndi.gaussian_laplace(flat, max(0.8, reference_b / 7.0))
        reference, score = img2mol.ring_candidates(flat, R, reference_b)
        reference = reference[(score[:, 1] > 0.7) & (score[:, 2] > 0.45)]
        if len(reference) != 19:
            raise ValueError(f'Expected 19 reviewed big.png ring centers, found {len(reference)}')
        for n in names:
            run_template(os.path.join(IMGDIR, n), os.path.join(OUTDIR, n.replace('.png', '')), reference, reference_b)
        sys.exit(0)
    for n in names:
        run(os.path.join(IMGDIR, n), os.path.join(OUTDIR, n.replace('.png', '')), bpx=args.bpx, addH=args.H)
