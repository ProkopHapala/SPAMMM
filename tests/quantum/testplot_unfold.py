#!/usr/bin/env python
"""Band-structure & supercell-unfolding pipeline for edge-passivated zigzag ribbons.

Ribbons: MoleculeEditorBackend.build_zigzag_ribbon via build_acene()
(width_chains: w4=1 ring row, w6=2, w8=3, w10=4, w12=5; per-site passivation
lists select the edge chemistry per cell site — see PASSIVATION_GROUPS).
All k in ABSOLUTE units [1/A] on the FULL primitive BZ (Gamma-centred mesh,
k_shift=(0,0,0) so BZ edges are sampled).

Modes:
  (default)   canonical folding validation: x1/x2/x4 cells of pristine CH acene;
              replica overlay (x2 x2 copies, x4 x4 copies), direct extended-zone
              x4 run (klist), fold-down eigenvalue check (0.1 meV), unfolded
              weight dots.  --perturb / --rdistort break the symmetry.
  --passiv L  x1 bands of comma-list passivations, chemistry colors (C=g,N=b,O=r), +-4 eV
  --switch    per system C/N/O: aromatic ref (black) vs hydrogenated (red), one fig each
  --sys3      relaxed ref x1 + one-site-hydrogenated (both edges) x2/x4 + weights

Canonical overlay layering (DO NOT regress): x4 lw 0.25 back, x2 lw 0.5,
x1 lw 2.0 on top; BZ edges in each spectrum's color; weight dots filled black,
size prop. W, behind all lines, duplicated at +-q (inversion-reduced output).

Usage:  python tests/quantum/testplot_unfold.py [--nkx 64] [--width 4]
        [--passiv "CH,N,C-OH"] [--switch] [--sys3] [--perturb A] [--rdistort A]
Output: debug/unfold/  (test_unfold_acene.png, test_unfold_passiv_w*_*,
        test_unfold_switch_w*_*, test_unfold_sys{C,N,O}.png, index.html gallery)
Docs:   doc/TopicalAudit/BandUnfolding_Ribbons.md
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'unfold')


def build_acene(width_chains, ncells, passivation='CH'):
    """Plain periodic polyacene-ish ribbon: edge passivation (str or per-site list), periodic x, vacuum y/z."""
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    from spammm.topology.ribbon_pbc import A_CC
    b = MoleculeEditorBackend(a_CC=A_CC)
    b.build_zigzag_ribbon(width_chains=width_chains, length_cells=ncells,
                          passivation_bottom=passivation, passivation_top=passivation, bPeriodicX=True)
    b._sync_sys()
    apos = np.asarray(b.sys.apos, float).copy()
    enames = list(b.sys.enames)
    Lx = ncells * 2.0 * A_CC * np.cos(np.pi / 6.0)
    h = apos[:, 1].ptp()
    apos[:, 1] -= apos[:, 1].min() - 6.0
    lvs = np.array([[Lx, 0.0, 0.0], [0.0, h + 12.0, 0.0], [0.0, 0.0, 12.0]])
    return apos, enames, lvs


def run_sp(apos, enames, lvs, nkx=None, wd=None, klist=None, eigenvecs=True,
           mixer='DIIS { Generations = 8 }'):
    """1D k SP -> (kw, eigs, fills, C|None, E).  klist overrides the MP mesh."""
    from spammm.quantum.DFTB_utils import run_pbc
    from spammm.quantum.pi_bond_order import read_band_out, read_eigenvec_bin
    extra = 'Analysis { WriteEigenvectors = Yes }' if eigenvecs else ''
    E, _, _ = run_pbc(apos, enames, lvs, nk=(nkx or 1, 1, 1), k_shift=(0.0, 0.0, 0.0),
                      klist=klist, workdir=wd, Temperature=300,
                      Mixer=mixer, MaxScc=400, extra_hsd=extra)
    kw, eigs, fills = read_band_out(os.path.join(wd, 'band.out'))
    C = read_eigenvec_bin(os.path.join(wd, 'eigenvec.bin'), len(kw), eigs.shape[1]) if eigenvecs else None
    return kw, eigs, fills, C, E


def run_passiv_compare(args, elements, plt, Rectangle):
    """x1 polyacene ribbons with different edge passivations: bands overlay + gaps."""
    pss = args.passiv.split(',')
    fig = plt.figure(figsize=(16, 10))
    res = {}
    for ip, ps in enumerate(pss):
        apos, en, lvs = build_acene(args.width, 1, passivation=ps)
        kw, eigs, fills, C, E = run_sp(apos, en, lvs, args.nkx, os.path.join(OUTDIR, f'pass_{ps.replace("=", "").replace("-", "_")}'))
        G = 2 * np.pi / lvs[0, 0]
        kabs = np.arange(len(kw)) / args.nkx * G
        occ = fills > 1.9
        H = eigs[occ].max()
        lumo = np.where(occ, +1e9, eigs).min(1)            # per-k LUMO
        homo = np.where(occ, eigs, -1e9).max(1)            # per-k HOMO
        gap = lumo.min() - H; dgap = (lumo - homo).min()
        res[ps] = dict(apos=apos, en=en, lvs=lvs, kabs=kabs, eigs=eigs, H=H, gap=gap, dgap=dgap, E=E)
        print(f'{ps:5s}: {len(apos)} atoms, {eigs.shape[1]} bands, E/at={E*27.2114/len(apos):+.4f} eV, '
              f'HOMO={H:+.3f} eV, gap={gap*1000:.0f} meV (direct {dgap*1000:.0f})')
    # geometries — cell rectangle in the same color as that system's band lines
    elcol = {'O': 'r', 'N': 'b', 'C': 'g'}
    for ip, ps in enumerate(pss):
        c = res[ps]
        col = 'r' if 'O' in ps else ('b' if 'N' in ps else 'g')
        axi = fig.add_subplot(2, len(pss), ip + 1)
        apos, en, lvs = c['apos'], c['en'], c['lvs']
        bonds = np.asarray([[i, j] for i in range(len(apos)) for j in range(i + 1, len(apos))])
        dd = apos[bonds[:, 1]] - apos[bonds[:, 0]]
        dd[:, 0] -= np.round(dd[:, 0] / lvs[0, 0]) * lvs[0, 0]
        ok = np.linalg.norm(dd, axis=1) < 1.9
        for (i, j), dvec in zip(bonds[ok], dd[ok]):
            axi.plot([apos[i, 0], apos[i, 0] + dvec[0]], [apos[i, 1], apos[i, 1] + dvec[1]], '-', c='0.55', lw=1.0)
        axi.scatter(apos[:, 0], apos[:, 1], c=[elements.ELEMENT_DICT[e][8] for e in en],
                    s=[elements.ELEMENT_DICT[e][6] * 20 for e in en], linewidths=0)
        axi.add_patch(Rectangle((0, apos[:, 1].min() - 1), lvs[0, 0], apos[:, 1].ptp() + 2, fill=False, edgecolor=col, lw=2.0))
        axi.set_aspect('equal'); axi.axis('off'); axi.set_title(f'{ps}: {len(apos)} atoms')
    # bands overlay (absolute kx, full primitive BZ); color by chemistry: O=red, N=blue, C=green
    ax = fig.add_subplot(2, 1, 2)
    for ip, ps in enumerate(pss):
        c = res[ps]
        col = 'r' if 'O' in ps else ('b' if 'N' in ps else 'g')   # O-systems red, N blue, C green
        kf, ef = mirror(c['kabs'], c['eigs'])
        srt = np.argsort(kf)
        ax.plot(kf[srt], ef[srt] - c['H'], '-', lw=1.0, color=col)
        ax.plot([], [], '-', lw=1.5, color=col,
                label=f'{ps}: {c["eigs"].shape[1]} bands, gap {c["gap"]*1000:.0f} meV')
    ax.axhline(0, color='k', ls='--', lw=0.7)
    ax.set(xlim=(-res[pss[0]]['kabs'][-1], res[pss[0]]['kabs'][-1]),
           ylim=(-4, 4), xlabel='kx [1/A]', ylabel='E - E_HOMO [eV]',
           title=f'polyacene x1 edge-passivation comparison (nkx={args.nkx})')
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    png = os.path.join(OUTDIR, f'test_unfold_passiv_w{args.width}_{"-".join(pss)}.png')
    fig.savefig(png, dpi=140)
    print(f'REVIEW: {png}')


SWITCH = {'C': ('CH', 'CH2'), 'N': ('N', 'NH'), 'O': ('C-OH', 'C=O')}   # ref aromatic -> hydrogenated, x1 comparison


def run_switch(args, elements, plt, Rectangle):
    """Per-system x1 switch: aromatic ref (solid) vs hydrogenated (dashed), chemistry color."""
    elcol = {'C': 'g', 'N': 'b', 'O': 'r'}
    for sy, (ref, hyd) in SWITCH.items():
        fig = plt.figure(figsize=(14, 8))
        res = {}
        for ps in (ref, hyd):
            apos, en, lvs = build_acene(args.width, 1, passivation=ps)
            kw, eigs, fills, C, E = run_sp(apos, en, lvs, args.nkx, os.path.join(OUTDIR, f'sw{args.width}_{ps.replace("=", "").replace("-", "_")}'))
            G = 2 * np.pi / lvs[0, 0]
            kabs = np.arange(len(kw)) / args.nkx * G
            occ = fills > 1.9
            H = eigs[occ].max()
            lumo = np.where(occ, +1e9, eigs).min(1)
            homo = np.where(occ, eigs, -1e9).max(1)
            gap = lumo.min() - H; dgap = (lumo - homo).min()
            res[ps] = dict(apos=apos, en=en, lvs=lvs, kabs=kabs, eigs=eigs, H=H, gap=gap, dgap=dgap)
            print(f'{sy} {ps:5s}: {len(apos)} atoms, {eigs.shape[1]} bands, HOMO={H:+.3f} eV, gap={gap*1000:.0f} meV (direct {dgap*1000:.0f})')
        for ip, (ps, pcol) in enumerate(((ref, 'k'), (hyd, 'r'))):
            c = res[ps]
            axi = fig.add_subplot(2, 2, ip + 1)
            apos, en, lvs = c['apos'], c['en'], c['lvs']
            bonds = np.asarray([[i, j] for i in range(len(apos)) for j in range(i + 1, len(apos))])
            dd = apos[bonds[:, 1]] - apos[bonds[:, 0]]
            dd[:, 0] -= np.round(dd[:, 0] / lvs[0, 0]) * lvs[0, 0]
            ok = np.linalg.norm(dd, axis=1) < 1.9
            for (i, j), dvec in zip(bonds[ok], dd[ok]):
                axi.plot([apos[i, 0], apos[i, 0] + dvec[0]], [apos[i, 1], apos[i, 1] + dvec[1]], '-', c='0.55', lw=1.0)
            axi.scatter(apos[:, 0], apos[:, 1], c=[elements.ELEMENT_DICT[e][8] for e in en],
                        s=[elements.ELEMENT_DICT[e][6] * 20 for e in en], linewidths=0)
            axi.add_patch(Rectangle((0, apos[:, 1].min() - 1), lvs[0, 0], apos[:, 1].ptp() + 2, fill=False, edgecolor=pcol, lw=2.0))
            axi.set_aspect('equal'); axi.axis('off'); axi.set_title(f'{ps}: {len(apos)} atoms')
        ax = fig.add_subplot(2, 1, 2)
        for ps, col, tag in ((ref, 'k', 'ref'), (hyd, 'r', 'hydrogenated')):
            c = res[ps]
            kf, ef = mirror(c['kabs'], c['eigs'])
            srt = np.argsort(kf)
            ax.plot(kf[srt], ef[srt] - c['H'], '-', lw=1.2, color=col)
            ax.plot([], [], '-', lw=1.8, color=col, label=f'{ps} ({tag}): {c["eigs"].shape[1]} bands, gap {c["gap"]*1000:.0f} meV')
        ax.axhline(0, color='k', ls='--', lw=0.7)
        ax.set(xlim=(-res[ref]['kabs'][-1], res[ref]['kabs'][-1]), ylim=(-4, 4),
               xlabel='kx [1/A]', ylabel='E - E_HOMO [eV]',
               title=f'{sy}-edge switch w{args.width} x1: {ref} (black) vs {hyd} (red), nkx={args.nkx}')
        ax.legend(fontsize=9, loc='lower left')
        fig.tight_layout()
        png = os.path.join(OUTDIR, f'test_unfold_switch_w{args.width}_{sy}_{ref}-vs-{hyd}.png')
        fig.savefig(png, dpi=140)
        print(f'REVIEW: {png}')


SYS3 = {'C': ('CH', 'CH2'), 'N': ('N', 'NH'), 'O': ('C-OH', 'CHOH')}   # ref aromatic -> hydrogenated


def run_sys3(args, elements, plt, Rectangle):
    """C/N/O edge systems: relaxed aromatic-ref x1 vs one-site-hydrogenated (both edges)
    x2/x4 — folded-band overlay + unfolded weights (canonical format)."""
    from spammm.quantum.DFTB_utils import run_pbc
    from spammm.quantum.pi_bond_order import subcell_group_indices, unfold_spectral_weights_DEPRECATED

    def relax_bands(apos, en, lvs, nkx, tag, eig=True):
        E_r, apos_r, _ = run_pbc(apos, en, lvs, nk=(8, 1, 1), do_relax=True,
                                 workdir=os.path.join(OUTDIR, f'{tag}_relax'),
                                 Temperature=300, Mixer='DIIS { Generations = 8 }', MaxScc=400)
        kw, eigs, fills, C, E = run_sp(apos_r, en, lvs, nkx, os.path.join(OUTDIR, f'{tag}_bands'),
                                     eigenvecs=eig, mixer=None)   # Broyden: DIIS goes singular on Gamma-mesh metals
        G = 2 * np.pi / lvs[0, 0]
        kabs = np.arange(len(kw)) / nkx * G
        return dict(apos=apos_r, apos_id=apos, en=en, lvs=lvs, eigs=eigs, fills=fills, C=C,
                    E=E, E_r=E_r, G=G, kabs=kabs, nk=nkx)

    def geom_panel(ax, c, title):
        """Geometry + heavy-atom bonds colored by length (seismic_r, 1.30-1.55 A)."""
        import matplotlib.colors as mcolors
        apos, en, lvs = c['apos'], c['en'], c['lvs']
        bonds = np.asarray([[i, j] for i in range(len(apos)) for j in range(i + 1, len(apos))])
        dd = apos[bonds[:, 1]] - apos[bonds[:, 0]]
        dd[:, 0] -= np.round(dd[:, 0] / lvs[0, 0]) * lvs[0, 0]
        bl = np.linalg.norm(dd, axis=1)
        ok = bl < 1.9
        bonds, dd, bl = bonds[ok], dd[ok], bl[ok]
        enh = np.array([e.split('_')[0] for e in en])
        heavy = (enh[bonds[:, 0]] != 'H') & (enh[bonds[:, 1]] != 'H')
        norm = mcolors.Normalize(vmin=1.30, vmax=1.55)
        cmap = plt.get_cmap('seismic_r')
        for (i, j), dvec, L, hv in zip(bonds, dd, bl, heavy):
            col = cmap(norm(min(max(L, 1.30), 1.55))) if hv else '0.7'
            ax.plot([apos[i, 0], apos[i, 0] + dvec[0]], [apos[i, 1], apos[i, 1] + dvec[1]],
                    '-', c=col, lw=2.2 if hv else 0.8)
        ax.scatter(apos[:, 0], apos[:, 1], c=[elements.ELEMENT_DICT[e][8] for e in en],
                   s=[elements.ELEMENT_DICT[e][6] * 20 for e in en], linewidths=0)
        for i, (e, p) in enumerate(zip(enh, apos)):
            ax.text(p[0], p[1] + 0.28, f'{i}:{e}', fontsize=5, ha='center', va='center', zorder=6)
        ax.add_patch(Rectangle((0, apos[:, 1].min() - 1), lvs[0, 0], apos[:, 1].ptp() + 2, fill=False, edgecolor='magenta', lw=1.5))
        ax.set_aspect('equal'); ax.axis('off')
        bh = bl[heavy]
        ax.set_title(f'{title}\nC-X bonds: {bh.min():.3f}..{bh.max():.3f} A (spread {1000*(bh.max()-bh.min()):.0f} mA)', fontsize=8)

    # pristine all-carbon acene reference (same for all 3 systems)
    aposC, enC, lvsC = build_acene(args.width, 1, 'CH')
    cCH = relax_bands(aposC, enC, lvsC, args.nkx, 'sysAcene_ref')
    occC = cCH['fills'] > 1.9
    HC = cCH['eigs'][occC].max()
    kfC, efC = mirror(cCH['kabs'], cCH['eigs'])

    for sname, (ref, hyd) in SYS3.items():
        apos1, en1, lvs1 = build_acene(args.width, 1, ref)
        c1 = relax_bands(apos1, en1, lvs1, args.nkx, f'sys{sname}_ref')
        occ1 = c1['fills'] > 1.9
        H1 = c1['eigs'][occ1].max()
        lumo = np.where(occ1, 1e9, c1['eigs']).min(1)
        print(f'== {sname}: ref={ref} hyd={hyd}; ref x1: {len(en1)} atoms, HOMO={H1:+.3f} eV, gap={1000*(lumo.min()-H1):.0f} meV')
        cs = {1: c1}
        for n in [2, args.ncells]:
            plist = [hyd] + [ref] * (n - 1)                    # site 0 hydrogenated, both edges
            apos, en, lvs = build_acene(args.width, n, plist)
            cs[n] = relax_bands(apos, en, lvs, args.nkx // n, f'sys{sname}_x{n}h')
        fig = plt.figure(figsize=(16, 10))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 6, figure=fig)
        geom_panel(fig.add_subplot(gs[0, 0:2]), c1, f'{sname} ref x1 ({ref}), {len(en1)} atoms')
        for ip, n in enumerate([2, args.ncells]):
            geom_panel(fig.add_subplot(gs[0, 2 * (ip + 1):2 * (ip + 2)]), cs[n],
                       f'{sname} x{n} 1x{hyd} relaxed, {len(cs[n]["en"])} atoms')
        for ip, n in enumerate([2, args.ncells]):
            cn = cs[n]
            IDX = subcell_group_indices(cn['apos_id'], cn['lvs'], cn['en'], n, strict=False)
            q, W = unfold_spectral_weights_DEPRECATED(cn['C'], cn['kabs'] / cn['G'], IDX, n)
            sharp = (W.max(1) > 0.9).mean() * 100
            print(f'   x{n} 1H({hyd}): {len(cn["en"])} atoms, E/at={cn["E"]*27.2114/len(cn["en"]):+.4f} eV, sharp={sharp:.1f}%')
            # -- canonical overlay: ref x1 red on top, hyd xN thin black replicas, weight dots
            ax = fig.add_subplot(gs[1, 3 * ip:3 * (ip + 1)])
            qabs = q * c1['G']
            qabs = np.where(qabs > c1['G'] / 2, qabs - c1['G'], qabs)
            qS = np.tile(qabs[:, :, None], (1, 1, cn['eigs'].shape[1]))
            ES = np.tile(cn['eigs'][:, None, :], (1, n, 1)) - H1
            sel = (ES > -8) & (ES < 8) & (W > 0.02)
            ax.scatter(np.r_[qS[sel], -qS[sel]], np.r_[ES[sel], ES[sel]],
                       s=np.r_[W[sel], W[sel]] * 60, c='k', lw=0, zorder=0)
            for m in range(-n // 2, n // 2 + 1):
                kf, ef = mirror(cn['kabs'], cn['eigs'])
                srt = np.argsort(kf)
                ax.plot(kf[srt] + m * cn['G'], ef[srt] - H1, '-', lw=0.25, color='k', zorder=1)
            kf1, ef1 = mirror(c1['kabs'], c1['eigs'])
            srtC = np.argsort(kfC)
            ax.plot(kfC[srtC], efC[srtC] - HC, '-', lw=1.0, color='tab:blue', zorder=4)
            ax.plot(np.sort(kf1), ef1[np.argsort(kf1)] - H1, '-', lw=2.0, color='r', zorder=5)
            ax.plot([], [], '-', lw=1.0, color='tab:blue', label='pristine acene (CH) x1')
            ax.plot([], [], '-', lw=2.0, color='r', label=f'{ref} x1 (ref)')
            ax.plot([], [], '-', lw=0.6, color='k', label=f'x{n} 1x{hyd} folded')
            ax.plot([], [], 'o', ms=5, color='k', label='unfold W (size)')
            ax.legend(fontsize=7, loc='lower left')
            for e, col in ((c1['G'] / 2, 'r'), (cn['G'] / 2, 'k')):
                ax.axvline(+e, color=col, lw=1.0, ls='--'); ax.axvline(-e, color=col, lw=1.0, ls='--')
            ax.axhline(0, color='b', ls='--', lw=0.7)
            ax.set(xlim=(-c1['G'] / 2, c1['G'] / 2), ylim=(-8, 8), xlabel='kx [1/A]',
                   ylabel='E - E_HOMO(ref) [eV]',
                   title=f'{sname} x{n}: 1x {hyd} site relaxed; black=x{n} bands (n periods), red={ref} x1, dots=unfold W; sharp {sharp:.0f}%')
        fig.tight_layout()
        png = os.path.join(OUTDIR, f'test_unfold_sys{sname}.png')
        fig.savefig(png, dpi=140)
        print(f'REVIEW: {png}')


def mirror(kabs, eigs):
    """Extend reduced [0, BZ/2] data to the full BZ by inversion E(-k)=E(k)."""
    kf = np.concatenate([-kabs[1:][::-1], kabs])
    ef = np.concatenate([eigs[1:][::-1], eigs])
    return kf, ef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=4, help='width_chains (4=polyacene)')
    ap.add_argument('--ncells', type=int, default=4, help='largest supercell multiplier along x')
    ap.add_argument('--nkx', type=int, default=64, help='k-mesh along x of the LARGEST cell')
    ap.add_argument('--perturb', type=float, default=0.0, help='pull one edge H of the xN cell this far [A] along C-H (breaks supercell->primitive symmetry)')
    ap.add_argument('--rdistort', type=float, default=0.0, help='random displacement [A] applied to one carbon of the xN cell')
    ap.add_argument('--passiv', type=str, default='', help="comma list of edge passivations (e.g. CH,N,NH,CH2,'C=O','C-OH') -> compare x1 bands instead of unfolding")
    ap.add_argument('--sys3', action='store_true', help='C/N/O systems: relaxed aromatic-ref x1 vs 1-site-hydrogenated x2/x4 + unfolded weights')
    ap.add_argument('--switch', action='store_true', help='per-system x1 switch: aromatic ref (solid) vs hydrogenated (dashed), one fig per C/N/O')
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    from spammm.quantum.pi_bond_order import subcell_group_indices, unfold_spectral_weights_DEPRECATED
    from spammm import elements
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if args.passiv:
        run_passiv_compare(args, elements, plt, Rectangle)
        return
    if args.sys3:
        run_sys3(args, elements, plt, Rectangle)
        return
    if args.switch:
        run_switch(args, elements, plt, Rectangle)
        return

    N = args.ncells
    cells = {}
    for n in [1, 2, N]:
        apos, en, lvs = build_acene(args.width, n)
        apos_id = apos.copy()
        if n == N and args.rdistort > 0:                     # random displacement of ONE carbon
            rng = np.random.default_rng(0)
            iC = int(np.where(np.array(en) == 'C')[0][0])
            v = rng.normal(size=3)
            apos[iC] += args.rdistort * v / np.linalg.norm(v)
            print(f'x{N}: random {args.rdistort} A displacement of C{iC}')
        if n == N and args.perturb > 0:                      # pull one edge H away from its C
            enarr = np.array(en)
            iH = int(np.argmax(np.where(enarr == 'H', apos[:, 1], -1e9)))
            d = apos - apos[iH]
            d[:, 0] -= np.round(d[:, 0] / lvs[0, 0]) * lvs[0, 0]
            iC = int(np.argmin(np.where(enarr == 'C', np.linalg.norm(d, axis=1), 1e9)))
            v = apos[iH] - apos[iC]
            apos[iH] += args.perturb * v / np.linalg.norm(v)
            print(f'x{N}: pulled H{iH} (bonded to C{iC}) by {args.perturb} A along C-H')
        nkx = args.nkx // n                                  # same k-density per atom
        kw, eigs, fills, C, E = run_sp(apos, en, lvs, nkx, os.path.join(OUTDIR, f'acene_x{n}'))
        G = 2 * np.pi / lvs[0, 0]                            # reciprocal lattice |G| [1/A]
        assert len(kw) == nkx // 2 + 1, f'x{n}: expected {nkx//2+1} reduced kpts, got {len(kw)}'
        kabs = np.arange(len(kw)) / nkx * G                  # Gamma-centred, inversion-reduced -> 0..G/2
        assert abs(kabs[-1] - G / 2) < 1e-9, 'BZ edge not sampled'
        cells[n] = dict(apos=apos, apos_id=apos_id, en=en, lvs=lvs, eigs=eigs, fills=fills, C=C, E=E, G=G, kabs=kabs, nk=nkx)
        print(f'x{n}: {len(apos)} atoms, Lx={lvs[0,0]:.3f} A, nk={nkx}->{len(kw)} (incl. BZ edge), E/at={E * 27.2114 / len(apos):.6f} eV')

    # --- direct extended-zone run: x4 cell at kf in [-2,+2] = full x1 BZ ------
    c4 = cells[N]
    nk4 = c4['nk']
    fext = np.arange(-2 * nk4, 2 * nk4 + 1) / nk4            # step 1/nk4 over [-2,+2]
    klist = [((f, 0.0, 0.0), 1.0 / len(fext)) for f in fext]
    kw_x, eigs_x, _, _, E_x = run_sp(c4['apos'], c4['en'], c4['lvs'], klist=klist,
                                   wd=os.path.join(OUTDIR, 'acene_x4_ext'), eigenvecs=False)
    assert len(kw_x) == len(fext), f'extended run: expected {len(fext)} kpts, got {len(kw_x)}'
    kext = fext * c4['G']                                    # absolute kx [1/A]
    print(f'x{N} extended-zone run: {len(kw_x)} kpts over kf [-2,+2], E/at={E_x * 27.2114 / len(c4["apos"]):.6f} eV')

    # extended run must reproduce the reduced-BZ eigenvalues (fold + sort)
    kred = np.abs(((kext / c4['G']) % 1.0 + 0.5) % 1.0 - 0.5)  # |kf| mod 1 -> [0,.5]
    dev = []
    for ik in range(len(kext)):
        j = np.argmin(np.abs(c4['kabs'] / c4['G'] - kred[ik]))
        dev.append(np.abs(np.sort(eigs_x[ik]) - np.sort(c4['eigs'][j])).max())
    print(f'extended-zone vs folded x{N}: max dev = {max(dev) * 1000:.2f} meV')

    Gp = cells[1]['G']
    H1 = cells[1]['eigs'][cells[1]['fills'] > 1.9].max()     # common reference: x1 HOMO

    # --- eigenvector unfold of xN -> primitive channels (needed for the plot) --
    c4, c1 = cells[N], cells[1]
    IDX = subcell_group_indices(c4['apos_id'], c4['lvs'], c4['en'], N)   # grouping on ideal lattice
    q, W = unfold_spectral_weights_DEPRECATED(c4['C'], c4['kabs'] / c4['G'], IDX, N)
    print(f'unfold x{N}->x1: sum_m w per eig = {W.sum(1).min():.3f}..{W.sum(1).max():.3f}; '
          f'sharp (w>0.9): {(W.max(1) > 0.9).mean() * 100:.1f}%')
    qabs = q * c1['G']                                     # primitive frac -> 1/A, [0,G1)
    qabs = np.where(qabs > c1['G'] / 2, qabs - c1['G'], qabs)   # wrap to [-G1/2, G1/2]
    qS = np.tile(qabs[:, :, None], (1, 1, c4['eigs'].shape[1]))  # [k,m,band]
    ES = np.tile(c4['eigs'][:, None, :], (1, N, 1)) - H1

    fig = plt.figure(figsize=(16, 10))
    # --- top row: geometries -------------------------------------------------
    for ip, n in enumerate([1, 2, N]):
        ax = fig.add_subplot(2, 3, ip + 1)
        c = cells[n]
        apos, en, lvs = c['apos'], c['en'], c['lvs']
        bonds = np.asarray([[i, j] for i in range(len(apos)) for j in range(i + 1, len(apos))])
        dd = apos[bonds[:, 1]] - apos[bonds[:, 0]]
        Lx = lvs[0, 0]
        dd[:, 0] -= np.round(dd[:, 0] / Lx) * Lx
        ok = np.linalg.norm(dd, axis=1) < 1.9
        colors = [elements.ELEMENT_DICT[e][8] for e in en]
        sizes = [elements.ELEMENT_DICT[e][6] * 20 for e in en]
        for (i, j), dvec in zip(bonds[ok], dd[ok]):
            ax.plot([apos[i, 0], apos[i, 0] + dvec[0]], [apos[i, 1], apos[i, 1] + dvec[1]], '-', c='0.55', lw=1.0)
        ax.scatter(apos[:, 0], apos[:, 1], c=colors, s=sizes, linewidths=0)
        ax.add_patch(Rectangle((0, apos[:, 1].min() - 1), Lx, apos[:, 1].ptp() + 2, fill=False, edgecolor='magenta', lw=1.5))
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(f'acene x{n}: {len(apos)} atoms, Lx={Lx:.2f} A')

    # --- bottom: full primitive BZ overlay -----------------------------------
    # x1 bands (black) + x2 replicated 2x (green) + x4 replicated 4x (red)
    # + x4 computed directly on the extended k-range (cyan) — must overlay red.
    ax = fig.add_subplot(2, 1, 2)
    reps = {1: [0], 2: [-1, 0, 1], N: list(range(-N // 2, N // 2 + 1))}
    sel = (ES > -8) & (ES < 8) & (W > 0.02)
    ax.scatter(np.r_[qS[sel], -qS[sel]], np.r_[ES[sel], ES[sel]], s=np.r_[W[sel], W[sel]] * 60,
               c='k', lw=0, zorder=0, label=f'x{N} unfolded weight (size=W)')
    for n, col, lw in [(N, 'k', 0.25), (2, 'tab:green', 0.5), (1, 'r', 2.0)]:
        c = cells[n]
        kf, ef = mirror(c['kabs'], c['eigs'])
        srt = np.argsort(kf)
        for m in reps[n]:
            ax.plot(kf[srt] + m * c['G'], ef[srt] - H1, '-', lw=lw, color=col)
        if n == N:
            ax.plot(kext, eigs_x - H1, '-', lw=0.35, color='cyan')
        for e in (-c['G'] / 2, c['G'] / 2):
            ax.axvline(e, color=col, lw=1.0, ls='--')
        ax.plot([], [], '-', color=col,
                label=f'x{n}: {c["eigs"].shape[1]} bands, period G={c["G"]:.3f} 1/A = {n}x per prim.BZ, BZ edge {c["G"]/2:.3f}')
    ax.plot([], [], '-', lw=0.8, color='cyan', label=f'x{N} direct extended-zone run')
    ax.axhline(0, color='b', ls='--', lw=0.7)
    ax.set(xlim=(-Gp / 2, Gp / 2), ylim=(-8, 8), xlabel='kx [1/A]', ylabel='E - E_HOMO [eV]',
           title='full primitive BZ: x1 bands (black); x2,x4 replicated by m*G (green,red); cyan = x4 computed directly at extended k')
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    png = os.path.join(OUTDIR, 'test_unfold_acene.png')
    fig.savefig(png, dpi=140)
    print(f'REVIEW: {png}')

    # --- numeric fold-down check: every x_n eig must equal some x4 eig at the
    #     mapped supercell k  (f_N = (N/n)*f_n mod 1) -----------------------------
    def fold_f(kf, factor):
        f = (kf * factor) % 1.0
        return np.where(f > 0.5, 1 - f, f)
    kf4 = cells[N]['kabs'] / cells[N]['G']
    e4s = cells[N]['eigs']
    for n in [1, 2]:
        c = cells[n]
        kf = c['kabs'] / c['G']
        dev = []
        for ik in range(len(kf)):
            j = np.argmin(np.abs(kf4 - fold_f(kf[ik], N / n)))
            dev.append(np.abs(c['eigs'][ik][:, None] - e4s[j][None, :]).min(1).max())
        print(f'fold-check x{n}->x{N}: worst-band max dev = {max(dev) * 1000:.1f} meV')


if __name__ == '__main__':
    main()
