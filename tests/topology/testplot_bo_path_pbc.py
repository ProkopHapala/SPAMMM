#!/usr/bin/env python3
"""Bond-length + pi-BO maps along the relaxed proton-transfer paths (PBC chain cells).

Loads a recovered scan.pkl (corners + relaxed path frames), computes per-frame
bond lengths and pi bond orders (pi_bond_orders_pbc: 2 local SP runs/frame,
~seconds) and plots canonical bond-scalar maps (same style as ribbon enum maps):
rows = [bond length | pi-BO], cols = path frames, cell tiled 2x along y.

Two figures per system: stepwise LL->RL->RR and concerted diagonal LL->RR.
Corner labels use the proton-HOST convention: letter j = molecule holding proton j
('L' = lower molecule, 'R' = upper), i.e. LL = both H on the same molecule.
(The corner_* dirs keep the old u-coordinate names; proton_corner_names maps them.)

Run:
  python tests/topology/testplot_bo_path_pbc.py --name st1x1Nn --work-root /tmp/cluster_sim
  python tests/topology/testplot_bo_path_pbc.py --name st1x1Nn,st1x1OO --plot-only
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spammm.topology.ascii_art_heterocycle import build_pbc_cell
from spammm.quantum.coordinate_scan import load_scan
from spammm.quantum.pi_bond_order import HAU2EV

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_bo_path_pbc')
MIXER = 'DIIS { Generations = 8 }'      # same as ribbon maps — DIIS >> Broyden for charge sloshing


def bond_lengths_minimage_y(apos, bonds, lv):
    """Bond lengths [A], minimum-image along lattice vector lv (periodic axis)."""
    d = apos[np.asarray(bonds)[:, 1]] - apos[np.asarray(bonds)[:, 0]]
    d -= np.round(d[:, 1] / lv[1])[:, None] * lv[None, :]
    return np.linalg.norm(d, axis=1)


def _heavy_mask(atoms):
    en = np.array([e.split('_')[0] for e in atoms.enames])
    b = np.asarray(atoms.bonds)
    return (en[b[:, 0]] != 'H') & (en[b[:, 1]] != 'H')


def _cc_mask(atoms):
    en = np.array([e.split('_')[0] for e in atoms.enames])
    b = np.asarray(atoms.bonds)
    return (en[b[:, 0]] == 'C') & (en[b[:, 1]] == 'C')


def _mol_ids(atoms):
    """Molecule id per atom = connected component of atoms.bonds, ordered by mean y (mol 0 = lower)."""
    n = atoms.natoms
    comp = -np.ones(n, int)
    adj = [[] for _ in range(n)]
    for a, b in atoms.bonds:
        adj[a].append(b); adj[b].append(a)
    nc = 0
    for i in range(n):
        if comp[i] >= 0:
            continue
        stack = [i]; comp[i] = nc
        while stack:
            for j in adj[stack.pop()]:
                if comp[j] < 0:
                    comp[j] = nc; stack.append(j)
        nc += 1
    order = np.argsort([atoms.apos[comp == c, 1].mean() for c in range(nc)])
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[c] for c in comp])


def proton_corner_names(atoms, hbonds):
    """Corner name by proton HOST molecule (user convention): letter_j = 'L' if proton j is
    covalently bonded to the lower molecule (mol 0), 'R' otherwise.  LL = both H on one molecule.
    Maps u=(u1,u2) -> name; u_j<0.5 -> proton on donor, else on acceptor."""
    comp = _mol_ids(atoms)
    return {u: ''.join('L' if comp[hb.donor_idx if u[j] < 0.5 else hb.acceptor_idx] == 0 else 'R' for j, hb in enumerate(hbonds)) for u in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]}


def _tile_y(atoms, apos_f, bvals, lvs):
    """Cell + replica along lvs[1]; seam bonds (crossing y boundary) drawn i->j+n."""
    from types import SimpleNamespace
    n = atoms.natoms
    bonds = np.asarray(atoms.bonds)
    dvec = lvs[1].copy()
    seam = np.abs(apos_f[bonds[:, 1], 1] - apos_f[bonds[:, 0], 1]) > 0.5 * lvs[1, 1]
    hvy = _heavy_mask(atoms)
    nb, bs, hb = bonds[~seam], bonds[seam], hvy[~seam]
    bonds2 = np.concatenate([nb, nb + n, np.stack([bs[:, 0], bs[:, 1] + n], 1)])
    bv2 = np.concatenate([bvals[~seam], bvals[~seam], bvals[seam]])
    mk2 = np.concatenate([hb, hb, np.zeros(len(bs), bool)])
    ns = SimpleNamespace(bonds=bonds2, enames=list(atoms.enames) * 2, natoms=2 * n)
    return ns, np.concatenate([apos_f, apos_f + dvec]), bv2, mk2


def plot_path_maps(rows, title, png, lvs, hbonds):
    """rows = [(label, apos, bl, bpi, atoms, q)]; 3 rows [BL|pi-BO|q+ESP] x cols, 2-cell tile along y."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from spammm import elements
    from spammm.quantum import pi_bond_order as pbo
    from spammm.topology.hbond_utils import hbond_positions
    all_bl = np.concatenate([r[2][_cc_mask(r[4])] for r in rows])
    all_bp = np.concatenate([r[3][_cc_mask(r[4])] for r in rows])
    all_bp = all_bp[np.isfinite(all_bp)]
    all_q = np.concatenate([r[5] for r in rows])
    Lcc = 1.42                                        # canonical aromatic C-C = colormap center
    dl = np.abs(all_bl - Lcc).max()
    lnorm = mcolors.TwoSlopeNorm(vmin=Lcc - dl, vcenter=Lcc, vmax=Lcc + dl)
    db = np.abs(all_bp - 0.5).max() if len(all_bp) else 0.5
    bnorm = mcolors.TwoSlopeNorm(vmin=0.5 - db, vcenter=0.5, vmax=0.5 + db)
    dq = np.abs(all_q).max() or 0.1
    qnorm = mcolors.TwoSlopeNorm(vmin=-dq, vcenter=0.0, vmax=dq)
    nst = len(rows)
    Lx0, Ly0 = lvs[0, 0], lvs[1, 1]
    cw, ch = Lx0 + abs(lvs[1, 0]) + 1.0, 2 * Ly0 + 3.2
    pw = 1.8
    fig, axs = plt.subplots(3, nst, figsize=(nst * pw + 1.5, 3 * pw * ch / cw + 1.0), squeeze=False)
    lcs, sms = {}, {}
    # ESP grid on wide margins (macroscopic view); potential of Gaussian-smeared Mulliken
    # charges incl. periodic images along y (plummer a=0.8 A).  One scale for all columns.
    ns0, apos20, _, _ = _tile_y(rows[0][4], rows[0][1], np.zeros(len(rows[0][4].bonds)), lvs)
    xg = np.linspace(apos20[:, 0].min() - 6.0, apos20[:, 0].max() + 6.0, 140)
    yg = np.linspace(apos20[:, 1].min() - 4.0, apos20[:, 1].max() + 4.0, 280)
    X, Y = np.meshgrid(xg, yg)
    Vs = []
    for r in rows:
        V = np.zeros_like(X)
        for rep in (-1, 0, 1, 2):
            for a in range(len(r[1])):
                V += r[5][a] / np.sqrt((X - r[1][a, 0] - rep * lvs[1, 0])**2 + (Y - r[1][a, 1] - rep * lvs[1, 1])**2 + 0.64)
        Vs.append(V)
    vlim = np.percentile(np.abs(np.stack(Vs)), 92)
    for i, (lab, apos_f, bl, bpi, atoms, q) in enumerate(rows):
        for j, (bvals, norm) in enumerate([(bl, lnorm), (bpi, bnorm)]):
            ax = axs[j, i]
            ns, apos2, bv2, mk2 = _tile_y(atoms, apos_f, bvals, lvs)
            lcs[j], sms[j] = pbo.plot_bond_scalar_map(ax, ns, apos2, bv2, bonds=ns.bonds, cmap='coolwarm', norm=norm, mask=mk2, bAtoms=False)
            en2 = [e.split('_')[0] for e in ns.enames]
            ax.scatter(apos2[:, 0], apos2[:, 1], s=6, c=[elements.ELEMENT_DICT[e][8] for e in en2], zorder=6, linewidths=0)
        # row 3: ESP field from Mulliken point charges + atoms colored by q
        ax = axs[2, i]
        ns, apos2, _, _ = _tile_y(atoms, apos_f, np.zeros(len(atoms.bonds)), lvs)
        q2 = np.concatenate([q, q])
        ax.contourf(X, Y, Vs[i], levels=np.linspace(-vlim, vlim, 21), cmap='RdBu_r', alpha=0.55, zorder=1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection([[(apos2[b[0], 0], apos2[b[0], 1]), (apos2[b[1], 0], apos2[b[1], 1])] for b in ns.bonds], colors='0.4', linewidths=0.6, zorder=3))
        en2 = [e.split('_')[0] for e in ns.enames]
        sms[2] = ax.scatter(apos2[:, 0], apos2[:, 1], s=26, c=q2, cmap='coolwarm', norm=qnorm, zorder=6, linewidths=0)
        for hb in hbonds:
            pD, pH, pA = hbond_positions(apos_f, hb, lvs)
            for rep in (-1, 0, 1):
                D, H, A = pD + rep * lvs[1], pH + rep * lvs[1], pA + rep * lvs[1]
                # solid green = covalent (bonded, shorter) side; magenta dashed = H-bond side
                (cov, hbd) = ((D, H), (H, A)) if np.linalg.norm(H - D) <= np.linalg.norm(H - A) else ((H, A), (D, H))
                for j in range(3):
                    axs[j, i].plot([cov[0][0], cov[1][0]], [cov[0][1], cov[1][1]], 'g-', lw=1.6, zorder=5)
                    axs[j, i].plot([hbd[0][0], hbd[1][0]], [hbd[0][1], hbd[1][1]], 'm--', lw=0.8, zorder=5)
        for j in range(3):
            ax = axs[j, i]
            if j == 0:
                ax.set_title(lab, fontsize=8)
            if j == 2:                                  # ESP row: wide macroscopic view
                ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(yg[0], yg[-1])
            else:
                ax.set_xlim(apos2[:, 0].min() - 1.5, apos2[:, 0].max() + 1.5)
                ax.set_ylim(apos2[:, 1].min() - 1.5, apos2[:, 1].max() + 1.5)
            ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle(title, fontsize=11)
    for j, lab in enumerate(['bond length [A]', 'pi bond order', 'Mulliken q + ESP']):
        sm = sms[j]
        cax = axs[j, -1].inset_axes([1.04, 0.15, 0.04, 0.7])
        fig.colorbar(sm, cax=cax)
        axs[j, 0].text(-0.06, 0.5, lab, rotation=90, va='center', ha='right', fontsize=9, transform=axs[j, 0].transAxes)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png}\nREVIEW: {png}")


def run_corners(name, work_root, nk=(1, 8, 1), plot_only=False):
    """BL + pi-BO maps of the 4 relaxed corner states (LL/RL/LR/RR) — endpoints only,
    no path frames needed.  Reads geom.out.gen/xyz from <work_root>/<name>/corner_*/."""
    from spammm.quantum import pi_bond_order as pbo
    from spammm.quantum.DFTB_utils import read_relaxed_geometry, parse_energy_out, parse_mulliken_charges
    from spammm.quantum.coordinate_scan import CORNER_US, CORNER_NAMES
    work_dir = os.path.join(work_root, name)
    atoms, lvs, hbonds = build_pbc_cell(name)
    bonds = np.asarray(atoms.bonds)
    enames = atoms.enames

    cache = os.path.join(work_dir, 'bo_corners.npz')
    data = dict(np.load(cache, allow_pickle=True)) if plot_only and os.path.exists(cache) else {}
    cwd = os.getcwd()
    if not data:
        for u in CORNER_US:
            cname = CORNER_NAMES[u]
            d = os.path.join(work_dir, f'corner_{cname}')
            os.chdir(d)
            apos_r = read_relaxed_geometry(atoms.apos, do_relax=True)
            E = parse_energy_out('OUT') * HAU2EV
            os.chdir(cwd)
            bl = bond_lengths_minimage_y(apos_r, bonds, lvs[1])
            bpi, info = pbo.pi_bond_orders_pbc(enames, apos_r, bonds, lvs, os.path.join(d, 'bo'), nk=nk, filling_temp=300, scctol=1e-6, maxscc=2000, mixer=MIXER, verbose=False)
            q = parse_mulliken_charges(os.path.join(d, 'bo', 'scf', 'detailed.out'))
            data[f'apos_{cname}'] = apos_r; data[f'bl_{cname}'] = bl; data[f'bpi_{cname}'] = bpi; data[f'E_{cname}'] = E; data[f'q_{cname}'] = q
            print(f"  {cname}: E={E:.4f} eV  E_bo={info['E_ha']*HAU2EV:.4f} eV", flush=True)
        np.savez(cache, **data)
    for u in CORNER_US:                      # backfill q into older caches (detailed.out persists)
        cname = CORNER_NAMES[u]
        qf = os.path.join(work_dir, f'corner_{cname}', 'bo', 'scf', 'detailed.out')
        if f'q_{cname}' not in data and os.path.exists(qf):
            data[f'q_{cname}'] = parse_mulliken_charges(qf)
            np.savez(cache, **data)

    pname = proton_corner_names(atoms, hbonds)        # u -> display name (letter = proton host molecule)
    name2u = {v: k for k, v in pname.items()}
    Emin = min(float(data[f'E_{CORNER_NAMES[u]}']) for u in CORNER_US)
    rows = [(f"{pname[u]}\ndE={float(data[f'E_{CORNER_NAMES[u]}'])-Emin:+.2f} eV",
             data[f'apos_{CORNER_NAMES[u]}'], data[f'bl_{CORNER_NAMES[u]}'], data[f'bpi_{CORNER_NAMES[u]}'], atoms, data[f'q_{CORNER_NAMES[u]}'])
            for u in [name2u[n] for n in ('LL', 'RL', 'LR', 'RR')]]
    os.makedirs(OUTDIR, exist_ok=True)
    E = {n: float(data[f'E_{CORNER_NAMES[name2u[n]]}']) for n in name2u}
    J = E['RR'] + E['LL'] - E['RL'] - E['LR']
    plot_path_maps(rows, f'{name} corners   J={J:+.3f} eV   (label = proton host molecule)', os.path.join(OUTDIR, f'bomap_{name}_corners.png'), lvs, hbonds)


def run_one(name, work_root, nk=(1, 8, 1), plot_only=False):
    from spammm.quantum import pi_bond_order as pbo
    from spammm.quantum.DFTB_utils import parse_mulliken_charges
    work_dir = os.path.join(work_root, name)
    scan = load_scan(os.path.join(work_dir, 'scan.pkl'))
    atoms, lvs, hbonds = build_pbc_cell(name)
    enames = scan['enames']
    bonds = np.asarray(atoms.bonds)
    Emin = min(c['e_ev'] for c in scan['corners'].values())

    cache = os.path.join(work_dir, 'bo_path.npz')
    data = {}
    if plot_only and os.path.exists(cache):
        data = dict(np.load(cache, allow_pickle=True))
    if not data:
        # (path_idx, frame_idx) -> relaxed apos; compute bl + bpi per frame (2 SP runs each, ~seconds)
        for ip, pth in enumerate(scan['paths']):
            apos_frames = pth.get('apos_frames_relaxed')
            if apos_frames is None:
                apos_frames = pth['apos_frames']
            for i, apos_f in enumerate(apos_frames):
                if apos_f is None:
                    continue
                key = f'{ip}_{i}'
                wd = os.path.join(work_dir, 'bo', key)
                bl = bond_lengths_minimage_y(apos_f, bonds, lvs[1])
                bpi, info = pbo.pi_bond_orders_pbc(enames, np.asarray(apos_f), bonds, lvs, wd, nk=nk, filling_temp=300, scctol=1e-6, maxscc=2000, mixer=MIXER, verbose=False)
                data[f'bl_{key}'] = bl; data[f'bpi_{key}'] = bpi; data[f'apos_{key}'] = np.asarray(apos_f)
                data[f'q_{key}'] = parse_mulliken_charges(os.path.join(wd, 'scf', 'detailed.out'))
                print(f"  {key} {pth['label']}: E_bo={info['E_ha']*HAU2EV:.4f} eV", flush=True)
        np.savez(cache, **data)
    # backfill q for frames cached before charges existed (detailed.out persists)
    for k in [k for k in list(data) if k.startswith('apos_')]:
        key = k[5:]
        if f'q_{key}' not in data:
            qf = os.path.join(work_dir, 'bo', key, 'scf', 'detailed.out')
            if os.path.exists(qf):
                data[f'q_{key}'] = parse_mulliken_charges(qf)
                np.savez(cache, **data)

    pname = proton_corner_names(atoms, hbonds)        # u -> display name (letter = proton host molecule)
    name2u = {v: k for k, v in pname.items()}

    def _edge(u_from, u_to):
        """Find path index connecting two corners; returns (ip, reverse?)."""
        for ip, pth in enumerate(scan['paths']):
            if np.allclose(pth['uA'], u_from) and np.allclose(pth['uB'], u_to):
                return ip, False
            if np.allclose(pth['uA'], u_to) and np.allclose(pth['uB'], u_from):
                return ip, True
        raise KeyError(f'no path {u_from}->{u_to}')

    def _frame_rows(segments):
        """Concatenate path segments [(ip, reverse)] into frame rows (drop shared joint frames)."""
        rows = []
        for ip, rev in segments:
            pth = scan['paths'][ip]
            Er = pth.get('energies_ev_relaxed')
            if Er is None:
                Er = pth['energies_ev']
            n = len(pth['fracs'])
            first = True
            for i in (range(n - 1, -1, -1) if rev else range(n)):
                key = f'{ip}_{i}'
                if f'apos_{key}' not in data:
                    continue
                if rows and first:
                    first = False; continue             # shared endpoint already plotted
                first = False
                fr = (1.0 - pth['fracs'][i]) if rev else pth['fracs'][i]
                u_end = pth['uA'] if i == 0 else (pth['uB'] if i == n - 1 else None)
                tag = f" {pname[tuple(float(x) for x in u_end)]}" if u_end is not None else ''
                lab = f"{tag} f={fr:.2f}  dE={Er[i]-Emin:+.2f} eV".strip()
                rows.append((lab, data[f'apos_{key}'], data[f'bl_{key}'], data[f'bpi_{key}'], atoms, data.get(f'q_{key}', np.zeros(atoms.natoms))))
        return rows

    uLL, uRR, uRL = name2u['LL'], name2u['RR'], name2u['RL']
    os.makedirs(OUTDIR, exist_ok=True)
    plot_path_maps(_frame_rows([_edge(uLL, uRL), _edge(uRL, uRR)]), f'{name} stepwise LL->RL->RR', os.path.join(OUTDIR, f'bopath_{name}_step.png'), lvs, hbonds)
    plot_path_maps(_frame_rows([_edge(uLL, uRR)]), f'{name} concerted LL->RR', os.path.join(OUTDIR, f'bopath_{name}_diag.png'), lvs, hbonds)


def main():
    ap = argparse.ArgumentParser(description='Bond-length + pi-BO maps along relaxed PT paths')
    ap.add_argument('--name', required=True, help='system name(s), comma-separated')
    ap.add_argument('--work-root', default=None, help='dir containing <name>/scan.pkl (default: debug/test_corner_scan_pbc)')
    ap.add_argument('--nk', type=int, default=8, help='k-points along chain for pi-BO SPs (default: 8)')
    ap.add_argument('--plot-only', action='store_true', help='replot from cached bo_path.npz')
    ap.add_argument('--corners', action='store_true', help='map only the 4 relaxed corner states (endpoints) — needs corner_*/ outputs, no path frames')
    args = ap.parse_args()
    wr = args.work_root or os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_corner_scan_pbc')
    for name in args.name.split(','):
        print(f"\n=== {name} ===")
        (run_corners if args.corners else run_one)(name, wr, nk=(1, args.nk, 1), plot_only=args.plot_only)


if __name__ == '__main__':
    main()
