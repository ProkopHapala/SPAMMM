#!/usr/bin/env python3
"""Bond-length + pi bond-order maps for corner-scan H-bond dimers (DFTBcore density matrix).

Re-reads relaxed corner geometries from debug/test_corner_scan/<name>/corner_*/geom.out.xyz,
re-runs each through DFTBcore (same Fermi protocol as the scan) to get the dense density
matrix P and overlap S, then extracts Lowdin pi bond orders (pi_bond_order.pi_bond_order_matrix)
and C-C bond lengths, drawn as colormaps on the molecular skeleton.

Also produces the response maps phi1 = B(RL)-B(LL), phi2 = B(LR)-B(LL) and the
non-additive second difference dB = B(RR)+B(LL)-B(RL)-B(LR) (bond-order analogue of J).

With --paths, additionally recomputes DMs for every path frame and plots pi-BO evolution
of the most-responsive bonds along each square edge/diagonal.

Outputs: debug/test_bond_order/<name>_bondmap.png, <name>_bo_response.png [, <name>_bo_paths.png, <name>_bo.npz]
Run: python tests/topology/testplot_bond_order.py --name 2NCI [--paths] [--temp 600]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spammm import atomicUtils as au
from spammm.quantum.hbond_scan import build_ascii_hbond_system, ascii_examples_with_hbonds, HAU2EV
from spammm.quantum.coordinate_scan import CORNER_US, CORNER_NAMES, SQUARE_PATHS
from spammm.quantum import pi_bond_order as pbo

SCAN_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_corner_scan')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_bond_order')
CORNER_DIR_NAMES = ['LL', 'RL', 'LR', 'RR']  # plot order; CORNER_US = (0,0),(1,0),(0,1),(1,1)


def _last_total_energy(out_file):
    """Last 'Total Energy' [H] line of a dftb+ OUT file; nan if missing."""
    e = np.nan
    with open(out_file) as f:
        for line in f:
            if 'Total Energy' in line:
                e = float(line.split(':')[1].split()[0])
    return e


def load_corner_geometry(scan_root, name, cname, enames_ref):
    """Read relaxed corner geometry (geom.out.xyz) or input geom.xyz as fallback."""
    cdir = os.path.join(scan_root, name, f'corner_{cname}')
    for fn in ('geom.out.xyz', 'geom.xyz'):
        p = os.path.join(cdir, fn)
        if os.path.isfile(p):
            apos, Zs, enames, qs, _ = au.load_xyz(p)
            apos = np.asarray(apos, dtype=float)
            assert list(enames) == list(enames_ref), f"{p}: element order mismatch {enames} vs {enames_ref}"
            return apos, cdir
    raise FileNotFoundError(f"no geometry in {cdir}")


def compute_corner_state(enames, apos, work_dir, layout, bonds, filling_temp, e_ref=None, verbose=True):
    """DFTBcore SCF -> Lowdin pi BO per bond + bond lengths. Returns dict."""
    res = pbo.run_dftbcore_sp(enames, apos, work_dir, filling_temp=filling_temp, verbose=verbose)
    Pld, pi_atoms = pbo.pi_bond_order_matrix(res['dm'], res['S'], layout, apos=apos, enames=enames)
    bo = pbo.bond_orders_from_pmat(Pld, pi_atoms, bonds)
    out = {'E_ha': res['E_ha'], 'E_ev': res['E_ha'] * HAU2EV, 'bo': bo, 'bl': pbo.bond_lengths(apos, bonds), 'Ppi': Pld, 'apos': apos}
    if e_ref is not None and np.isfinite(e_ref):
        out['dE_ref_mev'] = (res['E_ha'] - e_ref) * HAU2EV * 1000.0
        if verbose:
            print(f"      parity vs stored OUT: dE = {out['dE_ref_mev']:+.1f} meV")
    return out


def plot_corner_maps(name, atoms, corners, bonds, cc_mask, hb_mask, J_ev, png):
    """2-row figure: row0 C-C bond length map, row1 pi bond-order map; cols = LL,RL,LR,RR."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    cn = CORNER_DIR_NAMES
    bl_all = np.concatenate([corners[c]['bl'][cc_mask] for c in cn])
    bo_all = np.concatenate([corners[c]['bo'][hb_mask] for c in cn])
    # diverging cmap centered at data midpoint: red = double-like, blue = single-like
    # (length map uses reversed cmap so short bonds = red in both rows)
    n1 = mcolors.TwoSlopeNorm(vmin=bl_all.min(), vcenter=0.5 * (bl_all.min() + bl_all.max()), vmax=bl_all.max())
    n2 = mcolors.TwoSlopeNorm(vmin=bo_all.min(), vcenter=0.5 * (bo_all.min() + bo_all.max()), vmax=bo_all.max())
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for k, c in enumerate(cn):
        st = corners[c]
        e_tag = f"  dE={st['dE_ref_mev']:+.0f}meV" if 'dE_ref_mev' in st else ''
        pbo.plot_bond_scalar_map(axes[0, k], atoms, st['apos'], st['bl'], bonds=bonds, cmap='seismic_r', norm=n1, mask=cc_mask)
        axes[0, k].set_title(f"{c}  E={st['E_ev']:.3f}eV{e_tag}", fontsize=9)
        pbo.plot_bond_scalar_map(axes[1, k], atoms, st['apos'], st['bo'], bonds=bonds, cmap='seismic', norm=n2, mask=hb_mask)
    fig.colorbar(plt.cm.ScalarMappable(cmap='seismic_r', norm=n1), ax=list(axes[0, :]), shrink=0.7, label='C-C bond length [A]', fraction=0.02)
    fig.colorbar(plt.cm.ScalarMappable(cmap='seismic', norm=n2), ax=list(axes[1, :]), shrink=0.7, label='pi bond order (Lowdin p-perp)', fraction=0.02)
    fig.suptitle(f"{name}: bond maps over proton states  (J = {J_ev:+.4f} eV)" if np.isfinite(J_ev) else f"{name}: bond maps")
    fig.savefig(png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png}")


def plot_response_maps(name, atoms, corners, bonds, hb_mask, png):
    """phi1 = RL-LL, phi2 = LR-LL, dB = RR+LL-RL-LR (diverging, shared norm)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    phi1 = corners['RL']['bo'] - corners['LL']['bo']
    phi2 = corners['LR']['bo'] - corners['LL']['bo']
    dB = corners['RR']['bo'] + corners['LL']['bo'] - corners['RL']['bo'] - corners['LR']['bo']
    v = max(np.nanmax(np.abs(phi1[hb_mask])), np.nanmax(np.abs(phi2[hb_mask])), np.nanmax(np.abs(dB[hb_mask])))
    norm = mcolors.Normalize(vmin=-v, vmax=+v)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, vals, tit in zip(axes, (phi1, phi2, dB), ('phi_1 = B(RL)-B(LL)', 'phi_2 = B(LR)-B(LL)', 'dB = RR+LL-RL-LR')):
        pbo.plot_bond_scalar_map(ax, atoms, corners['LL']['apos'], vals, bonds=bonds, cmap='seismic', norm=norm, mask=hb_mask)
        ax.set_title(tit, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    fig.colorbar(sm, ax=list(axes), shrink=0.7, label='delta pi bond order')
    fig.suptitle(f"{name}: pi bond-order response / non-additivity")
    fig.savefig(png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png}")


def path_frames(scan_root, name):
    """Yield (ip, i, geom.xyz path) for every stored path frame."""
    root = os.path.join(scan_root, name)
    for d in sorted(os.listdir(root)):
        if d.startswith('path_') and os.path.isfile(os.path.join(root, d, 'geom.xyz')):
            ip, i = int(d.split('_')[1]), int(d.split('_')[2])
            yield ip, i, os.path.join(root, d)


def run_paths(scan_root, name, atoms, layout, bonds, hb_mask, filling_temp, work_root, verbose=True):
    """DM per stored path frame -> bo_frames[ip][i]. Returns dict {ip: (fracs, bo[nfr,nbond])}."""
    out = {}
    for ip, i, d in path_frames(scan_root, name):
        apos, _, _, _, _ = au.load_xyz(os.path.join(d, 'geom.xyz'))
        st = compute_corner_state(atoms.enames, np.asarray(apos, float), os.path.join(work_root, f'dm_p{ip:02d}_{i:03d}'), layout, bonds, filling_temp, verbose=verbose)
        out.setdefault(ip, {})[i] = st['bo']
    return {ip: (np.array(sorted(fr)) / max(sorted(fr)), np.array([fr[i] for i in sorted(fr)])) for ip, fr in out.items()}


def plot_bo_paths(name, atoms, bo_paths, bonds, hb_mask, png):
    """Per square path: pi-BO vs path fraction for the most-responsive bonds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    allbo = np.concatenate([bo for _, bo in bo_paths.values()])
    var = np.nanstd(allbo, axis=0)
    top = np.argsort(var)[::-1]
    top = top[np.isfinite(var[top])][:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ip, (frac, bo) in sorted(bo_paths.items()):
        ax = axes.flat[ip]
        uA, uB, lab = SQUARE_PATHS[ip]
        for b in top:
            ax.plot(frac, bo[:, b], 'o-', ms=3, lw=1, label=f"{atoms.enames[bonds[b][0]]}{bonds[b][0]}-{atoms.enames[bonds[b][1]]}{bonds[b][1]}")
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    for ax in axes[-1, :]:
        ax.set_xlabel('path fraction')
    for ax in axes[:, 0]:
        ax.set_ylabel('pi bond order')
    fig.suptitle(f"{name}: pi bond order along transfer paths")
    fig.savefig(png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {png}")


def run_one(name, filling_temp=600.0, b_paths=False, sk_set=None, verbose=True):
    atoms = build_ascii_hbond_system(name)
    if atoms.bonds is None:
        atoms.findBonds()
    bonds = np.asarray(atoms.bonds)
    heavy = np.array([e != 'H' for e in atoms.enames])
    hb_mask = np.array([heavy[i] and heavy[j] for i, j in bonds])          # heavy-heavy bonds (pi network)
    cc_mask = np.array([atoms.enames[i] == 'C' and atoms.enames[j] == 'C' for i, j in bonds])
    layout = pbo.ao_layout(atoms.enames)
    work_root = os.path.join(OUT_DIR, name)
    os.makedirs(work_root, exist_ok=True)

    corners = {}
    E = {}
    for u, cn in zip(CORNER_US, CORNER_DIR_NAMES):
        apos, cdir = load_corner_geometry(SCAN_DIR, name, cn, atoms.enames)
        e_ref = _last_total_energy(os.path.join(cdir, 'OUT')) if os.path.isfile(os.path.join(cdir, 'OUT')) else np.nan
        E[cn] = e_ref
        if verbose:
            print(f"  corner {cn}: DFTBcore SCF ...")
        corners[cn] = compute_corner_state(atoms.enames, apos, os.path.join(work_root, f'dm_{cn}'), layout, bonds, filling_temp, e_ref=e_ref, verbose=verbose)
    J_ev = E['RR'] + E['LL'] - E['RL'] - E['LR'] if all(np.isfinite(list(E.values()))) else np.nan
    J_ev *= HAU2EV
    print(f"  J (from stored OUTs) = {J_ev:+.4f} eV")

    os.makedirs(OUT_DIR, exist_ok=True)
    png1 = os.path.join(OUT_DIR, f'{name}_bondmap.png')
    png2 = os.path.join(OUT_DIR, f'{name}_bo_response.png')
    plot_corner_maps(name, atoms, corners, bonds, cc_mask, hb_mask, J_ev, png1)
    plot_response_maps(name, atoms, corners, bonds, hb_mask, png2)
    print(f"REVIEW: {png1}")
    print(f"REVIEW: {png2}")

    npz = {'bonds': bonds, 'enames': np.array(atoms.enames)}
    for cn in CORNER_DIR_NAMES:
        npz[f'bo_{cn}'] = corners[cn]['bo']
        npz[f'bl_{cn}'] = corners[cn]['bl']
        npz[f'E_{cn}'] = corners[cn]['E_ha']
        npz[f'apos_{cn}'] = corners[cn]['apos']

    if b_paths:
        bo_paths = run_paths(SCAN_DIR, name, atoms, layout, bonds, hb_mask, filling_temp, work_root, verbose=verbose)
        png3 = os.path.join(OUT_DIR, f'{name}_bo_paths.png')
        plot_bo_paths(name, atoms, bo_paths, bonds, hb_mask, png3)
        print(f"REVIEW: {png3}")
        for ip, (idx, bo) in bo_paths.items():
            npz[f'bo_path{ip}'] = bo
    npz_file = os.path.join(OUT_DIR, f'{name}_bo.npz')
    np.savez_compressed(npz_file, **npz)
    print(f"Saved: {npz_file}")
    return corners


def main():
    ap = argparse.ArgumentParser(description='pi bond-order maps from DFTBcore density matrices')
    ap.add_argument('--name', default='2NCI', help='ASCII example name (default 2NCI; "all" = all 2-junction examples)')
    ap.add_argument('--paths', action='store_true', help='also recompute DMs for all stored path frames')
    ap.add_argument('--temp', type=float, default=600.0, help='Fermi smearing T [K] matching the scan protocol (default 600)')
    ap.add_argument('--sk_set', default=None)
    args = ap.parse_args()
    names = ascii_examples_with_hbonds() if args.name == 'all' else [args.name]
    for name in names:
        print(f"\n=== {name} ===")
        run_one(name, filling_temp=args.temp, b_paths=args.paths, sk_set=args.sk_set)


if __name__ == '__main__':
    main()
