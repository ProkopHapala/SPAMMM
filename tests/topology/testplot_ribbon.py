#!/usr/bin/env python3
"""testplot_ribbon.py — N-terminated zigzag graphene nanoribbons, periodic along x.

Thin driver over spammm/topology/ribbon_pbc.py:
  build_ribbon_cell          — single ribbon PBC cell (wrap bonds in atoms.bonds)
  build_ribbon_junction_cell — two stacked ribbons -> periodic N...H-N interfaces
  scan_junction_gap          — E(d_DA) lattice-y scan via DFTB+ (3-pt parabola)
Plotting in spammm/plotUtils.py: plot_ribbon_pbc_cell, plot_ribbon_junction_cell,
plot_gap_scan.

Usage:
    python tests/topology/testplot_ribbon.py --widths 4,6,8 --ncells 4 [--passivation N] [--dftb --nk 8]
    python tests/topology/testplot_ribbon.py --two --widths 4,6,8 --ncells 4
    python tests/topology/testplot_ribbon.py --scan-ly --two --widths 4 --ncells 2 --lmin 2.4 --lmax 3.6 --npts 13

Outputs: debug/ribbon/<name>.png + <name>.xyz   (xyz comment line carries lvs)
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm.topology.ribbon_pbc import (build_ribbon_cell, build_ribbon_junction_cell,
                                       check_degrees, junction_geometry_report,
                                       save_xyz_lvs, scan_junction_gap, HAU2EV)
from spammm.plotUtils import plot_ribbon_pbc_cell, plot_ribbon_junction_cell, plot_gap_scan

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'ribbon')


def run_junction(args):
    """Two-ribbon N...H-N junction cell (periodic x, optional periodic y stack)."""
    from collections import Counter
    for w in [int(x) for x in args.widths.split(',')]:
        name = f'jgnr_w{w}_n{args.ncells}_{args.bottom}-{args.top}'
        atoms, lvs, hbonds = build_ribbon_junction_cell(width_chains=w, ncells=args.ncells, d_DA=args.dda,
                                                        shift_x=args.shift_x, bottom=args.bottom, top=args.top,
                                                        state=args.state, pbc_y=not args.no_pbc_y)
        nint = sum(1 for h in hbonds if h.a_shift == 0)
        print(f"\n=== {name}: natoms={atoms.natoms} {dict(Counter(atoms.enames))}  cell={lvs[0,0]:.2f}x{lvs[1,1]:.2f} A  junctions={len(hbonds)} ({nint} internal + {len(hbonds)-nint} boundary) ===")
        check_degrees(atoms, name)
        junction_geometry_report(atoms.apos, hbonds, lvs)
        png = os.path.join(OUTDIR, f'{name}.png')
        plot_ribbon_junction_cell(atoms, lvs, hbonds, savepath=png, title=f'{name}: {atoms.natoms} atoms/cell, {len(hbonds)} junction sites, cell {lvs[0,0]:.2f}x{lvs[1,1]:.2f} A')
        xyz = os.path.join(OUTDIR, f'{name}.xyz')
        save_xyz_lvs(xyz, atoms, lvs, name)
        print(f"  wrote {xyz}\nREVIEW: {png}")
        if args.dftb:
            from spammm.quantum.DFTB_utils import run_pbc
            E_ha, apos_out, forces = run_pbc(atoms.apos, atoms.enames, lvs, nk=(args.nk, args.nky, 1), workdir=os.path.join(OUTDIR, 'dftb_' + name), Temperature=300)
            print(f"  DFTB+ PBC SP (nk={args.nk}x{args.nky}): E = {E_ha*HAU2EV:.4f} eV   ({E_ha*HAU2EV/atoms.natoms:.3f} eV/atom)")


def run_scan_ly(args):
    """Scan junction N...N distance d_DA (lattice-y optimization) for given states."""
    n = args.ncells
    states = args.states.split(',') if args.states else None
    gaps = np.linspace(args.lmin, args.lmax, args.npts)
    labels = {'p' * (2 * n): 'all-p', 'd' * (2 * n): 'all-d', 'pd' * n: 'alternating'}
    mode = 'relax' if args.relax else 'SP'
    for w in [int(x) for x in args.widths.split(',')]:
        res, fits, parity = scan_junction_gap(w, n, gaps, states=states, nk=(args.nk, args.nky, 1),
                                            relax=args.relax, workdir=os.path.join(OUTDIR, f'scanly_w{w}'))
        png = os.path.join(OUTDIR, f'scanly_w{w}_n{n}.png')
        plot_gap_scan(gaps, res, fits, labels=labels, savepath=png,
                      title=f'junction gap scan w{w} n{n} ({mode}, nk={args.nk}x{args.nky})')
        print(f"REVIEW: {png}")
        np.savez(png.replace('.png', '.npz'), gaps=gaps,
                 **{f"E_{labels.get(st, st)}": res[st] for st in res},
                 **{f"fit_{labels.get(st, st)}": np.array([fits[st][1], fits[st][2], 2 * fits[st][0][0]]) for st in fits})   # [d_opt, E_opt, k] per state


def main():
    ap = argparse.ArgumentParser(description='PBC zigzag ribbons with N-terminated edges')
    ap.add_argument('--widths', default='4,6,8', help='comma list of atom-row counts across the ribbon (default: 4,6,8)')
    ap.add_argument('--ncells', type=int, default=4, help='unit cells along x (default: 4; >=2)')
    ap.add_argument('--passivation', default='N', help="edge termination: 'N' (pyridinic), 'NH' (protonated), or passivation string e.g. 'NnNn'")
    ap.add_argument('--dftb', action='store_true', help='run DFTB+ PBC single point per ribbon (verification)')
    ap.add_argument('--nk', type=int, default=8, help='k-points along the ribbon axis x (default: 8)')
    ap.add_argument('--two', action='store_true', help='two-ribbon junction cell: bottom + top ribbons stacked along y with N...H-N interfaces')
    ap.add_argument('--bottom', default='N', help="--two: bottom-ribbon edge passivation (default 'N' = acceptor)")
    ap.add_argument('--top', default='NH', help="--two: top-ribbon edge passivation (default 'NH' = donor)")
    ap.add_argument('--dda', type=float, default=2.9, help='--two: N...N distance across junction [A] (default 2.9)')
    ap.add_argument('--state', default=None, help="--two: per-site state string, 2*ncells chars over {p,d,0} (p=H up, d=H down, 0=bare); '|' separates the two junctions")
    ap.add_argument('--shift-x', type=float, default=0.0, help='--two: lateral shift of top ribbon (fraction of Lx)')
    ap.add_argument('--no-pbc-y', action='store_true', help='--two: vacuum along y instead of periodic junction stack')
    ap.add_argument('--nky', type=int, default=1, help='--two --dftb: k-points along the stacking direction y (default: 1)')
    ap.add_argument('--scan-ly', action='store_true', help='--two: scan junction N...N distance (lattice y optimization) with DFTB+ SP')
    ap.add_argument('--states', default=None, help="--scan-ly: comma list of state strings (default: all-p, alternating pd.., all-d)")
    ap.add_argument('--lmin', type=float, default=2.4, help='--scan-ly: min N...N distance [A] (default 2.4)')
    ap.add_argument('--lmax', type=float, default=3.4, help='--scan-ly: max N...N distance [A] (default 3.4)')
    ap.add_argument('--npts', type=int, default=9, help='--scan-ly: number of scan points (default 9)')
    ap.add_argument('--relax', action='store_true', help='--scan-ly: ionic relax at each point (fixed cell) instead of SP')
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.scan_ly:
        run_scan_ly(args)
        return
    if args.two:
        run_junction(args)
        return

    from collections import Counter
    for w in [int(x) for x in args.widths.split(',')]:
        name = f'zgnr_w{w}_n{args.ncells}_{args.passivation}'
        atoms, lvs, seam = build_ribbon_cell(w, args.ncells, passivation=args.passivation)
        print(f"\n=== {name}: natoms={atoms.natoms} {dict(Counter(atoms.enames))}  Lx={lvs[0,0]:.3f} A  seam_bonds={int(seam.sum())}/{len(atoms.bonds)} ===")
        check_degrees(atoms, name)
        png = os.path.join(OUTDIR, f'{name}.png')
        plot_ribbon_pbc_cell(atoms, lvs, seam, n_cells=3, savepath=png, title=f'{name}: {atoms.natoms} atoms/cell, Lx={lvs[0,0]:.2f} A')
        xyz = os.path.join(OUTDIR, f'{name}.xyz')
        save_xyz_lvs(xyz, atoms, lvs, name)
        print(f"  wrote {xyz}\nREVIEW: {png}")
        if args.dftb:
            from spammm.quantum.DFTB_utils import run_pbc
            E_ha, apos_out, forces = run_pbc(atoms.apos, atoms.enames, lvs, nk=(args.nk, 1, 1), workdir=os.path.join(OUTDIR, 'dftb_' + name), Temperature=300)
            print(f"  DFTB+ PBC SP ({args.nk} kpts): E = {E_ha*HAU2EV:.4f} eV   ({E_ha*HAU2EV/atoms.natoms:.3f} eV/atom)")


if __name__ == '__main__':
    main()
