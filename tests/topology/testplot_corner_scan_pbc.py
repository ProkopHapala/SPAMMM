#!/usr/bin/env python3
"""Periodic four-corner square scan for 1D H-bond chain cells (DFTB+ PBC).

Cell = build_pbc_cell(name) from PBC_CHAIN_ARTS — one junction internal, one
crossing the cell boundary (HbondRecord.a_shift/d_shift).  Corners relaxed with
`run_pbc` (pinned scan-H + bond partner), edges+diagonals SP on interpolated
frames; J = E_RR+E_LL-E_RL-E_LR.

Outputs: debug/test_corner_scan_pbc/corner_<name>.{png,xyz} + _overlay.png
Run: python tests/topology/testplot_corner_scan_pbc.py --name quinolone4
     python tests/topology/testplot_corner_scan_pbc.py --name all
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spammm.topology.ascii_art_heterocycle import build_pbc_cell, PBC_CHAIN_ARTS
from spammm.quantum.coordinate_scan import run_corner_scan_pbc, plot_corner_scan, plot_corner_overlay, write_corner_scan_xyz

DEBUG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_corner_scan_pbc')


def run_one(name, dx=0.25, relax_corners=True, sk_set=None, work_root=None, nk=(1, 8, 1), filling_temp=300.0, rescue_temp=600.0, hbond_length=2.8, tilt=45.0):
    atoms, lvs, hbonds = build_pbc_cell(name, hbond_length=hbond_length, tilt=tilt)
    if len(hbonds) < 2:
        print(f"  SKIP {name}: only {len(hbonds)} junctions (need 2 for corner square)")
        return None
    mapping = [0, 1]
    work_dir = os.path.join(work_root or DEBUG_DIR, name)
    print(f"\n=== {name}: PBC corner square ({len(hbonds)} junctions, Ly={lvs[1, 1]:.2f} A, nk={nk}, dx={dx}) ===")
    scan = run_corner_scan_pbc(atoms.enames, atoms.apos, lvs, hbonds, mapping, dx=dx, relax_corners=relax_corners, sk_set=sk_set, work_dir=work_dir, nk=nk, filling_temp=filling_temp, rescue_temp=rescue_temp, verbose=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    for u in scan['corner_us']:
        c0 = scan['corners'][u]
        print(f"  {c0['name']} u={u}:  E = {c0['e_ev']:.4f} eV")
    print(f"  J = E_RR+E_LL-E_RL-E_LR = {scan['J_ev']:+.4f} eV  ({scan['J_ev']*1000:+.1f} meV)")
    png = os.path.join(DEBUG_DIR, f'corner_{name}.png')
    png_ov = os.path.join(DEBUG_DIR, f'corner_{name}_overlay.png')
    xyz = os.path.join(DEBUG_DIR, f'corner_{name}.xyz')
    plot_corner_scan(scan, atoms, f'{name} PBC corner square', png)
    plot_corner_overlay(scan, atoms, png_ov)
    write_corner_scan_xyz(scan, atoms, xyz)
    print(f"REVIEW: {png}")
    print(f"REVIEW: {png_ov}")
    print(f"REVIEW: {xyz}")
    return scan


def main():
    parser = argparse.ArgumentParser(description='PBC 4-corner square scan for H-bond chain cells')
    parser.add_argument('--name', default='quinolone4', help=f'PBC_CHAIN_ARTS key (default: quinolone4; "all" = every entry; choices: {sorted(PBC_CHAIN_ARTS)})')
    parser.add_argument('--dx', type=float, default=0.25, help='Path fraction step (default: 0.25)')
    parser.add_argument('--sk_set', default=None, help='DFTB SK set (default: from config)')
    parser.add_argument('--nk', type=int, default=8, help='k-points along chain (default: 8)')
    parser.add_argument('--hbond', type=float, default=2.8, help='D..A junction gap [A] (default: 2.8)')
    parser.add_argument('--tilt', type=float, default=45.0, help='herringbone tilt [deg], alternate molecules +/-tilt (default: 45)')
    parser.add_argument('--no-relax', action='store_true', help='Skip corner relax (rigid corners, SP only)')
    parser.add_argument('--filling-temp', type=float, default=300.0, help='Fermi smearing T [K] for all PBC runs (default: 300)')
    parser.add_argument('--rescue-temp', type=float, default=600.0, help='Retry failed SCC at this Fermi T [K] (0 disables)')
    args = parser.parse_args()
    names = sorted(PBC_CHAIN_ARTS) if args.name == 'all' else [args.name]
    for name in names:
        run_one(name, dx=args.dx, relax_corners=not args.no_relax, sk_set=args.sk_set, nk=(1, args.nk, 1), filling_temp=args.filling_temp, rescue_temp=args.rescue_temp or None, hbond_length=args.hbond, tilt=args.tilt)


if __name__ == '__main__':
    main()
