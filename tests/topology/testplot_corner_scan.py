#!/usr/bin/env python3
"""Four-corner relaxed square scan for two-junction ASCII H-bond dimers (DFTB).

Corners (u1,u2) ∈ {0,1}² = both protons donor(L)/acceptor(R) side are DFTB-relaxed
first, then connected by interpolated paths along the 4 square edges + both diagonals.

Outputs: debug/test_corner_scan/corner_<name>.{png,xyz}
Run: python tests/topology/testplot_corner_scan.py
     python tests/topology/testplot_corner_scan.py --name 2Quinolone --dx 0.25
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spammm.quantum.hbond_scan import build_ascii_hbond_system, ascii_examples_with_hbonds
from spammm.topology.hbond_utils import find_hbonds_sys
from spammm.quantum.coordinate_scan import run_corner_scan, plot_corner_scan, plot_corner_overlay, write_corner_scan_xyz

DEBUG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_corner_scan')


def run_one(name, dx=0.25, relax_corners=True, sk_set=None, work_root=None, on_fail='skip', filling_temp=None, rescue_temp=600.0):
    atoms = build_ascii_hbond_system(name)
    hbonds = find_hbonds_sys(atoms, bPrint=False)
    if len(hbonds) < 2:
        print(f"  SKIP {name}: only {len(hbonds)} H-bonds (need 2 for corner square)")
        return None
    hbonds = hbonds[:2]
    mapping = [0, 1]
    work_dir = os.path.join(work_root or DEBUG_DIR, name)
    print(f"\n=== {name}: corner square ({len(hbonds)} junctions, dx={dx}) ===")
    scan = run_corner_scan(atoms.enames, atoms.apos, hbonds, mapping, dx=dx, relax_corners=relax_corners, sk_set=sk_set, work_dir=work_dir, verbose=True, on_fail=on_fail, filling_temp=filling_temp, rescue_temp=rescue_temp)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    for u in scan['corner_us']:
        c0 = scan['corners'][u]
        print(f"  {c0['name']} u={u}:  E = {c0['e_ev']:.4f} eV")
    print(f"  J = E_RR+E_LL-E_RL-E_LR = {scan['J_ev']:+.4f} eV  ({scan['J_ev']*1000:+.1f} meV)")
    png = os.path.join(DEBUG_DIR, f'corner_{name}.png')
    png_ov = os.path.join(DEBUG_DIR, f'corner_{name}_overlay.png')
    xyz = os.path.join(DEBUG_DIR, f'corner_{name}.xyz')
    plot_corner_scan(scan, atoms, f'{name} corner square', png)
    plot_corner_overlay(scan, atoms, png_ov)
    write_corner_scan_xyz(scan, atoms, xyz)
    print(f"REVIEW: {png}")
    print(f"REVIEW: {png_ov}")
    print(f"REVIEW: {xyz}")
    return scan


def main():
    parser = argparse.ArgumentParser(description='Relaxed 4-corner square scan for ASCII H-bond dimers')
    parser.add_argument('--name', default='2Quinolone', help='ASCII example name (default: 2Quinolone; "all" = every 2-junction example)')
    parser.add_argument('--dx', type=float, default=0.25, help='Path fraction step along edges/diagonals (default: 0.25)')
    parser.add_argument('--sk_set', default=None, help='DFTB SK set (default: from config)')
    parser.add_argument('--no-relax', action='store_true', help='Skip corner DFTB relax (rigid corners, SP only)')
    parser.add_argument('--on-fail', default='skip', choices=['raise', 'skip'])
    parser.add_argument('--filling-temp', type=float, default=None, help='Fermi smearing T [K] applied uniformly to all corners/paths (uniform protocol for SCC-difficult systems)')
    parser.add_argument('--rescue-temp', type=float, default=600.0, help='Retry failed SCC at this Fermi T [K] (0 disables)')
    args = parser.parse_args()
    names = ascii_examples_with_hbonds() if args.name == 'all' else [args.name]
    for name in names:
        run_one(name, dx=args.dx, relax_corners=not args.no_relax, sk_set=args.sk_set, on_fail=args.on_fail, filling_temp=args.filling_temp, rescue_temp=args.rescue_temp or None)


if __name__ == '__main__':
    main()
