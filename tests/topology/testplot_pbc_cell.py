#!/usr/bin/env python3
"""testplot_pbc_cell.py — build + visualise the periodic 1D H-bond chain cells.

Each PBC_CHAIN_ARTS entry is a stack of molecules along y separated by ':'
junction rows; the LAST block is the periodic image of the first (defines the
cell vector).  Cell contents = all blocks except the last.  Junctions between
in-cell blocks are internal; the junction into the last block crosses the cell
boundary (partner gets HbondRecord.a_shift/d_shift = +1).

Corner states for the 2-junction square:
    LL = both H on donors | RR = both transferred | RL/LR = mixed
    J = E_LL + E_RR - E_RL - E_LR   (<0 cooperative)

Usage:
    python tests/topology/testplot_pbc_cell.py [name|all] [--hbond 2.8]

Outputs: debug/pbc_cell/<name>_cell.png + <name>_cell.xyz
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm.topology.ascii_art_heterocycle import build_pbc_cell, PBC_CHAIN_ARTS
from spammm.topology.hbond_utils import junction_bond_lengths
from spammm import plotUtils as pu

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'pbc_cell')


def build_one(name, hbond_length=2.8, tilt=0.0):
    cell, lvs, hbonds = build_pbc_cell(name, hbond_length=hbond_length, tilt=tilt)
    print(f"\n=== {name}: natoms={cell.natoms}  Ly={lvs[1, 1]:.3f} A ===")
    e = lambda i: cell.enames[i]
    for j, hb in enumerate(hbonds):
        side = 'INTERNAL' if hb.d_shift == 0 and hb.a_shift == 0 else 'BOUNDARY'
        print(f"  j{j + 1} [{side}]: D={e(hb.donor_idx)}{hb.donor_idx}(sh{hb.d_shift})  H{hb.h_idx}  A={e(hb.acceptor_idx)}{hb.acceptor_idx}(sh{hb.a_shift})  D..A={hb.dist_ha:.3f} A")
    png = os.path.join(OUTDIR, f'{name}_cell.png')
    pu.plot_pbc_chain_cell(cell, lvs, hbonds, n_cells=3, savepath=png, title=f'{name}: cell = {cell.natoms} atoms, Ly={lvs[1, 1]:.2f} A')
    xyz = os.path.join(OUTDIR, f'{name}_cell.xyz')
    with open(xyz, 'w') as f:
        f.write(f"{cell.natoms}\n{name} PBC cell, Ly={lvs[1, 1]:.3f}\n")
        for el, p in zip(cell.enames, cell.apos):
            f.write(f"{el:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
    print(f"  wrote {xyz}")
    return cell, lvs, hbonds


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?', default='all')
    ap.add_argument('--hbond', type=float, default=2.8)
    ap.add_argument('--tilt', type=float, default=45.0, help='herringbone tilt [deg], alternate blocks +/-tilt (default: 45)')
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    names = sorted(PBC_CHAIN_ARTS) if args.name == 'all' else [args.name]
    for name in names:
        build_one(name, args.hbond, tilt=args.tilt)
