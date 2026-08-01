#!/usr/bin/env python3
"""Offline reaction-coordinate scan (no GUI). Same geometry path as tests/GUI script.

  python demos/gui_scripts/rc_scan_offline.py
  python demos/gui_scripts/rc_scan_offline.py --name 2Quinolone --dx 0.2
"""
import argparse
import os
import sys

import numpy as np

from spammm import elements as el
from spammm.quantum.hbond_scan import build_ascii_hbond_system
from spammm.topology.hbond_utils import find_hbonds_sys, default_mapping
from spammm.quantum.coordinate_scan import run_pm_neb


def run_offline(name='2Quinolone', dx=0.2, relax_endpoints=True, on_fail='raise', work_dir=None, out_npz=None, verbose=True):
    atoms = build_ascii_hbond_system(name)
    atoms.neighs()
    hbonds = find_hbonds_sys(atoms, bPrint=False)
    if not hbonds:
        raise RuntimeError(f"No H-bonds in {name}")
    mapping = default_mapping(len(hbonds), m=1)
    etype = np.array([el.ELEMENT_DICT[e][0] for e in atoms.enames], dtype=np.int32)
    bonds = np.asarray(atoms.bonds, dtype=np.int32)
    root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'debug', 'rc_scan_offline')
    work_dir = work_dir or os.path.join(root, name)
    ds = run_pm_neb(atoms.enames, atoms.apos, hbonds, mapping, dx=dx, relax_endpoints=relax_endpoints, run_sp=False, etype=etype, bonds=bonds, work_dir=work_dir, verbose=verbose, on_fail=on_fail)
    out_npz = out_npz or os.path.join(root, f'{name}_sym_pm_neb_relaxed.npz')
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    ds.save_npz(out_npz)
    tag = ds.meta.get('scan_type')
    ok = ds.meta.get('endpoints_relaxed', False)
    is_heavy = np.array([e != 'H' for e in atoms.enames])
    heavy_bi = [bi for bi, b in enumerate(bonds) if is_heavy[b[0]] and is_heavy[b[1]]]
    bl_delta = np.max(np.abs(ds.bond_len[-1, heavy_bi] - ds.bond_len[0, heavy_bi])) if heavy_bi else 0.0
    print(f"REVIEW: {name} — {ds.nframes} frames, scan_type={tag}, endpoints_relaxed={ok}")
    print(f"REVIEW: heavy bond max Δ (u=1 vs u=0) = {bl_delta:.4f} Å")
    print(f"REVIEW: E0={ds.meta.get('endpoint_E0_ev')} eV  E1={ds.meta.get('endpoint_E1_ev')} eV")
    print(f"REVIEW: {out_npz}")
    if relax_endpoints and not ok:
        raise RuntimeError("Endpoint DFTB relax failed — see OUT tails above")
    return ds


def main(argv=None):
    p = argparse.ArgumentParser(description='Offline symmetric H-bond RC scan (DFTB pm-NEB relaxed)')
    p.add_argument('--name', default='2Quinolone')
    p.add_argument('--dx', type=float, default=0.2)
    p.add_argument('--no-relax-endpoints', action='store_true')
    p.add_argument('--on-fail', default='raise', choices=['raise', 'skip'])
    p.add_argument('--work-dir', default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args(argv)
    try:
        run_offline(name=args.name, dx=args.dx, relax_endpoints=not args.no_relax_endpoints, on_fail=args.on_fail, work_dir=args.work_dir, out_npz=args.out)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
