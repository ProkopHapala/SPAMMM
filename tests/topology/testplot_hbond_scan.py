#!/usr/bin/env python3
"""Rigid DFTB H-bond proton-transfer scan for ASCII art molecules with ':' H-bonds.

Outputs: debug/test_hbond_scan/hbond_<name>_p<pair>.{png,xyz}
Run: python tests/topology/testplot_hbond_scan.py
     python tests/topology/testplot_hbond_scan.py --name 2Quinolone --step 0.1
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spammm.quantum.hbond_scan import (
    DEBUG_DIR, DEFAULT_DS, build_ascii_hbond_system, identify_hbond_from_ascii, run_hbond_transfer_scan,
    ascii_examples_with_hbonds, save_hbond_scan_artifacts,
)


def run_one(name, pair_idx=0, ds=DEFAULT_DS, sk_set=None, work_root=None):
    atoms = build_ascii_hbond_system(name)
    ih, ido, iac = identify_hbond_from_ascii(atoms, pair_idx=pair_idx)
    label = f"{atoms.enames[ido]}{ido}-H{ih}...{atoms.enames[iac]}{iac}"
    print(f"\n=== {name} pair {pair_idx}: {label}  ds={ds}Å ===")
    work_dir = os.path.join(work_root or DEBUG_DIR, f'{name}_p{pair_idx}')
    result = run_hbond_transfer_scan(atoms.enames, atoms.apos, ih, ido, iac, ds=ds, sk_set=sk_set, work_dir=work_dir, verbose=True, on_fail='skip')
    ok = np.isfinite(result['energies_ev'])
    if not ok.any():
        print("  ERROR: no converged points")
        return result
    e_min_i = int(np.nanargmin(result['energies_ev']))
    s = result['s_axis']
    print(f"  E_min at s={s[e_min_i]:.3f}Å  barrier={np.nanmax(result['rel_ev']):.3f} eV  converged={ok.sum()}/{len(ok)}  npts={len(s)}")
    png, xyz = save_hbond_scan_artifacts(result, atoms, name, pair_idx=pair_idx, out_dir=DEBUG_DIR)
    print(f"REVIEW: {png}")
    print(f"REVIEW: {xyz}")
    return result


def main():
    parser = argparse.ArgumentParser(description='DFTB rigid H-bond transfer scan for ASCII molecules')
    parser.add_argument('--name', default=None, help='ASCII example name (default: all with : H-bonds)')
    parser.add_argument('--pair', type=int, default=0, help='H-bond pair index')
    parser.add_argument('--step', type=float, default=DEFAULT_DS, help='Path step along D-A axis [Å] (default: 0.1)')
    parser.add_argument('--sk_set', default=None, help='DFTB SK set (default: from config)')
    parser.add_argument('--all_pairs', action='store_true', help='Scan every H-bond pair in molecule')
    args = parser.parse_args()
    os.makedirs(DEBUG_DIR, exist_ok=True)
    names = [args.name] if args.name else ascii_examples_with_hbonds()
    print(f"H-bond ASCII examples: {names}")
    for name in names:
        atoms = build_ascii_hbond_system(name)
        n_pairs = len(getattr(atoms, 'hbonds_ascii', []) or [])
        pairs = range(n_pairs) if args.all_pairs else [args.pair]
        for p in pairs:
            if p >= n_pairs:
                print(f"  SKIP {name} pair {p}: only {n_pairs} H-bonds")
                continue
            run_one(name, pair_idx=p, ds=args.step, sk_set=args.sk_set)


if __name__ == '__main__':
    main()
