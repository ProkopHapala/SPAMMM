#!/usr/bin/env python3
"""Convert existing .npz reference curves to ASCII .dat files and rename .npz to include atom index.

Does NOT recompute anything — just reads existing .npz and writes .dat + renamed .npz.

Usage:
  python tests/SPM/convert_npz_to_dat.py
"""
import os, sys, glob, shutil
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.realpath(os.path.join(_THIS_DIR, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
REF_DIR = os.path.join(_ROOT, 'tests', 'ref_data')
EZ_DIR = os.path.join(REF_DIR, 'Ez_FDBM')

# Import MOLECULES to get atom indices
from tests.SPM.run_zscan_reference import MOLECULES, METHODS

def build_filename_lookup():
    """Build mapping: old_filename -> (mol, method, site_label, atom_idx, new_filename)."""
    lookup = {}
    for mol_name, mol_info in MOLECULES.items():
        for target_label, atom_idx in mol_info['targets']:
            for method_name in METHODS:
                old_fname = f'zscan_{mol_name}_{method_name}_{target_label}.npz'
                new_label = f'{target_label}{atom_idx}'
                new_fname = f'zscan_{mol_name}_{method_name}_{new_label}.npz'
                lookup[old_fname] = (mol_name, method_name, target_label, atom_idx, new_fname)
    return lookup

def main():
    lookup = build_filename_lookup()
    npz_files = sorted(glob.glob(os.path.join(REF_DIR, 'zscan_*.npz')))
    print(f"Found {len(npz_files)} .npz files in {REF_DIR}")
    os.makedirs(EZ_DIR, exist_ok=True)

    n_converted = 0
    n_renamed = 0
    for path in npz_files:
        fname = os.path.basename(path)
        if fname not in lookup:
            print(f"  SKIP (unknown): {fname}")
            continue
        mol_name, method_name, site_label, atom_idx, new_fname = lookup[fname]
        new_path = os.path.join(REF_DIR, new_fname)

        # Load data
        data = np.load(path)
        z = data['z']
        e_rel = data['e_rel']
        e_abs = data['e_abs'] if 'e_abs' in data else None

        # Write .dat file to Ez_FDBM
        dat_fname = new_fname.replace('.npz', '.dat')
        dat_path = os.path.join(EZ_DIR, dat_fname)
        header = f"# z-scan: {mol_name} / {site_label} (atom {atom_idx}) / {method_name}\n"
        header += f"# CO tip: O apex at target + z, C at target + z + 1.13 Å\n"
        header += f"# E_rel = E(z) - E(z_max) in eV\n"
        header += f"z[A]   E_rel[eV]"
        if e_abs is not None:
            header += "   E_abs[eV]"
        header += "\n"
        with open(dat_path, 'w') as f:
            f.write(header)
            for i in range(len(z)):
                line = f"{z[i]:.4f}   {e_rel[i]:.6f}"
                if e_abs is not None:
                    line += f"   {e_abs[i]:.6f}"
                f.write(line + "\n")
        print(f"  DAT: {dat_fname}")
        n_converted += 1

        # Rename .npz if needed
        if new_fname != fname:
            if os.path.exists(new_path):
                print(f"  NPZ already renamed: {new_fname}")
            else:
                shutil.copy2(path, new_path)
                print(f"  NPZ: {fname} -> {new_fname}")
                n_renamed += 1

    print(f"\nDone: {n_converted} .dat files written to {EZ_DIR}, {n_renamed} .npz files renamed")
    print(f"Original .npz files preserved (not deleted)")

if __name__ == '__main__':
    main()
