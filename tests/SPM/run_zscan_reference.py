#!/usr/bin/env python3
"""Compute E(z) reference curves for CO tip approaching molecules.

For each molecule and each target atom:
  - Rigid z-scan: CO tip (O apex) descends along z above target atom
  - 4 quantum methods: DFTB mio-1-1, DFTB 3ob-3-1, pySCF PBE/6-31G*, pySCF B3LYP/6-31G*
  - Non-uniform z-grid: fine (0.1 Å) near contact, coarser at large distance
  - Save relative interaction energy E(z) - E(z_max) to tests/ref_data/

Usage:
  python tests/SPM/run_zscan_reference.py
  python tests/SPM/run_zscan_reference.py --molecules C2H4 --methods dftb_mio
"""
import os, sys, argparse, time, json, multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.realpath(os.path.join(_THIS_DIR, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATA_DIR = os.path.join(_ROOT, 'data', 'xyz')
REF_DIR = os.path.join(_ROOT, 'tests', 'ref_data', 'Ez_FDBM')
DEBUG_DIR = os.path.join(_ROOT, 'debug', 'zscan_reference')

HAU2EV = 27.211386245988

# CO tip: O at apex (bottom), C above at 1.13 Å
CO_BOND = 1.13
CO_NAMES = ['O', 'C']
CO_POS = np.array([[0, 0, 0], [0, 0, CO_BOND]], dtype=np.float64)

# Molecule definitions: (xyz_file, [(label, atom_index), ...])
# Small molecules
# Small molecules: all 4 methods (DFTB + pySCF)
# Big molecules: DFTB only (pySCF too expensive)
_ALL_METHODS = ['dftb_mio', 'dftb_3ob', 'pyscf_pbe', 'pyscf_b3lyp']
_DFTB_ONLY  = ['dftb_mio', 'dftb_3ob']

MOLECULES = {
    'C2H4': {
        'xyz': 'C2H4.xyz',
        'targets': [('C', 0), ('H', 2)],
        'methods': _ALL_METHODS,
    },
    'CH2O': {
        'xyz': 'CH2O.xyz',
        'targets': [('C', 0), ('O', 1), ('H', 2)],
        'methods': _ALL_METHODS,
    },
    'H2O': {
        'xyz': 'H2O.xyz',
        'targets': [('O', 0)],
        'methods': _ALL_METHODS,
    },
    'NH3': {
        'xyz': 'NH3.xyz',
        'targets': [('N', 0)],
        'methods': _ALL_METHODS,
    },
    'CH2NH': {
        'xyz': 'CH2NH.xyz',
        'targets': [('C', 0), ('N', 1), ('H', 2)],
        'methods': _ALL_METHODS,
    },
    # Big aromatic molecules: DFTB only
    'benzene': {
        'xyz': 'benzene.xyz',
        'targets': [('C', 0), ('H', 6)],
        'methods': _DFTB_ONLY,
    },
    'pyridine': {
        'xyz': 'pyridine.xyz',
        'targets': [('N', 0), ('C', 2)],
        'methods': _DFTB_ONLY,
    },
    'pyrrole': {
        'xyz': 'pyrrole.xyz',
        'targets': [('N', 0), ('C', 2)],
        'methods': _DFTB_ONLY,
    },
    # PTCDA: large aromatic with anhydride groups
    # O idx 26 = carbonyl O (=O) on left side
    # O idx 24 = bridging O (-O-) on left side
    # C idx 11 = C in carboxylic/anhydride group (left)
    # C idx 6  = C near center of perylene core
    'PTCDA': {
        'xyz': 'PTCDA.xyz',
        'targets': [('O_eq', 26), ('O_br', 24), ('C_anh', 11), ('C_core', 6)],
        'methods': _DFTB_ONLY,
    },
}

# Method definitions
METHODS = {
    'dftb_mio':   {'type': 'dftb',   'basis': 'mio-1-1'},
    'dftb_3ob':   {'type': 'dftb',   'basis': '3ob-3-1'},
    'pyscf_pbe':  {'type': 'pyscf',  'basis': '6-31g*', 'xc': 'pbe'},
    'pyscf_b3lyp':{'type': 'pyscf',  'basis': '6-31g*', 'xc': 'b3lyp'},
}


def load_molecule(xyz_file):
    """Load molecule from XYZ. Returns (atomPos, enames)."""
    from spammm.atomicUtils import load_xyz
    path = os.path.join(DATA_DIR, xyz_file)
    pos, _, names, _, _ = load_xyz(path)
    return np.array(pos, dtype=np.float64), list(names)


def run_dftb_singlepoint(combined_names, combined_pos, sk_prefix, work_dir):
    """Run DFTB+ single-point, return energy in eV. Raises on failure."""
    from spammm.quantum.DFTB_utils import run_dftb_sp
    e_ha = run_dftb_sp(work_dir, combined_names, combined_pos, sk_prefix)
    return e_ha * HAU2EV


def run_pyscf_singlepoint(combined_names, combined_pos, basis, xc=None, work_dir=None, dm_prev=None):
    """Run pySCF single-point, return (energy_eV, dm) for density reuse."""
    from pyscf import gto, scf, dft
    atom_str = '\n'.join([f'{e} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}'
                          for e, p in zip(combined_names, combined_pos)])
    mol = gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0, spin=0, charge=0)
    if xc is None:
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = xc
    mf.verbose = 0
    mf.max_memory = 4000  # 4 GB
    mf.direct_scf = True  # avoid storing full integrals
    # Reuse density matrix from previous z-point as initial guess
    if dm_prev is not None:
        mf.kernel(dm_prev)
    else:
        mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    return mf.e_tot * HAU2EV, mf.make_rdm1()


def _compute_zscan_chunk(mol_pos, target_pos, combined_names, z_chunk, method,
                         out_dir, chunk_id, mol_name, target_label, method_name):
    """Worker function (must be module-level for multiprocessing).

    Computes one contiguous z-chunk. For pySCF, density matrix is reused
    within the chunk. Returns (chunk_id, z_chunk, energies_chunk).
    """
    mol_pos = np.array(mol_pos, dtype=np.float64)
    target_pos = np.array(target_pos, dtype=np.float64)
    z_chunk = np.array(z_chunk, dtype=np.float64)
    energies = []
    failed_z = []
    dm_prev = None
    for z in z_chunk:
        o_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + z])
        c_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + z + CO_BOND])
        co_pos = np.array([o_pos, c_pos])
        combined_pos = np.vstack([mol_pos, co_pos])
        try:
            if method['type'] == 'dftb':
                from spammm.quantum.DFTB_utils import SK_PATHS
                sk = SK_PATHS[method['basis']]
                wd = os.path.join(out_dir, f'chunk_{chunk_id}', f'z_{z:.2f}')
                e_ev = run_dftb_singlepoint(combined_names, combined_pos, sk, wd)
            elif method['type'] == 'pyscf':
                e_ev, dm_prev = run_pyscf_singlepoint(combined_names, combined_pos,
                                                      method['basis'], method.get('xc'),
                                                      dm_prev=dm_prev)
            else:
                raise ValueError(f"Unknown method type: {method['type']}")
        except Exception as exc:
            energies.append(np.nan)
            failed_z.append(z)
            dm_prev = None
            continue
        energies.append(e_ev)
    return chunk_id, z_chunk, np.array(energies), failed_z


def split_z_grid(z_distances, n_chunks):
    """Split z-grid into contiguous chunks of approximately equal size."""
    n = len(z_distances)
    if n_chunks >= n:
        return [[i, i+1] for i in range(n)]
    chunk_size = n // n_chunks
    extra = n % n_chunks
    splits = []
    start = 0
    for i in range(n_chunks):
        size = chunk_size + (1 if i < extra else 0)
        if size == 0:
            continue
        splits.append([start, start + size])
        start += size
    return splits


def run_zscan(mol_name, mol_pos, mol_names, target_idx, target_label,
              method_name, method, z_distances, out_dir, n_workers=None):
    """Run z-scan for one (molecule, target_atom, method) combination.

    Splits the z-grid into contiguous chunks, runs chunks in parallel subprocesses.
    Within a chunk, density matrix is reused for pySCF.
    Returns (z_vals, e_rel_vals).
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, f'zscan_cache.npz')

    target_pos = mol_pos[target_idx]
    combined_names = list(mol_names) + CO_NAMES

    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        if cache['z'].shape == z_distances.shape and np.allclose(cache['z'], z_distances):
            print(f"  [CACHE] {mol_name}/{target_label}/{method_name}: {len(z_distances)} points")
            return cache['z'], cache['e_rel']
        else:
            print(f"  [CACHE MISS] z-range mismatch, recomputing")

    n_workers = n_workers or max(1, os.cpu_count() - 1)
    chunk_indices = split_z_grid(z_distances, n_workers)
    print(f"  [{method_name}] {mol_name}/{target_label} (atom {target_idx} at {target_pos}): {len(z_distances)} z-points, {len(chunk_indices)} chunks on {n_workers} workers")

    tasks = []
    for cid, (lo, hi) in enumerate(chunk_indices):
        z_chunk = z_distances[lo:hi]
        tasks.append((mol_pos, target_pos, combined_names, z_chunk, method,
                      out_dir, cid, mol_name, target_label, method_name))

    chunks = []
    if len(tasks) == 1:
        chunks.append(_compute_zscan_chunk(*tasks[0]))
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            chunks = list(pool.map(_compute_zscan_chunk_worker, tasks))

    # Sort chunks by chunk_id and merge
    chunks.sort(key=lambda x: x[0])
    z_list = np.concatenate([c[1] for c in chunks])
    energies = np.concatenate([c[2] for c in chunks])
    failed_z = []
    for c in chunks:
        failed_z.extend(c[3])

    # Verify order matches z_distances
    sort_idx = np.argsort(z_list)
    z_list = z_list[sort_idx]
    energies = energies[sort_idx]
    assert np.allclose(z_list, z_distances), "Chunk merge produced wrong z order"

    valid_mask = ~np.isnan(energies)
    if not valid_mask.any():
        raise RuntimeError(f"All z-points failed for {mol_name}/{target_label}/{method_name}")
    e_ref = energies[valid_mask][-1]
    e_rel = energies - e_ref
    if failed_z:
        print(f"  [WARNING] {len(failed_z)} z-points failed: {failed_z}")

    np.savez(cache_path, z=z_distances, e_abs=energies, e_rel=e_rel)
    print(f"  => {method_name}/{mol_name}/{target_label}: E_rel range [{np.nanmin(e_rel):.4f}, {np.nanmax(e_rel):.4f}] eV")
    return z_distances, e_rel


def _compute_zscan_chunk_worker(task):
    """Wrapper to unpack the tuple for ProcessPoolExecutor."""
    return _compute_zscan_chunk(*task)


def save_reference(mol_name, target_label, method_name, z, e_rel, e_abs=None):
    """Save reference curve to tests/ref_data/."""
    os.makedirs(REF_DIR, exist_ok=True)
    fname = f'zscan_{mol_name}_{method_name}_{target_label}.npz'
    path = os.path.join(REF_DIR, fname)
    save_dict = {'z': z, 'e_rel': e_rel}
    if e_abs is not None:
        save_dict['e_abs'] = e_abs
    np.savez(path, **save_dict)
    print(f"  SAVED: {path}")


def make_z_grid():
    """Non-uniform z-grid: fine near contact, coarser at large distance.
    1.5..3.0 dz=0.1, 3.0..5.0 dz=0.2, 5.0..8.0 dz=0.5
    """
    z1 = np.arange(1.5, 3.0, 0.1)
    z2 = np.arange(3.0, 5.0, 0.2)
    z3 = np.arange(5.0, 8.01, 0.5)
    z = np.unique(np.round(np.concatenate([z1, z2, z3]), 4))
    return z


def main():
    parser = argparse.ArgumentParser(description='Compute E(z) reference curves for Pauli fitting')
    parser.add_argument('--molecules', type=str, default='all',
                        help='Comma-separated molecule names (default: all)')
    parser.add_argument('--methods', type=str, default='all',
                        help='Comma-separated method names (default: all)')
    parser.add_argument('--save_ref', action='store_true', default=True,
                        help='Save results to tests/ref_data/')
    args = parser.parse_args()

    z_distances = make_z_grid()
    print(f"Z-scan: {len(z_distances)} points from {z_distances[0]:.2f} to {z_distances[-1]:.2f} Å")
    print(f"  Grid: 1.5-3.0 dz=0.1 ({len(z_distances[z_distances<3.0])} pts), "
          f"3.0-5.0 dz=0.2 ({len(z_distances[(z_distances>=3.0)&(z_distances<5.0)])} pts), "
          f"5.0-8.0 dz=0.5 ({len(z_distances[z_distances>=5.0])} pts)")

    mol_names = list(MOLECULES.keys()) if args.molecules == 'all' else args.molecules.split(',')

    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(REF_DIR, exist_ok=True)

    all_results = {}

    for mol_name in mol_names:
        if mol_name not in MOLECULES:
            print(f"WARNING: Unknown molecule {mol_name}, skipping")
            continue
        mol_info = MOLECULES[mol_name]
        mol_pos, mol_atom_names = load_molecule(mol_info['xyz'])
        print(f"\n{'='*60}")
        print(f"Molecule: {mol_name} ({len(mol_atom_names)} atoms)")
        print(f"{'='*60}")

        # Use per-molecule method list (big molecules = DFTB only)
        mol_methods = mol_info.get('methods', list(METHODS.keys()))
        if args.methods != 'all':
            mol_methods = [m for m in mol_methods if m in args.methods.split(',')]

        n_workers = max(1, os.cpu_count() - 1)
        for target_label, target_idx in mol_info['targets']:
            for method_name in mol_methods:
                if method_name not in METHODS:
                    print(f"WARNING: Unknown method {method_name}, skipping")
                    continue
                method = METHODS[method_name]

                out_dir = os.path.join(DEBUG_DIR, mol_name, f'{target_label}_{method_name}')
                z, e_rel = run_zscan(mol_name, mol_pos, mol_atom_names, target_idx,
                                     target_label, method_name, method, z_distances, out_dir,
                                     n_workers=n_workers)

                key = f'{mol_name}/{target_label}/{method_name}'
                all_results[key] = {'z': z, 'e_rel': e_rel}

                if args.save_ref:
                    save_reference(mol_name, target_label, method_name, z, e_rel)

    # Save summary JSON
    summary_path = os.path.join(REF_DIR, 'zscan_summary.json')
    summary = {
        'description': 'E(z) reference curves for CO tip approaching molecules',
        'z_grid': list(z_distances),
        'n_points': len(z_distances),
        'co_bond': CO_BOND,
        'n_workers': n_workers,
        'methods': {k: v for k, v in METHODS.items()},
        'molecules': {k: v for k, v in MOLECULES.items() if k in mol_names},
        'curves': list(all_results.keys()),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")
    print(f"Total curves: {len(all_results)}")


if __name__ == '__main__':
    main()
