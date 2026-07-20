#!/usr/bin/env python3
"""Compute E(z) reference curves for CO tip approaching molecules — **one CLI**.

SSOT for AFM CO rigid z-scans in SPAMMM. Do not add parallel ad-hoc scan drivers.

For each molecule × target atom × method:
  - Tip: CO O-apex above the **target atom** (site-correct xy)
  - Distance array: ``make_z_grid(z_max)`` or ``--distances-file`` (change only this for grid)
  - pySCF: DM warm-start along z via ``spammm.quantum.pySCF_utils-new.run_co_zscan``
      * prefer local-fork **GPU** OpenCL when available
      * else stock **CPU** pySCF (same API)
  - DFTB: mio / 3ob (unchanged)
  - Saves E_rel = E(z)−E(z_max) under ``tests/ref_data/Ez_FDBM/``

Usage:
  python tests/SPM/run_zscan_reference.py --molecules PTCDA --methods pyscf_pbe --z-max 15
  python tests/SPM/run_zscan_reference.py --molecules C2H4 --methods dftb_mio
  python tests/SPM/run_zscan_reference.py --distances-file my_z.dat --backend cpu

Related (pySCF fork, dimer XC-path benches — not AFM CO refs):
  /home/prokop/git/pyscf/expamples_prokop/profile_dimer_scan.py
"""
import os, sys, argparse, json, importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.realpath(os.path.join(_THIS_DIR, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATA_DIR = os.path.join(_ROOT, 'data', 'xyz')
REF_DIR = os.path.join(_ROOT, 'tests', 'ref_data', 'Ez_FDBM')
REF_DIR_GPU = os.path.join(_ROOT, 'tests', 'ref_data', 'CO_scan_pyscf_gpu')
DEBUG_DIR = os.path.join(_ROOT, 'debug', 'zscan_reference')

HAU2EV = 27.211386245988
CO_BOND = 1.13
CO_NAMES = ['O', 'C']

# Load hyphenated module (SSOT workhorse)
_spec = importlib.util.spec_from_file_location(
    'pySCF_utils_new', os.path.join(_ROOT, 'spammm', 'quantum', 'pySCF_utils-new.py'))
_pu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pu)

# Small molecules: DFTB + stock pySCF (6-31g*) + optional GPU PBE/def2-SVP
# Large aromatics: DFTB + GPU PBE (auto→CPU fallback)
_SMALL = ['dftb_mio', 'dftb_3ob', 'pyscf_pbe', 'pyscf_b3lyp', 'pyscf_gpu_pbe']
_LARGE = ['dftb_mio', 'dftb_3ob', 'pyscf_gpu_pbe']

MOLECULES = {
    'C2H4': {'xyz': 'C2H4.xyz', 'targets': [('C', 0), ('H', 2)], 'methods': _SMALL},
    'CH2O': {'xyz': 'CH2O.xyz', 'targets': [('C', 0), ('O', 1), ('H', 2)], 'methods': _SMALL},
    'H2O': {'xyz': 'H2O.xyz', 'targets': [('O', 0)], 'methods': _SMALL},
    'NH3': {'xyz': 'NH3.xyz', 'targets': [('N', 0)], 'methods': _SMALL},
    'CH2NH': {'xyz': 'CH2NH.xyz', 'targets': [('C', 0), ('N', 1), ('H', 2)], 'methods': _SMALL},
    'benzene': {'xyz': 'benzene.xyz', 'targets': [('C', 0), ('H', 6)], 'methods': _LARGE},
    'pyridine': {'xyz': 'pyridine.xyz', 'targets': [('N', 0), ('C', 2)], 'methods': _LARGE},
    'pyrrole': {'xyz': 'pyrrole.xyz', 'targets': [('N', 0), ('C', 2)], 'methods': _LARGE},
    'PTCDA': {
        'xyz': 'PTCDA.xyz',
        'targets': [('O_eq', 26), ('O_br', 24), ('C_anh', 11), ('C_core', 6)],
        'methods': _LARGE,
    },
    'pentacene': {
        'xyz': 'pentacene.xyz',
        'targets': [('C_end', 0), ('C_mid', 10), ('C_inner', 16)],
        'methods': _LARGE,
    },
}

# prefer_backend: 'auto' → GPU if NVIDIA OpenCL else stock CPU; 'cpu' forces stock
METHODS = {
    'dftb_mio': {'type': 'dftb', 'basis': 'mio-1-1'},
    'dftb_3ob': {'type': 'dftb', 'basis': '3ob-3-1'},
    # Legacy small-mol Ez_FDBM parity (stock CPU, 6-31g*)
    'pyscf_pbe': {
        'type': 'pyscf', 'basis': '6-31g*', 'xc': 'pbe',
        'prefer_backend': 'cpu', 'profile': None, 'ref_subdir': 'Ez_FDBM',
    },
    'pyscf_b3lyp': {
        'type': 'pyscf', 'basis': '6-31g*', 'xc': 'b3lyp',
        'prefer_backend': 'cpu', 'profile': None, 'ref_subdir': 'Ez_FDBM',
    },
    # Production AFM DFT refs: GPU preferred, stock CPU fallback; def2-SVP
    'pyscf_gpu_pbe': {
        'type': 'pyscf', 'basis': 'def2-SVP', 'xc': 'PBE',
        'prefer_backend': 'auto', 'profile': _pu.DEFAULT_GPU_PROFILE,
        'ref_subdir': 'CO_scan_pyscf_gpu', 'max_memory_mb': 20000,
    },
}


def load_molecule(xyz_file):
    from spammm.atomicUtils import load_xyz
    path = os.path.join(DATA_DIR, xyz_file)
    pos, _, names, _, _ = load_xyz(path)
    return np.array(pos, dtype=np.float64), list(names)


def load_distances_file(path):
    zs = []
    with open(path) as f:
        for line in f:
            s = line.split('#', 1)[0].strip()
            if not s:
                continue
            zs.append(float(s.split()[0]))
    if not zs:
        raise ValueError(f'no distances in {path}')
    return np.unique(np.round(np.asarray(zs, dtype=np.float64), 4))


def run_dftb_singlepoint(combined_names, combined_pos, sk_prefix, work_dir):
    from spammm.quantum.DFTB_utils import run_dftb_sp
    e_ha = run_dftb_sp(work_dir, combined_names, combined_pos, sk_prefix)
    return e_ha * HAU2EV


def _compute_dftb_zscan_chunk(mol_pos, target_pos, combined_names, z_chunk, method,
                              out_dir, chunk_id):
    """DFTB worker (module-level for multiprocessing). No DM reuse across chunks."""
    mol_pos = np.asarray(mol_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    z_chunk = np.asarray(z_chunk, dtype=np.float64)
    from spammm.quantum.DFTB_utils import SK_PATHS
    sk = SK_PATHS[method['basis']]
    energies, failed_z = [], []
    for z in z_chunk:
        o_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + z])
        c_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + z + CO_BOND])
        combined_pos = np.vstack([mol_pos, [o_pos, c_pos]])
        try:
            wd = os.path.join(out_dir, f'chunk_{chunk_id}', f'z_{z:.2f}')
            energies.append(run_dftb_singlepoint(combined_names, combined_pos, sk, wd))
        except Exception:
            energies.append(np.nan)
            failed_z.append(z)
    return chunk_id, z_chunk, np.array(energies), failed_z


def _dftb_chunk_worker(task):
    return _compute_dftb_zscan_chunk(*task)


def split_z_grid(z_distances, n_chunks):
    n = len(z_distances)
    if n_chunks >= n:
        return [[i, i + 1] for i in range(n)]
    chunk_size, extra = n // n_chunks, n % n_chunks
    splits, start = [], 0
    for i in range(n_chunks):
        size = chunk_size + (1 if i < extra else 0)
        if size == 0:
            continue
        splits.append([start, start + size])
        start += size
    return splits


def run_zscan_dftb(mol_name, mol_pos, mol_names, target_idx, target_label,
                   method_name, method, z_distances, out_dir, n_workers=None):
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'zscan_cache.npz')
    target_pos = mol_pos[target_idx]
    combined_names = list(mol_names) + CO_NAMES

    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        if cache['z'].shape == z_distances.shape and np.allclose(cache['z'], z_distances):
            print(f"  [CACHE] {mol_name}/{target_label}/{method_name}: {len(z_distances)} points")
            return cache['z'], cache['e_rel'], cache.get('e_abs')

    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    chunk_indices = split_z_grid(z_distances, n_workers)
    print(f"  [{method_name}] {mol_name}/{target_label} atom{target_idx}: "
          f"{len(z_distances)} z, {len(chunk_indices)} chunks")

    tasks = []
    for cid, (lo, hi) in enumerate(chunk_indices):
        tasks.append((mol_pos, target_pos, combined_names, z_distances[lo:hi], method, out_dir, cid))

    if len(tasks) == 1:
        chunks = [_compute_dftb_zscan_chunk(*tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            chunks = list(pool.map(_dftb_chunk_worker, tasks))

    chunks.sort(key=lambda x: x[0])
    z_list = np.concatenate([c[1] for c in chunks])
    energies = np.concatenate([c[2] for c in chunks])
    failed_z = [z for c in chunks for z in c[3]]
    order = np.argsort(z_list)
    z_list, energies = z_list[order], energies[order]
    assert np.allclose(z_list, z_distances)

    valid = ~np.isnan(energies)
    if not valid.any():
        raise RuntimeError(f"All z-points failed for {mol_name}/{target_label}/{method_name}")
    e_rel = energies - energies[valid][-1]
    if failed_z:
        print(f"  [WARNING] failed z: {failed_z}")
    np.savez(cache_path, z=z_distances, e_abs=energies, e_rel=e_rel)
    print(f"  => {method_name}/{mol_name}/{target_label}: E_rel [{np.nanmin(e_rel):.4f}, {np.nanmax(e_rel):.4f}] eV")
    return z_distances, e_rel, energies


def run_zscan_pyscf(mol_name, mol_pos, mol_names, target_idx, target_label,
                    method_name, method, z_distances, out_dir, backend_override=None,
                    E_mol_eV=None, E_CO_eV=None):
    """Sequential CO z-scan via run_co_zscan (DM warm-start). GPU auto or stock CPU."""
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, 'zscan_cache.npz')
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        if cache['z'].shape == z_distances.shape and np.allclose(cache['z'], z_distances):
            print(f"  [CACHE] {mol_name}/{target_label}/{method_name}: {len(z_distances)} points")
            return cache['z'], cache['e_rel'], cache.get('e_abs'), None

    prefer = backend_override or method.get('prefer_backend', 'auto')
    backend = _pu.resolve_backend(prefer)
    profile = method.get('profile') or _pu.DEFAULT_GPU_PROFILE
    kw = {}
    if method.get('max_memory_mb'):
        kw['max_memory_mb'] = method['max_memory_mb']
        kw['df_storage'] = 'incore'

    print(f"  [{method_name}] {mol_name}/{target_label} atom{target_idx}: "
          f"{len(z_distances)} z  backend={backend}  basis={method['basis']}  xc={method['xc']}")
    if backend == 'gpu':
        print(f"           profile={profile}  (DM warm-start along z)")

    r = _pu.run_co_zscan(
        mol_names, mol_pos, target_idx, z_distances,
        basis=method['basis'], xc=method['xc'], backend=backend, profile=profile,
        apex='O', bond=CO_BOND, warm_start=True, verbosity=1,
        E_mol_eV=E_mol_eV, E_CO_eV=E_CO_eV, **kw,
    )
    e_abs = r['E_abs_eV']
    e_int = r['E_int_eV']
    e_rel = e_int - e_int[-1]
    np.savez(cache_path, z=z_distances, e_abs=e_abs, e_rel=e_rel, e_int=e_int,
             E_mol_eV=r['E_mol_eV'], E_CO_eV=r['E_CO_eV'],
             cycles=np.asarray(r['cycles']), wall_s=np.asarray(r['wall_s']),
             backend=backend)
    print(f"  => {method_name}/{mol_name}/{target_label}: E_int [{e_int.min():.4f}, {e_int.max():.4f}] eV  "
          f"backend={backend}")
    return z_distances, e_rel, e_abs, r


def ref_dir_for(method):
    sub = method.get('ref_subdir', 'Ez_FDBM')
    return os.path.join(_ROOT, 'tests', 'ref_data', sub)


def save_reference_dat(mol_name, target_label, target_idx, method_name, method, z, e_rel,
                       e_abs=None, e_int=None, scan_meta=None):
    """Human-readable .dat (Ez_FDBM style). Also .npz cache name for legacy tools."""
    out_dir = ref_dir_for(method)
    os.makedirs(out_dir, exist_ok=True)
    # Filename: include atom idx for uniqueness (matches existing Ez_FDBM PTCDA names)
    label = f'{target_label}{target_idx}' if not target_label[-1:].isdigit() else target_label
    # Keep legacy small-mol names without forcing idx when already in label pattern used before
    if method_name in ('pyscf_pbe', 'pyscf_b3lyp', 'dftb_mio', 'dftb_3ob'):
        # Historical: zscan_C2H4_pyscf_pbe_C0.dat — label already has style from caller
        fname_base = f'zscan_{mol_name}_{method_name}_{target_label}{target_idx}'
        # Prefer matching existing if target_label is like C and we used C0
        fname_base = f'zscan_{mol_name}_{method_name}_{target_label}{target_idx}'
    else:
        fname_base = f'zscan_{mol_name}_{method_name}_{target_label}{target_idx}'

    npz_path = os.path.join(out_dir, fname_base + '.npz')
    save = {'z': z, 'e_rel': e_rel}
    if e_abs is not None:
        save['e_abs'] = e_abs
    if e_int is not None:
        save['e_int'] = e_int
    np.savez(npz_path, **save)

    dat_path = os.path.join(out_dir, fname_base + '.dat')
    backend = (scan_meta or {}).get('backend', method.get('prefer_backend'))
    with open(dat_path, 'w') as f:
        f.write(f'# z-scan: {mol_name} / {target_label} (atom {target_idx}) / {method_name}\n')
        f.write(f'# CO tip: O apex at target + z, C at target + z + {CO_BOND} Å (site-correct xy)\n')
        f.write(f'# method: basis={method["basis"]} xc={method.get("xc")} backend={backend}\n')
        if method.get('profile') and backend == 'gpu':
            f.write(f'# GPU profile: {method["profile"]}\n')
        f.write('# UNITS: z[A], energies in eV (PySCF mf.e_tot is Hartree; × HAU2EV)\n')
        f.write('# E_rel = E(z) - E(z_max) in eV\n')
        if e_int is not None:
            f.write('# E_int = E(mol+CO@z) - E(mol) - E(CO) in eV\n')
            f.write('z[A]   E_rel[eV]   E_int[eV]\n')
            for i in range(len(z)):
                f.write(f'{z[i]:.4f}   {e_rel[i]:.8f}   {e_int[i]:.8f}\n')
        else:
            f.write('z[A]   E_rel[eV]\n')
            for i in range(len(z)):
                f.write(f'{z[i]:.4f}   {e_rel[i]:.8f}\n')
    print(f"  SAVED: {dat_path}")
    return dat_path


def main():
    parser = argparse.ArgumentParser(
        description='CO tip E(z) references — single entry (GPU pySCF preferred, stock CPU fallback)')
    parser.add_argument('--molecules', type=str, default='all')
    parser.add_argument('--methods', type=str, default='all')
    parser.add_argument('--z-max', type=float, default=8.0,
                        help='Non-uniform grid upper bound (Å). Use 15 for full AFM well.')
    parser.add_argument('--distances-file', type=str, default=None,
                        help='Override grid: one z[Å] per line (same idea as profile_dimer_scan)')
    parser.add_argument('--backend', type=str, default=None, choices=['auto', 'gpu', 'cpu', 'smalldft'],
                        help='Override pySCF backend (default: per-method prefer_backend)')
    parser.add_argument('--save_ref', action='store_true', default=True)
    parser.add_argument('--no-cache', action='store_true', help='Ignore debug zscan_cache.npz')
    args = parser.parse_args()

    if args.distances_file:
        z_distances = load_distances_file(args.distances_file)
        print(f"Z-scan from file: {len(z_distances)} points [{z_distances[0]:.2f} … {z_distances[-1]:.2f}] Å")
    else:
        z_distances = _pu.make_z_grid(args.z_max)
        print(f"Z-scan: {len(z_distances)} points [{z_distances[0]:.2f} … {z_distances[-1]:.2f}] Å  (z_max={args.z_max})")

    print(f"pySCF GPU available: {_pu.gpu_opencl_available()}  (auto → gpu|cpu)")

    mol_names = list(MOLECULES.keys()) if args.molecules == 'all' else args.molecules.split(',')
    os.makedirs(DEBUG_DIR, exist_ok=True)

    all_results = {}
    shared_refs = {}  # (basis,xc,backend) → (E_mol, E_CO) per molecule geometry

    for mol_name in mol_names:
        if mol_name not in MOLECULES:
            print(f"WARNING: Unknown molecule {mol_name}, skipping")
            continue
        mol_info = MOLECULES[mol_name]
        mol_pos, mol_atom_names = load_molecule(mol_info['xyz'])
        print(f"\n{'='*60}\nMolecule: {mol_name} ({len(mol_atom_names)} atoms)\n{'='*60}")

        mol_methods = list(mol_info.get('methods', list(METHODS.keys())))
        if args.methods != 'all':
            # Explicit CLI list (may include methods not in the molecule default list)
            mol_methods = [m for m in args.methods.split(',') if m in METHODS]

        n_workers = max(1, (os.cpu_count() or 2) - 1)
        for target_label, target_idx in mol_info['targets']:
            for method_name in mol_methods:
                method = METHODS[method_name]
                out_dir = os.path.join(DEBUG_DIR, mol_name, f'{target_label}_{method_name}')
                if args.no_cache:
                    cache = os.path.join(out_dir, 'zscan_cache.npz')
                    if os.path.isfile(cache):
                        os.remove(cache)

                if method['type'] == 'dftb':
                    z, e_rel, e_abs = run_zscan_dftb(
                        mol_name, mol_pos, mol_atom_names, target_idx, target_label,
                        method_name, method, z_distances, out_dir, n_workers=n_workers)
                    scan_meta = None
                    e_int = None
                elif method['type'] == 'pyscf':
                    # Share E_mol/E_CO across sites for same molecule+method
                    ref_key = (mol_name, method_name)
                    E_mol = E_CO = None
                    if ref_key in shared_refs:
                        E_mol, E_CO = shared_refs[ref_key]
                    z, e_rel, e_abs, r = run_zscan_pyscf(
                        mol_name, mol_pos, mol_atom_names, target_idx, target_label,
                        method_name, method, z_distances, out_dir,
                        backend_override=args.backend, E_mol_eV=E_mol, E_CO_eV=E_CO)
                    if r is not None:
                        shared_refs[ref_key] = (r['E_mol_eV'], r['E_CO_eV'])
                        e_int = r['E_int_eV']
                        scan_meta = {'backend': r['backend']}
                    else:
                        e_int = None
                        scan_meta = None
                else:
                    raise ValueError(method['type'])

                key = f'{mol_name}/{target_label}/{method_name}'
                all_results[key] = {'z': z.tolist() if hasattr(z, 'tolist') else z, 'e_rel_minmax': [float(np.nanmin(e_rel)), float(np.nanmax(e_rel))]}
                if args.save_ref:
                    save_reference_dat(mol_name, target_label, target_idx, method_name, method,
                                       z, e_rel, e_abs=e_abs, e_int=e_int, scan_meta=scan_meta)

    summary_path = os.path.join(REF_DIR, 'zscan_summary.json')
    summary = {
        'description': 'E(z) CO tip refs — single CLI run_zscan_reference.py',
        'z_grid': list(map(float, z_distances)),
        'n_points': len(z_distances),
        'co_bond': CO_BOND,
        'gpu_available': _pu.gpu_opencl_available(),
        'methods': {k: {kk: vv for kk, vv in v.items() if kk != 'profile' or True} for k, v in METHODS.items()},
        'molecules': {k: MOLECULES[k] for k in mol_names if k in MOLECULES},
        'curves': list(all_results.keys()),
    }
    # JSON-serialize methods cleanly
    summary['methods'] = {
        k: {'type': v['type'], 'basis': v.get('basis'), 'xc': v.get('xc'),
            'prefer_backend': v.get('prefer_backend'), 'ref_subdir': v.get('ref_subdir', 'Ez_FDBM')}
        for k, v in METHODS.items()
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")
    print(f"Total curves: {len(all_results)}")


if __name__ == '__main__':
    main()
