#!/usr/bin/env python3
"""Optimize Slater-tail basis parameters via simulated annealing.

Uses project_density_dense_points kernel for fast evaluation at z-profile points only.
SCF runs once; basis is updated on GPU without kernel recompilation.

Usage:
  cd /home/prokop/git/SPAMMM
  python examples/density_comparison/optimize_basis.py --molecule PTCDA_PBE_500eV --n-iter 2000
"""
import os, sys, re, time, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

GPAW_DIR = "/home/prokop/SIMULATIONS/Fukui_AFM/gpaw_fukui_cluster/jobs/results"
BOHR_TO_ANG = 0.529177
ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


def parse_gpaw_txt(txt_path):
    with open(txt_path) as f: text = f.read()
    pos_section = re.search(r"Positions:\s*\n(.*?)\n\s*\n", text, re.DOTALL)
    atoms = []
    if pos_section:
        for line in pos_section.group(1).strip().split("\n"):
            parts = line.split()
            if len(parts) >= 5:
                atoms.append((int(parts[0]), parts[1], float(parts[2]), float(parts[3]), float(parts[4])))
    cell_lines = re.findall(r"\d+\.\s*axis:\s+yes\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)", text)
    cell = np.array([[float(c[0]), float(c[1]), float(c[2])] for c in cell_lines])
    coarse_grid = np.array([int(c[3]) for c in cell_lines])
    fine_match = re.search(r"Fine grid:\s*(\d+)\*(\d+)\*(\d+)", text)
    fine_grid = np.array([int(fine_match.group(i)) for i in range(1, 4)]) if fine_match else coarse_grid * 2
    return atoms, cell, fine_grid


def main():
    parser = argparse.ArgumentParser(description="Optimize Slater-tail basis via simulated annealing")
    parser.add_argument('--molecule', default='PTCDA_PBE_500eV')
    parser.add_argument('--basis', default='3ob-3-1')
    parser.add_argument('--n-iter', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='/home/prokop/SIMULATIONS/Fukui_AFM/jobs/results/dftb_4way')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    os.environ.setdefault('PYOPENCL_CTX', '0')
    os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd, convert_wfc_to_species_list_ang
    from spammm.quantum.DFTB import Grid_dftb as dg
    from spammm.quantum.DFTB.DFTBcore import DFTBcore
    from spammm.quantum.DFTB.basis_optimizer import (
        amplitude_match_params, make_single_exponent_species_list,
        build_z_profile_points, extract_z_profiles, optimize_basis_sa
    )
    from spammm import plotUtils as pu
    from spammm import atomicUtils as au

    mol_name = args.molecule
    mol_dir = os.path.join(GPAW_DIR, mol_name)
    txt_path = os.path.join(mol_dir, 'N.txt')
    if not os.path.exists(txt_path): print(f"No data for {mol_name}"); return

    atoms, cell, fine_grid = parse_gpaw_txt(txt_path)
    z0 = np.mean([a[4] for a in atoms])
    print(f"Molecule: {mol_name}, {len(atoms)} atoms, z0={z0:.3f}")

    # Load GPAW reference
    rho_gpaw = np.load(os.path.join(mol_dir, 'rho_N.npy'))
    dL_gpaw = np.array([cell[i, i] for i in range(3)]) / fine_grid
    ref_profiles, z_vals = extract_z_profiles(rho_gpaw, atoms, np.zeros(3), dL_gpaw, fine_grid, z0)
    del rho_gpaw

    # Prepare atom positions
    atomPos = np.array([[a[2], a[3], a[4]] for a in atoms], dtype=np.float64)
    atomTypes = np.array([ELEM_Z[a[1]] for a in atoms], dtype=np.int32)
    enames = [a[1] for a in atoms]

    # Run SCF once
    basis_hsd_path = get_dftb_basis_path(args.basis)
    basis_data = parse_wfc_hsd(basis_hsd_path)
    basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
    norb_per_atom, orb_offsets, max_l = afm_utils.build_orbital_layout(basis_data, enames)
    max_shells = 3 if max_l >= 2 else 2

    work_dir = f'/tmp/sa_opt_{mol_name}'
    os.makedirs(work_dir, exist_ok=True)
    print("Running SCF...")
    from spammm.quantum.DFTB_utils import SK_PATHS as _SK_PATHS
    import shutil
    basis_name = args.basis
    sk_dir = _SK_PATHS.get(basis_name, os.path.join(os.environ.get('DFTB_SK_PATH', ''), basis_name))
    xyz_path = os.path.join(work_dir, 'geom.xyz')
    hsd_path = os.path.join(work_dir, 'dftb_in.hsd')
    au.save_xyz(xyz_path, enames, atomPos)
    species = sorted(set(enames))
    max_am_map = {0: 's', 1: 'p', 2: 'd'}
    max_ang_lines = []
    for elem in species:
        elem_data = basis_data[elem]
        max_l_e = max(orb['AngularMomentum'] for orb in elem_data['orbitals'])
        max_ang_lines.append(f'    {elem} = "{max_am_map[max_l_e]}"')
    max_ang_str = '\n'.join(max_ang_lines)
    with open(hsd_path, 'w') as f:
        f.write(f'''Geometry = xyzFormat {{
  <<< "geom.xyz"
}}
Hamiltonian = DFTB {{
  SCC = Yes
  SCCTolerance = 1e-7
  MaxSCCIterations = 200
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_dir}/"
    Separator = "-"
    Suffix = ".skf"
    LowerCaseTypeName = No
  }}
  MaxAngularMomentum = {{
{max_ang_str}
  }}
}}
''')
    for i, elem1 in enumerate(species):
        for elem2 in species[i:]:
            for sk_file in [f"{elem1}-{elem2}.skf", f"{elem2}-{elem1}.skf"]:
                src = os.path.join(sk_dir, sk_file)
                if os.path.exists(src): shutil.copy(src, work_dir)

    old_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        dftb = DFTBcore()
        dftb.init('dftb_in.hsd')
        dftb.enable_matrix_collection(dm=True, h=False, s=False)
        energy = dftb.run_scf()
        dm_dense = dftb.get_dm_dense()
        dftb.finalize()
    finally:
        os.chdir(old_cwd)
    print(f"DM: {dm_dense.shape}, norb_total={orb_offsets[-1]}")

    # Build z-profile points
    points, z_vals_pts = build_z_profile_points(atoms, z0, z_max=3.0, dz=0.1)
    n_atoms = len(atoms); n_z = len(z_vals_pts)

    # Setup projector with initial basis (only for elements present in molecule)
    all_params = amplitude_match_params(basis_ang)
    elems_present = set(enames)
    initial_params = {k: v for k, v in all_params.items() if k in elems_present}
    print(f"Initial params: {initial_params}")
    species_list_init = make_single_exponent_species_list(basis_ang, initial_params)

    coords_bohr = atomPos * 1.8897259886
    dftb_data = {'coords_bohr': coords_bohr, 'species_per_atom': list(range(len(enames))), 'species_names': enames}
    projector, atoms_dict = dg.setup_gridprojector_from_dftb(dftb_data, species_list_init, verbosity=0, max_shells=max_shells)

    # Run SA optimization
    best_params, best_obj, history = optimize_basis_sa(
        projector, dm_dense, norb_per_atom, orb_offsets, atoms_dict,
        points, ref_profiles, z_vals_pts, initial_params, basis_ang,
        n_iter=args.n_iter, seed=args.seed, verbosity=1)

    # Plot results
    def eval_with_params(params):
        species_list = make_single_exponent_species_list(basis_ang, params)
        projector.update_basis_sto(species_list)
        rho_pts = projector.project_density_dense_points(points, dm_dense, norb_per_atom, orb_offsets, atoms_dict)
        return rho_pts.reshape(n_atoms, n_z) * (1.0 / (BOHR_TO_ANG**3))

    rho_best = eval_with_params(best_params)
    rho_init = eval_with_params(initial_params)

    methods = [
        {'name': 'GPAW', 'color': 'r', 'profiles': ref_profiles, 'lw': 0.5, 'fit': True},
        {'name': 'DFTB+Slater (initial)', 'color': 'b', 'profiles': rho_init, 'lw': 0.5, 'fit': True},
        {'name': 'DFTB+Slater (SA-opt)', 'color': 'm', 'profiles': rho_best, 'lw': 0.5, 'fit': True},
    ]
    out = os.path.join(args.outdir, f"{mol_name}_sa_optimized.png")
    pu.plot_density_per_element(atoms, z_vals_pts, methods,
                                suptitle=f"{mol_name} — SA-optimized Slater-tail vs GPAW", fname=out)
    print(f"Saved: {out}")

    out2 = os.path.join(args.outdir, f"{mol_name}_sa_history.png")
    pu.plot_sa_history(history, title=f"{mol_name} SA convergence", fname=out2)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
