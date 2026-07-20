#!/usr/bin/env python3
"""Optimize Slater-tail (prolonged) DFTB projection basis vs a reference density.

Small-molecule / Fukui path (GPAW):
  python examples/density_comparison/optimize_basis.py --molecule H2O_PBE_500eV --n-iter 300

PTCDA vs local pySCF GPU density (same level as CO Ez refs):
  python examples/density_comparison/optimize_basis.py \\
      --ref-rho debug/densities/rho_PTCDA_pyscf_gpu_pbe.npy \\
      --ref-meta debug/densities/rho_PTCDA_pyscf_gpu_pbe.meta.npz \\
      --xyz data/xyz/PTCDA.xyz --molecule PTCDA --n-iter 2000 \\
      --fit-lo 1.0 --fit-hi 2.5 --z-max 4.0 \\
      --outdir debug/dftb_basis_sa_ptcda --project-density --compare-pauli

SCF once with stock 3ob; only the projection STOs (N, ζ) are annealed.
"""
import os, sys, re, time, argparse, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def load_local_ref(rho_path, meta_path, xyz_path):
    """Load pySCF/local ρ + meta + XYZ → atoms, ref_profiles, grid info."""
    from spammm import atomicUtils as au
    from spammm.quantum.DFTB.basis_optimizer import extract_z_profiles

    rho = np.load(rho_path)
    meta = np.load(meta_path)
    origin = np.asarray(meta['origin'], dtype=float)
    step = float(meta['step'])
    ngrid = np.asarray(meta['ngrid'] if 'ngrid' in meta.files else rho.shape, dtype=int)
    atom_pos = np.asarray(meta['atom_pos'], dtype=float) if 'atom_pos' in meta.files else None
    atom_names = list(meta['atom_names']) if 'atom_names' in meta.files else None
    if atom_pos is None or atom_names is None:
        atom_pos, _, atom_names, _, _ = au.load_xyz(xyz_path)
    atoms = [(i, n, float(p[0]), float(p[1]), float(p[2])) for i, (n, p) in enumerate(zip(atom_names, atom_pos))]
    z0 = float(np.mean([a[4] for a in atoms]))
    return atoms, atom_pos, atom_names, rho, origin, step, ngrid, z0


def run_dftb_scf(atomPos, enames, basis_name, work_dir, basis_data):
    """One SCF; return dense DM."""
    from spammm.quantum.DFTB.DFTBcore import DFTBcore
    from spammm.quantum.DFTB_utils import SK_PATHS as _SK_PATHS
    from spammm import atomicUtils as au
    import shutil

    os.makedirs(work_dir, exist_ok=True)
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
    print(f"  SCF E={energy}  DM={dm_dense.shape}")
    return dm_dense


def pauli_overlay_log(rho_dict, origin, step, atom_pos, atom_names, targets, ez_method,
                      out_png, tip_mode='co', sigma=0.7, A=40.0, beta=1.15):
    """Log-scale FDBM Pauli (several ρ) vs pySCF Ez refs."""
    from spammm.SPM import AFM as afm
    from spammm.SPM.AFM_utils import get_tip_densities
    from tests.SPM.extract_pauli_zscan import load_ez_reference, extract_z_line

    os.environ.setdefault('SPAMMM_AFM_CPU_FFT', '1')  # PTCDA ny=176 not clFFT-friendly
    nx, ny, nz = next(iter(rho_dict.values())).shape
    rho_tip, _ = get_tip_densities(tip_mode, (nx, ny, nz), step, sigma=sigma)
    # CO tip: use rho_tip_total; already rolled to (0,0,0)
    tip_rolled = True

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {'stock': 'C0', 'sa': 'C4', 'pyscf': '0.4'}
    for key, rho in rho_dict.items():
        overlap = afm.compute_pauli_overlap(rho, rho_tip, step, tip_rolled=tip_rolled)
        E = afm.scale_pauli_field(overlap, step, A, beta, return_grads=False)
        for label, idx in targets:
            z_abs, line = extract_z_line(E, origin, step, atom_pos[idx])
            z_rel = z_abs - atom_pos[idx][2]
            m = line > 1e-12
            if m.any():
                ax.semilogy(z_rel[m], line[m], '-', lw=1.2, color=colors.get(key, None),
                            alpha=0.85, label=f'{key} {label}')

    # Ez refs (once per site)
    for label, idx in targets:
        ez = load_ez_reference('PTCDA', ez_method, label, idx)
        if ez is None:
            print(f"  WARNING: no Ez for {label}{idx}")
            continue
        m = ez['e_rel'] > 1e-12
        if m.any():
            ax.semilogy(ez['z'][m], ez['e_rel'][m], ':', lw=1.8, label=f'Ez {label}')

    ax.set_xlim(1.2, 8.0)
    ax.set_xlabel('z (Å)'); ax.set_ylabel('E (eV) [log]')
    ax.set_title(f'FDBM Pauli (tip={tip_mode}) vs Ez — stock / SA / DFT  A={A:.1f} β={beta:.3f}')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"REVIEW: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Optimize Slater-tail basis via simulated annealing")
    parser.add_argument('--molecule', default='PTCDA')
    parser.add_argument('--basis', default='3ob-3-1')
    parser.add_argument('--n-iter', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default=os.path.join(_ROOT, 'debug', 'dftb_basis_sa'))
    # Local pySCF / compute_densities refs (preferred for PTCDA)
    parser.add_argument('--ref-rho', default=None, help='Reference density .npy (e/Å³)')
    parser.add_argument('--ref-meta', default=None, help='Matching .meta.npz')
    parser.add_argument('--xyz', default=None, help='XYZ if meta lacks atoms')
    parser.add_argument('--fit-lo', type=float, default=0.5)
    parser.add_argument('--fit-hi', type=float, default=1.5)
    parser.add_argument('--z-max', type=float, default=3.0)
    parser.add_argument('--project-density', action='store_true',
                        help='Write full 3D ρ with SA basis to debug/densities/')
    parser.add_argument('--compare-pauli', action='store_true',
                        help='Log Pauli overlay: stock vs SA vs Ez (pySCF GPU)')
    parser.add_argument('--tip-mode', default='co', choices=['co', 'gaussian'])
    parser.add_argument('--ez-method', default='pyscf_gpu_pbe')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # PTCDA grids often have ny with factor 11 → clFFT rejects; NumPy FFT is fine for z-scans
    os.environ['SPAMMM_AFM_CPU_FFT'] = '1'

    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd, convert_wfc_to_species_list_ang
    from spammm.quantum.DFTB import Grid_dftb as dg
    from spammm.quantum.DFTB.basis_optimizer import (
        amplitude_match_params, make_single_exponent_species_list,
        build_z_profile_points, extract_z_profiles, optimize_basis_sa
    )
    from spammm import plotUtils as pu

    mol_name = args.molecule
    grid_spec = None
    origin = step = ngrid = None

    if args.ref_rho:
        rho_path = args.ref_rho if os.path.isabs(args.ref_rho) else os.path.join(_ROOT, args.ref_rho)
        meta_path = args.ref_meta if args.ref_meta else rho_path.replace('.npy', '.meta.npz')
        if not os.path.isabs(meta_path):
            meta_path = os.path.join(_ROOT, meta_path)
        xyz_path = args.xyz or os.path.join(_ROOT, 'data', 'xyz', f'{mol_name}.xyz')
        if not os.path.isabs(xyz_path):
            xyz_path = os.path.join(_ROOT, xyz_path)
        print(f"Local ref: {rho_path}")
        atoms, atomPos, enames, rho_ref, origin, step, ngrid, z0 = load_local_ref(rho_path, meta_path, xyz_path)
        ref_profiles, z_vals = extract_z_profiles(rho_ref, atoms, origin, step, ngrid, z0,
                                                  z_max=args.z_max, dz=0.1)
        del rho_ref
        grid_spec = {
            'origin': origin, 'ngrid': tuple(int(x) for x in ngrid),
            'dA': [step, 0.0, 0.0], 'dB': [0.0, step, 0.0], 'dC': [0.0, 0.0, step],
        }
        ref_label = 'pySCF'
    else:
        # Legacy GPAW Fukui path
        mol_dir = os.path.join(GPAW_DIR, mol_name if mol_name.endswith('eV') else f'{mol_name}_PBE_500eV')
        txt_path = os.path.join(mol_dir, 'N.txt')
        if not os.path.exists(txt_path):
            print(f"No GPAW data at {txt_path}; pass --ref-rho for local pySCF density")
            return
        atoms, cell, fine_grid = parse_gpaw_txt(txt_path)
        z0 = np.mean([a[4] for a in atoms])
        rho_gpaw = np.load(os.path.join(mol_dir, 'rho_N.npy'))
        dL_gpaw = np.array([cell[i, i] for i in range(3)]) / fine_grid
        ref_profiles, z_vals = extract_z_profiles(rho_gpaw, atoms, np.zeros(3), dL_gpaw, fine_grid, z0,
                                                  z_max=args.z_max, dz=0.1)
        del rho_gpaw
        atomPos = np.array([[a[2], a[3], a[4]] for a in atoms], dtype=np.float64)
        enames = [a[1] for a in atoms]
        ref_label = 'GPAW'

    print(f"Molecule: {mol_name}, {len(atoms)} atoms, z0={z0:.3f}, fit=[{args.fit_lo},{args.fit_hi}] Å")

    basis_hsd_path = get_dftb_basis_path(args.basis)
    basis_data = parse_wfc_hsd(basis_hsd_path)
    basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
    norb_per_atom, orb_offsets, max_l = afm_utils.build_orbital_layout(basis_data, enames)
    max_shells = 3 if max_l >= 2 else 2

    work_dir = os.path.join(args.outdir, f'scf_{mol_name}')
    print("Running SCF...")
    dm_dense = run_dftb_scf(atomPos, enames, args.basis, work_dir, basis_data)
    print(f"DM: {dm_dense.shape}, norb_total={orb_offsets[-1]}")

    points, z_vals_pts = build_z_profile_points(atoms, z0, z_max=args.z_max, dz=0.1)
    n_atoms = len(atoms); n_z = len(z_vals_pts)
    assert ref_profiles.shape == (n_atoms, n_z), (
        f"ref_profiles {ref_profiles.shape} vs expected ({n_atoms},{n_z}) — check z_max/dz")

    all_params = amplitude_match_params(basis_ang)
    elems_present = set(enames)
    initial_params = {k: v for k, v in all_params.items() if k in elems_present}
    print(f"Initial params: {initial_params}")
    species_list_init = make_single_exponent_species_list(basis_ang, initial_params)

    coords_bohr = atomPos * 1.8897259886
    dftb_data = {'coords_bohr': coords_bohr, 'species_per_atom': list(range(len(enames))), 'species_names': enames}
    projector, atoms_dict = dg.setup_gridprojector_from_dftb(dftb_data, species_list_init, verbosity=0, max_shells=max_shells)

    best_params, best_obj, history = optimize_basis_sa(
        projector, dm_dense, norb_per_atom, orb_offsets, atoms_dict,
        points, ref_profiles, z_vals_pts, initial_params, basis_ang,
        n_iter=args.n_iter, seed=args.seed, verbosity=1,
        fit_lo=args.fit_lo, fit_hi=args.fit_hi)

    params_path = os.path.join(args.outdir, f'{mol_name}_sa_params.json')
    with open(params_path, 'w') as f:
        json.dump({
            'molecule': mol_name, 'basis': args.basis, 'ref': ref_label,
            'fit_lo': args.fit_lo, 'fit_hi': args.fit_hi, 'z_max': args.z_max,
            'best_obj': float(best_obj),
            'initial_params': {k: [float(x) for x in v] for k, v in initial_params.items()},
            'best_params': {k: [float(x) for x in v] for k, v in best_params.items()},
        }, f, indent=2)
    print(f"Saved: {params_path}")
    print(f"REVIEW: {params_path}")

    def eval_with_params(params):
        species_list = make_single_exponent_species_list(basis_ang, params)
        projector.update_basis_sto(species_list)
        rho_pts = projector.project_density_dense_points(points, dm_dense, norb_per_atom, orb_offsets, atoms_dict)
        return rho_pts.reshape(n_atoms, n_z) * (1.0 / (BOHR_TO_ANG**3))

    rho_best = eval_with_params(best_params)
    rho_init = eval_with_params(initial_params)

    methods = [
        {'name': ref_label, 'color': 'r', 'profiles': ref_profiles, 'lw': 0.5, 'fit': True},
        {'name': 'DFTB+Slater (initial)', 'color': 'b', 'profiles': rho_init, 'lw': 0.5, 'fit': True},
        {'name': 'DFTB+Slater (SA-opt)', 'color': 'm', 'profiles': rho_best, 'lw': 0.5, 'fit': True},
    ]
    out = os.path.join(args.outdir, f"{mol_name}_sa_optimized.png")
    pu.plot_density_per_element(atoms, z_vals_pts, methods,
                                suptitle=f"{mol_name} — SA Slater-tail vs {ref_label}  fit=[{args.fit_lo},{args.fit_hi}]Å",
                                fname=out)
    print(f"REVIEW: {out}")

    out2 = os.path.join(args.outdir, f"{mol_name}_sa_history.png")
    pu.plot_sa_history(history, title=f"{mol_name} SA convergence", fname=out2)
    print(f"REVIEW: {out2}")

    dens_dir = os.path.join(_ROOT, 'debug', 'densities')
    rho_sa_3d = rho_stock_3d = None
    if args.project_density or args.compare_pauli:
        os.makedirs(dens_dir, exist_ok=True)
        if grid_spec is None:
            raise RuntimeError("--project-density/--compare-pauli need a local --ref-rho grid (grid_spec)")
        print("Projecting full 3D density with SA basis (reuse SCF DM)...")
        t0 = time.time()
        projector.update_basis_sto(make_single_exponent_species_list(basis_ang, best_params))
        # Extend cutoffs: update_basis_sto may not enlarge Rcut in atoms_dict — rebuild projector if needed
        rho_sa_3d = projector.project_density_dense(
            dm_dense.astype(np.float32), norb_per_atom, orb_offsets, atoms_dict, grid_spec)
        print(f"  SA 3D done in {time.time()-t0:.1f}s  q={rho_sa_3d.sum()*step**3:.2f}")
        npy_sa = os.path.join(dens_dir, f'rho_{mol_name}_dftb_3ob_sa.npy')
        meta_sa = os.path.join(dens_dir, f'rho_{mol_name}_dftb_3ob_sa.meta.npz')
        np.save(npy_sa, rho_sa_3d)
        np.savez(meta_sa, origin=origin, ngrid=ngrid, step=step,
                 atom_pos=atomPos, atom_names=np.array(enames))
        print(f"Saved: {npy_sa}")

        stock_npy = os.path.join(dens_dir, f'rho_{mol_name}_dftb_3ob.npy')
        if os.path.isfile(stock_npy):
            rho_stock_3d = np.load(stock_npy)
            print(f"  Loaded stock 3ob density {stock_npy} shape={rho_stock_3d.shape}")
        else:
            print("  Projecting stock multi-zeta 3ob density...")
            projector.update_basis_sto(basis_ang)
            rho_stock_3d = projector.project_density_dense(
                dm_dense.astype(np.float32), norb_per_atom, orb_offsets, atoms_dict, grid_spec)
            np.save(stock_npy, rho_stock_3d)
            np.savez(stock_npy.replace('.npy', '.meta.npz'), origin=origin, ngrid=ngrid, step=step,
                     atom_pos=atomPos, atom_names=np.array(enames))

    if args.compare_pauli and rho_sa_3d is not None:
        if origin is None:
            origin = res_sa['origin']; step = float(res_sa['grid_spec']['dA'][0])
        targets = [('O_eq', 26), ('O_br', 24), ('C_anh', 11), ('C_core', 6)] if mol_name == 'PTCDA' else [
            (enames[i], i) for i in range(min(3, len(enames)))]
        rho_dict = {'sa': rho_sa_3d}
        if rho_stock_3d is not None:
            # Align shapes if stock was on different grid
            if rho_stock_3d.shape == rho_sa_3d.shape:
                rho_dict = {'stock': rho_stock_3d, 'sa': rho_sa_3d}
            else:
                print(f"  WARNING: stock shape {rho_stock_3d.shape} != SA {rho_sa_3d.shape}; overlay SA only")
        # Use Gaussian-tip fitted A,β as starting scale (absolute scale less important on log plot)
        from spammm.SPM import AFM as afm
        defs = afm.PAULI_FITTED_DEFAULTS.get('3ob-3-1', {'A': 40.0, 'beta': 1.15})
        out_png = os.path.join(args.outdir, f'{mol_name}_pauli_stock_vs_sa_vs_ez.png')
        pauli_overlay_log(rho_dict, origin, step, atomPos, enames, targets, args.ez_method,
                          out_png, tip_mode=args.tip_mode, A=defs['A'], beta=defs['beta'])


if __name__ == "__main__":
    main()
