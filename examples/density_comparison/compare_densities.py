#!/usr/bin/env python3
"""Compare DFTB+ projected density vs GPAW vs PySCF density for molecules.

Produces 4 plot types per molecule:
  - per_element: 1D z-profiles per element with log-linear fits
  - by_atom: 1D z-profiles per individual atom
  - methods_panel: side-by-side panels per method, all atoms overlaid
  - 2d_slices: 2D density maps at molecular plane

Always runs both original DFTB and DFTB+Slater-tail projection.

Usage:
  cd /home/prokop/git/SPAMMM
  python examples/density_comparison/compare_densities.py [--basis 3ob-3-1] [--molecules H2O pentacene]
"""
import os, sys, re, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

GPAW_DIR = "/home/prokop/SIMULATIONS/Fukui_AFM/gpaw_fukui_cluster/jobs/results"
PYSCF_DIR = "/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/jobs/results"
BOHR_TO_ANG = 0.529177
ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
Z_TO_SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}


def parse_cube(cube_path):
    """Parse Gaussian cube file: atoms, origin, grid spacing, grid dims."""
    with open(cube_path) as f: lines = f.readlines()
    p = lines[2].split(); natoms = int(p[0])
    origin = np.array([float(p[1]), float(p[2]), float(p[3])]) * BOHR_TO_ANG
    nx, dx = int(lines[3].split()[0]), float(lines[3].split()[1]) * BOHR_TO_ANG
    ny, dy = int(lines[4].split()[0]), float(lines[4].split()[2]) * BOHR_TO_ANG
    nz, dz = int(lines[5].split()[0]), float(lines[5].split()[3]) * BOHR_TO_ANG
    atoms = []
    for i in range(natoms):
        p = lines[6 + i].split(); Z = int(p[0])
        atoms.append((Z, float(p[2])*BOHR_TO_ANG, float(p[3])*BOHR_TO_ANG, float(p[4])*BOHR_TO_ANG))
    return atoms, origin, np.array([dx, dy, dz]), np.array([nx, ny, nz])


def gpaw_to_pyscf_name(gpaw_name):
    return gpaw_name.replace('_500eV', '_def2-SVP')


def parse_gpaw_txt(txt_path):
    """Parse GPAW output txt: positions, cell, grid info."""
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
    parser = argparse.ArgumentParser(description="Compare DFTB vs GPAW vs PySCF electron density")
    parser.add_argument('--basis', default='3ob-3-1')
    parser.add_argument('--step', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=6.0)
    parser.add_argument('--outdir', default='/home/prokop/SIMULATIONS/Fukui_AFM/jobs/results/dftb_4way')
    parser.add_argument('--molecules', nargs='*', default=None)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    os.environ.setdefault('PYOPENCL_CTX', '0')
    os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list
    from spammm.quantum.DFTB.basis_optimizer import extract_z_profiles
    from spammm import plotUtils as pu

    basis_hsd_path = get_dftb_basis_path(args.basis)
    if not basis_hsd_path or not os.path.exists(basis_hsd_path):
        print(f"ERROR: basis file not found for '{args.basis}'"); sys.exit(1)
    print(f"Basis: {args.basis} -> {basis_hsd_path}")

    mol_dirs = sorted([d for d in os.listdir(GPAW_DIR) if os.path.isdir(os.path.join(GPAW_DIR, d))])
    if args.molecules:
        mol_dirs = [d for d in mol_dirs if any(m.lower() in d.lower() for m in args.molecules)]

    for mol_name in mol_dirs:
        mol_dir = os.path.join(GPAW_DIR, mol_name)
        print(f"\n{'='*60}\nProcessing: {mol_name}")

        txt_path = os.path.join(mol_dir, "N.txt")
        if not os.path.exists(txt_path): print("  No N.txt, skipping"); continue
        atoms, cell, fine_grid = parse_gpaw_txt(txt_path)
        z0 = np.mean([a[4] for a in atoms])
        dL_gpaw = np.array([cell[i, i] for i in range(3)]) / fine_grid
        print(f"  Atoms: {len(atoms)}, z0={z0:.3f}, GPAW grid: {fine_grid}")

        symbols = set(a[1] for a in atoms)
        if symbols - set(ELEM_Z.keys()): print(f"  Unsupported elements, skipping"); continue

        gpaw_rho_path = os.path.join(mol_dir, "rho_N.npy")
        if not os.path.exists(gpaw_rho_path): print("  No rho_N.npy, skipping"); continue
        print(f"  Loading GPAW density ({os.path.getsize(gpaw_rho_path)/1e6:.0f} MB)...")
        rho_gpaw = np.load(gpaw_rho_path)
        gpaw_integral = rho_gpaw.sum() * np.prod([cell[i, i] for i in range(3)]) / np.prod(rho_gpaw.shape)
        print(f"  GPAW integral: {gpaw_integral:.4f}")

        atomPos = np.array([[a[2], a[3], a[4]] for a in atoms], dtype=np.float64)
        atomTypes = np.array([ELEM_Z[a[1]] for a in atoms], dtype=np.int32)

        # Run DFTB SCF + project density (both original and Slater-tail)
        work_dir = os.path.join(args.outdir, mol_name + '_dftb_work')
        os.makedirs(work_dir, exist_ok=True)
        print(f"  Running DFTB+ SCF + density projection...")

        basis_data = parse_wfc_hsd(basis_hsd_path)
        basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
        slater_basis = make_slater_tail_species_list(basis_ang)
        print(f"  Using Slater-tail basis for projection")

        try:
            d_orig = afm_utils.get_density_from_dftb_dense(
                atomPos, atomTypes, basis_hsd_path, work_dir + '_orig',
                step=args.step, margin=args.margin, z_extra=6.0, verbosity=0)
            d_slat = afm_utils.get_density_from_dftb_dense(
                atomPos, atomTypes, basis_hsd_path, work_dir + '_slat',
                step=args.step, margin=args.margin, z_extra=6.0, verbosity=0,
                projection_basis_ang=slater_basis)
        except Exception as e:
            print(f"  DFTB failed: {e}"); import traceback; traceback.print_exc(); continue

        rho_dftb = d_orig['rho_scf']; rho_slater = d_slat['rho_scf']
        dftb_origin = d_orig['origin']; dftb_ngrid = d_orig['ngrid']; dftb_step = args.step
        print(f"  DFTB integral: {rho_dftb.sum()*dftb_step**3:.4f}  Slater: {rho_slater.sum()*dftb_step**3:.4f}")

        # Load PySCF density
        pyscf_name = gpaw_to_pyscf_name(mol_name)
        pyscf_dir = os.path.join(PYSCF_DIR, pyscf_name)
        rho_pyscf = None; has_pyscf = False
        pyscf_rho_path = os.path.join(pyscf_dir, 'rho_N.npy')
        pyscf_cube_path = os.path.join(pyscf_dir, 'rho_N.cube')
        if os.path.exists(pyscf_rho_path) and os.path.exists(pyscf_cube_path):
            print(f"  Loading PySCF density...")
            rho_pyscf = np.load(pyscf_rho_path) / (BOHR_TO_ANG**3)
            pyscf_atoms, pyscf_origin, dL_pyscf, pyscf_ngrid = parse_cube(pyscf_cube_path)
            pyscf_integral = rho_pyscf.sum() * np.prod(dL_pyscf)
            print(f"  PySCF integral: {pyscf_integral:.4f}")
            has_pyscf = True
            z0_pyscf = np.mean([a[3] for a in pyscf_atoms])

        # Extract z-profiles for all methods
        Z_MAX = 3.0; DZ = 0.1
        z_vals = np.arange(0, Z_MAX + DZ/2, DZ)

        gpaw_profs, _ = extract_z_profiles(rho_gpaw, atoms, np.zeros(3), dL_gpaw, fine_grid, z0, Z_MAX, DZ)
        dftb_profs, _ = extract_z_profiles(rho_dftb, atoms, dftb_origin, dftb_step, dftb_ngrid, z0, Z_MAX, DZ)
        slater_profs, _ = extract_z_profiles(rho_slater, atoms, dftb_origin, dftb_step, dftb_ngrid, z0, Z_MAX, DZ)

        methods_1d = [
            {'name': 'GPAW', 'color': 'r', 'profiles': gpaw_profs, 'lw': 0.5, 'fit': True},
            {'name': f'DFTB {args.basis}', 'color': 'b', 'profiles': dftb_profs, 'lw': 0.5, 'fit': True},
            {'name': 'DFTB+Slater', 'color': 'm', 'profiles': slater_profs, 'lw': 0.5, 'fit': True},
        ]
        if has_pyscf:
            pyscf_atoms_fmt = [(i, Z_TO_SYM.get(a[0], '?'), a[1], a[2], a[3]) for i, a in enumerate(pyscf_atoms)]
            pyscf_profs, _ = extract_z_profiles(rho_pyscf, pyscf_atoms_fmt, pyscf_origin, dL_pyscf, pyscf_ngrid, z0_pyscf, Z_MAX, DZ)
            methods_1d.append({'name': 'PySCF', 'color': 'g', 'profiles': pyscf_profs, 'lw': 0.5, 'fit': True})

        # Plot 1: per-element
        out1 = os.path.join(args.outdir, f"{mol_name}_per_element.png")
        pu.plot_density_per_element(atoms, z_vals, methods_1d,
                                    suptitle=f"{mol_name} — density per element (0–3 A)", fname=out1)
        print(f"  Saved: {out1}")

        # Plot 2: methods panel
        out2 = os.path.join(args.outdir, f"{mol_name}_methods_panel.png")
        pu.plot_density_methods_panel(atoms, z_vals, methods_1d,
                                      suptitle=f"{mol_name} — density above each atom", fname=out2)
        print(f"  Saved: {out2}")

        # Plot 3: 2D density slices
        iz_g = int(round(z0 / dL_gpaw[2]))
        iz_d = int(round((z0 - dftb_origin[2]) / dftb_step))
        atoms_2d = [(a[1], a[2], a[3]) for a in atoms]
        extent_g = [0, cell[0, 0], 0, cell[1, 1]]
        extent_d = [dftb_origin[0], dftb_origin[0] + dftb_ngrid[0]*dftb_step,
                    dftb_origin[1], dftb_origin[1] + dftb_ngrid[1]*dftb_step]

        methods_2d = [
            {'name': 'GPAW', 'slice_2d': rho_gpaw[:, :, iz_g], 'extent': extent_g, 'atoms': atoms_2d},
            {'name': f'DFTB {args.basis}', 'slice_2d': rho_dftb[:, :, iz_d], 'extent': extent_d, 'atoms': atoms_2d},
            {'name': 'DFTB+Slater', 'slice_2d': rho_slater[:, :, iz_d], 'extent': extent_d, 'atoms': atoms_2d},
        ]
        if has_pyscf:
            iz_p = int(round((z0_pyscf - pyscf_origin[2]) / dL_pyscf[2]))
            extent_p = [pyscf_origin[0], pyscf_origin[0] + pyscf_ngrid[0]*dL_pyscf[0],
                        pyscf_origin[1], pyscf_origin[1] + pyscf_ngrid[1]*dL_pyscf[1]]
            atoms_2d_p = [(Z_TO_SYM.get(a[0], '?'), a[1], a[2]) for a in pyscf_atoms]
            methods_2d.append({'name': 'PySCF', 'slice_2d': rho_pyscf[:, :, iz_p], 'extent': extent_p, 'atoms': atoms_2d_p})

        out3 = os.path.join(args.outdir, f"{mol_name}_2d_slices.png")
        pu.plot_2d_density_panel(methods_2d, suptitle=f"{mol_name} — 2D density at molecular plane", fname=out3)
        print(f"  Saved: {out3}")

        # Plot 4: multi-z 2D density grid (rows=z-heights, cols=methods)
        z_heights_rel = [1.5, 2.0, 2.5]
        methods_multi = [
            {'name': 'GPAW', 'rho_3d': rho_gpaw, 'origin': np.zeros(3), 'step': dL_gpaw, 'atoms': atoms_2d, 'z0': z0},
            {'name': f'DFTB {args.basis}', 'rho_3d': rho_dftb, 'origin': dftb_origin, 'step': dftb_step, 'atoms': atoms_2d, 'z0': z0},
            {'name': 'DFTB+Slater', 'rho_3d': rho_slater, 'origin': dftb_origin, 'step': dftb_step, 'atoms': atoms_2d, 'z0': z0},
        ]
        if has_pyscf:
            atoms_2d_p = [(Z_TO_SYM.get(a[0], '?'), a[1], a[2]) for a in pyscf_atoms]
            methods_multi.append({'name': 'PySCF', 'rho_3d': rho_pyscf, 'origin': pyscf_origin, 'step': dL_pyscf, 'atoms': atoms_2d_p, 'z0': z0_pyscf})
        # Reorder: GPAW, PySCF, DFTB+Slater, DFTB 3ob
        order = []
        for name in ['GPAW', 'PySCF', 'DFTB+Slater', f'DFTB {args.basis}']:
            for i, m in enumerate(methods_multi):
                if m['name'] == name and i not in order:
                    order.append(i); break
        methods_multi = [methods_multi[i] for i in order]
        out4 = os.path.join(args.outdir, f"{mol_name}_multi_z.png")
        pu.plot_density_multi_z(methods_multi, z_heights_rel,
                                suptitle=f"{mol_name} — 2D density at z=1.5, 2.0, 2.5 A above plane", fname=out4)
        print(f"  Saved: {out4}")

        del rho_gpaw, rho_dftb, rho_slater
        if has_pyscf: del rho_pyscf

    print(f"\nDone. Plots in: {args.outdir}")


if __name__ == "__main__":
    main()
