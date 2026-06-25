#!/usr/bin/env python3
"""Run manipulation (relaxed scan) and export .xyz movie.

Usage:
  python tests/run_manipulation.py --mol H2O --export-xyz movie.xyz
  python tests/run_manipulation.py --mol PTCDA --export-xyz movie.xyz --dx 0.1
  python tests/run_manipulation.py --mol PTCDA --z-pin 6.0 --x-end 8.0 --export-xyz ptcda_movie.xyz
"""
import os, sys, argparse, numpy as np

os.environ.setdefault('PYOPENCL_CTX', '0')
os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from tests.helpers.folded_rigid import (
    fit_folded_for_molecule, setup_rigid_folded, relaxed_scan,
    load_substrate, replicate_substrate, find_bonds,
    plot_relaxed_scan, plot_manipulation_trail, save_xyz_trajectory,
    Z_SURF_TOP,
)


def main():
    parser = argparse.ArgumentParser(description="Run manipulation (relaxed scan) and export .xyz movie")
    parser.add_argument('--mol', choices=['H2O', 'PTCDA'], default='H2O', help='Molecule to manipulate')
    parser.add_argument('--substrate', default='data/substrates/NaCl_1x1_L3.xyz', help='Substrate XYZ file')
    parser.add_argument('--dx', type=float, default=0.1, help='Step size for tip movement [Å]')
    parser.add_argument('--x-start', type=float, default=0.0, help='Start x position [Å]')
    parser.add_argument('--x-end', type=float, default=None, help='End x position [Å] (default: 4 for H2O, 8 for PTCDA)')
    parser.add_argument('--z-pin', type=float, default=None, help='Pin height above surface top [Å] (default: 4 for H2O, 6 for PTCDA)')
    parser.add_argument('--k-spring', type=float, default=10.0, help='Spring constant [eV/Å²]')
    parser.add_argument('--n-relax', type=int, default=200, help='Relaxation steps per path point')
    parser.add_argument('--dt', type=float, default=0.005, help='Timestep')
    parser.add_argument('--outdir', default='debug/run_manipulation', help='Output directory')
    parser.add_argument('--export-xyz', default=None, help='Export .xyz movie to this path')
    args = parser.parse_args()

    mol_file = os.path.join(_proj_root, 'data', 'xyz', f'{args.mol}.xyz')
    sub_file = os.path.join(_proj_root, args.substrate)
    os.makedirs(args.outdir, exist_ok=True)

    if args.x_end is None:
        args.x_end = 4.0 if args.mol == 'H2O' else 8.0
    if args.z_pin is None:
        args.z_pin = 4.0 if args.mol == 'H2O' else 6.0

    print(f'[Manip] Molecule: {args.mol}, substrate: {os.path.basename(sub_file)}')
    print(f'[Manip] Fitting folded basis...')
    fit = fit_folded_for_molecule(mol_file, substrate_file=sub_file)

    rbd = setup_rigid_folded(mol_file, fit, z_init=2.5, xy_init=(0.0, 0.0))
    bonds = find_bonds(rbd.atom_body_host.reshape(rbd.num_atoms, 4)[:, :3], rbd.enames, Rcut=1.8)

    if args.mol == 'H2O':
        pin_idx = 1  # H
        opp_idx = 0  # O
    else:
        pin_idx = 24  # O at one end
        opp_idx = 25  # O at opposite end

    z_pin_abs = Z_SURF_TOP + args.z_pin
    n_path = int(round((args.x_end - args.x_start) / args.dx)) + 1
    path = np.zeros((n_path, 3), dtype=np.float32)
    path[:, 0] = np.linspace(args.x_start, args.x_end, n_path)
    path[:, 1] = 0.0
    path[:, 2] = z_pin_abs

    print(f'[Manip] Pinning atom {pin_idx} ({rbd.enames[pin_idx]}) at z={z_pin_abs:.2f}, '
          f'dragging x: {args.x_start}→{args.x_end} Å ({n_path} steps, dx={args.dx})')
    traj = relaxed_scan(rbd, pin_atom_idx=pin_idx, path=path, k_spring=args.k_spring,
                        n_relax=args.n_relax, dt=args.dt, lin_damp=0.99, ang_damp=0.95, record_interval=50)

    sub_apos, sub_enames, _, sub_lvec = load_substrate(sub_file)
    rep_range = 6 if args.mol == 'H2O' else 10
    sub_rep_apos, sub_rep_enames = replicate_substrate(
        sub_apos, sub_enames, sub_lvec, (-rep_range, rep_range), (-rep_range, rep_range), z_min=-8.0)

    name = f'{args.mol.lower()}_manip'
    plot_relaxed_scan(traj, rbd.enames, sub_rep_apos, sub_rep_enames, args.outdir, name=name,
                      bonds=bonds, pin_atom_idx=pin_idx, highlight_element='O', target_element='Na')
    plot_manipulation_trail(traj, rbd.enames, sub_rep_apos, sub_rep_enames, args.outdir, name=name,
                            pin_atom_idx=pin_idx, opp_atom_idx=opp_idx)

    # Export .xyz movie
    xyz_path = args.export_xyz or os.path.join(args.outdir, f'{name}_movie.xyz')
    rec_per_path = len(traj['atom_positions']) // n_path
    snap_atoms = [traj['atom_positions'][min((i+1)*rec_per_path-1, len(traj['atom_positions'])-1)]
                  for i in range(n_path)]
    save_xyz_trajectory(xyz_path, rbd.enames, snap_atoms, sub_rep_apos, sub_rep_enames)
    print(f'[Manip] Saved .xyz movie: {xyz_path}')
    print(f'[Manip] Done. Check {args.outdir}/ for plots.')


if __name__ == '__main__':
    main()
