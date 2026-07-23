#!/usr/bin/env python3
"""Demo script for RigidBodyPairFF + Vispy visualization.

Loads uracil (static) and HCOOH (dynamic) molecules, sets up pairwise
force field interactions with electron pairs, and launches interactive
Vispy visualization with mouse picking.

Usage:
    python3 demos/demo_pairff.py                    # interactive Vispy (unified kernel)
    python3 demos/demo_pairff.py --pairff-mode legacy
    python3 demos/demo_pairff.py --no-vis           # headless relaxation test
    python3 demos/demo_pairff.py --no-vis --steps 500  # custom step count

Kernels (switchable from GUI or --pairff-mode):
  unified — rigid_body_pairff_unified_kernel: single compact-exp loop (default)
            (y=(1-b*rho)^8, rho=r2/(sqrt(r2+w*w)+w))
  legacy  — rigid_body_pairff_kernel: Morse+Coulomb / Lorentzian
See:
  - examples/density_comparison/HBondFF/fit_radial.py (--compact-exp-demo)
  - kernels/rigid.cl (rigid_body_pairff_unified_kernel)
"""
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_XYZ = os.path.join(REPO_ROOT, 'data', 'xyz')

from spammm.topology.FFparams import load_xyz_with_REQs
from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF


def load_molecule(fname):
    """Load XYZ molecule with REQ parameters."""
    apos, REQs, enames, Zs, lvec = load_xyz_with_REQs(fname)
    return np.asarray(apos, dtype=np.float32), REQs, enames


def main():
    import argparse
    parser = argparse.ArgumentParser(description='RigidBodyPairFF demo')
    parser.add_argument('--no-vis', action='store_true', help='Headless mode (no Vispy)')
    parser.add_argument('--steps', type=int, default=300, help='Max relaxation steps (headless)')
    parser.add_argument('--dt', type=float, default=0.02, help='Time step')
    parser.add_argument('--he', type=float, default=-1.0, help='Hbond energy coefficient (epair pseudo-charge)')
    parser.add_argument('--hs', type=float, default=1.0, help='Sigma-hole pseudo charge (0=disabled)')
    parser.add_argument('--rc', type=float, default=3.0, help='Hbond cutoff radius (legacy)')
    parser.add_argument('--alpha', type=float, default=1.8, help='Morse alpha (legacy) / ignored if --beta set')
    parser.add_argument('--beta', type=float, default=None, help='Compact-exp beta (unified; default 1.7)')
    parser.add_argument('--w', type=float, default=0.7, help='Soft-radius / Lorentzian width')
    parser.add_argument('--kz', type=float, default=5.0, help='Z-constraint strength')
    parser.add_argument('--epair-dist', type=float, default=1.4, help='Epair distance from host [Å]')
    parser.add_argument('--sigma-dist', type=float, default=1.0, help='Sigma hole distance from H [Å] (0=disabled)')
    parser.add_argument('--pairff-mode', choices=['legacy', 'unified'], default='unified',
                        help='Force kernel: legacy Morse+Lorentzian or unified compact-exp')
    args = parser.parse_args()

    # --- Load molecules ---
    static_apos, static_REQs, static_enames = load_molecule(os.path.join(DATA_XYZ, 'uracil.xyz'))
    dyn_apos, dyn_REQs, dyn_enames = load_molecule(os.path.join(DATA_XYZ, 'HCOOH.xyz'))

    print(f"Static (uracil): {len(static_enames)} atoms — {static_enames}")
    print(f"Dynamic (HCOOH): {len(dyn_enames)} atoms — {dyn_enames}")
    print(f"PairFF mode: {args.pairff_mode}")

    # Position dynamic molecule above static molecule center
    static_center = static_apos[:, :2].mean(axis=0)
    body_pos = np.array([static_center[0], static_center[1], 3.0], dtype=np.float32)

    # --- Build RigidBodyPairFF ---
    rbd = RigidBodyPairFF.from_two_molecules(
        dyn_apos=dyn_apos, dyn_enames=dyn_enames, dyn_REQs=dyn_REQs,
        static_apos=static_apos, static_enames=static_enames, static_REQs=static_REQs,
        body_pos=body_pos,
        He=args.he, rc=args.rc, w=args.w, morse_alpha=args.alpha, k_z=args.kz,
        z_target=0.0, Hs=args.hs,
        epair_dist=args.epair_dist, sigma_dist=args.sigma_dist,
        mode=args.pairff_mode, beta=args.beta,
    )

    # Print electron pair info
    print(f"Dynamic atoms+epairs: {len(rbd.enames)} — {rbd.enames}")
    print(f"Static atoms+epairs: {len(rbd.static_enames)} — {rbd.static_enames}")
    print(f"Dynamic types: {rbd.dyn_type_host}")
    print(f"Static types:  {rbd.static_type_host}")

    if args.no_vis:
        # --- Headless relaxation test ---
        print(f"\nRunning FIRE relaxation (max {args.steps} steps, dt={args.dt})...")
        result = rbd.relax_pairff(max_steps=args.steps, dt=args.dt, f_tol=1e-4, t_tol=1e-4, record=True)
        print(f"Converged: {result['converged']} in {result['steps']} steps")
        print(f"  F={result['F']:.6f}  T={result['T']:.6f}  E={result['E']:.6f}")

        # Print final positions
        out = rbd.download_outputs()
        print(f"\nFinal CoM pos: {out['pos'][0, :3]}")
        print(f"Final quat:    {out['quats'][0]}")
        print(f"Final atom positions (world):")
        for i, (e, t) in enumerate(zip(rbd.enames, rbd.dyn_type_host)):
            tag = 'epair' if t == 1 else ('sigma' if t == 2 else e)
            print(f"  [{i:2d}] {tag:6s}  {out['atom_positions'][0, i, :3]}")
    else:
        # --- Interactive Vispy visualization ---
        from spammm.GUI.RigidBodyVispy import RigidBodyVispy
        vis = RigidBodyVispy(rbd, dt=args.dt, steps_per_frame=10, fire=False)
        print("\nVispy+PyQt5 window opened.")
        print("Controls:")
        print("  LMB click+drag atoms to pull (anchor springs)")
        print("  Mouse wheel = zoom, Arrow keys = pan")
        print("  SPACE = run/stop simulation")
        print("  R = reset velocities, F = toggle FIRE")
        print("  ESC = quit")
        print("  Side panel: Kernel mode, FF params, probe atom, potential map")
        vis.run()


if __name__ == '__main__':
    main()
