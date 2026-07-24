#!/usr/bin/env python3
"""Demo script for RigidBodyPairFF + Vispy visualization.

Loads uracil (static) and HCOOH (dynamic) molecules, sets up pairwise
force field interactions with electron pairs, and launches interactive
Vispy visualization with mouse picking.

Multi-body (shared allmol buffers):
    python3 demos/demo_pairff.py --bodies 4 --active 0
    # N copies of HCOOH on a grid; click any molecule to make it mobile
    python3 demos/demo_pairff.py --bodies 4 --active 2 --no-vis
    # Mixed species (any XYZ list; formamide.xyz = HCONH2; no NTCDA.xyz yet → use PTCDA):
    python3 demos/demo_pairff.py --mols PTCDA.xyz HCOOH.xyz formamide.xyz --spacing 12
    python3 demos/demo_pairff.py --mols PTCDA.xyz HCOOH.xyz formamide.xyz --no-vis --steps 50

FAF substrate (NaCl folded basis; map = PairFF + FAF at molecule z):
    python3 demos/demo_pairff.py --bodies 4 --faf
    python3 demos/demo_pairff.py --bodies 4 --faf --faf-fit data/fits/hcooh_nacl.npz --z-init 3.5

Usage:
    python3 demos/demo_pairff.py                    # interactive Vispy (unified kernel)
    python3 demos/demo_pairff.py --pairff-mode legacy
    python3 demos/demo_pairff.py --no-vis           # headless relaxation test
    python3 demos/demo_pairff.py --no-vis --steps 500  # custom step count

Kernels (switchable from GUI or --pairff-mode):
  unified — rigid_body_pairff_unified_kernel: single compact-exp loop (default)
            (y=(1-b*rho)^8, rho=r2/(sqrt(r2+w*w)+w))
  legacy  — rigid_body_pairff_kernel: Morse+Coulomb / Lorentzian
  multi   — allmol shared buffers when --bodies > 1 or --mols
See:
  - examples/density_comparison/HBondFF/fit_radial.py (--compact-exp-demo)
  - doc/Tasks/PairFF_FAF_Substrate.md
  - kernels/rigid.cl (rigid_body_pairff_unified_allmol[_faf]_kernel)
"""
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_XYZ = os.path.join(REPO_ROOT, 'data', 'xyz')
DATA_FITS = os.path.join(REPO_ROOT, 'data', 'fits')
DEFAULT_HCOOH_FIT = os.path.join(DATA_FITS, 'hcooh_nacl.npz')

from spammm.topology.FFparams import load_xyz_with_REQs
from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF


def load_molecule(fname):
    """Load XYZ molecule with REQ parameters."""
    apos, REQs, enames, Zs, lvec = load_xyz_with_REQs(fname)
    return np.asarray(apos, dtype=np.float32), REQs, enames


def _resolve_xyz(path_or_name):
    """Accept absolute/relative path or basename under data/xyz/."""
    if os.path.isfile(path_or_name):
        return os.path.abspath(path_or_name)
    cand = os.path.join(DATA_XYZ, path_or_name)
    if os.path.isfile(cand):
        return cand
    if not path_or_name.endswith('.xyz'):
        cand2 = os.path.join(DATA_XYZ, path_or_name + '.xyz')
        if os.path.isfile(cand2):
            return cand2
    raise FileNotFoundError(f"Molecule XYZ not found: {path_or_name} (tried {cand})")


def _grid_positions(n, spacing=6.0, z=0.0):
    """Place N CoMs on an XY grid centered at origin."""
    nx = int(np.ceil(np.sqrt(n)))
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        ix, iy = i % nx, i // nx
        pos[i, 0] = (ix - 0.5 * (nx - 1)) * spacing
        pos[i, 1] = (iy - 0.5 * (nx - 1)) * spacing
        pos[i, 2] = z
    return pos


def _build_multibody(molecules, labels, active, spacing, args):
    """Shared constructor for identical or mixed multi-body scenes."""
    n = len(molecules)
    if active < 0 or active >= n:
        raise SystemExit(f'--active {active} out of range for {n} molecules')
    if args.pairff_mode != 'unified':
        raise SystemExit('multi-body mode requires --pairff-mode unified')
    body_pos = _grid_positions(n, spacing=spacing, z=0.0)
    print(f"Multi-body PairFF: {n} molecules, active={active}, spacing={spacing}")
    for i, lab in enumerate(labels):
        print(f"  [{i}] {lab}  CoM={body_pos[i]}")
    rbd = RigidBodyPairFF.from_molecules(
        molecules, body_pos, active_body=active,
        He=args.he, rc=args.rc, w=args.w, morse_alpha=args.alpha, k_z=args.kz,
        z_target=0.0, Hs=args.hs,
        epair_dist=args.epair_dist, sigma_dist=args.sigma_dist, beta=args.beta,
    )
    print(f"Active atoms+epairs: {len(rbd.enames)} — {rbd.enames}")
    print(f"Env molecules: {rbd.n_env_mols}  env sites: {rbd.n_env_sites}")
    print(f"Active types: {rbd.dyn_type_host}")
    return rbd


def _load_or_fit_faf(mol_xyz, fit_path=None):
    """Load cached FAF fit or fit HCOOH@NaCl and save under data/fits/."""
    from spammm.surfaces.FoldedRigid import fit_folded_for_molecule, load_fit, save_fit
    path = fit_path or DEFAULT_HCOOH_FIT
    if os.path.isfile(path):
        print(f"Loading FAF fit: {path}")
        return load_fit(path)
    print(f"Fitting FAF for {mol_xyz} → {path} (first run; may take a minute)...")
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fit = fit_folded_for_molecule(mol_xyz)
    save_fit(fit, path)
    print(f"Saved FAF fit: {path}")
    return fit


def _attach_faf(rbd, args, mol_xyz_for_fit):
    """Raise to surface, bind FAF, enable fused kernels (map uses rbd.faf_fit)."""
    from spammm.surfaces.FoldedRigid import Z_SURF_TOP
    fit = _load_or_fit_faf(mol_xyz_for_fit, args.faf_fit)
    rbd.attach_pairff_faf(fit, z_init=args.z_init, k_z=0.0, enable=True)
    print(f"FAF ON: z = Z_SURF_TOP({Z_SURF_TOP}) + {args.z_init} = {rbd.map_z:.3f}  k_z=0")
    print(f"  map = PairFF(env) + FAF(probe) at z=active CoM; types={fit['coeffs'].shape[0]}")
    return rbd


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
    parser.add_argument('--kz', type=float, default=5.0, help='Z-constraint strength (ignored / set 0 when --faf)')
    parser.add_argument('--epair-dist', type=float, default=1.4, help='Epair distance from host [Å]')
    parser.add_argument('--sigma-dist', type=float, default=1.0, help='Sigma hole distance from H [Å] (0=disabled)')
    parser.add_argument('--pairff-mode', choices=['legacy', 'unified'], default='unified',
                        help='Force kernel: legacy Morse+Lorentzian or unified compact-exp')
    parser.add_argument('--bodies', type=int, default=0,
                        help='Multi-body: N copies of HCOOH (shared allmol buffers). 0 = classic uracil+HCOOH')
    parser.add_argument('--mols', nargs='+', default=None,
                        help='Multi-body mixed species: XYZ paths or basenames under data/xyz/ '
                             '(e.g. PTCDA.xyz HCOOH.xyz formamide.xyz). Overrides --bodies.')
    parser.add_argument('--active', type=int, default=0, help='Active body index for multi-body mode')
    parser.add_argument('--spacing', type=float, default=6.0, help='XY grid spacing for multi-body')
    parser.add_argument('--faf', action='store_true',
                        help='Enable FAF NaCl substrate (fused PairFF+FAF; map shows PairFF+FAF)')
    parser.add_argument('--faf-fit', default=None,
                        help='Path to FAF .npz fit (default: data/fits/hcooh_nacl.npz; fit if missing)')
    parser.add_argument('--z-init', type=float, default=3.5,
                        help='Molecule height above surface top when --faf (Å); CoM = Z_SURF_TOP + z_init')
    args = parser.parse_args()

    if args.mols:
        paths = [_resolve_xyz(p) for p in args.mols]
        molecules, labels = [], []
        for p in paths:
            apos, REQs, enames = load_molecule(p)
            molecules.append((apos, enames, REQs))
            labels.append(os.path.basename(p))
        rbd = _build_multibody(molecules, labels, int(args.active), args.spacing, args)
        faf_mol = paths[0]
    elif args.bodies and args.bodies > 1:
        dyn_apos, dyn_REQs, dyn_enames = load_molecule(os.path.join(DATA_XYZ, 'HCOOH.xyz'))
        n = int(args.bodies)
        molecules = [(dyn_apos, dyn_enames, dyn_REQs)] * n
        labels = [f'HCOOH#{i}' for i in range(n)]
        rbd = _build_multibody(molecules, labels, int(args.active), args.spacing, args)
        faf_mol = os.path.join(DATA_XYZ, 'HCOOH.xyz')
    else:
        # --- Classic: uracil static + HCOOH dynamic ---
        static_apos, static_REQs, static_enames = load_molecule(os.path.join(DATA_XYZ, 'uracil.xyz'))
        dyn_apos, dyn_REQs, dyn_enames = load_molecule(os.path.join(DATA_XYZ, 'HCOOH.xyz'))

        print(f"Static (uracil): {len(static_enames)} atoms — {static_enames}")
        print(f"Dynamic (HCOOH): {len(dyn_enames)} atoms — {dyn_enames}")
        print(f"PairFF mode: {args.pairff_mode}")

        static_center = static_apos[:, :2].mean(axis=0)
        body_pos = np.array([static_center[0], static_center[1], 3.0], dtype=np.float32)

        rbd = RigidBodyPairFF.from_two_molecules(
            dyn_apos=dyn_apos, dyn_enames=dyn_enames, dyn_REQs=dyn_REQs,
            static_apos=static_apos, static_enames=static_enames, static_REQs=static_REQs,
            body_pos=body_pos,
            He=args.he, rc=args.rc, w=args.w, morse_alpha=args.alpha, k_z=args.kz,
            z_target=0.0, Hs=args.hs,
            epair_dist=args.epair_dist, sigma_dist=args.sigma_dist,
            mode=args.pairff_mode, beta=args.beta,
        )
        print(f"Dynamic atoms+epairs: {len(rbd.enames)} — {rbd.enames}")
        print(f"Static atoms+epairs: {len(rbd.static_enames)} — {rbd.static_enames}")
        print(f"Dynamic types: {rbd.dyn_type_host}")
        print(f"Static types:  {rbd.static_type_host}")
        faf_mol = os.path.join(DATA_XYZ, 'HCOOH.xyz')

    if args.faf:
        if args.pairff_mode != 'unified':
            raise SystemExit('--faf requires --pairff-mode unified')
        rbd = _attach_faf(rbd, args, faf_mol)

    if args.no_vis:
        print(f"\nRunning FIRE relaxation (max {args.steps} steps, dt={args.dt})...")
        result = rbd.relax_pairff(max_steps=args.steps, dt=args.dt, f_tol=1e-4, t_tol=1e-4, record=True)
        print(f"Converged: {result['converged']} in {result['steps']} steps")
        print(f"  F={result['F']:.6f}  T={result['T']:.6f}  E={result['E']:.6f}")

        out = rbd.download_outputs()
        a = int(getattr(rbd, 'active_body', 0))
        print(f"\nFinal CoM pos [active={a}]: {out['pos'][a, :3]}")
        print(f"Final quat:    {out['quats'][a]}")
        print(f"Final atom positions (world):")
        for i, (e, t) in enumerate(zip(rbd.enames, rbd.dyn_type_host)):
            tag = 'epair' if t == 1 else ('sigma' if t == 2 else e)
            print(f"  [{i:2d}] {tag:6s}  {out['atom_positions'][0, i, :3]}")
        if getattr(rbd, '_mb_packs', None) is not None:
            rbd.sync_active_pose_from_gpu()
            print(f"Host multi-body CoMs:\n{rbd._mb_pos}")
    else:
        from spammm.GUI.RigidBodyVispy import RigidBodyVispy
        vis = RigidBodyVispy(rbd, dt=args.dt, steps_per_frame=10, fire=True)
        print("\nVispy+PyQt5 window opened.")
        print("Controls:")
        print("  LMB click+drag atoms to pull (anchor springs)")
        if getattr(rbd, '_mb_packs', None) is not None:
            print("  Multi-body: LMB on any molecule → make it active (index only; poses persist)")
        if getattr(rbd, 'faf_mode', False):
            print("  Map = PairFF(env molecules) + FAF(NaCl) at active CoM height")
        print("  Mouse wheel = zoom, Arrow keys = pan")
        print("  SPACE = run/stop simulation")
        print("  R = reset velocities, F = toggle FIRE (default ON)")
        print("  ESC = quit")
        print("  Side panel: Kernel mode, FF params, probe atom, potential map")
        vis.run()


if __name__ == '__main__':
    main()
