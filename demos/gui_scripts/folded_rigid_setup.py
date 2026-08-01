#!/usr/bin/env python3
"""GUI control script: load molecule, fold-rigid setup, optionally run.

Usage:
    ./run_gui.sh --script demos/gui_scripts/folded_rigid_setup.py -- --mol data/xyz/H2O.xyz --fit data/fits/h2o_nacl.npz --run
    ./run_gui.sh --script demos/gui_scripts/folded_rigid_setup.py -- --mol data/xyz/H2O.xyz --fit data/fits/h2o_nacl.npz --manip
    ./run_gui.sh --script demos/gui_scripts/folded_rigid_setup.py -- --mol data/xyz/H2O.xyz --fit-fit
"""
import argparse


import sys


def _parse_argv(argv):
    p = argparse.ArgumentParser(description='Folded-Rigid GUI setup — one-command load/run/manip')
    p.add_argument('--mol', default='data/xyz/H2O.xyz', help='Molecule XYZ to load')
    p.add_argument('--substrate', default='data/substrates/NaCl_1x1_L3.xyz', help='Substrate XYZ')
    p.add_argument('--fit', default=None, help='Pre-fitted .npz file (skip fitting)')
    p.add_argument('--fit-fit', action='store_true', help='Run on-the-fly fit instead of loading')
    p.add_argument('--run', action='store_true', help='Run n_iter steps')
    p.add_argument('--step', action='store_true', help='Run a single step')
    p.add_argument('--manip', action='store_true', help='Switch to FR Manip (drag-atom) mode')
    p.add_argument('--n', type=int, default=None, help='Override number of iterations')
    p.add_argument('--dt', type=float, default=None, help='Override timestep')
    p.add_argument('--x', type=float, default=None, help='Initial COM x')
    p.add_argument('--y', type=float, default=None, help='Initial COM y')
    p.add_argument('--z', type=float, default=2.5, help='Initial COM z')
    p.add_argument('--k', type=float, default=None, help='Override manipulation spring constant')
    return p.parse_args(argv)


def run(window, argv=None):
    args = _parse_argv(argv or [])
    from spammm.GUI import FoldedRigidExtension as FR
    from spammm.GUI.gui_script_utils import set_spin_value

    fit = None if args.fit_fit else args.fit

    if args.k is not None:
        set_spin_value(window.fr_k_spring_spin, args.k)

    FR.prepare_folded_rigid(
        window,
        mol=args.mol,
        fit=fit,
        substrate=args.substrate,
        run=args.run,
        step=args.step,
        manip=args.manip,
        n=args.n,
        dt=args.dt,
        x=args.x,
        y=args.y,
        z=args.z,
    )
    return window.fr_rbd


if __name__ == '__main__':
    print("Use: ./run_gui.sh --script demos/gui_scripts/folded_rigid_setup.py", file=sys.stderr)
    raise SystemExit(1)
