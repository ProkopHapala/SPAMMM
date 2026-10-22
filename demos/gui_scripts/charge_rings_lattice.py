#!/usr/bin/env python3
"""GUI control script: open Charge Rings panel, load a lattice PME8 preset, run Map.

Usage:
    ./run_gui.sh --script demos/gui_scripts/charge_rings_lattice.py
    ./run_gui.sh --script demos/gui_scripts/charge_rings_lattice.py -- --preset /path/preset.json --V 1.8 --live
"""
import argparse
import sys


def _parse_argv(argv):
    p = argparse.ArgumentParser(description='ChargeRings lattice PME8 (periodic) mode setup')
    p.add_argument('--preset', default='/home/prokop/git/SPAMMM_export_for_ChaoLi/results/temirov_gui_presets/temirov_7T.json', help='lattice_pme8 preset JSON')
    p.add_argument('--V', type=float, default=None, help='override VBias')
    p.add_argument('--solver', choices=['auto', 'pme4', 'pme8'], default=None, help='lattice solver (pme4 needs <=4 sites; auto-sets 2x2)')
    p.add_argument('--npix', type=int, default=None, help='map resolution')
    p.add_argument('--live', action='store_true', help='enable Live recompute on spin edits')
    return p.parse_args(argv)


def run(window, argv=None):
    args = _parse_argv(argv or [])
    from spammm.GUI import gui_script_utils as GSU
    from spammm.GUI import ChargeRingsExtension as CR
    GSU.expand_extension_panel(window, 'charge_rings', open=True)
    CR.lat_load_json(window, args.preset)
    if args.solver is not None:
        window.lat_solver_combo.setCurrentText(args.solver)  # goes through _lat_solver_changed (pme4 -> 2x2)
    if args.npix is not None:
        GSU.set_spin_value(window.lat_spins['npix'], args.npix)
    if args.V is not None:
        GSU.set_spin_value(window.lat_spins['VBias'], args.V)
    if args.live:
        GSU.set_check(window.lat_live_chk, True)
    CR.lat_run_map(window)
    GSU.process_events(window)
    return window._lat_model


if __name__ == '__main__':
    print("Use: ./run_gui.sh --script demos/gui_scripts/charge_rings_lattice.py", file=sys.stderr)
    raise SystemExit(1)
