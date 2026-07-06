#!/usr/bin/env python3
"""GUI control script: symmetric dual-H RC scan with DFTB-relaxed isomers.

  # Full relaxed path (DFTB opt both isomers, interpolate all atoms — bond lengths change):
  ./run_gui.sh --script spammm/GUI/gui_scripts/rc_scan_review.py

  # Fast replay from cached relaxed npz (after one full run above):
  ./run_gui.sh --script spammm/GUI/gui_scripts/rc_scan_review.py -- --preview

  # Force rigid H-only morph (no cache, no C-C changes):
  ./run_gui.sh --script spammm/GUI/gui_scripts/rc_scan_review.py -- --preview --no-cache
"""
import argparse

from spammm.GUI.rc_scan_gui_script import prepare_rc_scan_review


def _parse_argv(argv):
    p = argparse.ArgumentParser(description='RC scan GUI review — symmetric H exchange + relaxed pm-NEB')
    p.add_argument('--name', default='2Quinolone')
    p.add_argument('--pair', type=int, default=0)
    p.add_argument('--dx', type=float, default=0.2)
    p.add_argument('--method', default='pm-NEB relaxed', choices=['pm-NEB relaxed', 'pm-NEB SP', 'rigid DFTB'])
    p.add_argument('--relax', type=int, default=3, help='Jacobi steps in ASCII builder')
    p.add_argument('--preview', action='store_true', help='Skip DFTB; load cached relaxed npz if available, else rigid H-only')
    p.add_argument('--no-cache', action='store_true', help='With --preview: never load cached npz')
    p.add_argument('--single-hbond', action='store_true', help='Scan one H-bond pair only')
    p.add_argument('--no-relax-endpoints', action='store_true')
    p.add_argument('--frame', default='mid', choices=['start', 'mid', 'end'])
    p.add_argument('--no-bond-viz', action='store_true')
    return p.parse_args(argv)


def run(window, argv=None):
    args = _parse_argv(argv or [])
    return prepare_rc_scan_review(
        window,
        name=args.name,
        pair=args.pair,
        dx=args.dx,
        method=args.method,
        relax_steps=args.relax,
        all_hbonds=not args.single_hbond,
        relax_endpoints=not args.no_relax_endpoints and not args.preview,
        run_dftb=not args.preview,
        use_cache=not args.no_cache,
        start_frame=args.frame,
        enable_bond_viz=not args.no_bond_viz,
    )


if __name__ == '__main__':
    print("Use: ./run_gui.sh --script spammm/GUI/gui_scripts/rc_scan_review.py", file=__import__('sys').stderr)
    raise SystemExit(1)
