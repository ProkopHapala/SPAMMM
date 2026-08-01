#!/usr/bin/env python3
"""Prepare four charged PTCDA molecules on NaCl for interactive stick-slip dragging.

Run:
  ./run_gui.sh --script spammm/GUI/gui_scripts/ptcda_interactive_drag.py

Pick an oxygen atom and drag it toward the center. The spring acts on that atom,
while the concurrent rigid-body kernel moves all four molecules under intermolecular
PairFF and the charge-sensitive FAF NaCl surface force.
"""
import argparse
import numpy as np

from spammm.GUI import gui_script_utils as GSU


def run(window, argv=None, ctx=None):
    p = argparse.ArgumentParser(description='Interactive 4×PTCDA drag/stick-slip setup')
    p.add_argument('--nmol', type=int, default=4)
    p.add_argument('--spacing', type=float, default=16.0)
    p.add_argument('--spring', type=float, default=0.2, help='oxygen anchor spring [internal PairFF units]')
    p.add_argument('--relax', type=int, default=20, help='all-mobile MD steps per mouse move')
    p.add_argument('--dt', type=float, default=0.02)
    args = p.parse_args(argv or [])

    yield ctx.frame('Configuring charged PTCDA/NaCl drag system…')
    GSU.expand_extension_panel(window, 'rigid_assembly', open=True)
    GSU.set_combo_text(window.ra_source_combo, 'From file')
    GSU.set_combo_text(window.ra_mol_combo, 'PTCDA')
    GSU.set_spin_value(window.ra_nmol_spin, args.nmol)
    GSU.set_spin_value(window.ra_spacing_spin, args.spacing)
    GSU.set_check(window.ra_no_qeq_chk, False)
    GSU.set_check(window.ra_no_faf_chk, False)
    GSU.set_spin_value(window.ra_k_spring_spin, args.spring)
    GSU.set_spin_value(window.ra_drag_nrelax_spin, args.relax)
    GSU.set_spin_value(window.ra_drag_dt_spin, args.dt)
    GSU.click_button(window.ra_build_btn)
    if window.ra_ensemble is None or window.ra_rbd is None:
        raise RuntimeError('PTCDA rigid assembly build failed')

    qmax = max(float(np.abs(pack['REQ_base'][:, 2]).max()) for pack in window.ra_rbd._mb_packs)
    if qmax < 1e-3:
        raise RuntimeError('PTCDA QEq charges are zero; O→Na electrostatic stick-slip cannot be exercised')
    if not window.ra_rbd.faf_mode:
        raise RuntimeError('NaCl FAF is not enabled')

    GSU.set_edit_mode(window, 'ra_drag')
    if hasattr(window.scene, 'fit_to_atoms'):
        window.scene.fit_to_atoms(margin=3.0)
        GSU.process_events(window)
    msg = f'RA Drag ready: pick a red O and pull toward center; all {args.nmol} molecules are mobile (max |Q|={qmax:.3f} e)'
    window.statusBar().showMessage(msg)
    print(f'[ptcda_interactive_drag] {msg}', flush=True)
    print(f'[ptcda_interactive_drag] spring={args.spring} relax/move={args.relax} dt={args.dt} FAF={window.ra_rbd.faf_mode}', flush=True)
    yield ctx.frame(msg)
    return {'nmol': args.nmol, 'qmax': qmax, 'spring': args.spring, 'relax': args.relax, 'dt': args.dt}


if __name__ == '__main__':
    import sys
    print('Use: ./run_gui.sh --script spammm/GUI/gui_scripts/ptcda_interactive_drag.py', file=sys.stderr)
    raise SystemExit(1)
