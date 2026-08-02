#!/usr/bin/env python3
"""Four-PTCDA demo: deterministic windmill assembly → default AFM → BR-STM.

Paced generator script (yields ctx.frame / ctx.barrier). Run fast from CLI or slow
with visible frames from the Script Runner panel / Scripts menu.

  # Fast (defaults)
  ./run_gui.sh --script demos/gui_scripts/conference_demo.py

  # Paced, 5 MC steps per frame, 300 ms delay, honor barriers
  ./run_gui.sh --script demos/gui_scripts/conference_demo.py \
    --script-delay-ms 300 --script-points-per-frame 5 --script-barriers

Doc: doc/Tasks/GUI_Scripting_DemoRunner.md
"""
import argparse
import os
import numpy as np

from spammm.GUI import gui_script_utils as GSU


def run(window, argv=None, ctx=None):
    p = argparse.ArgumentParser(description='SPAMMM conference demo')
    p.add_argument('--n-step', type=int, default=1000, help='greedy MC steps (1000 reproduces the reference windmill candidate)')
    p.add_argument('--ntrial', type=int, default=512, help='best-of-batch trials per MC step')
    p.add_argument('--mol', type=str, default='PTCDA', help='molecule name in MOL_PATHS')
    p.add_argument('--nmol', type=int, default=4, help='number of molecules')
    p.add_argument('--pme', action='store_true', help='also run the optional PME charge-rings panel')
    args = p.parse_args(argv or [])
    print(f"[conference_demo] start: mol={args.mol} nmol={args.nmol} n_step={args.n_step} ntrial={args.ntrial}", flush=True)

    from spammm.GUI import RigidAssemblyExtension as RA
    from spammm.GUI import AFMExtension as AFM
    from spammm.GUI.RigidAssemblyExtension import MOL_PATHS

    # Validate mol name before any GUI action
    if args.mol not in MOL_PATHS:
        raise ValueError(f"--mol {args.mol!r} not in MOL_PATHS: {sorted(MOL_PATHS)}")
    if not os.path.isfile(MOL_PATHS[args.mol]):
        raise FileNotFoundError(f"MOL_PATHS[{args.mol!r}] missing on disk: {MOL_PATHS[args.mol]}")

    # === Phase 1: Build 4×PTCDA assembly on substrate ===
    yield ctx.frame(f'Configuring {args.nmol}×{args.mol} assembly…')
    GSU.set_label_mode(window, 'None')  # hide atom labels — distracting in demos
    GSU.expand_extension_panel(window, 'rigid_assembly', open=True)
    GSU.set_combo_text(window.ra_source_combo, 'From file')
    GSU.set_combo_text(window.ra_mol_combo, args.mol)
    GSU.set_spin_value(window.ra_nmol_spin, args.nmol)
    GSU.set_spin_value(window.ra_ntrial_spin, args.ntrial)
    # NOTE: do NOT override ra_z_spin — use GUI default (3.0).
    # AFM scan heights are RELATIVE to mol_z (atomPos[:,2].max()), so z_mol
    # only affects MC substrate physics, not the AFM scan distance.
    GSU.click_button(window.ra_build_btn)
    if window.ra_ensemble is None:
        raise RuntimeError(f'Build failed — ra_ensemble is None after clicking Build for {args.nmol}×{args.mol}')
    # Zoom out so all molecules are clearly visible (fit to all atoms + margin)
    if hasattr(window.scene, 'fit_to_atoms'):
        window.scene.fit_to_atoms(margin=3.0)
        GSU.process_events(window)
    yield ctx.frame(f'Built {args.nmol}×{args.mol} ({len(window.ra_ensemble)} mols)')

    # === Phase 2: Greedy MC assembly optimization ===
    yield ctx.frame('Running greedy Monte Carlo assembly…')
    last = None
    n_accept = 0
    for i0, i1 in ctx.batches(args.n_step):
        for i in range(i0, i1):
            last = RA._on_mc_step(window, update_ui=(i + 1 == i1))
            n_accept += int(last is not None and last['accepted'])
        if last is not None:
            print(f"[conference_demo] MC {i1}/{args.n_step}: E={last['E']:.6f} accepted={last['accepted']}", flush=True)
            yield ctx.frame(f"Greedy assembly: {i1}/{args.n_step}, E={last['E']:.6f}")
        else:
            yield ctx.frame(f"Greedy assembly: {i1}/{args.n_step} (no summary)")
    print(f"[conference_demo] MC done: {args.n_step} steps, accepted={n_accept}, E={last['E']:.6f}" if last else "[conference_demo] MC done", flush=True)
    # Re-fit viewport after MC (poses may have shifted)
    if hasattr(window.scene, 'fit_to_atoms'):
        window.scene.fit_to_atoms(margin=3.0)
        GSU.process_events(window)

    # === Phase 3: default AFM product (same function as the AFM button) ===
    yield ctx.barrier('Assembly ready — Continue to AFM')
    GSU.expand_extension_panel(window, 'afm', open=True)
    actual_defaults = {
        'basis': window.afm_basis_combo.currentText(),
        'backend': window.afm_backend_combo.currentText(),
        'projection': window.afm_projection_combo.currentText(),
        'z_plot': float(window.afm_z_height_spin.value()),
    }
    expected_defaults = {'basis': '3ob-3-1', 'backend': 'DFTB FDBM (prolonged)', 'projection': 'prolonged', 'z_plot': 3.0}
    if actual_defaults != expected_defaults:
        raise RuntimeError(f'AFM widgets are not at the required GUI defaults; script will not overwrite them: actual={actual_defaults}, expected={expected_defaults}')
    print(f"[conference_demo] AFM defaults unchanged: {actual_defaults}", flush=True)
    yield ctx.frame('Default 3ob DFTB+ AFM: S1–S4…')
    AFM.run_afm_full_pipeline(window)
    tip = window._afm_results['tip_disp']
    dxy_max = float(np.hypot(tip['dx'], tip['dy']).max())
    bond_length = float(window.afm_bond_length_spin.value())
    if not np.isfinite(dxy_max) or dxy_max < 0.01 or dxy_max > bond_length:
        raise RuntimeError(f'AFM probe distortion is outside physical postcondition: |dxy|_max={dxy_max:.6f} Å, bond_length={bond_length:.3f} Å')
    print(f"[conference_demo] AFM image complete: |dxy|_max={dxy_max:.4f} Å", flush=True)
    yield ctx.frame(f'AFM image complete; |dxy| max={dxy_max:.3f} Å')

    # === Phase 4: default bond-resolved STM (default relative +1 = LUMO) ===
    yield ctx.barrier('Continue to bond-resolved STM')
    print("[conference_demo] BR-STM (GUI-default LUMO)…", flush=True)
    AFM.run_br_stm(window)
    br = np.asarray(window._afm_results['br_stm_grid'])
    if not np.isfinite(br).all() or float(np.max(np.abs(br))) == 0.0:
        raise RuntimeError('BR-STM postcondition failed: grid is non-finite or identically zero')
    print("[conference_demo] BR-STM complete", flush=True)
    yield ctx.frame('BR-STM complete')

    # Optional legacy conference phase.
    if args.pme:
        yield ctx.barrier('Continue to PME charge rings')
        GSU.expand_extension_panel(window, 'rigid_assembly', open=True)
        print("[conference_demo] PME charge rings XY scan…", flush=True)
        RA._on_pme_scan_xy(window)
        print("[conference_demo] PME XY complete", flush=True)
        yield ctx.frame('PME charge-rings XY image complete')
        print("[conference_demo] PME charge rings xV scan (NDR)…", flush=True)
        RA._on_pme_scan_xv(window)
        print("[conference_demo] PME xV complete", flush=True)
        yield ctx.frame('PME charge-rings xV (NDR) complete')

    print("[conference_demo] done", flush=True)
    return {'n_step': args.n_step, 'ntrial': args.ntrial, 'accepted': n_accept, 'E': None if last is None else last['E'], 'dxy_max': dxy_max, 'mol': args.mol, 'nmol': args.nmol}


if __name__ == '__main__':
    import sys
    print('Use: ./run_gui.sh --script demos/gui_scripts/conference_demo.py', file=sys.stderr)
    raise SystemExit(1)
