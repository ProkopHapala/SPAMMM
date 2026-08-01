#!/usr/bin/env python3
"""Conference demo — full SPAMMM workflow: build 4×PTCDA → greedy assembly →
AFM → BR-STM → PME charge rings.

Paced generator script (yields ctx.frame / ctx.barrier). Run fast from CLI or slow
with visible frames from the Script Runner panel / Scripts menu.

  # Fast (defaults)
  ./run_gui.sh --script spammm/GUI/gui_scripts/conference_demo.py -- --n-step 200

  # Paced, 5 MC steps per frame, 300 ms delay, honor barriers
  ./run_gui.sh --script spammm/GUI/gui_scripts/conference_demo.py \
    --script-delay-ms 300 --script-points-per-frame 5 --script-barriers -- --n-step 200

Doc: doc/Tasks/GUI_Scripting_DemoRunner.md
"""
import argparse
import os

from spammm.GUI import gui_script_utils as GSU


def run(window, argv=None, ctx=None):
    p = argparse.ArgumentParser(description='SPAMMM conference demo')
    p.add_argument('--n-step', type=int, default=200, help='greedy MC steps')
    p.add_argument('--mol', type=str, default='PTCDA', help='molecule name in MOL_PATHS')
    p.add_argument('--nmol', type=int, default=4, help='number of molecules')
    args = p.parse_args(argv or [])
    print(f"[conference_demo] start: mol={args.mol} nmol={args.nmol} n_step={args.n_step}", flush=True)

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
    GSU.expand_extension_panel(window, 'rigid_assembly', open=True)
    GSU.set_combo_text(window.ra_source_combo, 'From file')
    GSU.set_combo_text(window.ra_mol_combo, args.mol)
    GSU.set_spin_value(window.ra_nmol_spin, args.nmol)
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
    for i0, i1 in ctx.batches(args.n_step):
        for i in range(i0, i1):
            last = RA._on_mc_step(window, update_ui=(i + 1 == i1))
        if last is not None:
            print(f"[conference_demo] MC {i1}/{args.n_step}: E={last['E']:.6f} accepted={last['accepted']}", flush=True)
            yield ctx.frame(f"Greedy assembly: {i1}/{args.n_step}, E={last['E']:.6f}")
        else:
            yield ctx.frame(f"Greedy assembly: {i1}/{args.n_step} (no summary)")
    print(f"[conference_demo] MC done: {args.n_step} steps, E={last['E']:.6f}" if last else "[conference_demo] MC done", flush=True)
    # Re-fit viewport after MC (poses may have shifted)
    if hasattr(window.scene, 'fit_to_atoms'):
        window.scene.fit_to_atoms(margin=3.0)
        GSU.process_events(window)

    # === Phase 3: AFM simulation (staged for visible boundaries) ===
    yield ctx.barrier('Assembly ready — Continue to AFM')
    GSU.expand_extension_panel(window, 'afm', open=True)
    # Ensure DFTB FDBM backend (BR-STM requires it; no silent Morse fallback)
    if hasattr(window, 'afm_backend_combo'):
        GSU.set_combo_text(window.afm_backend_combo, 'DFTB FDBM (prolonged)')
        print(f"[conference_demo] AFM backend: {window.afm_backend_combo.currentText()}", flush=True)
    # NOTE: do NOT override AFM scan params (hmin/hmax/hstep/amp/scan_range/z_plot).
    # Use GUI defaults — the script must use the same code path as manual clicking.
    # AFM heights are RELATIVE to mol_z, so defaults (hmin=3.7, hmax=4.7, z_plot=3.0)
    # always scan 3.7-4.7 Å above the molecule regardless of z_mol.
    print(f"[conference_demo] AFM params (GUI defaults): hmin={window.afm_hmin_spin.value()} hmax={window.afm_hmax_spin.value()} z_plot={window.afm_z_height_spin.value()} scan_range={window.afm_scan_range_spin.value()}", flush=True)
    # Select LUMO for STM/BR-STM: MO list "1" relative to HOMO (0=HOMO, +1=LUMO)
    if hasattr(window, 'afm_stm_mo_list'):
        window.afm_stm_mo_list.setText('1')
    if hasattr(window, 'afm_stm_relative_mo'):
        window.afm_stm_relative_mo.setChecked(True)
    print("[conference_demo] STM MO selection: LUMO (relative +1 from HOMO)", flush=True)
    print("[conference_demo] AFM S1: DFTB+ SCF…", flush=True)
    yield ctx.frame('AFM S1: DFTB+ SCF…')
    AFM.run_afm_stage1(window)
    print("[conference_demo] AFM S2: density projection…", flush=True)
    yield ctx.frame('AFM S1 complete; projecting density…')
    AFM.run_afm_stage2(window)
    print("[conference_demo] AFM S3: potentials…", flush=True)
    yield ctx.frame('AFM S2 complete; building potentials…')
    AFM.run_afm_stage3(window)
    print("[conference_demo] AFM S4: probe relaxation…", flush=True)
    yield ctx.frame('AFM S3 complete; relaxing probe particle…')
    AFM.run_afm_stage4(window)
    print("[conference_demo] AFM image complete", flush=True)
    yield ctx.frame('AFM image complete')

    # === Phase 4: Bond-resolved STM (LUMO of PTCDA) ===
    yield ctx.barrier('Continue to bond-resolved STM')
    print("[conference_demo] BR-STM (LUMO)…", flush=True)
    AFM.run_br_stm(window)
    print("[conference_demo] BR-STM complete", flush=True)
    yield ctx.frame('BR-STM complete')

    # === Phase 5: PME charge rings ===
    yield ctx.barrier('Continue to PME charge rings')
    GSU.expand_extension_panel(window, 'rigid_assembly', open=True)
    print("[conference_demo] PME charge rings XY scan…", flush=True)
    RA._on_pme_scan_xy(window)
    print("[conference_demo] PME complete", flush=True)
    yield ctx.frame('PME charge-rings image complete')

    print("[conference_demo] done", flush=True)
    return {'n_step': args.n_step, 'mol': args.mol, 'nmol': args.nmol}


if __name__ == '__main__':
    import sys
    print('Use: ./run_gui.sh --script spammm/GUI/gui_scripts/conference_demo.py', file=sys.stderr)
    raise SystemExit(1)
