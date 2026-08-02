#!/usr/bin/env python3
"""Test: run BR-STM, then lower z to trigger _ensure_height_covers recompute."""
import os
from spammm.GUI import gui_script_utils as GSU
from spammm.GUI import AFMExtension as AE


def run(window, argv=None, ctx=None):
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_brstm_zlower'))
    os.makedirs(out, exist_ok=True)

    # Build 4xPTCDA via conference demo path
    GSU.expand_extension_panel(window, 'rigid_assembly', open=True)
    GSU.set_combo_text(window.ra_source_combo, 'From file')
    GSU.set_combo_text(window.ra_mol_combo, 'PTCDA')
    GSU.set_spin_value(window.ra_nmol_spin, 4)
    GSU.set_spin_value(window.ra_ntrial_spin, 512)
    GSU.process_events(window)
    window.ra_build_btn.click()
    GSU.process_events(window)
    print(f'Built {len(window.backend.sys.apos)} atoms')

    # Run BR-STM
    AE.run_br_stm(window)
    GSU.process_events(window)
    print(f'BR-STM done: stm_grid shape={window._afm_results["stm_grid"].shape}, heights={len(window._afm_results["heights"])}')

    # Now try lowering z to trigger _ensure_height_covers
    z_orig = window.afm_z_height_spin.value()
    z_low = max(1.5, float(window._afm_results['heights'].min()) - 1.0)
    print(f'Lowering z from {z_orig:.2f} to {z_low:.2f} (heights range=[{window._afm_results["heights"].min():.2f},{window._afm_results["heights"].max():.2f}])')
    window.afm_z_height_spin.setValue(z_low)
    GSU.process_events(window)
    # Call plot directly (bypasses debounce timer) to test _ensure_height_covers recompute
    AE.plot_afm_slice(window)
    GSU.process_events(window)

    # Check results
    if 'stm_grid' in window._afm_results and window._afm_results['stm_grid'] is not None:
        print(f'OK: stm_grid shape={window._afm_results["stm_grid"].shape}, heights={len(window._afm_results["heights"])}')
    else:
        print(f'FAIL: stm_grid missing or None')
    if 'br_stm_grid' in window._afm_results and window._afm_results['br_stm_grid'] is not None:
        print(f'OK: br_stm_grid shape={window._afm_results["br_stm_grid"].shape}')
    else:
        print(f'FAIL: br_stm_grid missing or None')
    print('done')
