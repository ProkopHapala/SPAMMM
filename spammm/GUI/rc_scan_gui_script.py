"""Programmatic GUI workflow for reaction-coordinate scan review."""
import os
import glob

from spammm.GUI.gui_script_utils import expand_extension_panel, process_events, set_spin_value
from spammm.GUI.AsciiArtExtension import load_ascii_example
from spammm.GUI.ReactionCoordinateExtension import import_from_graph, configure_scan, run_scan, run_preview_scan, show_scan_frame, enable_bond_length_visualization, load_npz_path
from spammm.quantum.hbond_scan import build_ascii_hbond_system
from spammm.topology.AtomicGraph import AtomicGraph


def _cache_npz_path(name):
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'testplot_rc_scan_gui')
    hits = sorted(glob.glob(os.path.join(out_dir, f'{name}_sym_pm_neb_relaxed.npz')), key=os.path.getmtime, reverse=True)
    return hits[0] if hits else os.path.join(out_dir, f'{name}_sym_pm_neb_relaxed.npz')


def prepare_rc_scan_review(window, name='2Quinolone', pair=0, dx=0.2, method='pm-NEB relaxed', relax_steps=3, all_hbonds=True, relax_endpoints=True, run_dftb=True, use_cache=True, start_frame='mid', enable_bond_viz=True):
    """Drive GUI into ready-to-review RC scan state.

    Default: DFTB-relaxed isomers at u=0/u=1, all atoms interpolated (bond lengths change).
    run_dftb=False (--preview): load cached relaxed npz if present, else rigid H-only morph.
    """
    expand_extension_panel(window, 'ascii', open=True)
    load_ascii_example(window, name)
    set_spin_value(window.kek_relax_spin, relax_steps)
    atoms = build_ascii_hbond_system(name)
    # Clear graph from previous demos so ensure_sys() won't overwrite our sys
    window.backend.graph = AtomicGraph()
    window.backend.sys = atoms
    if hasattr(window, 'refresh_view'):
        window.refresh_view()
    process_events(window)

    expand_extension_panel(window, 'reaction_coord', open=True)
    import_from_graph(window)
    configure_scan(window, pair=pair, dx=dx, method=method, all_hbonds=all_hbonds, relax_endpoints=relax_endpoints)

    cache_path = _cache_npz_path(name)
    if not run_dftb and use_cache and os.path.isfile(cache_path):
        print(f"REVIEW: loading cached relaxed trajectory: {cache_path}")
        load_npz_path(window, cache_path)
    elif run_dftb:
        ds = run_scan(window)
        if ds is None:
            n_hb = len(getattr(window, 'rc_hbonds', []))
            raise RuntimeError(f'run_scan returned None — no H-bonds found (rc_hbonds={n_hb}). '
                               f'The graph may not have been cleared from a previous demo.')
    else:
        print("NOTE: --preview without cache — rigid endpoints, only H atoms move along slider.")
        print("      Run once WITHOUT --preview for DFTB-relaxed path (~1–2 min), then --preview reuses cache.")
        run_preview_scan(window)

    if window.rc_dataset is None:
        raise RuntimeError('RC scan dataset missing after run')

    if enable_bond_viz:
        enable_bond_length_visualization(window, True)

    if run_dftb and window.rc_dataset is not None and window.rc_dataset.charges is not None:
        from spammm.GUI.rc_esp_view import open_rc_esp_animation
        open_rc_esp_animation(window)

    n = window.rc_dataset.nframes
    fi = {'start': 0, 'mid': n // 2, 'end': max(0, n - 1)}.get(start_frame, n // 2)
    show_scan_frame(window, fi)
    process_events(window)

    tag = window.rc_dataset.meta.get('scan_type', 'scan')
    out_dir = os.path.dirname(cache_path)
    os.makedirs(out_dir, exist_ok=True)
    if run_dftb and tag == 'pm_neb_relaxed' and window.rc_dataset.meta.get('endpoints_relaxed'):
        window.rc_dataset.save_npz(cache_path)
    n_hb = len(window.rc_hbonds) if window.rc_all_hbonds_chk.isChecked() else 1
    rigid = window.rc_dataset.meta.get('rigid_endpoints', tag == 'pm_neb_preview')
    print(f"REVIEW: GUI ready — {n_hb} H-bond(s), {n} frames ({tag}), bond viz={'on' if enable_bond_viz else 'off'}")
    if rigid:
        print("REVIEW: rigid endpoints — scrub slider moves H only; run without --preview for full geometry")
    else:
        print("REVIEW: scrub slider — all atoms + bond Δ vs u=0 (blue=shorter, red=longer)")
    print(f"REVIEW: {cache_path if os.path.isfile(cache_path) else out_dir}")
    return window.rc_dataset
