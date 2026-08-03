#!/usr/bin/env python3
"""Static-obstacle drag demo: drag a dynamic molecule through frozen obstacles.

Loads a benzoic acid dimer from XYZ, splits it into two rigid bodies via
connected components (From editor), freezes one as a static obstacle, and drags
the other TOWARD it so they interact. Mid-drag, toggles the static molecule to
dynamic and back to demonstrate the static↔dynamic switching.

Frames are captured from the VisPy canvas → GIF/MP4.

Run:
  ./run_gui.sh --script demos/gui_scripts/static_obstacle_drag_demo.py

  # Bigger drag, more relaxation:
  ./run_gui.sh --script demos/gui_scripts/static_obstacle_drag_demo.py -- --drag-x 10 --n-relax 300
"""
import argparse
import os
import time
import numpy as np
from PIL import Image

from spammm.GUI import gui_script_utils as GSU


def run(window, argv=None, ctx=None):
    p = argparse.ArgumentParser(description='Static-obstacle drag demo: dimer split + drag + toggle')
    p.add_argument('--dimer', type=str, default=None,
                   help='path to dimer XYZ (default: data/xyz/benzoicacid_dimer.xyz)')
    p.add_argument('--spring', type=float, default=0.2, help='anchor spring [PairFF internal units]')
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--n-relax', type=int, default=200, help='FIRE relaxation steps per drag step')
    p.add_argument('--drag-step', type=float, default=0.15, help='anchor movement per drag step [Å]')
    p.add_argument('--drag-x', type=float, default=8.0, help='total drag distance [Å]')
    p.add_argument('--out', type=str, default=None, help='output dir (default: debug/static_obstacle_drag_demo)')
    p.add_argument('--format', type=str, default='both', choices=['gif', 'mp4', 'both'], help='output format (default: both)')
    args = p.parse_args(argv or [])

    from spammm.GUI import RigidAssemblyExtension as RA
    from spammm.GUI.RigidAssemblyExtension import (
        _set_anchors, _sync_ensemble_from_gpu, _sync_display,
        _update_ra_substrate_overlay, _update_anchor_visuals,
        _toggle_body_state, _recompute_ra_combined_map, _body_state_counts,
        _on_probe_preset)

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dimer_path = args.dimer or os.path.join(REPO_ROOT, 'data', 'xyz', 'benzoicacid_dimer.xyz')
    if not os.path.isfile(dimer_path):
        raise FileNotFoundError(f'Dimer XYZ not found: {dimer_path}')

    outdir = args.out or os.path.join('debug', 'static_obstacle_drag_demo')
    os.makedirs(outdir, exist_ok=True)

    # === Phase 1: Load dimer into editor ===
    yield ctx.frame(f'Loading benzoic acid dimer…')
    GSU.set_label_mode(window, 'None')
    GSU.load_molecule(window, dimer_path)
    print(f'[static_obstacle_drag_demo] Loaded dimer: {len([a for a in window.backend.graph.atoms.values() if a.alive])} atoms', flush=True)

    # Check connected components
    comps = window.backend.graph.find_connected_components()
    print(f'[static_obstacle_drag_demo] Connected components: {len(comps)} (sizes: {[len(c) for c in comps]})', flush=True)
    if len(comps) < 2:
        raise RuntimeError(f'Dimer has only {len(comps)} connected component(s) — need 2 for static/dynamic split')

    # === Phase 2: Build RA from editor (splits by connected components) ===
    yield ctx.frame(f'Building rigid assembly from editor (2 fragments)…')
    GSU.expand_extension_panel(window, 'rigid_assembly', open=True)
    GSU.set_combo_text(window.ra_source_combo, 'From editor')
    GSU.set_check(window.ra_no_qeq_chk, False)
    GSU.set_check(window.ra_no_faf_chk, False)  # FAF on — NaCl substrate
    GSU.set_spin_value(window.ra_k_spring_spin, args.spring)
    GSU.set_spin_value(window.ra_drag_dt_spin, args.dt)
    GSU.set_spin_value(window.ra_drag_nrelax_spin, args.n_relax)
    GSU.click_button(window.ra_build_btn)
    if window.ra_ensemble is None or window.ra_rbd is None:
        raise RuntimeError('RA build from editor failed')
    rbd = window.ra_rbd
    ens = window.ra_ensemble
    n_bodies = len(ens)
    print(f'[static_obstacle_drag_demo] Built {n_bodies} rigid bodies from editor', flush=True)
    if n_bodies < 2:
        raise RuntimeError(f'Expected ≥2 bodies, got {n_bodies}')

    # Fit viewport
    if hasattr(window.scene, 'fit_to_atoms'):
        window.scene.fit_to_atoms(margin=3.0)
        GSU.process_events(window)
    yield ctx.frame(f'Built {n_bodies} rigid bodies (dimer split by connected components)')

    # === Phase 3: Freeze body 0 as static obstacle, keep body 1 dynamic ===
    yield ctx.frame(f'Freezing body 0 as static obstacle…')
    _toggle_body_state(window, 0)  # body 0 → static
    n_dyn, n_stat, n_del = _body_state_counts(window)
    print(f'[static_obstacle_drag_demo] States: dynamic={n_dyn} static={n_stat} deleted={n_del}', flush=True)

    # Compute combined probe map — H+ probe shows attractive blue minima at electron pairs
    _on_probe_preset(window, 'Hp')
    _recompute_ra_combined_map(window)
    GSU.process_events(window)
    yield ctx.frame(f'Body 0 frozen (static), body 1 dynamic — probe map computed')

    # === Phase 4: Find O atom in dynamic body 1 and drag TOWARD static body 0 ===
    # Get CoM positions to determine drag direction
    pos_all, _ = ens.get_poses()
    com0 = pos_all[0, :2]  # static body CoM (xy)
    com1 = pos_all[1, :2]  # dynamic body CoM (xy)
    drag_dir = com0 - com1  # direction from dynamic → static
    drag_dir_norm = np.linalg.norm(drag_dir)
    if drag_dir_norm < 1e-6:
        drag_dir = np.array([1.0, 0.0])
        drag_dir_norm = 1.0
    drag_unit = (drag_dir / drag_dir_norm).astype(np.float32)
    print(f'[static_obstacle_drag_demo] Drag direction: {drag_unit} (distance={drag_dir_norm:.2f} Å)', flush=True)

    # Find a real O atom in the dynamic body (body 1)
    pack1 = rbd._mb_packs[1]
    real_types1 = np.asarray(pack1['types'])
    real_enames1 = pack1['enames']
    o_local = -1
    for j, (e, t) in enumerate(zip(real_enames1, real_types1)):
        if t == 0 and e == 'O':
            o_local = j
            break
    if o_local < 0:
        # fallback: use first real atom
        for j, t in enumerate(real_types1):
            if t == 0:
                o_local = j
                break
    if o_local < 0:
        raise RuntimeError('No real atom found in dynamic body pack')
    n_per_mol1 = len(real_types1)
    anchor_flat = int(rbd.mol_offsets[1] + o_local)
    print(f'[static_obstacle_drag_demo] Anchor: body=1, atom[{o_local}] ({real_enames1[o_local]}), flat={anchor_flat}', flush=True)

    # Get initial world position of anchored atom
    atoms0 = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms0); rbd.queue.finish()
    anchor_base = atoms0[anchor_flat, :3].copy()
    print(f'[static_obstacle_drag_demo] Anchor base position: {anchor_base}', flush=True)

    # === Phase 5: Relaxation (settle, no anchor) ===
    yield ctx.frame(f'Relaxing on surface…')
    _set_anchors(window, -1, np.zeros(3, dtype=np.float32))
    rbd.run_multimol_md(300, dt=args.dt, lin_damp=0.95, ang_damp=0.95, faf=None)
    _sync_ensemble_from_gpu(window)
    _sync_display(window)
    GSU.process_events(window)
    atoms_relax = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms_relax); rbd.queue.finish()
    anchor_base = atoms_relax[anchor_flat, :3].copy()
    # Use eval_energy_system for a proper total energy (kernel .w channel double-counts pairs)
    pos_r, quat_r = window.ra_ensemble.get_poses()
    E_relax = float(rbd.eval_energy_system(pos_r, quat_r, k_pack=0.0))
    print(f'[static_obstacle_drag_demo] Relaxed: E={E_relax:.4f}, anchor={anchor_base}', flush=True)

    # Set anchor at relaxed position
    _set_anchors(window, anchor_flat, anchor_base)
    rbd.reset_dynamics_state()

    # === Trajectory lines ===
    import vispy.scene as vscene
    traj_dragged = [anchor_base.copy()]
    traj_anchor  = [anchor_base.copy()]
    traj_dragged_line = vscene.visuals.Line(parent=window.scene.view.scene, color=(1, 0, 0, 0.9),
                                            width=2.0, antialias=True, method='gl', connect='strip')
    traj_anchor_line  = vscene.visuals.Line(parent=window.scene.view.scene, color=(0.2, 0.5, 1, 0.9),
                                            width=2.0, antialias=True, method='gl', connect='strip')
    for v in (traj_dragged_line, traj_anchor_line):
        v.set_gl_state('translucent', depth_test=False)
        v.order = 9

    # === Phase 6: Drag loop — drag TOWARD static body, with mid-drag toggle ===
    n_steps = int(np.ceil(args.drag_x / args.drag_step))
    # Toggle at 50% of drag: static→dynamic→static
    toggle_step = n_steps // 2
    print(f'[static_obstacle_drag_demo] Drag: {n_steps} steps × {args.drag_step:.2f} Å = {args.drag_x:.1f} Å total, '
          f'toggle at step {toggle_step}', flush=True)
    yield ctx.frame(f'Dragging dynamic body TOWARD static obstacle ({n_steps} steps)…')

    pil_frames = []
    t0 = time.perf_counter()

    def capture_frame(step, E, anchor_pos, anchor_target, label=''):
        traj_dragged.append(np.asarray(anchor_pos[:3], dtype=np.float32))
        traj_anchor.append(np.asarray(anchor_target[:3], dtype=np.float32))
        traj_dragged_line.set_data(np.array(traj_dragged, dtype=np.float32))
        traj_anchor_line.set_data(np.array(traj_anchor, dtype=np.float32))
        _update_anchor_visuals(window, anchor_pos, anchor_target)
        GSU.process_events(window)
        png_path = os.path.join(outdir, f'_frame_{step:04d}.png')
        GSU.capture_canvas_png(window, png_path, fit=False)
        pil_frames.append(Image.open(png_path).convert('RGB'))
        print(f'[static_obstacle_drag_demo] step={step:3d}/{n_steps}  E={E:.4f}  {label}', flush=True)

    capture_frame(0, E_relax, anchor_base, anchor_base, label='start')

    for step in range(1, n_steps + 1):
        # Move anchor target toward static body
        dist = step * args.drag_step
        target_xy = anchor_base[:2] + drag_unit * dist
        target = np.array([target_xy[0], target_xy[1], anchor_base[2]], dtype=np.float32)
        _set_anchors(window, anchor_flat, target)

        # Mid-drag toggle: at toggle_step, switch body 0 static→dynamic, run a few steps, then back
        if step == toggle_step:
            yield ctx.frame(f'Toggling body 0: static → dynamic (both molecules free)…')
            _toggle_body_state(window, 0)  # static → dynamic
            n_dyn, n_stat, n_del = _body_state_counts(window)
            print(f'[static_obstacle_drag_demo] TOGGLE: body 0 now dynamic (dyn={n_dyn} stat={n_stat})', flush=True)
            # Recompute map (no static bodies → PairFF part is zero)
            _recompute_ra_combined_map(window)
            GSU.process_events(window)
            # Run a few relaxation steps with both dynamic
            rbd.run_multimol_md(50, args.dt, fire=True, faf=None)
            _sync_ensemble_from_gpu(window)
            _sync_display(window)
            GSU.process_events(window)
            atoms = np.empty((rbd.total_atoms, 4), dtype=np.float32)
            rbd.fromGPU('apos_world', atoms); rbd.queue.finish()
            E = float(atoms[:, 3].sum())
            anchor_act = atoms[anchor_flat, :3].copy()
            capture_frame(step, E, anchor_act, target, label='BOTH DYNAMIC')

            yield ctx.frame(f'Toggling body 0 back: dynamic → static…')
            _toggle_body_state(window, 0)  # dynamic → static
            n_dyn, n_stat, n_del = _body_state_counts(window)
            print(f'[static_obstacle_drag_demo] TOGGLE: body 0 now static (dyn={n_dyn} stat={n_stat})', flush=True)
            _recompute_ra_combined_map(window)
            GSU.process_events(window)
            continue

        rbd.run_multimol_md(args.n_relax, args.dt, fire=True, faf=None)
        _sync_ensemble_from_gpu(window)
        _sync_display(window)
        GSU.process_events(window)
        atoms = np.empty((rbd.total_atoms, 4), dtype=np.float32)
        rbd.fromGPU('apos_world', atoms); rbd.queue.finish()
        # Use eval_energy_system for proper total energy (.w channel double-counts pairs)
        pos_e, quat_e = window.ra_ensemble.get_poses()
        E = float(rbd.eval_energy_system(pos_e, quat_e, k_pack=0.0))
        anchor_act = atoms[anchor_flat, :3].copy()
        capture_frame(step, E, anchor_act, target)
        if step % 10 == 0 or step == n_steps:
            yield ctx.frame(f'Drag: {step}/{n_steps} steps, E={E:.4f}')

    elapsed = time.perf_counter() - t0
    print(f'[static_obstacle_drag_demo] Drag done in {elapsed:.1f}s ({n_steps} steps)', flush=True)

    # === Phase 7: Save GIF and/or MP4 ===
    frame_paths = [os.path.join(outdir, f'_frame_{i:04d}.png') for i in range(len(pil_frames))]
    gif_path = None
    if args.format in ('gif', 'both'):
        gif_path = os.path.join(outdir, 'static_obstacle_drag_demo.gif')
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           duration=150, loop=0, optimize=True)
        print(f'REVIEW: {gif_path}', flush=True)
    if args.format in ('mp4', 'both'):
        try:
            mp4_path = os.path.join(outdir, 'static_obstacle_drag_demo.mp4')
            GSU.frames_to_video(frame_paths, mp4_path, fps=10)
            print(f'REVIEW: {mp4_path}', flush=True)
        except Exception as e:
            print(f'[static_obstacle_drag_demo] MP4 export skipped: {e}', flush=True)
    pil_frames[0].save(os.path.join(outdir, 'frame_first.png'))
    pil_frames[-1].save(os.path.join(outdir, 'frame_last.png'))
    print(f'REVIEW: {os.path.join(outdir, "frame_first.png")}', flush=True)
    print(f'REVIEW: {os.path.join(outdir, "frame_last.png")}', flush=True)
    for p in frame_paths:
        if os.path.exists(p):
            os.remove(p)

    # Release anchor
    _set_anchors(window, -1, np.zeros(3, dtype=np.float32))
    _update_anchor_visuals(window, None, None)
    traj_dragged_line.visible = False
    traj_anchor_line.visible = False
    rbd.reset_dynamics_state()
    _sync_ensemble_from_gpu(window)
    _sync_display(window)
    GSU.process_events(window)

    yield ctx.frame(f'Demo done: {len(pil_frames)} frames → {gif_path}')
    return {'n_frames': len(pil_frames), 'gif_path': gif_path, 'n_steps': n_steps}


if __name__ == '__main__':
    import sys
    print('Use: ./run_gui.sh --script demos/gui_scripts/static_obstacle_drag_demo.py', file=sys.stderr)
    raise SystemExit(1)
