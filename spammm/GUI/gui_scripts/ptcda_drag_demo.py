#!/usr/bin/env python3
"""Automatic PTCDA drag demo on NaCl: stick-slip GIF via the GUI drag handler.

Builds 2×PTCDA on NaCl via the GUI (same as ptcda_interactive_drag.py), enters
ra_drag mode, then programmatically drags one molecule's end-O through the
other using the SAME code path as interactive dragging:
  - _set_anchors (anchor spring on O atom)
  - run_multimol_md(fire=True, n_relax=N)  — FIRE relaxation per drag step
  - _sync_ensemble_from_gpu + _sync_display — updates the VisPy canvas

Frames are captured from the VisPy canvas (GSU.capture_canvas_png), which now
shows the NaCl substrate overlay (shared VispyUtils.update_substrate_overlay).

Run:
  ./run_gui.sh --script spammm/GUI/gui_scripts/ptcda_drag_demo.py

  # Bigger drag, more relaxation:
  ./run_gui.sh --script spammm/GUI/gui_scripts/ptcda_drag_demo.py --drag-x 20 --n-relax 300
"""
import argparse
import os
import time
import numpy as np
from PIL import Image

from spammm.GUI import gui_script_utils as GSU


def run(window, argv=None, ctx=None):
    p = argparse.ArgumentParser(description='Automatic PTCDA drag stick-slip GIF via GUI drag handler')
    p.add_argument('--nmol', type=int, default=2, help='molecules (1 dragged + rest obstacles)')
    p.add_argument('--spacing', type=float, default=16.0)
    p.add_argument('--spring', type=float, default=0.2, help='anchor spring [PairFF internal units]')
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--n-relax', type=int, default=200, help='FIRE relaxation steps per drag step')
    p.add_argument('--drag-step', type=float, default=0.2, help='anchor movement per drag step [Å]')
    p.add_argument('--drag-x', type=float, default=16.0, help='total drag distance [Å]')
    p.add_argument('--anchor-atom', type=int, default=29, help='local index of anchored O atom (29=top-right corner carbonyl O)')
    p.add_argument('--opposite-atom', type=int, default=27, help='local index of opposite corner O atom (27=bottom-left corner carbonyl O)')
    p.add_argument('--out', type=str, default=None, help='output dir (default: debug/ptcda_drag_demo)')
    args = p.parse_args(argv or [])

    from spammm.GUI import RigidAssemblyExtension as RA
    from spammm.GUI.RigidAssemblyExtension import _set_anchors, _sync_ensemble_from_gpu, _sync_display, _update_ra_substrate_overlay, _update_anchor_visuals

    outdir = args.out or os.path.join('debug', 'ptcda_drag_demo')
    os.makedirs(outdir, exist_ok=True)

    # === Phase 1: Build via GUI (same as ptcda_interactive_drag) ===
    yield ctx.frame(f'Configuring {args.nmol}×PTCDA on NaCl…')
    GSU.expand_extension_panel(window, 'rigid_assembly', open=True)
    GSU.set_combo_text(window.ra_source_combo, 'From file')
    GSU.set_combo_text(window.ra_mol_combo, 'PTCDA')
    GSU.set_spin_value(window.ra_nmol_spin, args.nmol)
    GSU.set_spin_value(window.ra_spacing_spin, args.spacing)
    GSU.set_check(window.ra_no_qeq_chk, False)
    GSU.set_check(window.ra_no_faf_chk, False)
    GSU.set_spin_value(window.ra_k_spring_spin, args.spring)
    GSU.set_spin_value(window.ra_drag_dt_spin, args.dt)
    GSU.set_spin_value(window.ra_drag_nrelax_spin, args.n_relax)
    GSU.click_button(window.ra_build_btn)
    if window.ra_ensemble is None or window.ra_rbd is None:
        raise RuntimeError('PTCDA rigid assembly build failed')
    rbd = window.ra_rbd
    fit = window.ra_fit
    if fit is None:
        raise RuntimeError('FAF fit is None — substrate not enabled')
    if not rbd.faf_mode:
        raise RuntimeError('NaCl FAF is not enabled')
    qmax = max(float(np.abs(pack['REQ_base'][:, 2]).max()) for pack in rbd._mb_packs)
    if qmax < 1e-3:
        raise RuntimeError('PTCDA QEq charges are zero; O→Na stick-slip cannot work')
    print(f'[ptcda_drag_demo] Built {args.nmol}×PTCDA, FAF on, max|Q|={qmax:.3f} e', flush=True)

    # Substrate overlay is now wired into RA build (shared VispyUtils)
    # Make sure it's visible
    window.ra_show_substrate = True
    _update_ra_substrate_overlay(window)
    GSU.process_events(window)

    # Fit viewport to see molecules + substrate
    if hasattr(window.scene, 'fit_to_atoms'):
        window.scene.fit_to_atoms(margin=3.0)
        GSU.process_events(window)
    yield ctx.frame(f'Built {args.nmol}×PTCDA on NaCl (FAF on, |Q|max={qmax:.3f}, substrate visible)')

    # === Phase 2: Setup anchor on end-O of mol 0 ===
    n_per_mol = len(rbd._mb_packs[0]['types'])
    anchor_ia = args.anchor_atom  # local index within mol 0
    anchor_flat = 0 * n_per_mol + anchor_ia  # flat index in mol 0
    print(f'[ptcda_drag_demo] Anchor: mol=0, O[{anchor_ia}], flat={anchor_flat}', flush=True)

    # Get initial world position of anchored atom
    atoms0 = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms0); rbd.queue.finish()
    anchor_base = atoms0[anchor_flat, :3].copy()
    print(f'[ptcda_drag_demo] Anchor base position: {anchor_base}', flush=True)

    # === Phase 3: Relaxation (settle on surface, no anchor) ===
    yield ctx.frame(f'Relaxing on surface…')
    _set_anchors(window, -1, np.zeros(3, dtype=np.float32))  # no anchor
    rbd.run_multimol_md(300, dt=args.dt, lin_damp=0.95, ang_damp=0.95, faf=True)
    _sync_ensemble_from_gpu(window)
    _sync_display(window)
    GSU.process_events(window)
    # Read relaxed anchor position
    atoms_relax = np.empty((rbd.total_atoms, 4), dtype=np.float32)
    rbd.fromGPU('apos_world', atoms_relax); rbd.queue.finish()
    anchor_base = atoms_relax[anchor_flat, :3].copy()
    E_relax = float(atoms_relax[:, 3].sum())
    print(f'[ptcda_drag_demo] Relaxed: E={E_relax:.4f}, anchor_O={anchor_base}', flush=True)

    # Set anchor at relaxed position (same as drag handler on_press)
    _set_anchors(window, anchor_flat, anchor_base)
    rbd.reset_dynamics_state()

    # === Trajectory lines: dragged atom (red), anchor target (blue), opposite corner (green) ===
    import vispy.scene as vscene
    opposite_flat = 0 * n_per_mol + args.opposite_atom
    traj_dragged = [anchor_base.copy()]      # dragged atom world positions
    traj_anchor  = [anchor_base.copy()]      # anchor target positions
    traj_opposite = [atoms_relax[opposite_flat, :3].copy()]  # opposite corner positions
    traj_dragged_line = vscene.visuals.Line(parent=window.scene.view.scene, color=(1, 0, 0, 0.9),
                                            width=2.0, antialias=True, method='gl', connect='strip')
    traj_anchor_line  = vscene.visuals.Line(parent=window.scene.view.scene, color=(0.2, 0.5, 1, 0.9),
                                            width=2.0, antialias=True, method='gl', connect='strip')
    traj_opposite_line = vscene.visuals.Line(parent=window.scene.view.scene, color=(0.0, 0.8, 0.2, 0.9),
                                             width=2.0, antialias=True, method='gl', connect='strip')
    for v in (traj_dragged_line, traj_anchor_line, traj_opposite_line):
        v.set_gl_state('translucent', depth_test=False)
        v.order = 9
    print(f'[ptcda_drag_demo] Trajectory: dragged O[{anchor_ia}] (red), anchor target (blue), opposite O[{args.opposite_atom}] (green)', flush=True)

    # === Phase 4: Drag loop — bigger steps, full FIRE relaxation per step ===
    n_steps = int(np.ceil(args.drag_x / args.drag_step))
    print(f'[ptcda_drag_demo] Drag: {n_steps} steps × {args.drag_step:.2f} Å = {args.drag_x:.1f} Å total, {args.n_relax} FIRE steps/step', flush=True)
    yield ctx.frame(f'Dragging O atom {args.drag_x:.1f} Å ({n_steps} steps × {args.n_relax} FIRE)…')

    pil_frames = []
    t0 = time.perf_counter()

    def capture_frame(step, E, target_x, anchor_pos, anchor_target, opposite_pos):
        """Capture VisPy canvas → PIL frame. Camera fixed; anchor line + 3 growing trajectories."""
        # Append trajectory points
        traj_dragged.append(np.asarray(anchor_pos[:3], dtype=np.float32))
        traj_anchor.append(np.asarray(anchor_target[:3], dtype=np.float32))
        traj_opposite.append(np.asarray(opposite_pos[:3], dtype=np.float32))
        # Update growing polylines
        traj_dragged_line.set_data(np.array(traj_dragged, dtype=np.float32))
        traj_anchor_line.set_data(np.array(traj_anchor, dtype=np.float32))
        traj_opposite_line.set_data(np.array(traj_opposite, dtype=np.float32))
        # Update anchor visuals (red line atom→target + red cross at target)
        _update_anchor_visuals(window, anchor_pos, anchor_target)
        GSU.process_events(window)
        png_path = os.path.join(outdir, f'_frame_{step:04d}.png')
        GSU.capture_canvas_png(window, png_path, fit=False)  # fit=False: don't move camera
        pil_frames.append(Image.open(png_path).convert('RGB'))
        print(f'[ptcda_drag_demo] step={step:3d}/{n_steps}  E={E:.4f}  target_x={target_x:.2f}', flush=True)

    # Capture initial frame (anchor at relaxed position)
    opposite_init = atoms_relax[opposite_flat, :3].copy()
    capture_frame(0, E_relax, float(anchor_base[0]), anchor_base, anchor_base, opposite_init)

    for step in range(1, n_steps + 1):
        # Move anchor target by drag_step along x (same as drag handler on_move)
        target_x = float(anchor_base[0]) + step * args.drag_step
        target = np.array([target_x, anchor_base[1], anchor_base[2]], dtype=np.float32)
        _set_anchors(window, anchor_flat, target)
        # FIRE relaxation (same as drag handler on_move: run_multimol_md with fire=True)
        rbd.run_multimol_md(args.n_relax, args.dt, fire=True, faf=None)
        _sync_ensemble_from_gpu(window)
        _sync_display(window)
        GSU.process_events(window)
        # Read energy + actual positions (for anchor line + trajectories)
        atoms = np.empty((rbd.total_atoms, 4), dtype=np.float32)
        rbd.fromGPU('apos_world', atoms); rbd.queue.finish()
        E = float(atoms[:, 3].sum())
        anchor_act = atoms[anchor_flat, :3].copy()
        opposite_act = atoms[opposite_flat, :3].copy()
        capture_frame(step, E, target_x, anchor_act, target, opposite_act)
        if step % 10 == 0 or step == n_steps:
            yield ctx.frame(f'Drag: {step}/{n_steps} steps, E={E:.4f}')

    elapsed = time.perf_counter() - t0
    print(f'[ptcda_drag_demo] Drag done in {elapsed:.1f}s ({n_steps} steps)', flush=True)

    # === Phase 5: Save GIF ===
    gif_path = os.path.join(outdir, 'ptcda_drag_demo.gif')
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                       duration=150, loop=0, optimize=True)
    print(f'REVIEW: {gif_path}', flush=True)
    # Save first and last frame
    pil_frames[0].save(os.path.join(outdir, 'frame_first.png'))
    pil_frames[-1].save(os.path.join(outdir, 'frame_last.png'))
    print(f'REVIEW: {os.path.join(outdir, "frame_first.png")}', flush=True)
    print(f'REVIEW: {os.path.join(outdir, "frame_last.png")}', flush=True)
    # Clean up temp frames
    for i in range(len(pil_frames)):
        p = os.path.join(outdir, f'_frame_{i:04d}.png')
        if os.path.exists(p):
            os.remove(p)

    # Release anchor
    _set_anchors(window, -1, np.zeros(3, dtype=np.float32))
    _update_anchor_visuals(window, None, None)  # hide anchor line + marker
    traj_dragged_line.visible = False
    traj_anchor_line.visible = False
    traj_opposite_line.visible = False
    rbd.reset_dynamics_state()

    # Sync GUI display (don't refit camera — keep it where it was during drag)
    _sync_ensemble_from_gpu(window)
    _sync_display(window)
    GSU.process_events(window)

    yield ctx.frame(f'Drag demo done: {len(pil_frames)} frames → {gif_path}')
    return {'n_frames': len(pil_frames), 'gif_path': gif_path, 'spring': args.spring, 'n_steps': n_steps}


if __name__ == '__main__':
    import sys
    print('Use: ./run_gui.sh --script spammm/GUI/gui_scripts/ptcda_drag_demo.py', file=sys.stderr)
    raise SystemExit(1)
