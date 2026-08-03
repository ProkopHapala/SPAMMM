"""Menu-loaded continuous static-obstacle role hand-off animation macro.

This companion to ``static_obstacle_drag_demo.py`` is selected from Scripts →
Bundled.  It deliberately has no command-line parameter contract: the GUI script
runner supplies the existing window and engine context, and this file describes a
repeatable animation sequence.
"""
import os
import time
import numpy as np
from PIL import Image

from spammm.GUI import gui_script_utils as GSU


def run(window, argv=None, ctx=None):
    _ = argv
    spring, dt, n_relax = 0.2, 0.02, 200
    drag_step, drag_x = 0.15, 8.0
    output_format = 'both'
    from spammm.GUI.RigidAssemblyExtension import (
        _set_anchors, _sync_ensemble_from_gpu, _sync_display,
        _toggle_body_state, _recompute_ra_combined_map, _body_state_counts,
        _update_anchor_visuals,
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dimer_path = os.path.join(repo_root, 'data', 'xyz', 'benzoicacid_dimer.xyz')
    if not os.path.isfile(dimer_path):
        raise FileNotFoundError(dimer_path)
    outdir = os.path.join('debug', 'static_obstacle_drag_demo', 'alternate')
    frames_dir = os.path.join(outdir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    yield ctx.frame('Loading benzoic acid dimer…')
    GSU.set_label_mode(window, 'None')
    GSU.load_molecule(window, dimer_path)
    comps = window.backend.graph.find_connected_components()
    if len(comps) != 2:
        raise RuntimeError(f'Expected exactly 2 connected components, got {len(comps)}')

    yield ctx.frame('Building two rigid bodies from editor fragments…')
    GSU.expand_extension_panel(window, 'rigid_assembly', open=True)
    GSU.set_combo_text(window.ra_source_combo, 'From editor')
    GSU.set_check(window.ra_no_qeq_chk, False)
    GSU.set_check(window.ra_no_faf_chk, False)
    GSU.set_spin_value(window.ra_k_spring_spin, spring)
    GSU.set_spin_value(window.ra_drag_dt_spin, dt)
    GSU.set_spin_value(window.ra_drag_nrelax_spin, n_relax)
    GSU.click_button(window.ra_build_btn)
    if window.ra_ensemble is None or window.ra_rbd is None:
        raise RuntimeError('RA build from editor failed')
    rbd, ens = window.ra_rbd, window.ra_ensemble
    if len(ens) != 2:
        raise RuntimeError(f'Expected 2 rigid bodies, got {len(ens)}')
    if hasattr(window.scene, 'fit_to_atoms'):
        window.scene.fit_to_atoms(margin=3.0)
        GSU.process_events(window)

    def select_leading_anchor(body, direction):
        pack = rbd._mb_packs[body]
        real = np.asarray(pack['types']) == 0
        candidates = [i for i, (e, t) in enumerate(zip(pack['enames'], pack['types'])) if real[i] and e == 'O']
        if not candidates:
            candidates = list(np.flatnonzero(real))
        if not candidates:
            raise RuntimeError(f'body {body} has no real anchor site')
        rel = np.asarray(pack['rel'])[:, :3]
        local = max(candidates, key=lambda i: float(np.dot(rel[i, :2], direction)))
        return int(local), int(rbd.mol_offsets[body] + local), str(pack['enames'][local])

    def world_anchor(flat):
        atoms = np.empty((rbd.total_atoms, 4), dtype=np.float32)
        rbd.fromGPU('apos_world', atoms)
        rbd.queue.finish()
        return atoms[flat, :3].copy()

    def pose_energy():
        pos, quat = ens.get_poses()
        return float(rbd.eval_energy_system(pos, quat, k_pack=0.0))

    def save_frame(pil_frames, step, label):
        GSU.process_events(window)
        png = os.path.join(frames_dir, f'_frame_{step:04d}.png')
        GSU.capture_canvas_png(window, png, fit=False)
        pil_frames.append(Image.open(png).convert('RGB'))
        print(f'[static_obstacle_drag_demo_alt] step={step:3d} E={pose_energy():.4f} {label}', flush=True)

    # First pass: body 0 static, body 1 dynamic, moving toward body 0.
    _toggle_body_state(window, 0)
    _on_probe = getattr(window, 'ra_map_layer_combo', None)
    if _on_probe is not None:
        _on_probe.setCurrentText('Total')
    _recompute_ra_combined_map(window)
    pos, _ = ens.get_poses()
    direction = pos[0, :2] - pos[1, :2]
    direction /= max(np.linalg.norm(direction), 1e-12)
    local, anchor_flat, ename = select_leading_anchor(1, direction)
    _set_anchors(window, -1, np.zeros(3, dtype=np.float32))
    rbd.run_multimol_md(300, dt=dt, lin_damp=0.95, ang_damp=0.95, faf=None)
    _sync_ensemble_from_gpu(window); _sync_display(window)
    anchor_base = world_anchor(anchor_flat)
    _set_anchors(window, anchor_flat, anchor_base)
    print(f'[static_obstacle_drag_demo_alt] first anchor body=1 local={local} element={ename} flat={anchor_flat}', flush=True)

    n_steps = int(np.ceil(drag_x / drag_step))
    pil_frames = []
    t0 = time.perf_counter()
    save_frame(pil_frames, 0, 'first pass start')
    for step in range(1, n_steps + 1):
        target_xy = anchor_base[:2] + direction * (step * drag_step)
        target = np.array([target_xy[0], target_xy[1], anchor_base[2]], dtype=np.float32)
        _set_anchors(window, anchor_flat, target)
        rbd.run_multimol_md(n_relax, dt, fire=True, faf=None)
        _sync_ensemble_from_gpu(window); _sync_display(window)
        _update_anchor_visuals(window, world_anchor(anchor_flat), target)
        save_frame(pil_frames, step, 'first pass')
        if step % 10 == 0 or step == n_steps:
            yield ctx.frame(f'First pass: {step}/{n_steps}')

    # Continuous role hand-off at closest approach: body 1 becomes static and
    # body 0 becomes dynamic without resetting either pose.
    _set_anchors(window, -1, np.zeros(3, dtype=np.float32))
    _toggle_body_state(window, 0)
    _toggle_body_state(window, 1)
    states = rbd._body_state_host.copy()
    if not np.array_equal(states, np.array([1, 0], dtype=states.dtype)):
        raise RuntimeError(f'role hand-off produced unexpected body states {states.tolist()}')
    _recompute_ra_combined_map(window)
    pos, _ = ens.get_poses()
    direction = pos[0, :2] - pos[1, :2]
    direction /= max(np.linalg.norm(direction), 1e-12)
    local, anchor_flat, ename = select_leading_anchor(0, direction)
    anchor_base = world_anchor(anchor_flat)
    _set_anchors(window, anchor_flat, anchor_base)
    print(f'[static_obstacle_drag_demo_alt] hand-off states={states.tolist()} anchor body=0 local={local} element={ename}', flush=True)
    yield ctx.frame('Role hand-off: body 1 static, body 0 pulling away')

    for step in range(1, n_steps + 1):
        target_xy = anchor_base[:2] + direction * (step * drag_step)
        target = np.array([target_xy[0], target_xy[1], anchor_base[2]], dtype=np.float32)
        _set_anchors(window, anchor_flat, target)
        rbd.run_multimol_md(n_relax, dt, fire=True, faf=None)
        _sync_ensemble_from_gpu(window); _sync_display(window)
        _update_anchor_visuals(window, world_anchor(anchor_flat), target)
        save_frame(pil_frames, n_steps + step, 'second pass away')
        if step % 10 == 0 or step == n_steps:
            yield ctx.frame(f'Second pass: {step}/{n_steps}')

    frame_paths = [os.path.join(frames_dir, f'_frame_{i:04d}.png') for i in range(len(pil_frames))]
    gif_path = None
    if output_format in ('gif', 'both'):
        gif_path = os.path.join(outdir, 'static_obstacle_drag_demo_alt.gif')
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=150, loop=0, optimize=True)
        print(f'REVIEW: {gif_path}', flush=True)
    if output_format in ('mp4', 'both'):
        try:
            mp4_path = os.path.join(outdir, 'static_obstacle_drag_demo_alt.mp4')
            GSU.frames_to_video(frame_paths, mp4_path, fps=10)
            print(f'REVIEW: {mp4_path}', flush=True)
        except Exception as exc:
            print(f'[static_obstacle_drag_demo_alt] MP4 export skipped: {exc}', flush=True)
    pil_frames[0].save(os.path.join(outdir, 'frame_first.png'))
    pil_frames[-1].save(os.path.join(outdir, 'frame_last.png'))
    print(f'REVIEW: {os.path.join(outdir, "frame_first.png")}', flush=True)
    print(f'REVIEW: {os.path.join(outdir, "frame_last.png")}', flush=True)
    _set_anchors(window, -1, np.zeros(3, dtype=np.float32))
    _update_anchor_visuals(window, None, None)
    rbd.reset_dynamics_state()
    _sync_ensemble_from_gpu(window); _sync_display(window)
    if hasattr(window, 'set_manipulation_context'):
        window.set_manipulation_context('rigid_assembly')
    GSU.set_edit_mode(window, 'Manipulate')
    print(f'[static_obstacle_drag_demo_alt] done in {time.perf_counter() - t0:.1f}s states={states.tolist()}', flush=True)
    yield ctx.frame(f'Alternate demo done: {len(pil_frames)} frames → {gif_path}')
    return {'n_frames': len(pil_frames), 'gif_path': gif_path, 'n_steps': n_steps}
