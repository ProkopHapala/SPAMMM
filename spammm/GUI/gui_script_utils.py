"""gui_script_utils.py — drive SPAMMM_GUI like a user; capture demo PNG/GIF/MP4 frames.

Motivation: control scripts must click the same widgets / set the same toggles a
human would, so scripted setups stay valid when the UI layout changes. Capture
helpers exist so draw demos can prove real VisPy chrome (hover foreshadow, δ/φ
handles) in presentation GIFs — not only headless SVG.

Design notes:
- Widget helpers (``set_edit_mode``, ``set_atom_combo``, …) always ``process_events``.
- **apply_demo_overlays** mirrors ``EditModeHandlers`` mouse-move visuals; never
  ``set_data`` empty on ``ring_preview_line`` (VisPy Line + offscreen render can segfault).
- **capture_window_png** composites ``canvas.render()`` into the Qt grab — grab alone
  often blanks OpenGL (offscreen / Wayland).
- **frames_to_gif** — Pillow pack of ordered PNGs (256-color, large files).
- **frames_to_video** — ffmpeg H.264 with ``-tune animation`` (best for mostly-static
  screen content; ~11x smaller than GIF). Pads to even dimensions (H.264 requirement).
  Also supports VP9 (WebM) and AV1 via ``codec=`` parameter.

Doc: ``doc/Topics/GUI_DrawDemo_Scripts.md``, ``doc/Reports/GUI_Scripts_Consolidation_2026-08-01.md``.
"""
import os

from PyQt5 import QtWidgets, QtCore


def process_events(window=None):
    """Pump Qt event loop so layout/repaint catches up."""
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()


def expand_extension_panel(window, name_or_title, open=True):
    """Expand/collapse an extension CollapsibleSection by registry key or display title."""
    sections = getattr(window, '_extension_sections', {})
    sec = sections.get(name_or_title)
    if sec is None:
        raise KeyError(f"Extension panel {name_or_title!r} not found; keys={list(sections.keys())}")
    sec._toggle.setChecked(open)
    process_events(window)
    return sec


def click_button(btn):
    """Invoke a QPushButton's connected slot (same as user click)."""
    btn.click()
    process_events()


def set_combo_text(combo, text):
    combo.setCurrentText(text)
    process_events()


def set_line_edit(line_edit, text):
    line_edit.setText(text)
    process_events()


def set_spin_value(spin, value):
    spin.setValue(value)
    process_events()


def set_slider_value(slider, value):
    slider.blockSignals(True)
    slider.setValue(int(value))
    slider.blockSignals(False)
    process_events()


def set_check(check, checked):
    """Set a QCheckBox state."""
    check.setChecked(bool(checked))
    process_events()


def load_molecule(window, path):
    """Load an XYZ file into the GUI backend and refresh the view."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    window.backend.load_xyz(path)
    window.refresh_view()
    process_events(window)


def set_edit_mode(window, mode):
    """Switch the main editor interaction mode."""
    window.set_edit_mode(mode)
    process_events(window)


def extension_panel(window, key, open=True):
    """Backward-compatible alias for expand_extension_panel."""
    return expand_extension_panel(window, key, open=open)


def set_auto_h_cap(window, on):
    """Mirror the Auto H toggle button."""
    on = bool(on)
    if window.auto_h_cap_btn.isChecked() != on:
        window.auto_h_cap_btn.setChecked(on)
        window.toggle_auto_h_cap()
    else:
        window.backend.auto_h_cap = on
    process_events(window)


def set_ring_size(window, n):
    """Set Ring-mode n-gon size spinbox (same as user / numpad ±)."""
    set_spin_value(window.ring_size_spinbox, int(n))


def set_atom_combo(window, element):
    """Set element combo (Atom-mode click applies this type)."""
    set_combo_text(window.atom_combo, str(element))
    window.cur_atom_type = str(element)
    process_events(window)


def apply_demo_overlays(window, *, cursor_xy=None, hover_hex=None, bond_highlight=None,
                        ring_preview=None, hover_atom_id=None, clear=True):
    """Drive VisPy hover/cursor chrome the same way EditModeHandlers do on mouse move.

    Used by GUI demo scripts so GIF frames show foreshadow rings, bond hover, etc.

    NOTE: never ``set_data`` empty arrays on ``ring_preview_line`` — VisPy Line can
    segfault on the next ``canvas.render()`` offscreen after empty→refill.
    """
    import numpy as np
    from spammm.topology.HexGrid import snap_to_grid

    if clear:
        # Hide only — do not empty ring_preview_line buffers
        if hasattr(window, 'hover_markers'):
            window.hover_markers.visible = False
        window.scene.ring_preview_line.visible = False
        window.scene.hover_bond_line.visible = False
        try:
            window.scene.hover_atom_marker.visible = False
        except Exception:
            pass
        if hasattr(window, 'cursor_markers') and cursor_xy is None:
            window.cursor_markers.visible = False

    # Mouse cursor (same Markers as on_mouse_move)
    if cursor_xy is not None and hasattr(window, 'cursor_markers'):
        p = np.array([[float(cursor_xy[0]), float(cursor_xy[1]), 0.0]], dtype=np.float32)
        window.cursor_markers.set_data(pos=p, symbol='cross', edge_width=2,
                                       edge_color='red', face_color='transparent', size=14)
        window.cursor_markers.visible = True

    # Hex tile hover — orange node discs (RingMode)
    if hover_hex is not None:
        q, r = hover_hex
        ring_nodes = window.backend.grid.ring_nodes(q, r)
        hover_pos = [[snap_to_grid(n)[0], snap_to_grid(n)[1], -0.08] for n in ring_nodes]
        if hover_pos:
            window.hover_markers.set_data(pos=np.array(hover_pos, dtype=np.float32), symbol='disc',
                                         edge_width=2, edge_color='orange', face_color='transparent', size=12)
            window.hover_markers.visible = True
        nodes = np.asarray(ring_nodes, dtype=np.float32)
        if nodes.ndim == 2 and nodes.shape[1] == 2:
            nodes = np.column_stack([nodes, np.zeros(len(nodes), dtype=np.float32)])
        closed = np.vstack([nodes, nodes[:1]]).astype(np.float32)
        window.scene.ring_preview_line.set_data(pos=closed, color=(0.2, 0.8, 0.8, 0.6))
        window.scene.ring_preview_line.visible = True

    # Bond hover (lime line) — set_data only when we have endpoints
    if bond_highlight is not None:
        ids = list(bond_highlight)
        a = window.backend.graph.atoms.get(ids[0])
        b = window.backend.graph.atoms.get(ids[1])
        if a is not None and b is not None and a.alive and b.alive:
            window.scene.hover_bond_line.set_data(pos=np.array([a.pos, b.pos], dtype=np.float32))
            window.scene.hover_bond_line.visible = True

    # Cyan adjacent-ring foreshadow (overrides hex outline if both set)
    if ring_preview is not None:
        rp = np.asarray(ring_preview, dtype=np.float64)
        if rp.ndim != 2:
            raise ValueError(f'ring_preview shape {rp.shape}')
        if rp.shape[1] == 2:
            rp3 = np.column_stack([rp, np.zeros(len(rp))])
        else:
            rp3 = rp[:, :3]
        closed = np.vstack([rp3, rp3[:1]]).astype(np.float32)
        window.scene.ring_preview_line.set_data(pos=closed, color=(0.2, 0.8, 0.8, 0.6))
        window.scene.ring_preview_line.visible = True

    # Atom hover marker (same args as EditModeHandlers._hover_atom)
    if hover_atom_id is not None:
        atom = window.backend.graph.atoms.get(hover_atom_id)
        if atom is not None and atom.alive:
            window.scene.hover_atom_marker.set_data(
                pos=np.array([atom.pos], dtype=np.float32),
                symbol='disc', edge_width=3, edge_color='yellow', face_color='transparent', size=20)
            window.scene.hover_atom_marker.visible = True

    process_events(window)


def capture_canvas_png(window, path, size=None, fit=True, zoom_out=2.0):
    """Rasterize the VisPy editor canvas to PNG (viewport only).

    zoom_out: extra factor on fit_to_atoms margin (2 ≈ show selection box comfortably).
    """
    from vispy.io import write_png
    process_events(window)
    if fit and hasattr(window.scene, 'fit_to_atoms'):
        window.scene.fit_to_atoms(margin=1.8 * float(zoom_out))
    window.scene.canvas.update()
    process_events(window)
    kwargs = {}
    if size is not None:
        kwargs['size'] = tuple(size)
    img = window.scene.canvas.render(alpha=False, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    write_png(path, img)
    return path


def capture_window_png(window, path, include_frame=True, fit=True, zoom_out=2.0):
    """Screenshot of the Qt window including side panels / status bar.

    Composites VisPy ``canvas.render()`` into the Qt grab at the native canvas
    widget rect — ``QWidget.grab()`` / ``grabWindow`` often miss OpenGL content
    (offscreen / Wayland). Titlebar included via ``grabWindow(winId)`` when possible.
    """
    from PyQt5 import QtGui
    import numpy as np
    from PIL import Image

    process_events(window)
    if fit and hasattr(window, 'scene') and hasattr(window.scene, 'fit_to_atoms'):
        window.scene.fit_to_atoms(margin=1.8 * float(zoom_out))
        window.scene.canvas.update()
        process_events(window)

    used_frame = False
    pix = None
    if include_frame:
        try:
            screen = window.screen() or QtWidgets.QApplication.primaryScreen()
            if screen is not None and window.winId():
                pix = screen.grabWindow(int(window.winId()))
                used_frame = pix is not None and not pix.isNull()
        except Exception:
            pix = None
            used_frame = False
    if pix is None or pix.isNull():
        pix = window.grab()
        used_frame = False

    try:
        canvas = window.scene.canvas
        native = getattr(canvas, 'native', None)
        gl = canvas.render(alpha=False)
        if native is not None and gl is not None and getattr(gl, 'size', 0):
            top_left = native.mapTo(window, native.rect().topLeft())
            ox, oy = int(top_left.x()), int(top_left.y())
            if used_frame:
                try:
                    geo = window.frameGeometry()
                    cgeo = window.geometry()
                    ox += int(cgeo.x() - geo.x())
                    oy += int(cgeo.y() - geo.y())
                except Exception:
                    pass
            gl_img = np.asarray(gl)
            if gl_img.ndim == 3 and gl_img.shape[2] >= 3:
                if gl_img.dtype != np.uint8:
                    gl_img = (np.clip(gl_img, 0, 1) * 255).astype(np.uint8) if float(gl_img.max()) <= 1.0 else gl_img.astype(np.uint8)
                h, w = gl_img.shape[:2]
                nw, nh = max(1, int(native.width())), max(1, int(native.height()))
                pil_gl = Image.fromarray(gl_img[:, :, :3])
                if (w, h) != (nw, nh):
                    pil_gl = pil_gl.resize((nw, nh), Image.BILINEAR)
                    w, h = nw, nh
                qimg = pix.toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
                ptr = qimg.bits(); ptr.setsize(qimg.byteCount())
                arr = np.frombuffer(ptr, np.uint8).reshape(qimg.height(), qimg.width(), 4).copy()
                base = Image.fromarray(arr[:, :, :3])
                if ox < base.width and oy < base.height and ox + w > 0 and oy + h > 0:
                    base.paste(pil_gl, (ox, oy))
                    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
                    base.save(path)
                    return path
    except Exception as e:
        print(f'[capture_window_png] VisPy composite skipped: {e}')

    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    pix.save(path)
    return path


def frames_to_gif(frame_paths, out_gif, duration_ms=700):
    """Pack ordered PNG frames into an animated GIF (Pillow)."""
    from PIL import Image
    imgs = [Image.open(p).convert('RGB') for p in frame_paths]
    if not imgs:
        raise ValueError('frames_to_gif: empty frame list')
    os.makedirs(os.path.dirname(os.path.abspath(out_gif)) or '.', exist_ok=True)
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:], duration=int(duration_ms), loop=0)
    for im in imgs:
        im.close()
    return out_gif


def frames_to_video(frame_paths, out_video, fps=10, codec='libx264', crf=23, extra_args=None):
    """Encode ordered PNG frames into a video (MP4/WebM) via ffmpeg.

    Defaults: H.264 with ``-tune animation`` (best for mostly-static screen content
    with small moving parts — flat colors compress well). CRF 23 = x264 default
    (good quality, reasonable size). ``-pix_fmt yuv420p`` for universal playback.

    Args:
        frame_paths: list of PNG file paths (ordered, zero-padded names expected).
        out_video: output path (``.mp4`` → H.264, ``.webm`` → VP9).
        fps: frames per second.
        codec: ffmpeg video codec (``libx264``, ``libvpx-vp9``, ``libaom-av1``).
        crf: quality (lower=better; 18=visually lossless, 23=default, 28=small).
        extra_args: list of additional ffmpeg args appended before output.

    Returns the output path. Raises if ffmpeg fails or is not found.
    """
    import subprocess
    if not frame_paths:
        raise ValueError('frames_to_video: empty frame list')
    os.makedirs(os.path.dirname(os.path.abspath(out_video)) or '.', exist_ok=True)
    # Use image2 demuxer with sequential frame pattern (frames must be zero-padded)
    # Detect pattern from first frame path
    first = frame_paths[0]
    n = len(frame_paths)
    # Build a printf-style pattern: replace the zero-padded number with %0Nd
    import re
    m = re.search(r'^(.*?)(\d+)(\.\w+)$', first)
    if m and len(m.group(2)) >= 2:
        prefix = m.group(1)
        pad = len(m.group(2))
        ext = m.group(3)
        pattern = f'{prefix}%0{pad}d{ext}'
    else:
        # Fallback: write a concat list (no duration — rely on fps)
        import tempfile
        fd, list_path = tempfile.mkstemp(suffix='.txt', dir=os.path.dirname(os.path.abspath(out_video)) or '.')
        with os.fdopen(fd, 'w') as f:
            for p in frame_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")
        cmd_input = ['-f', 'concat', '-safe', '0', '-i', list_path]
        pattern = None
    if pattern is not None:
        cmd_input = ['-framerate', str(fps), '-i', pattern]
    cmd = ['ffmpeg', '-y'] + cmd_input
    # Pad to even dimensions (H.264/VP9 require even width/height)
    cmd += ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']
    if codec == 'libx264':
        cmd += ['-c:v', 'libx264', '-tune', 'animation', '-crf', str(crf),
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    elif codec == 'libvpx-vp9':
        cmd += ['-c:v', 'libvpx-vp9', '-crf', str(crf), '-b:v', '0', '-pix_fmt', 'yuv420p']
    else:
        cmd += ['-c:v', codec, '-crf', str(crf)]
    if extra_args:
        cmd += list(extra_args)
    cmd.append(out_video)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError('ffmpeg not found — install it to use frames_to_video')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'ffmpeg failed (code {e.returncode}):\n{e.stderr}')
    finally:
        if pattern is None:
            os.unlink(list_path)
    return out_video
