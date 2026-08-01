#!/usr/bin/env python3
"""GUI azaindol draw demo — same sequence as offline SVG, full-window PNG → GIF.

Uses the same editor tools as a user: Ring/Atom/Select modes, Auto H, element combo,
backend ring ops, Ctrl-C/V, translate_selected / rotate_selected.

  ./run_gui.sh --script demos/gui_scripts/azaindol_draw_demo.py
  ./run_gui.sh --script demos/gui_scripts/azaindol_draw_demo.py -- --gif-ms 500
  ./run_gui.sh --script demos/gui_scripts/azaindol_draw_demo.py -- --canvas-only
"""
import argparse
import os
import sys

from spammm.GUI.azaindol_draw_sequence import SequenceHost, run_azaindol_draw
from spammm.GUI import gui_script_utils as GSU


class GuiHost(SequenceHost):
    def __init__(self, window, out_dir, full_window=True, zoom_out=2.0):
        self.w = window
        self.out_dir = out_dir
        self.full_window = full_window
        self.zoom_out = zoom_out
        self._last_paste_ids = []

    def backend(self):
        return self.w.backend

    def snapshot(self, name, title='', **kw):
        GSU.process_events(self.w)
        if title:
            self.w.statusBar().showMessage(title)
        # Apply VisPy hover/cursor chrome (same visuals as real mouse move)
        GSU.apply_demo_overlays(
            self.w,
            cursor_xy=kw.get('cursor_xy'),
            hover_hex=kw.get('hover_hex'),
            bond_highlight=kw.get('bond_highlight'),
            ring_preview=kw.get('ring_preview'),
            hover_atom_id=kw.get('hover_atom_id'),
        )
        xform = kw.get('xform_mode')
        if xform in ('move', 'rotate') and self.w.scene.get_selected_ids():
            self.w.scene.enter_xform(xform, mouse_pos=None)
        GSU.process_events(self.w)
        path = os.path.join(self.out_dir, f'{name}.png')
        if self.full_window:
            return GSU.capture_window_png(self.w, path, include_frame=True, fit=True, zoom_out=self.zoom_out)
        return GSU.capture_canvas_png(self.w, path, fit=True, zoom_out=self.zoom_out)

    def set_auto_h(self, on):
        GSU.set_auto_h_cap(self.w, on)

    def set_edit_mode(self, mode):
        GSU.set_edit_mode(self.w, mode)
        if hasattr(self.w, 'mode_combo'):
            idx = self.w.mode_combo.findText(mode)
            if idx >= 0:
                self.w.mode_combo.blockSignals(True)
                self.w.mode_combo.setCurrentIndex(idx)
                self.w.mode_combo.blockSignals(False)
        GSU.process_events(self.w)

    def set_ring_size(self, n):
        GSU.set_ring_size(self.w, n)

    def set_element(self, element):
        GSU.set_atom_combo(self.w, element)

    def add_hex(self, q, r):
        self.w._push_undo()
        self.w.backend.add_ring(q, r)

    def add_adj_ring(self, bond, n_members, side):
        self.w._push_undo()
        return self.w.backend.add_adjacent_ring(bond, n_members=n_members, ename=self.w.cur_atom_type, side=side)

    def set_atom_element(self, atom_id, element):
        self.w._push_undo()
        self.w.backend.set_atom_type_by_id(atom_id, element)

    def add_h_caps(self):
        self.w._push_undo()
        self.w.backend.add_h_caps()

    def select_ids(self, ids):
        self.w.scene.exit_xform(commit=True)
        self.w.scene.set_selected_ids(list(ids))
        GSU.process_events(self.w)

    def copy_selection(self):
        self.w.copy_selected_atoms()
        GSU.process_events(self.w)

    def paste_selection(self):
        self.w._last_p_world = None
        self.w.scene.exit_xform(commit=True)
        ids = self.w.paste_copied_atoms()
        self.w.scene.exit_xform(commit=True)
        self._last_paste_ids = list(ids or [])
        GSU.process_events(self.w)
        return self._last_paste_ids

    def exit_xform(self):
        self.w.scene.exit_xform(commit=True)
        GSU.process_events(self.w)

    def translate_selected(self, dx, dy):
        self.w.scene.exit_xform(commit=True)
        self.w.translate_selected(dx, dy)
        GSU.process_events(self.w)

    def rotate_selected(self, deg):
        self.w.scene.exit_xform(commit=True)
        self.w.rotate_selected(deg)
        GSU.process_events(self.w)

    def refresh(self):
        self.w.refresh_view()
        GSU.process_events(self.w)


def _parse_argv(argv):
    p = argparse.ArgumentParser(description='GUI azaindol draw demo → PNG + GIF')
    p.add_argument('--out', default=None)
    p.add_argument('--relax', action='store_true')
    p.add_argument('--canvas-only', action='store_true',
                   help='VisPy viewport only (default: full window + panels)')
    p.add_argument('--zoom-out', type=float, default=2.0,
                   help='Extra zoom-out factor for fit (default 2)')
    p.add_argument('--gif-ms', type=int, default=700, help='GIF frame duration ms')
    p.add_argument('--no-gif', action='store_true')
    return p.parse_args(argv)


def run(window, argv=None):
    args = _parse_argv(argv or [])
    out = args.out or os.path.join(os.path.dirname(__file__), '..', '..', '..', 'debug', 'azaindol_draw_demo')
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    if hasattr(window, 'b2Dview_chk') and not window.b2Dview_chk.isChecked():
        window.b2Dview_chk.setChecked(True)
        GSU.process_events(window)
    host = GuiHost(window, out, full_window=not args.canvas_only, zoom_out=args.zoom_out)
    paths = run_azaindol_draw(host, do_relax=args.relax, out_dir=out)
    xyz = os.path.join(out, 'azaindol_dimer_drawn.xyz')
    window.backend.save_xyz(xyz, comment='azaindol_draw_demo')
    print(f'REVIEW: {xyz}')
    gif = None
    if not args.no_gif and paths:
        gif = os.path.join(out, 'azaindol_draw_demo.gif')
        GSU.frames_to_gif(paths, gif, duration_ms=args.gif_ms)
        print(f'REVIEW: {gif}')
    print(f'REVIEW: {len(paths)} PNG frames in {out}')
    window.statusBar().showMessage(f'Azaindol draw demo done → {out}')
    return {'paths': paths, 'gif': gif, 'out': out}


if __name__ == '__main__':
    print('Use: ./run_gui.sh --script demos/gui_scripts/azaindol_draw_demo.py', file=sys.stderr)
    raise SystemExit(1)
