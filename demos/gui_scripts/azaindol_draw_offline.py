#!/usr/bin/env python3
"""Headless azaindol draw demo → one SVG per step (same ops as GUI script).

  PYTHONPATH=. python demos/gui_scripts/azaindol_draw_offline.py
  PYTHONPATH=. python demos/gui_scripts/azaindol_draw_offline.py --relax
"""
import argparse
import os
import sys

import numpy as np

from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
from spammm.topology.PackedMolecule import PackedMolecule, _z_to_ename
from spammm.GUI.azaindol_draw_sequence import SequenceHost, run_azaindol_draw, render_editor_svg


class OfflineHost(SequenceHost):
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.b = MoleculeEditorBackend()
        self.b.auto_h_cap = False
        self._clipboard = None
        self._selected = []

    def backend(self):
        return self.b

    def snapshot(self, name, title='', **kw):
        path = os.path.join(self.out_dir, f'{name}.svg')
        keys = ('cursor_xy', 'highlight_ids', 'bond_highlight', 'selection_ids',
                'xform_mode', 'hover_atom_id', 'hover_hex', 'ring_preview', 'cursor_style')
        ov = {k: kw[k] for k in keys if k in kw}
        render_editor_svg(self.b, path, title=title, **ov)
        png = os.path.join(self.out_dir, f'{name}.png')
        render_editor_svg(self.b, png, title=title, **ov)
        return path

    def set_auto_h(self, on):
        self.b.auto_h_cap = bool(on)

    def set_edit_mode(self, mode):
        pass

    def set_ring_size(self, n):
        pass

    def set_element(self, element):
        pass

    def add_hex(self, q, r):
        self.b.add_ring(q, r)

    def add_adj_ring(self, bond, n_members, side):
        return self.b.add_adjacent_ring(bond, n_members=n_members, ename='C', side=side)

    def set_atom_element(self, atom_id, element):
        self.b.set_atom_type_by_id(atom_id, element)

    def add_h_caps(self):
        self.b.add_h_caps()

    def select_ids(self, ids):
        self._selected = list(ids)

    def copy_selection(self):
        if not self._selected:
            return
        atom_list, *_ = self.b.graph.to_arrays()
        id_to_idx = {a._id: i for i, a in enumerate(atom_list)}
        indices = [id_to_idx[i] for i in self._selected if i in id_to_idx]
        self._clipboard = PackedMolecule.from_graph(self.b.graph, atom_indices=indices)

    def paste_selection(self):
        packed = self._clipboard
        if packed is None:
            return []
        paste_shift = np.array([1.5, 1.5, 0.0])  # GUI default when no mouse hit
        new_ids, new_atoms = [], []
        for i in range(len(packed.etype)):
            ename = _z_to_ename(int(packed.etype[i]))
            npi_i = int(packed.npi[i]) if i < len(packed.npi) else 1
            pos = list(packed.apos[i] + paste_shift)
            a = self.b._append_atom(pos=pos, ename=ename, pin=None, parent=None, npi=npi_i)
            new_ids.append(a._id)
            new_atoms.append(a)
        for col in packed.bonds:
            i, j = int(col[0]), int(col[1])
            if 0 <= i < len(new_atoms) and 0 <= j < len(new_atoms):
                self.b.graph.add_bond(new_atoms[i], new_atoms[j])
        self.b.graph.sync_neighbor_lists()
        self.b._sync_sys()
        self._selected = list(new_ids)
        return new_ids

    def translate_selected(self, dx, dy):
        for aid in self._selected:
            a = self.b.graph.atoms.get(aid)
            if a is None or not a.alive:
                continue
            a.pos[0] += float(dx)
            a.pos[1] += float(dy)
        self.b._sync_sys()

    def rotate_selected(self, deg):
        atoms = [self.b.graph.atoms[i] for i in self._selected if i in self.b.graph.atoms and self.b.graph.atoms[i].alive]
        if not atoms:
            return
        com = np.mean([a.pos for a in atoms], axis=0)
        ang = np.deg2rad(float(deg))
        c, s = np.cos(ang), np.sin(ang)
        for a in atoms:
            d = a.pos - com
            a.pos[0] = com[0] + c * d[0] - s * d[1]
            a.pos[1] = com[1] + s * d[0] + c * d[1]
        self.b._sync_sys()


def main(argv=None):
    p = argparse.ArgumentParser(description='Offline azaindol draw → SVG frames')
    p.add_argument('--out', default=None, help='Output dir (default debug/azaindol_draw_offline)')
    p.add_argument('--relax', action='store_true', help='Run DFTB relax after H caps')
    p.add_argument('--save-xyz', default=None, help='Optional final XYZ path')
    args = p.parse_args(argv)
    out = args.out or os.path.join(os.path.dirname(__file__), '..', '..', '..', 'debug', 'azaindol_draw_offline')
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    host = OfflineHost(out)
    paths = run_azaindol_draw(host, do_relax=args.relax, out_dir=out)
    xyz = args.save_xyz or os.path.join(out, 'azaindol_dimer_drawn.xyz')
    host.b.save_xyz(xyz, comment='azaindol_draw_offline')
    print(f'REVIEW: {xyz}')
    print(f'REVIEW: {len(paths)} SVG frames in {out}')
    return paths


if __name__ == '__main__':
    main()
