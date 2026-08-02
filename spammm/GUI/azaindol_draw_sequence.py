"""azaindol_draw_sequence.py — SSOT step sequence for hex→7-azaindole→H-bond dimer demos.

Motivation: one scripted workflow that both presentation SVG and live GUI GIF can
replay, so slides and screenshots stay semantically identical while only rendering
differs. Mirrors real SPAMMM_GUI tools (Ring / Atom / Select, Auto H, Ctrl-C/V, δ/φ)
rather than inventing a separate builder API.

Design:
- **run_azaindol_draw(host)** — host protocol: same editor ops; only **snapshot()**
  differs (matplotlib SVG vs VisPy/Qt PNG).
- Geometry picks (rightmost hex edge, N sites relative to fusion bridge) — atom
  ``_id`` is a global counter and must not be hardcoded across backend instances.
- Pyrrole NH needs ``nπ=0`` after C→N so ``add_h_caps`` adds H; pyridine stays ``nπ=1``.
- Dimer pose: 180° about selection COM + COM shift from ``data/xyz/azaindol_dimer.xyz``.

Use: ``gui_scripts/azaindol_draw_offline.py``, ``azaindol_draw_demo.py``.
Doc: ``doc/Topics/GUI_DrawDemo_Scripts.md``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_REF_MONO = _REPO / 'data' / 'xyz' / 'azaindol.xyz'
_REF_DIMER = _REPO / 'data' / 'xyz' / 'azaindol_dimer.xyz'
_ELEM_COLORS = {'H': '#bbbbbb', 'C': '#222222', 'N': '#2244cc', 'O': '#cc2222', 'S': '#cccc22'}


def hex_center_xy(backend, q=0, r=0):
    nodes = np.asarray(backend.grid.ring_nodes(q, r), dtype=np.float64)
    return nodes.mean(axis=0)


def pick_rightmost_bond(backend):
    """Hex edge with largest midpoint-x (stable fusion edge for azaindol demo)."""
    bonds = [bd for bd in backend.graph.bonds.values() if bd.alive]
    if not bonds:
        raise RuntimeError('pick_rightmost_bond: no bonds')
    return max(bonds, key=lambda bd: 0.5 * (bd.a.pos[0] + bd.b.pos[0]))


def azaindol_nitrogen_ids(backend, fusion_bond, new_pent_atoms):
    """7-azaindole N sites: hex + pent neighbors of the upper fusion bridgehead."""
    fusion = {fusion_bond.a, fusion_bond.b}
    pent = set(new_pent_atoms)
    bridge = max(fusion, key=lambda a: float(a.pos[1]))
    hex_n = pent_n = None
    for n in bridge.neighbors:
        if not n.alive or n.ename == 'H' or n.npi == -1 or n in fusion:
            continue
        if n in pent:
            pent_n = n._id
        else:
            hex_n = n._id
    if hex_n is None or pent_n is None:
        raise RuntimeError(f'azaindol_nitrogen_ids: hex_n={hex_n} pent_n={pent_n} bridge={bridge._id}')
    return [hex_n, pent_n]


def dimer_com_delta_xy(mono_xyz=_REF_MONO, dimer_xyz=_REF_DIMER):
    """Relative COM shift of second monomer in reference dimer (heavy atoms, XY)."""
    def heavy_xy(path):
        rows = []
        with open(path) as f:
            for ln in f.readlines()[2:]:
                p = ln.split()
                if p[0] == 'H':
                    continue
                rows.append([float(p[1]), float(p[2])])
        return np.asarray(rows, dtype=np.float64)
    m = heavy_xy(mono_xyz)
    d = heavy_xy(dimer_xyz)
    if len(d) < 2 * len(m):
        raise RuntimeError(f'dimer_com_delta_xy: dimer heavy={len(d)} mono={len(m)}')
    n = len(m)
    return d[n:2 * n].mean(axis=0) - d[:n].mean(axis=0)


def alive_atom_ids(backend, heavy_only=False):
    out = []
    for a in backend.graph.atoms.values():
        if not a.alive:
            continue
        if heavy_only and (a.ename == 'H' or a.npi == -1):
            continue
        out.append(a._id)
    return out


def selection_com_xy(backend, ids):
    atoms = [backend.graph.atoms[i] for i in ids if i in backend.graph.atoms and backend.graph.atoms[i].alive]
    if not atoms:
        return np.zeros(2)
    return np.mean([a.pos[:2] for a in atoms], axis=0)


# ── SVG renderer (headless path) ─────────────────────────────────────────────

def _selection_aabb(backend, ids, pad=0.4):
    atoms = [backend.graph.atoms[i] for i in ids if i in backend.graph.atoms and backend.graph.atoms[i].alive]
    if not atoms:
        return None
    xs = [a.pos[0] for a in atoms]; ys = [a.pos[1] for a in atoms]
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    if xmax - xmin < 0.8:
        c = 0.5 * (xmin + xmax); xmin, xmax = c - 0.4, c + 0.4
    if ymax - ymin < 0.8:
        c = 0.5 * (ymin + ymax); ymin, ymax = c - 0.4, c + 0.4
    return xmin, xmax, ymin, ymax


def render_editor_svg(backend, path, title='', cursor_xy=None, highlight_ids=None,
                      guide_qrange=(-2, 3), guide_rrange=(-2, 3), bond_highlight=None,
                      selection_ids=None, xform_mode=None, hover_atom_id=None,
                      hover_hex=None, ring_preview=None, cursor_style='pointer'):
    """Ball-stick + GUI-like overlays → SVG/PNG via matplotlib.

    Overlays (mirror VisPy editor chrome):
      cursor_xy       — mouse pointer
      highlight_ids   — green atom halos (new/target atoms)
      bond_highlight  — orange thick bond (hover / fusion edge)
      hover_atom_id   — lime atom hover marker
      hover_hex       — (q,r) orange hex-node hover like RingMode
      ring_preview    — (n,2)/(n,3) ghost n-gon outline (cyan)
      selection_ids   — AABB + δ (BL) / φ (TR) handles
      xform_mode      — 'move'|'rotate' to warm-highlight active handle
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Polygon

    highlight_ids = set(highlight_ids or [])
    selection_ids = list(selection_ids) if selection_ids is not None else None
    atom_list, enames, apos, atypes, bonds_idx, bond_list, ring_list = backend.graph.to_arrays()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # Hex guide dots
    try:
        guides = backend.get_guide_points(qrange=guide_qrange, rrange=guide_rrange)
        if guides is not None and len(guides):
            g = np.asarray(guides)
            ax.scatter(g[:, 0], g[:, 1], s=8, c='#cccccc', zorder=0, marker='.')
    except Exception:
        pass
    # Detected rings (magenta dashed)
    for r in backend.graph.rings.values():
        if not r.alive:
            continue
        ring_atoms = [a for a in r.atoms if a.alive]
        if len(ring_atoms) < 3:
            continue
        xs = [a.pos[0] for a in ring_atoms] + [ring_atoms[0].pos[0]]
        ys = [a.pos[1] for a in ring_atoms] + [ring_atoms[0].pos[1]]
        ax.plot(xs, ys, color='magenta', lw=2.0, ls='--', alpha=0.45, zorder=0.5)
    # Hover hex tile (orange nodes + outline) — RingMode hex hover
    if hover_hex is not None:
        q, r = hover_hex
        nodes = np.asarray(backend.grid.ring_nodes(q, r), dtype=np.float64)
        xs = list(nodes[:, 0]) + [nodes[0, 0]]
        ys = list(nodes[:, 1]) + [nodes[0, 1]]
        ax.plot(xs, ys, color='orange', lw=2.0, ls='-', alpha=0.85, zorder=0.6)
        ax.scatter(nodes[:, 0], nodes[:, 1], s=60, facecolors='none', edgecolors='orange',
                   linewidths=2.0, zorder=0.7)
    # Ring preview ghost (adjacent n-gon)
    if ring_preview is not None:
        rp = np.asarray(ring_preview, dtype=np.float64)
        xs = list(rp[:, 0]) + [rp[0, 0]]
        ys = list(rp[:, 1]) + [rp[0, 1]]
        ax.plot(xs, ys, color='#22cccc', lw=2.5, ls='--', alpha=0.85, zorder=0.8)
    # Bonds
    for i, j in bonds_idx:
        pair = (atom_list[i]._id, atom_list[j]._id)
        col, lw = 'k', 1.8
        if bond_highlight is not None:
            bh = set(bond_highlight) if not isinstance(bond_highlight, set) else bond_highlight
            if set(pair) == set(bh) or frozenset(pair) == frozenset(bh):
                col, lw = '#ff6600', 4.0
        ax.plot([apos[i, 0], apos[j, 0]], [apos[i, 1], apos[j, 1]], color=col, lw=lw, zorder=1)
    # Atoms
    id_to_pos = {}
    for ia, (e, p) in enumerate(zip(enames, apos)):
        aid = atom_list[ia]._id
        id_to_pos[aid] = p[:2]
        c = _ELEM_COLORS.get(e, '#8844aa')
        s = 120 if e != 'H' else 40
        ax.scatter(p[0], p[1], c=c, s=s, zorder=2, edgecolors='black', linewidths=0.6)
        if e != 'H':
            ax.text(p[0], p[1], e, fontsize=8, ha='center', va='center', zorder=4,
                    color='white' if e in ('C', 'N') else 'black')
        if aid in highlight_ids:
            ax.scatter(p[0], p[1], facecolors='none', edgecolors='#33cc33', s=s * 2.2, lw=2.0, zorder=5)
    # Hover atom (lime) — like hover_atom_marker
    if hover_atom_id is not None and hover_atom_id in id_to_pos:
        hp = id_to_pos[hover_atom_id]
        ax.scatter(hp[0], hp[1], facecolors='none', edgecolors='lime', s=380, lw=2.5, zorder=6)
    # Selection AABB + δ/φ handles (VisPy Select mode)
    if selection_ids:
        aabb = _selection_aabb(backend, selection_ids)
        if aabb is not None:
            xmin, xmax, ymin, ymax = aabb
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, edgecolor='#268cff', lw=2.0, zorder=7))
            # selection halos
            for aid in selection_ids:
                if aid in id_to_pos:
                    p = id_to_pos[aid]
                    ax.scatter(p[0], p[1], facecolors='none', edgecolors='#268cff', s=280, lw=1.5, zorder=6)
            move_on = xform_mode == 'move'
            rot_on = xform_mode == 'rotate'
            for (hx, hy), label, on, idle_c in (
                ((xmin, ymin), 'δ', move_on, '#26bff2'),
                ((xmax, ymax), 'φ', rot_on, '#d940e6'),
            ):
                face = '#ffd91a' if on else idle_c
                size = 0.45 if on else 0.35
                ax.add_patch(Rectangle((hx - size / 2, hy - size / 2), size, size,
                                       facecolor=face, edgecolor='black', lw=1.2, zorder=8))
                ax.text(hx, hy, label, fontsize=11, fontweight='bold', ha='center', va='center',
                        zorder=9, color='black')
    # Mouse cursor
    if cursor_xy is not None:
        cx, cy = float(cursor_xy[0]), float(cursor_xy[1])
        if cursor_style == 'cross':
            ax.plot(cx, cy, 'x', color='red', markersize=14, markeredgewidth=2.5, zorder=10)
            ax.add_patch(Circle((cx, cy), 0.35, fill=False, color='red', ls='--', lw=1.2, zorder=10))
        else:
            # Simple pointer triangle + hotspot
            tip = np.array([cx, cy])
            poly = tip + np.array([[0, 0], [0.0, -0.55], [0.22, -0.42]])
            ax.add_patch(Polygon(poly, closed=True, facecolor='black', edgecolor='white', lw=0.8, zorder=11))
            ax.plot(cx, cy, 'o', color='red', markersize=4, zorder=12)
    ax.set_aspect('equal')
    ax.set_title(title or path)
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    # Autoscale with margin (include overlays)
    xs_all, ys_all = [], []
    if len(apos):
        xs_all.extend(apos[:, 0]); ys_all.extend(apos[:, 1])
    if cursor_xy is not None:
        xs_all.append(float(cursor_xy[0])); ys_all.append(float(cursor_xy[1]))
    if selection_ids:
        aabb = _selection_aabb(backend, selection_ids)
        if aabb is not None:
            xs_all.extend([aabb[0], aabb[1]]); ys_all.extend([aabb[2], aabb[3]])
    if hover_hex is not None:
        nodes = np.asarray(backend.grid.ring_nodes(*hover_hex), dtype=np.float64)
        xs_all.extend(nodes[:, 0]); ys_all.extend(nodes[:, 1])
    if ring_preview is not None:
        rp = np.asarray(ring_preview)
        xs_all.extend(rp[:, 0]); ys_all.extend(rp[:, 1])
    if xs_all:
        pad = 1.8
        ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
        ax.set_ylim(min(ys_all) - pad, max(ys_all) + pad)
    else:
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    ext = os.path.splitext(path)[1].lower().lstrip('.') or 'svg'
    fig.savefig(path, format=ext, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path


# ── Host protocol (implemented by offline + GUI runners) ─────────────────────

class SequenceHost:
    """Minimal surface both runners implement — maps 1:1 to GUI editor tools."""

    def backend(self):
        raise NotImplementedError

    def snapshot(self, name, title='', **kw):
        raise NotImplementedError

    def set_auto_h(self, on): raise NotImplementedError
    def set_edit_mode(self, mode): raise NotImplementedError
    def set_ring_size(self, n): raise NotImplementedError
    def set_element(self, element): raise NotImplementedError
    def add_hex(self, q, r): raise NotImplementedError
    def add_adj_ring(self, bond, n_members, side): raise NotImplementedError
    def set_atom_element(self, atom_id, element): raise NotImplementedError
    def add_h_caps(self): raise NotImplementedError
    def select_ids(self, ids): raise NotImplementedError
    def copy_selection(self): raise NotImplementedError
    def paste_selection(self): raise NotImplementedError
    def exit_xform(self): pass
    def translate_selected(self, dx, dy): raise NotImplementedError
    def rotate_selected(self, deg): raise NotImplementedError
    def refresh(self): pass


def run_azaindol_draw(host: SequenceHost, *, do_relax=False, out_dir=None):
    """Execute the shared draw sequence. Generator: yields title after each snapshot.

    Paths are stored on ``host.paths`` after iteration completes.
    """
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    b = host.backend()
    paths = []
    delta = dimer_com_delta_xy()

    def snap(name, title, **kw):
        p = host.snapshot(name, title=title, **kw)
        if p:
            paths.append(p)
            print(f'REVIEW: {p}')
        return p

    host.set_auto_h(False)
    host.set_edit_mode('Ring')
    host.set_ring_size(6)
    host.refresh()
    cx = hex_center_xy(b, 0, 0)

    # --- Drawing: empty → hover hex → click hex → hover bond+preview → click pent → N×2 ---
    snap('00_empty', '0) Empty canvas — Ring mode, Auto H off')
    yield '0) Empty canvas — Ring mode, Auto H off'
    snap('00b_hover_hex', '0b) Hover hex tile (orange nodes + cyan foreshadow)',
         cursor_xy=cx, hover_hex=(0, 0))
    yield '0b) Hover hex tile (orange nodes + cyan foreshadow)'

    host.add_hex(0, 0)
    host.refresh()
    snap('01_hex', '1) Click — hex ring materialized',
         cursor_xy=cx, hover_hex=(0, 0))
    yield '1) Click — hex ring materialized'

    host.set_ring_size(5)
    bond = pick_rightmost_bond(b)
    mid = 0.5 * (bond.a.pos[:2] + bond.b.pos[:2])
    side = +1
    d = bond.b.pos[:2] - bond.a.pos[:2]
    dn = np.linalg.norm(d)
    perp = np.array([-d[1], d[0]]) / dn * side
    cursor_bond = mid + 0.4 * perp
    preview = MoleculeEditorBackend.compute_adjacent_ring_positions(bond, n_members=5, side=side)
    # Critical UX frame: cyan ghost BEFORE click
    snap('01b_hover_bond', '1b) Hover bond — cyan 5-ring foreshadow (before click)',
         cursor_xy=cursor_bond, bond_highlight=(bond.a._id, bond.b._id),
         ring_preview=preview)
    yield '1b) Hover bond — cyan 5-ring foreshadow (before click)'

    new_pent = host.add_adj_ring(bond, 5, side)
    host.refresh()
    snap('02_pentagon', '2) Click — pentagon fused to hex',
         cursor_xy=cursor_bond, highlight_ids=[a._id for a in new_pent])
    yield '2) Click — pentagon fused to hex'

    n_ids = azaindol_nitrogen_ids(b, bond, new_pent)
    # Prefer pyridine (hex) then pyrrole (pent) for storytelling
    pent_ids = {a._id for a in new_pent}
    n_hex = next(i for i in n_ids if i not in pent_ids)
    n_pent = next(i for i in n_ids if i in pent_ids)
    host.set_edit_mode('Atom')
    host.set_element('N')

    a_hex = b.graph.atoms[n_hex]
    snap('02b_hover_N1', '2b) Atom mode N — hover pyridine site (hex)',
         cursor_xy=a_hex.pos[:2], hover_atom_id=n_hex, highlight_ids=[n_hex, n_pent])
    yield '2b) Atom mode N — hover pyridine site (hex)'
    host.set_atom_element(n_hex, 'N')
    host.refresh()
    snap('02c_N1_done', '2c) Click — first C→N (pyridine)',
         cursor_xy=a_hex.pos[:2], highlight_ids=[n_hex])
    yield '2c) Click — first C→N (pyridine)'

    a_pent = b.graph.atoms[n_pent]
    snap('02d_hover_N2', '2d) Hover pyrrole site (pentagon)',
         cursor_xy=a_pent.pos[:2], hover_atom_id=n_pent, highlight_ids=[n_hex, n_pent])
    yield '2d) Hover pyrrole site (pentagon)'
    host.set_atom_element(n_pent, 'N')
    b.set_atom_npi_by_id(n_pent, 0)  # pyrrole NH: nπ=0
    host.refresh()
    snap('03_azaindol_skel', '3) Click — second C→N (pyrrole nπ=0) → azaindol',
         cursor_xy=a_pent.pos[:2], highlight_ids=[n_hex, n_pent])
    yield '3) Click — second C→N (pyrrole nπ=0) → azaindol'

    host.set_auto_h(True)
    host.add_h_caps()
    host.refresh()
    snap('04_hydrogens', '4) Explicit hydrogens (Auto H / add caps)')
    yield '4) Explicit hydrogens (Auto H / add caps)'

    if do_relax:
        host.set_edit_mode('Select')
        b.run_relaxation(workdir=os.path.join(out_dir or '.', 'dftb_relax'))
        host.refresh()
        snap('04b_relaxed', '4b) DFTB relaxation')
        yield '4b) DFTB relaxation'

    all_ids = alive_atom_ids(b)
    host.set_edit_mode('Select')
    host.select_ids(all_ids)
    host.refresh()
    com = selection_com_xy(b, all_ids)
    snap('05_selected', '5) Block-select monomer — δ/φ handles',
         selection_ids=all_ids, cursor_xy=com + np.array([0.3, -0.3]))
    yield '5) Block-select monomer — δ/φ handles'
    host.copy_selection()
    snap('05b_copied', '5b) Copy (Ctrl-C)',
         selection_ids=all_ids, cursor_xy=com + np.array([0.3, -0.3]))
    yield '5b) Copy (Ctrl-C)'

    mono_com = selection_com_xy(b, all_ids)
    new_ids = host.paste_selection()
    host.exit_xform()
    if not new_ids:
        raise RuntimeError('paste_selection returned no atom ids')
    host.select_ids(new_ids)
    host.refresh()
    paste_com0 = selection_com_xy(b, new_ids)
    # Sticky δ-move: cursor near move handle
    aabb = _selection_aabb(b, new_ids)
    snap('06_pasted', '6) Paste (Ctrl-V) — sticky δ-move',
         selection_ids=new_ids, xform_mode='move',
         cursor_xy=np.array([aabb[0], aabb[2]]) if aabb else paste_com0)
    yield '6) Paste (Ctrl-V) — sticky δ-move'

    host.rotate_selected(180.0)
    host.refresh()
    aabb = _selection_aabb(b, new_ids)
    snap('07_rotated', '7) Rotate selection 180° — φ handle',
         selection_ids=new_ids, xform_mode='rotate',
         cursor_xy=np.array([aabb[1], aabb[3]]) if aabb else selection_com_xy(b, new_ids))
    yield '7) Rotate selection 180° — φ handle'

    paste_com = selection_com_xy(b, new_ids)
    target = mono_com + delta
    host.translate_selected(float(target[0] - paste_com[0]), float(target[1] - paste_com[1]))
    host.refresh()
    aabb = _selection_aabb(b, new_ids)
    snap('08_dimer', '8) Translate → azaindol dimer — δ handle',
         selection_ids=new_ids, xform_mode='move',
         cursor_xy=np.array([aabb[0], aabb[2]]) if aabb else selection_com_xy(b, new_ids))
    yield '8) Translate → azaindol dimer — δ handle'

    host.select_ids([])
    host.refresh()
    snap('09_done', '9) Done — azaindol dimer')
    yield '9) Done — azaindol dimer'
    host.paths = paths
