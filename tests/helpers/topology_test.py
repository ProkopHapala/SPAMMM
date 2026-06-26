"""
topology_test.py — Helpers for headless testing of molecular editing operations.

Two-layer testing:
  L1: TopologySnapshot / TopologyDiff — capture graph state, assert diffs (always, fast)
  L2: render_graph / render_before_after — matplotlib PNG output for human review (--visual)

Usage in pytest:
  snap_before = TopologySnapshot(backend.graph)
  backend.add_ring(0, 0)
  snap_after = TopologySnapshot(backend.graph)
  diff = snap_before.diff(snap_after)
  diff.assert_counts('add_ring', added_atoms=6, added_bonds=6)
  if visual_output_dir:
      render_before_after(backend.graph, savepath=..., diff=diff)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend — no display required
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# L1: Topology Snapshot + Diff
# ═══════════════════════════════════════════════════════════════════════════════

class TopologySnapshot:
    """Captures AtomicGraph state for before/after comparison.

    Stores atoms, bonds, rings, and neighbor lists keyed by stable object _id.
    Does NOT store positions (those change on geometry ops, not topology ops).
    """

    def __init__(self, graph):
        self.atoms = {a._id: (a.ename, a.subtype)
                      for a in graph.atoms.values() if a.alive}
        self.bonds = {frozenset((b.a._id, b.b._id)): b.order
                      for b in graph.bonds.values() if b.alive and b.a.alive and b.b.alive}
        self.rings = {r._id: tuple(sorted(a._id for a in r.atoms if a.alive))
                      for r in graph.rings.values() if r.alive}
        self.n_atoms = len(self.atoms)
        self.n_bonds = len(self.bonds)
        self.n_rings = len(self.rings)

    def diff(self, other):
        return TopologyDiff(
            added_atoms   = set(other.atoms) - set(self.atoms),
            removed_atoms = set(self.atoms) - set(other.atoms),
            added_bonds   = set(other.bonds) - set(self.bonds),
            removed_bonds = set(self.bonds) - set(other.bonds),
            added_rings   = set(other.rings) - set(self.rings),
            removed_rings = set(self.rings) - set(other.rings),
        )


class TopologyDiff:
    """Structured diff between two TopologySnapshots."""

    def __init__(self, added_atoms, removed_atoms, added_bonds, removed_bonds, added_rings, removed_rings):
        self.added_atoms   = added_atoms
        self.removed_atoms = removed_atoms
        self.added_bonds   = added_bonds
        self.removed_bonds = removed_bonds
        self.added_rings   = added_rings
        self.removed_rings = removed_rings

    def assert_empty(self, name=''):
        assert not self.added_atoms,   f'{name}: unexpected added atoms {self.added_atoms}'
        assert not self.removed_atoms, f'{name}: unexpected removed atoms {self.removed_atoms}'
        assert not self.added_bonds,   f'{name}: unexpected added bonds {self.added_bonds}'
        assert not self.removed_bonds, f'{name}: unexpected removed bonds {self.removed_bonds}'

    def assert_counts(self, name='', **expected):
        for key in ('added_atoms', 'removed_atoms', 'added_bonds', 'removed_bonds', 'added_rings', 'removed_rings'):
            if key in expected:
                actual = len(getattr(self, key))
                assert actual == expected[key], f'{name}: {key} expected {expected[key]}, got {actual} (got={getattr(self, key)})'

    def __repr__(self):
        return (f'TopologyDiff(+{len(self.added_atoms)}a/-{len(self.removed_atoms)}a, '
                f'+{len(self.added_bonds)}b/-{len(self.removed_bonds)}b, '
                f'+{len(self.added_rings)}r/-{len(self.removed_rings)}r)')


# ═══════════════════════════════════════════════════════════════════════════════
# L2: Visual Rendering (headless matplotlib Agg)
# ═══════════════════════════════════════════════════════════════════════════════

_ELEM_COLORS = {'H': 'lightgray', 'C': 'black', 'O': 'red', 'N': 'blue', 'S': 'yellow', 'Cl': 'green', 'E': 'orange'}


def render_graph(ax, graph, title='', proj='xy',
                 highlight_atoms=None, highlight_color='orange',
                 cursor_pos=None, cursor_radius=0.5,
                 diff=None):
    """Render AtomicGraph to matplotlib Axes with annotations.

    Args:
        ax: matplotlib Axes
        graph: AtomicGraph (will call to_arrays())
        title: plot title
        proj: projection plane ('xy', 'xz', 'yz')
        highlight_atoms: set of atom _ids to highlight (selection halo)
        highlight_color: color for highlight halo
        cursor_pos: (x,y) world coords — draws crosshair to simulate mouse click
        cursor_radius: radius for cursor circle
        diff: TopologyDiff for coloring changes (green=added, red=removed, blue=new bonds)
    """
    axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    ax_i, ax_j = axis_map.get(proj, (0, 1))
    highlight_atoms = highlight_atoms or set()
    added_atom_ids = diff.added_atoms if diff else set()
    removed_atom_ids = diff.removed_atoms if diff else set()
    added_bond_pairs = diff.added_bonds if diff else set()
    removed_bond_pairs = diff.removed_bonds if diff else set()

    atom_list, enames, apos, atypes, bonds_idx, bond_list, ring_list = graph.to_arrays()
    id_to_idx = {a._id: i for i, a in enumerate(atom_list)}

    # Draw bonds
    for bi, (i, j) in enumerate(bonds_idx):
        pair = frozenset((atom_list[i]._id, atom_list[j]._id))
        r = float(np.linalg.norm(apos[i] - apos[j]))
        mx, my = (apos[i, ax_i] + apos[j, ax_i]) / 2, (apos[i, ax_j] + apos[j, ax_j]) / 2
        if pair in added_bond_pairs:
            ax.plot([apos[i, ax_i], apos[j, ax_i]], [apos[i, ax_j], apos[j, ax_j]], 'b-', linewidth=3, zorder=1)
        else:
            ax.plot([apos[i, ax_i], apos[j, ax_i]], [apos[i, ax_j], apos[j, ax_j]], 'k-', linewidth=2, zorder=1)
        ax.text(mx, my, f'{r:.2f}', fontsize=6, color='blue', ha='center', va='center', zorder=3)

    # Draw removed bonds (dashed red) — need positions from before-snapshot, skip if not available
    # (removed atoms are gone from graph, so we can't draw their bonds here)

    # Draw rings (hexagonal outlines)
    for r in graph.rings.values():
        if not r.alive:
            continue
        ring_atoms = [a for a in r.atoms if a.alive]
        if len(ring_atoms) < 3:
            continue
        xs = [a.pos[ax_i] for a in ring_atoms] + [ring_atoms[0].pos[ax_i]]
        ys = [a.pos[ax_j] for a in ring_atoms] + [ring_atoms[0].pos[ax_j]]
        ax.plot(xs, ys, color='magenta', linewidth=3, linestyle='--', zorder=0.5, alpha=0.6)
        # Draw ring center marker
        cx = sum(a.pos[ax_i] for a in ring_atoms) / len(ring_atoms)
        cy = sum(a.pos[ax_j] for a in ring_atoms) / len(ring_atoms)
        ax.scatter(cx, cy, marker='*', c='magenta', s=100, zorder=0.7)

    # Draw atoms
    for ia, (e, p) in enumerate(zip(enames, apos)):
        aid = atom_list[ia]._id
        c = _ELEM_COLORS.get(e, 'purple')
        if aid in added_atom_ids:
            c = 'green'
        ax.scatter(p[ax_i], p[ax_j], c=c, s=200, zorder=2, edgecolors='black')
        ax.text(p[ax_i], p[ax_j], f'{aid}:{e}', fontsize=7, ha='center', va='center', zorder=4, color='white' if c == 'black' else 'black')
        if aid in highlight_atoms:
            ax.scatter(p[ax_i], p[ax_j], facecolors='none', edgecolors=highlight_color, s=350, linewidth=2.5, zorder=5)

    # Draw cursor (simulated mouse click)
    if cursor_pos is not None:
        cx, cy = cursor_pos
        ax.plot(cx, cy, 'x', color='red', markersize=15, markeredgewidth=3, zorder=10)
        circle = plt.Circle((cx, cy), cursor_radius, fill=False, color='red', linestyle='--', linewidth=1.5, zorder=10)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(f'{proj[0]} [A]')
    ax.set_ylabel(f'{proj[1]} [A]')


def render_before_after(graph_before, graph_after, savepath,
                        title_before='Before', title_after='After',
                        highlight_atoms=None, cursor_pos=None, diff=None,
                        proj='xy'):
    """Save side-by-side before/after PNG for human review.

    Args:
        graph_before: AtomicGraph state before operation (will be rendered as-is)
        graph_after: AtomicGraph state after operation
        savepath: output .png path
        diff: TopologyDiff for coloring changes in the 'after' panel
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    render_graph(ax1, graph_before, title_before, proj=proj,
                 highlight_atoms=highlight_atoms, cursor_pos=cursor_pos)
    render_graph(ax2, graph_after, title_after, proj=proj,
                 highlight_atoms=highlight_atoms, diff=diff)
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [visual] Saved: {savepath}')


# ═══════════════════════════════════════════════════════════════════════════════
# L2b: AtomicSystem Rendering (for electron pairs test)
# ═══════════════════════════════════════════════════════════════════════════════

def render_sys(ax, sys, title='', proj='xy', epairs_added=None):
    """Render AtomicSystem to matplotlib Axes.

    Args:
        ax: matplotlib Axes
        sys: AtomicSystem with apos, enames, bonds, ngs
        title: plot title
        proj: projection plane
        epairs_added: number of electron pairs added (for annotation)
    """
    axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    ax_i, ax_j = axis_map.get(proj, (0, 1))
    epairs_added = epairs_added or 0

    # Draw bonds with length labels
    if sys.bonds is not None:
        for (i, j) in sys.bonds:
            r = float(np.linalg.norm(sys.apos[i] - sys.apos[j]))
            mx = (sys.apos[i, ax_i] + sys.apos[j, ax_i]) / 2
            my = (sys.apos[i, ax_j] + sys.apos[j, ax_j]) / 2
            clr = 'orange' if (sys.enames[i] == 'E' or sys.enames[j] == 'E') else 'black'
            ax.plot([sys.apos[i, ax_i], sys.apos[j, ax_i]],
                    [sys.apos[i, ax_j], sys.apos[j, ax_j]], color=clr, linewidth=2, zorder=1)
            ax.text(mx, my, f'{r:.2f}', fontsize=6, color='blue', ha='center', va='center', zorder=3)

    # Draw atoms
    for ia, (e, p) in enumerate(zip(sys.enames, sys.apos)):
        c = _ELEM_COLORS.get(e, 'purple')
        sz = 200 if e != 'E' else 120
        ax.scatter(p[ax_i], p[ax_j], c=c, s=sz, zorder=2, edgecolors='black')
        label = f'{ia}:{e}'
        if e == 'E':
            label = 'EP'
        ax.text(p[ax_i], p[ax_j], label, fontsize=7, ha='center', va='center',
                zorder=4, color='white' if c == 'black' else 'black')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(f'{proj[0]} [A]')
    ax.set_ylabel(f'{proj[1]} [A]')


def render_sys_multiview(sys_before, sys_after, savepath,
                         title_before='Before', title_after='After',
                         projections=('xy', 'xz', 'yz')):
    """Save 3×2 grid PNG: rows=projections, cols=before/after.

    Shows the molecule from all three orthogonal projections so out-of-plane
    atoms (e.g. H in NH3) are visible in at least one view.
    """
    n_proj = len(projections)
    fig, axes = plt.subplots(n_proj, 2, figsize=(16, 5 * n_proj), squeeze=False)
    for row, proj in enumerate(projections):
        render_sys(axes[row, 0], sys_before, f'{title_before} ({proj})', proj=proj)
        render_sys(axes[row, 1], sys_after,  f'{title_after} ({proj})',  proj=proj)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [visual] Saved: {savepath}')
