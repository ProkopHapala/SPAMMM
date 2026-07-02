#!/usr/bin/env python3
"""Visual test script: render ASCII art molecules + Kekule solver results to PNG.

Outputs go to debug/visual_kekule/*.png
Run: python tests/topology/visual_kekule.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from spammm.AtomicSystem import AtomicSystem
from spammm import plotUtils as pu
from spammm.topology.KekulePure import KekulePure, make_n_pi, make_pi_mask
from spammm.topology.ascii_art_heterocycle import (
    parse_ascii_art, run_kekule_solver, ASCII_EXAMPLES, mol_bond_types,
    resolve_hbond_pairs, _build_target_valence, jacobi_relax_bond_lengths,
)
from spammm.topology.heterocycle_generator import EXAMPLES as GRID_EXAMPLES, build_atomic_system

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'visual_kekule')
os.makedirs(OUT_DIR, exist_ok=True)


def _bond_midpoints(atoms):
    bonds = np.asarray(atoms.bonds)
    p0 = atoms.apos[bonds[:, 0]]
    p1 = atoms.apos[bonds[:, 1]]
    return 0.5 * (p0 + p1)


def plot_molecule(atoms, title=None, bond_orders=None, n_pi=None, fname=None, sz=50.):
    """Plot a single AtomicSystem with optional bond order coloring. Save as PNG."""
    fig, ax = plt.subplots(figsize=(8, 8))
    pu.plotSystem(atoms, axes=(0, 1), bBonds=True, bLabels=True, sz=sz)

    if bond_orders is not None and atoms.bonds is not None:
        bo = np.asarray(bond_orders)
        total = 1.0 + bo
        lws = np.ones(len(bo), dtype=float)
        colors = np.array(['k'] * len(bo), dtype=object)
        aromatic = np.abs(total - 1.5) < 0.15
        double = total > 1.7
        lws[aromatic] = 2.5
        colors[aromatic] = 'green'
        lws[double] = 3.5
        pu.plotBonds(links=atoms.bonds, ps=atoms.apos, lws=lws, colors=colors, axes=(0, 1))

        mids = _bond_midpoints(atoms)[:, [0, 1]]
        for (x, y), b in zip(mids, bo):
            ax.text(x, y, f"{b:.2f}", fontsize=8, ha='center', va='center',
                    color='darkblue', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.7))

    if n_pi is not None:
        n_pi = np.asarray(n_pi)
        nlab = min(len(n_pi), len(atoms.apos))
        for i, (x, y) in enumerate(atoms.apos[:nlab, [0, 1]]):
            ax.text(x, y, f" {int(n_pi[i])}", fontsize=8, ha='left', va='top',
                    color='red', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='yellow',
                              edgecolor='none', alpha=0.6))

    hb = getattr(atoms, 'hbonds_ascii', None)
    if hb:
        for ih, ia in hb:
            p0 = atoms.apos[ih]
            p1 = atoms.apos[ia]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linestyle='--',
                    color=(0.8, 0.2, 0.8), linewidth=1.2, alpha=0.7, zorder=3)

    if title:
        ax.set_title(title, fontsize=14)
    ax.text(0.02, 0.98, f"N = {atoms.natoms}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', color='black')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
    plt.close(fig)


def plot_kekule_phases_png(atoms, k, bo_raw=None, bo_snap=None, title=None,
                           fname=None, sz=50.):
    """Plot raw and snapped Kekule bond orders side by side. Save as PNG."""
    if bo_raw is None:
        bo_raw = k.pi_bond_orders()
    if bo_snap is None:
        bo_snap = k.snap().copy()
    n_pi = getattr(k, 'n_pi', None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for ax, bo, subtitle in [(ax1, bo_raw, 'raw'), (ax2, bo_snap, 'snapped')]:
        plt.sca(ax)
        pu.plotSystem(atoms, axes=(0, 1), bBonds=True, bLabels=True, sz=sz)
        total = 1.0 + bo
        lws = np.ones(len(bo), dtype=float)
        colors = np.array(['k'] * len(bo), dtype=object)
        aromatic = np.abs(total - 1.5) < 0.15
        double = total > 1.7
        lws[aromatic] = 2.5
        colors[aromatic] = 'green'
        lws[double] = 3.5
        pu.plotBonds(links=atoms.bonds, ps=atoms.apos, lws=lws, colors=colors, axes=(0, 1))

        if n_pi is not None:
            n_pi_arr = np.asarray(n_pi)
            nlab = min(len(n_pi_arr), len(atoms.apos))
            for i, (x, y) in enumerate(atoms.apos[:nlab, [0, 1]]):
                ax.text(x, y, f" {int(n_pi_arr[i])}", fontsize=8, ha='left', va='top',
                        color='red', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='yellow',
                                  edgecolor='none', alpha=0.6))

        mids = _bond_midpoints(atoms)[:, [0, 1]]
        for (x, y), b in zip(mids, bo):
            ax.text(x, y, f"{b:.2f}", fontsize=8, ha='center', va='center',
                    color='darkblue', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.7))

        hb = getattr(atoms, 'hbonds_ascii', None)
        if hb:
            for ih, ia in hb:
                p0 = atoms.apos[ih]
                p1 = atoms.apos[ia]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linestyle='--',
                        color=(0.8, 0.2, 0.8), linewidth=1.2, alpha=0.7, zorder=3)

        ax.set_title(f'phase: {subtitle}', fontsize=12)
        ax.set_aspect('equal')
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Render all ASCII art examples (molecule only, no Kekule)
# ---------------------------------------------------------------------------
print("\n=== ASCII art examples: molecule plots ===")
for name, art in sorted(ASCII_EXAMPLES.items()):
    try:
        atoms = parse_ascii_art(art)
        atoms.neighs()
        n_pi = make_n_pi(atoms)
        # Add hydrogens for visualization
        tv = _build_target_valence(atoms, n_pi)
        atoms.add_capping_h_sp2(target_valence=tv)
        atoms.neighs()
        resolve_hbond_pairs(atoms)
        fname = os.path.join(OUT_DIR, f'ascii_{name}.png')
        plot_molecule(atoms, title=f'ASCII: {name}', n_pi=n_pi, fname=fname)
    except Exception as e:
        print(f"  SKIP {name}: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# 2. Render Kekule solver phases for key examples
# ---------------------------------------------------------------------------
print("\n=== Kekule solver phases ===")
for name in ['naphthalene', 'naphthalene2', 'pyridine', 'purin', 'cytosin', 'uracil', 'guanin', 'biphenyl', 'phenanthrene']:
    if name not in ASCII_EXAMPLES:
        continue
    try:
        art = ASCII_EXAMPLES[name]
        atoms = parse_ascii_art(art)
        atoms.neighs()

        # Run solver on heavy-atom graph (before H-capping)
        r = run_kekule_solver(atoms, Kval=50.0, Kloc=5.0, Karo=0.5, allow_aromatic=True)
        if r['err'] is not None:
            print(f"  SKIP {name}: solver error: {r['err']}")
            continue

        k = r['k']
        bo_raw = r['bo_raw']
        bo_snap = r['bo_snap']

        # Plot Kekule phases (heavy atoms only, before H-capping)
        fname = os.path.join(OUT_DIR, f'kekule_{name}.png')
        plot_kekule_phases_png(atoms, k, bo_raw=bo_raw, bo_snap=bo_snap,
                               title=f'Kekule: {name}', fname=fname)
        rep = r['report']
        print(f"  {name}: single={rep['single']} aromatic={rep['aromatic']} double={rep['double']} max_err={rep['max_err']:.2e}")

        # Also plot H-capped version (molecule only, no bond orders)
        n_pi0 = make_n_pi(atoms)
        tv = _build_target_valence(atoms, n_pi0)
        atoms.add_capping_h_sp2(target_valence=tv)
        atoms.neighs()
        resolve_hbond_pairs(atoms)
        fname_h = os.path.join(OUT_DIR, f'ascii_{name}_withH.png')
        plot_molecule(atoms, title=f'{name} (with H)', n_pi=n_pi0, fname=fname_h)
    except Exception as e:
        print(f"  SKIP {name}: {e}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# 3. Benzene Kekule: aromatic vs discrete comparison
# ---------------------------------------------------------------------------
print("\n=== Benzene: aromatic vs discrete ===")
def _make_benzene():
    a = 1.42
    angles = np.arange(6) * (np.pi / 3.0)
    apos = np.column_stack([a * np.cos(angles), a * np.sin(angles), np.zeros(6)])
    bonds = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]], dtype=np.int32)
    sys = AtomicSystem(apos=apos, enames=['C']*6)
    sys.bonds = bonds
    sys.natoms = 6
    return sys

benz = _make_benzene()

# Aromatic
k_aro = KekulePure(benz, n_pi=np.ones(6), Kval=50.0, Karo=0.5, allow_aromatic=True)
k_aro.solve_quadratic(Kloc=0.0)
bo_aro = k_aro.pi_bond_orders()
fname = os.path.join(OUT_DIR, 'benzene_aromatic.png')
plot_molecule(benz, title='Benzene (aromatic)', bond_orders=bo_aro, n_pi=np.ones(6), fname=fname)

# Discrete
k_disc = KekulePure(benz, n_pi=np.ones(6), Kval=50.0, Karo=0.5, allow_aromatic=False)
k_disc.solve_quadratic(Kloc=0.0)
k_disc.Kloc = 5.0
k_disc.solve_snap(niter=50)
bo_disc = k_disc.snap()
fname = os.path.join(OUT_DIR, 'benzene_discrete.png')
plot_molecule(benz, title='Benzene (discrete)', bond_orders=bo_disc, n_pi=np.ones(6), fname=fname)

# Side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
for ax, bo, title in [(ax1, bo_aro, 'aromatic'), (ax2, bo_disc, 'discrete')]:
    plt.sca(ax)
    pu.plotSystem(benz, axes=(0, 1), bBonds=True, bLabels=True, sz=50)
    total = 1.0 + bo
    lws = np.ones(len(bo), dtype=float)
    colors = np.array(['k'] * len(bo), dtype=object)
    aromatic = np.abs(total - 1.5) < 0.15
    double = total > 1.7
    lws[aromatic] = 2.5
    colors[aromatic] = 'green'
    lws[double] = 3.5
    pu.plotBonds(links=benz.bonds, ps=benz.apos, lws=lws, colors=colors, axes=(0, 1))
    mids = _bond_midpoints(benz)[:, [0, 1]]
    for (x, y), b in zip(mids, bo):
        ax.text(x, y, f"{b:.2f}", fontsize=9, ha='center', va='center',
                color='darkblue', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7))
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')
    ax.axis('off')
fig.suptitle('Benzene: aromatic vs discrete Kekule', fontsize=14)
fig.tight_layout()
fname = os.path.join(OUT_DIR, 'benzene_comparison.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
print(f"Saved: {fname}")
plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Grid-based heterocycle examples
# ---------------------------------------------------------------------------
print("\n=== Grid heterocycle examples ===")
for name in sorted(GRID_EXAMPLES.keys()):
    try:
        atoms = build_atomic_system(GRID_EXAMPLES[name])
        n_pi = make_n_pi(atoms)
        fname = os.path.join(OUT_DIR, f'grid_{name}.png')
        plot_molecule(atoms, title=f'Grid: {name}', n_pi=n_pi, fname=fname)
    except Exception as e:
        print(f"  SKIP {name}: {e}")
        import traceback; traceback.print_exc()

print(f"\nAll images saved to: {os.path.abspath(OUT_DIR)}")
