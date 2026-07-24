"""L0 (+ optional L1/L2) tests for spammm.topology.smiles."""
import os
import numpy as np
import pytest

from spammm.topology.smiles import (
    parse_smiles, smiles_to_system, graph_counts,
    SMILES_EXAMPLES, SMILES_HEAVY_COUNTS, SMILES_ATOM_COUNTS_WITH_H,
)


@pytest.mark.parametrize('name', list(SMILES_EXAMPLES.keys()))
def test_smiles_atom_counts(name):
    smi = SMILES_EXAMPLES[name]
    g = parse_smiles(smi, engine='pure', add_h=True, embed='2d')
    n, n_heavy, n_bonds, comp = graph_counts(g)
    assert n_heavy == SMILES_HEAVY_COUNTS[name], f'{name}: heavy {n_heavy}'
    assert n == SMILES_ATOM_COUNTS_WITH_H[name], f'{name}: natoms {n} comp={comp}'
    assert n_bonds >= n_heavy - 1
    assert np.all(np.isfinite(np.array([a.pos for a in g.atoms.values() if a.alive])))


@pytest.mark.parametrize('name', list(SMILES_EXAMPLES.keys()))
def test_smiles_heavy_only(name):
    g = parse_smiles(SMILES_EXAMPLES[name], engine='pure', add_h=False, embed='none')
    n, n_heavy, _, _ = graph_counts(g)
    assert n == n_heavy == SMILES_HEAVY_COUNTS[name]


def test_benzene_ring_bonds_aromatic():
    g = parse_smiles('c1ccccc1', add_h=False, engine='pure')
    bonds = [b for b in g.bonds.values() if b.alive]
    assert len(bonds) == 6
    assert all(abs(b.order - 1.5) < 1e-6 for b in bonds)


def test_smiles_to_system_matches_graph():
    sys = smiles_to_system(SMILES_EXAMPLES['naphthalene'], engine='pure')
    assert sys.natoms == SMILES_ATOM_COUNTS_WITH_H['naphthalene']
    assert sys.bonds is not None and len(sys.bonds) >= 19
    assert np.all(np.isfinite(sys.apos))


def test_bad_smiles_fails_loud():
    with pytest.raises(ValueError):
        parse_smiles('c1ccccc', engine='pure')  # unclosed ring
    with pytest.raises(ValueError):
        parse_smiles('', engine='pure')


def test_smiles_panel_visual(visual_output_dir, make_review):
    """L2: grid of recognizable molecules from SMILES (REVIEW PNG)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spammm import elements as el

    names = list(SMILES_EXAMPLES.keys())
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for k, name in enumerate(names):
        ax = axes[k // ncols, k % ncols]
        g = parse_smiles(SMILES_EXAMPLES[name], engine='pure')
        atoms = [a for a in g.atoms.values() if a.alive]
        bonds = [b for b in g.bonds.values() if b.alive and b.a.alive and b.b.alive]
        for b in bonds:
            xs = [b.a.pos[0], b.b.pos[0]]
            ys = [b.a.pos[1], b.b.pos[1]]
            lw = 2.5 if abs(b.order - 1.5) < 0.1 else (3.0 if b.order >= 2 else 1.2)
            color = 'green' if abs(b.order - 1.5) < 0.1 else 'k'
            ax.plot(xs, ys, '-', color=color, lw=lw, zorder=1)
        for a in atoms:
            c = el.getColor(a.ename, bFloat=True)
            s = 40 if a.ename != 'H' else 18
            ax.scatter([a.pos[0]], [a.pos[1]], c=[c], s=s, zorder=2, edgecolors='k', linewidths=0.3)
            if a.ename != 'H':
                ax.text(a.pos[0], a.pos[1], a.ename, fontsize=7, ha='center', va='center', zorder=3)
        n, nh, nb, comp = graph_counts(g)
        ax.set_title(f'{name}\n{SMILES_EXAMPLES[name]}\nN={n} heavy={nh}', fontsize=8)
        ax.set_aspect('equal')
        ax.axis('off')
    for k in range(len(names), nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')
    fig.suptitle('SMILES → AtomicGraph (pure parser)', fontsize=12)
    fig.tight_layout()
    out = os.path.join(visual_output_dir, 'smiles_examples_panel.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)

    rv = make_review('test_smiles_panel')
    rv.out(f'Intent: parse named SMILES examples into recognizable 2D graphs\n')
    for name in names:
        g = parse_smiles(SMILES_EXAMPLES[name], engine='pure')
        n, nh, nb, comp = graph_counts(g)
        rv.out(f'{name}: smiles={SMILES_EXAMPLES[name]} n={n} heavy={nh} bonds={nb} {comp}\n')
        rv.graph_table(g, pos=False)
    rv.out(f'REVIEW: {out}\n')
    print(f'REVIEW: {out}')
