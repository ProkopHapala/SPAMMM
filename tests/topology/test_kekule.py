"""Kekule solver — data-driven tests over ASCII art molecules.

Add molecules to KEKULE_MOLECULES; each must parse, pass feasibility, and
localize to a chemically valid discrete Kekule form.

Run:
  pytest tests/topology/test_kekule.py -q
  pytest tests/topology/test_kekule.py --develop -s   # L1 + one PNG (naphthalene)
"""

import os
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spammm.AtomicSystem import AtomicSystem
from spammm import plotUtils as pu
from spammm.topology.ascii_art_heterocycle import parse_ascii_art, ASCII_EXAMPLES
from spammm.topology.KekulePure import (
    make_n_pi, make_pi_mask, run_kekule_solver, mol_bond_types,
    analyze_kekule_feasibility, validate_kekule_solution,
)
from tests.helpers.review import review_trace

pytestmark = pytest.mark.review

# ASCII example names — extend this list when adding molecules
KEKULE_MOLECULES = [
    'naphthalene', 'naphthalene2',
    'phenanthrene', 'phenanthrene2',
    'perylene', 'perylene2',
    'biphenyl', 'biphenylene',
    'fulvalene',
    'purin',
    'cytosin', 'guanin',
    'karbazol', '7azaindol',
    'TAP',
]

KEKULE_VISUAL = 'naphthalene'  # L2 demo molecule for --develop / --visual

_SOLVER_KW = dict(Kval=50.0, Kloc=5.0, Karo=0.5, allow_aromatic=True, sym_break=0.2, seed=42)


def _load_ascii(name):
    assert name in ASCII_EXAMPLES, f'unknown ASCII example {name!r}'
    atoms = parse_ascii_art(ASCII_EXAMPLES[name])
    atoms.neighs()
    return atoms


def _plot_kekule_phases(atoms, n_pi, bo_raw, bo_snap, fname, title=None):
    """Side-by-side delocalized vs localized with pi bond order labels."""
    bonds = np.asarray(atoms.bonds, dtype=np.int32)
    pi = (n_pi[bonds[:, 0]] > 0) & (n_pi[bonds[:, 1]] > 0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    def _panel(ax, bo_full, subtitle):
        plt.sca(ax)
        pu.plotSystem(atoms, axes=(0, 1), bBonds=True, bLabels=True, sz=40)
        if np.any(pi):
            bo = np.asarray(bo_full)[pi]
            total = 1.0 + bo
            lws = np.where(total > 1.7, 3.5, np.where(np.abs(total - 1.5) < 0.15, 2.5, 1.0))
            colors = np.array(['green' if abs(t - 1.5) < 0.15 else 'k' for t in total], dtype=object)
            pu.plotBonds(links=bonds[pi], ps=atoms.apos, lws=lws, colors=colors, axes=(0, 1))
            p0 = atoms.apos[bonds[pi, 0]]
            p1 = atoms.apos[bonds[pi, 1]]
            mids = 0.5 * (p0 + p1)[:, [0, 1]]
            for (x, y), b in zip(mids, bo):
                ax.text(x, y, f'{b:.2f}', fontsize=7, ha='center', va='center', color='darkblue', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.75))
        ax.set_title(subtitle, fontsize=11)
        ax.set_aspect('equal')
        ax.axis('off')

    _panel(axes[0], bo_raw, 'delocalized')
    _panel(axes[1], bo_snap, 'localized')
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fname}', flush=True)


def _plot_localized(atoms, bo_snap, fname, title=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    pu.plotSystem(atoms, axes=(0, 1), bBonds=True, bLabels=True, sz=40)
    bonds = np.asarray(atoms.bonds, dtype=np.int32)
    n_pi = make_n_pi(atoms)
    pi = (n_pi[bonds[:, 0]] > 0) & (n_pi[bonds[:, 1]] > 0)
    if np.any(pi):
        bo = np.asarray(bo_snap)[pi]
        total = 1.0 + bo
        lws = np.where(total > 1.7, 3.5, np.where(np.abs(total - 1.5) < 0.15, 2.5, 1.0))
        pu.plotBonds(links=bonds[pi], ps=atoms.apos, lws=lws, axes=(0, 1))
    if title:
        ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fname}', flush=True)


@pytest.mark.parametrize('name', KEKULE_MOLECULES)
def test_kekule_ascii_valid(name):
    """Parse → feasibility → solve → validate for each ASCII molecule."""
    atoms = _load_ascii(name)
    pre = analyze_kekule_feasibility(atoms)
    assert not pre['impossible'], f'{name}: {pre["reasons"]}'

    r = run_kekule_solver(atoms, **_SOLVER_KW)
    assert r['err'] is None, f'{name}: solver error {r["err"]}'
    assert r['report'].get('valid'), f'{name}: {r["report"].get("validity", {})}'
    assert r['report']['max_err'] < 1e-6

    val = validate_kekule_solution(atoms, r['k'])
    assert val['valid'], f'{name}: {val["issues"]}'

    bt = mol_bond_types(atoms, bo_snap=r['bo_snap'], kekule=True)
    assert bt is not None and np.any(bt >= 2), f'{name}: no double/aromatic bonds in MOL types'


def test_kekule_impossible_odd_pi_sites():
    """Three sp2 sites (odd count) → proven impossible before solving."""
    apos = np.zeros((3, 3))
    bonds = np.array([[0, 1], [1, 2]], dtype=np.int32)
    sys = AtomicSystem(apos=apos, enames=['C', 'C', 'C'])
    sys.bonds = bonds
    sys.natoms = 3
    pre = analyze_kekule_feasibility(sys, n_pi=np.ones(3))
    assert pre['impossible']
    assert any('odd' in msg for msg in pre['reasons'])


class TestMakeNPi:
    def test_uppercase_sp2(self):
        sys = AtomicSystem(apos=np.zeros((3, 3)), enames=['C', 'N', 'O'])
        sys.natoms = 3
        assert np.allclose(make_n_pi(sys), 1.0)

    def test_lowercase_sp3(self):
        sys = AtomicSystem(apos=np.zeros((3, 3)), enames=['c', 'n', 'o'])
        sys.natoms = 3
        assert np.allclose(make_n_pi(sys), 0.0)

    def test_pi_mask(self):
        sys = AtomicSystem(apos=np.zeros((4, 3)), enames=['C', 'N', 'O', 'H'])
        sys.natoms = 4
        assert make_pi_mask(sys).tolist() == [True, True, True, False]


def test_kekule_develop_review(request, visual_output_dir, make_review):
    """L1 + L2 for one representative molecule (--develop / --visual)."""
    review_on = request.config.getoption('--review', default=False) or request.config.getoption('--develop', default=False)
    if visual_output_dir is None and not review_on:
        pytest.skip('review/visual off')
    name = KEKULE_VISUAL
    atoms = _load_ascii(name)
    r = run_kekule_solver(atoms, **_SOLVER_KW)
    val = validate_kekule_solution(atoms, r['k'])
    assert val['valid']

    rv = make_review(f'test_kekule_{name}')
    with review_trace(rv) as out:
        out.out_section('Intent')
        out.out(f'Representative Kekule regression: ASCII {name}')
        out.out_section('Metrics')
        out.out(f'valid={val["valid"]} ring_doubles={val["ring_doubles"]} max_err={r["report"]["max_err"]:.3e}')
        out.out(f'localize_seed={r["report"].get("localize_seed")} trials={r["report"].get("localize_trials")}')
        out.checklist('Feasibility pre-check passed', 'Localized bond orders chemically valid')
    rv.finish()

    if visual_output_dir is not None:
        png = os.path.join(visual_output_dir, f'test_kekule_{name}.png')
        _plot_localized(atoms, r['bo_snap'], png, title=f'{name} (localized)')


@pytest.mark.slow
@pytest.mark.visual
def test_kekule_all_ascii_plots(visual_output_dir):
    """L2: Kekule phase plots for every ASCII_EXAMPLES entry (--visual / --develop)."""
    if visual_output_dir is None:
        pytest.skip('--visual or --develop required')
    names = sorted(ASCII_EXAMPLES.keys())
    print(f'Plotting {len(names)} ASCII examples → {visual_output_dir}', flush=True)
    ok, skip = [], []
    for name in names:
        try:
            atoms = _load_ascii(name)
            pre = analyze_kekule_feasibility(atoms)
            if pre['impossible']:
                skip.append((name, 'impossible: ' + '; '.join(pre['reasons'][:1])))
                continue
            r = run_kekule_solver(atoms, **_SOLVER_KW)
            if r['err'] is not None:
                skip.append((name, f'solver: {r["err"]}'))
                continue
            n_pi = make_n_pi(atoms)
            val = validate_kekule_solution(atoms, r['k'])
            tag = 'valid' if val['valid'] else 'INVALID'
            fname = os.path.join(visual_output_dir, f'kekule_{name}.png')
            _plot_kekule_phases(atoms, n_pi, r['bo_raw'], r['bo_snap'], fname,
                                title=f'{name} ({tag}) seed={r["report"].get("localize_seed")}')
            ok.append(name)
        except Exception as e:
            skip.append((name, str(e)))
    print(f'OK: {len(ok)}  skipped: {len(skip)}', flush=True)
    for name, why in skip:
        print(f'  SKIP {name}: {why}', flush=True)
    assert len(ok) > 0, 'no molecules plotted'
