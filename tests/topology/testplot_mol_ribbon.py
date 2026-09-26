#!/usr/bin/env python3
"""testplot_mol_ribbon.py — molecule-bridged ribbon junction cells (periodic x+y).

One r3 ribbon (width_chains=8 = 3 ring rows, ncells=4 = '4u' along x) + one
molecule per cell.  The molecule's bottom end H-bonds the ribbon top edge
(internal junction) and its top end the bottom edge of the +y image (boundary
junction) — 2 junctions per molecule; the stack ...|rib|gap|mol|gap|rib'|...
is the molecule analog of build_self_junction_cell.

Corner states per molecule (state = [internal, boundary] junction; 'A' = H on
the molecule end -> mol donor X-H ... N-ribbon, 'B' = H on the ribbon junction
site -> mol X ... H-N-ribbon):
    AA: mol XH2 (both ends donor)  — ribbon edges all-N
    BB: mol X   (both ends bare)   — ribbon 'NH' at the two junction sites
    AB: mol XH  (bottom donates / top accepts)
    BA: mol XHb (bottom accepts / top donates)  — mirror image of AB -> parity check
BB vs AA is the H-transfer comparison of doc/ERC_private/mol_ribbon_Htransfer.md.

Gaps: each junction gets its own optimal heavy-atom D...A distance from
ribbon_pbc.D_DA_OPT ({N,N}->2.90, {N,O}->2.80, {O,O}->2.75 A) and Ly is
assembled so both junctions hold it — override globally with --dda.

Outputs: debug/mol_ribbon/<mol>/<state>/cell.xyz (lvs in comment) + cell.png
(2x2 tiling), overview build_check.png, results.json.

--relax runs a DFTB PBC geometry optimization per (mol, state, width): junction
scaffold (all edge-site + mol-tip N/O atoms) pinned via junction_site_atoms so
every corner sees the same d_DA; workdirs debug/mol_ribbon/<mol>/<state>/dftb_w<W>/.
With the default 3ob SK set the DFTB3 block (ThirdOrderFull + HubbardDerivs +
DampXH) is patched into dftb_in.hsd via testplot_muH._patch_hsd.  Energies +
relaxed junction distances land in results.json; LL/RR/LR/RL label = proton
host per junction (L = moLecule, R = Ribbon): LL=AA, RR=BB, LR=AB, RL=BA.

Usage: python tests/topology/testplot_mol_ribbon.py [--mols st1x1N,st3x1O]
       [--states AA,BB,AB,BA] [--dda 2.9] [--site 2] [--shiftx 0.0]
       [--alt --tilt 60] [--relax --widths 8,12 --nk 4,2,1 --sk 3ob-1-1]
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm.topology.ribbon_pbc import (build_mol_ribbon_cell, check_degrees, relax_cell,
                                       junction_site_atoms, junction_geometry_report, save_xyz_lvs)
from spammm.topology.hbond_utils import hbond_positions
from spammm.topology.ascii_art_heterocycle import make_strip_mol_art
from spammm.plotUtils import plot_ribbon_junction_cell

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'mol_ribbon')

# molecule menu: symmetric tip-ended strip molecules (single apex at each end)
# — the 'st<L>x<T><kind>' family already used for the PBC chains in
# debug/test_bo_path_pbc.  L = odd ring-rows along the spine, T = thickness;
# symmetric = T in {1,3} (T even is the asymmetric zigzag).  N tips = N-H...N
# junctions (st1x1N = pyrazine), O tips = exocyclic O-H...N (st1x1O =
# hydroquinone).
# O-tip (quinone/diol) family first, then N-tip (pyrazine) family -> size trend per family.
MOLS = {'st1x1O': 'hydroquinone',        'st3x1O': 'biphenol rod',        'st5x1O': 'terphenyl-diol rod',
        'st3x3O': 'O-tip strip',         'st5x3O': 'O-tip long strip',
        'st1x1N': 'pyrazine',            'st3x1N': 'bipyridine rod',      'st5x1N': 'terpyridine rod',
        'st3x3N': 'N-tip strip',         'st5x3N': 'N-tip long strip'}

# corner state -> (etop, ebot); don/acc = donor/acceptor tip char per kind.
# 'A' = H on the molecule end (mol donates); internal junction = bottom end.
_TIP = {'N': ('n', 'N'), 'O': ('o', 'O')}
_STATE_TIPS = {'AA': (0, 0), 'BB': (1, 1), 'AB': (1, 0), 'BA': (0, 1)}


def state_art(mol, st):
    """ASCII art for (molecule, corner state); mol = 'st<L>x<T><N|O>'."""
    import re
    m = re.fullmatch(r'st(\d+)x(\d+)([NO])', mol)
    assert m, f'molecule name must be st<L>x<T><N|O>, got {mol!r}'
    L, T = int(m.group(1)), int(m.group(2))
    don, acc = _TIP[m.group(3)]
    it, ib = _STATE_TIPS[st]
    etop, ebot = (don, acc)[it], (don, acc)[ib]
    return make_strip_mol_art(L, T, etop=etop, ebot=ebot)


def _draw_cell_panel(ax, atoms, hbonds, lvs, sz=26.):
    """Single-cell panel in the canonical PBC-cell style: thin grey skeleton
    (seam bonds skipped), element-colored atom dots, junction D-H solid green +
    H...A dashed magenta, magenta cell parallelogram (lvs[1] may be tilted)."""
    from matplotlib.patches import Polygon
    from spammm import elements
    apos = np.asarray(atoms.apos)
    seam = getattr(atoms, 'seam', np.zeros(len(atoms.bonds), bool))
    for (i, j), sm in zip(atoms.bonds, seam):
        if sm:
            continue
        ax.plot([apos[i, 0], apos[j, 0]], [apos[i, 1], apos[j, 1]], '-', c='0.55', lw=1.0, zorder=1)
    enames = [e.split('_')[0] for e in atoms.enames]
    ax.scatter(apos[:, 0], apos[:, 1], c=[elements.ELEMENT_DICT[e][8] for e in enames],
               s=[elements.ELEMENT_DICT[e][6] * sz for e in enames], zorder=3, linewidths=0)
    for hb in hbonds:
        pD, pH, pA = hbond_positions(apos, hb, lvs)
        ax.plot([pD[0], pH[0]], [pD[1], pH[1]], 'g-', lw=1.2, zorder=5)
        ax.plot([pH[0], pA[0]], [pH[1], pA[1]], 'm--', lw=1.0, zorder=5)
    c0, c1 = lvs[0, :2], lvs[1, :2]
    ax.add_patch(Polygon(np.array([[0.0, 0.0], c0, c0 + c1, c1]), fill=False, edgecolor='magenta', lw=1.2))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.margins(0.08)


def fig_overview(panels, mols, states, fname):
    """Grid (rows = molecules, cols = corner states) of single cells."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grid = {(p[0], p[1]): p[2:] for p in panels}
    fig, axs = plt.subplots(len(mols), len(states), figsize=(3.4 * len(states), 2.6 * len(mols)), squeeze=False)
    for r, m in enumerate(mols):
        for c, s in enumerate(states):
            ax = axs[r, c]
            if (m, s) not in grid:
                ax.axis('off')
                continue
            atoms, lvs, hbonds = grid[(m, s)]
            _draw_cell_panel(ax, atoms, hbonds, lvs)
            if r == 0:
                ax.set_title(s, fontsize=10)
            if c == 0:
                ax.annotate(m, (-0.02, 0.5), xycoords='axes fraction', rotation=90,
                            va='center', ha='right', fontsize=10, weight='bold')
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


_LR = {'AA': 'LL', 'BB': 'RR', 'AB': 'LR', 'BA': 'RL'}   # proton host per junction: L = moLecule, R = Ribbon


def _load_xyz_apos(path):
    """Read enames + apos from an save_xyz_lvs-style xyz file."""
    lines = open(path).read().split('\n')
    n = int(lines[0].strip())
    rows = [l.split() for l in lines[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])


def fig_dftb_summary(results, mols, width, fname, ncells=4, alt_CH=False, tilt_deg=0.0):
    """Per-width DFTB summary figure, 3 rows: relaxed LL geometry (top),
    corner energies vs LL (middle), relaxed RR geometry (bottom)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = results.get('dftb', {})
    colors = {'AA': 'tab:blue', 'BB': 'tab:red', 'AB': 'tab:green', 'BA': 'tab:purple'}
    marks = {'AA': 'o', 'BB': 's', 'AB': '^', 'BA': 'v'}
    n = len(mols)
    fig = plt.figure(figsize=(1.7 * n + 1.5, 12.5))
    gs = fig.add_gridspec(3, n, height_ratios=[3.4, 1.5, 3.4], hspace=0.06, wspace=0.05)
    ax_e = fig.add_subplot(gs[1, :])
    for i, mol in enumerate(mols):
        for r, st in ((0, 'AA'), (2, 'BB')):
            ax = fig.add_subplot(gs[r, i])
            art = state_art(mol, st)
            atoms, lvs, hbonds = build_mol_ribbon_cell(mol_art=art, width_chains=width, ncells=ncells, label=f'{mol}/{st}', alt_CH=alt_CH, tilt_deg=tilt_deg)
            xyz = os.path.join(OUTDIR, mol, st, f'dftb_w{width}', 'relaxed.xyz')
            if os.path.exists(xyz):
                _, atoms.apos = _load_xyz_apos(xyz)
            _draw_cell_panel(ax, atoms, hbonds, lvs, sz=34.)
            ax.set_title(f'{mol} {_LR[st]}' if r == 0 else _LR[st], fontsize=7)
    xs = np.arange(n)
    Eall = {st: np.array([d.get(f'{mol}_{st}_w{width}', {}).get('E_ev', np.nan) or np.nan for mol in mols]) for st in ('AA', 'BB', 'AB', 'BA')}
    for st in ('AA', 'BB', 'AB', 'BA'):
        ax_e.plot(xs, Eall[st] - Eall['AA'], marker=marks[st], color=colors[st], ms=4, lw=0.9, label=_LR[st])
    ax_e.plot(xs, 0.5 * (Eall['BB'] - Eall['AA']), 'k--x', ms=5, lw=0.9, label='(LL+RR)/2')
    for i in range(n):
        E = {st: Eall[st][i] for st in Eall}
        if np.isfinite(list(E.values())).all():
            J = E['BB'] + E['AA'] - E['AB'] - E['BA']
            ax_e.annotate(f'J={J:+.2f}\ndE={0.5 * (E["BB"] - E["AA"]):+.2f}', (i, 0.03), ha='center', va='bottom', fontsize=6.5)
    ax_e.set_xlim(-0.6, n - 0.4)
    ax_e.set_xticks(xs)
    ax_e.set_xticklabels(mols, rotation=45, ha='right', fontsize=8)
    ax_e.axhline(0, c='0.7', lw=0.6)
    ax_e.grid(True, lw=0.4, alpha=0.6)
    ax_e.set_ylabel('E_state - E_LL [eV]')
    ax_e.legend(fontsize=7, loc='lower right', ncols=5, title='proton host (L=mol, R=ribbon)', title_fontsize=7)
    fig.suptitle(f'mol->ribbon corner relaxes, w{width} ({"r3" if width == 8 else "r5" if width == 12 else "?"}) — 3ob-3-1, scaffold-pinned; top=LL relaxed, bottom=RR relaxed', fontsize=11)
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {fname}', flush=True)


def fig_dE_widths(results, mols, widths, fname):
    """Transfer energy dE = E_RR - E_LL vs molecule size — one line per
    (ribbon width, tip) -> 4 lines r3-N/r3-O/r5-N/r5-O."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = results.get('dftb', {})
    sizes = list(dict.fromkeys(m[:-1] for m in mols))           # ordered unique st<L>x<T>
    xi = {s: i for i, s in enumerate(sizes)}
    rname = {8: 'r3', 12: 'r5'}
    col = {'N': 'tab:blue', 'O': 'tab:red'}
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for w in widths:
        for tip in ('N', 'O'):
            pts = [(xi[m[:-1]], d[f'{m}_BB_w{w}']['E_ev'] - d[f'{m}_AA_w{w}']['E_ev'])
                   for m in mols if m.endswith(tip) and d.get(f'{m}_BB_w{w}', {}).get('E_ev') is not None and d.get(f'{m}_AA_w{w}', {}).get('E_ev') is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker='o', ms=4, lw=0.9, ls='-' if w == widths[0] else '--', color=col[tip], label=f'{rname.get(w, f"w{w}")}-{tip}')
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes, fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.6)
    ax.axhline(0, c='0.7', lw=0.6)
    ax.set_ylabel('E_RR - E_LL  [eV / cell = 2H]')
    ax.set_xlabel('molecule (size along spine x thickness)')
    ax.legend(fontsize=8)
    fig.suptitle('mol->ribbon H-transfer energy vs molecule size (3ob-3-1)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {fname}', flush=True)


def print_summary(results, mols, states, widths):
    d = results.get('dftb', {})
    print('\n=== DFTB corner energies (L=H on moLecule, R=H on Ribbon) ===', flush=True)
    for mol in mols:
        for w in widths:
            Es = {st: d.get(f'{mol}_{st}_w{w}', {}).get('E_ev') for st in states}
            line = '  '.join(f'{_LR[st]}({st})={E:.4f}' if E is not None else f'{_LR[st]}=nan' for st, E in Es.items())
            if all(E is not None for E in Es.values()) and len(states) == 4:
                J = Es['BB'] + Es['AA'] - Es['AB'] - Es['BA']
                dE = 0.5 * (Es['BB'] - Es['AA'])
                line += f'   |  J={J:+.3f} eV  dE_transfer/H={dE:+.3f} eV'
            print(f'{mol} w{w}:  {line} eV', flush=True)


def bake_gpaw(mols, states, widths, bundle_root, ncells=4, nk=(7, 2, 1), alt_CH=False, tilt_deg=0.0, relax=True):
    """Bake a GPAW lcao/dzp/PBE job bundle per (mol,state,width) in the
    cluster_scan_*_gpaw convention: geom.xyz (extxyz, pbc T T F), job.py
    (BFGS relax with the same junction-scaffold pins as DFTB, then energy +
    hs.npz/all.gpw export), gpaw_hs_export.py, jobs.txt, run_all.sh, submit.pbs.
    Input geometry = the DFTB-relaxed cell when available (dftb_w<W>/relaxed.xyz),
    else the as-built cell."""
    import shutil
    from ase import Atoms
    from ase.io import write as ase_write
    JOB = '''#!/usr/bin/env python3
"""GPAW LCAO/dzp/PBE on the DFTB-relaxed mol_ribbon cell (baked by
testplot_mol_ribbon.py --bake-gpaw).  Same pinned junction scaffold as the
DFTB relax.  Run: python job.py  (or mpirun -np N gpaw python job.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read, write
from ase.constraints import FixCartesian
from gpaw import GPAW
from gpaw_hs_export import export_hs

RELAX = {relax}      # relax with junction D/A atoms pinned along y only (as in DFTB)
NK = {nk}            # k-mesh incl. Gamma (odd grid along x; y flat-band -> 1 pt)
PINS = {pins}        # 0-based indices in geom.xyz order; constrained along y only

atoms = read('geom.xyz')                     # cell + pbc baked into the extxyz header
atoms.calc = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts={{'size': NK, 'gamma': True}},
                  txt='gpaw.out', symmetry='off')
if RELAX:
    from ase.optimize import BFGS
    if PINS:
        atoms.set_constraint([FixCartesian(i, mask=[0, 1, 0]) for i in PINS])
    BFGS(atoms, trajectory='relax.traj', logfile='relax.log').run(fmax=0.05)
    write('relaxed.xyz', atoms)
E = atoms.get_potential_energy()
with open('energy.txt', 'w') as f:
    f.write(f'{{E:.8f}}\\n')
export_hs(atoms.calc, atoms, 'hs.npz')
atoms.calc.write('all.gpw', mode='all')
print(f'E = {{E:.6f}} eV')
'''
    RUN_ALL = '''#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")" && pwd)
while IFS= read -r d; do
  [ -z "$d" ] && continue
  [ -f "$ROOT/$d/energy.txt" ] && continue
  echo "[$(date +%H:%M:%S)] $d"
  (cd "$ROOT/$d" && python job.py > stdout.txt 2> stderr.txt)
done < "$ROOT/jobs.txt"
echo ALL DONE
'''
    SUBMIT = '''#!/bin/bash
# Metacentrum PBS array job: one task per line of jobs.txt
# Adjust module/venv activation to your setup before qsub.
#PBS -N molribbon_hs
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=24:00:00
#PBS -J 0-%d
cd "$PBS_O_WORKDIR"
# module load gpaw  # or: source /path/to/venv/bin/activate
d=$(sed -n "$((PBS_ARRAY_INDEX + 1))p" jobs.txt)
[ -f "$d/energy.txt" ] && exit 0
cd "$d" && OMP_NUM_THREADS=$PBS_NCPUS python job.py > stdout.txt 2> stderr.txt
'''
    hs_src = os.path.join(os.path.dirname(__file__), '..', '..', 'spammm', 'quantum', 'gpaw_hs_export.py')
    jobs = []
    for mol in mols:
        for st in states:
            for w in widths:
                atoms, lvs, hbonds = build_mol_ribbon_cell(mol_art=state_art(mol, st), width_chains=w,
                                                         ncells=ncells, label=f'{mol}/{st}', alt_CH=alt_CH, tilt_deg=tilt_deg)
                xyz = os.path.join(OUTDIR, mol, st, f'dftb_w{w}', 'relaxed.xyz')
                if os.path.exists(xyz):
                    _, atoms.apos = _load_xyz_apos(xyz)              # start GPAW from the DFTB-relaxed geometry
                pins = sorted({h.donor_idx for h in hbonds} | {h.acceptor_idx for h in hbonds})
                name = f'{mol}_{st}_w{w}'
                jd = os.path.join(bundle_root, name)
                os.makedirs(jd, exist_ok=True)
                ase_write(os.path.join(jd, 'geom.xyz'),
                          Atoms(symbols=list(atoms.enames), positions=atoms.apos, cell=lvs, pbc=[True, True, False]))
                with open(os.path.join(jd, 'job.py'), 'w') as f:
                    f.write(JOB.format(relax=relax, nk=tuple(nk), pins=pins))
                shutil.copy(hs_src, os.path.join(jd, 'gpaw_hs_export.py'))
                jobs.append(name)
                print(f'baked {jd}  pins={len(pins)}', flush=True)
    with open(os.path.join(bundle_root, 'jobs.txt'), 'w') as f:
        f.write('\n'.join(jobs) + '\n')
    with open(os.path.join(bundle_root, 'run_all.sh'), 'w') as f:
        f.write(RUN_ALL)
    os.chmod(os.path.join(bundle_root, 'run_all.sh'), 0o755)
    with open(os.path.join(bundle_root, 'submit.pbs'), 'w') as f:
        f.write(SUBMIT % (len(jobs) - 1))
    print(f'REVIEW: {bundle_root}/  ({len(jobs)} jobs)', flush=True)


def _dftb3_patch():
    """patch_hsd callback for run_pbc: inject the 3ob DFTB3 block via testplot_muH._patch_hsd."""
    from tests.topology.testplot_muH import _patch_hsd
    return lambda hsd, enames: _patch_hsd(hsd, enames, dftb3=True)


def run_dftb(mol, st, width, atoms, lvs, hbonds, wd, nk, k_shift, sk_set, results):
    """PBC DFTB relax of one (mol,state,width) cell; junction D/A atoms y-pinned.

    Minimal constraint: only the heavy atoms forming each junction (donor +
    acceptor) are fixed along y (the stack direction, so d_DA is preserved)
    while x/z stay free — prevents the bare-N SCC meltdown and H hop-back
    without biasing in-plane stress."""
    from spammm.quantum import DFTB_utils as DU
    name = f'{mol}_{st}_w{width}'
    sk = sk_set or DU.DEFAULT_SK_SET
    patch = _dftb3_patch() if sk.startswith('3ob') else None
    jat = sorted({h.donor_idx for h in hbonds} | {h.acceptor_idx for h in hbonds})
    print(f'  [{name}] relax: sk={sk}  nk={nk}+{k_shift}  y-pinned={len(jat)} junction atoms {jat} -> {wd}', flush=True)
    E, apos_r = relax_cell(atoms, lvs, fixed_atoms=None, cart_constraints=[(jat, (0., 1., 0.))],
                           nk=nk, k_shift=k_shift, Temperature=300,
                           Mixer='DIIS { Generations = 8 }', workdir=wd, sk_set=sk, patch_hsd=patch)
    atoms_r = type(atoms)(apos=apos_r, enames=list(atoms.enames))
    atoms_r.bonds = atoms.bonds
    atoms_r.atypes = atoms.atypes
    atoms_r.seam = getattr(atoms, 'seam', None)
    bls = junction_geometry_report(apos_r, hbonds, lvs)
    save_xyz_lvs(os.path.join(wd, 'relaxed.xyz'), atoms_r, lvs, name)
    png = os.path.join(wd, 'relaxed.png')
    plot_ribbon_junction_cell(atoms_r, lvs, hbonds, nx=2, ny=2, sz=80, savepath=png,
                              title=f'{name} ({_LR[st]}): E={E:.4f} eV  relaxed D-H/H..A={"/".join(f"{a:.2f}:{b:.2f}" for a, b in bls)} A')
    print(f'  [{name}] E={E:.4f} eV\nREVIEW: {png}', flush=True)
    results.setdefault('dftb', {})[name] = {'E_ev': float(E), 'junctions_relaxed': [[float(a), float(b)] for a, b in bls], 'sk_set': sk, 'nk': list(nk), 'workdir': os.path.relpath(wd, OUTDIR)}


def main():
    ap = argparse.ArgumentParser(description='molecule-bridged ribbon junction cells')
    ap.add_argument('--mols', default=','.join(MOLS))
    ap.add_argument('--states', default='AA,BB,AB,BA')
    ap.add_argument('--dda', type=float, default=None, help='override junction D...A distance [A] (default: per-pair optimum)')
    ap.add_argument('--ncells', type=int, default=4)
    ap.add_argument('--width', type=int, default=8, help='width_chains (r3 = 8 chains)')
    ap.add_argument('--widths', default=None, help='comma list of width_chains for --relax (r3=8, r5=12; default: --width)')
    ap.add_argument('--site', type=int, default=None, help='junction edge-site index (default: center)')
    ap.add_argument('--shiftx', type=float, default=0.0, help='mol x shift, fraction of Lx')
    ap.add_argument('--no-pbc-y', action='store_true', help='vacuum y instead of the bridging stack')
    ap.add_argument('--alt', action='store_true', help='alternate edge sites N/CH (junction site stays N); outputs go to debug/mol_ribbon_alt/')
    ap.add_argument('--tilt', type=float, default=0.0, help='rotate molecular plane about tip-tip axis [deg] (clears edge C-H)')
    ap.add_argument('--relax', action='store_true', help='run DFTB PBC relax per (mol,state,width)')
    ap.add_argument('--bake-gpaw', metavar='BUNDLE_DIR', default=None, help='bake GPAW lcao/dzp/PBE job bundle (uses DFTB-relaxed geometry when present)')
    ap.add_argument('--gpaw-nk', default='7,1,1', help='GPAW k-mesh, comma list (default: 7,1,1 — odd grid incl. Gamma along x only; y flat-band)')
    ap.add_argument('--gpaw-sp', action='store_true', help='bake single-point GPAW jobs (no relax)')
    ap.add_argument('--summary', action='store_true', help='no builds/runs — print energy table + summary figure from existing results.json')
    ap.add_argument('--nk', default='7,1,1', help='k-points for relax, comma list (default: 7,1,1 Gamma-centered — y is non-bonded, flat bands)')
    ap.add_argument('--kshift', default='0,0,0', help='k-point shift (default: 0,0,0 -> grid includes Gamma)')
    ap.add_argument('--sk', default=None, help='DFTB SK set (default: config DEFAULT_SK_SET = 3ob-3-1)')
    args = ap.parse_args()
    global OUTDIR
    if args.alt:
        OUTDIR = OUTDIR.rstrip('/') + '_alt'
    os.makedirs(OUTDIR, exist_ok=True)

    if args.summary:
        widths_s = [int(w) for w in args.widths.split(',')] if args.widths else [8, 12]
        results_s = json.load(open(os.path.join(OUTDIR, 'results.json')))
        print_summary(results_s, args.mols.split(','), args.states.split(','), widths_s)
        for w in widths_s:
            fig_dftb_summary(results_s, args.mols.split(','), w, os.path.join(OUTDIR, f'dftb_summary_w{w}.png'), ncells=args.ncells, alt_CH=args.alt, tilt_deg=args.tilt)
        fig_dE_widths(results_s, args.mols.split(','), widths_s, os.path.join(OUTDIR, 'dftb_dE_widths.png'))
        return

    mols = args.mols.split(',')
    states = args.states.split(',')
    widths = [int(w) for w in args.widths.split(',')] if args.widths else [args.width]
    nk = tuple(int(k) for k in args.nk.split(','))
    k_shift = tuple(float(k) for k in args.kshift.split(','))
    if args.bake_gpaw:
        bake_gpaw(mols, states, widths, args.bake_gpaw, ncells=args.ncells,
                  nk=tuple(int(k) for k in args.gpaw_nk.split(',')),
                  alt_CH=args.alt, tilt_deg=args.tilt, relax=not args.gpaw_sp)
        return
    results, panels = {}, []
    for mol in mols:
        for st in states:
            art = state_art(mol, st)
            for w in widths:
                atoms, lvs, hbonds = build_mol_ribbon_cell(mol_art=art, width_chains=w,
                                                         ncells=args.ncells, d_DA=args.dda,
                                                         shift_x=args.shiftx, site=args.site,
                                                         pbc_y=not args.no_pbc_y, label=f'{mol}/{st}',
                                                         alt_CH=args.alt, tilt_deg=args.tilt)
                name = f'{mol}_{st}' + (f'_w{w}' if len(widths) > 1 else '')
                nint = sum(1 for h in hbonds if h.a_shift == 0)
                d_da = [float(np.linalg.norm((lambda p: p[2] - p[0])(hbond_positions(atoms.apos, h, lvs)))) for h in hbonds]
                print(f'\n=== {name} ({_LR[st]}): natoms={atoms.natoms}  cell={lvs[0,0]:.2f}x{lvs[1,1]:.2f} A (tilt dx={lvs[1,0]:+.2f})  junctions={len(hbonds)} ({nint} internal + {len(hbonds)-nint} boundary)  d_DA={"/".join(f"{d:.2f}" for d in d_da)} ===', flush=True)
                check_degrees(atoms, name, deg_heavy=(1, 2, 3))
                junction_geometry_report(atoms.apos, hbonds, lvs)
                wd = os.path.join(OUTDIR, mol, st)
                os.makedirs(wd, exist_ok=True)
                if args.relax:
                    run_dftb(mol, st, w, atoms, lvs, hbonds, os.path.join(wd, f'dftb_w{w}'), nk, k_shift, args.sk, results)
                else:
                    xyz = save_xyz_lvs(os.path.join(wd, 'cell.xyz'), atoms, lvs, name)
                    png = os.path.join(wd, 'cell.png')
                    plot_ribbon_junction_cell(atoms, lvs, hbonds, nx=2, ny=2, sz=80, savepath=png,
                                              title=f'{name} (w{w}/4u): {atoms.natoms} atoms/cell, {len(hbonds)} junctions, d_DA={"/".join(f"{d:.2f}" for d in d_da)} A')
                    print(f'  wrote {xyz}\nREVIEW: {png}', flush=True)
                    panels.append((mol, st, atoms, lvs, hbonds))
                results[name] = {'natoms': atoms.natoms, 'lvs': lvs.tolist(), 'd_DA': d_da,
                                 'junctions': [h.to_dict() for h in hbonds]}

    if args.relax:
        print_summary(results, mols, states, widths)
        for w in widths:
            fig_dftb_summary(results, mols, w, os.path.join(OUTDIR, f'dftb_summary_w{w}.png'), ncells=args.ncells, alt_CH=args.alt, tilt_deg=args.tilt)
        fig_dE_widths(results, mols, widths, os.path.join(OUTDIR, 'dftb_dE_widths.png'))
    else:
        png = os.path.join(OUTDIR, 'build_check.png')
        fig_overview(panels, mols, states, png)
        print(f'\nREVIEW: {png}', flush=True)
    with open(os.path.join(OUTDIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1, sort_keys=True)
    print(f'wrote {os.path.join(OUTDIR, "results.json")}', flush=True)


if __name__ == '__main__':
    main()
