#!/usr/bin/env python3
"""DEBUG: dump + plot the poly8 core centers (atoms + bond midpoints) per molecule.

Usage: python3 tests/SPM/testplot_core_centers.py [azaindol PTCDA pentacene]
Writes debug/testplot_coremesh/<mol>_centers.xyz (bond centers as 'He') and
<mol>_centers.png (atoms = element-colored dots, bond centers = magenta x).
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm import atomicUtils as au
from spammm.surfaces.PICCore import core_centers_atoms_bonds, POLY_CORE_R

MOLS = {'azaindol': 'data/xyz/azaindol.xyz', 'PTCDA': 'data/xyz/PTCDA.xyz',
        'pentacene': 'data/xyz/pentacene.xyz', 'benzene': 'data/xyz/benzene.xyz',
        'C2H4': 'data/xyz/C2H4.xyz'}
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'testplot_coremesh')
os.makedirs(OUT, exist_ok=True)
ECOL = {'H': 'lightgray', 'C': 'k', 'N': 'b', 'O': 'r', 'F': 'g'}
ESZ = {'H': 25, 'C': 60, 'N': 60, 'O': 60, 'F': 60}

mols = sys.argv[1:] or ['azaindol', 'PTCDA', 'pentacene']
fig, axs = plt.subplots(1, len(mols), figsize=(5 * len(mols), 5))
if len(mols) == 1: axs = [axs]
for ax, m in zip(axs, mols):
    apos, _, enames, _, _ = au.load_xyz(MOLS[m])
    apos = np.asarray(apos, float); apos[:, 2] = 0.0
    enames = [e.capitalize() for e in enames]
    centers, is_bond = core_centers_atoms_bonds(apos, dbond=1.6)
    B = centers[is_bond]
    fxyz = os.path.join(OUT, f'{m}_centers.xyz')
    with open(fxyz, 'w') as f:
        f.write(f'{len(centers)}\n{m}: atoms as elements, bond centers as He (poly8 centers, dbond<1.6)\n')
        for i, c in enumerate(centers):
            f.write(f'{"He" if is_bond[i] else enames[np.argmin(np.linalg.norm(apos - c, axis=1))]}  {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n')
    for e in set(enames):
        m_ = np.array([en == e for en in enames]); ax.scatter(apos[m_, 0], apos[m_, 1], s=[ESZ[e]] * m_.sum(), c=ECOL[e], zorder=3, label=e)
    ax.scatter(B[:, 0], B[:, 1], marker='x', c='magenta', s=50, lw=1.5, zorder=4, label='bond ctr')
    iu = np.triu_indices(len(apos), 1); b_ = np.linalg.norm(apos[iu[0]] - apos[iu[1]], axis=1) < 1.6
    for i, j in zip(iu[0][b_], iu[1][b_]): ax.plot(apos[[i, j], 0], apos[[i, j], 1], '0.5', lw=0.8, zorder=2)
    ax.set_aspect('equal'); ax.set_title(f'{m}: {len(apos)} at + {len(B)} bd = {len(centers)} centers'); ax.legend(fontsize=7)
    print(f'{m}: {len(apos)} atoms, {len(B)} bond centers, coeffs {len(centers)}x5={5*len(centers)} ({5*len(centers)*4} B)  REVIEW: {os.path.abspath(fxyz)}', flush=True)
f = os.path.join(OUT, 'centers_all.png'); fig.tight_layout(); fig.savefig(f, dpi=140); plt.close(fig)
print(f'REVIEW: {os.path.abspath(f)}', flush=True)
