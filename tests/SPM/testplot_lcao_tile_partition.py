#!/usr/bin/env python3
"""Illustrate LCAO tiled density projection: 8³ voxel blocks vs atom cutoff spheres.

Mirrors ``GridProjector.build_tasks`` / ``count_atoms_per_block`` (sphere–AABB):
each non-empty block becomes one GPU task; denmap pair count = na*(na+1)/2
(diagonal sparse path, nj < 0).

Plots only the block layer that contains the molecular plane (z ≈ 0 for pentacene).
Writes SVG under ``debug/lcao_tile_partition/``.

Usage:
  python tests/SPM/testplot_lcao_tile_partition.py
  python tests/SPM/testplot_lcao_tile_partition.py --xyz data/xyz/pentacene.xyz --step 0.15
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

BOHR2ANG = 0.5291772109


def check_overlap_sphere_aabb(center, radius, box_min, box_max):
    """Same test as GridProjector.check_overlap_sphere_aabb."""
    closest_p = np.clip(center, box_min, box_max)
    return float(np.sum((center - closest_p) ** 2)) < (radius ** 2)


def partition_blocks(atom_pos, atom_rcut, grid_spec, block_res=8):
    """CPU replica of GridProjector.build_tasks geometry (all blocks).

    Returns
    -------
    n_blocks : (nxb, nyb, nzb)
    na_map : int32[nxb, nyb, nzb] — atoms overlapping each block
    atoms_map : list[list[list[list[int]]]] — atom indices per block
    """
    nx, ny, nz = [int(x) for x in grid_spec['ngrid'][:3]]
    n_blocks = (
        (nx + block_res - 1) // block_res,
        (ny + block_res - 1) // block_res,
        (nz + block_res - 1) // block_res,
    )
    origin = np.asarray(grid_spec['origin'][:3], dtype=np.float64)
    dA = np.asarray(grid_spec['dA'][:3], dtype=np.float64)
    dB = np.asarray(grid_spec['dB'][:3], dtype=np.float64)
    dC = np.asarray(grid_spec['dC'][:3], dtype=np.float64)
    bx = block_res * dA[0]
    by = block_res * dB[1]
    bz = block_res * dC[2]

    na_map = np.zeros(n_blocks, dtype=np.int32)
    atoms_map = [[[[] for _ in range(n_blocks[2])] for _ in range(n_blocks[1])] for _ in range(n_blocks[0])]
    natoms = len(atom_pos)
    for fix in range(n_blocks[0]):
        for fiy in range(n_blocks[1]):
            for fiz in range(n_blocks[2]):
                block_min = origin + np.array([fix * bx, fiy * by, fiz * bz])
                block_max = block_min + np.array([bx, by, bz])
                hit = []
                for ia in range(natoms):
                    if check_overlap_sphere_aabb(atom_pos[ia], atom_rcut[ia], block_min, block_max):
                        hit.append(ia)
                na_map[fix, fiy, fiz] = len(hit)
                atoms_map[fix][fiy][fiz] = hit
    return n_blocks, na_map, atoms_map, (bx, by, bz), origin


def rcut_from_wfc(enames, wfc_path, fallback_bohr=6.0):
    """Per-atom Rcut [Å] = max orbital Cutoff in WFC (Bohr → Å)."""
    from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd
    basis = parse_wfc_hsd(wfc_path)
    by_z = {}
    for name, sp in basis.items():
        z = int(sp['AtomicNumber'])
        rc_b = max(float(orb['Cutoff']) for orb in sp['orbitals'])
        by_z[z] = rc_b * BOHR2ANG
    zmap = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
    default = fallback_bohr * BOHR2ANG
    return np.array([by_z.get(zmap.get(e, 6), default) for e in enames], dtype=np.float64)


def plot_plane_partition(atom_pos, enames, atom_rcut, grid_spec, block_res, out_path,
                         title_extra='', bond_rcut=1.8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.cm import ScalarMappable
    from spammm.surfaces.FoldedRigid import find_bonds

    n_blocks, na_map, atoms_map, (bx, by, bz), origin = partition_blocks(
        atom_pos, atom_rcut, grid_spec, block_res=block_res)

    z_mol = float(np.median(atom_pos[:, 2]))
    # Block layer whose [zmin,zmax) contains the molecular plane
    fiz = int(np.floor((z_mol - origin[2]) / bz))
    fiz = int(np.clip(fiz, 0, n_blocks[2] - 1))
    z0 = origin[2] + fiz * bz
    z1 = z0 + bz

    na_xy = na_map[:, :, fiz]
    n_pairs = na_xy * (na_xy + 1) // 2  # denmap pair lookups (diagonal nj<0)

    nxb, nyb = n_blocks[0], n_blocks[1]
    print(f'REVIEW: plane z={z_mol:.3f} Å → block layer fiz={fiz}  z∈[{z0:.3f},{z1:.3f}) Å')
    print(f'  n_blocks XY = {nxb}×{nyb}  block_res={block_res}  box=({bx:.3f}×{by:.3f}×{bz:.3f}) Å')
    print(f'  non-empty blocks: {int((na_xy > 0).sum())} / {nxb * nyb}')
    print(f'  na  : min={int(na_xy[na_xy > 0].min()) if (na_xy > 0).any() else 0}  '
          f'max={int(na_xy.max())}  sum={int(na_xy.sum())}')
    print(f'  pairs (denmap): min={int(n_pairs[n_pairs > 0].min()) if (n_pairs > 0).any() else 0}  '
          f'max={int(n_pairs.max())}  sum={int(n_pairs.sum())}')

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    vmax = max(int(n_pairs.max()), 1)
    cmap = plt.cm.YlOrRd
    norm = Normalize(vmin=0, vmax=vmax)

    for fix in range(nxb):
        for fiy in range(nyb):
            x0 = origin[0] + fix * bx
            y0 = origin[1] + fiy * by
            na = int(na_xy[fix, fiy])
            npair = int(n_pairs[fix, fiy])
            face = cmap(norm(npair)) if na > 0 else (0.92, 0.92, 0.92, 1.0)
            ax.add_patch(Rectangle((x0, y0), bx, by, facecolor=face,
                                   edgecolor='0.35', linewidth=0.6, zorder=1))
            if na > 0:
                ax.text(x0 + 0.5 * bx, y0 + 0.5 * by, f'{npair}\n({na})',
                        ha='center', va='center', fontsize=8 if bx >= 1.8 else 6.5,
                        color='k', zorder=3, linespacing=0.9)

    # Cutoff spheres: solid, thin, light/transparent
    for p, rc in zip(atom_pos, atom_rcut):
        ax.add_patch(Circle((p[0], p[1]), rc, fill=False, linestyle='-',
                            linewidth=0.5, edgecolor='#4a90c8', alpha=0.28, zorder=4))

    # Bond skeleton
    bonds = find_bonds(atom_pos, enames, Rcut=bond_rcut)
    if bonds:
        segs = [[atom_pos[i, :2], atom_pos[j, :2]] for i, j in bonds]
        ax.add_collection(LineCollection(
            segs, colors=[(0.15, 0.15, 0.15, 0.85)] * len(segs),
            linewidths=1.4, capstyle='round', zorder=5))
        print(f'  bonds: {len(bonds)} (Rcut={bond_rcut} Å)')

    # Atoms (larger markers, on top of bonds)
    for e, p in zip(enames, atom_pos):
        col = '#1f77b4' if e == 'C' else '#7f7f7f'
        ms = 9.0 if e == 'C' else 6.5
        ax.plot(p[0], p[1], 'o', color=col, markersize=ms,
                zorder=6, markeredgecolor='k', markeredgewidth=0.45)

    # Legend proxy
    ax.plot([], [], 'o', color='#1f77b4', markersize=8, label='C')
    ax.plot([], [], 'o', color='#7f7f7f', markersize=6, label='H')
    ax.plot([], [], '-', color='#4a90c8', alpha=0.45, lw=0.8,
            label=f'Rcut (max≈{atom_rcut.max():.2f} Å)')
    ax.plot([], [], '-', color='0.2', lw=1.4, label='bonds')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('denmap pairs / block  =  na(na+1)/2')

    ax.set_aspect('equal')
    pad = 0.5
    ax.set_xlim(origin[0] - pad, origin[0] + nxb * bx + pad)
    ax.set_ylim(origin[1] - pad, origin[1] + nyb * by + pad)
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_title(
        f'LCAO tile partition (molecular plane)\n'
        f'8×8×8 voxel blocks · fiz={fiz} · label = pairs (na)'
        f'{title_extra}'
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'REVIEW: {out_path}')
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Illustrate LCAO block tiling for density projection')
    ap.add_argument('--xyz', default='data/xyz/pentacene.xyz')
    ap.add_argument('--step', type=float, default=0.25,
                    help='grid step [Å] (block edge = block*step; 0.25 → 2 Å tiles, clearer)')
    ap.add_argument('--margin', type=float, default=4.0)
    ap.add_argument('--block', type=int, default=8, help='voxels per block edge')
    ap.add_argument('--wfc', default=None, help='WFC HSD for Rcut (default: bundled 3ob)')
    ap.add_argument('--outdir', default='debug/lcao_tile_partition')
    args = ap.parse_args()

    import spammm.atomicUtils as au
    from spammm.SPM.AFM import setup_density_grid

    xyz = args.xyz if os.path.isabs(args.xyz) else os.path.join(_ROOT, args.xyz)
    pos, _, names, _, _ = au.load_xyz(xyz)
    atom_pos = np.asarray(pos, dtype=np.float64)
    enames = list(names)

    wfc = args.wfc
    if wfc is None:
        wfc = os.path.join(_ROOT, 'spammm', 'quantum', 'DFTB', 'data', 'wfc.3ob-3-1.hsd')
    atom_rcut = rcut_from_wfc(enames, wfc)
    print(f'Molecule: {os.path.basename(xyz)}  natoms={len(enames)}')
    print(f'Rcut [Å]: C={atom_rcut[enames.index("C")]:.3f}  H={atom_rcut[enames.index("H")]:.3f}')

    grid_spec, origin, ngrid = setup_density_grid(
        atom_pos, step=args.step, margin=args.margin, z_extra=args.margin, block=args.block)
    print(f'Grid ngrid={ngrid} origin={origin} step={args.step}')

    stem = os.path.splitext(os.path.basename(xyz))[0]
    out = os.path.join(_ROOT, args.outdir, f'{stem}_tile_xy_step{args.step:g}.svg')
    plot_plane_partition(
        atom_pos, enames, atom_rcut, grid_spec, args.block, out,
        title_extra=f' · step={args.step} Å · {os.path.basename(xyz)}')


if __name__ == '__main__':
    main()
