#!/usr/bin/env python3
"""
testplot_assembly.py — Rigid-body on-surface assembly (hexagonal SAM, 6 orientations/cell).

Experimental model: fixed unit cell, six C6-related orientations per cell, r=1.0 Å
(softened vs VdW to approximate residual flexibility in rigid search).

Run:  python tests/testplot_assembly.py
"""

import os
import sys
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from spammm.AtomicSystem import AtomicSystem
from spammm.forcefields.Assembly import parse_lattice_vectors, run_assembly_search, min_z_span_oriented
from spammm.forcefields.AssemblyPlot import (
    plot_assembly_height, plot_assembly_xy_xz_panel, plot_translations, plot_rotations, plot_pareto, replicate_bonds,
    plot_assembly_diagnostics, write_assembly_xyz, select_top_atoms,
)

PLOT_DIR = os.path.join(_proj, 'debug', 'testplot_assembly')
HELICENE_CELL = parse_lattice_vectors('lvs 32.7 0 0 16.35 28.319 0 0 0 40')

PRESETS = {
    'tetraceno': dict(mol='data/xyz/DiTetraceno_helicene_1a.xyz', rot_mode='tilt', nrot=16, n_tilt=5, tilt_range=0.25, nshift=10, shift_range=0.4, n_sym=6),
    'triptyceno': dict(mol='data/xyz/DiTriptyceno_helicene_3a.xyz', rot_mode='tilt', nrot=16, n_tilt=5, tilt_range=0.25, nshift=10, shift_range=0.4, n_sym=6),
}


def _ensure_bonds(mol):
    if mol.bonds is None or len(mol.bonds) == 0:
        mol.findBonds(Rcut=3.0, RvdwCut=0.5)
    return np.asarray(mol.bonds, dtype=np.int32)


def _append_xyz_frame(f, apos, enames, natoms, comment):
    f.write(f'{len(apos)}\n{comment}\n')
    for i, p in enumerate(apos):
        elem = enames[i % natoms]
        elem = elem.split('_')[0] if '_' in elem else elem
        f.write(f'{elem} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')


def run_one(args, mol_path, tag, outdir):
    mol_path = mol_path if os.path.isabs(mol_path) else os.path.join(_proj, mol_path)
    print(f'\n{"="*60}\n{tag}: {mol_path}\n{"="*60}')
    mol = AtomicSystem(fname=mol_path)
    cell = parse_lattice_vectors(args.cell) if args.cell else (mol.lvec if mol.lvec is not None else HELICENE_CELL)
    mol.lvec = cell
    z_flat = min_z_span_oriented(mol.apos)
    print(f'natoms={mol.natoms}  z-flat(SO3)={z_flat:.2f} Å  n_sym={args.n_sym}  radius={args.radius} Å')
    print(f'cell (fixed):\n{cell}')

    t0 = time.time()
    res = run_assembly_search(
        mol, cell, cell_scale=args.cell_scale, nrot=args.nrot, rot_mode=args.rot_mode, n_tilt=args.n_tilt, tilt_range=args.tilt_range,
        nshift=args.nshift, shift_range=args.shift_range, shift_region=args.shift_region, shift_sum_max=args.shift_sum_max,
        n_pbc_test=args.nPBC_test, n_pbc_xyz=args.nPBC_xyz, n_sym=args.n_sym, zspan_max=args.zspan_max, zspan_slack=args.zspan_slack,
        clash_max=args.clash_max, dist_min=args.dist_min, dist_max=args.dist_max,
        zpenalty=args.zpenalty, pack_weight=args.pack_weight, clash_weight=args.clash_weight,
        penalty=args.penalty, radius=args.radius, export_max=args.export_max, top_k=args.top_k,
        dedup=bool(args.dedup), align_flat=not args.no_align_flat, wg=args.wg, device=args.device, simple=args.simple,
    )
    dt = time.time() - t0
    scores, min_dists, z_spans = res['scores'], res['min_dists'], res['z_spans']
    print(f'zspan_max={res["zspan_max"]:.2f} Å  clash_max={args.clash_max}')
    print(f'search {dt:.2f}s — {res["n_confs"]} configs, nmols_eval={res["nmols_eval"]} (incl. {args.n_sym}×sym/cell)')
    print(f'clash: min={scores.min():.4f} med={np.median(scores):.4f}  export_pass={res["export_sorted"].size}')
    bi = res['best_idx']
    print(f'best idx={bi} clash={scores[bi]:.4f} mindist={min_dists[bi]:.3f} zspan={z_spans[bi]:.2f} total={res["total_scores"][bi]:.2f}')

    os.makedirs(outdir, exist_ok=True)
    if args.plot_trans:
        p = os.path.join(outdir, f'trans_{tag}.png')
        plot_translations(res['T_conf'], cell, p)
        print(f'REVIEW: {p}')
    if args.plot_rot:
        p = os.path.join(outdir, f'rot_{tag}.png')
        plot_rotations(res['R_conf'], p)
        print(f'REVIEW: {p}')

    p_pareto = os.path.join(outdir, f'pareto_{tag}.png')
    plot_pareto(scores, z_spans, min_dists, res['export_sorted'], bi, args.clash_max, res['zspan_max'], args.dist_min, args.penalty, p_pareto)
    print(f'REVIEW: {p_pareto}')

    base_bonds = _ensure_bonds(mol)
    ocl, nmols_out, sel = res['ocl'], res['nmols_out'], res['sel_indices']
    plot_indices = list(res['best_indices'][:args.plot_best_k]) if len(res['best_indices']) else [bi]

    movie_path = os.path.join(outdir, f'assembly_{tag}_movie.xyz')
    movie_f = open(movie_path, 'w') if args.dump else None
    dump_list = list(res['export_sorted']) if args.dump_all else plot_indices

    for rank, idx in enumerate(plot_indices):
        best_tr = res['transforms'][idx][sel]
        out_atoms = ocl.emit_configuration(best_tr, nmols_out)
        apos = out_atoms[:, :3].copy()
        z0 = apos[:, 2].min()
        apos[:, 2] -= z0
        out_atoms[:, 2] -= z0
        bonds = replicate_bonds(base_bonds, mol.natoms, nmols_out)
        stem = os.path.join(outdir, f'assembly_{tag}_rank{rank}')
        meta = f'idx={idx} rank={rank} clash={scores[idx]:.4f} mindist={min_dists[idx]:.4f} zspan={z_spans[idx]:.4f}'
        xyz_path = f'{stem}.xyz'
        clash, clearance, strain, diag_paths = plot_assembly_diagnostics(apos, out_atoms, mol.natoms, bonds, cell, stem, n_pbc_super=args.nPBC_xyz, rank=rank)
        write_assembly_xyz(xyz_path, apos, mol.enames, mol.natoms, cell, meta, scalars=clash)
        print(f'REVIEW: {xyz_path}')
        for p in diag_paths:
            print(f'REVIEW: {p}')
        title = f'rank {rank+1}  clash={scores[idx]:.2f}  z={z_spans[idx]:.1f}Å  dmin={min_dists[idx]:.2f}Å  ({args.n_sym} sym/cell)'
        top_idx, zmax, z_cut = select_top_atoms(apos, args.z_highlight)
        img = f'{stem}.png'
        plot_assembly_xy_xz_panel(
            apos, bonds=bonds, cell_lvs=cell, highlight_dz=args.z_highlight, n_pbc_super=args.nPBC_xyz,
            cmap_name=args.cmap, atom_size=max(5.0, args.atom_size * 0.45), bond_lw=0.3, cell_lw=0.5,
            guide_lw=0.5, title=f'{title}  top_dz={args.z_highlight:.2f}Å  n_top={len(top_idx)}',
            fname=img, dpi=200)
        print(f'  top atoms: n={len(top_idx)}  zmax={zmax:.3f}  z_cut={z_cut:.3f}  (z > zmax−{args.z_highlight:.2f})')
        with open(f'{stem}.diag', 'w') as df:
            df.write(f'# {meta}\n')
            df.write(f'clash_sum={clash.sum():.6f} max_atom={clash.max():.6f}\n')
            df.write(f'strain_max={strain.max():.6f} clearance_min={clearance.min():.6f}\n')
            top = np.argsort(-clash)[:15]
            df.write('# top clash atoms: idx clash clearance strain\n')
            for i in top:
                if clash[i] <= 0:
                    break
                df.write(f'{i} {clash[i]:.6f} {clearance[i]:.6f} {strain[i]:.6f}\n')
        print(f'REVIEW: {stem}.diag')

    if movie_f is not None:
        for rank, idx in enumerate(dump_list):
            best_tr = res['transforms'][idx][sel]
            out_atoms = ocl.emit_configuration(best_tr, nmols_out)
            apos = out_atoms[:, :3].copy()
            apos[:, 2] -= apos[:, 2].min()
            meta = f'idx={idx} rank={rank} clash={scores[idx]:.4f} mindist={min_dists[idx]:.4f} zspan={z_spans[idx]:.4f}'
            _append_xyz_frame(movie_f, apos, mol.enames, mol.natoms, meta)
        movie_f.close()
        print(f'REVIEW: {movie_path}')

    out_path = os.path.join(outdir, f'{tag}.out')
    with open(out_path, 'w') as f:
        f.write(f'# assembly {tag}\n# elapsed={dt:.2f}s\n\n')
        f.write(f'mol: {mol_path}\n')
        f.write(f'cell: fixed experimental\n')
        f.write(f'n_sym: {args.n_sym}\n')
        f.write(f'radius: {args.radius} Å\n')
        f.write(f'z_flat: {z_flat:.3f} Å\n')
        f.write(f'export_pass: {res["export_sorted"].size}\n')
        f.write(f'best clash={scores[bi]:.4f} mindist={min_dists[bi]:.4f} zspan={z_spans[bi]:.4f}\n')
        f.write('\n## Agent checklist\n')
        f.write('1. Unit cell shows 6 distinct orientations (C6 symmetry)\n')
        f.write('2. Layer is flat (z-span near z_flat, not upright)\n')
        f.write('3. Low clash at r=1.0; rank favors tight packing + flatness\n')
        f.write('4. Per-rank: assembly_*_rankN.xyz + _clash/_strain/_clearance.png + .diag\n')
    print(f'REVIEW: {out_path}')
    return res


def _apply_preset(args, preset_kw):
    args.mol = preset_kw['mol']
    for k in ('rot_mode', 'nrot', 'n_tilt', 'tilt_range', 'nshift', 'shift_range', 'shift_sum_max', 'n_sym'):
        if k in preset_kw:
            setattr(args, k, preset_kw[k])


def main():
    parser = argparse.ArgumentParser(description='Hexagonal SAM assembly (6 orientations/cell)')
    parser.add_argument('--preset', choices=['tetraceno', 'triptyceno', 'both', 'none'], default='both')
    parser.add_argument('--mol', type=str, default=None)
    parser.add_argument('--cell', type=str, default=None)
    parser.add_argument('--cell_scale', type=float, default=1.0, help='Optional xy scale (default 1.0 = fixed cell)')
    parser.add_argument('--nrot', type=int, default=16)
    parser.add_argument('--rot_mode', choices=['full3d', 'inplane', 'tilt'], default='tilt')
    parser.add_argument('--tilt_range', type=float, default=0.25)
    parser.add_argument('--n_tilt', type=int, default=5)
    parser.add_argument('--nshift', type=int, default=10)
    parser.add_argument('--shift_range', type=float, default=0.4)
    parser.add_argument('--shift_region', choices=['square', 'triangle'], default='triangle')
    parser.add_argument('--shift_sum_max', type=float, default=0.8)
    parser.add_argument('--penalty', type=float, default=50.0)
    parser.add_argument('--zpenalty', type=float, default=2.0)
    parser.add_argument('--clash_weight', type=float, default=1.0)
    parser.add_argument('--pack_weight', type=float, default=1.0)
    parser.add_argument('--zspan_max', type=float, default=None)
    parser.add_argument('--zspan_slack', type=float, default=1.15)
    parser.add_argument('--clash_max', type=float, default=5.0)
    parser.add_argument('--export_max', type=int, default=100)
    parser.add_argument('--nPBC_test', type=int, default=2)
    parser.add_argument('--nPBC_xyz', type=int, default=1)
    parser.add_argument('--n_sym', type=int, default=6, choices=[1, 6])
    parser.add_argument('--dist_min', type=float, default=1.0)
    parser.add_argument('--dist_max', type=float, default=None)
    parser.add_argument('--dedup', type=int, default=0, help='Rotation dedup (slow O(N²); off by default)')
    parser.add_argument('--radius', type=float, default=1.0, help='Collision radius (1.0 = rigid-search softened)')
    parser.add_argument('--no_align_flat', action='store_true', help='Skip SO(3) flat pre-alignment')
    parser.add_argument('--wg', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--dump_all', action='store_true')
    parser.add_argument('--plot_trans', action='store_true', default=True)
    parser.add_argument('--plot_rot', action='store_true', default=True)
    parser.add_argument('--plot_best_k', type=int, default=3)
    parser.add_argument('--z_highlight', type=float, default=0.4)
    parser.add_argument('--cmap', type=str, default='viridis')
    parser.add_argument('--atom_size', type=float, default=14.0)
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--outdir', type=str, default=PLOT_DIR)
    args = parser.parse_args()

    if args.preset == 'both':
        for tag in ('tetraceno', 'triptyceno'):
            a = argparse.Namespace(**vars(args))
            _apply_preset(a, PRESETS[tag])
            run_one(a, a.mol, tag, args.outdir)
    elif args.preset != 'none':
        _apply_preset(args, PRESETS[args.preset])
        run_one(args, args.mol, args.preset, args.outdir)
    elif args.mol:
        run_one(args, args.mol, 'custom', args.outdir)
    else:
        parser.error('Provide --mol or --preset')


if __name__ == '__main__':
    main()
