#!/usr/bin/env python3
"""Assembly → AFM imaging pipeline (Morse PP-AFM, multi-rank).

Phase 1 (doc/Tasks/Assembly_AFM_Pipeline.md):
  - Search → annotate score-twins → AFM for n_afm best (kept, not collapsed)
  - Per rank: outdir/rankXX_idxYY/{geometry_xy_xz.png, afm_df_Fz_heights.png, …}

Examples:
  python run_assembly_afm.py --preset tetraceno --n-afm 10 --scan-dx 0.15
  python run_assembly_afm.py --xyz debug/helicene_afm_pipeline/assembly_unitcell.xyz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault('PYOPENCL_CTX', '0')

from spammm.SPM.AFM import AFMulator, compute_df, compute_df_amp
from spammm.SPM import AFM_utils as afm_u
from spammm.AtomicSystem import AtomicSystem
from spammm.forcefields.Assembly import parse_lattice_vectors, run_assembly_search, annotate_score_twins
from spammm.forcefields.AssemblyPlot import (
    write_assembly_xyz, replicate_bonds, plot_assembly_xy_xz_panel, select_top_atoms,
)

HELICENE_CELL = 'lvs 32.7 0 0 16.35 28.319 0 0 0 40'
PARAMS = os.path.join(_ROOT, 'data', 'ElementTypes.dat')
PRESETS = {
    'tetraceno': dict(mol='data/xyz/DiTetraceno_helicene_1a.xyz', cell=HELICENE_CELL),
    'triptyceno': dict(mol='data/xyz/DiTriptyceno_helicene_3a.xyz', cell=HELICENE_CELL),
}


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def _parse_cell_from_xyz(path: str):
    with open(path) as f:
        f.readline()
        line = f.readline().strip()
    cell_lvs, nmols = None, None
    if 'lvs:' in line:
        vals = [float(x) for x in line.split('lvs:')[1].split()]
        cell_lvs = np.array(vals, dtype=np.float64).reshape(3, 3)
    if 'nmols=' in line:
        nmols = int(line.split('nmols=')[1].split()[0])
    return cell_lvs, nmols


def _load_bonds(mol: AtomicSystem, natoms_per_mol: int):
    nmols = max(1, mol.natoms // natoms_per_mol)
    tpl = AtomicSystem()
    tpl.natoms = natoms_per_mol
    tpl.apos = mol.apos[:natoms_per_mol]
    tpl.enames = list(mol.enames[:natoms_per_mol])
    tpl.findBonds(Rcut=3.0, RvdwCut=0.5)
    return replicate_bonds(np.asarray(tpl.bonds, dtype=np.int32), natoms_per_mol, nmols)


def _fit(afm: AFMulator, method: str, margin: float, bspl_dx: float, n_iter: int, pic_reg: float):
    t0 = time.time()
    if method == 'contact-sep':
        afm.fit_contact_surface(margin=margin, bspl_dx=bspl_dx, fit_z_adaptive=(1.0, 5.0, 0.2, 0.8),
                                m_start=4, nz=5, n_iter=n_iter, fit_force_weight=0.5, bPrint=True)
    elif method == 'contact-pic':
        afm.fit_pic_contact_surface(margin=margin, poly_R=4.0, m_start=4, nz=4, cell_size=10.0,
                                    fit_z_adaptive=(1.0, 5.0, 0.2, 0.8), n_iter=n_iter, reg=pic_reg, bPrint=True)
    elif method == 'morse-3d':
        pass
    else:
        raise ValueError(f'unknown method={method!r}')
    print(f'  fit/setup: {time.time() - t0:.2f}s')


def _run_scan(afm: AFMulator, method: str, nxy, scan_p0, scan_da, scan_db, nz: int, dtip: float):
    kw = dict(nxy=nxy, nz=nz, dtip=dtip, scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db)
    if method == 'contact-sep':
        return afm.run_scan_contact(**kw)
    if method == 'contact-pic':
        return afm.run_scan_pic(**kw)
    if method == 'morse-3d':
        afm.setup_grid(n=(max(40, nxy[0]), max(40, nxy[1]), max(30, nz)), margin=3.0, z_top=14.0)
        afm.make_forcefield()
        return afm.run_scan(**kw)
    raise ValueError(method)


def _enames_for_assembly(enames_tpl, natoms_per_mol, n_atoms):
    """Expand molecule-template enames to one name per atom in the assembly."""
    tpl = list(enames_tpl[:natoms_per_mol])
    if len(tpl) != natoms_per_mol:
        raise ValueError(f'enames template len={len(tpl)} != natoms_per_mol={natoms_per_mol}')
    if n_atoms % natoms_per_mol != 0:
        raise ValueError(f'n_atoms={n_atoms} not divisible by natoms_per_mol={natoms_per_mol}')
    return tpl * (n_atoms // natoms_per_mol)


def _write_flat_xyz(path, apos, enames, comment):
    """Write plain XYZ; fail loud if lengths mismatch (never zip-truncate)."""
    apos = np.asarray(apos, dtype=np.float64)
    enames = list(enames)
    if len(enames) != len(apos):
        raise ValueError(f'_write_flat_xyz: len(enames)={len(enames)} != len(apos)={len(apos)} → would truncate')
    with open(path, 'w') as f:
        f.write(f'{len(apos)}\n{comment}\n')
        for e, p in zip(enames, apos):
            f.write(f'{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')


def _prepare_wrapped(apos, enames, cell_lvs, natoms_per_mol, outdir, tag, ff_pbc: int):
    """Wrap to cell; write primary + optional FF PBC xyz. Returns apos_primary, xyz_ff, bonds."""
    la, lb, ang, area = afm_u.cell_alat_info(cell_lvs)
    print(f'Unit cell: |a|={la:.3f} Å  |b|={lb:.3f} Å  angle={ang:.2f}°  area={area:.1f} Å²')
    apos_w = afm_u.wrap_atoms_to_cell(apos, cell_lvs, natoms_per_mol=natoms_per_mol)
    apos_w[:, 2] -= apos_w[:, 2].min()
    enames_w = _enames_for_assembly(enames, natoms_per_mol, len(apos_w))
    xyz_w = os.path.join(outdir, f'{tag}_wrapped.xyz')
    write_assembly_xyz(xyz_w, apos_w, enames, natoms_per_mol, cell_lvs,
                       f'wrapped  |a|={la:.3f} |b|={lb:.3f} ang={ang:.2f}')
    print(f'REVIEW: {xyz_w}  ({len(apos_w)} atoms, {len(apos_w)//natoms_per_mol} mols)')
    mol_b = AtomicSystem(fname=xyz_w)
    if mol_b.natoms != len(apos_w):
        raise RuntimeError(f'reload wrapped xyz: got {mol_b.natoms} expected {len(apos_w)}')
    bonds = _load_bonds(mol_b, natoms_per_mol)
    if ff_pbc > 0:
        apos_ff, enames_ff, n_prim = afm_u.replicate_cell_pbc(apos_w, enames_w, cell_lvs, n_pbc=ff_pbc)
        n_img = (2 * ff_pbc + 1) ** 2
        if len(apos_ff) != n_prim * n_img:
            raise RuntimeError(f'PBC replicate size {len(apos_ff)} != {n_prim}×{n_img}')
        xyz_ff = os.path.join(outdir, f'{tag}_wrapped_pbc{ff_pbc}.xyz')
        _write_flat_xyz(xyz_ff, apos_ff, enames_ff,
                        f'wrapped + {ff_pbc} PBC shells ({n_img} cells)  |a|={la:.3f}')
        with open(xyz_ff) as f:
            n_hdr = int(f.readline().split()[0])
        n_body = sum(1 for _ in open(xyz_ff)) - 2
        if n_hdr != len(apos_ff) or n_body != len(apos_ff):
            raise RuntimeError(f'PBC xyz corrupt: header={n_hdr} body_lines={n_body} expected={len(apos_ff)}')
        print(f'REVIEW: {xyz_ff}  ({len(apos_ff)} atoms = primary {n_prim} × {n_img} images, '
              f'{len(apos_ff)//natoms_per_mol} mols)')
    else:
        xyz_ff = xyz_w
    return apos_w, xyz_ff, bonds, (la, lb, ang, area)


def run_afm_one(args, apos_primary, xyz_ff, bonds, cell_lvs, outdir, tag, title_extra=''):
    """Fit+scan one structure; write geometry_xy_xz.png + afm_df_Fz_heights.png."""
    method = args.method
    afm = AFMulator(use_morse=True, use_fire=False)
    afm.load_molecule(xyz_ff)
    with open(xyz_ff) as f:
        n_expect = int(f.readline().split()[0])
    if afm.mol.natoms != n_expect:
        raise RuntimeError(f'AFMulator loaded {afm.mol.natoms} atoms but xyz header says {n_expect} ({xyz_ff})')
    print(f'AFM FF atoms: {afm.mol.natoms}  (must be full PBC layer)')
    afm.assign_params(params_path=PARAMS)
    mol_z = float(np.max(apos_primary[:, 2]))
    print(f'\nAFM method: {method}  tag={tag}')
    _fit(afm, method, args.margin, args.bspl_dx, args.n_iter, args.pic_reg)

    cell_origin = (0.0, 0.0)
    if cell_lvs is not None and args.wrap_cell:
        nxy, scan_p0, scan_da, scan_db, _ext = afm_u.cell_scan_grid(
            cell_lvs, mol_z, args.z_clearance, afm.dpos0[2], args.scan_dx, margin=args.scan_margin)
    else:
        nxy, scan_p0, scan_da, scan_db, _ext = afm.scan_bbox(margin=args.scan_margin, dx=args.scan_dx)
    print(f'Scan grid {nxy}  dx={args.scan_dx}Å  nz={args.nz_scan}')
    t0 = time.time()
    FEs, pts3 = _run_scan(afm, method, nxy, scan_p0, scan_da, scan_db, args.nz_scan, args.dtip)
    print(f'  scan: {time.time() - t0:.3f}s  shape={FEs.shape}')

    Fz = FEs[:, :, :, 2]
    df = compute_df_amp(Fz, abs(args.dtip), amp=args.amp) if args.amp > 0 else compute_df(Fz, args.dtip)
    scan_xs, scan_ys = afm_u.scan_axes_from_pts(pts3)
    extent = afm_u.scan_extent(scan_xs, scan_ys)
    h_tip, h_probe = afm_u.afm_scan_probe_heights(afm, args.nz_scan, args.dtip, mol_z=mol_z, z_clearance=args.z_clearance)
    L = abs(float(afm.dpos0[2]))
    print(afm_u.afm_height_geometry_note(mol_z, L_lever=L, amp=args.amp, z_clearance=args.z_clearance))
    print(f'  h_tip ∈[{h_tip[0]:.2f},{h_tip[-1]:.2f}]  h_probe ∈[{h_probe[0]:.2f},{h_probe[-1]:.2f}] Å above zmax')

    top_idx, zmax_mol, z_cut = select_top_atoms(apos_primary, args.top_dz)
    print(f'Top atoms: n={len(top_idx)}  zmax={zmax_mol:.3f}  z_cut={z_cut:.3f}  top_dz={args.top_dz:.2f}Å')

    # Alignment sanity: Fz repulsive peaks vs top atoms (same XY frame as imshow)
    iz_chk = int(np.linspace(0, args.nz_scan - 1, min(6, args.nz_scan))[len(np.linspace(0, args.nz_scan - 1, min(6, args.nz_scan))) // 2])
    Fz_sl = Fz[:, :, iz_chk]
    peaks = afm_u.find_afm_map_extrema(Fz_sl, scan_xs, scan_ys, mode='max', n=6, exclude_edge=3)
    if len(top_idx) and peaks:
        from spammm.forcefields.AssemblyPlot import top_atoms_with_pbc
        top_xyz, _, _ = top_atoms_with_pbc(apos_primary, cell_lvs, args.top_dz, n_pbc=1) if cell_lvs is not None else (apos_primary[top_idx], None, None)
        for pi, (_ix, _iy, x, y, val) in enumerate(peaks[:3]):
            d = float(np.linalg.norm(top_xyz[:, :2] - np.array([x, y]), axis=1).min())
            print(f'  align Fz_max@{h_probe[iz_chk]:.1f}Å peak{pi+1} ({x:.2f},{y:.2f}) Fz={val:.4f}  dist→top(+PBC)={d:.2f}Å')

    iz = [int(round(i)) for i in np.linspace(0, args.nz_scan - 1, min(6, args.nz_scan))]
    iz_mid = iz[len(iz) // 2]
    probe_z = zmax_mol + float(h_probe[iz_mid])

    geo_title = f'{tag}  top_dz={args.top_dz:.2f}Å  n_top={len(top_idx)}{title_extra}'
    p_geo = os.path.join(outdir, 'geometry_xy_xz.png')
    plot_assembly_xy_xz_panel(
        apos_primary, bonds=bonds, cell_lvs=cell_lvs, highlight_dz=args.top_dz,
        n_pbc_super=0, atom_size=5.0, bond_lw=0.3, cell_lw=0.5, guide_lw=0.5,
        probe_z=probe_z, title=geo_title, fname=p_geo, dpi=160)

    p_afm = os.path.join(outdir, 'afm_df_Fz_heights.png')
    afm_u.plot_afm_df_Fz_height_strip(
        df, Fz, h_probe, extent=extent, iz=iz, h_tip=h_tip,
        title=f'{tag} — {method}{title_extra}', apos=apos_primary, cell_lvs=cell_lvs,
        cell_origin=cell_origin, top_dz=args.top_dz, cell_lw=0.5, top_ms=2.2,
        fname='afm_df_Fz_heights.png', save_dir=outdir, amp=args.amp if args.amp > 0 else None)

    with open(os.path.join(outdir, 'SUMMARY.out'), 'w') as f:
        f.write(f'tag={tag}\nmethod={method}\nnatoms={len(apos_primary)}\n')
        if cell_lvs is not None:
            la, lb, ang, area = afm_u.cell_alat_info(cell_lvs)
            f.write(f'|a|={la:.3f} |b|={lb:.3f} ang={ang:.2f} area={area:.1f}\n')
        f.write(f'scan={nxy} dx={args.scan_dx} nz={args.nz_scan} amp={args.amp} z_clearance={args.z_clearance}\n')
        f.write(f'h_probe [{h_probe[0]:.2f},{h_probe[-1]:.2f}]  h_tip [{h_tip[0]:.2f},{h_tip[-1]:.2f}]  '
                f'zmax={zmax_mol:.3f} z_cut={z_cut:.3f} n_top={len(top_idx)}\n')
        f.write(afm_u.afm_height_geometry_note(zmax_mol, L_lever=abs(float(afm.dpos0[2])),
                                               amp=args.amp, z_clearance=args.z_clearance) + '\n')
        f.write(f'{title_extra.strip()}\n')
    print(f'REVIEW: {os.path.join(outdir, "SUMMARY.out")}')
    return dict(Fz=Fz, df=df, h_probe=h_probe, zmax=zmax_mol, n_top=len(top_idx), p_geo=p_geo, p_afm=p_afm)


def _emit_rank_xyz(res, idx, enames, natoms_per_mol, cell, out_xyz, meta):
    tr = res['transforms'][idx][res['sel_indices']]
    atoms = res['ocl'].emit_configuration(tr, res['nmols_out'])
    ap = atoms[:, :3].copy()
    ap[:, 2] -= ap[:, 2].min()
    write_assembly_xyz(out_xyz, ap, enames, natoms_per_mol, cell, meta)
    return ap


def _setup_gridff_cell_box(afm, cell_lvs, mol_z, z_clearance, dx_grid, margin):
    """GridFF box = cell AABB + margin in xy; z from -1 to tip-start+2. World coords (no shift)."""
    a = np.asarray(cell_lvs[0, :2], float)
    b = np.asarray(cell_lvs[1, :2], float)
    corners = np.array([[0., 0.], a, a + b, b])
    x0 = float(corners[:, 0].min()) - margin
    y0 = float(corners[:, 1].min()) - margin
    x1 = float(corners[:, 0].max()) + margin
    y1 = float(corners[:, 1].max()) + margin
    z0 = -1.0
    z1 = float(mol_z) + float(z_clearance) + abs(float(afm.dpos0[2])) + 2.0
    L = np.array([x1 - x0, y1 - y0, z1 - z0], dtype=np.float64)
    n = (max(8, int(np.ceil(L[0] / dx_grid))),
         max(8, int(np.ceil(L[1] / dx_grid))),
         max(8, int(np.ceil(L[2] / dx_grid))))
    afm.setup_grid_world((x0, y0, z0), L, n)
    return n, L


def _pick_two_tops(apos, top_dz):
    """Two well-separated top atoms for z-profiles."""
    idx, zmax, z_cut = select_top_atoms(apos, top_dz)
    if len(idx) == 0:
        raise RuntimeError('no top atoms')
    tops = apos[idx]
    i0 = int(np.argmax(tops[:, 2]))
    d = np.linalg.norm(tops[:, :2] - tops[i0, :2], axis=1)
    i1 = int(np.argmax(d))
    return [(f'T{k}', tops[i]) for k, i in enumerate((i0, i1))], zmax, z_cut


def compare_contact_vs_gridff(args, rank_dir: str) -> int:
    """One-rank: contact-sep vs Morse+Coulomb GridFF (same scan); maps + E/Fz profiles at 2 tops."""
    rank_dir = _abs(rank_dir)
    xyz_ff = os.path.join(rank_dir, 'assembly_wrapped_pbc1.xyz')
    xyz_w = os.path.join(rank_dir, 'assembly_wrapped.xyz')
    if not os.path.isfile(xyz_ff):
        raise SystemExit(f'missing {xyz_ff} — run assembly AFM first')
    cell_lvs, _ = _parse_cell_from_xyz(xyz_w if os.path.isfile(xyz_w) else xyz_ff)
    if cell_lvs is None:
        cell_lvs = parse_lattice_vectors(args.cell)
    mol_p = AtomicSystem(fname=xyz_w if os.path.isfile(xyz_w) else xyz_ff)
    if not os.path.isfile(xyz_w):
        raise SystemExit(f'missing {xyz_w}')
    apos_primary = AtomicSystem(fname=xyz_w).apos[:, :3].copy()
    mol_z = float(apos_primary[:, 2].max())
    sites, zmax, z_cut = _pick_two_tops(apos_primary, args.top_dz)
    print(f'Compare contact-sep vs GridFF Morse+Coulomb  dir={rank_dir}')
    print(f'  zmax={zmax:.3f}  tops: ' + ', '.join(f'{n}=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})' for n, p in sites))

    # --- shared scan geometry ---
    nxy, scan_p0, scan_da, scan_db, extent = afm_u.cell_scan_grid(
        cell_lvs, mol_z, args.z_clearance, -4.0, args.scan_dx, margin=args.scan_margin)

    # --- contact-sep ---
    afm_c = AFMulator(use_morse=True, use_fire=False)
    afm_c.load_molecule(xyz_ff)
    afm_c.assign_params(params_path=PARAMS)
    print(f'AFM FF atoms (contact): {afm_c.mol.natoms}')
    _fit(afm_c, 'contact-sep', args.margin, args.bspl_dx, args.n_iter, args.pic_reg)
    # fix dpos0 lever for height labels
    scan_p0 = scan_p0.copy()
    scan_p0[2] = mol_z + args.z_clearance + abs(float(afm_c.dpos0[2]))
    t0 = time.time()
    FEs_c, pts3 = afm_c.run_scan_contact(nxy=nxy, nz=args.nz_scan, dtip=args.dtip,
                                         scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db)
    print(f'  contact scan: {time.time() - t0:.2f}s')
    Fz_c = FEs_c[:, :, :, 2]
    df_c = compute_df_amp(Fz_c, abs(args.dtip), amp=args.amp) if args.amp > 0 else compute_df(Fz_c, args.dtip)

    # --- GridFF Morse+Coulomb (cell AABB only — not full 3×3 bbox) ---
    afm_g = AFMulator(use_morse=True, use_fire=False)
    afm_g.load_molecule(xyz_ff)
    afm_g.assign_params(params_path=PARAMS)
    print(f'AFM FF atoms (grid3d): {afm_g.mol.natoms}')
    dx_g = float(args.grid_dx)
    n_g, L_g = _setup_gridff_cell_box(afm_g, cell_lvs, mol_z, args.z_clearance, dx_g, args.scan_margin + 1.0)
    t0 = time.time()
    afm_g.make_forcefield()
    print(f'  make_forcefield: {time.time() - t0:.2f}s  n={n_g} L={L_g}')
    t0 = time.time()
    FEs_g, _ = afm_g.run_scan(nxy=nxy, nz=args.nz_scan, dtip=args.dtip,
                              scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db)
    print(f'  grid3d scan: {time.time() - t0:.2f}s')
    Fz_g = FEs_g[:, :, :, 2]
    df_g = compute_df_amp(Fz_g, abs(args.dtip), amp=args.amp) if args.amp > 0 else compute_df(Fz_g, args.dtip)

    h_tip, h_probe = afm_u.afm_scan_probe_heights(afm_c, args.nz_scan, args.dtip, mol_z=mol_z, z_clearance=args.z_clearance)
    iz = [int(round(i)) for i in np.linspace(0, args.nz_scan - 1, min(6, args.nz_scan))]

    fig, axes = plt.subplots(4, len(iz), figsize=(2.3 * len(iz), 9.2), squeeze=False)
    rows_plot = [
        (df_c, 'df contact-sep', 'gray', False),
        (df_g, 'df GridFF', 'gray', False),
        (Fz_c, 'Fz contact-sep', 'bwr', True),
        (Fz_g, 'Fz GridFF', 'bwr', True),
    ]
    for col, i in enumerate(iz):
        for row, (arr, ylab, cmap, sym) in enumerate(rows_plot):
            ax = axes[row, col]
            ttl = f'probe {h_probe[i]:.1f}Å' if row == 0 else ''
            afm_u.imshow_afm(ax, arr[:, :, i], extent=extent, cmap=cmap, symmetric=sym, title=ttl,
                             colorbar=(col == len(iz) - 1))
            afm_u.overlay_afm_geometry(ax, apos=apos_primary, cell_lvs=cell_lvs, show_bonds=False,
                                       show_cell=True, show_atoms=False, show_top_atoms=True,
                                       top_dz=args.top_dz, cell_lw=0.5, top_ms=2.5)
            # mark profile sites
            for name, p in sites:
                ax.plot(p[0], p[1], 'c+', ms=8, mew=1.2, zorder=20)
            afm_u.set_axes_to_cell(ax, cell_lvs, (0, 0), margin=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(ylab, fontsize=8)
    fig.suptitle(f'{os.path.basename(rank_dir)}  contact-sep vs Morse+Coulomb GridFF  (cyan += profile sites)', fontsize=10)
    fig.tight_layout()
    out_maps = os.path.join(rank_dir, 'compare_contact_vs_gridff_maps.png')
    fig.savefig(out_maps, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {out_maps}')

    # --- E(z), Fz(z) at 2 tops: brute + contact-raw + grid-raw ---
    hp = np.arange(float(args.profile_h_max), float(args.profile_h_min) - 1e-9, -abs(args.profile_dh))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    # axes: [0,0] E site0; [0,1] E site1; [1,0] Fz site0; [1,1] Fz site1
    for si, (name, p) in enumerate(sites):
        x, y = float(p[0]), float(p[1])
        zq = np.column_stack([np.full(len(hp), x), np.full(len(hp), y), mol_z + hp]).astype(np.float32)
        E_br, F_br = afm_c._brute_afm_morse_c_queries(zq)
        Fz_br = F_br[:, 2]
        zd = np.zeros(3, dtype=np.float32)
        dtip_use = float(hp[1] - hp[0]) if len(hp) > 1 else -abs(args.profile_dh)
        scan0 = np.array([x, y, mol_z + float(hp[0]) - float(afm_c.dpos0[2])], dtype=np.float32)
        Fraw_c, _ = afm_c.get_raw_FE_contact(nxy=(1, 1), nz=len(hp), dtip=dtip_use, scan_p0=scan0, scan_da=zd, scan_db=zd)
        Fpp_c, _ = afm_c.run_scan_contact(nxy=(1, 1), nz=len(hp), dtip=dtip_use, scan_p0=scan0, scan_da=zd, scan_db=zd)
        Fraw_g, _ = afm_g.get_raw_FE(nxy=(1, 1), nz=len(hp), dtip=dtip_use, scan_p0=scan0, scan_da=zd, scan_db=zd)
        Fpp_g, _ = afm_g.run_scan(nxy=(1, 1), nz=len(hp), dtip=dtip_use, scan_p0=scan0, scan_da=zd, scan_db=zd)
        E_c, Fz_c1 = Fraw_c[0, 0, :, 3], Fraw_c[0, 0, :, 2]
        E_g, Fz_g1 = Fraw_g[0, 0, :, 3], Fraw_g[0, 0, :, 2]
        Fz_pp_c, Fz_pp_g = Fpp_c[0, 0, :, 2], Fpp_g[0, 0, :, 2]

        axE, axF = axes[0, si], axes[1, si]
        axE.plot(hp, E_br, 'k-', lw=1.8, label='brute Morse+Coul')
        axE.plot(hp, E_c, 'C0--', lw=1.3, label='contact-sep raw')
        axE.plot(hp, E_g, 'C1-.', lw=1.3, label='GridFF raw')
        axE.set_title(f'{name}  xy=({x:.2f},{y:.2f})  z_atom={p[2]:.2f}')
        axE.set_ylabel('E [eV]')
        axE.legend(fontsize=7); axE.grid(True, alpha=0.3); axE.axhline(0, color='k', lw=0.4)

        axF.plot(hp, Fz_br, 'k-', lw=1.8, label='brute Fz')
        axF.plot(hp, Fz_c1, 'C0--', lw=1.3, label='contact raw Fz')
        axF.plot(hp, Fz_g1, 'C1-.', lw=1.3, label='GridFF raw Fz')
        axF.plot(hp, Fz_pp_c, 'C0:', lw=1.2, label='contact PP Fz')
        axF.plot(hp, Fz_pp_g, 'C1:', lw=1.2, label='GridFF PP Fz')
        axF.set_xlabel('h_probe [Å above zmax]')
        axF.set_ylabel('Fz [eV/Å]')
        axF.legend(fontsize=7); axF.grid(True, alpha=0.3); axF.axhline(0, color='k', lw=0.4)
        axF.axvline(3.38, color='0.5', ls=':', lw=0.8, label='R0≈3.38')
    fig.suptitle('Forcefield E(z) / Fz(z) at top atoms — brute vs contact-sep vs GridFF', fontsize=11)
    fig.tight_layout()
    out_prof = os.path.join(rank_dir, 'compare_contact_vs_gridff_profiles.png')
    fig.savefig(out_prof, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {out_prof}')

    with open(os.path.join(rank_dir, 'compare_SUMMARY.out'), 'w') as f:
        f.write(f'rank_dir={rank_dir}\n')
        f.write(f'natoms_ff={afm_c.mol.natoms}  grid_n={n_g}  grid_dx={dx_g}\n')
        f.write(f'scan={nxy} dx={args.scan_dx} nz={args.nz_scan} z_clearance={args.z_clearance}\n')
        f.write(f'h_probe [{h_probe[0]:.2f},{h_probe[-1]:.2f}]\n')
        f.write(afm_u.afm_height_geometry_note(mol_z, amp=args.amp, z_clearance=args.z_clearance) + '\n')
        for name, p in sites:
            f.write(f'site {name}: xy=({p[0]:.3f},{p[1]:.3f}) z={p[2]:.3f}\n')
        f.write(f'REVIEW: {out_maps}\nREVIEW: {out_prof}\n')
    print(f'REVIEW: {os.path.join(rank_dir, "compare_SUMMARY.out")}')
    return 0


def run_afm(args) -> int:
    if getattr(args, 'compare_dir', None):
        return compare_contact_vs_gridff(args, args.compare_dir)
    out_root = _abs(args.outdir)
    os.makedirs(out_root, exist_ok=True)
    method = args.method

    # --- single xyz path ---
    if args.xyz and not args.preset:
        xyz = _abs(args.xyz)
        mol = AtomicSystem(fname=xyz)
        cell_lvs, nmols_xyz = _parse_cell_from_xyz(xyz)
        if cell_lvs is None and args.cell:
            cell_lvs = parse_lattice_vectors(args.cell)
        n_mols = nmols_xyz if nmols_xyz is not None else args.n_mols
        natoms_per_mol = mol.natoms // max(1, n_mols)
        tag = os.path.splitext(os.path.basename(xyz))[0]
        rank_dir = os.path.join(out_root, 'rank00_xyz')
        os.makedirs(rank_dir, exist_ok=True)
        apos = mol.apos[:, :3].copy()
        if cell_lvs is not None and args.wrap_cell:
            apos_p, xyz_ff, bonds, _ = _prepare_wrapped(
                apos, mol.enames, cell_lvs, natoms_per_mol, rank_dir, tag, args.ff_pbc)
        else:
            apos_p, xyz_ff, bonds = apos, xyz, _load_bonds(mol, natoms_per_mol)
        run_afm_one(args, apos_p, xyz_ff, bonds, cell_lvs, rank_dir, tag)
        with open(os.path.join(out_root, 'SUMMARY.out'), 'w') as f:
            f.write(f'mode=single_xyz\nxyz={xyz}\nmethod={method}\n')
            f.write(f'REVIEW: {rank_dir}/geometry_xy_xz.png\nREVIEW: {rank_dir}/afm_df_Fz_heights.png\n')
        print(f'REVIEW: {os.path.join(out_root, "SUMMARY.out")}')
        return 0

    # --- preset: assembly search → multi-rank AFM ---
    if not args.preset:
        raise SystemExit('Provide --xyz or --preset')
    pr = PRESETS[args.preset]
    mol = AtomicSystem(fname=_abs(pr['mol']))
    cell = parse_lattice_vectors(pr['cell'])
    mol.lvec = cell
    print(f'Assembly search ({args.preset}) …')
    t0 = time.time()
    res = run_assembly_search(
        mol, cell, nrot=args.nrot, n_tilt=args.n_tilt, nshift=args.nshift,
        n_pbc_xyz=args.n_pbc_xyz, n_sym=args.n_sym, export_max=max(args.export_max, args.n_afm),
        top_k=max(args.top_k, args.n_afm), wg=128, device=0)
    print(f'  search: {time.time() - t0:.2f}s  export_pass={res["export_sorted"].size}')

    export = list(res['export_sorted'][: args.n_afm])
    if not export:
        raise SystemExit('No exportable configs — relax clash_max / zspan / dist_min')
    twins = annotate_score_twins(export, res['scores'], res['z_spans'], res['min_dists'],
                                 eps_clash=args.eps_clash, eps_z=args.eps_zspan, eps_d=args.eps_dmin)

    root_lines = [
        f'preset={args.preset} method={method} n_afm={len(export)}',
        f'search_export_pass={res["export_sorted"].size}',
        '# rank idx clash z_span min_dist total twin_of_rank',
    ]
    for tw in twins:
        r, idx = tw['rank'], tw['idx']
        twin_note = ''
        if tw['twin_of_rank'] is not None:
            twin_note = f'  [score-twin of rank{tw["twin_of_rank"]:02d} idx={tw["twin_of_idx"]}]'
            print(f'  rank{r:02d} idx={idx} SCORE-TWIN of rank{tw["twin_of_rank"]:02d} '
                  f'(clash={tw["clash"]:.6f} z={tw["z_span"]:.4f} dmin={tw["min_dist"]:.4f}) — kept for judgment')
        title_extra = twin_note
        rank_dir = os.path.join(out_root, f'rank{r:02d}_idx{idx}')
        os.makedirs(rank_dir, exist_ok=True)
        meta = (f'idx={idx} rank={r} clash={tw["clash"]:.6f} zspan={tw["z_span"]:.4f} '
                f'dmin={tw["min_dist"]:.4f}{twin_note}')
        xyz_rank = os.path.join(rank_dir, 'assembly.xyz')
        ap = _emit_rank_xyz(res, idx, mol.enames, mol.natoms, cell, xyz_rank, meta)
        print(f'REVIEW: {xyz_rank}')

        apos_p, xyz_ff, bonds, _ = _prepare_wrapped(
            ap, mol.enames, cell, mol.natoms, rank_dir, f'assembly', args.ff_pbc)
        run_afm_one(args, apos_p, xyz_ff, bonds, cell, rank_dir, f'rank{r:02d}_idx{idx}', title_extra=title_extra)

        root_lines.append(
            f'{r:02d} {idx} {tw["clash"]:.6f} {tw["z_span"]:.4f} {tw["min_dist"]:.4f} '
            f'{res["total_scores"][idx]:.4f} {tw["twin_of_rank"] if tw["twin_of_rank"] is not None else "-"}')
        root_lines.append(f'  REVIEW: {rank_dir}/geometry_xy_xz.png')
        root_lines.append(f'  REVIEW: {rank_dir}/afm_df_Fz_heights.png')

    summary = os.path.join(out_root, 'SUMMARY.out')
    with open(summary, 'w') as f:
        f.write('\n'.join(root_lines) + '\n')
        f.write('\n# Score-twins are KEPT (annotate only). GPU atom-cloud match = Phase 2.\n')
        f.write('# min_dist = closest inter-molecular atom–atom distance (Å).\n')
    print(f'REVIEW: {summary}')
    return 0


def build_parser():
    p = argparse.ArgumentParser(description='Assembly → AFM (multi-rank)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--xyz', help='Single assembly .xyz (skips search)')
    p.add_argument('--compare-dir', dest='compare_dir', default=None,
                   help='Compare contact-sep vs Morse+Coulomb GridFF for one rank dir (needs *_wrapped*.xyz)')
    p.add_argument('--grid-dx', type=float, default=0.25, help='GridFF voxel step [Å] for --compare-dir')
    p.add_argument('--profile-h-max', type=float, default=10.0)
    p.add_argument('--profile-h-min', type=float, default=2.0)
    p.add_argument('--profile-dh', type=float, default=0.05)
    p.add_argument('--preset', choices=list(PRESETS), help='Assembly search then AFM on n_afm best')
    p.add_argument('--outdir', default='debug/helicene_afm_pipeline')
    p.add_argument('--n-afm', type=int, default=10, help='How many best exports to AFM (kept incl. score-twins)')
    p.add_argument('--method', choices=['contact-sep', 'contact-pic', 'morse-3d'], default='contact-sep')
    p.add_argument('--cell', default=HELICENE_CELL)
    p.add_argument('--n-mols', type=int, default=6)
    p.add_argument('--margin', type=float, default=3.0)
    p.add_argument('--bspl-dx', type=float, default=0.2)
    p.add_argument('--n-iter', type=int, default=60)
    p.add_argument('--pic-reg', type=float, default=1e-2)
    p.add_argument('--scan-dx', type=float, default=0.15)
    p.add_argument('--scan-margin', type=float, default=3.0)
    p.add_argument('--nz-scan', type=int, default=40)
    p.add_argument('--dtip', type=float, default=-0.1)
    p.add_argument('--z-clearance', type=float, default=8.0,
                   help='Initial h_probe above molecular zmax [Å]. Was 5 (=tip 9); raise for onset contrast')
    p.add_argument('--amp', type=float, default=1.0)
    p.add_argument('--wrap-cell', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--ff-pbc', type=int, default=1, dest='ff_pbc')
    p.add_argument('--top-dz', type=float, default=0.25)
    p.add_argument('--eps-clash', type=float, default=1e-4)
    p.add_argument('--eps-zspan', type=float, default=1e-3)
    p.add_argument('--eps-dmin', type=float, default=1e-3)
    p.add_argument('--nrot', type=int, default=16)
    p.add_argument('--n-tilt', type=int, default=5)
    p.add_argument('--nshift', type=int, default=10)
    p.add_argument('--n-pbc-xyz', type=int, default=0, dest='n_pbc_xyz')
    p.add_argument('--n-sym', type=int, default=6)
    p.add_argument('--export-max', type=int, default=50)
    p.add_argument('--top-k', type=int, default=10)
    return p


def main(argv=None) -> int:
    return run_afm(build_parser().parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
