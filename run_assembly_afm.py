#!/usr/bin/env python3
"""Assembly → AFM imaging pipeline (Morse PP-AFM, multiple field backends).

Examples:
  python run_assembly_afm.py --xyz debug/helicene_afm_pipeline/assembly_unitcell.xyz
  python run_assembly_afm.py --preset tetraceno --method contact-sep --scan-dx 0.15
  python run_assembly_afm.py --xyz assembly.xyz --method contact-pic --show-bonds --bond-alpha 0.3
  python run_assembly_afm.py --xyz assembly.xyz --method morse-3d --scan-dx 0.2

Methods (see doc/Tasks/Fast_2p5D_AFM_ContactSurface.md):
  contact-sep  — separable 2.5D contact surface (default; no img_FF)
  contact-pic  — radial PIC atom-bounded 2.5D field
  morse-3d     — full 3D img_FF GridFF + run_scan (reference / slow)
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
from spammm.forcefields.Assembly import parse_lattice_vectors, run_assembly_search
from spammm.forcefields.AssemblyPlot import write_assembly_xyz, replicate_bonds

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
    """Intramolecular bonds replicated across assembly (no inter-molecular links)."""
    nmols = max(1, mol.natoms // natoms_per_mol)
    tpl = AtomicSystem()
    tpl.natoms = natoms_per_mol
    tpl.apos = mol.apos[:natoms_per_mol]
    tpl.enames = list(mol.enames[:natoms_per_mol])
    tpl.findBonds(Rcut=3.0, RvdwCut=0.5)
    return replicate_bonds(np.asarray(tpl.bonds, dtype=np.int32), natoms_per_mol, nmols)


def _fit_and_scan(afm: AFMulator, method: str, margin: float, bspl_dx: float, n_iter: int, pic_reg: float):
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
    return method


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


def _dense_profile(afm: AFMulator, method: str, x: float, y: float, mol_z: float, h_min: float, h_max: float, dh: float, z_clearance: float):
    """Dense Fz(h_probe) via brute + contact raw/PP at fixed xy."""
    hp = np.arange(float(h_max), float(h_min) - 0.5 * dh, -abs(dh))
    zd = np.zeros(3, dtype=np.float32)
    nz = len(hp)
    h_tip = hp - float(afm.dpos0[2])
    ztips = mol_z + h_tip
    dtip = float(dh) if len(hp) > 1 else -0.05
    scan_p0 = np.array([x, y, float(ztips[0])], dtype=np.float32)
    if method in ('contact-sep', 'contact-pic'):
        Fpp, _ = _run_scan(afm, method, (1, 1), scan_p0, zd, zd, nz, -abs(dtip))
        Fraw, _ = (afm.get_raw_FE_contact if method == 'contact-sep' else afm.get_raw_FE_pic)(
            nxy=(1, 1), nz=nz, dtip=-abs(dtip), scan_p0=scan_p0, scan_da=zd, scan_db=zd)
        Fz_pp, Fz_raw = Fpp[0, 0, :, 2], Fraw[0, 0, :, 2]
    else:
        Fpp, _ = _run_scan(afm, method, (1, 1), scan_p0, zd, zd, nz, -abs(dtip))
        Fz_pp, Fz_raw = Fpp[0, 0, :, 2], Fpp[0, 0, :, 2]
    zq = np.column_stack([np.full(nz, x), np.full(nz, y), mol_z + hp]).astype(np.float32)
    _, F_br = afm._brute_afm_morse_c_queries(zq)
    return hp, Fz_raw, Fz_pp, F_br[:, 2]


def _pick_sites(args, df_map, scan_xs, scan_ys, apos, natoms_per_mol):
    if args.sites == 'com':
        sites = []
        for name, x, y, _z in afm_u.molecule_com_sites(apos, natoms_per_mol):
            ix = int(np.argmin(np.abs(np.asarray(scan_xs) - x)))
            iy = int(np.argmin(np.abs(np.asarray(scan_ys) - y)))
            val = float(df_map[ix, iy])
            sites.append((name, x, y, val))
        return sites[: args.n_spots] if args.n_spots < len(sites) else sites
    spots = afm_u.find_afm_map_extrema(df_map, scan_xs, scan_ys, mode='min', n=args.n_spots)
    return [(f'S{si+1}', x, y, val) for si, (_ix, _iy, x, y, val) in enumerate(spots)]


def run_afm(args) -> int:
    xyz = _abs(args.xyz) if args.xyz else None
    cell_lvs = None
    if args.preset and not args.xyz:
        pr = PRESETS[args.preset]
        mol = AtomicSystem(fname=_abs(pr['mol']))
        cell = parse_lattice_vectors(pr['cell'])
        mol.lvec = cell
        print(f'Assembly search ({args.preset}) n_pbc_xyz={args.n_pbc_xyz} ...')
        res = run_assembly_search(mol, cell, nrot=args.nrot, n_tilt=args.n_tilt, nshift=args.nshift,
                                  n_pbc_xyz=args.n_pbc_xyz, n_sym=args.n_sym, export_max=args.export_max,
                                  top_k=args.top_k, wg=128, device=0)
        idx = int(res['export_sorted'][0]) if res['export_sorted'].size else int(res['best_idx'])
        tr = res['transforms'][idx][res['sel_indices']]
        atoms = res['ocl'].emit_configuration(tr, res['nmols_out'])
        ap = atoms[:, :3].copy()
        ap[:, 2] -= ap[:, 2].min()
        os.makedirs(_abs(args.outdir), exist_ok=True)
        xyz = os.path.join(_abs(args.outdir), f'assembly_{args.preset}.xyz')
        write_assembly_xyz(xyz, ap, mol.enames, mol.natoms, cell,
                           f'idx={idx} clash={res["scores"][idx]:.4f} nmols={res["nmols_out"]}')
        print(f'REVIEW: {xyz}')
        natoms_per_mol = mol.natoms
        cell_lvs = cell
        mol = AtomicSystem(fname=xyz)
    else:
        if not xyz:
            raise SystemExit('Provide --xyz or --preset')
        mol = AtomicSystem(fname=xyz)
        cell_lvs, nmols_xyz = _parse_cell_from_xyz(xyz)
        n_mols = nmols_xyz if nmols_xyz is not None else args.n_mols
        natoms_per_mol = mol.natoms // max(1, n_mols)

    if args.show_cell or args.wrap_cell:
        if cell_lvs is None and args.cell:
            cell_lvs = parse_lattice_vectors(args.cell)
    else:
        cell_lvs = cell_lvs  # may still be from xyz

    os.makedirs(_abs(args.outdir), exist_ok=True)
    tag = os.path.splitext(os.path.basename(xyz))[0]

    apos = mol.apos[:, :3].copy()
    if cell_lvs is not None and args.wrap_cell:
        la, lb, ang, area = afm_u.cell_alat_info(cell_lvs)
        print(f'Unit cell: |a|={la:.3f} Å  |b|={lb:.3f} Å  angle={ang:.2f}°  area={area:.1f} Å²  (all axes in Å)')
        print(f'  Wrapping molecules into crystallographic cell [0,1)×[0,1) by COM')
        apos = afm_u.wrap_atoms_to_cell(apos, cell_lvs, natoms_per_mol=natoms_per_mol)
        apos[:, 2] -= apos[:, 2].min()
        xyz_w = os.path.join(_abs(args.outdir), f'{tag}_wrapped.xyz')
        write_assembly_xyz(xyz_w, apos, mol.enames, natoms_per_mol, cell_lvs,
                           f'wrapped  |a|={la:.3f} |b|={lb:.3f} ang={ang:.2f} nmols={mol.natoms // natoms_per_mol}')
        print(f'REVIEW: {xyz_w}')
        apos_primary = apos.copy()
        enames_primary = list(mol.enames)
        if args.ff_pbc > 0:
            apos_ff, enames_ff, n_prim = afm_u.replicate_cell_pbc(apos, mol.enames, cell_lvs, n_pbc=args.ff_pbc)
            xyz = os.path.join(_abs(args.outdir), f'{tag}_wrapped_pbc{args.ff_pbc}.xyz')
            # write flat xyz for AFMulator (no special assembly format needed)
            with open(xyz, 'w') as f:
                f.write(f'{len(apos_ff)}\nwrapped + {args.ff_pbc} PBC shells for FF  |a|={la:.3f}\n')
                for e, p in zip(enames_ff, apos_ff):
                    f.write(f'{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
            print(f'REVIEW: {xyz}  ({len(apos_ff)} atoms = primary {n_prim} × {(2*args.ff_pbc+1)**2} images)')
            apos = apos_primary  # overlays use primary cell only
        else:
            xyz = xyz_w
            mol = AtomicSystem(fname=xyz)
            apos = mol.apos[:, :3]
            apos_primary = apos
    elif cell_lvs is not None:
        la, lb, ang, area = afm_u.cell_alat_info(cell_lvs)
        print(f'Unit cell: |a|={la:.3f} Å  |b|={lb:.3f} Å  angle={ang:.2f}°  (--no-wrap-cell)')
        apos_primary = apos
    else:
        apos_primary = apos

    cell_origin = (0.0, 0.0)
    if args.show_bonds:
        mol_for_bonds = AtomicSystem(fname=xyz_w) if (cell_lvs is not None and args.wrap_cell) else mol
        bonds = _load_bonds(mol_for_bonds, natoms_per_mol)
    else:
        bonds = None
    apos = apos_primary

    afm = AFMulator(use_morse=True, use_fire=False)
    afm.load_molecule(xyz)
    afm.assign_params(params_path=PARAMS)
    mol_z = float(np.max(apos_primary[:, 2]))

    method = args.method
    print(f'\nAFM method: {method}  (Morse+Coulomb PP-AFM; see doc/Tasks/Fast_2p5D_AFM_ContactSurface.md)')
    _fit_and_scan(afm, method, args.margin, args.bspl_dx, args.n_iter, args.pic_reg)

    if cell_lvs is not None and args.wrap_cell:
        nxy, scan_p0, scan_da, scan_db, _ext = afm_u.cell_scan_grid(
            cell_lvs, mol_z, args.z_clearance, afm.dpos0[2], args.scan_dx, margin=args.scan_margin)
    else:
        nxy, scan_p0, scan_da, scan_db, _ext = afm.scan_bbox(margin=args.scan_margin, dx=args.scan_dx)
    print(f'Scan grid {nxy}  dx={args.scan_dx}Å  nz={args.nz_scan}  dtip={args.dtip}  (coords in Å)')
    t0 = time.time()
    FEs, pts3 = _run_scan(afm, method, nxy, scan_p0, scan_da, scan_db, args.nz_scan, args.dtip)
    print(f'  scan: {time.time() - t0:.3f}s  shape={FEs.shape}')

    Fz = FEs[:, :, :, 2]
    df = compute_df_amp(Fz, abs(args.dtip), amp=args.amp) if args.amp > 0 else compute_df(Fz, args.dtip)
    scan_xs, scan_ys = afm_u.scan_axes_from_pts(pts3)
    extent = afm_u.scan_extent(scan_xs, scan_ys)
    h_tip, h_probe = afm_u.afm_scan_probe_heights(afm, args.nz_scan, args.dtip, mol_z=mol_z, z_clearance=args.z_clearance)
    print(f'  h_tip∈[{h_tip[0]:.2f},{h_tip[-1]:.2f}]  h_probe∈[{h_probe[0]:.2f},{h_probe[-1]:.2f}] Å above zmax')

    iz = [int(round(i)) for i in np.linspace(0, args.nz_scan - 1, min(6, args.nz_scan))]
    geo = dict(apos=apos_primary, bonds=bonds, cell_lvs=cell_lvs if args.show_cell else None,
               show_bonds=args.show_bonds, show_cell=args.show_cell, show_atoms=args.show_atoms,
               cell_pbc=0, bond_lw=args.bond_lw, bond_alpha=args.bond_alpha, cell_origin=cell_origin)

    fig, axes = afm_u.plot_afm_height_panel(Fz, h_probe, iz=iz, extent=extent, label=f'Fz ({method})', cmap='bwr',
                                            h_tip=h_tip, height_label='probe', **geo)
    if cell_lvs is not None:
        la, lb, ang, _ = afm_u.cell_alat_info(cell_lvs)
        for ax in np.atleast_1d(axes).ravel():
            afm_u.set_axes_to_cell(ax, cell_lvs, cell_origin, margin=args.scan_margin)
        fig.suptitle(f'{tag} — {method}  |a|={la:.1f}Å ∠{ang:.0f}°  natoms={len(apos_primary)} (Å)', fontsize=10, y=1.02)
    else:
        fig.suptitle(f'{tag} — {method}  natoms={mol.natoms}', fontsize=10, y=1.02)
    p_fz = os.path.join(_abs(args.outdir), f'Fz_{tag}_{method}.png')
    fig.savefig(p_fz, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {p_fz}')

    fig, axes = afm_u.plot_afm_height_panel(df, h_probe, iz=iz, extent=extent, label=f'df ({method})', cmap='gray',
                                            h_tip=h_tip, height_label='probe', **geo)
    if cell_lvs is not None:
        for ax in np.atleast_1d(axes).ravel():
            afm_u.set_axes_to_cell(ax, cell_lvs, cell_origin, margin=args.scan_margin)
        fig.suptitle(f'{tag} — df amp={args.amp}Å  |a|={la:.1f}Å', fontsize=10, y=1.02)
    p_df = os.path.join(_abs(args.outdir), f'df_{tag}_{method}.png')
    fig.savefig(p_df, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {p_df}')

    # Diagnostic: one clear panel — AFM + atoms + cell
    iz_mid = iz[len(iz) // 2]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ttl = f'Fz  probe={h_probe[iz_mid]:.1f}Å'
    if cell_lvs is not None:
        ttl += f'  |a|={la:.1f}Å ∠{ang:.0f}° (Å)'
    afm_u.imshow_afm(ax, Fz[:, :, iz_mid], extent=extent, cmap='bwr', title=ttl)
    afm_u.overlay_afm_geometry(ax, apos=apos_primary, bonds=bonds, cell_lvs=cell_lvs, cell_origin=cell_origin,
                               show_bonds=args.show_bonds, show_cell=(cell_lvs is not None), show_atoms=True,
                               bond_lw=args.bond_lw, bond_alpha=args.bond_alpha)
    if cell_lvs is not None:
        afm_u.set_axes_to_cell(ax, cell_lvs, cell_origin, margin=args.scan_margin)
    for name, x, y, _z in afm_u.molecule_com_sites(apos_primary, natoms_per_mol):
        ax.plot(x, y, 'yo', ms=7, mec='k')
        ax.annotate(name, (x, y), fontsize=8, fontweight='bold')
    fig.tight_layout()
    p_diag = os.path.join(_abs(args.outdir), f'cell_atoms_Fz_{tag}.png')
    fig.savefig(p_diag, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {p_diag}')

    iz_diag = int(np.clip(args.profile_iz, 0, args.nz_scan - 1))
    sites = _pick_sites(args, df[:, :, iz_diag], scan_xs, scan_ys, apos_primary, natoms_per_mol)
    if sites:
        afm_u.plot_afm_sites_legend(df[:, :, iz_diag], scan_xs, scan_ys,
            [(n, x, y, v) for n, x, y, v in sites], extent=extent,
            title=f'df sites ({args.sites}) probe {h_probe[iz_diag]:.1f}Å — {method}',
            fname=f'sites_legend_{tag}.png', save_dir=_abs(args.outdir), **geo)

        prof_items = []
        for name, x, y, val in sites:
            hp, fr, fp, fb = _dense_profile(afm, method, x, y, mol_z, args.profile_h_min, args.profile_h_max,
                                            args.profile_dh, args.z_clearance)
            prof_items.append((name, val, hp, fr, fp, fb))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        afm_u.imshow_afm(axes[0], df[:, :, iz_diag], extent=extent, cmap='gray', symmetric=False,
                         title=f'df  h_probe={h_probe[iz_diag]:.1f}Å')
        afm_u.imshow_afm(axes[1], Fz[:, :, iz_diag], extent=extent, cmap='bwr', title=f'Fz  h_probe={h_probe[iz_diag]:.1f}Å')
        geo_ov = {k: v for k, v in geo.items() if k != 'cell_pbc'}
        for ax in axes:
            afm_u.overlay_afm_geometry(ax, **geo_ov)
            if cell_lvs is not None:
                afm_u.set_axes_to_cell(ax, cell_lvs, cell_origin, margin=args.scan_margin)
        for (name, x, y, val), (_n, _v, hp, fr, fp, fb) in zip(sites, prof_items):
            for ax in axes:
                ax.plot(x, y, 'o', mec='k', mfc='yellow', ms=8, zorder=10)
                ax.annotate(name, (x, y), xytext=(4, 4), textcoords='offset points', fontsize=9, fontweight='bold')
        fig.tight_layout()
        p_map = os.path.join(_abs(args.outdir), f'spots_map_{tag}.png')
        fig.savefig(p_map, dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f'REVIEW: {p_map}')

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for name, val, hp, fr, fp, fb in prof_items:
            axes[0].plot(hp, fr, '--', lw=1.0, alpha=0.8, label=f'{name} raw')
            axes[0].plot(hp, fp, '-', lw=1.5, label=f'{name} PP')
            axes[1].plot(hp, fb, '-', lw=1.2, label=f'{name} brute')
        for ax, ylab, ttl in zip(axes, ['Fz contact [eV/Å]', 'Fz brute Morse [eV/Å]'],
                                 ['Contact surface: raw (--) vs PP relax (-)', 'Brute atomistic Morse reference']):
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel(ylab)
            ax.set_title(ttl, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
        axes[1].set_xlabel('probe height h_probe [Å above molecular zmax]')
        fig.suptitle(f'Fz(z) at sites — vdW minimum typically h_probe≈2.5–3.5Å (Fz<0); kinks in PP = relax jumps', fontsize=9)
        fig.tight_layout()
        p_prof = os.path.join(_abs(args.outdir), f'spots_Fz_profiles_{tag}.png')
        fig.savefig(p_prof, dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f'REVIEW: {p_prof}')

        with open(os.path.join(_abs(args.outdir), f'spots_{tag}.out'), 'w') as f:
            f.write(f'method={method}\nsites={args.sites}\niz={iz_diag} h_probe={h_probe[iz_diag]:.3f}\n')
            f.write(f'cell_origin=(0,0) after wrap into [0,1)×[0,1)\n')
            if cell_lvs is not None:
                la, lb, ang, area = afm_u.cell_alat_info(cell_lvs)
                f.write(f'|a|={la:.3f} Å  |b|={lb:.3f} Å  angle={ang:.2f}°  area={area:.1f} Å²\n')
            f.write('M0..M5 = molecule centre-of-mass (wrapped); S* = darkest df pixel\n')
            f.write('Axes/extent in Å. Fz<0 attractive; Fz>0 repulsive.\n\n')
            for name, x, y, val in sites:
                f.write(f'{name}: xy=({x:.3f},{y:.3f}) df@slice={val:.6e}\n')

    summary = os.path.join(_abs(args.outdir), 'SUMMARY.out')
    with open(summary, 'w') as f:
        f.write(f'xyz={xyz}\nnatoms_primary={len(apos_primary)}\nmethod={method}\n')
        if cell_lvs is not None:
            la, lb, ang, area = afm_u.cell_alat_info(cell_lvs)
            f.write(f'|a|={la:.3f} Å  |b|={lb:.3f} Å  angle={ang:.2f}°  area={area:.1f} Å²\n')
        f.write(f'scan={nxy} dx={args.scan_dx} nz={args.nz_scan} dtip={args.dtip} amp={args.amp}\n')
        f.write(f'h_probe range [{h_probe[0]:.2f},{h_probe[-1]:.2f}] Å above zmax\n')
        f.write(f'wrap_cell={args.wrap_cell} ff_pbc={args.ff_pbc}\n')
        f.write(f'show_cell={args.show_cell} show_bonds={args.show_bonds} show_atoms={args.show_atoms}\n')
    print(f'REVIEW: {summary}')
    return 0


def build_parser():
    p = argparse.ArgumentParser(description='Assembly → AFM imaging (Morse PP-AFM)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--xyz', help='Assembly or molecule .xyz')
    p.add_argument('--preset', choices=list(PRESETS), help='Run assembly search then AFM on best export')
    p.add_argument('--outdir', default='debug/helicene_afm_pipeline')
    p.add_argument('--method', choices=['contact-sep', 'contact-pic', 'morse-3d'], default='contact-sep',
                   help='contact-sep=separable 2.5D; contact-pic=PIC 2.5D; morse-3d=full img_FF')
    p.add_argument('--cell', default=HELICENE_CELL, help='Unit cell lvs string (overlay)')
    p.add_argument('--n-mols', type=int, default=6, help='Molecules per cell (bond replication)')
    p.add_argument('--margin', type=float, default=3.0, help='Contact-surface fit margin [Å]')
    p.add_argument('--bspl-dx', type=float, default=0.2, help='Separable B-spline xy step [Å]')
    p.add_argument('--n-iter', type=int, default=60, help='Contact-surface fit CG iterations')
    p.add_argument('--pic-reg', type=float, default=1e-2, help='PIC regularization')
    p.add_argument('--scan-dx', type=float, default=0.15, help='Lateral scan step [Å]')
    p.add_argument('--scan-margin', type=float, default=3.0)
    p.add_argument('--nz-scan', type=int, default=40)
    p.add_argument('--dtip', type=float, default=-0.1)
    p.add_argument('--z-clearance', type=float, default=5.0)
    p.add_argument('--amp', type=float, default=1.0, help='Oscillation amplitude for df [Å]; 0=simple gradient')
    p.add_argument('--wrap-cell', action=argparse.BooleanOptionalAction, default=True,
                   help='Wrap molecules into crystallographic cell [0,1)×[0,1) before AFM')
    p.add_argument('--ff-pbc', type=int, default=1, dest='ff_pbc',
                   help='PBC shells of atoms for forcefield (1 → 3×3 cells); overlays still show primary cell')
    p.add_argument('--show-cell', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--show-bonds', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--show-atoms', action=argparse.BooleanOptionalAction, default=True,
                   help='Overlay atom dots so unit-cell occupancy is obvious')
    p.add_argument('--bond-lw', type=float, default=0.35)
    p.add_argument('--bond-alpha', type=float, default=0.22)
    p.add_argument('--cell-pbc', type=int, default=1, help='Draw dashed periodic cell replicas (0=off)')
    p.add_argument('--sites', choices=['com', 'df_min'], default='com',
                   help='com=molecule COM (M0..); df_min=darkest df pixels (S1..)')
    p.add_argument('--n-spots', type=int, default=6, help='Max sites for legend / profiles')
    p.add_argument('--profile-iz', type=int, default=25, help='df slice index for df_min site picking')
    p.add_argument('--profile-h-max', type=float, default=9.0, help='Dense profile: max h_probe [Å]')
    p.add_argument('--profile-h-min', type=float, default=0.8, help='Dense profile: min h_probe [Å]')
    p.add_argument('--profile-dh', type=float, default=0.05, help='Dense profile height step [Å]')
    p.add_argument('--nrot', type=int, default=8)
    p.add_argument('--n-tilt', type=int, default=3)
    p.add_argument('--nshift', type=int, default=6)
    p.add_argument('--n-pbc-xyz', type=int, default=0, dest='n_pbc_xyz')
    p.add_argument('--n-sym', type=int, default=6)
    p.add_argument('--export-max', type=int, default=5)
    p.add_argument('--top-k', type=int, default=3)
    return p


def main(argv=None) -> int:
    return run_afm(build_parser().parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
