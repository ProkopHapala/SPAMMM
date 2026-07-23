#!/usr/bin/env python3
"""SPAMMM SPM CLI — AFM / STM imaging from the repo root (no GUI).

User entry point for density-based AFM (FDBM), Morse+Coulomb AFM, Kriging GridFF AFM,
and STM/orbital imaging (DFTB vs pySCF). Physics and plotting live in ``spammm.SPM``.

Docs: user_guide/SPM_CLI.md

Examples:
  python run_spm.py afm --xyz data/xyz/benzene.xyz --basis 3ob-3-1 --projection stock
  python run_spm.py afm-morse --xyz data/xyz/pentacene.xyz
  python run_spm.py afm-kriging --endgroup HHO-h-p_1 --tip H2O_O
  python run_spm.py stm orbitals --molecule pentacene --n-near 5
  python run_spm.py stm current --molecule pentacene --stm-tips s,pz,py
  python run_spm.py stm panel --molecule pentacene,PTCDA
  python run_spm.py panel-fukui --molecule PTCDA pentacene
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _abs_path(p: str | None) -> str | None:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def _add_common_afm_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group('geometry / density')
    g.add_argument('--xyz',      default='data/xyz/benzene.xyz',  help='Sample geometry (.xyz)')
    g.add_argument('--cube',     default=None,                    help='Sample density cube or directory')
    g.add_argument('--esp-cube', default=None,                    help='Optional ESP cube')

    b = p.add_argument_group('DFTB basis / projection')
    b.add_argument('--basis',      default='3ob-3-1',    choices=['3ob-3-1', 'mio-1-1'])
    b.add_argument('--projection', default='stock',      choices=['stock', 'prolonged', 'both'])
    b.add_argument('--tip-mode',   default='co',         choices=['co', 'gaussian'])

    grid = p.add_argument_group('grid')
    grid.add_argument('--step',        type=float, default=0.15)
    grid.add_argument('--margin',      type=float, default=4.0)
    grid.add_argument('--z-extra',     type=float, default=6.0)
    grid.add_argument('--cpu-fft',     type=bool,  default=True,  dest='cpu_fft')
    grid.add_argument('--gpu-fft',     type=bool,  default=False)

    scan = p.add_argument_group('PP scan / df')
    scan.add_argument('--h-min',       type=float, default=2.5)
    scan.add_argument('--h-max',       type=float, default=5.7)
    scan.add_argument('--h-step',      type=float, default=0.4)
    scan.add_argument('--amp',         type=float, default=1.0)
    scan.add_argument('--K-LAT',       type=float, default=0.5,  dest='K_LAT')
    scan.add_argument('--K-RAD',       type=float, default=20.0,   dest='K_RAD')
    scan.add_argument('--bond-length', type=float, default=3.0)
    scan.add_argument('--scan-margin', type=float, default=2.0)

    out = p.add_argument_group('output / plotting')
    out.add_argument('--outdir',             default='debug/spm_afm')
    out.add_argument('--cmap',               default='seismic')
    out.add_argument('--df-cmap',            default='gray', dest='df_cmap')
    out.add_argument('--height', type=float, default=2.5)
    out.add_argument('--scale',    default='per_column', choices=['per_image', 'per_column', 'common'])


def cmd_afm(args: argparse.Namespace) -> int:
    """Single-molecule FDBM AFM (xyz and/or cube)."""
    import numpy as np
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import (
        parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list,
    )
    from tests.SPM import testplot_fdbm_relax as diag

    if args.gpu_fft:
        os.environ.pop('SPAMMM_AFM_CPU_FFT', None)
    else:
        os.environ['SPAMMM_AFM_CPU_FFT'] = '1'

    os.makedirs(args.outdir, exist_ok=True)
    xyz = _abs_path(args.xyz)
    ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'Br': 35, 'I': 53}
    import spammm.atomicUtils as au
    pos, _, names, _, _ = au.load_xyz(xyz)
    atomPos = np.array(pos, dtype=np.float64)
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in names], dtype=np.int32)

    d_cube = None
    if args.cube:
        d_cube = afm_utils.get_density_from_cube(
            _abs_path(args.cube), esp_path=_abs_path(args.esp_cube), use_esp_cube=False, verbosity=0)
        atomPos = np.asarray(d_cube['atomPos'], dtype=np.float64)
        atomTypes = np.array([int(round(z)) for z in d_cube['atomZ']], dtype=np.int32)

    grid_spec, origin, ngrid, step = diag._fft_friendly_grid_spec(
        atomPos, args.step, args.margin, z_extra=args.z_extra)
    nx, ny, nz = [int(x) for x in ngrid]
    print(f'grid={nx}x{ny}x{nz} step={step} origin={origin}')

    variants = {}
    basis_hsd = get_dftb_basis_path(args.basis)
    pa_3ob = afm.PAULI_FITTED_DEFAULTS.get(args.basis, afm.PAULI_FITTED_DEFAULTS['3ob-3-1'])
    pa_cube = afm.PAULI_FITTED_DEFAULTS.get('pyscf_6-31g*', {'A': 40.0, 'beta': 1.15})

    V_ES_stock = None
    need_stock_es = args.projection in ('stock', 'both', 'prolonged') or not args.cube
    if need_stock_es and (args.projection in ('stock', 'both', 'prolonged')):
        work = os.path.join(args.outdir, 'dftb_work_stock')
        print('DFTB stock density...')
        res = afm_utils.get_density_from_dftb_dense(
            atomPos, atomTypes, basis_hsd, work, grid_spec=grid_spec, step=step, verbosity=0)
        V_ES_stock = res['V_ES']
        if args.projection in ('stock', 'both'):
            variants['stock'] = diag._run_from_density(
                'stock', res['rho_scf'], V_ES_stock, atomPos, atomTypes, origin, step, ngrid,
                pa_3ob['A'], pa_3ob['beta'], args.tip_mode, args.outdir, args)

    if args.projection in ('prolonged', 'both'):
        basis_data = parse_wfc_hsd(basis_hsd)
        basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
        prol = make_slater_tail_species_list(basis_ang)
        work = os.path.join(args.outdir, 'dftb_work_prolonged')
        print('DFTB prolonged (Pauli ρ; ES=stock)...')
        if V_ES_stock is None:
            res0 = afm_utils.get_density_from_dftb_dense(
                atomPos, atomTypes, basis_hsd, os.path.join(args.outdir, 'dftb_work_stock'),
                grid_spec=grid_spec, step=step, verbosity=0)
            V_ES_stock = res0['V_ES']
        res_p = afm_utils.get_density_from_dftb_dense(
            atomPos, atomTypes, basis_hsd, work, grid_spec=grid_spec, step=step,
            verbosity=0, projection_basis_ang=prol)
        variants['prolonged'] = diag._run_from_density(
            'prolonged', res_p['rho_scf'], V_ES_stock, atomPos, atomTypes, origin, step, ngrid,
            pa_3ob['A'], pa_3ob['beta'], args.tip_mode, args.outdir, args)

    if d_cube is not None:
        rho_g = afm_utils.resample_field_to_grid(
            d_cube['rho_scf'], d_cube['origin'], d_cube['step'], origin, step, ngrid)
        rho_d = afm_utils.resample_field_to_grid(
            d_cube['rho_diff'], d_cube['origin'], d_cube['step'], origin, step, ngrid)
        dV = step ** 3
        vol = float(nx * ny * nz) * dV
        qd = float(rho_d.sum() * dV)
        if abs(qd) > 1e-4:
            rho_d = (rho_d - qd / vol).astype(np.float32)
        V_cube = afm.fft_poisson_cpu(rho_d, step)
        variants['cube'] = diag._run_from_density(
            'cube', rho_g, V_cube, atomPos, atomTypes, origin, step, ngrid,
            pa_cube['A'], pa_cube['beta'], args.tip_mode, args.outdir, args)

    if not variants:
        print('Nothing to run: set --projection and/or --cube', file=sys.stderr)
        return 1

    order = [k for k in ('cube', 'stock', 'prolonged') if k in variants]
    row_specs = []
    for k in order:
        pa = pa_cube if k == 'cube' else pa_3ob
        row_specs.append(('df', k, f'df {k}\nA={pa["A"]:.1f} β={pa["beta"]:.2f}', args.df_cmap))
    for k in order:
        row_specs.append(('Fz', k, f'Fz {k}', args.cmap))

    heights = next(iter(variants.values()))['heights']
    title = f'FDBM AFM  tip={args.tip_mode}  basis={args.basis}  projection={args.projection}'
    out_png = os.path.join(args.outdir, f'compare_{args.scale}.png')
    afm_utils.plot_afm_variant_height_strip(
        variants, row_specs, heights, out_png, scale=args.scale, title=title, dpi=140)
    if args.scale != 'per_image':
        per = os.path.join(args.outdir, 'per_image')
        os.makedirs(per, exist_ok=True)
        afm_utils.plot_afm_variant_height_strip(
            variants, row_specs, heights, os.path.join(per, 'compare_per_image.png'),
            scale='per_image', title=title, dpi=140)

    summary = os.path.join(args.outdir, 'SUMMARY.out')
    with open(summary, 'w') as f:
        f.write(title + '\n')
        f.write(f'xyz={xyz}\n')
        f.write(f'cube={args.cube}\n')
        f.write(f'grid={nx}x{ny}x{nz} step={step}\n')
        f.write(f'REVIEW: {out_png}\n')
    print(f'REVIEW: {summary}')
    print(f'REVIEW: {os.path.abspath(args.outdir)}/')
    return 0


def cmd_afm_morse(args: argparse.Namespace) -> int:
    """Morse + point-charge Coulomb AFM (no electron density)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from spammm.SPM.AFM import AFMulator, compute_df

    os.makedirs(args.outdir, exist_ok=True)
    xyz = _abs_path(args.xyz)
    params = _abs_path(args.params) or os.path.join(_ROOT, 'data', 'ElementTypes.dat')
    afm = AFMulator(use_morse=not args.lj)
    afm.load_molecule(xyz)
    afm.assign_params(params_path=params)
    afm.setup_grid(n=(args.nx, args.ny, args.nz), margin=args.margin, z_top=args.z_top)
    afm.make_forcefield()

    nz_scan, dtip = args.nz_scan, args.dtip
    nxy = (args.scan_nx, args.scan_ny)
    FEs, _pts = afm.run_scan(nxy=nxy, nz=nz_scan, dtip=dtip)
    Fz = FEs[:, :, :, 2]
    df = compute_df(Fz, abs(dtip))
    mol_z = afm.mol_z
    z0_tip = mol_z + 5.0 + abs(float(afm.dpos0[2]))
    heights = z0_tip + np.arange(nz_scan) * dtip - mol_z

    sel = [i for i in args.slice_indices if 0 <= i < nz_scan]
    mol_tag = os.path.splitext(os.path.basename(xyz))[0]
    pot = 'Morse' if afm.use_morse else 'LJ'

    for kind, data, cmap in (('Fz', Fz, 'bwr'), ('df', df, 'bwr')):
        fig, axes = plt.subplots(1, len(sel), figsize=(3 * len(sel), 3))
        if len(sel) == 1:
            axes = [axes]
        for ax, iz in zip(axes, sel):
            arr = data[:, :, iz].T
            vabs = max(float(np.percentile(np.abs(arr), 99)), 1e-6)
            im = ax.imshow(arr, origin='lower', cmap=cmap, aspect='equal', vmin=-vabs, vmax=vabs)
            ax.set_title(f'{kind} h={heights[iz]:.2f}Å', fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f'AFM {kind} ({pot}+Coulomb) — {mol_tag}', fontsize=10)
        fig.tight_layout()
        out_png = os.path.join(args.outdir, f'afm_{kind}_{mol_tag}.png')
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f'REVIEW: {out_png}')

    np.savez(os.path.join(args.outdir, f'afm_morse_{mol_tag}.npz'),
             Fz=Fz, df=df, heights=heights)
    summary = os.path.join(args.outdir, 'SUMMARY.out')
    open(summary, 'w').write(
        f'AFM {pot}+Coulomb  xyz={xyz}\nREVIEW: {args.outdir}/\n')
    print(f'REVIEW: {summary}')
    return 0


def cmd_afm_kriging(args: argparse.Namespace) -> int:
    from tests.SPM import testplot_kriging_relax as kr
    argv = [
        'testplot_kriging_relax.py',
        '--endgroup', args.endgroup,
        '--tip', args.tip,
        '--dx', str(args.dx),
        '--h_min', str(args.h_min),
        '--h_max', str(args.h_max),
        '--h_step', str(args.h_step),
        '--klat', args.klat,
        '--bond_length', str(args.bond_length),
        '--outdir', _abs_path(args.outdir),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        kr.main()
    finally:
        sys.argv = old_argv
    return 0


def cmd_panel_fukui(args: argparse.Namespace) -> int:
    from tests.SPM import testplot_fdbm_relax as diag
    os.environ['SPAMMM_AFM_CPU_FFT'] = '1'
    ns = argparse.Namespace(
        xyz='data/xyz/PTCDA.xyz', basis='3ob-3-1', step=args.step, margin=args.margin,
        tip_mode='co', outdir=args.outdir, K_LAT=args.K_LAT, K_RAD=args.K_RAD,
        bond_length=args.bond_length, h_min=args.h_min, h_max=args.h_max, h_step=args.h_step,
        amp=args.amp, scan_margin=args.scan_margin, height=args.height,
        cmap=args.cmap, df_cmap=args.df_cmap, molecule=args.molecule,
        sa_params='debug/dftb_basis_sa_ptcda/PTCDA_sa_params.json',
    )
    diag.run_fukui_panel(ns)
    return 0


def cmd_replot_panel(args: argparse.Namespace) -> int:
    from tests.SPM import testplot_fdbm_relax as diag
    if args.scale != 'per_image':
        print('Note: replot-panel currently writes per_image strips (experimental contrast).')
    diag.replot_fukui_per_image(
        args.panel_dir, molecules=args.molecule, cmap=args.cmap, df_cmap=args.df_cmap)
    return 0


def _prepare_stm_outdir(args):
    args.outdir = _abs_path(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)


def cmd_stm_orbitals(args: argparse.Namespace) -> int:
    from spammm.SPM import stm_compare as stm
    _prepare_stm_outdir(args)
    for mol_name, pos, names, types, _info in stm.resolve_molecules(args.molecule, xyz=args.xyz):
        stm.run_frontier_orbitals(mol_name, pos, names, types, args)
    return 0


def cmd_stm_current(args: argparse.Namespace) -> int:
    from spammm.SPM import stm_compare as stm
    _prepare_stm_outdir(args)
    for mol_name, pos, names, types, _info in stm.resolve_molecules(args.molecule, xyz=args.xyz):
        stm.run_frontier_stm_current(mol_name, pos, names, types, args)
    return 0


def cmd_stm_panel(args: argparse.Namespace) -> int:
    from spammm.SPM import stm_compare as stm
    _prepare_stm_outdir(args)
    for mol_name, pos, names, types, info in stm.resolve_molecules(args.molecule, xyz=args.xyz):
        stm.run_stm_vacuum_panel(mol_name, pos, names, types, info, args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from spammm.SPM import stm_compare as stm

    p = argparse.ArgumentParser(
        prog='run_spm.py',
        description='SPAMMM SPM CLI — AFM/STM imaging without the GUI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    p_afm = sub.add_parser('afm', help='FDBM AFM from .xyz and/or density .cube')
    _add_common_afm_args(p_afm)
    p_afm.set_defaults(func=cmd_afm)

    p_morse = sub.add_parser('afm-morse', help='Classical Morse/LJ + Coulomb AFM (no density)')
    p_morse.add_argument('--xyz', default='data/xyz/benzene.xyz')
    p_morse.add_argument('--params', default=None, help='ElementTypes.dat path')
    p_morse.add_argument('--lj', action='store_true', help='Use LJ instead of Morse')
    p_morse.add_argument('--margin', type=float, default=4.0)
    p_morse.add_argument('--z-top', type=float, default=14.0)
    p_morse.add_argument('--nx', type=int, default=60, help='FF grid nx')
    p_morse.add_argument('--ny', type=int, default=60, help='FF grid ny')
    p_morse.add_argument('--nz', type=int, default=40, help='FF grid nz')
    p_morse.add_argument('--scan-nx', type=int, default=40)
    p_morse.add_argument('--scan-ny', type=int, default=40)
    p_morse.add_argument('--nz-scan', type=int, default=25, dest='nz_scan', help='Approach steps')
    p_morse.add_argument('--dtip', type=float, default=-0.15)
    p_morse.add_argument('--slice-indices', nargs='+', type=int, default=[0, 5, 10, 15, 20])
    p_morse.add_argument('--outdir', default='debug/spm_afm_morse')
    p_morse.set_defaults(func=cmd_afm_morse)

    p_krig = sub.add_parser('afm-kriging', help='DFT Kriging GridFF → PP-AFM (Mithun data)')
    p_krig.add_argument('--endgroup', default='HHO-h-p_1')
    p_krig.add_argument('--tip', default='H2O_O')
    p_krig.add_argument('--dx', type=float, default=0.1)
    p_krig.add_argument('--klat', default='0.5,1.0,2.0')
    p_krig.add_argument('--bond-length', type=float, default=4.0)
    p_krig.add_argument('--h-min', type=float, default=3.5)
    p_krig.add_argument('--h-max', type=float, default=5.5)
    p_krig.add_argument('--h-step', type=float, default=0.2)
    p_krig.add_argument('--outdir', default='debug/spm_afm_kriging')
    p_krig.set_defaults(func=cmd_afm_kriging)

    p_panel = sub.add_parser('panel-fukui', help='Fukui cube panel: cube vs stock vs prolonged')
    _add_common_afm_args(p_panel)
    p_panel.add_argument('--molecule', nargs='*', default=None)
    p_panel.set_defaults(func=cmd_panel_fukui, outdir='debug/fdbm_fukui_panel')

    p_rep = sub.add_parser('replot-panel', help='Replot saved panel npz (per-image contrast)')
    p_rep.add_argument('--panel-dir', default='debug/fdbm_fukui_panel')
    p_rep.add_argument('--molecule', nargs='*', default=None)
    p_rep.add_argument('--scale', default='per_image', choices=['per_image', 'per_column', 'common'])
    p_rep.add_argument('--cmap', default='seismic')
    p_rep.add_argument('--df-cmap', default='gray', dest='df_cmap')
    p_rep.set_defaults(func=cmd_replot_panel)

    p_stm = sub.add_parser('stm', help='STM / orbital imaging (DFTB vs pySCF)')
    stm_sub = p_stm.add_subparsers(dest='stm_mode', required=True)

    p_orb = stm_sub.add_parser('orbitals', help='Frontier MO ψ maps (signed phase) + spectrum')
    stm.add_stm_common_args(p_orb)
    stm.add_orbital_args(p_orb)
    p_orb.set_defaults(func=cmd_stm_orbitals)

    p_cur = stm_sub.add_parser('current', help='MO-resolved STM current I≥0 + spectrum')
    stm.add_stm_common_args(p_cur)
    stm.add_stm_current_args(p_cur)
    p_cur.set_defaults(func=cmd_stm_current)

    p_stmp = stm_sub.add_parser('panel', help='HOMO/LUMO vacuum STM panel (stock/prolonged/pySCF)')
    stm.add_stm_common_args(p_stmp)
    stm.add_panel_args(p_stmp)
    p_stmp.set_defaults(func=cmd_stm_panel)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
