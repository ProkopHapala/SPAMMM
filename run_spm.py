#!/usr/bin/env python3
"""SPAMMM SPM CLI — AFM / STM imaging from the repo root (no GUI).

User entry point for density-based AFM (FDBM), Morse+Coulomb AFM, Kriging GridFF AFM,
and STM/orbital imaging (DFTB vs pySCF). Physics and plotting live in ``spammm.SPM``.

Docs: user_guide/SPM_CLI.md

Examples:
  python run_spm.py afm --xyz data/xyz/benzene.xyz --basis 3ob-3-1 --projection stock
  python run_spm.py afm --smiles-example naphthalene --projection prolonged --show-atoms
  python run_spm.py opt --smiles-example benzene --method uff
  python run_spm.py smiles-afm --method uff          # all SMILES_EXAMPLES → planar opt → prolonged AFM
  python run_spm.py afm-morse --xyz data/xyz/pentacene.xyz
  python run_spm.py afm-kriging --endgroup HHO-h-p_1 --tip H2O_O
  python run_spm.py stm orbitals --molecule pentacene --n-near 5
  python run_spm.py stm current --molecule pentacene --stm-tips s,pz,py
  python run_spm.py stm panel --molecule pentacene,PTCDA
  python run_spm.py stm br --xyz data/xyz/PTCDA.xyz --show-atoms
  python run_spm.py panel-fukui --molecule PTCDA pentacene
  python run_spm.py basis-tails --molecule pentacene,PTCDA
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


def _parse_plots(s: str | None) -> set[str]:
    """Parse --plots CSV: compare,stage (default) | tip,df,fz,per_image | all | none."""
    if s is None:
        s = 'compare,stage'
    parts = [p.strip().lower() for p in str(s).replace(';', ',').split(',') if p.strip()]
    if not parts or 'none' in parts or 'off' in parts:
        return set()
    if 'all' in parts or 'debug' in parts:
        return {'compare', 'stage', 'tip', 'df', 'fz', 'per_image'}
    return set(parts)


def _add_common_afm_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group('geometry / density')
    g.add_argument('--xyz',      default=None,  help='Sample geometry (.xyz); default benzene if no SMILES')
    g.add_argument('--smiles',   default=None,  help='SMILES string (alternative to --xyz)')
    g.add_argument('--smiles-example', default=None, dest='smiles_example',
                   help='Named SMILES from SMILES_EXAMPLES (e.g. benzene, naphthalene)')
    g.add_argument('--cube',     default=None,                    help='Sample density cube or directory')
    g.add_argument('--esp-cube', default=None,                    help='Optional ESP cube')
    g.add_argument('--no-orient', action='store_true', dest='no_orient',
                   help='Skip PCA long-axis→x orientation (default: orient)')
    g.add_argument('--no-planar', action='store_true', dest='no_planar',
                   help='Keep 3D sample geometry (default: force all atom z=0 for AFM)')

    b = p.add_argument_group('DFTB basis / projection')
    b.add_argument('--basis',      default='3ob-3-1',    choices=['3ob-3-1', 'mio-1-1'])
    b.add_argument('--projection', default='prolonged', choices=['stock', 'prolonged', 'both'],
                   help='Pauli ρ basis (GUI/BR-STM SSOT = prolonged; ES always stock Δρ)')
    b.add_argument('--tip-mode',   default='co',         choices=['co', 'gaussian'])

    grid = p.add_argument_group('grid')
    grid.add_argument('--step',        type=float, default=0.1,
                      help='Density/FF grid spacing [Å] (GUI SSOT = 0.1)')
    grid.add_argument('--margin',      type=float, default=4.0)
    grid.add_argument('--z-extra',     type=float, default=6.0)
    grid.add_argument('--cpu-fft', action='store_true', dest='cpu_fft',
                      help='Force NumPy FFT Stage-3 (slow; parity). Default: FAST_S3 GPU.')
    grid.add_argument('--gpu-fft', action='store_true', dest='gpu_fft',
                      help='Deprecated no-op (GPU FAST_S3 is already the default).')

    scan = p.add_argument_group('PP scan / df')
    # Display window = df probe heights; Fz panels use h−amp (amp-align) by default
    scan.add_argument('--h-min', '--zmin', type=float, default=3.7, dest='h_min',
                      help='df probe z_min [Å] (column labels; Fz shown at h−amp)')
    scan.add_argument('--h-max', '--zmax', type=float, default=4.7, dest='h_max',
                      help='df probe z_max [Å]')
    scan.add_argument('--h-step', '--dz',  type=float, default=0.1, dest='h_step',
                      help='Height step dz [Å]')
    scan.add_argument('--amp',         type=float, default=1.0)
    scan.add_argument('--no-amp-align', action='store_true', dest='no_amp_align',
                      help='Plot Fz at same h as df (default: Fz at h−amp for fair morph. match)')
    scan.add_argument('--K-LAT',       type=float, default=0.5,  dest='K_LAT')
    scan.add_argument('--K-RAD',       type=float, default=20.0,   dest='K_RAD')
    scan.add_argument('--bond-length', type=float, default=3.0)
    scan.add_argument('--scan-margin', type=float, default=2.0)

    out = p.add_argument_group('output / plotting')
    out.add_argument('--outdir',             default='debug/spm_afm')
    out.add_argument('--cmap',               default='seismic')
    out.add_argument('--df-cmap',            default='gray', dest='df_cmap')
    out.add_argument('--height', type=float, default=4.2,
                     help='Stage-plot slice height above mol [Å]')
    out.add_argument('--scale',    default='per_image', choices=['per_image', 'per_column', 'common'],
                     help='Color scale (default per_image; df always per-panel clim)')
    out.add_argument('--plots', default='compare,stage',
                     help='CSV: compare,stage (default); tip,df,fz,per_image; all; none')
    out.add_argument('--show-atoms', action='store_true', dest='show_atoms',
                     help='Overlay atom positions as small dots on AFM panels')


def _resolve_geometry(args):
    """Return (name, atomPos, atomTypes, enames, bonds_or_None, smiles_or_None) from CLI args."""
    import numpy as np
    from spammm.topology.smiles import SMILES_EXAMPLES, smiles_to_system, parse_smiles
    ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53}

    smi = getattr(args, 'smiles', None)
    ex = getattr(args, 'smiles_example', None)
    if ex:
        if ex not in SMILES_EXAMPLES:
            raise SystemExit(f"Unknown --smiles-example {ex!r}; choose from {sorted(SMILES_EXAMPLES)}")
        smi = SMILES_EXAMPLES[ex]
        name = ex
    elif smi:
        name = 'smiles'
    else:
        smi = None
        name = None

    if smi is not None:
        sys = smiles_to_system(smi, engine='pure')
        atomPos = np.asarray(sys.apos, dtype=np.float64)
        enames = list(sys.enames)
        atomTypes = np.array([int(z) for z in sys.atypes], dtype=np.int32)
        bonds = np.asarray(sys.bonds, dtype=np.int32) if sys.bonds is not None else None
        if name is None:
            name = 'smiles'
        return name, atomPos, atomTypes, enames, bonds, smi

    xyz = _abs_path(getattr(args, 'xyz', None) or 'data/xyz/benzene.xyz')
    import spammm.atomicUtils as au
    pos, _, names, _, _ = au.load_xyz(xyz)
    atomPos = np.array(pos, dtype=np.float64)
    enames = list(names)
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in enames], dtype=np.int32)
    name = os.path.splitext(os.path.basename(xyz))[0]
    return name, atomPos, atomTypes, enames, None, None


def cmd_afm(args: argparse.Namespace) -> int:
    """Single-molecule FDBM AFM (xyz and/or cube / SMILES).

    Physics: ``AFM_utils.run_fdbm_pp_from_density`` (FAST_S3 GPU = ModularPipeline Stage3–4).
    Plotting: ``plot_afm_variant_height_strip`` (skill:`afm-plotting` SSOT).
    No dependency on ``tests/SPM/testplot_fdbm_relax``.
    """
    import numpy as np
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    from spammm.quantum.DFTB.DFTBplusParser import (
        parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list,
    )

    use_fast = not bool(getattr(args, 'cpu_fft', False))
    if use_fast:
        os.environ.pop('SPAMMM_AFM_CPU_FFT', None)
    else:
        os.environ['SPAMMM_AFM_CPU_FFT'] = '1'

    os.makedirs(args.outdir, exist_ok=True)
    plots = _parse_plots(getattr(args, 'plots', 'compare,stage'))
    mol_name, atomPos, atomTypes, enames, bonds, smi = _resolve_geometry(args)
    xyz = _abs_path(args.xyz) if args.xyz else None

    d_cube = None
    if args.cube:
        d_cube = afm_utils.get_density_from_cube(
            _abs_path(args.cube), esp_path=_abs_path(args.esp_cube), use_esp_cube=False, verbosity=0)
        atomPos = np.asarray(d_cube['atomPos'], dtype=np.float64)
        atomTypes = np.array([int(round(z)) for z in d_cube['atomZ']], dtype=np.int32)

    # Default: perfectly flat sample (z=0). Skip when density comes from a cube
    # (atoms must stay aligned with the cube) or when --no-planar is set.
    do_planar = (d_cube is None) and (not getattr(args, 'no_planar', False))
    if do_planar:
        from spammm.forcefields.FFController import make_planar_xy
        atomPos[:] = make_planar_xy(atomPos)
        print(f'planarize → z=0  zspan={atomPos[:,2].ptp():.3e}Å')
    if not getattr(args, 'no_orient', False):
        from spammm.forcefields.FFController import orient_long_axis_x
        orient_long_axis_x(atomPos)
        if do_planar:
            atomPos[:, 2] = 0.0
        print(f'orientPCA long→x  span_xy=({atomPos[:,0].ptp():.3f},{atomPos[:,1].ptp():.3f})'
              f'  zspan={atomPos[:,2].ptp():.3e}Å')

    z_vac = float(args.z_extra) if args.z_extra is not None else 6.0
    grid_spec, origin, ngrid, step = afm_utils.make_fdbm_grid_com_zsym(
        atomPos, args.step, args.margin, z_vac=z_vac)
    nx, ny, nz = [int(x) for x in ngrid]
    print(f'grid={nx}x{ny}x{nz} step={step} origin={origin} mol={mol_name}  '
          f'Stage3={"FAST_S3" if use_fast else "LEGACY_CPU_FFT"}')

    variants = {}
    basis_hsd = get_dftb_basis_path(args.basis)
    pa = afm.PAULI_FITTED_DEFAULTS.get(args.basis, afm.PAULI_FITTED_DEFAULTS['3ob-3-1'])
    A_pauli, beta_pauli = float(pa['A']), float(pa['beta'])
    print(f'Pauli EVAL defaults ({args.basis}): A={A_pauli:.3f} β={beta_pauli:.4f}')

    plot_diag = plots & {'tip', 'stage', 'df', 'fz'}

    def _pp(tag, rho_scf, rho_diff, V_ES=None):
        return afm_utils.run_fdbm_pp_from_density(
            tag, rho_scf, atomPos, atomTypes, origin, step, ngrid,
            A_pauli, beta_pauli, args.tip_mode, args.outdir,
            rho_diff=rho_diff, V_ES=V_ES,
            basis=args.basis, margin=args.margin,
            h_min=args.h_min, h_max=args.h_max, h_step=args.h_step,
            amp=args.amp, amp_align=not getattr(args, 'no_amp_align', False),
            K_LAT_Nm=args.K_LAT, K_RAD=args.K_RAD, bond_length=args.bond_length,
            scan_margin=args.scan_margin, plots=plot_diag,
            df_cmap=args.df_cmap, cmap=args.cmap, stage_height=args.height,
            use_fast_s3=use_fast,
        )

    V_ES_stock = None
    rho_diff_stock = None
    need_stock_es = args.projection in ('stock', 'both', 'prolonged') or not args.cube
    if need_stock_es and (args.projection in ('stock', 'both', 'prolonged')):
        work = os.path.join(args.outdir, 'dftb_work_stock')
        print('DFTB stock density...')
        res = afm_utils.get_density_from_dftb_dense(
            atomPos, atomTypes, basis_hsd, work, grid_spec=grid_spec, step=step, verbosity=0)
        V_ES_stock = res['V_ES']
        rho_diff_stock = res['rho_diff']
        if args.projection in ('stock', 'both'):
            variants['stock'] = _pp('stock', res['rho_scf'], rho_diff_stock, V_ES_stock)

    if args.projection in ('prolonged', 'both'):
        basis_data = parse_wfc_hsd(basis_hsd)
        basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
        prol = make_slater_tail_species_list(basis_ang)
        work = os.path.join(args.outdir, 'dftb_work_prolonged')
        print('DFTB prolonged (Pauli ρ; ES=stock Δρ)...')
        if rho_diff_stock is None:
            res0 = afm_utils.get_density_from_dftb_dense(
                atomPos, atomTypes, basis_hsd, os.path.join(args.outdir, 'dftb_work_stock'),
                grid_spec=grid_spec, step=step, verbosity=0)
            V_ES_stock = res0['V_ES']
            rho_diff_stock = res0['rho_diff']
        res_p = afm_utils.get_density_from_dftb_dense(
            atomPos, atomTypes, basis_hsd, work, grid_spec=grid_spec, step=step,
            verbosity=0, projection_basis_ang=prol)
        variants['prolonged'] = _pp('prolonged', res_p['rho_scf'], rho_diff_stock, V_ES_stock)

    if d_cube is not None:
        prep = afm_utils.allelectron_cube_to_fdbm_grid(
            d_cube['rho_scf'], d_cube['origin'], d_cube['step'],
            d_cube['atomPos'], d_cube['atomZ'],
            origin, step, ngrid, rc_na=0.6, R_sphere=0.6, verbosity=0)
        V_cube = afm.fft_poisson_cpu(prep['rho_diff'], step) if not use_fast else None
        variants['cube'] = _pp('cube', prep['rho_scf'], prep['rho_diff'], V_cube)

    if not variants:
        print('Nothing to run: set --projection and/or --cube', file=sys.stderr)
        return 1

    order = [k for k in ('cube', 'prolonged', 'stock') if k in variants]
    amp_align = not getattr(args, 'no_amp_align', False)
    amp = float(args.amp)
    row_specs = []
    for k in order:
        row_specs.append(('df', k, f'df {k}\nA={A_pauli:.1f} β={beta_pauli:.2f}', args.df_cmap))
    for k in order:
        fz_lab = f'Fz {k}\n@h−{amp:.1f}Å' if amp_align else f'Fz {k}'
        row_specs.append(('Fz', k, fz_lab, args.cmap))

    heights = next(iter(variants.values()))['heights']
    v0 = next(iter(variants.values()))
    extent = None
    if 'scan_xs' in v0 and 'scan_ys' in v0:
        extent = afm_utils.scan_extent(v0['scan_xs'], v0['scan_ys'])
    title = f'FDBM AFM  {mol_name}  tip={args.tip_mode}  basis={args.basis}  projection={args.projection}'
    show_atoms = bool(getattr(args, 'show_atoms', False))
    out_png = None
    if 'compare' in plots:
        scale = getattr(args, 'scale', 'per_image') or 'per_image'
        out_png = os.path.join(args.outdir, f'compare_{scale}.png')
        afm_utils.plot_afm_variant_height_strip(
            variants, row_specs, heights, out_png, scale=scale, title=title, dpi=140,
            apos=atomPos if show_atoms else None, show_atoms=show_atoms, extent=extent,
            amp=args.amp, amp_align=amp_align, long_axis_vertical=True, tight=True)
        print(f'REVIEW: {out_png}')
    if 'per_image' in plots and getattr(args, 'scale', 'per_image') != 'per_image':
        per = os.path.join(args.outdir, 'per_image')
        os.makedirs(per, exist_ok=True)
        pi = os.path.join(per, 'compare_per_image.png')
        afm_utils.plot_afm_variant_height_strip(
            variants, row_specs, heights, pi, scale='per_image', title=title, dpi=140,
            apos=atomPos if show_atoms else None, show_atoms=show_atoms, extent=extent,
            amp=args.amp, amp_align=amp_align, long_axis_vertical=True, tight=True)
        print(f'REVIEW: {pi}')

    summary = os.path.join(args.outdir, 'SUMMARY.out')
    with open(summary, 'w') as f:
        f.write(title + '\n')
        f.write(f'xyz={xyz}\n')
        f.write(f'smiles={smi}\n')
        f.write(f'cube={args.cube}\n')
        f.write(f'grid={nx}x{ny}x{nz} step={step} Stage3={"FAST_S3" if use_fast else "LEGACY"}\n')
        f.write(f'plots={sorted(plots)} h=[{args.h_min},{args.h_max}] dz={args.h_step}\n')
        if out_png:
            f.write(f'REVIEW: {out_png}\n')
        for k, v in variants.items():
            if v.get('stage_path'):
                f.write(f'REVIEW: {v["stage_path"]}\n')
    print(f'REVIEW: {summary}')
    print(f'REVIEW: {os.path.abspath(args.outdir)}/')
    return 0


def cmd_opt(args: argparse.Namespace) -> int:
    """Vacuum geometry optimization (UFF / SPFF / LFF / DFTB), optional planarize."""
    import numpy as np
    from spammm.AtomicSystem import AtomicSystem
    from spammm.forcefields.FFController import optimize_vacuum
    from spammm import atomicUtils as au
    from spammm.topology.smiles import smiles_to_system, SMILES_EXAMPLES

    os.makedirs(args.outdir, exist_ok=True)
    if args.smiles_example:
        smi = SMILES_EXAMPLES[args.smiles_example]
        name = args.smiles_example
        mol = smiles_to_system(smi, engine='pure')
    elif args.smiles:
        smi = args.smiles
        name = 'smiles'
        mol = smiles_to_system(smi, engine='pure')
    else:
        xyz = _abs_path(args.xyz or 'data/xyz/benzene.xyz')
        name = os.path.splitext(os.path.basename(xyz))[0]
        mol = AtomicSystem(fname=xyz)
        if mol.bonds is None:
            mol.findBonds()
        smi = None

    work = os.path.join(args.outdir, f'dftb_opt_{name}') if args.method == 'dftb' else args.outdir
    info = optimize_vacuum(mol, method=args.method, nsteps=args.nsteps, fmax_tol=args.fmax,
                           planar=not args.no_planar, orient_pca=not getattr(args, 'no_orient', False),
                           workdir=work, sk_set=args.basis, verbose=True)
    out_xyz = os.path.join(args.outdir, f'{name}_opt.xyz')
    au.save_xyz(out_xyz, mol.enames, mol.apos, comment=f"opt method={args.method} {info}")
    summary = os.path.join(args.outdir, 'SUMMARY.out')
    with open(summary, 'w') as f:
        f.write(f'opt {name} method={args.method}\n')
        f.write(f'smiles={smi}\n')
        for k, v in info.items():
            f.write(f'{k}={v}\n')
        f.write(f'REVIEW: {out_xyz}\n')
    print(f'REVIEW: {out_xyz}')
    print(f'REVIEW: {summary}')
    return 0


def cmd_smiles_afm(args: argparse.Namespace) -> int:
    """SMILES → vacuum opt (planar) → prolonged FDBM AFM with atom-dot overlay."""
    import numpy as np
    from spammm.topology.smiles import SMILES_EXAMPLES, smiles_to_system
    from spammm.forcefields.FFController import optimize_vacuum
    from spammm import atomicUtils as au

    names = args.example if args.example else list(SMILES_EXAMPLES.keys())
    for name in names:
        if name not in SMILES_EXAMPLES:
            print(f'SKIP unknown example {name!r}', file=sys.stderr)
            continue
        outdir = os.path.join(args.outdir, name)
        os.makedirs(outdir, exist_ok=True)
        print(f'\n======== SMILES-AFM {name} ========')
        mol = smiles_to_system(SMILES_EXAMPLES[name], engine='pure')
        work = os.path.join(outdir, 'dftb_opt') if args.method == 'dftb' else outdir
        info = optimize_vacuum(mol, method=args.method, nsteps=args.nsteps, fmax_tol=args.fmax,
                               planar=True, orient_pca=not getattr(args, 'no_orient', False),
                               workdir=work, sk_set=args.basis, verbose=True)
        xyz_path = os.path.join(outdir, f'{name}_opt.xyz')
        au.save_xyz(xyz_path, mol.enames, mol.apos,
                    comment=f"{name} {SMILES_EXAMPLES[name]} opt={args.method} {info}")
        print(f'REVIEW: {xyz_path}')

        ns = argparse.Namespace(**vars(args))
        ns.xyz = xyz_path
        ns.smiles = None
        ns.smiles_example = None
        ns.outdir = outdir
        ns.projection = 'prolonged'
        ns.show_atoms = True
        ns.cube = None
        ns.esp_cube = None
        # Already PCA-oriented in optimize_vacuum; skip second pass
        ns.no_orient = True
        rc = cmd_afm(ns)
        if rc != 0:
            return rc
    print(f'\nREVIEW: {os.path.abspath(args.outdir)}/')
    return 0


def cmd_afm_morse(args: argparse.Namespace) -> int:
    """Morse + point-charge Coulomb AFM (no electron density).

    Shared backend with GUI: ``AFM_utils.run_morse_coulomb_afm`` (no fork).
    """
    from spammm.SPM import AFM_utils as afm_utils

    os.makedirs(args.outdir, exist_ok=True)
    xyz = _abs_path(args.xyz)
    params = _abs_path(args.params) or os.path.join(_ROOT, 'data', 'ElementTypes.dat')
    afm_utils.run_morse_coulomb_afm(
        xyz, args.outdir,
        params_path=params, use_morse=not args.lj,
        n=(args.nx, args.ny, args.nz), margin=args.margin, z_top=args.z_top,
        nxy=(args.scan_nx, args.scan_ny), nz_scan=args.nz_scan, dtip=args.dtip,
        slice_indices=list(args.slice_indices), save_png=True,
    )
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
    """Fukui cube vs DFTB stock vs prolonged — same height SSOT as ``afm``.

    Physics: ``AFM_utils.run_fukui_panel`` → ``run_fdbm_pp_from_density`` (FAST_S3 default).
    Plotting: ``plot_afm_variant_height_strip`` (skill:`afm-plotting` SSOT).
    No dependency on ``tests/SPM/testplot_fdbm_relax``.
    """
    from spammm.SPM import AFM_utils as afm_utils

    use_fast = not bool(getattr(args, 'cpu_fft', False))
    if use_fast:
        os.environ.pop('SPAMMM_AFM_CPU_FFT', None)
    else:
        os.environ['SPAMMM_AFM_CPU_FFT'] = '1'
    afm_utils.run_fukui_panel(
        args.outdir, molecules=args.molecule, use_fast_s3=use_fast,
        step=args.step, margin=args.margin, basis='3ob-3-1', tip_mode='co',
        h_min=args.h_min, h_max=args.h_max, h_step=args.h_step,
        amp=args.amp, amp_align=not getattr(args, 'no_amp_align', False),
        K_LAT=args.K_LAT, K_RAD=args.K_RAD, bond_length=args.bond_length,
        scan_margin=args.scan_margin, height=args.height,
        cmap=args.cmap, df_cmap=args.df_cmap,
    )
    return 0


def cmd_replot_panel(args: argparse.Namespace) -> int:
    from spammm.SPM import AFM_utils as afm_utils
    if args.scale != 'per_image':
        print('Note: replot-panel currently writes per_image strips (experimental contrast).')
    afm_utils.replot_fukui_per_image(
        args.panel_dir, molecules=args.molecule, cmap=args.cmap, df_cmap=args.df_cmap)
    return 0


def cmd_es_diag(args: argparse.Namespace) -> int:
    """Cube ES chain diagnostics: ρ, Δρ, V_ES=Poisson(Δρ), E_ES, tip + mirror metrics."""
    from tests.SPM import testplot_fdbm_relax as diag
    os.environ['SPAMMM_AFM_CPU_FFT'] = '1'
    ns = argparse.Namespace(
        step=args.step, margin=args.margin, outdir=args.outdir, basis=args.basis,
        molecule=args.molecule, z_above=tuple(args.z_above),
    )
    diag.run_fukui_es_diag(ns)
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


def cmd_stm_fgr(args: argparse.Namespace) -> int:
    """Overlap_exp vs long-tail FGR I_S / I_H / I_τ (Level-B EH tables)."""
    from spammm.SPM import stm_compare as stm
    if args.outdir is None or args.outdir == stm.DEFAULT_OUT:
        args.outdir = os.path.join(_ROOT, 'debug', 'stm_fgr_compare')
    _prepare_stm_outdir(args)
    for mol_name, pos, names, types, info in stm.resolve_molecules(args.molecule, xyz=args.xyz):
        stm.run_fgr_transfer_compare(mol_name, pos, names, types, info, args)
    return 0


def cmd_stm_br_fgr(args: argparse.Namespace) -> int:
    """BR-STM compare: overlap vs FGR (I_S/I_H/I_τ) with PP-AFM tip displacement."""
    import numpy as np
    from spammm.SPM import stm_compare as stm
    from spammm import atomicUtils as au
    if args.outdir is None:
        args.outdir = os.path.join(_ROOT, 'debug', 'stm_br_fgr_compare')
    os.makedirs(args.outdir, exist_ok=True)
    for mol_name in args.molecule:
        xyz = _abs_path(f'data/xyz/{mol_name}.xyz')
        if not os.path.isfile(xyz):
            print(f'  skip {mol_name}: {xyz} not found')
            continue
        pos, _, names, _, _ = au.load_xyz(xyz)
        atomPos = np.asarray(pos, dtype=np.float64)
        # Planarize + center (same as stm br)
        from spammm.AtomicSystem import AtomicSystem
        mol = AtomicSystem(fname=xyz)
        if hasattr(mol, 'orientPCA'):
            mol.orientPCA()
            atomPos = np.asarray(mol.apos, dtype=np.float64)
            names = list(mol.enames)
        atomPos = atomPos.copy()
        atomPos[:, 2] = float(atomPos[:, 2].mean())
        stm.run_br_stm_fgr_compare(mol_name, atomPos, names, args)
    return 0


def cmd_stm_br(args: argparse.Namespace) -> int:
    """Three-stage BR-STM: (1) pure STM, (2) df+|dxy|, (3) STM vs BR-STM."""
    import numpy as np
    from spammm.SPM import AFM_utils as afm_utils
    from spammm import atomicUtils as au

    xyz = _abs_path(args.xyz or 'data/xyz/PTCDA.xyz')
    name = os.path.splitext(os.path.basename(xyz))[0]
    outdir = _abs_path(args.outdir) or os.path.join(_ROOT, 'debug', 'spm_brstm', name)
    os.makedirs(outdir, exist_ok=True)

    pos, _, names, _, _ = au.load_xyz(xyz)
    atomPos = np.asarray(pos, dtype=np.float64)
    enames = list(names)
    if not getattr(args, 'no_orient', False):
        from spammm.AtomicSystem import AtomicSystem
        mol = AtomicSystem(fname=xyz)
        if hasattr(mol, 'orientPCA'):
            mol.orientPCA()
            atomPos = np.asarray(mol.apos, dtype=np.float64)
            enames = list(mol.enames)
        atomPos = atomPos.copy()
        atomPos[:, 2] = float(atomPos[:, 2].mean())

    mo_rel = [int(x) for x in str(args.mo).replace(',', ' ').split() if x.strip()]
    if not mo_rel:
        mo_rel = [0, 1]  # HOMO + LUMO
    stm_heights = tuple(float(x) for x in str(args.stm_heights).replace(',', ' ').split() if x.strip())
    amp_align = not bool(getattr(args, 'no_amp_align', False))
    os.environ.setdefault('SPAMMM_AFM_CPU_FFT', '1')

    res = afm_utils.run_br_stm_afm_panel(
        atomPos, enames, outdir,
        basis=args.basis, step=args.step, margin=args.margin, z_extra=args.z_extra,
        scan_range=args.scan_range, scan_step=args.scan_step,
        h_min=args.h_min, h_max=args.h_max, h_step=args.h_step,
        amp=args.amp, amp_align=amp_align, stm_heights=stm_heights,
        K_LAT_Nm=args.K_LAT, K_RAD=args.K_RAD, bond_length=args.bond_length,
        mo_relative=mo_rel, field=args.field, projection=args.projection,
        tip_mode=args.tip_mode,
        df_cmap=args.df_cmap, fz_cmap=args.cmap, stm_cmap=args.stm_cmap,
        scale=args.scale, show_atoms=bool(args.show_atoms),
        force_recompute=bool(args.force), pp_stride=int(args.pp_stride),
        stm_mode=args.stm_mode, tip_orbital=args.tip_orbital, tip_elem=args.tip_elem,
        eh_K=args.eh_K, rcut=args.rcut, taper_w=args.taper_w,
        degen_thresh_eV=args.degen_thresh,
    )
    print(f'REVIEW: {os.path.abspath(outdir)}/')
    print(f'REVIEW: {res["png_stm"]}')
    print(f'REVIEW: {res["png_afm"]}')
    print(f'REVIEW: {res["png_brstm"]}')
    return 0


def cmd_basis_tails(args) -> int:
    """Central-C ρ(z) + Pauli E(z) log compare: GPAW / pySCF / DFTB stock / prolonged."""
    from spammm.SPM import AFM_utils as afm_utils
    mols = args.molecule or ['pentacene', 'PTCDA']
    formats = tuple(x.strip() for x in str(args.formats).split(',') if x.strip())
    afm_utils.run_basis_tails_compare(
        molecules=mols,
        outdir=_abs_path(args.outdir) or args.outdir,
        basis=args.basis,
        tip_mode=args.tip_mode,
        sigma=args.sigma,
        A_pauli=args.A,
        beta_pauli=args.beta,
        z_max=args.z_max,
        dz=args.dz,
        formats=formats,
        pyscf_basis=args.pyscf_basis,
        pyscf_xc=args.pyscf_xc,
        pyscf_z_extra=args.pyscf_z_extra,
        force_pyscf=bool(args.force_pyscf),
        verbosity=1,
    )
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

    p_opt = sub.add_parser('opt', help='Vacuum geometry opt (UFF/SPFF/LFF/DFTB); keep planar')
    p_opt.add_argument('--xyz', default=None)
    p_opt.add_argument('--smiles', default=None)
    p_opt.add_argument('--smiles-example', default=None, dest='smiles_example')
    p_opt.add_argument('--method', default='uff', choices=['uff', 'spff', 'lff', 'dftb'])
    p_opt.add_argument('--nsteps', type=int, default=1000)
    p_opt.add_argument('--fmax', type=float, default=0.05)
    p_opt.add_argument('--basis', default='3ob-3-1', choices=['3ob-3-1', 'mio-1-1'])
    p_opt.add_argument('--no-planar', action='store_true', dest='no_planar')
    p_opt.add_argument('--no-orient', action='store_true', dest='no_orient',
                       help='Skip PCA long-axis→x after opt')
    p_opt.add_argument('--outdir', default='debug/spm_opt')
    p_opt.set_defaults(func=cmd_opt)

    p_safm = sub.add_parser('smiles-afm', help='SMILES → planar vacuum opt → prolonged FDBM AFM (+ atom dots)')
    _add_common_afm_args(p_safm)
    p_safm.add_argument('--example', nargs='*', default=None,
                        help='SMILES_EXAMPLES names (default: all)')
    p_safm.add_argument('--method', default='uff', choices=['uff', 'spff', 'lff', 'dftb'])
    p_safm.add_argument('--nsteps', type=int, default=1000)
    p_safm.add_argument('--fmax', type=float, default=0.05)
    p_safm.set_defaults(func=cmd_smiles_afm, outdir='debug/spm_smiles_afm',
                        projection='prolonged', show_atoms=True, xyz=None)

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

    p_es = sub.add_parser('es-diag', help='Cube ES chain diag: ρ, Δρ, V_ES, E_ES, tip + mirror asym')
    p_es.add_argument('--molecule', nargs='*', default=None,
                      help='Fukui panel names (default: all). e.g. PTCDA pentacene phtalo_1-dftb-relax')
    p_es.add_argument('--outdir', default='debug/fdbm_fukui_panel_flat')
    p_es.add_argument('--step', type=float, default=0.15)
    p_es.add_argument('--margin', type=float, default=4.0)
    p_es.add_argument('--basis', default='3ob-3-1', choices=['3ob-3-1', 'mio-1-1'])
    p_es.add_argument('--z-above', nargs=2, type=float, default=[1.0, 5.0],
                      help='Slice heights above molecule plane [Å]')
    p_es.set_defaults(func=cmd_es_diag)

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

    p_fgr = stm_sub.add_parser('fgr', help='FGR H−ES vs overlap_exp (pentacene/PTCDA Level-B)')
    stm.add_stm_common_args(p_fgr)
    stm.add_fgr_args(p_fgr)
    p_fgr.set_defaults(func=cmd_stm_fgr)

    p_brfgr = stm_sub.add_parser('br-fgr',
        help='BR-STM FGR compare: overlap vs I_S/I_H/I_τ with PP-AFM tip displacement')
    p_brfgr.add_argument('--molecule', nargs='*', default=['pentacene', 'PTCDA'])
    p_brfgr.add_argument('--outdir', default=None, help='Default: debug/stm_br_fgr_compare')
    p_brfgr.add_argument('--bases', default='3ob-3-1', choices=['3ob-3-1', 'mio-1-1'])
    # FGR args
    stm.add_fgr_args(p_brfgr)
    # PP-AFM args (shared with stm br) — uses same h_Fz stack as stm br
    p_brfgr.add_argument('--step', type=float, default=0.1)
    p_brfgr.add_argument('--margin', type=float, default=4.0)
    p_brfgr.add_argument('--z-extra', type=float, default=6.0, dest='z_extra')
    p_brfgr.add_argument('--scan-range', type=float, default=3.0, dest='scan_range')
    p_brfgr.add_argument('--scan-step', type=float, default=0.1, dest='scan_step')
    p_brfgr.add_argument('--h-min', type=float, default=3.7, dest='h_min')
    p_brfgr.add_argument('--h-max', type=float, default=4.7, dest='h_max')
    p_brfgr.add_argument('--h-step', type=float, default=0.1, dest='h_step')
    p_brfgr.add_argument('--amp', type=float, default=1.0)
    p_brfgr.add_argument('--no-amp-align', action='store_true', dest='no_amp_align')
    p_brfgr.add_argument('--K-LAT', type=float, default=0.5, dest='K_LAT')
    p_brfgr.add_argument('--K-RAD', type=float, default=20.0, dest='K_RAD')
    p_brfgr.add_argument('--bond-length', type=float, default=3.0, dest='bond_length')
    p_brfgr.add_argument('--tip-mode', default='co', choices=['co', 'gaussian'], dest='tip_mode')
    p_brfgr.add_argument('--projection', default='prolonged', choices=['stock', 'prolonged'])
    p_brfgr.add_argument('--force', action='store_true')
    p_brfgr.set_defaults(func=cmd_stm_br_fgr)

    p_br = stm_sub.add_parser('br', help='3-stage BR-STM: pure STM → df+|dxy| → STM vs BR-STM')
    p_br.add_argument('--xyz', default='data/xyz/PTCDA.xyz')
    p_br.add_argument('--outdir', default=None, help='Default: debug/spm_brstm/<mol>')
    p_br.add_argument('--basis', default='3ob-3-1', choices=['3ob-3-1', 'mio-1-1'])
    p_br.add_argument('--projection', default='prolonged', choices=['stock', 'prolonged'],
                      help='AFM Pauli (+ Stage1 STM STO table)')
    p_br.add_argument('--tip-mode', default='co', choices=['co', 'gaussian'], dest='tip_mode')
    p_br.add_argument('--step', type=float, default=0.1)
    p_br.add_argument('--margin', type=float, default=4.0)
    p_br.add_argument('--z-extra', type=float, default=6.0, dest='z_extra')
    p_br.add_argument('--scan-range', type=float, default=3.0, dest='scan_range')
    p_br.add_argument('--scan-step', type=float, default=0.1, dest='scan_step')
    p_br.add_argument('--h-min', '--zmin', type=float, default=3.7, dest='h_min',
                      help='AFM df window start [Å]')
    p_br.add_argument('--h-max', '--zmax', type=float, default=4.7, dest='h_max')
    p_br.add_argument('--h-step', '--dz', type=float, default=0.1, dest='h_step')
    p_br.add_argument('--amp', type=float, default=1.0)
    p_br.add_argument('--no-amp-align', action='store_true', dest='no_amp_align')
    p_br.add_argument('--stm-heights', default='0.5,1.5,2.5', dest='stm_heights',
                      help='Stage1 pure-STM heights [Å] (0.5 matches frontier orbital diag)')
    p_br.add_argument('--K-LAT', type=float, default=0.5, dest='K_LAT')
    p_br.add_argument('--K-RAD', type=float, default=20.0, dest='K_RAD')
    p_br.add_argument('--bond-length', type=float, default=3.0, dest='bond_length')
    p_br.add_argument('--mo', default='0 1',
                      help='MO offsets vs HOMO for Stage1 (default: HOMO and LUMO)')
    p_br.add_argument('--field', default='psi2', choices=['ldos', 'psi2', 'psi'])
    p_br.add_argument('--cmap', default='seismic')
    p_br.add_argument('--df-cmap', default='gray', dest='df_cmap')
    p_br.add_argument('--stm-cmap', default='viridis', dest='stm_cmap')
    p_br.add_argument('--scale', default='per_image', choices=['per_image', 'per_column', 'common'])
    p_br.add_argument('--show-atoms', action='store_true', dest='show_atoms')
    p_br.add_argument('--pp-stride', type=int, default=4, dest='pp_stride',
                      help='Every Nth pixel for PP xy red-dot overlay (ppafm plotDistortions)')
    p_br.add_argument('--no-orient', action='store_true', dest='no_orient')
    p_br.add_argument('--force', action='store_true')
    # FGR transfer STM mode (Stage 3): 'overlap' (legacy) or 'fgr' (H−E·S kernel)
    p_br.add_argument('--stm-mode', default='overlap', choices=['overlap', 'fgr'], dest='stm_mode',
                      help="Stage 3 STM kernel: 'overlap' (legacy exp) or 'fgr' (H−E·S transfer)")
    p_br.add_argument('--tip-orbital', default='s', choices=['s', 'pz', 'py'], dest='tip_orbital',
                      help='FGR tip orbital (only used when --stm-mode=fgr)')
    p_br.add_argument('--tip-elem', default='C', dest='tip_elem',
                      help='FGR phantom tip atom element (only used when --stm-mode=fgr)')
    p_br.add_argument('--eh-K', type=float, default=1.75, dest='eh_K',
                      help='Extended-Hückel K for FGR tables (only used when --stm-mode=fgr)')
    p_br.add_argument('--rcut', type=float, default=15.0,
                      help='Atom-pair cutoff [Å] (FGR mode)')
    p_br.add_argument('--taper-w', type=float, default=2.0, dest='taper_w',
                      help='Cosine taper width at rcut [Å] (FGR mode)')
    p_br.add_argument('--degen-thresh', type=float, default=0.005, dest='degen_thresh',
                      help='Degeneracy threshold [eV] for Stage 3 MO cluster sum (0=off)')
    p_br.set_defaults(func=cmd_stm_br)

    p_bt = sub.add_parser(
        'basis-tails',
        help='Central-C ρ(z)+Pauli log: GPAW / pySCF / DFTB stock vs prolonged (SVG talk plots)')
    p_bt.add_argument('--molecule', nargs='*', default=['pentacene', 'PTCDA'],
                      help='pentacene and/or PTCDA (comma or space separated)')
    p_bt.add_argument('--outdir', default='debug/presentation_basis_tails')
    p_bt.add_argument('--basis', default='3ob-3-1', choices=['3ob-3-1', 'mio-1-1'])
    p_bt.add_argument('--tip-mode', default='gaussian', choices=['gaussian', 'co'], dest='tip_mode',
                      help='Pauli tip (gaussian correlates ρ tails cleanly)')
    p_bt.add_argument('--sigma', type=float, default=0.35, help='Gaussian tip σ [Å] (small → tracks ρ tails)')
    p_bt.add_argument('--A', type=float, default=1.0, help='Pauli A (same for all ρ; 1→raw overlap)')
    p_bt.add_argument('--beta', type=float, default=1.0, help='Pauli β (same for all ρ)')
    p_bt.add_argument('--z-max', type=float, default=3.0, dest='z_max')
    p_bt.add_argument('--dz', type=float, default=0.05, help='z sampling step [Å] (finer → smoother)')
    p_bt.add_argument('--pyscf-basis', default='def2-SVP', dest='pyscf_basis',
                      help='pySCF GTO basis for live dens (not Fukui cube)')
    p_bt.add_argument('--pyscf-xc', default='pbe', dest='pyscf_xc')
    p_bt.add_argument('--pyscf-z-extra', type=float, default=6.0, dest='pyscf_z_extra',
                      help='Vacuum padding above mol for live pySCF grid [Å]')
    p_bt.add_argument('--force-pyscf', action='store_true', dest='force_pyscf',
                      help='Recompute live pySCF ρ even if cache exists')
    p_bt.add_argument('--formats', default='svg,png', help='Output formats CSV')
    p_bt.set_defaults(func=cmd_basis_tails)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
