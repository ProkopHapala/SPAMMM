#!/usr/bin/env python3
"""Run full FDBM pipeline + PP relaxation and plot results.

Uses existing AFM_utils functions. Computes CO tip density via DFTB
(get_density_from_dftb_dense on CO.xyz) for proper delta-density electrostatics.

Generates:
  1. XY slices of Fz_relax at each probe height
  2. XY slices of df (frequency shift) at each probe height
  3. XY slices of tip displacement (dz) at each probe height
  4. 1D Fz vs height above a carbon atom

Usage:
  python tests/SPM/plot_fdbm_relax.py [--xyz data/xyz/H2O.xyz] [--basis 3ob-3-1] [--outdir debug/fdbm_relax]
"""
import os, sys, argparse, numpy as np

os.environ.setdefault('PYOPENCL_CTX', '0')
os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

def compute_co_tip_dftb(basis, step, margin, target_shape, outdir):
    """Compute CO tip densities (total + delta) using DFTB via get_density_from_dftb_dense.

    Returns (co_rho_total, co_rho_delta) padded+rolled to target_shape.
    """
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.SPM import AFM as afm
    from spammm.config_utils import get_dftb_basis_path
    import spammm.atomicUtils as au

    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17,'Na':11,'Mg':12,'K':19,'Ca':20}
    co_pos, _, co_names, _, _ = au.load_xyz('data/xyz/CO.xyz')
    co_atomPos = np.array(co_pos, dtype=np.float64)
    co_atomTypes = np.array([ELEM_Z.get(e, 6) for e in co_names], dtype=np.int32)

    co_grid_spec, co_origin, co_ngrid = afm.setup_density_grid(co_atomPos, step=step, margin=margin, z_extra=2.0)
    basis_hsd_path = get_dftb_basis_path(basis)
    co_work = os.path.join(outdir, 'co_tip_dftb_work')
    os.makedirs(co_work, exist_ok=True)

    print(f"  Computing CO tip via DFTB (grid={co_ngrid})...")
    co_result = afm_utils.get_density_from_dftb_dense(
        co_atomPos, co_atomTypes, basis_hsd_path, co_work,
        grid_spec=co_grid_spec, step=step, margin=margin, z_extra=2.0, verbosity=0
    )
    co_rho_total_raw = co_result['rho_scf']
    co_rho_delta_raw = co_result['rho_diff']
    print(f"  CO tip: total integral={co_rho_total_raw.sum()*step**3:.4f}, delta integral={co_rho_delta_raw.sum()*step**3:.6f}")

    co_rho_total = afm_utils._pad_and_roll_co_tip(co_rho_total_raw, target_shape)
    co_rho_delta = afm_utils._pad_and_roll_co_tip(co_rho_delta_raw, target_shape)
    return co_rho_total, co_rho_delta

def main():
    parser = argparse.ArgumentParser(description="Run FDBM relaxation and plot results")
    parser.add_argument('--xyz', default='data/xyz/H2O.xyz', help='XYZ file path')
    parser.add_argument('--basis', default='3ob-3-1', help='DFTB basis set')
    parser.add_argument('--step', type=float, default=0.15, help='Grid step [Ang]')
    parser.add_argument('--margin', type=float, default=6.0, help='Grid margin [Ang]')
    parser.add_argument('--outdir', default='debug/plot_fdbm_relax', help='Output directory')
    parser.add_argument('--K_LAT', type=float, default=0.5, help='Lateral stiffness [eV/Ang^2]')
    parser.add_argument('--K_RAD', type=float, default=20.0, help='Radial stiffness [eV/Ang^2]')
    parser.add_argument('--bond_length', type=float, default=2.0, help='CO bond length [Ang]')
    parser.add_argument('--h_min', type=float, default=3.0, help='Min probe height above mol [Ang]')
    parser.add_argument('--h_max', type=float, default=6.0, help='Max probe height above mol [Ang]')
    parser.add_argument('--h_step', type=float, default=0.25, help='Probe height step [Ang]')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    import spammm.atomicUtils as au

    os.makedirs(args.outdir, exist_ok=True)

    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17,'Na':11,'Mg':12,'K':19,'Ca':20}
    pos, _, names, _, _ = au.load_xyz(args.xyz)
    atomPos = np.array(pos, dtype=np.float64)
    enames = list(names)
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in enames], dtype=np.int32)
    mol_z = float(atomPos[:,2].max())
    print(f"Molecule: {os.path.basename(args.xyz)}  ({len(enames)} atoms)  mol_z={mol_z:.2f}")

    grid_spec, origin, ngrid = afm.setup_density_grid(atomPos, step=args.step, margin=args.margin, z_extra=6.0)
    nx, ny, nz = ngrid
    print(f"Grid: {ngrid}  step={args.step}  origin={origin}")

    # Step 1: Sample density via DFTB
    basis_hsd_path = get_dftb_basis_path(args.basis)
    work_dir = os.path.join(args.outdir, 'dftb_work')
    os.makedirs(work_dir, exist_ok=True)
    result = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd_path, work_dir,
        grid_spec=grid_spec, step=args.step, margin=args.margin, z_extra=6.0, verbosity=0
    )
    rho_scf = result['rho_scf']
    rho_na = result['rho_na']
    rho_diff = result['rho_diff']
    V_ES = result['V_ES']

    # Step 2: CO tip density via DFTB (proper delta density, not Gaussian)
    print("Computing CO tip density...")
    co_rho_total, co_rho_delta = compute_co_tip_dftb(args.basis, args.step, args.margin, (nx, ny, nz), args.outdir)

    # Step 3: Pauli — uses rho_scf (sample) x co_rho_total (tip)
    print("Computing Pauli...")
    overlap_raw = afm.compute_pauli_overlap(rho_scf, co_rho_total, args.step, tip_rolled=True)
    pauli_params = afm.PAULI_FITTED_DEFAULTS.get(args.basis, {'A': 787.22, 'beta': 1.2371})
    E_pauli = afm.scale_pauli_field(overlap_raw, args.step, pauli_params['A'], pauli_params['beta'], return_grads=False)

    # Step 4: Electrostatics — V_ES from sample rho_diff, convolved with CO tip delta density
    print("Computing Electrostatics...")
    E_ES = afm.compute_es_conv_field(V_ES, co_rho_delta, args.step, tip_rolled=True, return_grads=False)

    # Step 5: Dispersion
    print("Computing Dispersion...")
    E_vdw = afm.compute_dispersion_grid(atomPos, atomTypes, origin, args.step, ngrid, C6_CO=30.0, return_grads=False)

    E_total = E_pauli + E_ES + E_vdw
    print(f"E_pauli: [{E_pauli.min():.4e}, {E_pauli.max():.4e}]")
    print(f"E_ES:    [{E_ES.min():.4e}, {E_ES.max():.4e}]")
    print(f"E_vdw:   [{E_vdw.min():.4e}, {E_vdw.max():.4e}]")
    print(f"E_total: [{E_total.min():.4e}, {E_total.max():.4e}] eV")

    print("Computing force field gradient (GPU)...")
    afmulator = afm.AFMulator(use_morse=False, nloc=32)
    F_total = afmulator.compute_gradient_cl(E_total, args.step, bAlloc=True)
    print(f"F_total: Fz=[{F_total[...,2].min():.4e}, {F_total[...,2].max():.4e}]")

    # Setup scan grid
    scan_margin = 1.0
    x_min = float(atomPos[:,0].min() - scan_margin)
    x_max = float(atomPos[:,0].max() + scan_margin)
    y_min = float(atomPos[:,1].min() - scan_margin)
    y_max = float(atomPos[:,1].max() + scan_margin)
    scan_xs = np.arange(x_min, x_max, args.step, dtype=np.float32)
    scan_ys = np.arange(y_min, y_max, args.step, dtype=np.float32)
    heights = np.arange(args.h_min, args.h_max, args.h_step, dtype=np.float32)
    print(f"Scan: {len(scan_xs)}x{len(scan_ys)}  heights={heights[0]:.1f}..{heights[-1]:.1f} Å")

    afmulator.setup_fdbm_grid(F_total, origin, args.step)
    print("Running PP relaxation (GPU)...")
    FEs_relax, tip_disp = afmulator.scan_fdbm(
        scan_xs, scan_ys, heights, mol_z=mol_z,
        K_LAT=args.K_LAT, K_RAD=args.K_RAD, bond_length=args.bond_length,
        ppm_mode=True, use_fire=True
    )
    Fz_relax = FEs_relax[:,:,:,2]
    df = afm.compute_df(Fz_relax, float(heights[1] - heights[0]))
    print(f"Fz_relax: [{Fz_relax.min():.4e}, {Fz_relax.max():.4e}]  mean={Fz_relax.mean():.4e}")
    print(f"df:       [{df.min():.4e}, {df.max():.4e}]  mean={df.mean():.4e}")
    print(f"tip dz:   [{tip_disp['dz'].min():.4e}, {tip_disp['dz'].max():.4e}]")

    # Check for NaN
    n_nan = np.sum(~np.isfinite(Fz_relax))
    print(f"NaN/inf in Fz_relax: {n_nan} / {Fz_relax.size}")

    x_ext = [float(scan_xs[0]), float(scan_xs[-1])]
    y_ext = [float(scan_ys[0]), float(scan_ys[-1])]
    base = os.path.basename(args.xyz).replace('.xyz','')

    # === 1-3: Use existing AFM_utils.plot_grid_Fz (per-slice symmetric safe_norm, bwr) ===
    afm_utils.plot_grid_Fz(Fz_relax, heights, f'Fz_relax — {base}',
        f'Fz_relax_{base}.png', x_ext=x_ext, y_ext=y_ext, save_dir=args.outdir)
    afm_utils.plot_grid_Fz(df, heights, f'df — {base}',
        f'df_{base}.png', x_ext=x_ext, y_ext=y_ext, save_dir=args.outdir)
    afm_utils.plot_grid_Fz(tip_disp['dz'], heights, f'tip dz — {base}',
        f'tip_dz_{base}.png', x_ext=x_ext, y_ext=y_ext, save_dir=args.outdir)

    # === 4. 1D Fz vs height above carbon atom ===
    carbon_atoms = [(i, atomPos[i]) for i, e in enumerate(enames) if e == 'C']
    if not carbon_atoms:
        carbon_atoms = [(i, atomPos[i]) for i in range(len(enames))]
    cx, cy = float(atomPos[:,0].mean()), float(atomPos[:,1].mean())
    best_i, best_d = None, 1e9
    for i, p in carbon_atoms:
        d = (p[0]-cx)**2 + (p[1]-cy)**2
        if d < best_d: best_i, best_d = i, d
    atom_p = atomPos[best_i]
    ix_atom = int(np.argmin(np.abs(scan_xs - atom_p[0])))
    iy_atom = int(np.argmin(np.abs(scan_ys - atom_p[1])))
    print(f"1D curve above atom {best_i} ({enames[best_i]}) at ({atom_p[0]:.2f}, {atom_p[1]:.2f}, {atom_p[2]:.2f})")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(heights, Fz_relax[ix_atom, iy_atom, :], 'k-', label='Fz_relax', linewidth=2)
    ax.plot(heights, df[ix_atom, iy_atom, :], 'r--', label='df', linewidth=1.5)
    ax.set_xlabel('Probe height above molecule (Å)', fontsize=11)
    ax.set_ylabel('Fz (eV/Å) / df (a.u.)', fontsize=11)
    ax.set_title(f'Fz and df vs height above {enames[best_i]} — {os.path.basename(args.xyz)}', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_1d = os.path.join(args.outdir, f'Fz_df_1d_{os.path.basename(args.xyz).replace(".xyz","")}.png')
    fig.savefig(png_1d, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_1d}")

    print(f"\nDone! Check {args.outdir}/ for outputs.")

if __name__ == '__main__':
    main()
