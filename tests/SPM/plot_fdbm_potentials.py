#!/usr/bin/env python3
"""Diagnostic: plot FDBM potentials (Pauli, Electrostatics, Dispersion, Total).

Generates:
  1. XY slices at ~2.0 Å above molecule plane (4 panels: Pauli, ES, vdw, Total)
  2. XZ cross-section through molecule center (4 panels)
  3. 1D curves along z above a carbon atom (4 curves: Pauli, ES, vdw, Total)

Usage:
  python tests/SPM/plot_fdbm_potentials.py [--xyz data/xyz/H2O.xyz] [--basis 3ob-3-1] [--outdir debug/fdbm_potentials]
"""
import os, sys, argparse, numpy as np

os.environ.setdefault('PYOPENCL_CTX', '0')
os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

def main():
    parser = argparse.ArgumentParser(description="Plot FDBM potentials for diagnostics")
    parser.add_argument('--xyz', default='data/xyz/H2O.xyz', help='XYZ file path')
    parser.add_argument('--basis', default='3ob-3-1', help='DFTB basis set')
    parser.add_argument('--step', type=float, default=0.15, help='Grid step [Ang]')
    parser.add_argument('--margin', type=float, default=6.0, help='Grid margin [Ang]')
    parser.add_argument('--outdir', default='debug/plot_fdbm_potentials', help='Output directory')
    parser.add_argument('--height', type=float, default=2.0, help='Height above molecule for XY slice [Ang]')
    parser.add_argument('--vlim', type=float, default=0.1, help='Y-axis limit for 1D plot [eV]')
    parser.add_argument('--zmin', type=float, default=2.0, help='Min height above mol for 1D plot x-axis [Ang]')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path
    import spammm.atomicUtils as au

    os.makedirs(args.outdir, exist_ok=True)

    # Load molecule
    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17,'Na':11,'Mg':12,'K':19,'Ca':20}
    pos, _, names, _, _ = au.load_xyz(args.xyz)
    atomPos = np.array(pos, dtype=np.float64)
    enames = list(names)
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in enames], dtype=np.int32)
    print(f"Molecule: {os.path.basename(args.xyz)}  ({len(enames)} atoms)")

    # Setup grid
    grid_spec, origin, ngrid = afm.setup_density_grid(atomPos, step=args.step, margin=args.margin, z_extra=6.0)
    nx, ny, nz = ngrid
    print(f"Grid: {ngrid}  step={args.step}  origin={origin}")

    # Run DFTB+ SCF + density projection
    basis_hsd_path = get_dftb_basis_path(args.basis)
    work_dir = os.path.join(args.outdir, 'dftb_work')
    os.makedirs(work_dir, exist_ok=True)
    result = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd_path, work_dir,
        grid_spec=grid_spec, step=args.step, margin=args.margin, z_extra=6.0, verbosity=0
    )
    rho_scf = result['rho_scf']
    V_ES = result['V_ES']

    # Build tip density
    sigma_tip = 0.7
    rho_tip_total = afm.build_gaussian_tip((nx, ny, nz), args.step, sigma_tip)
    rho_tip_delta = rho_tip_total

    # Compute individual potentials
    print("Computing Pauli...")
    overlap_raw = afm.compute_pauli_overlap(rho_scf, rho_tip_total, args.step, tip_rolled=True)
    pauli_params = afm.PAULI_FITTED_DEFAULTS.get(args.basis, {'A': 787.22, 'beta': 1.2371})
    E_pauli = afm.scale_pauli_field(overlap_raw, args.step, pauli_params['A'], pauli_params['beta'], return_grads=False)

    print("Computing Electrostatics...")
    E_ES = afm.compute_es_conv_field(V_ES, rho_tip_delta, args.step, tip_rolled=True, return_grads=False)

    print("Computing Dispersion...")
    E_vdw = afm.compute_dispersion_grid(atomPos, atomTypes, origin, args.step, ngrid, C6_CO=30.0, return_grads=False)

    E_total = E_pauli + E_ES + E_vdw

    mol_z = float(atomPos[:,2].max())
    extent_xy = [float(origin[0]), float(origin[0] + nx*args.step),
                 float(origin[1]), float(origin[1] + ny*args.step)]
    extent_xz = [float(origin[0]), float(origin[0] + nx*args.step),
                 float(origin[2]), float(origin[2] + nz*args.step)]

    print(f"\n=== Potential Ranges ===")
    print(f"E_pauli: [{E_pauli.min():.4e}, {E_pauli.max():.4e}] eV")
    print(f"E_ES:    [{E_ES.min():.4e}, {E_ES.max():.4e}] eV")
    print(f"E_vdw:   [{E_vdw.min():.4e}, {E_vdw.max():.4e}] eV")
    print(f"E_total: [{E_total.min():.4e}, {E_total.max():.4e}] eV")

    components = [
        ('Pauli',         E_pauli, 'hot'),
        ('Electrostatic', E_ES,    'bwr'),
        ('Dispersion',    E_vdw,   'bwr'),
        ('Total',         E_total, 'bwr'),
    ]

    # === 1. XY slice at height above molecule ===
    z_slice = mol_z + args.height
    iz_slice = int((z_slice - origin[2]) / args.step)
    iz_slice = max(0, min(nz-1, iz_slice))
    z_actual = origin[2] + iz_slice * args.step
    print(f"\nXY slice at z={z_actual:.2f} Å (mol_z={mol_z:.2f}, height={z_actual-mol_z:.2f} Å)")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for col, (label, field, cmap) in enumerate(components):
        ax = axes[col]
        data = field[:,:,iz_slice].T
        if cmap == 'hot':
            vmax = max(float(np.percentile(data, 99)), 1e-30)
            im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', vmin=0, vmax=vmax, extent=extent_xy)
        else:
            vabs = max(float(np.percentile(np.abs(data), 99)), 1e-30)
            im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', vmin=-vabs, vmax=vabs, extent=extent_xy)
        ax.set_title(f'{label} (z={z_actual:.1f}Å)', fontsize=10)
        ax.set_xlabel('x (Å)', fontsize=8)
        if col == 0: ax.set_ylabel('y (Å)', fontsize=8)
        for i, (x, y, z) in enumerate(atomPos):
            if abs(z - mol_z) < args.step * 2:
                ax.plot(x, y, 'w+', markersize=6, markeredgewidth=1.0)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'FDBM Potentials XY slice — {os.path.basename(args.xyz)} (h={z_actual-mol_z:.1f}Å above mol)', fontsize=12)
    fig.tight_layout()
    png_xy = os.path.join(args.outdir, f'pot_xy_{os.path.basename(args.xyz).replace(".xyz","")}.png')
    fig.savefig(png_xy, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_xy}")

    # === 2. XZ cross-section through molecule center (y = mean y) ===
    y_center = float(atomPos[:,1].mean())
    iy_center = int((y_center - origin[1]) / args.step)
    iy_center = max(0, min(ny-1, iy_center))
    y_actual = origin[1] + iy_center * args.step
    print(f"XZ slice at y={y_actual:.2f} Å")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for col, (label, field, cmap) in enumerate(components):
        ax = axes[col]
        data = field[:,iy_center,:].T  # (nx, nz) -> transpose for (z, x)
        if cmap == 'hot':
            vmax = max(float(np.percentile(data, 99)), 1e-30)
            im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', vmin=0, vmax=vmax, extent=extent_xz)
        else:
            vabs = max(float(np.percentile(np.abs(data), 99)), 1e-30)
            im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', vmin=-vabs, vmax=vabs, extent=extent_xz)
        ax.set_title(f'{label} (y={y_actual:.1f}Å)', fontsize=10)
        ax.set_xlabel('x (Å)', fontsize=8)
        if col == 0: ax.set_ylabel('z (Å)', fontsize=8)
        for i, (x, y, z) in enumerate(atomPos):
            if abs(y - y_actual) < args.step * 2:
                ax.plot(x, z, 'w+', markersize=6, markeredgewidth=1.0)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'FDBM Potentials XZ cross-section — {os.path.basename(args.xyz)}', fontsize=12)
    fig.tight_layout()
    png_xz = os.path.join(args.outdir, f'pot_xz_{os.path.basename(args.xyz).replace(".xyz","")}.png')
    fig.savefig(png_xz, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_xz}")

    # === 3. 1D curves along z above a carbon atom ===
    carbon_atoms = [(i, atomPos[i]) for i, e in enumerate(enames) if e == 'C']
    if not carbon_atoms:
        carbon_atoms = [(i, atomPos[i]) for i in range(len(enames))]
    # Pick the carbon atom closest to molecule center
    cx, cy, cz = float(atomPos[:,0].mean()), float(atomPos[:,1].mean()), mol_z
    best_i, best_d = None, 1e9
    for i, p in carbon_atoms:
        d = (p[0]-cx)**2 + (p[1]-cy)**2
        if d < best_d: best_i, best_d = i, d
    atom_p = atomPos[best_i]
    print(f"1D curve above atom {best_i} ({enames[best_i]}) at ({atom_p[0]:.2f}, {atom_p[1]:.2f}, {atom_p[2]:.2f})")

    ix_atom = int((atom_p[0] - origin[0]) / args.step)
    iy_atom = int((atom_p[1] - origin[1]) / args.step)
    ix_atom = max(0, min(nx-1, ix_atom))
    iy_atom = max(0, min(ny-1, iy_atom))

    z_coords = np.array([origin[2] + iz * args.step for iz in range(nz)])
    z_rel = z_coords - mol_z  # height above molecule plane

    curves = [
        ('Pauli',         E_pauli[ix_atom, iy_atom, :], 'r'),
        ('Electrostatic', E_ES[ix_atom, iy_atom, :],    'b'),
        ('Dispersion',    E_vdw[ix_atom, iy_atom, :],   'g'),
        ('Total',         E_total[ix_atom, iy_atom, :], 'k'),
    ]

    mask = z_rel >= args.zmin
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for label, vals, color in curves:
        ax.plot(z_rel[mask], vals[mask], color, label=label, linewidth=1.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='mol plane')
    ax.set_xlabel('z above molecule plane (Å)', fontsize=11)
    ax.set_ylabel('Energy (eV)', fontsize=11)
    ax.set_ylim(-args.vlim, args.vlim)
    ax.set_title(f'FDBM Potentials above {enames[best_i]} atom — {os.path.basename(args.xyz)} (z>{args.zmin:.0f}Å, ±{args.vlim}eV)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_1d = os.path.join(args.outdir, f'pot_1d_{os.path.basename(args.xyz).replace(".xyz","")}.png')
    fig.savefig(png_1d, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_1d}")

    # Also print numerical values at key heights
    print(f"\n=== 1D values at key heights above {enames[best_i]} ===")
    print(f"{'h(Å)':>6s}  {'Pauli':>12s}  {'ES':>12s}  {'vdW':>12s}  {'Total':>12s}")
    for h in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        iz = int((mol_z + h - origin[2]) / args.step)
        if 0 <= iz < nz:
            print(f"{h:6.1f}  {E_pauli[ix_atom,iy_atom,iz]:12.4e}  {E_ES[ix_atom,iy_atom,iz]:12.4e}  {E_vdw[ix_atom,iy_atom,iz]:12.4e}  {E_total[ix_atom,iy_atom,iz]:12.4e}")

    print(f"\nDone! Check {args.outdir}/ for outputs.")

if __name__ == '__main__':
    main()
