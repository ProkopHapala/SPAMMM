#!/usr/bin/env python3
"""
run_afm_morse_visual.py — Generate AFM visualizations for pentacene and PTCDA.

Produces:
  1. 2D slices of Morse potential energy (repulsive "Pauli" part, attractive part, full Morse, Coulomb, total)
  2. 2D slices of relaxed probe-particle Fz forces at multiple z heights
  3. Fz(z) curves at center and over atoms
  4. df (frequency shift) maps

Usage:
  python tests/SPM/run_afm_morse_visual.py
"""
import os, sys, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('PYOPENCL_CTX', '0')
os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
PARAMS_PATH = os.path.join(DATA_DIR, 'ElementTypes.dat')

DX = 0.1  # isotropic grid spacing in Angstrom

def make_afmulator(xyz_path, use_morse=True, margin=4.0, z_top=16.0, dx=DX):
    """Build AFMulator with isotropic grid spacing dx=dy=dz=dx.
    Grid size is computed from molecule bounding box + margin."""
    from spammm.SPM.AFM import AFMulator
    afm = AFMulator(use_morse=use_morse)
    mol = afm.load_molecule(xyz_path)
    afm.assign_params(params_path=PARAMS_PATH)
    # Compute grid dimensions from molecule size + margin
    apos = afm.atoms_arr[:,:3].copy()
    mn, mx = apos.min(axis=0), apos.max(axis=0)
    Lx = float((mx[0]-mn[0]) + 2*margin)
    Ly = float((mx[1]-mn[1]) + 2*margin)
    Lz = float((mx[2]-mn[2]) + margin/2 + z_top)
    nx = int(np.ceil(Lx/dx))
    ny = int(np.ceil(Ly/dx))
    nz = int(np.ceil(Lz/dx))
    # Make nx,ny multiples of 8 for GPU alignment
    nx = ((nx+7)//8)*8; ny = ((ny+7)//8)*8
    afm.setup_grid(n=(nx,ny,nz), margin=margin, z_top=z_top)
    return afm, mol

def download_ff_grid(afm):
    """Download force field grid from GPU.
    OpenCL 3D images store as (z,y,x) in memory — must use (nz,ny,nx,4) shape
    then transpose to (nx,ny,nz,4) and copy to contiguous array."""
    import pyopencl as cl
    nx, ny, nz = int(afm.n[0]), int(afm.n[1]), int(afm.n[2])
    img_raw = np.zeros((nz, ny, nx, 4), dtype=np.float32)
    cl.enqueue_copy(afm.queue, img_raw, afm.img_FF, origin=(0,0,0), region=(nx,ny,nz))
    afm.queue.finish()
    # Transpose (nz,ny,nx,4) -> (nx,ny,nz,4) and make contiguous
    img_h = np.ascontiguousarray(img_raw.transpose(2,1,0,3))
    return img_h

R2SAFE = 1e-4
E_CLAMP = 100.0

def compute_morse_parts(afm, grid_pts):
    """Compute Morse repulsive and attractive parts at grid points, matching GPU kernel.
    GPU uses R2SAFE=1e-4 and clamps |E| to 100.
    Returns (E_rep, E_attr) each (N,) array."""
    atoms = afm.atoms_arr[:afm.mol.natoms]
    cMs = afm.cLJs_arr[:afm.mol.natoms]  # (R0, E0, alpha, 0) for Morse
    E_rep = np.zeros(len(grid_pts), dtype=np.float32)
    E_attr = np.zeros(len(grid_pts), dtype=np.float32)
    for ia in range(len(atoms)):
        dp = grid_pts - atoms[ia,:3]
        r = np.sqrt(np.sum(dp**2, axis=1) + R2SAFE)  # match GPU R2SAFE
        R0, E0, alpha = cMs[ia,0], cMs[ia,1], cMs[ia,2]
        expar = np.exp(alpha * (r - R0))
        E_rep += E0 * expar * expar           # repulsive: E0 * exp(2*alpha*(r-R0))
        E_attr += -2.0 * E0 * expar           # attractive: -2*E0 * exp(alpha*(r-R0))
    # Clamp to match GPU
    E_rep = np.clip(E_rep, -E_CLAMP, E_CLAMP)
    E_attr = np.clip(E_attr, -E_CLAMP, E_CLAMP)
    return E_rep, E_attr

def run_molecule(mol_file, save_dir):
    print(f"\n{'='*60}")
    print(f"Processing {mol_file}")
    print(f"{'='*60}")
    xyz_path = os.path.join(DATA_DIR, 'xyz', mol_file)
    if not os.path.exists(xyz_path):
        print(f"  ERROR: {xyz_path} not found, skipping")
        return

    # --- Setup and compute force field ---
    margin = 4.0
    z_top = 16.0
    afm, mol = make_afmulator(xyz_path, use_morse=True, margin=margin, z_top=z_top)
    afm.make_forcefield()
    nx, ny, nz = int(afm.n[0]), int(afm.n[1]), int(afm.n[2])
    dx_ff = float(afm.dA[0])
    print(f"  Grid: {nx}x{ny}x{nz}  dx={dx_ff:.3f}Å  L={afm.L}  mol_z={afm.mol_z:.2f}")

    # Download total FF grid (Morse + Coulomb)
    ff_total = download_ff_grid(afm)  # (nx,ny,nz,4): Fx,Fy,Fz,E
    E_total = ff_total[...,3]

    # Run with zero tip charges to get Morse-only (no Coulomb)
    afm.tipQs = np.array([0., 0., 0., 0.], dtype=np.float32)
    afm.make_forcefield()
    ff_morse = download_ff_grid(afm)
    E_morse = ff_morse[...,3]

    # Coulomb = total - morse
    E_coulomb = E_total - E_morse

    # Compute Morse repulsive/attractive parts on CPU at grid centers
    xs = afm.p0[0] + np.arange(nx) * afm.dA[0]
    ys = afm.p0[1] + np.arange(ny) * afm.dB[1]
    zs = afm.p0[2] + np.arange(nz) * afm.dC[2]
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float32)
    print(f"  Computing Morse repulsive/attractive parts on CPU ({len(grid_pts)} points)...")
    E_rep_flat, E_attr_flat = compute_morse_parts(afm, grid_pts)
    E_rep = E_rep_flat.reshape(nx, ny, nz)
    E_attr = E_attr_flat.reshape(nx, ny, nz)

    # Print ranges
    print(f"  E_morse:  [{E_morse.min():.4f}, {E_morse.max():.4f}]")
    print(f"  E_rep:    [{E_rep.min():.4f}, {E_rep.max():.4f}]")
    print(f"  E_attr:   [{E_attr.min():.4f}, {E_attr.max():.4f}]")
    print(f"  E_coulomb:[{E_coulomb.min():.4f}, {E_coulomb.max():.4f}]")
    print(f"  E_total:  [{E_total.min():.4f}, {E_total.max():.4f}]")

    # --- Plot 1: Potential energy slices at selected z heights ---
    mol_z = afm.mol_z
    z_coords = afm.p0[2] + np.arange(nz) * afm.dC[2]
    z_rel = z_coords - mol_z  # relative to molecule top
    target_heights = [2.0, 3.0, 4.0, 5.0, 7.0]
    sel_iz = []
    for h in target_heights:
        iz = int(np.argmin(np.abs(z_rel - h)))
        if 0 <= iz < nz:
            sel_iz.append(iz)
    print(f"  Selected z slices: iz={sel_iz}  z_rel={[f'{z_rel[iz]:.1f}' for iz in sel_iz]}")

    # Plot extent in Angstroms (in molecule frame, before mol_shift)
    extent_xy = [float(afm.p0[0] - afm.mol_shift[0]), float(afm.p0[0] + afm.L[0] - afm.mol_shift[0]),
                 float(afm.p0[1] - afm.mol_shift[1]), float(afm.p0[1] + afm.L[1] - afm.mol_shift[1])]

    components = [
        ('E_rep (Pauli-like)', E_rep, 'Reds'),
        ('E_attr (attractive)', E_attr, 'Blues_r'),
        ('E_morse (full)', E_morse, 'bwr'),
        ('E_coulomb', E_coulomb, 'seismic'),
        ('E_total', E_total, 'bwr'),
    ]

    fig, axes = plt.subplots(len(components), len(sel_iz), figsize=(2.8*len(sel_iz), 2.8*len(components)))
    for row, (label, E_field, cmap) in enumerate(components):
        for col, iz in enumerate(sel_iz):
            ax = axes[row, col]
            data = E_field[:,:,iz].T  # (ny,nx) for imshow
            vabs = max(float(np.percentile(np.abs(data), 99)), 1e-6)
            im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', vmin=-vabs, vmax=vabs, extent=extent_xy)
            ax.set_title(f'z={z_rel[iz]:.1f}Å', fontsize=7)
            ax.tick_params(labelsize=5)
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
            if row == len(components)-1:
                ax.set_xlabel('x (Å)', fontsize=7)
            plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f'Potential Energy Components (Morse+Coulomb) — {mol_file}', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'potential_slices_{mol_file.replace(".xyz","")}.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved potential slices")

    # --- Run relaxed scan with isotropic pixels ---
    afm.tipQs = afm.DEFAULT_tipQs.copy()
    afm.make_forcefield()
    # Scan grid: cover molecule bounding box + 1Å margin, dx=dy=0.1Å
    apos = afm.atoms_arr[:,:3]
    mn_s, mx_s = apos.min(axis=0), apos.max(axis=0)
    scan_margin = 1.0
    scan_x0 = float(mn_s[0] - scan_margin)
    scan_y0 = float(mn_s[1] - scan_margin)
    scan_Lx = float((mx_s[0]-mn_s[0]) + 2*scan_margin)
    scan_Ly = float((mx_s[1]-mn_s[1]) + 2*scan_margin)
    nx_scan = int(np.ceil(scan_Lx / DX))
    ny_scan = int(np.ceil(scan_Ly / DX))
    nxy = (nx_scan, ny_scan)
    nz_scan = 30
    dtip = -0.15
    # Explicit scan grid in kernel-space (post mol_shift)
    scan_p0 = np.array([scan_x0, scan_y0, float(mx_s[2]) + 5.0 + abs(float(afm.dpos0[2]))], dtype=np.float32)
    scan_da = np.array([DX, 0., 0.], dtype=np.float32)
    scan_db = np.array([0., DX, 0.], dtype=np.float32)
    print(f"  Scan grid: {nxy}  dx={DX}Å  area={scan_Lx:.1f}x{scan_Ly:.1f}Å  nz={nz_scan} dtip={dtip}")
    FEs_relax, pts = afm.run_scan(nxy=nxy, nz=nz_scan, dtip=dtip, scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db)
    Fz_relax = FEs_relax[:,:,:,2]
    print(f"  Fz_relax: [{Fz_relax.min():.4f}, {Fz_relax.max():.4f}]")

    # Raw scan for comparison
    print(f"  Running raw scan...")
    FEs_raw, _ = afm.get_raw_FE(nxy=nxy, nz=nz_scan, dtip=dtip, scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db)
    Fz_raw = FEs_raw[:,:,:,2]
    print(f"  Fz_raw:   [{Fz_raw.min():.4f}, {Fz_raw.max():.4f}]")

    from spammm.SPM.AFM import compute_df
    df = compute_df(Fz_relax, abs(dtip))
    print(f"  df:        [{df.min():.4f}, {df.max():.4f}]")

    # Scan heights (relative to molecule top)
    z0_tip = float(mx_s[2]) + 5.0 + abs(float(afm.dpos0[2]))
    scan_heights = z0_tip + np.arange(nz_scan) * dtip - mol_z

    # Scan extent in molecule frame (un-shift x,y)
    scan_extent = [scan_x0 - afm.mol_shift[0], scan_x0 + (nx_scan-1)*DX - afm.mol_shift[0],
                   scan_y0 - afm.mol_shift[1], scan_y0 + (ny_scan-1)*DX - afm.mol_shift[1]]

    # --- Plot 2: Relaxed Fz slices ---
    sel_iz_scan = [0, 5, 10, 15, 20, 25]
    sel_iz_scan = [iz for iz in sel_iz_scan if iz < nz_scan]

    fig, axes = plt.subplots(2, len(sel_iz_scan), figsize=(2.8*len(sel_iz_scan), 5.6))
    for col, iz in enumerate(sel_iz_scan):
        ax = axes[0, col]
        data = Fz_relax[:,:,iz].T
        vabs = max(float(np.percentile(np.abs(data), 99)), 1e-6)
        im = ax.imshow(data, origin='lower', cmap='bwr', aspect='equal', vmin=-vabs, vmax=vabs, extent=scan_extent)
        ax.set_title(f'h={scan_heights[iz]:.1f}Å', fontsize=7)
        ax.tick_params(labelsize=5)
        if col == 0: ax.set_ylabel('Fz relaxed', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax = axes[1, col]
        data = df[:,:,iz].T
        vabs = max(float(np.percentile(np.abs(data), 99)), 1e-6)
        im = ax.imshow(data, origin='lower', cmap='bwr', aspect='equal', vmin=-vabs, vmax=vabs, extent=scan_extent)
        ax.set_title(f'h={scan_heights[iz]:.1f}Å', fontsize=7)
        ax.tick_params(labelsize=5)
        if col == 0: ax.set_ylabel('df', fontsize=8)
        ax.set_xlabel('x (Å)', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f'Relaxed PP Forces & df — {mol_file}', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'relaxed_forces_{mol_file.replace(".xyz","")}.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved relaxed force slices")

    # --- Plot 3: Fz(z) curves at center and over an atom ---
    atomPos = afm.atoms_arr[:afm.mol.natoms,:3]
    # Scan pixel coordinates in kernel-space
    scan_xs = scan_p0[0] + np.arange(nx_scan) * DX
    scan_ys = scan_p0[1] + np.arange(ny_scan) * DX
    # Center pixel
    ix_c, iy_c = nx_scan//2, ny_scan//2
    scan_x0_center = scan_xs[ix_c]
    scan_y0_center = scan_ys[iy_c]
    # Find atom closest to scan center
    dists = np.sqrt((atomPos[:,0]-scan_x0_center)**2 + (atomPos[:,1]-scan_y0_center)**2)
    i_atom = int(np.argmin(dists))
    ix_atom = int(np.argmin(np.abs(scan_xs - atomPos[i_atom,0])))
    iy_atom = int(np.argmin(np.abs(scan_ys - atomPos[i_atom,1])))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(scan_heights, Fz_raw[ix_c, iy_c, :], 'b--', lw=1, alpha=0.7, label='Fz raw (center)')
    ax.plot(scan_heights, Fz_relax[ix_c, iy_c, :], 'b-', lw=1.5, marker='o', ms=3, label='Fz relaxed (center)')
    ax.plot(scan_heights, Fz_raw[ix_atom, iy_atom, :], 'r--', lw=1, alpha=0.7, label=f'Fz raw (over atom {i_atom})')
    ax.plot(scan_heights, Fz_relax[ix_atom, iy_atom, :], 'r-', lw=1.5, marker='s', ms=3, label=f'Fz relaxed (over atom {i_atom})')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Height above mol top (Å)'); ax.set_ylabel('Fz (eV/Å)')
    ax.set_title(f'Fz(z) curves — {mol_file}')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(scan_heights, df[ix_c, iy_c, :], 'b-', lw=1.5, marker='o', ms=3, label='df (center)')
    ax.plot(scan_heights, df[ix_atom, iy_atom, :], 'r-', lw=1.5, marker='s', ms=3, label=f'df (over atom {i_atom})')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Height above mol top (Å)'); ax.set_ylabel('df (eV/Å²)')
    ax.set_title(f'df(z) curves — {mol_file}')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'fz_curves_{mol_file.replace(".xyz","")}.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved Fz/df curves")

    # --- Plot 4: Raw vs Relaxed comparison ---
    fig, axes = plt.subplots(2, len(sel_iz_scan), figsize=(2.8*len(sel_iz_scan), 5.6))
    for col, iz in enumerate(sel_iz_scan):
        ax = axes[0, col]
        data = Fz_raw[:,:,iz].T
        vabs = max(float(np.percentile(np.abs(data), 99)), 1e-6)
        im = ax.imshow(data, origin='lower', cmap='bwr', aspect='equal', vmin=-vabs, vmax=vabs, extent=scan_extent)
        ax.set_title(f'h={scan_heights[iz]:.1f}Å', fontsize=7)
        ax.tick_params(labelsize=5)
        if col == 0: ax.set_ylabel('Fz raw', fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax = axes[1, col]
        data = Fz_relax[:,:,iz].T
        im = ax.imshow(data, origin='lower', cmap='bwr', aspect='equal', vmin=-vabs, vmax=vabs, extent=scan_extent)
        ax.set_title(f'h={scan_heights[iz]:.1f}Å', fontsize=7)
        ax.tick_params(labelsize=5)
        if col == 0: ax.set_ylabel('Fz relaxed', fontsize=8)
        ax.set_xlabel('x (Å)', fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f'Raw vs Relaxed Fz — {mol_file}', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'raw_vs_relaxed_{mol_file.replace(".xyz","")}.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved raw vs relaxed comparison")

    # Save raw data
    np.savez(os.path.join(save_dir, f'afm_data_{mol_file.replace(".xyz","")}.npz'),
             E_morse=E_morse, E_rep=E_rep, E_attr=E_attr, E_coulomb=E_coulomb, E_total=E_total,
             Fz_raw=Fz_raw, Fz_relax=Fz_relax, df=df, scan_heights=scan_heights,
             z_rel=z_rel, sel_iz=sel_iz)
    print(f"  Saved raw data")

    # Cleanup GPU
    del afm

def main():
    today = datetime.date.today()
    save_dir = os.path.join('debug', f'{today}_afm_morse_visual')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")

    molecules = ['pentacene.xyz', 'PTCDA.xyz']
    for mol_file in molecules:
        try:
            run_molecule(mol_file, save_dir)
        except Exception as e:
            print(f"  ERROR for {mol_file}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone! All plots saved to {save_dir}/")

if __name__ == '__main__':
    main()
