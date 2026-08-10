#!/usr/bin/env python3
"""testplot_morse_direct_compare.py — Side-by-side AFM: GridFF Morse vs Direct Morse.

Visual proof that the brute-force direct kernel (relaxStrokesTiltedMorseDirect)
produces the same AFM contrast as the grid-based path (scan_fdbm + relaxStrokes),
AND that it is differentiable (VJP gradient visualization).

Uses SSOT plotting (plot_afm_variant_height_strip) with CLI-default params:
  h_min=3.7, h_max=4.7, h_step=0.1, amp=1.0, scan_margin=2.0, margin=4.0
  K_LAT=0.5 N/m, K_RAD=20, bond_length=3.0

Run:
  python tests/SPM/testplot_morse_direct_compare.py
  pytest tests/SPM/testplot_morse_direct_compare.py -s --develop
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── path bootstrap ──────────────────────────────────────────────────────────
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, _ROOT)

from spammm.SPM import AFM as afm_mod
from spammm.SPM import AFM_utils as afm_utils
from spammm.atomicUtils import load_xyz

ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53}

# ── CLI-default params (skill:afm-plotting SSOT) ────────────────────────────
H_MIN, H_MAX, H_STEP = 3.7, 4.7, 0.1
AMP = 0.5          # Å peak — reduced from CLI default 1.0 to avoid non-convergence at closest approach with K_RAD=20
SCAN_MARGIN = 2.0
MARGIN = 4.0
K_LAT_NM = 0.5       # N/m (CLI default)
K_RAD = 20.0         # eV/Å² (CLI default — stiff radial spring → sharp PP relaxation features)
BOND_LENGTH = 3.0    # Å (CLI default)
STEP = 0.1           # grid spacing [Å]
OSC_DIR = (0., 0., 1.)

OUTDIR = os.path.join(_ROOT, 'debug', 'test_afm_morse', 'morse_direct_compare')


def _load_benzene():
    xyz_path = os.path.join(_ROOT, 'data', 'xyz', 'benzene.xyz')
    pos, _, names, _, _ = load_xyz(xyz_path)
    atomPos = np.array(pos, dtype=np.float64)
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in names], dtype=np.int32)
    return atomPos, atomTypes, names


def _grid_origin_step_ngrid(atomPos, margin=MARGIN, step=STEP, z_extra=6.0):
    """Build grid origin/step/ngrid matching run_morse_pp_afm CLI defaults.

    z_extra: extra space above molecule top to avoid PBC wrap when PP relaxes
    outside the grid box (CLK_ADDRESS_MIRRORED_REPEAT sampler). Must cover
    max(h_scan) + mol_z + PP_displacement + safety.
    """
    mn = atomPos[:, :3].min(axis=0) - margin
    mx_xy = atomPos[:, :2].max(axis=0) + margin
    # z: from margin below molecule to z_extra above (covers full scan range)
    mx = np.array([mx_xy[0], mx_xy[1], float(atomPos[:, 2].max()) + z_extra])
    ngrid = tuple(int(np.ceil((mx[i] - mn[i]) / step)) + 1 for i in range(3))
    origin = mn.astype(np.float64)
    return origin, step, ngrid


def run_grid_morse(atomPos, atomTypes, outdir):
    """Grid-based Morse+Q via run_morse_pp_afm (same as `run_spm.py afm --model morse`)."""
    origin, step, ngrid = _grid_origin_step_ngrid(atomPos)
    h_df, h_Fz, h_scan = afm_utils.afm_df_height_stacks(
        H_MIN, H_MAX, H_STEP, amp=AMP, amp_align=True, osc_dir=OSC_DIR)
    scan_xs = np.arange(float(atomPos[:, 0].min() - SCAN_MARGIN),
                        float(atomPos[:, 0].max() + SCAN_MARGIN), step, dtype=np.float32)
    scan_ys = np.arange(float(atomPos[:, 1].min() - SCAN_MARGIN),
                        float(atomPos[:, 1].max() + SCAN_MARGIN), step, dtype=np.float32)
    K_LAT = afm_mod.stiffness_Nm_to_eVA2(K_LAT_NM)
    scan_spec = afm_utils.ScanSpec(
        scan_xs=scan_xs, scan_ys=scan_ys, h_df=h_df, h_Fz=h_Fz, h_scan=h_scan,
        amplitude=AMP, osc_dir=OSC_DIR, K_LAT=K_LAT, K_RAD=K_RAD,
        bond_length=BOND_LENGTH, scan_margin=SCAN_MARGIN)
    params_path = os.path.join(_ROOT, 'data', 'ElementTypes.dat')
    result = afm_utils.run_morse_pp_afm(
        'grid_morse', atomPos, atomTypes, origin, step, ngrid, outdir,
        scan_spec=scan_spec, params_path=params_path,
        margin=MARGIN, plots={'df', 'fz'}, df_cmap='gray', cmap='seismic')
    return result, scan_spec


def run_direct_morse(atomPos, atomTypes, scan_spec):
    """Direct brute-force Morse+Q via run_scan_morse_direct (no grid, differentiable)."""
    h_df, h_Fz, h_scan = scan_spec.h_df, scan_spec.h_Fz, scan_spec.h_scan
    scan_xs, scan_ys = scan_spec.scan_xs, scan_spec.scan_ys
    nx, ny = len(scan_xs), len(scan_ys)
    nz = len(h_scan)
    dtip = -H_STEP  # downward approach

    # Build atoms/cMs matching the grid path (same ElementTypes.dat, same tip params)
    atoms_arr, cLJs_arr = afm_utils._morse_atoms_from_Z(
        atomPos, atomTypes,
        params_path=os.path.join(_ROOT, 'data', 'ElementTypes.dat'))

    # AFMulator with PPM params matching CLI
    afmulator = afm_mod.AFMulator(use_morse=True, nloc=32, use_fire=True)
    afmulator.atoms_arr = atoms_arr
    afmulator.cLJs_arr = cLJs_arr
    afmulator.use_morse = True
    # Match stiffness/dpos0 to CLI PPM params
    K_LAT = afm_mod.stiffness_Nm_to_eVA2(K_LAT_NM)
    afmulator.stiffness = np.array([-K_LAT, -K_LAT, -K_LAT, -K_RAD], dtype=np.float32)
    afmulator.dpos0 = np.array([0., 0., -BOND_LENGTH, BOND_LENGTH], dtype=np.float32)
    # Tighter FIRE dt for close-approach convergence
    afmulator.relax_pars = np.array([0.2, 0.1, 0.02, 0.3], dtype=np.float32)

    # Scan geometry: match grid path exactly
    # Grid path: z_start = max(h_scan) + mol_z + bond_length (tip apex world z)
    mol_z = float(atomPos[:, 2].max())
    z_start = float(np.max(h_scan)) + mol_z + BOND_LENGTH
    dx = float(scan_xs[1] - scan_xs[0]) if len(scan_xs) > 1 else 0.
    dy = float(scan_ys[1] - scan_ys[0]) if len(scan_ys) > 1 else 0.
    scan_p0 = np.array([float(scan_xs[0]), float(scan_ys[0]), z_start], dtype=np.float32)
    scan_da = np.array([dx, 0., 0.], dtype=np.float32)
    scan_db = np.array([0., dy, 0.], dtype=np.float32)

    print(f"\n=== Direct Morse: nxy=({nx},{ny}) nz={nz} dtip={dtip} z_start={z_start:.2f} ===")
    FEs, points_xyz, PPs = afmulator.run_scan_morse_direct(
        nxy=(nx, ny), nz=nz, dtip=dtip,
        scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db,
        workgroup_size=64, bAlloc=True, return_pp=True)

    # CRITICAL: run_scan_morse_direct stores iz=0=highest z (start of descent),
    # but shared_postprocess expects iz=0=lowest z (like scan_fdbm after flip).
    # Flip z-axis for postprocessing ONLY. Keep unflipped PPs for VJP (the adjoint
    # kernel reconstructs tipPos from iz index and expects the original kernel order).
    FEs_flipped = FEs[:, :, ::-1, :]
    # Postprocess through shared_postprocess (same df/Fz extraction as grid path)
    result = afm_utils.shared_postprocess(FEs_flipped, scan_spec, backend_name='morse_direct')
    result.tag = 'direct_morse'
    result.atomPos = atomPos
    # Return FEs flipped (for plotting), but PPs unflipped (for VJP)
    return result, afmulator, FEs_flipped, points_xyz, PPs


def plot_z_curves(grid_result, direct_result, scan_spec, atomPos, atomTypes, names, outdir):
    """E(z) and Fz(z) curves above a carbon atom — GridFF vs Direct.

    Locates the nearest scan pixel to a carbon atom and plots the raw force
    volume (Fz, E) vs actual probe z-height for both backends. This reveals
    any z-alignment offset between the grid and direct paths.
    """
    scan_xs, scan_ys = scan_spec.scan_xs, scan_spec.scan_ys
    h_scan = scan_spec.h_scan
    dx = float(scan_xs[1] - scan_xs[0]) if len(scan_xs) > 1 else 0.
    dy = float(scan_ys[1] - scan_ys[0]) if len(scan_ys) > 1 else 0.

    # Find a carbon atom (Z=6)
    C_idx = np.where(atomTypes == 6)[0]
    if len(C_idx) == 0:
        print("  WARNING: no carbon atom found, using atom 0")
        C_idx = np.array([0])
    # Pick the carbon closest to scan center
    cx = float((scan_xs[0] + scan_xs[-1]) * 0.5)
    cy = float((scan_ys[0] + scan_ys[-1]) * 0.5)
    i_atom = C_idx[np.argmin(np.sum((atomPos[C_idx, :2] - [cx, cy])**2, axis=1))]
    ax, ay, az = atomPos[i_atom]
    print(f"  z-curve probe atom: {names[i_atom]} (Z={atomTypes[i_atom]}) at ({ax:.3f},{ay:.3f},{az:.3f})")

    # Nearest scan pixel
    ix = int(round((ax - float(scan_xs[0])) / dx)) if dx > 0 else 0
    iy = int(round((ay - float(scan_ys[0])) / dy)) if dy > 0 else 0
    ix = max(0, min(ix, len(scan_xs) - 1))
    iy = max(0, min(iy, len(scan_ys) - 1))
    px, py = float(scan_xs[ix]), float(scan_ys[iy])
    print(f"  nearest pixel: ix={ix} iy={iy} ({px:.3f},{py:.3f})  dist={np.hypot(px-ax, py-ay):.3f}Å")

    # Extract raw FEs at this pixel for both paths
    # grid_result.FEs and direct_result.FEs are (nx, ny, nz_scan, 4) after shared_postprocess
    # Both should be iz=0=lowest z after the flip in direct path
    grid_FEs = grid_result.FEs[ix, iy]   # (nz_scan, 4)
    direct_FEs = direct_result.FEs[ix, iy]  # (nz_scan, 4)

    # Probe z-heights = h_scan (probe O-apex above mol_z=0)
    z = np.asarray(h_scan, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Fz(z)
    ax = axes[0]
    ax.plot(z, grid_FEs[:, 2], 'b-o', ms=3, lw=1.5, label='GridFF')
    ax.plot(z, direct_FEs[:, 2], 'r-s', ms=3, lw=1.5, label='Direct')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('probe z [Å]'); ax.set_ylabel('Fz [eV/Å]')
    ax.set_title(f'Fz(z) above {names[i_atom]} @({px:.2f},{py:.2f})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # E(z)
    ax = axes[1]
    ax.plot(z, grid_FEs[:, 3], 'b-o', ms=3, lw=1.5, label='GridFF')
    ax.plot(z, direct_FEs[:, 3], 'r-s', ms=3, lw=1.5, label='Direct')
    ax.set_xlabel('probe z [Å]'); ax.set_ylabel('E [eV]')
    ax.set_title(f'E(z) above {names[i_atom]} @({px:.2f},{py:.2f})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # ΔFz(z) and ΔE(z)
    ax = axes[2]
    dFz = direct_FEs[:, 2] - grid_FEs[:, 2]
    dE  = direct_FEs[:, 3] - grid_FEs[:, 3]
    ax.plot(z, dFz, 'g-o', ms=3, lw=1.5, label='ΔFz (Direct−Grid)')
    ax.plot(z, dE,  'm-s', ms=3, lw=1.5, label='ΔE (Direct−Grid)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('probe z [Å]'); ax.set_ylabel('Δ [eV/Å or eV]')
    ax.set_title(f'Difference above {names[i_atom]}')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f'E(z)/Fz(z) above carbon — GridFF vs Direct Morse  (K_LAT={K_LAT_NM}N/m K_RAD={K_RAD} L={BOND_LENGTH}Å)', fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(outdir, 'z_curves_above_carbon.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    # Print numeric table for debugging
    print(f"  z[Å]   Fz_grid    Fz_direct   ΔFz        E_grid      E_direct    ΔE")
    for i in range(len(z)):
        print(f"  {z[i]:.2f}  {grid_FEs[i,2]:+.6e}  {direct_FEs[i,2]:+.6e}  {dFz[i]:+.6e}  {grid_FEs[i,3]:+.6e}  {direct_FEs[i,3]:+.6e}  {dE[i]:+.6e}")
    print(f"REVIEW: {out_png}")
    return out_png


def plot_compare_strip(grid_result, direct_result, scan_spec, atomPos, outdir):
    """Side-by-side df/Fz height strip: GridFF Morse vs Direct Morse."""
    heights = grid_result.heights  # h_df (probe heights)
    extent = afm_utils.scan_extent(scan_spec.scan_xs, scan_spec.scan_ys)
    amp_z = AMP * abs(float(np.asarray(OSC_DIR)[2]))

    variants = {
        'grid':   {'df': grid_result.df,   'Fz': grid_result.Fz},
        'direct': {'df': direct_result.df, 'Fz': direct_result.Fz},
    }
    row_specs = [
        ('df', 'grid',   f'df GridFF\n@h−{amp_z:.1f}Å', 'gray'),
        ('df', 'direct', f'df Direct\n@h−{amp_z:.1f}Å', 'gray'),
        ('Fz', 'grid',   f'Fz GridFF\n@same z',          'seismic'),
        ('Fz', 'direct', f'Fz Direct\n@same z',          'seismic'),
    ]
    title = f'Morse+Q AFM benzene  GridFF vs Direct  K_LAT={K_LAT_NM} N/m  K_RAD={K_RAD}  L={BOND_LENGTH}Å  amp={AMP}Å'
    out_png = os.path.join(outdir, 'compare_grid_vs_direct.png')
    afm_utils.plot_afm_variant_height_strip(
        variants, row_specs, heights, out_png,
        scale='per_image', title=title, dpi=140,
        apos=atomPos, show_atoms=True, extent=extent,
        amp=AMP, amp_align=True, amp_z=amp_z,
        long_axis_vertical=True, tight=True)
    print(f"REVIEW: {out_png}")
    return out_png


def plot_difference_strip(grid_result, direct_result, scan_spec, atomPos, outdir):
    """Difference (Direct − Grid) strip to show parity."""
    heights = grid_result.heights
    extent = afm_utils.scan_extent(scan_spec.scan_xs, scan_spec.scan_ys)
    amp_z = AMP * abs(float(np.asarray(OSC_DIR)[2]))

    df_diff = direct_result.df - grid_result.df
    Fz_diff = direct_result.Fz - grid_result.Fz
    variants = {'diff': {'df': df_diff, 'Fz': Fz_diff}}
    row_specs = [
        ('df', 'diff', f'Δdf (Direct−Grid)\n@h−{amp_z:.1f}Å', 'bwr'),
        ('Fz', 'diff', 'ΔFz (Direct−Grid)\n@same z',          'bwr'),
    ]
    out_png = os.path.join(outdir, 'diff_direct_minus_grid.png')
    afm_utils.plot_afm_variant_height_strip(
        variants, row_specs, heights, out_png,
        scale='common', title=f'Direct − GridFF difference  benzene  (scale=common)', dpi=140,
        apos=atomPos, show_atoms=True, extent=extent,
        amp=AMP, amp_align=True, amp_z=amp_z,
        long_axis_vertical=True, tight=True)
    # Report worst errors
    df_rms = float(np.sqrt(np.mean(df_diff**2)))
    df_max = float(np.max(np.abs(df_diff)))
    Fz_rms = float(np.sqrt(np.mean(Fz_diff**2)))
    Fz_max = float(np.max(np.abs(Fz_diff)))
    df_scale = max(float(np.abs(grid_result.df).max()), 1e-30)
    Fz_scale = max(float(np.abs(grid_result.Fz).max()), 1e-30)
    print(f"  Δdf: rms={df_rms:.6e} max={df_max:.6e}  (rel max={df_max/df_scale:.4e})")
    print(f"  ΔFz: rms={Fz_rms:.6e} max={Fz_max:.6e}  (rel max={Fz_max/Fz_scale:.4e})")
    print(f"REVIEW: {out_png}")
    return out_png


def plot_vjp_gradient(direct_result, afmulator, FEs, PPs, points_xyz, dtip,
                      scan_spec, atomPos, atomTypes, names, outdir):
    """Visual proof of differentiability: VJP gradient w.r.t. atom positions.

    Uses df_loss_seed to compute a loss (df vs zero target), then runs
    vjp_scan_morse_direct to get dL/d(atom positions). Shows:
      1. Per-atom |grad_xyz| as scatter overlaid on molecular structure
      2. Per-atom gradient components bar chart
    """
    # ── compute df loss seed (loss = 0.5 * mean(df^2) vs zero target) ──
    # FEs here is the flipped version (iz=0=lowest z, for postprocess/plotting).
    # But the VJP kernel expects the ORIGINAL kernel order (iz=0=highest z).
    # Compute loss/dL_dFEs from flipped FEs, then flip dL_dFEs back for the VJP.
    nz_scan = FEs.shape[2]
    target_df_full = np.zeros((FEs.shape[0], FEs.shape[1], FEs.shape[2]), dtype=np.float32)
    loss, df, dL_dFEs = afm_mod.df_loss_seed(FEs, target_df_full, dtip=dtip)
    # Flip dL_dFEs back to kernel order (iz=0=highest z) to match unflipped PPs
    dL_dFEs_kernel = dL_dFEs[:, :, ::-1, :].astype(np.float32)
    print(f"\n=== VJP: df loss vs zero target: L={loss:.6e} ===")

    # ── run VJP (PPs and dL_dFEs in original kernel order: iz=0=highest z) ──
    grad_theta, diag = afmulator.vjp_scan_morse_direct(
        PPs, points_xyz, dtip, dL_dFEs_kernel,
        workgroup_size=64, bAlloc=True)
    print(f"  grad_theta shape={grad_theta.shape}")
    print(f"  status: all zero = {np.all(diag['status_codes'] == 0)}")
    print(f"  residual worst = {diag['residual_norms'].max():.6e}")
    print(f"  lambda_min worst = {diag['lambda_mins'].min():.6e}")

    # ── plot 1: per-atom |grad_xyz| overlaid on molecule top view ──
    grad_xyz = grad_theta[:, :3]  # (nAtoms, 3)
    grad_mag = np.linalg.norm(grad_xyz, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: molecule top view with gradient-colored atoms
    ax = axes[0]
    apos = np.asarray(atomPos, dtype=np.float64)
    sc = ax.scatter(apos[:, 0], apos[:, 1], c=grad_mag, s=200, cmap='hot',
                    edgecolors='cyan', linewidths=0.5, zorder=5)
    for i, (x, y) in enumerate(apos[:, :2]):
        ax.annotate(names[i], (x, y), fontsize=7, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points', color='cyan')
    ax.set_aspect('equal')
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    ax.set_title(f'|∇_atom L| (df loss vs zero)\nmax={grad_mag.max():.4e}, min={grad_mag.min():.4e}')
    plt.colorbar(sc, ax=ax, label='|dL/d(atom xyz)|')
    ax.grid(True, alpha=0.2)

    # Right: per-atom gradient components bar chart
    ax = axes[1]
    na = len(atomPos)
    x = np.arange(na)
    w = 0.25
    ax.bar(x - w, grad_theta[:, 0], w, label='dx', color='red', alpha=0.7)
    ax.bar(x,     grad_theta[:, 1], w, label='dy', color='green', alpha=0.7)
    ax.bar(x + w, grad_theta[:, 2], w, label='dz', color='blue', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f'{n}({i})' for i, n in enumerate(names)], fontsize=7, rotation=45)
    ax.set_ylabel('dL/d(atom coord) [eV/Å²]')
    ax.set_title('Per-atom VJP gradient components')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_png = os.path.join(outdir, 'vjp_gradient_atoms.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"REVIEW: {out_png}")

    # ── plot 2: gradient also for R0, E0, Q parameters ──
    fig, ax = plt.subplots(figsize=(8, 4))
    grad_R0 = grad_theta[:, 3]
    grad_E0 = grad_theta[:, 4]
    grad_Q  = grad_theta[:, 5]
    x = np.arange(na)
    w = 0.25
    ax.bar(x - w, grad_R0, w, label='dR0', color='orange', alpha=0.7)
    ax.bar(x,     grad_E0, w, label='dE0', color='purple', alpha=0.7)
    ax.bar(x + w, grad_Q,  w, label='dQ',  color='brown',  alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f'{n}({i})' for i, n in enumerate(names)], fontsize=7, rotation=45)
    ax.set_ylabel('dL/d(param)')
    ax.set_title('Per-atom VJP gradient: Morse params (R0, E0) + Coulomb Q')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_png2 = os.path.join(outdir, 'vjp_gradient_params.png')
    fig.savefig(out_png2, dpi=140)
    plt.close(fig)
    print(f"REVIEW: {out_png2}")

    # ── plot 3: Finite-difference validation of the VJP ──
    # The whole point: analytic VJP vs brute-force re-relax-and-observe.
    # Perturb each atom's z by ±h, re-run forward, compute loss, central difference.
    print("\n  --- VJP finite-difference validation ---")
    h_pert = 0.01  # Å — perturbation for central difference
    na = len(atomPos)
    atoms_orig = afmulator.atoms_arr.copy()
    scan_p0 = np.array([float(scan_spec.scan_xs[0]), float(scan_spec.scan_yz[0]) if False else float(scan_spec.scan_ys[0]),
                        float(np.max(scan_spec.h_scan)) + float(atomPos[:, 2].max()) + BOND_LENGTH], dtype=np.float32)
    dx = float(scan_spec.scan_xs[1] - scan_spec.scan_xs[0]) if len(scan_spec.scan_xs) > 1 else 0.
    dy = float(scan_spec.scan_ys[1] - scan_spec.scan_ys[0]) if len(scan_spec.scan_ys) > 1 else 0.
    scan_da = np.array([dx, 0., 0.], dtype=np.float32)
    scan_db = np.array([0., dy, 0.], dtype=np.float32)
    nx, ny = len(scan_spec.scan_xs), len(scan_spec.scan_ys)
    nz = len(scan_spec.h_scan)

    grad_dz_fd = np.zeros(na, dtype=np.float64)
    for ia in range(na):
        # +h perturbation
        afmulator.atoms_arr = atoms_orig.copy()
        afmulator.atoms_arr[ia, 2] += h_pert
        FEs_p, _, PPs_p = afmulator.run_scan_morse_direct(
            nxy=(nx, ny), nz=nz, dtip=dtip,
            scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db,
            workgroup_size=64, bAlloc=True, return_pp=True)
        FEs_p = FEs_p[:, :, ::-1, :]
        _, _, dL_dFEs_p = afm_mod.df_loss_seed(FEs_p, np.zeros((nx, ny, nz), dtype=np.float32), dtip=dtip)
        loss_p = 0.5 * float(np.sum(dL_dFEs_p * FEs_p)) / max(float(np.sum(np.ones_like(FEs_p[:, :, :, 2]))), 1.0)
        # Actually recompute loss properly
        target = np.zeros((nx, ny, nz), dtype=np.float32)
        loss_p, _, _ = afm_mod.df_loss_seed(FEs_p, target, dtip=dtip)

        # -h perturbation
        afmulator.atoms_arr = atoms_orig.copy()
        afmulator.atoms_arr[ia, 2] -= h_pert
        FEs_m, _, PPs_m = afmulator.run_scan_morse_direct(
            nxy=(nx, ny), nz=nz, dtip=dtip,
            scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db,
            workgroup_size=64, bAlloc=True, return_pp=True)
        FEs_m = FEs_m[:, :, ::-1, :]
        loss_m, _, _ = afm_mod.df_loss_seed(FEs_m, target, dtip=dtip)

        grad_dz_fd[ia] = (loss_p - loss_m) / (2.0 * h_pert)
        print(f"  atom {ia} ({names[ia]}): analytic dz={grad_theta[ia, 2]:+.6e}  FD dz={grad_dz_fd[ia]:+.6e}  "
              f"rel_err={abs(grad_theta[ia, 2] - grad_dz_fd[ia]) / max(abs(grad_dz_fd[ia]), 1e-30):.4e}")

    # Restore original atoms
    afmulator.atoms_arr = atoms_orig.copy()

    # Plot analytic vs FD
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(na)
    ax.bar(x - 0.2, grad_theta[:, 2], 0.4, label='Analytic VJP (dL/dz)', color='blue', alpha=0.7)
    ax.bar(x + 0.2, grad_dz_fd,       0.4, label=f'Finite diff (h={h_pert}Å)',   color='red',  alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([f'{n}({i})' for i, n in enumerate(names)], fontsize=7, rotation=45)
    ax.set_ylabel('dL/d(atom z) [eV/Å]')
    ax.set_title(f'VJP validation: analytic vs finite-difference (df loss vs zero)\nh={h_pert}Å central difference')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    # Report worst relative error
    rel_errs = np.abs(grad_theta[:, 2] - grad_dz_fd) / np.maximum(np.abs(grad_dz_fd), 1e-30)
    worst = np.argmax(rel_errs)
    ax.text(0.02, 0.98, f'worst rel_err: atom {worst} ({names[worst]}) = {rel_errs[worst]:.4e}',
            transform=ax.transAxes, fontsize=8, va='top', bbox=dict(facecolor='yellow', alpha=0.5))
    fig.tight_layout()
    out_png3 = os.path.join(outdir, 'vjp_fd_validation.png')
    fig.savefig(out_png3, dpi=140)
    plt.close(fig)
    print(f"REVIEW: {out_png3}")
    return out_png, out_png2, out_png3


def plot_vjp_visual(direct_result, afmulator, FEs, PPs, points_xyz, dtip,
                    scan_spec, atomPos, atomTypes, names, outdir):
    """Visual differentiability: gradient arrows + perturbation effect on df.

    Two figures:
      1. Gradient arrows overlaid on molecule (top view) and on the df AFM image.
         Arrows show dL/d(x,y) — which direction to move each atom to most change the image.
      2. Perturbation effect: pick the atom with largest |grad|, displace it by a
         visible amount, re-run forward, show df_before / df_after / Δdf.
    """
    # ── re-run forward + VJP (atoms_arr may have been changed by FD validation) ──
    scan_xs, scan_ys = scan_spec.scan_xs, scan_spec.scan_ys
    nx, ny = len(scan_xs), len(scan_ys)
    nz = len(scan_spec.h_scan)
    z_start = float(np.max(scan_spec.h_scan)) + float(atomPos[:, 2].max()) + BOND_LENGTH
    dx = float(scan_xs[1] - scan_xs[0]) if len(scan_xs) > 1 else 0.
    dy = float(scan_ys[1] - scan_ys[0]) if len(scan_ys) > 1 else 0.
    scan_p0 = np.array([float(scan_xs[0]), float(scan_ys[0]), z_start], dtype=np.float32)
    scan_da = np.array([dx, 0., 0.], dtype=np.float32)
    scan_db = np.array([0., dy, 0.], dtype=np.float32)
    FEs_fresh, points_fresh, PPs_fresh = afmulator.run_scan_morse_direct(
        nxy=(nx, ny), nz=nz, dtip=dtip,
        scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db,
        workgroup_size=64, bAlloc=True, return_pp=True)
    FEs_flipped = FEs_fresh[:, :, ::-1, :]

    # ── compute VJP gradient ──
    target_df_full = np.zeros((nx, ny, nz), dtype=np.float32)
    loss, _, dL_dFEs = afm_mod.df_loss_seed(FEs_flipped, target_df_full, dtip=dtip)
    dL_dFEs_kernel = dL_dFEs[:, :, ::-1, :].astype(np.float32)
    grad_theta, diag = afmulator.vjp_scan_morse_direct(
        PPs_fresh, points_fresh, dtip, dL_dFEs_kernel, workgroup_size=64, bAlloc=True)
    grad_xyz = grad_theta[:, :3]
    grad_mag = np.linalg.norm(grad_xyz, axis=1)
    print(f"  VJP visual: |grad|_max={grad_mag.max():.4e} loss={loss:.6e}")

    # ── Figure 1: gradient arrows on molecule + df image ──
    apos = np.asarray(atomPos, dtype=np.float64)
    scan_xs, scan_ys = scan_spec.scan_xs, scan_spec.scan_ys
    extent = afm_utils.scan_extent(scan_xs, scan_ys)
    # Pick a representative df slice (middle height)
    df_map = direct_result.df[:, :, len(direct_result.heights) // 2]
    # Scale arrows for visibility — normalize to a fraction of the scan extent
    arrow_scale = float(max(scan_xs[-1] - scan_xs[0], scan_ys[-1] - scan_ys[0])) * 0.15
    gxy = grad_xyz[:, :2].copy()
    gmax = max(np.linalg.norm(gxy, axis=1).max(), 1e-30)
    gxy = gxy / gmax * arrow_scale  # normalize to arrow_scale for visibility

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: molecule top view with gradient arrows
    ax = axes[0]
    sc = ax.scatter(apos[:, 0], apos[:, 1], c=grad_mag, s=200, cmap='hot',
                    edgecolors='cyan', linewidths=0.8, zorder=5)
    ax.quiver(apos[:, 0], apos[:, 1], gxy[:, 0], gxy[:, 1],
              angles='xy', scale_units='xy', scale=1.0,
              color='lime', width=0.005, headwidth=4, headlength=5, zorder=6)
    for i, (x, y) in enumerate(apos[:, :2]):
        ax.annotate(names[i], (x, y), fontsize=7, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points', color='cyan')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    ax.set_title(f'∇L w.r.t. atom positions (top view)\n|∇|_max={grad_mag.max():.4e}, arrow=direction of max image change')
    plt.colorbar(sc, ax=ax, label='|dL/d(atom xyz)|')
    ax.grid(True, alpha=0.2)

    # Right: df image with gradient arrows overlaid at atom positions
    ax = axes[1]
    im = ax.imshow(df_map.T, origin='lower', extent=extent, cmap='gray', aspect='equal')
    ax.quiver(apos[:, 0], apos[:, 1], gxy[:, 0], gxy[:, 1],
              angles='xy', scale_units='xy', scale=1.0,
              color='lime', width=0.006, headwidth=4, headlength=5, zorder=10)
    ax.scatter(apos[:, 0], apos[:, 1], s=50, c='cyan', edgecolors='white',
               linewidths=0.5, zorder=11)
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    h_mid = direct_result.heights[len(direct_result.heights) // 2]
    ax.set_title(f'df image @h={h_mid:.1f}Å with ∇L arrows\narrows point toward max image change')
    plt.colorbar(im, ax=ax, label='df [Hz]')
    fig.suptitle(f'VJP gradient visualization — df loss vs zero, benzene (K_LAT={K_LAT_NM}N/m K_RAD={K_RAD} L={BOND_LENGTH}Å)', fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(outdir, 'vjp_gradient_arrows.png')
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"REVIEW: {out_png}")

    # ── Figure 2: perturbation effect on df ──
    # Pick atom with largest |grad_dz| (most sensitive to z-displacement)
    grad_dz = grad_theta[:, 2]
    i_pert = int(np.argmax(np.abs(grad_dz)))
    pert_z = 0.1  # Å — visible perturbation (10x the FD step)
    print(f"  Perturbing atom {i_pert} ({names[i_pert]}) by dz={pert_z:+.2f}Å "
          f"(grad_dz={grad_dz[i_pert]:+.4e})")

    # Re-run forward with perturbed atom
    atoms_orig = afmulator.atoms_arr.copy()
    afmulator.atoms_arr[i_pert, 2] += pert_z
    FEs_pert, _, _ = afmulator.run_scan_morse_direct(
        nxy=(nx, ny), nz=nz, dtip=dtip,
        scan_p0=scan_p0, scan_da=scan_da, scan_db=scan_db,
        workgroup_size=64, bAlloc=True, return_pp=True)
    FEs_pert = FEs_pert[:, :, ::-1, :]
    result_pert = afm_utils.shared_postprocess(FEs_pert, scan_spec, backend_name='morse_direct_perturbed')
    # Restore
    afmulator.atoms_arr = atoms_orig.copy()

    # Show df at 3 heights: before, after, difference
    n_show = min(3, len(direct_result.heights))
    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 10))
    if n_show == 1:
        axes = axes.reshape(3, 1)
    h_indices = np.linspace(0, len(direct_result.heights) - 1, n_show, dtype=int)

    for col, ih in enumerate(h_indices):
        h = direct_result.heights[ih]
        df_before = direct_result.df[:, :, ih]
        df_after = result_pert.df[:, :, ih]
        df_diff = df_after - df_before
        # Row 0: before
        ax = axes[0, col]
        im0 = ax.imshow(df_before.T, origin='lower', extent=extent, cmap='gray', aspect='equal')
        ax.scatter(apos[:, 0], apos[:, 1], s=30, c='cyan', edgecolors='white', linewidths=0.3, zorder=5)
        # Mark perturbed atom
        ax.scatter(apos[i_pert, 0], apos[i_pert, 1], s=120, c='red', marker='x',
                   linewidths=2, zorder=6)
        ax.set_title(f'df before @h={h:.1f}Å')
        if col == 0: ax.set_ylabel('BEFORE perturbation')
        plt.colorbar(im0, ax=ax, fraction=0.046)
        # Row 1: after
        ax = axes[1, col]
        im1 = ax.imshow(df_after.T, origin='lower', extent=extent, cmap='gray', aspect='equal')
        ax.scatter(apos[:, 0], apos[:, 1], s=30, c='cyan', edgecolors='white', linewidths=0.3, zorder=5)
        apos_pert = apos.copy(); apos_pert[i_pert, 2] += pert_z
        ax.scatter(apos_pert[i_pert, 0], apos_pert[i_pert, 1], s=120, c='red', marker='x',
                   linewidths=2, zorder=6)
        ax.set_title(f'df after @h={h:.1f}Å')
        if col == 0: ax.set_ylabel(f'AFTER (atom {i_pert} dz={pert_z:+.1f}Å)')
        plt.colorbar(im1, ax=ax, fraction=0.046)
        # Row 2: difference
        ax = axes[2, col]
        # Use symmetric clim
        vmax = max(abs(df_diff.min()), abs(df_diff.max()))
        im2 = ax.imshow(df_diff.T, origin='lower', extent=extent, cmap='bwr', aspect='equal',
                        vmin=-vmax, vmax=vmax)
        ax.scatter(apos[:, 0], apos[:, 1], s=30, c='cyan', edgecolors='white', linewidths=0.3, zorder=5)
        ax.scatter(apos[i_pert, 0], apos[i_pert, 1], s=120, c='red', marker='x',
                   linewidths=2, zorder=6)
        ax.set_title(f'Δdf @h={h:.1f}Å (range ±{vmax:.2e})')
        if col == 0: ax.set_ylabel('DIFFERENCE (after − before)')
        plt.colorbar(im2, ax=ax, fraction=0.046)

    fig.suptitle(f'Perturbation effect: atom {i_pert} ({names[i_pert]}) displaced dz={pert_z:+.1f}Å → df image change\n'
                 f'VJP predicts dL/dz={grad_dz[i_pert]:+.4e} (direction {"↑" if grad_dz[i_pert]>0 else "↓"} increases loss)',
                 fontsize=10)
    fig.tight_layout()
    out_png2 = os.path.join(outdir, 'vjp_perturbation_effect.png')
    fig.savefig(out_png2, dpi=140)
    plt.close(fig)
    print(f"REVIEW: {out_png2}")
    return out_png, out_png2


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"=== Morse Direct vs GridFF comparison ===")
    print(f"    params: h=[{H_MIN},{H_MAX}] dz={H_STEP} amp={AMP} K_LAT={K_LAT_NM}N/m K_RAD={K_RAD} L={BOND_LENGTH}Å")
    print(f"    outdir: {OUTDIR}")

    atomPos, atomTypes, names = _load_benzene()
    print(f"    benzene: {len(atomPos)} atoms, z_top={atomPos[:,2].max():.2f}Å")

    # ── 1. Grid-based Morse (same as CLI: run_spm.py afm --model morse) ──
    print("\n--- GridFF Morse ---")
    grid_result, scan_spec = run_grid_morse(atomPos, atomTypes, OUTDIR)

    # ── 2. Direct brute-force Morse (differentiable kernel) ──
    print("\n--- Direct Morse ---")
    direct_result, afmulator, FEs, points_xyz, PPs = run_direct_morse(atomPos, atomTypes, scan_spec)

    # ── 3. E(z)/Fz(z) curves above a carbon atom (z-alignment diagnostic) ──
    print("\n--- Plot: z-curves above carbon ---")
    plot_z_curves(grid_result, direct_result, scan_spec, atomPos, atomTypes, names, OUTDIR)

    # ── 4. Side-by-side comparison strip ──
    print("\n--- Plot: compare strip ---")
    plot_compare_strip(grid_result, direct_result, scan_spec, atomPos, OUTDIR)

    # ── 4. Difference strip ──
    print("\n--- Plot: difference strip ---")
    plot_difference_strip(grid_result, direct_result, scan_spec, atomPos, OUTDIR)

    # ── 5. VJP gradient (differentiability proof) ──
    print("\n--- Plot: VJP gradient ---")
    plot_vjp_gradient(direct_result, afmulator, FEs, PPs, points_xyz, -H_STEP,
                      scan_spec, atomPos, atomTypes, names, OUTDIR)

    # ── 6. VJP visual: gradient arrows + perturbation effect ──
    print("\n--- Plot: VJP visual ---")
    plot_vjp_visual(direct_result, afmulator, FEs, PPs, points_xyz, -H_STEP,
                    scan_spec, atomPos, atomTypes, names, OUTDIR)

    print(f"\n=== DONE — all plots in {OUTDIR} ===")


if __name__ == '__main__':
    main()
