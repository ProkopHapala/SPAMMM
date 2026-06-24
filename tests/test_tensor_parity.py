#!/usr/bin/env python3
"""
test_tensor_parity.py — Verify GPU tensor kernels match CPU numpy reference.

Builds directly on test_folded_surface_scan.py:
  - Same coordinate system (Z_SURF_TOP = -3.25, z_rel relative to surface)
  - Same reference generation (GPU brute-force for Morse, Ewald2D for Coulomb)
  - Same z-scan range, fit range, alphas, sites, probes

Strategy:
  1. Fit folded basis coefficients using fit_folded_surface_basis with
     powered basis (Phi^3 for pauli, Phi^2 for london, Phi^1 for coulomb)
     to match the kernel's cubic formula E = B*(cCoulomb + B*(cLondon + B*cPauli)).
  2. Build CPU reference: evaluate the cubic formula at uvz points.
  3. Evaluate GPU tensor kernel at same grid points.
  4. Compare GPU vs CPU for:
     - Morse only (coulomb coeffs = 0)
     - Coulomb only (pauli/london coeffs = 0)
     - Combined (all 3 components)
  5. Plot z-scans with brute-force reference + CPU fit + GPU tensor, matching
     the style of test_folded_surface_scan.py plots.

Run:  python tests/test_tensor_parity.py
"""

import os, sys, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from spammm.forcefields.MolecularDynamics import MolecularDynamics
from spammm.AtomicSystem import AtomicSystem
from spammm.surfaces.Ewald2D import Ewald2D

# =============================================================================
# Configuration — MUST match test_folded_surface_scan.py exactly
# =============================================================================

NACL_SUBSTRATE = os.path.join(_proj_root, 'data', 'substrates', 'NaCl_1x1_L3.xyz')
PLOT_DIR = os.path.join(_proj_root, 'debug', f'{datetime.date.today()}_tensor_parity')

R_O = 1.7500; SQRT_E_O = np.sqrt(0.00260184625)
PROBE_REQ = np.array([[R_O, SQRT_E_O, -0.5, 0.0]], dtype=np.float32)
Q_PROBE = -0.5  # charge of probe for Coulomb reference

Z_SURF_TOP = -3.25
SITES = {'Na': np.array([0.0, 0.0]), 'Cl': np.array([2.0, 2.0])}
LATTICE_A = 4.0

Z_SCAN_REL = np.linspace(0.3, 10.0, 120)
FIT_Z_RANGE = (1.5, 8.0)  # relative to surface top
FIT_NXY = 32
FIT_NZ_SAMP = 60

MORSE_NPBC = (4, 4, 0)
MORSE_ALPHA = 1.8

# Z-basis alphas (from test_folded_surface_scan.py)
MORSE_ALPHAS = np.array([1.0, 1.8, 2.7, 3.6, 5.0], dtype=np.float32)  # /Å

# Poly basis config (from test_folded_surface_scan.py)
POLY_R = 14.0  # cutoff [Å]
# Kernel uses SEQUENTIAL powers: m_start, m_start+1, ..., m_start+Nz-1
# (unlike scan test which used arbitrary powers [4,8,16,32,64])
MORSE_M_START = 8    # powers [8,9,10,11,12], α_eff = 0.57..0.86 /Å
COULOMB_M_START = 0  # powers [0,1,2,3,4], α_eff = 0..0.29 /Å (n=0=constant)

NU = 4; NV = 4; NZ = 5

RTOL = 1e-3
ATOL = 1e-4

# Absolute z helpers
Z0_ABS = Z_SURF_TOP + FIT_Z_RANGE[0]  # = -1.75
Z1_ABS = Z_SURF_TOP + FIT_Z_RANGE[1]  # =  4.75
Z_SCAN_ABS = Z_SURF_TOP + Z_SCAN_REL   # absolute z for scan


# =============================================================================
# CPU reference: evaluate E = sum_i B_i * (cCoulomb_i + B_i*(cLondon_i + B_i*cPauli_i))
# =============================================================================

def cpu_eval_energy(uvz, basis_params, coeff4):
    """Evaluate E = B*(c.z + B*(c.y + B*c.x)) at uvz points.
    uvz: (N, 3) absolute (u, v, z) — z is absolute coordinate.
    basis_params: (nbasis, 4) array of (ku, kv, alpha, z0) — z0 is absolute.
    coeff4: (ntypes, nbasis, 4) array of (cPauli, cLondon, cCoulomb, cH).
    Returns: E (ntypes, N)
    """
    uvz = np.asarray(uvz, dtype=np.float64)
    bp = np.asarray(basis_params, dtype=np.float64)
    u = uvz[:, 0][:, None]; v = uvz[:, 1][:, None]; z = uvz[:, 2][:, None]
    ku = bp[None, :, 0]; kv = bp[None, :, 1]; az = bp[None, :, 2]; z0 = bp[None, :, 3]
    bx = np.cos((2.0 * np.pi) * ku * u)
    by = np.cos((2.0 * np.pi) * kv * v)
    bz = np.exp(-az * np.maximum(0.0, z - z0))
    B = bx * by * bz  # (N, nbasis)
    coeff4 = np.asarray(coeff4, dtype=np.float64)
    cP = coeff4[:, :, 0]; cL = coeff4[:, :, 1]; cC = coeff4[:, :, 2]
    E = cC[:, None, :] * B[None, :, :] + cL[:, None, :] * B[None, :, :]**2 + cP[:, None, :] * B[None, :, :]**3
    return E.sum(axis=2)  # (ntypes, N)


def cpu_eval_force(uvz, basis_params, coeff4, lvec2d):
    """Evaluate force (fx, fy, fz) at uvz points. Returns F: (ntypes, N, 3)."""
    uvz = np.asarray(uvz, dtype=np.float64)
    bp = np.asarray(basis_params, dtype=np.float64)
    coeff4 = np.asarray(coeff4, dtype=np.float64)
    ntypes = coeff4.shape[0]; N = uvz.shape[0]; nbasis = bp.shape[0]
    u = uvz[:, 0][:, None]; v = uvz[:, 1][:, None]; z = uvz[:, 2][:, None]
    ku = bp[None, :, 0]; kv = bp[None, :, 1]; az = bp[None, :, 2]; z0 = bp[None, :, 3]
    phix = 2.0 * np.pi * ku; phiy = 2.0 * np.pi * kv
    bx = np.cos(phix * u); sx = np.sin(phix * u)
    by = np.cos(phiy * v); sy = np.sin(phiy * v)
    dz = np.maximum(0.0, z - z0)
    bz = np.exp(-az * dz)
    B = bx * by * bz  # (N, nbasis)
    dBdu = -phix * sx * by * bz
    dBdv = bx * (-phiy * sy) * bz
    dBdz = bx * by * (-az * bz)
    dBdz[z <= z0] = 0.0
    cP = coeff4[:, :, 0]; cL = coeff4[:, :, 1]; cC = coeff4[:, :, 2]
    dEdB = cC[:, None, :] + B[None, :, :] * (2.0 * cL[:, None, :] + B[None, :, :] * 3.0 * cP[:, None, :])
    dEdu = (dEdB * dBdu[None, :, :]).sum(axis=2)
    dEdv = (dEdB * dBdv[None, :, :]).sum(axis=2)
    dEdz = (dEdB * dBdz[None, :, :]).sum(axis=2)
    ax, bx_l, ay, by_l = lvec2d
    det = ax * by_l - bx_l * ay
    inv = np.array([by_l / det, -bx_l / det, -ay / det, ax / det])
    Fx = -(dEdu * inv[0] + dEdv * inv[2])
    Fy = -(dEdu * inv[1] + dEdv * inv[3])
    Fz = -dEdz
    return np.stack([Fx, Fy, Fz], axis=-1)


# =============================================================================
# Reference generation — same as test_folded_surface_scan.py
# =============================================================================

def _make_transforms(positions):
    T = np.zeros((len(positions), 3, 4), dtype=np.float32)
    T[:, 0, 0] = 1.0; T[:, 1, 1] = 1.0; T[:, 2, 2] = 1.0
    T[:, :, 3] = positions
    return T.reshape(-1, 12)


def _z_scan_positions(site_xy, z_abs):
    n = len(z_abs)
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 0] = site_xy[0]; pos[:, 1] = site_xy[1]; pos[:, 2] = z_abs
    return pos


def compute_morse_reference(positions, probe_REQ=PROBE_REQ, nPBC=MORSE_NPBC, alpha_morse=MORSE_ALPHA):
    """GPU brute-force Morse reference: pauli + london."""
    md = MolecularDynamics(nloc=32, debug_build_options='-DDBG_UFF=0')
    md.init_rigid_molecule_batch(np.zeros((1, 3), dtype=np.float32), probe_REQ, nSystems=max(len(positions), 1))
    md.set_surface(NACL_SUBSTRATE, nPBC=nPBC, alpha_morse=alpha_morse, bMacro=True)
    T = _make_transforms(positions)
    out = md.eval_rigid_getSurfMorse_components(T, chunk_size=md.nSystems, components=('pauli', 'london'))
    return out['pauli'] + out['london']


_ewald_cache = None

def _get_ewald():
    """Lazy-init Ewald2D (cached so we don't re-read substrate every call)."""
    global _ewald_cache
    if _ewald_cache is None:
        mol = AtomicSystem(fname=NACL_SUBSTRATE)
        apos = np.asarray(mol.apos, dtype=float)
        surf_lvec = np.asarray(mol.lvec, dtype=float)
        a_vec = surf_lvec[0, :2]; b_vec = surf_lvec[1, :2]
        _ewald_cache = Ewald2D(a_vec, b_vec, apos[:, 0], apos[:, 1], apos[:, 2],
                               np.asarray(mol.qs, dtype=float), n_harm=6)
    return _ewald_cache


def compute_coulomb_reference_ewald(positions, q_probe=Q_PROBE):
    """Ewald2D periodic Coulomb reference: phi * q_probe.
    For z-scans (few unique x,y): use phi_full_1d per (x,y) pair.
    For grids: use phi_vacuum_xy per z-layer (vectorized over XY)."""
    ew = _get_ewald()
    positions = np.asarray(positions, dtype=np.float64)
    N = len(positions)
    # Detect if this is a regular grid (many unique z, few unique x,y → z-scan)
    # vs a 3D grid (many unique z, many unique x,y)
    unique_xy = set(zip(positions[:, 0].round(6), positions[:, 1].round(6)))
    if len(unique_xy) <= 4:
        # Z-scan mode: few (x,y) sites, many z values → use phi_full_1d
        E = np.zeros(N, dtype=np.float64)
        for (x0, y0) in unique_xy:
            mask = (positions[:, 0].round(6) == x0) & (positions[:, 1].round(6) == y0)
            z_arr = positions[mask, 2]
            phi = ew.phi_full_1d(float(x0), float(y0), z_arr)
            E[mask] = phi * q_probe
        return E
    else:
        # Grid mode: use phi_vacuum_xy per z-layer (vectorized over XY)
        E = np.zeros(N, dtype=np.float64)
        unique_z = np.unique(positions[:, 2].round(6))
        for z_abs in unique_z:
            mask = positions[:, 2].round(6) == z_abs
            X = positions[mask, 0].reshape(1, -1)
            Y = positions[mask, 1].reshape(1, -1)
            phi = ew.phi_vacuum_xy(X, Y, float(z_abs))
            E[mask] = phi.ravel() * q_probe
        return E


def compute_combined_reference(positions):
    """Morse + Coulomb reference."""
    E_morse = compute_morse_reference(positions)
    E_coul = compute_coulomb_reference_ewald(positions)
    return E_morse + E_coul


# =============================================================================
# GPU tensor kernel evaluation
# =============================================================================

def _eval_gpu_zscan(md, site_xy, z_abs, chunk_size=256):
    """Run GPU tensor kernel at z-scan positions, return E and F."""
    n = len(z_abs)
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[:, 0] = site_xy[0]; pos[:, 1] = site_xy[1]; pos[:, 2] = z_abs
    transforms = np.zeros((n, 3, 4), dtype=np.float32)
    transforms[:, 0, 0] = 1.0; transforms[:, 1, 1] = 1.0; transforms[:, 2, 2] = 1.0
    transforms[:, :, 3] = pos
    out = md.eval_rigid_getSurfFolded(transforms.reshape(-1, 12), chunk_size=chunk_size)
    E = out['total']
    # Get forces
    import pyopencl as cl
    float_size = np.float32().itemsize
    sys_bytes = md.nvecs * 4 * float_size
    nch = min(chunk_size, n)
    all_F = np.zeros((n, 3), dtype=np.float32)
    for i0 in range(0, n, nch):
        nch_i = min(nch, n - i0)
        md.upload_rigid_transforms(transforms[i0:i0+nch_i].reshape(-1, 3, 4), iSys0=0)
        cl.enqueue_fill_buffer(md.queue, md.buffer_dict['aforce'], np.zeros(1, dtype=np.float32), 0, nch_i * sys_bytes)
        md.queue.finish()
        md.run_getSurfFolded(nSystems=nch_i)
        md.queue.finish()
        aforce = np.empty((nch_i, md.nvecs, 4), dtype=np.float32)
        md.fromGPU('aforce', aforce)
        md.queue.finish()
        all_F[i0:i0+nch_i] = aforce[:, :md.natoms, :3].reshape(nch_i, 3)
    return E, all_F


def _eval_gpu_2d(md, z_abs, nxy, chunk_size=256):
    """Run GPU on 2D xy grid at fixed z, return E reshaped (nxy, nxy)."""
    us = np.linspace(0.0, 1.0, nxy, endpoint=False, dtype=np.float32)
    vs = np.linspace(0.0, 1.0, nxy, endpoint=False, dtype=np.float32)
    pos = []
    for v in vs:
        for u in us:
            pos.append([u * LATTICE_A, v * LATTICE_A, z_abs])
    pos = np.array(pos, dtype=np.float32)
    transforms = np.zeros((len(pos), 3, 4), dtype=np.float32)
    transforms[:, 0, 0] = 1.0; transforms[:, 1, 1] = 1.0; transforms[:, 2, 2] = 1.0
    transforms[:, :, 3] = pos
    out = md.eval_rigid_getSurfFolded(transforms.reshape(-1, 12), chunk_size=chunk_size)
    return out['total'].reshape(nxy, nxy)


# =============================================================================
# Plotting — matching test_folded_surface_scan.py style
# =============================================================================

def _safe_name(s):
    return s.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')


def _set_ylim_symmetric(ax, E_ref, z_rel=None, fit_range=None):
    if z_rel is not None and fit_range is not None:
        mask = (z_rel >= fit_range[0]) & (z_rel <= fit_range[1])
        E_sub = E_ref[mask]
    else:
        E_sub = E_ref
    vmax = float(np.max(np.abs(E_sub)))
    ax.set_ylim(-vmax * 1.2, vmax * 1.2)


def _z_basis_values(z_rel, alphas):
    """Compute z-basis functions exp(-alpha * max(0, z-z0)) for plotting."""
    z0 = FIT_Z_RANGE[0]
    dz = np.maximum(0.0, z_rel - z0)
    return np.stack([np.exp(-a * dz) for a in alphas], axis=1)


def plot_zscan(z_rel, E_ref, E_cpu, E_gpu, F_cpu, F_gpu, label, site, save_dir, alphas=None):
    """Plot E(z) + Fz(z) + z-basis functions: Reference + CPU fit + GPU tensor.
    Style matches test_folded_surface_scan.py: top=energy, middle=force, bottom=basis."""
    if alphas is None:
        alphas = MORSE_ALPHAS
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 2]})
    # --- Top: Energy ---
    ax = axes[0]
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.plot(z_rel, E_ref, 'k:', label='Reference (brute-force)', linewidth=2)
    ax.plot(z_rel, E_cpu, 'b-', label='CPU (folded fit)', linewidth=1)
    ax.plot(z_rel, E_gpu, 'r--', label='GPU (tensor kernel)', linewidth=1)
    ax.axvspan(FIT_Z_RANGE[0], FIT_Z_RANGE[1], alpha=0.1, color='orange', label='Fit region')
    ax.set_ylabel('Energy [eV]')
    # Symmetric y-limits from fit-region reference (like scan test)
    fit_mask = (z_rel >= FIT_Z_RANGE[0]) & (z_rel <= FIT_Z_RANGE[1])
    if fit_mask.any() and np.any(E_ref[fit_mask] != 0):
        _set_ylim_symmetric(ax, E_ref, z_rel=z_rel, fit_range=FIT_Z_RANGE)
    ax.set_title(f'{label} — E(z) above {site}')
    ax.legend(loc='best', fontsize=8)
    # --- Middle: Force Fz ---
    ax = axes[1]
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.plot(z_rel, F_cpu[:, 2], 'b-', label='CPU Fz', linewidth=1)
    ax.plot(z_rel, F_gpu[:, 2], 'r--', label='GPU Fz', linewidth=1)
    ax.axvspan(FIT_Z_RANGE[0], FIT_Z_RANGE[1], alpha=0.1, color='orange')
    ax.set_ylabel('Fz [eV/Å]')
    ax.set_title(f'{label} — Fz(z) above {site}')
    ax.legend(fontsize=8)
    # --- Bottom: Z-basis functions ---
    ax = axes[2]
    bz_vals = _z_basis_values(z_rel, alphas)
    cmap = plt.cm.plasma
    for i in range(len(alphas)):
        ax.plot(z_rel, bz_vals[:, i], color=cmap(i / max(len(alphas)-1, 1)),
                label=f'α={alphas[i]:.1f}', linewidth=1.5)
    ax.axvspan(FIT_Z_RANGE[0], FIT_Z_RANGE[1], alpha=0.1, color='orange')
    ax.set_xlabel('z above surface [Å]'); ax.set_ylabel('Basis amplitude')
    ax.set_title(f'Z-basis functions (exp decay)')
    ax.legend(fontsize=8, ncol=len(alphas))
    fig.tight_layout()
    fname = f'zscan_{_safe_name(label)}_{site}.png'
    fig.savefig(os.path.join(save_dir, fname), dpi=120)
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_2d_map(E_cpu_2d, E_gpu_2d, z_rel, label, save_dir):
    """Plot 2D energy maps: CPU, GPU, |difference| at a fixed z."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = max(np.abs(E_cpu_2d).max(), np.abs(E_gpu_2d).max())
    for ax, data, title in [(axes[0], E_cpu_2d, 'CPU (folded fit)'), (axes[1], E_gpu_2d, 'GPU (tensor kernel)')]:
        im = ax.imshow(data, origin='lower', extent=[0, LATTICE_A, 0, LATTICE_A],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{label} — {title} (z={z_rel:.2f} Å)')
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
        plt.colorbar(im, ax=ax, label='eV')
    diff = E_gpu_2d - E_cpu_2d
    dmax = np.abs(diff).max()
    im = axes[2].imshow(diff, origin='lower', extent=[0, LATTICE_A, 0, LATTICE_A],
                        cmap='RdBu_r', vmin=-dmax if dmax > 0 else -1, vmax=dmax if dmax > 0 else 1)
    axes[2].set_title(f'{label} — |GPU-CPU| max={dmax:.2e}')
    axes[2].set_xlabel('x [Å]'); axes[2].set_ylabel('y [Å]')
    plt.colorbar(im, ax=axes[2], label='eV')
    fig.tight_layout()
    fname = f'map_{_safe_name(label)}_z{z_rel:.2f}.png'
    fig.savefig(os.path.join(save_dir, fname), dpi=120)
    plt.close(fig)
    print(f'  Saved {fname}')


# =============================================================================
# Test
# =============================================================================

def run_test():
    print("=" * 70)
    print("Tensor Kernel Parity Test: GPU vs CPU")
    print("=" * 70)
    print(f"  Z_SURF_TOP = {Z_SURF_TOP}")
    print(f"  Fit region (rel): {FIT_Z_RANGE[0]}-{FIT_Z_RANGE[1]} Å above surface")
    print(f"  Fit region (abs): {Z0_ABS}-{Z1_ABS} Å")
    print(f"  Z-scan (rel): {Z_SCAN_REL[0]}-{Z_SCAN_REL[-1]} Å ({len(Z_SCAN_REL)} pts)")
    print(f"  Alphas: {MORSE_ALPHAS.tolist()}")

    # --- Setup MD with surface ---
    md = MolecularDynamics(nloc=32, debug_build_options='-DDBG_UFF=0')
    md.init_rigid_molecule_batch(np.zeros((1, 3), dtype=np.float32), PROBE_REQ, nSystems=256)
    md.set_surface(NACL_SUBSTRATE, nPBC=MORSE_NPBC, alpha_morse=MORSE_ALPHA, bMacro=True)

    # --- Fit folded basis with 3 components ---
    # Use Ewald2D for Coulomb reference (same as scan test)
    print("\n--- Fitting folded basis (pauli, london, coulomb) ---")
    md.set_folded_kernel_kind('tensor')
    md.fit_folded_surface_basis(
        surf_xyz=NACL_SUBSTRATE, nPBC=MORSE_NPBC,
        z_range=(Z0_ABS, Z1_ABS),  # absolute z for fit region
        nu=NU, nv=NV, nz=NZ,
        nxy=FIT_NXY, nz_samp=FIT_NZ_SAMP,
        alpha_morse=MORSE_ALPHA,
        components=('pauli', 'london', 'coulomb'),
        coulomb_solver='ewald2d',
        custom_alphas=MORSE_ALPHAS,
    )

    fp = md.folded_params
    basis_params = fp['basis_params']
    coeff_sets = fp['coeff_sets']
    ntypes = coeff_sets['pauli'].shape[0]
    nbasis = basis_params.shape[0]
    lvec2d = fp['basis_lvec2d']
    lvec2d_flat = np.array([lvec2d[0, 0], lvec2d[1, 0], lvec2d[0, 1], lvec2d[1, 1]], dtype=np.float32)

    print(f"  ntypes={ntypes}, nbasis={nbasis}, nu={NU}, nv={NV}, nz={NZ}")
    print(f"  coeff_sets keys: {sorted(coeff_sets.keys())}")
    print(f"  basis z0={basis_params[0, 3]:.4f} (abs), alphas={np.unique(basis_params[:, 2]).tolist()}")

    # --- Build float4 coefficients (cPauli, cLondon, cCoulomb, cH=0) ---
    coeff4_natural = np.zeros((ntypes, nbasis, 4), dtype=np.float64)
    coeff4_natural[:, :, 0] = coeff_sets['pauli']
    coeff4_natural[:, :, 1] = coeff_sets['london']
    coeff4_natural[:, :, 2] = coeff_sets['coulomb']

    # --- Generate test grid (absolute z) ---
    us = np.linspace(0.0, 1.0, 16, endpoint=False, dtype=np.float64)
    vs = np.linspace(0.0, 1.0, 16, endpoint=False, dtype=np.float64)
    zs_abs = np.linspace(Z0_ABS + 0.5, Z1_ABS - 0.5, 20, dtype=np.float64)
    xyz_pts = []; uvz_pts = []
    for z in zs_abs:
        for v in vs:
            for u in us:
                xyz_pts.append([u * LATTICE_A, v * LATTICE_A, z])
                uvz_pts.append([u, v, z])
    xyz_pts = np.array(xyz_pts, dtype=np.float32)
    uvz_pts = np.array(uvz_pts, dtype=np.float64)
    N = len(xyz_pts)
    print(f"\n  Test grid: 16x16x20 = {N} points (abs z {zs_abs[0]:.2f}-{zs_abs[-1]:.2f})")

    # --- CPU reference (folded basis evaluation) ---
    print("\n--- CPU reference evaluation ---")
    E_cpu = cpu_eval_energy(uvz_pts, basis_params, coeff4_natural)
    F_cpu = cpu_eval_force(uvz_pts, basis_params, coeff4_natural, lvec2d_flat)
    print(f"  E_cpu shape={E_cpu.shape}, range=[{E_cpu.min():.6f}, {E_cpu.max():.6f}]")
    print(f"  F_cpu shape={F_cpu.shape}, |F|max={np.max(np.linalg.norm(F_cpu, axis=-1)):.6f}")

    # --- GPU tensor kernel evaluation ---
    print("\n--- GPU tensor kernel evaluation ---")
    transforms = np.zeros((N, 3, 4), dtype=np.float32)
    transforms[:, 0, 0] = 1.0; transforms[:, 1, 1] = 1.0; transforms[:, 2, 2] = 1.0
    transforms[:, :, 3] = xyz_pts
    out = md.eval_rigid_getSurfFolded(transforms.reshape(-1, 12), chunk_size=256)
    E_gpu = out['total']
    print(f"  E_gpu shape={E_gpu.shape}, range=[{E_gpu.min():.6f}, {E_gpu.max():.6f}]")

    # GPU forces (first 256 points)
    import pyopencl as cl
    float_size = np.float32().itemsize
    sys_bytes = md.nvecs * 4 * float_size
    nch = min(256, N)
    md.upload_rigid_transforms(transforms[:nch].reshape(-1, 3, 4), iSys0=0)
    cl.enqueue_fill_buffer(md.queue, md.buffer_dict['aforce'], np.zeros(1, dtype=np.float32), 0, nch * sys_bytes)
    md.queue.finish()
    md.run_getSurfFolded(nSystems=nch)
    md.queue.finish()
    aforce = np.empty((nch, md.nvecs, 4), dtype=np.float32)
    md.fromGPU('aforce', aforce)
    md.queue.finish()
    F_gpu = aforce[:, :md.natoms, :3].reshape(nch, 3)

    # --- Parity comparison (GPU vs CPU folded eval) ---
    print("\n" + "=" * 70)
    print("PARITY: GPU tensor vs CPU folded eval (combined)")
    print("=" * 70)
    E_cpu_flat = E_cpu[0]; F_cpu_flat = F_cpu[0]
    E_diff = np.abs(E_gpu - E_cpu_flat)
    E_rel = E_diff / (np.abs(E_cpu_flat) + 1e-10)
    print(f"  Energy: max|diff|={E_diff.max():.6e}, max|rel|={E_rel.max():.6e}")
    F_diff = np.linalg.norm(F_gpu - F_cpu_flat[:nch], axis=-1)
    F_rel = F_diff / (np.linalg.norm(F_cpu_flat[:nch], axis=-1) + 1e-10)
    print(f"  Force:  max|diff|={F_diff.max():.6e}, max|rel|={F_rel.max():.6e}")

    # --- Morse-only and Coulomb-only parity ---
    orig_coulomb = coeff_sets['coulomb'].copy()
    orig_pauli = coeff_sets['pauli'].copy()
    orig_london = coeff_sets['london'].copy()

    # Morse only
    coeff4_morse = coeff4_natural.copy(); coeff4_morse[:, :, 2] = 0.0
    E_cpu_morse = cpu_eval_energy(uvz_pts, basis_params, coeff4_morse)[0]
    coeff_sets['coulomb'] = np.zeros_like(orig_coulomb)
    if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
    md._set_folded_coefficients(None); md._tensor_coeffs_set = True
    E_gpu_morse = md.eval_rigid_getSurfFolded(transforms.reshape(-1, 12), chunk_size=256)['total']
    E_diff_m = np.abs(E_gpu_morse - E_cpu_morse); E_rel_m = E_diff_m / (np.abs(E_cpu_morse) + 1e-10)
    print(f"\n  Morse only:  max|diff|={E_diff_m.max():.6e}, max|rel|={E_rel_m.max():.6e}")
    coeff_sets['coulomb'] = orig_coulomb

    # Coulomb only
    coeff4_coul = coeff4_natural.copy(); coeff4_coul[:, :, 0] = 0.0; coeff4_coul[:, :, 1] = 0.0
    E_cpu_coul = cpu_eval_energy(uvz_pts, basis_params, coeff4_coul)[0]
    coeff_sets['pauli'] = np.zeros_like(orig_pauli); coeff_sets['london'] = np.zeros_like(orig_london)
    if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
    md._set_folded_coefficients(None); md._tensor_coeffs_set = True
    E_gpu_coul = md.eval_rigid_getSurfFolded(transforms.reshape(-1, 12), chunk_size=256)['total']
    E_diff_c = np.abs(E_gpu_coul - E_cpu_coul); E_rel_c = E_diff_c / (np.abs(E_cpu_coul) + 1e-10)
    print(f"  Coulomb only: max|diff|={E_diff_c.max():.6e}, max|rel|={E_rel_c.max():.6e}")
    coeff_sets['pauli'] = orig_pauli; coeff_sets['london'] = orig_london

    # --- Plots ---
    print("\n" + "=" * 70)
    print("PLOTS")
    print("=" * 70)
    os.makedirs(PLOT_DIR, exist_ok=True)

    def _cpu_zscan(site_xy, z_abs_arr, coeff4):
        uvz = np.stack([site_xy[0] / LATTICE_A * np.ones(len(z_abs_arr)),
                        site_xy[1] / LATTICE_A * np.ones(len(z_abs_arr)),
                        z_abs_arr], axis=1)
        E = cpu_eval_energy(uvz, basis_params, coeff4)[0]
        F = cpu_eval_force(uvz, basis_params, coeff4, lvec2d_flat)[0]
        return E, F

    def _cpu_2d(z_abs, nxy, coeff4):
        us = np.linspace(0.0, 1.0, nxy, endpoint=False)
        vs = np.linspace(0.0, 1.0, nxy, endpoint=False)
        uvz = []
        for v in vs:
            for u in us:
                uvz.append([u, v, z_abs])
        return cpu_eval_energy(np.array(uvz), basis_params, coeff4)[0].reshape(nxy, nxy)

    # --- Combined: z-scans + 2D maps ---
    print("\n--- Combined (pauli+london+coulomb) ---")
    if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
    md._set_folded_coefficients(None); md._tensor_coeffs_set = True

    for site_name, site_xy in SITES.items():
        pos = _z_scan_positions(site_xy, Z_SCAN_ABS)
        E_ref = compute_combined_reference(pos)
        E_cpu_z, F_cpu_z = _cpu_zscan(site_xy, Z_SCAN_ABS, coeff4_natural)
        E_gpu_z, F_gpu_z = _eval_gpu_zscan(md, site_xy, Z_SCAN_ABS)
        plot_zscan(Z_SCAN_REL, E_ref, E_cpu_z, E_gpu_z, F_cpu_z, F_gpu_z, 'combined', site_name, PLOT_DIR)

    for z_rel in [2.5, 4.75, 7.0]:
        z_abs = Z_SURF_TOP + z_rel
        E_cpu_map = _cpu_2d(z_abs, 32, coeff4_natural)
        E_gpu_map = _eval_gpu_2d(md, z_abs, 32)
        plot_2d_map(E_cpu_map, E_gpu_map, z_rel, 'combined', PLOT_DIR)

    # --- Morse only ---
    print("\n--- Morse only ---")
    coeff_sets['coulomb'] = np.zeros_like(orig_coulomb)
    if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
    md._set_folded_coefficients(None); md._tensor_coeffs_set = True

    for site_name, site_xy in SITES.items():
        pos = _z_scan_positions(site_xy, Z_SCAN_ABS)
        E_ref = compute_morse_reference(pos)
        E_cpu_z, F_cpu_z = _cpu_zscan(site_xy, Z_SCAN_ABS, coeff4_morse)
        E_gpu_z, F_gpu_z = _eval_gpu_zscan(md, site_xy, Z_SCAN_ABS)
        plot_zscan(Z_SCAN_REL, E_ref, E_cpu_z, E_gpu_z, F_cpu_z, F_gpu_z, 'Morse', site_name, PLOT_DIR)

    coeff_sets['coulomb'] = orig_coulomb

    # --- Coulomb only ---
    print("\n--- Coulomb only ---")
    coeff_sets['pauli'] = np.zeros_like(orig_pauli); coeff_sets['london'] = np.zeros_like(orig_london)
    if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
    md._set_folded_coefficients(None); md._tensor_coeffs_set = True

    for site_name, site_xy in SITES.items():
        pos = _z_scan_positions(site_xy, Z_SCAN_ABS)
        E_ref = compute_coulomb_reference_ewald(pos)
        E_cpu_z, F_cpu_z = _cpu_zscan(site_xy, Z_SCAN_ABS, coeff4_coul)
        E_gpu_z, F_gpu_z = _eval_gpu_zscan(md, site_xy, Z_SCAN_ABS)
        plot_zscan(Z_SCAN_REL, E_ref, E_cpu_z, E_gpu_z, F_cpu_z, F_gpu_z, 'Coulomb', site_name, PLOT_DIR)

    coeff_sets['pauli'] = orig_pauli; coeff_sets['london'] = orig_london

    print(f"\n  All plots saved to: {PLOT_DIR}")

    # --- Verdict ---
    print("\n" + "=" * 70)
    print("VERDICT (GPU-CPU parity)")
    print("=" * 70)
    all_pass = True
    for label, E_d, E_r in [("combined", E_diff, E_rel), ("morse", E_diff_m, E_rel_m), ("coulomb", E_diff_c, E_rel_c)]:
        ok = E_r.max() < RTOL or E_d.max() < ATOL
        status = "PASS" if ok else "FAIL"
        if not ok: all_pass = False
        print(f"  {label:10s}: {status}  max|rel|={E_r.max():.6e}  max|abs|={E_d.max():.6e}")

    if all_pass:
        print("\n  ALL TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED")

    return all_pass


# =============================================================================
# CPU reference for POLY basis: E = sum_i B_i * (cCoulomb_i + B_i*(cLondon_i + B_i*cPauli_i))
# where B_i = cos(2π*ku*u) * cos(2π*kv*v) * t^n,  t = 1 - min(dz/R, 1)
# =============================================================================

def _poly_basis_matrix(uvz, nu, nv, nz, m_start, zmin, R):
    """Build (N, nbasis) poly basis matrix. nbasis = nu*nv*nz.
    Layout: iu varies slowest, iz fastest (matches kernel coeff layout for poly)."""
    uvz = np.asarray(uvz, dtype=np.float64)
    N = uvz.shape[0]
    u = uvz[:, 0]; v = uvz[:, 1]; z = uvz[:, 2]
    dz = np.maximum(0.0, z - zmin)
    x = np.minimum(dz / R, 1.0)
    t = 1.0 - x
    cols = []
    for iu in range(nu):
        for iv in range(nv):
            for iz in range(nz):
                n = m_start + iz
                bx = np.cos(2.0 * np.pi * iu * u)
                by = np.cos(2.0 * np.pi * iv * v)
                bz = t ** n if n > 0 else np.ones(N)
                cols.append(bx * by * bz)
    return np.array(cols).T  # (N, nbasis)


def cpu_eval_energy_poly(uvz, nu, nv, nz, m_start, zmin, R, coeff4):
    """Evaluate E = B*(c.z + B*(c.y + B*c.x)) with poly basis.
    coeff4: (ntypes, nbasis, 4) = (cPauli, cLondon, cCoulomb, cH).
    Returns: E (ntypes, N)"""
    B = _poly_basis_matrix(uvz, nu, nv, nz, m_start, zmin, R)  # (N, nbasis)
    coeff4 = np.asarray(coeff4, dtype=np.float64)
    cP = coeff4[:, :, 0]; cL = coeff4[:, :, 1]; cC = coeff4[:, :, 2]
    E = cC[:, None, :] * B[None, :, :] + cL[:, None, :] * B[None, :, :]**2 + cP[:, None, :] * B[None, :, :]**3
    return E.sum(axis=2)


def cpu_eval_force_poly(uvz, nu, nv, nz, m_start, zmin, R, coeff4, lvec2d):
    """Evaluate force with poly basis. Returns F: (ntypes, N, 3)."""
    uvz = np.asarray(uvz, dtype=np.float64)
    coeff4 = np.asarray(coeff4, dtype=np.float64)
    ntypes = coeff4.shape[0]; N = uvz.shape[0]
    u = uvz[:, 0]; v = uvz[:, 1]; z = uvz[:, 2]
    dz = np.maximum(0.0, z - zmin)
    x = np.minimum(dz / R, 1.0)
    t = 1.0 - x
    invR = 1.0 / R
    nbasis = nu * nv * nz
    # Build basis and derivatives
    B = np.zeros((N, nbasis)); dBdu = np.zeros((N, nbasis))
    dBdv = np.zeros((N, nbasis)); dBdz = np.zeros((N, nbasis))
    ib = 0
    for iu in range(nu):
        phix = 2.0 * np.pi * iu
        bx = np.cos(phix * u); sx = np.sin(phix * u)
        for iv in range(nv):
            phiy = 2.0 * np.pi * iv
            by = np.cos(phiy * v); sy = np.sin(phiy * v)
            for iz in range(nz):
                n = m_start + iz
                if n > 0:
                    tpow = t ** n
                    dtpow = -n * invR * t ** max(n - 1, 0) if n > 0 else np.zeros(N)
                    dtpow[x >= 1.0] = 0.0
                else:
                    tpow = np.ones(N); dtpow = np.zeros(N)
                B[:, ib] = bx * by * tpow
                dBdu[:, ib] = -phix * sx * by * tpow
                dBdv[:, ib] = bx * (-phiy * sy) * tpow
                dBdz[:, ib] = bx * by * dtpow
                ib += 1
    cP = coeff4[:, :, 0]; cL = coeff4[:, :, 1]; cC = coeff4[:, :, 2]
    dEdB = cC[:, None, :] + B[None, :, :] * (2.0 * cL[:, None, :] + B[None, :, :] * 3.0 * cP[:, None, :])
    dEdu = (dEdB * dBdu[None, :, :]).sum(axis=2)
    dEdv = (dEdB * dBdv[None, :, :]).sum(axis=2)
    dEdz = (dEdB * dBdz[None, :, :]).sum(axis=2)
    ax, bx_l, ay, by_l = lvec2d
    det = ax * by_l - bx_l * ay
    inv = np.array([by_l / det, -bx_l / det, -ay / det, ax / det])
    Fx = -(dEdu * inv[0] + dEdv * inv[2])
    Fy = -(dEdu * inv[1] + dEdv * inv[3])
    Fz = -dEdz
    return np.stack([Fx, Fy, Fz], axis=-1)


def _z_basis_values_poly(z_rel, m_start, nz, R):
    """Compute poly z-basis functions t^n for plotting."""
    z0 = FIT_Z_RANGE[0]
    dz = np.maximum(0.0, z_rel - z0)
    x = np.minimum(dz / R, 1.0)
    t = 1.0 - x
    return np.stack([t ** (m_start + iz) if (m_start + iz) > 0 else np.ones(len(z_rel)) for iz in range(nz)], axis=1)


def plot_zscan_poly(z_rel, E_ref, E_cpu, E_gpu, F_cpu, F_gpu, label, site, save_dir, m_start, nz, R):
    """Plot E(z) + Fz(z) + poly z-basis functions."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 2]})
    # --- Top: Energy ---
    ax = axes[0]
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.plot(z_rel, E_ref, 'k:', label='Reference (brute-force)', linewidth=2)
    ax.plot(z_rel, E_cpu, 'b-', label='CPU (poly fit)', linewidth=1)
    ax.plot(z_rel, E_gpu, 'r--', label='GPU (tensor poly)', linewidth=1)
    ax.axvspan(FIT_Z_RANGE[0], FIT_Z_RANGE[1], alpha=0.1, color='orange', label='Fit region')
    ax.set_ylabel('Energy [eV]')
    fit_mask = (z_rel >= FIT_Z_RANGE[0]) & (z_rel <= FIT_Z_RANGE[1])
    if fit_mask.any() and np.any(E_ref[fit_mask] != 0):
        _set_ylim_symmetric(ax, E_ref, z_rel=z_rel, fit_range=FIT_Z_RANGE)
    ax.set_title(f'{label} [poly] — E(z) above {site}')
    ax.legend(loc='best', fontsize=8)
    # --- Middle: Force Fz ---
    ax = axes[1]
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.plot(z_rel, F_cpu[:, 2], 'b-', label='CPU Fz', linewidth=1)
    ax.plot(z_rel, F_gpu[:, 2], 'r--', label='GPU Fz', linewidth=1)
    ax.axvspan(FIT_Z_RANGE[0], FIT_Z_RANGE[1], alpha=0.1, color='orange')
    ax.set_ylabel('Fz [eV/Å]')
    ax.set_title(f'{label} [poly] — Fz(z) above {site}')
    ax.legend(fontsize=8)
    # --- Bottom: Z-basis functions ---
    ax = axes[2]
    bz_vals = _z_basis_values_poly(z_rel, m_start, nz, R)
    cmap = plt.cm.plasma
    for i in range(nz):
        n = m_start + i
        ax.plot(z_rel, bz_vals[:, i], color=cmap(i / max(nz-1, 1)),
                label=f'n={n} (α_eff={n/R:.2f})', linewidth=1.5)
    ax.axvspan(FIT_Z_RANGE[0], FIT_Z_RANGE[1], alpha=0.1, color='orange')
    ax.set_xlabel('z above surface [Å]'); ax.set_ylabel('Basis amplitude')
    ax.set_title(f'Z-basis functions (poly, R={R} Å)')
    ax.legend(fontsize=8, ncol=nz)
    fig.tight_layout()
    fname = f'zscan_poly_{_safe_name(label)}_{site}.png'
    fig.savefig(os.path.join(save_dir, fname), dpi=120)
    plt.close(fig)
    print(f'  Saved {fname}')


def run_poly_test():
    """Test poly tensor kernel with sequential powers."""
    print("\n" + "=" * 70)
    print("POLY Tensor Kernel Parity Test")
    print("=" * 70)
    print(f"  POLY_R = {POLY_R} Å")
    print(f"  Morse m_start={MORSE_M_START}, powers=[{MORSE_M_START}..{MORSE_M_START+NZ-1}]")
    print(f"  Coulomb m_start={COULOMB_M_START}, powers=[{COULOMB_M_START}..{COULOMB_M_START+NZ-1}]")

    # --- Setup MD with surface ---
    md = MolecularDynamics(nloc=32, debug_build_options='-DDBG_UFF=0')
    md.init_rigid_molecule_batch(np.zeros((1, 3), dtype=np.float32), PROBE_REQ, nSystems=256)
    md.set_surface(NACL_SUBSTRATE, nPBC=MORSE_NPBC, alpha_morse=MORSE_ALPHA, bMacro=True)

    # --- Generate fitting grid (same as scan test) ---
    us = np.linspace(0.0, 1.0, FIT_NXY, endpoint=False, dtype=np.float32)
    vs = np.linspace(0.0, 1.0, FIT_NXY, endpoint=False, dtype=np.float32)
    zs_abs = np.linspace(Z0_ABS, Z1_ABS, FIT_NZ_SAMP, dtype=np.float32)
    grid_xyz = []; grid_uvz = []
    for z in zs_abs:
        for v in vs:
            for u in us:
                grid_xyz.append([u * LATTICE_A, v * LATTICE_A, z])
                grid_uvz.append([u, v, z])
    grid_xyz = np.array(grid_xyz, dtype=np.float32)
    grid_uvz = np.array(grid_uvz, dtype=np.float64)
    print(f"  Fit grid: {FIT_NXY}x{FIT_NXY}x{FIT_NZ_SAMP} = {len(grid_xyz)} points")

    # --- Compute references on grid ---
    print("  Computing Morse reference...")
    E_morse_grid = compute_morse_reference(grid_xyz)
    print("  Computing Coulomb reference...")
    E_coul_grid = compute_coulomb_reference_ewald(grid_xyz)
    E_combined_grid = E_morse_grid + E_coul_grid

    # --- Fit each component with poly basis (powered for tensor cubic formula) ---
    # E = cCoulomb*B + cLondon*B^2 + cPauli*B^3
    # Fit pauli with B^3, london with B^2, coulomb with B^1
    def _fit_poly_powered(E_ref, power, m_start, fit_mask=None):
        """Fit coefficients for B^power using poly basis with sequential powers."""
        Phi = _poly_basis_matrix(grid_uvz, NU, NV, NZ, m_start, Z0_ABS, POLY_R)  # (N, nbasis)
        Phi_p = Phi ** power  # powered basis
        if fit_mask is not None:
            w = fit_mask.astype(np.float64)
        else:
            w = np.ones(len(E_ref), dtype=np.float64)
        Phiw = Phi_p * w[:, None]
        yw = E_ref * w
        S, *_ = np.linalg.lstsq(Phiw, yw, rcond=None)
        return S.astype(np.float32)

    # Morse mask: exclude high-repulsive region (same as scan test)
    E_min = E_morse_grid.min()
    morse_mask = E_morse_grid < -E_min
    print(f"  Morse mask: E_min={E_min:.6f}, kept={morse_mask.sum()}/{len(morse_mask)}")

    # Fit with MORSE_M_START for Morse components, COULOMB_M_START for Coulomb
    # But the kernel uses a SINGLE m_start for all components!
    # So we must use the same m_start for all. Use a compromise.
    # Let's test two configurations: Morse-optimized and Coulomb-optimized

    results = {}
    for config_name, m_start in [('morse_opt', MORSE_M_START), ('coulomb_opt', COULOMB_M_START)]:
        print(f"\n--- Poly fit: {config_name} (m_start={m_start}) ---")
        c_pauli = _fit_poly_powered(E_morse_grid, power=3, m_start=m_start, fit_mask=morse_mask)
        c_london = _fit_poly_powered(E_morse_grid, power=2, m_start=m_start, fit_mask=morse_mask)
        c_coulomb = _fit_poly_powered(E_coul_grid, power=1, m_start=m_start)
        # RMSE check
        Phi = _poly_basis_matrix(grid_uvz, NU, NV, NZ, m_start, Z0_ABS, POLY_R)
        E_morse_fit = (Phi**3 @ c_pauli + Phi**2 @ c_london)
        E_coul_fit = (Phi @ c_coulomb)
        rmse_m = np.sqrt(np.mean((E_morse_fit - E_morse_grid)[morse_mask]**2)) if morse_mask.any() else float('nan')
        rmse_c = np.sqrt(np.mean((E_coul_fit - E_coul_grid)**2))
        print(f"  Morse RMSE (masked): {rmse_m:.6f} eV")
        print(f"  Coulomb RMSE: {rmse_c:.6f} eV")
        results[config_name] = {
            'm_start': m_start, 'c_pauli': c_pauli, 'c_london': c_london, 'c_coulomb': c_coulomb,
            'rmse_morse': rmse_m, 'rmse_coulomb': rmse_c,
        }

    # --- For each config, set up GPU and test parity ---
    ntypes = 1; nbasis = NU * NV * NZ
    lvec2d_flat = np.array([LATTICE_A, 0.0, 0.0, LATTICE_A], dtype=np.float32)

    for config_name, res in results.items():
        m_start = res['m_start']
        print(f"\n{'=' * 70}")
        print(f"POLY PARITY: {config_name} (m_start={m_start})")
        print(f"{'=' * 70}")

        # Pack coeff4 for CPU reference
        coeff4 = np.zeros((ntypes, nbasis, 4), dtype=np.float64)
        coeff4[:, :, 0] = res['c_pauli'][None, :]
        coeff4[:, :, 1] = res['c_london'][None, :]
        coeff4[:, :, 2] = res['c_coulomb'][None, :]

        # Set up GPU with poly kernel
        # Poly kernel ignores folded_kxyz (uses scalar zmin/zcut/m_start),
        # but _set_folded_coefficients still uploads it, so provide dummy.
        dummy_bp = np.zeros((NU * NV * NZ, 4), dtype=np.float32)
        for iu in range(NU):
            for iv in range(NV):
                for iz in range(NZ):
                    idx = (iu * NV + iv) * NZ + iz
                    dummy_bp[idx, 0] = iu; dummy_bp[idx, 1] = iv
                    dummy_bp[idx, 2] = 1.0; dummy_bp[idx, 3] = Z0_ABS
        md.folded_params = {
            'basis_params': dummy_bp,
            'coeffs': np.zeros((1, nbasis), dtype=np.float32),  # dummy for _set_folded_coefficients
            'coeff_sets': {'pauli': res['c_pauli'][None, :], 'london': res['c_london'][None, :],
                           'coulomb': res['c_coulomb'][None, :]},
            'atom_type_ids': np.array([0], dtype=np.int32),
            'unique_REQs': PROBE_REQ,
            'basis_lvec2d': np.array([[LATTICE_A, 0, 0, 0], [0, LATTICE_A, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
            'nu': NU, 'nv': NV, 'nz': NZ,
            'z_range': (Z0_ABS, Z1_ABS),
            'basis_type': 1,  # poly
            'poly_R': POLY_R,
            'm_start': m_start,
        }
        md.folded_lvec_basis = md.folded_params['basis_lvec2d']
        md.set_folded_kernel_kind('tensor')
        md.kernel_params['folded_lvec2d'] = np.array([LATTICE_A, 0.0, 0.0, LATTICE_A], dtype=np.float32)
        if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
        md._set_folded_coefficients(None); md._tensor_coeffs_set = True

        # --- Test grid ---
        us_t = np.linspace(0.0, 1.0, 16, endpoint=False, dtype=np.float64)
        vs_t = np.linspace(0.0, 1.0, 16, endpoint=False, dtype=np.float64)
        zs_t = np.linspace(Z0_ABS + 0.5, Z1_ABS - 0.5, 20, dtype=np.float64)
        xyz_t = []; uvz_t = []
        for z in zs_t:
            for v in vs_t:
                for u in us_t:
                    xyz_t.append([u * LATTICE_A, v * LATTICE_A, z])
                    uvz_t.append([u, v, z])
        xyz_t = np.array(xyz_t, dtype=np.float32); uvz_t = np.array(uvz_t, dtype=np.float64)
        N = len(xyz_t)

        # CPU reference
        E_cpu = cpu_eval_energy_poly(uvz_t, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4)[0]
        F_cpu = cpu_eval_force_poly(uvz_t, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4, lvec2d_flat)[0]

        # GPU evaluation
        transforms = np.zeros((N, 3, 4), dtype=np.float32)
        transforms[:, 0, 0] = 1.0; transforms[:, 1, 1] = 1.0; transforms[:, 2, 2] = 1.0
        transforms[:, :, 3] = xyz_t
        out = md.eval_rigid_getSurfFolded(transforms.reshape(-1, 12), chunk_size=256)
        E_gpu = out['total']

        # GPU forces (first 256)
        import pyopencl as cl
        nch = min(256, N)
        sys_bytes = md.nvecs * 4 * np.float32().itemsize
        md.upload_rigid_transforms(transforms[:nch].reshape(-1, 3, 4), iSys0=0)
        cl.enqueue_fill_buffer(md.queue, md.buffer_dict['aforce'], np.zeros(1, dtype=np.float32), 0, nch * sys_bytes)
        md.queue.finish(); md.run_getSurfFolded(nSystems=nch); md.queue.finish()
        aforce = np.empty((nch, md.nvecs, 4), dtype=np.float32); md.fromGPU('aforce', aforce); md.queue.finish()
        F_gpu = aforce[:, :md.natoms, :3].reshape(nch, 3)

        E_diff = np.abs(E_gpu - E_cpu); E_rel = E_diff / (np.abs(E_cpu) + 1e-10)
        F_diff = np.linalg.norm(F_gpu - F_cpu[:nch], axis=-1)
        F_rel = F_diff / (np.linalg.norm(F_cpu[:nch], axis=-1) + 1e-10)
        print(f"  Combined:  E max|diff|={E_diff.max():.6e}, max|rel|={E_rel.max():.6e}")
        print(f"             F max|diff|={F_diff.max():.6e}, max|rel|={F_rel.max():.6e}")

        # Morse-only and Coulomb-only
        coeff4_m = coeff4.copy(); coeff4_m[:, :, 2] = 0.0
        E_cpu_m = cpu_eval_energy_poly(uvz_t, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4_m)[0]
        md.folded_params['coeff_sets']['coulomb'] = np.zeros_like(res['c_coulomb'][None, :])
        if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
        md._set_folded_coefficients(None); md._tensor_coeffs_set = True
        E_gpu_m = md.eval_rigid_getSurfFolded(transforms.reshape(-1, 12), chunk_size=256)['total']
        E_diff_m = np.abs(E_gpu_m - E_cpu_m); E_rel_m = E_diff_m / (np.abs(E_cpu_m) + 1e-10)
        print(f"  Morse:     E max|diff|={E_diff_m.max():.6e}, max|rel|={E_rel_m.max():.6e}")
        md.folded_params['coeff_sets']['coulomb'] = res['c_coulomb'][None, :]

        coeff4_c = coeff4.copy(); coeff4_c[:, :, 0] = 0.0; coeff4_c[:, :, 1] = 0.0
        E_cpu_c = cpu_eval_energy_poly(uvz_t, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4_c)[0]
        md.folded_params['coeff_sets']['pauli'] = np.zeros_like(res['c_pauli'][None, :])
        md.folded_params['coeff_sets']['london'] = np.zeros_like(res['c_london'][None, :])
        if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
        md._set_folded_coefficients(None); md._tensor_coeffs_set = True
        E_gpu_c = md.eval_rigid_getSurfFolded(transforms.reshape(-1, 12), chunk_size=256)['total']
        E_diff_c = np.abs(E_gpu_c - E_cpu_c); E_rel_c = E_diff_c / (np.abs(E_cpu_c) + 1e-10)
        print(f"  Coulomb:   E max|diff|={E_diff_c.max():.6e}, max|rel|={E_rel_c.max():.6e}")
        md.folded_params['coeff_sets']['pauli'] = res['c_pauli'][None, :]
        md.folded_params['coeff_sets']['london'] = res['c_london'][None, :]

        # --- Plots ---
        os.makedirs(PLOT_DIR, exist_ok=True)
        # Restore full coeffs for combined plots
        if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
        md._set_folded_coefficients(None); md._tensor_coeffs_set = True

        for site_name, site_xy in SITES.items():
            pos = _z_scan_positions(site_xy, Z_SCAN_ABS)
            E_ref = compute_combined_reference(pos)
            uvz_scan = np.stack([site_xy[0] / LATTICE_A * np.ones(len(Z_SCAN_ABS)),
                                 site_xy[1] / LATTICE_A * np.ones(len(Z_SCAN_ABS)),
                                 Z_SCAN_ABS], axis=1)
            E_cpu_z = cpu_eval_energy_poly(uvz_scan, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4)[0]
            F_cpu_z = cpu_eval_force_poly(uvz_scan, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4, lvec2d_flat)[0]
            E_gpu_z, F_gpu_z = _eval_gpu_zscan(md, site_xy, Z_SCAN_ABS)
            plot_zscan_poly(Z_SCAN_REL, E_ref, E_cpu_z, E_gpu_z, F_cpu_z, F_gpu_z,
                            f'combined_{config_name}', site_name, PLOT_DIR, m_start, NZ, POLY_R)

        # Morse-only plots
        md.folded_params['coeff_sets']['coulomb'] = np.zeros_like(res['c_coulomb'][None, :])
        if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
        md._set_folded_coefficients(None); md._tensor_coeffs_set = True
        for site_name, site_xy in SITES.items():
            pos = _z_scan_positions(site_xy, Z_SCAN_ABS)
            E_ref = compute_morse_reference(pos)
            uvz_scan = np.stack([site_xy[0] / LATTICE_A * np.ones(len(Z_SCAN_ABS)),
                                 site_xy[1] / LATTICE_A * np.ones(len(Z_SCAN_ABS)),
                                 Z_SCAN_ABS], axis=1)
            E_cpu_z = cpu_eval_energy_poly(uvz_scan, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4_m)[0]
            F_cpu_z = cpu_eval_force_poly(uvz_scan, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4_m, lvec2d_flat)[0]
            E_gpu_z, F_gpu_z = _eval_gpu_zscan(md, site_xy, Z_SCAN_ABS)
            plot_zscan_poly(Z_SCAN_REL, E_ref, E_cpu_z, E_gpu_z, F_cpu_z, F_gpu_z,
                            f'Morse_{config_name}', site_name, PLOT_DIR, m_start, NZ, POLY_R)
        md.folded_params['coeff_sets']['coulomb'] = res['c_coulomb'][None, :]

        # Coulomb-only plots
        md.folded_params['coeff_sets']['pauli'] = np.zeros_like(res['c_pauli'][None, :])
        md.folded_params['coeff_sets']['london'] = np.zeros_like(res['c_london'][None, :])
        if hasattr(md, '_tensor_coeffs_set'): del md._tensor_coeffs_set
        md._set_folded_coefficients(None); md._tensor_coeffs_set = True
        for site_name, site_xy in SITES.items():
            pos = _z_scan_positions(site_xy, Z_SCAN_ABS)
            E_ref = compute_coulomb_reference_ewald(pos)
            uvz_scan = np.stack([site_xy[0] / LATTICE_A * np.ones(len(Z_SCAN_ABS)),
                                 site_xy[1] / LATTICE_A * np.ones(len(Z_SCAN_ABS)),
                                 Z_SCAN_ABS], axis=1)
            E_cpu_z = cpu_eval_energy_poly(uvz_scan, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4_c)[0]
            F_cpu_z = cpu_eval_force_poly(uvz_scan, NU, NV, NZ, m_start, Z0_ABS, POLY_R, coeff4_c, lvec2d_flat)[0]
            E_gpu_z, F_gpu_z = _eval_gpu_zscan(md, site_xy, Z_SCAN_ABS)
            plot_zscan_poly(Z_SCAN_REL, E_ref, E_cpu_z, E_gpu_z, F_cpu_z, F_gpu_z,
                            f'Coulomb_{config_name}', site_name, PLOT_DIR, m_start, NZ, POLY_R)
        md.folded_params['coeff_sets']['pauli'] = res['c_pauli'][None, :]
        md.folded_params['coeff_sets']['london'] = res['c_london'][None, :]

    print(f"\n  All poly plots saved to: {PLOT_DIR}")


if __name__ == '__main__':
    ok = run_test()
    run_poly_test()
    sys.exit(0 if ok else 1)
