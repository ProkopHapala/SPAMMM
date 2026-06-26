#!/usr/bin/env python3
"""Fit Pauli parameters A and beta for E_pauli = A * overlap^beta.

For each molecule/method/target, we have:
  - overlap(z): raw Pauli overlap from FFT convolution (A=1, beta=1)
  - Ez_ref(z): reference energy from DFTB+ z-scan

We fit A and beta so that A * overlap(z)^beta ≈ Ez_ref(z)
in the z-range [z_min, z_max] (default 1.7–2.3 Å above atom).

Two fitting approaches:
  1. Log-log linear regression: log(Ez) = log(A) + beta * log(overlap)
     → fast, robust, but biased toward high-overlap points
  2. Nonlinear least squares (scipy.optimize.curve_fit): minimize sum (A*overlap^beta - Ez)^2
     → more accurate in linear scale

We fit per-molecule, per-method, and also globally (all curves pooled per method).

Usage:
  python tests/SPM/fit_pauli_params.py
  python tests/SPM/fit_pauli_params.py --z_min 1.7 --z_max 2.3
"""
import os, sys, argparse, json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.realpath(os.path.join(_THIS_DIR, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.SPM.run_zscan_reference import MOLECULES, METHODS, load_molecule
from spammm.SPM import AFM as afm

DENS_DIR = os.path.join(_ROOT, 'debug', 'densities')
EZ_DIR = os.path.join(_ROOT, 'tests', 'ref_data', 'Ez_FDBM')
OUT_DIR = os.path.join(_ROOT, 'debug', 'pauli_fit')


def load_density(mol_name, method_name):
    npy_path = os.path.join(DENS_DIR, f'rho_{mol_name}_{method_name}.npy')
    meta_path = os.path.join(DENS_DIR, f'rho_{mol_name}_{method_name}.meta.npz')
    if not os.path.exists(npy_path) or not os.path.exists(meta_path):
        return None
    rho = np.load(npy_path)
    meta = np.load(meta_path)
    return rho, meta['origin'], float(meta['step']), meta['atom_pos'], list(meta['atom_names'])


def atom_to_grid_idx(atom_pos, origin, step):
    ix = int(round((atom_pos[0] - origin[0]) / step))
    iy = int(round((atom_pos[1] - origin[1]) / step))
    return ix, iy


def extract_z_line(field, origin, step, atom_pos):
    ix, iy = atom_to_grid_idx(atom_pos, origin, step)
    nx, ny, nz = field.shape
    ix = max(0, min(ix, nx - 1))
    iy = max(0, min(iy, ny - 1))
    z_values = origin[2] + np.arange(nz) * step
    return z_values, field[ix, iy, :].copy()


def load_ez_reference(mol_name, method_name, site_label, atom_idx):
    fname = f'zscan_{mol_name}_{method_name}_{site_label}{atom_idx}.dat'
    path = os.path.join(EZ_DIR, fname)
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, skiprows=4)
    return data[:, 0], data[:, 1]


def fit_loglog(z, overlap, ez, z_min, z_max):
    """Log-log linear regression: log(Ez) = log(A) + beta * log(overlap)."""
    mask = (z >= z_min) & (z <= z_max) & (ez > 1e-10) & (overlap > 1e-30)
    if mask.sum() < 3:
        return None
    log_ov = np.log(overlap[mask])
    log_ez = np.log(ez[mask])
    # Linear fit: log_ez = log_A + beta * log_ov
    A_mat = np.vstack([np.ones_like(log_ov), log_ov]).T
    coeffs, residuals, rank, sv = np.linalg.lstsq(A_mat, log_ez, rcond=None)
    log_A, beta = coeffs
    A = np.exp(log_A)
    # R^2
    pred = A_mat @ coeffs
    ss_res = np.sum((log_ez - pred) ** 2)
    ss_tot = np.sum((log_ez - log_ez.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'A': A, 'beta': beta, 'r2_log': r2, 'n_points': int(mask.sum())}


def fit_nonlinear(z, overlap, ez, z_min, z_max):
    """Nonlinear least squares: Ez = A * overlap^beta."""
    mask = (z >= z_min) & (z <= z_max) & (ez > 1e-10) & (overlap > 1e-30)
    if mask.sum() < 3:
        return None
    ov = overlap[mask]
    ez = ez[mask]

    def model(x, A, beta):
        return A * x ** beta

    try:
        popt, pcov = curve_fit(model, ov, ez, p0=[500.0, 1.1], maxfev=10000)
        A, beta = popt
        pred = model(ov, A, beta)
        ss_res = np.sum((ez - pred) ** 2)
        ss_tot = np.sum((ez - ez.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {'A': A, 'beta': beta, 'r2_lin': r2, 'n_points': int(mask.sum())}
    except Exception as e:
        return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Fit Pauli A and beta parameters')
    parser.add_argument('--z_min', type=float, default=1.7, help='Min z for fitting [Ang]')
    parser.add_argument('--z_max', type=float, default=2.3, help='Max z for fitting [Ang]')
    parser.add_argument('--sigma_tip', type=float, default=0.7, help='Gaussian tip sigma [Ang]')
    parser.add_argument('--outdir', default=OUT_DIR)
    args = parser.parse_args()

    outdir = os.path.join(_ROOT, args.outdir) if not os.path.isabs(args.outdir) else args.outdir
    os.makedirs(outdir, exist_ok=True)

    mol_names = list(MOLECULES.keys())
    method_names = list(METHODS.keys())

    # Collect all (overlap, ez) pairs per method for global fitting
    pooled = {m: {'overlap': [], 'ez': []} for m in method_names}
    results = []

    for mol_name in mol_names:
        if mol_name not in MOLECULES:
            continue
        mol_info = MOLECULES[mol_name]
        mol_methods = mol_info.get('methods', list(METHODS.keys()))
        atom_pos_mol, _ = load_molecule(mol_info['xyz'])

        for method_name in mol_methods:
            result = load_density(mol_name, method_name)
            if result is None:
                continue
            rho_mol, origin_mol, step_mol, _, _ = result
            nx, ny, nz = rho_mol.shape

            rho_tip = afm.build_gaussian_tip((nx, ny, nz), step_mol, args.sigma_tip)
            overlap_raw = afm.compute_pauli_overlap(rho_mol, rho_tip, step_mol, tip_rolled=True)

            for target_label, atom_idx in mol_info['targets']:
                target_pos = atom_pos_mol[atom_idx]
                z_abs, overlap_line = extract_z_line(overlap_raw, origin_mol, step_mol, target_pos)
                z_rel = z_abs - target_pos[2]

                ez_ref = load_ez_reference(mol_name, method_name, target_label, atom_idx)
                if ez_ref is None:
                    continue

                # Interpolate Ez reference onto our z_rel grid
                ez_interp = np.interp(z_rel, ez_ref[0], ez_ref[1], left=0, right=0)

                # Fit in the requested z-range
                fit_ll = fit_loglog(z_rel, overlap_line, ez_interp, args.z_min, args.z_max)
                fit_nl = fit_nonlinear(z_rel, overlap_line, ez_interp, args.z_min, args.z_max)

                entry = {
                    'mol': mol_name, 'method': method_name, 'site': target_label,
                    'atom_idx': atom_idx, 'z_rel': z_rel, 'overlap': overlap_line,
                    'ez_interp': ez_interp, 'loglog': fit_ll, 'nonlinear': fit_nl,
                }
                results.append(entry)

                # Pool for global fit
                mask = (z_rel >= args.z_min) & (z_rel <= args.z_max) & (ez_interp > 1e-10) & (overlap_line > 1e-30)
                if mask.any():
                    pooled[method_name]['overlap'].append(overlap_line[mask])
                    pooled[method_name]['ez'].append(ez_interp[mask])

                A_ll = fit_ll['A'] if fit_ll else float('nan')
                b_ll = fit_ll['beta'] if fit_ll else float('nan')
                r2_ll = fit_ll['r2_log'] if fit_ll else float('nan')
                A_nl = fit_nl['A'] if fit_nl and 'A' in fit_nl else float('nan')
                b_nl = fit_nl['beta'] if fit_nl and 'beta' in fit_nl else float('nan')
                print(f"  {mol_name:10s} {method_name:12s} {target_label:8s}({atom_idx:2d})  "
                      f"loglog: A={A_ll:10.3f} β={b_ll:.4f} R²={r2_ll:.4f}  "
                      f"nonlin: A={A_nl:10.3f} β={b_nl:.4f}")

    # Global fit per method (all curves pooled)
    print(f"\n{'='*80}")
    print(f"Global fit per method (z=[{args.z_min}, {args.z_max}] Å, all curves pooled)")
    print(f"{'='*80}")
    global_fits = {}
    for method_name in method_names:
        ov_all = np.concatenate(pooled[method_name]['overlap']) if pooled[method_name]['overlap'] else np.array([])
        ez_all = np.concatenate(pooled[method_name]['ez']) if pooled[method_name]['ez'] else np.array([])
        if len(ov_all) < 3:
            print(f"  {method_name}: not enough data")
            continue
        # Already filtered by z-range in the pooling step; use wide z range
        z_all = np.concatenate([np.full(len(o), 0.0) for o in pooled[method_name]['overlap']])
        fit_ll = fit_loglog(z_all, ov_all, ez_all, -1e9, 1e9)
        fit_nl = fit_nonlinear(z_all, ov_all, ez_all, -1e9, 1e9)
        global_fits[method_name] = {'loglog': fit_ll, 'nonlinear': fit_nl}
        A_ll = fit_ll['A'] if fit_ll else float('nan')
        b_ll = fit_ll['beta'] if fit_ll else float('nan')
        A_nl = fit_nl['A'] if fit_nl and 'A' in fit_nl else float('nan')
        b_nl = fit_nl['beta'] if fit_nl and 'beta' in fit_nl else float('nan')
        print(f"  {method_name:12s}  loglog: A={A_ll:10.3f} β={b_ll:.4f} R²={fit_ll['r2_log']:.4f}  "
              f"nonlin: A={A_nl:10.3f} β={b_nl:.4f}")

    # Save results as JSON
    json_results = {
        'z_range': [args.z_min, args.z_max],
        'sigma_tip': args.sigma_tip,
        'per_curve': [],
        'global': {},
    }
    for r in results:
        entry = {
            'mol': r['mol'], 'method': r['method'], 'site': r['site'], 'atom_idx': r['atom_idx'],
            'loglog': r['loglog'], 'nonlinear': r['nonlinear'],
        }
        json_results['per_curve'].append(entry)
    for m, f in global_fits.items():
        json_results['global'][m] = f
    json_path = os.path.join(outdir, 'pauli_fit_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # === Plot 1: Per-method overlay with global fit ===
    for method_name in method_names:
        curves = [r for r in results if r['method'] == method_name]
        if not curves:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, fit_key, title in [(axes[0], 'loglog', 'Log-log fit'), (axes[1], 'nonlinear', 'Nonlinear fit')]:
            for r in curves:
                label = f"{r['mol']}/{r['site']}({r['atom_idx']})"
                z = r['z_rel']
                ov = r['overlap']
                ez = r['ez_interp']
                mask = (z >= args.z_min - 0.5) & (z <= args.z_max + 0.5) & (ez > 1e-10) & (ov > 1e-30)
                ax.semilogy(z[mask], ez[mask], 'r.', markersize=2, alpha=0.5)
                fit = r[fit_key]
                if fit and 'A' in fit:
                    ax.semilogy(z[mask], fit['A'] * ov[mask] ** fit['beta'], 'b-', alpha=0.3, linewidth=0.5)
            # Global fit line
            gf = global_fits.get(method_name, {})
            gfit = gf.get(fit_key)
            if gfit and 'A' in gfit:
                ov_range = np.logspace(-5, 1, 100)
                ax.semilogy([], [], 'b-', label=f"Global fit: A={gfit['A']:.1f} β={gfit['beta']:.4f}")
                ax.semilogy([], [], 'r.', label='Ez reference')
            ax.set_xlabel('z (Å)')
            ax.set_ylabel('E (eV, log)')
            ax.set_title(f'{method_name} — {title}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'Pauli fit — {method_name} (z=[{args.z_min},{args.z_max}] Å)', fontsize=13)
        fig.tight_layout()
        path = os.path.join(outdir, f'fit_{method_name}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    # === Plot 2: 2D scatter of fitted A vs beta, colored by method, labeled by system ===
    method_colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(METHODS.keys())}
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for r in results:
        fit = r['nonlinear']
        if not fit or 'A' not in fit:
            continue
        A = fit['A']; beta = fit['beta']
        color = method_colors.get(r['method'], 'gray')
        label = f"{r['mol']}/{r['site']}({r['atom_idx']})"
        ax.scatter(beta, A, c=[color], s=80, edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(label, (beta, A), fontsize=5, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points', alpha=0.8)
    # Legend for methods
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=method_colors[m],
                      markersize=8, label=m) for m in METHODS.keys()]
    ax.legend(handles=handles, fontsize=9, loc='upper left')
    ax.set_xlabel('beta', fontsize=12)
    ax.set_ylabel('A (eV)', fontsize=12)
    ax.set_title(f'Fitted Pauli parameters (nonlinear, z=[{args.z_min},{args.z_max}] Å)', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'fit_params_scatter.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # === Plot 3: Best fit comparison curves (linear + log, for a few representative curves) ===
    for method_name in method_names:
        curves = [r for r in results if r['method'] == method_name]
        if not curves:
            continue
        gf = global_fits.get(method_name, {})
        gfit = gf.get('nonlinear')
        if not gfit or 'A' not in gfit:
            continue
        A_g, beta_g = gfit['A'], gfit['beta']
        n = min(len(curves), 8)
        # Linear scale
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()
        for i in range(n):
            r = curves[i]
            ax = axes[i]
            z = r['z_rel']
            mask = (z >= 1.0) & (z <= 6.0)
            ax.plot(z[mask], r['ez_interp'][mask], 'r-', label='Ez ref', linewidth=1.5)
            ov = r['overlap']
            ax.plot(z[mask], A_g * ov[mask] ** beta_g, 'b--', label=f'Fit A={A_g:.0f} β={beta_g:.3f}', linewidth=1.5)
            ax.axvspan(args.z_min, args.z_max, alpha=0.1, color='green', label='fit range')
            ax.set_xlabel('z (Å)')
            ax.set_ylabel('E (eV)')
            ax.set_title(f"{r['mol']}/{r['site']}({r['atom_idx']})", fontsize=9)
            ax.set_xlim([1.0, 6.0])
            ax.set_ylim([-0.5, 5.0])
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'Pauli fit comparison (linear) — {method_name} (global A={A_g:.1f}, β={beta_g:.4f})', fontsize=13)
        fig.tight_layout()
        path = os.path.join(outdir, f'fit_comparison_{method_name}_linear.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")
        # Log scale
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()
        for i in range(n):
            r = curves[i]
            ax = axes[i]
            z = r['z_rel']
            mask = (z >= 1.0) & (z <= 6.0)
            ez = r['ez_interp'][mask]
            ov = r['overlap'][mask]
            pos_ez = ez > 1e-10
            pos_fit = ov > 1e-30
            if pos_ez.any():
                ax.semilogy(z[mask][pos_ez], ez[pos_ez], 'r-', label='Ez ref', linewidth=1.5)
            if pos_fit.any():
                fit_vals = np.clip(A_g * ov[pos_fit] ** beta_g, 1e-30, None)
                ax.semilogy(z[mask][pos_fit], fit_vals, 'b--', label=f'Fit A={A_g:.0f} β={beta_g:.3f}', linewidth=1.5)
            ax.axvspan(args.z_min, args.z_max, alpha=0.1, color='green')
            ax.set_xlabel('z (Å)')
            ax.set_ylabel('E (eV, log)')
            ax.set_title(f"{r['mol']}/{r['site']}({r['atom_idx']})", fontsize=9)
            ax.set_xlim([1.0, 6.0])
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'Pauli fit comparison (log) — {method_name} (global A={A_g:.1f}, β={beta_g:.4f})', fontsize=13)
        fig.tight_layout()
        path = os.path.join(outdir, f'fit_comparison_{method_name}_log.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    print(f"\nAll outputs in {outdir}/")


if __name__ == '__main__':
    main()
