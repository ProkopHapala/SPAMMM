#!/usr/bin/env python3
"""Plot all E(z) reference curves from tests/ref_data/.

Generates:
  1. Per-molecule overlay: all methods × all sites on one plot
  2. Per-method overlay: all molecules × all sites on one plot
  3. Log-scale plot for repulsive region

Usage:
  python tests/SPM/testplot_zscan_reference.py [--outdir debug/zscan_plots]
  python tests/SPM/testplot_zscan_reference.py --review-external
"""
import os, sys, glob, json, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.realpath(os.path.join(_THIS_DIR, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
REF_DIR = os.path.join(_ROOT, 'tests', 'ref_data', 'Ez_FDBM')
DATA_DIR = os.path.join(_ROOT, 'data', 'xyz')

from tests.SPM.run_zscan_reference import MOLECULES, METHODS, load_molecule

METHOD_STYLES = {
    'dftb_mio':    {'color': 'tab:red',   'ls': '-',  'label': 'DFTB mio-1-1'},
    'dftb_3ob':    {'color': 'tab:orange','ls': '-',  'label': 'DFTB 3ob-3-1'},
    'pyscf_pbe':   {'color': 'tab:blue',  'ls': '--', 'label': 'pySCF PBE/6-31G*'},
    'pyscf_b3lyp': {'color': 'tab:green', 'ls': '--', 'label': 'pySCF B3LYP/6-31G*'},
    'pyscf_gpu_pbe': {'color': 'tab:purple', 'ls': '-.', 'label': 'pySCF GPU PBE/def2-SVP'},
}

ATOM_COLORS = {'C': 'gray', 'O': 'red', 'H': 'white', 'N': 'blue'}
ATOM_SIZES  = {'C': 300, 'O': 400, 'H': 150, 'N': 350}

def _build_filename_lookup():
    """Build mapping: filename -> (mol, method, site_label, atom_idx).
    Matches .dat files with atom index appended to site label."""
    lookup = {}
    for mol_name, mol_info in MOLECULES.items():
        for target_label, atom_idx in mol_info['targets']:
            for method_name in METHODS:
                new_label = f'{target_label}{atom_idx}'
                new_fname = f'zscan_{mol_name}_{method_name}_{new_label}.dat'
                lookup[new_fname] = (mol_name, method_name, target_label, atom_idx)
    return lookup

_FILE_LOOKUP = _build_filename_lookup()

def load_all_curves():
    """Load all zscan curves from Ez_FDBM .dat files. Returns dict {key: {z, e_rel, ...}}."""
    curves = {}
    for path in sorted(glob.glob(os.path.join(REF_DIR, 'zscan_*.dat'))):
        fname = os.path.basename(path)
        if fname not in _FILE_LOOKUP:
            print(f"  SKIP (unknown filename): {fname}")
            continue
        mol, method, site_label, atom_idx = _FILE_LOOKUP[fname]
        key = f'{mol}/{method}/{site_label}'
        data = np.loadtxt(path, skiprows=4)
        curves[key] = {
            'z': data[:, 0], 'e_rel': data[:, 1],
            'mol': mol, 'method': method, 'site': site_label, 'atom_idx': atom_idx,
        }
    return curves


def plot_molecule_skeleton(mol_name, atom_names, atom_pos, target_indices, ax):
    """Draw molecule skeleton (xy projection) with all target atoms highlighted."""
    if isinstance(target_indices, (int, np.integer)):
        target_indices = [target_indices]
    target_set = set(target_indices)
    # Simple bond detection by covalent radius sum
    COV_R = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05}
    n = len(atom_names)
    for i in range(n):
        for j in range(i+1, n):
            ri = COV_R.get(atom_names[i], 0.7)
            rj = COV_R.get(atom_names[j], 0.7)
            d = np.linalg.norm(atom_pos[i] - atom_pos[j])
            if d < (ri + rj) * 1.3:
                ax.plot([atom_pos[i, 0], atom_pos[j, 0]], [atom_pos[i, 1], atom_pos[j, 1]],
                        'k-', lw=1.5, alpha=0.6)
    for i in range(len(atom_names)):
        elem = atom_names[i]
        c = ATOM_COLORS.get(elem, 'purple')
        s = ATOM_SIZES.get(elem, 200)
        edge = 'black'
        lw = 0.5
        if i in target_set:
            s *= 2.5
            edge = 'gold'
            lw = 2.5
        ax.scatter(atom_pos[i, 0], atom_pos[i, 1], c=c, s=s, edgecolors=edge, linewidths=lw, zorder=5)
        ax.text(atom_pos[i, 0], atom_pos[i, 1], str(i), fontsize=5, ha='center', va='center', zorder=6)
    ax.set_aspect('equal')
    ax.set_xlabel('x (Å)', fontsize=8)
    ax.set_ylabel('y (Å)', fontsize=8)
    target_strs = [f'{i}({atom_names[i]})' for i in target_indices]
    ax.set_title(f'{mol_name}: atoms {", ".join(target_strs)}', fontsize=10)


def plot_per_molecule(curves, outdir):
    """One figure per molecule: skeleton + all methods × sites."""
    molecules = sorted(set(c['mol'] for c in curves.values()))
    for mol in molecules:
        mol_curves = {k: v for k, v in curves.items() if v['mol'] == mol}
        sites = sorted(set(c['site'] for c in mol_curves.values()))
        n_sites = len(sites)
        # Load molecule geometry
        mol_info = MOLECULES[mol]
        atom_pos, atom_names = load_molecule(mol_info['xyz'])
        # Build site -> atom_idx mapping
        site_atomidx = {}
        for k, v in mol_curves.items():
            site_atomidx[v['site']] = v['atom_idx']

        # Linear plot (±0.1 eV) with skeleton as first subplot
        n_cols = n_sites + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), squeeze=False)
        # Skeleton — highlight all target atoms
        all_target_indices = [site_atomidx[s] for s in sites]
        plot_molecule_skeleton(mol, atom_names, atom_pos, all_target_indices, axes[0, 0])
        for col, site in enumerate(sites):
            ax = axes[0, col + 1]
            for method_name in ['dftb_mio', 'dftb_3ob', 'pyscf_pbe', 'pyscf_b3lyp', 'pyscf_gpu_pbe']:
                key = f'{mol}/{method_name}/{site}'
                if key not in mol_curves:
                    continue
                c = mol_curves[key]
                style = METHOD_STYLES.get(method_name, {'color': 'gray', 'ls': '-', 'label': method_name})
                ax.plot(c['z'], c['e_rel'], color=style['color'], ls=style['ls'],
                        label=style['label'], linewidth=1.5)
            ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax.set_xlabel('z (Å)', fontsize=10)
            if col == 0:
                ax.set_ylabel('E_rel (eV)', fontsize=10)
            ax.set_title(f'above {site} (atom {site_atomidx[site]})', fontsize=11)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([1.5, 6.0])
            ax.set_ylim([-0.1, 0.1])
        fig.suptitle(f'E(z) — {mol} (linear, ±0.1 eV)', fontsize=13)
        fig.tight_layout()
        path = os.path.join(outdir, f'zscan_{mol}_linear.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

        # Log-scale version with skeleton
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), squeeze=False)
        plot_molecule_skeleton(mol, atom_names, atom_pos, all_target_indices, axes[0, 0])
        for col, site in enumerate(sites):
            ax = axes[0, col + 1]
            for method_name in ['dftb_mio', 'dftb_3ob', 'pyscf_pbe', 'pyscf_b3lyp', 'pyscf_gpu_pbe']:
                key = f'{mol}/{method_name}/{site}'
                if key not in mol_curves:
                    continue
                c = mol_curves[key]
                style = METHOD_STYLES.get(method_name, {'color': 'gray', 'ls': '-', 'label': method_name})
                pos_mask = c['e_rel'] > 1e-6
                if pos_mask.any():
                    ax.semilogy(c['z'][pos_mask], c['e_rel'][pos_mask],
                               color=style['color'], ls=style['ls'],
                               label=style['label'], linewidth=1.5)
            ax.set_xlabel('z (Å)', fontsize=10)
            if col == 0:
                ax.set_ylabel('|E_rel| (eV, log)', fontsize=10)
            ax.set_title(f'above {site} (atom {site_atomidx[site]})', fontsize=11)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([1.5, 6.0])
        fig.suptitle(f'E(z) log-scale — {mol}', fontsize=13)
        fig.tight_layout()
        path = os.path.join(outdir, f'zscan_{mol}_log.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


ELEM_LS = {'C': '-', 'O': '--', 'H': ':', 'N': '-.'}
ELEM_LABEL = {'C': 'C', 'O': 'O', 'H': 'H', 'N': 'N'}

def _get_elem(site_label):
    """Extract element symbol from site label (e.g. 'C_anh' -> 'C', 'O_eq' -> 'O', 'H' -> 'H')."""
    return site_label.split('_')[0]

def plot_per_method_overlay(curves, outdir):
    """For each method: overlay all molecules × sites on one plot (linear + log).
    Line style per element, color per molecule."""
    methods = sorted(set(c['method'] for c in curves.values()))
    mol_list = sorted(set(c['mol'] for c in curves.values()))
    mol_colors = {m: plt.cm.tab10(i / max(1, len(mol_list))) for i, m in enumerate(mol_list)}
    for method in methods:
        method_curves = {k: v for k, v in curves.items() if v['method'] == method}
        if not method_curves:
            continue
        style = METHOD_STYLES.get(method, {'label': method})
        # Linear (±0.1 eV)
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        for key, c in sorted(method_curves.items()):
            elem = _get_elem(c['site'])
            ls = ELEM_LS.get(elem, '-')
            color = mol_colors.get(c['mol'], 'gray')
            label = f"{c['mol']}/{c['site']}({c['atom_idx']})"
            ax.plot(c['z'], c['e_rel'], color=color, ls=ls, linewidth=1.5, label=label)
        ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
        ax.set_xlabel('z (Å)', fontsize=11)
        ax.set_ylabel('E_rel (eV)', fontsize=11)
        ax.set_title(f"{style['label']} — all sites (linear, ±0.1 eV)", fontsize=12)
        ax.set_xlim([1.5, 6.0])
        ax.set_ylim([-0.1, 0.1])
        ax.legend(fontsize=6, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, f'zscan_method_{method}_linear.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
        # Log
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        for key, c in sorted(method_curves.items()):
            elem = _get_elem(c['site'])
            ls = ELEM_LS.get(elem, '-')
            color = mol_colors.get(c['mol'], 'gray')
            label = f"{c['mol']}/{c['site']}({c['atom_idx']})"
            pos_mask = c['e_rel'] > 1e-6
            if pos_mask.any():
                ax.semilogy(c['z'][pos_mask], c['e_rel'][pos_mask],
                           color=color, ls=ls, linewidth=1.5, label=label)
        ax.set_xlabel('z (Å)', fontsize=11)
        ax.set_ylabel('|E_rel| (eV, log)', fontsize=11)
        ax.set_title(f"{style['label']} — all sites (log scale)", fontsize=12)
        ax.set_xlim([1.5, 6.0])
        ax.legend(fontsize=6, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, f'zscan_method_{method}_log.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_per_element_overlay(curves, outdir):
    """For each element (C, O, H, N): subplots per method, each showing all molecules.
    Line style per method, color per molecule."""
    elements = ['C', 'O', 'H', 'N']
    method_order = ['dftb_mio', 'dftb_3ob', 'pyscf_pbe', 'pyscf_b3lyp', 'pyscf_gpu_pbe']
    mol_list = sorted(set(c['mol'] for c in curves.values()))
    mol_colors = {m: plt.cm.tab10(i / max(1, len(mol_list))) for i, m in enumerate(mol_list)}
    for elem in elements:
        elem_curves = {k: c for k, c in curves.items() if _get_elem(c['site']) == elem}
        if not elem_curves:
            continue
        methods_present = [m for m in method_order if any(c['method'] == m for c in elem_curves.values())]
        n_meth = len(methods_present)
        # Linear (±0.1 eV) — one subplot per method
        fig, axes = plt.subplots(1, n_meth, figsize=(5 * n_meth, 4), squeeze=False)
        for col, method in enumerate(methods_present):
            ax = axes[0, col]
            mstyle = METHOD_STYLES.get(method, {'label': method})
            for key, c in sorted(elem_curves.items()):
                if c['method'] != method:
                    continue
                color = mol_colors.get(c['mol'], 'gray')
                label = f"{c['mol']}/{c['site']}({c['atom_idx']})"
                ax.plot(c['z'], c['e_rel'], color=color, linewidth=1.5, label=label)
            ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax.set_xlabel('z (Å)', fontsize=10)
            if col == 0:
                ax.set_ylabel('E_rel (eV)', fontsize=10)
            ax.set_title(mstyle['label'], fontsize=11)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([1.5, 6.0])
            ax.set_ylim([-0.1, 0.1])
        fig.suptitle(f"Element {elem} — per method (linear, ±0.1 eV)", fontsize=13)
        fig.tight_layout()
        path = os.path.join(outdir, f'zscan_elem_{elem}_linear.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
        # Log — one subplot per method
        fig, axes = plt.subplots(1, n_meth, figsize=(5 * n_meth, 4), squeeze=False)
        for col, method in enumerate(methods_present):
            ax = axes[0, col]
            mstyle = METHOD_STYLES.get(method, {'label': method})
            for key, c in sorted(elem_curves.items()):
                if c['method'] != method:
                    continue
                color = mol_colors.get(c['mol'], 'gray')
                label = f"{c['mol']}/{c['site']}({c['atom_idx']})"
                pos_mask = c['e_rel'] > 1e-6
                if pos_mask.any():
                    ax.semilogy(c['z'][pos_mask], c['e_rel'][pos_mask],
                               color=color, linewidth=1.5, label=label)
            ax.set_xlabel('z (Å)', fontsize=10)
            if col == 0:
                ax.set_ylabel('|E_rel| (eV, log)', fontsize=10)
            ax.set_title(mstyle['label'], fontsize=11)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([1.5, 6.0])
        fig.suptitle(f"Element {elem} — per method (log scale)", fontsize=13)
        fig.tight_layout()
        path = os.path.join(outdir, f'zscan_elem_{elem}_log.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def fit_exponential_decay(z, e, z_min=2.5, z_max=6.0):
    """Fit log(E) = a + b*z for E > 0 in [z_min, z_max]. Returns (a, b, r2) or None."""
    mask = (z >= z_min) & (z <= z_max) & (e > 1e-10)
    if mask.sum() < 3:
        return None
    log_e = np.log(e[mask])
    z_fit = z[mask]
    A_mat = np.vstack([np.ones_like(z_fit), z_fit]).T
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, log_e, rcond=None)
    a, b = coeffs
    pred = A_mat @ coeffs
    ss_res = np.sum((log_e - pred) ** 2)
    ss_tot = np.sum((log_e - log_e.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'a': a, 'b': b, 'r2': r2, 'E0': np.exp(a), 'decay': b, 'n': int(mask.sum())}


def plot_log_linear(curves, outdir):
    """Plot log(E) vs z for all curves with fitted exponential decay lines overlaid.

    Generates per-method figures: each curve as log(E) vs z with linear fit in tail region.
    Also generates a per-element figure with subplots per method.
    """
    method_order = ['dftb_mio', 'dftb_3ob', 'pyscf_pbe', 'pyscf_b3lyp', 'pyscf_gpu_pbe']
    mol_list = sorted(set(c['mol'] for c in curves.values()))
    mol_colors = {m: plt.cm.tab10(i / max(1, len(mol_list))) for i, m in enumerate(mol_list)}

    # === Per-method: all curves on one plot, log(E) vs z, with fitted lines ===
    for method in method_order:
        method_curves = {k: v for k, v in curves.items() if v['method'] == method}
        if not method_curves:
            continue
        mstyle = METHOD_STYLES.get(method, {'label': method})
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for key, c in sorted(method_curves.items()):
            elem = _get_elem(c['site'])
            ls = ELEM_LS.get(elem, '-')
            color = mol_colors.get(c['mol'], 'gray')
            label = f"{c['mol']}/{c['site']}({c['atom_idx']})"
            z = c['z']; e = c['e_rel']
            pos_mask = e > 1e-10
            if pos_mask.any():
                ax.plot(z[pos_mask], np.log(e[pos_mask]), color=color, ls=ls, linewidth=1, label=label, alpha=0.7)
                # Fit exponential decay in tail
                fit = fit_exponential_decay(z, e, z_min=2.5, z_max=6.0)
                if fit:
                    z_line = np.linspace(2.5, 6.0, 50)
                    log_e_line = fit['a'] + fit['b'] * z_line
                    ax.plot(z_line, log_e_line, color=color, ls=':', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('z (Å)', fontsize=11)
        ax.set_ylabel('log(E_rel) [eV]', fontsize=11)
        ax.set_title(f"{mstyle['label']} — log-linear (dotted = exp fit in 2.5–6.0 Å)", fontsize=12)
        ax.set_xlim([1.5, 6.0])
        ax.legend(fontsize=5, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, f'zscan_method_{method}_loglinear.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # === Per-element: subplots per method, log(E) vs z with fitted lines ===
    elements = ['C', 'O', 'H', 'N']
    for elem in elements:
        elem_curves = {k: c for k, c in curves.items() if _get_elem(c['site']) == elem}
        if not elem_curves:
            continue
        methods_present = [m for m in method_order if any(c['method'] == m for c in elem_curves.values())]
        n_meth = len(methods_present)
        fig, axes = plt.subplots(1, n_meth, figsize=(5 * n_meth, 4), squeeze=False)
        for col, method in enumerate(methods_present):
            ax = axes[0, col]
            mstyle = METHOD_STYLES.get(method, {'label': method})
            for key, c in sorted(elem_curves.items()):
                if c['method'] != method:
                    continue
                color = mol_colors.get(c['mol'], 'gray')
                label = f"{c['mol']}/{c['site']}({c['atom_idx']})"
                z = c['z']; e = c['e_rel']
                pos_mask = e > 1e-10
                if pos_mask.any():
                    ax.plot(z[pos_mask], np.log(e[pos_mask]), color=color, linewidth=1, label=label, alpha=0.7)
                    fit = fit_exponential_decay(z, e, z_min=2.5, z_max=6.0)
                    if fit:
                        z_line = np.linspace(2.5, 6.0, 50)
                        log_e_line = fit['a'] + fit['b'] * z_line
                        ax.plot(z_line, log_e_line, color=color, ls=':', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('z (Å)', fontsize=10)
            if col == 0:
                ax.set_ylabel('log(E_rel) [eV]', fontsize=10)
            ax.set_title(mstyle['label'], fontsize=11)
            ax.legend(fontsize=5, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([1.5, 6.0])
        fig.suptitle(f"Element {elem} — log-linear (dotted = exp fit 2.5–6.0 Å)", fontsize=13)
        fig.tight_layout()
        path = os.path.join(outdir, f'zscan_elem_{elem}_loglinear.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # === Summary table of exponential decay parameters ===
    print(f"\n{'='*90}")
    print(f"{'Mol':>8} {'Site':>8} {'Method':>14} {'E0 (eV)':>12} {'decay (1/Å)':>12} {'R²':>8} {'n':>4}")
    print(f"{'-'*90}")
    for key in sorted(curves.keys()):
        c = curves[key]
        fit = fit_exponential_decay(c['z'], c['e_rel'], z_min=2.5, z_max=6.0)
        if fit:
            print(f"{c['mol']:>8} {c['site']:>8} {c['method']:>14} {fit['E0']:12.4f} {fit['decay']:12.4f} {fit['r2']:8.4f} {fit['n']:4d}")
        else:
            print(f"{c['mol']:>8} {c['site']:>8} {c['method']:>14} {'N/A':>12} {'N/A':>12} {'N/A':>8} {'N/A':>4}")
    print(f"{'='*90}")


def print_summary(curves):
    """Print numerical summary table."""
    print(f"\n{'='*80}")
    print(f"{'Mol':>6} {'Site':>4} {'Method':>14} {'E_rel(z=2)':>12} {'E_rel(z=3)':>12} {'E_rel(z=5)':>12} {'E_min':>12}")
    print(f"{'-'*80}")
    for key in sorted(curves.keys()):
        c = curves[key]
        z = c['z']; e = c['e_rel']
        e2 = np.interp(2.0, z, e)
        e3 = np.interp(3.0, z, e)
        e5 = np.interp(5.0, z, e)
        emin = e.min()
        print(f"{c['mol']:>6} {c['site']:>4} {c['method']:>14} {e2:12.4f} {e3:12.4f} {e5:12.4f} {emin:12.4f}")
    print(f"{'='*80}")


# External Fukui CO-scan jobs (GPAW / pySCF short) — review for large-molecule refs
PYSCF_CO_DIR = '/home/prokop/SIMULATIONS/Fukui_AFM/jobs_CO_scan_pyscf_short'
GPAW_CO_DIR  = '/home/prokop/SIMULATIONS/Fukui_AFM/jobs_CO_scan_gpaw'


def _parse_run_script_geometry(py_path):
    """Parse tip/target geometry from a generated run_*.py job script."""
    with open(py_path) as f:
        txt = f.read()
    m_idx = re.search(r'ATOM_IDX\s*=\s*(\d+)', txt)
    m_str = re.search(r'MOL_ATOM_STR\s*=\s*"([^"]+)"', txt)
    if not (m_idx and m_str):
        return None
    atoms = []
    for part in m_str.group(1).split(';'):
        p = part.strip().split()
        if len(p) >= 4:
            atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    idx = int(m_idx.group(1))
    ax, ay, az = atoms[idx][1], atoms[idx][2], atoms[idx][3]
    # Tip C is always at (0,0,r) in these jobs (C-down); lateral offset to labeled atom
    d_lat = float(np.hypot(ax, ay))
    return {'atom_idx': idx, 'atom_xyz': (ax, ay, az), 'd_lat': d_lat, 'n_atoms': len(atoms)}


def load_unique_pyscf_short_scans(root=PYSCF_CO_DIR):
    """Load unique pySCF short CO-scan curves (dedupe identical E(z) within a molecule).

    Returns list of dicts: mol, path, z, E_eV, Fz_eV_A, n_copies, d_lat_mean, tip_ok.
    """
    curves = []
    res = os.path.join(root, 'results')
    for mol_dir in sorted(glob.glob(os.path.join(res, 'CO_scan_*'))):
        mol_tag = os.path.basename(mol_dir)  # e.g. CO_scan_PTCDA_PBE_def2-SVP
        # molecule name between CO_scan_ and _PBE_
        m = re.match(r'CO_scan_(.+)_PBE_', mol_tag)
        mol = m.group(1) if m else mol_tag
        by_hash = {}
        for dat in sorted(glob.glob(os.path.join(mol_dir, '*_scan.dat'))):
            data_lines = [l for l in open(dat) if l.strip() and not l.startswith('#')]
            key = ''.join(data_lines)
            by_hash.setdefault(key, []).append(dat)
        # geometry from matching run scripts
        d_lats = []
        for py in glob.glob(os.path.join(root, f'run_{mol}_*.py')):
            geo = _parse_run_script_geometry(py)
            if geo:
                d_lats.append(geo['d_lat'])
        for copies in by_hash.values():
            data = np.loadtxt(copies[0], comments='#')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            z = data[:, 0]
            e_ev = data[:, 2] if data.shape[1] >= 3 else data[:, 1] * 27.2114
            fz = -np.gradient(e_ev, z)
            d_lat_mean = float(np.mean(d_lats)) if d_lats else float('nan')
            tip_ok = bool(d_lats) and max(d_lats) < 0.05
            curves.append({
                'mol': mol, 'path': copies[0], 'n_copies': len(copies),
                'z': z, 'E_eV': e_ev, 'Fz': fz,
                'd_lat_mean': d_lat_mean, 'tip_ok': tip_ok,
                'copy_names': [os.path.basename(p) for p in copies],
            })
    return curves


def inventory_gpaw_co_scans(root=GPAW_CO_DIR):
    """Return list of (mol, site, n_r_ok, co_bytes, status) for GPAW CO-scan results."""
    rows = []
    res = os.path.join(root, 'results')
    for mol_dir in sorted(glob.glob(os.path.join(res, 'CO_scan_*'))):
        mol = os.path.basename(mol_dir)
        # group by site prefix before _mol/_CO/_r
        sites = {}
        for f in os.listdir(mol_dir):
            m = re.match(r'(.+?)_(mol|CO|r[\d.]+)\.txt$', f)
            if not m:
                continue
            sites.setdefault(m.group(1), []).append(f)
        for site, files in sorted(sites.items()):
            co_b = next((os.path.getsize(os.path.join(mol_dir, f)) for f in files if f.endswith('_CO.txt')), 0)
            r_ok = sum(1 for f in files if '_r' in f and os.path.getsize(os.path.join(mol_dir, f)) > 0)
            status = 'OK' if (r_ok >= 5 and co_b > 0) else ('PARTIAL' if r_ok else 'FAIL')
            rows.append({'mol': mol, 'site': site, 'n_r_ok': r_ok, 'co_bytes': co_b, 'status': status})
    return rows


def review_external_co_scans(outdir):
    """L1/L2 review of Fukui GPAW + pySCF short CO-scan jobs → debug plots + .out."""
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, 'review.out')
    lines = []
    def log(s=''):
        print(s); lines.append(s)

    log('REVIEW: external CO-scan references (GPAW + pySCF short)')
    log(f'GPAW dir:  {GPAW_CO_DIR}')
    log(f'pySCF dir: {PYSCF_CO_DIR}')
    log('')

    # --- GPAW ---
    gpaw_rows = inventory_gpaw_co_scans()
    n_ok = sum(1 for r in gpaw_rows if r['status'] == 'OK')
    log(f'GPAW sites: {len(gpaw_rows)}  OK={n_ok}  FAIL={len(gpaw_rows)-n_ok}')
    for r in gpaw_rows:
        log(f"  {r['status']:7} {r['mol']}/{r['site']}: r_ok={r['n_r_ok']} CO_bytes={r['co_bytes']}")
    log('')
    log('VERDICT GPAW: unusable — jobs OOM-killed; no complete E(z) scan files.')
    log('')

    # --- pySCF ---
    curves = load_unique_pyscf_short_scans()
    log(f'pySCF unique curves (after dedupe): {len(curves)}')
    for c in curves:
        mono = bool(np.all(np.diff(c['E_eV']) < 0))
        log(f"  {c['mol']:12} n_copies={c['n_copies']:2d} tip_ok={c['tip_ok']} "
            f"d_lat_mean={c['d_lat_mean']:.3f} Å  "
            f"E[{c['E_eV'][0]:.3f}..{c['E_eV'][-1]:.3f}] eV  mono_dec={mono}  "
            f"file={os.path.basename(c['path'])}")
    log('')
    n_tip_ok = sum(1 for c in curves if c['tip_ok'])
    log(f'VERDICT pySCF: {len(curves)} molecules have scan.dat; only {n_tip_ok} have tip above labeled atom.')
    log('  Tip geometry: C-down at xy=(0,0); molecule not recentered → site labels are duplicates.')
    log('  Range: z=1.5–1.9 Å only (Pauli wall). Useful as one curve per molecule, NOT site-resolved.')
    log('')

    # --- Plots: unique E + Fz ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in curves:
        axes[0].plot(c['z'], c['E_eV'], 'o-', lw=1, ms=4, label=f"{c['mol']} (×{c['n_copies']})")
        axes[1].plot(c['z'], c['Fz'], 'o-', lw=1, ms=4, label=c['mol'])
    axes[0].set_xlabel('z (Å)'); axes[0].set_ylabel('E_int (eV)')
    axes[0].set_title('pySCF PBE/def2-SVP CO short — E_int (unique)')
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('z (Å)'); axes[1].set_ylabel('Fz ≈ −dE/dz (eV/Å)')
    axes[1].set_title('Numerical Fz (unique curves)')
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p_all = os.path.join(outdir, 'pyscf_short_E_Fz_all.png')
    fig.savefig(p_all, dpi=150); plt.close(fig)
    log(f'REVIEW: {p_all}')

    # Per-molecule panel (large ones emphasized)
    large = [c for c in curves if c['mol'] in ('PTCDA', 'pentacene', 'pyridine', 'pyrrol')]
    if large:
        fig, axes = plt.subplots(len(large), 2, figsize=(10, 3.2 * len(large)), squeeze=False)
        for i, c in enumerate(large):
            axes[i, 0].plot(c['z'], c['E_eV'], 'bo-', lw=1.5)
            axes[i, 0].set_ylabel('E_int (eV)'); axes[i, 0].set_title(f"{c['mol']} E  tip_ok={c['tip_ok']} d_lat={c['d_lat_mean']:.2f}")
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 1].plot(c['z'], c['Fz'], 'rs-', lw=1.5)
            axes[i, 1].set_ylabel('Fz (eV/Å)'); axes[i, 1].set_title(f"{c['mol']} Fz  copies={c['n_copies']}")
            axes[i, 1].grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel('z (Å)'); axes[-1, 1].set_xlabel('z (Å)')
        fig.tight_layout()
        p_large = os.path.join(outdir, 'pyscf_short_E_Fz_large.png')
        fig.savefig(p_large, dpi=150); plt.close(fig)
        log(f'REVIEW: {p_large}')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f'REVIEW: {out_path}')
    return curves


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot E(z) reference curves')
    parser.add_argument('--outdir', default='debug/zscan_plots', help='Output directory')
    parser.add_argument('--review-external', action='store_true',
                        help='Review Fukui GPAW/pySCF CO-scan jobs → debug/co_scan_ref_review/')
    args = parser.parse_args()

    if args.review_external:
        outdir = os.path.join(_ROOT, 'debug', 'co_scan_ref_review')
        review_external_co_scans(outdir)
        return

    outdir = os.path.join(_ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    curves = load_all_curves()
    print(f"Loaded {len(curves)} curves from {REF_DIR}")
    print_summary(curves)
    plot_per_molecule(curves, outdir)
    plot_per_method_overlay(curves, outdir)
    plot_per_element_overlay(curves, outdir)
    plot_log_linear(curves, outdir)
    print(f"\nAll plots saved to {outdir}/")

if __name__ == '__main__':
    main()
