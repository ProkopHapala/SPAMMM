#!/usr/bin/env python3
"""Plot all E(z) reference curves from tests/ref_data/.

Generates:
  1. Per-molecule overlay: all methods × all sites on one plot
  2. Per-method overlay: all molecules × all sites on one plot
  3. Log-scale plot for repulsive region

Usage:
  python tests/SPM/plot_zscan_reference.py [--outdir debug/zscan_plots]
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
            for method_name in ['dftb_mio', 'dftb_3ob', 'pyscf_pbe', 'pyscf_b3lyp']:
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
            for method_name in ['dftb_mio', 'dftb_3ob', 'pyscf_pbe', 'pyscf_b3lyp']:
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
    method_order = ['dftb_mio', 'dftb_3ob', 'pyscf_pbe', 'pyscf_b3lyp']
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot E(z) reference curves')
    parser.add_argument('--outdir', default='debug/zscan_plots', help='Output directory')
    args = parser.parse_args()
    outdir = os.path.join(_ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    curves = load_all_curves()
    print(f"Loaded {len(curves)} curves from {REF_DIR}")
    print_summary(curves)
    plot_per_molecule(curves, outdir)
    plot_per_method_overlay(curves, outdir)
    plot_per_element_overlay(curves, outdir)
    print(f"\nAll plots saved to {outdir}/")

if __name__ == '__main__':
    main()
