#!/usr/bin/env python3
"""Extract 1D Pauli z-scan from precomputed densities and compare with Ez reference.

Uses the existing FDBM pipeline from AFM.py / AFM_utils.py:
  - afm.build_gaussian_tip() for tip density (rolled, normalized)
  - afm.compute_pauli_overlap(rho_scf, rho_tip, step, tip_rolled=True) for raw overlap
  - afm.scale_pauli_field(overlap, step, A, beta) for scaled Pauli energy

For each molecule/method/target:
  1. Load rho_scf from .npy (precomputed by compute_densities.py)
  2. Build Gaussian tip density on the same grid
  3. Compute Pauli overlap via FFT cross-correlation (tip_rolled=True)
  4. Scale with PAULI_FITTED_DEFAULTS to get E_pauli in eV
  5. Find the grid pixel closest to the target atom's (x,y) position
  6. Extract the 1D z-line through that pixel
  7. Load the corresponding Ez reference curve from tests/ref_data/Ez_FDBM/
  8. Plot overlay: Ez_ref vs E_pauli (both raw overlap and scaled)

Key careful step: mapping atom (x,y) to grid pixel (ix, iy).
  Grid point (ix, iy, iz) has real-space position:
    r = origin + ix*step*x̂ + iy*step*ŷ + iz*step*ẑ
  So: ix = round((atom_x - origin[0]) / step)
      iy = round((atom_y - origin[1]) / step)
  The z-line then covers: z_values = origin[2] + iz*step for iz=0..nz-1
  The "tip height above atom" is: z_tip = z_values - atom_z

Usage:
  python tests/SPM/extract_pauli_zscan.py
  python tests/SPM/extract_pauli_zscan.py --molecules H2O --methods dftb_3ob
"""
import os, sys, argparse
import numpy as np
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
OUT_DIR = os.path.join(_ROOT, 'debug', 'pauli_zscan')


def load_density(mol_name, method_name):
    """Load rho_scf .npy and metadata. Returns (rho, origin, step, atom_pos, atom_names)."""
    npy_path = os.path.join(DENS_DIR, f'rho_{mol_name}_{method_name}.npy')
    meta_path = os.path.join(DENS_DIR, f'rho_{mol_name}_{method_name}.meta.npz')
    if not os.path.exists(npy_path) or not os.path.exists(meta_path):
        return None
    rho = np.load(npy_path)
    meta = np.load(meta_path)
    return rho, meta['origin'], float(meta['step']), meta['atom_pos'], list(meta['atom_names'])


def atom_to_grid_idx(atom_pos, origin, step):
    """Map atom (x,y) position to nearest grid pixel (ix, iy).
    
    Grid point (ix, iy, iz) real-space position:
      r = origin + ix*step*x̂ + iy*step*ŷ + iz*step*ẑ
    So: ix = round((atom_x - origin[0]) / step)
        iy = round((atom_y - origin[1]) / step)
    """
    ix = int(round((atom_pos[0] - origin[0]) / step))
    iy = int(round((atom_pos[1] - origin[1]) / step))
    return ix, iy


def extract_z_line(overlap, origin, step, atom_pos):
    """Extract 1D z-line from 3D overlap field at the pixel closest to atom (x,y).
    
    Returns (z_values, overlap_values) where z_values are absolute z coordinates.
    The "tip height above atom" = z_values - atom_pos[2].
    """
    ix, iy = atom_to_grid_idx(atom_pos, origin, step)
    nx, ny, nz = overlap.shape
    # Clamp to valid range
    ix = max(0, min(ix, nx - 1))
    iy = max(0, min(iy, ny - 1))
    
    z_values = origin[2] + np.arange(nz) * step
    overlap_line = overlap[ix, iy, :].copy()
    
    print(f"    Atom grid pixel: (ix={ix}, iy={iy})  atom_pos=({atom_pos[0]:.3f}, {atom_pos[1]:.3f})")
    print(f"    Pixel real pos:  ({origin[0]+ix*step:.3f}, {origin[1]+iy*step:.3f})")
    print(f"    z range: [{z_values[0]:.2f}, {z_values[-1]:.2f}] Å, atom_z={atom_pos[2]:.3f} Å")
    
    return z_values, overlap_line


def load_ez_reference(mol_name, method_name, site_label, atom_idx):
    """Load Ez reference .dat file. Returns (z, e_rel) or None."""
    fname = f'zscan_{mol_name}_{method_name}_{site_label}{atom_idx}.dat'
    path = os.path.join(EZ_DIR, fname)
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, skiprows=4)
    return data[:, 0], data[:, 1]


def main():
    parser = argparse.ArgumentParser(description='Extract Pauli z-scan from densities and compare with Ez reference')
    parser.add_argument('--molecules', type=str, default='all')
    parser.add_argument('--methods', type=str, default='all')
    parser.add_argument('--outdir', default=OUT_DIR)
    parser.add_argument('--sigma_tip', type=float, default=0.7, help='Gaussian tip sigma [Ang]')
    args = parser.parse_args()
    
    outdir = os.path.join(_ROOT, args.outdir) if not os.path.isabs(args.outdir) else args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    mol_names = list(MOLECULES.keys()) if args.molecules == 'all' else args.molecules.split(',')
    method_names = list(METHODS.keys()) if args.methods == 'all' else args.methods.split(',')
    
    # Collect all curves for summary plots
    all_curves = []
    
    for mol_name in mol_names:
        if mol_name not in MOLECULES:
            continue
        mol_info = MOLECULES[mol_name]
        mol_methods = mol_info.get('methods', list(METHODS.keys()))
        mol_methods = [m for m in mol_methods if m in method_names]
        
        atom_pos_mol, atom_names_mol = load_molecule(mol_info['xyz'])
        
        for method_name in mol_methods:
            print(f"\n{'='*50}")
            print(f"{mol_name} / {method_name}")
            print(f"{'='*50}")
            
            # Load molecule density
            result = load_density(mol_name, method_name)
            if result is None:
                print(f"  No density found, skipping")
                continue
            rho_mol, origin_mol, step_mol, _, _ = result
            nx, ny, nz = rho_mol.shape
            
            # Build Gaussian tip density on the SAME grid as molecule (rolled, normalized)
            rho_tip = afm.build_gaussian_tip((nx, ny, nz), step_mol, args.sigma_tip)
            
            # Compute Pauli overlap using FDBM pipeline (tip_rolled=True)
            overlap_raw = afm.compute_pauli_overlap(rho_mol, rho_tip, step_mol, tip_rolled=True)
            print(f"  Overlap: range=[{overlap_raw.min():.4e}, {overlap_raw.max():.4e}]")
            
            # Scale with fitted Pauli parameters
            # Map method_name to basis name for PAULI_FITTED_DEFAULTS lookup
            method = METHODS[method_name]
            basis_key = method.get('basis', '3ob-3-1')  # mio-1-1, 3ob-3-1, etc.
            pauli_params = afm.PAULI_FITTED_DEFAULTS.get(basis_key, {'A': 509.28, 'beta': 1.0586})
            E_pauli = afm.scale_pauli_field(overlap_raw, step_mol, pauli_params['A'], pauli_params['beta'], return_grads=False)
            print(f"  E_pauli (A={pauli_params['A']:.2f}, β={pauli_params['beta']:.4f}): range=[{E_pauli.min():.4e}, {E_pauli.max():.4e}] eV")
            
            # Extract z-lines for each target atom
            for target_label, atom_idx in mol_info['targets']:
                print(f"\n  Target: {target_label} (atom {atom_idx})")
                target_pos = atom_pos_mol[atom_idx]
                
                z_abs, overlap_line = extract_z_line(overlap_raw, origin_mol, step_mol, target_pos)
                _, pauli_line = extract_z_line(E_pauli, origin_mol, step_mol, target_pos)
                # Convert to "tip height above atom" = z_abs - atom_z
                z_rel = z_abs - target_pos[2]
                
                # Load Ez reference
                ez_ref = load_ez_reference(mol_name, method_name, target_label, atom_idx)
                
                # Store for summary
                all_curves.append({
                    'mol': mol_name, 'method': method_name, 'site': target_label,
                    'atom_idx': atom_idx, 'z_rel': z_rel, 'overlap': overlap_line,
                    'pauli': pauli_line, 'ez_ref': ez_ref,
                })
                
                # Plot comparison: 3 panels (linear, log, linear zoomed)
                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                # Panel 1: linear — E_pauli (scaled) vs Ez ref
                ax1 = axes[0]
                ax1.plot(z_rel, pauli_line, 'b-', label=f'E_pauli (A={pauli_params["A"]:.1f}, β={pauli_params["beta"]:.3f})', linewidth=1.5)
                if ez_ref is not None:
                    ax1.plot(ez_ref[0], ez_ref[1], 'r--', label='Ez reference', linewidth=1.5)
                ax1.set_xlabel('z (tip height above atom, Å)')
                ax1.set_ylabel('Energy (eV)')
                ax1.set_title(f'{mol_name}/{target_label}({atom_idx}) {method_name} — linear')
                ax1.set_xlim([1.0, 6.0])
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
                # Panel 2: log — raw overlap vs Ez ref
                ax2 = axes[1]
                pos_mask = overlap_line > 1e-30
                if pos_mask.any():
                    ax2.semilogy(z_rel[pos_mask], overlap_line[pos_mask], 'b-', label='Raw overlap (A=1,β=1)', linewidth=1.5)
                if ez_ref is not None:
                    pos_ez = ez_ref[1] > 1e-6
                    if pos_ez.any():
                        ax2.semilogy(ez_ref[0][pos_ez], ez_ref[1][pos_ez], 'r--', label='Ez reference', linewidth=1.5)
                ax2.set_xlabel('z (tip height above atom, Å)')
                ax2.set_ylabel('|Overlap / Energy| (log)')
                ax2.set_title(f'{mol_name}/{target_label}({atom_idx}) {method_name} — log')
                ax2.set_xlim([1.0, 6.0])
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
                # Panel 3: linear zoomed ±0.1 eV
                ax3 = axes[2]
                ax3.plot(z_rel, pauli_line, 'b-', label='E_pauli', linewidth=1.5)
                if ez_ref is not None:
                    ax3.plot(ez_ref[0], ez_ref[1], 'r--', label='Ez reference', linewidth=1.5)
                ax3.set_xlabel('z (tip height above atom, Å)')
                ax3.set_ylabel('Energy (eV)')
                ax3.set_title(f'{mol_name}/{target_label}({atom_idx}) {method_name} — ±0.1 eV')
                ax3.set_xlim([1.5, 6.0])
                ax3.set_ylim(-0.1, 0.1)
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
                ax3.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
                fig.tight_layout()
                fname = f'pauli_vs_ez_{mol_name}_{method_name}_{target_label}{atom_idx}.png'
                path = os.path.join(outdir, fname)
                fig.savefig(path, dpi=150)
                plt.close(fig)
                print(f"  Saved: {path}")
    
    # Summary plot: all curves per method (log scale, raw overlap)
    methods_present = sorted(set(c['method'] for c in all_curves))
    for method in methods_present:
        curves = [c for c in all_curves if c['method'] == method]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for c in curves:
            label = f"{c['mol']}/{c['site']}({c['atom_idx']})"
            pos_mask = c['overlap'] > 1e-30
            if pos_mask.any():
                ax.semilogy(c['z_rel'][pos_mask], c['overlap'][pos_mask], linewidth=1, label=label)
        ax.set_xlabel('z (Å)')
        ax.set_ylabel('Raw Pauli overlap (log)')
        ax.set_title(f'Pauli overlap — {method} (all molecules/sites)')
        ax.set_xlim([1.0, 6.0])
        ax.legend(fontsize=6, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, f'pauli_overlap_{method}_log.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")
    
    print(f"\nAll plots saved to {outdir}/")
    print(f"Total curves: {len(all_curves)}")


if __name__ == '__main__':
    main()
