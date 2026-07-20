#!/usr/bin/env python3
"""Extract 1D Pauli z-scan from precomputed densities and compare with Ez reference.

Uses the existing FDBM pipeline from AFM.py / AFM_utils.py:
  - afm.build_gaussian_tip() for tip density (rolled, normalized)
  - afm.compute_pauli_overlap(rho_scf, rho_tip, step, tip_rolled=True) for raw overlap
  - afm.scale_pauli_field / AFM_utils._fit_pauli_powerlaw vs site-correct Ez

For each molecule/method/target:
  1. Load rho_scf from .npy (precomputed by compute_densities.py)
  2. Build Gaussian tip density on the same grid
  3. Compute Pauli overlap via FFT cross-correlation
  4. Load Ez reference (Ez_FDBM or CO_scan_pyscf_gpu)
  5. Fit A,β so E_pauli ≈ Ez in the contact window (default 1.7–2.5 Å)
  6. Plot overlay + write comparison_summary.out

Usage:
  python tests/SPM/extract_pauli_zscan.py --molecules PTCDA --methods pyscf_gpu_pbe \\
      --outdir debug/pauli_zscan_pyscf_ptcda
"""
import os, sys, argparse, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.realpath(os.path.join(_THIS_DIR, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.SPM.run_zscan_reference import MOLECULES, METHODS, load_molecule, ref_dir_for
from spammm.SPM import AFM as afm
from spammm.SPM.AFM_utils import _fit_pauli_powerlaw

DENS_DIR = os.path.join(_ROOT, 'debug', 'densities')
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
    ix = int(round((atom_pos[0] - origin[0]) / step))
    iy = int(round((atom_pos[1] - origin[1]) / step))
    return ix, iy


def extract_z_line(overlap, origin, step, atom_pos):
    ix, iy = atom_to_grid_idx(atom_pos, origin, step)
    nx, ny, nz = overlap.shape
    ix = max(0, min(ix, nx - 1))
    iy = max(0, min(iy, ny - 1))
    z_values = origin[2] + np.arange(nz) * step
    overlap_line = overlap[ix, iy, :].copy()
    print(f"    Atom grid pixel: (ix={ix}, iy={iy})  atom_pos=({atom_pos[0]:.3f}, {atom_pos[1]:.3f})")
    return z_values, overlap_line


def load_ez_reference(mol_name, method_name, site_label, atom_idx):
    """Load Ez .dat from method ref_subdir (Ez_FDBM or CO_scan_pyscf_gpu).

    Returns dict z, e_rel, e_int (optional) or None.
    """
    method = METHODS.get(method_name, {})
    ref_dir = ref_dir_for(method) if method else os.path.join(_ROOT, 'tests', 'ref_data', 'Ez_FDBM')
    candidates = [
        f'zscan_{mol_name}_{method_name}_{site_label}{atom_idx}.dat',
        f'zscan_{mol_name}_{method_name}_def2svp_{site_label}{atom_idx}.dat',
        f'zscan_{mol_name}_pyscf_gpu_pbe_def2svp_{site_label}{atom_idx}.dat',
    ]
    path = None
    for fname in candidates:
        p = os.path.join(ref_dir, fname)
        if os.path.isfile(p):
            path = p
            break
    if path is None:
        import glob
        pat = os.path.join(ref_dir, f'zscan_{mol_name}_*{site_label}{atom_idx}.dat')
        hits = sorted(glob.glob(pat))
        path = hits[0] if hits else None
    if path is None:
        return None

    rows = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#') or not (s[0].isdigit() or s[0] in '+-.'):
                continue
            rows.append([float(x) for x in s.split()])
    if not rows:
        return None
    raw = np.asarray(rows, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    z = raw[:, 0]
    if raw.shape[1] >= 3:
        e_int, e_rel = raw[:, 1], raw[:, 2]
    else:
        e_rel = raw[:, 1]
        e_int = e_rel
    return {'z': z, 'e_rel': e_rel, 'e_int': e_int, 'path': path}


def interp_overlap_on_ez(z_rel, overlap, z_ez):
    return np.interp(z_ez, z_rel, overlap, left=np.nan, right=np.nan)


def main():
    parser = argparse.ArgumentParser(description='FDBM Pauli z-scan vs Ez reference')
    parser.add_argument('--molecules', type=str, default='all')
    parser.add_argument('--methods', type=str, default='all')
    parser.add_argument('--outdir', default=OUT_DIR)
    parser.add_argument('--sigma_tip', type=float, default=0.7, help='Gaussian tip sigma [Ang]')
    parser.add_argument('--fit-zmin', type=float, default=1.7)
    parser.add_argument('--fit-zmax', type=float, default=2.5)
    parser.add_argument('--no-fit', action='store_true', help='Use PAULI_FITTED_DEFAULTS only')
    args = parser.parse_args()

    outdir = os.path.join(_ROOT, args.outdir) if not os.path.isabs(args.outdir) else args.outdir
    os.makedirs(outdir, exist_ok=True)

    mol_names = list(MOLECULES.keys()) if args.molecules == 'all' else args.molecules.split(',')
    method_names = list(METHODS.keys()) if args.methods == 'all' else args.methods.split(',')

    all_curves = []
    summary_lines = [
        'FDBM Pauli (Gaussian tip) vs site-correct Ez',
        f'sigma_tip={args.sigma_tip}  fit_window=[{args.fit_zmin},{args.fit_zmax}] Å',
        'NOTE: Ez is full E_int; FDBM curve is Pauli-only (contact wall is the fair test)',
        '',
    ]

    for mol_name in mol_names:
        if mol_name not in MOLECULES:
            continue
        mol_info = MOLECULES[mol_name]
        mol_methods = [m for m in method_names if m in METHODS]
        if args.methods == 'all':
            mol_methods = [m for m in mol_info.get('methods', list(METHODS.keys())) if m in method_names]

        atom_pos_mol, atom_names_mol = load_molecule(mol_info['xyz'])

        for method_name in mol_methods:
            print(f"\n{'='*50}\n{mol_name} / {method_name}\n{'='*50}")
            result = load_density(mol_name, method_name)
            if result is None:
                print(f"  No density found (run compute_densities.py --molecules {mol_name} --methods {method_name})")
                summary_lines.append(f'{mol_name}/{method_name}: NO DENSITY')
                continue
            rho_mol, origin_mol, step_mol, _, _ = result
            nx, ny, nz = rho_mol.shape
            q = float(rho_mol.sum() * step_mol**3)
            n_elec = sum({'H': 1, 'C': 6, 'N': 7, 'O': 8}.get(n, 6) for n in atom_names_mol)
            print(f"  Density shape={rho_mol.shape}  q={q:.2f} (Z_sum={n_elec})")

            rho_tip = afm.build_gaussian_tip((nx, ny, nz), step_mol, args.sigma_tip)
            overlap_raw = afm.compute_pauli_overlap(rho_mol, rho_tip, step_mol, tip_rolled=True)
            print(f"  Overlap: [{overlap_raw.min():.4e}, {overlap_raw.max():.4e}]")

            method = METHODS[method_name]
            basis_key = method.get('basis', '3ob-3-1')
            default_key = {
                'mio-1-1': 'mio-1-1', '3ob-3-1': '3ob-3-1',
                '6-31g*': 'pyscf_6-31g*', 'def2-SVP': 'pyscf_6-31g*',
            }.get(basis_key, 'pyscf_6-31g*')
            pauli_default = afm.PAULI_FITTED_DEFAULTS.get(default_key, {'A': 40.0, 'beta': 1.15})

            site_data = []
            for target_label, atom_idx in mol_info['targets']:
                target_pos = atom_pos_mol[atom_idx]
                z_abs, overlap_line = extract_z_line(overlap_raw, origin_mol, step_mol, target_pos)
                z_rel = z_abs - target_pos[2]
                ez = load_ez_reference(mol_name, method_name, target_label, atom_idx)
                site_data.append({
                    'label': target_label, 'atom_idx': atom_idx, 'z_rel': z_rel,
                    'overlap': overlap_line, 'ez': ez, 'pos': target_pos,
                })
                if ez is None:
                    print(f"  WARNING: no Ez ref for {target_label}{atom_idx}")
                else:
                    print(f"  Ez ref: {ez['path']}")

            A_fit, beta_fit, r2_fit = pauli_default['A'], pauli_default['beta'], None
            if not args.no_fit:
                o_pool, e_pool, z_pool = [], [], []
                for sd in site_data:
                    if sd['ez'] is None:
                        continue
                    o_on = interp_overlap_on_ez(sd['z_rel'], sd['overlap'], sd['ez']['z'])
                    e_tgt = sd['ez']['e_rel']
                    try:
                        A_s, b_s, r2_s, _ = _fit_pauli_powerlaw(
                            sd['ez']['z'], o_on, e_tgt, z_min=args.fit_zmin, z_max=args.fit_zmax)
                        print(f"  fit {sd['label']}: A={A_s:.3f} β={b_s:.4f} R²={r2_s:.4f}")
                        summary_lines.append(f"{mol_name}/{method_name}/{sd['label']}: A={A_s:.3f} β={b_s:.4f} R²={r2_s:.4f}")
                        m = ((sd['ez']['z'] >= args.fit_zmin) & (sd['ez']['z'] <= args.fit_zmax)
                             & np.isfinite(o_on) & np.isfinite(e_tgt) & (o_on > 1e-15) & (e_tgt > 1e-15))
                        o_pool.append(o_on[m]); e_pool.append(e_tgt[m]); z_pool.append(sd['ez']['z'][m])
                    except ValueError as ex:
                        print(f"  fit {sd['label']}: FAILED ({ex})")
                        summary_lines.append(f"{mol_name}/{method_name}/{sd['label']}: FIT FAILED ({ex})")
                if o_pool:
                    A_fit, beta_fit, r2_fit, _ = _fit_pauli_powerlaw(
                        np.concatenate(z_pool), np.concatenate(o_pool), np.concatenate(e_pool),
                        z_min=args.fit_zmin, z_max=args.fit_zmax)
                    print(f"  fit POOLED: A={A_fit:.3f} β={beta_fit:.4f} R²={r2_fit:.4f}")
                    summary_lines.append(f"{mol_name}/{method_name} POOLED: A={A_fit:.3f} β={beta_fit:.4f} R²={r2_fit:.4f}")

            E_pauli = afm.scale_pauli_field(overlap_raw, step_mol, A_fit, beta_fit, return_grads=False)
            print(f"  E_pauli A={A_fit:.3f} β={beta_fit:.4f}: [{E_pauli.min():.4e}, {E_pauli.max():.4e}] eV")

            for sd in site_data:
                _, pauli_line = extract_z_line(E_pauli, origin_mol, step_mol, sd['pos'])
                ez = sd['ez']
                metrics = {}
                if ez is not None:
                    for zc in (1.5, 1.8, 2.0, 2.2):
                        ep = float(np.interp(zc, sd['z_rel'], pauli_line))
                        er = float(np.interp(zc, ez['z'], ez['e_rel']))
                        metrics[f'z{zc}'] = {'E_pauli': ep, 'E_rel': er, 'ratio': ep / er if abs(er) > 1e-6 else np.nan}
                    summary_lines.append(
                        f"  {sd['label']}{sd['atom_idx']} @1.8Å: E_pauli={metrics['z1.8']['E_pauli']:.3f}  "
                        f"Ez={metrics['z1.8']['E_rel']:.3f}  ratio={metrics['z1.8']['ratio']:.3f}")

                all_curves.append({
                    'mol': mol_name, 'method': method_name, 'site': sd['label'],
                    'atom_idx': sd['atom_idx'], 'z_rel': sd['z_rel'], 'overlap': sd['overlap'],
                    'pauli': pauli_line, 'ez': ez, 'A': A_fit, 'beta': beta_fit, 'metrics': metrics,
                })

                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
                ax1, ax2, ax3 = axes
                ax1.plot(sd['z_rel'], pauli_line, 'b-', lw=1.2,
                         label=f'FDBM Pauli (A={A_fit:.1f}, β={beta_fit:.3f})')
                if ez is not None:
                    ax1.plot(ez['z'], ez['e_rel'], 'r:', lw=2, label='Ez E_rel')
                    ax1.plot(ez['z'], ez['e_int'], 'r-', lw=1, alpha=0.5, label='Ez E_int')
                ax1.axvspan(args.fit_zmin, args.fit_zmax, color='yellow', alpha=0.15, label='fit window')
                ax1.set_xlabel('z tip above atom (Å)'); ax1.set_ylabel('E (eV)')
                ax1.set_title(f'{mol_name}/{sd["label"]}{sd["atom_idx"]} {method_name}')
                ax1.set_xlim(1.0, 6.0); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
                ax1.axhline(0, color='gray', lw=0.5)

                pos = sd['overlap'] > 1e-30
                if pos.any():
                    ax2.semilogy(sd['z_rel'][pos], sd['overlap'][pos], 'b-', lw=1, label='overlap')
                if ez is not None:
                    m = ez['e_rel'] > 1e-6
                    if m.any():
                        ax2.semilogy(ez['z'][m], ez['e_rel'][m], 'r:', lw=2, label='Ez E_rel')
                ax2.set_xlim(1.0, 6.0); ax2.set_xlabel('z (Å)'); ax2.set_ylabel('log')
                ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3); ax2.set_title('log')

                ax3.plot(sd['z_rel'], -np.gradient(pauli_line, sd['z_rel']), 'b-', lw=1, label='Fz Pauli')
                if ez is not None:
                    ax3.plot(ez['z'], -np.gradient(ez['e_rel'], ez['z']), 'r:', lw=2, label='Fz Ez')
                ax3.set_xlim(1.0, 4.0); ax3.set_xlabel('z (Å)'); ax3.set_ylabel('Fz (eV/Å)')
                ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3); ax3.axhline(0, color='gray', lw=0.5)
                ax3.set_title('Fz')

                fig.tight_layout()
                fname = f'pauli_vs_ez_{mol_name}_{method_name}_{sd["label"]}{sd["atom_idx"]}.png'
                path = os.path.join(outdir, fname)
                fig.savefig(path, dpi=150); plt.close(fig)
                print(f"  Saved: {path}")
                print(f"  REVIEW: {path}")

    methods_present = sorted(set(c['method'] for c in all_curves))
    for method in methods_present:
        curves = [c for c in all_curves if c['method'] == method]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for c in curves:
            m = c['pauli'] > 1e-12
            if m.any():
                ax.semilogy(c['z_rel'][m], c['pauli'][m], '-', lw=1.2, label=f"FDBM {c['site']}")
            if c['ez'] is not None:
                m = c['ez']['e_rel'] > 1e-12
                if m.any():
                    ax.semilogy(c['ez']['z'][m], c['ez']['e_rel'][m], ':', lw=1.5, label=f"Ez {c['site']}")
        ax.set_xlim(1.2, 8.0)
        ax.set_xlabel('z (Å)'); ax.set_ylabel('E (eV)  [log]')
        ax.set_title(f'FDBM Pauli vs Ez — {method} (log)')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        p = os.path.join(outdir, f'pauli_vs_ez_overlay_{method}.png')
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"REVIEW: {p}")

    out_path = os.path.join(outdir, 'comparison_summary.out')
    summary_lines += ['', f'REVIEW: {out_path}', f'Total curves: {len(all_curves)}']
    text = '\n'.join(summary_lines) + '\n'
    open(out_path, 'w').write(text)
    print(text)
    print(f"REVIEW: {out_path}")

    with open(os.path.join(outdir, 'comparison_metrics.json'), 'w') as f:
        json.dump([{
            'mol': c['mol'], 'method': c['method'], 'site': c['site'], 'atom_idx': c['atom_idx'],
            'A': c['A'], 'beta': c['beta'], 'metrics': c['metrics'],
        } for c in all_curves], f, indent=2)


if __name__ == '__main__':
    main()
