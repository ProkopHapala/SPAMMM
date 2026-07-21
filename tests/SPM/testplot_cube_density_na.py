#!/usr/bin/env python3
"""Diagnostics: cube Dt → Gaussian ρ_NA → Δρ with ∫Δρ≈0 and atoms on NA peaks.

Usage:
  python tests/SPM/testplot_cube_density_na.py
  python tests/SPM/testplot_cube_density_na.py --mol H2O_O --sigma 0.3
"""
from __future__ import annotations
import argparse, os
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', nargs='+', default=['H2O_O', 'HHO-h-p_1'],
                   help='neutral/<mol> under data/mithun_afm_tip_fukui')
    p.add_argument('--sigma', type=float, default=0.3)
    p.add_argument('--outdir', default=None)
    args = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')

    from spammm.SPM.AFM_utils import get_density_from_cube, plot_cube_density_diagnostics

    root = os.path.join(os.path.dirname(__file__), '..', '..')
    fukui = os.path.join(root, 'data', 'mithun_afm_tip_fukui', 'neutral')
    outdir = args.outdir or os.path.abspath(os.path.join(root, 'debug', 'testplot_cube_density_na'))
    os.makedirs(outdir, exist_ok=True)

    summary = [f'sigma_na={args.sigma}', f'outdir={outdir}', '']
    all_pass = True
    for mol in args.mol:
        cube_dir = os.path.join(fukui, mol)
        print(f'\n=== {mol} ===')
        d = get_density_from_cube(cube_dir, sigma_na=args.sigma, rescale_na=True, verbosity=0)
        png, txt = plot_cube_density_diagnostics(d, outdir, tag=mol, z_above=0.0)
        ok_q = abs(d['q_diff']) < 0.05
        # local atom-peak check from CHARGE.out
        with open(txt) as f:
            body = f.read()
        ok_atoms = 'PASS_atoms_on_NA=True' in body
        summary.append(f'{mol}: q_diff={d["q_diff"]:.3e} PASS_charge={ok_q} PASS_atoms={ok_atoms}')
        summary.append(f'  REVIEW: {png}')
        summary.append(f'  REVIEW: {txt}')
        all_pass = all_pass and ok_q and ok_atoms
        assert ok_q, f'{mol}: |q_diff|={abs(d["q_diff"])} too large'
        assert ok_atoms, f'{mol}: Gaussian NA peaks not on atoms — see {txt}'

    summary.append('')
    summary.append(f'ALL_PASS={all_pass}')
    summary_path = os.path.join(outdir, 'SUMMARY.out')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary) + '\n')
    print('\n'.join(summary))
    print(f'REVIEW: {summary_path}')
    print(f'REVIEW: {outdir}/')

if __name__ == '__main__':
    main()
