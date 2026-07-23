#!/usr/bin/env python3
"""L2 STM orbital panel + frontier MO diagnostic (DFTB vs pySCF).

Thin wrapper — SSOT: ``spammm.SPM.stm_compare`` + ``run_spm.py stm *``.

Task: doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md
Artifacts: debug/stm_orbital_compare/<mol>/

Usage:
  python run_spm.py stm panel --molecule pentacene,PTCDA
  python run_spm.py stm orbitals --molecule pentacene --n-near 5
  python run_spm.py stm current --molecule pentacene --stm-tips s,pz,py

  # legacy entry (same backend):
  python tests/SPM/testplot_stm_basis_compare.py --frontier-diag --molecules pentacene
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.realpath(os.path.join(_THIS, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from spammm.SPM import stm_compare as stm

OUT_ROOT = stm.DEFAULT_OUT


def main():
    ap = argparse.ArgumentParser(description='STM / frontier MO DFTB vs pySCF panels (dev wrapper)')
    ap.add_argument('--frontier-diag', action='store_true',
                    help='±N MOs at z=0.5Å + eigspectrum + SCF/projection timings')
    ap.add_argument('--frontier-stm-diag', action='store_true',
                    help='±N MO-resolved STM current at stm-z-above; per tip (s,pz,py)')
    stm.add_stm_common_args(ap)
    stm.add_orbital_args(ap)
    ap.add_argument('--stm-z-above', type=float, default=3.0)
    ap.add_argument('--stm-tips', default='s,pz,py')
    ap.add_argument('--heights', default='2.5,3.0,3.5')
    ap.add_argument('--field', default='psi2', choices=('psi', 'psi2', 'ldos'))
    ap.add_argument('--molecules', default='pentacene,PTCDA', help='Comma list (legacy flag)')
    args = ap.parse_args()
    args.outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(_ROOT, args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    mol_args = [m.strip() for m in args.molecules.split(',') if m.strip()]
    timings = []
    for mol_name, pos, names, types, info in stm.resolve_molecules(mol_args, xyz=args.xyz):
        if args.frontier_stm_diag:
            timings.append((mol_name, stm.run_frontier_stm_current(mol_name, pos, names, types, args)))
        elif args.frontier_diag:
            timings.append((mol_name, stm.run_frontier_orbitals(mol_name, pos, names, types, args)))
        else:
            stm.run_stm_vacuum_panel(mol_name, pos, names, types, info, args)

    if timings:
        if args.frontier_stm_diag:
            print('\n======== STM DIAG SUMMARY ========')
            hdr = (f'{"mol":12s}  {"scan":>8s}  {"nMO":>4s}  {"DFTB_SCF":>9s}  {"DFTB_STM":>9s}  '
                   f'{"pySCF_SCF":>10s}  {"pySCF_STM":>10s}')
            print(hdr)
            rows = []
            for name, t in timings:
                n_mo = t.get('n_mo', 12)
                row = (f'{name:12s}  {t.get("scan", ""):>8s}  {n_mo:4d}  '
                       f'{t["t_scf_dftb"]:9.3f}  {t.get("t_proj_dftb", 0):9.3f}  '
                       f'{t["t_scf_pyscf"]:10.3f}  {t.get("t_proj_pyscf", 0):10.3f}')
                print(row)
                rows.append(
                    f'{name}: scan={t.get("scan")} nMO={n_mo} tips={t["tips"]} z={t["z_plane"]:.2f}Å  '
                    f'DFTB_SCF={t["t_scf_dftb"]:.3f}s DFTB_STM_proj={t.get("t_proj_dftb", 0):.3f}s  '
                    f'pySCF_SCF={t["t_scf_pyscf"]:.3f}s pySCF_STM_proj={t.get("t_proj_pyscf", 0):.3f}s')
            timing_path = os.path.join(args.outdir, 'TIMING_frontier_stm_diag.out')
            open(timing_path, 'w').write('\n'.join(rows) + '\n')
            print(f'REVIEW: {timing_path}')
        else:
            print('\n======== TIMING TABLE (all molecules) ========')
            print(f'{"mol":12s}  {"DFTB_SCF":>10s}  {"DFTB_proj":>10s}  {"pySCF_SCF":>10s}  {"pySCF_proj":>10s}  '
                  f'{"E_H_DFTB":>10s}  {"E_H_py":>10s}')
            rows = []
            for name, t in timings:
                row = (f'{name:12s}  {t["t_scf_dftb"]:10.3f}  {t["t_proj_dftb"]:10.3f}  '
                       f'{t["t_scf_pyscf"]:10.3f}  {t["t_proj_pyscf"]:10.3f}  '
                       f'{t["E_homo_dftb"]:10.3f}  {t["E_homo_pyscf"]:10.3f}')
                print(row)
                rows.append(
                    f'{name:12s}  DFTB_SCF={t["t_scf_dftb"]:.3f}s  DFTB_proj={t["t_proj_dftb"]:.3f}s  '
                    f'pySCF_SCF={t["t_scf_pyscf"]:.3f}s  pySCF_proj={t["t_proj_pyscf"]:.3f}s  '
                    f'HOMO_DFTB={t["E_homo_dftb"]:.3f}eV  HOMO_py={t["E_homo_pyscf"]:.3f}eV')
            timing_path = os.path.join(args.outdir, 'TIMING_frontier_diag.out')
            open(timing_path, 'w').write('\n'.join(rows) + '\n')
            print(f'REVIEW: {timing_path}')

    print(f'\nDone. Review gallery under {args.outdir}/')


if __name__ == '__main__':
    main()
