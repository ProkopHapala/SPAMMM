#!/usr/bin/env python3
"""Systematic FDBM AFM gallery: DFT cube vs DFTB stock vs prolonged (3+3 rows).

SSOT heights (do NOT override unless USER asks):
  df  3.7–4.7 Å @ dz=0.1
  Fz  amp-aligned @ h−amp  (amp=1 → 2.7–3.7)

Pauli EVAL (same for all mols / all rows): AFM.PAULI_FITTED_DEFAULTS['3ob-3-1']
  A=124.84  β=1.4330
  Fitting scripts (e.g. PTCDA Ez) must NOT override this path.

Output layout:
  debug/AFM_CLI_FDBM/
    README.md
    SUMMARY.out
    <mol>/
      compare_cube_stock_prolonged.png   # 6 rows: df×3 + Fz×3
      …

Usage: python tests/SPM/run_afm_cli_fdbm_gallery.py
See user_guide/SPM_CLI.md § Pauli A,β; doc/AGENTS/skills/afm-plotting/SKILL.md.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Molecule order for the clean gallery table (all have rho_N.cube)
GALLERY_MOLS = [
    'pentacene',
    'PTCDA',
    'adenine-uracil',
    'adenine-uracil-iso',
    'azaindol_dimer',
    'azaindol_isodimer',
    'benzoicacid_dimer',
    'benzoicamid_dimer',
    'phtalo_1-dftb-relax',
    'phtalo_2-dftb-relax',
]

OUTDIR_DEFAULT = 'debug/AFM_CLI_FDBM'

README = """# AFM_CLI_FDBM — systematic DFT vs DFTB FDBM gallery

One subfolder per molecule. Strip = **6 rows**: df(cube, prolonged, stock) + Fz(cube, prolonged, stock).

## Height SSOT (CLI defaults — do not invent)

| | Value |
|--|-------|
| df window | **3.7 – 4.7 Å**, dz=**0.1** |
| amp | **1.0 Å** (peak) |
| Fz columns | **amp-aligned** @ **h − amp** → **2.7 – 3.7 Å** |

## Pauli A,β SSOT (evaluation — transferable)

| | Value |
|--|-------|
| Source | `AFM.PAULI_FITTED_DEFAULTS['3ob-3-1']` |
| **A** | **124.84** |
| **β** | **1.4330** |

Same `(A,β)` for **every molecule** and **cube / stock / prolonged**.  
Molecule-specific fits (e.g. `PTCDA_PAULI_FIT`) are **fitting-only** — never used here.

## Re-run

```bash
python tests/SPM/run_afm_cli_fdbm_gallery.py
python tests/SPM/run_afm_cli_fdbm_gallery.py --molecule pentacene PTCDA
```

## Per-molecule REVIEW

Open `<mol>/compare_cube_stock_prolonged.png` (and `per_image/` for independent clim).
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--molecule', nargs='*', default=None, help=f'Subset (default: all {len(GALLERY_MOLS)})')
    p.add_argument('--outdir', default=OUTDIR_DEFAULT)
    args = p.parse_args(argv)

    root = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(root, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    open(os.path.join(outdir, 'README.md'), 'w').write(README)

    mols = list(args.molecule) if args.molecule else list(GALLERY_MOLS)
    unknown = set(mols) - set(GALLERY_MOLS)
    if unknown:
        print(f'Unknown molecules {unknown}; choose from {GALLERY_MOLS}', file=sys.stderr)
        return 2

    cmd = [
        sys.executable, os.path.join(root, 'run_spm.py'), 'panel-fukui',
        '--outdir', outdir,
        '--molecule', *mols,
        # height SSOT: omit overrides → CLI defaults 3.7–4.7 / dz=0.1 / amp-align
        '--basis', '3ob-3-1',
        '--tip-mode', 'co',
        '--scale', 'per_column',
        '--df-cmap', 'gray',
        '--cmap', 'seismic',
    ]
    print('RUN:', ' '.join(cmd), flush=True)
    log_path = os.path.join(outdir, 'RUN.log')
    with open(log_path, 'w') as log:
        log.write(' '.join(cmd) + '\n\n')
        log.flush()
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    print(f'REVIEW: {outdir}/README.md')
    print(f'REVIEW: {outdir}/SUMMARY.out')
    print(f'REVIEW: {log_path}')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
