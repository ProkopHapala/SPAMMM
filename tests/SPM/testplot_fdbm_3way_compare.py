"""testplot_fdbm_3way_compare.py — FDBM vs coremesh vs separable df strips.

Joins the per-method comparison_arrays.npz produced by testplot_fdbm_fields_compress.py
(--method coremesh and --method sep) into ONE df-only height strip per molecule:
rows = FDBM reference / coremesh / separable 2.5D, labelled with archive size + fit time.

    python3 tests/SPM/testplot_fdbm_3way_compare.py sw_00 duo_03 uracil graft_02 genflake_00
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm.SPM import AFM_utils as afm_utils  # noqa: E402

OUTROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'testplot_fdbm_fields_compress')
OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'testplot_fdbm_3way_compare')
CM_TAG = 'coremesh_poly8sp-As3p2Bs11'


def _stats(summary_path):
    t = open(summary_path).read()
    fit = re.search(r'fit_sec=([0-9.]+)', t)
    arch = re.search(r'archive_(?:pickle_)?bytes=(\d+)', t)
    return (float(fit.group(1)) if fit else None,
            int(arch.group(1)) // 1024 if arch else None)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    mols = sys.argv[1:] or ['sw_00']
    for mol in mols:
        sep_dir = os.path.join(OUTROOT, f'{mol}_s0.1_sep0.8')
        cm_dir = os.path.join(OUTROOT, f'{mol}_s0.1_{CM_TAG}')
        zs = np.load(os.path.join(sep_dir, 'comparison_arrays.npz'))
        zc = np.load(os.path.join(cm_dir, 'comparison_arrays.npz'))
        s_fit, s_kb = _stats(os.path.join(sep_dir, 'SUMMARY.out'))
        c_fit, c_kb = _stats(os.path.join(cm_dir, 'SUMMARY.out'))
        # dense float4 grid for this field (bytes on disk if stored raw)
        dense_mb = int(zs['df_ref'].size) * 16 / 1e6  # df slice only is not the field; note in title instead

        variants = {
            'FDBM': {'df': zs['df_ref']},
            'sep': {'df': zs['contact_0.8_df']},
            'coremesh': {'df': zc['coremesh_1_df']},
        }
        row_specs = [
            ('df', 'FDBM', 'FDBM (dense ref)', 'gray'),
            ('df', 'sep', f'sep 0.8A | {s_kb}kB | fit {s_fit:.2f}s', 'gray'),
            ('df', 'coremesh', f'coremesh h1 | {c_kb}kB | fit {c_fit:.2f}s', 'gray'),
        ]
        h = np.asarray(zs['h_Fz'])
        extent = afm_utils.scan_extent(zs['scan_xs'], zs['scan_ys'])
        out = os.path.join(OUTDIR, f'{mol}_df_3way.png')
        afm_utils.plot_afm_variant_height_strip(
            variants, row_specs, h, out,
            title=f'{mol}  df: FDBM dense grid vs compressed stores (amp=1A, h_Fz labels)',
            extent=extent, amp=None, amp_align=False, dpi=150)
        print(f'REVIEW: {os.path.abspath(out)}', flush=True)


if __name__ == '__main__':
    main()
