#!/usr/bin/env python3
"""testplot_fdbm_projection_ab.py — A/B of Pauli density-variant × (A,beta)
on the dense FDBM oracle, for the "images too electrostatically distorted"
question (USER 2026-10-23).

Physics under test: PAULI_FITTED_DEFAULTS['3ob-3-1'] (A=124.84, beta=1.4330)
was globally fitted on STOCK rho_scf x GAUSSIAN (sigma=0.7) tip overlaps at
z=1.7-2.3 (tests/ref_data/Ez_FDBM/pauli_fit_results.json). Production feeds it
PROLONGED rho x DFTB CO tip — different density and different tip. PTCDA-
specific CO-tip Ez fits (testplot_fdbm_relax.PTCDA_PAULI_FIT, fitting-only)
are ~10x smaller A: stock 12.82/0.651, SA-prolonged 11.76/0.852.

Grid: rows = (rho variant, A, beta) combos, columns = df heights (canonical
3.7..4.7, amp=1, Fz at h-amp). Also prints E_pauli vs E_ES magnitudes at
contact height to quantify the ES-dominance hypothesis.

Run:  python3 tests/SPM/testplot_fdbm_projection_ab.py [--mol NTCDA PTCDA]
"""
import os
import sys
import argparse
import numpy as np

_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('Agg')

from spammm.SPM import AFM as afm_mod, AFM_utils as afm_utils
from spammm.SPM.FDBMPipeline import FDBMPipeline
from spammm.config_utils import get_dftb_basis_path
from spammm import atomicUtils as au
from spammm.forcefields.FFController import make_planar_xy, orient_long_axis_x

ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16}
BOND_L = 3.0
SCAN_MARGIN = 2.0

# (print_label, plot_label, rho_variant, A, beta)
# stock ruled out by USER 2026-10-23: stock-rho Pauli is dead at scan height -> ES-only blob.
COMBOS = [
    ('prolonged + 124.8/1.43 (current prod)',    'prolonged\nA=124.8\nβ=1.43\n(prod)', 'prolonged', 124.84, 1.4330),
    ('prolonged + 11.76/0.852 (SA Ez CO-tip)',   'prolonged\nA=11.76\nβ=0.85\n(SA-fit)', 'prolonged', 11.762, 0.8519),
]


def load_xyz_planar(path):
    pos, _, names, _, _ = au.load_xyz(path)
    atomPos = np.asarray(pos, dtype=np.float64)
    atomPos[:] = make_planar_xy(atomPos)
    orient_long_axis_x(atomPos)
    atomPos[:, 2] = 0.0
    atomPos -= (atomPos.min(0) + atomPos.max(0)) / 2
    return atomPos, np.array([ELEM_Z[e] for e in names], np.int32), list(names)


def run_mol(pipe, mol_name, xyz, outdir):
    atomPos, atomTypes, enames = load_xyz_planar(xyz)
    mol_z = 0.0
    basis_hsd = get_dftb_basis_path('3ob-3-1')
    step = 0.1
    # PRODUCTION grid (pyinvafm.fdbm.gen.grid_for convention): bbox+margin+1 halo,
    # +MESH_HALO*h extra, COM-centred, big symmetric z vacuum — do NOT shrink:
    # FFT-Poisson ES is periodic, undersized boxes contaminate the molecule field.
    sys.path.insert(0, '/home/prokop/git/invPPAFM')
    from pyinvafm.fdbm.gen import grid_for
    grid_spec, origin, ngrid, step, qb = grid_for(atomPos, mol_z, step, 1.0)
    # both variants: stock rho + prolonged rho + shared stock-diff ES
    res = pipe.run_fields(atomPos, atomTypes, basis_hsd, os.path.join(outdir, 'dftb_work'),
                          grid_spec, origin, step, ngrid, 124.84, 1.4330, 'co',
                          projection='prolonged', oracle=False)
    bufs, a = res['bufs'], pipe.ensure_afm()
    h_df, h_Fz, h_scan = afm_utils.afm_df_height_stacks(3.7, 4.7, 0.1, amp=1.0, amp_align=True)
    xs = np.arange(atomPos[:, 0].min() - SCAN_MARGIN, atomPos[:, 0].max() + SCAN_MARGIN, 0.1, np.float32)
    ys = np.arange(atomPos[:, 1].min() - SCAN_MARGIN, atomPos[:, 1].max() + SCAN_MARGIN, 0.1, np.float32)

    # component magnitudes at the closest df height (oracle fields, stock ES shared)
    rows, comp = [], {}
    for label, plabel, var, A, beta in COMBOS:
        rho = bufs['stock'] if var == 'stock' else bufs['prol']
        fd = pipe.fields_for_fit(rho, bufs['diff'], atomPos, atomTypes, origin, step, ngrid,
                                 A, beta, tip_mode='co', basis='3ob-3-1', oracle=True)
        a.queue.finish()
        FEs, _ = a.scan_fdbm(xs, ys, h_scan, mol_z=mol_z, bond_length=BOND_L,
                             K_LAT=afm_mod.stiffness_Nm_to_eVA2(0.5), K_RAD=20.0, use_fire=True)
        df_full = afm_mod.compute_df_amp(FEs[..., 2], 0.1, amp=1.0)   # df[i] = osc-center at h_scan[i]
        idx_df = [int(np.argmin(np.abs(h_scan - h))) for h in h_df]   # SSOT: scan_and_post:504
        rows.append((plabel, df_full[:, :, idx_df]))
        # E_pauli/E_ES are raw (nx,ny,nz) cl arrays on fft scratch — grab z-slice at h~3.2
        iz = int(round((mol_z + 3.2 - origin[2]) / step))
        iz = max(0, min(iz, int(ngrid[2]) - 1))
        Ep = fd['E_pauli'].get()[:, :, iz]
        Ee = fd['E_ES'].get()[:, :, iz]
        comp.setdefault(var, []).append((label, float(np.abs(Ep).max()), float(np.abs(Ee).max())))
        del fd
        print(f'  {label:48s} df@3.7 [{df_full[...,idx_df[0]].min():+.3f},{df_full[...,idx_df[0]].max():+.3f}] '
              f'|E_pauli|max={comp[var][-1][1]:.4f} |E_ES|max={comp[var][-1][2]:.4f}', flush=True)

    # figure: rows = combos, cols = df heights (canonical strip helper)
    import matplotlib.pyplot as plt
    ncol = len(h_df)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(ncol * 1.5, len(rows) * 1.45), squeeze=False)
    fig.subplots_adjust(hspace=0.04, wspace=0.03, left=0.09, top=0.82)
    extent = afm_utils.scan_extent(xs, ys)
    for ir, (label, df) in enumerate(rows):
        for ih in range(ncol):
            ax = axs[ir, ih]
            afm_utils.imshow_afm(ax, df[..., ih], extent=extent, cmap='gray',
                                 symmetric=False, colorbar=False)
            # small alpha-blended atom dots on every row — contrast-vs-geometry check
            ecol = {'H': 'cyan', 'C': 'lime', 'N': 'blue', 'O': 'red'}
            ax.scatter(atomPos[:, 0], atomPos[:, 1], s=2, alpha=0.6,
                       c=[ecol.get(e, 'yellow') for e in enames], linewidths=0)
            ax.set_xticks([]); ax.set_yticks([])
            if ir == 0:
                ax.set_title(f'df h={h_df[ih]:.2f}\nFz@{h_Fz[ih]:.2f}', fontsize=8)
            if ih == 0:
                ax.set_ylabel(label, fontsize=7)
    fig.suptitle(f'{mol_name} FDBM projection x (A,beta) A/B | dual basis: ES = stock d-rho (shared)', fontsize=11, y=0.97)
    out = os.path.join(outdir, f'{mol_name}_projection_ab.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'REVIEW: {out}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mol', nargs='+', default=['NTCDA', 'PTCDA'])
    p.add_argument('--outdir', default=os.path.join(_ROOT, 'debug', 'testplot_fdbm_projection_ab'))
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    gallery_dir = '/home/prokop/git/invPPAFM/debug/mol_gen/sheets/gallery'
    pipe = FDBMPipeline()
    for m in args.mol:
        if m.endswith('_real') and os.path.exists(os.path.join(args.outdir, f'{m}.xyz')):
            xyz = os.path.join(args.outdir, f'{m}.xyz')          # explicit file (e.g. mol2 conversion)
        elif m.startswith('real_'):
            xyz = os.path.join(_ROOT, 'data/xyz', f'{m[5:]}.xyz')  # real_<name> -> data/xyz/<name>.xyz
        else:
            xyz = os.path.join(gallery_dir, f'{m}.xyz')
            if not os.path.exists(xyz):
                xyz = os.path.join(_ROOT, 'data/xyz', f'{m}.xyz')
        print(f'=== {m} ({xyz}) ===', flush=True)
        run_mol(pipe, m, xyz, args.outdir)


if __name__ == '__main__':
    main()
