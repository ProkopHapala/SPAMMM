"""Diagnostic visual demo: run long relaxation of H2O on NaCl and plot trajectory statistics.

Run: python tests/SPM/testplot_folded_rigid_diag.py
"""
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from spammm.surfaces import FoldedRigid
from spammm.surfaces.FoldedRigid import load_fit, setup_rigid_folded, relax_folded_diag, plot_relax_diag

FIT_PATH = os.path.join(REPO_ROOT, 'data', 'fits', 'h2o_nacl.npz')
MOL_PATH = os.path.join(REPO_ROOT, 'data', 'xyz', 'H2O.xyz')
DEBUG_DIR = os.path.join(REPO_ROOT, 'debug', 'folded_rigid_diag')


def main():
    """Run 5000-step relaxation and save diagnostic plots."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    fit = load_fit(FIT_PATH)
    rbd = setup_rigid_folded(MOL_PATH, fit, z_init=2.5, xy_init=(0.0, 0.0), debug=False)
    recs = relax_folded_diag(rbd, n_steps=5000, dt=0.02, lin_damp=0.95, ang_damp=0.90, record_interval=10)

    # Print summary statistics
    print(f"\n=== Relaxation Diagnostics (5000 steps) ===")
    print(f"Energy:  initial={recs['E'][0]:.6f}  final={recs['E'][-1]:.6f}  min={recs['E'].min():.6f}")
    print(f"|F|:     initial={recs['Fmag'][0]:.6f}  final={recs['Fmag'][-1]:.6f}")
    print(f"|T|:     initial={recs['Tmag'][0]:.6f}  final={recs['Tmag'][-1]:.6f}")
    print(f"|vrot|:  initial={recs['vrot_mag'][0]:.6f}  final={recs['vrot_mag'][-1]:.6f}  max={recs['vrot_mag'].max():.6f}")
    print(f"rot/rec: max={recs['rot_per_step'].max():.6f} rad ({np.degrees(recs['rot_per_step'].max()):.2f} deg)")
    print(f"tilt:    initial={np.degrees(recs['tilt'][0]):.2f}  final={np.degrees(recs['tilt'][-1]):.2f}  max={np.degrees(recs['tilt'].max()):.2f} deg")
    print(f"COM z:   initial={recs['com'][0,2]:.4f}  final={recs['com'][-1,2]:.4f}")
    print(f"quat:    initial={recs['quat'][0]}  final={recs['quat'][-1]}")

    # Save plot
    plot_path = os.path.join(DEBUG_DIR, 'relax_diag.png')
    plot_relax_diag(recs, title="H2O/NaCl Relaxation Diagnostics (BUG: world-frame vrot)", save_path=plot_path)
    print(f"\nREVIEW: {plot_path}")

    # Save raw data
    npz_path = os.path.join(DEBUG_DIR, 'relax_diag.npz')
    np.savez(npz_path, **recs)
    print(f"Data saved to {npz_path}")

    # Sanity checks (not pytest assertions — this is a visual demo)
    if recs['E'][-1] >= recs['E'][0]:
        print(f"WARNING: Energy did not decrease ({recs['E'][0]:.6f} → {recs['E'][-1]:.6f})")
    if recs['Fmag'][-1] >= recs['Fmag'][0] * 2:
        print(f"WARNING: Force grew ({recs['Fmag'][0]:.6f} → {recs['Fmag'][-1]:.6f})")


if __name__ == '__main__':
    main()
