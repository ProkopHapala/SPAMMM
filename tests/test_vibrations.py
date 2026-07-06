"""
test_vibrations.py — Gradual vibrational analysis: H2O → benzene → PTCDA.

UFF backend (GPU) by default; optional DFTB on H2O when DFTB_EXE is set.
"""

import os
import pytest
import numpy as np

from spammm.dynamics.Vibrations import run_vibrations
from spammm.dynamics.VibrationPlot import plot_softest_modes, save_summary

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'xyz')


def _run_case(mol_file, backend='uff', outdir=None, n_plot=6):
    path = os.path.join(DATA, mol_file)
    result = run_vibrations(path, backend=backend)
    print(f"\n=== {mol_file} ({backend}) ===")
    print(result.format_table())
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        save_summary(result, os.path.join(outdir, f'{mol_file.replace(".xyz", "")}_modes.txt'))
        pngs = plot_softest_modes(result, outdir, n=n_plot, prefix=mol_file.replace('.xyz', ''))
        for p in pngs:
            print(f"REVIEW: {p}")
        print(f"REVIEW: {os.path.join(outdir, mol_file.replace('.xyz', '') + '_modes.txt')}")
    return result


@pytest.mark.gpu
def test_vibrations_h2o_uff(visual_output_dir):
    """H2O UFF: 3 vibrational modes after rigid-body removal."""
    out = visual_output_dir
    result = _run_case('H2O.xyz', backend='uff', outdir=out, n_plot=3)
    assert len(result.mode_info) == 3
    assert np.all(np.isfinite(result.frequencies_cm1))
    assert np.all(result.frequencies_cm1 > 0)
    # bending mode ~ lower frequency, mostly in-plane for planar H2O geometry
    bend = min(result.mode_info, key=lambda m: m.freq_cm1)
    assert bend.f_xy > 0.5, f"bend should be mostly in-plane, f_xy={bend.f_xy}"


@pytest.mark.gpu
@pytest.mark.slow
def test_vibrations_benzene_uff(visual_output_dir):
    """Benzene UFF: 30 internal modes; several low-frequency out-of-plane bends."""
    result = _run_case('benzene.xyz', backend='uff', outdir=visual_output_dir, n_plot=6)
    assert len(result.mode_info) == 30
    oop = sum(1 for m in result.mode_info if m.character == 'out-of-plane')
    ip = sum(1 for m in result.mode_info if m.character == 'in-plane')
    assert oop >= 3, f"expected several OOP modes for benzene, got {oop}"
    assert ip >= 3


@pytest.mark.gpu
@pytest.mark.slow
def test_vibrations_ptcda_uff(visual_output_dir):
    """PTCDA UFF: large planar molecule — many low OOP modes."""
    result = _run_case('PTCDA.xyz', backend='uff', outdir=visual_output_dir, n_plot=6)
    n_exp = 3 * len(result.enames) - 6
    assert len(result.mode_info) == n_exp
    soft6 = result.mode_info[:6]
    assert all(m.freq_cm1 < 2000 for m in soft6), "softest modes should be low-frequency"


@pytest.mark.slow
def test_vibrations_h2o_dftb(visual_output_dir):
    """H2O DFTB+ native Hessian (optional, requires DFTB_EXE)."""
    pytest.importorskip('ase', reason='ASE not installed')
    if not os.environ.get('DFTB_EXE'):
        pytest.skip('DFTB_EXE not set')
    result = _run_case('H2O.xyz', backend='dftb', outdir=visual_output_dir, n_plot=3)
    assert len(result.mode_info) >= 2
    assert np.all(np.isfinite(result.frequencies_cm1))
