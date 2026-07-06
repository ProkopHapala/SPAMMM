"""L0 tests for rigid H-bond proton-transfer scan (DFTB, slow)."""
import os
import numpy as np
import pytest

from spammm.quantum.hbond_scan import build_ascii_hbond_system, identify_hbond_from_ascii, run_hbond_transfer_scan, ascii_examples_with_hbonds, make_hbond_transfer_path, DEBUG_DIR, save_hbond_scan_artifacts

DEBUG = DEBUG_DIR


def test_ascii_examples_with_hbonds_nonempty():
    names = ascii_examples_with_hbonds()
    assert '2Quinolone' in names
    assert all(':' not in n for n in names)


def test_hbond_transfer_path_endpoints():
    atoms = build_ascii_hbond_system('2Quinolone')
    ih, ido, iac = identify_hbond_from_ascii(atoms, pair_idx=0)
    f, path, h0, h1, s_axis = make_hbond_transfer_path(atoms.apos, ih, ido, iac, fractions=[0.0, 1.0])
    assert np.linalg.norm(path[0] - h0) < 0.2 or np.allclose(path[0], h0, atol=1e-6)
    assert np.linalg.norm(path[1] - atoms.apos[iac]) > 0.9


@pytest.mark.slow
def test_hbond_scan_2quinolone():
    """Rigid DFTB scan for O...H-N in 2Quinolone dimer; barrier should be finite."""
    name = '2Quinolone'
    atoms = build_ascii_hbond_system(name)
    ih, ido, iac = identify_hbond_from_ascii(atoms, pair_idx=0)
    work_dir = os.path.join(DEBUG, f'{name}_pytest')
    result = run_hbond_transfer_scan(atoms.enames, atoms.apos, ih, ido, iac, fractions=np.linspace(0, 1, 5), work_dir=work_dir, verbose=False, on_fail='skip')  # coarse grid for pytest
    ok = np.isfinite(result['energies_ev'])
    assert ok.sum() >= 3, f"too few converged points: {ok.sum()}"
    assert np.nanmax(result['rel_ev']) < 5.0
    assert np.isfinite(result['rel_ev'][ok]).all()
    png, xyz = save_hbond_scan_artifacts(result, atoms, name, pair_idx=0, out_dir=DEBUG)
    assert os.path.isfile(png), png
    assert os.path.isfile(xyz), xyz
    print(f"REVIEW: {png}")
    print(f"REVIEW: {xyz}")
