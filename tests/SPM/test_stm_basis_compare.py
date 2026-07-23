"""L0 smoke: DFTB STM stock vs prolonged differs at vacuum height (benzene).

Task: doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md
"""
import os
import numpy as np
import pytest

_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))

ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


@pytest.mark.gpu
@pytest.mark.slow
def test_stm_prolonged_differs_from_stock_benzene(tmp_path):
    from spammm import atomicUtils as au
    from spammm.quantum.DFTB_utils import WFC_HSD_PATHS
    from spammm.SPM import AFM_utils as afm_utils

    xyz = os.path.join(_ROOT, 'data', 'xyz', 'benzene.xyz')
    pos, _, names, _, _ = au.load_xyz(xyz)
    pos = np.asarray(pos, dtype=np.float64)
    types = np.array([ELEM_Z[e] for e in names], dtype=np.int32)

    lo = pos[:, :2].min(axis=0) - 2.0
    hi = pos[:, :2].max(axis=0) + 2.0
    xs = np.arange(lo[0], hi[0], 0.4)
    ys = np.arange(lo[1], hi[1], 0.4)
    heights = np.array([3.0], dtype=np.float64)

    res = afm_utils.compute_stm_basis_variants(
        pos, types, WFC_HSD_PATHS['3ob-3-1'], str(tmp_path / 'dftb'),
        xs, ys, heights, projection_variants=('stock', 'prolonged'), field='psi2', verbosity=0)

    stock = res['maps']['stock']['HOMO'][:, :, 0]
    prol = res['maps']['prolonged']['HOMO'][:, :, 0]
    assert np.isfinite(stock).all()
    assert float(stock.max()) > 0.0
    assert float(np.max(np.abs(prol - stock))) > 0.0
    assert float(prol.max()) > 0.0
    # HOMO must be valence (not legacy eigvals<0 near-zero virtual)
    assert float(res['E_homo']) < -0.05, f"HOMO energy {res['E_homo']} Ha looks like a virtual (use valence n_occ)"
    assert res['homo'] == afm_utils.dftb_n_valence_electrons(enames=names) // 2 - 1
