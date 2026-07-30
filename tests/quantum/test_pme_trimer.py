#!/usr/bin/env python3
"""L0: Symmetric equilateral trimer PME — mirror I(x)=I(−x) + NDR in dI/dV."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module')
def solver():
    from spammm.quantum.PauliSolverCL import PauliSolverCL
    sol = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=False)
    name = sol.ctx.devices[0].name.lower()
    assert any(k in name for k in ('nvidia', 'geforce', 'rtx', 'quadro')), sol.ctx.devices[0].name
    return sol


def test_symmetric_trimer_geometry():
    from spammm.quantum import pauli_scan as ps
    params = ps.symmetric_trimer_params()
    spos, rots, angles = ps.make_site_geom(params)
    assert spos.shape == (3, 4)
    # apex on +y, base symmetric about y
    assert spos[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert spos[0, 1] > 0
    assert spos[1, 0] == pytest.approx(-spos[2, 0], abs=1e-6)
    assert spos[1, 1] == pytest.approx(spos[2, 1], abs=1e-6)
    print(f'[geom] sites:\n{spos}')


def test_symmetric_trimer_xV_mirror_and_ndr(solver, make_review):
    from spammm.quantum import pauli_scan as ps
    review = make_review('test_symmetric_trimer_xV_mirror_and_ndr')
    params = ps.symmetric_trimer_params(npix=64)
    spos, rots, _ = ps.make_site_geom(params)
    xv = ps.scan_xV(solver, spos, rots, params, nx=81, nV=60, Vmin=0.0, Vmax=0.85, return_probs=True)

    STM, dIdV = xv['STM'], xv['dIdV']
    assert np.isfinite(STM).all()
    assert STM.max() > 0

    # Mirror symmetry along horizontal cut (odd nx → center sample)
    I_fwd, I_rev = STM, STM[:, ::-1]
    scale = max(float(np.max(np.abs(STM))), 1e-30)
    rel_asym = float(np.max(np.abs(I_fwd - I_rev))) / scale
    print(f'[trimer xV] I∈[{STM.min():.3e},{STM.max():.3e}] mirror_rel={rel_asym:.3e}')
    assert rel_asym < 1e-3, f'mirror asymmetry {rel_asym}'

    ndr_min = float(dIdV.min())
    print(f'[trimer xV] dIdV min={ndr_min:.3e} (NDR)')
    assert ndr_min < -1e-7, 'expected negative dI/dV (NDR) for Qzz=0 trimer'

    if xv.get('probs') is not None:
        P = xv['probs']
        # V≈0 rows may be zeroed when PME singular — check V>0.05 only
        V = xv['Vbiases']
        mask = V > 0.05
        psum = P[mask].sum(axis=2)
        assert np.allclose(psum, 1.0, atol=2e-3), f'ΣP not 1: {psum.min()}..{psum.max()}'

    if review.active:
        review.out('Symmetric trimer xV: mirror I(x)=I(-x) + NDR')
        review.out(f'sites:\n{spos}')
        review.out(f'mirror_rel={rel_asym:.6e} ndr_min={ndr_min:.6e}')
        review.checklist('apex on +y', 'I(x)≈I(-x)', 'dIdV has NDR', 'ΣP≈1')
        review.finish()


def test_symmetric_trimer_xy_rings(solver):
    from spammm.quantum import pauli_scan as ps
    params = ps.symmetric_trimer_params(npix=64, VBias=0.69)
    spos, rots, _ = ps.make_site_geom(params)
    xy = ps.scan_xy(solver, spos, rots, params)
    assert np.isfinite(xy['STM']).all() and xy['STM'].max() > 0
    assert xy['dIdV'] is not None and float(xy['dIdV'].min()) < 0
    # C2v: I(x,y) ≈ I(-x,y)
    STM = xy['STM']
    rel = float(np.max(np.abs(STM - STM[:, ::-1]))) / max(float(np.max(np.abs(STM))), 1e-30)
    print(f'[trimer xy] V=0.69 mirror_x rel={rel:.3e} dIdVmin={xy["dIdV"].min():.3e}')
    assert rel < 5e-3, f'xy mirror asymmetry {rel}'
