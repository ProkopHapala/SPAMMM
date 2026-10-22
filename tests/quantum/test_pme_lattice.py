#!/usr/bin/env python3
"""L0: pme_lattice — multipole slot mapping, Wij shells, translation invariance,
dipole mirror symmetry. Parity target: export scripts/temirov_pme.py."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

# Minimal self-contained lattice params (no geometry file needed for unit tests
# of Wij/sites; a synthetic geometry dict is injected via a tmp file).
_GEO = dict(cell_A=[[15.04, 0.0], [0.0, 11.57]], basis_A=[[0.0, 0.0]],
            origin_pixel=[48.0, 72.0], pixels_per_A=9.6,
            screen_transform_xy_to_image=[[1.0, 0.0], [0.0, 1.0]], image_y='down')


def _params(tmp_path, **over):
    import json
    g = tmp_path / 'geo.json'
    g.write_text(json.dumps(_GEO))
    p = dict(model='lattice_pme8', geometry_file=str(g), supercell=[2, 4],
             z_charge=17.6, Q0=1.0, Qzz=9.45, zVd=20.4, E_ref=-0.0585,
             W_a_meV=41.8, W_b_meV=3.5, W_ab_meV=0.7, occupied_env_energy=True,
             Rtip=3.0, zV0=-0.9, Temp=1.0, GammaT=0.01, decay=0.3,
             bMirror=True, bRamp=True, VBias=1.7, dV=0.001)
    p.update(over)
    return p


@pytest.fixture(scope='module')
def solver():
    from spammm.quantum.PauliSolverCL8 import PauliSolverCL8
    sol = PauliSolverCL8(preferred_vendor='nvidia', bPrint=False, max_iter=20000, tol=1e-12,
                         stationary_method='gmres', stationary_precision='double')
    assert 'nvidia' in sol.queue.device.vendor.lower()
    return sol


@pytest.fixture(scope='module')
def model(solver, tmp_path_factory):
    from spammm.quantum.pme_lattice import LatticePME
    return LatticePME(_params(tmp_path_factory.mktemp('lat')), solver=solver)


def test_multipole_cs_slots():
    """cs slot mapping must match the PME8.cl eval_multipole formula."""
    from spammm.quantum.pme_lattice import make_multipole_cs
    rng = np.random.default_rng(7)
    cs, order = make_multipole_cs(1.3, p=(0.2, -0.4, 0.9), Qxx=1.1, Qyy=-2.2, Qzz=9.4, Qxy=0.7, Qxz=-0.3, Qyz=0.5)
    assert order == 2 and cs.shape == (10,)
    for _ in range(50):
        d = rng.normal(size=3) * 5
        r2 = d @ d
        expect = (1 / np.sqrt(r2)) * (1.3 + (0.2 * d[0] - 0.4 * d[1] + 0.9 * d[2]) / r2
                  + (1.1 * d[0] ** 2 - 2.2 * d[1] ** 2 + 9.4 * d[2] ** 2 + 0.7 * d[0] * d[1] - 0.3 * d[2] * d[0] + 0.5 * d[1] * d[2]) / r2 ** 2)
        got = (1 / np.sqrt(r2)) * (cs[0] + (cs[1] * d[0] + cs[2] * d[1] + cs[3] * d[2]) / r2
               + ((cs[4] * d[0] + cs[9] * d[1]) * d[0] + (cs[5] * d[1] + cs[7] * d[2]) * d[1] + (cs[6] * d[2] + cs[8] * d[0]) * d[2]) / r2 ** 2)
        assert got == pytest.approx(expect, rel=1e-6), f'kernel formula mismatch at d={d}'


def test_wij_rectangular_shells(model):
    """2x4 rectangular minimum-image Wij: each row 1 long + 2 short + 2 diag."""
    W = model.W
    la = np.linalg.norm(model.cell[0])
    assert W.shape == (8, 8)
    np.testing.assert_allclose(W, W.T, atol=0)
    a_mask, b_mask, ab_mask = model.contact_masks
    per_site = dict(long=a_mask.sum(1), short=b_mask.sum(1), diag=ab_mask.sum(1))
    np.testing.assert_array_equal(per_site['long'], np.ones(8))
    np.testing.assert_array_equal(per_site['short'], np.full(8, 2))
    np.testing.assert_array_equal(per_site['diag'], np.full(8, 2))
    np.testing.assert_allclose(W.sum(1), np.full(8, W.sum(1)[0]), rtol=1e-12)  # all sites equivalent
    # occupied-environment site energies: E_i = E_ref - row sum
    np.testing.assert_allclose(model.sites[:, 3], model.params['E_ref'] - W.sum(1), atol=1e-12)


def test_translation_invariance(model):
    """Current identical under +supercell vector and under +a (site permutation)."""
    x0 = np.array([[3.0, -5.0], [30.0, 25.0]])
    I0 = model.current(1.7, x0)[0]
    for shift in (model.supercell[0], model.supercell[1], model.cell[0], model.cell[1], -model.cell[0] - 2 * model.cell[1]):
        I1 = model.current(1.7, x0 + shift)[0]
        np.testing.assert_allclose(I1, I0, rtol=1e-5, atol=1e-13, err_msg=f'shift {shift}')  # float32 tip-pos rounding


def test_dipole_mirror_symmetry(model):
    """+px current at x0+d equals -px current at x0-d (mirror about a site)."""
    p = dict(model.params)
    site0 = model.sites[0, :2].copy()
    d = np.array([[2.0, 3.0], [-1.0, 4.5], [5.0, -2.0]])
    model.configure(dict(p, px=0.8))
    Ip = model.current(1.7, site0 + d)[0]
    model.configure(dict(p, px=-0.8))
    Im = model.current(1.7, site0 - d)[0]
    np.testing.assert_allclose(Ip, Im, rtol=1e-6)
    model.configure(p)  # restore
    model.configure(p)


def test_pme4_vs_padded_pme8_2x2(tmp_path):
    """2x2 lattice: pme4 direct solve matches pme8 gmres/double within fp32."""
    from spammm.quantum.pme_lattice import LatticePME
    shared = {}
    m = LatticePME(_params(tmp_path, supercell=[2, 2]), solvers=shared)
    rng = np.random.default_rng(3)
    xy = rng.uniform(-5, 40, (100, 2))
    for v in (1.55, 1.85):
        m.configure(dict(m.params, solver='pme8'))
        I8 = m.current(v, xy)[0]
        m.configure(dict(m.params, solver='pme4'))
        I4 = m.current(v, xy)[0]
        assert np.isfinite(I8).all() and np.isfinite(I4).all()
        rel = np.abs(I4 - I8).max() / np.abs(I8).max()
        assert rel < 1e-3, f'V={v}: pme4 vs pme8 rel err {rel:.2e}'
    # per-site contact counts: 2x2 has one long, one short, one diagonal partner
    m.configure(dict(m.params, solver='pme8'))
    a_mask, b_mask, ab_mask = m.contact_masks
    np.testing.assert_array_equal(a_mask.sum(1), np.ones(4))
    np.testing.assert_array_equal(b_mask.sum(1), np.ones(4))
    np.testing.assert_array_equal(ab_mask.sum(1), np.ones(4))


def test_translation_invariance_pme4(tmp_path):
    """Nearest-replica tip PBC in PME.cl reproduces lattice translation."""
    from spammm.quantum.pme_lattice import LatticePME
    m = LatticePME(_params(tmp_path, supercell=[2, 2], solver='pme4'))
    x0 = np.array([[3.0, -5.0], [30.0, 25.0]])
    I0 = m.current(1.7, x0)[0]
    for shift in (m.supercell[0], m.supercell[1], m.cell[0], m.cell[1]):
        I1 = m.current(1.7, x0 + shift)[0]
        np.testing.assert_allclose(I1, I0, rtol=1e-5, atol=1e-13, err_msg=f'shift {shift}')


def test_scan_map_and_cuts_smoke(model):
    r = model.scan_map(1.7, 9, 64, 64)
    assert r['dIdV'].shape == (9, 9) and np.isfinite(r['dIdV']).all()
    c = model.scan_cuts(np.array([1.55, 1.70, 1.85]), 21)
    assert c['dIdV'].shape == (3, 3, 21) and np.isfinite(c['dIdV']).all()
    pix = model.site_pixels(256, 256)
    assert len(pix) >= 4 and np.isfinite(pix).all()
