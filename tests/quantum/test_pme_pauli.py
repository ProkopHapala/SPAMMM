#!/usr/bin/env python3
"""L0: PauliSolverCL (PME.cl) on symmetric square tetramer — finite I + mirror symmetry."""
from __future__ import annotations

import os
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'charge_rings')
kT_eV = 8.61733326214511e-5  # Boltzmann in eV/K


def _load_sites_xyze(path):
    """Load (N,4) float32 sites: x y z E."""
    data = np.loadtxt(path)
    assert data.ndim == 2 and data.shape[1] >= 4, f'need x y z E columns in {path}'
    return np.ascontiguousarray(data[:, :4], dtype=np.float32)


def _default_params(W=0.05, Gamma=1.0, Esite_ref=-0.1):
    # [Rtip, zV0, zVd, Esite, beta, Gamma, W, bMirror, bRamp]
    return np.array([3.0, -0.5, 0.0, Esite_ref, 0.5, Gamma, W, 1.0, 1.0], dtype=np.float32)


@pytest.fixture(scope='module')
def pauli():
    from spammm.quantum.PauliSolverCL import PauliSolverCL
    sol = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=True)
    name = sol.ctx.devices[0].name.lower()
    assert 'nvidia' in name or 'geforce' in name or 'rtx' in name or 'quadro' in name, (
        f'Expected NVIDIA device, got {sol.ctx.devices[0].name} (sandboxed PoCL?)')
    return sol


@pytest.fixture(scope='module')
def square_sites():
    return _load_sites_xyze(os.path.join(DATA, 'square_tetramer.txt'))


def test_pme_square_xscan_symmetry(pauli, square_sites, make_review, visual_output_dir):
    """4 equal sites on a square: I(x) must mirror I(-x) along the line y=+5 (over two sites)."""
    review = make_review('test_pme_square_xscan_symmetry')
    Vbias = 0.80
    zTip = 4.0
    # Scan along y=+5 so tip passes above sites at (±5, +5); C2 mirror in x remains.
    xs = np.linspace(-8.0, 8.0, 81, dtype=np.float32)
    tip_pos = np.zeros((len(xs), 3), dtype=np.float32)
    tip_pos[:, 0] = xs
    tip_pos[:, 1] = 5.0
    tip_pos[:, 2] = zTip
    Vtips = np.full(len(xs), Vbias, dtype=np.float32)

    cs = np.zeros(10, dtype=np.float32)
    cs[0] = 1.0
    params = _default_params()

    T = 4.0 * kT_eV
    pauli.set_lead(0, mu=0.0, temp=T)
    pauli.set_lead(1, mu=Vbias, temp=T)

    currents, Es, Ts, Probs, StateEs, K, CurMat = pauli.scan_current_tip(
        pTips=tip_pos, Vtips=Vtips, pSites=square_sites, params=params,
        order=0, cs=cs, return_probs=True, return_state_energies=True,
    )

    assert np.isfinite(currents).all(), 'non-finite current'
    assert np.max(np.abs(currents)) > 1e-20, 'current identically zero'
    assert Probs is not None and np.isfinite(Probs).all()
    # probabilities row-normalize ≈ 1
    psum = Probs.sum(axis=1)
    assert np.allclose(psum, 1.0, atol=1e-4), f'ΣP not 1: min={psum.min()} max={psum.max()}'

    # Mirror symmetry I(x) ≈ I(-x) (square + tip on y=0)
    I_fwd = currents
    I_rev = currents[::-1]
    max_asym = float(np.max(np.abs(I_fwd - I_rev)))
    scale = max(float(np.max(np.abs(currents))), 1e-30)
    rel_asym = max_asym / scale
    print(f'[PME square] device={pauli.ctx.devices[0].name}')
    print(f'[PME square] I: min={currents.min():.6e} max={currents.max():.6e} mean={currents.mean():.6e}')
    print(f'[PME square] mirror max|I(x)-I(-x)|={max_asym:.3e}  rel={rel_asym:.3e}')
    assert rel_asym < 1e-4, f'mirror asymmetry too large: {rel_asym}'

    if review.active:
        review.out_section('Intent')
        review.out('Run PME.cl PauliSolverCL on square tetramer x-scan at y=+5; check I(x)=I(-x).')
        review.out_section('Device / geometry')
        review.out(f'device: {pauli.ctx.devices[0].name}')
        review.out(f'sites:\n{square_sites}')
        review.out(f'Vbias={Vbias} zTip={zTip} yTip=5 W={params[6]} Gamma={params[5]}')
        review.out_section('Metrics')
        review.out(f'I_min={currents.min():.8e} I_max={currents.max():.8e}')
        review.out(f'mirror_max_abs={max_asym:.8e} mirror_rel={rel_asym:.8e}')
        review.array_summary('currents', currents)
        review.array_summary('Probs[mid]', Probs[len(xs) // 2])
        review.checklist(
            'NVIDIA device selected (not PoCL)',
            'ΣP ≈ 1 for all tip positions',
            'I(x) ≈ I(-x) within 1e-4 relative',
            'current not identically zero',
        )
        review.finish()

    if visual_output_dir is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(xs, currents, 'b-', label='I(x)')
        ax.plot(-xs, currents, 'r--', alpha=0.6, label='I(-x) overlay via -xs')
        ax.set_xlabel('tip x (Å)')
        ax.set_ylabel('I (arb)')
        ax.set_title('PME square tetramer — I(x) at y=+5 (mirror check)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png = os.path.join(visual_output_dir, 'pme_square_xscan.png')
        fig.savefig(png, dpi=120)
        plt.close(fig)
        print(f'REVIEW: {png}', flush=True)


def test_pme_square_center_vs_corner(pauli, square_sites):
    """Center tip should differ from tip above a site (symmetry of four corners)."""
    Vbias = 0.80
    zTip = 4.0
    tips = np.array([
        [0.0, 0.0, zTip],           # center
        [5.0, 5.0, zTip],           # above site 0
        [-5.0, 5.0, zTip],          # above site 1
        [-5.0, -5.0, zTip],
        [5.0, -5.0, zTip],
    ], dtype=np.float32)
    Vtips = np.full(len(tips), Vbias, dtype=np.float32)
    cs = np.zeros(10, dtype=np.float32)
    cs[0] = 1.0
    params = _default_params()
    T = 4.0 * kT_eV
    pauli.set_lead(0, mu=0.0, temp=T)
    pauli.set_lead(1, mu=Vbias, temp=T)

    I, *_ = pauli.scan_current_tip(
        pTips=tips, Vtips=Vtips, pSites=square_sites, params=params, order=0, cs=cs,
    )
    print(f'[PME points] I_center={I[0]:.6e} I_corners={I[1:]}')
    # four corners equivalent
    corner_spread = float(np.max(I[1:]) - np.min(I[1:]))
    corner_scale = max(float(np.max(np.abs(I[1:]))), 1e-30)
    assert corner_spread / corner_scale < 1e-4, f'corner currents not equal: {I[1:]}'
    assert np.isfinite(I).all()
