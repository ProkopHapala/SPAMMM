"""
test_scan_contract.py — L0 regression for the shared scan/result contract.

Unify_Morse_FDBM_Pipeline (Agent_1 Wave 1):
  - build_scan_points_vectorized (spammm/SPM/AFM.py) matches a nested-loop
    reference for a 5x5x3 grid.
  - ScanSpec / ScanResult dataclasses construct and hold the frozen fields.
  - shared_postprocess extracts Fz at amp-aligned heights and computes df via
    compute_df_amp_dir on a synthetic force volume, matching the reference
    extraction logic in run_fdbm_pp_from_density.
"""
import numpy as np
import pytest

from spammm.SPM.AFM import build_scan_points_vectorized
from spammm.SPM.AFM_utils import ScanSpec, ScanResult, shared_postprocess, afm_df_height_stacks


# ── build_scan_points_vectorized ──────────────────────────────────────────────

def test_build_scan_points_matches_nested_loop_5x5x3():
    """Vectorized builder must match a triple nested-loop reference (5x5x3)."""
    scan_xs = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    scan_ys = np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32)
    h_scan = np.array([3.0, 3.5, 4.0], dtype=np.float32)
    nx, ny, nz = len(scan_xs), len(scan_ys), len(h_scan)

    # reference: triple nested loop, ix-major then iy then iz (C-order, ij)
    ref = np.zeros((nx * ny * nz, 4), dtype=np.float32)
    k = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                ref[k, 0] = scan_xs[ix]
                ref[k, 1] = scan_ys[iy]
                ref[k, 2] = h_scan[iz]
                k += 1

    pts = build_scan_points_vectorized(scan_xs, scan_ys, h_scan)
    assert pts.shape == (nx * ny * nz, 4), f"shape {pts.shape} != {(nx*ny*nz, 4)}"
    assert pts.dtype == np.float32
    assert pts.flags['C_CONTIGUOUS']
    np.testing.assert_allclose(pts, ref, atol=0.0, rtol=0.0)


def test_build_scan_points_w_component_zero():
    """4th channel must be exactly 0 (reserved for kernel secondary results)."""
    scan_xs = np.linspace(0, 1, 4, dtype=np.float32)
    scan_ys = np.linspace(0, 1, 3, dtype=np.float32)
    h_scan = np.array([2.0, 3.0], dtype=np.float32)
    pts = build_scan_points_vectorized(scan_xs, scan_ys, h_scan)
    np.testing.assert_array_equal(pts[:, 3], np.zeros(pts.shape[0]))


def test_build_scan_points_single_height():
    """nz=1 must still produce a valid (nx*ny, 4) grid."""
    scan_xs = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    scan_ys = np.array([0.0, 0.5], dtype=np.float32)
    h_scan = np.array([4.0], dtype=np.float32)
    pts = build_scan_points_vectorized(scan_xs, scan_ys, h_scan)
    assert pts.shape == (3 * 2 * 1, 4)
    np.testing.assert_allclose(pts[:, 2], 4.0)


# ── ScanSpec / ScanResult dataclass contract ──────────────────────────────────

def _make_scan_spec():
    h_df, h_Fz, h_scan = afm_df_height_stacks(3.7, 4.7, 0.1, amp=1.0, amp_align=True)
    return ScanSpec(
        scan_xs=np.arange(-2.0, 2.0, 0.1, dtype=np.float32),
        scan_ys=np.arange(-1.0, 1.0, 0.1, dtype=np.float32),
        h_df=h_df, h_Fz=h_Fz, h_scan=h_scan,
        amplitude=1.0, osc_dir=(0.0, 0.0, 1.0),
        K_LAT=0.0312, K_RAD=20.0, bond_length=3.0, scan_margin=2.0,
    )


def test_scanspec_holds_all_frozen_fields():
    spec = _make_scan_spec()
    for name in ['scan_xs', 'scan_ys', 'h_df', 'h_Fz', 'h_scan',
                 'amplitude', 'osc_dir', 'K_LAT', 'K_RAD', 'bond_length',
                 'scan_margin']:
        assert hasattr(spec, name), f"ScanSpec missing field {name}"
    assert spec.scan_margin == 2.0  # default
    assert spec.amplitude == 1.0
    assert spec.osc_dir == (0.0, 0.0, 1.0)


def test_scanresult_default_metadata():
    res = ScanResult(
        df=np.zeros((2, 2, 3), dtype=np.float32),
        Fz=np.zeros((2, 2, 3), dtype=np.float32),
        heights=np.arange(3, dtype=np.float32),
        heights_Fz=np.arange(3, dtype=np.float32),
        scan_xs=np.arange(2, dtype=np.float32),
        scan_ys=np.arange(2, dtype=np.float32),
        amp_align=True,
        FEs=np.zeros((2, 2, 5, 4), dtype=np.float32),
        tip_disp={'dx': np.zeros((2, 2, 5)), 'dy': np.zeros((2, 2, 5)), 'dz': np.zeros((2, 2, 5))},
        E_diss=np.zeros((2, 2, 3), dtype=np.float32),
    )
    assert res.backend_name == 'unknown'
    assert res.fft_path == 'none'


# ── shared_postprocess ────────────────────────────────────────────────────────

def test_shared_postprocess_fz_df_extraction():
    """shared_postprocess must extract Fz at amp-aligned heights and df via
    compute_df_amp_dir, matching the reference logic in run_fdbm_pp_from_density."""
    spec = _make_scan_spec()
    nx, ny, nz_scan = len(spec.scan_xs), len(spec.scan_ys), len(spec.h_scan)

    # Synthetic force volume: Fz = -z (repulsive downward grows with height),
    # Fx = Fy = 0, E = 0. Deterministic, smooth → df well-defined.
    FEs = np.zeros((nx, ny, nz_scan, 4), dtype=np.float32)
    z3 = spec.h_scan[np.newaxis, np.newaxis, :, np.newaxis]
    FEs[..., 2] = -z3[..., 0]  # Fz = -z

    res = shared_postprocess(FEs, spec, backend_name='fdbm', fft_path='GPU')

    # Fz extracted at idx_Fz (nearest h_scan to each h_Fz)
    idx_Fz = [int(np.argmin(np.abs(spec.h_scan - h))) for h in spec.h_Fz]
    Fz_ref = FEs[..., 2][:, :, idx_Fz]
    np.testing.assert_allclose(res.Fz, Fz_ref, rtol=1e-6, atol=1e-6)

    # df via compute_df_amp_dir then sliced at idx_df
    spacing = (float(spec.scan_xs[1] - spec.scan_xs[0]),
               float(spec.scan_ys[1] - spec.scan_ys[0]),
               float(spec.h_scan[1] - spec.h_scan[0]))
    from spammm.SPM import AFM as afm
    osc_n = np.asarray(spec.osc_dir, dtype=np.float64)
    osc_n = osc_n / np.linalg.norm(osc_n)
    df_full = afm.compute_df_amp_dir(FEs, spacing, osc_dir=osc_n, amp=float(spec.amplitude))
    idx_df = [int(np.argmin(np.abs(spec.h_scan - h))) for h in spec.h_df]
    df_ref = df_full[:, :, idx_df]
    np.testing.assert_allclose(res.df, df_ref, rtol=1e-6, atol=1e-6)

    # No backward stroke → E_diss zeros
    assert res.E_diss.shape == (nx, ny, len(spec.h_df))
    np.testing.assert_array_equal(res.E_diss, np.zeros_like(res.E_diss))

    # Metadata propagated
    assert res.backend_name == 'fdbm'
    assert res.fft_path == 'GPU'
    assert res.amp_align is True  # osc_dir along z
    # tip_disp auto-filled with zeros
    for k in ('dx', 'dy', 'dz'):
        assert res.tip_disp[k].shape == (nx, ny, nz_scan)


def test_shared_postprocess_ediss_with_backward_stroke():
    """When FEs_bwd is provided, E_diss comes from compute_dissipation (nonzero
    where forward/backward forces differ)."""
    spec = _make_scan_spec()
    nx, ny, nz_scan = len(spec.scan_xs), len(spec.scan_ys), len(spec.h_scan)
    FEs_fwd = np.zeros((nx, ny, nz_scan, 4), dtype=np.float32)
    FEs_bwd = np.zeros((nx, ny, nz_scan, 4), dtype=np.float32)
    # Make Fz differ: fwd Fz = -z, bwd Fz = -2z → hysteresis → E_diss > 0
    FEs_fwd[..., 2] = -spec.h_scan[np.newaxis, np.newaxis, :]
    FEs_bwd[..., 2] = -2.0 * spec.h_scan[np.newaxis, np.newaxis, :]

    res = shared_postprocess(FEs_fwd, spec, FEs_bwd=FEs_bwd, backend_name='morse', fft_path='none')
    from spammm.SPM import AFM as afm
    osc_n = np.asarray(spec.osc_dir, dtype=np.float64)
    osc_n = osc_n / np.linalg.norm(osc_n)
    E_diss_ref = afm.compute_dissipation(FEs_fwd, FEs_bwd, spec.h_scan, spec.h_df,
                                         float(spec.amplitude), osc_dir=osc_n)
    np.testing.assert_allclose(res.E_diss, E_diss_ref, rtol=1e-6, atol=1e-6)
    assert res.backend_name == 'morse'
    assert res.fft_path == 'none'
    assert np.all(res.E_diss >= 0.0)


def test_shared_postprocess_lateral_osc_amp_align_false():
    """Pure lateral oscillation (n_z=0) → amp_align False."""
    h_df, h_Fz, h_scan = afm_df_height_stacks(3.7, 4.7, 0.1, amp=1.0, amp_align=True)
    spec = ScanSpec(
        scan_xs=np.arange(-1.0, 1.0, 0.1, dtype=np.float32),
        scan_ys=np.arange(-1.0, 1.0, 0.1, dtype=np.float32),
        h_df=h_df, h_Fz=h_Fz, h_scan=h_scan,
        amplitude=1.0, osc_dir=(1.0, 0.0, 0.0),
        K_LAT=0.0312, K_RAD=20.0, bond_length=3.0, scan_margin=2.0,
    )
    nx, ny, nz_scan = len(spec.scan_xs), len(spec.scan_ys), len(spec.h_scan)
    FEs = np.zeros((nx, ny, nz_scan, 4), dtype=np.float32)
    FEs[..., 2] = -spec.h_scan[np.newaxis, np.newaxis, :]
    res = shared_postprocess(FEs, spec)
    assert res.amp_align is False
