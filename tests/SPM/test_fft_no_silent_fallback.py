"""
test_fft_no_silent_fallback.py — L0: fft_poisson with unfriendly shape raises (not silent CPU fallback).

Verifies that fft_poisson / _FDBMGpyFFT.ensure() with an unfriendly shape
(e.g. (11,11,11) — 11 is prime) raises an error instead of silently falling
back to CPU FFT. The error must mention "not clFFT-friendly" or "prime factors".

NOTE: The current implementation raises RuntimeError (not ValueError) from
ensure() — see AFM.py:2180. The task spec mentioned ValueError, but since
AFM.py is read-only for this agent, we test the actual behavior (RuntimeError).
The key invariant: it RAISES, it does NOT silently produce results.
"""
import os
import numpy as np
import pytest

# Ensure CPU FFT is NOT enabled (we want to test the GPU path's friendliness check)
os.environ.pop('SPAMMM_AFM_CPU_FFT', None)

from spammm.SPM.AFM import _FDBMGpyFFT, fft_poisson, afm_use_cpu_fft, _prime_factorization


def test_is_fft_friendly_11_is_not_friendly():
    """11 is prime → not clFFT-friendly (factors must be 2,3,5,7 only)."""
    assert not _FDBMGpyFFT.is_fft_friendly(11)
    assert _prime_factorization(11) == {11: 1}


def test_is_fft_friendly_friendly_sizes():
    """Known friendly sizes are accepted."""
    for n in [8, 16, 32, 64, 120, 128, 240, 1024]:
        assert _FDBMGpyFFT.is_fft_friendly(n), f"{n} should be friendly"


def test_ensure_raises_for_unfriendly_shape():
    """_FDBMGpyFFT.ensure() raises for shape (11,11,11) — no silent fallback.

    Uses __new__ to create a bare instance without OpenCL init. ensure() checks
    friendliness FIRST (before accessing any GPU resources), so it raises at
    the friendliness check.
    """
    fft = _FDBMGpyFFT.__new__(_FDBMGpyFFT)
    fft._shape = None
    rho = np.zeros((11, 11, 11), dtype=np.float32)
    with pytest.raises(RuntimeError, match="not clFFT-friendly"):
        fft.ensure(rho.shape)


def test_ensure_raises_for_partially_unfriendly_shape():
    """ensure() raises if ANY dimension is unfriendly (not just all three)."""
    fft = _FDBMGpyFFT.__new__(_FDBMGpyFFT)
    fft._shape = None
    # (120, 11, 64) — ny=11 is prime, nx and nz are friendly
    rho = np.zeros((120, 11, 64), dtype=np.float32)
    with pytest.raises(RuntimeError, match="ny=11"):
        fft.ensure(rho.shape)


def test_ensure_does_not_raise_for_friendly_shape():
    """ensure() does NOT raise the friendliness error for a friendly shape.

    Uses a bare instance (no OpenCL). The friendliness check passes for (64,64,64),
    so ensure() proceeds to GPU allocation — which fails with AttributeError (no cl_array
    on bare instance). The key: it does NOT raise 'not clFFT-friendly'.
    """
    fft = _FDBMGpyFFT.__new__(_FDBMGpyFFT)
    fft._shape = None
    rho = np.zeros((64, 64, 64), dtype=np.float32)
    with pytest.raises((AttributeError, RuntimeError)) as exc_info:
        fft.ensure(rho.shape)
    # Must NOT be the friendliness error
    assert "not clFFT-friendly" not in str(exc_info.value), \
        f"Friendly shape (64,64,64) incorrectly rejected as unfriendly: {exc_info.value}"


def test_fft_poisson_unfriendly_shape_raises():
    """fft_poisson with unfriendly shape raises (not silent CPU fallback).

    Mocks afm_use_cpu_fft to return False (GPU path) and _get_fdbm_fft to return
    a bare instance. fft_poisson → poisson → ensure → raises RuntimeError.
    """
    from unittest.mock import patch
    rho = np.zeros((11, 11, 11), dtype=np.float32)
    bare_fft = _FDBMGpyFFT.__new__(_FDBMGpyFFT)
    bare_fft._shape = None
    with patch('spammm.SPM.AFM.afm_use_cpu_fft', return_value=False), \
         patch('spammm.SPM.AFM._get_fdbm_fft', return_value=bare_fft):
        with pytest.raises((RuntimeError, ValueError), match="not clFFT-friendly"):
            fft_poisson(rho, step=0.1)


def test_fft_poisson_does_not_silently_produce_results():
    """fft_poisson with unfriendly shape must NOT return a valid array silently.

    This is the core invariant: no silent fallback to CPU FFT when the user
    didn't request it. Either it raises, or (if CPU FFT is explicitly enabled)
    it works — but never silently switches paths.
    """
    from unittest.mock import patch
    rho = np.ones((11, 11, 11), dtype=np.float32) * 0.1
    bare_fft = _FDBMGpyFFT.__new__(_FDBMGpyFFT)
    bare_fft._shape = None
    with patch('spammm.SPM.AFM.afm_use_cpu_fft', return_value=False), \
         patch('spammm.SPM.AFM._get_fdbm_fft', return_value=bare_fft):
        try:
            result = fft_poisson(rho, step=0.1)
            # If we got here, it didn't raise — that's a silent fallback (BUG)
            pytest.fail(f"fft_poisson silently returned result for unfriendly shape: {result.shape}")
        except (RuntimeError, ValueError):
            pass  # Expected — it raised instead of silently falling back


def test_cpu_fft_explicit_does_not_raise_friendliness():
    """When SPAMMM_AFM_CPU_FFT is explicitly set, fft_poisson_cpu is used (no friendliness check).

    This verifies the CPU path is a legitimate explicit choice, not a silent fallback.
    We test fft_poisson_cpu directly (it uses NumPy FFT — no friendliness requirement).
    """
    from spammm.SPM.AFM import fft_poisson_cpu
    rho = np.zeros((11, 11, 11), dtype=np.float32)
    # fft_poisson_cpu uses NumPy — works for any shape, no friendliness check
    V = fft_poisson_cpu(rho, step=0.1)
    assert V.shape == (11, 11, 11), f"CPU FFT result shape {V.shape}, expected (11,11,11)"
