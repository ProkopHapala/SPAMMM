import pytest, numpy as np
from spammm.utils.Lingebra_ocl import LingebraOCL
from tests.helpers.parity import rmse, max_err


@pytest.fixture(scope='module')
def lingebra():
    return LingebraOCL(bPrint=False)


def _sort_eigh(w, V):
    idx = np.argsort(w)
    return w[idx], V[:, idx]


def _align_signs(ref_V, test_V):
    aligned = test_V.copy()
    for j in range(ref_V.shape[1]):
        if np.dot(ref_V[:, j], test_V[:, j]) < 0:
            aligned[:, j] = -test_V[:, j]
    return aligned


def _check_eigh_parity(A_sym, w_gpu, V_gpu, m, i):
    w_ref, V_ref = np.linalg.eigh(A_sym[i].astype(np.float64))
    idx = np.argsort(w_gpu[i])
    w_s = w_gpu[i][idx]
    V_s = V_gpu[i][:, idx]
    V_a = _align_signs(V_ref, V_s)
    wr = rmse(w_ref, w_s)
    wm = max_err(w_ref, w_s)
    vr = rmse(V_ref, V_a)
    vm = max_err(V_ref, V_a)
    print(f"  m={m} batch={i}: w RMSE={wr:.2e} Max={wm:.2e} | V RMSE={vr:.2e} Max={vm:.2e}")
    assert wr < 1e-3, f"m={m} batch={i}: eigenvalue RMSE={wr:.2e}"
    assert wm < 1e-2, f"m={m} batch={i}: eigenvalue MaxErr={wm:.2e}"
    assert vr < 1e-3, f"m={m} batch={i}: eigenvector RMSE={vr:.2e}"
    assert vm < 1e-2, f"m={m} batch={i}: eigenvector MaxErr={vm:.2e}"


@pytest.mark.gpu
@pytest.mark.parametrize('m', [4, 8, 16, 32])
def test_eigh_random_symmetric(lingebra, m):
    """Jacobi eigendecomposition vs numpy.linalg.eigh on random symmetric matrices."""
    np.random.seed(42)
    batch = 8
    B = np.random.randn(batch, m, m).astype(np.float32)
    A_sym = ((B + np.transpose(B, (0, 2, 1))) * 0.5).astype(np.float32)
    w, V = lingebra.jacobi_eigh(A_sym)
    for i in range(batch):
        _check_eigh_parity(A_sym, w, V, m, i)


@pytest.mark.gpu
def test_eigh_diagonal(lingebra):
    """Already-diagonal matrix: eigenvalues = diagonal, eigenvectors = I."""
    m = 8
    batch = 4
    np.random.seed(123)
    diag = np.random.randn(batch, m).astype(np.float32)
    A = np.zeros((batch, m, m), dtype=np.float32)
    for i in range(batch):
        for j in range(m):
            A[i, j, j] = diag[i, j]
    w, V = lingebra.jacobi_eigh(A, max_sweeps=10)
    for i in range(batch):
        w_ref = np.sort(diag[i].astype(np.float64))
        assert rmse(np.sort(w[i]), w_ref) < 1e-4, f"diagonal batch={i}: RMSE too large"


@pytest.mark.gpu
def test_eigh_identity(lingebra):
    """Identity matrix: all eigenvalues = 1, eigenvectors = I."""
    m = 8
    batch = 2
    A = np.tile(np.eye(m, dtype=np.float32), (batch, 1, 1))
    w, V = lingebra.jacobi_eigh(A, max_sweeps=10)
    for i in range(batch):
        assert rmse(w[i], np.ones(m)) < 1e-4
        assert rmse(V[i], np.eye(m)) < 1e-4
