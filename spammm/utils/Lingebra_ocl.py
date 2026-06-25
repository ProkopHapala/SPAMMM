"""
Lingebra_ocl.py — OpenCL-accelerated linear algebra for small dense matrices.

Purpose: Batched eigendecomposition of small symmetric matrices via parallel
Jacobi rotations in local memory. One workgroup per matrix, all threads
collaborate on each Givens rotation.

Key functionality:
  - jacobi_eigh() — batched symmetric eigendecomposition (eigenvalues + eigenvectors)

Role in SPAMMM: Reusable linear algebra primitive for GPU modules that need
small-matrix diagonalization (e.g. inertia tensor diagonalization, Hessian
eigenmodes, fragment Hamiltonians).

Design: Inherits OpenCLBase for device/queue/buffer management. Kernels live
in kernels/lingebra.cl.
"""

import os
import numpy as np
import pyopencl as cl
from .OpenCLBase import OpenCLBase

KERNEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kernels', 'lingebra.cl')


class LingebraOCL(OpenCLBase):
    """Batched small-matrix linear algebra on GPU via OpenCL."""

    def __init__(self, preferred_vendor='nvidia', device_index=0, bPrint=False):
        super().__init__(nloc=64, preferred_vendor=preferred_vendor, device_index=device_index, bPrint=bPrint)
        self.load_program(kernel_path=KERNEL_PATH, bPrint=bPrint)
        self._krnl_jacobi = cl.Kernel(self.prg, 'local_jacobi_blocks_parallel')

    def jacobi_eigh(self, A, max_sweeps=100, tol=1e-6):
        """Batched symmetric eigendecomposition via parallel Jacobi.

        Args:
            A: (batch, m, m) float32 symmetric matrices, row-major
            max_sweeps: max Jacobi sweeps
            tol: convergence tolerance (relative off-diagonal Frobenius norm)

        Returns:
            eigvals: (batch, m) float32, sorted ascending per matrix
            eigvecs: (batch, m, m) float32, columns are eigenvectors
        """
        A = np.ascontiguousarray(A, dtype=np.float32)
        if A.ndim != 3:
            raise ValueError(f"A must be (batch, m, m), got shape {A.shape}")
        batch, m, m2 = A.shape
        if m != m2:
            raise ValueError(f"Matrices must be square, got {m}x{m2}")
        n2 = m * m
        lsz = m  # 1 thread per row

        A_flat = A.reshape(batch, n2)
        eigvals = np.zeros((batch, m), dtype=np.float32)
        eigvecs = np.zeros((batch, n2), dtype=np.float32)

        mf = cl.mem_flags
        g_A = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_flat)
        g_eigvals = cl.Buffer(self.ctx, mf.WRITE_ONLY, batch * m * 4)
        g_eigvecs = cl.Buffer(self.ctx, mf.WRITE_ONLY, batch * n2 * 4)

        local_A = cl.LocalMemory(n2 * 4)
        local_V = cl.LocalMemory(n2 * 4)
        local_scratch = cl.LocalMemory(lsz * 4)

        self._krnl_jacobi(
            self.queue, (batch * lsz,), (lsz,),
            np.int32(m), np.int32(max_sweeps), np.float32(tol),
            g_A, g_eigvals, g_eigvecs,
            local_A, local_V, local_scratch,
        )
        self.queue.finish()

        cl.enqueue_copy(self.queue, eigvals, g_eigvals)
        cl.enqueue_copy(self.queue, eigvecs, g_eigvecs)
        self.queue.finish()

        return eigvals, eigvecs.reshape(batch, m, m)
