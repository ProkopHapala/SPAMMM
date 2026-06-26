// ------------------------------------------------------------------
// local_jacobi_blocks_parallel — refactored
//
// Parallel Jacobi eigendecomposition of small symmetric matrices
// in local memory. One workgroup per matrix (block). All threads
// collaborate on each Givens rotation: each thread updates its
// assigned rows of A and V.
//
// --- Optimizations vs original ---
//
//  (1) Inline (c,s) computation: every thread reads A[p*p], A[q*q],
//      A[p*q] from local memory and computes the Givens angle
//      independently. Eliminates the thread-0-broadcast barrier
//      (saves 1 barrier / rotation). Cost: ~10 redundant flops/thread,
//      negligible vs barrier latency (~100s of cycles on most GPUs).
//
//  (2) Fused 2x2 diagonal update: the thread that processes row k==p
//      in the row loop also updates A[p*p], A[q*q], A[p*q], A[q*p]
//      inline. The off-diagonal row loop skips k==p and k==q for A
//      updates, so the 2x2 block is unmodified when the k==p thread
//      reaches it. Eliminates the separate diagonal-update barrier
//      (saves 1 barrier / rotation).
//      Race-safety: off-diagonal writes touch A[k*m+p], A[k*m+q],
//      A[p*m+k], A[q*m+k] for k!=p,q — disjoint from the 2x2 block
//      indices A[p*m+p], A[q*m+q], A[p*m+q], A[q*m+p].
//
//  (3) Skip trivial rotations: when |A[p*q]| < tol, all threads
//      skip the row update (c=1, s=0 is a no-op). Saves compute
//      for already-converged off-diagonal elements. Barrier is
//      still needed (next rotation may read values from prior
//      rotations), but no writes → barrier is effectively free.
//
//  Net: 1 barrier per rotation (down from 3).
//  For m=64: 2016 barriers/sweep (down from 6048).
//
// --- Physics & Numerics ---
//
// Jacobi method: diagonalize symmetric A by applying sequence of
// Givens rotations G(p,q,theta) that zero out A[p*q]. Each rotation
// is an orthogonal similarity transform A' = G^T A G, preserving
// eigenvalues. Accumulating rotations into V gives eigenvectors:
//   A_final = diag(lambda_i),  V_final = G_1 G_2 ... G_k
//
// Rotation angle (Golub & Van Loan §8.4, numerically stable):
//   tau = (A[q*q] - A[p*p]) / (2 A[p*q])
//   t   = sign(tau) / (|tau| + sqrt(1 + tau^2))
//   c   = 1 / sqrt(1 + t^2),   s = t * c
// The formula avoids catastrophic cancellation when tau is large.
//
// Convergence: Frobenius norm of off-diagonal strictly decreases
// with each rotation (Jacobi's theorem). After m*(m-1)/2 rotations
// per sweep and ~6-10 sweeps, off-diagonal is below tolerance.
// Convergence check: ||off(A)||_F / ||off(A_0)||_F < tol.
//
// --- Performance: Local Memory & Bank Conflicts ---
//
// A and V are m x m row-major in local memory.
// Access pattern: thread lid reads A[k*m+p] and A[k*m+q] for its
// assigned rows k. For consecutive lid (consecutive k), addresses
// are stride-m apart.
//
// On NVIDIA (32 banks, 4B wide), bank = (k*m + p) % 32.
// For m=64 (divisible by 32): bank = p % 32 for ALL k → all threads
// hit the SAME bank → 32-way bank conflict. Very bad.
// For m not divisible by 32 (e.g. m=33): bank = (33k+p)%32, banks
// differ by k → no conflict.
//
// Mitigation (NOT implemented, backward compat): pad rows to stride
// m+1. Then bank = (k*(m+1)+p) % 32. For m=64: bank = (k+p)%32,
// which distributes across banks. Requires host to allocate
// m*(m+1) local memory and adjust stride everywhere.
//
// Workgroup size: optimal when lsz ≈ m (1 row/thread, no inner
// loop). For m=64, lsz=64 is ideal. Larger lsz wastes threads;
// smaller lsz requires inner k-loop iterations.
//
// --- Why not Brent-Luk parallel pairs? ---
//
// Brent-Luk round-robin schedules m/2 disjoint (p,q) pairs per
// round, potentially executing them in parallel. However, with
// row-major layout, disjoint pairs still share matrix rows: pair
// (p,q) writes A[p][k] for all k, including k=p' where (p',q') is
// another pair in the same round. Pair (p',q') reads A[p'][p] for
// its rotation. These are the same memory location → read-write
// race. Resolving requires per-pair synchronization or a different
// data layout (e.g. column-partitioned), adding complexity that
// outweighs the benefit for small m.
// ------------------------------------------------------------------

#ifndef JACOBI_MAX_M
#define JACOBI_MAX_M 64
#endif

__kernel void local_jacobi_blocks_parallel(
    const int m,
    const int max_sweeps,
    const float tol,
    __global const float* blocks,
    __global float* eigvals,
    __global float* eigvecs,
    __local float* A,
    __local float* V,
    __local float* scratch
) {
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsz = get_local_size(0);
    const int n2 = m * m;
    __global const float* gA = blocks + gid * n2;
    __global float* gV = eigvecs + gid * n2;

    // --- Load A from global, init V = I ---
    // Coalesced global read: consecutive lid load consecutive elements.
    // Local memory write: A[i] and V[i] for i = lid, lid+lsz, ...
    for (int i = lid; i < n2; i += lsz) {
        A[i] = gA[i];
        int r = i / m;
        int c = i - r * m;
        V[i] = (r == c) ? 1.0f : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Initial off-diagonal Frobenius norm ---
    // ||off(A)||_F = sqrt(sum_{i!=j} A[i][j]^2). Used as reference
    // for relative convergence check. Tree reduction in scratch[].
    // Assumes lsz is power of 2.
    float off_part = 0.0f;
    for (int i = lid; i < n2; i += lsz) {
        int r = i / m;
        int c = i - r * m;
        if (r != c) off_part += A[i] * A[i];
    }
    scratch[lid] = off_part;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = lsz >> 1; off > 0; off >>= 1) {
        if (lid < off) scratch[lid] += scratch[lid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float off0 = sqrt(fmax(scratch[0], tol * tol));
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Main sweep loop ---
    // Each sweep visits all m*(m-1)/2 upper-triangular (p,q) pairs
    // in lexicographic order. ~6-10 sweeps typically suffice.
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        for (int p = 0; p < m; ++p) {
            for (int q = p + 1; q < m; ++q) {

                // --- Compute Givens rotation (c,s) inline ---
                // All threads read the same 3 local-memory elements
                // and compute c,s independently. No broadcast needed.
                float apq = A[p * m + q];
                float c, s;
                if (fabs(apq) < tol) {
                    c = 1.0f;
                    s = 0.0f;
                } else {
                    float app = A[p * m + p];
                    float aqq = A[q * m + q];
                    float tau = (aqq - app) / (2.0f * apq);
                    float t = (tau >= 0.0f)
                        ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                        : -1.0f / (-tau + sqrt(1.0f + tau * tau));
                    c = 1.0f / sqrt(1.0f + t * t);
                    s = t * c;
                }

                // --- Apply rotation to A and V ---
                // Skip when |apq| < tol (c=1, s=0 → no-op).
                // Row loop: each thread handles rows k = lid, lid+lsz, ...
                //
                // Off-diagonal A update (k != p, q):
                //   A[k][p]' = c*A[k][p] - s*A[k][q]
                //   A[k][q]' = s*A[k][p] + c*A[k][q]
                //   Symmetric: A[p][k]' = A[k][p]', A[q][k]' = A[k][q]'
                //
                // V update (all k, including p and q):
                //   V[k][p]' = c*V[k][p] - s*V[k][q]
                //   V[k][q]' = s*V[k][p] + c*V[k][q]
                //
                // Fused 2x2 diagonal update (k == p only):
                //   A[p][p]' = c^2*A[p][p] - 2cs*A[p][q] + s^2*A[q][q]
                //   A[q][q]' = s^2*A[p][p] + 2cs*A[p][q] + c^2*A[q][q]
                //   A[p][q] = A[q][p] = 0
                // Uses `apq` saved before the loop (A[p*q] is not modified
                // by off-diagonal updates since k!=p,q skips it).
                if (fabs(apq) >= tol) {
                    for (int k = lid; k < m; k += lsz) {
                        if (k != p && k != q) {
                            float akp = A[k * m + p];
                            float akq = A[k * m + q];
                            float npv = c * akp - s * akq;
                            float nqv = s * akp + c * akq;
                            A[k * m + p] = npv;
                            A[p * m + k] = npv;
                            A[k * m + q] = nqv;
                            A[q * m + k] = nqv;
                        }
                        float vkp = V[k * m + p];
                        float vkq = V[k * m + q];
                        V[k * m + p] = c * vkp - s * vkq;
                        V[k * m + q] = s * vkp + c * vkq;

                        if (k == p) {
                            float app = A[p * m + p];
                            float aqq = A[q * m + q];
                            A[p * m + p] = c * c * app - 2.0f * c * s * apq + s * s * aqq;
                            A[q * m + q] = s * s * app + 2.0f * c * s * apq + c * c * aqq;
                            A[p * m + q] = 0.0f;
                            A[q * m + p] = 0.0f;
                        }
                    }
                }
                // Single barrier per rotation: ensures all row + diagonal
                // updates are visible before the next rotation reads A.
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        // --- Convergence check after sweep ---
        // Recompute ||off(A)||_F and compare to initial norm.
        off_part = 0.0f;
        for (int i = lid; i < n2; i += lsz) {
            int r = i / m;
            int c = i - r * m;
            if (r != c) off_part += A[i] * A[i];
        }
        scratch[lid] = off_part;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int off = lsz >> 1; off > 0; off >>= 1) {
            if (lid < off) scratch[lid] += scratch[lid + off];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (sqrt(scratch[0]) / off0 < tol) break;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Write eigenvalues and eigenvectors to global memory ---
    for (int i = lid; i < m; i += lsz) eigvals[gid * m + i] = A[i * m + i];
    for (int i = lid; i < n2; i += lsz) gV[i] = V[i];
}
