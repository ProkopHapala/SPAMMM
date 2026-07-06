"""esp_grid.py — Coulomb ESP on 2D slices for charge-driven scan visualization.

Same physics as QEq panel (`KE/r`, eV): precompute **`[nframes, ny, nx]`** stacks so the GUI
slider can update heatmaps without recomputing pairwise sums. Grid extent follows molecule
bounding box (`plotUtils.compute_grid_extent`).

- **Input:** Mulliken charges from DFTB (`coordinate_scan` / `ScanDataset.charges`).
- **Docs:** `doc/Topics/ReactionCoordinateScan.md`, blit caveats in `doc/Takeways.md`
"""
import numpy as np

from spammm.forcefields.QEq import KE
from spammm.plotUtils import compute_grid_extent, make_2d_grid

__all__ = ['KE', 'coulomb_esp_frame', 'compute_esp_stack']


def coulomb_esp_frame(apos, charges, points):
    """ESP at grid points. apos (natoms,3), charges (natoms,), points (npts,3) → (npts,)."""
    apos = np.asarray(apos, dtype=np.float64)
    q = np.asarray(charges, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    r = pts[:, None, :] - apos[None, :, :]
    d = np.maximum(np.linalg.norm(r, axis=2), 1e-3)
    return KE * (q[None, :] / d).sum(axis=1)


def compute_esp_stack(apos_stack, charges_stack, z_height, n=128, padding_factor=0.15):
    """Precompute ESP maps for all frames. Returns (stack, extent, nx, ny, z_abs).

    stack: [nframes, ny, nx] eV
    """
    apos_stack = np.asarray(apos_stack, dtype=np.float64)
    charges_stack = np.asarray(charges_stack, dtype=np.float64)
    nframes = apos_stack.shape[0]
    flat = apos_stack.reshape(-1, 3)
    grid_origin, size_xy, center_z = compute_grid_extent(flat, padding_factor=padding_factor)
    points, extent, nx, ny = make_2d_grid(grid_origin, size_xy, center_z, z_height, n=n)
    z_abs = center_z + float(z_height)
    stack = np.empty((nframes, ny, nx), dtype=np.float64)
    for i in range(nframes):
        stack[i] = coulomb_esp_frame(apos_stack[i], charges_stack[i], points).reshape(ny, nx)
    return stack, extent, nx, ny, z_abs
