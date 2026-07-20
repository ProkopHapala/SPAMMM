"""
KrigingGridFF.py — DFT AFM z-scan → regular GridFF for AFMulator.

Essence: Load Mithun-style points + z-scans; Kriging/RBF interpolate E,Fx,Fy; Fz via Δz;
  emit F_total (nx,ny,nz,4)=(Fx,Fy,Fz,E) for AFMulator.setup_fdbm_grid.
Design: Port of ppafm tests/Interpolation/interp_zscan_to_grid*.py; no ppafm GridUtils.
  Units: kcal/mol → eV (KCAL_TO_EV). Default z0=1.6, dz=0.1 (Mithun).
Open issues / caveats:
  - Subtract large-z reference (e.g. z≈20 Å) before absolute FDBM fits — see topic doc.
  - points_clean already has outer `grid` sites; augmenter not required for Mithun set.
  - Data via symlink data/mithun_afm_scans (do not copy).
  - Science map: doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from .InterpolatorKriging import InterpolatorKriging
from .InterpolatorRBF import InterpolatorRBF

KCAL_TO_EV = 0.043364115

# Repo-relative defaults (symlinks under data/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
MITHUN_SCANS = _REPO_ROOT / "data" / "mithun_afm_scans"
MITHUN_SCANS_FLAT = _REPO_ROOT / "data" / "mithun_afm_scans_flat"
MITHUN_FUKUI = _REPO_ROOT / "data" / "mithun_afm_tip_fukui"


def load_clean_points(fname):
    """Load points: header 'type x y' or 'index type x y'. Returns types, pts (N,2)."""
    types, xs, ys = [], [], []
    with open(fname, 'r') as f:
        header = f.readline().strip().split()
        if header and header[0] == 'type':
            col_type, col_x, col_y = 0, 1, 2
        elif header and header[0] == 'index' and len(header) >= 4:
            col_type, col_x, col_y = 1, 2, 3
        else:
            f.seek(0)
            col_type, col_x, col_y = 0, 1, 2
        for line in f:
            parts = line.strip().split()
            if not parts or len(parts) <= max(col_type, col_x, col_y):
                continue
            types.append(parts[col_type])
            xs.append(float(parts[col_x]))
            ys.append(float(parts[col_y]))
    return types, np.stack([xs, ys], axis=1)


def load_point_info(fname):
    """Load raw *_point_info.txt: 'index atom[x y][]'."""
    types, xs, ys = [], [], []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_end = line.find(' ')
            if idx_end == -1:
                continue
            atom_part = line[idx_end + 1:]
            atom_start = atom_part.find('[')
            if atom_start == -1:
                continue
            atom_type = atom_part[:atom_start]
            coord_end = atom_part.find(']', atom_start + 1)
            if coord_end == -1:
                continue
            coords = atom_part[atom_start + 1:coord_end].strip().split()
            if len(coords) < 2:
                continue
            types.append(atom_type)
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
    return types, np.stack([xs, ys], axis=1)


def load_zscan(fname):
    """Load 'pNNN zMM value' → values (n_points, n_z). Fail loud on missing entries."""
    data = {}
    with open(fname, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p_label, z_label, val_str = parts
            if not p_label.startswith('p') or not z_label.startswith('z'):
                continue
            try:
                i_point = int(p_label[1:])
                i_z = int(z_label[1:])
                val = float(val_str)
            except ValueError:
                continue
            data.setdefault(i_point, {})[i_z] = val
    if not data:
        raise RuntimeError(f"No z-scan data parsed from {fname}")
    point_indices = sorted(data.keys())
    z_indices = sorted(next(iter(data.values())).keys())
    n_points, n_z = len(point_indices), len(z_indices)
    values = np.zeros((n_points, n_z), dtype=float)
    for ip, p in enumerate(point_indices):
        row = data[p]
        for iz, z in enumerate(z_indices):
            try:
                values[ip, iz] = row[z]
            except KeyError:
                raise RuntimeError(f"Missing value for point p{p:03d} z{z:02d} in {fname}")
    return values


def subtract_z_reference(zscan_vals, iz_ref=-1):
    """Subtract per-point reference z-slice (default last = largest z in file)."""
    return zscan_vals - zscan_vals[:, iz_ref][:, None]


def build_grid(points_xy, nx, ny, nz, z0, dz, pad=0.1, dx=None, dy=None):
    """Build XY mesh + z stack. If dx/dy set, use exact arange spacing (not linspace)."""
    xmin, xmax = points_xy[:, 0].min(), points_xy[:, 0].max()
    ymin, ymax = points_xy[:, 1].min(), points_xy[:, 1].max()
    Lx, Ly = xmax - xmin, ymax - ymin
    xmin -= pad * Lx
    xmax += pad * Lx
    ymin -= pad * Ly
    ymax += pad * Ly
    if (dx is not None) and (dy is not None):
        xs = np.arange(xmin, xmax + 0.5 * dx, dx, dtype=float)
        ys = np.arange(ymin, ymax + 0.5 * dy, dy, dtype=float)
        if len(xs) < 2:
            xs = np.array([xmin, xmin + dx], dtype=float)
        if len(ys) < 2:
            ys = np.array([ymin, ymin + dy], dtype=float)
    else:
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
    zs = z0 + dz * np.arange(nz, dtype=float)
    Xs, Ys = np.meshgrid(xs, ys)
    grid_points = np.stack([Xs.ravel(), Ys.ravel()], axis=1)
    return xs, ys, zs, grid_points


def make_interpolator(kind, points_xy, R_basis, kriging_nugget=0.0, kriging_global_eval=False,
                      rbf_normalized=False, rbf_eps_norm=0.0, verbose=False):
    if kind == 'rbf':
        return InterpolatorRBF(points_xy, R_basis, normalized=rbf_normalized, eps_norm=rbf_eps_norm, verbose=verbose)
    if kind == 'kriging':
        return InterpolatorKriging(points_xy, R_basis, nugget=kriging_nugget, global_eval=bool(kriging_global_eval), verbose=verbose)
    raise ValueError(f"Unknown interpolator kind '{kind}', expected 'rbf' or 'kriging'")


def auto_support_radii(points_xy, k=6, scale=1.3, rmin=0.5, rmax=1e9, percentile=None):
    points_xy = np.asarray(points_xy, dtype=float)
    n = points_xy.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    k = int(k)
    if k < 1:
        raise ValueError(f"auto_support_radii: k must be >=1, got {k}")
    kk = min(n - 1, k)
    if kk < 1:
        return np.full(n, max(rmin, 1.0), dtype=float)
    tree = KDTree(points_xy)
    dists, _ = tree.query(points_xy, k=kk + 1)
    dk = dists[:, kk]
    if percentile is not None:
        Ri = np.full(n, float(np.percentile(dk, float(percentile)) * float(scale)), dtype=float)
    else:
        Ri = dk * float(scale)
    return np.clip(Ri, float(rmin), float(rmax))


def interpolate_volume(points_xy, zscan_vals, nx, ny, nz, z0, dz, R_basis, kind='kriging',
                       dx=None, dy=None, kriging_nugget=0.0, kriging_global_eval=False,
                       rbf_normalized=False, rbf_eps_norm=0.0, verbose=False):
    """Return xs, ys, zs, vol[nz,ny,nx] in input energy units (kcal/mol for Mithun)."""
    xs, ys, zs, grid_points = build_grid(points_xy, nx, ny, nz, z0, dz, dx=dx, dy=dy)
    nx_eff, ny_eff = len(xs), len(ys)
    interp = make_interpolator(kind, points_xy, R_basis, kriging_nugget=kriging_nugget,
                               kriging_global_eval=kriging_global_eval, rbf_normalized=rbf_normalized,
                               rbf_eps_norm=rbf_eps_norm, verbose=verbose)
    vol = np.zeros((nz, ny_eff, nx_eff), dtype=float)
    for iz in range(nz):
        if not interp.update_weights(zscan_vals[:, iz]):
            raise RuntimeError(f"Failed to update weights for z-index {iz}")
        vals_xy = interp.evaluate(grid_points)
        if vals_xy is None:
            raise RuntimeError(f"Interpolation failed at z-index {iz}")
        vol[iz, :, :] = vals_xy.reshape((ny_eff, nx_eff))
    return xs, ys, zs, vol


def interpolate_volume_and_forces(points_xy, zscan_vals, nx, ny, nz, z0, dz, R_basis, kind='kriging',
                                  dx=None, dy=None, kriging_nugget=0.0, kriging_global_eval=False,
                                  rbf_normalized=False, rbf_eps_norm=0.0, to_eV=True, verbose=False):
    """
    Interpolate E + forces.

    Returns
    -------
    xs, ys, zs : 1D grids
    F_afm : (nx, ny, nz, 4) float64  — (Fx,Fy,Fz,E) for AFMulator.setup_fdbm_grid
    vol_znyx : (nz, ny, nx) energy (same units as F_afm[...,3])
    """
    xs, ys, zs, grid_points = build_grid(points_xy, nx, ny, nz, z0, dz, dx=dx, dy=dy)
    nx_eff, ny_eff = len(xs), len(ys)
    interp = make_interpolator(kind, points_xy, R_basis, kriging_nugget=kriging_nugget,
                               kriging_global_eval=kriging_global_eval, rbf_normalized=rbf_normalized,
                               rbf_eps_norm=rbf_eps_norm, verbose=verbose)
    vol = np.zeros((nz, ny_eff, nx_eff), dtype=float)
    vol_Fxy = np.zeros((nz, ny_eff, nx_eff, 2), dtype=float)
    for iz in range(nz):
        if not interp.update_weights(zscan_vals[:, iz]):
            raise RuntimeError(f"Failed weights at z-index {iz}")
        vals = interp.evaluate(grid_points)
        if vals is None:
            raise RuntimeError(f"Interp failed at z-index {iz}")
        vol[iz, :, :] = vals.reshape((ny_eff, nx_eff))
        grads = interp.evaluate_gradient(grid_points)
        if grads is None:
            raise RuntimeError(f"Grad failed at z-index {iz}")
        vol_Fxy[iz, :, :, 0] = -grads[:, 0].reshape((ny_eff, nx_eff))
        vol_Fxy[iz, :, :, 1] = -grads[:, 1].reshape((ny_eff, nx_eff))
    vol_Fz = np.zeros((nz, ny_eff, nx_eff), dtype=float)
    if nz >= 2:
        for iz in range(1, nz - 1):
            vol_Fz[iz, :, :] = -(vol[iz + 1, :, :] - vol[iz - 1, :, :]) / (2.0 * dz)
        vol_Fz[0, :, :] = -(vol[1, :, :] - vol[0, :, :]) / dz
        vol_Fz[nz - 1, :, :] = -(vol[nz - 1, :, :] - vol[nz - 2, :, :]) / dz
    scale = KCAL_TO_EV if to_eV else 1.0
    # ppafm storage was (nz,ny,nx,4); AFMulator wants (nx,ny,nz,4)
    F_afm = np.zeros((nx_eff, ny_eff, nz, 4), dtype=float)
    for iz in range(nz):
        F_afm[:, :, iz, 0] = vol_Fxy[iz, :, :, 0].T * scale
        F_afm[:, :, iz, 1] = vol_Fxy[iz, :, :, 1].T * scale
        F_afm[:, :, iz, 2] = vol_Fz[iz, :, :].T * scale
        F_afm[:, :, iz, 3] = vol[iz, :, :].T * scale
    vol_out = vol * scale
    return xs, ys, zs, np.ascontiguousarray(F_afm), vol_out


def grid_origin_step(xs, ys, zs, atol=1e-6):
    """Uniform-grid origin + isotropic step for AFMulator.setup_fdbm_grid."""
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    dz = float(zs[1] - zs[0]) if len(zs) > 1 else 1.0
    if abs(dx - dy) > atol or abs(dx - dz) > atol:
        raise ValueError(
            f"AFMulator.setup_fdbm_grid needs isotropic step; got dx={dx} dy={dy} dz={dz}. "
            f"Call interpolate with dx=dy=dz (e.g. 0.1)."
        )
    origin = np.array([xs[0], ys[0], zs[0]], dtype=np.float64)
    return origin, dx


def demo_paths(endgroup='HHO-h-p_1', tip='H2O_O'):
    """Default Mithun demo file paths (via data/ symlink)."""
    root = MITHUN_SCANS
    points = root / 'points_clean' / f'{endgroup}_points_clean.txt'
    zscan = root / 'results' / f'{endgroup}-{tip}.dat'
    return points, zscan


def interpolate_mithun_pair(endgroup='HHO-h-p_1', tip='H2O_O', kind='kriging', R_basis=8.0,
                            dx=0.1, dy=0.1, dz=0.1, z0=1.6, nz=None, to_eV=True,
                            kriging_nugget=0.0, kriging_global_eval=False, verbose=False):
    """Convenience: load linked Mithun pair → AFMulator GridFF (isotropic dx=dy=dz)."""
    points_path, zscan_path = demo_paths(endgroup, tip)
    if not points_path.is_file():
        raise FileNotFoundError(f"Missing points (check symlink data/mithun_afm_scans): {points_path}")
    if not zscan_path.is_file():
        raise FileNotFoundError(f"Missing z-scan: {zscan_path}")
    _, points_xy = load_clean_points(points_path)
    zscan_vals = load_zscan(zscan_path)
    if nz is None:
        nz = zscan_vals.shape[1]
    else:
        nz = min(int(nz), zscan_vals.shape[1])
    xs, ys, zs, F_afm, vol = interpolate_volume_and_forces(
        points_xy, zscan_vals[:, :nz], nx=50, ny=50, nz=nz, z0=z0, dz=dz, R_basis=R_basis,
        kind=kind, dx=dx, dy=dy, kriging_nugget=kriging_nugget,
        kriging_global_eval=kriging_global_eval, to_eV=to_eV, verbose=verbose,
    )
    origin, step = grid_origin_step(xs, ys, zs)
    return dict(xs=xs, ys=ys, zs=zs, F_afm=F_afm, vol=vol, origin=origin, step=step,
                points_xy=points_xy, zscan_vals=zscan_vals, points_path=str(points_path), zscan_path=str(zscan_path))
