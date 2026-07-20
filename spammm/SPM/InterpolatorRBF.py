"""
InterpolatorRBF.py — Wendland C2 RBF interpolator (AFM z-scans).

Essence: Build Phi, solve weights per z-slice, evaluate E and ∇E at query XY.
Design: Port of ppafm InterpolatorRBF; optional normalized mode; per-point R_i.
Open issues / caveats:
  - Prefer Kriging for AFM visuals (ppafm tutorial). verbose=False by default.
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import solve

from .interpy import wendland_c2, pairwise_distances, wendland_c2_varR, wendland_c2_deriv, wendland_c2_deriv_varR


class InterpolatorRBF:
    def __init__(self, data_points, R_basis, C_peak=1.0, normalized=False, eps_norm=0.0, verbose=False):
        self.data_points = np.asarray(data_points, dtype=float)
        self.ndata = self.data_points.shape[0]
        self.R_basis = R_basis
        self.R_i = None
        self.R_max = None
        self.verbose = bool(verbose)
        if np.ndim(R_basis) == 0:
            self.R_basis = float(R_basis)
        else:
            self.R_i = np.asarray(R_basis, dtype=float).reshape(-1)
            if self.R_i.shape[0] != self.ndata:
                raise ValueError(f"InterpolatorRBF: per-point R_basis must have shape (N,), got {self.R_i.shape} for N={self.ndata}")
            self.R_max = float(np.max(self.R_i)) if self.ndata > 0 else 0.0
        self.C_peak = float(C_peak)
        self.normalized = bool(normalized)
        self.eps_norm = float(eps_norm)
        if self.ndata == 0:
            if self.verbose:
                print("WARNING: InterpolatorRBF initialized with no data points.")
            self.phi_matrix = None
            return
        if self.verbose:
            if self.R_i is None:
                print(f"InterpolatorRBF.init(): Building {self.ndata}x{self.ndata} Phi (R_basis={self.R_basis})")
            else:
                print(f"InterpolatorRBF.init(): Building {self.ndata}x{self.ndata} Phi (R_i: min={self.R_i.min():.3g} max={self.R_i.max():.3g})")
        distances = pairwise_distances(self.data_points, self.data_points)
        if self.R_i is None:
            self.phi_matrix = wendland_c2(distances, self.R_basis, C=self.C_peak)
        else:
            R_pair = np.minimum(self.R_i[:, None], self.R_i[None, :])
            self.phi_matrix = wendland_c2_varR(distances, R_pair, C=self.C_peak)
        self.weights = None

    def update_weights(self, data_vals):
        if self.phi_matrix is None or self.ndata == 0:
            print("ERROR in InterpolatorRBF.update_weights(): setup failed or no data.")
            self.weights = None
            return False
        z = np.asarray(data_vals, dtype=float)
        if z.shape[0] != self.ndata:
            print(f"ERROR in InterpolatorRBF.update_weights(): data_vals size ({z.shape[0]}) != N ({self.ndata}).")
            self.weights = None
            return False
        if self.verbose:
            print(f"InterpolatorRBF.update_weights(): Solving for {self.ndata} weights...")
        try:
            self.weights = solve(self.phi_matrix, z)
            return True
        except np.linalg.LinAlgError:
            print("ERROR in InterpolatorRBF.update_weights(): singular / ill-conditioned system.")
            self.weights = None
            return False

    def evaluate(self, query_points):
        if self.weights is None:
            print("ERROR in InterpolatorRBF.evaluate(): call update_weights first.")
            return None
        if self.ndata == 0:
            return np.zeros(query_points.shape[0], dtype=float)
        query_points = np.asarray(query_points, dtype=float)
        nqps = query_points.shape[0]
        if nqps == 0:
            return np.array([], dtype=float)
        if self.verbose:
            print(f"InterpolatorRBF.evaluate(): {nqps} points...")
        data_kdtree = KDTree(self.data_points)
        r = self.R_basis if self.R_i is None else self.R_max
        neighbor_indices_list = data_kdtree.query_ball_point(query_points, r=r)
        interpolated_values = np.zeros(nqps, dtype=float)
        for i in range(nqps):
            q = query_points[i]
            neighbors_q_indices = neighbor_indices_list[i]
            if not neighbors_q_indices:
                interpolated_values[i] = 0.0
                continue
            neighbor_pts = self.data_points[neighbors_q_indices, :]
            dists = np.linalg.norm(neighbor_pts - q, axis=1)
            if self.R_i is None:
                phi_vals = wendland_c2(dists, self.R_basis, C=self.C_peak)
                neighbor_weights = self.weights[neighbors_q_indices]
            else:
                Ri = self.R_i[neighbors_q_indices]
                mask = dists < Ri
                if not np.any(mask):
                    interpolated_values[i] = 0.0
                    continue
                phi_vals = wendland_c2_varR(dists[mask], Ri[mask], C=self.C_peak)
                neighbor_weights = self.weights[neighbors_q_indices][mask]
            base_val = np.sum(neighbor_weights * phi_vals)
            if self.normalized:
                interpolated_values[i] = base_val / (np.sum(phi_vals) + self.eps_norm)
            else:
                interpolated_values[i] = base_val
        return interpolated_values

    def evaluate_gradient(self, query_points):
        """∇E at query points (not force). F = -∇E."""
        if self.weights is None:
            print("ERROR in InterpolatorRBF.evaluate_gradient(): call update_weights first.")
            return None
        if self.ndata == 0:
            return np.zeros((query_points.shape[0], query_points.shape[1]), dtype=float)
        query_points = np.asarray(query_points, dtype=float)
        nqps = query_points.shape[0]
        D = query_points.shape[1]
        if nqps == 0:
            return np.array([], dtype=float).reshape(0, D)
        data_kdtree = KDTree(self.data_points)
        r = self.R_basis if self.R_i is None else self.R_max
        neighbor_indices_list = data_kdtree.query_ball_point(query_points, r=r)
        gradients = np.zeros((nqps, D), dtype=float)
        for i in range(nqps):
            q = query_points[i]
            neighbors_q_indices = neighbor_indices_list[i]
            if not neighbors_q_indices:
                continue
            neighbor_pts = self.data_points[neighbors_q_indices, :]
            neighbor_weights = self.weights[neighbors_q_indices]
            diffs = q - neighbor_pts
            dists = np.linalg.norm(diffs, axis=1)
            if self.R_i is None:
                deriv_vals = wendland_c2_deriv(dists, self.R_basis, C=self.C_peak)
            else:
                Ri = self.R_i[neighbors_q_indices]
                mask = dists < Ri
                if not np.any(mask):
                    continue
                deriv_vals = wendland_c2_deriv_varR(dists[mask], Ri[mask], C=self.C_peak)
                diffs = diffs[mask]
                neighbor_weights = neighbor_weights[mask]
                dists = dists[mask]
            with np.errstate(divide='ignore', invalid='ignore'):
                direction = diffs / dists[:, None]
            direction[np.isnan(direction)] = 0.0
            gradients[i] = np.sum(neighbor_weights[:, None] * deriv_vals[:, None] * direction, axis=0)
        return gradients
