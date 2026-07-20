"""
InterpolatorKriging.py — Ordinary Kriging with Wendland C2 covariance (AFM z-scans).

Essence: Build [C 1; 1^T 0], solve weights per z-slice, evaluate E and ∇E at query XY.
Design: Port of ppafm InterpolatorKriging; per-point R_i via min(R_i,R_j); optional global_eval.
Open issues / caveats:
  - verbose=False by default (ppafm printed every slice).
  - F = -∇E; Fz from finite Δz is in KrigingGridFF, not here.
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import solve

from .interpy import compact_c2_covariance, pairwise_distances, wendland_c2_varR, wendland_c2_deriv, wendland_c2_deriv_varR


class InterpolatorKriging:
    def __init__(self, data_points, R_basis, nugget=0.0, global_eval=False, verbose=False):
        """
        data_points: (N, 2). R_basis: float or (N,) per-point radii.
        """
        self.data_points = np.asarray(data_points, dtype=float)
        self.ndata = self.data_points.shape[0]
        self.R_basis = R_basis
        self.nugget = float(nugget)
        self.global_eval = bool(global_eval)
        self.verbose = bool(verbose)
        self.R_i = None
        self.R_max = None
        if np.ndim(R_basis) == 0:
            self.R_basis = float(R_basis)
        else:
            self.R_i = np.asarray(R_basis, dtype=float).reshape(-1)
            if self.R_i.shape[0] != self.ndata:
                raise ValueError(f"InterpolatorKriging: per-point R_basis must have shape (N,), got {self.R_i.shape} for N={self.ndata}")
            self.R_max = float(np.max(self.R_i)) if self.ndata > 0 else 0.0
        if self.ndata == 0:
            if self.verbose:
                print("WARNING: InterpolatorKriging initialized with no data points.")
            self.kriging_matrix = None
            return
        if self.verbose:
            if self.R_i is None:
                print(f"InterpolatorKriging.init(): Building {self.ndata+1}x{self.ndata+1} matrix (R_basis={self.R_basis})")
            else:
                print(f"InterpolatorKriging.init(): Building {self.ndata+1}x{self.ndata+1} matrix (R_i: min={self.R_i.min():.3g} max={self.R_i.max():.3g})")

        distances = pairwise_distances(self.data_points, self.data_points)
        if self.R_i is None:
            covariance_matrix = compact_c2_covariance(distances, self.R_basis)
        else:
            R_pair = np.minimum(self.R_i[:, None], self.R_i[None, :])
            covariance_matrix = wendland_c2_varR(distances, R_pair)

        self.kriging_matrix = np.zeros((self.ndata + 1, self.ndata + 1), dtype=float)
        self.kriging_matrix[:self.ndata, :self.ndata] = covariance_matrix
        if self.nugget > 0.0:
            self.kriging_matrix[:self.ndata, :self.ndata][np.diag_indices(self.ndata)] += self.nugget
        self.kriging_matrix[:self.ndata, self.ndata] = 1.0
        self.kriging_matrix[self.ndata, :self.ndata] = 1.0
        self.kriging_matrix[self.ndata, self.ndata] = 0.0
        self.coefficients = None

    def update_weights(self, data_vals):
        if self.kriging_matrix is None or self.ndata == 0:
            print("ERROR in InterpolatorKriging.update_weights(): setup failed or no data.")
            self.coefficients = None
            return False
        z = np.asarray(data_vals, dtype=float)
        if z.shape[0] != self.ndata:
            print(f"ERROR in InterpolatorKriging.update_weights(): data_vals size ({z.shape[0]}) != N ({self.ndata}).")
            self.coefficients = None
            return False
        if self.verbose:
            print(f"InterpolatorKriging.update_weights(): Solving for {self.ndata+1} coefficients...")
        rhs = np.zeros(self.ndata + 1, dtype=float)
        rhs[:self.ndata] = z
        try:
            self.coefficients = solve(self.kriging_matrix, rhs)
            return True
        except np.linalg.LinAlgError:
            print("ERROR in InterpolatorKriging.update_weights(): singular / ill-conditioned system.")
            self.coefficients = None
            return False

    def evaluate(self, query_points):
        if self.coefficients is None:
            print("ERROR in InterpolatorKriging.evaluate(): call update_weights first.")
            return None
        if self.ndata == 0:
            return np.zeros(query_points.shape[0], dtype=float)
        query_points = np.asarray(query_points, dtype=float)
        nqps = query_points.shape[0]
        if nqps == 0:
            return np.array([], dtype=float)
        if self.verbose:
            print(f"InterpolatorKriging.evaluate(): {nqps} points...")
        c_coeffs = self.coefficients[:self.ndata]
        mu = self.coefficients[self.ndata]
        if self.global_eval:
            neighbor_indices_list = [np.arange(self.ndata, dtype=int) for _ in range(nqps)]
        else:
            data_kdtree = KDTree(self.data_points)
            r = self.R_basis if self.R_i is None else self.R_max
            neighbor_indices_list = data_kdtree.query_ball_point(query_points, r=r)
        interpolated_values = np.zeros(nqps, dtype=float)
        for i in range(nqps):
            q = query_points[i]
            neighbors_q_indices = neighbor_indices_list[i]
            val = mu
            if neighbors_q_indices is None or len(neighbors_q_indices) == 0:
                interpolated_values[i] = val
                continue
            neighbor_pts = self.data_points[neighbors_q_indices, :]
            neighbor_c_coeffs = c_coeffs[neighbors_q_indices]
            dists = np.linalg.norm(neighbor_pts - q, axis=1)
            if self.R_i is None:
                cov_vals = compact_c2_covariance(dists, self.R_basis)
            else:
                Ri = self.R_i[neighbors_q_indices]
                mask = dists < Ri
                if not np.any(mask):
                    interpolated_values[i] = val
                    continue
                cov_vals = wendland_c2_varR(dists[mask], Ri[mask])
                neighbor_c_coeffs = neighbor_c_coeffs[mask]
            val += np.sum(neighbor_c_coeffs * cov_vals)
            interpolated_values[i] = val
        return interpolated_values

    def evaluate_gradient(self, query_points):
        """∇E at query points (not force). F = -∇E."""
        if self.coefficients is None:
            print("ERROR in InterpolatorKriging.evaluate_gradient(): call update_weights first.")
            return None
        if self.ndata == 0:
            return np.zeros((query_points.shape[0], query_points.shape[1]), dtype=float)
        query_points = np.asarray(query_points, dtype=float)
        nqps = query_points.shape[0]
        D = query_points.shape[1]
        if nqps == 0:
            return np.array([], dtype=float).reshape(0, D)
        c_coeffs = self.coefficients[:self.ndata]
        if self.global_eval:
            neighbor_indices_list = [np.arange(self.ndata, dtype=int) for _ in range(nqps)]
        else:
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
            neighbor_c_coeffs = c_coeffs[neighbors_q_indices]
            diffs = q - neighbor_pts
            dists = np.linalg.norm(diffs, axis=1)
            if self.R_i is None:
                deriv_vals = wendland_c2_deriv(dists, self.R_basis)
            else:
                Ri = self.R_i[neighbors_q_indices]
                mask = dists < Ri
                if not np.any(mask):
                    continue
                deriv_vals = wendland_c2_deriv_varR(dists[mask], Ri[mask])
                diffs = diffs[mask]
                neighbor_c_coeffs = neighbor_c_coeffs[mask]
                dists = dists[mask]
            with np.errstate(divide='ignore', invalid='ignore'):
                direction = diffs / dists[:, None]
            direction[np.isnan(direction)] = 0.0
            gradients[i] = np.sum(neighbor_c_coeffs[:, None] * deriv_vals[:, None] * direction, axis=0)
        return gradients
