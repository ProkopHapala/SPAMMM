"""PauliSolverCL8 — PyOpenCL wrapper for kernels/PME8.cl (PME, N_SITES≤8).

# === AUTO-DOC BEGIN ===
Essence: GPU steady-state Pauli Master Equation currents for up to 8 molecular
sites (256 states). Sparse iterative Jacobi solver with normalisation.
Design: mirrors PauliSolverCL API (scan_current_tip) but uses PME8.cl kernel
with 256-thread workgroups. Tip μ from Vtips; Gamma convention (Gamma/π)².
Caveats: Jacobi convergence depends on tol/max_iter; for n<8 sites, pad with
far high-E spectators (same as PME4 embed).
# === AUTO-DOC END ===
"""
import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from ..utils.OpenCLBase import OpenCLBase


def pack_float4(xyz_arr, w_arr=None, w_default=0.0):
    n = len(xyz_arr)
    packed = np.zeros((n, 4), dtype=np.float32)
    if xyz_arr.shape[1] >= 3:
        packed[:, 0:3] = xyz_arr[:, 0:3]
    if xyz_arr.shape[1] == 4:
        packed[:, 3] = xyz_arr[:, 3]
    elif w_arr is not None:
        packed[:, 3] = w_arr
    else:
        packed[:, 3] = w_default
    return np.ascontiguousarray(packed)


class PauliSolverCL8(OpenCLBase):
    """8-site / 256-state PME solver via PME8.cl."""

    def __init__(self, nSingle=8, nLeads=2, verbosity=0, ctx=None, queue=None,
                 preferred_vendor='nvidia', device_index=0, bPrint=True,
                 max_iter=2000, tol=1e-6):
        if nSingle > 8:
            raise ValueError(f"PME8.cl supports at most 8 sites (got {nSingle})")
        # nloc = 256 threads per workgroup
        super().__init__(nloc=256, preferred_vendor=preferred_vendor, device_index=device_index,
                         bPrint=bPrint, ctx=ctx, queue=queue)
        self.nSingle = nSingle
        self.nStates = 2 ** nSingle
        self.nLeads = nLeads
        self.verbosity = verbosity
        self.max_iter = max_iter
        self.tol = tol

        kdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kernels')
        kpath = os.path.abspath(os.path.join(kdir, 'PME8.cl'))
        if not self.load_program(kernel_path=kpath, bMakeHeaders=False, bPrint=bPrint):
            raise FileNotFoundError(f"Kernel file not found or failed to build: {kpath}")

        self.krn_compute_tip_interaction = cl.Kernel(self.prg, 'compute_tip_interaction')
        self.krn_solve_pme8 = cl.Kernel(self.prg, 'solve_pme8')
        self._init_lookups()

    def _init_lookups(self):
        self.W = 0.0
        self.lead_params = np.array([0.0, 0.01, 0.0, 0.01], dtype=np.float32)

    def set_lead(self, lead_idx, mu, temp):
        if lead_idx < 2:
            self.lead_params[lead_idx * 2] = mu
            self.lead_params[lead_idx * 2 + 1] = temp

    def scan_current_tip(self, pTips, Vtips, pSites, params, order, cs,
                         rots=None, Wij=None,
                         return_probs=False, return_state_energies=False):
        """Main simulation function (mirrors PauliSolverCL.scan_current_tip).

        pTips: (N, 3) tip positions.
        Vtips: (N,) voltages.
        pSites: (n_sites, 3) or (n_sites, 4). Padded to 8 by caller.
        params: [Rtip, zV0, zVd, Esite, beta, Gamma, W, bMirror, bRamp]
        """
        n_pixels = len(pTips)
        n_sites = self.nSingle

        p_tips_packed = pack_float4(pTips)
        p_tips_cl = cl_array.to_device(self.queue, p_tips_packed)

        e0_default = params[3] if len(params) > 3 else 0.0
        p_sites_packed = pack_float4(pSites, w_default=e0_default)
        p_sites_cl = cl_array.to_device(self.queue, p_sites_packed)

        if rots is None:
            rots_host = np.tile(np.eye(3, dtype=np.float32), (n_sites, 1, 1))
        else:
            rots_host = np.asarray(rots, dtype=np.float32).reshape(n_sites, 3, 3)
        rots_cl = cl_array.to_device(self.queue, rots_host.reshape(n_sites * 3 * 3))

        v_tips_cl = cl_array.to_device(self.queue, np.array(Vtips, dtype=np.float32))
        cs_cl = cl_array.to_device(self.queue, np.array(cs, dtype=np.float32))
        params_cl = cl_array.to_device(self.queue, np.array(params, dtype=np.float32))

        h_shifts_cl = cl_array.zeros(self.queue, (n_pixels, n_sites), dtype=np.float32)
        t_factors_cl = cl_array.zeros(self.queue, (n_pixels, n_sites), dtype=np.float32)

        # Kernel 1: tip interaction
        global_size_1 = (n_pixels,)
        self.krn_compute_tip_interaction(
            self.queue, global_size_1, None,
            np.int32(n_pixels), np.int32(n_sites),
            p_tips_cl.data,
            p_sites_cl.data,
            rots_cl.data,
            v_tips_cl.data,
            cs_cl.data,
            params_cl.data,
            np.int32(order),
            h_shifts_cl.data,
            t_factors_cl.data
        )

        # Physics constants
        gamma_val = params[5]
        w_val = params[6]
        gamma_input = (gamma_val / np.pi) ** 2  # match C++ convention

        lead_params_cl = cl_array.to_device(self.queue, self.lead_params)
        H_single_cl = cl_array.zeros(self.queue, (n_sites, n_sites), dtype=np.float32)
        Wij_cl = None
        if Wij is not None:
            Wij = np.ascontiguousarray(Wij, dtype=np.float32)
            Wij_cl = cl_array.to_device(self.queue, Wij)

        out_current_cl = cl_array.zeros(self.queue, n_pixels, dtype=np.float32)
        out_probs_cl = None
        out_stateEs_cl = None
        if return_probs:
            out_probs_cl = cl_array.zeros(self.queue, n_pixels * self.nStates, dtype=np.float32)
        if return_state_energies:
            out_stateEs_cl = cl_array.zeros(self.queue, n_pixels * self.nStates, dtype=np.float32)

        # Kernel 2: PME8 solver — 256 threads per pixel
        local_size_2 = (256,)
        global_size_2 = (n_pixels * 256,)

        self.krn_solve_pme8(
            self.queue, global_size_2, local_size_2,
            np.int32(n_pixels), np.int32(n_sites), np.int32(self.nStates),
            h_shifts_cl.data, t_factors_cl.data,
            v_tips_cl.data,
            lead_params_cl.data, H_single_cl.data,
            Wij_cl.data if Wij_cl is not None else None,
            np.float32(w_val),
            np.float32(gamma_input), np.float32(gamma_input),
            out_current_cl.data,
            out_probs_cl.data if out_probs_cl is not None else None,
            out_stateEs_cl.data if out_stateEs_cl is not None else None,
            np.int32(self.max_iter),
            np.float32(self.tol)
        )

        currents = out_current_cl.get().astype(np.float64)

        Es = h_shifts_cl.get().astype(np.float64) if return_state_energies else None
        Ts = t_factors_cl.get().astype(np.float64)
        Probs = None
        StateEs = None
        if out_probs_cl is not None:
            Probs = out_probs_cl.get().astype(np.float64).reshape(n_pixels, self.nStates)
        if out_stateEs_cl is not None:
            StateEs = out_stateEs_cl.get().astype(np.float64).reshape(n_pixels, self.nStates)

        # is_valid_point cut (same as PME4)
        Tmin_cut = 0.0
        EW_cut = 2.0
        W_scalar = float(w_val)
        Es_sp = Es.reshape(n_pixels, n_sites) if Es is not None else None
        Ts_sp = Ts.reshape(n_pixels, n_sites)
        if Es_sp is not None:
            Emax = Es_sp.max(axis=1)
            gamma_amp = np.sqrt(gamma_input)
            Tmax = np.max(np.abs(gamma_amp * Ts_sp), axis=1)
            invalid = (Emax + W_scalar * EW_cut < 0.0) | (Tmax < Tmin_cut)
            if np.any(invalid):
                currents = currents.copy()
                currents[invalid] = 0.0

        return currents, Es, Ts, Probs, StateEs

    def cleanup(self):
        pass
