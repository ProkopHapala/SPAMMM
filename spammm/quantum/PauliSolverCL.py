"""PauliSolverCL — PyOpenCL wrapper for kernels/PME.cl (full PME, N_SITES=4).

# === AUTO-DOC BEGIN ===
Essence: GPU steady-state Pauli Master Equation currents for ≤4 molecular sites.
Design: FireCore ``pauli_ocl.py`` API preserved; device/kernels via SPAMMM
``OpenCLBase`` (``preferred_vendor='nvidia'``). Tip μ from Vtips; Gamma convention
``(Gamma/π)²`` matches C++ ``VS=Gamma/π``.
Caveats: nSingle must be 4 (kernel hardcode); pad smaller systems in ``pauli_scan``.
# === AUTO-DOC END ===
"""
import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from ..utils.OpenCLBase import OpenCLBase


def pack_float4(xyz_arr, w_arr=None, w_default=0.0):
    """Pack (N,3) or (N,4) into contiguous float32 (N,4) for OpenCL float4."""
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


class PauliSolverCL(OpenCLBase):
    def __init__(self, nSingle=4, nLeads=2, verbosity=0, ctx=None, queue=None,
                 preferred_vendor='nvidia', device_index=0, bPrint=True):
        if nSingle != 4:
            raise ValueError(f"PME.cl is hardcoded for nSingle=4 (got {nSingle})")
        super().__init__(nloc=16, preferred_vendor=preferred_vendor, device_index=device_index,
                         bPrint=bPrint, ctx=ctx, queue=queue)
        self.nSingle = nSingle
        self.nStates = 2 ** nSingle
        self.nLeads = nLeads
        self.verbosity = verbosity

        kdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'kernels')
        kpath = os.path.abspath(os.path.join(kdir, 'PME.cl'))
        if not self.load_program(kernel_path=kpath, bMakeHeaders=False, bPrint=bPrint):
            raise FileNotFoundError(f"Kernel file not found or failed to build: {kpath}")

        self.krn_compute_tip_interaction = cl.Kernel(self.prg, 'compute_tip_interaction')
        self.krn_solve_pme = cl.Kernel(self.prg, 'solve_pme')
        self._init_lookups()

    def _init_lookups(self):
        # Identity state ordering (matches C++/Python pauli.make_state_order for nsite=4)
        self.state_order_host = np.arange(self.nStates, dtype=np.int32)
        self.state_order_dev = cl_array.to_device(self.queue, self.state_order_host)
        self.W = 0.0
        # [mu0, T0, mu1, T1]
        self.lead_params = np.array([0.0, 0.01, 0.0, 0.01], dtype=np.float32)

    def set_lead(self, lead_idx, mu, temp):
        if lead_idx < 2:
            self.lead_params[lead_idx * 2] = mu
            self.lead_params[lead_idx * 2 + 1] = temp

    def scan_current_tip(self, pTips, Vtips, pSites, params, order, cs,
                         state_order=None, rots=None, bOmp=False,
                         bMakeArrays=True, Ts=None, return_probs=False,
                         return_state_energies=False, externTs=False, return_curmat=False,
                         Wij=None):
        """
        Main simulation function.

        pTips: (N, 3) tip positions.
        Vtips: (N,) voltages.
        pSites: (4, 3) or (4, 4). If (4,3), W component from params[3] (Esite).
        params: [Rtip, zV0, zVd, Esite, beta, Gamma, W, bMirror, bRamp]
        """
        n_pixels = len(pTips)
        n_sites = self.nSingle

        p_tips_packed = pack_float4(pTips)
        p_tips_cl = cl_array.to_device(self.queue, p_tips_packed)

        pSites = np.array(pSites)
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

        # ----------------------------------------------------------------
        # 3. Kernel 2: Pauli Master Equation
        # ----------------------------------------------------------------
        
        # Prepare Physics Constants
        # params structure: [Rtip, zV0, zVd, Esite, beta, Gamma, W, ...]
        gamma_val = params[5]
        w_val = params[6]
        
        # Gamma in C++ code usually means Gamma/PI in the rate eq context?
        # The kernel uses standard rate = 2*PI * |T|^2.
        # If C++ VS = Gamma/PI, then C++ Rate = VS * 2*PI = 2*Gamma. 
        # Let's adhere to the standard: 
        # C++ input "Gamma" usually implies the broadening Gamma = 2*pi*|V|^2*rho.
        # To match C++ exactly, we pass Gamma/PI as the base factor if C++ does so.
        # Based on C++ `evalSitesTipsTunneling` using `Amp * exp` and solver using `VS=Gamma/PI`,
        # We should pass `Gamma/PI` to the kernel so kernel doing `2*PI*...` results in `2*Gamma`.
        # Wait, exact match check: 
        # C++: TLeads[...] = Gamma/PI * exp(...). Coupling = T^2. 
        # Rate = Coupling * 2PI = (Gamma/PI)^2 * exp^2 * 2PI = 2/PI * Gamma^2 * exp^2. 
        # This seems odd physically (Gamma squared?).
        # 
        # Let's assume the standard: Rate ~ Gamma. 
        # If user passes Gamma, we pass Gamma/(2*PI) to kernel as 'Gamma0'? 
        # No, let's look at C++ `solve_pme`:
        # `pauli_factors[0] = coupling_val * fermi * 2 * PI;`
        # `coupling_val = tij * tji`.
        # `tij` comes from `TLeads`. 
        # `TLeads` initialized as `Gamma/PI` (VS) or `(Gamma/PI)*exp` (VT).
        # So Rate ~ (Gamma/PI)^2 * 2PI. 
        # This is the "C++ Convention" we must keep.
        
        # Match C++ scan_current_tip_ convention:
        #   VS = Gamma/pi; VT = Gamma/pi; coupling ~ (VS)^2, rates use *2*pi
        # Here kernel uses:
        # C++ sets VS=VT=Gamma/pi, and rates use (TLeads)^2 * 2*pi.
        # Kernel multiplies by T^2, so pass (Gamma/pi)^2 to match.
        gamma_input = (gamma_val / np.pi) ** 2

        lead_params_cl = cl_array.to_device(self.queue, self.lead_params)
        H_single_cl = cl_array.zeros(self.queue, (n_sites, n_sites), dtype=np.float32)
        Wij_cl = None
        if Wij is not None:
            Wij = np.ascontiguousarray(Wij, dtype=np.float32)
            Wij_cl = cl_array.to_device(self.queue, Wij)

        out_current_cl = cl_array.zeros(self.queue, n_pixels, dtype=np.float32)

        out_probs_cl = None
        out_stateEs_cl = None
        out_K_cl = None
        out_curmat_cl = None
        if return_probs:
            out_probs_cl = cl_array.zeros(self.queue, n_pixels * self.nStates, dtype=np.float32)
        if return_state_energies:
            out_stateEs_cl = cl_array.zeros(self.queue, n_pixels * self.nStates, dtype=np.float32)
        if (return_probs or return_state_energies):
            out_K_cl = cl_array.zeros(self.queue, n_pixels * self.nStates * self.nStates, dtype=np.float32)
        if return_curmat:
            out_curmat_cl = cl_array.zeros(self.queue, n_pixels * self.nStates * self.nStates, dtype=np.float32)

        global_size_2 = (n_pixels * 16,)
        local_size_2 = (16,)

        self.krn_solve_pme(
            self.queue, global_size_2, local_size_2,
            np.int32(n_pixels), np.int32(n_sites), np.int32(self.nStates),
            h_shifts_cl.data, t_factors_cl.data,
            v_tips_cl.data,
            lead_params_cl.data, H_single_cl.data,
            Wij_cl.data if Wij_cl is not None else None,
            np.float32(w_val),
            np.float32(gamma_input), np.float32(gamma_input),
            self.state_order_dev.data,
            out_current_cl.data,
            out_curmat_cl.data if out_curmat_cl is not None else None,
            out_K_cl.data if out_K_cl is not None else None,
            out_probs_cl.data if out_probs_cl is not None else None,
            out_stateEs_cl.data if out_stateEs_cl is not None else None
        )

        currents = out_current_cl.get().astype(np.float64)

        Es = None
        Ts = None
        Probs = None
        StateEs = None
        K = None
        CurMat = None

        Es_tmp = h_shifts_cl.get().astype(np.float64)
        Ts_tmp = t_factors_cl.get().astype(np.float64)
        if return_state_energies:
            Es = Es_tmp
        Ts = Ts_tmp

        # Match C++ is_valid_point() cut (Emax + W*EW_cut < 0 or Tmax < Tmin_cut)
        Tmin_cut = 0.0
        EW_cut = 2.0
        W_scalar = float(w_val)
        if Es_tmp is not None and Ts_tmp is not None:
            Es_sp = Es_tmp.reshape(n_pixels, n_sites)
            Ts_sp = Ts_tmp.reshape(n_pixels, n_sites)
            Emax = Es_sp.max(axis=1)
            gamma_amp = np.sqrt(gamma_input)
            Tmax = np.max(np.abs(gamma_amp * Ts_sp), axis=1)
            invalid = (Emax + W_scalar * EW_cut < 0.0) | (Tmax < Tmin_cut)
            if np.any(invalid):
                currents = currents.copy()
                currents[invalid] = 0.0

        if out_probs_cl is not None:
            Probs = out_probs_cl.get().astype(np.float64).reshape(n_pixels, self.nStates)
        if out_stateEs_cl is not None:
            StateEs = out_stateEs_cl.get().astype(np.float64).reshape(n_pixels, self.nStates)
        if out_K_cl is not None:
            K = out_K_cl.get().astype(np.float64).reshape(n_pixels, self.nStates, self.nStates)
        if out_curmat_cl is not None:
            CurMat = out_curmat_cl.get().astype(np.float64).reshape(n_pixels, self.nStates, self.nStates)

        return currents, Es, Ts, Probs, StateEs, K, CurMat

    def cleanup(self):
        pass
