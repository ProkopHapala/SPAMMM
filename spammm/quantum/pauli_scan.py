"""pauli_scan — slim xy / xV scan engine for PauliSolverCL (PME.cl).

# === AUTO-DOC BEGIN ===
Essence: OpenCL-first charge-ring STM scans (xy rings, xV diamonds/NDR) without
pulling in ppafm's 3k-LOC GUI engine.
Design: 2/3-site geometries embed into PME.cl's fixed 4-site basis via far
high-E spectators; capacitive W only among active sites (``make_Wij_active``).
Two parametric regimes SSOT: Ruslan NTCDA dimer (Qzz≠0) vs fig3 circle trimer
(Qzz=0 — NDR). Tip μ = per-pixel Vtips; Temp[K]→eV via kB.
Caveats: scalar params['W'] must not be used alone when n_active<4 (spectators).
Hubbard/MQCA/GUI/MC-fit not here — see ``doc/TopicalAudit/ChargeRings_PME.md``.
# === AUTO-DOC END ===
"""
from __future__ import annotations

import json
import os
import numpy as np

from .PauliSolverCL import PauliSolverCL

kB_eV = 8.61733326214511e-5  # eV/K


def makePosXY(n=100, L=10.0, axs=(0, 1, 2), p0=(0.0, 0.0, 0.0)):
    """Tip xy grid at fixed z (ppafm utils.makePosXY)."""
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    Xs, Ys = np.meshgrid(x, y)
    ps = np.zeros((n * n, 3), dtype=np.float64)
    ps[:, axs[0]] = p0[axs[0]] + Xs.flatten()
    ps[:, axs[1]] = p0[axs[1]] + Ys.flatten()
    ps[:, axs[2]] = p0[axs[2]]
    return ps, Xs, Ys


def makeCircle(n=10, R=1.0, p0=(0.0, 0.0, 0.0), axs=(0, 1, 2), phi0=0.0):
    """Equally spaced sites on a circle (ppafm utils.makeCircle)."""
    phis = np.linspace(0, 2 * np.pi, n, endpoint=False) + phi0
    ps = np.zeros((n, 3), dtype=np.float64)
    ps[:, axs[0]] = p0[axs[0]] + np.cos(phis) * R
    ps[:, axs[1]] = p0[axs[1]] + np.sin(phis) * R
    ps[:, axs[2]] = p0[axs[2]]
    return ps, phis


def makeRotMats(phis, nsite=None):
    """In-plane rotation matrices from angles (rad)."""
    phis = np.asarray(phis, dtype=np.float64).ravel()
    if nsite is None:
        nsite = len(phis)
    rot = np.zeros((nsite, 3, 3), dtype=np.float32)
    ca = np.cos(phis)
    sa = np.sin(phis)
    rot[:, 0, 0] = ca
    rot[:, 1, 1] = ca
    rot[:, 0, 1] = sa
    rot[:, 1, 0] = -sa
    rot[:, 2, 2] = 1.0
    return rot


def make_pTips_line(start_point, end_point, npts=100, zT=0.0):
    x1, y1 = start_point
    x2, y2 = end_point
    ts = np.linspace(0.0, 1.0, npts)
    pTips = np.zeros((npts, 3), dtype=np.float64)
    pTips[:, 0] = x1 + (x2 - x1) * ts
    pTips[:, 1] = y1 + (y2 - y1) * ts
    pTips[:, 2] = zT
    dist = float(np.hypot(x2 - x1, y2 - y1))
    return pTips, ts, dist


def make_cpp_params(params):
    """[Rtip, zV0, zVd, Esite, beta, Gamma, W, bMirror, bRamp] — same as ppafm."""
    bMirror = params.get('bMirror', True)
    bRamp = params.get('bRamp', True)
    return np.array([
        params['Rtip'], params['zV0'], params['zVd'], params['Esite'],
        params['decay'], params['GammaT'], params['W'],
        float(bMirror), float(bRamp),
    ], dtype=np.float32)


def make_quadrupole_Coeffs(Q0, Qzz):
    """Multipole coeffs + order=2 (ppafm pauli.make_quadrupole_Coeffs)."""
    cs = np.array([Q0, 0.0, 0.0, 0.0, 0.0, Qzz, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return cs, 2


def load_site_geometry(filename):
    """Load Ruslan-style site file: x y angle_deg[, Esite] → spos(N,4), rots, angles_rad."""
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Geometry '{filename}' must have >=3 columns (x,y,angle[,E]).")
    n_sites = len(data)
    spos = np.zeros((n_sites, 4), dtype=np.float64)
    spos[:, :2] = data[:, :2]
    if data.shape[1] >= 4:
        spos[:, 3] = data[:, 3]
    angles_rad = np.radians(data[:, 2])
    rots = makeRotMats(angles_rad, nsite=n_sites)
    return spos, rots, angles_rad


def make_site_geom(params):
    """Circle (nsite,radius,…) or geometry_file — same dispatch as ppafm pauli_scan."""
    if 'geometry_file' in params and params['geometry_file']:
        return load_site_geometry(params['geometry_file'])
    nsite = int(params['nsite'])
    xyz, phis = makeCircle(n=nsite, R=float(params['radius']), phi0=float(params.get('phiRot', 0.0)))
    xyz[:, 2] = float(params.get('zQd', 0.0))
    angles = phis + float(params.get('phi0_ax', 0.0))
    spos = np.zeros((nsite, 4), dtype=np.float64)
    spos[:, :3] = xyz
    spos[:, 3] = float(params.get('Esite', 0.0))
    rots = makeRotMats(angles, nsite=nsite)
    return spos, rots, angles


def load_json_params(path):
    """Load ppafm-style params JSON (bools may be True/true)."""
    with open(path) as f:
        raw = f.read().replace('True', 'true').replace('False', 'false')
    return json.loads(raw)


def embed_sites_pme4(spos, rots=None):
    """Pad ≤4 sites to PME.cl N_SITES=4 with far, high-E spectators."""
    spos = np.asarray(spos, dtype=np.float64)
    n = spos.shape[0]
    if n > 4:
        raise ValueError(f'PME.cl supports at most 4 sites (got {n})')
    out = np.zeros((4, 4), dtype=np.float32)
    out[:n, :spos.shape[1]] = spos[:n, :min(4, spos.shape[1])]
    for i in range(n, 4):
        out[i] = [1e3 * (i + 1), 1e3 * (i + 1), 0.0, 1e3]
    if rots is None:
        rots_out = np.tile(np.eye(3, dtype=np.float32), (4, 1, 1))
    else:
        rots = np.asarray(rots, dtype=np.float32)
        rots_out = np.tile(np.eye(3, dtype=np.float32), (4, 1, 1))
        rots_out[:n] = rots[:n]
    return out, rots_out, n


def make_Wij_active(n_active, W, n_embed=4):
    """Constant capacitive coupling W between active sites only (spectators decoupled)."""
    Wij = np.zeros((n_embed, n_embed), dtype=np.float32)
    for i in range(n_active):
        for j in range(i + 1, n_active):
            Wij[i, j] = Wij[j, i] = float(W)
    return Wij


def ruslan_default_params(**overrides):
    """Baseline matching ppafm results/NTCDA Ruslan long/short (solver_0)."""
    p = dict(
        nsite=2,
        VBias=2.0,
        Rtip=3.0,
        z_tip=5.0,
        zV0=-1.0,
        zVd=15.0,
        zQd=0.0,
        Q0=1.0,
        Qzz=10.0,
        Esite=-0.09,
        W=0.05,
        Temp=3.0,          # Kelvin → set_lead uses Temp*kB (eV)
        decay=0.3,
        GammaS=0.01,
        GammaT=0.01,
        L=20.0,
        npix=120,
        bMirror=True,
        bRamp=True,
        p1_x=-15.0,
        p1_y=0.0,
        p2_x=15.0,
        p2_y=0.0,
        dQ=0.02,
        verbosity=0,
    )
    p.update(overrides)
    return p


def fig3_trimer_params(**overrides):
    """ppafm fig3 / pauli_scan_results trimer — Qzz=0 monopole, NDR regime.

    SSOT copy: ``data/charge_rings/fig3_trimer.json`` (from fig3_data/fig_1/params.json).
    Note: historical phiRot makes the triangle arbitrarily rotated; prefer
    ``symmetric_trimer_params()`` for tests/GUI defaults.
    """
    path = default_geometry_path('fig3_trimer.json')
    p = load_json_params(path) if os.path.isfile(path) else {}
    base = dict(
        nsite=3,
        radius=5.77,
        phiRot=1.31,
        phi0_ax=-0.28,
        VBias=1.0,
        V_slice=0.79,
        Rtip=3.0,
        z_tip=6.0,
        zV0=-0.9,
        zVd=20.0,
        zQd=0.0,
        Q0=1.0,
        Qzz=0.0,           # monopole — key difference vs Ruslan Qzz=10
        Esite=-0.09,
        W=0.05,
        Temp=2.6,
        decay=0.3,
        GammaS=0.01,
        GammaT=0.01,
        L=20.0,
        npix=120,
        dQ=0.02,
        p1_x=15.0,
        p1_y=-15.0,
        p2_x=-15.0,
        p2_y=15.0,
        bMirror=True,
        bRamp=True,
        verbosity=0,
    )
    base.update(p)
    base.update(overrides)
    return base


def symmetric_trimer_params(**overrides):
    """Equilateral trimer, apex on +y (φ₀=π/2), mirror about y; fig3 NDR physics.

    Horizontal xV cut y=0 through the center → I(x)=I(−x). Same Esite/W/Qzz=0
    as fig3 so charging rings + NDR remain.
    """
    p = fig3_trimer_params(
        phiRot=float(np.pi / 2),
        phi0_ax=0.0,
        p1_x=-15.0,
        p1_y=0.0,
        p2_x=15.0,
        p2_y=0.0,
        VBias=0.85,
    )
    p.update(overrides)
    return p


def make_state_labels(n_active, n_embed=4):
    """Binary labels for embed masks; spectators always empty in the low bits we care about."""
    n_states = 1 << n_embed
    return [format(s & ((1 << n_active) - 1), f'0{n_active}b') + ('' if n_active == n_embed else f'|…') for s in range(n_states)]


def configure_leads(solver, params, Vbias_mu=0.0):
    """Temp in Kelvin → eV; tip mu overwritten per-pixel by Vtips in PME.cl."""
    T_eV = float(params.get('Temp', 3.0)) * kB_eV
    solver.set_lead(0, 0.0, T_eV)
    solver.set_lead(1, float(Vbias_mu), T_eV)


def scan_xy(solver, spos, rots, params, Wij=None, return_probs=False):
    """Constant-VBias xy map → STM[npix,npix], optional dIdV via dQ finite difference."""
    npix = int(params['npix'])
    L = float(params['L'])
    zT = float(params['z_tip']) + float(params['Rtip'])
    pTips, Xs, Ys = makePosXY(n=npix, L=L, p0=(0.0, 0.0, zT))
    Vtips = np.full(len(pTips), float(params['VBias']), dtype=np.float32)
    cpp = make_cpp_params(params)
    # Scalar W in cpp unused when Wij provided — keep W=0 to avoid spectator coupling
    if Wij is not None:
        cpp = cpp.copy()
        cpp[6] = 0.0
    cs, order = make_quadrupole_Coeffs(params['Q0'], params['Qzz'])
    p4, r4, n_act = embed_sites_pme4(spos, rots)
    if Wij is None:
        Wij = make_Wij_active(n_act, params['W'])
    configure_leads(solver, params)
    I, Es, Ts, Probs, *_rest = solver.scan_current_tip(
        pTips=pTips, Vtips=Vtips, pSites=p4, params=cpp, order=order, cs=cs, rots=r4, Wij=Wij,
        return_probs=return_probs,
    )
    STM = I.reshape(npix, npix)
    if not np.isfinite(STM).all():
        STM = np.nan_to_num(STM, nan=0.0, posinf=0.0, neginf=0.0)
    dIdV = None
    dQ = float(params.get('dQ', 0.0))
    if dQ != 0.0:
        Vtips2 = np.full(len(pTips), float(params['VBias']) + dQ, dtype=np.float32)
        I2, *_ = solver.scan_current_tip(
            pTips=pTips, Vtips=Vtips2, pSites=p4, params=cpp, order=order, cs=cs, rots=r4, Wij=Wij,
        )
        STM2 = I2.reshape(npix, npix)
        if not np.isfinite(STM2).all():
            STM2 = np.nan_to_num(STM2, nan=0.0, posinf=0.0, neginf=0.0)
        dIdV = (STM2 - STM) / dQ
    extent = [-L, L, -L, L]
    out = dict(STM=STM, dIdV=dIdV, Es=Es, Ts=Ts, Xs=Xs, Ys=Ys, extent=extent, spos=spos, pTips=pTips, n_active=n_act)
    if return_probs and Probs is not None:
        out['probs'] = Probs.reshape(npix, npix, -1)
    return out


def scan_xV(solver, spos, rots, params, nx=100, nV=120, Vmin=0.0, Vmax=None, Wij=None, return_probs=False):
    """Line×voltage scan → STM[nV,nx], dIdV[nV,nx]; optional probs[nV,nx,nStates]."""
    if Vmax is None:
        Vmax = float(params['VBias'])
    zT = float(params['z_tip']) + float(params['Rtip'])
    start = (float(params['p1_x']), float(params['p1_y']))
    end = (float(params['p2_x']), float(params['p2_y']))
    pTips, ts, dist = make_pTips_line(start, end, npts=nx, zT=zT)
    Vbiases = np.linspace(Vmin, Vmax, nV, dtype=np.float64)
    pTips_rep = np.tile(pTips, (nV, 1))
    Vtips_rep = np.repeat(Vbiases, nx).astype(np.float32)
    cpp = make_cpp_params(params)
    if Wij is not None:
        cpp = cpp.copy()
        cpp[6] = 0.0
    cs, order = make_quadrupole_Coeffs(params['Q0'], params['Qzz'])
    p4, r4, n_act = embed_sites_pme4(spos, rots)
    if Wij is None:
        Wij = make_Wij_active(n_act, params['W'])
    configure_leads(solver, params)
    I, Es, Ts, Probs, *_rest = solver.scan_current_tip(
        pTips=pTips_rep, Vtips=Vtips_rep, pSites=p4, params=cpp, order=order, cs=cs, rots=r4, Wij=Wij,
        return_probs=return_probs,
    )
    STM = I.reshape(nV, nx)
    if not np.isfinite(STM).all():
        # V≈0 / extreme tips can singularize Gauss–Jordan at low T — treat as I=0
        STM = np.nan_to_num(STM, nan=0.0, posinf=0.0, neginf=0.0)
    dIdV = np.gradient(STM, Vbiases, axis=0)
    extent = [min(start[0], end[0]), max(start[0], end[0]), Vmin, Vmax]
    # Along-cut distance for 1D overlays (0 … |p2-p1|)
    dist_axis = np.linspace(0.0, dist, nx)
    out = dict(
        STM=STM, dIdV=dIdV, Es=Es, Ts=Ts, pTips=pTips, Vbiases=Vbiases,
        extent=extent, spos=spos, dist=dist, dist_axis=dist_axis, start=start, end=end, n_active=n_act,
    )
    if return_probs and Probs is not None:
        P = Probs.reshape(nV, nx, -1)
        if not np.isfinite(P).all():
            P = np.nan_to_num(P, nan=0.0)
        out['probs'] = P
    return out


def scan_1d(solver, spos, rots, params, nx=100, Wij=None, return_probs=True):
    """Fixed-VBias line cut (params p1→p2) → I(s), optional P_state(s)."""
    zT = float(params['z_tip']) + float(params['Rtip'])
    start = (float(params['p1_x']), float(params['p1_y']))
    end = (float(params['p2_x']), float(params['p2_y']))
    pTips, ts, dist = make_pTips_line(start, end, npts=nx, zT=zT)
    Vtips = np.full(nx, float(params['VBias']), dtype=np.float32)
    cpp = make_cpp_params(params)
    if Wij is not None:
        cpp = cpp.copy(); cpp[6] = 0.0
    cs, order = make_quadrupole_Coeffs(params['Q0'], params['Qzz'])
    p4, r4, n_act = embed_sites_pme4(spos, rots)
    if Wij is None:
        Wij = make_Wij_active(n_act, params['W'])
    configure_leads(solver, params)
    I, Es, Ts, Probs, *_ = solver.scan_current_tip(
        pTips=pTips, Vtips=Vtips, pSites=p4, params=cpp, order=order, cs=cs, rots=r4, Wij=Wij,
        return_probs=return_probs,
    )
    I = np.nan_to_num(I, nan=0.0)
    dist_axis = np.linspace(0.0, dist, nx)
    out = dict(I=I, Es=Es, Ts=Ts, pTips=pTips, dist_axis=dist_axis, dist=dist,
               start=start, end=end, spos=spos, n_active=n_act, VBias=float(params['VBias']))
    if return_probs and Probs is not None:
        out['probs'] = np.nan_to_num(Probs, nan=0.0)
    return out


def default_geometry_path(name='Ruslan_long.txt'):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'charge_rings'))
    return os.path.join(root, name)
