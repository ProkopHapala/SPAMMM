#!/usr/bin/env python3
"""Systematic near-EF Tersoff-Hamann LDOS + unfolded bands for the vacuum ribbons.

Extends test_ribbon_bloch.py (same DFTBcore C(k) -> project_bloch_points path;
see doc/prokop/userguide/bloch_slice.md) to the full enum:

  chem {C,N,O} x width r1..r5 x 7 protonation states (0H, 1H-p/d, 2H-ss/os-adj/sep)

Differences vs test_ribbon_bloch.py:
  * dense k-mesh along the ribbon (NKX, default 48 supercell points = 24 stored
    after the +/-k fold) instead of copying the 8x1x1 relaxation mesh
  * TWO half-windows per state: occupied [EF-W, EF) and empty [EF, EF+W]
    (~ STM at -W / +W bias); per-side frontier fallback for real gaps
  * band structure unfolded to the primitive (1-cell) BZ via SPAMMM's
    unfold_T_weights (subcell-translation spectral projectors in the true
    S(k) metric — required: DFTB eigvecs are S-orthonormal, |S_off|~0.4);
    pristine 0H unfolded bands overlaid as reference
  * green '+' marks the switched sites (heavy atom whose H-count differs
    from the 0H cell)

Outputs (debug/ribbon_mio/stm/, or stm_xscan/ with --xscan):
  {tag}_{chem}_r{w}_ldos_{lin,sat5,log}.png   states in a row, shared norm
  {tag}_{chem}_r{w}_bands.png                 unfolded bands, one row per state
  index.html                                  gallery
  cache: debug/ribbon_mio/stm_work/{enumv,xscan}/

Usage:
  python test_ribbon_stm_sys.py                      # everything (cache-aware)
  python test_ribbon_stm_sys.py --chem C --widths 1  # subset
  python test_ribbon_stm_sys.py --plot-only          # replot from work/*.npz

Per-case cache lives in tests/grid/work/ribbon_stm_sys/*.npz (gitignored).
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np

DFTB_REPO = Path('/home/prokop/git/dftbplus')        # DFTB+ build + pyBall (external dep)
sys.path.insert(0, str(DFTB_REPO))
sys.path.insert(0, str(DFTB_REPO / 'tests' / 'grid'))  # test_ribbon_bloch helpers
sys.path.insert(0, '/home/prokop/git/SPAMMM')          # spammm builders + unfold helpers

from pyBall.DFTBcore import DFTBcore
from pyBall.OCL.DFTBplusGridProjector import DFTBplusGridProjector
from pyBall.OCL.DFTBplusParser import parse_basis_hsd_ang
from pyBall.plotUtils import plot_ldos_row, scatter_atoms

from test_ribbon_bloch import (read_gen, sk_prefix, write_case, fermi_ev,
                               run_dftb, cells_x, plane, overlay, HA2EV)

from spammm.topology.ribbon_pbc import build_edge_switch_cell, junction_state_strings, xscan_state_strings
from spammm.quantum.pi_bond_order import subcell_group_indices, unfold_T_weights

RIBBON_ROOT = Path('/home/prokop/git/SPAMMM/debug/ribbon_mio')
WFC = Path(__file__).parent / 'dftb_ptcda' / 'wfc.mio-1-1.hsd'
WORK = RIBBON_ROOT / 'stm_work' / 'enumv'              # per-case cache npz + run dirs
OUT = RIBBON_ROOT / 'stm'                              # figures + index.html

NKX = 48          # supercell mesh along x (24 irreducible -> 96 primitive channels)
WINS = (0.1, 0.2, 0.4)    # eV, TH-LDOS windows each side of EF
BAND_WIN = 4.0    # eV, unfolded-bands y-range around EF
Z_ABOVE = 3.0     # slice height over the molecular plane (pz is zero at z=0)
DX = 0.12
NCELLS = 4        # supercell = 4 primitive cells (ncells of the enum; 8 for --xscan)
TAG = 'enumv'     # enum dir prefix (xscanv for --xscan)
GPAW_ROOT = None  # --gpaw <root>: read hs.npz + all.gpw instead of running DFTB
NAO_DZP = {'H': 5, 'C': 13, 'N': 13, 'O': 13}   # GPAW dzp AOs per element (unfold groups)
METHOD = 'mio-1-1'  # figure label; 'GPAW/dzp' under --gpaw
STATES = ['0H', '1H-p', '1H-d', '2H-ss-adj', '2H-ss-sep', '2H-os-adj', '2H-os-sep']
STATE_STRS = junction_state_strings(NCELLS)
WIDTHS = (1, 2, 3, 4, 5)


def dense_kblock(nkx):
    """SupercellFolding text: nkx along x, half-shifted (same convention as the enum inputs)."""
    return f'{nkx} 0 0\n    0 1 0\n    0 0 1\n    0.5 0.0 0.0'


def ideal_geom(chem, width, state):
    """Builder geometry (same atom order as the enum's gen) for subcell grouping."""
    st = STATE_STRS[state].split('|')[0]
    atoms, lvs, _ = build_edge_switch_cell(2 * (width + 1), NCELLS, st, chem=chem)
    return atoms, lvs


def switched_sites_xy(atoms0, atoms1, lvs, rch=1.35):
    """x,y of heavy atoms whose H-count differs from the 0H cell (min-image in x)."""
    apos0, apos1 = np.asarray(atoms0.apos), np.asarray(atoms1.apos)
    hvy0 = np.array([not e.startswith('H') for e in atoms0.enames])
    hvy1 = np.array([not e.startswith('H') for e in atoms1.enames])
    p0, p1 = apos0[hvy0], apos1[hvy1]
    h0, h1 = apos0[~hvy0], apos1[~hvy1]
    Lx = float(lvs[0, 0])

    def n_h(ph, hh):
        if len(hh) == 0:
            return np.zeros(len(ph), int)
        d = ph[:, None, :] - hh[None, :, :]
        d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
        return (np.linalg.norm(d, axis=-1) < rch).sum(axis=1)

    c0, c1 = n_h(p0, h0), n_h(p1, h1)
    dx = p1[:, 0, None] - p0[None, :, 0]
    dx -= Lx * np.round(dx / Lx)
    dy = p1[:, 1, None] - p0[None, :, 1]
    j0 = np.argmin(dx * dx + dy * dy, axis=1)
    return p1[c1 != c0[j0], :2]


def select_window(C, E, kpts, kw, lo, hi, side, tag):
    """States with E in [lo,hi); if empty take the single frontier state on `side` per k."""
    ev = E * HA2EV
    pick = (ev >= lo) & (ev < hi)
    if not np.any(pick):
        pick = np.zeros(ev.shape, dtype=bool)
        for ik in range(ev.shape[0]):
            m = np.where(ev[ik] < hi)[0] if side == 'occ' else np.where(ev[ik] >= lo)[0]
            if m.size == 0:
                raise RuntimeError(f'{tag}: k{ik} no state on {side} side of [{lo:.2f},{hi:.2f}]')
            pick[ik, m[np.argmax(ev[ik, m])] if side == 'occ' else m[np.argmin(ev[ik, m])]] = True
        print(f'  [window] {side} empty; frontier state at each k', flush=True)
    rows, ks, ws, es = [], [], [], []
    for ik, mo in zip(*np.nonzero(pick)):
        rows.append(C[ik, mo])
        ks.append(kpts[ik])
        ws.append(2.0 * kw[ik])
        es.append(ev[ik, mo])
    return np.array(rows), np.array(ks), np.array(ws), np.array(es)


def run_primitive(dftb, chem, width, nkx):
    """Primitive-cell (x1) bands of the pristine 0H ribbon -> clean reference lines.

    Ideal builder geometry at ncells=1 (cutting relaxed subcells is fragile for
    N/O chem); eigenvalues on a dense primitive mesh, mirrored to the full BZ.
    """
    tag = f'{TAG}_{chem}_r{width}_prim'
    src = RIBBON_ROOT / f'{TAG}_{chem}_r{width}' / '0H'
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    from spammm.topology.ribbon_pbc import EDGE_SWITCH, A_CC
    b = MoleculeEditorBackend(a_CC=A_CC)
    base = EDGE_SWITCH[chem][0]
    b.build_zigzag_ribbon(width_chains=2 * (width + 1), length_cells=1,
                          passivation_bottom=base, passivation_top=base, bPeriodicX=True)
    b._sync_sys()
    apos = np.asarray(b.sys.apos, float).copy()
    hvy = np.array([e.split('_')[0] != 'H' for e in b.sys.enames])
    h = apos[hvy, 1].ptp()
    apos[:, 1] -= apos[hvy, 1].min() - 6.0
    lvs = np.array([[2.0 * A_CC * np.cos(np.pi / 6.0), 0.0, 0.0], [0.0, h + 12.0, 0.0], [0.0, 0.0, 20.0]])
    sym = [e.split('_')[0] for e in b.sys.enames]
    wd = WORK / tag
    write_case(wd, sym, apos, lvs, dense_kblock(nkx),
               sk_prefix((src / 'dftb_in.hsd').read_text(), src / 'dftb_in.hsd'))
    cwd0 = os.getcwd()
    os.chdir(wd)
    try:
        energy, C, E, kpts, kw = run_dftb(dftb, 'dftb_in.hsd')
        ef = fermi_ev(wd / 'detailed.out')
    finally:
        os.chdir(cwd0)
    G = 2.0 * np.pi / float(lvs[0, 0])
    kf = np.concatenate([kpts[:, 0], -kpts[:, 0]])            # half-BZ -> full
    ev = np.concatenate([E, E], axis=0) * HA2EV
    return {'pkx': kf * G, 'pev': ev, 'pef': ef, 'pE': energy, 'G': G}


def run_dftb_hs(dftb, hsd_name):
    """run_dftb + complex S(k).  Needed because DFTBcore eigenvectors are
    S-orthonormal (c* S c^T = I) and the stored overlap is NOT identity even
    for mio (|S_offdiag| ~0.4) — unfold_T_weights needs the true metric."""
    dftb.init(hsd_name, 'dftb.log')
    energy = dftb.run_scf()
    C, E = dftb.get_eigvecs_cplx()
    kpts, kw = dftb.get_kpoints()
    Sk = np.asarray(dftb.get_s_cplx())
    dftb.finalize()
    return energy, C, E, kpts, kw, Sk


def run_case(dftb, proj, basis, by_name, chem, width, state, nkx, wins, dx):
    """DFTB SP on the relaxed gen + LDOS maps + unfolded bands -> cache dict."""
    tag = f'{TAG}_{chem}_r{width}_{state}'
    src = RIBBON_ROOT / f'{TAG}_{chem}_r{width}' / state
    gen, hsd_src = src / 'geom.out.gen', src / 'dftb_in.hsd'
    if not gen.is_file() or not hsd_src.is_file():
        raise RuntimeError(f'missing relaxed input in {src}')
    sym, pos, lat = read_gen(gen)
    a1 = lat[0]
    if max(abs(a1[1]), abs(a1[2]), abs(lat[1, 0]), abs(lat[1, 2])) > 1e-3:
        raise RuntimeError(f'{tag}: non-orthogonal cell\n{lat}')
    wd = WORK / tag
    write_case(wd, sym, pos, lat, dense_kblock(nkx), sk_prefix(hsd_src.read_text(), hsd_src))
    cwd0 = os.getcwd()
    os.chdir(wd)
    try:
        energy, C, E, kpts, kw, Sk = run_dftb_hs(dftb, 'dftb_in.hsd')
        ef = fermi_ev(wd / 'detailed.out')
    finally:
        os.chdir(cwd0)
    ispec = np.array([by_name[s] for s in sym], dtype=np.int32)
    atoms_d = proj.prepare_atoms_dftb(pos, ispec, basis)
    norb = int(atoms_d['i0orb'][-1] + atoms_d['norb'][-1])
    if norb != C.shape[2]:
        raise RuntimeError(f'{tag}: basis norb {norb} != DFTB norb {C.shape[2]}')
    heavy = np.array([s != 'H' for s in sym])
    pts, extent, shape = plane(pos, a1, nrep=1, heavy=heavy)
    cell_cart, cell_n = cells_x(a1, n=2)
    rhos = {'occ': [], 'unocc': []}
    spans = {'occ': [], 'unocc': []}
    nst = {'occ': [], 'unocc': []}
    for wv in wins:
        for side, (lo, hi) in (('occ', (ef - wv, ef)), ('unocc', (ef, ef + wv))):
            cs, ks, ws, es = select_window(C, E, kpts, kw, lo, hi, side, tag)
            rho, _ = proj.project_bloch_points(pts, atoms_d, cell_cart, cell_n, cs, ks, ws, write_psi=False)
            if not np.isfinite(rho).all() or float(rho.max()) <= 0.0:
                raise RuntimeError(f'{tag}/{side} w={wv}: LDOS max={float(np.max(rho))}')
            rhos[side].append(rho.reshape(shape))
            spans[side].append((float(es.min()), float(es.max())))
            nst[side].append(len(es))

    # unfolded bands: C[nk,norb,nband] (get_eigvecs_cplx gives [nk,band,orb])
    atoms_i, lvs = ideal_geom(chem, width, state)
    IDX = subcell_group_indices(np.asarray(atoms_i.apos), np.asarray(lvs), list(atoms_i.enames), NCELLS, strict=False)
    q, W = unfold_T_weights(np.ascontiguousarray(C.transpose(0, 2, 1)), Sk, kpts[:, 0], IDX, NCELLS)
    ev = E * HA2EV
    nkb = ev.shape[1]
    qq = np.tile(q[:, :, None], (1, 1, nkb)).ravel()          # [nk,nc,nband] -> flat
    ee = np.tile(ev[:, None, :], (1, NCELLS, 1)).ravel()
    ww = W.ravel()
    qq = np.concatenate([qq, (-qq) % 1.0])                    # stored k is half-BZ; mirror by time reversal
    ee = np.concatenate([ee, ee])
    ww = np.concatenate([ww, ww])
    G = 2.0 * np.pi / (float(a1[0]) / NCELLS)                 # primitive G [1/A]
    kx = ((qq % 1.0 + 0.5) % 1.0 - 0.5) * G                   # centered primitive BZ
    keep = (np.abs(ee - ef) < BAND_WIN) & (ww > 0.02)
    kband = (np.abs(ev - ef) < BAND_WIN + 0.5).any(axis=0)    # bands near window -> structured arrays
    a0, _ = ideal_geom(chem, width, '0H')
    sites = switched_sites_xy(a0, atoms_i, lvs)
    at, en = overlay(sym, pos, a1, extent, nrep=1)
    return {
        'rho_occ': np.stack(rhos['occ']), 'rho_unocc': np.stack(rhos['unocc']),
        'extent': np.array(extent), 'wins': np.asarray(wins, float),
        'span_occ': np.array(spans['occ']), 'span_unocc': np.array(spans['unocc']),
        'nst_occ': np.array(nst['occ']), 'nst_unocc': np.array(nst['unocc']), 'ef': ef, 'E': energy,
        'kx': kx[keep], 'be': ee[keep], 'bw': ww[keep], 'G': G,
        'qA': q, 'eA': ev[:, kband], 'wA': W[:, :, kband],
        'atoms': at, 'enames': np.array(en), 'sites': sites,
    }


def gpaw_window_pick(eps, lo, hi, side, tag):
    """(k,n) mask with eps in [lo,hi); empty -> frontier state on `side` per k.
    Same semantics as select_window but on GPAW eps_kn [nk,nb] (eV)."""
    pick = (eps >= lo) & (eps < hi)
    if not np.any(pick):
        pick = np.zeros(eps.shape, dtype=bool)
        for ik in range(eps.shape[0]):
            m = np.where(eps[ik] < hi)[0] if side == 'occ' else np.where(eps[ik] >= lo)[0]
            if m.size == 0:
                raise RuntimeError(f'{tag}: k{ik} no state on {side} side of [{lo:.2f},{hi:.2f}]')
            pick[ik, m[np.argmax(eps[ik, m])] if side == 'occ' else m[np.argmin(eps[ik, m])]] = True
        print(f'  [window] {side} empty; frontier state at each k', flush=True)
    return pick


def run_case_gpaw(chem, width, state, wins):
    """GPAW hs.npz + all.gpw -> same cache dict as run_case (TH-LDOS + unfold).

    rho = sum_k 2 w_k |psi_kn|^2 on the real-space grid z-slice at the molecular
    plane + Z_ABOVE (GPAW native grid, ~0.2 A spacing).  kpts already span the
    full BZ (job ran symmetry='off') -> no +-k mirroring.  Unfolding reuses
    subcell_group_indices/unfold_T_weights with the dzp AO count map.
    """
    from gpaw import GPAW
    from spammm.quantum import gpaw_hs
    tag = f'{TAG}_{chem}_r{width}_{state}'
    sdir = Path(GPAW_ROOT) / f'{TAG}_{chem}_r{width}' / state
    hs = gpaw_hs.load_hs(sdir / 'hs.npz')
    calc = GPAW(str(sdir / 'all.gpw'), txt=None)
    kpts, wk = np.asarray(hs['kpts_kc']), np.asarray(hs['w_k'])
    if not np.allclose(calc.get_ibz_k_points(), kpts, atol=1e-6):
        raise RuntimeError(f'{tag}: gpw kpts != hs.npz kpts')
    eps = np.asarray(hs['eps_kn'])                                # [nk,nb] eV
    if not np.allclose(calc.get_eigenvalues(kpt=0), eps[0], atol=1e-4):
        raise RuntimeError(f'{tag}: gpw eigenvalues != hs.npz eps_kn')
    ef, E = float(hs['E_fermi']), float(hs['E_total'])
    pos, cell = np.asarray(hs['pos_ac']), np.asarray(hs['cell_cv'])
    sym = [gpaw_hs.Z2E[int(z)] for z in hs['Z_a']]
    heavy = np.array([s != 'H' for s in sym])
    a1 = cell[0]
    ngx, ngy, ngz = calc.get_pseudo_wave_function(band=0, kpt=0).shape
    hz, hy = cell[2, 2] / ngz, cell[1, 1] / ngy
    iz = int(round((float(np.median(pos[heavy, 2])) + Z_ABOVE) / hz)) % ngz
    iy0 = max(0, int(np.floor((pos[heavy, 1].min() - 1.5) / hy)))
    iy1 = min(ngy, int(np.ceil((pos[heavy, 1].max() + 1.5) / hy)))
    extent = [0.0, float(a1[0]), iy0 * hy, iy1 * hy]
    rhos = {'occ': [], 'unocc': []}
    spans = {'occ': [], 'unocc': []}
    nst = {'occ': [], 'unocc': []}
    for wv in wins:
        for side, (lo, hi) in (('occ', (ef - wv, ef)), ('unocc', (ef, ef + wv))):
            pick = gpaw_window_pick(eps, lo, hi, side, tag)
            rho = np.zeros((ngx, iy1 - iy0))
            es = []
            for ik, n in zip(*np.nonzero(pick)):
                psi = calc.get_pseudo_wave_function(band=int(n), kpt=int(ik))
                rho += 2.0 * wk[ik] * np.abs(psi[:, iy0:iy1, iz]) ** 2
                es.append(eps[ik, n])
            if not np.isfinite(rho).all() or float(rho.max()) <= 0.0:
                raise RuntimeError(f'{tag}/{side} w={wv}: LDOS max={float(rho.max())}')
            rhos[side].append(rho.T.copy())                       # (ny,nx) like plane()
            spans[side].append((float(min(es)), float(max(es))))
            nst[side].append(len(es))
    calc = None                                                   # drop wfs before unfold

    # unfolded bands: C_knM [nk,nb,nao] -> [nk,nao,nb]; dzp AO map; full BZ (no mirror)
    atoms_i, lvs = ideal_geom(chem, width, state)
    IDX = subcell_group_indices(np.asarray(atoms_i.apos), np.asarray(lvs), list(atoms_i.enames),
                                NCELLS, nao=NAO_DZP, strict=False)
    q, W = unfold_T_weights(np.ascontiguousarray(hs['C_knM'].transpose(0, 2, 1)),
                            hs['S_kMM'], kpts[:, 0], IDX, NCELLS)
    ev = eps                                                       # already eV
    nkb = ev.shape[1]
    qq = np.tile(q[:, :, None], (1, 1, nkb)).ravel()
    ee = np.tile(ev[:, None, :], (1, NCELLS, 1)).ravel()
    ww = W.ravel()
    G = 2.0 * np.pi / (float(a1[0]) / NCELLS)
    kx = ((qq % 1.0 + 0.5) % 1.0 - 0.5) * G
    keep = (np.abs(ee - ef) < BAND_WIN) & (ww > 0.02)
    kband = (np.abs(ev - ef) < BAND_WIN + 0.5).any(axis=0)
    a0, _ = ideal_geom(chem, width, '0H')
    sites = switched_sites_xy(a0, atoms_i, lvs)
    at, en = overlay(sym, pos, a1, extent, nrep=1)
    return {
        'rho_occ': np.stack(rhos['occ']), 'rho_unocc': np.stack(rhos['unocc']),
        'extent': np.array(extent), 'wins': np.asarray(wins, float),
        'span_occ': np.array(spans['occ']), 'span_unocc': np.array(spans['unocc']),
        'nst_occ': np.array(nst['occ']), 'nst_unocc': np.array(nst['unocc']), 'ef': ef, 'E': E,
        'kx': kx[keep], 'be': ee[keep], 'bw': ww[keep], 'G': G,
        'qA': q, 'eA': ev[:, kband], 'wA': W[:, :, kband],
        'atoms': at, 'enames': np.array(en), 'sites': sites,
    }


def run_primitive_gpaw(chem, width, nkx):
    """Primitive-cell (x1) GPAW bands of the pristine 0H ribbon -> reference lines.

    Same builder as run_primitive; GPAW settings mirror the cluster job.py
    (lcao/dzp/PBE, non-gamma mesh, symmetry off).  Full BZ -> no mirroring.
    """
    from gpaw import GPAW
    from ase import Atoms
    from ase.io import write as ase_write
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    from spammm.topology.ribbon_pbc import EDGE_SWITCH, A_CC
    tag = f'{TAG}_{chem}_r{width}_prim'
    b = MoleculeEditorBackend(a_CC=A_CC)
    base = EDGE_SWITCH[chem][0]
    b.build_zigzag_ribbon(width_chains=2 * (width + 1), length_cells=1,
                          passivation_bottom=base, passivation_top=base, bPeriodicX=True)
    b._sync_sys()
    apos = np.asarray(b.sys.apos, float).copy()
    hvy = np.array([e.split('_')[0] != 'H' for e in b.sys.enames])
    h = apos[hvy, 1].ptp()
    apos[:, 1] -= apos[hvy, 1].min() - 6.0
    apos[:, 2] += 10.0                                            # off the z boundary (Lz=20)
    lvs = np.array([[2.0 * A_CC * np.cos(np.pi / 6.0), 0.0, 0.0], [0.0, h + 12.0, 0.0], [0.0, 0.0, 20.0]])
    wd = WORK / tag
    wd.mkdir(parents=True, exist_ok=True)
    atoms = Atoms(symbols=[e.split('_')[0] for e in b.sys.enames], positions=apos,
                  cell=lvs, pbc=[True, True, False])
    ase_write(str(wd / 'geom.xyz'), atoms)
    atoms.calc = GPAW(mode='lcao', xc='PBE', basis='dzp',
                      kpts={'size': (nkx, 1, 1), 'gamma': False},
                      txt=str(wd / 'gpaw.out'), symmetry='off')
    E = atoms.get_potential_energy()
    kc = atoms.calc.get_ibz_k_points()
    ev = np.array([atoms.calc.get_eigenvalues(kpt=ik) for ik in range(len(kc))])
    ef = atoms.calc.get_fermi_level()
    G = 2.0 * np.pi / float(lvs[0, 0])
    return {'pkx': kc[:, 0] * G, 'pev': ev, 'pef': ef, 'pE': E, 'G': G}


def recompute_weights_gpaw(chem, width, state):
    """Recompute only the unfold arrays (qA/eA/wA + kx/be/bw) of an existing
    cache from hs.npz — no all.gpw access.  Used to swap the weight formula
    (S-metric projector) without re-extracting wavefunctions."""
    from spammm.quantum import gpaw_hs
    tag = f'{TAG}_{chem}_r{width}_{state}'
    npz = WORK / f'{tag}.npz'
    hs = gpaw_hs.load_hs(Path(GPAW_ROOT) / f'{TAG}_{chem}_r{width}' / state / 'hs.npz')
    atoms_i, lvs = ideal_geom(chem, width, state)
    IDX = subcell_group_indices(np.asarray(atoms_i.apos), np.asarray(lvs), list(atoms_i.enames),
                                NCELLS, nao=NAO_DZP, strict=False)
    q, W = unfold_T_weights(np.ascontiguousarray(hs['C_knM'].transpose(0, 2, 1)),
                            hs['S_kMM'], np.asarray(hs['kpts_kc'])[:, 0], IDX, NCELLS)
    ev, ef = np.asarray(hs['eps_kn']), float(hs['E_fermi'])
    d = dict(np.load(npz, allow_pickle=True))
    nkb = ev.shape[1]
    qq = np.tile(q[:, :, None], (1, 1, nkb)).ravel()
    ee = np.tile(ev[:, None, :], (1, NCELLS, 1)).ravel()
    G = float(d['G'])
    kx = ((qq % 1.0 + 0.5) % 1.0 - 0.5) * G
    keep = (np.abs(ee - ef) < BAND_WIN) & (W.ravel() > 0.02)
    kband = (np.abs(ev - ef) < BAND_WIN + 0.5).any(axis=0)
    d['kx'], d['be'], d['bw'] = kx[keep], ee[keep], W.ravel()[keep]
    d['qA'], d['eA'], d['wA'] = q, ev[:, kband], W[:, :, kband]
    np.savez_compressed(npz, **d)
    print(f'[re-w] {tag}  Wmax med {np.median(W.max(axis=1)):.3f}  '
          f'frac>0.9 {(W.max(axis=1) > 0.9).mean():.2f}', flush=True)


def recompute_weights_dftb(dftb, chem, width, state):
    """DFTB analog of recompute_weights_gpaw: rerun the eig solve in the
    existing work dir, recompute unfold arrays with T-spectral weights in the
    true S(k) metric, patch the cache.  Skips LDOS projection entirely."""
    tag = f'{TAG}_{chem}_r{width}_{state}'
    npz, wd = WORK / f'{tag}.npz', WORK / tag
    cwd0 = os.getcwd()
    os.chdir(wd)
    try:
        energy, C, E, kpts, kw, Sk = run_dftb_hs(dftb, 'dftb_in.hsd')
        ef = fermi_ev(wd / 'detailed.out')
    finally:
        os.chdir(cwd0)
    atoms_i, lvs = ideal_geom(chem, width, state)
    IDX = subcell_group_indices(np.asarray(atoms_i.apos), np.asarray(lvs), list(atoms_i.enames),
                                NCELLS, strict=False)
    q, W = unfold_T_weights(np.ascontiguousarray(C.transpose(0, 2, 1)), Sk, kpts[:, 0], IDX, NCELLS)
    ev = E * HA2EV
    d = dict(np.load(npz, allow_pickle=True))
    nkb = ev.shape[1]
    qq = np.tile(q[:, :, None], (1, 1, nkb)).ravel()
    ee = np.tile(ev[:, None, :], (1, NCELLS, 1)).ravel()
    ww = W.ravel()
    qq = np.concatenate([qq, (-qq) % 1.0])                       # half-BZ -> full (same as run_case)
    ee = np.concatenate([ee, ee])
    ww = np.concatenate([ww, ww])
    G = float(d['G'])
    kx = ((qq % 1.0 + 0.5) % 1.0 - 0.5) * G
    keep = (np.abs(ee - ef) < BAND_WIN) & (ww > 0.02)
    kband = (np.abs(ev - ef) < BAND_WIN + 0.5).any(axis=0)
    d['kx'], d['be'], d['bw'] = kx[keep], ee[keep], ww[keep]
    d['qA'], d['eA'], d['wA'] = q, ev[:, kband], W[:, :, kband]
    np.savez_compressed(npz, **d)
    print(f'[re-w] {tag}  Wmax med {np.median(W.max(axis=1)):.3f}  '
          f'frac>0.9 {(W.max(axis=1) > 0.9).mean():.2f}', flush=True)


def prim_from_unfold(c0):
    """Primitive-cell reference bands folded back from the relaxed-0H supercell —
    no extra DFT run.  Each eigenstate is assigned its dominant primitive
    channel j* = argmax_m W[ik,m,n] (T-spectral projector weights) and
    contributes one exact (q_j*, eps_n) point — these ARE the true primitive
    Bloch eigenvalues at the nk*ncells sampled q's.

    Connectivity for line drawing: greedy band-tracking across the sorted
    q-columns — each column's energies are matched to open tracks by nearest
    energy (|dE| <= TOL); unmatched energies open new tracks, unmatched tracks
    close.  Lines only interpolate measured eigenvalues, but may swap identity
    at exact crossings (visually harmless: bands cross anyway)."""
    q, ev, W = np.asarray(c0['qA']), np.asarray(c0['eA']), np.asarray(c0['wA'])
    G, ef = float(c0['G']), float(c0['ef'])
    nk, nc, nb = W.shape
    TOL = 0.30                                                        # max |dE| per column step [eV]
    jstar = W.argmax(axis=1)                                            # [nk,nb]
    cols = {}                                                           # kx -> [eps]
    for ik in range(nk):
        for n in range(nb):
            x = ((q[ik, jstar[ik, n]] % 1.0 + 0.5) % 1.0 - 0.5) * G
            cols.setdefault(round(float(x), 9), []).append(float(ev[ik, n]))
    open_tr, done = [], []                                              # tracks = [xs, ys, last_e]
    for x in sorted(cols):
        es = sorted(cols[x])
        cand = sorted((abs(e - t[2]), i, e) for i, t in enumerate(open_tr) for e in es)
        ti_used, e_used = set(), set()
        for d, i, e in cand:
            if d > TOL:
                break
            if i in ti_used or e in e_used:
                continue
            ti_used.add(i); e_used.add(e)
            open_tr[i][0].append(x); open_tr[i][1].append(e); open_tr[i][2] = e
        done += [t for i, t in enumerate(open_tr) if i not in ti_used]
        open_tr = [t for i, t in enumerate(open_tr) if i in ti_used]
        open_tr += [[[x], [e], e] for e in es if e not in e_used]
    done += open_tr
    plines = [(np.asarray(t[0]), np.asarray(t[1])) for t in done if len(t[0]) > 1]
    qq = np.array([x for x in sorted(cols) for _ in cols[x]])
    ee = np.array([e for x in sorted(cols) for e in sorted(cols[x])])
    return {'pkx': qq, 'pev': ee[:, None], 'pef': ef, 'pE': np.nan, 'G': G,
            'plines': plines}


def unfold_segments(c, G):
    """Structured unfold arrays -> LineCollection (segments, alphas).

    Connectivity: fixed primitive channel j and band index, sorted along kx.
    Per-segment alpha = max endpoint weight (0..1); drawn for +/-kx (time reversal).
    """
    qA, eA, wA = c['qA'], c['eA'], c['wA']                       # [nk,nc], [nk,nb], [nk,nc,nb]
    nk, nc, nb = wA.shape
    segs, al = [], []
    for j in range(nc):
        xs0 = ((qA[:, j] % 1.0 + 0.5) % 1.0 - 0.5) * G          # centered primitive BZ
        o = np.argsort(xs0)
        xs = xs0[o]
        for b in range(nb):
            m = (np.abs(eA[:, b] - c['ef']) < BAND_WIN)
            if not m.any():
                continue
            ys = eA[o, b] - c['ef']
            ws = wA[o, j, b]
            wm = np.maximum(ws[:-1], ws[1:])                  # segment weight = brighter end
            good = wm > 0.02
            if not good.any():
                continue
            p = np.empty((int(good.sum()), 2, 2))
            p[:, 0, 0] = xs[:-1][good]; p[:, 0, 1] = ys[:-1][good]
            p[:, 1, 0] = xs[1:][good];  p[:, 1, 1] = ys[1:][good]
            pm = p.copy(); pm[:, :, 0] *= -1.0                # time-reversal mirror
            segs += [p, pm]
            al += [wm[good], wm[good]]
    return np.concatenate(segs), np.concatenate(al)


def bands_figure(chem, width, caches, prim, win):
    """One row per state: unfolded bands as alpha-weighted line segments
    (alpha = unfold weight) + pristine primitive bands (black lines).
    Also emits a pixelated spectral-density variant *_bands_pix.png."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    pkx, pev, pef = prim['pkx'], prim['pev'], prim['pef']
    order = np.argsort(pkx)
    pkx, pev = pkx[order], pev[order]
    G = float(prim['G'])

    def alpha_lc(c, base):
        segs, al = unfold_segments(c, G)
        lc = LineCollection(segs, linewidths=1.3, capstyle='round', zorder=3)
        rgba = np.tile(np.array(base, float), (len(al), 1))
        rgba[:, 3] = np.clip(al, 0.0, 1.0) ** 0.7             # slight gamma: keep faint bands visible
        lc.set_color(rgba)
        return lc, al

    fig, axes = plt.subplots(len(STATES), 1, figsize=(9.5, 2.15 * len(STATES)), sharex=True)
    ref = caches['0H']
    for ax, st in zip(axes, STATES):
        c = caches[st]
        if 'plines' in prim:
            for xs_, ys_ in prim['plines']:
                ax.plot(xs_, ys_ - pef, c='tab:green', lw=0.6, zorder=1)
        else:
            for nb in range(pev.shape[1]):
                ax.plot(pkx, pev[:, nb] - pef, c='tab:green', lw=0.5, zorder=1)
        lc0, _ = alpha_lc(ref, (0.55, 0.55, 0.55, 1.0))
        ax.add_collection(lc0)
        lc1, al1 = alpha_lc(c, (0.85, 0.05, 0.05, 1.0))       # red, alpha = weight
        ax.add_collection(lc1)
        ax.axhspan(-win, 0.0, color='tab:blue', alpha=0.10, lw=0)
        ax.axhspan(0.0, win, color='tab:orange', alpha=0.10, lw=0)
        ax.axhline(0.0, color='0.4', lw=0.7)
        ax.set_ylabel('E−EF [eV]', fontsize=8)
        ax.set_ylim(-BAND_WIN, BAND_WIN)
        ax.tick_params(labelsize=7)
        frac = float((c['bw'] > 0.5).mean())
        glab = '0H->x1 folded bands (green)' if 'plines' in prim else 'pristine x1 bands (green)'
        ax.set_title(f'{st}   unfolded (red, alpha=W; sharp {frac * 100:.0f}%)  '
                     f'vs  {glab} + relaxed 0H unfold (grey)', fontsize=8, loc='left')
    axes[-1].set_xlabel('kx [1/Å]  primitive BZ')
    axes[-1].set_xlim(-G / 2, G / 2)
    fig.suptitle(f'{TAG}_{chem} r{width}  {METHOD}  x{NCELLS} supercell unfolded -> primitive BZ; '
                 f'shaded = STM windows ±{win} eV', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    path = OUT / f'{TAG}_{chem}_r{width}_bands.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[plot] {path}', flush=True)

    # pixelated variant: max unfold weight per (kx,E) bin, Gaussian-smoothed
    # -> spectral-function-like density; vmax = p99.5 for contrast.
    # nkx_ = 2x the number of sampled q's — much wider bins leave empty columns
    # between the sparse GPAW samples that Gaussian smoothing dilutes to a wash.
    from scipy.ndimage import gaussian_filter
    from scipy.stats import binned_statistic_2d
    nkx_ = 2 * int(caches[STATES[0]]['qA'].size)
    ne_ = 400
    fig, axes = plt.subplots(len(STATES), 1, figsize=(9.5, 2.15 * len(STATES)), sharex=True)
    hss = []
    for st in STATES:
        c = caches[st]
        H, _, _, _ = binned_statistic_2d(c['kx'], c['be'] - c['ef'], c['bw'], statistic='max',
                                         bins=(nkx_, ne_), range=[[-G / 2, G / 2], [-BAND_WIN, BAND_WIN]])
        hss.append(gaussian_filter(np.nan_to_num(H.T), sigma=(1.6, 1.2)))
    vmax = np.percentile(np.concatenate([h[h > 0] for h in hss]), 99.7)
    for ax, st, Hs in zip(axes, STATES, hss):
        ax.set_facecolor('white')
        im = ax.imshow(np.ma.masked_less(Hs, 1e-4), cmap='gist_heat_r', vmin=0, vmax=vmax,
                       origin='lower', aspect='auto', extent=[-G / 2, G / 2, -BAND_WIN, BAND_WIN],
                       interpolation='bilinear', zorder=2)
        if 'plines' in prim:
            for xs_, ys_ in prim['plines']:
                ax.plot(xs_, ys_ - pef, c='tab:green', lw=0.6, zorder=3)
        else:
            for nb in range(pev.shape[1]):
                ax.plot(pkx, pev[:, nb] - pef, c='tab:green', lw=0.5, zorder=3)
        ax.axhspan(-win, 0.0, color='tab:blue', alpha=0.08, lw=0, zorder=1)
        ax.axhspan(0.0, win, color='tab:orange', alpha=0.08, lw=0, zorder=1)
        ax.axhline(0.0, color='0.4', lw=0.7, zorder=4)
        ax.set_ylabel('E−EF [eV]', fontsize=8)
        ax.set_ylim(-BAND_WIN, BAND_WIN)
        ax.tick_params(labelsize=7)
        ax.set_title(f'{st}   unfolded spectral density (max W per bin)', fontsize=8, loc='left')
    axes[-1].set_xlabel('kx [1/Å]  primitive BZ')
    axes[-1].set_xlim(-G / 2, G / 2)
    fig.suptitle(f'{TAG}_{chem} r{width}  {METHOD}  x{NCELLS} supercell unfolded -> primitive BZ '
                 f'(pixelated max, vmax=p99.5)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.95, 0.985])
    fig.colorbar(im, ax=axes, label='max W per bin (shared vmax)', fraction=0.03, pad=0.01)
    path2 = OUT / f'{TAG}_{chem}_r{width}_bands_pix.png'
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f'[plot] {path2}', flush=True)
    return [path, path2]


def ldos_figure(chem, width, caches, wins):
    """(2*nw rows: occ windows then unocc windows) x states; shared norm per row
    so states are comparable within a window. lin / sat5 / log variants."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    tag = f'{TAG}_{chem}_r{width}'
    wins = [float(w) for w in wins]
    nw = len(wins)
    rows = [('occ', i) for i in range(nw)] + [('unocc', i) for i in range(nw)]
    vmax_row = {}
    for side, iw in rows:
        vmax_row[(side, iw)] = max(float(np.max(caches[st][f'rho_{side}'][iw])) for st in STATES)
        if not np.isfinite(vmax_row[(side, iw)]) or vmax_row[(side, iw)] <= 0.0:
            raise RuntimeError(f'{tag} {side} w={wins[iw]}: shared vmax={vmax_row[(side, iw)]}')
    variants = [('sat5', 'viridis', 'lin', 0.2), ('log', 'gnuplot2', 'log', 1.0),
                ('sqrt', 'viridis', 'sqrt', 0.5)]
    paths = []
    for vname, cmap, mode, vmaxf in variants:
        fig, axes = plt.subplots(2 * nw, len(STATES),
                                 figsize=(2.7 * len(STATES), 2.4 * 2 * nw), squeeze=False)
        for irow, (side, iw) in enumerate(rows):
            vmax = vmax_row[(side, iw)]
            for ax, st in zip(axes[irow], STATES):
                c = caches[st]
                rho = np.asarray(c[f'rho_{side}'][iw])
                if mode == 'log':
                    ax.imshow(np.clip(rho, vmax * 1e-3, None), origin='lower', cmap=cmap,
                              extent=list(c['extent']), interpolation='nearest',
                              norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax))
                elif mode == 'sqrt':
                    ax.imshow(np.sqrt(rho), origin='lower', cmap=cmap, extent=list(c['extent']),
                              interpolation='nearest', vmin=0.0, vmax=np.sqrt(vmax) * vmaxf)
                else:
                    ax.imshow(rho, origin='lower', cmap=cmap, extent=list(c['extent']),
                              interpolation='nearest', vmin=0.0, vmax=vmax * vmaxf)
                scatter_atoms(ax, c['atoms'], list(c['enames']), s=8)
                if len(c['sites']):
                    ax.plot(c['sites'][:, 0], c['sites'][:, 1], '+', color='lime', ms=5, mew=1.0, zorder=11)
                lo, hi = c[f'span_{side}'][iw]
                ax.set_title(f"{st}  {lo:.2f}..{hi:.2f} eV  n={c[f'nst_{side}'][iw]}\n"
                             f"max {float(np.max(rho)):.3g}", fontsize=7)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=6)
            lab = {'occ': f'occ EF−{wins[iw]}..EF', 'unocc': f'unocc EF..EF+{wins[iw]}'}[side]
            axes[irow, 0].set_ylabel(lab + '   y [Å]', fontsize=8)
        mode = {'sat5': 'vmax=rowmax/5 saturated', 'log': 'log10 3 decades',
                'sqrt': 'sqrt(rho) ~ |psi|, vmax=rowmax/2'}[vname]
        fig.suptitle(f'{tag}  {METHOD}  Σ 2w_k|ψ|²  z=+{Z_ABOVE:.1f} Å  '
                     f'[{mode}]  (green + = switched sites)', fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        path = OUT / f'{tag}_ldos_{vname}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f'[plot] {path}', flush=True)
        paths.append(path)
    return paths


def ldos_cmp_figure(width, allc, chems, wins):
    """Cross-chem comparison: one figure per window, 6 rows = {chems} occ then
    {chems} unocc, columns = states.  Shared vmax per row.  Variants:
    sat5 (vmax=row/5), log (gnuplot2), sqrt (|psi|~sqrt(rho))."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    wins = [float(w) for w in wins]
    variants = [('sat5', 'viridis', 'lin', 0.2), ('log', 'gnuplot2', 'log', 1.0),
                ('sqrt', 'viridis', 'sqrt', 0.5)]
    paths = []
    crow = ('C', 'N', 'O')                                            # canonical row order; missing chems -> blank
    for iw, wv in enumerate(wins):
        rows = [(s, ch) for s in ('occ', 'unocc') for ch in crow]
        vmax_row = {}
        for side, ch in rows:
            if ch not in allc:
                continue
            vmax_row[(side, ch)] = max(float(np.max(allc[ch][st][f'rho_{side}'][iw])) for st in STATES)
            if not np.isfinite(vmax_row[(side, ch)]) or vmax_row[(side, ch)] <= 0.0:
                raise RuntimeError(f'r{width} {side} {ch} w={wv}: vmax={vmax_row[(side, ch)]}')
        for vname, cmap, mode, vmaxf in variants:
            fig, axes = plt.subplots(2 * len(crow), len(STATES),
                                     figsize=(2.7 * len(STATES), 2.4 * 2 * len(crow)), squeeze=False)
            for irow, (side, ch) in enumerate(rows):
                if ch not in allc:
                    for ax, st in zip(axes[irow], STATES):
                        ax.set_title(st, fontsize=7)
                        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center',
                                va='center', fontsize=8, color='0.5')
                        ax.set_xticks([]); ax.set_yticks([])
                    axes[irow, 0].set_ylabel(
                        f"{ch} {'occ EF−' if side == 'occ' else 'unocc EF+'}{wv}  y [Å]", fontsize=8)
                    continue
                vmax = vmax_row[(side, ch)]
                for ax, st in zip(axes[irow], STATES):
                    c = allc[ch][st]
                    rho = np.asarray(c[f'rho_{side}'][iw])
                    if mode == 'log':
                        ax.imshow(np.clip(rho, vmax * 1e-3, None), origin='lower', cmap=cmap,
                                  extent=list(c['extent']), interpolation='nearest',
                                  norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax))
                    elif mode == 'sqrt':
                        ax.imshow(np.sqrt(rho), origin='lower', cmap=cmap, extent=list(c['extent']),
                                  interpolation='nearest', vmin=0.0, vmax=np.sqrt(vmax) * vmaxf)
                    else:
                        ax.imshow(rho, origin='lower', cmap=cmap, extent=list(c['extent']),
                                  interpolation='nearest', vmin=0.0, vmax=vmax * vmaxf)
                    scatter_atoms(ax, c['atoms'], list(c['enames']), s=8)
                    if len(c['sites']):
                        ax.plot(c['sites'][:, 0], c['sites'][:, 1], '+', color='lime', ms=5, mew=1.0, zorder=11)
                    lo, hi = c[f'span_{side}'][iw]
                    ax.set_title(f"{st}  {lo:.2f}..{hi:.2f} eV  n={c[f'nst_{side}'][iw]}", fontsize=7)
                    ax.set_aspect('equal')
                    ax.tick_params(labelsize=6)
                axes[irow, 0].set_ylabel(
                    f"{ch} {'occ EF−' if side == 'occ' else 'unocc EF+'}{wv}  y [Å]", fontsize=8)
            fig.suptitle(f'{TAG} r{width}  {METHOD}  window ±{wv} eV  z=+{Z_ABOVE:.1f} Å  [{vname}]  '
                         f'(green + = switched sites)', fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.99])
            path = OUT / f'{TAG}_r{width}_ldoscmp_w{wv:g}_{vname}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            print(f'[plot] {path}', flush=True)
            paths.append(path)
    return paths


def write_index(figs):
    # *_bands.png (transparent alpha-line method) omitted: too messy — the
    # pixelated *_bands_pix.png is the canonical unfold view.
    figs = [f for f in figs if not f.name.endswith('_bands.png')]
    rows = '\n'.join(f'<p><a href="{f.name}">{f.name}</a><br><img src="{f.name}" width="1100"></p>'
                     for f in sorted(figs))
    (OUT / 'index.html').write_text(
        f'<html><body><h1>ribbon_stm_sys — near-EF TH-LDOS + unfolded bands</h1>'
        f'<p>{METHOD}, nkx={NKX}, windows ±{WINS} eV, z=+{Z_ABOVE} Å, dx={DX} Å. '
        f'bands: red/grey segments = unfolded channel weight W (subcell-translation spectral '
        f'projector in the true S(k) metric, exact for periodic states); '
        f'green = primitive reference (real x1 run lines, or 0H folded-back dots).</p>{rows}</body></html>')


def main():
    global NCELLS, TAG, STATES, STATE_STRS, WORK, OUT, GPAW_ROOT, METHOD
    ap = argparse.ArgumentParser()
    ap.add_argument('--chem', default='C,N,O')
    ap.add_argument('--widths', default=','.join(map(str, WIDTHS)))
    ap.add_argument('--states', default=None)
    ap.add_argument('--nkx', type=int, default=NKX)
    ap.add_argument('--wins', default=','.join(map(str, WINS)),
                    help='comma list of eV half-windows for TH-LDOS (each side of EF)')
    ap.add_argument('--dx', type=float, default=DX)
    ap.add_argument('--xscan', action='store_true',
                    help='separation-scan set (xscanv_* dirs, ncells=8, os/ss-dK states)')
    ap.add_argument('--tagp', default=None,
                    help='dataset dir prefix override (e.g. xscanz for the z-fixed tilted-OH set); '
                         'default xscanv with --xscan')
    ap.add_argument('--gpaw', default=None, metavar='ROOT',
                    help='GPAW mode: read hs.npz + all.gpw from ROOT/<TAG>_<chem>_r<w>/<state>/; '
                         'caches/figures go to debug/ribbon_gpaw/stm_work*/stm* (same layout)')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--re-w', dest='re_w', action='store_true',
                    help='GPAW: recompute unfold weights in existing caches from hs.npz '
                         '(no all.gpw), then continue to plotting')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.xscan:
        NCELLS, TAG = 8, (args.tagp or 'xscanv')
        STATE_STRS = xscan_state_strings(NCELLS)
        STATES = list(STATE_STRS)
    if args.gpaw:
        GPAW_ROOT = Path(args.gpaw)
        METHOD = 'GPAW/dzp'
        root = Path('/home/prokop/git/SPAMMM/debug/ribbon_gpaw')
        WORK = root / 'stm_work' / (TAG.replace('xscanv', 'xscan') if args.xscan else 'enumv')
        OUT = root / (f'stm_{TAG.replace("xscanv", "xscan")}' if args.xscan else 'stm')
    elif args.xscan:
        WORK = RIBBON_ROOT / 'stm_work' / TAG.replace('xscanv', 'xscan')
        OUT = RIBBON_ROOT / f'stm_{TAG.replace("xscanv", "xscan")}'
    if not args.gpaw and not WFC.is_file():
        raise RuntimeError(f'missing basis {WFC}')
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    chems = args.chem.split(',')
    widths = [int(x) for x in args.widths.split(',')]
    if args.gpaw:                                                  # no O data: keep only chems present on disk
        chems = [c for c in chems if any(
            (GPAW_ROOT / f'{TAG}_{c}_r{w}').is_dir() for w in widths)]
        if not chems:
            raise RuntimeError(f'no GPAW data dirs under {GPAW_ROOT} for {args.chem}')
    states = args.states.split(',') if args.states else list(STATES)
    wins = [float(x) for x in args.wins.split(',')]
    figs = []

    if not args.plot_only and not args.gpaw:
        (WORK / 'basis.hsd').write_text(
            'Basis {\n  Resolution = 0.1\n  <<+ "%s"\n}\n' % os.path.relpath(WFC, WORK))
        species = parse_basis_hsd_ang(str(WORK / 'basis.hsd'))
        by_name = {sp['name']: i for i, sp in enumerate(species)}
        basis = {'species': species}
        proj = DFTBplusGridProjector(verbosity=0)
        proj.load_basis_dftb(basis)
        dftb = DFTBcore()
    elif not args.plot_only:
        dftb = proj = basis = by_name = None
    else:
        dftb = proj = basis = by_name = None

    if not args.plot_only:
        for chem in chems:
            for w in widths:
                if not args.gpaw:
                    pnpz = WORK / f'{TAG}_{chem}_r{w}_prim.npz'
                    if not pnpz.is_file() or args.force:
                        d = run_primitive(dftb, chem, w, args.nkx * NCELLS)
                        np.savez_compressed(pnpz, **d)
                        print(f'[{TAG}_{chem}_r{w}_prim] E={d["pE"]:.6f}  EF={d["pef"]:.3f} eV', flush=True)
                for st in states:
                    tag = f'{TAG}_{chem}_r{w}_{st}'
                    npz = WORK / f'{tag}.npz'
                    if npz.is_file() and not args.force:
                        continue
                    if args.gpaw:
                        d = run_case_gpaw(chem, w, st, wins)
                    else:
                        d = run_case(dftb, proj, basis, by_name, chem, w, st, args.nkx, wins, args.dx)
                    np.savez_compressed(npz, **d)
                    print(f'[{tag}] E={d["E"]:.6f}  EF={d["ef"]:.3f} eV  '
                          f'occ n={d["nst_occ"].tolist()}  unocc n={d["nst_unocc"].tolist()}',
                          flush=True)

    if args.re_w:
        for chem in chems:
            for w in widths:
                for st in states:
                    npz = WORK / f'{TAG}_{chem}_r{w}_{st}.npz'
                    if not npz.is_file():
                        continue
                    if args.gpaw:
                        recompute_weights_gpaw(chem, w, st)
                    else:
                        if dftb is None:
                            dftb = DFTBcore()
                        recompute_weights_dftb(dftb, chem, w, st)

    for w in widths:
        allc, prims = {}, {}
        for chem in chems:
            caches = {}
            for st in STATES:
                npz = WORK / f'{TAG}_{chem}_r{w}_{st}.npz'
                if not npz.is_file():
                    raise RuntimeError(f'missing cache {npz} — run without --plot-only first')
                caches[st] = dict(np.load(npz, allow_pickle=True))
            allc[chem] = caches
            pnpz = WORK / f'{TAG}_{chem}_r{w}_prim.npz'
            prims[chem] = (dict(np.load(pnpz)) if pnpz.is_file()
                           else prim_from_unfold(allc[chem]['0H']) if args.gpaw
                           else dict(np.load(pnpz)))
        for chem in chems:
            figs.extend(bands_figure(chem, w, allc[chem], prims[chem], max(wins)))
            figs.extend(ldos_figure(chem, w, allc[chem], wins))
        if len(chems) > 1:
            figs.extend(ldos_cmp_figure(w, allc, chems, wins))
    write_index(figs)
    print(f'[plot] {OUT / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
