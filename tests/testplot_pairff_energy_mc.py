#!/usr/bin/env python3
"""Greedy planar MC assembly with PairFF energy_replica kernel + FAF substrate + packing well.

Energy-only GPU kernel (kernel 14, `rigid_body_pairff_energy_replica_kernel`) evaluates
PairFF + FAF for rigid molecular assemblies across many replicas; the host harness does
greedy best-of-batch acceptance. Supports 8 molecules (PTCDA, formic acid, terephthalic
acid, NTCDI, TBTAP, azaindol, uracil, adenine) with QEq charges, FAF substrate (NaCl) via
remapped folded-basis fits, charge-colored atom visualization, and animated GIF trajectory.
**Multi-species** runs (e.g. `--mol adenine,uracil`) place nmol copies of each species on
a shared grid with a single FAF fit covering all elements.

Design:
- **FAF = molecule↔surface only.** Molecules interact with each other via PairFF directly.
  Existing fits are reused for new molecules via `remap_fit_for_molecule` (REQ similarity).
- **Unified plotting** via `plot_assembly_on_faf` — every plot (PNG + GIF) shows FAF
  substrate heatmap + bonded skeleton (plotUtils.plotBonds) + charge-colored atoms
  (plotUtils.plotAtoms) + CoM markers. No separate styles.
- **Per-molecule subfolders** in `debug/testplot_pairff_energy_mc/<mol>/` so runs don't
  overwrite each other.
- **Bigger MC steps** (dxy=1.5 Å, dphi=0.8 rad) than the original 0.55/0.35 to escape
  local minima; 1000 steps default.

Caveats:
- Greedy MC still stalls in deep minima (21–75/1000 accepted); simulated annealing
  (task phase H2) is the proper fix — kernel already supports it, only harness acceptance
  logic needs changing.
- FAF type remapping uses (R,E,Q) similarity; for molecules with elements not in any fit
  (e.g. N in formic acid), the closest type is picked — may be less accurate than a
  molecule-specific fit.
- No pytest L0 regression yet (only testplot demo); parity check runs inline.

Usage:
  python3 tests/testplot_pairff_energy_mc.py
  python3 tests/testplot_pairff_energy_mc.py --mol NTCDI --steps 1000 --ntrial 512
  python3 tests/testplot_pairff_energy_mc.py --mol formic_acid --nmol 6 --spacing 8

Artifacts → debug/testplot_pairff_energy_mc/<mol>/
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from spammm import elements
from spammm.AtomicSystem import AtomicSystem
from spammm.forcefields.QEq import solve_from_elements
from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF, _body_sites_world
from spammm.plotUtils import plotAtoms, plotBonds
from spammm.surfaces.FoldedRigid import load_fit, eval_folded_potential_grid, Z_SURF_TOP
from spammm.topology.FFparams import (
    read_atom_types, read_element_types, make_REQs_from_enames, load_xyz_with_REQs, _DATA_PATH,
)

OUT_ROOT = os.path.join(REPO, 'debug', 'testplot_pairff_energy_mc')
NTCDI = os.path.join(REPO, 'data', 'mol', 'NTCDI.mol2')
PTCDA = os.path.join(REPO, 'data', 'xyz', 'PTCDA.xyz')
HCOOH = os.path.join(REPO, 'data', 'xyz', 'HCOOH.xyz')
TEREPHTHALIC = os.path.join(REPO, 'data', 'xyz', 'terephthalic_acid.xyz')
TBTAP = os.path.join(REPO, 'data', 'mol', 'TBTAP.mol2')
AZAINDOL = os.path.join(REPO, 'data', 'xyz', 'azaindol.xyz')
URACIL = os.path.join(REPO, 'data', 'xyz', 'uracil.xyz')
ADENINE = os.path.join(REPO, 'data', 'xyz', 'adenine.xyz')
FAF_FITS = {
    'formic_acid':       os.path.join(REPO, 'data', 'fits', 'hcooh_nacl.npz'),
    'PTCDA':             os.path.join(REPO, 'data', 'fits', 'ptcda_nacl.npz'),
    'NTCDI':             os.path.join(REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,O,H — same elements
    'terephthalic_acid': os.path.join(REPO, 'data', 'fits', 'ptcda_nacl.npz'),  # C,O,H — same elements
    'TBTAP':             os.path.join(REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,O,H — Br maps to closest
    'azaindol':          os.path.join(REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,H — same elements
    'uracil':            os.path.join(REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,O,H — same elements
    'adenine':           os.path.join(REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,H — same elements
}
# Default fit for multi-species runs (must cover all elements)
FAF_FIT_DEFAULT = os.path.join(REPO, 'data', 'fits', 'ptcdi_nacl.npz')  # C,N,O,H — broadest coverage


def _atom_types_dict():
    etypes = read_element_types(os.path.join(_DATA_PATH, 'ElementTypes.dat'))
    return etypes, read_atom_types(os.path.join(_DATA_PATH, 'AtomTypes.dat'), etypes)


def remap_fit_for_molecule(fit, mol_REQs):
    """Remap a FAF fit's atom types for a different molecule by (R,E,Q) similarity.

    The fit's coeffs/basis_params/lvec2d are reused; only atom_type_ids is rebuilt
    so each real atom picks the closest type in the fit's unique_REQs.
    """
    unique = np.asarray(fit['unique_REQs'], dtype=np.float64)
    reqs = np.asarray(mol_REQs, dtype=np.float64)[:, :4]
    scale = np.array([1.0, 20.0, 3.0, 0.0])  # R~1, E~0.05→1, Q~0.3→1, w ignored
    atids = np.zeros(len(reqs), dtype=np.int32)
    for i, r in enumerate(reqs):
        d = np.sqrt(((unique[:, :3] - r[:3]) * scale[:3])**2).sum(axis=1)
        atids[i] = int(np.argmin(d))
    fit2 = dict(fit)
    fit2['atom_type_ids'] = atids
    return fit2


def load_ntcdi():
    etypes, atom_types = _atom_types_dict()
    mol = AtomicSystem(fname=NTCDI)
    apos = np.asarray(mol.apos, dtype=np.float32)
    enames = [str(e) for e in mol.enames]
    qs = np.asarray(mol.qs, dtype=np.float32) if mol.qs is not None else np.zeros(len(enames), np.float32)
    REQs = make_REQs_from_enames(enames, qs, atom_types)
    apos[:, :2] -= apos[:, :2].mean(axis=0)
    apos[:, 2] = 0.0
    bonds = _bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def load_ptcda(qeq=True):
    """PTCDA with optional QEq charges (needed for O↔H quadrupole / windmill)."""
    etypes, atom_types = _atom_types_dict()
    apos, REQs, enames, Zs, lvec = load_xyz_with_REQs(PTCDA, atom_types=atom_types)
    apos = np.asarray(apos, dtype=np.float32)
    enames = [str(e) for e in enames]
    apos[:, :2] -= apos[:, :2].mean(axis=0)
    apos[:, 2] = 0.0
    if qeq:
        # Physical partial charges = negate electron-occupancy QEq (PairFF audit SSOT)
        q = -solve_from_elements(apos, enames, etypes, Q_target=0.0)
        REQs = make_REQs_from_enames(enames, q.astype(np.float32), atom_types)
        print(f'  QEq: sum={q.sum():.4f}  O=[{q[[i for i,e in enumerate(enames) if e=="O"]].min():.3f},'
              f'{q[[i for i,e in enumerate(enames) if e=="O"]].max():.3f}]  '
              f'H=[{q[[i for i,e in enumerate(enames) if e=="H"]].min():.3f},'
              f'{q[[i for i,e in enumerate(enames) if e=="H"]].max():.3f}]')
    bonds = _bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def load_formic_acid(qeq=True):
    """Formic acid (HCOOH) with optional QEq charges."""
    etypes, atom_types = _atom_types_dict()
    apos, REQs, enames, Zs, lvec = load_xyz_with_REQs(HCOOH, atom_types=atom_types)
    apos = np.asarray(apos, dtype=np.float32)
    enames = [str(e) for e in enames]
    apos[:, :2] -= apos[:, :2].mean(axis=0)
    apos[:, 2] = 0.0
    if qeq:
        q = -solve_from_elements(apos, enames, etypes, Q_target=0.0)
        REQs = make_REQs_from_enames(enames, q.astype(np.float32), atom_types)
        print(f'  QEq: sum={q.sum():.4f}  Q per atom: {np.round(q,3)}')
    bonds = _bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def load_terephthalic_acid(qeq=True):
    """Terephthalic acid (C8H6O4) with optional QEq charges."""
    etypes, atom_types = _atom_types_dict()
    apos, REQs, enames, Zs, lvec = load_xyz_with_REQs(TEREPHTHALIC, atom_types=atom_types)
    apos = np.asarray(apos, dtype=np.float32)
    enames = [str(e) for e in enames]
    apos[:, :2] -= apos[:, :2].mean(axis=0)
    apos[:, 2] = 0.0
    if qeq:
        q = -solve_from_elements(apos, enames, etypes, Q_target=0.0)
        REQs = make_REQs_from_enames(enames, q.astype(np.float32), atom_types)
        print(f'  QEq: sum={q.sum():.4f}  O=[{q[[i for i,e in enumerate(enames) if e=="O"]].min():.3f},'
              f'{q[[i for i,e in enumerate(enames) if e=="O"]].max():.3f}]')
    bonds = _bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def _load_xyz_generic(path, qeq=True, name=''):
    """Generic XYZ loader with QEq charges, centering, and planarization."""
    etypes, atom_types = _atom_types_dict()
    apos, REQs, enames, Zs, lvec = load_xyz_with_REQs(path, atom_types=atom_types)
    apos = np.asarray(apos, dtype=np.float32)
    enames = [str(e) for e in enames]
    apos[:, :2] -= apos[:, :2].mean(axis=0)
    apos[:, 2] = 0.0
    if qeq:
        q = -solve_from_elements(apos, enames, etypes, Q_target=0.0)
        REQs = make_REQs_from_enames(enames, q.astype(np.float32), atom_types)
        print(f'  {name} QEq: sum={q.sum():.4f}  Q range=[{q.min():.3f},{q.max():.3f}]')
    bonds = _bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def _load_mol2_generic(path, qeq=True, name=''):
    """Generic mol2 loader via AtomicSystem with QEq charges."""
    etypes, atom_types = _atom_types_dict()
    mol = AtomicSystem(fname=path)
    apos = np.asarray(mol.apos, dtype=np.float32)
    enames = [str(e) for e in mol.enames]
    apos[:, :2] -= apos[:, :2].mean(axis=0)
    apos[:, 2] = 0.0
    if qeq:
        q = -solve_from_elements(apos, enames, etypes, Q_target=0.0)
    else:
        q = np.asarray(mol.qs, dtype=np.float32) if mol.qs is not None else np.zeros(len(enames), np.float32)
    REQs = make_REQs_from_enames(enames, q.astype(np.float32), atom_types)
    print(f'  {name} QEq: sum={q.sum():.4f}  Q range=[{q.min():.3f},{q.max():.3f}]  atoms={len(enames)}  elems={sorted(set(enames))}')
    bonds = _bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def load_uracil(qeq=True):     return _load_xyz_generic(URACIL, qeq=qeq, name='uracil')
def load_adenine(qeq=True):    return _load_xyz_generic(ADENINE, qeq=qeq, name='adenine')
def load_azaindol(qeq=True):   return _load_xyz_generic(AZAINDOL, qeq=qeq, name='azaindol')
def load_tbtap(qeq=True):      return _load_mol2_generic(TBTAP, qeq=qeq, name='TBTAP')


def _bonds_from_geom(apos, enames):
    atypes = [elements.ELEMENT_DICT[e][0] if e in elements.ELEMENT_DICT else 6 for e in enames]
    mol = AtomicSystem(apos=np.asarray(apos, dtype=np.float32).copy(), atypes=atypes, enames=list(enames))
    mol.neighs(bBond=True)
    if mol.bonds is None or len(mol.bonds) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(mol.bonds, dtype=np.int32)


def grid_pos(n, spacing, z=0.0):
    nx = int(np.ceil(np.sqrt(n)))
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        ix, iy = i % nx, i // nx
        pos[i, 0] = (ix - 0.5 * (nx - 1)) * spacing
        pos[i, 1] = (iy - 0.5 * (nx - 1)) * spacing
        pos[i, 2] = z
    return pos


def assembly_real_atoms(packs, pos, quat, bonds0):
    """Concatenate real-atom world frames + replicated intramolecular bonds.

    bonds0: single (N,2) array (reused for all packs, single-species) or list of
    (N_i,2) arrays (one per pack, multi-species).
    """
    worlds, enames, bonds = [], [], []
    off = 0
    bonds_list = bonds0 if isinstance(bonds0, (list, tuple)) else [bonds0] * len(packs)
    for j, pack in enumerate(packs):
        w = _body_sites_world(pack['rel'], pos[j], quat[j])
        m = pack['types'] == 0
        wr = w[m]
        er = [e for e, t in zip(pack['enames'], pack['types']) if t == 0]
        worlds.append(wr)
        enames.extend(er)
        b0 = bonds_list[j] if j < len(bonds_list) else bonds_list[0]
        if b0 is not None and len(b0):
            bonds.extend([(int(a) + off, int(b) + off) for a, b in b0])
        off += len(wr)
    return np.vstack(worlds).astype(np.float32), enames, np.asarray(bonds, dtype=np.int32)


def write_xyz(path, packs, pos, quat, comment=''):
    lines = []
    for j, pack in enumerate(packs):
        w = _body_sites_world(pack['rel'], pos[j], quat[j])
        for e, t, p in zip(pack['enames'], pack['types'], w):
            if t != 0:
                continue
            lines.append(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}')
    with open(path, 'w') as f:
        f.write(f'{len(lines)}\n{comment}\n')
        f.write('\n'.join(lines) + '\n')


def parity_check(rbd, pos, quat):
    E0 = rbd.eval_energy_system(pos, quat, k_pack=0.0)
    pos2 = pos.copy(); quat2 = quat.copy()
    pos2[0, 0] += 0.35
    half = 0.5 * 0.2
    dq = np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)
    quat2[0] = rbd._quat_normalize(rbd._quat_mul(quat[0], dq))
    E1 = rbd.eval_energy_system(pos2, quat2, k_pack=0.0)
    dE_full = E1 - E0
    nmol = rbd.n_bodies
    poss = np.zeros((2, nmol, 4), dtype=np.float32)
    qrots = np.zeros((2, nmol, 4), dtype=np.float32)
    for r, (p, q) in enumerate([(pos, quat), (pos2, quat2)]):
        poss[r, :, :3] = p
        qrots[r] = q
    Echan = rbd.eval_energy_replicas(poss, qrots, active_mols=[0])
    E_part = rbd.energy_changed(Echan)
    dE_part = float(E_part[1] - E_part[0])
    return dict(E0=E0, E1=E1, dE_full=dE_full, dE_part=dE_part, err=abs(dE_full - dE_part))


def plot_energy(hist, path, title):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(hist, lw=0.8, color='C0')
    ax.set_xlabel('MC step')
    ax.set_ylabel('E_tot + E_pack [eV]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_assembly_on_faf(ax, packs, pos, quat, bonds0, title, charges=None, fit=None, z_eval=None,
                          xlim=None, ylim=None, extent_pad=4.0, ngrid=120):
    """Unified plot: FAF substrate heatmap + bonded skeleton + charge-colored atoms + CoM markers.

    Uses plotUtils.plotBonds + plotUtils.plotAtoms (no reinvention).
    If fit is None, draws only bonds+atoms (vacuum).
    """
    apos, enames, bonds = assembly_real_atoms(packs, pos, quat, bonds0)
    # Axis extent
    all_xy = apos[:, :2]
    pad = extent_pad
    xmin, xmax = all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad
    ymin, ymax = all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad
    if xlim is not None: xmin, xmax = xlim
    if ylim is not None: ymin, ymax = ylim

    plt.sca(ax)
    # 1) FAF substrate heatmap (if available)
    if fit is not None and z_eval is not None:
        at_ids = np.asarray(fit['atom_type_ids'], dtype=np.int32)
        rep_type = int(np.bincount(at_ids).argmax())
        xs = np.linspace(xmin, xmax, ngrid)
        ys = np.linspace(ymin, ymax, ngrid)
        V = eval_folded_potential_grid(fit, rep_type, xs, ys, z_eval)
        im = ax.imshow(V, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='RdYlBu_r',
                       aspect='equal', interpolation='bilinear', alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=f'V_FAF(type {rep_type}) [eV]')

    # 2) Bonded skeleton
    if len(bonds):
        plotBonds(links=bonds, ps=apos, axes=(0, 1), colors='#333333', lws=0.8)

    # 3) Atoms — charge colored (red=negative, blue=positive) or element colored
    if charges is not None:
        q = np.asarray(charges)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=min(q.min(), -0.05), vcenter=0.0, vmax=max(q.max(), 0.05))
        cmap = plt.cm.RdBu_r
        colors = [cmap(norm(qi)) for qi in q]
        sizes = [40.0 + 600.0 * abs(qi) for qi in q]
    else:
        colors = [elements.ELEMENT_DICT[e][8] for e in enames]
        sizes = [elements.ELEMENT_DICT[e][6] * 90.0 for e in enames]
    plotAtoms(apos=apos, es=enames, colors=colors, sizes=sizes, axes=(0, 1), marker='o')
    if charges is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label='Q [e]')

    # 4) CoM markers
    for j in range(len(packs)):
        ax.plot(pos[j, 0], pos[j, 1], 'kx', ms=7, mew=1.0, zorder=5)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
    ax.grid(True, alpha=0.2)
    return (xmin, xmax, ymin, ymax)


def plot_before_after(packs, pos0, quat0, pos1, quat1, bonds0, path, mol_name,
                      charges=None, fit=None, z_eval=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    lims = plot_assembly_on_faf(axs[0], packs, pos0, quat0, bonds0, 'before',
                                charges=charges, fit=fit, z_eval=z_eval)
    plot_assembly_on_faf(axs[1], packs, pos1, quat1, bonds0, 'after',
                         charges=charges, fit=fit, z_eval=z_eval, xlim=lims[:2], ylim=lims[2:])
    fig.suptitle(f'{mol_name} PairFF+FAF greedy MC')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def make_trajectory_gif(packs, frames, bonds0, path, mol_name, charges=None, fit=None, z_eval=None,
                        fps=4, dpi=100):
    """Animate the MC trajectory: FAF substrate + bonds + charge-colored atoms per frame."""
    from PIL import Image
    if len(frames) < 2:
        return
    # Fixed axis limits across all frames
    all_xy = np.vstack([_body_sites_world(p['rel'], f[1][j], f[2][j])[:, :2]
                        for f in frames for j, p in enumerate(packs)])
    pad = 4.0
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    pil_frames = []
    for label, pos, quat, E in frames:
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_assembly_on_faf(ax, packs, pos, quat, bonds0, f'{mol_name}  {label}  E={E:.4f} eV',
                             charges=charges, fit=fit, z_eval=z_eval, xlim=xlim, ylim=ylim)
        fig.canvas.draw()
        pil_frames.append(Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()))
        plt.close(fig)
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=1000 // fps, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mol', type=str, default='PTCDA',
                    help='molecule name or comma-separated list for multi-species '
                         '(e.g. adenine,uracil). Single: PTCDA, NTCDI, formic_acid, '
                         'terephthalic_acid, TBTAP, azaindol, uracil, adenine')
    ap.add_argument('--nmol', type=int, default=4, help='copies per species (total = nmol * n_species)')
    ap.add_argument('--spacing', type=float, default=16.0)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--ntrial', type=int, default=512)
    ap.add_argument('--dxy', type=float, default=1.5, help='translation step sigma [Å] (bigger to escape minima)')
    ap.add_argument('--dphi', type=float, default=0.8, help='rotation step sigma [rad] (bigger to escape minima)')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--rmin', type=float, default=0.0, help='reject CoM–CoM distances below this (0=off)')
    ap.add_argument('--rmin-atom', type=float, default=1.6, help='reject real-atom contacts closer than this')
    ap.add_argument('--k-pack', type=float, default=0.03, help='centripetal packing spring [eV/Å²]')
    ap.add_argument('--no-qeq', action='store_true', help='keep XYZ charges (usually 0)')
    ap.add_argument('--no-faf', action='store_true', help='disable FAF substrate (vacuum)')
    ap.add_argument('--z-init', type=float, default=3.0, help='initial height above NaCl surface [Å]')
    args = ap.parse_args()

    mol_names = [m.strip() for m in args.mol.split(',')]
    is_multi = len(mol_names) > 1
    mol_label = '+'.join(mol_names) if is_multi else mol_names[0]
    OUT = os.path.join(OUT_ROOT, mol_label.replace(',', '_'))
    os.makedirs(OUT, exist_ok=True)

    LOADERS = {
        'NTCDI':             lambda: load_ntcdi(),
        'PTCDA':             lambda: load_ptcda(qeq=not args.no_qeq),
        'formic_acid':       lambda: load_formic_acid(qeq=not args.no_qeq),
        'terephthalic_acid': lambda: load_terephthalic_acid(qeq=not args.no_qeq),
        'TBTAP':             lambda: load_tbtap(qeq=not args.no_qeq),
        'azaindol':          lambda: load_azaindol(qeq=not args.no_qeq),
        'uracil':            lambda: load_uracil(qeq=not args.no_qeq),
        'adenine':           lambda: load_adenine(qeq=not args.no_qeq),
    }
    for mn in mol_names:
        if mn not in LOADERS:
            raise ValueError(f'unknown molecule {mn!r}; available: {list(LOADERS.keys())}')

    # Load each species
    species = []
    for mn in mol_names:
        print(f'Loading {mn}...')
        species.append(LOADERS[mn]())

    # Build molecules list: nmol copies of each species, interleaved on the grid
    n_total = args.nmol * len(species)
    molecules = []
    all_bonds0 = []
    all_REQs_concat = []
    for sp_idx, (apos, enames, REQs, b0) in enumerate(species):
        for _ in range(args.nmol):
            molecules.append((apos, enames, REQs))
    # Per-pack bonds list for plotting (one bond array per pack, cycling species)
    bonds0 = [species[i % len(species)][3] for i in range(n_total)]

    # FAF substrate fit — use one fit covering all species
    fit = None
    z_mol = 0.0
    if not args.no_faf:
        # For multi-species: use the default fit (ptcdi has C,N,O,H — broadest)
        # For single-species: use the species-specific fit if available
        if is_multi:
            fit_path = FAF_FIT_DEFAULT
        elif mol_names[0] in FAF_FITS:
            fit_path = FAF_FITS[mol_names[0]]
        else:
            fit_path = FAF_FIT_DEFAULT
        fit = load_fit(fit_path)
        # Remap each species and concatenate atom_type_ids
        at_ids_parts = []
        for sp_idx, (apos, enames, REQs, _) in enumerate(species):
            sp_fit = remap_fit_for_molecule(fit, REQs)
            n_real = int((np.asarray(REQs)[:, 3] >= 0).sum())  # all REQs are real atoms
            at_ids_parts.append(sp_fit['atom_type_ids'])
            print(f'  FAF remap {mol_names[sp_idx]}: {sp_fit["atom_type_ids"]}')
        fit['atom_type_ids'] = np.concatenate(at_ids_parts).astype(np.int32)
        # For multi-species, tile the per-species IDs nmol times
        # _folded_types_all_sites slices per pack, so we need total = nmol * n_real_per_species
        tiled_parts = []
        for sp_idx in range(len(species)):
            n_real = len(at_ids_parts[sp_idx])
            for _ in range(args.nmol):
                tiled_parts.append(at_ids_parts[sp_idx])
        fit['atom_type_ids'] = np.concatenate(tiled_parts).astype(np.int32)
        z_mol = Z_SURF_TOP + args.z_init
        print(f'FAF: loaded {fit_path}  ntypes={fit["coeffs"].shape[0]}  '
              f'nbasis={fit["coeffs"].shape[1]}  z_mol={z_mol:.2f} Å  '
              f'total_atom_type_ids={fit["atom_type_ids"].shape[0]}')

    # Grid positions — interleave species for better mixing
    pos = grid_pos(n_total, args.spacing, z=z_mol)
    rng = np.random.default_rng(args.seed)
    pos[:, 0] += rng.normal(0, 0.6, size=n_total).astype(np.float32)
    pos[:, 1] += rng.normal(0, 0.6, size=n_total).astype(np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n_total, 1))
    for i in range(n_total):
        phi0 = (i * 0.5 * np.pi) + float(rng.uniform(-0.35, 0.35))
        quat[i] = np.array([0, 0, np.sin(0.5 * phi0), np.cos(0.5 * phi0)], dtype=np.float32)

    print(f'Building PairFF ({mol_label}, {n_total} mols, k_pack={args.k_pack}, FAF={"on" if fit else "off"})...')
    rbd = RigidBodyPairFF.from_molecules(
        molecules, pos, quats=quat, active_body=0,
        He=-0.1, rc=3.0, w=0.7, k_z=0.0, z_target=z_mol, Hs=1.0, beta=1.7,
    )
    if fit is not None:
        rbd.attach_pairff_faf(fit, z_init=args.z_init, k_z=0.0, enable=True)
        # Re-upload poses to the replica path (attach_pairff_faf rewrites poss, not replica buffers)
    print(f'  device: {rbd.ctx.devices[0].name}')
    print(f'  sites/mol={rbd.atom_counts[0]}  bonds0={len(bonds0)}  n_bodies={rbd.n_bodies}')

    par = parity_check(rbd, pos, quat)
    print(f'Parity ΔE_full={par["dE_full"]:.6f}  ΔE_part={par["dE_part"]:.6f}  |err|={par["err"]:.3e}')
    assert par['err'] < 1e-3, f'parity failed: {par}'

    # Extract per-atom charges for coloring (from REQs[:,2], real atoms only)
    charges_per_mol = []
    for pack in rbd._mb_packs:
        m = pack['types'] == 0
        charges_per_mol.append(pack['REQ_ext'][m, 2].copy())
    charges_all = np.concatenate(charges_per_mol)

    pos0, quat0 = pos.copy(), quat.copy()
    E = rbd.eval_energy_system(pos, quat, k_pack=args.k_pack)
    hist = [E]
    n_acc = 0
    print(f'E_initial = {E:.6f} eV  (PairFF{"+FAF" if fit else ""}+pack)')

    frames = [('initial', pos0.copy(), quat0.copy(), hist[0])]
    for step in range(args.steps):
        moved = [step % n_total]
        pos, quat, E0, Ebest, acc, Ebatch = rbd.greedy_energy_step(
            pos, quat, moved, n_trial=args.ntrial, dxy=args.dxy, dphi=args.dphi,
            seed=args.seed + 1000 + step, rmin_com=args.rmin, rmin_atom=args.rmin_atom,
            k_pack=args.k_pack,
        )
        if acc:
            n_acc += 1
        E = rbd.eval_energy_system(pos, quat, k_pack=args.k_pack)
        hist.append(E)
        if acc or step % 50 == 0:
            frames.append((f'step{step:04d}', pos.copy(), quat.copy(), E))
        finite = Ebatch[np.isfinite(Ebatch)]
        if step % 50 == 0 or acc:
            print(f'  step {step:04d} moved={moved[0]}  E={E:10.5f}  dE_trial={Ebest-E0:10.5f}  '
                  f'acc={int(acc)}  batch_min={finite.min():.5f}')

    frames.append(('final', pos.copy(), quat.copy(), hist[-1]))
    write_xyz(os.path.join(OUT, 'after.xyz'), rbd._mb_packs, pos, quat, comment=f'E={E:.6f}')
    write_xyz(os.path.join(OUT, 'before.xyz'), rbd._mb_packs, pos0, quat0, comment=f'E={hist[0]:.6f}')

    traj = os.path.join(OUT, 'traj.xyz')
    with open(traj, 'w') as f:
        for label, p, q, Ee in frames:
            lines = []
            for j, pack in enumerate(rbd._mb_packs):
                w = _body_sites_world(pack['rel'], p[j], q[j])
                for e, t, xyz in zip(pack['enames'], pack['types'], w):
                    if t == 0:
                        lines.append(f'{e:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}')
            f.write(f'{len(lines)}\n{label} E={Ee:.6f}\n')
            f.write('\n'.join(lines) + '\n')

    e_png = os.path.join(OUT, 'energy_history.png')
    ba_png = os.path.join(OUT, 'assembly_before_after.png')
    traj_gif = os.path.join(OUT, 'trajectory.gif')
    faf_label = '+FAF' if fit else ''
    z_eval = z_mol if fit is not None else None
    plot_energy(hist, e_png, f'Greedy PairFF{faf_label}+pack ({mol_label}, k_pack={args.k_pack})')
    plot_before_after(rbd._mb_packs, pos0, quat0, pos, quat, bonds0, ba_png, mol_label,
                      charges=charges_all, fit=fit, z_eval=z_eval)
    print('Generating trajectory GIF...')
    make_trajectory_gif(rbd._mb_packs, frames, bonds0, traj_gif, mol_label,
                        charges=charges_all, fit=fit, z_eval=z_eval)

    # Inter-mol contact stats (per-pack atom counts — multi-species safe)
    apos_f, _, _ = assembly_real_atoms(rbd._mb_packs, pos, quat, bonds0)
    mins = []
    offsets = [0]
    for pack in rbd._mb_packs:
        offsets.append(offsets[-1] + int((pack['types'] == 0).sum()))
    for i in range(n_total):
        for j in range(i + 1, n_total):
            di = apos_f[offsets[i]:offsets[i + 1]]
            dj = apos_f[offsets[j]:offsets[j + 1]]
            d = np.linalg.norm(di[:, None, :] - dj[None, :, :], axis=-1).min()
            mins.append((i, j, float(d)))

    out_txt = os.path.join(OUT, 'summary.out')
    with open(out_txt, 'w') as f:
        f.write(f'mol={mol_label} nmol_per_species={args.nmol} n_species={len(species)} n_total={n_total} steps={args.steps} ntrial={args.ntrial}\n')
        f.write(f'k_pack={args.k_pack} rmin={args.rmin} rmin_atom={args.rmin_atom} spacing={args.spacing}\n')
        f.write(f'dxy={args.dxy} dphi={args.dphi} z_init={args.z_init} faf={"on" if fit else "off"}\n')
        f.write(f'device={rbd.ctx.devices[0].name}\n')
        f.write(f'parity_err={par["err"]:.6e}\n')
        f.write(f'E_initial={hist[0]:.6f}\nE_final={hist[-1]:.6f}\ndE={hist[-1]-hist[0]:.6f}\n')
        f.write(f'E_pack_final={RigidBodyPairFF.packing_energy(pos, args.k_pack):.6f}\n')
        f.write(f'accepted={n_acc}/{args.steps}\n')
        f.write('min_intermol_dist:\n')
        for i, j, d in mins:
            f.write(f'  {i}-{j}: {d:.3f} Å\n')
        f.write(f'pos_final=\n{pos}\n')

    print(f'\nE_final = {hist[-1]:.6f} eV  (ΔE = {hist[-1]-hist[0]:.6f})  accepted {n_acc}/{args.steps}')
    print('min inter-mol:', ', '.join(f'{i}-{j}:{d:.2f}' for i, j, d in mins))
    print(f'charges: min={charges_all.min():.3f} max={charges_all.max():.3f}')
    print(f'REVIEW: {out_txt}')
    print(f'REVIEW: {e_png}')
    print(f'REVIEW: {ba_png}')
    print(f'REVIEW: {traj_gif}')
    print(f'REVIEW: {os.path.join(OUT, "before.xyz")}')
    print(f'REVIEW: {os.path.join(OUT, "after.xyz")}')
    print(f'REVIEW: {traj}')


if __name__ == '__main__':
    main()
