#!/usr/bin/env python3
"""Greedy planar MC assembly with PairFF energy_replica kernel + packing well.

Default: PTCDA (QEq charges) — O end-groups attract peripheral H → T / windmill motifs.
Optional centripetal packing potential pulls CoMs toward origin for a tight layer.

Usage:
  python3 tests/testplot_pairff_energy_mc.py
  python3 tests/testplot_pairff_energy_mc.py --mol NTCDI --k-pack 0.02
  python3 tests/testplot_pairff_energy_mc.py --mol PTCDA --nmol 4 --k-pack 0.04 --spacing 16

Artifacts → debug/testplot_pairff_energy_mc/
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
from spammm.topology.FFparams import (
    read_atom_types, read_element_types, make_REQs_from_enames, load_xyz_with_REQs, _DATA_PATH,
)

OUT = os.path.join(REPO, 'debug', 'testplot_pairff_energy_mc')
NTCDI = os.path.join(REPO, 'data', 'mol', 'NTCDI.mol2')
PTCDA = os.path.join(REPO, 'data', 'xyz', 'PTCDA.xyz')


def _atom_types_dict():
    etypes = read_element_types(os.path.join(_DATA_PATH, 'ElementTypes.dat'))
    return etypes, read_atom_types(os.path.join(_DATA_PATH, 'AtomTypes.dat'), etypes)


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
    """Concatenate real-atom world frames + replicated intramolecular bonds."""
    worlds, enames, bonds = [], [], []
    off = 0
    for j, pack in enumerate(packs):
        w = _body_sites_world(pack['rel'], pos[j], quat[j])
        m = pack['types'] == 0
        wr = w[m]
        er = [e for e, t in zip(pack['enames'], pack['types']) if t == 0]
        worlds.append(wr)
        enames.extend(er)
        if bonds0 is not None and len(bonds0):
            bonds.extend([(int(a) + off, int(b) + off) for a, b in bonds0])
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


def plot_assembly_panel(ax, packs, pos, quat, bonds0, title):
    """Bonded skeleton + element colors from elements.ELEMENT_DICT via plotUtils."""
    apos, enames, bonds = assembly_real_atoms(packs, pos, quat, bonds0)
    plt.sca(ax)
    if len(bonds):
        plotBonds(links=bonds, ps=apos, axes=(0, 1), colors='#555555', lws=0.7)
    colors = [elements.ELEMENT_DICT[e][8] for e in enames]
    sizes = [elements.ELEMENT_DICT[e][6] * 90.0 for e in enames]
    plotAtoms(apos=apos, es=enames, colors=colors, sizes=sizes, axes=(0, 1), marker='o')
    for j in range(len(packs)):
        ax.plot(pos[j, 0], pos[j, 1], 'kx', ms=7, mew=1.0, zorder=5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.grid(True, alpha=0.2)


def plot_before_after(packs, pos0, quat0, pos1, quat1, bonds0, path, mol_name):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    plot_assembly_panel(axs[0], packs, pos0, quat0, bonds0, 'before')
    plot_assembly_panel(axs[1], packs, pos1, quat1, bonds0, 'after')
    fig.suptitle(f'{mol_name} PairFF greedy MC (bonds + element colors)')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mol', choices=['NTCDI', 'PTCDA'], default='PTCDA')
    ap.add_argument('--nmol', type=int, default=4)
    ap.add_argument('--spacing', type=float, default=16.0)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--ntrial', type=int, default=512)
    ap.add_argument('--dxy', type=float, default=0.55)
    ap.add_argument('--dphi', type=float, default=0.35)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--rmin', type=float, default=0.0, help='reject CoM–CoM distances below this (0=off)')
    ap.add_argument('--rmin-atom', type=float, default=1.6, help='reject real-atom contacts closer than this')
    ap.add_argument('--k-pack', type=float, default=0.03, help='centripetal packing spring [eV/Å²]')
    ap.add_argument('--no-qeq', action='store_true', help='PTCDA: keep XYZ charges (usually 0)')
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.mol == 'NTCDI':
        apos, enames, REQs, bonds0 = load_ntcdi()
    else:
        apos, enames, REQs, bonds0 = load_ptcda(qeq=not args.no_qeq)

    molecules = [(apos, enames, REQs)] * args.nmol
    pos = grid_pos(args.nmol, args.spacing, z=0.0)
    rng = np.random.default_rng(args.seed)
    pos[:, 0] += rng.normal(0, 0.6, size=args.nmol).astype(np.float32)
    pos[:, 1] += rng.normal(0, 0.6, size=args.nmol).astype(np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (args.nmol, 1))
    for i in range(args.nmol):
        # Prefer near-orthogonal starts for PTCDA windmill (≈45° + jitter)
        phi0 = (i * 0.5 * np.pi) + float(rng.uniform(-0.35, 0.35))
        quat[i] = np.array([0, 0, np.sin(0.5 * phi0), np.cos(0.5 * phi0)], dtype=np.float32)

    print(f'Building PairFF ({args.mol} x{args.nmol}, k_pack={args.k_pack})...')
    rbd = RigidBodyPairFF.from_molecules(
        molecules, pos, quats=quat, active_body=0,
        He=-0.1, rc=3.0, w=0.7, k_z=0.0, z_target=0.0, Hs=1.0, beta=1.7,
    )
    print(f'  device: {rbd.ctx.devices[0].name}')
    print(f'  sites/mol={rbd.atom_counts[0]}  bonds0={len(bonds0)}  Q range=[{REQs[:,2].min():.3f},{REQs[:,2].max():.3f}]')

    par = parity_check(rbd, pos, quat)
    print(f'Parity ΔE_full={par["dE_full"]:.6f}  ΔE_part={par["dE_part"]:.6f}  |err|={par["err"]:.3e}')
    assert par['err'] < 1e-3, f'parity failed: {par}'

    pos0, quat0 = pos.copy(), quat.copy()
    E = rbd.eval_energy_system(pos, quat, k_pack=args.k_pack)
    hist = [E]
    n_acc = 0
    print(f'E_initial = {E:.6f} eV  (PairFF+pack)')

    frames = [('initial', pos0.copy(), quat0.copy(), hist[0])]
    for step in range(args.steps):
        moved = [step % args.nmol]
        pos, quat, E0, Ebest, acc, Ebatch = rbd.greedy_energy_step(
            pos, quat, moved, n_trial=args.ntrial, dxy=args.dxy, dphi=args.dphi,
            seed=args.seed + 1000 + step, rmin_com=args.rmin, rmin_atom=args.rmin_atom,
            k_pack=args.k_pack,
        )
        if acc:
            n_acc += 1
        E = rbd.eval_energy_system(pos, quat, k_pack=args.k_pack)
        hist.append(E)
        if acc or step % 10 == 0:
            frames.append((f'step{step:03d}', pos.copy(), quat.copy(), E))
        finite = Ebatch[np.isfinite(Ebatch)]
        print(f'  step {step:03d} moved={moved[0]}  E={E:10.5f}  dE_trial={Ebest-E0:10.5f}  '
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
    plot_energy(hist, e_png, f'Greedy PairFF+pack ({args.mol}, k_pack={args.k_pack})')
    plot_before_after(rbd._mb_packs, pos0, quat0, pos, quat, bonds0, ba_png, args.mol)

    # Inter-mol contact stats
    apos_f, _, _ = assembly_real_atoms(rbd._mb_packs, pos, quat, bonds0)
    na = int((rbd._mb_packs[0]['types'] == 0).sum())
    mins = []
    for i in range(args.nmol):
        for j in range(i + 1, args.nmol):
            di = apos_f[i * na:(i + 1) * na]
            dj = apos_f[j * na:(j + 1) * na]
            d = np.linalg.norm(di[:, None, :] - dj[None, :, :], axis=-1).min()
            mins.append((i, j, float(d)))

    out_txt = os.path.join(OUT, 'summary.out')
    with open(out_txt, 'w') as f:
        f.write(f'mol={args.mol} nmol={args.nmol} steps={args.steps} ntrial={args.ntrial}\n')
        f.write(f'k_pack={args.k_pack} rmin={args.rmin} rmin_atom={args.rmin_atom} spacing={args.spacing}\n')
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
    print(f'REVIEW: {out_txt}')
    print(f'REVIEW: {e_png}')
    print(f'REVIEW: {ba_png}')
    print(f'REVIEW: {os.path.join(OUT, "before.xyz")}')
    print(f'REVIEW: {os.path.join(OUT, "after.xyz")}')
    print(f'REVIEW: {traj}')


if __name__ == '__main__':
    main()
