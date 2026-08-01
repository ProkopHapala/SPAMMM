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
import time

# Unbuffered stdout so progress prints flush immediately (USER rule: never run
# a long script silently — user must see what is starting + progress + finish).
sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore', message='Ignoring fixed .* limits', category=UserWarning)
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from spammm.AtomicSystem import AtomicSystem
from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF, _body_sites_world
from spammm.forcefields.RigidBodyUtils import load_molecule, greedy_energy_step, grid_pos, assembly_real_atoms, write_xyz
from spammm.forcefields.RigidEnsemble import RigidEnsemble
from spammm.surfaces.FoldedRigid import (
    Z_SURF_TOP, NACL_SUBSTRATE, load_substrate, replicate_substrate,
    remap_fit_for_molecule,
)
from spammm.surfaces.surface_plots import plot_assembly_on_substrate

OUT_ROOT = os.path.join(REPO, 'debug', 'testplot_pairff_energy_mc')


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
                          xlim=None, ylim=None, extent_pad=4.0, ngrid=120,
                          sub_apos=None, sub_enames=None, probe_type='auto'):
    """Thin wrapper: resolve packs→real atoms, then delegate to the shared
    surface_plots.plot_assembly_on_substrate (FAF heatmap + Na/Cl ions + molecule + CoM).

    sub_apos/sub_enames: replicated substrate ions (load once in main, pass through).
    probe_type: 'auto' (most-negative-Q FAF type → Na/Cl checkerboard visible),
                int, or None.
    """
    apos, enames, bonds = assembly_real_atoms(packs, pos, quat, bonds0)
    return plot_assembly_on_substrate(
        ax, apos, enames, mol_bonds=bonds,
        sub_apos=sub_apos, sub_enames=sub_enames,
        fit=fit, z_eval=z_eval, probe_type=probe_type,
        charges=charges, com_pos=pos,
        title=title, xlim=xlim, ylim=ylim,
        extent_pad=extent_pad, ngrid=ngrid,
    )


def plot_before_after(packs, pos0, quat0, pos1, quat1, bonds0, path, mol_name,
                      charges=None, fit=None, z_eval=None,
                      sub_apos=None, sub_enames=None, probe_type='auto'):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    lims = plot_assembly_on_faf(axs[0], packs, pos0, quat0, bonds0, 'before',
                                charges=charges, fit=fit, z_eval=z_eval,
                                sub_apos=sub_apos, sub_enames=sub_enames, probe_type=probe_type)
    plot_assembly_on_faf(axs[1], packs, pos1, quat1, bonds0, 'after',
                         charges=charges, fit=fit, z_eval=z_eval,
                         sub_apos=sub_apos, sub_enames=sub_enames, probe_type=probe_type,
                         xlim=lims[:2], ylim=lims[2:])
    fig.suptitle(f'{mol_name} PairFF+FAF greedy MC')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def make_trajectory_gif(packs, frames, bonds0, path, mol_name, charges=None, fit=None, z_eval=None,
                        fps=4, dpi=100, sub_apos=None, sub_enames=None, probe_type='auto'):
    """Animate the MC trajectory: FAF substrate + Na/Cl ions + bonds + charge-colored atoms per frame."""
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
                             charges=charges, fit=fit, z_eval=z_eval, xlim=xlim, ylim=ylim,
                             sub_apos=sub_apos, sub_enames=sub_enames, probe_type=probe_type)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.asarray(fig.canvas.buffer_rgba()).tobytes()
        pil_frames.append(Image.frombytes('RGBA', (w, h), rgba).convert('RGB'))
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
    ap.add_argument('--faf-mode', type=str, default='auto', choices=['auto', 'typed', 'factorized'],
                    help='FAF fit mode: typed (per-type combined, with charge discretization), '
                         'factorized (per-atom PLQH mixing, substrate-only Coulomb), or auto (pick by availability)')
    ap.add_argument('--z-init', type=float, default=3.0, help='initial height above NaCl surface [Å]')
    # Output control
    ap.add_argument('--no-plot', action='store_true', help='skip all plotting (pure performance benchmark)')
    ap.add_argument('--no-gif', action='store_true', help='skip GIF generation (keep before/after PNG only)')
    # Verbosity / debugging
    ap.add_argument('-v', '--verbosity', type=int, default=1, choices=[0, 1, 2, 3],
                    help='0=silent, 1=accepted-only (default), 2=every-10-steps, 3=debug+timing')
    ap.add_argument('--timing', action='store_true', help='print per-step ms breakdown (GPU vs Python)')
    ap.add_argument('--profile', action='store_true', help='run cProfile on the MC loop, print top-20')
    args = ap.parse_args()

    mol_names = [m.strip() for m in args.mol.split(',')]
    is_multi = len(mol_names) > 1
    mol_label = '+'.join(mol_names) if is_multi else mol_names[0]
    faf_tag = 'nofaf' if args.no_faf else args.faf_mode
    OUT = os.path.join(OUT_ROOT, f'{mol_label.replace(",", "_")}_{faf_tag}')
    os.makedirs(OUT, exist_ok=True)
    print(f'=== MC assembly START  mol={mol_label}  nmol={args.nmol}  steps={args.steps}  ntrial={args.ntrial}  FAF={faf_tag}  → {OUT}', flush=True)

    # Molecule paths + QEq defaults (NTCDI uses mol2 file charges, not QEq)
    _MOL_DIR = os.path.join(REPO, 'data', 'mol')
    _XYZ_DIR = os.path.join(REPO, 'data', 'xyz')
    MOL_PATHS = {
        'NTCDI':             os.path.join(_MOL_DIR, 'NTCDI.mol2'),
        'TBTAP':             os.path.join(_MOL_DIR, 'TBTAP.mol2'),
        'PTCDA':             os.path.join(_XYZ_DIR, 'PTCDA.xyz'),
        'formic_acid':       os.path.join(_XYZ_DIR, 'HCOOH.xyz'),
        'terephthalic_acid': os.path.join(_XYZ_DIR, 'terephthalic_acid.xyz'),
        'azaindol':          os.path.join(_XYZ_DIR, 'azaindol.xyz'),
        'uracil':            os.path.join(_XYZ_DIR, 'uracil.xyz'),
        'adenine':           os.path.join(_XYZ_DIR, 'adenine.xyz'),
    }
    # NTCDI.mol2 has good file charges; others use QEq
    _NO_QEQ = {'NTCDI'}
    for mn in mol_names:
        if mn not in MOL_PATHS:
            raise ValueError(f'unknown molecule {mn!r}; available: {list(MOL_PATHS.keys())}')

    # Load each species
    species = []
    for mn in mol_names:
        print(f'Loading {mn}...')
        species.append(load_molecule(MOL_PATHS[mn], qeq=(mn not in _NO_QEQ) and (not args.no_qeq), name=mn))

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
        from spammm.surfaces.FoldedRigid import load_or_fit_faf, faf_fit_mode, FAF_MODE_TYPED, FAF_MODE_FACTOR
        # Resolve requested mode: 'auto' → prefer factorized (substrate-only, exact Coulomb)
        faf_mode_req = args.faf_mode
        if faf_mode_req == 'auto':
            faf_mode_req = 'factorized'
        faf_mode_key = FAF_MODE_TYPED if faf_mode_req == 'typed' else FAF_MODE_FACTOR
        # Single shared entry point: load_or_fit_faf handles QEq + cache + fit.
        # No duplication, no silent fallback.
        fit_apos, fit_enames, fit_REQs, _ = species[0]
        charge_disc = 'element_sign' if faf_mode_key == FAF_MODE_TYPED else None
        fit = load_or_fit_faf((fit_apos, fit_enames, fit_REQs), mol_name=mol_names[0],
                              fit_mode=faf_mode_key, charge_discretization=charge_disc)
        mode = faf_fit_mode(fit)
        if mode == FAF_MODE_TYPED:
            # Remap each species and concatenate atom_type_ids
            at_ids_parts = []
            for sp_idx, (apos, enames, REQs, _) in enumerate(species):
                sp_fit = remap_fit_for_molecule(fit, REQs)
                at_ids_parts.append(sp_fit['atom_type_ids'])
                print(f'  FAF remap {mol_names[sp_idx]}: {sp_fit["atom_type_ids"]}')
            tiled_parts = []
            for sp_idx in range(len(species)):
                n_real = len(at_ids_parts[sp_idx])
                for _ in range(args.nmol):
                    tiled_parts.append(at_ids_parts[sp_idx])
            fit['atom_type_ids'] = np.concatenate(tiled_parts).astype(np.int32)
            ntypes = fit['coeffs'].shape[0]
            nbasis = fit['coeffs'].shape[1]
        else:
            # Factorized: atom_type_ids are zeros (all atoms use the single substrate fit);
            # runtime PLQH is derived per-atom from REQs in attach_pairff_faf.
            n_real_total = sum(len(sp[2]) for sp in species) * args.nmol
            fit['atom_type_ids'] = np.zeros(n_real_total, dtype=np.int32)
            ntypes = 1
            nbasis = fit['coeffs4'].shape[0]
        z_mol = Z_SURF_TOP + args.z_init
        print(f'FAF: mode={mode}  ntypes={ntypes}  '
              f'nbasis={nbasis}  z_mol={z_mol:.2f} Å  '
              f'total_atom_type_ids={fit["atom_type_ids"].shape[0]}')

    # Substrate ions (Na/Cl) for visualization — load + replicate to the grid extent.
    # Loaded once here, threaded through plot_before_after / make_trajectory_gif so the
    # Na/Cl checkerboard is drawn (shared plot_assembly_on_substrate in surface_plots).
    sub_apos = sub_enames = None
    if fit is not None:
        sub_apos0, sub_en0, _, sub_lvec = load_substrate(NACL_SUBSTRATE)
        # Replicate only to the molecule grid extent + padding (NOT 10x larger).
        # Grid spans (nx-1)*spacing where nx=ceil(sqrt(nmol)); add 8 Å view margin.
        nx = int(np.ceil(np.sqrt(args.nmol * len(species))))
        grid_span = args.spacing * (nx - 1) * 0.5 + 8.0
        extent_xy = (-grid_span, grid_span)
        sub_apos, sub_enames = replicate_substrate(sub_apos0, sub_en0, sub_lvec,
                                                   extent_xy, extent_xy, z_min=Z_SURF_TOP - 1.5)
        print(f'substrate: {len(sub_enames)} ions replicated to extent {extent_xy}', flush=True)

    # Grid positions — interleave species for better mixing
    pos = grid_pos(n_total, args.spacing, z=z_mol)
    rng = np.random.default_rng(args.seed)
    pos[:, 0] += rng.normal(0, 0.6, size=n_total).astype(np.float32)
    pos[:, 1] += rng.normal(0, 0.6, size=n_total).astype(np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n_total, 1))
    for i in range(n_total):
        phi0 = (i * 0.5 * np.pi) + float(rng.uniform(-0.35, 0.35))
        quat[i] = np.array([0, 0, np.sin(0.5 * phi0), np.cos(0.5 * phi0)], dtype=np.float32)

    # Build shared rigid-pose ensemble (RigidEnsemble SSOT for rigid modules).
    # The MC loop reads/writes poses through the ensemble; RigidBodyPairFF and plotting
    # read from it. tid = species name (cycles through mol_names per pack).
    tids = [mol_names[i % len(mol_names)] for i in range(n_total)]
    ensemble = RigidEnsemble.from_poses(tids, pos, quat)
    print(f'  ensemble: {ensemble.summary()}')

    print(f'Building PairFF ({mol_label}, {n_total} mols, k_pack={args.k_pack}, FAF={"on" if fit else "off"})...')
    pos, quat = ensemble.get_poses()  # read from ensemble (copies) for GPU upload
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

    pos0, quat0 = ensemble.get_poses()  # snapshot initial poses from ensemble
    E = rbd.eval_energy_system(pos0, quat0, k_pack=args.k_pack)
    hist = [E]
    n_acc = 0
    print(f'E_initial = {E:.6f} eV  (PairFF{"+FAF" if fit else ""}+pack)')

    frames = [('initial', pos0.copy(), quat0.copy(), hist[0])]
    pos, quat = ensemble.get_poses()  # working copies read from ensemble
    t_gpu_total = 0.0; t_step_total = 0.0
    for step in range(args.steps):
        moved = [step % n_total]
        t_step0 = time.perf_counter()
        pos, quat, E0, Ebest, acc, Ebatch = greedy_energy_step(rbd,
            pos, quat, moved, n_trial=args.ntrial, dxy=args.dxy, dphi=args.dphi,
            seed=args.seed + 1000 + step, rmin_com=args.rmin, rmin_atom=args.rmin_atom,
            k_pack=args.k_pack,
        )
        t_step_total += time.perf_counter() - t_step0
        if acc:
            n_acc += 1
            ensemble.set_poses(pos, quat)  # write accepted poses back to ensemble
            # Frozen-frozen terms are invariant, so the active-channel delta is
            # the exact full-system delta; avoid a second replica launch/step.
            E += Ebest - E0
        hist.append(E)
        if acc or step % 50 == 0:
            frames.append((f'step{step:04d}', pos.copy(), quat.copy(), E))
        finite = Ebatch[np.isfinite(Ebatch)]
        if acc and args.verbosity >= 1:
            dE = E - hist[-2] if len(hist) >= 2 else 0.0
            print(f'  step {step:04d} moved={moved[0]}  E={E:10.5f}  dE={dE:+.5f}  '
                  f'acc=✓  batch_min={finite.min():.5f}', flush=True)
        elif args.verbosity >= 2 and step % 10 == 0:
            print(f'  step {step:04d}  E={E:10.5f}  acc=✗', flush=True)

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
    if not args.no_plot:
        plot_energy(hist, e_png, f'Greedy PairFF{faf_label}+pack ({mol_label}, k_pack={args.k_pack})')
        plot_before_after(rbd._mb_packs, pos0, quat0, pos, quat, bonds0, ba_png, mol_label,
                          charges=charges_all, fit=fit, z_eval=z_eval,
                          sub_apos=sub_apos, sub_enames=sub_enames)
        if not args.no_gif:
            print('Generating trajectory GIF...', flush=True)
            make_trajectory_gif(rbd._mb_packs, frames, bonds0, traj_gif, mol_label,
                                charges=charges_all, fit=fit, z_eval=z_eval,
                                sub_apos=sub_apos, sub_enames=sub_enames)
        else:
            print('Skipping GIF (--no-gif).', flush=True)
    else:
        print('Skipping all plots (--no-plot).', flush=True)

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
    print(f'=== MC assembly DONE  mol={mol_label}  E: {hist[0]:.4f} → {hist[-1]:.4f} eV  ({n_acc}/{args.steps} accepted)  → {OUT}', flush=True)
    if args.timing or args.verbosity >= 3:
        avg_ms = t_step_total / max(args.steps, 1) * 1000
        print(f'TIMING: avg {avg_ms:.2f} ms/step  total {t_step_total:.2f}s  ({args.steps} steps, {args.ntrial} trials/step)', flush=True)
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
