"""
RigidBodyUtils.py — High-level workflow functions for rigid-body simulation.

Standalone functions that take an `rbd` (RigidBodyDynamics / RigidBodyPairFF)
object and orchestrate workflows: molecule loading, assembly building,
MC/GA optimization, AFM manipulation, grid placement, trajectory I/O.

Pattern: same as `spammm/surfaces/FoldedRigid.py` — functions take `rbd` as
first arg, call its GPU methods, return results. No GPU code here.

Used by: GUI extensions (RigidAssemblyExtension, FoldedRigidExtension),
demos (demo_pairff), tests (testplot_pairff_energy_mc, test_forcefield).
"""
import numpy as np

from spammm import elements
from spammm.AtomicSystem import AtomicSystem
from spammm.topology.FFparams import make_REQs_from_enames
from .QEq import solve_from_elements, get_atom_types, compute_qeq_reqs
from .RigidBodyDynamics import _body_sites_world


# =============================================================================
# Molecule loading
# =============================================================================

def _bonds_from_geom(apos, enames):
    """Infer intramolecular bonds from geometry via AtomicSystem.neighs(bBond=True)."""
    atypes = [elements.ELEMENT_DICT[e][0] if e in elements.ELEMENT_DICT else 6 for e in enames]
    mol = AtomicSystem(apos=np.asarray(apos, dtype=np.float32).copy(), atypes=atypes, enames=list(enames))
    mol.neighs(bBond=True)
    if mol.bonds is None or len(mol.bonds) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(mol.bonds, dtype=np.int32)


def load_molecule(path, qeq=True, planarize=True, name=''):
    """Load any molecule (.xyz, .mol2, .mol) for rigid-body simulation.

    General loader — uses AtomicSystem (handles all formats) + QEq charges.
    Returns (apos, enames, REQs, bonds) ready for RigidBodyPairFF.from_molecules.
    """
    mol = AtomicSystem(fname=path)
    apos = np.asarray(mol.apos, dtype=np.float32)
    enames = [str(e) for e in mol.enames]
    if planarize:
        apos = apos.copy()
        apos[:, :2] -= apos[:, :2].mean(axis=0)
        apos[:, 2] = 0.0
    if qeq:
        REQs = compute_qeq_reqs(apos, enames, name=name)
    else:
        q = np.asarray(mol.qs, dtype=np.float32) if mol.qs is not None else np.zeros(len(enames), np.float32)
        _, atom_types = get_atom_types()
        REQs = make_REQs_from_enames(enames, q, atom_types)
    bonds = _bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def graph_to_rigid_fragments(graph, qeq=True, planarize=True):
    """Split an AtomicGraph into independent rigid-body fragments (connected components).

    Returns (fragments, coms):
      fragments: list of (apos_rel (n,3) f32, enames list[str], REQs (n,4) f32, bonds (m,2) i32)
      coms: (n_frags, 3) f32 — mass-weighted center of mass per fragment
    """
    components = graph.find_connected_components()
    if not components:
        return [], np.zeros((0, 3), dtype=np.float32)

    all_atoms = [a for comp in components for a in comp]
    all_apos = np.array([a.pos for a in all_atoms], dtype=np.float32)
    all_enames = [str(a.ename) for a in all_atoms]
    if planarize:
        all_apos = all_apos.copy()
        all_apos[:, 2] = 0.0

    if qeq:
        etypes, atom_types = get_atom_types()
        q_all = -solve_from_elements(all_apos, all_enames, etypes, Q_target=0.0)
        print(f'  graph_to_rigid_fragments: QEq sum={q_all.sum():.4f}  Q range=[{q_all.min():.3f},{q_all.max():.3f}]')
    else:
        q_all = np.array([float(a.charge) for a in all_atoms], dtype=np.float32)
        etypes, atom_types = get_atom_types()

    atom_to_idx = {a._id: i for i, a in enumerate(all_atoms)}
    fragments = []
    coms = np.zeros((len(components), 3), dtype=np.float32)

    for fi, comp in enumerate(components):
        apos = np.array([a.pos for a in comp], dtype=np.float32)
        enames = [str(a.ename) for a in comp]
        masses = np.array([elements.ELEMENT_DICT[e][10] if e in elements.ELEMENT_DICT else 12.0 for e in enames], dtype=np.float32)
        mtot = masses.sum()
        com = apos.mean(axis=0) if mtot <= 0 else (apos * masses[:, None]).sum(axis=0) / mtot
        coms[fi] = com.astype(np.float32)
        apos_rel = (apos - com).astype(np.float32)
        if planarize:
            apos_rel[:, 2] = 0.0
        q_frag = np.array([q_all[atom_to_idx[a._id]] for a in comp], dtype=np.float32)
        REQs = make_REQs_from_enames(enames, q_frag, atom_types)
        bonds = _bonds_from_geom(apos_rel, enames)
        fragments.append((apos_rel, enames, REQs, bonds))

    return fragments, coms


# =============================================================================
# Grid placement + assembly I/O
# =============================================================================

def grid_pos(n, spacing, z=0.0):
    """Place N CoMs on an XY grid centered at origin."""
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
    """Write real-atom world-frame positions to XYZ file from rigid packs + poses."""
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


# =============================================================================
# MC / Greedy optimization
# =============================================================================

def greedy_energy_step(rbd, pos, quat, moved, n_trial=128, dxy=0.4, dphi=0.25, seed=None,
                       rmin_com=0.0, rmin_atom=0.0, k_pack=0.0, pack_center=(0.0, 0.0)):
    """Greedy best-of-batch planar move of molecules in ``moved``.

    Score = PairFF channel energy of the active set + optional packing well.
    Reject trials with CoM–CoM < rmin_com or any real-atom pair < rmin_atom.
    Returns (pos', quat', E_before, E_best, accepted, E_batch).
    """
    rng = np.random.default_rng(seed)
    nmol = int(rbd.n_bodies)
    pos = np.asarray(pos, dtype=np.float32).reshape(nmol, 3).copy()
    quat = np.asarray(quat, dtype=np.float32).reshape(nmol, 4).copy()
    moved = np.asarray(moved, dtype=np.int32).ravel()
    n_trial = int(n_trial)
    if n_trial < 1:
        raise ValueError(f"n_trial must be positive, got {n_trial}")
    poss = np.zeros((n_trial, nmol, 4), dtype=np.float32)
    poss[:, :, :3] = pos[None, :, :]
    qrots = np.tile(quat[None, :, :], (n_trial, 1, 1))
    if n_trial > 1:
        rnd = rng.normal(size=(n_trial - 1, moved.size, 3))
        poss[1:, moved, 0] += (rnd[..., 0] * dxy).astype(np.float32)
        poss[1:, moved, 1] += (rnd[..., 1] * dxy).astype(np.float32)
        s = np.sin(0.5 * rnd[..., 2] * dphi)
        c = np.cos(0.5 * rnd[..., 2] * dphi)
        q0 = quat[moved][None, :, :]
        qr = np.empty((n_trial - 1, moved.size, 4), dtype=np.float32)
        qr[..., 0] = q0[..., 0] * c + q0[..., 1] * s
        qr[..., 1] = q0[..., 1] * c - q0[..., 0] * s
        qr[..., 2] = q0[..., 3] * s + q0[..., 2] * c
        qr[..., 3] = q0[..., 3] * c - q0[..., 2] * s
        qr /= np.linalg.norm(qr, axis=-1, keepdims=True)
        qrots[1:, moved, :] = qr
    E_chan = rbd.eval_energy_replicas(poss, qrots, active_mols=moved, rmin_com=rmin_com, rmin_atom=rmin_atom)
    E = rbd.energy_changed(E_chan)
    if k_pack > 0.0:
        d = poss[:, :, :2] - np.asarray(pack_center, dtype=np.float32)[None, None, :2]
        E += 0.5 * k_pack * np.sum(d * d, axis=(1, 2))
    if (rmin_com > 0.0) or (rmin_atom > 0.0):
        clash = np.any(E_chan[..., 3] > 0.0, axis=1)
        clash[0] = False
        E[clash] = np.inf
    E0 = float(E[0])
    ibest = int(np.argmin(E))
    Ebest = float(E[ibest])
    accepted = np.isfinite(Ebest) and Ebest < E0 - 1e-8
    if accepted:
        pos = poss[ibest, :, :3].copy()
        quat = qrots[ibest].copy()
    return pos, quat, E0, Ebest, accepted, E


def run_greedy_mc_assembly(rbd, ensemble, n_steps, n_trial, dxy, dphi, k_pack,
                           rmin_com=0.0, rmin_atom=0.0, seed=0, verbosity=1, record_every=50):
    """Run greedy MC assembly with round-robin moved index.

    Returns: (pos, quat, energy_history, n_accepted, frames)
    where frames is a list of (label, pos, quat, E) tuples.
    """
    pos0, quat0 = ensemble.get_poses()
    E = rbd.eval_energy_system(pos0, quat0, k_pack=k_pack)
    hist = [E]
    n_acc = 0
    n_total = len(ensemble)
    frames = [('initial', pos0.copy(), quat0.copy(), hist[0])]
    pos, quat = ensemble.get_poses()

    for step in range(n_steps):
        moved = [step % n_total]
        pos, quat, E0, Ebest, acc, Ebatch = greedy_energy_step(
            rbd, pos, quat, moved, n_trial=n_trial, dxy=dxy, dphi=dphi,
            seed=seed + 1000 + step, rmin_com=rmin_com, rmin_atom=rmin_atom, k_pack=k_pack,
        )
        if acc:
            n_acc += 1
            ensemble.set_poses(pos, quat)
            E += Ebest - E0
        hist.append(E)
        if acc or step % record_every == 0:
            frames.append((f'step{step:04d}', pos.copy(), quat.copy(), E))
        finite = Ebatch[np.isfinite(Ebatch)]
        if acc and verbosity >= 1:
            dE = E - hist[-2] if len(hist) >= 2 else 0.0
            print(f'  step {step:04d} moved={moved[0]}  E={E:10.5f}  dE={dE:+.5f}  '
                  f'acc=✓  batch_min={finite.min():.5f}', flush=True)
        elif verbosity >= 2 and step % 10 == 0:
            print(f'  step {step:04d}  E={E:10.5f}  acc=✗', flush=True)

    frames.append(('final', pos.copy(), quat.copy(), hist[-1]))
    return pos, quat, hist, n_acc, frames


# =============================================================================
# AFM manipulation
# =============================================================================

def tip_pull_scan(rbd, pin_local_idx, path, k_spring=20.0, n_relax=100, dt=0.02,
                  fire=True, record_every=1):
    """AFM-like tip pull: spring on one active-molecule atom, move target along path.

    ``pin_local_idx`` is local to the active molecule (0..na-1). In allmol_mode the
    GPU anchor is written at ``mol_offsets[active] + pin_local_idx``.
    Uses ``run_pairff`` (optional FAF). Inactive bodies stay frozen but force the active one.

    Returns dict with CoM/quat/pin trails, per-frame world sites (real atoms), tip path.
    """
    path = np.asarray(path, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError(f"path must be (N,3), got {path.shape}")
    a = int(rbd.active_body)
    i0 = int(rbd.mol_offsets[a]) if getattr(rbd, 'allmol_mode', False) and rbd.mol_offsets is not None else 0
    gi = i0 + int(pin_local_idx)
    if gi < 0 or gi >= rbd.total_atoms:
        raise ValueError(f"pin global index {gi} out of range [0,{rbd.total_atoms})")

    positions, quats, pin_xyz, E_list, frames = [], [], [], [], []
    for ip, target in enumerate(path):
        anchors = np.zeros((rbd.total_atoms, 4), dtype=np.float32)
        anchors[:, 3] = -1.0
        anchors[gi, :3] = target
        anchors[gi, 3] = float(k_spring)
        rbd.anchors = anchors
        rbd.upload_anchors()
        rbd.run_pairff(int(n_relax), float(dt), lin_damp=0.9, ang_damp=0.88, fire=fire)
        out = rbd.download_outputs()
        sites = rbd.world_sites_all_bodies(real_only=True)
        pin_w = sites[a]['world'][int(pin_local_idx), :3].copy()
        atom_E = out['atom_positions'][0]
        E = float(atom_E[:, 3].sum())
        if (ip % max(int(record_every), 1)) == 0:
            positions.append(out['pos'][a, :3].copy())
            quats.append(out['quats'][a].copy())
            pin_xyz.append(pin_w)
            E_list.append(E)
            frames.append(sites)
        print(f"  tip_pull {ip+1}/{len(path)}  CoM={out['pos'][a,:3]}  pin={pin_w}  E={E:.4f}")

    anc = np.zeros((rbd.total_atoms, 4), dtype=np.float32)
    anc[:, 3] = -1.0
    rbd.anchors = anc
    rbd.upload_anchors()
    return {
        'path': path,
        'pos': np.asarray(positions, dtype=np.float32),
        'quat': np.asarray(quats, dtype=np.float32),
        'pin': np.asarray(pin_xyz, dtype=np.float32),
        'E': np.asarray(E_list, dtype=np.float64),
        'frames': frames,
        'active_body': a,
        'pin_local_idx': int(pin_local_idx),
        'k_spring': float(k_spring),
    }
