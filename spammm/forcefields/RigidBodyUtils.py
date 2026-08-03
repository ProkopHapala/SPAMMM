"""
RigidBodyUtils.py — High-level workflow functions for rigid-body simulation.

Standalone functions that take an `rbd` (RigidBodyDynamics / RigidBodyPairFF)
object and orchestrate workflows: molecule loading, assembly building,
MC/GA optimization, AFM manipulation, grid placement, trajectory I/O.

Pattern: same as `spammm/surfaces/FoldedRigid.py` — functions take `rbd` as
first arg, call its GPU methods, return results. No GPU code here.

Key functions:
- **`graph_to_rigid_fragments`** — split AtomicGraph into rigid bodies via
  `find_connected_components`. Atoms within each fragment are in BFS order
  (not file order) — callers must use the same order for display sync.
- **`build_mixed_species_assembly`** — round-robin body ordering for multi-species
  assemblies. Deterministic, interleaved by copy.
- **`compute_combined_probe_map`** — E_map = E_PairFF(static) + E_FAF(NaCl).
  Dynamic molecules excluded. Headless (no Qt/Vispy); the CPU path is an independent
  reference for the production GPU evaluator.

Used by: GUI extensions (RigidAssemblyExtension, FoldedRigidExtension),
demos (demo_pairff, static_obstacle_drag_demo), tests (test_body_state,
testplot_pairff_energy_mc, test_forcefield).

Caveats:
- `graph_to_rigid_fragments` returns atoms in BFS order — the display graph must
  be rebuilt to match (see `RigidAssemblyExtension._ensure_backend_matched`).
- `compute_combined_probe_map` remains available as a parity/reference path; GUI
  recompute may use `RigidBodyPairFF.eval_probe_grid_gpu` when the device supports it.
"""
import numpy as np

from spammm import elements
from spammm.AtomicSystem import AtomicSystem
from spammm.topology.FFparams import make_REQs_from_enames
from .QEq import solve_from_elements, get_atom_types, compute_qeq_reqs
from .RigidBodyDynamics import _body_sites_world

COULOMB_CONST = 14.3996448915
R2SAFE = 1e-4


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
        # Extract bonds from the authoritative AtomicGraph (not re-inferred from geometry)
        comp_ids = {a._id for a in comp}
        comp_id_list = [a._id for a in comp]
        id_to_local = {aid: li for li, aid in enumerate(comp_id_list)}
        frag_bonds = []
        if hasattr(graph, 'bonds'):
            for b in graph.bonds.values():
                if not getattr(b, 'alive', True):
                    continue
                a_id, b_id = b.a._id, b.b._id
                if a_id in comp_ids and b_id in comp_ids:
                    frag_bonds.append([id_to_local[a_id], id_to_local[b_id]])
        if frag_bonds:
            bonds = np.array(frag_bonds, dtype=np.int32)
        else:
            # Fallback: infer from geometry if graph has no bonds
            bonds = _bonds_from_geom(apos_rel, enames)
        fragments.append((apos_rel, enames, REQs, bonds))

    return fragments, coms


# =============================================================================
# Mixed-species assembly builder (shared by GUI + testplot)
# =============================================================================

def build_mixed_species_assembly(mol_names, nmol, mol_paths, no_qeq_set, spacing, z_body, seed,
                                 qeq=True):
    """Build aligned mixed-species assembly data in round-robin body order.

    Body order is deterministic and interleaved:
      copy 0: species[0], species[1], ..., species[N-1]
      copy 1: species[0], species[1], ..., species[N-1]

    This ensures molecules, tids, bonds_list, and display atom ranges all use
    the same order.  Do NOT use the testplot's grouped-by-species ordering.

    Args:
        mol_names: list of species name strings (keys in mol_paths)
        nmol: copies per species (total bodies = nmol * len(mol_names))
        mol_paths: dict {name: path}
        no_qeq_set: set of names that should skip QEq (use file charges)
        spacing: grid spacing in Angstrom
        z_body: body CoM z coordinate
        seed: RNG seed for jitter + rotation init
        qeq: whether to run QEq (overridden by no_qeq_set per species)

    Returns:
        molecules: list of (apos, enames, REQs) tuples, length n_total
        tids: list of species name strings, length n_total
        bonds_list: list of bond arrays, length n_total
        pos: (n_total, 3) f32 grid positions with jitter
        quat: (n_total, 4) f32 quaternions with rotation init
        species_data: list of (apos, enames, REQs, bonds) per species (for FAF)
    """
    n_species = len(mol_names)
    n_total = nmol * n_species

    # Load each species once
    species_data = []
    for mn in mol_names:
        if mn not in mol_paths:
            raise ValueError(f'Unknown molecule: {mn}. Available: {sorted(mol_paths.keys())}')
        sd = load_molecule(mol_paths[mn], qeq=qeq and (mn not in no_qeq_set), name=mn)
        species_data.append(sd)

    # Round-robin body order: copy 0 of each species, then copy 1, etc.
    molecules = []
    tids = []
    bonds_list = []
    for copy_idx in range(nmol):
        for sp_idx in range(n_species):
            apos, enames, REQs, bonds = species_data[sp_idx]
            molecules.append((apos, enames, REQs))
            tids.append(mol_names[sp_idx])
            bonds_list.append(bonds)

    # Grid positions + deterministic jitter/rotation (same pattern as _on_build)
    pos = grid_pos(n_total, spacing=spacing, z=z_body)
    rng = np.random.default_rng(int(seed))
    pos[:, 0] += rng.normal(0, 0.6, size=n_total).astype(np.float32)
    pos[:, 1] += rng.normal(0, 0.6, size=n_total).astype(np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n_total, 1))
    for i in range(n_total):
        phi0 = (i * 0.5 * np.pi) + float(rng.uniform(-0.35, 0.35))
        quat[i] = np.array([0, 0, np.sin(0.5 * phi0), np.cos(0.5 * phi0)], dtype=np.float32)

    return molecules, tids, bonds_list, pos, quat, species_data


# =============================================================================
# Headless combined PairFF+FAF probe map (shared by GUI + tests)
# =============================================================================

def plan_probe_grid(xy, margin=4.0, step=0.1, aspect=None, max_points=2_000_000):
    """Plan a snapped ``(ny,nx)`` XY grid around real-atom coordinates.

    ``aspect`` is canvas width/height.  Only the shorter world-space axis is
    expanded, so atom+margin bounds are never cropped.  The point limit fails
    loudly instead of silently changing scientific sampling.
    """
    step = float(step)
    margin = float(margin)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f'grid step must be finite and positive, got {step}')
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(f'grid margin must be finite and non-negative, got {margin}')
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if xy.size:
        lo = xy.min(axis=0) - margin
        hi = xy.max(axis=0) + margin
    else:
        lo = np.array([-margin, -margin], dtype=np.float64)
        hi = np.array([margin, margin], dtype=np.float64)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, step)
    if aspect is not None:
        aspect = float(aspect)
        if not np.isfinite(aspect) or aspect <= 0.0:
            raise ValueError(f'grid aspect must be finite and positive, got {aspect}')
        current = span[0] / span[1]
        if current < aspect:
            span[0] = span[1] * aspect
        elif current > aspect:
            span[1] = span[0] / aspect
    lo = np.floor((center - 0.5 * span) / step) * step
    hi = np.ceil((center + 0.5 * span) / step) * step
    xs = np.arange(lo[0], hi[0] + 0.5 * step, step, dtype=np.float64)
    ys = np.arange(lo[1], hi[1] + 0.5 * step, step, dtype=np.float64)
    n_points = int(xs.size) * int(ys.size)
    if n_points > int(max_points):
        raise ValueError(f'probe grid needs {n_points} points ({ys.size}x{xs.size}); '
                         f'increase step or reduce margin (limit={int(max_points)})')
    return xs, ys, [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]


def _compute_unified_probe_pair_map(static_apos, static_REQ, static_types, probe_REQ,
                                    z_probe, xs, ys, beta):
    """Independent NumPy reference for the unified GPU site-pair energy."""
    apos = np.asarray(static_apos, dtype=np.float64).reshape(-1, 3)
    req = np.asarray(static_REQ, dtype=np.float64).reshape(-1, 4)
    types = np.asarray(static_types, dtype=np.int32).reshape(-1)
    if len(apos) != len(req) or len(req) != len(types):
        raise ValueError('static site positions, REQ, and types must have equal lengths')
    probe = np.asarray(probe_REQ, dtype=np.float64).reshape(4)
    X, Y = np.meshgrid(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), indexing='xy')
    Emap = np.zeros_like(X, dtype=np.float64)
    gi = 1.0
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f'PairFF beta must be finite and positive, got {beta}')
    for j, typ in enumerate(types):
        gj = 1.0 if int(typ) == 0 else 0.0
        gij = gi * gj
        R0 = gij * (probe[0] + req[j, 0])
        w = probe[3] + req[j, 3]
        alpha = gij
        attr = -min(0.0, probe[2] * req[j, 2])
        both_dummy = 1.0 - min(gi + gj, 1.0)
        E0 = (attr * (1.0 - gij) + probe[1] * req[j, 1] * gij) * (1.0 - both_dummy)
        dx = X - apos[j, 0]
        dy = Y - apos[j, 1]
        dz = float(z_probe) - apos[j, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if E0 != 0.0:
            rho_c = R0 + 8.0 / beta
            rc2 = rho_c * (rho_c + 2.0 * w)
            mask = r2 <= rc2
            rw = np.sqrt(r2 + w * w)
            rho = r2 / np.maximum(rw + w, 1e-12)
            u = np.maximum(0.0, 1.0 - (beta / 8.0) * (rho - R0))
            u2 = u * u
            u4 = u2 * u2
            y = u4 * u4
            Emap += np.where(mask, E0 * y * (alpha * y - (1.0 + alpha)), 0.0)
        if gij > 0.5:
            Emap += COULOMB_CONST * probe[2] * req[j, 2] / np.sqrt(r2 + R2SAFE)
    return Emap


def nuclear_exclusion_mask(xs, ys, z_probe, static_apos, static_types, r=1.0):
    """Boolean mask (ny, nx): True where pixel is within r Å of any real atom.

    Used to exclude nuclear singularities from vmin/vmax estimation without
    punching NaN holes in the displayed map.
    """
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    apos = np.asarray(static_apos, dtype=np.float64).reshape(-1, 3)
    types = np.asarray(static_types, dtype=np.int32).reshape(-1)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    excluded = np.zeros_like(X, dtype=bool)
    r2 = float(r) * float(r)
    for j, typ in enumerate(types):
        if int(typ) != 0:
            continue
        dx = X - apos[j, 0]
        dy = Y - apos[j, 1]
        dz = float(z_probe) - apos[j, 2]
        excluded |= (dx * dx + dy * dy + dz * dz) < r2
    return excluded

def compute_combined_probe_map(rbd, fit, frozen_mask, probe_R0, probe_E0, probe_q,
                               z_probe, margin=4.0, step=0.1, beta=None, aspect=None):
    """Compute E_map = E_PairFF(static mols) + E_FAF(NaCl) on a 2D grid at z_probe.

    Headless (no Qt/VisPy).  Dynamic molecules are excluded — only frozen (static)
    bodies contribute to the PairFF part.  The CPU PairFF calculation consumes the
    packed ``REQ_ext`` values and includes the same compact-exp + damped Coulomb terms
    as unified dynamics.

    Args:
        rbd: RigidBodyPairFF with _mb_packs populated
        fit: FAF fit dict (or None for no FAF)
        frozen_mask: (n_bodies,) bool array — True = static (contributes to map)
        probe_R0, probe_E0, probe_q: probe parameters
        z_probe: absolute z height for map evaluation
        margin, step: grid extent margin and step in Angstrom
        beta: optional parity assertion; production beta comes from ``rbd``.

    Returns:
        E_sum: (ny, nx) float64 — combined PairFF + FAF potential
        E_pairff: (ny, nx) float64 — PairFF contribution only
        E_faf: (ny, nx) float64 or None — FAF contribution only
        xs: (nx,) float64 — x axis
        ys: (ny,) float64 — y axis
        extent: [xmin, xmax, ymin, ymax]
        exclude_mask: (ny, nx) bool — True within 1 Å of a real atom (for vmin/vmax)
    """
    from spammm.surfaces.FoldedRigid import eval_folded_potential_grid, faf_type_idx_for_probe, faf_fit_mode, FAF_MODE_FACTOR

    params = getattr(rbd, 'pairff_params_host', None)
    if params is None or params.get('beta') is None:
        raise RuntimeError('RigidBodyPairFF map requires packed pairff_params_host[beta]')
    beta_live = float(params['beta'])
    if beta is not None and not np.isclose(float(beta), beta_live, rtol=0.0, atol=1e-7):
        raise ValueError(f'caller beta={beta} disagrees with live PairFF beta={beta_live}')
    beta = beta_live

    # Gather world sites from frozen (static) bodies only (for PairFF energy)
    # but compute grid extent from ALL live real atoms (so dynamic bodies stay in view)
    frozen_mask = np.asarray(frozen_mask, dtype=bool)
    body_state = getattr(rbd, '_body_state_host', None)
    sites_pos, sites_REQ, sites_enames, sites_types = [], [], [], []
    all_live_xy = []  # xy of all live real atoms for extent
    for j, pack in enumerate(rbd._mb_packs):
        is_deleted = body_state is not None and j < len(body_state) and body_state[j] < 0
        pos_j = rbd._mb_pos[j]
        quat_j = rbd._mb_quat[j]
        world = _body_sites_world(pack['rel'], pos_j, quat_j)
        if not is_deleted:
            m_real = pack['types'] == 0
            all_live_xy.append(world[m_real, :2])
        if j >= len(frozen_mask) or not frozen_mask[j] or is_deleted:
            continue
        sites_pos.append(world)
        sites_REQ.append(pack['REQ_ext'])
        sites_enames.extend(pack['enames'])
        sites_types.append(pack['types'])

    # Grid extent from all live real atoms (not just static)
    all_xy = np.vstack(all_live_xy).astype(np.float64) if all_live_xy else np.zeros((0, 2), dtype=np.float64)
    xs, ys, extent = plan_probe_grid(all_xy, margin=margin, step=step, aspect=aspect)

    if sites_pos:
        static_apos = np.vstack(sites_pos).astype(np.float64)
        static_REQ = np.vstack(sites_REQ).astype(np.float64)
        static_types = np.concatenate(sites_types).astype(np.int32)
        probe_REQ = np.array([probe_R0, np.sqrt(max(float(probe_E0), 0.0)), probe_q, 0.0], dtype=np.float64)
        if getattr(rbd, 'ctx', None) is not None and hasattr(rbd, 'eval_probe_grid_gpu'):
            gpu_map = rbd.eval_probe_grid_gpu(
                static_apos, static_REQ, static_types, probe_REQ, xs, ys, z_probe,
                beta=beta, probe_faf=fit is not None)
            E_pairff = np.asarray(gpu_map[..., 1], dtype=np.float64)
            E_faf_gpu = np.asarray(gpu_map[..., 2], dtype=np.float64)
        else:
            E_pairff = _compute_unified_probe_pair_map(
                static_apos, static_REQ, static_types, probe_REQ, z_probe, xs, ys, beta)
            E_faf_gpu = None
    else:
        E_pairff = np.zeros((len(ys), len(xs)), dtype=np.float64)
        E_faf_gpu = None

    # FAF contribution
    E_faf = None
    if fit is not None and E_faf_gpu is None:
        mode = faf_fit_mode(fit)
        if mode == FAF_MODE_FACTOR:
            probe_REQH = np.array([float(probe_R0), float(np.sqrt(max(float(probe_E0), 0.0))),
                                   float(probe_q), 0.0], dtype=np.float32)
            E_faf = eval_folded_potential_grid(fit, 0, xs, ys, z_probe, atom_REQH=probe_REQH)
        else:
            ityp = faf_type_idx_for_probe(fit, probe_R0, probe_E0, probe_q)
            E_faf = eval_folded_potential_grid(fit, ityp, xs, ys, z_probe)

    if E_faf_gpu is not None:
        E_faf = E_faf_gpu
    E_sum = E_pairff + (E_faf if E_faf is not None else 0.0)
    # Nuclear exclusion mask: pixels within 1 Å of any real atom (for vmin/vmax only)
    if sites_pos:
        exclude_mask = nuclear_exclusion_mask(xs, ys, z_probe, static_apos, static_types, r=1.0)
    else:
        exclude_mask = np.zeros((len(ys), len(xs)), dtype=bool)
    return E_sum, E_pairff, E_faf, xs, ys, extent, exclude_mask


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
