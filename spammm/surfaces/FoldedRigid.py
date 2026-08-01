"""
FoldedRigid.py — Core functions for folded-basis rigid-body molecular simulation on surfaces.

Purpose: Workflow orchestration for fitting folded basis potentials, setting up
rigid-body dynamics, running relaxations, lateral scans, and manipulation
trajectories on periodic substrates. Also supplies **CPU FAF map helpers** used by
PairFF Vispy (`eval_folded_potential_grid`, `faf_type_idx_for_probe`) so the
background map can show E_PairFF + E_FAF at the same height.

Key functionality:
  - fit_folded_for_molecule() — fit folded basis coefficients for molecule+substrate
  - setup_rigid_folded() — create RigidBodyDynamics from fit result
  - relax_folded() — run relaxation recording trajectory
  - lateral_scan() — scan molecule across substrate at fixed z
  - relaxed_scan() — pinned-atom manipulation scan
  - manipulation_trajectory() — lateral manipulation simulation
  - replicate_substrate() — periodic replication of substrate atoms
  - load_substrate() — load substrate from XYZ
  - nearest_substrate_distance() — distance to nearest substrate atom of given element
  - find_bonds() — find bonds by distance cutoff
  - save_xyz_trajectory() — save multi-frame XYZ with substrate context
  - random_quaternion() — random rotation quaternion
  - eval_folded_potential(_slice|_grid) — CPU FAF eval for diagnostics / PairFF map

Role in SPAMMM: Core simulation workflow module. Used by tests, FoldedRigid GUI,
and PairFF `--faf` demo. Plotting functions are in surface_plots.py.
"""

import os
import numpy as np

from spammm.forcefields.SPFF_cl import SPFF_cl as MolecularDynamics
from spammm.forcefields.RigidBodyDynamics import RigidBodyDynamics, _guess_mass, _quat_to_matrix_np, _reqs_to_plq, compute_mass_properties
from spammm.topology.FFparams import load_xyz_with_REQs
from spammm.AtomicSystem import AtomicSystem

# =============================================================================
# Constants
# =============================================================================

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NACL_SUBSTRATE = os.path.join(_proj_root, 'data', 'substrates', 'NaCl_1x1_L3.xyz')
Z_SURF_TOP = -3.25
LATTICE_A = 4.0

MORSE_ALPHAS = np.array([1.0, 1.8, 2.7, 3.6, 5.0], dtype=np.float32)
COULOMB_ALPHAS = np.array([0.0, 0.3, 0.6, 1.0, 1.5], dtype=np.float32)
COMBINED_ALPHAS = np.array([0.3, 0.6, 1.0, 1.8, 2.7, 3.6, 5.0], dtype=np.float32)
FAF_FORMAT_VERSION = 2
FAF_MODE_TYPED = 'typed_combined'
FAF_MODE_FACTOR = 'factorized_plqh'


# =============================================================================
# Geometry & math utilities
# =============================================================================

def _charge_moments(apos, q, origin):
    r = np.asarray(apos, dtype=np.float64) - np.asarray(origin, dtype=np.float64)[None, :]
    q = np.asarray(q, dtype=np.float64)
    M2 = np.einsum('i,ij,ik->jk', q, r, r)
    Q = 3.0*M2 - np.eye(3)*np.trace(M2)
    return {'M0': float(q.sum()), 'M1': np.einsum('i,ij->j', q, r), 'M2': M2, 'Q': Q}


def discretize_charges(apos, enames, q_per_atom, type_scheme='element_sign', sign_threshold=1e-3, origin=None, dipole_weight=1.0, quadrupole_weight=1.0, regularization=1e-8):
    """Replace per-atom charges by per-type charges with exact total charge.

    Dipole and traceless-quadrupole errors are minimized in coordinates centred
    at ``origin`` and scaled by molecular RMS radius, so Å and Å² rows have
    comparable conditioning. The equality constraint is solved through the
    null space, not by a large penalty.
    """
    apos = np.asarray(apos, dtype=np.float64)
    q = np.asarray(q_per_atom, dtype=np.float64).reshape(-1)
    enames = np.asarray(enames, dtype=str).reshape(-1)
    if apos.ndim != 2 or apos.shape[1] != 3 or len(apos) != len(q) or len(enames) != len(q):
        raise ValueError(f"discretize_charges(): expected apos(N,3), enames(N), q(N); got {apos.shape}, {enames.shape}, {q.shape}")
    if len(q) == 0 or not np.all(np.isfinite(apos)) or not np.all(np.isfinite(q)):
        raise ValueError("discretize_charges(): inputs must be non-empty and finite")
    if origin is None:
        origin = apos.mean(axis=0)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    r = apos - origin[None, :]
    rscale = float(np.sqrt(np.mean(np.einsum('ij,ij->i', r, r))))
    if not np.isfinite(rscale) or rscale < 1e-12:
        rscale = 1.0
    x = r/rscale
    if type_scheme == 'element':
        labels = enames.tolist()
    elif type_scheme == 'element_sign':
        s = np.where(q > float(sign_threshold), '+', np.where(q < -float(sign_threshold), '-', '0'))
        labels = [f'{e}{si}' for e, si in zip(enames, s)]
    else:
        raise ValueError(f"discretize_charges(): unknown type_scheme '{type_scheme}', expected 'element' or 'element_sign'")
    type_names, type_ids = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    type_ids = type_ids.astype(np.int32)
    K = len(type_names)
    G = np.equal(type_ids[:, None], np.arange(K)[None, :]).astype(np.float64)
    q_mean = (G.T @ q)/G.sum(axis=0)
    n_t = G.sum(axis=0)
    C = n_t[None, :]
    d = np.array([q.sum()], dtype=np.float64)
    dip_atom = x
    r2 = np.einsum('ij,ij->i', x, x)
    quad_atom = np.stack([3*x[:, 0]*x[:, 0]-r2, 3*x[:, 0]*x[:, 1], 3*x[:, 0]*x[:, 2], 3*x[:, 1]*x[:, 1]-r2, 3*x[:, 1]*x[:, 2]], axis=1)
    A = np.vstack([float(dipole_weight)*(dip_atom.T @ G), float(quadrupole_weight)*(quad_atom.T @ G)])
    b = np.concatenate([float(dipole_weight)*(dip_atom.T @ q), float(quadrupole_weight)*(quad_atom.T @ q)])
    U, sC, Vt = np.linalg.svd(C, full_matrices=True)
    rank_C = int(np.sum(sC > np.finfo(np.float64).eps*max(C.shape)*sC[0]))
    x0 = C.T @ np.linalg.solve(C @ C.T, d)
    Z = Vt[rank_C:].T
    if Z.shape[1]:
        AZ = A @ Z
        rhs = b - A @ x0
        lam = max(0.0, float(regularization))
        if lam > 0.0:
            AZ = np.vstack([AZ, np.sqrt(lam)*Z])
            rhs = np.concatenate([rhs, np.sqrt(lam)*(q_mean-x0)])
        y, _, rank_soft, s_soft = np.linalg.lstsq(AZ, rhs, rcond=None)
        Q_type = x0 + Z @ y
    else:
        rank_soft, s_soft = 0, np.zeros(0, dtype=np.float64)
        Q_type = x0
    Q_type += C.T @ np.linalg.solve(C @ C.T, d-C @ Q_type)
    q_disc = Q_type[type_ids]
    mo = _charge_moments(apos, q, origin)
    md = _charge_moments(apos, q_disc, origin)
    return {
        'type_ids': type_ids, 'type_names': type_names.tolist(), 'Q_type': Q_type, 'q_disc': q_disc,
        'moments_orig': mo, 'moments_disc': md, 'origin': origin, 'length_scale': rscale,
        'diagnostics': {'constraint_error': float(abs(q_disc.sum()-q.sum())), 'soft_rank': int(rank_soft), 'soft_singular_values': np.asarray(s_soft), 'charge_rms_error': float(np.sqrt(np.mean((q_disc-q)**2)))},
    }

def random_quaternion(max_angle=np.pi):
    """Generate a random rotation quaternion for rotation up to max_angle radians."""
    axis = np.random.randn(3)
    axis /= max(np.linalg.norm(axis), 1e-30)
    angle = np.random.uniform(0, max_angle)
    s = np.sin(angle * 0.5)
    c = np.cos(angle * 0.5)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, c], dtype=np.float32)


def nearest_substrate_distance(atom_pos, substrate_apos, substrate_enames, target_element):
    """Compute distance from a single atom position to nearest substrate atom of target_element.

    Args:
        atom_pos: (3,) array
        substrate_apos: (N, 3) or (N, 4) array
        substrate_enames: list of N element names
        target_element: e.g. 'Na' or 'Cl'
    Returns: (distance, substrate_atom_index)
    """
    p = np.asarray(atom_pos[:3], dtype=np.float64)
    apos = np.asarray(substrate_apos[:, :3], dtype=np.float64)
    enames = np.asarray(substrate_enames)
    mask = enames == target_element
    if not np.any(mask):
        return float('inf'), -1
    d = np.linalg.norm(apos[mask] - p[None, :], axis=1)
    idx = int(np.argmin(d))
    return float(d[idx]), int(np.where(mask)[0][idx])


def find_bonds(apos, enames, Rcut=1.8):
    """Find bonds by distance cutoff."""
    na = len(apos)
    bonds = []
    for i in range(na):
        for j in range(i + 1, na):
            r = np.linalg.norm(apos[i] - apos[j])
            if r < Rcut:
                bonds.append((i, j))
    return bonds


# =============================================================================
# Substrate utilities
# =============================================================================

def replicate_substrate(sub_apos, sub_enames, lvec, x_range, y_range, z_min=-10.0):
    """Periodically replicate substrate atoms within a given XY area.

    Args:
        sub_apos: (N, 3) or (N, 4) substrate positions in unit cell
        sub_enames: list of element names
        lvec: (3, 3) or (4, 3) lattice vectors (rows = a, b, c)
        x_range: (xmin, xmax)
        y_range: (ymin, ymax)
        z_min: only include atoms with z >= z_min (filter deep layers)

    Returns: (apos_rep, enames_rep) — replicated positions and names.
    """
    a = np.asarray(lvec[0, :3], dtype=np.float64)
    b = np.asarray(lvec[1, :3], dtype=np.float64)
    apos = np.asarray(sub_apos[:, :3], dtype=np.float64)
    na = len(sub_enames)

    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cell_extent = max(abs(ax) + abs(bx), abs(ay) + abs(by))
    n_rep = int(np.ceil(max(x_range[1] - x_range[0], y_range[1] - y_range[0]) / max(cell_extent, 1e-6))) + 2

    rep_pos = []
    rep_names = []
    for ix in range(-n_rep, n_rep + 1):
        for iy in range(-n_rep, n_rep + 1):
            shift = ix * a + iy * b
            for i in range(na):
                p = apos[i] + shift
                if p[0] < x_range[0] - 1 or p[0] > x_range[1] + 1:
                    continue
                if p[1] < y_range[0] - 1 or p[1] > y_range[1] + 1:
                    continue
                if p[2] < z_min:
                    continue
                rep_pos.append(p)
                rep_names.append(sub_enames[i])
    return np.array(rep_pos, dtype=np.float32), rep_names


def load_substrate(substrate_file=NACL_SUBSTRATE):
    mol = AtomicSystem(fname=substrate_file)
    apos = np.asarray(mol.apos, dtype=np.float32)
    enames = list(mol.enames) if hasattr(mol, 'enames') else []
    if not enames:
        from spammm import atomicUtils as au
        _, _, enames, _, _ = au.load_xyz(fname=substrate_file, bReadN=True)
    qs = np.asarray(mol.qs, dtype=np.float32) if hasattr(mol, 'qs') else None
    lvec = np.asarray(mol.lvec, dtype=np.float32) if hasattr(mol, 'lvec') else None
    return apos, enames, qs, lvec


# =============================================================================
# Folded basis workflow
# =============================================================================

def fit_folded_for_molecule(mol, substrate_file=NACL_SUBSTRATE, z_range_rel=(1.5, 8.0), nu=4, nv=4, nPBC=(4, 4, 0), alpha_morse=1.8, custom_alphas=None, substrate_R_override=None, q_override=None, fit_mode=FAF_MODE_TYPED, charge_discretization=None, nxy=32, nz_samp=60, ewald_n_harm=6):
    """Fit folded basis coefficients for a molecule on a substrate.

    Morse (pauli+london) and Coulomb are fitted independently via
    fit_folded_surface_basis with coulomb_solver='ewald2d'.
    Returns dict with total_coeffs, basis_params, atom_type_ids, basis_lvec2d.

    mol: either a file path (str) loaded via load_xyz_with_REQs, or a tuple
        (apos, enames, REQs) of pre-loaded molecule data. Using a tuple avoids
        the XYZ-only restriction — mol2 or any other format can be fitted.
    substrate_R_override: optional dict element->RvdW (Å), e.g. {'Na': 1.45}
    q_override: optional dict element->charge or one charge per atom
    fit_mode: ``typed_combined`` (fastest, molecule-specific) or
        ``factorized_plqh`` (one molecule-independent substrate float4 fit)
    charge_discretization: optional ``element`` or ``element_sign`` for typed mode
    """
    if isinstance(mol, (tuple, list)) and len(mol) >= 3:
        apos_mol, enames, reqs = mol[0], mol[1], mol[2]
        apos_mol = np.asarray(apos_mol, dtype=np.float32)
        reqs = np.asarray(reqs, dtype=np.float32).copy()
    else:
        apos_mol, reqs, enames, _, _ = load_xyz_with_REQs(mol)
        reqs = np.asarray(reqs, dtype=np.float32).copy()
    if q_override is not None:
        if isinstance(q_override, dict):
            for i, e in enumerate(enames):
                if e in q_override:
                    reqs[i, 2] = float(q_override[e])
        else:
            q_arr = np.asarray(q_override, dtype=np.float32).reshape(-1)
            if len(q_arr) != len(reqs):
                raise ValueError(f"fit_folded_for_molecule(): q_override length {len(q_arr)} != natoms {len(reqs)}")
            reqs[:, 2] = q_arr
    mode = str(fit_mode).lower()
    aliases = {'typed': FAF_MODE_TYPED, 'combined': FAF_MODE_TYPED, 'factorized': FAF_MODE_FACTOR, 'plqh': FAF_MODE_FACTOR}
    mode = aliases.get(mode, mode)
    if mode not in (FAF_MODE_TYPED, FAF_MODE_FACTOR):
        raise ValueError(f"fit_folded_for_molecule(): unknown fit_mode '{fit_mode}', expected '{FAF_MODE_TYPED}' or '{FAF_MODE_FACTOR}'")
    charge_fit = None
    if charge_discretization is not None:
        if mode != FAF_MODE_TYPED:
            raise ValueError("fit_folded_for_molecule(): charge_discretization only applies to typed_combined mode")
        charge_fit = discretize_charges(apos_mol, enames, reqs[:, 2], type_scheme=charge_discretization)
        reqs[:, 2] = charge_fit['q_disc'].astype(np.float32)
    z_range_abs = (Z_SURF_TOP + z_range_rel[0], Z_SURF_TOP + z_range_rel[1])
    if custom_alphas is None:
        custom_alphas = COMBINED_ALPHAS
    nz = len(custom_alphas)

    reqs_fit = reqs if mode == FAF_MODE_TYPED else np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    md = MolecularDynamics(nloc=32, debug_build_options='-DDBG_UFF=0')
    md.init_rigid_molecule_batch(np.zeros((len(reqs_fit), 3), dtype=np.float32), reqs_fit, nSystems=8192)
    md.set_surface(substrate_file, nPBC=nPBC, alpha_morse=alpha_morse, bMacro=True)
    if substrate_R_override:
        for ia, e in enumerate(md.surface_enames):
            if e in substrate_R_override:
                md.surface_REQs[ia, 0] = float(substrate_R_override[e])
        md.toGPU('REQ_s', md.surface_REQs)

    params = md.fit_folded_surface_basis(
        surf_xyz=substrate_file, components=('pauli', 'london', 'coulomb'),
        coulomb_solver='ewald2d', z_range=z_range_abs,
        nu=nu, nv=nv, nz=nz, custom_alphas=custom_alphas,
        nPBC=nPBC, alpha_morse=alpha_morse, nxy=int(nxy), nz_samp=int(nz_samp), ewald_n_harm=int(ewald_n_harm),
        substrate_R_override=substrate_R_override,
    )

    coeff_sets = params['coeff_sets']
    lvec2d = params['basis_lvec2d']
    a = np.array(lvec2d[0, :3], dtype=np.float32)
    b = np.array(lvec2d[1, :3], dtype=np.float32)
    folded_lvec2d = np.array([a[0], b[0], a[1], b[1]], dtype=np.float32)

    out = {
        'format_version': FAF_FORMAT_VERSION,
        'fit_mode': mode,
        'basis_params': params['basis_params'].astype(np.float32),
        'folded_lvec2d': folded_lvec2d,
        'z_range': params['z_range'],
        'enames': enames,
        'reqs': reqs,
        'apos_mol': apos_mol,
        'alpha_morse': float(alpha_morse),
        'substrate_R_override': dict(substrate_R_override) if substrate_R_override else {},
    }
    if mode == FAF_MODE_TYPED:
        out['coeffs'] = (coeff_sets['pauli'] + coeff_sets['london'] + coeff_sets['coulomb']).astype(np.float32)
        out['coeff_sets'] = {k: np.asarray(v, dtype=np.float32) for k, v in coeff_sets.items()}
        out['atom_type_ids'] = params['atom_type_ids'].astype(np.int32)
        out['unique_REQs'] = params['unique_REQs']
        if charge_fit is not None:
            out['charge_discretization'] = charge_fit
    else:
        nbasis = len(params['basis_params'])
        coeffs4 = np.zeros((nbasis, 4), dtype=np.float32)
        coeffs4[:, 0] = coeff_sets['pauli'][0, :nbasis]
        coeffs4[:, 1] = coeff_sets['london'][0, :nbasis]
        coeffs4[:, 2] = coeff_sets['coulomb'][0, :nbasis]
        out['coeffs4'] = coeffs4
        out['atom_plqh'] = _reqs_to_plq(reqs, alpha=float(alpha_morse))
        out['atom_type_ids'] = np.zeros(len(reqs), dtype=np.int32)
        out['unique_REQs'] = np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    return out


def save_fit(fit, fname):
    """Save a fit result dict to a small .npz file."""
    mode = faf_fit_mode(fit)
    payload = {
        'format_version': np.int32(fit.get('format_version', FAF_FORMAT_VERSION)),
        'fit_mode_str': np.array(mode),
        'basis_params': fit['basis_params'], 'atom_type_ids': fit['atom_type_ids'],
        'folded_lvec2d': fit['folded_lvec2d'], 'unique_REQs': fit['unique_REQs'],
        'z_range': np.array(fit['z_range'], dtype=np.float32), 'enames_str': np.array(','.join(fit['enames'])),
        'reqs': fit['reqs'], 'apos_mol': fit['apos_mol'], 'alpha_morse': np.float32(fit.get('alpha_morse', 1.8)),
    }
    if mode == FAF_MODE_FACTOR:
        payload['coeffs4'] = fit['coeffs4']
        payload['atom_plqh'] = fit.get('atom_plqh', _reqs_to_plq(fit['reqs'], alpha=float(payload['alpha_morse'])))
    else:
        payload['coeffs'] = fit['coeffs']
    np.savez(fname, **payload)


def load_fit(fname):
    """Load a fit result dict from .npz file written by save_fit."""
    d = np.load(fname, allow_pickle=False)
    mode = str(d['fit_mode_str'].item()) if 'fit_mode_str' in d.files else FAF_MODE_TYPED
    out = {
        'format_version': int(d['format_version']) if 'format_version' in d.files else 1,
        'fit_mode': mode, 'basis_params': d['basis_params'], 'atom_type_ids': d['atom_type_ids'],
        'folded_lvec2d': d['folded_lvec2d'], 'unique_REQs': d['unique_REQs'], 'z_range': tuple(d['z_range']),
        'enames': list(str(d['enames_str'].item()).split(',')), 'reqs': d['reqs'], 'apos_mol': d['apos_mol'],
        'alpha_morse': float(d['alpha_morse']) if 'alpha_morse' in d.files else 1.8,
    }
    if mode == FAF_MODE_FACTOR:
        if 'coeffs4' not in d.files:
            raise ValueError(f"load_fit(): factorized fit '{fname}' is missing coeffs4")
        out['coeffs4'] = d['coeffs4']
        out['atom_plqh'] = d['atom_plqh'] if 'atom_plqh' in d.files else _reqs_to_plq(out['reqs'], alpha=out['alpha_morse'])
    else:
        out['coeffs'] = d['coeffs']
        coeff_sets = {}
        for key in ('pauli', 'london', 'coulomb', 'h_bond', 'total'):
            if f'coeff_{key}' in d.files:
                coeff_sets[key] = d[f'coeff_{key}']
        if coeff_sets:
            out['coeff_sets'] = coeff_sets
    return out


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


# =============================================================================
# Shared load-or-fit entry point — single code path for all callers
# (GUI extensions, demo_pairff, testplot scripts). No duplication, no fallback.
# =============================================================================

_FIT_DIR = os.path.join(_proj_root, 'data', 'fits')


def load_or_fit_faf(mol, mol_name='mol', fit_mode=FAF_MODE_FACTOR, substrate_file=NACL_SUBSTRATE,
                    z_range_rel=(1.5, 8.0), charge_discretization=None, force_refit=False,
                    fit_path=None, **fit_kwargs):
    """Load cached FAF fit or fit molecule on substrate. SINGLE shared entry point.

    Accepts the molecule in any form and normalizes it to (apos, enames, REQs)
    with QEq charges. Without QEq, Coulomb is zero (no Na/Cl checkerboard).

    Args:
        mol: one of:
            - str: path to XYZ file (loaded with QEq via load_xyz_generic)
            - tuple (apos, enames, REQs): pre-loaded with QEq charges in REQs[:, 2]
            - tuple (apos, enames, REQs, bonds): same, bonds ignored
        mol_name: name for cache filename and log messages (e.g. 'PTCDA')
        fit_mode: FAF_MODE_FACTOR (default) or FAF_MODE_TYPED
        substrate_file: path to substrate XYZ
        z_range_rel: (z_min, z_max) relative to Z_SURF_TOP
        charge_discretization: 'element' or 'element_sign' for typed mode
        force_refit: if True, re-fit even if cache exists
        fit_path: explicit cache path (overrides default). Use when substrate or
            fit params differ from defaults (e.g. FoldedRigidExtension custom substrate)
        **fit_kwargs: passed to fit_folded_for_molecule (nu, nv, alpha_morse, etc.)

    Returns: fit dict (from load_fit or fit_folded_for_molecule)
    """
    # 1. Normalize mol to (apos, enames, REQs) with QEq charges
    if isinstance(mol, str):
        from spammm.AtomicSystem import AtomicSystem
        from spammm.forcefields.QEq import compute_qeq_reqs
        sys = AtomicSystem(fname=mol)
        apos = np.asarray(sys.apos, dtype=np.float32)
        enames = [str(e) for e in sys.enames]
        apos[:, :2] -= apos[:, :2].mean(axis=0)
        apos[:, 2] = 0.0
        REQs = compute_qeq_reqs(apos, enames, name=f'fit({mol_name})')
    elif isinstance(mol, (tuple, list)) and len(mol) >= 3:
        apos, enames, REQs = mol[0], mol[1], mol[2]
        apos = np.asarray(apos, dtype=np.float32)
        REQs = np.asarray(REQs, dtype=np.float32)
    else:
        raise ValueError(f"load_or_fit_faf(): mol must be str path or (apos, enames, REQs) tuple, got {type(mol)}")

    # 2. Cache path: default is per-molecule-per-mode; caller can override
    if fit_path is None:
        mode_tag = 'factorized' if fit_mode == FAF_MODE_FACTOR else 'typed'
        fit_path = os.path.join(_FIT_DIR, f'{mol_name.lower()}_nacl_{mode_tag}.npz')

    # 3. Load cached or fit
    if not force_refit and os.path.isfile(fit_path):
        return load_fit(fit_path)

    os.makedirs(os.path.dirname(fit_path) or '.', exist_ok=True)
    fit = fit_folded_for_molecule((apos, enames, REQs), substrate_file=substrate_file,
                                  z_range_rel=z_range_rel, fit_mode=fit_mode,
                                  charge_discretization=charge_discretization, **fit_kwargs)
    save_fit(fit, fit_path)
    return fit


def faf_fit_mode(fit_result):
    """Return the explicit FAF storage/evaluation mode, including legacy fits."""
    mode = str(fit_result.get('fit_mode', FAF_MODE_FACTOR if 'coeffs4' in fit_result else FAF_MODE_TYPED))
    if mode not in (FAF_MODE_TYPED, FAF_MODE_FACTOR):
        raise ValueError(f"faf_fit_mode(): unsupported fit mode '{mode}'")
    if mode == FAF_MODE_FACTOR and 'coeffs4' not in fit_result:
        raise ValueError("faf_fit_mode(): factorized_plqh fit is missing coeffs4")
    if mode == FAF_MODE_TYPED and 'coeffs' not in fit_result:
        raise ValueError("faf_fit_mode(): typed_combined fit is missing coeffs")
    return mode


def materialize_factorized_coeffs(fit_result, reqs):
    """Combine substrate float4 coefficients with atom REQH on the host."""
    if faf_fit_mode(fit_result) != FAF_MODE_FACTOR:
        raise ValueError("materialize_factorized_coeffs(): expected factorized_plqh fit")
    plqh = _reqs_to_plq(np.asarray(reqs, dtype=np.float32).reshape(-1, 4), alpha=float(fit_result.get('alpha_morse', 1.8)))
    return (plqh @ np.asarray(fit_result['coeffs4'], dtype=np.float32).reshape(-1, 4).T).astype(np.float32)


def compare_faf_fit_modes(mol_file, charge_discretization='element_sign', **fit_kwargs):
    """Fit both architectures and compare equivalent discretized-type rows."""
    import time
    q_override = fit_kwargs.pop('q_override', None)
    t0 = time.perf_counter()
    typed = fit_folded_for_molecule(mol_file, q_override=q_override, fit_mode=FAF_MODE_TYPED, charge_discretization=charge_discretization, **fit_kwargs)
    t1 = time.perf_counter()
    factor = fit_folded_for_molecule(mol_file, q_override=q_override, fit_mode=FAF_MODE_FACTOR, **fit_kwargs)
    t2 = time.perf_counter()
    materialized = materialize_factorized_coeffs(factor, typed['unique_REQs'])
    delta = np.asarray(typed['coeffs'], dtype=np.float64) - np.asarray(materialized, dtype=np.float64)
    return {
        'typed': typed, 'factorized': factor,
        'timing_s': {'typed': t1-t0, 'factorized': t2-t1, 'speedup_fit': (t1-t0)/max(t2-t1, 1e-30)},
        'coefficient_parity': {'rms': float(np.sqrt(np.mean(delta*delta))), 'max_abs': float(np.max(np.abs(delta))), 'reference_rms': float(np.sqrt(np.mean(np.asarray(typed['coeffs'], dtype=np.float64)**2)))},
    }


def setup_rigid_folded(mol_file, fit_result, z_init=3.0, xy_init=(0.0, 0.0), quats=None, mass_trans=1.0, debug=False):
    """Create RigidBodyDynamics with folded basis from fit result.

    Args:
        mol_file: path to molecule XYZ, or None to use fit_result['apos_mol'] etc.
        fit_result: dict from fit_folded_for_molecule
        z_init: initial height above surface top in Angstrom
        xy_init: initial (x, y) position
        quats: (4,) initial quaternion, or None for identity
        mass_trans: effective translational/rotational mass. Inertia is scaled
            by mass_trans / physical total mass, preserving its shape.
    """
    if mol_file is None:
        apos_mol = fit_result['apos_mol']
        reqs = fit_result['reqs']
        enames = fit_result['enames']
    else:
        apos_mol, reqs, enames, _, _ = load_xyz_with_REQs(mol_file)
    apos_mol = np.asarray(apos_mol, dtype=np.float32)
    masses = np.ones(len(enames), dtype=np.float32)
    com0 = (apos_mol * masses[:, None]).sum(axis=0) / masses.sum()
    rel = apos_mol - com0[None, :]
    mtot, I, Iinv = compute_mass_properties(rel, masses)
    mass_trans = float(mass_trans)
    if mass_trans <= 0.0:
        raise ValueError(f'mass_trans must be > 0, got {mass_trans}')
    I_relax = I * (mass_trans / mtot)
    Iinv_relax = Iinv * (mtot / mass_trans)

    n_bodies = 1
    n_atoms = len(enames)
    pos4 = np.zeros((n_bodies, 4), dtype=np.float32)
    pos4[0, :3] = [xy_init[0], xy_init[1], Z_SURF_TOP + z_init]
    pos4[0, 3] = mass_trans
    quat4 = np.zeros((n_bodies, 4), dtype=np.float32)
    quat4[0, 3] = 1.0
    if quats is not None:
        quat4[0, :] = np.asarray(quats, dtype=np.float32)
        quat4[0] /= max(np.linalg.norm(quat4[0]), 1e-30)
    zero4 = np.zeros((n_bodies, 4), dtype=np.float32)
    atom_body = rel[None, :, :].astype(np.float32)

    rbd = RigidBodyDynamics(debug=debug)
    rbd.realloc(n_bodies=n_bodies, num_atoms=n_atoms)
    rbd.enames = list(enames)
    rbd.atom_REQ = reqs.copy()
    rbd.atom_masses = masses.copy()
    rbd.mass_physical = float(mtot)
    rbd.mass_trans = mass_trans
    rbd.mass_rot = mass_trans
    rbd.upload_state(
        pos4, quat4, zero4, zero4,
        mass_trans, 1.0 / mass_trans,
        np.repeat(Iinv_relax[None, :, :], n_bodies, axis=0),
        atom_body,
        inertia=np.repeat(I_relax[None, :, :], n_bodies, axis=0),
    )

    mode = faf_fit_mode(fit_result)
    coeffs = fit_result['coeffs'] if mode == FAF_MODE_TYPED else fit_result['coeffs4']
    kxyz = fit_result['basis_params']
    atype = fit_result['atom_type_ids'] if mode == FAF_MODE_TYPED else _reqs_to_plq(reqs, alpha=float(fit_result.get('alpha_morse', 1.8)))
    lvec2d = fit_result['folded_lvec2d']
    nbasis = int(coeffs.shape[-2] if mode == FAF_MODE_FACTOR else coeffs.shape[1])
    folded_meta = np.array([nbasis, -1 if mode == FAF_MODE_FACTOR else coeffs.shape[0], 0, 0], dtype=np.int32)
    rbd.init_folded(coeffs, kxyz, atype, lvec2d, folded_meta=folded_meta)
    return rbd


def setup_rigid_folded_replicas(fit_result, xs, ys, z_init=3.0, quats=None, mass_trans=None,
                                pin_atom_idx=None, z_pin=None, k_spring=10.0, debug=False):
    """Create RigidBodyDynamics replicas grid for imaging on folded FAF substrate.

    One replica per (x,y) pixel; identity (flat) quaternion by default.
    xs, ys are 1D scan axes (Å, PBC).

    If pin_atom_idx is set (AFM-like tip constraint):
      - Scan (x,y) is the *pin target* (tip) position, not free COM.
      - Anchor spring holds that atom at (x, y, z_pin) with stiffness k_spring.
      - Molecule COM is initialized so the pin atom starts exactly on the tip
        (flat orientation). z_pin defaults to Z_SURF_TOP + z_init.
    Else:
      - Free molecule with COM on the scan grid at height Z_SURF_TOP + z_init.
    """
    apos_mol = np.asarray(fit_result['apos_mol'], dtype=np.float32)
    reqs = fit_result['reqs']
    enames = fit_result['enames']
    masses = np.ones(len(enames), dtype=np.float32)
    com0 = (apos_mol * masses[:, None]).sum(axis=0) / masses.sum()
    rel = apos_mol - com0[None, :]
    mtot, I, Iinv = compute_mass_properties(rel, masses)
    if mass_trans is None:
        mass_trans = float(mtot)
    mass_trans = float(mass_trans)
    if mass_trans <= 0.0:
        raise ValueError(f'mass_trans must be > 0, got {mass_trans}')
    I_relax = I * (mass_trans / mtot)
    Iinv_relax = Iinv * (mtot / mass_trans)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    nx, ny = len(xs), len(ys)
    n_rep = nx * ny
    na = len(enames)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')  # (ny, nx)
    sx = XX.ravel()
    sy = YY.ravel()

    quat4 = np.zeros((n_rep, 4), dtype=np.float32)
    if quats is None:
        quat4[:, 3] = 1.0
    else:
        q = np.asarray(quats, dtype=np.float32).reshape(-1)
        q = q / max(np.linalg.norm(q), 1e-30)
        quat4[:] = q

    pos4 = np.zeros((n_rep, 4), dtype=np.float32)
    anchors = None
    if pin_atom_idx is not None:
        ia = int(pin_atom_idx)
        if ia < 0 or ia >= na:
            raise ValueError(f'pin_atom_idx={ia} out of range [0,{na})')
        r_pin = rel[ia]
        r_pin_world = _quat_to_matrix_np(quat4[0]) @ r_pin
        z_tip = float(Z_SURF_TOP + z_init if z_pin is None else z_pin)
        pos4[:, 0] = sx - r_pin_world[0]
        pos4[:, 1] = sy - r_pin_world[1]
        pos4[:, 2] = z_tip - r_pin_world[2]
        pos4[:, 3] = mass_trans
        anchors = np.zeros((n_rep, na, 4), dtype=np.float32)
        anchors[:, :, 3] = -1.0
        anchors[:, ia, 0] = sx
        anchors[:, ia, 1] = sy
        anchors[:, ia, 2] = z_tip
        anchors[:, ia, 3] = float(k_spring)
    else:
        pos4[:, 0] = sx
        pos4[:, 1] = sy
        pos4[:, 2] = Z_SURF_TOP + float(z_init)
        pos4[:, 3] = mass_trans

    zero4 = np.zeros((n_rep, 4), dtype=np.float32)
    rbd = RigidBodyDynamics(debug=debug)
    rbd.realloc_replicas(n_replicas=n_rep, num_atoms=na)
    rbd.enames = list(enames)
    rbd.atom_REQ = reqs.copy()
    rbd.atom_masses = masses.copy()
    rbd.mass_physical = float(mtot)
    rbd.mass_trans = mass_trans
    rbd.mass_rot = mass_trans
    rbd.upload_replicas_state(
        pos4, quat4, zero4, zero4, mass_trans, Iinv_relax, rel.astype(np.float32),
        anchors=anchors, inertia=I_relax,
    )
    mode = faf_fit_mode(fit_result)
    coeffs = fit_result['coeffs'] if mode == FAF_MODE_TYPED else fit_result['coeffs4']
    kxyz = fit_result['basis_params']
    atype = fit_result['atom_type_ids'] if mode == FAF_MODE_TYPED else _reqs_to_plq(reqs, alpha=float(fit_result.get('alpha_morse', 1.8)))
    lvec2d = fit_result['folded_lvec2d']
    nbasis = int(coeffs.shape[-2] if mode == FAF_MODE_FACTOR else coeffs.shape[1])
    folded_meta = np.array([nbasis, -1 if mode == FAF_MODE_FACTOR else coeffs.shape[0], na, n_rep], dtype=np.int32)
    rbd.init_replicas(n_rep, coeffs, kxyz, atype, lvec2d, folded_meta=folded_meta)
    rbd._scan_nx = nx
    rbd._scan_ny = ny
    rbd._scan_xs = xs
    rbd._scan_ys = ys
    rbd._pin_atom_idx = pin_atom_idx
    rbd._z_pin = None if pin_atom_idx is None else float(Z_SURF_TOP + z_init if z_pin is None else z_pin)
    rbd._k_spring = float(k_spring) if pin_atom_idx is not None else 0.0
    return rbd


def relax_folded(rbd, n_steps=500, dt=0.05, lin_damp=0.1, ang_damp=0.1, record_interval=25,
                 fire=True, f_tol=1e-4, t_tol=1e-4):
    """Run relaxation, recording trajectory.

    Default: FIRE quench (zero v when v·F<0 / ω when ω·τ<0). Early-exits when
    |F|<f_tol and |τ|<t_tol. Set fire=False for plain damped MD.

    Returns dict with 'energies', 'forces', 'torques', 'positions', 'quaternions',
    'atom_positions', and 'steps' (actual steps taken).
    """
    energies = []
    forces = []
    torques = []
    positions = []
    quaternions = []
    atom_positions_list = []
    steps_done = 0
    interval = max(1, int(record_interval))

    while steps_done < n_steps:
        n = min(interval, n_steps - steps_done)
        rbd.run_folded(n, dt, lin_damp=lin_damp, ang_damp=ang_damp, fire=fire)
        steps_done += n
        out = rbd.download_outputs()
        atom_pos = out['atom_positions'][0]
        E = float(atom_pos[:, 3].sum())
        f = out['body_force'][0]
        tq = out['body_torque'][0]
        Fmag = float(np.linalg.norm(f[:3]))
        Tmag = float(np.linalg.norm(tq[:3]))
        energies.append(E)
        forces.append(Fmag)
        torques.append(Tmag)
        positions.append(out['pos'][0].copy())
        quaternions.append(out['quats'][0].copy())
        atom_positions_list.append(atom_pos[:, :3].copy())
        if Fmag < f_tol and Tmag < t_tol:
            # Confirm with force recompute at final pose (last body_force is pre-drift)
            F2, Tw2, _, _, _ = rbd.eval_force_torque('folded')
            Fmag = float(np.linalg.norm(F2)); Tmag = float(np.linalg.norm(Tw2))
            forces[-1] = Fmag; torques[-1] = Tmag
            if Fmag < f_tol and Tmag < t_tol:
                break
            # Not actually converged — continue (velocities were cleared by eval; OK near min)

    return {
        'energies': np.array(energies),
        'forces': np.array(forces),
        'torques': np.array(torques),
        'positions': np.array(positions),
        'quaternions': np.array(quaternions),
        'atom_positions': atom_positions_list,
        'steps': steps_done,
    }


def relax_folded_newton(rbd, max_iter=20, trust0=0.5, f_tol=1e-5, t_tol=1e-5, record=False):
    """Pure GPU Newton (one kernel launch). Prefer this over host FD."""
    return rbd.run_folded_newton(niter=max_iter, trust0=trust0, f_tol=f_tol, t_tol=t_tol)


def relax_folded_diag(rbd, n_steps=5000, dt=0.02, lin_damp=0.95, ang_damp=0.90, record_interval=10):
    """Run relaxation with comprehensive diagnostics for debugging rotation dynamics.

    Records thinned trajectory of: energy, |force|, |torque|, COM, quaternion,
    angular velocity (vrot), linear velocity (vpos), per-atom forces, tilt angle,
    rotation angle per step, and atom positions.

    Returns dict with arrays of shape (n_record, ...).
    """
    from spammm.forcefields.RigidBodyDynamics import _quat_to_matrix_np
    n_record = max(1, n_steps // record_interval)
    recs = {
        'step': np.zeros(n_record, dtype=np.int32),
        'E': np.zeros(n_record, dtype=np.float64),
        'Fmag': np.zeros(n_record, dtype=np.float64),
        'Tmag': np.zeros(n_record, dtype=np.float64),
        'com': np.zeros((n_record, 3), dtype=np.float64),
        'quat': np.zeros((n_record, 4), dtype=np.float64),
        'vpos': np.zeros((n_record, 3), dtype=np.float64),
        'vrot': np.zeros((n_record, 3), dtype=np.float64),
        'vrot_mag': np.zeros(n_record, dtype=np.float64),
        'rot_per_step': np.zeros(n_record, dtype=np.float64),
        'tilt': np.zeros(n_record, dtype=np.float64),
        'atom_pos': np.zeros((n_record, rbd.num_atoms, 3), dtype=np.float64),
        'atom_force': np.zeros((n_record, rbd.num_atoms, 4), dtype=np.float64),
        'body_force': np.zeros((n_record, 4), dtype=np.float64),
        'body_torque': np.zeros((n_record, 4), dtype=np.float64),
    }
    body_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    surf_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    irec = 0
    for i in range(0, n_steps, record_interval):
        steps = min(record_interval, n_steps - i)
        rbd.run_folded(steps, dt, lin_damp=lin_damp, ang_damp=ang_damp)
        out = rbd.download_outputs()
        atom_pos = out['atom_positions'][0]
        atom_f = out['atom_force'][0]
        E = float(atom_pos[:, 3].sum())
        f = out['body_force'][0][:3]
        tq = out['body_torque'][0][:3]
        pos = out['pos'][0][:3]
        quat = out['quats'][0]
        vpos = out['lin_mom'][0][:3]
        vrot = out['ang_mom'][0][:3]
        R = _quat_to_matrix_np(quat)
        body_z_world = R @ body_axis
        tilt = float(np.arccos(np.clip(np.dot(body_z_world, surf_normal), -1, 1)))
        rot_angle = float(np.linalg.norm(vrot) * dt * steps)
        recs['step'][irec] = i + steps
        recs['E'][irec] = E
        recs['Fmag'][irec] = float(np.linalg.norm(f))
        recs['Tmag'][irec] = float(np.linalg.norm(tq))
        recs['com'][irec] = pos
        recs['quat'][irec] = quat
        recs['vpos'][irec] = vpos
        recs['vrot'][irec] = vrot
        recs['vrot_mag'][irec] = float(np.linalg.norm(vrot))
        recs['rot_per_step'][irec] = rot_angle
        recs['tilt'][irec] = tilt
        recs['atom_pos'][irec] = atom_pos[:, :3]
        recs['atom_force'][irec] = atom_f
        recs['body_force'][irec] = out['body_force'][0]
        recs['body_torque'][irec] = out['body_torque'][0]
        irec += 1
    return recs


def plot_relax_diag(recs, title="Relaxation Diagnostics", save_path=None):
    """Plot diagnostic relaxation data as a multi-panel summary.

    Panels: Energy, |F| & |T|, COM z, tilt angle, |vrot| & rot/step, quaternion components.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps = recs['step']
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14)

    ax = axes[0, 0]
    ax.plot(steps, recs['E'], 'b-', lw=0.8)
    ax.set_ylabel('Energy [eV]')
    ax.set_title('Total Energy')

    ax = axes[0, 1]
    ax.plot(steps, recs['Fmag'], 'r-', lw=0.8, label='|F|')
    ax.plot(steps, recs['Tmag'], 'g-', lw=0.8, label='|T|')
    ax.set_ylabel('Force / Torque')
    ax.set_title('|Force| & |Torque|')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[1, 0]
    ax.plot(steps, recs['com'][:, 2], 'b-', lw=0.8, label='z')
    ax.plot(steps, recs['com'][:, 0], 'r-', lw=0.5, label='x')
    ax.plot(steps, recs['com'][:, 1], 'g-', lw=0.5, label='y')
    ax.set_ylabel('Position [Å]')
    ax.set_title('COM Position')
    ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, np.degrees(recs['tilt']), 'm-', lw=0.8)
    ax.set_ylabel('Tilt [deg]')
    ax.set_title('Tilt angle (body z vs surface normal)')

    ax = axes[2, 0]
    ax.plot(steps, recs['vrot_mag'], 'r-', lw=0.8, label='|vrot|')
    ax.plot(steps, recs['rot_per_step'], 'b-', lw=0.5, label='rot/record')
    ax.set_ylabel('Angular velocity / rotation')
    ax.set_title('Angular velocity magnitude')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[2, 1]
    for j, c in enumerate('xyzw'):
        ax.plot(steps, recs['quat'][:, j], lw=0.5, label=f'q{j}({c})')
    ax.set_ylabel('Quaternion')
    ax.set_title('Quaternion components')
    ax.legend()
    ax.set_xlabel('Step')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_relax_diag] Saved to {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# Folded potential evaluation (for debugging / visualization)
# =============================================================================

def eval_folded_potential(fit_result, atom_type_idx, xyz, atom_REQH=None, component='total'):
    """Evaluate a typed row or factorized atom/component at arbitrary positions.

    Args:
        fit_result: dict from fit_folded_for_molecule / load_fit
        atom_type_idx: typed coefficient-row index; factorized molecule-atom
            index when atom_REQH is not passed
        xyz: (N, 3) array of world coordinates
        atom_REQH: explicit factorized probe (R,sqrt(E),Q,H)
        component: total, pauli, london, coulomb, h_bond, or coulomb_phi

    Returns: (N,) array of potential energy values
    """
    mode = faf_fit_mode(fit_result)
    kxyz = np.asarray(fit_result['basis_params'], dtype=np.float64)
    lvec2d = np.asarray(fit_result['folded_lvec2d'], dtype=np.float64)
    ax, bx, ay, by = lvec2d
    det = ax * by - bx * ay
    invLvec = np.array([by / det, -bx / det, -ay / det, ax / det])
    xyz = np.asarray(xyz, dtype=np.float64)
    u = invLvec[0] * xyz[:, 0] + invLvec[1] * xyz[:, 1]
    v = invLvec[2] * xyz[:, 0] + invLvec[3] * xyz[:, 1]
    u = u - np.floor(u)
    v = v - np.floor(v)
    z = xyz[:, 2]
    ku = kxyz[None, :, 0]
    kv = kxyz[None, :, 1]
    az = kxyz[None, :, 2]
    z0 = kxyz[None, :, 3]
    bx_ = np.cos(2.0 * np.pi * ku * u[:, None])
    by_ = np.cos(2.0 * np.pi * kv * v[:, None])
    bz_ = np.exp(-az * np.maximum(0.0, z[:, None] - z0))
    basis = bx_ * by_ * bz_
    if mode == FAF_MODE_TYPED:
        if component != 'total':
            coeff_sets = fit_result.get('coeff_sets', {})
            if component not in coeff_sets:
                raise ValueError(f"eval_folded_potential(): typed fit has no component '{component}'")
            coeffs = np.asarray(coeff_sets[component], dtype=np.float64)
        else:
            coeffs = np.asarray(fit_result['coeffs'], dtype=np.float64)
        c = coeffs[int(atom_type_idx), :basis.shape[1]]
    else:
        coeffs4 = np.asarray(fit_result['coeffs4'], dtype=np.float64).reshape(-1, 4)[:basis.shape[1]]
        if component == 'coulomb_phi':
            mix = np.array([0.0, 0.0, 1.0, 0.0])
        else:
            if atom_REQH is None:
                reqs = np.asarray(fit_result['reqs'], dtype=np.float32)
                ia = int(atom_type_idx)
                if ia < 0 or ia >= len(reqs):
                    raise ValueError(f"eval_folded_potential(): factorized atom index {ia} outside [0,{len(reqs)})")
                atom_REQH = reqs[ia]
            req = np.asarray(atom_REQH, dtype=np.float32).reshape(1, 4)
            mix = _reqs_to_plq(req, alpha=float(fit_result.get('alpha_morse', 1.8)))[0].astype(np.float64)
            component_index = {'pauli': 0, 'london': 1, 'coulomb': 2, 'h_bond': 3}
            if component != 'total':
                if component not in component_index:
                    raise ValueError(f"eval_folded_potential(): unknown factorized component '{component}'")
                mask = np.zeros(4, dtype=np.float64)
                mask[component_index[component]] = 1.0
                mix *= mask
        c = coeffs4 @ mix
    return (basis * c[None, :]).sum(axis=1)


def eval_folded_potential_slice(fit_result, atom_type_idx, plane='xy', fixed_val=0.0, extent=(-8, 8, -8, 8), n=64):
    """Evaluate folded basis potential for a single atom type on a 2D grid.

    Args:
        fit_result: dict from fit_folded_for_molecule / load_fit
        atom_type_idx: index into coeffs rows
        plane: 'xy' (top view), 'xz' or 'yz' (side views)
        fixed_val: value of the fixed coordinate (z for 'xy', y for 'xz', x for 'yz')
        extent: (min1, max1, min2, max2) for the two varying coordinates
        n: grid resolution (n x n)

    Returns: (c1, c2, E) where c1, c2 are 1D coordinate arrays and E is (n, n) potential matrix
    """
    a = np.linspace(extent[0], extent[1], n)
    b = np.linspace(extent[2], extent[3], n)
    A, B = np.meshgrid(a, b, indexing='ij')
    if plane == 'xy':
        xyz = np.stack([A.ravel(), B.ravel(), np.full(A.size, fixed_val)], axis=1)
    elif plane == 'xz':
        xyz = np.stack([A.ravel(), np.full(A.size, fixed_val), B.ravel()], axis=1)
    elif plane == 'yz':
        xyz = np.stack([np.full(A.size, fixed_val), A.ravel(), B.ravel()], axis=1)
    else:
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got {plane!r}")
    E = eval_folded_potential(fit_result, atom_type_idx, xyz).reshape(n, n)
    return a, b, E


def eval_folded_potential_grid(fit_result, atom_type_idx, xs, ys, z, component='total', atom_REQH=None):
    """FAF energy on an arbitrary XY grid (same layout as Vispy PairFF maps).

    xs, ys: 1D axes; returns E with shape (len(ys), len(xs)) via meshgrid(xs, ys).
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    xyz = np.stack([X.ravel(), Y.ravel(), np.full(X.size, float(z))], axis=1)
    return eval_folded_potential(fit_result, int(atom_type_idx), xyz, atom_REQH=atom_REQH, component=component).reshape(X.shape)


def faf_type_idx_for_probe(fit_result, probe_R0, probe_E0, probe_q):
    """Pick unique_REQs row closest to map-probe (R0, Q); E0 used as weak tie-break."""
    ureq = fit_result.get('unique_REQs')
    if ureq is None or len(ureq) == 0:
        return 0
    ureq = np.asarray(ureq, dtype=np.float64)
    R0, E0, Q = float(probe_R0), float(probe_E0), float(probe_q)
    best_i, best_d = 0, 1e99
    for i, req in enumerate(ureq):
        d = (req[0] - R0) ** 2 + 4.0 * (req[2] - Q) ** 2 + 0.05 * (req[1] ** 2 - E0) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return int(best_i)


# =============================================================================
# Scanning & manipulation
# =============================================================================

def lateral_scan(rbd, xs, ys, z, n_relax=50, dt=0.01):
    """Scan molecule across substrate at fixed z, measuring force at each (x,y).

    At each position, runs a short relaxation (n_relax steps) with strong damping
    to let forces settle, then records force and energy.

    Returns dict with 'X', 'Y', 'Fz', 'Fx', 'Fy', 'E', 'atom_positions' arrays.
    """
    from spammm.forcefields.RigidBodyDynamics import _ensure_float4
    nx, ny = len(xs), len(ys)
    Fz = np.zeros((nx, ny), dtype=np.float32)
    Fx = np.zeros((nx, ny), dtype=np.float32)
    Fy = np.zeros((nx, ny), dtype=np.float32)
    E_grid = np.zeros((nx, ny), dtype=np.float32)
    atom_pos_grid = []

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            pos = np.array([[x, y, z, rbd.mass_trans]], dtype=np.float32)
            quat = np.array([[0, 0, 0, 1]], dtype=np.float32)
            rbd.reset_pose(pos, quat)
            rbd.run_folded(n_relax, dt, lin_damp=0.98, ang_damp=0.95)
            out = rbd.download_outputs()
            f = out['body_force'][0]
            atoms = out['atom_positions'][0]
            E = float(atoms[:, 3].sum())
            Fx[ix, iy] = f[0]
            Fy[ix, iy] = f[1]
            Fz[ix, iy] = f[2]
            E_grid[ix, iy] = E
            atom_pos_grid.append(atoms[:, :3].copy())

    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return {
        'X': X, 'Y': Y,
        'Fx': Fx, 'Fy': Fy, 'Fz': Fz, 'E': E_grid,
        'atom_positions': atom_pos_grid,
    }


def relaxed_scan(rbd, pin_atom_idx, path, k_spring=5.0, n_relax=200, dt=0.005,
                 lin_damp=0.99, ang_damp=0.95, record_interval=10):
    """Relaxed scan: pin one atom with a spring and move it along a path.

    At each path point, the pinned atom is held by a harmonic spring to the target
    position. The rest of the molecule relaxes on the surface potential. This simulates
    AFM manipulation: dragging a molecule by one atom across the substrate.

    Args:
        rbd: RigidBodyDynamics instance (with folded basis initialized)
        pin_atom_idx: index of the atom to pin with spring
        path: (N, 3) array of target (x,y,z) positions for the pinned atom
        k_spring: spring constant in eV/Å² (higher = stiffer constraint)
        n_relax: relaxation steps per path point
        dt: timestep for relaxation
        lin_damp: linear damping factor per step
        ang_damp: angular damping factor per step
        record_interval: record state every N relaxation steps within each path point

    Returns:
        dict with 'positions', 'quaternions', 'forces', 'torques', 'atom_positions',
        'path', 'pin_forces' (spring force on pinned atom at each path point)
    """
    n_path = len(path)
    positions = []
    quaternions = []
    forces = []
    torques = []
    atom_positions_list = []
    pin_forces = []

    for i in range(n_path):
        target = np.asarray(path[i], dtype=np.float32)
        anchors = np.zeros((rbd.total_atoms, 4), dtype=np.float32)
        anchors[:, 3] = -1.0
        anchors[pin_atom_idx, :3] = target
        anchors[pin_atom_idx, 3] = k_spring
        rbd.update_anchors(anchors)

        for j in range(0, n_relax, record_interval):
            nrun = min(record_interval, n_relax - j)
            rbd.run_folded(nrun, dt, lin_damp=lin_damp, ang_damp=ang_damp)
            out = rbd.download_outputs()
            positions.append(out['pos'][0].copy())
            quaternions.append(out['quats'][0].copy())
            forces.append(out['body_force'][0][:3].copy())
            torques.append(out['body_torque'][0][:3].copy())
            atom_positions_list.append(out['atom_positions'][0][:, :3].copy())

        out = rbd.download_outputs()
        atom_f = out['atom_force'][0]
        pin_forces.append(atom_f[pin_atom_idx][:3].copy())

    n_rec = len(positions)
    return {
        'positions': np.array(positions),
        'quaternions': np.array(quaternions),
        'forces': np.array(forces),
        'torques': np.array(torques),
        'atom_positions': atom_positions_list,
        'path': np.asarray(path, dtype=np.float32),
        'pin_forces': np.array(pin_forces),
        'n_path': n_path,
        'n_relax': n_relax,
        'record_interval': record_interval,
    }


def manipulation_trajectory(rbd, x0, y0, z, dx, n_steps=50, dt=0.02, n_relax_per_step=20):
    """Simulate manipulation: move molecule laterally in small increments.

    At each step, shift target position by dx/n_steps, relax briefly, record state.
    Returns dict with 'positions', 'forces', 'atom_positions', 'path'.
    """
    positions = []
    forces = []
    atom_positions_list = []
    path = np.zeros((n_steps, 3), dtype=np.float32)

    for i in range(n_steps):
        frac = (i + 1) / n_steps
        x = x0 + dx[0] * frac
        y = y0 + dx[1] * frac
        pos = np.array([[x, y, z, rbd.mass_trans]], dtype=np.float32)
        quat = np.array([[0, 0, 0, 1]], dtype=np.float32)
        rbd.reset_pose(pos, quat)
        rbd.run_folded(n_relax_per_step, dt, lin_damp=0.98, ang_damp=0.95)
        out = rbd.download_outputs()
        f = out['body_force'][0]
        atoms = out['atom_positions'][0]
        positions.append(out['pos'][0].copy())
        forces.append(f[:3].copy())
        atom_positions_list.append(atoms[:, :3].copy())
        path[i] = [x, y, z]

    return {
        'positions': np.array(positions),
        'forces': np.array(forces),
        'atom_positions': atom_positions_list,
        'path': path,
    }


# =============================================================================
# I/O
# =============================================================================

def save_xyz_trajectory(filename, mol_enames, mol_positions_list, sub_apos=None, sub_enames=None, comments=None):
    """Save trajectory as multi-frame XYZ including substrate atoms for context."""
    na_mol = len(mol_enames)
    na_sub = len(sub_enames) if sub_apos is not None else 0
    na_total = na_mol + na_sub
    with open(filename, 'w') as f:
        for idx, mol_pos in enumerate(mol_positions_list):
            comment = comments[idx] if comments else f'frame {idx}'
            f.write(f'{na_total}\n{comment}\n')
            if sub_apos is not None:
                for e, p in zip(sub_enames, sub_apos):
                    f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
            for e, p in zip(mol_enames, mol_pos):
                f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
