"""
molecule_loaders.py — Shared rigid-body molecule loaders (XYZ/mol2 + QEq + bonds) + graph-to-fragments splitter.

Purpose: load molecules for rigid-body workflows (PairFF, Assembly, RigidAssembly GUI
extension, MC/GA) — planarized, XY-centered, with QEq charges and intramolecular bonds.
Also splits an `AtomicGraph` into connected components (independent fragments) for the
"From editor" build path. Reused by `tests/testplot_pairff_energy_mc.py` and
`spammm/GUI/RigidAssemblyExtension.py`.

Returns (apos, enames, REQs, bonds) tuples ready for `_prepare_molecule_pack` /
`RigidBodyPairFF.from_molecules`. `graph_to_rigid_fragments` returns
(fragments, coms) where each fragment is a body-frame (CoM-centered) tuple and coms
are the mass-weighted centers of mass.

Conventions:
  - apos: float32 (n,3), XY-centered (mean subtracted), z planarized to 0
  - enames: list[str] element symbols
  - REQs: float32 (n,4) from `make_REQs_from_enames` with QEq charges
    (physical partial charges = -QEq occupancy; PairFF audit SSOT)
  - bonds: int32 (m,2) from `AtomicSystem.neighs(bBond=True)`

No plotting, no OpenCL — pure data loading. QEq uses `spammm.forcefields.QEq`.
Atomic masses from `spammm.elements.ELEMENT_DICT` (index 10).
"""
import os
import numpy as np

from spammm import elements
from spammm.AtomicSystem import AtomicSystem
from spammm.forcefields.QEq import solve_from_elements
from spammm.topology.FFparams import (
    read_atom_types, read_element_types, make_REQs_from_enames, load_xyz_with_REQs, _DATA_PATH,
)

# Repo root (parents: this file → forcefields/ → spammm/ → repo)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MOL_DIR = os.path.join(_REPO, 'data', 'mol')
_XYZ_DIR = os.path.join(_REPO, 'data', 'xyz')

# Canonical molecule data paths (shared between testplot and GUI extension)
MOL_PATHS = {
    'NTCDI':             os.path.join(_MOL_DIR, 'NTCDI.mol2'),
    'TBTAP':             os.path.join(_MOL_DIR, 'TBTAP.mol2'),
    'PTCDA':             os.path.join(_XYZ_DIR, 'PTCDA.xyz'),
    'HCOOH':             os.path.join(_XYZ_DIR, 'HCOOH.xyz'),
    'terephthalic_acid': os.path.join(_XYZ_DIR, 'terephthalic_acid.xyz'),
    'azaindol':          os.path.join(_XYZ_DIR, 'azaindol.xyz'),
    'uracil':            os.path.join(_XYZ_DIR, 'uracil.xyz'),
    'adenine':           os.path.join(_XYZ_DIR, 'adenine.xyz'),
}

# FAF substrate fits per molecule (key = molecule name; value = .npz fit path).
# Molecules sharing element sets reuse the broadest fit (ptcdi: C,N,O,H).
FAF_FITS = {
    'formic_acid':       os.path.join(_REPO, 'data', 'fits', 'hcooh_nacl.npz'),
    'PTCDA':             os.path.join(_REPO, 'data', 'fits', 'ptcda_nacl.npz'),
    'NTCDI':             os.path.join(_REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,O,H
    'terephthalic_acid': os.path.join(_REPO, 'data', 'fits', 'ptcda_nacl.npz'),  # C,O,H
    'TBTAP':             os.path.join(_REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,O,H — Br maps to closest
    'azaindol':          os.path.join(_REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,H
    'uracil':            os.path.join(_REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,O,H
    'adenine':           os.path.join(_REPO, 'data', 'fits', 'ptcdi_nacl.npz'),  # C,N,H
}
# Default fit for multi-species runs (must cover all elements)
FAF_FIT_DEFAULT = os.path.join(_REPO, 'data', 'fits', 'ptcdi_nacl.npz')  # C,N,O,H — broadest coverage


def atom_types_dict():
    """Load (element_types, atom_types) from FFparams data files (cached on call site)."""
    etypes = read_element_types(os.path.join(_DATA_PATH, 'ElementTypes.dat'))
    return etypes, read_atom_types(os.path.join(_DATA_PATH, 'AtomTypes.dat'), etypes)


def bonds_from_geom(apos, enames):
    """Infer intramolecular bonds from geometry via AtomicSystem.neighs(bBond=True)."""
    atypes = [elements.ELEMENT_DICT[e][0] if e in elements.ELEMENT_DICT else 6 for e in enames]
    mol = AtomicSystem(apos=np.asarray(apos, dtype=np.float32).copy(), atypes=atypes, enames=list(enames))
    mol.neighs(bBond=True)
    if mol.bonds is None or len(mol.bonds) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(mol.bonds, dtype=np.int32)


def load_xyz_generic(path, qeq=True, name='', etypes_atom_types=None):
    """Generic XYZ loader with QEq charges, XY-centering, and z-planarization.

    Returns (apos (n,3) f32, enames list[str], REQs (n,4) f32, bonds (m,2) i32).
    etypes_atom_types: optional (etypes, atom_types) tuple to avoid re-reading data files.
    """
    etypes, atom_types = etypes_atom_types if etypes_atom_types is not None else atom_types_dict()
    apos, REQs, enames, Zs, lvec = load_xyz_with_REQs(path, atom_types=atom_types)
    apos = np.asarray(apos, dtype=np.float32)
    enames = [str(e) for e in enames]
    apos[:, :2] -= apos[:, :2].mean(axis=0)
    apos[:, 2] = 0.0
    if qeq:
        q = -solve_from_elements(apos, enames, etypes, Q_target=0.0)
        REQs = make_REQs_from_enames(enames, q.astype(np.float32), atom_types)
        print(f'  {name} QEq: sum={q.sum():.4f}  Q range=[{q.min():.3f},{q.max():.3f}]')
    bonds = bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def load_mol2_generic(path, qeq=True, name='', etypes_atom_types=None):
    """Generic mol2 loader via AtomicSystem with QEq charges.

    Returns (apos (n,3) f32, enames list[str], REQs (n,4) f32, bonds (m,2) i32).
    etypes_atom_types: optional (etypes, atom_types) tuple to avoid re-reading data files.
    """
    etypes, atom_types = etypes_atom_types if etypes_atom_types is not None else atom_types_dict()
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
    bonds = bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def graph_to_rigid_fragments(graph, qeq=True, planarize=True):
    """Split an AtomicGraph into independent rigid-body fragments (connected components).

    For each connected component (fragment not linked by any bond):
      - extract (apos, enames)
      - compute mass-weighted CoM (body center) using atomic masses from `elements`
      - build body-frame relative positions: apos - CoM
      - build REQs via `make_REQs_from_enames` with charges from QEq (if qeq=True) or
        from `atom.charge` (if already set on the graph) or 0
      - build bonds array from the graph's bonds within the fragment

    Returns (fragments, coms):
      fragments: list of (apos_rel (n,3) f32, enames list[str], REQs (n,4) f32, bonds (m,2) i32)
                 apos_rel is body-frame (CoM-centered); z planarized to 0 if planarize=True
      coms: (n_frags, 3) f32 — mass-weighted center of mass per fragment (body positions)

    Reuses: `AtomicGraph.find_connected_components`, `make_REQs_from_enames`, `QEq.solve_from_elements`,
    `molecule_loaders.bonds_from_geom`, `elements.ELEMENT_DICT` for atomic masses.
    """
    from spammm import elements as _el
    from spammm.forcefields.QEq import solve_from_elements

    components = graph.find_connected_components()
    if not components:
        return [], np.zeros((0, 3), dtype=np.float32)

    # Collect full-system arrays for QEq (run once on the whole graph)
    all_atoms = [a for comp in components for a in comp]
    all_apos = np.array([a.pos for a in all_atoms], dtype=np.float32)
    all_enames = [str(a.ename) for a in all_atoms]
    if planarize:
        all_apos = all_apos.copy()
        all_apos[:, 2] = 0.0

    # Charges: run QEq on the whole system once, then split per fragment
    if qeq:
        etypes, atom_types = atom_types_dict()
        q_all = -solve_from_elements(all_apos, all_enames, etypes, Q_target=0.0)
        print(f'  graph_to_rigid_fragments: QEq sum={q_all.sum():.4f}  Q range=[{q_all.min():.3f},{q_all.max():.3f}]')
    else:
        # Use existing atom.charge if set, else 0
        q_all = np.array([float(a.charge) for a in all_atoms], dtype=np.float32)

    # Map atom → index in all_atoms for charge lookup
    atom_to_idx = {a._id: i for i, a in enumerate(all_atoms)}

    etypes, atom_types = atom_types_dict() if not qeq else (etypes, atom_types)

    fragments = []
    coms = np.zeros((len(components), 3), dtype=np.float32)

    for fi, comp in enumerate(components):
        n = len(comp)
        apos = np.array([a.pos for a in comp], dtype=np.float32)
        enames = [str(a.ename) for a in comp]
        masses = np.array([_el.ELEMENT_DICT[e][10] if e in _el.ELEMENT_DICT else 12.0 for e in enames], dtype=np.float32)
        # Mass-weighted CoM
        mtot = masses.sum()
        if mtot <= 0:
            com = apos.mean(axis=0)
        else:
            com = (apos * masses[:, None]).sum(axis=0) / mtot
        coms[fi] = com.astype(np.float32)
        # Body-frame relative positions
        apos_rel = (apos - com).astype(np.float32)
        if planarize:
            apos_rel[:, 2] = 0.0
        # Charges for this fragment
        q_frag = np.array([q_all[atom_to_idx[a._id]] for a in comp], dtype=np.float32)
        REQs = make_REQs_from_enames(enames, q_frag, atom_types)
        # Bonds within this fragment (use geometry since graph bonds may not map cleanly)
        bonds = bonds_from_geom(apos_rel, enames)
        fragments.append((apos_rel, enames, REQs, bonds))

    return fragments, coms


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


# ─── per-molecule loaders (canonical set) ────────────────────────────────────
# Each returns (apos, enames, REQs, bonds). Cached (etypes, atom_types) is read once
# per process via a module-level lazy cache to avoid re-reading FFparams files.
_ETA_CACHE = None
def _eta():
    global _ETA_CACHE
    if _ETA_CACHE is None:
        _ETA_CACHE = atom_types_dict()
    return _ETA_CACHE


def load_ntcdi(qeq=False):
    """NTCDI from mol2 (uses file charges; QEq off by default — mol2 has good charges)."""
    etypes, atom_types = _eta()
    mol = AtomicSystem(fname=MOL_PATHS['NTCDI'])
    apos = np.asarray(mol.apos, dtype=np.float32)
    enames = [str(e) for e in mol.enames]
    qs = np.asarray(mol.qs, dtype=np.float32) if mol.qs is not None else np.zeros(len(enames), np.float32)
    REQs = make_REQs_from_enames(enames, qs, atom_types)
    apos[:, :2] -= apos[:, :2].mean(axis=0)
    apos[:, 2] = 0.0
    bonds = bonds_from_geom(apos, enames)
    return apos, enames, REQs, bonds


def load_ptcda(qeq=True):
    return load_xyz_generic(MOL_PATHS['PTCDA'], qeq=qeq, name='PTCDA', etypes_atom_types=_eta())


def load_formic_acid(qeq=True):
    return load_xyz_generic(MOL_PATHS['HCOOH'], qeq=qeq, name='formic_acid', etypes_atom_types=_eta())


def load_terephthalic_acid(qeq=True):
    return load_xyz_generic(MOL_PATHS['terephthalic_acid'], qeq=qeq, name='terephthalic_acid', etypes_atom_types=_eta())


def load_uracil(qeq=True):
    return load_xyz_generic(MOL_PATHS['uracil'], qeq=qeq, name='uracil', etypes_atom_types=_eta())


def load_adenine(qeq=True):
    return load_xyz_generic(MOL_PATHS['adenine'], qeq=qeq, name='adenine', etypes_atom_types=_eta())


def load_azaindol(qeq=True):
    return load_xyz_generic(MOL_PATHS['azaindol'], qeq=qeq, name='azaindol', etypes_atom_types=_eta())


def load_tbtap(qeq=True):
    return load_mol2_generic(MOL_PATHS['TBTAP'], qeq=qeq, name='TBTAP', etypes_atom_types=_eta())


# Registry: name → loader callable (for GUI extension dropdown / CLI --mol)
LOADERS = {
    'NTCDI':             load_ntcdi,
    'PTCDA':             load_ptcda,
    'formic_acid':       load_formic_acid,
    'terephthalic_acid': load_terephthalic_acid,
    'TBTAP':             load_tbtap,
    'azaindol':          load_azaindol,
    'uracil':            load_uracil,
    'adenine':           load_adenine,
}
