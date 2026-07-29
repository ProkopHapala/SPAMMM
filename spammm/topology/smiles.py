"""
smiles.py — SMILES → AtomicGraph (topology SSOT).

Purpose: Build editable molecular topology from SMILES strings for CLI / GUI /
tests. Prefer optional RDKit when available; otherwise a pure-Python OpenSMILES
organic-subset parser + 2D spring embedding (networkx).

Public API:
  parse_smiles(s, add_h=True, embed='2d') -> AtomicGraph
  smiles_to_system(s, ...) -> AtomicSystem
  SMILES_EXAMPLES  — named demo molecules

See: doc/Tasks/SPM_CLI_Headless.md §C, ARCHITECTURE_ROADMAP §9 / T07.
"""
from __future__ import annotations

import numpy as np

from spammm import elements as el
from spammm.topology.AtomicGraph import AtomicGraph
from spammm.AtomicSystem import AtomicSystem


# ---------------------------------------------------------------------------
# Named examples (recognizable organics for demos / tests)
# ---------------------------------------------------------------------------

SMILES_EXAMPLES = {
    'benzene':           'c1ccccc1',
    'hydroquinone':      'c1cc(ccc1O)O',
    'benzoic_acid':      'O=C(O)c1ccccc1',
    'maleic_anhydride':  'C1=CC(=O)OC1=O',
    'terephthalic_acid': 'O=C(O)c1ccc(C(O)=O)cc1',
    'naphthalene':       'c1c2ccccc2ccc1',
    'azulene':           'c1cccc2cccc2c1',
    'guanine':           'OC1=C2N=CNC2=NC(=N1)N',
    'thymine':           'O=C1NC(=O)NC=C1C',
}

SMILES_HEAVY_COUNTS = {
    'benzene': 6,
    'hydroquinone': 8,
    'benzoic_acid': 9,
    'maleic_anhydride': 7,
    'terephthalic_acid': 12,
    'naphthalene': 10,
    'azulene': 10,
    'guanine': 11,
    'thymine': 9,
}

SMILES_ATOM_COUNTS_WITH_H = {
    'benzene': 12,            # C6H6
    'hydroquinone': 14,       # C6H6O2
    'benzoic_acid': 15,       # C7H6O2
    'maleic_anhydride': 9,    # C4H2O3
    'terephthalic_acid': 18,  # C8H6O4
    'naphthalene': 18,        # C10H8
    'azulene': 18,            # C10H8
    'guanine': 16,            # C5H5N5O
    'thymine': 15,            # C5H6N2O2
}


def _try_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return Chem, AllChem
    except ImportError:
        return None, None


def _parse_smiles_rdkit(s: str, add_h: bool, embed: str) -> AtomicGraph:
    Chem, AllChem = _try_rdkit()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {s!r}")
    if add_h:
        mol = Chem.AddHs(mol)
    if embed == '2d':
        AllChem.Compute2DCoords(mol)
    elif embed == '3d':
        AllChem.EmbedMolecule(mol, randomSeed=0)
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    elif embed not in ('none', None, False):
        raise ValueError(f"Unknown embed={embed!r}")

    conf = mol.GetConformer() if mol.GetNumConformers() else None
    graph = AtomicGraph()
    atoms = []
    for i, a in enumerate(mol.GetAtoms()):
        sym = a.GetSymbol()
        z = int(a.GetAtomicNum())
        if conf is not None:
            p = conf.GetAtomPosition(i)
            pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        else:
            pos = np.array([float(i), 0.0, 0.0], dtype=np.float64)
        if sym == 'H':
            npi = -1
        elif a.GetIsAromatic() or a.GetHybridization().name in ('SP2', 'SP'):
            npi = 2 if a.GetHybridization().name == 'SP' else 1
        else:
            npi = 0
        atoms.append(graph.add_atom(pos, sym, z, pin=None, parent=None, npi=npi))

    heavies = [a for a in atoms if a.ename != 'H']
    for a in atoms:
        if a.ename != 'H':
            continue
        best, bd = None, 1e9
        for h in heavies:
            d = float(np.linalg.norm(a.pos - h.pos))
            if d < bd:
                best, bd = h, d
        a.parent = best

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        order = 1.5 if b.GetIsAromatic() else float(b.GetBondTypeAsDouble())
        graph.add_bond(atoms[i], atoms[j], order=order)
    return graph


_ORGANIC = {
    'B': 'B', 'C': 'C', 'N': 'N', 'O': 'O', 'P': 'P', 'S': 'S',
    'F': 'F', 'Cl': 'Cl', 'Br': 'Br', 'I': 'I',
    'b': 'B', 'c': 'C', 'n': 'N', 'o': 'O', 'p': 'P', 's': 'S',
}
_AROMATIC_TOKENS = set('bcnops')

_VALENCE = {
    'B': 3, 'C': 4, 'N': 3, 'O': 2, 'P': 3, 'S': 2,
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'H': 1,
}

_BOND_CHAR = {'-': 1.0, '=': 2.0, '#': 3.0, ':': 1.5}


class _AtomRec:
    __slots__ = ('ename', 'aromatic', 'hcount', 'charge', 'explicit_h')

    def __init__(self, ename, aromatic=False, hcount=None, charge=0):
        self.ename = ename
        self.aromatic = aromatic
        self.hcount = hcount
        self.charge = charge
        self.explicit_h = hcount is not None


def _read_organic_atom(s, i):
    if i + 1 < len(s) and s[i:i + 2] in ('Cl', 'Br'):
        return _AtomRec(s[i:i + 2], aromatic=False), i + 2
    c = s[i]
    if c not in _ORGANIC:
        raise ValueError(f"Unsupported atom token {c!r} at pos {i} in {s!r}")
    aromatic = c in _AROMATIC_TOKENS
    return _AtomRec(_ORGANIC[c], aromatic=aromatic), i + 1


def _read_bracket_atom(s, i):
    assert s[i] == '['
    j = s.find(']', i)
    if j < 0:
        raise ValueError(f"Unclosed '[' in SMILES: {s!r}")
    body = s[i + 1:j]
    k = 0
    while k < len(body) and body[k].isdigit():
        k += 1
    if k >= len(body):
        raise ValueError(f"Empty element in bracket atom [{body}]")
    if body[k].isupper() and k + 1 < len(body) and body[k + 1].islower():
        sym = body[k:k + 2]
        aromatic = False
        k += 2
    elif body[k].isupper() or body[k] in 'bcnops':
        sym = body[k]
        aromatic = sym in _AROMATIC_TOKENS
        if aromatic:
            sym = _ORGANIC[sym]
        k += 1
    else:
        raise ValueError(f"Bad element in [{body}]")
    hcount = 0
    if k < len(body) and body[k] == 'H':
        k += 1
        if k < len(body) and body[k].isdigit():
            hcount = int(body[k]); k += 1
        else:
            hcount = 1
    charge = 0
    if k < len(body) and body[k] in '+-':
        sign = 1 if body[k] == '+' else -1
        k += 1
        if k < len(body) and body[k].isdigit():
            charge = sign * int(body[k]); k += 1
        else:
            n = 1
            while k < len(body) and body[k] in '+-':
                n += 1; k += 1
            charge = sign * n
    return _AtomRec(sym, aromatic=aromatic, hcount=hcount, charge=charge), j + 1


def _parse_smiles_topology(s: str):
    s = s.strip()
    if not s:
        raise ValueError("Empty SMILES")
    atoms: list[_AtomRec] = []
    bonds: list[tuple[int, int, float]] = []
    branch: list[int] = []
    rings: dict[int, tuple[int, float | None]] = {}
    prev: int | None = None
    pending_bond: float | None = None
    i = 0
    n = len(s)

    def add_bond(a, b, order):
        if a == b:
            raise ValueError(f"Self-bond in SMILES: {s!r}")
        a0, b0 = (a, b) if a < b else (b, a)
        for x, y, _ in bonds:
            if x == a0 and y == b0:
                return
        bonds.append((a0, b0, order))

    def default_order(ia, ib):
        if atoms[ia].aromatic and atoms[ib].aromatic:
            return 1.5
        return 1.0

    while i < n:
        c = s[i]
        if c == '(':
            if prev is None:
                raise ValueError(f"Branch '(' with no previous atom in {s!r}")
            branch.append(prev); i += 1; continue
        if c == ')':
            if not branch:
                raise ValueError(f"Unmatched ')' in {s!r}")
            prev = branch.pop(); i += 1; continue
        if c in _BOND_CHAR:
            pending_bond = _BOND_CHAR[c]; i += 1; continue
        if c == '%':
            if i + 2 >= n or not s[i + 1:i + 3].isdigit():
                raise ValueError(f"Bad ring %%nn at pos {i} in {s!r}")
            rnum = int(s[i + 1:i + 3]); i += 3
            if prev is None:
                raise ValueError(f"Ring digit with no atom in {s!r}")
            if rnum in rings:
                j, bo_open = rings.pop(rnum)
                bo = pending_bond if pending_bond is not None else (bo_open if bo_open is not None else default_order(prev, j))
                add_bond(prev, j, bo); pending_bond = None
            else:
                rings[rnum] = (prev, pending_bond); pending_bond = None
            continue
        if c.isdigit():
            rnum = int(c); i += 1
            if prev is None:
                raise ValueError(f"Ring digit with no atom in {s!r}")
            if rnum in rings:
                j, bo_open = rings.pop(rnum)
                bo = pending_bond if pending_bond is not None else (bo_open if bo_open is not None else default_order(prev, j))
                add_bond(prev, j, bo); pending_bond = None
            else:
                rings[rnum] = (prev, pending_bond); pending_bond = None
            continue
        if c == '[':
            rec, i = _read_bracket_atom(s, i)
        elif c.isalpha():
            rec, i = _read_organic_atom(s, i)
        else:
            raise ValueError(f"Unexpected {c!r} at pos {i} in SMILES {s!r}")

        ia = len(atoms)
        atoms.append(rec)
        if prev is not None:
            bo = pending_bond if pending_bond is not None else default_order(prev, ia)
            add_bond(prev, ia, bo)
            pending_bond = None
        prev = ia

    if branch:
        raise ValueError(f"Unclosed '(' in SMILES: {s!r}")
    if rings:
        raise ValueError(f"Unclosed rings {sorted(rings)} in SMILES: {s!r}")
    if pending_bond is not None:
        raise ValueError(f"Trailing bond symbol in SMILES: {s!r}")
    return atoms, bonds


def _bond_order_sum(ia, bonds):
    """Sum bond orders for implicit-H. Aromatic bonds count as 1.5 (→ benzene CH)."""
    s = 0.0
    for a, b, o in bonds:
        if a == ia or b == ia:
            s += float(o)
    return s


def _needed_h(rec: _AtomRec, ia: int, bonds) -> int:
    if rec.explicit_h:
        return int(rec.hcount or 0)
    val = _VALENCE.get(rec.ename)
    if val is None:
        return 0
    used = _bond_order_sum(ia, bonds)
    if rec.ename == 'N' and used > 3:
        val = 5
    if rec.ename == 'P' and used > 3:
        val = 5
    if rec.ename == 'S' and used > 2:
        val = 4 if used <= 4 else 6
    n = int(round(val - used - rec.charge))
    return max(0, n)


def _ideal_bond_length(order: float, ea: str, eb: str) -> float:
    if 'H' in (ea, eb):
        return 1.09
    if abs(order - 3.0) < 1e-6:
        return 1.20
    if abs(order - 2.0) < 1e-6:
        return 1.34
    if abs(order - 1.5) < 1e-6:
        return 1.40
    return 1.50


def _embed_2d(n: int, bonds, enames, n_iters=80, seed=0) -> np.ndarray:
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, o in bonds:
        G.add_edge(a, b, weight=_ideal_bond_length(o, enames[a], enames[b]))
    if n == 1:
        return np.zeros((1, 3), dtype=np.float64)
    if n == 2:
        L = G[0][1]['weight']
        return np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]], dtype=np.float64)

    pos = nx.spring_layout(G, weight='weight', seed=seed, dim=2, iterations=200)
    try:
        if nx.is_connected(G) and n <= 80:
            pos = nx.kamada_kawai_layout(G, weight='weight', dim=2)
    except Exception:
        pass

    xy = np.array([pos[i] for i in range(n)], dtype=np.float64)
    lengths, ideals = [], []
    for a, b, o in bonds:
        lengths.append(np.linalg.norm(xy[a] - xy[b]))
        ideals.append(_ideal_bond_length(o, enames[a], enames[b]))
    mean_L = float(np.mean(lengths)) if lengths else 1.0
    mean_I = float(np.mean(ideals)) if ideals else 1.4
    if mean_L > 1e-8:
        xy *= (mean_I / mean_L)

    for _ in range(n_iters):
        forces = np.zeros_like(xy)
        for a, b, o in bonds:
            L0 = _ideal_bond_length(o, enames[a], enames[b])
            d = xy[b] - xy[a]
            r = float(np.linalg.norm(d))
            if r < 1e-8:
                d = np.array([1e-3, 0.0]); r = 1e-3
            f = 0.5 * (r - L0) * (d / r)
            forces[a] += f
            forces[b] -= f
        if n <= 40:
            for i in range(n):
                for j in range(i + 1, n):
                    if G.has_edge(i, j):
                        continue
                    d = xy[j] - xy[i]
                    r2 = float(np.dot(d, d))
                    if r2 < 1e-8 or r2 > 9.0:
                        continue
                    r = np.sqrt(r2)
                    if r < 1.6:
                        f = 0.05 * (1.6 - r) * (d / r)
                        forces[i] -= f
                        forces[j] += f
        xy += 0.35 * forces

    out = np.zeros((n, 3), dtype=np.float64)
    out[:, :2] = xy
    out -= out.mean(axis=0, keepdims=True)
    out[:, 2] = 0.0
    return out


def _place_hydrogens(apos, bonds, enames, n_h_per_atom):
    apos = [np.asarray(p, dtype=np.float64) for p in apos]
    enames = list(enames)
    bonds = list(bonds)
    parents = []
    n0 = len(enames)
    neigh = [[] for _ in range(n0)]
    for a, b, o in bonds:
        neigh[a].append(b)
        neigh[b].append(a)

    for ia in range(n0):
        nh = n_h_per_atom[ia]
        if nh <= 0:
            continue
        p = apos[ia]
        vecs = []
        for j in neigh[ia]:
            v = apos[j] - p
            nrm = np.linalg.norm(v)
            if nrm > 1e-8:
                vecs.append(v / nrm)
        if not vecs:
            dirs = [np.array([np.cos(2 * np.pi * k / max(nh, 1)), np.sin(2 * np.pi * k / max(nh, 1)), 0.0]) for k in range(nh)]
        else:
            mean = np.sum(vecs, axis=0)
            mn = np.linalg.norm(mean)
            mean = np.array([1.0, 0.0, 0.0]) if mn < 1e-8 else (-mean / mn)
            t = np.array([-mean[1], mean[0], 0.0])
            tn = np.linalg.norm(t)
            t = np.array([0.0, 1.0, 0.0]) if tn < 1e-8 else (t / tn)
            if nh == 1:
                dirs = [mean]
            else:
                dirs = []
                for k in range(nh):
                    ang = (k - 0.5 * (nh - 1)) * (np.pi / max(nh, 2))
                    d = np.cos(ang) * mean + np.sin(ang) * t
                    dn = np.linalg.norm(d)
                    dirs.append(d / dn if dn > 1e-8 else mean)
        for d in dirs[:nh]:
            ih = len(enames)
            apos.append(p + 1.09 * d)
            enames.append('H')
            bonds.append((ia, ih, 1.0))
            parents.append(ia)
            neigh.append([])
            neigh[ia].append(ih)
    return np.array(apos, dtype=np.float64), enames, bonds, parents


def _npi_from_bonds(i, bonds, n_heavy, aromatic_flag):
    if aromatic_flag:
        return 1
    for a, b, o in bonds:
        if (a == i or b == i) and a < n_heavy and b < n_heavy and o >= 1.5:
            return 1
    return 0


def _parse_smiles_pure(s: str, add_h: bool, embed: str) -> AtomicGraph:
    recs, bonds = _parse_smiles_topology(s)
    n = len(recs)
    enames = [r.ename for r in recs]
    n_h = [_needed_h(recs[i], i, bonds) for i in range(n)]

    if embed in ('2d', '3d', True):
        apos = _embed_2d(n, bonds, enames)
    else:
        apos = np.zeros((n, 3), dtype=np.float64)
        apos[:, 0] = np.arange(n, dtype=np.float64)

    parents_h = []
    if add_h:
        apos, enames, bonds, parents_h = _place_hydrogens(apos, bonds, enames, n_h)

    graph = AtomicGraph()
    atoms = []
    for i, e in enumerate(enames):
        z = int(el.ELEMENT_DICT[e][0])
        if e == 'H':
            at = graph.add_atom(apos[i], e, z, pin=None, parent=None, npi=-1)
        else:
            npi = _npi_from_bonds(i, bonds, n, recs[i].aromatic)
            at = graph.add_atom(apos[i], e, z, pin=None, parent=None, npi=npi)
            at.charge = float(recs[i].charge)
        atoms.append(at)

    h_k = 0
    for i, e in enumerate(enames):
        if e != 'H':
            continue
        atoms[i].parent = atoms[parents_h[h_k]]
        h_k += 1

    for a, b, o in bonds:
        graph.add_bond(atoms[a], atoms[b], order=float(o))
    return graph


def parse_smiles(s: str, add_h: bool = True, embed: str = '2d', engine: str = 'auto') -> AtomicGraph:
    """Parse SMILES into an AtomicGraph.

    Args:
        s: SMILES string
        add_h: add implicit hydrogens
        embed: '2d' | '3d' | 'none'
        engine: 'auto' | 'rdkit' | 'pure' — auto uses RDKit if importable
    """
    Chem, _ = _try_rdkit()
    use_rdkit = (engine == 'rdkit') or (engine == 'auto' and Chem is not None)
    if engine == 'rdkit' and Chem is None:
        raise ImportError("RDKit requested but not installed")
    if use_rdkit:
        return _parse_smiles_rdkit(s, add_h=add_h, embed=embed)
    return _parse_smiles_pure(s, add_h=add_h, embed=embed)


def smiles_to_system(s: str, **kwargs) -> AtomicSystem:
    """Parse SMILES → AtomicSystem (flat arrays for FF / DFTB / SPM).

    Default 2D embed is perfectly planar (all atom z = 0).
    """
    graph = parse_smiles(s, **kwargs)
    _atom_list, enames, apos, atypes, bonds, _bond_list, _rings = graph.to_arrays()
    apos = np.asarray(apos, dtype=np.float64).copy()
    if kwargs.get('embed', '2d') in ('2d', True, None):
        apos[:, 2] = 0.0
    sys = AtomicSystem(apos=apos, atypes=atypes, enames=list(enames), bonds=bonds, bPreinit=True)
    sys.graph = graph
    return sys


def graph_counts(graph: AtomicGraph):
    """Return (n_atoms, n_heavy, n_bonds, composition dict)."""
    atoms = [a for a in graph.atoms.values() if a.alive]
    bonds = [b for b in graph.bonds.values() if b.alive and b.a.alive and b.b.alive]
    comp = {}
    for a in atoms:
        comp[a.ename] = comp.get(a.ename, 0) + 1
    n_heavy = sum(1 for a in atoms if a.ename != 'H')
    return len(atoms), n_heavy, len(bonds), comp
