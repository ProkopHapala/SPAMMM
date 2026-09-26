#!/usr/bin/env python3
"""
ascii_art_heterocycle.py — ASCII art heterocycle builder.

Supports two ASCII input formats:

1. Single-atom format: every row is an atom layer, characters are element
   symbols, spaces are empty.  Bonds are inferred from the zig-zag lattice
   topology.

       O o O
        c c
        c c
       c c c
       c c c
        c c
       O o O

2. Dimer format: rows alternate between atom rows and bond rows.
   Atom rows contain atom symbols (always converted to carbon in this format).
   Bond rows contain '|' (vertical dimer) and '-' (horizontal dimer).

       O o O
        | |
       | | |
        | |
       O o O

Each character column is 0.5 of the zig-zag lattice constant (dx/2).
Each character row is 0.5 of the armchair lattice constant (dy/2).
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from spammm.AtomicSystem import AtomicSystem
from spammm import elements as _elements
from spammm.topology.KekulePure import KekulePure, make_n_pi, run_kekule_solver, localize_kekule, mol_bond_types, export_mol
from spammm.topology.heterocycle_generator import plot_system, plot_kekule_phases

A_CC = 1.42


# ---------------------------------------------------------------------------
# Reusable H-bond helpers
# ---------------------------------------------------------------------------

def _resolve_hbond_donor(ia, ib, enames, eo=None):
    """Return (donor_idx, acceptor_idx) for an ASCII ':' H-bond pair.

    Uses standard chemistry rules:
      - N ... O  → N is donor, O is acceptor
      - N ... N  → prefer sp3 (lowercase in *eo*) as donor
    """
    ea = enames[ia]; eb = enames[ib]
    eau = ea.upper(); ebu = eb.upper()
    if (eau == 'N' and ebu == 'O'):
        return ia, ib
    elif (eau == 'O' and ebu == 'N'):
        return ib, ia
    elif (eau == 'N' and ebu == 'N') and (eo is not None):
        # Prefer the sp3 (lowercase in original ASCII) as donor.
        if (not eo[ia].isupper()) and eo[ib].isupper():
            return ia, ib
        elif (not eo[ib].isupper()) and eo[ia].isupper():
            return ib, ia
    # fallback: arbitrary ordering
    return (ia, ib) if ia < ib else (ib, ia)


def _build_target_valence(atoms, n_pi0):
    """Build target_valence dict for AtomicSystem.add_capping_h_sp2.

    Baseline uses MoleculeEditorBackend._target_sigma(element, npi).
    This is purely electron-counting from (element, n_pi) and should work for
    both uppercase/lowercase ASCII element symbols.
    """
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    kb = MoleculeEditorBackend()
    tv = {}
    for ia, e in enumerate(atoms.enames):
        eu = e.upper()
        if eu in ('C', 'N', 'O'):
            npi = 1 if float(n_pi0[ia]) > 0.5 else 0
            tv[ia] = kb._target_sigma(eu, npi)
    return tv


def resolve_hbond_pairs(atoms):
    """Resolve ASCII ':' H-bond heavy-atom pairs into (H_idx, acceptor_idx) segments.

    Donor selection is **data-driven**:
      - If exactly one side already has an attached H, that side is the donor.
      - Otherwise we fall back to element/case heuristics via ``_resolve_hbond_donor``.

    Then, among donor-attached H atoms, we pick the one most aligned with
    the donor→acceptor vector.

    Sets ``atoms.hbonds_ascii`` as a side effect.

    Returns:
        list of (int, int): (H_atom_index, acceptor_atom_index) pairs
    """
    hb_pairs = getattr(atoms, '_hbonds_pairs', [])
    if not hb_pairs:
        return []
    hb_ascii = []
    eo = getattr(atoms, '_enames_original', None)
    for ia, ib in hb_pairs:
        if atoms.ngs is None:
            atoms.neighs()
        na = list(atoms.ngs[ia].keys()) if hasattr(atoms.ngs[ia], 'keys') else list(atoms.ngs[ia])
        nb = list(atoms.ngs[ib].keys()) if hasattr(atoms.ngs[ib], 'keys') else list(atoms.ngs[ib])
        ha = [j for j in na if atoms.enames[j] == 'H']
        hb = [j for j in nb if atoms.enames[j] == 'H']
        if (len(ha) > 0) and (len(hb) == 0):
            donor, acc, h_neighs = ia, ib, ha
        elif (len(hb) > 0) and (len(ha) == 0):
            donor, acc, h_neighs = ib, ia, hb
        else:
            donor, acc = _resolve_hbond_donor(ia, ib, atoms.enames, eo)
            neighs = list(atoms.ngs[donor].keys()) if hasattr(atoms.ngs[donor], 'keys') else list(atoms.ngs[donor])
            h_neighs = [j for j in neighs if atoms.enames[j] == 'H']
        if not h_neighs:
            raise RuntimeError(f"ASCII ':' H-bond requires donor to have an attached H; donor atom {donor}({atoms.enames[donor]}) has none")
        pD = atoms.apos[donor]
        vA = atoms.apos[acc] - pD
        best = h_neighs[0]
        best_dot = -1e9
        for ih in h_neighs:
            vH = atoms.apos[ih] - pD
            d = float(np.dot(vH, vA))
            if d > best_dot:
                best_dot = d
                best = ih
        hb_ascii.append((best, acc))
    atoms.hbonds_ascii = hb_ascii
    return hb_ascii


# Solver orchestration functions (run_kekule_solver, localize_kekule, mol_bond_types, export_mol)
# have been moved to KekulePure.py — imported above.


# ---------------------------------------------------------------------------
# Bond-length relaxation
# ---------------------------------------------------------------------------
def jacobi_relax_bond_lengths(atoms, L0=1.42, n_iters=1, bmix=0.0):
    """Multi-step Jacobi bond-length relaxation with momentum acceleration.

    Each step is order-independent: a frozen copy of the current positions is
    used to compute all bond corrections, the per-atom displacements are
    accumulated, and only then the positions are updated.

    For each bond (i,j) we compute rij = ri - rj, the current distance d,
    and the vector delta = rij * (L0/d - 1) which would bring the bond exactly
    to length L0 if applied alone.  Half of delta is added to atom i and
    subtracted from atom j.

    Momentum: on the first step the full Jacobi displacement is applied.  From
    the second step onward the displacement is ``v = bmix * v + acc``, where
    ``v`` is the previous step's displacement.
    """
    pos = atoms.apos.copy()
    v = np.zeros_like(pos)
    for step in range(n_iters):
        acc = np.zeros_like(pos)
        for i, j in atoms.bonds:
            rij = pos[i] - pos[j]
            d = np.linalg.norm(rij)
            if d == 0.0:
                continue
            delta = rij * (L0 / d - 1.0)  # vector that takes this bond to length L0
            acc[i] += 0.5 * delta
            acc[j] -= 0.5 * delta
        if step == 0:
            v = acc
        else:
            v = bmix * v + acc
        pos += v
    atoms.apos = pos


def _lines(text):
    return [line.rstrip('\n') for line in text.strip('\n').splitlines()]


def _is_bond_row(line):
    return '|' in line or '-' in line


def _atom_tokens(line):
    """Return list of (col, char) for non-space characters."""
    return [(c, ch) for c, ch in enumerate(line) if ch != ' ']


def _col_to_xidx(r, c):
    """Map ASCII column to lattice x-index (one x-unit = 2 columns)."""
    return c // 2


# ---------------------------------------------------------------------------
# Dimer format -> direct AtomicSystem (x-shift is explicit via spaces)
# ---------------------------------------------------------------------------
def _build_dimer(lines, aCC=A_CC, hbond_length=None):
    dx = np.sqrt(3.0) * aCC
    pos = []
    enames = []
    enames_original = []  # keep case to distinguish sp2 (upper) vs sp3 (lower)
    rows = []  # per-atom row index
    xidxs = []  # per-atom x index (integer column//2)
    bonds = set()
    hbond_marks = []  # (r, c) where ':' denotes an H-bond between row r-1 and r+1

    row_kind = {}
    row_parity = {}
    for r, line in enumerate(lines):
        tokens = _atom_tokens(line)
        if not tokens:
            continue
        for c, ch in tokens:
            if ch == ':':
                hbond_marks.append((r, c))
        row_kind[r] = 'E' if any(ch.isalpha() for _, ch in tokens) else 'D'
        if row_kind[r] == 'E':
            for c, ch in tokens:
                if ch.isalpha():
                    row_parity[r] = c % 2
                    break

    y_list = {}
    y_pos = 0.0
    for r in range(len(lines)):
        if r == 0:
            y_list[r] = y_pos
            continue
        prev = row_kind.get(r - 1, 'E')
        cur = row_kind.get(r, 'E')
        if prev == 'E' and cur == 'E':
            p0 = row_parity.get(r - 1)
            p1 = row_parity.get(r)
            if (p0 is not None) and (p1 is not None) and (p0 != p1):
                y_pos += aCC / 2.0
            else:
                y_pos += aCC
        else:
            y_pos += (1.5 * aCC) if (prev == 'D' and cur == 'D') else aCC
        y_list[r] = y_pos

    if hbond_marks and (hbond_length is not None):
        hbond_row_pairs = sorted(set((r - 1, r + 1) for r, _ in hbond_marks), key=lambda x: x[0])
        for donor_row, acceptor_row in hbond_row_pairs:
            if donor_row < 0 or acceptor_row >= len(y_list):
                continue
            current_sep = abs(y_list[acceptor_row] - y_list[donor_row])
            if current_sep < 1e-6:
                continue
            extra = hbond_length - current_sep
            if extra > 0.0:
                for i in range(acceptor_row, len(lines)):
                    y_list[i] += extra

    def _add_atom(x, y, e, r, xi):
        i = len(pos)
        pos.append([x, y, 0.0])
        enames.append(e.upper())
        enames_original.append(e)
        rows.append(r)
        xidxs.append(xi)
        return i

    row_atoms = {}
    for r, line in enumerate(lines):
        y = -y_list.get(r, r * aCC)
        tokens = _atom_tokens(line)
        if not tokens:
            continue
        # mixed rows allowed: atom letters AND '|','-','.' marks on one line
        for c, ch in tokens:
            xi = c // 2
            x = c * dx / 2.0
            if ch.isalpha():
                i = _add_atom(x, y, ch, r, xi)
                row_atoms.setdefault(r, []).append(i)
            elif ch == '.':
                i = _add_atom(x, y, 'C', r, xi)
                row_atoms.setdefault(r, []).append(i)
            elif ch == '|':
                i1 = _add_atom(x, y - aCC / 2.0, 'C', r, xi)
                i2 = _add_atom(x, y + aCC / 2.0, 'C', r, xi)
                bonds.add((min(i1, i2), max(i1, i2)))
                row_atoms.setdefault(r, []).extend([i1, i2])
            elif ch == '-':
                i1 = _add_atom(x - aCC / 2.0, y, 'C', r, xi)
                i2 = _add_atom(x + aCC / 2.0, y, 'C', r, xi)
                bonds.add((min(i1, i2), max(i1, i2)))
                row_atoms.setdefault(r, []).extend([i1, i2])

    row_x = {}
    for i, (r, xi) in enumerate(zip(rows, xidxs)):
        row_x.setdefault(r, {}).setdefault(xi, []).append(i)

    for r in range(len(lines) - 1):
        rs = row_atoms.get(r, [])
        rt = row_atoms.get(r + 1, [])
        if (not rs) or (not rt):
            continue
        for i in rs:
            xi = xidxs[i]
            p = np.array(pos[i])
            for dx_i in (-1, 0, 1):
                for j in row_x.get(r + 1, {}).get(xi + dx_i, []):
                    if np.linalg.norm(p - np.array(pos[j])) < 1.2 * aCC:
                        bonds.add((min(i, j), max(i, j)))

    atoms = AtomicSystem(apos=np.array(pos), enames=enames)
    atoms.atypes = [_elements.ELEMENT_DICT[e][0] - 1 for e in enames]
    atoms.bonds = np.array(sorted(bonds), dtype=np.int32)
    atoms._enames_original = enames_original
    if hbond_marks:
        def _closest_in_row(r, c):
            xi = c // 2
            xr = row_x.get(r)
            if not xr:
                return None
            ks = list(xr.keys())
            k = min(ks, key=lambda kk: (abs(kk - xi), kk))
            return xr[k][0]

        hb_pairs = []
        for r, c in hbond_marks:
            ia = _closest_in_row(r - 1, c)
            ib = _closest_in_row(r + 1, c)
            if (ia is None) or (ib is None) or (ia == ib):
                continue
            hb_pairs.append((ia, ib))
        atoms._hbonds_pairs = list({tuple(sorted(p)) for p in hb_pairs})
    atoms._rows = rows                    # per-atom source row (drawn atoms only)
    atoms._hbond_rows = sorted({r for r, _ in hbond_marks})  # ':' rows delimit molecule blocks
    return atoms


# ---------------------------------------------------------------------------
# Single-atom format -> direct AtomicSystem
# ---------------------------------------------------------------------------
def _build_single(lines, aCC=A_CC, hbond_length=None):
    """Build an AtomicSystem from single-atom ASCII using honeycomb topology."""
    dx = np.sqrt(3.0) * aCC

    atoms = {}  # (r, c) -> element (preserves case)
    atoms_original = {}
    row_parity = {}  # r -> c % 2 of first atom in row
    bond_marks = []  # (r, c, ch) where ch in {'_', '/', '\\'}
    hbond_marks = []  # (r, c) where ':' denotes an H-bond between row r-1 and r+1
    for r, line in enumerate(lines):
        tokens = _atom_tokens(line)
        if not tokens:
            continue
        atom_tokens = [(c, ch) for c, ch in tokens if ch.isalpha()]
        if atom_tokens:
            row_parity[r] = atom_tokens[0][0] % 2
        for c, ch in tokens:
            if ch in ('_', '/', '\\'):
                bond_marks.append((r, c, ch))
                continue
            if ch == ':':
                hbond_marks.append((r, c))
                continue
            if not ch.isalpha():
                continue
            atoms[(r, c)] = ch.upper()
            atoms_original[(r, c)] = ch

    # Cumulative y: adjacent rows of the same parity are dimer-bonded (dy=A_CC),
    # adjacent rows of different parity are diagonal-bonded (dy=A_CC/2).
    y = [0.0]
    for r in range(1, len(lines)):
        same_parity = row_parity.get(r) == row_parity.get(r - 1)
        y.append(y[-1] - (aCC if same_parity else aCC / 2.0))

    # Adjust vertical spacing for H-bond rows so donor-acceptor distance matches target
    if hbond_marks and hbond_length is not None:
        hbond_row_pairs = sorted(set((r - 1, r + 1) for r, _ in hbond_marks), key=lambda x: x[0])
        for donor_row, acceptor_row in hbond_row_pairs:
            if donor_row < 0 or acceptor_row >= len(y):
                continue
            current_sep = y[donor_row] - y[acceptor_row]
            if current_sep < 1e-6:
                continue
            extra = hbond_length - current_sep
            if extra > 0.0:
                for i in range(acceptor_row, len(y)):
                    y[i] -= extra

    bonds = []
    for (r, c) in atoms:
        for nr, nc in [(r + 1, c), (r + 1, c - 1), (r + 1, c + 1)]:
            if (nr, nc) in atoms:
                bonds.append(tuple(sorted(((r, c), (nr, nc)))))

    row_atoms_cols = {}
    for (r, c) in atoms:
        row_atoms_cols.setdefault(r, []).append(c)
    for r in row_atoms_cols:
        row_atoms_cols[r].sort()

    def _nearest_left(r, c):
        cols = row_atoms_cols.get(r)
        if not cols:
            return None
        for cc in reversed(cols):
            if cc < c:
                return (r, cc)
        return None

    def _nearest_right(r, c):
        cols = row_atoms_cols.get(r)
        if not cols:
            return None
        for cc in cols:
            if cc > c:
                return (r, cc)
        return None

    def _closest_in_row(r, c):
        cols = row_atoms_cols.get(r)
        if not cols:
            return None
        cc = min(cols, key=lambda x: (abs(x - c), x))
        return (r, cc)

    for r, c, ch in bond_marks:
        if ch == '_':
            a = _nearest_left(r, c)
            b = _nearest_right(r, c)
            if (a is not None) and (b is not None):
                bonds.append(tuple(sorted((a, b))))
        elif ch == '/':
            # shortcut bond through an omitted atom at (r,c): (r-1,c) -- (r+1,c-1)
            a = _closest_in_row(r - 1, c)
            b = _closest_in_row(r + 1, c - 1)
            if (a is not None) and (b is not None):
                bonds.append(tuple(sorted((a, b))))
            else:
                # fallback: local diagonal mark
                a = _nearest_left(r, c)
                b = _nearest_left(r + 1, c)
                if (a is not None) and (b is not None):
                    bonds.append(tuple(sorted((a, b))))
        elif ch == '\\':
            # shortcut bond through an omitted atom at (r,c): (r-1,c) -- (r+1,c+1)
            a = _closest_in_row(r - 1, c)
            b = _closest_in_row(r + 1, c + 1)
            if (a is not None) and (b is not None):
                bonds.append(tuple(sorted((a, b))))
            else:
                # fallback: local diagonal mark
                a = _nearest_right(r, c)
                b = _nearest_right(r + 1, c)
                if (a is not None) and (b is not None):
                    bonds.append(tuple(sorted((a, b))))

    bonds = list(set(bonds))

    idx_map = {}
    pos = []
    enames = []
    rows = []
    for i, ((r, c), el) in enumerate(atoms.items()):
        idx_map[(r, c)] = i
        pos.append([c * dx / 2.0, y[r], 0.0])
        enames.append(el)
        rows.append(r)

    atoms = AtomicSystem(apos=np.array(pos), enames=enames)
    atoms.bonds = np.array([(idx_map[a], idx_map[b]) for a, b in bonds], dtype=np.int32)
    atoms.atypes = [_elements.ELEMENT_DICT[e][0] - 1 for e in enames]
    atoms._enames_original = [atoms_original[k] for k in atoms_original.keys()]
    atoms._rows = rows                                    # per-atom source row (drawn atoms only)
    atoms._hbond_rows = sorted({r for r, _ in hbond_marks})  # ':' rows delimit molecule blocks
    if hbond_marks:
        hb_pairs = []
        for r, c in hbond_marks:
            a = _closest_in_row(r - 1, c)
            b = _closest_in_row(r + 1, c)
            if (a is None) or (b is None):
                continue
            ia = idx_map.get(a)
            ib = idx_map.get(b)
            if (ia is None) or (ib is None) or (ia == ib):
                continue
            hb_pairs.append((ia, ib))
        atoms._hbonds_pairs = list({tuple(sorted(p)) for p in hb_pairs})
    return atoms


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def parse_ascii_art(text, hbond_length=None):
    """Return an AtomicSystem built from ASCII art."""
    lines = _lines(text)
    if any(_is_bond_row(line) for line in lines):
        return _build_dimer(lines, hbond_length=hbond_length)
    return _build_single(lines, hbond_length=hbond_length)


def mol_from_art(art, relax_bonds=True, n_iters=3, bmix=0.3):
    """ASCII art -> capped monomer AtomicSystem (no cell, no junctions).

    Same pipeline as tests/topology/testplot_muH.py::build_mol (parse ->
    neighs -> pi-count -> target-valence capping H) plus optional Jacobi
    bond-length relaxation of the heavy skeleton — run BEFORE capping so the
    L0=1.42 A target does not stretch the X-H caps.  Lowercase art atoms
    ('n','o') mark sp3/donor sites that receive the capping H.
    """
    atoms = parse_ascii_art(art)
    atoms.neighs()
    if relax_bonds:
        jacobi_relax_bond_lengths(atoms, n_iters=n_iters, bmix=bmix)
    n_pi0 = make_n_pi(atoms)
    tv = _build_target_valence(atoms, n_pi0)
    atoms.add_capping_h_sp2(target_valence=tv)
    eo = getattr(atoms, '_enames_original', None)
    if eo is not None and len(eo) < atoms.natoms:
        atoms._enames_original = list(eo) + ['H'] * (atoms.natoms - len(eo))
    atoms.neighs()
    return atoms


# ---------------------------------------------------------------------------
# ASCII equivalents of the built-in examples from heterocycle_generator.py.
# In dimer format atom symbols are only visual; the parser converts them to C.
# ---------------------------------------------------------------------------
ASCII_EXAMPLES = {
    'naphthalene': """
  C C
 | | |
  C C
""",
    'naphthalene2': """
  C C
 C C C
 C C C
  C C
""",
    'fulvalene': """
  C C
 | - |
  C C
""",
    'pentacross': """
  C C
 | . |
  C C
""",
    'biphenyl': """
  C 
 | | 
  |
 | | 
  C
""", 
  'biphenyl2': """
  C 
 C C
 C C 
  C
  C
 C C
 C C
  C 
""",
    'phenanthrene': """
  C 
 | | 
  | |
 | | 
  C 
""",
    'phenanthrene2': """
  C
 C C
 C C
  C C
  C C
 C C
 C C
  C
""",
    'perylene': """
  C C
 | | |
  | |
 | | |
  C C
""",
    'perylene2': """
  C C
 C C C
 C C C
  C C
  C C
 C C C
 C C C
  C C  
""",
    'purin': """
  C N
 N C C
 C C /
  N n
""",

    'purin_x': """
  C C
 N C N
  \\C C
  n N
""",
    'purin_y': """
  N n
 C C\\
 N C C
  C N
""",
    '7azaindol': """
  N n
 | | .
  C C
# """,
    'karbazol': """
  C n C
 C C C C
 C C_C C
  C   C
""",
    'biphenylene': """
  C   C
 C C_C C
 C C_C C
  C   C
""",
    'uracil': """
O n O
 C C
 n C
  C
""",
    'cytosin': """
O N n
 C C
 n C
  C
""",
    'guanin': """
n n O
 C C
 N C
  C n
   -
""",
    'NTCDA': """
O o O
 | |
| | |
 | |
O o O
""",

    'NTCDI': """
O n O
 | |
| | |
 | |
O n O
""",

    'TAP': """
  C
 N N 
 C C
| | |
 C C
 N N
  C 
""",
"Quinolone":"""
 O n
  C C
  C C
   C
""",
"2Quinolone":"""
 C
C C
C C
 n O
 : :
 O n
  C C
  C C
   C
""",
    '2purin': """
  C C
 C C C
 C C /
  N n
  : :
  n N
 / C C
 C C C
  C C
""",


    "Quinolinone":"""
  C C
 C C C
 C C C
  C n O
""",
    "2Quinolinone":"""
 C C
C C C
C C C
 C n O
   : :
   O n C
    C C C
    C C C
     C C
""",

    'NTCDI': """
O n O
 | |
| | |
 | |
O n O
""",
    '2NCI': """
 C C
| | | 
 | |
O n O
  : :
  O n O
   | |
  | | |
   C C
""",
}


# ---------------------------------------------------------------------------
# Edge-donor molecules for mol<->ribbon junction cells (build_mol_ribbon_cell).
# From doc/ERC_private/Ascci_Art_heterocycles.md — ACS Nano Fig.3 DD/DDD/DAD/
# DDA donors + classic 2H carriers.  Drawn state = corner 'A': lowercase
# junction tips ('n','o') carry the mol-side proton (donor), uppercase are bare
# acceptors.  mol_art_state() turns an end into 'B' = all tips uppercase (all
# junction protons on the ribbon).  Junction tips = N/O atoms of the extreme
# heavy-atom row at each end; several tips share one edge ('n n n' = 3-site
# donor edge — the art lattice spacing equals the ribbon site pitch).
# (hydroquinone = st1x1O, 1,4-dihydropyrazine = st1x1N are already covered by
# the parametric st<L>x<T> family — not duplicated here.)
MOL_EDGE_ARTS = {
    # --- classic 2H carriers (single tip each end -> bridge like st family) ---
    # para-NH2 tips; B end -> =NH (p-quinonediimine)
    'p_phenylenediamine': """
  n
  C
 C C
 C C
  C
  n
""",
    # -COOH per end: only the hydroxyl 'o' sits on the tip row (1 junction per
    # end); the carbonyl 'O' is drawn one row inside the molecule via '_'
    'terephthalic_acid': """
  o
  C_O
  C
 C C
 C C
  C
O_C
  o
""",
    # --- DD edges (Fig.3) ---
    'HH-h_1': """
n n
 C C
 C C
  n
""",
    # donors 2 site-pitches apart (skip 1 site)
    'HH-h_2': """
n C n
 C C
 C C
  n
""",
    # bare bottom edge -> single (boundary) junction
    'HH-p_1': """
n n
 C C
 C_C
""",
    # --- DDD edges ---
    'HHH-h': """
n n n
 C C
 C C
  n
""",
    'HHH-p': """
n n n
 C C
 C_C
""",
    # --- DAD edges (D-A-D triad: 2 donors + central ring-N acceptor) ---
    # 2,6-diaminopyridine motif; bare bottom CH apex
    'HNH-h': """
n N n
 C C
 C C
  C
""",
    # bottom 'n' is a 5-ring N-H -> bridges
    'HNH-p': """
n N n
 C C
 n_C
""",
    # --- DDA edges ---
    'HHO-h': """
n n O
 C C
 C C
  C
""",
    'HHO-p': """
n n O
 C C
 C_n
""",
    # fused 6+5; bottom '-' dimer edge is bare C-C
    'guanin': """
n n O
 C C
 N C
  C n
   -
""",
    # --- fused multi-ring donors (dimer format; mixed rows OK) ---
    # fused 6+6: 'n n' pyrrolic NH donors both ends (DD/DD bridge)
    'HH-hh': """
 n n
| | |
 n n
""",
    # fused 6+5 (pyrrole fused to the diazine): DD top edge, 'n c' bottom
    'HH-hp': """
 n n
| | c
 n c
""",
    # fused 5+5 (dipyrrrole): 'n n' donor edge; bare CH edge -> binds down only
    'HH-pp': """
 n n
C | C
 C C
""",
}


def mol_art_tip_cells(art):
    """(row,col) cells of the junction-tip atoms of a molecule art: the N/O
    atoms sitting in the extreme heavy-atom rows (within 0.15 A).
    Returns (bot_cells, top_cells), each sorted by column."""
    atoms = parse_ascii_art(art)
    dxh = np.sqrt(3.0) * A_CC / 2.0        # char column -> x
    y = atoms.apos[:, 1]
    iNO = [i for i, e in enumerate(atoms.enames) if e in ('N', 'O')]
    bot = sorted((atoms._rows[i], int(round(atoms.apos[i, 0] / dxh))) for i in iNO if y[i] < y.min() + 0.15)
    top = sorted((atoms._rows[i], int(round(atoms.apos[i, 0] / dxh))) for i in iNO if y[i] > y.max() - 0.15)
    return bot, top


def mol_art_state(art, etop='A', ebot='A'):
    """Corner-state variant of a MOL_EDGE_ARTS art.  'A' = end as drawn;
    'B' = every junction-tip heteroatom at that end uppercased (bare
    acceptor -> all junction protons moved to the ribbon)."""
    if etop == 'A' and ebot == 'A':
        return art
    bot, top = mol_art_tip_cells(art)
    lines = art.strip('\n').split('\n')
    for cells, st in ((top, etop), (bot, ebot)):
        if st == 'B':
            for r, c in cells:
                lines[r] = lines[r][:c] + lines[r][c].upper() + lines[r][c + 1:]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Periodic 1D H-bond chain cells
# ---------------------------------------------------------------------------
# Motif: molecules stacked along +y, separated by ':' junction rows. The LAST
# molecule in the art is the periodic image of the FIRST (same type) — the unit
# cell contains all molecules between the first and the last. Each ':' row gives
# one proton-transfer junction; junctions touching the last block partner atoms in
# the next cell (recorded via HbondRecord.d_shift/a_shift = +1).
#
# Corner states for the 2-junction cell: LL = both H on donors, RR = both
# transferred, RL/LR = mixed.  J = E_LL + E_RR - E_RL - E_LR < 0 cooperative.
# See doc/ERC_private/pbc_proton_transfer_chains.md.
PBC_CHAIN_ARTS = {
    # quinone/hydroquinone: O-H...O junctions; HQ donates to Q on both ends.
    'hq2q': """
  O
  C
 C C
 C C
  C
  O
  :
  o
  C
 C C
 C C
  C
  o
  :
  O
  C
 C C
 C C
  C
  O
""",
    # pyridine-substituted hq2q variants: the inner ortho C-H's clash across the
    # zigzag junctions (~0.4 A), so one clashing CH per junction -> pyridinic N.
    #   hq2qNa: N alternates side — junction-internal N on the Q acceptor,
    #           boundary N on the HQ donor (N's on DIFFERENT molecules).
    #   hq2qNb: both N's on the Q acceptor molecule (same molecule).
    # (left/right column of the ortho 'C C' rows = which side faces the junction)
    'hq2qNa': """
  O
  C
 C C
 N C
  C
  O
  :
  o
  C
 C C
 C N
  C
  o
  :
  O
  C
 C C
 N C
  C
  O
""",
    'hq2qNb': """
  O
  C
 C N
 N C
  C
  O
  :
  o
  C
 C C
 C C
  C
  o
  :
  O
  C
 C N
 N C
  C
  O
""",
    # pyrazine / 1,4-dihydropyrazine: N-H...N junctions (6-ring, N apex atoms).
    'pyr2hpyr': """
  N
 C C
 C C
  N
  :
  n
 C C
 C C
  n
  :
  N
 C C
 C C
  N
""",
    # mono-hydrogenated pyrazine homo-chain: self-complementary (N acceptor top,
    # n-H donor bottom) -> N-H...N junction at every interface.
    'pyrH2': """
  N
 C C
 C C
  n
  :
  N
 C C
 C C
  n
  :
  N
 C C
 C C
  n
""",
    # quinoxaline / dihydroquinoxaline (benzopyrazine): same as the QX/HQ chain
    # already built in dftbplus/rust_dftb/scripts/make_qxhq_chain.py.
    'qx2hqx': """
   N C
  | | |
   N C
   :
 C n
| | |
 C n
   :
   N C
  | | |
   N C
""",
    # phenazine / 5,10-dihydrophenazine: 3 fused rings, N at central-ring apexes.
    'phz2hphz': """
 C N C
C C C C
C C C C
 C N C
   :
 C n C
C C C C
C C C C
 C n C
   :
 C N C
C C C C
C C C C
 C N C
""",
    # 4-quinolone homo-chain: self-complementary (O acceptor top, n-H donor
    # bottom) -> N-H...O=C junction at EVERY interface; cell = 2 molecules so the
    # square has one internal and one boundary-crossing junction.
    'quinolone4': """
   O
 C C
C C C
C C C
 C n
   :
   O
 C C
C C C
C C C
 C n
   :
   O
 C C
C C C
C C C
 C n
""",
}


PBC_CELL_PARAMS = {  # per-system build defaults (overridable by explicit args)
    # hydroquinone/quinone: O-H...O is bent ~120 deg at the acceptor.  Default is
    # now the AXIAL geometry: molecules unrotated along y, blocks offset so every
    # junction leans slant deg off y.  slant=60 gives C-O-H ~115-125 deg at the
    # donor and C=O...H ~120 deg at the acceptor (the phenol-like kink; the
    # proton path is oblique along the true D...A direction).  All junctions
    # identical + sites related by half-cell translation -> E_LL = E_RR by
    # construction.  (Old alternative: zigzag=60, tilt=30, slant=0.)
    'hq2q': dict(tilt=0.0, zigzag=0.0, slant=60.0),
    # pyridine-substituted variants (clashing ortho C-H -> N): same axial slant
    # geometry; also clash-free under zigzag=60.
    'hq2qNa': dict(tilt=0.0, zigzag=0.0, slant=60.0),
    'hq2qNb': dict(tilt=0.0, zigzag=0.0, slant=60.0),
}
PBC_TILT_DEFAULT = 45.0   # herringbone tilt for all other chain systems


# ---------------------------------------------------------------------------
# Parametric acene-rhombus chains: ASCII art generated from (ncols, nrows, kind)
# ---------------------------------------------------------------------------
#
# Molecule = rhombus of ncols x nrows fused benzene rings on the honeycomb
# (col,k) lattice: pointy-top hexagon centers on q*(-1,3)+r*(+1,3), so all
# atoms sit at col==k (mod 2) and the patch has exactly one apex atom at the
# top and bottom corners -> the junction sites.  1x1 = benzene (quinone),
# 1x2 = naphthalene, 2x2 = pyrene, 3x3 = coronene-like rhombus.
#
# Junction terminations per block end (etop/ebot):
#   'O' = exocyclic =O on the apex C  (acceptor, C=O)
#   'o' = exocyclic -OH on the apex C (donor)
#   'N' = pyridinic N at the apex     (acceptor)
#   'n' = pyrrolic N-H at the apex    (donor)
#
# ACENE_KINDS maps a short name to the per-block (etop,ebot) specs cycled over
# the 3 drawn blocks (block0, block1, image-of-block0):
ACENE_KINDS = {
    'OO': [('O', 'O'), ('o', 'o')],   # alternating quinone/hydroquinone (hq2q-like): O-H...O=C
    'NN': [('N', 'n')],               # self-complementary N-H...N   (pyrH2-like)
    'Oo': [('O', 'o')],               # self-complementary O-H...O=C (hydroxy-quinone)
    'nO': [('n', 'O')],               # self-complementary N-H...O=C (quinolone-like)
    'No': [('N', 'o')],               # self-complementary O-H...N   (phenol-pyridine)
    'Nn': [('n', 'N')],               # reversed-orientation N-H...N
}


def _rhombus_patch(nrows=1, ncols=1):
    """(col,k)-grid atoms of an nrows x ncols rhombus of fused benzene rings.
    Returns ({(col,k): 'C'}, (col_top,k_top), (col_bot,k_bot))."""
    verts = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]
    atoms = {}
    for q in range(ncols):
        for r in range(nrows):
            cq, ck = r - q, 3 * (q + r)
            for dc, dk in verts:
                atoms[(cq + dc, ck + dk)] = 'C'
    kt = max(k for _, k in atoms); kb = min(k for _, k in atoms)
    tops = [c for c, k in atoms if k == kt]; bots = [c for c, k in atoms if k == kb]
    assert len(tops) == 1 and len(bots) == 1, f"rhombus {nrows}x{ncols} has no unique apex atoms"
    return atoms, (tops[0], kt), (bots[0], kb)


def _acene_block(nrows, ncols, etop='O', ebot='O'):
    """One molecule block {(col,k): char} with junction terminations at the apexes."""
    patch, (ct, kt), (cb, kb) = _rhombus_patch(nrows, ncols)
    blk = dict(patch)
    for c, k, e, sgn in [(ct, kt, etop, +2), (cb, kb, ebot, -2)]:
        if e in 'Oo':
            blk[(c, k + sgn)] = e            # exocyclic =O / -OH one row past the apex C
        else:
            blk[(c, k)] = e                  # pyridinic N / N-H at the apex atom
    return blk


def make_acene_chain_art(nrows=1, ncols=1, kind='OO'):
    """3-block stack art (block0, block1, image-of-block0) for build_pbc_cell.

    Rasterizes the (col,k) lattice: every occupied k-level -> one text line,
    atoms at character columns; ':' junction rows at the apex column between
    blocks.  Adjacent lines carry the correct parity so the single-atom parser
    reproduces the honeycomb y-spacing (aCC/2 diagonal, aCC vertical edges).
    """
    specs = ACENE_KINDS[kind]
    blocks = [_acene_block(nrows, ncols, *specs[i % len(specs)]) for i in range(3)]
    c0 = 1 - min(c for blk in blocks for c, _ in blk)          # global left margin
    lines = []
    for ib, blk in enumerate(blocks):
        if ib:
            cj = [c for (c, k) in blk if k == max(k2 for _, k2 in blk)][0] + c0
            lines.append(' ' * cj + ':')
        for k in sorted({k for _, k in blk}, reverse=True):
            row = {c: ch for (c, kk), ch in blk.items() if kk == k}
            lines.append(''.join(row.get(c - c0, ' ') for c in range(0, max(row) + c0 + 1)))
    return '\n'.join(lines)


# STRIP family (aligned chains, laterally thickened — armchair ribbons along y):
#   T=1 'link'    : phenylene spine - | | edge-rows alternate with single | link
#                   rows (apex-to-apex C-C bonds):  L=1 benzene, L=3 biphenyl,
#                   L=5 p-terphenyl.  Needs odd L; ends aligned on the spine.
#   T even 'zigzag': constant T/2 fused hexagons per row, consecutive rows offset
#                   by +-1 column (zigzag fusion) — inherently asymmetric, ends
#                   carry T/2 apex atoms.  T=2: phenanthrene-like; T=4: 2-wide.
#   T odd >=3 'symm': centered rows alternating (T-1)/2 and (T+1)/2 hexagons —
#                   mirror-symmetric about the spine axis (PTCDA-like), ends
#                   carry (T-1)/2 apexes.  T=3: pyrene / perylene-homolog strips.
def _strip_marks(nrows, ncols):
    """'|' mark columns per dimer row + apex atom columns at both ends.
    Returns (marks, apex_top, apex_bot).  nrows must be odd so both ends carry
    the same apex set (junction interfaces then pair apex-to-apex)."""
    assert nrows % 2 == 1, f"strip needs odd nrows (apexes must match at both ends), got {nrows}"
    if ncols <= 1:
        return [[-1, 1] if i % 2 == 0 else [0] for i in range(nrows)], [0], [0]
    if ncols % 2 == 0:                                   # even T: constant zigzag
        h = ncols // 2
        s, hexes = 1 - h, []
        for i in range(nrows):
            if i:
                s = (3 - 2 * h) - s                      # zigzag: offsets alternate +-1
            hexes.append(list(range(s, s + 2 * h, 2)))
    else:                                                # odd T: centered alternating
        m = (ncols - 1) // 2
        hexes = [list(range(1 - hh, hh + 1, 2)) for hh in (m + i % 2 for i in range(nrows))]
    marks = [sorted({c for o in row for c in (o - 1, o + 1)}) for row in hexes]
    return marks, hexes[0], hexes[-1]


def _strip_block_lines(marks, apex_top, apex_bot, etop, ebot, c0):
    """Text lines for one strip block: junction atoms at each apex col, '|' marks."""
    w = c0 + max(c for row in marks for c in row) + 1
    def atom_row(cols, ch):
        line = [' '] * w
        for c in cols:
            line[c + c0] = ch
        return ''.join(line).rstrip()
    lines = []
    if etop in 'Oo':
        lines += [atom_row(apex_top, etop), atom_row(apex_top, 'C')]   # exocyclic =O/-OH
    else:
        lines += [atom_row(apex_top, etop)]                            # N / N-H at apex
    for row in marks:
        line = [' '] * w
        for c in row:
            line[c + c0] = '|'
        lines.append(''.join(line).rstrip())
    if ebot in 'Oo':
        lines += [atom_row(apex_bot, 'C'), atom_row(apex_bot, ebot)]
    else:
        lines += [atom_row(apex_bot, ebot)]
    return lines


def make_strip_chain_art(nrows=1, ncols=1, kind='OO'):
    """3-block stack art for build_pbc_cell: aligned spine strip.
    nrows = dimer rows along the spine ('ring-rows'), ncols = thickness T:
    1 -> phenylene link chain, even -> constant zigzag strip (T/2 apexes),
    odd >=3 -> mirror-symmetric alternating strip ((T-1)/2 apexes)."""
    specs = ACENE_KINDS[kind]
    marks, atop, abot = _strip_marks(nrows, ncols)
    c0 = 1 - min(c for row in marks for c in row)      # spine column -> char index
    assert atop == abot, f"junction apexes differ top/bot ({atop} vs {abot})"
    w = c0 + max(c for row in marks for c in row) + 1
    jline = [' '] * w
    for c in atop:
        jline[c + c0] = ':'                            # one ':' per junction apex pair
    jline = ''.join(jline).rstrip()
    lines = []
    for ib in range(3):
        if ib:
            lines.append(jline)
        lines += _strip_block_lines(marks, atop, abot, *specs[ib % len(specs)], c0)
    return '\n'.join(lines)


def make_strip_mol_art(nrows=1, ncols=1, etop='N', ebot='N'):
    """Single strip-block monomer art (no ':' junction rows) for mol_from_art.

    Same family as make_strip_chain_art: nrows = ring-rows along the spine
    (must be odd), ncols = thickness T (1 = link chain, odd >=3 =
    mirror-symmetric strip).  etop/ebot in {'O','o','N','n'} set the tip
    terminations independently ('o'/'n' = donor, 'O'/'N' = acceptor), so one
    call yields the AA/BB/AB/BA corner-state arts of a tip-ended molecule.
    """
    marks, atop, abot = _strip_marks(nrows, ncols)
    assert atop == abot, f"strip {nrows}x{ncols} tip columns differ top/bot ({atop} vs {abot})"
    c0 = 1 - min(c for row in marks for c in row)
    return '\n' + '\n'.join(_strip_block_lines(marks, atop, abot, etop, ebot, c0)) + '\n'


def get_pbc_chain_art(name):
    """PBC_CHAIN_ARTS lookup; falls back to generated 'ac<ncols>x<nrows><kind>' rhombus
    or 'st<nrows>x<ncols><kind>' spine-strip chains."""
    import re
    if name in PBC_CHAIN_ARTS:
        return PBC_CHAIN_ARTS[name]
    m = re.fullmatch(r'ac(\d+)x(\d+)([A-Za-z]+)', name)
    if m and m.group(3) in ACENE_KINDS:
        return make_acene_chain_art(int(m.group(2)), int(m.group(1)), m.group(3))
    m = re.fullmatch(r'st(\d+)x(\d+)([A-Za-z]+)', name)
    if m and m.group(3) in ACENE_KINDS:
        return make_strip_chain_art(int(m.group(1)), int(m.group(2)), m.group(3))
    raise KeyError(f"unknown PBC chain '{name}' (not in PBC_CHAIN_ARTS, ac<ncols>x<nrows><kind>, or st<nrows>x<ncols><kind>)")


def _pbc_defaults(name):
    """PBC_CELL_PARAMS lookup; generated acene names get per-junction-kind defaults."""
    if name in PBC_CELL_PARAMS:
        return PBC_CELL_PARAMS[name]
    if name.startswith('st') and 'x' in name:
        # multi-apex ends (T>=4 -> >=2 parallel junction pairs per interface):
        if 'O' in name or 'o' in name:
            return dict(tilt=0.0, zigzag=0.0, slant=60.0)   # bent X-H...O kink like hq2q
        return dict(tilt=0.0, zigzag=0.0, slant=0.0)
    if name.startswith('ac') and ('O' in name or 'o' in name):
        return dict(tilt=0.0, zigzag=0.0, slant=60.0)   # bent X-H...O kink like hq2q
    return {}


def build_pbc_cell(name=None, art=None, hbond_length=2.8, vac_x=10.0, vac_z=10.0, relax_bonds=True, tilt=None, zigzag=None, slant=None, jkink=None):
    """Build a 1D-periodic H-bond chain unit cell from a PBC_CHAIN_ARTS stack.

    The art is a stack of molecule blocks separated by ':' junction rows; the
    last block is the periodic image of the first and is NOT included in the
    cell. Junctions between two in-cell blocks are internal; the junction to the
    last block becomes a boundary junction whose partner gets a cell shift of +1
    (HbondRecord.d_shift/a_shift).

    Geometry assembly (after the flat stack is built):
      1. zigzag : rotate molecule block k by (-1)^k * zigzag deg in-plane (about
         z through the block centroid) — for bent X-H...Y bonds (~120 deg at the
         acceptor, e.g. O-H...O in hq2q) so the junction can kink.
      2. junction realignment : translate each block so every junction D...A
         pair sits exactly hbond_length apart on a line parallel to y
         (straightens staggered arts; no-op for symmetric ones).
      3. the cell is rotated so the lattice vector is pure +y.
      4. tilt : herringbone rotation of block k by (-1)^k * tilt deg about the
         axis through the block's two junction heavy atoms — junction D...A is
         preserved EXACTLY (atoms on the axis don't move); +/-45 makes
         consecutive molecular planes ~perpendicular (same trick as
         make_qxhq_chain.py), relieving H...H steric clash across junctions.

    Args:
        tilt   : herringbone tilt [deg]; None -> PBC_CELL_PARAMS default (45).
        zigzag : in-plane zigzag angle [deg]; None -> PBC_CELL_PARAMS default (0).

    Returns:
        atoms  : AtomicSystem for one cell (capping H included, junction H's on donors)
        lvs    : (3,3) lattice vectors [A], chain along y, vacuum in x/z
        hbonds : list[HbondRecord] with d_shift/a_shift marking image partners
    """
    defaults = _pbc_defaults(name)
    if tilt is None:
        tilt = defaults.get('tilt', PBC_TILT_DEFAULT)
    if zigzag is None:
        zigzag = defaults.get('zigzag', 0.0)
    if slant is None:
        slant = defaults.get('slant', 0.0)
    if jkink is None:
        jkink = defaults.get('jkink', 0.0)
    if art is None:
        art = get_pbc_chain_art(name)
    atoms = parse_ascii_art(art, hbond_length=hbond_length)
    atoms.neighs()
    n_pi0 = make_n_pi(atoms)
    tv = _build_target_valence(atoms, n_pi0)
    atoms.add_capping_h_sp2(target_valence=tv)
    atoms.neighs()
    if relax_bonds:
        jacobi_relax_bond_lengths(atoms, n_iters=3, bmix=0.3)
    resolve_hbond_pairs(atoms)

    rows = atoms._rows
    hb_rows = atoms._hbond_rows
    ndrawn = len(rows)
    if len(hb_rows) < 2:
        raise ValueError("PBC cell art needs >= 2 ':' junction rows (last block is the periodic image)")

    def _block_of(i):
        if i < ndrawn:
            return sum(1 for cr in hb_rows if rows[i] > cr)
        # capping H: same block as its heavy neighbour
        for j in atoms.ngs[i]:
            if atoms.enames[j] != 'H':
                return _block_of(j)
        raise ValueError(f"atom {i} has no heavy neighbour")

    nblocks = len(hb_rows) + 1
    blocks = [[] for _ in range(nblocks)]
    for i in range(atoms.natoms):
        blocks[_block_of(i)].append(i)
    if zigzag or tilt:
        assert (nblocks - 1) % 2 == 0, "zigzag/tilt need an even number of in-cell blocks (image parity)"

    # junction endpoint heavy atoms per ':' row, grouped by interface:
    # jints[j] = [(upper_atom, lower_atom), ...] connecting blocks j and j+1.
    # Multi-apex junctions (e.g. T>=4 strips) give >1 pair per interface —
    # alignment/straighten then work on the interface MEAN position.
    jints = [[] for _ in range(nblocks - 1)]
    for ih, ia in atoms.hbonds_ascii:
        ngh = [jj for jj in atoms.ngs[ih] if atoms.enames[jj] != 'H']
        assert len(ngh) == 1, f"junction H {ih} has {len(ngh)} heavy neighbours"
        idon = ngh[0]
        bd, ba = _block_of(idon), _block_of(ia)
        assert abs(bd - ba) == 1, f"junction must connect adjacent blocks (got {bd},{ba})"
        jints[min(bd, ba)].append((idon, ia) if bd < ba else (ia, idon))
    assert all(jints), "every ':' interface must yield >=1 H-bond pair"

    def _jpos(k):
        """(top, bottom) junction-site positions of block k in its own frame —
        mean over the interface's apex atoms; image block via the img map."""
        pa = (np.mean([atoms.apos[l] for _, l in jints[k - 1]], axis=0) if k >= 1
              else np.mean([atoms.apos[img[l]] for _, l in jints[-1]], axis=0))
        pb = (np.mean([atoms.apos[u] for u, _ in jints[k]], axis=0) if k < len(jints)
              else np.mean([atoms.apos[inv_img[u]] for u, _ in jints[0]], axis=0))
        return pa, pb

    # (1) zigzag: rotate block k in-plane by (-1)^k * zigzag about z through its centroid
    if zigzag:
        for k, blk in enumerate(blocks):
            idx = np.asarray(blk)
            ctr = atoms.apos[idx].mean(axis=0)
            th = np.radians((1.0 if k % 2 == 0 else -1.0) * zigzag)
            c, s = np.cos(th), np.sin(th)
            d = atoms.apos[idx] - ctr
            atoms.apos[idx, 0] = ctr[0] + d[:, 0] * c - d[:, 1] * s
            atoms.apos[idx, 1] = ctr[1] + d[:, 0] * s + d[:, 1] * c

    # last block = periodic image of first: match atoms by (row, col) order
    first, last = blocks[0], blocks[-1]
    drawn_first = [i for i in first if i < ndrawn]
    drawn_last = [i for i in last if i < ndrawn]
    assert len(drawn_first) == len(drawn_last), f"image block mismatch: {len(drawn_first)} vs {len(drawn_last)} atoms"
    # match image atoms to first-block atoms by (art row, rounded x, y):
    # the bond relax leaves bit-level asymmetry in x (~1e-15 A) which would
    # scramble the two same-column atoms of a '|' pair differently per block;
    # rounding x and sorting by y as tiebreak makes the match deterministic.
    drawn_first.sort(key=lambda i: (rows[i], round(atoms.apos[i, 0], 4), atoms.apos[i, 1]))
    drawn_last.sort(key=lambda i: (rows[i], round(atoms.apos[i, 0], 4), atoms.apos[i, 1]))
    for i, j in zip(drawn_first, drawn_last):
        assert atoms.enames[i] == atoms.enames[j], f"image block ename mismatch at {i}/{j}"
    img = {j: i for i, j in zip(drawn_first, drawn_last)}  # last-block atom -> first-block atom

    # (2a) straighten each block — rotate rigidly about its centroid so the axis
    #      through its two junction heavy atoms is parallel to y (the bond relax
    #      can skew junction atoms off-axis, and generated diagonal acene arts have
    #      genuinely tilted apex axes; straight molecules make the lattice pure-y;
    #      skipped under zigzag where junction axes are intentionally rotated).
    #      The image block gets the same treatment via the inverse img map so it
    #      stays a pure translation.
    if not zigzag:
        inv_img = {i: j for j, i in img.items()}
        for k, blk in enumerate(blocks):
            pa, pb = _jpos(k)
            axv = pb - pa
            axv /= np.linalg.norm(axv)
            uy = np.array([0.0, np.sign(axv[1]), 0.0])          # keep up/down order
            rot = np.cross(axv, uy)
            th = np.arcsin(np.clip(np.linalg.norm(rot), -1.0, 1.0))
            if th < 1e-9:
                continue
            rot = rot / np.linalg.norm(rot) * th
            c, s = np.cos(th), np.sin(th)
            u = rot / th
            idx = np.asarray(blk)
            ctr = atoms.apos[idx].mean(axis=0)
            d = atoms.apos[idx] - ctr
            atoms.apos[idx] = ctr + d * c + np.cross(u, d) * s + u * (d @ u)[:, None] * (1.0 - c)

    # (2b) junction kink: rotate each DONOR-side exocyclic junction atom ('o','O')
    #      with its attached H's outward about its apex carbon by jkink deg.
    #      The splayed X-H arms let each parallel multi-apex junction lean
    #      (C-O-H ~120 deg) without the donor landing on the NEIGHBOUR acceptor
    #      — the reason block slant is capped ~30 deg for T>=4.  Symmetric splay
    #      keeps the interface mean on-axis; image counterparts get identical
    #      splay via the img map so the image block stays a pure translation.
    if jkink:
        inv_img = {i: j for j, i in img.items()}
        ops = {}
        def _splay_end(end_atoms):
            cx = np.mean([atoms.apos[a][0] for a in end_atoms])
            for ja in end_atoms:
                hs = [jj for jj in atoms.ngs[ja] if atoms.enames[jj] == 'H']
                if not hs:
                    continue                              # acceptor: no H -> skip
                heavy = [jj for jj in atoms.ngs[ja] if atoms.enames[jj] != 'H']
                if len(heavy) != 1:
                    continue                              # in-ring junction atom ('n','N')
                cap = heavy[0]
                om = np.radians(jkink) * (-np.sign(atoms.apos[ja][0] - cx) * np.sign(atoms.apos[ja][1] - atoms.apos[cap][1]))
                if om == 0.0:
                    continue                              # lone apex: no outward direction
                if ja not in ops:
                    ops[ja] = (hs, cap, om)
        for pairs in jints:
            _splay_end([u for u, _ in pairs])
            _splay_end([l for _, l in pairs])
        for ja, (hs, cap, om) in list(ops.items()):       # image counterparts: identical splay
            for m in (img.get(ja), inv_img.get(ja)):
                if m is not None and m not in ops:
                    mh = [jj for jj in atoms.ngs[m] if atoms.enames[jj] == 'H']
                    mc = [jj for jj in atoms.ngs[m] if atoms.enames[jj] != 'H'][0]
                    ops[m] = (mh, mc, om)
        for ja, (hs, cap, om) in ops.items():
            c, s = np.cos(om), np.sin(om)
            for a in [ja] + hs:
                v = atoms.apos[a] - atoms.apos[cap]
                atoms.apos[a] = atoms.apos[cap] + np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]])

    # (2) realign junctions: translate block j+1 so its junction atom sits exactly
    #     hbond_length below block j's junction atom, on a line leaning by slant
    #     deg off y (alternating sign per junction -> offsets cancel, lattice stays
    #     pure y, molecules stay axial; slant mimics the ~120 deg C-O-H kink so the
    #     H-bond/proton path is oblique).  slant=0 -> junctions vertical.
    sa = np.radians(slant)
    for j, pairs in enumerate(jints):
        pu = np.mean([atoms.apos[u] for u, _ in pairs], axis=0)
        pl = np.mean([atoms.apos[l] for _, l in pairs], axis=0)
        sgn = 1.0 if j % 2 == 0 else -1.0
        dx_t = sgn * np.sin(sa) * hbond_length
        dy_t = -np.cos(sa) * hbond_length
        dy = pl[1] - pu[1]
        shift = np.array([pu[0] + dx_t - pl[0], dy_t - dy, 0.0])
        atoms.apos[np.asarray(blocks[j + 1])] += shift

    t = np.median(np.array([atoms.apos[j] - atoms.apos[i] for i, j in zip(drawn_first, drawn_last)]), axis=0)
    resid = max(np.linalg.norm(atoms.apos[j] - atoms.apos[i] - t) for i, j in zip(drawn_first, drawn_last))
    assert resid < 0.05, f"image block not a pure translation of first (max resid {resid:.3f} A)"

    # cell contents: all blocks except the last
    cell_atoms = [i for b in blocks[:-1] for i in b]
    remap = {old: new for new, old in enumerate(cell_atoms)}
    enames_c = [atoms.enames[i] for i in cell_atoms]
    apos_c = np.array([atoms.apos[i] for i in cell_atoms], dtype=float)
    bonds_c = [(remap[i], remap[j]) for i, j in atoms.bonds if i in remap and j in remap]

    # (3) rotate cell so the lattice vector is pure +y (diagonal molecules give t
    #     an x-component; rotation by +phi aligns t with +y)
    if t[1] < 0:
        apos_c[:, 1] *= -1.0
        t = t * np.array([1.0, -1.0, 1.0])
    phi = np.arctan2(t[0], t[1])
    if abs(phi) > 1e-9:
        c, s = np.cos(phi), np.sin(phi)
        xy = apos_c[:, :2].copy()
        apos_c[:, 0] = xy[:, 0] * c - xy[:, 1] * s
        apos_c[:, 1] = xy[:, 0] * s + xy[:, 1] * c
        t = np.array([0.0, np.linalg.norm(t), 0.0])
    Ly = t[1]
    apos_c[:, 1] -= apos_c[:, 1].min() - 0.5 * hbond_length   # margin so boundary junction sits inside plot range

    # (4) herringbone tilt: rotate block k by (-1)^k * tilt about the axis through
    #     the block's two junction heavy atoms -> junction D...A preserved exactly
    if tilt:
        for k, blk in enumerate(blocks[:-1]):
            itops = [l for _, l in jints[k - 1]] if k >= 1 else [img[l] for _, l in jints[-1]]
            ibots = [u for u, _ in jints[k]]
            pa = np.mean([apos_c[remap[i]] for i in itops], axis=0)
            pb = np.mean([apos_c[remap[i]] for i in ibots], axis=0)
            axv = pb - pa
            axv /= np.linalg.norm(axv)
            p0 = 0.5 * (pa + pb)
            th = np.radians((1.0 if k % 2 == 0 else -1.0) * tilt)
            c, s = np.cos(th), np.sin(th)
            idx = np.array([remap[i] for i in blk])
            d = apos_c[idx] - p0
            apos_c[idx] = p0 + d * c + np.cross(axv, d) * s + axv * (d @ axv)[:, None] * (1.0 - c)

    lvs = np.array([[apos_c[:, 0].ptp() + vac_x, 0.0, 0.0], [0.0, Ly, 0.0], [0.0, 0.0, apos_c[:, 2].ptp() + vac_z]])
    lvec = np.array([0.0, Ly, 0.0])

    # junctions: hbonds_ascii = (h_idx, acc_idx); donor = heavy neighbour of h;
    # a partner in the last (image) block is remapped to its first-block atom with shift +1.
    # The junction H is placed on the D->A axis at r_xh (the corner scan repositions it anyway).
    from spammm.topology.hbond_utils import HbondRecord
    r_xh = 1.01
    hbonds = []
    for ih, ia in atoms.hbonds_ascii:
        ngh = [j for j in atoms.ngs[ih] if atoms.enames[j] != 'H']
        idon = ngh[0]
        d_sh = +1 if _block_of(idon) == nblocks - 1 else 0
        a_sh = +1 if _block_of(ia) == nblocks - 1 else 0
        id_c = remap[img.get(idon, idon)]
        ia_c = remap[img.get(ia, ia)]
        if ih in remap:
            ih_c = remap[ih]
        else:
            # junction H on an image-block donor (donor-oriented-down kinds):
            # reuse the corresponding H on the in-cell donor atom (position is
            # recomputed on the D->A axis below anyway)
            hs = [j for j in atoms.ngs[img.get(idon, idon)] if atoms.enames[j] == 'H' and j in remap]
            assert hs, f"in-cell donor {img.get(idon, idon)} has no H for image junction"
            ih_c = remap[hs[0]]
        pD, pA = apos_c[id_c] + d_sh * lvec, apos_c[ia_c] + a_sh * lvec
        axv = pA - pD
        dist = float(np.linalg.norm(axv))
        apos_c[ih_c] = pD + r_xh * axv / dist
        hbonds.append(HbondRecord(id_c, ih_c, ia_c, dist, 180.0, d_shift=d_sh, a_shift=a_sh))

    cell = AtomicSystem(apos=apos_c, enames=enames_c)
    cell.atypes = [_elements.ELEMENT_DICT[e][0] - 1 for e in enames_c]
    cell.bonds = np.array(bonds_c, dtype=np.int32)
    cell._pbc_img = img
    cell._pbc_remap = remap

    # steric-clash check on the finished cell (incl. boundary image pairs);
    # junction D..A/H..A pairs are close by design -> excluded
    from spammm.atomicUtils import check_clashes
    jexc = [p for hb in hbonds for p in ((hb.donor_idx, hb.h_idx), (hb.h_idx, hb.acceptor_idx), (hb.donor_idx, hb.acceptor_idx))]
    check_clashes(cell.apos, cell.enames, cell.bonds, lvec=lvec, exclude=jexc, label=name or 'pbc_cell')
    return cell, lvs, hbonds
def main():
    parser = argparse.ArgumentParser(description='Generate heterocycle geometry from ASCII art')
    parser.add_argument('--out', '-o', default='/tmp/kekule/heterocycle.svg', help='Output SVG file')
    parser.add_argument('--xyz', default=None, help='Optional output XYZ file')
    parser.add_argument('--example', '-e', default='naphthalene', choices=list(ASCII_EXAMPLES.keys()),  help='Built-in ASCII example to use')
    parser.add_argument('--title', '-t', default=None, help='Plot title')
    parser.add_argument('--size', type=float, default=120, help='Atom marker size')
    parser.add_argument('--aCC', type=float, default=A_CC, help='C-C bond length [Angstrom]')
    parser.add_argument('--plot', type=int, default=1, help='Show matplotlib figure (1) or suppress it (0)')
    parser.add_argument('--dump_bonds', type=int, default=0, help='Print explicit bond list (1) or not (0)')
    parser.add_argument('--kekule', type=int, default=1, help='Run Kekule pi-bond optimization and draw bond orders (1) or not (0)')
    parser.add_argument('--kekule_single', type=int, default=0, help='When --kekule 1, draw a single-panel system plot instead of two-phase raw+snapped')
    parser.add_argument('--relax_bonds', type=int, default=0, help='Number of Jacobi bond-length relaxation steps (0=off)')
    parser.add_argument('--relax_bmix', type=float, default=0.5, help='Momentum mixing factor from 2nd relaxation step onward')
    parser.add_argument('--hydrogens', type=int, default=1, help='Add H passivation based on topology (1) or keep heavy-atom skeleton only (0)')
    parser.add_argument('--hbond_length', type=float, default=3.0, help='Target heavy-atom donor-acceptor distance for H-bonds in Angstrom')
    parser.add_argument('--mol', default='auto', help='Save MOL file ("auto" -> same as --out but .mol; "off" -> disable; or provide path)')
    parser.add_argument('--sym_break', type=float, default=0.0, help='Symmetry-breaking noise added to bond orders before phase-2 localization (0=off)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed used for --sym_break (0 means do not set seed)')
    parser.add_argument('--Kval', type=float, default=50.0, help='Atom-sum stiffness (K_atom_sum)')
    parser.add_argument('--Kloc', type=float, default=5.0, help='Snap/localization stiffness (K_snap)')
    parser.add_argument('--Karo', type=float, default=0.5, help='Aromatization stiffness (K_arom)')
    parser.add_argument('--aromatic', type=int, default=1, help='Allow aromatic (0.5) bond orders (1) or force integer (0/1) localization (0)')
    parser.add_argument('--solver', default='linsolve', choices=['linsolve', 'gd'], help='Solver: linsolve (recommended, quadratic) or gradient descent (fallback)')
    parser.add_argument('--kkt', type=int, default=0, help='Save KKT matrix plot (1) or not (0)')
    parser.add_argument('--kkt_mode', default='signed', choices=['signed', 'logabs'], help='KKT plot mode')
    parser.add_argument('--kkt_cmap', default=None, help='KKT colormap (signed: seismic; logabs: magma/inferno)')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    art = ASCII_EXAMPLES[args.example]
    atoms = parse_ascii_art(art, hbond_length=args.hbond_length)
    atoms.neighs()

    n_pi0 = make_n_pi(atoms)

    if args.hydrogens:
        tv = _build_target_valence(atoms, n_pi0)
        atoms.add_capping_h_sp2(target_valence=tv)
        enames_original = getattr(atoms, '_enames_original', None)
        if (enames_original is not None) and (len(enames_original) < len(atoms.apos)):
            atoms._enames_original = list(enames_original) + ['H'] * (len(atoms.apos) - len(enames_original))
        atoms.neighs()

    resolve_hbond_pairs(atoms)

    if args.relax_bonds:
        jacobi_relax_bond_lengths(atoms, L0=args.aCC, n_iters=args.relax_bonds, bmix=args.relax_bmix)

    lengths = [np.linalg.norm(atoms.apos[i] - atoms.apos[j]) for i, j in atoms.bonds]
    uniq = sorted({round(d, 3) for d in lengths})
    print(f"{args.example:12s}: atoms={atoms.natoms:2d} bonds={len(atoms.bonds):2d} lengths={uniq}")

    if args.dump_bonds:
        enames_original = getattr(atoms, '_enames_original', None)
        if (enames_original is not None) and (len(enames_original) != len(atoms.apos)):
            enames_original = None
        print("ATOMS:")
        for i, p in enumerate(atoms.apos):
            e = atoms.enames[i]
            if enames_original is not None:
                e = enames_original[i]
            print(f"  {i+1:2d} {e:2s}  ({p[0]:7.3f},{p[1]:7.3f},{p[2]:7.3f})")
        print("BONDS (1-indexed):")
        for i, j in atoms.bonds:
            print(f"  {i+1:2d} - {j+1:2d}")
        if atoms.natoms >= 9:
            i, j = 5, 9
            has = (min(i-1, j-1), max(i-1, j-1)) in set((min(a,b), max(a,b)) for a, b in atoms.bonds)
            print(f"CHECK bond {i}-{j}: {has}")

    if args.kekule:
        if args.kkt:
            # KKT plot needs a solver instance; create one briefly for the matrix
            n_pi_kkt = make_n_pi(atoms)
            bonds_all_kkt = np.asarray(atoms.bonds, dtype=np.int32) if (atoms.bonds is not None) else np.zeros((0, 2), dtype=np.int32)
            is_heavy_kkt = np.array([e not in ('H', 'E') for e in atoms.enames], dtype=bool)
            heavy_mask_kkt = is_heavy_kkt[bonds_all_kkt[:, 0]] & is_heavy_kkt[bonds_all_kkt[:, 1]] if len(bonds_all_kkt) else np.zeros(0, dtype=bool)
            k_kkt = KekulePure(atoms, n_pi=n_pi_kkt, bonds=bonds_all_kkt[heavy_mask_kkt], Kval=args.Kval, Kloc=0.0, Karo=args.Karo, Kbound=1.0,   allow_aromatic=(args.aromatic != 0))
            if args.kkt_cmap is None:
                kkt_cmap = 'seismic' if args.kkt_mode == 'signed' else 'magma'
            else:
                kkt_cmap = args.kkt_cmap
            root, ext = os.path.splitext(os.path.abspath(args.out))
            kkt_out = root + f"_kkt_{args.kkt_mode}.png"
            k_kkt.plot_kkt_matrix(mode=args.kkt_mode, cmap=kkt_cmap, fname=kkt_out, show=False)
            print(f"Saved KKT matrix: {kkt_out}")

        r = run_kekule_solver(atoms, Kval=args.Kval, Kloc=args.Kloc, Karo=args.Karo, allow_aromatic=(args.aromatic != 0), solver=args.solver, sym_break=args.sym_break, seed=args.seed)
        bo_raw, bo_snap, n_pi, k, err = r['bo_raw'], r['bo_snap'], r['n_pi'], r['k'], r['err']
        rep = r['report']
        if err is None:
            print(f"  phase 1 F2={rep['phase1_F2']:.3e}  pi_BO={np.round(rep['phase1_bo'],2)}")
            print(f"  n_pi={np.asarray(rep['n_pi'], dtype=int)}")
            print(f"  phase 2 F2={rep['phase2_F2']:.3e}  pi_BO={np.round(rep['phase2_bo'],2)}")
            print(f"  single={rep['single']:2d} aromatic={rep['aromatic']:2d} double={rep['double']:2d}")
            print(f"  phase 2 constraint max|A@bo-n_pi|={rep['max_err']:.3e}")
        else:
            print(f"ERROR: {err}")
        title = args.title
        if err is not None:
            title = (title + '\n' if title else '') + f"ERROR: {err}"
        if args.kekule_single:
            bo_total = 1.0 + bo_snap
            plot_system(atoms, title=title, fname=args.out, show=args.plot != 0, sz=args.size, n_pi=n_pi, bond_orders=bo_total, ascii_art=art)
        else:
            plot_kekule_phases(atoms, k, bo_raw=bo_raw, bo_snap=bo_snap, title=title, fname=args.out, show=args.plot != 0, sz=args.size)
    else:
        n_pi = make_n_pi(atoms)
        plot_system(atoms, title=args.title, fname=args.out, show=args.plot != 0, sz=args.size, n_pi=n_pi, ascii_art=art)

    bt = mol_bond_types(atoms, bo_snap=bo_snap if args.kekule else None,  allow_aromatic=(args.aromatic != 0), kekule=args.kekule)
    mol_fname = export_mol(atoms, mol_opt=args.mol, out_path=args.out,title=args.example, bond_types=bt)
    if mol_fname:
        print(f"Saved MOL: {mol_fname}")
    if args.xyz:
        atoms.saveXYZ(args.xyz)


if __name__ == '__main__':
    main()
