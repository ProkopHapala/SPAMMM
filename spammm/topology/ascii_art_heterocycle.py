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
        if any(ch.isalpha() for _, ch in tokens):
            for c, ch in tokens:
                xi = c // 2
                i = _add_atom(c * dx / 2.0, y, ch, r, xi)
                row_atoms.setdefault(r, []).append(i)
        else:
            for c, ch in tokens:
                if ch not in ('|', '-', '.'):
                    continue
                xi = c // 2
                x = c * dx / 2.0
                if ch == '.':
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
                    if np.linalg.norm(p - np.array(pos[j])) < 1.6 * aCC:
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


def build_pbc_cell(name=None, art=None, hbond_length=2.8, vac_x=10.0, vac_z=10.0, relax_bonds=True, tilt=None, zigzag=None, slant=None):
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
    defaults = PBC_CELL_PARAMS.get(name, {})
    if tilt is None:
        tilt = defaults.get('tilt', PBC_TILT_DEFAULT)
    if zigzag is None:
        zigzag = defaults.get('zigzag', 0.0)
    if slant is None:
        slant = defaults.get('slant', 0.0)
    if art is None:
        art = PBC_CHAIN_ARTS[name]
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

    # junction endpoint heavy atoms per ':' row: (upper_atom, lower_atom),
    # sorted so jpairs[j] connects blocks j and j+1 (hbonds_ascii order != row order)
    jpairs = []
    for ih, ia in atoms.hbonds_ascii:
        ngh = [jj for jj in atoms.ngs[ih] if atoms.enames[jj] != 'H']
        assert len(ngh) == 1, f"junction H {ih} has {len(ngh)} heavy neighbours"
        idon = ngh[0]
        bd, ba = _block_of(idon), _block_of(ia)
        assert abs(bd - ba) == 1, f"junction must connect adjacent blocks (got {bd},{ba})"
        jpairs.append((min(bd, ba), (idon, ia) if bd < ba else (ia, idon)))
    jpairs = [p for _, p in sorted(jpairs)]
    assert len(jpairs) == nblocks - 1

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
    drawn_first.sort(key=lambda i: (rows[i], atoms.apos[i, 0]))
    drawn_last.sort(key=lambda i: (rows[i], atoms.apos[i, 0]))
    for i, j in zip(drawn_first, drawn_last):
        assert atoms.enames[i] == atoms.enames[j], f"image block ename mismatch at {i}/{j}"
    img = {j: i for i, j in zip(drawn_first, drawn_last)}  # last-block atom -> first-block atom

    # (2a) slant mode: straighten each block — rotate rigidly about its centroid so
    #      the axis through its two junction heavy atoms is parallel to y (the bond
    #      relax can skew junction atoms off-axis; straight axial molecules make the
    #      slant offsets accumulate into a pure-y lattice; skipped under zigzag
    #      where junction axes are intentionally rotated)
    if slant and not zigzag:
        for k, blk in enumerate(blocks[:-1]):
            itop = jpairs[k - 1][1] if k >= 1 else img[jpairs[-1][1]]
            ibot = jpairs[k][0]
            pa, pb = atoms.apos[itop], atoms.apos[ibot]
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

    # (2) realign junctions: translate block j+1 so its junction atom sits exactly
    #     hbond_length below block j's junction atom, on a line leaning by slant
    #     deg off y (alternating sign per junction -> offsets cancel, lattice stays
    #     pure y, molecules stay axial; slant mimics the ~120 deg C-O-H kink so the
    #     H-bond/proton path is oblique).  slant=0 -> junctions vertical.
    sa = np.radians(slant)
    for j, (au, al) in enumerate(jpairs):
        pu, pl = atoms.apos[au], atoms.apos[al]
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

    # (3) rotate cell so the lattice vector is pure +y (zigzag may give t an x-component)
    if t[1] < 0:
        apos_c[:, 1] *= -1.0
        t = t * np.array([1.0, -1.0, 1.0])
    phi = np.arctan2(t[0], t[1])
    if abs(phi) > 1e-9:
        c, s = np.cos(-phi), np.sin(-phi)
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
            itop = jpairs[k - 1][1] if k >= 1 else img[jpairs[-1][1]]
            ibot = jpairs[k][0]
            pa, pb = apos_c[remap[itop]], apos_c[remap[ibot]]
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
        id_c, ih_c, ia_c = remap[img.get(idon, idon)], remap[ih], remap[img.get(ia, ia)]
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
