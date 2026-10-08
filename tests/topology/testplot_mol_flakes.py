#!/usr/bin/env python3
"""testplot_mol_flakes.py — symmetric PAH flake generator + battery for the
molecular spin/charge switch study (doc/ERC_private/task_Molecules.md).

Generator: ring-centre row profiles -> fused-hexagon atom patch -> AtomicGraph
-> per-site edge chemistry -> adjust_h() (H caps along real missing-bond
directions).  Shapes:
  profile_rings(counts)  rows of `counts[j]` fused rings, centred & half-offset
  hex_rings(R)           coronene discs (all-zigzag hexagon)
  tri_rings(n)           triangulene triangles
  random_patch_grow(n)   random connected cell cluster (frontier growth)
  random_patch_cut(set,n) random cut of a graphene patch, connectivity-checked

Random PAH mode (--rand N): connected hex-cell patch -> optional defects ->
adjust_h().  Defect ops on the AtomicGraph (pre-passivation):
  pentagon_by_cut(mb)  remove a degree-2 rim C (a CH site) + bond its 2 nbrs
                       -> that ring shrinks 6->5 (new edge ~sqrt(3)*aCC —
                       stretched, relaxes under FF)
  stone_wales(mb)      flip an interior bond by 90 deg -> 4 hexagons become
                       a 5-7-7-5 Stone-Wales defect (bond length preserved)
SYMMETRY (lattice fact): adjacent ring rows have opposite col parity, so
x-mirror needs every row centred on one axis (parity-consistent counts) and
y-mirror needs odd nrow -> BOTH mirrors = odd-nrow centred palindromes with
odd end counts (acenes [m], kites [1,2,..,1], tapered [a,b,c,b,a]).
Rectangles >1 row are never x-symmetric; trapezoids/triangles are x-only.
check_mirror_sym() verifies per flake; substitution site = central zigzag tip
(odd top/bottom row -> site lies on the mirror axis).

Two flake sets:
  BASES — convex symmetric set (row profiles): benzene, acene3, pyrene,
          tap343 (ovalene), kite3; validation radicals phenalenyl/triangulene.
  EXT   — non-convex grid fillers defined as ASCII arts via build_art_flake():
          biphenyl (w1h3), pterphenyl (w1h5), w2h5 zigzag strip, w3h3 hourglass.
Overview = width x height matrix (debug/mol_flakes/overview.png), future
substitution sites marked with green circles (central top/bottom apex atoms).

Site chemistry (per-site char strings, applied before adjust_h):
  'C' sp2 =CH- | 'c' sp3 -CH2- | 'N' pyridinic =N- | 'n' -NH- (in-plane H) |
  'O' keto -C(=O)- | 'o' enol =C(OH)-   (exocyclic O added BEFORE adjust_h so
  the tip C stays uncapped; keto O npi=1 no H, enol O npi=0 gets OH-H itself)

Outputs per flake:
  debug/mol_flakes/<name>.png   — canonical skeleton+dot plot
  debug/mol_flakes/<name>.xyz
  debug/mol_flakes/manifest.txt — name, natoms, formula, edge-site counts

Usage:
    python tests/topology/testplot_mol_flakes.py [name|all] [--list] [--sublat]
    python tests/topology/testplot_mol_flakes.py --rand 12 [--disc R] [--pent K] [--sw K] [--seed S] [--grid G]
"""
import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
from spammm.AtomicSystem import AtomicSystem
from spammm import elements

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'mol_flakes')


# ---------------------------------------------------------------- builders

VERTS = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]  # hexagon verts in (col,k)


def profile_rings(counts, axis='auto'):
    """Axial (q,r) ring centres from a row-length profile (bottom->top).

    Rows are centred on a shared col axis via shifts along the row direction
    (q+s,r-s) which preserve ring heights ck=3j.  Consecutive rows have
    opposite lattice parity, so a shared axis exists only when
      even j: m_j % 2 == 1 - a_parity,   odd j: m_j % 2 == a_parity
    for some global a_parity in {0,1}.  axis='auto' picks the consistent
    parity; rows that cannot centre fall back to nearest (flake then fails
    check_mirror_sym -> caught by the manifest).
    Symmetric families found empirically:
      [m]            acene (both mirrors)
      [a,b] trapezoid/perylene family (x-mirror only)
      [a,b,a], [a,b,c,b,a] palindromic odd-nrow (both mirrors if centred)
      [m]*n n>1      rectangles — x-mirror IMPOSSIBLE (row parity conflict)
    """
    if axis == 'auto':
        ok = {}
        for a in (0, 1):
            ok[a] = all((m + a) % 2 == (0 if j % 2 else 1) for j, m in enumerate(counts))
        axis = next((a for a in (0, 1) if ok[a]), 0)
    rings = []
    for j, m in enumerate(counts):
        base = -(m - 1) if j % 2 == 0 else -m                  # unshifted row centre (col units)
        s = int(round((base - axis) / 2))                    # shift (q+s,r-s) centres row on `axis`
        for i in range(m):
            if j % 2 == 0: q, r = j // 2 + i + s, j // 2 - i - s
            else:          q, r = (j + 1) // 2 + i + s, (j - 1) // 2 - i - s
            rings.append((q, r))
    return rings


def hex_rings(R):
    """Coronene family: axial disc max(|q|,|r|,|s|)<=R -> 6 all-zigzag edges."""
    return [(q, r) for q in range(-R, R + 1) for r in range(-R, R + 1)
            if max(abs(q), abs(r), abs(q + r)) <= R]


def tri_rings(n):
    """Triangulene family: axial triangle q,r>=0, q+r<=n (n+1 rows 1..n+1)."""
    return [(q, r) for q in range(n + 1) for r in range(n + 1) if q + r <= n]


def patch_atoms(rings):
    """(col,k)->'C' dict: rasterize axial ring centres to fused-hexagon atoms."""
    atoms = {}
    for q, r in rings:
        cq, ck = r - q, 3 * (q + r)                           # ring centre on (col,k)
        for dc, dk in VERTS:
            atoms[(cq + dc, ck + dk)] = 'C'
    return atoms


def _patch_to_art(atoms):
    """Rasterize (col,k)->char dict to ASCII lines (same convention as make_acene_chain_art)."""
    c0 = 1 - min(c for c, _ in atoms)
    lines = []
    for k in sorted({k for _, k in atoms}, reverse=True):
        row = {c: ch for (c, kk), ch in atoms.items() if kk == k}
        lines.append(''.join(row.get(c - c0, ' ') for c in range(0, max(row) + c0 + 1)))
    return '\n'.join(lines)


def build_rect_flake(nrow, ncol, bot=None, top=None):
    """Rectangular PAH flake (back-compat wrapper)."""
    return build_patch_flake(profile_rings([ncol] * nrow), bot=bot, top=top)


def build_patch_flake(rings, bot=None, top=None):
    """PAH flake from axial ring list: ASCII skeleton -> _flake_from_skel."""
    from spammm.topology.ascii_art_heterocycle import parse_ascii_art
    return _flake_from_skel(parse_ascii_art(_patch_to_art(patch_atoms(rings))), bot=bot, top=top)


def build_art_flake(art, bot=None, top=None):
    """PAH flake from an explicit ASCII art (single-atom or dimer format)."""
    from spammm.topology.ascii_art_heterocycle import parse_ascii_art
    return _flake_from_skel(parse_ascii_art(art), bot=bot, top=top)


def _mb_from_skel(skel):
    """Skeleton AtomicSystem -> MoleculeEditorBackend (graph+sys synced, no H, no site chem)."""
    mb = MoleculeEditorBackend()
    mb.auto_h_cap = False
    amap = [mb._append_atom(list(p), e, pin=None, parent=None, npi=1)
            for e, p in zip(skel.enames, skel.apos)]
    for i, j in skel.bonds:
        mb.graph.add_bond(amap[i], amap[j])
    mb.graph.sync_neighbor_lists()
    mb._sync_sys()
    return mb


def _flake_from_skel(skel, bot=None, top=None):
    """Skeleton AtomicSystem -> AtomicGraph -> site chemistry -> adjust_h.

    bot/top: per-site chars applied to the bottom/top zigzag edge atoms in
    x-sorted order:
      'C' = edge C-H (sp2, 1 H along missing bond dir)
      'c' = edge CH2 (sp3, tetrahedral 2 H)
      'N' = bare pyridinic edge N (no H)
      'n' = protonated edge N-H (in-plane H, added manually)
    Armchair side atoms always stay 'C'.  H caps come from adjust_h(), i.e. they
    point along the real missing-bond directions (NOT fixed cartesian offsets).
    Returns the synced AtomicSystem.
    """
    mb = _mb_from_skel(skel)

    nh_sites = []
    o_sites = []
    def _apply(edge, spec):
        if spec is None:
            return
        atom_list, *_ = mb.graph.to_arrays()
        for idx, ia in enumerate(edge):       # cyclic over edge sites (x-sorted)
            ch = spec[idx % len(spec)]
            a = atom_list[ia]
            if ch in 'Nn':
                a.ename, a.atype = 'N', elements.ELEMENT_DICT['N'][0]
            if ch == 'c':
                a.npi = 0                    # sp3 CH2 -> adjust_h gives tetrahedral 2H
            if ch == 'n':
                nh_sites.append(a)           # pyridinium N-H: sp2 N + in-plane H below
            if ch in 'Oo':
                o_sites.append((a, ch))      # exocyclic =O / -OH (quinone/hydroquinone)
    bots, tops, _sides = edge_sites(mb.sys)
    _apply(bots, bot)
    _apply(tops, top)

    # exocyclic O sites: add the O BEFORE adjust_h so the apex C already has 3
    # heavy neighbours -> no H cap on it.  'O' = keto C=O (O npi=1 -> target
    # sigma 1 -> uncapped); 'o' = enol C-OH (O npi=0 -> target 2 -> adjust_h
    # adds the O-H itself at a tetrahedral angle).
    for a, ch in o_sites:
        heavy = [n for n in a.neighbors if n.alive and n.npi != -1]
        d = np.zeros(3)
        for n in heavy:
            v = n.pos - a.pos
            d += v / np.linalg.norm(v)
        d[2] = 0.0
        d /= np.linalg.norm(d)
        o = mb.graph.add_atom(a.pos - d * (1.23 if ch == 'O' else 1.36), 'O',
                              elements.ELEMENT_DICT['O'][0], pin=None, parent=a,
                              npi=(1 if ch == 'O' else 0))
        mb.graph.add_bond(a, o)
    if o_sites:
        mb.graph.sync_neighbor_lists()

    mb.adjust_h()   # caps along true missing-bond directions; CH2 -> tetrahedral pair

    # 'n' sites (pyridinium N-H): sp2 N gets no cap from adjust_h; add the H
    # manually along the in-plane outward bisector of its two ring bonds.
    for a in nh_sites:
        heavy = [n for n in a.neighbors if n.alive and n.npi != -1]
        d = np.zeros(3)
        for n in heavy:
            v = n.pos - a.pos
            d += v / np.linalg.norm(v)
        d[2] = 0.0
        d /= np.linalg.norm(d)
        h = mb.graph.add_atom(a.pos - d * 1.01, 'H', elements.ELEMENT_DICT['H'][0],
                              pin=None, parent=a, npi=-1)
        mb.graph.add_bond(a, h)
    mb.graph.sync_neighbor_lists()
    mb._sync_sys()
    return mb.sys


def build_ring_flake(rings):
    """Fused-ring molecule (phenalenyl, triangulene) from axial ring coords."""
    mb = MoleculeEditorBackend()
    for qr in rings:
        mb.add_ring(*qr)
    mb._sync_sys()
    return mb.sys


def edge_sites(atoms):
    """Heavy atoms with <3 heavy neighbors, split top/bottom zigzag vs armchair sides.

    Returns (bottom, top, side) index lists, each x-sorted — matches the order
    in which per-site passivation lists are applied by the builder.
    """
    en = [str(e).split('_')[0] for e in atoms.enames]
    nbr = {}
    for i, j in atoms.bonds:
        nbr.setdefault(i, []).append(j)
        nbr.setdefault(j, []).append(i)
    heavy = [i for i in range(len(en)) if en[i] not in ('H', 'E')]
    edge = [i for i in heavy if sum(1 for j in nbr.get(i, []) if en[j] not in ('H', 'E')) < 3]
    ys = atoms.apos[edge, 1]
    y_min, y_max = ys.min(), ys.max()
    # zigzag tip atoms sit exactly at y_max/y_min; everything else -> armchair sides
    # (corner atoms go to sides so both ends are treated symmetrically)
    top = [i for i, y in zip(edge, ys) if y > y_max - 0.4]
    bot = [i for i, y in zip(edge, ys) if y < y_min + 0.4]
    side = [i for i in edge if i not in set(top) | set(bot)]
    key = lambda i: atoms.apos[i, 0]
    return sorted(bot, key=key), sorted(top, key=key), sorted(side, key=key)


def site_str(n, base, switches):
    """Per-site edge string: `base` char everywhere except {site: char}."""
    lst = [base] * n
    for site, ch in switches.items():
        lst[site] = ch
    return ''.join(lst)


def sublattice(atoms):
    """Bipartite A/B coloring of the heavy-atom bond graph (BFS). Returns (n,) int in {0,1,-1}."""
    en = [str(e).split('_')[0] for e in atoms.enames]
    nbr = {}
    for i, j in atoms.bonds:
        if en[i] == 'H' or en[j] == 'H':
            continue
        nbr.setdefault(i, []).append(j)
        nbr.setdefault(j, []).append(i)
    color = np.full(len(en), -1, dtype=int)
    for s in nbr:
        if color[s] != -1:
            continue
        color[s] = 0
        stack = [s]
        while stack:
            i = stack.pop()
            for j in nbr.get(i, []):
                if color[j] == -1:
                    color[j] = 1 - color[i]
                    stack.append(j)
    return color


# ---------------------------------------------------------------- random PAH
# Random connected clusters of hexagonal cells + defect ops that turn hexagons
# into pentagons/heptagons.  Defects act on the AtomicGraph BEFORE adjust_h()
# passivation, so the H caps land along the real missing-bond directions.

RING_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))  # axial neighbours


def _ring_nbs(qr):
    q, r = qr
    return [(q + dq, r + dr) for dq, dr in RING_DIRS]


def patch_connected(rings):
    """True if the cell set is one connected piece (BFS on axial ring adjacency)."""
    rings = set(rings)
    seen = {next(iter(rings))}
    stack = list(seen)
    while stack:
        for nb in _ring_nbs(stack.pop()):
            if nb in rings and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(rings)


def random_patch_grow(n_rings, rng):
    """Random connected cluster of n_rings hex cells: repeatedly attach a random frontier cell."""
    rings = {(0, 0)}
    while len(rings) < n_rings:
        front = sorted({nb for qr in rings for nb in _ring_nbs(qr)} - rings)
        rings.add(front[rng.integers(len(front))])
    return rings


def random_patch_cut(rings, n_target, rng, max_trials=100000):
    """Randomly remove cells down to n_target, rejecting cuts that would disconnect the patch."""
    rings = set(rings)
    for _ in range(max_trials):
        if len(rings) <= n_target:
            return rings
        trial = rings - {tuple(rings)[rng.integers(len(rings))]}
        if patch_connected(trial):
            rings = trial
    raise RuntimeError(f'random_patch_cut: could not reach {n_target} connected cells')


def _heavy_nbrs(a):
    """Alive non-H neighbours of an Atom."""
    return [n for n in a.neighbors if n.alive and n.npi != -1]


def _avoid_dist(x, y, avoid):
    """BFS edge-distance x->y over alive heavy atoms, excluding `avoid`; None if unreachable."""
    dist = {x: 0}
    front = [x]
    while front:
        nxt = []
        for a in front:
            if a is y:
                return dist[a]
            for n in _heavy_nbrs(a):
                if n in dist or n in avoid:
                    continue
                dist[n] = dist[a] + 1
                nxt.append(n)
        front = nxt
    return None


def pentagon_by_cut(mb, rng):
    """Hexagon -> pentagon: remove a random degree-2 rim carbon (a CH site whose
    2 neighbours sit 4 steps apart = in one 6-ring) and bond the neighbours.
    The new edge is stretched (~sqrt(3)*a_CC) — topology is an exact pentagon,
    geometry relaxes under FF.  Returns [u_id, w_id] of the new bond or None."""
    cands = []
    for a in mb.graph.atoms.values():
        if not (a.alive and a.npi == 1 and a.ename == 'C'):
            continue
        nb = _heavy_nbrs(a)
        if len(nb) != 2:
            continue
        if mb.graph.get_bond(nb[0], nb[1]) is not None:
            continue                                        # would close a triangle
        if _avoid_dist(nb[0], nb[1], {a}) != 4:
            continue                                        # only shrink 6-rings
        cands.append(a)
    if not cands:
        return None
    v = cands[rng.integers(len(cands))]
    u, w = _heavy_nbrs(v)
    for h in mb.graph.h_children(v):
        mb.graph.remove_atom(h)
    mb.graph.remove_atom(v)                                 # soft-del atom + both bonds
    mb.graph.add_bond(u, w)
    return [u._id, w._id]


def stone_wales(mb, rng):
    """Flip a random interior C-C bond by 90 deg (Stone-Wales): the two hexagons
    sharing the bond become pentagons and the two flanking rings become
    heptagons (5-7-7-5).  Each end keeps its wing atom on its own side and gains
    the partner's wing on that side; |a-b| is preserved.  Returns the flipped
    atoms' ids, or None if no interior bond between two 6-ring faces qualifies."""
    hb = [b for b in mb.graph.bonds.values()
          if b.alive and b.a.alive and b.b.alive and b.a.npi == 1 and b.b.npi == 1]
    for bi in rng.permutation(len(hb)):
        a1, a2 = hb[bi].a, hb[bi].b
        wa = [n for n in _heavy_nbrs(a1) if n is not a2]
        wb = [n for n in _heavy_nbrs(a2) if n is not a1]
        if len(wa) != 2 or len(wb) != 2:
            continue                                        # rim bond — not interior
        if len({id(x) for x in wa + wb}) < 4:
            continue                                        # wings not 4 distinct atoms
        mid = 0.5 * (a1.pos + a2.pos)
        d = a2.pos - a1.pos
        dl = np.linalg.norm(d)
        perp = np.array([-d[1], d[0], 0.0]) / dl
        for s in ((perp, -perp) if rng.integers(2) else (-perp, perp)):
            pap = [n for n in wa if np.dot(n.pos - mid, s) > 0]   # a-wing on +s side
            qbp = [n for n in wb if np.dot(n.pos - mid, s) > 0]   # b-wing on +s side
            if len(pap) != 1 or len(qbp) != 1:
                continue
            pam = wa[0] if wa[1] is pap[0] else wa[1]
            qbm = wb[0] if wb[1] is qbp[0] else wb[1]
            if mb.graph.get_bond(a1, qbp[0]) or mb.graph.get_bond(a2, pam):
                continue                                        # already bonded
            avoid = {a1, a2}
            if _avoid_dist(pap[0], qbp[0], avoid) != 3 or _avoid_dist(pam, qbm, avoid) != 3:
                continue                                        # shared faces must be 6-rings
            mb.graph.remove_bond(mb.graph.get_bond(a1, pam))
            mb.graph.remove_bond(mb.graph.get_bond(a2, qbp[0]))
            mb.graph.add_bond(a1, qbp[0])
            mb.graph.add_bond(a2, pam)
            a1.pos = mid + s * (dl / 2)
            a2.pos = mid - s * (dl / 2)
            mb._rings_dirty = True
            return [a1._id, a2._id]
    return None


def build_random_pah(n_rings, seed=None, n_pent=0, n_sw=0, base_R=None):
    """Random all-carbon PAH flake with H-passivated edges.

    Patch: base_R -> cut hex_rings(base_R) disc down to n_rings cells,
    else grow a random connected cluster of n_rings cells.
    Defects: n_pent rim-vertex cuts (6->5 ring), n_sw Stone-Wales flips (5-7-7-5).
    Returns (AtomicSystem, [sys indices of defect-touched atoms]).
    """
    from spammm.topology.ascii_art_heterocycle import parse_ascii_art
    rng = np.random.default_rng(seed)
    rings = random_patch_cut(hex_rings(base_R), n_rings, rng) if base_R else random_patch_grow(n_rings, rng)
    assert len(rings) == n_rings and patch_connected(rings)
    mb = _mb_from_skel(parse_ascii_art(_patch_to_art(patch_atoms(rings))))
    ids = []
    # Bond flips (Stone-Wales) produce heptagons — generally discouraged.
    # Pentagons are made ONLY by cutting an atom from a hexagon
    # (pentagon_by_cut); never via bond flips. 'sw' kept only for explicit
    # legacy requests; do not use it as a fallback for 'pent'.
    for kind in ['pent'] * n_pent + ['sw'] * n_sw:
        r = pentagon_by_cut(mb, rng) if kind == 'pent' else stone_wales(mb, rng)
        if r:
            ids += r
    mb.adjust_h()
    mb._sync_sys()
    return mb.sys, [mb._atom_idx_map[i] for i in ids]


def relax_uff(atoms, nsteps=1500, dt=0.05, damp=0.80, block=250, ftol=0.05, tag='', uff=None, bKekule=True):
    """GPU UFF relax via fused relax_nsteps_global_UFF (all steps in one kernel launch
    per block — no host round-trips; UFF_Builder treewalk must stay per-flake, it
    explodes on merged graphs). Blocks of `block` steps print E/fmax; early-stop
    at fmax<ftol. Planar input + mirror-symmetric FF => z drift ~1e-6; z zeroed.
    `uff` = shared UFF_cl instance (context/program reused across flakes).
    bKekule=False skips the O(n^3) limit-resonance treewalk — valid for
    all-aromatic (fully sp2/resonant) systems.
    Returns (apos_relaxed, E, fmax, zdrift)."""
    from spammm.forcefields.UFF_cl import UFF_cl
    if uff is None:
        uff = UFF_cl()
    uff.toUFF(atoms, bKekule=bKekule)
    mass_map = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0}
    uff.upload_positions(atoms.apos, masses=np.array([mass_map[str(e).split('_')[0]] for e in atoms.enames], dtype=np.float32))
    done = 0; fmax = np.nan; E = np.nan
    while done < nsteps:
        uff.relax_global(nsteps=min(block, nsteps - done), dt=dt, damp=damp)
        uff.run_eval_step()                                    # refresh fapos forces + energies
        E = float(uff.get_total_energy()[0]); f = uff.get_forces(0)
        fmax = float(np.linalg.norm(f, axis=1).max()); done += min(block, nsteps - done)
        print(f'  relax{tag} {done:5d}/{nsteps}: E={E:10.3f} fmax={fmax:.3f}', flush=True)
        if fmax < ftol:
            break
    p = uff.get_positions().copy(); zd = float(np.abs(p[:, 2]).max()); p[:, 2] = 0
    return p, E, fmax, zd


def face_sizes(atoms, max_ring=14):
    """Ring sizes of fused-patch faces: shortest cycle through each heavy-heavy
    bond (BFS with that bond removed), deduplicated by atom set."""
    en = [str(e).split('_')[0] for e in atoms.enames]
    hb = [(i, j) for i, j in atoms.bonds if en[i] != 'H' and en[j] != 'H']
    nbr = {}
    for i, j in hb:
        nbr.setdefault(i, []).append(j)
        nbr.setdefault(j, []).append(i)
    faces = set()
    for i, j in hb:
        dist = {i: 0}
        prev = {i: -1}
        front = [i]
        while front and j not in dist:
            nxt = []
            for x in front:
                if dist[x] >= max_ring:
                    continue
                for y in nbr.get(x, []):
                    if y in dist or (x == i and y == j):
                        continue
                    dist[y] = dist[x] + 1
                    prev[y] = x
                    nxt.append(y)
            front = nxt
        if j not in dist:
            continue
        cyc = set()
        x = j
        while x != -1:
            cyc.add(x)
            x = prev[x]
        faces.add(frozenset(cyc))
    return sorted(len(f) for f in faces)


# ---------------------------------------------------------------- battery
# patch entries: ('patch', rings, bottom_sites, top_sites) — rings = axial (q,r)
#   list from profile_rings()/hex_rings()/tri_rings(); per-site char strings
#   cyclic over x-sorted zigzag edge atoms: 'C'=CH 'c'=CH2 'N'=bare-N 'n'=N-H
# All shapes convex + x/y-mirror symmetric (palindromic centred profiles),
# triangulene = y-symmetric limit (apex up).

# SYMMETRIC battery: both mirrors, width<=4 rings, height<=5 rows, odd
# top/bottom row -> central zigzag tip lies ON the mirror axis = the single
# future =N-/-NH- substitution site (marked by a green circle; NO substitution
# is applied yet).  Convex odd-nrow palindromes only — rectangles with >1 row
# are never x-symmetric on this lattice (row parity conflict).
#   height\width:  w1        w2          w3            w4
#   h1             [1]       -           [3]           -
#   h3             -         [1,2,1]     -             [3,4,3]
#   h5             -         -           [1,2,3,2,1]   -
BASES = {   # name: row-count profile (width = max count, height = nrow)
    'benzene':    [1],
    'acene3':     [3],
    'pyrene':     [1, 2, 1],
    'tap343':     [3, 4, 3],            # ovalene skeleton
    'kite3':      [1, 2, 3, 2, 1],
}
VALIDATION = {   # spin signposts — triangles are x-mirror only (apex up)
    'phenalenyl':  tri_rings(1),
    'triangulene': tri_rings(2),
}

FLAKES = {name: ('patch', profile_rings(prof), None, None) for name, prof in BASES.items()}
FLAKES.update({name: ('patch', rings, None, None) for name, rings in VALIDATION.items()})

# ---------------------------------------------------------------------------
# EXT battery — NON-convex fillers for the empty w x h cells (user-specified;
# relaxes the taper/convexity rule).  ('art', gridpos(w,h), ascii) entries.
# Marked sites = apex atoms of the extreme atom row (the '*' in the sketches).
# ---------------------------------------------------------------------------
ART_BIPHENYL = """
  C
 C C
 C C
  C
  C
 C C
 C C
  C
"""
ART_TERPHENYL = """
  C
 C C
 C C
  C
  C
 C C
 C C
  C
  C
 C C
 C C
  C
"""
ART_W2H5 = """
  C
 | |
| | |
 | |
| | |
 | |
  C
"""
ART_W3H3 = """
 C C C
| | | |
 | | |
| | | |
 C C C
"""
EXT = {   # name: (w, h) grid cell
    'biphenyl':   (1, 3),
    'pterphenyl': (1, 5),
    'w2h5':       (2, 5),
    'w3h3':       (3, 3),
}
EXT_ARTS = {
    'biphenyl':   ART_BIPHENYL,
    'pterphenyl': ART_TERPHENYL,
    'w2h5':       ART_W2H5,
    'w3h3':       ART_W3H3,
}
FLAKES.update({name: ('art', EXT_ARTS[name]) for name in EXT})

# future states per base (to enable when substitution runs start):
#   <b>_2N    central top+bottom edge atoms -> 'N'  (bare pyridinic)
#   <b>_2N_1H one central N protonated      -> 'n'
#   <b>_2N_2H both protonated               -> 'n','n'
#   <b>_1CH2  central top site              -> 'c' (sp3)


def central_sites(atoms):
    """The two marked substitution sites: central atom of top & bottom zigzag edge."""
    bots, tops, _ = edge_sites(atoms)
    return bots[len(bots) // 2], tops[len(tops) // 2]


# ---------------------------------------------------------------- plotting

def draw_flake(ax, atoms, sz=30., bSublattice=False, highlight=None):
    """Canonical molecule style: thin grey skeleton + element dots (no rims).
    bSublattice: tint heavy atoms by bipartite sublattice (A=red, B=blue ring).
    `highlight` = atom indices circled.
    """
    apos = atoms.apos
    en = [str(e).split('_')[0] for e in atoms.enames]
    for i, j in atoms.bonds:
        ax.plot([apos[i, 0], apos[j, 0]], [apos[i, 1], apos[j, 1]], '-', c='0.55', lw=1.0, zorder=1)
    colors = [elements.ELEMENT_DICT[e][8] for e in en]
    sizes = [elements.ELEMENT_DICT[e][6] * sz for e in en]
    ax.scatter(apos[:, 0], apos[:, 1], c=colors, s=sizes, zorder=3, linewidths=0)
    if bSublattice:
        col = sublattice(atoms)
        for i in range(len(en)):
            if en[i] == 'H' or col[i] < 0:
                continue
            ec = 'tab:red' if col[i] == 0 else 'tab:blue'
            ax.scatter([apos[i, 0]], [apos[i, 1]], facecolors='none', edgecolors=ec,
                       s=elements.ELEMENT_DICT[en[i]][6] * sz * 2.2, linewidths=0.7, zorder=2)
    if highlight is not None:
        for i in highlight:
            ax.scatter([apos[i, 0]], [apos[i, 1]], facecolors='none', edgecolors='lime',
                       s=elements.ELEMENT_DICT[en[i]][6] * sz * 3.4, linewidths=1.4, zorder=4)
    ax.set_aspect('equal')
    ax.axis('off')


def check_mirror_sym(atoms, tol=0.05):
    """Return (sym_x, sym_y): heavy-atom skeleton invariant under vertical/horizontal mirror."""
    en = [str(e).split('_')[0] for e in atoms.enames]
    heavy = np.array([i for i in range(len(en)) if en[i] not in ('H', 'E')])
    P = atoms.apos[heavy]
    c0 = 0.5 * (P.min(0) + P.max(0))                    # mirror through bbox centre
    def sym(axis):
        Q = P.copy(); Q[:, axis] = 2 * c0[axis] - Q[:, axis]
        for p in Q:                                   # every point has a mirror twin
            if np.abs(P - p).sum(1).min() > tol:
                return False
        return True
    return sym(0), sym(1)


def formula(atoms):
    en = [str(e).split('_')[0] for e in atoms.enames]
    return ' '.join(f'{e}{en.count(e)}' for e in sorted(set(en)))


def build_one(name, spec):
    if spec[0] == 'rings':
        return build_ring_flake(spec[1])
    if spec[0] == 'art':
        return build_art_flake(spec[1])
    _tag, rings, bot, top = spec
    return build_patch_flake(rings, bot=bot, top=top)


def run_one(name, spec, bSublattice=False):
    atoms = build_one(name, spec)
    bot_s, top_s, side_s = edge_sites(atoms)
    hl = list(central_sites(atoms))          # mark the two =N-/-NH- target sites
    sx, sy = check_mirror_sym(atoms)
    fig, ax = plt.subplots(figsize=(7, 4))
    draw_flake(ax, atoms, bSublattice=bSublattice, highlight=hl or None)
    ax.set_title(f'{name}: {formula(atoms)}  sym x={sx} y={sy}  |edge| bot{len(bot_s)} top{len(top_s)} side{len(side_s)}')
    png = os.path.join(OUTDIR, f'{name}.png')
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    xyz = os.path.join(OUTDIR, f'{name}.xyz')
    with open(xyz, 'w') as f:
        f.write(f'{len(atoms.enames)}\n{name} {formula(atoms)}\n')
        for e, p in zip(atoms.enames, atoms.apos):
            f.write(f'{str(e).split("_")[0]:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
    print(f'REVIEW: {png}\n        {xyz}  ({formula(atoms)})')
    return name, len(atoms.enames), formula(atoms), len(bot_s), len(top_s), len(side_s)


def run_rand(args):
    """--rand driver: (args.grid or 1) random flakes -> per-flake PNG+XYZ + grid overview."""
    n = args.grid or 1
    gfig = gax = None
    if args.grid:
        nc = int(np.ceil(np.sqrt(n)))
        nr = int(np.ceil(n / nc))
        gfig, gax = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.2 * nr), squeeze=False)
        gax = gax.ravel()
    os.makedirs(OUTDIR, exist_ok=True)
    wdef = np.array([float(x) for x in args.ndef.split(',')], float) if args.ndef else None
    if wdef is not None:
        wdef /= wdef.sum()
    tb0 = time.time()
    batch = []
    for k in range(n):
        seed = args.seed + k
        n_pent, n_sw = args.pent, args.sw
        if wdef is not None:                                   # sample defect count ~ wdef, then pent/SW 50:50
            rng = np.random.default_rng(seed)
            nd = int(rng.choice(len(wdef), p=wdef)) + 1
            n_pent = int(rng.binomial(nd, 0.5)); n_sw = nd - n_pent
        atoms, hl = build_random_pah(args.rand, seed=seed, n_pent=n_pent, n_sw=n_sw, base_R=args.disc or None)
        name = f'rand{args.rand}' + (f'd{args.disc}' if args.disc else '') + (f'p{n_pent}' if n_pent else '') + (f'w{n_sw}' if n_sw else '') + f'_s{seed}'
        batch.append([atoms, hl, name, atoms.apos.copy(), n_pent, n_sw])
    t_build = time.time() - tb0
    stats = None
    if args.relax:
        from spammm.forcefields.UFF_cl import UFF_cl
        uff = UFF_cl()
        t0 = time.time()
        stats = []; k = extra = 0
        while k < n:
            try:
                stats.append(relax_uff(batch[k][0], nsteps=args.relax, tag=f'[{k}]', uff=uff))
                k += 1
            except Exception as e:                           # UFF_Builder can fail on some defected topologies -> resample this flake
                extra += 1
                print(f'  [{k}] UFF failed: {e} -> resample', flush=True)
                seed = args.seed + 10000 * extra + k
                n_pent, n_sw = batch[k][4], batch[k][5]
                atoms, hl = build_random_pah(args.rand, seed=seed, n_pent=n_pent, n_sw=n_sw, base_R=args.disc or None)
                name = f'rand{args.rand}' + (f'd{args.disc}' if args.disc else '') + (f'p{n_pent}' if n_pent else '') + (f'w{n_sw}' if n_sw else '') + f'_s{seed}'
                batch[k][:4] = [atoms, hl, name, atoms.apos.copy()]
        t_relax = time.time() - t0
        print(f'UFF relax: {n} flakes x <= {args.relax} steps, {t_relax:.2f}s total, E={[f"{s[1]:.2f}" for s in stats]}')
    tr0 = time.time()
    for k, (atoms, hl, name, pos0, _, _) in enumerate(batch):
        fs = face_sizes(atoms)
        fstr = ','.join(f'{s}:{fs.count(s)}' for s in sorted(set(fs)))
        sx, sy = check_mirror_sym(atoms)
        title = f'{name}: {formula(atoms)}  rings[{fstr}]  sym x={sx} y={sy}'
        if args.relax:
            apos_r, E, fmax, zd = stats[k]
            atoms.apos = apos_r
            title += f'   UFF E={E:.2f} fmax={fmax:.2f} zdrift={zd:.1e}' + ('  BAD-FMAX' if fmax > 1.0 else '')
            name += 'r'
        fig, axs = plt.subplots(1, 2 if args.relax else 1, figsize=(14 if args.relax else 7, 4), squeeze=False)
        if args.relax:
            draw_flake(axs[0, 0], AtomicSystem(apos=pos0, enames=list(atoms.enames), bonds=atoms.bonds), bSublattice=args.sublat, highlight=hl or None)
            axs[0, 0].set_title(f'{name[:-1]} input', fontsize=9)
            draw_flake(axs[0, 1], atoms, bSublattice=args.sublat, highlight=hl or None)
            axs[0, 1].set_title(title, fontsize=9)
        else:
            draw_flake(axs[0, 0], atoms, bSublattice=args.sublat, highlight=hl or None)
            axs[0, 0].set_title(title)
        png = os.path.join(OUTDIR, f'{name}.png')
        fig.savefig(png, dpi=140, bbox_inches='tight')
        plt.close(fig)
        xyz = os.path.join(OUTDIR, f'{name}.xyz')
        with open(xyz, 'w') as f:
            f.write(f'{len(atoms.enames)}\n{name} {formula(atoms)}\n')
            for e, p in zip(atoms.enames, atoms.apos):
                f.write(f'{str(e).split("_")[0]:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
        print(f'REVIEW: {png}\n        {xyz}  ({formula(atoms)}, rings[{fstr}])')
        if gax is not None:
            draw_flake(gax[k], atoms, sz=26., highlight=hl or None)
            gax[k].set_title(f'{name}\n{formula(atoms)}', fontsize=8)
    if gfig is not None:
        for ax in gax[n:]:
            ax.axis('off')
        gfig.tight_layout()
        gpng = os.path.join(OUTDIR, f'rand{args.rand}_grid_s{args.seed}.png')
        gfig.savefig(gpng, dpi=150, bbox_inches='tight')
        plt.close(gfig)
        print(f'REVIEW: {gpng}')
    t_render = time.time() - tr0
    print(f'TIMING: build={t_build:.2f}s relax={t_relax if args.relax else 0.0:.2f}s render={t_render:.2f}s  (per flake: build={t_build / n * 1e3:.1f}ms relax={(t_relax / n if args.relax else 0.0) * 1e3:.1f}ms render={t_render / n * 1e3:.1f}ms)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?', default='all')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--sublat', action='store_true', help='draw A/B sublattice tint rings')
    ap.add_argument('--rand', type=int, default=0, metavar='N', help='random connected patch of N hex cells')
    ap.add_argument('--disc', type=int, default=0, metavar='R', help='with --rand: cut cells from hex_rings(R) disc down to N')
    ap.add_argument('--pent', type=int, default=0, metavar='K', help='rim-vertex cuts (hexagon->pentagon each)')
    ap.add_argument('--sw', type=int, default=0, metavar='K', help='interior Stone-Wales bond flips (->5-7-7-5)')
    ap.add_argument('--ndef', type=str, default=None, metavar='W', help="defect-count weights, e.g. '0.8,0.15,0.05' => 1/2/3 defects; each defect randomly pent or SW (overrides --pent/--sw)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grid', type=int, default=0, metavar='G', help='with --rand: sample G flakes (seeds seed..seed+G-1) into an overview grid')
    ap.add_argument('--relax', type=int, default=0, metavar='NSTEPS', help='GPU UFF relax, max steps budget (fused kernel, blocks of 200 with E/fmax print, early-stop fmax<0.05; z kept 0)')
    args = ap.parse_args()
    if args.list:
        for k in FLAKES:
            print(' ', k)
        return
    if args.rand:
        run_rand(args)
        return
    names = list(FLAKES) if args.name == 'all' else [args.name]
    rows = []
    for name in names:
        if name not in FLAKES:
            raise KeyError(f'unknown flake {name}; --list to see all')
        rows.append(run_one(name, FLAKES[name], bSublattice=args.sublat))
    if args.name == 'all':   # overview panel — width x height matrix + validation row
        widths = [1, 2, 3, 4]
        heights = [1, 3, 5]
        fig, axes = plt.subplots(len(heights) + 1, len(widths), figsize=(3.2 * len(widths), 3.2 * (len(heights) + 1)))
        grid = {(max(p), len(p)): n for n, p in BASES.items()}
        grid.update({pos: n for n, pos in EXT.items()})    # non-convex fillers
        for irow, h in enumerate(heights):
            axes[irow, 0].text(-0.15, 0.5, f'h{h}', ha='right', va='center', fontsize=11, transform=axes[irow, 0].transAxes)
            for icol, w in enumerate(widths):
                ax = axes[irow, icol]
                name = grid.get((w, h))
                if name is None:
                    ax.set_title(f'w{w}' if irow == 0 else '', fontsize=9)  # column header
                    ax.text(0.5, 0.5, '-', ha='center', va='center', fontsize=16, c='0.7', transform=ax.transAxes)
                    ax.axis('off')
                    continue
                atoms = build_one(name, FLAKES[name])
                draw_flake(ax, atoms, sz=26., bSublattice=args.sublat, highlight=list(central_sites(atoms)))
                tcol = 'teal' if name in EXT else 'k'      # EXT = non-convex fillers
                ax.set_title((f'w{w}\n' if irow == 0 else '') + f'{name}\n{formula(atoms)}', fontsize=9, color=tcol)
        for ax, (name, rings) in zip(axes[len(heights), :], VALIDATION.items()):
            atoms = build_one(name, FLAKES[name])
            draw_flake(ax, atoms, sz=26., bSublattice=args.sublat)
            ax.set_title(f'{name}\n{formula(atoms)}', fontsize=9)
        for ax in axes[len(heights), len(VALIDATION):]:
            ax.axis('off')
        fig.tight_layout()
        png = os.path.join(OUTDIR, 'overview.png')
        fig.savefig(png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'REVIEW: {png}')
    mf = os.path.join(OUTDIR, 'manifest.txt')
    with open(mf, 'w') as f:
        f.write('name  natoms  formula  n_bot n_top n_side\n')
        for r in rows:
            f.write('%-14s %4d  %-18s %3d %3d %3d\n' % r)
    print(f'manifest -> {mf}')


if __name__ == '__main__':
    main()
