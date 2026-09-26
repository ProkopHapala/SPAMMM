"""ribbon_pbc.py — periodic zigzag graphene nanoribbons and junction cells.

Reusable builders for the PBC ribbon/junction system (N-terminated zigzag edges,
periodic along the ribbon axis x; junction interfaces are N...H-N contacts
along y — either between two stacked ribbons, or of one ribbon bound to
itself across the y boundary).  All builders return the `build_pbc_cell`
contract ``(atoms, lvs, ...)`` so results feed `run_pbc` and
`run_corner_scan_pbc` directly.

Conventions
-----------
- Ribbon axis = x, periodic; stack direction = y; vacuum = z.
- `a_CC` = C-C bond length (1.42 A).  Cell length Lx = ncells * 2*a_CC*cos(30deg).
- Edge termination via PASSIVATION_GROUPS ('N' pyridinic, 'NH' protonated, ...)
  or per-site strings parsed by parse_passivation_string ('n'->NH, 'N'->N, ...).
- Junction site states: string of 2*ncells chars over {p,d,0}; first ncells =
  internal-junction sites (x-ordered), last ncells = boundary sites.
    'p' = proton on the UPPER edge of the gap, 'd' = LOWER edge, '0' = bare N...N.
  Exact symmetry: inversion p<->d is a mirror image -> E must be identical
  (parity check used by scan_junction_gap).
- d_DA = N...N heavy-atom distance across a junction interface; H sits 1.01 A
  from its donor (H...A = d_DA - 1.01).

Driver: tests/topology/testplot_ribbon.py -> debug/ribbon/.
Lab note: doc/ERC_private/ribbon_junction_cells.md.
"""
import numpy as np
from spammm.AtomicSystem import AtomicSystem
from spammm.topology.hbond_utils import HbondRecord, junction_bond_lengths

A_CC = 1.42       # default C-C bond length [A]
HAU2EV = 27.211386245988


def _decode_passivation(p):
    """Accept a group name, a per-site passivation string, or a list of groups."""
    from spammm.topology.MoleculeEditorBackend import parse_passivation_string, PASSIVATION_GROUPS
    return parse_passivation_string(p) if isinstance(p, str) and p not in PASSIVATION_GROUPS else p


def build_ribbon_cell(width_chains, ncells, passivation='N', aCC=A_CC, vac_y=12.0, vac_z=12.0):
    """Single zigzag ribbon cell, periodic along x, both edges passivated alike.

    Args:
        width_chains: atom rows across the ribbon (thickness).
        ncells: unit cells along x (>= 2).
        passivation: group name ('N', 'NH', ...) or per-site string ('NnNn').
        vac_y/vac_z: vacuum padding perpendicular to the ribbon [A].

    Returns (atoms, lvs, seam_mask):
        atoms: AtomicSystem; atoms.bonds includes wrap bonds across the x seam.
        lvs: (3,3) — lvs[0]=(Lx,0,0) periodic; y/z are vacuum padding.
        seam_mask: bool per bond — True for bonds crossing the x=0/Lx boundary.
    """
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    assert ncells >= 2, 'need >=2 cells for zigzag periodic topology'
    assert width_chains >= 2, 'need >=2 atom rows for a zigzag ribbon (w=1 is a bare chain, not a ribbon)'
    b = MoleculeEditorBackend(a_CC=aCC)
    passivation = _decode_passivation(passivation)
    b.build_zigzag_ribbon(width_chains=width_chains, length_cells=ncells,
                          passivation_bottom=passivation, passivation_top=passivation,
                          bPeriodicX=True)
    b._sync_sys()
    s = b.sys
    Lx = ncells * 2.0 * aCC * np.cos(np.pi / 6.0)

    apos = np.asarray(s.apos, dtype=float).copy()
    bonds = np.asarray(s.bonds, dtype=np.int32)
    # bond crosses the seam when |dx| > Lx/2 (min-image partner is in the next cell)
    seam = np.abs(apos[bonds[:, 1], 0] - apos[bonds[:, 0], 0]) > 0.5 * Lx

    # center ribbon in the (non-periodic) y direction with vac_y total padding
    h = apos[:, 1].ptp()
    apos[:, 1] -= apos[:, 1].min() - 0.5 * vac_y
    lvs = np.array([[Lx, 0.0, 0.0], [0.0, h + vac_y, 0.0], [0.0, 0.0, vac_z]])

    atoms = AtomicSystem(apos=apos, enames=list(s.enames))
    atoms.bonds = bonds
    return atoms, lvs, seam


EDGE_SWITCH = {'N': ('N', 'NH'), 'C': ('CH', 'CH2'), 'O': ('C-OH', 'C=O')}


def build_edge_switch_cell(width_chains, ncells, state, chem='N', vac_y=12.0, a_CC=A_CC):
    """Single zigzag ribbon, vacuum in y (NO junction), per-site edge-chemistry
    switch — the H-bond-free counterpart of build_self_junction_cell.

    `state`: ncells chars over {p,d,b,0}; 'p' = bottom-edge site switched to the
    hydrogenated group, 'd' = top-edge site, 'b' = BOTH edges of that cell
    switched (needed for os-d0 in the separation scan).  `chem` picks the pair:
      'N': N -> NH   (pyridinic N protonated)
      'C': CH -> CH2 (aromatic edge C -> sp3)
      'O': C-OH -> C=O
    Same 7-state enumeration (junction_state_strings) as the junction cell, so
    J/decomposition/nonlin analyses compare C-vs-N switching directly, without
    the dipole-sheet/H-bond channel the junction stack carries.

    Returns (atoms, lvs, hbonds=[]) — same contract as build_self_junction_cell
    minus hbonds (no junctions to draw).
    """
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    assert ncells >= 2 and width_chains >= 2
    base, sw = EDGE_SWITCH[chem]
    st = [c for c in state.lower() if c in 'pdb0']
    assert len(st) == ncells, f"state needs ncells={ncells} chars from {{p,d,b,0}}, got '{state}'"
    b = MoleculeEditorBackend(a_CC=a_CC)
    b.build_zigzag_ribbon(width_chains=width_chains, length_cells=ncells,
                          passivation_bottom=[sw if c in 'pb' else base for c in st],
                          passivation_top=[sw if c in 'db' else base for c in st],
                          bPeriodicX=True)
    b._sync_sys()
    s = b.sys
    Lx = ncells * 2.0 * a_CC * np.cos(np.pi / 6.0)
    apos = np.asarray(s.apos, dtype=float).copy()
    bonds = np.asarray(s.bonds, dtype=np.int32)
    seam = np.abs(apos[bonds[:, 1], 0] - apos[bonds[:, 0], 0]) > 0.5 * Lx
    # cell height from the heavy-atom backbone ONLY: identical Ly across states
    # (protruding switch-group H's must not change the supercell)
    hvy = np.array([e.split('_')[0] != 'H' for e in s.enames])
    h = apos[hvy, 1].ptp()
    apos[:, 1] -= apos[hvy, 1].min() - 0.5 * vac_y
    lvs = np.array([[Lx, 0.0, 0.0], [0.0, h + vac_y, 0.0], [0.0, 0.0, 20.0]])
    atoms = AtomicSystem(apos=apos, enames=list(s.enames))
    atoms.bonds = bonds
    atoms.seam = seam
    return atoms, lvs, []


def build_ribbon_junction_cell(width_chains=6, ncells=4, d_DA=2.9, shift_x=0.0,
                               bottom='N', top='NH', state=None, pbc_y=True, vac_y=14.0, Lz=20.0,
                               a_CC=A_CC, d_max=2.2, bAlignX=True):
    """Two stacked zigzag ribbons -> periodic N...H-N junction interfaces.

    Bottom ribbon A gets `bottom` passivation on BOTH edges, top ribbon B gets `top`
    (default A='N' pyridinic acceptor edges, B='NH' donor edges with H's pointing
    outward toward A's edges).  With pbc_y=True the cell stacks ...A|B|A'|B'... along
    y so EVERY gap is a junction interface — 2 per cell (internal A.top<->B.bot and
    boundary B.top<->A'.bot), like the molecular PBC chains of build_pbc_cell.
    d_DA is the N...N heavy-atom distance across each junction interface (the H sits
    1.01 A from its donor, i.e. H...A = d_DA-1.01).  Each interface holds `ncells`
    switchable sites along x.

    `state` (optional) overrides bottom/top: a string of 2*ncells chars over
    {p,d,0} — first ncells chars = internal-junction sites, last ncells = boundary.
      'p' = proton on the UPPER edge of the gap (internal: B.bot-NH; boundary: A'.bot-NH)
      'd' = proton on the LOWER edge of the gap (internal: A.top-NH; boundary: B.top-NH)
      'b' = BOTH edges of that x-site protonated (H...H close pair inside one gap)
      '0' = no proton, bare N...N contact.
    Non-pd0 characters are ignored, so 'pppp|dddd' reads as two junction groups.

    Returns (atoms, lvs, hbonds) — same contract as
    ascii_art_heterocycle.build_pbc_cell: `atoms` is an AtomicSystem whose bonds
    include the x-seam wrap bonds, lvs rows are [ribbon dir x, stack dir y, vacuum z],
    and hbonds are HbondRecord with a_shift=+-1 on the boundary-junction partners.
    Feed directly to coordinate_scan.run_corner_scan_pbc with a mapping grouping
    the sites into controls (e.g. [0]*ncells+[1]*ncells = whole-interface transfer).

    Note: zigzag edge tips are staggered by half a cell between the two edges of a
    ribbon for some widths — bAlignX (default) registers the two facing edge rows
    in x; shift_x adds a further manual offset (fraction of Lx).
    """
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    assert width_chains >= 2, 'need >=2 atom rows for a zigzag ribbon (w=1 is a bare chain, not a ribbon)'

    def _build(pass_bot, pass_top):
        b = MoleculeEditorBackend(a_CC=a_CC)
        b.build_zigzag_ribbon(width_chains=width_chains, length_cells=ncells,
                              passivation_bottom=_decode_passivation(pass_bot),
                              passivation_top=_decode_passivation(pass_top),
                              bPeriodicX=True)
        b._sync_sys()
        return b

    if state is not None:
        st = [c for c in state.lower() if c in 'pdb0']
        assert len(st) == 2 * ncells, f"state needs 2*ncells={2*ncells} chars from {{p,d,b,0}}, got '{state}'"
        si, sb = st[:ncells], st[ncells:]
        # internal gap: upper=B.bot, lower=A.top ; boundary gap: upper=A'.bot, lower=B.top
        bA = _build(['NH' if c in 'pb' else 'N' for c in sb], ['NH' if c in 'db' else 'N' for c in si])
        bB = _build(['NH' if c in 'pb' else 'N' for c in si], ['NH' if c in 'db' else 'N' for c in sb])
    else:
        bA = _build(bottom, bottom)
        bB = _build(top, top)
    nA = len(bA.sys.apos)
    be = MoleculeEditorBackend(a_CC=a_CC)
    be.combine_ribbons(bA, bB, L_Hb=0.0, shift_x=shift_x)
    be._sync_sys()

    apos = np.asarray(be.sys.apos, dtype=float).copy()
    enames = list(be.sys.enames)
    assert be.sys.bonds is not None and len(be.sys.bonds) > 0, 'combined ribbons have no bonds'
    bonds = np.asarray(be.sys.bonds, dtype=np.int32)
    Lx = ncells * 2.0 * a_CC * np.cos(np.pi / 6.0)
    pitch = Lx / ncells

    neigh = {}
    for i, j in bonds:
        neigh.setdefault(i, []).append(j)
        neigh.setdefault(j, []).append(i)

    iNO_A = np.array([i for i in range(nA) if enames[i] in ('N', 'O')])
    iNO_B = np.array([i for i in range(nA, len(apos)) if enames[i] in ('N', 'O')])
    y_topN_A, y_botN_A = apos[iNO_A, 1].max(), apos[iNO_A, 1].min()      # edge N/O rows of A
    y_botN_B, y_topN_B = apos[iNO_B, 1].min(), apos[iNO_B, 1].max()      # edge N/O rows of B
    dyB = y_topN_A + d_DA - y_botN_B          # B.bot edge N/O exactly d_DA above A.top edge
    apos[nA:, 1] += dyB

    if bAlignX:
        # register the facing edge rows in x (zigzag tips can be staggered by half a cell)
        edgeB = [i for i in iNO_B if apos[i, 1] < y_botN_B + dyB + 0.1]
        edgeA = [i for i in iNO_A if apos[i, 1] > y_topN_A - 0.1]
        if edgeB and edgeA:
            xd = min(apos[i, 0] for i in edgeB)
            xa_ = min(edgeA, key=lambda i: abs(apos[i, 0] - xd))
            dx = (apos[xa_, 0] - xd + 0.5 * pitch) % pitch - 0.5 * pitch
            if abs(dx) > 1e-6:
                apos[nA:, 0] += dx
                print(f"build_ribbon_junction_cell: auto-aligned top ribbon by {dx:+.3f} A (staggered edges)")
        apos[nA:, 0] %= Lx

    if pbc_y:
        Ly = (y_topN_B + dyB + d_DA) - y_botN_A          # boundary gap = same d_DA
        apos[:, 1] -= y_botN_A - 0.6                    # small margin below bottom edge
    else:
        Ly = apos[:, 1].ptp() + vac_y
        apos[:, 1] -= apos[:, 1].min() - 0.5 * vac_y
    apos[:, 2] += 0.5 * Lz
    lvs = np.array([[Lx, 0.0, 0.0], [0.0, Ly, 0.0], [0.0, 0.0, Lz]])

    # junction records: every H bonded to N|O is a donor; its acceptor is the
    # nearest N|O of the OTHER ribbon block (min-image along the stack dir y)
    lvec = np.array([0.0, Ly, 0.0])
    shifts = (-1, 0, 1) if pbc_y else (0,)
    acc = [np.array([0, nA]), np.array([nA, len(apos)])]          # acceptor index range per donor block
    hbonds = []
    for ih in range(len(apos)):
        if enames[ih] != 'H':
            continue
        heavy = [j for j in neigh.get(ih, []) if enames[j] != 'H']
        if not heavy or enames[heavy[0]] not in ('N', 'O'):
            continue
        idon = heavy[0]
        lo, hi = acc[0] if idon >= nA else acc[1]
        best = None
        for ia in range(lo, hi):
            if enames[ia] not in ('N', 'O'):
                continue
            for s in shifts:
                d = np.linalg.norm(apos[ih] - (apos[ia] + s * lvec))
                if best is None or d < best[0]:
                    best = (d, ia, s)
        if best is None or best[0] > d_max:
            continue
        d, ia, s = best
        vh_d = apos[idon] - apos[ih]                      # D-H...A angle at H (180 = collinear)
        vh_a = apos[ia] + s * lvec - apos[ih]
        ang = np.degrees(np.arccos(np.clip(np.dot(vh_d, vh_a) / (np.linalg.norm(vh_d) * np.linalg.norm(vh_a)), -1.0, 1.0)))
        hbonds.append(HbondRecord(idon, ih, ia, float(d), float(ang), d_shift=0, a_shift=s))
    hbonds.sort(key=lambda h: (h.a_shift != 0, apos[h.donor_idx, 0]))   # internal junction first, x-ordered

    cell = AtomicSystem(apos=apos, enames=enames)
    cell.atypes = np.asarray(be.sys.atypes, dtype=np.int32)
    cell.bonds = bonds
    return cell, lvs, hbonds


def build_self_junction_cell(width_chains=6, ncells=4, state=None, d_DA=3.0,
                             a_CC=A_CC, Lz=20.0, d_max=2.2):
    """Single-ribbon self-junction cell: the ribbon binds to ITSELF across the y
    cell boundary — its top edge forms N...H-N junctions with the bottom edge of
    its own periodic image above (and vice versa below).  Every edge is a
    junction edge: no dangling/bare N edges anywhere in the stack.

    `width_chains` = number of ATOM chains across the ribbon; ring rows =
    width_chains/2 - 1 (w_chains=4 -> 1 ring row, polyacene-like; =6 -> 2;
    =10 -> 4; =18 -> 8).  `state`: ncells chars over {p,d,0}, one junction site
    per x position:
      'p' = proton on the UPPER edge of the gap -> this cell's BOTTOM-edge site
            is protonated (it is the upper edge of the junction below; the image
            above carries the same H over this cell's top edge), H points down.
      'd' = proton on the LOWER edge of the gap -> TOP-edge site, H points up.
      'b' = BOTH edges of that x-site protonated (as in build_edge_switch_cell).
      '0' = bare N...N contact.
    Inversion p<->d is a mirror image (top<->bottom edges swap, identical
    junctions on both boundaries) -> parity check.

    Ly = (edge-N to edge-N distance) + d_DA; the cell boundary sits mid-gap.
    Zigzag top/bottom edges are staggered by half a cell in x, so lvs[1] is
    tilted (dx) to register the image's bottom edge with the home top edge —
    junctions stay vertical with 1:1 N...N pairing at distance d_DA.

    Returns (atoms, lvs, hbonds) — same contract as build_ribbon_junction_cell
    (hbonds carry a_shift=+-1 since every junction crosses the y boundary).
    """
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    assert width_chains >= 2, 'need >=2 atom rows for a zigzag ribbon (w=1 is a bare chain, not a ribbon)'
    st = [c for c in (state or '0' * ncells).lower() if c in 'pdb0']
    assert len(st) == ncells, f"state needs ncells={ncells} chars from {{p,d,b,0}}, got '{state}'"
    b = MoleculeEditorBackend(a_CC=a_CC)
    b.build_zigzag_ribbon(width_chains=width_chains, length_cells=ncells,
                          passivation_bottom=['NH' if c in 'pb' else 'N' for c in st],
                          passivation_top=['NH' if c in 'db' else 'N' for c in st],
                          bPeriodicX=True)
    b._sync_sys()

    apos = np.asarray(b.sys.apos, dtype=float).copy()
    enames = list(b.sys.enames)
    assert b.sys.bonds is not None and len(b.sys.bonds) > 0, 'ribbon has no bonds'
    bonds = np.asarray(b.sys.bonds, dtype=np.int32)
    Lx = ncells * 2.0 * a_CC * np.cos(np.pi / 6.0)
    pitch = Lx / ncells

    iNO = np.array([i for i, e in enumerate(enames) if e in ('N', 'O')], dtype=int)
    y_bot, y_top = apos[iNO, 1].min(), apos[iNO, 1].max()          # edge N/O rows
    eb = sorted(iNO[np.abs(apos[iNO, 1] - y_bot) < 0.2], key=lambda i: apos[i, 0])
    et = sorted(iNO[np.abs(apos[iNO, 1] - y_top) < 0.2], key=lambda i: apos[i, 0])
    # tilt the y lattice so the +y image's bottom edge registers with the top edge;
    # RAW difference keeps same-index 1:1 site pairing (a centered wrap to -pitch/2
    # would shift pairing by one site and force x-wrapped junction partners)
    dx = apos[et[0], 0] - apos[eb[0], 0]
    Ly = (y_top - y_bot) + d_DA
    apos[:, 1] += 0.5 * d_DA - y_bot        # junction gaps centered on y=0 and y=Ly
    apos[:, 2] += 0.5 * Lz
    lvs = np.array([[Lx, 0.0, 0.0], [dx, Ly, 0.0], [0.0, 0.0, Lz]])

    neigh = {}
    for i, j in bonds:
        neigh.setdefault(i, []).append(j)
        neigh.setdefault(j, []).append(i)
    lv = lvs[1]
    hbonds = []
    for ih in range(len(apos)):
        if enames[ih] != 'H':
            continue
        heavy = [j for j in neigh.get(ih, []) if enames[j] != 'H']
        if not heavy or enames[heavy[0]] not in ('N', 'O'):
            continue
        idon = heavy[0]
        best = None
        for ia in iNO:                                   # junction partner is always across the y boundary
            for s in (-1, 1):
                d = np.linalg.norm(apos[ih] - (apos[ia] + s * lv))
                if best is None or d < best[0]:
                    best = (d, ia, s)
        if best is None or best[0] > d_max:
            continue
        d, ia, s = best
        vh_d = apos[idon] - apos[ih]
        vh_a = apos[ia] + s * lv - apos[ih]
        ang = np.degrees(np.arccos(np.clip(np.dot(vh_d, vh_a) / (np.linalg.norm(vh_d) * np.linalg.norm(vh_a)), -1.0, 1.0)))
        hbonds.append(HbondRecord(idon, ih, ia, float(d), float(ang), d_shift=0, a_shift=s))

    cell = AtomicSystem(apos=apos, enames=enames)
    cell.atypes = np.asarray(b.sys.atypes, dtype=np.int32) if b.sys.atypes is not None else None
    cell.bonds = bonds
    return cell, lvs, hbonds


D_DA_OPT = {'NN': 2.90, 'NO': 2.80, 'OO': 2.75}   # initial D...A optima per junction heavy-atom element pair [A]; refine via a d_DA mini-scan


def build_mol_ribbon_cell(mol_art=None, mol=None, width_chains=8, ncells=4, d_DA=None,
                          shift_x=0.0, site=None, pbc_y=True, vac_y=14.0, Lz=20.0,
                          a_CC=A_CC, relax_bonds=True, alt_CH=False, tilt_deg=0.0, label='mol_ribbon'):
    """Molecule bridging two ribbon edges: cell = 1 ribbon + 1 molecule.

    The molecule's bottom end H-bonds the ribbon top edge (internal junction)
    and its top end H-bonds the bottom edge of the +y image (boundary
    junction) — 2 junctions per molecule, stack ...|rib|gap|mol|gap|rib'|...
    (doc/ERC_private/mol_ribbon_Htransfer.md).

    Donor/acceptor chemistry is detected from the molecule art: a junction end
    carrying an H (lowercase 'n'/'o' donor in the art -> capped H) faces an
    'N' acceptor edge site; a bare 'N'/'O' end is an acceptor and faces an
    'NH' donor site.  So 'AA' (mol XH2, donates at both ends) uses all-N
    edges, 'BB' (mol X) puts NH at the two junction sites, 'AB'/'BA' mix.
    Non-junction edge sites stay 'N' (bare pyridinic) — or, with
    alt_CH=True, the edge alternates N/CH (N at the junction site and every
    2nd site of its parity; CH elsewhere) — a half-N-doped edge.  tilt_deg
    rigidly rotates the molecule about its tip-tip axis (junction atoms are
    ON the axis, junction registration unchanged), tilting the molecular
    plane out of the ribbon plane to clear the edge C-H's.

    d_DA: None -> per-junction optimum from D_DA_OPT keyed by the sorted
    donor/acceptor element pair ({mol end element, N}); scalar -> both
    junctions; (d_lo, d_hi) -> internal/boundary gaps separately.  Ly is
    assembled so BOTH gaps hold their own d_DA (the molecule spans between
    the two edge rows exactly).

    Returns (atoms, lvs, hbonds) — build_pbc_cell contract; hbonds carry
    a_shift=+-1 on boundary-junction partners; lvs[1] is tilted (dx) so the
    image's bottom edge registers over the molecule's top junction atom
    (same trick as build_self_junction_cell).  atoms.n_ribbon = ribbon atom
    count (mol atoms are appended after).
    """
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    from spammm.topology.ascii_art_heterocycle import mol_from_art
    assert ncells >= 2 and width_chains >= 2
    if mol is None:
        assert mol_art is not None, 'need mol_art or a prebuilt AtomicSystem mol'
        mol = mol_from_art(mol_art, relax_bonds=relax_bonds)
    apos_m = np.asarray(mol.apos, dtype=float).copy()
    enames_m = list(mol.enames)
    bonds_m = np.asarray(mol.bonds, dtype=np.int32)

    ngs_m = {}
    for i, j in bonds_m:
        ngs_m.setdefault(i, []).append(j); ngs_m.setdefault(j, []).append(i)
    iNO_m = np.array([i for i, e in enumerate(enames_m) if e in ('N', 'O')], dtype=int)
    assert len(iNO_m) >= 2, 'molecule needs N/O junction atoms at both ends'
    iNO_m = iNO_m[np.argsort(apos_m[iNO_m, 1])]
    jb, jt = int(iNO_m[0]), int(iNO_m[-1])
    assert apos_m[iNO_m[1], 1] - apos_m[jb, 1] > 0.1 and apos_m[jt, 1] - apos_m[iNO_m[-2], 1] > 0.1, \
        'need a single unique junction N/O at each molecule end'
    don_b = any(enames_m[j] == 'H' for j in ngs_m.get(jb, ()))   # bottom end donates?
    don_t = any(enames_m[j] == 'H' for j in ngs_m.get(jt, ()))   # top end donates?

    if site is None:
        site = ncells // 2
    # ribbon edge chemistry complementary to the facing molecule end:
    # top edge faces mol bottom end (internal junction), bottom edge faces the
    # image mol top end (boundary junction).  Acceptor end -> 'NH' at site.
    # alt_CH: N only at every 2nd site (junction-site parity), CH elsewhere.
    def _edge_pass(don_end):
        return ['NH' if (i == site and not don_end) else
                ('N' if (not alt_CH or (i - site) % 2 == 0) else 'CH') for i in range(ncells)]
    pass_top = _edge_pass(don_b)
    pass_bot = _edge_pass(don_t)
    b = MoleculeEditorBackend(a_CC=a_CC)
    b.build_zigzag_ribbon(width_chains=width_chains, length_cells=ncells,
                          passivation_bottom=pass_bot, passivation_top=pass_top,
                          bPeriodicX=True)
    b._sync_sys()
    apos = np.asarray(b.sys.apos, dtype=float).copy()
    enames = list(b.sys.enames)
    assert b.sys.bonds is not None and len(b.sys.bonds) > 0, 'ribbon has no bonds'
    bonds_r = np.asarray(b.sys.bonds, dtype=np.int32)
    nR = len(apos)
    Lx = ncells * 2.0 * a_CC * np.cos(np.pi / 6.0)

    ihv = np.array([i for i, e in enumerate(enames) if not e.startswith('H')], dtype=int)
    y_bot_r, y_top_r = apos[ihv, 1].min(), apos[ihv, 1].max()          # edge heavy-atom rows
    et = sorted(ihv[apos[ihv, 1] > y_top_r - 0.2], key=lambda i: apos[i, 0])
    eb = sorted(ihv[apos[ihv, 1] < y_bot_r + 0.2], key=lambda i: apos[i, 0])
    assert len(et) == ncells and len(eb) == ncells, f'expected {ncells} edge sites per edge'
    assert enames[et[site]] == 'N' and enames[eb[site]] == 'N', f'{label}: junction sites must be edge N'
    x_t, x_b = apos[et[site], 0], apos[eb[site], 0]

    if d_DA is None:
        d_lo = D_DA_OPT[''.join(sorted((enames_m[jb], 'N')))]          # internal junction
        d_hi = D_DA_OPT[''.join(sorted((enames_m[jt], 'N')))]          # boundary junction
    elif np.isscalar(d_DA):
        d_lo = d_hi = float(d_DA)
    else:
        d_lo, d_hi = (float(v) for v in d_DA)

    if tilt_deg:
        # tilt the molecular plane about the tip-tip axis (Rodrigues); the
        # junction atoms lie on the axis -> registration and h_m unchanged
        v = apos_m[jt] - apos_m[jb]
        v /= np.linalg.norm(v)
        th = np.deg2rad(tilt_deg)
        c, s = np.cos(th), np.sin(th)
        p = apos_m - apos_m[jb]
        apos_m = apos_m[jb] + p * c + np.cross(v, p) * s + np.outer(p @ v, v) * (1.0 - c)

    h_m = apos_m[jt, 1] - apos_m[jb, 1]                # mol junction-to-junction span
    apos_m[:, 0] += x_t - apos_m[jb, 0] + shift_x * Lx
    apos_m[:, 0] %= Lx
    apos_m[:, 1] += y_top_r + d_lo - apos_m[jb, 1]     # bottom end exactly d_lo above the top edge
    nM = len(apos_m)
    apos = np.vstack([apos, apos_m])
    enames += enames_m
    bonds = np.vstack([bonds_r, bonds_m + nR])
    jb_c, jt_c = nR + jb, nR + jt
    if pbc_y:
        Ly = (y_top_r - y_bot_r) + d_lo + h_m + d_hi
        dx = apos[jt_c, 0] - x_b                       # tilt registers image bottom site over mol top end
        apos[:, 1] += 0.5 * d_hi - y_bot_r             # boundary junction gap centered on y=0/Ly
        lvec = np.array([dx, Ly, 0.0])
    else:
        span = apos[:, 1].ptp()
        apos[:, 1] -= apos[:, 1].min() - 0.5 * vac_y
        Ly = span + vac_y
        lvec = np.array([0.0, Ly, 0.0])
    apos[:, 2] += 0.5 * Lz
    lvs = np.array([[Lx, 0.0, 0.0], lvec, [0.0, 0.0, Lz]])
    seam = np.abs(apos[bonds[:, 1], 0] - apos[bonds[:, 0], 0]) > 0.5 * Lx

    # junction records from construction (the pairs are known — geometric H..A
    # search would miss donors whose cap-H points off-axis, e.g. exocyclic O-H):
    #   internal: mol bottom end <-> top-edge site;  boundary: mol top end <->
    #   bottom-edge site of the +y image (donor-side record keeps a_shift).
    neigh = {}
    for i, j in bonds:
        neigh.setdefault(i, []).append(j)
        neigh.setdefault(j, []).append(i)
    pairs = [(jb_c, et[site], 0) if don_b else (et[site], jb_c, 0)]
    if pbc_y:
        pairs.append((jt_c, eb[site], +1) if don_t else (eb[site], jt_c, -1))
    r_xh = 1.01
    hbonds = []
    for idon, iacc, a_sh in pairs:
        hs = [j for j in neigh.get(idon, []) if enames[j] == 'H']
        assert hs, f'{label}: junction donor atom {idon}({enames[idon]}) has no H'
        pD = apos[idon]
        pA = apos[iacc] + a_sh * lvec
        axv = pA - pD
        dist = np.linalg.norm(axv)
        ih = max(hs, key=lambda j: np.dot(apos[j] - pD, axv))   # donor H most aligned with D->A
        apos[ih] = pD + r_xh * axv / dist                      # junction H exactly on the axis
        hbonds.append(HbondRecord(int(idon), int(ih), int(iacc), float(dist - r_xh), 180.0, d_shift=0, a_shift=a_sh))
    hbonds.sort(key=lambda h: (h.a_shift != 0, apos[h.donor_idx, 0]))  # internal junction first

    cell = AtomicSystem(apos=apos, enames=enames)
    cell.atypes = np.concatenate([np.asarray(b.sys.atypes), np.asarray(mol.atypes)]).astype(np.int32)
    cell.bonds = bonds
    cell.seam = seam
    cell.n_ribbon = nR

    # steric-clash check on the finished cell (incl. boundary image pairs);
    # junction D..A/H..A pairs are close by design -> excluded
    from spammm.atomicUtils import check_clashes
    jexc = [p for hb in hbonds for p in ((hb.donor_idx, hb.h_idx), (hb.h_idx, hb.acceptor_idx), (hb.donor_idx, hb.acceptor_idx))]
    check_clashes(cell.apos, cell.enames, cell.bonds, lvec=lvec if pbc_y else None, exclude=jexc, label=label)
    return cell, lvs, hbonds


def build_ribbon(passivation, width_chains, length_cells, Lx, a_CC=A_CC):
    """Build a ribbon and return arrays (mirrors deprecated GrapheneRibbonBuilder.build_ribbon API).

    Returns (pos2d, atypes, elems): 2D positions, atomic numbers, element symbols.
    """
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    backend = MoleculeEditorBackend(a_CC=a_CC)
    xa_nom = a_CC * np.cos(np.pi / 6)
    scale_x = Lx / (2.0 * xa_nom)
    backend.build_zigzag_ribbon(width_chains=width_chains, length_cells=length_cells,
                                  passivation=passivation, scale_x=scale_x, bPeriodicX=False)
    return backend.sys.apos[:, :2].copy(), backend.sys.atypes, list(backend.sys.enames)


def build_two_ribbon_cell(width_chains=4, length_cells=1, Lx=2.4, a_CC=A_CC, L_Hb=2.0, shift_x=0.0):
    """Legacy API: two ribbons separated by an atom-extent gap L_Hb (FireCore convention).

    Prefer build_ribbon_junction_cell (gap set by d_DA = N...N distance).
    Returns (apos, atypes, elems, lvs).
    """
    from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
    backend1 = MoleculeEditorBackend(a_CC=a_CC)
    backend2 = MoleculeEditorBackend(a_CC=a_CC)
    xa_nom = a_CC * np.cos(np.pi / 6)
    scale_x = Lx / (2.0 * xa_nom)
    backend1.build_zigzag_ribbon(width_chains, length_cells, passivation='N', scale_x=scale_x, bPeriodicX=False)
    backend2.build_zigzag_ribbon(width_chains, length_cells, passivation='NH', scale_x=scale_x, bPeriodicX=False)

    apos_N = backend1.sys.apos.copy()
    apos_NH = backend2.sys.apos.copy()
    apos_N[:, 1]  -= apos_N[:, 1].mean()
    apos_NH[:, 1] -= apos_NH[:, 1].mean()
    apos_NH[:, 0] += shift_x * Lx

    y_max_N  = np.max(apos_N[:, 1])
    y_min_NH = np.min(apos_NH[:, 1])
    apos_NH[:, 1] += (y_max_N + L_Hb) - y_min_NH

    y_span_N  = np.max(apos_N[:, 1])  - np.min(apos_N[:, 1])
    y_span_NH = np.max(apos_NH[:, 1]) - np.min(apos_NH[:, 1])
    Ly = y_span_N + y_span_NH + 2 * L_Hb

    apos   = np.vstack([apos_N, apos_NH])
    atypes = np.concatenate([backend1.sys.atypes, backend2.sys.atypes])
    elems  = list(backend1.sys.enames) + list(backend2.sys.enames)

    apos[:, 2] = 0.0
    apos[:, 1] -= apos[:, 1].mean()
    Lz = 20.0
    apos[:, 2] += 0.5 * Lz
    lvs = np.array([[Lx, 0.0, 0.0], [0.0, Ly, 0.0], [0.0, 0.0, Lz]])
    return apos, atypes, elems, lvs


def junction_state_strings(ncells):
    """Unique {p,d,0} protonation configs for ONE junction interface (pbc_y=False).

    Sites = ncells along x; boundary sites forced '0' (outer edges bare N).
    Returns {label: state_string} — unique modulo x-translation and p<->d mirror:
      '0H'       bare N...N at all sites
      '1H-p/d'   single proton, upper/lower edge (mirror pair -> parity check)
      '2H-ss-adj/sep' two protons, SAME edge, adjacent / maximally separated
      '2H-os-adj/sep' two protons, OPPOSITE edges, adjacent / maximally separated
    'adjacent across the x seam' = 'adjacent' under PBC, so not listed separately.
    """
    b = '0' * ncells                     # boundary sites: no junction (pbc_y=False)
    k = max(1, ncells // 2)              # separation offset for 'sep' patterns
    def s(map_):
        return ''.join(map_.get(i, '0') for i in range(ncells)) + '|' + b
    return {'0H'       : s({}),
            '1H-p'     : s({0: 'p'}),
            '1H-d'     : s({0: 'd'}),
            '2H-ss-adj': s({0: 'p', 1: 'p'}),
            '2H-ss-sep': s({0: 'p', k: 'p'}),
            '2H-os-adj': s({0: 'p', 1: 'd'}),
            '2H-os-sep': s({0: 'p', k: 'd'})}


def xscan_state_strings(ncells):
    """Separation scan on an ncells ring (designed for ncells=8): bottom-edge
    switch fixed at site 0, second switch shifted by k primitive cells.
      '2H-os-dK' = p@0 + d@K  opposite edges, K=0..ncells//2 (K=0 -> 'b' at cell 0)
      '2H-ss-dK' = p@0 + p@K  same edge,     K=1..ncells//2
    'b' in a cell char = BOTH edges of that cell switched.
    Singles are translation-invariant, so one 1H-p/1H-d reference serves all K.
    """
    km = ncells // 2
    out = {'0H'  : '0' * ncells,
           '1H-p': 'p' + '0' * (ncells - 1),
           '1H-d': 'd' + '0' * (ncells - 1)}
    for k in range(km + 1):
        st = ['0'] * ncells
        st[k] = 'd'
        st[0] = 'b' if k == 0 else 'p'
        out[f'2H-os-d{k}'] = ''.join(st)
    for k in range(1, km + 1):
        st = ['0'] * ncells
        st[0] = st[k] = 'p'
        out[f'2H-ss-d{k}'] = ''.join(st)
    return out


def bond_lengths_minimage(apos, bonds, Lx):
    """Bond lengths [A] with minimum-image convention along x (seam bonds corrected)."""
    d = apos[np.asarray(bonds)[:, 1]] - apos[np.asarray(bonds)[:, 0]]
    d[:, 0] -= np.round(d[:, 0] / Lx) * Lx
    return np.linalg.norm(d, axis=1)


def check_degrees(atoms, name='', deg_heavy=(2, 3)):
    """Coordination sanity: H degree 1, heavy atoms degree 2 (edge) or 3 (interior).
    deg_heavy=(2,3,4) allows sp3 edge sites (CH2 switch in carbon ribbons).

    Raises AssertionError on violation (fail-loud topology check).
    """
    deg = np.zeros(atoms.natoms, dtype=int)
    for i, j in atoms.bonds:
        deg[i] += 1
        deg[j] += 1
    bad = [(i, e, int(deg[i])) for i, e in enumerate(atoms.enames)
           if (e == 'H' and deg[i] != 1) or (e != 'H' and deg[i] not in deg_heavy)]
    if bad:
        raise AssertionError(f"{name}: bad coordination {bad}")
    n_edge = int(np.sum(deg == 2))
    print(f"  degree check OK: {n_edge} edge atoms (deg 2), {atoms.natoms - n_edge} interior (deg 3)")


def junction_site_atoms(atoms, lvs=None, pbc_y=False):
    """Indices of the N/O edge atoms forming the junction interface(s).

    Finds the distinct N/O edge-row y-levels and returns those of the rows
    nearest the inter-ribbon gap: for pbc_y=False the two middle rows (one
    interface); for pbc_y=True all four rows are junction rows anyway.
    Use for fixed_atoms pinning in relaxes — keeps d_DA (hence the cell
    comparison geometry) fixed while the rest relaxes; also prevents the
    SCC-meltdown drift of bare-N junction sites.
    """
    en = atoms.enames
    apos = np.asarray(atoms.apos)
    iNO = np.array([i for i, e in enumerate(en) if e in ('N', 'O')], dtype=int)
    ys = np.array(sorted(set(np.round(apos[iNO, 1], 1))))
    if pbc_y or len(ys) <= 2:
        rows = ys
    else:
        ym = 0.5 * (apos[:, 1].min() + apos[:, 1].max())
        rows = sorted(ys, key=lambda y: abs(y - ym))[:2]
    return sorted(i for i in iNO if any(abs(apos[i, 1] - y) < 0.2 for y in rows))


def junction_geometry_report(apos, hbonds, lvs):
    """Print D-H / H...A distances per junction site (from junction_bond_lengths)."""
    bls = junction_bond_lengths(apos, hbonds, lvs)
    print('  junction D-H/H..A:', '  '.join(f'{d1:.2f}/{d2:.2f}' for d1, d2 in bls))
    return bls


def save_xyz_lvs(path, atoms, lvs, name=''):
    """Write XYZ with the lattice vectors encoded in the comment line."""
    with open(path, 'w') as f:
        f.write(f"{atoms.natoms}\n{name} lvs=({lvs[0,0]:.4f},0,0) (0,{lvs[1,1]:.4f},0) (0,0,{lvs[2,2]:.4f})\n")
        for el, p in zip(atoms.enames, atoms.apos):
            f.write(f"{el:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
    return path


def read_xyz_frames(path):
    """Read a multi-frame xyz (e.g. DFTB+ geom.out.xyz) -> [(comment, enames, apos)]."""
    out = []
    with open(path) as f:
        lines = f.read().split('\n')
    i = 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        fr = lines[i + 2:i + 2 + n]
        out.append((lines[i + 1].strip(), [l.split()[0] for l in fr],
                    np.array([[float(x) for x in l.split()[1:4]] for l in fr])))
        i += n + 2
    return out


def relax_cell(atoms, lvs, fixed_atoms=None, nk=(8, 1, 1), k_shift=(0.5, 0.0, 0.0),
               cart_constraints=None, Temperature=300, Mixer=None,
               MaxScc=300, workdir='relax', chunks=3, verbose=True,
               sk_set=None, SCCTolerance=1e-5, params=None, patch_hsd=None):
    """run_pbc ionic relax with automatic chunked restart on SCC failure.

    DFTB+ relax occasionally poisons the SCF after a large LBFGS step (stale
    Hessian): SCC explodes at a perfectly reasonable geometry.  The cure is a
    fresh run from the last converged geom.out.xyz frame — a new Hessian (and
    charge guess) gets through.  Retries up to `chunks` times; raises the last
    RuntimeError if all chunks fail (fail loud, no silent fallback).

    Returns (E_eV, apos_relaxed).
    """
    from spammm.quantum.DFTB_utils import run_pbc
    enames, apos = list(atoms.enames), np.asarray(atoms.apos, dtype=float)
    last_err = None
    for c in range(chunks):
        wd = f'{workdir}_c{c}' if c else workdir
        try:
            E_ha, apos_r, _ = run_pbc(apos, enames, lvs, nk=nk, k_shift=k_shift, do_relax=True, fixed_atoms=fixed_atoms,
                                      workdir=wd, Temperature=Temperature, Mixer=Mixer, MaxScc=MaxScc,
                                      sk_set=sk_set, SCCTolerance=SCCTolerance, params=params, patch_hsd=patch_hsd,
                                      cart_constraints=cart_constraints)
            if c:
                print(f"  relax converged after restart chunk {c}", flush=True)
            return E_ha * HAU2EV, np.asarray(apos_r, dtype=float)
        except RuntimeError as e:
            last_err = e
            import os
            traj = os.path.join(wd, 'geom.out.xyz')
            frames = read_xyz_frames(traj) if os.path.exists(traj) else []
            if not frames:
                raise e
            enames, apos = frames[-1][1], frames[-1][2]
            if verbose:
                print(f"  ! relax SCC-failed ({frames[-1][0]}) -> restart chunk {c+1} from last frame", flush=True)
    raise last_err


def scan_junction_gap(width_chains, ncells, gaps, states=None, nk=(8, 2, 1), relax=False,
                      workdir='scanly', Temperature=300, verbose=True):
    """E(d_DA) lattice-y scan of the junction cell for a set of {p,d,0} states.

    For each state string and each gap d_DA the junction cell is rebuilt and a
    DFTB+ PBC single point (or ionic relax at fixed cell with relax=True) is
    evaluated via run_pbc.  'p'*2n vs 'd'*2n are mirror images — their energies
    must agree exactly (parity check, reported in the result dict).

    Args:
        gaps: array of N...N distances [A].
        states: list of state strings (default: all-p, alternating pd.., all-d).
        nk: (nkx, nky, 1) k-point grid; relax: ionic relax at fixed cell.
        workdir: prefix for per-point DFTB+ workdirs (<workdir>_<state>_<k>).

    Returns (res, fits, parity):
        res:    {state: np.array of energies [eV] (NaN where failed)}
        fits:   {state: (coef, d_opt, E_opt, sel)} — exact parabola through the
                3 lowest points (analytic interpolation, not least squares);
                coef are [a,b,c] of E = a d^2 + b d + c, curvature k = 2a.
        parity: {('p'*2n, 'd'*2n): max |dE| [eV]} when both states present.
    """
    from spammm.quantum.DFTB_utils import run_pbc
    n = ncells
    if states is None:
        states = ['p' * (2 * n), 'pd' * n, 'd' * (2 * n)]
    labels = {'p' * (2 * n): 'all-p', 'd' * (2 * n): 'all-d', 'pd' * n: 'alternating'}
    mode = 'relax' if relax else 'SP'
    res, fits = {}, {}
    for st in states:
        Es = np.full(len(gaps), np.nan)
        if verbose:
            print(f"\n=== w{width_chains} state '{st}' ({labels.get(st, st)}) {mode} scan, nk={nk[0]}x{nk[1]} ===")
        for k, d in enumerate(gaps):
            atoms, lvs, hbonds = build_ribbon_junction_cell(width_chains=width_chains, ncells=n, d_DA=float(d), state=st)
            wd = f'{workdir}_{labels.get(st, st)}_{k:02d}'
            E_ha, apos_out, forces = run_pbc(atoms.apos, atoms.enames, lvs, nk=nk,
                                           do_relax=relax, workdir=wd, Temperature=Temperature)
            Es[k] = E_ha * HAU2EV                       # run_pbc returns Hartree
            if verbose:
                print(f"  d_DA={d:.2f} A  Ly={lvs[1,1]:.2f}  E={Es[k]:.4f} eV  ({Es[k]/atoms.natoms:.3f} eV/atom)", flush=True)
        res[st] = Es
        ok = np.isfinite(Es)
        if ok.sum() >= 3:
            # exact parabola through the 3 lowest points (analytic interpolation)
            i3 = np.argsort(np.where(ok, Es, np.inf))[:3]
            i3 = i3[np.argsort(gaps[i3])]
            c = np.polyfit(gaps[i3], Es[i3], 2)            # passes through all 3 pts
            d_opt, E_opt = -c[1] / (2 * c[0]), np.polyval(c, -c[1] / (2 * c[0]))
            sel = np.zeros(len(Es), bool); sel[i3] = True
            fits[st] = (c, d_opt, E_opt, sel)
            if verbose:
                print(f"  -> grid min d_DA={gaps[np.nanargmin(Es)]:.2f} A;  3-pt parabola: d_opt={d_opt:.3f} A  E_min={E_opt:.4f} eV  (k={2*c[0]:.2f} eV/A^2)")
    parity = {}
    sp, sd = res.get('p' * (2 * n)), res.get('d' * (2 * n))
    if sp is not None and sd is not None:
        dE = np.nanmax(np.abs(sp - sd))
        parity[('p' * (2 * n), 'd' * (2 * n))] = dE
        if verbose:
            print(f"  PARITY all-p vs all-d (w{width_chains}): max |dE| = {dE*1000:.3f} meV  {'OK' if dE < 1e-4 else 'FAIL'}")
    return res, fits, parity
