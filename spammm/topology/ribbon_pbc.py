"""ribbon_pbc.py — periodic zigzag graphene nanoribbons and two-ribbon junction cells.

Reusable builders for the PBC ribbon/junction system (N-terminated zigzag edges,
periodic along the ribbon axis x; two stacked ribbons give periodic N...H-N
junction interfaces along y).  All builders return the `build_pbc_cell` contract
``(atoms, lvs, ...)`` so results feed `run_pbc` and `run_corner_scan_pbc` directly.

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

    def _build(pass_bot, pass_top):
        b = MoleculeEditorBackend(a_CC=a_CC)
        b.build_zigzag_ribbon(width_chains=width_chains, length_cells=ncells,
                              passivation_bottom=_decode_passivation(pass_bot),
                              passivation_top=_decode_passivation(pass_top),
                              bPeriodicX=True)
        b._sync_sys()
        return b

    if state is not None:
        st = [c for c in state.lower() if c in 'pd0']
        assert len(st) == 2 * ncells, f"state needs 2*ncells={2*ncells} chars from {{p,d,0}}, got '{state}'"
        si, sb = st[:ncells], st[ncells:]
        # internal gap: upper=B.bot, lower=A.top ; boundary gap: upper=A'.bot, lower=B.top
        bA = _build(['NH' if c == 'p' else 'N' for c in sb], ['NH' if c == 'd' else 'N' for c in si])
        bB = _build(['NH' if c == 'p' else 'N' for c in si], ['NH' if c == 'd' else 'N' for c in sb])
    else:
        bA = _build(bottom, bottom)
        bB = _build(top, top)
    nA = len(bA.sys.apos)
    be = MoleculeEditorBackend(a_CC=a_CC)
    be.combine_ribbons(bA, bB, L_Hb=0.0, shift_x=shift_x)
    be._sync_sys()

    apos = np.asarray(be.sys.apos, dtype=float).copy()
    enames = list(be.sys.enames)
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


def check_degrees(atoms, name=''):
    """Coordination sanity: H degree 1, heavy atoms degree 2 (edge) or 3 (interior).

    Raises AssertionError on violation (fail-loud topology check).
    """
    deg = np.zeros(atoms.natoms, dtype=int)
    for i, j in atoms.bonds:
        deg[i] += 1
        deg[j] += 1
    bad = [(i, e, int(deg[i])) for i, e in enumerate(atoms.enames)
           if (e == 'H' and deg[i] != 1) or (e != 'H' and deg[i] not in (2, 3))]
    if bad:
        raise AssertionError(f"{name}: bad coordination {bad}")
    n_edge = int(np.sum(deg == 2))
    print(f"  degree check OK: {n_edge} edge atoms (deg 2), {atoms.natoms - n_edge} interior (deg 3)")


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
