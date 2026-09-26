#!/usr/bin/env python3
"""testplot_ribbon.py — N-terminated zigzag graphene nanoribbons, periodic along x.

Thin driver over spammm/topology/ribbon_pbc.py:
  build_ribbon_cell          — single ribbon PBC cell (wrap bonds in atoms.bonds)
  build_ribbon_junction_cell — two stacked ribbons -> periodic N...H-N interfaces
  scan_junction_gap          — E(d_DA) lattice-y scan via DFTB+ (3-pt parabola)
Plotting in spammm/plotUtils.py: plot_ribbon_pbc_cell, plot_ribbon_junction_cell,
plot_gap_scan.

Usage:
    python tests/topology/testplot_ribbon.py --widths 4,6,8 --ncells 4 [--passivation N] [--dftb --nk 8]
    python tests/topology/testplot_ribbon.py --two --widths 4,6,8 --ncells 4
    python tests/topology/testplot_ribbon.py --scan-ly --two --widths 4 --ncells 2 --lmin 2.4 --lmax 3.6 --npts 13

Outputs: debug/ribbon/<name>.png + <name>.xyz   (xyz comment line carries lvs)
"""
import os
import sys
import shutil
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm.topology.ribbon_pbc import (build_ribbon_cell, build_ribbon_junction_cell,
                                       build_self_junction_cell, build_edge_switch_cell,
                                       check_degrees, junction_geometry_report,
                                       save_xyz_lvs, scan_junction_gap, HAU2EV)
from spammm.plotUtils import plot_ribbon_pbc_cell, plot_ribbon_junction_cell, plot_gap_scan

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'ribbon')


def _enum_mode(args):
    """(file tag, builder) for the enum family.  Default: self-junction stack
    (N<->NH, periodic y, H-bonded).  --vac: vacuum-y ribbon, --chem selects the
    switch pair (N: N->NH, C: CH->CH2, O: C-OH->C=O).  Builder signature is
    (width_chains, ncells, state) -> (atoms, lvs, hbonds)."""
    if args.vac:
        tagp = getattr(args, 'tagp', None) or ('xscanv' if getattr(args, 'xscan', False) else 'enumv')
        tag = f'{tagp}_{args.chem}'
        return tag, lambda wc, n, st: build_edge_switch_cell(wc, n, st, chem=args.chem, vac_y=args.vacy)
    return 'enumsj', lambda wc, n, st: build_self_junction_cell(width_chains=wc, ncells=n, d_DA=args.dda, state=st)


def _state_dict(args):
    """enum state dict: standard 7-state junction set, or the --xscan separation
    scan (xscan_state_strings: os-d0..d{n/2} + ss-d1..d{n/2} + refs)."""
    from spammm.topology.ribbon_pbc import junction_state_strings, xscan_state_strings
    return xscan_state_strings(args.ncells) if getattr(args, 'xscan', False) else junction_state_strings(args.ncells)


def _jdc_tag(args):
    """jdecomp cache/workdir tag matching the active enum mode."""
    if not args.vac:
        return 'jdecomp'
    tp = getattr(args, 'tagp', None) or ('xscanv' if getattr(args, 'xscan', False) else 'enumv')
    return f'jdecompv_{args.chem}' if tp == 'enumv' else f'jdecomp_{tp}_{args.chem}'


def _jdc_tag_for(tagp, chem):
    """jdecomp tag for an enum tag-prefix + chem (run_compare loops all chems)."""
    return f'jdecompv_{chem}' if tagp == 'enumv' else f'jdecomp_{tagp}_{chem}'


def run_junction(args):
    """Two-ribbon N...H-N junction cell (periodic x, optional periodic y stack)."""
    from collections import Counter
    for w in [int(x) for x in args.widths.split(',')]:
        name = f'jgnr_w{w}_n{args.ncells}_{args.bottom}-{args.top}'
        atoms, lvs, hbonds = build_ribbon_junction_cell(width_chains=w, ncells=args.ncells, d_DA=args.dda,
                                                        shift_x=args.shift_x, bottom=args.bottom, top=args.top,
                                                        state=args.state, pbc_y=not args.no_pbc_y)
        nint = sum(1 for h in hbonds if h.a_shift == 0)
        print(f"\n=== {name}: natoms={atoms.natoms} {dict(Counter(atoms.enames))}  cell={lvs[0,0]:.2f}x{lvs[1,1]:.2f} A  junctions={len(hbonds)} ({nint} internal + {len(hbonds)-nint} boundary) ===")
        check_degrees(atoms, name)
        junction_geometry_report(atoms.apos, hbonds, lvs)
        png = os.path.join(OUTDIR, f'{name}.png')
        plot_ribbon_junction_cell(atoms, lvs, hbonds, savepath=png, title=f'{name}: {atoms.natoms} atoms/cell, {len(hbonds)} junction sites, cell {lvs[0,0]:.2f}x{lvs[1,1]:.2f} A')
        xyz = os.path.join(OUTDIR, f'{name}.xyz')
        save_xyz_lvs(xyz, atoms, lvs, name)
        print(f"  wrote {xyz}\nREVIEW: {png}")
        if args.dftb:
            from spammm.quantum.DFTB_utils import run_pbc
            E_ha, apos_out, forces = run_pbc(atoms.apos, atoms.enames, lvs, nk=(args.nk, args.nky, 1), workdir=os.path.join(OUTDIR, 'dftb_' + name), Temperature=300)
            print(f"  DFTB+ PBC SP (nk={args.nk}x{args.nky}): E = {E_ha*HAU2EV:.4f} eV   ({E_ha*HAU2EV/atoms.natoms:.3f} eV/atom)")


def _enum_resume(wd, atoms):
    """If relaxed.xyz + detailed.out exist in wd, reuse them -> (E_eV, apos_r) or None."""
    import re
    rlx, det = os.path.join(wd, 'relaxed.xyz'), os.path.join(wd, 'detailed.out')
    if not (os.path.isfile(rlx) and os.path.isfile(det)):
        return None
    m = re.findall(r'Total energy:\s+(-?\d+\.\d+)\s+H', open(det).read())
    ls = open(rlx).read().strip().split('\n')
    if not m or len(ls) - 2 != atoms.natoms:
        return None
    apos_r = np.array([[float(v) for v in l.split()[1:4]] for l in ls[2:]])
    return float(m[-1]) * HAU2EV, apos_r


def _enum_one(task):
    """Relax one (width,state) + periodic pi-BO.  Module-level so it is picklable
    for the --jobs process pool; each state owns its workdir -> no collisions.
    Resumes from relaxed.xyz/detailed.out when present (relax is the expensive
    part; the DM SP is always redone, ~seconds)."""
    (w, lab, st, wc, n, vac, chem, vacy, dda, nk, nky, sk, outdir, tag, omp) = task
    os.environ['OMP_NUM_THREADS'] = str(omp)
    from spammm.topology.ribbon_pbc import (build_edge_switch_cell, build_self_junction_cell,
                                          junction_site_atoms, bond_lengths_minimage, relax_cell, save_xyz_lvs)
    from spammm.quantum import pi_bond_order as pbo
    builder = ((lambda wc_, n_, st_: build_edge_switch_cell(wc_, n_, st_, chem=chem, vac_y=vacy)) if vac
               else (lambda wc_, n_, st_: build_self_junction_cell(width_chains=wc_, ncells=n_, d_DA=dda, state=st_)))
    atoms, lvs, hbonds = builder(wc, n, st)
    wd = os.path.join(outdir, f'{tag}_r{w}', lab)
    MIXER = 'DIIS { Generations = 8 }'
    res = _enum_resume(wd, atoms)
    if res is not None:
        E, apos_r = res
        atoms.apos = apos_r
        print(f"  {lab:10s} w{w} '{st}': resume E_relax={E:10.4f} eV", flush=True)
    else:
        pins = None if vac else junction_site_atoms(atoms)
        tight = dict(Optimizer='LBFGS{  Memory = 20 }', MaxSteps=2000, GradElem=1e-5)
        try:
            E, apos_r = relax_cell(atoms, lvs, fixed_atoms=pins, nk=(nk, nky, 1),
                                   Temperature=300, Mixer=MIXER, MaxScc=1000, workdir=wd,
                                   sk_set=sk, SCCTolerance=1e-8, params=tight)
        except RuntimeError:
            return lab, np.nan, np.nan, None
        apos_r = np.asarray(apos_r, dtype=float)
        atoms.apos = apos_r
        save_xyz_lvs(os.path.join(wd, 'relaxed.xyz'), atoms, lvs, f'w{w}_{lab}')
    Lx = lvs[0, 0]
    bonds = np.asarray(atoms.bonds)
    bl = bond_lengths_minimage(apos_r, bonds, Lx)
    if bl.max() > 2.0:
        return lab, np.nan, np.nan, None
    try:
        bpi, info = pbo.pi_bond_orders_pbc(atoms.enames, apos_r, bonds, lvs,
                                           os.path.join(wd, 'dm'), nk=(nk, nky, 1), filling_temp=300,
                                           scctol=1e-8, maxscc=2000, mixer=MIXER, sk_set=sk, verbose=False)
        Eg = info['E_ha'] * HAU2EV
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"  {lab:10s} w{w} '{st}': pi-BO DM failed (energy kept)", flush=True)
        bpi, Eg = None, np.nan
    seam = np.abs(apos_r[bonds[:, 1], 0] - apos_r[bonds[:, 0], 0]) > 0.5 * Lx
    print(f"  {lab:10s} w{w} '{st}': E_relax={E:10.4f} eV  E_Gamma={Eg:10.4f} eV", flush=True)
    return lab, E, Eg, dict(apos=apos_r, bonds=bonds, bl=bl, bpi=bpi, seam=seam)


def run_enum(args):
    """Enumerate protonation states on the SELF-JUNCTION cell (one ribbon/cell,
    bound to itself across the y boundary — every edge is a junction edge).

    For each width x state (junction_state_strings: 0H/1H/2H same-side vs
    opposite-side, adjacent vs separated): build -> DFTB+ ionic relax (fixed
    tilted cell, nk 8x2) -> relaxed xyz + energy; then Gamma-point DFTBcore DM ->
    pi bond orders. Composite figure per width: rows=[bond lengths, pi-BO],
    cols=states, 2-cell replicas along x and y.
    Parity: 1H-p vs 1H-d are mirror images -> energies must agree.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from spammm.quantum.DFTB_utils import run_pbc
    from spammm.quantum import pi_bond_order as pbo
    from spammm.topology.ribbon_pbc import junction_state_strings, junction_site_atoms, bond_lengths_minimage, relax_cell
    from spammm.topology.hbond_utils import hbond_positions

    n = args.ncells
    states = _state_dict(args)
    tag, builder = _enum_mode(args)
    nky = 1 if args.vac else 2                     # vacuum y -> single k-point
    MIXER = 'DIIS { Generations = 8 }'      # DIIS >> Broyden for junction charge sloshing
    for w in [int(x) for x in args.widths.split(',')]:
        wc = 2 * (w + 1)                    # --widths = RING ROWS (w1=polyacene); width_chains = 2*rings + 2
        rows, Es, Es_g = [], {}, {}
        jobs = int(getattr(args, 'jobs', 1))
        print(f"\n########## {tag} w{w} (rings; {wc} atom chains): {len(states)} states, ncells={n}, d_DA={args.dda}, jobs={jobs} ##########")
        omp = max(1, (3 * 16 // 4) // jobs)                      # <=3/4 of 16 threads total
        tasks = [(w, lab, st.split('|')[0], wc, n, args.vac, args.chem, args.vacy, args.dda,
                  args.nk, nky, args.sk, OUTDIR, tag, omp) for lab, st in states.items()]
        if jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            res = list(ProcessPoolExecutor(jobs).map(_enum_one, tasks))
        else:
            res = [_enum_one(t) for t in tasks]
        for lab, E, Eg, pl in res:
            Es[lab], Es_g[lab] = E, Eg
            if pl is None or pl['bpi'] is None:
                print(f"  {lab:<10}: {'RELAX FAILED/DISTORTED' if pl is None else 'DM/pi-BO FAILED'} - energy kept, no map row")
                continue
            atoms, lvs, hbonds = builder(wc, n, states[lab].split('|')[0])
            atoms.apos = pl['apos']
            rows.append((lab, atoms, lvs, hbonds, pl['bl'], pl['bpi'], pl['seam']))

        E0 = np.nanmin(np.array([Es[l] for l in states]))
        print(f"\n  == w{w} energies (relaxed, nk={args.nk}) relative to ground state ==")
        for lab in states:
            print(f"    {lab:10s}: {Es[lab]:10.4f} eV   dE={1000*(Es[lab]-E0):+7.1f} meV")
        dp = abs(Es['1H-p'] - Es['1H-d'])
        print(f"    PARITY 1H-p vs 1H-d: |dE| = {dp*1000:.3f} meV  {'OK' if dp < 1e-4 else 'FAIL'}")

        png = os.path.join(OUTDIR, f'{tag}_r{w}_maps.png')
        plot_enum_maps(rows, Es, E0, w, args.dda, png, tile=0 if args.vac else 1,
                       ncell=1 if getattr(args, 'xscan', False) else 2)
        print(f"Saved: {png}\nREVIEW: {png}")
        np.savez(os.path.join(OUTDIR, f'{tag}_r{w}.npz'), labels=list(states.keys()),
                 E_relax=np.array([Es[l] for l in states]), E_gamma=np.array([Es_g[l] for l in states]),
                 lvs=lvs, width_chains=wc)
        # per-state analysis arrays (keyed by label) -> replotting via --enum-plot needs no recompute
        kw = {}
        for (lab, atoms, lvs, hbonds, bl, bpi, seam) in rows:
            kw[f'apos_{lab}'] = np.asarray(atoms.apos); kw[f'bonds_{lab}'] = np.asarray(atoms.bonds)
            kw[f'bl_{lab}'] = bl; kw[f'bpi_{lab}'] = bpi; kw[f'seam_{lab}'] = seam
        np.savez(os.path.join(OUTDIR, f'{tag}_r{w}_data.npz'), **kw)


def _heavy_mask(atoms):
    """True for bonds between heavy atoms (C-C/C-N); X-H junction bonds -> grey, out of colormap."""
    en = np.array([e.split('_')[0] for e in atoms.enames])
    b = np.asarray(atoms.bonds)
    return (en[b[:, 0]] != 'H') & (en[b[:, 1]] != 'H')


def _cc_mask(atoms):
    """True for C-C bonds only -> these set the colormap span (edge C-N may saturate)."""
    en = np.array([e.split('_')[0] for e in atoms.enames])
    b = np.asarray(atoms.bonds)
    return (en[b[:, 0]] == 'C') & (en[b[:, 1]] == 'C')


def _tile2(atoms, bvals, seam, lvs, axis=0, ncell=2):
    """Cell + replicas along lattice `axis` (ncell cells drawn).  Seam bonds
    (x-PBC crossings) are drawn as short bonds crossing the boundary: the
    LEFT-edge endpoint is replicated at +lvs[0] (min-image direction, not raw
    index order).  Extra wrap-atom copies are appended so each displayed cell
    shows its seam bonds.
    """
    from types import SimpleNamespace
    n = atoms.natoms
    bonds = np.asarray(atoms.bonds)
    apos = np.asarray(atoms.apos)
    dvec = lvs[axis].copy()                     # replica shift (lvs[1] may be tilted)
    hvy = _heavy_mask(atoms)
    nb, bs, hb = bonds[~seam], bonds[seam], hvy[~seam]
    xi, xj = apos[bs[:, 0], 0], apos[bs[:, 1], 0]
    hi = np.where(xi > xj, bs[:, 0], bs[:, 1])  # right-edge endpoint
    lo = np.where(xi > xj, bs[:, 1], bs[:, 0])  # left-edge endpoint -> replicated +Lx
    ulo, idx = np.unique(lo, return_inverse=True)
    if axis == 0:
        w0 = ncell * n + idx                                             # lo + lvs[0]
        bonds2 = np.concatenate([nb + k * n for k in range(ncell)] + [np.stack([hi, w0], 1)])
        bv2 = np.concatenate([bvals[~seam]] * ncell + [bvals[seam]])
        mk2 = np.concatenate([hb] * ncell + [hvy[seam]])
        apos2 = np.concatenate([apos + k * dvec for k in range(ncell)] + [apos[ulo] + lvs[0]])
        en2 = list(atoms.enames) * ncell + [atoms.enames[i] for i in ulo]
    else:
        w0 = 2 * n + idx                                                   # lo + lvs[0]           (cell 0)
        w1 = w0 + len(ulo)                                                 # lo + lvs[0] + lvs[1]  (cell 1)
        bonds2 = np.concatenate([nb, nb + n, np.stack([hi, w0], 1), np.stack([hi + n, w1], 1)])
        bv2 = np.concatenate([bvals[~seam], bvals[~seam], bvals[seam], bvals[seam]])
        mk2 = np.concatenate([hb, hb, hvy[seam], hvy[seam]])
        apos2 = np.concatenate([apos, apos + dvec, apos[ulo] + lvs[0], apos[ulo] + lvs[0] + dvec])
        en2 = list(atoms.enames) * 2 + [atoms.enames[i] for i in ulo] * 2
    ns = SimpleNamespace(bonds=bonds2, enames=en2)
    return ns, apos2, bv2, mk2


def _cell_boxes(ax, lvs, axis=0, ncell=2):
    """Faint thin dashed outlines of the displayed cells (origin k*lvs[axis])."""
    a0, a1 = lvs[0][:2], lvs[1][:2]
    for k in range(ncell):
        o = k * lvs[axis][:2]
        c = np.array([o, o + a0, o + a0 + a1, o + a1, o])
        ax.plot(c[:, 0], c[:, 1], '--', color=(0, 0, 0, 0.3), lw=0.8, zorder=1)


def plot_enum_maps(rows, Es, E0, w, dda, png, tile=0, ncell=2):
    """Composite map: rows=[bond length | pi-BO], cols=states; each panel shows
    `ncell` cells along lattice axis `tile` (0=x periodic, 1=y replica)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from spammm import elements
    from spammm.quantum import pi_bond_order as pbo
    from spammm.topology.hbond_utils import hbond_positions
    # norms from C-C (non-seam) bonds only: X-H grey, edge C-N colored but may saturate
    cc = [~r[6] & _cc_mask(r[1]) for r in rows]
    all_bl = np.concatenate([r[4][m] for r, m in zip(rows, cc)])
    all_bp = np.concatenate([r[5][m] for r, m in zip(rows, cc) if np.isfinite(r[5][m]).any()])
    Lcc = 1.42                                    # canonical aromatic C-C length = colormap center
    dl = 0.2 * np.abs(all_bl - Lcc).max()         # ~2.5x oversaturated span -> minor differences visible
    lnorm = mcolors.TwoSlopeNorm(vmin=Lcc - dl, vcenter=Lcc, vmax=Lcc + dl)
    db = 0.2 * np.abs(all_bp - 0.5).max()         # BO centered exactly at 0.5, symmetric
    bnorm = mcolors.TwoSlopeNorm(vmin=0.5 - db, vcenter=0.5, vmax=0.5 + db)
    nst = len(rows)
    Lx0, Ly0, dxl = rows[0][2][0, 0], rows[0][2][1, 1], abs(rows[0][2][1, 0])
    cw = ncell * Lx0 + 1.0 if tile == 0 else Lx0 + dxl + 1.0
    ch = Ly0 + 3.2 if tile == 0 else 2 * Ly0 + 3.2      # cell span + junction margin
    pw = 1.8                                           # panel width [in]
    fig, axs = plt.subplots(2, nst, figsize=(nst * pw + 1.5, 2 * pw * ch / cw + 1.0), squeeze=False)
    atoms0 = next((r[1] for r in rows if r[0] == '0H'), None)
    lcs = {}
    for i, (lab, atoms, lvs, hbonds, bl, bpi, seam) in enumerate(rows):
        Lx, Ly = lvs[0, 0], lvs[1, 1]
        for j, (bvals, norm) in enumerate([(bl, lnorm), (bpi, bnorm)]):
            ax = axs[j, i]
            ns, apos2, bv2, mk2 = _tile2(atoms, bvals, seam, lvs, axis=tile, ncell=ncell)
            lcs[j] = pbo.plot_bond_scalar_map(ax, ns, apos2, bv2, bonds=ns.bonds, cmap='coolwarm', norm=norm, mask=mk2, bAtoms=False, lws=6.0)
            en2 = [e.split('_')[0] for e in ns.enames]
            ax.scatter(apos2[:, 0], apos2[:, 1], s=6, c=[elements.ELEMENT_DICT[e][8] for e in en2], zorder=6, linewidths=0)
            if atoms0 is not None:
                _mark_sites(ax, atoms, atoms0, lvs, axis=tile, ncell=ncell)
            for hb in hbonds:
                for rep in ((-1, 0, 1) if tile == 1 else (0, 1)):
                    drep = rep * lvs[tile]
                    pD, pH, pA = hbond_positions(atoms.apos, hb, lvs)
                    ax.plot([pD[0] + drep[0], pH[0] + drep[0]], [pD[1] + drep[1], pH[1] + drep[1]], 'g-', lw=1.2, zorder=5)
                    ax.plot([pH[0] + drep[0], pA[0] + drep[0]], [pH[1] + drep[1], pA[1] + drep[1]], 'm--', lw=1.0, zorder=5)
            _cell_boxes(ax, lvs, axis=tile, ncell=ncell)
            ax.set_aspect('equal'); ax.axis('off'); ax.margins(0.02)
            if tile == 1:                        # show the boundary junctions crossing the outer edges
                ax.set_ylim(-1.6, 2.0 * Ly + 1.6)
            if j == 0:
                ax.set_title(f'{lab}\nE={Es[lab]-E0:+.3f} eV', fontsize=8)
    for j, (lc, sm) in lcs.items():
        cax = axs[j, -1].inset_axes([1.04, 0.15, 0.04, 0.7])
        fig.colorbar(sm, cax=cax)
        axs[j, 0].text(-0.06, 0.5, ['bond length [A]', 'pi bond order'][j], rotation=90,
                       va='center', ha='right', fontsize=9, transform=axs[j, 0].transAxes)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(f'r{w} ({w} ring rows, {2*(w+1)} chains): bond lengths & pi bond orders (relaxed, d_DA={dda})')
    fig.savefig(png, dpi=140, bbox_inches='tight')
    plt.close(fig)


def run_enum_diff(args):
    """Relative maps: dBL and d(pi-BO) of each protonated state vs relaxed 0H reference,
    loaded from enumsj_r<w>_data.npz (no recompute). Rows=[dBL, dBO], cols=states."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from spammm.quantum import pi_bond_order as pbo
    from spammm.topology.ribbon_pbc import junction_state_strings, bond_lengths_minimage
    from spammm import elements

    n = args.ncells
    states = _state_dict(args)
    tag, builder = _enum_mode(args)
    t_ax = 0 if args.vac else 1                        # replica axis: x for vacuum, y across the junction
    for w in [int(x) for x in args.widths.split(',')]:
        wc = 2 * (w + 1)
        npz = os.path.join(OUTDIR, f'{tag}_r{w}_data.npz')
        d = np.load(npz, allow_pickle=True)
        b0, bd0, bl0 = d['bpi_0H'], d['bonds_0H'], d['bl_0H']
        ref = {tuple(sorted(b)): i for i, b in enumerate(bd0)}          # sorted-pair -> 0H bond index
        labs = [s for s in states if f'bpi_{s}' in d.files]
        # matched differences (extra N-H bonds get NaN -> grey)
        dbl, dbp = {}, {}
        for s in labs:
            bds = d[f'bonds_{s}']
            m = np.array([tuple(sorted(b)) in ref for b in bds])
            ridx = np.array([ref[tuple(sorted(b))] if m[i] else 0 for i, b in enumerate(bds)])
            dbl[s] = np.where(m, d[f'bl_{s}'] - bl0[np.clip(ridx, 0, len(bl0) - 1)], np.nan)
            dbp[s] = np.where(m & np.isfinite(d[f'bpi_{s}']), d[f'bpi_{s}'] - b0[np.clip(ridx, 0, len(b0) - 1)], np.nan)
        all_l = np.concatenate([v[np.isfinite(v)] for v in dbl.values()])
        all_p = np.concatenate([v[np.isfinite(v)] for v in dbp.values()])
        ncl = 1 if getattr(args, 'xscan', False) else 2
        nl = mcolors.TwoSlopeNorm(vmin=-0.2 * np.abs(all_l).max(), vcenter=0, vmax=0.2 * np.abs(all_l).max())
        nb = mcolors.TwoSlopeNorm(vmin=-0.2 * np.abs(all_p).max(), vcenter=0, vmax=0.2 * np.abs(all_p).max())
        nst = len(labs)
        lvs = None
        for s in labs:                                                   # need cell dims for panel aspect
            atoms_t, lvs_t, _ = builder(wc, n, states[s].split('|')[0])
            lvs = lvs_t; break
        if t_ax == 1:
            ch = 2 * lvs[1, 1] + 3.2; cw = lvs[0, 0] + abs(lvs[1, 0]) + 1.0
        else:
            ch = lvs[1, 1] + 3.2; cw = ncl * lvs[0, 0] + 1.0
        pw = 1.8
        fig, axs = plt.subplots(2, nst, figsize=(nst * pw + 1.5, 2 * pw * ch / cw + 1.0), squeeze=False)
        lcs = {}
        atoms0, _, _ = builder(wc, n, states['0H'].split('|')[0])
        atoms0.apos = d['apos_0H']
        for i, s in enumerate(labs):
            atoms, lvs, hbonds = builder(wc, n, states[s].split('|')[0])
            atoms.apos = d[f'apos_{s}']
            seam = d[f'seam_{s}']
            for j, (dv, norm) in enumerate([(dbl[s], nl), (dbp[s], nb)]):
                ax = axs[j, i]
                ns, apos2, bv2, mk2 = _tile2(atoms, dv, seam, lvs, axis=t_ax, ncell=ncl)
                lcs[j] = pbo.plot_bond_scalar_map(ax, ns, apos2, bv2, bonds=ns.bonds, cmap='coolwarm', norm=norm, mask=mk2 & np.isfinite(bv2), bAtoms=False, lws=6.0)
                en2 = [e.split('_')[0] for e in ns.enames]
                ax.scatter(apos2[:, 0], apos2[:, 1], s=6, c=[elements.ELEMENT_DICT[e][8] for e in en2], zorder=6, linewidths=0)
                _mark_sites(ax, atoms, atoms0, lvs, axis=t_ax, ncell=ncl)
                _cell_boxes(ax, lvs, axis=t_ax, ncell=ncl)
                if t_ax == 1:
                    ax.set_ylim(-1.6, 2.0 * lvs[1, 1] + 1.6)
                if j == 0:
                    ax.set_title(s, fontsize=8)
        for j, (lc, sm) in lcs.items():
            cax = axs[j, -1].inset_axes([1.04, 0.15, 0.04, 0.7])
            fig.colorbar(sm, cax=cax)
            axs[j, 0].text(-0.06, 0.5, ['d(bond length) vs 0H [A]', 'd(pi BO) vs 0H'][j], rotation=90,
                           va='center', ha='right', fontsize=9, transform=axs[j, 0].transAxes)
        fig.suptitle(f'{tag} r{w} ({w} ring rows, {wc} chains): CHANGE vs relaxed 0H (red=longer/higher BO, blue=shorter/lower)')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        png = os.path.join(OUTDIR, f'{tag}_r{w}_diffmaps.png')
        fig.savefig(png, dpi=140, bbox_inches='tight'); plt.close(fig)
        print(f'  r{w}: max|dbl|={np.abs(all_l).max()*1000:.1f} mA  max|dbpi|={np.abs(all_p).max():.3f}\nREVIEW: {png}')


def _nonlin_core(d, builder, wc, n, states, labs2):
    """Nonlinear superposition residuals for the 4 two-site states.

    db/dl[s]  : state-vs-0H differences in 0H-backbone (bd0) order.
    Rb/Rl[s]  : residual R = X(2H) - X(1H_A) - X(1H_B) in bd0 order.
    Rbf/Rlf[s]: same residuals expanded onto the STATE's own bond list
                (NaN on non-backbone bonds) — ready for _tile2/map plotting.
    Backbone bonds are matched by endpoint POSITIONS (atom indices shift for
    O chem where the removed H renumbers atoms)."""
    bd0, b0, bl0 = d['bonds_0H'], d['bpi_0H'], d['bl_0H']
    atoms0, lvs, _ = builder(wc, n, states['0H'].split('|')[0])
    en0 = np.array([e.split('_')[0] for e in atoms0.enames])
    mH0 = (en0[bd0[:, 0]] != 'H') & (en0[bd0[:, 1]] != 'H')          # backbone only (0H may carry passivation C-H)
    bd0, b0, bl0 = bd0[mH0], b0[mH0], bl0[mH0]
    hvy0 = np.where(en0 != 'H')[0]
    ap0 = np.asarray(atoms0.apos)[hvy0]

    Lx, ax_ = lvs[0, 0], lvs[0, 0] / n
    dv = atoms0.apos[bd0[:, 1]] - atoms0.apos[bd0[:, 0]]
    dv[:, 0] -= np.round(dv[:, 0] / Lx) * Lx                          # min-image seam
    mid = atoms0.apos[bd0[:, 0]] + 0.5 * dv; mid[:, 0] %= Lx

    perms = {0: np.arange(len(bd0))}
    def perm(s):
        if s not in perms:
            P = np.empty(len(bd0), int)
            for j in range(len(bd0)):
                tgt = mid[j].copy(); tgt[0] = (tgt[0] - s * ax_) % Lx
                P[j] = np.argmin(np.linalg.norm(mid - tgt, axis=1))
            perms[s] = P
        return perms[s]

    def align(s):
        """-> (mH: backbone mask on state bond list, pm: bd0 row -> state backbone pos)."""
        at, _, _ = builder(wc, n, states[s].split('|')[0])
        en = np.array([e.split('_')[0] for e in at.enames])
        hvy = np.where(en != 'H')[0]
        M = np.argmin(np.linalg.norm(np.asarray(at.apos)[hvy][:, None, :2] - ap0[None, :, :2], axis=2), axis=1)
        bd = d[f'bonds_{s}']
        mH = (en[bd[:, 0]] != 'H') & (en[bd[:, 1]] != 'H')
        ks = np.searchsorted(hvy, bd[mH])                            # full idx -> heavy-list position
        pos = {tuple(sorted((int(hvy0[M[a]]), int(hvy0[M[b]])))): k for k, (a, b) in enumerate(ks)}
        pm = np.array([pos[tuple(sorted(b))] for b in bd0])          # bd0 row -> state's backbone position
        return mH, pm

    labs1 = ['1H-p', '1H-d']
    db, dl, mHs, pms = {}, {}, {}, {}
    for s in labs1 + labs2:
        mHs[s], pms[s] = align(s)
        db[s] = d[f'bpi_{s}'][mHs[s]][pms[s]] - b0
        dl[s] = d[f'bl_{s}'][mHs[s]][pms[s]] - bl0
    def exp_for(s):
        """Expected superposition: translate each single-site response to the
        sites encoded in the state string ('p'/'b'->bottom, 'd'/'b'->top)."""
        eb, el = np.zeros(len(bd0)), np.zeros(len(bd0))
        for i, c in enumerate(states[s].split('|')[0].lower()):
            P = perm(i)
            if c in 'pb':
                eb += db['1H-p'][P]; el += dl['1H-p'][P]
            if c in 'db':
                eb += db['1H-d'][P]; el += dl['1H-d'][P]
        return eb, el
    exp_b, exp_l = {}, {}
    for s in labs2:
        exp_b[s], exp_l[s] = exp_for(s)
    Rb, Rl, Rbf, Rlf = {}, {}, {}, {}
    for s in labs2:
        Rb[s], Rl[s] = db[s] - exp_b[s], dl[s] - exp_l[s]
        mH_idx = np.where(mHs[s])[0]
        Rbf[s] = np.full(len(d[f'bonds_{s}']), np.nan); Rbf[s][mH_idx[pms[s]]] = Rb[s]
        Rlf[s] = np.full(len(d[f'bonds_{s}']), np.nan); Rlf[s][mH_idx[pms[s]]] = Rl[s]
    return dict(atoms0=atoms0, lvs=lvs, db=db, dl=dl, Rb=Rb, Rl=Rl, Rbf=Rbf, Rlf=Rlf)


def run_nonlin(args):
    """NONLINEAR residual maps: R = X(2H) - X(1H_A) - X(1H_B) for X in {bpi, bl}.

    The single-site response at junction site s is the site-0 response translated
    by s*ax (exact symmetry of the ideal cell; permutation built on min-image bond
    midpoints).  R = where the two perturbations actually interact in the density
    matrix — the quantity whose energy counterpart is J_noSCC.  Linear
    superposition (R~0) means no through-bond coupling however large each
    single-site response is.  Loads <tag>_r<w>_data.npz; no recompute.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from spammm.quantum import pi_bond_order as pbo
    from spammm.topology.ribbon_pbc import junction_state_strings
    from spammm import elements

    n = args.ncells
    states = _state_dict(args)
    tag, builder = _enum_mode(args)
    t_ax = 0 if args.vac else 1
    labs2 = [l for l in states if l.startswith('2H')]
    for w in [int(x) for x in args.widths.split(',')]:
        wc = 2 * (w + 1)
        d = np.load(os.path.join(OUTDIR, f'{tag}_r{w}_data.npz'), allow_pickle=True)
        nc = _nonlin_core(d, builder, wc, n, states, labs2)
        atoms0, lvs = nc['atoms0'], nc['lvs']
        db, Rb, Rl, Rbf, Rlf = nc['db'], nc['Rb'], nc['Rl'], nc['Rbf'], nc['Rlf']
        # J_noSCC for correlation, if jdecomp cache exists
        Jn = {}
        jdc = os.path.join(OUTDIR, f'{_jdc_tag(args)}_r{w}.npz')
        if os.path.exists(jdc):
            z = np.load(jdc)
            Jn = dict(zip([str(l) for l in z['labels']], z['J_noscc'] * 1000))
        print(f'--- {tag} r{w}: nonlinear residual (DM superposition error) ---')
        for s in labs2:
            j = f"  J_noSCC={Jn[s]:+.1f} meV" if s in Jn else ''
            print(f'  {s:<10} max|Dbpi|={np.abs(db[s]).max():.3f}  max|R_bpi|={np.abs(Rb[s]).max():.4f}  '
                  f'rms|R|={np.sqrt((Rb[s]**2).mean()):.4f}  max|R_bl|={np.abs(Rl[s]).max()*1000:.1f} mA{j}')

        nb = mcolors.TwoSlopeNorm(vmin=-0.2 * max(np.abs(Rb[s]).max() for s in labs2), vcenter=0,
                                  vmax=0.2 * max(np.abs(Rb[s]).max() for s in labs2))
        nl = mcolors.TwoSlopeNorm(vmin=-0.2 * max(np.abs(Rl[s]).max() for s in labs2), vcenter=0,
                                  vmax=0.2 * max(np.abs(Rl[s]).max() for s in labs2))
        ncl = 1 if getattr(args, 'xscan', False) else 2
        ch = (2 * lvs[1, 1] + 3.2) if t_ax == 1 else (lvs[1, 1] + 3.2)
        cw = (lvs[0, 0] + abs(lvs[1, 0]) + 1.0) if t_ax == 1 else (ncl * lvs[0, 0] + 1.0)
        fig, axs = plt.subplots(2, len(labs2), figsize=(len(labs2) * 1.8 + 1.5, 2 * 1.8 * ch / cw + 1.0), squeeze=False)
        lcs = {}
        for i, s in enumerate(labs2):
            atoms, lvs_s, _ = builder(wc, n, states[s].split('|')[0])
            atoms.apos = d[f'apos_{s}']
            seam = d[f'seam_{s}']
            for j, (rv, norm) in enumerate([(Rlf[s], nl), (Rbf[s], nb)]):
                ax = axs[j, i]
                ns, apos2, bv2, mk2 = _tile2(atoms, rv, seam, lvs, axis=t_ax, ncell=ncl)
                lcs[j] = pbo.plot_bond_scalar_map(ax, ns, apos2, bv2, bonds=ns.bonds, cmap='coolwarm',
                                                  norm=norm, mask=np.isfinite(bv2), bAtoms=False, lws=6.0)
                en2 = [e.split('_')[0] for e in ns.enames]
                ax.scatter(apos2[:, 0], apos2[:, 1], s=6, c=[elements.ELEMENT_DICT[e][8] for e in en2], zorder=6, linewidths=0)
                _mark_sites(ax, atoms, atoms0, lvs, axis=t_ax, ncell=ncl)
                _cell_boxes(ax, lvs, axis=t_ax, ncell=ncl)
                if t_ax == 1:
                    ax.set_ylim(-1.6, 2.0 * lvs[1, 1] + 1.6)
                if j == 0:
                    ax.set_title(f'{s}\nmax|R_bpi|={np.abs(Rb[s]).max():.3f}', fontsize=8)
        for j, (lc, sm) in lcs.items():
            cax = axs[j, -1].inset_axes([1.04, 0.15, 0.04, 0.7])
            fig.colorbar(sm, cax=cax)
            axs[j, 0].text(-0.06, 0.5, ['nonlin d(bond len) [A]', 'nonlin d(pi BO)'][j], rotation=90,
                           va='center', ha='right', fontsize=9, transform=axs[j, 0].transAxes)
        fig.suptitle(f'{tag} r{w}: NONLINEAR residual X(2H)-X(1H_A)-X(1H_B) = where the two sites interact')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        png = os.path.join(OUTDIR, f'{tag}_r{w}_nonlin.png')
        fig.savefig(png, dpi=140, bbox_inches='tight'); plt.close(fig)
        print(f'REVIEW: {png}')


def run_compare(args):
    """Cross-chemistry comparison at each width in --widths (vacuum cells).

    cmp_BL_r<w>.png / cmp_BO_r<w>.png : rows = chem x {abs | d vs 0H | nonlin},
    cols = states, SHARED color scale across all rows -> magnitudes compare
    directly.  Nonlin rows (R = X(2H)-X(1H_A)-X(1H_B)) fill only the 2H columns.
    cmp_J_r<w>.png : rows = chem; J split noSCC / electrostatic / elastic / total.
    Needs enumv_{C,N,O}_r<w>_data.npz (+ jdecompv_*_r<w>.npz for the J figure).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from spammm.quantum import pi_bond_order as pbo
    from spammm.topology.ribbon_pbc import junction_state_strings, build_edge_switch_cell
    from spammm import elements

    n = args.ncells
    states = _state_dict(args)
    labs = list(states)
    tagp = getattr(args, 'tagp', None) or ('xscanv' if getattr(args, 'xscan', False) else 'enumv')
    pfx = '' if tagp == 'enumv' else f'{tagp}_'                # don't clobber enumv cmp_*
    chems = ['C', 'N', 'O']
    cnames = {'C': 'CH->CH2', 'N': 'N->NH', 'O': 'C-OH->C=O'}
    for w in [int(x) for x in args.widths.split(',')]:
        wc = 2 * (w + 1)
        DD, lvs0 = {}, None
        for c in chems:
            dpf = os.path.join(OUTDIR, f'{tagp}_{c}_r{w}_data.npz')
            if not os.path.exists(dpf):                      # partial sets (e.g. xscanz = O only)
                print(f'  r{w}: skip chem {c} (no {dpf})')
                continue
            d = np.load(dpf, allow_pickle=True)
            bd0, b0, bl0 = d['bonds_0H'], d['bpi_0H'], d['bl_0H']
            ref = {tuple(sorted(b)): i for i, b in enumerate(bd0)}
            a0, lv, _ = build_edge_switch_cell(wc, n, states['0H'].split('|')[0], chem=c, vac_y=args.vacy)
            a0.apos = d['apos_0H']; lvs0 = lv
            per = {'atoms0': a0}
            for s in labs:
                bds = d[f'bonds_{s}']
                m = np.array([tuple(sorted(b)) in ref for b in bds])
                ridx = np.array([ref[tuple(sorted(b))] if m[i] else 0 for i, b in enumerate(bds)])
                a, _, _ = build_edge_switch_cell(wc, n, states[s].split('|')[0], chem=c, vac_y=args.vacy)
                a.apos = d[f'apos_{s}']
                per[s] = dict(atoms=a, seam=d[f'seam_{s}'], bl=d[f'bl_{s}'], bpi=d[f'bpi_{s}'],
                              dbl=np.where(m, d[f'bl_{s}'] - bl0[np.clip(ridx, 0, len(bl0) - 1)], np.nan),
                              dbp=np.where(m & np.isfinite(d[f'bpi_{s}']), d[f'bpi_{s}'] - b0[np.clip(ridx, 0, len(b0) - 1)], np.nan))
            per['data'] = d
            DD[c] = per
        chems = [c for c in chems if c in DD]                # partial sets keep only loaded chems
        if not chems:
            continue
        # shared norms across ALL chems so magnitudes compare directly;
        # range from C-C bonds only (X-H grey, C-N/C-O may saturate), ~2.5x oversaturated
        en0 = np.array([e.split('_')[0] for e in DD[chems[0]]['atoms0'].enames])
        hv = np.array([e.split('_')[0] != 'H' for e in DD[chems[0]]['atoms0'].enames])
        SAT = 0.2                                              # 1/2.5 of max -> 2.5x oversaturation
        def ccm(c, s):                                         # C-C bonds of that state
            return _cc_mask(DD[c][s]['atoms'])
        allbl = np.concatenate([DD[c][s]['bl'][ccm(c, s)] for c in chems for s in labs])
        allbp = np.concatenate([v[np.isfinite(v)] for c in chems for s in labs for v in [DD[c][s]['bpi'][ccm(c, s)]]])
        alldl = np.concatenate([v[np.isfinite(v)] for c in chems for s in labs for v in [DD[c][s]['dbl'][ccm(c, s)]]])
        alldb = np.concatenate([v[np.isfinite(v)] for c in chems for s in labs for v in [DD[c][s]['dbp'][ccm(c, s)]]])
        Lcc = 1.42
        nl_abs = mcolors.TwoSlopeNorm(vmin=Lcc - SAT * np.abs(allbl - Lcc).max(), vcenter=Lcc, vmax=Lcc + SAT * np.abs(allbl - Lcc).max())
        nb_abs = mcolors.TwoSlopeNorm(vmin=0.5 - SAT * np.abs(allbp - 0.5).max(), vcenter=0.5, vmax=0.5 + SAT * np.abs(allbp - 0.5).max())
        nl_dif = mcolors.TwoSlopeNorm(vmin=-SAT * np.abs(alldl).max(), vcenter=0, vmax=SAT * np.abs(alldl).max())
        nb_dif = mcolors.TwoSlopeNorm(vmin=-SAT * np.abs(alldb).max(), vcenter=0, vmax=SAT * np.abs(alldb).max())
        # nonlinear residuals per chem (on each state's own bond list)
        labs2 = [l for l in states if l.startswith('2H')]
        NC = {c: _nonlin_core(DD[c]['data'], lambda w_, n_, s_, c_=c: build_edge_switch_cell(w_, n_, s_, chem=c_, vac_y=args.vacy), wc, n, states, labs2) for c in chems}
        allrb = np.concatenate([v[np.isfinite(v)] for c in chems for v in NC[c]['Rbf'].values()])
        allrl = np.concatenate([v[np.isfinite(v)] for c in chems for v in NC[c]['Rlf'].values()])
        nb_nl = mcolors.TwoSlopeNorm(vmin=-SAT * np.abs(allrb).max(), vcenter=0, vmax=SAT * np.abs(allrb).max())
        nl_nl = mcolors.TwoSlopeNorm(vmin=-SAT * np.abs(allrl).max(), vcenter=0, vmax=SAT * np.abs(allrl).max())
        k2d = {'bl': 'dbl', 'bpi': 'dbp'}
        nlkey = {'bl': 'Rlf', 'bpi': 'Rbf'}
        ncl = 1 if getattr(args, 'xscan', False) else 2        # xscan: 1 cell per panel (12 cols)
        ch, cw = lvs0[1, 1] + 3.2, ncl * lvs0[0, 0] + 1.0

        def panel(ax, c, ir, s, nm):
            """One map panel: ir = 0 abs | 1 d-vs-0H | 2 nonlin (2H cols only)."""
            pr = DD[c][s]
            if ir == 2:
                bv = NC[c][nlkey_][s] if s in labs2 else np.full(len(pr['atoms'].bonds), np.nan)
            else:
                bv = pr[key_ if ir == 0 else k2d[key_]]
            ns, apos2, bv2, mk2 = _tile2(pr['atoms'], bv, pr['seam'], lvs0, axis=0, ncell=ncl)
            lc = pbo.plot_bond_scalar_map(ax, ns, apos2, bv2, bonds=ns.bonds, cmap='coolwarm', norm=nm, mask=mk2 & np.isfinite(bv2), bAtoms=False, lws=6.0)
            en2 = [e.split('_')[0] for e in ns.enames]
            ax.scatter(apos2[:, 0], apos2[:, 1], s=5, c=[elements.ELEMENT_DICT[e][8] for e in en2], zorder=6, linewidths=0)
            _mark_sites(ax, pr['atoms'], DD[c]['atoms0'], lvs0, axis=0, ncell=ncl)
            _cell_boxes(ax, lvs0, axis=0, ncell=ncl)
            ax.set_aspect('equal'); ax.axis('off'); ax.margins(0.02)
            return lc

        def finish(fig, axs, lcs, rnames, title, png):
            for j, (lc, sm) in lcs.items():
                cax = axs[j, -1].inset_axes([1.04, 0.15, 0.04, 0.7])
                fig.colorbar(sm, cax=cax)
                axs[j, 0].text(-0.06, 0.5, rnames[j], rotation=90, va='center', ha='right', fontsize=9, transform=axs[j, 0].transAxes)
            for i, s in enumerate(labs):
                axs[0, i].set_title(s, fontsize=8)
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            p = os.path.join(OUTDIR, png)
            fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
            print(f'REVIEW: {p}')

        for key_, na, nd, nn, ylab, png in [('bl', nl_abs, nl_dif, nl_nl, 'bond length [A]', 'cmp_BL'),
                                           ('bpi', nb_abs, nb_dif, nb_nl, 'pi bond order', 'cmp_BO')]:
            nlkey_ = nlkey[key_]
            norms = [na, nd, nn]; rtypes = ['abs', 'd vs 0H', 'nonlin']
            # (a) per-chem figure: 3 rows {abs, d vs 0H, nonlin} with CHEM-OWN norms
            for c in chems:
                dc = DD[c]
                va = np.concatenate([dc[s][key_][ccm(c, s)] for s in labs])
                vd = np.concatenate([v[np.isfinite(v)] for s in labs for v in [dc[s][k2d[key_]][ccm(c, s)]]])
                vn = np.concatenate([v[np.isfinite(v)] for v in NC[c][nlkey_].values()])
                cen = Lcc if key_ == 'bl' else 0.5
                nc_norms = [mcolors.TwoSlopeNorm(vmin=cen - SAT * np.abs(va - cen).max(), vcenter=cen, vmax=cen + SAT * np.abs(va - cen).max()),
                            mcolors.TwoSlopeNorm(vmin=-SAT * np.abs(vd).max(), vcenter=0, vmax=SAT * np.abs(vd).max()),
                            mcolors.TwoSlopeNorm(vmin=-SAT * np.abs(vn).max(), vcenter=0, vmax=SAT * np.abs(vn).max())]
                fig, axs = plt.subplots(3, len(labs), figsize=(len(labs) * 1.8 + 1.5, 3 * 1.8 * ch / cw + 1.0), squeeze=False)
                lcs = {}
                for ir, nm in enumerate(nc_norms):
                    for i, s in enumerate(labs):
                        lcs[ir] = panel(axs[ir, i], c, ir, s, nm)
                finish(fig, axs, lcs, [f'{cnames[c]} {t}' for t in rtypes],
                       f'r{w} vac {cnames[c]}: {ylab} — abs / d vs 0H / nonlin residual', f'{pfx}{png}_{c}_r{w}.png')
            # (b) per-type figures: 3 rows = chems, SHARED norms across chems
            for ir, (rt, nm) in enumerate(zip(rtypes, norms)):
                fig, axs = plt.subplots(3, len(labs), figsize=(len(labs) * 1.8 + 1.5, 3 * 1.8 * ch / cw + 1.0), squeeze=False)
                lcs = {}
                for ic, c in enumerate(chems):
                    for i, s in enumerate(labs):
                        lcs[ic] = panel(axs[ic, i], c, ir, s, nm)
                finish(fig, axs, lcs, [cnames[c] for c in chems],
                       f'r{w} vac: {ylab} [{rt}] — rows: CH->CH2 / N->NH / C-OH->C=O; shared scale', f'{pfx}{png}-{rt.replace(" ", "")}_r{w}.png')
        # (c) combined figure: 18 rows = {BL | BO} x {abs | d0H | nonlin} x {C,N,O}, cols=states
        nrm = {('bl', 0): nl_abs, ('bl', 1): nl_dif, ('bl', 2): nl_nl,
               ('bpi', 0): nb_abs, ('bpi', 1): nb_dif, ('bpi', 2): nb_nl}
        qnames = {'bl': 'bond length [A]', 'bpi': 'pi bond order'}
        rtnames = {0: 'abs', 1: 'd vs 0H', 2: 'nonlin'}
        rspec = [(q, rt, c) for q, rt in (('bl', 0), ('bl', 1), ('bl', 2),
                                          ('bpi', 0), ('bpi', 1), ('bpi', 2)) for c in chems]
        pw = 3.4
        fig, axs = plt.subplots(len(rspec), len(labs), figsize=(len(labs) * pw + 2.0, len(rspec) * pw * ch / cw + 1.2), squeeze=False)
        lcb = {}
        for ir, (q, rt, c) in enumerate(rspec):
            key_ = q; nlkey_ = nlkey[q]                        # read by panel()
            for i, s in enumerate(labs):
                lcb[ir // len(chems)] = panel(axs[ir, i], c, rt, s, nrm[(q, rt)])
            axs[ir, 0].text(-0.05, 0.5, f'{cnames[c]}\n{qnames[q]} {rtnames[rt]}',
                            rotation=90, va='center', ha='right', fontsize=8, transform=axs[ir, 0].transAxes)
        for i, s in enumerate(labs):
            axs[0, i].set_title(s, fontsize=9)
            axs[-1, i].text(0.5, -0.4, s, fontsize=9, ha='center', va='top', transform=axs[-1, i].transAxes)
        fig.suptitle(f'r{w} vac: BL & pi-BO — row blocks: BL abs | BL d0H | BL nonlin | BO abs | BO d0H | BO nonlin;  rows within block: CH->CH2 / N->NH / C-OH->C=O;  shared scale per block')
        fig.tight_layout(rect=[0.03, 0, 0.985, 0.94])
        for ib in range(6):                                  # one colorbar per len(chems)-row block
            b0 = axs[ib * len(chems), -1].get_position()
            b2 = axs[(ib + 1) * len(chems) - 1, -1].get_position()
            fig.colorbar(lcb[ib][1], cax=fig.add_axes([b0.x1 + 0.004, b2.y0, 0.0035, b0.y1 - b2.y0]))
        p = os.path.join(OUTDIR, f'{pfx}cmp_BLBO_r{w}.png')
        fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
        print(f'REVIEW: {p}')
        # --- J breakdown per chem ---
        stl = labs2
        x = np.arange(len(stl))
        fig, axs = plt.subplots(len(chems), 1, figsize=(9, 2.7 * len(chems)), sharex=True, squeeze=False)
        axs = axs[:, 0]
        for ic, c in enumerate(chems):
            ax = axs[ic]
            e = np.load(os.path.join(OUTDIR, f'{tagp}_{c}_r{w}.npz'))
            Es = dict(zip([str(l) for l in e['labels']], e['E_relax']))
            E1 = 0.5 * (Es['1H-p'] + Es['1H-d'])
            Jr = np.array([1000 * (Es[l] - 2 * E1 + Es['0H']) for l in stl])          # meV
            jf = os.path.join(OUTDIR, f"{_jdc_tag_for(tagp, c)}_r{w}.npz")
            if not os.path.exists(jf):
                ax.plot(np.arange(len(stl)), Jr, 'ko-', label='J total (relaxed)')
                ax.set_ylabel(f'{cnames[c]}\nJ [meV]', fontsize=9)
                continue
            z = np.load(jf)
            Jri = dict(zip([str(l) for l in z['labels']], z['J_rigid']))
            Jno = dict(zip([str(l) for l in z['labels']], z['J_noscc']))
            Jn = np.array([1000 * Jno[l] for l in stl]); Jes = np.array([1000 * (Jri[l] - Jno[l]) for l in stl]); Jel = Jr - np.array([1000 * Jri[l] for l in stl])
            ax.plot(x, Jr, 'ko-', label='J total (relaxed)')
            ax.plot(x, Jn, 's--', color='0.45', mfc='none', label='noSCC (pi/band + rep)')
            ax.plot(x, Jes, 'b^-', mfc='none', label='electrostatic (SCC charge resp.)')
            ax.plot(x, Jel, 'gv-', mfc='none', label='elastic (relaxation)')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel(f'{cnames[c]}\nJ [meV]', fontsize=9)
            if ic == 0:
                ax.legend(fontsize=8, ncol=2)
        axs[-1].set_xticks(x, stl)
        fig.suptitle(f'r{w} vac: J = E(2H)-2E(1H)+E(0H) breakdown — same scale comparison of switch chemistries')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = os.path.join(OUTDIR, f'{pfx}cmp_J_r{w}.png')
        fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
        print(f'REVIEW: {p}')


def run_enum_plot(args):
    """Replot enum maps WITHOUT recomputation: loads enumsj_r<w>_data.npz; if absent,
    recovers rows from saved relaxed.xyz + one periodic DM SP each (~seconds/state,
    NO relaxation) and writes the npz for future replots."""
    from spammm.quantum import pi_bond_order as pbo
    from spammm.topology.ribbon_pbc import junction_state_strings, bond_lengths_minimage
    n = args.ncells
    states = _state_dict(args)
    tag, builder = _enum_mode(args)
    for w in [int(x) for x in args.widths.split(',')]:
        wc = 2 * (w + 1)                    # --widths = RING ROWS
        epath = os.path.join(OUTDIR, f'{tag}_r{w}.npz')
        dpath = os.path.join(OUTDIR, f'{tag}_r{w}_data.npz')
        edat = np.load(epath)
        labels = [str(l) for l in edat['labels']]
        Es = dict(zip(labels, edat['E_relax']))
        Es_g = dict(zip(labels, edat['E_gamma']))
        rows, kw = [], {}
        d = np.load(dpath) if os.path.exists(dpath) else None
        for lab, st in states.items():
            if not np.isfinite(Es[lab]):
                continue
            atoms, lvs, hbonds = builder(wc, n, st.split('|')[0])
            if d is not None and f'apos_{lab}' in d:
                atoms.apos = d[f'apos_{lab}']
                rows.append((lab, atoms, lvs, hbonds, d[f'bl_{lab}'], d[f'bpi_{lab}'], d[f'seam_{lab}']))
                continue
            # recovery: relaxed geometry from disk + periodic DM SP only (no relax)
            xfile = os.path.join(OUTDIR, f'{tag}_r{w}', lab, 'relaxed.xyz')
            lines = open(xfile).read().split('\n')
            na = int(lines[0])
            en = [l.split()[0] for l in lines[2:2 + na]]
            apos = np.array([[float(x) for x in l.split()[1:4]] for l in lines[2:2 + na]])
            atoms.apos = apos
            # periodic pi-BO: dftb+ binary eigenvec.bin + oversqr.dat -> per-k Lowdin (E_gamma kept from npz)
            bpi, info = pbo.pi_bond_orders_pbc(en, apos, np.asarray(atoms.bonds), lvs,
                                               os.path.join(OUTDIR, f'{tag}_r{w}', lab, 'dm'),
                                               nk=(args.nk, 1 if args.vac else 2, 1), filling_temp=300, scctol=1e-6,
                                               maxscc=2000, mixer='DIIS { Generations = 8 }', sk_set=args.sk, verbose=False)
            Lx = lvs[0, 0]
            bonds = np.asarray(atoms.bonds)
            seam = np.abs(apos[bonds[:, 1], 0] - apos[bonds[:, 0], 0]) > 0.5 * Lx
            bl = bond_lengths_minimage(apos, bonds, Lx)
            rows.append((lab, atoms, lvs, hbonds, bl, bpi, seam))
            kw[f'apos_{lab}'] = apos; kw[f'bonds_{lab}'] = bonds
            kw[f'bl_{lab}'] = bl; kw[f'bpi_{lab}'] = bpi; kw[f'seam_{lab}'] = seam
            Es_g[lab] = info['E_ha'] * HAU2EV
            print(f"  w{w} {lab}: recovered (periodic eigenvec.bin DM; E_pbc={Es_g[lab]:.4f} eV, |dE_relax|={abs(Es_g[lab]-Es[lab])*1000:.1f} meV)")
        if kw:
            if d is not None:                       # merge keys already cached in data.npz
                for k in d.files:
                    if k not in kw: kw[k] = d[k]
            np.savez(dpath, **kw)
            np.savez(epath, labels=list(states.keys()),
                     E_relax=np.array([Es[l] for l in states]), E_gamma=np.array([Es_g[l] for l in states]),
                     lvs=lvs, width_chains=wc)
        E0 = np.nanmin(np.array([Es[l] for l in states]))
        png = os.path.join(OUTDIR, f'{tag}_r{w}_maps.png')
        plot_enum_maps(rows, Es, E0, w, args.dda, png, tile=0 if args.vac else 1,
                       ncell=1 if getattr(args, 'xscan', False) else 2)
        print(f"REVIEW: {png}")


def _prot_sites(atoms, atoms0=None):
    """Positions of switched/protonated edge sites (the perturbation sites).

    With atoms0 (0H reference): heavy atoms whose H-coordination EXCEEDS the
    reference — works for CH->CH2 (1->2 H) as well as N->NH (0->1 H); heavy-atom
    order is identical between states (extra H's are appended).  Without atoms0:
    every heavy atom bonded to an H (junction mode: only protonated N's have H).
    """
    def hcount(at):
        en = np.array([e.split('_')[0] for e in at.enames])
        cnt = np.zeros(at.natoms, int)
        for i, j in np.asarray(at.bonds):
            if en[j] == 'H': cnt[i] += 1
            if en[i] == 'H': cnt[j] += 1
        return en != 'H', cnt
    hvy, cnt = hcount(atoms)
    if atoms0 is not None:                                   # match heavy atoms by position
        hvy0, cnt0 = hcount(atoms0)
        ap = np.asarray(atoms.apos)[hvy]; ap0 = np.asarray(atoms0.apos)[hvy0]
        j0 = np.argmin(np.linalg.norm(ap[None, :, :] - ap0[:, None, :], axis=2), axis=0)
        j0f = np.where(hvy0)[0][j0]                      # heavy-subset index -> full atom index
        idx = np.where(hvy)[0][cnt[hvy] != cnt0[j0f]]    # != catches both +H (NH,CH2) and -H (C=O) switches
    else:
        idx = np.where(hvy & (cnt > 0))[0]
    return np.asarray(atoms.apos)[idx]


def _mark_sites(ax, atoms, atoms0, lvs, axis=0, ncell=2):
    """Magenta + on each switched/protonated site, replicated into each displayed cell."""
    for s in _prot_sites(atoms, atoms0):
        for k in range(ncell):
            p = s + k * lvs[axis]
            ax.plot(p[0], p[1], '+', color='green', ms=5, mew=1.3, zorder=9)


def _dist_to_sites(apos_b, sites, Lx):
    """Min distance from each bond midpoint to any protonated site
    (min-image along x only — y is a single ribbon, not translational)."""
    mid = 0.5 * (apos_b[:, 0] + apos_b[:, 1])
    r = np.full(len(mid), np.inf)
    for s in sites:
        d = mid - s
        d[:, 0] -= np.round(d[:, 0] / Lx) * Lx
        r = np.minimum(r, np.linalg.norm(d[:, :2], axis=1))
    return r


def run_analyze(args):
    """Quantitative analysis from saved enumsj_r*_data.npz — no recompute.

    Per width: state energies, interaction energies J(2H) = E_2H - E_1H - E_1H' + E_0H
    (cost of second protonation), and pi-BO response decay |Dbpi(r)| vs distance
    from the protonated site with an exponential fit -> penetration length xi.
    """
    import matplotlib.pyplot as plt
    from spammm.topology.ribbon_pbc import junction_state_strings
    n = args.ncells
    states = _state_dict(args)
    tag, builder = _enum_mode(args)
    jdc_tag = _jdc_tag(args)
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.5))
    ws_in = [int(x) for x in args.widths.split(',')]
    rows_J, colors, Jdec = [], plt.cm.viridis(np.linspace(0.05, 0.9, len(ws_in))), {}
    for iw, w in enumerate(ws_in):
        e = np.load(os.path.join(OUTDIR, f'{tag}_r{w}.npz'))
        d = np.load(os.path.join(OUTDIR, f'{tag}_r{w}_data.npz'))
        labels = [str(l) for l in e['labels']]
        Es = dict(zip(labels, e['E_relax']))
        Lx = float(e['lvs'][0, 0])
        wc = int(e['width_chains'])
        E1 = 0.5 * (Es['1H-p'] + Es['1H-d'])          # mirror-pair mean
        Js = {l: Es[l] - 2 * E1 + Es['0H'] for l in labels if l.startswith('2H')}
        jdc = os.path.join(OUTDIR, f'{jdc_tag}_r{w}.npz')          # electrostatic decomposition cache (from --j-decomp)
        if os.path.exists(jdc):
            z = np.load(jdc)
            Jrn = dict(zip([str(l) for l in z['labels']], z['J_rigid']))
            Jnn = dict(zip([str(l) for l in z['labels']], z['J_noscc']))
            Jdec[w] = {'res': Jnn, 'es': {l: Jrn[l] - Jnn[l] for l in Jrn}}   # residual(kekule) + electrostatic
        rows_J.append((w, Js, Es))
        print(f"\n=== r{w} ({wc} chains, Lx={Lx:.2f} A) ===")
        print(f"  J(second proton) [meV]: " + '  '.join(f'{l}={1000*v:+.1f}' for l, v in Js.items()))
        # PBE comparison (GPAW bundle, if energy.txt exists for all states);
        # baked layout is <root>/<tag>_r<w>/<lab>, legacy junction used <root>/r<w>/<lab>
        Eg = {}
        for lab in labels:
            ef_new = os.path.join('cluster_scan_ribbon_gpaw', f'{tag}_r{w}', lab, 'energy.txt')
            ef_old = os.path.join('cluster_scan_ribbon_gpaw', f'r{w}', lab, 'energy.txt')
            ef = ef_new if os.path.exists(ef_new) else ef_old if (not args.vac and os.path.exists(ef_old)) else None
            Eg[lab] = float(open(ef).read()) if ef else np.nan
        if np.isfinite(list(Eg.values())).all():
            E1g = 0.5 * (Eg['1H-p'] + Eg['1H-d'])
            Jg = {l: Eg[l] - 2 * E1g + Eg['0H'] for l in labels if l.startswith('2H')}
            print(f"  J(second proton) PBE : " + '  '.join(f'{l}={1000*v:+.1f}' for l, v in Jg.items()))
            rows_J[-1] += (Jg, 'PBE/dzp')
        else:
            # cross-method overlay: sibling debug/ribbon_<sk> cache (gpaw <-> mio)
            alt = {'gpaw': 'ribbon_mio', 'mio': 'ribbon_gpaw'}.get(args.sk.split('-')[0])
            ap = os.path.join(os.path.dirname(OUTDIR.rstrip('/')), alt, f'{tag}_r{w}.npz') if alt else None
            if ap and os.path.exists(ap):
                ea = np.load(ap)
                El = dict(zip([str(l) for l in ea['labels']], ea['E_relax']))
                if all(l in El for l in ('0H', '1H-p', '1H-d')):
                    E1a = 0.5 * (El['1H-p'] + El['1H-d'])
                    Ja = {l: El[l] - 2 * E1a + El['0H'] for l in El if l.startswith('2H')}
                    nm = alt.split('_')[1]
                    print(f"  J(second proton) {nm:4s}: " + '  '.join(f'{l}={1000*v:+.1f}' for l, v in Ja.items()))
                    rows_J[-1] += (Ja, nm)
        # --- pi-BO decay vs distance from protonated site(s) ---
        atoms0, lvs, hb = builder(wc, n, states['0H'].split('|')[0])
        en0 = np.array([x.split('_')[0] for x in atoms0.enames])
        hvy0 = np.where(en0 != 'H')[0]
        b0, bd0 = d['bpi_0H'], d['bonds_0H']
        mH0 = (en0[bd0[:, 0]] != 'H') & (en0[bd0[:, 1]] != 'H')     # backbone only (vac 0H has passivation C-H)
        b0, bd0 = b0[mH0], bd0[mH0]
        for lab, st in states.items():
            if lab == '0H' or f'bpi_{lab}' not in d:
                continue
            atoms, lvs, hb = builder(wc, n, st.split('|')[0])
            en = np.array([x.split('_')[0] for x in atoms.enames])
            atoms.apos = d[f'apos_{lab}']
            b1, bd1 = d[f'bpi_{lab}'], d[f'bonds_{lab}']
            mH = (en[bd1[:, 0]] != 'H') & (en[bd1[:, 1]] != 'H')
            hvy = np.where(en != 'H')[0]                                  # atom indices differ per state (O chem) -> match by position
            M = np.argmin(np.linalg.norm(atoms.apos[hvy][:, None, :2] - np.asarray(atoms0.apos)[hvy0][None, :, :2], axis=2), axis=1)
            ks = np.searchsorted(hvy, bd1[mH])
            pos = {tuple(sorted((int(hvy0[M[a]]), int(hvy0[M[b]])))): k for k, (a, b) in enumerate(ks)}
            pm = np.array([pos[tuple(sorted(b))] for b in bd0])           # bd0 row -> state's backbone position
            db = np.abs(b1[mH][pm] - b0)
            sites = _prot_sites(atoms, atoms0)
            r = _dist_to_sites(atoms.apos[bd1[mH][pm]], sites, Lx)   # state-space bond pairs in bd0 order
            mC = (en0[bd0[:, 0]] == 'C') & (en0[bd0[:, 1]] == 'C')
            ax = axs[0] if lab == '1H-p' else axs[1] if lab in ('2H-os-sep', f'2H-os-d{n // 2}') else None
            if ax is not None:
                ax.loglog(r[mC], db[mC] + 1e-4, 'o', ms=3, color=colors[iw], alpha=0.6, label=f'r{w}')
                # power-law fit on binned maxima: Friedel-type tail ~ r^-p
                edges = np.linspace(0, r[mC].max(), 9)
                rc = 0.5 * (edges[:-1] + edges[1:])
                rb = np.array([db[mC][(r[mC] >= lo) & (r[mC] < hi)].max() if ((r[mC] >= lo) & (r[mC] < hi)).any() else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
                ok = np.isfinite(rb) & (rb > 1e-4) & (rc > 1.5)
                if ok.sum() > 2:
                    c = np.polyfit(np.log(rc[ok]), np.log(rb[ok]), 1)
                    ax.loglog(rc, np.exp(np.polyval(c, np.log(rc))), '-', color=colors[iw], lw=1)
                    print(f"  {lab:10s}: max|Dbpi|={db.max():.3f}  sum={db.sum():.3f}  tail ~ r^{c[0]:.2f}")
    axs[0].set(xlabel='|r - site| [A]', ylabel='|Delta pi-BO|', title='1H-p response decay')
    axs[1].set(xlabel='|r - site| [A]', title='2H-os-sep response decay')
    for ax in axs[:2]:
        ax.legend(); ax.set_ylim(1e-4, 1)
    ws = [r[0] for r in rows_J]
    labs2 = [l for l in states if l.startswith('2H')]
    for l in labs2:
        axs[2].plot(ws, [1000 * r[1][l] for r in rows_J], 'o-', label=l)
        ov = [r for r in rows_J if len(r) > 3]
        if ov:
            axs[2].plot([r[0] for r in ov], [1000 * r[3][l] for r in ov], 's--', mfc='none', label=f'{l} {ov[0][4]}')
    axs[2].axhline(0, color='k', lw=0.5)
    ovnm = rows_J[0][4] if len(rows_J[0]) > 3 else 'none'
    axs[2].set(xlabel='ring rows', ylabel='J [meV]', title=f'2-proton interaction J (solid={args.sk}, dashed={ovnm})')
    axs[2].legend(fontsize=7, ncol=2)
    # --- J decomposition vs width: solid = electrostatic (rigid_SCC - noSCC),
    #     dashed = residual electronic/kekule (noSCC) — from jdecomp_r*.npz ---
    if Jdec:
        stl = labs2
        stc = dict(zip(stl, plt.cm.tab10(np.arange(len(stl)))))
        wsd = sorted(Jdec)
        for l in stl:
            axs[3].plot(wsd, [1000 * Jdec[w]['es'][l] for w in wsd], 'o-', color=stc[l], label=f'{l} ES')
            axs[3].plot(wsd, [1000 * Jdec[w]['res'][l] for w in wsd], 's--', mfc='none', color=stc[l], label=f'{l} res')
        axs[3].axhline(0, color='k', lw=0.5)
        axs[3].plot([], [], 'k-', label='electrostatic (SCC-noSCC)')
        axs[3].plot([], [], 'k--', label='residual (noSCC, kekule)')
        axs[3].set(xlabel='ring rows', ylabel='J [meV]', title='J decomposed: solid=electrostatic, dashed=residual/kekule')
        axs[3].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    png = os.path.join(OUTDIR, f'{tag}_analysis.png')
    fig.savefig(png, dpi=140)
    print(f"\nREVIEW: {png}")


GPAW_JOB = '''#!/usr/bin/env python3
"""GPAW LCAO/dzp/PBE job on the DFTB-relaxed cell: optional pinned relax, then
export of the AO Hamiltonian / overlap / density matrix / orbital coefficients
to hs.npz (see gpaw_hs_export.py, copied next to this file by bake_gpaw).
Run:  python job.py   (or: mpirun -np N gpaw python job.py)
Outputs: gpaw.out, energy.txt, hs.npz, all.gpw (wfs for STM/TH maps);
RELAX adds relax.traj/relaxed.xyz/relax.log.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ase.io import read, write
from ase.constraints import FixAtoms, FixCartesian
from gpaw import GPAW
from gpaw_hs_export import export_hs

RELAX = %r               # True = relax with junction-site N/O atoms pinned (as in DFTB)
NK = %r                  # same mesh as the DFTB reference runs ((nk,1,1) for vacuum cells)
PINS = %r                # junction scaffold indices (empty = no constraint)
ZFIX = %r                # True = FixCartesian z on ALL atoms (xscanz tilted-OH: blocks proton hops)

atoms = read('geom.xyz')                     # cell + pbc baked into the extxyz header
atoms.calc = GPAW(mode='lcao', xc='PBE', basis='dzp', kpts={'size': NK, 'gamma': False},
                  txt='gpaw.out', symmetry='off')
if RELAX:
    from ase.optimize import BFGS
    if ZFIX:
        atoms.set_constraint([FixCartesian(i, mask=(0, 0, 1)) for i in range(len(atoms))])
    elif PINS:
        atoms.set_constraint(FixAtoms(indices=PINS))
    BFGS(atoms, trajectory='relax.traj', logfile='relax.log').run(fmax=0.05)
    write('relaxed.xyz', atoms)
E = atoms.get_potential_energy()
with open('energy.txt', 'w') as f:
    f.write(f'{E:.8f}\\n')
export_hs(atoms.calc, atoms, 'hs.npz')
atoms.calc.write('all.gpw', mode='all')      # wfs on grid -> Tersoff-Hamann / STM maps via ase.dft.stm.STM
print(f'E = {E:.6f} eV')
'''


def bake_gpaw(args):
    """Bake a cluster bundle of GPAW LCAO/dzp/PBE jobs: one dir per width x state,
    geometry = DFTB-relaxed cell (extxyz, cell+pbc in header). RELAX=False -> SP.
    """
    from spammm.topology.ribbon_pbc import junction_state_strings, junction_site_atoms, bond_lengths_minimage
    n = args.ncells
    states = _state_dict(args)
    tag, builder = _enum_mode(args)
    root = args.bake_gpaw
    jobs = []
    for w in [int(x) for x in args.widths.split(',')]:
        wc = 2 * (w + 1)
        for lab, st in states.items():
            st = st.split('|')[0]
            atoms, lvs, hb = builder(wc, n, st)
            xfile = os.path.join(OUTDIR, f'{tag}_r{w}', lab, 'relaxed.xyz')
            ok = os.path.isfile(xfile)
            if ok:                                                  # reject distorted/proton-transferred relaxes
                lines = open(xfile).read().split('\n')
                na = int(lines[0])
                pos = np.array([[float(x) for x in l.split()[1:4]] for l in lines[2:2 + na]])
                ok = (len(pos) == atoms.natoms) and bond_lengths_minimage(pos, np.asarray(atoms.bonds), lvs[0, 0]).max() < 2.0
            if ok:
                atoms.apos = pos
            else:
                print(f"  {tag}_r{w}/{lab}: no usable relaxed.xyz -> IDEAL builder geometry", flush=True)
            atoms.apos[:, 2] += 0.5 * lvs[2, 2]                      # GPAW needs atoms off the z boundary
            pins = [] if args.vac else junction_site_atoms(atoms)
            nk = (args.nk, 1, 1) if args.vac else (args.nk, 2, 1)
            jd = os.path.join(root, f'{tag}_r{w}', lab)
            os.makedirs(jd, exist_ok=True)
            lat = ' '.join(f'{v:.6f}' for v in np.asarray(lvs).ravel())
            with open(os.path.join(jd, 'geom.xyz'), 'w') as f:
                f.write(f'{na}\nLattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T F"\n')
                for el, p in zip(atoms.enames, atoms.apos):
                    f.write(f'{el.split("_")[0]:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
            with open(os.path.join(jd, 'job.py'), 'w') as f:
                f.write(GPAW_JOB % (args.gpaw_relax, nk, list(pins), args.gpaw_zfix))
            shutil.copy(os.path.join(OUTDIR, '../../spammm/quantum/gpaw_hs_export.py'), os.path.join(jd, 'gpaw_hs_export.py'))
            jobs.append(os.path.relpath(jd, root))
    jf = os.path.join(root, 'jobs.txt')
    if os.path.exists(jf):                                    # merge with jobs baked earlier (other chems)
        jobs = sorted(set(jobs) | set(open(jf).read().split()))
    with open(jf, 'w') as f:
        f.write('\n'.join(jobs) + '\n')
    with open(os.path.join(root, 'run_all.sh'), 'w') as f:
        f.write('#!/usr/bin/env bash\nset -u\nROOT=$(cd "$(dirname "$0")" && pwd)\n'
                'while IFS= read -r d; do\n'
                '  [ -z "$d" ] && continue\n'
                '  [ -f "$ROOT/$d/energy.txt" ] && continue\n'
                '  echo "[$(date +%H:%M:%S)] $d"\n'
                '  (cd "$ROOT/$d" && python job.py > stdout.txt 2> stderr.txt)\n'
                'done < "$ROOT/jobs.txt"\necho ALL DONE\n')
    os.chmod(os.path.join(root, 'run_all.sh'), 0o755)
    with open(os.path.join(root, 'submit.pbs'), 'w') as f:
        f.write('#!/bin/bash\n'
                '# Metacentrum PBS array job: one task per line of jobs.txt\n'
                '# Adjust module/venv activation to your setup before qsub.\n'
                '#PBS -N ribbon_hs\n'
                '#PBS -l select=1:ncpus=8:mem=16gb\n'
                '#PBS -l walltime=24:00:00\n'
                '#PBS -J 0-%d\n'
                'cd "$PBS_O_WORKDIR"\n'
                '# module load gpaw  # or: source /path/to/venv/bin/activate\n'
                'd=$(sed -n "$((PBS_ARRAY_INDEX + 1))p" jobs.txt)\n'
                '[ -f "$d/energy.txt" ] && exit 0\n'
                'cd "$d" && OMP_NUM_THREADS=$PBS_NCPUS python job.py > stdout.txt 2> stderr.txt\n' % (len(jobs) - 1))
    print(f"baked {len(jobs)} GPAW jobs -> {root}  (run_all.sh + jobs.txt + submit.pbs)")


def run_gpaw_import(args):
    """Import a GPAW hs.npz bundle -> standard npz caches under debug/ribbon_gpaw/.

    Layout: <root>/<tag>_r<w>/<lab>/hs.npz (as baked by --bake-gpaw; tag =
    xscanv_<chem> or enumv_<chem>).  For every state: rebuild the cell via the
    same builder (atom order is preserved through geom.xyz -> GPAW), take
    pos_ac (GPAW-relaxed geometry), compute bond lengths + pi bond orders from
    rho_kMM/S_kMM (per-k Lowdin on the multi-channel dzp pi subspace,
    spammm/quantum/gpaw_hs.py).  Writes {tag}_r{w}.npz (labels, E_relax=E_total,
    E_band=eigenvalue sum, lvs, width_chains) + {tag}_r{w}_data.npz
    (apos_/bonds_/bl_/bpi_/seam_<lab>) — so --enum-plot/--enum-diff/--nonlin/
    --analyze run unchanged with `--sk gpaw` (-> OUTDIR=debug/ribbon_gpaw).
    Note: no SCC on/off in GPAW -> no noSCC/electrostatic J split; J_band is
    stored as the closest transferable decomposition.
    """
    import glob
    from spammm.quantum import gpaw_hs
    from spammm.topology.ribbon_pbc import (xscan_state_strings, junction_state_strings,
                                          build_edge_switch_cell, bond_lengths_minimage)
    root = args.gpaw_import
    GOUT = os.path.join(os.path.dirname(OUTDIR.rstrip('/')), 'ribbon_gpaw')
    os.makedirs(GOUT, exist_ok=True)
    A_CC = 1.42
    for d in sorted(glob.glob(os.path.join(root, '*_r*'))):
        base = os.path.basename(d)                                # xscanv_C_r1
        tag, ws = base.rsplit('_r', 1)
        if not ws.isdigit():
            continue
        w = int(ws); wc = 2 * (w + 1); chem = tag.split('_')[-1]
        labs = sorted(os.listdir(d))
        hsd = {lab: os.path.join(d, lab, 'hs.npz') for lab in labs}
        hsd = {lab: p for lab, p in hsd.items() if os.path.isfile(p)}
        if not hsd:
            continue
        # infer ncells + canonical state order from the 0H cell
        hs0 = gpaw_hs.load_hs(hsd['0H']) if '0H' in hsd else gpaw_hs.load_hs(hsd[sorted(hsd)[0]])
        n = int(round(hs0['cell_cv'][0, 0] / (2 * A_CC * np.cos(np.pi / 6))))
        states = (xscan_state_strings(n) if tag.startswith('xscanv') else junction_state_strings(n))
        labels = [l for l in states if l in hsd] + [l for l in hsd if l not in states]
        print(f'\n##### {base}: ncells={n}, wc={wc}, {len(labels)} states #####', flush=True)
        kw, Es, Eb, Ef = {}, {}, {}, {}
        lvs = None
        for lab in labels:
            hs = gpaw_hs.load_hs(hsd[lab])
            st = states.get(lab, '0' * n).split('|')[0]
            atoms, lvs_b, _ = build_edge_switch_cell(wc, n, st, chem=chem, vac_y=args.vacy)
            en = np.array([e.split('_')[0] for e in atoms.enames])
            assert atoms.natoms == len(hs['pos_ac']), f'{base}/{lab}: natoms {atoms.natoms} != {len(hs["pos_ac"])}'
            assert all(gpaw_hs.Z2E[z] == e for z, e in zip(hs['Z_a'], en)), f'{base}/{lab}: element order mismatch'
            lvs = hs['cell_cv']
            assert np.abs(lvs - lvs_b).max() < 1e-3, f'{base}/{lab}: cell mismatch {np.abs(lvs - lvs_b).max()}'
            apos = hs['pos_ac'].copy(); apos[:, 2] -= apos[:, 2].mean()      # undo bake's +Lz/2 shift (z~0 like builder)
            dd = apos - np.asarray(atoms.apos); dd[:, 0] -= np.round(dd[:, 0] / lvs[0, 0]) * lvs[0, 0]
            dmax = np.linalg.norm(dd, axis=1).max()
            assert dmax < 1.0, f'{base}/{lab}: atoms moved {dmax:.2f} A vs builder — wrong cell/order?'
            bonds = np.asarray(atoms.bonds)
            bl = bond_lengths_minimage(apos, bonds, lvs[0, 0])
            assert bl.max() < 2.0, f'{base}/{lab}: distorted bond {bl.max():.2f} A'
            seam = np.abs(apos[bonds[:, 1], 0] - apos[bonds[:, 0], 0]) > 0.5 * lvs[0, 0]
            bpi, info = gpaw_hs.pi_bond_orders_hs(hs, bonds, apos=apos, lvs=lvs)
            Es[lab], Eb[lab], Ef[lab] = hs['E_total'], info['E_band'], hs['E_fermi']
            kw[f'apos_{lab}'] = apos; kw[f'bonds_{lab}'] = bonds
            kw[f'bl_{lab}'] = bl; kw[f'bpi_{lab}'] = bpi; kw[f'seam_{lab}'] = seam
            print(f'  {lab:10s}: E={Es[lab]:12.4f} eV  E_band={Eb[lab]:12.4f}  EF={Ef[lab]:7.3f}  |d_bld|={dmax:.3f} A', flush=True)
        np.savez(os.path.join(GOUT, f'{tag}_r{w}.npz'), labels=labels,
                 E_relax=np.array([Es[l] for l in labels]), E_gamma=np.array([Es[l] for l in labels]),
                 E_band=np.array([Eb[l] for l in labels]), E_fermi=np.array([Ef[l] for l in labels]),
                 lvs=lvs, width_chains=wc)
        np.savez(os.path.join(GOUT, f'{tag}_r{w}_data.npz'), **kw)
        if '1H-p' in Es and '1H-d' in Es:
            print(f'  PARITY 1H-p vs 1H-d: |dE| = {abs(Es["1H-p"] - Es["1H-d"]) * 1000:.3f} meV')
        if '0H' in Es and '1H-p' in Es:
            E1 = 0.5 * (Es['1H-p'] + Es['1H-d']); E1b = 0.5 * (Eb['1H-p'] + Eb['1H-d'])
            print(f'  J(d) [meV]:  ' + '  '.join(f"{l.replace('2H-','')}={1000 * (Es[l] - 2 * E1 + Es['0H']):+.0f}"
                                               for l in labels if l.startswith('2H')))
            print(f'  J_band[meV]: ' + '  '.join(f"{l.replace('2H-','')}={1000 * (Eb[l] - 2 * E1b + Eb['0H']):+.0f}"
                                               for l in labels if l.startswith('2H')))
    print(f'\ncaches -> {GOUT};  figures: --sk gpaw --vac --xscan --chem <C|N> --ncells {n} --widths <w> [--enum-plot|--enum-diff|--nonlin|--analyze|--compare]')


def run_geom_plot(args):
    """Geometry overview: rows=widths, cols=states; relaxed cells (2 replicas along y),
    element-colored atoms, grey bonds, junction D-H green / H...A magenta dashed,
    protonated N sites circled red."""
    import matplotlib.pyplot as plt
    from spammm import elements
    from spammm.quantum import pi_bond_order as pbo
    from spammm.topology.ribbon_pbc import junction_state_strings
    from spammm.topology.hbond_utils import hbond_positions
    n = args.ncells
    states = _state_dict(args)
    tag, builder = _enum_mode(args)
    t_ax = 0 if args.vac else 1
    ws = [int(x) for x in args.widths.split(',')]
    fig, axs = plt.subplots(len(ws), len(states), figsize=(len(states) * 1.9, len(ws) * 4.4), squeeze=False)
    for iw, w in enumerate(ws):
        wc = 2 * (w + 1)
        d = np.load(os.path.join(OUTDIR, f'{tag}_r{w}_data.npz'))
        atoms0, _, _ = builder(wc, n, states['0H'].split('|')[0])
        for i, (lab, st) in enumerate(states.items()):
            ax = axs[iw, i]
            atoms, lvs, hbonds = builder(wc, n, st.split('|')[0])
            atoms.apos = d[f'apos_{lab}']
            seam = d[f'seam_{lab}']
            ns, apos2, _, _ = _tile2(atoms, np.zeros(len(atoms.bonds)), seam, lvs, axis=t_ax)
            pbo.plot_bond_scalar_map(ax, ns, apos2, np.zeros(len(ns.bonds)), bonds=ns.bonds,
                                     cmap='Greys', norm=plt.Normalize(0, 1), mask=np.zeros(len(ns.bonds), bool), bAtoms=False)
            en2 = [e.split('_')[0] for e in ns.enames]
            ax.scatter(apos2[:, 0], apos2[:, 1], s=7, c=[elements.ELEMENT_DICT[e][8] for e in en2], zorder=6, linewidths=0)
            for hb in hbonds:
                for rep in (-1, 0, 1):
                    drep = rep * lvs[1]
                    pD, pH, pA = hbond_positions(atoms.apos, hb, lvs)
                    ax.plot([pD[0] + drep[0], pH[0] + drep[0]], [pD[1] + drep[1], pH[1] + drep[1]], 'g-', lw=1.2, zorder=5)
                    ax.plot([pH[0] + drep[0], pA[0] + drep[0]], [pH[1] + drep[1], pA[1] + drep[1]], 'm--', lw=1.0, zorder=5)
            for s in _prot_sites(atoms, atoms0):       # switched/protonated edge sites
                reps = (0, 1) if t_ax == 1 else (0,)
                for rep in reps:
                    ax.add_patch(plt.Circle((s[0], s[1] + rep * lvs[1, 1]), 0.55, fill=False, color='red', lw=1.2, zorder=7))
            _cell_boxes(ax, lvs, axis=t_ax)
            ax.set_aspect('equal'); ax.axis('off'); ax.margins(0.02)
            if t_ax == 1:
                ax.set_ylim(-1.6, 2.0 * lvs[1, 1] + 1.6)
            if iw == 0:
                ax.set_title(lab, fontsize=9)
            if i == 0:
                ax.text(-0.06, 0.5, f'r{w}', rotation=90, va='center', ha='right', fontsize=11, transform=ax.transAxes)
    fig.suptitle(f'{tag} ribbon cells (relaxed; switched sites circled)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(OUTDIR, f'{tag}_geometries.png')
    fig.savefig(png, dpi=140, bbox_inches='tight')
    print(f"REVIEW: {png}")


def run_jdecomp(args):
    """J decomposition per width: total (relaxed) = electronic (noSCC, fixed
    skeleton) + electrostatic (SCC charge response, rigid) + elastic (relax).
    All on the 0H-relaxed skeleton, H at fixed 1.01 A -> rigid/noSCC are SPs.
    Waterfall bars per 2H state; flat electronic baseline = no pi coupling."""
    import matplotlib.pyplot as plt
    from spammm.topology.ribbon_pbc import junction_state_strings
    from spammm.topology.hbond_utils import hbond_positions
    from spammm.quantum.DFTB_utils import run_pbc
    n = args.ncells
    states = _state_dict(args)
    tag, builder = _enum_mode(args)
    jdc_tag = _jdc_tag(args)
    nky = 1 if args.vac else 2
    labs2 = [l for l in states if l.startswith('2H')]
    ws = [int(x) for x in args.widths.split(',')]
    fig, axs = plt.subplots(1, len(ws), figsize=(3.2 * len(ws) + 1, 4.5), sharey=True, squeeze=False)
    for iw, w in enumerate(ws):
        wc = 2 * (w + 1)
        cache = os.path.join(OUTDIR, f'{jdc_tag}_r{w}.npz')
        if os.path.exists(cache):
            z = np.load(cache)
            Jr, Jn = dict(zip(z['labels'], z['J_rigid'])), dict(zip(z['labels'], z['J_noscc']))
        else:
            d = np.load(os.path.join(OUTDIR, f'{tag}_r{w}_data.npz'))
            apos0 = d['apos_0H']
            atoms0, _, _ = builder(wc, n, states['0H'].split('|')[0])
            en0 = np.array([e.split('_')[0] for e in atoms0.enames])
            assert len(apos0) == atoms0.natoms and len(apos0[en0 != 'H']) == int((en0 != 'H').sum())
            Er, En = {}, {}
            for lab, st in states.items():
                atoms, lvs, hbs = builder(wc, n, st.split('|')[0])
                en = np.array([e.split('_')[0] for e in atoms.enames])
                apos = atoms.apos.copy()
                if getattr(args, 'tagp', None):              # custom set (e.g. xscanz): inherit FULL relaxed-0H skeleton incl. tilted OH H's
                    for i in range(len(apos)):
                        j0 = np.where(en0 == en[i])[0]
                        dd = apos0[j0] - apos[i]; dd[:, 0] -= np.round(dd[:, 0] / lvs[0, 0]) * lvs[0, 0]
                        apos[i] = apos0[j0[np.argmin(np.linalg.norm(dd, axis=1))]]
                else:
                    apos[en != 'H'] = apos0[en0 != 'H']      # rigid 0H heavy skeleton; state H's at built positions (base-H offset cancels in J)
                for hb in hbs:
                    if hb.h_idx is None:
                        continue
                    pD, _, pA = hbond_positions(apos, hb, lvs)
                    ax_ = (pA - pD) / np.linalg.norm(pA - pD)
                    apos[hb.h_idx] = pD + 1.01 * ax_
                for scc, dd in [(True, Er), (False, En)]:
                    E, _, _ = run_pbc(apos, list(atoms.enames), lvs, nk=(args.nk, nky, 1), SCC=scc,
                                      workdir=os.path.join(OUTDIR, f'{jdc_tag}_r{w}', lab + ('' if scc else '_noscc')),
                                      Temperature=300, Mixer='DIIS { Generations = 8 }' if scc else None,
                                      MaxScc=1000, SCCTolerance=1e-8, sk_set=args.sk)
                    dd[lab] = E * HAU2EV
                print(f'  r{w} {lab} done', flush=True)
            E0, E1 = Er['0H'], 0.5 * (Er['1H-p'] + Er['1H-d'])
            E0n, E1n = En['0H'], 0.5 * (En['1H-p'] + En['1H-d'])
            Jr = {l: Er[l] - 2 * E1 + E0 for l in labs2}
            Jn = {l: En[l] - 2 * E1n + E0n for l in labs2}
            np.savez(cache, labels=labs2, J_rigid=np.array([Jr[l] for l in labs2]), J_noscc=np.array([Jn[l] for l in labs2]))
        e = np.load(os.path.join(OUTDIR, f'{tag}_r{w}.npz'))
        Es = dict(zip([str(l) for l in e['labels']], e['E_relax']))
        E0r, E1r = Es['0H'], 0.5 * (Es['1H-p'] + Es['1H-d'])
        Jrel = {l: Es[l] - 2 * E1r + E0r for l in labs2}
        ax = axs[0, iw]
        xs = np.arange(len(labs2))
        Ja = np.array([Jn[l] * 1000 for l in labs2])                     # electronic (noSCC)
        Jb = np.array([(Jr[l] - Jn[l]) * 1000 for l in labs2])           # electrostatic (SCC charge response)
        Jc = np.array([(Jrel[l] - Jr[l]) * 1000 for l in labs2])         # elastic (relaxation)
        Jt = np.array([Jrel[l] * 1000 for l in labs2])                   # total relaxed J
        ax.plot(xs, Jt, 'k-o', lw=1.8, label='total J (relaxed)' if iw == 0 else None)
        ax.plot(xs, Ja, 's--', color='0.45', label='electronic residual (noSCC)' if iw == 0 else None)
        ax.plot(xs, Jb, '^-', color='#1f77b4', label='electrostatic (SCC)' if iw == 0 else None)
        ax.plot(xs, Jc, 'v-', color='#2ca02c', label='elastic (relax)' if iw == 0 else None)
        ax.plot(xs, Ja + Jb + Jc, 'k:', lw=1.0)                          # sanity: components sum to total
        ax.axhline(np.mean(Ja), color='0.4', ls='--', lw=1)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(range(len(labs2)))
        ax.set_xticklabels([l.replace('2H-', '') for l in labs2], rotation=45, ha='right', fontsize=8)
        ax.set_title(f'r{w}', fontsize=10)
        if iw == 0:
            ax.set_ylabel('J = E_2H - 2E_1H + E_0H [meV]')
    axs[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('J decomposition (lines): total (black) = electronic residual (grey --) + electrostatic (blue ^) + elastic (green v)\n'
                 'residual electronic baseline is FLAT -> no pi/kekule coupling in J')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    png = os.path.join(OUTDIR, f'{tag}_J_decomp.png')
    fig.savefig(png, dpi=140)
    print(f"REVIEW: {png}")


def _xyz(p):
    import re
    ls = open(p).read().strip().split('\n')
    es = np.array([l.split()[0] for l in ls[2:]])
    xy = np.array([[float(v) for v in l.split()[1:4]] for l in ls[2:]])
    Lx = float(re.findall(r'\(([-\d.]+),', ls[1])[0])
    return es, xy, Lx


def _switched_arr(e0, x0, e2, x2, Lx):
    """heavy-atom indices in state (e2,x2) whose H-neighbour count differs vs (e0,x0)."""
    def hc(pos, es):
        ih = np.where(es == 'H')[0]; iv = np.where(es != 'H')[0]
        d = pos[ih][:, None, :] - pos[iv][None, :, :]
        d[..., 0] -= np.round(d[..., 0] / Lx) * Lx
        return (np.linalg.norm(d, axis=2) < 1.3).sum(0), iv
    c0, iv0 = hc(x0, e0); c2, iv2 = hc(x2, e2)
    d = x2[iv2][:, None, :] - x0[iv0][None, :, :]
    d[..., 0] -= np.round(d[..., 0] / Lx) * Lx
    return iv2[c2 != c0[np.linalg.norm(d, axis=2).argmin(1)]]


def _switched(p0, p2):
    e0, x0, Lx = _xyz(p0)
    e2, x2, _ = _xyz(p2)
    return _switched_arr(e0, x0, e2, x2, Lx)


def _sketch_arr(ax, e0, x, sw, Lx):
    import matplotlib.pyplot as plt
    ii, jj = np.triu_indices(len(x), 1)
    d = x[jj] - x[ii]
    d[:, 0] -= np.round(d[:, 0] / Lx) * Lx
    r = np.linalg.norm(d, axis=1)
    heavy = np.array([e != 'H' for e in e0])
    m = r < np.where(heavy[ii] & heavy[jj], 1.7, 1.3)
    for a, b in zip(ii[m], jj[m]):
        ax.plot([x[a, 0], x[b, 0] - np.round((x[b, 0] - x[a, 0]) / Lx) * Lx],
                [x[a, 1], x[b, 1]], 'k-', lw=0.8, alpha=0.3, zorder=1)
    from spammm import plotUtils as pu, elements
    plt.sca(ax)
    pu.plotAtoms(apos=x, es=list(e0),
                 sizes=[elements.ELEMENT_DICT[e][6] * 20. for e in e0],
                 colors=[elements.ELEMENT_DICT[e][8] for e in e0], marker='o', axes=(0, 1))
    from matplotlib.patches import Circle
    for a in sw:
        ax.add_patch(Circle(x[a, :2], 1.35, fill=False, ec='r', lw=1.6, zorder=5))
    ax.set_xlim(-1.5, Lx + 1.5)                   # shared zoom: fixed cell view,
    ax.set_ylim(x[:, 1].min() - 1.5, x[:, 1].max() + 1.5)  # not per-panel autoscale
    ax.set_aspect('equal'); ax.axis('off')


def _sketch(ax, p0, p2):
    e0, x, Lx = _xyz(p0)
    _sketch_arr(ax, e0, x, _switched(p0, p2), Lx)


def run_jterms_gpaw(args):
    """GPAW analog of run_jterms — same 3-row layout, honest 2-term split.

    GPAW has no detailed.out H0/SCC/3rd/rep terms; the relaxed J decomposes as
      J_tot = J_band + J_rest,   J_band from E_band = sum_k w_k Tr(H_k rho_k)
    (band-structure part; J_rest = double-counting/XC/entropy remainder).
    Bottom row: GPAW J_tot (solid) vs DFTB/mio J_tot (dashed, sibling caches).
    Sketches from cache apos (same atom order as the builder)."""
    import matplotlib.pyplot as plt
    from spammm.topology.ribbon_pbc import build_edge_switch_cell
    n = args.ncells
    states = _state_dict(args)
    tagf = getattr(args, 'tagp', None) or ('xscanv' if getattr(args, 'xscan', False) else 'enumv')
    CHEMS = ['C', 'N', 'O']
    ws = [int(x) for x in args.widths.split(',')]
    for w in ws:
        wc = 2 * (w + 1)
        DD = {}
        for c in CHEMS:
            pf, df = os.path.join(OUTDIR, f'{tagf}_{c}_r{w}.npz'), os.path.join(OUTDIR, f'{tagf}_{c}_r{w}_data.npz')
            if not (os.path.exists(pf) and os.path.exists(df)):
                continue
            e, d = np.load(pf), np.load(df, allow_pickle=True)
            labs = [str(l) for l in e['labels']]
            E, Eb = dict(zip(labs, e['E_relax'])), dict(zip(labs, e['E_band']))
            pl = [l for l in labs if l.startswith('2H')]
            J = {l: {'tot': (E[l] - E['1H-p'] - E['1H-d'] + E['0H']) * 1000,
                     'band': (Eb[l] - Eb['1H-p'] - Eb['1H-d'] + Eb['0H']) * 1000} for l in pl}
            for l in pl:
                J[l]['rest'] = J[l]['tot'] - J[l]['band']
            DD[c] = (pl, J, d)
        if not DD:
            continue
        # DFTB sibling totals for the overlay row
        dftb = {}
        for c in DD:
            pf = os.path.join(os.path.dirname(OUTDIR), 'ribbon_mio', f'{tagf}_{c}_r{w}.npz')
            if os.path.exists(pf):
                e = np.load(pf)
                Es = dict(zip([str(l) for l in e['labels']], e['E_relax']))
                dftb[c] = {l: (Es[l] - Es['1H-p'] - Es['1H-d'] + Es['0H']) * 1000
                           for l in [str(l) for l in e['labels']] if str(l).startswith('2H')}
        PAIRLABS = DD[list(DD)[0]][0]
        oslabs = [l for l in PAIRLABS if '-os-' in l]
        NP = len(PAIRLABS)
        NC = NP + len(oslabs) + 1
        glo = [min(J['tot'] for _, J, _ in DD.values() for J in J.values()),
               max(J['tot'] for _, J, _ in DD.values() for J in J.values())]
        zg = [min(J['tot'] for _, J, _ in DD.values() for l, J in J.items() if '-os-' in l),
              max(J['tot'] for _, J, _ in DD.values() for l, J in J.items() if '-os-' in l)]
        glo = [glo[0] * 1.08 - 40, glo[1] * 1.08 + 40]
        zm = max(-zg[0], zg[1], 1.0); zg = [-zm * 1.12 - 10, zm * 1.12 + 10]
        fig = plt.figure(figsize=(16, 10.5))
        gs = fig.add_gridspec(3, len(DD) * NC, height_ratios=[0.32, 1.3, 1.3], hspace=0.35,
                              wspace=0.15, left=0.05, right=0.97, top=0.93, bottom=0.06)
        e12 = np.linspace(0, len(DD) * NC, NP + 1).astype(int)
        c0 = list(DD)[0]
        d0 = DD[c0][2]
        a0, lvs, _ = build_edge_switch_cell(wc, n, states['0H'].split('|')[0], chem=c0, vac_y=args.vacy)
        en0 = np.array([e.split('_')[0] for e in a0.enames])
        x0 = np.asarray(d0['apos_0H']); Lx = float(lvs[0, 0])
        for ip, lab in enumerate(PAIRLABS):
            ax = fig.add_subplot(gs[0, e12[ip]:e12[ip + 1]])
            a2, _, _ = build_edge_switch_cell(wc, n, states[lab].split('|')[0], chem=c0, vac_y=args.vacy)
            en2 = np.array([e.split('_')[0] for e in a2.enames])
            x2 = np.asarray(d0[f'apos_{lab}'])
            _sketch_arr(ax, en0, x0, _switched_arr(en0, x0, en2, x2, Lx), Lx)
            ax.set_title(lab.replace('2H-', ''), fontsize=10)

        def _fin(ax, labs, zoom=False):
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(range(len(labs)))
            ax.set_xticklabels([l.replace('2H-', '') for l in labs], fontsize=7, rotation=45, ha='right')
            if zoom:
                ax.set_facecolor('#f3f3f3')
                ax.set_title(f'os zoomed x{(glo[1] - glo[0]) / (zg[1] - zg[0]):.1f}', fontsize=8)

        ax0 = az0 = None
        for ix, c in enumerate(DD):                            # row 1: band/rest split
            pl, J, _ = DD[c]
            axm = fig.add_subplot(gs[1, ix * NC:ix * NC + NP], sharey=ax0); ax0 = ax0 or axm
            axz = fig.add_subplot(gs[1, ix * NC + NP:(ix + 1) * NC], sharey=az0); az0 = az0 or axz
            for ax_, labs in ((axm, pl), (axz, oslabs)):
                for ip, lab in enumerate(labs):
                    b, r = J[lab]['band'], J[lab]['rest']
                    ax_.bar(ip, b, width=0.7, color='tab:blue',
                            label='band (E_band)' if ix == 0 and ip == 0 and ax_ is axm else None)
                    ax_.bar(ip, r, bottom=b, width=0.7, color='tab:gray',
                            label='rest (XC/double-count)' if ix == 0 and ip == 0 and ax_ is axm else None)
                    ax_.plot(ip, J[lab]['tot'], 'kD', ms=5)
            _fin(axm, pl); _fin(axz, oslabs, zoom=True)
            axm.set_title(f'{c} chem — relaxed (band/rest split)', fontsize=9)
            if ix == 0:
                axm.set_ylabel('J [meV]'); axm.legend(fontsize=8, title='diamond=total')
                axz.set_ylabel('J [meV]')
            else:
                axm.tick_params(labelleft=False); axz.tick_params(labelleft=False)
        for ix, c in enumerate(DD):                            # row 2: GPAW vs DFTB totals
            pl, J, _ = DD[c]
            axm = fig.add_subplot(gs[2, ix * NC:ix * NC + NP], sharey=ax0)
            axz = fig.add_subplot(gs[2, ix * NC + NP:(ix + 1) * NC], sharey=az0)
            for ax_, labs in ((axm, pl), (axz, oslabs)):
                for ip, lab in enumerate(labs):
                    ax_.plot(ip, J[lab]['tot'], 'ko-', ms=5, lw=1.4,
                             label='GPAW/dzp' if ix == 0 and ip == 0 and ax_ is axm else None)
                if c in dftb:
                    ax_.plot(range(len(labs)), [dftb[c].get(l, np.nan) for l in labs], 's--',
                             color='0.45', mfc='none', ms=5,
                             label='DFTB/mio' if ix == 0 and ax_ is axm else None)
            _fin(axm, pl); _fin(axz, oslabs, zoom=True)
            axm.set_title(f'{c} chem — J_tot GPAW vs DFTB', fontsize=9)
            if ix == 0:
                axm.set_ylabel('J [meV]'); axm.legend(fontsize=8)
                axz.set_ylabel('J [meV]')
            else:
                axm.tick_params(labelleft=False); axz.tick_params(labelleft=False)
        if ax0 is not None:
            ax0.set_ylim(glo)
        if az0 is not None:
            az0.set_ylim(zg)
        fig.suptitle(f'r{w} vac ribbon (GPAW/dzp): J = E(2H)-E(1H-p)-E(1H-d)+E(0H)   —   top: relaxed cells, red rings = switched sites\n'
                     'mid: relaxed, band-structure (E_band) vs remainder | bottom: GPAW vs DFTB/mio J_tot', fontsize=11)
        png = os.path.join(OUTDIR, f"Jterms{'_xscan' if tagf == 'xscanv' else ''}_r{w}.png")
        fig.savefig(png, dpi=160); fig.savefig(png[:-4] + '.svg'); plt.close(fig)
        print('REVIEW: ' + png)


def run_jterms(args):
    """detailed.out TERM decomposition of J = E(2H)-E(1H-p)-E(1H-d)+E(0H)
    per (chem, width, arrangement) — same partition as the molecular site
    scan (site_j_analysis in testplot_mol_flakes_dftb.py):
      J = J_H0 + J_SCC [+ J_3rd for 3ob] + J_rep.
    Two sources, both already on disk (no new DFTB runs):
      enumv_<c>_r<w>/<lab>/    = fully relaxed states (J includes elastic)
      jdecompv_<c>_r<w>/<lab>/ = rigid 0H skeleton, SCC on (pure electronic
                                 split: H0 band vs SCC charge response)
    Fig per width: Jterms_r<w>.png — top row = cell sketches with switched
    sites ringed, mid = relaxed J bars, bottom = rigid-skeleton J bars."""
    import re
    import matplotlib.pyplot as plt
    if args.sk == 'gpaw':
        return run_jterms_gpaw(args)
    FIELDS = [('H0', 'Energy H0'), ('SCC', 'Energy SCC'), ('3rd', 'Energy 3rd'),
              ('rep', 'Repulsive energy'), ('tot', 'Total energy')]
    CHEMS = ['C', 'N', 'O']
    tagf = getattr(args, 'tagp', None) or ('xscanv' if getattr(args, 'xscan', False) else 'enumv')
    jdcf = 'jdecompv' if tagf == 'enumv' else f'jdecomp_{tagf}'

    def _comps(d, lab):
        f = os.path.join(OUTDIR, d, lab, 'detailed.out')
        if not os.path.isfile(f):
            return None
        txt = open(f).read()
        out = {}
        for k, fld in FIELDS:
            m = re.findall(re.escape(fld) + r':\s+(-?\d+\.\d+)\s+H', txt)
            out[k] = float(m[-1]) if m else np.nan
        return out

    def Jset(d, suffix=''):
        E0, Ep, Ed = (_comps(d, l + suffix) for l in ('0H', '1H-p', '1H-d'))
        if E0 is None or Ep is None or Ed is None:
            return None
        out = {}
        for lab in PAIRLABS:
            E2 = _comps(d, lab + suffix)
            out[lab] = {k: (E2[k] - Ep[k] - Ed[k] + E0[k]) * HAU2EV for k, _ in FIELDS} \
                if E2 is not None else {k: np.nan for k, _ in FIELDS}
        return out

    ws = [int(x) for x in args.widths.split(',')]
    CC = {'H0': 'tab:blue', 'SCC': 'tab:orange', '3rd': 'tab:purple', 'rep': 'tab:gray'}
    W = {}
    glo = [0.0, 0.0]                                   # global y-range across all widths
    zg = [0.0, 0.0]                                    # zoomed y-range (os totals only)
    for w in ws:
        e = None
        for c0 in CHEMS:                                     # labels from first available chem npz (partial sets)
            pf = os.path.join(OUTDIR, f'{tagf}_{c0}_r{w}.npz')
            if os.path.exists(pf):
                e = np.load(pf); break
        if e is None:
            continue
        PAIRLABS = [str(l) for l in e['labels'] if str(l).startswith('2H')]
        relax = {c: Jset(f'{tagf}_{c}_r{w}') for c in CHEMS}
        rigid = {c: Jset(f'{jdcf}_{c}_r{w}') for c in CHEMS}
        noscc = {c: Jset(f'{jdcf}_{c}_r{w}', '_noscc') for c in CHEMS}
        W[w] = (PAIRLABS, relax, rigid, noscc)
        for c in CHEMS:                                # accumulate global y-range
            if relax[c] is not None:
                for lab in PAIRLABS:
                    bot = 0.0
                    for k in ('H0', 'SCC', '3rd', 'rep'):
                        bot += relax[c][lab][k] * 1000
                        glo[0], glo[1] = min(glo[0], bot), max(glo[1], bot)
                    if '-os-' in lab:                  # zoom range tracks the diamonds
                        zg[0], zg[1] = min(zg[0], relax[c][lab]['tot'] * 1000), \
                                       max(zg[1], relax[c][lab]['tot'] * 1000)
            if rigid[c] is not None and noscc[c] is not None:
                for lab in PAIRLABS:
                    for v in (noscc[c][lab]['tot'], rigid[c][lab]['tot']):
                        glo[0], glo[1] = min(glo[0], v * 1000), max(glo[1], v * 1000)
                        if '-os-' in lab:
                            zg[0], zg[1] = min(zg[0], v * 1000), max(zg[1], v * 1000)
    glo[0], glo[1] = glo[0] * 1.08 - 40, glo[1] * 1.08 + 40
    zm = max(-zg[0], zg[1], 1.0)                       # symmetric zoom, all os diamonds visible
    zg = [-zm * 1.12 - 10, zm * 1.12 + 10]
    for w in ws:
        PAIRLABS, relax, rigid, noscc = W[w]
        for mode, Js in (('relaxed', relax), ('rigid', rigid)):
            print(f'\n==== r{w} {mode}: J(lab) = E(2H)-E(1H-p)-E(1H-d)+E(0H) [meV] ====')
            print(f'{"X":2s} {"pair":10s} {"J_tot":>8s} {"J_H0":>8s} {"J_SCC":>8s} {"J_3rd":>8s} {"J_rep":>8s}')
            for c in CHEMS:
                if Js[c] is None:
                    continue
                for lab in PAIRLABS:
                    J = Js[c][lab]
                    print(f'{c:2s} {lab.replace("2H-", ""):10s} {J["tot"]*1000:8.1f} {J["H0"]*1000:8.1f} '
                          f'{J["SCC"]*1000:8.1f} {J["3rd"]*1000:8.1f} {J["rep"]*1000:8.1f}')
        print(f'\n==== r{w} rigid skeleton PHYSICAL split: J_rigid = J_noSCC + J_electrostatic [meV] ====')
        print(f'{"X":2s} {"pair":10s} {"J_rigid":>8s} {"J_noSCC":>8s} {"J_el":>8s}')
        for c in CHEMS:
            if rigid[c] is None or noscc[c] is None:
                continue
            for lab in PAIRLABS:
                Jr, Jn = rigid[c][lab]['tot'], noscc[c][lab]['tot']
                print(f'{c:2s} {lab.replace("2H-", ""):10s} {Jr*1000:8.1f} {Jn*1000:8.1f} {(Jr-Jn)*1000:8.1f}')
        d0 = None
        for c0 in CHEMS:                                     # sketches from first existing chem dir
            dd = os.path.join(OUTDIR, f'{tagf}_{c0}_r{w}')
            if os.path.isdir(dd):
                d0 = dd; break
        if d0 is None:
            continue
        fig = plt.figure(figsize=(16, 10.5))
        oslabs = [l for l in PAIRLABS if '-os-' in l]
        NP = len(PAIRLABS)
        NC = NP + len(oslabs) + 1                      # grid units per chem: main + os-zoom + gap
        gs = fig.add_gridspec(3, 3 * NC, height_ratios=[0.32, 1.3, 1.3], hspace=0.35,
                              wspace=0.15, left=0.05, right=0.97, top=0.93, bottom=0.06)
        e12 = np.linspace(0, 3 * NC, NP + 1).astype(int)
        for ip, lab in enumerate(PAIRLABS):
            ax = fig.add_subplot(gs[0, e12[ip]:e12[ip + 1]])
            p0 = os.path.join(d0, '0H', 'relaxed.xyz')
            p2 = os.path.join(d0, lab, 'relaxed.xyz')
            if os.path.isfile(p0) and os.path.isfile(p2):
                _sketch(ax, p0, p2)
            ax.set_title(lab.replace('2H-', ''), fontsize=10)

        def _fin(ax, labs, zoom=False):
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xticks(range(len(labs)))
            ax.set_xticklabels([l.replace('2H-', '') for l in labs], fontsize=7,
                               rotation=45, ha='right')
            if zoom:
                ax.set_facecolor('#f3f3f3')
                ax.set_title(f'os zoomed ×{(glo[1] - glo[0]) / (zg[1] - zg[0]):.1f}', fontsize=8)

        ax0 = az0 = None
        for ix, c in enumerate(CHEMS):                 # relaxed row (detailed.out terms)
            axm = fig.add_subplot(gs[1, ix * NC:ix * NC + NP], sharey=ax0); ax0 = ax0 or axm
            axz = fig.add_subplot(gs[1, ix * NC + NP:(ix + 1) * NC], sharey=az0); az0 = az0 or axz
            if relax[c] is not None:
                for ax_, labs in ((axm, PAIRLABS), (axz, oslabs)):
                    for ip, lab in enumerate(labs):
                        J = relax[c][lab]; bot = 0.0
                        for k in ('H0', 'SCC', '3rd', 'rep'):
                            v = J[k] * 1000
                            if not np.isfinite(v):
                                continue
                            ax_.bar(ip, v, bottom=bot, width=0.7, color=CC[k],
                                    label=k if (ix == 0 and ip == 0 and ax_ is axm) else None)
                            bot += v
                        ax_.plot(ip, J['tot'] * 1000, 'kD', ms=5)
            _fin(axm, PAIRLABS); _fin(axz, oslabs, zoom=True)
            axm.set_title(f'{c} chem — relaxed (detailed.out terms)', fontsize=9)
            if ix == 0:
                axm.set_ylabel('J [meV]'); axm.legend(fontsize=8, title='diamond=total')
                axz.set_ylabel('J [meV]')
            else:
                axm.tick_params(labelleft=False); axz.tick_params(labelleft=False)
        for ix, c in enumerate(CHEMS):                 # rigid row (physical split)
            axm = fig.add_subplot(gs[2, ix * NC:ix * NC + NP], sharey=ax0)
            axz = fig.add_subplot(gs[2, ix * NC + NP:(ix + 1) * NC], sharey=az0)
            if rigid[c] is not None and noscc[c] is not None:
                for ax_, labs in ((axm, PAIRLABS), (axz, oslabs)):
                    for ip, lab in enumerate(labs):
                        Jn = noscc[c][lab]['tot'] * 1000
                        Jr = rigid[c][lab]['tot'] * 1000
                        ax_.bar(ip, Jn, width=0.7, color='tab:blue',
                                label='noSCC (band+rep, SCC off)' if ix == 0 and ip == 0 and ax_ is axm else None)
                        ax_.bar(ip, Jr - Jn, bottom=Jn, width=0.7, color='tab:orange',
                                label='electrostatic (SCC resp.)' if ix == 0 and ip == 0 and ax_ is axm else None)
                        ax_.plot(ip, Jr, 'kD', ms=5)
            _fin(axm, PAIRLABS); _fin(axz, oslabs, zoom=True)
            axm.set_title(f'{c} chem — rigid skeleton (physical split)', fontsize=9)
            if ix == 0:
                axm.set_ylabel('J [meV]'); axm.legend(fontsize=8, title='diamond=J_rigid')
                axz.set_ylabel('J [meV]')
            else:
                axm.tick_params(labelleft=False); axz.tick_params(labelleft=False)
        if ax0 is not None:
            ax0.set_ylim(glo)                          # shared scale, same for all widths
        if az0 is not None:
            az0.set_ylim(zg)                           # shared zoomed scale (os columns)
        fig.suptitle(f'r{w} vac ribbon: J = E(2H)-E(1H-p)-E(1H-d)+E(0H)   —   top: relaxed cells, red rings = switched sites\n'
                     'mid: relaxed, detailed.out terms — CAUTION: "H0" incl. charge-response band shift | '
                     'bottom: rigid skeleton, noSCC vs electrostatic (= molecule convention)', fontsize=11)
        sfx = {'enumv': '', 'xscanv': '_xscan'}.get(tagf, '_' + tagf)
        png = os.path.join(OUTDIR, f"Jterms{sfx}_r{w}.png")
        fig.savefig(png, dpi=160)
        fig.savefig(png[:-4] + '.svg')
        plt.close(fig)
        print('REVIEW: ' + png)


def run_bands(args):
    """Band structure E(kx) of the relaxed 0H cell per width — dense kx mesh
    (nkx x nky MP grid, SP on the saved relaxed geometry), bands near EF,
    HOMO/LUMO gap annotated."""
    import matplotlib.pyplot as plt
    from spammm.quantum.DFTB_utils import run_pbc
    from spammm.quantum.pi_bond_order import read_band_out
    tag, builder = _enum_mode(args)
    nky = 1 if args.vac else 2
    fig, axs = plt.subplots(1, len(args.widths.split(',')), figsize=(4.5 * len(args.widths.split(',')), 5), squeeze=False)
    for iw, w in enumerate([int(x) for x in args.widths.split(',')]):
        wc = 2 * (w + 1)
        atoms, lvs, hb = builder(wc, args.ncells, '0' * args.ncells)
        d = np.load(os.path.join(OUTDIR, f'{tag}_r{w}_data.npz'))
        apos = d['apos_0H']
        wd = os.path.join(OUTDIR, f'bands_r{w}' if not args.vac else f'bands_{tag}_r{w}')
        E, _, _ = run_pbc(apos, list(atoms.enames), lvs, nk=(args.nkx, nky, 1), workdir=wd, Temperature=300,
                          Mixer='DIIS { Generations = 8 }', MaxScc=1000, SCCTolerance=1e-8, sk_set=args.sk,
                          extra_hsd='Analysis { WriteEigenvectors = Yes }')
        kw, eigs, fills = read_band_out(os.path.join(wd, 'band.out'))   # eigs in eV
        # MP mesh (ky-inner ordering as written by SupercellFolding): kx=(i+.5)/nkx, ky=(j)/nky
        nkp = len(kw)
        kf = np.array([[(i // nky + 0.5) / args.nkx, (i % nky) * (0.5 if nky == 2 else 0.0), 0.0] for i in range(nkp)])
        occ = fills > 1.9
        H, L = eigs[occ].max(), eigs[~occ].min()                       # eV
        ih = occ.sum(1)[0]                                           # n occupied bands
        dg = eigs[:, ih] - eigs[:, ih - 1]                           # direct gap per k
        kmin = int(dg.argmin())
        ax = axs[0, iw]
        for ky_i, ky_v in enumerate(sorted(set(kf[:, 1]))):
            mk = np.abs(kf[:, 1] - ky_v) < 1e-6
            xs = kf[mk, 0]
            srt = np.argsort(xs)
            ax.plot(xs[srt], eigs[mk][srt] - H, '-', lw=0.6, color=f'C{ky_i}')
        ax.plot([], [], '-', color='C0', label='ky=0.0')
        ax.plot([], [], '-', color='C1', label='ky=0.5')
        ax.axhline(0, color='r', ls='--', lw=0.8)
        ax.axhline(L - H, color='r', ls='--', lw=0.8)
        ax.axvline(kf[kmin, 0], color='g', ls=':', lw=0.8)
        ax.set_ylim(-4, 4)
        ax.set_title(f'r{w}: gap = {1000 * (L - H):.1f} meV  (nk={args.nkx}x{nky}, kmin={kf[kmin, 0]:.3f},{kf[kmin, 1]:.1f})', fontsize=9)
        ax.set_xlabel('kx [frac. recip.]')
        if iw == 0:
            ax.set_ylabel('E - E_HOMO [eV]')
        ax.legend(fontsize=7, loc='upper left', markerscale=3)
        print(f'r{w}: gap={1000 * (L - H):.2f} meV  HOMO={H:.3f}  LUMO={L:.3f}  mindirectgap={1000 * dg.min():.1f} meV at kf=({kf[kmin, 0]:.3f},{kf[kmin, 1]:.1f})')
    fig.tight_layout()
    png = os.path.join(OUTDIR, f'{tag}_bands.png')
    fig.savefig(png, dpi=140)
    print(f'REVIEW: {png}')


def run_unfold(args):
    """Unfold supercell band structure to primitive cell: spectral weight
    A(k_prim,E) from eigenvec.bin (per-orbital subcell Fourier projection).
    Reuses bands_r<w>/ dirs (must have eigenvec.bin — written by --bands).
    -> enumsj_unfold.png"""
    import matplotlib.pyplot as plt
    from spammm.quantum.pi_bond_order import read_band_out, read_eigenvec_bin, subcell_group_indices, unfold_spectral_weights_DEPRECATED
    widths = [int(x) for x in args.widths.split(',')]
    tag, builder = _enum_mode(args)
    nky = 1 if args.vac else 2
    fig, axs = plt.subplots(nky, len(widths), figsize=(5 * len(widths), 3.5 * nky), sharex=True, sharey=True, squeeze=False)
    for iw, w in enumerate(widths):
        wc = 2 * (w + 1)
        atoms, lvs, hb = builder(wc, args.ncells, '0' * args.ncells)
        wd = os.path.join(OUTDIR, f'bands_r{w}' if not args.vac else f'bands_{tag}_r{w}')
        kw, eigs, fills = read_band_out(os.path.join(wd, 'band.out'))
        nk, norb = eigs.shape
        C = read_eigenvec_bin(os.path.join(wd, 'eigenvec.bin'), nk, norb)
        IDX = subcell_group_indices(np.asarray(atoms.apos), lvs, list(atoms.enames), args.ncells)
        kf = np.array([(i // nky + 0.5) / args.nkx for i in range(nk)])
        kyv = np.array([(i % nky) * (0.5 if nky == 2 else 0.0) for i in range(nk)])
        q, W = unfold_spectral_weights_DEPRECATED(C, kf, IDX, args.ncells)
        occ = fills > 1.9
        H = eigs[occ].max()
        for ir, kv in enumerate([0.0, 0.5][:nky]):
            ax = axs[ir, iw]
            mk = np.abs(kyv - kv) < 1e-6
            qs = np.tile(q[mk][:, :, None], (1, 1, eigs.shape[1]))          # [k,q,band]
            es = np.tile(eigs[mk][:, None, :], (1, args.ncells, 1)) - H
            ws = W[mk]
            qs = np.where(qs > 0.5, 1 - qs, qs)                            # fold to primitive IBZ
            sel = (es > -3) & (es < 3)
            ax.scatter(qs[sel], es[sel], s=ws[sel] * 6, c=ws[sel], cmap='viridis', vmin=0, vmax=1, lw=0)
            ax.axhline(0, color='r', ls='--', lw=0.7)
            ax.set_xlim(0, 0.5)
            ax.set_title(f'r{w} ky={kv}  (unfolded to {args.ncells}x smaller cell)', fontsize=9)
            ax.set_xlabel('kx [primitive frac.]')
            if iw == 0:
                ax.set_ylabel('E - E_HOMO [eV]')
        print(f'r{w}: unfold done, mean sharpness (frac eigs w>0.8) = {(W.max(1) > 0.8).mean():.2f}')
    fig.tight_layout()
    png = os.path.join(OUTDIR, f'{tag}_unfold.png')
    fig.savefig(png, dpi=140)
    print(f'REVIEW: {png}')


def run_scan_ly(args):
    """Scan junction N...N distance d_DA (lattice-y optimization) for given states."""
    n = args.ncells
    states = args.states.split(',') if args.states else None
    gaps = np.linspace(args.lmin, args.lmax, args.npts)
    labels = {'p' * (2 * n): 'all-p', 'd' * (2 * n): 'all-d', 'pd' * n: 'alternating'}
    mode = 'relax' if args.relax else 'SP'
    for w in [int(x) for x in args.widths.split(',')]:
        res, fits, parity = scan_junction_gap(w, n, gaps, states=states, nk=(args.nk, args.nky, 1),
                                            relax=args.relax, workdir=os.path.join(OUTDIR, f'scanly_w{w}'))
        png = os.path.join(OUTDIR, f'scanly_w{w}_n{n}.png')
        plot_gap_scan(gaps, res, fits, labels=labels, savepath=png,
                      title=f'junction gap scan w{w} n{n} ({mode}, nk={args.nk}x{args.nky})')
        print(f"REVIEW: {png}")
        np.savez(png.replace('.png', '.npz'), gaps=gaps,
                 **{f"E_{labels.get(st, st)}": res[st] for st in res},
                 **{f"fit_{labels.get(st, st)}": np.array([fits[st][1], fits[st][2], 2 * fits[st][0][0]]) for st in fits})   # [d_opt, E_opt, k] per state


def main():
    ap = argparse.ArgumentParser(description='PBC zigzag ribbons with N-terminated edges')
    ap.add_argument('--widths', default='4,6,8', help='comma list of widths; for --enum/--enum-plot these are RING ROWS (w1=polyacene strip, chains=2*w+2); otherwise atom chains')
    ap.add_argument('--ncells', type=int, default=4, help='unit cells along x (default: 4; >=2)')
    ap.add_argument('--passivation', default='N', help="edge termination: 'N' (pyridinic), 'NH' (protonated), or passivation string e.g. 'NnNn'")
    ap.add_argument('--dftb', action='store_true', help='run DFTB+ PBC single point per ribbon (verification)')
    ap.add_argument('--nk', type=int, default=8, help='k-points along the ribbon axis x (default: 8)')
    ap.add_argument('--two', action='store_true', help='two-ribbon junction cell: bottom + top ribbons stacked along y with N...H-N interfaces')
    ap.add_argument('--bottom', default='N', help="--two: bottom-ribbon edge passivation (default 'N' = acceptor)")
    ap.add_argument('--top', default='NH', help="--two: top-ribbon edge passivation (default 'NH' = donor)")
    ap.add_argument('--dda', type=float, default=2.9, help='--two: N...N distance across junction [A] (default 2.9)')
    ap.add_argument('--state', default=None, help="--two: per-site state string, 2*ncells chars over {p,d,0} (p=H up, d=H down, 0=bare); '|' separates the two junctions")
    ap.add_argument('--shift-x', type=float, default=0.0, help='--two: lateral shift of top ribbon (fraction of Lx)')
    ap.add_argument('--no-pbc-y', action='store_true', help='--two: vacuum along y instead of periodic junction stack')
    ap.add_argument('--nky', type=int, default=1, help='--two --dftb: k-points along the stacking direction y (default: 1)')
    ap.add_argument('--scan-ly', action='store_true', help='--two: scan junction N...N distance (lattice y optimization) with DFTB+ SP')
    ap.add_argument('--states', default=None, help="--scan-ly: comma list of state strings (default: all-p, alternating pd.., all-d)")
    ap.add_argument('--lmin', type=float, default=2.4, help='--scan-ly: min N...N distance [A] (default 2.4)')
    ap.add_argument('--lmax', type=float, default=3.4, help='--scan-ly: max N...N distance [A] (default 3.4)')
    ap.add_argument('--npts', type=int, default=9, help='--scan-ly: number of scan points (default 9)')
    ap.add_argument('--relax', action='store_true', help='--scan-ly: ionic relax at each point (fixed cell) instead of SP')
    ap.add_argument('--enum', action='store_true', help='enumerate protonation states on a SINGLE junction (pbc_y=False): relax + pi-BO maps per state')
    ap.add_argument('--enum-plot', action='store_true', help='replot enum maps from saved results only (no recompute; recovers data npz from relaxed.xyz + Gamma DM once if missing)')
    ap.add_argument('--enum-diff', action='store_true', help='relative maps: d(bond length) and d(pi-BO) of each state vs relaxed 0H reference -> enumsj_r<w>_diffmaps.png')
    ap.add_argument('--nonlin', action='store_true', help='nonlinear residual maps X(2H)-X(1H_A)-X(1H_B) for X in {bpi,bl} -> <tag>_r<w>_nonlin.png')
    ap.add_argument('--compare', action='store_true', help='cross-chemistry vac comparison (C,N,O): shared-scale BL + BO map figures (abs & diff rows) + J breakdown -> cmp_{BL,BO,J}_r<w>.png')
    ap.add_argument('--vac', action='store_true', help='vacuum-y edge-switch cell instead of the self-junction stack (no H-bonds; --chem selects switch pair)')
    ap.add_argument('--chem', default='N', choices=['N', 'C', 'O'], help='--vac: edge switch pair N:N->NH, C:CH->CH2, O:C-OH->C=O (default N)')
    ap.add_argument('--xscan', action='store_true', help='--vac: separation scan instead of the 7-state enum — p fixed at site 0, second switch shifted by k cells (os-d0..d{n/2}, ss-d1..d{n/2}); use --ncells 8; tag xscanv_<chem>')
    ap.add_argument('--tagp', default=None, help="override dataset tag prefix (e.g. 'xscanz' for the z-constrained tilted-OH set); dirs/npz become <tagp>_<chem>_r<w>")
    ap.add_argument('--jobs', type=int, default=1, help='--enum: parallel states (process pool); OMP threads/job capped so jobs*threads <= 12 (3/4 of 16)')
    ap.add_argument('--vacy', type=float, default=12.0, help='--vac: y vacuum padding [A] (default 12)')
    ap.add_argument('--analyze', action='store_true', help='quantitative analysis from saved enum data: J couplings + pi-BO decay lengths -> debug/ribbon/enumsj_analysis.png')
    ap.add_argument('--bake-gpaw', default=None, metavar='DIR', help='bake GPAW LCAO/dzp/PBE cluster jobs (one dir per width x state) from relaxed.xyz; each job exports hs.npz (H/S/rho/C per k)')
    ap.add_argument('--gpaw-import', default=None, metavar='ROOT', help='import GPAW hs.npz bundle (ROOT/<tag>_r<w>/<lab>/hs.npz) into npz caches under debug/ribbon_gpaw/ + J tables')
    ap.add_argument('--gpaw-relax', action='store_true', help='--bake-gpaw: relax geometry in GPAW before the export (default: SP on DFTB-relaxed cell)')
    ap.add_argument('--gpaw-zfix', action='store_true', help='--bake-gpaw: FixCartesian z on all atoms during GPAW relax (xscanz tilted-OH set)')
    ap.add_argument('--geom-plot', action='store_true', help='geometry overview figure: widths x states, relaxed cells, protonated sites circled')
    ap.add_argument('--j-decomp', action='store_true', help='J decomposition waterfall (electronic/electrostatic/elastic) via rigid + noSCC SPs; caches to jdecomp_r<w>.npz')
    ap.add_argument('--j-terms', action='store_true', help='J = E(2H)-E(1H-p)-E(1H-d)+E(0H) decomposed into detailed.out terms (H0/SCC/3rd/rep), relaxed + rigid skeleton; no new DFTB runs -> Jterms_r<w>.png')
    ap.add_argument('--bands', action='store_true', help='band structure E(kx) of relaxed 0H per width (dense kx SP) -> enumsj_bands.png')
    ap.add_argument('--unfold', action='store_true', help='unfold bands to primitive cell (needs bands_r<w>/eigenvec.bin from --bands) -> enumsj_unfold.png')
    ap.add_argument('--nkx', type=int, default=64, help='--bands: k-points along x (default 64)')
    ap.add_argument('--sk', default='3ob-3-1', help='Slater-Koster set (default 3ob-3-1); non-default -> OUTDIR becomes debug/ribbon_<sk>/ so caches never mix')
    args = ap.parse_args()
    if args.sk != '3ob-3-1':
        global OUTDIR
        OUTDIR = OUTDIR.rstrip('/') + '_' + args.sk.split('-')[0]   # e.g. debug/ribbon_mio
        print(f'## SK set = {args.sk} -> OUTDIR = {OUTDIR}')
    os.makedirs(OUTDIR, exist_ok=True)
    if args.enum:
        run_enum(args)
        return
    elif args.enum_plot:
        run_enum_plot(args)
        return
    elif args.enum_diff:
        run_enum_diff(args)
        return
    elif args.nonlin:
        run_nonlin(args)
        return
    elif args.compare:
        run_compare(args)
        return
    elif args.analyze:
        run_analyze(args)
        return
    elif args.bake_gpaw:
        os.makedirs(args.bake_gpaw, exist_ok=True)
        bake_gpaw(args)
        return
    elif args.gpaw_import:
        run_gpaw_import(args)
        return
    elif args.geom_plot:
        run_geom_plot(args)
        return
    elif args.j_decomp:
        run_jdecomp(args)
        return
    elif args.j_terms:
        run_jterms(args)
        return
    elif args.bands:
        run_bands(args)
        return
    elif args.unfold:
        run_unfold(args)
        return
    if args.scan_ly:
        run_scan_ly(args)
        return
    if args.two:
        run_junction(args)
        return

    from collections import Counter
    for w in [int(x) for x in args.widths.split(',')]:
        name = f'zgnr_w{w}_n{args.ncells}_{args.passivation}'
        atoms, lvs, seam = build_ribbon_cell(w, args.ncells, passivation=args.passivation)
        print(f"\n=== {name}: natoms={atoms.natoms} {dict(Counter(atoms.enames))}  Lx={lvs[0,0]:.3f} A  seam_bonds={int(seam.sum())}/{len(atoms.bonds)} ===")
        check_degrees(atoms, name)
        png = os.path.join(OUTDIR, f'{name}.png')
        plot_ribbon_pbc_cell(atoms, lvs, seam, n_cells=3, savepath=png, title=f'{name}: {atoms.natoms} atoms/cell, Lx={lvs[0,0]:.2f} A')
        xyz = os.path.join(OUTDIR, f'{name}.xyz')
        save_xyz_lvs(xyz, atoms, lvs, name)
        print(f"  wrote {xyz}\nREVIEW: {png}")
        if args.dftb:
            from spammm.quantum.DFTB_utils import run_pbc
            E_ha, apos_out, forces = run_pbc(atoms.apos, atoms.enames, lvs, nk=(args.nk, 1, 1), workdir=os.path.join(OUTDIR, 'dftb_' + name), Temperature=300)
            print(f"  DFTB+ PBC SP ({args.nk} kpts): E = {E_ha*HAU2EV:.4f} eV   ({E_ha*HAU2EV/atoms.natoms:.3f} eV/atom)")


if __name__ == '__main__':
    main()
