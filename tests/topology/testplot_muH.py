#!/usr/bin/env python3
"""testplot_muH.py — per-site hydrogen chemical potentials mu_H(X -> XH) via gas-phase DFTB+.

Concept (doc/ERC_private/proof_of_concept.chat.md):
    mu1 = E(XH) - E(X);  mu2 = E(XH2) - E(XH);  J = mu2 - mu1 = E(XH2)+E(X)-2E(XH)
J is "the change of local hydrogen chemical potential caused by hydrogenating the
other site".  mu's are RELATIVE quantities: all element offsets cancel, so no H2
reservoir is needed to compare sites/molecules (H2 computed anyway as optional zero).

Stage 1 (monomers): BQ, NQ, Pyr, QX, Phz x {X, XH*, XH2} -> mu1, mu2, J.
    XH are doublets -> SpinPolarisation=Collinear{UnpairedElectrons=1} + mio
    shell-resolved SpinConstants (3ob ships none).  Restricted variant also run
    to estimate the restricted-vs-spin bias.  'XHb' variants = parity check.
    4-quinolone: lactam <-> lactim intramolecular transfer (both closed shell).
Stage 2 (dimers): finite ':' junction dimers; relax LL (H on donor) and RR (H on
    acceptor) with junction {D,A,H} pinned (the H pin is required — the
    transferred state otherwise ejects the proton to infinity) ->
    dE_direct = E_RR - E_LL vs dE_pred = mu1(acceptor mol) - mu2(donor mol);
    residual = H-bond/env term.  RR corners = radical pairs ->
    UnpairedElectrons=2, except qnol2 whose double PT is closed-shell
    (RR_SPIN_OVERRIDE).  Charged monomer variants XH-/XH+ (same XH geometry,
    closed shell) give DPE/PA -> the ionic proton-transfer prediction
    dE_PT = DPE(D) - PA(A), shown in transfer_calib.png next to the PCET one.
    _dissociated_atoms() post-relax check catches LBFGS ejecting atoms (the
    junction distance alone missed a C-H flown to 52 A).
Stage 3 (closed-shell 2H): the credible channel — XH2 + Y -> X + YH2 is
    closed-shell on BOTH ends, so the isolated monomers ARE the limiting
    structures and dE = mu2h(Y) - mu2h(X) needs no new DFTB.  Printed as a
    donor/acceptor matrix + transfer_2h_matrix.png heatmap + rxn2h_*.png
    didactic before/after panels (plotUtils.draw_mol_junctions style).

JUNCTION ART WARNING: ':' resolves to the atom whose CHARACTER column is
nearest (see _closest_in_row) — ' C N C' has atoms at cols 1,3,5 so ':' must
sit at col 3 to hit N; col 2 silently gives a C-H...C junction (~+4 eV
garbage that mimics a physics anomaly).  Check atoms.hbonds_ascii element
pairs before trusting a dimer number.

Hamiltonian: 3ob-3-1 requires the FULL DFTB3 model (skf docs: "no other level
of DFTB theory is recommended") — ThirdOrderFull + HubbardDerivs (published
3ob values; not in the .skf) + HCorrection=Damping{Exponent=4.0} (DampXH
zeta=4; this build predates the 'DampXH' name).  All patched into the .hsd
after write_dftb_input_relax() writes it (DFTB_utils.py is under other agents).

Artifacts: debug/muH/   Usage: python tests/topology/testplot_muH.py [--mols BQ,Pyr] [--dimers q+hq] [--figonly]
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm import atomicUtils as au
from spammm import elements as _elements
from spammm.topology.ascii_art_heterocycle import parse_ascii_art, _build_target_valence, resolve_hbond_pairs
from spammm.topology.KekulePure import make_n_pi
from spammm.topology.hbond_utils import HbondRecord, junction_bond_lengths
from spammm.quantum import DFTB_utils as DU

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'muH')
HAU2EV = 27.211386245988
RESULTS = os.path.join(OUTDIR, 'results.json')

# mio-style shell-resolved spin constants W [Ha]: {ss sp ps pp} (H: s only).
# 3ob .skf files carry no spin constants; these are the standard DFTB+ values.
SPIN_CONSTANTS = {'H': '-0.072',
                  'C': '-0.031 -0.025 -0.025 -0.023',
                  'N': '-0.033 -0.027 -0.027 -0.026',
                  'O': '-0.035 -0.030 -0.030 -0.028'}

# ---------------------------------------------------------------------------
# ASCII arts — junction sites: uppercase 'N'/'O' = acceptor, lowercase 'n'/'o'
# = donor (gets capping H).  Blocks copied from PBC_CHAIN_ARTS.
# ---------------------------------------------------------------------------
MONOMER_ARTS = {
    'BQ': {   # p-benzoquinone: 2 exocyclic O para on benzene
        'X':   "\n  O\n  C\n C C\n C C\n  C\n  O\n",
        'XH':  "\n  O\n  C\n C C\n C C\n  C\n  o\n",
        'XHb': "\n  o\n  C\n C C\n C C\n  C\n  O\n",
        'XH2': "\n  o\n  C\n C C\n C C\n  C\n  o\n",
    },
    'NQ': {   # naphthoquinone: exocyclic O at the rhombus apexes
        'X':   "\n   O\n   C\n  C C\n  C C\n C C\n C C\n  C\n  O\n",
        'XH':  "\n   O\n   C\n  C C\n  C C\n C C\n C C\n  C\n  o\n",
        'XH2': "\n   o\n   C\n  C C\n  C C\n C C\n C C\n  C\n  o\n",
    },
    'Pyr': {  # pyrazine: N at para ring apexes
        'X':   "\n  N\n C C\n C C\n  N\n",
        'XH':  "\n  N\n C C\n C C\n  n\n",
        'XHb': "\n  n\n C C\n C C\n  N\n",
        'XH2': "\n  n\n C C\n C C\n  n\n",
    },
    'QX': {   # quinoxaline (benzopyrazine): dimer-format art from qx2hqx
        'X':   "\n   N C\n  | | |\n   N C\n",
        'XH':  "\n   N C\n  | | |\n   n C\n",
        'XH2': "\n   n C\n  | | |\n   n C\n",
    },
    'Phz': {  # phenazine: N at central-ring apexes (5,10)
        'X':   "\n C N C\nC C C C\nC C C C\n C N C\n",
        'XH':  "\n C N C\nC C C C\nC C C C\n C n C\n",
        'XH2': "\n C n C\nC C C C\nC C C C\n C n C\n",
    },
    'Qnol': { # 4-quinolone: ONE mobile H, two sites (lactam <-> lactim)
        'lactam': "\n   O\n C C\nC C C\nC C C\n C n\n",
        'lactim': "\n   o\n C C\nC C C\nC C C\n C N\n",
    },
}
DOUBLET_STATES = {'XH', 'XHb'}          # odd-electron states -> spin polarized
MAIN_STATES = ('X', 'XH', 'XH2')

# acceptor block on top, ':' junction, donor block below.
# (art, acc_mol, don_mol): dE_pred = mu1(acc_mol) - mu2(don_mol)
# RR electronic state: single H transfer leaves a radical pair -> triplet (spin=2);
# qnol2 transfers BOTH junction H's (lactam->lactim on each) -> closed shell (spin=0),
# and its prediction is 2*dE_tautomer, not mu1-mu2.
# NOTE: blocks carry NO leading newline — ':' must sit on the row between the two atom rows.
RR_SPIN_OVERRIDE = {'qnol2': 0}   # qnol2 RR = 2 closed-shell lactim (double PT)
Q_BLK  = "  O\n  C\n C C\n C C\n  C\n  O\n"
HQ_BLK = "  o\n  C\n C C\n C C\n  C\n  o\n"
PYR_BLK   = "  N\n C C\n C C\n  N\n"
PYRH2_BLK = "  n\n C C\n C C\n  n\n"
PHZ_BLK   = " C N C\nC C C C\nC C C C\n C N C\n"
PHZH2_BLK = " C n C\nC C C C\nC C C C\n C n C\n"
DIMER_ARTS = {
    'q+hq':     (Q_BLK   + "  :\n" + HQ_BLK,   'BQ',  'BQ'),
    'pyr+hq':   (PYR_BLK + "  :\n" + HQ_BLK,   'Pyr', 'BQ'),   # cross-family O-H...N
    'pyr+pyrh2':(PYR_BLK + "  :\n" + PYRH2_BLK,'Pyr', 'Pyr'),
    'phz+phzh2':(PHZ_BLK + "   :\n" + PHZH2_BLK,'Phz', 'Phz'),  # ':' at col 3 = apex N;
        # col 2 is equidistant between C(col1) and N(col3) -> tie-break picks C (C-H...C junction, garbage)
    'qx+hqx':   ("   N C\n  | | |\n   N C\n   :\n C n\n| | |\n C n\n", 'QX', 'QX'),
    'qnol2':    (" C\nC C\nC C\n n O\n : :\n O n\n  C C\n  C C\n   C\n", 'Qnol', 'Qnol'),  # symmetric double transfer -> ~0
}


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def build_mol(art, hbond_length=None):
    """ASCII art -> AtomicSystem with capping H (same path as ascii_art_heterocycle.main)."""
    atoms = parse_ascii_art(art, hbond_length=hbond_length)
    atoms.neighs()
    n_pi0 = make_n_pi(atoms)
    tv = _build_target_valence(atoms, n_pi0)
    atoms.add_capping_h_sp2(target_valence=tv)
    eo = getattr(atoms, '_enames_original', None)
    if eo is not None and len(eo) < atoms.natoms:
        atoms._enames_original = list(eo) + ['H'] * (atoms.natoms - len(eo))
    atoms.neighs()
    return atoms


def check_degrees(atoms, name=''):
    """Fail-loud coordination check: H deg 1, heavy deg 2-3."""
    deg = np.zeros(atoms.natoms, dtype=int)
    for i, j in atoms.bonds:
        deg[i] += 1
        deg[j] += 1
    bad = [(i, e, int(d)) for i, e, d in zip(range(atoms.natoms), atoms.enames, deg)
           if (e == 'H' and d != 1) or (e != 'H' and not (1 <= d <= 3))]
    assert not bad, f"{name}: bad coordination {bad}"
    return deg


# ---------------------------------------------------------------------------
# DFTB relax (with optional spin block patched into the hsd)
# ---------------------------------------------------------------------------
# 3ob-3-1 requires the full DFTB3 model ("no other level of DFTB theory is
# recommended").  Plain SCC-DFTB on donor-acceptor pairs over-delocalizes
# charge (all dimer corners converge to +/- ~1 e ionic SCF branches);
# ThirdOrderFull + HubbardDerivs + DampXH is the published remedy.
HUBBARD_DERIVS = {'C': -0.1492, 'H': -0.1857, 'N': -0.1535, 'O': -0.1575}  # 3ob-3-1


def _patch_hsd(hsd_path, enames, unpaired=0, charge=0.0, extra_hsd='', dftb3=True):
    """Insert DFTB3 (ThirdOrderFull+HubbardDerivs+DampXH) and optional
    Charge/SpinPolarisation inside the 'Hamiltonian = DFTB { ... }' block
    (matched by brace depth — the SP writer appends an Options{} block after
    the Hamiltonian, so "last } of file" is NOT always the Hamiltonian).
    dftb3=False -> plain SCC-DFTB (mio etc. — ThirdOrder/DampXH are 3ob-only).
    extra_hsd: arbitrary extra Hamiltonian-level lines (e.g. Mixer override)."""
    txt = open(hsd_path).read()
    i = txt.index('Hamiltonian')
    depth, j = 0, txt.index('{', i)
    for k in range(j, len(txt)):
        depth += (txt[k] == '{') - (txt[k] == '}')
        if depth == 0:
            j = k
            break
    assert depth == 0, 'unbalanced braces in hsd'
    extra = ''
    if dftb3:
        hd = '\n'.join(f'    {s} = {HUBBARD_DERIVS[s]}' for s in sorted(set(enames)))
        extra += ('  ThirdOrderFull = Yes\n'
                  f'  HubbardDerivs {{\n{hd}\n  }}\n'
                  '  HCorrection = Damping {\n    Exponent = 4.0\n  }\n')  # = DampXH zeta=4 (3ob protocol; this build predates the 'DampXH' name)
    if charge:
        extra += f'  Charge = {charge}\n'
    if unpaired:
        sc = '\n'.join(f'    {s} = {SPIN_CONSTANTS[s]}' for s in sorted(set(enames)))
        extra += (f'  SpinPolarisation = Collinear {{ UnpairedElectrons = {unpaired} }}\n'
                  f'  SpinConstants {{\n    ShellResolvedSpin = Yes\n{sc}\n  }}\n')
    extra += extra_hsd
    with open(hsd_path, 'w') as f:
        f.write(txt[:j] + extra + txt[j:])


def relax_cell(enames, apos, workdir, spin=0, charge=0.0, fixed_atoms=None, filling_temp=300.0,
               scctol=1e-6, maxscc=500, rescue_temps=(600.0,), sk_set=None, grad_elem=1e-4):
    """Gas-phase DFTB+ relax -> (E_ha, apos_relaxed).  spin = # unpaired e (0=restricted).
    Rescue retries at higher Fermi T + Anderson mixer + larger maxscc (hot-T
    alone can 'converge' to a dissociated molecule — atoms fly apart until the
    forces vanish).  Post-check: every atom must keep a neighbor < 2.5 A,
    else RuntimeError (exploded geometry is never returned).
    sk_set: 'mio-1-1' etc -> plain SCC (no 3ob ThirdOrder/DampXH patch)."""
    os.makedirs(workdir, exist_ok=True)
    enames = list(enames)
    apos = np.asarray(apos, dtype=float)
    xyz = os.path.join(workdir, 'geom.xyz')
    hsd = os.path.join(workdir, 'dftb_in.hsd')
    sk_prefix = DU.get_sk_path(sk_set)
    d3 = (sk_set or DU.DEFAULT_SK_SET or '').startswith('3ob')
    au.save_xyz(xyz, enames, apos)
    DU.write_dftb_input_relax(enames, xyz, hsd, sk_prefix, scctol=scctol, maxscc=maxscc,
                              fixed_atoms=fixed_atoms, filling_temp=filling_temp, grad_elem=grad_elem)
    _patch_hsd(hsd, enames, unpaired=spin, charge=charge, dftb3=d3)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        ret = os.system(f'{DU.DFTB_EXE} > OUT 2> ERR')
        for T in rescue_temps:
            if ret == 0:
                break
            print(f'    ! relax failed in {workdir} -> retry Fermi T={T:.0f} K + Anderson mixer, maxscc={maxscc * 4}', flush=True)
            DU.write_dftb_input_relax(enames, xyz, hsd, sk_prefix, scctol=scctol, maxscc=maxscc * 4,
                                      fixed_atoms=fixed_atoms, filling_temp=T, grad_elem=grad_elem)
            _patch_hsd(hsd, enames, unpaired=spin, charge=charge, dftb3=d3,
                       extra_hsd='  Mixer = Anderson {\n    MixingParameter = 0.05\n    Generations = 8\n  }\n')
            ret = os.system(f'{DU.DFTB_EXE} > OUT 2> ERR')
        if ret != 0:
            raise RuntimeError(f'DFTB+ failed in {workdir}\n    {DU.dftb_failure_summary(workdir)}')
        E = DU.parse_energy_out('OUT')
        apos_out = DU.read_relaxed_geometry(apos, do_relax=True)
        d = np.linalg.norm(apos_out[:, None, :] - apos_out[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min(axis=1).max() > 2.5:
            raise RuntimeError(f'DFTB+ relax exploded in {workdir} (max nearest-neighbour distance {d.min(axis=1).max():.1f} A) — molecule dissociated')
        return E, apos_out
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Results cache (idempotent reruns)
# ---------------------------------------------------------------------------
def load_results():
    if os.path.isfile(RESULTS):
        return json.load(open(RESULTS))
    return {}


def save_results(res):
    with open(RESULTS, 'w') as f:
        json.dump(res, f, indent=1, sort_keys=True)


def cached_relax(res, key, enames, apos, workdir, **kw):
    """relax_cell with JSON cache; key -> {E_ha, apos}."""
    if key in res and np.isfinite(res[key]['E_ha']):
        return res[key]['E_ha'], np.asarray(res[key]['apos'])
    E, apos_out = relax_cell(enames, apos, workdir, **kw)
    res[key] = {'E_ha': float(E), 'apos': np.asarray(apos_out).tolist()}
    save_results(res)
    return E, apos_out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def draw_mol(ax, atoms, sz=80, hbonds=None):
    pos, en = np.asarray(atoms.apos), list(atoms.enames)
    if atoms.bonds is not None:
        for i, j in atoms.bonds:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'k-', lw=1.0, zorder=1)
    if hbonds:
        for hb in hbonds:
            ax.plot([pos[hb.donor_idx, 0], pos[hb.acceptor_idx, 0]],
                    [pos[hb.donor_idx, 1], pos[hb.acceptor_idx, 1]], 'g--', lw=1.0, zorder=1)
    heavy = [i for i, e in enumerate(en) if e != 'H']
    hs = [i for i, e in enumerate(en) if e == 'H']
    ax.scatter(pos[heavy, 0], pos[heavy, 1], s=sz, c=[_elements.ELEMENT_DICT[e][8] for e in np.array(en)[heavy]], zorder=2, linewidths=0)
    if hs:
        ax.scatter(pos[hs, 0], pos[hs, 1], s=sz * 0.35, c='0.7', zorder=2, linewidths=0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.margins(0.15)


def fig_build_check(built, dimers, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nm, nd = len(built), len(dimers)
    fig, axs = plt.subplots(1, nm + nd, figsize=(1.6 * (nm + nd), 2.2), squeeze=False)
    for k, (name, atoms, hbs) in enumerate(list(built) + list(dimers)):
        draw_mol(axs[0, k], atoms, hbonds=hbs)
        axs[0, k].set_title(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_mu_ladder(table, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mols = [m for m in table if 'mu1' in table[m]]
    fig, ax = plt.subplots(figsize=(1.5 * len(mols) + 1.5, 3.2))
    for i, m in enumerate(mols):
        mu1, mu2 = table[m]['mu1'], table[m]['mu2']
        E0 = 0.0
        ax.plot([i - .3, i + .3], [E0, E0], 'k-', lw=2)
        ax.plot([i - .3, i + .3], [mu1, mu1], 'b-', lw=2)
        ax.plot([i - .3, i + .3], [mu1 + mu2, mu1 + mu2], 'r-', lw=2)
        for lvl, c in [(E0, 'k'), (mu1, 'b'), (mu1 + mu2, 'r')]:
            ax.plot([i, i], [lvl, lvl], '_', color=c)
        ax.plot([i, i], [E0, mu1], ':', color='b', lw=0.8)
        ax.plot([i, i], [mu1, mu1 + mu2], ':', color='r', lw=0.8)
        ax.text(i + .33, mu1, f'{mu1:+.2f}', fontsize=7, va='center', color='b')
        ax.text(i + .33, mu1 + mu2, f'{mu1+mu2:+.2f}', fontsize=7, va='center', color='r')
        ax.text(i + .33, (mu1 + mu2) / 2, f'J={table[m]["J"]:+.2f}', fontsize=7, va='center')
    ax.set_xticks(range(len(mols)))
    ax.set_xticklabels(mols)
    ax.set_ylabel('E rel. to X  [eV]  (level i = E(XH_i)-E(X))')
    ax.set_title('hydrogenation ladder: X (black) -> XH (blue) -> XH2 (red)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_transfer_calib(rows, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labs = [r['name'] for r in rows]
    pred = np.array([r['dE_pred'] for r in rows])
    dire = np.array([r['dE_direct'] for r in rows])
    pt   = np.array([r.get('dE_pred_pt', np.nan) for r in rows])
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(1.3 * len(labs) + 2.0, 3.2))
    ax.bar(x - 0.27, pred, 0.27, label='predicted mu1(A)-mu2(D) [PCET]')
    ax.bar(x,        dire, 0.27, label='direct dimer relax')
    ax.bar(x + 0.27, pt,   0.27, label='predicted DPE(D)-PA(A) [PT ionic]', color='#999999')
    for xi, (p, d) in enumerate(zip(pred, dire)):
        if np.isfinite(p) and np.isfinite(d):
            ax.text(xi, max(p, d) + 0.05, f'res {d - p:+.2f}', ha='center', fontsize=7)
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel('dE(LL->RR) [eV]')
    ax.set_title('junction H transfer: monomer-mu prediction vs direct dimer')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 1: monomer table
# ---------------------------------------------------------------------------
def run_monomers(res, mol_filter=None, figonly=False):
    table = {}
    built = []
    for mol, states in MONOMER_ARTS.items():
        if mol_filter and mol not in mol_filter:
            continue
        if mol == 'Qnol':
            continue                      # handled separately (tautomer pair)
        row = {}
        for st in MAIN_STATES + ('XHb',):
            if st not in states:
                continue
            if st == 'XHb' and mol not in ('BQ', 'Pyr'):   # parity check on 2 mols only
                continue
            art = states[st]
            atoms = build_mol(art)
            check_degrees(atoms, f'{mol}/{st}')
            if st == 'X':
                built.append((f'{mol} X', atoms, None))
            spin = 1 if st in DOUBLET_STATES else 0
            key = f'mol/{mol}/{st}/spin{spin}'
            if not figonly:
                E, apos_r = cached_relax(res, key, atoms.enames, atoms.apos,
                                         os.path.join(OUTDIR, 'mol', mol, st), spin=spin)
                row[st] = E * HAU2EV
                # restricted comparison for doublets: bias estimate
                if spin:
                    key_r = f'mol/{mol}/{st}/spin0'
                    Er, _ = cached_relax(res, key_r, atoms.enames, atoms.apos,
                                         os.path.join(OUTDIR, 'mol', mol, st + '_restr'), spin=0)
                    row[st + '_restr'] = Er * HAU2EV
            # charged XH variants (same geometry) -> proton-transfer channel:
            #   XHm = XH2 - H+ (donor anion),  XHp = X + H+ (acceptor cation)
            if st == 'XH' and not figonly:
                for chg, tag in ((-1.0, 'XHm'), (1.0, 'XHp')):
                    Ec, _ = cached_relax(res, f'mol/{mol}/{tag}/spin0', atoms.enames, atoms.apos,
                                         os.path.join(OUTDIR, 'mol', mol, tag), spin=0, charge=chg)
                    row[tag] = Ec * HAU2EV
            print(f'  {mol:5s} {st:4s}: natom={atoms.natoms} E={row.get(st, float("nan")):.4f} eV', flush=True)
        table[mol] = row

    # Qnol tautomer pair (closed shell both)
    if not mol_filter or 'Qnol' in mol_filter:
        row = {}
        for st, art in MONOMER_ARTS['Qnol'].items():
            atoms = build_mol(art)
            check_degrees(atoms, f'Qnol/{st}')
            if not figonly:
                E, _ = cached_relax(res, f'mol/Qnol/{st}/spin0', atoms.enames, atoms.apos,
                                    os.path.join(OUTDIR, 'mol', 'Qnol', st), spin=0)
                row[st] = E * HAU2EV
            print(f'  Qnol  {st:7s}: natom={atoms.natoms} E={row.get(st, float("nan")):.4f} eV', flush=True)
        if 'lactam' in row and 'lactim' in row:
            row['dE_tautomer'] = row['lactim'] - row['lactam']
        table['Qnol'] = row
        built.append(('Qnol lactam', build_mol(MONOMER_ARTS['Qnol']['lactam']), None))

    # H2 reservoir (optional absolute zero)
    if not figonly:
        h2_xyz = np.array([[0., 0., 0.], [0., 0., 0.74]])
        E_H2, _ = cached_relax(res, 'mol/H2/X/spin0', ['H', 'H'], h2_xyz,
                               os.path.join(OUTDIR, 'mol', 'H2'), spin=0)
        table['H2'] = {'X': E_H2 * HAU2EV, 'mu_H': 0.5 * E_H2 * HAU2EV}
        print(f'  H2: E={E_H2*HAU2EV:.4f} eV  (mu_H^0 = {0.5*E_H2*HAU2EV:.4f} eV)', flush=True)

    # derived quantities
    for mol, row in table.items():
        if all(k in row for k in MAIN_STATES):
            row['mu1'] = row['XH'] - row['X']
            row['mu2'] = row['XH2'] - row['XH']
            row['mu2h'] = row['XH2'] - row['X']       # full 2H affinity (closed shell)
            row['J'] = row['mu2'] - row['mu1']
            if 'XHb' in row:
                row['parity_meV'] = 1000 * abs(row['XHb'] - row['XH'])
            if 'XH_restr' in row:
                row['spin_bias_meV'] = 1000 * (row['XH_restr'] - row['XH'])
            if 'XHm' in row:
                row['DPE'] = row['XHm'] - row['XH2']     # deprotonation XH2 -> XH- + H+
                row['PA'] = row['X'] - row['XHp']        # proton affinity X + H+ -> XH+
    return table, built


def print_table(table):
    print('\n==== monomer hydrogen chemical potentials (eV; lower = more favourable H uptake) ====')
    print(f'{"mol":6s} {"E(X)":>10s} {"mu1":>8s} {"mu2":>8s} {"J=mu2-mu1":>10s} {"parity":>8s} {"spinbias":>9s} {"DPE":>8s} {"PA":>8s}')
    for mol, row in table.items():
        if 'mu1' not in row:
            continue
        print(f'{mol:6s} {row["X"]:10.3f} {row["mu1"]:8.3f} {row["mu2"]:8.3f} {row["J"]:10.3f} '
              f'{row.get("parity_meV", float("nan")):8.2f} {row.get("spin_bias_meV", float("nan")):9.1f} '
              f'{row.get("DPE", float("nan")):8.3f} {row.get("PA", float("nan")):8.3f}')
    if 'Qnol' in table and 'dE_tautomer' in table['Qnol']:
        print(f'Qnol lactam->lactim: dE = {table["Qnol"]["dE_tautomer"]:+.3f} eV')
    if 'H2' in table:
        print(f'mu_H^0 = E(H2)/2 = {table["H2"]["mu_H"]:+.3f} eV')


# ---------------------------------------------------------------------------
# Closed-shell 2H transfer:  XH2 + Y -> X + YH2
# Both endpoints closed-shell -> no radical/ionic channel ambiguity, and the
# limiting structures are the isolated monomers themselves, so the additive
# prediction dE = mu2h(Y) - mu2h(X) IS the full answer for separated species
# (junction environment adds only the residual measured in the dimers).
# ---------------------------------------------------------------------------
def _mu2h(row):
    """Full 2H affinity E(XH2)-E(X); cached rows may predate the 'mu2h' key."""
    if 'mu2h' in row:
        return row['mu2h']
    if 'XH2' in row and 'X' in row:
        return row['XH2'] - row['X']
    return None


def print_2h_matrix(table):
    mols = [m for m, r in table.items() if _mu2h(r) is not None]
    print('\n==== closed-shell 2H transfer  dE = mu2h(acceptor) - mu2h(donor)  [eV] ====')
    print('        donor XH2 -> Y    (negative = downhill)')
    hdr = 'donor\\acc ' + ' '.join(f'{m:>7s}' for m in mols)
    print(hdr)
    for xd in mols:
        print(f'{xd:>9s} ' + ' '.join(f'{_mu2h(table[y]) - _mu2h(table[xd]):+7.3f}' for y in mols))


def fig_2h_matrix(table, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mols = [m for m, r in table.items() if _mu2h(r) is not None]
    n = len(mols)
    dE = np.array([[_mu2h(table[y]) - _mu2h(table[x]) for y in mols] for x in mols])
    v = np.nanmax(np.abs(dE))
    fig, ax = plt.subplots(figsize=(0.9 * n + 2.2, 0.8 * n + 1.6))
    im = ax.imshow(dE, cmap='RdBu_r', vmin=-v, vmax=v)
    ax.set_xticks(range(n), mols); ax.set_yticks(range(n), mols)
    ax.set_xlabel('acceptor Y'); ax.set_ylabel('donor XH$_2$')
    ax.set_title('2H transfer  XH$_2$+Y$\\rightarrow$X+YH$_2$   $\\Delta E=\\mu_{2H}(Y)-\\mu_{2H}(X)$ [eV]', fontsize=10)
    for i in range(n):
        for j in range(n):
            ax.annotate(f'{dE[i, j]:+.2f}', (j, i), ha='center', va='center', fontsize=9,
                        color='k' if abs(dE[i, j]) < 0.7 * v else 'w')
    fig.colorbar(im, ax=ax, label='dE [eV]', shrink=0.8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


RXN2H_PAIRS = [('BQ', 'Pyr'), ('BQ', 'Phz'), ('BQ', 'NQ'), ('Pyr', 'Phz'), ('QX', 'Phz'), ('Pyr', 'QX')]
H_NAMES = {'BQ': ('p-benzoquinone', 'hydroquinone'), 'NQ': ('naphthoquinone', 'dihydronaphthoquinone'),
           'Pyr': ('pyrazine', '1,4-dihydropyrazine'), 'QX': ('quinoxaline', '1,2-dihydroquinoxaline'),
           'Phz': ('phenazine', '5,10-dihydrophenazine')}


def _reactive_hs(atoms):
    """Indices of capping H's sitting on N/O apex atoms (the transferable H's)."""
    hs = set()
    for i, j in atoms.bonds:
        for h, x in ((i, j), (j, i)):
            if atoms.enames[h] == 'H' and atoms.enames[x] in ('N', 'O'):
                hs.add(h)
    return sorted(hs)


def fig_2h_reaction(res, table, don_mol, acc_mol, fname):
    """Didactic before->after panel:  donor XH2 + Y  ->  X + YH2  (all relaxed geometries)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spammm.plotUtils import draw_mol_junctions
    dE = _mu2h(table[acc_mol]) - _mu2h(table[don_mol])
    species = [(don_mol, 'XH2'), (acc_mol, 'X'), (don_mol, 'X'), (acc_mol, 'XH2')]
    labels = [H_NAMES[don_mol][1], H_NAMES[acc_mol][0], H_NAMES[don_mol][0], H_NAMES[acc_mol][1]]
    fig, axs = plt.subplots(1, 7, figsize=(13.5, 3.6), gridspec_kw={'width_ratios': [1, 0.12, 1, 0.22, 1, 0.12, 1]})
    mol_axs = [axs[0], axs[2], axs[4], axs[6]]
    for ax, (mol, st), lab in zip(mol_axs, species, labels):
        atoms = build_mol(MONOMER_ARTS[mol][st])
        apos = np.asarray(res[f'mol/{mol}/{st}/spin0']['apos'])
        draw_mol_junctions(ax, atoms, apos, [], sz=70)
        hs = _reactive_hs(atoms)
        if hs:   # green rings on the transferable H's (junction convention)
            ax.scatter(apos[hs, 0], apos[hs, 1], s=300, facecolors='none', edgecolors='lime', linewidths=1.6, zorder=6)
        ax.annotate(lab, (0.5, 0.02), xycoords='axes fraction', fontsize=9.5, weight='bold', ha='center', va='bottom',
                    color='#2222aa' if st == 'X' else '#aa5500')
    for ax, t in ((axs[1], '+'), (axs[5], '+')):
        ax.annotate(t, (0.5, 0.5), fontsize=18, ha='center', va='center'); ax.axis('off')
    axs[3].annotate('→', (0.5, 0.58), fontsize=26, ha='center', va='center')
    axs[3].annotate(f'ΔE = {dE:+.2f} eV', (0.5, 0.38), fontsize=10.5, ha='center', va='center')
    axs[3].axis('off')
    fig.suptitle(f'{H_NAMES[don_mol][1]} + {H_NAMES[acc_mol][0]}  →  {H_NAMES[don_mol][0]} + {H_NAMES[acc_mol][1]}', fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 2: dimer calibration
# ---------------------------------------------------------------------------
def _dissociated_atoms(atoms, apos, hbs):
    """Bonded pairs stretched past threshold -> atoms that escaped during relax.
    The junction (donor,H) pair is whitelisted: in RR it is broken BY DESIGN
    (the H now sits on the acceptor)."""
    bonds = np.asarray(atoms.bonds); en = np.asarray(atoms.enames)
    apos = np.asarray(apos)
    whitelist = {(hb.donor_idx, hb.h_idx) for hb in hbs} | {(hb.h_idx, hb.donor_idx) for hb in hbs}
    d = np.linalg.norm(apos[bonds[:, 0]] - apos[bonds[:, 1]], axis=1)
    bad = set()
    for (i, j), dd in zip(bonds, d):
        thr = 1.6 if ('H' in (en[i], en[j])) else 2.0
        if dd > thr and (i, j) not in whitelist:
            bad.update((int(i), int(j)))
    return bad


def run_dimers(res, table, dimer_filter=None, figonly=False):
    rows = []
    built = []
    for name, (art, acc_mol, don_mol) in DIMER_ARTS.items():
        if dimer_filter and name not in dimer_filter:
            continue
        atoms = build_mol(art, hbond_length=2.9)
        check_degrees(atoms, f'dim/{name}')
        resolve_hbond_pairs(atoms)
        hbs = [HbondRecord(next(j for j in atoms.ngs[ih] if atoms.enames[j] != 'H'), ih, ia)
               for ih, ia in atoms.hbonds_ascii]
        built.append((name, atoms, hbs))
        print(f'\n=== {name}: {atoms.natoms} atoms, {len(hbs)} junction(s) ===', flush=True)
        # corner-scan pin base: junction heteroatoms AND the scan H (else the
        # transferred state ejects the junction H).  Post-relax bond-integrity
        # check catches OTHER atoms that LBFGS ejects to infinity (observed:
        # a phz C-H stretched to 52 A -> garbage energy); on violation expand
        # the pin set and rerun BOTH corners so the constraint is symmetric.
        pins_base = {i for hb in hbs for i in (hb.donor_idx, hb.acceptor_idx, hb.h_idx)}
        pins_extra = set()
        corners = {}
        for attempt in range(3):
            for corner in ('LL', 'RR'):
                apos_g = np.asarray(atoms.apos, float).copy()
                if corner == 'RR':
                    for hb in hbs:
                        pD, pA = apos_g[hb.donor_idx], apos_g[hb.acceptor_idx]
                        axv = (pA - pD) / np.linalg.norm(pA - pD)
                        apos_g[hb.h_idx] = pA - 1.01 * axv
                pins = sorted(pins_base | pins_extra)
                key = f'dim/{name}/{corner}'
                spin = RR_SPIN_OVERRIDE.get(name, 2) if corner == 'RR' else 0
                if not figonly:
                    if attempt > 0:
                        res.pop(key, None)          # old result was dissociated -> recompute
                    try:
                        E, apos_r = cached_relax(res, key, atoms.enames, apos_g,
                                                 os.path.join(OUTDIR, 'dim', name, corner),
                                                 fixed_atoms=pins, spin=spin)
                    except RuntimeError as ex:
                        print(f'  {corner}: RELAX FAILED {ex}', flush=True)
                        E, apos_r = np.nan, apos_g
                    corners[corner] = (E, apos_r)
            if figonly:
                break
            bad = set()
            for corner in ('LL', 'RR'):
                E, apos_r = corners[corner]
                if np.isfinite(E):
                    bad |= _dissociated_atoms(atoms, apos_r, hbs)
            if not bad:
                break
            pins_extra |= bad
            print(f'    ! dissociated atoms {sorted(bad)} -> expand pins, retry (attempt {attempt + 2})', flush=True)
        for corner in ('LL', 'RR'):
            if corner in corners:
                E, apos_r = corners[corner]
                bl = junction_bond_lengths(apos_r, hbs)
                bstr = '  '.join(f'DH={a:.2f} HA={b:.2f}' for a, b in bl)
                corners[corner] = {'E': E * HAU2EV if np.isfinite(E) else np.nan, 'apos': apos_r, 'bl': bl}
                print(f'  {corner}: E={corners[corner]["E"]:.4f} eV   {bstr}', flush=True)
        if not figonly and all(c in corners for c in ('LL', 'RR')):
            dE_dir = corners['RR']['E'] - corners['LL']['E']
            dE_pred = np.nan
            if name == 'qnol2' and 'dE_tautomer' in table.get('Qnol', {}):
                dE_pred = 2.0 * table['Qnol']['dE_tautomer']   # LL = 2 lactam, RR = 2 lactim
            elif 'mu1' in table.get(acc_mol, {}) and 'mu2' in table.get(don_mol, {}):
                dE_pred = table[acc_mol]['mu1'] - table[don_mol]['mu2']
            # PT channel (D- ... H-A+): deprotonation of donor + protonation of acceptor
            dE_pred_pt = np.nan
            if 'DPE' in table.get(don_mol, {}) and 'PA' in table.get(acc_mol, {}):
                dE_pred_pt = table[don_mol]['DPE'] - table[acc_mol]['PA']
            print(f'  >> dE_direct={dE_dir:+.3f} eV   dE_pred(mu1({acc_mol})-mu2({don_mol}))={dE_pred:+.3f} eV   residual={dE_dir - dE_pred:+.3f} eV   dE_pred_PT={dE_pred_pt:+.3f} eV', flush=True)
            rows.append({'name': name, 'dE_direct': dE_dir, 'dE_pred': dE_pred, 'dE_pred_pt': dE_pred_pt})
        elif figonly:
            rows.append({'name': name, 'dE_direct': np.nan, 'dE_pred': np.nan})
    return rows, built


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='site hydrogen chemical potentials via DFTB+')
    ap.add_argument('--mols', default=None, help='comma list of monomers (default all)')
    ap.add_argument('--dimers', default=None, help='comma list of dimers (default all)')
    ap.add_argument('--skip-dimers', action='store_true')
    ap.add_argument('--skip-mols', action='store_true')
    ap.add_argument('--figonly', action='store_true', help='only rebuild figures from cache')
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    res = load_results()

    mol_filter = args.mols.split(',') if args.mols else None
    dim_filter = args.dimers.split(',') if args.dimers else None

    table, built_m = ({}, [])
    if not args.skip_mols:
        print('########## stage 1: monomer relaxes ##########', flush=True)
        table, built_m = run_monomers(res, mol_filter, figonly=args.figonly)
        if not args.figonly:
            print_table(table)
            res['mu_table'] = table
            save_results(res)
    else:
        table = res.get('mu_table', {})
    if not args.figonly:
        print_2h_matrix(table)

    rows, built_d = ([], [])
    if not args.skip_dimers:
        print('\n########## stage 2: dimer calibration ##########', flush=True)
        rows, built_d = run_dimers(res, table, dim_filter, figonly=args.figonly)

    if built_m or built_d:
        png1 = os.path.join(OUTDIR, 'build_check.png')
        fig_build_check(built_m, built_d, png1)
        print(f'\nREVIEW: {png1}', flush=True)
    if not args.figonly:
        png2 = os.path.join(OUTDIR, 'mu_ladder.png')
        fig_mu_ladder(table, png2)
        print(f'REVIEW: {png2}', flush=True)
        png4 = os.path.join(OUTDIR, 'transfer_2h_matrix.png')
        fig_2h_matrix(table, png4)
        print(f'REVIEW: {png4}', flush=True)
        for don, acc in RXN2H_PAIRS:
            if don in table and acc in table and _mu2h(table.get(don, {})) is not None and _mu2h(table.get(acc, {})) is not None:
                png5 = os.path.join(OUTDIR, f'rxn2h_{don.lower()}_{acc.lower()}.png')
                fig_2h_reaction(res, table, don, acc, png5)
                print(f'REVIEW: {png5}', flush=True)
        if rows:
            png3 = os.path.join(OUTDIR, 'transfer_calib.png')
            fig_transfer_calib(rows, png3)
            print(f'REVIEW: {png3}', flush=True)
        res['dimer_calib'] = rows
        save_results(res)


if __name__ == '__main__':
    main()
