#!/usr/bin/env python3
"""testplot_mol_flakes_dftb.py — DFTB battery on the PAH flake set
(doc/ERC_private/task_Molecules.md, §6 steps 2-5).

Reuses the builders of testplot_mol_flakes.py (defs-only exec) and the
DFTB3+spin hsd patcher of testplot_muH.py. Nothing shared is modified.

BATTERY (--group bat): 9 backbones (BASES + EXT of testplot_mol_flakes.py)
x 3 site chemistries x 5 states = 135 fixed-skeleton SP + 27 triplet checks.
  chemistries on the two marked central sites (TIPCHEM lo->hi chars):
    C:  =CH- -> -CH2-     N: =N- -> -NH-     O: -C(=O)- keto -> =C(OH)- enol
  states: 0H both lo | 2H both hi (+ _T triplet) | 1Hr top hi (q0 doublet)
          | 1Hp top hi q+1 | 1Hm top hi q-1
  NB: for O the lo state (keto) is the QUINONIC tautomer and hi (enol) is
  AROMATIC — opposite of C/N.  The bond maps below use the flipped
  aromatic-first convention (ARO=enol, QUI=keto); the energy battery keeps
  the original low-H -> high-H labelling (0H=keto for O).
Caches: results.json (SP energies+spin), sp/<key>/, relax_bat/<key>/ +
results_relax.json (relaxed geometries for bond maps), bo/<key>.npz.

ENERGY PLOTS (--group bat):
  battery_report.png    per-backbone bar chart + molecule thumbnail
  battery_lines.png     hydrogenation ladder 0H-1H-2H per chemistry
  battery_reservoir.png grand potential with electron reservoir:
                        Omega(q)=E(q)-q*mu_e, mu_e=-phi; phi_Au=5.3,
                        phi_Gr=4.6 eV; one polyline 0H->1H@Au->1H@Gr->2H
  H reservoir: mu_H = E(H2)/2 = -9.123 eV (DFTB ref; D(H-H)=3.30 vs 4.75 exp)

BOND MAPS (--maps NAME|all): maps_<variant>/bo_<name>.png + bl_<name>.png
(+.svg), per backbone: rows = C/N/O, cols = Aromatic | Quinonic | Qui-Aro |
1H*-Aro | 1H+-Aro | 1H--Aro — every diff is (final state) - (aromatic ref)
on the final state's geometry/bond list.  BO = pi bond orders from the
DFTBcore SP density matrix (Lowdin pi projection, pi_bond_order.py);
BL = relaxed-geometry bond lengths (heavy-heavy bonds only).  Absolute
panels share a clipped viridis norm; diffs share symmetric coolwarm;
delta computed on shared bonds only (added/removed bonds drawn grey).
NOTE: BL deltas need relaxed geoms — SP skeletons are identical by
construction.

BASIS/TOLERANCE VARIANTS (--sk SET --tight): map caches + figures get a
variant tag suffix so parameterizations coexist for comparison:
  --sk mio-1-1 --tight -> bo_miot/, relax_bat_miot/, results_relax_miot.json,
                         bo_<name>_miot.png  (mio = plain SCC, no 3ob
                         ThirdOrder/DampXH; patched automatically)
  --sk 3ob-3-1 --tight -> *_3obt
  --tight = SP scctol 1e-10, relax scctol 1e-9 + GradElem 1e-6 (standard:
            1e-7 / 1e-6 / 1e-4)
Validation result (see task_Molecules.md): 3ob loose->tight is converged
(BO identical <1e-4, BL <0.3 mA); 3ob->mio gives max dBO 0.04 / dBL 0.02 A
systematic shift, ARO-QUI pattern correlation >= 0.994 — maps are robust.

Legacy groups: --group flake/mol/ref (small-molecule ladders, muH refs,
spin validation phenalenyl/triangulene), --export-pyscf DIR.

Usage:
    python tests/topology/testplot_mol_flakes_dftb.py [--group G] [--states k,k] [--maps all] [--sk mio-1-1] [--tight]
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# defs + data import of testplot_mol_flakes (battery section is pure dict
# construction — no builds; main() is __name__-guarded so exec is safe)
_FL = os.path.join(os.path.dirname(__file__), 'testplot_mol_flakes.py')
_FNS = {'__file__': _FL, '__name__': 'mol_flakes_defs'}
exec(compile(open(_FL).read(), _FL, 'exec'), _FNS)
build_ring_flake = _FNS['build_ring_flake']
build_rect_flake = _FNS['build_rect_flake']
edge_sites = _FNS['edge_sites']
site_str = _FNS['site_str']
sublattice = _FNS['sublattice']
FLAKES = _FNS['FLAKES']
BASES = _FNS['BASES']
EXT = _FNS['EXT']
_build_one = _FNS['build_one']
build_art_flake = _FNS['build_art_flake']
build_patch_flake = _FNS['build_patch_flake']
profile_rings = _FNS['profile_rings']
draw_flake = _FNS['draw_flake']
formula = _FNS['formula']
central_sites = _FNS['central_sites']

import testplot_muH as M
from spammm.quantum import DFTB_utils as DU
from spammm.topology.ascii_art_heterocycle import _rhombus_patch, ASCII_EXAMPLES

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'mol_flakes_dftb')
RESULTS = os.path.join(OUTDIR, 'results.json')
HAU2EV = 27.211386245988
Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}

NROW, NCOL = 2, 4   # smallest flake: 2x4 fused rings (~perylene+), zigzag top/bot


def n_electrons(enames, charge=0.0):
    return int(round(sum(Z[e.split('_')[0]] for e in enames) - charge))


def sp_cell(enames, apos, workdir, spin=0, charge=0.0, filling_temp=100.0, scctol=1e-6, maxscc=500):
    """Fixed-skeleton SP under DFTB3 protocol -> (E_ha, spin_moments).
    One rescue retry at 300 K on failure; raises RuntimeError after."""
    os.makedirs(workdir, exist_ok=True)
    enames = list(enames)
    apos = np.asarray(apos, dtype=float)
    xyz = os.path.join(workdir, 'geom.xyz')
    hsd = os.path.join(workdir, 'dftb_in.hsd')
    sk_prefix = DU.get_sk_path()
    import spammm.atomicUtils as au
    au.save_xyz(xyz, enames, apos)
    ret = -1
    for T in (filling_temp, 300.0):
        DU.write_dftb_input_sp(enames, xyz, hsd, sk_prefix, scctol=scctol, maxscc=maxscc, filling_temp=T)
        M._patch_hsd(hsd, enames, unpaired=spin, charge=charge)
        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            ret = os.system(f'{DU.DFTB_EXE} > OUT 2> ERR')
        finally:
            os.chdir(cwd)
        if ret == 0:
            break
        print(f'    ! SP failed in {workdir} (T={T:.0f} K) -> retry', flush=True)
    if ret != 0:
        raise RuntimeError(f'DFTB+ failed in {workdir}\n    {DU.dftb_failure_summary(workdir)}')
    E = DU.parse_energy_out(os.path.join(workdir, 'OUT'))
    m = parse_spin_moments(os.path.join(workdir, 'detailed.out'))
    return E, m


def parse_spin_moments(detailed_path):
    """Per-atom Mulliken spin populations from detailed.out (spin runs only).

    Looks for 'Atom populations (up)' / '(down)' blocks; returns list m_i =
    up_i - down_i, or None for restricted runs."""
    if not os.path.isfile(detailed_path):
        return None
    blocks = {}
    cur = None
    for ln in open(detailed_path):
        if 'Atom populations (up)' in ln:
            cur = 'up'; blocks[cur] = []; continue
        if 'Atom populations (down)' in ln:
            cur = 'dn'; blocks[cur] = []; continue
        if cur and 'l-shell' in ln:
            cur = None; continue
        if cur:
            tok = ln.split()
            if len(tok) == 2 and tok[0].isdigit():
                blocks[cur].append(float(tok[1]))
    up, dn = blocks.get('up'), blocks.get('dn')
    if up and dn and len(up) == len(dn):
        return (np.asarray(up) - np.asarray(dn)).tolist()
    return None


def cached(res, key, fn):
    if key in res:
        return res[key]
    out = fn()
    res[key] = out
    os.makedirs(OUTDIR, exist_ok=True)
    with open(RESULTS, 'w') as f:
        json.dump(res, f, indent=1, sort_keys=True)
    return out


# ---------------------------------------------------------------- job table

_T0 = build_rect_flake(NROW, NCOL)
_BOTS, _TOPS, _ = edge_sites(_T0)
_C, _CB = len(_TOPS) // 2, len(_BOTS) // 2   # central zigzag sites, top/bottom
print(f'flake {NROW}x{NCOL}: {len(_TOPS)} top / {len(_BOTS)} bot zigzag sites, switch at {_C}/{_CB}')


def flake(base_top='N', top_sw=None, bot_sw=None, base_bot=None):
    """(nrow,ncol) flake with per-edge-site chemistry chars
    ('C' edge CH, 'c' sp3 CH2, 'N' bare pyridinic N, 'n' protonated N-H)."""
    top = site_str(len(_TOPS), base_top, top_sw or {})
    bot = site_str(len(_BOTS), base_bot or base_top, bot_sw or {})
    return build_rect_flake(NROW, NCOL, bot=bot, top=top)


def jobs():
    """(key, atoms_builder, spin, charge)."""
    c, cb = _C, _CB
    j = [
        ('phenalenyl',    lambda: build_ring_flake([(0, 0), (1, 0), (0, 1)]),                              1, 0.0),
        ('phenalenyl_S0', lambda: build_ring_flake([(0, 0), (1, 0), (0, 1)]),                              0, 0.0),
        ('triangulene_S', lambda: build_ring_flake([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]),      0, 0.0),
        ('triangulene_T', lambda: build_ring_flake([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]),      2, 0.0),
        ('fl_CH',         lambda: build_rect_flake(NROW, NCOL),                                            0, 0.0),
        ('fl_N',          lambda: flake('N'),                                                            0, 0.0),
        ('N_1H',          lambda: flake('N', {c: 'n'}),                                                  1, 0.0),
        ('N_1Hp',         lambda: flake('N', {c: 'n'}),                                                  0, +1.0),
        ('N_1Hm',         lambda: flake('N', {c: 'n'}),                                                  0, -1.0),
        ('N_2Hs_S',       lambda: flake('N', {c: 'n', c + 1: 'n'}),                                      0, 0.0),
        ('N_2Hs_T',       lambda: flake('N', {c: 'n', c + 1: 'n'}),                                      2, 0.0),
        ('N_2Ho_S',       lambda: flake('N', {c: 'n'}, {cb: 'n'}),                                       0, 0.0),
        ('N_2Ho_T',       lambda: flake('N', {c: 'n'}, {cb: 'n'}),                                       2, 0.0),
        ('C_1CH2',        lambda: flake('C', {c: 'c'}),                                                  1, 0.0),
        ('C_2CH2s_S',     lambda: flake('C', {c: 'c', c + 1: 'c'}),                                      0, 0.0),
        ('C_2CH2s_T',     lambda: flake('C', {c: 'c', c + 1: 'c'}),                                      2, 0.0),
        ('C_2CH2o_S',     lambda: flake('C', {c: 'c'}, {cb: 'c'}),                                       0, 0.0),
        ('C_2CH2o_T',     lambda: flake('C', {c: 'c'}, {cb: 'c'}),                                       2, 0.0),
    ]
    return j


# ------------------------------------------------------------- small mols
# Two-TIP switch test (mirrors the flake N_2Hs vs N_2Ho protocol): both apex
# atoms carry the base chemistry ('NN' = N passivation on BOTH sides), then
# hydrogenate ONE tip only ('nN' = 1H radical) or BOTH tips ('nn' = 2H,
# closed shell) — and likewise 'cC'/'cc' for CH2 site removal.  Charge states
# q=0,-1 (+1 where useful); nn/cc get singlet AND triplet (annihilation test).
#
# Channels derived per molecule:
#   H* 1-side : E(nN,q0)  - E(NN,q0)        PCET radical affinity (one tip)
#   2H 2-side : E(nn,S,q0)- E(NN,q0)        closed-shell double hydrogenation
#   annihilation gain: E(nn,S)+E(NN)-2*E(nN)  <0 => radicals pair up profitably
#   E_T-E_S   : nn/cc triplet vs singlet (Lieb signature of the two-tip geometry)

# tip chars: 'C' =CH- | 'c' sp3 -CH2- | 'N' pyridinic | 'n' -NH- |
#            'O' exocyclic =O (quinone) | 'o' exocyclic -OH (hydroquinone)
# 'O'/'o' atoms go one row BEYOND the apex C (exocyclic), apex keeps 'C'.
MOLPAIRS = ['CC', 'cC', 'cc', 'NN', 'nN', 'nn', 'OO', 'oO', 'oo']   # (top, bot)
MOLS = {'benzene': (1, 1), 'biphenyl': None, 'pyrene': (2, 2), 'phenanthrene': None}


def _rasterize(blk):
    """(col,k)->char patch -> ASCII lines (same convention as make_acene_chain_art)."""
    c0 = 1 - min(c for c, _ in blk)
    lines = []
    for k in sorted({k for _, k in blk}, reverse=True):
        row = {c: ch for (c, kk), ch in blk.items() if kk == k}
        lines.append(''.join(row.get(c - c0, ' ') for c in range(0, max(row) + c0 + 1)))
    return '\n'.join(lines)


def fix_sp3_caps(atoms):
    """add_capping_h_sp2 stacks both caps at the same in-plane spot on sp3 'c'
    sites (2 heavy nbrs, 2 identical H) -> spread tetrahedrally above/below plane."""
    en = [str(e) for e in atoms.enames]
    nbr = {}
    for i, j in atoms.bonds:
        nbr.setdefault(i, []).append(j)
        nbr.setdefault(j, []).append(i)
    for i, e in enumerate(en):
        hs = [j for j in nbr.get(i, []) if en[j] == 'H']
        hv = [j for j in nbr.get(i, []) if en[j] != 'H']
        if e == 'H' or not (len(hs) == 2 and len(hv) == 2):
            continue
        if np.linalg.norm(atoms.apos[hs[0]] - atoms.apos[hs[1]]) > 0.1:
            continue
        v = sum((atoms.apos[j] - atoms.apos[i]) / np.linalg.norm(atoms.apos[j] - atoms.apos[i]) for j in hv)
        b = v / np.linalg.norm(v)
        for k, s in zip(hs, (+1, -1)):
            atoms.apos[k] = atoms.apos[i] + (-0.5 * b + s * (np.sqrt(3) / 2) * np.array([0., 0., 1.])) * 1.09
    return atoms


def _apply_tip(lines, apex_idx, ch, where):
    """Switch apex atom in an ASCII-art line list: 'Oo' = exocyclic row beyond
    the apex (apex stays C); other chars replace the apex char in place."""
    i = apex_idx
    c = lines[i].index('C')
    if ch in 'Oo':
        row = ' ' * c + ch
        if where == 'top':
            lines.insert(i, row)
        else:
            lines.insert(i + 1, row)
    else:
        lines[i] = lines[i].replace('C', ch, 1)
    return lines


MOL_ARTS = {'biphenyl': 'biphenyl2', 'phenanthrene': 'phenanthrene2'}   # single-apex-atom tip arts


def build_switch_mol(mol, top='C', bot='C'):
    """Small molecule with BOTH apex/tip sites switched -> AtomicSystem."""
    if mol in MOL_ARTS:
        lines = [l for l in ASCII_EXAMPLES[MOL_ARTS[mol]].split('\n')]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        it = next(i for i, l in enumerate(lines) if 'C' in l)          # top apex
        lines = _apply_tip(lines, it, top, 'top')
        ib = next(i for i in range(len(lines) - 1, -1, -1) if 'C' in lines[i])  # bottom apex
        lines = _apply_tip(lines, ib, bot, 'bot')
        return fix_sp3_caps(M.build_mol('\n'.join(lines)))
    nr, nc = MOLS[mol]
    patch, (ct, kt), (cb, kb) = _rhombus_patch(nr, nc)
    for (c, k), ch, sgn in [((ct, kt), top, +2), ((cb, kb), bot, -2)]:
        if ch in 'Oo':
            patch[(c, k + sgn)] = ch            # exocyclic =O / -OH past the apex
        else:
            patch[(c, k)] = ch
    return fix_sp3_caps(M.build_mol(_rasterize(patch)))


def mol_jobs():
    """(key, builder, q, spin) — spin explicit for the nn/cc S-vs-T pairs."""
    out = []
    for mol in MOLS:
        for pair in MOLPAIRS:
            tp, bp = pair
            for q in (0.0, -1.0, +1.0):
                out.append((f'{mol}_{pair}_q{q:+.0f}',
                            (lambda mol=mol, tp=tp, bp=bp: build_switch_mol(mol, tp, bp)), q, None))  # spin auto = Ne parity
            if pair in ('cc', 'nn', 'oo'):                    # closed-shell 2-tip: S vs T
                out.append((f'{mol}_{pair}_q+0_T',
                            (lambda mol=mol, tp=tp, bp=bp: build_switch_mol(mol, tp, bp)), 0.0, 2))
    return out


# ---------------------------------------------------------------- flake battery
# The 9 symmetric backbones (BASES + EXT, no triangles) x 3 tip chemistries
# x state ladder.  States per chemistry (lo/hi tip chars):
#   0H   both tips lo        q=0  closed shell
#   2H   both tips hi        q=0  closed shell  (+ _T triplet variant)
#   1Hr  top tip hi only     q=0  radical doublet
#   1Hp  top tip hi only     q=+1 closed-shell cation  (H+ channel)
#   1Hm  top tip hi only     q=-1 closed-shell anion   (H- channel)
# => 6 jobs per chemistry x 3 x 9 = 162 single points.

BACKBONES = list(BASES) + list(EXT)
TIPCHEM = {'C': ('C', 'c'), 'N': ('N', 'n'), 'O': ('O', 'o')}   # lo, hi tip chars


def flake_tip(name, top_ch='C', bot_ch='C'):
    """Backbone with chars applied to the two marked central edge sites."""
    spec = FLAKES[name]
    a0 = _build_one(name, spec)
    bots, tops, _ = edge_sites(a0)
    bot = site_str(len(bots), 'C', {len(bots) // 2: bot_ch})
    top = site_str(len(tops), 'C', {len(tops) // 2: top_ch})
    if spec[0] == 'art':
        return build_art_flake(spec[1], bot=bot, top=top)
    return build_patch_flake(spec[1], bot=bot, top=top)


def battery_jobs():
    """(key, builder, spin, q) — same convention as jobs()."""
    out = []
    for name in BACKBONES:
        for X, (lo, hi) in TIPCHEM.items():
            out += [
                (f'{name}_{X}_0H',  (lambda n=name, lo=lo, hi=hi: flake_tip(n, lo, lo)),   0, 0.0),
                (f'{name}_{X}_2H',  (lambda n=name, lo=lo, hi=hi: flake_tip(n, hi, hi)),   0, 0.0),
                (f'{name}_{X}_2H_T',(lambda n=name, lo=lo, hi=hi: flake_tip(n, hi, hi)),   2, 0.0),
                (f'{name}_{X}_1Hr', (lambda n=name, lo=lo, hi=hi: flake_tip(n, hi, lo)),   1, 0.0),
                (f'{name}_{X}_1Hp', (lambda n=name, lo=lo, hi=hi: flake_tip(n, hi, lo)),   0, +1.0),
                (f'{name}_{X}_1Hm', (lambda n=name, lo=lo, hi=hi: flake_tip(n, hi, lo)),   0, -1.0),
            ]
    return out


def fig_battery_report(res, path):
    """9-panel summary: per backbone, per chemistry bar chart of the
    hydrogenation channels relative to the 0H state:
      d1Hr = E(1H*,q0) - E(0H) - mu_H      radical (PCET) affinity
      d1Hp = E(1H,q+1) - E(0H) - mu_H      cationic (H+) channel*
      d1Hm = E(1H,q-1) - E(0H) - mu_H      anionic  (H-) channel*
      d2H  = E(2H,q0)  - E(0H) - E(H2)     closed-shell double
    (* the ionic channels miss the H+/H- reservoir offset — only relative
    trends across backbones are meaningful.)
    Annihilation gains (cooperativity of the 2-site switch) printed to stdout.
    """
    import matplotlib.pyplot as plt
    E = lambda k: (res[k]['E_ha'] or float('nan')) * HAU2EV if k in res and res[k]['E_ha'] is not None else float('nan')
    muH = E('ref_H2') / 2
    sts = [('1Hr', '1H*'), ('1Hp', '1H+'), ('1Hm', '1H-'), ('2H', '2H')]
    fig, axs = plt.subplots(3, 6, figsize=(17, 10), gridspec_kw={'width_ratios': [0.30, 1.0] * 3})
    for ib, name in enumerate(BACKBONES):
        r, c = ib // 3, 2 * (ib % 3)
        axm = axs[r, c]                          # molecule thumbnail
        atoms0 = _build_one(name, FLAKES[name])
        draw_flake(axm, atoms0, sz=36., highlight=list(central_sites(atoms0)))
        axm.set_title(name, fontsize=10)
        ax = axs[r, c + 1]
        for j, (X, col) in enumerate(zip(TIPCHEM, ('tab:blue', 'tab:red', 'tab:green'))):
            xs = np.arange(len(sts)) + (j - 1) * 0.26
            for i, (st, _) in enumerate(sts):
                dE = E(f'{name}_{X}_{st}') - E(f'{name}_{X}_0H') - (muH if st != '2H' else 2 * muH)
                ax.bar(xs[i], dE, width=0.24, color=col)
            t_s = E(f'{name}_{X}_2H_T') - E(f'{name}_{X}_2H')
            if np.isfinite(t_s):
                ylim = ax.get_ylim()
                ax.text(0.98, 0.98 - 0.07 * j, f'{X} T-S {1000 * t_s:+.0f}meV',
                        fontsize=6.5, ha='right', va='top', color=col, transform=ax.transAxes)
        ax.axhline(0, c='k', lw=0.7)
        ax.set_xticks(range(len(sts)))
        ax.set_xticklabels([l for _, l in sts], fontsize=8)
        ax.tick_params(labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('tab:blue', 'tab:red', 'tab:green')]
    fig.legend(handles, [f'{X}: {TIPCHEM[X][0]}/{TIPCHEM[X][1]}' for X in TIPCHEM], loc='lower right', fontsize=9)
    fig.suptitle('Flake battery: hydrogenation channels dE = E(state)-E(0H)-n*mu_H [eV]\n(2H bar annotated with E_T-E_S)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print('REVIEW: ' + path)


def fig_battery_lines(res, path):
    """Hydrogenation-ladder variant of the battery report: x = nH (0,1,2),
    y = E(state)-E(0H)-n*mu_H; per chemistry one colour (C green, N blue,
    O red), the three 1H charge channels as line styles (* solid, + dashed,
    - dotted).  Faint chord 0H--2H: points above the chord midpoint mean the
    1H intermediate is skipped (disproportionation / cooperativity)."""
    import matplotlib.pyplot as plt
    E = lambda k: (res[k]['E_ha'] or float('nan')) * HAU2EV if k in res and res[k]['E_ha'] is not None else float('nan')
    muH = E('ref_H2') / 2
    CHEMCOL = {'C': 'tab:green', 'N': 'tab:blue', 'O': 'tab:red'}
    fig, axs = plt.subplots(3, 6, figsize=(17, 10), gridspec_kw={'width_ratios': [0.30, 1.0] * 3})
    for ib, name in enumerate(BACKBONES):
        r, c = ib // 3, 2 * (ib % 3)
        axm = axs[r, c]
        atoms0 = _build_one(name, FLAKES[name])
        draw_flake(axm, atoms0, sz=36., highlight=list(central_sites(atoms0)))
        axm.set_title(name, fontsize=10)
        ax = axs[r, c + 1]
        for X, col in CHEMCOL.items():
            e0, e2 = E(f'{name}_{X}_0H'), E(f'{name}_{X}_2H')
            d2 = e2 - e0 - 2 * muH
            ax.plot([0, 2], [0, d2], '-', color=col, lw=0.7, alpha=0.4)            # chord = cooperative reference
            for st, ls in (('1Hr', '-'), ('1Hp', '--'), ('1Hm', ':')):
                d1 = E(f'{name}_{X}_{st}') - e0 - muH
                ax.plot([0, 1, 2], [0, d1, d2], ls, color=col, lw=1.3,
                        marker='o', ms=3.5, markevery=[1])
        ax.axhline(0, c='k', lw=0.7)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['0H', '1H', '2H'], fontsize=8)
        ax.set_xlim(-0.25, 2.25)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([0], [0], color=c, lw=1.5) for c in CHEMCOL.values()] + \
              [plt.Line2D([0], [0], color='k', ls=ls, lw=1.2) for ls in ('-', '--', ':')]
    fig.legend(handles, [f'{X}: {TIPCHEM[X][0]}/{TIPCHEM[X][1]}' for X in CHEMCOL] + ['1H* q0', '1H+ q+1', '1H- q-1'],
               loc='lower right', fontsize=8, ncol=2)
    fig.suptitle('Flake battery: hydrogenation ladders E(nH)-E(0H)-n·mu_H [eV]\n'
                 '(1H point above the 0H-2H chord => intermediate skipped = cooperative switch)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print('REVIEW: ' + path)


def fig_battery_reservoir(res, path):
    """Ladder with electron-reservoir correction of the charged 1H states:
    Omega(q) = E(q) - q*mu_e,  mu_e = -phi (work function) ->
    dE(1H+) -> dE - phi_electrode,  dE(1H-) -> dE + phi_electrode.
    x positions: 0H | 1H@Au (phi=5.3 eV) | 1H@Gr (phi=4.6 eV) | 2H.
    1H* (neutral radical) is reservoir-independent -> same value at both
    middle columns.  Markers: filled = Au column, open = graphene column;
    line style = charge channel as in fig_battery_lines."""
    import matplotlib.pyplot as plt
    PHI = {'Au': 5.3, 'Gr': 4.6}            # work functions [eV]: Au(111), graphene Dirac
    E = lambda k: (res[k]['E_ha'] or float('nan')) * HAU2EV if k in res and res[k]['E_ha'] is not None else float('nan')
    muH = E('ref_H2') / 2
    CHEMCOL = {'C': 'tab:green', 'N': 'tab:blue', 'O': 'tab:red'}
    fig, axs = plt.subplots(3, 6, figsize=(17, 10), gridspec_kw={'width_ratios': [0.30, 1.0] * 3})
    for ib, name in enumerate(BACKBONES):
        r, c = ib // 3, 2 * (ib % 3)
        axm = axs[r, c]
        atoms0 = _build_one(name, FLAKES[name])
        draw_flake(axm, atoms0, sz=36., highlight=list(central_sites(atoms0)))
        axm.set_title(name, fontsize=10)
        ax = axs[r, c + 1]
        for X, col in CHEMCOL.items():
            e0 = E(f'{name}_{X}_0H'); d2 = E(f'{name}_{X}_2H') - e0 - 2 * muH
            dr = E(f'{name}_{X}_1Hr') - e0 - muH                    # reservoir-independent
            ax.plot([0, 1, 2, 3], [0, dr, dr, d2], '-', color=col, lw=1.4, marker='o', ms=3.5, markevery=[1, 2])
            for st, ls, sgn in (('1Hp', '--', -1.0), ('1Hm', ':', +1.0)):
                d1 = E(f'{name}_{X}_{st}') - e0 - muH
                ax.plot([0, 1, 2, 3], [0, d1 + sgn * PHI['Au'], d1 + sgn * PHI['Gr'], d2],
                        ls, color=col, lw=1.1, marker='o', ms=3.5, markevery=[1, 2],
                        mfc='none')                                   # open markers at both middle cols
                ax.plot([1], [d1 + sgn * PHI['Au']], 'o', ms=3.5, color=col)
            ax.plot([0, 3], [0, d2], '-', color=col, lw=0.6, alpha=0.35)          # chord
        ax.axhline(0, c='k', lw=0.7)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['0H', '1H@Au', '1H@Gr', '2H'], fontsize=8)
        ax.set_xlim(-0.25, 3.25)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([0], [0], color=c, lw=1.5) for c in CHEMCOL.values()] + \
              [plt.Line2D([0], [0], color='k', ls=ls, lw=1.2) for ls in ('-', '--', ':')] + \
              [plt.Line2D([0], [0], color='k', marker='o', ls='', ms=4, mfc='k'),
               plt.Line2D([0], [0], color='k', marker='o', ls='', ms=4, mfc='none')]
    fig.legend(handles, [f'{X}: {TIPCHEM[X][0]}/{TIPCHEM[X][1]}' for X in CHEMCOL] +
               ['1H* q0', '1H+ q+1', '1H- q-1', '@Au phi=5.3 (filled)', '@Gr phi=4.6 (open)'],
               loc='lower right', fontsize=7.5, ncol=2)
    fig.suptitle('Flake battery: hydrogenation ladders with electron reservoir correction [eV]\n'
                 'charged 1H states shifted by work function: 1H+ -phi, 1H- +phi', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print('REVIEW: ' + path)


# ---------------------------------------------------------------- bond maps
# Per-backbone BO (pi bond order, SP density matrix) and BL (bond length,
# relaxed geometry) maps: rows = C/N/O chemistry, cols = aromatic | quinonic |
# diff(Aro-Qui) | mixed(*) mixed(+) mixed(-).  Aromatic reference: =CH-, =N-,
# =C(OH)- (enol); quinonic: -CH2-, -NH-, -C(=O)- (keto).  NOTE the O flip —
# for O the ENOL is aromatic and the KETO is quinonic (opposite of C/N).

AROQUI = {'C': ('C', 'c'), 'N': ('N', 'n'), 'O': ('o', 'O')}   # (aromatic, quinonic) tip chars
MAP_STATES = [('ARO', 0, 0.0), ('QUI', 0, 0.0),               # both sites
              ('M1r', 1, 0.0), ('M1p', 0, +1.0), ('M1m', 0, -1.0)]  # top=quinonic

# basis/tolerance variant tag for the map caches+figures ('' = 3ob standard)
SK_SET = None     # None -> DU.DEFAULT_SK_SET (3ob-3-1); 'mio-1-1' etc via --sk
TIGHT = False     # --tight: scctol 1e-10 (SP/DM) + GradElem 1e-6, scctol 1e-9 (relax)
SK_TAG = ''
RESR = os.path.join(OUTDIR, 'results_relax.json')


def bat_state_atoms(name, X, st):
    """(atoms, spin, q) for backbone `name`, chemistry X, map-state tag."""
    aro, qui = AROQUI[X]
    top, bot = {'ARO': (aro, aro), 'QUI': (qui, qui),
                'M1r': (qui, aro), 'M1p': (qui, aro), 'M1m': (qui, aro)}[st]
    spin, q = {s: (sp, qq) for s, sp, qq in MAP_STATES}[st]
    return flake_tip(name, top, bot), spin, q


def run_bo(atoms, spin, q, npz):
    """DFTBcore SP -> Lowdin pi-BO array; cached to npz (spin/q patched hsd)."""
    from spammm.quantum.pi_bond_order import ao_layout, pi_project, lowdin_orthogonalize, \
        bond_orders_from_pmat, mol_plane_normal
    if os.path.isfile(npz):
        return np.load(npz)['bo']
    from spammm.quantum.DFTB.DFTBcore import DFTBcore
    import spammm.atomicUtils as au
    wd = npz[:-4]
    os.makedirs(wd, exist_ok=True)
    enames = [str(e).split('_')[0] for e in atoms.enames]
    xyz = os.path.join(wd, 'geom.xyz')
    hsd = os.path.join(wd, 'dftb_in.hsd')
    au.save_xyz(xyz, enames, atoms.apos)
    sk = SK_SET or DU.DEFAULT_SK_SET
    DU.write_dftb_input_sp(enames, xyz, hsd, DU.get_sk_path(sk), filling_temp=100.0,
                           scctol=1e-10 if TIGHT else 1e-7, maxscc=500)
    M._patch_hsd(hsd, enames, unpaired=spin, charge=q, dftb3=sk.startswith('3ob'))
    cwd = os.getcwd()
    try:
        os.chdir(wd)
        with DFTBcore() as d:
            d.init('dftb_in.hsd')
            d.enable_matrix_collection(dm=True, h=False, s=True)
            d.run_scf()
            P, S = d.get_dm_dense(), d.get_s_dense()
    finally:
        os.chdir(cwd)
    lay = ao_layout(enames)
    Ppi = pi_project(P, lay['pxyz'], mol_plane_normal(atoms.apos), lay['pi_atoms'])
    Spi = pi_project(S, lay['pxyz'], mol_plane_normal(atoms.apos), lay['pi_atoms'])
    bo = bond_orders_from_pmat(lowdin_orthogonalize(Ppi, Spi), lay['pi_atoms'], atoms.bonds)
    np.savez(npz, bo=bo)
    return bo


def bat_bo(name, X, st):
    """pi bond orders over skeleton bonds via DFTBcore SP (spin/q patched hsd).
    Cached to bo<SK_TAG>/<key>.npz."""
    key = f'{name}_{X}_{st}'
    npz = os.path.join(OUTDIR, 'bo' + SK_TAG, key + '.npz')
    atoms, spin, q = bat_state_atoms(name, X, st)
    return atoms, run_bo(atoms, spin, q, npz)


def run_relax(atoms, spin, q, key, resr, resr_path):
    """DFTB+ relax -> relaxed-geometry AtomicSystem (skeleton bonds kept).
    Cached in resr_path json + workdir relax_<key>."""
    import spammm.atomicUtils as au
    wd = os.path.join(OUTDIR, 'relax_bat' + SK_TAG, key)
    geom = os.path.join(wd, 'geom.out.xyz')
    if key not in resr or not os.path.isfile(geom):
        enames = [str(e).split('_')[0] for e in atoms.enames]
        try:
            E, _ = M.relax_cell(enames, atoms.apos, wd, spin=spin, charge=q,
                                scctol=1e-9 if TIGHT else 1e-6, maxscc=500,
                                grad_elem=1e-6 if TIGHT else 1e-4, sk_set=SK_SET)
            resr[key] = {'E_ha': E}
        except RuntimeError as e:
            print(f'  !! relax {key}: FAILED — {str(e).splitlines()[0]}', flush=True)
            resr[key] = {'E_ha': None, 'failed': str(e).splitlines()[0]}
        with open(resr_path, 'w') as f:
            json.dump(resr, f, indent=1, sort_keys=True)
    if os.path.isfile(geom):
        xyzs, _Zs, enames, _qs, _ = au.load_xyz(geom)
        out = xyz_mol(enames, xyzs)
        out.bonds = atoms.bonds          # keep skeleton bond topology (no re-detect)
        return out
    return atoms                          # failed relax -> skeleton


def bat_relax_geom(name, X, st, resr):
    """Relaxed geometry + energy; cached in results_relax<SK_TAG>.json + relax_bat<SK_TAG>/<key>."""
    key = f'{name}_{X}_{st}'
    atoms, spin, q = bat_state_atoms(name, X, st)
    return run_relax(atoms, spin, q, key, resr, RESR)


# ---------------------------------------------------------------------------
# EDGE-SITE SCAN (--sites): every zigzag site on BOTH edges carries chemistry X
# (aromatic =CH-, =N-, -C(OH)-), then one top + one bottom site are switched to
# quinonic (-CH2-, -NH-, -C(=O)-).  All states closed-shell, q=0.
# Site indices 1..3 = x-sorted apex sites per edge (2 = mirror-axis centre).
# Unique site pairs under both mirrors: q11 (same side), q13 (opposite sides),
# q12 (edge+centre), q22 (both centre).
# ---------------------------------------------------------------------------
SITE_FLAKES = {'tap343': [3, 4, 3], 'tap34343': [3, 4, 3, 4, 3]}   # 3-layer + 5-layer same width
ARO_CH = {'C': 'C', 'N': 'N', 'O': 'o'}        # aromatic site chars
QUI_CH = {'C': 'c', 'N': 'n', 'O': 'O'}        # quinonic site chars
SITE_CONF = ['AROall', 'q11', 'q12', 'q13', 'q22']
SINGLE_CONF = ['s1', 's2']          # top-only single switches (bottom = mirror)
RESS = os.path.join(OUTDIR, 'results_site.json')


def site_atoms(name, X, conf):
    """Backbone with all edge sites in chemistry X; conf 'AROall' = all aromatic,
    'qij' = top site i and bottom site j (1-based, x-sorted) quinonic,
    'sK' = top site K only quinonic (single-switch reference for J)."""
    rings = profile_rings(SITE_FLAKES[name])
    a0 = build_patch_flake(rings)
    bots, tops, _ = edge_sites(a0)
    top = [ARO_CH[X]] * len(tops)
    bot = [ARO_CH[X]] * len(bots)
    if conf[0] == 'q':
        i, j = int(conf[1]) - 1, int(conf[2]) - 1
        top[i] = bot[j] = QUI_CH[X]
    elif conf[0] == 's':
        top[int(conf[1]) - 1] = QUI_CH[X]
    return build_patch_flake(rings, bot=''.join(bot), top=''.join(top))


def site_bo(name, X, conf):
    key = f'site_{name}_{X}_{conf}'
    npz = os.path.join(OUTDIR, 'bo' + SK_TAG, key + '.npz')
    atoms = site_atoms(name, X, conf)
    return atoms, run_bo(atoms, 0, 0.0, npz)


def site_relax(name, X, conf, resr):
    key = f'site_{name}_{X}_{conf}'
    return run_relax(site_atoms(name, X, conf), 0, 0.0, key, resr, RESS)


def fig_site_maps(name, resr):
    """Per flake: bo_<name>_sites.png + bl_<name>_sites.png.  Rows = C/N/O;
    cols = AROall (absolute) | q11 | q12 | q13 | q22 — all as delta vs AROall,
    drawn on the switched state's geometry/bond list."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from spammm import plotUtils as pu, elements
    from spammm.quantum.pi_bond_order import plot_bond_scalar_map, bond_lengths
    rows = []
    for X in TIPCHEM:
        row = {'X': X}
        for cf in SITE_CONF:
            a, bo = site_bo(name, X, cf)
            ar = site_relax(name, X, cf, resr)
            row[cf] = (a, bo, bond_lengths(ar.apos, a.bonds), ar)
        rows.append(row)

    def delta_on(row, iv, cf):
        """v(cf) - v(AROall) on the cf bond list; nan/mask for state-unique bonds."""
        a, v = row[cf][0], row[cf][iv]
        v0 = row['AROall'][iv]
        ref = {frozenset(b): k for k, b in enumerate(row['AROall'][0].bonds)}
        d = np.full(len(v), np.nan)
        mask = np.zeros(len(v), bool)
        for k, b in enumerate(a.bonds):
            j = ref.get(frozenset(b))
            if j is not None and np.isfinite(v[k]) and np.isfinite(v0[j]):
                d[k], mask[k] = v[k] - v0[j], True
        return d, mask

    def atoms_on_top(ax, a, pos):
        en = [str(e).split('_')[0] for e in a.enames]
        pu.plotAtoms(apos=pos, es=en, sizes=[elements.ELEMENT_DICT[e][6] * 28. for e in en],
                     colors=[elements.ELEMENT_DICT[e][8] for e in en], marker='o', axes=(0, 1))

    def heavy_mask(a):
        en = [str(e).split('_')[0] for e in a.enames]
        return np.array([en[i] != 'H' and en[j] != 'H' for i, j in a.bonds])

    for itag, (tag, label) in enumerate([('bo', 'pi bond order'), ('bl', 'bond length')]):
        iv = 1 + itag
        va = np.concatenate([row['AROall'][iv][heavy_mask(row['AROall'][0]) if tag == 'bl' else slice(None)]
                             for row in rows])
        va = va[np.isfinite(va)]
        anrm = mcolors.Normalize(vmin=np.percentile(va, 10), vmax=np.percentile(va, 90))
        vd = np.abs(np.concatenate([delta_on(row, iv, cf)[0][delta_on(row, iv, cf)[1] &
                                    (heavy_mask(row[cf][0]) if tag == 'bl' else True)]
                                    for row in rows for cf in SITE_CONF[1:]]))
        vmax = np.nanpercentile(vd, 80)
        dnrm = mcolors.TwoSlopeNorm(vcenter=0., vmin=-vmax, vmax=vmax)
        fig, axs = plt.subplots(3, len(SITE_CONF), figsize=(17, 9))
        fig.subplots_adjust(left=0.02, right=0.90, top=0.93, bottom=0.02, wspace=0.02, hspace=0.12)
        pos_all = np.concatenate([row[cf][3].apos[:, :2] if tag == 'bl' else row[cf][0].apos[:, :2]
                                  for row in rows for cf in SITE_CONF])
        (x0, y0), (x1, y1) = pos_all.min(0) - 1.6, pos_all.max(0) + 1.6
        side = max(x1 - x0, y1 - y0) * 0.5
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1, y0, y1 = cx - side, cx + side, cy - side, cy + side
        for r, row in enumerate(rows):
            for c, cf in enumerate(SITE_CONF):
                ax = axs[r, c]
                a = row[cf][0]
                pos = row[cf][3].apos if tag == 'bl' else a.apos
                if cf == 'AROall':
                    d, msk = row[cf][iv], None
                    ttl = 'ARO all'
                else:
                    d, msk = delta_on(row, iv, cf)
                    ttl = f'{cf[1]}{cf[2]} quinonic - ARO'
                if tag == 'bl':
                    msk = heavy_mask(a) if msk is None else msk & heavy_mask(a)
                plot_bond_scalar_map(ax, a, pos, d, cmap='viridis' if cf == 'AROall' else 'coolwarm',
                                     norm=anrm if cf == 'AROall' else dnrm, mask=msk, bAtoms=False, lws=6.)
                atoms_on_top(ax, a, pos)
                ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
                ax.set_title(f"{row['X']} {ttl}", fontsize=9)
        cax1 = fig.add_axes([0.905, 0.56, 0.010, 0.36])
        cax2 = fig.add_axes([0.905, 0.07, 0.010, 0.36])
        fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=anrm), cax=cax1).set_label('absolute', fontsize=8)
        fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=dnrm), cax=cax2).set_label('delta vs AROall', fontsize=8)
        fig.suptitle(f'{name}: {label} | all 3+3 edge sites = chemistry | diffs coolwarm +/-{vmax:.3f}', fontsize=11)
        mdir = os.path.join(OUTDIR, 'maps' + (SK_TAG or '_3ob'))
        os.makedirs(mdir, exist_ok=True)
        png = os.path.join(mdir, f'{tag}_{name}_sites.png')
        fig.savefig(png, dpi=160)
        fig.savefig(png[:-4] + '.svg')
        plt.close(fig)
        print('REVIEW: ' + png)


def _detailed_comps(f):
    """Energy components from detailed.out (last geometry step), Ha."""
    import re
    FIELDS = [('H0', 'Energy H0'), ('SCC', 'Energy SCC'), ('3rd', 'Energy 3rd'),
              ('rep', 'Repulsive energy'), ('tot', 'Total energy')]
    if not os.path.isfile(f):
        return {k: np.nan for k, _ in FIELDS}
    txt = open(f).read()
    out = {}
    for k, field in FIELDS:
        m = re.findall(re.escape(field) + r':\s+(-?\d+\.\d+)\s+H', txt)
        out[k] = float(m[-1]) if m else np.nan
    return out


def site_rigid_comps(name, X, conf, resr, scc):
    """SP (SCC on/off) of `conf` chemistry on the relaxed AROall heavy skeleton —
    same trick as the ribbon jdecompv (heavy atoms transplanted, state H caps at
    built positions).  J(SCC on) - J(SCC off) = electrostatic/charge-response part."""
    a = site_atoms(name, X, conf)
    a0 = site_relax(name, X, 'AROall', resr)
    e = np.array([str(x).split('_')[0] for x in a.enames])
    e0 = np.array([str(x).split('_')[0] for x in a0.enames])
    apos = a.apos.copy()
    iv, iv0 = np.where(e != 'H')[0], np.where(e0 != 'H')[0]
    for i in iv:                                     # transplant heavy atoms
        same = iv0[e0[iv0] == e[i]]
        apos[i] = a0.apos[same[np.argmin(np.linalg.norm(a0.apos[same] - a.apos[i], axis=1))]]
    wd = os.path.join(OUTDIR, ('sp_rigid' if scc else 'sp_rigid_noscc') + SK_TAG,
                      f'site_{name}_{X}_{conf}')
    det = os.path.join(wd, 'detailed.out')
    if not os.path.isfile(det):
        enames = [str(x).split('_')[0] for x in a.enames]
        DU.run_dftb_sp(wd, enames, apos, DU.get_sk_path(SK_SET or DU.DEFAULT_SK_SET),
                       maxscc=500, filling_temp=100., scc=scc)
    return _detailed_comps(det)


def site_j_analysis(name, resr):
    """Pair coupling J(i,j) = E(qij) + E(AROall) - E(s_i) - E(s_j) decomposed
    into DFTB energy components (H0 band / SCC charge / 3rd / repulsive) from
    detailed.out.  Composition-neutral (the +/-H bookkeeping cancels).
    bottom singles by y-mirror; s3 = s1 by x-mirror.
    Second pass: same J on the rigid AROall skeleton, SCC on vs off —
    J_el = J_scc - J_noscc is the true electrostatic share (the SCC potential
    shifts the band energy, so 'Energy H0' alone is NOT non-electrostatic)."""
    for X in TIPCHEM:                       # ensure single-switch relaxes exist
        for cf in SINGLE_CONF:
            site_relax(name, X, cf, resr)
    FIELDS = [('H0', 'Energy H0'), ('SCC', 'Energy SCC'), ('3rd', 'Energy 3rd'),
              ('rep', 'Repulsive energy'), ('tot', 'Total energy')]

    def comps(X, cf):
        return _detailed_comps(os.path.join(OUTDIR, 'relax_bat' + SK_TAG,
                                            f'site_{name}_{X}_{cf}', 'detailed.out'))

    PAIRS = [('q11', 's1', 's1'), ('q12', 's1', 's2'), ('q13', 's1', 's1'), ('q22', 's2', 's2')]
    Jall = {}
    print(f'\n==== {name} site-pair coupling J(i,j) = E(ij)+E(ARO)-E(s_i)-E(s_j) [eV] ====')
    print(f'{"X":2s} {"pair":4s} {"J_tot":>8s} {"J_H0":>8s} {"J_SCC":>8s} {"J_3rd":>8s} {"J_rep":>8s}')
    for X in TIPCHEM:
        aro = comps(X, 'AROall')
        for qij, si, sj in PAIRS:
            cq, ca, ci, cj = (comps(X, c) for c in (qij, 'AROall', si, sj))
            J = {k: (cq[k] + ca[k] - ci[k] - cj[k]) * HAU2EV for k, _ in FIELDS}
            Jall[(X, qij)] = J
            print(f'{X:2s} {qij:4s} {J["tot"]:8.3f} {J["H0"]:8.3f} {J["SCC"]:8.3f} {J["3rd"]:8.3f} {J["rep"]:8.3f}')
    # rigid-skeleton pass: SP of the same four states on the relaxed AROall
    # skeleton, SCC on vs off -> J_el = electrostatic (charge-response) share
    CONF_ALL = ['AROall', 's1', 's2'] + [p[0] for p in PAIRS]
    print('   rigid AROall skeleton [eV]: J_rigid = J_noSCC + J_el')
    print(f'{"X":2s} {"pair":4s} {"J_rigid":>8s} {"J_noSCC":>8s} {"J_el":>8s}')
    Jrig = {}
    for X in TIPCHEM:
        for qij, si, sj in PAIRS:
            Js = {}
            for scc in (True, False):
                cq = site_rigid_comps(name, X, qij, resr, scc)
                ca = site_rigid_comps(name, X, 'AROall', resr, scc)
                ci = site_rigid_comps(name, X, si, resr, scc)
                cj = site_rigid_comps(name, X, sj, resr, scc)
                Js[scc] = (cq['tot'] + ca['tot'] - ci['tot'] - cj['tot']) * HAU2EV
            Jrig[(X, qij)] = {'rigid': Js[True], 'noscc': Js[False], 'el': Js[True] - Js[False]}
            print(f'{X:2s} {qij:4s} {Js[True]:8.3f} {Js[False]:8.3f} {Js[True] - Js[False]:8.3f}')
    # figure: top row = molecule sketch per pair (switched sites ringed),
    # mid = relaxed J term bars, bottom = rigid-skeleton noSCC/electrostatic bars
    import matplotlib.pyplot as plt
    from spammm import plotUtils as pu, elements
    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(3, 12, height_ratios=[1.0, 1.4, 1.4], hspace=0.35, wspace=0.15,
                          left=0.05, right=0.97, top=0.93, bottom=0.06)
    a0 = site_atoms(name, 'C', 'AROall')            # skeleton identical for all chem
    bots, tops, _ = edge_sites(a0)
    pos = a0.apos
    en = [str(e).split('_')[0] for e in a0.enames]
    for ip, (qij, _s, _t) in enumerate(PAIRS):
        ax = fig.add_subplot(gs[0, ip * 3:(ip + 1) * 3])
        for b in a0.bonds:
            ax.plot(pos[b, 0], pos[b, 1], 'k-', lw=0.8, alpha=0.3, zorder=1)
        plt.sca(ax)
        pu.plotAtoms(apos=pos, es=en,
                     sizes=[elements.ELEMENT_DICT[e][6] * 20. for e in en],
                     colors=[elements.ELEMENT_DICT[e][8] for e in en], marker='o', axes=(0, 1))
        i, j = int(qij[1]) - 1, int(qij[2]) - 1
        for idx, cc in ((tops[i], 'r'), (bots[j], 'b')):
            ax.scatter([pos[idx, 0]], [pos[idx, 1]], s=900., facecolors='none',
                       edgecolors=cc, linewidths=2.5, zorder=5)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'{qij}: top{i+1} + bot{j+1}  (red/blue)', fontsize=10)
    CC = {'H0': 'tab:blue', 'SCC': 'tab:orange', '3rd': 'tab:purple', 'rep': 'tab:gray'}
    ax0 = None
    for ix, X in enumerate(TIPCHEM):
        ax = fig.add_subplot(gs[1, ix * 4:(ix + 1) * 4], sharey=ax0)
        ax0 = ax0 or ax
        for ip, (qij, _s, _t) in enumerate(PAIRS):
            J = Jall[(X, qij)]
            bot = 0.0
            for k in ('H0', 'SCC', '3rd', 'rep'):
                ax.bar(ip, J[k], bottom=bot, width=0.7, color=CC[k],
                       label=k if ix == 0 and ip == 0 else None)
                bot += J[k]
            ax.plot(ip, J['tot'], 'kD', ms=5)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(range(len(PAIRS)))
        ax.set_xticklabels([p[0] for p in PAIRS])
        ax.set_title(f'{X} chem — relaxed (detailed.out terms)')
        if ix == 0:
            ax.set_ylabel('J [eV]')
            ax.legend(fontsize=8, title='diamond=total')
        else:
            ax.tick_params(labelleft=False)
    ax0 = None
    for ix, X in enumerate(TIPCHEM):
        ax = fig.add_subplot(gs[2, ix * 4:(ix + 1) * 4], sharey=ax0)
        ax0 = ax0 or ax
        for ip, (qij, _s, _t) in enumerate(PAIRS):
            J = Jrig[(X, qij)]
            ax.bar(ip, J['noscc'], width=0.7, color='tab:blue',
                   label='noSCC (band+rep, SCC off)' if ix == 0 and ip == 0 else None)
            ax.bar(ip, J['el'], bottom=J['noscc'], width=0.7, color='tab:orange',
                   label='electrostatic (SCC resp.)' if ix == 0 and ip == 0 else None)
            ax.plot(ip, J['rigid'], 'kD', ms=5)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(range(len(PAIRS)))
        ax.set_xticklabels([p[0] for p in PAIRS])
        ax.set_title(f'{X} chem — rigid skeleton')
        if ix == 0:
            ax.set_ylabel('J [eV]')
            ax.legend(fontsize=8, title='diamond=J_rigid')
        else:
            ax.tick_params(labelleft=False)
    fig.suptitle(f'{name}: site-pair coupling J(i,j) = E(pair)+E(ARO)-E(top)-E(bot)\n'
                 'mid: relaxed, detailed.out terms — CAUTION: "H0" incl. charge-response band shift | '
                 'bottom: rigid skeleton, noSCC vs electrostatic (= ribbon convention)')
    mdir = os.path.join(OUTDIR, 'maps' + (SK_TAG or '_3ob'))
    os.makedirs(mdir, exist_ok=True)
    png = os.path.join(mdir, f'J_{name}_sites.png')
    fig.savefig(png, dpi=160)
    fig.savefig(png[:-4] + '.svg')
    plt.close(fig)
    print('REVIEW: ' + png)


def fig_site_energies(name, resr):
    """dOmega of each site pair vs the all-aromatic reference, per chemistry.
    q states differ from AROall by +2H (C,N) / -2H (O) -> correct by
    dnH*mu_H so all bars are grand-potential differences at fixed mu_H."""
    import matplotlib.pyplot as plt
    res = json.load(open(RESULTS)) if os.path.isfile(RESULTS) else {}
    muH = (res['ref_H2']['E_ha'] / 2) if 'ref_H2' in res else -9.123 / HAU2EV
    DN_H = {'C': +2, 'N': +2, 'O': -2}          # AROall -> q pair: added/removed H
    confs = SITE_CONF[1:]
    COLX = {'C': 'tab:green', 'N': 'tab:blue', 'O': 'tab:red'}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xw = 0.25
    for ix, X in enumerate(TIPCHEM):
        de = []
        for cf in confs:
            e0 = resr.get(f'site_{name}_{X}_AROall', {}).get('E_ha')
            e1 = resr.get(f'site_{name}_{X}_{cf}', {}).get('E_ha')
            de.append(np.nan if e0 is None or e1 is None else (e1 - e0 - DN_H[X] * muH) * HAU2EV)
        xs = np.arange(len(confs)) + (ix - 1) * xw
        ax.bar(xs, de, width=xw * 0.9, color=COLX[X], label=X)
        for x, v in zip(xs, de):
            if np.isfinite(v):
                ax.text(x, v, f'{v:+.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(np.arange(len(confs)))
    ax.set_xticklabels(['top' + cf[1] + '/bot' + cf[2] for cf in confs])
    ax.set_ylabel('dOmega(pair) - dOmega(ARO all) [eV]')
    ax.set_title(f'{name}: pair-switch cost vs AROall, mu_H={muH * HAU2EV:.2f} eV (relaxed)')
    ax.legend(title='chemistry')
    fig.tight_layout()
    mdir = os.path.join(OUTDIR, 'maps' + (SK_TAG or '_3ob'))
    os.makedirs(mdir, exist_ok=True)
    png = os.path.join(mdir, f'E_{name}_sites.png')
    fig.savefig(png, dpi=160)
    fig.savefig(png[:-4] + '.svg')
    plt.close(fig)
    print('REVIEW: ' + png)


def fig_bond_maps(name, resr):
    """Two figures per backbone: bo_<name>.png (pi BO) and bl_<name>.png (bond
    lengths).  Rows = C/N/O chemistry; cols = Aromatic | Quinonic | Qui-Aro |
    1H*-Aro | 1H+-Aro | 1H--Aro — all diffs are (final - ARO reference),
    drawn on the FINAL state's geometry/bond list.  Absolute panels share one
    viridis norm per figure; diffs share a symmetric coolwarm norm.  Atom
    dots drawn on top of bonds."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from spammm import plotUtils as pu, elements
    from spammm.quantum.pi_bond_order import plot_bond_scalar_map, bond_lengths
    rows = []                     # per chemistry: {X, st -> (atoms, bo, bl, atoms_relax)}
    for X in TIPCHEM:
        row = {'X': X}
        for st, _s, _q in MAP_STATES:
            a, bo = bat_bo(name, X, st)
            ar = bat_relax_geom(name, X, st, resr)
            row[st] = (a, bo, bond_lengths(ar.apos, a.bonds), ar)
        rows.append(row)

    def delta_on(row, iv, st_draw, st_sub):
        """v(st_draw) - v(st_sub) on the st_draw bond list; nan/mask for bonds
        that exist only in one state (added X-H / removed bonds)."""
        a, v = row[st_draw][0], row[st_draw][iv]
        v0 = row[st_sub][iv]
        ref = {frozenset(b): k for k, b in enumerate(row[st_sub][0].bonds)}
        d = np.full(len(v), np.nan)
        mask = np.zeros(len(v), bool)
        for k, b in enumerate(a.bonds):
            j = ref.get(frozenset(b))
            if j is not None and np.isfinite(v[k]) and np.isfinite(v0[j]):
                d[k], mask[k] = v[k] - v0[j], True
        return d, mask

    def atoms_on_top(ax, a, pos):
        en = [str(e).split('_')[0] for e in a.enames]
        pu.plotAtoms(apos=pos, es=en, sizes=[elements.ELEMENT_DICT[e][6] * 28. for e in en],
                     colors=[elements.ELEMENT_DICT[e][8] for e in en], marker='o', axes=(0, 1))

    def heavy_mask(a):
        en = [str(e).split('_')[0] for e in a.enames]
        return np.array([en[i] != 'H' and en[j] != 'H' for i, j in a.bonds])

    COLS = ['ARO', 'QUI', 'QUI-ARO', 'M1r', 'M1p', 'M1m']
    LAB = {'M1r': '1H*-ARO', 'M1p': '1H+-ARO', 'M1m': '1H--ARO'}
    for itag, (tag, label) in enumerate([('bo', 'pi bond order'), ('bl', 'bond length')]):
        iv = 1 + itag             # tuple index: 1=bo, 2=bl
        va = np.concatenate([row[st][iv][heavy_mask(row[st][0]) if tag == 'bl' else slice(None)]
                             for row in rows for st in ('ARO', 'QUI')])
        va = va[np.isfinite(va)]
        anrm = mcolors.Normalize(vmin=np.percentile(va, 10), vmax=np.percentile(va, 90))  # shared, clipped
        def dmag(row, a, b):
            d, msk = delta_on(row, iv, a, b)
            if tag == 'bl':
                msk = msk & heavy_mask(row[a][0])
            return d[msk]
        vd = np.abs(np.concatenate([dmag(row, a, b)
                   for row in rows for a, b in [('QUI', 'ARO'), ('M1r', 'ARO'), ('M1p', 'ARO'), ('M1m', 'ARO')]]))
        vmax = np.nanpercentile(vd, 80)               # tight scale: saturate extremes, keep contrast
        dnrm = mcolors.TwoSlopeNorm(vcenter=0., vmin=-vmax, vmax=vmax)
        fig, axs = plt.subplots(3, 6, figsize=(19, 9))
        fig.subplots_adjust(left=0.02, right=0.90, top=0.93, bottom=0.02, wspace=0.02, hspace=0.12)
        # common zoom: one bounding box over all positions drawn in this figure
        pos_all = np.concatenate([row[st][3].apos[:, :2] if tag == 'bl' else row[st][0].apos[:, :2]
                                  for row in rows for st, _s, _q in MAP_STATES])
        (x0, y0), (x1, y1) = pos_all.min(0) - 1.6, pos_all.max(0) + 1.6   # margin for H caps
        side = max(x1 - x0, y1 - y0) * 0.5
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1, y0, y1 = cx - side, cx + side, cy - side, cy + side
        for r, row in enumerate(rows):
            for c, col in enumerate(COLS):
                ax = axs[r, c]
                if col == 'QUI-ARO':
                    a = row['QUI'][0]
                    pos = row['QUI'][3].apos if tag == 'bl' else a.apos
                    d, msk = delta_on(row, iv, 'QUI', 'ARO')
                elif col in ('ARO', 'QUI'):
                    a = row[col][0]
                    pos = row[col][3].apos if tag == 'bl' else a.apos
                    d, msk = row[col][iv], None
                else:                              # mixed states: delta vs aromatic
                    a = row[col][0]
                    pos = row[col][3].apos if tag == 'bl' else a.apos
                    d, msk = delta_on(row, iv, col, 'ARO')
                if tag == 'bl':                    # heavy-heavy bonds only (drop X-H 1A)
                    msk = heavy_mask(a) if msk is None else msk & heavy_mask(a)
                if col in ('ARO', 'QUI'):
                    plot_bond_scalar_map(ax, a, pos, d, cmap='viridis', norm=anrm, mask=msk, bAtoms=False, lws=6.)
                else:
                    plot_bond_scalar_map(ax, a, pos, d, cmap='coolwarm', norm=dnrm, mask=msk, bAtoms=False, lws=6.)
                atoms_on_top(ax, a, pos)
                ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
                ax.set_title(f"{row['X']} {LAB.get(col, col)}", fontsize=9)
        cax1 = fig.add_axes([0.905, 0.56, 0.010, 0.36])   # absolute — fixed size, right edge
        cax2 = fig.add_axes([0.905, 0.07, 0.010, 0.36])   # delta
        fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=anrm), cax=cax1).set_label('absolute', fontsize=8)
        fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=dnrm), cax=cax2).set_label('delta', fontsize=8)
        fig.suptitle(f'{name}: {label} | Aro & Qui absolute (shared scale) | diffs = final - ARO, coolwarm +/-{vmax:.3f}', fontsize=11)
        mdir = os.path.join(OUTDIR, 'maps' + (SK_TAG or '_3ob'))   # maps_<variant>/ subfolders
        os.makedirs(mdir, exist_ok=True)
        png = os.path.join(mdir, f'{tag}_{name}.png')
        fig.savefig(png, dpi=160)
        fig.savefig(png[:-4] + '.svg')
        plt.close(fig)
        print('REVIEW: ' + png)


def battery_verdicts(res):
    """Printed cooperativity table: radical vs ionic annihilation channels."""
    E = lambda k: (res[k]['E_ha'] or float('nan')) * HAU2EV if k in res and res[k]['E_ha'] is not None else float('nan')
    muH = E('ref_H2') / 2
    print('\n==== flake battery: 2-site switch channels (eV) ====')
    print('d1H* = E(1Hr)-E(0H)-mu_H | d2H = E(2H)-E(0H)-E(H2)')
    print('ann*  = E(0H)+E(2H)-2E(1Hr)        radical pairing (<0 cooperative)')
    print('ann+- = E(0H)+E(2H)-E(1Hp)-E(1Hm)  ionic pairing (charge-conserving)')
    for name in BACKBONES:
        print(f'-- {name}')
        for X in TIPCHEM:
            k = lambda s: f'{name}_{X}_{s}'
            if k('0H') not in res:
                continue
            e0, e2, er, ep, em = (E(k(s)) for s in ('0H', '2H', '1Hr', '1Hp', '1Hm'))
            t_s = f' | E_T-E_S {1000 * (E(k("2H_T")) - e2):+.0f} meV' if k('2H_T') in res and res[k('2H_T')]['E_ha'] is not None else ''
            print(f'   {X}: d1H* {er - e0 - muH:+7.3f} | d2H {e2 - e0 - 2 * muH:+7.3f} | '
                  f'ann* {e0 + e2 - 2 * er:+7.3f} | ann+- {e0 + e2 - ep - em:+7.3f}{t_s}')


# ---------------------------------------------------------------- references
# H chemical-potential references: total energies only become comparable once
# expressed per E(H2).  Benchmarks: ethylene+H2->ethane ~ -1.4 eV exp.,
# benzene+H2->1,4-cyclohexadiene ~ +1.0 eV exp. (aromaticity loss).

def xyz_mol(enames, coords):
    from spammm.AtomicSystem import AtomicSystem
    a = AtomicSystem(apos=np.asarray(coords, dtype=float), enames=list(enames))
    a.findBonds(Rcut=3.0)
    return a


def cyclohexane():
    """1x1 rhombus with ALL sites 'c' -> sp3 ring -> C6H12 (relaxes to chair)."""
    patch, _, _ = _rhombus_patch(1, 1)
    return fix_sp3_caps(M.build_mol(_rasterize({k: 'c' for k in patch})))


def ref_jobs():
    """(key, builder, q, spin) — H reservoir + benchmark hydrogenation pairs."""
    a1, a2 = 1.09 * np.cos(np.radians(70.5)), 1.09 * np.sin(np.radians(70.5))
    ethane_H = []                                        # staggered C2H6
    for s in (+1, -1):
        for phi in (0, 120, 240):
            ph = np.radians(phi + (0 if s > 0 else 60))
            ethane_H.append([s * (0.77 + a1), a2 * np.cos(ph), a2 * np.sin(ph)])
    out = [
        ('ref_H',        lambda: xyz_mol(['H'], [[0, 0, 0]]),                                     0.0, 1),
        ('ref_H2',       lambda: xyz_mol(['H', 'H'], [[0, 0, 0], [0, 0, 0.74]]),                  0.0, 0),
        ('ref_ethylene', lambda: xyz_mol(['C', 'C', 'H', 'H', 'H', 'H'],
                                        [[0.67, 0, 0], [-0.67, 0, 0], [1.23, 0.94, 0], [1.23, -0.94, 0],
                                         [-1.23, 0.94, 0], [-1.23, -0.94, 0]]),                   0.0, 0),
        ('ref_ethane',   lambda: xyz_mol(['C', 'C'] + ['H'] * 6, [[0.77, 0, 0], [-0.77, 0, 0]] + ethane_H), 0.0, 0),
        ('ref_cyclohex', cyclohexane,                                                              0.0, 0),
    ]
    return out


def all_jobs():
    return jobs() + mol_jobs() + battery_jobs() + ref_jobs()


def fig_geometry(path):
    """Geometry sanity panel (canonical plotSystem style) — each flake state
    with switched sites ringed and its DFTB result annotated."""
    import matplotlib.pyplot as plt
    from spammm import plotUtils as pu
    panels = [
        ('phenalenyl',  build_ring_flake([(0, 0), (1, 0), (0, 1)]),                        None),
        ('triangulene', build_ring_flake([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]), None),
        ('fl_N',        flake('N'),                                                        None),
        ('N_1H',        flake('N', {_C: 'n'}),                                             [_TOPS[_C]]),
        ('N_2Hs',       flake('N', {_C: 'n', _C + 1: 'n'}),                                [_TOPS[_C], _TOPS[_C + 1]]),
        ('N_2Ho',       flake('N', {_C: 'n'}, {_CB: 'n'}),                                 [_TOPS[_C], _BOTS[_CB]]),
        ('C_1CH2',      flake('C', {_C: 'c'}),                                             [_TOPS[_C]]),
    ]
    fig, axs = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 3.0))
    for ax, (name, atoms, hi) in zip(axs, panels):
        plt.sca(ax)
        pu.plotSystem(atoms, bLabels=False, sz=40.)
        if hi:
            en = [str(e).split('_')[0] for e in atoms.enames]
            col = sublattice(atoms)
            for i in hi:   # switched site: lime ring; sublattice tint rings on heavy atoms
                ax.scatter([atoms.apos[i, 0]], [atoms.apos[i, 1]], facecolors='none', edgecolors='lime', s=260, linewidths=1.6, zorder=5)
            for i in range(atoms.natoms):
                if en[i] != 'H' and col[i] >= 0:
                    ax.scatter([atoms.apos[i, 0]], [atoms.apos[i, 1]], facecolors='none', edgecolors=('tab:red' if col[i] == 0 else 'tab:blue'), s=90, linewidths=0.6, zorder=2)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print('REVIEW: ' + path)


def fig_mol_geometry(path):
    """mol x two-tip-switch panel (canonical plotSystem); N sites and sp3 CH2 ringed."""
    import matplotlib.pyplot as plt
    from spammm import plotUtils as pu
    fig, axs = plt.subplots(len(MOLS), len(MOLPAIRS), figsize=(2.2 * len(MOLPAIRS), 2.2 * len(MOLS) + 0.4))
    for r, mol in enumerate(MOLS):
        for cix, pair in enumerate(MOLPAIRS):
            ax = axs[r, cix]
            plt.sca(ax)
            atoms = build_switch_mol(mol, pair[0], pair[1])
            pu.plotSystem(atoms, bLabels=False, sz=45.)
            en = [str(e).split('_')[0] for e in atoms.enames]
            nbr = {}
            for a, b in atoms.bonds:
                nbr.setdefault(a, []).append(b); nbr.setdefault(b, []).append(a)
            for i, e in enumerate(en):   # ring: N / O tips + sp3 CH2 (C with 2 H)
                if e in 'NO' or (e == 'C' and sum(en[j] == 'H' for j in nbr.get(i, [])) == 2):
                    ax.scatter([atoms.apos[i, 0]], [atoms.apos[i, 1]], facecolors='none', edgecolors='lime', s=300, linewidths=1.4, zorder=5)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title(f'{mol} {pair}', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print('REVIEW: ' + path)


MOLSERIES = [('CH2', 'CC', 'cC', 'cc', '=CH-/-CH2- tips'),
             ('NH',  'NN', 'nN', 'nn', '=N-/-NH- tips'),
             ('OH',  'OO', 'oO', 'oo', '=O/-OH tips')]


def relaxed_geom(key, res=None):
    """Relaxed geometry of a 'relax' job -> AtomicSystem; falls back to the
    built (unrelaxed) geometry when the relax failed (results entry E=None)."""
    import spammm.atomicUtils as au
    if res is not None and res.get(key, {}).get('E_ha') is None:
        for k, build, q, s in ref_jobs() + mol_jobs():
            if k == key:
                return build()
    wd = os.path.join(OUTDIR, 'relax', key)
    xyzs, _Zs, enames, _qs, _ = au.load_xyz(os.path.join(wd, 'geom.out.xyz'))
    return xyz_mol(enames, xyzs)


def fig_mol_report(mol, res, path):
    """Per-molecule didactic figure: rows = tip series (CH2/NH/OH); per row the
    relaxed 0H/1H/2H structures + grand-potential diagram Omega_n(mu_H) =
    E(nH,q)-E(0H,q)-n*mu_H split by charge (neutral / anion / cation).
    Lower envelope = stable hydrogenation state; where 1H never reaches the
    envelope, the radical intermediate is skipped (annihilation-driven 2H)."""
    import matplotlib.pyplot as plt
    from spammm import plotUtils as pu
    E = lambda k: (res[k]['E_ha'] or float('nan')) * HAU2EV if res[k]['E_ha'] is not None else float('nan')
    muH2 = E('ref_H2') / 2                                  # H2 reservoir
    muC2 = (E('ref_ethane') - E('ref_ethylene')) / 2        # C2H4+H2<->C2H6 reservoir
    mus = np.linspace(-14.0, -6.0, 300)
    qs = [(0.0, 'neutral'), (-1.0, 'anion q=-1'), (+1.0, 'cation q=+1')]
    fig, axs = plt.subplots(len(MOLSERIES), 6, figsize=(19, 9.6))
    for r, (tag, base, v1, v2, _desc) in enumerate(MOLSERIES):
        for c, (k, lab) in enumerate([(f'{mol}_{base}_q+0', '0H'), (f'{mol}_{v1}_q+0', '1H'), (f'{mol}_{v2}_q+0', '2H')]):
            ax = axs[r, c]
            plt.sca(ax)
            pu.plotSystem(relaxed_geom(k, res), bLabels=False, sz=45.)
            ax.set_aspect('equal'); ax.axis('off')
            ax.set_title(f'{tag} {lab}', fontsize=9)
        for c, (q, qlab) in enumerate(qs):
            ax = axs[r, 3 + c]
            Eb = E(f'{mol}_{base}_q{q:+.0f}')
            lines = {}
            for n, v, col in ((0, base, '0.4'), (1, v1, 'tab:orange'), (2, v2, 'tab:green')):
                om = E(f'{mol}_{v}_q{q:+.0f}') - Eb - n * mus
                lines[n] = om
                ax.plot(mus, om, '-', c=col, lw=1.2, label=f'{n}H')
            env = np.nanmin(np.stack(list(lines.values())), axis=0)
            ax.plot(mus, env, 'k-', lw=2.5, alpha=0.55)
            ax.axvline(muH2, color='k', ls='--', lw=0.9)
            ax.axvline(muC2, color='k', ls=':', lw=0.9)
            ax.set_ylim(-6, 3)
            ax.set_title(qlab, fontsize=9)
            if c == 0:
                ax.set_ylabel(r'$\Omega = E(nH) - E(0H) - n\mu_H$  [eV]', fontsize=8)
            if r == len(MOLSERIES) - 1:
                ax.set_xlabel(r'$\mu_H$ [eV]', fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc='upper left')
    fig.suptitle(f'{mol}: hydrogenation ladder vs hydrogen chemical potential\n'
                 f'(dashed = H2 reservoir {muH2:.2f} eV, dotted = C2H4+H2<->C2H6 {muC2:.2f} eV; '
                 f'black = stable state)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print('REVIEW: ' + path)


def deflect_tips(atoms, dz=0.3):
    """Small out-of-plane nudge on the two tip atoms + their substituents
    (NH-H, CH2-H's, =O/-OH): planar starts are saddle points if the molecule
    wants to pucker (pyrrole-like).  Tips = extreme-|y| atoms (apex C, or the
    exocyclic O where present); substituents follow; +/-dz alternating keeps
    approximate C2."""
    en = [str(e).split('_')[0] for e in atoms.enames]
    nbr = {}
    for i, j in atoms.bonds:
        nbr.setdefault(i, []).append(j)
        nbr.setdefault(j, []).append(i)
    tips = [min(range(atoms.natoms), key=lambda i: atoms.apos[i, 1]),
            max(range(atoms.natoms), key=lambda i: atoms.apos[i, 1])]
    for s, i in zip((+1., -1.), tips):
        atoms.apos[i, 2] += s * dz * 0.5                       # tip atom itself
        for j in nbr[i]:
            if en[j] in ('H', 'O'):
                atoms.apos[j, 2] += s * dz
                for k in nbr[j]:                            # O-H hydrogen follows the O
                    if en[k] == 'H' and en[j] == 'O':
                        atoms.apos[k, 2] += s * dz
    return atoms


def export_pyscf(outdir, res, dz=0.3, overview=True):
    """Export the full job set for independent B3LYP recalculation:
    xyz/<key>.xyz (DFTB-relaxed geometry where available) + jobs.csv manifest
    (key, xyz, charge, n_unpaired) + png/ thumbnails + index.html overview.
    Run with tests/topology/run_pyscf_b3lyp.py."""
    import matplotlib.pyplot as plt
    from spammm import plotUtils as pu
    xdir = os.path.join(outdir, 'xyz')
    pdir = os.path.join(outdir, 'png')
    os.makedirs(xdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    rows = []

    def add(key, atoms, q, spin, deflect=False, tier='main'):
        src = 'builder'
        if res.get(key, {}).get('E_ha') is not None and \
           os.path.exists(os.path.join(OUTDIR, 'relax', key, 'geom.out.xyz')):
            atoms = relaxed_geom(key, res)          # DFTB-relaxed start for B3LYP opt
            src = 'DFTB-relaxed'
        if deflect:
            atoms = deflect_tips(atoms, dz)         # seed for out-of-plane pucker
        en = [str(e).split('_')[0] for e in atoms.enames]
        with open(os.path.join(xdir, key + '.xyz'), 'w') as f:
            f.write(f'{len(en)}\n{key}\n')
            for e, p in zip(en, atoms.apos):
                f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
        if spin is None:
            spin = n_electrons(en, q) % 2           # lowest spin per parity
        # sanity: per-atom nearest-neighbour distances (catch exploded/stacked geoms)
        if len(en) > 1:
            d = np.linalg.norm(atoms.apos[None, :, :] - atoms.apos[:, None, :], axis=-1)
            d += np.eye(len(d)) * 1e9
            nn_min, nn_max = d.min(), d.min(axis=1).max()
        else:
            nn_min = nn_max = float('nan')
        flag = 'OK' if (len(en) == 1 or (nn_max < 1.9 and nn_min > 0.6)) else 'SUSPECT'
        rows.append((key, 'xyz/' + key + '.xyz', f'{q:+.0f}', spin, src, len(en),
                     n_electrons(en, q), nn_min, nn_max, flag, tier))
        if overview:
            fig = plt.figure(figsize=(1.7, 1.7))
            ax = fig.add_subplot(111)
            plt.sca(ax)
            pu.plotSystem(atoms, bLabels=False, sz=45., bBonds=(atoms.natoms > 1))
            ax.set_aspect('equal'); ax.axis('off')
            fig.savefig(os.path.join(pdir, key + '.png'), dpi=110,
                        bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
        if flag != 'OK':
            print(f'  !! {key}: nn_min={nn_min:.2f} nn_max={nn_max:.2f} A', flush=True)

    # --- main set: the open-shell 1H states with all 3 charge channels
    #     (radical q0 / cation q+1 / anion q-1) + the closed-shell 0H and 2H
    #     singlet references.  bonus tier: ions of 0H/2H and all 2H triplets.
    def mol_tier(key):
        if key.endswith('_T'):
            return 'bonus'                          # 2H triplet diagnostic
        pair = key.split('_')[1]
        q = key.split('_q')[1].split('_')[0]
        rad = pair in ('cC', 'nN', 'oO')            # 1H one-tip states
        return 'main' if (rad or q == '+0') else 'bonus'

    for key, build, q, spin in ref_jobs():
        add(key, build(), q, spin)
    for key, build, q, spin in mol_jobs():
        add(key, build(), q, spin, deflect=True,    # tips may pucker out of plane
            tier=mol_tier(key))
    for key, build, spin, q in jobs():              # old flake set (SP geoms)
        if key == 'phenalenyl_S0':
            continue                                # odd Ne + spin 0: not a legal UKS/RKS job
        add(key, build(), q, spin, deflect=True)
    for key, build, spin, q in battery_jobs():      # 9-backbone tip battery (SP geoms)
        add(key, build(), q, spin, deflect=True,
            tier='bonus' if key.endswith('2H_T') else 'main')

    for name, sel in (('jobs.csv', 'main'), ('jobs_bonus.csv', 'bonus')):
        with open(os.path.join(outdir, name), 'w') as f:
            f.write('key,xyz,charge,spin\n')
            for r in rows:
                if r[10] == sel:
                    f.write(','.join(str(x) for x in r[:4]) + '\n')
        n = sum(r[10] == sel for r in rows)
        print(f'  {name}: {n} jobs', flush=True)
    print(f'exported {len(rows)} jobs -> {outdir}/ (+ {xdir}/, {pdir}/)')

    if overview:
        write_overview_html(outdir, rows)


def write_overview_html(outdir, rows):
    """index.html — table per backbone: one row per (backbone, tip chemistry
    C/N/O), columns 0H | 2H | 1H* | 1H+ | 1H-.  Extra charge/triplet variants
    are stacked inside their state cell.  Suspect geometries are red."""
    by_key = {r[0]: r for r in rows}
    PAIR2X = {'C': 'Cc', 'N': 'Nn', 'O': 'Oo'}     # series -> (lo,hi) tip chars

    def mk_bat(key):
        for b in BACKBONES:
            for X in TIPCHEM:
                p = f'{b}_{X}_'
                if key.startswith(p):
                    st = key[len(p):]
                    col = {'0H': '0H', '2H': '2H', '2H_T': '2H',
                           '1Hr': '1Hr', '1Hp': '1Hp', '1Hm': '1Hm'}.get(st)
                    if col:
                        return (b, X), col
        return None

    # mol pairs carry explicit _q{+-1} suffixes -> route by charge
    def mk_mol2(key):
        for m in MOLS:
            if not key.startswith(m + '_'):
                continue
            pair = key.split('_')[1]
            q = '+0' if '_q' not in key else key.split('_q')[1].split('_')[0]
            for X, (lo, hi) in PAIR2X.items():
                base, rad, dbl = lo * 2, hi + lo, hi * 2
                if pair == base:
                    return (m, X), ('0H' if q == '+0' else '0H_q')
                if pair == dbl:
                    return (m, X), ('2H' if q == '+0' else '2H_q')
                if pair == rad:
                    return (m, X), {'+0': '1Hr', '+1': '1Hp', '-1': '1Hm'}[q]
            return None
        return None

    def cell2(keys):
        out = []
        for k in keys:
            if k not in by_key:
                continue
            r = by_key[k]
            sus = ' style="background:#fdd"' if r[9] != 'OK' else ''
            sty = 'opacity:0.45;border:1px dashed #999;' if r[10] == 'bonus' else ''
            tag = 'q' + r[2] + (' T' if r[3] == 2 else ' &#8226;' if r[3] == 1 else '')
            out.append(f'<div{sus}><div style="{sty}"><img src="png/{k}.png"><br>{tag}'
                       f'{" bonus" if r[10] == "bonus" else ""} <i>{r[4][:4]}</i></div></div>')
        return '<td>' + ''.join(out) + '</td>' if out else '<td class="e">—</td>'

    def table2(tag, mk):
        seen, tab = [], {}
        for r in rows:
            g = mk(r[0])
            if g is None:
                continue
            rk, col = g
            if rk not in seen:
                seen.append(rk)
            tab.setdefault(rk, {}).setdefault(col, []).append(r[0])
        h = [f'<h2>{tag}</h2><table><tr><th></th><th>0H</th><th>0H &plusmn;1</th>'
             '<th>2H (S,T)</th><th>2H &plusmn;1</th>'
             '<th>1H&#8226; q0</th><th>1H q+1</th><th>1H q-1</th></tr>']
        for rk in seen:
            c = tab[rk]
            h.append(f'<tr><th>{rk[0]} <b>{rk[1]}</b></th>' +
                     ''.join(cell2(c.get(col, [])) for col in
                             ('0H', '0H_q', '2H', '2H_q', '1Hr', '1Hp', '1Hm')) + '</tr>')
        h.append('</table>')
        return '\n'.join(h)

    SPINLAB = {0: 'S', 1: 'D&#8226;', 2: 'T'}
    html = ['<html><head><meta charset="utf-8"><style>',
            'body{font-family:monospace;font-size:11px}',
            'table{border-collapse:collapse;margin-bottom:18px}',
            'td,th{border:1px solid #bbb;padding:2px;vertical-align:top;text-align:center}',
            'td.e{color:#999}', 'img{width:110px;display:block;margin:auto}',
            'td div{margin:1px 0}', 'h2{margin:14px 0 4px}',
            '</style></head><body>',
            f'<h1>PySCF job set — {len(rows)} jobs</h1>',
            'columns: 0H base | 2H closed-shell (+T) | 1H radical/cation/anion. ',
            'Cell labels: manifest charge/spin; <i>DFTB</i> = relaxed start, <i>buil</i> = builder. ',
            'Dimmed/dashed = <b>bonus tier</b> (jobs_bonus.csv: 0H/2H ions + triplets) — not in the main set. ',
            'Red = suspect geometry (nearest-neighbour check).<br>']

    html.append(table2('small molecules (two tips)', mk_mol2))
    html.append(table2('flake battery (central edge sites)', mk_bat))

    # leftover jobs (refs + legacy flakes) as a flat grid
    used = {k for g in (mk_mol2, mk_bat) for r in rows
            for k in ([r[0]] if g(r[0]) else [])}
    rest = [r for r in rows if r[0] not in used]
    if rest:
        html.append('<h2>references + legacy flakes</h2><table><tr>')
        for i, r in enumerate(rest):
            if i % 8 == 0:
                html.append('</tr><tr>')
            sus = ' style="background:#fdd"' if r[9] != 'OK' else ''
            html.append(f'<td><div{sus}><img src="png/{r[0]}.png"><br><b>{r[0]}</b><br>'
                        f'q={r[2]} {SPINLAB[r[3]]} <i>{r[4][:4]}</i></div></td>')
        html.append('</tr></table>')

    nsus = sum(r[9] != 'OK' for r in rows)
    html.append('</body></html>')
    with open(os.path.join(outdir, 'index.html'), 'w') as f:
        f.write('\n'.join(html))
    print(f'overview -> {outdir}/index.html   ({nsus} suspect geometries)', flush=True)


def run_job(res, key, atoms, spin, q, mode):
    Ne = n_electrons(atoms.enames, q)
    if Ne % 2 and spin == 0:
        print(f'  ! {key}: odd Ne={Ne} but spin=0 (restricted) — bias reference only', flush=True)
    wd = os.path.join(OUTDIR, mode, key)
    if mode == 'sp':
        def run():
            E, m = sp_cell(atoms.enames, atoms.apos, wd, spin=spin, charge=q)
            return {'E_ha': E, 'm': m, 'Ne': Ne}
    else:
        def run():
            E, _ = M.relax_cell(atoms.enames, atoms.apos, wd, spin=spin, charge=q)
            m = parse_spin_moments(os.path.join(wd, 'detailed.out'))
            return {'E_ha': E, 'm': m, 'Ne': Ne}
    try:
        r = cached(res, key, run)
    except RuntimeError as e:
        print(f'  !! {key}: FAILED — {str(e).splitlines()[0]}', flush=True)
        r = res[key] = {'E_ha': None, 'failed': str(e).splitlines()[0]}
        with open(RESULTS, 'w') as f:
            json.dump(res, f, indent=1, sort_keys=True)
    if r.get('m') is None and spin and r.get('E_ha') is not None:   # re-parse older caches (parser added later)
        r['m'] = parse_spin_moments(os.path.join(wd, 'detailed.out'))
        if r['m'] is not None:
            with open(RESULTS, 'w') as f:
                json.dump(res, f, indent=1, sort_keys=True)
    return r, Ne


def print_row(key, atoms, Ne, spin, q, E, m):
    sm = mx = ipr = float('nan'); mab = ''
    if m:
        m = np.asarray(m)
        col = sublattice(atoms)
        sm, mx, ipr = m.sum(), np.abs(m).max(), float(np.sum(m**2) / max(np.sum(np.abs(m))**2, 1e-12))
        mab = f'{m[col == 0].sum():+.2f}/{m[col == 1].sum():+.2f}'
    Es = f'{E * HAU2EV:12.4f}' if E is not None else '      FAILED'
    print(f'{key:16s} {atoms.natoms:4d} {Ne:4d} {spin:4d} {q:+4.0f} {Es} {sm:6.2f} {mx:7.3f} {ipr:7.4f} {mab:>11s}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', default=None, help='comma list of job keys')
    ap.add_argument('--group', default='all', choices=['flake', 'mol', 'ref', 'bat', 'all'])
    ap.add_argument('--export-pyscf', default=None, metavar='DIR',
                    help='export all jobs as xyz + jobs.csv for B3LYP (run_pyscf_b3lyp.py)')
    ap.add_argument('--maps', default=None, metavar='NAME|all',
                    help='per-backbone BO/BL map figures (DFTBcore SP bo + relaxed bl)')
    ap.add_argument('--sites', default=None, metavar='NAME|all',
                    help='edge-site scan: all zigzag sites substituted, one top+bottom pair switched')
    ap.add_argument('--sk', default=None, choices=list(DU.AVAILABLE_SK_SETS),
                    help='Slater-Koster set for map computations (default 3ob-3-1; mio-1-1 = plain SCC, no DFTB3)')
    ap.add_argument('--tight', action='store_true',
                    help='tight convergence for --maps: SP scctol 1e-10, relax scctol 1e-9 + GradElem 1e-6')
    args = ap.parse_args()
    global SK_SET, TIGHT, SK_TAG, RESR, RESS
    SK_SET, TIGHT = args.sk, args.tight
    if SK_SET or TIGHT:      # variant tag: _3obt / _mio / _miot — keeps 3ob-standard caches intact
        SK_TAG = '_' + (SK_SET or DU.DEFAULT_SK_SET).split('-')[0] + ('t' if TIGHT else '')
        RESR = os.path.join(OUTDIR, f'results_relax{SK_TAG}.json')
        RESS = os.path.join(OUTDIR, f'results_site{SK_TAG}.json')
    res = json.load(open(RESULTS)) if os.path.isfile(RESULTS) else {}
    wanted = set(args.states.split(',')) if args.states else None

    print(f'{"job":16s} {"nat":>4s} {"Ne":>4s} {"spin":>4s} {"q":>4s} {"E [eV]":>12s} {"sum m":>6s} {"max|m|":>7s} {"IPR(m)":>7s} {"mA/mB":>11s}', flush=True)
    if args.group in ('flake', 'all'):
        for key, build, spin, q in jobs():
            if wanted and key not in wanted:
                continue
            atoms = build()
            r, Ne = run_job(res, key, atoms, spin, q, 'sp')
            print_row(key, atoms, Ne, spin, q, r['E_ha'], r.get('m'))
    if args.group in ('mol', 'ref', 'all'):
        for key, build, q, spin in ref_jobs():
            if wanted and key not in wanted:
                continue
            atoms = build()
            r, Ne = run_job(res, key, atoms, spin, q, 'relax')
            print_row(key, atoms, Ne, spin, q, r['E_ha'], r.get('m'))
    if args.group in ('mol', 'all'):
        for key, build, q, spin in mol_jobs():
            if wanted and key not in wanted:
                continue
            atoms = build()
            if spin is None:
                spin = n_electrons(atoms.enames, q) % 2      # lowest spin per parity
            r, Ne = run_job(res, key, atoms, spin, q, 'relax')
            print_row(key, atoms, Ne, spin, q, r['E_ha'], r.get('m'))
    if args.group in ('bat', 'all'):
        for key, build, spin, q in battery_jobs():
            if wanted and key not in wanted:
                continue
            atoms = build()
            r, Ne = run_job(res, key, atoms, spin, q, 'sp')
            print_row(key, atoms, Ne, spin, q, r['E_ha'], r.get('m'))

    # ---- go/no-go verdicts ------------------------------------------------
    def E(key):
        v = res[key]['E_ha']
        return v * HAU2EV if v is not None else float('nan')
    print('\n==== validation / verdicts ====')
    if 'triangulene_T' in res and 'triangulene_S' in res:
        print(f'triangulene  E_T - E_S = {1000 * (E("triangulene_T") - E("triangulene_S")):+.1f} meV   (expect < 0: triplet GS)')
    for st, tag in (('N_2Hs', 'same edge'), ('N_2Ho', 'opp edge'), ('C_2CH2s', 'same edge'), ('C_2CH2o', 'opp edge')):
        if f'{st}_T' in res and f'{st}_S' in res:
            print(f'{st:9s} ({tag})  E_T - E_S = {1000 * (E(st + "_T") - E(st + "_S")):+.1f} meV')
    if all(k in res for k in ('N_1H', 'N_1Hp', 'N_1Hm', 'fl_N')):
        print(f'1H on N edge vs fl_N:  PCET radical {E("N_1H") - E("fl_N"):+.3f} | proton-only {E("N_1Hp") - E("fl_N"):+.3f} | H+e- {E("N_1Hm") - E("fl_N"):+.3f} eV')

    # ---- H chemical-potential references -----------------------------------
    if 'ref_H2' in res:
        print('\n==== H reservoir references (relaxed, eV) ====')
        EH, EH2 = E('ref_H'), E('ref_H2')
        print(f'E(H atom) {EH:+.3f} | E(H2) {EH2:+.3f} | mu_H = E(H2)/2 {EH2 / 2:+.3f} | D(H-H) {2 * EH - EH2:+.3f} eV (exp 4.75)')
        if 'ref_ethane' in res:
            print(f'ethylene+H2->ethane  {E("ref_ethane") - E("ref_ethylene") - EH2:+.3f} eV  (exp ~ -1.36)')
        if 'ref_cyclohex' in res and 'benzene_CC_q+0' in res:
            print(f'benzene+H2->1,4-cyclohexadiene (cc) {E("benzene_cc_q+0") - E("benzene_CC_q+0") - EH2:+.3f} eV  (exp ~ +0.9..1.0)')
            print(f'benzene+3H2->cyclohexane           {E("ref_cyclohex") - E("benzene_CC_q+0") - 3 * EH2:+.3f} eV  (exp ~ -2.1)')

    # ---- small-molecule two-tip comparison --------------------------------
    if args.group in ('mol', 'all') and 'ref_H2' in res:
        EH2, muH = E('ref_H2'), E('ref_H2') / 2
        print('\n==== two-tip switch channels (relaxed, eV vs H2 reservoir) ====')
        print('1H  = E(1H)-E(0H)-mu_H     one-side hydrogenation (radical)')
        print('2H  = E(2H,S)-E(0H)-E(H2)  both-sides hydrogenation (closed shell)')
        print('ann = E(2H,S)+E(0H)-2*E(1H)  <0 => radical pairing/annihilation gain')
        for mol in MOLS:
            if f'{mol}_NN_q+0' not in res:
                continue
            print(f'-- {mol}')
            for v2, v1, base, tag in (('cc', 'cC', 'CC', 'CH2 tips'), ('nn', 'nN', 'NN', 'NH tips'), ('oo', 'oO', 'OO', 'OH tips')):
                h1 = E(f'{mol}_{v1}_q+0') - E(f'{mol}_{base}_q+0') - muH
                h2 = E(f'{mol}_{v2}_q+0') - E(f'{mol}_{base}_q+0') - EH2
                an = E(f'{mol}_{v2}_q+0') + E(f'{mol}_{base}_q+0') - 2 * E(f'{mol}_{v1}_q+0')
                t_s = ''
                kT = f'{mol}_{v2}_q+0_T'
                if kT in res and res[kT]['E_ha'] is not None and res[f'{mol}_{v2}_q+0']['E_ha'] is not None:
                    t_s = f' | E_T-E_S {1000 * (res[kT]["E_ha"] - res[f"{mol}_{v2}_q+0"]["E_ha"]) * HAU2EV:+.1f} meV'
                print(f'   {tag:9s}  1H {h1:+7.3f} | 2H {h2:+7.3f} | ann {an:+7.3f}{t_s}')
            print(f'   aza NN-for-CC (not an addition; bookkeeping only)')
        for mol in MOLS:
            if f'{mol}_NN_q+1' in res:
                fig_mol_report(mol, res, os.path.join(OUTDIR, f'report_{mol}.png'))
    if args.group in ('bat', 'all'):
        battery_verdicts(res)
        if 'ref_H2' in res:
            fig_battery_report(res, os.path.join(OUTDIR, 'battery_report.png'))
            fig_battery_lines(res, os.path.join(OUTDIR, 'battery_lines.png'))
            fig_battery_reservoir(res, os.path.join(OUTDIR, 'battery_reservoir.png'))
    if args.group in ('flake', 'all'):
        fig_geometry(os.path.join(OUTDIR, 'flakes_check.png'))
    if args.group in ('mol', 'all'):
        fig_mol_geometry(os.path.join(OUTDIR, 'mols_check.png'))
    if args.maps:
        resr = json.load(open(RESR)) if os.path.isfile(RESR) else {}
        for name in list(BASES) + list(EXT) if args.maps == 'all' else args.maps.split(','):
            print(f'-- bond maps {name}', flush=True)
            fig_bond_maps(name, resr)
    if args.sites:
        resr = json.load(open(RESS)) if os.path.isfile(RESS) else {}
        for name in list(SITE_FLAKES) if args.sites == 'all' else args.sites.split(','):
            print(f'-- site scan {name}', flush=True)
            fig_site_maps(name, resr)
            fig_site_energies(name, resr)
            site_j_analysis(name, resr)
    print(f'\nresults -> {RESULTS}')
    if args.export_pyscf:
        export_pyscf(args.export_pyscf, res)


if __name__ == '__main__':
    main()
