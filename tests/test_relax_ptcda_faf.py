"""PTCDA on NaCl via FAF substrate + fused UFF/SPFF/LFF multi-step kernels.

Uses charges from data/xyz/PTCDA.xyz (4th column on O/H) so FAF Coulomb
types distinguish anhydride O from C/H.

Artifacts: debug/test_relax_ptcda_faf/
  - ff_topology / lff_topology — intramolecular bonds and LFF K12/K13/K14 sticks
  - before/after .xyz + geometry png (xy+xz)
  - lff_outer_sweep.out — iterations vs geometry
  - speed_summary.out
"""
import os
import time
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spammm.AtomicSystem import AtomicSystem
from spammm.forcefields.UFF_cl import UFF_cl
from spammm.forcefields.LFFSolver import LFFSolver
from spammm.forcefields.FFController import FFController
from spammm.surfaces.FoldedRigid import (
    fit_folded_for_molecule, save_fit, load_fit, load_substrate, replicate_substrate,
    NACL_SUBSTRATE, eval_folded_potential,
)

PTCDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'xyz', 'PTCDA.xyz')
SUB = NACL_SUBSTRATE

NSTEPS = 8000
DT = 0.01
DAMP = 0.9
FLIMIT = 100.0
# LFF: large outer dt + Jacobi inners (projective); fewer outers than explicit MD
LFF_DT = 0.04
LFF_NINNER = 16
LFF_NOUTER = 200
LFF_OUTER_SWEEP = (50, 100, 200, 500, 1000)
# Start off-registry + in-plane rotation so lateral O→Na relaxation is visible
XY_SHIFT = (1.5, 1.0)  # Å
ROT_Z_DEG = 18.0       # in-plane rotation about molecule COM
Z_REL = 3.0            # start above expected contact
# AtomTypes Na RvdW≈1.45; R0(O–Na)=1.75+1.45=3.20 Å. Electrostatics from xyz charges.
R_NA_ION = 1.45
# Charges live in data/xyz/PTCDA.xyz (O=-0.4, H=+0.3 neutral) and NaCl (±1).
Q_O = -0.4
Q_H = 0.3
FIT_CACHE = f'ptcda_nacl_faf_NaR{R_NA_ION:.2f}_Qo{abs(Q_O):.1f}_qSub1.npz'


def _outdir():
    d = os.path.join('debug', 'test_relax_ptcda_faf')
    os.makedirs(d, exist_ok=True)
    return d


def _ensure_fit(outdir):
    cache = os.path.join(outdir, FIT_CACHE)
    if os.path.isfile(cache):
        print(f'[FAF] loading cached fit {cache}', flush=True)
        return load_fit(cache)
    print(f'[FAF] fitting PTCDA on NaCl R_Na={R_NA_ION} Q_O={Q_O} Q_H={Q_H} substrate±1 ...', flush=True)
    t0 = time.perf_counter()
    fit = fit_folded_for_molecule(
        os.path.abspath(PTCDA), substrate_file=os.path.abspath(SUB),
        substrate_R_override={'Na': R_NA_ION},
        q_override={'O': Q_O, 'H': Q_H},
        z_range_rel=(1.2, 8.0),
    )
    print(f'[FAF] fit done in {time.perf_counter()-t0:.1f}s  ntypes={fit["coeffs"].shape[0]} '
          f'nbasis={fit["coeffs"].shape[1]} unique_REQs=\n{fit["unique_REQs"]}', flush=True)
    save_fit(fit, cache)
    print(f'REVIEW: {cache}', flush=True)
    return fit


def _load_ptcda_placed():
    mol = AtomicSystem(fname=PTCDA)
    mol.enames = np.array([str(e) for e in mol.enames], dtype=object)
    if mol.bonds is None:
        mol.findBonds()
    mol.neighs()
    mol.atom_types_spff = ['C_R' if e == 'C' else e for e in mol.enames]
    if mol.qs is None:
        from spammm import atomicUtils as au
        _, _, _, qs, _ = au.load_xyz(fname=PTCDA, bReadN=True)
        mol.qs = np.asarray(qs, dtype=np.float64)
    else:
        mol.qs = np.asarray(mol.qs, dtype=np.float64).copy()
    for i, e in enumerate(mol.enames):
        if e == 'O':
            mol.qs[i] = Q_O
        elif e == 'H':
            mol.qs[i] = Q_H
    qtot = float(np.sum(mol.qs))
    assert abs(qtot) < 1e-9, f'PTCDA not charge-neutral: qtot={qtot}'
    sub = AtomicSystem(fname=SUB, bPreinit=False)
    z_top = float(np.max(sub.apos[:, 2]))
    mol.apos = np.asarray(mol.apos, dtype=np.float64).copy()
    # in-plane rotation about COM, then center on substrate + XY shift
    com = mol.apos.mean(axis=0)
    th = np.deg2rad(ROT_Z_DEG)
    c, s = np.cos(th), np.sin(th)
    xy = mol.apos[:, :2] - com[:2]
    mol.apos[:, 0] = com[0] + c * xy[:, 0] - s * xy[:, 1]
    mol.apos[:, 1] = com[1] + s * xy[:, 0] + c * xy[:, 1]
    mol.apos[:, 0] += (sub.apos[:, 0].mean() - mol.apos[:, 0].mean()) + XY_SHIFT[0]
    mol.apos[:, 1] += (sub.apos[:, 1].mean() - mol.apos[:, 1].mean()) + XY_SHIFT[1]
    mol.apos[:, 2] += (z_top + Z_REL) - float(mol.apos[:, 2].min())
    return mol, sub, z_top


def _o_na_stats(mol_apos, mol_enames, sub_apos, sub_enames):
    """Nearest Na and Cl for each O by full 3D distance: (io, dNa, dCl, dxy_Na, na_pos)."""
    Oidx = [i for i, e in enumerate(mol_enames) if e == 'O']
    Na = np.array([sub_apos[i] for i, e in enumerate(sub_enames) if e == 'Na'], dtype=np.float64)
    Cl = np.array([sub_apos[i] for i, e in enumerate(sub_enames) if e == 'Cl'], dtype=np.float64)
    rows = []
    for io in Oidx:
        dNa = np.linalg.norm(Na - mol_apos[io], axis=1)
        dCl = np.linalg.norm(Cl - mol_apos[io], axis=1)
        j = int(np.argmin(dNa))
        dxy = float(np.linalg.norm(Na[j, :2] - mol_apos[io, :2]))
        rows.append((io, float(dNa[j]), float(dCl.min()), dxy, Na[j].copy()))
    return rows


def _oc_delta_z(apos, enames):
    oi = [i for i, e in enumerate(enames) if e == 'O']
    ci = [i for i, e in enumerate(enames) if e == 'C']
    return float(apos[oi, 2].mean() - apos[ci, 2].mean())


def _mean_bond_len(apos, bonds):
    return float(np.mean([np.linalg.norm(apos[i] - apos[j]) for i, j in bonds]))


def _dump_ff_topology(outdir, mol):
    """Print + plot intramolecular FF bonds (findBonds / UFF neighs)."""
    bonds = list(mol.bonds)
    lines = [
        'PTCDA intramolecular FF topology (findBonds → UFF/SPFF)',
        f'natoms={len(mol.enames)} nbonds={len(bonds)}',
        '',
        f'{"ib":>4s} {"ia":>4s} {"ja":>4s} {"ei":>2s} {"ej":>2s} {"r":>8s}',
    ]
    for ib, (ia, ja) in enumerate(bonds):
        r = float(np.linalg.norm(mol.apos[ia] - mol.apos[ja]))
        lines.append(f'{ib:4d} {ia:4d} {ja:4d} {mol.enames[ia]:>2s} {mol.enames[ja]:>2s} {r:8.4f}')
    deg = np.zeros(len(mol.enames), dtype=int)
    for ia, ja in bonds:
        deg[ia] += 1; deg[ja] += 1
    lines.append('')
    lines.append('degree: ' + ', '.join(f'{i}:{mol.enames[i]}={deg[i]}' for i in range(len(mol.enames))))
    # UFF neighs after toUFF (same bond graph)
    uff = UFF_cl(bPrint=False)
    data = uff.toUFF(mol)
    neighs = data['neighs']
    lines.append('')
    lines.append('UFF neighs (per atom, -1 pad):')
    for ia in range(len(mol.enames)):
        ng = [int(x) for x in neighs[ia] if x >= 0]
        lines.append(f'  {ia:3d} {mol.enames[ia]} -> {ng}')
    out = os.path.join(outdir, 'ff_topology.out')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'REVIEW: {out}', flush=True)
    for L in lines[:8] + ['...'] + lines[-12:]:
        print(L, flush=True)

    # plot bonds on molecule
    png = os.path.join(outdir, 'ff_topology.png')
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    apos = np.asarray(mol.apos, dtype=np.float64)
    for ia, ja in bonds:
        ax.plot([apos[ia, 0], apos[ja, 0]], [apos[ia, 1], apos[ja, 1]], 'k-', lw=1.0, zorder=1)
    for e, c, s in (('C', '0.2', 40), ('O', 'tab:red', 70), ('H', '0.7', 22)):
        m = [i for i, ee in enumerate(mol.enames) if ee == e]
        if not m:
            continue
        p = apos[m]
        ax.scatter(p[:, 0], p[:, 1], c=c, s=s, label=e, zorder=2)
        for i in m:
            ax.text(apos[i, 0], apos[i, 1], str(i), fontsize=6, ha='center', va='bottom', zorder=3)
    ax.set_aspect('equal')
    ax.set_xlabel('x [A]'); ax.set_ylabel('y [A]')
    ax.set_title(f'PTCDA FF bonds (n={len(bonds)})')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f'REVIEW: {png}', flush=True)
    return bonds, out, png


def _faf_force_report(outdir, fit, apos, enames, z_top):
    """Finite-diff FAF Fz per element at current geometry (diagnostic)."""
    at = np.asarray(fit['atom_type_ids'], dtype=np.int32)
    h = 1e-3
    lines = ['FAF Fz (finite-diff) at initial placement',
             f'z_top={z_top:.4f} z_rel_min={apos[:,2].min()-z_top:.4f}',
             f'{"ia":>4s} {"e":>2s} {"typ":>3s} {"Fz":>10s} {"E":>10s}']
    by_e = {}
    for ia, e in enumerate(enames):
        t = int(at[ia])
        p = apos[ia].copy()
        pp, pm = p.copy(), p.copy(); pp[2] += h; pm[2] -= h
        Ep = float(eval_folded_potential(fit, t, pp[None, :])[0])
        Em = float(eval_folded_potential(fit, t, pm[None, :])[0])
        E = float(eval_folded_potential(fit, t, p[None, :])[0])
        Fz = -(Ep - Em) / (2 * h)
        lines.append(f'{ia:4d} {e:>2s} {t:3d} {Fz:10.5f} {E:10.5f}')
        by_e.setdefault(e, []).append(Fz)
    lines.append('')
    for e, fzs in by_e.items():
        lines.append(f'mean Fz({e})={np.mean(fzs):+.5f}  (negative = toward substrate)')
    path = os.path.join(outdir, 'faf_forces_init.out')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'REVIEW: {path}', flush=True)
    for L in lines[:3] + lines[-6:]:
        print(L, flush=True)
    return path


def _save_xyz_with_sub(path, tag, mol_e, mol_p, sub_e, sub_p, comment):
    with open(path, 'a' if tag == 'end' else 'w') as f:
        n = len(mol_e) + len(sub_e)
        f.write(f'{n}\n{comment}\n')
        for e, p in zip(sub_e, sub_p):
            f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
        for e, p in zip(mol_e, mol_p):
            f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')


def _plot_with_sub(path, title, mol_e, mol_init, mol_final, sub_e, sub_p, ona_init, ona_final, bonds, z_top):
    """2×2: xy (init/final) on top, xz side view (init/final) below to show O bending."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # --- xy ---
    for ax, pos, ona, ttl in (
        (axes[0, 0], mol_init, ona_init, 'initial'),
        (axes[0, 1], mol_final, ona_final, 'relaxed'),
    ):
        for e, c, s in (('Na', 'tab:blue', 18), ('Cl', 'tab:green', 12)):
            m = [i for i, ee in enumerate(sub_e) if ee == e]
            if not m:
                continue
            p = sub_p[m]
            ax.scatter(p[:, 0], p[:, 1], c=c, s=s, alpha=0.35, label=e, zorder=1)
        for ia, ja in bonds:
            ax.plot([pos[ia, 0], pos[ja, 0]], [pos[ia, 1], pos[ja, 1]], color='0.55', lw=0.8, zorder=2)
        for e, c, s in (('C', 'k', 28), ('O', 'tab:red', 55), ('H', '0.7', 16)):
            m = [i for i, ee in enumerate(mol_e) if ee == e]
            if not m:
                continue
            p = pos[m]
            ax.scatter(p[:, 0], p[:, 1], c=c, s=s, label=e, zorder=3)
        for io, dNa, dCl, dxy, na in ona:
            o = pos[io]
            ax.plot([o[0], na[0]], [o[1], na[1]], 'r-', lw=1.2, zorder=2)
            mid = 0.5 * (o[:2] + na[:2])
            prefer = 'Na' if dNa <= dCl else 'Cl'
            ax.text(mid[0], mid[1], f'{dNa:.2f}/{dCl:.2f}', color='darkred' if prefer == 'Na' else 'purple',
                    fontsize=7, ha='center', zorder=4)
        ax.set_aspect('equal')
        ax.set_xlabel('x [A]'); ax.set_ylabel('y [A]')
        ax.set_title(f'{title} {ttl} xy (dNa/dCl)')
        ax.legend(loc='upper right', fontsize=7, markerscale=0.8)

    # --- xz side view (O bending toward substrate) ---
    # show only top substrate layer near molecule x-range
    z_layer = float(np.max(sub_p[:, 2]))
    top = np.abs(sub_p[:, 2] - z_layer) < 0.15
    for ax, pos, ttl in (
        (axes[1, 0], mol_init, 'initial'),
        (axes[1, 1], mol_final, 'relaxed'),
    ):
        ax.axhline(z_top, color='0.3', ls='--', lw=0.9, label='z_top')
        ax.scatter(sub_p[top, 0], sub_p[top, 2], c='0.75', s=12, alpha=0.5, zorder=1)
        for ia, ja in bonds:
            ax.plot([pos[ia, 0], pos[ja, 0]], [pos[ia, 2], pos[ja, 2]], color='0.55', lw=0.8, zorder=2)
        for e, c, s in (('C', 'k', 28), ('O', 'tab:red', 70), ('H', '0.7', 16)):
            m = [i for i, ee in enumerate(mol_e) if ee == e]
            if not m:
                continue
            p = pos[m]
            ax.scatter(p[:, 0], p[:, 2], c=c, s=s, label=e, zorder=3)
        # drop lines from each O to z_top
        oi = [i for i, e in enumerate(mol_e) if e == 'O']
        ci = [i for i, e in enumerate(mol_e) if e == 'C']
        for io in oi:
            o = pos[io]
            ax.plot([o[0], o[0]], [o[2], z_top], 'r:', lw=0.7, alpha=0.6, zorder=2)
        dOCdz = float(pos[oi, 2].mean() - pos[ci, 2].mean()) if oi and ci else 0.0
        ax.set_aspect('equal')
        ax.set_xlabel('x [A]'); ax.set_ylabel('z [A]')
        ax.set_title(f'{title} {ttl} xz  dOCdz={dOCdz:+.3f} Å')
        ax.legend(loc='upper right', fontsize=7, markerscale=0.8)
        # zoom y around molecule + surface
        zmin = min(float(pos[:, 2].min()) - 0.5, z_top - 0.3)
        zmax = float(pos[:, 2].max()) + 1.0
        ax.set_ylim(zmin, zmax)
        xmid = float(pos[:, 0].mean())
        ax.set_xlim(xmid - 8, xmid + 8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f'REVIEW: {path}', flush=True)


def _permute_fit_atypes(fit, mol):
    at = np.asarray(fit['atom_type_ids'], dtype=np.int32).copy()
    perm = getattr(mol, 'perm_nodes_first', None)
    if perm is not None and len(perm) == len(at):
        at = at[np.asarray(perm, dtype=np.int32)]
    fit2 = dict(fit)
    fit2['atom_type_ids'] = at
    return fit2


def _dump_lff_topology(outdir, mol, sticks):
    """Plot K12/K13/K14 linearized springs for LFF."""
    from matplotlib.lines import Line2D
    pos = np.asarray(mol.apos, dtype=np.float64)
    enames = list(mol.enames)
    colors = {'bond': 'tab:red', 'angle': 'tab:green', 'dihedral': 'tab:blue'}
    counts = {t: 0 for t in colors}
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for i, j, l0, k, tag in sticks:
        counts[tag] = counts.get(tag, 0) + 1
        c = colors.get(tag, '0.5')
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color=c, lw=0.9 if tag == 'bond' else 0.55, alpha=0.75, zorder=1)
    for e, c, s in (('C', 'k', 28), ('O', 'tab:red', 70), ('H', '0.7', 16)):
        m = [i for i, ee in enumerate(enames) if ee == e]
        if m:
            ax.scatter(pos[m, 0], pos[m, 1], c=c, s=s, zorder=3)
    ax.set_aspect('equal')
    ax.set_title(f'LFF linearized springs  K12={counts.get("bond",0)} K13={counts.get("angle",0)} K14={counts.get("dihedral",0)}')
    ax.legend(handles=[
        Line2D([0], [0], color='tab:red', lw=2, label=f'K12 bond ({counts.get("bond",0)})'),
        Line2D([0], [0], color='tab:green', lw=2, label=f'K13 angle ({counts.get("angle",0)})'),
        Line2D([0], [0], color='tab:blue', lw=2, label=f'K14 dihedral ({counts.get("dihedral",0)})'),
    ], loc='upper right', fontsize=8)
    png = os.path.join(outdir, 'lff_topology.png')
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    plt.close(fig)
    out = os.path.join(outdir, 'lff_topology.out')
    lines = [
        'LFF linearized topology from UFF (K12 bonds, K13 angles, K14 dihedrals)',
        f'nsticks={len(sticks)}  K12={counts.get("bond",0)} K13={counts.get("angle",0)} K14={counts.get("dihedral",0)}',
        'i j tag l0 k',
    ]
    for i, j, l0, k, tag in sticks:
        lines.append(f'{i:3d} {j:3d} {tag:8s} {l0:8.4f} {k:10.4f}')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'REVIEW: {out}', flush=True)
    print(f'REVIEW: {png}', flush=True)
    return out, png


@pytest.mark.gpu
@pytest.mark.slow
def test_ptcda_faf_fused_pipelines(make_review):
    """SPFF serial/global + UFF local/global + LFF FAF on PTCDA+NaCl; artifacts + speed table."""
    rev = make_review('ptcda_faf')
    outdir = _outdir()
    assert os.path.isfile(PTCDA), PTCDA
    assert os.path.isfile(SUB), SUB

    fit = _ensure_fit(outdir)
    assert fit['coeffs'].shape[1] <= 128, f"nbasis={fit['coeffs'].shape[1]} > FAF_BASIS_MAX"

    mol0, sub, z_top = _load_ptcda_placed()
    enames0 = list(mol0.enames)
    apos0 = np.asarray(mol0.apos, dtype=np.float64).copy()
    qs = np.asarray(mol0.qs, dtype=np.float64)
    print(f'[PTCDA] natoms={len(enames0)} qs(O)={qs[[i for i,e in enumerate(enames0) if e=="O"]]}', flush=True)

    bonds, topo_out, topo_png = _dump_ff_topology(outdir, mol0)
    rev.out(f'topology nbonds={len(bonds)} REVIEW: {topo_out} REVIEW: {topo_png}')
    _faf_force_report(outdir, fit, apos0, enames0, z_top)

    # LFF spring topology (from UFF linearization)
    uff_topo = UFF_cl(bPrint=False)
    uff_data0 = uff_topo.toUFF(mol0)
    lff_probe = LFFSolver(bPrint=False)
    sticks0 = lff_probe.from_uff(uff_data0, mol=mol0, mass=1.0)
    lff_topo_out, lff_topo_png = _dump_lff_topology(outdir, mol0, sticks0)
    rev.out(f'LFF sticks={len(sticks0)} REVIEW: {lff_topo_out} REVIEW: {lff_topo_png}')

    sub_apos, sub_enames, _, sub_lvec = load_substrate(SUB)
    sub_rep, sub_rep_e = replicate_substrate(
        sub_apos, sub_enames, sub_lvec, (-12, 12), (-12, 12), z_min=float(np.min(sub_apos[:, 2])) - 1.0)
    sub_rep = np.asarray(sub_rep, dtype=np.float64)
    sub_rep_e = list(sub_rep_e)

    summary = [
        f'PTCDA + FAF NaCl fused pipelines  z_top={z_top:.3f} XY_SHIFT={XY_SHIFT} ROT_Z={ROT_Z_DEG} Z_REL={Z_REL}',
        f'R_Na={R_NA_ION} Q_O={Q_O} Q_H={Q_H} (Morse R0(O-Na)={1.75+R_NA_ION:.2f}; substrate q=±1; COULOMB_CONST=14.40)',
        f'nsteps={NSTEPS} dt={DT} damp={DAMP}  nbasis={fit["coeffs"].shape[1]} ntypes={fit["coeffs"].shape[0]}',
        f'LFF: nOuter={LFF_NOUTER} nInner={LFF_NINNER} dt={LFF_DT} damp={DAMP} (sweep {LFF_OUTER_SWEEP})',
        f'device=NVIDIA (select_device preferred_vendor=nvidia)',
        '',
        f'{"tag":22s} {"t_s":>8s} {"E":>10s} {"fmax":>8s} {"z_min":>7s} {"meanBL":>7s} {"dONa3":>7s} {"dOCdz":>7s}',
    ]
    results = []

    def _run_case(tag, title, runner_factory):
        mol, _, _ = _load_ptcda_placed()
        apos_init = np.asarray(mol.apos, dtype=np.float64).copy()
        enames = list(mol.enames)
        t_relax, E, fmax, apos_final = runner_factory(mol, fit)
        ona_i = _o_na_stats(apos_init, enames, sub_rep, sub_rep_e)
        ona_f = _o_na_stats(apos_final, enames, sub_rep, sub_rep_e)
        mean_d3 = float(np.mean([dNa for _, dNa, _, _, _ in ona_f]))
        n_pref_Na = sum(1 for _, dNa, dCl, _, _ in ona_f if dNa <= dCl)
        mean_bl = _mean_bond_len(apos_final, bonds)
        dOCdz = _oc_delta_z(apos_final, enames)
        z_min = float(apos_final[:, 2].min())
        xyz = os.path.join(outdir, f'{tag}_init_final.xyz')
        if os.path.isfile(xyz):
            os.remove(xyz)
        _save_xyz_with_sub(xyz, 'start', enames, apos_init, sub_rep_e, sub_rep,
                           f'{tag} start mean_ONa_d3={np.mean([dNa for _,dNa,_,_,_ in ona_i]):.3f}')
        _save_xyz_with_sub(xyz, 'end', enames, apos_final, sub_rep_e, sub_rep,
                           f'{tag} end E={E:.4f} mean_ONa_d3={mean_d3:.3f} dOCdz={dOCdz:.4f} nNa={n_pref_Na}/6')
        print(f'REVIEW: {xyz}', flush=True)
        png = os.path.join(outdir, f'{tag}_geometry.png')
        _plot_with_sub(png, title, enames, apos_init, apos_final, sub_rep_e, sub_rep, ona_i, ona_f, bonds, z_top)
        line = (f'{tag:22s} {t_relax:8.4f} {E:10.4f} {fmax:8.4f} {z_min:7.4f} {mean_bl:7.4f} {mean_d3:7.4f} {dOCdz:7.4f}')
        summary.append(line)
        results.append((tag, t_relax, E, fmax, z_min, mean_bl, mean_d3, dOCdz, ona_i, ona_f))
        out = os.path.join(outdir, f'{tag}.out')
        with open(out, 'w') as f:
            f.write('\n'.join([
                title,
                f't_relax={t_relax:.4f}s E={E:.6f} fmax={fmax:.6f} z_min={z_min:.4f}',
                f'meanBL={mean_bl:.4f} (init {_mean_bond_len(apos_init, bonds):.4f})  dOCdz={dOCdz:.4f} (O_z-C_z; <0 => O toward substrate)',
                f'O closer to Na: {n_pref_Na}/6',
                'O-Na/Cl_d3 init:  ' + ', '.join(f'O{io}:{dNa:.3f}/{dCl:.3f}' for io, dNa, dCl, _, _ in ona_i),
                'O-Na/Cl_d3 final: ' + ', '.join(f'O{io}:{dNa:.3f}/{dCl:.3f}' for io, dNa, dCl, _, _ in ona_f),
                f'REVIEW: {xyz}', f'REVIEW: {png}',
            ]) + '\n')
        print(f'REVIEW: {out}', flush=True)
        for L in (line, f'  O-Na/Cl final: {[(round(dNa,3), round(dCl,3)) for _,dNa,dCl,_,_ in ona_f]}  nNa={n_pref_Na}/6 dOCdz={dOCdz:.4f}'):
            rev.out(L)
            print(L, flush=True)
        assert np.isfinite(apos_final).all() and np.isfinite(E)
        assert z_min - z_top > 0.8, f'{tag} crashed into surface: z_min={z_min} z_top={z_top}'
        # intramolecular: do not collapse (UFF angle bug previously drove meanBL~1.06)
        assert mean_bl > 1.20, f'{tag} intramolecular collapse meanBL={mean_bl}'
        spanx = float(apos_final[:, 0].max() - apos_final[:, 0].min())
        assert spanx > 8.0, f'{tag} lateral crumple spanx={spanx:.2f}'
        return mean_d3, dOCdz, n_pref_Na

    def spff_serial(mol, fit0):
        ctrl = FFController(enable_nonbond=False)
        info = ctrl.build_ff(mol, ff_type='spff')
        fit_u = _permute_fit_atypes(fit0, mol)
        ctrl.md.upload_folded_fit(fit_u)
        device = ctrl.md.ctx.devices[0].name
        assert 'nvidia' in device.lower() or 'rtx' in device.lower() or 'geforce' in device.lower()
        print(f'[SPFF serial] {info} device={device}', flush=True)
        t0 = time.perf_counter()
        ctrl.md.relax_serial(nsteps=NSTEPS, dt=DT, damp=DAMP, Flimit=FLIMIT, do_faf=True)
        t = time.perf_counter() - t0
        E = float(ctrl.md.get_total_energy())
        fmax = float(ctrl.get_fmax())
        apos = np.asarray(ctrl.get_positions(), dtype=np.float64)
        ctrl.teardown()
        return t, E, fmax, apos

    def spff_global(mol, fit0):
        ctrl = FFController(enable_nonbond=False)
        ctrl.build_ff(mol, ff_type='spff')
        fit_u = _permute_fit_atypes(fit0, mol)
        ctrl.md.upload_folded_fit(fit_u)
        t0 = time.perf_counter()
        ctrl.md.relax_global(nsteps=NSTEPS, dt=DT, damp=DAMP, Flimit=FLIMIT, wg=256, do_faf=True)
        t = time.perf_counter() - t0
        E = float(ctrl.md.get_total_energy())
        fmax = float(ctrl.get_fmax())
        apos = np.asarray(ctrl.get_positions(), dtype=np.float64)
        ctrl.teardown()
        return t, E, fmax, apos

    def uff_local(mol, fit0):
        uff = UFF_cl(bPrint=False)
        uff.toUFF(mol)
        masses = np.array([{'H': 1., 'C': 12., 'O': 16.}.get(e, 12.) for e in mol.enames], dtype=np.float32)
        uff.upload_positions(mol.apos, masses=masses)
        uff.upload_folded_fit(fit0)
        t0 = time.perf_counter()
        E = float(np.asarray(uff.relax_serial(nsteps=NSTEPS, dt=DT, damp=0.95, Flimit=FLIMIT, wg=128, do_faf=True)).ravel()[0])
        t = time.perf_counter() - t0
        fmax = float(np.max(np.linalg.norm(uff.get_forces()[0], axis=1)))
        apos = np.asarray(uff.get_positions(), dtype=np.float64)
        return t, E, fmax, apos

    def uff_global(mol, fit0):
        uff = UFF_cl(bPrint=False)
        uff.toUFF(mol)
        masses = np.array([{'H': 1., 'C': 12., 'O': 16.}.get(e, 12.) for e in mol.enames], dtype=np.float32)
        uff.upload_positions(mol.apos, masses=masses)
        uff.upload_folded_fit(fit0)
        t0 = time.perf_counter()
        E = float(np.asarray(uff.relax_global(nsteps=NSTEPS, dt=DT, damp=0.95, Flimit=FLIMIT, wg=256, do_faf=True)).ravel()[0])
        t = time.perf_counter() - t0
        fmax = float(np.max(np.linalg.norm(uff.get_forces()[0], axis=1)))
        apos = np.asarray(uff.get_positions(), dtype=np.float64)
        return t, E, fmax, apos

    def lff_faf(mol, fit0, n_outer=LFF_NOUTER):
        uff = UFF_cl(bPrint=False)
        uff_data = uff.toUFF(mol)
        lff = LFFSolver(bPrint=False)
        lff.from_uff(uff_data, mol=mol, mass=1.0)
        lff.upload_folded_fit(fit0)
        t0 = time.perf_counter()
        st = lff.relax(n_outer=n_outer, n_inner=LFF_NINNER, dt=LFF_DT, damp=DAMP, do_faf=True)
        t = time.perf_counter() - t0
        apos = np.asarray(st['pos'][:, :3], dtype=np.float64)
        # LFF has no force buffer; report displacement from init as proxy
        disp = float(np.max(np.linalg.norm(apos - np.asarray(mol.apos, dtype=np.float64), axis=1)))
        E = 0.0  # no energy accumulator in LFF kernel yet
        return t, E, disp, apos

    mean_spff, dOCdz_spff, nNa_spff = _run_case('spff_serial_faf', 'SPFF serial+FAF', spff_serial)
    _run_case('spff_global_faf', 'SPFF global+FAF', spff_global)
    _run_case('uff_local_faf', 'UFF local+FAF', uff_local)
    _run_case('uff_global_faf', 'UFF global+FAF', uff_global)
    _run_case('lff_faf', f'LFF+FAF nOuter={LFF_NOUTER}', lambda mol, fit0: lff_faf(mol, fit0, LFF_NOUTER))

    # LFF outer-step sweep (iteration count vs geometry)
    sweep_lines = ['LFF outer sweep (nInner=%d dt=%.3f)' % (LFF_NINNER, LFF_DT),
                   f'{"nOuter":>8s} {"t_s":>8s} {"meanBL":>8s} {"dOCdz":>8s} {"dONa3":>8s}']
    for nout in LFF_OUTER_SWEEP:
        mol_s, _, _ = _load_ptcda_placed()
        t, E, fmax, apos_s = lff_faf(mol_s, fit, n_outer=nout)
        mb = _mean_bond_len(apos_s, bonds)
        dz = _oc_delta_z(apos_s, list(mol_s.enames))
        ona_s = _o_na_stats(apos_s, list(mol_s.enames), sub_rep, sub_rep_e)
        md3 = float(np.mean([dNa for _, dNa, _, _, _ in ona_s]))
        sweep_lines.append(f'{nout:8d} {t:8.4f} {mb:8.4f} {dz:8.4f} {md3:8.4f}')
        print(f'[LFF sweep] nOuter={nout} t={t:.4f}s meanBL={mb:.4f} dOCdz={dz:.4f} dONa3={md3:.4f}', flush=True)
    sweep_path = os.path.join(outdir, 'lff_outer_sweep.out')
    with open(sweep_path, 'w') as f:
        f.write('\n'.join(sweep_lines) + '\n')
    print(f'REVIEW: {sweep_path}', flush=True)
    for L in sweep_lines:
        rev.out(L)

    ona0 = _o_na_stats(apos0, enames0, sub_rep, sub_rep_e)
    mean0_d3 = float(np.mean([dNa for _, dNa, _, _, _ in ona0]))
    summary.append('')
    summary.append(f'mean_ONa_d3_init={mean0_d3:.4f}  mean_ONa_d3_spff_serial={mean_spff:.4f}  dOCdz_spff={dOCdz_spff:.4f}  nNa_spff={nNa_spff}/6')
    summary.append('Labels on PNG: dNa/dCl. Prefer Na (red); purple label => closer to Cl (bad).')
    summary.append(f'Target: O→Na (not Cl). See O_neg_potential_Na_vs_Cl.png for P/L/Q components.')
    summary.append(f'Start: ROT_Z={ROT_Z_DEG}° + XY_SHIFT={XY_SHIFT}.')
    sum_path = os.path.join(outdir, 'speed_summary.out')
    with open(sum_path, 'w') as f:
        f.write('\n'.join(summary) + '\n')
    print(f'REVIEW: {sum_path}', flush=True)
    for L in summary:
        rev.out(L)
        print(L, flush=True)
    rev.finish()

    assert mean_spff < mean0_d3 + 0.5, f'O-Na d3 worsened: init={mean0_d3} final={mean_spff}'
    # Registry: FAF 1D scan prefers Na; 3D d(O–Cl)<d(O–Na) can still happen if O sits
    # between ions at low z — report, don't hard-fail until MD is fully settled (fmax).
    if nNa_spff < 4:
        print(f'WARNING: only {nNa_spff}/6 O closer to Na in 3D — check geometry PNG + fmax', flush=True)
