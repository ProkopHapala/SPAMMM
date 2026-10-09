#!/usr/bin/env python3
"""DEBUG: systematic core-basis survey for the poly8 core+mesh FDBM representation.

Step 1 (--cache): run FDBM (FDBMPipeline.run_fields, same params as
  testplot_fdbm_fields_compress) and sample E,F of the oracle on a 0.25 Å grid
  covering atoms ± XY_MARGIN, z in [mol_z-1, mol_z+8]. -> debug/testplot_coremesh_basis_survey/<mol>_cache.npz
Step 2 (default): for each basis spec fit core (LSQ on shell rin<r_min<rcut, z>mol_z+z0,
  optional force rows), then weighted B-spline mesh on E-core (h=1 Å), and score
  E shell rms, Fz residual (exact oracle F) at h=2.5/3.0/3.5 for core-only and core+mesh.

Basis spec string (SSOT: spammm.surfaces.CoreBasisStudy): 'A:s3p2d1' atoms with
  3 s-, 2 p-, 1 d-radials; '+B:s2' bond centers; '@dz0.5' shifts atom centers up.
  <l><n> = n radii geomspace(RHI[l],RLO,n) (n=1 -> R1[l]); <l>[r1,r2,...] explicit.
Canonical core-only residual figure: surface_plots.plot_core_residual_study.
"""
import os, sys, time, argparse
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from spammm.surfaces.CoreBasisStudy import parse_spec, build_terms, cost_per_query, fit_core_lsq, eval_terms, core_residual_study, spec_radials
from spammm.surfaces.surface_plots import plot_core_residual_study

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'testplot_coremesh_basis_survey')
MOLS = {'azaindol': 'data/xyz/azaindol.xyz', 'PTCDA': 'data/xyz/PTCDA.xyz', 'pentacene': 'data/xyz/pentacene.xyz'}
XY_MARGIN, Z_LO, Z_HI, DG = 5.0, -1.0, 8.0, 0.25

# ======================= step 1: cache =======================
def make_cache(mol, step=0.1):
    from spammm import atomicUtils as au
    from spammm.SPM import AFM as afm, AFM_utils as afm_utils
    from spammm.SPM.FDBMPipeline import FDBMPipeline
    from spammm.config_utils import get_dftb_basis_path
    from spammm.forcefields.FFController import make_planar_xy, orient_long_axis_x
    atomPos, _, enames, _, _ = au.load_xyz(MOLS[mol])
    ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8}
    atomTypes = np.array([ELEM_Z[e] for e in enames], np.int32)
    atomPos = np.asarray(atomPos, np.float64); atomPos[:] = make_planar_xy(atomPos); orient_long_axis_x(atomPos); atomPos[:, 2] = 0.0
    grid_spec, origin, ngrid, step = afm_utils.make_fdbm_grid_com_zsym(atomPos, step, XY_MARGIN + 3.0, z_vac=Z_HI + 3.0)
    pipe = FDBMPipeline(); pa = afm.PAULI_FITTED_DEFAULTS['3ob-3-1']
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = pipe.run_fields(atomPos, atomTypes, get_dftb_basis_path('3ob-3-1'), os.path.join(OUT, f'{mol}_work'),
                          grid_spec, origin, step, ngrid, float(pa['A']), float(pa['beta']), 'co', oracle=True)
    sample = res['prolonged']['sample']
    lo = np.array([np.floor(atomPos[:, 0].min() - XY_MARGIN), np.floor(atomPos[:, 1].min() - XY_MARGIN), Z_LO])
    hi = np.array([np.ceil(atomPos[:, 0].max() + XY_MARGIN), np.ceil(atomPos[:, 1].max() + XY_MARGIN), Z_HI])
    ns = np.round((hi - lo) / DG).astype(int) + 1                       # integer-Å box -> (n-1)%4==0
    P = np.stack(np.meshgrid(*[lo[a] + np.arange(ns[a]) * DG for a in range(3)], indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    E, F = [], []
    for i in range(0, len(P), 262144):
        e, f = sample(P[i:i + 262144]); E.append(np.asarray(e)); F.append(np.asarray(f))
    E = np.concatenate(E); F = np.concatenate(F)
    assert np.isfinite(E).all() and np.isfinite(F).all(), 'oracle NaN (sample box outside FDBM grid?)'
    f = os.path.join(OUT, f'{mol}_cache.npz')
    np.savez(f, apos=atomPos, enames=np.array(enames), lo=lo, dg=DG, ns=ns, E=E.reshape(ns), F=F.reshape(*ns, 3))
    print(f'[cache] {mol}: {len(atomPos)} atoms grid {tuple(ns)} ({len(P)} pts) FDBM+sample {time.time()-t0:.1f}s -> {os.path.abspath(f)}', flush=True)

# ======================= step 2: survey =======================
def survey(mol, specs, wF=0.0, hm=1.0, lam=1e-3, rin=2.5, rcut=4.0, z0=0.5, bPlot=True):
    from spammm.surfaces.CoarseMesh import fit_coremesh_lsq
    d = np.load(os.path.join(OUT, f'{mol}_cache.npz'))
    A = d['apos']; lo = d['lo']; ns = tuple(d['ns']); dg = float(d['dg']); E3 = d['E'].astype(float); F3 = d['F'].astype(float)
    P3 = np.stack(np.meshgrid(*[lo[a] + np.arange(ns[a]) * dg for a in range(3)], indexing='ij'), -1)
    P = P3.reshape(-1, 3); E = E3.ravel(); F = F3.reshape(-1, 3); mol_z = A[:, 2].max()
    rm = np.min(np.linalg.norm(P[:, None, :] - A[None], axis=-1), axis=1)
    mfit = (rm > rin) & (rm < rcut) & (P[:, 2] > mol_z + z0)
    wmesh = ((rm > rin) & (P[:, 2] > mol_z + z0)).reshape(ns).astype(float)
    s = int(round(hm / dg))
    iz_h = {h: int(round((mol_z + h - lo[2]) / dg)) for h in (2.5, 3.0, 3.5)}
    mxy = (P3[..., 0] > A[:, 0].min() - 2) & (P3[..., 0] < A[:, 0].max() + 2) & (P3[..., 1] > A[:, 1].min() - 2) & (P3[..., 1] < A[:, 1].max() + 2)
    print(f'\n=== {mol}: {len(A)} atoms grid {ns} fit samples {mfit.sum()} wF={wF} ===', flush=True)
    print(f'{"spec":26s} {"nc":>4s} {"ncoef":>5s} {"cost":>6s} {"cond":>8s} {"Eshell":>7s} | {"core Fz 2.5/3.0/3.5 [meV/A]":>28s} | {"core+mesh Fz 2.5/3.0/3.5":>26s} | {"cm E<0.3":>8s}', flush=True)
    rows = []; maps = {}
    Pf = P[mfit]
    for spec in specs:
        sets = build_terms(A, parse_spec(spec)); t0 = time.time()
        c, ME = fit_core_lsq(Pf, E[mfit], F[mfit], sets, wF=wF)
        cond = np.linalg.cond(ME); es = 1e3 * np.sqrt(np.mean((ME @ c - E[mfit])**2))
        Ec, Fzc = eval_terms(P, sets, c)
        mesh, _, _ = fit_coremesh_lsq((E - Ec).reshape(ns), np.empty((0, 3)), lo, hm, np.empty(0), np.empty(0), s=s, weights=wmesh, lam=lam, tol=1e-7, maxiter=20000)
        g = ((P - lo) / hm).T; dz = 0.01
        Em = map_coordinates(mesh.coeffs, g, order=3, mode='nearest', prefilter=False)
        gp = g.copy(); gp[2] += dz / hm; gm = g.copy(); gm[2] -= dz / hm
        Fzm = -(map_coordinates(mesh.coeffs, gp, order=3, mode='nearest', prefilter=False) - map_coordinates(mesh.coeffs, gm, order=3, mode='nearest', prefilter=False)) / (2 * dz)
        Fz_ref = F[:, 2]; err_c = (Fzc - Fz_ref).reshape(ns); err_cm = (Fzc + Fzm - Fz_ref).reshape(ns)
        fz = lambda e: [1e3 * np.sqrt(np.mean(e[..., iz][mxy[..., iz] & (rm.reshape(ns)[..., iz] > rin)]**2)) for iz in iz_h.values()]
        low = (E < 0.3) & (rm > rin) & (P[:, 2] > mol_z + z0)
        ecm = 1e3 * np.sqrt(np.mean((Ec + Em - E)[low]**2))
        r = dict(spec=spec, nc=sum(len(C) for C, _ in sets), ncoef=len(c), cost=cost_per_query(sets), cond=cond, Eshell=es, fzc=fz(err_c), fzcm=fz(err_cm), ecm=ecm, maxc=np.abs(c).max())
        rows.append(r); maps[spec] = (err_c, err_cm)
        print(f'{spec:26s} {r["nc"]:4d} {r["ncoef"]:5d} {r["cost"]:6d} {cond:8.1e} {es:7.2f} | {r["fzc"][0]:8.1f} {r["fzc"][1]:8.2f} {r["fzc"][2]:8.2f}   | {r["fzcm"][0]:8.1f} {r["fzcm"][1]:7.2f} {r["fzcm"][2]:7.2f}  | {ecm:7.2f}   ({time.time()-t0:.1f}s maxc={r["maxc"]:.1e})', flush=True)
    return rows, maps, (P3, A, iz_h, mol_z, ns)

def plot_pareto(all_rows, fname):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for mol, rows in all_rows.items():
        for k, (ax, key, idx, ttl) in enumerate(((axs[0], 'fzcm', 0, 'core+mesh Fz rms @h=2.5 [meV/Å]'), (axs[1], 'fzcm', 1, 'core+mesh Fz rms @h=3.0'), (axs[2], 'ecm', None, 'core+mesh E rms (E<0.3) [meV]'))):
            x = [r['cost'] / rows[0]['cost'] for r in rows]; y = [r[key][idx] if idx is not None else r[key] for r in rows]
            ax.plot(x, y, 'o', label=mol)
            for xi, yi, r in zip(x, y, rows): ax.annotate(r['spec'], (xi, yi), fontsize=6)
            ax.set_xlabel('eval cost relative to baseline (1st spec)'); ax.set_title(ttl); ax.set_yscale('log'); ax.grid(alpha=0.3)
    axs[0].legend(); fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)

def plot_maps(mol, maps, geo, specs, fname, h=2.5):
    P3, A, iz_h, mol_z, ns = geo; iz = iz_h[h]
    fig, axs = plt.subplots(2, len(specs), figsize=(3.2 * len(specs), 6.0), squeeze=False)
    ext = [P3[0, 0, 0, 0], P3[-1, 0, 0, 0], P3[0, 0, 0, 1], P3[0, -1, 0, 1]]
    for j, sp in enumerate(specs):
        for i, (lab, M) in enumerate(zip(('core', 'core+mesh'), maps[sp])):
            S = M[..., iz]; v = max(0.005, np.nanpercentile(np.abs(S), 99.5))
            axs[i, j].imshow(S.T, origin='lower', extent=ext, cmap='bwr', vmin=-v, vmax=v, aspect='equal')
            axs[i, j].plot(A[:, 0], A[:, 1], 'k.', ms=1.5); axs[i, j].set_title(f'{sp}\n{lab} ±{1e3*v:.0f} meV/Å', fontsize=7)
    fig.suptitle(f'{mol}: Fz residual (model − FDBM) at h={h} Å'); fig.tight_layout(); fig.savefig(fname, dpi=120); plt.close(fig)

def core_study(mol, specs, rcuts, wFs, rins=(2.5,), z0=0.5, sigma=1.0, hs=(2.5, 3.0, 3.5, 4.0)):
    """Thin runner: load cache -> CoreBasisStudy.core_residual_study over variants
    (wF x rcut x rin) -> print table. Figures via surface_plots.plot_core_residual_study.
    Returns {spec: res}, {spec: geo}."""
    d = np.load(os.path.join(OUT, f'{mol}_cache.npz'))
    A = d['apos']; lo = d['lo']; dg = float(d['dg']); E3 = d['E'].astype(float); F3 = d['F'].astype(float)
    bands = ((2.5, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 7.0))
    print(f'\n=== {mol} CORE-ONLY residual (sigma_HP={sigma} A) ===', flush=True)
    print(f'{"spec":16s} {"wF":>4s} {"rcut":>4s} {"rin":>4s} {"ncoef":>5s} {"cond":>9s} {"maxc":>9s} | E rms by r_min band [meV] ' +
          ' '.join(f'{a:.1f}-{b:.1f}' for a, b in bands) + ' | Fz rms h=' + '/'.join(f'{h:g}' for h in hs) + ' [meV/A] | HP: E [meV] Fz [meV/A]', flush=True)
    res_all, geos, rows_by = {}, {}, {}
    for spec in specs:
        variants = [dict(label=variant_label(rc, ri, wF), rcut=rc, rin=ri, wF=wF) for wF in wFs for rc in rcuts for ri in rins]
        res, geo = core_residual_study(A, E3, F3, lo, dg, spec, variants, z0=z0, sigma=sigma, hs=hs, bands=bands)
        for v in variants:
            r = res[v['label']]
            print(f'{spec:16s} {v["wF"]:4.1f} {v["rcut"]:4.1f} {v["rin"]:4.1f} {r["ncoef"]:5d} {r["cond"]:9.1e} {r["maxc"]:9.1e} | ' +
                  ' '.join(f'{e:7.2f}' for e in r['eb']) + ' | ' + '/'.join(f'{x:.1f}' for x in r['fz']) + f' | {r["hpe"]:6.2f} {r["hpf"]:6.2f}', flush=True)
        res_all[spec] = res; geos[spec] = geo
        rows_by[spec] = {wF: [variant_label(rc, ri, wF) for rc in rcuts for ri in rins] for wF in wFs}
    return res_all, geos, rows_by

def variant_label(rcut, rin, wF):
    return f'rcut={rcut:g} rin={rin:g} wF={wF:g}'

SPECS = ['A:s3+B:s2',                              # current baseline (at3bd2)
         'A:s3', 'A:s4',                           # atom-only radial
         'A:s3@dz0.5', 'A:s3@dz1.0',               # shifted centers (vs pz)
         'A:s3p1', 'A:s3p2', 'A:s3p3',             # sp, decreasing p radials
         'A:s2p1d1', 'A:s3p1d1', 'A:s3p2d1',       # with d
         'A:s3p1+B:s1', 'A:s3p2+B:s1', 'A:s3+B:s1', # hybrid
         'A:s3p2d1@dz0.5']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mols', nargs='*', default=['azaindol', 'PTCDA', 'pentacene'])
    ap.add_argument('--cache', action='store_true'); ap.add_argument('--wF', type=float, default=0.0)
    ap.add_argument('--specs', nargs='*', default=None)
    ap.add_argument('--rcuts', nargs='*', type=float, default=None, help='core-only study over fit-shell outer cutoffs')
    ap.add_argument('--wFs', nargs='*', type=float, default=[0.0, 0.3])
    ap.add_argument('--rins', nargs='*', type=float, default=[2.5], help='fit-shell inner cutoffs (r_min)')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.cache:
        for m in args.mols: make_cache(m)
    elif args.rcuts:
        specs = args.specs or ['A:s3p2+B:s1', 'A:s3p2']
        for m in args.mols:
            res_all, geos, rows_by = core_study(m, specs, args.rcuts, args.wFs, rins=tuple(args.rins))
            for sp in specs:
                print(f'  [{sp}] radials: {spec_radials(sp)}', flush=True)
                for wF in args.wFs:
                    tag = ''.join(c if c.isalnum() else '_' for c in sp)
                    f = os.path.join(OUT, f'{m}_corestudy_{tag}_wF{wF:g}' + (f'_rin{min(args.rins):g}-{max(args.rins):g}' if args.rins != [2.5] else '') + '.png')
                    plot_core_residual_study(res_all[sp], geos[sp], rows_by[sp][wF], f,
                        title=f'{m} CORE-ONLY residual (FDBM - core), basis {sp}, wF={wF}; shared scale/col; NaN=excluded')
                    print(f'REVIEW: {os.path.abspath(f)}', flush=True)
    else:
        specs = args.specs or SPECS; all_rows = {}
        for m in args.mols:
            rows, maps, geo = survey(m, specs, wF=args.wF); all_rows[m] = rows
            best = sorted(rows, key=lambda r: r['fzcm'][0])[:3]
            fm = os.path.join(OUT, f'{m}_maps_wF{args.wF:g}.png'); plot_maps(m, maps, geo, [specs[0]] + [b['spec'] for b in best if b['spec'] != specs[0]][:3], fm)
            print(f'REVIEW: {os.path.abspath(fm)}', flush=True)
        fp = os.path.join(OUT, f'pareto_wF{args.wF:g}.png'); plot_pareto(all_rows, fp); print(f'REVIEW: {os.path.abspath(fp)}', flush=True)
