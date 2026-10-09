"""
CoreBasisStudy.py — experimental core-basis machinery for the core+mesh FDBM fit.

Essence: grammar + design matrix + LSQ fit for compact atom/bond-centered bases
(phi=(1-r/R)_+^N times s/p/d angular factors) and the CORE-ONLY residual study
(FDBM - core) used to judge whether the leftover is smooth enough for a coarse
cubic B-spline mesh. Canonical figure: surface_plots.plot_core_residual_study.

Design:
  - This is the CPU survey/design path, NOT the production GPU basis — the
    compiled kernel still uses the fixed 5-slot poly8 ladder (PICCore.POLY_CORE_*).
    Once a spec is chosen here it must be ported to cs_pme_core_basis.
  - Spec string: 'A:s3p2d1+B:s2@dz0.5'. A=atoms, B=bond midpoints (dbond=1.6 A).
    Per group, <l><n> adds n radii geomspace(RHI[l]->RLO) (n=1 -> R1[l]);
    <l>[r1,r2,...] gives explicit radii in Ang. Angular multiplicities:
    s=1, p=(nx,ny,nz), d=5 real (nx*ny, nx*nz, ny*nz, nx^2-ny^2, 3nz^2-1).
  - Fit is weighted LSQ on shell rin<r_min<rcut, z>mol_z+z0; wF>0 appends
    force rows wF*dE/dx against -wF*F (force-aware fit).
  - Smoothness metric HP: residual minus masked normalized Gaussian smooth of
    itself (sigma, default 1 A) = the part a ~sigma-resolution mesh cannot hold.
"""
import numpy as np

POLY_N = 8                                                     # power of (1-r/R)_+^N
RHI = {'s': 9.0, 'p': 7.0, 'd': 6.0}                           # outermost radius per channel
R1 = {'s': 5.0, 'p': 4.5, 'd': 4.0}                            # radius used when n==1
RLO = 3.5                                                      # innermost ladder radius


def ang_funcs(l):
    """List of (g(n), G(n)) — g angular value, G = dg/dn (n treated as free 3-vector)."""
    if l == 's': return [(lambda n: np.ones(n.shape[:-1]), lambda n: np.zeros(n.shape))]
    if l == 'p': return [(lambda n, a=a: n[..., a], lambda n, a=a: np.broadcast_to(np.eye(3)[a], n.shape)) for a in range(3)]
    def G2(n, a, b):                                     # d(n_a n_b)/dn
        g = np.zeros(n.shape); g[..., a] += n[..., b]; g[..., b] += n[..., a]; return g
    return [(lambda n: n[..., 0] * n[..., 1], lambda n: G2(n, 0, 1)),
            (lambda n: n[..., 0] * n[..., 2], lambda n: G2(n, 0, 2)),
            (lambda n: n[..., 1] * n[..., 2], lambda n: G2(n, 1, 2)),
            (lambda n: n[..., 0]**2 - n[..., 1]**2, lambda n: G2(n, 0, 0) - G2(n, 1, 1)),
            (lambda n: 3 * n[..., 2]**2 - 1, lambda n: 3 * G2(n, 2, 2))]


def radii(l, n):
    return np.array([R1[l]]) if n == 1 else np.geomspace(RHI[l], RLO, n)


def parse_spec(spec):
    """'A:s3p[7,3.5]d1+B:s2@dz0.5' -> dict(groups=[('A',{'s':[R..],'p':[R..]}),('B',{...})], dz).
    <l><n> = n radii from radii(); <l>[r1,r2,...] = explicit radii [Ang]."""
    dz = 0.0
    if '@dz' in spec: spec, d_ = spec.split('@dz'); dz = float(d_)
    groups = []
    for g in spec.split('+'):
        kind, ch = g.split(':'); cnt = {}; i = 0
        while i < len(ch):
            l = ch[i]
            if ch[i + 1] == '[':
                j = ch.index(']', i); cnt.setdefault(l, []).extend(float(x) for x in ch[i + 2:j].split(',')); i = j + 1
            else:
                j = i + 1
                while j < len(ch) and ch[j].isdigit(): j += 1
                cnt.setdefault(l, []).extend(radii(l, int(ch[i + 1:j]))); i = j
        groups.append((kind, cnt))
    return dict(groups=groups, dz=dz)


def spec_radials(spec):
    """Human-readable radial ladder per group/channel, for plot/table labels."""
    return {k: {l: [f'{R:g}' for R in Rs] for l, Rs in cnt.items()} for grp in parse_spec(spec)['groups'] for k, cnt in [grp]}


def build_terms(apos, spec):
    """Expand spec (string or parse_spec dict) into [(centers (nc,3), [(R,l,angidx),...]), ...]."""
    from spammm.surfaces.PICCore import core_centers_atoms_bonds
    spec = parse_spec(spec) if isinstance(spec, str) else spec
    c_all, is_b = core_centers_atoms_bonds(apos, dbond=1.6)
    sets = []
    for kind, cnt in spec['groups']:
        C = (apos + np.array([0, 0, spec['dz']])) if kind == 'A' else c_all[is_b]
        modes = [(R, l, k) for l, Rs in cnt.items() for R in Rs for k in range(len(ang_funcs(l)))]
        sets.append((C, modes))
    return sets


def design(p, sets, grad=False):
    """E-design (np, ncoef) and optionally gradient design (np, ncoef, 3) [dE/dx]."""
    colsE, colsG = [], []
    for C, modes in sets:
        v = p[:, None, :] - C[None]; r = np.linalg.norm(v, axis=-1); n = v / np.maximum(r, 1e-9)[..., None]
        cache = {}
        for R, l, k in modes:
            if R not in cache:
                u = np.clip(1 - r / R, 0, None); u7 = u**7; cache[R] = (u7 * u, -(POLY_N / R) * u7)
            ph, dph = cache[R]; g, G = ang_funcs(l)[k]; gv = g(n)
            colsE.append(ph * gv)                                         # (np, nc)
            if grad:
                Gv = G(n); Gt = Gv - np.sum(Gv * n, -1)[..., None] * n     # tangential part
                colsG.append(dph[..., None] * gv[..., None] * n + ph[..., None] * Gt / np.maximum(r, 1e-9)[..., None])
    ME = np.concatenate(colsE, axis=1)
    return (ME, np.concatenate(colsG, axis=1)) if grad else ME


def cost_per_query(sets):
    """~flops per query: per center 15 (dist,1/r) + 8/radial + 6/term (E+grad)."""
    return sum(len(C) * (15 + 8 * len({m[0] for m in modes}) + 6 * len(modes)) for C, modes in sets)


def fit_core_lsq(Pf, Ef, Ff, sets, wF=0.0, rcond=1e-12):
    """Weighted LSQ: E rows, plus wF*(dE/dx) rows against -wF*F when wF>0.
    Returns (coef, ME) — ME is the energy-only design on Pf for diagnostics."""
    if wF > 0:
        ME, MG = design(Pf, sets, grad=True)
        M = np.vstack([ME] + [wF * MG[:, :, a] for a in range(3)])
        y = np.concatenate([Ef] + [-wF * Ff[:, a] for a in range(3)])
    else:
        ME = design(Pf, sets); M = ME; y = Ef
    c, *_ = np.linalg.lstsq(M, y, rcond=rcond)
    return c, ME


def eval_terms(P, sets, coef, chunk=8000):
    """Evaluate core E and Fz at points P. Returns (E (np,), Fz (np,))."""
    E = np.empty(len(P)); Fz = np.empty(len(P))
    for i in range(0, len(P), chunk):
        me, mg = design(P[i:i + chunk], sets, grad=True)
        E[i:i + chunk] = me @ coef; Fz[i:i + chunk] = -(mg[:, :, 2] @ coef)
    return E, Fz


def core_residual_study(apos, E3, F3, lo, dg, spec, variants, z0=0.5, sigma=1.0,
                        hs=(2.5, 3.0, 3.5, 4.0), bands=((2.5, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 7.0)), rin=2.5):
    """CORE-ONLY residual (oracle - core) for a list of fit variants — no mesh.

    variants: list of dicts {label, wF=0.0, rcut=4.0, rin=<rin>}.
    Returns (res, geo):
      res[label] = dict(RE, RF 3D residual fields, eb band rms [meV], fz slice rms
        [meV/A], hpe/hpf high-pass rms, ncoef, cond, maxc, rcut, rin)
      geo = dict(P3, apos, iz_h, mol_z, ns, rm3, w3, hs, bands, spec) — w3 is the
        display mask rm>min(rin over variants) & z>mol_z+z0.
    """
    from scipy.ndimage import gaussian_filter
    ns = E3.shape
    P3 = np.stack(np.meshgrid(*[lo[a] + np.arange(ns[a]) * dg for a in range(3)], indexing='ij'), -1)
    P = P3.reshape(-1, 3); E = E3.ravel(); F = F3.reshape(-1, 3); mol_z = apos[:, 2].max()
    from spammm.surfaces.PICCore import min_dist_to_atoms
    rm = min_dist_to_atoms(P, apos); rm3 = rm.reshape(ns)
    up = P[:, 2] > mol_z + z0
    rin_min = min(v.get('rin', rin) for v in variants)
    w3 = ((rm > rin_min) & up).reshape(ns).astype(float)                     # display mask
    Gw = gaussian_filter(w3, sigma / dg, mode='constant')
    iz_h = {h: int(round((mol_z + h - lo[2]) / dg)) for h in hs}
    sets = build_terms(apos, spec)
    res = {}
    for v in variants:
        rin_v = v.get('rin', rin); rcut = v['rcut']; wF = v.get('wF', 0.0); lab = v['label']
        mfit = (rm > rin_v) & (rm < rcut) & up
        c, ME = fit_core_lsq(P[mfit], E[mfit], F[mfit], sets, wF=wF)
        Ec, Fzc = eval_terms(P, sets, c)
        RE = (E - Ec).reshape(ns); RF = (F[:, 2] - Fzc).reshape(ns)
        wv = ((rm > rin_v) & up).reshape(ns).astype(float)                    # this variant's domain (float: gaussian_filter truncates bools)
        Gwv = np.maximum(gaussian_filter(wv, sigma / dg, mode='constant'), 1e-6)
        HPE = (RE - gaussian_filter(wv * RE, sigma / dg, mode='constant') / Gwv) * wv
        HPF = (RF - gaussian_filter(wv * RF, sigma / dg, mode='constant') / Gwv) * wv
        m_ok = wv > 0
        eb = [1e3 * np.sqrt(np.mean(RE[m_ok & (rm3 >= a) & (rm3 < b)]**2)) for a, b in bands]
        fz = [1e3 * np.sqrt(np.mean(RF[..., iz][m_ok[..., iz]]**2)) for iz in iz_h.values()]
        res[lab] = dict(RE=RE, RF=RF, eb=eb, fz=fz, hpe=1e3 * np.sqrt(np.mean(HPE[m_ok]**2)), hpf=1e3 * np.sqrt(np.mean(HPF[m_ok]**2)),
                        ncoef=len(c), cond=np.linalg.cond(ME), maxc=np.abs(c).max(), rcut=rcut, rin=rin_v, wF=wF)
    geo = dict(P3=P3, apos=apos, iz_h=iz_h, mol_z=mol_z, ns=ns, rm3=rm3, w3=w3, hs=list(hs), bands=bands, spec=spec)
    return res, geo
