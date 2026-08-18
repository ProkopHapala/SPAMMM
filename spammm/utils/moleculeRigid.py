"""
moleculeRigid.py — Extract molecular rigid bodies (position + rotation) from atomic geometry.

Identifies molecules in a periodic unit cell by:
  1. PBC-aware bond recovery (vdW radii, minimum-image convention)
  2. Connected components on the bond graph
  3. Per-molecule unwrapping (make each molecule contiguous across cell boundaries)
  4. PCA on each molecule → center-of-geometry + rotation matrix (principal axes)

Reuses: atomicUtils.findBondsNP / rotMatPCA / findCOG, elements.ELEMENT_DICT,
plotUtils.ELEM_COLOR_2D.

Plotting: plot_rigid_bodies() draws atoms (colored by molecule), bonds, unit-cell
outline, COM markers, and principal-axis frames — a visualization type not covered
by existing plotUtils functions (plotGeometry has no per-molecule coloring or rigid
body frames).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from spammm import elements, atomicUtils as au
from spammm.plotUtils import ELEM_COLOR_2D

# ── PBC-aware bond finding ───────────────────────────────────────────────────

def _wrap_frac(frac):
    """Wrap fractional coordinates to [0,1)."""
    return frac - np.floor(frac)

def findBondsPBC(apos, atypes, lvec, RvdwCut=1.2):
    """PBC-aware bond finding via minimum-image convention.
    Returns (bonds (n,2) int32, rs (n,) float). Uses vdW radii * RvdwCut as cutoff."""
    inv_lvec = np.linalg.inv(lvec)
    frac = _wrap_frac(apos @ inv_lvec)
    apos_w = frac @ lvec
    RvdWs = au.getAtomRadiusNP(atypes, eparams=elements.ELEMENTS)
    natoms = len(apos)
    bonds = []; rs = []
    for i in range(natoms):
        d = apos_w[i+1:] - apos_w[i][None,:]
        d_frac = d @ inv_lvec
        d_frac = d_frac - np.round(d_frac)      # minimum image
        d = d_frac @ lvec
        r = np.sqrt(np.sum(d*d, axis=1))
        cut = (RvdWs[i+1:] + RvdWs[i]) * RvdwCut
        mask = r < cut
        for j in np.where(mask)[0]:
            jj = i + 1 + int(j)
            bonds.append((i, jj)); rs.append(r[j])
    return np.array(bonds, dtype=np.int32), np.array(rs)

# ── Connected components ─────────────────────────────────────────────────────

def connected_components(natoms, bonds):
    """BFS connected components on bond graph. Returns list of index arrays."""
    adj = [set() for _ in range(natoms)]
    for a, b in bonds:
        adj[a].add(b); adj[b].add(a)
    visited = set(); comps = []
    for i in range(natoms):
        if i in visited: continue
        comp = []; stack = [i]; visited.add(i)
        while stack:
            a = stack.pop(); comp.append(a)
            for nb in adj[a]:
                if nb not in visited: visited.add(nb); stack.append(nb)
        comps.append(np.array(comp, dtype=np.int32))
    return comps

# ── Molecule unwrapping ──────────────────────────────────────────────────────

def unwrap_molecule(frac, lvec, indices):
    """Unwrap fractional positions of one molecule so it's contiguous in real space.
    Uses first atom as reference, minimum-image for each subsequent atom."""
    f = frac[indices].copy()
    ref = f[0]
    for i in range(1, len(f)):
        d = f[i] - ref
        d = d - np.round(d)
        f[i] = ref + d
    return f @ lvec

# ── Rigid body extraction ────────────────────────────────────────────────────

def extract_rigid_bodies(apos, enames, lvec, RvdwCut=1.2, min_size=3):
    """Split periodic geometry into molecular rigid bodies.

    Returns list of dicts, one per molecule:
      { 'indices': global atom indices, 'com': (3,) center of geometry,
        'R': (3,3) rotation matrix (rows = principal axes, sorted by eigenvalue desc),
        'apos_local': (n,3) positions in principal frame (centered + rotated),
        'enames': element names for this molecule, 'bonds': (m,2) local bond indices }

    Uses PBC-aware bond finding + connected components + per-molecule unwrapping + PCA.
    Molecules with fewer than min_size atoms are discarded (likely stray atoms).
    """
    atypes = [elements.ELEMENT_DICT[e][0] if e in elements.ELEMENT_DICT else 6 for e in enames]
    bonds, rs = findBondsPBC(apos, atypes, lvec, RvdwCut=RvdwCut)
    print(f"extract_rigid_bodies: {len(bonds)} bonds, R range [{rs.min():.3f}, {rs.max():.3f}]")
    comps = connected_components(len(apos), bonds)
    comps = [c for c in comps if len(c) >= min_size]
    print(f"extract_rigid_bodies: {len(comps)} molecules (min_size={min_size})")

    inv_lvec = np.linalg.inv(lvec)
    frac_all = _wrap_frac(apos @ inv_lvec)

    # build local bond list per molecule
    bond_of = {}
    for bi, (a, b) in enumerate(bonds):
        bond_of.setdefault(a, []).append(bi)
        bond_of.setdefault(b, []).append(bi)

    bodies = []
    for ci, comp in enumerate(comps):
        idx_set = {int(x) for x in comp}
        local_idx = {int(g): li for li, g in enumerate(comp)}
        apos_unwrapped = unwrap_molecule(frac_all, lvec, comp)
        com = au.findCOG(apos_unwrapped)
        R = au.rotMatPCA(apos_unwrapped)          # (3,3), rows = principal axes
        apos_local = (R @ (apos_unwrapped - com).T).T
        enames_local = [enames[i] for i in comp]
        # local bonds
        loc_bonds = []
        for g in comp:
            for bi in bond_of.get(int(g), []):
                a, b = bonds[bi]
                other = b if a == int(g) else a
                if other in idx_set and int(g) < other:
                    loc_bonds.append([local_idx[int(g)], local_idx[other]])
        loc_bonds = np.array(loc_bonds, dtype=np.int32) if loc_bonds else np.zeros((0,2), dtype=np.int32)
        bodies.append({
            'indices': comp, 'com': com, 'R': R,
            'apos_local': apos_local, 'apos_unwrapped': apos_unwrapped,
            'enames': enames_local, 'bonds': loc_bonds,
        })
    return bodies

# ── Plotting ─────────────────────────────────────────────────────────────────

_MOL_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']

def plot_rigid_bodies(bodies, lvec, axes=(0,1), title='Molecular rigid bodies',
                      fname=None, figsize=(10,8), axis_scale=5.0, bLabels=True):
    """Plot molecules as rigid bodies: atoms colored by molecule, bonds, unit-cell
    outline, COM markers, and principal-axis frames.

    axes: which 2 axes to project (default (0,1) = xy top-down view).
    axis_scale: length of principal-axis arrows relative to molecule extent.
    """
    ax1, ax2 = axes
    fig, ax = plt.subplots(figsize=figsize)

    # unit cell outline
    corners = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0]]) @ lvec
    ax.plot(corners[:,ax1], corners[:,ax2], 'k--', lw=1.5, alpha=0.5)

    for mi, body in enumerate(bodies):
        color = _MOL_COLORS[mi % len(_MOL_COLORS)]
        pos = body['apos_unwrapped']
        ens = body['enames']
        com = body['com']
        R = body['R']
        # atoms
        for p, e in zip(pos, ens):
            ec = ELEM_COLOR_2D.get(e, 'magenta')
            size = 30 if e == 'H' else 80
            ax.scatter(p[ax1], p[ax2], c=ec, s=size, edgecolors=color, linewidths=1.0, zorder=10)
        # bonds
        for a, b in body['bonds']:
            ax.plot([pos[a,ax1], pos[b,ax1]], [pos[a,ax2], pos[b,ax2]], '-', color=color, lw=1.0, alpha=0.6, zorder=5)
        # COM marker
        ax.scatter(com[ax1], com[ax2], c=color, s=200, marker='x', linewidths=2.5, zorder=15)
        # principal axes arrows
        extent = np.max(np.linalg.norm(pos - com, axis=1))
        arrow_len = extent * axis_scale * 0.15
        for k, (axis_color, label) in enumerate(zip(['red','green','blue'], ['PC1','PC2','PC3'])):
            d = R[k] * arrow_len
            ax.annotate('', xy=(com[ax1]+d[ax1], com[ax2]+d[ax2]), xytext=(com[ax1], com[ax2]),
                        arrowprops=dict(arrowstyle='->', color=axis_color, lw=2), zorder=14)
            if bLabels:
                ax.text(com[ax1]+d[ax1]*1.1, com[ax2]+d[ax2]*1.1, f'{label}', color=axis_color, fontsize=8, zorder=16)
        if bLabels:
            ax.text(com[ax1], com[ax2]+extent*0.3, f'mol{mi}', color=color, fontsize=9, ha='center', fontweight='bold', zorder=16)

    ax.set_xlabel(f"{['x','y','z'][ax1]} (Å)")
    ax.set_ylabel(f"{['x','y','z'][ax2]} (Å)")
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    if fname:
        plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    return fig

# ── Approximate symmetry analysis ────────────────────────────────────────────

def find_approx_symmetry(coms_frac, tol=0.05):
    """Find approximate crystallographic symmetry operations that map the set of
    molecular COMs (in fractional coords) onto itself.

    Searches all signed permutation matrices (orthogonal integer 3x3) + fractional
    translations. Returns list of (R, t, max_dev, perm_map) sorted by max_dev ascending,
    where perm_map[i] = j means COM i maps to COM j under this operation.
    """
    from itertools import permutations, product
    nmol = len(coms_frac)
    ops = []
    for perm in permutations(range(3)):
        for signs in product([-1, 1], repeat=3):
            R = np.zeros((3, 3), dtype=int)
            for i in range(3): R[i, perm[i]] = signs[i]
            if abs(int(np.linalg.det(R))) != 1: continue
            for i in range(nmol):
                for j in range(nmol):
                    t = coms_frac[j] - R @ coms_frac[i]
                    t = t - np.round(t)
                    mapped = [(R @ c + t) for c in coms_frac]
                    mapped_w = [m - np.round(m) for m in mapped]
                    all_match = True; perm_map = []
                    for m in mapped_w:
                        best = -1; best_d = 1e9
                        for ji, c in enumerate(coms_frac):
                            d = m - c; d = d - np.round(d)
                            dd = np.max(np.abs(d))
                            if dd < best_d: best_d = dd; best = ji
                        if best_d > tol: all_match = False; break
                        perm_map.append(best)
                    if not all_match: continue
                    max_dev = 0.0
                    for k, m in enumerate(mapped_w):
                        d = m - coms_frac[perm_map[k]]
                        d = d - np.round(d)
                        max_dev = max(max_dev, np.max(np.abs(d)))
                    key = (tuple(R.flatten()), tuple(np.round(t, 3)))
                    if key not in [(tuple(o[0].flatten()), tuple(np.round(o[1], 3))) for o in ops]:
                        ops.append((R.copy(), t.copy(), max_dev, perm_map))
    ops.sort(key=lambda x: x[2])
    return ops

def equivalence_classes(ops, nmol):
    """Group molecule indices into equivalence classes from symmetry operations.
    ops: list of (R, t, dev, perm_map). Returns list of sets."""
    classes = [{i} for i in range(nmol)]
    for _, _, _, pm in ops:
        for i, j in enumerate(pm):
            if i == j: continue
            # merge class of i and class of j
            ci = next((c for c in classes if i in c), None)
            cj = next((c for c in classes if j in c), None)
            if ci is not None and cj is not None and ci is not cj:
                ci |= cj; classes.remove(cj)
    return classes

# ── Molecule neighborhood plotting ───────────────────────────────────────────

def plot_molecule_neighborhood(bodies, lvec, center_idx, radius=17.0, axes=(0,1),
                               title=None, fname=None, figsize=(10,8), bDistLabels=True):
    """Plot the local neighborhood of one molecule: central molecule + all other
    molecules (including periodic images) within `radius` Å of the central COM.

    Central molecule drawn with full opacity + principal axes; neighbors drawn faded.
    Draws COM-to-COM vectors (center-to-center) from central molecule to each neighbor,
    with distance labels.
    """
    ax1, ax2 = axes
    fig, ax = plt.subplots(figsize=figsize)
    com_center = bodies[center_idx]['com']

    # gather all molecule positions including periodic images
    shifts = [(ix, iy, iz) for ix in (-1,0,1) for iy in (-1,0,1) for iz in (-1,0,1)]
    neighbors = []  # (mi, shift, com_shifted, dist)
    for mi, body in enumerate(bodies):
        for (ix, iy, iz) in shifts:
            shift = lvec[0]*ix + lvec[1]*iy + lvec[2]*iz
            com_s = body['com'] + shift
            d = np.linalg.norm(com_s - com_center)
            if d < radius:
                neighbors.append((mi, shift, com_s, d))
    neighbors.sort(key=lambda x: x[3])

    # draw COM-to-COM vectors first (behind everything)
    for mi, shift, com_s, d in neighbors:
        if mi == center_idx and np.allclose(shift, 0): continue
        ax.annotate('', xy=(com_s[ax1], com_s[ax2]), xytext=(com_center[ax1], com_center[ax2]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, alpha=0.6), zorder=2)
        if bDistLabels:
            mid = 0.5*(com_center + com_s)
            ax.text(mid[ax1], mid[ax2], f'{d:.1f}', color='gray', fontsize=7, ha='center', va='bottom', zorder=3)

    # draw neighbors (faded)
    for mi, shift, com_s, d in neighbors:
        if mi == center_idx and np.allclose(shift, 0): continue  # skip center, draw later
        body = bodies[mi]
        pos = body['apos_unwrapped'] + shift
        ens = body['enames']
        alpha = max(0.15, 1.0 - d/radius * 0.7)
        mol_color = _MOL_COLORS[mi % len(_MOL_COLORS)]
        for p, e in zip(pos, ens):
            ec = ELEM_COLOR_2D.get(e, 'magenta')
            size = 20 if e == 'H' else 50
            ax.scatter(p[ax1], p[ax2], c=ec, s=size, edgecolors=mol_color, linewidths=0.5, alpha=alpha, zorder=5)
        for a, b in body['bonds']:
            ax.plot([pos[a,ax1], pos[b,ax1]], [pos[a,ax2], pos[b,ax2]], '-', color=mol_color, lw=0.7, alpha=alpha*0.6, zorder=3)
        ax.scatter(com_s[ax1], com_s[ax2], c=mol_color, s=60, marker='x', linewidths=1.0, alpha=alpha, zorder=6)
        ax.text(com_s[ax1], com_s[ax2]+0.5, f'mol{mi}', color=mol_color, fontsize=7, ha='center', alpha=alpha, zorder=7)

    # draw central molecule
    body = bodies[center_idx]
    pos = body['apos_unwrapped']
    ens = body['enames']
    com = body['com']
    R = body['R']
    cc = _MOL_COLORS[center_idx % len(_MOL_COLORS)]
    for p, e in zip(pos, ens):
        ec = ELEM_COLOR_2D.get(e, 'magenta')
        size = 30 if e == 'H' else 90
        ax.scatter(p[ax1], p[ax2], c=ec, s=size, edgecolors=cc, linewidths=1.5, zorder=15)
    for a, b in body['bonds']:
        ax.plot([pos[a,ax1], pos[b,ax1]], [pos[a,ax2], pos[b,ax2]], '-', color=cc, lw=1.5, zorder=10)
    # principal axes
    extent = np.max(np.linalg.norm(pos - com, axis=1))
    arrow_len = extent * 0.4
    for k, (ac, lab) in enumerate(zip(['red','green','blue'], ['PC1','PC2','PC3'])):
        d = R[k] * arrow_len
        ax.annotate('', xy=(com[ax1]+d[ax1], com[ax2]+d[ax2]), xytext=(com[ax1], com[ax2]),
                    arrowprops=dict(arrowstyle='->', color=ac, lw=2.5), zorder=14)
    ax.scatter(com[ax1], com[ax2], c=cc, s=250, marker='*', linewidths=1.5, edgecolors='black', zorder=20)

    # draw radius circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(com[ax1] + radius*np.cos(theta), com[ax2] + radius*np.sin(theta), 'k--', lw=0.5, alpha=0.3, zorder=1)

    n_neigh = len([n for n in neighbors if not (n[0]==center_idx and np.allclose(n[1],0))])
    ax.set_xlabel(f"{['x','y','z'][ax1]} (Å)")
    ax.set_ylabel(f"{['x','y','z'][ax2]} (Å)")
    ax.set_aspect('equal', adjustable='box')
    if title: ax.set_title(f'{title}  ({n_neigh} neighbors)')
    else:     ax.set_title(f'mol{center_idx} neighborhood ({n_neigh} neighbors)')
    ax.grid(True, alpha=0.2)
    if fname:
        plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close()
    return fig
