"""
AssemblyPlot.py — Assembly visualization and per-atom contact diagnostics.

Essence: Matplotlib top views for SAM configs — height shading, clash/strain/clearance maps,
XYZ export with lvec and optional clash column.

Design: Pure plotting; no OpenCL. analyze_assembly_contacts() is O(N²) CPU post-process
on emitted supercells — use plot_best_k sparingly for large exports.

Open issues / caveats:
  - Diagnostic contact analysis duplicates kernel overlap logic on CPU (parity intent, not bit-exact).
  - Large supercells (~5k atoms) take ~1 min/rank for contact maps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors


def analyze_assembly_contacts(atoms4, natoms):
    """Per-atom clash overlap (Å² sum) and contact clearance (Å) vs other molecules.

    clash[i]     — sum_j (r_i+r_j-d_ij)² over inter-molecular overlaps (kernel-consistent)
    clearance[i] — min_j (d_ij - r_i - r_j); negative = buried in overlap
    strain[i]    — max(0, -clearance[i]) overlap depth (Å)
    """
    atoms4 = np.asarray(atoms4, dtype=np.float64)
    pos, rad = atoms4[:, :3], atoms4[:, 3]
    n = len(pos)
    mol_id = np.arange(n, dtype=np.int32) // natoms
    clash = np.zeros(n, dtype=np.float64)
    clearance = np.full(n, np.inf, dtype=np.float64)
    for ia in range(n):
        d = np.linalg.norm(pos - pos[ia], axis=1)
        cross = (mol_id != mol_id[ia]) & (np.arange(n) != ia)
        if not np.any(cross):
            continue
        clear = d - (rad[ia] + rad)
        clearance[ia] = min(clearance[ia], float(np.min(clear[cross])))
        for j in np.where(cross)[0]:
            c = clear[j]
            if c < clearance[j]:
                clearance[j] = c
            if c < 0:
                ov2 = c * c
                clash[ia] += ov2
                clash[j] += ov2
    clearance[np.isinf(clearance)] = 0.0
    strain = np.maximum(0.0, -clearance)
    return clash, clearance, strain


def write_assembly_xyz(path, apos, enames, natoms, cell_lvs, comment, scalars=None):
    """Write assembly xyz; optional 4th column = per-atom scalar (e.g. clash)."""
    with open(path, 'w') as f:
        f.write(f'{len(apos)}\n')
        lvs = 'lvs: ' + ' '.join(f'{v:.6f}' for row in cell_lvs for v in row)
        f.write(f'{comment}  {lvs}\n')
        for i, p in enumerate(apos):
            elem = enames[i % natoms]
            elem = elem.split('_')[0] if '_' in elem else elem
            if scalars is not None:
                f.write(f'{elem} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {scalars[i]:.6f}\n')
            else:
                f.write(f'{elem} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')


def plot_assembly_scalar(ax, apos, values, bonds=None, cell_lvs=None, n_pbc_super=1, cmap_name='hot', vmin=None, vmax=None, bond_lw=0.5, title='', draw_supercell=True):
    """Top view colored by per-atom scalar (clash, strain, clearance, …)."""
    apos = np.asarray(apos, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    vmin = float(values.min()) if vmin is None else vmin
    vmax = float(values.max()) if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-9
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')

    if bonds:
        segs = [[apos[i, :2], apos[j, :2]] for i, j in bonds]
        lc = LineCollection(segs, colors=[(0.3, 0.3, 0.3, 0.35)], linewidths=bond_lw, capstyle='round', zorder=1)
        ax.add_collection(lc)

    order = np.argsort(values)
    for j in order:
        rgba = cmap(norm(values[j]))
        ax.scatter(apos[j, 0], apos[j, 1], s=14, c=[rgba], edgecolors='none', zorder=5 + norm(values[j]))

    if cell_lvs is not None:
        a, b = cell_lvs[0, :2], cell_lvs[1, :2]
        cpts = np.array([[0, 0], a, a + b, b, [0, 0]])
        ax.plot(cpts[:, 0], cpts[:, 1], color='0.15', ls='-', lw=1.2, zorder=20)
        if draw_supercell and n_pbc_super >= 1:
            pts = np.array([[-a[0] - b[0], -a[1] - b[1]], [2 * a[0] - b[0], 2 * a[1] - b[1]], [2 * a[0] + 2 * b[0], 2 * a[1] + 2 * b[1]], [-a[0] + 2 * b[0], -a[1] + 2 * b[1]], [-a[0] - b[0], -a[1] - b[1]]])
            ax.plot(pts[:, 0], pts[:, 1], color='0.45', ls='--', lw=1.0, zorder=19)

    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    return ax


def plot_assembly_diagnostics(apos, atoms4, natoms, bonds, cell_lvs, out_prefix, n_pbc_super=1, rank=0):
    """Write clash + strain + clearance diagnostic PNGs for one configuration."""
    clash, clearance, strain = analyze_assembly_contacts(atoms4, natoms)
    panels = [
        ('clash', clash, 'hot', f'rank {rank+1}  clash overlap (Å²)  max={clash.max():.3f}'),
        ('strain', strain, 'Reds', f'rank {rank+1}  overlap depth (Å)  max={strain.max():.3f}'),
        ('clearance', clearance, 'viridis_r', f'rank {rank+1}  min clearance (Å)  min={clearance.min():.3f}'),
    ]
    paths = []
    for tag, vals, cmap, title in panels:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        if tag == 'clearance':
            plot_assembly_scalar(ax, apos, vals, bonds=bonds, cell_lvs=cell_lvs, n_pbc_super=n_pbc_super, cmap_name=cmap, vmin=clearance.min(), vmax=max(0.5, float(np.percentile(clearance, 95))), title=title)
        else:
            vmax = max(float(vals.max()), 1e-6)
            plot_assembly_scalar(ax, apos, vals, bonds=bonds, cell_lvs=cell_lvs, n_pbc_super=n_pbc_super, cmap_name=cmap, vmin=0.0, vmax=vmax, title=title)
        path = f'{out_prefix}_{tag}.png'
        fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        paths.append(path)
    return clash, clearance, strain, paths


def _z_norm(z):
    zmin, zmax = float(z.min()), float(z.max())
    t = (z - zmin) / max(zmax - zmin, 1e-9)
    return t, zmin, zmax


def replicate_bonds(base_bonds, natoms, n_replicas):
    """Replicate base (i,j) bonds across n_replicas molecules."""
    if base_bonds is None or len(base_bonds) == 0:
        return []
    base_bonds = np.asarray(base_bonds, dtype=np.int32)
    links = []
    for k in range(n_replicas):
        off = k * natoms
        for i, j in base_bonds:
            links.append((int(i) + off, int(j) + off))
    return links


def plot_assembly_height(ax, apos, bonds=None, cell_lvs=None, n_pbc_super=1, cmap_name='viridis', atom_size=12.0, bond_lw=0.5, title='', highlight_dz=None, draw_supercell=True):
    """Top-view (xy) assembly: skeleton sticks + height-shaded atoms on white background."""
    apos = np.asarray(apos, dtype=np.float64)
    z = apos[:, 2]
    t, zmin, zmax = _z_norm(z)
    cmap = plt.get_cmap(cmap_name)

    ax.set_facecolor('white')
    ax.figure.patch.set_facecolor('white')

    if bonds:
        segs, seg_alphas = [], []
        for i, j in bonds:
            p1, p2 = apos[i, :2], apos[j, :2]
            zm = 0.5 * (z[i] + z[j])
            tn = (zm - zmin) / max(zmax - zmin, 1e-9)
            segs.append([p1, p2])
            seg_alphas.append(0.06 + 0.40 * tn)
        lc = LineCollection(segs, colors=[(0.25, 0.25, 0.25, a) for a in seg_alphas], linewidths=bond_lw, capstyle='round', zorder=1)
        ax.add_collection(lc)

    order = np.argsort(z)
    for j in order:
        tn = float(t[j])
        rgba = cmap(tn)
        alpha = 0.12 + 0.88 * tn
        s = atom_size
        if highlight_dz is not None and z[j] > (zmax - highlight_dz):
            ax.scatter(apos[j, 0], apos[j, 1], s=s * 1.5, c=[(0.85, 0.12, 0.12, 1.0)], edgecolors='none', zorder=12)
        else:
            ax.scatter(apos[j, 0], apos[j, 1], s=s, c=[(*rgba[:3], alpha)], edgecolors='none', zorder=5 + tn)

    if cell_lvs is not None:
        a, b = cell_lvs[0, :2], cell_lvs[1, :2]
        cpts = np.array([[0, 0], a, a + b, b, [0, 0]])
        ax.plot(cpts[:, 0], cpts[:, 1], color='0.15', ls='-', lw=1.2, zorder=20)
        if draw_supercell and n_pbc_super >= 1:
            pts = np.array([
                [-a[0] - b[0], -a[1] - b[1]],
                [2 * a[0] - b[0], 2 * a[1] - b[1]],
                [2 * a[0] + 2 * b[0], 2 * a[1] + 2 * b[1]],
                [-a[0] + 2 * b[0], -a[1] + 2 * b[1]],
                [-a[0] - b[0], -a[1] - b[1]],
            ])
            ax.plot(pts[:, 0], pts[:, 1], color='0.45', ls='--', lw=1.0, zorder=19)

    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    return ax


def plot_translations(T_conf, cell_lvs, outpath):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    T_unique = np.unique(T_conf, axis=0)
    ax.scatter(T_unique[:, 0], T_unique[:, 1], s=12, c='steelblue', alpha=0.8)
    cpts = np.array([[0, 0], cell_lvs[0, :2], cell_lvs[0, :2] + cell_lvs[1, :2], cell_lvs[1, :2], [0, 0]])
    ax.plot(cpts[:, 0], cpts[:, 1], 'k--', lw=1.5)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.set_title(f'Translation sampling ({len(T_unique)} unique)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor='white')
    plt.close(fig)


def plot_rotations(R_conf, outpath):
    fig = plt.figure(figsize=(14, 4.5), facecolor='white')
    R_u = np.unique(R_conf, axis=0)
    axes_names = ['a (x)', 'b (y)', 'c (z)']
    colors = ['#c0392b', '#27ae60', '#2980b9']
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d', facecolor='white')
        vecs = R_u[:, :, i]
        ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], c=colors[i], s=6, alpha=0.7)
        ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1]); ax.set_zlim([-1.1, 1.1])
        ax.set_title(axes_names[i], fontsize=10)
    fig.suptitle(f'Rotation sampling ({len(R_u)} unique orientations)', fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor='white')
    plt.close(fig)


def plot_pareto(scores, z_spans, min_dists, export_idx, best_idx, clash_max, zspan_max, dist_min, penalty, outpath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='white')
    for ax in (ax1, ax2):
        ax.set_facecolor('white')

    ax1.scatter(scores, z_spans, c='0.35', alpha=0.35, s=8, label='all', zorder=1)
    if len(export_idx):
        ax1.scatter(scores[export_idx], z_spans[export_idx], c='#e74c3c', alpha=0.75, s=18, label='exported', zorder=3)
    ax1.scatter(scores[best_idx], z_spans[best_idx], c='#2ecc71', marker='*', s=160, edgecolors='0.2', linewidths=0.5, label='best', zorder=5)
    ax1.axvline(clash_max, color='0.6', ls='--', lw=1)
    ax1.axhline(zspan_max, color='0.6', ls='--', lw=1)
    ax1.set_xlabel('clash penalty'); ax1.set_ylabel('z-span (Å)'); ax1.set_title('Clash vs z-span')
    ax1.set_xlim(-0.1, min(penalty, max(clash_max * 2, float(np.percentile(scores, 90)))))
    ax1.legend(fontsize=8, framealpha=0.9)

    ax2.scatter(min_dists, z_spans, c='0.35', alpha=0.35, s=8, label='all', zorder=1)
    if len(export_idx):
        ax2.scatter(min_dists[export_idx], z_spans[export_idx], c='#e74c3c', alpha=0.75, s=18, label='exported', zorder=3)
    ax2.scatter(min_dists[best_idx], z_spans[best_idx], c='#2ecc71', marker='*', s=160, edgecolors='0.2', linewidths=0.5, label='best', zorder=5)
    ax2.axvline(dist_min, color='0.6', ls='--', lw=1)
    ax2.axhline(zspan_max, color='0.6', ls='--', lw=1)
    ax2.set_xlabel('min inter-mol distance (Å)'); ax2.set_ylabel('z-span (Å)'); ax2.set_title('Min-dist vs z-span')
    ax2.set_xlim(0, max(float(np.percentile(min_dists, 99)), dist_min * 2) * 1.05)
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, facecolor='white')
    plt.close(fig)
