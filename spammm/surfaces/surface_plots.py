"""
surface_plots.py — Visualization functions for molecule-on-substrate simulations.

Purpose: Plotting utilities for relaxation trajectories, lateral scans,
manipulation trajectories, and relaxed scans. All functions use matplotlib
with non-interactive backend for headless use.

Key functionality:
  - plot_relaxation() — energy/force/torque vs step
  - plot_molecule_substrate_xy() / _xz() — top/side view of molecule on substrate
  - plot_relax_overview() — comprehensive relaxation visualization
  - plot_force_map() — lateral force map
  - plot_manipulation() — manipulation trajectory snapshots
  - plot_relaxed_scan() — relaxed scan snapshots + force/torque curves
  - plot_manipulation_trail() — pin/opp atom trail visualization

Role in SPAMMM: Visualization layer for FoldedRigid.py workflows.
Separated from core compute to keep modules focused.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .FoldedRigid import nearest_substrate_distance

ELEMENT_COLORS = {'H': 'lightgray', 'C': 'black', 'O': 'red', 'N': 'blue', 'Na': 'goldenrod', 'Cl': 'green', 'S': 'yellow'}
ELEMENT_SIZES = {'H': 50, 'C': 120, 'O': 140, 'N': 130, 'Na': 180, 'Cl': 180, 'S': 150}


def plot_relaxation(traj, save_dir, name='relax'):
    """Plot energy, force, torque vs step."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    steps = np.arange(len(traj['energies']))
    axes[0].plot(steps, traj['energies'], 'b-')
    axes[0].set_ylabel('Energy [eV]')
    axes[0].set_title(f'{name}: Relaxation')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, traj['forces'], 'r-')
    axes[1].set_ylabel('|Force| [eV/Å]')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, traj['torques'], 'g-')
    axes[2].set_ylabel('|Torque| [eV]')
    axes[2].set_xlabel('Step')
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name}_relaxation.png'), dpi=150)
    plt.close(fig)


def plot_molecule_substrate_xy(ax, mol_apos, mol_enames, sub_apos, sub_enames, title='', bonds=None, highlight_element=None, highlight_color='magenta'):
    """Top-view (XY) plot of molecule on substrate.

    Args:
        ax: matplotlib Axes
        mol_apos: (natoms, 3) molecule positions
        mol_enames: list of element names
        sub_apos: (N, 3) or (N, 4) substrate positions
        sub_enames: list of substrate element names
        title: plot title
        bonds: list of (i,j) tuples for molecule bonds
        highlight_element: element name to highlight in molecule (e.g. 'O')
    """
    sub_apos = np.asarray(sub_apos[:, :3], dtype=np.float64)
    for i, (e, p) in enumerate(zip(sub_enames, sub_apos)):
        c = ELEMENT_COLORS.get(e, 'gray')
        s = ELEMENT_SIZES.get(e, 100)
        ax.scatter(p[0], p[1], c=c, s=s, zorder=2, edgecolors='black', linewidths=0.5)
        ax.text(p[0], p[1], e, fontsize=6, ha='center', va='center', zorder=4)
    if bonds:
        for i, j in bonds:
            ax.plot([mol_apos[i, 0], mol_apos[j, 0]], [mol_apos[i, 1], mol_apos[j, 1]], 'k-', linewidth=1.5, zorder=3)
    for i, (e, p) in enumerate(zip(mol_enames, mol_apos)):
        c = ELEMENT_COLORS.get(e, 'purple')
        s = ELEMENT_SIZES.get(e, 100) * 0.7
        edge = highlight_color if (highlight_element and e == highlight_element) else 'black'
        lw = 2.0 if (highlight_element and e == highlight_element) else 0.5
        ax.scatter(p[0], p[1], c=c, s=s, zorder=5, edgecolors=edge, linewidths=lw)
    ax.set_aspect('equal')
    ax.set_xlabel('X [Å]')
    ax.set_ylabel('Y [Å]')
    ax.set_title(title)


def plot_molecule_substrate_xz(ax, mol_apos, mol_enames, sub_apos, sub_enames, title='', bonds=None, highlight_element=None, highlight_color='magenta'):
    """Side-view (XZ) plot of molecule on substrate."""
    sub_apos = np.asarray(sub_apos[:, :3], dtype=np.float64)
    for i, (e, p) in enumerate(zip(sub_enames, sub_apos)):
        c = ELEMENT_COLORS.get(e, 'gray')
        s = ELEMENT_SIZES.get(e, 100)
        ax.scatter(p[0], p[2], c=c, s=s, zorder=2, edgecolors='black', linewidths=0.5)
        ax.text(p[0], p[2], e, fontsize=6, ha='center', va='center', zorder=4)
    if bonds:
        for i, j in bonds:
            ax.plot([mol_apos[i, 0], mol_apos[j, 0]], [mol_apos[i, 2], mol_apos[j, 2]], 'k-', linewidth=1.5, zorder=3)
    for i, (e, p) in enumerate(zip(mol_enames, mol_apos)):
        c = ELEMENT_COLORS.get(e, 'purple')
        s = ELEMENT_SIZES.get(e, 100) * 0.7
        edge = highlight_color if (highlight_element and e == highlight_element) else 'black'
        lw = 2.0 if (highlight_element and e == highlight_element) else 0.5
        ax.scatter(p[0], p[2], c=c, s=s, zorder=5, edgecolors=edge, linewidths=lw)
    ax.set_aspect('equal')
    ax.set_xlabel('X [Å]')
    ax.set_ylabel('Z [Å]')
    ax.set_title(title)


def plot_relax_overview(traj, mol_enames, sub_apos, sub_enames, save_dir, name, bonds=None, highlight_element='O', target_element='Na'):
    """Create comprehensive visualization: relaxation curves + XY/XZ views (initial & final).

    Also annotates distances from highlight_element atoms to nearest target_element substrate atom.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.figure(figsize=(16, 14))

    ax_e = fig.add_subplot(3, 3, 1)
    ax_f = fig.add_subplot(3, 3, 4)
    ax_t = fig.add_subplot(3, 3, 7)
    steps = np.arange(len(traj['energies']))
    ax_e.plot(steps, traj['energies'], 'b-')
    ax_e.set_ylabel('Energy [eV]'); ax_e.set_title(f'{name}: Energy'); ax_e.grid(True, alpha=0.3)
    ax_f.plot(steps, traj['forces'], 'r-')
    ax_f.set_ylabel('|Force| [eV/Å]'); ax_f.set_title('Force'); ax_f.grid(True, alpha=0.3)
    ax_t.plot(steps, traj['torques'], 'g-')
    ax_t.set_ylabel('|Torque| [eV]'); ax_t.set_xlabel('Step'); ax_t.set_title('Torque'); ax_t.grid(True, alpha=0.3)

    ax_xy0 = fig.add_subplot(3, 3, 2)
    ax_xy1 = fig.add_subplot(3, 3, 3)
    mol0 = traj['atom_positions'][0]
    mol1 = traj['atom_positions'][-1]
    plot_molecule_substrate_xy(ax_xy0, mol0, mol_enames, sub_apos, sub_enames, title=f'{name} XY (initial)', bonds=bonds, highlight_element=highlight_element)
    plot_molecule_substrate_xy(ax_xy1, mol1, mol_enames, sub_apos, sub_enames, title=f'{name} XY (final)', bonds=bonds, highlight_element=highlight_element)

    ax_xz0 = fig.add_subplot(3, 3, 5)
    ax_xz1 = fig.add_subplot(3, 3, 6)
    plot_molecule_substrate_xz(ax_xz0, mol0, mol_enames, sub_apos, sub_enames, title=f'{name} XZ (initial)', bonds=bonds, highlight_element=highlight_element)
    plot_molecule_substrate_xz(ax_xz1, mol1, mol_enames, sub_apos, sub_enames, title=f'{name} XZ (final)', bonds=bonds, highlight_element=highlight_element)

    if highlight_element and target_element:
        for label, mol_pos, ax in [('initial', mol0, ax_xy0), ('final', mol1, ax_xy1)]:
            for ia, e in enumerate(mol_enames):
                if e == highlight_element:
                    d, idx = nearest_substrate_distance(mol_pos[ia], sub_apos, sub_enames, target_element)
                    ax.annotate(f'd({e}-{target_element})={d:.2f}Å', xy=(mol_pos[ia, 0], mol_pos[ia, 1]),
                                fontsize=7, color='red', ha='center', va='bottom',
                                arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name}_overview.png'), dpi=150)
    plt.close(fig)

    for proj, plot_fn, suffix in [('xy', plot_molecule_substrate_xy, 'XY'), ('xz', plot_molecule_substrate_xz, 'XZ')]:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        plot_fn(ax2, mol1, mol_enames, sub_apos, sub_enames, title=f'{name} {suffix} (final)', bonds=bonds, highlight_element=highlight_element)
        if highlight_element and target_element:
            for ia, e in enumerate(mol_enames):
                if e == highlight_element:
                    d, idx = nearest_substrate_distance(mol1[ia], sub_apos, sub_enames, target_element)
                    ax2.annotate(f'd({e}-{target_element})={d:.2f}Å', xy=(mol1[ia, 0] if proj == 'xy' else mol1[ia, 0], mol1[ia, 1] if proj == 'xy' else mol1[ia, 2]),
                                 fontsize=8, color='red', ha='center', va='bottom',
                                 arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_dir, f'{name}_{suffix}_final.png'), dpi=150)
        plt.close(fig2)


def plot_force_map(scan_result, save_dir, name='scan'):
    """Plot Fz and |F| lateral force maps."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    X, Y = scan_result['X'], scan_result['Y']
    Fz = scan_result['Fz']
    Fmag = np.sqrt(scan_result['Fx']**2 + scan_result['Fy']**2 + Fz**2)
    E = scan_result['E']

    for ax, data, title, cmap in [
        (axes[0], Fz, 'Fz [eV/Å]', 'RdBu_r'),
        (axes[1], Fmag, '|F| [eV/Å]', 'hot'),
        (axes[2], E, 'Energy [eV]', 'viridis'),
    ]:
        im = ax.pcolormesh(X, Y, data, shading='auto', cmap=cmap)
        ax.set_aspect('equal')
        ax.set_xlabel('X [Å]')
        ax.set_ylabel('Y [Å]')
        ax.set_title(f'{name}: {title}')
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name}_force_map.png'), dpi=150)
    plt.close(fig)


def plot_manipulation(traj, mol_enames, sub_apos, sub_enames, save_dir, name='manip', bonds=None, highlight_element='O', target_element='Na'):
    """Visualize manipulation trajectory with multi-snapshot XY and XZ views."""
    os.makedirs(save_dir, exist_ok=True)
    n_snap = len(traj['atom_positions'])
    n_show = min(6, n_snap)
    indices = np.linspace(0, n_snap - 1, n_show, dtype=int)

    fig_xy, axes_xy = plt.subplots(2, 3, figsize=(18, 12))
    for k, idx in enumerate(indices):
        ax = axes_xy[k // 3][k % 3]
        mol_pos = traj['atom_positions'][idx]
        plot_molecule_substrate_xy(ax, mol_pos, mol_enames, sub_apos, sub_enames,
                                   title=f'{name} XY step {idx}', bonds=bonds,
                                   highlight_element=highlight_element)
        if highlight_element and target_element:
            for ia, e in enumerate(mol_enames):
                if e == highlight_element:
                    d, _ = nearest_substrate_distance(mol_pos[ia], sub_apos, sub_enames, target_element)
                    ax.annotate(f'd={d:.2f}', xy=(mol_pos[ia, 0], mol_pos[ia, 1]),
                                fontsize=7, color='red', ha='center', va='bottom')
    fig_xy.tight_layout()
    fig_xy.savefig(os.path.join(save_dir, f'{name}_xy_snapshots.png'), dpi=150)
    plt.close(fig_xy)

    fig_xz, axes_xz = plt.subplots(2, 3, figsize=(18, 12))
    for k, idx in enumerate(indices):
        ax = axes_xz[k // 3][k % 3]
        mol_pos = traj['atom_positions'][idx]
        plot_molecule_substrate_xz(ax, mol_pos, mol_enames, sub_apos, sub_enames,
                                   title=f'{name} XZ step {idx}', bonds=bonds,
                                   highlight_element=highlight_element)
    fig_xz.tight_layout()
    fig_xz.savefig(os.path.join(save_dir, f'{name}_xz_snapshots.png'), dpi=150)
    plt.close(fig_xz)

    fig_f, ax_f = plt.subplots(figsize=(10, 5))
    steps = np.arange(n_snap)
    Fmag = np.linalg.norm(traj['forces'], axis=1)
    ax_f.plot(steps, Fmag, 'b-', label='|F|')
    ax_f.set_xlabel('Step')
    ax_f.set_ylabel('|Force| [eV/Å]')
    ax_f.set_title(f'{name}: Force along manipulation path')
    ax_f.grid(True, alpha=0.3)
    ax_f.legend()
    fig_f.tight_layout()
    fig_f.savefig(os.path.join(save_dir, f'{name}_force_path.png'), dpi=150)
    plt.close(fig_f)


def plot_relaxed_scan(traj, mol_enames, sub_apos, sub_enames, save_dir, name='rscan',
                      bonds=None, pin_atom_idx=None, highlight_element='O', target_element='Na'):
    """Visualize relaxed scan: snapshots, force/torque curves, pin force."""
    os.makedirs(save_dir, exist_ok=True)
    n_rec = len(traj['atom_positions'])
    n_path = traj['n_path']
    rec_per_path = n_rec // n_path

    snap_indices = [min((i + 1) * rec_per_path - 1, n_rec - 1) for i in range(n_path)]
    n_show = min(6, n_path)
    show_indices = np.linspace(0, n_path - 1, n_show, dtype=int)

    fig_xy, axes_xy = plt.subplots(2, 3, figsize=(18, 12))
    for k, pi in enumerate(show_indices):
        ax = axes_xy[k // 3][k % 3]
        si = snap_indices[pi]
        mol_pos = traj['atom_positions'][si]
        path_pt = traj['path'][pi]
        plot_molecule_substrate_xy(ax, mol_pos, mol_enames, sub_apos, sub_enames,
                                   title=f'{name} XY path {pi}', bonds=bonds,
                                   highlight_element=highlight_element)
        if pin_atom_idx is not None:
            ax.plot(path_pt[0], path_pt[1], 'rx', markersize=12, markeredgewidth=2)
            ax.annotate(f'pin', xy=(path_pt[0], path_pt[1]), fontsize=7, color='red', ha='left')
    fig_xy.tight_layout()
    fig_xy.savefig(os.path.join(save_dir, f'{name}_xy_snapshots.png'), dpi=150)
    plt.close(fig_xy)

    fig_xz, axes_xz = plt.subplots(2, 3, figsize=(18, 12))
    for k, pi in enumerate(show_indices):
        ax = axes_xz[k // 3][k % 3]
        si = snap_indices[pi]
        mol_pos = traj['atom_positions'][si]
        path_pt = traj['path'][pi]
        plot_molecule_substrate_xz(ax, mol_pos, mol_enames, sub_apos, sub_enames,
                                   title=f'{name} XZ path {pi}', bonds=bonds,
                                   highlight_element=highlight_element)
        if pin_atom_idx is not None:
            ax.plot(path_pt[0], path_pt[2], 'rx', markersize=12, markeredgewidth=2)
    fig_xz.tight_layout()
    fig_xz.savefig(os.path.join(save_dir, f'{name}_xz_snapshots.png'), dpi=150)
    plt.close(fig_xz)

    fig_f, (ax_f, ax_t, ax_pf) = plt.subplots(3, 1, figsize=(10, 12))
    path_steps = np.arange(n_path)
    Fmag = np.linalg.norm(traj['forces'][snap_indices], axis=1)
    Tmag = np.linalg.norm(traj['torques'][snap_indices], axis=1)
    Pmag = np.linalg.norm(traj['pin_forces'], axis=1)
    ax_f.plot(path_steps, Fmag, 'b-o', label='|F|')
    ax_f.set_ylabel('|Force| [eV/Å]')
    ax_f.set_title(f'{name}: Force/torque along relaxed scan')
    ax_f.grid(True, alpha=0.3)
    ax_f.legend()
    ax_t.plot(path_steps, Tmag, 'r-o', label='|τ|')
    ax_t.set_ylabel('|Torque| [eV]')
    ax_t.grid(True, alpha=0.3)
    ax_t.legend()
    ax_pf.plot(path_steps, Pmag, 'g-o', label='|F_pin|')
    ax_pf.set_ylabel('|Pin force| [eV/Å]')
    ax_pf.set_xlabel('Path step')
    ax_pf.grid(True, alpha=0.3)
    ax_pf.legend()
    fig_f.tight_layout()
    fig_f.savefig(os.path.join(save_dir, f'{name}_forces.png'), dpi=150)
    plt.close(fig_f)


def plot_manipulation_trail(traj, mol_enames, sub_apos, sub_enames, save_dir, name='rscan',
                            pin_atom_idx=None, opp_atom_idx=None,
                            highlight_element='O', target_element='Na'):
    """Plot manipulation trail: pin atom and opposite atom connected by thin lines.

    Instead of full molecule snapshots, shows the trail of two key atoms:
    - Pinned atom (red dots) — the one being dragged
    - Opposite atom (blue dots) — the far end of the molecule
    - Thin alpha-blended line connecting them for each snapshot
    This reveals how the molecule tilts and follows the tip.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_rec = len(traj['atom_positions'])
    n_path = traj['n_path']
    rec_per_path = n_rec // n_path
    snap_indices = [min((i + 1) * rec_per_path - 1, n_rec - 1) for i in range(n_path)]

    pin_pos = np.array([traj['atom_positions'][si][pin_atom_idx] for si in snap_indices])
    opp_pos = np.array([traj['atom_positions'][si][opp_atom_idx] for si in snap_indices])
    path_pts = traj['path']

    colors = plt.cm.viridis(np.linspace(0, 1, n_path))

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 7))

    sub = np.asarray(sub_apos[:, :3], dtype=np.float64)
    sub_z = sub[:, 2]
    z_top = sub_z.mean()
    layer_mask = np.abs(sub_z - z_top) < 0.5
    sub1 = sub[layer_mask]
    sub1_names = [sub_enames[i] for i in range(len(sub_enames)) if layer_mask[i]]

    for ax, proj, xlabel, ylabel, title_proj in [
        (ax_xy, [0, 1], 'x (Å)', 'y (Å)', 'XY'),
        (ax_xz, [0, 2], 'x (Å)', 'z (Å)', 'XZ'),
    ]:
        for e, p in zip(sub1_names, sub1):
            c = 'blue' if e in ['Na', 'K', 'Ca', 'Mg'] else 'green'
            ax.plot(p[proj[0]], p[proj[1]], 'o', color=c, markersize=4, alpha=0.3)

        ax.plot(path_pts[:, proj[0]], path_pts[:, proj[1]], 'k--', alpha=0.3, linewidth=1)

        for i in range(n_path):
            x = [pin_pos[i, proj[0]], opp_pos[i, proj[0]]]
            y = [pin_pos[i, proj[1]], opp_pos[i, proj[1]]]
            ax.plot(x, y, '-', color=colors[i], alpha=0.3, linewidth=0.8)
            ax.plot(pin_pos[i, proj[0]], pin_pos[i, proj[1]], '.', color=colors[i], markersize=3, alpha=0.6)
            ax.plot(opp_pos[i, proj[0]], opp_pos[i, proj[1]], '.', color=colors[i], markersize=3, alpha=0.6)

        ax.plot(pin_pos[0, proj[0]], pin_pos[0, proj[1]], 'r^', markersize=8, label='pin start')
        ax.plot(pin_pos[-1, proj[0]], pin_pos[-1, proj[1]], 'rv', markersize=8, label='pin end')
        ax.plot(opp_pos[0, proj[0]], opp_pos[0, proj[1]], 'b^', markersize=8, label='opp start')
        ax.plot(opp_pos[-1, proj[0]], opp_pos[-1, proj[1]], 'bv', markersize=8, label='opp end')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} {title_proj} trail')
        ax.set_aspect('equal')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'{name}: pin atom {pin_atom_idx} ({mol_enames[pin_atom_idx]}) → '
                 f'opp atom {opp_atom_idx} ({mol_enames[opp_atom_idx]})', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name}_trail.png'), dpi=150)
    plt.close(fig)

    fig_d, (ax_d, ax_tilt) = plt.subplots(2, 1, figsize=(10, 8))
    dists = np.linalg.norm(pin_pos - opp_pos, axis=1)
    dz = opp_pos[:, 2] - pin_pos[:, 2]
    dx = np.linalg.norm(opp_pos[:, :2] - pin_pos[:, :2], axis=1)
    tilts = np.degrees(np.arctan2(dz, dx))
    path_x = path_pts[:, 0]

    ax_d.plot(path_x, dists, 'k-o', markersize=3)
    ax_d.set_ylabel('pin–opp distance (Å)')
    ax_d.set_title(f'{name}: pin–opp distance and tilt along path')
    ax_d.grid(True, alpha=0.3)

    ax_tilt.plot(path_x, tilts, 'r-o', markersize=3)
    ax_tilt.set_ylabel('tilt angle (°)')
    ax_tilt.set_xlabel('pin x position (Å)')
    ax_tilt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_tilt.grid(True, alpha=0.3)

    fig_d.tight_layout()
    fig_d.savefig(os.path.join(save_dir, f'{name}_dist_tilt.png'), dpi=150)
    plt.close(fig_d)
