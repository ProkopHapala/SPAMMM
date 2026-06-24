import numpy as np

def bond_lengths(apos, bonds):
    return [float(np.linalg.norm(apos[i] - apos[j])) for i, j in bonds]

def bond_angle(apos, i, j, k):
    a, b = apos[i] - apos[j], apos[k] - apos[j]
    return float(np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))))

def planarity(apos, indices):
    p = apos[indices]; c = p.mean(axis=0)
    _, s, _ = np.linalg.svd(p - c)
    return float(s[-1])

def distort(apos, amplitude=0.2, seed=42):
    rng = np.random.default_rng(seed)
    return apos + rng.normal(0, amplitude, apos.shape)

def check_geometry(apos, expected_bonds):
    """expected_bonds: { (i,j): (r0, tol_frac) }. Returns dict of bool."""
    results = {}
    for (i, j), (r0, tol) in expected_bonds.items():
        r = float(np.linalg.norm(apos[i] - apos[j]))
        results[f'bond_{i}_{j}'] = abs(r - r0) < r0 * tol
    return results

def assert_geometry(apos, expected_bonds, name=''):
    checks = check_geometry(apos, expected_bonds)
    for k, v in checks.items():
        assert v, f'{name}: {k} failed'

def find_bonds(apos, enames, Rcut=1.8):
    """Find bonds by distance cutoff. Returns list of (i, j) tuples."""
    na = len(apos)
    bonds = []
    for i in range(na):
        for j in range(i+1, na):
            r = np.linalg.norm(apos[i] - apos[j])
            if r < Rcut:
                bonds.append((i, j))
    return bonds

def save_xyz_frames(filename, enames, positions_list, comments=None):
    """Save multiple frames to a single XYZ file (trajectory).
    
    Args:
        filename: output .xyz path
        enames: list of element names
        positions_list: list of (natoms, 3) arrays
        comments: list of comment strings (one per frame), or None
    """
    na = len(enames)
    with open(filename, 'w') as f:
        for idx, apos in enumerate(positions_list):
            comment = comments[idx] if comments else f'frame {idx}'
            f.write(f'{na}\n{comment}\n')
            for e, p in zip(enames, apos):
                f.write(f'{e:2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')

def plot_geometry(ax, apos, enames, bonds, title='', proj='xy'):
    """Plot molecular geometry on a matplotlib Axes with bond lengths annotated.
    
    Args:
        ax: matplotlib Axes
        apos: (natoms, 3) positions
        enames: list of element names
        bonds: list of (i, j) tuples
        title: plot title
        proj: which plane to project ('xy', 'xz', 'yz')
    """
    axis_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    ax_i, ax_j = axis_map.get(proj, (0, 1))
    colors = {'H': 'lightgray', 'C': 'black', 'O': 'red', 'N': 'blue', 'S': 'yellow', 'Cl': 'green'}
    for i, j in bonds:
        r = np.linalg.norm(apos[i] - apos[j])
        mx, my = (apos[i, ax_i] + apos[j, ax_i]) / 2, (apos[i, ax_j] + apos[j, ax_j]) / 2
        ax.plot([apos[i, ax_i], apos[j, ax_i]], [apos[i, ax_j], apos[j, ax_j]], 'k-', linewidth=2, zorder=1)
        ax.text(mx, my, f'{r:.3f}', fontsize=7, color='blue', ha='center', va='center', zorder=3)
    for ia, (e, p) in enumerate(zip(enames, apos)):
        c = colors.get(e, 'purple')
        ax.scatter(p[ax_i], p[ax_j], c=c, s=200, zorder=2, edgecolors='black')
        ax.text(p[ax_i], p[ax_j], f'{ia}:{e}', fontsize=8, ha='center', va='center', zorder=4)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(f'{proj[0]} [A]')
    ax.set_ylabel(f'{proj[1]} [A]')
