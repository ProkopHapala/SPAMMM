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
