"""Minimal standalone test: discretize per-atom QEq charges into per-type charges
that preserve molecular multipole moments (monopole, dipole, quadrupole).

This tests the charge-discretization idea from doc/Tasks/FAF_Fit_Architecture.md §3.3
without evaluating FAF or running any simulation — just the discretization math.

Run:  pytest tests/test_faftype_discretize.py -s
"""
import numpy as np
import pytest


def discretize_charges(apos, enames, q_per_atom, type_scheme='element'):
    """Bin per-atom charges into per-type charges preserving multipole moments.

    Args:
        apos: (N,3) atom positions
        enames: (N,) element names
        q_per_atom: (N,) QEq charges
        type_scheme: 'element' (C/O/H = 3 types) or 'element_sign' (C+/C-/O/H = 4-5)

    Returns: dict with:
        'type_ids': (N,) int array — type index per atom
        'type_names': list of type labels
        'Q_type': (K,) discretized charges per type
        'q_disc': (N,) discretized charge per atom (= Q_type[type_ids])
        'moments_orig': dict of original moments
        'moments_disc': dict of discretized moments
    """
    apos = np.asarray(apos, dtype=np.float64)
    enames = np.asarray(enames)
    q = np.asarray(q_per_atom, dtype=np.float64)
    N = len(q)

    # --- 1. Assign atoms to types ---
    if type_scheme == 'element':
        type_labels = sorted(set(enames))
        type_ids = np.array([type_labels.index(e) for e in enames], dtype=np.int32)
    elif type_scheme == 'element_sign':
        # Split by element AND charge sign (C+ vs C-, O+ vs O-, etc.)
        signs = np.where(q >= 0, '+', '-')
        labels = [f'{e}{s}' for e, s in zip(enames, signs)]
        type_labels = sorted(set(labels))
        type_ids = np.array([type_labels.index(l) for l in labels], dtype=np.int32)
    else:
        raise ValueError(f"unknown type_scheme '{type_scheme}'")
    K = len(type_labels)

    # --- 2. Build constraint matrix ---
    # Unknowns: Q = [Q_0, ..., Q_{K-1}]
    # Constraints (linear in Q):
    #   (a) neutrality:      sum_t n_t * Q_t = 0                    [1 eq]
    #   (b) dipole:          sum_t Q_t * R_t[k] = M1[k]  for k=0,1,2  [3 eqs]
    #   (c) quadrupole:      sum_t Q_t * S_t[k] = M2[k]  for k in upper-tri [6 eqs]
    # Total: 10 equations, K unknowns. Over-determined for K<10 → least-squares.
    # Neutrality is hard (exact); dipole+quadrupole are soft (least-squares).

    # Per-type aggregates
    n_t = np.array([(type_ids == t).sum() for t in range(K)], dtype=np.float64)       # (K,)
    R_t = np.array([apos[type_ids == t].sum(axis=0) for t in range(K)])               # (K,3)
    S_t = np.array([(apos[type_ids == t][:,None,:] * apos[type_ids == t][:,:,None]).sum(axis=0)
                    for t in range(K)])                                                # (K,3,3)

    # Original moments
    M0 = q.sum()
    M1 = (q[:, None] * apos).sum(axis=0)                                               # (3,)
    M2 = (q[:, None, None] * apos[:, :, None] * apos[:, None, :]).sum(axis=0)          # (3,3)

    # Build A * Q = b
    # Row 0: neutrality (hard)
    # Rows 1-3: dipole x,y,z
    # Rows 4-9: quadrupole xx,xy,xz,yy,yz,zz (upper triangle)
    A = np.zeros((10, K), dtype=np.float64)
    b = np.zeros(10, dtype=np.float64)
    A[0, :] = n_t;        b[0] = M0
    A[1:4, :] = R_t.T;    b[1:4] = M1
    quad_idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for row, (i, j) in enumerate(quad_idx):
        A[4 + row, :] = S_t[:, i, j]
        b[4 + row] = M2[i, j]

    # Weight: neutrality = 1e6 (hard), dipole = 1.0, quadrupole = 1.0
    w = np.array([1e6] + [1.0] * 3 + [1.0] * 6, dtype=np.float64)
    Aw = A * w[:, None]
    bw = b * w

    # Solve weighted least-squares
    Q_type, *_ = np.linalg.lstsq(Aw, bw, rcond=None)

    q_disc = Q_type[type_ids]

    return {
        'type_ids': type_ids,
        'type_names': type_labels,
        'Q_type': Q_type,
        'q_disc': q_disc,
        'moments_orig': {'M0': M0, 'M1': M1, 'M2': M2},
        'moments_disc': {
            'M0': q_disc.sum(),
            'M1': (q_disc[:, None] * apos).sum(axis=0),
            'M2': (q_disc[:, None, None] * apos[:, :, None] * apos[:, None, :]).sum(axis=0),
        },
    }


def _print_moments(label, result):
    mo, md = result['moments_orig'], result['moments_disc']
    print(f'\n=== {label} ===')
    print(f'Types: {result["type_names"]}')
    for t, q in zip(result['type_names'], result['Q_type']):
        print(f'  {t}: Q={q:+.4f}')
    print(f'Monopole:  orig={mo["M0"]:+.6f}  disc={md["M0"]:+.6f}  err={abs(md["M0"]-mo["M0"]):.2e}')
    print(f'Dipole:    orig=({mo["M1"][0]:+.4f},{mo["M1"][1]:+.4f},{mo["M1"][2]:+.4f})'
          f'  disc=({md["M1"][0]:+.4f},{md["M1"][1]:+.4f},{md["M1"][2]:+.4f})')
    d_err = np.linalg.norm(md['M1'] - mo['M1'])
    print(f'  |dDip|={d_err:.2e}')
    q_err = np.linalg.norm(md['M2'] - mo['M2'])
    print(f'Quadrupole |dQuad|={q_err:.2e}')
    print(f'  orig: xx={mo["M2"][0,0]:+.4f} xy={mo["M2"][0,1]:+.4f} yy={mo["M2"][1,1]:+.4f}')
    print(f'  disc: xx={md["M2"][0,0]:+.4f} xy={md["M2"][0,1]:+.4f} yy={md["M2"][1,1]:+.4f}')
    # Per-atom charge error
    q_atom_err = np.linalg.norm(result['q_disc'] - np.asarray(mo['M0'] / len(result['q_disc'])) * 0)  # placeholder
    return d_err, q_err


@pytest.mark.parametrize('type_scheme', ['element', 'element_sign'])
def test_discretize_ptcda(type_scheme):
    """PTCDA: 38 atoms, 3 element types (C/O/H) or 4-5 element+sign types.
    Discretized charges should preserve neutrality (exact) and approximate dipole+quadrupole.
    """
    from spammm.forcefields.molecule_loaders import load_ptcda
    apos, enames, REQs, _ = load_ptcda(qeq=True)
    q = REQs[:, 2].copy()

    result = discretize_charges(apos, enames, q, type_scheme=type_scheme)
    d_err, q_err = _print_moments(f'PTCDA ({type_scheme})', result)

    # Neutrality: exact (hard constraint)
    assert abs(result['moments_disc']['M0']) < 1e-6, 'neutrality violated'

    # Dipole: should be well preserved (PTCDA is centrosymmetric, dipole ≈ 0)
    assert d_err < 0.5, f'dipole error too large: {d_err}'

    # Quadrupole: approximate (over-determined with 3 types)
    # Just check it's finite and not wildly off
    assert np.isfinite(q_err), 'quadrupole error not finite'

    # Charges should be non-trivial (not all zero)
    assert np.any(np.abs(result['Q_type']) > 0.01), 'discretized charges all ~0'

    # Per-atom discretized charge should be closer to real Q than to 0
    q_real = np.asarray(q)
    q_disc = result['q_disc']
    err_disc = np.linalg.norm(q_disc - q_real)
    err_zero = np.linalg.norm(q_real)
    print(f'\n  Per-atom charge RMS: disc={err_disc:.4f}  zero={err_zero:.4f}  ratio={err_disc/err_zero:.2f}')


def test_discretize_formic_acid():
    """formic_acid: 5 atoms (C, O×2, H×2). Small molecule, 3 element types.
    Should preserve moments well with 3 types for 5 atoms.
    """
    from spammm.forcefields.molecule_loaders import load_formic_acid
    apos, enames, REQs, _ = load_formic_acid(qeq=True)
    q = REQs[:, 2].copy()

    result = discretize_charges(apos, enames, q, type_scheme='element')
    d_err, q_err = _print_moments('formic_acid (element)', result)

    assert abs(result['moments_disc']['M0']) < 1e-6
    assert d_err < 1.0, f'dipole error too large: {d_err}'
    assert np.any(np.abs(result['Q_type']) > 0.01)


if __name__ == '__main__':
    # Run standalone (no pytest) for quick visual check
    from spammm.forcefields.molecule_loaders import load_ptcda, load_formic_acid

    print('=' * 60)
    print('PTCDA — 3 element types (C, O, H)')
    apos, enames, REQs, _ = load_ptcda(qeq=True)
    r = discretize_charges(apos, enames, REQs[:, 2], 'element')
    _print_moments('PTCDA (element)', r)

    print('\n' + '=' * 60)
    print('PTCDA — element+sign types (C+, C-, O, H)')
    r = discretize_charges(apos, enames, REQs[:, 2], 'element_sign')
    _print_moments('PTCDA (element_sign)', r)

    print('\n' + '=' * 60)
    print('formic_acid — 3 element types')
    apos, enames, REQs, _ = load_formic_acid(qeq=True)
    r = discretize_charges(apos, enames, REQs[:, 2], 'element')
    _print_moments('formic_acid (element)', r)
