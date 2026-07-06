"""Rigid DFTB single-point scan for H-bond proton transfer (donor-H...acceptor ↔ donor...H-acceptor).

Essence: build ASCII ``':'`` H-bond systems, slide H along donor→acceptor axis, DFTB+ SP energy profile.

Design: path grid in Å along D-A axis (default ``ds=0.1``); charge restart between steps; artifacts in
``debug/test_hbond_scan/``. Reuses ``save_xyz_movie`` from ``DFTB_utils``.

Open issues:
- Rigid scan only (heavy atoms fixed); SCC may fail near acceptor — use ``on_fail='skip'``.
- ASCII 2D geometry (z=0); Kekule localization not required for DFTB element types.
"""
import os
import numpy as np

HAU2EV = 27.211386245988
DEBUG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'debug', 'test_hbond_scan')
DEFAULT_DS = 0.1


def identify_hbond_from_ascii(atoms, pair_idx=0):
    """Return (h_idx, donor_idx, acceptor_idx) for ASCII ``':'`` H-bond pair *pair_idx*."""
    from spammm.topology.ascii_art_heterocycle import resolve_hbond_pairs
    resolve_hbond_pairs(atoms)
    hb = getattr(atoms, 'hbonds_ascii', None)
    if not hb:
        raise RuntimeError("No ASCII H-bonds (':' marks); call after add_capping_h_sp2")
    ih, acc = hb[pair_idx]
    heavy = [j for j in atoms.ngs[ih] if atoms.enames[j] != 'H']
    if len(heavy) != 1:
        raise RuntimeError(f"H atom {ih} must have exactly one heavy neighbor, got {heavy}")
    return ih, heavy[0], acc


def make_hbond_transfer_path(apos, h_idx, donor_idx, acceptor_idx, ds=DEFAULT_DS, fractions=None, r_xh=1.01):
    """H positions along donor→acceptor axis (rigid proton transfer).

    Grid: uniform step *ds* [Å] along axis from donor-side H (s0) to acceptor-side X-H (s1).
    Pass *fractions* explicitly to override (e.g. coarse pytest grid).
    """
    apos = np.asarray(apos, dtype=float)
    h0 = apos[h_idx].copy()
    pD, pA = apos[donor_idx], apos[acceptor_idx]
    axis = pA - pD
    L = float(np.linalg.norm(axis))
    assert L > 1e-8, "Donor-acceptor distance too small"
    axis_hat = axis / L
    s0 = float(np.dot(h0 - pD, axis_hat))
    s1 = L - r_xh
    if fractions is not None:
        fractions = np.asarray(fractions, dtype=float)
        s_axis = s0 + fractions * (s1 - s0)
    else:
        lo, hi = (s0, s1) if s0 <= s1 else (s1, s0)
        s_axis = np.arange(lo, hi + ds * 0.5, ds)
        if abs(s_axis[0] - s0) > 1e-6:
            s_axis = np.concatenate([[s0], s_axis])
        if abs(s_axis[-1] - s1) > 1e-6:
            s_axis = np.concatenate([s_axis, [s1]])
        s_axis = np.unique(np.round(s_axis, 8))
        span = s1 - s0
        fractions = (s_axis - s0) / span if abs(span) > 1e-8 else np.zeros_like(s_axis)
    path = np.array([pD + s * axis_hat for s in s_axis])
    h1 = pD + s1 * axis_hat
    return fractions, path, h0, h1, s_axis


def run_hbond_transfer_scan(enames, apos, h_idx, donor_idx, acceptor_idx, ds=DEFAULT_DS, fractions=None, sk_set=None, work_dir='.', r_xh=1.01, verbose=True, on_fail='raise'):
    """Rigid scan: move H along transfer path; all other atoms fixed. Returns dict with energies in eV."""
    from spammm.quantum.DFTB_utils import get_sk_path, run_dftb_sp
    enames = list(enames)
    apos = np.asarray(apos, dtype=float).copy()
    user_fractions = fractions is not None
    fractions, path, h0, h1, s_axis = make_hbond_transfer_path(apos, h_idx, donor_idx, acceptor_idx, ds=ds, fractions=fractions, r_xh=r_xh)
    sk_prefix = get_sk_path(sk_set)
    os.makedirs(work_dir, exist_ok=True)
    energies_ha = []
    prev_charges = None
    for i, (f, hp, s) in enumerate(zip(fractions, path, s_axis)):
        apos_i = apos.copy()
        apos_i[h_idx] = hp
        pt_dir = os.path.join(work_dir, f'pt_{i:03d}')
        if verbose:
            d_d = np.linalg.norm(hp - apos[donor_idx])
            d_a = np.linalg.norm(hp - apos[acceptor_idx])
            print(f"  pt {i:2d} s={s:.3f}Å f={f:.3f}  d(donor-H)={d_d:.3f}  d(acceptor-H)={d_a:.3f} Å")
        try:
            e_ha = run_dftb_sp(pt_dir, enames, apos_i, sk_prefix, maxscc=400, restart_charges_from=prev_charges)
        except RuntimeError as exc:
            if on_fail == 'skip':
                if verbose:
                    print(f"    SKIP: {exc}")
                energies_ha.append(np.nan)
                continue
            raise
        prev_charges = os.path.join(pt_dir, 'charges.bin')
        energies_ha.append(e_ha)
        if verbose:
            print(f"    E = {e_ha * HAU2EV:.4f} eV")
    energies_ha = np.array(energies_ha, dtype=float)
    ok = np.isfinite(energies_ha)
    energies_ev = np.where(ok, energies_ha * HAU2EV, np.nan)
    rel_ev = energies_ev - np.nanmin(energies_ev)
    return {'fractions': fractions, 's_axis': s_axis, 'path': path, 'h0': h0, 'h1': h1, 'energies_ha': energies_ha, 'energies_ev': energies_ev, 'rel_ev': rel_ev, 'h_idx': h_idx, 'donor_idx': donor_idx, 'acceptor_idx': acceptor_idx, 'ds': None if user_fractions else ds}


def write_hbond_scan_xyz(enames, apos, result, fname):
    """Multi-frame XYZ with f, E [eV], dE [eV] in comment line (jmol/movie-friendly)."""
    from spammm.quantum.DFTB_utils import save_xyz_movie
    apos = np.asarray(apos, dtype=float)
    h_idx = result['h_idx']
    frames = []
    for i, (f, hp, s) in enumerate(zip(result['fractions'], result['path'], result['s_axis'])):
        apos_i = apos.copy()
        apos_i[h_idx] = hp
        e_ev = result['energies_ev'][i] if i < len(result['energies_ev']) else np.nan
        dE = result['rel_ev'][i] if i < len(result['rel_ev']) else np.nan
        frames.append({'apos': apos_i, 'enames': list(enames), 's': float(s), 'f': float(f), 'E': float(e_ev), 'dE': float(dE)})
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
    save_xyz_movie(frames, fname, key_order=['s', 'f', 'E', 'dE'])


def plot_hbond_scan(result, atoms, title, savepath):
    """Energy profile + inset geometry sketch (H path)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rel = result['rel_ev']
    ok = np.isfinite(rel)
    x = result.get('s_axis', result['fractions'])
    xlabel = 's along D-A axis [Å]' if 's_axis' in result else 'transfer fraction (0=donor, 1=acceptor)'
    fig, (ax_e, ax_g) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]})
    ax_e.plot(x[ok], rel[ok], 'o-', lw=1.2, ms=3)
    ax_e.set_xlabel(xlabel)
    ax_e.set_ylabel('E − E_min [eV]')
    ax_e.set_title(title)
    ax_e.grid(True, alpha=0.3)
    ih, ido, iac = result['h_idx'], result['donor_idx'], result['acceptor_idx']
    pD, pA = atoms.apos[ido, :2], atoms.apos[iac, :2]
    path2 = result['path'][:, :2]
    ax_g.plot(pD[0], pD[1], 'o', color='blue', ms=10, label=f"{atoms.enames[ido]} (donor)")
    ax_g.plot(pA[0], pA[1], 's', color='red', ms=10, label=f"{atoms.enames[iac]} (acceptor)")
    ax_g.plot(path2[:, 0], path2[:, 1], '.-', color='green', lw=1, ms=6, label='H path')
    ax_g.plot(path2[0, 0], path2[0, 1], '*', color='green', ms=14, label='H start')
    ax_g.plot(path2[-1, 0], path2[-1, 1], 'x', color='darkgreen', ms=12, mew=2, label='H end')
    ax_g.set_aspect('equal'); ax_g.legend(fontsize=8); ax_g.set_title('H transfer path (xy)'); ax_g.axis('off')
    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {savepath}")


def save_hbond_scan_artifacts(result, atoms, name, pair_idx=0, out_dir=None):
    """Write PNG profile + XYZ movie to debug/test_hbond_scan/. Returns (png_path, xyz_path)."""
    out_dir = out_dir or DEBUG_DIR
    os.makedirs(out_dir, exist_ok=True)
    ido, iac, ih = result['donor_idx'], result['acceptor_idx'], result['h_idx']
    label = f"{atoms.enames[ido]}{ido}-H{ih}...{atoms.enames[iac]}{iac}"
    stem = f'hbond_{name}_p{pair_idx}'
    png = os.path.join(out_dir, f'{stem}.png')
    xyz = os.path.join(out_dir, f'{stem}.xyz')
    plot_hbond_scan(result, atoms, f'{name}: {label}', png)
    write_hbond_scan_xyz(atoms.enames, atoms.apos, result, xyz)
    return png, xyz


def build_ascii_hbond_system(name, art=None, hbond_length=3.0, relax_bonds=True):
    """Parse ASCII example, cap H, resolve H-bonds. Returns AtomicSystem."""
    from spammm.topology.ascii_art_heterocycle import parse_ascii_art, ASCII_EXAMPLES, resolve_hbond_pairs, _build_target_valence, jacobi_relax_bond_lengths
    from spammm.topology.KekulePure import make_n_pi
    if art is None:
        art = ASCII_EXAMPLES[name]
    atoms = parse_ascii_art(art, hbond_length=hbond_length)
    atoms.neighs()
    n_pi = make_n_pi(atoms)
    tv = _build_target_valence(atoms, n_pi)
    atoms.add_capping_h_sp2(target_valence=tv)
    atoms.neighs()
    if relax_bonds:
        jacobi_relax_bond_lengths(atoms, n_iters=3, bmix=0.3)
    resolve_hbond_pairs(atoms)
    return atoms


def ascii_examples_with_hbonds():
    """ASCII_EXAMPLES keys whose art contains ``':'`` H-bond markers."""
    from spammm.topology.ascii_art_heterocycle import ASCII_EXAMPLES
    return sorted(k for k, art in ASCII_EXAMPLES.items() if ':' in art)
