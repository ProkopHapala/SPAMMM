"""coordinate_scan.py — DFTB reaction-coordinate paths and pm-NEB trajectories for H-bond transfer.

Generalizes hbond_scan from fixed 0.1 Å axis grids to **m-dimensional control grids** on
any geometry with detected bridging H-bonds. Poor-man's NEB: optional DFTB relax at u=0/u=1,
linear interpolation of **all** atoms, Mulliken SP per frame for charges.

- **SSOT output:** `ScanDataset` via `dataset_from_frames` (fractions derived from controls + mapping, not stored).
- **Methods:** `run_rigid_dftb_scan`, `run_pm_neb` (relax / SP / charges-only pass).
- **Corner square (2 junctions):** `run_corner_scan` relaxes the 4 proton states LL/RL/LR/RR
  (H+bond-partner pinned per corner so mixed states cannot collapse), connects them by
  interpolated edge + diagonal paths, and evaluates J = E_RR+E_LL−E_RL−E_LR (cooperativity).
  `plot_corner_scan` → 2×2 geometry panels + closed perimeter E loop + diagonals;
  `plot_corner_overlay` → skeleton overlays (absolute / Kabsch / junction-frame);
  `write_corner_scan_xyz` → corner+path movie.
- **Shared helpers used:** `plotUtils.draw_mol_junctions` / `plot_geom_overlay`,
  `atomicUtils.rigid_align`/`kabsch_rotation`/`rmsd`, `hbond_utils.junction_bond_lengths`.
- **Caveats:** `endpoints_relaxed` in meta only when both DFTB opts succeed; charge restart chains across frames.
- **Docs:** `doc/Topics/ReactionCoordinateScan.md`
"""
import os
import numpy as np

from spammm.topology.hbond_utils import controls_to_fractions, HbondRecord, junction_bond_lengths
from spammm.topology.scan_dataset import bond_lengths
from spammm.quantum.hbond_scan import make_hbond_transfer_path, DEFAULT_DS, HAU2EV

__all__ = [
    'DEFAULT_DS', 'HAU2EV', 'build_control_grid', 'position_h_at_fraction', 'build_frame',
    'build_pm_neb_endpoints', 'interpolate_all_atoms', 'run_rigid_dftb_scan', 'run_pm_neb',
    'run_pm_neb_sp', 'dataset_from_frames', 'make_hbond_transfer_path',
    'CORNER_US', 'CORNER_NAMES', 'SQUARE_PATHS', 'run_corner_scan', 'plot_corner_scan',
    'run_corner_scan_pbc', 'plot_corner_diagram',
    'plot_corner_overlay', 'write_corner_scan_xyz', 'junction_bond_lengths',
]


def _axis_grid(lo, hi, dx):
    lo, hi = float(lo), float(hi)
    if hi < lo:
        lo, hi = hi, lo
    u = np.arange(lo, hi + dx * 0.5, dx)
    if len(u) == 0 or abs(u[0] - lo) > 1e-8:
        u = np.concatenate([[lo], u])
    if abs(u[-1] - hi) > 1e-8:
        u = np.concatenate([u, [hi]])
    return np.unique(np.round(u, 8))


def build_control_grid(ranges, dx=DEFAULT_DS):
    """Build control grid. ranges: list of (lo, hi) per control → controls [nframes, m]."""
    axes = [_axis_grid(lo, hi, dx) for lo, hi in ranges]
    if len(axes) == 1:
        return axes[0][:, np.newaxis]
    mg = np.meshgrid(*axes, indexing='ij')
    return np.column_stack([g.ravel() for g in mg])


def position_h_at_fraction(apos, h_idx, donor_idx, acceptor_idx, f, r_xh=1.01):
    _, path, _, _, _ = make_hbond_transfer_path(apos, h_idx, donor_idx, acceptor_idx, fractions=[f], r_xh=r_xh)
    return path[0]


def build_frame(apos_ref, hbonds, control_row, mapping, r_xh=1.01):
    """Rigid frame: move scan H atoms; all others at apos_ref."""
    apos = np.asarray(apos_ref, dtype=float).copy()
    fracs = controls_to_fractions(control_row, mapping)
    for hb, f in zip(hbonds, fracs):
        apos[hb.h_idx] = position_h_at_fraction(apos_ref, hb.h_idx, hb.donor_idx, hb.acceptor_idx, f, r_xh=r_xh)
    return apos


def interpolate_all_atoms(apos_start, apos_end, fractions):
    """Linear interp of **all** atom coordinates. fractions: [nframes] in [0,1]."""
    t = np.asarray(fractions, dtype=float).reshape(-1, 1, 1)
    a0 = np.asarray(apos_start, dtype=float)
    a1 = np.asarray(apos_end, dtype=float)
    return a0[np.newaxis, :, :] + t * (a1 - a0)[np.newaxis, :, :]


def _enames_to_etype(enames):
    from spammm import elements as el
    return np.array([el.ELEMENT_DICT[e][0] for e in enames], dtype=np.int32)


def _hbonds_from_meta(hbond_dicts):
    return [HbondRecord.from_dict(d) if isinstance(d, dict) else d for d in hbond_dicts]


def dataset_from_frames(etype, bonds, atom_ids, apos_frames, controls, energies_ev, meta, charges=None, esp_xy=None):
    from spammm.topology.scan_dataset import ScanDataset
    apos_frames = np.asarray(apos_frames, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, np.newaxis]
    bl = bond_lengths(apos_frames, bonds)
    return ScanDataset(etype, bonds, atom_ids, apos_frames, controls, bond_len=bl, energies_ev=energies_ev, meta=meta, charges=charges, esp_xy=esp_xy)


def _dftb_sp_charges(pt_dir, enames, apos_i, sk_prefix, prev_charges, on_fail, verbose, frame_label='', filling_temp=None, rescue_temp=None):
    from spammm.quantum.DFTB_utils import run_dftb_sp
    try:
        e_ha, q = run_dftb_sp(pt_dir, enames, apos_i, sk_prefix, maxscc=400, restart_charges_from=prev_charges, return_charges=True, filling_temp=filling_temp)
    except RuntimeError as exc:
        if rescue_temp:
            try:
                if verbose:
                    print(f"    retry{frame_label} with Fermi T={rescue_temp:.0f} K ({exc})")
                e_ha, q = run_dftb_sp(pt_dir, enames, apos_i, sk_prefix, maxscc=400, restart_charges_from=prev_charges, return_charges=True, filling_temp=rescue_temp)
                return e_ha, q, os.path.join(pt_dir, 'charges.bin')
            except RuntimeError as exc2:
                exc = exc2
        if on_fail == 'skip':
            if verbose:
                print(f"    SKIP charges{frame_label}: {exc}")
            return np.nan, None, prev_charges
        raise
    return e_ha, q, os.path.join(pt_dir, 'charges.bin')


def run_rigid_dftb_scan(enames, apos_ref, hbonds, mapping, ranges=None, dx=DEFAULT_DS, etype=None, bonds=None, atom_ids=None, sk_set=None, work_dir='.', r_xh=1.01, verbose=True, on_fail='skip', meta=None, collect_charges=True):
    """Rigid DFTB SP along control grid; H moves, heavy atoms fixed."""
    from spammm.quantum.DFTB_utils import get_sk_path, run_dftb_sp
    enames = list(enames)
    apos_ref = np.asarray(apos_ref, dtype=float)
    hbonds = _hbonds_from_meta(hbonds)
    m = max(mapping) + 1
    ranges = ranges or [(0.0, 1.0)] * m
    controls = build_control_grid(ranges, dx=dx)
    sk_prefix = get_sk_path(sk_set)
    os.makedirs(work_dir, exist_ok=True)
    apos_frames, energies_ha, charge_rows = [], [], []
    prev_charges = None
    for i, u_row in enumerate(controls):
        apos_i = build_frame(apos_ref, hbonds, u_row, mapping, r_xh=r_xh)
        apos_frames.append(apos_i)
        pt_dir = os.path.join(work_dir, f'pt_{i:03d}')
        if verbose:
            print(f"  frame {i:3d} controls={u_row}")
        e_ha, q, prev_charges = _dftb_sp_charges(pt_dir, enames, apos_i, sk_prefix, prev_charges, on_fail, verbose)
        if not np.isfinite(e_ha):
            energies_ha.append(np.nan)
            charge_rows.append(np.full(len(enames), np.nan))
            continue
        energies_ha.append(e_ha)
        charge_rows.append(q if collect_charges and q is not None else np.full(len(enames), np.nan))
        if verbose:
            print(f"    E = {e_ha * HAU2EV:.4f} eV")
    apos_stack = np.array(apos_frames, dtype=np.float64)
    energies_ha = np.array(energies_ha, dtype=float)
    energies_ev = np.where(np.isfinite(energies_ha), energies_ha * HAU2EV, np.nan)
    charges = np.array(charge_rows, dtype=np.float64) if collect_charges and charge_rows else None
    if charges is not None and not np.any(np.isfinite(charges)):
        charges = None
    meta = dict(meta or {})
    meta.update(scan_type='rigid_dftb', dx=dx, mapping=list(mapping), hbond_records=[h.to_dict() for h in hbonds], sk_set=sk_set, charge_type='mulliken' if charges is not None else None)
    if etype is None:
        etype = _enames_to_etype(enames)
    if bonds is None:
        raise ValueError("bonds required for ScanDataset")
    if atom_ids is None:
        atom_ids = np.arange(len(enames), dtype=np.int64)
    return dataset_from_frames(etype, bonds, atom_ids, apos_stack, controls, energies_ev, meta, charges=charges)


def build_pm_neb_endpoints(apos_ref, hbonds, mapping, r_xh=1.01):
    """Rigid isomer endpoints u=0 (H at donors) and u=1 (H at acceptors)."""
    m = max(mapping) + 1 if mapping else 1
    apos_start = build_frame(apos_ref, hbonds, np.zeros(m), mapping, r_xh=r_xh)
    apos_end = build_frame(apos_ref, hbonds, np.ones(m), mapping, r_xh=r_xh)
    return apos_start, apos_end


def run_pm_neb(enames, apos_ref, hbonds, mapping, dx=DEFAULT_DS, relax_endpoints=False, run_sp=False, etype=None, bonds=None, atom_ids=None, sk_set=None, work_dir='.', r_xh=1.01, verbose=True, on_fail='skip', meta=None, collect_charges=True):
    """Poor-man's NEB: optional DFTB relax at u=0/u=1, linear interp all atoms, optional SP along path."""
    from spammm.quantum.DFTB_utils import get_sk_path, run_dftb_relax
    enames = list(enames)
    apos_ref = np.asarray(apos_ref, dtype=float)
    hbonds = _hbonds_from_meta(hbonds)
    m = max(mapping) + 1
    apos_start, apos_end = build_pm_neb_endpoints(apos_ref, hbonds, mapping, r_xh=r_xh)
    os.makedirs(work_dir, exist_ok=True)
    endpoint_meta = {}
    if relax_endpoints:
        e0, apos_start = run_dftb_relax(os.path.join(work_dir, 'endpoint_u0'), enames, apos_start, sk_set=sk_set, verbose=verbose, on_fail=on_fail)
        charges_u0 = os.path.join(work_dir, 'endpoint_u0', 'charges.bin')
        e1, apos_end = run_dftb_relax(os.path.join(work_dir, 'endpoint_u1'), enames, apos_end, sk_set=sk_set, restart_charges_from=charges_u0 if os.path.isfile(charges_u0) and np.isfinite(e0) else None, verbose=verbose, on_fail=on_fail)
        endpoints_ok = np.isfinite(e0) and np.isfinite(e1)
        endpoint_meta = dict(endpoint_E0_ev=float(e0) * HAU2EV if np.isfinite(e0) else np.nan, endpoint_E1_ev=float(e1) * HAU2EV if np.isfinite(e1) else np.nan, endpoints_relaxed=endpoints_ok)
        if not endpoints_ok and verbose:
            print("  WARN: endpoint DFTB relax incomplete — trajectory uses unrelaxed endpoint(s); bond lengths may not change")
    u1d = build_control_grid([(0.0, 1.0)], dx=dx)[:, 0]
    apos_stack = interpolate_all_atoms(apos_start, apos_end, u1d)
    controls = u1d[:, np.newaxis] if m == 1 else np.column_stack([u1d] * m)
    energies_ha = np.full(len(apos_stack), np.nan)
    charge_rows = None
    sk_prefix = get_sk_path(sk_set)
    if run_sp:
        prev_charges = os.path.join(work_dir, 'endpoint_u1', 'charges.bin') if relax_endpoints and os.path.isfile(os.path.join(work_dir, 'endpoint_u1', 'charges.bin')) else None
        charge_rows = []
        for i, apos_i in enumerate(apos_stack):
            pt_dir = os.path.join(work_dir, f'pm_{i:03d}')
            if verbose:
                print(f"  pm-neb SP frame {i:3d} u={u1d[i]:.3f}")
            e_ha, q, prev_charges = _dftb_sp_charges(pt_dir, enames, apos_i, sk_prefix, prev_charges, on_fail, verbose, frame_label=f' frame {i}')
            if not np.isfinite(e_ha):
                charge_rows.append(np.full(len(enames), np.nan))
                continue
            energies_ha[i] = e_ha
            charge_rows.append(q if collect_charges and q is not None else np.full(len(enames), np.nan))
            if verbose:
                print(f"    E = {e_ha * HAU2EV:.4f} eV")
    elif collect_charges:
        prev_charges = os.path.join(work_dir, 'endpoint_u1', 'charges.bin') if relax_endpoints and os.path.isfile(os.path.join(work_dir, 'endpoint_u1', 'charges.bin')) else None
        charge_rows = []
        for i, apos_i in enumerate(apos_stack):
            pt_dir = os.path.join(work_dir, f'chg_{i:03d}')
            if verbose:
                print(f"  Mulliken SP frame {i:3d} u={u1d[i]:.3f}")
            e_ha, q, prev_charges = _dftb_sp_charges(pt_dir, enames, apos_i, sk_prefix, prev_charges, on_fail, verbose, frame_label=f' frame {i}')
            charge_rows.append(q if q is not None else np.full(len(enames), np.nan))
            if verbose and q is not None:
                print(f"    q_sum={q.sum():.4f} e")
    charges = np.array(charge_rows, dtype=np.float64) if charge_rows is not None and np.any(np.isfinite(np.array(charge_rows))) else None
    energies_ev = np.where(np.isfinite(energies_ha), energies_ha * HAU2EV, np.nan)
    if run_sp:
        scan_type = 'pm_neb_sp'
    elif relax_endpoints and endpoint_meta.get('endpoints_relaxed'):
        scan_type = 'pm_neb_relaxed'
    else:
        scan_type = 'pm_neb_preview'
    meta = dict(meta or {})
    meta.update(scan_type=scan_type, dx=dx, mapping=list(mapping), hbond_records=[h.to_dict() for h in hbonds], sk_set=sk_set, pm_neb_all_atoms=True, charge_type='mulliken' if charges is not None else None, **endpoint_meta)
    if etype is None:
        etype = _enames_to_etype(enames)
    if bonds is None:
        raise ValueError("bonds required for ScanDataset")
    if atom_ids is None:
        atom_ids = np.arange(len(enames), dtype=np.int64)
    return dataset_from_frames(etype, bonds, atom_ids, apos_stack, controls, energies_ev, meta, charges=charges)


def run_pm_neb_sp(enames, apos_ref, hbonds, mapping, dx=DEFAULT_DS, etype=None, bonds=None, atom_ids=None, sk_set=None, work_dir='.', r_xh=1.01, verbose=True, on_fail='skip', meta=None):
    """Poor-man's NEB: rigid endpoints, interpolate all atoms, DFTB SP each frame."""
    return run_pm_neb(enames, apos_ref, hbonds, mapping, dx=dx, relax_endpoints=False, run_sp=True, etype=etype, bonds=bonds, atom_ids=atom_ids, sk_set=sk_set, work_dir=work_dir, r_xh=r_xh, verbose=verbose, on_fail=on_fail, meta=meta)


# ---------------------------------------------------------------------------
# Four-corner square (two H-bond junctions, u1/u2 ∈ {0,1})
# ---------------------------------------------------------------------------

CORNER_US = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
CORNER_NAMES = {(0.0, 0.0): 'LL', (1.0, 0.0): 'RL', (0.0, 1.0): 'LR', (1.0, 1.0): 'RR'}
SQUARE_PATHS = [
    ((0.0, 0.0), (1.0, 0.0), 'edge u2=0: LL→RL'),
    ((0.0, 1.0), (1.0, 1.0), 'edge u2=1: LR→RR'),
    ((0.0, 0.0), (0.0, 1.0), 'edge u1=0: LL→LR'),
    ((1.0, 0.0), (1.0, 1.0), 'edge u1=1: RL→RR'),
    ((0.0, 0.0), (1.0, 1.0), 'diag LL→RR'),
    ((1.0, 0.0), (0.0, 1.0), 'diag RL→LR'),
]


def run_corner_scan(enames, apos_ref, hbonds, mapping, dx=0.25, relax_corners=True, fix_scan_h=True, sk_set=None, work_dir='.', r_xh=1.01, verbose=True, on_fail='skip', filling_temp=None, rescue_temp=600.0, meta=None):
    """Four-state thermodynamic square for two H-bond junctions.

    Protocol:
    1. Relax each corner (u1,u2) ∈ {0,1}² (L = proton on donor, R = proton on acceptor).
       LL is relaxed first; RL/LR/RR guesses are then built on the RELAXED LL scaffold —
       only the scan H's move (to r_xh from the partner they should bond to), so the
       scaffold starts from a real minimum and drifts minimally.
       With *fix_scan_h*, each corner's relax pins the scan H **and the atom it should
       bond to** (donor at L, acceptor at R) via MovedAtoms exclusion — proton assignment
       cannot hop and mixed corners cannot collapse; free junction partners + the rest
       of the molecule still relax.
    2. Connect relaxed corners by all-atom linear interpolation along the 4 square edges
       and both diagonals; DFTB SP per frame (charge restart along each path).
    3. J = E_RR + E_LL − E_RL − E_LR  (cooperative if < 0).

    *filling_temp*: uniform Fermi smearing T [K] for all calculations (uniform protocol for
    SCC-difficult systems). *rescue_temp*: one retry of failed SCC runs at this T (None off).

    Each corner stores 'bonds' = junction_bond_lengths (per-junction D–H / H···A [Å]) —
    always check these to verify the corner states are genuinely distinct.

    Returns dict with 'corners' {u: {name,apos,e_ev,bonds}}, 'paths' [{uA,uB,label,fracs,
    energies_ev,apos_frames}], 'J_ev', 'apos_ref', 'hbonds', 'meta'.
    """
    from spammm.quantum.DFTB_utils import get_sk_path, run_dftb_relax
    enames = list(enames)
    apos_ref = np.asarray(apos_ref, dtype=float)
    hbonds = _hbonds_from_meta(hbonds)
    m = max(mapping) + 1
    assert m == 2, f"corner square needs m=2 controls, got m={m}"
    sk_prefix = get_sk_path(sk_set)
    os.makedirs(work_dir, exist_ok=True)

    def _corner_geometry(u, apos_base):
        """Corner guess on scaffold *apos_base*: each scan H at covalent distance r_xh from the
        atom it should bond to (donor at L, acceptor at R) along that junction's D→A axis;
        ALL other atoms stay at apos_base — only the H's move."""
        apos = apos_base.copy()
        for j, uj in enumerate(u):
            hb = hbonds[mapping[j]]
            pD, pA = apos_base[hb.donor_idx], apos_base[hb.acceptor_idx]
            axis = pA - pD
            axis /= np.linalg.norm(axis)
            apos[hb.h_idx] = (pD + r_xh * axis) if uj < 0.5 else (pA - r_xh * axis)
        return apos

    def _fixed_for(u):
        """Pin each scan H + the atom it should bond to (donor at L, acceptor at R) so the
        proton assignment cannot hop and the H–X bond keeps its length; the free junction
        partner and the rest of the molecule still relax."""
        if not fix_scan_h:
            return None
        return sorted({i for j, uj in enumerate(u) for i in (hbonds[mapping[j]].h_idx, hbonds[mapping[j]].donor_idx if uj < 0.5 else hbonds[mapping[j]].acceptor_idx)})

    corners = {}
    prev_charges = None
    scaffold = apos_ref
    for u in CORNER_US:
        cname = CORNER_NAMES[u]
        cdir = os.path.join(work_dir, f'corner_{cname}')
        apos_c = _corner_geometry(u, scaffold)
        if relax_corners:
            if verbose:
                print(f"  relax corner {cname} u={u}")
            e_ha, apos_r = run_dftb_relax(cdir, enames, apos_c, sk_set=sk_set, restart_charges_from=prev_charges, verbose=verbose, on_fail=on_fail, fixed_atoms=_fixed_for(u), filling_temp=filling_temp)
            if not np.isfinite(e_ha) and rescue_temp:
                if verbose:
                    print(f"    retry corner {cname} with Fermi T={rescue_temp:.0f} K")
                e_ha, apos_r = run_dftb_relax(cdir, enames, apos_c, sk_set=sk_set, restart_charges_from=prev_charges, verbose=verbose, on_fail=on_fail, fixed_atoms=_fixed_for(u), filling_temp=rescue_temp)
            chg = os.path.join(cdir, 'charges.bin')
            if np.isfinite(e_ha) and os.path.isfile(chg):
                prev_charges = chg
        else:
            e_ha, _, prev_charges = _dftb_sp_charges(cdir, enames, apos_c, sk_prefix, prev_charges, on_fail, verbose, frame_label=f' corner {cname}', filling_temp=filling_temp, rescue_temp=rescue_temp)
            apos_r = apos_c
        corners[u] = dict(name=cname, apos=apos_r, e_ev=float(e_ha) * HAU2EV if np.isfinite(e_ha) else np.nan, bonds=junction_bond_lengths(apos_r, hbonds))
        if u == CORNER_US[0] and np.isfinite(e_ha):
            scaffold = apos_r  # build RL/LR/RR guesses on the relaxed LL scaffold → minimal scaffold drift
        if verbose:
            bs = '  '.join(f"DH={d1:.3f} HA={d2:.3f}" for d1, d2 in corners[u]['bonds'])
            print(f"    {cname} junctions: {bs}")

    fr = _axis_grid(0.0, 1.0, dx)
    paths = []
    for ip, (uA, uB, lab) in enumerate(SQUARE_PATHS):
        if verbose:
            print(f"  path {ip}: {lab}")
        stack = interpolate_all_atoms(corners[uA]['apos'], corners[uB]['apos'], fr)
        energies_ev = np.full(len(fr), np.nan)
        prev_charges = None
        for i, apos_i in enumerate(stack):
            pt_dir = os.path.join(work_dir, f'path_{ip:02d}_{i:03d}')
            e_ha, _, prev_charges = _dftb_sp_charges(pt_dir, enames, apos_i, sk_prefix, prev_charges, on_fail, verbose, frame_label=f' {lab} f={fr[i]:.2f}', filling_temp=filling_temp, rescue_temp=rescue_temp)
            if np.isfinite(e_ha):
                energies_ev[i] = e_ha * HAU2EV
        paths.append(dict(uA=uA, uB=uB, label=lab, fracs=fr, energies_ev=energies_ev, apos_frames=stack))

    E00, E10, E01, E11 = (corners[u]['e_ev'] for u in CORNER_US)
    J = E11 + E00 - E10 - E01 if np.isfinite([E00, E10, E01, E11]).all() else np.nan
    meta = dict(meta or {})
    meta.update(scan_type='corner_square', dx=dx, relax_corners=relax_corners, fix_scan_h=fix_scan_h, filling_temp=filling_temp, rescue_temp=rescue_temp, mapping=list(mapping), hbond_records=[h.to_dict() for h in hbonds], sk_set=sk_set, J_ev=J)
    return dict(corners=corners, corner_us=CORNER_US, paths=paths, J_ev=J, hbonds=hbonds, enames=enames, apos_ref=apos_ref, meta=meta)


def run_corner_scan_pbc(enames, apos_ref, lvs, hbonds, mapping, dx=0.25, relax_corners=True, fix_scan_h=True, sk_set=None, work_dir='.', r_xh=1.01, nk=(1, 8, 1), k_shift=(0.5, 0.5, 0.5), filling_temp=300.0, rescue_temp=600.0, verbose=True, on_fail='skip', meta=None):
    """Four-state corner square for a PERIODIC H-bond chain cell (DFTB+ PBC).

    Same protocol as `run_corner_scan` (relax corners pinned H+bond-partner, then
    interpolated edge/diagonal paths with SP), but energies come from `run_pbc`.
    Junctions may partner periodic images (HbondRecord.d_shift/a_shift) — all
    junction geometry uses `hbond_positions` (image-aware), so the boundary
    junction's H is placed toward the image acceptor across the cell boundary.

    Pinning note: `fixed_atoms` carries IN-CELL indices — under PBC an atom's
    periodic image moves with it, so pinning the in-cell acceptor index pins the
    image partner of a boundary junction as intended.

    Args:
        lvs: (3,3) lattice vectors [Å] (chain along lvs[1]).
        nk, k_shift: k-point folding/shift for `run_pbc` (1D chain: nk=(1,Nk,1)).
        filling_temp: Fermi smearing T [K] used uniformly (default 300 K for PBC).
        rescue_temp: one retry at this T on SCC failure (None disables).

    Returns the same dict shape as `run_corner_scan`, plus 'lvs' (needed by
    `plot_corner_scan` to draw boundary junctions to image atoms).
    """
    from spammm.quantum.DFTB_utils import run_pbc
    from spammm.topology.hbond_utils import hbond_positions
    enames = list(enames)
    apos_ref = np.asarray(apos_ref, dtype=float)
    lvs = np.asarray(lvs, dtype=float)
    hbonds = _hbonds_from_meta(hbonds)
    assert max(mapping) + 1 == 2, "corner square needs m=2 controls"
    os.makedirs(work_dir, exist_ok=True)

    def _corner_geometry(u, apos_base):
        apos = apos_base.copy()
        for j, uj in enumerate(u):
            hb = hbonds[mapping[j]]
            pD, _, pA = hbond_positions(apos_base, hb, lvs)
            axis = (pA - pD) / np.linalg.norm(pA - pD)
            apos[hb.h_idx] = (pD + r_xh * axis) if uj < 0.5 else (pA - r_xh * axis)
        return apos

    def _fixed_for(u):
        """Pin the whole junction scaffold: all 4 junction heteroatoms + both scan H's.
        The H's are pinned at their corner position (H + bond partner fixed by the
        heteroatom pins) — prevents proton hop AND molecule sliding/re-registration
        along the chain; all internal bonds still relax freely."""
        if not fix_scan_h:
            return None
        return sorted({i for j in range(len(mapping)) for i in (hbonds[mapping[j]].donor_idx, hbonds[mapping[j]].h_idx, hbonds[mapping[j]].acceptor_idx)})

    def _try_pbc(cdir, apos_in, do_relax, fixed=None, label=''):
        """run_pbc with escalating rescue tiers (T, mixing, MaxScc); returns (E_ha, apos_out)."""
        tiers = [(filling_temp, 0.2, 200)]
        if rescue_temp:
            tiers += [(rescue_temp, 0.2, 200), (rescue_temp, 0.05, 500)]
        for T, mix, maxscc in tiers:
            try:
                e_ha, apos_out, _ = run_pbc(apos_in, enames, lvs, sk_set=sk_set, do_relax=do_relax, fixed_atoms=fixed, nk=nk, k_shift=k_shift, workdir=cdir, Temperature=T, MixingParameter=mix, MaxScc=maxscc)
                return e_ha, apos_out
            except RuntimeError as ex:
                if verbose:
                    print(f"    {label}: DFTB+ failed at T={T:.0f} K mix={mix} ({str(ex).splitlines()[0]})")
        return np.nan, apos_in

    corners = {}
    scaffold = apos_ref
    for u in CORNER_US:
        cname = CORNER_NAMES[u]
        apos_c = _corner_geometry(u, scaffold)
        cdir = os.path.join(work_dir, f'corner_{cname}')
        if relax_corners:
            if verbose:
                print(f"  relax corner {cname} u={u} (PBC, nk={nk})")
            e_ha, apos_r = _try_pbc(cdir, apos_c, True, fixed=_fixed_for(u), label=f'corner {cname}')
        else:
            e_ha, apos_r = _try_pbc(cdir, apos_c, False, label=f'corner {cname} SP')
        corners[u] = dict(name=cname, apos=apos_r, e_ev=float(e_ha) * HAU2EV if np.isfinite(e_ha) else np.nan, bonds=junction_bond_lengths(apos_r, hbonds, lvs))
        if u == CORNER_US[0] and np.isfinite(e_ha):
            scaffold = apos_r
        if verbose:
            bs = '  '.join(f"DH={d1:.3f} HA={d2:.3f}" for d1, d2 in corners[u]['bonds'])
            print(f"    {cname} junctions: {bs}  E={corners[u]['e_ev']:.4f} eV")

    fr = _axis_grid(0.0, 1.0, dx)
    paths = []
    for ip, (uA, uB, lab) in enumerate(SQUARE_PATHS):
        if verbose:
            print(f"  path {ip}: {lab}")
        stack = interpolate_all_atoms(corners[uA]['apos'], corners[uB]['apos'], fr)
        energies_ev = np.full(len(fr), np.nan)
        for i, apos_i in enumerate(stack):
            e_ha, _ = _try_pbc(os.path.join(work_dir, f'path_{ip:02d}_{i:03d}'), apos_i, False, label=f'{lab} f={fr[i]:.2f}')
            if np.isfinite(e_ha):
                energies_ev[i] = e_ha * HAU2EV
        paths.append(dict(uA=uA, uB=uB, label=lab, fracs=fr, energies_ev=energies_ev, apos_frames=stack))

    E00, E10, E01, E11 = (corners[u]['e_ev'] for u in CORNER_US)
    J = E11 + E00 - E10 - E01 if np.isfinite([E00, E10, E01, E11]).all() else np.nan
    meta = dict(meta or {})
    meta.update(scan_type='corner_square_pbc', dx=dx, relax_corners=relax_corners, fix_scan_h=fix_scan_h, filling_temp=filling_temp, rescue_temp=rescue_temp, nk=list(nk), k_shift=list(k_shift), lvs=[list(v) for v in lvs], mapping=list(mapping), hbond_records=[h.to_dict() for h in hbonds], sk_set=sk_set, J_ev=J)
    scan = dict(corners=corners, corner_us=CORNER_US, paths=paths, J_ev=J, hbonds=hbonds, enames=enames, apos_ref=apos_ref, lvs=lvs, meta=meta)
    save_scan(scan, os.path.join(work_dir, 'scan.pkl'))
    return scan


def save_scan(scan, path):
    """Pickle the corner-scan dict so figures can be replotted without rerunning DFTB."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(scan, f)


def load_scan(path):
    """Load a scan dict saved by `save_scan` (HbondRecord objects restore as-is)."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_corner_overlay(scan, atoms, savepath, sz=30.):
    """4 relaxed corners overlaid as thin bond skeletons (thin wrapper over
    `plotUtils.plot_geom_overlay`), 3 panels:
    absolute coordinates | Kabsch on all atoms | Kabsch on the 4 junction heteroatoms
    (D1,A1,D2,A2 — pins the junction frame, shows monomer swing)."""
    import matplotlib
    matplotlib.use('Agg')
    from spammm import plotUtils as _pu
    us = scan['corner_us']
    hbs = scan['hbonds']
    junc_heavy = [i for hb in hbs for i in (hb.donor_idx, hb.acceptor_idx)]
    geoms = [scan['corners'][u]['apos'] for u in us]
    labels = [scan['corners'][u]['name'] for u in us]
    modes = [('absolute', None, 'absolute coordinates (no alignment)'),
             ('kabsch', None, f'Kabsch aligned to {labels[0]} (all atoms)'),
             ('kabsch', junc_heavy, 'Kabsch on junction heteroatoms (D1,A1,D2,A2)')]
    markers = [(i, f'{t}{j+1}') for j, hb in enumerate(hbs) for i, t in ((hb.donor_idx, 'D'), (hb.h_idx, 'H'), (hb.acceptor_idx, 'A'))]
    _pu.plot_geom_overlay(geoms, atoms.bonds, labels, ref=0, modes=modes, markers=markers, savepath=savepath)


def plot_corner_scan(scan, atoms, title, savepath):
    """Square figure: 2×2 relaxed-corner geometry panels + closed perimeter E loop + diagonals.

    Energy panel: 4 square edges concatenated into one closed traversal LL→RL→RR→LR→LL
    (x = perimeter coordinate 0..4, corner names labeled at integer nodes); both diagonals
    vs their own fraction in a second panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spammm import plotUtils as _pu
    from spammm.atomicUtils import rigid_align
    corners, paths = scan['corners'], scan['paths']
    Emin = np.nanmin([c['e_ev'] for c in corners.values()] + [np.nanmin(p['energies_ev']) for p in paths if np.isfinite(p['energies_ev']).any()])
    pos = {(0.0, 0.0): (1, 0), (1.0, 0.0): (1, 1), (0.0, 1.0): (0, 0), (1.0, 1.0): (0, 1)}
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.35], wspace=0.05, hspace=0.3)
    anchor = sorted({i for hb in scan['hbonds'] for i in (hb.donor_idx, hb.h_idx, hb.acceptor_idx)})
    ref = scan.get('apos_ref')
    for u, (r, c) in pos.items():
        ax = fig.add_subplot(gs[r, c])
        c0 = corners[u]
        dE = c0['e_ev'] - Emin
        bs = '  '.join(f"{d1:.2f}/{d2:.2f}" for d1, d2 in c0.get('bonds', []))
        lab = f"{c0['name']}  u={u}\n" + (f"ΔE={dE:.3f} eV" if np.isfinite(dE) else "relax FAILED") + (f"\nD–H/H···A: {bs}" if bs else "")
        # PBC: lattice pins orientation -> plot absolute coords (Kabsch on the
        # near-collinear junction atoms would leave a free twist about the
        # junction axis and rotate the tilted molecular planes edge-on)
        apos_c = c0['apos'] if scan.get('lvs') is not None else (rigid_align(c0['apos'], ref, idx=anchor) if ref is not None else c0['apos'])
        # PBC scan pins the whole junction scaffold (D,H,A of each junction)
        fixed = sorted({i for hb in scan['hbonds'] for i in (hb.donor_idx, hb.h_idx, hb.acceptor_idx)}) if scan.get('lvs') is not None and scan.get('meta', {}).get('fix_scan_h') else None
        _pu.draw_mol_junctions(ax, atoms, apos_c, scan['hbonds'], label=lab, lvs=scan.get('lvs'), jnames=[str(j + 1) for j in range(len(scan['hbonds']))], fixed_idx=fixed)

    # --- perimeter loop: edges concatenated LL→RL→RR→LR→LL -------------------
    ax_l = fig.add_subplot(gs[0, 2])
    edge_map = {p['label']: p for p in paths if 'diag' not in p['label']}
    loop_segs = [('edge u2=0: LL→RL', False), ('edge u1=1: RL→RR', False), ('edge u2=1: LR→RR', True), ('edge u1=0: LL→LR', True)]
    loop_nodes = ['LL', 'RL', 'RR', 'LR', 'LL']
    seg_colors = ['C0', 'C3', 'C1', 'C2']
    for k, (lab, rev) in enumerate(loop_segs):
        p = edge_map.get(lab)
        if p is None:
            continue
        E = p['energies_ev'] - Emin
        x = p['fracs']
        if rev:
            E, x = E[::-1], 1.0 - x[::-1]
        ok = np.isfinite(E)
        ax_l.plot(k + x[ok], E[ok], 'o-', color=seg_colors[k], lw=1.4, ms=4, label=lab.split(':')[0])
    name2u = {c['name']: u for u, c in corners.items()}
    for k, nm in enumerate(loop_nodes):
        c0 = corners.get(name2u.get(nm))
        if c0 is not None and np.isfinite(c0['e_ev']):
            ax_l.plot(k, c0['e_ev'] - Emin, 's', ms=10, mfc='none', mew=2, color='k', zorder=6)
    ax_l.set_xticks(range(5))
    ax_l.set_xticklabels(loop_nodes)
    ax_l.set_xlim(-0.15, 4.15)
    ax_l.set_ylabel('E − E_min [eV]')
    ax_l.set_xlabel('perimeter coordinate (corners at integers)')
    J = scan['J_ev']
    ax_l.set_title(f"{title}\nJ = E_RR+E_LL−E_RL−E_LR = {J:+.3f} eV" if np.isfinite(J) else f"{title}\nJ = NaN (corner relax failed)")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=7.5, loc='best')

    # --- diagonals ------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 2])
    for p in paths:
        if 'diag' not in p['label']:
            continue
        E = p['energies_ev'] - Emin
        ok = np.isfinite(E)
        ax_d.plot(p['fracs'][ok], E[ok], 's--', lw=1.4, ms=4, color='k' if 'LL' in p['label'] else '0.55', label=p['label'])
    for p in paths:
        if 'diag' not in p['label']:
            continue
        for f, u in ((0.0, p['uA']), (1.0, p['uB'])):
            c0 = corners[u]
            if np.isfinite(c0['e_ev']):
                ax_d.plot(f, c0['e_ev'] - Emin, 'o', ms=7, mfc='none', mew=1.5, color='k')
                ax_d.annotate(c0['name'], (f, c0['e_ev'] - Emin), textcoords='offset points', xytext=(6, -3), fontsize=8)
    ax_d.set_xlabel('diagonal fraction')
    ax_d.set_ylabel('E − E_min [eV]')
    ax_d.set_title('diagonals (concerted / swap)')
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=7.5, loc='best')

    os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {savepath}")


def plot_corner_diagram(scan, atoms, savepath, sz=18.):
    """Energy diagram of the LL->RR transition.  TOP half: row of the four
    corner geometries (LL, RL, LR, RR — matching their x positions below),
    each in a box with an arrow down to its point.  BOTTOM half: energy vs
    reaction progress — stepwise paths via RL (red) / via LR (blue), concerted
    LL->RR diagonal (black dashed), RL<->LR swap (thin gray vertical).

    SVG-compatible: pass savepath ending in .svg to get editable vector output.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spammm import plotUtils as _pu
    corners = scan['corners']
    n2u = {c['name']: u for u, c in corners.items()}
    E = {c['name']: c['e_ev'] for c in corners.values()}
    Emin = np.nanmin(list(E.values()))
    Erel = {k: v - Emin for k, v in E.items()}

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_axes([0.08, 0.08, 0.86, 0.40])
    ax.set_xlabel('reaction progress:  LL → RR')
    ax.set_ylabel('E − E_min [eV]')
    J = scan['J_ev']
    ax.set_title(f"{scan.get('meta', {}).get('name', '')}  J = {J:+.3f} eV" if np.isfinite(J) else 'J = NaN')

    edge = {p['label']: p for p in scan['paths'] if 'diag' not in p['label']}
    diag = {p['label']: p for p in scan['paths'] if 'diag' in p['label']}

    def _edge_xy(lab, x0, x1):
        p = edge.get(lab)
        if p is None:
            return None
        ok = np.isfinite(p['energies_ev'])
        return x0 + (x1 - x0) * p['fracs'][ok], p['energies_ev'][ok] - Emin

    for labs, col, name in ((('edge u2=0: LL→RL', 'edge u1=1: RL→RR'), 'tab:red', 'via RL'),
                            (('edge u1=0: LL→LR', 'edge u2=1: LR→RR'), 'tab:blue', 'via LR')):
        for k, lab in enumerate(labs):
            seg = _edge_xy(lab, 0.5 * k, 0.5 * (k + 1))
            if seg is not None:
                ax.plot(*seg, 'o-', color=col, lw=0.9, ms=3, label=name if k == 0 else None)
        ax.plot(0.5, Erel[name.split()[-1]], 's', color=col, ms=7, mfc='none', mew=1.4, zorder=6)
    p = diag.get('diag LL→RR')
    if p is not None:
        ok = np.isfinite(p['energies_ev'])
        ax.plot(p['fracs'][ok], p['energies_ev'][ok] - Emin, 's--', color='k', lw=0.9, ms=3, label='diag LL→RR (concerted)')
    p = diag.get('diag RL→LR')
    if p is not None:
        ok = np.isfinite(p['energies_ev'])
        ax.plot(0.5 + 0.0 * p['fracs'][ok], p['energies_ev'][ok] - Emin, 'd:', color='0.55', lw=0.7, ms=3, label='diag RL→LR (swap)')
    for x, nm in ((0.0, 'LL'), (1.0, 'RR')):
        ax.plot(x, Erel[nm], 'o', color='k', ms=8, mfc='none', mew=1.6, zorder=7)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(['LL', 'LR / RL', 'RR'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    # top row: corner geometries LL, RL, LR, RR (matching x positions below) +
    # narrow xz side view (axes=(0,2)) next to each -> shows the herringbone tilt
    geo_order = ['LL', 'RL', 'LR', 'RR']
    bw, bwz, bh, y0 = 0.17, 0.055, 0.40, 0.55
    xs = [0.01, 0.26, 0.51, 0.76]
    fixed = sorted({i for hb in scan['hbonds'] for i in (hb.donor_idx, hb.h_idx, hb.acceptor_idx)})
    tgt = dict(LL=(0.0, Erel['LL']), RR=(1.0, Erel['RR']), RL=(0.5, Erel['RL']), LR=(0.5, Erel['LR']))
    for nm, bx in zip(geo_order, xs):
        u = n2u[nm]
        c0 = corners[u]
        axb = fig.add_axes([bx, y0, bw, bh])
        bs = '  '.join(f"{d1:.2f}/{d2:.2f}" for d1, d2 in c0.get('bonds', []))
        _pu.draw_mol_junctions(axb, atoms, c0['apos'], scan['hbonds'], label=f"{nm}  ΔE={Erel[nm]:.2f} eV\n{bs}", sz=sz, lvs=scan.get('lvs'), jnames=[str(j + 1) for j in range(len(scan['hbonds']))], lw=0.8, label_off=0.55, fixed_idx=fixed, frame=True)
        axz = fig.add_axes([bx + bw + 0.004, y0 + 0.10, bwz, 0.24])
        _pu.draw_mol_junctions(axz, atoms, c0['apos'], scan['hbonds'], sz=sz, axes=(0, 2), lvs=scan.get('lvs'), lw=0.8, fixed_idx=fixed, annotate=False, frame=True)
        ax.annotate('', xy=tgt[nm], xycoords='data', xytext=(bx + 0.5 * bw, y0), textcoords='figure fraction',
                    arrowprops=dict(arrowstyle='->', color='0.3', lw=0.9, shrinkA=2, shrinkB=4))

    os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {savepath}")


def write_corner_scan_xyz(scan, atoms, fname):
    """XYZ movie of the 4 relaxed corners (comment: corner name + E)."""
    from spammm.quantum.DFTB_utils import save_xyz_movie
    frames = []
    for u in scan['corner_us']:
        c0 = scan['corners'][u]
        frames.append({'apos': c0['apos'], 'enames': list(scan['enames']), 'corner': c0['name'], 'E': c0['e_ev']})
    os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
    save_xyz_movie(frames, fname, key_order=['corner', 'E'])
