"""coordinate_scan.py — DFTB reaction-coordinate paths and pm-NEB trajectories for H-bond transfer.

Generalizes hbond_scan from fixed 0.1 Å axis grids to **m-dimensional control grids** on
any geometry with detected bridging H-bonds. Poor-man's NEB: optional DFTB relax at u=0/u=1,
linear interpolation of **all** atoms, Mulliken SP per frame for charges.

- **SSOT output:** `ScanDataset` via `dataset_from_frames` (fractions derived from controls + mapping, not stored).
- **Methods:** `run_rigid_dftb_scan`, `run_pm_neb` (relax / SP / charges-only pass).
- **Caveats:** `endpoints_relaxed` in meta only when both DFTB opts succeed; charge restart chains across frames.
- **Docs:** `doc/Topics/ReactionCoordinateScan.md`
"""
import os
import numpy as np

from spammm.topology.hbond_utils import controls_to_fractions, HbondRecord
from spammm.topology.scan_dataset import bond_lengths
from spammm.quantum.hbond_scan import make_hbond_transfer_path, DEFAULT_DS, HAU2EV

__all__ = [
    'DEFAULT_DS', 'HAU2EV', 'build_control_grid', 'position_h_at_fraction', 'build_frame',
    'build_pm_neb_endpoints', 'interpolate_all_atoms', 'run_rigid_dftb_scan', 'run_pm_neb',
    'run_pm_neb_sp', 'dataset_from_frames', 'make_hbond_transfer_path',
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


def _dftb_sp_charges(pt_dir, enames, apos_i, sk_prefix, prev_charges, on_fail, verbose, frame_label=''):
    from spammm.quantum.DFTB_utils import run_dftb_sp
    try:
        e_ha, q = run_dftb_sp(pt_dir, enames, apos_i, sk_prefix, maxscc=400, restart_charges_from=prev_charges, return_charges=True)
    except RuntimeError as exc:
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
