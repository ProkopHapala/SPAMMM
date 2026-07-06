"""hbond_utils.py — Bridging H-bond discovery on edited molecular graphs for RC scans.

Unlike ASCII `resolve_hbond_pairs` (marker-based), **`find_hbonds_graph`** uses geometry on
synced `backend.sys` so assemblies and imported graphs share one detection path with the
3D H-bond overlay. `HbondRecord` carries dense indices for frame building and stable labels.

- **Mapping:** `default_mapping(n_hbonds, m)` + `controls_to_fractions` link slider controls to per-H f ∈ [0,1].
- **Docs:** `doc/Topics/ReactionCoordinateScan.md`
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class HbondRecord:
    donor_idx: int
    h_idx: int
    acceptor_idx: int
    dist_ha: float = 0.0
    angle: float = 180.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def find_hbonds_sys(sys, d_max=2.5, a_min=150.0, bPrint=False):
    """Return HbondRecord list from AtomicSystem (calls find_hbonds)."""
    sys.neighs()
    raw = sys.find_hbonds(d_max=d_max, a_min=a_min, bPrint=bPrint)
    return [HbondRecord(d, h, a, dist, ang) for d, h, a, dist, ang in raw]


def find_hbonds_graph(backend, d_max=2.5, a_min=150.0, bPrint=False):
    """Return HbondRecord list from MoleculeEditorBackend (syncs graph → sys when graph populated)."""
    backend.ensure_sys()
    return find_hbonds_sys(backend.sys, d_max=d_max, a_min=a_min, bPrint=bPrint)


def default_mapping(n_hbonds, m=None):
    """Map each H-bond to a control index. Default: one shared control (m=1)."""
    if m is None:
        m = 1
    if m == 1:
        return [0] * n_hbonds
    if m == n_hbonds:
        return list(range(n_hbonds))
    raise ValueError(f"default_mapping: need m=1 or m=n_hbonds, got m={m} n_hbonds={n_hbonds}")


def controls_to_fractions(control_row, mapping):
    """Per-H-bond transfer fraction f ∈ [0,1] from control vector and mapping."""
    u = np.asarray(control_row, dtype=float).ravel()
    return np.array([float(u[mapping[i]]) for i in range(len(mapping))], dtype=float)
