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
    # PBC: integer cell shifts of donor/acceptor partners along the chain
    # lattice vector (image position = apos[idx] + shift*lvec). 0 = in-cell.
    d_shift: int = 0
    a_shift: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


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


def hbond_positions(apos, hb, lvs=None):
    """(pD, pH, pA) with periodic image shifts applied (pX = apos[idx] + shift*lvec)."""
    apos = np.asarray(apos, dtype=float)
    lvec = np.zeros(3) if lvs is None else np.asarray(lvs)[1]
    return apos[hb.donor_idx] + hb.d_shift * lvec, apos[hb.h_idx], apos[hb.acceptor_idx] + hb.a_shift * lvec


def junction_bond_lengths(apos, hbonds, lvs=None):
    """Per-junction (D–H, H···A) distances [Å] for each HbondRecord — the state-defining
    numbers of a proton-transfer corner (2 junctions → 4 lengths)."""
    out = []
    for h in hbonds:
        pD, pH, pA = hbond_positions(apos, h, lvs)
        out.append((float(np.linalg.norm(pH - pD)), float(np.linalg.norm(pH - pA))))
    return out
