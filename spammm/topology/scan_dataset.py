"""scan_dataset.py — Compressed trajectory I/O for reaction-coordinate scans (.npz).

Stores static topology once plus time series: coordinates, controls, bond lengths, optional
Mulliken charges and precomputed ESP maps. Per-H-bond transfer fractions are **recomputed**
from `controls` + `meta.mapping` — storing them would duplicate data when m=1 symmetric scan.

- **Caveats:** `charges` / `esp_xy` optional; older npz files load without them.
- **Docs:** `doc/Topics/ReactionCoordinateScan.md`
"""
import json
import os
import numpy as np

from spammm import elements as el


def bond_lengths(apos, bonds):
    """Vectorized bond lengths for one frame or stack [nframes, natoms, 3]."""
    apos = np.asarray(apos, dtype=float)
    bonds = np.asarray(bonds, dtype=np.int32)
    if apos.ndim == 2:
        p0, p1 = apos[bonds[:, 0]], apos[bonds[:, 1]]
        return np.linalg.norm(p1 - p0, axis=1)
    p0, p1 = apos[:, bonds[:, 0]], apos[:, bonds[:, 1]]
    return np.linalg.norm(p1 - p0, axis=2)


def cc_bond_indices(etype, bonds):
    """Dense bond indices where both endpoints are carbon (Z=6)."""
    etype = np.asarray(etype, dtype=np.int32)
    bonds = np.asarray(bonds, dtype=np.int32)
    zc = el.ELEMENT_DICT['C'][0]
    mask = (etype[bonds[:, 0]] == zc) & (etype[bonds[:, 1]] == zc)
    return np.where(mask)[0].astype(np.int32)


class ScanDataset:
    __slots__ = ('etype', 'bonds', 'atom_ids', 'apos', 'controls', 'bond_len', 'energies_ev', 'pi_bo_cc', 'charges', 'esp_xy', 'meta')

    def __init__(self, etype, bonds, atom_ids, apos, controls, bond_len=None, energies_ev=None, pi_bo_cc=None, charges=None, esp_xy=None, meta=None):
        self.etype = np.asarray(etype, dtype=np.int32)
        self.bonds = np.asarray(bonds, dtype=np.int32)
        self.atom_ids = np.asarray(atom_ids, dtype=np.int64)
        self.apos = np.asarray(apos, dtype=np.float64)
        self.controls = np.asarray(controls, dtype=np.float64)
        if self.apos.ndim != 3:
            raise ValueError(f"apos must be [nframes, natoms, 3], got {self.apos.shape}")
        nframes, natoms, _ = self.apos.shape
        if len(self.etype) != natoms:
            raise ValueError("etype length mismatch")
        if self.controls.shape[0] != nframes:
            raise ValueError("controls nframes mismatch")
        self.bond_len = np.asarray(bond_len, dtype=np.float64) if bond_len is not None else bond_lengths(self.apos, self.bonds)
        self.energies_ev = np.asarray(energies_ev, dtype=np.float64) if energies_ev is not None else np.full(nframes, np.nan)
        self.pi_bo_cc = np.asarray(pi_bo_cc, dtype=np.float64) if pi_bo_cc is not None else None
        self.charges = np.asarray(charges, dtype=np.float64) if charges is not None else None
        if self.charges is not None and self.charges.shape != (nframes, natoms):
            raise ValueError(f"charges must be [nframes, natoms], got {self.charges.shape}")
        self.esp_xy = np.asarray(esp_xy, dtype=np.float64) if esp_xy is not None else None
        if self.esp_xy is not None and self.esp_xy.shape[0] != nframes:
            raise ValueError("esp_xy nframes mismatch")
        self.meta = dict(meta or {})

    @property
    def nframes(self):
        return self.apos.shape[0]

    @property
    def natoms(self):
        return self.apos.shape[1]

    @property
    def m(self):
        return self.controls.shape[1] if self.controls.ndim > 1 else 1

    def frame(self, i):
        return self.apos[int(i)].copy()

    def enames(self):
        from spammm.topology.PackedMolecule import _z_to_ename
        return np.array([_z_to_ename(int(z)) for z in self.etype], dtype=object)

    def recompute_bond_len(self):
        self.bond_len = bond_lengths(self.apos, self.bonds)
        return self.bond_len

    def save_npz(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        payload = dict(etype=self.etype, bonds=self.bonds, atom_ids=self.atom_ids, apos=self.apos, controls=self.controls, bond_len=self.bond_len, energies_ev=self.energies_ev, meta_json=json.dumps(self.meta, default=str))
        if self.pi_bo_cc is not None:
            payload['pi_bo_cc'] = self.pi_bo_cc
        if self.charges is not None:
            payload['charges'] = self.charges
        if self.esp_xy is not None:
            payload['esp_xy'] = self.esp_xy
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load_npz(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data['meta_json']))
        pi_bo_cc = data['pi_bo_cc'] if 'pi_bo_cc' in data.files else None
        charges = data['charges'] if 'charges' in data.files else None
        esp_xy = data['esp_xy'] if 'esp_xy' in data.files else None
        return cls(data['etype'], data['bonds'], data['atom_ids'], data['apos'], data['controls'], bond_len=data['bond_len'], energies_ev=data['energies_ev'], pi_bo_cc=pi_bo_cc, charges=charges, esp_xy=esp_xy, meta=meta)

    def export_xyz(self, path):
        from spammm.quantum.DFTB_utils import save_xyz_movie
        enames = self.enames()
        frames = []
        for i in range(self.nframes):
            meta_line = {'frame': i}
            for j in range(self.m):
                meta_line[f'u{j}'] = float(self.controls[i, j])
            e = self.energies_ev[i]
            if np.isfinite(e):
                meta_line['E'] = float(e)
            frames.append({'apos': self.apos[i], 'enames': list(enames), **meta_line})
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        keys = ['frame'] + [f'u{j}' for j in range(self.m)] + (['E'] if np.any(np.isfinite(self.energies_ev)) else [])
        save_xyz_movie(frames, path, key_order=keys)
        return path
