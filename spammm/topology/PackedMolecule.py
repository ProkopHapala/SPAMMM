"""
PackedMolecule.py — Dense numpy representation of molecular topology.

Purpose: Compact, serializable snapshot of an AtomicGraph for undo history,
clipboard, and fast binary I/O (.npz). No strings — all element types are
atomic numbers (int32). No object references — bonds are index pairs.

Fields:
  etype : int32[natoms]        — atomic numbers (Z)
  apos  : float32[natoms, 3]   — positions in Angstrom
  bonds : int32[nbonds, 2]     — 0-based index pairs
  npi   : int8[natoms]         — pi-orbital count (-1=H_cap, 0=sp3, 1=sp2, 2=sp)

Memory: ~21 bytes/atom + 8 bytes/bond. 1000 atoms + 2000 bonds ≈ 37 KB.
"""

import numpy as np
from spammm import elements as el
from spammm.topology.AtomicGraph import AtomicGraph, Atom, Bond


def _z_to_ename(z):
    for e in el.ELEMENTS:
        if e[0] == z: return e[1]
    return 'X'

def _ename_to_z(ename):
    return el.ELEMENT_DICT[ename][0]


class PackedMolecule:
    __slots__ = ('etype', 'apos', 'bonds', 'npi')

    def __init__(self, etype, apos, bonds, npi):
        self.etype = np.asarray(etype, dtype=np.int32)
        self.apos  = np.asarray(apos,  dtype=np.float32)
        self.bonds = np.asarray(bonds, dtype=np.int32) if bonds is not None else np.zeros((0, 2), dtype=np.int32)
        self.npi   = np.asarray(npi,   dtype=np.int8)

    def __repr__(self):
        return f"PackedMolecule(natoms={len(self.etype)}, nbonds={len(self.bonds)})"

    @classmethod
    def from_graph(cls, graph, atom_indices=None):
        """Extract PackedMolecule from AtomicGraph.
        If atom_indices is given (list of dense indices from to_arrays()), only
        include those atoms and bonds between them (internal bonds only)."""
        atom_list, enames, apos, atypes, bonds_idx, bond_list, ring_list = graph.to_arrays()
        if atom_indices is not None:
            sel = set(atom_indices)
            idx_map = {old: new for new, old in enumerate(sorted(sel))}
            mask = np.array([(i in sel) for i in range(len(atom_list))], dtype=bool)
            etype = atypes[mask].astype(np.int32)
            pos   = apos[mask].astype(np.float32)
            npi   = np.array([atom_list[i].npi for i in range(len(atom_list)) if i in sel], dtype=np.int8)
            # Filter bonds: both endpoints must be in selection
            bond_pairs = []
            for col in bonds_idx:
                i, j = int(col[0]), int(col[1])
                if i in sel and j in sel:
                    bond_pairs.append((idx_map[i], idx_map[j]))
            bonds = np.array(bond_pairs, dtype=np.int32).reshape(-1, 2) if bond_pairs else np.zeros((0, 2), dtype=np.int32)
        else:
            etype = atypes.astype(np.int32)
            pos   = apos.astype(np.float32)
            npi   = np.array([a.npi for a in atom_list], dtype=np.int8)
            bonds = bonds_idx.astype(np.int32) if len(bonds_idx) else np.zeros((0, 2), dtype=np.int32)
        return cls(etype, pos, bonds, npi)

    def to_graph(self):
        """Rebuild a fresh AtomicGraph from packed data.
        Pins=None (caller can reassign). Parent reconstructed for H caps by distance.
        Rings re-detected by caller."""
        graph = AtomicGraph()
        atoms = []
        for i in range(len(self.etype)):
            z = int(self.etype[i])
            ename = _z_to_ename(z)
            atype = z
            npi_i = int(self.npi[i]) if i < len(self.npi) else 1
            pos = np.array(self.apos[i], dtype=np.float64)
            a = graph.add_atom(pos, ename, atype, pin=None, parent=None, npi=npi_i)
            atoms.append(a)
        # Bonds
        for col in self.bonds:
            i, j = int(col[0]), int(col[1])
            if 0 <= i < len(atoms) and 0 <= j < len(atoms):
                graph.add_bond(atoms[i], atoms[j])
        # Reconstruct parent for H caps: nearest heavy atom within 1.5 Å
        heavy = [a for a in atoms if a.ename not in ('H', 'E')]
        for a in atoms:
            if a.ename == 'H':
                best_d, best_a = float('inf'), None
                for ha in heavy:
                    d = float(np.linalg.norm(a.pos - ha.pos))
                    if d < best_d: best_d, best_a = d, ha
                if best_a is not None and best_d < 1.5:
                    a.parent = best_a
                    a.npi = -1
        graph.sync_neighbor_lists()
        return graph

    def save_npz(self, fname):
        np.savez(fname, etype=self.etype, apos=self.apos, bonds=self.bonds, npi=self.npi)

    @classmethod
    def load_npz(cls, fname):
        d = np.load(fname)
        return cls(d['etype'], d['apos'], d['bonds'], d['npi'])

    def to_xyz_text(self):
        lines = [str(len(self.etype)), "#SPAMMM clipboard"]
        for i in range(len(self.etype)):
            e = _z_to_ename(int(self.etype[i]))
            p = self.apos[i]
            lines.append(f"{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        return "\n".join(lines) + "\n"

    def to_mol_text(self):
        from spammm.atomicUtils import save_mol
        import io
        enames = [_z_to_ename(int(z)) for z in self.etype]
        # save_mol writes to file, so use StringIO
        buf = io.StringIO()
        n = len(self.etype)
        nb = len(self.bonds)
        buf.write("  SPAMMM\n\n")
        buf.write(f"{n:>3d}{nb:>3d}  0  0  0  0  0  0  0 0999 V2000\n")
        for i in range(n):
            x, y, z = self.apos[i]
            buf.write(f"{x:10.4f}{y:10.4f}{z:10.4f} {enames[i]:<3s}" + "  0"*12 + "\n")
        for i, (a1, a2) in enumerate(self.bonds):
            buf.write(f"{int(a1)+1:>3d}{int(a2)+1:>3d}   1  0  0  0  0\n")
        buf.write("M  END\n")
        return buf.getvalue()

    @classmethod
    def from_text(cls, text):
        """Parse .xyz or .mol text → PackedMolecule. Auto-detect format."""
        lines = text.strip().split('\n')
        if not lines: return None
        # Try XYZ: first line is integer (atom count)
        try:
            n = int(lines[0].strip())
            if n > 0 and n < 100000:
                return cls._parse_xyz(lines, n)
        except ValueError:
            pass
        # Try MOL: look for "V2000" in first 5 lines
        for line in lines[:5]:
            if 'V2000' in line:
                return cls._parse_mol(lines)
        return None

    @classmethod
    def _parse_xyz(cls, lines, n):
        etypes = []
        apos = []
        for line in lines[2:2+n]:
            wds = line.split()
            if len(wds) < 4: continue
            try:
                ename = wds[0]
                x, y, z = float(wds[1]), float(wds[2]), float(wds[3])
                z_num = int(wds[0]) if wds[0].isdigit() else _ename_to_z(ename)
                etypes.append(z_num)
                apos.append([x, y, z])
            except (ValueError, KeyError):
                continue
        return cls(
            np.array(etypes, dtype=np.int32),
            np.array(apos, dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
            np.ones(len(etypes), dtype=np.int8),
        )

    @classmethod
    def _parse_mol(cls, lines):
        # Find counts line (contains V2000)
        counts_idx = None
        for i, line in enumerate(lines):
            if 'V2000' in line:
                counts_idx = i
                break
        if counts_idx is None: return None
        wds = lines[counts_idx].split()
        n_atoms = int(wds[0])
        n_bonds = int(wds[1])
        # Skip blank lines, collect non-blank lines after counts
        data_lines = []
        for line in lines[counts_idx + 1:]:
            if line.strip():
                data_lines.append(line)
            if len(data_lines) >= n_atoms + n_bonds:
                break
        etypes = []
        apos = []
        for i in range(n_atoms):
            line = data_lines[i]
            wds = line.split()
            if len(wds) < 4: continue
            # Standard MOL V2000: x y z symbol ...
            # Some writers add atom_id prefix: atom_id x y z symbol ...
            try:
                float(wds[0]); float(wds[1]); float(wds[2])
                symbol = wds[3]; x, y, z = float(wds[0]), float(wds[1]), float(wds[2])
            except ValueError:
                symbol = wds[4]; x, y, z = float(wds[1]), float(wds[2]), float(wds[3])
            etypes.append(_ename_to_z(symbol))
            apos.append([x, y, z])
        bonds = []
        for i in range(n_bonds):
            line = data_lines[n_atoms + i]
            wds = line.split()
            if len(wds) < 3: continue
            # Standard MOL: a1 a2 order ... (1-based)
            try:
                a1, a2 = int(wds[0]) - 1, int(wds[1]) - 1
            except ValueError:
                continue
            bonds.append([a1, a2])
        return cls(
            np.array(etypes, dtype=np.int32),
            np.array(apos, dtype=np.float32),
            np.array(bonds, dtype=np.int32).reshape(-1, 2) if bonds else np.zeros((0, 2), dtype=np.int32),
            np.ones(len(etypes), dtype=np.int8),
        )


class EditorSnapshot:
    """Lossless undo snapshot of an AtomicGraph + editor backend state.

    Unlike PackedMolecule (which is lossy: float32, no pins/parents/charges/
    bond-orders/IDs), EditorSnapshot preserves every authoritative graph field
    needed for exact restoration:
      - Atom: _id, ename, atype, pos (float64), pin, parent (by _id), npi, charge
      - Bond: _id, endpoint _ids, order
      - backend.hex_tiles: set of (q,r) tuples

    Only alive atoms/bonds are serialized. On restore, a fresh AtomicGraph is
    built with exact IDs, pins, parents, charges, and bond orders. The pin cache
    is rebuilt from stored pins (with duplicate-pin assertion). hex_tiles is
    returned to the caller for exact restoration.

    H caps are NOT regenerated during restore — they are restored exactly as
    they were in the snapshot, preserving parent links and positions.
    """
    __slots__ = ('atom_ids', 'etypes', 'apos', 'pins', 'parent_ids', 'npis', 'charges',
                 'bond_ids', 'bond_a_ids', 'bond_b_ids', 'bond_orders', 'hex_tiles')

    def __init__(self, atom_ids, etypes, apos, pins, parent_ids, npis, charges,
                 bond_ids, bond_a_ids, bond_b_ids, bond_orders, hex_tiles):
        self.atom_ids    = atom_ids
        self.etypes      = etypes
        self.apos        = apos
        self.pins        = pins
        self.parent_ids  = parent_ids
        self.npis        = npis
        self.charges     = charges
        self.bond_ids    = bond_ids
        self.bond_a_ids  = bond_a_ids
        self.bond_b_ids  = bond_b_ids
        self.bond_orders = bond_orders
        self.hex_tiles   = hex_tiles

    def __repr__(self):
        return f"EditorSnapshot(natoms={len(self.atom_ids)}, nbonds={len(self.bond_ids)}, ntiles={len(self.hex_tiles)})"

    @classmethod
    def from_graph(cls, graph, hex_tiles):
        """Build a lossless snapshot from an AtomicGraph + backend.hex_tiles."""
        atom_list, enames, apos, atypes, bonds_idx, bond_list, ring_list = graph.to_arrays()
        n = len(atom_list)
        atom_ids   = [a._id for a in atom_list]
        etypes     = [a.atype for a in atom_list]
        apos_arr   = np.array([a.pos for a in atom_list], dtype=np.float64)
        pins       = [a.pin for a in atom_list]
        parent_ids = [a.parent._id if a.parent is not None else -1 for a in atom_list]
        npis       = [a.npi for a in atom_list]
        charges    = [a.charge for a in atom_list]
        id_to_idx  = {a._id: i for i, a in enumerate(atom_list)}
        bond_ids    = [b._id for b in bond_list]
        bond_a_ids  = [id_to_idx[b.a._id] for b in bond_list]
        bond_b_ids  = [id_to_idx[b.b._id] for b in bond_list]
        bond_orders = [b.order for b in bond_list]
        return cls(atom_ids, etypes, apos_arr, pins, parent_ids, npis, charges,
                   bond_ids, bond_a_ids, bond_b_ids, bond_orders, set(hex_tiles))

    def to_graph(self):
        """Rebuild a fresh AtomicGraph with exact IDs, pins, parents, charges, bond orders.
        Returns (graph, hex_tiles_copy)."""
        graph = AtomicGraph()
        atoms = [None] * len(self.atom_ids)
        for i in range(len(self.atom_ids)):
            z = int(self.etypes[i])
            ename = _z_to_ename(z)
            pos = np.array(self.apos[i], dtype=np.float64)
            pin = self.pins[i]
            npi_i = int(self.npis[i])
            a = graph.add_atom(pos, ename, z, pin=pin, parent=None, npi=npi_i, _id=int(self.atom_ids[i]))
            a.charge = float(self.charges[i])
            atoms[i] = a
        # Restore parent links by stored parent _id
        id_to_atom = {a._id: a for a in atoms}
        for i in range(len(atoms)):
            pid = int(self.parent_ids[i])
            if pid >= 0 and pid in id_to_atom:
                atoms[i].parent = id_to_atom[pid]
        # Restore bonds with exact IDs and orders
        for k in range(len(self.bond_ids)):
            ia = int(self.bond_a_ids[k])
            ib = int(self.bond_b_ids[k])
            if 0 <= ia < len(atoms) and 0 <= ib < len(atoms):
                graph.add_bond(atoms[ia], atoms[ib], order=float(self.bond_orders[k]), _id=int(self.bond_ids[k]))
        graph.sync_neighbor_lists()
        graph.rebuild_pin_cache_from_atoms()
        return graph, set(self.hex_tiles)


class UndoStack:
    """Rolling buffer of PackedMolecule snapshots. O(1) push/pop."""
    def __init__(self, maxlen=100):
        from collections import deque
        self._stack = deque(maxlen=maxlen)
        self.enabled = True

    def push(self, packed):
        if not self.enabled: return
        self._stack.append(packed)

    def pop(self):
        if not self._stack: return None
        return self._stack.pop()

    def clear(self):
        self._stack.clear()

    def __len__(self):
        return len(self._stack)
