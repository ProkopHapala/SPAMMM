"""
AtomicGraph.py — Object-graph representation of an atomic structure.

Purpose: Provide an editable molecular topology where Atom, Bond, and Ring objects
have stable Python identities. Deleting an atom does not renumber others — crucial
for interactive GUI editing where indices must remain valid across operations.

Key functionality:
  - Atom, Bond, Ring objects with stable IDs and bidirectional references
  - Graph traversal: neighbors, rings, paths
  - to_arrays() — export to NumPy arrays for force field evaluation and rendering
  - from_arrays() — import from NumPy arrays

Role in SPAMMM: Canonical molecular representation for the editor (MoleculeEditorBackend).
AtomicGraph is the "authoritative state" that SPAMMM_GUI edits directly.
It is periodically synced to AtomicSystem for rendering and force field computation.

Design principles:
  - Atom, Bond, Ring are plain Python objects with stable identity (not integer indices).
  - Deletion of any object does NOT renumber or invalidate any other object.
  - Integer indices for interop with numpy/vispy are generated on demand via to_arrays().
  - No parallel arrays that must be kept in sync; all per-atom data lives on the Atom object.

Public API:
  graph = AtomicGraph()
  a = graph.add_atom(pos, ename, pin=None, parent=None, npi=1)  # npi: -1=H_cap, 0=sp3, 1=sp2, 2=sp
  graph.remove_atom(a)          # removes a and all its bonds; caller handles rings
  b = graph.add_bond(a1, a2)
  graph.remove_bond(b)
  r = graph.add_ring(q, r_coord, atoms)
  graph.remove_ring(r)
  atoms, enames, apos, bonds = graph.to_arrays()   # for numpy/vispy rendering
"""

import numpy as np

DEBUG = True  # Set True to enable topology consistency checks

# ─── Atom ───────────────────────────────────────────────────────────────────

class Atom:
    __slots__ = ('pos', 'ename', 'atype', 'pin', 'parent', 'npi', 'bonds', 'neighbors', '_id', 'alive', 'charge')
    _counter = 0

    def __init__(self, pos, ename, atype, pin=None, parent=None, npi=1):
        Atom._counter += 1
        self._id = Atom._counter
        self.alive   = True         # False = marked for deletion, will be cleaned up
        self.pos     = np.asarray(pos, dtype=np.float64)   # (3,)
        self.ename   = ename        # element symbol string
        self.atype   = atype        # integer Z
        self.pin     = pin          # (rx, ry) grid node key or None
        self.parent  = parent       # Atom object (heavy atom this H belongs to) or None
        # npi: pi-orbital count. -1=H_cap, 0=sp3, 1=sp2 (default), 2=sp
        self.npi     = npi
        self.bonds   = []           # list of Bond objects involving this atom
        self.neighbors = []         # list of neighboring Atoms (derived from bonds)
        self.charge  = 0.0          # partial charge (e.g. from QEq), in electrons

    def __repr__(self):
        status = "" if self.alive else "[DEAD]"
        return f"Atom({self._id}{status} {self.ename} pin={self.pin} pos={self.pos[:2]})"


# ─── Bond ───────────────────────────────────────────────────────────────────

class Bond:
    __slots__ = ('a', 'b', 'order', '_id', 'alive')
    _counter = 0

    def __init__(self, a: Atom, b: Atom, order=1.0):
        Bond._counter += 1
        self._id   = Bond._counter
        self.alive = True           # False = marked for deletion, will be cleaned up
        self.a     = a
        self.b     = b
        self.order = order

    def other(self, atom: Atom) -> Atom:
        return self.b if atom is self.a else self.a

    def __repr__(self):
        status = "" if self.alive else "[DEAD]"
        return f"Bond({self._id}{status} {self.a._id}-{self.b._id} o={self.order})"


# ─── Ring ───────────────────────────────────────────────────────────────────

class Ring:
    __slots__ = ('atoms', 'bonds', 'cog', '_id', 'alive')
    _counter = 0

    def __init__(self, atoms, bonds):
        """Ring as real geometry cycle (n-gon).
        Args:
            atoms: list[Atom] - ordered list of atoms in the cycle
            bonds: list[Bond] - ordered list of bonds in the cycle
        """
        Ring._counter += 1
        self._id   = Ring._counter
        self.alive = True           # False = marked for deletion, will be cleaned up
        self.atoms = list(atoms)    # [Atom, ...] — ordered cyclically
        self.bonds = list(bonds)    # [Bond, ...] — ordered cyclically
        self.cog   = self._compute_cog()

    def _compute_cog(self):
        """Compute center of geometry as average of atom positions."""
        # Only count alive atoms
        alive_atoms = [a for a in self.atoms if a.alive]
        if not alive_atoms:
            return np.zeros(3)
        positions = np.array([a.pos for a in alive_atoms])
        return np.mean(positions, axis=0)

    def __repr__(self):
        status = "" if self.alive else "[DEAD]"
        return f"Ring({self._id}{status} natoms={len(self.atoms)})"


# ─── AtomicGraph ────────────────────────────────────────────────────────────

class AtomicGraph:
    """Mutable graph of atoms, bonds, and rings.
    All collections are dicts keyed by object id for O(1) lookup and deletion.
    """

    def __init__(self):
        self.atoms  = {}    # id -> Atom
        self.bonds  = {}    # id -> Bond
        self.rings  = {}    # id -> Ring
        self._pin_to_atom = {}   # (rx,ry) -> Atom  — rebuildable cache, not primary index

    # ── Pin cache management ─────────────────────────────────────────────────

    def invalidate_pin_cache(self):
        """Clear all pins and the pin→atom cache. Call when grid transform changes."""
        for a in self.atoms.values():
            if a.alive:
                a.pin = None
        self._pin_to_atom.clear()

    def rebuild_pin_cache(self, pin_map):
        """Rebuild pin cache from a {Atom: pin_key} mapping. Overwrites all pins.
        Args:
            pin_map: dict {Atom: (rx, ry)} or None entries (None = no pin for that atom)
        """
        self._pin_to_atom.clear()
        for a, pin in pin_map.items():
            a.pin = pin
            if pin is not None:
                self._pin_to_atom[pin] = a

    def ensure_pin_cache(self):
        """No-op: cache is maintained in real-time by add_atom/remove_atom.
        Kept for API symmetry with future lazy-cache designs."""
        pass

    # ── Atom operations ──────────────────────────────────────────────────────

    def add_atom(self, pos, ename, atype, pin=None, parent=None, npi=1) -> Atom:
        a = Atom(pos, ename, atype, pin=pin, parent=parent, npi=npi)
        self.atoms[a._id] = a
        if pin is not None:
            assert pin not in self._pin_to_atom, f"Duplicate pin {pin} (existing={self._pin_to_atom[pin]}, new={a})"
            self._pin_to_atom[pin] = a
        return a

    def remove_atom(self, atom: Atom, soft=True):
        """Remove atom. If soft=True, mark as dead and cleanup later.
        If soft=False, immediate hard removal."""
        if atom._id not in self.atoms:
            return
        if soft:
            # Soft deletion: mark as dead, will be cleaned up later
            atom.alive = False
            # Also mark all its bonds as dead
            for b in atom.bonds:
                b.alive = False
            # Remove from pin cache so dead atoms don't block re-adding at same grid node
            if atom.pin is not None:
                self._pin_to_atom.pop(atom.pin, None)
                atom.pin = None
        else:
            # Hard deletion: immediate removal
            for b in list(atom.bonds):
                self._remove_bond_internal(b, hard=True)
            if atom.pin is not None:
                self._pin_to_atom.pop(atom.pin, None)
            del self.atoms[atom._id]

    def cleanup_invalid(self):
        """Remove all dead (alive=False) atoms, bonds, and rings.
        Clean up references from other objects."""
        # First, remove dead bonds from atom bond lists
        for atom in self.atoms.values():
            if atom.alive:
                atom.bonds = [b for b in atom.bonds if b.alive]
        
        # Remove dead rings
        dead_ring_ids = [rid for rid, r in self.rings.items() if not r.alive]
        for rid in dead_ring_ids:
            del self.rings[rid]
        
        # Remove dead bonds from main bonds dict
        dead_bond_ids = [bid for bid, b in self.bonds.items() if not b.alive]
        for bid in dead_bond_ids:
            del self.bonds[bid]
        
        # Remove dead atoms and update pin mapping
        dead_atom_ids = [aid for aid, a in self.atoms.items() if not a.alive]
        for aid in dead_atom_ids:
            atom = self.atoms[aid]
            if atom.pin is not None:
                self._pin_to_atom.pop(atom.pin, None)
            del self.atoms[aid]
        
        return len(dead_atom_ids), len(dead_bond_ids), len(dead_ring_ids)

    def sync_neighbor_lists(self):
        """Rebuild neighbor lists from alive bonds.
        Call this after any bond topology change."""
        # Clear all neighbor lists
        for atom in self.atoms.values():
            if atom.alive:
                atom.neighbors = []
        # Rebuild from alive bonds
        for bond in self.bonds.values():
            if bond.alive:
                bond.a.neighbors.append(bond.b)
                bond.b.neighbors.append(bond.a)

    def h_children(self, heavy_atom: Atom) -> list:
        """Return list of H cap atoms (npi=-1) that have parent=heavy_atom."""
        return [a for a in self.atoms.values() 
                if a.alive and a.npi == -1 and a.parent is heavy_atom]

    def atom_at_pin(self, pin) -> 'Atom | None':
        return self._pin_to_atom.get(pin)

    # ── Bond operations ──────────────────────────────────────────────────────

    def add_bond(self, a: Atom, b: Atom, order=None) -> Bond:
        for bond in a.bonds:
            if bond.other(a) is b:
                if not bond.alive:
                    bond.alive = True  # revive dead bond
                    a.neighbors.append(b)
                    b.neighbors.append(a)
                return bond   # already exists (now alive)
        if order is None:
            order = 1.5 if (a.npi > 0 and b.npi > 0) else 1.0
        bond = Bond(a, b, order)
        self.bonds[bond._id] = bond
        a.bonds.append(bond)
        b.bonds.append(bond)
        a.neighbors.append(b)
        b.neighbors.append(a)
        return bond

    def remove_bond(self, bond: Bond):
        self._remove_bond_internal(bond)

    def _remove_bond_internal(self, bond: Bond, hard=False):
        if bond._id not in self.bonds:
            return
        # Validate neighbor consistency before removal (debug only)
        if DEBUG and (bond.b not in bond.a.neighbors or bond.a not in bond.b.neighbors):
            raise RuntimeError(f"remove_bond: inconsistent topology — bond {bond._id} atoms not in each other's neighbor lists")
        if hard:
            bond.a.bonds = [b for b in bond.a.bonds if b is not bond]
            bond.b.bonds = [b for b in bond.b.bonds if b is not bond]
            bond.a.neighbors.remove(bond.b)
            bond.b.neighbors.remove(bond.a)
            del self.bonds[bond._id]
        else:
            bond.alive = False
            bond.a.neighbors.remove(bond.b)
            bond.b.neighbors.remove(bond.a)

    def get_bond(self, a: Atom, b: Atom) -> 'Bond | None':
        for bond in a.bonds:
            if bond.other(a) is b:
                return bond
        return None

    # ── Ring operations ──────────────────────────────────────────────────────

    def add_ring(self, atoms, bonds) -> Ring:
        """Add a geometry-based ring (n-gon cycle).
        Args:
            atoms: list[Atom] - ordered list of atoms in the cycle
            bonds: list[Bond] - ordered list of bonds in the cycle
        """
        ring = Ring(atoms, bonds)
        self.rings[ring._id] = ring
        return ring

    def remove_ring(self, ring: Ring):
        self.rings.pop(ring._id, None)

    def detect_rings(self, max_ring_size=8):
        """Detect all rings (cycles) in the bond graph using DFS.
        Returns list of Ring objects.
        """
        # Build adjacency list (only for alive atoms with alive bonds to alive atoms)
        adj = {a._id: [b.other(a) for b in a.bonds if b.alive and b.other(a).alive] for a in self.atoms.values() if a.alive}
        visited = set()
        rings = []

        def dfs(start, current, path_atoms, path_bonds, visited_edges):
            if len(path_atoms) > max_ring_size:
                return
            if current._id in visited:
                return
            for neighbor in adj.get(current._id, []):
                if neighbor._id not in adj:
                    continue  # Skip dead/removed neighbors
                edge = self.get_bond(current, neighbor)
                if edge is None:
                    continue
                edge_key = frozenset((current._id, neighbor._id))
                if edge_key in visited_edges:
                    continue
                if neighbor._id == start._id and len(path_atoms) >= 3:
                    # Found a cycle
                    rings.append(self.add_ring(path_atoms + [neighbor], path_bonds + [edge]))
                    continue
                if neighbor._id not in [a._id for a in path_atoms]:
                    dfs(start, neighbor, path_atoms + [neighbor], path_bonds + [edge],
                        visited_edges | {edge_key})

        for atom in self.atoms.values():
            if atom._id in visited or not atom.alive:
                continue
            dfs(atom, atom, [], [], set())

        return rings

    # ── Bulk bond rebuild ─────────────────────────────────────────────────────

    def recalc_bonds(self, bond_length=1.42, tol_factor=0.35):
        """Remove all bonds and recompute from distance threshold."""
        # Use hard delete to immediately remove bonds from atom bond lists
        for bond in list(self.bonds.values()):
            self._remove_bond_internal(bond, hard=True)
        # Only consider alive atoms
        atoms = [a for a in self.atoms.values() if a.alive]
        threshold = bond_length * (1.0 + tol_factor)
        threshold_sq = threshold ** 2
        for i, a in enumerate(atoms):
            for j in range(i + 1, len(atoms)):
                b = atoms[j]
                d2 = float(np.sum((a.pos - b.pos) ** 2))
                if d2 < threshold_sq:
                    self.add_bond(a, b)

    # ── Export for numpy/vispy ─────────────────────────────────────────────────

    def to_arrays(self):
        """Return (atom_list, enames, apos, atypes, bonds_idx, bond_list, ring_list) for rendering.
        atom_list[i] is the Atom object at index i.
        bonds_idx is (N,2) int array of indices into atom_list.
        bond_list[i] is the Bond object at index i (parallel to bonds_idx).
        ring_list is list of Ring objects.
        Index assignment is stable within one call; call again after mutations.
        Only alive objects are included.
        """
        # Only include alive atoms
        atom_list = [a for a in self.atoms.values() if a.alive]
        idx = {a._id: i for i, a in enumerate(atom_list)}
        enames = np.array([a.ename for a in atom_list], dtype=object)
        apos   = np.array([a.pos   for a in atom_list], dtype=np.float64)
        atypes = np.array([a.atype for a in atom_list], dtype=np.int32)
        
        # Only include alive bonds between alive atoms
        bond_pairs = []
        bond_list = []
        for bond in self.bonds.values():
            if not bond.alive:
                continue
            # Both atoms must be alive
            if not bond.a.alive or not bond.b.alive:
                continue
            ia = idx.get(bond.a._id)
            ib = idx.get(bond.b._id)
            if ia is not None and ib is not None:
                bond_pairs.append((ia, ib))
                bond_list.append(bond)
        bonds = np.array(bond_pairs, dtype=np.int32).reshape(-1, 2) if bond_pairs else np.zeros((0, 2), dtype=np.int32)
        
        # Only include alive rings
        ring_list = [r for r in self.rings.values() if r.alive]
        return atom_list, enames, apos, atypes, bonds, bond_list, ring_list

    # ── Position update ───────────────────────────────────────────────────────

    def update_positions_from_array(self, apos):
        """Update atom positions from array (same order as to_arrays()).
        
        Args:
            apos: (N,3) array of positions, where N matches len(atoms) and order matches to_arrays()
        
        This updates geometry only (atom positions), not topology (bonds, rings).
        Used after external geometry relaxation (e.g., DFTB) to sync relaxed positions back to graph.
        """
        atom_list = [a for a in self.atoms.values() if a.alive]
        if len(atom_list) != len(apos):
            raise ValueError(f"Position array length {len(apos)} does not match number of alive atoms {len(atom_list)}")
        for i, atom in enumerate(atom_list):
            atom.pos[:] = apos[i]

    # ── Convenience queries ───────────────────────────────────────────────────

    def heavy_atoms(self):
        return [a for a in self.atoms.values() if a.alive and a.ename not in ('H', 'E')]

    def h_children(self, atom: Atom):
        return [a for a in self.atoms.values() if a.alive and a.parent is atom]

    def neighbors(self, atom: Atom):
        return [b.other(atom) for b in atom.bonds if b.alive]

    # ── Graph analysis: fragmentation ─────────────────────────────────────────

    def find_connected_components(self):
        """Return list of lists of alive Atom objects, one per connected component.
        Components are found by BFS over alive bonds."""
        alive = [a for a in self.atoms.values() if a.alive]
        visited = set()
        components = []
        for atom in alive:
            if atom._id in visited:
                continue
            comp = []
            stack = [atom]
            visited.add(atom._id)
            while stack:
                a = stack.pop()
                comp.append(a)
                for b in a.bonds:
                    if not b.alive:
                        continue
                    nb = b.other(a)
                    if nb.alive and nb._id not in visited:
                        visited.add(nb._id)
                        stack.append(nb)
            components.append(comp)
        return components

    def find_bridges(self, heavy_only=True):
        """Return list of alive Bond objects that are bridges.
        A bridge is a bond whose removal disconnects the graph.
        If heavy_only=True (default), only consider bonds between non-H atoms —
        C-H bonds are never bridges.
        Uses Tarjan's bridge-finding algorithm (DFS with disc/low times)."""
        if heavy_only:
            alive = self.heavy_atoms()
            def is_valid_edge(b, u):
                return b.alive and b.other(u).alive and b.other(u).ename != 'H'
        else:
            alive = [a for a in self.atoms.values() if a.alive]
            def is_valid_edge(b, u):
                return b.alive and b.other(u).alive
        disc, low = {}, {}
        timer = [0]
        visited = set()
        bridges = []

        def dfs(u, parent_id):
            visited.add(u._id)
            disc[u._id] = low[u._id] = timer[0]; timer[0] += 1
            for b in u.bonds:
                if not is_valid_edge(b, u): continue
                v = b.other(u)
                if v._id == parent_id: continue
                if v._id in visited:
                    low[u._id] = min(low[u._id], disc[v._id])
                else:
                    dfs(v, u._id)
                    low[u._id] = min(low[u._id], low[v._id])
                    if low[v._id] > disc[u._id]:
                        bridges.append(b)

        for atom in alive:
            if atom._id not in visited:
                dfs(atom, -1)
        return bridges

    def find_articulation_points(self):
        """Return list of alive Atom objects whose removal disconnects the graph.
        Uses Tarjan's articulation point algorithm."""
        alive = [a for a in self.atoms.values() if a.alive]
        disc, low = {}, {}
        timer = [0]
        visited = set()
        aps = []

        def dfs(u, parent_id, is_root):
            visited.add(u._id)
            disc[u._id] = low[u._id] = timer[0]; timer[0] += 1
            children = 0
            for b in u.bonds:
                if not b.alive: continue
                v = b.other(u)
                if not v.alive: continue
                if v._id == parent_id: continue
                if v._id in visited:
                    low[u._id] = min(low[u._id], disc[v._id])
                else:
                    children += 1
                    dfs(v, u._id, False)
                    low[u._id] = min(low[u._id], low[v._id])
                    if not is_root and low[v._id] >= disc[u._id]:
                        aps.append(u)
            if is_root and children > 1:
                aps.append(u)

        for atom in alive:
            if atom._id not in visited:
                dfs(atom, -1, True)
        return aps

    def find_local_bridges(self, max_dist=3):
        """Return list of alive Bond objects that are local bridges of order max_dist.
        A local bridge is a bond (u,v) where removing it makes the shortest
        path between u and v > max_dist. Unlike global bridges, the graph may
        remain connected through longer paths.
        For max_dist=2: bonds where endpoints share no common neighbor (not in any triangle).
        For max_dist=3: bonds where no alternate path of length <= 3 exists."""
        alive_bonds = [b for b in self.bonds.values() if b.alive and b.a.alive and b.b.alive]
        local_bridges = []
        for bond in alive_bonds:
            u, v = bond.a, bond.b
            depth = {u._id: 0}
            queue = [u]
            found = False
            while queue and not found:
                a = queue.pop(0)
                d = depth[a._id]
                if d >= max_dist: continue
                for b in a.bonds:
                    if not b.alive: continue
                    nb = b.other(a)
                    if not nb.alive: continue
                    if b is bond: continue
                    if nb._id not in depth:
                        depth[nb._id] = d + 1
                        if nb._id == v._id:
                            found = True
                            break
                        queue.append(nb)
            if not found:
                local_bridges.append(bond)
        return local_bridges

    def find_biconnected_components(self, heavy_only=True):
        """Return list of (atoms, bonds) tuples, one per biconnected component (block).
        A block is either a maximal 2-connected subgraph (ring system) or a single
        bridge bond. If heavy_only=True (default), only consider non-H atoms.
        Uses DFS with edge stack."""
        if heavy_only:
            alive = self.heavy_atoms()
            def is_valid_edge(b, u):
                return b.alive and b.other(u).alive and b.other(u).ename != 'H'
        else:
            alive = [a for a in self.atoms.values() if a.alive]
            def is_valid_edge(b, u):
                return b.alive and b.other(u).alive
        disc, low = {}, {}
        timer = [0]
        visited = set()
        edge_stack = []
        blocks = []

        def dfs(u, parent_id):
            visited.add(u._id)
            disc[u._id] = low[u._id] = timer[0]; timer[0] += 1
            children = 0
            for b in u.bonds:
                if not is_valid_edge(b, u): continue
                v = b.other(u)
                if v._id == parent_id: continue
                if v._id not in visited:
                    children += 1
                    edge_stack.append(b)
                    dfs(v, u._id)
                    low[u._id] = min(low[u._id], low[v._id])
                    if (parent_id == -1 and children > 1) or (parent_id != -1 and low[v._id] >= disc[u._id]):
                        block_bonds = []
                        while True:
                            eb = edge_stack.pop()
                            block_bonds.append(eb)
                            if eb is b: break
                        block_atoms = set()
                        for bb in block_bonds:
                            block_atoms.add(bb.a)
                            block_atoms.add(bb.b)
                        blocks.append((list(block_atoms), block_bonds))
                elif disc[v._id] < disc[u._id]:
                    edge_stack.append(b)
                    low[u._id] = min(low[u._id], disc[v._id])

        for atom in alive:
            if atom._id not in visited:
                dfs(atom, -1)
                if edge_stack:
                    block_bonds = list(edge_stack)
                    edge_stack.clear()
                    block_atoms = set()
                    for bb in block_bonds:
                        block_atoms.add(bb.a)
                        block_atoms.add(bb.b)
                    blocks.append((list(block_atoms), block_bonds))
        return blocks

    def find_fragments(self, min_size=2):
        """Split molecule into fragments by cutting heavy-atom bridges.

        Returns (fragments, cut_bridges) where:
          fragments   = list of lists of Atom objects (including H atoms assigned
                        to their parent heavy atom's fragment)
          cut_bridges = list of Bond objects that separate the fragments

        Fragments with fewer than min_size heavy atoms are merged back into the
        largest adjacent fragment (their bridge is restored and not counted as a cut).
        """
        heavy = self.heavy_atoms()
        heavy_ids = {a._id for a in heavy}
        bridges = self.find_bridges(heavy_only=True)
        bridge_set = {frozenset((b.a._id, b.b._id)) for b in bridges}

        # Connected components on heavy atoms, excluding bridge bonds
        visited = set()
        comps = []
        for atom in heavy:
            if atom._id in visited: continue
            comp = []
            stack = [atom]
            visited.add(atom._id)
            while stack:
                a = stack.pop()
                comp.append(a)
                for b in a.bonds:
                    if not b.alive: continue
                    nb = b.other(a)
                    if nb._id not in heavy_ids or nb._id in visited: continue
                    if frozenset((a._id, nb._id)) in bridge_set: continue  # skip bridge
                    visited.add(nb._id)
                    stack.append(nb)
            comps.append(comp)

        # Build adjacency via bridges: comp_idx -> list of (other_comp_idx, bridge_bond)
        comp_of = {}
        for i, comp in enumerate(comps):
            for a in comp:
                comp_of[a._id] = i
        bridge_adj = [[] for _ in range(len(comps))]
        for b in bridges:
            i, j = comp_of[b.a._id], comp_of[b.b._id]
            bridge_adj[i].append((j, b))
            bridge_adj[j].append((i, b))

        # Merge fragments smaller than min_size into largest neighbor
        merged = [False] * len(comps)
        merge_map = list(range(len(comps)))  # comp_idx -> representative comp_idx

        def find(x):
            while merge_map[x] != x:
                merge_map[x] = merge_map[merge_map[x]]
                x = merge_map[x]
            return x

        # Process smallest first
        order = sorted(range(len(comps)), key=lambda i: len(comps[i]))
        for i in order:
            if len(comps[i]) >= min_size: continue
            # Find largest adjacent component
            best_j, best_size = None, 0
            for j, b in bridge_adj[i]:
                rj = find(j)
                if rj == find(i): continue  # already same
                if len(comps[rj]) > best_size:
                    best_size = len(comps[rj])
                    best_j = rj
            if best_j is not None:
                merge_map[find(i)] = best_j

        # Collect final fragments
        final_comps = {}
        for i in range(len(comps)):
            r = find(i)
            final_comps.setdefault(r, []).extend(comps[i])

        # Determine which bridges are actual cuts (between different final fragments)
        cut_bridges = []
        for b in bridges:
            if comp_of.get(b.a._id) is not None and comp_of.get(b.b._id) is not None:
                if find(comp_of[b.a._id]) != find(comp_of[b.b._id]):
                    cut_bridges.append(b)

        # Assign H atoms to their parent's fragment
        fragments = []
        for r, heavy_atoms in final_comps.items():
            frag = list(heavy_atoms)
            for a in heavy_atoms:
                for h in self.h_children(a):
                    frag.append(h)
            fragments.append(frag)

        return fragments, cut_bridges

    # ── Picking helpers ────────────────────────────────────────────────────────

    def pick_atom(self, pos, radius=0.5):
        """Find atom within radius of position. Returns Atom or None."""
        for atom in self.atoms.values():
            if atom.alive and np.linalg.norm(atom.pos - pos) < radius:
                return atom
        return None

    def pick_bond(self, pos, radius=0.5):
        """Find bond whose center is within radius of position. Returns Bond or None."""
        for bond in self.bonds.values():
            if not bond.alive:
                continue
            # Both atoms must be alive
            if not bond.a.alive or not bond.b.alive:
                continue
            center = (bond.a.pos + bond.b.pos) / 2
            if np.linalg.norm(center - pos) < radius:
                return bond
        return None

    def pick_ring(self, pos, radius=1.0):
        """Find ring whose COG is within radius of position. Returns Ring or None."""
        for ring in self.rings.values():
            if ring.alive and np.linalg.norm(ring.cog - pos) < radius:
                return ring
        return None

    def format_table(self, pos=False, neighbors=True, bond_orders=False, charge=False, hyb=True, alive_only=True):
        """Compact one-line-per-atom table for L1 agent review. uid elem [hyb npi] [neighs] [x y z]."""
        _hyb = {-1: 'cap', 0: 'sp3', 1: 'sp2', 2: 'sp'}
        cols = ['uid', 'elem']
        if hyb:
            cols += ['hyb', 'npi']
        if neighbors:
            cols.append('neighs')
        if bond_orders:
            cols.append('bonds')
        if charge:
            cols.append('charge')
        if pos:
            cols += ['x', 'y', 'z']
        lines = ['# ' + '  '.join(cols)]
        for a in sorted(self.atoms.values(), key=lambda x: x._id):
            if alive_only and not a.alive:
                continue
            row = [str(a._id), a.ename]
            if hyb:
                row += [_hyb.get(a.npi, '?'), str(a.npi)]
            if neighbors:
                nids = sorted(n._id for n in a.neighbors if (not alive_only or n.alive))
                row.append(','.join(str(i) for i in nids) if nids else '-')
            if bond_orders:
                parts = []
                for b in a.bonds:
                    if not b.alive:
                        continue
                    other = b.other(a)
                    if alive_only and not other.alive:
                        continue
                    parts.append(f'{other._id}:{b.order:g}')
                row.append(';'.join(parts) if parts else '-')
            if charge:
                row.append(f'{a.charge:.4f}')
            if pos:
                row += [f'{a.pos[0]:.4f}', f'{a.pos[1]:.4f}', f'{a.pos[2]:.4f}']
            lines.append('  '.join(row))
        return '\n'.join(lines)

    def __repr__(self):
        return f"AtomicGraph(atoms={len(self.atoms)}, bonds={len(self.bonds)}, rings={len(self.rings)})"
