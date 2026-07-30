"""
RigidEnsemble.py — Shared Python numpy representation of rigid-molecule poses (pos + qrot).

Purpose: A thin, dependency-free (numpy-only) shared store of per-molecule 6-DOF rigid
poses — the rigid-body-modules analogue of `AtomicGraph` for atoms. All rigid-body
consumers (PairFF, FoldedRigid, Assembly, ChargeRings PME, MC/GA) import this and read
`get_poses()` so they stop forking geometry among themselves.

Scope (locked USER 2026-07-30, see doc/Tasks/RigidMoleculePose_SSOT.md):
  - **Poses only.** No template data (atoms_body, multipole_cs, Esite) — those stay in
    consuming modules. The ensemble holds: stable `id`, `tid` (species ref), `pos (3,)`,
    `qrot (4,) xyzw`, optional `pin`/`active`/`in_pme_subset`.
  - **NOT a global SSOT.** `AtomicGraph`/`SPAMM_GUI` stay independent and must work fine
    when no rigid module is loaded. The ensemble is optional, scoped to rigid modules.
  - **Dependency direction: modules → ensemble, never the reverse.** The ensemble exposes
    fast flat numpy arrays; each module reads them and does its OWN conversion to its own
    GPU buffers / internal format using its OWN template data. No converters live here.
  - **One-way ensemble → AtomicGraph** on demand (manual, when a rigid module wants to
    update display). Reverse (atoms → poses) is deferred — out of scope.
  - **GPU buffers stay as-is**, per-algorithm optimized — NOT touched/converted/mirrored.

Conventions (locked from RigidBody audit — do not reinvent):
  - Quaternion: `qrot = (qx, qy, qz, qw)` xyzw; identity `[0,0,0,1]`.
  - World atoms: `apos_world = pos + R(q) @ apos_body` (R from `_quat_to_matrix_np` in
    RigidBodyDynamics — consuming modules call that, not anything here).
  - CoM: rigid-body center; mass-weighted for dynamics, but settable by other means.

Stable ids: `RigidBody._counter` (class-level), `_id` per body — mirrors `Atom._id` in
AtomicGraph. Ids are never reused. Array index ≠ id; use `id_to_idx` / `set_pose_by_id`
to bridge. Dense indices are stable for the lifetime of the ensemble (bodies are only
appended, never removed in the current scope).

Role in SPAMMM: shared rigid-pose rep consumed by PairFF/FoldedRigid/Assembly/ChargeRings.
This module does NO physics, NO OpenCL, NO plotting — pure numpy pose storage.
"""

import numpy as np

# Default preallocation chunk (grown by doubling when exceeded). Tuned for typical
# on-surface assembly sizes (a few to a few hundred molecules).
_DEFAULT_CAP = 16
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class RigidBody:
    """One rigid molecule's pose slot (metadata only — arrays live in the ensemble)."""
    __slots__ = ('id', 'tid', 'pin', 'active', 'in_pme_subset', 'alive')
    _counter = 0

    def __init__(self, tid, pin=None, active=True, in_pme_subset=True):
        RigidBody._counter += 1
        self.id = RigidBody._counter          # stable id (never reused)
        self.tid = tid                        # species ref (hashable: int/str)
        self.pin = pin                        # optional pin constraint (None or (3,))
        self.active = active                  # participates in dynamics / scans
        self.in_pme_subset = in_pme_subset    # included in PME active subset
        self.alive = True

    def __repr__(self):
        return f"RigidBody(id={self.id} tid={self.tid} active={self.active})"


class RigidEnsemble:
    """Shared numpy store of rigid-molecule poses (pos + qrot).

    Internal layout: preallocated contiguous `_pos (cap,3)` and `_qrot (cap,4)` float32
    arrays; `n_bodies` tracks the active count. `get_poses()` returns the active slice.
    Bodies are append-only in the current scope (no removal) so dense indices are stable.

    Conventions: qrot xyzw, identity [0,0,0,1]. CoM = rigid-body center (mass-weighted
    for dynamics, settable otherwise). dtype float32 by default (matches GPU upload).
    """

    def __init__(self, capacity=_DEFAULT_CAP, dtype=np.float32):
        self._cap = max(int(capacity), 1)
        self._dtype = np.dtype(dtype)
        self._pos = np.zeros((self._cap, 3), dtype=self._dtype)
        self._qrot = np.tile(IDENTITY_QUAT.astype(self._dtype), (self._cap, 1))
        self._bodies = []                     # list[RigidBody], index = dense index
        self._id_to_idx = {}                  # id -> dense index
        self.n_bodies = 0

    # ─── construction / growth ───────────────────────────────────────────────
    def _ensure_cap(self, need):
        if need <= self._cap: return
        newcap = self._cap
        while newcap < need: newcap *= 2
        pos = np.zeros((newcap, 3), dtype=self._dtype)
        qrot = np.tile(IDENTITY_QUAT.astype(self._dtype), (newcap, 1))
        pos[:self.n_bodies] = self._pos[:self.n_bodies]
        qrot[:self.n_bodies] = self._qrot[:self.n_bodies]
        self._pos, self._qrot, self._cap = pos, qrot, newcap

    def add_body(self, tid, pos, qrot=None, pin=None, active=True, in_pme_subset=True):
        """Append one body; returns its dense index. qrot defaults to identity."""
        self._ensure_cap(self.n_bodies + 1)
        i = self.n_bodies
        b = RigidBody(tid, pin=pin, active=active, in_pme_subset=in_pme_subset)
        self._bodies.append(b)
        self._id_to_idx[b.id] = i
        self._pos[i] = np.asarray(pos, dtype=self._dtype)[:3]
        self._qrot[i] = (np.asarray(qrot, dtype=self._dtype)[:4] if qrot is not None else IDENTITY_QUAT.astype(self._dtype))
        self.n_bodies = i + 1
        return i

    def add_bodies(self, tids, pos, qrot=None, pins=None, active=None, in_pme_subset=None):
        """Append N bodies from arrays. tids: list of length N. pos (N,3), qrot (N,4) or None."""
        tids = list(tids)
        n = len(tids)
        if n == 0: return []
        pos = np.asarray(pos, dtype=self._dtype)
        if pos.shape != (n, 3): raise ValueError(f"pos shape {pos.shape} != ({n},3)")
        if qrot is None:
            qrot = np.tile(IDENTITY_QUAT.astype(self._dtype), (n, 1))
        else:
            qrot = np.asarray(qrot, dtype=self._dtype)
            if qrot.shape != (n, 4): raise ValueError(f"qrot shape {qrot.shape} != ({n},4)")
        self._ensure_cap(self.n_bodies + n)
        idxs = []
        for k in range(n):
            i = self.n_bodies
            b = RigidBody(tids[k],
                          pin=(pins[k] if pins is not None else None),
                          active=(active[k] if active is not None else True),
                          in_pme_subset=(in_pme_subset[k] if in_pme_subset is not None else True))
            self._bodies.append(b)
            self._id_to_idx[b.id] = i
            self._pos[i] = pos[k]
            self._qrot[i] = qrot[k]
            self.n_bodies = i + 1
            idxs.append(i)
        return idxs

    @classmethod
    def from_poses(cls, tids, pos, qrot=None, capacity=None, dtype=np.float32, **kw):
        """Build an ensemble from arrays in one call. tids: list of length N."""
        n = len(tids)
        cap = max(int(capacity) if capacity is not None else n, 1)
        ens = cls(capacity=cap, dtype=dtype)
        ens.add_bodies(tids, pos, qrot, **kw)
        return ens

    # ─── reads ───────────────────────────────────────────────────────────────
    def get_poses(self, copy=True, subset=None):
        """Return (pos (n,3), qrot (n,4)) for the active slice (or a subset of indices).

        copy=True (default): returns fresh copies — safe for callers that mutate.
        copy=False: returns views into the internal buffers — fastest, but caller must
        NOT mutate in place (would corrupt the ensemble). Use copy=False only on hot
        paths where the caller immediately copies or reads read-only.
        subset: optional list of dense indices to return (e.g. PME active subset).
        """
        if subset is None:
            sl = slice(0, self.n_bodies)
        else:
            sub = np.asarray(subset, dtype=np.int64)
            sl = sub  # fancy indexing
        pos = self._pos[sl]
        qrot = self._qrot[sl]
        if copy:
            return pos.copy(), qrot.copy()
        return pos, qrot

    def get_pose(self, i, copy=True, by_id=False):
        """Return (pos (3,), qrot (4,)) for one body. by_id=True to look up by stable id."""
        idx = self._id_to_idx[i] if by_id else int(i)
        pos = self._pos[idx]
        qrot = self._qrot[idx]
        if copy:
            return pos.copy(), qrot.copy()
        return pos, qrot

    def get_ids(self):
        """Return list of stable ids for all bodies (dense order)."""
        return [b.id for b in self._bodies]

    def get_tids(self):
        """Return list of species tids for all bodies (dense order)."""
        return [b.tid for b in self._bodies]

    def get_body(self, i, by_id=False):
        """Return the RigidBody metadata object for index/id i."""
        idx = self._id_to_idx[i] if by_id else int(i)
        return self._bodies[idx]

    def id_to_idx(self, id):
        return self._id_to_idx[id]

    # ─── writes ──────────────────────────────────────────────────────────────
    def set_poses(self, pos, qrot=None, subset=None):
        """Overwrite poses for all bodies (or a subset). pos (n,3); qrot (n,4) or None.

        If subset is None, n must equal n_bodies. If subset is given, len(subset) must
        match pos.shape[0]. qrot=None keeps existing quaternions.
        """
        pos = np.asarray(pos, dtype=self._dtype)
        if subset is None:
            if pos.shape != (self.n_bodies, 3):
                raise ValueError(f"set_poses: pos shape {pos.shape} != ({self.n_bodies},3)")
            self._pos[:self.n_bodies] = pos
            if qrot is not None:
                qrot = np.asarray(qrot, dtype=self._dtype)
                if qrot.shape != (self.n_bodies, 4):
                    raise ValueError(f"set_poses: qrot shape {qrot.shape} != ({self.n_bodies},4)")
                self._qrot[:self.n_bodies] = qrot
        else:
            sub = np.asarray(subset, dtype=np.int64)
            if pos.shape[0] != len(sub):
                raise ValueError(f"set_poses: pos rows {pos.shape[0]} != subset len {len(sub)}")
            self._pos[sub] = pos
            if qrot is not None:
                qrot = np.asarray(qrot, dtype=self._dtype)
                if qrot.shape[0] != len(sub):
                    raise ValueError(f"set_poses: qrot rows {qrot.shape[0]} != subset len {len(sub)}")
                self._qrot[sub] = qrot

    def set_pose(self, i, pos, qrot=None, by_id=False):
        """Overwrite one body's pose. by_id=True to look up by stable id."""
        idx = self._id_to_idx[i] if by_id else int(i)
        self._pos[idx] = np.asarray(pos, dtype=self._dtype)[:3]
        if qrot is not None:
            self._qrot[idx] = np.asarray(qrot, dtype=self._dtype)[:4]

    def normalize_quats(self, subset=None):
        """Normalize all (or a subset of) quaternions to unit length in place."""
        if subset is None:
            sl = slice(0, self.n_bodies)
        else:
            sl = np.asarray(subset, dtype=np.int64)
        q = self._qrot[sl]
        n = np.linalg.norm(q, axis=1, keepdims=True)
        n[n == 0] = 1.0  # guard against zero quats (leave as-is rather than NaN)
        self._qrot[sl] = q / n

    # ─── convenience ─────────────────────────────────────────────────────────
    def __len__(self): return self.n_bodies

    def __repr__(self):
        return f"RigidEnsemble(n_bodies={self.n_bodies}, cap={self._cap}, dtype={self._dtype.name})"

    def summary(self):
        """One-line human-readable summary for debug prints."""
        if self.n_bodies == 0: return "RigidEnsemble(empty)"
        pos = self._pos[:self.n_bodies]
        q = self._qrot[:self.n_bodies]
        qn = np.linalg.norm(q, axis=1)
        tids = self.get_tids()
        return (f"RigidEnsemble(n={self.n_bodies} tids={sorted(set(map(str,tids)))} "
                f"pos_range=[{pos.min():.2f},{pos.max():.2f}] |q|=[{qn.min():.4f},{qn.max():.4f}]")
