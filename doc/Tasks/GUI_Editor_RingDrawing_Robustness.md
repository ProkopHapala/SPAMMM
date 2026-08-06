# Task: GUI Editor — Ring Drawing & Undo Robustness

**Status:** Investigating (root cause confirmed by code analysis and a headless runtime probe; fix not implemented or USER-verified)
**Priority:** P1 (editor produces corrupt topology → DFTB+ crashes `Atoms N and M too close together`)
**Related:** `spammm/topology/PackedMolecule.py`, `spammm/topology/MoleculeEditorBackend.py`, `spammm/topology/AtomicGraph.py`, `spammm/GUI/SPAMMM_GUI.py`, `spammm/GUI/EditModeHandlers.py`, skill:`molecular-structure-sync`

---

## 1. Goal

Make the hex/ring drawing + undo workflow **structurally robust** so that the editor cannot produce a topology with two heavy atoms at (near-)identical positions or place H caps inside C–C bonds. The current architecture has a **state-loss bug in the undo snapshot**: Ctrl+Z replaces the authoritative `AtomicGraph` with a graph whose atoms have lost their grid pins. The ring drawer then treats occupied grid nodes as empty and creates duplicate carbons.

This is not an `undo()`-only quick patch. The snapshot must preserve the authoritative graph state that later editor operations consume, and restore must reconcile backend-derived state without inferring or silently changing the restored molecule.

---

## 2. Symptom (reproduced)

Workflow that triggers it:
1. Draw a hex (or several) — carbons pinned to grid, H caps placed correctly.
2. Accidentally click empty space → a free C atom is added.
3. Ctrl+Z to undo.
4. Draw a new hex on top of / adjacent to existing atoms.
5. Run relaxation → DFTB+ fails: `Atoms 13 and 7 too close together` (two atoms at ~0 Å).

Secondary visible symptom: H atoms appear **inside** C–C bonds after undo+redraw.

Terminal log (representative):
```
[ADD_FREE_ATOM] type=C pos=(4.17,0.26)
...
Undo: restored 20 atoms
...
[UNIFIED_PRESS] target=hex ...
DEBUG: run_relaxation starting with 30 atoms: {'C': 16, 'H': 14}
ERROR! -> Atoms 13 and 7 too close together
```

---

## 3. Root cause (confirmed by code reading)

### 3.1 The undo snapshot is lossy w.r.t. grid-painting state

`PackedMolecule` stores only:
```python
__slots__ = ('etype', 'apos', 'bonds', 'npi')
```
<ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/topology/PackedMolecule.py" lines="31-38" />

It does **not** store:
- `Atom.pin` (grid-node key)
- `Atom.parent` (H→heavy ownership)
- `Atom._id` / `Bond._id` (stable identity)
- `Atom.charge`
- `Bond.order`
- `backend.hex_tiles` (set of painted hexagons)

`from_graph` reads `graph.to_arrays()`, which emits only alive atoms/bonds with positions+types — no pins. `to_graph` rebuilds a **fresh** `AtomicGraph` with every atom added as `pin=None, parent=None` <ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/topology/PackedMolecule.py" lines="70-101" />, then tries to reconstruct H parents **by distance** (nearest heavy within 1.5 Å). Bond orders are lost (all bonds come back as default `1.0`).

Positions are also converted from authoritative `float64` graph coordinates to `float32` in the packed representation. That is sufficient for display, but it is not a lossless undo snapshot.

**Consequence:** after any undo, `graph._pin_to_atom` is **empty** and every atom has `pin=None`; other authoritative properties are silently altered as well.

### 3.2 `add_ring` keys entirely off the pin cache to detect existing atoms

```python
n2a = self.graph._pin_to_atom
for node in self.grid.ring_nodes(q, r):
    nk = snap_to_grid(node)
    if nk not in n2a:          # <-- "empty" test is pin-based, NOT position-based
        a = self.graph.add_atom(... pin=nk ...)
```
<ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/topology/MoleculeEditorBackend.py" lines="380-397" />

After undo, `_pin_to_atom` is empty → **every node looks empty** → `add_ring` creates six new carbons, including duplicates at nodes shared geometrically with the restored structure. `_create_bonds_for_ring_atoms` bonds the six newly pinned ring atoms to each other when they are within `a_CC*1.1` <ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/topology/MoleculeEditorBackend.py" lines="528-553" />.

More precisely, `all_ring_atoms` contains only atoms currently reachable through `_pin_to_atom`. After undo, those are the newly added atoms; the restored pin-less atoms are not included. Therefore the new six-membered ring is bonded internally, while coincident restored/new carbons are **not bonded to each other**. The result is overlapping disconnected topology, which is sufficient to trigger `Atoms N and M too close together`.

### 3.3 H caps are generated for two overlapping disconnected fragments

The visible H-inside-bond symptom is real, but the previous zero-vector explanation was incorrect. The coincident old/new carbons are not neighbors, so `_calc_h_directions_atom` does not receive a zero-length vector between the twins. Also, `atomicUtils.normalize()` returns the original near-zero vector rather than NaN.

The actual mechanism is:
1. The restored ring and newly created adjacent ring overlap at shared grid nodes but remain topologically disconnected there.
2. Each duplicate carbon sees only its two neighbors within its own ring.
3. `adjust_h()` therefore passivates both carbons independently as under-coordinated sp2 atoms.
4. An H generated for one disconnected copy can point toward a heavy atom belonging to the other copy, placing it visually inside a C–C bond.

A headless probe of `add_ring(0,0) → PackedMolecule round-trip → add_ring(1,0)` produced:
```text
after restore: alive=12, pins=0, hex_tiles={(0,0)}, bond_orders={1.0}
after adjacent ring: heavy=12, H=12, overlapping_heavy_pairs=2
minimum heavy-heavy distance = 2.57e-8 Å
minimum H-to-non-parent-heavy distance = 0.330 Å
```
Both overlapping carbon pairs were unbonded and every duplicated carbon had two heavy neighbors.

The user's intuition — *"hydrogens should be added only after bonds between node atoms are settled"* — is already satisfied **within a single `add_ring` call** (bonds → `sync_neighbor_lists` → `adjust_h` <ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/topology/MoleculeEditorBackend.py" lines="391-401" />). The problem is not intra-call ordering; the heavy-atom graph is already corrupt before H regeneration begins.

### 3.4 `undo()` restores the graph but not backend state

```python
def undo(self):
    packed = self.undo_stack.pop()
    ...
    self.backend.graph = packed.to_graph()
    self.backend._sync_sys()
    self.refresh_view()
```
<ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py" lines="1385-1394" />

It does **not** call:
- `backend.reassign_pins()` (exists, rebuilds `_pin_to_atom` by snapping positions to grid nodes <ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/topology/MoleculeEditorBackend.py" lines="1335-1345" />)
- `backend._guess_rings()` (exists, infers and adds `hex_tiles` from pins <ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/topology/MoleculeEditorBackend.py" lines="1595-1612" />)
- any bond-order, charge, pin, parent, or identity restore
- any explicit `_rings_dirty` invalidation
- any reconciliation of backend state that still refers to old atom IDs

`reassign_pins()` is appropriate after a **grid transform**, where pins intentionally must be inferred against a changed grid <ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py" lines="1018-1095" />. It is not the preferred undo restore path:
- undo already had exact pin assignments before the mutation and should restore them directly;
- tolerance-based snapping can newly pin an originally free or dragged atom;
- `AtomicGraph.rebuild_pin_cache()` silently overwrites duplicate pin keys instead of rejecting collisions.

Likewise, `_guess_rings()` is currently additive: it does not clear stale `hex_tiles` or existing graph rings and scans only a hard-coded `q,r ∈ [-20,20]`. Calling it directly is not a complete rebuild.

---

## 4. Architectural findings that constrain the fix

### A. `AtomicGraph` is the molecular SSOT; `Atom.pin` is graph state
`AtomicGraph` is authoritative for positions, types, hybridization, charges, connectivity, bond orders, parents, and stable identity. `Atom.pin` is optional per-atom graph state used by grid editing. `graph._pin_to_atom` is explicitly documented as a **rebuildable cache**, not primary state.

Therefore:
- undo must restore `Atom.pin`;
- `_pin_to_atom` must be rebuilt or populated from those exact pins;
- cache rebuild must assert that two alive atoms never claim the same pin;
- tolerance-based geometric snapping must not substitute for stored pin state during undo.

### B. `PackedMolecule` currently mixes three fidelity requirements
`PackedMolecule` is used for:
- undo history;
- internal/external clipboard and append-import;
- `.npz` binary I/O.

Undo requires exact identity-preserving restoration. Paste requires **new** identities and usually no grid pins at the destination. External XYZ/MOL text may not carry all editor metadata. A global “store only heavy atoms” change would alter clipboard/import behavior and discard explicit H geometry.

The fix must either:
1. make `PackedMolecule` capable of carrying optional lossless graph metadata while each consumer explicitly chooses restoration semantics; or
2. introduce a dedicated editor undo snapshot that does not change clipboard/import semantics.

Do not assume that all H atoms can be omitted globally. The current graph stores H caps as normal `Atom`/`Bond` objects with `npi=-1` and `parent`, and the topology skill does not define them as non-authoritative.

### C. H lifecycle and dead-H accumulation are real but separate
`adjust_h()` removes and recreates H caps, and `remove_h_caps()` soft-deletes without compaction <ref_snippet file="/home/prokophapala/git/SPAMMM/spammm/topology/MoleculeEditorBackend.py" lines="1125-1195" />. Repeated editing therefore accumulates dead H atoms and bonds.

This deserves a separate lifecycle/performance task. It is not required to correct undo:
- an exact undo snapshot can restore existing H atoms, parents, bonds, positions, and IDs directly;
- calling `adjust_h()` during undo would destroy that fidelity;
- moving `adjust_h()` into `refresh_view()` would make a rendering refresh mutate authoritative topology and must not be done.

### D. `hex_tiles` semantics are not yet formally defined
`backend.hex_tiles` is used by `Hex2` removal to decide whether node atoms are shared with another painted tile. It can drift from the graph because atom deletion does not update it.

It may be possible to define it as a derived cache: “a tile exists iff all six grid nodes are pinned.” That matches `_guess_rings()` for tile insertion, but it can also classify six individually placed atoms as a painted tile. Until that semantic is explicitly accepted, the surgical undo behavior is to snapshot and restore `hex_tiles` exactly. Any derived-cache refactor must use a true `clear → rebuild → validate` path, not the current additive `_guess_rings()`.

### E. Bond order, charge, precision, and identity loss are part of undo fidelity
Undo currently:
- converts all bonds to `order=1.0`, losing user-edited and Kekulé orders;
- resets all atom charges to `0.0`;
- round-trips positions through `float32`;
- mints new `Atom._id` and `Bond._id` values.

Identity loss already affects persistent ID consumers: scene selections, `backend.constraint_set`, fragment caches, and extension-held atom IDs. Stable IDs therefore should be preserved for objects that existed in the restored snapshot. Global counters may remain monotonic above all minted/restored IDs; they must never be rewound into a collision.

### F. Derived geometry-ring state must be invalidated
`graph.rings` can be re-detected from the restored bond graph. Undo should set `backend._rings_dirty = True` so the normal detection path rebuilds it. This is distinct from `hex_tiles`, which controls grid painting behavior.

---

## 5. The corruption chain, end to end

1. Draw hex → carbons pinned, `hex_tiles` updated, H caps placed (correct).
2. Accidentally click empty → `_add_free_atom` pushes a lossy snapshot, adds an off-grid atom, and adjusts H.
3. Ctrl+Z → `undo()` replaces `graph` with a fresh pin-less graph; `_pin_to_atom` is empty, stable IDs change, charges reset, positions are float32-round-tripped, and bond orders flatten to single.
4. `hex_tiles` is not restored with the graph. In this exact free-atom repro its value happens to remain unchanged; undoing ring/atom operations can leave it stale.
5. Draw an adjacent hex → `add_ring` sees every restored node as empty and creates a complete new six-carbon ring. At shared geometric nodes, old and new carbons overlap but are not bonded to each other.
6. `adjust_h()` passivates the two disconnected components independently. Extra H caps can point toward non-parent atoms in the overlapping component and appear inside C–C bonds.
7. Run relaxation → DFTB+ sees two atoms at ~0 Å → `Atoms N and M too close together`.

---

## 6. Required changes (implementation direction)

### 6.1 Add an undo representation with explicit fidelity
The undo snapshot must preserve, for every alive graph object:
- atom: `_id`, element/type, `float64 pos`, `pin`, `parent` reference by snapshot index/ID, `npi`, and `charge`;
- bond: `_id`, endpoint references, and `order`;
- backend editor state: at minimum an exact copy of `hex_tiles`.

Explicit H atoms/caps remain in the undo snapshot. Do not regenerate them during restore. Clipboard paste/import may use the same packed data, but must intentionally mint new IDs, clear destination pins, and remap parents/bond endpoints within the pasted subset.

Whether this is implemented as optional lossless fields on `PackedMolecule` or as a dedicated editor snapshot remains a code-structure decision. The behavior above is required either way.

### 6.2 Restore exact graph state, then rebuild only derived caches
The undo restore sequence should be:
1. Reconstruct the graph with stored IDs, positions, pins, parents, charges, bonds, and bond orders.
2. Populate/rebuild `_pin_to_atom` from stored `Atom.pin` values and reject duplicate pins.
3. Restore `hex_tiles` exactly for the initial surgical fix.
4. Set `_rings_dirty = True`; geometry rings are derived from bonds.
5. Validate graph/cache invariants.
6. Call `_sync_sys()` once, reconcile transient GUI ID holders, and refresh the view.

Do **not** call `reassign_pins()`, `_guess_rings()`, or `adjust_h()` as substitutes for restoring missing state.

### 6.3 Preserve stable identity safely
Restored atoms/bonds that existed in the snapshot keep their `_id`. Newly created objects after undo still receive fresh monotonically increasing IDs. Restoration must verify:
- graph dictionary keys equal object IDs;
- no two alive atoms/bonds share an ID;
- global counters remain at least as large as every ID ever minted and are not rewound.

Transient GUI/extension ID sets should be intersected with restored live IDs where appropriate. With identity preservation, surviving selections and constraints continue to reference the intended atoms.

### 6.4 Validate grid and overlap invariants
Add a reusable validation path for editor mutations and restore:
- every non-`None` `Atom.pin` is unique among alive atoms;
- `_pin_to_atom` equals the mapping derived from alive `Atom.pin` values;
- no two alive heavy atoms are closer than the chosen overlap tolerance;
- every alive bond connects two distinct alive atoms and has finite positive length;
- parent references of H caps point to alive heavy atoms.

### 6.5 Detect-and-refuse positional overlap in `add_ring`
The pin lookup remains the normal fast path. If a ring node has no cached pin but an alive heavy atom already lies within the overlap tolerance, `add_ring` must fail loudly with the conflicting atom IDs/position instead of silently creating another atom. Do not silently merge or skip: those policies can hide a stale cache and leave incomplete topology.

This guard is secondary containment. Correct undo fidelity is the primary fix.

### 6.6 Defer broader H and tile-cache refactors
Do not bundle the following into the first correction:
- removal of eager `adjust_h()` calls from every mutation;
- H-cap compaction/lifecycle redesign;
- conversion of `hex_tiles` into a derived cache;
- general undo/redo/branch/compression work.

Each may be worthwhile, but each changes behavior beyond the demonstrated root cause.

---

## 7. Decisions and verification required before implementation

### 7.1 Choose the code container for exact undo state
The behavior is settled by §6.1; only code placement remains:
- extend `PackedMolecule` with optional lossless metadata and backward-compatible defaults; or
- introduce a dedicated undo snapshot, preferably reusing existing packing helpers rather than duplicating graph serialization.

Inventory already confirms that `PackedMolecule` is used by undo, clipboard/paste, append-import, text conversion, and NPZ I/O. Any extension must test all of those consumers.

### 7.2 Confirm `hex_tiles` semantics
For the first fix, snapshot/restore the set exactly. Separately decide whether the long-term invariant should be:

> A hex tile is present iff all six ring nodes have unique alive pinned atoms.

If accepted, replace `_guess_rings()` with a true bounded-independent rebuild and update every grid/atom mutation through one reconciliation path. If not accepted, `hex_tiles` remains explicit editor history and must be included in every editor snapshot.

### 7.3 Define the overlap tolerance once
The current relevant tolerance is approximately `0.3 Å`, also used for pin snapping. The implementation should define one named editor-overlap tolerance and use it consistently in:
- the `add_ring` fail-fast guard;
- graph validation;
- L0 assertions.

This tolerance must be well below any legitimate heavy-heavy bond length.

### 7.4 Test design (TDD — write failing L0 first)
Per `doc/TEST_DESIGN.md`, add backend-level pytest coverage rather than automating Qt clicks for the core invariant.

Required L0 checks:
1. **Authoritative snapshot parity:** IDs, element/type, `float64` positions, pins, parents, `npi`, charges, bond endpoints/IDs/orders, and `hex_tiles` survive one undo round-trip.
2. **Pin-cache parity:** `_pin_to_atom` exactly matches alive `Atom.pin` values; expected count is restored; duplicate pins raise.
3. **Primary reproduction:** `add_ring(0,0) → snapshot/restore → add_ring(1,0)` produces C10H8, not C12H12; minimum heavy-heavy separation exceeds the overlap tolerance.
4. **H geometry:** after the reproduction, every H is bonded/parented correctly and no H lies within `a_CC*0.5` of a non-parent heavy atom.
5. **Bond-order and charge parity:** aromatic/double/user-cycled orders and nonzero atom charges survive undo.
6. **Identity continuity:** pre-existing selected/constrained atom IDs remain valid after undo; IDs of newly created objects do not collide.
7. **Fail-fast containment:** intentionally clear/corrupt the pin cache while leaving a heavy atom at a ring node; `add_ring` must raise before adding a duplicate.
8. **Normal-path regression:** adjacent ring drawing without undo still produces C10H8; atom add/delete and bond cycling retain existing behavior.

L1 review:
- dump `AtomicGraph.format_table(pos=True, neighbors=True, bond_orders=True, charge=True)` before mutation and after restore;
- report pin/cache maps, IDs, `hex_tiles`, minimum heavy-heavy distance, and minimum H-to-non-parent-heavy distance.

L2 human review:
- before/after image of benzene → undo → adjacent ring;
- confirm shared carbons are single nodes and all H caps point outside the fused-ring skeleton.

Integration verification:
- generate the relaxation input from a structure built with undo in its history;
- assert no overlapping heavy atoms before launching DFTB+;
- run DFTB+ relaxation when the executable/environment is available and report its result separately from the L0 topology verdict.

---

## 8. Out of scope (do not bundle)

- General undo/redo stack improvements (redo, branching, compression).
- Global H-cap transaction/lifecycle redesign and dead-object compaction.
- Converting `hex_tiles` to a derived cache before its semantics are approved.
- Unrelated ring-detection algorithm changes.
- Relaxation/DFTB+ error handling beyond rejecting invalid topology before launch.

Bond orders, charges, pins, parents, float precision, and stable IDs are **not** out of scope: they are authoritative state already lost by the same undo serialization path and must be handled by an exact snapshot.

---

## 9. Acceptance criteria (USER must confirm before marking Done)

1. The repro sequence in §2 no longer produces overlapping atoms — verified by L0 test.
2. After undo, `_pin_to_atom` and `hex_tiles` are consistent with the visible graph — verified by L0 test.
3. Pins, parents, charges, bond orders, stable IDs, and `float64` positions survive undo — verified by L0 parity checks.
4. Adjacent-ring undo reproduction produces C10H8 with two shared carbons, not overlapping disconnected C12H12.
5. H caps do not appear inside C–C bonds after undo+redraw — verified by L0 geometry checks and L2 visual review.
6. Stale/missing pin-cache corruption fails before any duplicate heavy atom is added.
7. DFTB+ relaxation succeeds on a structure built with undo in the history.
8. No regression in normal (no-undo) hex drawing, atom add/delete, bond cycling, clipboard paste, append-import, or NPZ round-trip.

Per AGENTS.md: **do not mark Done until USER confirms after seeing the test results.**
