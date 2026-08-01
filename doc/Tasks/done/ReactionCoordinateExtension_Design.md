# ReactionCoordinateExtension — Design Spec (v2)

**Topical doc (OKF):** [`doc/Topics/ReactionCoordinateScan.md`](../Topics/ReactionCoordinateScan.md)  
**Pitfalls:** [`doc/Takeways.md`](../Takeways.md)

**Status:** implemented — pm-NEB relaxed, Mulliken charges, ESP blit, GUI/offline scripts  
**Supersedes:** [`ReactionCoordinateScan_Design.md`](ReactionCoordinateScan_Design.md) (v1 draft)  
**Extension name:** `ReactionCoordinateExtension` (`spammm/GUI/ReactionCoordinateExtension.py`)  
**Registry key:** `'reaction_coord'` in `ExtensionManager`

---

## 1. Scope and philosophy

A **separate GUI extension** for reaction-coordinate exploration on structures held in **`AtomicGraph`** (assemblies, ribbons, ASCII-built systems, etc.). It is **not** part of Kekule solver, ASCII builder, or the hex editor core.

**Near-term goal:** fast **rigid DFTB** profiles along H-bond (and similar) coordinates, stored in a **binary scan dataset**, previewed with **m sliders** in the GUI.

**Medium-term goal:** “poor man’s NEB” and **partially relaxed** scans (pin H + junction heavy atoms, relax the rest).

**Long-term goal:** full NEB / relaxed pathways; **Kekulé-sensitive** observables (C–C bond lengths and π bond orders in aromatic cores as H moves between heterocycles). Bond-length **coloring in Vispy is deferred** until relaxed / NEB phases need it.

**Principles:**

- **SSOT:** authoritative geometry = `AtomicGraph`; preview = visualization overlay until explicit sync.
- **Reuse:** `find_hbonds`, `hbond_scan` path math, `run_dftb_sp`, `save_xyz_movie`, `constrained_scan`, `PackedMolecule` I/O patterns.
- **No auto-coarsening:** user sets step `dx` (or control step) manually; compute once, load many times.
- **KISS phases:** rigid before relax; relax endpoints before full path relaxation.

---

## 2. User decisions (round 2 + round 3)

| Topic | Decision |
|-------|----------|
| Extension | **Separate** `ReactionCoordinateExtension`, not folded into Kekule/FF panels |
| Preview sliders | Modify **visualization only**; **explicit Sync → Graph** / **Import ← Graph** buttons for `AtomicGraph` |
| Atom pinning | **`backend.constraint_set: set[Atom._id]`** — shared by RC, DFTB relax, FF (migrate FF to read it in phase 1b) |
| Grid step | **Manual** `dx` (Å along path or control step); **no** automatic coarsening; **one common `dx` for all controls** (v1) |
| Storage | Primary: **`.npz`** arrays; `.xyz` movie **optional export** |
| Phase 1 QM | **Gas-phase** DFTB rigid SP default |
| PBC | **After first gas-phase demo** |
| Bond coloring | **Deferred** (Vispy); **C–C length + Kekulé π-BO analysis in library** from Phase B |
| Poor-man NEB | **Linear interpolation of all atoms** between endpoint frames (u=0, u=1), then DFTB SP |
| Sync to Graph | **Full interpolated frame** (all atoms at current control) |
| Roadmap | Rigid → poor-man NEB (+ Kekulé/C–C) → pinned partial relax → (future) full NEB |

---

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│  SPAMMM_GUI (SPAMMMWindow)                                       │
│    MoleculeEditorBackend ──► AtomicGraph (SSOT)                  │
│         │                    ▲                                   │
│         │ _sync_sys()        │ Sync to Graph (explicit)          │
│         ▼                    │                                   │
│    AtomicSystem (dense)      │                                   │
│         │                    │                                   │
│    Vispy AtomScene ◄─────────┘ Preview overlay (sliders)         │
│         ▲                                                        │
│         │ fixed mask, hbond lines, (future: replicas)          │
├─────────┴───────────────────────────────────────────────────────┤
│  ReactionCoordinateExtension (panel)                             │
│    • detect / select H-bonds                                     │
│    • m controls + mapping (n hbonds → m params)                  │
│    • pin mode (writes `backend.constraint_set`)                  │
│    • Run rigid DFTB → ScanDataset.npz                            │
│    • Load scan → m sliders + E readout                           │
├──────────────────────────────────────────────────────────────────┤
│  spammm/topology/hbond_utils.py      — discovery + HbondRecord   │
│  spammm/topology/scan_dataset.py     — ScanDataset .npz I/O      │
│  spammm/quantum/coordinate_scan.py   — paths, grids, DFTB runner │
│  spammm/topology/scan_kekule.py      — C–C bond_len + Kekulé π-BO on dataset │
└──────────────────────────────────────────────────────────────────┘
         FFExtension ──► FFController pins ← same fixed_atom_ids
```

---

## 4. Authoritative graph vs preview overlay

### 4.1 Two layers

| Layer | Holds | Mutated by |
|-------|--------|------------|
| **Authoritative** | `AtomicGraph` atom positions (`Atom.pos`) | Editor, Sync from RC panel, DFTB relax import, explicit “Apply frame” |
| **Preview** | `window.rc_preview_pos` or `scene` overlay buffer | m-sliders only |

Default slider motion **does not** write to `AtomicGraph` (undo-safe, no accidental commit).

### 4.2 Explicit sync buttons

| Button | Action |
|--------|--------|
| **Import from Graph** | Snapshot `backend._sync_sys()` → `reference_apo`, H-bond list, pin mask; reset sliders to start |
| **Sync to Graph** | Write current preview positions to `Atom.pos` for selected/moved atoms (at minimum: all scan H atoms + any atoms moved in preview); `_sync_sys()`; `refresh_view()` |
| **Reset preview** | Discard overlay; show graph geometry |

**Decision (round 3):** Sync to Graph writes the **full frame** at current control values (all atoms in preview buffer).

---

## 5. General atom constraints (pinning)

Pinning is **not** RC-specific. FF already pins via `FFController`; RC uses the same semantics **without requiring FF build**.

### 5.1 Current state (inventory)

| Mechanism | Location | Semantics |
|-----------|----------|-----------|
| `Atom.pin` | `AtomicGraph` | Hex **grid** key — **not** spatial fix |
| `backend.constraint_set` | `MoleculeEditorBackend` | **Spatial pin** by `Atom._id` (implemented) |
| `FFController._pinned_mask` | FF | Harmonic constraint in GPU MD |
| `AtomScene._fixed` | Vispy | Display + exclude from some picks |
| `makeDFTBjob_pbc(fixed_atoms=…)` | DFTB | `MovedAtoms` for **relaxation** |

### 5.2 Implemented: `backend.constraint_set`

On `MoleculeEditorBackend`:

```python
constraint_set: set[int]   # Atom._id pinned in space
toggle_constraint(atom_id) / toggle_constraint_by_index(idx)
constraint_mask() -> bool[natoms]
fixed_atom_indices() -> list[int]   # for DFTB MovedAtoms
clear_constraints()
```

RC panel **RC pin** edit mode toggles this set and updates `scene.set_fixed_mask`.

**Future:** migrate `FFController` to read `constraint_set` instead of maintaining a separate mask.

### 5.3 Deferred: `Atom.fixed` on graph

Optional later field if pins must survive graph round-trips without backend reference. Not required for v1.

---

## 6. H-bond detection and selection

### 6.1 Detection SSOT

**Geometry on synced system:** `AtomicSystem.find_hbonds(d_max=2.5, a_min=150)` — already in `refresh_view`.

Wrapper `find_hbonds_graph(backend) → list[HbondRecord]`:

- Stable IDs: `donor_id`, `h_id`, `acceptor_id` (`Atom._id`)
- Ephemeral dense indices refreshed on each `_sync_sys`
- Label: `"N12-H45…O3"` for UI

ASCII `resolve_hbond_pairs` remains for ASCII pipeline only.

### 6.2 Visualization (phase 1)

- Keep purple H…A lines (existing).
- **Enhance:** optional D–H segment (lighter); highlight **selected** scan H-bonds.
- Checkbox list in RC panel; invalid pairs (shared H) flagged at setup.

### 6.3 Control mapping (n H-bonds, m parameters)

- **n** = selected H-bonds (typically 1–3, max ~5).
- **m** = independent control parameters (typically 1–2, max ~5).
- **`mapping[i]`** = control index driving H-bond `i` fraction along its D→A axis.

Examples:

| Scenario | n | m | mapping | Grid |
|----------|---|---|---------|------|
| Single transfer | 1 | 1 | `[0]` | 1D |
| Two H, same coordinate | 2 | 1 | `[0,0]` | 1D |
| Two H, independent | 2 | 2 | `[0,1]` | 2D tensor product |
| Three H, two controls | 3 | 2 | `[0,0,1]` | 2D |

**Fraction → position:** reuse `make_hbond_transfer_path` per H-bond (store `h0`, `h1`, axis at setup from **reference** geometry).

**Step control:** user sets **`dx`** (Å) along each control axis or uniform step in fraction space — **manual only**. UI shows total frame count before Run.

---

## 7. Scan computation and storage

### 7.1 Compute once, load many times

Workflow:

1. Import from Graph → reference geometry + H-bond list.
2. Configure mapping, `dx`, method (**rigid DFTB**).
3. **Run scan** (background worker; progress bar).
4. Save **`ScanDataset.npz`** (primary artifact).
5. Optional: export **`scan.xyz`** via `save_xyz_movie` for external tools.

Subsequent sessions: **Load .npz** → instant slider exploration (no DFTB).

### 7.2 `ScanDataset` (new — not quite `PackedMolecule`)

`PackedMolecule` = **single frame** topology snapshot (~21 B/atom). Scan needs **trajectory + scalars + metadata**.

Proposed **`spammm/topology/scan_dataset.py`**:

```python
class ScanDataset:
    # Static topology (frame 0 reference)
    etype: int32[natoms]
    bonds: int32[nbonds, 2]      # ibond → (i,j) atom indices
    atom_ids: int32[natoms]      # Atom._id at scan time (for graph round-trip)

    # Trajectory
    apos: float64[nframes, natoms, 3]
    controls: float64[nframes, m]           # independent control values (u0, u1, …)

    # Precomputed observables
    bond_len: float64[nframes, nbonds]      # |r_i - r_j| each frame (incl. C–C)
    energies_ev: float64[nframes]           # NaN if preview-only trajectory
    pi_bo_cc: float64[nframes, n_cc]         # optional: Kekulé π bond orders on C–C bonds

    # NOT stored: fractions[nframes, n_hbonds] — see below

    # Metadata (JSON-serializable dict in npz)
    meta: {
        scan_type: 'rigid_dftb' | 'pm_neb_sp' | 'relaxed_dftb',
        dx, sk_set, mapping, hbond_records,
        fixed_atom_ids, gas_phase: bool,
        lvec: (3,3) or None,
        created, git_rev?, ...
    }
```

**I/O:**

```python
ScanDataset.save_npz(path)
ScanDataset.load_npz(path) -> ScanDataset
ScanDataset.export_xyz(path)  # thin wrapper → save_xyz_movie
ScanDataset.frame(i) -> apos_i
```

**Bond lengths:** compute on save from `apos` + `bonds` (vectorized). Enables future Kekulé analysis without re-parsing XYZ.

**Why not only XYZ:** loading `apos[nframes,natoms,3]` from npz is **O(1)** mmap-friendly; XYZ parse is slow and lossy for metadata.

### 7.2.1 `controls` vs `fractions` (not stored)

| Array | Meaning |
|-------|---------|
| **`controls[k, j]`** | Independent reaction-coordinate parameter **u_j** for frame *k* (what sliders and grid iterate). Typically u ∈ [0,1] = donor-side → acceptor-side for that control. |
| **`fractions[k, i]`** (derived) | Per-H-bond transfer fraction **f_i** along D→A axis for H-bond *i* at frame *k*. |

**Relation:** `f_i = controls[k, mapping[i]]` via `controls_to_fractions()` in `hbond_utils.py`.

**Why not store `fractions`:** redundant with `controls + mapping + hbond_records` in `meta`. Recompute when building frames or plotting. For m=1 shared control, `fractions[:,0] == controls[:,0]` for all selected H-bonds — storing both would duplicate data.

### 7.3 Quantum backend (phase 1)

| Function | Role |
|----------|------|
| `build_control_grid(ranges, dx, m)` | 1D/2D/… grid from manual step |
| `build_frame(apos0, hbonds, control_values, mapping)` | Place all H positions |
| `run_rigid_dftb_scan(...)` | Loop `run_dftb_sp` + charge restart |
| `dataset_from_scan(...)` | Build `ScanDataset` |

**Gas phase:** existing cluster `run_dftb_sp` (no `lvs`).

**No duplication** of DFTB input or XYZ writer.

---

## 8. Periodic boundary conditions (later phase)

### 8.1 Current inventory

| Piece | Status |
|-------|--------|
| `MoleculeEditorBackend.pbc_x`, `pbc_y`, `build_lattice_vectors()` | Flags + DFTB ribbon relax |
| `DFTB_utils.run_pbc`, `constrained_scan` | PBC DFTB with `fixed_atoms` |
| `AtomicSystem.clonePBC`, `clonePBC_central(nPBC)` | Replicate atoms for **plotting/FF** |
| `plotUtils.plotGeometry(..., replicate=)` | Matplotlib cell boxes |
| **Vispy replica rendering** | **Not implemented** in main editor |
| Assembly `n_pbc_xyz` | SAM packing context only |

### 8.2 Target: `nPBC_vis` for GUI

User-facing **`nPBC_vis = (nx, ny)`** (integer **radius** of cell images, same convention as `clonePBC_central`):

| `nPBC_vis` | Renders |
|------------|---------|
| `(0, 0)` | Unit cell only |
| `(0, 1)` | ±1 cells along **b** (y) — **polymer / ribbon** (up/down) |
| `(1, 0)` | ±1 cells along **a** (x) |
| `(1, 1)` | 3×3 grid in xy — **2D periodic** |

Implementation sketch:

- `AtomicSystem.lvec` from `backend.build_lattice_vectors()` when PBC enabled.
- `clonePBC_central(nPBC_vis + (0,))` → ghost positions for **draw only** (dimmed bonds/atoms).
- Unit cell box lines in Vispy (new `bbox_lines` use or lattice visual).

**RC scan with PBC:** use `run_pbc` / `constrained_scan` with same `lvec`; store `lvec` in `ScanDataset.meta`; energies k-point sampled (reuse ribbon `nk` heuristics or user `nk`).

**Phase:** after gas-phase rigid scan works in GUI.

---

## 9. Interactive exploration UI

### 9.1 Panel sections (`ReactionCoordinateExtension`)

1. **Geometry** — Import from Graph | Sync to Graph | Reset preview  
2. **H-bonds** — Refresh | list + checkboxes | count  
3. **Constraints** — Pin mode link | fixed count | clear all  
4. **Scan setup** — m controls | mapping table | dx | method dropdown (rigid DFTB)  
5. **Compute** — Run | progress | Save npz | Load npz | Export xyz  
6. **Explore** — m sliders | frame index | E, dE labels | (1D plot widget later)

### 9.2 Sliders

- **m sliders** bound to control values; on change → interpolate frame from loaded `ScanDataset` (or live preview path before DFTB).
- Update **Vispy positions** from preview buffer only.
- Display **E** if energies present in dataset.

### 9.3 Bond coloring — **deferred**

Planned when **relaxed / NEB** phases exist:

- Track **C–C aromatic bonds** (and X–H scan bonds) vs reference frame.
- Blue = shortened, red = elongated (Kekulé localization signal when H moves between heterocycles).
- Reuse / extend `set_bond_orders` line coloring in `VispyUtils`.

Not in phase 1–2.

---

## 10. Roadmap: scan methods (physics phases)

### Phase A — Rigid DFTB (implement first)

- Move H along D→A axis; all other atoms fixed.
- Fast; matches existing `hbond_scan` tests.
- Output: `ScanDataset` + optional xyz.

### Phase B — Poor man’s NEB + Kekulé / C–C analysis

User motivation: avoid full NEB cost; still capture **endpoint chemistry** and **aromatic bond response**.

1. Build **endpoint geometries** (u=0, u=1) via `build_frame` (H at donor-side vs acceptor-side).
2. *(Optional later)* Relax endpoints with DFTB opt before interpolation.
3. Generate interior frames by **linear interpolation of all atom coordinates** (`interpolate_all_atoms`).
4. **Single-point DFTB** on each frame.
5. Store as `scan_type='pm_neb_sp'`; `bond_len[:, ibond_cc]` already in dataset.
6. Run **`analyze_kekule_cc(dataset)`** (`scan_kekule.py`) on trajectory — π bond orders on C–C bonds vs control (library + tests; GUI plot later).

**Not** true NEB (no spring chain). Good enough for qualitative barriers and Kekulé-sensitive observables when H moves between heterocycles.

### Phase C — Partial relaxed scan

Along path (or pm-NEB frames):

- **Pin:** H (on path) + **nearest donor/acceptor heavy atoms** at junction (N…H…N).
- **Relax:** all other atoms (gas or PBC DFTB opt).
- Reuse `constrained_scan` / `run_pbc(do_relax=True, fixed_atoms=fixed_idx)`.

Pins come from `backend.constraint_set` + scan recipe (auto-pin junction atoms when enabling this mode).

### Phase D — Full NEB / CI-NEB (future)

Out of scope for initial extension; architecture leaves room in `ScanDataset.meta.scan_type` and separate runner.

---

## 11. Extension registration

```python
# ExtensionManager.py
'reaction_coord': dict(
    module='spammm.GUI.ReactionCoordinateExtension',
    dependencies=[],
    req_paths=[],   # DFTB checked at run time
    build_ui='build_ui',
),
DEFAULT_CONFIG['reaction_coord'] = dict(enabled=True)
```

Panel title: **Reaction coordinate**.

Optional edit mode: `'rc_pin'` (toggle fixed), `'rc_select_hbond'` (toggle H-bond in scan list).

---

## 12. Testing strategy

| Level | What |
|-------|------|
| L0 | `find_hbonds_graph` on dimer fixture; `build_control_grid`; frame builder respects pins |
| L0 | `ScanDataset` round-trip npz; bond_len consistency |
| L0 | `build_frame` + mapping cases (shared vs 2D) |
| Slow | `run_rigid_dftb_scan` (existing `test_hbond_scan` path, generalized) |
| L1 | RC panel smoke: load npz, slider changes preview positions (headless Qt optional) |
| L2 | PNG: energy vs control from dataset |

Follow `doc/TEST_DESIGN.md`; artifacts under `debug/test_reaction_coord/`.

---

## 13. File plan (implementation)

| File | Status |
|------|--------|
| `spammm/GUI/ReactionCoordinateExtension.py` | **Done** — panel, import/sync, run, load/save, slider, bond Δ viz, ESP controls |
| `spammm/GUI/rc_esp_view.py` | **Done** — blitted ESP animation synced to slider |
| `spammm/GUI/mpl_blit.py` | **Done** — reusable Qt matplotlib blit helper |
| `spammm/GUI/rc_scan_gui_script.py` | **Done** — programmatic review setup |
| `demos/gui_scripts/rc_scan_review.py` | **Done** — `./run_gui.sh --script …` |
| `demos/gui_scripts/rc_scan_offline.py` | **Done** — headless DFTB path |
| `spammm/quantum/esp_grid.py` | **Done** — ESP stack precompute |
| `spammm/quantum/DFTB_utils.py` | **Done** — Mulliken parse, relax diagnostics, `return_charges` |
| `spammm/topology/hbond_utils.py` | **Done** |
| `spammm/topology/scan_dataset.py` | **Done** — incl. `charges`, `esp_xy` |
| `spammm/topology/scan_kekule.py` | **Done** — `analyze_kekule_cc`, `cc_length_vs_control` |
| `spammm/quantum/coordinate_scan.py` | **Done** — pm-NEB relaxed + Mulliken per frame |
| `spammm/topology/MoleculeEditorBackend.py` | **Done** — `constraint_set` |
| `spammm/GUI/ExtensionManager.py` | **Done** — register `reaction_coord` |
| `tests/topology/test_scan_dataset.py` | **Done** — incl. slow `test_pm_neb_relaxed_dftb` |
| `tests/GUI/test_rc_scan_gui_script.py` | **Done** — offscreen script smoke |
| `doc/Topics/ReactionCoordinateScan.md` | **Done** — topical OKF doc |
| `doc/Takeways.md` | **Done** — blit + geometry + DFTB pitfalls |
| `spammm/GUI/VispyUtils.py` | **Later** — PBC replicas, bond coloring |
| `spammm/forcefields/FFController.py` | **Later** — read pins from `constraint_set` |
| `tests/topology/test_coordinate_scan.py` | Optional — slow DFTB (see `test_hbond_scan.py`) |

Keep `hbond_scan.py` for ASCII pipeline and existing tests.

---

## 14. Remaining open points

1. **Background DFTB in GUI:** thread vs subprocess — avoid blocking UI during Run scan.
2. **PBC priority:** after first gas-phase demo (agreed).
3. **Auto-pin junction atoms** in Phase C: default on or opt-in?
4. **Endpoint relax** before pm-NEB interpolation: DFTB opt vs skip (currently skip).

---

## 15. Summary

`ReactionCoordinateExtension` is a **standalone plugin** that:

- Detects H-bonds on real **`AtomicGraph`** assemblies  
- Uses **`backend.constraint_set`** pinning (RC pin mode)  
- Builds **m-dimensional** control grids with explicit **n→m mapping**  
- Stores **`controls`** only; **`fractions` derived** at load/build time  
- Runs **rigid DFTB** or **pm-NEB (all-atom interp + SP)** → **`ScanDataset.npz`**  
- **Kekulé / C–C analysis** on trajectories via `scan_kekule.py` (Phase B)  
- Explores via **preview sliders**; **Sync to Graph** writes **full frame**  
- Defers **Vispy bond coloring** and **PBC visualization** to later phases  
- Sequences **rigid → pm-NEB + Kekulé → partial relax → (future) NEB**

Next: GUI polish (non-blocking DFTB, multi-H-bond mapping UI, energy plot widget).

---

## 16. GUI control script (efficient review)

For repeatable interactive review without manual clicking:

| File | Role |
|------|------|
| `spammm/GUI/rc_scan_gui_script.py` | `prepare_rc_scan_review(window, …)` — orchestrates widget-parity workflow |
| `spammm/GUI/gui_script_utils.py` | Expand panels, pump events, set spin/combo values |
| `tests/GUI/testplot_rc_scan_gui.py` | Launch GUI pre-configured for user review |

**Workflow (same as manual):**

1. ASCII Builder → Load example → Generate (with relax steps)
2. Reaction coordinate → Import from Graph → configure pair/dx/method
3. **Preview path** (fast, no DFTB) or **Run scan** (`--dftb`)
4. Bond-length visualization on; slider at mid-frame

```bash
./run_gui.sh --script demos/gui_scripts/rc_scan_review.py
./run_gui.sh --script demos/gui_scripts/rc_scan_review.py -- --dftb --dx 0.1
```

Script module must define `run(window, argv=None)`. Runner: `spammm/GUI/gui_script_runner.py`.

Public RC API (called by buttons and script): `import_from_graph`, `configure_scan`, `run_preview_scan`, `run_scan`, `show_scan_frame`, `enable_bond_length_visualization`.
