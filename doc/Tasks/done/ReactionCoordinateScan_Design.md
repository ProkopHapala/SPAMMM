# Reaction-Coordinate Scan Extension — Design Spec (draft)

> **Superseded by** [`ReactionCoordinateExtension_Design.md`](ReactionCoordinateExtension_Design.md) (v2, discussion round 2).  
> Kept for history only.

**Status:** superseded  
**Audience:** SPAMMM GUI + topology + quantum  
**Goal:** General plugin for H-bond (and similar) coordinate scans on `AtomicGraph` assemblies, with DFTB rigid profiles, labeled XYZ movies, and interactive m-slider preview.

---

## 1. Problem statement

We can already run **rigid DFTB proton-transfer scans** on ASCII molecules with `:` H-bonds (`spammm/quantum/hbond_scan.py`, `tests/topology/testplot_hbond_scan.py`). The GUI can **draw** H-bonds (`AtomicSystem.find_hbonds` in `refresh_view`) and **pin atoms** during FF relaxation (`FFController` + `AtomScene._fixed`).

What is missing is a **unified, graph-native workflow** for real editor-built or assembly-loaded structures:

1. Detect & display H-bonds on the current `AtomicGraph` / assembly  
2. Let the user **pin/fix** atoms (reuse existing pin UX where possible)  
3. Choose which H-bond(s) to scan and map them to **m control parameter(s)** (1D shared, 2D independent, …)  
4. Run **rigid DFTB** (later: constrained relax) and save **labeled multi-frame XYZ**  
5. **Interactive preview**: m sliders animate geometry + bond-length deviation coloring (blue shortened / red elongated vs reference)

This document inventories existing code and proposes architecture **before coding**.

---

## 2. Code inventory (reuse — do not duplicate)

### 2.1 H-bond identification

| Location | What it does | Scope |
|----------|--------------|-------|
| `AtomicSystem.find_hbonds(d_max, a_min)` | Geometry: D–H…A, D/A ∈ {N,O}, angle at H | **Any** `AtomicSystem` after `_sync_sys()` — **already used in GUI** |
| `ascii_art.resolve_hbond_pairs(atoms)` | ASCII `:` marks → `(H_idx, acceptor_idx)` + donor from H neighbor | ASCII pipeline only |
| `hbond_scan.identify_hbond_from_ascii` | Thin wrapper over `resolve_hbond_pairs` | Tests / ASCII |
| `DFTB_utils.identify_hbond_transfer` | Pick one H + two closest heavy atoms by axis/mode | Legacy mol/ribbon scans |

**Recommendation:** SSOT for GUI/assembly = **`find_hbonds` on synced `backend.sys`**, enriched with stable **`Atom._id`** via `backend._atom_ids`. Wrap in one function:

```python
# spammm/topology/hbond_utils.py  (new, small)
def find_hbonds_graph(backend) -> list[HbondRecord]:
    """Return list of HbondRecord with graph IDs + dense indices."""
```

`HbondRecord`: `donor_id`, `h_id`, `acceptor_id`, optional `(d,h,a)` dense indices, `dist_ha`, `angle_dha`, label string.

ASCII `resolve_hbond_pairs` stays for ASCII builder; GUI path uses geometry.

### 2.2 Transfer path (rigid)

| Location | What it does |
|----------|--------------|
| `hbond_scan.make_hbond_transfer_path` | H slides on D→A axis; grid `ds=0.1` Å; returns `fractions`, `s_axis`, `path` |
| `hbond_scan.run_hbond_transfer_scan` | Loop `run_dftb_sp` + charge restart; returns energies + path |
| `DFTB_utils.make_axis_path` | Generic axis interpolation (legacy) |
| `DFTB_utils.constrained_scan` | **Relaxed** scan: `moved_idx`, `path`, `fixed_idx`, PBC — for phase 2 |

**Recommendation:** Rename/generalize `hbond_scan.py` → **`spammm/quantum/coordinate_scan.py`**:

- Keep `make_hbond_transfer_path(h_idx, donor, acceptor, …)` as one **path generator**
- Add **`build_scan_grid(mapping, ds)`** for m-dimensional control grid
- Add **`run_rigid_dftb_scan(enames, apos0, frames, sk_set, …)`** — single loop calling existing `run_dftb_sp`
- **`write_scan_xyz`** — thin alias over `DFTB_utils.save_xyz_movie` (already used by `write_hbond_scan_xyz`)

Do **not** copy XYZ movie writer or DFTB input logic.

### 2.3 Fixed / pinned atoms

| Mechanism | Layer | Purpose |
|-----------|-------|---------|
| `Atom.pin` | `AtomicGraph` | **Hex grid** node key — *not* spatial constraint |
| `FFController.set_pinned` / `pin_selected` / `get_pinned_mask` | FF | Harmonic pin during **GPU MD** |
| `AtomScene._fixed`, `set_fixed_mask`, `toggle_fixed` | Vispy | Visual + skip in picking; wired from **FFExtension** |
| `makeDFTBjob_pbc(fixed_atoms=…)` / `constrained_scan(fixed_idx=…)` | DFTB | **DFTB+ MovedAtoms** constraint for relaxation |
| Rigid DFTB scan | — | Implicit: only update H coords; all else unchanged |

**Gap:** Pin state lives on **`FFController`**, not on **`MoleculeEditorBackend` / `AtomicGraph`**. Scan extension needs pins **without building FF**.

**Recommendation (phase 1):**

- Add **`ConstraintSet`** (or `backend.scan_constraints`) on the **window** or **backend**:
  - `fixed_atom_ids: set[int]`  — `Atom._id` values user pinned for scan
  - Sync to `scene.set_fixed_mask` for display (reuse FFExtension pin/unpin UI or duplicate minimal buttons in scan panel)
- Rigid scan: `fixed_atom_ids` = all atoms except moving H(s) **plus** user pins (redundant but explicit)
- Phase 2 relaxed scan: map `fixed_atom_ids` → dense `fixed_idx` for `constrained_scan`

**Do not** overload `Atom.pin` (hex grid semantics differ).

### 2.4 GUI infrastructure

| Location | Pattern |
|----------|---------|
| `ExtensionManager` + `build_ui(window) → UIComponents` | Lazy-loaded panel + optional edit/view modes |
| `KekuleExtension`, `FFExtension`, `AsciiArtExtension` | Reference implementations |
| `SPAMMM_GUI.refresh_view()` | Central render; already draws H-bonds, bond orders |
| `VispyUtils.set_bond_orders` | Colored bond lines by π order — **pattern for X–H length coloring** |

**Recommendation:** New extension **`spammm/GUI/ScanExtension.py`**, registry key `'scan'` (or `'rc_scan'`).

### 2.5 Assembly / self-assembly

| Location | Notes |
|----------|-------|
| `forcefields/Assembly.py` | Rigid packing of **multiple copies** of a monomer on hex cell |
| `AssemblyPlot.write_assembly_xyz` | Multi-molecule XYZ export |
| Editor `AtomicGraph` | Can hold full assembly after import or manual build |

H-bonds **between** molecules in an assembly are found by `find_hbonds` on the combined system — no special assembly API needed for detection. Scan paths are per H-bond instance (each bridging H identified separately).

---

## 3. Proposed architecture

```
AtomicGraph (SSOT)
       │
       ├─ MoleculeEditorBackend._sync_sys() → AtomicSystem
       │
       ├─ hbond_utils.find_hbonds_graph(backend) → [HbondRecord]
       │
       ├─ ConstraintSet (fixed Atom._id set) ← GUI pin mode
       │
       └─ ScanDefinition
              ├─ hbonds: [HbondRecord] selected for scan
              ├─ mapping: (n_hbonds × m_controls) or list[f_i = u_k]
              ├─ grid: per-control ranges + step ds
              └─ method: 'rigid_dftb' | (later 'constrained_dftb')

coordinate_scan.build_frames(apos0, scan_def) → list[apos]
coordinate_scan.run_rigid_dftb_scan(...) → ScanResult
coordinate_scan.write_scan_xyz(...) → save_xyz_movie

ScanExtension (GUI)
       ├─ list/detect hbonds, checkboxes
       ├─ mapping UI (m controls, assign hbond → control)
       ├─ pin/unpin (reuse scene fixed mask)
       ├─ Run DFTB / Preview sliders
       └─ matplotlib energy plot (reuse plot_hbond_scan pattern)
```

**Separation of concerns (SoC):**

| Module | Role |
|--------|------|
| `topology/hbond_utils.py` | Discovery + `HbondRecord` + graph ID resolution |
| `topology/scan_constraints.py` | Optional: `ConstraintSet` if not on backend |
| `quantum/coordinate_scan.py` | Path math, grid, DFTB loop, XYZ export (no Qt) |
| `GUI/ScanExtension.py` | Panel, sliders, calls into above |
| `MoleculeEditorBackend` | Optional: `get_reference_geometry()` snapshot — or store on window |

---

## 4. Data model

### 4.1 `HbondRecord`

```python
@dataclass
class HbondRecord:
    index: int           # 0..n-1 in current detection list
    donor_id: int        # Atom._id
    h_id: int
    acceptor_id: int
    donor_idx: int       # dense, ephemeral — valid after _sync_sys
    h_idx: int
    acceptor_idx: int
    dist_ha: float
    angle_dha: float
    label: str           # e.g. "N12-H45...O3"
```

### 4.2 `ScanDefinition`

```python
@dataclass
class ScanDefinition:
    hbonds: list[HbondRecord]      # selected subset
    n_controls: int                # m
    # mapping[i] = control index (0..m-1) driving hbond i fraction
    mapping: list[int]             # len = n_hbonds
    control_ranges: list[tuple]    # m entries: (0, 1) or (s_min, s_max) in Å along axis
    ds: float = 0.1                # grid step per control (1D) or tensor product in m-D
    r_xh: float = 1.01
    fixed_atom_ids: set[int]       # user pins
    sk_set: str | None = None
    rigid: bool = True
```

**Mapping examples (user req. 3):**

| Case | n hbonds | m | mapping | Grid |
|------|----------|---|---------|------|
| Single H-bond 1D | 1 | 1 | `[0]` | `u0 ∈ [0,1]` step 0.1 |
| Two H-bonds, same coordinate | 2 | 1 | `[0, 0]` | both `f_i = u0` |
| Two H-bonds, 2D | 2 | 2 | `[0, 1]` | `(u0,u1)` tensor product |
| Three H-bonds, 2 controls | 3 | 2 | `[0, 0, 1]` | hbonds 0,1 share u0; hbond 2 uses u1 |

For m=2 with ds=0.1: ~11×11 ≈ 121 DFTB points — acceptable for small m,n; warn in UI if product > threshold (e.g. 200).

Each hbond fraction `f_i ∈ [0,1]` maps to H position via existing `make_hbond_transfer_path` (store `h0`, `h1` per hbond at setup time).

### 4.3 `ScanResult`

```python
{
  'controls': ndarray (n_frames, m),   # control values per frame
  'fractions': list[ndarray],          # per-hbond f_i per frame
  'frames': list[apos],                # or build on demand
  'energies_ev': ndarray,
  'rel_ev': ndarray,
  'hbonds': [...],
  'reference_apo': apos0,              # for bond-length diff coloring
}
```

XYZ comment line (extend existing):

```
u0=0.300 u1=0.000 f0=0.300 f1=0.300 E=-867.12 dE=0.05
```

---

## 5. Feature spec (mapped to user list)

### 5.1 Detect H-bonds in assembly + draw in Vispy

**Existing:** `refresh_view` calls `sys.find_hbonds`, draws purple H…A segments.

**Enhancements:**

- [ ] Run detection once on panel open / "Refresh H-bonds" → populate list with labels (`HbondRecord.label`)
- [ ] Draw **D–H** and **H…A** separately (dashed vs dotted) or label H with hbond index on hover
- [ ] Store `window._hbond_records` for scan extension (invalidate on `_sync_sys` / geometry edit)
- [ ] Toggle "show H-bonds" checkbox (hide without deleting records)

**Reuse:** `find_hbonds`, `scene.hbond_lines`, `_line_set`.

### 5.2 Pin / fix atoms

**Existing:**

- FF panel: Pin Selected / Unpin All / pin_unpin edit mode → `FFController` + `scene.set_fixed_mask`
- Vispy: fixed atoms excluded from drag (see `is_fixed` in pick handlers)

**Plan:**

- [ ] **Option A (minimal):** Scan panel buttons call same `FFExtension.handle_pin_click` / `pin_selected` — requires FF built (heavy)
- [ ] **Option B (preferred):** **`ConstraintSet` on window** independent of FF:
  - `window.scan_fixed_ids: set`
  - Pin mode toggles membership + `scene.set_fixed_mask` from backend indices
  - No GPU FF required for scan setup
- [ ] Persist pins until user clears or loads new structure
- [ ] Rigid DFTB: pins informational only (geometry frozen anyway except scan H's); matter for phase-2 relax

**Future home:** `MoleculeEditorBackend` method `set_fixed_atoms(ids)` if we want SSOT on backend — optional; window-level OK for v1.

### 5.3 Select H-bonds + control mapping (n, m)

**UI sketch:**

```
H-bonds detected: 3
 [x] #0  N5-H18...O7   (1.72 Å, 165°)
 [x] #1  N8-H19...O6   (1.68 Å, 158°)
 [ ] #2  ...

Controls m: [1 ▼]  (1–5)

Mapping:
  H-bond #0 → control [0 ▼]
  H-bond #1 → control [0 ▼]

Step ds: 0.1 Å   Grid: 1D (11 pts)  [Run DFTB]
```

For m=2, add second slider range + 2D grid size readout.

**Logic:** `coordinate_scan.build_control_grid(m, ranges, ds)` → array of shape `(n_frames, m)`; for each row, compute each hbond's `f_i` from `mapping` and control values (for shared mapping: same control value drives multiple f's).

### 5.4 DFTB rigid + labeled XYZ

**Reuse verbatim:**

- `run_dftb_sp(..., maxscc=400, restart_charges_from=...)`
- `save_xyz_movie` via `write_scan_xyz`
- Charge restart along path order (important for SCC)

**Workflow:**

1. `backend._sync_sys()` → `apos0`, `enames`
2. Build frames (preview without DFTB)
3. On "Run DFTB": background thread or subprocess queue (GUI must not freeze — follow `DFTBExtension` / AFM patterns)
4. Save to user-chosen path or `debug/scan_<timestamp>/scan.xyz`
5. Optional PNG: generalize `plot_hbond_scan` → `plot_scan_energy(result, x_control_index=0)`

### 5.5 Interactive m-sliders + bond-length coloring

**Without DFTB (preview mode):**

- m sliders (QSlider + QLabel) bound to control values
- On change: apply frame to `scene.update_positions` **or** temporary overlay positions (don't mutate graph until "Apply" — **discuss**)
- Compute bond lengths for each **scan-relevant** X–H and H…X pair vs `reference_apo`:
  - `Δr = r - r_ref`
  - Color bond segment: blue if Δr < 0, red if Δr > 0, magnitude → saturation/width
- Reuse `bond_colored_lines` or add `set_bond_length_deviation(bonds, delta_r)`

**With precomputed DFTB grid:**

- Sliders snap to nearest computed grid point; show E from cache
- 2D: two sliders + optional contour plot in panel (matplotlib widget or popup)

**Reference geometry:** equilibrium `apos0` at scan setup (or minimum-E frame after DFTB — later).

---

## 6. GUI extension registration

```python
# ExtensionManager.py
'scan': dict(
    module='spammm.GUI.ScanExtension',
    dependencies=[],
    req_paths=[],  # DFTB validated at run time via DFTB_utils import
    build_ui='build_ui',
),
```

Panel sections:

1. **Detect** — refresh, list, visibility  
2. **Constraints** — pin mode, clear pins, fixed count  
3. **Scan setup** — select hbonds, m, mapping, ds, SK set  
4. **Compute** — Run DFTB (rigid), progress, save XYZ  
5. **Explore** — m sliders, E readout, bond deviation legend  

Optional edit mode: `'Pick H-bond'` (click H to toggle selection).

---

## 7. Phased implementation

| Phase | Deliverable | Depends on |
|-------|-------------|------------|
| **P0** | `hbond_utils.find_hbonds_graph` + GUI list + improved drawing | None |
| **P1** | `ConstraintSet` + pin UI in scan panel (no FF) | P0 |
| **P2** | Refactor `hbond_scan` → `coordinate_scan` with mapping + m-D grid | P0 |
| **P3** | `ScanExtension` preview sliders + bond coloring | P1, P2 |
| **P4** | DFTB run + XYZ + energy plot in GUI | P2, DFTB_EXE |
| **P5** | 2D scan UI + cached energy lookup | P4 |
| **P6** | Constrained relax (`constrained_scan`) | P4, pins → fixed_idx |

**Tests (TDD):**

- L0: `find_hbonds_graph` on known dimer geometry (from editing_ops fixtures)
- L0: `build_control_grid` mapping cases (shared vs independent)
- L0: frame geometry respects pin constraints (pinned atoms unchanged)
- Slow: rigid DFTB 1D reuse existing `test_hbond_scan` patterns

---

## 8. Open questions for discussion

1. **Preview mutates graph?** Should slider preview update `AtomicGraph` positions temporarily, or only Vispy overlay? (Overlay safer for undo.)

2. **FF pin reuse vs independent ConstraintSet?** Independent avoids requiring FF build; duplicate pin UX slightly.

3. **Maximum grid size guard?** Auto-coarsen ds if m-D product > N_max?

4. **Assembly with PBC:** Rigid cluster OK in gas-phase DFTB; periodic assembly needs `run_pbc` / `constrained_scan` — defer or support early?

5. **Simultaneous scan of 2+ H on same molecule:** Any exclusion if H-bonds share atoms? (Likely invalid — validate at setup.)

6. **Extension name:** `ScanExtension` vs `ReactionCoordinateExtension` vs `HbondScanExtension`?

7. **Bond-length coloring scope:** Only X–H bonds involved in selected scans, or all H-bonds?

8. **Kekule coupling:** Scan needs 3D geometry only; Kekule optional for FF afterward — confirm no solver in loop for v1.

---

## 9. Files to touch (when implementing)

| Action | File |
|--------|------|
| **New** | `spammm/topology/hbond_utils.py` |
| **New** | `spammm/quantum/coordinate_scan.py` (evolve from `hbond_scan.py`) |
| **New** | `spammm/GUI/ScanExtension.py` |
| **Edit** | `ExtensionManager.py` — register `'scan'` |
| **Edit** | `SPAMMM_GUI.refresh_view` — optional hbond labels / richer draw |
| **Edit** | `VispyUtils` — bond deviation coloring helper |
| **Keep** | `hbond_scan.py` as thin re-export during migration, then remove |
| **Tests** | `tests/topology/test_coordinate_scan.py`, `tests/topology/test_hbond_utils.py` |

---

## 10. Summary

Most building blocks **exist** but are **fragmented**:

- Detection + draw: **GUI already** (`find_hbonds`)  
- Path + DFTB + XYZ: **`hbond_scan` + `DFTB_utils.save_xyz_movie`**  
- Pins: **FF + Vispy**, not graph-native  
- m-D mapping + interactive preview: **missing**

The general tool should be a **thin GUI extension** over a **small quantum module** (`coordinate_scan`) and **topology helper** (`hbond_utils`), not a fork of the ASCII test script. Generalize path/mapping first; keep DFTB and XYZ as shared utilities.

**Next step:** Agree on open questions (especially preview overlay vs graph mutation, ConstraintSet placement, and 2D grid limits), then implement P0–P2 before GUI polish.
