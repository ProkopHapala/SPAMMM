# topology/

Editable molecular topology and structure generation. **`AtomicGraph` is the SSOT** — rendering, force-field export (`AtomicSystem`), `PackedMolecule`, and GUI editing all derive from it.

See also: `kernels/README.md` (GPU force fields consume topology via `AtomicSystem` / packed arrays).

## Naming (editor vs Kekulé)

| Name | Role |
|------|------|
| **MoleculeEditorBackend** | Hex-grid molecular editor: atoms, rings, passivation, H-caps, export (`MoleculeEditorBackend.py`) |
| **KekulePure** | π-bond-order optimizer only (`KekulePure.py`) |

## Modules

| File | Purpose |
|------|---------|
| `AtomicGraph.py` | Object-graph: stable Atom/Bond/Ring `_id`; `format_table()` for test dumps |
| `MoleculeEditorBackend.py` | Editing engine: hex grid, passivation, rings, `_sync_sys()`, MOL/XYZ export |
| `KekulePure.py` | Kekulé π-bond optimizer: feasibility, multi-seed localization, 6-ring validation; planned input to RI density (`doc/Tasks/Kekule_ExponentialDensityFit.md`) |
| `PackedMolecule.py` | Dense NumPy snapshot of `AtomicGraph` for undo/clipboard |
| `FFparams.py` | Parse UFF/SPFF `.dat` parameter files → atom types, REQs |
| `HexGrid.py` | Hexagonal grid snapping and transforms |
| `heterocycle_generator.py` | Build heterocycles from rectangular hex-lattice specs |
| `ascii_art_heterocycle.py` | ASCII art → 2D geometry; `:` H-bond marks; `resolve_hbond_pairs()` |
| `hbond_utils.py` | H-bond records on synced graphs; RC control→fraction mapping |
| `scan_dataset.py` | `ScanDataset` `.npz` trajectories (geometry, charges, optional ESP) |
| `scan_kekule.py` | C–C bond length vs control along scan paths |

## Data flow

```
ASCII art / GUI clicks
       ↓
  AtomicGraph  ←── MoleculeEditorBackend
       ↓
  to_arrays() → AtomicSystem → SPFF/UFF MD (kernels/SPFF.cl, UFF.cl)
       ↓
  KekulePure.run_kekule_solver()  (π bond orders, optional)
```

## Kekulé workflow

1. Build or parse graph → `make_n_pi()` → `run_kekule_solver()`
2. Localization: `localize_kekule(..., validate=True, ntrials=5)`
3. H-caps: `add_capping_h_sp2(target_valence=...)`

**Tests:** `pytest tests/topology/test_kekule_pure.py`  
**Visual:** `python tests/topology/testplot_kekule.py` → `debug/testplot_kekule/`

## Editing operations (headless)

`tests/topology/test_editing_ops.py` — L0 `TopologyDiff` + optional L1/L2:

```bash
pytest tests/topology/test_editing_ops.py -s
pytest tests/topology/test_editing_ops.py --develop -s   # + .out/.log + PNG
```

Artifacts: `debug/test_editing_ops/`

| `ascii_art_heterocycle.py` | ASCII art → 2D geometry; `:` H-bond marks; `resolve_hbond_pairs()` |
| `hbond_utils.py` | H-bond records on synced graphs; RC control→fraction mapping |
| `scan_dataset.py` | `ScanDataset` `.npz` trajectories (geometry, charges, optional ESP) |
| `scan_kekule.py` | C–C bond length vs control along scan paths |

## Reaction-coordinate scan

Graph-backed H-bond scans: **`find_hbonds_graph`** + **`ScanDataset`**. Topical doc: `doc/Topics/ReactionCoordinateScan.md`. Tests: `tests/topology/test_scan_dataset.py`.

## ASCII H-bonds & proton transfer (legacy)

`spammm/quantum/hbond_scan.py` — see `spammm/quantum/README.md`  
**Tests:** `tests/topology/test_ascii_art.py`, `tests/topology/test_hbond_scan.py`

## OpenCL kernels (downstream)

Topology does not load kernels directly. After `_sync_sys()`, force evaluation uses:

| Physics | Kernel stack | Python |
|---------|--------------|--------|
| SPFF MD | `SPFF.cl` + `nonbonded.cl` + … | `forcefields/SPFF_cl.py` |
| UFF MD | `UFF.cl` + `nonbonded.cl` | `forcefields/UFF_cl.py` |
| On-surface assembly | `assembly.cl` | `forcefields/Assembly.py` |

Full index: `kernels/README.md`.
