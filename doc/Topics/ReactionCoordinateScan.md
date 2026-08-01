---
type: TopicalAudit
title: Reaction-Coordinate Scan (H-bond transfer)
tags: [reaction-coordinate, hbond, dftb, pm-neb, gui, scan-dataset]
---

# Reaction-Coordinate Scan (H-bond transfer)

## Summary

Explores proton-transfer (or shared-H) pathways on **`AtomicGraph`** assemblies: detect bridging H-bonds, drive one or more control parameters **u ∈ [0,1]**, optionally relax isomers with DFTB+ and interpolate **all atom** coordinates (poor-man's NEB). Trajectories are stored in **`ScanDataset`** (`.npz`); the GUI extension scrubs frames on a slider, colors bonds by Δlength, and animates Coulomb ESP from **Mulliken charges** (same KE/r model as QEq).

**SSOT for compute:** `spammm/quantum/coordinate_scan.py`. **SSOT for I/O:** `spammm/topology/scan_dataset.py`. **SSOT for H-bond discovery on edited graphs:** `spammm/topology/hbond_utils.py`. **GUI:** `spammm/GUI/ReactionCoordinateExtension.py` (`reaction_coord`). **Design spec:** [Tasks/ReactionCoordinateExtension_Design.md](../Tasks/ReactionCoordinateExtension_Design.md).

ASCII-only rigid scans remain in `hbond_scan.py` (0.1 Å axis steps) — use `coordinate_scan` for graph-backed workflows.

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `spammm/quantum/coordinate_scan.py` | **active** | Grids, frame builder, rigid DFTB, pm-NEB (relax + interp + Mulliken SP) |
| `spammm/topology/scan_dataset.py` | **active** | `.npz` I/O: `apos`, `controls`, `bond_len`, `charges`, optional `esp_xy` |
| `spammm/topology/hbond_utils.py` | **active** | `HbondRecord`, `find_hbonds_graph`, `controls_to_fractions` |
| `spammm/topology/scan_kekule.py` | **active** | C–C length vs control on trajectories (library; not GUI yet) |
| `spammm/quantum/esp_grid.py` | **active** | Precompute ESP stack `[nframes, ny, nx]` from Mulliken charges |
| `spammm/quantum/DFTB_utils.py` | **active** | `run_dftb_sp(return_charges)`, `run_dftb_relax`, `parse_mulliken_charges` |
| `spammm/quantum/hbond_scan.py` | **active** | Legacy ASCII rigid scan; kept for existing tests |
| `spammm/GUI/ReactionCoordinateExtension.py` | **active** | Panel + slider + bond viz + ESP controls |
| `spammm/GUI/rc_esp_view.py` | **active** | Blitted ESP animation synced to RC slider |
| `spammm/GUI/mpl_blit.py` | **active** | Reusable Qt matplotlib blit helper |
| `spammm/GUI/rc_scan_gui_script.py` | **active** | Programmatic setup for review scripts |
| `demos/gui_scripts/rc_scan_review.py` | **active** | `./run_gui.sh --script …` entry |
| `demos/gui_scripts/rc_scan_offline.py` | **active** | Same geometry path as tests, no Qt |
| `tests/topology/test_scan_dataset.py` | **active** | L0 + slow `test_pm_neb_relaxed_dftb` |
| `tests/GUI/test_rc_scan_gui_script.py` | **active** | Offscreen GUI script smoke |

## Pipeline

```
AtomicGraph / build_ascii_hbond_system  →  import_from_graph (GUI)
    → find_hbonds_graph → mapping (symmetric: all H-bonds share u)
    → method:
        • rigid DFTB     — H moves, heavy atoms fixed, DFTB SP each frame
        • pm-NEB SP      — rigid endpoints, interp all atoms, DFTB SP each frame
        • pm-NEB relaxed — DFTB opt u=0 & u=1, interp all atoms, Mulliken SP each frame
    → ScanDataset (charges [nframes, natoms], optional esp_xy)
    → GUI slider → Vispy scene + optional ESP blit animation
```

**Fractions not stored:** `controls[k,j]` + `meta.mapping` → `controls_to_fractions()` → per-H-bond **f** at frame *k*.

## GUI (`reaction_coord` extension)

| Control | Action |
|---------|--------|
| **Import from Graph** | Snapshot `backend.sys.apos` + refresh H-bonds |
| **All H-bonds (symmetric)** | One shared **u** for all bridging H (dimer scans) |
| **Relax endpoints (DFTB)** | Used with pm-NEB relaxed |
| **dx** | Control grid step (default 0.2) |
| **Method** | rigid DFTB / pm-NEB SP / pm-NEB relaxed |
| **Run scan** | DFTB work dir: `debug/rc_scan/` |
| **Preview path** | Rigid endpoints, no DFTB |
| **Save / Load npz** | `ScanDataset` round-trip |
| **Slider** | Frame index; updates 3D view + open ESP window |
| **ESP z / grid / ESP animation** | Precompute Coulomb map; blit viewer (`mpl_blit.py`) |

### GUI control script

```bash
# Full DFTB relaxed path + cache + ESP animation
./run_gui.sh --script demos/gui_scripts/rc_scan_review.py

# Replay cached npz (after one full run)
./run_gui.sh --script demos/gui_scripts/rc_scan_review.py -- --preview
```

Cache: `debug/testplot_rc_scan_gui/{name}_sym_pm_neb_relaxed.npz`

### Offline (no Qt)

```bash
PYTHONPATH=. python3 demos/gui_scripts/rc_scan_offline.py
pytest tests/topology/test_scan_dataset.py::test_pm_neb_relaxed_dftb -s
```

## ScanDataset fields

| Array | Shape | Notes |
|-------|-------|-------|
| `apos` | `[nframes, natoms, 3]` | Full geometry per frame |
| `controls` | `[nframes, m]` | Independent RC parameters |
| `bond_len` | `[nframes, nbonds]` | Derived from `apos` |
| `energies_ev` | `[nframes]` | NaN if not computed |
| `charges` | `[nframes, natoms]` | Mulliken (DFTB `detailed.out`) |
| `esp_xy` | `[nframes, ny, nx]` | Optional precomputed ESP (eV) |
| `meta` | JSON | `scan_type`, `mapping`, `hbond_records`, `endpoints_relaxed`, ESP grid params |

## Parity / validation

| Check | Reference | Tolerance |
|-------|-----------|-----------|
| H-only rigid preview | Heavy atoms identical frame 0 vs mid | exact |
| pm-NEB relaxed | Heavy bond Δ at mid frame | > 0 (qualitative) |
| Mulliken sum | Neutral molecule | \|Σq\| < 0.05 e |
| GUI vs offline geometry | `build_ascii_hbond_system('2Quinolone')` | same `natoms` |
| Endpoints relaxed flag | Both endpoint energies finite | metadata honest |

## Open Issues

- **Geometry SSOT:** GUI review scripts must use `build_ascii_hbond_system`, not stripped ASCII text alone — see [Takeways.md](../Takeways.md).
- **pm-NEB relaxed cost:** Mulliken SP on every interpolated frame (~6× DFTB for default dx).
- **Kekulé along path:** `scan_kekule.py` implemented; not wired to GUI yet.
- **Constraint pins:** `rc_pin` mode toggles `backend.constraint_set`; not yet passed into DFTB relax.
- **Legacy overlap:** `hbond_scan.py` (0.1 Å axis grid) vs `coordinate_scan` (control grid dx) — two scan drivers until unified CLI.

## Related

- [ReactionCoordinateExtension_Design.md](../Tasks/ReactionCoordinateExtension_Design.md) — full spec
- [Takeways.md](../Takeways.md) — blit caveats, GUI vs test geometry, DFTB failure diagnostics
- [TEST_DESIGN.md](../TEST_DESIGN.md) — L0/L1/L2 review levels
- `spammm/GUI/mpl_blit.py` — matplotlib blit with Qt
