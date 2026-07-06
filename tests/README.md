# tests/

Test suite for SPAMMM. See **`doc/TEST_DESIGN.md`** (SSOT).

## Review levels

| Level | Mechanism |
|-------|-----------|
| L0 | `assert` / `TopologyDiff` / `ref_data/` |
| L1 | `.out` + `.log` in `debug/<script>/` (`--review` or `--develop`) |
| L2 | `.png` (`--visual` or `--develop`) |

## File classes

| Pattern | pytest? | Purpose |
|---------|---------|---------|
| `test_*.py` | Yes | Automatic tests (+ optional L1/L2 flags) |
| `testplot_*.py` | No | Visual demos: `python tests/...` |
| `run_*.py` | No | CLI utilities |
| `helpers/` | No | Shared utilities |

## Run

```bash
pytest -m "not slow"                                    # routine
pytest tests/topology/test_editing_ops.py --develop -s  # new feature debug
```

## Key files

| Script | Purpose |
|--------|---------|
| `conftest.py` | Fixtures, `--develop`/`--review`/`--visual` flags |
| `test_topology.py` | Bond/angle/hybridization/type assignment |
| `test_forcefield.py` | UFF/SPFF optimization, NVE |
| `test_surface.py` | Ewald, GridFF, folded function |
| `test_folded_relax.py` | Rigid body relax + manipulation |
| `topology/test_editing_ops.py` | Molecular editing (L0+L1+L2 pilot) |
| `testplot_tensor_parity.py` | GPU tensor kernel parity plots |
| `testplot_folded_surface_scan.py` | Folded basis fitting plots |
| `testplot_assembly.py` | Hexagonal SAM assembly search, clash/strain maps, XYZ export |
| `testplot_contact_surface.py` | GPU contact surface vs brute Morse (separable + PIC) |
| `SPM/test_afm_*.py` | AFM pytest tests |
| `SPM/testplot_*.py` | AFM visual diagnostics |
| `ref_data/` | Git-tracked regression references |

## Subfolders

| Folder | Purpose |
|--------|---------|
| `topology/` | Editing, Kekule, ascii art |
| `SPM/` | Scanning probe microscopy |
| `surfaces/` | GridFF utilities, contact-surface demo |
| `helpers/` | parity, geometry, review, topology_test |
