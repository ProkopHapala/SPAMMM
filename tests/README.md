# tests/

Test suite for SPAMMM. Three classes of scripts:

- **Class 1 — Pytest tests**: `def test_*` with `assert`, run via `pytest`
- **Class 2 — Standalone scripts**: `python tests/<script>.py`, produce plots/metrics for visual review
- **Class 3 — Utility modules**: helpers imported by Class 1 & 2, never run directly

## Files

| Script | Class | Purpose |
|--------|-------|---------|
| `conftest.py` | 3 | Pytest fixtures: data paths, molecule loader, `--update-refs` |
| `test_topology.py` | 1 | Bond/angle/hybridization/atom-type assignment |
| `test_forcefield.py` | 1 | UFF/SPFF geometry optimization, NVE invariants, force-energy correspondence |
| `test_surface.py` | 1 | Ewald2D vs brute-force, surface GridFF, lateral scans |
| `test_folded_relax.py` | 1 | Rigid body relaxation + manipulation on folded basis (NaCl substrate) |
| `test_lingebra.py` | 1 | Linear algebra: eigenvalue decomposition parity |
| `test_integration.py` | 1 | Relaxed scan stubs (molecule on substrate) — TODO |
| `test_afm.py` | 3 | Re-exports pointer to `SPM/test_afm_morse.py` |
| `test_folded_surface_scan.py` | 2 | Fit folded basis to NaCl surface potential, plot fits + residuals |
| `test_tensor_parity.py` | 2 | GPU tensor kernel vs CPU numpy reference for Morse/Coulomb |
| `run_manipulation.py` | 2 | CLI: run relaxed scan, export `.xyz` trajectory movie |
| `TEST_RESULTS.md` | — | Human-readable test results log |
| `ref_data/` | — | Git-tracked reference files (`.ref.json`, `.ref.xyz`) for regression tests |

## Subfolders

| Folder | Purpose |
|--------|---------|
| `SPM/` | AFM / scanning probe microscopy tests and diagnostic plots |
| `helpers/` | Shared utility modules (parity, geometry, scan, folded rigid body) |
| `surfaces/` | Surface GridFF construction and sampling utilities |
| `forcefields/` | Forcefield-specific tests (empty — placeholder) |
| `integration/` | Integration tests (empty — placeholder) |
| `quantum/` | Quantum method tests (empty — placeholder) |
| `topology/` | Topology tests (empty — placeholder) |
