# quantum/

Quantum chemistry integration: DFTB+ scans, Hessians for vibrations, electron densities for FDBM AFM.

- **DFTB_utils.py** — DFTB+ I/O, `run_dftb_sp` / `run_dftb_relax`, Mulliken parse, Hessian, `clean_dftb_workdir`, failure diagnostics
- **coordinate_scan.py** — Reaction-coordinate engine: control grids, pm-NEB (endpoint relax + all-atom interp), Mulliken charges per frame → `ScanDataset`
- **esp_grid.py** — Precompute Coulomb ESP stacks `[nframes, ny, nx]` from charges (KE/r, same as QEq)
- **hbond_scan.py** — Legacy ASCII rigid proton-transfer scan (0.1 Å axis steps); kept for existing tests
- **pySCF_utils.py** — pySCF RHF/DFT grid densities
- **DFTB/** — ctypes wrapper, basis parser, GPU density projection (dense NA DM default for FDBM), basis optimizer — see `DFTB/README.md`

## Reaction-coordinate scan (graph / GUI)

**SSOT:** `coordinate_scan.py` + `topology/scan_dataset.py`. Topical doc: `doc/Topics/ReactionCoordinateScan.md`.

```bash
PYTHONPATH=. python3 spammm/GUI/gui_scripts/rc_scan_offline.py
pytest tests/topology/test_scan_dataset.py::test_pm_neb_relaxed_dftb -s
./run_gui.sh --script spammm/GUI/gui_scripts/rc_scan_review.py
```

**Requires:** `DFTB_EXE`, `DFTB_SK_PATH`.

**Caveats:** GUI geometry must use `build_ascii_hbond_system` (see `doc/Takeways.md`); pm-NEB relaxed runs Mulliken SP on every interpolated frame.

## H-bond proton-transfer scan (ASCII legacy)

**Pipeline:** `build_ascii_hbond_system(name)` → `run_hbond_transfer_scan(ds=0.1)` → `save_hbond_scan_artifacts()`

```bash
python tests/topology/testplot_hbond_scan.py --name 2Quinolone --step 0.1
pytest tests/topology/test_hbond_scan.py -s
```

**Artifacts:** `debug/test_hbond_scan/hbond_*.{png,xyz}`
