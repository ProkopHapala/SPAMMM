# quantum/

Quantum chemistry integration for electron density computation and constrained scans. Densities feed into the FDBM AFM method in `SPM/`.

- **DFTB_utils.py** — DFTB+ integration: input generation, output parsing, SK parameter management; `run_dftb_sp`, `save_xyz_movie`, `constrained_scan`, H-transfer helpers (`identify_hbond_transfer`, `make_axis_path`)
- **hbond_scan.py** — Rigid DFTB proton-transfer scan for ASCII `:` H-bond systems: H slides along donor→acceptor axis at **0.1 Å** steps (default); PNG + multi-frame XYZ with `s`, `f`, `E`, `dE` in comment lines
- **pySCF_utils.py** — pySCF RHF/DFT calculations for electron densities on 3D grids (alternative to DFTB for higher accuracy)
- **DFTB/** — DFTB+ ctypes wrapper, basis parser, GPU density projection, basis optimizer

## H-bond proton-transfer scan

**Pipeline:** `build_ascii_hbond_system(name)` → `identify_hbond_from_ascii()` → `run_hbond_transfer_scan(ds=0.1)` → `save_hbond_scan_artifacts()`

**Requires:** `DFTB_EXE`, `DFTB_SK_PATH` (validated at import of `DFTB_utils`).

**Run:**
```bash
python tests/topology/testplot_hbond_scan.py --name 2Quinolone --step 0.1
pytest tests/topology/test_hbond_scan.py::test_hbond_scan_2quinolone -s   # slow, coarse grid
```

**Artifacts:** `debug/test_hbond_scan/hbond_{name}_p{pair}.{png,xyz}` + per-point DFTB dirs `{name}_p{pair}/pt_*`

**Caveats:** rigid scan (heavy atoms fixed); DFTB SCC may fail near acceptor — `on_fail='skip'`; charge restart from previous converged point.

**Topology input:** `spammm/topology/ascii_art_heterocycle.py` — see `topology/README.md`.
