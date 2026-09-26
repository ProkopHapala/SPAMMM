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
| `test_forcefield.py` | UFF/SPFF optimization and NVE; PairFF replica clash and concurrent-launch parity |
| `test_relax_serial.py` | SPFF serial vs batch parity (L0) |
| `test_relax_flat1.py` | flat_1 PAH vacuum/substrate timing (`--develop`) |
| `test_relax_ptcda_faf.py` | PTCDA+FAF: SPFF/UFF fused + **LFF** topology/sweep (`--develop`) |
| `test_surface.py` | Ewald, GridFF, folded function |
| `test_folded_relax.py` | Rigid body relax + manipulation |
| `topology/test_editing_ops.py` | Molecular editing (L0+L1+L2 pilot) |
| `testplot_tensor_parity.py` | GPU tensor kernel parity plots |
| `testplot_folded_surface_scan.py` | Folded basis fitting plots |
| `testplot_assembly.py` | Hexagonal SAM assembly search, clash/strain maps, XYZ export |
| `testplot_pairff_energy_mc.py` | PairFF+FAF rigid-body greedy MC assembly (8 mols, multi-species, charge colors, GIF trajectory) |
| `testplot_contact_surface.py` | GPU contact surface vs brute Morse (separable + PIC) |
| `topology/testplot_ribbon.py` | Thin driver over `spammm/topology/ribbon_pbc.py`: PBC zigzag GNRs, N-terminated edges (`--widths --ncells --passivation --dftb`); `--two` = two-ribbon N···H-N junction cell (`--bottom N --top NH --dda --state`); `--scan-ly` = d_DA lattice-y scan (3-pt parabola + p↔d parity, `--relax` for ionic relax) → `debug/ribbon/` |
| `topology/testplot_bond_order.py` | π bond-order maps from DFTBcore DM over corner-scan states → `debug/test_bond_order/` |
| `topology/testplot_muH.py` | Per-site H chemical potentials via DFTB3/3ob: monomer μ1/μ2/J/μ2H + dimer calibration + closed-shell 2H matrix → `debug/muH/`; results reusable via `debug/muH/results.json` (see `doc/ERC_private/muH_monomers.md`) |
| `topology/testplot_mol_flakes.py` | Symmetric PAH flake generator (row profiles / coronene / triangulene + ASCII-art EXT fillers) + per-site edge chemistry (`C/c/N/n/O/o`) → `debug/mol_flakes/` PNG+XYZ+w×h overview; mirror-symmetry checked per flake (see `doc/ERC_private/task_Molecules.md`) |
| `topology/testplot_mol_flakes_dftb.py` | DFTB battery on the 9-backbone flake set: 3 chemistries × 5 states SP (162 jobs) → `debug/mol_flakes_dftb/`; energy report/lines, electron-reservoir plot (φ_Au/φ_Gr), `--maps` per-backbone π-BO + relaxed bond-length maps (Aro/Qui/Mixed columns) |
| `topology/run_pyscf_b3lyp.py` | B3LYP recalculation of jobs exported by `--export-pyscf` (jobs.csv manifest → per-job npz: S, h1e, fock, dm, mo_energy; resumable results.csv) |
| `quantum/testplot_unfold.py` | Ribbon band structure + supercell unfolding: x1/x2/x4 canonical overlay (replicas, extended-zone check, weight dots), `--passiv` edge-chemistry comparison (w-encoded filenames), `--switch` ref-vs-hydrogenated per C/N/O system, `--sys3` relaxed 1-site-hydrogenated supercells → `debug/unfold/` (+ `index.html` gallery; doc `doc/TopicalAudit/BandUnfolding_Ribbons.md`) |
| `SPM/test_afm_*.py` | AFM pytest (Morse + FDBM; FAST_S3 parity) |
| `SPM/bench_fdbm.py` | Headless FDBM timing (`SPAMMM_AFM_BENCH`); see `doc/Tasks/PerfBenchmark_FDBM.md` |
| `SPM/testplot_*.py` | AFM visual diagnostics |
| `ref_data/` | Git-tracked regression references |

## Subfolders

| Folder | Purpose |
|--------|---------|
| `topology/` | Editing, Kekule, ascii art |
| `quantum/` | PME charge rings (pytest), band unfolding (`testplot_unfold.py`) |
| `SPM/` | Scanning probe microscopy |
| `surfaces/` | GridFF utilities, contact-surface demo |
| `helpers/` | parity, geometry, review, topology_test |
