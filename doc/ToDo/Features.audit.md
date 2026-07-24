
# SPAMMM Repository Audit — Comprehensive Inventory

## 1. Architecture Overview

SPAMMM (FireCore) is a Python + PyOpenCL scientific simulation platform for **AFM/STM SPM simulation pipelines**. The architecture follows a staged modular design:

```
Topology (AtomicGraph SSOT) → Type Assignment → Force Fields → Surface Interactions → AFM/STM → QM Integration
```

**Tech stack:** Python orchestration + OpenCL `.cl` kernels (GPU compute), NumPy (CPU), VisPy (GUI), DFTB+ & pySCF (QM backends). No C++/JS/Fortran/CUDA.

---

## 2. Module Inventory by Category

### 2.1 Molecular Topology (✅ Working)

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `AtomicGraph.py` | ✅ Active | [test_topology.py](cci:7://file:///home/prokop/git/SPAMMM/tests/test_topology.py:0:0-0:0) (6 tests) | SSOT for molecular topology |
| `KekulePure.py` | ✅ Active | [test_kekule.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_kekule.py:0:0-0:0), [testplot_kekule.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/testplot_kekule.py:0:0-0:0) | Bond order solver, multi-seed localization |
| `MoleculeEditorBackend.py` | ✅ Active | [test_editing_ops.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_editing_ops.py:0:0-0:0) (42 tests) | Hex-grid editing, passivation, ring ops (edge + corner + hex) |
| `ascii_art_heterocycle.py` | ✅ Active | [test_ascii_art.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_ascii_art.py:0:0-0:0), [test_heterocycle_generator.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_heterocycle_generator.py:0:0-0:0) | ASCII → 2D geometry |
| `FFparams.py` | ✅ Active | Implicit via FF tests | Atom type assignment (sp1/sp2/sp3) |
| `scan_dataset.py` | ✅ Active | [test_scan_dataset.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_scan_dataset.py:0:0-0:0) | Trajectory I/O for RC scans |
| `hbond_utils.py` | ✅ Active | [test_hbond_scan.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_hbond_scan.py:0:0-0:0) | H-bond detection |

### 2.2 Force Fields (⚠️ Partially Working)

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| [UFF_cl.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/UFF_cl.py:0:0-0:0) | ⚠️ Mostly works | [test_forcefield.py](cci:7://file:///home/prokop/git/SPAMMM/tests/test_forcefield.py:0:0-0:0) | H2O/benzene relax pass, CH4 fails (bond assertion), energy=0 bug (nonbonded disabled by default), NVE shape mismatch bug |
| [SPFF_cl.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/SPFF_cl.py:0:0-0:0) | ⚠️ Partially | [test_forcefield.py](cci:7://file:///home/prokop/git/SPAMMM/tests/test_forcefield.py:0:0-0:0) | EF correspondence ✅ (pi-sigma bug fixed), relaxation stubs only, no non-bonded tests |
| [UFFbuilder.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/UFFbuilder.py:0:0-0:0) | ✅ Active | Implicit | UFF parameter assignment |
| [SPFFbuilder.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/SPFFbuilder.py:0:0-0:0) | ✅ Active | Implicit | SPFF topology builder |
| [RigidBodyDynamics.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:0:0-0:0) | ✅ Active | [test_folded_relax.py](cci:7://file:///home/prokop/git/SPAMMM/tests/test_folded_relax.py:0:0-0:0) (5 tests) | 6-DOF rigid body, GPU, folded basis + **PairFF** (`RigidBodyPairFF`, allmol ± FAF) |
| [RigidBodyAFM.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyAFM.py:0:0-0:0) | ✅ Active | [test_folded_relax.py](cci:7://file:///home/prokop/git/SPAMMM/tests/test_folded_relax.py:0:0-0:0) | Anchor springs, manipulation |
| `GUI/RigidBodyVispy.py` + `demos/demo_pairff.py` | ✅ Active | Demo / headless | PairFF Vispy: FIRE ON, click-to-select, `--mols`, `--faf` map compose — [PairFF_manual.md](../../demos/PairFF_manual.md) |
| [QEq.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/QEq.py:0:0-0:0) | ✅ Active | No test | Charge equilibration |
| [Assembly.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/Assembly.py:0:0-0:0) | ⚠️ Visual only | [testplot_assembly.py](cci:7://file:///home/prokop/git/SPAMMM/tests/testplot_assembly.py:0:0-0:0) (L2 only) | No L0 pytest yet |
| [FFController.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/FFController.py:0:0-0:0) | ✅ Active | No direct test | FF orchestration |
| [FFEvaluator.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/FFEvaluator.py:0:0-0:0) | ✅ Active | No direct test | FF evaluation wrapper |

### 2.3 Surface Interactions (⚠️ Mixed)

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| [Ewald2D.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/Ewald2D.py:0:0-0:0) | ✅ Working | [test_surface.py](cci:7://file:///home/prokop/git/SPAMMM/tests/test_surface.py:0:0-0:0) (11 tests) | All pass, human-reviewed, GPU+CPU parity |
| [SurfaceEwald.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/SurfaceEwald.py:0:0-0:0) | ✅ Working | [test_surface.py](cci:7://file:///home/prokop/git/SPAMMM/tests/test_surface.py:0:0-0:0) | GPU Ewald, parity confirmed |
| [GridFF.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/GridFF.py:0:0-0:0) | ⚠️ Untested | No test | B-spline interpolation, exists but no test |
| [GridFFRelaxedScan.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/GridFFRelaxedScan.py:0:0-0:0) | ⚠️ Untested | No test | Imports fixed, functionality untested |
| [FoldedRigid.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/FoldedRigid.py:0:0-0:0) | ✅ Working | [test_folded_relax.py](cci:7://file:///home/prokop/git/SPAMMM/tests/test_folded_relax.py:0:0-0:0) | Folded basis relaxation + manipulation |
| [ContactSurface.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/ContactSurface.py:0:0-0:0) | ✅ Working | [test_afm_contact_surface.py](cci:7://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_contact_surface.py:0:0-0:0) (2 tests) | Separable + PIC, parity ~14-20 meV/Å |
| [SubstrateBuilder.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/SubstrateBuilder.py:0:0-0:0) | ⚠️ Minimal | No test | NaCl/CaF2 only, no CIF parsing |
| [Surface_utils.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/Surface_utils.py:0:0-0:0) | ✅ Active | Implicit | Surface utilities |

### 2.4 SPM / AFM / STM (✅ FDBM interactive; Morse solid; **CLI ready**)

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| `AFM.py` (AFMulator) | ✅ Morse + FDBM FAST_S3 | [test_afm_morse.py](cci:7://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_morse.py:0:0-0:0), [test_afm_fdbm.py](cci:7://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_fdbm.py:0:0-0:0) | Morse/LJ + Coulomb OK; FDBM R1+R2 perf (`SPAMMM_AFM_FAST_S3`); fused ES + GPU pad/scale |
| `AFM_utils.py` | ✅ Active | FDBM + tip helpers | Tip `pad_mode='none'` for GPU roll; `compose_and_relax_total(reuse_fdbm_grid=…)`; strip plots `plot_afm_variant_height_strip` |
| `ModularPipeline.py` | ✅ Active | `tests/SPM/bench_fdbm.py` | S1–S6 + `AFMBench`; dual S3 path (fast/legacy); **not yet a CLI subcommand** |
| `KrigingGridFF.py` | ✅ Active | kriging testplots | Mithun DFT → GridFF; CLI: `run_spm.py afm-kriging` |
| `stm_compare.py` | ✅ Active | `testplot_stm_basis_compare.py` | SSOT for `run_spm.py stm {orbitals,current,panel}` |
| STM kernels (`LCAO_STM.cl`, `LCAO_grid.cl`) | ⚠️ Kernels + `compute_stm` | No dedicated STM L0 | Campaign: `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md` |

**Headless CLI (2026-07-24):** repo-root [`run_spm.py`](../../run_spm.py) + user docs [`user_guide/SPM_CLI.md`](../../user_guide/SPM_CLI.md). Task / gaps: [`doc/Tasks/SPM_CLI_Headless.md`](../Tasks/SPM_CLI_Headless.md).

| CLI command | Status |
|-------------|--------|
| `afm` (FDBM stock/prolonged/cube; SMILES; amp-align; `--plots`) | ✅ wired — defaults df 3.7–4.7 @ dz=0.1, Fz @ h−amp |
| `opt` (UFF/SPFF/LFF/DFTB; planar + PCA long→x) | ✅ wired — science OK pending |
| `smiles-afm` (SMILES → planar opt → prolonged AFM) | ✅ wired — gallery `debug/spm_smiles_afm/` |
| `afm-morse` | ✅ wired (plot SSOT polish open) |
| `afm-kriging` | ✅ wired |
| `panel-fukui` / `replot-panel` | ✅ wired |
| `stm orbitals` / `current` / `panel` | ✅ wired |
| Bond-resolved STM (BR-STM) | GUI S6 ✅; CLI ✗ — consolidate task |
| Shared GUI↔CLI job-spec protocol | ✗ ToDo (`Consolidate_GUI_CLI_Backend_Input_Protocol.md`) |
| Inputs: ASCII / `.mol`/`.mol2` shared flags | ✗ ToDo — SMILES done; ASCII builder exists (`SPM_CLI_Headless.md` §A/D) |
| Substrate `dock` (GridFF / FAF) | ✗ **Future** — harder; see `SPM_CLI_Headless.md` §E + folded-basis task |
| light-STM; charge-rings | ✗ ToDo |


**FDBM perf (T01, 2026-07-19) — measured on RTX 3090:** benzene warm ~**0.18 s** (was ~1.65 s); flat_1 S2 NA **5.87→0.03 s**; S3 cache **~10→0.4 s**; flat_1 warm S3+S4 ~**1.4 s**. Spec: `doc/Tasks/PerfBenchmark_FDBM.md`.

| `ScanUtils.py` | ✅ Active | Implicit | Scan utilities |
| `ManipulationPathOpt.py` | ⚠️ Imports fixed | No test | Autonomous manipulation |

### 2.5 Quantum Backends (⚠️ Partially Working)

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| [DFTBcore.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/quantum/DFTB/DFTBcore.py:0:0-0:0) | ✅ Works | [test_afm_fdbm.py](cci:7://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_fdbm.py:0:0-0:0) (stage 1) | ctypes interface to DFTB+, SCF runs |
| [DFTBplusParser.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/quantum/DFTB/DFTBplusParser.py:0:0-0:0) | ✅ Active | Implicit | HSD parser, eigenvector export |
| [Grid_dftb.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/quantum/DFTB/Grid_dftb.py:0:0-0:0) | ✅ Works | [test_afm_fdbm.py](cci:7://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_fdbm.py:0:0-0:0) (stage 2) | GPU density projection, STO basis |
| [basis_optimizer.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/quantum/DFTB/basis_optimizer.py:0:0-0:0) | ✅ Active | No direct test | Basis optimization |
| [DFTB_utils.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/quantum/DFTB_utils.py:0:0-0:0) | ⚠️ Imports fixed | [test_hbond_scan.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_hbond_scan.py:0:0-0:0) | Subprocess runner, Hessian writer |
| [pySCF_utils.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/quantum/pySCF_utils.py:0:0-0:0) | ⚠️ Minimal | No test | Basic SCF + geometry opt, no density grid |
| [coordinate_scan.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/quantum/coordinate_scan.py:0:0-0:0) | ✅ Active | [test_scan_dataset.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_scan_dataset.py:0:0-0:0) | RC scan, pm-NEB prototype |
| [hbond_scan.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/quantum/hbond_scan.py:0:0-0:0) | ✅ Active | [test_hbond_scan.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_hbond_scan.py:0:0-0:0) | DFTB rigid scan |
| [esp_grid.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/quantum/esp_grid.py:0:0-0:0) | ✅ Active | No direct test | ESP grid computation |

### 2.6 GUI (⚠️ Partially Working)

| Module | Status | Tests | Notes |
|--------|--------|-------|-------|
| [SPAMMM_GUI.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/SPAMMM_GUI.py:0:0-0:0) | ✅ Active | No test | Main window; **`b2Dview` / Enter** ortho 2D↔3D; mouse dispatch; grid transforms |
| [EditModeHandlers.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/EditModeHandlers.py:0:0-0:0) | ✅ Active | No test | Unified/Atom/Bond/Ring/Hex; Ring 3D = closest atom/bond/ring-COG along ray; hex 2D-only |
| [VispyUtils.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/VispyUtils.py:0:0-0:0) | ✅ Active | No test | `AtomScene`: ortho camera, presets, RMB orbit, depth test in 3D |
| [AFMExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/AFMExtension.py:0:0-0:0) | ✅ Active | No pytest (manual GUI) | Dirty flags S1–S6; FDBM FAST_S3; plot z default 3.0 Å; atom overlay toggle + `'.'` dots |
| [KekuleExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/KekuleExtension.py:0:0-0:0) | ✅ Active | No test | Bond order visualization |
| [FoldedRigidExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/FoldedRigidExtension.py:0:0-0:0) | ✅ Active | No test (manual L2) | Interactive manipulation |
| [FFExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/FFExtension.py:0:0-0:0) | ✅ Active | No test | FF relaxation panel |
| [QEqExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/QEqExtension.py:0:0-0:0) | ✅ Active | No test | QEq panel |
| [VibrationExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/VibrationExtension.py:0:0-0:0) | ✅ Active | No test | Vibrational analysis panel |
| [ReactionCoordinateExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/ReactionCoordinateExtension.py:0:0-0:0) | ✅ Active | `test_rc_scan_gui_script.py` | RC scan GUI |
| [MolecularBrowser.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/MolecularBrowser.py:0:0-0:0) | ⚠️ Needs VisPy port | No test | Molecular browser |
| [FragmentExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/FragmentExtension.py:0:0-0:0) | ✅ Active | [test_fragmentation.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_fragmentation.py:0:0-0:0) | Fragment detection |
| [AsciiArtExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/AsciiArtExtension.py:0:0-0:0) | ✅ Active | [test_ascii_art.py](cci:7://file:///home/prokop/git/SPAMMM/tests/topology/test_ascii_art.py:0:0-0:0) | ASCII art input |
| [DFTBExtension.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/GUI/DFTBExtension.py:0:0-0:0) | ✅ Active | No test | DFTB panel |

### 2.7 Infrastructure

| Module | Status | Notes |
|--------|--------|-------|
| `OpenCLBase.py` | ✅ Working | Auto-selects NVIDIA GPU, kernel compilation |
| `Lingebra_ocl.py` | ✅ Working | Jacobi eigendecomposition (6 tests pass) |
| [config_utils.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/config_utils.py:0:0-0:0) | ✅ Working | DFTB path config |
| [atomicUtils.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/atomicUtils.py:0:0-0:0) | ✅ Working | XYZ/MOL2 I/O |
| [AtomicSystem.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/AtomicSystem.py:0:0-0:0) | ✅ Working | Atom system wrapper |
| pytest infrastructure | ✅ Working | [conftest.py](cci:7://file:///home/prokop/git/SPAMMM/tests/conftest.py:0:0-0:0), markers, helpers, ref_data |

---

## 3. Test Coverage Summary

### Current test counts (from [TEST_RESULTS.md](cci:7://file:///home/prokop/git/SPAMMM/tests/TEST_RESULTS.md:0:0-0:0)):

| Category | Tests | Passed | Failed | Stubs |
|----------|-------|--------|--------|-------|
| Non-GPU (topology + Ewald2D) | 23 | 23 | 0 | 0 |
| GPU (UFF, SurfaceEwald, AFM) | 13 | 8 | 5 | — |
| Stubs (AFM, integration, SPFF) | 9 | 9 | 0 | 9 |
| **Total** | **45** | **40** | **5** | **9** |

Plus: 42 topology editing tests, 5 folded rigid tests, 2 contact surface tests, 6 lingebra tests, 11 FDBM tests (require DFTB+).

### Test quality levels:

- **L0 (automated asserts):** ~60+ tests with `assert` checks — topology counts, finiteness, energy conservation, force-energy correspondence, Ewald parity
- **L1 (agent review):** `--develop` mode produces `.out`/`.log` artifacts, topology tests use `TopologySnapshot`/`TopologyDiff`
- **L2 (visual human review):** Ewald scans, AFM images, FDBM slices, folded rigid manipulation — all produce `.png` for human inspection

### Major test gaps:

| Area | Gap | Priority |
|------|-----|----------|
| FDBM hex/rot60 image QA | Coarse `step=0.15` symmetry; see AFMTesting lessons | Medium |
| ModularPipeline S1-S6 | Bench exists (`bench_fdbm.py`); more L0 E2E asserts welcome | Medium |
| STM simulation | CLI `stm *` wired; L0 thin; basis campaign open | **High** — `STM_ExtendedBasis_OrbitalCompare.md`, `SPM_CLI_Headless.md` |
| Headless SPM CLI gaps | BR-STM, substrate relax, light-STM, charge-rings | **High** — `SPM_CLI_Headless.md` |
| Pauli \(A,\beta\) site maps | Global fits only; H≠N/C in pyridine | **High** — `Pauli_A_beta_KrigingTransferability.md` |
| Fukui FDBM molecule panel | Ran cube/stock/prolonged; cube ES open; USER review | **High** — `ProlongedRadialBasis_DFTB.md`, `Fukui_FDBM_panel_notes_2026-07-23.md` |
| Kekulé RI density | No atom+bond exponential fit yet | Medium — `Kekule_ExponentialDensityFit.md` |
| 2.5D contact surface | Sphere h₀ + coarse dx; USER visual vs GridFF | Medium — `ContactSurface_2p5D_vs_GridFF_2026-07-24.md` |
| DFTB+ density projection | Covered via FDBM tests; dedicated unit tests thin | Medium |
| pySCF backend | No test | Medium |
| GridFF construction/interpolation | No test | Medium |
| SPFF relaxation | Stubs only | Medium |
| UFF NVE conservation | Shape mismatch bug | Medium |
| UFF energy finite | Energy=0 (nonbonded disabled) | Low |
| GUI | No automated tests | Low |
| Assembly | No L0 pytest | Low |

---

## 4. Known Bugs & Issues

### Critical (pipeline-breaking):

1. ~~**FDBM df blank / relaxStrokes**~~ — **stale as of 2026-07:** GUI FDBM + PP relax produces finite non-zero `df` (benzene `|df|max`~0.2 Hz in `bench_fdbm`). Re-verify old FIRE/`tipForce` notes in `AFM.cl` only if blank images reappear.

### Algorithmic:

3. **UFF CH4 relaxation**: Bond assertion fails — likely topology/parameter assignment issue in `UFFbuilder`
4. **UFF NVE conservation**: Shape mismatch `(5,3)` vs `(1,5,3)` — array broadcasting bug in MD code
5. **UFF energy=0**: `bDoNonBonded=False` by default, test should enable it
6. **Folded poly basis**: Power sequence wrong — kernel uses sequential powers `m_start, m_start+1, ...` but needs doubling powers `[4,8,16,32,64]` for good fit quality

### Infrastructure:

7. **No `pyproject.toml`/`setup.py`**: Package not formally pip-installable
8. **H atoms not connected after molecule load**
9. **[FeatureChecklist.md](cci:7://file:///home/prokop/git/SPAMMM/FeatureChecklist.md:0:0-0:0) stale**: Needs sync with current state

---

## 5. What Works Well (Conference-Ready)

- **Topology editing**: 42 tests, Kekule solver, ASCII art, heterocycle generation — all solid
- **Ring placement (3 modes)**: unified in Ring mode — Numpad +/- controls ring size (3-12). **2D:** bond → corner atom → hex. **3D:** closest of bond / atom / ring-COG along mouse ray (hex disabled); ring side uses ray∩z=0. RMB deletes bond/atom or whole ring at COG.
- **Ortho 2D/3D editor view**: `Enter` / `b2Dview` checkbox; RMB empty = rotate; digit presets (5=Top); depth test in 3D. Task: `doc/Tasks/GUI_Editor_3D_ViewMode.md`. Cheatsheet: `doc/GUI_CHEATSHEET.md`.
- **Ewald2D surface electrostatics**: 11 tests, GPU-CPU parity, human-reviewed visual tests
- **Morse/LJ AFM**: 9 tests, produces correct AFM contrast for pentacene/PTCDA
- **Folded basis rigid body**: 5 tests, relaxation + manipulation on NaCl(100), reference data system
- **Contact surface AFM**: 2 tests, separable + PIC parity, memory-efficient
- **Force-energy correspondence**: 4 tests (UFF+SPFF), 1 critical bug found and fixed (pi-sigma energy)
- **MD invariants**: 2 tests, 3 critical bugs found and fixed (angle energy, bond double-counting, MD parameter mapping)
- **Jacobi eigendecomposition**: 6 tests, GPU kernel matches numpy
- **Z-scan reference curves**: 60 curves, 4 QM methods, DFTB vs pySCF comparison
- **FDBM Pauli parameter fitting**: Global log-log fit, A/β parameters for mio-1-1, 3ob-3-1, pySCF
- **FDBM Fz/df imaging**: PTCDA + pentacene with fitted Pauli params (via `plot_fdbm_relax.py`, not pytest)

---

## 6. Testing Strategy Assessment

### Current approach (from [doc/TEST_DESIGN.md](cci:7://file:///home/prokop/git/SPAMMM/doc/TEST_DESIGN.md:0:0-0:0)):

The three-level system (L0/L1/L2) is well-designed:
- **L0**: Automated asserts — finiteness, counts, invariants, parity checks
- **L1**: Agent reads `.out`/`.log` artifacts in `--develop` mode
- **L2**: Human reviews `.png` visual outputs

### Physical correctness tests that exist:

| Type | Example | Status |
|------|---------|--------|
| **Energy conservation** | [test_invariants](cci:1://file:///home/prokop/git/SPAMMM/tests/test_forcefield.py:158:0-261:82) (NVE MD) | ✅ Working |
| **Force-energy correspondence** | [test_ef_correspondence](cci:1://file:///home/prokop/git/SPAMMM/tests/test_forcefield.py:504:0-585:95) (F = -dE/dx) | ✅ Working |
| **Newton's 3rd law** | [test_uff_force_newton3](cci:1://file:///home/prokop/git/SPAMMM/tests/test_forcefield.py:139:0-150:78) | ✅ Working |
| **Electron count** | [test_density_projection](cci:1://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_fdbm.py:224:0-237:84) (rho integrates to N_e) | ✅ Working |
| **Charge conservation** | [test_charge_conservation](cci:1://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_fdbm.py:257:0-267:90) (rho_diff → 0) | ✅ Working |
| **Far-field decay** | [test_poisson_potential](cci:1://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_fdbm.py:270:0-286:107) (boundary < peak) | ✅ Working |
| **Ewald parity** | [test_ewald_py_vs_cl](cci:1://file:///home/prokop/git/SPAMMM/tests/test_surface.py:161:0-175:85) (GPU vs CPU) | ✅ Working |
| **AFM physics** | [test_afm_raw_scan](cci:1://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_morse.py:94:0-114:113) (close > far), [test_afm_relaxed_scan](cci:1://file:///home/prokop/git/SPAMMM/tests/SPM/test_afm_morse.py:121:0-149:88) (raw ≠ relax) | ✅ Working |
| **Bond geometry** | [test_relax](cci:1://file:///home/prokop/git/SPAMMM/tests/test_forcefield.py:41:0-118:64) (relaxed bond lengths match expected) | ⚠️ Partially |
| **Reference data** | `test_folded_relax` (adsorption height, force convergence) | ✅ Working |

### Brainstorm: Physical testing possibilities

**Already implemented but could be strengthened:**

1. **Symmetry invariance tests**: Rotate molecule 90°, AFM image should rotate accordingly. Not tested.
2. **Scaling laws**: Pauli overlap should scale as `A * overlap^β` — could assert power-law fit quality.
3. **Dispersion asymptotics**: vdW should scale as `-C6/r^6` at large distances — could assert decay rate.
4. **Poisson solution properties**: V_ES should satisfy Laplace equation away from charges — could test `∇²V = 0`.
5. **Tip force balance**: At equilibrium, spring force = interaction force — could assert in relaxed scan.
6. **Reciprocity**: AFM image of molecule A over substrate B should relate to B over A (not generally true, but specific symmetries could be tested).

**New physical test ideas:**

7. **Density normalization**: `∫ρ_scf dV = N_electrons` already tested, but could also check `∫ρ_tip dV = N_tip` and `∫overlap dV ≤ min(N_sample, N_tip)`.
8. **Pauli sign**: Pauli energy must be **repulsive** (positive) everywhere. Already asserted.
9. **Dispersion sign**: vdW must be **attractive** (negative). Already asserted.
10. **Force direction**: At close approach, Fz should be **repulsive** (positive). Already asserted in Morse tests.
11. **Image contrast**: For known molecules (benzene, pentacene), AFM images should show expected symmetry (6-fold for benzene). Could assert via autocorrelation or symmetry detection.
12. **Convergence with grid resolution**: As step → 0, density integral should converge to exact N_e. Could test Richardson extrapolation.
13. **Kohn-Sham vs DFTB density comparison**: For small molecules, pySCF and DFTB densities should be qualitatively similar (same electron count, similar spatial distribution).
14. **STM DOS sum rule**: Sum of LDOS over all MOs should equal total DOS. Could assert `Σ LDOS(E_i) = DOS_total`.
15. **Thermodynamic consistency**: For folded basis, force = -gradient of fitted energy. Already tested via [test_ef_correspondence](cci:1://file:///home/prokop/git/SPAMMM/tests/test_forcefield.py:504:0-585:95) pattern.

---

## 7. Documentation System

The repo has a well-organized documentation hierarchy:

- [CODEMAP.md](cci:7://file:///home/prokop/git/SPAMMM/CODEMAP.md:0:0-0:0) — repo structure overview
- [MANIFEST.md](cci:7://file:///home/prokop/git/SPAMMM/MANIFEST.md:0:0-0:0) — background, dependency chains, known issues
- [FeatureChecklist.md](cci:7://file:///home/prokop/git/SPAMMM/FeatureChecklist.md:0:0-0:0) — feature inventory (stale, needs sync)
- [doc/TEST_DESIGN.md](cci:7://file:///home/prokop/git/SPAMMM/doc/TEST_DESIGN.md:0:0-0:0) — test system SSOT (L0/L1/L2)
- [doc/topical_audit.md](cci:7://file:///home/prokop/git/SPAMMM/doc/topical_audit.md:0:0-0:0) — cross-implementation topic map
- [doc/Takeways.md](cci:7://file:///home/prokop/git/SPAMMM/doc/Takeways.md:0:0-0:0) — debugging lessons
- [doc/ToDo/ToDo.agents.md](cci:7://file:///home/prokop/git/SPAMMM/doc/ToDo/ToDo.agents.md:0:0-0:0) — agent task index (Done/Soon/Later)
- [doc/AGENTS/skills/](cci:9://file:///home/prokop/git/SPAMMM/doc/AGENTS/skills:0:0-0:0) — 15 skill files for agent guidance
- [doc/AGENTS/protocols/](cci:9://file:///home/prokop/git/SPAMMM/doc/AGENTS/protocols:0:0-0:0) — domain + general protocols
- [doc/Topics/](cci:9://file:///home/prokop/git/SPAMMM/doc/Topics:0:0-0:0) — per-topic deep dives (AFM, Vibrations, RC Scan, RigidBody, PairFF)
- [doc/TopicalAudit/PairFF_RigidBody.md](../TopicalAudit/PairFF_RigidBody.md) — PairFF inventory
- [demos/PairFF_manual.md](../../demos/PairFF_manual.md) — PairFF user manual
- [tests/TEST_RESULTS.md](cci:7://file:///home/prokop/git/SPAMMM/tests/TEST_RESULTS.md:0:0-0:0) — 1109-line detailed test report with human-reviewed sections
- Per-module [README.md](cci:7://file:///home/prokop/git/SPAMMM/README.md:0:0-0:0) files in key directories
- [doc/GUI_CHEATSHEET.md](cci:7://file:///home/prokop/git/SPAMMM/doc/GUI_CHEATSHEET.md:0:0-0:0) — keyboard & mouse controls cheatsheet

---

## 8. Summary: What's Done, What's Not

### ✅ Done & Working (conference-demo-ready):
- Molecular topology editing (42 tests)
- Kekule bond order solver
- ASCII art → heterocycle generation
- UFF force field (relaxation, EF correspondence, Newton's 3rd law)
- SPFF force field (EF correspondence, pi-sigma bug fixed)
- Ewald2D surface electrostatics (11 tests, GPU+CPU parity)
- Morse/LJ AFM imaging (9 tests, pentacene/PTCDA images); CLI `afm-morse`
- **Headless SPM CLI** (`run_spm.py`): FDBM / Morse / Kriging AFM + STM orbitals/current/panel — docs `user_guide/SPM_CLI.md` (science sign-off pending; gaps in `SPM_CLI_Headless.md`)
- Folded basis rigid body relaxation + manipulation (5 tests, ref data)
- **PairFF rigid-body docking** — unified/env OpenCL kernels, Vispy demo (FIRE default, multi-body click-to-select, mixed XYZs); manual `demos/PairFF_manual.md`; main-GUI wire still open
- Contact surface AFM (2 tests, separable + PIC)
- DFTB+ SCF + GPU density projection
- FDBM Pauli parameter fitting (global log-log, 3 basis sets)
- FDBM Fz/df imaging via `plot_fdbm_relax.py` (PTCDA, pentacene)
- **FDBM ModularPipeline perf (T01 R1+R2):** benzene warm ~0.18 s; flat_1 S3+S4 ~1.4 s; fused ES + GPU pad/scale (`SPAMMM_AFM_FAST_S3`); see `doc/Tasks/PerfBenchmark_FDBM.md`
- Z-scan reference curves (60 curves, 4 QM methods)
- Jacobi eigendecomposition GPU kernel (6 tests)
- Reaction-coordinate scan + GUI
- Test infrastructure (L0/L1/L2, conftest, helpers, ref_data)

### ⚠️ Partially Working / Needs Fixing:
- **FDBM image hex symmetry** at coarse grids (`step=0.15`) — prefer ≤0.1 Å (`doc/Tasks/AFMTesting.md`)
- **FDBM perf polish:** async cache skip; optional interactive coarse grid (`PerfBenchmark_FDBM.md` T-next-4)
- **UFF CH4 relaxation**: Bond assertion failure
- **UFF NVE conservation**: Array shape mismatch
- **Folded poly basis**: Power sequence wrong
- **pySCF backend**: Minimal, no density grid test
- **STM simulation**: CLI `stm *` + `stm_compare`; systematic basis panel / L0 still open (`STM_ExtendedBasis_OrbitalCompare.md`)
- **SPM CLI gaps**: bond-resolved STM; gas-phase `opt` + ASCII/SMILES inputs; substrate GridFF/FAF `dock` (**future**); light-STM; charge-rings (`SPM_CLI_Headless.md`)
- **Pauli site maps / transferability**: global \(A,\beta\) only (`Pauli_A_beta_KrigingTransferability.md`)
- **Kekulé RI density**: not started (`Kekule_ExponentialDensityFit.md`)
- **2.5D contact surface**: prototype (separable+PIC); harden (`Fast_2p5D_AFM_ContactSurface.md`)
- **GUI**: No automated tests
- **Assembly**: No L0 pytest
- **GridFF**: No tests
- **SubstrateBuilder**: Minimal (no CIF parsing)

### ❌ Not Yet Implemented / Campaign backlog:
- Contact-surface elastic AFM (Winkler model) — static 2.5D task: `Fast_2p5D_AFM_ContactSurface.md`
- STM mio/3ob/prolonged vs pySCF cubes — `STM_ExtendedBasis_OrbitalCompare.md`
- Bond-resolved STM + gas-phase `opt` + ASCII/SMILES loaders + substrate GridFF/FAF `dock` (future) + light-STM + charge-ring imaging via CLI — `SPM_CLI_Headless.md`
- Site-resolved Pauli \(A,\beta\) Kriging + transferability — `Pauli_A_beta_KrigingTransferability.md`
- Kekulé π → exponential RI density — `Kekule_ExponentialDensityFit.md`
- RigidAtom XPBD / RRsp3
- ProjectiveDynamics implicit solver (beyond LFF)
- Reactive rigid-atom FF
- NEB / H-transfer workflows (full)
- Phonon / FTIR / Hessian
- FMM long-range electrostatics
- Molecular browser plugins + AFM thumbnails (Phase 1 VisPy browser exists)
- Pentagon/heptagon via corner ring (implemented); SMILES builder (not yet)
- LAMMPS/GROMACS export
- `pyproject.toml` / `setup.py` (not pip-installable)