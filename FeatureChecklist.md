# SPAMMM — Feature Checklist & Roadmap

> Comprehensive inventory of implemented, in-progress, and planned features.
> Maturity levels: **✅ Working** · **⚠️ In Progress** (implemented but broken/incomplete) · **🔲 Planned** (not started)

> **Source repository:** Several features are imported from / planned for porting from **FireCore** (`/home/prokop/git/FireCore/`), the parent project. FireCore contains mature implementations of rigid-atom force fields (RRsp3), web-based molecular GUI (molgui_webgpu), and structure editing utilities that serve as reference implementations for SPAMMM.

---

## 1. Molecular Topology & Editing

### ✅ Working
- **Bond detection** — distance-based topology from XYZ (`atomicUtils.py`)
- **Atom type assignment** — hybridized types from `AtomTypes.dat` (`FFparams.py`)
- **Neighbor consistency** — neighbor list validation
- **Geometry validation** — water, benzene, methane geometry checks
- **AtomicGraph** — object-graph representation with stable IDs, Atom/Bond/Ring objects (`AtomicGraph.py`)
- **KekuleBackend hex grid** — hexagonal grid painting, passivation groups (N, NH, CH, H, O, C=O, C-OH), pi/n-pi toggle (`KekuleBackend.py`)
- **Hybridization inference** — sp/sp2/sp3 from atom types, pi-orbital tracking
- **PBC support** — `pbc_x`, `pbc_y` flags for periodic structures in editor

### ✅ Working (updated)
- **AtomicGraph editing** — add/remove atom/bond/ring, 42 tests in `test_editing_ops.py`
- **Ring detection** — `detect_rings()` implemented + tested
- **KekuleSolver** — `KekulePure.py` multi-seed bond order solver, tested
- **Electron pair placement** — `make_epair_geom` for sp/sp2/sp3, 9 molecules tested
- **Fragment/bridge detection** — `test_fragmentation.py` passing
- **PackedMolecule** — undo stack + clipboard, tested

### ✅ Working (updated)
- **PAH/zigzag ribbon builder** — zigzag graphene ribbon (3×3) with H passivation via `adjust_h()`, tested in `test_editing_ops.py`

### 🔲 Planned
- **Pentagon/heptagon drawing** — extend hex grid to N-gon rings (5,7 membered). See `doc/ARCHITECTURE_ROADMAP.md` §9
- **SMILES builder** — parse SMILES → AtomicGraph. See `doc/ARCHITECTURE_ROADMAP.md` §9

### 🔲 Planned — Port from FireCore (`web/molgui_webgpu/`, `web/common_js/`)
- **Selection query system** — compile/select by element + neighbor-count constraints. Query syntax: `"C n{C}={2,3}"` selects carbons with 2 or 3 carbon neighbors; `"* deg={1}"` selects any atom with degree 1. Supports element wildcards (`*`), atom-type matching (`C_sp2`), set operations (replace/add/subtract). Source: `MoleculeSelection.js` (`compileSelectQuerySpec`, `applySelectQuery`).
- **Selection data structure** — ordered set with fast membership via Map, tombstone-based removal, set operations (subtract, intersect), `SelectionBanks` for multiple parallel selections. Source: `Selection.js` (`Selection` class, `SelectionBanks` class).
- **ScriptRunner command dispatcher** — whitelist-based command execution (no eval) for molecular editing scripts. 30+ commands: load, translate, rotate, select, select_query, recalculate_bonds, add_caps, build_substrate, build_polymer, build_nanocrystal, replicate, relax, relaxJacobi_CPU/GPU. Supports both JSON command lists and JS scripting with queued API. Source: `ScriptRunner.js`.
- **Bridge manipulation** — select bridge candidates (heavy atoms with 2 heavy + ≥2 H neighbors), collapse bridge (remove CH2 group, reconnect neighbors), insert bridge (add CH2 between two heavy atoms), collapse all bridges, random collapse/insert. Source: `MoleculeSelection.js` (`selectBridgeCandidates`, `findBridgeAtom`, `findBondPairForInsert`), `MoleculeUtils.js` (`collapseBridgeAt`, `insertBridge`).
- **Undercoordinated atom operations** — select by element + degree (e.g. `Si deg={1}`), remove undercoordinated atoms, stochastic collapse of fraction of undercoordinated atoms. Source: `ScriptRunner.js` (`removeUndercoordinatedAtoms`, `collapseUndercoordinatedAtoms`).
- **Polymer builder** — assemble polymers from monomer sequence tokens with head/tail anchor indices, monomer library + geometry loading. Source: `ScriptRunner.js` (`buildPolymer`), `BuildersGUI.js`.
- **Nanocrystal builder** — build nanocrystals from CIF files with symmetry operations, plane cutting (Miller indices), cell box visualization, bucket-based broad-phase. Source: `ScriptRunner.js` (`buildNanocrystal`), `CrystalUtils.js`.
- **Cap addition** — add capping atoms (H by default) to undercoordinated atoms, optionally only for current selection. Source: `ScriptRunner.js` (`addCaps`), `EditableMolecule.js` (`addCappingAtoms`).

---

## 2. Force Fields

### ✅ Working
- **UFF bonds + angles** — GPU kernels in `UFF.cl`, energy-force correspondence verified (CH4, CH2NH)
- **UFF torsions + inversions** — implemented in kernel, tested via EF correspondence
- **UFF Newton's 3rd law** — force symmetry verified
- **UFF visual relaxation** — H2O, benzene relax correctly
- **SPFF bonds + angles + pi-pi** — GPU kernels in `SPFF.cl`, EF correspondence verified (CH4, CH2NH)
- **SPFF pi-sigma orthogonalization** — fixed copy-paste bug, verified
- **MD invariants conservation** — energy, linear momentum, angular momentum conserved (CH4, CH2NH, semi-implicit Euler)
- **Energy-force correspondence tests** — `F = -dE/dx` verified for UFF and SPFF via finite differences
- **PairFF rigid-body nonbonded** — `RigidBodyPairFF` + `rigid.cl` unified/env kernels (Strategy M: one active body, frozen neighbors); epair/σ-hole H-bonds; Vispy demo with FIRE default, click-to-select active, `--mols` mixed species. Manual: `demos/PairFF_manual.md`. Main `SPAMMM_GUI` integration still planned (`doc/Tasks/PairFF_GUI_Integration.md`).

### ⚠️ In Progress
- **UFF relaxation (CH4)** — bond assertion failure for methane; likely topology/parameter issue in `UFFbuilder.py`
- **UFF energy finite** — energy=0.0 because `bDoNonBonded=False` by default; test logic issue
- **UFF NVE conservation** — shape mismatch `(5,3)` vs `(1,5,3)` — array broadcasting bug in MD code
- **UFF non-bonded (LJ + Coulomb)** — implemented in `UFF.cl` + `nonbonded.cl`, not tested with `bDoNonBonded=True`
- **SPFF relaxation** — test stub only, not implemented
- **SPFF pi-pi interactions** — no dedicated test
- **SPFF H-bond corrections** — no test
- **MolecularDynamics FIRE** — implemented in `MolecularDynamics.py`, no test
- **MolecularDynamics velocity Verlet** — implemented, no test
- **MolecularDynamics multi-system** — buffer management exists, no test
- **SPFF with pi-orbital rotation** — `getSPFFf4_rot` / `updateAtomsSPFFf4_rot` exist, untested
- **FAF substrate relaxation** — use ForcedAtomicFunction instead of GridFF for on-surface relaxation. See `doc/ARCHITECTURE_ROADMAP.md` §5

### ✅ Working (updated)
- **QEq module** — `spammm/forcefields/QEq.py` + GUI extension (`QEqExtension.py`)
- **RigidAtom forcefield (RRsp3)** — cluster-sorted rigid body dynamics with ARAP-style "ports" and PBD/Jacobi collisions with recoils. **Reference implementation in FireCore** (`pyBall/RigidAtomFF/RRsp3/`):
  - `RRsp3.cl` (1748 lines) — OpenCL kernels: `update_bboxes_rigid`, `build_local_topology_rigid`, `compute_collision_cluster_rigid`, `compute_ports_cluster_rigid`, `apply_corrections_rigid_ports`
  - `RRsp3.py` (42KB) — Python harness: buffer management, state upload/download, topology upload, solver execution
  - Cluster-sorted layout (64 atoms/workgroup), ghost atom halo for inter-cluster interactions, recoil buffers for momentum conservation
  - 4 rotation solver variants: XPBD (impulse-based), shapematch (polar decomposition), eigen (Davenport q-method), substep_optimized (Newton substeps)
  - Jacobi iteration with heavy-ball momentum acceleration, 1-2 and 1-3 neighbor exclusion
  - Known issue: slow convergence (~73-140 iterations vs expected 3-5) due to Jacobi diffusion on linear chains
  - Also: `XPDB_new/` (experimental force+position based), `XPDB_legacy/` (deprecated simple PBD), `XPBD_2D/` (2D specialization)
  - Test scripts: convergence, momentum conservation, smoke, debug, Vispy GUI
  - **SPAMMM port status:** not started. Would extend `RigidBodyDynamics.py` with per-atom orientation parameters and cluster-sorted layout.
- **Reactive forcefield** — port from `NumericalMathPlayground/topics/ReactiveFF/`. Angular-dependent pair potential with per-atom quaternion orientation. See `doc/ARCHITECTURE_ROADMAP.md` §8
- **SPFFL / LMMF** — linearized SPFF for fast assembly/placement without full relaxation. Exists in FireCore, not yet ported to SPAMMM.
- **AABB collision relaxation** — projective/position-based with bounding boxes for multi-molecule clusters. See `doc/ARCHITECTURE_ROADMAP.md` §7

---

## 3. Surface Interactions

### ✅ Working
- **Ewald2D (NumPy)** — pure Python reference implementation (`Ewald2D.py`): neutrality, vacuum decay, symmetry, vs brute-force — all passing
- **Ewald2D (OpenCL)** — production GPU implementation (`SurfaceEwald.py`): Py vs CL parity, CL full 1D, CL vs brute z-scan/x-scan, NaCl 8×8 — all passing
- **Ewald2D visual scans** — z-scan, x-scan, lateral scans — human-reviewed, passing
- **Folded basis fit (Morse)** — exponential + polynomial basis fit to NaCl(100) Morse potential, RMSE < 1e-4 eV
- **Folded basis fit (Coulomb)** — Ewald2D periodic reference, exp + poly basis, RMSE < 1e-3 eV
- **Folded tensor kernels (exp)** — GPU-CPU parity confirmed, `getSurfFolded_tensor_exp`, cubic energy formula
- **Folded tensor kernels (poly)** — GPU-CPU parity confirmed (but fit quality poor — see ⚠️)

### ⚠️ In Progress
- **Folded poly basis fit quality** — kernel uses sequential powers `[m_start, m_start+1, ...]` instead of doubling powers `[4, 8, 16, 32, 64]`. Critical blocker. Needs kernel modification to support arbitrary power sequences.
- **Per-component power sequences** — kernel uses single `m_start` for Pauli/London/Coulomb; scan test uses different sets per component.
- **GridFF construction** — `GridFF.py` exists, B-spline interpolation implemented, untested
- **GridFF B-spline interpolation** — no test
- **GridFF PLQH channels** — Pauli/London/Coulomb/H-bond channels, no test
- **GridFFRelaxedScan** — imports fixed, untested (relaxed PES scan: GridFF + SPFF + MD)
- **SubstrateBuilder** — NaCl, CaF2 slab generation, no test
- **Folded atomic functions** — `Surface_utils.py`, no test
- **GridFF consolidation** — unify tricubic B-spline (4ch) vs trilinear (12ch). See `doc/ARCHITECTURE_ROADMAP.md` §6

### 🔲 Planned
- **GridFF extensions** — precomputed grids, multi-species grids, additional GridFF utilities
- **RRsp3 surface interaction** — once RigidAtom forcefield is ported (see §2 Force Fields), integrate with GridFF/folded basis for rigid-atom surface docking

---

## 4. SPM / AFM / STM

### ✅ Working
- **AFMulator Morse + point-charge** — LJ/Morse + Coulomb AFM, `AFM.cl` kernel, 9 functional tests passing
- **AFM probe-particle relaxation** — `relaxStrokesTilted` kernel, FIRE + damped velocity modes
- **AFM frequency shift (df)** — `compute_df` from Fz(z) curves
- **AFM Morse vs LJ comparison** — correlated but different, verified
- **AFM visual images** — pentacene, PTCDA: potential slices, Fz slices, Fz(z) curves, raw vs relaxed comparison — human-reviewed
- **AFMulator QEq (CPU)** — `solve_QEq()` numpy implementation, builds screened Coulomb matrix, solves charge equilibration

### ⚠️ In Progress
- **FDBM pipeline** — DFTB SCF → density → Pauli → electrostatics → vdW → AFM scan. Runs without crash but produces degenerate/empty output (df=0, density slices empty, ES/vdW tilted). Root causes: `relaxStrokes` vs `relaxStrokesTilted` coordinate mismatch, grid layout/visualization issues.
- **AFMulator FDBM scan** — `scan_fdbm()` uses `relaxStrokes` (broken); should use `relaxStrokesTilted` (working for Morse path)
- **ModularPipeline S1-S6** — staged FDBM pipeline with disk caching, imports fixed, not tested end-to-end
- **STM orbital projection** — `LCAO_STM.cl` kernel exists, no test
- **STM DOS/LDOS** — no test
- **Bond-resolved STM** — sampling STM at relaxed PP positions, no test
- **Contact surface AFM** — `ContactSurface.py` with separable + PIC, 2 tests passing. Needs wiring into `AFMulator`

### 🔲 Planned
- **Lateral force microscopy with dissipation** — LFM mode measuring lateral forces and energy dissipation during tip oscillation. No implementation. Would require: lateral force channel in scan, dissipation integration over oscillation cycle.
- **Constant-current AFM** — feedback-controlled scan height maintaining constant tunneling current (or constant total current). No implementation. Would require: real-time height feedback loop, current calculation model.
- **Near-field photoluminescence simulation** — simulate near-field optical excitation and PL emission in SPM geometry. No implementation. Would require: electromagnetic near-field calculation, exciton/plasmon coupling model, PL emission spectrum.
- **Quantum dot solver for dIdV charging rings** — self-consistent quantum dot solver to simulate single-electron charging effects in dI/dV spectroscopy (Coulomb blockade rings in scanning gate microscopy). No implementation. Would require: quantum dot energy level solver, tunneling rate calculation, charging energy, dI/dV map generation with tip position as gate.

---

## 5. Rigid Body Dynamics

### ✅ Working (updated)
- **RigidBodyDynamics 6-DOF** — quaternion-based dynamics, `rigid.cl` kernel, 5 tests in `test_folded_relax.py`
- **RigidBodyAFM anchor springs** — harmonic constraint on specific atoms, tested (manipulation H2O + PTCDA)
- **RigidBody folded basis** — `init_folded()` / `run_folded()` for folded atomic function surface interaction, tested
- **Folded basis relaxation + manipulation** — 5 tests, reference data system, trail plots
- **PairFF multi-body docking** — one active rigid molecule vs frozen neighbors (Strategy M env kernel); Vispy click-to-select; FIRE default; mixed `--mols`. Manual: `demos/PairFF_manual.md`

### 🔲 Planned
- **Assembly collision** — `testplot_assembly.py` visual demo + XYZ/diag artifacts (`Assembly.py`, `AssemblyPlot.py`, `assembly.cl`); pytest L0 pending
- **AABB collision relaxation** — projective/position-based with bounding boxes for multi-molecule clusters. See `doc/ARCHITECTURE_ROADMAP.md` §7

---

## 6. Quantum Backends

### ✅ Working (updated)
- **DFTB+ SCF** — `DFTBcore.py` ctypes wrapper, runs correctly (H2O E=-342 eV, benzene E=-1033 eV)
- **DFTB+ density grid projection** — `Grid_dftb.py` + `LCAO_grid.cl`, electron count correct, density visualization works via `plot_density_projection.py`
- **DFTB+ wfc parsing** — `DFTBplusParser.py`, active
- **DFTB_utils subprocess runner** — `DFTB_utils.py`, imports fixed, tested via `test_hbond_scan.py`
- **Z-scan reference curves** — 60 curves, 4 QM methods (dftb_mio, dftb_3ob, pyscf_pbe, pyscf_b3lyp)
- **FDBM Pauli parameter fitting** — global log-log fit, A/β for mio-1-1, 3ob-3-1, pySCF

### ⚠️ In Progress
- **pySCF backend** — `pySCF_utils.py` minimal, needs robust integration with custom modified pySCF. See `doc/ARCHITECTURE_ROADMAP.md` §2

### 🔲 Planned
- **Fireball / OCL Hamiltonian** — OpenCL Hamiltonian assembly, CheFSI, OMM. May re-add if DFTB+ insufficient for STM accuracy.

---

## 7. GUI

### ✅ Working
- **SPAMMM_GUI / AtomScene** — VisPy + PyQt5 molecular editor; **ortho 2D/3D view mode** (`Enter` / `b2Dview`); Ring/atom/bond pick in 3D; hex 2D-only. See `doc/Tasks/GUI_Editor_3D_ViewMode.md`, `doc/GUI_CHEATSHEET.md`
- **AFMExtension** — AFM panel with S1-S6 pipeline UI
- **VispyUtils AtomScene** — reusable ortho scene: picking, drag, camera presets, depth test in 3D

### ⚠️ In Progress
- **MolecularBrowserVispy** — VisPy Phase 1 (thumbnail grid + 3D view), no plugin system yet

### 🔲 Planned
- **Molecular browser plugin system** — port from FireCore `VispyMolBrowser.py` plugin architecture. See `doc/ARCHITECTURE_ROADMAP.md` §1
- Topology fully independent of hex grid (hex as guideline only; off-grid atoms already OK)

---

## 8. Infrastructure

### ✅ Working
- **pytest marks** — `gpu`, `visual`, `slow` markers registered in `pytest.ini`
- **OpenCL device selection** — auto-selects NVIDIA GPU, sets `PYOPENCL_CTX`
- **Kernel include guards** — `gridFF.cl`, `surface.cl` have `#ifndef` guards preventing redefinition
- **Test helpers** — `parity.py` (RMSE, correlation), `geometry.py` (bond lengths, angles), `scan.py` (z/x scans)

### ⚠️ In Progress
- **No `pyproject.toml` / `setup.py`** — package not formally installable via pip
- **Legacy `ocl_GridFF_new.py`** — partial import fixes, still has hardcoded `PYOPENCL_CTX` override

### ✅ Working (updated)
- **`spammm/globals.py`** — centralized verbosity: `DEBUG_PRINT_LEVEL`, `DEBUG_SAVE_LEVEL`, `DEBUG_PLOT_LEVEL`, `set_develop_mode()`
- **Jacobi eigendecomposition** — `Lingebra_ocl.py` + `lingebra.cl`, 6 tests passing

### ⚠️ In Progress
- **GUI verbosity consolidation** — `globals.py` exists but 284 matches for `verbose`/`verbosity` across 44 files still use per-module state. See `doc/ARCHITECTURE_ROADMAP.md` §10

### 🔲 Planned
- **Codon compilation** — compile core geometry/topology/assembly modules with Codon for performance (after core functionality stable)
- **ManipulationTrajectory** — full manipulation trajectory recording and replay
- **Presentation tools** — `html_pres.py` for LLM-generated HTML slides. See `doc/ARCHITECTURE_ROADMAP.md` §3

---

## 9. Integration

### ⚠️ In Progress
- **Relaxed scan H2O/NaCl** — test stub
- **Relaxed scan benzene/NaCl** — test stub
- **Visual relaxed scan** — test stub

### 🔲 Planned
- **Full pipeline: molecule → relax → dock → AFM** — end-to-end integration test, not implemented
- **Autonomous adsorption-structure search** — GPU-parallelized global optimization of molecular geometries on surfaces
- **AFM manipulation trajectory optimization** — `ManipulationPathOpt.py` exists, needs extension

---
---

## Summary Table

| Category | ✅ Working | ⚠️ In Progress | 🔲 Planned | Total |
|----------|-----------|----------------|------------|-------|
| Topology & Editing | 14 | 2 | 8 | 24 |
| Force Fields | 9 | 10 | 6 | 25 |
| Surface Interactions | 8 | 9 | 2 | 19 |
| SPM / AFM / STM | 7 | 6 | 4 | 17 |
| Rigid Body Dynamics | 4 | 1 | 2 | 7 |
| Quantum Backends | 7 | 1 | 1 | 9 |
| GUI | 0 | 4 | 2 | 6 |
| Infrastructure | 6 | 1 | 3 | 10 |
| Integration | 0 | 3 | 3 | 6 |
| Testing Strategy | 0 | 0 | 1 | 1 |
| **Total** | **55** | **37** | **32** | **124** |

---

## Priority Recommendations

### High Priority (blocking core functionality)
1. **Fix FDBM pipeline** — switch `scan_fdbm` to `relaxStrokesTilted`, fix grid coordinate mapping, fix density visualization
2. **Fix UFF CH4 relaxation** — topology/parameter assignment bug in `UFFbuilder.py`
3. **Fix UFF NVE conservation** — array broadcasting `(5,3)` vs `(1,5,3)` shape mismatch
4. **Fix folded poly kernel power sequence** — support doubling powers instead of sequential
5. **Test GridFF + GridFFRelaxedScan** — core surface interaction path, completely untested

### Medium Priority (extending capabilities)
6. **Implement topology test helpers** — `TopologySnapshot`/`TopologyDiff` in `tests/helpers/topology_test.py`, `render_graph()`/`render_before_after()` in `geometry.py`, `--visual` pytest flag. Foundation for all editing tests (see §10).
7. **KekuleSolver** — automatic bond order assignment, enables correct SPFF parameter assignment for PAHs
8. **QEq standalone module + GPU** — generalize from AFMulator method, integrate with UFF/SPFF
9. **Test RigidBodyDynamics + RigidBodyAFM** — implemented but zero test coverage
10. **Test DFTB+ density projection** — fix empty density slices, verify against reference
11. **PAH / graphene ribbon builder** — high-level API on top of KekuleBackend hex grid

### Future / Research
12. **Reactive forcefield** — bond breaking/forming during manipulation
13. **RigidAtom forcefield (RRsp3)** — port from FireCore, anisotropic atom interactions with cluster-sorted PBD
14. **Structure editing from FireCore** — port selection query system, bridge manipulation, ScriptRunner, polymer/nanocrystal builders
15. **Lateral force microscopy with dissipation**
16. **Constant-current AFM**
17. **Near-field photoluminescence simulation**
18. **Quantum dot solver for dIdV charging rings**
