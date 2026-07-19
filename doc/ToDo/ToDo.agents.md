# Agent ToDo Index

High-level status for agent-driven work. Details in linked docs — do not duplicate prose here.

---

## Done

- [x] FireCore core port: SPFF, UFF, GridFF, Ewald2D, FAF, AFM, STM, DFTB, Assembly, QEq
- [x] OpenCL kernel modular split + `kernels/README.md`
- [x] `KekulePure` solver + topology tests + `testplot_kekule`
- [x] ASCII-art heterocycles + `ascii_art_heterocycle.py`
- [x] `MoleculeEditorBackend` rename + `KekuleBackend` compat shim
- [x] Headless molecular editing tests (`TopologySnapshot`, `test_editing_ops`)
- [x] `PackedMolecule` + undo stack + clipboard tests
- [x] Fragment/bridge detection (`test_fragmentation`)
- [x] Create bond between two existing atoms; reliable atom removal (ID-based pipeline)
- [x] MOL/MOL2/XYZ export + clipboard polish
- [x] Relaxation GUI extension (FFExtension)
- [x] Export Kekulé structures
- [x] QEq module + GUI extension
- [x] DFTB H-bond scan (`hbond_scan.py`) + topology tests
- [x] Reaction-coordinate scan: `coordinate_scan.py`, `ScanDataset`, GUI extension, `test_scan_dataset`
- [x] Contact-surface GPU prototype + static/elastic design docs
- [x] Assembly clash scoring + `AssemblyPlot` + `testplot_assembly`
- [x] Test system L0/L1/L2 (`TEST_DESIGN`, `review.py`, `--develop`, `testplot_*`)
- [x] Corner ring placement: n-gon sharing 2 bonds at inner corner (circumcircle, vector math)
- [x] Unified Ring mode: edge ring + corner ring + hex-grid ring in one mode (`EditModeHandlers.RingMode`)
- [x] Grid geometry transforms: transpose, flip X, flip Y (grid + atoms) (`transform_atoms`)

---

## Soon

- [ ] Wire contact surface into `AFMulator` / `RigidBodyAFM`
- [ ] Contact-surface L0 pytest parity (brute vs separable/PIC)
- [ ] Fix H not connected after molecule load
- [ ] Decouple `ascii_art_heterocycle.py` from `KekulePure.py`
- [ ] Molecule fragments/groups + automatic bridge search
- [ ] FireCore port: graph atom selection (SMARTS-like)
- [ ] FireCore port: bridge insert/collapse, group attach, polymer builder
- [ ] Folded poly basis power-sequence fix
- [ ] FDBM pipeline: `relaxStrokesTilted`, non-degenerate df/density
- [ ] **FDBM image asymmetry** — coronene df map lacks 6-fold symmetry; suspected grid misalignment or CO tip axis/roll issue (`doc/Tasks/AFMTesting.md` §Bug)
- [ ] **FDBM perf: < 1s end-to-end** — per-stage timing, eliminate Python overhead, fix `spammm.DFTB` import (T01, `doc/Tasks/PerfBenchmark_FDBM.md`)
- [ ] **UFF/SPFF relax perf: instant GUI relax** — isolate GPU vs GUI overhead, use serial kernel, reduce callback frequency (T02, `doc/Tasks/PerfBenchmark_Relaxation.md`)
- [ ] **UFF fused completeness** — add torsions/dihedrals (+ inversions) to `relax_nsteps_{local,global}_UFF` with parallel gather (not serial); parity vs multi-kernel (`doc/Tasks/PerfBenchmark_Relaxation.md` §Future work)
- [ ] **SPFF fused π terms audit** — ensure π–π / π–σ present and parity-checked in `relax_nsteps_serial` and `relax_nsteps_global` (`doc/Tasks/PerfBenchmark_Relaxation.md` §Future work)
- [ ] **Fused relax + FAF + non-covalent** — connect fused multi-step kernels to FAF substrate and NB/GridFF without per-step launch fallback (`doc/Tasks/PerfBenchmark_Relaxation.md` §Future work; §5 ARCH_ROADMAP)
- [ ] Roll L1 `review.py` artifacts to more topology tests
- [ ] Integration tests: relaxed scan H2O/benzene on NaCl (stubs removed, need real implementation)
- [ ] **Robust pySCF backend**: integrate custom modified pySCF, density grid projection, FDBM pipeline integration (§2 ARCH_ROADMAP)
- [ ] **GridFF consolidation**: unify tricubic B-spline (4ch) vs trilinear (12ch) via common interface + preprocessor kernel variants (§6 ARCH_ROADMAP)
- [ ] **Mol browser plugin port**: port `MolBrowserPlugin` system from FireCore `VispyMolBrowser.py`, write SPAMMM plugins (AFM, vibration, relax) (§1 ARCH_ROADMAP)
- [ ] **FAF substrate relaxation**: implement ForcedAtomicFunction as analytic substrate potential for on-surface molecule relaxation (§5 ARCH_ROADMAP) — prefer fused `relax_global(do_faf=True)` path once complete
- [ ] **Presentation tools**: `spammm/utils/html_pres.py` — HTML slide generator for LLM agents, auto-generate from test artifacts (§3 ARCH_ROADMAP)
- [ ] **Consolidate GUI verbosity → `spammm.globals`**: replace per-module `verbose`/`bPrint` with `globals.debug_print()` (§10 ARCH_ROADMAP)

---

## Later

- [ ] Contact-surface elastic AFM (Winkler h, K_z, indentation solve)
- [ ] FireCore port: RigidAtom XPBD / RRsp3
- [ ] FireCore port: ProjectiveDynamics implicit solver
- [ ] Reactive rigid-atom FF
- [ ] NEB / H-transfer workflows (full, beyond pm-NEB prototype)
- [ ] Phonon / FTIR / Hessian
- [ ] FMM long-range electrostatics
- [ ] Editor menu simplification + shortcuts
- [ ] 3D viewer; topology independent of hex grid
- [ ] SMILES builder (corner ring pentagon/heptagon drawing implemented)
- [ ] Molecular browser + AFM thumbnails
- [ ] Interactive MD drag constrained to mouse ray
- [ ] LAMMPS/GROMACS export; fragment library
- [ ] Sync stale `FeatureChecklist.md`
- [ ] **AABB collision relaxation**: projective/position-based with bounding boxes for multi-molecule clusters on surfaces (§7 ARCH_ROADMAP)
- [ ] **Reactive FF port**: port from `NumericalMathPlayground/topics/ReactiveFF/` for molecule editing/generation (§8 ARCH_ROADMAP)
- [ ] **Presentation planning**: define slides, test each slide's content, `test_presentation.py` (§4 ARCH_ROADMAP)

---

## References

### Test & agent workflow
| Item | Files |
|------|-------|
| Test SSOT | `doc/TEST_DESIGN.md` |
| Run modes, L0/L1/L2 | `doc/AGENTS/skills/running-tests/SKILL.md`, `tests/conftest.py`, `tests/helpers/review.py` |
| Visual demos | `tests/testplot_*.py`, `debug/README.md` |
| Reference data | `doc/AGENTS/skills/reference-data/SKILL.md`, `tests/ref_data/` |
| Repo map | `CODEMAP.md` |

### Topology & Kekulé
| Item | Files |
|------|-------|
| Module index | `spammm/topology/README.md` |
| SSOT graph | `spammm/topology/AtomicGraph.py` |
| Editor backend | `spammm/topology/MoleculeEditorBackend.py` |
| Kekulé solver | `spammm/topology/KekulePure.py` |
| ASCII → geometry | `spammm/topology/ascii_art_heterocycle.py` |
| H-bond detection | `spammm/topology/hbond_utils.py` |
| Tests | `tests/topology/test_editing_ops.py`, `test_kekule.py`, `test_ascii_art.py`, `test_fragmentation.py`, `testplot_kekule.py` |

### Reaction-coordinate / H-bond DFTB
| Item | Files |
|------|-------|
| Design | `doc/Topics/ReactionCoordinateScan.md`, `doc/Tasks/ReactionCoordinateExtension_Design.md` |
| 1D H-transfer scan | `spammm/quantum/hbond_scan.py` |
| m-D control grid + pm-NEB | `spammm/quantum/coordinate_scan.py` |
| Trajectory I/O | `spammm/topology/scan_dataset.py` |
| GUI | `spammm/GUI/ReactionCoordinateExtension.py`, `spammm/GUI/rc_esp_view.py` |
| Tests | `tests/topology/test_hbond_scan.py`, `test_scan_dataset.py`, `testplot_hbond_scan.py` |
| Vibrations context | `doc/Topics/Vibrations.md` |

### Contact surface (quasi-2D AFM)
| Item | Files |
|------|-------|
| Static design | `doc/Topics/AFM/ContactSurface_Static.md` |
| Elastic design (Phase 2) | `doc/Topics/AFM/ContactSurface_Elastic.md` |
| GPU + Python | `kernels/contact_surface.cl`, `spammm/surfaces/ContactSurface.py` |
| Integration target | `spammm/SPM/AFM.py`, `kernels/AFM.cl` |
| Visual test | `tests/testplot_contact_surface.py` |
| Parity patterns | `tests/test_surface.py`, `tests/testplot_folded_surface_scan.py` |

### Assembly & surfaces
| Item | Files |
|------|-------|
| Assembly OCL | `spammm/forcefields/Assembly.py`, `kernels/assembly.cl` |
| Assembly plots | `spammm/forcefields/AssemblyPlot.py`, `tests/testplot_assembly.py` |
| GridFF / Ewald / FAF | `spammm/surfaces/GridFF.py`, `Ewald2D.py`, `FoldedRigid.py`, `kernels/surface.cl`, `kernels/gridFF.cl` |
| Surface overview | `doc/surface_interactions.md`, `spammm/surfaces/README.md` |

### Kernels
| Item | Files |
|------|-------|
| Index + composition | `kernels/README.md` |
| Pairwise potentials | `kernels/Forces.cl`, `kernels/common.cl` |

### FireCore migration (future ports)
| Item | Files |
|------|-------|
| Migration map | `doc/FireCore_migration_codemap.md` |
| Feature inventory | `FeatureChecklist.md` |
| Rigid body + folded gap | `doc/Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md` |

### AFM / STM
| Item | Files |
|------|-------|
| Overview | `doc/afm_stm_simulation.md` |
| AFMulator | `spammm/SPM/AFM.py`, `AFM_utils.py`, `kernels/AFM.cl` |
| FDBM tests | `tests/SPM/test_afm_fdbm.py`, `tests/SPM/testplot_fdbm_relax.py` |

### Architecture & strategy
| Item | Files |
|------|-------|
| **Architecture roadmap** | `doc/ARCHITECTURE_ROADMAP.md` |
| Feature audit | `doc/ToDo/Features.audit.md` |
| Debugging lessons | `doc/Takeways.md` |

### External references
| Item | Files |
|------|-------|
| FireCore mol browser | `/home/prokop/git/FireCore/pyBall/GUI/VispyMolBrowser.py` |
| FireCore mol browser plugins | `/home/prokop/git/FireCore/pyBall/GUI/mol_browser_plugins/` |
| Reactive FF | `/home/prokop/git/NumericalMathPlayground/topics/ReactiveFF/` |
| FireCore RRsp3 | `/home/prokop/git/FireCore/pyBall/RigidAtomFF/RRsp3/` |

### Human ToDo (non-agent)
| Item | Files |
|------|-------|
| Quick items | `doc/ToDo/ToDo.md` |
| GUI wishes | `doc/Tasks/ToDO_GUI.md` |
