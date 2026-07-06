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
- [x] Export Kekulé structures
- [x] QEq module + GUI extension
- [x] DFTB H-bond scan (`hbond_scan.py`) + topology tests
- [x] Reaction-coordinate scan: `coordinate_scan.py`, `ScanDataset`, GUI extension, `test_scan_dataset`
- [x] Contact-surface GPU prototype + static/elastic design docs
- [x] Assembly clash scoring + `AssemblyPlot` + `testplot_assembly`
- [x] Test system L0/L1/L2 (`TEST_DESIGN`, `review.py`, `--develop`, `testplot_*`)

---

## Soon

- [ ] Wire contact surface into `AFMulator` / `RigidBodyAFM`
- [ ] Contact-surface L0 pytest parity (brute vs separable/PIC)
- [ ] Fix H not connected after molecule load
- [ ] Create bond between two existing atoms; reliable atom removal
- [ ] MOL/MOL2/XYZ export + clipboard polish
- [ ] Decouple `ascii_art_heterocycle.py` from `KekulePure.py`
- [ ] Molecule fragments/groups + automatic bridge search
- [ ] FireCore port: graph atom selection (SMARTS-like)
- [ ] FireCore port: bridge insert/collapse, group attach, polymer builder
- [ ] Folded poly basis power-sequence fix
- [ ] FDBM pipeline: `relaxStrokesTilted`, non-degenerate df/density
- [ ] Roll L1 `review.py` artifacts to more topology tests
- [ ] `test_integration.py` relaxed-scan stubs

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
- [ ] Pentagon/heptagon drawing; SMILES builder
- [ ] Molecular browser + AFM thumbnails
- [ ] Interactive MD drag constrained to mouse ray
- [ ] LAMMPS/GROMACS export; fragment library
- [ ] Consolidate GUI verbosity → `spammm.globals`
- [ ] Sync stale `FeatureChecklist.md`

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

### Human ToDo (non-agent)
| Item | Files |
|------|-------|
| Quick items | `doc/ToDo/ToDo.md` |
| GUI wishes | `doc/Tasks/ToDO_GUI.md` |
