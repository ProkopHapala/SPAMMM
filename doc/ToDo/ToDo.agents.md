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
- [x] **FDBM perf Round 1+2 (T01)** — GPU tasks/FFT, dense NA, uncompressed cache, fused Poisson–ES + GPU pad/scale (`SPAMMM_AFM_FAST_S3`); benzene warm ~0.2 s; flat_1 S3+S4 ~1.4 s; see `doc/Tasks/PerfBenchmark_FDBM.md`
- [x] **LFF projective path (bring-up)** — `kernels/LFF.cl` + `LFFSolver` + `ff_type='lff'`; UFF→K₁₂/K₁₃/K₁₄; FAF outer; PTCDA harness; topic `doc/Topics/ForceFields/LFF_ProjectiveRelax.md` (**physics/GUI polish unverified** — see Soon)
- [x] **UFF fused dihedrals/inversions + FAF** — tiled gather in `relax_nsteps_{local,global}_UFF`; PTCDA harness (`doc/Tasks/PerfBenchmark_Relaxation.md`; geometry/charges **unverified**)

---

## Soon

### Repo consolidation / nc-AFM week (priority order)

SSOT priorities: `doc/ARCHITECTURE_ROADMAP.md` §TOC. Strategy: `doc/Tasks/RepoConsolidation.md`.

**P0 — conference critical**
- [ ] **PairFF+FAF map display SSOT (tomorrow)** — tip-pull/report movies must reuse Vispy `potential_to_rgba` (attractive `|Emin|` scale), not softclip+add; rebuild PTCDI GIF after (`doc/Tasks/PairFF_MapDisplay_SSOT.md`; report `doc/Reports/PairFF_TipPull_PTCDI_QEq_2026-07-28.md`)
- [ ] **Prolonged DFTB radial basis (STM+AFM)** — analyze + wire existing fit (`basis_optimizer`, `testplot_3ob_basis_tails`); selectable WFC in projection (`doc/Tasks/ProlongedRadialBasis_DFTB.md`; `doc/DFTB_basis_fit.md`)
  - **Tip-first SA (2026-07-21):** prolonged Slater is **even more important for the tip** than sample; fit systematically on CO guinea-pig; try SA for tip / sample / both — tip-only may suffice (precomputed, simpler). Dual basis: stock ES + prolonged Pauli; never charge-normalize prolonged ρ.
  - **Fukui FDBM panel (2026-07-23):** pySCF PBE/def2-SVP densities at `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/` — `pentacene`, `PTCDA`, **new** `azaindol_dimer`, `azaindol_isodimer`, `benzoicacid_dimer`, `benzoicamid_dimer` (`rho_N`/`esp_N`). Run cube-FDBM reference vs DFTB stock vs DFTB prolonged (same pattern as PTCDA strip). XYZ: `data/xyz/*.xyz`.
- [ ] **All-electron Δρ (clamp → compact NA)** — CO tip guinea-pig first (`delta_rho_clamp_compact_na`); then sample. Distinguish all-e (Psi4/pySCF ∑Z) vs DFTB valence. See `Import_KrigingGridFF.md` §CO guinea-pig.
- [ ] **Molecule-on-surface relax polish** — PTCDA+FAF USER review / charge dial-back; LFF GUI combo; FoldedRigid stability; instant GUI relax T02 (`doc/Tasks/PerfBenchmark_Relaxation.md`, `doc/Topics/ForceFields/LFF_ProjectiveRelax.md`)

**P1 — packaging + AFM/STM demos**
- [~] **Contact-surface Morse+Coulomb parity** — sphere `h₀` + `h0_R_scale=0.75` + atom-scale `bspl/scan` dx done; E/Fz profiles track brute; XY still sharper vs GridFF. **USER visual pending.** Report `doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md`; task `Fast_2p5D_AFM_ContactSurface.md`; helicene `Assembly_ContactSurface_AFM_helicene_2026-07-24.md`; Caveats §6
- [ ] **Headless SPM CLI** — `run_spm.py` + `user_guide/SPM_CLI.md`: AFM/STM + **`opt` / `smiles-afm`** wired (amp-align heights, PCA planar); gaps: BR-STM, ASCII/.mol flags, light-STM, charge-rings, FDBM↔Kriging compare, cube ES (`doc/Tasks/SPM_CLI_Headless.md`)
  - [~] **Gas-phase `opt`/`smiles-afm`** — `FFController.optimize_vacuum` (UFF/SPFF/LFF/DFTB); planar + `orientPCA`; **USER science OK** pending
  - [~] **SMILES** — `spammm/topology/smiles.py` + CLI flags; remaining: ASCII / `.mol`/`.mol2` shared resolver; GUI text box
  - [ ] **Substrate `dock` (GridFF / FAF)** — **future / harder**; thin-wrap only after USER prioritizes; task §E + `RigidBodyDynamicsWithFoldedBasisSubstrate.md`
- [ ] **Consolidate GUI↔CLI SPM backend + input protocol** — Modular FAST_S3 ↔ CLI-legacy Stage3–4 **parity USER-confirmed** (~5.5×); **next: remove `_run_from_density` from `run_spm.py afm`**, fold onto ModularPipeline; GUI Pauli/FIRE/scan defaults aligned to CLI SSOT. Remaining: `SPMJobSpec` / `run_spm_job`. (`doc/Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md`, `doc/TopicalAudit/AFM_FDBM.md`)
- [ ] **pip install / packaging** — `pyproject.toml`, kernels+data findable (`doc/Tasks/PipInstall_Packaging.md`); expose `run_spm` console script
- [ ] **Kriging / RBF z-scan → GridFF** — ported modules exist; FDBM-cube ES vs Kriging still open; **grid z/XY alignment is CRITICAL** (`doc/Tasks/Import_KrigingGridFF.md`; topic `doc/Topics/AFM/KrigingGridFF_DFT_vs_FDBM.md`; reports `doc/Reports/Kriging_*.md`)
  - CO guinea-pig for ES Δρ + tip prolonged Pauli (notes in task file)
- [ ] **Site-resolved Pauli \(A,\beta\) + transferability** — fit \(A\) (and β) at atoms/bonds/rings vs Kriging; optionally Kriging-interpolate parameter maps; spread across site types & molecules (`doc/Tasks/Pauli_A_beta_KrigingTransferability.md`)
- [ ] **STM orbital compare (mio / 3ob / prolonged + pySCF cubes)** — pentacene + PTCDA HOMO/LUMO panels (`doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md`); CLI: `run_spm.py stm *`; depends on prolonged WFC
- [ ] **Charge rings PME + Hubbard/MQCA** — OpenCL into SPAMMM; later CLI imaging (`doc/Tasks/Import_ChargeRings_PME.md`)
- [ ] **Light-STM** — optically driven / excited-state STM channels (CLI ToDo; no module yet)

**P2 — secondary if time**
- [ ] **Kekulé → exponential RI density** — atom+bond Slater fit from Kekulé π orders vs DFT/DFTB (`doc/Tasks/Kekule_ExponentialDensityFit.md`)
- [ ] **Fast 2.5D AFM (contact surface) finish** — hybrid decision; ND flag; L0 asserts — **parity bisect promoted to P1 above**; elastic stays Later (`doc/Tasks/Fast_2p5D_AFM_ContactSurface.md`)
- [ ] **Frenkel Hamiltonian / TEPL** — design only today (`doc/Ideas/FrenkelRigidFF.chat.md`); no module yet

### Existing Soon

- [ ] Fix H not connected after molecule load
- [ ] Decouple `ascii_art_heterocycle.py` from `KekulePure.py`
- [ ] Molecule fragments/groups + automatic bridge search
- [ ] FireCore port: graph atom selection (SMARTS-like)
- [ ] FireCore port: bridge insert/collapse, group attach, polymer builder
- [ ] Folded poly basis power-sequence fix
- [ ] FDBM pipeline: `relaxStrokesTilted`, non-degenerate df/density
- [ ] **FDBM image asymmetry** — coronene df map lacks 6-fold symmetry; suspected grid misalignment or CO tip axis/roll issue (`doc/Tasks/AFMTesting.md` §Bug)
- [ ] **FDBM perf polish (T01 remaining)** — async/skip stage cache for interactive GUI; optional coarse `step=0.15` interactive (not default); optional skip F download when NO_IO (`doc/Tasks/PerfBenchmark_FDBM.md` T-next-4)
- [ ] **UFF/SPFF/LFF relax perf: instant GUI relax (T02)** — GUI callback overhead; wire UFF+LFF into FFExtension combo; prefer LFF for interactive morphing (`doc/Tasks/PerfBenchmark_Relaxation.md`)
- [ ] **SPFF fused π terms audit** — ensure π–π / π–σ present and parity-checked in `relax_nsteps_serial` and `relax_nsteps_global` (`doc/Tasks/PerfBenchmark_Relaxation.md`)
- [ ] **Fused relax + molecule–molecule NB / GridFF** — NB/GridFF in fused multi-step without per-step launch fallback (`doc/Tasks/PerfBenchmark_Relaxation.md`)
- [ ] **FAF + charge dial-back (PTCDA)** — fused FAF wired; USER noted charges “over-did it”; UFF vs SPFF `dOCdz` gap; pending review of `debug/test_relax_ptcda_faf/`
- [ ] **LFF polish** — energy accumulator; default `nOuter≈50` for SPFF-like bend; GUI combo; mols >64 atoms; optional JS-style K₁₄ \(l_0\) (`doc/Topics/ForceFields/LFF_ProjectiveRelax.md`)
- [ ] Roll L1 `review.py` artifacts to more topology tests
- [ ] Integration tests: relaxed scan H2O/benzene on NaCl (stubs removed, need real implementation)
- [ ] **Robust pySCF backend**: integrate custom modified pySCF, density grid projection, FDBM pipeline integration (§2 ARCH_ROADMAP)
- [ ] **GridFF consolidation**: unify tricubic B-spline (4ch) vs trilinear (12ch) via common interface + preprocessor kernel variants (§6 ARCH_ROADMAP)
- [ ] **Mol browser plugin port**: port `MolBrowserPlugin` system from FireCore `VispyMolBrowser.py`, write SPAMMM plugins (AFM, vibration, relax) (§1 ARCH_ROADMAP)
- [ ] **Presentation tools**: `spammm/utils/html_pres.py` — HTML slide generator for LLM agents, auto-generate from test artifacts (§3 ARCH_ROADMAP)
- [ ] **Consolidate GUI verbosity → `spammm.globals`**: replace per-module `verbose`/`bPrint` with `globals.debug_print()` (§10 ARCH_ROADMAP)

---

## Later

### Repo consolidation (P3 / post-conference)

- [ ] **Contact-surface elastic Phase 2** — Winkler \(h,K_z\) (`ContactSurface_Elastic.md`); static 2.5D tracked under Soon P2 `Fast_2p5D_AFM_ContactSurface.md`
- [ ] **Dyson Level-1/2/3** — `doc/Tasks/DysonOrbitals_DFTB_STM.md` (**P3**; after STM basis compare)
- [ ] **OpenCL/JIT FF fit driver** — `doc/Tasks/FF_Optimizer_OpenCL_Driver.md` (**P3**)
- [ ] **Stable Cosserat / cassette rods** — `doc/Tasks/Import_CosseratRods_PTCDA.md` (**P3**)
- [ ] Charge-rings MC fit vs full experimental NPZ / QmeQ
- [ ] Frenkel implementation (if not started under Soon P2)

### Existing Later

- [ ] **Refactor large modules** — analysis only for now: `doc/Tasks/Refactor_LargeModules.md` (`AFM_utils` ~5.3k L primary; Surface_utils / atomicUtils / editor / AFM.py secondary). **No code moves until USER approves split.**
- [ ] FireCore port: RigidAtom XPBD / RRsp3
- [ ] FireCore port: full ProjectiveDynamics / XPBD (LFF is the OpenCL spring-Jacobi path for adsorbates; broader XPBD still open)
- [ ] Reactive rigid-atom FF
- [ ] NEB / H-transfer workflows (full, beyond pm-NEB prototype)
- [ ] Phonon / FTIR / Hessian
- [ ] FMM long-range electrostatics
- [ ] Editor menu simplification + shortcuts
- [x] 3D editor view mode (`b2Dview` / Enter; ortho; Ring atom/bond/COG; hex 2D-only) — task `doc/Tasks/GUI_Editor_3D_ViewMode.md`
- [ ] Topology fully independent of hex grid (hex is guideline only; atoms off-grid already OK)
- [~] SMILES builder → `AtomicGraph` — parser + CLI wired (`spammm/topology/smiles.py`, `smiles-afm`); GUI text box open — `SPM_CLI_Headless.md` §C + `ARCHITECTURE_ROADMAP` §9 / T07
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

### Contact surface (quasi-2D / 2.5D AFM)
| Item | Files |
|------|-------|
| **Task SSOT** | `doc/Tasks/Fast_2p5D_AFM_ContactSurface.md` |
| **Assembly AFM task** | `doc/Tasks/Assembly_AFM_Pipeline.md` |
| **Parity report (SSOT)** | `doc/Reports/ContactSurface_2p5D_vs_GridFF_2026-07-24.md` |
| **Report (helicene pipeline)** | `doc/Reports/Assembly_ContactSurface_AFM_helicene_2026-07-24.md` |
| **Caveats** | `doc/Caveats.md` §6 |
| **Topical audit** | `doc/TopicalAudit/AFM_ContactSurface.md` |
| Static design | `doc/Topics/AFM/ContactSurface_Static.md` |
| Elastic design (Phase 2) | `doc/Topics/AFM/ContactSurface_Elastic.md` |
| GPU + Python | `kernels/contact_surface.cl`, `spammm/surfaces/ContactSurface.py` |
| Integration | `spammm/SPM/AFM.py` (`fit_*_contact*`, `run_scan_*`) |
| L0 / L2 PTCDA | `tests/SPM/test_afm_contact_surface.py`, `tests/testplot_contact_surface.py`, `tests/SPM/testplot_afm_contact_surface.py` |
| L2 helicene compare | `run_assembly_afm.py --compare-dir` |

### Assembly & surfaces
| Item | Files |
|------|-------|
| Assembly OCL | `spammm/forcefields/Assembly.py`, `kernels/assembly.cl` |
| Assembly plots | `spammm/forcefields/AssemblyPlot.py`, `tests/testplot_assembly.py` |
| Assembly→AFM CLI | `run_assembly_afm.py` |
| GridFF / Ewald / FAF | `spammm/surfaces/GridFF.py`, `Ewald2D.py`, `FoldedRigid.py`, `kernels/surface.cl`, `kernels/gridFF.cl` |
| Surface overview | `doc/surface_interactions.md`, `spammm/surfaces/README.md` |

### Kernels
| Item | Files |
|------|-------|
| Index + composition | `kernels/README.md` |
| Pairwise potentials | `kernels/Forces.cl`, `kernels/common.cl` |
| LFF projective | `kernels/LFF.cl`, `spammm/forcefields/LFFSolver.py`, `doc/Topics/ForceFields/LFF_ProjectiveRelax.md` |

### Force fields / relax
| Item | Files |
|------|-------|
| Module index | `spammm/forcefields/README.md` |
| Perf / PTCDA+FAF / LFF | `doc/Tasks/PerfBenchmark_Relaxation.md` |
| LFF topic (SSOT) | `doc/Topics/ForceFields/LFF_ProjectiveRelax.md` |
| Harness | `tests/test_relax_ptcda_faf.py`, `tests/test_relax_flat1.py`, `tests/test_relax_serial.py` |

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
| AFMulator + FAST_S3 | `spammm/SPM/AFM.py`, `AFM_utils.py`, `kernels/AFM.cl` (`fdbm_*`) |
| Modular pipeline + bench | `spammm/SPM/ModularPipeline.py`, `tests/SPM/bench_fdbm.py` |
| FDBM tests | `tests/SPM/test_afm_fdbm.py` (incl. `test_fdbm_fast_s3_parity_pauli_es`), `tests/SPM/testplot_fdbm_relax.py` |
| Perf report (T01) | `doc/Tasks/PerfBenchmark_FDBM.md` |
| STM basis compare (new) | `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md` |
| Pauli \(A,\beta\) maps (new) | `doc/Tasks/Pauli_A_beta_KrigingTransferability.md` |
| Kekulé RI density (new) | `doc/Tasks/Kekule_ExponentialDensityFit.md` |
| Kriging ↔ FDBM | `doc/Tasks/Import_KrigingGridFF.md`, `doc/Reports/Kriging_*.md` |

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
| ppafm Kriging export | `/home/prokop/git/ppafm/docs/export/interpolation.export.md` |
| ppafm charge rings export | `/home/prokop/git/ppafm/docs/export/charge_rings.export.md` |
| FireCore MQCA/Hubbard export | `/home/prokop/git/FireCore/doc/Topics/ManyBody/MQCA_Hubbard_Ising.export.md` |
| Utah Cosserat rods | https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/ |

### Repo consolidation tasks
| Item | Files |
|------|-------|
| Strategy + pri map | `doc/Tasks/RepoConsolidation.md`, `doc/ARCHITECTURE_ROADMAP.md` §TOC |
| pip install (P1) | `doc/Tasks/PipInstall_Packaging.md` |
| Prolonged DFTB basis (P0) | `doc/Tasks/ProlongedRadialBasis_DFTB.md`, `doc/DFTB_basis_fit.md` |
| Kriging → GridFF (P1) | `doc/Tasks/Import_KrigingGridFF.md` |
| Pauli \(A,\beta\) transferability (P1) | `doc/Tasks/Pauli_A_beta_KrigingTransferability.md` |
| STM mio/3ob/prolonged + DFT cubes (P1) | `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md` |
| Charge rings / PME / MQCA (P1) | `doc/Tasks/Import_ChargeRings_PME.md` |
| Kekulé exponential RI density (P2) | `doc/Tasks/Kekule_ExponentialDensityFit.md` |
| Fast 2.5D contact surface (P2) | `doc/Tasks/Fast_2p5D_AFM_ContactSurface.md` |
| Frenkel / TEPL (P2) | `doc/Ideas/FrenkelRigidFF.chat.md` |
| Dyson STM (P3) | `doc/Tasks/DysonOrbitals_DFTB_STM.md`, `doc/Dyson_orbitals_STM.chat.md`
| FF OpenCL fit driver (P3) | `doc/Tasks/FF_Optimizer_OpenCL_Driver.md` |
| Cosserat / cassette PTCDA (P3) | `doc/Tasks/Import_CosseratRods_PTCDA.md` |

### Human ToDo (non-agent)
| Item | Files |
|------|-------|
| Quick items + consolidation backlog | `doc/ToDo/ToDo.human.md` |
| GUI wishes | `doc/Tasks/ToDO_GUI.md` |
