# Topical Audit — SPAMMM

## Purpose

SPAMMM (Scanning Probe AFM/STM Molecular Modeling) is a Python + PyOpenCL scientific simulation package. This file is the index for topic-based audits — each section maps where a scientific concept is implemented, its status, and relationships to other topics.

## Technology Stack

- **Language**: Python (orchestration) + OpenCL `.cl` kernels (compute)
- **Acceleration**: PyOpenCL (GPU), NumPy (CPU vectorized)
- **GUI**: VisPy (Python, OpenGL)
- **QM**: DFTB+ (subprocess/C-API), pySCF
- **No C++, no JavaScript, no Fortran, no CUDA** — all compute is Python + OpenCL

## Content Guidelines

For each topic, document:

- **Overview** — What is the scientific problem being solved?
- **Implementations** — List all locations where this topic appears (file paths)
- **Status** — Active, experimental, deprecated, or unfinished
- **Relationships** — How different implementations relate
- **Notes** — Context about why multiple implementations exist, which to use, consolidation plans

## Usage

When working on a feature:
1. Check this file first to see if related work already exists
2. Update the relevant section when adding new implementations
3. Mark old implementations as deprecated when superseded
4. Add new sections for novel scientific areas

## Scope

This is **not** a replacement for:
- API documentation (use docstrings)
- Build instructions (use README)
- Tutorial examples (use `tests/`)

This **is** a supplement to:
- Help navigate the codebase
- Preserve context for long-term project continuity
- Enable systematic consolidation of duplicate efforts

---

# TOPIC HIERARCHY

## 1. Molecular Topology

**SSOT:** `AtomicGraph` (`spammm/topology/AtomicGraph.py`) — all other representations derive from it. See skill:`molecular-structure-sync`.

### 1a. Base Graph & Connectivity
- Graph representations, bond finding, neighbor lists, ring/bridge detection
- **Key files:** `spammm/topology/AtomicGraph.py`, `spammm/AtomicSystem.py`, `spammm/atomicUtils.py`
- **Audit Document:** [molecular_topology.md](molecular_topology.md)

### 1b. Type Assignment
- Atom type assignment (sp1/sp2/sp3), VSEPR geometry, parameter loading
- **Key files:** `spammm/topology/FFparams.py`
- **Audit Document:** [molecular_topology_types.md](molecular_topology_types.md)

### 1c. Kekule Bond Order Solver
- Pi-bond order optimization, aromatic detection, feasibility precheck, multi-seed localization with 6-ring validation
- **Key files:** `spammm/topology/KekulePure.py`, `spammm/topology/MoleculeEditorBackend.py`, `spammm/GUI/KekuleExtension.py`
- **Tests:** `tests/topology/test_kekule.py`, `tests/topology/test_editing_ops.py`, `tests/topology/testplot_kekule.py`
- **Artifacts:** `debug/test_kekule/` (develop-mode PNGs)
- **Caveats:** default seed can yield valence-valid but chemically invalid patterns — use `localize_kekule(validate=True, ntrials=5)`; some ASCII examples impossible (odd π count)

### 1d. Heterocycle Generation & ASCII Art
- Heterocycle structure generation and ASCII art → 2D geometry; `:` H-bond markers
- **Key files:** `spammm/topology/heterocycle_generator.py`, `spammm/topology/ascii_art_heterocycle.py`, `spammm/GUI/AsciiArtExtension.py`
- **Tests:** `tests/topology/test_heterocycle_generator.py`, `tests/topology/test_ascii_art.py`, `tests/topology/test_kekule.py`

### 1e. H-Bond Proton Transfer (DFTB rigid scan)
- Move bridging H along donor→acceptor axis; DFTB+ single-point energy profile
- **Key files:** `spammm/quantum/hbond_scan.py`, `spammm/quantum/DFTB_utils.py` (`run_dftb_sp`, `save_xyz_movie`)
- **Topology deps:** `resolve_hbond_pairs()`, `build_ascii_hbond_system()` from ASCII examples with `:`
- **Tests:** `tests/topology/test_hbond_scan.py` (L0 + slow DFTB), `tests/topology/testplot_hbond_scan.py`
- **Artifacts:** `debug/test_hbond_scan/hbond_*.{png,xyz}` — XYZ comment: `s=… f=… E=… dE=…`
- **Caveats:** 0.1 Å path grid default; rigid geometry; SCC failures skipped; uses `$DFTB_EXE` not PATH `dftb+`

### 1f. Molecular Editor (GUI backend)
- Hex-grid editing, passivation, ring ops — **not** the Kekule bond-order solver
- **Key files:** `spammm/topology/MoleculeEditorBackend.py`, `spammm/GUI/SPAMMM_GUI.py` (`SPAMMMWindow`)
- **Legacy alias:** `KekuleExplorerWindow` → `SPAMMMWindow`
- **Tests:** `tests/topology/test_editing_ops.py`, `tests/test_export_import.py`

### 1g. Editors & GUI
- Interactive molecular editor: VisPy-based GUI with topology editing
- **Key files:** `spammm/GUI/SPAMMM_GUI.py`, `spammm/GUI/VispyUtils.py`, `spammm/GUI/KekuleExtension.py`
- **Audit Document:** [molecular_topology_editors.md](molecular_topology_editors.md)
- **Feature audit:** [gui_audit.md](gui_audit.md)

## 2. Force Fields

### 2a. Intramolecular Force Fields
- UFF (universal), SPFF (sp3 with pi-orbital nodes), rigid body dynamics
- **Key files:** `spammm/forcefields/UFF_cl.py`, `spammm/forcefields/SPFF_cl.py`, `spammm/forcefields/UFFbuilder.py`, `spammm/forcefields/SPFFbuilder.py`, `spammm/forcefields/RigidBodyDynamics.py`
- **Kernels:** `kernels/UFF.cl`, `kernels/SPFF.cl`, `kernels/rigid.cl`
- **Controller:** `spammm/forcefields/FFController.py`
- **Audit Documents:** [forcefields_overview.md](forcefields_overview.md), [intramolecular_forcefields.md](intramolecular_forcefields.md)
- **Tests:** `tests/test_forcefield.py`

### 2b. Non-Bonding Force Fields
- LJ/Morse/Coulomb, exclusion schemes
- **Kernels:** `kernels/nonbonded.cl`, `kernels/nonbonded_grid.cl`
- **Audit Document:** [nonbonding_forcefields.md](nonbonding_forcefields.md)

### 2c. Charge Equilibration
- QEq charge transfer method
- **Key files:** `spammm/forcefields/QEq.py`, `spammm/GUI/QEqExtension.py`

### 2d. On-Surface Assembly (Rigid SAM)
- Rigid-body packing on a fixed hexagonal unit cell: 6 C6 orientations/cell, GPU clash scoring
- **Key files:** `spammm/forcefields/Assembly.py`, `spammm/forcefields/AssemblyPlot.py`
- **Kernel:** `kernels/assembly.cl`
- **Driver:** `tests/testplot_assembly.py` → `debug/testplot_assembly/`
- **FireCore reference:** `pyBall/OCL/Assembly.py`, `tests/tMMFF/test_assembly.py`
- **Caveats:** `radius=1.0` Å default (softened for rigid search); `align_flat` pre-aligns to surface; rotation dedup CPU-only, off by default
- **Tests:** visual demo only (`testplot_assembly.py`); no pytest L0 yet

## 3. Surface Interactions

### 3a. GridFF & Surface Potentials
- GridFF (B-spline interpolation), FoldedAtomicFunctions (compact basis), Ewald2D
- **Key files:** `spammm/surfaces/GridFF.py`, `spammm/surfaces/GridFFRelaxedScan.py`, `spammm/surfaces/FoldedRigid.py`, `spammm/surfaces/Ewald2D.py`, `spammm/surfaces/SurfaceEwald.py`, `spammm/surfaces/Surface_utils.py`, `spammm/surfaces/SubstrateBuilder.py`
- **Kernels:** `kernels/gridFF.cl`, `kernels/surface.cl`
- **Audit Document:** [surface_interactions.md](surface_interactions.md)
- **Consolidation needed:** `GridFF.py` vs `GridFFRelaxedScan.py` — overlapping functionality
- **Tests:** `tests/test_surface.py`, `tests/test_folded_relax.py`, `tests/surfaces/ocl_GridFF_new.py`

### 3b. Contact Surface (quasi-2D static AFM)
- Compact alternative to 3D GridFF for aperiodic rigid samples: separable B-spline×poly + radial PIC
- **Key files:** `spammm/surfaces/ContactSurface.py`
- **Kernel:** `kernels/contact_surface.cl` (with `common.cl` + `Forces.cl`)
- **Design:** [Topics/AFM/ContactSurface_Static.md](Topics/AFM/ContactSurface_Static.md)
- **Caveats:** h₀ height map on B-spline grid; poly doubling powers `t^(m_start·2^k)`; fit needs multi-z samples near scan plane for `Fz`; `F=∇E` convention; not wired to `AFMulator` yet
- **Tests:** visual demo `tests/testplot_contact_surface.py` → `debug/testplot_contact_surface/` (no pytest L0 yet)

## 4. AFM/STM Simulation

### 4a. AFM Simulation
- Morse/LJ force field, Full Density-Based Model (FDBM), modular pipeline (S1-S6 stages)
- **Key files:** `spammm/SPM/AFM.py`, `spammm/SPM/AFM_util.py`, `spammm/SPM/ModularPipeline.py`, `spammm/SPM/ScanUtils.py`, `spammm/SPM/ManipulationPathOpt.py`
- **Rigid body AFM:** `spammm/forcefields/RigidBodyAFM.py`
- **Kernel:** `kernels/AFM.cl`
- **GUI extension:** `spammm/GUI/AFMExtension.py`
- **Audit Document:** [afm_stm_simulation.md](afm_stm_simulation.md)
- **Tests:** `tests/SPM/test_afm_morse.py`, `tests/SPM/test_afm_fdbm.py`

### 4b. STM Simulation
- LCAO orbital projection, spectral function, DOS/STM current
- **Kernels:** `kernels/LCAO_STM.cl`, `kernels/LCAO_grid.cl`
- **Audit Document:** [afm_stm_simulation.md](afm_stm_simulation.md) (STM sections)

## 5. QM Integration (DFTB+)

- DFTB+ integration: subprocess, C-API, parsers, OpenCL grid projection, constrained scans
- **Key files:** `spammm/quantum/DFTB/DFTBcore.py`, `spammm/quantum/DFTB/DFTBplusParser.py`, `spammm/quantum/DFTB/Grid_dftb.py`, `spammm/quantum/DFTB/basis_optimizer.py`, `spammm/quantum/DFTB_utils.py`, `spammm/quantum/hbond_scan.py`, `spammm/quantum/pySCF_utils.py`
- **Audit Document:** [afm_stm_simulation.md](afm_stm_simulation.md) (DFTB sections)
- **Tests:** `tests/SPM/plot_dftb_vs_pyscf_basis.py`, `tests/SPM/plot_3ob_basis_tails.py`, `tests/topology/test_hbond_scan.py`

## 6. GUI & Visualization

- Main GUI: VisPy-based molecular editor with extension plugins
- **Key files:** `spammm/GUI/SPAMMM_GUI.py`, `spammm/GUI/BaseGUI.py`, `spammm/GUI/GLGUI.py`, `spammm/GUI/VispyUtils.py`, `spammm/GUI/MoleculeViewer.py`, `spammm/GUI/MolecularBrowser.py`, `spammm/GUI/ExtensionManager.py`
- **Extensions:** `KekuleExtension.py`, `AFMExtension.py`, `FFExtension.py`, `QEqExtension.py`
- **Design docs:** [GUI.desing.md](GUI.desing.md), [GUI_FF_Relaxation.md](GUI_FF_Relaxation.md), [GUI_topology_edit.desing.md](GUI_topology_edit.desing.md)
- **Audit Document:** [molecular_topology_editors.md](molecular_topology_editors.md), [gui_audit.md](gui_audit.md)

---

# TOPIC DEPENDENCY GRAPH

```
              ┌─────────────────────────────┐
              │  Molecular Topology (1)      │
              │  AtomicGraph (SSOT)          │
              └──────────┬──────────────────┘
                         │
              ┌──────────▼──────────────────┐
              │  Type Assignment (1b)        │
              │  → FF Parameter Loading      │
              └──────────┬──────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────────┐
  │ Intra FF (2a)│ │ Non-Bond(2b)│ │ Surface (3)     │
  │ UFF, SPFF   │ │ LJ, Morse   │ │ GridFF, Ewald2D │
  │ RigidBody   │ │ Coulomb     │ │ FoldedRigid     │
  │             │ │             │ │ ContactSurface  │
  └──────┬──────┘ └──────┬──────┘ └──────┬──────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
              ┌──────────▼──────────────────┐
              │  AFM/STM Simulation (4)     │
              │  AFM → RigidBody → GridFF   │
              │  → Tip Relaxation → Image   │
              └──────────┬──────────────────┘
                         │
              ┌──────────▼──────────────────┐
              │  QM Integration (5)         │
              │  DFTB+ / pySCF              │
              │  → H/S matrices → orbitals  │
              └─────────────────────────────┘
```

**Key dependencies:**
- Topology (1) → Type Assignment (1b): graph structure determines hybridization
- Type Assignment (1b) → Force Fields (2): types determine FF parameters
- Force Fields (2) + Surface Interactions (3) → AFM/STM (4): AFM uses both
- QM Integration (5) → AFM/STM (4): STM requires DFTB orbital data
- GUI (6) → All topics: GUI visualizes and edits everything

---

# OPEN ISSUES & CONSOLIDATION OPPORTUNITIES

| Topic | Issue | Related Files | Priority |
|-------|-------|---------------|----------|
| **GridFF variants** | `GridFF.py` vs `GridFFRelaxedScan.py` — overlapping functionality, needs consolidation | `spammm/surfaces/GridFF.py`, `spammm/surfaces/GridFFRelaxedScan.py` | Medium |
| **Contact surface** | Prototype in `ContactSurface.py`; no AFMulator integration; single-z fit underdetermines `Fz` | `spammm/surfaces/ContactSurface.py`, `kernels/contact_surface.cl` | Medium |
| **Substrate builder** | `SubstrateBuilder.py` is minimal — no CIF parsing, lattice replication, or slab cutting | `spammm/surfaces/SubstrateBuilder.py` | Low |
| **File I/O** | XYZ, MOL2 parsing in `atomicUtils.py` — no CIF or extended XYZ with Lattice support | `spammm/atomicUtils.py` | Low |
| **Linear algebra** | `Lingebra_ocl.py` wrapper exists but may overlap with NumPy functionality | `spammm/utils/Lingebra_ocl.py`, `kernels/lingebra.cl` | Low |

---

# WINDSURF CODEMAPS

* [All Windsurf Codemaps](https://windsurf.com/codemaps)

### AFM/STM Simulation
* [AFM Simulation: GPU Rigid Body Dynamics, CPU GridFF Relaxation, and Interactive GUI](https://windsurf.com/codemaps/594f7eaf-c3ab-4139-8f20-d1d2d7f8d401-fe86ab10a43f3d18)
* [AFM PyOpenCL System: Morse/LJ Path and FDBM Density-Based Path](https://windsurf.com/codemaps/9bb4c2a5-0c38-4943-abe9-254cfdcc75af-8796fe608a7d71c1)
* [AFM FDBM Pipeline: DFTB Backend & pySCF Integration Points](https://windsurf.com/codemaps/02d559c9-de47-4058-b07b-3318664b454e-fe86ab10a43f3d18)
* [Rigid Body Dynamics on Surfaces (pyOpenCL)](https://windsurf.com/codemaps/b5d9c2d2-50f0-4ba7-bc65-60db6e06e423-8796fe608a7d71c1)
* [Rigid Body Dynamics System for AFM Simulation](https://windsurf.com/codemaps/c9f13e1c-edfa-4702-814f-5036d03ea6c9-fe86ab10a43f3d18)

### STM Simulation
* [GPU Green's Function STM Implementation: Current Orbital Projection System & Planned GF Solver Integration](https://windsurf.com/codemaps/f398c2cf-5ff8-4d75-a398-c83e788e27b4-fe86ab10a43f3d18)
* [STM Simulation Pipeline: Orbital Projection & Quantum Transport](https://windsurf.com/codemaps/d0242216-c415-4f38-98f9-4c88b5dfeeb8-fe86ab10a43f3d18)
* [STM QMMM: DFTB Integration with GPU Density Projection](https://windsurf.com/codemaps/9fa40c64-e78c-42f2-9573-574936c8040d-fe86ab10a43f3d18)

### Surface Interactions
* [Interactive GridFF Scanning: PTCDA-on-CaF2 Constrained Relaxation System](https://windsurf.com/codemaps/99d506e2-223b-4ae7-bb60-8c2498fedfb9-8796fe608a7d71c1)
* [Surface Potential Evaluation: GridFF B-spline and XYZ Rigid Kernels](https://windsurf.com/codemaps/2a639fae-c9cb-407a-9d45-7b806c90c749-8796fe608a7d71c1)
* [FoldedAtomicFunctions: Surface Potential Basis Fitting System](https://windsurf.com/codemaps/c9fc44a7-57a2-47c5-906f-886fa301ccc7-8796fe608a7d71c1)
* [Molecule-Substrate Interaction Energy Scanning: Assembly, GUI Placement, Force Fields & Surface Evaluation](https://windsurf.com/codemaps/38bd3cb6-31c0-45b6-9e09-fda94257999c-8796fe608a7d71c1)
* [Molecule-on-Surface Systems: GridFF, XYZ Scanning, Surface Sampling, and Assembly](https://windsurf.com/codemaps/f8407e23-3a2e-41f1-abcf-9c15f3644c41-8796fe608a7d71c1)

### Force Fields
* [SPFF/UFF PyOpenCL Parity Infrastructure](https://windsurf.com/codemaps/8d1b056f-1502-4363-b52d-8257de4be453-8796fe608a7d71c1)
* [FitREQ_PN: Hydrogen-Bond Parameter Fitting System](https://windsurf.com/codemaps/d977d597-94b4-42c3-a92a-0cefe34a3e82-8796fe608a7d71c1)
* [FitREQ Interactive GUI: Monte Carlo Optimization & Energy Decomposition Integration](https://windsurf.com/codemaps/e25a0dfc-f9a8-42ab-b8bb-1d959037ca68-fe86ab10a43f3d18)
* [FitREQ Hydrogen Bond Fitting System - GPU-Accelerated Parameter Optimization](https://windsurf.com/codemaps/bf59a960-ac6c-4eea-b828-9bd18c3d44ac-fe86ab10a43f3d18)

### QM Integration
* [DFTB+ Python Integration: Library Interfaces, Parsers, and OpenCL Grid Projection](https://windsurf.com/codemaps/1d6b4b7c-04de-49ef-b581-12cf5bfef54a-fe86ab10a43f3d18)
* [DFTB+ Eigenvector Export for OpenCL Orbital Projection](https://windsurf.com/codemaps/845d1373-d23e-4f7d-a109-c0d8eccebea9-fe86ab10a43f3d18)
* [DFTB+ Calculation Flow: Standalone Program, C API, and Python Wrapper](https://windsurf.com/codemaps/2c157118-9d28-4a7c-a234-a49a3d464424-fe86ab10a43f3d18)
* [DFTB Reference Calculation & FDBM AFM Forcefield Comparison System](https://windsurf.com/codemaps/1153fe89-ff29-4d4b-b4a6-e97d8f37047f-fe86ab10a43f3d18)
