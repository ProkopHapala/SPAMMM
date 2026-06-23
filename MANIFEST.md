# SPAMMM — Scientific Python AFM/MM Molecular Manipulator

> **S**cientific **P**ython **A**FM/**M**olecular **M**anipulator  
> An integrated environment for molecular editing, assembly, and AFM/STM simulations on surfaces. Pure PyOpenCL — no C++ dependencies for core logic. Visualization via VisPy + PyQt5.

---

## What This System DOES

1. **Molecular editor** — 2D/3D drawing of molecules (hexagons, bonds, atoms, delete, pi-orbital/npi changes) via KekuleExplorerGUI (VisPy + PyQt5)
2. **Molecular browser** — browse databases of existing molecules (currently OpenGL-based; VisPy rewrite planned)
3. **Geometry relaxation** — PyOpenCL-accelerated force fields (UFF, MMFFsp3) for intra-molecular covalent + non-covalent interactions, with FIRE / velocity Verlet MD
4. **Substrate interaction** — molecule-surface interactions via GridFF (B-spline), Ewald2D, folded atomic functions
5. **AFM simulation** — both simple Morse+point-charge (AFMulator/relax.cl) and FDBM (density-based via DFTB or pySCF backends)
6. **STM simulation** — orbital projection, DOS, LDOS via DFTB `Grid_dftb.cl` (not Fireball)
7. **Rigid body dynamics** — 6-DOF rigid body AFM manipulation (molecule-on-tip scanning)
8. **Molecular assembly** — place molecules on surfaces, build assemblies, manipulation path optimization

---

## What This System DOES NOT Do

- **No Fireball** — Fortran FireCore library, OpenCL Hamiltonian, CheFSI, OMM, STM (Fireball variant), FdataParser, Grid projector (Fireball variant)
- **No psi4** — only pySCF as the ab initio backend for FDBM
- **No nanocrystals** — nanocrystal generation is out of scope
- **No C++ ctypes wrappers** — `MMFF.py` (top-level ctypes version), `cpp_utils_.py` were removed
- **No PME / Pauli_ocl / HubbardSolver** — separate research path, not part of core SPAMMM

---

## Directory Structure

```
SPAMMM/
├── cl_kernels/                 # OpenCL kernel source files
│   ├── relax.cl                # Probe-particle relaxation + LJ/Morse/charge AFM (AFMulator)
│   ├── relax_multi.cl          # MMFFsp3 multi-system MD kernel (MolecularDynamics.py)
│   ├── relax_multi_mini.cl     # Mini variant of relax_multi (historical; needs refactoring)
│   ├── UFF.cl                  # UFF force field kernels (bonds, angles, torsions, inversions)
│   ├── GridFF.cl               # B-spline grid force field interpolation kernels
│   ├── Rigid.cl                # 6-DOF rigid body dynamics + GridFF sampling
│   ├── Surface.cl              # Unified surface potential (Morse/LJ/Coulomb, folded atomic funcs)
│   └── Forces.cl               # Basic force computation kernels
│
├── data/                       # Parameter files and test geometries
│   ├── ElementTypes.dat        # Element params: RvdW, EvdW, Qbase, colors, masses
│   ├── AtomTypes.dat           # Hybridized types: C_sp2, O_hydroxyl, N_sp3, etc.
│   ├── BondTypes.dat           # Bond length l0 + stiffness k by type pair + order
│   ├── AngleTypes.dat          # Equilibrium angle + stiffness by type triple
│   ├── DihedralTypes.dat       # Torsion barrier, phase, periodicity by type quadruple
│   ├── xyz/                    # Test molecules (24 files: benzene, water, pentacene, DNA bases, etc.)
│   ├── mol/                    # Test molecules in MOL2 format (10 files)
│   └── substrates/             # Substrate surfaces (4 files: NaCl, CaF2)
│
├── doc/                        # Audit documents (architectural reference)
│   ├── afm_stm_simulation.md
│   ├── gui_audit.md
│   ├── forcefields_overview.md
│   ├── intramolecular_forcefields.md
│   ├── molecular_topology.md
│   ├── molecular_topology_editors.md
│   ├── molecular_topology_types.md
│   ├── nonbonding_forcefields.md
│   ├── surface_interactions.md
│   └── topical_audit.md
│
└── pyBall/                     # Python package (all PyOpenCL / VisPy / PyQt5 logic)
    ├── __init__.py
    ├── elements.py             # Element data: Z, colors, radii, masses, covalent radii
    ├── atomicUtils.py          # XYZ/MOL/MOL2 I/O, bond finding, geometry transforms, unit cell
    ├── AtomicGraph.py          # Object-graph molecular representation (Atom/Bond/Ring, stable IDs)
    ├── AtomicSystem.py         # Array-based molecular representation (FF conversion, file I/O)
    ├── KekuleBackend.py        # Backend for Kekule structure explorer (hex grid, passivation groups)
    ├── KekuleExplorerGUI.py    # Main GUI: VisPy 3D scene + PyQt5 panels (editor, AFM, settings)
    ├── VispyUtils.py           # AtomScene: reusable VisPy widget (atoms, bonds, forces, picking)
    ├── AFMExtension.py         # AFM panel for KekuleExplorerGUI (dirty flags, S1-S6 pipeline UI)
    ├── ExtensionManager.py     # Extension registry (AFM, MMFF, DFTB, etc.)
    ├── config_utils.py         # Configuration management (JSON config file)
    ├── globals.py              # Global debug/verbosity controls
    ├── dftb_utils.py           # DFTB+ integration (subprocess runner, sk-path management)
    ├── pyscf_utils.py          # pySCF integration for FDBM density computation
    ├── Ewald2D.py              # 2D Ewald summation (testing/parity, NOT production)
    ├── SubstrateBuilder.py     # Crystal slab generation (NaCl, CaF2 simple cubic)
    │
    ├── GUI/
    │   ├── BaseGUI.py          # PyQt5 widget helpers (button, spinBox, comboBox, etc.)
    │   ├── CollapsibleSection.py  # Collapsible UI section widget
    │   ├── MolecularBrowser.py    # Molecular database browser (OpenGL-based; needs VisPy port)
    │   └── GLGUI.py               # OpenGL widget base (dependency of MolecularBrowser)
    │   └── shaders/                 # GLSL shaders for GLGUI (cylinder, sphere, instances, text)
    │       ├── cylinder.glslf
    │       ├── instances.glslv, instances2.glslv
    │       ├── sphere.glslf, sphere_max.glslf
    │       └── text_billboard.glslf, text_billboard.glslv
    │
    ├── DFTB/
    │   ├── DFTBcore.py         # DFTB+ SCF backend (ctypes wrapper, dm/eigvec collection)
    │   ├── DFTBplusParser.py   # Parse DFTB+ wfc HSD basis files
    │   ├── Grid_dftb.py        # GPU density grid projection (STO basis via OpenCL)
    │   ├── data/
    │   │   ├── wfc.mio-1-1.hsd # DFTB basis data (mio-1-1 parameter set)
    │   │   └── wfc.3ob-3-1.hsd # DFTB basis data (3ob-3-1 parameter set)
    │   └── cl/
    │       └── Grid_dftb.cl    # OpenCL kernel for DFTB density projection
    │
    ├── OCL/                    # PyOpenCL force field & simulation modules
    │   ├── OpenCLBase.py       # Base class: buffer management, kernel loading, device selection
    │   ├── clUtils.py          # OpenCL device helpers, GridShape, GridCL
    │   ├── MMparams.py         # MMFF/UFF parameter parsing (loads .dat files)
    │   ├── MMFF.py             # PyOpenCL MMFFsp3 (bonds, angles, torsions, pi-pi, H-bond)
    │   ├── UFF.py              # PyOpenCL UFF force field (bonds, angles, torsions, inversions)
    │   ├── UFFbuilder.py       # UFF system builder (AtomicSystem → UFF topology arrays)
    │   ├── MolecularDynamics.py  # MD engine: FIRE, velocity Verlet, buffer allocation
    │   ├── RigidBodyDynamics.py  # 6-DOF rigid body dynamics (quaternion → matrix, Rigid.cl)
    │   ├── RigidBodyAFM.py     # High-level AFM scanning via rigid body dynamics
    │   ├── AFM.py              # AFMulator: LJ/Morse + point charge AFM (loads relax.cl)
    │   ├── AFM_utils.py        # AFM orchestration: FDBM helpers, density providers, plotting
    │   ├── ModularPipeline.py  # ModularAFMPipeline: staged S1-S6 with disk caching
    │   ├── GridFF.py           # GridFF_cl: B-spline grid interpolation + sampling
    │   ├── GridFFRelaxedScan.py  # Relaxed PES scan (GridFF + MMFF + MD)
    │   ├── InteractionEnergy.py  # Interaction energy scanner (OpenCL)
    │   ├── ScanUtils.py        # Quaternion utilities, scan coordinate helpers
    │   ├── SurfaceEwald.py     # PyOpenCL Ewald 2D summation (production)
    │   ├── Surface_utils.py    # GridFF metadata, folded atomic functions, surface sampling
    │   ├── Assembly.py         # Molecular assembly on surfaces (OpenCL)
    │   └── ManipulationPathOpt.py  # AFM manipulation path optimization (tip + molecule MD)
    │
    └── tests/
        ├── ocl_GridFF_new.py   # GridFF construction + sampling utilities
        └── utils.py            # Test utilities (PLQH, potential computation)
```

---

## Key Dependency Chains

### GUI
```
KekuleExplorerGUI.py
  → VispyUtils.py (AtomScene: 3D rendering, picking, dragging)
  → KekuleBackend.py (editing logic, hex grid, passivation)
    → AtomicGraph.py (object graph: Atom/Bond/Ring)
    → AtomicSystem.py (array representation)
      → atomicUtils.py (I/O, bond finding, geometry)
      → elements.py (element data)
  → GUI/BaseGUI.py (PyQt5 widget helpers)
  → GUI/CollapsibleSection.py
  → ExtensionManager.py (extension loading)
  → AFMExtension.py (AFM panel)
```

### AFM Simulation
```
AFMExtension.py → ModularPipeline.py
  → DFTB/DFTBcore.py (SCF, dm/eigvecs)  [DFTB backend]
  → DFTB/Grid_dftb.py + DFTB/cl/Grid_dftb.cl (density projection)
  → pyscf_utils.py (density from pySCF)    [pySCF backend]
  → OCL/AFM.py (AFMulator: relax.cl)
  → OCL/AFM_utils.py (FDBM helpers, CO tip, plotting)
```

### Force Fields
```
OCL/MMFF.py → OCL/OpenCLBase.py → OCL/clUtils.py (relax_multi.cl)
OCL/UFF.py → OCL/UFFbuilder.py → OCL/MMparams.py (UFF.cl)
OCL/MolecularDynamics.py → OCL/MMFF.py + OCL/OpenCLBase.py (relax_multi.cl)
OCL/RigidBodyDynamics.py → OCL/OpenCLBase.py (Rigid.cl)
OCL/RigidBodyAFM.py → OCL/RigidBodyDynamics.py
```

### Surface Interactions
```
OCL/GridFF.py → OCL/clUtils.py (GridFF.cl)
OCL/GridFFRelaxedScan.py → OCL/MMFF.py + OCL/MolecularDynamics.py
OCL/SurfaceEwald.py → OCL/clUtils.py
OCL/Surface_utils.py → GridFF metadata, folded atomic functions
ExplorerVisPy.py → OCL/GridFFRelaxedScan.py + SurfaceSampling.py
```

---

## Known Issues & Architectural Notes

### relax.cl / relax_multi.cl / relax_multi_mini.cl — Historical Mess
These three kernels exist for historical reasons and their responsibilities overlap:
- `relax.cl` — probe-particle AFM relaxation (LJ/Morse + point charges + tip dynamics). Loaded by `AFM.py`.
- `relax_multi.cl` — full MMFFsp3 multi-system molecular dynamics. Loaded by `MolecularDynamics.py`.
- `relax_multi_mini.cl` — stripped-down variant mentioned in comments but not currently loaded.

**Plan:** Significant refactoring needed. The ideal end state is a unified force field kernel or cleanly separated modules per task.

### MolecularBrowser — Needs VisPy Port
Current `MolecularBrowser.py` uses **PyOpenGL** (`OpenGL.GL`, `GLGUI.py`) not VisPy. This adds a `PyOpenGL` dependency and is architecturally inconsistent with the VisPy-based editor.

**Plan:** Port to VisPy using `AtomScene` from `VispyUtils.py`. The OpenGL code and GLSL shaders should be deprecated.

### Ewald2D.py vs SurfaceEwald.py
- `Ewald2D.py` — pure NumPy implementation. Used for testing/parity against PyOpenCL. **Not for production.**
- `SurfaceEwald.py` — PyOpenCL-accelerated Ewald 2D. **This is the production path.**

### ElementTypes.dat Location
Previously scattered in `cpp/common_resources/`. Moved to `data/` — the canonical data directory for SPAMMM.

---

## Future Considerations (Not Now)

These are noted for later expansion but **not included** in the current export:

1. **MMFFL / LMMF (Linearized MMFF)** — `pyBall/OCL/MMFFL.py` exists in FireCore. A linearized force field for fast assembly/placement without full relaxation. **Consider adding when assembly performance becomes critical.**

2. **RRsp3** — `pyBall/RigidAtomFF/RRsp3/RRsp3.cl` — cluster-sorted rigid ports + PBD/Jacobi collisions with recoils. A more advanced rigid body framework than the current `Rigid.cl`. **Consider adding for complex multi-rigid-body simulations.**

3. **GridFF Extensions** — additional GridFF utilities, precomputed grids, multi-species grids. Currently only basic GridFF is included.

4. **Fireball / OCL Hamiltonian** — `pyBall/FireballOCL/` — OpenCL Hamiltonian assembly, STM (Fireball variant), CheFSI, OMM. **May re-add if a pure-Python DFTB+ alternative is insufficient for STM accuracy.**

5. **Codon (native compiled Python)** — Consider compiling core geometry/topology preparation and assembly modules with Codon if it supports PyQt5, pyOpenCL, and VisPy. High-value targets: `KekuleBackend.py`, `UFFbuilder.py`, `Assembly.py`. **Investigate after core functionality is stable.**

6. **ManipulationTrajectory** — full manipulation trajectory recording and replay (related to `ManipulationPathOpt.py`). **Extend when AFM manipulation workflows mature.**

---

## Verification Checklist

- [ ] All source files in FireCore are still present (copy, not move)
- [ ] No files deleted from `/home/prokop/git/FireCore/`
- [ ] Kernel paths in Python files reference correct locations (update from `../../cpp/common_resources/cl/` to `cl_kernels/`)
- [ ] `data/*.dat` files are loaded from `data/` not `cpp/common_resources/`
- [ ] `pyBall/DFTB/` package is importable
- [ ] `pyBall/GUI/MolecularBrowser.py` runs (accepts OpenGL dependency for now)
- [ ] `KekuleExplorerGUI.py` launches without import errors
- [ ] `OCL/AFM.py` compiles `relax.cl`
- [ ] `OCL/MolecularDynamics.py` compiles `relax_multi.cl`
- [ ] `OCL/UFF.py` compiles `UFF.cl`
- [ ] `OCL/GridFF.py` compiles `GridFF.cl`
- [ ] `OCL/RigidBodyDynamics.py` compiles `Rigid.cl`

---

## Total Files: 120

- Python modules: 52
- OpenCL kernels: 9
- Data files (.dat): 5
- Test molecules (.xyz): 24
- Test molecules (.mol2): 10
- Substrates (.xyz): 4
- GLSL shaders: 7
- Audit documents (.md): 10
- DFTB basis data (.hsd): 2
- DFTB OpenCL kernel: 1
- Package inits: 5
- This manifest: 1
