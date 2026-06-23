# SPAMMM — Scientific Python AFM/MM Molecular Manipulator

> Scanning Probe Accelerated Modeling of Microscopy and Manipulation

## Background & Context

SPAMMM is also known as **FireCore**: an integrated QM/MM SPM simulation platform. Over the last decade, probe-particle model (PPM) and full-density-based model (FDBM) simulations, combined with Chen's derivative rules applied to molecular orbitals, have become standard tools for bond-resolved AFM and STM (BR-AFM, BR-STM). GPU-accelerated implementations can generate hundreds of simulated AFM volumes per second, making them attractive data sources for machine learning and other high-throughput applications.

The remaining bottleneck is preparing the inputs: DFT calculations of electron density and molecular orbitals, plus structural relaxation of molecules on substrates. When the adsorption geometry is unknown, this becomes a tedious, multi-tool workflow. FireCore/SPAMMM addresses this by integrating the preparation steps into one platform, combining fast local-orbital DFT(B) methods with classical grid-projected force fields.

The goal is to use GPU acceleration at every stage where it matters:

1. Load or draw a molecule (fast algorithms, no GPU needed here).
2. Relax it with a force field (UFF, SPFFsp3 — GPU) or DFTB.
3. Dock or drag the molecule on a surface (GridFF, rigid-body models, folded atomic functions — GPU).
4. Compute AFM and STM images (probe-particle relaxation, FDBM density convolution, Poisson solve, orbital projection — GPU).

The platform is accessible through an interactive GUI or via Python scripting, enabling end-to-end workflows from molecule design to simulated images, as well as autonomous searches for AFM manipulation trajectories and polymorph adsorption structures.

## What This System DOES

1. **Molecular editor** — 2D/3D drawing of molecules (hexagons, bonds, atoms, delete, pi-orbital/npi changes) via KekuleExplorerGUI (VisPy + PyQt5)
2. **Molecular browser** — browse databases of existing molecules (currently OpenGL-based; VisPy rewrite planned)
3. **Geometry relaxation** — PyOpenCL-accelerated force fields (UFF, SPFFsp3) for intra-molecular covalent + non-covalent interactions, with FIRE / velocity Verlet MD
4. **Substrate interaction** — molecule-surface interactions via GridFF (B-spline), Ewald2D, folded atomic functions
5. **AFM simulation** — both simple Morse+point-charge (AFMulator/AFM.cl) and FDBM (density-based via DFTB or pySCF backends)
6. **STM simulation** — orbital projection, DOS, LDOS via DFTB `LCAO_grid.cl` (not Fireball)
7. **Rigid body dynamics** — 6-DOF rigid body AFM manipulation (molecule-on-tip scanning)
8. **Molecular assembly** — place molecules on surfaces, build assemblies, manipulation path optimization

## Directory Structure

```
SPAMMM/
├── kernels/                    # ALL OpenCL kernel sources, flat
│   ├── AFM.cl                 # Probe-particle relaxation + LJ/Morse/charge AFM (AFMulator)
│   ├── SPFF.cl                # SPFFsp3 multi-system MD kernel (MolecularDynamics.py)
│   ├── UFF.cl                 # UFF force field kernels (bonds, angles, torsions, inversions)
│   ├── gridFF.cl              # B-spline grid force field interpolation kernels
│   ├── rigid.cl               # 6-DOF rigid body dynamics + GridFF sampling
│   ├── surface.cl             # Unified surface potential (Morse/LJ/Coulomb, folded atomic funcs)
│   ├── Forces.cl              # Basic force computation kernels
│   ├── assembly.cl            # Rigid-body assembly / collision kernels
│   ├── LCAO_grid.cl           # DFTB density/orbital grid projection
│   ├── LCAO_STM.cl            # STM orbital projection kernel
│   ├── common.cl              # Shared OpenCL utilities and defines
│   └── nonbonded.cl           # Non-bonded interaction kernels
│
├── data/                       # Parameter files and test geometries
│   ├── ElementTypes.dat        # Element params: RvdW, EvdW, Qbase, colors, masses
│   ├── AtomTypes.dat           # Hybridized types: C_sp2, O_hydroxyl, N_sp3, etc.
│   ├── BondTypes.dat           # Bond length l0 + stiffness k by type pair + order
│   ├── AngleTypes.dat          # Equilibrium angle + stiffness by type triple
│   ├── DihedralTypes.dat       # Torsion barrier, phase, periodicity by type quadruple
│   ├── xyz/                    # Test molecules (benzene, water, pentacene, DNA bases, etc.)
│   ├── mol/                    # Test molecules in MOL2 format
│   └── substrates/             # Substrate surfaces (NaCl, CaF2)
│
├── doc/                        # Audit documents + agent skills
│   └── AGENTS/
│       ├── agentic_debugging_principles.md
│       └── skills/
│
├── scripts/                    # Thin orchestration scripts
│
├── tests/                      # Pytest suite
│   ├── conftest.py             # Fixtures: xyz(), substrate(), dat()
│   ├── test_topology.py        # Bond detection, type assignment, geometry
│   ├── test_surface.py         # Ewald vs brute-force, GPU parity, visual scans
│   ├── test_forcefield.py      # UFF/SPFF relaxation, energy conservation
│   ├── test_afm.py             # AFM relaxation convergence, visual images
│   ├── test_integration.py     # Relaxed scans (molecule on substrate)
│   ├── helpers/                # Shared test utilities
│   │   ├── parity.py           # rmse, correlation, overlay_plot, assert_parity
│   │   ├── geometry.py         # bond_lengths, bond_angle, planarity, distort
│   │   └── scan.py             # z_scan, x_scan, compare_scans
│   ├── topology/               # (empty, reserved)
│   ├── forcefields/
│   ├── surfaces/
│   ├── quantum/
│   ├── SPM/
│   └── integration/
│
└── spammm/                     # Main Python package
    ├── __init__.py
    ├── elements.py             # Element data: Z, colors, radii, masses, covalent radii
    ├── atomicUtils.py          # XYZ/MOL/MOL2 I/O, bond finding, geometry transforms, unit cell
    ├── AtomicSystem.py         # Array-based molecular representation (FF conversion, file I/O)
    ├── globals.py              # Global debug/verbosity controls
    ├── config_utils.py         # Configuration management (JSON config file)
    │
    ├── utils/                  # Shared infrastructure (OpenCL, plotting, I/O, test helpers)
    │   ├── OpenCLBase.py       # Base class: buffer management, kernel loading, device selection
    │   ├── clUtils.py          # OpenCL device helpers, GridShape, GridCL
    │   └── test_utils.py       # Test utilities (PLQH, potential computation)
    │
    ├── topology/               # Molecular topology + editing operations
    │   ├── AtomicGraph.py      # Object-graph representation (Atom/Bond/Ring, stable IDs)
    │   ├── KekuleBackend.py    # Editing logic: hex grid, passivation, pi/n-pi toggle
    │   └── FFparams.py         # FF parameter parsing (loads .dat files; shared by UFF + SPFF)
    │
    ├── forcefields/            # Force field harnesses (flat, no subfolders)
    │   ├── UFF.py              # UFF: bonds + angles + torsions + inversions + non-bonded
    │   ├── UFFbuilder.py       # AtomicSystem → UFF topology arrays
    │   ├── SPFF.py             # SPFFsp3: bonds + angles + torsions + pi-pi + H-bond + non-bonded
    │   ├── MolecularDynamics.py  # MD engine: FIRE, velocity Verlet (tightly coupled to SPFF)
    │   ├── RigidBodyDynamics.py  # 6-DOF rigid body dynamics (quaternion → matrix, Rigid.cl)
    │   ├── RigidBodyAFM.py     # High-level AFM scanning via rigid body dynamics
    │   └── Assembly.py         # Molecular assembly: rigid-body collision between fragments
    │
    ├── surfaces/               # Molecule-surface interactions
    │   ├── GridFF.py           # GridFF_cl: B-spline grid interpolation + sampling
    │   ├── GridFFRelaxedScan.py  # Relaxed PES scan (GridFF + SPFF + MD)
    │   ├── SurfaceEwald.py     # PyOpenCL Ewald 2D summation (production)
    │   ├── Surface_utils.py    # GridFF metadata, folded atomic functions, surface sampling
    │   ├── SubstrateBuilder.py # Crystal slab generation (NaCl, CaF2 simple cubic)
    │   └── Ewald2D.py          # Pure NumPy Ewald 2D (testing/parity only, NOT production)
    │
    ├── quantum/                # QM backends (DFTB+, pySCF)
    │   ├── DFTB_utils.py       # DFTB+ integration (subprocess runner, sk-path management)
    │   ├── pySCF_utils.py      # pySCF integration for FDBM density computation
    │   └── DFTB/
    │       ├── DFTBcore.py     # DFTB+ SCF backend (ctypes wrapper, dm/eigvec collection)
    │       ├── DFTBplusParser.py  # Parse DFTB+ wfc HSD basis files
    │       ├── Grid_dftb.py    # GPU density grid projection (STO basis via OpenCL)
    │       └── data/
    │           ├── wfc.mio-1-1.hsd
    │           └── wfc.3ob-3-1.hsd
    │
    ├── SPM/                    # Scanning probe microscopy: AFM + STM + manipulation
    │   ├── AFM.py              # AFMulator: LJ/Morse + point charge AFM (loads AFM.cl)
    │   ├── AFM_utils.py        # FDBM orchestration: density providers, CO tip, plotting
    │   ├── ModularPipeline.py  # ModularAFMPipeline: staged S1-S6 with disk caching
    │   ├── ScanUtils.py        # Quaternion utilities, scan coordinate helpers
    │   └── ManipulationPathOpt.py  # AFM manipulation path optimization (tip + molecule MD)
    │
    └── GUI/                    # VisPy + PyQt5 interfaces (thin frontend only)
        ├── KekuleExplorerGUI.py   # Main GUI: VisPy 3D scene + PyQt5 panels
        ├── VispyUtils.py          # AtomScene: reusable VisPy widget (atoms, bonds, picking)
        ├── AFMExtension.py        # AFM panel (dirty flags, S1-S6 pipeline UI)
        ├── ExtensionManager.py    # Extension registry (AFM, SPFF, DFTB, etc.)
        ├── BaseGUI.py             # PyQt5 widget helpers (button, spinBox, comboBox, etc.)
        ├── CollapsibleSection.py  # Collapsible UI section widget
        ├── MolecularBrowser.py    # Molecular database browser (deprecated: needs VisPy port)
        ├── GLGUI.py               # OpenGL widget base (deprecated: dependency of MolecularBrowser)
        └── shaders/               # GLSL shaders for GLGUI (deprecated with MolecularBrowser)
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
  → quantum/DFTB/DFTBcore.py (SCF, dm/eigvecs)  [DFTB backend]
  → quantum/DFTB/Grid_dftb.py + kernels/LCAO_grid.cl (density projection)
  → quantum/pySCF_utils.py (density from pySCF)    [pySCF backend]
  → SPM/AFM.py (AFMulator: AFM.cl)
  → SPM/AFM_utils.py (FDBM helpers, CO tip, plotting)
```

### Force Fields
```
forcefields/SPFF.py → utils/OpenCLBase.py → utils/clUtils.py (SPFF.cl)
forcefields/UFF.py → forcefields/UFFbuilder.py → topology/FFparams.py (UFF.cl)
forcefields/MolecularDynamics.py → forcefields/SPFF.py + utils/OpenCLBase.py (SPFF.cl)
forcefields/RigidBodyDynamics.py → utils/OpenCLBase.py (rigid.cl)
forcefields/RigidBodyAFM.py → forcefields/RigidBodyDynamics.py
```

### Surface Interactions
```
surfaces/GridFF.py → utils/clUtils.py (gridFF.cl)
surfaces/GridFFRelaxedScan.py → forcefields/SPFF.py + forcefields/MolecularDynamics.py
surfaces/SurfaceEwald.py → utils/clUtils.py
surfaces/Surface_utils.py → GridFF metadata, folded atomic functions
GUI/KekuleExplorerGUI.py → surfaces/GridFFRelaxedScan.py + Surface_utils.py
```

---

## Known Issues & Architectural Notes

### AFM.cl / SPFF.cl — Historical Mess
These kernels exist for historical reasons and their responsibilities overlap:
- `AFM.cl` — probe-particle AFM relaxation (LJ/Morse + point charges + tip dynamics). Loaded by `SPM/AFM.py`.
- `SPFF.cl` — full SPFFsp3 multi-system molecular dynamics. Loaded by `forcefields/MolecularDynamics.py`.

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

1. **SPFFL / LMMF (Linearized SPFF)** — `spammm/forcefields/SPFFL.py` exists in FireCore. A linearized force field for fast assembly/placement without full relaxation. **Consider adding when assembly performance becomes critical.**

2. **RRsp3** — `spammm/forcefields/RRsp3/RRsp3.cl` — cluster-sorted rigid ports + PBD/Jacobi collisions with recoils. A more advanced rigid body framework than the current `rigid.cl`. **Consider adding for complex multi-rigid-body simulations.**

3. **GridFF Extensions** — additional GridFF utilities, precomputed grids, multi-species grids. Currently only basic GridFF is included.

4. **Fireball / OCL Hamiltonian** — `spammm/quantum/FireballOCL/` — OpenCL Hamiltonian assembly, STM (Fireball variant), CheFSI, OMM. **May re-add if a pure-Python DFTB+ alternative is insufficient for STM accuracy.**

5. **Codon (native compiled Python)** — Consider compiling core geometry/topology preparation and assembly modules with Codon if it supports PyQt5, pyOpenCL, and VisPy. High-value targets: `KekuleBackend.py`, `UFFbuilder.py`, `Assembly.py`. **Investigate after core functionality is stable.**

6. **ManipulationTrajectory** — full manipulation trajectory recording and replay (related to `SPM/ManipulationPathOpt.py`). **Extend when AFM manipulation workflows mature.**

---

## Verification Checklist

- [ ] All source files in FireCore are still present (copy, not move)
- [ ] No files deleted from `/home/prokop/git/FireCore/`
- [ ] Kernel paths in Python files reference `kernels/` directory
- [ ] `data/*.dat` files are loaded from `data/`
- [ ] `spammm/quantum/DFTB/` package is importable
- [ ] `spammm/GUI/MolecularBrowser.py` runs (accepts OpenGL dependency for now)
- [ ] `spammm/GUI/KekuleExplorerGUI.py` launches without import errors
- [ ] `spammm/SPM/AFM.py` compiles `AFM.cl`
- [ ] `spammm/forcefields/MolecularDynamics.py` compiles `SPFF.cl`
- [ ] `spammm/forcefields/UFF.py` compiles `UFF.cl`
- [ ] `spammm/surfaces/GridFF.py` compiles `gridFF.cl`
- [ ] `spammm/forcefields/RigidBodyDynamics.py` compiles `rigid.cl`

---

## Scratchpad / Workflow Notes

- **SPAMMM vs FireCore**: SPAMMM is the repository/package name; FireCore is the integrated simulation platform. In practice the terms are used interchangeably.
- **AFM approaches**:
  - Simple: LJ/Morse + point charges on the probe particle, then relax the probe particle in the force field.
  - FDBM: project electron density on a grid, convolve with tip density to get Pauli potential, solve Poisson for Hartree potential, add vdW, then project total potential back to the grid for relaxation.
- **STM approaches**:
  - FFT-based orbital projection.
  - Local basis set (DFTB) orbital projection on grid.
  - Bond-resolved STM: sample STM at the relaxed probe-particle positions from AFM; the distortion and discontinuities at bond positions produce the sharp edges.
- **Force fields**: UFF and SPFFsp3 for covalent and non-covalent intra-molecular interactions; GridFF and folded atomic functions for molecule-surface interactions.
- **Relaxation engines**: FIRE for static relaxation, velocity Verlet for MD, rigid-body 6-DOF dynamics for manipulation and docking.
- **GPU use**: OpenCL kernels in `kernels/` handle the heavy numerical work (MD, AFM, GridFF, surface sampling, DFTB density projection). GUI and topology editing stay on the CPU.
- **DFT backends**: DFTB+ via `spammm/quantum/DFTB/` and `spammm/quantum/DFTB_utils.py`; pySCF via `spammm/quantum/pySCF_utils.py` for FDBM density when needed.
- **High-throughput target**: autonomous adsorption-structure search and AFM manipulation trajectory optimization, sampling millions of geometries per second on GPU.