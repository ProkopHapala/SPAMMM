# forcefields/

GPU-accelerated force field implementations and molecular dynamics. All OpenCL modules inherit from `utils/OpenCLBase.py`.

- **FFController.py** — Pure-logic orchestrator: bridges AtomicSystem → forcefield build → GPU relaxation → positions/forces download (no Qt dependency)
- **UFF_cl.py** — PyOpenCL UFF runtime: bonds, angles, torsions, inversions, LJ + electrostatic non-bonded (FIRE + velocity Verlet)
- **UFFbuilder.py** — Converts AtomicSystem to UFF topology arrays (atom types, bonds, angles, torsions, inversions, exclusions)
- **SPFF_cl.py** — PyOpenCL SPFFsp3 runtime with pi-orbital DOFs (FIRE + damped velocity Verlet)
- **SPFFbuilder.py** — Converts AtomicSystem to SPFFsp3 topology arrays (positions, neighbors, bond/angle params, pi-orbitals)
- **RigidBodyDynamics.py** — 6-DOF rigid body GPU dynamics with quaternion integration (symplectic Euler, Taylor-series quaternion exp)
- **RigidBodyAFM.py** — High-level AFM scanning: molecule on tip via harmonic spring interacting with substrate GridFF (future: `spammm/surfaces/ContactSurface.py` quasi-2D sample potential)
- **QEq.py** — Charge Equilibration via direct matrix solve (Rappe & Goddard, Cholesky+Schur default, LU backup)
- **Assembly.py** — Hexagonal SAM rigid-body packing: GPU clash scoring + Python orchestration (`run_assembly_search`, `generate_assembly_transforms`, `AssemblyOCL`). Kernel: `kernels/assembly.cl`
- **AssemblyPlot.py** — Top-view figures (height shading, clash/strain/clearance maps, XYZ export). Used by `tests/testplot_assembly.py`

## Assembly (on-surface SAM)

**Model:** fixed experimental unit cell; **6 C6-related orientations per cell** (`n_sym=6`); collision radius **1.0 Å** (softened vs VdW for rigid search).

**Workflow:**
1. Sample rotations (tilt / inplane / full3d) × fractional translations × 6-fold sym × PBC replicas
2. GPU `evaluate_packing_3d` → clash sum + min inter-molecular distance per config
3. Rank: `clash + zpenalty×z_span + pack_weight×min_dist` (flat, steric, tight packing)
4. Export best configs via `AssemblyPlot` (PNG + XYZ + `.diag`)

**Run:**
```bash
python tests/testplot_assembly.py
python tests/testplot_assembly.py --preset tetraceno --plot_best_k 3
```

**Artifacts:** `debug/testplot_assembly/assembly_{preset}_rank{N}.{png,xyz,diag}` + `_*_{clash,strain,clearance}.png`

**FireCore reference:** `FireCore/tests/tMMFF/test_assembly.py` (driver); core ported from `pyBall/OCL/Assembly.py`
