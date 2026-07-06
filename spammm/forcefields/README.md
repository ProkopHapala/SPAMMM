# forcefields/

GPU-accelerated force field implementations and molecular dynamics. All OpenCL modules inherit from `utils/OpenCLBase.py`.

- **FFController.py** — Pure-logic orchestrator: bridges AtomicSystem → forcefield build → GPU relaxation → positions/forces download (no Qt dependency)
- **UFF_cl.py** — PyOpenCL UFF runtime: bonds, angles, torsions, inversions, LJ + electrostatic non-bonded (FIRE + velocity Verlet)
- **UFFbuilder.py** — Converts AtomicSystem to UFF topology arrays (atom types, bonds, angles, torsions, inversions, exclusions)
- **SPFF_cl.py** — PyOpenCL SPFFsp3 runtime with pi-orbital DOFs (FIRE + damped velocity Verlet)
- **SPFFbuilder.py** — Converts AtomicSystem to SPFFsp3 topology arrays (positions, neighbors, bond/angle params, pi-orbitals)
- **RigidBodyDynamics.py** — 6-DOF rigid body GPU dynamics with quaternion integration (symplectic Euler, Taylor-series quaternion exp)
- **RigidBodyAFM.py** — High-level AFM scanning: molecule on tip via harmonic spring interacting with substrate GridFF
- **QEq.py** — Charge Equilibration via direct matrix solve (Rappe & Goddard, Cholesky+Schur default, LU backup)
- **Assembly.py** — GPU-accelerated molecular placement on surfaces via rigid-body transforms and collision detection
