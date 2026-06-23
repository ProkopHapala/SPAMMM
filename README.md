# SPAMMM — FireCore

**Scientific Python AFM/MM Molecular Manipulator**

SPAMMM (Scanning Probe Accelerated Modeling of Microscopy and Manipulation) is the integrated SPM simulation platform also known as **FireCore**. It is designed for high-throughput, end-to-end simulation of bond-resolved atomic force microscopy (BR-AFM) and scanning tunneling microscopy (BR-STM), from molecular structure preparation and relaxation to image generation and surface manipulation.

The platform combines fast local-orbital DFT(B) methods with classical grid-projected force fields inside a unified Python framework, with GPU-accelerated engines at every performance-critical step. It can be used interactively through a GUI or scripted in Python.

## What SPAMMM Does

SPAMMM streamlines the full simulation pipeline for molecules on surfaces:

- **Molecular structure design** — Draw or load molecules in 2D/3D; perform interactive topology editing (hexagonal rings, bond creation/deletion, atom passivation, pi/n-pi toggling) inside the GUI.
- **Geometry relaxation** — Relax structures with fast GPU-accelerated force fields (UFF, SPFFsp3) or with DFTB, using FIRE or velocity-Verlet MD.
- **Surface docking and assembly** — Drag and place molecules on substrates, build assemblies, and run rigid-body or flexible docking using GridFF, Ewald2D, and folded-atomic-function models.
- **AFM simulation** — Generate AFM images using either a simple LJ/Morse + point-charge probe-particle model, or the full-density-based model (FDBM). For FDBM, electron density is projected onto a grid, Pauli and Hartree potentials are computed, and van der Waals contributions are added to build the total probe-sample interaction potential.
- **STM simulation** — Project molecular orbitals onto a grid via DFTB local basis set or FFT-based approaches. Bond-resolved STM is achieved by sampling STM at the probe-particle positions obtained from AFM relaxation, which distorts the orbital images and highlights bond edges.
- **Autonomous manipulation and optimization** — GPU-parallelized engines support global optimization of molecular geometries, AFM manipulation trajectories, and polymorph adsorption structures on surfaces, sampling millions of configurations per second.
- **Excitonic and charged systems** — Supports scanning-probe imaging of coupled excitonic and charged states in molecular assemblies.

## Why This Exists

Probe-particle model (PPM) and full-density-based model (FDBM) simulations, combined with Chen's derivative rules applied to molecular orbitals, have become standard tools for BR-AFM and BR-STM. GPU implementations can produce hundreds of simulated AFM volumes per second, but the upstream preparation of DFT inputs — electron density, molecular orbitals, and relaxed adsorption geometries — remains a bottleneck. SPAMMM removes that bottleneck by integrating drawing, relaxation, surface docking, and image generation in one platform, keeping every heavy stage on the GPU where possible.

## Key Components

- `spammm/GUI/KekuleExplorerGUI.py` — Interactive VisPy + PyQt5 molecular editor and main GUI.
- `spammm/forcefields/` — GPU-accelerated force fields (UFF, SPFF), MD, rigid-body dynamics, and assembly.
- `spammm/SPM/` — AFM simulation, FDBM pipeline, manipulation path optimization.
- `spammm/quantum/` — DFTB+ backend and GPU density/orbital grid projection.
- `spammm/surfaces/` — GridFF, Ewald2D, SurfaceEwald, folded atomic functions, substrate builder.
- `spammm/topology/` — Molecular topology, FF parameter parsing, Kekule editing backend.
- `kernels/` — OpenCL kernels for relaxation, force fields, surface sampling, and density projection.
- `data/` — Element, atom, bond, angle, and dihedral parameter files plus test molecules and substrates.
- `tests/` — Pytest suite (topology, surface/Ewald, forcefield, AFM, integration) with helpers.
- `doc/` — Architectural and topical audit documents, test design, agent protocols.

## Usage Modes

- **Interactive GUI** — Launch the molecular editor to draw, relax, dock, and simulate images in a single visual workflow.
- **Python scripting** — Call the workhorse modules directly to build batch pipelines, run high-throughput scans, or embed components into larger workflows.

## Documentation

For the detailed file layout, dependency chains, and current architectural notes, see [`MANIFEST.md`](MANIFEST.md).

## License

See the repository for licensing information.
