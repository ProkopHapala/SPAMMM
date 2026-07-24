[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ProkopHapala/SPAMMM)

# SPAMMM — FireCore

**Scanning Probe Accelerated Modeling of Microscopy and Manipulation**

SPAMMM (Scanning Probe Accelerated Modeling of Microscopy and Manipulation) is the integrated SPM simulation platform also known as **FireCore**. It is designed for high-throughput, end-to-end simulation of bond-resolved atomic force microscopy (BR-AFM) and scanning tunneling microscopy (BR-STM), from molecular structure preparation and relaxation to image generation and surface manipulation.

The platform combines fast local-orbital DFT(B) methods with classical grid-projected force fields inside a unified Python framework, with GPU-accelerated engines at every performance-critical step. It can be used interactively through a GUI or scripted in Python.

## What SPAMMM Does

SPAMMM streamlines the full simulation pipeline for molecules on surfaces:

- **Molecular structure design** — Draw or load molecules in 2D/3D; perform interactive topology editing (hexagonal rings, bond creation/deletion, atom passivation, pi/n-pi toggling) inside the GUI.
- **Geometry relaxation** — Relax structures with fast GPU-accelerated force fields (UFF, SPFFsp3) or with DFTB, using FIRE or velocity-Verlet MD.
- **Surface docking and assembly** — Drag and place molecules on substrates, build assemblies, and run rigid-body or flexible docking using GridFF, Ewald2D, and folded-atomic-function models. **PairFF** adds GPU rigid-body molecule–molecule docking with directional H-bonds (epairs / σ-holes), interactive FIRE, click-to-select active body, and optional **FAF** substrate with a combined PairFF+FAF potential map — see [`demos/PairFF_manual.md`](demos/PairFF_manual.md).
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
- `demos/` — User-facing demos (PairFF Vispy); start at [`demos/README.md`](demos/README.md) · [`demos/PairFF_manual.md`](demos/PairFF_manual.md).
- `data/` — Element, atom, bond, angle, and dihedral parameter files plus test molecules and substrates.
- `tests/` — Pytest suite (topology, surface/Ewald, forcefield, AFM, integration) with helpers.
- `doc/` — Architectural and topical audit documents, test design, agent protocols.
- `user_guide/` — End-user docs (CLI AFM/STM); start at [`user_guide/SPM_CLI.md`](user_guide/SPM_CLI.md).
- [`run_spm.py`](run_spm.py) — Headless SPM imaging CLI (see user guide).

## Usage Modes

- **GUI** — interactive editor and AFM/STM (`spammm/GUI/`).
- **CLI (headless)** — `python run_spm.py …` — see [`user_guide/SPM_CLI.md`](user_guide/SPM_CLI.md).
- **Python scripting / tests** — call `spammm/` APIs or `tests/` for batch pipelines and regression.

### Main entry points

| Entry | What |
|-------|------|
| [`run_spm.py`](run_spm.py) | **User CLI** — AFM (FDBM, Morse+Coulomb, Kriging), `opt` / `smiles-afm`, and STM (orbitals, current, vacuum panel) without GUI |
| [`demos/demo_pairff.py`](demos/demo_pairff.py) | **PairFF demo** — rigid-body H-bond docking (Vispy + FIRE); optional `--faf` NaCl; manual [`demos/PairFF_manual.md`](demos/PairFF_manual.md) |
| [`user_guide/`](user_guide/) | **User documentation** (start at [`user_guide/SPM_CLI.md`](user_guide/SPM_CLI.md)) |
| `spammm/GUI/` | Interactive VisPy + PyQt5 molecular editor / AFM–STM GUI |
| `spammm/SPM/` | Physics SSOT (`AFM.py`, `AFM_utils.py`, `stm_compare.py`, `KrigingGridFF.py`, `ModularPipeline.py`) |
| `tests/SPM/` | Developer diagnostics / L2 plots (thin wrappers; prefer CLI for routine imaging) |
| `doc/Tasks/SPM_CLI_Headless.md` | CLI roadmap, gaps (BR-STM, substrate relax, light-STM, charge-rings, …) |

```bash
python run_spm.py --help
python run_spm.py smiles-afm --example naphthalene --method uff
python run_spm.py afm --xyz data/xyz/benzene.xyz --projection stock
python run_spm.py afm-morse --xyz data/xyz/pentacene.xyz
python run_spm.py stm orbitals --molecule pentacene --n-near 5
python run_spm.py stm current --molecule pentacene --stm-tips s,pz,py
```

## Documentation

- **Users:** [`user_guide/README.md`](user_guide/README.md) · [`user_guide/SPM_CLI.md`](user_guide/SPM_CLI.md) · [`demos/PairFF_manual.md`](demos/PairFF_manual.md) (rigid-body PairFF)
- **Developers / layout:** [`MANIFEST.md`](MANIFEST.md) · [`doc/`](doc/) · [`doc/TopicalAudit/PairFF_RigidBody.md`](doc/TopicalAudit/PairFF_RigidBody.md) · [`doc/ToDo/ToDo.agents.md`](doc/ToDo/ToDo.agents.md)

## License

See the repository for licensing information.
