# SPM/

Scanning Probe Microscopy simulation — AFM and STM. GPU-accelerated via PyOpenCL.

- **AFM.py** — AFMulator: core AFM simulator with LJ/Morse + FDBM tip-sample interactions, probe-particle relaxation, image generation
- **AFM_utils.py** — High-level AFM utilities, plotting, FDBM orchestration (combines AFM.py physics with QM density providers)
- **ModularPipeline.py** — Staged AFM/STM pipeline (S1–S6) with disk caching; only recomputes stages affected by parameter changes
- **ManipulationPathOpt.py** — AFM manipulation path optimization via evolutionary/genetic algorithms
- **ScanUtils.py** — Trajectory generators: grid scans, line scans, rotational sweeps, tilted approaches (rigid-body transforms)
