# SPM/

Scanning Probe Microscopy simulation — AFM and STM. GPU-accelerated via PyOpenCL.

- **AFM.py** — AFMulator: LJ/Morse + FDBM tip-sample interactions, PP relaxation, image generation. **Stiffness SSOT:** `stiffness_Nm_to_eVA2` / `K_LAT_HAPALA_*` (internal = eV/Å²; GUI = N/m).
- **AFM_utils.py** — High-level AFM utilities, plotting, FDBM orchestration
- **ModularPipeline.py** — Staged AFM/STM pipeline (S1–S6) with disk caching
- **ManipulationPathOpt.py** — AFM manipulation path optimization
- **ScanUtils.py** — Trajectory generators (grid/line/rotational/tilted)

**Caveats (Jul 2026):** (1) PP `K_LAT` must not confuse N/m with eV/Å² — see `doc/Tasks/AFMTesting.md` lessons. (2) Prefer grid `step ≤ 0.1 Å` for ES hex symmetry.
