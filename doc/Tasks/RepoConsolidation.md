# Repo Consolidation Strategy — Eat FireCore / ppafm into SPAMMM

## Goal

SPAMMM is the **target product**: one PyOpenCL compute stack + one common GUI (`SPAMMM_GUI` + `ExtensionManager` plugins / mol-browser plugins). Source repos (FireCore, ppafm, NumericalMathPlayground, Utah Cosserat, …) stay as **reference + export docs**; we do **not** run their scattered CLIs/GUIs long-term.

```
FireCore / ppafm / …          SPAMMM (product)
─────────────────             ────────────────
many indep. programs    →     spammm/* modules + kernels/*.cl
many CLIs / GUIs        →     ExtensionManager + MolBrowser plugins
duplicate physics       →     one OpenCLBase path, NVIDIA-first
```

## Rules for every import task

1. **Inventory first** — use `docs/export/*.export.md` (ppafm) / FireCore topic exports; grep SPAMMM before copying.
2. **Compute in `spammm/` + `kernels/`** — no new parallel physics stacks. Thin `tests/test_*.py` / `testplot_*.py` only.
3. **GUI last** — wire as `spammm/GUI/*Extension.py` or mol-browser plugin after L0 parity.
4. **Parity** — CPU/ref vs GPU; never report PoCL as NVIDIA (Shell `all` for OpenCL).
5. **Surgical** — port one vertical slice per task; do not “clean up” the donor repo.

## Human backlog → task map (see also conference P0–P3 in `doc/ARCHITECTURE_ROADMAP.md` §TOC)

| # | Human item | Standalone task | Conf. pri |
|---|------------|-----------------|-----------|
| 4 | Prolonged DFTB radial basis | `doc/Tasks/ProlongedRadialBasis_DFTB.md` | **P0** |
| — | Molecule@surface relax (FAF/LFF) | `doc/Tasks/PerfBenchmark_Relaxation.md` | **P0** |
| — | pip install / packaging | `doc/Tasks/PipInstall_Packaging.md` | **P1** |
| 2 | PPAFM Kriging → GridFF | `doc/Tasks/Import_KrigingGridFF.md` | **P1** |
| 7 | STM mio/3ob/prolonged + pySCF cubes | `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md` | **P1** |
| — | Site Pauli \(A,\beta\) + transferability | `doc/Tasks/Pauli_A_beta_KrigingTransferability.md` | **P1** |
| 3 | PME charge rings + MC / Hubbard / MQCA | `doc/Tasks/Import_ChargeRings_PME.md` (A+D+F Done); pose glue `RigidMoleculePose_SSOT.md` | **P1** |
| 8 | Kekulé π → exponential RI density | `doc/Tasks/Kekule_ExponentialDensityFit.md` | **P2** |
| 9 | Fast 2.5D contact-surface AFM | `doc/Tasks/Fast_2p5D_AFM_ContactSurface.md` | **P1** (parity bisect; was P2) |
| — | Frenkel / TEPL | `doc/Ideas/FrenkelRigidFF.chat.md` (ideas only) | **P2** |
| 5 | Dyson orbitals + DFTB+ for STM | `doc/Tasks/DysonOrbitals_DFTB_STM.md` | **P3** |
| 6 | OpenCL/weave/jit FF optimizer driver | `doc/Tasks/FF_Optimizer_OpenCL_Driver.md` | **P3** |
| 1 | Stable Cosserat rods (coarse PTCDA) | `doc/Tasks/Import_CosseratRods_PTCDA.md` | **P3** |

## Related already in SPAMMM

- Architecture: `doc/ARCHITECTURE_ROADMAP.md` (mol browser plugins, GridFF, pySCF)
- Agent index: `doc/ToDo/ToDo.agents.md`
- FireCore map: `doc/FireCore_migration_codemap.md`
- AFM/STM overview: `doc/afm_stm_simulation.md`

## Status

**unverified** — strategy doc only; individual tasks track their own progress. Do not mark Done without USER confirmation.
