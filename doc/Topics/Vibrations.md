---
type: TopicalAudit
title: Vibrational Analysis
tags: [vibrations, hessian, normal-modes, forcefields, dftb, gui]
---

# Vibrational Analysis

## Summary

Normal-mode analysis for gas-phase molecules in the main GUI and test suite. Builds a Cartesian Hessian (DFTB+ native `SecondDerivatives` or GPU finite-difference on UFF/SPFF forces), projects out six rigid-body modes, diagonalizes the mass-weighted matrix, and classifies each mode by in-plane (xy) vs out-of-plane (z) displacement fraction. Visualization is a top-view quiver plot: in-plane arrows scaled to 1 Å max, atom-sized circles colored by seismic colormap for z amplitude.

**SSOT for compute:** `spammm/dynamics/Vibrations.py` (`run_vibrations`). **SSOT for plots:** `spammm/dynamics/VibrationPlot.py`. **GUI:** `spammm/GUI/VibrationExtension.py` via `ExtensionManager` key `vibrations`.

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `spammm/dynamics/Vibrations.py` | **active** | Hessian backends, rigid projection, `freq_cm1_to_unit`, `VibrationResult` |
| `spammm/dynamics/VibrationPlot.py` | **active** | `make_mode_figure`, file export helpers |
| `spammm/forcefields/FFEvaluator.py` | **active** | `make_ff_eval_fn` — single-point E,F for FD Hessian |
| `spammm/quantum/DFTB_utils.py` | **active** | `write_dftb_input_hessian`, `read_hessian` (also legacy ASE helpers) |
| `spammm/GUI/VibrationExtension.py` | **active** | Panel: backend, units, compute, clickable mode table |
| `tests/test_vibrations.py` | **active** | H2O (fast), benzene/PTCDA (`slow`), optional DFTB H2O |
| FireCore `Phonons.py` / phonon bands | **unfinished** | Not ported; see `doc/FireCore_migration_codemap.md` §2.2 |

## Pipeline

```
AtomicSystem (GUI backend.sys or .xyz)
    → Hessian H (3N×3N, eV/Å²)
        • DFTB: dftb+ Driver=SecondDerivatives → hessian.out
        • UFF/SPFF: central FD on GPU eval_fn (6N force evals)
    → H' = P H P   (P projects translations + rotations)
    → mass-weighted eigh → frequencies (cm⁻¹ stored) + mode vectors
    → drop |ν| < 20 cm⁻¹ remnants
    → ModeInfo: f_xy, f_z, character (in-plane / out-of-plane / mixed)
```

## GUI (`Vibrations` extension)

| Control | Action |
|---------|--------|
| **Backend** | `UFF` (default), `SPFF`, or `DFTB` |
| **Units** | `cm⁻¹`, `meV`, `THz`, `kcal/mol` — table + plot titles refresh without recompute |
| **Compute Modes** | Background `QThread`; fills mode table |
| **Mode table row click** | Opens/updates single plot window (`plotutils.show_in_plot_window`) |

Launch: `./run_gui.sh` → collapsible **Vibrations** panel.

## Unit conversions

Internal storage is always wavenumber (cm⁻¹). Display conversions in `freq_cm1_to_unit()`:

| Unit | Formula |
|------|---------|
| cm⁻¹ | identity |
| meV | \(E = hc\tilde\nu\) in meV |
| THz | \(\tilde\nu \cdot c / 10^{12}\) |
| kcal/mol | \(E = hc\tilde\nu\) in kcal/mol |

E_zpe column: ½ quantum in meV/kcal/mol when those units selected; otherwise eV.

## Tests

```bash
pytest tests/test_vibrations.py -m "not slow"          # H2O UFF (~0.3 s)
pytest tests/test_vibrations.py --develop -s           # + benzene, PTCDA plots
pytest tests/test_vibrations.py::test_vibrations_h2o_dftb -s   # needs DFTB_EXE
```

Artifacts: `debug/test_vibrations/*.{png,txt}`

## Parity / validation

| Check | Reference | Tolerance |
|-------|-----------|-----------|
| H2O mode count | 3N−6 = 3 | exact |
| Benzene OOP modes | several f_z > 0.75 | qualitative |
| PTCDA soft modes | OOP for planar molecule | qualitative |
| Absolute frequencies | UFF/DFTB stiff vs experiment | **not calibrated** — use for mode character, not spectroscopy |

## Open Issues

- **SPFF:** pi-orbitals frozen during FD — frequencies are partial derivatives w.r.t. nuclear positions only (same as `test_forcefield` EF tests).
- **UFF in FFController:** relaxation path still `NotImplementedError`; vibrations use `FFEvaluator` directly (works).
- **Large molecules:** FD Hessian is O(N) GPU force calls — PTCDA (38 atoms) ~1.5 s; DFTB native Hessian preferred for QM when available.
- **Phonon bands / PBC / FTIR:** not implemented — FireCore port deferred (`spammm/quantum/Phonons.py` planned).
- **Duplicate Hessian helpers:** `DFTB_utils.get_hessian_ase` exists but GUI uses native `SecondDerivatives` path.

## Related

- [intramolecular_forcefields.md](../intramolecular_forcefields.md) — UFF/SPFF force evaluation
- [FireCore_migration_codemap.md](../FireCore_migration_codemap.md) — phonon/FTIR future work
- [ContactSurface_Elastic.md](AFM/ContactSurface_Elastic.md) — planned molecular Hessian for indentation stiffness
