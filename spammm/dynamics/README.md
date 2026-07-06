# dynamics/

Normal-mode and vibrational analysis for isolated molecules — Hessian assembly, rigid-mode removal, frequency/mode extraction, and top-view plotting. Separate from MD relaxation (`FFController`) and from future phonon-band work.

- **Vibrations.py** — SSOT: `run_vibrations(mol, backend='uff'|'spff'|'dftb')`, rigid-body projector, unit conversion (`freq_cm1_to_unit`), `VibrationResult` dataclass
- **VibrationPlot.py** — `make_mode_figure` (xy arrows + seismic z circles), PNG export via `plot_softest_modes`
- **`__init__.py`** — exports `run_vibrations`, `VibrationResult`

**Backends:** DFTB+ `SecondDerivatives` (`DFTB_utils`); UFF/SPFF central finite difference on GPU forces (`forcefields/FFEvaluator.py`).

**GUI:** `spammm/GUI/VibrationExtension.py` (ExtensionManager key `vibrations`).

**Tests:** `tests/test_vibrations.py` → artifacts in `debug/test_vibrations/`.

**Doc:** [doc/Topics/Vibrations.md](../../doc/Topics/Vibrations.md)
