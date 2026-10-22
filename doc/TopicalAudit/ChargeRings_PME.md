---
type: TopicalAudit
title: Charge Rings PME (Pauli Master Equation)
tags: [STM, PME, charge-rings, OpenCL, many-body, NTCDA, NDR]
timestamp: 2026-07-30
---

# Charge Rings — Pauli Master Equation

## Summary

Many-body **charge-state STM** of molecular sites (quantum dots): tip multipole + exponential tunneling shift site energies and couplings; steady-state occupations from the **Pauli Master Equation** yield \(I\) and \(dI/dV\) maps (xy rings, xV charging diamonds, NDR). SPAMMM production path is OpenCL **`PME.cl`** via **`PauliSolverCL`** on `OpenCLBase` (NVIDIA-first). Full 2ᴺ basis is hardcoded to **N=4 sites / 16 states**; 2–3 site geometries embed inactive spectators. GUI **`ChargeRingsExtension`** is active. Hubbard MC / MQCA / MC-fit remain unported. **Sites today are abstract** (circle/JSON); connecting to the same rigid molecules PairFF/Assembly move needs pose SSOT ([`RigidBody.md`](RigidBody.md), [`RigidMoleculePose_SSOT.md`](../Tasks/RigidMoleculePose_SSOT.md)).

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| OpenCL | `kernels/PME.cl` | **active** | Identical md5 to FireCore/ppafm; tip interaction + Gauss–Jordan PME |
| OpenCL | `kernels/PME8.cl` | **active** | 8-site / 256-state; sparse iterative Euler solver; 256 threads/wg; ~13 KB local mem |
| Python | `spammm/quantum/PauliSolverCL.py` | **active** | FireCore `pauli_ocl.py` on SPAMMM `OpenCLBase` |
| Python | `spammm/quantum/PauliSolverCL8.py` | **active** | 8-site wrapper for PME8.cl; same API as PauliSolverCL |
| Python | `spammm/quantum/pauli_scan.py` | **active** | Slim xy/xV API; 2→4 embed; Wij; Ruslan + fig3 trimer params |
| Data | `data/charge_rings/` | **active** | `Ruslan_{long,short,kite}.txt`, `square_tetramer.txt`, `fig3_trimer.json` |
| Test L0 | `tests/quantum/test_pme_pauli.py` | **active** | Square mirror symmetry on RTX 3090 |
| Test L0 | `tests/quantum/test_pme8_smoke.py` | **active** | PME8 vs PME4 parity on 4-site square; max\|dI\|=4.14e-08 |
| Test L1 | `tests/quantum/test_trimer_pme8_ndr.py` | **active** | PME8 reproduces fig3 trimer NDR; max\|dSTM\|=3.6e-08; NDR min=-5.69e-06 vs PME4 -6.57e-06 |
| Demo L2 | `tests/quantum/testplot_charge_rings_ruslan.py` | **active** | Ruslan_long xV diamonds + xy V-stack |
| Demo L2 | `tests/quantum/testplot_charge_rings_trimer.py` | **active** | fig3 / symmetric trimer NDR |
| Test L0 | `tests/quantum/test_pme_trimer.py` | **active** | Symmetric apex-+y trimer: mirror + NDR |
| GUI | `spammm/GUI/ChargeRingsExtension.py` | **active** | JSON I/O, XY/xV/1D, cut overlay, state probs |
| OpenCL | FireCore `pyBall/OCL/cl/hubbard.cl` | unfinished (SPAMMM) | Dense PME ≤64; parity vs PME.cl done in FireCore |
| OpenCL | FireCore `MQCA.cl` / `MQCA_top8.cl` | unfinished (SPAMMM) | Ground state ≤16 |
| C++ | ppafm `cpp/pauli.hpp` | reference (external) | CPU/OpenMP; not primary SPAMMM path |
| Python | ppafm `tests/ChargeRings/pauli_scan.py` | reference (external) | Full scan+GUI engine (~3k LOC) |

## Parity Status

| Pair | Result | Artifacts |
|------|--------|-----------|
| Square tetramer I(x)=I(−x) | rel asymmetry ≈ 1.6×10⁻⁷ (RTX 3090) | `debug/test_pme_pauli/` |
| Four corner currents equal | exact to float print | same |
| Ruslan_long W=0.05 xy/xV | morphology matches ppafm NTCDA solver_0 | `debug/testplot_charge_rings_ruslan/` |
| fig3 trimer NDR | reproduced from `fig3_data` / `params.json` | `debug/testplot_charge_rings_trimer/` |
| PME.cl vs Hubbard dense (FireCore) | max\|dI\| ~ 1e-12 (external) | FireCore `test_pme_parity_*` — not re-run in SPAMMM yet |
| PME8 vs PME4 (4-site square, 4 spectators) | max\|dI\| = 4.14e-08 | `tests/quantum/test_pme8_smoke.py` — see [`MoleculeExtraction_PME8_2026-08-18.md`](../Reports/MoleculeExtraction_PME8_2026-08-18.md) |

## Design notes

- **PME.cl tip μ**: substrate `mu0` from `lead_params`; tip `mu1` = per-pixel `Vtips` (not `lead_params[2]`).
- **Temp**: `params['Temp']` in Kelvin → `set_lead(..., Temp*kB)` in eV (ppafm `make_configured_solver` convention).
- **W vs Wij**: scalar `W` couples all 4 embed sites including spectators — always pass **`make_Wij_active`** for n&lt;4.
- **NDR regime**: fig3 trimer uses **Qzz=0** (monopole), circle R≈5.77, diagonal xV cut — distinct from Ruslan dimer (Qzz=10).
- **V≈0 numerics**: PME Gauss–Jordan can yield NaN at zero bias / low T; `pauli_scan` maps non-finite \(I\) → 0.
- **Lever-arm gating**: `PauliSolverCL8.scan_current_tip(Vtips_gate=...)` decouples the site-energy gate voltage from tip `mu1=Vtips` — used for the Chao Li fitted 4-site model (ε_i = Δε_i + α(V_s−V0)·g_i, μ_tip = V_s); see `~/git/SPAMMM_export_for_ChaoLi/NOTES_fitted_PME4_model.md` + `scripts/run_pme4_fit.py`.

## Open Issues

### Temirov experimental fitting (2026-10-15)

The external export `SPAMMM_export_for_ChaoLi/scripts/fit_temirov_pme.py` fits
the registered 15.04 Å × 11.57 Å January Temirov maps using PME8 and one shared
energy in a 2×4 occupation supercell. Early fits hold Wij=0; later maps receive a
separate explicitly effective dipolar interaction screen. The geometry stays
fixed and each tip/site pair selects one nearest supercell image.

The wrapper now accepts optional `stationary_method='jacobi'` and
`stationary_precision='double'`. Damped diagonal preconditioning avoids slow
global Euler timesteps; double rate arithmetic retains small thermal hole
probabilities at 1 K. Both are opt-in; defaults remain Euler/single. Calling
`scan_current_tip(..., apply_validity_cut=False)` returns raw finite-temperature
currents consistently with or without diagnostic energies. The original
cutoff default is preserved. Independent dense 256-state comparisons and the
existing PME8/PME4 smoke check are documented in the external fit README.

Later interaction/image agreement is still limited; early-radius agreement does
not validate a unique tip shape or a full independently occupied infinite lattice.

- [ ] **Shared rigid pose SSOT** (`pos`+`qrot`) for PME ↔ PairFF ↔ Assembly ↔ FoldedRigid — audit: [`RigidBody.md`](RigidBody.md); design: [`Tasks/RigidMoleculePose_SSOT.md`](../Tasks/RigidMoleculePose_SSOT.md) (USER 2026-07-30; do not code until prioritized)
- [ ] Hubbard / MQCA OpenCL import (task slices B–C)
- [ ] MC fit + Wasserstein (E)
- [ ] Headless CLI imaging (`SPM_CLI_Headless.md`)
- [ ] Optional C++ CPU parity path
- [ ] Exp NPZ / QmeQ integration = Later

## Links

- Task: [`doc/Tasks/Import_ChargeRings_PME.md`](../Tasks/Import_ChargeRings_PME.md)
- Export SSOTs: `/home/prokop/git/ppafm/docs/export/charge_rings.export.md`, `/home/prokop/git/FireCore/doc/Topics/ManyBody/MQCA_Hubbard_Ising.export.md`
- Folder README: [`spammm/quantum/README.md`](../../spammm/quantum/README.md)

## Lattice PME8 (periodic) GUI mode (2026 added)

`spammm/quantum/pme_lattice.py` ports the export workhorse
`SPAMMM_export_for_ChaoLi/scripts/temirov_pme.py` (TemirovPME) into a
preset-driven `LatticePME`: registered rectangular cell + basis from
`geometry.json` (sha256-verified), n1×n2 occupation supercell (≤8 sites),
nearest-replica tip PBC via `tip_cell` (never sums replicas), distance-shell
Wij (long |a| / short |b| / diagonal |a+b|, meV in preset → eV), occupied-
environment site energies `E_i = E_ref − W.row_sum` (toggleable), and the full
10-coefficient PME8 tip multipole (`make_multipole_cs`; Qzz lives in cs6,
off-diagonals are single coefficients — `make_quadrupole_Coeffs` is NOT used).
Voltage = tip mu = gate. GUI section "Lattice PME8 (periodic)" in
`ChargeRingsExtension.py` gives live dI/dV/current maps vs the registered
experimental image (band-pass ridge scores) and long/short/diagonal x–V cuts
(branch-signal correlations). Presets written by
`export scripts/make_temirov_gui_presets.py` into
`results/temirov_gui_presets/`; parity vs saved candidate maps is exact
(`scripts/check_lattice_gui_parity.py`, rel diff 0 at 289 subsampled points).
Tests: `tests/quantum/test_pme_lattice.py`, `tests/GUI/test_charge_rings_lattice.py`.
