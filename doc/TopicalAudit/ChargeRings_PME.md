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
| Python | `spammm/quantum/PauliSolverCL.py` | **active** | FireCore `pauli_ocl.py` on SPAMMM `OpenCLBase` |
| Python | `spammm/quantum/pauli_scan.py` | **active** | Slim xy/xV API; 2→4 embed; Wij; Ruslan + fig3 trimer params |
| Data | `data/charge_rings/` | **active** | `Ruslan_{long,short,kite}.txt`, `square_tetramer.txt`, `fig3_trimer.json` |
| Test L0 | `tests/quantum/test_pme_pauli.py` | **active** | Square mirror symmetry on RTX 3090 |
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

## Design notes

- **PME.cl tip μ**: substrate `mu0` from `lead_params`; tip `mu1` = per-pixel `Vtips` (not `lead_params[2]`).
- **Temp**: `params['Temp']` in Kelvin → `set_lead(..., Temp*kB)` in eV (ppafm `make_configured_solver` convention).
- **W vs Wij**: scalar `W` couples all 4 embed sites including spectators — always pass **`make_Wij_active`** for n&lt;4.
- **NDR regime**: fig3 trimer uses **Qzz=0** (monopole), circle R≈5.77, diagonal xV cut — distinct from Ruslan dimer (Qzz=10).
- **V≈0 numerics**: PME Gauss–Jordan can yield NaN at zero bias / low T; `pauli_scan` maps non-finite \(I\) → 0.

## Open Issues

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
