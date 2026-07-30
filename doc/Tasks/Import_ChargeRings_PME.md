# Task: Import charge rings — Pauli Master Equation + MC / Hubbard / MQCA

**Status:** Done (slices A+D+F) — Hubbard/MQCA/MC-fit remain Later; pose SSOT follow-on in [`RigidMoleculePose_SSOT.md`](RigidMoleculePose_SSOT.md)  
**Priority:** P1 (nc-AFM — `doc/ARCHITECTURE_ROADMAP.md` §TOC)  

**Human ToDo:** item 3  
**Parent:** `doc/Tasks/RepoConsolidation.md`  
**Topical audit:** `doc/TopicalAudit/ChargeRings_PME.md`  
**Pose glue (sites = molecules):** `doc/TopicalAudit/RigidBody.md` · `doc/Tasks/RigidMoleculePose_SSOT.md`

## Objective

Consolidate **many-body charge-state STM** (Pauli Master Equation, tip multipoles, xy / xV scans, Monte Carlo fitting) and the FireCore **Hubbard / MQCA** OpenCL solvers into one SPAMMM stack + one GUI plugin — replacing ppafm’s multiple ChargeRings GUIs and FireCore’s parallel OCL demos.

## Export SSOTs (read first)

| Doc | Covers |
|-----|--------|
| `/home/prokop/git/ppafm/docs/export/charge_rings.export.md` | C++ `PauliSolver`, ctypes, `PME.cl`, `pauli_scan.py`, GUIs, MC fit vs experiment |
| `/home/prokop/git/FireCore/doc/Topics/ManyBody/MQCA_Hubbard_Ising.export.md` | `MQCA.cl`, `hubbard.cl` MC+dense PME, parity vs `PME.cl` |

**Note:** Human ToDo previously linked `interpolation.export.md` for item 3 — that is Kriging (item 2). Charge rings = `charge_rings.export.md` + MQCA/Hubbard export.

## Physics (one picture)

```
sites (geom, ε, W_ij) + tip multipole/tunneling
        ↓
many-body occupations (2^N or reduced basis)
        ↓
PME steady state P  →  I, dI/dV maps (xy, xV)
```

| Scale | Ground state | Non-eq current |
|-------|--------------|----------------|
| ≤4 sites full PME | — | `PME.cl` (ppafm + FireCore identical) |
| ≤16 sites | MQCA Gray code | — |
| ≤64 sites | Hubbard MC | Hubbard dense PME (≤64 basis states) |
| Fitting | — | MC optimizer + Wasserstein (ppafm) |

## Preferred SPAMMM architecture

1. **OpenCL-first** — port FireCore wrappers onto SPAMMM `OpenCLBase` (`preferred_vendor='nvidia'`):
   - `kernels/PME.cl`, `kernels/hubbard.cl`, `kernels/MQCA.cl`, `kernels/MQCA_top8.cl`
   - `spammm/quantum/` or `spammm/SPM/` solvers: `PauliSolverCL`, `HubbardSolver`, `MQCASolver`
2. **Optional C++** — ppafm `pauli.hpp` only if needed for CPU parity / OpenMP scan; do not make ctypes the primary path.
3. **Scan engine** — slim port of `pauli_scan.py` (xy / xV) as library functions; CLI = thin `run_*.py`.
4. **Fitting** — port `MonteCarloOptimizer` + Wasserstein as optional module; keep separate from solver kernels.
5. **GUI** — one `ChargeRingsExtension` (or mol-browser plugin), not three ppafm GUIs.

## Existing SPAMMM hooks

- `spammm/utils/OpenCLBase.py` — already mentions Hubbard-style solvers in comments.
- STM/AFM GUI: `AFMExtension` — charge-rings should be a **sibling** extension, not stuffed into AFM FDBM panes.
- NTCDA/PTCDA geometries: copy minimal `Ruslan_*.txt` fixtures into `data/` or `tests/ref_data/` (not 29 MB exp NPZ unless USER asks).

## Work plan (parallelizable slices)

| Slice | Owner focus | Exit criterion | Status |
|-------|-------------|----------------|--------|
| A | `PME.cl` + `PauliSolverCL` + 4-site x-scan L0 | parity vs FireCore `test_pme_parity_*` | **Done** (USER 2026-07-30) |
| B | `hubbard.cl` MC + dense PME | max\|dI\| ~ 1e-12 vs A on 2/4-site | Later |
| C | MQCA + top8 | ground-state / logic-map smoke | Later |
| D | `pauli_scan` xy/xV API + testplot | NTCDA dimer + fig3 trimer NDR | **Done** (USER 2026-07-30) |
| E | MC fit skeleton | optional; needs exp data policy | Later |
| F | GUI extension | after A+D | **Done** (ChargeRingsExtension) |

## Acceptance

- [x] Solvers live under SPAMMM tree; run on NVIDIA — `kernels/PME.cl` + `PauliSolverCL` + `pauli_scan`
- [x] Documented parity numbers in test stdout / `.out` — square mirror; Ruslan/fig3 plots
- [x] One NTCDA (or Ruslan) xy or xV plot for USER review — + fig3 trimer NDR
- [x] No silent PoCL fallback in benches — NVIDIA assert in tests
- [x] USER confirms before Done — confirmed 2026-07-30 (docs + mark done)

**Delivered:**
- `spammm/quantum/PauliSolverCL.py`, `pauli_scan.py`, `kernels/PME.cl`
- Fixtures `data/charge_rings/` (Ruslan_*, square_tetramer, fig3_trimer.json)
- L0 `tests/quantum/test_pme_pauli.py`; L2 `testplot_charge_rings_{ruslan,trimer}.py`
- Audit `doc/TopicalAudit/ChargeRings_PME.md`

## Caveats from exports

- Temperature units GUI→solver bug in ppafm (`temperature.md`) — SPAMMM uses Kelvin→eV via `make_configured_solver` convention.
- Hubbard PME: known historical bugs (pivoting, dE sign) already fixed in FireCore — port fixed kernels only (slice B).
- Full exp dataset / QmeQ integration = Later.
- **Two regimes:** Ruslan dimer (Qzz=10) vs fig3/symmetric trimer (Qzz=0 monopole → strong NDR).

## Follow-on (not in A–F)

- [ ] **Rigid sites = molecules** — ChargeRings `spos`/`rots` from shared pose SSOT (`pos`+`qrot`), same bodies as PairFF/Assembly/FoldedRigid. Inventory: [`doc/TopicalAudit/RigidBody.md`](../TopicalAudit/RigidBody.md). Design: [`RigidMoleculePose_SSOT.md`](RigidMoleculePose_SSOT.md). Do not implement until USER prioritizes.
