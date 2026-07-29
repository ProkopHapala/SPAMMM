---
type: TopicalAudit
title: STM FGR Transfer (H−ES)
tags: [STM, FGR, Bardeen, DFTB, OpenCL, LCAO, prolonged-basis]
timestamp: 2026-07-29
---

# STM FGR Transfer (\(H - ES\))

## Summary

Weak-coupling STM between known tip and sample MOs evaluates \(M = c_t^\dagger(H_{TS}-E S_{TS})c_s\) with **custom long-tail** Slater–Koster tables, not mio/3ob radial integrals. OpenCL production path is `stm_fgr_sk_tau_scan_real`. Legacy `mo_overlap_points_exp_sk` remains the overlap / artificial-exp baseline. Table Level B (exact prolonged \(S\) + EH \(H\)) is wired; Level A frozen \(H^0\) is unfinished. No SCC, Dyson, or bias integration in the scan kernel.

## Implementations

| Language | Location | Status | Notes |
|----------|----------|--------|-------|
| OpenCL | `kernels/LCAO_STM_FGR.cl` | **active** | τ build + real/complex scan + HS debug |
| OpenCL | `kernels/LCAO_grid.cl` `mo_overlap_points_exp_sk` | active (legacy baseline) | Artificial \(\exp(-\beta(r-r_0))\) SK; \(H'\sim 1\) |
| OpenCL | `kernels/LCAO_STM.cl` | experimental / later | Dyson / GF response — **not** FGR product path |
| Python | `spammm/quantum/DFTB/Grid_dftb.py` | **active** | Host buffers + `stm_fgr_sk_tau_scan_real` |
| Python | `spammm/quantum/DFTB/DFTBplusParser.py` | **active** | `build_longtail_eh_sk_tables`, STO 2c channels, SKF onsite |
| Python | `spammm/SPM/AFM_utils.py` | **active** | `project_mo_stm_fgr_slice`, `_stm_fgr_prepare_tables` |
| Python | `spammm/SPM/stm_compare.py` | **active** | `run_fgr_transfer_compare` |
| CLI | `run_spm.py stm fgr` | **active** | Gallery → `debug/stm_fgr_compare/` |
| Ideas | `doc/Ideas/LCAO_STM_FGR_WIRING.md` | active | Buffer/layout + Level A/B/C |
| Ideas | `doc/Ideas/STM_perturbation_H.chat.md` | reference | Derivation discussion |

## Parity Status

| Pair | Result | Artifacts |
|------|--------|-----------|
| FGR vs `overlap_exp` (pentacene, PTCDA, z=3 Å, tip \(s\)/\(p_z\)) | **agent-run**; morphology change documented | `debug/stm_fgr_compare/` — **awaiting USER L2** |
| Level B \(I_S\) vs \(I_\tau\) (C/H only) | nearly identical shape (EH \(H\propto S\)) | expected; not a bug |
| Level B vs Level A \(H^0\) | **not implemented** | — |
| Bardeen-plane FFT vs SK τ | **not implemented** | — |
| L0 pytest | **missing** | — |

## Design notes

- **SSOT matrix element:** \(\tau = H - E S\) prebuilt once per energy; scan reads only τ.
- **Orbital packing:** `[px,py,pz,s]` in FGR + overlap kernels; DFTB export remapped on host.
- **Directed SK:** \(u = (R_S-R_T)/R\); store signed `sp` and `ps` (do not hard-code a second minus).
- **Basis dirty trick:** short-basis DFTB \(c\) + prolonged \(\tilde\chi\) tables — nodes from \(c\), tails from tables.
- **NVIDIA first:** OpenCL Shell must see NVIDIA ICD (`preferred_vendor='nvidia'`); never report PoCL timings as GPU.

## Open Issues

- Level A numerical \(H^0\) (kinetic + frozen \(v_A^0+v_B^0\)).
- USER confirmation of pentacene/PTCDA panels before product promotion.
- PTCDA HOMO localization / broken-symmetry DFTB caveat (shared with other STM reports).
- Optional Bardeen surface reference; WKB barrier factor later.
- Keep Dyson/`LCAO_STM.cl` off the first-order FGR validation path until FGR is accepted.

## Related docs

- Report: [`doc/Reports/STM_FGR_Transfer_H_ES_2026-07-29.md`](../Reports/STM_FGR_Transfer_H_ES_2026-07-29.md)
- Orbital / prolonged compare: [`doc/Reports/STM_ExtendedBasis_OrbitalCompare.md`](../Reports/STM_ExtendedBasis_OrbitalCompare.md)
- Dyson task (later): [`doc/Tasks/DysonOrbitals_DFTB_STM.md`](../Tasks/DysonOrbitals_DFTB_STM.md)
- Index: [`doc/topical_audit.md`](../topical_audit.md) §4b
