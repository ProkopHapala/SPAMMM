# Task: Dyson orbitals for STM via DFTB+

**Status:** investigating  
**Priority:** Later (method) / Soon for Level-1 plumbing  
**Human ToDo:** item 5  
**Parent:** `doc/Tasks/RepoConsolidation.md`  
**Design chat:** `doc/Dyson_orbitals_STM.chat.md`

## Objective

Expose **Dyson-orbital (or controlled approximations)** as the STM imaging object instead of raw neutral HOMO/LUMO when charge-state / correlation matters — computed with DFTB+ where possible, projected on GPU with **prolonged radial basis** (`ProlongedRadialBasis_DFTB.md`).

## What the chat concludes (SSOT summary)

DFTB+ does **not** ship correlated Dyson orbitals. Practical ladder:

| Level | Method | Cost | When |
|-------|--------|------|------|
| 1 | Frozen: HOMO/LUMO ≈ Dyson; energies from ΔE(N↔N±1) | 1–2 SCC | Closed-shell, weak correlation |
| 2 | Relaxed single-determinant Dyson: SVD of orbital overlap \(B = (C^{N\pm1})^\dagger S C^N\) | 2 SCC + small dense | Orbital relaxation |
| 3 | Active-space DFTB–CI in (N,N±1) | Research | Polyradicals, MQCA-like molecules |

Vacuum caveat: even good \(d_p\) fail if AO tails die too fast → **prolonged basis for projection** (waveplot / `Grid_dftb`).

## SPAMMM inventory

| Piece | Status |
|-------|--------|
| DFTB+ SCC, charged calc, eigenvectors | `spammm/quantum/DFTB/`, `DFTB_utils.py` |
| Orbital → grid | `Grid_dftb.py` |
| STM path in AFM extension | `AFMExtension` / SPM docs |
| Many-body charge rings (separate) | `Import_ChargeRings_PME.md` — PME occupations ≠ Dyson shape |
| Chat | `doc/Dyson_orbitals_STM.chat.md` |

## Work plan

1. **Level 1 plumbing** — API: `compute_stm_channel(mol, mode='homo'|'lumo'|'dyson_frozen', charge=…)` using existing eigenvector export + prolonged projection; vertical IP/EA via charged SCC.
2. **Level 2** — obtain \(S_{\rm AO}\) (DFTB+ library or matrix dump); implement overlap SVD → Dyson AO coeffs; write same format as eigenvector for projector.
3. **L0** — closed-shell molecule: ‖ψ_Dyson_L2 − ψ_HOMO‖ small; charged ΔE finite.
4. **L2** — pentacene-like HOMO map vs experiment optional; USER review.
5. **Level 3** — only after USER prioritizes; tie to MQCA / charge-rings molecules.

## GUI

- STM/Orbitals sub-pane: channel selector `HOMO | LUMO | Dyson(N−1) | Dyson(N+1)`.
- Do not block on full CI.

## Acceptance

- [ ] Level 1 callable from library + testplot
- [ ] Prolonged basis used for vacuum slice
- [ ] Level 2 either implemented or explicitly deferred with USER OK
- [ ] USER confirms before Done

## Dependencies

- Prefer completing prolonged-basis selection (`ProlongedRadialBasis_DFTB.md`) before claiming STM quantitative tails.
- Charge-rings PME is complementary (rate equations on sites), not a substitute for Dyson maps.
