# Task: Lightweight OpenCL / JIT driver for force-field optimization

**Status:** investigating  
**Priority:** Later  
**Human ToDo:** item 6  
**Parent:** `doc/Tasks/RepoConsolidation.md`

## Objective

A **small, focused driver** so FF parameter fits (Pauli A/β, Morse, GridFF channels, maybe STO tails) can run **tight GPU loops** without heavy Python orchestration — options named in human ToDo: pyOpenCL “weave”, Numba/JIT, or [Codon](https://github.com/exaloop/codon)-style compiled Python.

**Not** a new FF physics stack — a **fit driver** around existing kernels.

## Context in SPAMMM

| Existing | Path |
|----------|------|
| Basis SA (density tails) | `spammm/quantum/DFTB/basis_optimizer.py` — already GPU projection + SA in Python |
| Pauli fitting design | `doc/Tasks/PauliFitting_TestDesign.md` |
| OpenCLBase | `spammm/utils/OpenCLBase.py` — persistent buffers, kernel cache |
| FDBM / grid eval | `kernels/AFM.cl`, `Forces.cl`, GridFF |

Bottleneck pattern: outer Python loop (trial params) → upload → kernel → download loss. Driver should keep params/loss on device or minimize round-trips.

## Design choices (ask USER before locking)

1. **Stay pure PyOpenCL** — parameter sweep / simplex / SA written as host loop calling one `eval_loss` kernel; simplest, matches repo.
2. **Numba** — only if host-side bookkeeping is proven hot (usually it is not vs GPU).
3. **Codon / weave** — only if USER wants compiled host; higher maintenance; justify with bench.

**Default recommendation:** PyOpenCL loss kernel + thin Python optimizer (reuse `basis_optimizer` patterns). Skip Codon unless measured need.

## Deliverables

1. One module e.g. `spammm/utils/ocl_fit_driver.py` (or under `forcefields/`): `FitProblem` protocol (`pack_params`, `launch_loss`, `unpack`).
2. Demo: refit Pauli A/β or STO ζ on a tiny system; wall-time vs current Python SA.
3. L0: loss decreases; best params finite; NVIDIA device.
4. Doc note in module header: when to use vs plain pytest benches.

## Out of scope

- Autodiff through OpenCL.
- Replacing DFTB+ SCC.
- GUI until driver proves ≥2–5× on a real fit.

## Acceptance

- [ ] Driver used by at least one real fit path (Pauli or basis)
- [ ] Bench numbers on NVIDIA shown to USER
- [ ] USER picks PyOpenCL vs Codon/Numba before Done
- [ ] USER confirms before Done
