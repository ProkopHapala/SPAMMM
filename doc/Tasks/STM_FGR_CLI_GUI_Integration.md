# Task: Integrate FGR STM into SPM_CLI and SPAMM_GUI

**Status:** investigating
**Priority:** Medium — after FGR Level-B USER acceptance and BR-STM wiring
**Parent:** `doc/TopicalAudit/STM_FGR_Transfer.md`
**Related:** `doc/Tasks/STM_FGR_BondResolved.md`, `doc/Tasks/SPM_CLI_Headless.md`, `doc/Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md`

## Objective

Promote the FGR transfer STM (`M = c_t†(H−ES)c_s`, Level B) from the current compare-gallery prototype to a first-class STM mode in both the CLI (`run_spm.py`) and the GUI (`SPAMMM_GUI`), alongside the legacy overlap STM.

## Current state

| Surface | STM mode | Status |
|---------|----------|--------|
| `run_spm.py stm fgr` | FGR compare gallery | **active** (4-column panel: overlap vs I_S vs I_H vs I_τ) |
| `run_spm.py stm current` | overlap (legacy) | active |
| `run_spm.py stm br` | overlap BR-STM | active (legacy kernel) |
| `SPAMMM_GUI` STM panel | overlap (legacy) | active |
| FGR as a **standalone** STM mode (not compare) | — | **missing** |

The FGR path only exists as a **comparison gallery** (`run_fgr_transfer_compare`). There is no single-mode FGR STM output (just I_τ, no overlap/I_S/I_H columns), and no GUI integration.

## Implementation plan

### Step 1: Add `--stm-mode {overlap,fgr}` to CLI STM subcommands

Affects: `run_spm.py stm current`, `run_spm.py stm panel`, `run_spm.py stm br`.

- `stm current` → `run_frontier_stm_current`: add FGR path alongside `project_mo_stm_sk_slice`
- `stm panel` → `run_stm_vacuum_panel`: add FGR column
- `stm br` → see `STM_FGR_BondResolved.md`

Each subcommand gets `--stm-mode` (default `overlap` for backward compat) and FGR-specific args (`--eh-K`, `--tip-elem`, `--rcut`, `--degen-thresh`) via `add_fgr_args`.

### Step 2: Extract single-mode FGR STM helper

Currently `project_mo_stm_fgr_slice` is the per-MO, per-tip scan function. Wrap it in a higher-level helper that:
- Builds tables once (reuse `_stm_fgr_prepare_tables`)
- Loops over MOs × tips × heights
- Handles degeneracy clusters (sum I over degenerate manifold)
- Returns the same dict shape as `compute_stm_basis_variants` so downstream plotting is shared

Place in `spammm/SPM/stm_compare.py` or `AFM_utils.py` (wherever `compute_stm_basis_variants` lives).

### Step 3: GUI integration

`SPAMMM_GUI` STM panel currently calls the overlap path. Add:
- STM mode selector (dropdown: "Overlap (legacy)" / "FGR (H−ES)")
- FGR parameter controls (K, tip element, rcut, degen threshold) — collapsible advanced section
- Backend: call the same helper from Step 2

See `Consolidate_GUI_CLI_Backend_Input_Protocol.md` for the GUI↔CLI↔backend protocol — the STM mode must flow through the same backend interface.

### Step 4: Degeneracy handling everywhere

The degeneracy-aware MO summation (sum I over near-degenerate cluster, implemented in `run_fgr_transfer_compare`) must be applied in **all** STM paths, not just the FGR compare gallery. The PTCDA HOMO asymmetry (1.8 meV split → arbitrary DFTB rotation) affects overlap STM too.

- Extract `degen_cluster(eigvals, center, direction, thresh_eV)` to a shared utility
- Apply in `compute_stm`, `compute_bond_resolved_stm`, `run_frontier_stm_current`, etc.
- Default threshold: 5 meV (configurable)

### Step 5: Documentation

- Update `user_guide/SPM_CLI.md` with FGR mode docs
- Update `doc/topical_audit.md` §4b status
- Update `tests/SPM/README.md`

## Open questions

- Should FGR become the **default** STM mode (replacing overlap), or remain opt-in? → USER decision after L2 review of pentacene/PTCDA panels.
- Level A (frozen H⁰) is not implemented — FGR Level B (EH H∝S) is the current prototype. See `STM_FGR_Transfer.md` open issues.
- GUI performance: FGR table build (~0.07s) + scan (~1ms/pixel) is fast enough for interactive use, but the first run compiles OpenCL kernels (~2s cache miss).

## Code map

```text
FGR scan kernel     →  kernels/LCAO_STM_FGR.cl :: stm_fgr_sk_tau_scan_real
FGR host scan       →  spammm/SPM/AFM_utils.py :: project_mo_stm_fgr_slice
FGR tables          →  spammm/SPM/AFM_utils.py :: _stm_fgr_prepare_tables
FGR compare gallery →  spammm/SPM/stm_compare.py :: run_fgr_transfer_compare
CLI                 →  run_spm.py stm {current,panel,br,fgr}
GUI                 →  SPAMMM_GUI STM panel (see Consolidate_GUI_CLI_Backend_Input_Protocol.md)
Degeneracy util     →  extract from run_fgr_transfer_compare → shared helper
```
