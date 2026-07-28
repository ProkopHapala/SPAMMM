# Task: Consolidate GUI ↔ CLI SPM backend + input protocol

**Status:** investigating — **FAST_S3 ↔ CLI-legacy Stage3–4 parity USER-confirmed (2026-07-28)**. Next: remove legacy CLI pipeline; finish GUI param SSOT. **Do not mark Done** until CLI cutover + GUI param parity verified by USER.

**Related:** [`SPM_CLI_Headless.md`](SPM_CLI_Headless.md) · [`user_guide/SPM_CLI.md`](../../user_guide/SPM_CLI.md) · topical [`../TopicalAudit/AFM_FDBM.md`](../TopicalAudit/AFM_FDBM.md) · GUI `spammm/GUI/AFMExtension.py` · `spammm/SPM/ModularPipeline.py`

---

## Decision (2026-07-28)

| Question | Answer |
|----------|--------|
| Can Modular FAST_S3 reproduce CLI `_run_from_density` Stage3–4? | **Yes** — corr ≥ 0.9996 on df; fields ~1.0 (pentacene, PTCDA) |
| Is FAST faster? | **Yes** — **~5.5×** on warm S3+S4 (RTX 3090) |
| Put CLI path into GUI? | **No** — CLI Stage-3 is the slow NumPy fork |
| Put ModularPipeline into CLI? | **Yes — required next** |

Parity script: `tests/SPM/testplot_cli_vs_modular_parity.py`  
Artifacts: `debug/cli_vs_modular_parity/{pentacene,PTCDA}/` — USER: “definitely good”.

### Remove legacy (todo — not Done)

1. Rewrite `run_spm.py afm` to call ModularPipeline / shared runner (FAST_S3, GPU FFT default — drop `SPAMMM_AFM_CPU_FFT=1` as default).
2. Point `panel-fukui` / `run_afm_cli_fdbm_gallery.py` at the same runner (keep multi-row compare as thin plotting).
3. Mark `testplot_fdbm_relax._run_from_density` **deprecated** (parity-only or delete after cutover).
4. Keep `SPAMMM_AFM_FAST_S3=0` / `SPAMMM_AFM_CPU_FFT=1` only as explicit debug switches.

---

## Why GUI did not reproduce CLI (parameter audit)

Physics code path for GUI FDBM is already ModularPipeline (correct/fast). Mismatch was mostly **wrong UI defaults / PP knobs**, not a second algorithm.

| Parameter | CLI SSOT | GUI (wrong / was) | Effect |
|-----------|----------|-------------------|--------|
| **Pauli A, β (3ob)** | **124.84 / 1.4330** (`PAULI_FITTED_DEFAULTS`) | Was **509.28 / 1.0586** | **Fixed** — spins load from `PAULI_FITTED_DEFAULTS` |
| **Pauli A, β (mio)** | 155.33 / 1.5507 | Was 787.22 / 1.2371 | **Fixed** (same) |
| df / heights | 3.7–4.7, amp=1, `compute_df_amp` | Was 2.8–3.6 + `compute_df` | **Fixed** |
| PP integrator | FIRE defaults | Soft `relax_pars` in `compose_and_relax_total` | **Fixed** — FIRE SSOT |
| Lateral scan pad | `--scan-margin` **2.0** | Was scan_range=3.0 | **Fixed** → 2.0 |
| Projection / step | prolonged / 0.1 | prolonged / 0.1 | OK |
| Lateral sampling | `arange` @ grid step | ModularPipeline `linspace` @ scan_step | Mild residual |

**Smoking gun for “CLI looks better / GUI strange”:** GUI Pauli spins still defaulted to commented-out old values in `AFM.PAULI_FITTED_DEFAULTS`, while CLI evaluation always used 124.84 / 1.4330.

---

## Discrepancy inventory (updated)

| Issue | Status |
|-------|--------|
| Silent Morse→FDBM | Fixed (fail loud + shared Morse runner) |
| GUI df / heights / amp | Fixed (CLI SSOT) |
| CLI `afm` projection/step defaults | Aligned prolonged / 0.1 |
| **FAST_S3 ↔ CLI Stage3–4 parity** | **USER confirmed** |
| **CLI `afm` still `_run_from_density`** | **Must remove** (see Decision) |
| GUI Pauli spins vs SSOT | **Fixed** → `PAULI_FITTED_DEFAULTS` |
| GUI soft `relax_pars` vs CLI FIRE | **Fixed** → FIRE defaults in `compose_and_relax_total` |
| GUI `scan_range` vs CLI `scan_margin` | **Fixed** → default 2.0 |
| `SPMJobSpec` / `run_spm_job` | Remaining protocol work |

**Shared backends**

| Mode | Shared | CLI | GUI |
|------|--------|-----|-----|
| FDBM product | ModularPipeline FAST_S3 | `stm br` ✓; `afm` **legacy until cutover** | AFM / BR-STM |
| Morse | `run_morse_coulomb_afm` | `afm-morse` | AFM when Morse selected |

---

## Design goals (do not forget)

### 1. Shared compute backend (maximum reuse)

```
GUI widgets ──┐
              ├──► SPMJobSpec ──► ModularAFMPipeline / runners ──► result arrays
CLI argparse ─┘
```

No second FDBM Stage-3 in `tests/`. Adapters only.

### 2. BR-STM first-class — CLI `stm br` + GUI BR-STM button share ModularPipeline S4–S6.

### 3. Plot backends stay separate — Agg stills vs VisPy/Qt; same arrays.

---

## Work plan (remaining)

1. **Cut over CLI `afm`** to ModularPipeline (remove default CPU FFT).
2. **GUI param SSOT** — Pauli spins, FIRE relax_pars, scan margin.
3. Extract `run_spm_job(cfg)` when adapters stabilize.
4. Do not delete `_run_from_density` until USER approves after gallery re-run on FAST path.
