# Task: Consolidate GUI ↔ CLI SPM backend + input protocol

**Status:** investigating — partial unification in progress (2026-07-28). **Do not mark Done** without USER confirmation.

**Related:** [`SPM_CLI_Headless.md`](SPM_CLI_Headless.md) · [`user_guide/SPM_CLI.md`](../../user_guide/SPM_CLI.md) · GUI `spammm/GUI/AFMExtension.py` · backend `spammm/SPM/ModularPipeline.py`

---

## Discrepancy inventory (2026-07-28) — why GUI ≠ CLI

| Issue | Was | Fix / remaining |
|-------|-----|-----------------|
| **Silent Morse→FDBM fallback** | GUI combo "Morse" silently ran DFTB FDBM | **Fixed:** fail loud; Morse uses shared `AFM_utils.run_morse_coulomb_afm` |
| **df without amp** | GUI `stage4` → `compute_df` on raw hmin–hmax | **Fixed:** GUI uses `afm_df_height_stacks` + `compute_df_amp` (same as `stm br`) |
| **Height defaults** | GUI 2.8–3.6; CLI/BR 3.7–4.7 | **Fixed:** GUI defaults → 3.7–4.7, amp=1.0, Z=4.2 |
| **CLI `afm` projection/step** | default stock / 0.15 | **Aligned:** prolonged / 0.1 |
| **CLI `afm` vs ModularPipeline** | `testplot_fdbm_relax._run_from_density` fork | **Remaining:** fold CLI `afm` into ModularPipeline or shared runner |
| **CLI BR-STM** | was missing | **Exists:** `run_spm.py stm br` → `run_br_stm_afm_panel` (ModularPipeline) |
| **SPMJobSpec** | no single job dict | **Remaining:** extract `run_spm_job(cfg)` |

**Shared backends today**

| Mode | Shared function | CLI | GUI |
|------|-----------------|-----|-----|
| FDBM AFM + BR | `ModularAFMPipeline` (+ amp heights in GUI / `run_br_stm_afm_panel`) | `stm br` ✓; `afm` still forked | AFM / BR-STM buttons |
| Morse+Coulomb | `AFM_utils.run_morse_coulomb_afm` | `afm-morse` | AFM button when Morse selected |
| Flat STM diagnostic | `stm_compare` | `stm orbitals/…` | (not product path) |

**No silent fallbacks:** Morse never becomes FDBM; STM/BR-STM with Morse raise `RuntimeError`.

---

## Design goals (do not forget)

### 1. Shared compute backend (maximum reuse)

| Layer | Shared? | Role |
|-------|---------|------|
| Physics / stages S1–S6 | **Yes** | `ModularAFMPipeline`, `AFM_utils.compute_*`, `stm_compare` where appropriate |
| Input gathering | **No** | PyQt widgets ↔ CLI argparse (only adapters differ) |
| Input **payload** | **Yes** | One JSON-serializable **dict / job spec** → shared `run_from_config(cfg)` |
| Output plotting | **Mostly no** | Still images (matplotlib Agg) vs interactive GUI (VisPy / blit); both consume the **same arrays** |

CLI and GUI must not reimplement SCF → density → FDBM → PP relax → STM / BR-STM. Adapters only:

```
GUI widgets ──┐
              ├──► SPMJobSpec (dict/JSON) ──► ModularAFMPipeline / runners ──► result arrays
CLI argparse ─┘                                                              │
                                                                             ├──► matplotlib stills (CLI / reports)
                                                                             └──► VisPy / Qt blit (GUI interactive)
```

### 2. Bond-resolved STM (BR-STM) is first-class

Not “STM or AFM separately only” — the product imaging mode is often **BR-STM**: sample the STM signal at **AFM-relaxed tip positions** (CO bending → bond sharpening).

| Mode | Meaning |
|------|---------|
| AFM | Fz / df from FDBM (or Morse / Kriging) |
| STM (vacuum / constant-height) | ψ / LDOS / tip-coupled current on a flat height lattice |
| **BR-STM** | STM evaluated at PP-relaxed \((x+\mathrm{d}x,\,y+\mathrm{d}y)\) from Stage 4 |

### 3. Plot backends stay separate on purpose

- **CLI / L2 reports:** matplotlib `Agg` still PNGs (`AFM_utils` plot SSOT).
- **GUI molecule canvas:** VisPy.
- **GUI AFM/STM slice dialogs today:** matplotlib Qt5Agg (see below — not yet VisPy blit).
- Future interactive AFM overlays may use VisPy/`mpl_blit`; still must not fork physics.

---

## Current state (code review 2026-07-23)

### What already works (shared physics)

| Piece | Location | Notes |
|-------|----------|-------|
| S1–S4 AFM FDBM | `ModularAFMPipeline` | GUI uses this |
| S5 STM | `stage5_stm` → `AFM_utils.compute_stm` | Constant-height lattice |
| **S6 BR-STM** | `stage6_br_stm` → `AFM_utils.compute_bond_resolved_stm` | Needs Stage 4 `tip_disp` |
| Dirty flags S1→S6 | `AFMDirtyFlags` in `AFMExtension` | GUI only |
| Frontier STM compare | `stm_compare.py` | CLI `stm orbitals/current/panel` — **parallel path**, not S5/S6 |

### GUI: BR-STM is implemented

`AFMExtension.py`:

- Checkbox / component **“BR-STM Signal”** + `bond_resolved` in `_get_stm_params_from_ui`.
- Full run: Stage 4 → Stage 5 → optional Stage 6 when `sp['bond_resolved']`.
- Auto path for plot component `"BR-STM Signal"` calls `pipe.stage6_br_stm(..., tip_disp=...)`.

**Unknown without USER re-test:** whether S6 still works after FDBM FAST_S3 / tip-roll / Pauli changes. Treat as **verify**, not rewrite, when wiring CLI.

### CLI: BR-STM is missing

`run_spm.py` does **not** import `ModularAFMPipeline`. 

- `afm` / `panel-fukui` → `testplot_fdbm_relax` helpers (FDBM only; no tip_disp → BR-STM).
- `stm *` → `stm_compare` (vacuum MO / tip-orbital current panels).

So CLI can do AFM **or** STM diagnostics, but **not** the GUI’s BR-STM product path.

### How far from a common input protocol?

| Piece | Today | Gap |
|-------|-------|-----|
| Param snapshot dict | `_get_pipeline_params(window)`, `_get_stm_params_from_ui(window)` | Widget-coupled; not a schema; not loaded from JSON |
| Pipeline ctor | GUI builds `ModularAFMPipeline(...)` from widget floats | CLI never builds it |
| Job runner | Logic inlined in `AFMExtension` (`_ensure_stages_for_component`, full run) | No `run_spm_job(cfg) -> results` |
| JSON save/load | None for AFM/STM job specs | Need `cfg.to_json()` / `from_json()` |
| CLI argparse | Separate flags per subcommand | Need `args → SPMJobSpec` mapper |
| GUI widgets | Read at call time | Need `widgets → SPMJobSpec` mapper (thin) |

**Verdict:** ~40% there on the **compute** side (ModularPipeline + `compute_bond_resolved_stm` exist). ~10% on the **protocol** side (ad-hoc GUI dicts only). CLI is on a **forked** path for AFM/STM and must be folded back for BR-STM.

### Plotting reality check

| Surface | Backend today |
|---------|----------------|
| CLI stills | matplotlib Agg |
| GUI AFM/STM slice window | **matplotlib Qt5Agg** dialog (`plot_afm_slice`, `plot_orbital_map`) |
| GUI 3D molecule | VisPy |
| Fast interactive overlays | `mpl_blit` / VisPy elsewhere — **not** yet the AFM slice path |

So “matplotlib vs VisPy” is the **goal**; currently GUI AFM imaging is still matplotlib-in-Qt. Consolidation should: (1) shared arrays from backend; (2) keep still-plot helpers in `AFM_utils`; (3) optionally add VisPy blit later without touching S1–S6.

---

## Proposed `SPMJobSpec` (sketch)

JSON-serializable dict (names illustrative):

```json
{
  "version": 1,
  "geometry": {"xyz": "data/xyz/pentacene.xyz"},
  "qm": {"backend": "dftb", "basis": "3ob-3-1", "projection": "stock"},
  "grid": {"step": 0.1, "margin": 4.0, "z_extra": 6.0},
  "afm": {
    "enabled": true,
    "pauli_A": 124.84, "pauli_beta": 1.433,
    "C6_CO": 30.0,
    "K_LAT_Nm": 0.5, "bond_length": 3.0, "amp": 1.0,
    "h_min": 2.5, "h_max": 5.5, "h_step": 0.1
  },
  "stm": {
    "enabled": true,
    "mode": "br_stm",
    "mo_indices": [0],
    "mo_relative_to_homo": true,
    "field": "ldos",
    "use_exp_basis": true,
    "exp_beta": 1.0, "exp_r0": 3.0
  },
  "output": {"outdir": "debug/spm_job", "save_npz": true}
}
```

`mode`: `"afm_only"` | `"stm"` | `"br_stm"` | `"afm+stm"` | …  
BR-STM implies AFM Stage 4 must run first.

---

## Work plan

1. **Extract** `spammm/SPM/spm_job.py` (or similar): `SPMJobSpec` validate/load/save; `run_spm_job(cfg) -> dict` calling `ModularAFMPipeline` S1–S6 as needed.
2. **Thin GUI:** `_get_*_from_ui` → fill `SPMJobSpec`; call `run_spm_job`; keep dirty-flag cache as GUI optimization (or push dirty logic into pipeline later).
3. **CLI:** `run_spm.py afm-pipeline` / `stm br` → build same spec; `--config job.json`.
4. **BR-STM CLI:** require tip_disp from S4; write `br_stm_*.png` via `AFM_utils` still plots.
5. **Verify GUI BR-STM** on benzene/pentacene after FAST_S3 (L2 screenshots).
6. **Do not** merge `stm_compare` into ModularPipeline blindly — keep as diagnostic tool; optionally call from CLI under `stm orbitals` as today.
7. Plot adapters: still = matplotlib; interactive = VisPy/blit later; never fork physics for plots.

---

## Acceptance

- [ ] One `run_spm_job(cfg)` used by GUI AFM extension and CLI for FDBM + STM + **BR-STM**
- [ ] Job spec load/save as JSON
- [ ] `python run_spm.py …` can produce BR-STM maps for a molecule without GUI
- [ ] GUI BR-STM re-verified (USER) after consolidation
- [ ] Docs: `user_guide/SPM_CLI.md` + this task; ToDo.agents lists the work
- [ ] Plot backends documented: shared arrays; matplotlib stills vs VisPy interactive

---

## Non-goals (this task)

- Rewriting VisPy molecule editor
- Light-STM / charge-rings (separate)
- Replacing `stm_compare` frontier diagnostics
- Marking cube-ES or prolonged campaigns Done
