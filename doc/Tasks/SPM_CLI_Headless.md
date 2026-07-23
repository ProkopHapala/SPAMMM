# Task: Headless SPM CLI (`run_spm.py`)

**Status:** investigating — CLI scaffolding + STM/AFM imaging wired; USER visual/science sign-off pending. **Do not mark Done** without confirmation.  
**Entry:** [`run_spm.py`](../../run_spm.py) · docs: [`user_guide/SPM_CLI.md`](../../user_guide/SPM_CLI.md) · index: [`user_guide/README.md`](../../user_guide/README.md)

---

## Goal

One **repo-root** CLI for all SPM imaging (and, gradually, pre-imaging geometry prep) so users never dig into `tests/SPM/` or open the GUI for batch work.

Physics SSOT stays in `spammm/SPM/` (`AFM.py`, `AFM_utils.py`, `stm_compare.py`, `KrigingGridFF.py`, `ModularPipeline.py`). The CLI is a thin argparse shell.

---

## Implemented (CLI surface)

| Command | Physics | Module / backend |
|---------|---------|------------------|
| `afm` | FDBM (stock / prolonged / cube) | `AFM_utils` + `testplot_fdbm_relax` helpers |
| `afm-morse` | Morse/LJ + point-charge Coulomb | `AFM.AFMulator` |
| `afm-kriging` | Mithun DFT → Kriging GridFF → PP | `testplot_kriging_relax` |
| `panel-fukui` / `replot-panel` | Fukui cube vs DFTB stock vs prolonged | `testplot_fdbm_relax` |
| `stm orbitals` | Frontier MO **ψ** (signed), DFTB + pySCF | `stm_compare` |
| `stm current` | MO-resolved STM **I≥0**, tip s/p_z/p_y | `stm_compare` |
| `stm panel` | HOMO/LUMO vacuum stock/prolonged/pySCF | `stm_compare` |

```bash
python run_spm.py --help
python run_spm.py stm orbitals --molecule pentacene --n-near 5
python run_spm.py afm --xyz data/xyz/benzene.xyz --projection both
```

---

## Gaps / open issues (ToDo)

### Imaging (near-term)

| Gap | Notes | Related |
|-----|-------|---------|
| Bond-resolved STM (BR-STM) | GUI has S6 via `ModularPipeline.stage6_br_stm`; **CLI missing** — must share backend | `AFMExtension`, `Consolidate_GUI_CLI_Backend_Input_Protocol.md` |
| GUI↔CLI shared job spec | Widget/argparse adapters → JSON dict → one runner | `Consolidate_GUI_CLI_Backend_Input_Protocol.md` |
| FDBM ↔ Kriging compare | Single CLI entry for method panels | `testplot_kriging_vs_fdbm_cube.py` |
| Pauli \(A,\beta\) fit modes | Contact / residual vs Kriging | `PauliFitting_TestDesign.md`, `Pauli_A_beta_KrigingTransferability.md` |
| Cube FDBM ES quality | Too strong / asymmetric (esp. PTCDA); Gaussian NA | `Fukui_FDBM_panel_notes_2026-07-23.md` |
| df↔Fz height labeling | amp=1.0 → ~1 Å morph shift; need Fz_u/Fz_r/df + dense Δz | skill:`afm-plotting`; same notes report |
| Chem-window defaults | df useful ~4.3–5.3 Å @ Δz=0.1 | panel currently h_step=0.4 |
| `afm-morse` plot SSOT | Still ad-hoc `imshow`; should use `AFM_utils` | skill:`afm-plotting` |
| STM molecule registry | Only pentacene/PTCDA/benzene/pyridine; Fukui dimers missing | `stm_compare.MOLECULES` |
| Full bias-window STM sum | Σ channels at fixed bias (not MO-resolved) | future |

### Pre-imaging geometry (medium-term)

| Gap | Notes | Related |
|-----|-------|---------|
| **`relax` / `dock` subcommand** | Molecule on substrate before AFM/STM | UFF/SPFF/LFF, DFTB, GridFF, FAF |
| GridFF substrate prep | Relax / scan on GridFF volume | `surfaces/GridFF.py` |
| FAF / folded atomic functions | FoldedRigid docking + relax | `FoldedRigid.py`, Cosserat rods task |
| DFTB surface SCF | Optional QM relax on slab | DFTBcore / ModularPipeline S1 |

### Future imaging modalities

| Gap | Notes | Related |
|-----|-------|---------|
| **Light-STM** | Optically driven / excited-state channels | Frenkel / TEPL ideas; no module yet |
| **Charge-ring imaging** | Ring current / magnetic contrast | `Import_ChargeRings_PME.md` |
| Contact-surface / 2.5D AFM | Fast AFM path | `Fast_2p5D_AFM_ContactSurface.md` |

### Packaging / UX

| Gap | Notes |
|-----|-------|
| pip entry point | `pyproject.toml` console script → `run_spm` (`PipInstall_Packaging.md`) |
| argparse `--cpu-fft` | `type=bool` is fragile; prefer `store_true` / `store_false` |
| Thin wrappers vs duplication | Prefer `stm_compare` / shared runners; avoid growing logic inside `run_spm.py` |
| **GUI↔CLI backend protocol** | Shared `SPMJobSpec` JSON + `run_spm_job`; BR-STM via ModularPipeline S6 (`Consolidate_GUI_CLI_Backend_Input_Protocol.md`) |

---

## Acceptance (when USER confirms)

- [ ] User can run Morse, FDBM, Kriging AFM and STM orbitals/current from repo root without GUI
- [ ] `user_guide/SPM_CLI.md` matches `--help` and lists open roadmap
- [ ] README points to CLI + `user_guide/`
- [ ] Science campaigns (cube ES, prolonged, STM panels) remain **investigating** until explicit USER OK
- [ ] Bond-resolved STM + substrate `relax` + light-STM / charge-rings documented as open (not silently claimed ready)
- [ ] BR-STM available from CLI using the **same** ModularPipeline path as GUI (see consolidate task)

---

## Related docs

- User guide — `user_guide/SPM_CLI.md`
- STM basis campaign — `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md`
- Prolonged / Fukui panel — `doc/Tasks/ProlongedRadialBasis_DFTB.md`
- Kriging import — `doc/Tasks/Import_KrigingGridFF.md`
- Charge rings — `doc/Tasks/Import_ChargeRings_PME.md`
- Features audit — `doc/ToDo/Features.audit.md` §2.4 / CLI
- Agent ToDo — `doc/ToDo/ToDo.agents.md`
- **GUI↔CLI shared backend / BR-STM** — `doc/Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md`
