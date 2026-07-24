# Task: Headless SPM CLI (`run_spm.py`)

**Status:** investigating — CLI scaffolding + STM/AFM imaging wired; USER visual/science sign-off pending. **Do not mark Done** without confirmation.  
**Entry:** [`run_spm.py`](../../run_spm.py) · docs: [`user_guide/SPM_CLI.md`](../../user_guide/SPM_CLI.md) · index: [`user_guide/README.md`](../../user_guide/README.md)

---

## Goal

One **repo-root** CLI for the full SPM workflow — not only AFM/STM imaging, but also **molecule construction** and **geometry prep** — so users never dig into `tests/SPM/` or open the GUI for batch work.

Physics SSOT stays in `spammm/SPM/` (`AFM.py`, `AFM_utils.py`, `stm_compare.py`, `KrigingGridFF.py`, `ModularPipeline.py`). Topology / FF / DFTB SSOT stays in `spammm/topology/`, `spammm/forcefields/`, `spammm/quantum/`. The CLI is a thin argparse shell.

**Target pipeline (conceptual):**

```
input (.xyz/.mol/.mol2 | ASCII | SMILES)
        ↓
   AtomicGraph / AtomicSystem   ← SSOT topology
        ↓
   [opt] gas-phase relax (UFF / SPFF / LFF / DFTB)
        ↓
   [opt, FUTURE] dock / relax on substrate (GridFF / FAF)
        ↓
   AFM / STM / BR-STM imaging
```

---

## Implemented (CLI surface)

| Command | Physics | Module / backend |
|---------|---------|------------------|
| `afm` | FDBM (stock / prolonged / cube); `--smiles` / `--smiles-example`; PCA long→x; amp-aligned df/Fz; `--plots` | `AFM_utils` + `testplot_fdbm_relax` helpers |
| `opt` | Gas-phase UFF/SPFF/LFF/DFTB; planar flatten + `orientPCA` | `FFController.optimize_vacuum` |
| `smiles-afm` | SMILES → planar opt → prolonged FDBM + atom dots | `topology/smiles` + `opt` + `afm` |
| `afm-morse` | Morse/LJ + point-charge Coulomb | `AFM.AFMulator` |
| `afm-kriging` | Mithun DFT → Kriging GridFF → PP | `testplot_kriging_relax` |
| `panel-fukui` / `replot-panel` | Fukui cube vs DFTB stock vs prolonged | `testplot_fdbm_relax` |
| `stm orbitals` | Frontier MO **ψ** (signed), DFTB + pySCF | `stm_compare` |
| `stm current` | MO-resolved STM **I≥0**, tip s/p_z/p_y | `stm_compare` |
| `stm panel` | HOMO/LUMO vacuum stock/prolonged/pySCF | `stm_compare` |

```bash
python run_spm.py --help
python run_spm.py smiles-afm --example naphthalene --method uff
python run_spm.py stm orbitals --molecule pentacene --n-near 5
python run_spm.py afm --xyz data/xyz/benzene.xyz --projection both
```

**AFM height defaults (2026-07-24):** df window `--h-min/--h-max` = **3.7–4.7** Å, `--h-step/--dz` = **0.1**; amp-align shows Fz at **h−amp** (default amp=1 → Fz 2.7–3.7). Plots default `--plots compare,stage`. Geometry: force planar (`z` identical) + PCA long axis ‖ x.

**Not yet in CLI:** ASCII / `.mol`/`.mol2` shared resolver flags, substrate `dock`, BR-STM.

---

## Inventory — reuse before writing (do not reimplement)

### Molecule I/O (files)

| Capability | Where | Notes |
|------------|-------|-------|
| `.xyz` → arrays | `spammm.atomicUtils.load_xyz` | Used ad-hoc in `run_spm.py afm` today |
| `.mol` / `.mol2` → arrays + bonds | `atomicUtils.loadMol` / `loadMol2` | |
| File → **AtomicGraph** (SSOT) | `MoleculeEditorBackend.load_structure` / `load_xyz` / `load_mol` | Also snaps to HexGrid; CLI should prefer a **headless** loader that does not require Qt/HexGrid UX |
| Save `.xyz`/`.mol`/`.mol2` | `MoleculeEditorBackend.save_structure`, `AtomicSystem.save_*` | Bond orders from graph via `_graph_bond_types_mol` |
| STM registry | `stm_compare.resolve_molecules` | Names + optional `--xyz` override |

### ASCII-art builder (already works; GUI is thin)

| Capability | Where | Notes |
|------------|-------|-------|
| Parse ASCII → `AtomicSystem` | `spammm.topology.ascii_art_heterocycle.parse_ascii_art` | Single-atom + dimer formats; `:` H-bonds |
| Add H / Jacobi bond relax / H-bond resolve | same module + `KekulePure.make_n_pi` | Used by CLI `main()` in that file **and** GUI |
| GUI wrapper | `spammm.GUI.AsciiArtExtension.generate_ascii_molecule` | **Duplicates** the post-parse pipeline (H, relax, hbonds) — extract shared `build_molecule_from_ascii(...)` and call from both |
| Examples | `ASCII_EXAMPLES` dict | naphthalene, PTCDA-like, etc. |
| Tests | `tests/topology/test_ascii_art.py` | |

### Gas-phase geometry optimization (already works in GUI)

| Backend | Shared module (logic) | GUI (thin) | Status |
|---------|----------------------|------------|--------|
| **SPFF** | `FFController.build_ff(..., 'spff')` + `relax_n` / `relax_until_converged` | `FFExtension` | **Ready** for headless CLI |
| **LFF** | `FFController` `ff_type='lff'` → `LFFSolver` | same panel combo | Ready; projective Jacobi |
| **UFF** | `UFF_cl` exists; `FFController` raises `NotImplementedError` for UFF relax path | combo lists UFF | **Wire UFF into FFController** (or document SPFF/LFF-only until T02 finishes) — do **not** fork a second optimizer in CLI |
| **DFTB+** | `DFTB_utils.run_dftb_relax(work_dir, enames, apos, ...)` → `(E_ha, apos)` | `DFTBExtension` → `MoleculeEditorBackend.run_relaxation` (PBC-aware `run_pbc`) | Prefer **`run_dftb_relax`** for gas-phase CLI; keep `run_relaxation` for editor PBC cases |

Also: vibrations already choose UFF/SPFF/DFTB (`VibrationExtension` / `dynamics/`) — same backends, different product.

### Substrate dock / relax (harder — FUTURE)

| Path | Module | Notes |
|------|--------|-------|
| Rigid on **GridFF** volume | `RigidBodyDynamics.run_gridff`, `from_xyz_and_grid` | Precomputed B-spline PLQ |
| Rigid on **FAF** (folded) | `FoldedRigid.relax_folded`, `RigidBodyDynamics` folded kernels | See `RigidBodyDynamicsWithFoldedBasisSubstrate.md`, `Report_PTCDA_NaCl_FAF_RigidImaging.md` |
| Flexible molecule + FAF | `SPFF_cl` / `UFF_cl` with `do_faf`, `FFController` upload folded fit | Atomistic MD on surface — heavier |
| Flexible + GridFF scan | `GridFFRelaxedScan` | Untested / partial |
| GUI folded dock | `FoldedRigidExtension` | Already wraps `FoldedRigid.*` |

**Do not** reimplement these for CLI yet — only document + eventually thin-wrap the same APIs under `run_spm.py dock`.

### SMILES (parser wired; GUI text box open)

| Item | Status |
|------|--------|
| Parser / builder | **`spammm/topology/smiles.py`** — pure OpenSMILES organic subset + soft RDKit; `parse_smiles` → `AtomicGraph`; `smiles_to_system`; `SMILES_EXAMPLES` |
| Tests | `tests/topology/test_smiles.py` (22 passed) |
| CLI | `--smiles` / `--smiles-example` on `afm`/`opt`; `smiles-afm` pipeline |
| Remaining | Shared `.mol`/`.mol2`/ASCII resolver; GUI SMILES text box (`ToDO_GUI.md`) |

---

## Gaps / open issues (ToDo)

### Imaging (near-term)

| Gap | Notes | Related |
|-----|-------|---------|
| Bond-resolved STM (BR-STM) | GUI has S6 via `ModularPipeline.stage6_br_stm`; **CLI missing** — must share backend | `AFMExtension`, `Consolidate_GUI_CLI_Backend_Input_Protocol.md` |
| GUI↔CLI shared job spec | Widget/argparse adapters → JSON dict → one runner | `Consolidate_GUI_CLI_Backend_Input_Protocol.md` |
| FDBM ↔ Kriging compare | Single CLI entry for method panels | `testplot_kriging_vs_fdbm_cube.py` |
| Pauli \(A,\beta\) fit modes | Contact / residual vs Kriging | `PauliFitting_TestDesign.md`, `Pauli_A_beta_KrigingTransferability.md` |
| Cube FDBM ES quality | Too strong / asymmetric (esp. PTCDA); prefer cube ρ_NA | `Fukui_FDBM_panel_notes_2026-07-23.md` |
| df↔Fz height labeling | **CLI default amp-align** (df @ h, Fz @ h−amp; dense dz=0.1; window 3.7–4.7). Panel-fukui may still use coarser / same-h until updated | skill:`afm-plotting`; Fukui notes §2 |
| Chem-window defaults | CLI: df **3.7–4.7** @ Δz=0.1, amp-align Fz **2.7–3.7** | `user_guide/SPM_CLI.md` |
| `afm-morse` plot SSOT | Still ad-hoc `imshow`; should use `AFM_utils` | skill:`afm-plotting` |
| STM molecule registry | Only pentacene/PTCDA/benzene/pyridine; Fukui dimers missing | `stm_compare.MOLECULES` |
| Full bias-window STM sum | Σ channels at fixed bias (not MO-resolved) | future |
| ASCII / `.mol`/`.mol2` flags | SMILES wired; ASCII builder exists but not CLI flags | §A/D |
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
| Thin wrappers vs duplication | Prefer shared runners; **no logic growth inside `run_spm.py`** |
| **GUI↔CLI backend protocol** | Shared `SPMJobSpec` JSON + `run_spm_job`; BR-STM via ModularPipeline S6 (`Consolidate_GUI_CLI_Backend_Input_Protocol.md`) |

---

## Pre-imaging geometry — detailed ToDo

Priority order: **(1) ~~SMILES + gas-phase `opt`~~ (wired 2026-07-24)**, **(2) ASCII + `.mol`/`.mol2` resolver**, **(3) USER science OK / polish**, **(4) FUTURE substrate dock**.

### A. Shared molecule resolution (SSOT for CLI + GUI scripts)

- [~] **SMILES path** — wired: `--smiles` / `--smiles-example` → `smiles_to_system` (`afm`, `opt`, `smiles-afm`). Gallery: `debug/spm_smiles_afm/`. **USER science OK** still open.
- [ ] **Inventory & extract headless loader** — one function e.g. `spammm.topology.molecule_io.load_molecule(path) -> AtomicSystem` covering `.xyz` / `.mol` / `.mol2`. Reuse `atomicUtils`; no Qt.
- [ ] **Deduplicate ASCII post-parse pipeline** — shared `build_from_ascii(...)` for GUI + CLI.
- [ ] **ASCII → AtomicGraph bridge** — populate `AtomicGraph` after ASCII build.
- [ ] **Wire remaining CLI input flags**:
  - `--xyz PATH` (existing)
  - `--mol` / `--mol2` PATH (or `--structure` by extension)
  - `--ascii` / `--ascii-example` (`ASCII_EXAMPLES`)
  - `--smiles` / `--smiles-example` (**done**)
  - Precedence documented in `user_guide/SPM_CLI.md` (**updated**)

### B. Gas-phase optimization (`run_spm.py opt`) — **wired; science OK pending**

- [~] **Subcommand** `opt` — `--method {uff,spff,lff,dftb}`; planar + PCA long→x via `FFController.optimize_vacuum`.
- [~] **Outputs** — `{name}_opt.xyz`, `SUMMARY.out`; `REVIEW:` paths.
- [~] **Chain** — `smiles-afm` = SMILES → opt → prolonged `afm` (or two-step `opt` → `afm --xyz`).
- [ ] **Tests** — L0 benzene/H2O SPFF finite E; DFTB opt mark `slow` if available.
- [ ] **USER confirmation** before acceptance (UFF may leave high fmax on some aromatics — images usable).

```bash
python run_spm.py opt --xyz data/xyz/benzene.xyz --method spff --outdir debug/spm_opt_benzene
python run_spm.py opt --smiles-example naphthalene --method uff
python run_spm.py smiles-afm --example naphthalene --method uff
```

### C. SMILES → AtomicGraph — **parser + CLI wired; GUI open**

Cross-links: `ARCHITECTURE_ROADMAP` §9, `TASKS.md` T07, `ToDO_GUI.md`, Wikipedia [SMILES](https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System).

- [~] **`spammm/topology/smiles.py`** — `parse_smiles` / `smiles_to_system` / `SMILES_EXAMPLES`; pure + soft RDKit.
- [~] **Tests** — `tests/topology/test_smiles.py` (22 passed).
- [~] **Wire** `--smiles` / `--smiles-example` + `smiles-afm`.
- [ ] **GUI** — SMILES text box calling the same `parse_smiles`.
- [ ] **USER OK** on example gallery under `debug/spm_smiles_afm/`.

### D. ASCII-art as first-class CLI input

- [ ] **`run_spm.py` flags** `--ascii` / `--ascii-example` → shared `build_from_ascii` (section A).
- [ ] **Keep** `python -m spammm.topology.ascii_art_heterocycle` working; thin-wrap shared builder (no duplicated H/relax logic).
- [ ] **Imaging path** — `afm` / `stm` accept ASCII the same way as `--xyz` once resolver exists.
- [ ] **Export** — after ASCII build (+ optional `opt`), write `.xyz`/`.mol2` for reproducibility.

### E. Substrate `dock` / surface relax — **FUTURE (harder)**

Keep out of near-term scope, but track explicitly so the CLI roadmap stays honest. Details live in surface/FAF tasks; CLI only adds a thin subcommand later.

- [ ] **Documented deferral** — molecule-on-substrate relaxation via **GridFF** or **FAF (folded atomic functions)** is **more difficult** (fit/cache substrate, pose DOF, pins, PBC, flexible vs rigid). **Do not start** until gas-phase `opt` + loaders are done and USER prioritizes.
- [ ] **Future subcommand** `dock` (name TBD) with `--substrate`, `--method {gridff,faf,spff+faf}`, rigid vs flexible.
- [ ] **Reuse (when started):**
  - GridFF rigid: `RigidBodyDynamics.from_xyz_and_grid` / `run_gridff`
  - FAF rigid: `FoldedRigid.relax_folded` / folded `RigidBodyDynamics` (see `RigidBodyDynamicsWithFoldedBasisSubstrate.md`, `TestDesign_FoldedBasisRigidBody.md`, `Report_PTCDA_NaCl_FAF_RigidImaging.md`)
  - Flexible + FAF: existing `do_faf` paths in SPFF/UFF + `FFController` — GUI `FoldedRigidExtension` / FF panel as reference, not copy-paste
- [ ] **CLI thin wrap only** — no new physics in `run_spm.py`; shared runner e.g. `spammm.surfaces.dock_molecule(...)`.
- [ ] **Optional later** — DFTB slab / adsorbate SCF (`run_pbc` / ModularPipeline S1); Cosserat / rods (`Import_CosseratRods_PTCDA.md`).

Cross-links for E: `doc/ToDo/ToDo.agents.md`, `doc/ToDo/Features.audit.md` §CLI, `doc/Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md`, `doc/Tasks/PairFF_GUI_Integration.md` (flexible vs rigid product split).

---

## Acceptance (when USER confirms)

- [ ] User can run Morse, FDBM, Kriging AFM and STM orbitals/current from repo root without GUI
- [ ] `user_guide/SPM_CLI.md` matches `--help` and lists open roadmap
- [ ] README points to CLI + `user_guide/`
- [ ] Science campaigns (cube ES, prolonged, STM panels) remain **investigating** until explicit USER OK
- [ ] Bond-resolved STM + substrate `dock` + light-STM / charge-rings documented as open (not silently claimed ready)
- [ ] BR-STM available from CLI using the **same** ModularPipeline path as GUI (see consolidate task)
- [ ] Shared molecule resolver: `.xyz`/`.mol`/`.mol2` + ASCII (+ SMILES when module lands)
- [ ] `run_spm.py opt` gas-phase SPFF (and DFTB) works on a small molecule; writes geometry + SUMMARY; **USER OK** before claiming done
- [ ] No duplicated FF/DFTB/ASCII logic in CLI — only imports from shared modules
- [ ] Surface GridFF/FAF `dock` remains explicitly **future** (unchecked) until prioritized

---

## Related docs

- User guide — `user_guide/SPM_CLI.md`
- STM basis campaign — `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md`
- Prolonged / Fukui panel — `doc/Tasks/ProlongedRadialBasis_DFTB.md`
- Kriging import — `doc/Tasks/Import_KrigingGridFF.md`
- Charge rings — `doc/Tasks/Import_ChargeRings_PME.md`
- Features audit — `doc/ToDo/Features.audit.md` §2.4 / CLI
- Agent ToDo — `doc/ToDo/ToDo.agents.md`
- GUI ToDo (ASCII/SMILES) — `doc/ToDo/ToDO_GUI.md`
- SMILES roadmap — `doc/ARCHITECTURE_ROADMAP.md` §9 · `doc/TASKS.md` T07
- SMILES→AFM CLI notes — `doc/Reports/SPM_CLI_smiles_afm_2026-07-24.md`
- Fukui df↔Fz / amp — `doc/Reports/Fukui_FDBM_panel_notes_2026-07-23.md`
- FF relax design — `doc/GUI_FF_Relaxation.md` · `doc/Tasks/done/RelaxationExtension_Design.md` · `PerfBenchmark_Relaxation.md`
- **GUI↔CLI shared backend / BR-STM** — `doc/Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md`
- **FAF / folded substrate** — `doc/Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md` · `Report_PTCDA_NaCl_FAF_RigidImaging.md`
- Topology index — `spammm/topology/README.md`
