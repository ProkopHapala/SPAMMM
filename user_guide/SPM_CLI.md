# SPAMMM SPM CLI — user guide

Headless AFM / STM imaging from the **repository root**, without opening the GUI.

| Entry | Role |
|-------|------|
| [`run_spm.py`](../run_spm.py) | User-facing CLI (this guide) |
| `spammm/SPM/` | Physics + plotting SSOT (`AFM_utils`, `stm_compare`, `KrigingGridFF`) |
| `tests/SPM/testplot_*.py` | Developer diagnostics / L2 plots (thin wrappers) |

---

## Quick start

```bash
python run_spm.py --help
python run_spm.py stm --help
python run_spm.py stm orbitals --help
```

**GPU:** OpenCL prefers NVIDIA. STM DFTB projection and Morse AFM need the GPU ICD visible (not sandboxed PoCL).

```bash
python -c "import pyopencl as cl; print([(p.name,[d.name for d in p.get_devices()]) for p in cl.get_platforms()])"
```

Awkward FDBM grid sizes: keep **CPU FFT** (`afm` default `--cpu-fft`).

---

## Command map

| Command | Physics | Status |
|---------|---------|--------|
| `afm` | FDBM (Pauli + ES), stock / prolonged / cube | **ready** |
| `panel-fukui` | Fukui DFT cube vs DFTB stock vs prolonged | **ready** |
| `replot-panel` | Replot panel NPZ (per-image contrast) | **ready** |
| `afm-morse` | Morse/LJ + point-charge Coulomb, no density | **ready** |
| `afm-kriging` | DFT Kriging GridFF → PP relax → Fz/df | **ready** |
| `stm orbitals` | Frontier MO **ψ** (signed phase), DFTB + pySCF | **ready** |
| `stm current` | MO-resolved **STM current** I≥0, s/p_z/p_y tips | **ready** |
| `stm panel` | HOMO/LUMO vacuum STM (stock vs prolonged vs pySCF) | **ready** |

### Orbital vs STM (plot convention)

| Kind | Quantity | CLI | Colormap |
|------|----------|-----|----------|
| **Orbital** | Signed ψ (phase visible) | `stm orbitals` | RdBu_r |
| **STM current** | I ≥ 0, no phase; tip picks φ_t | `stm current` | viridis |

Both `stm orbitals` and `stm current` write **vertical** (E↑) and **horizontal** (E→) spectrum↔map figures by default (`--layout both`).

---

## AFM

### `afm` — FDBM (density-based)

```bash
python run_spm.py afm --xyz data/xyz/benzene.xyz --basis 3ob-3-1 --projection stock
python run_spm.py afm --xyz data/xyz/PTCDA.xyz --projection prolonged
python run_spm.py afm --xyz data/xyz/pentacene.xyz --projection both --outdir debug/spm_pentacene
python run_spm.py afm --cube /path/to/rho_N.cube --xyz data/xyz/mol.xyz --projection both
```

Key flags: `--basis`, `--projection` (`stock`|`prolonged`|`both`), `--tip-mode` (`co`|`gaussian`), `--h-min`/`--h-max`/`--h-step`, `--scale` (`per_column`|`per_image`|`common`).

**Dual basis:** prolonged ρ is **Pauli only**; electrostatics always from stock Δρ.

Outputs: `compare_*.png`, `df_*.png`, `Fz_*.png`, `SUMMARY.out`.

### `afm-morse` — classical force field

No SCF / no cube — Morse (or `--lj`) + tip point charges on a 3D grid, then `run_scan` → Fz/df slices.

```bash
python run_spm.py afm-morse --xyz data/xyz/pentacene.xyz
python run_spm.py afm-morse --xyz data/xyz/benzene.xyz --lj
```

### `afm-kriging` — DFT GridFF → probe-particle AFM

Uses linked Mithun DFT z-scans (`data/mithun_afm_scans/…`) → Kriging volume → `AFMulator.scan_fdbm`.

```bash
python run_spm.py afm-kriging --endgroup HHO-h-p_1 --tip H2O_O
python run_spm.py afm-kriging --klat 0.5,1.0,2.0 --outdir debug/spm_afm_kriging
```

### `panel-fukui` / `replot-panel`

```bash
python run_spm.py panel-fukui --molecule PTCDA pentacene
python run_spm.py replot-panel --panel-dir debug/fdbm_fukui_panel --molecule PTCDA
```

Cubes: `/home/prokop/SIMULATIONS/Fukui_AFM/pyscf_fukui_cluster/<name>_PBE_def2-SVP/rho_N.cube`

---

## STM

### `stm orbitals` — frontier MO maps (DFTB vs pySCF)

Signed ψ at z ≈ 0.5 Å above the molecular plane; eigenvalue ladder + spectrum↔orbital connectors.

```bash
python run_spm.py stm orbitals --molecule pentacene --n-near 5
python run_spm.py stm orbitals --molecule pentacene PTCDA --z-above 0.5 --layout both
python run_spm.py stm orbitals --xyz data/xyz/benzene.xyz --outdir debug/stm_orbital_compare
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--molecule` | `pentacene` | Registry name(s); comma ok |
| `--xyz` | — | Override geometry |
| `--z-above` | `0.5` | Slice height above mean z [Å] |
| `--n-near` | `5` | ±N MOs around HOMO/LUMO |
| `--layout` | `both` | `vertical` / `horizontal` / `both` |
| `--bases` | `3ob` | DFTB basis (`mio`, `3ob`) |
| `--pyscf-basis` | `def2-SVP` | pySCF basis |
| `--pyscf-xc` | `PBE` | pySCF XC |

Outputs under `debug/stm_orbital_compare/<mol>/frontier_diag/`:

- `spectrum_orbitals_vertical_z0.5_*.png`
- `spectrum_orbitals_horizontal_z0.5_*.png`
- `orbitals_z0.5_*.png`, `spectrum_*.png`, `SUMMARY.out`

### `stm current` — MO-resolved STM (I ≥ 0)

Current maps at z ≈ 3 Å; separate figure set per tip orbital (s, p_z, p_y).

```bash
python run_spm.py stm current --molecule pentacene --stm-tips s,pz,py
python run_spm.py stm current --molecule pentacene --stm-z-above 3.0 --layout horizontal
```

Outputs: `debug/stm_orbital_compare/<mol>/frontier_stm_diag/tip_{s,pz,py}/spectrum_stm_{vertical,horizontal}_z3.0_*.png`

### `stm panel` — HOMO/LUMO vacuum compare

Multi-height panels: DFTB stock vs prolonged (+ optional pySCF).

```bash
python run_spm.py stm panel --molecule pentacene,PTCDA --field psi2
python run_spm.py stm panel --molecule pentacene --heights 2.5,3.0,3.5 --bases mio,3ob
```

Known molecules: `pentacene`, `PTCDA`, `benzene`, `pyridine` (or any `--xyz`).

---

## Plotting SSOT

- AFM E / Fz / df → `spammm.SPM.AFM_utils` (`imshow_afm`, `plot_afm_variant_height_strip`, …)
- STM / frontier spectrum↔maps → `plot_spectrum_with_orbitals`, `plot_stm_basis_compare_panel`
- Do not use ad-hoc `imshow` loops in user scripts

See `doc/AGENTS/skills/afm-plotting/SKILL.md`.

---

## Related docs

- STM basis compare report — `doc/Reports/STM_ExtendedBasis_OrbitalCompare.md`
- Task — `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md`
- Prolonged basis — `doc/Tasks/ProlongedRadialBasis_DFTB.md`
- Kriging import — `doc/Tasks/Import_KrigingGridFF.md`
- Fukui panel notes — `doc/Reports/Fukui_FDBM_panel_notes_2026-07-23.md`

---

## Roadmap / open issues

**Wired in CLI** (science campaigns stay *investigating* until you confirm):

1. FDBM AFM (`afm`) — stock / prolonged / cube
2. Fukui panel + per-image replot
3. Morse/LJ + Coulomb AFM (`afm-morse`)
4. Kriging GridFF AFM (`afm-kriging`)
5. STM orbitals + STM current (DFTB + pySCF; vertical + horizontal layouts)
6. STM HOMO/LUMO vacuum panel (`stm panel`)

**Planned / not yet in CLI** (see also `doc/Tasks/SPM_CLI_Headless.md`):

| Priority | Item | Notes |
|----------|------|-------|
| Near | Bond-resolved STM (**BR-STM**) | GUI: `ModularPipeline` S6 already; CLI missing — share via job-spec runner (`Consolidate_GUI_CLI_Backend_Input_Protocol.md`) |
| Near | **GUI↔CLI input protocol** | JSON `SPMJobSpec` + `run_spm_job`; widgets/argparse adapters only; plots separate (matplotlib stills vs VisPy/blit) |
| Near | FDBM ↔ Kriging compare | One entry wrapping `testplot_kriging_vs_fdbm_cube` |
| Near | Pauli \(A,\beta\) fit modes | Contact / residual vs Kriging |
| Near | Cube FDBM ES fix | Strong/asymmetric ES (esp. PTCDA); clamp→compact-NA |
| Near | df↔Fz presentation | Annotate amp; Fz_u / Fz_r / df; dense Δz in chem window |
| Medium | **`relax` / `dock`** | Pre-optimize molecule on substrate (UFF/SPFF/LFF, DFTB, **GridFF**, **FAF**/folded) before imaging |
| Later | **Light-STM** | Optically driven / excited-state STM channels |
| Later | **Charge-ring imaging** | Ring-current / magnetic contrast (`Import_ChargeRings_PME.md`) |
| Later | Full multi-channel STM sum | Σ\|⟨φ_s\|H'\|φ_t⟩\|² at fixed bias |
| Later | Contact-surface / 2.5D AFM | Fast AFM path |
| UX | pip console script | `run_spm` via `PipInstall_Packaging.md` |

Task tracker: [`doc/Tasks/SPM_CLI_Headless.md`](../doc/Tasks/SPM_CLI_Headless.md) · consolidate GUI/CLI: [`doc/Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md`](../doc/Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md) · agent list: [`doc/ToDo/ToDo.agents.md`](../doc/ToDo/ToDo.agents.md).
