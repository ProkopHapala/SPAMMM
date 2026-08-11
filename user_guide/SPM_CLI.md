# SPAMMM SPM CLI — user guide

Headless AFM / STM imaging from the **repository root**, without opening the GUI.

| Entry | Role |
|-------|------|
| [`run_spm.py`](../run_spm.py) | User-facing CLI (this guide) |
| `spammm/SPM/` | Physics + plotting SSOT (`AFM_utils`, `stm_compare`, `KrigingGridFF`) |
| `spammm/topology/smiles.py` | SMILES → `AtomicGraph` / `AtomicSystem` |
| `spammm/forcefields/FFController.py` | `optimize_vacuum` (UFF/SPFF/LFF/DFTB + planar + PCA) |
| `tests/SPM/testplot_*.py` | Developer diagnostics / L2 plots (thin wrappers) |

---

## Quick start

```bash
python run_spm.py --help
python run_spm.py smiles-afm --example naphthalene --method uff
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
| `afm` | FDBM (Pauli + ES), stock / prolonged / cube; SMILES / xyz | **ready** |
| `opt` | Gas-phase relax (UFF / SPFF / LFF / DFTB); planar + PCA | **ready** |
| `smiles-afm` | SMILES → planar opt → prolonged FDBM (+ atom dots) | **ready** |
| `panel-fukui` | Fukui DFT cube vs DFTB stock vs prolonged | **ready** |
| `replot-panel` | Replot panel NPZ (per-image contrast) | **ready** |
| `afm-morse` | Morse/LJ + point-charge Coulomb, no density | **ready** |
| `afm --model contact_pme` | PAW particle-mesh Morse(+Q), no 3D FF image | **ready** |
| `afm-kriging` | DFT Kriging GridFF → PP relax → Fz/df | **ready** |
| `stm orbitals` | Frontier MO **ψ** (signed phase), DFTB + pySCF | **ready** |
| `stm current` | MO-resolved **STM current** I≥0, s/p_z/p_y tips | **ready** |
| `stm panel` | HOMO/LUMO vacuum STM (stock vs prolonged vs pySCF) | **ready** |
| `stm fgr` | FGR \(H-ES\) vs legacy overlap_exp (Level-B long-tail) | **review** |
| `basis-tails` | Central-C **ρ(z)** + **Pauli E(z)** log (GPAW/pySCF/stock/prolonged) | **ready** |

### Orbital vs STM (plot convention)

| Kind | Quantity | CLI | Colormap |
|------|----------|-----|----------|
| **Orbital** | Signed ψ (phase visible) | `stm orbitals` | RdBu_r |
| **STM current** | I ≥ 0, no phase; tip picks φ_t | `stm current` | viridis |

Both `stm orbitals` and `stm current` write **vertical** (E↑) and **horizontal** (E→) spectrum↔map figures by default (`--layout both`).

---

## Basis tails (talk plots)

Central-carbon **ρ(z)** and **Pauli E(z)** on a log axis (6 decades), GPAW / pySCF / DFTB stock / prolonged:

```bash
python run_spm.py basis-tails --molecule pentacene,PTCDA
python run_spm.py basis-tails --molecule PTCDA --tip-mode gaussian --sigma 0.35 --A 1 --beta 1
```

Artifacts: `debug/presentation_basis_tails/*_central_C_{rho,pauli,rho_pauli}_*.{svg,png}`.  
Pauli uses the **same** (A,β) for every density so the vacuum-tail story tracks ρ (default A=β=1 → raw overlap; default Gaussian σ=0.35 Å so E_Pauli follows ρ tails).

---

## Molecule inputs

| Flag | Meaning |
|------|---------|
| `--xyz PATH` | Geometry file |
| `--smiles STRING` | OpenSMILES organic subset → `smiles_to_system` |
| `--smiles-example NAME` | Named entry from `SMILES_EXAMPLES` (benzene, naphthalene, terephthalic_acid, thymine, …) |
| `--cube` / `--esp-cube` | Density / ESP cubes (FDBM) |

**Not yet:** `--ascii` / `--mol` / `--mol2` as first-class shared flags (ASCII builder exists in `topology/ascii_art_heterocycle`; wire later).

**Default orientation (AFM / opt):** PCA long axis → **x** (`orientPCA` / `rotMatPCA`); skip with `--no-orient`.

---

## Geometry prep

### `opt` — vacuum relax

```bash
python run_spm.py opt --smiles-example benzene --method uff
python run_spm.py opt --xyz data/xyz/benzene.xyz --method spff --outdir debug/spm_opt
python run_spm.py opt --smiles 'c1ccccc1' --method dftb --basis 3ob-3-1
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--method` | `uff` | `uff` \| `spff` \| `lff` \| `dftb` |
| `--nsteps` / `--fmax` | 1000 / 0.05 | Relax stopping |
| `--no-planar` | off | Keep 3D after opt (default: flatten to xy) |
| `--no-orient` | off | Skip PCA long→x |

After opt (when planar): `make_planar_xy` then PCA, then **all atom z = 0**. Same default on `afm` for SMILES/xyz (skip with `--no-planar`; cube path keeps cube atom z). Planarity is **forced** — AFM “buckling” contrast is chemical/electronic, not from atom z.

Outputs: `{name}_opt.xyz`, `SUMMARY.out` under `--outdir`.

### `smiles-afm` — SMILES → planar opt → prolonged AFM

Batch-friendly path for aromatics:

```bash
python run_spm.py smiles-afm --method uff                    # all SMILES_EXAMPLES
python run_spm.py smiles-afm --example naphthalene terephthalic_acid
python run_spm.py smiles-afm --example thymine --method dftb
```

Per molecule under `debug/spm_smiles_afm/<name>/`:

- `{name}_opt.xyz` — planar, long axis ‖ x
- **`compare_per_image.png`** — default strip (df + Fz, amp-aligned; **per-panel clim**, long→vertical, tight)
- **`stage_prolonged.png`** — light field diagnostics
- `SUMMARY.out`

---

## AFM

### Height window, amplitude, and plots (important)

Probe heights on the CLI are **df probe heights** above the molecule plane (`mol_z = max atom z`).

| Parameter | Default | Notes |
|-----------|---------|--------|
| `--h-min` / `--zmin` | **3.7** Å | df window start |
| `--h-max` / `--zmax` | **4.7** Å | df window end |
| `--h-step` / `--dz` | **0.1** Å | dense stack for `compute_df_amp` |
| `--amp` | **1.0** Å | peak oscillation amplitude |

**Amp-align (default):** each column shows

- **df** at labeled \(h\) (e.g. 3.7…4.7)
- **Fz** at \(h - \mathrm{amp}\) (e.g. 2.7…3.7)

so morphologies match closest-approach physics. Use `--no-amp-align` only if you want Fz and df at the **same** labeled \(h\) (expect ~1 Å visual mismatch).

For vertical oscillation, the PP scan covers `[h_min − amp, h_max + amp]` so df at the window edges is valid.

For lateral/arbitrary oscillation, `--osc-dir nx,ny,nz` changes the force projection and
directional derivative, not the approach axis:

```bash
python run_spm.py afm --xyz data/xyz/PTCDA.xyz --projection prolonged --osc-dir 1,0,0
```

- The scan volume and displayed columns remain `(x,y,z)` with `--h-min/--h-max` as z heights.
- Pure x/y oscillation samples the finite-amplitude path on a padded lateral grid, then crops
  back to the requested XY image; Fz is shown at the same z.
- For arbitrary `n`, only `amp*abs(nz)` pads z and shifts the amp-aligned Fz comparison.
- `--base-pos` is only a constant world-coordinate scan offset; it does not set a lateral-AFM height.

**Which PNGs to write** (`--plots` CSV):

| Value | Meaning |
|-------|---------|
| `compare,stage` | **default** — `compare_per_image.png` + `stage_*.png` |
| `compare` | strip only |
| `tip,df,fz` | deep debug grids |
| `all` / `debug` | everything |
| `none` | compute only |

### `afm` — FDBM (density-based)

```bash
python run_spm.py afm --xyz data/xyz/benzene.xyz --basis 3ob-3-1 --projection stock
python run_spm.py afm --smiles-example naphthalene --projection prolonged --show-atoms
python run_spm.py afm --xyz data/xyz/PTCDA.xyz --projection both --plots all
python run_spm.py afm --cube /path/to/rho_N.cube --xyz data/xyz/mol.xyz --projection both
```

Key flags: `--basis`, `--projection` (`stock`|`prolonged`|`both`), `--tip-mode` (`co`|`gaussian`), `--scale` (`per_image` default | `per_column` | `common`), `--show-atoms`, `--plots`, height / amp flags above.

**Strip display SSOT** (skill:`afm-plotting` § overview strip display, USER 2026-07-28):

| | Policy |
|--|--------|
| df clim | **per panel** min/max (never shared df colorbar) |
| long axis | **vertical** on overview strips |
| packing | **tight** 6×N (no empty gutters) |
| CLI `--scale` | default **`per_image`** |

Helper: `AFM_utils.plot_afm_variant_height_strip`.

**Dual basis:** prolonged ρ is **Pauli only**; electrostatics always from stock Δρ.

### Pauli A,β — fit vs evaluate

| | Rule |
|--|------|
| **SSOT dict** | `AFM.PAULI_FITTED_DEFAULTS` |
| **CLI default (3ob-3-1)** | **A = 124.84**, **β = 1.4330** |
| **Evaluation** (`afm`, `panel-fukui`, gallery) | Same `(A,β)` for every molecule and every ρ row in the strip |
| **Fitting** | Separate scripts only; may explore mol-specific fits — **never** wire those into eval/panel |

Do not special-case PTCDA (or any molecule) in evaluation. See skill:`afm-plotting` § Pauli A,β.

### `afm-morse` — classical force field

No SCF / no cube — Morse (or `--lj`) + tip point charges on a 3D grid, then `run_scan` → Fz/df slices.

```bash
python run_spm.py afm-morse --xyz data/xyz/pentacene.xyz
python run_spm.py afm-morse --xyz data/xyz/benzene.xyz --lj
```

### `afm --model contact_pme` — PAW particle-mesh Morse(+Q)

Same ScanSpec / height SSOT / `plot_afm_variant_height_strip` as Morse and contact 2.5D.
Fits coarse mesh + compact PIC cores (`split_mode=paw` default), then PP-AFM scan.

```bash
python run_spm.py afm --model contact_pme --xyz data/xyz/pyridine.xyz --show-atoms
python run_spm.py afm --model contact_pme --xyz data/xyz/PTCDA.xyz --pme-q-tip -0.1 --show-atoms
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--pme-split` | `paw` | Long/short split (`paw`/`hermite`/`plateau`/`rho`/`softcore`) |
| `--pme-q-tip` | `0` | Radial tip charge (multi-site `tipQs` forced off) |
| `--pme-h-mesh` | `1.0` | Coarse mesh spacing [Å] |
| heights / margin | same as `afm` | `--h-min 3.7`…`--h-max 4.7`, `--scan-margin`, `--margin` |

### `afm-kriging` — DFT GridFF → probe-particle AFM

Uses linked Mithun DFT z-scans (`data/mithun_afm_scans/…`) → Kriging volume → `AFMulator.scan_fdbm`.

```bash
python run_spm.py afm-kriging --endgroup HHO-h-p_1 --tip H2O_O
python run_spm.py afm-kriging --klat 0.5,1.0,2.0 --outdir debug/spm_afm_kriging
```

### `panel-fukui` / `replot-panel`

**Same height SSOT as `afm`** (do not pass coarse `2.5–5.7` / `dz=0.4` unless USER asks):

| | Default |
|--|---------|
| df window | **3.7–4.7 Å**, **dz=0.1** |
| Fz in strip | **amp-aligned** @ **h−amp** (→ 2.7–3.7 with amp=1) |
| Rows | **DFT cube → prolonged → stock** for df and Fz (**6 rows** when `rho_N.cube` exists) |

```bash
# full systematic gallery → debug/AFM_CLI_FDBM/<mol>/
python tests/SPM/run_afm_cli_fdbm_gallery.py

python run_spm.py panel-fukui --molecule PTCDA pentacene --outdir debug/AFM_CLI_FDBM
python run_spm.py panel-fukui --molecule adenine-uracil azaindol_dimer \
  --outdir debug/AFM_CLI_FDBM
# uses CLI height defaults above — do NOT add --h-min 2.5 --h-step 0.4
python run_spm.py replot-panel --panel-dir debug/AFM_CLI_FDBM --molecule PTCDA
```

Cubes searched in order: `…/Fukui_AFM/new/<name>_PBE_def2-SVP/`, then `…/pyscf_fukui_cluster/<name>_PBE_def2-SVP/` (`rho_N.cube`).

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

### `stm br` — three-stage BR-STM campaign

Separate figures (do **not** mash STM into the AFM strip):

1. **Pure STM** (orbital identity, prolonged STO) — heights `--stm-heights` (default 0.5,1.5,2.5)
2. **AFM** df + Fz(amp-align) + |dxy|; plus **PP xy red dots** every `--pp-stride` pixels
3. **BR-STM vs flat STM** at Fz heights (where PP bend is large)

```bash
python run_spm.py stm br --xyz data/xyz/PTCDA.xyz --show-atoms --mo 1
python run_spm.py stm br --xyz data/xyz/benzene.xyz --mo 0 --outdir debug/spm_brstm/benzene
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--h-min` / `--h-max` / `--h-step` | **3.7 / 4.7 / 0.1** | AFM df window (same SSOT as `afm`) |
| `--amp` | **1.0** | df amplitude; Fz & |dxy| at **h−amp** unless `--no-amp-align` |
| `--stm-heights` | `0.5,1.5,2.5` | Stage1 pure-STM heights |
| `--mo` | `0 1` | offsets vs HOMO for Stage1 (`0`=HOMO, `1`=LUMO) |
| `--pp-stride` | `4` | every Nth pixel for PP xy red-dot overlay |
| `--K-LAT` | `0.5` | PP spring [N/m] |
| `--bond-length` | `3.0` | tip–probe lever L [Å] |
| `--field` | `psi2` | `ldos` / `psi2` / `psi` |

Outputs under `debug/spm_brstm/<mol>/`: `01_stm_pure.png`, `02_afm_df_tipdisp.png`, `02_pp_xy_dots.png`, `03_brstm_vs_stm.png`, `SUMMARY.out`.

### `stm panel` — HOMO/LUMO vacuum compare

Multi-height panels: DFTB stock vs prolonged (+ optional pySCF).

```bash
python run_spm.py stm panel --molecule pentacene,PTCDA --field psi2
python run_spm.py stm panel --molecule pentacene --heights 2.5,3.0,3.5 --bases mio,3ob
```

Known molecules: `pentacene`, `PTCDA`, `benzene`, `pyridine` (or any `--xyz`).

### `stm fgr` — FGR \(H-ES\) vs overlap

Compares legacy `mo_overlap_points_exp_sk` to long-tail Level-B tables: \(I_S\), \(I_H\), \(I_\tau=|c^\dagger(H-ES)c|^2\).

```bash
python run_spm.py stm fgr --molecule pentacene,PTCDA --stm-tips s,pz --stm-z-above 3.0
```

Outputs under `debug/stm_fgr_compare/<mol>/`. Report: `doc/Reports/STM_FGR_Transfer_H_ES_2026-07-29.md`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--stm-z-above` | `3.0` | Height above molecular plane [Å] |
| `--stm-tips` | `s,pz` | Point-tip orbitals |
| `--eh-K` | `1.75` | Extended-Hückel \(K\) |
| `--tip-elem` | `C` | Phantom tip element for SK types |
| `--rcut` | `10.0` | Atom-pair cutoff [Å] |

---

## Plotting SSOT

- AFM E / Fz / df → `spammm.SPM.AFM_utils` (`imshow_afm`, `plot_afm_variant_height_strip`, …)
- STM / frontier spectrum↔maps → `plot_spectrum_with_orbitals`, `plot_stm_basis_compare_panel`
- Do not use ad-hoc `imshow` loops in user scripts

See `doc/AGENTS/skills/afm-plotting/SKILL.md`.

---

## Related docs

- Task tracker — `doc/Tasks/SPM_CLI_Headless.md`
- STM basis compare — `doc/Reports/STM_ExtendedBasis_OrbitalCompare.md` · `doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md`
- STM FGR \(H-ES\) — `doc/Reports/STM_FGR_Transfer_H_ES_2026-07-29.md` · `doc/TopicalAudit/STM_FGR_Transfer.md`
- Prolonged basis — `doc/Tasks/ProlongedRadialBasis_DFTB.md`
- Kriging import — `doc/Tasks/Import_KrigingGridFF.md`
- Fukui panel / df↔Fz amp — `doc/Reports/Fukui_FDBM_panel_notes_2026-07-23.md`
- SMILES→AFM pipeline notes — `doc/Reports/SPM_CLI_smiles_afm_2026-07-24.md`
- Features audit — `doc/ToDo/Features.audit.md`

---

## Roadmap / open issues

**Wired in CLI** (science campaigns stay *investigating* until you confirm):

1. FDBM AFM (`afm`) — stock / prolonged / cube; SMILES; amp-aligned height strip
2. Gas-phase `opt` + `smiles-afm` (planar + PCA long→x)
3. Fukui panel + per-image replot
4. Morse/LJ + Coulomb AFM (`afm-morse`)
5. Kriging GridFF AFM (`afm-kriging`)
6. STM orbitals + STM current + vacuum panel + **BR-STM** (`stm br`)

**Still open** (see `doc/Tasks/SPM_CLI_Headless.md`):

| Priority | Item | Notes |
|----------|------|-------|
| **Near** | **GUI↔CLI param SSOT** | Heights/amp/Pauli/FIRE/scan-margin fixed in GUI; residual: scan sampling `arange` vs `linspace` |
| Near | **Confirm CLI FAST_S3 cutover** | `afm` + `panel-fukui` now use `AFM_utils.run_fdbm_pp_from_density` (FAST_S3 default). Smoke: `debug/spm_afm_fast_smoke/`. Await USER REVIEW before marking Done. |
| Near | **GUI↔CLI input protocol** | JSON `SPMJobSpec` + `run_spm_job` |
| Near | FDBM ↔ Kriging compare | Wrap `testplot_kriging_vs_fdbm_cube` |
| Near | Cube FDBM ES / NA multipoles | Prefer cube `ρ_NA`; see Fukui notes §1b |
| Near | Inputs: ASCII / `.mol` / `.mol2` | Shared resolver; ASCII builder exists |
| Near | `afm-morse` plot SSOT | Still ad-hoc `imshow` |
| Later | **`dock` (substrate)** | GridFF / FAF — deferred |
| Later | Light-STM / charge-ring imaging | |
| UX | pip console script | `run_spm` via packaging task |

**Parity gate (done):** `python tests/SPM/testplot_cli_vs_modular_parity.py` → `debug/cli_vs_modular_parity/`.

Task tracker: [`doc/Tasks/SPM_CLI_Headless.md`](../doc/Tasks/SPM_CLI_Headless.md) · consolidate GUI/CLI: [`doc/Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md`](../doc/Tasks/Consolidate_GUI_CLI_Backend_Input_Protocol.md) · agent list: [`doc/ToDo/ToDo.agents.md`](../doc/ToDo/ToDo.agents.md).
