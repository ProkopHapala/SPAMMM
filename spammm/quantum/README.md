# quantum/

Quantum chemistry integration: DFTB+ scans, Hessians for vibrations, electron densities for FDBM AFM.

- **DFTB_utils.py** — DFTB+ I/O, `run_dftb_sp` / `run_dftb_relax`, Mulliken parse, Hessian, `clean_dftb_workdir`, failure diagnostics. **SK path resolution SSOT**: `SK_PATHS` dict, `_get_sk_paths()`, `_check_sk_path()` — see `doc/Caveats.md` §7. **PBC runs**: `run_pbc`/`makeDFTBjob_pbc` — `nk` MP mesh with `k_shift` ((0,0,0)=Γ-centred incl. BZ edge; default 0.5 half-shift misses it), `klist=` explicit fractional k-points + weights (extended-zone runs), `do_relax`, `Mixer` (DIIS goes singular on Γ-mesh metals → use Broyden/`None`), `extra_hsd` (`Analysis{WriteEigenvectors=Yes}` → `eigenvec.bin`)
- **config_utils.py** — `get_dftb_sk_path()` smart search (config → env var + subdir patterns → parent fallback), `firecore_config.json` loader
- **coordinate_scan.py** — Reaction-coordinate engine: control grids, pm-NEB (endpoint relax + all-atom interp), Mulliken charges per frame → `ScanDataset`
- **esp_grid.py** — Precompute Coulomb ESP stacks `[nframes, ny, nx]` from charges (KE/r, same as QEq)
- **hbond_scan.py** — Legacy ASCII rigid proton-transfer scan (0.1 Å axis steps); kept for existing tests
- **pi_bond_order.py** — π bond orders from DFTB density matrix + **band unfolding**. Cluster: `run_dftbcore_sp` (minimal-HSD SCF → dense DM+S), `ao_layout` ([s,py,pz,px] per sp atom), `pi_bond_order_matrix` (p⊥ sub-block + Löwdin `Sπ^½ Pπ Sπ^½`), `plot_bond_scalar_map` (colored-bond maps). Periodic: `pi_bond_orders_pbc` (dftb+ binary → `oversqr.dat` S(k) + `eigenvec.bin` C(k) → per-k Löwdin, ICELL phase per pair). Unfolding: `subcell_group_indices` (primitive-orbital equivalence groups across supercell subcells, ideal-lattice grouping, `strict=False` tolerates defect atoms), `unfold_T_weights` (**canonical** — spectral projectors of the subcell-translation operator in the true S(k) metric, `w_j=(1/Nc)Σ_m e^{−i2πq_j m} c*T^m S c`; required for nonorthogonal bases: DFTB |S_off|~0.4, dzp ~0.5), `unfold_spectral_weights_DEPRECATED` (Euclidean subcell DFT — **wrong metric for S≠I**: leaks ~20-30% into adjacent channels, artificial k-edge collapse; kept for legacy drivers, see `doc/TopicalAudit/BandUnfolding_Ribbons.md`), `unfold_projector_weights` (SUPERSEDED S-metric oblique projector experiment), `read_band_out` / `read_eigenvec_bin` / `read_sqr_dat` / `read_overreal_pairs`. Drivers: `tests/topology/testplot_bond_order.py` → `debug/test_bond_order/`; `tests/quantum/testplot_unfold.py` → `debug/unfold/` (canonical overlay, see `doc/TopicalAudit/BandUnfolding_Ribbons.md`)
- **gpaw_hs.py** — same analysis for GPAW `hs.npz` exports (`gpaw_hs_export.py`): `load_hs`, `pi_channels` (per-shell l=1 triplets, GPAW m-order y,z,x → x,y,z), `pi_project_ch` (multi-channel π projection along plane normal), `pi_bond_orders_hs` (per-k Löwdin; `single=True` keeps the dominant pz channel per atom — the dzp 2nd shell is diffuse and makes Sπ nearly singular; pair cell-image picked by dominant Fourier component over R∈{−1,0,+1}, NOT from geometry — GPAW wraps boundary atoms like DFTB ICELL), `band_energy` (Σw Tr(Hρ), not a DFTB H0 analog). Importer: `testplot_ribbon.py --gpaw-import` → `debug/ribbon_gpaw/` caches, DFTB figures reused (`--sk gpaw` on `--enum-plot/--enum-diff/--nonlin/--analyze/--compare/--j-terms/--geom-plot`; j-terms splits J into E_band+rest since detailed.out terms don't exist). `test_ribbon_stm_sys.py --gpaw <root>` → TH-LDOS z-slices + dzp unfold from `all.gpw`, same cache/figure layout → `debug/ribbon_gpaw/stm_xscan/`. See `doc/ERC_private/ribbon_switch_recap.md` §7
- **pySCF_utils-new.py** — **SSOT today** (GPU OpenCL / smallDFT / stock CPU; CO z-scan; frontier MO cubes). Notes: `doc/AGENTS/notes/pyscf-gpu-scf.md`. Hyphen → importlib.
- **pySCF_utils.py** — **legacy** thin RHF/opt (~91 L); do not extend. **Merge plan:** replace with `-new` → single `pySCF_utils.py` (`doc/Tasks/Refactor_LargeModules.md` §12).
- **DFTB/** — ctypes wrapper, basis parser, GPU density projection (dense NA DM default for FDBM), basis optimizer — see `DFTB/README.md`
- **PauliSolverCL.py** — OpenCL full PME (`kernels/PME.cl`, hardcoded 4 sites); FireCore `pauli_ocl` on `OpenCLBase`
- **PauliSolverCL8.py** — OpenCL 8-site PME (`kernels/PME8.cl`, 256 states, sparse iterative Euler solver); same API as PauliSolverCL. Report: [`MoleculeExtraction_PME8_2026-08-18.md`](../../doc/Reports/MoleculeExtraction_PME8_2026-08-18.md)
- **pauli_scan.py** — slim xy/xV API (2/3→4 embed, Wij); Ruslan dimer + fig3 trimer (Qzz=0 NDR)
- Fixtures: `data/charge_rings/` (`symmetric_trimer.json`, Ruslan_*, fig3). Tests: `test_pme_pauli.py`, `test_pme_trimer.py`; demos `testplot_charge_rings_{ruslan,trimer}.py`
- GUI: `spammm/GUI/ChargeRingsExtension.py` (JSON load/save, Calc XY/xV/1D, cut overlay, state probs)
- Audit: `doc/TopicalAudit/ChargeRings_PME.md` · Task: `doc/Tasks/Import_ChargeRings_PME.md` (A+D+F Done)
- Pose glue (sites = molecules, design): `doc/TopicalAudit/RigidBody.md`, `doc/Tasks/RigidMoleculePose_SSOT.md`

**Open campaigns:** prolonged Slater STM panel (`doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md`, `tests/SPM/testplot_stm_basis_compare.py`); AFM prolonged (`doc/Tasks/ProlongedRadialBasis_DFTB.md`); Kekulé π → exponential RI density (`doc/Tasks/Kekule_ExponentialDensityFit.md`).

## Reaction-coordinate scan (graph / GUI)

**SSOT:** `coordinate_scan.py` + `topology/scan_dataset.py`. Topical doc: `doc/Topics/ReactionCoordinateScan.md`.

```bash
PYTHONPATH=. python3 demos/gui_scripts/rc_scan_offline.py
pytest tests/topology/test_scan_dataset.py::test_pm_neb_relaxed_dftb -s
./run_gui.sh --script demos/gui_scripts/rc_scan_review.py
```

**Requires:** `DFTB_EXE`, `DFTB_SK_PATH`.

**Caveats:** GUI geometry must use `build_ascii_hbond_system` (see `doc/Takeways.md`); pm-NEB relaxed runs Mulliken SP on every interpolated frame.

## H-bond proton-transfer scan (ASCII legacy)

**Pipeline:** `build_ascii_hbond_system(name)` → `run_hbond_transfer_scan(ds=0.1)` → `save_hbond_scan_artifacts()`

```bash
python tests/topology/testplot_hbond_scan.py --name 2Quinolone --step 0.1
pytest tests/topology/test_hbond_scan.py -s
```

**Artifacts:** `debug/test_hbond_scan/hbond_*.{png,xyz}`

## Ribbon band structure & unfolding

**Pipeline:** `build_zigzag_ribbon` (topology, `PASSIVATION_GROUPS` per-site edge chemistry) → `run_pbc` SP/relax → `band.out`+`eigenvec.bin` → canonical overlay + `unfold_spectral_weights_DEPRECATED` (deprecated Euclidean weights; new work should use `unfold_T_weights` with S(k)). Driver: `tests/quantum/testplot_unfold.py` (modes: folding validation, `--passiv`, `--switch`, `--sys3`); artifacts `debug/unfold/` + `index.html` gallery.

**Topical doc:** [`doc/TopicalAudit/BandUnfolding_Ribbons.md`](../../doc/TopicalAudit/BandUnfolding_Ribbons.md) — full detail incl. canonical plot layering, gap-vs-width table, fold-validation numbers (0.1–0.2 meV).
