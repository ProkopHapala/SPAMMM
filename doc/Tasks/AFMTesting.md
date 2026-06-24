# AFM Testing Plan: Morse+Coulomb (Phase 1) and FDBM (Phase 2)

## Objective

Write comprehensive pytest tests for SPAMMM's AFM imaging pipeline. Two tracks:
- **Phase 1:** Classical Morse + Coulomb force fields on GPU grid — **tests written, need to run**
- **Phase 2:** Full Density-Based Model (FDBM) using DFTB electron density — **code exists, tests needed**

**Key insight:** SPAMMM already has a complete FDBM pipeline (`ModularAFMPipeline` with S1-S6, `compute_fdbm_forcefield`, `fft_poisson`, `compute_pauli_overlap`, `scan_fdbm`, etc.). The task is **testing existing code**, not implementing it.

**Reference repo:** `/home/prokop/git/FireCore/` — tests in `tests/tAFM/`, docs in `doc/Topics/AFM/`

---

## Phase 1: Morse + Coulomb — ✅ Tests Written

### Status

**Tests implemented** in `tests/SPM/test_afm_morse.py` — 6 tests:

| Test | What It Checks |
|------|---------------|
| `test_afm_grid_finite` | FF grid finite, E<0 well, forces decay at top |
| `test_afm_raw_scan` | Raw FE scan: Fz finite, close > far |
| `test_afm_relaxed_scan` | PP-relaxed: Fz finite, r>0.5 vs raw, differs |
| `test_afm_df_finite` | `compute_df`: finite, non-zero |
| `test_afm_morse_vs_lj` | Morse vs LJ: correlated (r>0.3) but different |
| `test_visual_afm_morse_images` | 2D Fz+df slices at 5 z heights (visual) |

### SPAMMM Code Under Test

| File | Key Functions |
|------|--------------|
| `spammm/SPM/AFM.py` | `AFMulator.__init__`, `assign_params`, `setup_grid`, `make_forcefield`, `get_raw_FE`, `run_scan`, `compute_df` |
| `data/ElementTypes.dat` | vdW radii, EvdW, QEq params |
| `data/xyz/` | CO.xyz, benzene.xyz, pentacene.xyz |

### FireCore Reference

| File | Use |
|------|-----|
| `tests/tAFM/afm_morse_pbc.py` | **Best reference** — full Morse+PBC pipeline (~560 lines): grid setup, `evalMorseC_QZs_toImg`, raw scan, relaxed scan, df, plots |
| `tests/tAFM/test_ptcda.py` | Simpler — LJ/Morse on PTCDA, QEq charges, raw+relaxed scan, df |
| `doc/Topics/AFM/AFM.md` | 5-step workflow documentation (DFT→density→potential→relax→df) |

### Run

```bash
pytest tests/SPM/test_afm_morse.py -v
pytest tests/SPM/test_afm_morse.py -v -m "not visual"  # non-visual only
```

### TODO

- [ ] Run tests and verify they pass
- [ ] Review visual output in `debug/`
- [ ] Update `TEST_RESULTS.md`

---

## Phase 2: FDBM — 🚧 Code Exists, Tests Needed

### SPAMMM Already Has Full FDBM Pipeline

**This is the key correction:** SPAMMM already has a complete FDBM implementation. The task is to write tests for it, not to build it.

### Existing FDBM Code in SPAMMM

#### `spammm/SPM/AFM.py` — Physics Functions

| Function | Lines | What It Does |
|----------|-------|-------------|
| `fft_poisson(rho, step)` | 1158-1168 | FFT Poisson solver: V from charge density |
| `build_gaussian_tip(shape, step, sigma)` | 1170-1182 | Normalized Gaussian tip density kernel |
| `get_tip_kernel(rho_t)` | 1189-1196 | Reverse + roll tip density for FFT correlation |
| `get_pauli_convolution(rho_s, rho_t, dV, A, beta)` | 1198-1212 | Pauli repulsion via FFT convolution |
| `compute_pauli_overlap(rho_grid, rho_tip, step)` | 1583-1595 | Raw density overlap via FFT cross-correlation |
| `scale_pauli_field(overlap, step, A, beta)` | 1598-1616 | Scale overlap → energy: `E = A * overlap^beta` |
| `compute_es_conv_field(V_ES, rho_tip_delta, step)` | 1629-1654 | ES energy via convolution of V with tip delta-rho |
| `compute_vdw_field(atomPos, atomTypes, ...)` | 1656-1684 | C6/r^6 dispersion on grid with regularization |
| `compute_dispersion_grid(...)` | 1768+ | C6/r^6 dispersion (OpenCL or CPU) |
| `compute_fdbm_forcefield(...)` | 1361-1462 | **Full FDBM**: Pauli + ES + vdW → scan grid forces |
| `compute_df(Fz, dz)` | 1154-1156 | Frequency shift: `-dFz/dz` |
| `project_single_atom(Z, rho_4x4, step, ...)` | 1721-1760 | Single-atom density projection test helper |
| `run_fireball_scf(xyz_path, fdata_dir, nscf)` | 1219-1267 | Fireball SCF → sparse density matrix |
| `setup_density_grid(atomPos, step, ...)` | 1270-1290 | Grid spec for density projection |
| `build_neutral_atom_rho(atomTypes, neighs, natoms)` | 1302-1315 | Neutral-atom on-site density matrix |
| `project_density_grids(rho_sparse, rho_na, ...)` | 1318-1354 | Project SCF + NA densities → rho_grid, rho_na, rho_diff |

#### `spammm/SPM/AFM.py` — AFMulator Class (FDBM methods)

| Method | What It Does |
|--------|-------------|
| `setup_fdbm_grid(F_total, origin, step)` | Upload precomputed FF 3D image to GPU |
| `scan_fdbm(scan_xs, scan_ys, probe_heights, ...)` | GPU PP relaxation over scan grid using FDBM image |
| `scan_fdbm_2d(scan_xs, scan_ys, probe_heights, ...)` | 2D lateral-only relaxation (relaxStrokes2D kernel) |
| `compute_gradient_cl(E_field, step)` | GPU gradient of scalar field (central differences) |
| `compute_dispersion_grid_cl(...)` | GPU C6/r^6 dispersion energy grid |

#### `spammm/SPM/ModularPipeline.py` — ModularAFMPipeline

Full staged pipeline with caching. Supports DFTB and pySCF backends.

| Stage | Method | What It Does |
|-------|--------|-------------|
| S1 | `stage1_scf()` | DFTBcore SCF or pySCF SCF → density matrix, eigvecs |
| S2 | `stage2_project(dm_dense)` | GPU density projection → rho_scf, rho_na, rho_diff |
| S3 | `stage3_potentials(rho_scf, rho_na, rho_diff)` | Poisson + Pauli + ES + vdW → F_total (GPU gradient) |
| S4 | `stage4_relax(F_total)` | PP relaxation → df, tip_disp, FEs_relax |
| S5 | `stage5_stm(eigvecs, eigvals)` | Standard STM projection on height slices |
| S6 | `stage6_br_stm(eigvecs, eigvals, tip_disp)` | Bond-resolved STM at relaxed tip positions |

#### `spammm/SPM/AFM_utils.py` — Orchestration

| Function | What It Does |
|--------|-------------|
| `get_density_from_dftb_dense(...)` | DFTBcore SCF + dense matrix projection → rho grids |
| `get_density_from_pyscf(...)` | pySCF SCF + density on grid |
| `compose_and_relax_total(F_total, ...)` | Interpolate F_total to scan grid + PP relaxation → df |
| `compute_stm(...)` | STM orbital projection |
| `compute_bond_resolved_stm(...)` | BR-STM at relaxed tip positions |
| `_get_cached_co_tip(...)` | CO tip density caching |
| `_compute_co_tip_grid(...)` | CO tip grid spec |

#### `spammm/quantum/DFTB/Grid_dftb.py` — GPU Projection

| Function | What It Does |
|--------|-------------|
| `GridProjector` | GPU density projection from sparse/dense matrix |
| `setup_gridprojector_from_dftb(...)` | Setup projector from DFTB+ data |
| `project_neutral_density(...)` | Project neutral atom density |

### FDBM Physics Model

```
E_total(R) = E_Pauli(R) + E_ES(R) + E_vdW(R)

E_Pauli(R) = A_pauli * [overlap(R)]^beta_pauli
  where overlap(R) = dV * IFFT[FFT(ρ_sample) * conj(FFT(ρ_tip))]

E_ES(R) = dV * IFFT[FFT(V_ES) * FFT(tip_delta_kernel)]
  where V_ES = Poisson(Δρ),  Δρ = ρ_SCF - ρ_NA

E_vdW(R) = -Σ_i sqrt(C6_i * C6_CO) / (|R - r_i|² + RA²)³
  where RA = 1.5 Å (regularization)
```

**Fitted Pauli parameters** (`PAULI_FITTED_DEFAULTS` in AFM.py):
- `mio-1-1`: A=787.22, beta=1.2371
- `3ob-3-1`: A=509.28, beta=1.0586
- `pyscf_sto-3g`: A=1.15, beta=0.36

### FireCore Reference for FDBM

| File | Use |
|------|-----|
| `tests/tAFM/test_fdbm.py` | **Primary reference** — full FDBM with Fireball: SCF, delta-rho, Poisson, Pauli, vdW, relax, df, plots |
| `tests/tAFM/pyocl_fdbm/run_pyocl_fdbm.py` | **Latest PyOpenCL** — step-by-step validation, debug outputs per step, ~1300 lines |
| `tests/tAFM/pyocl_fdbm/compute_co_tip.py` | CO tip density via Fireball subprocess |
| `tests/tAFM/pyocl_fdbm/run_fitted_afm.py` | Run with fitted Pauli parameters |
| `tests/tAFM/pyocl_fdbm/test_full_pipeline.py` | End-to-end with DFTB+ + STM |
| `doc/Topics/AFM/AFM.md` | 5-step workflow: DFT→density→potential→relax→df |
| `doc/Topics/AFM/AFM_migration_plan.md` | Physics corrections, debugging policy, known issues |
| `doc/Topics/AFM/AFM_FDBM_DFTB.chat.md` | DFTB integration discussion (replacing Fireball) |
| `doc/Topics/AFM/AFM_FDBM_fitting.chat.md` | Fitting Pauli/LJ/ES to DFTB z-scan reference |
| `doc/Topics/FDBM_Fit/FDBM_fit.md` | Linear least-squares fitting methodology |

### Test Plan for Phase 2

**File:** `tests/SPM/test_afm_fdbm.py` (to be created)

**Strategy:** Test each physics function independently, then test the full pipeline. Use synthetic data where possible to avoid DFTB dependency in unit tests. Use DFTB only for integration tests (marked `gpu` + `slow`).

#### Tier 1: Unit Tests (No DFTB, No GPU Required)

These test individual physics functions with synthetic inputs. Fast, no external dependencies.

| Test | Function | Input | Checks |
|------|----------|-------|-------|
| `test_fft_poisson_point_charge` | `fft_poisson` | Delta-function rho on grid | V ~ 1/r decay, V finite, Laplacian matches |
| `test_fft_poisson_dipole` | `fft_poisson` | +q at z=0, -q at z=d | V has dipole pattern, sign correct |
| `test_build_gaussian_tip` | `build_gaussian_tip` | shape, step, sigma=0.7 | Integral ≈ 1.0, symmetric, peak at center |
| `test_get_tip_kernel` | `get_tip_kernel` | Simple 3x3x3 density | Reversed, rolled to (0,0,0) |
| `test_pauli_overlap_gaussian` | `compute_pauli_overlap` | Two identical Gaussians | Overlap peaks at R=0, decays with |R| |
| `test_scale_pauli_field` | `scale_pauli_field` | Known overlap field | E = A * overlap^beta, gradient finite |
| `test_es_conv_field` | `compute_es_conv_field` | V_ES = 1/r, tip = delta | E_ES finite, sign correct |
| `test_vdw_field_single_atom` | `compute_vdw_field` | One C atom at origin | E_vdw < 0 (attractive), ~1/r^6 decay, no divergence |
| `test_vdw_field_regularization` | `compute_vdw_field` | Tip at r=0 (on top of atom) | E finite (no NaN/inf), RA=1.5 prevents blow-up |
| `test_compute_df_sign` | `compute_df` | Fz = linear in z | df = -dFz/dz has correct sign |
| `test_compute_df_decreasing_z` | `compute_df` | Fz with decreasing z-coords | Correct sign even when dz < 0 |

#### Tier 2: Integration Tests (GPU, No DFTB)

These test GPU kernels and AFMulator FDBM methods with synthetic force fields.

| Test | Function | Input | Checks |
|------|----------|-------|-------|
| `test_compute_gradient_cl` | `AFMulator.compute_gradient_cl` | E = known analytic field | Gradient matches analytic, finite |
| `test_setup_fdbm_grid` | `AFMulator.setup_fdbm_grid` | Synthetic F_total (nx,ny,nz,4) | GPU image allocated, no error |
| `test_scan_fdbm_synthetic` | `AFMulator.scan_fdbm` | Simple repulsive field | FEs finite, PP displacement reasonable |
| `test_scan_fdbm_2d_synthetic` | `AFMulator.scan_fdbm_2d` | Simple repulsive field | FEs finite, lateral relaxation works |
| `test_dispersion_grid_cl` | `AFMulator.compute_dispersion_grid_cl` | Single atom | E_vdw < 0, finite, matches CPU version |

#### Tier 3: Full Pipeline Tests (GPU + DFTB, marked `slow`)

These exercise the complete `ModularAFMPipeline` end-to-end.

| Test | Molecule | Stages | Checks |
|------|----------|--------|-------|
| `test_fdbm_pipeline_benzene` | benzene.xyz | S1-S4 | df finite, AFM contrast visible, df range reasonable |
| `test_fdbm_pipeline_pentacene` | pentacene.xyz | S1-S4 | df finite, ring/bond features visible |
| `test_fdbm_vs_morse` | benzene.xyz | S1-S4 + Phase 1 | FDBM df correlated with Morse df but different |
| `test_fdbm_stage_caching` | CO.xyz | S1→S2→S3→S4 with cache | Each stage loads from previous cache, results identical |

#### Tier 4: Visual Tests (marked `visual`)

| Test | What It Produces |
|------|-----------------|
| `test_visual_fdbm_components` | 2D slices of E_Pauli, E_ES, E_vdW, E_total at z=3Å |
| `test_visual_fdbm_df_maps` | df maps at 3-5 z heights for benzene |
| `test_visual_fdbm_vs_morse` | Side-by-side df comparison: FDBM vs Morse |
| `test_visual_fdbm_fz_curve` | Fz(z) curve at center of benzene ring vs over C atom |

### Known Issues to Test For

**From FireCore `AFM_migration_plan.md` §4.5.7:**

1. **vdW divergence** — Pure C6/r^6 blows up at z < 2 Å. SPAMMM uses RA=1.5 Å regularization. Test: verify `compute_vdw_field` is finite at r=0.
2. **Force sign convention** — Store ∇E consistently, compute F = -∇E only during composition. Test: verify `compute_fdbm_forcefield` returns F = -∇E.
3. **`np.gradient` with decreasing z** — `compute_df` uses `abs(dz)` to handle this. Test: verify sign with both increasing and decreasing z.
4. **Charge conservation** — ∫Δρ dV ≈ 0. Test: verify `project_density_grids` output integrates to ~0 (tolerance < 0.5 e).
5. **Grid alignment for convolution** — Tip O-atom must be at grid origin for correct convolution shift. Test: verify Pauli overlap peaks at correct position.

### Debug Output Directory

```
tests/SPM/debug_fdbm/
├── step1_density/     # rho_scf, rho_na, rho_diff slices + integrals
├── step2_poisson/     # V_ES slices, Laplacian check
├── step3_pauli/       # E_Pauli slices, overlap vs distance
├── step4_es/          # E_ES slices, comparison with point charge
├── step5_vdw/         # E_vdw slices, 1/r^6 check
└── step6_composed/    # F_total, df maps, Fz(z) curves
```

### Validation Reference Values (Pentacene, from FireCore)

| Component | Fz Range [eV/Å] | Notes |
|---|---|---|
| Pauli | [0.0, 2.24] | Repulsive, dominant at z < 3.2 Å |
| vdW | [-0.36, -0.005] | Attractive, dominant at z > 3.5 Å |
| ES | [-0.008, 0.007] | Negligible with q_CO=-0.05 |
| Total (raw) | [-0.037, 1.88] | Balance point ~3.2-3.5 Å |
| Total (relax) | [-0.037, 1.85] | PP relaxation has small effect |
| df | [-3.79, 0.02] | Mostly negative (attractive regime) |

---

## FireCore Documentation Map

### `doc/Topics/AFM/`

| File | Content | Relevance |
|------|---------|-----------|
| `AFM.md` | 5-step AFM workflow (DFT→density→potential→relax→df), code map, kernel mapping | **High** — best overview |
| `AFM_migration_plan.md` | PyOpenCL migration plan, physics corrections, debugging policy, known issues, validation results | **High** — read before FDBM tests |
| `AFM_FDBM_DFTB.chat.md` | Discussion: replacing Fireball with DFTB for density projection | Medium — DFTB integration reference |
| `AFM_FDBM_fitting.chat.md` | Fitting Pauli/LJ/ES to DFTB z-scan reference data | Medium — parameter fitting |
| `AFM_FDBM_optimization.chat.md` | Performance optimization discussion | Low |
| `AFM_FDBM_profiling_optimization.chat.md` | Profiling and optimization | Low |
| `AFM_FDBM_pySCF.chat.md` | pySCF backend integration | Medium — SPAMMM also has pySCF |
| `AFM_migration.progress.md` | Migration progress tracking | Low |
| `AFM_migration_discusion.chat.md` | General migration discussion | Low |
| `DFTB_Perturbation_Pauli.chat.md` | DFTB perturbation theory for Pauli | Low — advanced |
| `GPU_CPU_transfer_analysis.md` | GPU↔CPU transfer analysis | Low |
| `IndentationForce2D.chat.md` | Indentation force discussion | Low |
| `dense_projection_integration_plan.md` | Dense matrix projection plan | Medium |
| `AFM_fast_DFTB_model_GPT55.py` | Fast DFTB model script | Low — reference code |
| `AFM_fast_DFTB_model_Kimi.py` | Alternative fast DFTB model | Low |

### `doc/Topics/FDBM_Fit/`

| File | Content | Relevance |
|------|---------|-----------|
| `FDBM_fit.md` | Linear least-squares fitting of P_i, L_i, Q_i coefficients to DFTB reference | **High** — parameter fitting strategy |

### `doc/Topics/STM/`

| File | Content | Relevance |
|------|---------|-----------|
| `STM_GF_new.chat.md` | Green's function STM implementation (~108K) | Future — STM tests |
| `STM_GPU_QMMM.chat.md` | GPU QMMM STM (~179K) | Future — STM tests |

### `doc/topical_audit/`

| File | Content | Relevance |
|------|---------|-----------|
| `afm_stm_simulation.md` | Cross-language audit of all AFM/STM components | **High** — component map |
| `RigidSurfPotential_GridFF.md` | GridFF substrate potential audit | Medium — surface interactions |
| `surface_interactions.md` | Surface electrostatics overview | Medium |
| `forcefields_overview.md` | Force field overview | Low |

### `tests/tAFM/`

| File | Status | Use |
|------|--------|-----|
| `afm_morse_pbc.py` | **Latest** | Phase 1 reference — Morse+PBC full pipeline |
| `test_ptcda.py` | Active | Phase 1 — simpler Morse/LJ on PTCDA |
| `test_fdbm.py` | **Latest** | Phase 2 reference — full FDBM with Fireball |
| `pyocl_fdbm/run_pyocl_fdbm.py` | **Latest** | Phase 2 — PyOpenCL FDBM with step-by-step validation |
| `pyocl_fdbm/compute_co_tip.py` | Active | CO tip density computation |
| `pyocl_fdbm/run_fitted_afm.py` | Active | Run with fitted Pauli parameters |
| `pyocl_fdbm/test_full_pipeline.py` | Active | End-to-end DFTB+ + STM |
| `test_single_atom.py` | Active | Single-atom density projection (FDBM prereq) |
| `test_gradient_simple.py` | Utility | GPU gradient kernel test |
| `test_gradient_kernel.py` | Utility | GPU gradient kernel test (detailed) |
| `test_gradient_visual.py` | Utility | GPU gradient visualization |
| `AGENTS.md` | Documentation | Test organization guide |

---

## Test Data

### Molecules (`data/xyz/`)

| File | Atoms | Use |
|------|-------|-----|
| `CO.xyz` | 2 | Fastest test, tip model |
| `H2O.xyz` | 3 | Polar molecule |
| `CH4.xyz` | 5 | Tetrahedral |
| `benzene.xyz` | 12 | Aromatic ring — good for visual AFM |
| `pentacene.xyz` | 36 | Large conjugated — AFM benchmark molecule |

### Substrates (`data/substrates/`)

| File | Ions | Use |
|------|------|-----|
| `NaCl_1x1_L2.xyz` | 4 | Minimal unit cell |
| `NaCl_1x1_L3.xyz` | 6 | 3 layers |
| `CaF2_3x3_6L.xyz` | 162 | Different material, larger |

### Parameters

| File | Content |
|------|---------|
| `data/ElementTypes.dat` | vdW radii, EvdW, QEq params |
| `data/AtomTypes.dat` | LJ/Morse parameters per atom type |

---

## Implementation Priority

### 1. Run Phase 1 tests (immediate)
```bash
pytest tests/SPM/test_afm_morse.py -v
```

### 2. Write Tier 1 unit tests (next)
No GPU, no DFTB — pure NumPy physics function tests. These are fast and catch sign errors, normalization bugs, and edge cases.

### 3. Write Tier 2 GPU integration tests
Test `AFMulator` FDBM methods with synthetic data. Requires GPU but not DFTB.

### 4. Write Tier 3 full pipeline tests (requires DFTB+)
Exercise `ModularAFMPipeline` S1-S4. Marked `slow` + `gpu`. These are the real validation.

### 5. Write Tier 4 visual tests
Generate comparison plots for human review.

### 6. Parameter fitting (future)
Fit Pauli A/beta to DFTB z-scan reference. See `doc/Topics/FDBM_Fit/FDBM_fit.md`.

### 7. STM tests (future)
Test `stage5_stm` and `stage6_br_stm`. See `doc/Topics/STM/`.

---

## Summary

| Phase | Code Status | Test Status |
|-------|-----------|-------------|
| Phase 1 (Morse+Coulomb) | ✅ Complete | ✅ Tests written, ⏳ need to run |
| Phase 2 (FDBM) | ✅ Complete (ModularAFMPipeline + physics functions) | ❌ No tests — 4 tiers planned |
| Parameter fitting | ✅ Methodology documented | ❌ Not started |
| STM | ✅ stage5/stage6 implemented | ❌ Not started |
