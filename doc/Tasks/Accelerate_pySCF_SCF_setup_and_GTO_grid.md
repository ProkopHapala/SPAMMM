# Accelerate pySCF for FDBM — SCF setup + GTO density-on-grid

**Goal:** Make DFT (pySCF) density generation for FDBM AFM competitive enough that GPU SCF wins are not eaten by setup + ρ(r) projection. Target: ρ-on-grid cost in the same ballpark as DFTB `project_density_dense` (tiled LCAO), and GPU SCF **setup ≪ one SCF cycle** (or clearly amortized across many geometry steps).

**Status:** investigating (benchmarks 2026-07-28; no fix claimed).  
**Host:** RTX 3090. **Code SSOT:** `spammm/quantum/pySCF_utils-new.py` + local fork `/home/prokop/git/pyscf`.  
**Related:** `doc/AGENTS/notes/pyscf-gpu-scf.md`, `doc/Tasks/PerfBenchmark_FDBM.md`, `kernels/LCAO_grid.cl`, fork `pyscf/OpenCL/` + `expamples_prokop/`.

---

## Timing vocabulary (do not confuse)

| Name | What it is | When paid | Bench column (2026-07-28) |
|------|------------|-----------|---------------------------|
| **Outer geometry step** | One nuclear geometry (MD step, opt step, CO z-point, …) | Once per geometry | not in table (single-geometry run) |
| **SCF job / run** | Build MF + one `mf.kernel()` to convergence at fixed geometry | Once per geometry (or warm-started reuse) | `setup` + `SCF` |
| **SCF setup** | Before first cycle: mol/DF build, XC grid plan, OpenCL buffers, `_cderi`, … | Once per job (ideally reused across geom if DF/plan allow) | **`setup`** |
| **SCF loop (total)** | All DIIS/SCF **cycles** inside `mf.kernel()` | Once per job | **`SCF`** (= total, **not** per cycle) |
| **SCF cycle** | One iteration inside the SCF loop (J/K/XC + DIIS update) | `n_cycles` times per job | `SCF / n_cycles` |
| **ρ projection** | Density matrix → ρ(r) on FDBM real-space grid | Once per geometry after SCF (Pauli/ES) | **`ρ_proj`** |
| **Disk I/O** | Writing `.npy` / `.cube` | optional | **excluded** from these benches |

**Outer vs inner (optimization / MD):**

```
for geom_step in optimization_or_MD:          # OUTER loop
    setup_or_reuse_mf(geom)                  # setup (amortize!)
    for scf_cycle in 1..n_cycles:            # INNER SCF loop
        update_density_matrices...
    rho = project_DM_to_grid(...)            # ρ_proj (today: slow for GTO)
    forces_or_FDBM(...)
```

Quote numbers with the row label above. Saying “SCF takes 4 s” without “total vs per cycle” is ambiguous.

---

## Benchmark snapshot (2026-07-28)

**Protocol:** single geometry; no ρ disk save; grid `step=0.1 Å`, `margin=5`, `z_extra=6`.  
pySCF: **PBE / def2-SVP**, DF, backends GPU (`production_radial_screened_splitk`) vs stock CPU.  
DFTB: **3ob-3-1**, `project_density_dense` only (no NA).  
`SCF` = wall of `mf.kernel()` / DFTB `run_scf` (+DFTB projector/HSD setup folded into DFTB SCF column).  
`setup` = `make_rks` + `apply_gpu_profile` (GPU) only.

| mol | method | N | nao | cycles | **setup** [s] | **SCF total** [s] | **≈ / cycle** [s] | **ρ_proj** [s] | total [s] |
|-----|--------|---|-----|--------|---------------|-------------------|-------------------|----------------|-----------|
| pentacene | DFTB 3ob | 36 | 102 | — | — | **0.16** | — | **0.02** | 0.18 |
| pentacene | pySCF GPU | 36 | 378 | 12 | **6.91** | **1.73** | **0.14** | **17.52** | 26.2 |
| pentacene | pySCF CPU | 36 | 378 | 13 | ~0 | **55.34** | **4.3** | **18.18** | 73.5 |
| PTCDA | DFTB 3ob | 38 | 128 | — | — | **0.15** | — | **0.02** | 0.17 |
| PTCDA | pySCF GPU | 38 | 460 | 19 | **10.11** | **3.88** | **0.20** | **24.38** | 38.4 |
| PTCDA | pySCF CPU | 38 | 460 | 17 | ~0 | **106.49** | **6.3** | **23.87** | 130.4 |

### Headline ratios (use carefully)

| Comparison | Metric | pentacene | PTCDA |
|------------|--------|-----------|-------|
| GPU vs CPU pySCF | **SCF total** | **32×** | **27×** |
| GPU vs CPU pySCF | **≈ per cycle** | ~30× | ~31× |
| GPU vs DFTB | SCF total | ~11× slower | ~26× slower |
| ρ_proj GPU-path vs DFTB | same grid | ~1000× slower | ~1000× slower |
| ρ_proj GPU vs CPU pySCF | both CPU `numint` | ~1× | ~1× |

**Interpretation:**

1. GPU acceleration of **XC/SCF cycles works** (~30× vs stock CPU on these mols).
2. **ρ(r) from DM dominates** the DFT→FDBM path today (~18–24 s ≫ SCF).
3. **GPU setup (7–10 s) ≫ one SCF cycle (0.14–0.20 s)** and even ≫ full SCF total on pentacene — **not acceptable** for single-point imaging; must be profiled and cut or amortized across many outer geometry steps.

**Caveat:** DFTB 3ob vs pySCF def2-SVP are **not** the same Hamiltonian / basis size (nao 100 vs 400). Speed vs DFTB is a product target, not a parity claim.

---

## Improvement 1 — Why is GPU SCF setup so expensive?

**USER expectation:** some overhead is fine; **setup should not exceed ~one SCF cycle** (~0.2 s here), unless it is clearly reused across many outer geometry steps.

**What setup does today** (`make_rks` → `apply_gpu_profile(..., setup=True)` in local `/home/prokop/git/pyscf`):

1. `mf.initialize_grids` — XC quadrature grid  
2. `setup_precomputed_gto` / `setup_xc_grid_gpu` — OpenCL XC plan, Hermite/radial tables, tiles  
3. `prepare_df_for_scf` — build **`_cderi`** (incore DF) + GPU DF-J buffers  

Suspicion (to verify with staged timers, not guess): **DF `_cderi` build** and/or **full XC plan rebuild** dominate; possible double work vs first SCF cycle; possible rebuild every geometry even when only nuclei moved slightly.

### Tasks

- [ ] Instrument `apply_gpu_profile` / `prepare_df_for_scf` / XC `setup_*` with **sub-timers** (grid init, AO/Hermite tables, DF build, GPU buffer upload). Print in one line per job.
- [ ] Compare setup vs **first SCF cycle** cost; list what is geometry-invariant vs geometry-dependent.
- [ ] For **outer geometry loops** (opt / MD / z-scan): design **reuse policy**
  - keep mol + DF basis auxiliaries when possible  
  - rebuild only displacement-dependent pieces  
  - document when `_cderi` **must** rebuild (fork note already warns DF rebuilds per geom)
- [ ] Target: **setup ≤ 1× mean SCF cycle** for fixed-geom single point after warm caches; for multi-geom, **setup / N_geom ≪ cycle**.
- [ ] Check we are not forcing expensive path (e.g. `df_storage='incore'` + `require_df_incore` building huge RAM tensors when unnecessary for AFM tol).

**Refs:** `pyscf/OpenCL/gpu_profiles.py` (`apply_gpu_profile`), `doc/df_storage_and_benchmark_hygiene.md` (fork), `spammm/quantum/pySCF_utils-new.py`.

---

## Improvement 2 — Project pySCF DM → ρ as fast as DFTB (tiled LCAO)

### Today (bottleneck)

`AFM_utils.get_density_from_pyscf` uses stock **CPU** `pyscf.dft.numint.eval_ao` + `eval_rho` in chunks.  
Even when SCF is GPU, **ρ_proj stays ~18–24 s** on ~6×10⁶ voxels × ~400 AOs.

This is **not** GridFF and **not** `LCAO_grid.cl` / `project_density_dense` (those are the **DFTB STO** path).

### Desired

Same methodology as DFTB density projection:

- Spatial **tiles / boxes** of grid points  
- Atoms (or shells) that contribute to a tile loaded to local memory  
- Screen far contributions (radial cutoffs)  
- Dense DM contraction on GPU  

def2-SVP is **double-ζ** (more functions per atom than 3ob STO). That raises:

- larger `nao` (~4× DFTB here)  
- more shells / primitives per atom  
- possibly need a **new GTO (or Hermite) kernel**, not a blind reuse of STO `LCAO_grid.cl`  

In principle still fast: cost scales with **active shells × tile points**, not global `npts × nao` if tiling + screening work.

### Assets already nearby

| Asset | Role |
|-------|------|
| `kernels/LCAO_grid.cl` + `Grid_dftb.project_density_dense` | DFTB STO tiled/dense projector (reference speed) |
| `/home/prokop/git/pyscf/pyscf/OpenCL/` Hermite AO / radial tables | GTO-on-grid for **XC** (not yet FDBM ρ provider) |
| Fork `expamples_prokop/test_opencl_xc_rho_precomp.py` etc. | ρ projection benches inside XC pipeline |

### Tasks

- [ ] Inventory: can XC Hermite/rho GPU path emit **full molecular ρ on a uniform FDBM grid** (Cartesian Å grid), or only XC quadrature grids?
- [ ] Spec FDBM GTO projector API: inputs `(mol, dm, origin, step, ngrid)` → `ρ` float32; parity vs `numint` on pentacene/PTCDA (∫ρ, RMSE).
- [ ] Design tile layout for **multi-ζ** (max shells/atom, contraction coeffs, cutoff by primitive exponent).
- [ ] Implement OpenCL kernel (new file or extend `LCAO_grid.cl` with GTO path); wire into `get_density_from_pyscf` when GPU available.
- [ ] Bench: **ρ_proj** target ≪ 1 s on PTCDA 0.1 Å grid (stretch: approach DFTB 0.02 s order-of-magnitude, accepting ~4× nao).
- [ ] Keep CPU `numint` as parity / fallback.

---

## Acceptance / reporting rules

When publishing new numbers, always state:

1. **setup** vs **SCF total** vs **SCF / cycle** vs **ρ_proj** vs **outer geom count**  
2. Basis / XC / grid / device  
3. Whether DF/XC plans were cold or warm  
4. Whether ρ path is CPU `numint` or GPU tiled GTO  

Do **not** mark this task done until USER confirms benches after the two improvements.

---

## Suggested order of work

1. **Setup profiling** (cheap, explains 7–10 s) → cut stupid work / amortize for z-scans.  
2. **GTO tiled ρ projector** (large win for FDBM AFM images from DFT).  
3. Re-run the table above + multi-geometry z-scan amortization demo (e.g. 10 CO heights, one setup).
