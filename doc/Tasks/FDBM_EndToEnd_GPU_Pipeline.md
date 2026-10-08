# Task: End-to-end GPU FDBM pipeline (no host round-trips, no duplicated work)

Status: implemented + parity-verified + **DEFAULT** — 2026-10-08 (`afm --pipeline gpu`
is the CLI default; `--pipeline legacy` = deprecated staged path kept for
debugging/back-compat; `run_fdbm_pp_from_density` emits `DeprecationWarning`;
GUI `ModularPipeline` shares one ctx: projector+AFMulator+FDBM FFT)
Motivation: invPPAFM dataset generation (`pme_dataset.py::fdbm_field`) runs this pipeline
per molecule; measured waste on PTCDA (0.15 Å grid, 192×160×168):

| stage | now | waste |
|---|---|---|
| `get_density_from_dftb_dense` stock | 0.41 s | computes ρ_scf (unused — only ρ_diff needed), V_ES Poisson skipped now |
| `get_density_from_dftb_dense` prolonged | 0.21 s | whole SCF skipped via `dm_in`; still re-does projector/basis/file I/O setup |
| `stage3_fdbm_fields_fast` | 0.74–1.18 s | downloads nothing needed, but **re-uploads ρ_scf/ρ_diff that just came off the GPU** |
| tip densities | ~0.3 s first | host-built, re-uploaded per molecule (shape changes per molecule!) |

## Waste inventory (measured, not guessed)

1. **Two separate OpenCL contexts.** `setup_gridprojector_from_dftb(dftb_data, proj_basis, …)`
   accepts `ctx`/`queue` but `get_density_from_dftb_dense` never passes them → GridProjector
   makes its own `OpenCLBase` context. Result: ρ computed on GPU in context A can only reach
   the FFT/AFMulator in context B via host. Every density field does GPU→CPU→GPU.
2. **Whole function run twice, half discarded.** Stock call's ρ_scf and prolonged call's
   ρ_na/ρ_diff/V_ES are all thrown away. (Already mitigated: `need_es=False, dm_in=res['dm']`
   skips the second SCF + neutral ρ + Poisson. Still pays projector setup, HSD/xyz/skf file
   I/O, basis parsing twice.)
3. **Host round-trip inside `project_density_dense`:** kernel writes `dproj_out_buff`,
   `enqueue_copy` → host, `× B3_FACTOR` on CPU, return np array; stage3 uploads it right back.
   B3_FACTOR is a scalar — should be folded into the projection kernel or applied on-device.
4. **ρ_diff computed on host** (`rho_scf - rho_na` in NumPy) though both operands sit on GPU
   moments earlier.
5. **Tip density rebuilt/re-uploaded per molecule** because `ngrid` differs per molecule
   (domain-fitted grids) → FFT plans, tip pad/roll, k-grids all re-created.
6. **clFFT plan creation ~1 s per unique grid shape** — amortized only if shapes repeat.
7. Scan itself reads `img_FF_fdbm` device-side (good); `F_total` host download only needed
   for host-side consumers (PME rasterization needs E on host anyway — legit, but could be
   async).

## Proposed structure — one context, three orthogonal functions

```
run_fdbm_pipeline(apos, Zs, grid_spec, basis_hsd, prol_ang, afmulator, work_dir)
├── dftb_scf(...)                    → dm_dense, eigvecs   (once; the ONLY DFTBcore call)
├── project_densities(device-resident):
│     projector_stock = GridProjector(ctx=afm.ctx, queue=afm.queue) + stock basis
│     projector_prol  = GridProjector(ctx=afm.ctx, queue=afm.queue) + prolonged basis
│     rho_scf_prol_cl = projector_prol.project_dm_cl(dm, grid_spec, out=fft.rho_scf_buf)
│     rho_scf_cl      = projector_stock.project_dm_cl(dm, grid_spec)          # on-device
│     rho_na_cl       = projector_stock.project_dm_cl(dm_na, grid_spec)       # on-device
│     rho_diff_cl     = rho_scf_cl − rho_na_cl                                # elementwise kernel
│     charge check    = device reduction, tiny download (4 floats)
├── stage3_fdbm_fields_fast(..., rho_scf_cl=…, rho_diff_cl=…)   # upload only when np given
└── scan reads img_FF_fdbm on device; single F_total download iff host needs it
```

### Concrete API changes (minimal, backward-compatible)

1. `project_density_dense(..., out_cl=None, keep_device=False)`:
   - `keep_device=True` → return `(cl.Buffer|cl_array, B3 already applied)`; apply B3_FACTOR
     via one elementwise kernel (or fold into `_krn_project_density_dense` — add a `scale`
     arg; kernel already writes the buffer, zero extra passes).
   - `out_cl` → write into caller-provided buffer (e.g. `fft._xyz_rho_scf`), zero copies.
2. `get_density_from_dftb_dense` split into two focused helpers (keep old fn as wrapper):
   - `dftb_scf_dm(apos, Zs, basis_hsd, work_dir)` → `dm_dense, eigvecs, enames, dftb_data`
   - `project_density_grids(dftb_ctx_dict, grid_spec, bases, need=('pauli','diff'))` →
     dict of device buffers + small host stats (q_scf, q_diff for the charge check).
   - `ρ_diff` subtraction fused on device: `rho_diff_cl = rho_scf_cl − rho_na_cl` via a
     trivial elementwise kernel (or pyopencl array subtraction — same context so it's free).
3. `setup_gridprojector_from_dftb(..., ctx=afmulator.ctx, queue=afmulator.queue)` — already
   supported, just pass it. Both projectors share the AFMulator context.
4. `_FdbmFFT` per-shape caching: `ensure()` already caches by shape internally;
   additionally keep a small dict of FFT instances keyed by `(nx,ny,nz,step)` in a module
   LRU so re-visiting a shape is free. Tip densities keyed the same way.
5. `stage3_fdbm_fields_fast(..., rho_scf_cl=None, rho_diff_cl=None)` — accept device arrays;
   host arrays still allowed (upload path, for parity tests).

## Expected steady-state per-molecule cost (PTCDA-scale, 0.15 Å)

| piece | now | after |
|---|---|---|
| SCF | 0.11 s | 0.11 s |
| projector setup ×2 + basis parse | ~0.12 s | ~0.12 s (cacheable → ~0) |
| projections (ρ_scf stock, ρ_na, ρ_scf prol) | ~0.35 s | ~0.35 s on-device |
| Poisson V_ES | 0 s (fused ES uses ρ_diff) | 0 |
| ρ round-trips | ~2×0.4 GB host↔dev | 0 |
| tip build/roll | ~0.3 s | cached per shape → ~0 |
| stage3 fused | ~0.7 s | ~0.7 s |
| FFT plans | ~1.0 s | ~0 when shape repeats |
| **total** | ~1.9 s | **~0.6–1.0 s** |

Second-order wins (not in scope): canonical grid sizes across molecules (snap to a few
FFT-friendly shapes) → plan/tip caches hit every time; async download of F_total while
next molecule's SCF runs.

## Invariants / verification

- `rho_diff` integral ≈ 0 (charge check) — keep the device reduction.
- Parity vs current host path: same grids → `np.allclose` on ρ_scf/ρ_diff (device path
  vs downloaded path, atol 1e-6).
- Parity of final df: PTCDA NCC vs `scan_fdbm` must stay ≥0.99.
- Deprecated CPU path stays deprecated — this pipeline removes the *reason* to reach for it.

## Do NOT

- Do not merge the two bases into one call — dual basis is mandatory physics (prolonged ρ
  for Pauli only; stock Δρ for ES). See `get_density_from_dftb_dense` docstring DUAL BASIS.
- Do not normalize prolonged ρ — ∫ρ_prol ≉ N_e by design; A,β absorb the scale.
- Do not drop `rho_na` projection — ρ_diff IS ρ_scf − ρ_na, it is the ES charge.
- Do not keep host copies of ρ fields in the device path (defeats the purpose).
- Do not modify the legacy orchestration (`get_density_from_dftb_dense`,
  `run_fdbm_pp_from_density`, `ModularPipeline`) — the new pipeline lives alongside;
  legacy stays functional for back-compat/debugging until USER confirms parity.

---

## Code-review corrections to the waste inventory (2026-10-07)

Measured `run_spm.py afm --xyz benzene.xyz` = **2.73 s wall** (12 atoms): ~1.0 s density
stage, ~0.32 s PP stage, ~0.4 s plotting, ~0.6–0.7 s imports+parser+glue. Findings that
refine/extend the table above:

1. **`dm_in`/`need_es`/`need_ves` merged (4f1a77a) but no caller uses them.** Both
   `run_spm.py::cmd_afm` and `run_fukui_panel` still call `get_density_from_dftb_dense`
   twice with defaults → duplicate DFTBcore init+SCF and two unused `fft_poisson` V_ES
   (~0.48 s on benzene). Quick fix exists but is superseded by the new pipeline —
   do NOT patch legacy callers (see Do NOT).
2. **`GridProjector.__init__` still creates a throwaway context** even when `ctx`/`queue`
   are passed: `super().__init__(nloc=nloc)` runs `select_device` (platform enum ~70 ms)
   before lines 81–83 overwrite `self.ctx`. Fix = forward ctx/queue to super (1 line,
   backward-compatible).
3. **ρ_diff needs only ONE projection, not two + subtraction.** `project_density_dense`
   is linear in dm → project `(dm_scf − dm_na)` directly. `B3_FACTOR` also folds into dm
   on host (norb² floats, trivial). Eliminates ρ_scf_stock + ρ_na grids AND the host
   subtract/`astype` entirely — no elementwise kernel needed.
4. **Projector output and FFT scratch share layout.** `dproj_out_buff` and
   `_FDBMGpyFFT._xyz_*` are both (nx,ny,nz) C-order float32 → `out_buff` can point
   straight at `fft._xyz_rho_diff.data`/`_xyz_rho_scf.data`. Zero copies, zero transfers.
5. **ModularPipeline (GUI) has the same two-context bug** — `stage2` projector created
   without ctx (`ModularPipeline.py:263`). Out of scope here; note for a follow-up.
6. **`F_total` host download in `stage3_fdbm_fields_fast` is unconditional** but only
   needed for stage plots / host consumers. Gate behind `download_F`.
7. **Backward `scan_fdbm` (E_diss) is unconditional** in `run_fdbm_pp_from_density` —
   2× relax cost even when `ediss` not requested. Gate in new pipeline.
8. **`get_tip_densities` uses `pad_mode='cpu'`** in `run_fdbm_pp_from_density` — the R2
   GPU pad/roll (`pad_mode='none'` + `tip_already_rolled=False`) exists but is unused
   there. New pipeline uses it.
9. `dm_na` comes from `dg.build_na_dm_diagonal` (already exists — reuse, don't
   re-project neutral atoms).
10. `build_tasks_selected` already runs on GPU by default; tasks rebuilt per projection
    are identical when basis+atoms+grid match (dm_diff & ρ_scf_stock share them) —
    optional micro-cache later.
11. `compute_df_amp_dir` on host is fine — FEs are the output and must come home anyway.
12. Eager `from spammm.SPM import stm_compare` in `run_spm.py::build_parser` costs
    ~0.3–0.5 s on every CLI call (separate minor cleanup, not pipeline scope).

## Kernel verdict: no new kernels needed

| need | existing mechanism |
|---|---|
| ρ output into caller buffer | add `out_cl` arg to `project_density_dense` (kernel already writes a buffer arg) |
| B3_FACTOR scaling | fold into `dm` on host before upload (projection linear in dm) |
| ρ_diff = ρ_scf − ρ_na | project `(dm_scf − dm_na)` — one launch, no subtraction kernel |
| ρ_na (dm) | `dg.build_na_dm_diagonal` (exists) |
| tip pad/roll on device | `fdbm_pad_roll_f32` via `fft.pad_roll_to_cl` (exists) |
| charge check ∫ρ | `cl_array.sum` reduction (pyopencl built-in) or keep optional host check |
| fields/compose/gradient/scan | unchanged (`stage3_fdbm_fields_fast` device path, `scan_fdbm`) |

## Architecture: `spammm/SPM/FDBMPipeline.py` (new module, new class)

`FDBMPipeline` owns EVERYTHING GPU for one process lifetime — single context, single
queue, lazily-built kernels/buffers keyed by shape/basis:

```
class FDBMPipeline:
    self.afm          = AFMulator(use_morse=False, use_fire=True)   # the ONE ctx+queue+prg
    self.ffts         = {}   # (nx,ny,nz) -> _FDBMGpyFFT(ctx=afm.ctx, queue=afm.queue)
    self.projectors   = {}   # 'stock'|'prol' -> GridProjector(ctx=afm.ctx, queue=afm.queue)
    self._tip_raw     = {}   # (step, margin, tip_mode, backend) -> host raw CO grids
    self._tip_cl      = {}   # shape -> device pad_rolled tip buffers (per fft)

    set_geometry(atomPos, atomTypes, grid_spec)      # atoms_dict, tasks, grid plan
    scf()                                            # ONE DFTBcore call -> dm (host, small)
    project(variants=('diff','prol'), basis_ang=…):  # device buffers only
        # 'diff' : project (dm_scf − dm_na)·B3   -> fft._xyz_rho_diff  (stock projector)
        # 'prol' : project dm_scf·B3             -> fft._xyz_rho_scf   (prol  projector)
        # 'stock': project dm_scf·B3             -> extra buf          (stock projector; only 'both')
    fields()                                         # pauli+ES+vdW+grad -> img_FF_fdbm
    scan(scan_spec, need_ediss=True) -> ScanResult   # fwd (+bwd iff ediss) scan_fdbm
    run(...) -> ScanResult                           # orchestration; F_total download lazy
```

- FFT per-shape LRU: `_FDBMGpyFFT.ensure` already caches per instance; pipeline keeps a
  dict shape→instance so revisiting a shape costs zero plans/allocations (cap ~4 shapes
  to bound VRAM; each ≈ 6·complex64 + ~10·float32 grids).
- Two persistent projectors on the shared ctx (doc sketch) — simpler than basis-swap
  (`_set_projector_species_basis` exists if we ever want one-projector mode).
- `project_density_dense` gets `out_cl=` writing into `fft._xyz_*` buffers directly —
  ρ never leaves the device between projection and FFT.
- `F_total` downloaded only on demand (stage plots, PME raster, debug) — `download_F` flag.
- `ScanResult` built via existing `shared_postprocess` — same contract as legacy.

## Phases

### Phase A — plumbing (legacy untouched, additive only)

- [x] A1 `GridProjector.__init__`: forward `ctx`/`queue` to `OpenCLBase.__init__` (kill throwaway ctx).
- [x] A2 `project_density_dense(..., out_cl=None)`: `out_cl` writes to caller buffer,
      B3 folded into uploaded DM (linear in dm), no host download. Default path unchanged.
- [x] A3 `_FDBMGpyFFT` accept raw `cl.Buffer` inputs (`getattr(x, 'data', x)` in
      `_xyz_cl_to_fft`/`_fft_real_to_xyz_cl`/`flip3_cl`/`pad_roll_to_cl`).
- [x] A4 `stage3_fdbm_fields_fast(..., rho_scf_cl=, rho_diff_cl=, tip_tot_cl=,
      tip_del_cl=, download_F=True)` — device inputs skip `_xyz_host_to_cl`;
      `download_F=False` skips the unconditional `download_image_rgba_xyz`.
- [x] A5 backward scan gated by `need_ediss` in `FDBMPipeline.scan_and_post` only
      (legacy `run_fdbm_pp_from_density` untouched).
- [x] A6 `pytest tests/SPM/test_afm_fdbm.py -m "not slow"` — 15 passed, no behavior change.

### Phase B — `FDBMPipeline` module (`spammm/SPM/FDBMPipeline.py`)

- [x] B1 lazy resource owners: `ensure_afm` (one ctx+queue+program), `ensure_fft(shape)`
      dict of `_FDBMGpyFFT`, `ensure_projector(key)` dict of GridProjector on shared ctx,
      `tip_raw`/`ensure_tip` caches (raw CO tip once, GPU pad_roll per shape).
- [x] B2 `dftb_prep` — single DFTBcore init/SCF/get_dm/finalize per molecule + basis
      parsing/orbital layout (mirrors the SCF half of `get_density_from_dftb_dense`;
      legacy fn untouched).
- [x] B3 `project_densities` — `build_na_dm_diagonal` + fused `(dm_scf−dm_na)` projection
      into `fft._xyz_rho_diff.data`; `dm_scf` → `fft._xyz_rho_scf.data` (prol projector)
      or pipeline stock buffer; charge check via `cl_array.dot(·,1)` device reduction.
- [x] B4 `fields` + `scan_and_post` — device-resident S3 (no F_total download unless
      'stage' plot), `need_ediss` gate, ScanResult + legacy extras contract.
- [x] B5 `run(...)` → `{variant: ScanResult}` — same consumer contract as
      `run_fdbm_pp_from_density`.

### Phase C — wiring

- [x] C1 `run_spm.py afm --pipeline {legacy,gpu}` (default `gpu`) — same ScanSpec,
      same `plot_afm_variant_height_strip` output. `smiles-afm` routes through
      `cmd_afm` → new pipeline automatically.
- [x] C1b GUI `ModularPipeline._init_dftb_backend` — projector now shares the
      AFMulator ctx/queue (was a separate context); FAST_S3 stage-3 already default.
- [ ] C2 `run_fukui_panel` + `smiles-afm` + dataset generator (`pme_dataset.py::fdbm_field`)
      reuse ONE `FDBMPipeline` across molecules (the real amortization).
- [x] C3 `tests/SPM/testplot_fdbm_pipeline_parity.py` — per-stage timing + host RSS /
      GPU VRAM deltas + compare strips, azaindol/pentacene/PTCDA.

### Phase D — parity gate → deprecation

- [x] D1 L0 test `tests/SPM/test_fdbm_pipeline_v2.py`: shared-ctx assert, ρ allclose,
      df corr ≥ 0.998 (coarse step=0.2; 0.9996–0.9998 at production step) — 3 passed.
- [x] D2 compare strips `debug/testplot_fdbm_pipeline_parity/<mol>/compare.png`
      (azaindol/pentacene/PTCDA) — legacy|gpu|Δ rows visually identical → USER review.
- [x] D3 USER confirmed 2026-10-08: `--pipeline` default flipped to `gpu`;
      `DeprecationWarning` added to `run_fdbm_pp_from_density` recommending
      `FDBMPipeline` (function stays); `cmd_afm` legacy branch prints a visible
      deprecation note. CODEMAP / AFM_FDBM audit updated.

### Measured results (RTX 3090, 2026-10-08, all at step=0.1 Å)

Cold = first touch on a fresh `FDBMPipeline` (per-shape FFT plan bake, projector
setup, tip upload, image allocs). Warm = same molecule re-run on the SAME pipeline
instance → the batch steady-state (caches hot, only real work remains).

| mol      | grid          | legacy dens+pp | gpu cold | **gpu warm** | speedup (warm) | host RSS Δ (leg→gpu) | GPU VRAM Δ (leg→gpu) | df corr |
|----------|---------------|----------------|----------|--------------|----------------|----------------------|----------------------|---------|
| azaindol | 160×128×128   | 0.42 s         | 0.26 s   | **0.11 s**   | 4.0×           | 237→79 MB            | 784→454 MB           | 0.9996  |
| pentacene| 224×144×128   | 0.82 s         | 0.50 s   | **0.16 s**   | 5.1×           | 272→145 MB           | 872→568 MB           | 0.9997  |
| PTCDA    | 200×160×128   | 1.30 s         | 0.50 s   | **0.20 s**   | 6.6×           | 682→226 MB           | 1002→572 MB          | 0.9998  |

Cold-run residual = mostly per-shape first-touch: `ensure_fft` gpyFFT plan bake
~0.2 s + fields first-touch ~0.2 s (tip upload, dispersion/FE images, first
kernel execs). Warm proj = ~0.01 s (device-resident tasks — no task-list
round-trip), warm fields ~0.01 s. Steady-state per-molecule = prep (~0.05 s
DFTBcore SCF — Fortran) + real GPU work + FEs downloads.

### Batch reuse contract (data generation — IMPORTANT)

One `FDBMPipeline` instance is designed to be reused across **different
molecules** — different natoms, positions, density matrices are all per-call
inputs. Reuse is safe and is the intended dataset-generation pattern:

```python
pipe = FDBMPipeline()                     # ONE ctx/queue/program for the batch
for mol in molecules:
    grid_spec, origin, ngrid, step = make_fdbm_grid_com_zsym(mol.atomPos, step, margin, z_vac)
    res = pipe.run(mol.atomPos, mol.atomTypes, basis_hsd, work_dir_i,
                   grid_spec, origin, step, ngrid, ...)   # ~0.1–0.2 s warm
```

Cache keys: FFT plans/scratch per **grid shape**; tip pad/roll per
(shape, tip params); projectors per (basis kind, hsd path); species/basis
per hsd path. **Same ngrid+step → all caches hit even with a different
origin** (origin is passed per-call in grid_spec). A different shape is
still correct — it just creates one more plan/scratch set (VRAM grows per
unique shape — for database runs, snap grids to canonical sizes so caches
always hit). Projector keys include the hsd path since 2026-10-08 — mixing
bases within one pipeline is safe.

### Fields-only mode (FDBM → PME/contact encode, dataset labels)

`run_fields(...)` / `fields_for_fit(...)` end the pipeline at the potentials —
for the `doc/export_invAFM/fdbm_compression.md` contract (FDBM → separable
2.5D contact field ~23 KB/mol) and any other field-label fitting:

- `oracle=True` (default): composed `img_FF_fdbm` + `out['sample'] =
  afm.sample_fdbm` — GPU tricubic (E,F) sampling at arbitrary points. This
  is the query oracle for `fit_contact_field` (mesh nodes + core-fit points)
  — replaces `setup_fdbm_grid(F_total)` so the 74 MB F_total host download
  never happens.
- `oracle=False`: component device fields only (`E_pauli`, `E_ES` cl_arrays
  + `img_vdw` E in channel 3) — no compose image, no scan buffers, no FEs
  downloads, no df/diss numpy. Matches the channel design (separate
  Pauli/ES/vdW fits for tip-charge/prefactor augmentation).

**Compression test (2026-10-08, azaindol + PTCDA):** the oracle feeds the
fast ContactPME dense-mesh encode (`cpm_params_from_samples`, direct
`_prefilter_3d` — no CG). At h_mesh=0.20 Å: fit ~0.05 s/mol (vs ~112 s
separable CG — now deprecated for batch), ~2–3 MB/mol ≈ 110–120× vs dense,
df_corr ≥0.99, PME rescan ~0.02 s. Harness:
`tests/SPM/testplot_fdbm_fields_compress.py --method pme`; full table in
`doc/export_invAFM/fdbm_compression.md` §PME dense-mesh variant.

Parity breakdown: device ρ_diff/ρ_scf max|Δ| ≈ 1–3e-5 abs (fp32 rounding of the fused
(dm−dm_na) projection); FEs corr ≥0.9999; E_diss corr 0.991–0.995 (hysteresis is the
most noise-sensitive derived quantity). Single-shot numbers — batch reuse amortizes
projector/FFT/program setup further (the per-process constants are paid once).

### Second-order (not in scope, list for later)

- ~~ModularPipeline projector ctx sharing (`ModularPipeline.py:263`)~~ — DONE 2026-10-08:
  projector shares the AFMulator ctx/queue (one ctx per GUI pipeline).
- ~~Task-list GPU→CPU→GPU round-trip~~ — DONE 2026-10-08: `build_tasks_dev` keeps the
  compacted task list on the GPU; `project_density_dense(tasks_dev=...)` binds the
  gtask output buffers directly (no host argsort — ordering is load-balance only).
- ~~Redundant .skf copies + double parse_wfc_hsd in dftb_prep~~ — DONE 2026-10-08:
  Prefix points at absolute sk_dir (no copies); basis_data cached per hsd path.
- Snap per-molecule grids to canonical FFT-friendly sizes → plan/tip caches always hit
  even across different molecules.
- Wire `pme_dataset.py::fdbm_field` + `run_fukui_panel` to ONE FDBMPipeline → warm
  steady-state ~0.11–0.20 s/mol (the 4–6.6× case above).
- Task-list cache per (atoms, grid, Rcut) in projector.
- Async F_total / FEs download overlapped with next molecule's SCF.
- Lazy `stm_compare` import in `build_parser` (~0.4 s/CLI call).
