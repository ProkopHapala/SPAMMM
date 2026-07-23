# Task: Refactor large modules (analysis — no code yet)

**Status:** investigating / design discussion (**rev. 5** — USER 2026-07-23)  
**Priority:** Later — **do not implement until USER approves**  
**Trigger:** `spammm/SPM/AFM_utils.py` ~5400 L  
**Rule:** **NO CODE MOVES** until this plan is accepted.

---

## 1. Goals (rev. 5)

1. Navigable modules; maximize **reuse**, minimize duplication.  
2. **`Grid_util`** — FDBM-agnostic field algebra **including cube file I/O** (read/write/resample cube ↔ arrays).  
3. **`SPM_plot`** — shared plotting (Morse AFM, FDBM, STM).  
4. **`SPM_debug`** — shared diagnostics (not in `tests/`).  
5. **`AFM_utils`** — basic Morse + Coulomb AFM only.  
6. **`fdbm.py`** — FDBM physics/pipeline only (Δρ recipes, tip, compose, run_*).  
7. **QM-engine density adapters → `spammm/quantum/`** (DFTB, pySCF, **Fireball**).  
8. STM / Pauli as separate SPM modules.

---

## 2. Target layout (rev. 5)

```
spammm/utils/Grid_util.py          # field ops + cube read/write/load-as-grid
spammm/SPM/SPM_plot.py             # shared AFM+STM(+FDBM) plotting SSOT
spammm/SPM/SPM_debug.py            # diagnostics / parity / stage dumps
spammm/SPM/AFM_utils.py            # slim: Morse+Coulomb AFM helpers
spammm/SPM/fdbm.py                 # FDBM only (no per-QM adapters)
spammm/SPM/stm_utils.py
spammm/SPM/pauli_fit.py
spammm/SPM/AFM.py                  # AFMulator (largely unchanged this round)

spammm/quantum/DFTB/…              # get_density_from_dftb*
spammm/quantum/pySCF_utils.py      # MERGED SSOT (today’s -new + legacy API); see §12
spammm/quantum/…                  # get_density_from_fireball (legacy Fireball SCF → grids)
```

**Contract:** QM adapters and cube loaders return a **common density/grid dict** (ρ, origin, step, …). `fdbm.py` / ModularPipeline consume that + `Grid_util` — they do not know DFTB vs pySCF vs Fireball internals.

**Cube ≠ QM engine:** Gaussian/Psi4/pySCF `.cube` is a **grid file format**. I/O lives in **`Grid_util`**, not under `quantum/`.

---

## 3. Estimated sizes (post-split)

| Module | Est. lines | Notes |
|--------|----------:|-------|
| **`Grid_util.py`** | **~850–1400** | Field algebra **+ cube I/O** (`get_density_from_cube` / read_cube / write_cube) |
| **`SPM_plot.py`** | **~700–1000** | E/Fz/df + FDBM z-diag + STM map panels |
| **`SPM_debug.py`** | **~450–600** | Interp parity, step dumps, tip/cube diag figures |
| **`AFM_utils.py`** | **~250–500** | Morse+Coulomb helpers only |
| **`fdbm.py`** | **~800–1100** | FDBM core; adapters in QM / cubes in Grid_util |
| **`stm_utils.py`** | **~400–550** | STM compute; plots via `SPM_plot` |
| **`pauli_fit.py`** | **~850–950** | Fitting campaigns |
| **QM adapters (DFTB/pySCF/Fireball)** | **~400–550** total | Engine-specific only (§6) |

---

## 4. `Grid_util.py`

**Path:** `spammm/utils/Grid_util.py`.

**Field algebra:** crop, line/z sample, resample, pad/extend, roll, axpy, origin/step helpers; optional moments; optional `GridsOCL` backend.

**Cube files (USER):** read/write Gaussian-style cubes; load cube → `(data, origin, step, atoms?)`; optional `get_density_from_cube`-style helper that builds the common density dict (ρ_scf from Dt.cube, attach NA via callable or simple recipes). May reuse/wrap existing parsers in `DFTBplusParser.read_cube` / `DFTB_utils.read_cube*` by **moving or thin-wrapping** into Grid_util so SPM/FDBM don’t import DFTB just to read a cube.

**Not here:** DFTB/pySCF/Fireball SCF.

---

## 5. `SPM_plot` / `SPM_debug` / slim `AFM_utils`

Same as rev. 3:

- **`SPM_plot`** — skill:`afm-plotting` SSOT target.  
- **`SPM_debug`** — shared diagnostics in `spammm/SPM/`.  
- **`AFM_utils`** — Morse+Coulomb non-plot glue only.

---

## 6. QM engines → `quantum/` ; cubes → `Grid_util` ; slim `fdbm.py`

### Engine adapters → `quantum/` (USER-confirmed)

| Function (today in `AFM_utils`) | ~L | Destination |
|--------------------------------|----:|-------------|
| `get_density_from_dftb_dense` | ~185 | `spammm/quantum/DFTB/` |
| `get_density_from_dftb` / `_dftb_plus` | ~70 | same |
| `get_density_from_pyscf` | ~196 | **merged** `spammm/quantum/pySCF_utils.py` (see §12) |
| `get_density_from_fireball` | ~43 | `spammm/quantum/` (Fireball SCF → grids; **yes, quantum**) |
| `build_orbital_layout`, `_project_densities` | ~30 | DFTB projector side |

### Cube → `Grid_util` (USER-confirmed, rev. 5)

| Function | ~L | Destination |
|----------|----:|-------------|
| `get_density_from_cube` | ~152 | **`spammm/utils/Grid_util.py`** (cube format ↔ grids; NA attach may call `fdbm` recipes or take NA callable) |
| related `read_cube` / `write_cube` helpers | various | consolidate into **`Grid_util`** (SSOT for cube I/O) |

Cube is a **grid file format**, not a QM code. `fdbm.build_fdbm_grid_from_cubes` orchestrates: `Grid_util` load cubes → tip → FDBM fields.

### What stays in `fdbm.py` (~800–1100 L)

| Block | Est. L | Role |
|-------|------:|------|
| Δρ / NA recipes | ~200–250 | FDBM ES science; callable from cube loader / QM adapters |
| `build_fdbm_grid_from_cubes` (orchestration) | ~80–120 | `Grid_util` cubes + tip + compose |
| CO tip cache / `get_tip_densities` | ~150–200 | pad/roll via `Grid_util` |
| `compose_and_relax*`, `run_afm_*` | ~450–550 | Pipeline |
| Optional backend dispatch | ~30–50 | imports QM adapters / `Grid_util` cube |

**Not in `fdbm.py`:** matplotlib → `SPM_plot`; debug dumps → `SPM_debug`; DFTB/pySCF/Fireball SCF → `quantum/`; cube parse → `Grid_util`.

### Δρ / NA ownership

**Default:** recipes in **`fdbm.py`**; `Grid_util.get_density_from_cube` may accept `na_builder=...` or call into fdbm NA helpers to avoid circular imports (prefer: Grid_util loads raw ρ; `fdbm` or caller attaches NA/Δρ).

### Size arithmetic

```
fdbm.py           ~800–1100   FDBM core only
quantum/          ~400–550    DFTB + pySCF + Fireball adapters
Grid_util         ~850–1400   field ops + cube I/O (~+150 from cube)
```

---

## 7. STM / Pauli

Unchanged estimates: `stm_utils` ~400–550; `pauli_fit` ~850–950. Pauli z-scan runners that invoke DFTB/pySCF should import from **`quantum/`**, not embed SCF.

---

## 8. Size summary (rev. 5)

```
Grid_util      850–1400     ← field ops + cube I/O
SPM_plot       700–1000
SPM_debug      450–600
AFM_utils      250–500
fdbm.py        800–1100
stm_utils      400–550
pauli_fit      850–950
quantum/*      400–550      ← DFTB + pySCF + Fireball only
```

---

## 9. Migration order (when approved)

1. `Grid_util` (field ops **+ cube I/O**, including move of `get_density_from_cube`)  
2. `SPM_plot` + skill update  
3. `SPM_debug`  
4. Move **engine adapters** into `quantum/` (DFTB / pySCF / **Fireball**)  
5. `fdbm.py` — Δρ + tip + pipeline only  
6. `stm_utils`, `pauli_fit`  
7. Trim `AFM_utils`  

---

## 10. Discussion leftovers

1. Exact DFTB file name: new `density_dftb.py` vs methods on `Grid_dftb`?  
2. Fireball: keep minimal adapter vs deprecate?  
3. If `AFM_utils` <~200 L after extract — keep vs fold into `AFM.py`?  
4. Avoid cycles: raw cube load in `Grid_util`, NA/Δρ attach in `fdbm` caller?  
5. **pySCF merge** — plan in §12 (approved direction: one `pySCF_utils.py`).

---

## 11. Other large modules / acceptance

Unchanged. No code until USER says go.

---

## 12. pySCF: `pySCF_utils.py` vs `pySCF_utils-new.py` → merge plan

### What they are (investigation)

| | **`pySCF_utils.py`** (~91 L) | **`pySCF_utils-new.py`** (~532 L) |
|--|------------------------------|-----------------------------------|
| Role | Legacy thin helper | **Production SSOT** (header + docs agree) |
| pySCF install | Stock / whatever is on `PYTHONPATH` (`import pyscf`) | Prefers **local fork** `SPAMMM_PYSCF_ROOT` or `/home/prokop/git/pyscf`; else stock |
| Backends | Implicit stock RHF/UHF only | Explicit `backend=gpu\|smalldft\|cpu` via `make_rks` / `resolve_backend` |
| GPU | No | Yes — `pyscf.OpenCL` + NVIDIA (`apply_gpu_profile`) |
| smallDFT | No | Yes — `pyscf.smallDFT` (CPU OpenMP XC; needs `libsmalldft.so`) |
| API | `pack_mol`, `unpack_mol`, `preparemol`, `evalHf`, `optHf` | **Superset** of those + `make_rks`, `run_scf*`, `run_co_zscan`, `make_z_grid`, frontier MO cubes/slices, STM tip orbitals |
| Density on grid | Header claims it; **not implemented** | MO/ψ grids yes; total-ρ FDBM adapter still in `AFM_utils.get_density_from_pyscf` |
| Import | Normal package import | **Hyphen** → callers use `importlib` (painful) |
| Callers | GUI `ExtensionManager` lists module; little real use | `run_zscan_reference.py`, `compute_densities.py`, STM testplots, reports |

**Conclusion:** Not “two parallel equal backends.” They are **legacy stub vs current workhorse**. The workhorse already supports **stock CPU + your modified fork (GPU OpenCL + smallDFT)** in one module via `backend=`. Legacy file is a subset (and its `from . import atomicUtils` is wrong — `atomicUtils` lives in `spammm`).

### Merge strategy (when implementing)

1. **Canonical name:** `spammm/quantum/pySCF_utils.py` (no hyphen — normal imports).  
2. **Body:** move content of `pySCF_utils-new.py` into that path (overwrite legacy).  
3. **Keep** `resolve_backend('auto'|'gpu'|'smalldft'|'cpu')` and `ensure_local_pyscf()` — single entry for fork vs stock.  
4. **Preserve** legacy symbols (`evalHf`, `preparemol`, `optHf`, pack/unpack) so old callers keep working.  
5. **Compat shim (short-lived):** either  
   - delete `-new` and fix all `importlib` call sites to `from spammm.quantum import pySCF_utils`, or  
   - leave `pySCF_utils-new.py` as 5-line re-export of `pySCF_utils` for one transition. Prefer delete after grep-clean.  
6. **During large-module refactor:** land `get_density_from_pyscf` here (from `AFM_utils`), using `make_rks` + numint ρ on `Grid_util` lattice.  
7. **Docs:** update `doc/AGENTS/notes/pyscf-gpu-scf.md`, `ARCHITECTURE_ROADMAP` §2, `quantum/README.md`, reports that cite `-new`.  
8. **Do not** maintain two code paths for “official vs modified” as separate files — modification is a **backend flag + sys.path fork**, not a second module.

### Acceptance for pySCF merge

- [ ] `from spammm.quantum.pySCF_utils import run_co_zscan, make_rks` works  
- [ ] `backend='cpu'` runs without local fork; `auto`/`gpu` use fork when NVIDIA OpenCL visible  
- [ ] No remaining hyphen imports in tests/scripts  
- [ ] USER OK before deleting `-new`

---

*Rev. 5 + §12: Fireball → quantum; cube → Grid_util; pySCF merge = `-new` becomes `pySCF_utils.py`.*
