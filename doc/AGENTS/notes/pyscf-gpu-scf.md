# pySCF GPU SCF (local fork) — quick notes

**SSOT workhorse (today):** `spammm/quantum/pySCF_utils-new.py`  
**Merge target:** replace legacy `pySCF_utils.py` with this file’s body (drop hyphen) — `doc/Tasks/Refactor_LargeModules.md` §12. Until then load via `importlib`.
**CLI for CO z-scans:** `tests/SPM/run_zscan_reference.py`  
**Cookbook (fork):** `/home/prokop/git/pyscf/doc/opencl_gpu_paths_cookbook.md`  
**Examples:** `/home/prokop/git/pyscf/expamples_prokop/`  
**NVIDIA ICD rule:** `.cursor/rules/opencl-nvidia-gpu.mdc` (Shell must use unrestricted / `all`)

## When to use which path

| Path | Module | Use for |
|------|--------|---------|
| **GPU OpenCL** | `pyscf.OpenCL` via `make_rks(..., backend='gpu')` | Medium/large molecules (benzene→PTCDA), production AFM DFT refs |
| **smallDFT** | `pyscf.smallDFT` via `backend='smalldft'` | Small molecules, CPU OpenMP XC (needs `libsmalldft.so`) |
| **Stock CPU** | `backend='cpu'` | Parity / when NVIDIA ICD missing |

Default GPU profile: **`production_radial_screened_splitk`** (RTX 3090).

## Minimal SCF snippet

```python
import importlib.util, os
root = '/home/prokop/git/SPAMMM'
spec = importlib.util.spec_from_file_location(
    'pu', os.path.join(root, 'spammm/quantum/pySCF_utils-new.py'))
pu = importlib.util.module_from_spec(spec); spec.loader.exec_module(pu)

backend = pu.resolve_backend('auto')   # gpu if NVIDIA OpenCL else cpu
mf = pu.make_rks(atom_str, basis='def2-SVP', xc='PBE', backend=backend)
e, dm, cycles, wall = pu.run_scf(mf)
# frontier MOs: mf.mo_coeff, mf.mo_energy, pu.homo_lumo_indices(mf)
pu.write_frontier_mo_cubes(mf, out_dir, names, pos)  # HOMO/LUMO .cube
pu.release_scf(mf)
```

Or one-shot geometry:

```python
r = pu.run_scf_geometry(names, pos, basis='def2-SVP', xc='PBE', backend='auto', release=False)
```

## CO tip z-scan (AFM Pauli refs)

```bash
python tests/SPM/run_zscan_reference.py --molecules PTCDA --methods pyscf_gpu_pbe --z-max 15
```

Uses `run_co_zscan` (DM warm-start, site-correct tip xy). Report: `doc/Reports/PySCF_GPU_CO_zscan_PTCDA.md`.

## STM frontier cubes

```bash
# DFTB mio/3ob/prolonged + pySCF HOMO/LUMO panel
python tests/SPM/testplot_stm_basis_compare.py --molecules pentacene,PTCDA
# Frontier ±5 MOs at z=0.5 Å (spectrum↔ψ, E↑)
python tests/SPM/testplot_stm_basis_compare.py --frontier-diag --molecules pentacene,PTCDA --n-near 5
# DFTB only while pySCF runs elsewhere:
python tests/SPM/testplot_stm_basis_compare.py --molecules pentacene,PTCDA --skip-pyscf
```

Artifacts: `debug/stm_orbital_compare/<mol>/`.

**MO projection:** `eval_mo_on_xy_slice` / `eval_mo_on_grid` use full `numint.eval_ao` × `mo_coeff` (complete GTO basis, e.g. def2-SVP). Not the DFTB OpenCL STO kernels in `LCAO_grid.cl`. DFTB HOMO must use valence count — see `dftb_frontier_mo_indices` / report.

## Caveats

- Agent Shell **must** see NVIDIA (`required_permissions: ["all"]`); sandbox often shows PoCL only.
- GPU AFM SCF tols: `conv_tol=1e-6`, `conv_tol_grad=1e-4`, `max_cycle=40` (below f32 XC noise).
- DF `_cderi` rebuilds each geometry; warm-start DM still helps cycle count.
- Hyphenated filename → load via `importlib` until merge (§12 Refactor_LargeModules).
- Legacy `pySCF_utils.py` is a thin subset of `-new`; **merge plan:** overwrite with `-new` body as single `pySCF_utils.py` (not two maintained modules).
