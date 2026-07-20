# Report: Fast site-correct CO z-scans with pySCF GPU

**Date:** 2026-07-20  
**Status:** reference archived (USER-confirmed curves)  
**Code (SSOT):** `spammm/quantum/pySCF_utils-new.py` → `run_co_zscan` (DM warm-start)  
**CLI (one script):** `tests/SPM/run_zscan_reference.py --methods pyscf_gpu_pbe --z-max 15`  
  (GPU OpenCL if available, else stock CPU). Do not add parallel ad-hoc scan drivers.  
**Refs:** `tests/ref_data/CO_scan_pyscf_gpu/`  
**Debug plots:** `debug/pyscf_ptcda_co_scan/`  
**Cousin (dimer XC benches only):** `/home/prokop/git/pyscf/expamples_prokop/profile_dimer_scan.py`

## Verdict

We can compute **site-correct rigid CO–molecule interaction energies** for **PTCDA-scale** systems on a single **RTX 3090** at roughly:

| Regime | Wall time | Notes |
|--------|-----------|-------|
| **~4 s / z-point** | short contact scan | end-to-end, production GPU-AFM tolerances |
| **~2.5 s / z-point** | SCF kernel only | `mf.kernel()` after DF is built |
| **~13 s / z-point** | full rebuild each z | first full-grid fill (DF+grid re-setup dominates) |

A **39-point** non-uniform grid to **15 Å** for **4 PTCDA sites** is practical overnight (~30 min for the missing points after caching z=1.5–1.9).

This unblocks DFT / prolonged-DFTB Pauli fitting for large aromatics that were previously “DFTB-only” in `run_zscan_reference.py`.

---

## Method (SSOT)

| Item | Value |
|------|-------|
| Molecule | PTCDA (`data/xyz/PTCDA.xyz`) |
| Sites (0-based) | O_eq=26, O_br=24, C_anh=11, C_core=6 |
| Tip | CO, **O-apex**, bond 1.13 Å |
| Tip placement | apex at `(atom_x, atom_y, atom_z + z)` — **site-correct** |
| XC / basis | PBE / **def2-SVP** |
| Backend | local pySCF OpenCL (`pyscf.OpenCL`) |
| Profile | `production_radial_screened_splitk` |
| Hardware | NVIDIA GeForce RTX 3090 |
| SCF tols (GPU-AFM) | `conv_tol=1e-6`, `conv_tol_grad=1e-4`, `max_cycle=40` |
| Warm-start | density matrix from previous z |
| Dispersion | **none** (bare PBE → shallow ~20–40 meV wells) |

### Energy definition and units

PySCF returns `mf.e_tot` in **Hartree** (atomic units). We never use kcal/mol in the pipeline.

```text
E_eV  = E_Ha × 27.211396641308          # HAU2EV in pySCF_utils-new.py
E_int = E(mol+CO@z) − E(mol) − E(CO)    # [eV]
E_rel = E_int(z) − E_int(z_max)         # [eV], Ez_FDBM-compatible
```

Cross-check on archived data: `E_abs − E_mol − E_CO − E_int == 0` (machine precision).  
Isolated refs: `E_mol = −1368.288874 Ha`, `E_CO = −113.099060 Ha`.

---

## Timing (detailed)

### Timing semantics (do not mix)

1. **`wall_scf`** — seconds inside `mf.kernel()` only (`run_scf`). Stored per point in `.dat` / `timing.json`.
2. **Job wall** — end-to-end process time (DF rebuild, host setup, Python). From `.out` logs.

For PTCDA, (2) ≈ 3–5× (1) when DF is rebuilt every z. Reusing DF/grid across z is the next big win.

### A. Short contact scan (z = 1.5–1.9 Å, Δz = 0.1)

Source: `debug/pyscf_ptcda_co_scan/PTCDA_co_scan_loose.out`

| Site | E(1.5) / eV | E(1.9) / eV | site wall / s | ≈ s/pt | cycles (5 pts) |
|------|-------------|-------------|---------------|--------|----------------|
| O_eq26 | 6.078 | 1.587 | 79.0 | 4.29 | 32/23/16/10/14 |
| O_br24 | 7.534 | 1.776 | 77.2 | 4.65 | 24/32/16/16/14 |
| C_anh11 | 4.934 | 1.539 | 74.2 | 4.06 | 27/16/16/18/15 |
| C_core6 | 6.059 | 2.116 | 74.8 | 3.90 | 29/11/15/16/15 |
| **Total** | | | **305.3** | **~4.1** | all converged |

### B. Full non-uniform grid → 15 Å (39 pts)

Grid (same recipe as `make_z_grid(15)`):

```text
1.5–3.0 / 0.1 · 3.0–5.0 / 0.2 · 5.0–8.0 / 0.5 · 8.0–15.0 / 1.0
```

Reused loose z=1.5–1.9; computed **34 missing × 4 sites**.

| | Value |
|--|-------|
| Missing-only job wall | **1742.9 s (~29 min)** |
| ≈ wall / new point | **~12.8 s** |
| SCF-only sum / site (39 pts) | **94–100 s** (~2.4–2.6 s mean) |

Per-site SCF stats (`timing.json`):

| Site | Σ wall_scf | mean | median | min | max | mean cycles |
|------|------------|------|--------|-----|-----|-------------|
| O_eq26 | 93.9 s | 2.41 | 2.36 | 1.08 | 6.90 | 10.2 |
| O_br24 | 100.1 s | 2.57 | 2.22 | 1.26 | 7.03 | 10.9 |
| C_anh11 | 96.0 s | 2.46 | 2.01 | 0.58 | 6.01 | 10.3 |
| C_core6 | 95.5 s | 2.45 | 2.28 | 0.89 | 6.38 | 10.1 |

Point-by-point tables: `tests/ref_data/CO_scan_pyscf_gpu/timing_PTCDA_*.dat`.

### C. Why early “non-convergence” was a red herring

First attempt used profile defaults `conv_tol=1e-8`, `conv_tol_grad=1e-5` with `max_cycle=50`. GPU f32 XC noise floor sits near that gradient tolerance → DIIS hunted forever. Chemically done by ~12–17 cycles; loose GPU-AFM tols (above) converge cleanly and cut wall ~2×.

---

## Physics results (PTCDA)

| Site | E_min / meV | z_min / Å | E(1.5) / eV | E(15) / eV |
|------|-------------|-----------|-------------|------------|
| O_br24 | −41.2 | 3.2 | 7.53 | ~0.0003 |
| C_anh11 | −38.4 | 3.2 | 4.93 | ~0.0007 |
| C_core6 | −24.7 | 3.4 | 6.06 | ~0.0005 |
| O_eq26 | −22.7 | 3.2 | 6.08 | ~0.0007 |

Shallow wells are expected without D3/D4. For vdW-depth AFM wells, add dispersion later; Pauli / contact region (z ≲ 2.5 Å) is the high-value part for FDBM fitting.

**REVIEW plots:**

- `debug/pyscf_ptcda_co_scan/PTCDA_co_scan_Ez_full15_well.png` (meV well)
- `debug/pyscf_ptcda_co_scan/PTCDA_co_scan_Ez_full15_main.png`

---

## Where the data live

```text
tests/ref_data/CO_scan_pyscf_gpu/
  README.md
  PTCDA.xyz / PTCDA_labeled.xyz     # atom indices + TARGET tags
  sites.json / timing.json
  zscan_PTCDA_pyscf_gpu_pbe_def2svp_<site><idx>.dat
  timing_PTCDA_<site><idx>.dat
```

Style matches `tests/ref_data/Ez_FDBM/zscan_*.dat` (comment header + columns), but kept in a **separate folder** so existing `testplot_zscan_reference` filename lookup is not broken. Columns include `E_int[eV]`, `E_rel[eV]`, `E_abs[eV]`, `E_int[Ha]`, `cycles`, `wall_scf[s]`.

Contrast: `Ez_FDBM` PTCDA curves are **DFTB only** and stop at 8 Å with `E_rel = E(z)−E(z_max)`.

---

## Relation to Fukui jobs

`/home/prokop/SIMULATIONS/Fukui_AFM/jobs_CO_scan_pyscf_short` only scanned z=1.5–1.9 and placed the tip at **lab xy=(0,0)** (site labels were filename-only). Those data are **not** site-correct references. This archive replaces them for PTCDA.

---

## Next optimizations (not done)

1. Persist DF / grid plan across z (only tip moves) → push wall toward `wall_scf` (~2.5 s).
2. Optional D3/D4 for deeper attractive wells.
3. Wire `pySCF_utils-new` into `run_zscan_reference.py` as `pyscf_gpu_pbe` for automated regeneration.
4. Promote selected points to L0 pytest once fitting tolerances are agreed.

---

## Links

- Roadmap §2: `doc/ARCHITECTURE_ROADMAP.md`
- Prolonged DFTB context: `doc/Tasks/ProlongedRadialBasis_DFTB.md`
- **Follow-on (Pauli + SA AFM):** `doc/Reports/PTCDA_FDBM_prolonged_basis.md` · `FOR_PRESENTATION.md`
- API: `spammm/quantum/pySCF_utils-new.py` (`make_z_grid`, `run_co_zscan`)
