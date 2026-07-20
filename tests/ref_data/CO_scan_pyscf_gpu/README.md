# CO rigid z-scan references — pySCF GPU (PBE/def2-SVP)

Site-correct **interaction energies** for a rigid CO tip above PTCDA.
Computed 2026-07-20 on **NVIDIA RTX 3090** via local pySCF OpenCL
(`spammm/quantum/pySCF_utils-new.py`, profile `production_radial_screened_splitk`).

## Units (important)

| Quantity | Unit | Source |
|----------|------|--------|
| `z` | Å | tip height above target atom |
| `E_int`, `E_rel`, `E_abs` | **eV** | after conversion |
| PySCF raw `mf.e_tot` | **Hartree** | never kcal/mol in this pipeline |
| Conversion | `E_eV = E_Ha × 27.211396641308` | `HAU2EV` in `pySCF_utils-new.py` |

Definition:

```
E_int(z) = E(mol + CO@z) − E(mol) − E(CO)     # interaction energy [eV]
E_rel(z) = E_int(z) − E_int(z_max)            # Ez_FDBM-compatible [eV]
```

## Geometry / sites (0-based indices)

| Label | Atom idx | Description | E_min | z_min |
|-------|----------|-------------|-------|-------|
| O_eq | 26 | =O carbonyl | −22.7 meV | 3.2 Å |
| O_br | 24 | -O- bridge | −41.2 meV | 3.2 Å |
| C_anh | 11 | anhydride C | −38.4 meV | 3.2 Å |
| C_core | 6 | perylene core C | −24.7 meV | 3.4 Å |

- `PTCDA.xyz` — plain geometry (same as `data/xyz/PTCDA.xyz`)
- `PTCDA_labeled.xyz` — same coords with `# idx=…` and `TARGET site=…` comments
- `sites.json` — machine-readable method + site metadata

**Tip:** O-apex CO; tip **xy follows the target atom** (not lab origin).

## Files

```
zscan_PTCDA_pyscf_gpu_pbe_def2svp_<site><idx>.dat   # human-readable E(z)
timing_PTCDA_<site><idx>.dat                        # per-z SCF timing
timing.json                                         # job wall summaries + per-point
sites.json                                          # method / units / sites SSOT
```

Naming mirrors `tests/ref_data/Ez_FDBM/zscan_*.dat` but lives in this folder so
legacy loaders (which only know DFTB/small-molecule pySCF keys) are not broken.

## Regenerate (one CLI)

```bash
python tests/SPM/run_zscan_reference.py --molecules PTCDA --methods pyscf_gpu_pbe --z-max 15 --no-cache
# optional: --backend cpu   # force stock pySCF
# optional: --distances-file path.dat
```

Workhorse: `spammm/quantum/pySCF_utils-new.py` → `run_co_zscan` (DM warm-start along z).

## Speed (headline)

| Job | Points | Wall | ≈ s/pt wall |
|-----|--------|------|-------------|
| Short z=1.5–1.9 (5×4), loose GPU tols | 20 | **305 s** | **~4 s** |
| Full grid →15 Å, missing 34×4 only | 136 | **1743 s** | **~13 s** |
| SCF kernel only (mean over full grid) | — | — | **~2.5 s** |

Details and caveats: `doc/Reports/PySCF_GPU_CO_zscan_PTCDA.md`.

## Plots (debug, not git-tracked)

`debug/pyscf_ptcda_co_scan/PTCDA_co_scan_Ez_full15_well.png`
