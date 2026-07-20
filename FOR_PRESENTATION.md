# FOR PRESENTATION — PTCDA FDBM / prolonged DFTB (2026-07-20)

Open the HTML gallery for clickable previews: **[`debug/presentation.html`](debug/presentation.html)**  
Technical write-up: [`doc/Reports/PTCDA_FDBM_prolonged_basis.md`](doc/Reports/PTCDA_FDBM_prolonged_basis.md)

---

## One-liner

**Site-correct pySCF Ez for PTCDA → SA Slater-tail projection (“SA-prolonged”) vs stock multi-ζ 3ob → FDBM CO-tip AFM images over 8 heights.**

| Term | Meaning |
|------|---------|
| **stock 3ob** | Multi-ζ **3ob-3-1** density projection (**not mio**) |
| **SA-prolonged** | **S**imulated **A**nnealing fit of single-exponent Slater tails to pySCF ρ; SCF still 3ob |

---

## Slide 1 — Why

Stock DFTB projection cutoffs kill vacuum tails → Pauli too short for AFM. Prolong projection STOs only; keep SCF DM.

## Slide 2 — DFT ground truth (Ez)

Site-correct CO (O-apex) PBE/def2-SVP, z→15 Å, 4 PTCDA sites.

| Artifact | Link |
|----------|------|
| Report | [doc/Reports/PySCF_GPU_CO_zscan_PTCDA.md](doc/Reports/PySCF_GPU_CO_zscan_PTCDA.md) |
| Ref data | [tests/ref_data/CO_scan_pyscf_gpu/](tests/ref_data/CO_scan_pyscf_gpu/) |
| Ez wells plot | [debug/pyscf_ptcda_co_scan/](debug/pyscf_ptcda_co_scan/) (if present) |

## Slide 3 — FDBM Pauli vs Ez (log)

Gaussian-tip path first; then CO tip + stock/SA densities.

| Artifact | Link |
|----------|------|
| pySCF-ρ Pauli vs Ez (log overlay) | [debug/pauli_zscan_pyscf_ptcda/pauli_vs_ez_overlay_pyscf_gpu_pbe.png](debug/pauli_zscan_pyscf_ptcda/pauli_vs_ez_overlay_pyscf_gpu_pbe.png) |
| Summary | [debug/pauli_zscan_pyscf_ptcda/comparison_summary.out](debug/pauli_zscan_pyscf_ptcda/comparison_summary.out) |
| Per-site panels | [debug/pauli_zscan_pyscf_ptcda/](debug/pauli_zscan_pyscf_ptcda/) |

## Slide 4 — SA Slater fit (density)

| Artifact | Link |
|----------|------|
| ρ(z) pySCF vs initial vs SA | [debug/dftb_basis_sa_ptcda/PTCDA_sa_optimized.png](debug/dftb_basis_sa_ptcda/PTCDA_sa_optimized.png) |
| SA history | [debug/dftb_basis_sa_ptcda/PTCDA_sa_history.png](debug/dftb_basis_sa_ptcda/PTCDA_sa_history.png) |
| Params (N,ζ) | [debug/dftb_basis_sa_ptcda/PTCDA_sa_params.json](debug/dftb_basis_sa_ptcda/PTCDA_sa_params.json) |
| Pauli log: stock vs SA vs Ez | [debug/dftb_basis_sa_ptcda/PTCDA_pauli_stock_vs_sa_vs_ez.png](debug/dftb_basis_sa_ptcda/PTCDA_pauli_stock_vs_sa_vs_ez.png) |

**Fitted Pauli (CO tip, Ez window 1.7–2.5 Å):** stock A=12.82 β=0.651 · SA A=11.76 β=0.852

## Slide 5 — AFM images (hero)

**Main figure — 8 heights, df then Fz, stock 3ob vs SA-prolonged:**

→ [debug/fdbm_ptcda_stock_vs_sa/compare_stock3ob_vs_SAprolonged_heights.png](debug/fdbm_ptcda_stock_vs_sa/compare_stock3ob_vs_SAprolonged_heights.png)

| Extra | Link |
|-------|------|
| SUMMARY | [debug/fdbm_ptcda_stock_vs_sa/SUMMARY.out](debug/fdbm_ptcda_stock_vs_sa/SUMMARY.out) |
| Scan arrays | [debug/fdbm_ptcda_stock_vs_sa/scan_stock_vs_sa.npz](debug/fdbm_ptcda_stock_vs_sa/scan_stock_vs_sa.npz) |
| Stage fields stock | [debug/fdbm_ptcda_stock_vs_sa/stage_stock.png](debug/fdbm_ptcda_stock_vs_sa/stage_stock.png) |
| Stage fields SA | [debug/fdbm_ptcda_stock_vs_sa/stage_sa.png](debug/fdbm_ptcda_stock_vs_sa/stage_sa.png) |
| df grids | [df_stock.png](debug/fdbm_ptcda_stock_vs_sa/df_stock.png) · [df_sa.png](debug/fdbm_ptcda_stock_vs_sa/df_sa.png) |

**Talking point at h≈3.3 Å:** stock already attractive (blue Fz); SA-prolonged still repulsive on the backbone.

## Slide 6 — Densities on disk

| ρ | Path |
|---|------|
| pySCF PBE/def2-SVP | [debug/densities/rho_PTCDA_pyscf_gpu_pbe.npy](debug/densities/rho_PTCDA_pyscf_gpu_pbe.npy) |
| stock 3ob | [debug/densities/rho_PTCDA_dftb_3ob.npy](debug/densities/rho_PTCDA_dftb_3ob.npy) |
| SA-prolonged | [debug/densities/rho_PTCDA_dftb_3ob_sa.npy](debug/densities/rho_PTCDA_dftb_3ob_sa.npy) |

---

## Reproduce (short)

```bash
SPAMMM_AFM_CPU_FFT=1 python tests/SPM/testplot_fdbm_relax.py --ptcda-stock-vs-sa
```

Full chain: see `doc/Reports/PTCDA_FDBM_prolonged_basis.md`.

## Docs updated

- `doc/Reports/PTCDA_FDBM_prolonged_basis.md` (this campaign)
- `doc/Tasks/ProlongedRadialBasis_DFTB.md`
- `doc/DFTB_basis_fit.md`
- `doc/Reports/PySCF_GPU_CO_zscan_PTCDA.md`
- `doc/ARCHITECTURE_ROADMAP.md` (P0 note)
- `tests/SPM/README.md`
