# Report: PTCDA FDBM — stock 3ob vs SA-prolonged Pauli (2026-07-20)

**Status:** results for USER review (not marked Done)  
**Presentation index:** [`FOR_PRESENTATION.md`](../FOR_PRESENTATION.md) · gallery [`debug/presentation.html`](../../debug/presentation.html)

## Verdict

With **site-correct pySCF GPU Ez** refs for PTCDA, we can:

1. Fit **Slater-tail (SA-prolonged)** projection parameters to pySCF density.
2. Refit **Pauli A,β** (CO tip) to Ez in the contact window.
3. Run **PP-AFM** images: **stock multi-ζ 3ob-3-1** vs **SA-prolonged** side-by-side over **8 heights**.

**SA** = Simulated Annealing of single-exponent Slater STOs `(N, ζ)` per element for **projection only** (SCF still uses stock 3ob). **Not mio.**

At AFM heights ~3.3 Å the prolonged basis keeps **Pauli repulsion** on the backbone while stock 3ob is already mostly attractive — the intended vacuum-tail effect.

---

## Pipeline (SSOT)

| Step | Tool | Output |
|------|------|--------|
| Ez refs (PBE/def2-SVP, CO O-apex) | `run_zscan_reference.py` / `pySCF_utils-new` | `tests/ref_data/CO_scan_pyscf_gpu/` |
| Sample ρ (pySCF) | `compute_densities.py --methods pyscf_gpu_pbe` | `debug/densities/rho_PTCDA_pyscf_gpu_pbe.*` |
| SA Slater fit vs ρ | `examples/density_comparison/optimize_basis.py --ref-rho …` | `debug/dftb_basis_sa_ptcda/` |
| Pauli FDBM vs Ez (log) | `extract_pauli_zscan.py` + CO tip | `debug/pauli_zscan_pyscf_ptcda/`, `…/PTCDA_pauli_stock_vs_sa_vs_ez.png` |
| AFM images | `testplot_fdbm_relax.py --ptcda-stock-vs-sa` | `debug/fdbm_ptcda_stock_vs_sa/` |

### Naming

| Label | Meaning |
|-------|---------|
| **stock 3ob** | Multi-ζ **3ob-3-1** WFC projection (NOT mio) |
| **SA-prolonged** | Single-exp Slater tails, SA-fit vs pySCF ρ; SCF DM unchanged |

### Fitted numbers (PTCDA, CO tip)

**Slater tails** (fit log-ρ, z∈[1.0, 2.5] Å; obj 18.1 → 0.79):

| Elem | N | ζ [1/Å] |
|------|---|--------|
| H | −3.36 | 2.57 |
| C | −4.98 | 2.65 |
| O | −10.53 | 3.57 |

**Pauli** `E = A · S^β` vs Ez, z∈[1.7, 2.5] Å, tip=`co`:

| Density | A | β | R² |
|---------|---|---|-----|
| stock 3ob | 12.82 | 0.651 | 0.973 |
| SA-prolonged | 11.76 | 0.852 | 0.980 |

Params JSON: `debug/dftb_basis_sa_ptcda/PTCDA_sa_params.json`

### Caveats

- SA `ρ_NA` / `V_ES` charge imbalance is large → AFM compare uses **shared stock V_ES**; only Pauli ρ + A,β differ.
- PTCDA grid `ny=176` (factor 11) → `SPAMMM_AFM_CPU_FFT=1` for FFT.
- Ez beyond ~3 Å flattens (non-Pauli / noise); fair Pauli test is the contact wall.
- L0 pytest + GUI WFC switch still open (see task).

---

## Key figures

1. **AFM height strip (main):** [`debug/fdbm_ptcda_stock_vs_sa/compare_stock3ob_vs_SAprolonged_heights.png`](../../debug/fdbm_ptcda_stock_vs_sa/compare_stock3ob_vs_SAprolonged_heights.png) — columns h=2.5…5.3 Å; rows df/Fz × stock/SA  
2. **Pauli log vs Ez:** [`debug/dftb_basis_sa_ptcda/PTCDA_pauli_stock_vs_sa_vs_ez.png`](../../debug/dftb_basis_sa_ptcda/PTCDA_pauli_stock_vs_sa_vs_ez.png)  
3. **ρ(z) SA vs pySCF:** [`debug/dftb_basis_sa_ptcda/PTCDA_sa_optimized.png`](../../debug/dftb_basis_sa_ptcda/PTCDA_sa_optimized.png)  
4. **pySCF Ez overlay (FDBM from pySCF ρ):** [`debug/pauli_zscan_pyscf_ptcda/pauli_vs_ez_overlay_pyscf_gpu_pbe.png`](../../debug/pauli_zscan_pyscf_ptcda/pauli_vs_ez_overlay_pyscf_gpu_pbe.png)

## Reproduce

```bash
# SA fit + project ρ
python examples/density_comparison/optimize_basis.py \
  --ref-rho debug/densities/rho_PTCDA_pyscf_gpu_pbe.npy \
  --ref-meta debug/densities/rho_PTCDA_pyscf_gpu_pbe.meta.npz \
  --xyz data/xyz/PTCDA.xyz --molecule PTCDA --n-iter 2000 \
  --fit-lo 1.0 --fit-hi 2.5 --z-max 4.0 \
  --outdir debug/dftb_basis_sa_ptcda --project-density --compare-pauli --tip-mode co

# AFM strip (8 heights)
SPAMMM_AFM_CPU_FFT=1 python tests/SPM/testplot_fdbm_relax.py --ptcda-stock-vs-sa \
  --outdir debug/fdbm_ptcda_stock_vs_sa
```

## Related docs

- Task: `doc/Tasks/ProlongedRadialBasis_DFTB.md`
- Design: `doc/DFTB_basis_fit.md`
- Ez refs: `doc/Reports/PySCF_GPU_CO_zscan_PTCDA.md`
- Roadmap P0: `doc/ARCHITECTURE_ROADMAP.md`
