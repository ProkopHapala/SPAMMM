---
type: TopicalAudit
title: Ribbon band structure & supercell unfolding
tags: [dftb, bands, unfolding, ribbons, edge-passivation]
---

# Ribbon band structure & supercell unfolding

## Summary

Pipeline for studying how edge chemistry and local hydrogenation of periodic
zigzag graphene nanoribbons changes the π electronic structure. One driver —
`tests/quantum/testplot_unfold.py` — builds N-terminated / C-terminated ribbons,
runs DFTB+ periodic single-points (optionally after relaxation), and produces the
**canonical band overlay**: all cell sizes in absolute kx [1/Å] on the full
primitive BZ, supercell spectra replicated by their reciprocal vectors, and
eigenvector-unfolded spectral weights as black dots behind the lines.

Motivation (ERC "hedgehog" project): does hydrogenation/protonation at one
junction switch the π channel elsewhere — measured via spectral-weight
delocalization, π bond orders, and cooperative energies J.

## Scientific results so far

- **Fold validation (pristine w4 acene):** x1→x4 and x2→x4 fold-down = 0.1 meV
  max dev; direct extended-zone x4 run (kf∈[−2,+2], 129 explicit kpts) vs
  replicated = 0.2 meV. Band counts 18/36/72 for x1/x2/x4. E/atom identical to
  6 decimals.
- **Unfold sharpness:** pristine x4 → 91.2% of eigenstates with W>0.9 in one
  primitive channel (rest are exact degeneracies at Γ / BZ edge — irreducible).
  One C displaced 0.1 Å → 65.9%; random 0.1 Å on all C → 4.7%. The sharp%
  is a working symmetry-breaking detector.
- **Edge chemistry, unrelaxed x1, gap [meV] vs width_chains:**

  | edge | w3 | w4 | w5 | w6 | w8 | w10 | w12 |
  |------|---:|---:|---:|---:|---:|----:|----:|
  | CH   | 8  | 23 | 17 | 41 | 34 | 35  | 41  |
  | N    | 10 | 38 | 11 | 16 | 2  | 7   | 4   |
  | C-OH | 8  | 81 | 8  | 66 | 51 | 53  | 49  |
  | CH2  | 16 |**3725**|6|**1910**|**975**|**517**|**278**|
  | NH   |**869**|24|47 | 15 | 35 | 31  | 42  |
  | C=O  | 23 | 17 | 18 | 22 | 34 | 31  | 23  |

  CH2 (sp3 edge, no pz) is the only true on/off switch — and **only on even
  width_chains**; its gap decays ~1/width as the interior π system bypasses the
  severed edge. NH and C=O switches are doping switches (HOMO shifts of ~5 eV
  and ~3.6 eV), not on/off. All sp2 edges stay semimetallic.

## Implementations

| Location | Status | Notes |
|----------|--------|-------|
| `tests/quantum/testplot_unfold.py` | active | Single driver, 4 modes (below) → `debug/unfold/` |
| `spammm/quantum/pi_bond_order.py` | active | `subcell_group_indices`, `unfold_T_weights` (canonical), `unfold_spectral_weights_DEPRECATED` (Euclidean — wrong for S≠I), `read_band_out`, `read_eigenvec_bin`, `pi_bond_orders_pbc` |
| `spammm/quantum/DFTB_utils.py` | active | `run_pbc`/`makeDFTBjob_pbc`: `nk`, `k_shift`, `klist` (explicit kpts), `do_relax`, `Mixer`, `extra_hsd` |
| `spammm/topology/MoleculeEditorBackend.py` | active | `PASSIVATION_GROUPS`, `PASSIVATION_ENCODING`, `build_zigzag_ribbon`, per-site passivation lists |
| `spammm/topology/ribbon_pbc.py` | active | `build_ribbon_cell`, junction cells, `A_CC`, `save_xyz_lvs` |
| `doc/chats/Band_Unfolging.chat.md` | reference | Derivation of the weight formula (PRB 89, 201412 suppl.) |
| `debug/unfold/index.html` | artifact | HTML gallery of all passiv/switch figures + gap table |

## Ribbon generation & passivation

`build_acene(width_chains, ncells, passivation)` in the driver wraps
`MoleculeEditorBackend.build_zigzag_ribbon(..., bPeriodicX=True)`:
**width_chains ↔ ring rows: w4=1 row, w6=2, w8=3, w10=4, w12=5**
(rows = w/2 − 1). Odd widths are half-integer rows — different edge symmetry.

`PASSIVATION_GROUPS` (edge-site chemistry, applied per edge site along x):
`'CH'` =CH- (aromatic ref), `'N'` pyridinic =N-, `'NH'` -NH- (protonated),
`'C=O'` -C(=O)-, `'C-OH'` =C(OH)-, `'CH2'` -CH2- (sp3, `H*` marker = bond to
host not chained), `'CHOH'` sp3 -CH(OH)- (added for the O-system hydrogenation).
`PASSIVATION_ENCODING` maps single chars (n/N/o/O/H/h/m) for CLI strings.
**Per-site passivation**: a list of length `ncells` gives each edge site its own
group, e.g. `['NH','N','N','N']` protonates site 0 on that edge — this is how
single-defect supercells are built.

## k-point handling in DFTB_utils

- `k_shift=(0,0,0)` → Γ-centered mesh that **includes the BZ edge** f=0.5
  (the default 0.5 MP half-shift misses it — was a real bug for folding plots).
- `klist=[(frac3, weight), ...]` → explicit `KPointsAndWeights` block; used for
  the extended-zone validation run (supercell computed directly at primitive-BZ
  k-points — zero folding bookkeeping on our side).
- `run_pbc` returns inversion-**reduced** k sets (only half the BZ); plots use
  `mirror()` = E(−k)=E(k) extension, and weight dots are duplicated at ±q.
- `Analysis { WriteEigenvectors = Yes }` via `extra_hsd` → `eigenvec.bin`
  (complex128 stream; read as `conj().T`, verified vs C†SC=I).

## Unfolding algorithm (`pi_bond_order.py`)

**Canonical method: `unfold_T_weights`** — spectral projectors of the
subcell-translation operator T (shift by one primitive pitch along x), evaluated
in the true overlap metric S(k):

    w_j(n) = (1/N_c) Σ_m exp(−i2π q_j m) ⟨ψ_n|T^m|ψ_n⟩,
             ⟨ψ|T^m|ψ⟩ = c† T^m S c,   q_j = (kf + j)/N_c

T is a unitary permutation mapping AO (g,m,a)→(g,m+1,a); the wrap column
m=N_c−1→0 carries the Bloch factor exp(+i2π·kf). For an exactly periodic
supercell every eigenstate is a T eigenstate → weight ~1 in a single channel;
defect-localized states dephase → weight spreads physically.
Σ_j w_j = ⟨ψ|S|ψ⟩ = 1 exactly (Parseval over the spectral measure).

**Why not the Euclidean formula** (`unfold_spectral_weights_DEPRECATED`, kept
for regression + legacy drivers): it is a projector only for an orthogonal
basis. DFTB mio eigenvectors are S-orthonormal with |S_offdiag| up to ~0.44 and
GPAW dzp ~0.5 → ~20–30% of every state's weight leaked into adjacent channels
*and* the error grew with |kf| (DFTB xscanv 0H: Wmax med ~1.0 near BZ centre →
~0.45 at zone edge, purely artificial — the relaxed geometry is exactly
8-fold periodic). After the fix all xscanv C,N 0H states give Wmax med 1.000,
100% sharp; GPAW 0H gives 0.88–1.00.

- `subcell_group_indices(apos_ideal, lvs, enames, ncells)` — equivalence groups
  of primitive atoms across the N_c subcells; grouping done on the **ideal**
  lattice (atom indexing survives relaxation). `strict=False` tolerates defect
  atoms whose group has 1 member — excluded from the projection.
- Needs S(k): DFTB side fetch `dftb.get_s_cplx()` **before** finalizing
  (`run_dftb_hs` in `test_ribbon_stm_sys.py`); GPAW side `S_kMM` is in hs.npz.
- Residual limitation: single-phase seam model — when the primitive pitch is so
  short that SK couplings span 2+ subcells (mio O ribbons, pitch 2.46 Å), seam
  blocks carry two image contributions and [T,S(k)] grows toward the zone edge
  (0.01→0.39) → O 0H Wmax med ~0.88, a representation error not physics.
- Caveat: reduced sharpness at exact degeneracies (Γ, BZ edge) is not a bug —
  degenerate eigenvectors mix arbitrarily.

## Canonical plot format (do not regress)

- Absolute kx [1/Å], full primitive BZ, each cell's BZ edge drawn in its own
  spectrum color.
- Layering: x4 lw=0.25 back → x2 lw=0.5 → x1 lw=2.0 on top; supercell spectra
  replicated by m·G to tile the primitive BZ.
- Unfolded weights: **filled black dots, size ∝ W, zorder below all lines**,
  plotted at both ±q (inversion partners).
- Geometry panels: cell rectangle in the same color as that system's band lines;
  bond-length coloring uses `seismic_r` on 1.30–1.55 Å for heavy-atom bonds.
- Chemistry colors: C family green, N blue, O red. Switch plots: ref black,
  hydrogenated red. Energy window ±4 eV (comparison) / ±8 eV (folding validation).

## Driver modes (`tests/quantum/testplot_unfold.py`)

```bash
# canonical folding validation: x1/x2/x4 overlay + replicas + extended-zone
python tests/quantum/testplot_unfold.py --nkx 64            # debug/unfold/test_unfold_acene.png

# edge-chemistry comparison, x1 only, chemistry colors, ±4 eV
python tests/quantum/testplot_unfold.py --nkx 128 --width 8 --passiv "CH,N,C-OH"
python tests/quantum/testplot_unfold.py --nkx 128 --width 8 --passiv "CH2,NH,C=O"
#   → test_unfold_passiv_w8_{CH-N-C-OH,CH2-NH-C=O}.png   (width IS in the name)

# per-system switch: aromatic ref (black) vs hydrogenated (red), one fig/system
python tests/quantum/testplot_unfold.py --nkx 128 --width 8 --switch
#   → test_unfold_switch_w8_{C_CH-vs-CH2,N_N-vs-NH,O_C-OH-vs-C=O}.png

# relaxed ref x1 + one-site-hydrogenated x2/x4, unfolded weight dots
python tests/quantum/testplot_unfold.py --nkx 64 --sys3
#   → test_unfold_sys{C,N,O}.png   (SWITCH vs SYS3: O-system hyd is C=O vs CHOH!)

# symmetry-breaking probes on the folding validation
python tests/quantum/testplot_unfold.py --perturb 0.2       # pull one edge H
python tests/quantum/testplot_unfold.py --rdistort 0.1      # displace one C
```

## Parity status

- x1→x4, x2→x4 eigenvalue fold-down: **0.1 meV** (per-KPT blocks, Γ-centered mesh).
- Extended-zone x4 vs replicated: **0.2 meV**.
- Σ_m W per eigenstate = 1.000; pristine sharp fraction 91.2% (nkx=64).
- Band counts scale exactly N× (18/36/72 at w4).

## Open issues / caveats

- **DIIS goes singular on Γ-centered metallic meshes** — sys3 band runs use
  `mixer=None` (Broyden). Keep in mind for other metals.
- Fold-check numbers are a consistency check **only for pristine cells**; for
  perturbed/hydrogenated cells the deviation *is* the band shift — don't read it
  as an error.
- Odd width_chains have different edge sublattice parity → qualitatively
  different physics (CH2 gap collapses). Don't mix parities in comparisons.
- `subcell_group_indices` requires the supercell to be built from ideal
  translations; pass `apos_id` (pre-relax) positions.
- O-system ambiguity: `--switch` pairs C-OH↔C=O (oxidation switch);
  `--sys3` pairs C-OH↔CHOH (hydrogenation). Pick consciously.
- The weights are normalized |a|², not the full S-corrected PRB formula —
  fine for pristine/sharp states, approximate for strongly perturbed ones.
  **Resolved in the xscan pipeline:** `unfold_T_weights` uses the true S(k)
  metric; this file's `testplot_unfold.py` drivers still call the deprecated
  Euclidean variant (historical figures; sharpness% there understates
  channel purity for mio bases).
- `debug/unfold/` filenames now encode width (`_w{n}_`); older fixed-name
  outputs were overwritten across widths (resolved 2026-09).
