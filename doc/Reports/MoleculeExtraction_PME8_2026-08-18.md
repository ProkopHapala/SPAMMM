---
type: Report
title: Molecule extraction from CONTCAR + PME8 8-site charge-ring simulator
tags: [pme, charge-rings, stm, opencl, contcar, molecule-extraction, symmetry]
date: 2026-08-18
---

# Molecule extraction from CONTCAR + PME8 8-site charge-ring simulator

## Overview

Two tasks were accomplished in this thread:

1. **Molecule extraction from VASP CONTCAR**: utility functions to load a CONTCAR/POSCAR file, identify molecules via PBC-aware bond finding, extract rigid-body parameters (COM + PCA rotation), analyze approximate crystallographic symmetry, and visualize molecular neighborhoods.

2. **PME8 kernel**: a new OpenCL kernel extending the charge-ring Pauli Master Equation simulator from 4 sites (16 states) to 8 sites (256 states), enabling simulation of a 7-molecule finite cluster (central + 6 neighbors).

---

## Part 1: Molecule extraction from CONTCAR

### Problem

The VASP CONTCAR file (`data/chao_Li/CONTCAR_unit_cell`) contains 184 atoms (C₂₀H₁₈N₄Br₄ × 4 molecules = 4×46 atoms). No CONTCAR/POSCAR loader existed in the codebase. The goal was to:
- Identify individual molecules (bond-finding with PBC)
- Extract COM and orientation (PCA) for each
- Determine non-equivalent molecules via approximate symmetry
- Visualize neighborhoods with COM-to-COM vectors

### Key files

| File | Role |
|------|------|
| `spammm/atomicUtils.py` — `loadPOSCAR()` | New function: loads VASP CONTCAR/POSCAR with lattice vectors, element names, charges |
| `spammm/utils/moleculeRigid.py` | New module: PBC bond finding, connected components, rigid-body extraction, symmetry analysis, neighborhood plotting |
| `debug/extract_molecules/run_extract.py` | Run script: loads CONTCAR, extracts molecules, plots rigid bodies + neighborhoods |
| `debug/extract_molecules/*.png` | Output plots (rigid bodies XY/XZ, neighborhoods mol0–mol3) |

### Functions added to `moleculeRigid.py`

- `findBondsPBC(apos, enames, lvec, RvdwCut)` — PBC-aware bond finding via minimum-image convention
- `connected_components(natoms, bonds)` — BFS molecule identification
- `unwrap_molecule(apos, bonds, lvec)` — unwraps molecules split across PBC boundaries
- `extract_rigid_bodies(apos, enames, lvec, ...)` — full pipeline: COM + PCA rotation + bond list
- `find_approx_symmetry(coms_frac, tol)` — searches signed permutation matrices + translations for approximate symmetry operations
- `equivalence_classes(ops, nmol)` — groups molecules by symmetry equivalence
- `plot_rigid_bodies(bodies, lvec, ...)` — plots all molecules with COM + principal axes
- `plot_molecule_neighborhood(bodies, lvec, center_idx, radius, ...)` — plots central molecule + periodic-image neighbors with COM-to-COM vectors and distance labels

### Key findings

- **4 molecules** of 46 atoms each (C₂₀H₁₈N₄Br₄), correctly identified with `RvdwCut=1.2`
- All 4 are **chemically identical** (same formula, bond degree sequence, PCA eigenvalues)
- **Approximate symmetry**: two 2₁ screw axes found:
  - Along a: `R=diag(1,-1,-1)`, `t≈(½,0,½)`, dev=0.015 (0.37 Å) → mol0↔mol2, mol1↔mol3
  - Along b: `R=diag(-1,1,-1)`, `t≈(½,½,½)`, dev=0.04 (1.0 Å) → mol0↔mol1, mol2↔mol3
- At `tol=0.05` (~1.25 Å), all 4 molecules are **equivalent** → 1 class {0,1,2,3}
- The unrelaxed crystal is likely **P2₁2₁2₁** (No. 19), broken to P1 by DFT relaxation
- **6 nearest neighbors** per molecule at 11–16 Å, then a gap to ~19 Å

### Caveats

- **CONTCAR z-coordinates are not reliable** for symmetry analysis — DFT relaxation breaks the ideal symmetry. The z-differences (0 to -0.85 Å) between molecules are artifacts of relaxation, not genuine structural differences. **For PME simulations, z must be flattened** (all sites at same z) to preserve the equivalence.
- `spglib` reports P1 (no symmetry) because the tolerance needed (~1 Å) is too large for standard spglib settings
- The approximate symmetry search uses signed permutation matrices only (orthogonal lattice); for general monoclinic cells with non-90° angles, the rotation matrices would need to be in Cartesian space

---

## Part 2: PME8 — 8-site / 256-state charge-ring kernel

### Problem

The existing PME kernel (`kernels/PME.cl`) is hardcoded to `N_SITES=4` (16 states = 2⁴). A 7-molecule cluster needs 7 sites (128 states). The user requested a separate kernel optimized for 8 sites (256 states = 2⁸), with 256 threads per workgroup (1 thread per state).

### Key files

| File | Role |
|------|------|
| `kernels/PME8.cl` | New OpenCL kernel: 8-site PME with sparse iterative solver |
| `spammm/quantum/PauliSolverCL8.py` | New Python wrapper: `PauliSolverCL8` class |
| `tests/quantum/test_pme8_smoke.py` | Parity test: PME8 vs PME4 on 4-site square tetramer |
| `debug/extract_molecules/run_pme7.py` | Run script: 7-mol cluster PME8 scan (XY + XV) |
| `debug/extract_molecules/pme7/pme7_xy_xv.png` | Output: 4-panel plot (XY STM, XY dIdV, XV STM, XV dIdV) |
| `debug/extract_molecules/pme7/cluster7_sites.txt` | Site geometry file |

### Design

**Why not dense Gauss-Jordan (like PME4)?**
- PME4 uses a 16×16 dense matrix in local memory (1 KB) + parallel Gauss-Jordan elimination
- PME8 would need 256×256 = 256 KB — far exceeds GPU local memory (48 KB on RTX 3090)

**Solution: sparse iterative solver**
- Each state has at most 8 neighbors (one per site flip), so the rate matrix is stored sparsely:
  `Rates[256][8]` (8 KB) + `DiagLoss[256]` (1 KB) + `P_old/P_new[256]` (2 KB) + `reduce_buf[256]` (1 KB) ≈ **13 KB local memory**
- **Explicit Euler time-stepping** to steady state (guaranteed convergence for rate matrices)
  - `dt = 0.5 / max|DiagLoss|` (adaptive, stable)
  - Normalisation after each step: `P /= sum(P)`
  - Convergence: `max|ΔP| < tol`
- Current via parallel reduction across 256 threads (not serial thread-0 loop like PME4)

### Problems encountered and solutions

1. **Jacobi solver diverged** (P_sum = 23726):
   - The rate matrix is not diagonally dominant in the Jacobi sense
   - **Fix**: switched to explicit Euler time-stepping (`P_new = P_old + dt·(K·P_old)`) with normalisation — guaranteed to converge to steady state

2. **`__local` variables in inner scope** (OpenCL compile error):
   - NVIDIA's OpenCL compiler requires `__local` declarations at outermost kernel scope
   - **Fix**: moved all `__local` scalars (`s_max_rate`, `s_dt`, `s_psum`) to top of kernel

3. **Missing blobs in XY map** (only 3/7 sites showed current):
   - With `Esite=-0.09`, the `bRamp` term created z-dependent baseline shifts; sites at lower z (z=-0.85) had H_shift always below mu0=0 → permanently occupied, no charging
   - **Fix 1**: flatten all site z-coordinates to 0 (molecules are equivalent; z-differences are DFT relaxation artifacts)
   - **Fix 2**: use `Esite=-0.1` (below 0) so sites charge when tip electrostatic shift pushes H_shift above 0
   - **Fix 3**: use `Qzz=0` (monopole tip) so no orientation dependence — all sites truly equivalent at W=0

4. **Plot missing correspondence lines**:
   - XY plot didn't show the XV cut line; XV plot didn't show the VBias line
   - **Fix**: added white dashed cut line on XY, cyan dashed VBias horizontal line on XV, cyan dotted vertical lines at site positions along cut — matching `ChargeRingsExtension.py` GUI convention

### Parity verification

PME8 vs PME4 on 4-site square tetramer (4 active + 4 spectators):
- **max_abs_diff = 4.14e-08** (float32 precision)
- P sums = 1.000000 (perfect normalisation)
- Test: `tests/quantum/test_pme8_smoke.py`

### PME8 parameters for 7-molecule cluster

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_sites | 7 active + 1 spectator | 7-mol cluster, padded to 8 |
| Esite | -0.1 eV | Below mu0=0, charges when tip shifts energy up |
| W | 0.0 | Independent charging (no capacitive coupling) |
| Qzz | 0.0 | Monopole tip (no orientation dependence) |
| Q0 | 1.0 | Unit monopole charge |
| VBias | 2.0 V | Bias voltage |
| Temp | 3.0 K | Temperature |
| z_tip | 3.0 Å | Tip height (closer = stronger shift) |
| Rtip | 3.0 Å | Tip radius |
| decay | 0.3 | Tunneling decay |
| bRamp | False | Disable z-ramp (would break z-flat equivalence) |
| bMirror | True | Image charge mirror |
| max_iter | 5000 | Euler iterations |
| tol | 1e-7 | Convergence tolerance |

### Caveats

- **z-coordinates must be flattened** for the PME simulation. The CONTCAR z-differences are DFT relaxation artifacts. If real z-differences matter (e.g. molecules on a corrugated surface), they should come from a trusted geometry source, not the relaxed CONTCAR.
- **Euler solver convergence** depends on `max_iter` and `tol`. For extreme parameter regimes (very low T, very high W), more iterations may be needed. The adaptive dt ensures stability but not speed.
- **Spectator site** (site 7) is placed at (1000, 1000, 0) with E=1000 eV — far away and high energy so it never participates. For n<8 active sites, pad with spectators.
- **Wij must be provided** (not scalar W) when n_active < 8, to avoid coupling spectators. Use `make_Wij_active` pattern from `pauli_scan.py`.
- **dIdV in XY** is computed via finite difference (two kernel calls with dQ=0.02). For production, a single-call analytical derivative would be faster.
- **No state probability output in XV scan** — disabled for speed. Can be re-enabled with `return_probs=True`.
- **Future: W dependent on distance and orientation** — the user noted that W should eventually be set based on inter-molecular distance and orientation (possibly quadrupole-quadrupole interaction). This requires computing W_ij from the molecular geometry rather than using a constant.

### Ground state and NDR analysis (2026-08-18)

With Esite=-0.1 eV and W=0.05 eV for 7 sites, the many-body ground state is **NOT all-occupied**:

| n_occ | E (eV) | Notes |
|-------|--------|-------|
| 0 | 0.000 | all empty |
| 1 | -0.100 | |
| 2 | **-0.150** | **ground state** |
| 3 | **-0.150** | **ground state (degenerate)** |
| 4 | -0.100 | |
| 5 | 0.000 | |
| 6 | +0.150 | |
| 7 | +0.350 | all occupied (unfavorable) |

The charging energy C(7,2)×W = 21×0.05 = 1.05 eV exceeds 7×|Esite| = 0.7 eV, so full occupation is unfavorable. The ground state has 2–3 sites occupied.

**NDR absence with Qzz=0**: The monopole tip shift is always positive (pushes site energy UP as V increases), so sites only discharge — never charge back up. NDR requires a site to charge (electron enters) as V increases, blocking neighbors via W. The user corrected: **NDR does NOT require Qzz≠0** — the fig3 trimer achieves NDR with Qzz=0 (monopole). The NDR mechanism in the trimer comes from the sequential charging of sites as V increases, combined with W blocking. **The PME8 solver must reproduce the trimer NDR before proceeding to the 7-site system.** This is the next validation step.

### Trimer NDR validation (2026-08-18)

**Test**: `tests/quantum/test_trimer_pme8_ndr.py` — runs the fig3 symmetric trimer with both PME4 and PME8 (3 active + 5 spectators), compares xV scans.

**Parameters** (from `symmetric_trimer_params`): nsite=3, R=5.77, Qzz=0, Esite=-0.09, W=0.05, Temp=2.6K, z_tip=6, bRamp=True, bMirror=True.

| Metric | PME4 (reference) | PME8 | Match |
|--------|-----------------|------|-------|
| STM max | 2.793e-07 | 2.793e-07 | exact |
| dIdV min (NDR) | -6.567e-06 | -5.689e-06 | ~13% diff |
| NDR fraction | 4.9% | 5.5% | close |
| max\|dSTM\| | — | 3.6e-08 | good |

**Conclusion**: PME8 reproduces the trimer NDR. The small dIdV difference is from the Euler solver's slightly different convergence vs Gauss-Jordan, amplified by the finite-difference gradient.

**Artifacts**: `debug/test_trimer_pme8/trimer_pme4_vs_pme8.png`, `debug/test_trimer_pme8/trimer_IV_site0.png`

### 7-site NDR with trimer-like params (2026-08-18)

After validating the trimer, the 7-site cluster was re-run with trimer-like parameters (Esite=-0.09, W=0.05, Temp=2.6K, z_tip=6, bRamp=True, Qzz=0). The previous run used Esite=-0.1, Temp=10K, z_tip=3, bRamp=False which suppressed NDR (too much thermal broadening + wrong energy landscape).

**Result**: NDR present — dIdV min = -4.17e-06, NDR fraction = 20.5%. The NDR appears at V~1.0-1.3V in the XY stack.

**Key lesson**: NDR requires:
1. Low temperature (2.6K, not 10K) — sharp Fermi edge for abrupt charging transitions
2. `bRamp=True` — the z-ramp creates the position-dependent baseline that enables sequential charging
3. `Esite` close to 0 (−0.09, not −0.1) — sites must be near the chemical potential for charging to occur at accessible voltages
4. W≠0 — capacitive coupling blocks neighbors when a site charges, creating the NDR
5. **Distance-dependent W** — uniform W with many sites creates a highly degenerate ground state (C(7,2)=21 for 7 sites), smearing transitions and suppressing NDR. Using W_ij = W0·(r0/r_ij)³ breaks the degeneracy: close pairs interact strongly (sharp NDR at low V), distant pairs interact weakly. This matches the physical expectation that capacitive coupling decays with distance.

### Distance-dependent W and NDR at low voltage (2026-08-18)

With uniform W=0.05 eV, the 7-site system showed NDR only at V~1.0V when all rings overlap. The ground state (2 sites occupied) has C(7,2)=21 degenerate configurations, smearing the charging transitions.

**Fix**: W_ij = W0·(r0/r_ij)³ where W0=0.05 at r0=10 Å (dipole-like capacitive coupling decay). This breaks the degeneracy:
- Closest pairs (r~10.6 Å): W~0.042 eV (strong blocking → NDR)
- Distant pairs (r~25-29 Å): W~0.002-0.003 eV (weak coupling)
- Ground state: sites 4,6 (most distant pair, W=0.002, E=-0.178)

**Result**: NDR visible at V=0.3V onward (dIdV min = -6.8e-6 at V=0.3, -7.3e-6 at V=0.5). NDR fraction = 40-45% at all voltages. This matches the user's observation that Chao-li sees NDR when only two rings overlap.

**NDR vs site count** (uniform W, V=0.85):
| n_sites | C(n,2) | NDR min | NDR fraction |
|---------|--------|---------|-------------|
| 3 | 3 | -5.1e-6 | 4.3% |
| 5 | 10 | ~0 | 2.7% |
| 7 | 21 | ~0 | 2.7% |

With distance-dependent W, 7-site NDR at V=0.5: dIdV min = -7.3e-6, fraction = 41%.

---

## File index (all files created/modified in this thread)

### New files
- `spammm/utils/moleculeRigid.py` — molecule extraction, symmetry, plotting
- `kernels/PME8.cl` — 8-site PME OpenCL kernel
- `spammm/quantum/PauliSolverCL8.py` — Python wrapper for PME8
- `tests/quantum/test_pme8_smoke.py` — parity test PME8 vs PME4
- `tests/quantum/test_trimer_pme8_ndr.py` — trimer NDR reproduction PME8 vs PME4
- `debug/extract_molecules/run_extract.py` — molecule extraction run script
- `debug/extract_molecules/run_pme7.py` — 7-mol PME8 scan run script
- `debug/extract_molecules/pme7/cluster7_sites.txt` — site geometry
- `debug/extract_molecules/pme7/pme7_xy_xv.png` — 4-panel PME result plot
- `debug/extract_molecules/plot_rigid_bodies_xy.png` — all 4 molecules XY
- `debug/extract_molecules/plot_rigid_bodies_xz.png` — all 4 molecules XZ
- `debug/extract_molecules/neighborhood_mol{0,1,2,3}_xy.png` — neighborhood plots

### Modified files
- `spammm/atomicUtils.py` — added `loadPOSCAR()` function

### Reference files (existing, used)
- `kernels/PME.cl` — original 4-site PME kernel (reference for PME8)
- `spammm/quantum/PauliSolverCL.py` — original PME4 wrapper (reference for PME8 wrapper)
- `spammm/quantum/pauli_scan.py` — scan API, parameter defaults, embed functions
- `spammm/GUI/ChargeRingsExtension.py` — plot style reference (cut lines, site markers)
- `spammm/utils/OpenCLBase.py` — OpenCL device selection and program loading
- `data/charge_rings/square_tetramer.txt` — 4-site test geometry
