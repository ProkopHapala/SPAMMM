# Performance Benchmark: UFF/SPFF Relaxation

**Goal:** GUI "Relax" button should feel instant (< 0.5s for typical molecules, < 2s for large PAHs).

**Status:** Benchmarks + PTCDA/FAF fused work (2026-07-19). Geometry / charge tuning **unverified** pending USER review of `debug/test_relax_*` artifacts. Do not mark closed without USER confirmation.

---

## Measured results — `flat_1.mol2` (96 atoms, 66 C + 30 H)

Input: `/home/prokop/svec_triptacene/flat_1.mol2`  
Harness: `tests/test_relax_flat1.py` (`pytest tests/test_relax_flat1.py --develop -s`)  
GPU: NVIDIA RTX 3090, `PYOPENCL_CTX=0`

| Case | Mode | Steps | Time | E | fmax | planarity_C | Notes |
|------|------|-------|------|---|------|-------------|-------|
| Vacuum UFF | multi-kernel/step | 2000 | **2.07 s** | 9.99 | 1.01 | 0.31 | mean C–C 1.41, C–H 1.08 |
| Vacuum SPFF | `relax_batch` | 2000 | **0.80 s** | −424.21 | 2.62 | 0.36 | damp=0.9 |
| Vacuum SPFF | `relax_serial` WG=192 | 2000 | **0.0049 s** | −424.21 | 2.62 | 0.36 | **~163× vs batch**, identical E/fmax |
| NaCl GridFF + SPFF | batch + GridFF ex2 | 1500 | **0.93 s** | −413.71 | 2.95 | — | z_rel(min−top)=3.55 Å (stays above surface) |

Artifacts (L2 review):

- `debug/test_relax_flat1/uff_vacuum_{geometry.png,init_final.xyz,out}`
- `debug/test_relax_flat1/spff_batch_vacuum_*`
- `debug/test_relax_flat1/spff_serial_vacuum_*`
- `debug/test_relax_flat1/spff_substrate_{geometry.png,init_final.xyz,out}`

### Findings

1. **Serial local-memory kernel works for flat_1** after memory-safe sizing (`WG_SIZE=192`, `MAX_NVEC=192`, `MAX_NATOM=128`, `MAX_NNODE=96` ≈ 38 KB local — fits NVIDIA 48 KB). Naive WG=256 with 12×WG float4 arrays would be ~64 KB and fail.
2. **Parity**: serial vs batch on flat_1 give identical energy and fmax (and `tests/test_relax_serial.py` still all-pass).
3. **GUI damp default was too low**: `DEFAULT_DAMP=0.1` left SPFF at fmax≈12 after 2000 steps; `damp=0.9` → fmax≈2.6. Default updated to **0.9**.
4. **UFF fused** local/global kernels now exist (see PTCDA session below); flat_1 still timed on multi-kernel path in the table above.
5. **UFF+GridFF** not wired; substrate case in flat_1 harness is SPFF+GridFF only. `nonbonded_grid.cl` loads when `enable_nonbond=True`.

### Serial limits (SPFF)

| Cap | Value |
|-----|-------|
| `WG_SIZE` | 192 |
| `MAX_NVEC` | 192 |
| `MAX_NATOM` | 128 |
| `MAX_NNODE` | 96 |
| Requires | `nSystems==1`, no non-bonded / no GridFF in serial kernel |

UFF fused local caps: `MAX_UFF_ATOMS=128`, `MAX_UFF_ANGLES=256`, tile `MAX_UFF_ANG_TILE=128`.

---

## Session report — PTCDA + FAF + fused UFF/SPFF (2026-07-19)

Harness: `tests/test_relax_ptcda_faf.py` (`pytest tests/test_relax_ptcda_faf.py --develop -s`)  
Artifacts: `debug/test_relax_ptcda_faf/`  
GPU: NVIDIA RTX 3090

This section is the working log of adsorbate relaxation on **folded atom-type fit (FAF)** NaCl, and of making **fused multi-step UFF** scientifically usable.

### What we implemented

#### 1. FAF in fused multi-step kernels

- **SPFF:** `relax_nsteps_serial` / `relax_nsteps_global` evaluate FAF substrate forces each MD step (same coeff/basis layout as `getSurfFolded`).
- **UFF:** `relax_nsteps_local_UFF` / `relax_nsteps_global_UFF` same optional FAF path via `upload_folded_fit()`.
- Fit helper: `fit_folded_for_molecule(..., substrate_R_override=, q_override=)` + NPZ cache under `debug/test_relax_ptcda_faf/`.
- Test plots: 2×2 **xy + xz** with substrate + O–Na/Cl distance labels.

#### 2. Fused UFF intramolecular completeness

| Term | Multi-kernel | Fused local/global (now) |
|------|--------------|---------------------------|
| Bonds (neighs) | `evalBondsAndHNeigh_UFF` | yes |
| Angles (Fourier) | `evalAngles_UFF` | yes (fixed force formula — see bugs) |
| Dihedrals / torsions | `evalDihedrals_UFF` | **yes** — `uff_eval_dihedral_forces`, tiled gather |
| Inversions | `evalInversions_UFF` | **yes** — `uff_eval_inversion_forces`, tiled gather |
| FAF substrate | separate | optional `do_faf` |
| Non-bonded LJ/Coulomb | optional | **not** in fused path |
| 1–4 NB subtract on torsions | `SubNBTorsionFactor` (default 0) | not in fused (matches default 0) |

Design: parallel eval → per-interaction force slots → gather onto atoms (no atomics, no `iL==0` serial loops). Dihedrals/inversions use `MAX_UFF_ANG_TILE` tiling; local kernel keeps angle atom cache in `s_angA` and uses separate `s_tileA` so tiles do not corrupt angles across MD steps.

Python: `UFF_cl.relax_serial` / `relax_global` pass `nDOFs2=(ndihedrals, ninversions)` + `dihAtoms`/`dihParams`/`invAtoms`/`invParams`.

#### 3. Charge / Morse geometry knobs (current SSOT in test + xyz)

| Quantity | Value | Notes |
|----------|-------|-------|
| Substrate Na / Cl | q = ±1 | `data/substrates/NaCl_1x1_L3.xyz` |
| PTCDA O / H | q = −0.4 / +0.3 | `data/xyz/PTCDA.xyz` (neutral: 6×−0.4 + 8×+0.3 = 0) |
| Na RvdW (Morse) | 1.45 Å | R0(O–Na) ≈ 1.75+1.45 = 3.20 Å |
| Placement | `ROT_Z=18°`, `XY_SHIFT=(1.5,1.0)`, `Z_REL=3.0` | off-registry so lateral O→Na motion is visible |
| Cache | `ptcda_nacl_faf_NaR1.45_Qo0.4_qSub1.npz` | |

USER noted charges/well depth are **a bit strong** (“over-did it”) — easy to dial back later; not treated as closed.

### Bugs / problems we hit (and root causes)

#### A. UFF “horrendous shrink” (meanBL ~1.06) — **root cause found**

- **Symptom:** fused UFF collapsed PTCDA bond lengths; looked like broken topology.
- **Not the cause:** bond graph was fine (44 bonds, correct neighs) — see `ff_topology.out` / `.png`.
- **Cause:** fused `uff_eval_angle_forces` used a broken Fourier force (missing `sin`/`inv_s` and `1/l` scaling) vs `evalAngles_UFF`.
- **Fix:** align fused angle force with multi-kernel / `UFF.h::evalAngle_Prokop`. Vacuum/FAF meanBL recovered ~1.3–1.4.

#### B. Missing dihedrals / inversions in fused UFF

- **Symptom:** SPFF stayed flatter / more reasonable; fused UFF crumpled more under substrate pull.
- **Cause:** fused path only had bonds + angles (+ FAF); multi-kernel already had dihedrals (120) + inversions (72) for PTCDA.
- **Fix:** port Prokop dihedral/inversion formulas to fused kernels (from positions, no `hneigh`).
- **Parity (buckled PTCDA so dih+inv ≠ 0):** fused local vs `run_eval_step` multi-kernel: `F_max ≈ 8×10⁻⁶`, `F_rms ≈ 3×10⁻⁶`. Dih+inv contribution alone `F_max ≈ 4.2`. Planar geometry has near-zero dih/inv forces (expected).

#### C. Weak / confusing substrate response (“O on Cl”)

- **Tiny forces initially:** Morse R0(O–Na)≈3.2 Å with start Z_REL=3.5 put O near soft wall; brief `R_Na=0.75` experiment was unphysical — reverted; **R_Na≈1.45 OK**.
- **Electrostatics:** `COULOMB_CONST=14.3996` correct. FAF uses **Ewald**-style Coulomb (Madelung-screened), not bare pairwise.
- **Diagnostic:** `O_neg_potential_Na_vs_Cl.png` — O(q=−0.4) attracted to Na, repelled from Cl; FAF ≈ Morse(P+L)+Ewald within ~0.001 eV.
- **3D `d(O–Cl) < d(O–Na)`** can be geometric (O between ions at low z), not true Cl binding — labels on PNG distinguish.

#### D. Charge strength

- Strengthened substrate to ±1 and O/H to −0.4/+0.3 so FAF response is visible.
- SPFF result accepted as **reasonable** (clear O bend); USER: slightly **over-strong**, easy to correct later.

### Latest measured numbers (PTCDA + FAF, 8000 steps, RTX 3090)

| tag | t_s | E | fmax | z_min | meanBL | dONa3 | dOCdz |
|-----|-----|---|------|-------|--------|-------|-------|
| spff_serial_faf | 0.072 | −78.67 | 2.79 | −1.54 | 1.340 | 2.91 | **−0.630** |
| spff_global_faf | 0.069 | −78.67 | 2.79 | −1.54 | 1.340 | 2.91 | −0.630 |
| uff_local_faf | 0.152 | 117.12 | 6.72 | −2.05 | 1.356 | 2.29 | **−1.085** |
| uff_global_faf | 0.144 | 117.12 | 6.72 | −2.05 | 1.356 | 2.29 | −1.085 |

Review paths:

- `debug/test_relax_ptcda_faf/speed_summary.out`
- `debug/test_relax_ptcda_faf/spff_serial_faf_geometry.png` (xy+xz)
- `debug/test_relax_ptcda_faf/uff_local_faf_geometry.png`
- `debug/test_relax_ptcda_faf/O_neg_potential_Na_vs_Cl.png`
- `debug/test_relax_ptcda_faf/ff_topology.out` / `.png`

**Interpretation (unverified):** With dihedrals+inversions present and force-parity-checked, UFF still bends more (`dOCdz≈−1.09`) than SPFF (`−0.63`) and ends at higher `fmax`. Likely **softer UFF torsion/inversion barriers vs SPFF**, and/or incomplete convergence (`fmax≈6.7`), not missing fused terms. Bonds remain healthy (no collapse).

### Code touchpoints

| Area | Files |
|------|-------|
| Fused UFF kernels | `kernels/UFF.cl` — `relax_nsteps_{local,global}_UFF`, `uff_eval_{angle,dihedral,inversion}_forces` |
| UFF host | `spammm/forcefields/UFF_cl.py` — `relax_serial` / `relax_global`, FAF upload |
| Fused SPFF + FAF | `kernels/SPFF.cl`, `spammm/forcefields/SPFF_cl.py` |
| FAF fit / substrate | `spammm/surfaces/FoldedRigid.py` |
| Harness | `tests/test_relax_ptcda_faf.py` |
| Charges / geometry | `data/xyz/PTCDA.xyz`, `data/substrates/NaCl_1x1_L3.xyz` |

---

## Current relaxation paths

### GUI path (`FFExtension._on_relax`)

`spammm/GUI/FFExtension.py`

1. `_ensure_built(window)` → `FFController.build_ff()` (if not built)
2. `ctrl.relax_until_converged(max_steps, dt, damp, callback, batch_size)`
3. Callback `_cb()` runs **every batch**: GPU→host sync, AtomicGraph update, Vispy refresh, `processEvents`

### GPU paths (`FFController.relax_n` / `relax_until_converged`)

- **`relax_serial` (SPFF):** Single-kernel local-memory, WG=192. Runs N steps in one kernel call. ~160× faster than batch on flat_1. Caps above; no molecule–molecule non-bonded; **FAF optional** when fit uploaded.
- **`relax_batch`:** Per-step kernel launches; sync at end (or each GUI callback).
- **UFF fused:** `UFF_cl.relax_serial` / `relax_global` — bonds+angles+dihedrals+inversions (+ optional FAF). Not yet the default GUI path (`FFController` UFF combo still incomplete).

### Remaining suspected bottlenecks (GUI)

1. **GUI callback overhead** every batch (`get_state` + refresh + `processEvents`)
2. **Serial unavailable** when non-bonded / GridFF / nvecs>192
3. Kernel launch overhead in batch mode for large systems

## Optimization targets (still open)

1. Reduce GUI callback frequency (refresh every N batches or only at end)
2. Optional WG=256 with even tighter local packing if needed for larger PAHs
3. Serial / fused kernel + GridFF (not started); bench FAF fused vs GridFF batch
4. Wire UFF into `FFController` GUI combo (still `NotImplementedError`)

## Fused kernels completeness — checklist

Status: **FAF + UFF torsions implemented**; geometry/charge dial-back and SPFF π-term audit **open / unverified**.

### Done (pending USER confirmation of plots / physics)

- [x] FAF in SPFF `relax_nsteps_serial` / `relax_nsteps_global` and UFF fused local/global
- [x] Harness `tests/test_relax_ptcda_faf.py` + artifacts under `debug/test_relax_ptcda_faf/`
- [x] UFF fused **Fourier angle force** fixed (parity vs multi-kernel)
- [x] UFF fused **dihedrals (torsions)** + **inversions** (tiled gather; force parity on buckled PTCDA)
- [ ] USER sign-off: SPFF/UFF FAF geometries and charge strength OK (or dial back)

### Still open

#### Physics / parameters (PTCDA+FAF)

- [ ] Dial back O/substrate charges and/or Morse well if USER confirms “over-did it”
- [ ] Understand / reduce UFF vs SPFF `dOCdz` gap (parameter softness vs need more steps / lower fmax)
- [ ] Lateral registry: only ~2/6 O closer to Na in 3D on last run — may improve after charge dial-back or longer relax
- [ ] Optional L0 asserts: fused UFF force parity vs multi-kernel on small buckled mol

#### SPFF fused (`kernels/SPFF.cl`)

- [ ] Audit / complete **π–π** (`evalPiAling`) and **π–σ** (`evalAngCos` / Ksp) in local and global fused loops; parity with `getSPFFf4` / `relax_batch`
- [ ] Document which terms are in serial vs global vs batch; L0 parity asserts if any term missing

#### Substrate + non-covalent (fused path)

- [ ] Wire **non-covalent** molecule–molecule (LJ/Coulomb exclusions) into fused kernels (not needed for single adsorbate)
- [ ] Optional: GridFF in fused path; bench FAF fused vs GridFF batch
- [ ] flat_1: re-bench **UFF fused** vacuum timing vs multi-kernel (table above still multi-kernel)

See also: `doc/ToDo/ToDo.agents.md` (Soon items), `doc/ARCHITECTURE_ROADMAP.md` §5, `doc/GUI_FF_Relaxation.md` § non-bonded gaps.

## Success criteria

- Benzene (12 atoms): < 0.1s to convergence (fmax < 0.05)
- Coronene (24 atoms): < 0.3s
- Pentacene (36 atoms): < 0.5s
- Large PAH (~100 atoms): < 2s — **flat_1 serial vacuum: 0.005 s / 2000 steps (GPU only)**
- GUI remains responsive during relaxation
- Serial kernel used for molecules with nvecs≤192, nnode≤96
- Fused UFF includes bonds+angles+dihedrals+inversions with multi-kernel force parity on non-planar test geometries

## References

- `spammm/forcefields/FFController.py` — relax_n, `_can_use_serial`
- `spammm/forcefields/SPFF_cl.py` — `relax_serial` / `relax_batch`, `SERIAL_*` caps
- `kernels/SPFF.cl` — `relax_nsteps_serial`
- `spammm/forcefields/UFF_cl.py` — UFF fused + multi-kernel
- `kernels/UFF.cl` — `relax_nsteps_{local,global}_UFF`
- `spammm/surfaces/FoldedRigid.py` — FAF fit / eval
- `spammm/GUI/FFExtension.py` — GUI wiring
- `tests/test_relax_flat1.py` — flat_1 systematic benchmarks
- `tests/test_relax_serial.py` — serial vs batch parity
- `tests/test_relax_ptcda_faf.py` — PTCDA + FAF fused pipelines
