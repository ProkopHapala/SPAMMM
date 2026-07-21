# Report: PTCDA @ NaCl FAF — GPU Rigid-Body Imaging (Newton vs FIRE)

**Date:** 2026-07-21  
**Status:** investigating (not marked fixed)  
**Durable student-facing report:** [`doc/Reports/PTCDA_NaCl_Rigid_Newton_FIRE_Relaxation.md`](../Reports/PTCDA_NaCl_Rigid_Newton_FIRE_Relaxation.md) — supersedes the early solver-performance conclusions below.
**Hardware:** NVIDIA GeForce RTX 3090  
**Driver script:** `tests/testplot_ptcda_nacl_replicas.py`  
**Artifacts:** `debug/testplot_ptcda_nacl_replicas/`

---

## 1. Goal

Evaluate **pure GPU** rigid-body pose optimization of **PTCDA** on a **FoldedAtomicFunctions (FAF)** NaCl(100) substrate for AFM-like imaging throughput:

- Scan grid: **256×256** pixels over a **NaCl 2×2** cell (**8×8 Å**, PBC inherent to FAF).
- Start **flat** (identity quaternion).
- Compare **Newton / Levenberg–Marquardt** vs **FIRE**.
- Diagnose non-convergence: pin one distant anhydride O, log peak force, inspect failing-pixel trajectories.

This is **not** tip-sample FDBM AFM; it is molecule-on-substrate rigid relaxation under an analytic folded basis, meant as a stress test of the imaging path (`*_replicas_*` kernels).

---

## 2. Physics & numerical setup

### 2.1 Substrate (FAF)

| Item | Value |
|------|--------|
| Substrate XYZ | `data/substrates/NaCl_1x1_L3.xyz` |
| Lattice | \(a = 4\) Å (orthogonal); scan = 2×2 → \(L = 8\) Å |
| Surface top | `Z_SURF_TOP = −3.25` Å |
| Fit | `fit_folded_for_molecule` (pauli+london+coulomb / Ewald2D), cached `ptcda_nacl_faf.npz` |
| Basis | `ntypes=3`, `nbasis=112`, `folded_lvec2d = (4,0,0,4)` |

FAF energy for atom type \(t\) at world \((x,y,z)\):

\[
E = \sum_b c_{t,b}\, \cos(2\pi k_u u)\,\cos(2\pi k_v v)\,\exp(-\alpha\max(0,z-z_0))
\]

with fractional \((u,v)\) from the 2D lattice (PBC via wrap).

### 2.2 Molecule

| Item | Value |
|------|--------|
| XYZ | `data/xyz/PTCDA.xyz` (38 atoms, planar) |
| Lateral size | ~11.7 × 6.9 Å |
| Pin atom | **O index 24** (anhydride O at one end; same as `test_manipulation_ptcda_nacl`) |
| Pin height | \(z_\mathrm{pin} = Z_\mathrm{SURF\_TOP} + 6.0 = 2.75\) Å absolute |
| Spring | \(k = 10\) eV/Å² (`anchors[ia].w > 0`) |

With identity orientation, COM is placed so the pin atom starts **exactly** on the tip:

\[
\mathbf{r}_\mathrm{COM} = \mathbf{r}_\mathrm{tip} - \mathbf{r}^\mathrm{body}_\mathrm{pin}
\]

Scan \((x,y)\) is the **tip / pin target**, not free COM.

### 2.3 Optimizers (GPU)

Documented in `kernels/rigid.cl` header. Short form:

| Kernel | Role | Parallelism |
|--------|------|-------------|
| `rigid_body_folded_newton_replicas_kernel` | Pure Newton + LM trust | 1 thread / pixel, WG=128; H 6×6 **private** |
| `rigid_body_folded_replicas_kernel` | MD / FIRE | 1 thread / pixel, WG=128 |

**Newton does not diagonalize \(H\)** (no Jacobi). Per iteration:

1. \(G = [F_\mathrm{world},\,\tau_\mathrm{body}] = -\nabla E\)
2. Forward FD Hessian on \(u = (\Delta x,\,\Delta\theta_\mathrm{body})\), then symmetrize
3. Solve \((H + \lambda I)\Delta = G\) by dense **6×6 Gaussian elimination + partial pivoting** (`rigid_solve6_lm`)
4. Trust-cap \(\|\Delta\|\); accept only if energy decreases

Parallelism is over **replicas** (and atoms inside each force eval), **not** inside the 6×6 solve (~216 flops).

| Parameter | Newton | FIRE |
|-----------|--------|------|
| Budget | **`NEWTON_NITER = 40`** | `FIRE_NITER = 2000` |
| Step | trust / λ adapt in-kernel | `dt = 0.02`, damp0 = 0.1 |
| Stop | \(\|F\|^2+\|\tau\|^2 < f_\mathrm{tol}^2+t_\mathrm{tol}^2\) | same early-exit when FIRE (`RIGID_FIRE_F2SAFE`) |
| Tol used in maps | \(f_\mathrm{tol}=t_\mathrm{tol}=10^{-4}\) | same |

Diagnostics written by kernels:

- `body_torque.w` → iterations actually used  
- `vposs.w` → **peak \|F\| over the trajectory** (`Fmax_traj`)

Python helpers: `setup_rigid_folded_replicas(..., pin_atom_idx=...)`, `run_folded_newton_replicas`, `run_folded_replicas(..., fire=True)`.

---

## 3. Experiments performed

### Exp A — Free molecule (no pin), first attempt

- COM on scan grid, \(z_\mathrm{init}=3\) Å above surface, flat.
- Newton: **93%** converged, clear NaCl-periodic E / COM maps, ~0.68 s (~9.7×10⁴ mol/s).
- FIRE: **99.9%** “converged” but **washed-out** E/COM (molecules laterally drift into nearby basins).

**Conclusion:** free 6-DOF imaging is not tip-like; FIRE especially destroys lateral contrast via COM drift.

### Exp B — Pinned O (AFM-like), main results

- Pin O#24 at tip \((x,y,z_\mathrm{pin})\), \(k=10\).
- Same 256×256 / 8×8 Å grid.

### Exp C — Fail-pixel trajectories

- From Newton fail mask, pick 5 tips in densest failure neighborhoods.
- Re-relax each as **single body**, recording:
  - Newton: every iteration (budget 40; often finishes earlier)
  - FIRE: first **100** steps only

---

## 4. Quantitative results (Exp B — pinned)

| Metric | Newton | FIRE |
|--------|--------|------|
| Wall time (RTX 3090) | **1.09 s** | **5.49 s** |
| Throughput | ~6.0×10⁴ mol/s | ~1.2×10⁴ mol/s |
| frac_conv (\(\|F\|,\|\tau\|<10^{-4}\)) | **0.739** | **0.846** |
| iters median / max | **33 / 40** | **1085 / 2000** |
| E median | −0.271 eV | −0.271 eV |
| COM height median (above surf) | ~4.52 Å | ~4.52 Å |
| Fmax_traj median / max | 0.246 / **0.623** eV/Å | 0.220 / **0.460** eV/Å |

**Crash check:** peak forces are **O(0.2–0.6) eV/Å**, not catastrophic spikes. Hard substrate penetration is **not** the dominant failure mode at \(z_\mathrm{pin}=6\) Å. `Fmax_traj` maps are structured (substrate periodicity), consistent with contact / torque build-up during tilt, not random blow-ups.

### Map phenomenology (pinned)

- **Energy:** clear vertical-band / checker-like contrast (weaker binding than free adsorbate at lower z).
- **Final \|F\| / \|τ\|:** mostly dark (near zero) with sparse / island failures.
- **COM_z / tilt:** molecule **hangs from the tip** — typical tilt ~15–30°, COM below pin height (~4.5 Å vs pin at 6 Å relative).
- **converged / iters:** failures are **spatially clustered** (periodic islands), not pure salt-and-pepper — though Newton also has finer noise at basin boundaries.
- FIRE failures form larger coherent “teardrop / lobe” regions that hit the 2000-step cap.

Primary overview plots:

- `newton_maps.png`, `fire_maps.png`
- `newton_Fmax_traj.png`, `fire_Fmax_traj.png`
- `summary.out`

---

## 5. Trajectory diagnostics (Exp C)

### 5.1 Selected tips (dense Newton-fail regions)

| ID | tip (x, y) [Å] | fail-neighbor density |
|----|----------------|------------------------|
| P0 | (6.44, 4.47) | 361 |
| P1 | (2.44, 4.22) | 360 |
| P2 | (6.44, 7.97) | 315 |
| P3 | (2.44, 7.97) | 310 |
| P4 | (2.44, 0.56) | 300 |

See `fail_pick_map.png`.

### 5.2 Newton single-pixel re-runs (`niter=1` × up to 40)

| ID | Newton steps | final E [eV] | final \|F\| | final \|τ\| | final tilt [°] |
|----|--------------|--------------|------------|------------|----------------|
| P0 | 19 | −0.268 | 6.2e-6 | 3.2e-5 | **19.6** |
| P1 | 19 | −0.113 | 2.1e-5 | 3.2e-5 | **64.3** |
| P2 | 40 | −0.121 | 3.4e-5 | 1.0e-4 | **62.4** |
| P3 | 18 | −0.132 | 3.1e-6 | 6.9e-6 | **13.4** |
| P4 | 23 | −0.271 | 8.8e-6 | 1.0e-5 | **19.7** |

**Typical Newton story (P0):** flat start → hinge about pinned O → COM drops ~6→4.5 Å → tilt settles ~20° → mid-trajectory **\|F\|/\|τ\| spike** then collapse under tolerance. XZ snapshots (`traj_newton_P*_xz.png`) show the molecule rotating downward about the cyan tip marker.

**Important caveat — batch vs step-by-step Newton:**  
Imaging used **one kernel launch with `niter=40`** (continuous trust/λ state). Trajectory recording used **`run_folded_newton(niter=1)` repeatedly**, which **resets** trust/λ each launch. Several pixels marked failed on the batch map **do converge** in the step-by-step re-run. Therefore:

- Some map “failures” may be **optimizer-path / budget** artifacts of the continuous LM loop, not unique unphysical states.
- Conversely, continuous LM can also get stuck where restarted 1-step Newton escapes (or vice versa).  
Treat batch frac_conv and single-pixel traj as **related but not identical** experiments.

### 5.3 FIRE first 100 steps (same tips)

At physical mass (\(m \sim N_\mathrm{atoms} \approx 38\)) and `dt=0.02`, the first 100 FIRE steps barely move PTCDA (ΔCOM_z ~10⁻³ Å, tilt ~0). Residual \|F\| ~ 10⁻², \|τ\| **increases**.  

**Interpretation:** 100 steps is too short for this mass/dt (batch FIRE needed ~10³ steps median). Also, energy read from `body_force.w` after the MD eval path is unreliable (often 0); prefer `sum(atom_positions[:,3])` for energy in MD diagnostics.

Plots: `traj_fire_P*.png`.

---

## 6. Interpretation

### What works

1. **GPU replicas path is fast enough for imaging** (~10⁴–10⁵ mol/s on 3090 for PTCDA+FAF).
2. **Newton preserves lateral contrast** better than free FIRE; pinned setup is the correct AFM-like constraint.
3. **Forces are not exploding** at \(z_\mathrm{pin}=6\) Å — “vigorous crash” hypothesis is **weak** for this height (`Fmax_traj` max &lt; 1 eV/Å).
4. Physics of the hanging molecule (tilt about tip O, COM drop) is **geometrically expected** for a long planar adsorbate on a soft periodic potential.

### What is wrong / concerning

1. **Pinned frac_conv is only ~74% (Newton) / ~85% (FIRE)** — not production-ready for clean images without masking or more iters.
2. Failures form **systematic islands** correlated with substrate periodicity and with high mid-trajectory torque / extreme final tilt (60°+ at P1/P2).
3. Long lever arm (PTCDA ~12 Å) + stiff pin spring (\(k=10\)) → **ill-conditioned 6-DOF problem** (small tip residuals ↔ large torques).
4. Newton map noise + incomplete convergence may mix: (a) true multi-basin / near-singular Hessian sites, (b) LM trust failures within 40 iters, (c) batch vs host recording path differences.
5. FIRE needs either **more steps**, **larger dt**, or **reduced mass** for fair short-trajectory comparison with Newton.

### Not claimed fixed

No change in this work is asserted to “fix” convergence. Status remains **investigating**.

---

## 7. How to reproduce

```bash
# Full maps: Newton then FIRE (pinned O)
python tests/testplot_ptcda_nacl_replicas.py

# Newton fail mask + per-pixel trajectories (skip re-plotting FIRE maps)
python tests/testplot_ptcda_nacl_replicas.py --traj-only
```

Requires NVIDIA OpenCL visible (agent Shell: unrestricted / `all`).  
Fit cache: `debug/testplot_ptcda_nacl_replicas/ptcda_nacl_faf.npz` (fallback from `debug/test_relax_ptcda_faf/`).

### Key code

| Path | Role |
|------|------|
| `kernels/rigid.cl` | Folded MD/FIRE/Newton kernels; `folded_FT_*`, `rigid_solve6_lm`; iters + Fmax in `.w` channels |
| `spammm/surfaces/FoldedRigid.py` | `setup_rigid_folded_replicas` (optional pin), fit/load |
| `spammm/forcefields/RigidBodyDynamics.py` | `run_folded_newton_replicas`, `run_folded_replicas` |
| `tests/testplot_ptcda_nacl_replicas.py` | Imaging demo + fail-pixel traj |

---

## 8. Artifact index

```
debug/testplot_ptcda_nacl_replicas/
  summary.out                 # pinned map scalars
  fail_traj.out               # P0–P4 trajectory endpoints
  fail_pick_map.png           # fail density + selected tips
  newton_maps.png / fire_maps.png
  newton_E.png, newton_Fmag.png, newton_Fmax_traj.png
  fire_E.png, fire_Fmag.png, fire_Fmax_traj.png
  traj_newton_P{0..4}.png     # E,|F|,|τ|,COM_z,tilt vs iter
  traj_newton_P{0..4}_xz.png  # XZ snapshots (hinge about tip)
  traj_fire_P{0..4}.png
  traj_fire_P{0..4}_xz.png
  ptcda_nacl_faf.npz          # FAF fit cache
```

---

## 9. Suggested next steps (not done)

1. **Align trajectory recording with batch Newton** — record inside one `niter=40` launch (or persist trust/λ) so fail pixels match the imaging kernel path.
2. **Sweep** \(z_\mathrm{pin}\), \(k_\mathrm{spring}\), and Newton `niter` (e.g. 80–100) on the same grid; plot frac_conv.
3. **Mask or inpaint** non-converged pixels for image products; report frac_conv as quality metric.
4. FIRE: raise `dt` or lower effective mass for short diagnostics; compare to Newton at equal wall-time.
5. Dump `.xyz` movie for one 60°-tilt pixel (P1/P2) if geometry looks unphysical vs expected adsorption.
6. Optional: constrain only **z + orientation** or pin **xy of COM** instead of a tip O — different imaging protocols.

---

## 10. Bottom line

Pinned PTCDA@NaCl FAF imaging on GPU works at **imaging-relevant speed**. Newton gives structured, tip-local contrast; free FIRE does not. At tip height 6 Å, molecules **tilt about the pinned O** rather than violently crash (`Fmax_traj ≲ 0.6` eV/Å). **Convergence is incomplete (~74% Newton / ~85% FIRE)** with systematic failure islands and occasional extreme tilts — enough to distrust raw maps without masking or further optimizer/constraint tuning. Single-pixel Newton trajectories show physically plausible hinge relaxation, but **batch LM (40 iters) and step-by-step Newton are not identical**, which complicates attributing every failed pixel to a single root cause.

---

## 11. Follow-up solver audit (2026-07-21, awaiting user confirmation)

**Status remains investigating.** The changes below have GPU verification but are not marked fixed/resolved until user review.

### Root causes found

1. Newton trust adaptation was inverted: an accepted boundary-limited step did not grow the trust radius, while a small unconstrained step did. LM damping also remained high after most accepted steps.
2. Rejected Newton trials could leave `lambda` below the useful recovery range; resetting the kernel happened to restore `lambda0`, explaining why one-step relaunches performed better.
3. Kernel early exit used a combined force+torque norm, inconsistent with the separately reported `|F|` and `|tau|` tolerances.
4. `setup_rigid_folded(mass_trans=...)` ignored its argument, and the replica setup changed translational mass without scaling rotational inertia. The body/world torque transform itself passed finite-difference parity.

### Corrections under verification

- Accepted boundary steps now grow trust; every accepted step reduces LM damping.
- Rejection recovery returns to at least `lambda0`, with one bounded on-stall reset inside the replica kernel.
- Near float32 energy ties are accepted only when energy differs by at most summation noise and the force/torque residual drops by at least 1%.
- Folded setup now reuses the established scaling `I_relax = I*(m_eff/mtot)`, `Iinv_relax = Iinv*(mtot/m_eff)`.
- FIRE and Newton early exits now require force and torque to pass separately.

### NVIDIA RTX 3090 verification

| Configuration | frac_conv | wall | Median iterations | Notes |
|---|---:|---:|---:|---|
| Newton, old report, 40 iterations | 0.739 | 1.09 s | 33 | continuous trust state |
| Newton, corrected, 40 iterations | 0.999 | 0.84 s | 14 | same 256x256 pinned scan |
| Newton, corrected, 80 iterations | 1.000 | 0.85 s | 14 | max `|F|<1e-4`, max `|tau|<1e-4` |
| FIRE, old report, physical effective mass 38, 2000 steps | 0.846 | 5.49 s | 1085 | old combined early exit |
| FIRE, corrected mass scaling, effective mass 4, 4000 steps | 0.999 | 7.17 s | 593 | stable sweep choice; slower than Newton |

For the former P0 failure at tip `(6.44, 4.47)` A, continuous Newton now reaches `|F|=6.4e-6`, `|tau|=4.6e-5`, `E=-0.26792848` eV in 16 iterations without relaunching. The focused regression is `test_ptcda_pinned_newton_single_replica` in `tests/test_folded_relax.py`.

The existing rigid-body physics audit reports force finite-difference error `1.27e-7`, torque-frame error `1.68e-5`, and free-body angular-momentum/rotational-energy drift about `1.8e-4`; this supports the current body-frame inertia and quaternion transform. FIRE performance was primarily an effective-mass/setup issue, not an inertia-frame transform error.
