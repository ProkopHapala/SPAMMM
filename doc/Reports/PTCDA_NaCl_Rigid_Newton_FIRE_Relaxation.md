---
type: Report
title: Stable rigid-body relaxation for PTCDA on NaCl — damped Newton and FIRE
description: Numerical diagnosis, solver design, validation, and practical guidance for pinned rigid-body imaging with the folded NaCl force field.
tags: [rigid-body, Newton, Levenberg-Marquardt, FIRE, OpenCL, PTCDA, NaCl, imaging]
timestamp: 2026-07-21
---

# Stable rigid-body relaxation for PTCDA on NaCl

**Status:** results for USER review; investigating, not marked fixed  
**Hardware:** NVIDIA GeForce RTX 3090  
**Imaging driver:** [`tests/testplot_ptcda_nacl_replicas.py`](../../tests/testplot_ptcda_nacl_replicas.py)  
**Main kernel:** [`kernels/rigid.cl`](../../kernels/rigid.cl)  
**Artifacts:** [`debug/testplot_ptcda_nacl_replicas/`](../../debug/testplot_ptcda_nacl_replicas/)

## Verdict

Rigid PTCDA relaxation on the folded NaCl potential can produce smooth, mutually consistent FIRE and Newton images, but ordinary aggressive Newton is not suitable from the high, flat initial pose. It converges its residuals while selecting spatially scattered attraction basins, which creates discontinuous energy, COM-height, tilt, and constraint-force maps.

The effective Newton solution is a two-stage Levenberg–Marquardt procedure:

1. use float32-safe centered Hessian differences and persistent damping so the early trajectory behaves approximately like gradient descent;
2. after a CPU convergence check, reduce damping and let Newton finish the remaining pixels from their persistent poses.

For the 256×256 PTCDA scan this reduced the 99th-percentile neighboring-pixel discontinuities by one to two orders of magnitude. The resulting fields resemble FIRE, which is important evidence that both algorithms are following the same physical basin structure rather than generating solver-specific image texture. Agreement between optimizers is still not proof that the folded force field itself is physically exact.

## 1. Test problem

| Item | Value |
|---|---|
| Molecule | PTCDA, 38 atoms, `data/xyz/PTCDA.xyz` |
| Surface | NaCl(100), folded analytic Pauli + London + Ewald2D Coulomb fit |
| Scan | 256×256 pixels over 8×8 Å (2×2 NaCl cells) |
| Constraint | anhydride O atom 24 attached to the tip by a harmonic spring |
| Tip height | 6.0 Å above `Z_SURF_TOP` |
| Spring stiffness | 10 eV/Å² |
| Degrees of freedom | COM translation plus body-frame rotation, 6 total |
| Strict stopping test | `|F_total| < 1e-4` and `|tau_total| < 1e-4` separately |

Each pixel is an independent replica. The scan coordinate is the spring-anchor position, not the free molecular COM. The initial COM is calculated so the pinned atom starts on the anchor:

\[
\mathbf r_\mathrm{COM}=\mathbf r_\mathrm{anchor}-R(q)\mathbf r^\mathrm{body}_\mathrm{pin}.
\]

Using `R(q)` is essential. The earlier initializer assumed the identity quaternion even when a tilted initial quaternion was supplied. The corrected setup aligns a non-identity start to within `4.8e-7 Å` in the tested float32 geometry.

## 2. Observables: residual force is not the AFM signal

After a successful relaxation, total force and torque should be nearly zero. Plotting total `Fz` therefore mainly visualizes convergence error, not the force carried by the external constraint.

For pin displacement

\[
\mathbf d=\mathbf r_\mathrm{pin}-\mathbf r_\mathrm{anchor},
\]

the force of the spring on the molecule is

\[
\mathbf F_\mathrm{spring\rightarrow mol}=-k\mathbf d,
\]

and the equal-and-opposite reaction exerted on the external tip/constraint is

\[
\mathbf F_\mathrm{constraint}=+k\mathbf d.
\]

The diagnostic maps now include `Fconstraint_z` and `|Fconstraint|` as well as total residual force. The sign in the plots is the force on the external constraint. Energy includes both folded substrate energy and harmonic spring energy.

## 3. Why ordinary Newton produced noisy images

### 3.1 Newton step used here

Define generalized coordinates and generalized downhill force

\[
u=(x,y,z,\theta_x,\theta_y,\theta_z),\qquad G=(F_x,F_y,F_z,\tau_x,\tau_y,\tau_z)=-\nabla_u E.
\]

The Levenberg–Marquardt step solves

\[
(H+\lambda I)\Delta u=G
\]

and caps `|Delta u|` by the trust radius. A trial pose is accepted only when energy decreases, apart from a small float32 energy-tie allowance that also requires a lower residual.

Near a well-conditioned minimum, small `lambda` gives fast quadratic-like Newton convergence. Far from a minimum, the Hessian may be indefinite and its eigenvectors can point across attraction-basin boundaries. Energy descent alone does not guarantee that a finite multidimensional Newton step follows the same basin as continuous gradient flow.

### 3.2 Direct boundary trace

The worst 64×64 boundary pair was separated by only 0.125 Å:

| Pixel | Final energy | COM height | Tilt | Energy increases |
|---|---:|---:|---:|---:|
| `(1.250, 4.500)` | −0.1373 eV | 5.242 Å | 38.31° | 0 |
| `(1.250, 4.625)` | −0.3790 eV | 4.338 Å | 17.04° | 0 |

Both trajectories decreased energy monotonically and met strict residual tolerances, but they reached distant minima. Thus the original noisy map was not simply a collection of unconverged pixels. It was solver-dependent basin selection.

Artifacts: [`newton_worst_boundary_trace.out`](../../debug/testplot_ptcda_nacl_replicas/newton_worst_boundary_trace.out), [`newton_boundary_P0_traj.png`](../../debug/testplot_ptcda_nacl_replicas/newton_boundary_P0_traj.png), and [`newton_boundary_P1_traj.png`](../../debug/testplot_ptcda_nacl_replicas/newton_boundary_P1_traj.png).

### 3.3 Initial orientation was not the main cure

Flat and correctly anchored ±2° starts were compared at 64×64. A smaller trust radius helped common jumps, but all three starts retained rare jumps around 50° tilt and 1.5 Å COM height. Initial orientation can break a symmetry and bias a branch, but a fixed arbitrary tilt does not guarantee local gradient-flow behavior and can introduce its own directional bias.

Artifact: [`newton64_initial_condition_comparison.out`](../../debug/testplot_ptcda_nacl_replicas/newton64_initial_condition_comparison.out).

## 4. Float32 Hessian differentiation

The original Hessian used forward differences with `eps_t=eps_r=1e-4`. This is too close to cancellation for forces accumulated in float32. The imaging replica kernel now uses centered differences:

\[
H_{ij}=-\frac{G_i(u+\epsilon_j e_j)-G_i(u-\epsilon_j e_j)}{2\epsilon_j}.
\]

The selected default is

```text
eps_t = 0.1 Å
eps_r = 0.1 rad
```

This deliberately gives up small-step truncation accuracy to retain roughly two useful decimal orders after float32 subtraction. For this relaxation problem a smooth, approximate curvature is more useful than a nominally precise but noisy Hessian.

Caveats:

- `0.1 rad` is about 5.7°, so the rotational Hessian is an averaged local curvature, not an infinitesimal one.
- Translation and rotation have different dimensions. A future solver should expose a rotational coordinate scale rather than treating Å and radians identically in `H + lambda I` and the trust norm.
- The many-replica imaging Newton kernel uses centered differences. The single-body workgroup Newton kernel still uses forward differences; do not assume bitwise or trajectory parity between those two implementations.
- A central 6×6 Hessian needs 12 perturbed force/torque evaluations per Newton iteration, before trial evaluations.

## 5. Persistent LM damping: the central stabilization

For large damping,

\[
\Delta u=(H+\lambda I)^{-1}G\approx G/\lambda,
\]

so damped Newton approaches a scaled steepest-descent step. It therefore tends to remain in the local gradient-flow basin. The trust radius still limits the combined translation/rotation step.

The previous code treated `lambda0` only as a starting value. Every accepted step multiplied `lambda` by 0.3 until damping was nearly absent. Consequently, even `lambda0=10` became aggressive Newton after a few accepted steps. Resetting trust/LM state on every host launch appeared to improve the map because it accidentally restored damping repeatedly.

The kernel now treats `lambda0` as both the initial damping and its lower bound. Trust, damping, recovery state, FIRE state, pose, and velocities remain persistent across kernel launches.

### Damping tradeoff measured at 64×64

| LM floor | Strict convergence after 80 | Max neighbor energy jump | Max tilt jump | Interpretation |
|---:|---:|---:|---:|---|
| 0.1 | 99.54% | 0.233 eV | 60.6° | too Newton-like; scattered basin switching |
| 1.0 | 70.19% | 0.0507 eV | 12.9° | smooth basin following, slower convergence |
| 10.0 | 0% | 0.0010 eV | 0.54° | extremely smooth but barely moves in 80 steps |

This is not a paradox: stronger damping improves path locality while reducing step size and asymptotic speed.

## 6. Recommended staged Newton workflow

The tested compromise is:

```text
Stage 1: 80 iterations, centered eps=0.1, trust=0.1, lambda floor=1.0
CPU check: evaluate separate |F| and |tau| convergence masks
Stage 2: if any pixel is unfinished, continue the same persistent state for
         up to 80 iterations with lambda floor=0.01
```

Converged pixels enter the second kernel launch but exit immediately. Compacting only unfinished replicas could reduce wasted scheduling and memory traffic, but it is not implemented; the extra index management is not justified until profiling shows that it matters.

The first stage chooses a basin smoothly. The second stage accelerates local convergence after the pose is already close to that basin. This is still Newton/LM, not FIRE: it has no velocity or inertial trajectory.

## 7. Full 256×256 results

### 7.1 Spatial continuity

| Neighbor-jump metric | Aggressive Newton | Staged damped Newton | Improvement |
|---|---:|---:|---:|
| Energy p99 | 0.1148 eV | **0.00659 eV** | 17.4× |
| COM-height p99 | 0.222 Å | **0.0281 Å** | 7.9× |
| Tilt p99 | 17.46° | **1.28°** | 13.6× |
| Constraint-Fz p99 | 0.0206 eV/Å | **0.00443 eV/Å** | 4.7× |

Staged Newton reached strict convergence on 99.7467% of pixels: 166 of 65,536 remained above at least one `1e-4` threshold after the two 80-step stages. The remaining maximum jumps were `0.064 eV`, `0.349 Å`, and `14.5°`. They are sparse line-like boundaries rather than the former area-filling fractal texture, but they still require human physical review.

Main artifacts:

- [`newton256_staged_damped80_finish80_maps.png`](../../debug/testplot_ptcda_nacl_replicas/newton256_staged_damped80_finish80_maps.png) — E, residuals, constraint force, COM, tilt, convergence, iterations
- [`newton256_staged_damped80_finish80_E.png`](../../debug/testplot_ptcda_nacl_replicas/newton256_staged_damped80_finish80_E.png) — energy map
- [`newton256_staged_damped80_finish80.npz`](../../debug/testplot_ptcda_nacl_replicas/newton256_staged_damped80_finish80.npz) — raw maps
- [`newton256_staged_damped80_finish80.out`](../../debug/testplot_ptcda_nacl_replicas/newton256_staged_damped80_finish80.out) — exact numerical summary

### 7.2 Timing

| Stage | RTX 3090 wall time | Comment |
|---|---:|---|
| Damped Newton 80 | 3.83 s | all 65,536 pixels active initially |
| Fast finishing Newton 80 | 1.96 s | many pixels exit immediately |
| Total staged Newton | **5.79 s** | 99.7467% at strict tolerance |
| Archived FIRE 4000 | **7.36 s** | 99.89% at strict tolerance, median 593 iterations |

The archived FIRE run has more than the few outliers seen in the user's newest visual review; no newer FIRE scalar summary was saved, so the table deliberately uses the reproducible archived numbers. The qualitative conclusion is unchanged: FIRE and damped Newton now give similar smooth fields, with only sparse convergence outliers.

## 8. Newton versus FIRE

### Per-iteration cost

FIRE needs one force/torque evaluation per step. Centered-difference Newton needs 12 perturbed evaluations for its 6×6 Hessian plus one or more trial evaluations. A Newton iteration is therefore roughly an order of magnitude more expensive in force-field evaluations.

The dense 6×6 solve itself is cheap, about `O(6^3)`, and is not the bottleneck. The expensive part is repeatedly evaluating all atom–basis interactions. This distinction matters when reasoning about optimization: parallelizing the 6×6 solve would not recover the Hessian-evaluation cost.

### FIRE strengths

- Hessian-free and naturally follows a dissipative local path.
- One force evaluation per iteration.
- Usually robust when Hessian curvature is indefinite or noisy.
- Extends more naturally to many flexible degrees of freedom, where a dense Hessian becomes prohibitive.

### FIRE costs and caveats

- Often needs hundreds or thousands of steps.
- Depends on effective mass, moment of inertia, time step, damping, and persistent velocity/adaptive state.
- Translational and rotational dynamics must use consistent frames: angular velocity and inertia are body-frame quantities, while total torque is accumulated in world space and transformed by `R^T`.
- Changing effective translational mass without scaling `I` and `I^-1` consistently changes rotational versus translational time scales.
- A few stuck pixels can usually be given another persistent batch. If only approximately two pixels remain, special compaction logic is probably not worth the complexity; first inspect their residual and geometry.

### Damped-Newton strengths

- No physical/artificial momentum, mass, or time step.
- Energy-monotone accepted trajectory.
- Very fast finishing near a minimum.
- Attractive for this six-DOF rigid problem and massively parallel independent pixels.
- Explicit damping provides a tunable bridge from gradient-like basin following to Newton finishing.

### Damped-Newton costs and caveats

- Central Hessian evaluation is expensive.
- The unscaled combination of translation and rotation makes `lambda` and trust system-dependent.
- Too little persistent damping returns to basin-jumping Newton; too much damping becomes slow gradient descent.
- Large finite-difference steps stabilize float32 but average the curvature.
- A formally converged Newton point can be a different local minimum from its neighbor; convergence masks alone cannot diagnose image continuity.

### Practical choice

For the current six-DOF PTCDA imaging problem, staged damped Newton is competitive with FIRE in wall time because it uses far fewer iterations and the GPU exposes many replicas. FIRE remains the simpler and more scalable choice when the number of relaxed degrees of freedom grows, when a reliable Hessian step is unavailable, or when solver-development effort matters more than the last factor in wall time.

For production-quality images, running both on a reduced grid is valuable. Agreement of E, COM, tilt, and constraint force is a stronger diagnostic than either solver's residual alone.

## 9. Convergence policy

The strict `1e-4` force and torque thresholds are useful for solver debugging but may be unnecessarily expensive for imaging. A `1e-3` residual may be adequate if changing the threshold does not visibly or numerically change E and constraint-force maps. This must be validated as an image-error criterion, not accepted only because the scalar residual is small.

Recommended staged execution:

1. run 80 iterations;
2. download only the small body force/torque arrays and compute the convergence mask on CPU;
3. if needed, run another 10–20 or 80 persistent iterations;
4. stop when the map difference or unconverged fraction is acceptable;
5. save the mask rather than silently treating unfinished pixels as valid.

Possible later optimization: build a compact list of unfinished replica indices and relaunch only those. This is intentionally not implemented yet.

## 10. Validation performed

### Focused one-pixel regression

```bash
python -m pytest tests/test_folded_relax.py::test_ptcda_pinned_newton_single_replica --develop -s
```

RTX 3090 result after the final float32 defaults:

```text
E = -0.21744700 eV
|F| = 1.054e-5
|tau| = 7.078e-5
iterations = 21
PASS
```

The test also verifies that 80 one-iteration launches reproduce one continuous 80-iteration launch when persistent Newton state is used. It similarly checks persistent FIRE state.

### Rigid-body physics audit

The separate audit reported force finite-difference error `1.27e-7`, torque-frame error `1.68e-5`, and free-body angular-momentum/rotational-energy drift around `1.8e-4`. This supports the quaternion, torque-frame, and inertia transforms. It does not validate attraction-basin choice.

## 11. Reproduction

```bash
# Full visual driver; requires NVIDIA OpenCL visibility
python tests/testplot_ptcda_nacl_replicas.py

# Focused L0/L1/L2 Newton persistence regression
python -m pytest tests/test_folded_relax.py::test_ptcda_pinned_newton_single_replica --develop -s

# Independent mechanics audit
python debug/rigid_body_physics_audit.py
```

The staged calls are equivalent to:

```python
out = rbd.run_folded_newton_replicas(niter=80, eps_t=0.1, eps_r=0.1, trust0=0.1, lambda0=1.0, f_tol=1e-4, t_tol=1e-4)
# CPU checks |F| and |tau| here.
out = rbd.run_folded_newton_replicas(niter=80, eps_t=0.1, eps_r=0.1, trust0=0.1, lambda0=0.01, f_tol=1e-4, t_tol=1e-4)
```

## 12. Open issues and cautions

1. **Not USER-confirmed fixed.** The new maps are much smoother, but remaining branch lines need physical review.
2. **Newton replica versus single-body implementation.** Centered differences are currently specific to the replica imaging kernel; the single-body workgroup kernel uses forward differences.
3. **Coordinate scaling.** Translation and body rotation should eventually be nondimensionalized before applying one LM damping and one trust norm.
4. **Tolerance semantics.** Force and torque have different units; identical numerical thresholds are a practical convention, not a dimensionless error norm.
5. **Physical model.** Newton–FIRE agreement establishes solver consistency, not accuracy of the fitted FAF potential, spring model, rigid-molecule approximation, or chosen pin height.
6. **Initial-condition dependence.** Correct pin alignment is necessary. A deliberately tilted start selects a biased basin and should be treated as a physical protocol choice.
7. **Constraint force sign.** Report whether a plotted force is spring-on-molecule or molecule-on-tip; they differ by a minus sign.
8. **Do not hide failures.** Keep convergence and iteration maps next to scientific fields. Do not inpaint until the failure mechanism and desired presentation policy are agreed.

## 13. Student checklist

Before trusting a rigid-relaxation image:

- Verify the pinned atom starts on its anchor for the actual initial quaternion.
- Confirm quaternion normalization and body/world torque conventions with a one-body test.
- Use a finite-difference step compatible with the arithmetic precision.
- Preserve optimizer state across host batches.
- Inspect energy monotonicity for representative difficult pixels.
- Plot E, COM height, orientation, convergence, and constraint reaction together.
- Quantify neighboring-pixel jumps; convergence fraction alone is insufficient.
- Compare at least one reduced map with a qualitatively different optimizer.
- Treat smooth branch lines as potentially physical and scattered area-filling texture as a numerical warning.
- Record exact solver parameters, hardware, tolerances, wall time, and artifact paths.

## Related documentation

- Earlier evolving task report: [`doc/Tasks/Report_PTCDA_NaCl_FAF_RigidImaging.md`](../Tasks/Report_PTCDA_NaCl_FAF_RigidImaging.md)
- Folded rigid-body design: [`doc/Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md`](../Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md)
- Test design: [`doc/Tasks/TestDesign_FoldedBasisRigidBody.md`](../Tasks/TestDesign_FoldedBasisRigidBody.md)
- Repository test policy: [`doc/TEST_DESIGN.md`](../TEST_DESIGN.md)

