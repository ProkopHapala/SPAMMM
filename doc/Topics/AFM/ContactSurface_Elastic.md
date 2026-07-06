# AFM Contact Surface — Elastic (Flexible Sample, Stiffness Enrichment)

**Status:** Design (not implemented)  
**Scope:** Extend the contact-surface representation to capture **sample deformation** under indentation — local compliance/stiffness, not just a rigid height map.  
**Prerequisite:** [ContactSurface_Static.md](ContactSurface_Static.md) (spatial discretization, fitting infrastructure, GPU eval patterns).  
**External brainstorm:** [FireCore IndentationForce2D.chat.md](file:///home/prokophapala/git/FireCore/doc/Topics/AFM/IndentationForce2D.chat.md)

---

## 1. Problem Statement

### Limit of rigid static model

Phase 1 (static) assumes fixed nuclear positions. For flexible molecules (UFF/SPFF spring networks, AFM soft CO tips), the observable AFM contrast depends on **how much the sample deforms** under the tip. Precomputing a single 3D grid of forces on rigid atoms:

- Cannot represent indentation-induced relaxation without recomputing the grid.
- Forces `GridFFRelaxedScan`-style full relaxation at every scan pixel — correct but O(scan × MD) cost.

### What elasticity adds

At each lateral position (x,y), store not only **where** contact occurs but **how stiff** the sample is when pushed:

| Field | Symbol | Meaning |
|-------|--------|---------|
| Contact height | h(x,y) | z where tip first feels repulsion (may include elastic pre-relaxation) |
| Normal stiffness | K_z(x,y) or C_z | dF_z/dz at contact (or compliance C_z = 1/K_z) |
| Surface slope | ∇h | (∂h/∂x, ∂h/∂y) — geometric sliding under vertical load |
| Lateral compliance | C_xx, C_xy, C_yy | Optional 2×2 tensor for true sideways elasticity |

**Runtime (Winkler foundation):**

```
Δz = h(x,y) - z_probe     (indentation depth, > 0 in contact)
F_z  = K_z · Δz
F_xy ≈ -F_z · ∇h  +  C_lat · Δr_xy
```

This reduces per-step cost to **2D texture lookups** vs 3D trilinear interpolation, while capturing the dominant mechanical DoF of flexible organic samples.

---

## 2. Separation from Phase 1

| Aspect | Static (Phase 1) | Elastic (Phase 2) |
|--------|------------------|-------------------|
| Sample | Rigid nuclei | Flexible FF (springs, angles, optional NB) |
| Precompute | V(x,y,z) or atom radial fold | h, K_z, ∇h (+ optional C_lat) |
| Physics | Pairwise Morse/LJ/Coulomb | + linearized Hessian or PBD indentation |
| Cost driver | Grid memory | Indentation solves at sample points |
| PP-AFM use | Direct force from V | Rubber-surface law + CO tip spring |

Phase 2 **reuses** Phase 1 spatial structures:

- Coarse xy bins (1–3 Å) + one hotspot per top atom
- GPU sphere-casting for h(x,y) (Minkowski dilation)
- Dense 0.2 Å rasterization for PP runtime texture
- Option A/B evaluators become **sampling backends** for validation

---

## 3. Data Structure (from second-iteration design)

### Hierarchy

```
Layer 0: Top atoms (contact shell)
Layer 1: Coarse xy bins (~3 Å) — spatial hash, O(1) lookup
Layer 2: Adaptive mesh — coarse vertices + atom hotspots
         snap rules: vertex / edge / face (ε margins)
Layer 3: Dense regular grid (~0.2 Å) — GPU runtime texture
```

**Avoid global Delaunay/Voronoi.** Use **grid-anchored triangulation**: regular quad topology, vertices perturbed to atom (x,y). Barycentric interpolation to dense grid — local, GPU-parallel rasterization.

### Runtime texture payload

**Minimal (MVP):**

```
Texture A (float4):  h, K_z, ∂h/∂x, ∂h/∂y
```

**Extended (if lateral artefacts in PP sliding):**

```
Texture B (float4):  C_xx, C_yy, C_xy, (reserved)
```

Store **compliance** C_z = 1/K_z in fitting/interpolation where soft regions make K_z huge and numerically noisy.

---

## 4. Precomputation Pipeline

### Step 1 — Geometry pass (cheap, GPU)

Per dense pixel (x,y):

```
h(x,y) = max_i [ z_i + sqrt((R_i + R_tip)² - r_xy²) ]
```

- Workgroup tile (16×16) + LDS atom cache (same pattern as proposed in IndentationForce2D).
- Track `atom_id` of argmax — identifies contact patch for Step 2.
- Finite differences on tile → ∇h.

### Step 2 — Stiffness pass (expensive, sparse)

Only at **coarse mesh nodes** (not every dense pixel):

**Option 2a — Analytic Winkler (fast, first validation):**

Per atom i: local spring k_i from Morse curvature at equilibrium. Weight by contact participation:

```
K_z(x,y) ≈ Σ_i k_i · w_i(x,y) · exp(-r_xy²/σ²)
```

**Option 2b — Linearized FF (accurate):**

1. Build molecular Hessian H at equilibrium (UFF/SPFF), with pinning on substrate anchor atoms.
2. Regularize: H' = H + αI (projective dynamics / PBD pinning term).
3. Apply virtual vertical force f at contact atoms under (x,y).
4. Solve H' Δx = f via **Jacobi** (GPU-friendly, matrix-free).
5. Extract K_z = |f| / Δz_probe.

Effective stiffness formula (scalar push):

```
K_eff = (f^T f) / (f^T H'^{-1} f)    [Rayleigh quotient — no full inverse]
```

**Warm-start:** displacement from pixel (i,j) seeds solve at (i+1,j) — often 1–3 Jacobi iterations.

**Option 2c — Nonlinear PBD (fallback):**

Full constraint projection when indentation is large (floppy chains, steric between surface molecules). Use only where linear K_z differs >10% from short PBD test.

### Step 3 — Rasterize to dense grid

Barycentric interpolation of coarse-node (h, K_z, ∇h, C_lat) → 0.2 Å texture. One-time offline cost.

---

## 5. Stiffness Solver Design Notes

### Linear regime (default)

CO tip + organic molecule → small deformations → H' Δx = f is adequate.

| Solver | When | GPU fit |
|--------|------|---------|
| Jacobi / Gauss-Seidel | General, matrix-free | Excellent — local updates |
| Cholesky LDL^T | Small molecule <100 DoF per hotspot | CPU preprocess once |
| Full PBD iterations | Nonlinear, large deformation | Moderate — constraint loops |

Regularization α:

- Physical: M/(dt²) inertia (projective dynamics)
- Practical: substrate pinning stiffness

### Lateral compliance

Only if validation shows geometric ∇h sliding is insufficient:

- Finite-difference probe: displace (x,y) by δ, re-solve → C_lat entries.
- 2–4 extra linear solves per coarse node — still << full 3D grid cost.

### Non-covalent collisions

Surface molecules colliding under load: PBD inequality constraints (unilateral). Scope tied to force-field design elsewhere; document interface only.

---

## 6. Connection to Phase 1 Representations

| Phase 1 option | Elastic extension |
|----------------|-------------------|
| **A: B-spline × exp** | Add K_z as separate separable field fitted on same grid; or derive K_z from second derivative of V_Pauli w.r.t. z at h |
| **B: Radial fold + PIC** | Per-atom k_i in folding; K_z(x,y) = Σ_i K_i · φ_i(|r-r_i|) with same PIC |

Phase 2 does not require choosing A or B exclusively — stiffness can live on the **coarse mesh** regardless of how rigid reference V was represented.

---

## 7. PP-AFM Runtime Integration

Modify probe relaxation (`kernels/AFM.cl`):

```
// Current:
fe = interpFE(pos, ..., imgIn_3D);

// Elastic:
(h, Kz, dhdx, dhdy) = sampleContact2D(pos.x, pos.y, texA);
dz = h - pos.z;
if (dz > 0) {
    Fz = Kz * dz;
    Fx = -Fz * dhdx;   // geometric slide
    Fy = -Fz * dhdy;
}
// + CO tip internal spring (tipForce) — unchanged
// + optional far-field vdW/ES atom sum
```

Far-field attraction (London tail beyond contact) may still use direct atom sum or Phase 1 separable field — orthogonal to rubber-surface repulsion.

---

## 8. Implementation Plan

### Depends on Phase 1

- [ ] Contact surface spatial types (coarse bins, mesh, dense texture)
- [ ] GPU tiled geometry pass (h, ∇h)
- [ ] Reference brute indentation for validation

### Phase 2 specific

- [ ] `IndentationSolver` — Jacobi on GPU for local atom patch
- [ ] Coarse mesh builder with snap rules
- [ ] Rasterizer: coarse → dense texture
- [ ] `relaxStrokesTilted` branch: 2D contact texture mode
- [ ] Staged rollout: analytic Winkler → Jacobi → optional C_lat

---

## 9. Validation Protocol

| Stage | Compare against | Criterion |
|-------|-----------------|-----------|
| h(x,y) only | Rigid dilation + 3D grid iso-surface | <0.05 Å RMS |
| K_z | Full FF relaxation F(z) at 10 (x,y) points | <5% force error |
| AFM image | `GridFFRelaxedScan` or full sample MD | <2% RMS contrast |
| Lateral | PP sliding trajectory | Qualitative L2 review |

Test systems: benzene on surface (rigid ring), H-bonded dimer (directional compliance), floppy alkane (nonlinear fallback).

---

## 10. Staged Complexity (do not over-build)

1. **MVP:** h + K_z only, analytic Winkler, dense 0.2 Å grid.
2. **+∇h** from geometry pass (already cheap).
3. **+Jacobi K_z** at coarse hotspots where |∇K| is large.
4. **+C_lat** only if sliding artefacts appear.
5. **+3×3 compliance kernel patch** (FireCore ChatGPT idea) only if Winkler too stiff on floppy chains.

---

## 11. Open Questions

1. **Who relaxes at scan time?** CO apex only (standard PP-AFM) vs coupled sample+tip MD?
2. **Substrate anchor:** Which atoms pinned in Hessian — affects K_z magnitude?
3. **Consistency:** If sample also relaxed in MD, does offline K_z remain valid?
4. **FDBM path:** Elastic surface orthogonal to density-based Pauli — hybrid possible?
5. **Cache invalidation:** K field depends on anchor geometry; h recomputable from positions alone.

---

## 12. References

| Resource | Path |
|----------|------|
| FireCore brainstorm + meta-analysis | `FireCore/doc/Topics/AFM/IndentationForce2D.chat.md` |
| Static contact surface design | `SPAMMM/doc/Topics/AFM/ContactSurface_Static.md` |
| GridFF relaxed scan (expensive reference) | `spammm/surfaces/GridFFRelaxedScan.py` |
| PBD / projective dynamics | `spammm/forcefields/` (SPFF_cl, MolecularDynamics) |
| Test design | `doc/TEST_DESIGN.md` |
