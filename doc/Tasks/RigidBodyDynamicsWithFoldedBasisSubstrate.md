
# Inventory of Existing Components

## 1. Rigid Body Dynamics ([RigidBodyDynamics.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:0:0-0:0) + [rigid.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/rigid.cl:0:0-0:0))

**What exists:** A fully functional 6-DOF rigid body engine on GPU.

- **[RigidBodyDynamics](cci:2://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:208:0-670:27)** class (`@/home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:209`) — OpenCL wrapper
  - Each rigid body = 1 workgroup of 32 threads
  - State: position (CoM), quaternion, linear/angular momentum
  - Supports up to 128 atoms/body, 4 atoms/thread
  - **Two kernels:**
    - `rigid_body_dynamics_kernel` — pairwise forces (LJ/Morse/Coulomb) + anchor springs + E-field
    - `rigid_body_gridff_kernel` — forces from **precomputed GridFF** (B-spline interpolated 3D grid) + anchor springs
  - Integration: symplectic Euler with Taylor-series quaternion exponential, velocity damping
  - [from_xyz_and_grid()](cci:1://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:556:4-641:18) factory method loads molecule + grid in one call
  - [run_gridff(num_steps, dt, lin_damp, ang_damp)](cci:1://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:447:4-457:27) runs relaxation

- **`RigidBodyAFM`** class (`@/home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyAFM.py:164`) — higher-level wrapper for AFM scanning
  - `relax_to_constraint()` — runs damped MD until force/torque convergence
  - Anchor spring system for tip attachment

**Key limitation:** The `rigid_body_gridff_kernel` currently only reads from a **precomputed B-spline grid** (`BsplinePLQ`). It does NOT support the folded basis (`cos(kx*u)*cos(ky*v)*exp(-az*z)`) directly. The grid must be precomputed on a 3D Cartesian mesh.

## 2. Folded Basis Surface Potential ([MolecularDynamics.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/MolecularDynamics.py:0:0-0:0) + [surface.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/surface.cl:0:0-0:0))

**What exists:** A complete pipeline for fitting and evaluating surface potentials in a Fourier-like basis.

- **Basis functions:** `cos(2π·k_u·u) * cos(2π·k_v·v) * exp(-α·max(0, z-z0))` — see `@/home/prokop/git/SPAMMM/kernels/surface.cl:99-104`
  - `u, v` = fractional coords w.r.t. 2D lattice
  - `z0` = surface top, `α` = decay rate
  - Parameters `(k_u, k_v, α, z0)` stored as `float4` per basis function

- **Fitting pipeline** ([fit_folded_surface_basis()](cci:1://file:///home/prokop/git/SPAMMM/spammm/forcefields/MolecularDynamics.py:1185:4-1320:33) at `@/home/prokop/git/SPAMMM/spammm/forcefields/MolecularDynamics.py:1183`):
  1. Samples brute-force `getSurfMorse` (Pauli + London + Coulomb) on a (u,v,z) grid
  2. For Coulomb: can use `coulomb_solver='ewald2d'` → calls [SurfaceEwaldCL.eval_full()](cci:1://file:///home/prokop/git/SPAMMM/spammm/surfaces/SurfaceEwald.py:297:4-346:39) to get Ewald potential, multiplies by probe charge
  3. Least-squares fit of basis coefficients per atom type
  4. Uploads `folded_coeffs[ntypes, nbasis]`, `folded_kxyz[nbasis]`, `folded_atom_type[natoms]` to GPU

- **GPU evaluation kernels** in [surface.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/surface.cl:0:0-0:0):
  - `getSurfFolded` — per-atom, evaluates `E = Σ c_ib * basis_ib(u,v,z)`, `F = -∇E`
  - `getSurfFolded_workgroup` — optimized version with shared memory
  - Both compute forces directly from the analytic gradient of the folded basis

- **Key buffers:** `folded_coeffs`, `folded_kxyz`, `folded_atom_type`, `folded_lvec2d`, `folded_meta`

## 3. 2D Ewald Electrostatics ([Ewald2D.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/Ewald2D.py:0:0-0:0) + [SurfaceEwald.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/surfaces/SurfaceEwald.py:0:0-0:0))

**What exists:** Both NumPy reference and GPU production implementations.

- **`Ewald2D`** (NumPy, `@/home/prokop/git/SPAMMM/spammm/surfaces/Ewald2D.py:223`) — reference implementation
  - Potential: `φ(ρ,z) = φ₀(z) + Σ_{G≠0} C_G · exp(iG·ρ) · exp(-|G||z-z_i|)`
  - Vacuum form: `φ = Σ_G C_G cos(G·ρ) exp(-|G|z)` — **exactly the folded basis form!**
  - `C_G = (2π/(A|G|)) Σ_i q_i exp(|G|z_i) exp(-iG·ρ_i)`

- **[SurfaceEwaldCL](cci:2://file:///home/prokop/git/SPAMMM/spammm/surfaces/SurfaceEwald.py:43:0-448:39)** (GPU, `@/home/prokop/git/SPAMMM/spammm/surfaces/SurfaceEwald.py:44`) — production
  - [prepare_system()](cci:1://file:///home/prokop/git/SPAMMM/spammm/surfaces/SurfaceEwald.py:164:4-233:62) → computes `C_G` and `w[g,i]` coefficients on GPU
  - [eval_vacuum()](cci:1://file:///home/prokop/git/SPAMMM/spammm/surfaces/SurfaceEwald.py:235:4-295:39) → potential above slab (z > z_max)
  - [eval_full()](cci:1://file:///home/prokop/git/SPAMMM/spammm/surfaces/SurfaceEwald.py:297:4-346:39) → potential at any z

## 4. Morse/Pauli/Dispersion (`Forces.cl` + [surface.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/surface.cl:0:0-0:0))

- **`getMorsePLQH()`** (`@/home/prokop/git/SPAMMM/kernels/Forces.cl:235`) — pairwise Morse with Pauli/London/Coulomb/H-bond decomposition
- **`getSurfMorse`** kernel (`@/home/prokop/git/SPAMMM/kernels/surface.cl:176`) — brute-force molecule-substrate with PBC replicas
- **GridFF** (`nonbonded_grid.cl`) — precomputed B-spline grid with PLQ channels (Pauli, London, Coulomb, H-bond)

---

# Gap Analysis: What's Missing

The key gap is that **the rigid body dynamics engine ([rigid.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/rigid.cl:0:0-0:0)) cannot currently evaluate the folded basis potential**. It only supports:
1. Pairwise atom-atom forces (`rigid_body_dynamics_kernel`)
2. Precomputed B-spline grid interpolation (`rigid_body_gridff_kernel`)

The folded basis evaluation (`getSurfFolded`) exists only in [surface.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/surface.cl:0:0-0:0) and is wired into [MolecularDynamics.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/MolecularDynamics.py:0:0-0:0), NOT into [RigidBodyDynamics.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:0:0-0:0).

---

# Implementation Plan

There are **two approaches** to connect the rigid body dynamics with the folded basis surface potential:

## Option A: New `rigid_body_folded_kernel` (Recommended)

Add a third kernel to [rigid.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/rigid.cl:0:0-0:0) that evaluates the folded basis directly inside the rigid body MD loop.

**Steps:**

1. **Add kernel `rigid_body_folded_kernel`** to [rigid.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/rigid.cl:0:0-0:0):
   - Same structure as `rigid_body_gridff_kernel` (per-body workgroup, quaternion rotation, force/torque reduction)
   - Instead of `fe3d_pbc_comb()` (B-spline grid sampling), call `folded_eval_basis()` + `folded_eval_grad()` from [surface.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/surface.cl:0:0-0:0)
   - Needs additional buffers: `folded_coeffs`, `folded_kxyz`, `folded_atom_type`, `folded_lvec2d`, `folded_meta`
   - Each atom: compute world position → fractional (u,v) → evaluate `Σ c_ib * basis_ib(u,v,z)` → force = `-∇E`

2. **Extend [RigidBodyDynamics.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:0:0-0:0):**
   - Add `init_folded()` method (upload folded coefficients + basis params + lattice)
   - Add `run_folded(num_steps, dt, ...)` method
   - Add kernel header string for `rigid_body_folded_kernel`
   - Load [surface.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/surface.cl:0:0-0:0) alongside [rigid.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/rigid.cl:0:0-0:0) (for `folded_eval_basis`/`folded_eval_grad`)

3. **Prepare the folded coefficients** (reuse existing pipeline):
   - Use [MolecularDynamics.fit_folded_surface_basis()](cci:1://file:///home/prokop/git/SPAMMM/spammm/forcefields/MolecularDynamics.py:1185:4-1320:33) with `coulomb_solver='ewald2d'` to fit Pauli + London + Coulomb(Ewald) onto the folded basis
   - Extract the fitted `folded_coeffs`, `folded_kxyz`, `folded_atom_type`, `folded_lvec2d` from [MolecularDynamics](cci:2://file:///home/prokop/git/SPAMMM/spammm/forcefields/MolecularDynamics.py:78:0-2078:21)
   - Pass them to `RigidBodyDynamics.init_folded()`

4. **Relax PTCDA on NaCl:**
   - Load PTCDA xyz + NaCl substrate xyz
   - Fit folded basis (Pauli+London from `getSurfMorse`, Coulomb from [SurfaceEwaldCL](cci:2://file:///home/prokop/git/SPAMMM/spammm/surfaces/SurfaceEwald.py:43:0-448:39))
   - Initialize rigid body at some height above surface
   - Run `run_folded()` with damping until convergence

**Pros:** Self-contained GPU kernel, no grid precomputation, analytic forces, naturally periodic.
**Cons:** Need to write a new kernel variant.

## Option B: Precompute GridFF from Folded Basis, Then Use Existing `rigid_body_gridff_kernel`

1. Evaluate the folded basis on a 3D Cartesian grid → produce a B-spline PLQ grid
2. Feed that grid to the existing [RigidBodyDynamics.init_gridff()](cci:1://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:409:4-429:74)
3. Run [run_gridff()](cci:1://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:447:4-457:27) as usual

**Pros:** No new kernel needed, reuses existing path completely.
**Cons:** Loses analytic gradients (B-spline interpolation error), requires grid memory, redundant computation (folded basis → grid → B-spline interpolation).

---

# Recommendation

**Option A** is the cleanest and most physically correct. The folded basis `cos(kx·u)·cos(ky·v)·exp(-α·z)` is exactly the natural representation for periodic surface potentials, and the Ewald 2D vacuum form `C_G·cos(G·ρ)·exp(-|G|·z)` maps directly onto it. The existing `folded_eval_basis` and `folded_eval_grad` functions in [surface.cl](cci:7://file:///home/prokop/git/SPAMMM/kernels/surface.cl:0:0-0:0) already provide analytic forces. We just need to wire them into the rigid body kernel.

The implementation is relatively small:
- ~80 lines of new OpenCL kernel code (copy `rigid_body_gridff_kernel`, replace the force evaluation)
- ~50 lines of Python glue in [RigidBodyDynamics.py](cci:7://file:///home/prokop/git/SPAMMM/spammm/forcefields/RigidBodyDynamics.py:0:0-0:0)
- Reuse the existing [fit_folded_surface_basis()](cci:1://file:///home/prokop/git/SPAMMM/spammm/forcefields/MolecularDynamics.py:1185:4-1320:33) pipeline for coefficient preparation

Shall I proceed with Option A?