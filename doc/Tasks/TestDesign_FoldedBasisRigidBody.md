# Test Design: Folded Basis Surface Interaction → Rigid Body Relaxation → AFM Scan

## Overview

Three-level test series for molecule-substrate interaction using the folded basis (`cos(k·u)·cos(k·v)·exp(-α·z)`) surface potential, integrated into the pytest system in `tests/`.

Each level builds on the previous:
1. **Level 1**: Single-atom ↔ folded basis substrate (fitting + force validation)
2. **Level 2**: Rigid molecule relaxation on folded basis forcefield
3. **Level 3**: Lateral scan / AFM imaging / manipulation trajectory with visualization

---

## Level 1: Single-Atom Folded Basis Fit & Force Validation

### Review of existing test: `test_folded_surface_scan.py`

**Status**: Standalone script (not pytest). Already working, human-reviewed. Results documented in `TEST_RESULTS.md` (lines 364–397).

**What it does**:
- Fits folded basis (4×4 lateral cosine modes × 5 z-exponentials = 80 basis functions) to NaCl(100) surface potential
- Morse (Pauli+London): GPU brute-force reference with 4×4 PBC, energy-masked to exclude repulsive wall
- Coulomb: Ewald2D periodic reference (replaces finite-cluster GPU brute-force which had lateral-average artifact)
- Both exp and poly z-basis tested
- Produces z-scan plots at Na and Cl sites

**Results** (fit-region RMSE, eV):
| Component | Site | exp RMSE | poly RMSE |
|-----------|------|----------|-----------|
| Morse     | Na   | 2.2e-5   | 9.6e-5    |
| Morse     | Cl   | 4.0e-6   | 6.9e-5    |
| Coulomb   | Na   | 8.2e-4   | 3.4e-4    |
| Coulomb   | Cl   | 8.2e-4   | 3.4e-4    |

**Key findings**:
- Exp basis fits better than poly for Morse; poly slightly better for Coulomb
- Na≡Cl Coulomb RMSE identical (physical correctness check)
- Finite-cluster Coulomb artifact fixed by Ewald2D
- Poly kernel power sequence issue identified (sequential vs doubling powers) — not blocking for exp basis

### Proposed pytest integration

**File**: `tests/test_folded_atom_surface.py`

**Test 1a**: `test_folded_fit_morse` — `@pytest.mark.gpu`
- Fit folded basis to Morse (Pauli+London) reference on NaCl(100)
- Use `MolecularDynamics.fit_folded_surface_basis()` with `components=('pauli','london')`
- Evaluate fit at z-scan points above Na and Cl sites
- Assert: RMSE < 0.01 eV in fit region (z ∈ [1.5, 8.0] Å)
- Assert: Na and Cl site fits have comparable RMSE (Morse is site-dependent but charge-independent)
- Save plots to `debug/<date>_folded_atom/`

**Test 1b**: `test_folded_fit_coulomb_ewald` — `@pytest.mark.gpu`
- Fit folded basis to Coulomb reference using `coulomb_solver='ewald2d'`
- Use `components=('coulomb',)` with `coulomb_solver='ewald2d'`
- Evaluate at z-scan points above Na and Cl sites, for probe charge Q=±0.5
- Assert: RMSE < 0.01 eV in fit region
- Assert: Na(+Q) ≡ Cl(-Q) and Na(-Q) ≡ Cl(+Q) (symmetry: |RMSE_Na_Q+ - RMSE_Cl_Q-| < 1e-4)
- Save plots

**Test 1c**: `test_folded_force_vs_finite_diff` — `@pytest.mark.gpu`
- Evaluate folded basis forces (analytic gradient) at a grid of (x,y,z) points
- Compare to central finite-difference of folded basis energy (eps=1e-4 Å)
- Assert: max|F_analytic - F_numeric| < 1e-3 eV/Å (float32 precision)
- This validates the `folded_eval_grad_rigid` function in `rigid.cl`

**Test 1d** (visual): `test_visual_folded_surface_map` — `@pytest.mark.visual @pytest.mark.gpu`
- Plot 2D lateral energy maps E(x,y) at several z heights
- Plot z-scan curves E(z) at Na, Cl, and bridge sites
- Show reference (brute-force/Ewald) vs folded fit side by side
- User checks: contrast pattern matches NaCl checkerboard, decay with z, symmetry

---

## Level 2: Rigid Molecule Relaxation on Folded Basis Forcefield

### Infrastructure available

- `RigidBodyDynamics.init_folded(coeffs, kxyz, atom_type, lvec2d, meta)` — uploads folded basis to GPU
- `RigidBodyDynamics.run_folded(num_steps, dt, lin_damp, ang_damp)` — runs rigid body MD with folded basis forces
- `RigidBodyDynamics.download_outputs()` — gets positions, quaternions, forces, torques
- `MolecularDynamics.fit_folded_surface_basis()` — fits coefficients, can extract them for RigidBodyDynamics
- `rigid_body_folded_kernel` in `rigid.cl` — already implemented, evaluates folded basis per atom

### Setup procedure (shared helper)

```python
def setup_rigid_folded(mol_file, substrate_file, z_init=3.0, ...):
    # 1. Load molecule XYZ, get apos, enames, REQs
    # 2. Compute COM, relative positions, mass properties
    # 3. Fit folded basis using MolecularDynamics.fit_folded_surface_basis()
    #    with coulomb_solver='ewald2d', components=('pauli','london','coulomb')
    # 4. Extract folded_coeffs, folded_kxyz, folded_atom_type, folded_lvec2d
    # 5. Create RigidBodyDynamics, realloc, upload_state, init_folded
    # 6. Return rbd object ready for run_folded()
```

### Test 2a: `test_relax_h2o_nacl` — `@pytest.mark.gpu @pytest.mark.slow`

**Physical expectation**: H2O on NaCl(100) — O atom orients toward Na+ ion (weak dative coordination via free electron pairs). H atoms point away from surface or toward Cl⁻.

- Place H2O at ~3 Å above NaCl surface, centered near Na site
- Initial orientation: O down, H atoms up (roughly)
- Run `run_folded(num_steps=2000, dt=0.01, lin_damp=0.95, ang_damp=0.90)`
- Assert: `|body_force| < 0.5 eV/Å` (converged)
- Assert: `|body_torque| < 0.5 eV` (converged)
- Assert: final z (COM height) in range [2.0, 4.0] Å (physical adsorption height)
- Assert: O atom is closer to nearest Na than to nearest Cl (dative coordination)
  - Compute: for each O atom in molecule, find nearest substrate Na and Cl from substrate XYZ
  - Check: `d(O, Na_nearest) < d(O, Cl_nearest)`
- Save: XYZ trajectory (initial + final), energy vs step plot, geometry plot

### Test 2b: `test_relax_pyridine_nacl` — `@pytest.mark.gpu @pytest.mark.slow`

**Physical expectation**: Pyridine (C5H5N) on NaCl — N lone pair orients toward Na+ (similar dative coordination as H2O's O). Molecule lies roughly flat (π-system parallel to surface) or slightly tilted.

- Place pyridine at ~3.5 Å, flat orientation (ring plane parallel to surface)
- Center near Na site
- Run relaxation
- Assert: forces converged
- Assert: N atom closer to Na than to Cl
- Assert: molecule height (ring plane z) in [2.5, 4.5] Å
- Save: trajectory, energy plot, geometry

### Test 2c: `test_relax_ptcda_nacl` — `@pytest.mark.gpu @pytest.mark.slow`

**Physical expectation**: PTCDA (C24H8O6) on NaCl(100) — large planar molecule, lies flat. O atoms (anhydride) may align with Na rows. Known adsorption: flat-lying, commensurate with substrate lattice.

- Place PTCDA at ~3.5 Å, flat orientation
- Run relaxation (more steps: 5000, larger molecule = slower convergence)
- Assert: forces converged (`|body_force| < 1.0 eV/Å`)
- Assert: molecule remains roughly flat (tilt angle < 15° from surface plane)
- Assert: height in [2.5, 4.5] Å
- Assert: O atoms preferentially near Na sites (check at least 2 of 6 O atoms)
- Save: trajectory, energy plot, geometry (top view showing O-Na alignment)

### Test 2d (visual): `test_visual_relax_trajectory` — `@pytest.mark.visual @pytest.mark.gpu @pytest.mark.slow`

- For each molecule (H2O, pyridine, PTCDA):
  - Save XYZ trajectory every N steps (e.g. every 100 steps)
  - Plot energy + force vs step
  - Plot final geometry overlaid with substrate atoms (top view)
  - Highlight O-Na and N-Na distances
  - User checks: molecule relaxes to expected orientation, O/N toward Na

---

## Level 3: Lateral Scan / AFM Imaging / Manipulation Trajectory

### Test 3a: `test_lateral_scan_energy_landscape` — `@pytest.mark.gpu @pytest.mark.slow`

**Goal**: Map the adsorption energy landscape by scanning molecule laterally across the substrate.

- For a grid of (x, y) positions (e.g. 8×8 or 16×16 across one unit cell):
  - Place molecule at each (x, y) at fixed height z_ads (from Level 2 relaxation)
  - Run short relaxation (100-500 steps) to let molecule orient locally
  - Record: final energy, final orientation, O-Na distance (for H2O)
- Plot: 2D energy map E(x,y), 2D O-Na distance map, 2D tilt angle map
- Assert: energy map has periodicity matching NaCl lattice (4 Å)
- Assert: energy minima at expected adsorption sites (Na site for H2O)
- Save: 2D maps to `debug/<date>_lateral_scan/`

### Test 3b (visual): `test_visual_afm_scan` — `@pytest.mark.visual @pytest.mark.gpu @pytest.mark.slow`

**Goal**: AFM-like force measurement — hold molecule at fixed height, scan laterally, record vertical force Fz.

- For a grid of (x, y) positions at several z heights:
  - Place molecule, run short relaxation with z-constraint (anchor spring on COM z)
  - Record Fz (vertical force on molecule)
- Plot: Fz(x,y) maps at different z heights (AFM contrast images)
- Plot: Fz(z) curves at characteristic sites (Na, Cl, bridge)
- User checks: AFM contrast shows NaCl checkerboard pattern, contrast fades with height

### Test 3c (visual): `test_visual_manipulation_trajectory` — `@pytest.mark.visual @pytest.mark.gpu @pytest.mark.slow`

**Goal**: Show how molecule moves when dragged across substrate (manipulation).

- Define a path across substrate (e.g. straight line Na→Cl→Na)
- At each path point:
  - Move molecule COM to path position (with anchor spring)
  - Run relaxation (let molecule orient + settle)
  - Record: position, orientation, energy, O-Na distance
- Visualization:
  - **Multi-snapshot overlay**: substrate atoms (Na blue, Cl green) as fixed background, molecule snapshots at several path positions overlaid with varying transparency
  - **Energy profile**: E vs path position
  - **O-Na distance**: d(O, Na_nearest) vs path position
  - **XYZ trajectory**: all snapshots saved as multi-frame XYZ for external visualization (VMD, OVITO)
- User checks:
  - Molecule follows expected path (no sudden jumps)
  - O atom tracks Na sites (stays close to nearest Na)
  - Energy barriers at transition points (Cl bridge) are visible
  - No unphysical behavior (molecule flipping, escaping surface)

### Visualization implementation notes

For the multi-snapshot overlay plot:
```python
# Substrate atoms as scatter (Na=blue, Cl=green, size ~ ionic radius)
# For each snapshot (every Nth path point):
#   - Plot molecule atoms as scatter with alpha=0.3 + thin lines for bonds
#   - Color by element (O=red, N=blue, C=black, H=gray)
#   - Annotate O-Na distance for key snapshots
# Use proj='xy' (top view) since substrate is in xy-plane
```

For XYZ trajectory:
- Use `save_xyz_frames()` from `tests/helpers/geometry.py`
- Include substrate atoms in each frame (for context in VMD/OVITO)
- Comment line: `step=N x=... y=... E=... d_O_Na=...`

---

## Implementation Notes

### Shared helper: `tests/helpers/folded_rigid.py`

Extract common setup code:
- `fit_folded_for_molecule(mol_file, substrate_file, ...)` → returns fitted params
- `setup_rigid_body_folded(mol_file, substrate_file, z_init, ...)` → returns RigidBodyDynamics object
- `download_trajectory(rbd, n_steps, step_interval)` → returns trajectory arrays
- `compute_nearest_substrate_distances(atom_positions, substrate_positions, element_filter)` → O-Na, O-Cl distances
- `plot_molecule_on_substrate(ax, mol_positions, mol_enames, substrate_positions, substrate_enames, ...)` → top-view overlay

### Test file structure

```
tests/
  test_folded_atom_surface.py    # Level 1 (integrate test_folded_surface_scan.py into pytest)
  test_folded_relax.py           # Level 2 (molecule relaxation)
  test_folded_scan.py            # Level 3 (lateral scan, AFM, manipulation)
  helpers/
    folded_rigid.py              # shared helpers
```

### Marks

- Level 1: `@pytest.mark.gpu` (fast, ~10s)
- Level 2: `@pytest.mark.gpu @pytest.mark.slow` (moderate, ~30-60s per molecule)
- Level 3: `@pytest.mark.gpu @pytest.mark.slow` (slow, ~2-5 min per scan)
- Visual tests: add `@pytest.mark.visual`

### Key physical checks summary

| Molecule | Key check | Metric |
|----------|-----------|--------|
| H2O      | O→Na dative coordination | d(O,Na) < d(O,Cl), d(O,Na) < 3.0 Å |
| Pyridine | N→Na dative coordination | d(N,Na) < d(N,Cl), d(N,Na) < 3.0 Å |
| PTCDA    | Flat adsorption, O near Na | tilt < 15°, ≥2/6 O atoms near Na |
| All      | Converged forces | \|F\| < 1.0 eV/Å, \|τ\| < 1.0 eV |
| All      | Physical height | z_COM ∈ [2.0, 4.5] Å |

### Dependencies

- `MolecularDynamics.fit_folded_surface_basis()` with `coulomb_solver='ewald2d'` — already implemented
- `RigidBodyDynamics.init_folded()` / `run_folded()` — already implemented
- `rigid_body_folded_kernel` in `rigid.cl` — already implemented
- Substrate data: `data/substrates/NaCl_1x1_L3.xyz` (3-layer, has lattice vectors)
- Molecule data: `data/xyz/H2O.xyz`, `data/xyz/pyridine.xyz`, `data/xyz/PTCDA.xyz`

### Potential issues to watch

1. **Folded basis coefficient extraction**: `fit_folded_surface_basis()` stores results in `MolecularDynamics.folded_params`. Need to extract `coeffs`, `basis_params` (→ kxyz), `atom_type_ids`, `basis_lvec2d` and pass to `RigidBodyDynamics.init_folded()`. The z0 in basis_params is relative to surface top — need to ensure coordinate convention matches between MolecularDynamics fitting and RigidBodyDynamics evaluation.

2. **Atom type mapping**: `fit_folded_surface_basis()` fits per unique REQ type. The `folded_atom_type` array maps each atom to its type index. Need to ensure this mapping is consistent when transferring from MolecularDynamics to RigidBodyDynamics.

3. **Coordinate origin**: Substrate XYZ has atoms at specific z positions. The folded basis z0 is set to `z_range[0]` relative to surface top. RigidBodyDynamics uses absolute coordinates. Need to verify that the z-coordinate convention is consistent (surface top = max z of substrate atoms).

4. **PTCDA size**: 38 atoms. RigidBodyDynamics supports up to 128 atoms/body. Should work, but convergence may be slow. Consider larger `nsteps` and stronger damping.

5. **Lateral scan performance**: 8×8 grid × 100 relax steps = 64 relaxations. Can batch multiple positions as separate rigid bodies (n_bodies > 1) in a single `run_folded()` call for GPU parallelism.
