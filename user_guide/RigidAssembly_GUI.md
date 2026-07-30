# Rigid Assembly GUI — user guide

Interactive rigid-body molecular assembly in the **SPAMMM GUI**: build multi-molecule
systems, optimize their packing with Monte Carlo, drag individual molecules with
anchor springs, and run Pauli Master Equation (PME) charge-ring STM scans — all in
one panel, without leaving the editor.

| Entry | Role |
|-------|------|
| `./run_gui.sh` → "Rigid Assembly" tab | This guide |
| `spammm/GUI/RigidAssemblyExtension.py` | Extension source (glue + Qt panel) |
| `spammm/forcefields/RigidEnsemble.py` | Shared pose store (pos + qrot per molecule) |
| `spammm/forcefields/RigidBodyDynamics.py` → `RigidBodyPairFF` | GPU physics backend |
| `spammm/forcefields/molecule_loaders.py` | Molecule loaders + graph-to-fragments splitter |
| `spammm/quantum/pauli_scan.py` | PME scan engine |
| `tests/GUI/test_rigid_assembly_extension.py` | L0 regression tests |

---

## Quick start

```bash
./run_gui.sh
```

1. Switch to the **"Rigid Assembly"** tab (right panel).
2. **Source: "From file"** → select a molecule (e.g. PTCDA) → set nmol=4 → press **Build**.
3. The editor scene now shows 4 PTCDA molecules on a grid.
4. Press **Step** (MC/GA section) to run one greedy Monte Carlo move, or **Run** for N steps.
5. Watch the status bar: `MC step N: E=X.XXXXX acc=1 batch_min=Y.YYYYY`.

Or, to use molecules you drew yourself:

1. Draw 2+ disconnected molecules in the editor (they must **not** be bonded to each other).
2. **Source: "From editor"** → press **Build**.
3. Each connected component becomes one rigid body at its mass-weighted center of mass.

---

## The three modes

### 1. MC/GA Optimization (greedy Monte Carlo)

Greedy best-of-batch planar moves: for each molecule in round-robin order, generate
`n_trial` random (Δx, Δy, Δφ) perturbations, evaluate the PairFF energy of all trials
in parallel on the GPU, and accept the best one if it lowers the energy.

| Control | Default | Effect |
|---------|---------|--------|
| `n_trial` | 128 | Number of random perturbations per step (more = better sampling, slower) |
| `dxy` | 1.5 Å | Standard deviation of lateral Gaussian perturbation |
| `dphi` | 0.8 rad | Standard deviation of rotational perturbation |
| `k_pack` | 0.03 | Packing well strength (favors compact assemblies) |
| `rmin_atom` | 1.6 Å | Minimum inter-molecular atom distance (clash rejection) |
| `n_steps` | 50 | Steps per **Run** press |
| `seed` | 3 | RNG seed (same seed → same trajectory, for reproducibility) |

**Step** runs one molecule move; **Run** runs `n_steps` moves; **Reset** zeroes the
step counter (does not reset poses).

**Physics:** PairFF compact-exp + optional FAF substrate (NaCl folded basis). Molecules
interact via pairwise compact-exp (H-bond epairs, sigma holes); FAF is molecule↔surface
only. See `doc/TopicalAudit/RigidBody.md`, `demos/PairFF_manual.md`.

### 2. Drag (anchor spring)

Interactive molecule pulling with a harmonic anchor spring — the same pattern as
FoldedRigid's `FRManipMode`, but writing poses back to `RigidEnsemble` on release.

1. Activate the **"RA Drag"** edit mode (toolbar dropdown).
2. **LMB click+drag** on any atom in the scene to pin it and pull the molecule.
3. While dragging, `n_relax` FIRE steps per mouse-move event relax the molecule
   under the spring force.
4. **Release LMB** → the spring clears, and the molecule's pose is written to the ensemble.

| Control | Default | Effect |
|---------|---------|--------|
| `k_spring` | 20.0 | Anchor spring stiffness (eV/Å²) |
| `n_relax` | 20 | FIRE steps per mouse-move (more = smoother drag, slower) |
| `dt` | 0.02 | Dynamics timestep for drag relaxation |

### 3. PME (Pauli Master Equation charge-ring STM)

STM scan over the assembly where each rigid molecule is a charge-ring "site" at its
CoM, oriented by the full rotation matrix R(q) from the ensemble (not just φ-only).

| Control | Default | Effect |
|---------|---------|--------|
| `Esite` | 0.0 eV | On-site energy offset per molecule |
| `W` | 0.05 eV | Inter-site hopping |
| `Q0`, `Qzz` | 0.0, 0.0 | Molecular quadrupole (calibrate per species) |
| `VBias` | 1.0 V | Bias voltage |
| `z_tip` | 5.0 Å | Tip height above molecular plane |
| `Temp` | 1.0 K | Temperature (Fermi broadening) |
| `GammaT` | 0.01 | Tip coupling |
| `decay` | 0.5 | Wavefunction decay length |
| `L` | 20.0 Å | Scan area half-size |
| `npix` | 80 | Scan resolution (npix × npix) |

Press **Scan XY** to run `pauli_scan.scan_xy` over the first min(n_bodies, 4) molecules.
A matplotlib popup shows the STM map with site markers.

**Note:** Q0/Qzz are currently placeholders (0.0). Calibrate per species from QM
multipole calculations before using PME for quantitative work.

---

## Build section

### Source: "From file"

Loads `nmol` copies of a pre-defined molecule from `data/xyz/` or `data/mol/`, places
them on a square grid with `spacing` Å between centers, at height `z_mol` Å.

Available molecules (see `LOADERS` in `molecule_loaders.py`):
PTCDA, NTCDI, formic_acid, terephthalic_acid, TBTAP, azaindol, uracil, adenine.

Each molecule is loaded with QEq charges (unless "no QEq" is checked) and planarized
(z=0 in body frame). Optional FAF substrate (NaCl folded basis) is loaded from
`data/fits/` unless "no FAF" is checked.

### Source: "From editor"

Splits the current `AtomicGraph` (what you drew in the editor) into **connected
components** — groups of atoms linked by bonds. Each disconnected fragment becomes
one rigid body. The body center is the **mass-weighted center of mass** (using atomic
masses from `elements.ELEMENT_DICT`).

- `nmol` and `spacing` are ignored — the number of bodies = number of fragments.
- Body positions = fragment CoMs; z is set to `z_mol`.
- QEq charges are computed on the whole graph, then split per fragment.
- Bonds within each fragment are inferred from geometry (`bonds_from_geom`).

**Caveat:** Linear molecules (e.g. CO₂) have a singular inertia tensor and will fail
at `compute_mass_properties`. Draw non-collinear geometries (bent H₂O, PTCDA, etc.).

---

## Data flow

```
RigidEnsemble (pose SSOT: pos, qrot)  ◄── extension reads/writes poses here
        │
        ├─► RigidBodyPairFF GPU (built once via from_molecules; poses uploaded from ensemble)
        │       ├─ Mode: Drag   → anchor springs; on release, pose → ensemble
        │       ├─ Mode: MC/GA  → greedy_energy_step; accepted poses → ensemble
        │       └─ Mode: PME    → read ensemble subset, build spos+R(q), pauli_scan.scan_xy
        │
        └─► AtomicGraph (display) — one-way ensemble→graph after each accepted step / drag release
```

The extension never creates a second VisPy window — everything renders in the main
`AtomScene`. When the assembly atom count differs from the editor graph (e.g. loading
4×PTCDA=104 atoms into an empty editor), the graph is **rebuilt** from the assembly's
world atoms + element names (same pattern as `FoldedRigidExtension`).

---

## GUI-script entry point

For headless / scripted use (mirrors `FoldedRigidExtension.prepare_folded_rigid`):

```python
from spammm.GUI.RigidAssemblyExtension import prepare_rigid_assembly
prepare_rigid_assembly(window, mol='PTCDA', nmol=4, spacing=16.0, z=3.0,
                        run_mc=True, n_steps=50)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "Build FAILED: Singular inertia tensor" | Molecule is linear (all atoms collinear) | Draw a non-collinear geometry (bent, planar with ≥3 non-collinear atoms) |
| "No backend.graph available" | Editor not initialized | Draw at least one atom first, or use "From file" source |
| "Graph is empty" | No alive atoms in the editor | Draw molecules before using "From editor" |
| Scene doesn't update after Build | Graph rebuild failed silently | Check stderr for traceback; ensure `ra_bonds0` is set |
| PME scan shows flat map | Q0/Qzz = 0 (placeholders) | Set non-zero quadrupole values per species |
| MC accepts nothing | Energy already at minimum, or `dxy`/`dphi` too small | Increase `dxy`/`dphi`, or use simulated annealing (not yet implemented) |

---

## Related documentation

- [PairFF GUI Integration](../doc/Tasks/PairFF_GUI_Integration.md) — design decisions, acceptance checklist
- [Rigid Molecule Pose SSOT](../doc/Tasks/RigidMoleculePose_SSOT.md) — why `RigidEnsemble` exists
- [RigidBody Topical Audit](../doc/TopicalAudit/RigidBody.md) — cross-implementation map
- [PairFF Manual](../demos/PairFF_manual.md) — standalone PairFF demo (headless)
- [Folded Rigid](../doc/TopicalAudit/RigidBody.md) — folded-basis rigid-body manipulation (sibling extension)
- [Charge Rings PME](../doc/TopicalAudit/ChargeRings_PME.md) — PME physics details
