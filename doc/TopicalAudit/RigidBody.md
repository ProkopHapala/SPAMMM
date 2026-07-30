---
type: TopicalAudit
title: Rigid-body molecule pose (pos + qrot)
tags: [rigid-body, pose, SSOT, PairFF, Assembly, FoldedRigid, PME, OpenCL, quaternion]
timestamp: 2026-07-30
---

# Rigid-body molecule pose

## Summary

SPAMMM treats adsorbed molecules as **6-DOF rigid bodies** (CoM translation + quaternion orientation) for PairFF, FoldedRigid surface dynamics, Assembly packing, RigidBodyAFM, and (intended) ChargeRings PME gating. Physics and kernels are mature; what is **missing** is a shared authoritative host store of per-molecule `(pos, qrot)` analogous to `AtomicGraph` for atoms. Today poses live in GPU buffers (`poss`/`qrots`), PairFF host mirrors (`_mb_pos`/`_mb_quat`), Assembly/SPFF transform matrices, and ChargeRings ad-hoc `spos`+φ-only `rots` — none of which sync to each other. Connecting PME tip–sample multipoles to the same molecules that PairFF/Assembly/FoldedRigid move **requires** a pose SSOT (design: [`Tasks/RigidMoleculePose_SSOT.md`](../Tasks/RigidMoleculePose_SSOT.md)).

**Quaternion convention (SSOT where used):** `qrot = (qx, qy, qz, qw)` — xyzw, identity `[0,0,0,1]` (`kernels/rigid.cl` L17; `_quat_to_matrix_np`).

**World atoms:** `apos_world = pos + R(q) @ apos_body` (kernel `quat_to_a/b/c`; host `_body_sites_world`: `pos + rel @ R.T`).

**COM:** mass-weighted over real atoms (`compute_mass_properties` / `_prepare_molecule_pack`); epair/σ-hole dummies share the body origin but do not contribute mass.

## Design verdict (USER 2026-07-30)

**Yes** — ChargeRings sites should be the same physical objects as rigid molecules:

| Quantity | Meaning |
|----------|---------|
| Site position | molecule **COM/COG** |
| Site rotation frame | molecule **quaternion → R(3×3)** orienting molecular multipoles for tip–sample electrostatic **gating** (`PME.cl` already rotates tip vector into `rots[i*9]`) |

`AtomicGraph` remains SSOT for atomic topology/positions. Rigid sessions need a dual SSOT for **molecule pose**. Without it, GUI “connect PME ↔ PairFF ↔ Assembly” will keep forking geometry.

Proposed name (TBD): `RigidEnsemble` / `MoleculePoseGraph` — see SSOT task. **Design locked USER 2026-07-30:** `RigidEnsemble` in `spammm/forcefields/`, poses-only (templates stay in modules), rigid-optional shared numpy rep (NOT global SSOT — AtomicGraph/GUI independent), one-way ensemble→AtomicGraph on demand, reverse deferred. Modules depend on ensemble; ensemble never reaches back. GPU buffers stay per-algorithm, untouched.

## Implementations

| Language | Location | Status | Pose storage | Orientation | Notes |
|----------|----------|--------|--------------|-------------|-------|
| OpenCL | `kernels/rigid.cl` | **active** | `poss`, `qrots`, `vposs`, `vrots` float4 | quat xyzw | Dynamics + PairFF + folded + Newton + replica energy |
| OpenCL | `kernels/PME.cl` | **active** | `p_sites.xyz` + `rots[i*9]` | **3×3 only** | Consumer of frames; no quat |
| OpenCL | `kernels/assembly.cl` | **active** | 4×4 transforms | 3×3 + T | Packing clash; no persistent pose store |
| Python | `spammm/forcefields/RigidBodyDynamics.py` | **active** | GPU buffers; `upload_state` / `_set_pose` / `download_outputs` | quat xyzw | Physics mirror; `_quat_to_matrix_np`, `_body_sites_world` |
| Python | same → `RigidBodyPairFF` | **active** | GPU **+** host `_mb_pos (N,3)`, `_mb_quat (N,4)` | quat xyzw | `from_molecules`, `set_active_body`, `sync_active_pose_from_gpu`, replica energy |
| Python | `spammm/surfaces/FoldedRigid.py` | **active** | via RBD upload | quat / identity / random | Folded substrate workflow |
| Python | `spammm/forcefields/RigidBodyAFM.py` | **active** | RBD `poss`/`qrots` | identity default | GridFF AFM path |
| Python | `spammm/forcefields/Assembly.py` | **active** | `(n_conf,nmol,4,4)` / `R_conf`+`T_conf` | **3×3 mats** (scipy xyzw→mat) | Discrete packing search only |
| Python | `spammm/forcefields/SPFF_cl.py` | **active** | `(N,3,4) [R\|t]` | no quat | Batch rigid surface eval; parallel pose language |
| Python | `spammm/quantum/pauli_scan.py` | **active** | local `spos (N,4)`, `rots (N,3,3)` | **φ about z** (`makeRotMats`) | Rebuilt per scan; abstract QD sites |
| Python | `spammm/quantum/PauliSolverCL.py` | **active** | uploads `pSites` + flat `rots` | 3×3 | PME consumer |
| GUI | `spammm/GUI/FoldedRigidExtension.py` | **active** | `window.fr_rbd` + COM spins | identity at setup | GPU→`AtomicGraph.update_positions_from_array` one-way |
| GUI | `spammm/GUI/RigidBodyVispy.py` | **active** | reads RBD/PairFF + `_mb_*` | from GPU/host | PairFF demo Vispy (not main GUI) |
| GUI | `spammm/GUI/ChargeRingsExtension.py` | **active** | rebuilt each scan | φ only | No molecule registry |
| Topology | `spammm/topology/AtomicGraph.py` | **active** | per-atom `Atom.pos` | none | Atomic SSOT only |
| Toolbox | `spammm/AtomicSystem.py` | **active** | flat `apos` | 3×3 rot mats | `orient*`, `addSystems`; not RBD store |
| Dynamics | `spammm/dynamics/Vibrations.py` | **active** | N/A | removes 6 rigid DOF | Orthogonal (mode projection) |
| Tip QM | `spammm/quantum/DFTB/Grid_dftb.py` | **active** | tip scan `tip_quat` | default `[0,0,0,1]` | Tip frame only |
| Cosserat / Frenkel | docs only | unfinished | — | would need poses | `Tasks/Import_CosseratRods_PTCDA.md`, `Ideas/FrenkelRigidFF.chat.md` |
| **Pose SSOT** | `spammm/forcefields/RigidEnsemble.py` | **implemented (core)** | `pos`+`qrot` per body, stable ids | quat xyzw | [`RigidMoleculePose_SSOT.md`](../Tasks/RigidMoleculePose_SSOT.md); L0 tests `tests/forcefields/test_rigid_ensemble.py` |

### Related topical audits (do not merge)

| Audit | Scope |
|-------|-------|
| [`PairFF_RigidBody.md`](PairFF_RigidBody.md) | PairFF force model, allmol kernels, FAF, tip-pull |
| [`ChargeRings_PME.md`](ChargeRings_PME.md) | PME solver / scans; open issue → pose SSOT |
| This file | **Cross-module pose representation** and consolidation plan |

## Kernel map (`kernels/rigid.cl`) — pose role

| Kernel family | Pose role |
|---------------|-----------|
| `rigid_body_dynamics_kernel` | Integrate `poss`/`qrots` (pairwise + anchors) |
| `rigid_body_*_gridff_*` | GridFF forces → same integration |
| `rigid_body_folded_*` / Newton / replicas | Folded basis; pose MD / optimizer |
| PairFF unified allmol (± FAF) | Multi-mol layout; `active_mol` index; all poses stay on GPU |
| `rigid_body_pairff_energy_replica_kernel` | **Read-only** replica `poss`/`qrots` → energies (MC/GA consumer pattern) |

Helpers: `quat_mult`, `make_qrot` / `make_qrot_taylor`, `qrot_omega*`, `quat_to_a/b/c`.

## Authority map (who owns truth today)

```
                    ┌─────────────────────┐
                    │   AtomicGraph       │  atoms only
                    │   Atom.pos          │
                    └──────────▲──────────┘
                               │ one-way (FoldedRigid _update_graph)
┌──────────────┐    ┌──────────┴──────────┐    ┌─────────────────┐
│ PairFF host  │◄──►│ GPU poss / qrots    │◄───│ FoldedRigid /   │
│ _mb_pos/quat │sync│ (RBD / PairFF)      │    │ RigidBodyAFM    │
└──────────────┘    └─────────────────────┘    └─────────────────┘
                               ✕ no link
                    ┌─────────────────────┐    ┌─────────────────┐
                    │ ChargeRings spos/   │    │ Assembly 4×4 /  │
                    │ rots (φ-z only)     │    │ SPFF [R\|t]     │
                    └─────────────────────┘    └─────────────────┘
```

**Target (when implemented):**

```
RigidEnsemble.bodies[i].pos, qrot   ← sole write authority
        │
        ├─► GPU poss/qrots          (RBD / PairFF mirror)
        ├─► PME spos + R(q)         (ChargeRings)
        ├─► Assembly/SPFF [R\|t]    (packing / batch eval)
        └─► AtomicGraph atoms_world (display; derived)
```

## Sync points

| Direction | Where | What |
|-----------|-------|------|
| Host → GPU | `upload_state`, `upload_replicas_state`, `reset_pose`, `_set_pose`, `from_molecules` | poss, qrots, apos_body |
| Host → GPU | `upload_replica_poses` | MC/GA trial poses |
| GPU → Host | `download_outputs` | pos, quats, apos_world, forces |
| GPU → Host PairFF | `sync_active_pose_from_gpu` | `_mb_pos`, `_mb_quat` |
| GPU → AtomicGraph | `FoldedRigidExtension._update_graph` | `update_positions_from_array` then `_sync_sys` |
| PME | `PauliSolverCL.scan_current_tip` | fresh `spos`/`rots` every scan — **no reverse sync** |
| Assembly → RBD | *(none)* | winners stay as transform arrays |

Documented dual language today (`doc/topical_audit.md` §3b): *“AtomicGraph SSOT for atoms; RigidBodyDynamics SSOT for physics.”* Pose SSOT would make RBD a **consumer/mirror**, not a second authority.

## Parity / convention status

| Pair | Status |
|------|--------|
| `_quat_to_matrix_np` ↔ `quat_to_a/b/c` | Matched (xyzw) |
| Host `_body_sites_world` ↔ kernel `apos_world` | Same SE(3) math |
| scipy Assembly quat → mat | xyzw-compatible |
| FoldedRigid graph sync | One-way GPU→graph; graph edits do not push pose back |
| PairFF host↔GPU | Manual sync; Vispy calls it for maps/picking |
| ChargeRings ↔ PairFF/Folded | **None** |
| Assembly winners → PairFF/RBD | **None** |
| SPFF `[R\|t]` ↔ quat | Convertible; no shared API |
| ChargeRings φ-z ↔ full SO(3) | PME.cl accepts full `rots`; site builders drop tilt |

## Conflicting / duplicate pose sources

1. **PairFF dual mirror** — GPU buffers and `_mb_*`; stale if `sync_active_pose_from_gpu` skipped.
2. **FoldedRigid GUI** — `fr_rbd` GPU vs COM spinboxes vs `AtomicGraph` after `_update_graph`.
3. **ChargeRings** — circle / Ruslan / JSON geometry, rebuilt per scan; not molecules.
4. **Assembly / SPFF** — matrix transforms; no shared quat store with RBD.
5. **Demos / MC** — local `pos`/`quat` arrays (`demo_pairff.py`, `testplot_pairff_energy_mc.py`).
6. **Replica imaging** — one pose per scan pixel (`testplot_ptcda_nacl_replicas.py`); scan-grid authority, not ensemble.
7. **Main GUI PairFF** — unfinished ([`PairFF_GUI_Integration.md`](../Tasks/PairFF_GUI_Integration.md)); risk of inventing another bridge.

## Open issues

- [x] **`RigidEnsemble` module implemented** — `spammm/forcefields/RigidEnsemble.py` (poses only, numpy, stable ids); L0 tests pass; `testplot_pairff_energy_mc.py` rerouted with bit-exact parity vs reference. Remaining consumers (ChargeRings, Assembly, FoldedRigid, PairFF `_mb_*`) to be rerouted incrementally.
- [ ] Sync policy undecided: **pose-primary** (recommended for multi-mol / PME / PairFF) vs **graph-primary** (draw→promote)
- [ ] ChargeRings historically φ-only; must bridge via full `R(q)` (PME already supports it)
- [ ] CoM definition: mass CoM (RBD) vs geometric center vs charge centroid — pick per template
- [ ] PME `n_sites ≤ 4` vs large ensembles — need active subset / spectator pad
- [ ] FoldedRigid “dual SSOT” language must be rewritten once pose SSOT exists
- [ ] Main GUI: one shared “Rigid molecules” pose list for ChargeRings + PairFF + FoldedRigid
- [ ] Cosserat / Frenkel deferred — both should consume the same pose SSOT if built
- [ ] Formal L0: move one body → PME Esite / gating at tip-above-CoM changes with `R(q)`
- [ ] Decision gate in task: sync policy, module name, package location — before any PR

## Key file index

**Core:** `spammm/forcefields/RigidBodyDynamics.py`, `kernels/rigid.cl`, `spammm/surfaces/FoldedRigid.py`, `spammm/forcefields/RigidBodyAFM.py`, `spammm/forcefields/Assembly.py`, `spammm/forcefields/SPFF_cl.py`

**GUI:** `spammm/GUI/FoldedRigidExtension.py`, `spammm/GUI/RigidBodyVispy.py`, `spammm/GUI/ChargeRingsExtension.py`, `spammm/GUI/gui_scripts/folded_rigid_setup.py`

**PME:** `spammm/quantum/pauli_scan.py`, `spammm/quantum/PauliSolverCL.py`, `kernels/PME.cl`

**Topology bridge:** `spammm/topology/AtomicGraph.py`, `spammm/topology/MoleculeEditorBackend.py`, `spammm/AtomicSystem.py`

**Tests / demos:** `demos/demo_pairff.py`, `tests/testplot_pairff_energy_mc.py`, `tests/testplot_ptcda_nacl_replicas.py`, `tests/test_folded_relax.py`, `tests/quantum/test_pme_trimer.py`, `debug/rigid_body_physics_audit.py`

## Links

- Design (implement when prioritized): [`Tasks/RigidMoleculePose_SSOT.md`](../Tasks/RigidMoleculePose_SSOT.md)
- PairFF physics audit: [`PairFF_RigidBody.md`](PairFF_RigidBody.md)
- ChargeRings: [`ChargeRings_PME.md`](ChargeRings_PME.md)
- Folded substrate task: [`Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md`](../Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md)
- PairFF GUI wiring: [`Tasks/PairFF_GUI_Integration.md`](../Tasks/PairFF_GUI_Integration.md)
- MC/GA replica poses: [`Tasks/PairFF_RigidEnergy_MC_GA.md`](../Tasks/PairFF_RigidEnergy_MC_GA.md)
- Index: [`../topical_audit.md`](../topical_audit.md) §2b2 / §2d / §3b
