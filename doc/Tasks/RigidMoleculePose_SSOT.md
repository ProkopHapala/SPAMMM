---
type: Task
title: Rigid-molecule pose SSOT (pos + qrot)
tags: [rigid-body, pose, SSOT, PairFF, Assembly, FoldedRigid, PME, ChargeRings]
timestamp: 2026-07-30
status: design — do not implement until USER prioritizes
priority: P1
---

# Task: Rigid-molecule pose SSOT (`pos` + `qrot`)

**Status:** design only — **do not code until USER prioritizes** and picks sync policy  
**Priority:** P1 glue — ChargeRings PME ↔ PairFF / Assembly ↔ FoldedRigid / RBD / MC-GA replicas  
**Inventory SSOT (read first):** [`doc/TopicalAudit/RigidBody.md`](../TopicalAudit/RigidBody.md)  
**Related:** skill `molecular-structure-sync`; [`PairFF_GUI_Integration.md`](PairFF_GUI_Integration.md); [`PairFF_RigidBody.md`](../TopicalAudit/PairFF_RigidBody.md); [`ChargeRings_PME.md`](../TopicalAudit/ChargeRings_PME.md); [`Import_ChargeRings_PME.md`](Import_ChargeRings_PME.md); [`PairFF_RigidEnergy_MC_GA.md`](PairFF_RigidEnergy_MC_GA.md)

---

## Objective

Introduce a **host-side authoritative store** of per-molecule rigid poses — the dual of `AtomicGraph` for atoms — so every rigid consumer reads the same `(pos, qrot)`:

| Consumer | Needs from pose |
|----------|-----------------|
| PairFF / FoldedRigid / RigidBodyAFM | GPU `poss` / `qrots` upload |
| Assembly / SPFF batch | `[R\|t]` or 4×4 transforms |
| ChargeRings PME | `spos` CoM + `rots = R(q)` for multipole **gating** |
| VisPy / AtomicGraph display | `apos_world = pos + R(q) @ apos_body` |
| MC/GA replicas | trial pose arrays (already pattern in kernel 14) |

**USER verdict (2026-07-30):** ChargeRings **sites = rigid molecules** (COM + molecular frame). Without a shared pose SSOT, GUI “connect PME to PairFF” will keep forking geometry.

---

## Conventions (locked from RigidBody audit — do not reinvent)

| Item | SSOT value | Where |
|------|------------|-------|
| Quaternion layout | **`qrot = (qx, qy, qz, qw)`** xyzw; identity `[0,0,0,1]` | `kernels/rigid.cl` L17; `_quat_to_matrix_np` |
| World atoms | `apos_world = pos + R(q) @ apos_body` | kernel `quat_to_a/b/c`; host `_body_sites_world` uses `pos + rel @ R.T` (same SE(3)) |
| COM | **mass-weighted** over real atoms | `compute_mass_properties` / `_prepare_molecule_pack`; epair/σ-hole dummies share origin, **no mass** |
| PME frame | full **3×3** `rots[i*9]` (already); builders today only fill φ-z | `PME.cl` `compute_tip_interaction` |

Reuse existing converters — do **not** write a second quat math stack:
- `_quat_to_matrix_np`, `_body_sites_world`, `_set_pose` / `download_outputs` / `sync_active_pose_from_gpu`
- Assembly scipy xyzw → mat (already compatible)

---

## Motivation — why AtomicGraph is not enough

`AtomicGraph` is SSOT for **atomic** topology/`Atom.pos`. Rigid modules need **molecule-level** 6-DOF state. Today poses are forked (detail in [`RigidBody.md`](../TopicalAudit/RigidBody.md) §Authority / §Conflicting sources):

```
AtomicGraph (atoms)  ←── one-way ──  FoldedRigid GPU
PairFF _mb_pos/quat  ←── sync ──►  GPU poss/qrots  ←── FoldedRigid / AFM
ChargeRings spos/rots (φ-z)     ✕ no link
Assembly 4×4 / SPFF [R|t]       ✕ no link
```

**Target:**

```
RigidEnsemble.bodies[i].pos, qrot   ← sole write authority
        │
        ├─► GPU poss/qrots          (RBD / PairFF mirror)
        ├─► PME spos + R(q)         (ChargeRings)
        ├─► Assembly/SPFF [R|t]     (packing / batch)
        └─► AtomicGraph atoms_world (display; derived)
```

Documented dual language in `topical_audit.md` §3b (“AtomicGraph for atoms; RBD for physics”) must be **rewritten**: RBD becomes a **consumer/mirror**, not a second authority.

---

## Analogy to AtomicGraph

| Layer | Atomic (exists) | Rigid molecules (proposed) |
|-------|-----------------|----------------------------|
| **SSOT** | `AtomicGraph` | **`RigidEnsemble`** (name TBD; aka `MoleculePoseGraph`) |
| Bridge | `MoleculeEditorBackend._sync_sys()` | `sync_poses()` → GPU + PME + optional graph |
| Consumers | VisPy, FF, export, Kekule | PairFF, Assembly, FoldedRigid, ChargeRings, MC replicas |
| Templates | ElementTypes / atom data | per-species body-frame: `atoms_body`, PairFF sites, **multipole frame**, default `Esite` |

**Rule:** write poses only to the ensemble; never treat GPU buffers, `_mb_*`, demo locals, or `pauli_scan` circle geometry as truth.

---

## Proposed SSOT shape (sketch)

```
RigidEnsemble
  templates[tid]:          # immutable per species (NTCDA, PTCDI, …)
    atoms_body             # (na,3) relative to mass CoM
    pairff_sites           # epairs / σ-holes in body frame (optional)
    multipole_cs           # body-frame Q0/Qzz (or axes) for PME
    default_Esite, label
  bodies[i]:
    id                     # stable id (like Atom._id)
    tid
    pos                    # (3,) mass CoM — SSOT
    qrot                   # (4,) xyzw — SSOT
    # optional: pin, active, Esite override, in_pme_subset
```

Derived only:
- `R = quat_to_mat(qrot)` → PME `rots`, Assembly `[R|t]`
- `apos_world` → VisPy / `AtomicGraph.update_positions_from_array`
- PairFF: `upload_state` / `from_molecules` from `get_poses()`

### Sync policy vs AtomicGraph — **USER must pick before coding**

| Policy | When | Behavior |
|--------|------|----------|
| **(1) Pose-primary** (recommended for multi-mol / PME / PairFF / Assembly) | Rigid session | Ensemble owns placement; atoms rebuilt for display; bond edit may be locked or “break rigid” |
| **(2) Graph-primary** | Draw → promote | Edit atoms; “make rigid bodies” fits CoM+qrot; ensemble owns pose until “bake to graph” |

Document which GUI mode uses which. FoldedRigid single-adsorbate demos can stay (2); NTCDA rings + packing use (1).

---

## How PME plugs in

```
ensemble.bodies[i].pos, qrot
        ↓  R = _quat_to_matrix_np(qrot)
PME compute_tip_interaction:
  d rotated into molecular frame (PME.cl already)
  multipole → Δε_i (gating); T_i(r) tunneling
        ↓
PME solve → P({n}), I, dI/dV
```

- After FIRE / drag / Assembly accept: **invalidate** ChargeRings caches; next scan pulls fresh `poses_to_pme_sites`.
- `W_ij` from CoM distances (existing) or later oriented multipoles — still from shared poses.
- Symmetric trimer demo → 3× NTCDA template poses (not abstract `makeCircle`).
- **PME `n_sites ≤ 4`:** ensemble may be large; scan uses **active subset** (+ spectator pad) — UI required.

---

## Converters to implement (thin wrappers on existing math)

| Function | In → Out | Reuse |
|----------|----------|-------|
| `poses_to_pme_sites(ens, subset, Esite=…)` | bodies → `spos (n,4)`, `rots (n,3,3)` | `_quat_to_matrix_np` |
| `poses_to_rbd_arrays(ens)` | → `pos (N,3)`, `quat (N,4)` for upload | same layout as `from_molecules` |
| `poses_to_Rt(ens)` | → `(N,3,4)` or 4×4 | Assembly / SPFF |
| `poses_to_apos_world(ens, templates)` | → flat atoms | `_body_sites_world` |
| `ensemble_from_pairff(rbd)` | GPU/`_mb_*` → ensemble (migration) | `download_outputs` / `sync_active_pose_from_gpu` |
| `ensemble_from_assembly(T)` | winners → bodies | one-shot import |

---

## Implementation checklist (when prioritized)

### A. Core module first
- [ ] `RigidEnsemble` in `spammm/topology/` (preferred, next to `AtomicGraph`) or `spammm/forcefields/` — **poses only, no OpenCL**
- [ ] Stable body ids; template registry; `get_poses()` / `set_pose(i, pos, q)` / `normalize_quats()`
- [ ] Converters table above + L0: quat→R matches `_quat_to_matrix_np`; round-trip pose unchanged
- [ ] Rewrite FoldedRigid / topical_audit wording: RBD = mirror, not SSOT

### B. Consumers read from ensemble
- [ ] `pauli_scan` / ChargeRingsExtension: `from_ensemble=…` path (keep circle/Ruslan as fallback fixtures)
- [ ] `RigidBodyPairFF`: init/refresh from ensemble; stop treating `_mb_*` as authority (mirror only)
- [ ] FoldedRigid: read/write active body via ensemble
- [ ] Assembly: write accepted packing into ensemble
- [ ] MC/GA (`PairFF_RigidEnergy_MC_GA.md`): trial poses from / to ensemble (kernel 14 remains read-only GPU)

### C. GUI
- [ ] Shared “Rigid molecules” pose list used by ChargeRings + PairFF section + FoldedRigid (**before** inventing another bridge in [`PairFF_GUI_Integration.md`](PairFF_GUI_Integration.md))
- [ ] ChargeRings: “sites = rigid bodies”; cut in CoM xy; after FIRE → Rescan
- [ ] Do not duplicate pose editors per extension

### D. Data / demos / tests
- [ ] Templates: NTCDA/PTCDA/NTCDI (`data/mol/*.mol2`) → `atoms_body` + default multipole frame
- [ ] L0: move/rotate one body → PME Esite at tip-above-CoM changes with `R(q)`
- [ ] L2: three NTCDA poses → charging rings (replace pure circle for “molecule” story)

### E. Out of scope (Later)
- Flexible atoms as PME sites
- Full multipole–multipole \(W_{ij}\) from density
- GPU buffers as SSOT
- Cosserat / Frenkel — when built, **consume** this ensemble ([`Import_CosseratRods_PTCDA.md`](Import_CosseratRods_PTCDA.md), Frenkel ideas)

---

## Risks (from RigidBody conflicting sources)

1. **PairFF dual mirror** — GPU vs `_mb_*`; ensemble must own writes; sync becomes ensemble→GPU only.
2. **FoldedRigid triple** — `fr_rbd` / COM spins / graph; spins become views of ensemble.
3. **ChargeRings φ-z** — must upload full `R(q)`; do not silently drop tilt.
4. **Assembly / SPFF matrix language** — convertible; one shared API (`poses_to_Rt`).
5. **Replica imaging / MC** — scan-grid or trial poses are *sessions* over the ensemble, not parallel authorities.
6. **CoM definition** — lock mass CoM (RBD) for templates; document if charge-centroid ever needed for gating.
7. **Main GUI PairFF** — must depend on this SSOT; do not invent a fourth pose store in Option A/B wiring.

---

## Decision gate (USER)

Before any PR:

1. Sync policy: **pose-primary (1)** vs **graph-primary (2)** (or both with documented modes)?
2. Module name: `RigidEnsemble` vs `MoleculePoseGraph`?
3. Package: `spammm/topology/` vs `spammm/forcefields/`?
4. Priority relative to PairFF main-GUI wiring and Hubbard PME import?

---

## Cross-doc updates (this session)

| Doc | Role |
|-----|------|
| [`TopicalAudit/RigidBody.md`](../TopicalAudit/RigidBody.md) | Inventory + authority map (SSOT for “what exists”) |
| This file | What to code + conventions + checklist |
| [`ChargeRings_PME.md`](../TopicalAudit/ChargeRings_PME.md) | Open issue → pose SSOT |
| [`PairFF_RigidBody.md`](../TopicalAudit/PairFF_RigidBody.md) | Open issue → pose SSOT |
| [`PairFF_GUI_Integration.md`](PairFF_GUI_Integration.md) | Depends on pose SSOT before dual-panel wiring |
| [`Import_ChargeRings_PME.md`](Import_ChargeRings_PME.md) | Follow-on: sites from rigid molecules |

---

## Verdict

**Shared `pos`+`qrot` per molecule is the correct SSOT**, dual to `AtomicGraph`. Conventions and converters already exist in RBD — the work is **consolidation + thin adapters**, not new quaternion physics. ChargeRings must stop inventing geometry once the ensemble exists; PairFF/Assembly/FoldedRigid must stop treating GPU/demo arrays as truth.
