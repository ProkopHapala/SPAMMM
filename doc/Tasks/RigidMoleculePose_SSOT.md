---
type: Task
title: Rigid-molecule pose SSOT (pos + qrot)
tags: [rigid-body, pose, SSOT, PairFF, Assembly, FoldedRigid, PME, ChargeRings]
timestamp: 2026-07-30
status: design locked — ready to implement (USER 2026-07-30)
priority: P1 — before PairFF GUI wiring
---

# Task: Shared rigid-molecule pose representation (`pos` + `qrot`)

**Status:** design locked by USER 2026-07-30 — ready to implement  
**Priority:** P1 — **before** PairFF main-GUI wiring (so GUI does not invent a 4th pose store)  
**Inventory SSOT (read first):** [`doc/TopicalAudit/RigidBody.md`](../TopicalAudit/RigidBody.md)  
**Related:** skill `molecular-structure-sync`; [`PairFF_GUI_Integration.md`](PairFF_GUI_Integration.md); [`PairFF_RigidBody.md`](../TopicalAudit/PairFF_RigidBody.md); [`ChargeRings_PME.md`](../TopicalAudit/ChargeRings_PME.md); [`Import_ChargeRings_PME.md`](Import_ChargeRings_PME.md); [`PairFF_RigidEnergy_MC_GA.md`](PairFF_RigidEnergy_MC_GA.md)

---

## Objective

Introduce a **shared Python numpy representation** of per-molecule rigid poses (`pos` + `qrot`) that all rigid-body-related modules import and sync **to** — so they stop forking geometry among themselves.

**This is NOT a global SSOT.** `AtomicGraph` and `SPAMM_GUI` remain independent and must work fine when no rigid-body module is loaded. `RigidEnsemble` is **optional**, scoped to rigid-body modules only. The dependency direction is **modules → ensemble, never the reverse**: the ensemble exposes fast flat numpy arrays; each module reads them and does its own conversion to its own GPU buffers / internal format using its own template data.

| Consumer (depends on ensemble) | Reads from ensemble, converts in-module to |
|----------|-----------------|
| PairFF / FoldedRigid / RigidBodyAFM | GPU `poss` / `qrots` upload (GPU buffers stay as-is, per-algorithm optimized) |
| Assembly / SPFF batch | `[R\|t]` or 4×4 transforms (module's own matrix math) |
| ChargeRings PME | `spos` CoM + `rots = R(q)` for multipole **gating** (module owns multipole frame) |
| VisPy / AtomicGraph display | `apos_world = pos + R(q) @ apos_body` (module owns `atoms_body`) — **one-way ensemble→graph, on demand** |
| MC/GA | separate module's business; reads ensemble poses, not the ensemble's concern |

**USER verdict (2026-07-30):** ChargeRings **sites = rigid molecules** (COM + molecular frame). The shared rep stops rigid modules from forking geometry among themselves. `AtomicGraph`/GUI are **not** made to depend on it.

---

## Conventions (locked from RigidBody audit — do not reinvent)

| Item | SSOT value | Where |
|------|------------|-------|
| Quaternion layout | **`qrot = (qx, qy, qz, qw)`** xyzw; identity `[0,0,0,1]` | `kernels/rigid.cl` L17; `_quat_to_matrix_np` |
| World atoms | `apos_world = pos + R(q) @ apos_body` | kernel `quat_to_a/b/c`; host `_body_sites_world` uses `pos + rel @ R.T` (same SE(3)) |
| COM | **rigid-body center**; mass-weighted for dynamics, but **settable by other means** when needed | `compute_mass_properties` / `_prepare_molecule_pack`; epair/σ-hole dummies share origin, **no mass** |
| PME frame | full **3×3** `rots[i*9]` (already); builders today only fill φ-z | `PME.cl` `compute_tip_interaction` |

Reuse existing converters — do **not** write a second quat math stack:
- `_quat_to_matrix_np`, `_body_sites_world`, `_set_pose` / `download_outputs` / `sync_active_pose_from_gpu`
- Assembly scipy xyzw → mat (already compatible)

---

## Motivation — why a shared rigid rep is needed

`AtomicGraph` is SSOT for **atomic** topology/`Atom.pos` and stays so. Rigid modules need a **molecule-level** 6-DOF state that they currently fork among themselves (detail in [`RigidBody.md`](../TopicalAudit/RigidBody.md) §Authority / §Conflicting sources):

```
AtomicGraph (atoms)  ←── one-way ──  FoldedRigid GPU
PairFF _mb_pos/quat  ←── sync ──►  GPU poss/qrots  ←── FoldedRigid / AFM
ChargeRings spos/rots (φ-z)     ✕ no link
Assembly 4×4 / SPFF [R|t]       ✕ no link
```

**Target (rigid-optional, modules depend on ensemble — not vice versa):**

```
RigidEnsemble  (shared Python numpy: pos (N,3), qrot (N,4) — fast flat arrays)
        ▲
        │ modules import ensemble, read get_poses(), do their own conversion
        │
   ┌────┴────┬──────────────┬───────────────┬──────────────────┐
   │         │              │               │                  │
 PairFF    FoldedRigid    Assembly       ChargeRings        (one-way, on demand)
 _mb_* ←──  fr_rbd ←──    [R|t] ←──      spos+R(q) ←──      → AtomicGraph atoms_world
 upload    upload          (own math)     (own multipole)    (display; derived)
 (GPU buffers stay as-is, per-algorithm optimized — NOT touched/converted into SSOT)
```

**GPU buffers are NOT made into mirrors of the ensemble.** They stay exactly as they are — per-algorithm, optimized. The new shared thing is the **Python numpy representation**. Modules read it and upload to their own buffers.

`AtomicGraph`/`SPAMM_GUI` do **not** depend on the ensemble. The ensemble → `AtomicGraph` write is **one-way, on demand** (manual), only when a rigid module wants to update display. The reverse (atoms → poses) is **deferred** — harder, out of current scope.

Documented dual language in `topical_audit.md` §3b (“AtomicGraph for atoms; RBD for physics”) must be **rewritten**: the shared numpy rep is the rigid-modules truth; RBD GPU buffers remain per-algorithm working storage.

---

## Analogy to AtomicGraph (scoped)

| Layer | Atomic (exists) | Rigid molecules (this task) |
|-------|-----------------|----------------------------|
| **SSOT** | `AtomicGraph` (global) | **`RigidEnsemble`** (rigid-modules-only, optional) |
| Bridge | `MoleculeEditorBackend._sync_sys()` | modules call `ensemble.get_poses()`; one-way `ensemble → AtomicGraph` on demand |
| Consumers | VisPy, FF, export, Kekule | PairFF, Assembly, FoldedRigid, ChargeRings (each does own conversion) |
| Templates | ElementTypes / atom data | **stay in existing modules** — ensemble holds poses only |

**Rule:** rigid modules import the ensemble and read `get_poses()`; they never treat their own GPU buffers, `_mb_*`, demo locals, or `pauli_scan` circle geometry as the shared truth *among rigid modules*. The ensemble does **not** reach into modules — dependency is one-directional (modules → ensemble).

---

## Proposed shape (sketch) — poses only

```
RigidEnsemble                       # spammm/forcefields/RigidEnsemble.py
  bodies[i]:
    id                     # stable id (like Atom._id)
    tid                    # species id (reference; template data stays in modules)
    pos                    # (3,) rigid-body center — mass-weighted for dynamics, settable otherwise
    qrot                   # (4,) xyzw — identity [0,0,0,1]
    # optional: pin, active, in_pme_subset
```

**No templates in the ensemble.** `atoms_body`, `pairff_sites`, `multipole_cs`, `default_Esite` stay in their existing modules. The ensemble only holds per-body poses + stable ids + tid.

**Fast flat arrays (the point of this module):**
- `get_poses()` → `(pos (N,3) float32/64, qrot (N,4) float32/64)` contiguous numpy arrays
- `set_poses(pos, qrot)` / `set_pose(i, pos, q)` / `normalize_quats()`
- Preallocate; no per-call allocation in hot paths

**Derived by consuming modules (not by the ensemble):**
- `R = quat_to_mat(qrot)` → PME `rots`, Assembly `[R|t]` (module's own call to existing `_quat_to_matrix_np`)
- `apos_world` → VisPy / `AtomicGraph.update_positions_from_array` (module owns `atoms_body`)
- PairFF: `upload_state` / `from_molecules` from `get_poses()` (module's own upload)

### Sync policy (locked USER 2026-07-30)

**Rigid-optional shared store — NOT a global SSOT.**

| Direction | Behavior |
|-----------|----------|
| modules → ensemble | rigid modules import ensemble, read `get_poses()`, do own conversion. Ensemble never reaches into modules. |
| ensemble → AtomicGraph | **one-way, on demand** (manual write when a rigid module wants to update display) |
| AtomicGraph → ensemble | **deferred** (harder; out of current scope) |
| GUI / AtomicGraph independence | `SPAMM_GUI` and `AtomicGraph` work fine with no rigid module loaded; ensemble is optional |

`AtomicGraph` remains the global atomic SSOT. The ensemble is the shared truth **among rigid-body modules only**. FoldedRigid single-adsorbate demos adopt the ensemble (uniform path; parity tests guard regressions).

---

## How PME plugs in (ChargeRings module's responsibility)

```
ensemble.get_poses()  →  pos, qrot
        ↓  ChargeRings module does its own:
           R = _quat_to_matrix_np(qrot)   (reuse existing; no new math)
           spos = pos                     (CoM = rigid-body center)
           rots = R                       (full 3×3, not φ-only)
PME compute_tip_interaction:
  d rotated into molecular frame (PME.cl already)
  multipole → Δε_i (gating); T_i(r) tunneling
        ↓
PME solve → P({n}), I, dI/dV
```

- The ensemble only provides `pos, qrot`. The ChargeRings module owns the multipole frame and does the conversion.
- After FIRE / drag / Assembly accept: **invalidate** ChargeRings caches; next scan reads fresh `get_poses()`.
- `W_ij` from CoM distances (existing) or later oriented multipoles — still from shared poses.
- Symmetric trimer demo → 3× NTCDA body poses (not abstract `makeCircle`).
- **PME `n_sites ≤ 4`:** ensemble may be large; scan uses **active subset** (+ spectator pad) — UI required.

---

## Conversion responsibility (modules own it, not the ensemble)

The ensemble exposes only `get_poses()` → `(pos, qrot)` numpy arrays. Each consuming module does its own conversion using its own template data and existing math helpers. **No converters live in `RigidEnsemble`.**

| Module | Reads from ensemble | Converts in-module to | Reuses |
|--------|---------------------|----------------------|--------|
| PairFF / RBD | `get_poses()` | GPU `poss`/`qrots` upload | `upload_state` / `from_molecules` layout |
| FoldedRigid | `get_poses()` | `fr_rbd` upload | existing upload path |
| Assembly / SPFF | `get_poses()` | `[R\|t]` / 4×4 | scipy xyzw→mat (already compatible) |
| ChargeRings PME | `get_poses()` + subset | `spos`, `rots = R(q)` | `_quat_to_matrix_np` |
| VisPy / AtomicGraph display | `get_poses()` + own `atoms_body` | `apos_world` | `_body_sites_world` |

One-shot **importers** (migration helpers, called once when wiring a module to the ensemble) live **in the consuming module**, not in the ensemble:
- `ensemble_from_pairff` — read `_mb_*` / `download_outputs`, write into ensemble via `set_poses`
- `ensemble_from_assembly` — read winner transforms, write into ensemble via `set_poses`

---

## Implementation checklist (design locked — ready to implement)

### A. Core module first
- [ ] `RigidEnsemble` in `spammm/forcefields/RigidEnsemble.py` — **poses only, no OpenCL, no template data**
- [ ] Stable body ids; `tid` reference; `get_poses()` → fast contiguous `(pos (N,3), qrot (N,4))` numpy; `set_poses()` / `set_pose(i, pos, q)` / `normalize_quats()`; preallocate, no per-call alloc in hot paths
- [ ] L0: round-trip `set_poses` → `get_poses` unchanged; quat normalization preserves identity; `R = _quat_to_matrix_np(qrot)` matches kernel `quat_to_a/b/c`
- [ ] Rewrite `topical_audit.md` §3b + FoldedRigid wording: shared numpy rep is rigid-modules truth; GPU buffers stay per-algorithm working storage

### B. Consumers import ensemble and do own conversion (one at a time, parity test after each)
- [ ] ChargeRings / `pauli_scan`: read `get_poses()` + subset, build `spos`+`R(q)` in-module (keep circle/Ruslan as fallback fixtures)
- [ ] `RigidBodyPairFF`: read `get_poses()`, upload to GPU; `_mb_*` populated from ensemble (GPU buffers untouched)
- [ ] FoldedRigid: read/write active body via ensemble (demos adopt — run each demo first, capture plots for USER review before rerouting)
- [ ] Assembly: write accepted packing into ensemble via `set_poses` (one-shot importer in Assembly module)
- [ ] MC/GA (`PairFF_RigidEnergy_MC_GA.md`): **separate module's business** — reads ensemble poses, kernel 14 stays read-only GPU; ensemble is unaware

### C. GUI
- [ ] Shared “Rigid molecules” pose list used by ChargeRings + PairFF section + FoldedRigid (**before** inventing another bridge in [`PairFF_GUI_Integration.md`](PairFF_GUI_Integration.md))
- [ ] ChargeRings: “sites = rigid bodies”; cut in CoM xy; after FIRE → Rescan
- [ ] Do not duplicate pose editors per extension
- [ ] GUI/AtomicGraph stay independent when no rigid module loaded (ensemble optional)

### D. Data / demos / tests
- [ ] Templates: NTCDA/PTCDA/NTCDI (`data/mol/*.mol2`) → `atoms_body` + default multipole frame **stay in existing modules**
- [ ] L0: move/rotate one body → PME Esite at tip-above-CoM changes with `R(q)`
- [ ] L2: three NTCDA poses → charging rings (replace pure circle for “molecule” story)

### E. Out of scope (Later)
- Flexible atoms as PME sites
- Full multipole–multipole \(W_{ij}\) from density
- **Reverse sync AtomicGraph → ensemble** (atoms → poses; harder, deferred)
- GPU buffers as SSOT (they stay per-algorithm working storage)
- Cosserat / Frenkel — when built, **consume** this ensemble ([`Import_CosseratRods_PTCDA.md`](Import_CosseratRods_PTCDA.md), Frenkel ideas)

---

## Risks (from RigidBody conflicting sources)

1. **PairFF `_mb_*` population** — must be populated from ensemble reads; GPU buffers stay as-is. Risk: a code path writes `_mb_*` directly, diverging from ensemble. Guard with parity test.
2. **FoldedRigid demos** — adopting the ensemble touches working demos. **Run each demo, capture plots for USER review before rerouting** (refactoring discipline). Spinboxes become reads of ensemble, not independent writers.
3. **ChargeRings φ-z → full SO(3)** — `PME.cl` already accepts full `rots`; site builders drop tilt. Uploading full `R(q)` is correct, but downstream `W_ij`/symmetry may have assumed φ-only. L0 move/rotate test is the gate.
4. **Assembly / SPFF matrix language** — each module does its own `get_poses()` → `[R|t]`; no shared converter in ensemble. Accepted packings must write back via `set_poses` (easy to forget).
5. **MC/GA** — separate module; reads ensemble, kernel 14 stays read-only GPU. Do not promote scan-grid trial poses to ensemble authority.
6. **CoM definition** — rigid-body center; mass-weighted for dynamics, settable otherwise. Document per use; do not silently mix mass-CoM and charge-centroid.
7. **Main GUI PairFF** — must depend on this shared rep; do not invent a fourth pose store in Option A/B wiring. But GUI/AtomicGraph stay independent when no rigid module loaded.
8. **Dependency direction** — modules → ensemble, never reverse. If a converter creeps into `RigidEnsemble`, coupling inverts. Keep ensemble dependency-free (only numpy).

---

## Decision gate (USER) — RESOLVED 2026-07-30

1. ~~Sync policy~~ → **rigid-optional shared store** (not global SSOT); one-way ensemble→AtomicGraph on demand; reverse deferred.
2. ~~Module name~~ → **`RigidEnsemble`**.
3. ~~Package~~ → **`spammm/forcefields/`**.
4. ~~Priority~~ → **before** PairFF main-GUI wiring.
5. ~~Scope~~ → **poses only** (templates stay in modules).
6. ~~FoldedRigid demos~~ → **adopt ensemble** (parity tests guard regressions).

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

**A shared Python numpy `(pos, qrot)` per molecule is the correct rigid-modules representation** — optional, scoped to rigid-body modules, with one-way `ensemble → AtomicGraph` on demand. `AtomicGraph`/`SPAMM_GUI` stay independent. GPU buffers stay per-algorithm and untouched. Conventions and math already exist in RBD — the work is **one thin dependency-free numpy module + per-module read/convert wiring**, not new quaternion physics. Modules depend on the ensemble; the ensemble never reaches back. ChargeRings stops inventing geometry; PairFF/Assembly/FoldedRigid read the shared numpy rep instead of forking among themselves.
