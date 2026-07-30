# Task: PairFF integration into SPAMMM GUI

**Status:** implemented (core) — `RigidAssemblyExtension` provides Drag + MC/GA + PME in one panel; USER L2 review pending  
**Priority:** P1 after multi-body kernel demo (`PairFF_MultiBody_Kernel.md`)  
**Depends on:** `demos/demo_pairff.py`, `spammm/GUI/RigidBodyVispy.py`, `RigidBodyPairFF` (unified kernel); **shared rigid pose SSOT** ([`RigidMoleculePose_SSOT.md`](RigidMoleculePose_SSOT.md), inventory [`TopicalAudit/RigidBody.md`](../TopicalAudit/RigidBody.md)) — do **not** invent a parallel pose store when wiring PairFF into the main GUI  
**Related:** `FoldedRigidExtension.py`, `ChargeRingsExtension.py`, `FFExtension.py`, `doc/GUI_FF_Relaxation.md`, `doc/Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md`

---

## Objective

Wire **unified PairFF** (`rigid_body_pairff_unified_kernel`) into the main **SPAMMM_GUI** editor workflow so users can relax / manipulate molecules with nonbonded pairwise interactions (H-bond epairs, sigma holes, compact-exp) inside the same window as structure editing — not only via the standalone `RigidBodyVispy` demo window.

---

## Inventory: what already exists

### PairFF stack (demo-ready, not GUI-registered)

| Component | Path | Role |
|-----------|------|------|
| Physics | `spammm/forcefields/RigidBodyDynamics.py` → `RigidBodyPairFF` | 6-DOF rigid body + unified/legacy pair kernels |
| GPU | `kernels/rigid.cl` kernels 7–8, `kernels/Forces.cl` `compact_exp_pair_EF` | Forces + FIRE integration |
| Standalone viewer | `spammm/GUI/RigidBodyVispy.py` | PyQt5 panel + VisPy canvas (uracil+HCOOH demo) |
| CLI demo | `demos/demo_pairff.py` | Headless FIRE + optional viewer launch |
| Reference map | `examples/density_comparison/HBondFF/fit_radial.py` | CPU potential map for GUI overlay |

### GUI extensions touching “relaxation on surface”

| Extension | Registry key | Physics model | Rigid body? | Substrate / surface |
|-----------|--------------|---------------|-------------|---------------------|
| **FoldedRigidExtension** | `folded_rigid` | Folded-basis Morse/Coulomb on **periodic NaCl FAF** | **Yes** — 1 adsorbate, 6-DOF | Fit folded coefficients; `run_folded` / Newton |
| **FFExtension** | `ff` | **SPFF / UFF** flexible MD on `AtomicSystem` | No — all atoms move | Optional FAF via fused relax path in `SPFF_cl` / `UFF_cl` (atomistic, not rigid) |
| **AFMExtension** | `afm` | FDBM/STM pipeline | No (PP relax is separate) | Sample geometry only |
| **DFTBExtension** | `dftb` | DFTB+ opt | No | Gas-phase / implicit |

**Answer to “do we already have rigid-body FF on FAF?”**  
**Yes — `FoldedRigidExtension`** (`spammm/GUI/FoldedRigidExtension.py` + `spammm/surfaces/FoldedRigid.py`). It uses `RigidBodyDynamics.run_folded()` against a **fitted folded basis** on a lattice substrate, not PairFF.

**Answer to “do we have UFF/SPFF on FAF surface?”**  
**Yes — `FFExtension`** via `FFController` → GPU `SPFF_cl` / `UFF_cl` with optional FAF substrate coupling (flexible intramolecular + nonbonded to surface). This is **atomistic MD**, not rigid-body 6-DOF.

PairFF is a **third** surface/intermolecular model: **rigid molecules + pairwise compact-exp** (good for H-bond directionality, fast assembly scoring). It is **not** a drop-in replacement for folded basis or SPFF/UFF.

---

## Integration options (preference: existing extension)

### Option A — **Extend FoldedRigidExtension** (recommended template, dual backend)

**Idea:** Keep one “molecule on / near surface” panel; add **Interaction mode** combo:

| Mode | Backend | Use case |
|------|---------|----------|
| `Folded basis` (current) | `FoldedRigid.setup_rigid_folded` + `run_folded` | Adsorbate on NaCl(100) FAF, imaging scans |
| `PairFF unified` (new) | `RigidBodyPairFF` + multi-body env (`PairFF_MultiBody_Kernel.md`) | Adsorbate–adsorbate, soft assembly, H-bond docking on flat / pre-relaxed geometry |

**Shared GUI assets (reuse, do not duplicate):**

- Edit modes: `FR Manip` (anchor spring drag), pin/COM placement patterns from `FoldedRigidExtension`
- Timer / Step / Relax buttons pattern
- `_sync_sys()` / `refresh_view()` geometry write-back
- Substrate overlay markers (static molecules = “environment”)

**New PairFF-specific UI** (can live in same panel or collapsible section):

- Kernel mode: unified (default) / legacy
- He, Hs, w, beta, k_z, epair_dist, sigma_dist
- **Active molecule** selector (which body integrates)
- Environment list: which editor fragments / loaded XYZs are rigid obstacles

**Why not only FoldedRigid?** Physics differ; folded basis is substrate-periodic; PairFF is pairwise between explicit rigid bodies. Combining under one extension avoids a third panel while keeping backends separate.

### Option B — New `PairFFExtension` (cleaner SoC, more registry noise)

Thin wrapper: register `pairff` in `ExtensionManager`, embed or dock `RigidBodyVispy` `ControlPanel` into main window, build `RigidBodyPairFF` from `AtomicGraph` selections.

**Pros:** No risk of breaking FoldedRigid users.  
**Cons:** Duplicates manip/timer/sync patterns; user preference was existing extension.

### Option C — Extend **FFExtension** (not recommended)

`FFController` assumes one flexible `AtomicSystem`. PairFF needs per-molecule rigid frames, epair/sigma dummy atoms, and a different OpenCL kernel family. Would require a parallel code path inside FF panel — high confusion (“Relax” would mean different physics per combo).

---

## Recommended path

1. **Short term (demo):** ~~Extend `demos/demo_pairff.py` for multi-body + pick-one-active~~ — **done** (`PairFF_MultiBody_Kernel.md`): allmol shared buffers, `--bodies` / `--mols`, LMB selects active (index only). ~~FAF substrate + map compose~~ — **done** (`PairFF_FAF_Substrate.md`): `--faf`.
2. **GUI phase:** **Option A (variant: new `RigidAssemblyExtension`)** — implemented 2026-07-30. New extension `spammm/GUI/RigidAssemblyExtension.py` (registry key `rigid_assembly`, title "Rigid Assembly") provides one panel with three modes (Drag / MC-GA / PME), all sharing a single `RigidEnsemble` pose SSOT + single `RigidBodyPairFF` GPU backend. Reuses `greedy_energy_step`, `update_anchors`, `pauli_scan.scan_xy` — no new physics. L0 tests: `tests/GUI/test_rigid_assembly_extension.py` (5 pass). USER L2 review pending.
3. **Do not** merge PairFF into `FFExtension` without explicit USER decision.

---

## GUI wiring checklist (when implementing)

| Step | File | Action |
|------|------|--------|
| 1 | `ExtensionManager.py` | If new key: `'pairff'` + `DEFAULT_CONFIG`; if Option A: no new key |
| 2 | `SPAMMM_GUI.py` | Extension title; optional edit-mode registration |
| 3 | `FoldedRigidExtension.py` (or new file) | `build_ui()`: mode combo, PairFF spins, active-body combo |
| 4 | `spammm/forcefields/RigidBodyDynamics.py` | Multi-body `RigidBodyPairFF` API (see other task) |
| 5 | `AtomScene` / `VispyUtils` | Reuse picking; avoid second VisPy window — draw in main `AtomScene` |
| 6 | `gui_scripts/` | `pairff_setup.py` headless bootstrap (mirror `folded_rigid_setup.py`) |

**Entry points to study:**

- `FoldedRigidExtension.build_ui()` — panel layout SSOT
- `FoldedRigidExtension.prepare_folded_rigid()` — scripted GUI prep
- `FoldedRigidExtension._on_relax()` — relax → sync geometry
- `FRManipMode` — anchor picking (same pattern as `RigidBodyVispy._set_anchor`)
- `doc/GUI_FF_Relaxation.md` — `sig_geometry_changed`, pinning TODOs

---

## Data flow (target)

```
RigidEnsemble (pose SSOT: pos, qrot)  ◄── write authority for rigid sessions
        │
        ├─► AtomicGraph atoms_world (display; derived) ──► AtomScene
        ├─► RigidBodyPairFF GPU poss/qrots (mirror)
        └─► (optional) ChargeRings PME spos + R(q)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     active body: integrate            fixed bodies: env only
```

Until [`RigidMoleculePose_SSOT.md`](RigidMoleculePose_SSOT.md) exists in code, do **not** treat GPU/`_mb_*` as a second GUI authority — stage poses through one host array path shared with FoldedRigid / ChargeRings.

**Open design questions for USER:**

1. Sync policy: pose-primary vs graph-primary ([`RigidMoleculePose_SSOT.md`](RigidMoleculePose_SSOT.md) decision gate)?
2. After relax, update **only active molecule** in `AtomicGraph`, or all bodies?
3. Substrate: keep folded NaCl overlay + PairFF adsorbates, or PairFF-only (no lattice fit)?
4. Embed PairFF map in main scene vs keep side potential map from `RigidBodyVispy`?

---

## Acceptance (do not mark Done without USER confirmation)

- [x] USER picks Option A vs B → **Option A variant: new `RigidAssemblyExtension`** (USER-approved 2026-07-30)
- [x] One molecule pickable as "active"; others fixed; visible in main GUI — `RigidBodyPairFF.set_active_body` + main `AtomScene` (no second VisPy window)
- [x] Unified kernel default; legacy available for A/B — `RigidBodyPairFF.from_molecules` uses unified kernel
- [x] Geometry sync documented (which atoms update in graph) — one-way `ensemble → AtomicGraph` via `_update_graph` after each accepted MC step / drag release
- [x] `gui_scripts/pairff_setup.py` or extended `folded_rigid_setup.py` reproduces demo in GUI — `prepare_rigid_assembly(window, mol, nmol, ...)` GUI-script entry point implemented
- [ ] **USER L2 review** of the extension in a live GUI session (build, MC run, drag, PME scan)

## Out of scope (this task)

- Replacing folded-basis substrate fitting
- SPFF/UFF intramolecular relax inside PairFF (rigid bodies stay rigid)
- AFM/STM pipeline coupling (separate from PairFF assembly)
