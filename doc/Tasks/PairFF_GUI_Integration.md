# Task: PairFF integration into SPAMMM GUI

**Status:** investigating — design notes only (no implementation yet)  
**Priority:** P1 after multi-body kernel demo (`PairFF_MultiBody_Kernel.md`)  
**Depends on:** `demos/demo_pairff.py`, `spammm/GUI/RigidBodyVispy.py`, `RigidBodyPairFF` (unified kernel)  
**Related:** `FoldedRigidExtension.py`, `FFExtension.py`, `doc/GUI_FF_Relaxation.md`, `doc/Tasks/RigidBodyDynamicsWithFoldedBasisSubstrate.md`

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

1. **Short term (demo):** Extend `demos/demo_pairff.py` for multi-body + pick-one-active (`PairFF_MultiBody_Kernel.md`) using standalone `RigidBodyVispy`.
2. **GUI phase:** **Option A** — add PairFF section to `FoldedRigidExtension` *or* rename panel to `SurfaceRigidExtension` if scope grows; keep `folded_rigid` registry key for backward compatibility.
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
AtomicGraph (SSOT) ──► per-molecule rigid bodies (poses, quats, REQ, epairs)
                              │
                              ▼
                    RigidBodyPairFF (multi-body)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     active body: integrate            fixed bodies: env only
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    AtomScene markers (world apos)
                              │
                              ▼
              backend._sync_sys() on relax stop (active body only?)
```

**Open design questions for USER:**

1. After relax, update **only active molecule** in `AtomicGraph`, or all bodies?
2. Substrate: keep folded NaCl overlay + PairFF adsorbates, or PairFF-only (no lattice fit)?
3. Embed PairFF map in main scene vs keep side potential map from `RigidBodyVispy`?

---

## Acceptance (do not mark Done without USER confirmation)

- [ ] USER picks Option A vs B
- [ ] One molecule pickable as “active”; others fixed; visible in main GUI
- [ ] Unified kernel default; legacy available for A/B
- [ ] Geometry sync documented (which atoms update in graph)
- [ ] `gui_scripts/pairff_setup.py` or extended `folded_rigid_setup.py` reproduces demo in GUI

## Out of scope (this task)

- Replacing folded-basis substrate fitting
- SPFF/UFF intramolecular relax inside PairFF (rigid bodies stay rigid)
- AFM/STM pipeline coupling (separate from PairFF assembly)
