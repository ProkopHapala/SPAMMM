---
type: Strategy
title: SPAMMM Architecture Roadmap — High-Level Decisions & Plans
tags: [architecture, roadmap, gui, gridff, pyscf, presentation, reactive-ff, mol-browser]
---

# SPAMMM Architecture Roadmap

High-level architectural decisions, plans, and notes that must not be forgotten.
This document captures strategic direction — for implementation status see
`FeatureChecklist.md`, for agent task tracking see `doc/ToDo/ToDo.agents.md`,
for comprehensive audit see `doc/ToDo/Features.audit.md`.

---

## 1. Molecular Browser (ACDSee-style)

**Goal:** Fast molecule browser like ACDSee — thumbnail grid, click to view/edit,
compute AFM images, vibrations, etc. on selected molecule.

**Reference implementation (FireCore):**
- `/home/prokop/git/FireCore/pyBall/GUI/VispyMolBrowser.py` (815 lines) — mature PyQt5+VisPy browser
- `/home/prokop/git/FireCore/pyBall/GUI/mol_browser_plugins/` — plugin system:
  - `base.py` — `MolBrowserPlugin` ABC + `MolBrowserContext`
  - `registry.py` — `MolBrowserPluginRegistry` + `MolBrowserPluginHost` (east tab strip)
  - `vibration_spectrum.py` — FTIR histogram, mode pick, 3D arrows/animation
  - `__init__.py` — `default_plugin_registry()` wiring

**Current SPAMMM state:**
- `spammm/GUI/MolecularBrowser.py` — old PyOpenGL version (deprecated, needs removal)
- `spammm/GUI/MolecularBrowserVispy.py` — VisPy Phase 1 (thumbnail grid + 3D view, no plugins)
- `spammm/GUI/DirectoryNavigator.py`, `ThumbnailCache.py`, `MoleculeViewer.py` — supporting modules

**Browser ↔ SPAMMM_GUI interaction (current state: none):**

Currently `MolecularBrowserVispy` and `SPAMMMWindow` (`SPAMMM_GUI.py`) are completely
independent. Browser opens `MoleculeViewer` (read-only 3D) on Enter. Editor has no
awareness of browser. We want: open molecule from browser in editor (single click),
batch AFM on selected molecules, save from editor → browser refreshes.

**Architecture:**

```
┌──────────────────────────────────────────────────────────┐
│  MolecularBrowserVispy                                   │
│                                                          │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │Thumbnail │  │ 3D Preview  │  │ Plugin Panel         │  │
│  │ Grid     │  │(MoleculeView│  │ (AFM, Vibration,     │  │
│  │          │  │er, read-only│  │  Relax, Edit...)     │  │
│  └──────────┘  └────────────┘  └──────────────────────┘  │
│       │              │                   │                │
│       │   Enter / Double-click           │                │
│       │              ▼                   │                │
│       │     ┌──────────────┐             │                │
│       │     │ editor_window│             │                │
│       │     │ exists?      │             │                │
│       │     └──┬────────┬──┘             │                │
│       │      Yes        No               │                │
│       │       ▼         ▼                │                │
│       │  load_molecule  MoleculeViewer   │                │
│       │  (SPAMMMWindow) (read-only 3D)   │                │
│       │       │                          │                │
│       │       ▼                          ▼                │
│       │  ┌────────────┐    ┌─────────────────────────┐   │
│       │  │SPAMMMWindow │    │ AFMulator (GPU headless)│   │
│       │  │(Editor)     │    │ - select 10 mols        │   │
│       │  │- edit atoms │    │ - loop: assign_params   │   │
│       │  │- save XYZ   │    │   → make_forcefield     │   │
│       │  │- Kekule     │    │   → run_scan → save PNG │   │
│       │  └────────────┘    └─────────────────────────┘   │
│       │       │                          │                │
│       │       ▼                          ▼                │
│       │  save → refresh thumbnails    AFM PNGs → overlay  │
│       │                              on thumbnails or     │
│       │                              show in side panel   │
└──────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Editing**: Browser gets optional `editor_window` ref (a `SPAMMMWindow` instance).
  Enter/double-click → if editor exists, call `gui_script_utils.load_molecule(editor_window, path)`
  (brings editor to front, loads XYZ into `MoleculeEditorBackend`); else opens
  `MoleculeViewer` as read-only 3D (current behavior).

- **AFM batch**: Browser calls `AFMulator` **directly** (headless GPU), NOT through
  SPAMMM_GUI. Reasons:
  - AFM is pure GPU computation, doesn't need GUI widgets
  - Batch processing 10 molecules doesn't make sense in single-molecule editor
  - `AFMulator` API is clean: `assign_params()` → `make_forcefield()` → `run_scan()` → save PNG
  - Results displayed as AFM thumbnails in browser grid or side panel

- **Save from editor → browser refresh**: `SPAMMMWindow.export_structure()` saves XYZ.
  Browser watches directory for changes (QFileSystemWatcher) and refreshes thumbnails.

- **Multi-select**: Shift+arrow or Ctrl+click for batch AFM on selected molecules.

**Implementation plan:**
1. Add `editor_window` parameter to `MolecularBrowserVispy.__init__()` (optional, None = standalone)
2. Modify `enter_view_mode()`: if `self.editor_window` → `load_molecule()` + raise editor; else current behavior
3. Add "AFM" action (keyboard shortcut or button): calls `AFMulator` headless for selected molecule(s)
4. Add multi-select support (Shift+arrow, Ctrl+click) for batch AFM
5. Port plugin system from FireCore (`MolBrowserPlugin`, `MolBrowserPluginHost`, `MolBrowserContext`)
6. Write SPAMMM plugins: AFM image, FDBM pipeline, vibration, relaxation
7. Deprecate/remove old PyOpenGL `MolecularBrowser.py`
8. Add `QFileSystemWatcher` for directory refresh after editor save

**Key design from FireCore:**
- CPU QPainter thumbnails (avoid second GL context)
- Lazy molecule loading (stat only for directory scan)
- Plugin relevance filtering per directory/molecule type
- Shared `MolBrowserContext` (directory, selection, loaded molecule, 3D viewer)

---

## 2. Robust pySCF Backend

**Goal:** Make pySCF a first-class quantum backend, using our own modified pySCF
optimized for SPM/FDBM workflows.

**Current state:**
- `spammm/quantum/pySCF_utils.py` — minimal (91 lines): basic SCF, geometry opt, density on grid
- No test coverage
- No integration with FDBM pipeline

**Plan:**
1. Locate and integrate our modified pySCF (user has custom version optimized for this purpose)
2. Add density grid projection (analogous to `Grid_dftb.py` for DFTB+)
3. Add pySCF as FDBM backend in `ModularPipeline.py` (alternative to DFTB+)
4. Test: electron count, density comparison vs DFTB, z-scan reference curves
5. The z-scan reference system already supports pySCF (`pyscf_pbe`, `pyscf_b3lyp`) — extend to full pipeline

---

## 3. Presentation Generation (HTML / Jupyter / ODP)

**Goal:** Enable LLM coding agents to efficiently generate presentations of results.

### 3a. HTML Presentations

**Approach:** Create a utility module that wraps result data into self-contained HTML slides.

**Design:**
- `spammm/utils/html_pres.py` — lightweight HTML slide generator
  - `Slide(title, content_html, image_path=None)` → `<section>` element
  - `Presentation(title, slides)` → self-contained `.html` with embedded CSS/JS
  - Use reveal.js or similar lightweight framework (embedded, no external deps)
  - Image embedding via base64 data URIs (portable, no file dependencies)
  - Agent-friendly: just pass data + images, get HTML
- Template: title slide, content slides with image+caption, summary slide
- Auto-generate from test artifacts: scan `debug/` for `.png` + `.json` results

### 3b. Jupyter Notebooks

**Approach:** Jupyter as interactive working mode combining visual + bash + code.

**Design:**
- Template notebooks for common workflows: AFM imaging, FDBM pipeline, relaxation
- `%matplotlib inline` + NumPy + pyopencl cells
- Bash cells (`!python tests/SPM/run_zscan_reference.py`) for running scripts
- Export to HTML for portable presentation

### 3c. ODP (LibreOffice/OpenOffice)

**Feasibility:** Direct `.odp` generation is possible but complex (ODF is XML-based).
- Python `odfpy` library can create ODP presentations programmatically
- Alternative: generate HTML → convert via `libreoffice --convert-to odp`
- **Recommendation:** Use HTML as primary format, convert to ODP/PPTX on demand via LibreOffice CLI
- This avoids maintaining ODP generation code; HTML is more agent-friendly

---

## 4. Presentation Planning

**Goal:** Decide what features to present, what should be on each slide, test thoroughly.

**Proposed slide structure (conference presentation):**

1. **Title** — SPAMMM: Scanning Probe Accelerated Modeling of Microscopy and Manipulation
2. **Architecture overview** — Python + OpenCL pipeline diagram
3. **Molecular topology editing** — hex grid, Kekule solver, electron pairs (show editing test plots)
4. **Force fields** — UFF/SPFF, energy-force correspondence, MD invariants (show conservation plots)
5. **Surface electrostatics** — Ewald2D, GPU-CPU parity (show z-scan/x-scan plots)
6. **Morse AFM imaging** — pentacene/PTCDA AFM images (show Fz/df maps)
7. **FDBM pipeline** — DFTB+ SCF → density → Pauli → ES → AFM (show Fz/df maps)
8. **Z-scan reference curves** — DFTB vs pySCF comparison (show overlay plots)
9. **Folded basis rigid body** — relaxation + manipulation on NaCl(100) (show trail plots)
10. **Contact surface AFM** — memory-efficient quasi-2D approach (show parity plots)
11. **GPU acceleration** — Jacobi eigendecomposition, OpenCL kernel design (show speedup table)
12. **Molecular browser** — ACDSee-style workflow demo (screenshot/animation)
13. **Conclusions & future work**

**Testing strategy for presentation:**
- Each slide's content must be backed by a passing test
- Visual outputs (`.png`) must be regenerated and reviewed before presentation
- Run `pytest --develop` to produce L1+L2 artifacts for all slides
- Create `tests/test_presentation.py` that asserts all needed plots exist and are non-empty

---

## 5. Molecule-on-Substrate Relaxation with FAF

**Goal:** Relax molecules on substrate using ForcedAtomicFunction (FAF) instead of full GridFF.

**Why FAF over GridFF:**
- FAF is smaller in memory (analytic functions vs precomputed 3D grid)
- GridFF requires B-spline fitting (expensive setup time)
- FAF can be evaluated on-the-fly at any point

**Current state:**
- `GridFF.py` — B-spline grid (tricubic), expensive setup, 4 channels (Pauli/London/Coulomb/Hbond energy)
- AFM uses trilinear `img_FF` (cl.Image) — 12 channels (Fx,Fy,Fz,E × Pauli/London/Coulomb)
- `Surface_utils.py` has `sample_gridff_trilinear()` for CPU trilinear sampling
- FAF mentioned in `elements.py` but not fully implemented as substrate relaxation backend

**Plan:**
1. Implement FAF as analytic substrate potential (Morse + Ewald2D Coulomb, no grid precomputation)
2. Integrate with `MolecularDynamics.py` for on-substrate relaxation
3. Compare FAF vs GridFF accuracy and performance
4. Use FAF as default for interactive GUI relaxation (fast setup), GridFF for production (fast evaluation)

---

## 6. GridFF Consolidation (Tricubic B-spline vs Trilinear)

**Goal:** Unify the two GridFF variants so they can be easily switched, not treated as separate systems.

**Current situation:**

| Variant | Interpolation | Channels | Setup | Used by |
|---------|--------------|----------|-------|---------|
| Tricubic B-spline | Analytic derivatives from B-spline coefs | 4 (E_Pauli, E_London, E_Coulomb, E_Hbond) | Expensive B-spline fit | `GridFF.py`, `MolecularDynamics.py` |
| Trilinear | Explicit force channels (Fx,Fy,Fz,E) | 12 (4 per component × 3 components) | Direct grid fill | `AFM.py` (`img_FF`), `AFM.cl` |

**Complication:** Molecule-molecule and molecule-substrate interactions are fused into one kernel
(for efficiency — eliminates kernel call overhead). This means we need multiple variants of the
combined kernel because:
- Trilinear GridFF requires explicit force channels (12 channels: {Fx,Fy,Fz} × {Pauli,London,Coulomb})
- B-spline GridFF requires only energy channels (4 channels: {Pauli,London,Coulomb,Hbond}) — derivatives computed analytically

**Plan:**
1. Define common `GridFFBase` interface: `sample(pos) → (E, F)` regardless of backend
2. `GridFFTrilinear` — current AFM-style, 12 channels, direct interpolation
3. `GridFFBspline` — current GridFF-style, 4 channels, analytic derivatives
4. `GridFFFAF` — analytic ForcedAtomicFunction, no grid at all
5. Kernel variants: generate combined kernel with `#define GRIDFF_TRILINEAR` / `#define GRIDFF_BSPLINE` / `#define GRIDFF_FAF` preprocessor switches
6. Switch at runtime via parameter, not separate code paths
7. Consolidate `gridFF.cl` and AFM grid interpolation into unified kernel module

---

## 7. Fast Relaxation with Collision Groups (AABB Bounding Boxes)

**Goal:** Projective/position-based relaxation with AABB collision detection for building
non-covalently bonded molecular clusters on surfaces.

**Use case:** Multiple PTCDA molecules on NaCl(100) — each molecule has its bounding box,
relax them simultaneously without inter-molecular overlap.

**Current state:**
- `RigidBodyDynamics.py` — 6-DOF rigid body, GPU, works for single molecule on folded basis
- `Assembly.py` — assembly clash scoring, visual only (no L0 test)
- No AABB collision detection or multi-body relaxation

**Reference (FireCore):**
- `RRsp3.cl` — `update_bboxes_rigid`, `build_local_topology_rigid`, `compute_collision_cluster_rigid`
- Cluster-sorted layout (64 atoms/workgroup), ghost atom halo, recoil buffers
- Jacobi iteration with heavy-ball momentum

**Plan:**
1. Add AABB bounding box computation per rigid body (GPU kernel)
2. Implement broad-phase collision detection (AABB overlap test)
3. Add collision response to `RigidBodyDynamics.py` (positional correction, not force-based)
4. Test: 2-10 PTCDA molecules on NaCl(100), relax to non-overlapping configuration
5. Extend to general molecular clusters (non-covalent assembly)

---

## 8. Reactive Force Field for Molecule Editing/Generation

**Goal:** Use reactive force field for editing and automatic generation of molecules.

**Reference implementation:**
- `/home/prokop/git/NumericalMathPlayground/topics/ReactiveFF/`
  - `reactiveff_ocl_app.py` — 2D OpenCL reactive FF with angular terms, torque, quaternion rotation
  - `reactiveff_ocl_app3d.py` — 3D version
  - `BoundingBoxBalancing.md` — design notes on AABB balancing
  - `RigidAtomicRotatingFrameFF.chat.md` — design discussion
  - `plot_basis_functions.py` — basis function visualization

**Key concepts from the code:**
- Per-atom orientation (quaternion) with rotated local frame
- Angular-dependent pair potential: `fang = x*(x²-3y²)` (cosine of 3θ)
- Force + torque evaluation in local frame, rotated back to lab frame
- Cutoff function: `(1-q)²` where `q = r²/rc²`
- OpenCL kernel with `PairEval` struct (force + torque)

**Plan:**
1. Study the ReactiveFF code for molecule generation patterns
2. Port relevant kernels to SPAMMM kernel module
3. Integrate with topology editing — use reactive FF to relax/generate new molecular structures
4. Automatic molecule generation: place atoms, let reactive FF find stable geometry
5. Test: generate simple molecules (H2O, NH3, benzene) from atom placement + reactive relaxation

---

## 9. Pentagon/Heptagon Drawing & SMILES Builder

**Status:** N-gon ring drawing implemented (edge + corner). SMILES builder not yet implemented.

**Current state:**
- `heterocycle_generator.py` mentions pentagons/heptagons in comments (zigzag rows create them)
- **N-gon ring placement implemented** in `MoleculeEditorBackend.py` + `EditModeHandlers.py`:
  - `compute_adjacent_ring_positions(bond, n_members, side)` — edge ring: n-gon sharing 1 bond
  - `compute_corner_ring_positions(atom, n_members, mouse_pos)` — corner ring: n-gon sharing 2 bonds at inner corner (angle < 180°); circumcircle through B-A-C, vector math (dot/cross products, no atan2); outer corners fall through to edge/hex
  - `add_corner_ring(atom, n_members, ename, mouse_pos)` — creates atoms + bonds for corner ring
  - `RingMode` in `EditModeHandlers.py` unifies all 3: bond → corner atom → hex center (priority order)
  - Ring size selectable via spinbox (3-8+), works with any n_members ≥ 3
  - Fused ring systems with mixed ring sizes supported (e.g. pentagon fused to hexagon)
- No SMILES parser or builder exists
- `doc/Tasks/ToDO_GUI.md` lists "ASCII-art molecule builder, as well as smiles"

**Remaining plan:**
1. SMILES builder: parse SMILES strings → AtomicGraph
   - Use RDKit or hand-rolled parser
   - SMILES → atom list + bonds → `AtomicGraph` → geometry assignment
   - Integrate with Kekule solver for bond order validation

---

## 10. GUI Verbosity Consolidation → `spammm.globals`

**Status:** `spammm/globals.py` exists (88 lines) with `DEBUG_PRINT_LEVEL`, `DEBUG_SAVE_LEVEL`,
`DEBUG_PLOT_LEVEL`, `debug_print()`, `set_develop_mode()`. But many modules still use their
own `verbose` / `verbosity` / `bPrint` variables.

**Current `globals.py` features:**
- `DEBUG_PRINT_LEVEL` — 0=silent, 1=warnings, 2=info, 3=debug (env: `SPAMMM_VERBOSITY`)
- `DEBUG_SAVE_LEVEL` — controls auxiliary array saving (env: `AFM_DEBUG_SAVE_LEVEL`)
- `DEBUG_PLOT_LEVEL` — controls figure generation (env: `AFM_DEBUG_PLOT_LEVEL`)
- `debug_print(level, msg)` — conditional print
- `set_develop_mode(on)` — bump levels for `pytest --develop`

**Problem:** 284 matches for `verbose|verbosity` across 44 files — many modules have
their own `self.verbose = True` or `bPrint=True` parameters that don't use `globals.py`.

**Plan:**
1. Audit all `verbose`/`verbosity`/`bPrint` parameters across codebase
2. Replace with `globals.debug_print(level, msg)` calls
3. Keep `bPrint` as function parameter but default to `None` → check `globals.DEBUG_PRINT_LEVEL`
4. Remove per-module verbosity state where possible
5. Document the convention: `from spammm.globals import debug_print, DEBUG_PRINT_LEVEL`

---

## 11. Fragment/Group Library & Substitution

**Status:** Not implemented. Design specified below. Fragment *detection* works (`AtomicGraph.find_fragments()`, `FragmentExtension.py`), but no library, attachment, or substitution exists.

**Current state:**
- `PASSIVATION_GROUPS` dict in `MoleculeEditorBackend.py` — 6 hardcoded edge terminations (N, NH, CH, H, O, C=O, C-OH). Applied via `_apply_passivation_group()`. Not extensible.
- `AtomicGraph.find_fragments()` — splits molecule by bridge bonds (tested, 5 tests)
- `FragmentExtension.py` — GUI panel for analysis: components, bridges, articulation points, biconnected components. Read-only visualization, no editing.
- FireCore has `attachGroupByMarker()` (marker-pair attachment), `attachParsedByDirection()` (directional attachment), `insertBridge()` / `collapseBridgeAt()` — none ported.

**Design: Group attachment by terminal count**

Groups are reusable molecular fragments stored in a library. Each group has 1 or 2 **terminal atoms** (dangling bonds) that connect to the host molecule. Attachment follows the same interaction pattern as ring placement (hover → preview → click to attach).

### Type 1: Single-terminal groups (1 dangling bond)

Simple substitution — group bonds to one atom.

```
Hover over atom A:          Click to attach:

    A                          A—K
                                |
                               group
```

- Hover over any atom → preview group attached at that atom
- Click → replace H cap (if present) with group, create bond A—K
- Example groups: –OH, –NH₂, –CH₃, –F, –Cl, –CHO, –COOH
- Terminal atom K replaces the H cap or bonds to the selected atom

### Type 2: Two-terminal (vicinal) groups (2 dangling bonds)

Group has two terminal atoms K, L with dangling bonds close together. Attaches across a bond or at a corner — exactly analogous to ring placement modes.

#### 2a. Edge attachment (like edge ring on bond A–B)

Group bridges across bond A–B. The bond A–B may be kept or broken depending on group type (addition vs substitution).

**Substitution** (A–B bond broken, replaced by A–K and B–L):
```
Before:        After:

A—B            A   B
                |   |
                K—L
               (group)
```

**Addition** (A–B bond kept, new bonds A–K and B–L added):
```
Before:        After:

A—B            A—B
                |   |
                K—L
               (group)
```

- Hover over bond → preview group on chosen side
- Side selection: mouse position determines which side of bond
- Example groups: –CH=CH– (vinylene), –N=N– (azo), –NH–NH– (hydrazine), fused ring systems

#### 2b. Corner attachment (like corner ring at inner corner atom B)

Group attaches at an inner corner atom B where angle A–B–C < 180°. Terminal K bonds to A, terminal L bonds to C. The corner atom B may be kept or removed depending on group type.

**Corner substitution** (B removed, A–K and C–L bonds formed):
```
Before:        After:

A—B—C          A     C
                 \   /
                  K—L
                 (group)
```

**Corner addition** (B kept, A–K and C–L bonds added alongside A–B and B–C):
```
Before:        After:

A—B—C          A—B—C
                |     |
                K—————L
               (group)
```

- Hover over inner corner atom B (angle < 180°) → preview group
- Outer corners (> 180°) fall through to edge or single-terminal mode
- Uses same corner detection as `compute_corner_ring_positions()` (dot/cross products, no atan2)

### Group library format

Each group entry stores:
- **Atoms**: list of (element, x, y, z) relative coordinates
- **Terminals**: list of terminal atom indices (1 or 2)
- **Terminal geometry**: for 2-terminal groups, the K–L distance and orientation
- **Bond mode**: 'substitute' (break host bond) or 'add' (keep host bond)
- **Name**: e.g. "hydroxyl", "vinylene", "azo"

Library stored as JSON or Python dict, loadable at runtime. Initial groups:
- 1-terminal: –H, –F, –Cl, –OH, –NH₂, –CH₃, –CHO, –COOH, –CN, –NO₂
- 2-terminal: –CH=CH–, –N=N–, –NH–NH–, –C≡C–, –CH₂–CH₂–, fused ring (e.g. naphthalene → anthracene)

### Interaction design (reuses Ring mode patterns)

- **Mode**: "Group" edit mode (or extension of Unified mode)
- **Group selector**: dropdown or palette to pick group from library
- **Hover priority** (same as Ring mode):
  1. Bond → edge attachment (2-terminal) or substitution at bond endpoint (1-terminal)
  2. Inner corner atom → corner attachment (2-terminal)
  3. Atom → single-terminal attachment
- **Preview**: ghost atoms + bonds shown on hover, like ring preview line
- **Click**: attach group, push undo
- **RMB**: remove group (reverse operation — detach + restore H caps)

### Implementation plan

1. Define `GroupLibrary` class with JSON-based group definitions
2. Implement `attach_group_single(backend, atom, group)` — single-terminal
3. Implement `attach_group_edge(backend, bond, group, side)` — 2-terminal edge
4. Implement `attach_group_corner(backend, atom, group, mouse_pos)` — 2-terminal corner (reuse corner detection from ring code)
5. Add "Group" edit mode to `EditModeHandlers.py` with hover priority: bond → corner → atom
6. Add group palette UI (dropdown + preview)
7. Port FireCore `attachGroupByMarker()` for marker-based attachment (advanced)
8. Port `insertBridge()` / `collapseBridgeAt()` for bridge operations

**Key reference:** FireCore `MoleculeUtils.js::attachGroupByMarker()`, `attachParsedByDirection()`
**Target module:** `spammm/topology/MoleculeBuilder.py` (new) or extend `MoleculeEditorBackend.py`

---

## Cross-References

| Topic | This doc section | FeatureChecklist | ToDo.agents | Other docs |
|-------|-----------------|-------------------|-------------|------------|
| Mol Browser | §1 | §7 GUI | Soon | `doc/Tasks/ToDO_GUI.md` |
| pySCF | §2 | §6 Quantum | — | `spammm/quantum/pySCF_utils.py` |
| Presentations | §3, §4 | — | — | — |
| FAF substrate | §5 | §3 Surface | — | `doc/surface_interactions.md` |
| GridFF consolidation | §6 | §3 Surface | — | `spammm/surfaces/README.md` |
| AABB collision | §7 | §5 RigidBody | Later | FireCore `RRsp3.cl` |
| Reactive FF | §8 | §2 ForceFields | Later | `NumericalMathPlayground/topics/ReactiveFF/` |
| Pentagon/SMILES | §9 | §1 Topology | Later | `doc/Tasks/ToDO_GUI.md` |
| GUI verbosity | §10 | §8 Infrastructure | Later | `spammm/globals.py` |
| Fragment/Group library | §11 | §1 Topology | Soon | `doc/Tasks/ToDO_GUI.md`, FireCore `MoleculeUtils.js` |
