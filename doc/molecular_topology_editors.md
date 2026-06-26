# Molecular Topology — Editors & Advanced Editing

> Cross-language audit of interactive molecular editors: GUI architectures, advanced editing operations, and consolidation roadmap. See [base topology](molecular_topology.md) for graph representations, [type assignment](molecular_topology_types.md) for typing, and [interactive codemap](https://windsurf.com/codemaps/692593e6-1efe-495f-bbf6-2ad291a285c9-fe86ab10a43f3d18).

## Quick Navigation

| I want to... | Go to |
|--------------|-------|
| Understand graph representations | [molecular_topology.md](molecular_topology.md) |
| Understand type assignment | [molecular_topology_types.md](molecular_topology_types.md) |
| See which editor to use | [Editor Decision Tree](#which-editor-should-i-use) |
| See what needs consolidation | [Redundancies & Roadmap](#redundancies--consolidation-roadmap) |

---

## All GUIs Compared

| GUI | Technology | Backend | Status | Maintained? |
|-----|-----------|---------|--------|-------------|
| **KekuleExplorerGUI** | Python / Vispy / PyQt5 | `KekuleBackend` + `AtomicGraph` | **Active** | Yes |
| **MoleculeEditor2D** | Python / Matplotlib / PyQt5 | `AtomicSystem` directly | Experimental | **No** |
| **molgui_web** | JS / WebGL | `EditableMolecule` + `MMParams` | **Legacy** | No |
| **molgui_webgpu** | JS / WebGPU | `EditableMolecule` + `MMParams` + `SPFFLTopology` | **Active** | Yes |

---

## Which Editor Should I Use?

```
Need to edit molecules in Python?
├── Need 3D visualization + hex grid? → KekuleExplorerGUI
└── Need simple 2D sketching? → MoleculeEditor2D (experimental)

Need to edit molecules in browser?
├── New project / production → molgui_webgpu
└── Legacy WebGL code → migrate to molgui_webgpu
```

**Recommendation**: Use `KekuleExplorerGUI` (Python) or `molgui_webgpu` (JS). `MoleculeEditor2D` and `molgui_web` are effectively deprecated.

---

## Editor Architecture Patterns

### Pattern A: Direct Manipulation (KekuleExplorerGUI)
- GUI widgets directly call backend mutation methods
- Backend emits signals for GUI refresh
- No command queue / undo stack

### Pattern B: Document Model (MoleculeEditor2D)
- Document wrapper around AtomicSystem
- Explicit `auto_bonds()`, `auto_types()` calls
- Grid snapping modes (triangular, square, none)

### Pattern C: Versioned Object Graph (EditableMolecule)
- `topoVersion` counter invalidates cached indices
- `lockTopology()` / `unlockTopology()` for batch ops
- `id2atom` Map for stable object identity

```javascript
// EditableMolecule.js
lockTopology() {
    this.lockDepth++;
    if (this.lockDepth === 1) this._topoVersionLocked = this.topoVersion;
}

addAtom(x, y, z, Z) {
    this._assertUnlocked("addAtom");
    this._touchTopo();  // increments topoVersion
}
```

---

## Python Editors

### KekuleExplorerGUI (Active)

**File**: `pyBall/KekuleExplorerGUI.py` (1413 lines)

- Vispy-based 3D scene with PyQt5 widgets
- Hexagonal grid snapping for graphene-like structures
- Atom type selector panel
- Auto hydrogen capping
- Passivation groups (N, NH, CH, H, O, C=O, C-OH)
- Bond visualization
- XYZ export

**Backend**: `KekuleBackend.py` (1975 lines)
- `AtomicGraph` as authoritative state
- `AtomicSystem` for rendering/export (synced via `_sync_sys()`)
- `_target_sigma()`: Calculate target sigma bonds from valence + hybridization
- Ring detection: `detect_geometry_rings()` → `graph.detect_rings()`

**Key method**: `add_atom(pos, ename, atype, pin=None, parent=None, subtype='')`
- `pin`: hex grid node key `(rx, ry)`
- `parent`: parent atom (for H capping)
- `subtype`: 'C_sp2', 'H_cap', etc.

### MoleculeEditor2D (Deprecated)

**File**: `pyBall/GUI/MoleculeEditor2D.py` (1602 lines)

- Matplotlib-based 2D canvas
- Operates directly on `AtomicSystem`
- Modes: draw, move, bond, rotate
- Element canonicalization
- Bond order management
- **Status**: Less maintained than KekuleExplorerGUI. Consider for removal or archival.

---

## JavaScript Editors

### molgui_webgpu (Active)

**File**: `web/molgui_webgpu/main.js` + supporting modules

- WebGPU compute shaders for real-time MD
- XPBD/SPFFL force fields
- Parameter inspection
- Editing + simulation in same interface

**Key modules**:
- `EditableMolecule.js` — topology editing
- `MMParams.js` — parameter loading
- `SPFFLTopology.js` — topology → XPDB packing
- `MoleculeRenderer.js` — WebGPU rendering
- `GUI.js` — UI components

### molgui_web (Legacy)

**File**: `web/molgui_web/` directory

- Original WebGL implementation
- Same `EditableMolecule.js` and `MMParams.js` duplicated
- **Status**: Deprecated, plan migration to WebGPU

---

## Advanced Editing Operations

### Insert Ring / Remove Ring

| Language | Function | Backend |
|----------|----------|---------|
| **Python** | `KekuleBackend.add_ring()`, `remove_ring()` | `AtomicGraph` |
| **C++** | Not explicit | — |
| **JavaScript** | Not explicit | — |

**Python hex grid**: `KekuleBackend` defines hexagonal grid coordinates (`pin = (rx, ry)`). Adding a ring places atoms at hex vertices.

### Collapse Bond / Split by Bond

| Language | Function | File |
|----------|----------|------|
| **C++** | `MM::Builder::splitByBond()` | `SPFFBuilder.h` |
| **Python** | Not explicit | — |
| **JavaScript** | Not explicit | — |

**C++**: `splitByBond(ib)` removes bond `ib` and reassigns atoms to separate fragments.

### Fragment Manipulation

| Language | Class/Function | Features |
|----------|---------------|----------|
| **C++** | `Fragment` struct | `atomIds`, `bondIds`, `bounds`, `isClosed` |
| **JavaScript** | `Fragment` class | `atomIds`, `bondIds`, `bounds`, `updateBounds()` |
| **Python** | Not explicit | — |

---

## Geometry Transforms

| Language | Translate | Rotate | Orient |
|----------|-----------|--------|--------|
| **C++** | `move_atoms(dshift)` | `rotate_atoms(angle, axis)` | `orient_atoms(fw, up)` |
| **Python** | `AtomicSystem.translate()` | `AtomicSystem.rotate()` | — |
| **JavaScript** | `EditableMolecule.translateAtoms()` | `EditableMolecule.rotateAtoms()` | — |

**Critical bug fixed** (see memory): JS `translateAtoms` and `rotateAtoms` were using atom IDs as array indices instead of looking up `id2atom` Map. Fixed by using `mol.getAtomIndex(id)`.

---

## Redundancies & Consolidation Roadmap

### Critical: EditableMolecule.js Duplicate

| File | Lines | Status |
|------|-------|--------|
| `web/molgui_web/js/EditableMolecule.js` | 1057 | **Legacy** |
| `web/molgui_webgpu/EditableMolecule.js` | 1057 | **Active** |

**Action**: Move shared code to `web/common_js/EditableMolecule.js`. Update both molgui_web and molgui_webgpu to import from common location. Remove duplicate files.

### Critical: MMParams.js Versions

| File | Lines | Features | Status |
|------|-------|----------|--------|
| `web/molgui_web/js/MMParams.js` | 466 | Basic parsing | **Legacy** |
| `web/molgui_webgpu/MMParams.js` | 524 | Logger, verbosity, QEq | **Active** |

**Action**: Use WebGPU version as canonical. Update molgui_web to import from molgui_webgpu or common location. Deprecate WebGL version.

### Moderate: Selection Systems

| Language | Implementation | Status |
|----------|---------------|--------|
| **C++** | `std::unordered_set<int>` in SPFFBuilder | Keep |
| **Python** | Set-based in AtomicSystem | Keep |
| **JavaScript** | `Selection.js` with banks | Keep |

**Rationale**: Different approaches for different use cases; no consolidation needed.

### Moderate: Graph Representations

| Language | Representations | Recommendation |
|----------|----------------|---------------|
| **Python** | `AtomicGraph` (object) + `AtomicSystem` (array) | Standardize on one |
| **C++** | Integer-indexed arrays only | Keep |
| **JavaScript** | Object ID maps only | Keep |

**Python decision needed**: `AtomicGraph` for editing, `AtomicSystem` for FF conversion. Document when to use each.

### Low: Parameter File Parsing

Three separate parsers for same `.dat` format (C++, Python, JS).

**Options**:
1. Keep separate (language-specific needs)
2. Add JSON export from one canonical parser
3. Document parity and keep as-is

---

## Crystal Building — Cross-Language Gap Analysis

### Overview

Crystal building is a **critical gap** in the Python ecosystem. The JavaScript `CrystalUtils.js` (1188 lines) provides a complete crystallography toolkit with no Python equivalent. This blocks the goal of a unified Python VisPy GUI.

### JavaScript Implementation (Active, Complete)

**`web/molgui_webgpu/CrystalUtils.js`** (1188 lines) — the canonical crystal builder:

| Function | Lines | Purpose |
|----------|-------|---------|
| `latticeVectorsFromParams(params)` | 229-255 | Build 3 lattice vectors from (a, b, c, α, β, γ) |
| `fracToCart(p, lvec, out)` | 257-264 | Fractional → Cartesian coordinate conversion |
| `cartToFrac(p, lvec, out)` | 266-279 | Cartesian → Fractional coordinate conversion |
| `parseLatticeText(txt)` | 281-293 | Parse 3-line text input as lattice vectors |
| `parseSitesTextXYZ(txt, mode)` | 308-327 | Parse atomic sites (fractional or Cartesian) |
| `parseSymOpsText(txt)` | 382-391 | Parse symmetry operation strings |
| `parseSymOpXYZ(opStr)` | 485-498 | Parse single sym op "x,y,z" → matrix + translation |
| `parseCIF(cifText)` | 418-466 | Full CIF parser (tags + loops) |
| `cifToCrystalData(cifText)` | 540-580 | Extract lattice, sites, symmetry from CIF |
| `applySymmetryOpsFracSites(sites, symOps)` | 500-528 | Apply symmetry operations to fractional sites |
| `dedupFracSitesByTolA(sites, lvec, tol)` | 104-157 | Grid-bucket deduplication of fractional sites |
| `dedupMolAtomsByTolA(mol, tol)` | 159-227 | Dedup atoms in EditableMolecule by distance |
| `cellDataFromFracSites(lvec, sites)` | 582-602 | Convert fractional sites → flat arrays (pos, types, charges) |
| `genReplicatedCell(params)` | 928-1026 | Replicate unit cell (bulk) with optional bond building |
| `genReplicatedCellSlab(params)` | 667-750 | Replicate with slab cut (HKL normal, cmin/cmax range) |
| `genReplicatedCellCutPlanes(params)` | 752-864 | Replicate with arbitrary plane cuts (multiple planes) |
| `reciprocalLattice(lvec)` | 1063-1083 | Compute reciprocal lattice vectors |
| `expandPlaneTemplates(templates, cSym)` | 11-29 | Expand {100}, {110}, {111} plane families |
| `_computeBasisBonds(lvec, basisPos, types, mmParams)` | 877-926 | Compute bonds within unit cell + 27 neighbor offsets |

**`web/molgui_webgpu/Nanocrystals.js`** (631 lines):
- `buildCrystalFromCIFText(cifText, opts)` — high-level CIF → crystal pipeline
- `buildPlanesFromTemplates(lvec, templates, ...)` — plane template expansion for slabs
- `loadMMParams(args)` — load FF parameters for bond building

**`web/molgui_webgpu/BuildersGUI.js`** (1130 lines):
- Unit cell editor GUI (lattice vectors, atoms, symmetry, dedup controls)
- CIF file loading → fill editor fields
- Crystal generation (bulk, slab, cut planes) with replication counts
- Preset crystal structures (NaCl, CaF2, diamond, etc.)
- Preview rendering with cell box, atom spheres, bond lines

### Python Implementation (Missing)

| Capability | JS Status | Python Status |
|-----------|-----------|---------------|
| Lattice vectors from params | ✅ Complete | ❌ **Missing** |
| Frac↔Cart conversion | ✅ Complete | ❌ **Missing** |
| CIF parsing | ✅ Complete | ❌ **Missing** (no Python CIF parser in pyBall) |
| Symmetry operations | ✅ Complete | ❌ **Missing** |
| Site deduplication | ✅ Grid-bucket | ❌ **Missing** |
| Bulk cell replication | ✅ Complete | ❌ **Missing** |
| Slab cutting (HKL) | ✅ Complete | ❌ **Missing** |
| Plane cutting | ✅ Complete | ❌ **Missing** |
| Bond building across cells | ✅ Complete | ❌ **Missing** |
| Reciprocal lattice | ✅ Complete | ❌ **Missing** (exists in `doc/Julia/EwaldGrid.jl` only) |
| Nanocrystal generation | ✅ Complete | ❌ **Missing** |
| Preset crystals | ✅ Complete | ❌ **Missing** |

**Note:** `KekuleBackend.py` (1975 lines) builds hexagonal graphene grids only — not general crystallography.

### C++ Implementation (Partial)

`SPFFBuilder` (`cpp/common/molecular/SPFFBuilderBase.h`, 808 lines) has:
- `insertAtoms(n, pos, types)` — add atoms to builder
- `autoBonds(rCut)` — distance-based bond finding
- No CIF parsing, no lattice vectors, no symmetry, no slab cutting

### Recommended Python Port Plan

Create `pyBall/crystal/CrystalBuilder.py` with these functions (mapping from JS):

```python
# Core crystallography
def lattice_vectors_from_params(a, b=None, c=None, alpha=90, beta=90, gamma=90) -> np.ndarray(3,3)
def frac_to_cart(frac, lvec) -> np.ndarray(N,3)
def cart_to_frac(cart, lvec) -> np.ndarray(N,3)
def reciprocal_lattice(lvec) -> np.ndarray(3,3)

# CIF parsing
def parse_cif(cif_text) -> dict  # tags + loops
def cif_to_crystal_data(cif_text) -> dict  # lattice, sites, sym_ops

# Symmetry
def parse_sym_op_xyz(op_str) -> (matrix, translation)
def apply_symmetry_ops_frac_sites(sites, sym_ops, tol=1e-6) -> list

# Deduplication
def dedup_frac_sites_by_tol(sites, lvec, tol=0.1) -> list

# Cell generation
def gen_replicated_cell(lvec, basis_pos, basis_types, n_rep=(1,1,1), ...) -> AtomicGraph
ndef gen_replicated_cell_slab(lvec, basis_pos, basis_types, n_rep, n_hat, cmin, cmax, ...) -> AtomicGraph
def gen_replicated_cell_cut_planes(lvec, basis_pos, basis_types, n_rep, planes, ...) -> AtomicGraph

# Bond building
def compute_basis_bonds(lvec, basis_pos, basis_types, mm_params, bond_tol=0.2) -> list
```

**Key design decisions:**
- Use NumPy arrays (not Vec3 objects) for positions — vectorized, cache-friendly
- Return `AtomicGraph` instances (not raw arrays) for GUI integration
- Use `mm_params` from existing Python FF parameter loading (`pyBall/OCL/UFFbuilder.py`)
- Fail loudly on invalid inputs (matching JS error behavior)

---

## Consolidation Checklist

- [ ] **Port `CrystalUtils.js` → Python `CrystalBuilder.py`** (CRITICAL — blocks unified GUI)
- [ ] Move `EditableMolecule.js` to `web/common_js/`
- [ ] Consolidate `MMParams.js` to single version
- [ ] Archive or remove `MoleculeEditor2D.py`
- [ ] Document `AtomicGraph` vs `AtomicSystem` usage in Python
- [ ] Plan molgui_web → molgui_webgpu migration
- [ ] Add cross-language parity tests for topology operations
- [ ] Port VSEPR capping from `EditableMolecule.js` → Python
- [ ] Create `crystal_building.md` audit document
- [ ] Build unified VisPy GUI combining KekuleExplorer + crystal builder + VSEPR

---

## File Index

### Python
- `pyBall/KekuleExplorerGUI.py` (1413 lines) — active 3D editor
- `pyBall/KekuleBackend.py` (1975 lines) — hex grid backend (graphene only, NOT general crystallography)
- `pyBall/GUI/MoleculeEditor2D.py` (1602 lines) — deprecated 2D editor
- `pyBall/AtomicGraph.py` (392 lines) — object-based molecular graph
- `pyBall/AtomicSystem.py` (1314 lines) — array-based molecular system

### JavaScript (Active)
- `web/molgui_webgpu/CrystalUtils.js` (1188 lines) — **crystal builder (no Python equivalent)**
- `web/molgui_webgpu/Nanocrystals.js` (631 lines) — nanocrystal generation from CIF
- `web/molgui_webgpu/BuildersGUI.js` (1130 lines) — crystal builder GUI
- `web/molgui_webgpu/EditableMolecule.js` (1057 lines) — topology editor
- `web/molgui_webgpu/SPFFLTopology.js` (826 lines) — XPDB packing
- `web/molgui_webgpu/MMParams.js` (524 lines) — parameters
- `web/molgui_webgpu/MoleculeRenderer.js` — WebGPU rendering
- `web/molgui_webgpu/GUI.js` — UI components
- `web/molgui_webgpu/Selection.js` — selection system

### JavaScript (Legacy)
- `web/molgui_web/js/EditableMolecule.js` (1057 lines) — **[DUPLICATE]**
- `web/molgui_web/js/MMParams.js` (466 lines) — **[LEGACY]**

### C++
- `cpp/common/molecular/SPFFBuilderBase.h` (808 lines) — FF builder with autoBonds (no crystallography)
- `cpp/apps/MolecularEditor/MolGUI.h` — C++ SDL-based GUI

---

## See Also

- [Base Topology](molecular_topology.md) — graph representations, bond finding, rings, bridges
- [Type Assignment](molecular_topology_types.md) — atom type assignment, parameter files
- [GUI Feature Audit](gui_audit.md) — detailed visualization & editor feature matrices, consolidation plan
- [Topical Audit Index](topical_audit.md) — priority ranking, dependency graph, missing topics
- [Interactive Codemap](https://windsurf.com/codemaps/692593e6-1efe-495f-bbf6-2ad291a285c9-fe86ab10a43f3d18) — visual navigation
- Canonical default parameters: `tests/tUFF/data_UFF/{ElementTypes,AtomTypes,BondTypes,AngleTypes,DihedralTypes}.dat`

---



Let me also check the actual SPAMMM KekuleBackend for all public mutation/editing methods:






Now let me check what the FireCore ScriptRunner offers (from the previous session's analysis) and also check the AtomicSystem for selection operations:

Now I have a comprehensive picture. Let me compile the full review:

Now I have the full picture. Here's the comprehensive review:

---

## Complete Inventory of Molecular Editing Operations

Based on [doc/molecular_topology.md](cci:7://file:///home/prokop/git/SPAMMM/doc/molecular_topology.md:0:0-0:0), [doc/molecular_topology_editors.md](cci:7://file:///home/prokop/git/SPAMMM/doc/molecular_topology_editors.md:0:0-0:0), and actual SPAMMM code, here are all editing operations/actions that can be done, organized by category and testability:

### 1. Atom-Level Operations (SPAMMM Python — implemented)

| # | Operation | API | What Changes | Testable? |
|---|-----------|-----|--------------|-----------|
| 1 | **Add atom at grid position** | [KekuleBackend.set_atom_type(node_key, element)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:432:4-451:27) | New atom at hex grid node, auto H-caps | ✅ L1+L2 |
| 2 | **Add atom at arbitrary position** | [KekuleBackend.add_atom_at_position(pos, element, npi)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:528:4-546:17) | New atom off-grid | ✅ L1+L2 |
| 3 | **Remove atom by grid key** | [KekuleBackend.remove_atom(node_key)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:548:4-564:27) | Atom + its H caps removed, bonds broken | ✅ L1+L2 |
| 4 | **Remove atom by index** | [KekuleBackend.remove_atom_by_index(atom_idx)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:566:4-577:27) | Same, index-based | ✅ L1+L2 |
| 5 | **Change element type** | [KekuleBackend.set_atom_type(node_key, new_element)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:432:4-451:27) | Atom.ename, atype, subtype change, H-caps re-adjust | ✅ L1+L2 |
| 6 | **Change element by index** | [KekuleBackend.set_atom_type_by_index(idx, element)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:453:4-463:28) | Same, index-based | ✅ L1+L2 |
| 7 | **Change hybridization (npi)** | [KekuleBackend.set_atom_valency(node_key, npi)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:519:4-526:27) | Atom.subtype (sp2↔sp3↔sp), H-caps re-adjust | ✅ L1+L2 |
| 8 | **Set subtype by index** | [KekuleBackend.set_atom_subtype_by_index(idx, subtype)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:182:4-186:49) | Direct subtype assignment | ✅ L1 |

### 2. Bond-Level Operations (SPAMMM Python — implemented)

| # | Operation | API | What Changes | Testable? |
|---|-----------|-----|--------------|-----------|
| 9 | **Insert atom into bond** | [KekuleBackend.insert_atom_into_bond(bond, ename)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:579:4-657:23) | A-B → A-C-B, original atoms pushed aside, 1 bond removed, 2 added | ✅ L1+L2 |
| 10 | **Collapse bond** | [KekuleBackend.collapse_bond(bond, mouse_pos)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:659:4-766:23) | A-B → survivor (farther from mouse), neighbor bonds transferred, survivor moved to center | ✅ L1+L2 |
| 11 | **Add bond** | `AtomicGraph.add_bond(a, b, order)` | New bond between two atoms | ✅ L1 |
| 12 | **Remove bond** | `AtomicGraph.remove_bond(bond)` | Bond removed, atoms remain | ✅ L1 |
| 13 | **Recalculate all bonds** | [KekuleBackend.recalc_bonds()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:929:4-943:54) / [AtomicGraph.recalc_bonds()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:929:4-943:54) | All bonds cleared, recomputed from distance | ✅ L1+L2 |

### 3. Ring/Hex Operations (SPAMMM Python — implemented)

| # | Operation | API | What Changes | Testable? |
|---|-----------|-----|--------------|-----------|
| 14 | **Add hex ring** | [KekuleBackend.add_ring(q, r)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:311:4-339:27) | Up to 6 C atoms + 6 bonds at hex vertices, H-caps adjusted | ✅ L1+L2 |
| 15 | **Remove hex ring** | [KekuleBackend.remove_ring(q, r)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:341:4-386:27) | Atoms at 6 nodes removed (shared atoms handled by hex_mode), H-caps adjusted | ✅ L1+L2 |
| 16 | **Toggle hex ring** | [KekuleBackend.toggle_ring(q, r)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:388:4-393:31) | Add if absent, remove if present | ✅ L1+L2 |
| 17 | **Detect rings** | [KekuleBackend.detect_geometry_rings()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:252:4-263:20) / `AtomicGraph.detect_rings()` | Ring objects created from bond graph DFS | ✅ L1 |

### 4. Hydrogen Cap Management (SPAMMM Python — implemented)

| # | Operation | API | What Changes | Testable? |
|---|-----------|-----|--------------|-----------|
| 18 | **Auto-adjust H caps** | [KekuleBackend.adjust_h()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:820:4-832:78) | Add/remove H to match valence: `nsigma = nvalence - npi - nepair` | ✅ L1+L2 |
| 19 | **Add all H caps** | [KekuleBackend.add_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:774:4-818:28) | Add H to all undercoordinated heavy atoms | ✅ L1+L2 |
| 20 | **Remove all H caps** | [KekuleBackend.remove_h_caps()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:768:4-772:36) | All H_cap atoms soft-deleted | ✅ L1+L2 |

### 5. Picking / Selection (SPAMMM Python — implemented)

| # | Operation | API | What Changes | Testable? |
|---|-----------|-----|--------------|-----------|
| 21 | **Pick atom** | [graph.pick_atom(pos, radius=0.5)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:267:4-269:48) | Returns Atom or None | ✅ L1 |
| 22 | **Pick bond** | [graph.pick_bond(pos, radius=0.5)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:271:4-273:48) | Returns Bond or None | ✅ L1 |
| 23 | **Pick ring** | [graph.pick_ring(pos, radius=1.0)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:275:4-277:48) | Returns Ring or None | ✅ L1 |
| 24 | **Grow selection** | `AtomicSystem.grow_selection(mask)` | Expand by 1 bonded shell | ✅ L1 |
| 25 | **Shrink selection** | `AtomicSystem.shrink_selection(mask)` | Contract by 1 shell | ✅ L1 |
| 26 | **Select connected** | `AtomicSystem.select_all_connected(i0)` | BFS/DFS from atom i0 | ✅ L1 |

### 6. Geometry Transforms (SPAMMM Python — implemented)

| # | Operation | API | What Changes | Testable? |
|---|-----------|-----|--------------|-----------|
| 27 | **Translate system** | [AtomicSystem.translate(vec)](cci:1://file:///home/prokop/git/FireCore/web/molgui_webgpu/ScriptRunner.js:411:4-432:5) | All atom positions shifted | ✅ L1 |
| 28 | **Rotate system** | `AtomicSystem.rotate_ax(ang, ax)` | All atom positions rotated | ✅ L1 |
| 29 | **Rotate subset** | `AtomicSystem.rotate_subset(indices, ang, pivot)` | Only selected atoms rotated | ✅ L1 |

### 7. Ribbon/Structure Builders (SPAMMM Python — implemented)

| # | Operation | API | What Changes | Testable? |
|---|-----------|-----|--------------|-----------|
| 30 | **Build zigzag ribbon** | [KekuleBackend.build_zigzag_ribbon(width, length, passivation, ...)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:1236:4-1288:19) | Full graphene zigzag ribbon with edge passivation | ✅ L1+L2 |
| 31 | **Build two-ribbon cell** | [KekuleBackend.build_two_ribbon_cell(width, length, Lx, L_Hb, ...)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:1674:4-1732:19) | Two ribbons + H-bond gap, periodic | ✅ L1+L2 |
| 32 | **Build ribbon (module-level)** | [KekuleBackend.build_ribbon(passivation, width, length, Lx)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:1888:0-1921:31) | Returns arrays (pos2d, atypes, elems) | ✅ L1 |

### 8. I/O Operations (SPAMMM Python — implemented)

| # | Operation | API | What Changes | Testable? |
|---|-----------|-----|--------------|-----------|
| 33 | **Save XYZ** | [KekuleBackend.save_xyz(fname)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:1028:4-1034:76) | Write system to XYZ file | ✅ L1 (roundtrip) |
| 34 | **Load XYZ** | [KekuleBackend.load_xyz(fname)](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:1036:4-1070:40) | Read XYZ, rebuild graph + grid mapping | ✅ L1 (roundtrip) |
| 35 | **Export to string** | `KekuleBackend.to_xyz_string()` | XYZ format string | ✅ L1 |

### 9. Graph Maintenance (SPAMMM Python — implemented)

| # | Operation | API | What Changes | Testable? |
|---|-----------|-----|--------------|-----------|
| 36 | **Cleanup dead objects** | `AtomicGraph.cleanup_invalid()` | Remove soft-deleted atoms/bonds/rings | ✅ L1 |
| 37 | **Sync neighbor lists** | `AtomicGraph.sync_neighbor_lists()` | Rebuild neighbor lists from alive bonds | ✅ L1 |
| 38 | **Sync sys from graph** | [KekuleBackend._sync_sys()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:239:4-250:35) | Rebuild AtomicSystem arrays from graph | ✅ L1 |
| 39 | **Build lattice vectors** | [KekuleBackend.build_lattice_vectors()](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:962:4-983:73) | Compute lattice from bounding box + PBC | ✅ L1 |

### 10. FireCore JS Operations (not yet ported — reference implementations)

| # | Operation | Source | Testable after port? |
|---|-----------|--------|---------------------|
| 40 | **Selection query** | `MoleculeSelection.js: compileSelectQuerySpec, applySelectQuery` | ✅ L1+L2 |
| 41 | **Select bridge candidates** | `MoleculeSelection.js: selectBridgeCandidates` | ✅ L1+L2 |
| 42 | **Collapse bridge** | `MoleculeUtils.js: collapseBridgeAt` | ✅ L1+L2 |
| 43 | **Insert bridge** | `MoleculeUtils.js: insertBridge` | ✅ L1+L2 |
| 44 | **Remove undercoordinated atoms** | `ScriptRunner.js: removeUndercoordinatedAtoms` | ✅ L1+L2 |
| 45 | **Collapse undercoordinated atoms** | `ScriptRunner.js: collapseUndercoordinatedAtoms` | ✅ L1+L2 |
| 46 | **Add caps** | `ScriptRunner.js: addCaps` / `EditableMolecule.js: addCappingAtoms` | ✅ L1+L2 |
| 47 | **Build polymer** | `ScriptRunner.js: buildPolymer` | ✅ L1+L2 |
| 48 | **Build nanocrystal** | `ScriptRunner.js: buildNanocrystal` | ✅ L1+L2 |
| 49 | **Build substrate (crystal)** | `ScriptRunner.js: buildSubstrate` / [CrystalUtils.js](cci:7://file:///home/prokop/git/FireCore/web/molgui_webgpu/CrystalUtils.js:0:0-0:0) | ✅ L1+L2 |
| 50 | **Replicate cell** | `CrystalUtils.js: genReplicatedCell` | ✅ L1+L2 |
| 51 | **Slab cutting (HKL)** | `CrystalUtils.js: genReplicatedCellSlab` | ✅ L1+L2 |
| 52 | **Plane cutting** | `CrystalUtils.js: genReplicatedCellCutPlanes` | ✅ L1+L2 |
| 53 | **CIF parsing** | `CrystalUtils.js: parseCIF, cifToCrystalData` | ✅ L1 |
| 54 | **Symmetry operations** | `CrystalUtils.js: applySymmetryOpsFracSites` | ✅ L1 |
| 55 | **Site deduplication** | `CrystalUtils.js: dedupFracSitesByTolA` | ✅ L1 |
| 56 | **Bonds across cells** | `CrystalUtils.js: _computeBasisBonds` | ✅ L1 |

### 57. C++ Only Operations (not in Python)

| # | Operation | Source | Notes |
|---|-----------|--------|-------|
| 57 | **Split by bond** | `SPFFBuilder.h: splitByBond(ib)` | Remove bond, split into fragments |
| 58 | **Bridge finding (Tarjan)** | `LimitedGraph.h: bridge()` | Find critical bonds — **not in Python** |

### Summary by Test Priority

**Already testable in SPAMMM (operations 1-39):** All have Python API, operate on `AtomicGraph`/[KekuleBackend](cci:2://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:118:0-1883:95)/`AtomicSystem`. Ready for `TopologySnapshot`/`TopologyDiff` testing.

**Most important to test first (high impact, likely to have bugs):**
1. [add_ring](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:311:4-339:27) / [remove_ring](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:341:4-386:27) — core hex grid editing, shared atom handling
2. [insert_atom_into_bond](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:579:4-657:23) — bond splitting, position adjustment
3. [collapse_bond](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:659:4-766:23) — neighbor transfer, mouse-position-dependent survivor selection
4. [adjust_h](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:820:4-832:78) — valence counting, H direction calculation
5. [set_atom_type](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:432:4-451:27) — element change with H-cap re-adjustment
6. [recalc_bonds](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:929:4-943:54) — distance-based rebuild
7. [load_xyz](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:1036:4-1070:40) / [save_xyz](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:1028:4-1034:76) roundtrip — graph reconstruction from file
8. [build_zigzag_ribbon](cci:1://file:///home/prokop/git/SPAMMM/spammm/topology/KekuleBackend.py:1236:4-1288:19) — complex builder with passivation

**Critical gaps (no Python implementation):**
- Bridge finding (Tarjan's algorithm) — C++ only
- Selection query system — JS only
- Bridge collapse/insert — JS only
- Crystal building (CIF, lattice, symmetry, slab) — JS only
- Fragment manipulation — C++/JS only