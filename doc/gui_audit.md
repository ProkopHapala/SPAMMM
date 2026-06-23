# GUI and Molecular Viewer/Editor Audit

## Executive Summary

FireCore has **significant GUI fragmentation** across 3 languages (C++, Python, JavaScript) with **multiple overlapping implementations** of molecular viewers, editors, and browsers. This audit identifies **22+ distinct GUI applications** with substantial code duplication and inconsistent architectures.

**Key Findings:**
- **Python**: 10+ GUI applications using 3 different rendering backends (VisPy, PyQt+OpenGL, PyQt+Matplotlib)
- **C++**: 1 massive monolithic GUI (MolGUI.h: 2922 lines) plus multiple editor apps
- **JavaScript**: 2 parallel web implementations (WebGL legacy vs WebGPU modern) with duplicate code
- **Major duplication**: MolViewer.py ≈ MolBrowser.py (near-identical code)
- **Technology sprawl**: VisPy, PyQt5, OpenGL, SDL2, WebGL, WebGPU, Matplotlib all used

---

## Python GUI Applications

### 1. VisPy-Based (GPU-Accelerated 3D)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `pyBall/ExplorerVisPy.py` | Molecule-substrate interaction scanning (GridFF) | 1104 | Active |
| `pyBall/KekuleExplorerGUI.py` | Kekulé structure explorer with hex grid editing | 1413 | Active |
| `pyBall/SequencePlacerVisPy.py` | Molecule placement on NaCl step edges | 376 | Active |
| `pyBall/VispyUtils.py` | Shared VisPy utilities (AtomScene, bond colors) | 1579 | Shared |
| `pyBall/RigidAtomFF/RRsp3/test_RRsp3_vispy.py` | Rigid-atom force field visualization | 861 | Test |

**Technology Stack:** VisPy + PyQt5 + GPU acceleration

**Overlap Analysis:**
- All use `VispyUtils.AtomScene` for 3D rendering
- All use `BaseGUI` for PyQt5 widgets
- Similar camera control patterns
- **Consolidation opportunity:** Extract common VisPy+PyQt5 application template

### 2. PyQt5 + OpenGL (Traditional 3D)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `pyBall/GUI/MolViewer.py` | Molecular viewer with trajectory support | 452 | **DUPLICATE** |
| `pyBall/GUI/MolBrowser.py` | Molecular browser (near-identical to MolViewer) | 520 | **DUPLICATE** |
| `pyBall/GUI/MolBrowser_refactored.py` | Refactored MolBrowser attempt | 319 | Experimental |
| `pyBall/GUI/GLGUI.py` | OpenGL widget base class (BaseGLWidget) | 845 | Shared |
| `pyBall/GUI/BaseGUI.py` | PyQt5 widget utilities (polymorphic helpers) | 229 | Shared |

**Technology Stack:** PyQt5 + PyOpenGL + custom OpenGL shaders

**Critical Issue - Code Duplication:**
- `MolViewer.py` and `MolBrowser.py` are **95% identical**
- Both use `BaseGLWidget`, `FrameData`, same bond computation
- Same import structure, same class hierarchy
- **Recommendation:** Delete one, keep the other as single canonical viewer

### 3. PyQt5 + Matplotlib (2D/Plotting)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `pyBall/GUI/MoleculeEditor2D.py` | 2D planar molecule editor (deprecated) | 1602 | **Deprecated** |
| `pyBall/GUI/FitREQInteractiveGUI.py` | Interactive parameter fitting GUI | 1300 | Active |
| `pyBall/GUI/FitREQUtil.py` | Matplotlib blitting utilities | 98 | Shared |

**Technology Stack:** PyQt5 + Matplotlib + VisPy (for 3D preview)

**Note:** `MoleculeEditor2D.py` documented as "deprecated; superseded by KekuleExplorerGUI"

### 4. GLCL (OpenGL + OpenCL Hybrid)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `pyBall/GLCL/GLCLBrowser.py` | Scientific simulation framework (v1) | 908 | Legacy |
| `pyBall/GLCL2/GLCLBrowser.py` | Scientific simulation framework (v2) | 443 | Newer |

**Technology Stack:** PyQt5 + OpenGL + PyOpenCL

**Overlap:** Two versions of same functionality with similar structure

---

## C++ GUI Applications

### SDL2 + OpenGL Applications

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `cpp/common_SDL/SDL2OGL/MolGUI.h` | **Monolithic molecular GUI** | 2922 | **MASSIVE** |
| `cpp/common/browser.h` | File browser utility | 99 | Shared |
| `cpp/common_SDL/SDL2OGL/GUI.h` | General GUI framework (panels, widgets) | ~600 | Shared |
| `cpp/apps/MolecularEditor/MolecularEditor_main.cpp` | Molecular editor app | ~400 | Active |
| `cpp/apps_OCL/MolecularEditorOCL/MolecularEditorOCL_main.cpp` | OpenCL molecular editor | ~400 | Active |

**Technology Stack:** SDL2 + OpenGL + custom GUI framework

**Critical Issue - Monolithic Design:**
- `MolGUI.h` is **2922 lines** - violates single responsibility
- Combines: rendering, editing, console, spline GUI, marching cubes, raytracing
- Should be split into modular components

**GUI Framework:**
- Custom `GUI.h` with panels: `GUIPanel`, `MultiPanel`, `CheckBoxList`, `DropDownList`, `TreeView`, `TableView`
- Similar to PyQt5 but C++ implementation
- **No reuse with Python GUIs**

---

## JavaScript/Web Applications

### WebGL (Legacy)

| Directory | Files | Purpose | Status |
|-----------|-------|---------|--------|
| `web/molgui_web/` | 16 JS files | WebGL molecular editor | **Legacy** |

**Key Files:**
- `EditableMolecule.js` - Molecular topology (1057 lines)
- `GUI.js` - Main GUI class (1133 lines)
- `MMParams.js` - Parameter parser
- `MoleculeRenderer.js` - WebGL rendering
- `ProjectiveDynamics.js` - Physics solver

### WebGPU (Modern)

| Directory | Files | Purpose | Status |
|-----------|-------|---------|--------|
| `web/molgui_webgpu/` | 26 JS files | WebGPU molecular editor + physics | **Active** |

**Key Files:**
- `EditableMolecule.js` - Molecular topology (1188 lines) - **DUPLICATE**
- `GUI.js` - Main GUI class (1357 lines) - **DUPLICATE**
- `MMParams.js` - Parameter parser - **DUPLICATE**
- `Draw3D_webgpu.js` - WebGPU rendering
- `XPDBTopology.js` - XPBD physics
- `MMFFLTopology.js` - MMFF topology

**Critical Issue - Code Duplication:**
- `molgui_web/` and `molgui_webgpu/` have **near-identical files**:
  - `EditableMolecule.js` (1057 vs 1188 lines) - same VSEPR, orthonormal basis code
  - `GUI.js` (1133 vs 1357 lines) - same sequence parsing, file I/O
  - `MMParams.js` - same parameter loading
- **Recommendation:** Consolidate to single WebGPU implementation, deprecate WebGL

### Shared JavaScript Utilities

| Directory | Purpose | Status |
|-----------|---------|--------|
| `web/common_js/` | Shared math, spatial partitioning, selection | Shared |

**Files:** `Vec3.js`, `Mat3.js`, `Buckets.js`, `BucketGrid3D.js`, `Selection.js`, `MeshBuilder.js`

**Good:** Already consolidated across web implementations

---

## Consolidation Opportunities

### High Priority (Critical Duplication)

1. **Python: MolViewer vs MolBrowser**
   - Files: `pyBall/GUI/MolViewer.py` vs `pyBall/GUI/MolBrowser.py`
   - Issue: 95% code duplication
   - Action: Delete one, keep single canonical viewer
   - Savings: ~500 lines

2. **JavaScript: WebGL vs WebGPU**
   - Directories: `web/molgui_web/` vs `web/molgui_webgpu/`
   - Issue: Duplicate `EditableMolecule.js`, `GUI.js`, `MMParams.js`
   - Action: Deprecate WebGL, consolidate to WebGPU
   - Savings: ~3000 lines

3. **C++: MolGUI.h Monolith**
   - File: `cpp/common_SDL/SDL2OGL/MolGUI.h` (2922 lines)
   - Issue: Violates single responsibility, combines rendering+editing+console+splines+marching cubes
   - Action: Split into modular components
   - Impact: Major refactoring, but necessary for maintainability

### Medium Priority (Architecture Cleanup)

4. **Python: VisPy Application Template**
   - Files: `ExplorerVisPy.py`, `KekuleExplorerGUI.py`, `SequencePlacerVisPy.py`
   - Issue: Similar structure, repeated boilerplate
   - Action: Extract common `VisPyApp` base class
   - Benefit: Easier to create new VisPy applications

5. **Python: GLCL Version Consolidation**
   - Files: `pyBall/GLCL/GLCLBrowser.py` vs `pyBall/GLCL2/GLCLBrowser.py`
   - Issue: Two versions of same functionality
   - Action: Merge to single version, delete legacy
   - Savings: ~900 lines

6. **Deprecated Code Removal**
   - File: `pyBall/GUI/MoleculeEditor2D.py` (1602 lines)
   - Status: Documented as "deprecated; superseded by KekuleExplorerGUI"
   - Action: Delete or move to archive
   - Savings: 1600 lines

### Low Priority (Cross-Language Consistency)

7. **Parameter File Loading**
   - Python: `MMFFparams.h` (C++), `MMParams.js` (JS), manual parsing (Python)
   - Issue: Three different implementations for same `.dat` files
   - Action: Consider single source of truth or well-documented parity

8. **GUI Framework Parity**
   - C++: Custom `GUI.h` with panels
   - Python: `BaseGUI.py` with polymorphic helpers
   - JS: Custom GUI classes
   - Issue: No cross-language consistency
   - Action: Document patterns, accept language-specific implementations

---

## Technology Stack Summary

| Technology | Language | Files | Purpose | Status |
|------------|----------|-------|---------|--------|
| **VisPy** | Python | 5 | GPU-accelerated 3D visualization | Active |
| **PyQt5 + PyOpenGL** | Python | 5 | Traditional 3D OpenGL | Active |
| **PyQt5 + Matplotlib** | Python | 3 | 2D editing, plotting | Active |
| **SDL2 + OpenGL** | C++ | 10+ | Desktop applications | Active |
| **WebGL** | JavaScript | 16 | Browser-based (legacy) | **Legacy** |
| **WebGPU** | JavaScript | 26 | Browser-based (modern) | Active |

**Recommendation:** Reduce to 3 primary stacks:
1. Python: VisPy (for interactive 3D) + Matplotlib (for 2D/plotting)
2. C++: SDL2 + OpenGL (for desktop apps)
3. JavaScript: WebGPU (for browser)

---

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)
1. Delete `pyBall/GUI/MolBrowser.py` (duplicate of MolViewer)
2. Delete `pyBall/GUI/MoleculeEditor2D.py` (deprecated)
3. Delete `pyBall/GLCL/GLCLBrowser.py` (legacy GLCL)
4. Mark `web/molgui_web/` as deprecated in documentation

### Phase 2: Web Consolidation (3-5 days)
1. Audit differences between `molgui_web/` and `molgui_webgpu/`
2. Merge unique features from WebGL to WebGPU
3. Delete `web/molgui_web/`
4. Update all references to use WebGPU

### Phase 3: Python Refactoring (5-7 days)
1. Extract common `VisPyApp` base class from existing VisPy apps
2. Refactor `ExplorerVisPy.py`, `KekuleExplorerGUI.py`, `SequencePlacerVisPy.py` to use base class
3. Consolidate `VispyUtils.py` (already good, minor cleanup)

### Phase 4: C++ Refactoring (10+ days)
1. Split `MolGUI.h` into modular components:
   - `MolGUI_renderer.h` (rendering)
   - `MolGUI_editor.h` (editing)
   - `MolGUI_console.h` (console)
   - `MolGUI_splines.h` (spline GUI)
2. Update all includes
3. Test all C++ apps

### Phase 5: Documentation (2-3 days)
1. Update `CODEMAP.md` with consolidated architecture
2. Add GUI development guide
3. Document technology choices (why VisPy vs OpenGL vs WebGPU)

---

## File Inventory

### Python GUI Files (22 files)

**VisPy-based:**
- `pyBall/ExplorerVisPy.py` (1104 lines)
- `pyBall/KekuleExplorerGUI.py` (1413 lines)
- `pyBall/SequencePlacerVisPy.py` (376 lines)
- `pyBall/VispyUtils.py` (1579 lines)
- `pyBall/RigidAtomFF/RRsp3/test_RRsp3_vispy.py` (861 lines)

**PyQt5 + OpenGL:**
- `pyBall/GUI/MolViewer.py` (452 lines)
- `pyBall/GUI/MolBrowser.py` (520 lines) - **DUPLICATE**
- `pyBall/GUI/MolBrowser_refactored.py` (319 lines)
- `pyBall/GUI/GLGUI.py` (845 lines)
- `pyBall/GUI/BaseGUI.py` (229 lines)

**PyQt5 + Matplotlib:**
- `pyBall/GUI/MoleculeEditor2D.py` (1602 lines) - **DEPRECATED**
- `pyBall/GUI/FitREQInteractiveGUI.py` (1300 lines)
- `pyBall/GUI/FitREQUtil.py` (98 lines)

**GLCL:**
- `pyBall/GLCL/GLCLBrowser.py` (908 lines) - **LEGACY**
- `pyBall/GLCL2/GLCLBrowser.py` (443 lines)

**Other:**
- `pyBall/MoleculeViewer.py` (separate from GUI/MolViewer.py)
- `pyBall/MolecularPlacerVisPy.py`
- `pyBall/SequencePlacer.py` (has GUI class)

### C++ GUI Files (20+ files)

**Core:**
- `cpp/common_SDL/SDL2OGL/MolGUI.h` (2922 lines) - **MONOLITHIC**
- `cpp/common/browser.h` (99 lines)
- `cpp/common_SDL/browser_sdl.h`
- `cpp/common_SDL/SDL2OGL/GUI.h` (~600 lines)
- `cpp/common_SDL/SDL2OGL/GUI.cpp`

**Applications:**
- `cpp/apps/MolecularEditor/MolecularEditor_main.cpp`
- `cpp/apps/MolecularEditor/ConfSearch.cpp`
- `cpp/apps_OCL/MolecularEditorOCL/MolecularEditorOCL_main.cpp`
- `cpp/apps_OCL/MolecularEditorOCL/MolecularEditorOCL_scanner.cpp`
- `cpp/apps_CUDA/MolGUIapp_cuda.cpp`

**Utilities:**
- `cpp/common_SDL/SDL2OGL/Draw3D.h`
- `cpp/common_SDL/SDL2OGL/Draw3D_Molecular.h`
- `cpp/common_SDL/SDL2OGL/MolecularDraw.h`
- `cpp/common_SDL/SDL2OGL/MarchingCubes.h`
- `cpp/common_SDL/SDL2OGL/EditorGizmo.h`
- `cpp/common_SDL/SDL2OGL/SplineGUI.h`
- `cpp/common_SDL/SDL2OGL/Console.h`

### JavaScript/Web Files (42 files)

**WebGPU (modern):**
- `web/molgui_webgpu/EditableMolecule.js` (1188 lines)
- `web/molgui_webgpu/GUI.js` (1357 lines)
- `web/molgui_webgpu/MMParams.js`
- `web/molgui_webgpu/Draw3D_webgpu.js`
- `web/molgui_webgpu/BuildersGUI.js`
- `web/molgui_webgpu/CrystalUtils.js`
- `web/molgui_webgpu/Editor.js`
- `web/molgui_webgpu/MoleculeIO.js`
- `web/molgui_webgpu/MoleculeRenderer.js`
- `web/molgui_webgpu/MoleculeSelection.js`
- `web/molgui_webgpu/MoleculeUtils.js`
- `web/molgui_webgpu/MMFFLTopology.js`
- `web/molgui_webgpu/XPDBTopology.js`
- `web/molgui_webgpu/XPDB_CPU.js`
- `web/molgui_webgpu/XPDB_WebGPU.js`
- `web/molgui_webgpu/ScriptRunner.js`
- `web/molgui_webgpu/ShortcutManager.js`
- `web/molgui_webgpu/Nanocrystals.js`
- `web/molgui_webgpu/LinearizedTopologyNpz.js`
- `web/molgui_webgpu/LinearizedTopologyViewer.js`
- `web/molgui_webgpu/MeshRenderer_webgpu.js`
- `web/molgui_webgpu/RawWebGPUAtomsRenderer.js`
- `web/molgui_webgpu/XPDBTopologyRenderer.js`
- `web/molgui_webgpu/headless_webgpu.js`
- `web/molgui_webgpu/debugBuffers.js`
- `web/molgui_webgpu/main.js`

**WebGL (legacy):**
- `web/molgui_web/js/EditableMolecule.js` (1057 lines) - **DUPLICATE**
- `web/molgui_web/js/GUI.js` (1133 lines) - **DUPLICATE**
- `web/molgui_web/js/MMParams.js` - **DUPLICATE**
- `web/molgui_web/js/BuildersGUI.js`
- `web/molgui_web/js/CrystalUtils.js`
- `web/molgui_web/js/Editor.js`
- `web/molgui_web/js/MoleculeIO.js`
- `web/molgui_web/js/MoleculeRenderer.js`
- `web/molgui_web/js/MoleculeSelection.js`
- `web/molgui_web/js/MoleculeUtils.js`
- `web/molgui_web/js/ProjectiveDynamics.js`
- `web/molgui_web/js/ScriptRunner.js`
- `web/molgui_web/js/ShortcutManager.js`
- `web/molgui_web/js/main.js`

**Shared:**
- `web/common_js/Vec3.js`
- `web/common_js/Mat3.js`
- `web/common_js/Buckets.js`
- `web/common_js/BucketGrid3D.js`
- `web/common_js/Buckets_SoA.js`
- `web/common_js/BucketAABBs.js`
- `web/common_js/Selection.js`
- `web/common_js/MeshBuilder.js`
- `web/common_js/MeshRenderer.js`
- `web/common_js/MeshesUV.js`
- `web/common_js/GUIutils.js`
- `web/common_js/Graph.js`
- `web/common_js/GridIndexer.js`
- `web/common_js/Logger.js`
- `web/common_js/MolIO.js`
- `web/common_js/Nonbonded.js`
- `web/common_js/SDfuncs.js`
- `web/common_js/exportFF.js`
- `web/common_js/nanocrystalSvg.js`
- `web/common_js/npzIO.js`

---

## Conclusion

FireCore has **severe GUI fragmentation** with:
- **22+ Python GUI files** across 4 technology stacks
- **20+ C++ GUI files** with one 2922-line monolith
- **42 JavaScript files** with duplicate WebGL/WebGPU implementations

**Estimated cleanup potential:** 8000+ lines of duplicate/legacy code

**Primary recommendation:** Consolidate to:
1. Python: VisPy (3D) + Matplotlib (2D) - single viewer, single editor
2. C++: Modularized MolGUI components
3. JavaScript: WebGPU only (deprecate WebGL)

This audit provides the foundation for systematic consolidation.

---

# Detailed Feature Analysis: Visualization and Editor Operations

## Part 1: Visualization/Rendering Features

### Python VisPy+PyQt5 GUIs

#### KekuleExplorerGUI.py
**Rendering Options:**
- 3D molecular visualization with VisPy scene
- Atom spheres with element-based colors
- Bond lines with color by length option
- Debug view mode
- Multiple label modes:
  - Element+Index
  - Atomic Type
  - Pi Orbitals
  - Z-Height
  - Charge
  - Bond Lengths
- Bond coloring options
- Hex grid drawing toggle

#### ExplorerVisPy.py
**Rendering Options:**
- 3D VisPy canvas with orthographic camera
- Molecule and substrate rendering
- Surface map visualization
- Matplotlib 2D plot for scan data
- Interactive relaxation visualization
- Physics toggles (LJ, Coulomb, H-bond, Morse)

#### MoleculeEditor2D.py
**Rendering Options:**
- 2D planar Matplotlib canvas
- Atom circles with element colors
- Bond lines
- Grid snapping visualization (triangular, square)
- Selection highlighting
- Rotation pivot visualization

### Python PyQt5+OpenGL GUIs

#### MolViewer.py
**Rendering Options:**
- OpenGL-based 3D rendering
- Atom spheres with colors
- Bond lines
- Frame/trajectory support
- Opacity control
- Multiple shader render modes:
  - raytrace
  - max_vol
- GL frame saving to images
- Atom labels

#### MolBrowser.py / MolBrowser_refactored.py
**Rendering Options:**
- OpenGL-based 3D rendering
- Trajectory browsing
- Frame data management (atoms, electrons, bonds)
- Instanced rendering for performance

### C++ SDL2+OpenGL GUI

#### MolGUI.h
**Rendering Options:**
- OpenGL rendering via SDL2
- Atom spheres (mm_Rsc, mm_Rsub parameters)
- Bond rendering
- Multiple visualization toggles:
  - bViewBuilder
  - bViewAxis
  - bViewCell
  - bViewMolCharges
  - bViewHBondCharges
  - bViewAtomLabels
  - bViewAtomTypes
  - bViewColorFrag
  - bViewBondLabels
  - bViewAtomSpheres
  - bViewAtomForces
  - bViewBondLenghts
  - bViewBonds
  - bViewPis
  - bViewSubstrate
  - bViewGroupBoxes
- Isosurface rendering (isoSurfRenderType)
- AFM scan visualization
- ESP rendering
- Orbital rendering
- Density rendering
- Non-bond particle rendering
- Dipole map visualization
- Hex grid drawing
- Trajectory rendering

### JavaScript WebGPU/WebGL GUIs

#### molgui_webgpu/EditableMolecule.js
**Rendering Options:**
- WebGPU-based 3D rendering
- Atom and bond rendering
- Fragment bounds visualization
- Selection highlighting

#### molgui_web/EditableMolecule.js
**Rendering Options:**
- WebGL-based 3D rendering
- Similar to WebGPU version (legacy)

---

## Part 2: Editor Operations

### Python VisPy+PyQt5 GUIs

#### KekuleExplorerGUI.py
**Editor Modes:**
- Hex1 mode - Hexagonal ring drawing (variant 1)
- Hex2 mode - Hexagonal ring drawing (variant 2)
- Atom mode - Add individual atoms
- Bond mode - Add bonds between atoms
- pi mode - Pi orbital manipulation
- Select mode - Selection operations

**Atom Operations:**
- Add atoms by clicking
- Change atom type via selection
- Auto hydrogen capping
- Auto bond recalculation
- Delete selected atoms

**Bond Operations:**
- Add bonds by clicking
- Bond order cycling
- Auto bond recalculation

**Selection Operations:**
- Single atom selection
- Multiple selection
- Drag to move selected atoms
- Copy/paste selection

**Specialized Operations:**
- Ribbon generation (single ribbon)
- Ribbon generation (two-ribbon)
- Grid snapping
- Pick radius configuration
- Pi orbital visualization and manipulation
- Export to XYZ

**Alignment/Rotation:**
- Rotation pivots for PCA alignment
- Molecule pose controls (translation, rotation)

#### MoleculeEditor2D.py
**Editor Modes:**
- Draw mode - Add atoms
- Move mode - Move atoms
- Bond mode - Add bonds
- Rotate mode - Rotate atoms

**Atom Operations:**
- Add atoms by clicking
- Delete atoms
- Change element type
- Move atoms

**Bond Operations:**
- Add bonds
- Delete bonds
- Bond order cycling

**Selection Operations:**
- Single selection
- Rectangular selection
- Selection expansion (grow)
- Selection shrinkage
- Connected selection

**Transformations:**
- Translate atoms
- Rotate atoms with pivot
- Grid snapping (triangular, square)

**File I/O:**
- Load XYZ, MOL, MOL2
- Save XYZ, MOL, MOL2

### C++ SDL2+OpenGL GUI

#### MolGUI.h
**Selection Operations:**
- selectShorterSegment - Ray-based segment selection
- selectRect - Rectangular selection
- Selection modes: Atom, Bond, Angle, Torsion, Fragment
- GUI modes: base, edit, scan

**Editor Features:**
- EditorGizmo integration for transform operations
- SimplexRuler for organic molecule painting
- Builder pattern for molecular construction
- Non-bond particle manipulation
- Hex drawing mode

**Transformations:**
- Rotation center, axis, step parameters
- Camera controls
- Gizmo-based translation/rotation
- Auto-pivot at COG of selection

**Specialized Operations:**
- Non-bond particle relaxation
- Lattice scanning
- Bond length coloring
- AFM scan operations
- Dipole map visualization

### JavaScript WebGPU/WebGL GUIs

#### molgui_webgpu/EditableMolecule.js
**Atom Operations:**
- addAtom(x, y, z, Z) - Add atom with position and element
- addAtomZ(Z, x, y, z) - Alternative signature
- setAtomTypeByName(id, typeName, mmParams) - Change atom type
- setAtomPosById(id, x, y, z) - Set atom position
- removeAtomById(id) - Remove atom
- removeAtomByIndex(i) - Remove atom by index
- deleteSelectedAtoms() - Delete all selected atoms
- addAtomsFromArrays(pos3, types1) - Batch add from arrays

**Bond Operations:**
- addBond(aId, bId, order, type) - Add bond
- removeBondById(id) - Remove bond
- removeBondByIndex(ib) - Remove bond by index
- recalculateBonds(mmParams, opts) - Auto-recalculate bonds
- recalculateBondsBucketNeighbors(mmParams, buckets, opts) - Bucket-based bond recalculation
- recalculateBondsBucketAllPairsAABB(mmParams, buckets, opts) - AABB-optimized bond recalculation

**Selection Operations:**
- selectAtom(id, mode) - Select atom (replace/add/subtract)
- select(idOrIndex, mode) - Select by id or index
- selectAll() - Select all atoms
- clearSelection() - Clear selection
- Selection Set for tracking

**Transformations:**
- translateAtoms(ids, vec) - Translate atoms by vector
- rotateAtoms(ids, axis, deg, center) - Rotate atoms around axis

**Specialized Operations:**
- addCappingAtoms(mmParams, cap, opts) - Add H caps with VSEPR geometry
  - VSEPR direction calculation (linear, planar, tetrahedral)
  - Outward bias option
  - Clash resolution (resolveCapHClashes)
  - Optional H-H bond addition (addCapHHBonds)
- addExplicitEPairs(mmParams, opts) - Add explicit electron pairs
- replicate(nrep, lvec) - Replicate molecule in lattice
- updateNeighborList() - Rebuild atom-bond adjacency
- exportToMoleculeSystem(ms) - Export to MoleculeSystem

**Topology Management:**
- lockTopology() - Lock topology for batch operations
- unlockTopology() - Unlock topology
- assertLocked() - Assert topology is locked
- Dirty flags: dirtyTopo, dirtyGeom, dirtyFrags, dirtyExport
- Topology version tracking

**Data Structures:**
- Atom class: id, i, Z, atype, charge, flags, pos, bonds, frag, fragSlot
- Bond class: id, i, aId, bId, a, b, order, type, isBridge, isRingEdge
- Bounds class: min, max, center, radius, intersection tests
- Fragment class: id, isClosed, atomIds, bondIds, bounds

#### molgui_web/EditableMolecule.js
**Editor Operations:**
- Similar to WebGPU version (legacy WebGL implementation)
- Same core operations for atoms, bonds, selection, transformations
- VSEPR-based capping
- Bond recalculation

---

## Part 3: Feature Richness Summary

### Visualization Feature Matrix

| Feature | KekuleExplorer | ExplorerVisPy | MoleculeEditor2D | MolViewer | MolGUI (C++) | molgui_webgpu |
|---------|---------------|---------------|-----------------|-----------|--------------|---------------|
| 3D Rendering | ✓ | ✓ | ✗ (2D) | ✓ | ✓ | ✓ |
| Atom Spheres | ✓ | ✓ | ✓ (2D circles) | ✓ | ✓ | ✓ |
| Bond Lines | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Element Colors | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Atom Labels | ✓ (multiple modes) | Limited | ✓ | ✓ | ✓ | ? |
| Bond Labels | ✓ | ✗ | ✗ | ✗ | ✓ | ? |
| Force Vectors | ✗ | ✗ | ✗ | ✗ | ✓ | ? |
| Trajectory Support | ✗ | ✗ | ✗ | ✓ | ✓ | ? |
| Surface/Isosurface | ✗ | ✓ (surface map) | ✗ | ✗ | ✓ | ? |
| ESP Visualization | ✗ | ✗ | ✗ | ✗ | ✓ | ? |
| AFM Scan | ✗ | ✓ | ✗ | ✗ | ✓ | ? |
| Pi Orbital Vis | ✓ | ✗ | ✗ | ✗ | ✓ | ? |
| Grid Snapping | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Shader Modes | ✗ | ✗ | ✗ | ✓ | ? | ? |
| Opacity Control | ✗ | ✗ | ✗ | ✓ | ? | ? |
| **Crystal Cell Box** | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| **Lattice Vector Vis** | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| **Slab/Plane Preview** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **Nanocrystal Vis** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

### Editor Operation Matrix

| Operation | KekuleExplorer | MoleculeEditor2D | MolGUI (C++) | molgui_webgpu |
|-----------|---------------|-----------------|--------------|---------------|
| Add Atom | ✓ | ✓ | ✓ (builder) | ✓ |
| Delete Atom | ✓ | ✓ | ✓ | ✓ |
| Move Atom | ✓ (drag) | ✓ | ✓ (gizmo) | ✓ (translate) |
| Change Element | ✓ | ✓ | ✓ | ✓ |
| Add Bond | ✓ | ✓ | ✓ | ✓ |
| Delete Bond | ✓ | ✓ | ✓ | ✓ |
| Bond Order Cycle | ✓ | ✓ | ? | ? |
| Select Single | ✓ | ✓ | ✓ | ✓ |
| Select Multiple | ✓ | ✓ | ✓ | ✓ |
| Rectangular Select | ✗ | ✓ | ✓ | ? |
| Connected Select | ✗ | ✓ | ? | ? |
| Translate | ✓ (drag) | ✓ | ✓ (gizmo) | ✓ |
| Rotate | ✓ (pivot) | ✓ (pivot) | ✓ (gizmo) | ✓ |
| Grid Snapping | ✓ | ✓ | ✗ | ✗ |
| H Capping | ✓ (auto) | ✗ | ? | ✓ (VSEPR) |
| Ribbon Gen | ✓ | ✗ | ✗ | ✗ |
| Pi Orbital Edit | ✓ | ✗ | ✓ (view) | ✗ |
| PCA Alignment | ✓ (pivot) | ✗ | ✗ | ✗ |
| Bond Recalc | ✓ (auto) | ✗ | ? | ✓ |
| Replicate | ✗ | ✗ | ? | ✓ |
| E-Pair Add | ✗ | ✗ | ✗ | ✓ |
| **CIF Import** | ✗ | ✗ | ✗ | ✓ |
| **Unit Cell Editor** | ✗ | ✗ | ✗ | ✓ |
| **Symmetry Ops** | ✗ | ✗ | ✗ | ✓ |
| **Slab Cutting (HKL)** | ✗ | ✗ | ✗ | ✓ |
| **Plane Cutting** | ✗ | ✗ | ✗ | ✓ |
| **Crystal Presets** | ✗ | ✗ | ✗ | ✓ |
| **Bond Building (cells)** | ✗ | ✗ | ✓ (single cell) | ✓ |
| **Site Deduplication** | ✗ | ✗ | ✗ | ✓ |

---

## Part 4: Consolidation Recommendations

### High Priority Consolidation

#### 1. Shared Editor Operations Module
**Target:** `pyBall/GUI/SharedEditor.py`

Extract common editor operations:
- Atom add/delete/move/change type
- Bond add/delete/order cycling
- Selection management (single, multiple, rectangular, connected)
- Transform operations (translate, rotate with pivot)
- VSEPR-based capping (from JavaScript, port to Python)
- Bond recalculation algorithms

#### 2. Shared Visualization Module
**Target:** `pyBall/GUI/SharedRenderer.py`

Unify rendering:
- Atom sphere rendering with element colors
- Bond line/cylinder rendering
- Label placement and styling
- Color mapping utilities (element, charge, fragment, bond-length)
- Force vector rendering
- Selection highlighting

#### 3. Selection System
**Target:** `pyBall/GUI/SelectionManager.py`

Unified selection API:
- Selection modes (atom, bond, angle, torsion, fragment)
- Selection operations (add, subtract, replace, clear, select all)
- Selection queries (connected, expand, shrink)
- Selection persistence (save/load)

### Medium Priority Consolidation

#### 4. VSEPR Geometry Module
**Target:** `pyBall/geometry/VSEPR.py`

Port VSEPR direction calculation from JavaScript to Python:
- Linear (2 domains)
- Planar (3 domains)
- Tetrahedral (4 domains)
- Outward bias for capping
- Clash resolution

**Source:** `EditableMolecule.js:missingDirsVSEPR()` and `addCappingAtoms()` (lines ~400-500)

#### 5. Bond Recalculation Module
**Target:** `pyBall/topology/BondRecalculation.py`

Consolidate bond recalculation algorithms:
- Distance-based cutoff
- Bucket-optimized neighbor search
- AABB-based acceleration
- Bond order inference

**Sources:** `EditableMolecule.js:recalculateBonds()`, `recalculateBondsBucketNeighbors()`, `recalculateBondsBucketAllPairsAABB()`

#### 6. Crystal Builder Module (CRITICAL — blocks unified GUI)
**Target:** `pyBall/crystal/CrystalBuilder.py`

Port from `web/molgui_webgpu/CrystalUtils.js` (1188 lines) to Python:
- Lattice vectors from (a, b, c, α, β, γ)
- Fractional ↔ Cartesian coordinate conversion
- CIF parsing (tags + loops)
- Symmetry operation parsing and application
- Site deduplication (grid-bucket algorithm)
- Bulk cell replication with optional bond building
- Slab cutting (HKL normal via reciprocal lattice)
- Plane cutting (multiple arbitrary planes)
- Reciprocal lattice computation
- Preset crystal structures (NaCl, CaF2, diamond, etc.)

**See:** [molecular_topology_editors.md](molecular_topology_editors.md) § "Crystal Building — Cross-Language Gap Analysis" for detailed function inventory

#### 7. File I/O Module
**Target:** `pyBall/io/MoleculeIO.py`

Consolidate file I/O:
- XYZ, MOL, MOL2 loading/saving
- CIF loading (via CrystalBuilder)
- Trajectory loading
- Export utilities

### Low Priority / Keep Separate

#### Web vs Desktop
- Keep web (WebGPU/WebGL) separate due to deployment constraints
- Share data structures and algorithms via common JavaScript/Python modules

#### C++ vs Python
- Keep C++ GUI separate for performance-critical applications
- Share algorithms via common mathematical utilities

#### Specialized GUIs
- ExplorerVisPy - Keep as specialized physics explorer
- MolViewer - Keep as trajectory viewer
- KekuleExplorerGUI - **Evolve into unified GUI** (see VisPy plan below)

### Deprecation Plan

#### Immediate Deprecation
- **MoleculeEditor2D.py** - Replaced by KekuleExplorerGUI, marked as deprecated

#### Future Deprecation
- **molgui_web** - Replace with molgui_webgpu once feature parity achieved
- **MolBrowser.py** - Replace with MolBrowser_refactored.py once stable

### Migration Path

1. **Phase 1:** Create shared modules (editor, visualization, selection)
2. **Phase 2:** Port `CrystalUtils.js` → `CrystalBuilder.py` (CRITICAL)
3. **Phase 3:** Port VSEPR and bond recalculation from JavaScript to Python
4. **Phase 4:** Refactor KekuleExplorerGUI to use shared modules + CrystalBuilder
5. **Phase 5:** Update other Python GUIs to use shared modules
6. **Phase 6:** Deprecate legacy implementations

---

## Part 5: Unified VisPy GUI Architecture

### Goal

Consolidate all molecular GUI functionality into a single Python VisPy-based application, replacing the fragmented ecosystem (KekuleExplorer, MoleculeEditor2D, molgui_webgpu crystal features, MolGUI C++ editor).

### Proposed Module Structure

```
pyBall/GUI/
├── MolecularGUI.py           # Main application (VisPy + PyQt5)
├── SharedEditor.py           # Atom/bond/selection operations
├── SharedRenderer.py         # VisPy rendering (spheres, bonds, labels, cell box)
├── SelectionManager.py       # Unified selection API
├── CrystalGUI.py             # Crystal builder panel (CIF, lattice, slab, presets)
└── panels/
    ├── EditorPanel.py        # Atom/bond editing controls
    ├── CrystalPanel.py       # Unit cell editor, symmetry, replication
    ├── ViewPanel.py          # Rendering options, labels, colors
    └── PhysicsPanel.py       # FF parameters, MD controls (future)

pyBall/crystal/
└── CrystalBuilder.py         # Ported from CrystalUtils.js

pyBall/geometry/
└── VSEPR.py                  # Ported from EditableMolecule.js
```

### Feature Target for Unified GUI

| Feature Category | Source | Priority |
|-----------------|--------|----------|
| 3D atom/bond rendering | KekuleExplorer | High |
| Element colors + labels | KekuleExplorer | High |
| Atom add/delete/move | KekuleExplorer + EditableMolecule.js | High |
| Bond add/delete/order | KekuleExplorer + EditableMolecule.js | High |
| Selection (single, rect, connected) | All GUIs | High |
| Transform (translate, rotate) | KekuleExplorer | High |
| **CIF import** | CrystalUtils.js | **Critical** |
| **Unit cell editor** | BuildersGUI.js | **Critical** |
| **Bulk crystal generation** | CrystalUtils.js | **Critical** |
| **Slab cutting (HKL)** | CrystalUtils.js | High |
| **Plane cutting** | CrystalUtils.js | Medium |
| **Crystal presets** | BuildersGUI.js | Medium |
| **Symmetry operations** | CrystalUtils.js | High |
| **Site deduplication** | CrystalUtils.js | High |
| **Bonds across cells** | CrystalUtils.js | High |
| VSEPR H-capping | EditableMolecule.js | High |
| Bond recalculation | EditableMolecule.js | Medium |
| Grid snapping (hex) | KekuleBackend | Medium |
| Ribbon generation | KekuleBackend | Low |
| Pi orbital visualization | KekuleExplorer | Medium |
| Force vector rendering | MolGUI (C++) | Low |
| Trajectory support | MolViewer | Low |
| AFM scan visualization | ExplorerVisPy | Low (future) |

### Key Design Principles

1. **AtomicGraph as canonical editing representation** — object-based graph for interactive editing
2. **NumPy arrays for crystal operations** — vectorized frac/cart conversions, batch atom generation
3. **Fail loudly** — all invalid inputs raise exceptions (matching JS behavior)
4. **Topology locking** — port `lockTopology()`/`unlockTopology()` from EditableMolecule.js for batch operations
5. **Lazy index caching** — port dirty flag system (dirtyTopo, dirtyGeom, dirtyFrags) from EditableMolecule.js
6. **Stable atom IDs** — atoms maintain integer IDs across operations (not just array indices)

### What NOT to Port

- **WebGPU/WebGL rendering** — keep in JS for browser deployment
- **C++ SDL2 GUI** — keep for performance-critical native applications
- **MoleculeEditor2D** — deprecated, 2D-only, no unique features
- **MolBrowser** — trajectory viewing is separate concern

---

## See Also

- [Topical Audit Index](topical_audit.md) — priority ranking, dependency graph, missing topics
- [Molecular Topology](molecular_topology.md) — graph representations, bond finding, rings, bridges
- [Molecular Topology Editors](molecular_topology_editors.md) — crystal building gap analysis, consolidation roadmap
- [Molecular Topology Types](molecular_topology_types.md) — atom type assignment, parameter files
- [Forcefields Web Implementation](forcefields_web_implementation.md) — WebGPU/WebGL shader implementations
- [Intramolecular Forcefields](intramolecular_forcefields.md) — FF evaluation used by GUI physics panels
