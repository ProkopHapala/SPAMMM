
- Allow rotate or flip the Hex-Grid by 
   - 90deg  (x,y flip)
   - general rotation
   - arbitrary shift?
   - Hex grid should be just guideline for drawing, tolology should not rely on it, atoms should be possible to place outside the grid

- [x] Export molecule with topology e.g. to .mol / mol2, later .pdb (implemented: .xyz, .mol, .mol2)
- [x] Allow undo (Ctrl+Z) and redo (Ctrl+Y). How? 
   - Simplets is just keep in memroy few last topologies/geometries of the molecule
   - Implemented: UndoStack with PackedMolecule snapshots, Ctrl+Z. Redo not yet.

- [x] We have bond-insert but we do not have creating bond between existing atoms?
   - Implemented: Ctrl+LMB drag in Atom mode creates bond between two atoms
- [x] Removing atoms still does not work reliably
   - Fixed: ID-based pipeline (Atom._id), soft-delete, no index shifting

- [x] Ctrl+C/Ctrl+Z direcly as .xyz or .mol
   - Implemented: Ctrl+C copies PackedMolecule + puts .mol/.xyz text on Qt clipboard

- Drawing pentagon, hepagon etc ?

- [x] ASCII-art molecule builder — `AsciiArtExtension` + `ascii_art_heterocycle` (dedupe shared builder for CLI: `SPM_CLI_Headless.md` §A/D)
- [ ] SMILES → `AtomicGraph` — parser + CLI done (`spammm/topology/smiles.py`, `run_spm.py --smiles*` / `smiles-afm`); **GUI text box** still open (`SPM_CLI_Headless.md` §C, `ARCHITECTURE_ROADMAP` §9 / T07)

- [x] 3D view (ortho in `SPAMMM_GUI`: Enter toggles `b2Dview`; RMB empty = rotate; Ring mode ray pick) — see `doc/Tasks/GUI_Editor_3D_ViewMode.md`

- Molecular Browser with thumbnails (also with AFM images)

- Kekule Solver ?  - fast estimation of pi-orbitals structures?

- Forming hydrogen bonds when dragging ?


- Interactive relaxation drag constraint:
  When dragging an atom during interactive MD mode, the dragged atom should be
  properly constrained to the mouse ray (projection onto the camera plane).
  Currently the atom has "authority problems" — the MD integrator fights the
  mouse drag. Need to implement: during drag, pin the dragged atom to a target
  position that follows the mouse ray (updated each frame), with high K spring.
  The mouse ray → 3D position mapping should use the VisPy camera projection
  and intersect with the atom's current depth plane.
  Status: NOT YET IMPLEMENTED — future task.


- Export LAMMPS topology file (or GROMAS?)

- Function to attach selected chemical group anywhere 

- Define fragments, store into fragment library — see ARCHITECTURE_ROADMAP §11 for full design
  - **1-terminal groups** (–OH, –NH₂, –CH₃, ...): hover over atom → preview → click to attach
  - **2-terminal vicinal groups** (–CH=CH–, –N=N–, fused rings, ...):
    - Edge attachment: hover over bond A–B → group bridges across (like edge ring)
    - Corner attachment: hover over inner corner atom B (angle < 180°) → group spans A–C (like corner ring)
  - Group library: JSON-based, extensible, with terminal atoms + bond mode (substitute/add)
  - Interaction reuses Ring mode hover priority: bond → corner → atom

