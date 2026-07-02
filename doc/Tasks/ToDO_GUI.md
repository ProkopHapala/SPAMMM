
- Allow rotate or flip the Hex-Grid by 
   - 90deg  (x,y flip)
   - general rotation
   - arbitrary shift?
   - Hex grid should be just guideline for drawing, tolology should not rely on it, atoms should be possible to place outside the grid

- Export molecule with tpopology e.g. to .mol / mol2, later .pdb (we perhaps do not have implemented that yet)
- Allow undo (Ctrl+Z) and redo (Ctrl+Y). How? 
   - Simplets is just keep in memroy few last topologies/geometries of the molecule

- We have bond-insert but we do not have creating bond between existing atoms?
- Removing atoms still does not work reliably

- Ctrl+C/Ctrl+Z direcly as .xyz or .mol

- Drawing pentagon, hepagon etc ?

- ASCII-art molecule builder, as well as smiles

- move to 3D viewwe

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

- Define fragments, store it into fragment library, figgre out format for subrituting attachmetnd from the fragment library 
  - easy when it is bonded by just one bond
  - there can be also gridging groups and vicinal groups - like attaching ataracene to whatever

