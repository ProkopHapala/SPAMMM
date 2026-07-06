# topology/

Editable molecular topology and structure generation. AtomicGraph is the SSOT — all other representations (rendering, force field export, PackedMolecule) derive from it.

- **AtomicGraph.py** — Object-graph molecular representation with stable Atom/Bond/Ring identities; deleting an atom doesn't renumber others (crucial for GUI editing)
- **KekuleBackend.py** — Backend logic for the Kekule Structure Explorer: atom/bond editing, ring insertion, passivation groups, hexagonal grid snapping, hybridization inference
- **KekulePure.py** — Pure-NumPy Kekule pi-bond-order optimizer (parabolic valence penalty + aromatic bond energy + gradient descent)
- **PackedMolecule.py** — Dense NumPy snapshot of AtomicGraph for undo history, clipboard, and .npz I/O (~21 bytes/atom)
- **FFparams.py** — Force field parameter parsing from .dat files (ElementTypes, AtomTypes, BondTypes, AngleTypes, DihedralTypes) for SPFF and UFF
- **HexGrid.py** — Hexagonal grid with offset/rotation/transpose transforms for hex drawing and atom snapping
- **heterocycle_generator.py** — Heterocycle generator from sparse rectangular-grid description of hexagonal lattice (E/D layers, dimer modes)
- **ascii_art_heterocycle.py** — ASCII art heterocycle builder (single-atom and dimer input formats, bonds inferred from zig-zag topology)
