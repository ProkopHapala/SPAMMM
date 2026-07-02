# Kekule Solver Visualization in SPAMMM GUI

## Overview

This document describes the implementation of Kekule pi-bond-order visualization integrated into the SPAMMM GUI. The feature allows users to draw molecules on a hexagonal grid, manually set per-atom pi-electron counts (n_pi), run a Kekule bond-order solver, and see the resulting localized/delocalized double bonds rendered directly in the 3D viewport.

## Components

### 1. ASCII Art Heterocycle Builder (`spammm/topology/ascii_art_heterocycle.py`)

The ASCII art system provides a text-based interface for specifying molecular topology. Two formats are supported:

- **Single-atom format**: Each row is an atom layer. Characters are element symbols (uppercase = sp2, lowercase = sp3). Spaces are empty. Bonds are inferred from zig-zag lattice topology.

```
    O o O
     c c
     c c
    c c c
    c c c
     c c
    O o O
```

- **Dimer format**: Rows alternate between atom rows and bond rows. Bond rows contain `|` (vertical dimer) and `-` (horizontal dimer).

The `run_kekule_solver()` function is the core entry point. It:
1. Filters bonds to only pi-active atoms (both endpoints must have `n_pi > 0`)
2. Constructs a `KekulePure` optimizer with the filtered bonds and n_pi array
3. Runs a two-phase solve: Phase 1 (delocalized/aromatic, Kloc=0), Phase 2 (localization with Kloc>0)
4. Returns a dict with `bo_raw`, `bo_snap`, `n_pi`, `k` (optimizer instance), `err`, and `report`

### 2. KekulePure Solver (`spammm/topology/KekulePure.py`)

A NumPy-based pi-bond-order optimizer. It uses an active-set constrained KKT solver:

- **Atom valence constraint**: `A @ bo = n_pi` (sum of pi bond orders at each atom must match its pi-electron count)
- **Aromatic energy**: Parabolic penalty centered at pi=0.5 (bond order 1.5)
- **Localization snap**: Three-piece parabola around 0, 0.5, 1.0 to push bonds toward discrete Kekule structures
- **Hard bounds**: [0, 1] clipping on bond orders

The `solve_constrained()` method iteratively solves the KKT system, clips violating bonds, fixes them, and re-solves until the active set stabilizes.

### 3. KekuleBackend (`spammm/topology/KekuleBackend.py`)

The GUI backend that manages molecular editing state on a hexagonal grid. Key attributes:

- `atom_npi`: Property returning a list of per-atom npi values. H caps have `npi = -1`, sp3 atoms have `npi = 0`, sp2 atoms have `npi = 1` (default), sp atoms have `npi = 2`.
- `set_atom_npi_by_id()`: Sets npi for a specific atom by stable ID.
- `set_atom_npi_by_index()`: Sets npi by array index.

The GUI's "pi" edit mode toggles atoms between sp2 (npi=1) and sp3 (npi=0) by clicking on them.

### 4. KekuleExtension GUI Panel (`spammm/GUI/KekuleExtension.py`)

A GUI extension panel providing:

- **ASCII art text editor**: Multi-line text input for specifying molecules
- **Example dropdown**: Pre-loaded example molecules (benzene, naphthalene, purine, etc.)
- **Generate button**: Parses ASCII art, builds an AtomicSystem, loads it into the GUI backend
- **Solve button**: Runs the Kekule solver on the ASCII-generated system
- **Solve Current button**: Runs the solver on the currently drawn molecule (from the hex grid), using `backend.atom_npi` for per-atom pi-electron counts
- **Solver parameters**: Kval (valence stiffness), Kloc (localization stiffness), Karo (aromatic stiffness), allow_aromatic toggle
- **Export buttons**: Save XYZ or MOL files
- **Bond order label toggle**: Show/hide numeric bond order values at bond midpoints

### 5. Bond Order Visualization (`spammm/GUI/VispyUtils.py`)

The `AtomScene` class was extended with:

- `bond_order_lines`: A Vispy `Line` visual for rendering bond order lines (double bonds as parallel lines, aromatic as green lines, single as thin gray lines)
- `bond_order_text`: A Vispy `Text` visual for optional bond order value labels
- `set_bond_orders(bonds, bond_orders, show_labels)`: Method to set bond order data and trigger redraw
- `set_bond_order_labels(show)`: Toggle label visibility

Rendering logic in `_redraw()`:
- **Aromatic bonds** (total bond order ~1.5): Single green line
- **Double bonds** (total > 1.7): Two parallel gray lines offset perpendicular to the bond direction
- **Single bonds** (total ~1.0): One thin gray line
- **Labels**: Numeric pi bond order value at bond midpoint (optional)

### 6. GUI Integration (`spammm/GUI/SPAMMM_GUI.py`)

The `KekuleExplorerWindow` was extended with:
- `bond_orders`, `bond_order_bonds`, `show_bond_order_labels` instance attributes
- In `refresh_view()`: Calls `scene.set_bond_orders()` with stored bond order data

### 7. Extension Manager (`spammm/GUI/ExtensionManager.py`)

The 'kekule' extension was registered in `EXTENSION_REGISTRY` with `enabled=True` and no dependencies, making it available in the GUI sidebar.

## How It Works

### Workflow

1. **Draw molecule**: User draws atoms on the hexagonal grid using Atom/Hex1/Hex2 edit modes
2. **Set hybridization**: User switches to "pi" edit mode and clicks atoms to toggle between sp2 (npi=1) and sp3 (npi=0). The backend stores this in `atom_npi`.
3. **Solve**: User clicks "Solve Current" in the Kekule extension panel
4. **Visualization**: The solver returns bond orders, which are stored on the window and rendered in the next `refresh_view()` call

### Data Flow

```
backend.atom_npi  (user-set, includes H caps with npi=-1)
       |
       v
run_kekule_solver(sys, n_pi=atom_npi)
       |
       v
  Filter bonds: only keep bonds where both atoms have n_pi > 0
       |
       v
  KekulePure(atoms, n_pi=n_pi, bonds=bonds_pi)
       |
       v
  solve_quadratic() -> solve_snap()  (two-phase optimization)
       |
       v
  Returns: bo_raw, bo_snap, n_pi, report
       |
       v
  _store_bond_orders_on_window()
       |
       v
  refresh_view() -> scene.set_bond_orders() -> _redraw()
```

## Problems Encountered and Solutions

### Problem 1: Solver ignored user-set n_pi

**Symptom**: When user set n_pi=0 on two para carbons in benzene, the solver still showed fully delocalized (aromatic) bonds.

**Root cause**: `run_kekule_solver()` called `make_n_pi(atoms)` internally, which derives n_pi from element name **case** (uppercase = sp2, lowercase = sp3). The GUI always stores uppercase element names and tracks hybridization separately in `backend.atom_npi`. So the solver always saw all atoms as sp2 regardless of user settings.

**Fix**: Added `n_pi=None` parameter to `run_kekule_solver()`. When provided, uses it directly instead of calling `make_n_pi()`. The `_on_solve_current()` callback in KekuleExtension passes `window.backend.atom_npi` as this parameter.

### Problem 2: Solver crashed on unsatisfiable constraints

**Symptom**: `RuntimeError: KekulePure.solve_constrained(): atom-sum constraint not satisfied, max|A@bo-n_pi|=1.000e+00`

**Root cause**: The solver raised an exception when it couldn't perfectly satisfy the `A@bo = n_pi` constraint. This happened whenever the n_pi configuration was chemically unsatisfiable (e.g., odd electron count, disconnected pi system).

**Fix**: Changed `solve_constrained()` to print a warning and return the best-effort solution instead of raising. This allows the GUI to still display partial results.

### Problem 3: H cap atoms caused spurious constraint violations

**Symptom**: Persistent `WARNING: atom-sum constraint not satisfied, max|A@bo-n_pi|=1.000e+00` warnings even for valid configurations.

**Root cause**: H cap atoms in the backend have `npi = -1`. The solver's constraint matrix `_A` has shape `(natom, nbond)` covering ALL atoms. H caps have all-zero rows in `_A` (no pi bonds), so `A@bo = 0` for them, but `n_pi = -1`, giving `|0 - (-1)| = 1.0`.

**Fix**: Two-part fix:
1. Bond filtering: Changed from "heavy-atom bonds" to "pi-active bonds" — only bonds where **both** atoms have `n_pi > 0` are included in the solver. This excludes H caps and sp3 atoms entirely.
2. Constraint check: Changed `solve_constrained()` to only check the constraint for pi-active atoms (`n_pi > 0`), since non-pi atoms trivially have `A@bo = 0` and shouldn't be checked against `n_pi = -1` or `n_pi = 0`.

### Problem 4: Error messages not copyable from GUI

**Symptom**: Solver error messages displayed in a QLabel could not be selected/copied to clipboard.

**Fix**: Added `setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)` to the status label.

### Problem 5: Errors not printed to terminal

**Symptom**: Solver errors only appeared in the GUI status label, not in the terminal, making debugging difficult.

**Fix**: Added `print()` calls with `[KEKULE]` prefix to all error paths. Also preserved the full traceback in `run_kekule_solver` by storing `traceback.format_exc()` in `report['traceback']`, since the exception is caught internally.

### Problem 6: Debug visibility into solver inputs

**Symptom**: No way to see what n_pi configuration and bonds were being sent to the solver.

**Fix**: Added debug prints in `run_kekule_solver()` showing:
- Number of atoms, total bonds, and pi-active bonds
- Per-atom n_pi values and element names
- List of pi-active bond index pairs

Example output:
```
[KEKULE] Solving: natoms=14 nbonds_all=14 nbonds_pi=2
[KEKULE] n_pi=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0, -1.0, ...]  enames=['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', ...]
[KEKULE] pi-bonds=[[0, 5], [2, 3]]
```

## Files Modified

| File | Changes |
|------|---------|
| `spammm/topology/ascii_art_heterocycle.py` | Added `n_pi` parameter to `run_kekule_solver()`, changed bond filtering to pi-active atoms, added debug prints, preserved traceback in report |
| `spammm/topology/KekulePure.py` | Changed `solve_constrained()` to warn instead of raise, constraint check only on pi-active atoms |
| `spammm/GUI/KekuleExtension.py` | Created module with UI panel, solver callbacks, bond order storage, error printing, status label selectability |
| `spammm/GUI/ExtensionManager.py` | Registered 'kekule' extension |
| `spammm/GUI/VispyUtils.py` | Added `bond_order_lines`, `bond_order_text` visuals, `set_bond_orders()` method, rendering in `_redraw()` |
| `spammm/GUI/SPAMMM_GUI.py` | Added bond order attributes to window, integrated into `refresh_view()` |
