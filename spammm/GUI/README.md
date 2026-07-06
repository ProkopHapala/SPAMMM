# GUI/

PyQt5 GUI for molecular editing and AFM simulation. Main window combines VisPy 3D scene with lazy-loaded extension panels.

## Naming

| Current | Role |
|---------|------|
| **SPAMMMWindow** (`SPAMMM_GUI.py`) | Main window — molecular editor + extensions |
| **MoleculeEditorBackend** | Topology editing backend (not the Kekule solver) |
| **KekuleExtension** | GUI panel for **KekulePure** bond-order solver only |
| ~~KekuleExplorerWindow~~ | Legacy alias → `SPAMMMWindow` |

- **SPAMMM_GUI.py** — Main application window (`SPAMMMWindow`): VisPy 3D scene + hex-grid editor + extension panels
- **BaseGUI.py** — Reusable PyQt5 widget helpers
- **ExtensionManager.py** — Lazy-loading extension system (AFM, FF, QEq, Kekule solver, ASCII builder, …)
- **VispyUtils.py** — VisPy 3D molecular visualization (`AtomScene`)
- **KekuleExtension.py** — Kekule π-bond-order solver panel (calls `KekulePure`, writes `Bond.order`)
- **AsciiArtExtension.py** — ASCII art molecule builder + H-bond resolution
- **AFMExtension.py**, **FFExtension.py**, **QEqExtension.py** — simulation / FF / charge panels
- **EditModeHandlers.py** — per-mode mouse/keyboard dispatch for the editor
