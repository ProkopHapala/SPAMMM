# GUI/

PyQt5 GUI for molecular editing and AFM simulation. Main window combines VisPy 3D scene with lazy-loaded extension panels.

- **SPAMMM_GUI.py** — Main application window (KekuleExplorerGUI): VisPy 3D molecular scene + editor + extension panels
- **BaseGUI.py** — Reusable PyQt5 widget helpers: buttons, checkboxes, combo boxes, spin boxes, layout containers
- **ExtensionManager.py** — Lazy-loading extension system: dynamically load optional functionality (AFM, DFTB, SPFF, POV-Ray) without hard dependencies
- **VispyUtils.py** — Reusable VisPy 3D molecular visualization: AtomScene widget (spheres, cylinders, arrows, text labels)
- **MoleculeViewer.py** — Standalone modular 3D molecular viewer (VisPy): composable rendering, picking, measurement layers
- **GLGUI.py** — OpenGL widget base and rendering utilities for PyQt5 (instanced rendering, mesh, shader management)
- **MolecularBrowser.py** — Database browser for molecular structures with 3D ball-and-stick preview (OpenGL)
- **MolecularBrowserVispy.py** — ACDSee-style molecular file browser using VisPy (thumbnail grid + detail view)
- **ThumbnailCache.py** — Offscreen molecule thumbnail rendering with lazy job queue (in-memory, no disk cache)
- **AFMExtension.py** — AFM/STM simulation panel: setup, run, dirty-flag incremental recomputation via ModularAFMPipeline
- **FFExtension.py** — Forcefield relaxation/MD panel: edit/view modes built on FFController + ExtensionManager.UIComponents
- **QEqExtension.py** — Charge Equilibration panel: compute partial atomic charges via QEq direct matrix solve
- **KekuleExtension.py** — Heterocycle generation and Kekule bond-order optimization panel
- **CollapsibleSection.py** — Animated foldable panel widget for organizing UI sections
- **DirectoryNavigator.py** — Directory reading and navigation for molecular browser (extension filtering)
- **plotutils.py** — Qt-specific 2D plotting: re-exports pure-matplotlib functions from plotUtils.py, adds Qt-FigureCanvas wrapper
- **shaders/** — GLSL shader files (sphere, cylinder, text billboard, instanced rendering)
