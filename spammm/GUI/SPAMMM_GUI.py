#!/usr/bin/env python3
"""
KekuleExplorerGUI.py — Main application window for molecular editing and AFM simulation.

Purpose: Provide the primary user interface combining a VisPy 3D molecular scene
with PyQt5 control panels. Supports atom/bond editing, hexagonal grid drawing,
AFM/STM simulation setup, and geometry relaxation.

Key functionality:
  - VisPy AtomScene for 3D molecular visualization (atoms, bonds, forces, picking)
  - PyQt5 panels: element selector, passivation groups, AFM controls, settings
  - Extension manager integration (AFM, DFTB, SPFF extensions)
  - Hexagonal grid snapping mode for graphene-like structures
  - XYZ export and screenshot capture

Role in SPAMMM: The central GUI hub. All user interaction flows through here:
editing commands → KekuleBackend, AFM commands → AFMExtension, rendering → VispyUtils.

CODE STYLE POLICIES:
- Strive for concise, general, and reusable code
- Modularity and composability over duplication
- Minimize code duplication
- Prefer single-line function calls and messages
- Extract repeated logic to shared utilities (e.g., VispyUtils.py)
- Use BaseGUI helper methods for widget creation to reduce boilerplate
- Use polymorphic functions with default arguments instead of specialized variants
- Consolidate similar functions (e.g., spinBox/spinBoxInt → spinBox(int_mode=True))
- Refactor if-else labyrinths into general functions with callbacks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from spammm.GUI.BaseGUI import BaseGUI
from spammm.GUI.VispyUtils import AtomScene
from spammm.topology.KekuleBackend import KekuleBackend
import spammm.topology.KekuleBackend as KB
from spammm import atomicUtils as au
from vispy import scene

# Global verbosity level for debug prints
# 0: Only exceptions and explicit prints
# 1: Warnings and complex operation reports
# 2: Click and action prints (default)
# 3: Hovered prints (most verbose)
VERBOSITY_LEVEL = 2

def debug_print(level, message):
    """Print message if verbosity level is >= specified level."""
    global VERBOSITY_LEVEL
    if VERBOSITY_LEVEL >= level:
        print(message)

from spammm.GUI import VispyUtils as vu
from spammm import elements
from spammm import atomicUtils as au
from spammm import elements
from spammm.topology.PackedMolecule import PackedMolecule, UndoStack, _z_to_ename
from spammm.GUI.BaseGUI import BaseGUI
from spammm.GUI.VispyUtils import compute_bond_colors_by_length, generate_atom_labels

from spammm.GUI.ExtensionManager import ExtensionManager, ExtensionNotAvailableError
from spammm.GUI.CollapsibleSection import CollapsibleSection

class KekuleExplorerWindow(BaseGUI):
    sig_geometry_changed = QtCore.pyqtSignal()  # Emitted whenever atom geometry changes

    def __init__(self, output_dir=None, fdata_path=None, verbosity=None):
        super().__init__("Kekule Structure Explorer")
        self.resize(1024, 768)

        self.extensions = ExtensionManager()
        self.backend = KekuleBackend()
        self.cur_atom_type = 'C'
        self.edit_mode = 'Hex1'  # 'Hex1' (paint), 'Hex2' (toggle), 'Atom', 'Bond', 'pi', 'Select'
        self.label_mode = 'Element+Index'
        self.pick_radius = 0.5  # Distance in Angstroms for atom picking (matches spinbox default)
        self.bond_orders = None  # pi bond orders array (set by KekuleExtension solver)
        self.bond_order_bonds = None  # (m,2) heavy-atom bond indices matching bond_orders
        self.show_bond_order_labels = False

        # Output directory for saved images (screenshots, plots)
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = output_dir or os.path.join(_repo_root, 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Override global verbosity if specified
        global VERBOSITY_LEVEL
        if verbosity is not None:
            VERBOSITY_LEVEL = verbosity

        # Load settings
        self.settings = QtCore.QSettings("FireCore", "KekuleExplorer")
        self.fdata_path = fdata_path or self.settings.value("fdata_path", "/home/prokop/Fireball/Fdata_HCNOS")
        # Sync fdata_path into ExtensionManager config so FireCore/Grid can find it
        self.extensions.set_config('firecore', 'fdata_dir', self.fdata_path)
        self.initUI()
        # Scene drag signal: update AtomicGraph and sys.apos after drag end
        self.scene.sig_drag_state.connect(self.on_drag_state)
        self.scene.sig_rmb_remove.connect(self.on_atom_remove)
        self.scene.sig_camera_changed.connect(self.refresh_view)
        self.scene.sig_link_bond.connect(self.on_link_bond)
        self.scene.sig_atom_clicked.connect(self.on_atom_clicked)
        self.refresh_view()

    def initUI(self):
        # --- Central Widget (Vispy Scene) ---
        self.scene = vu.AtomScene(bgcolor=(0.95, 0.95, 0.95), backend=self.backend)
        self.scene.pick_radius = self.pick_radius
        
        # Link axes to view
        self.scene.view.parent = None # Re-parent from central_widget to grid
        grid = self.scene.canvas.central_widget.add_grid(spacing=0, margin=10)
        
        self.axis_x = scene.AxisWidget(orientation='bottom', axis_label='x (A)', font_size=8)
        self.axis_y = scene.AxisWidget(orientation='left', axis_label='y (A)', font_size=8)
        
        self.axis_x.height_max = 30
        self.axis_y.width_max = 40

        grid.add_widget(self.axis_y, row=0, col=0)
        grid.add_widget(self.scene.view, row=0, col=1)
        grid.add_widget(self.axis_x, row=1, col=1)
        
        self.scene.view.stretch = (1, 1)
        
        self.axis_x.link_view(self.scene.view)
        self.axis_y.link_view(self.scene.view)
        
        # Configure axis after linking to view
        self.axis_x.axis.text_color = 'black'
        self.axis_y.axis.text_color = 'black'
        self.axis_x.axis.tick_color = 'black'
        self.axis_y.axis.tick_color = 'black'

        # --- Main Layout with Side Panel ---
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        
        # Create side panel
        side_panel = QtWidgets.QFrame()
        side_panel.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        side_panel.setFixedWidth(280)
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setSpacing(1)
        
        # Add sections
        side_layout.addWidget(self.create_editors_section())
        side_layout.addWidget(self.create_grid_section())
        side_layout.addWidget(self.create_ribbon_section())
        self._build_extension_panels(side_layout)
        side_layout.addStretch()
        
        # Add to main layout
        main_layout.addWidget(side_panel)
        main_layout.addWidget(self.scene.canvas.native)
        
        self.setCentralWidget(main_widget)

        # Add grid guide markers
        self.grid_markers = scene.visuals.Markers(parent=self.scene.view.scene)
        self.grid_markers.set_gl_state('translucent', depth_test=False)
        self.grid_markers.order = -1  # Behind everything

        # Add mouse cursor (cross) for debugging
        self.cursor_markers = scene.visuals.Markers(parent=self.scene.view.scene)
        self.cursor_markers.set_gl_state('translucent', depth_test=False)
        self.cursor_markers.order = 10  # On top
        self.cursor_markers.set_data(
            pos=np.zeros((1, 3)),
            symbol='cross',
            edge_width=2,
            edge_color='red',
            face_color='transparent',
            size=10
        )

        # Add debug markers for grid node -> atom mappings (cyan)
        self.debug_markers = scene.visuals.Markers(parent=self.scene.view.scene)
        self.debug_markers.set_gl_state('translucent', depth_test=False)
        self.debug_markers.order = 5  # Behind atoms but in front of grid

        # Add hover markers for highlighting hexagon under mouse
        self.hover_markers = scene.visuals.Markers(parent=self.scene.view.scene)
        self.hover_markers.set_gl_state('translucent', depth_test=False)
        self.hover_markers.order = 6  # On top of debug markers

        # Add debug lines for node -> atom connections
        self.debug_lines = scene.visuals.Line(parent=self.scene.view.scene)
        self.debug_lines.set_gl_state('translucent', depth_test=False)
        self.debug_lines.order = 4  # Behind debug markers

        # Help / Status
        self.statusBar().showMessage("LMB: Add/Toggle | RMB: Remove | Middle-Click: Toggle H | Scroll: Zoom | Arrow Keys: Pan")
        self.scene.lock_drag = True   # Default mode is Ring, no dragging
        self.scene.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.scene.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.scene.sig_selection_changed.connect(self.on_selection_changed)
        self.copied_packed = None  # PackedMolecule for internal copy/paste
        self.undo_stack = UndoStack(maxlen=100)
        self.undo_enabled = True
        self.scene.canvas.events.key_press.connect(self.on_key_press)
        self.create_menus()
        self.error_print = True      # Print to stdout
        self.error_raise = True      # Raise exception
        self.error_dialog = True     # Show QMessageBox
        self.error_statusbar = True  # Update status bar

    def _raise(self, msg, title="Error", dialog_type="critical"):
        """Reusable error handling function.
        
        Args:
            msg: Error message
            title: Dialog title
            dialog_type: 'critical', 'warning', or 'information'
        """
        if self.error_print:
            print(msg)
        if self.error_statusbar:
            self.statusBar().showMessage(msg)
        if self.error_dialog:
            if dialog_type == "critical":
                QtWidgets.QMessageBox.critical(self, title, str(msg))
            elif dialog_type == "warning":
                QtWidgets.QMessageBox.warning(self, title, str(msg))
            elif dialog_type == "information":
                QtWidgets.QMessageBox.information(self, title, str(msg))
        if self.error_raise:
            raise RuntimeError(msg)

    def install_mpl_canvas_screenshot_menu(self, canvas, fig, *, default_name='plot.png'):
        def _on_menu(pos):
            menu = QtWidgets.QMenu(canvas)
            act_save = menu.addAction('Save Screenshot...')
            action = menu.exec_(canvas.mapToGlobal(pos))
            if action is act_save:
                start_dir = getattr(self, 'output_dir', os.getcwd())
                if hasattr(self, 'settings'):
                    try:
                        start_dir = str(self.settings.value('last_screenshot_dir', start_dir))
                    except Exception:
                        pass
                start_path = os.path.join(start_dir, default_name)
                fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Screenshot', start_path, 'PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)')
                if not fname:
                    return
                if hasattr(self, 'settings'):
                    try:
                        self.settings.setValue('last_screenshot_dir', os.path.dirname(fname))
                    except Exception:
                        pass
                fig.savefig(fname, dpi=200, bbox_inches='tight')

        canvas.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        canvas.customContextMenuRequested.connect(_on_menu)

    def create_editors_section(self):
        """Merged Builder and Editor section as collapsible panel."""
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(3)
        
        # Edit Mode (from Builder)
        self.label("Edit Mode:", layout=layout)
        self.mode_combo = self.comboBox(["Hex1", "Hex2", "Atom", "Bond", "Ring", "pi", "Select"], self.set_edit_mode, layout=layout)
        
        # Atom type and auto h-cap (from Builder)
        row = QtWidgets.QHBoxLayout()
        # Ring size spinbox (visible in Ring mode)
        self.ring_size_label = QtWidgets.QLabel("Ring size:")
        self.ring_size_spinbox = self.spinBox(5, 1.0, max_width=50, vmin=3, vmax=12, int_mode=True)
        self.ring_size_label.setVisible(False)
        self.ring_size_spinbox.setVisible(False)
        self.label("Type:", layout=row)
        self.atom_combo = self.comboBox(["C", "N", "O"], self.set_atom_type, layout=row)
        self.auto_h_cap_btn = self.button("Auto H", self.toggle_auto_h_cap, layout=row)
        self.auto_h_cap_btn.setCheckable(True)
        self.auto_h_cap_btn.setChecked(self.backend.auto_h_cap)
        self.auto_bonds_btn = self.button("Auto Bonds", self.toggle_auto_recalc_bonds, layout=row)
        self.auto_bonds_btn.setCheckable(True)
        self.auto_bonds_btn.setChecked(self.backend.auto_recalc_bonds)
        layout.addLayout(row)

        # Ring size row (visible only in Ring mode)
        row_ring = QtWidgets.QHBoxLayout()
        row_ring.addWidget(self.ring_size_label)
        row_ring.addWidget(self.ring_size_spinbox)
        layout.addLayout(row_ring)

        # Pick radius (kept in main editors section)
        row_grid = QtWidgets.QHBoxLayout()
        self.label("Pick Radius:", layout=row_grid)
        self.pick_radius_spinbox = self.spinBox(0.5, 0.1, max_width=60, vmin=0.1, vmax=5.0)
        self.pick_radius_spinbox.valueChanged.connect(self.set_pick_radius)
        row_grid.addWidget(self.pick_radius_spinbox)
        layout.addLayout(row_grid)

        # Editor buttons (from Editor)
        row1 = QtWidgets.QHBoxLayout()
        self.button("Snap", self.reset_offsets, layout=row1)
        self.button("Adj H", self.adjust_h, layout=row1)
        self.button("AutoBonds", self.recalc_bonds, layout=row1)
        layout.addLayout(row1)
        
        # Labels combo (from Editor)
        row2 = QtWidgets.QHBoxLayout()
        self.label("Labels:", layout=row2)
        self.label_combo = self.comboBox(["Element+Index", "Atomic Type", "Pi Orbitals", "Z-Height", "Charge", "Bond Lengths"], self.set_label_mode, layout=row2)
        layout.addLayout(row2)
        
        # Visualization buttons (from Editor)
        row3 = QtWidgets.QHBoxLayout()
        self.bond_viz_mode = False
        self.button("Bond Colors", self.toggle_bond_viz, layout=row3).setCheckable(True)
        self.debug_view_mode = True
        self.debug_btn = self.button("Debug View", self.toggle_debug_view, layout=row3)
        self.debug_btn.setCheckable(True)
        self.debug_btn.setChecked(True)
        layout.addLayout(row3)
        
        # Export/Import buttons
        row4 = QtWidgets.QHBoxLayout()
        self.button("Show", self.show_xyz, layout=row4)
        self.button("Export", self.export_structure, layout=row4)
        self.button("Import", self.import_structure, layout=row4)
        layout.addLayout(row4)
        
        # Wrap layout in QWidget for CollapsibleSection
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        
        # Wrap in CollapsibleSection
        sec = CollapsibleSection("Editors", collapsed=True, parent=self)
        sec.setContent(widget)
        return sec

    def create_grid_section(self):
        """Grid transform controls in a compact collapsible panel."""
        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(2)

        # Row 1: a_CC (lattice constant) | Transpose toggle
        row1 = QtWidgets.QHBoxLayout()
        self.label("a_CC:", layout=row1)
        self.a_CC_spin = self.spinBox(1.42, 0.01, max_width=55, vmin=0.5, vmax=5.0, decimals=3)
        self.a_CC_spin.valueChanged.connect(self.set_a_CC)
        row1.addWidget(self.a_CC_spin)
        self.grid_transpose_btn = self.button("Transpose", self.transpose_grid, layout=row1)
        self.grid_transpose_btn.setCheckable(True)
        self.grid_transpose_btn.setChecked(self.backend.grid.transpose)
        layout.addLayout(row1)

        # Row 2: Rotate° | Offset unit checkbox
        row2 = QtWidgets.QHBoxLayout()
        self.label("Rot°:", layout=row2)
        self.grid_rotate_spin = self.spinBox(0.0, 1.0, max_width=55, vmin=-180.0, vmax=180.0, decimals=1)
        self.grid_rotate_spin.valueChanged.connect(self.set_grid_rotation)
        row2.addWidget(self.grid_rotate_spin)
        self.grid_offset_unit_chk = self.checkBox("grid units", checked=True, callback=self.toggle_offset_unit, layout=row2)
        layout.addLayout(row2)

        # Row 3: Offset X, Y (step depends on unit mode)
        row3 = QtWidgets.QHBoxLayout()
        self.label("Off:", layout=row3)
        self.grid_offset_x_spin = self.spinBox(0.0, 1.0, max_width=50, vmin=-20.0, vmax=20.0, decimals=3)
        self.grid_offset_x_spin.valueChanged.connect(self.set_grid_offset_x)
        row3.addWidget(self.grid_offset_x_spin)
        self.grid_offset_y_spin = self.spinBox(0.0, 1.0, max_width=50, vmin=-20.0, vmax=20.0, decimals=3)
        self.grid_offset_y_spin.valueChanged.connect(self.set_grid_offset_y)
        row3.addWidget(self.grid_offset_y_spin)
        layout.addLayout(row3)

        # Row 4: Reset
        row4 = QtWidgets.QHBoxLayout()
        self.button("Reset Grid", self.reset_grid_transform, layout=row4)
        layout.addLayout(row4)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        sec = CollapsibleSection("Grid Transform", collapsed=True, parent=self)
        sec.setContent(widget)
        return sec

    def _build_extension_panels(self, side_layout):
        """Dynamically add collapsible panels for each enabled extension."""
        # Register edit/view mode lists for dynamic dispatch
        self._ext_edit_modes = {}   # label -> callback
        self._ext_view_modes = {}   # label -> callback

        EXTENSION_TITLES = {'ff': 'Force Field', 'afm': 'AFM', 'dftb': 'DFTB', 'firecore': 'FireCore'}
        for name in self.extensions.enabled_extensions():
            ui = self.extensions.build_ui(name, self)
            title = EXTENSION_TITLES.get(name, name.capitalize())
            sec = CollapsibleSection(title, collapsed=True, parent=self)
            if ui.panel is not None:
                sec.setContent(ui.panel)
                ok = self.extensions.is_loaded(name)
                sec.set_status(ok, '' if ok else self.extensions.status(name).replace('error: ', '')[:30])
            else:
                # Extension failed to load: show reason
                reason = self.extensions.status(name)
                lbl = QtWidgets.QLabel(reason.replace('error: ', ''))
                lbl.setWordWrap(True)
                lbl.setStyleSheet('color: gray; font-style: italic;')
                sec.setContent(lbl)
                sec.set_status(False)
            side_layout.addWidget(sec)

            for label, cb in ui.edit_modes:
                self._ext_edit_modes[label] = cb
                self.mode_combo.addItem(label)
            for label, cb in ui.view_modes:
                self._ext_view_modes[label] = cb

    def set_view_mode(self, mode: str):
        """Called by extension view-mode callbacks."""
        debug_print(2, f"View mode: {mode}")

    def create_ribbon_section(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Shared inputs (used by both single and two-ribbon)
        shared_layout = QtWidgets.QHBoxLayout()
        shared_layout.setContentsMargins(0, 0, 0, 0)
        shared_layout.setSpacing(2)
        
        rows_label = QtWidgets.QLabel("Rows:")
        rows_label.setFixedWidth(30)
        self.ribbon_rows_spinbox = self.spinBox(4, 1.0, max_width=50, vmin=1, vmax=20, int_mode=True)
        shared_layout.addWidget(rows_label)
        shared_layout.addWidget(self.ribbon_rows_spinbox)
        
        bottom_label = QtWidgets.QLabel("Bot:")
        bottom_label.setFixedWidth(25)
        self.ribbon_bottom_edit = QtWidgets.QLineEdit()
        self.ribbon_bottom_edit.setPlaceholderText("n/N/o/O/H/h")
        self.ribbon_bottom_edit.setMaximumWidth(80)
        shared_layout.addWidget(bottom_label)
        shared_layout.addWidget(self.ribbon_bottom_edit)
        
        top_label = QtWidgets.QLabel("Top:")
        top_label.setFixedWidth(25)
        self.ribbon_top_edit = QtWidgets.QLineEdit()
        self.ribbon_top_edit.setPlaceholderText("n/N/o/O/H/h")
        self.ribbon_top_edit.setMaximumWidth(80)
        shared_layout.addWidget(top_label)
        shared_layout.addWidget(self.ribbon_top_edit)
        
        layout.addLayout(shared_layout)
        
        # Generate buttons side by side
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(2)
        self.button("Single", self.generate_single_ribbon, layout=btn_layout)
        self.button("Two", self.generate_two_ribbons, layout=btn_layout)
        layout.addLayout(btn_layout)
        
        # Two-ribbon specific inputs (collapsible)
        two_ribbon_group = QtWidgets.QGroupBox("Two-Ribbon Options")
        two_ribbon_group.setCheckable(True)
        two_ribbon_group.setChecked(False)
        two_ribbon_group.toggled.connect(lambda checked: two_ribbon_group.setVisible(checked))
        two_ribbon_layout = QtWidgets.QVBoxLayout()
        two_ribbon_layout.setContentsMargins(0, 0, 0, 0)
        two_ribbon_layout.setSpacing(2)
        
        # Ribbon 2 inputs
        r2_layout = QtWidgets.QHBoxLayout()
        r2_layout.setContentsMargins(0, 0, 0, 0)
        r2_layout.setSpacing(2)
        
        r2_rows_label = QtWidgets.QLabel("R2:")
        r2_rows_label.setFixedWidth(20)
        self.ribbon2_rows_spinbox = self.spinBox(4, 1.0, max_width=50, vmin=1, vmax=20, int_mode=True)
        r2_layout.addWidget(r2_rows_label)
        r2_layout.addWidget(self.ribbon2_rows_spinbox)
        
        r2_bottom_label = QtWidgets.QLabel("Bot:")
        r2_bottom_label.setFixedWidth(25)
        self.ribbon2_bottom_edit = QtWidgets.QLineEdit()
        self.ribbon2_bottom_edit.setPlaceholderText("n/N/o/O/H/h")
        self.ribbon2_bottom_edit.setMaximumWidth(80)
        r2_layout.addWidget(r2_bottom_label)
        r2_layout.addWidget(self.ribbon2_bottom_edit)
        
        r2_top_label = QtWidgets.QLabel("Top:")
        r2_top_label.setFixedWidth(25)
        self.ribbon2_top_edit = QtWidgets.QLineEdit()
        self.ribbon2_top_edit.setPlaceholderText("n/N/o/O/H/h")
        self.ribbon2_top_edit.setMaximumWidth(80)
        r2_layout.addWidget(r2_top_label)
        r2_layout.addWidget(self.ribbon2_top_edit)
        
        two_ribbon_layout.addLayout(r2_layout)
        
        # H-bond spacing
        hb_layout = QtWidgets.QHBoxLayout()
        hb_layout.setContentsMargins(0, 0, 0, 0)
        hb_layout.setSpacing(2)
        hb_label = QtWidgets.QLabel("H-bond:")
        hb_label.setFixedWidth(50)
        self.ribbon_L_Hb_spinbox = self.spinBox(3.0, 0.1, max_width=60, vmin=2.0, vmax=10.0)
        hb_layout.addWidget(hb_label)
        hb_layout.addWidget(self.ribbon_L_Hb_spinbox)
        two_ribbon_layout.addLayout(hb_layout)
        
        two_ribbon_group.setLayout(two_ribbon_layout)
        layout.addWidget(two_ribbon_group)
        
        # Wrap layout in QWidget for CollapsibleSection
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        
        # Wrap in CollapsibleSection
        sec = CollapsibleSection("Ribbon", collapsed=True, parent=self)
        sec.setContent(widget)
        return sec

    def generate_single_ribbon(self):
        """Generate single periodic ribbon from passivation strings."""
        bottom_str = self.ribbon_bottom_edit.text().strip()
        top_str = self.ribbon_top_edit.text().strip()
        
        if not bottom_str or not top_str:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please provide passivation strings for both bottom and top edges.")
            return
        
        try:
            bottom_passivation = parse_passivation_string(bottom_str)
            top_passivation = parse_passivation_string(top_str)
            length_cells = len(bottom_passivation)
            width_chains = self.ribbon_rows_spinbox.value()
            Lx = 2.4
            
            self.backend = KekuleBackend()
            self.backend.build_zigzag_ribbon(width_chains=width_chains, length_cells=length_cells, passivation_bottom=bottom_passivation, passivation_top=top_passivation, scale_x=Lx / (2.0 * 1.42 * np.cos(np.pi / 6)), bPeriodicX=True)
            
            self.scene.backend = self.backend
            
            n_C = sum(1 for e in self.backend.sys.enames if e == 'C')
            n_N = sum(1 for e in self.backend.sys.enames if e == 'N')
            n_O = sum(1 for e in self.backend.sys.enames if e == 'O')
            n_H = sum(1 for e in self.backend.sys.enames if e == 'H')
            
            msg = f"Generated single ribbon: C={n_C}, N={n_N}, O={n_O}, H={n_H}"
            debug_print(1, msg)
            self.statusBar().showMessage(msg)
            self.refresh_view()
            
        except Exception as e:
            self._raise(f"Ribbon generation FAILED: {e}", title="Ribbon Error")

    def generate_two_ribbons(self):
        """Generate two-ribbon system from passivation strings."""
        bottom1_str = self.ribbon_bottom_edit.text().strip()
        top1_str = self.ribbon_top_edit.text().strip()
        bottom2_str = self.ribbon2_bottom_edit.text().strip()
        top2_str = self.ribbon2_top_edit.text().strip()
        
        if not all([bottom1_str, top1_str, bottom2_str, top2_str]):
            QtWidgets.QMessageBox.warning(self, "Warning", "Please provide passivation strings for all four edges.")
            return
        
        try:
            bottom1_passivation = parse_passivation_string(bottom1_str)
            top1_passivation = parse_passivation_string(top1_str)
            bottom2_passivation = parse_passivation_string(bottom2_str)
            top2_passivation = parse_passivation_string(top2_str)
            length_cells = len(bottom1_passivation)
            width_chains1 = self.ribbon_rows_spinbox.value()
            width_chains2 = self.ribbon2_rows_spinbox.value()
            Lx = 2.4
            L_Hb = self.ribbon_L_Hb_spinbox.value()
            
            # Build bottom ribbon
            bottom_ribbon = KekuleBackend()
            bottom_ribbon.build_zigzag_ribbon(width_chains=width_chains1, length_cells=length_cells, passivation_bottom=bottom1_passivation, passivation_top=top1_passivation, scale_x=Lx / (2.0 * 1.42 * np.cos(np.pi / 6)), bPeriodicX=True)
            
            # Build top ribbon
            top_ribbon = KekuleBackend()
            top_ribbon.build_zigzag_ribbon(width_chains=width_chains2, length_cells=length_cells, passivation_bottom=bottom2_passivation, passivation_top=top2_passivation,  scale_x=Lx / (2.0 * 1.42 * np.cos(np.pi / 6)), bPeriodicX=True)
            
            # Combine ribbons
            self.backend = KekuleBackend()
            self.backend.combine_ribbons(bottom_ribbon, top_ribbon, L_Hb=L_Hb, shift_x=0.0)
            
            self.scene.backend = self.backend
            
            n_C = sum(1 for e in self.backend.sys.enames if e == 'C')
            n_N = sum(1 for e in self.backend.sys.enames if e == 'N')
            n_O = sum(1 for e in self.backend.sys.enames if e == 'O')
            n_H = sum(1 for e in self.backend.sys.enames if e == 'H')
            
            msg = f"Generated two-ribbon system: C={n_C}, N={n_N}, O={n_O}, H={n_H}"
            debug_print(1, msg)
            self.statusBar().showMessage(msg)
            self.refresh_view()
            
        except Exception as e:
            self._raise(f"Two-ribbon generation FAILED: {e}", title="Ribbon Error")

    def create_menus(self):
        # Settings Menu
        self.settings_menu = self.menuBar().addMenu("Settings")
        # Fdata path is now in Fireball section, but keep menu for convenience
        self.settings_menu.addAction("Set Fdata Path", self.set_fdata_path)

    def update_orbital_energy_label(self, value):
        """Update orbital energy label when spinbox value changes."""
        if hasattr(self, '_eigen'):
            idx = int(value)
            if 0 <= idx < len(self._eigen):
                self.orbital_info_label.setText(f"Orbital {idx} E: {self._eigen[idx]:.3f} eV")
            else:
                self.orbital_info_label.setText("Orbital: Invalid index")

    def plot_orbital_from_spinbox(self):
        """Plot orbital at the index selected in spinbox."""
        if not hasattr(self, '_eigen'):
            msg = "Please run Compute SCF first."
            debug_print(1, f"INFO: {msg}")
            QtWidgets.QMessageBox.information(self, "Info", msg)
            return
        mo_idx = self.orbital_spinbox.value()
        if mo_idx < 0 or mo_idx >= len(self._eigen):
            msg = f"Invalid orbital index: {mo_idx}"
            debug_print(1, f"WARNING: {msg}")
            QtWidgets.QMessageBox.warning(self, "Warning", msg)
            return
        z_height = self.z_height_spinbox.value()
        self.statusBar().showMessage(f"Projecting MO {mo_idx + 1}...")
        QtWidgets.QApplication.processEvents()
        try:
            apos = self.backend.sys.apos
            grid_origin, size, center_z = self._compute_extent_from_geometry(apos)
            points, extent, n = self._make_2d_grid(grid_origin, size, center_z, z_height)
            flat_data = self._evaluate_on_grid(points, 'orbital', orb_index=mo_idx)
            data_2d = np.asarray(flat_data, dtype=np.float64).reshape(n, n)
            E = self._eigen[mo_idx]
            pos = apos.astype(np.float32)
            enames = self.backend.sys.enames
            self._plot_2d_projection(data_2d, extent, title=f"MO {mo_idx + 1} E={E:+.3f} eV  z={z_height:.1f}Å", cmap='bwr', symmetric=True, atom_pos=pos, atom_types=enames)
            msg = f"Plotted MO {mo_idx + 1}"
            debug_print(1, msg)
            self.statusBar().showMessage(msg)
        except Exception as e:
            self._raise(f"Plot FAILED: {e}", title="Plot Error")

    def set_edit_mode(self, mode):
        # Dispatch to extension edit mode callbacks first
        if hasattr(self, '_ext_edit_modes') and mode in self._ext_edit_modes:
            try:
                self._ext_edit_modes[mode]()
            except ExtensionNotAvailableError as e:
                self.statusBar().showMessage(str(e))
            return
        self.edit_mode = mode
        debug_print(2, f"Edit Mode: {mode}")
        # Sync backend hex_mode when switching Hex1/Hex2
        if mode == 'Hex1':
            self.backend.hex_mode = 'Hex1'
        elif mode == 'Hex2':
            self.backend.hex_mode = 'Hex2'
        # Auto-switch label mode to Pi Orbitals when in pi mode
        if mode == 'pi':
            self.set_label_mode('Pi Orbitals')
        # Scene and UI settings
        if mode == 'Select':
            self.scene.set_selection_mode(True)
            self.scene.lock_drag = False
            self.scene._link_mode = False
            self.ring_size_label.setVisible(False)
            self.ring_size_spinbox.setVisible(False)
            self.scene.ring_preview_line.visible = False
            self.statusBar().showMessage("LMB: Select/Deselect | RMB: Delete | Scroll: Zoom")
        elif mode == 'Bond':
            self.scene.set_selection_mode(False)
            self.scene.lock_drag = True   # No atom dragging in bond mode
            self.scene._link_mode = False
            self.ring_size_label.setVisible(False)
            self.ring_size_spinbox.setVisible(False)
            self.scene.ring_preview_line.visible = False
            self.statusBar().showMessage("LMB: Insert atom (Ctrl: push aside) | RMB: Delete bond (Ctrl: collapse) | Scroll: Zoom")
        elif mode == 'Ring':
            self.scene.set_selection_mode(False)
            self.scene.lock_drag = True   # No atom dragging in ring mode
            self.scene._link_mode = False
            self.ring_size_label.setVisible(True)
            self.ring_size_spinbox.setVisible(True)
            self.scene.ring_preview_line.visible = False
            self.statusBar().showMessage("LMB: Add n-gon ring on bond | RMB: Delete bond | Numpad +/-: Ring size | Scroll: Zoom")
        elif mode in ('Hex1', 'Hex2'):
            self.scene.set_selection_mode(False)
            self.scene.lock_drag = True   # No atom dragging in hex mode
            self.scene._link_mode = False
            self.ring_size_label.setVisible(False)
            self.ring_size_spinbox.setVisible(False)
            self.scene.ring_preview_line.visible = False
            mode_str = "Hex1 (paint: force add/remove)" if mode == 'Hex1' else "Hex2 (toggle: preserve shared)"
            self.statusBar().showMessage(f"{mode_str}: LMB: Add | RMB: Remove")
        else:
            self.scene.set_selection_mode(False)
            self.scene.lock_drag = False
            self.scene._link_mode = (mode == 'Atom')  # Enable Ctrl+drag bond creation in Atom mode
            self.ring_size_label.setVisible(False)
            self.ring_size_spinbox.setVisible(False)
            self.scene.ring_preview_line.visible = False
            if mode == 'Atom':
                self.statusBar().showMessage("LMB: Add/Change type/Drag move | Ctrl+LMB drag: Create bond | RMB: Delete (Ctrl: bridge neighbors) | Scroll: Zoom")
            else:
                self.statusBar().showMessage("LMB: Add/Toggle | RMB: Remove | Middle-Click: Toggle H | Scroll: Zoom")

    def set_atom_type(self, atype):
        self.cur_atom_type = atype
        debug_print(2, f"Atom Type: {atype}")

    def toggle_auto_h_cap(self):
        self.backend.auto_h_cap = self.auto_h_cap_btn.isChecked()
        debug_print(2, f"Auto H-cap: {self.backend.auto_h_cap}")

    def toggle_auto_recalc_bonds(self):
        self.backend.auto_recalc_bonds = self.auto_bonds_btn.isChecked()
        debug_print(2, f"Auto Recalc Bonds: {self.backend.auto_recalc_bonds}")

    def set_a_CC(self, value):
        self.backend.grid.a_CC = value
        self.backend.reassign_pins()
        self.refresh_view()
        debug_print(2, f"a_CC = {value}")

    def transpose_grid(self):
        self.backend.grid.toggle_transpose()
        self.grid_transpose_btn.setChecked(self.backend.grid.transpose)
        self.backend.reassign_pins()
        self.refresh_view()
        debug_print(2, f"Grid transpose: {self.backend.grid.transpose}")

    def set_grid_rotation(self, degrees):
        self.backend.grid.set_rotation(np.radians(degrees))
        self.backend.reassign_pins()
        self.refresh_view()
        debug_print(2, f"Grid rotation: {degrees}°")

    def toggle_offset_unit(self):
        grid_units = self.grid_offset_unit_chk.isChecked()
        a = self.backend.grid.a_CC
        if grid_units:
            self.grid_offset_x_spin.setSingleStep(1.0)
            self.grid_offset_y_spin.setSingleStep(1.0)
        else:
            self.grid_offset_x_spin.setSingleStep(0.1)
            self.grid_offset_y_spin.setSingleStep(0.1)
        debug_print(2, f"Offset unit: {'grid' if grid_units else 'Å'}")

    def _offset_to_angstrom(self, value):
        """Convert spinbox value to Å based on unit checkbox."""
        if self.grid_offset_unit_chk.isChecked():
            a = self.backend.grid.a_CC
            s3 = np.sqrt(3.0)
            return value * s3 * a  # grid vector along x = sqrt(3)*a_CC
        return value

    def set_grid_offset_x(self, value):
        self.backend.grid.set_offset(self._offset_to_angstrom(value), self.backend.grid.offset[1])
        self.backend.reassign_pins()
        self.refresh_view()

    def set_grid_offset_y(self, value):
        self.backend.grid.set_offset(self.backend.grid.offset[0], self._offset_to_angstrom(value))
        self.backend.reassign_pins()
        self.refresh_view()

    def reset_grid_transform(self):
        self.backend.grid.reset_transform()
        self.backend.reassign_pins()
        self.grid_rotate_spin.setValue(0.0)
        self.grid_offset_x_spin.setValue(0.0)
        self.grid_offset_y_spin.setValue(0.0)
        self.grid_transpose_btn.setChecked(False)
        self.a_CC_spin.setValue(1.42)
        self.refresh_view()
        debug_print(2, "Grid transform reset")

    def set_pick_radius(self, value):
        self.pick_radius = value
        self.scene.pick_radius = value
        debug_print(2, f"Pick Radius: {self.pick_radius}")

    def find_nearest_atom_index(self, pos, radius):
        """Find index of nearest atom within radius of pos."""
        if len(self.backend.sys.apos) == 0:
            return None
        apos = self.backend.sys.apos[:, :2]  # Only x,y for distance
        distances = np.linalg.norm(apos - pos[:2], axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] <= radius:
            return min_idx
        return None

    def find_nearest_atom_id(self, pos, radius):
        """Find Atom._id of nearest atom within radius of pos. O(1) via _atom_ids."""
        idx = self.find_nearest_atom_index(pos, radius)
        if idx is None:
            return None
        if hasattr(self.backend, '_atom_ids') and self.backend._atom_ids is not None:
            return int(self.backend._atom_ids[idx])
        return idx  # fallback to index if _atom_ids not available

    def set_label_mode(self, mode):
        """Set label display mode."""
        self.label_mode = mode
        # Update combo box to reflect current mode
        index = self.label_combo.findText(mode)
        if index >= 0:
            self.label_combo.blockSignals(True)
            self.label_combo.setCurrentIndex(index)
            self.label_combo.blockSignals(False)
        self.refresh_view()

    def toggle_bond_viz(self):
        """Toggle bond color visualization mode."""
        self.bond_viz_mode = not self.bond_viz_mode
        self.refresh_view()

    def toggle_debug_view(self):
        """Toggle debug visualization mode."""
        self.debug_view_mode = not self.debug_view_mode
        self.refresh_view()

    def on_selection_changed(self, selected_indices):
        """Handle selection change from Vispy scene."""
        n_selected = len(selected_indices)
        if n_selected > 0:
            self.statusBar().showMessage(f"Selected {n_selected} atoms | Delete: Remove | Ctrl-C: Copy | Ctrl-V: Paste | LMB: Drag selected")
        elif self.edit_mode == 'Select':
            self.statusBar().showMessage("Selection Mode: RMB drag to select | Delete: Remove | Ctrl-C: Copy | Ctrl-V: Paste | LMB: Drag selected")

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        selected = self.scene.get_selected_ids()

        # Check for Control modifier (Vispy uses tuple of strings)
        ctrl_pressed = 'Control' in event.modifiers if isinstance(event.modifiers, (tuple, list)) else False

        # Ctrl-V works even without selection (paste from clipboard)
        if event.key == 'V' and ctrl_pressed:
            self.paste_copied_atoms()
            return
        # Ctrl-Z works even without selection (undo)
        if event.key == 'Z' and ctrl_pressed:
            self.undo()
            return

        # Numpad +/- changes ring size in Ring mode (synced with spinbox)
        if self.edit_mode == 'Ring' and event.key in ('+', 'KP_ADD', '='):
            val = int(self.ring_size_spinbox.value()) + 1
            self.ring_size_spinbox.setValue(min(val, self.ring_size_spinbox.maximum()))
            return
        if self.edit_mode == 'Ring' and event.key in ('-', 'KP_SUBTRACT', '_'):
            val = int(self.ring_size_spinbox.value()) - 1
            self.ring_size_spinbox.setValue(max(val, self.ring_size_spinbox.minimum()))
            return

        if not selected:
            return

        # Delete key - remove selected atoms
        if event.key == 'Delete':
            self.delete_selected_atoms()
        # Ctrl-C - copy selected atoms
        elif event.key == 'C' and ctrl_pressed:
            self.copy_selected_atoms()

    def delete_selected_atoms(self):
        """Delete currently selected atoms by Atom._id."""
        ids = list(self.scene.get_selected_ids())
        if not ids:
            return
        self._push_undo()
        self.backend.remove_atoms_by_id(ids)
        self.scene.clear_selection()
        self.refresh_view()
        debug_print(2, f"Deleted {len(ids)} atoms")

    def copy_selected_atoms(self):
        """Copy selected atoms to internal PackedMolecule + Qt clipboard text."""
        ids = list(self.scene.get_selected_ids())
        if not ids:
            return
        # Map IDs to dense indices
        indices = []
        for aid in ids:
            idx = self.scene._id_to_idx_safe(aid)
            if idx >= 0:
                indices.append(idx)
        if not indices:
            return
        # Build PackedMolecule from selected atoms (includes internal bonds)
        self.copied_packed = PackedMolecule.from_graph(self.backend.graph, atom_indices=indices)
        # Also put text on Qt clipboard for external paste
        clip_text = self.copied_packed.to_mol_text() if len(self.copied_packed.bonds) > 0 else self.copied_packed.to_xyz_text()
        QtWidgets.QApplication.clipboard().setText(clip_text)
        debug_print(2, f"Copied {len(indices)} atoms ({len(self.copied_packed.bonds)} bonds) to clipboard")

    def paste_copied_atoms(self):
        """Paste atoms from internal PackedMolecule or parse Qt clipboard text."""
        packed = self.copied_packed
        if packed is None:
            # Try parsing Qt clipboard text as .xyz or .mol
            clip_text = QtWidgets.QApplication.clipboard().text()
            if clip_text.strip():
                packed = PackedMolecule.from_text(clip_text)
                if packed is not None:
                    debug_print(2, f"Parsed clipboard text: {packed}")
        if packed is None:
            debug_print(2, "No atoms to paste")
            return
        self._push_undo()
        # Add atoms from packed data to graph
        new_atom_ids = []
        new_atoms = []
        for i in range(len(packed.etype)):
            z = int(packed.etype[i])
            ename = _z_to_ename(z)
            npi_i = int(packed.npi[i]) if i < len(packed.npi) else 1
            pos = list(packed.apos[i].copy())
            a = self.backend._append_atom(pos=pos, ename=ename, pin=None, parent=None, npi=npi_i)
            new_atom_ids.append(a._id)
            new_atoms.append(a)
        # Add internal bonds from packed data
        for col in packed.bonds:
            i, j = int(col[0]), int(col[1])
            if 0 <= i < len(new_atoms) and 0 <= j < len(new_atoms):
                self.backend.graph.add_bond(new_atoms[i], new_atoms[j])
        self.backend.graph.sync_neighbor_lists()
        self.refresh_view()
        self.scene.set_selected_ids(new_atom_ids)
        debug_print(2, f"Pasted {len(new_atoms)} atoms ({len(packed.bonds)} bonds)")

    def _push_undo(self):
        """Push current graph state to undo stack before a mutation."""
        if not self.undo_enabled: return
        self.undo_stack.push(PackedMolecule.from_graph(self.backend.graph))

    def undo(self):
        """Undo last graph mutation by restoring from UndoStack."""
        packed = self.undo_stack.pop()
        if packed is None:
            debug_print(2, "Nothing to undo")
            return
        self.backend.graph = packed.to_graph()
        self.backend._sync_sys()
        self.refresh_view()
        debug_print(2, f"Undo: restored {len(packed.etype)} atoms")

    def on_drag_state(self, state, atom_id, pos):
        """Handle drag state changes from scene.
        
        Args:
            state: 1 = drag start, 0 = drag end
            atom_id: Atom._id being dragged
            pos: position of dragged atom
        
        On drag end (state=0), sync scene positions back to AtomicGraph (authoritative)
        and then to sys.apos via _sync_sys(). This ensures all geometry sources stay in sync.
        Then refresh view to update bond visualization immediately.
        """
        if state == 0:  # Drag end
            # Update AtomicGraph atom positions from scene._pos
            atom_list, enames, apos, atypes, bonds, bond_list, ring_list = self.backend.graph.to_arrays()
            scene_pos = self.scene._pos
            if len(atom_list) != len(scene_pos):
                debug_print(1, f"WARNING: on_drag_state: atom count mismatch {len(atom_list)} vs {len(scene_pos)}")
                return
            for i, atom in enumerate(atom_list):
                atom.pos[:] = scene_pos[i]

            # Check for atom merge: if dragged atom overlaps another heavy atom
            if atom_id >= 0:
                dragged = self.backend.graph.atoms.get(atom_id)
                if dragged is not None and dragged.alive and dragged.npi != -1:
                    # Find nearest other heavy atom (excluding dragged itself)
                    drag_pos = dragged.pos[:2]
                    nearest_other = None
                    nearest_dist = self.pick_radius
                    for a in self.backend.graph.atoms.values():
                        if a is dragged or not a.alive or a.npi == -1: continue
                        dist = np.linalg.norm(a.pos[:2] - drag_pos)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_other = a
                    if nearest_other is not None:
                        debug_print(2, f"Drag end: merge Atom({atom_id}) onto Atom({nearest_other._id}) (dist={nearest_dist:.3f})")
                        self._push_undo()
                        self.backend.merge_atoms(atom_id, nearest_other._id)
                        self.refresh_view()
                        self.sig_geometry_changed.emit()
                        return

            # Sync sys.apos from AtomicGraph (now authoritative)
            self.backend._sync_sys()
            # Update scene's internal _pos array from sys.apos to keep in sync
            self.scene.update_positions(self.backend.sys.apos.astype(np.float32))
            # Refresh view to update bond visualization immediately
            self.refresh_view()
            self.sig_geometry_changed.emit()
            debug_print(2, f"Drag end: synced {len(atom_list)} atom positions to graph and sys")

    def on_mouse_move(self, event):
        """Update cursor cross position on mouse move and highlight atom/bond/ring on hover."""
        r0, rd = self.scene._ray_from_mouse(event.pos)
        p_world = self.scene._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
        if p_world is not None:
            self.cursor_markers.set_data(pos=np.array([p_world]),symbol='cross',edge_width=2,edge_color='red',face_color='transparent',size=10 )

            # Detect geometry rings before picking
            self.backend.detect_geometry_rings()

            # Picking: atom, bond, ring (in priority order)
            hovered_atom = self.backend.pick_atom(p_world, radius=self.pick_radius)
            hovered_bond = self.backend.pick_bond(p_world, radius=0.5)
            hovered_ring = self.backend.pick_ring(p_world, radius=1.0)

            # Clear hover visuals
            self.scene.hover_bond_line.set_data(pos=np.zeros((0,3)))
            self.scene.hover_ring_lines.set_data(pos=np.zeros((0,3)))
            self.scene.hover_ring_markers.set_data(pos=np.zeros((0,3)))
            self.scene.hover_ring_text.text = ''
            self.scene.hover_atom_marker.set_data(pos=np.zeros((0,3)))

            # Mode-specific hover highlighting
            if self.edit_mode in ('Atom', 'pi', 'Select'):
                # Atom modes: highlight atoms only
                if hovered_atom:
                    self.scene.hover_atom_marker.set_data(
                        pos=np.array([hovered_atom.pos], dtype=np.float32),
                        symbol='disc', edge_width=3, edge_color='yellow', 
                        face_color='transparent', size=20
                    )
                    debug_print(3, f"Hovered atom: {hovered_atom}")
            elif self.edit_mode in ('Bond', 'Ring'):
                # Bond/Ring modes: highlight bonds only
                if hovered_bond:
                    pos_a = hovered_bond.a.pos
                    pos_b = hovered_bond.b.pos
                    self.scene.hover_bond_line.set_data(pos=np.array([pos_a, pos_b], dtype=np.float32))
                    debug_print(3, f"Hovered bond: {hovered_bond}")
                # Ring mode: show ghost n-gon preview on mouse side of bond
                if self.edit_mode == 'Ring' and hovered_bond:
                    n = int(self.ring_size_spinbox.value())
                    side = self.backend.compute_ring_side(hovered_bond, p_world)
                    verts = self.backend.compute_adjacent_ring_positions(hovered_bond, n, side)
                    # Close the loop for visualization
                    closed = np.vstack([verts, verts[:1]]).astype(np.float32)
                    self.scene.ring_preview_line.set_data(pos=closed, color=(0.2, 0.8, 0.8, 0.6))
                    self.scene.ring_preview_line.visible = True
                else:
                    self.scene.ring_preview_line.visible = False
            elif self.edit_mode in ('Hex1', 'Hex2'):
                # Hex modes: highlight rings only (existing hex highlighting below)
                pass
            else:
                # Other modes: no hover highlighting
                pass

            # Highlight hovered ring (polygon + CoG lines + atom count) - only in hex modes
            if self.edit_mode in ('Hex1', 'Hex2') and hovered_ring:
                # Draw polygon around ring
                ring_pos = np.array([a.pos for a in hovered_ring.atoms] + [hovered_ring.atoms[0].pos], dtype=np.float32)
                self.scene.hover_ring_lines.set_data(pos=ring_pos)
                # Draw lines from CoG to each atom
                cog_lines = []
                for atom in hovered_ring.atoms:
                    cog_lines.append(hovered_ring.cog)
                    cog_lines.append(atom.pos)
                self.scene.hover_ring_markers.set_data(pos=np.array(cog_lines, dtype=np.float32))
                # Show atom count at CoG
                self.scene.hover_ring_text.pos = hovered_ring.cog
                self.scene.hover_ring_text.text = str(len(hovered_ring.atoms))
                debug_print(3, f"Hovered ring: {hovered_ring} (n={len(hovered_ring.atoms)})")

            # Highlight hexagon under mouse if in hex mode
            if self.edit_mode in ('Hex1', 'Hex2') and hasattr(self.backend, 'snap_to_ring'):
                from spammm.topology.HexGrid import snap_to_grid
                q, r = self.backend.snap_to_ring(p_world[0], p_world[1])
                ring_nodes = self.backend.grid.ring_nodes(q, r)
                hover_pos = []
                for node in ring_nodes:
                    nk = snap_to_grid(node)
                    hover_pos.append([nk[0], nk[1], -0.08])
                if hover_pos:
                    self.hover_markers.set_data(
                        pos=np.array(hover_pos, dtype=np.float32),
                        symbol='disc', edge_width=2, edge_color='orange', face_color='transparent', size=12
                    )
                    self.hover_markers.visible = True
                else:
                    self.hover_markers.visible = False
            else:
                self.hover_markers.visible = False

    def on_atom_remove(self, atom_id):
        """Remove atom by Atom._id and refresh view. Only in Atom/pi/Select modes."""
        if self.edit_mode in ('Hex1', 'Hex2', 'Bond'):
            return   # In Hex/Bond modes, RMB is handled by handle_click (hex removal / bond collapse)
        debug_print(2, f"[RMB_REMOVE] atom_id={atom_id} ctrl={self.scene._last_ctrl}")
        self._push_undo()
        if self.scene._last_ctrl:
            self.backend.remove_atom_with_bridge(atom_id)
        else:
            self.backend.remove_atom_by_id(atom_id)
        self.refresh_view()
        self.sig_geometry_changed.emit()

    def on_link_bond(self, from_id, to_id):
        """Create bond between two atoms (Ctrl+drag in Atom mode)."""
        debug_print(2, f"[LINK_BOND] from={from_id} to={to_id}")
        self._push_undo()
        a = self.backend.graph.atoms.get(from_id)
        b = self.backend.graph.atoms.get(to_id)
        if a is not None and b is not None and a.alive and b.alive:
            self.backend.graph.add_bond(a, b)  # local neighbor update in add_bond
            if self.backend.auto_h_cap:
                self.backend.adjust_h()
            self.backend._sync_sys()
        self.refresh_view()
        self.sig_geometry_changed.emit()

    def on_atom_clicked(self, atom_id):
        """Handle atom click without drag in Atom mode (change atom type)."""
        debug_print(2, f"[ATOM_CLICKED] atom_id={atom_id}")
        self._push_undo()
        self.backend.set_atom_type_by_id(atom_id, self.cur_atom_type)
        self.refresh_view()
        self.sig_geometry_changed.emit()

    def on_mouse_press(self, event):
        # In Select mode, let Vispy handle everything (RMB selection, LMB drag)
        if self.edit_mode == 'Select':
            return

        # In Hex/Ring/Bond modes, prevent dragging - we only want add/remove operations
        if self.edit_mode in ('Hex1', 'Hex2', 'Ring', 'Bond'):
            # Continue to handle_click for operations, but don't let Vispy handle drag
            pass
        else:
            # For non-ring modes, if atoms are selected and LMB, let Vispy handle dragging
            selected = self.scene.get_selected_ids()
            if selected and event.button == 1:
                return

        # If atom picked and LMB in atom/pi mode, handle atom change instead of drag
        picked = self.scene._pick_idx
        picked_id = self.scene._pick_id
        if picked >= 0 and event.button == 1:
            if self.edit_mode == 'Atom':
                # Change atom type by Atom._id
                self._push_undo()
                self.backend.set_atom_type_by_id(picked_id, self.cur_atom_type)
                self.refresh_view()
                return
            elif self.edit_mode == 'pi':
                # Cycle pi orbitals on picked atom: sp3(0) -> sp2(1) -> sp(2) -> sp3(0)
                self._push_undo()
                current_npi = self.backend.atom_npi[picked]
                new_npi = (current_npi + 1) % 3
                self.backend.set_atom_npi_by_id(picked_id, new_npi)
                if self.backend.auto_h_cap:
                    self.backend.adjust_h()
                self.refresh_view()
                return
            elif self.edit_mode == 'Ring':
                # In ring mode, don't allow dragging atoms - ignore atom picks
                return

        ctrl_pressed = 'Control' in event.modifiers if isinstance(event.modifiers, (tuple, list)) else False
        if event.button == 1: # LMB
            self.handle_click(event.pos, action='add', ctrl=ctrl_pressed)
        elif event.button == 2: # RMB
            # Atom/pi/Select modes rely on sig_rmb_remove from VispyUtils;
            # calling handle_click here too would double-fire removal
            if self.edit_mode not in ('Atom', 'pi', 'Select'):
                self.handle_click(event.pos, action='remove', ctrl=ctrl_pressed)
        elif event.button == 3: # Middle / Scroll click
            self.handle_click(event.pos, action='toggle_h')

    def reset_offsets(self):
        self._push_undo()
        self.backend.snap_atoms_to_grid()
        self.refresh_view()

    def show_xyz(self):
        """Show current structure in a text dialog."""
        xyz_str = self.backend.get_xyz_string()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Current XYZ Structure")
        layout = QtWidgets.QVBoxLayout(dialog)
        self.textEdit(xyz_str, read_only=True, min_size=(400, 500), layout=layout, plain=True)
        self.button("Close", dialog.accept, layout=layout)
        dialog.exec_()

    def export_structure(self):
        """Export current structure to .xyz, .mol, or .mol2 file."""
        fname = self.fileDialog(mode="save", title="Export Structure",
                                filter_str="Molecular Files (*.xyz *.mol *.mol2);;XYZ (*.xyz);;MOL (*.mol);;MOL2 (*.mol2)")
        if fname:
            self.backend.save_structure(fname)
            self.statusBar().showMessage(f"Exported to {fname}")
            debug_print(2, f"Exported to {fname}")

    def import_structure(self):
        """Import structure from .xyz, .mol, or .mol2 file."""
        fname = self.fileDialog(mode="open", title="Import Structure",
                                filter_str="Molecular Files (*.xyz *.mol *.mol2);;XYZ (*.xyz);;MOL (*.mol);;MOL2 (*.mol2)")
        if fname:
            self.backend.load_structure(fname)
            self.refresh_view()
            self.statusBar().showMessage(f"Imported from {fname}")
            debug_print(2, f"Imported from {fname}")

    def adjust_h(self):
        """Manually trigger H passivation."""
        self._push_undo()
        self.backend.adjust_h()
        self.refresh_view()

    def recalc_bonds(self):
        """Manually trigger bond recalculation and refresh view.
        
        Removes H caps before recalc, then adds H caps based on new topology.
        """
        self._push_undo()
        # Step 1: Remove all H caps
        self.backend.remove_h_caps()
        
        # Step 2: Recalculate bonds from distance
        self.backend.recalc_bonds()
        
        # Step 3: Add H caps based on new topology (if auto-H is enabled)
        if self.backend.auto_h_cap:
            self.backend.add_h_caps()
        
        self.refresh_view()

    def handle_click(self, mouse_pos, action='add', ctrl=False):
        self._push_undo()
        # 1. Get world coordinates on z=0 plane
        # Vispy mouse pos is (x, y) from top-left
        # We need to use Vispy's internal ray casting
        # ray = self.scene._ray_from_mouse(mouse_pos)
        # intersect = self.scene._intersect_ray_plane(ray[0], ray[1], np.zeros(3), np.array([0,0,1]))
        
        # Helper to get world pos
        r0, rd = self.scene._ray_from_mouse(mouse_pos)
        p_world = self.scene._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0,0,1]))
        
        if p_world is None: return
        x, y = p_world[0], p_world[1]
        pos_2d = np.array([x, y])

        # 2. Mode-specific target resolution
        if self.edit_mode in ('Hex1', 'Hex2'):
            # Hex modes: use grid snapping for ring placement
            q, r = self.backend.snap_to_ring(x, y)
            nearest_atom_idx = None
            nearest_atom_id = None
        elif self.edit_mode in ('Bond', 'Ring'):
            # Bond/Ring modes: pick bond
            bond = self.backend.pick_bond(p_world)
            nearest_atom_idx = None
            nearest_atom_id = None
            q, r = (None, None)
        else:
            # Atom/pi/Select modes: position-based atom picking, no grid snapping
            nearest_atom_idx = self.find_nearest_atom_index(p_world, self.pick_radius)
            nearest_atom_id = self.find_nearest_atom_id(p_world, self.pick_radius)
            q, r = (None, None)
            debug_print(2, f"Click at ({x:.2f}, {y:.2f}) -> Mode={self.edit_mode} AtomIdx={nearest_atom_idx} | Action: {action}")

        # 3. Modify backend
        if action == 'add':
            if self.edit_mode in ('Hex1', 'Hex2'):
                if q is not None and r is not None:
                    self.backend.add_ring(q, r)
            elif self.edit_mode == 'Bond':
                if bond is not None:
                    new_atom = self.backend.insert_atom_into_bond(bond, self.cur_atom_type, push_aside=ctrl)
                    debug_print(2, f"Inserted atom into bond {bond._id}, new atom {new_atom._id} (push_aside={ctrl})")
            elif self.edit_mode == 'Ring':
                if bond is not None:
                    n = int(self.ring_size_spinbox.value())
                    side = self.backend.compute_ring_side(bond, p_world)
                    new_atoms = self.backend.add_adjacent_ring(bond, n_members=n, ename=self.cur_atom_type, side=side)
                    debug_print(2, f"Added {n}-ring on bond {bond._id} side={side}, {len(new_atoms)} new atoms")
            elif self.edit_mode == 'pi':
                # Cycle pi orbitals: sp3(0) -> sp2(1) -> sp(2) -> sp3(0)
                if nearest_atom_idx is not None:
                    current_npi = self.backend.atom_npi[nearest_atom_idx]
                    new_npi = (current_npi + 1) % 3
                    self.backend.set_atom_npi_by_id(nearest_atom_id, new_npi)
                    if self.backend.auto_h_cap:
                        self.backend.adjust_h()
                    self.refresh_view()
                    return
            elif nearest_atom_idx is not None:
                # Change atom type by Atom._id
                self.backend.set_atom_type_by_id(nearest_atom_id, self.cur_atom_type)
            else:
                # Free placement at exact position
                self.backend._append_atom(pos=[x, y, 0.0], ename=self.cur_atom_type, pin=None, parent=None, npi=self.backend._get_element_default_npi(self.cur_atom_type))
                atom_list, *_ = self.backend.graph.to_arrays()
                if atom_list:
                    new_atom = atom_list[-1]
                    self.backend._create_bond_to_nearest_heavy(new_atom)
                    self.backend.graph.sync_neighbor_lists()
                if self.backend.auto_h_cap:
                    self.backend.adjust_h()
                self.backend._sync_sys()
        elif action == 'remove':
            if self.edit_mode in ('Hex1', 'Hex2'):
                if q is not None and r is not None:
                    self.backend.remove_ring(q, r)
            elif self.edit_mode in ('Bond', 'Ring'):
                if bond is not None:
                    if ctrl and self.edit_mode == 'Bond':
                        survivor = self.backend.collapse_bond(bond, np.array([x, y]))
                        debug_print(2, f"Collapsed bond {bond._id}, survivor atom {survivor._id}")
                    else:
                        self.backend.delete_bond(bond)
                        debug_print(2, f"Deleted bond {bond._id}")
            elif nearest_atom_idx is not None:
                self.backend.remove_atom_by_id(nearest_atom_id)
        elif action == 'toggle_h':
            if nearest_atom_idx is not None:
                self.backend.toggle_h_state(nearest_atom_idx)
        
        self.refresh_view()
        self.sig_geometry_changed.emit()

    def refresh_view(self):
        # 0. Update Guide Grid
        guides = self.backend.get_guide_points()
        self.grid_markers.set_data(
            pos=np.column_stack([guides, np.full(len(guides), -0.1)]).astype(np.float32),
            symbol='disc', edge_width=0, size=2,
            face_color=(0.3, 0.3, 0.3, 0.3)
        )

        # 0.5. Debug view: for each atom, draw its pin node (cyan disc) and a line atom->pin
        if self.debug_view_mode and hasattr(self.backend, 'atom_pin'):
            pin_pos = []
            line_segs = []
            for ia, pin in enumerate(self.backend.atom_pin):
                if pin is not None and ia < len(self.backend.sys.apos):
                    atom_pos = self.backend.sys.apos[ia]
                    pin_pos.append([pin[0], pin[1], 0.05])          # pin node in z=0 plane
                    line_segs.append([pin[0], pin[1], 0.05])        # line: pin -> atom
                    line_segs.append([atom_pos[0], atom_pos[1], atom_pos[2]])
            if pin_pos:
                self.debug_markers.set_data(
                    pos=np.array(pin_pos, dtype=np.float32),
                    symbol='disc', edge_width=0, face_color=(0.0, 1.0, 1.0, 0.7), size=6
                )
                self.debug_markers.visible = True
                segs = np.array(line_segs, dtype=np.float32)
                conn = np.zeros(len(segs), dtype=bool); conn[0::2] = True  # isolated pairs
                self.debug_lines.set_data(pos=segs, connect=conn, color=(0.0, 1.0, 1.0, 0.6), width=1.5)
                self.debug_lines.visible = True
            else:
                self.debug_markers.visible = False
                self.debug_lines.visible = False
        else:
            self.debug_markers.visible = False
            self.debug_lines.visible = False

        # 1. Use persistent sys directly
        sys = self.backend.sys
        pos = sys.apos.astype(np.float32)

        if pos.size == 0:
            self.scene.set_data(np.zeros((0,3)))
            return

        # Colors based on elements
        colors = []
        sizes = []
        for e in sys.enames:
            c = elements.getColor(e)
            if e == 'H':
                colors.append((0.4, 0.4, 0.4, 1.0))
                sizes.append(8.0)
            else:
                colors.append((c[0], c[1], c[2], 1.0))
                sizes.append(15.0)
        
        colors = np.array(colors, dtype=np.float32)
        sizes = np.array(sizes, dtype=np.float32)
        
        # Bonds
        if sys.bonds is not None:
            is_heavy = np.array([sys.enames[i] != 'H' for i in range(len(sys.enames))])
            bonds_heavy = []
            bonds_h = []
            
            for b in sys.bonds:
                if is_heavy[b[0]] and is_heavy[b[1]]:
                    bonds_heavy.append(b)
                else:
                    bonds_h.append(b)
            
            # Bond color visualization mode
            if self.bond_viz_mode and bonds_heavy:
                bond_segs, bond_colors = compute_bond_colors_by_length(bonds_heavy, pos)
                self.scene._line_set("bonds-colored", self.scene.bond_colored_lines, bond_segs, color=bond_colors, width=5.0)
                self.scene.bond_colored_lines.visible = True
                self.scene.bond_lines.visible = False
                self.scene.set_data(pos, colors=colors, sizes=sizes, bonds=None)
            elif self.bond_viz_mode:
                # Bond viz mode but no bonds - hide both bond visuals
                self.scene.bond_colored_lines.visible = False
                self.scene.bond_lines.visible = False
                self.scene.set_data(pos, colors=colors, sizes=sizes, bonds=None)
            else:
                # Normal bond rendering - use standard bond_lines
                self.scene.bond_colored_lines.visible = False
                self.scene.bond_lines.visible = True
                self.scene.set_data(pos, colors=colors, sizes=sizes, bonds=bonds_heavy)
            
            if bonds_h:
                h_segs = pos[np.array(bonds_h)].reshape(-1, 3)
                self.scene._line_set("CH-bonds", self.scene.ch_bond_lines, h_segs, color=(0.4, 0.4, 0.4, 0.6), width=1.0)
            else:
                self.scene.ch_bond_lines.set_data(np.zeros((0, 3), dtype=np.float32))

            hbonds = sys.find_hbonds(bPrint=False)
            if hbonds:
                hb_segs = []
                for d, h, a, dist, ang in hbonds:
                    hb_segs.append(pos[h])
                    hb_segs.append(pos[a])
                hb_segs = np.array(hb_segs, dtype=np.float32)
                self.scene._line_set("H-bonds", self.scene.hbond_lines, hb_segs, color=(0.8, 0.2, 0.8, 0.5), width=1.5)
            else:
                self.scene.hbond_lines.set_data(np.zeros((0, 3), dtype=np.float32))

            # Bond orders (from Kekule solver)
            if self.bond_orders is not None and self.bond_order_bonds is not None:
                self.scene.set_bond_orders(self.bond_order_bonds, self.bond_orders, show_labels=self.show_bond_order_labels)
            else:
                self.scene.set_bond_orders(None, None)

            # Labels based on label_mode
            lbl_pos, lbl_texts = generate_atom_labels(self.label_mode, pos, sys.enames, self.backend.atom_npi, self.backend, bonds_heavy)
            if lbl_pos:
                self.scene.text_labels.text = lbl_texts
                self.scene.text_labels.pos = np.array(lbl_pos, dtype=np.float32)
                self.scene.text_labels.color = np.array([(0, 0, 0, 1)] * len(lbl_texts), dtype=np.float32)
                self.scene.text_labels.visible = True
            else:
                self.scene.text_labels.visible = False
        
        # Force immediate canvas update to avoid async rendering lag
        self.scene.canvas.update()
        QtWidgets.QApplication.processEvents()

    def run_relaxation(self):
        self.dftb_status_label.setText("Status: Relaxing...")
        self.statusBar().showMessage("Relaxing... please wait")
        QtWidgets.QApplication.processEvents()

        try:
            E, forces, lvs = self.backend.run_relaxation(workdir='gui_relax')
            msg = f"Relaxation done. E = {E:.4f} eV"
            self.statusBar().showMessage(msg)
            self.dftb_status_label.setText(f"Status: Done\nE = {E:.4f} eV")
            self.refresh_view()
        except Exception as e:
            msg = f"Relaxation FAILED: {e}"
            self.statusBar().showMessage(msg)
            self.dftb_status_label.setText(f"Status: FAILED\n{e}")
            self._raise(msg, title="Relaxation Error")

    def compute_orbitals(self):
        if len(self.backend.sys.apos) == 0:
            msg = "No atoms to compute orbitals for."
            debug_print(1, f"WARNING: {msg}")
            QtWidgets.QMessageBox.warning(self, "Warning", msg)
            return

        # Setup Fdata symlink (needed by FireCore)
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        FDATA_DIR = os.path.join(_THIS_DIR, "Fdata")
        FDATA_TARGET = self.fdata_path

        if not os.path.exists(FDATA_DIR):
            if os.path.exists(FDATA_TARGET):
                os.symlink(FDATA_TARGET, FDATA_DIR)
                debug_print(1, f"Created symlink: {FDATA_DIR} -> {FDATA_TARGET}")
            else:
                msg = f"Neither {FDATA_DIR} nor {FDATA_TARGET} exists. Please download Fdata_HC_minimal from fireball-qmd.github.io"
                print(f"ERROR: {msg}")
                QtWidgets.QMessageBox.critical(self, "Fdata Error", msg)
                return

        self.statusBar().showMessage("Running FireCore SCF...")
        QtWidgets.QApplication.processEvents()
        try:
            atypes = np.array([elements.ELEMENT_DICT[e][0] for e in self.backend.sys.enames], dtype=np.int32)
            apos = np.array(self.backend.sys.apos, dtype=np.float64)
            fc = self.extensions.require('firecore')
            fc.setVerbosity(0)
            fc.initialize(atomType=atypes, atomPos=apos)
            fc.evalForce(apos, nmax_scf=200)
            dims = fc.get_HS_dims()
            norb = int(dims.norbitals)
            self._eigen = fc.get_eigen(ikp=1, norb=norb)
            self._wfcoef = fc.get_wfcoef(norb=norb)
            self._norb = norb
            occ = np.where(self._eigen < 0.0)[0]
            self._homo = int(occ[-1]) if len(occ) > 0 else len(self._eigen) // 2 - 1
            self._lumo = self._homo + 1

            # Calculate total valence electrons
            valence_dict = {'H': 1, 'C': 4, 'N': 5, 'O': 6}
            total_electrons = sum([valence_dict.get(e, 0) for e in self.backend.sys.enames])
            occupied_orbitals = total_electrons // 2

            # Update orbital info label
            info_text = (f"Total Orbitals: {norb}\n"
                        f"HOMO: {self._homo + 1} (E={self._eigen[self._homo]:.3f} eV)\n"
                        f"LUMO: {self._lumo + 1} (E={self._eigen[self._lumo]:.3f} eV)\n"
                        f"Occupied: {occupied_orbitals} (e-/2)")
            self.orbital_info_label.setText(info_text)

            # Enable orbital controls
            self.orbital_spinbox.setEnabled(True)
            self.setSpinBox(self.orbital_spinbox, vmin=0, vmax=norb-1, value=self._homo)
            self.plot_orb_btn.setEnabled(True)
            self.plot_density_btn.setEnabled(True)
            self.plot_delta_btn.setEnabled(True)

            # Update orbital energy label for current selection
            self.update_orbital_energy_label(self._homo)

            msg = f"SCF done. HOMO={self._homo + 1} E={self._eigen[self._homo]:.3f} eV  LUMO={self._lumo + 1} E={self._eigen[self._lumo]:.3f} eV"
            debug_print(1, msg)
            self.statusBar().showMessage(msg)
            QtWidgets.QMessageBox.information(self, "SCF Done", f"HOMO={self._homo + 1} E={self._eigen[self._homo]:.3f} eV\nLUMO={self._lumo + 1} E={self._eigen[self._lumo]:.3f} eV")
        except Exception as e:
            self._raise(f"SCF FAILED: {e}", title="SCF Error")

    def _compute_extent_from_geometry(self, apos, padding_factor=0.1, default_size=14.0):
        """Compute grid extent and origin from atomic positions."""
        if len(apos) > 0:
            apos_2d = apos[:, :2]  # Only x,y
            min_pos = apos_2d.min(axis=0)
            max_pos = apos_2d.max(axis=0)
            center_z = apos[:, 2].mean()
            # Add padding
            padding = (max_pos - min_pos) * padding_factor
            grid_origin = min_pos - padding
            size = (max_pos - min_pos + 2 * padding).max()
        else:
            # No atoms - cannot compute extent from geometry
            # This should not happen in normal usage (plotting requires atoms)
            raise ValueError("Cannot compute grid extent: no atoms in system. Add atoms first.")
        return grid_origin, size, center_z

    def _make_2d_grid(self, grid_origin, size, center_z, z_height, n=100):
        """Generate 2D grid points for projection."""
        xs = np.linspace(grid_origin[0], grid_origin[0] + size, n)
        ys = np.linspace(grid_origin[1], grid_origin[1] + size, n)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros_like(X) + (center_z + z_height)
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        extent = [grid_origin[0], grid_origin[0] + size, grid_origin[1], grid_origin[1] + size]
        return points, extent, n

    def _evaluate_on_grid(self, points, what, orb_index=None):
        """Call Fortran evaluator on grid points.
        
        Args:
            points: (npoints, 3) array of grid points
            what: 'orbital', 'density', or 'delta_rho'
            orb_index: orbital index (1-based, required for 'orbital')
        
        Returns:
            flat array of evaluated values
        """
        fc = self.extensions.require('firecore')
        points_f64 = points.astype(np.float64)
        if what == 'orbital':
            if orb_index is None:
                raise ValueError("orb_index required for orbital evaluation")
            return fc.orb2points(points_f64, iMO=int(orb_index + 1), ikpoint=1)
        elif what == 'density':
            return fc.dens2points(points_f64, f_den=1.0, f_den0=0.0)
        elif what == 'delta_rho':
            return fc.dens2points(points_f64, f_den=1.0, f_den0=-1.0)
        else:
            raise ValueError(f"Unknown what={what}, expected 'orbital', 'density', or 'delta_rho'")

    def _plot_2d_projection(self, data_2d, extent, title, cmap, symmetric, atom_pos, atom_types):
        """Plot 2D projection using VisPy heatmap."""
        canvas, view = vu.create_heatmap_window(data_2d, extent, title=title,  cmap=cmap,  symmetric=symmetric, atom_pos=atom_pos, atom_types=atom_types )
        return canvas, view

    def plot_density(self):
        if not hasattr(self, '_eigen'):
            msg = "Please run Compute SCF first."
            debug_print(1, f"INFO: {msg}")
            QtWidgets.QMessageBox.information(self, "Info", msg)
            return
        self.statusBar().showMessage("Computing electron density...")
        QtWidgets.QApplication.processEvents()
        try:
            apos = self.backend.sys.apos
            z_height = self.z_height_spinbox.value()
            grid_origin, size, center_z = self._compute_extent_from_geometry(apos)
            points, extent, n = self._make_2d_grid(grid_origin, size, center_z, z_height)
            flat_data = self._evaluate_on_grid(points, 'density')
            data_2d = np.asarray(flat_data, dtype=np.float64).reshape(n, n)
            pos = apos.astype(np.float32)
            enames = list(self.backend.sys.enames)
            self._plot_2d_projection( data_2d, extent,title=f"Electron Density (z={z_height:.1f}Å)",cmap='bwr',   symmetric=False,  atom_pos=pos, atom_types=enames  )
            msg = "Density plotted"
            debug_print(1, msg)
            self.statusBar().showMessage(msg)
        except Exception as e:
            self._raise(f"Density plot FAILED: {e}", title="Plot Error")

    def plot_delta_rho(self):
        if not hasattr(self, '_eigen'):
            msg = "Please run Compute SCF first."
            debug_print(1, f"INFO: {msg}")
            QtWidgets.QMessageBox.information(self, "Info", msg)
            return
        self.statusBar().showMessage("Computing delta-rho (rho_SCF - rho_NA)...")
        QtWidgets.QApplication.processEvents()
        try:
            apos = self.backend.sys.apos
            z_height = self.z_height_spinbox.value()
            grid_origin, size, center_z = self._compute_extent_from_geometry(apos)
            points, extent, n = self._make_2d_grid(grid_origin, size, center_z, z_height)
            flat_data = self._evaluate_on_grid(points, 'delta_rho')
            data_2d = np.asarray(flat_data, dtype=np.float64).reshape(n, n)
            pos = apos.astype(np.float32)
            enames = list(self.backend.sys.enames)
            self._plot_2d_projection(data_2d, extent, title=f"Delta-Rho (z={z_height:.1f}Å)", cmap='bwr', symmetric=True, atom_pos=pos, atom_types=enames)
            msg = "Delta-rho plotted"
            debug_print(1, msg)
            self.statusBar().showMessage(msg)
        except Exception as e:
            self._raise(f"Delta-rho plot FAILED: {e}", title="Plot Error")

    def set_fdata_path(self):
        """Open dialog to set Fdata path and save to settings."""
        selected = self.fileDialog(mode="directory", title="Select Fdata Directory", start_dir=self.fdata_path)
        if selected:
            self.fdata_path = selected
            self.settings.setValue("fdata_path", selected)
            self.extensions.set_config('firecore', 'fdata_dir', selected)
            self.extensions.save_config()
            debug_print(2, f"Set Fdata path to: {selected}")
            self.statusBar().showMessage(f"Fdata path set to: {selected}")
            QtWidgets.QMessageBox.information(self, "Settings Saved", f"Fdata path set to:\n{selected}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SPAMMM GUI — Molecular editor and AFM simulation')
    parser.add_argument('--output-dir', '-o', type=str, default=None, help='Directory for saved images (default: <repo>/output)')
    parser.add_argument('--fdata-path', '-f', type=str, default=None, help='Path to Fdata directory')
    parser.add_argument('--verbosity', '-v', type=int, default=None, choices=[0, 1, 2, 3], help='Verbosity level 0-3 (default: 2)')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = KekuleExplorerWindow(output_dir=args.output_dir, fdata_path=args.fdata_path, verbosity=args.verbosity)
    window.show()
    sys.exit(app.exec_())
