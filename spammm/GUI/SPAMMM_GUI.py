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
editing commands → MoleculeEditorBackend, AFM commands → AFMExtension, rendering → VispyUtils.

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
from spammm.GUI.EditModeHandlers import (
    EditModeHandler, UnifiedMode, AtomMode, PiMode, BondMode,
    RingMode, Hex1Mode, Hex2Mode, SelectMode,
)
from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
import spammm.topology.MoleculeEditorBackend as MEB
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

class SPAMMMWindow(BaseGUI):
    sig_geometry_changed = QtCore.pyqtSignal()  # Emitted whenever atom geometry changes

    def __init__(self, output_dir=None, fdata_path=None, verbosity=None, work_dir=None):
        super().__init__("SPAMMM")
        self.resize(1024, 768)

        self.extensions = ExtensionManager()
        self.backend = MoleculeEditorBackend()
        self.work_dir = work_dir or os.path.expanduser("~")
        self.cur_atom_type = 'C'
        self.edit_mode = 'Unified'  # 'Unified', 'Hex1' (paint), 'Hex2' (toggle), 'Atom', 'Bond', 'Ring', 'pi', 'Select'
        self.label_mode = 'Element+Index'
        self.pick_radius = 0.5  # Distance in Angstroms for atom picking (matches spinbox default)
        self.bond_orders = None  # pi bond orders array (set by KekuleExtension solver)
        self.bond_order_bonds = None  # (m,2) heavy-atom bond indices matching bond_orders
        self.show_bond_order_labels = False
        self.b2Dview = True  # True: top planar edit; False: ortho 3D inspect/edit-lite

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
        self.mode_handlers = {}
        self._init_mode_handlers()
        self.initUI()
        # Scene drag signal: update AtomicGraph and sys.apos after drag end
        self.scene.sig_drag_state.connect(self.on_drag_state)
        self.scene.sig_rmb_remove.connect(self.on_atom_remove)
        self.scene.sig_camera_changed.connect(self.refresh_view)
        self.scene.sig_link_bond.connect(self.on_link_bond)
        self.scene.sig_link_to_pos.connect(self.on_link_to_pos)
        self.scene.sig_atom_clicked.connect(self.on_atom_clicked)
        self.set_edit_mode('Unified')  # Sync scene flags with default mode
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
        
        # Create side panel content (inside a scroll area)
        side_content = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_content)
        side_layout.setSpacing(1)
        side_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add sections
        side_layout.addWidget(self.create_editors_section())
        side_layout.addWidget(self.create_grid_section())
        side_layout.addWidget(self.create_ribbon_section())
        self._build_extension_panels(side_layout)
        side_layout.addStretch()
        
        # Wrap in scroll area so expanding sections doesn't resize the window
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(side_content)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameStyle(QtWidgets.QFrame.NoFrame)
        
        # Add to main layout
        main_layout.addWidget(scroll)
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
        self.statusBar().showMessage("LMB: Add/Toggle | RMB: Remove | Enter: 2D/3D | Space: Run FF | Scroll: Zoom | Arrows: Pan/Rotate")
        self.scene.lock_drag = False  # Default mode is Unified, allow dragging
        self.scene.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.scene.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.scene.canvas.events.mouse_release.connect(self.on_mouse_release)
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
        self.apply_view_mode()

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
        self.mode_combo = self.comboBox(["Unified", "Hex1", "Hex2", "Atom", "Bond", "Ring", "pi", "Select"], self.set_edit_mode, layout=layout)
        
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

        # 2D / 3D view mode
        row_view = QtWidgets.QHBoxLayout()
        self.b2Dview_chk = QtWidgets.QCheckBox("2D view (planar edit)")
        self.b2Dview_chk.setChecked(True)
        self.b2Dview_chk.setToolTip("Checked: top-down hex/empty edit. Unchecked: ortho 3D (Enter toggles). Space = run/stop FF.")
        self.b2Dview_chk.toggled.connect(self._on_b2Dview_toggled)
        row_view.addWidget(self.b2Dview_chk)
        self.view_debug_chk = QtWidgets.QCheckBox("Ray debug")
        self.view_debug_chk.setChecked(False)
        self.view_debug_chk.setToolTip("Draw mouse ray + hit point (transform sanity)")
        self.view_debug_chk.toggled.connect(lambda c: self.scene.set_view_debug(c))
        row_view.addWidget(self.view_debug_chk)
        row_view.addStretch()
        layout.addLayout(row_view)
        
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
        self.grid_transpose_btn = self.button("T-Grid", self.transpose_grid_only, layout=row1)
        self.grid_transpose_btn.setCheckable(True)
        self.grid_transpose_btn.setChecked(self.backend.grid.transpose)
        self.button("T-All", self.transpose_grid, layout=row1)
        self.button("Flip X", self.flip_x_geometry, layout=row1)
        self.button("Flip Y", self.flip_y_geometry, layout=row1)
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
        self._extension_sections = {}  # extension key / panel title -> CollapsibleSection

        EXTENSION_TITLES = {'ff': 'Force Field', 'afm': 'AFM', 'dftb': 'DFTB', 'firecore': 'FireCore', 'qeq': 'QEq Charges', 'kekule': 'Kekule Solver', 'ascii': 'ASCII Builder', 'fragments': 'Fragments', 'reaction_coord': 'Reaction coordinate', 'vibrations': 'Vibrations', 'folded_rigid': 'Folded Rigid'}
        for name in self.extensions.enabled_extensions():
            ui = self.extensions.build_ui(name, self)
            title = EXTENSION_TITLES.get(name, name.capitalize())
            sec = CollapsibleSection(title, collapsed=True, parent=self)
            self._extension_sections[name] = sec
            self._extension_sections[title] = sec
            if ui.panel is not None:
                if ui.help_text:
                    wrapper = QtWidgets.QWidget()
                    w_layout = QtWidgets.QVBoxLayout(wrapper)
                    w_layout.setContentsMargins(0, 0, 0, 0)
                    w_layout.setSpacing(2)
                    help_btn = QtWidgets.QPushButton("?")
                    help_btn.setMaximumWidth(24)
                    help_btn.setToolTip("Show help")
                    help_btn.clicked.connect(lambda checked=False, n=name, t=ui.help_text: self._show_extension_help(n, t))
                    hbox = QtWidgets.QHBoxLayout()
                    hbox.addStretch()
                    hbox.addWidget(help_btn)
                    w_layout.addLayout(hbox)
                    w_layout.addWidget(ui.panel)
                    sec.setContent(wrapper)
                else:
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
                # Register a minimal handler — extension uses its own callback via _ext_edit_modes
                self.register_mode_handler(label, EditModeHandler(self))
            for label, cb in ui.view_modes:
                self._ext_view_modes[label] = cb

    def _show_extension_help(self, name, help_text):
        """Open a dialog with extension help text."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"{name} Help")
        layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        if isinstance(help_text, dict):
            lines = [f"{k}: {v}" for k, v in help_text.items()]
            text.setPlainText("\n".join(lines))
        else:
            text.setPlainText(str(help_text))
        layout.addWidget(text)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        dialog.resize(450, 350)
        dialog.show()

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
            
            self.backend = MoleculeEditorBackend()
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
            bottom_ribbon = MoleculeEditorBackend()
            bottom_ribbon.build_zigzag_ribbon(width_chains=width_chains1, length_cells=length_cells, passivation_bottom=bottom1_passivation, passivation_top=top1_passivation, scale_x=Lx / (2.0 * 1.42 * np.cos(np.pi / 6)), bPeriodicX=True)
            
            # Build top ribbon
            top_ribbon = MoleculeEditorBackend()
            top_ribbon.build_zigzag_ribbon(width_chains=width_chains2, length_cells=length_cells, passivation_bottom=bottom2_passivation, passivation_top=top2_passivation,  scale_x=Lx / (2.0 * 1.42 * np.cos(np.pi / 6)), bPeriodicX=True)
            
            # Combine ribbons
            self.backend = MoleculeEditorBackend()
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

    def _init_mode_handlers(self):
        """Instantiate handler objects for each built-in edit mode."""
        self.mode_handlers = {
            'Unified': UnifiedMode(self),
            'Atom':    AtomMode(self),
            'pi':      PiMode(self),
            'Bond':    BondMode(self),
            'Ring':    RingMode(self),
            'Hex1':    Hex1Mode(self),
            'Hex2':    Hex2Mode(self),
            'Select':  SelectMode(self),
        }

    def register_mode_handler(self, name, handler):
        """Register an EditModeHandler instance for an extension-defined edit mode."""
        self.mode_handlers[name] = handler

    # ── Common helpers (used by on_mouse_move preamble) ─────────────────────

    def _clear_hover(self):
        self.scene.hover_bond_line.set_data(pos=np.zeros((0,3)))
        self.scene.hover_ring_lines.set_data(pos=np.zeros((0,3)))
        self.scene.hover_ring_markers.set_data(pos=np.zeros((0,3)))
        self.scene.hover_ring_text.text = ''
        self.scene.hover_atom_marker.set_data(pos=np.zeros((0,3)))
        self.hover_markers.visible = False
        self.scene.ring_preview_line.visible = False

    def _add_free_atom(self, p_world):
        """Add a free atom at world position, bond to nearest heavy, sync."""
        debug_print(2, f"[ADD_FREE_ATOM] type={self.cur_atom_type} pos=({p_world[0]:.2f},{p_world[1]:.2f})")
        self._push_undo()
        self.backend._append_atom(pos=[p_world[0], p_world[1], 0.0], ename=self.cur_atom_type, pin=None, parent=None, npi=self.backend._get_element_default_npi(self.cur_atom_type))
        atom_list, *_ = self.backend.graph.to_arrays()
        if atom_list:
            new_atom = atom_list[-1]
            self.backend._create_bond_to_nearest_heavy(new_atom)
            self.backend.graph.sync_neighbor_lists()
        if self.backend.auto_h_cap:
            self.backend.adjust_h()
        self.backend._sync_sys()
        self.refresh_view()
        self.sig_geometry_changed.emit()

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
        h = self.mode_handlers.get(mode)
        if h is None: return
        self.scene.set_selection_mode(h.selection_mode)
        self.scene.lock_drag = h.lock_drag
        self.scene._link_mode = h.link_mode
        self.ring_size_label.setVisible(h.ring_size_visible)
        self.ring_size_spinbox.setVisible(h.ring_size_visible)
        self.scene.ring_preview_line.visible = False
        if h.status_msg:
            self.statusBar().showMessage(h.status_msg)
        h.on_activate()

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

    def transpose_grid_only(self):
        """Transpose grid axes only (swap X/Y), without moving atoms."""
        self._push_undo()
        self.backend.grid.toggle_transpose()
        self.grid_transpose_btn.setChecked(self.backend.grid.transpose)
        self.backend.reassign_pins()
        self.refresh_view()
        debug_print(2, f"Grid transpose (grid only): {self.backend.grid.transpose}")

    def transpose_grid(self):
        """Transpose grid axes AND transform atom geometry (swap X/Y for both)."""
        self._push_undo()
        self.backend.grid.toggle_transpose()
        self.grid_transpose_btn.setChecked(self.backend.grid.transpose)
        self.backend.transform_atoms('transpose')
        self.backend.reassign_pins()
        self.refresh_view()
        self.sig_geometry_changed.emit()
        debug_print(2, f"Grid transpose (grid+atoms): {self.backend.grid.transpose}")

    def flip_x_geometry(self):
        self._push_undo()
        self.backend.transform_atoms('flip_x')
        self.backend.reassign_pins()
        self.refresh_view()
        self.sig_geometry_changed.emit()
        debug_print(2, "Flip X geometry")

    def flip_y_geometry(self):
        self._push_undo()
        self.backend.transform_atoms('flip_y')
        self.backend.reassign_pins()
        self.refresh_view()
        self.sig_geometry_changed.emit()
        debug_print(2, "Flip Y geometry")

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

    def _on_b2Dview_toggled(self, checked):
        self.b2Dview = bool(checked)
        self.apply_view_mode()

    def apply_view_mode(self):
        """Sync camera/pick/depth with b2Dview. Default True = current planar editor."""
        if self.b2Dview:
            self.scene.set_pick_mode('2d')
            self.scene.set_lock_top_view(True)
            self.scene.set_depth_test(False)
            self.scene.set_camera_preset('top')
            msg = "2D view: planar edit (hex/empty OK) | Enter: 3D | Space: Run FF"
        else:
            self.scene.set_pick_mode('3d')
            self.scene.set_lock_top_view(False)
            self.scene.set_depth_test(True)
            msg = "3D ortho: RMB-drag empty=rotate | Arrows: rotate | Shift+Arrows: pan | 5:Top | Enter: 2D"
        if hasattr(self, 'b2Dview_chk') and self.b2Dview_chk.isChecked() != self.b2Dview:
            self.b2Dview_chk.blockSignals(True)
            self.b2Dview_chk.setChecked(self.b2Dview)
            self.b2Dview_chk.blockSignals(False)
        self.statusBar().showMessage(msg)
        debug_print(2, f"[VIEW] b2Dview={self.b2Dview} pick={self.scene._pick_mode} lock_top={self.scene._lock_top_view}")

    def toggle_b2Dview(self):
        self.b2Dview = not self.b2Dview
        if hasattr(self, 'b2Dview_chk'):
            self.b2Dview_chk.blockSignals(True)
            self.b2Dview_chk.setChecked(self.b2Dview)
            self.b2Dview_chk.blockSignals(False)
        self.apply_view_mode()

    def toggle_run_simulation(self):
        """Space: toggle interactive FF / simulation if the FF panel exists."""
        btn = getattr(self, 'relax_interactive_btn', None)
        if btn is None:
            self.statusBar().showMessage("Space: no interactive FF panel (enable FF extension)")
            return
        btn.click()

    def _planar_ops_blocked(self):
        if self.b2Dview:
            return False
        self.statusBar().showMessage("2D-only (hex/empty) — press Enter for 2D view")
        return True

    def _mouse_world_and_ray(self, event):
        """Return (p_world, r0, rd).

        Construction hit ``p_world`` is always ray ∩ z=0 (XY molecular plane) when
        possible — needed for ring side / hex / planar topology even in tilted 3D.
        Falls back to view-plane ∩ cam.center if the ray is parallel to XY.
        """
        r0, rd = self.scene._ray_from_mouse(event.pos)
        self._last_r0, self._last_rd = r0, rd
        self._last_mouse_pos = event.pos
        p = self.scene._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0.0, 0.0, 1.0]))
        if p is None and not self.b2Dview:
            cam = self.scene.view.camera
            center = np.array(cam.center, dtype=np.float32)
            p = self.scene._intersect_ray_plane(r0, rd, center, rd)
            if p is None:
                p = center.copy()
        if p is not None:
            self._last_p_world = np.asarray(p, dtype=np.float64)
        return p, r0, rd

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
        # Enter: 2D/3D view toggle; Space: run/stop interactive FF
        if event.key in ('Enter', 'Return'):
            self.toggle_b2Dview()
            event.handled = True
            return
        if event.key in ('Space', ' '):
            self.toggle_run_simulation()
            event.handled = True
            return

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
        """Paste atoms from clipboard; switch to Select and enter sticky δ-move."""
        packed = self.copied_packed
        if packed is None:
            clip_text = QtWidgets.QApplication.clipboard().text()
            if clip_text.strip():
                packed = PackedMolecule.from_text(clip_text)
                if packed is not None:
                    debug_print(2, f"Parsed clipboard text: {packed}")
        if packed is None:
            debug_print(2, "No atoms to paste")
            return
        self._insert_packed(packed, status_fmt="Pasted {n} atoms — sticky δ-move on; LMB click to drop")

    def _insert_packed(self, packed, status_fmt=None):
        """Append PackedMolecule into current graph (like paste). Returns new atom ids."""
        self._push_undo()
        src_com = packed.apos.mean(axis=0) if len(packed.apos) else np.zeros(3)
        if getattr(self, '_last_p_world', None) is not None:
            target = np.asarray(self._last_p_world, dtype=np.float64)
            target[2] = float(src_com[2])
        else:
            target = src_com + np.array([1.5, 1.5, 0.0])
        paste_shift = target - src_com
        new_atom_ids = []
        new_atoms = []
        for i in range(len(packed.etype)):
            z = int(packed.etype[i])
            ename = _z_to_ename(z)
            npi_i = int(packed.npi[i]) if i < len(packed.npi) else 1
            pos = list(packed.apos[i].copy() + paste_shift)
            a = self.backend._append_atom(pos=pos, ename=ename, pin=None, parent=None, npi=npi_i)
            new_atom_ids.append(a._id)
            new_atoms.append(a)
        for col in packed.bonds:
            i, j = int(col[0]), int(col[1])
            if 0 <= i < len(new_atoms) and 0 <= j < len(new_atoms):
                self.backend.graph.add_bond(new_atoms[i], new_atoms[j])
        # XYZ clipboard may lack bonds — distance-bond the new heavy atoms
        if len(packed.bonds) == 0:
            self.backend._create_bonds_by_distance(atoms=[a for a in new_atoms if a.ename not in ('H', 'E') and a.npi != -1])
        self.backend.graph.sync_neighbor_lists()
        self.backend._sync_sys()
        self.refresh_view()
        msg = None
        if status_fmt:
            msg = status_fmt.format(n=len(new_atoms))
        self._enter_select_manip(new_atom_ids, status=msg)
        debug_print(2, f"[INSERT] n={len(new_atoms)} bonds={len(packed.bonds)} → sticky move")
        return new_atom_ids

    def _enter_select_manip(self, atom_ids, status=None):
        """Force Select mode + select atoms + sticky δ-move (paste/import)."""
        if self.edit_mode != 'Select':
            self.set_edit_mode('Select')
            if hasattr(self, 'mode_combo'):
                idx = self.mode_combo.findText('Select')
                if idx >= 0:
                    self.mode_combo.blockSignals(True)
                    self.mode_combo.setCurrentIndex(idx)
                    self.mode_combo.blockSignals(False)
        ids = list(atom_ids) if atom_ids is not None else []
        self.scene.set_selected_ids(ids)
        mouse = getattr(self, '_last_mouse_pos', None)
        if ids:
            self.scene.enter_xform('move', mouse_pos=mouse)
        if status:
            self.statusBar().showMessage(status)

    def _hide_manip_overlays(self):
        """Hide stale CH/H-bond/label/hover overlays while sticky δ/φ runs."""
        self._clear_hover()
        self.cursor_markers.visible = False
        self.debug_markers.visible = False
        self.debug_lines.visible = False
        self.hover_markers.visible = False
        self.scene.hide_xform_overlays()

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
        if state == 1 and int(atom_id) < 0:
            # Sticky δ/φ xform start — snapshot for Ctrl+Z; hide stale overlays
            self._push_undo()
            self._hide_manip_overlays()
            return
        if state == 0:  # Drag end
            self.cursor_markers.visible = True
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
            if self.scene._selected_ids:
                self.scene._highlight_selected()
                self.scene._update_selection_bbox()
            self.sig_geometry_changed.emit()
            debug_print(2, f"Drag end: synced {len(atom_list)} atom positions to graph and sys")

    def on_mouse_move(self, event):
        """Update cursor cross + dispatch to mode handler for hover highlighting."""
        p_world, r0, rd = self._mouse_world_and_ray(event)
        if p_world is None: return
        self.cursor_markers.set_data(pos=np.array([p_world]), symbol='cross', edge_width=2, edge_color='red', face_color='transparent', size=10)
        self.scene.update_view_debug(event.pos, hit=p_world)
        self.backend.detect_geometry_rings()
        self._clear_hover()
        h = self.mode_handlers.get(self.edit_mode)
        if h and h.on_move:
            h.on_move(p_world, r0, rd)
            if getattr(h, 'capture_move', False):
                event.handled = True

    def on_mouse_release(self, event):
        """Dispatch mouse release to mode handler."""
        if getattr(event, 'handled', False):
            return
        h = self.mode_handlers.get(self.edit_mode)
        if h is None or h.on_release is None: return
        ctrl = 'Control' in event.modifiers if isinstance(event.modifiers, (tuple, list)) else False
        p_world, r0, rd = self._mouse_world_and_ray(event)
        if p_world is None: return
        h.on_release(event, p_world, ctrl)

    def on_atom_remove(self, atom_id):
        """Signal callback: RMB on atom dispatched to mode handler."""
        h = self.mode_handlers.get(self.edit_mode)
        if h and h.on_rmb_atom:
            h.on_rmb_atom(atom_id, self.scene._last_ctrl)

    def on_link_bond(self, from_id, to_id):
        """Signal callback: Ctrl+drag bond creation dispatched to mode handler."""
        h = self.mode_handlers.get(self.edit_mode)
        if h and h.on_link:
            h.on_link(from_id, to_id)

    def on_link_to_pos(self, from_id, x, y):
        """Signal callback: Ctrl+drag to empty space — create new atom at (x,y) + bond to from_id."""
        if self._planar_ops_blocked():
            return
        debug_print(2, f"[ON_LINK_TO_POS] from={from_id} pos=({x:.2f},{y:.2f}) type={self.cur_atom_type}")
        self._push_undo()
        self.backend._append_atom(pos=[x, y, 0.0], ename=self.cur_atom_type, pin=None, parent=None, npi=self.backend._get_element_default_npi(self.cur_atom_type))
        atom_list, *_ = self.backend.graph.to_arrays()
        if atom_list:
            new_atom = atom_list[-1]
            # Bond to source atom
            src = self.backend.graph.atoms.get(from_id)
            if src is not None and src.alive:
                self.backend.graph.add_bond(src, new_atom)
            self.backend.graph.sync_neighbor_lists()
        if self.backend.auto_h_cap:
            self.backend.adjust_h()
        self.backend._sync_sys()
        self.refresh_view()
        self.sig_geometry_changed.emit()

    def on_atom_clicked(self, atom_id):
        """Signal callback: atom click without drag dispatched to mode handler."""
        debug_print(2, f"[ON_ATOM_CLICKED] atom_id={atom_id} mode={self.edit_mode}")
        h = self.mode_handlers.get(self.edit_mode)
        if h and h.on_atom_click:
            h.on_atom_click(atom_id)

    def on_mouse_press(self, event):
        """Dispatch mouse press to mode handler."""
        if getattr(event, 'handled', False):
            debug_print(2, f"[GUI_PRESS] SKIPPED (event.handled=True) mode={self.edit_mode}")
            return
        h = self.mode_handlers.get(self.edit_mode)
        if h is None or h.on_press is None: return
        ctrl = 'Control' in event.modifiers if isinstance(event.modifiers, (tuple, list)) else False
        p_world, r0, rd = self._mouse_world_and_ray(event)
        if p_world is None: return
        debug_print(2, f"[GUI_PRESS] mode={self.edit_mode} b2D={self.b2Dview} pos=({p_world[0]:.2f},{p_world[1]:.2f},{p_world[2]:.2f}) ctrl={ctrl}")
        h.on_press(event, p_world, ctrl)

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
                                filter_str="Molecular Files (*.xyz *.mol *.mol2);;XYZ (*.xyz);;MOL (*.mol);;MOL2 (*.mol2)",
                                start_dir=self.work_dir)
        if fname:
            self.backend.save_structure(fname)
            self.statusBar().showMessage(f"Exported to {fname}")
            debug_print(2, f"Exported to {fname}")

    def import_structure(self):
        """Import structure from file and APPEND to current graph (same as Ctrl+V)."""
        fname = self.fileDialog(mode="open", title="Import Structure",
                                filter_str="Molecular Files (*.xyz *.mol *.mol2);;XYZ (*.xyz);;MOL (*.mol);;MOL2 (*.mol2)",
                                start_dir=self.work_dir)
        if not fname:
            return
        # Load into a temp backend so we do not replace the current molecule
        from spammm.topology.MoleculeEditorBackend import MoleculeEditorBackend
        tmp = MoleculeEditorBackend()
        tmp.load_structure(fname)
        packed = PackedMolecule.from_graph(tmp.graph)
        if packed is None or len(packed.etype) == 0:
            self.statusBar().showMessage(f"Import failed / empty: {fname}")
            return
        self._insert_packed(packed, status_fmt=f"Imported {{n}} atoms from {fname} — sticky δ-move; LMB to drop")
        debug_print(2, f"Imported append from {fname} n={len(packed.etype)}")

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

        # 0.6. Debug view: ring COG + bounding circles for all detected rings
        if self.debug_view_mode:
            self.backend.detect_geometry_rings()
            rings = self.backend.geometry_rings
            if rings:
                cog_pos = []
                circle_segs = []
                n_seg = 32
                for ring in rings:
                    if not ring.alive: continue
                    cog_pos.append(ring.cog)
                    # Draw bounding circle as line segments in xy plane
                    z = 0.02
                    for k in range(n_seg):
                        a0 = 2 * np.pi * k / n_seg
                        a1 = 2 * np.pi * (k + 1) / n_seg
                        circle_segs.append([ring.cog[0] + ring.radius * np.cos(a0), ring.cog[1] + ring.radius * np.sin(a0), z])
                        circle_segs.append([ring.cog[0] + ring.radius * np.cos(a1), ring.cog[1] + ring.radius * np.sin(a1), z])
                if cog_pos:
                    self.scene.hover_ring_markers.set_data(pos=np.array(cog_pos, dtype=np.float32), symbol='cross', edge_width=2, edge_color='yellow', face_color='yellow', size=10)
                    self.scene.hover_ring_markers.visible = True
                    if circle_segs:
                        segs = np.array(circle_segs, dtype=np.float32)
                        conn = np.zeros(len(segs), dtype=bool); conn[0::2] = True
                        self.scene.hover_ring_lines.set_data(pos=segs, connect=conn, color=(1.0, 0.9, 0.0, 0.3), width=1.0)
                        self.scene.hover_ring_lines.visible = True
                    self.scene.hover_ring_text.text = ''
                else:
                    self.scene.hover_ring_markers.visible = False
                    self.scene.hover_ring_lines.visible = False
        # When debug_view_mode is off, ring visuals are managed by mode handlers (on_move)

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
        
        # Fragment extension: component coloring
        if getattr(self, '_frag_component_colors', None):
            idx_map = getattr(self.backend, '_atom_idx_map', {})
            for aid, color in self._frag_component_colors.items():
                i = idx_map.get(aid)
                if i is not None and i < len(colors):
                    colors[i] = color
        
        # Bonds
        bonds_heavy = []
        bonds_h = []
        if sys.bonds is not None:
            is_heavy = np.array([sys.enames[i] != 'H' for i in range(len(sys.enames))])
            for b in sys.bonds:
                if is_heavy[b[0]] and is_heavy[b[1]]:
                    bonds_heavy.append(b)
                else:
                    bonds_h.append(b)

        # Bond visuals
        if self.bond_viz_mode and bonds_heavy:
            bond_segs, bond_colors = compute_bond_colors_by_length(bonds_heavy, pos)
            self.scene._line_set("bonds-colored", self.scene.bond_colored_lines, bond_segs, color=bond_colors, width=5.0)
            self.scene.bond_colored_lines.visible = True
            self.scene.bond_lines.visible = False
            bonds_arg = None
        elif self.bond_viz_mode:
            self.scene.bond_colored_lines.visible = False
            self.scene.bond_lines.visible = False
            bonds_arg = None
        else:
            self.scene.bond_colored_lines.visible = False
            self.scene.bond_lines.visible = bool(bonds_heavy)
            bonds_arg = bonds_heavy if bonds_heavy else None

        if bonds_h:
            h_segs = pos[np.array(bonds_h)].reshape(-1, 3)
            self.scene._line_set("CH-bonds", self.scene.ch_bond_lines, h_segs, color=(0.4, 0.4, 0.4, 0.6), width=1.0)
        else:
            self.scene.ch_bond_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        hbonds = sys.find_hbonds(bPrint=False) if sys.bonds is not None else []
        if hbonds:
            hb_segs = []
            for d, h, a, dist, ang in hbonds:
                hb_segs.append(pos[h])
                hb_segs.append(pos[a])
            hb_segs = np.array(hb_segs, dtype=np.float32)
            self.scene._line_set("H-bonds", self.scene.hbond_lines, hb_segs, color=(0.8, 0.2, 0.8, 0.5), width=1.5)
        else:
            self.scene.hbond_lines.set_data(np.zeros((0, 3), dtype=np.float32))

        # Fragment extension: bond + bbox highlights via dedicated visuals
        # Update atom markers FIRST — set_frag_highlights/_redraw must not run on stale _pos/_colors
        self.scene.set_data(pos, colors=colors, sizes=sizes, bonds=bonds_arg)

        frag_data = getattr(self, '_frag_overlay', None)
        if frag_data:
            self.scene.set_frag_highlights(**frag_data)
        else:
            self.scene.set_frag_highlights()

        # Bond orders (from Bond.order on AtomicGraph — authoritative store)
        bo_bonds, bo_vals = self.backend.get_graph_bond_orders()
        if bo_bonds is not None:
            self.scene.set_bond_orders(bo_bonds, bo_vals, show_labels=self.show_bond_order_labels)
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

# FireCore / legacy alias
KekuleExplorerWindow = SPAMMMWindow

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SPAMMM GUI — Molecular editor and AFM simulation')
    parser.add_argument('--output-dir', '-o', type=str, default=None, help='Directory for saved images (default: <repo>/output)')
    parser.add_argument('--fdata-path', '-f', type=str, default=None, help='Path to Fdata directory')
    parser.add_argument('--verbosity', '-v', type=int, default=None, choices=[0, 1, 2, 3], help='Verbosity level 0-3 (default: 2)')
    parser.add_argument('--mol', '-m', type=str, default=None, metavar='PATH', help='Molecule file to load on startup (.xyz/.mol/.mol2)')
    parser.add_argument('--dir', '-d', type=str, default=None, metavar='PATH', help='Working directory for save/load dialogs')
    parser.add_argument('--script', '-s', type=str, default=None, metavar='PATH', help='GUI control script (must define run(window, argv)); args after -- go to script')
    args, script_argv = parser.parse_known_args()

    app = QtWidgets.QApplication(sys.argv)
    window = SPAMMMWindow(output_dir=args.output_dir, fdata_path=args.fdata_path, verbosity=args.verbosity, work_dir=args.dir)
    window.show()
    QtWidgets.QApplication.processEvents()
    if args.mol:
        atom_ids = window.backend.load_structure(args.mol)
        window.refresh_view()
        window._enter_select_manip(atom_ids, status=f"Loaded {args.mol} — sticky δ-move on; LMB click to drop")
        debug_print(2, f"Loaded molecule from CLI: {args.mol}")
        QtWidgets.QApplication.processEvents()
    if args.script:
        from spammm.GUI.gui_script_runner import run_gui_script
        script_argv = list(script_argv)
        if script_argv and script_argv[0] == '--':
            script_argv = script_argv[1:]
        print(f"[GUI] Running control script: {args.script} argv={script_argv}")
        run_gui_script(window, args.script, script_argv)
        QtWidgets.QApplication.processEvents()
    sys.exit(app.exec_())
