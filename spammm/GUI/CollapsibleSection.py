"""
CollapsibleSection.py — Foldable panel widget for PyQt5.

Purpose: Provide a collapsible section widget with a header button that
instantly shows/hides the content area. Used for organizing complex
control panels into tidy hierarchical sections.

Key functionality:
  - Toggle button with arrow indicator (►/▼)
  - Instant show/hide via setVisible — no animation, no delay
  - Content widget container (any QWidget can be added)

Role in SPAMMM: UI component for AFMExtension and other panels that need
many grouped controls (simulation params, rendering options, etc.).
See doc/GUI.desing.md §Visual Design Principles.

Usage:
    sec = CollapsibleSection("Fireball", parent=self)
    sec.setContent(some_widget)
    layout.addWidget(sec)
"""

from PyQt5 import QtWidgets, QtCore, QtGui

from spammm.GUI.LayoutPolicy import MARGIN, SPACING, ROW_SPACING


class CollapsibleSection(QtWidgets.QWidget):
    """A titled panel whose content area can be toggled open/closed."""

    def __init__(self, title: str, collapsed: bool = False, parent=None):
        super().__init__(parent)

        # --- header button ---
        self._toggle = QtWidgets.QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(not collapsed)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            QtCore.Qt.DownArrow if not collapsed else QtCore.Qt.RightArrow
        )
        self._toggle.setText(f" {title}")
        self._toggle.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; text-align: left; }"
        )
        self._toggle.toggled.connect(self._on_toggle)

        # --- content area ---
        self._content = QtWidgets.QWidget()
        self._content.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(MARGIN, 0, 0, MARGIN)
        self._content_layout.setSpacing(ROW_SPACING)

        # --- separator line ---
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        # --- outer layout ---
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._toggle)
        outer.addWidget(self._content)
        outer.addWidget(line)

        if collapsed:
            self._content.setVisible(False)

    def setContent(self, widget: QtWidgets.QWidget):
        """Set (or replace) the widget shown inside this section."""
        # clear existing
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self._content_layout.addWidget(widget)

    def _on_toggle(self, checked: bool):
        self._toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )
        self._content.setVisible(checked)

    def is_open(self) -> bool:
        return self._toggle.isChecked()

    def set_status(self, ok: bool, msg: str = ""):
        """Append a small status indicator to the title."""
        icon = "✓" if ok else "✗"
        title = self._toggle.text().lstrip().split(" [")[0]
        self._toggle.setText(f" {title} [{icon} {msg}]" if msg else f" {title} [{icon}]")
