"""
LayoutPolicy.py — Centralized layout/style SSOT for SPAMMM GUI.

Purpose: Define all layout constants (margins, spacing, widget widths, panel
width) in one place, and provide helper methods that extensions use instead
of raw Qt calls. This ensures a consistent, tight layout across all extensions
without each one independently picking values.

Key functionality:
  - Constants: PANEL_TARGET_WIDTH, MARGIN=0, SPACING=1, SPIN_MAX_WIDTH
  - apply_tight(layout): sets zero margins + minimal spacing on any QLayout
  - FlowLayout: QLayout subclass that wraps items to next line when they
    exceed available width (replaces fixed QHBoxLayout rows)
  - tight_button/tight_spin/tight_combo: create widgets with Maximum size
    policy (natural width — only as wide as content, no expansion)

Role in SPAMMM: All GUI extensions and BaseGUI use this module. Extensions
call LayoutPolicy.apply_tight(layout) instead of manual setSpacing/
setContentsMargins. Button rows use FlowLayout to wrap automatically.
Widgets use Maximum size policy so they're only as wide as their content.
"""

from PyQt5 import QtWidgets, QtCore, QtGui


# ---------------------------------------------------------------------------
# Constants (SSOT)
# ---------------------------------------------------------------------------
PANEL_TARGET_WIDTH = 320   # intended sidebar width
PANEL_MIN_WIDTH = 200
PANEL_MAX_WIDTH = 450

MARGIN = 0          # eliminate margins — tightest possible
SPACING = 1         # minimal spacing between widgets
ROW_SPACING = 1     # between rows in vertical layouts

# Spinboxes: wide enough to show values (5-6 digits + arrows), but capped
# so they don't take excessive space. With Preferred policy they fill
# available space up to this cap.
SPIN_MAX_WIDTH = 70

# Backward-compat aliases (deprecated — use setSizePolicy(Maximum, Fixed) instead)
BUTTON_MAX_WIDTH = 9999  # effectively no cap — size policy controls width
COMBO_MAX_WIDTH = 9999

FONT_SIZE = 8       # pt — set in BaseGUI.__init__


# ---------------------------------------------------------------------------
# apply_tight — apply policy margins + spacing to any layout
# ---------------------------------------------------------------------------
def apply_tight(layout, margins=MARGIN, spacing=SPACING):
    """Set minimal margins + spacing on a QLayout. Returns the layout."""
    layout.setContentsMargins(margins, margins, margins, margins)
    layout.setSpacing(spacing)
    return layout


class AutoGridPlacer:
    """Excel-like grid layout with AUTOMATIC span calculation.

    Panel width divided into `cols` EQUAL cells. Each widget's span is
    auto-calculated from its sizeHint() — no manual span parameter needed.

    Label+input pairs are treated as ONE component (measured together).

    Auto-wraps to next row when remaining cells can't fit the widget.
    All cells are equal width (enforced via column stretch), so alignment
    is perfect and predictable.

    Usage:
        g = AutoGridPlacer(cols=4)
        g.add(button)                    # span auto from sizeHint
        g.add_pair("dt:", spinbox)       # label+input measured together
        g.newrow()                       # optional forced break
        layout.addLayout(g.layout())
    """
    def __init__(self, cols=4, cell_width=80):
        self._grid = QtWidgets.QGridLayout()
        apply_tight(self._grid)
        self._cols = cols
        self._cell_w = cell_width  # reference width for span calculation
        self._row = 0
        self._col = 0
        # Equal columns via stretch only (NO minimum width — prevents overflow
        # in scroll areas / narrow panels). Stretch=1 makes all columns equal.
        for i in range(cols):
            self._grid.setColumnStretch(i, 1)

    def _text_width(self, widget, text):
        """Measure text pixel width using widget's fontMetrics."""
        fm = widget.fontMetrics()
        if hasattr(fm, 'horizontalAdvance'):
            return fm.horizontalAdvance(text)
        return fm.width(text)  # PyQt5 fallback

    def _auto_span(self, widget):
        """Calculate span from ACTUAL text width via fontMetrics.

        Uses widget's text (buttons, labels) or longest item (combos).
        Padding is minimal — just enough for borders/decorations.
        """
        cw = self._cell_w
        # Buttons: text width + 10px padding (5px each side)
        if isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QToolButton)):
            text = widget.text()
            total = self._text_width(widget, text) + 10
            return max(1, (total + cw - 1) // cw)
        # Checkboxes: text width + 20px (checkbox square + spacing)
        if isinstance(widget, QtWidgets.QCheckBox):
            text = widget.text()
            total = self._text_width(widget, text) + 20
            return max(1, (total + cw - 1) // cw)
        # Labels: text width + 2px (minimal)
        if isinstance(widget, QtWidgets.QLabel):
            text = widget.text()
            total = self._text_width(widget, text) + 2
            return max(1, (total + cw - 1) // cw)
        # Spinboxes: 1 cell (they have maximumWidth set)
        if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return 1
        # LineEdits: 2 cells (need space for typing)
        if isinstance(widget, QtWidgets.QLineEdit):
            return 2
        # Combos: longest item text + 24px (dropdown arrow + padding)
        if isinstance(widget, QtWidgets.QComboBox):
            n = widget.count()
            if n == 0:
                return 1
            longest = max((widget.itemText(i) for i in range(n)), key=len)
            total = self._text_width(widget, longest) + 24
            return max(1, (total + cw - 1) // cw)
        # Fallback: 1 cell
        return 1

    def _place(self, span):
        """Auto-wrap if needed, return (row, col, span)."""
        if self._col + span > self._cols and self._col > 0:
            self._row += 1
            self._col = 0
        start = self._col
        self._col += span
        return self._row, start, span

    def add(self, widget, span=None):
        """Add a widget. Span auto-calculated from sizeHint if not given."""
        if span is None:
            span = self._auto_span(widget)
        r, c, s = self._place(span)
        self._grid.addWidget(widget, r, c, 1, s)
        return self

    def add_pair(self, label_text, widget, label_span=None, input_span=None):
        """Add label + input as one unit. Spans auto-calculated if not given."""
        lbl = QtWidgets.QLabel(label_text)
        if label_span is None:
            label_span = self._auto_span(lbl)
        if input_span is None:
            input_span = self._auto_span(widget)
        total = label_span + input_span
        # If total exceeds cols, give label 1 cell, input gets the rest
        if total > self._cols:
            label_span = 1
            input_span = self._cols - 1
            total = self._cols
        # Auto-wrap
        if self._col + total > self._cols and self._col > 0:
            self._row += 1
            self._col = 0
        self._grid.addWidget(lbl, self._row, self._col, 1, label_span)
        self._grid.addWidget(widget, self._row, self._col + label_span, 1, input_span)
        self._col += total
        return self

    def newrow(self):
        """Force next row (optional — auto-wrap handles most cases)."""
        self._row += 1
        self._col = 0
        return self

    def layout(self):
        """Return the underlying QGridLayout."""
        return self._grid


# Backward-compatible alias
GridPlacer = AutoGridPlacer


_GROUPBOX_STYLE = """
QGroupBox {
    border: 1px solid #888;
    border-radius: 3px;
    margin-top: 14px;
    padding-top: 2px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 4px;
    padding: 0 3px;
}
"""


def tight_groupbox(title):
    """Create a QGroupBox with tight margins but room for the title label."""
    gb = QtWidgets.QGroupBox(title)
    gb.setStyleSheet(_GROUPBOX_STYLE)
    return gb


def make_vbox(parent=None, tight=True):
    """Create a QVBoxLayout with policy margins/spacing applied."""
    lay = QtWidgets.QVBoxLayout(parent)
    if tight:
        apply_tight(lay)
    return lay


def make_hbox(parent=None, tight=True):
    """Create a QHBoxLayout with policy margins/spacing applied."""
    lay = QtWidgets.QHBoxLayout(parent)
    if tight:
        apply_tight(lay)
    return lay


# ---------------------------------------------------------------------------
# FlowLayout — wraps items to next line when they exceed available width
# ---------------------------------------------------------------------------
class FlowLayout(QtWidgets.QLayout):
    """QLayout that arranges items left-to-right, wrapping to next line
    when the available width is exceeded. Like text flow but for widgets.

    Used for button rows that should wrap instead of forcing the panel wider.
    """

    def __init__(self, parent=None, spacing=SPACING, margins=MARGIN):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self._h_spacing = spacing
        self._v_spacing = spacing
        self.setContentsMargins(margins, margins, margins, margins)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            wid = item.widget()
            if wid is not None:
                size = size.expandedTo(wid.minimumSize())
            else:
                size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        eff_rect = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = eff_rect.x()
        y = eff_rect.y()
        line_height = 0
        for item in self._items:
            # Use expandedTo(minimumSize) so widgets with setFixedSize work
            # (QWidgetItem.sizeHint() returns 0,0 for widgets without a layout)
            item_size = item.sizeHint().expandedTo(item.minimumSize())
            if item_size.width() <= 0:
                item_size = QtCore.QSize(50, 30)  # fallback for unsized widgets
            next_x = x + item_size.width() + self._h_spacing
            if next_x - self._h_spacing > eff_rect.right() and line_height > 0:
                x = eff_rect.x()
                y = y + line_height + self._v_spacing
                next_x = x + item_size.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item_size))
            x = next_x
            line_height = max(line_height, item_size.height())
        return y + line_height - rect.y() + m.bottom()


def make_flow(parent=None, spacing=SPACING, margins=MARGIN):
    """Create a FlowLayout with policy defaults."""
    return FlowLayout(parent, spacing=spacing, margins=margins)


# ---------------------------------------------------------------------------
# Widget helpers — create widgets with policy widths
# ---------------------------------------------------------------------------
def tight_button(text, parent=None):
    """Create a QPushButton at natural text width (no expansion)."""
    btn = QtWidgets.QPushButton(text, parent)
    btn.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    return btn


def tight_spin(value=0.0, step=0.1, max_width=SPIN_MAX_WIDTH,
               vmin=-1e9, vmax=1e9, decimals=4, int_mode=False, parent=None):
    """Create a QSpinBox/QDoubleSpinBox with policy max width."""
    if int_mode:
        sp = QtWidgets.QSpinBox(parent)
        sp.setRange(int(vmin), int(vmax))
        sp.setValue(int(value))
        sp.setSingleStep(int(step))
    else:
        sp = QtWidgets.QDoubleSpinBox(parent)
        sp.setDecimals(decimals)
        sp.setRange(vmin, vmax)
        sp.setSingleStep(step)
        sp.setValue(value)
    sp.setMaximumWidth(max_width)
    sp.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    return sp


def tight_combo(items=None, parent=None):
    """Create a QComboBox that shows full content (Preferred policy)."""
    cb = QtWidgets.QComboBox(parent)
    cb.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
    cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
    if items:
        cb.addItems(items)
    return cb


# ---------------------------------------------------------------------------
# TIGHT_STYLESHEET — global Qt stylesheet that shrinks intrinsic widget sizes
# ---------------------------------------------------------------------------
# This is the PRIMARY layout-tightening mechanism. Qt's default widget padding
# (6px for buttons, 4px for spinboxes, etc.) makes widgets far wider than their
# text content. This stylesheet eliminates that padding globally, so widgets
# are only as wide as their actual text + minimal chrome.
#
# Combined with Maximum size policy (via enforce_tight), widgets become
# truly minimal-width: text content + 1-2px padding.
TIGHT_STYLESHEET = """
QPushButton {
    padding: 1px 4px;
    min-width: 0;
}
QSpinBox, QDoubleSpinBox {
    padding: 0px 2px;
    min-width: 0;
}
QComboBox {
    padding: 0px 4px;
    min-width: 0;
}
QComboBox QAbstractItemView {
    padding: 0px;
}
QLabel {
    padding: 0px;
    margin: 0px;
}
QCheckBox {
    padding: 0px;
    spacing: 2px;
}
QGroupBox {
    padding: 0px;
    margin: 0px;
    border: 1px solid #888;
}
QGroupBox::title {
    padding: 0px 2px;
}
QToolButton {
    padding: 0px 2px;
    min-width: 0;
}
QSlider {
    padding: 0px;
    margin: 0px;
}
QScrollBar:vertical {
    width: 8px;
}
QScrollBar:horizontal {
    height: 8px;
}
QScrollArea {
    border: none;
}
"""


# ---------------------------------------------------------------------------
# enforce_tight — recursive sweep that forces Maximum size policy on all widgets
# ---------------------------------------------------------------------------
def enforce_tight(widget):
    """Set appropriate size policy on ALL widgets in the tree.

    ROBUST enforcement — runs AFTER the entire UI is built, catches everything.

    Policy per widget type:
      - Buttons: Preferred (fill grid cell, don't expand beyond sizeHint)
      - Labels, Checkboxes: Maximum (natural width, don't expand)
      - Combos, Spinboxes: Preferred (show full content, fill available space)

    Uses findChildren() which returns ALL descendants recursively.

    Call this after initUI() in SPAMMM_GUI and after each build_ui() in
    ExtensionManager.
    """
    Max = QtWidgets.QSizePolicy.Maximum
    Fixed = QtWidgets.QSizePolicy.Fixed
    Pref = QtWidgets.QSizePolicy.Preferred

    # Buttons → Preferred (fill grid cell, don't expand beyond sizeHint)
    for w in widget.findChildren(QtWidgets.QPushButton):
        w.setSizePolicy(Pref, Fixed)
    for w in widget.findChildren(QtWidgets.QToolButton):
        w.setSizePolicy(Pref, Fixed)
    # Labels, checkboxes → Maximum (natural width, don't expand)
    for w in widget.findChildren(QtWidgets.QLabel):
        w.setSizePolicy(Max, Fixed)
    for w in widget.findChildren(QtWidgets.QCheckBox):
        w.setSizePolicy(Max, Fixed)

    # Combos, spinboxes, lineedits → Preferred (show content, fill space)
    # These must be READABLE — don't squeeze them to minimum width.
    for w in widget.findChildren(QtWidgets.QComboBox):
        w.setSizePolicy(Pref, Fixed)
        # Size to contents, not to minimum — so the full text is visible
        w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
    for w in widget.findChildren(QtWidgets.QSpinBox):
        w.setSizePolicy(Pref, Fixed)
    for w in widget.findChildren(QtWidgets.QDoubleSpinBox):
        w.setSizePolicy(Pref, Fixed)
    for w in widget.findChildren(QtWidgets.QLineEdit):
        w.setSizePolicy(Pref, Fixed)
