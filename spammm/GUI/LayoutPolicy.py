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

# No max width caps — widgets should be natural size (size policy = Maximum)
# Buttons/labels/combos shrink to fit their text content.
# Spinboxes keep a small cap since they have excessive default width.
SPIN_MAX_WIDTH = 55

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
    """Create a QComboBox at natural width (no expansion)."""
    cb = QtWidgets.QComboBox(parent)
    cb.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    if items:
        cb.addItems(items)
    return cb
