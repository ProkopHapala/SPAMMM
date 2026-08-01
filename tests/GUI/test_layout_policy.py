"""L0 tests for LayoutPolicy — centralized layout constants + FlowLayout.

Verifies:
  - Constants exist and have expected values
  - apply_tight() sets margins + spacing on layouts
  - FlowLayout wraps items to next line when width is exceeded
  - tight_button/tight_spin/tight_combo apply policy widths
  - make_vbox/make_hbox create tight layouts

Run: pytest tests/GUI/test_layout_policy.py -m "not slow"
"""
import pytest
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5 import QtWidgets
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication([])

from spammm.GUI.LayoutPolicy import (
    PANEL_TARGET_WIDTH, PANEL_MIN_WIDTH, PANEL_MAX_WIDTH,
    MARGIN, SPACING, ROW_SPACING,
    BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH,
    apply_tight, make_vbox, make_hbox, make_flow,
    FlowLayout, tight_button, tight_spin, tight_combo,
)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_constants_exist():
    assert PANEL_TARGET_WIDTH == 320
    assert PANEL_MIN_WIDTH == 200
    assert PANEL_MAX_WIDTH == 450
    assert MARGIN == 0
    assert SPACING == 1
    assert ROW_SPACING == 1
    assert SPIN_MAX_WIDTH == 70


# ── apply_tight ───────────────────────────────────────────────────────────────

def test_apply_tight_vbox():
    lay = QtWidgets.QVBoxLayout()
    apply_tight(lay)
    m = lay.contentsMargins()
    assert (m.left(), m.top(), m.right(), m.bottom()) == (MARGIN, MARGIN, MARGIN, MARGIN)
    assert lay.spacing() == SPACING

def test_apply_tight_custom_margins():
    lay = QtWidgets.QHBoxLayout()
    apply_tight(lay, margins=0, spacing=1)
    m = lay.contentsMargins()
    assert (m.left(), m.top(), m.right(), m.bottom()) == (0, 0, 0, 0)
    assert lay.spacing() == 1

def test_apply_tight_returns_layout():
    lay = QtWidgets.QVBoxLayout()
    ret = apply_tight(lay)
    assert ret is lay


# ── make_vbox / make_hbox ─────────────────────────────────────────────────────

def test_make_vbox_tight():
    lay = make_vbox()
    assert lay.spacing() == SPACING
    m = lay.contentsMargins()
    assert m.left() == MARGIN

def test_make_hbox_tight():
    lay = make_hbox()
    assert lay.spacing() == SPACING

def test_make_vbox_not_tight():
    # With MARGIN=0, non-tight mode still differs in spacing (Qt default=6 or 9)
    lay = make_vbox(tight=False)
    assert lay.spacing() != SPACING  # Qt default spacing is 6 or 9, not our 1


# ── FlowLayout ────────────────────────────────────────────────────────────────

def test_flow_layout_basic():
    flow = FlowLayout()
    btn1 = QtWidgets.QPushButton("A")
    btn2 = QtWidgets.QPushButton("B")
    flow.addWidget(btn1)
    flow.addWidget(btn2)
    assert flow.count() == 2

def test_flow_layout_wraps():
    """FlowLayout should wrap items to a second line when width is exceeded."""
    flow = FlowLayout()
    # Use widgets with fixed sizes (offscreen Qt returns 0,0 sizeHint otherwise)
    for i in range(10):
        w = QtWidgets.QWidget()
        w.setFixedSize(80, 30)
        flow.addWidget(w)
    # heightForWidth should return more height for narrow width (wrapping)
    h_narrow = flow.heightForWidth(100)
    h_wide = flow.heightForWidth(2000)
    assert h_narrow > h_wide, f"Expected wrapping: h(100)={h_narrow} should > h(2000)={h_wide}"

def test_flow_layout_has_height_for_width():
    flow = FlowLayout()
    assert flow.hasHeightForWidth() is True

def test_flow_layout_take_at():
    flow = FlowLayout()
    btn = QtWidgets.QPushButton("X")
    flow.addWidget(btn)
    item = flow.takeAt(0)
    assert item is not None
    assert flow.count() == 0

def test_flow_layout_minimum_size():
    flow = FlowLayout()
    w = QtWidgets.QWidget(); w.setFixedSize(80, 30)
    flow.addWidget(w)
    ms = flow.minimumSize()
    assert ms.width() >= 80
    assert ms.height() >= 30


# ── Widget helpers ────────────────────────────────────────────────────────────

def test_tight_button_natural_width():
    btn = tight_button("Test")
    # Maximum size policy — button doesn't expand beyond text width
    assert btn.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Maximum

def test_tight_spin_max_width():
    sp = tight_spin(value=1.0)
    assert sp.maximumWidth() == SPIN_MAX_WIDTH

def test_tight_spin_int_mode():
    sp = tight_spin(value=5, int_mode=True)
    assert isinstance(sp, QtWidgets.QSpinBox)
    assert sp.maximumWidth() == SPIN_MAX_WIDTH

def test_tight_combo_natural_width():
    cb = tight_combo(items=["a", "b"])
    assert cb.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Preferred
    assert cb.count() == 2

def test_make_flow_returns_flowlayout():
    flow = make_flow()
    assert isinstance(flow, FlowLayout)
