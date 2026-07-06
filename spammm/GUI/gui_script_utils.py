"""Helpers for programmatic GUI scripts — same state as user widget actions."""
from PyQt5 import QtWidgets, QtCore


def process_events(window=None):
    """Pump Qt event loop so layout/repaint catches up."""
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()


def expand_extension_panel(window, name_or_title, open=True):
    """Expand/collapse an extension CollapsibleSection by registry key or display title."""
    sections = getattr(window, '_extension_sections', {})
    sec = sections.get(name_or_title)
    if sec is None:
        raise KeyError(f"Extension panel {name_or_title!r} not found; keys={list(sections.keys())}")
    sec._toggle.setChecked(open)
    process_events(window)
    return sec


def click_button(btn):
    """Invoke a QPushButton's connected slot (same as user click)."""
    btn.click()
    process_events()


def set_combo_text(combo, text):
    combo.setCurrentText(text)
    process_events()


def set_spin_value(spin, value):
    spin.setValue(value)
    process_events()


def set_slider_value(slider, value):
    slider.blockSignals(True)
    slider.setValue(int(value))
    slider.blockSignals(False)
    process_events()
