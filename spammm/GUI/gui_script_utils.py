"""Helpers for programmatic GUI scripts — same state as user widget actions."""
import os

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


def set_line_edit(line_edit, text):
    line_edit.setText(text)
    process_events()


def set_spin_value(spin, value):
    spin.setValue(value)
    process_events()


def set_slider_value(slider, value):
    slider.blockSignals(True)
    slider.setValue(int(value))
    slider.blockSignals(False)
    process_events()


def set_check(check, checked):
    """Set a QCheckBox state."""
    check.setChecked(bool(checked))
    process_events()


def load_molecule(window, path):
    """Load an XYZ file into the GUI backend and refresh the view."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    window.backend.load_xyz(path)
    window.refresh_view()
    process_events(window)


def set_edit_mode(window, mode):
    """Switch the main editor interaction mode."""
    window.set_edit_mode(mode)
    process_events(window)


def extension_panel(window, key, open=True):
    """Backward-compatible alias for expand_extension_panel."""
    return expand_extension_panel(window, key, open=open)
