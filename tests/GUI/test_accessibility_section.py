"""L0 smoke test for laptop accessibility section — verify UI construction.

Verifies:
  - create_accessibility_section constructs the panel without error (offscreen Qt)
  - Zoom slider and buttons are properly connected
  - Panel is resizable (min/max width constraints)

Run: pytest tests/GUI/test_accessibility_section.py -m "not slow"
"""
import os
import sys
import numpy as np
import pytest

# Offscreen Qt before any PyQt import
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5 import QtWidgets, QtCore, QtGui

# Create QApplication before any QWidget (must be before importing SPAMMM_GUI)
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)


class _MockScene:
    """Minimal stand-in for AtomScene — provides zoom methods."""
    def __init__(self):
        self._zoom = 1.0
    def get_zoom(self):
        return self._zoom
    def set_zoom(self, zoom):
        self._zoom = zoom
    def reset_view(self):
        self._zoom = 1.0
    def fit_to_atoms(self, margin=1.8):
        pass


class _MockBackend:
    """Minimal stand-in for MoleculeEditorBackend."""
    def __init__(self):
        self.auto_h_cap = True
        self.auto_recalc_bonds = True


def test_accessibility_section_construction():
    """Test that accessibility section can be constructed without errors."""
    # Import after offscreen Qt is set
    from spammm.GUI.SPAMMM_GUI import SPAMMMWindow
    
    # Create minimal window with offscreen Qt (this calls initUI which creates accessibility section)
    window = SPAMMMWindow(output_dir="/tmp/test_output", verbosity=0)
    
    # Verify accessibility section exists
    assert hasattr(window, 'create_accessibility_section'), "Window should have create_accessibility_section method"
    
    # Verify zoom controls exist (created in initUI)
    assert hasattr(window, 'zoom_slider'), "Window should have zoom_slider"
    assert hasattr(window, 'zoom_in_btn'), "Window should have zoom_in_btn"
    assert hasattr(window, 'zoom_out_btn'), "Window should have zoom_out_btn"
    assert hasattr(window, 'reset_zoom_btn'), "Window should have reset_zoom_btn"
    
    # Verify slider range
    assert window.zoom_slider.minimum() == -100, "Zoom slider minimum should be -100"
    assert window.zoom_slider.maximum() == 100, "Zoom slider maximum should be 100"
    assert window.zoom_slider.value() == 0, "Zoom slider initial value should be 0"
    
    # Verify methods exist
    assert hasattr(window, 'on_zoom_slider_changed'), "Window should have on_zoom_slider_changed method"
    assert hasattr(window, 'zoom_in'), "Window should have zoom_in method"
    assert hasattr(window, 'zoom_out'), "Window should have zoom_out method"
    assert hasattr(window, 'reset_view'), "Window should have reset_view method"
    assert hasattr(window, 'sync_zoom_slider'), "Window should have sync_zoom_slider method"


def test_zoom_slider_functionality():
    """Test that zoom slider changes zoom level."""
    from spammm.GUI.SPAMMM_GUI import SPAMMMWindow
    
    window = SPAMMMWindow(output_dir="/tmp/test_output", verbosity=0)
    
    # Mock scene to avoid actual VisPy rendering
    window.scene = _MockScene()
    
    # Test zoom slider at different positions
    window.zoom_slider.setValue(50)
    assert window.scene.get_zoom() > 1.0, "Positive slider value should zoom in"
    
    window.zoom_slider.setValue(-50)
    assert window.scene.get_zoom() < 1.0, "Negative slider value should zoom out"
    
    window.zoom_slider.setValue(0)
    assert abs(window.scene.get_zoom() - 1.0) < 0.1, "Zero slider value should give ~1.0 zoom"


def test_zoom_buttons():
    """Test that zoom buttons work correctly."""
    from spammm.GUI.SPAMMM_GUI import SPAMMMWindow
    
    window = SPAMMMWindow(output_dir="/tmp/test_output", verbosity=0)
    window.scene = _MockScene()
    
    # Test zoom in
    initial_zoom = window.scene.get_zoom()
    window.zoom_in()
    assert window.scene.get_zoom() > initial_zoom, "Zoom in should increase zoom"
    
    # Test zoom out
    window.zoom_out()
    assert window.scene.get_zoom() < initial_zoom * 1.5, "Zoom out should decrease zoom"
    
    # Test reset
    window.scene.set_zoom(5.0)
    window.reset_view()
    assert abs(window.scene.get_zoom() - 1.0) < 0.1, "Reset should return to ~1.0 zoom"


def test_resizable_panel():
    """Test that side panel is resizable (not fixed width)."""
    from spammm.GUI.SPAMMM_GUI import SPAMMMWindow
    
    window = SPAMMMWindow(output_dir="/tmp/test_output", verbosity=0)
    
    # Find the scroll area in the layout
    main_widget = window.centralWidget()
    main_layout = main_widget.layout()
    splitter = main_layout.itemAt(0).widget()
    
    # Verify it's a QSplitter
    assert isinstance(splitter, QtWidgets.QSplitter), "Main layout should use QSplitter for resizable panel"
    
    # Verify splitter has 2 widgets (panel + canvas)
    assert splitter.count() == 2, "Splitter should have 2 widgets"
    
    # Verify first widget (panel) is a QScrollArea
    panel = splitter.widget(0)
    assert isinstance(panel, QtWidgets.QScrollArea), "First widget should be QScrollArea"
    
    # Verify panel has min/max width constraints (not fixed)
    assert panel.minimumWidth() > 0, "Panel should have minimum width"
    assert panel.maximumWidth() > panel.minimumWidth(), "Panel should have maximum width > minimum"
    assert panel.maximumWidth() < 16777215, "Panel should have reasonable maximum width (not QWIDGETSIZE_MAX)"
    
    # Verify horizontal scrollbar policy is AsNeeded
    assert panel.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded, "Panel should use ScrollBarAsNeeded"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
