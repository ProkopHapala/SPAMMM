"""
plotutils.py — Qt-specific 2D plotting utilities for SPAMMM GUI extensions.

Re-exports pure-matplotlib functions (compute_grid_extent, make_2d_grid,
overlay_atoms, plot_2d_scalar) from spammm.plotUtils, and adds the Qt-specific
show_in_plot_window for embedding matplotlib figures in reusable QDialog windows.

Used by: QEqExtension (ESP), AFMExtension (orbital/density/slice plots), and any
future extension that needs 2D scalar field visualization in the GUI.
"""

from spammm.plotUtils import (
    compute_grid_extent, make_2d_grid, overlay_atoms, plot_2d_scalar, imshow_array,
)

from PyQt5 import QtWidgets


def show_in_plot_window(window, fig, title="Plot", attr='_plot_window'):
    """Show a matplotlib Figure in a reusable Qt dialog window.

    Reuses the same dialog if already open (replaces figure). The dialog
    reference is stored on window as `attr`.

    Args:
        window: parent Qt widget (typically the main GUI window)
        fig: matplotlib Figure
        title: window title
        attr: attribute name on window to store the dialog reference
    """
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    dlg = getattr(window, attr, None)
    layout = getattr(window, f'{attr}_layout', None)

    if dlg is None:
        dlg = QtWidgets.QDialog(window)
        layout = QtWidgets.QVBoxLayout(dlg)
        setattr(window, attr, dlg)
        setattr(window, f'{attr}_layout', layout)
        dlg.resize(700, 600)
        def on_closed(*args, **kwargs):
            setattr(window, attr, None)
        dlg.finished.connect(on_closed)
    else:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    canvas = FigureCanvas(fig)
    if hasattr(window, 'install_mpl_canvas_screenshot_menu'):
        try:
            window.install_mpl_canvas_screenshot_menu(canvas, fig, default_name=f"{title.replace(' ','_')}.png")
        except Exception:
            pass
    layout.addWidget(canvas)
    dlg.setWindowTitle(title)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
