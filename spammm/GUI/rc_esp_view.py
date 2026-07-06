"""Blitted ESP animation for reaction-coordinate scans — synced to rc_slider.

Blit caveats: spammm/GUI/mpl_blit.py and doc/Takeways.md
- Frame counter is a Qt QLabel (not ax.set_title) to avoid title ghosting.
- MplBlitManager uses ax.bbox because colorbar lives on a sibling axes.
- capture_background() after show(); resize_event triggers re-snapshot.
"""
import numpy as np

from spammm.quantum.esp_grid import compute_esp_stack
from spammm.GUI.mpl_blit import MplBlitManager


class RCESPBlitView:
    """Matplotlib blit viewer: ESP heatmap + atom overlay, one frame at a time."""

    def __init__(self, window, esp_stack, extent, enames, apos_stack, z_abs, vmax=None):
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from PyQt5 import QtWidgets, QtCore

        self.window = window
        self.esp_stack = np.asarray(esp_stack, dtype=np.float64)
        self.extent = list(extent)
        self.enames = list(enames)
        self.apos_stack = np.asarray(apos_stack, dtype=np.float64)
        self.z_abs = float(z_abs)
        self.nframes = len(self.esp_stack)
        self.vmax = float(vmax) if vmax is not None else float(np.nanmax(np.abs(self.esp_stack)))
        if self.vmax <= 0:
            self.vmax = 1e-6

        dlg = QtWidgets.QDialog(window)
        dlg.setWindowTitle(f"RC ESP animation  z={self.z_abs:.2f} Å")
        dlg.resize(720, 620)
        layout = QtWidgets.QVBoxLayout(dlg)
        self._frame_label = QtWidgets.QLabel("frame 0")
        self._frame_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self._frame_label)

        self.fig, self.ax = plt.subplots(figsize=(6.5, 5.5))
        self.im = self.ax.imshow(self.esp_stack[0], origin='lower', extent=self.extent, cmap='seismic', vmin=-self.vmax, vmax=self.vmax, aspect='equal')
        self.fig.colorbar(self.im, ax=self.ax, label='ESP (eV)', shrink=0.85)
        self.ax.set_xlabel('x (Å)')
        self.ax.set_ylabel('y (Å)')
        self.ax.set_title(f'Electrostatic potential  z={self.z_abs:.2f} Å')
        self._atom_artists = []
        for p, e in zip(self.apos_stack[0], self.enames):
            c = 'white' if e == 'H' else ('gray' if e == 'C' else 'magenta')
            ln, = self.ax.plot(p[0], p[1], 'o', color=c, markersize=5, markeredgecolor='k', markeredgewidth=0.5, zorder=5)
            self._atom_artists.append(ln)
        self.fig.tight_layout()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self._blit = MplBlitManager(self.canvas, self.ax)
        self._blit.add_artist(self.im)
        for ln in self._atom_artists:
            self._blit.add_artist(ln)

        self.dlg = dlg
        self._frame = -1
        self.canvas.mpl_connect('resize_event', lambda evt: self._blit.capture_background())
        self.canvas.draw_idle()

        def _on_close(*args):
            self._blit.close()
            window._rc_esp_view = None
        dlg.finished.connect(_on_close)

    def show(self):
        self.dlg.show()
        self.dlg.raise_()
        self.dlg.activateWindow()
        self._blit.capture_background()

    def set_frame(self, frame_idx):
        fi = int(frame_idx)
        if fi < 0 or fi >= self.nframes:
            return
        if fi == self._frame and self.dlg.isVisible():
            return
        self._frame = fi
        self.im.set_array(self.esp_stack[fi])
        apos = self.apos_stack[fi]
        for ln, p in zip(self._atom_artists, apos):
            ln.set_data([p[0]], [p[1]])
        self._frame_label.setText(f"frame {fi}/{self.nframes - 1}")
        self.dlg.setWindowTitle(f"RC ESP animation  z={self.z_abs:.2f} Å  frame {fi}/{self.nframes - 1}")
        self._blit.update()


def ensure_rc_esp_stack(window, z_height=None, n_grid=None, force=False):
    """Precompute [nframe, ny, nx] ESP stack from ScanDataset Mulliken charges."""
    ds = window.rc_dataset
    if ds is None or ds.charges is None:
        return None
    z_height = float(z_height if z_height is not None else window.rc_esp_z_spin.value())
    n_grid = int(n_grid if n_grid is not None else window.rc_esp_n_spin.value())
    meta = ds.meta
    key = (z_height, n_grid)
    if not force and getattr(window, '_rc_esp_cache_key', None) == key and getattr(window, '_rc_esp_stack', None) is not None:
        return window._rc_esp_stack, window._rc_esp_extent, window._rc_esp_z_abs
    if ds.esp_xy is not None and meta.get('esp_z') == z_height and meta.get('esp_n') == n_grid:
        window._rc_esp_stack = ds.esp_xy
        window._rc_esp_extent = meta.get('esp_extent')
        window._rc_esp_z_abs = meta.get('esp_z_abs')
        window._rc_esp_cache_key = key
        return window._rc_esp_stack, window._rc_esp_extent, window._rc_esp_z_abs
    stack, extent, nx, ny, z_abs = compute_esp_stack(ds.apos, ds.charges, z_height, n=n_grid)
    window._rc_esp_stack = stack
    window._rc_esp_extent = extent
    window._rc_esp_z_abs = z_abs
    window._rc_esp_cache_key = key
    ds.esp_xy = stack
    meta.update(esp_z=z_height, esp_n=n_grid, esp_extent=list(extent), esp_nx=nx, esp_ny=ny, esp_z_abs=z_abs)
    ds.meta = meta
    return stack, extent, z_abs


def open_rc_esp_animation(window):
    """Open blitted ESP viewer synced to rc_slider."""
    ds = window.rc_dataset
    if ds is None:
        window.rc_status.setText("Run scan first")
        return None
    if ds.charges is None:
        window.rc_status.setText("No Mulliken charges — re-run DFTB scan")
        return None
    out = ensure_rc_esp_stack(window)
    if out is None:
        return None
    stack, extent, z_abs = out
    enames = list(ds.enames())
    view = RCESPBlitView(window, stack, extent, enames, ds.apos, z_abs)
    window._rc_esp_view = view
    view.show()
    view.set_frame(window.rc_slider.value())
    return view


def update_rc_esp_frame(window, frame_idx):
    """Called from show_scan_frame when ESP viewer is open."""
    view = getattr(window, '_rc_esp_view', None)
    if view is not None and view.dlg.isVisible():
        view.set_frame(frame_idx)
