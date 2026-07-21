#!/usr/bin/env python3
"""
npy_viewer — inspect .npy / .npz arrays (summary, ASCII/table, imshow).

CLI:
  python -m spammm.npy_viewer path/to/file.npy
  python -m spammm.npy_viewer path/to/file.npz --key coeffs --print
  python -m spammm.npy_viewer path/to/file.npy --image
  python -m spammm.npy_viewer path/to/file.npz --gui

GUI:
  python -m spammm.npy_viewer --gui [optional path]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ── load / analyze ──────────────────────────────────────────────────────────

def load_arrays(path: str) -> Dict[str, np.ndarray]:
    """Load .npy or .npz → {name: array}. .npy uses key 'arr'."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npz':
        z = np.load(path, allow_pickle=True)
        try:
            return {k: z[k] for k in z.files}
        finally:
            z.close()
    if ext == '.npy':
        return {'arr': np.load(path, allow_pickle=True)}
    # try anyway (some dumps omit extension)
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        try:
            return {k: obj[k] for k in obj.files}
        finally:
            obj.close()
    return {'arr': np.asarray(obj)}


def summarize_array(arr: np.ndarray, name: str = 'arr') -> Dict[str, Any]:
    """Return human-oriented stats for one array."""
    a = np.asarray(arr)
    info: Dict[str, Any] = {
        'name': name,
        'dtype': str(a.dtype),
        'shape': tuple(a.shape),
        'ndim': int(a.ndim),
        'size': int(a.size),
        'nbytes': int(a.nbytes),
        'C_contiguous': bool(a.flags['C_CONTIGUOUS']),
        'F_contiguous': bool(a.flags['F_CONTIGUOUS']),
    }
    if a.dtype == object or a.dtype.kind in 'OSUV':
        info['note'] = 'object/string-like; numeric stats skipped'
        return info
    if a.size == 0:
        info['note'] = 'empty'
        return info
    flat = a.ravel()
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.complexfloating):
        if np.iscomplexobj(flat):
            absv = np.abs(flat)
            info['abs_min'] = float(np.nanmin(absv))
            info['abs_max'] = float(np.nanmax(absv))
            info['abs_mean'] = float(np.nanmean(absv))
            info['n_nan'] = int(np.isnan(flat.real).sum() + np.isnan(flat.imag).sum())
            info['n_inf'] = int(np.isinf(flat.real).sum() + np.isinf(flat.imag).sum())
        else:
            info['min'] = float(np.nanmin(flat))
            info['max'] = float(np.nanmax(flat))
            info['mean'] = float(np.nanmean(flat))
            info['std'] = float(np.nanstd(flat))
            info['n_nan'] = int(np.isnan(flat).sum())
            info['n_inf'] = int(np.isinf(flat).sum())
    elif np.issubdtype(a.dtype, np.integer) or a.dtype.kind == 'b':
        info['min'] = a.dtype.type(flat.min())
        info['max'] = a.dtype.type(flat.max())
        info['mean'] = float(flat.mean())
        if a.size <= 64 or (a.dtype.kind == 'b'):
            uniq, counts = np.unique(flat, return_counts=True)
            if uniq.size <= 16:
                info['value_counts'] = {uniq.dtype.type(u): int(c) for u, c in zip(uniq, counts)}
    return info


def format_summary(info: Dict[str, Any], indent: str = '') -> str:
    """Pretty multi-line summary string."""
    lines = [
        f"{indent}{info['name']}: dtype={info['dtype']}  shape={info['shape']}  "
        f"ndim={info['ndim']}  size={info['size']}  nbytes={info['nbytes']}"
    ]
    contig = []
    if info.get('C_contiguous'): contig.append('C')
    if info.get('F_contiguous'): contig.append('F')
    if contig:
        lines.append(f"{indent}  contiguous: {','.join(contig)}")
    for key in ('min', 'max', 'mean', 'std', 'abs_min', 'abs_max', 'abs_mean', 'n_nan', 'n_inf'):
        if key in info:
            v = info[key]
            if isinstance(v, float):
                lines.append(f"{indent}  {key}: {v:.6g}")
            else:
                lines.append(f"{indent}  {key}: {v}")
    if 'value_counts' in info:
        lines.append(f"{indent}  value_counts: {info['value_counts']}")
    if 'note' in info:
        lines.append(f"{indent}  note: {info['note']}")
    return '\n'.join(lines)


def format_ascii(arr: np.ndarray, precision: int = 4, threshold: int = 200,
                 max_line_width: int = 120, edgeitems: int = 3) -> str:
    """Human-readable ASCII dump (numpy array2string)."""
    a = np.asarray(arr)
    with np.printoptions(precision=precision, threshold=threshold, linewidth=max_line_width,
                         edgeitems=edgeitems, suppress=True, floatmode='maxprec_equal'):
        return np.array2string(a)


def format_table(arr: np.ndarray, precision: int = 4, max_rows: int = 40, max_cols: int = 12) -> str:
    """Tabular ASCII for 1D/2D (truncate large arrays with …)."""
    a = np.asarray(arr)
    if a.ndim == 0:
        return f"{a.item():.{precision}g}" if np.issubdtype(a.dtype, np.number) else str(a.item())
    if a.ndim > 2:
        return format_ascii(a, precision=precision)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    nr, nc = a.shape
    row_idx = list(range(min(nr, max_rows)))
    if nr > max_rows:
        row_idx = list(range(max_rows // 2)) + [-1] + list(range(nr - max_rows // 2, nr))
    col_idx = list(range(min(nc, max_cols)))
    if nc > max_cols:
        col_idx = list(range(max_cols // 2)) + [-1] + list(range(nc - max_cols // 2, nc))

    def cell(i, j):
        if i < 0 or j < 0:
            return '…'
        v = a[i, j]
        if np.issubdtype(a.dtype, np.floating):
            return f"{float(v):.{precision}g}"
        if np.issubdtype(a.dtype, np.complexfloating):
            return f"{v:.{precision}g}"
        return str(v)

    header = 'i\\j | ' + ' '.join(f"{(j if j >= 0 else '…'):>10}" for j in col_idx)
    sep = '-' * len(header)
    lines = [header, sep]
    for i in row_idx:
        label = '…' if i < 0 else str(i)
        lines.append(f"{label:>3} | " + ' '.join(f"{cell(i, j):>10}" for j in col_idx))
    if nr > max_rows or nc > max_cols:
        lines.append(f"(showing truncated {nr}x{nc})")
    return '\n'.join(lines)


def select_2d_slice(arr: np.ndarray, index: Optional[Tuple[int, ...]] = None,
                    axis: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """Reduce to 2D for imshow. Default: take middle along leading axes."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a, 'full'
    if a.ndim == 1:
        return a.reshape(1, -1), 'reshape(1,-1)'
    if a.ndim == 0:
        return a.reshape(1, 1), 'scalar'
    # ndim >= 3
    if index is not None:
        sl = list(index)
        while len(sl) < a.ndim - 2:
            sl.append(a.shape[len(sl)] // 2)
        # last two dims stay full
        idx = tuple(sl[: a.ndim - 2]) + (slice(None), slice(None))
        out = a[idx]
        return np.asarray(out), f"index={idx}"
    # collapse all but last two (or chosen axis pair) via middle slice
    if axis is not None:
        # keep axis and axis+1
        axes = [axis % a.ndim, (axis + 1) % a.ndim]
        if axes[0] == axes[1]:
            axes[1] = (axes[0] + 1) % a.ndim
    else:
        axes = [a.ndim - 2, a.ndim - 1]
    idx = []
    desc = []
    for d in range(a.ndim):
        if d in axes:
            idx.append(slice(None))
        else:
            mid = a.shape[d] // 2
            idx.append(mid)
            desc.append(f"axis{d}={mid}")
    out = np.asarray(a[tuple(idx)])
    # ensure 2D order matches axes order
    if out.ndim == 2 and axes != [a.ndim - 2, a.ndim - 1]:
        # transpose so first kept axis is rows
        # after slicing, remaining dims are in original order
        pass
    return out, ('middle: ' + ', '.join(desc)) if desc else 'full'


def plot_array(arr: np.ndarray, title: str = '', cmap: str = 'viridis',
               slice_index: Optional[Tuple[int, ...]] = None, show: bool = True,
               save: Optional[str] = None):
    """imshow 2D (or mid-slice of ND). Uses plotUtils.imshow_array."""
    from spammm.plotUtils import imshow_array
    data2d, how = select_2d_slice(arr, index=slice_index)
    ttl = title or 'array'
    if how != 'full':
        ttl = f"{ttl} [{how}]"
    fig = imshow_array(data2d, title=ttl, cmap=cmap)
    if save:
        fig.savefig(save, dpi=120, bbox_inches='tight')
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig


def report_file(path: str, key: Optional[str] = None, do_print: bool = False,
                do_table: bool = False, do_image: bool = False,
                precision: int = 4, threshold: int = 200,
                slice_index: Optional[Tuple[int, ...]] = None,
                save_image: Optional[str] = None, show: bool = True) -> str:
    """Load file, print summaries (+ optional ASCII/table/image). Returns text report."""
    arrays = load_arrays(path)
    lines = [f"file: {path}", f"arrays: {list(arrays.keys())}", '']
    keys = [key] if key else list(arrays.keys())
    for k in keys:
        if k not in arrays:
            raise KeyError(f"key {k!r} not in {list(arrays.keys())}")
        arr = arrays[k]
        info = summarize_array(arr, name=k)
        lines.append(format_summary(info))
        if do_print:
            lines.append(format_ascii(arr, precision=precision, threshold=threshold))
        if do_table:
            lines.append(format_table(arr, precision=precision))
        lines.append('')
        if do_image:
            plot_array(arr, title=f"{os.path.basename(path)}:{k}",
                       slice_index=slice_index, show=show, save=save_image)
    return '\n'.join(lines).rstrip() + '\n'


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_slice(s: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not s:
        return None
    return tuple(int(x.strip()) for x in s.split(',') if x.strip() != '')


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Inspect .npy / .npz arrays (summary, ASCII, imshow).')
    p.add_argument('path', nargs='?', help='.npy or .npz file')
    p.add_argument('--key', '-k', help='array key inside .npz (default: all)')
    p.add_argument('--print', '-p', dest='do_print', action='store_true', help='ASCII dump')
    p.add_argument('--table', '-t', action='store_true', help='tabular ASCII for 1D/2D')
    p.add_argument('--image', '-i', action='store_true', help='imshow (2D or mid-slice)')
    p.add_argument('--slice', dest='slice_str', help='leading indices for ND slice, e.g. 5,10')
    p.add_argument('--precision', type=int, default=4)
    p.add_argument('--threshold', type=int, default=200, help='array2string print threshold')
    p.add_argument('--save-image', help='save imshow PNG instead of/in addition to show')
    p.add_argument('--no-show', action='store_true', help='do not plt.show() (use with --save-image)')
    p.add_argument('--gui', '-g', action='store_true', help='open PyQt5 GUI')
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.gui:
        return run_gui(args.path)
    if not args.path:
        build_argparser().print_help()
        return 2
    text = report_file(
        args.path, key=args.key, do_print=args.do_print, do_table=args.table,
        do_image=args.image, precision=args.precision, threshold=args.threshold,
        slice_index=_parse_slice(args.slice_str), save_image=args.save_image,
        show=not args.no_show,
    )
    sys.stdout.write(text)
    return 0


# ── GUI ─────────────────────────────────────────────────────────────────────

def run_gui(path: Optional[str] = None) -> int:
    from PyQt5 import QtWidgets, QtCore
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    class NpyViewerWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('NPY / NPZ viewer')
            self.resize(900, 700)
            self.arrays: Dict[str, np.ndarray] = {}
            self.path: Optional[str] = None

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)

            row = QtWidgets.QHBoxLayout()
            self.path_edit = QtWidgets.QLineEdit()
            self.path_edit.setPlaceholderText('path to .npy / .npz')
            row.addWidget(self.path_edit)
            btn_open = QtWidgets.QPushButton('Open…')
            btn_open.clicked.connect(self.browse_open)
            row.addWidget(btn_open)
            btn_load = QtWidgets.QPushButton('Load')
            btn_load.clicked.connect(self.reload)
            row.addWidget(btn_load)
            layout.addLayout(row)

            row2 = QtWidgets.QHBoxLayout()
            row2.addWidget(QtWidgets.QLabel('array:'))
            self.key_combo = QtWidgets.QComboBox()
            self.key_combo.currentTextChanged.connect(self.refresh_views)
            row2.addWidget(self.key_combo, stretch=1)
            self.slice_edit = QtWidgets.QLineEdit()
            self.slice_edit.setPlaceholderText('slice e.g. 5,10 (leading axes)')
            self.slice_edit.setMaximumWidth(180)
            row2.addWidget(self.slice_edit)
            btn_img = QtWidgets.QPushButton('Imshow')
            btn_img.clicked.connect(self.show_image)
            row2.addWidget(btn_img)
            layout.addLayout(row2)

            split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            self.summary = QtWidgets.QPlainTextEdit()
            self.summary.setReadOnly(True)
            self.summary.setMaximumBlockCount(5000)
            split.addWidget(self.summary)
            self.ascii = QtWidgets.QPlainTextEdit()
            self.ascii.setReadOnly(True)
            self.ascii.setFont(QtWidgets.QFont('monospace', 9))
            split.addWidget(self.ascii)
            layout.addWidget(split, stretch=1)

            self.canvas_host = QtWidgets.QWidget()
            self.canvas_layout = QtWidgets.QVBoxLayout(self.canvas_host)
            self.canvas_layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.canvas_host, stretch=2)
            self._canvas = None

            if path:
                self.path_edit.setText(path)
                self.reload()

        def browse_open(self):
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Open NumPy file', '', 'NumPy (*.npy *.npz);;All (*)')
            if fname:
                self.path_edit.setText(fname)
                self.reload()

        def reload(self):
            p = self.path_edit.text().strip()
            if not p:
                return
            self.arrays = load_arrays(p)
            self.path = p
            self.key_combo.blockSignals(True)
            self.key_combo.clear()
            self.key_combo.addItems(list(self.arrays.keys()))
            self.key_combo.blockSignals(False)
            self.refresh_views()

        def current_array(self):
            k = self.key_combo.currentText()
            if not k or k not in self.arrays:
                return None, None
            return k, self.arrays[k]

        def refresh_views(self, *_):
            k, arr = self.current_array()
            if arr is None:
                self.summary.setPlainText('')
                self.ascii.setPlainText('')
                return
            info = summarize_array(arr, name=k)
            head = f"file: {self.path}\n\n" + format_summary(info)
            self.summary.setPlainText(head)
            # prefer table for small 1D/2D
            if arr.ndim <= 2 and arr.size <= 500:
                self.ascii.setPlainText(format_table(arr))
            else:
                self.ascii.setPlainText(format_ascii(arr, threshold=100))

        def show_image(self):
            k, arr = self.current_array()
            if arr is None:
                return
            from spammm.plotUtils import imshow_array
            sl = _parse_slice(self.slice_edit.text().strip() or None)
            data2d, how = select_2d_slice(arr, index=sl)
            title = f"{os.path.basename(self.path or '')}:{k}"
            if how != 'full':
                title = f"{title} [{how}]"
            fig = imshow_array(data2d, title=title)
            if self._canvas is not None:
                self.canvas_layout.removeWidget(self._canvas)
                self._canvas.deleteLater()
            self._canvas = FigureCanvas(fig)
            self.canvas_layout.addWidget(self._canvas)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = NpyViewerWindow()
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
