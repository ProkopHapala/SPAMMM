"""
VibrationPlot.py — Top-view normal-mode figures (in-plane arrows + seismic z circles).

Arrows are globally scaled so the longest in-plane displacement is 1 Å; atom circles
use covalent radius × 0.35 with `seismic` fill for z amplitude. Used by tests
(`plot_softest_modes`) and GUI (`make_mode_figure` + `plotutils.show_in_plot_window`).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from .. import elements
from .Vibrations import VibrationResult, FreqUnit, format_freq, format_E_zpe

_RCOV_SCALE = 0.35  # display radius = covalent radius × this factor


def _atom_display_radii(enames):
    return np.array([elements.ELEMENT_DICT[e][elements.index_Rcov] * _RCOV_SCALE for e in enames], dtype=np.float64)


def make_mode_figure(result: VibrationResult, mode_index: int, scale=1.0, margin_A=1.0, max_arrow_A=1.0, unit: FreqUnit = 'cm-1'):
    """Build matplotlib Figure for one mode (caller owns figure lifecycle)."""
    pos = np.asarray(result.pos, dtype=np.float64)
    enames = result.enames
    u = result.modes[:, mode_index].reshape(-1, 3).astype(np.float64) * scale
    mi = result.mode_info[mode_index]
    u_xy = u[:, :2]
    uz = u[:, 2]
    radii = _atom_display_radii(enames)

    max_xy = float(np.max(np.linalg.norm(u_xy, axis=1)))
    u_xy_plot = u_xy * (max_arrow_A / max_xy) if max_xy > 1e-12 else np.zeros_like(u_xy)

    uz_max = float(np.max(np.abs(uz)))
    norm_z = Normalize(vmin=-uz_max if uz_max > 1e-12 else -1.0, vmax=uz_max if uz_max > 1e-12 else 1.0)
    cmap = plt.cm.seismic

    tips_x = np.concatenate([pos[:, 0], pos[:, 0] + u_xy_plot[:, 0]])
    tips_y = np.concatenate([pos[:, 1], pos[:, 1] + u_xy_plot[:, 1]])
    xlo, xhi = float(tips_x.min() - margin_A), float(tips_x.max() + margin_A)
    ylo, yhi = float(tips_y.min() - margin_A), float(tips_y.max() + margin_A)

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(len(pos)):
        ax.add_patch(Circle((pos[i, 0], pos[i, 1]), radii[i], facecolor=cmap(norm_z(uz[i])), edgecolor='0.2', linewidth=0.4, zorder=2))
    if max_xy > 1e-12:
        ax.quiver(pos[:, 0], pos[:, 1], u_xy_plot[:, 0], u_xy_plot[:, 1], angles='xy', scale_units='xy', scale=1.0, color='k', width=0.003, headwidth=4, headlength=5, zorder=3)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect('equal')
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_title(f"mode {mode_index}: {result.mode_freq_label(mode_index, unit)}  f_xy={mi.f_xy:.2f} f_z={mi.f_z:.2f}  ({mi.character})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_z)
    sm.set_array(uz)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label('z displacement (mode amplitude)')
    fig.tight_layout()
    return fig


def plot_mode_topview(result: VibrationResult, mode_index: int, savepath: str, scale=1.0, margin_A=1.0, max_arrow_A=1.0):
    """Plot one normal mode in xy projection and save to savepath."""
    fig = make_mode_figure(result, mode_index, scale=scale, margin_A=margin_A, max_arrow_A=max_arrow_A)
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_softest_modes(result: VibrationResult, outdir: str, n=6, prefix='mode', **kw):
    """Plot n softest vibrational modes; return list of saved paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    nplot = min(n, len(result.mode_info))
    for i in range(nplot):
        path = os.path.join(outdir, f'{prefix}_{i:02d}_f{result.mode_info[i].freq_cm1:.0f}.png')
        plot_mode_topview(result, i, path, **kw)
        paths.append(path)
    return paths


def save_summary(result: VibrationResult, path: str):
    """Write mode table to text file."""
    with open(path, 'w') as f:
        f.write(f"# Vibrations backend={result.backend}  N={len(result.enames)}  n_modes={len(result.mode_info)}\n")
        f.write(result.format_table())
        f.write('\n')
