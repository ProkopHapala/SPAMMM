#!/usr/bin/env python3
"""Render SPAMMM GUI/CLI pipeline block scheme as valid SVG (+ PNG).

Usage:
  python tests/SPM/testplot_spammm_pipeline_scheme.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUTDIR = os.path.join(_ROOT, 'debug', 'spammm_pipeline')

# stage header, fill, light chip fill, stroke
STAGES = [
    ('1  Geometry', '#2f6f8f', '#e8f3f8', '#9ec3d4', [
        ('Draw (GUI)', 'rings / bonds / passivate\nKekule, pi / n-pi edit'),
        ('Load file', '.xyz  .mol  .mol2\n(GUI yes; CLI mostly xyz)'),
        ('SMILES', 'CLI --smiles / examples\n(GUI text box open)'),
        ('ASCII-art', 'heterocycles + : H-bonds\n(GUI + library)'),
        ('-> AtomicGraph', 'topology SSOT'),
    ]),
    ('2  Vacuum relax', '#3d7a4a', '#eaf5ec', '#9bc4a4', [
        ('UFF', 'GPU  opt --method uff'),
        ('SPFF', 'sp3 + pi  --method spff'),
        ('LFF', 'projective Jacobi springs'),
        ('DFTB+', 'QM  3ob / mio  (fork)'),
        ('planar + PCA', 'optional; long axis -> x'),
    ]),
    ('3  On surface', '#2f7a72', '#e6f5f3', '#8fc4bd', [
        ('SURFACE: FAF', 'folded atomic fn\ncompact, periodic, fast'),
        ('SURFACE: GridFF', '3D B-spline channels'),
        ('MOL: Flexible', 'UFF / SPFF on surface'),
        ('MOL: Rigid 6-DOF', 'FoldedRigid / PairFF'),
        ('Assembly / PairFF', 'polymorph, multi-mol\ndemo yes; main GUI later'),
    ]),
    ('4  PP-AFM', '#b86a2a', '#fbf0e6', '#e0b48a', [
        ('Morse + Coulomb', 'classical GridFF, no density\nafm-morse'),
        ('FDBM', 'Pauli(rho)+Hartree+vdW\nrho DFTB <<1s / pySCF / cube\nstock / prolonged'),
        ('2.5D contact', 'FAF-like, memory-light\nContactSurface'),
        ('Kriging GridFF', 'DFT z-scan -> PP\nafm-kriging'),
        ('-> Fz / df', 'PP relax; tip path -> BR-STM'),
    ]),
    ('5  STM', '#5a4f7a', '#efecf5', '#b5adc8', [
        ('Orbital maps psi', 'DFTB LCAO +/- pySCF\nstm orbitals'),
        ('STM current I>=0', 'tips s / pz / py\nstm current'),
        ('Vacuum panel', 'stock / prolonged / pySCF\nstm panel'),
        ('BR-STM', 'MO at PP-relaxed tip xy\nbond-edge contrast\nstm br / GUI S6'),
        ('QM forks', 'DFTB+ + pySCF\nstock / prolonged basis'),
    ]),
]


def _round_box(ax, x, y, w, h, fc, ec, lw=1.0, z=1, rad=0.012):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.004,rounding_size={rad}',
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
                       mutation_aspect=None)
    ax.add_patch(p)
    return p


def render(outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    fig_w, fig_h = 14.8, 7.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor('#f7f5f1')
    ax.set_facecolor('#f7f5f1')

    # banner
    _round_box(ax, 0.01, 0.905, 0.98, 0.08, '#1e2a32', '#1e2a32', lw=0, rad=0.008)
    ax.text(0.03, 0.955, 'SPAMMM pipeline  -  GUI / CLI', color='#f7f5f1',
            fontsize=16, fontweight='bold', va='center', zorder=3)
    ax.text(0.03, 0.922, 'Geometry -> vacuum relax -> surface docking -> PP-AFM -> STM / BR-STM',
            color='#b8c4cc', fontsize=9, va='center', zorder=3)

    # entry chips
    ax.text(0.02, 0.875, 'ENTRY', fontsize=8, fontweight='bold', color='#3d4f5c', va='center')
    for i, (lab, x0) in enumerate([
        ('GUI  SPAMMM_GUI', 0.07),
        ('CLI  run_spm.py', 0.22),
        ('demo_pairff.py', 0.37),
    ]):
        _round_box(ax, x0, 0.858, 0.13, 0.032, '#2c3b45', '#4d6370', lw=0.8, rad=0.02)
        ax.text(x0 + 0.065, 0.874, lab, color='#e8eef2', fontsize=7.5, ha='center', va='center', zorder=3)
    ax.text(0.52, 0.874, 'SSOT: AtomicGraph | spammm/SPM | OpenCL (NVIDIA)',
            fontsize=8, color='#5a6570', va='center')

    n = len(STAGES)
    margin_l, margin_r = 0.018, 0.018
    gap = 0.012
    usable = 1.0 - margin_l - margin_r - gap * (n - 1)
    col_w = usable / n
    y_top = 0.835
    y_bot = 0.06
    col_h = y_top - y_bot
    header_h = 0.055

    col_centers = []
    for i, (title, hc, chip_fc, chip_ec, items) in enumerate(STAGES):
        x0 = margin_l + i * (col_w + gap)
        col_centers.append(x0 + 0.5 * col_w)

        # column body
        _round_box(ax, x0, y_bot, col_w, col_h, '#ffffff', '#d5d0c8', lw=0.9, rad=0.01, z=1)
        # header
        _round_box(ax, x0, y_top - header_h, col_w, header_h, hc, hc, lw=0, rad=0.01, z=2)
        ax.text(x0 + 0.012, y_top - 0.5 * header_h, title, color='white', fontsize=10,
                fontweight='bold', va='center', zorder=3)

        # chips
        pad = 0.012
        area_top = y_top - header_h - 0.012
        area_bot = y_bot + 0.014
        n_items = len(items)
        # leave small gaps between chips + "or" labels
        slot = (area_top - area_bot) / n_items
        chip_h = min(0.095, slot * 0.78)

        for j, (head, detail) in enumerate(items):
            cy_top = area_top - j * slot
            cy = cy_top - 0.5 * slot
            y = cy - 0.5 * chip_h
            _round_box(ax, x0 + pad, y, col_w - 2 * pad, chip_h, chip_fc, chip_ec, lw=0.8, rad=0.008, z=2)
            ax.text(x0 + pad + 0.01, y + chip_h - 0.018, head, fontsize=8.5, fontweight='bold',
                    color='#1e2a32', va='top', zorder=3)
            ax.text(x0 + pad + 0.01, y + chip_h - 0.038, detail, fontsize=7.0,
                    color='#5a6570', va='top', linespacing=1.15, zorder=3)
            if j < n_items - 1:
                ax.text(x0 + 0.5 * col_w, cy_top - slot + 0.008, 'or', fontsize=7,
                        color='#8a8580', ha='center', va='center', fontstyle='italic', zorder=3)

    # arrows between columns
    for i in range(n - 1):
        x1 = col_centers[i] + 0.5 * col_w - 0.004
        x2 = col_centers[i + 1] - 0.5 * col_w + 0.004
        # recompute edges from indices
        x_right = margin_l + i * (col_w + gap) + col_w
        x_left = margin_l + (i + 1) * (col_w + gap)
        arr = FancyArrowPatch((x_right + 0.002, 0.48), (x_left - 0.002, 0.48),
                             arrowstyle='-|>', mutation_scale=12,
                             color='#5a6570', lw=1.6, zorder=4)
        ax.add_patch(arr)

    # soft arrow AFM tip path -> BR-STM (col 4 -> BR chip in col 5)
    arr2 = FancyArrowPatch((margin_l + 4 * (col_w + gap) - 0.002, 0.28),
                          (margin_l + 4 * (col_w + gap) + 0.01, 0.38),
                          arrowstyle='-|>', mutation_scale=10,
                          color='#8a949c', lw=1.1, linestyle='--', zorder=4,
                          connectionstyle='arc3,rad=-0.25')
    ax.add_patch(arr2)
    ax.text(margin_l + 3.65 * (col_w + gap), 0.255, 'PP tip path', fontsize=7, color='#8a949c')

    # footer
    legend = [
        ('#2f6f8f', 'Geometry'),
        ('#3d7a4a', 'Vacuum'),
        ('#2f7a72', 'Surface'),
        ('#b86a2a', 'PP-AFM'),
        ('#5a4f7a', 'STM'),
    ]
    lx = 0.02
    for c, lab in legend:
        _round_box(ax, lx, 0.018, 0.012, 0.018, c, c, lw=0, rad=0.004)
        ax.text(lx + 0.016, 0.027, lab, fontsize=8, color='#5a6570', va='center')
        lx += 0.09
    ax.text(0.50, 0.027, 'Notes: debug/spammm_pipeline/PIPELINE_NOTES.md',
            fontsize=8, color='#5a6570', va='center')

    svg = os.path.join(outdir, 'spammm_pipeline.svg')
    png = os.path.join(outdir, 'spammm_pipeline.png')
    fig.savefig(svg, format='svg', bbox_inches='tight', pad_inches=0.08)
    fig.savefig(png, format='png', dpi=140, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)

    # validate SVG is well-formed XML
    from xml.etree import ElementTree as ET
    ET.parse(svg)
    print(f'REVIEW: {svg}')
    print(f'REVIEW: {png}')
    return svg, png


if __name__ == '__main__':
    render()
