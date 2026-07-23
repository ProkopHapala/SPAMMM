#!/usr/bin/env python3
"""
AFM_utils.py — High-level AFM utilities, plotting, and FDBM orchestration.

Purpose: Orchestrate AFM simulations by combining AFM.py physics with QM density
providers (DFTB, pySCF), CO tip models, plotting, and I/O. This module adds
orchestration on top of AFM.py's pure physics.

Key functionality:
  - Plotting: AFM frequency shift maps, tip trajectories, orbital densities
  - FDBM vs Kriging z-layout SSOT: plot_fdbm_vs_kriging_zlayout(), plot_fdbm_methods_zcompare_4panel()
  - Density providers: get_density_from_dftb(), get_density_from_pyscf(), get_density_from_cube()
  - CO tip: _co_tip_cache_dir(), _compute_co_tip_subprocess()
  - FDBM helpers: fft_poisson(), compute_pauli_field(), compute_es_conv_field()
  - STM: compute_stm(), compute_bond_resolved_stm(), compute_stm_basis_variants()

Role in SPAMMM: AFM orchestration layer. Used by ModularPipeline.py for all
stages and by AFMExtension.py for result visualization. Depends on AFM.py for
physics and DFTB/Grid_dftb.py for density projection.

Design principle: AFM.py contains pure physics (no matplotlib, no QM).
This module depends on AFM.py and adds plotting, I/O, and orchestration.

Open issues / caveats:
  - STM basis compare must use use_exp_basis=False so projector STO (stock vs prolonged)
    is the imaging object; exp(β,r0) bypasses WFC and hides the prolonged-tail effect.
  - Dual-basis AFM rule (prolonged Pauli only) does NOT apply to STM ψ — prolonged radial
    is the map. See doc/Tasks/STM_ExtendedBasis_OrbitalCompare.md.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Import core AFM physics
from . import AFM as afm

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting Utilities (moved from AFM.py)
# ═══════════════════════════════════════════════════════════════════════════════

def safe_norm(data_2d, pct=99):
    """Symmetric ±vabs TwoSlopeNorm for diverging colormaps."""
    vabs = max(float(np.percentile(np.abs(data_2d), pct)), 1e-6)
    return TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)


def scan_extent(scan_xs, scan_ys):
    """[xmin,xmax,ymin,ymax] for imshow from 1D scan axes."""
    return [float(scan_xs[0]), float(scan_xs[-1]), float(scan_ys[0]), float(scan_ys[-1])]


def crop_afm_xy(data, scan_xs, scan_ys, view_extent):
    """Crop (nx,ny[,nz]) AFM volume + axes to [xmin,xmax,ymin,ymax] for fair side-by-side compare.

    Returns (data_c, xs_c, ys_c, extent_c). If view_extent is None, returns inputs unchanged.
    """
    if view_extent is None:
        return np.asarray(data), np.asarray(scan_xs), np.asarray(scan_ys), scan_extent(scan_xs, scan_ys)
    xmin, xmax, ymin, ymax = [float(v) for v in view_extent]
    xs = np.asarray(scan_xs, float)
    ys = np.asarray(scan_ys, float)
    ix0 = int(np.searchsorted(xs, xmin, side='left'))
    ix1 = int(np.searchsorted(xs, xmax, side='right'))
    iy0 = int(np.searchsorted(ys, ymin, side='left'))
    iy1 = int(np.searchsorted(ys, ymax, side='right'))
    ix0 = max(0, min(ix0, len(xs) - 1)); ix1 = max(ix0 + 1, min(ix1, len(xs)))
    iy0 = max(0, min(iy0, len(ys) - 1)); iy1 = max(iy0 + 1, min(iy1, len(ys)))
    d = np.asarray(data)
    d = d[ix0:ix1, iy0:iy1, ...] if d.ndim >= 2 else d
    xs_c, ys_c = xs[ix0:ix1], ys[iy0:iy1]
    return d, xs_c, ys_c, scan_extent(xs_c, ys_c)


def imshow_afm(ax, arr_nxny, extent=None, cmap='bwr', symmetric=True, pct=99, title='', colorbar=True, transpose=True):
    """SSOT: one AFM XY map. `arr_nxny` is (nx,ny) scan layout; transposed for imshow by default.

    Returns the AxesImage. Prefer this over raw `ax.imshow(...)` for E/Fz/df maps.
    """
    data = np.asarray(arr_nxny).T if transpose else np.asarray(arr_nxny)
    kw = dict(origin='lower', cmap=cmap, aspect='equal')
    if extent is not None: kw['extent'] = extent
    if symmetric:
        kw['norm'] = safe_norm(data, pct=pct)
    im = ax.imshow(data, **kw)
    if title: ax.set_title(title)
    if colorbar: plt.colorbar(im, ax=ax, shrink=0.7, fraction=0.04, pad=0.02)
    return im


def plot_afm_height_panel(data, heights, iz=None, extent=None, label='Fz', cmap='bwr', fname=None, save_dir='.', figsize_col=2.8, dpi=150, ylabel=None, transpose=True):
    """SSOT: row of XY maps at selected heights. `data` is (nx,ny,nz).

    Args:
        data: (nx, ny, nz) E/Fz/df volume
        heights: (nz,) probe heights [Å]
        iz: list of z-indices (default: evenly spaced up to 6)
        extent: [xmin,xmax,ymin,ymax] or None
        label / ylabel: figure / left-axis label
        fname: if set, save under save_dir and close
    Returns:
        fig, axes
    """
    nz = data.shape[2]
    if iz is None:
        n = min(6, nz)
        iz = list(np.linspace(0, nz - 1, n, dtype=int))
    else:
        iz = [i for i in iz if 0 <= i < nz]
    fig, axes = plt.subplots(1, len(iz), figsize=(figsize_col * len(iz), figsize_col), squeeze=False)
    axes = axes[0]
    for ax, i in zip(axes, iz):
        h = float(heights[i]) if heights is not None else float(i)
        imshow_afm(ax, data[:, :, i], extent=extent, cmap=cmap, title=f'h={h:.1f}Å', transpose=transpose)
        ax.tick_params(labelsize=5)
    if ylabel or label:
        axes[0].set_ylabel(ylabel or label, fontsize=8)
    fig.suptitle(label, fontsize=11)
    fig.tight_layout()
    if fname is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved {path}")
    return fig, axes


def afm_panel_clim(arr, *, scale='per_image', pct=99.0, symmetric=False):
    """Color limits for one AFM XY panel.

    scale:
      'per_image' — this array only (experimental-style relative contrast)
      (common / per_column handled by caller who passes a pooled array)
    symmetric: if True, return ±max(|lo|,|hi|) (diverging); else independent min/max.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return -1e-30, 1e-30
    lo = float(np.percentile(a, 100.0 - float(pct)))
    hi = float(np.percentile(a, float(pct)))
    if not np.isfinite(lo): lo = float(np.nanmin(a)) if np.isfinite(np.nanmin(a)) else -1e-30
    if not np.isfinite(hi): hi = float(np.nanmax(a)) if np.isfinite(np.nanmax(a)) else 1e-30
    if hi <= lo:
        hi = lo + 1e-30
    if symmetric:
        v = max(abs(lo), abs(hi), 1e-30)
        return -v, v
    return lo, hi


def plot_afm_variant_height_strip(variants, row_specs, heights, out_path, *,
                                  scale='per_image', title='', dpi=140, pct=99.0,
                                  colorbar=True, figsize_col=1.55, figsize_row=1.35,
                                  amp=None):
    """Multi-row × multi-height AFM compare strip (df/Fz × methods).

    Args:
        variants: dict key → {'df': (nx,ny,nz), 'Fz': (nx,ny,nz), ...}
        row_specs: list of (qty, key, ylabel, cmap) e.g. ('df','cube','df cube','gray')
        heights: (nz,)  — probe heights; if both df and Fz rows are shown, remember
          df(h) mixes Fz over ±amp (default panel amp=1.0 Å) so morphologies shift in z.
          See compute_df_amp docstring / Fukui_FDBM_panel_notes_2026-07-23.md.
        out_path: save path (.png)
        scale:
          'per_image'  — each panel its own min/max (experimental relative contrast)
          'per_column' — shared clim across all variants at the same height+qty family
          'common'     — one clim for all panels of the same qty (df vs Fz separate)
        amp: if set, annotate title with peak oscillation amplitude [Å]
        pct: percentile for clim (99 ≈ robust min/max)
    """
    heights = np.asarray(heights, dtype=np.float64)
    n_h = len(heights)
    n_r = len(row_specs)
    fig, axes = plt.subplots(n_r, n_h,
                             figsize=(figsize_col * n_h + 1.4, figsize_row * n_r + 1.2),
                             squeeze=False)

    # Precompute common / per_column clims
    qty_keys = {}
    for qty, key, _, _ in row_specs:
        qty_keys.setdefault(qty, []).append(key)

    common_clim = {}
    if scale == 'common':
        for qty, keys in qty_keys.items():
            stack = np.concatenate([np.asarray(variants[k][qty], dtype=np.float64).ravel() for k in keys])
            common_clim[qty] = afm_panel_clim(stack, pct=pct, symmetric=(qty != 'df'))

    for ih in range(n_h):
        col_clim = {}
        if scale == 'per_column':
            for qty, keys in qty_keys.items():
                stack = np.concatenate([
                    np.asarray(variants[k][qty][:, :, ih], dtype=np.float64).ravel() for k in keys])
                col_clim[qty] = afm_panel_clim(stack, pct=pct, symmetric=(qty != 'df'))

        for ir, (qty, key, ylab, cmap) in enumerate(row_specs):
            ax = axes[ir, ih]
            arr = np.asarray(variants[key][qty][:, :, ih], dtype=np.float64)
            if scale == 'per_image':
                # Experimental: relative contrast within this image (not zero-forced)
                vmin, vmax = afm_panel_clim(arr, pct=pct, symmetric=False)
            elif scale == 'per_column':
                vmin, vmax = col_clim[qty]
            else:
                vmin, vmax = common_clim[qty]
            im = ax.imshow(arr.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if ir == 0:
                ax.set_title(f'h={heights[ih]:.2f}Å', fontsize=8)
            if ih == 0:
                ax.set_ylabel(ylab, fontsize=7)
            if colorbar and ih == n_h - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    scale_note = {
        'per_image': 'color scale = per panel min/max (relative contrast)',
        'per_column': 'color scale = shared per column (variants comparable)',
        'common': 'color scale = common across all panels of same qty',
    }.get(scale, scale)
    amp_note = ''
    if amp is not None:
        amp_note = (f'  |  df amp={float(amp):.2f}Å peak → mixes Fz over ±amp '
                    f'(closest≈h−amp); do not equate Fz(h) with df(h)')
    fig.suptitle(f'{title}\n{scale_note}{amp_note}', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f'REVIEW: {out_path}')
    return out_path


def plot_afm_df_Fz_tworow(
    df_relax, Fz_unrelax, heights_tip, heights_probe, extent=None, *,
    title=None, bond_length=None, cmap_df='gray', cmap_Fz='seismic',
    figsize_col=2.0, dpi=140, fname=None, save_dir='.', apos=None, Fz_relax=None,
    scan_xs=None, scan_ys=None, view_extent=None,
):
    """ Backward-compat wrapper → `plot_afm_Fz_df_threerow` (omit Fz_relax for 2-row)."""
    return plot_afm_Fz_df_threerow(
        Fz_unrelax, Fz_relax, df_relax, heights_tip, heights_probe, extent=extent,
        title=title, bond_length=bond_length, cmap_df=cmap_df, cmap_Fz=cmap_Fz,
        figsize_col=figsize_col, dpi=dpi, fname=fname, save_dir=save_dir, apos=apos,
        scan_xs=scan_xs, scan_ys=scan_ys, view_extent=view_extent,
    )


def plot_afm_Fz_df_threerow(
    Fz_unrelax, Fz_relax, df_relax, heights_tip, heights_probe, extent=None, *,
    title=None, bond_length=None, amp=None, cmap_df='gray', cmap_Fz='seismic',
    figsize_col=1.85, dpi=140, fname=None, save_dir='.', apos=None,
    scan_xs=None, scan_ys=None, view_extent=None,
):
    """SSOT: up to 3×nz panel — Fz_unrelax / Fz_relax / df at probe plane.

    Tip / lever: column i has tip-apex `heights_tip[i]`, probe `heights_probe[i]` (= tip − L).
    All maps at the **probe** plane. Pass `Fz_relax=None` to skip the middle row (2-row mode).

    Row order (top→bottom): Fz unrelax (GridFF) · Fz relax (PP OutFz) · df (amp-averaged).

    Compare consistency:
      - `view_extent=[xmin,xmax,ymin,ymax]` crops all rows to the same XY window (use FDBM
        extent when overlaying Kriging). Prefer over mismatched full GridFF boxes.
      - `apos` must be **atoms** (natoms×2/3), never Kriging `points_clean` sample sites.
    """
    Fu, Fr, df = Fz_unrelax, Fz_relax, df_relax
    if view_extent is not None:
        if scan_xs is None or scan_ys is None:
            raise ValueError('plot_afm_Fz_df_threerow: view_extent requires scan_xs and scan_ys')
        Fu, xs_c, ys_c, extent = crop_afm_xy(Fu, scan_xs, scan_ys, view_extent)
        if Fr is not None:
            Fr, _, _, _ = crop_afm_xy(Fr, scan_xs, scan_ys, view_extent)
        if df is not None:
            df, _, _, _ = crop_afm_xy(df, scan_xs, scan_ys, view_extent)
    elif extent is None and scan_xs is not None and scan_ys is not None:
        extent = scan_extent(scan_xs, scan_ys)

    nz = int(Fu.shape[2])
    assert len(heights_tip) == nz and len(heights_probe) == nz
    rows = [('Fz unrelax', Fu, cmap_Fz)]
    if Fr is not None:
        assert Fr.shape[2] == nz
        rows.append(('Fz relax', Fr, cmap_Fz))
    if df is not None:
        assert df.shape[2] == nz
        rows.append(('df relax', df, cmap_df))
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, nz, figsize=(figsize_col * nz, 2.15 * nrows), squeeze=False)
    Lnote = f'  L={float(bond_length):.2f}Å' if bond_length is not None else ''
    Anote = f'  amp={float(amp):.2f}Å' if amp is not None else ''
    if title is None:
        title = f'Fz unrelax / Fz relax / df{Lnote}{Anote}'
    fig.suptitle(title, fontsize=9)
    for i in range(nz):
        ht, hp = float(heights_tip[i]), float(heights_probe[i])
        for row, (ylab, arr, cmap) in enumerate(rows):
            ax = axes[row, i]
            imshow_afm(ax, arr[:, :, i], extent=extent, cmap=cmap, colorbar=(i == nz - 1),
                       title=(f'tip={ht:.1f}  probe={hp:.1f}' if row == 0 else ''),
                       transpose=True)
            if apos is not None:
                ax.plot(np.asarray(apos)[:, 0], np.asarray(apos)[:, 1], 'c.', ms=1.2, alpha=0.55)
            if view_extent is not None:
                ax.set_xlim(float(view_extent[0]), float(view_extent[1]))
                ax.set_ylim(float(view_extent[2]), float(view_extent[3]))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(ylab, fontsize=8)
    fig.tight_layout()
    if fname is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved {path}")
    return fig, axes


def plot_afm_z_profiles(z, profiles, xlabel='z [Å]', ylabel='value', title=None, fname=None, save_dir='.', ax=None, dpi=150):
    """SSOT: 1D E(z) / Fz(z) / df(z) curves.

    Args:
        z: (nz,) heights
        profiles: dict name->1d array, or list of (y, label, plot_kwargs)
        fname: if set, save under save_dir and close
    Returns:
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    if isinstance(profiles, dict):
        items = [(y, name, {}) for name, y in profiles.items()]
    else:
        items = []
        for p in profiles:
            if len(p) == 2: items.append((p[0], p[1], {}))
            else: items.append((p[0], p[1], p[2]))
    for y, lab, kw in items:
        ax.plot(z, y, label=lab, **kw)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    if fname is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved {path}")
    return fig, ax


# ── Canonical FDBM vs reference z-layout (arbitrary probe sites) ───────────────
# SSOT for diagnostic E(z)/V(z) panels (Import_KrigingGridFF / DFT↔DFTB).
# Normalization: E − E(z=6 Å), V − V(z=8 Å). Yellow band marks z ≥ z_well.
#
# Tip / probe (PP lever): tip_z = probe_z + bond_length. AFM GridFF / FDBM volumes
# are sampled at the **probe** (O apex) plane. df uses oscillation amp (peak) so
# closest approach ≈ probe_z − amp — see skill:afm-plotting.

_DEFAULT_SITE_COLORS = ('tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:red', 'tab:brown', 'tab:pink', 'tab:cyan')


def normalize_probe_sites(sites):
    """Accept list of (name, xy[, color]) or dicts {name, xy, color?} → list of (name, xy[2], color)."""
    out = []
    for i, s in enumerate(sites):
        if isinstance(s, dict):
            name = s['name']
            xy = np.asarray(s['xy'], float).ravel()[:2]
            c = s.get('color', _DEFAULT_SITE_COLORS[i % len(_DEFAULT_SITE_COLORS)])
        else:
            name = s[0]
            xy = np.asarray(s[1], float).ravel()[:2]
            c = s[2] if len(s) > 2 else _DEFAULT_SITE_COLORS[i % len(_DEFAULT_SITE_COLORS)]
        out.append((str(name), xy.copy(), c))
    if not out:
        raise ValueError('normalize_probe_sites: empty sites')
    return out


def fdbm_probe_sites_from_indices(apos, indices, names=None, colors=None):
    """Build probe site list from atom indices (any molecule / any count).

    Args:
        apos: (natoms, 3)
        indices: sequence of atom indices
        names: optional labels (default atom{i})
        colors: optional colors (cycles tab: palette)
    """
    apos = np.asarray(apos, float)
    indices = [int(i) for i in indices]
    if names is None:
        names = [f'atom{i}' for i in indices]
    if colors is None:
        colors = [_DEFAULT_SITE_COLORS[k % len(_DEFAULT_SITE_COLORS)] for k in range(len(indices))]
    if not (len(names) == len(indices) == len(colors)):
        raise ValueError('fdbm_probe_sites_from_indices: names/indices/colors length mismatch')
    return [(names[k], apos[indices[k], :2].copy(), colors[k]) for k in range(len(indices))]


def fdbm_probe_sites_nch(apos, Zs, colors=None):
    """Pyridine helper: N, farthest C from N (para-C), farthest H from N (para-H).

    Opposite-C is carbon (Z==6), never H. For general molecules use
    `fdbm_probe_sites_from_indices` or pass any list to `normalize_probe_sites`.
    """
    apos = np.asarray(apos, float)
    Zs = np.asarray(Zs)
    if colors is None:
        colors = {'N': 'tab:blue', 'para-C': 'tab:green', 'para-H': 'tab:purple'}
    iN = int(np.where(np.isclose(Zs, 7))[0][0])
    iCs = np.where(np.isclose(Zs, 6))[0]
    iHs = np.where(np.isclose(Zs, 1))[0]
    iC = int(iCs[np.argmax(np.sum((apos[iCs, :2] - apos[iN, :2]) ** 2, axis=1))])
    iH = int(iHs[np.argmax(np.sum((apos[iHs, :2] - apos[iN, :2]) ** 2, axis=1))])
    return [
        ('N', apos[iN, :2].copy(), colors['N']),
        ('para-C', apos[iC, :2].copy(), colors['para-C']),
        ('para-H', apos[iH, :2].copy(), colors['para-H']),
    ]


def afm_tip_probe_heights(tip_min, tip_max, tip_step, bond_length):
    """Tip-apex ladder and matching probe heights (probe = tip − L).

    Returns (heights_tip, heights_probe) float arrays.
    """
    tip = np.arange(float(tip_min), float(tip_max) + 0.5 * float(tip_step), float(tip_step))
    L = float(bond_length)
    return tip, tip - L


def sample_field_z_profile(F, origin, step, xy, zs, zref=None, order=1):
    """Sample 3D field along z at fixed XY; optional subtract F(zref).

    order=1 (default): trilinear via map_coordinates — smooth curves for plots.
    order=0: nearest voxel (stair-steps; avoid for diagnostics).
    """
    from scipy.ndimage import map_coordinates
    F = np.asarray(F, float)
    o = np.asarray(origin, float).ravel()[:3]
    s = float(np.asarray(step).reshape(-1)[0])
    zs = np.asarray(zs, float)
    fx = np.full(len(zs), (xy[0] - o[0]) / s)
    fy = np.full(len(zs), (xy[1] - o[1]) / s)
    fz = (zs - o[2]) / s
    out = map_coordinates(F, np.vstack([fx, fy, fz]), order=int(order), mode='nearest')
    if zref is not None:
        out = out - out[np.argmin(np.abs(zs - float(zref)))]
    return out


def _shade_zwell(ax, ylim, z0=2.5, z1=None):
    if z1 is None:
        z1 = ax.get_xlim()[1]
    ax.fill_betweenx([-ylim, ylim], z0, z1, color='yellow', alpha=0.08)


def plot_fdbm_rho_E_sites(
    sites, rho_curves, E_curves, *, mol_z=0.0,
    z_rho=(-2.0, 2.0), z_E=(1.0, 6.0), zref_E=6.0, z_well=2.5,
    ylim_E=0.2, n_z=321, title=None, figsize=None, fname=None, save_dir='.', dpi=140,
):
    """Two-row ρ + E z-profiles for a short probe-site list (default: N and para-H).

    rho_curves / E_curves: list of (label, field, origin, step, style_dict).
    style_dict keys go to Axes.plot (color, ls, lw, alpha, …).
    E curves are zeroed at mol_z+zref_E. Yellow band on E: z ≥ mol_z+z_well.
    ρ z-axis: [mol_z+z_rho[0], mol_z+z_rho[1]]; E: [mol_z+z_E[0], mol_z+z_E[1]].
    """
    sites = normalize_probe_sites(sites)
    n = len(sites)
    if figsize is None:
        figsize = (max(7.5, 3.6 * n), 7.0)
    zs_r = np.linspace(mol_z + float(z_rho[0]), mol_z + float(z_rho[1]), int(n_z))
    zs_e = np.linspace(mol_z + float(z_E[0]), mol_z + float(z_E[1]), int(n_z))
    zref = mol_z + float(zref_E)
    fig, axes = plt.subplots(2, n, figsize=figsize, sharex='row', squeeze=False)
    if title:
        fig.suptitle(title, fontsize=10)
    for ic, (name, xy, _col) in enumerate(sites):
        ax = axes[0, ic]
        for lab, F, origin, step, sty in rho_curves:
            ax.plot(zs_r, sample_field_z_profile(F, origin, step, xy, zs_r),
                    label=lab, **(sty or {}))
        ax.axhline(0, color='k', lw=0.4)
        ax.axvline(mol_z, color='gray', ls=':', lw=0.8)
        ax.set_title(f'{name}  xy=({xy[0]:+.2f},{xy[1]:+.2f})', fontsize=9)
        ax.set_xlim(zs_r[0], zs_r[-1])
        if ic == 0:
            ax.set_ylabel('ρ [e/Å³]', fontsize=8)
            ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)

        ax = axes[1, ic]
        for lab, F, origin, step, sty in E_curves:
            ax.plot(zs_e, sample_field_z_profile(F, origin, step, xy, zs_e, zref=zref),
                    label=lab, **(sty or {}))
        ax.axhline(0, color='k', lw=0.4)
        ax.axvspan(mol_z + float(z_well), zs_e[-1], color='yellow', alpha=0.08)
        ax.set_ylim(-float(ylim_E), float(ylim_E))
        ax.set_xlim(zs_e[0], zs_e[-1])
        ax.set_xlabel('z [Å]')
        if ic == 0:
            ax.set_ylabel(f'E−E(+{float(zref_E):.0f}) [eV] (±{ylim_E})', fontsize=8)
            ax.legend(fontsize=6, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94 if title else 0.98])
    if fname:
        import os
        path = fname if os.path.isabs(fname) else os.path.join(save_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return path
    return fig


def plot_fdbm_vs_kriging_zlayout(
    sites, origin, step, V_ES, E_pauli, E_es, E_vdw, E_tot, *,
    kriging_E=None, sites_krig=None, A_pauli=None, beta_pauli=None,
    title=None, title_extra='', zs=None, zs_v=None, z_well=2.5,
    ylim_V=0.2, ylim_Ees=0.05, ylim_PauliTot=0.1, ylim_site=0.1,
    zref_E=6.0, zref_V=8.0, figsize=None, fname=None, save_dir='.', dpi=140,
):
    """Canonical FDBM vs Kriging z-profile layout for **any number of probe sites**.

    Top row (3 panels, sites as colored lines):
      V_ES (±ylim_V) · E_es (±ylim_Ees) · E_Pauli(dashed)+E_tot(solid) (±ylim_PauliTot, lw=0.5)
    Bottom row (**one panel per site**): Kriging, E_tot, E_Pauli, E_es, E_vdW (±ylim_site)

    Energies zeroed at zref_E; V at zref_V. Yellow band = z ≥ z_well.

    Args:
        sites: any length — list of (name, xy[, color]) or dicts; see `normalize_probe_sites`
        kriging_E: callable (xy, zs)->1d already zeroed at zref_E, or None
        sites_krig: optional Kriging XY sites (same order/length as sites)
    """
    sites = normalize_probe_sites(sites)
    n_sites = len(sites)
    o = np.asarray(origin, float)
    s = float(np.asarray(step).reshape(-1)[0])
    if zs is None:
        zs = np.arange(1.6, 6.01, 0.1)
    if zs_v is None:
        zs_v = np.arange(1.5, 8.01, 0.1)
    if sites_krig is None:
        sites_krig = sites
    else:
        sites_krig = normalize_probe_sites(sites_krig)
        if len(sites_krig) != n_sites:
            raise ValueError(f'sites_krig length {len(sites_krig)} != sites {n_sites}')

    ncols = max(3, n_sites)
    if figsize is None:
        figsize = (max(11.0, 3.2 * ncols), 7.5)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.0, 1.05], hspace=0.32, wspace=0.28)
    ab = ''
    if A_pauli is not None and beta_pauli is not None:
        ab = f'  A={float(A_pauli):.2f} β={float(beta_pauli):.3f}'
    if title is None:
        title = (
            f'FDBM vs Kriging  |  E−E({zref_E:g}), V−V({zref_V:g})  |  n_sites={n_sites}\n'
            f'Top: V_ES ±{ylim_V} · E_es ±{ylim_Ees} · E_Pauli(dashed)+E_tot(solid) ±{ylim_PauliTot}  |  '
            f'Bottom: per-site ±{ylim_site}{ab}  {title_extra}'
        )
    fig.suptitle(title, fontsize=9)

    # Top: 3 component panels (left-aligned if ncols > 3)
    ax = fig.add_subplot(gs[0, 0])
    for name, xy, c in sites:
        ax.plot(zs_v, sample_field_z_profile(V_ES, o, s, xy, zs_v, zref_V), color=c, lw=1.5, label=name)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(z_well, color='gray', ls=':', lw=1)
    ax.set_xlim(float(zs_v[0]), float(zs_v[-1])); ax.set_ylim(-ylim_V, ylim_V)
    ax.set_title('V_ES (sites overlapped)'); ax.set_xlabel('z [Å]'); ax.set_ylabel(f'V−V({zref_V:g}) [eV]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); _shade_zwell(ax, ylim_V, z_well, float(zs_v[-1]))

    ax = fig.add_subplot(gs[0, 1])
    for name, xy, c in sites:
        ax.plot(zs, sample_field_z_profile(E_es, o, s, xy, zs, zref_E), color=c, lw=1.5, label=name)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(z_well, color='gray', ls=':', lw=1)
    ax.set_xlim(float(zs[0]), float(zs[-1])); ax.set_ylim(-ylim_Ees, ylim_Ees)
    ax.set_title('E_es (sites overlapped)'); ax.set_xlabel('z [Å]'); ax.set_ylabel(f'E_es−E({zref_E:g}) [eV]')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); _shade_zwell(ax, ylim_Ees, z_well)

    ax = fig.add_subplot(gs[0, 2])
    for name, xy, c in sites:
        ax.plot(zs, sample_field_z_profile(E_tot, o, s, xy, zs, zref_E), color=c, lw=0.5, ls='-', label=f'{name} E_tot')
        ax.plot(zs, sample_field_z_profile(E_pauli, o, s, xy, zs, zref_E), color=c, lw=0.5, ls='--', label=f'{name} E_Pauli')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(z_well, color='gray', ls=':', lw=1)
    ax.set_xlim(float(zs[0]), float(zs[-1])); ax.set_ylim(-ylim_PauliTot, ylim_PauliTot)
    ax.set_title('E_Pauli (dashed) + E_tot (solid)'); ax.set_xlabel('z [Å]'); ax.set_ylabel(f'E−E({zref_E:g}) [eV]')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3); _shade_zwell(ax, ylim_PauliTot, z_well)

    # Bottom: one panel per site
    for i, ((name, xy, c), (_, xyk, _)) in enumerate(zip(sites, sites_krig)):
        ax = fig.add_subplot(gs[1, i])
        ax.axhline(0, color='k', lw=0.5); ax.axvline(z_well, color='gray', ls=':', lw=1)
        if kriging_E is not None:
            ax.plot(zs, kriging_E(xyk, zs), 'k-', lw=2.2, label='Kriging')
        ax.plot(zs, sample_field_z_profile(E_tot, o, s, xy, zs, zref_E), color='k', lw=1.6, ls='--', label='E_tot')
        ax.plot(zs, sample_field_z_profile(E_pauli, o, s, xy, zs, zref_E), color='tab:blue', lw=1.5, label='E_Pauli')
        ax.plot(zs, sample_field_z_profile(E_es, o, s, xy, zs, zref_E), color='tab:red', lw=1.5, label='E_es')
        ax.plot(zs, sample_field_z_profile(E_vdw, o, s, xy, zs, zref_E), color='tab:green', lw=1.5, label='E_vdW')
        ax.set_xlim(float(zs[0]), float(zs[-1])); ax.set_ylim(-ylim_site, ylim_site)
        ax.set_title(name); ax.set_xlabel('z [Å]'); ax.set_ylabel(f'E−E({zref_E:g}) [eV]')
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3); _shade_zwell(ax, ylim_site, z_well)

    if fname is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved {path}")
    return fig, None


def plot_fdbm_methods_zcompare_4panel(
    methods, sites_per_method, *, keys=('V_ES', 'E_es', 'E_pauli', 'E_tot'),
    styles=None, zs=None, zs_v=None, z_well=2.5,
    ylims=None, zref_E=6.0, zref_V=8.0, title=None, figsize=(11, 8),
    fname=None, save_dir='.', dpi=140,
):
    """4-panel overlay of FDBM methods (e.g. DFT-cube solid vs DFTB dashed).

    methods: list of dicts with keys V_ES, E_es, E_pauli, E_tot, origin, step, label
    sites_per_method: list of site lists parallel to methods (any site count; same chemical order)
    styles: list of linestyle for each method (default ['-', '--', ...])
    """
    n_m = len(methods)
    if styles is None:
        styles = ['-', '--', ':', '-.'][:n_m]
    if zs is None:
        zs = np.arange(1.6, 6.01, 0.1)
    if zs_v is None:
        zs_v = np.arange(1.5, 8.01, 0.1)
    if ylims is None:
        ylims = {'V_ES': 0.2, 'E_es': 0.05, 'E_pauli': 0.1, 'E_tot': 0.1}
    sites_per_method = [normalize_probe_sites(s) for s in sites_per_method]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    if title is None:
        labs = ' / '.join(m.get('label', f'm{i}') for i, m in enumerate(methods))
        title = f'FDBM method compare ({labs})  |  site color · line style = method\nE−E({zref_E:g}), V−V({zref_V:g})'
    fig.suptitle(title, fontsize=10)
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    ylab = {
        'V_ES': f'V−V({zref_V:g}) [eV]', 'E_es': f'E_es−E({zref_E:g}) [eV]',
        'E_pauli': f'E_Pauli−E({zref_E:g}) [eV]', 'E_tot': f'E_tot−E({zref_E:g}) [eV]',
    }
    for ax, key in zip(ax_list, keys):
        zax = zs_v if key == 'V_ES' else zs
        zref = zref_V if key == 'V_ES' else zref_E
        ylim = float(ylims.get(key, 0.1))
        for m, sites, ls in zip(methods, sites_per_method, styles):
            lab = m.get('label', '')
            for name, xy, c in sites:
                y = sample_field_z_profile(m[key], m['origin'], m['step'], xy, zax, zref)
                ax.plot(zax, y, color=c, lw=1.4, ls=ls, label=f'{name} {lab}'.strip())
        ax.axhline(0, color='k', lw=0.5); ax.axvline(z_well, color='gray', ls=':', lw=1)
        ax.set_xlim(float(zax[0]), float(zax[-1])); ax.set_ylim(-ylim, ylim)
        ax.set_title(key); ax.set_xlabel('z [Å]'); ax.set_ylabel(ylab.get(key, 'E [eV]'))
        ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3); _shade_zwell(ax, ylim, z_well, float(zax[-1]))
    fig.tight_layout()
    if fname is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches='tight'); plt.close(fig)
        print(f"  Saved {path}")
    return fig, axes


def plot_xy_slice(data, origin, step, iz, title, fname, save_dir, sym=False, cmap='magma'):
    """Plot xy slice at given z-index. Prefer `imshow_afm` / `plot_afm_height_panel` for new code."""
    nx, ny = data.shape[0], data.shape[1]
    extent = [origin[0], origin[0] + nx * step, origin[1], origin[1] + ny * step]
    fig, ax = plt.subplots(figsize=(6, 5))
    imshow_afm(ax, data[:, :, iz], extent=extent, cmap=cmap, symmetric=sym, title=title)
    ax.set_xlabel('x [A]'); ax.set_ylabel('y [A]')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def save_afm_images(df, scan_xs, scan_ys, heights, out_dir, prefix='df', cmap='afmhot'):
    """Save AFM frequency-shift (or Fz/E) images at all heights. SSOT for full-height PNG dumps.

    Args:
        df: (nx, ny, nz) array
        scan_xs, scan_ys: 1D scan coordinate arrays
        heights: 1D probe height array
        out_dir: directory for PNG output
        prefix: filename prefix (e.g. 'df' -> df_h3.0.png)
    """
    ext = scan_extent(scan_xs, scan_ys)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(heights)):
        h = heights[i]
        fig, ax = plt.subplots(figsize=(5, 4))
        imshow_afm(ax, df[:, :, i], extent=ext, cmap=cmap, symmetric=(cmap in ('bwr', 'seismic', 'RdBu_r')), title=f"{prefix} at h={h:.1f} A")
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
        fname = os.path.join(out_dir, f"{prefix}_h{h:.1f}.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")


def plot_slices(data, title, fname, sym=False, cmap='magma', save_dir='.'):
    """Plot central XY/XZ/YZ slices + 1D profiles of a 3D field."""
    nx, ny, nz = data.shape
    cx, cy, cz = nx//2, ny//2, nz//2
    if sym: cmap = 'bwr'
    fig, axes = plt.subplots(2, 3, figsize=(16, 8)); fig.suptitle(title)
    norm = safe_norm(data) if sym else None
    kw = dict(origin='lower', cmap=cmap, aspect='auto', norm=norm)
    for ax, sl, tl in zip(axes[0],
        [data[cx,:,:].T, data[:,cy,:].T, data[:,:,cz].T],
        [f'ix={cx} (YZ)', f'iy={cy} (XZ)', f'iz={cz} (XY)']):
        im = ax.imshow(sl, **kw); ax.set_title(tl); plt.colorbar(im, ax=ax, shrink=0.8)
    axes[1,0].plot(data[cx,cy,:]); axes[1,0].set_xlabel('iz'); axes[1,0].set_title('z-profile center')
    axes[1,1].plot(data[:,cy,cz]); axes[1,1].set_xlabel('ix'); axes[1,1].set_title('x-profile center')
    axes[1,2].plot(data[cx,:,cz]); axes[1,2].set_xlabel('iy'); axes[1,2].set_title('y-profile center')
    for ax in axes[1]: ax.axhline(0, color='k', lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=90, bbox_inches='tight'); plt.close()
    print(f"Saved {fname}")


def plot_grid_Fz(Fz, heights, label, fname, x_ext=None, y_ext=None, ncols=7, save_dir='.', cmap='seismic'):
    """SSOT: full grid of 2D Fz/df/E images at all heights with per-slice colorbars."""
    nz_p = len(heights)
    nrows = int(np.ceil(nz_p / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), squeeze=False,
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.15})
    axes = np.array(axes).reshape(nrows, ncols)
    fig.suptitle(f"{label} (eV/Å) [per-slice]", fontsize=10)
    ext = [x_ext[0], x_ext[1], y_ext[0], y_ext[1]] if x_ext is not None and y_ext is not None else None
    for k in range(nz_p):
        r, c = divmod(k, ncols); ax = axes[r, c]
        vabs = max(float(np.percentile(np.abs(Fz[:,:,k]), 99)), 1e-6)
        imshow_afm(ax, Fz[:,:,k], extent=ext, cmap=cmap, title=f"h={heights[k]:.1f}Å ±{vabs:.2g}")
        ax.tick_params(labelsize=5)
        ax.title.set_fontsize(7)
    for k in range(nz_p, nrows*ncols):
        r, c = divmod(k, ncols); axes[r, c].set_visible(False)
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# Density Provider Adapters (standard interface)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_grid_spec(atomPos, step, margin, z_extra):
    """Build grid_spec + return (grid_spec, origin, ngrid, step). Wraps afm.setup_density_grid."""
    grid_spec, origin, ngrid = afm.setup_density_grid(atomPos, step=step, margin=margin, z_extra=z_extra)
    return grid_spec, origin, ngrid, step


def _project_densities(geo, evecs, basis, grid_spec, verbosity=0):
    """Shared projection logic: returns (rho_scf, rho_na, rho_diff) using Grid_dftb backends."""
    from spammm.quantum.DFTB import Grid_dftb as dg
    dftb_data = {k: geo[k] for k in ('coords_bohr', 'species_per_atom', 'species_names')}
    projector, atoms_dict = dg.setup_gridprojector_from_dftb(dftb_data, basis, verbosity=verbosity)
    rho_scf = dg.project_dftb_density(geo, evecs, projector, atoms_dict, grid_spec, basis)
    rho_na  = dg.project_neutral_density(geo, projector, atoms_dict, grid_spec, basis)
    return rho_scf, rho_na, (rho_scf - rho_na).astype(np.float32)


def build_orbital_layout(basis_data, enames):
    """Build norb_per_atom and orb_offsets from basis data.

    Args:
        basis_data: dict from parse_wfc_hsd (keys are element names)
        enames: list of element names for each atom

    Returns:
        norb_per_atom: (natoms,) number of orbitals per atom
        orb_offsets: (natoms+1,) cumulative orbital offsets
        max_l: maximum angular momentum in system
    """
    norb_per_atom = []
    orb_offsets = [0]
    max_l = 0
    for name in enames:
        sp = basis_data[name]
        norb = sum(2 * orb['AngularMomentum'] + 1 for orb in sp['orbitals'])
        for orb in sp['orbitals']:
            max_l = max(max_l, orb['AngularMomentum'])
        norb_per_atom.append(norb)
        orb_offsets.append(orb_offsets[-1] + norb)
    return (np.array(norb_per_atom, dtype=np.int32),
            np.array(orb_offsets, dtype=np.int32),
            max_l)


def get_density_from_dftb_dense(atomPos, atomTypes, basis_hsd_path, work_dir,
                                 grid_spec=None, step=0.1, margin=4.0, z_extra=6.0,
                                 verbosity=0, max_shells=None, projection_basis_ang=None,
                                 project_density=True):
    """Get density grids using DFTBcore dense matrix projection (supports d-orbitals).

    Uses direct DFTBcore library access (no file parsing) and dense density matrix
    projection, enabling support for d-orbitals (e.g., Br in 3ob-3-1 basis).

    Args:
        atomPos: (natoms, 3) positions in Angstrom
        atomTypes: (natoms,) atomic numbers
        basis_hsd_path: path to basis HSD file (e.g., 'wfc.3ob-3-1.hsd')
        work_dir: DFTB+ scratch directory for SCF
        grid_spec: dict with 'origin', 'dA', 'dB', 'dC', 'ngrid' (optional)
        step/margin/z_extra: grid parameters (used if grid_spec is None)
        verbosity: logging level
        max_shells: int (2=sp, 3=spd); auto-detected from basis if None
        projection_basis_ang: optional species_list to override basis used for **Pauli**
            projection only (SCF still uses stock mio/3ob). Typical: result of
            `make_slater_tail_species_list` / SA-prolonged STOs.

            DUAL BASIS (mandatory — do not "fix"):
              - Prolonged ρ is for Pauli overlap only. Never charge-normalize it;
                ∫ρ ≉ N_e by design; A,β absorb overall scale.
              - Electrostatics MUST use stock Δρ → V_ES (call this function twice
                or keep stock ρ_diff / V_ES from a separate stock projection).
              - See `make_slater_tail_species_list` docstring and `doc/DFTB_basis_fit.md`.
        project_density: if False, skip ρ/V_ES (STM orbital-only path still returns
            eigvecs + projector).

    Returns:
        dict with 'rho_scf', 'rho_na', 'rho_diff', 'V_ES', 'origin', 'ngrid', 'grid_spec'
        (density keys None when project_density=False)
    """
    from spammm.quantum.DFTB.DFTBcore import DFTBcore
    from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd, convert_wfc_to_species_list_ang
    from spammm.quantum.DFTB import Grid_dftb as dg
    from spammm import atomicUtils as au
    import multiprocessing as mp
    import shutil

    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'P':15,'S':16,'Br':35,'I':53}
    inv_z = {v:k for k,v in ELEM_Z.items()}
    enames = [inv_z.get(int(z), 'C') for z in atomTypes]

    # Ensure work_dir exists (use absolute path for subprocess)
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Setup grid
    if grid_spec is None:
        grid_spec, origin, ngrid, step = _make_grid_spec(atomPos, step, margin, z_extra)
    else:
        origin, ngrid, step = grid_spec['origin'], grid_spec['ngrid'], grid_spec['dA'][0]

    # Load basis
    basis_data = parse_wfc_hsd(basis_hsd_path)
    basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)

    # Build orbital layout
    norb_per_atom, orb_offsets, max_l = build_orbital_layout(basis_data, enames)
    if max_shells is None:
        max_shells = 3 if max_l >= 2 else 2

    # Prepare DFTB data for projector and neutral density
    coords_bohr = atomPos * 1.8897259886  # Ang -> Bohr
    species_per_atom = list(range(len(enames)))  # Each atom is unique species index
    dftb_data = {
        'coords_bohr': coords_bohr,
        'species_per_atom': species_per_atom,
        'species_names': enames
    }

    # Setup projector with max_shells for d-orbital support
    proj_basis = projection_basis_ang if projection_basis_ang is not None else basis_ang
    projector, atoms_dict = dg.setup_gridprojector_from_dftb(dftb_data, proj_basis, verbosity=verbosity, max_shells=max_shells)

    # Run DFTBcore SCF directly (single molecule - no Fortran state conflicts expected)
    basis_name = os.path.basename(basis_hsd_path).replace('wfc.', '').replace('.hsd', '')

    # Prepare DFTBcore input (minimal, no Analysis/Options blocks like DFTB+ needs)
    from spammm.quantum.DFTB_utils import SK_PATHS as _SK_PATHS
    sk_dir = _SK_PATHS.get(basis_name, os.path.join(os.environ.get('DFTB_SK_PATH', ''), basis_name))
    xyz_path = os.path.join(work_dir, 'geom.xyz')
    hsd_path = os.path.join(work_dir, 'dftb_in.hsd')

    # Write XYZ file
    au.save_xyz(xyz_path, enames, atomPos)

    # Compute MaxAngularMomentum from basis_data for each element
    species = sorted(set(enames))
    max_am_map = {0: 's', 1: 'p', 2: 'd'}
    max_ang_lines = []
    for elem in species:
        elem_data = basis_data[elem]
        max_l = max(orb['AngularMomentum'] for orb in elem_data['orbitals'])
        max_ang_lines.append(f'    {elem} = "{max_am_map[max_l]}"')

    # Write minimal DFTBcore-compatible HSD (no Analysis/Options blocks)
    max_ang_str = '\n'.join(max_ang_lines)
    with open(hsd_path, 'w') as f:
        f.write(f'''Geometry = xyzFormat {{
  <<< "geom.xyz"
}}
Hamiltonian = DFTB {{
  SCC = Yes
  SCCTolerance = 1e-7
  MaxSCCIterations = 200
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_dir}/"
    Separator = "-"
    Suffix = ".skf"
    LowerCaseTypeName = No
  }}
  MaxAngularMomentum = {{
{max_ang_str}
  }}
}}
''')

    # Copy required SK files to work directory (same as sparse method)
    for i, elem1 in enumerate(species):
        for elem2 in species[i:]:
            for sk_file in [f"{elem1}-{elem2}.skf", f"{elem2}-{elem1}.skf"]:
                src = os.path.join(sk_dir, sk_file)
                if os.path.exists(src):
                    shutil.copy(src, work_dir)

    # Run SCF
    old_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        dftb = DFTBcore()
        dftb.init('dftb_in.hsd')
        dftb.enable_matrix_collection(dm=True, h=False, s=False)
        energy = dftb.run_scf()
        dm_dense = dftb.get_dm_dense()
        eigvecs, eigvals = dftb.get_eigvecs_dense()  # Get eigenvectors for STM
        dftb.finalize()
        # Note: DM is in non-orthogonal basis, GPU kernel handles this correctly

    finally:
        os.chdir(old_cwd)

    rho_scf = rho_na = rho_diff = V_ES = None
    if project_density:
        # Project SCF density using dense method (supports d-orbitals)
        rho_scf = projector.project_density_dense(dm_dense.astype(np.float32), norb_per_atom, orb_offsets, atoms_dict, grid_spec)

        # Build geo dict for neutral density projection (sparse method)
        geo = {
            'natoms': len(enames),
            'species_per_atom': species_per_atom,
            'species_names': enames,
            'coords_bohr': coords_bohr
        }
        # Neutral-atom density: diagonal NA DM → one project_density_dense (same physics as AO loop)
        rho_na = dg.project_neutral_density(
            geo, projector, atoms_dict, grid_spec, proj_basis,
            norb_per_atom=norb_per_atom, orb_offsets=orb_offsets)

        rho_diff = (rho_scf - rho_na).astype(np.float32)

        # CRITICAL: Check charge conservation - rho_diff should integrate to ~0
        # Both rho_scf and rho_na should contain the same total number of electrons
        cell_volume = step**3
        q_scf = rho_scf.sum() * cell_volume
        q_na = rho_na.sum() * cell_volume
        q_diff_val = rho_diff.sum() * cell_volume
        print(f"  [CHARGE CHECK] step={step:.3f} Å, cell_vol={cell_volume:.6f} Å³")
        print(f"  [CHARGE CHECK] rho_scf.sum={rho_scf.sum():.1f}, rho_na.sum={rho_na.sum():.1f}")
        print(f"  [CHARGE CHECK] q_scf={q_scf:.3f}, q_na={q_na:.3f}, q_diff={q_diff_val:.6f} (should be ~0)")
        if abs(q_diff_val) > 2.0:  # More than 2.0 electron discrepancy is serious
            print(f"  WARNING: Large charge imbalance in rho_diff! Electrostatics may be unreliable.")
            print(f"           Consider increasing grid resolution or checking basis consistency.")

        V_ES = afm.fft_poisson(rho_diff, step)

    return {'rho_scf': rho_scf, 'rho_na': rho_na, 'rho_diff': rho_diff, 'V_ES': V_ES,
            'origin': origin, 'ngrid': ngrid, 'grid_spec': grid_spec,
            'eigvecs': eigvecs, 'eigvals': eigvals, 'dm': dm_dense,
            'norb_per_atom': norb_per_atom, 'orb_offsets': orb_offsets, 'atoms_dict': atoms_dict,
            'projector': projector, 'basis_ang': basis_ang, 'dftb_data': dftb_data,
            'enames': enames, 'basis_hsd_path': basis_hsd_path}


# Cache for atomic density matrices (neutral atom density computation)
_ATOMIC_DM_CACHE = {}

def get_density_from_pyscf(atomPos, atomTypes, grid_spec=None, step=0.1, margin=4.0, z_extra=6.0,
                            basis='sto-3g', method='RHF', xc=None, verbosity=0,
                            skip_na=False, use_df=True, mf=None, dm=None):
    """Get density grids using pySCF for SCF and direct grid evaluation (CPU-based).

    This is the Phase 1 pySCF backend: uses pySCF's eval_ao/eval_rho on CPU.
    Phase 2 (GPU-accelerated GTO projection) would use Grid_dftb with GTO kernels.

    Args:
        atomPos: (natoms, 3) positions in Angstrom
        atomTypes: (natoms,) atomic numbers
        grid_spec: dict with 'origin', 'dA', 'dB', 'dC', 'ngrid' (optional)
        step/margin/z_extra: grid parameters (used if grid_spec is None)
        basis: basis set name (default 'sto-3g' for minimal basis)
        method: 'RHF' or 'RKS' (DFT)
        xc: XC functional for DFT (e.g., 'lda,vwn', 'pbe')
        verbosity: pySCF verbosity level (0=silent)
        skip_na: if True, skip ρ_NA / Δρ / V_ES (Pauli-only FDBM needs ρ_scf only)
        use_df: density-fit SCF (much faster for PTCDA-scale)
        mf, dm: optional precomputed mean-field + DM (skip SCF; use mf.mol for AO eval)

    Returns:
        dict with 'rho_scf', 'rho_na', 'rho_diff', 'V_ES', 'origin', 'ngrid', 'grid_spec',
                 'eigvecs', 'eigvals', 'dm', 'mol', 'mf' (latter three for Phase 2 extension)
    """
    from pyscf import gto, scf, dft
    from pyscf.dft import numint
    import time

    BOHR_PER_ANGSTROM = 1.8897259886

    ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16, 'Br': 35, 'I': 53}
    inv_z = {v: k for k, v in ELEM_Z.items()}
    enames = [inv_z.get(int(z), 'C') for z in atomTypes]

    # Setup grid
    if grid_spec is None:
        grid_spec, origin, ngrid, step = _make_grid_spec(atomPos, step, margin, z_extra)
    else:
        origin, ngrid, step = grid_spec['origin'], grid_spec['ngrid'], grid_spec['dA'][0]

    t0 = time.time()

    if mf is not None and dm is not None:
        mol = mf.mol
        eigvecs = getattr(mf, 'mo_coeff', None)
        eigvals = getattr(mf, 'mo_energy', None)
        t1 = time.time()
        print(f"  [pySCF] Using precomputed SCF ({t1-t0:.3f}s), energy={float(mf.e_tot):.6f} Hartree")
    else:
        # Build pySCF molecule from atomPos (Angstrom) and enames
        atom_list = [[enames[i], atomPos[i]] for i in range(len(enames))]
        mol = gto.Mole()
        mol.atom = atom_list
        mol.basis = basis
        mol.verbose = verbosity
        mol.spin = 0
        mol.charge = 0
        mol.build()

        # Run SCF
        if method.upper() == 'RHF':
            mf = scf.RHF(mol)
        elif method.upper() == 'RKS':
            mf = dft.RKS(mol)
            if xc is not None:
                mf.xc = xc
        else:
            raise ValueError(f"Unknown method: {method}. Use 'RHF' or 'RKS'.")
        if use_df:
            mf = mf.density_fit()
        mf.kernel()
        dm = mf.make_rdm1()
        eigvecs = mf.mo_coeff
        eigvals = mf.mo_energy

        t1 = time.time()
        print(f"  [pySCF] SCF converged in {t1-t0:.3f}s, energy={mf.e_tot:.6f} Hartree")

    # Chunked density eval — full AO(npts×nao) for PTCDA/def2-SVP is ~20+ GB
    nx, ny, nz = ngrid
    origin_bohr = origin * BOHR_PER_ANGSTROM
    dA_bohr = np.array(grid_spec['dA']) * BOHR_PER_ANGSTROM
    dB_bohr = np.array(grid_spec['dB']) * BOHR_PER_ANGSTROM
    dC_bohr = np.array(grid_spec['dC']) * BOHR_PER_ANGSTROM
    npts = int(nx) * int(ny) * int(nz)
    chunk = max(8192, min(65536, npts))  # ~0.2 GB AO at nao~460
    rho_scf = np.empty(npts, dtype=np.float32)
    print(f"  [pySCF] Density eval chunked: npts={npts} nao={mol.nao_nr()} chunk={chunk}")
    for i0 in range(0, npts, chunk):
        i1 = min(i0 + chunk, npts)
        idx = np.arange(i0, i1)
        ix = idx // (ny * nz)
        rem = idx % (ny * nz)
        iy = rem // nz
        iz = rem % nz
        pts = (origin_bohr + ix[:, None] * dA_bohr + iy[:, None] * dB_bohr + iz[:, None] * dC_bohr)
        ao = numint.eval_ao(mol, pts, deriv=0)
        rho_scf[i0:i1] = numint.eval_rho(mol, ao, dm, xctype='LDA').astype(np.float32)
        if (i0 // chunk) % 20 == 0:
            print(f"    … {i1}/{npts} ({100.0*i1/npts:.0f}%)")
    rho_scf = rho_scf.reshape(nx, ny, nz)
    t2 = time.time()
    print(f"  [pySCF] Density evaluation: {t2-t1:.3f}s for {npts} points")

    if skip_na:
        rho_na = np.zeros_like(rho_scf)
        rho_diff = rho_scf.copy()
        V_ES = None
        cell_volume = step**3
        print(f"  [pySCF] skip_na=True — ρ_NA / V_ES not computed (Pauli-only)")
        print(f"  [pySCF CHARGE CHECK] q_scf={rho_scf.sum()*cell_volume:.3f}")
    else:
        # Compute neutral atom density (rho_NA) by summing isolated atoms
        rho_na = np.zeros_like(rho_scf)

        # Atomic numbers for determining spin
        ATOMIC_NUMBERS = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}

        # Cache atomic density matrices per element to avoid re-running SCF
        unique_elements = list(set(enames))
        for elem in unique_elements:
            if elem not in _ATOMIC_DM_CACHE:
                # Determine spin for atom (odd electron count = doublet)
                nelec = ATOMIC_NUMBERS.get(elem, 6)
                spin = 1 if nelec % 2 == 1 else 0
                # Run single-atom SCF and cache the density matrix
                atm = gto.M(atom=f'{elem} 0 0 0', basis=basis, verbose=0, spin=spin)
                atm.build()
                mf_atm = scf.RHF(atm)
                mf_atm.kernel()
                dm_atm = mf_atm.make_rdm1()
                _ATOMIC_DM_CACHE[elem] = (atm, dm_atm, spin)

        # Chunked NA density (same grid walk)
        for i, (elem, pos) in enumerate(zip(enames, atomPos)):
            atm_cache, dm_cache, spin = _ATOMIC_DM_CACHE[elem]
            atm_i = gto.M(atom=[[elem, pos]], basis=basis, verbose=0, spin=spin)
            atm_i.build()
            mf_i = scf.RHF(atm_i)
            mf_i.kernel()
            dm_i = mf_i.make_rdm1()
            rho_i = np.empty(npts, dtype=np.float32)
            for i0 in range(0, npts, chunk):
                i1 = min(i0 + chunk, npts)
                idx = np.arange(i0, i1)
                ix = idx // (ny * nz)
                rem = idx % (ny * nz)
                iy = rem // nz
                iz = rem % nz
                pts = (origin_bohr + ix[:, None] * dA_bohr + iy[:, None] * dB_bohr + iz[:, None] * dC_bohr)
                ao_i = numint.eval_ao(atm_i, pts, deriv=0)
                rho_i[i0:i1] = numint.eval_rho(atm_i, ao_i, dm_i, xctype='LDA').astype(np.float32)
            rho_na += rho_i.reshape(nx, ny, nz)

        rho_na = rho_na.astype(np.float32)

        t3 = time.time()
        print(f"  [pySCF] Neutral atom density: {t3-t2:.3f}s")

        rho_diff = (rho_scf - rho_na).astype(np.float32)

        # Charge check (same as DFTB path)
        cell_volume = step**3
        q_scf = rho_scf.sum() * cell_volume
        q_na = rho_na.sum() * cell_volume
        q_diff_val = rho_diff.sum() * cell_volume
        print(f"  [pySCF CHARGE CHECK] q_scf={q_scf:.3f}, q_na={q_na:.3f}, q_diff={q_diff_val:.6f}")

        # Electrostatic potential from rho_diff
        V_ES = afm.fft_poisson(rho_diff, step)

    print(f"  [pySCF] Total time: {time.time()-t0:.3f}s")

    # Return same format as get_density_from_dftb_dense
    # For Phase 1, we don't have a projector (CPU-based)
    # Phase 2 would include GTO basis data and a GTO-capable projector
    return {
        'rho_scf': rho_scf,
        'rho_na': rho_na,
        'rho_diff': rho_diff,
        'V_ES': V_ES,
        'origin': origin,
        'ngrid': ngrid,
        'grid_spec': grid_spec,
        'eigvecs': eigvecs,
        'eigvals': eigvals,
        'dm': dm,
        'mol': mol,
        'mf': mf,
        # These are None for pySCF backend (no STO projector)
        'norb_per_atom': None,
        'orb_offsets': None,
        'atoms_dict': None,
        'projector': None
    }


# ── Cube density provider (Psi4 / Gaussian Dt) + Gaussian ρ_NA ────────────────
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


# soft_clamp_density (tanh clamp) removed 2026-07-21 — only served deleted prepare_delta_rho_clamped.
# All-electron clamp SSOT: soft_clamp_rational → delta_rho_clamp_compact_na.

def soft_clamp_rational(y, y1, y2, dy=None):
    """Rational soft clamp (USER SSOT): above y1, approach y2 via 1/(1+z).

    For y > y1:  y' = y1 + (y2-y1)·(1 − 1/(1+z)),  z=(y−y1)/(y2−y1)
    dy' = dy / (1+z)²  (chain rule) if dy given.

    Same formula as `spammm.utils.test_utils.soft_clamp` — non-mutating copy here.
    Use for all-electron nuclear cusps before compact NA subtraction (CO guinea-pig).
    """
    y = np.asarray(y, dtype=np.float64)
    y1, y2 = float(y1), float(y2)
    if not (y2 > y1):
        raise ValueError(f'soft_clamp_rational: need y2 > y1, got y1={y1}, y2={y2}')
    y_new = y.copy()
    dy_new = None if dy is None else np.asarray(dy, dtype=np.float64).copy()
    mask = y_new > y1
    if not np.any(mask):
        return (y_new.astype(np.float32), dy_new)
    y12 = y2 - y1
    z = (y_new[mask] - y1) / y12
    y_new[mask] = y1 + y12 * (1.0 - 1.0 / (1.0 + z))
    if dy_new is not None:
        dy_new[mask] *= 1.0 / (1.0 + z) ** 2
    return y_new.astype(np.float32), dy_new


def delta_rho_clamp_compact_na(rho_scf, origin, step, atomPos, atomZ, *,
                               y1=None, y2=None, R_sphere=0.6, rc_na=0.6,
                               profile='r2', valence_Z=None):
    """All-electron Δρ recipe (CO guinea-pig SSOT): soft-clamp cores → compact NA.

    Distinguishes **all-electron** (Psi4/pySCF cubes, ∫ρ≈∑Z) from **DFTB valence**
    (∫ρ≈∑Z_val) — this path is for all-electron; DFTB already has orbital ρ_NA.

    Steps:
      1. ρ_c = soft_clamp_rational(ρ_scf; y1,y2) — kill nuclear spikes
      2. Per nucleus, Q_rem_i = ∫(ρ_scf−ρ_c) inside sphere R_sphere (charge removed)
      3. ρ_NA from compact f(r)=(1-(r/rc)^2)^2 (profile='r2') with per-atom charge
         q_i = Z_i − Q_rem_i  ("NA charge − clamped charge")
      4. Δρ = ρ_c − ρ_NA; require |∫Δρ| small (neutrality check)

    Plot valence region on a common axis with DFTB Δρ; ignore core spikes visually.

    Returns dict: rho_clamped, rho_na, rho_diff, Q_*, y1,y2, per-atom Q_rem, …
    """
    rho_scf = np.asarray(rho_scf, dtype=np.float64)
    atomPos = np.asarray(atomPos, dtype=np.float64)
    atomZ = np.asarray(atomZ, dtype=np.float64).reshape(-1)
    origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
    if np.ndim(step) == 0:
        dx = dy = dz = float(step)
    else:
        dx, dy, dz = float(step[0]), float(step[1]), float(step[2])
    dV = dx * dy * dz
    nx, ny, nz = rho_scf.shape
    R_sphere = float(R_sphere); rc_na = float(rc_na)

    # Default clamp: y1 ~ high valence (percentile away from nuclei), y2 = 2 y1
    if y1 is None or y2 is None:
        xs = origin[0] + dx * np.arange(nx)
        ys = origin[1] + dy * np.arange(ny)
        zs = origin[2] + dz * np.arange(nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        near = np.zeros(rho_scf.shape, dtype=bool)
        for p in atomPos:
            near |= ((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2) < (0.35 ** 2)
        far = rho_scf[~near & (rho_scf > 0)]
        y1_auto = float(np.percentile(far, 99.0)) if far.size else 5.0
        y1 = float(y1_auto if y1 is None else y1)
        y2 = float(2.0 * y1 if y2 is None else y2)

    rho_c, _ = soft_clamp_rational(rho_scf, y1, y2)
    rho_c64 = np.asarray(rho_c, dtype=np.float64)
    removed = rho_scf - rho_c64  # ≥0 where clamped

    xs = origin[0] + dx * np.arange(nx)
    ys = origin[1] + dy * np.arange(ny)
    zs = origin[2] + dz * np.arange(nz)
    X, Y, Zg = np.meshgrid(xs, ys, zs, indexing='ij')

    Q_rem = np.zeros(len(atomZ), dtype=np.float64)
    for i, (p, Zi) in enumerate(zip(atomPos, atomZ)):
        m = ((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Zg - p[2]) ** 2) <= (R_sphere ** 2)
        Q_rem[i] = float(removed[m].sum() * dV)

    Q_scf = float(rho_scf.sum() * dV)
    Q_c = float(rho_c64.sum() * dV)
    Q_rem_tot = float(Q_rem.sum())
    # Prefer sphere accounting; fall back to global excess if spheres miss
    Q_ex_global = Q_scf - Q_c
    Z_use = atomZ if valence_Z is None else np.asarray(valence_Z, dtype=np.float64).reshape(-1)
    q_na = np.maximum(Z_use - Q_rem, 0.0)
    # If sphere Q_rem under-counts vs global excess, scale q_na to match Q_c
    # Target: ∫ρ_NA = Q_c  ⇒  ∫Δρ = 0
    q_na_sum = float(q_na.sum())
    if q_na_sum > 1e-12:
        q_na = q_na * (Q_c / q_na_sum)

    rho_na = make_compact_rho_na(
        atomPos, q_na, origin, (dx, dy, dz) if (dx != dy or dy != dz) else dx,
        (nx, ny, nz), rc=rc_na, rescale_to_q=Q_c, profile=profile)
    rho_diff = (rho_c64 - np.asarray(rho_na, dtype=np.float64)).astype(np.float32)
    Q_diff = float(rho_diff.sum() * dV)

    return {
        'rho_scf_clamped': rho_c64.astype(np.float32),
        'rho_na': np.asarray(rho_na, dtype=np.float32),
        'rho_diff': rho_diff,
        'y1': float(y1), 'y2': float(y2),
        'R_sphere': R_sphere, 'rc_na': rc_na, 'profile': profile,
        'Q_scf': Q_scf, 'Q_clamped': Q_c, 'Q_ex_global': float(Q_ex_global),
        'Q_rem_spheres': Q_rem, 'Q_rem_tot': Q_rem_tot,
        'q_na_per_atom': q_na, 'Q_na': float(np.asarray(rho_na).sum() * dV),
        'Q_diff': Q_diff,
        'atomPos': atomPos, 'atomZ': atomZ,
    }


# REMOVED 2026-07-21: prepare_delta_rho_clamped (rescale cube ρ_NA to clamped ρ).
# That recipe inverted V_ES morphology vs USER SSOT delta_rho_clamp_compact_na
# (soft-clamp → rebuild compact NA from Z−Q_rem). Do not reintroduce.
# All-electron Δρ SSOT: delta_rho_clamp_compact_na


def make_gaussian_rho_na(atomPos, atomZ, origin, step, ngrid, sigma=0.3, rescale_to_q=None):
    """Neutral-atom density as sum of spherical Gaussians centered on atoms.

    Each atom Z contributes ∫ρ = Z (before optional rescale). Used for Δρ = ρ_scf − ρ_NA
    so that ∫Δρ ≈ 0 (charge neutrality for FFT Poisson). Shape is not physical DFT NA —
    only needs to cancel total charge and sit exactly on nuclear positions.
    Default σ=0.3 Å (pyridine same-cell V_ES: σ≲0.5 saturates AFM-height V; wider NA worsens repulsion).

    Args:
        atomPos: (natoms, 3) Angstrom — must match density grid frame
        atomZ: (natoms,) nuclear charges (electrons per Gaussian)
        origin: (3,) grid origin Angstrom
        step: float isotropic spacing Angstrom (or (3,) — uses step[0] if array)
        ngrid: (nx, ny, nz)
        sigma: Gaussian width [Å] (default 0.3)
        rescale_to_q: if set, scale ρ_NA so ∫ρ_NA = rescale_to_q (exact neutrality)

    Returns:
        rho_na: (nx, ny, nz) float32  e/Å³
    """
    atomPos = np.asarray(atomPos, dtype=np.float64)
    atomZ = np.asarray(atomZ, dtype=np.float64).reshape(-1)
    nx, ny, nz = [int(x) for x in ngrid]
    if np.ndim(step) == 0:
        dx = dy = dz = float(step)
    else:
        dx, dy, dz = float(step[0]), float(step[1]), float(step[2])
    origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"make_gaussian_rho_na: sigma must be > 0, got {sigma}")

    xs = origin[0] + dx * np.arange(nx, dtype=np.float64)
    ys = origin[1] + dy * np.arange(ny, dtype=np.float64)
    zs = origin[2] + dz * np.arange(nz, dtype=np.float64)
    # Broadcast: evaluate each atom with separable 1D Gaussians (exact for isotropic σ)
    norm = 1.0 / ((2.0 * np.pi * sigma ** 2) ** 1.5)
    inv2s2 = 1.0 / (2.0 * sigma ** 2)
    rho_na = np.zeros((nx, ny, nz), dtype=np.float64)
    for i in range(len(atomZ)):
        Zi = float(atomZ[i])
        if Zi == 0.0:
            continue
        px, py, pz = atomPos[i]
        gx = np.exp(-(xs - px) ** 2 * inv2s2)
        gy = np.exp(-(ys - py) ** 2 * inv2s2)
        gz = np.exp(-(zs - pz) ** 2 * inv2s2)
        # outer: gx[:,None,None] * gy[None,:,None] * gz[None,None,:]
        rho_na += Zi * norm * gx[:, None, None] * gy[None, :, None] * gz[None, None, :]

    dV = dx * dy * dz
    q_na = float(rho_na.sum() * dV)
    if rescale_to_q is not None and q_na != 0.0:
        rho_na *= float(rescale_to_q) / q_na
        q_na = float(rho_na.sum() * dV)
    return rho_na.astype(np.float32)


def compact_core_profile(r, rc, power=4):
    """Finite-support core shape f=(1-(r/rc)^p)^2 for r<rc, else 0.

    power=4: older default (same-cell V_ES sweeps).
    power=2: USER SSOT for all-electron NA after soft-clamp (smoother) — CO guinea-pig.
    """
    r = np.asarray(r, dtype=np.float64)
    rc = float(rc)
    p = int(power)
    if rc <= 0:
        raise ValueError(f"compact_core_profile: rc must be > 0, got {rc}")
    if p not in (2, 4):
        raise ValueError(f"compact_core_profile: power must be 2 or 4, got {p}")
    u = (r / rc) ** p
    f = np.where(r < rc, (1.0 - u) ** 2, 0.0)
    return f


def _compact_core_integral_unit(rc, power=4):
    """∫ f(r) 4π r² dr for unit amplitude, f=(1-(r/rc)^p)^2 on [0,rc]."""
    rc = float(rc)
    if power == 4:
        # ∫_0^rc (1-(r/rc)^4)^2 4π r² dr = 128 π rc³ / 231
        return 128.0 * np.pi * rc ** 3 / 231.0
    if power == 2:
        # ∫_0^rc (1-(r/rc)^2)^2 4π r² dr = 32 π rc³ / 105
        return 32.0 * np.pi * rc ** 3 / 105.0
    raise ValueError(f'_compact_core_integral_unit: power={power}')


def make_compact_rho_na(atomPos, atomZ, origin, step, ngrid, rc=0.5, rescale_to_q=None,
                        profile='r4'):
    """Neutral-atom density as sum of compact cores on atoms.

    profile:
      'r4' / power 4 — f=(1-(r/rc)^4)^2  (legacy)
      'r2' / power 2 — f=(1-(r/rc)^2)^2  (USER SSOT after soft-clamp)

    Support strictly r < rc. Amplitude so ∫ρ = Z (or atom charge) before optional rescale.

    Args:
        atomPos, atomZ, origin, step, ngrid: same as make_gaussian_rho_na
        rc: cutoff / core radius [Å]
        rescale_to_q: if set, scale so ∫ρ_NA = rescale_to_q
        profile: 'r2' or 'r4'

    Returns:
        rho_na: (nx, ny, nz) float32  e/Å³
    """
    atomPos = np.asarray(atomPos, dtype=np.float64)
    atomZ = np.asarray(atomZ, dtype=np.float64).reshape(-1)
    nx, ny, nz = [int(x) for x in ngrid]
    if np.ndim(step) == 0:
        dx = dy = dz = float(step)
    else:
        dx, dy, dz = float(step[0]), float(step[1]), float(step[2])
    origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
    rc = float(rc)
    if rc <= 0:
        raise ValueError(f"make_compact_rho_na: rc must be > 0, got {rc}")
    prof = str(profile).lower()
    power = 2 if prof in ('r2', '2', 'quad') else 4
    I1 = _compact_core_integral_unit(rc, power=power)

    xs = origin[0] + dx * np.arange(nx, dtype=np.float64)
    ys = origin[1] + dy * np.arange(ny, dtype=np.float64)
    zs = origin[2] + dz * np.arange(nz, dtype=np.float64)
    # local stamp
    n_pad = int(np.ceil(rc / min(dx, dy, dz))) + 2
    rho_na = np.zeros((nx, ny, nz), dtype=np.float64)
    for pos, Zi in zip(atomPos, atomZ):
        Zi = float(Zi)
        if abs(Zi) < 1e-30:
            continue
        A = Zi / I1
        ix = int(round((pos[0] - origin[0]) / dx))
        iy = int(round((pos[1] - origin[1]) / dy))
        iz = int(round((pos[2] - origin[2]) / dz))
        i0, i1 = max(0, ix - n_pad), min(nx, ix + n_pad + 1)
        j0, j1 = max(0, iy - n_pad), min(ny, iy + n_pad + 1)
        k0, k1 = max(0, iz - n_pad), min(nz, iz + n_pad + 1)
        xx = xs[i0:i1][:, None, None]
        yy = ys[j0:j1][None, :, None]
        zz = zs[k0:k1][None, None, :]
        r = np.sqrt((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2 + (zz - pos[2]) ** 2)
        rho_na[i0:i1, j0:j1, k0:k1] += A * compact_core_profile(r, rc, power=power)

    if rescale_to_q is not None:
        q = float(rho_na.sum() * dx * dy * dz)
        if abs(q) > 1e-30:
            rho_na *= float(rescale_to_q) / q
    return rho_na.astype(np.float32)


def get_density_from_cube(cube_path_or_dir, *, esp_path=None, sigma_na=0.3,
                          rescale_na=True, use_esp_cube=False, verbosity=0,
                          na_kind='gaussian', rc_na=0.5):
    """Load Dt.cube (+ optional ESP) → same dict as get_density_from_dftb_dense.

    ρ_scf from Dt (e/a0³ → e/Å³). ρ_NA from `na_kind`:
      - 'gaussian': Σ Z_i Gaussians, width `sigma_na`
      - 'compact': Σ Z_i (1-(r/rc_na)^4)^2 for r<rc_na (finite support)
    ρ_diff = ρ_scf − ρ_NA with ∫ρ_diff forced ≈ 0 via rescale when `rescale_na=True`.

    Atom positions come from the **cube header** (not geom.xyz) so NA sits
    on the density frame. Does not modify DFTB/pySCF providers.

    Args:
        cube_path_or_dir: path to Dt.cube or directory containing Dt.cube
        esp_path: optional ESP.cube (default: sibling ESP.cube)
        sigma_na: Gaussian NA width [Å] (default 0.3; used if na_kind='gaussian')
        rc_na: compact core radius [Å] (default 0.5; used if na_kind='compact')
        na_kind: 'gaussian' | 'compact'
        rescale_na: scale ρ_NA so ∫ρ_NA = ∫ρ_scf (charge neutrality)
        use_esp_cube: if True and ESP found, use it as V_ES; else fft_poisson(ρ_diff)
        verbosity: print charge checks

    Returns:
        dict with rho_scf, rho_na, rho_diff, V_ES, origin, ngrid, grid_spec, atomPos, atomZ, …
    """
    from spammm.quantum.DFTB.DFTBplusParser import read_cube

    path = cube_path_or_dir
    _RHO_NAMES = ('Dt.cube', 'rho_N.cube', 'rho.cube')
    _ESP_NAMES = ('ESP.cube', 'esp_N.cube', 'esp.cube')
    if os.path.isdir(path):
        dt_path = None
        for name in _RHO_NAMES:
            cand = os.path.join(path, name)
            if os.path.isfile(cand):
                dt_path = cand
                break
        if dt_path is None:
            raise FileNotFoundError(f"get_density_from_cube: no density cube in {path} (tried {_RHO_NAMES})")
        if esp_path is None:
            for name in _ESP_NAMES:
                cand = os.path.join(path, name)
                if os.path.isfile(cand):
                    esp_path = cand
                    break
    else:
        dt_path = path
        if esp_path is None:
            dname = os.path.dirname(path)
            base = os.path.basename(path)
            # rho_N.cube → esp_N.cube; Dt.cube → ESP.cube
            paired = None
            if base.startswith('rho_') and base.endswith('.cube'):
                paired = os.path.join(dname, 'esp_' + base[4:])
            for cand in ([paired] if paired else []) + [os.path.join(dname, n) for n in _ESP_NAMES]:
                if cand and os.path.isfile(cand):
                    esp_path = cand
                    break
    if not os.path.isfile(dt_path):
        raise FileNotFoundError(f"get_density_from_cube: missing Dt cube: {dt_path}")

    rho_b, origin_b, step_b, nPoints, atoms_b = read_cube(dt_path)
    nx, ny, nz = nPoints
    # Density e/a0³ → e/Å³: ρ_A = ρ_B * (a0/Å)³ = ρ_B / BOHR_TO_ANG³? 
    # ∫ ρ_B dV_B = N_e with dV_B in a0³. Same N_e = ∫ ρ_A dV_A with dV_A in Å³
    # ⇒ ρ_A = ρ_B * (dV_B/dV_A) = ρ_B / BOHR_TO_ANG³
    b3 = BOHR_TO_ANG ** 3
    rho_scf = (rho_b / b3).astype(np.float32)
    origin = np.asarray(origin_b, dtype=np.float64) * BOHR_TO_ANG
    step_vec = np.asarray(step_b, dtype=np.float64).ravel()[:3] * BOHR_TO_ANG
    step_mean = float(np.mean(step_vec))
    aniso = float(np.max(np.abs(step_vec - step_mean)) / max(step_mean, 1e-30))
    # pySCF cubes often have ~0.1% axis anisotropy from cell rounding — accept and use mean step
    if aniso > 0.01:
        raise ValueError(f"get_density_from_cube: non-isotropic cube step {step_vec} (aniso={aniso:.3e})")
    step = step_mean
    atomZ = np.array([a[0] for a in atoms_b], dtype=np.float64)
    atomPos = np.array([[a[1], a[2], a[3]] for a in atoms_b], dtype=np.float64) * BOHR_TO_ANG

    dV = float(np.prod(step_vec))  # exact voxel volume even if mildly anisotropic
    q_scf = float(rho_scf.sum() * dV)
    na_kind = str(na_kind).lower().strip()
    if na_kind == 'gaussian':
        rho_na = make_gaussian_rho_na(
            atomPos, atomZ, origin, step, (nx, ny, nz),
            sigma=sigma_na, rescale_to_q=(q_scf if rescale_na else None),
        )
        na_tag = f'sigma_na={sigma_na}'
    elif na_kind == 'compact':
        rho_na = make_compact_rho_na(
            atomPos, atomZ, origin, step, (nx, ny, nz),
            rc=rc_na, rescale_to_q=(q_scf if rescale_na else None),
        )
        na_tag = f'rc_na={rc_na}'
    else:
        raise ValueError(f"get_density_from_cube: na_kind must be 'gaussian' or 'compact', got {na_kind!r}")
    rho_diff = (rho_scf - rho_na).astype(np.float32)
    q_na = float(rho_na.sum() * dV)
    q_diff = float(rho_diff.sum() * dV)

    if verbosity >= 0:
        print(f"  [cube] {dt_path}")
        print(f"  [cube] grid={nx}x{ny}x{nz} step={step:.5f}Å origin={origin}")
        print(f"  [cube] natoms={len(atomZ)} Zsum={atomZ.sum():.1f} na_kind={na_kind} {na_tag}")
        print(f"  [CHARGE CHECK] q_scf={q_scf:.6f} q_na={q_na:.6f} q_diff={q_diff:.6e}")
        if abs(q_diff) > 0.05:
            print(f"  WARNING: |q_diff|={abs(q_diff):.4f} > 0.05 e — ES may be unreliable")

    V_ES = None
    if use_esp_cube and esp_path and os.path.isfile(esp_path):
        V_b, o2, s2, n2, _ = read_cube(esp_path)
        if n2 != nPoints:
            raise ValueError(f"ESP cube shape {n2} != Dt {nPoints}")
        # Psi4 ESP typically Hartree/e; convert later if needed — store raw for now as float32
        V_ES = V_b.astype(np.float32)
        if verbosity >= 0:
            print(f"  [cube] V_ES from {esp_path} (file units)")
    else:
        # Cube ngrid often has prime factors clFFT rejects; CPU Poisson is explicit here.
        # After resample onto FDBM-friendly grids, callers can re-Poisson on GPU.
        V_ES = afm.fft_poisson_cpu(rho_diff, step)
        if verbosity >= 0:
            print(f"  [cube] V_ES from fft_poisson_cpu (native cube grid)")

    grid_spec = {
        'origin': origin.copy(),
        'dA': np.array([step, 0.0, 0.0]),
        'dB': np.array([0.0, step, 0.0]),
        'dC': np.array([0.0, 0.0, step]),
        'ngrid': np.array([nx, ny, nz], dtype=int),
    }
    return {
        'rho_scf': rho_scf,
        'rho_na': rho_na,
        'rho_diff': rho_diff,
        'V_ES': V_ES,
        'origin': origin,
        'ngrid': np.array([nx, ny, nz], dtype=int),
        'grid_spec': grid_spec,
        'step': step,
        'atomPos': atomPos,
        'atomZ': atomZ,
        'q_scf': q_scf,
        'q_na': q_na,
        'q_diff': q_diff,
        'sigma_na': float(sigma_na),
        'rc_na': float(rc_na),
        'na_kind': na_kind,
        'cube_path': dt_path,
        'esp_path': esp_path,
    }


def plot_cube_density_diagnostics(d, save_dir, tag='cube', z_above=0.0):
    """XY slices of ρ_scf, ρ_NA, ρ_diff with atom markers; write CHARGE snippet.

    Atoms must sit on ρ_NA peaks. ∫ρ_diff must be ~0 (printed on figure).
    """
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    rho_scf, rho_na, rho_diff = d['rho_scf'], d['rho_na'], d['rho_diff']
    origin, step = d['origin'], float(d.get('step', d['grid_spec']['dA'][0]))
    atomPos, atomZ = d['atomPos'], d['atomZ']
    nx, ny, nz = rho_scf.shape
    z_mol = float(atomPos[:, 2].mean())
    z_target = z_mol + float(z_above)
    zs = origin[2] + step * np.arange(nz)
    iz = int(np.clip(np.argmin(np.abs(zs - z_target)), 0, nz - 1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    extent = [origin[0], origin[0] + (nx - 1) * step, origin[1], origin[1] + (ny - 1) * step]
    panels = [
        (rho_scf[:, :, iz], 'ρ_scf', 'magma', False),
        (rho_na[:, :, iz], 'ρ_NA (Gaussians)', 'magma', False),
        (rho_diff[:, :, iz], 'ρ_diff = scf−NA', 'bwr', True),
    ]
    for ax, (sl, title, cmap, sym) in zip(axes, panels):
        arr = sl.T
        if sym:
            v = max(float(np.percentile(np.abs(arr), 99)), 1e-12)
            im = ax.imshow(arr, origin='lower', extent=extent, cmap=cmap, vmin=-v, vmax=v, aspect='equal')
        else:
            im = ax.imshow(arr, origin='lower', extent=extent, cmap=cmap, aspect='equal')
        ax.scatter(atomPos[:, 0], atomPos[:, 1], c='cyan', s=40, marker='x', linewidths=1.5, label='atoms')
        for j, Z in enumerate(atomZ):
            ax.annotate(str(int(Z)), (atomPos[j, 0], atomPos[j, 1]), color='w', fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_title(f'{title}\nz={zs[iz]:.2f}Å', fontsize=9)
        ax.set_xlabel('x [Å]'); ax.set_ylabel('y [Å]')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f'{tag}  q_scf={d["q_scf"]:.4f}  q_na={d["q_na"]:.4f}  q_diff={d["q_diff"]:.3e}  σ_NA={d["sigma_na"]}Å',
        fontsize=10,
    )
    fig.tight_layout()
    out = os.path.join(save_dir, f'{tag}_rho_scf_na_diff.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'Saved {out}')

    # Peak of ρ_NA vs atom XY distance (must be small)
    peak = np.unravel_index(int(np.argmax(rho_na)), rho_na.shape)
    peak_xy = np.array([origin[0] + peak[0] * step, origin[1] + peak[1] * step])
    dmin = float(np.min(np.linalg.norm(atomPos[:, :2] - peak_xy[None, :], axis=1)))
    # Per-atom: nearest voxel should be a local max of ρ_NA (Gaussian sits on nucleus).
    # Allow ~1 voxel; H near O may shift slightly in the *combined* field.
    atom_dists = []
    for j in range(len(atomZ)):
        ix = int(np.clip(round((atomPos[j, 0] - origin[0]) / step), 0, nx - 1))
        iy = int(np.clip(round((atomPos[j, 1] - origin[1]) / step), 0, ny - 1))
        iz_a = int(np.clip(round((atomPos[j, 2] - origin[2]) / step), 0, nz - 1))
        grid_xyz = origin + step * np.array([ix, iy, iz_a], dtype=float)
        atom_dists.append(float(np.linalg.norm(atomPos[j] - grid_xyz)))
        x0, x1 = max(0, ix - 2), min(nx, ix + 3)
        y0, y1 = max(0, iy - 2), min(ny, iy + 3)
        z0, z1 = max(0, iz_a - 2), min(nz, iz_a + 3)
        local = rho_na[x0:x1, y0:y1, z0:z1]
        # value at atom voxel vs local max
        v_atom = float(rho_na[ix, iy, iz_a])
        v_max = float(local.max())
        if v_atom < 0.5 * v_max:
            atom_dists[-1] = 1e9  # fail marker
    max_atom_grid_dist = max(atom_dists) if atom_dists else float('nan')
    lines = [
        f'tag={tag}',
        f'cube={d.get("cube_path")}',
        f'q_scf={d["q_scf"]:.8f} q_na={d["q_na"]:.8f} q_diff={d["q_diff"]:.8e}',
        f'sigma_na={d["sigma_na"]} step={step}',
        f'rho_na global peak index={peak} peak_xy={peak_xy} min_dist_to_atom_xy={dmin:.4f} Å',
        f'max_atom_to_nearest_voxel={max_atom_grid_dist:.4f} Å (inf => NA not peaked at atom)',
        f'PASS_charge={abs(d["q_diff"]) < 0.05}',
        f'PASS_atoms_on_NA={max_atom_grid_dist < 0.75 * step * np.sqrt(3)}',
    ]
    out_txt = os.path.join(save_dir, f'{tag}_CHARGE.out')
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'REVIEW: {out_txt}')
    return out, out_txt


def resample_field_to_grid(field, origin_src, step_src, origin_dst, step_dst, ngrid_dst, order=1):
    """Trilinear resample 3D field from src grid onto dst grid (Å)."""
    from scipy.ndimage import map_coordinates
    field = np.asarray(field, dtype=np.float64)
    origin_src = np.asarray(origin_src, dtype=np.float64).ravel()[:3]
    origin_dst = np.asarray(origin_dst, dtype=np.float64).ravel()[:3]
    step_src = float(step_src) if np.ndim(step_src) == 0 else float(step_src[0])
    step_dst = float(step_dst) if np.ndim(step_dst) == 0 else float(step_dst[0])
    nx, ny, nz = [int(x) for x in ngrid_dst]
    ix = np.arange(nx, dtype=np.float64)
    iy = np.arange(ny, dtype=np.float64)
    iz = np.arange(nz, dtype=np.float64)
    X, Y, Z = np.meshgrid(ix, iy, iz, indexing='ij')
    wx = origin_dst[0] + X * step_dst
    wy = origin_dst[1] + Y * step_dst
    wz = origin_dst[2] + Z * step_dst
    coords = np.stack([
        (wx - origin_src[0]) / step_src,
        (wy - origin_src[1]) / step_src,
        (wz - origin_src[2]) / step_src,
    ], axis=0)
    out = map_coordinates(field, coords, order=order, mode='constant', cval=0.0)
    return out.astype(np.float32)


def tip_density_apex_down(rho, atomPos, atomZ, apex_Z, origin, step, z_tol=0.15):
    """Ensure tip apex atom has the lowest z (AFM approach along +z from above).

    Mithun cubes: CO_O/HF_*/NH3_H already apex-down along z — leave unchanged.
    Planar tips (H2O_O/H2O_H, all z≈0): permute (x,y,z)→(x,z,−y) then flip if needed.

    Returns (rho_out, atomPos_out, origin_out, reoriented:bool).
    """
    atomPos = np.asarray(atomPos, dtype=np.float64)
    atomZ = np.asarray(atomZ, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
    step = float(step)
    apex_mask = np.isclose(atomZ, float(apex_Z))
    if not np.any(apex_mask):
        raise ValueError(f"tip_density_apex_down: no atom with Z={apex_Z}")
    z_apex = float(atomPos[apex_mask, 2].mean())
    z_min = float(atomPos[:, 2].min())
    z_span = float(atomPos[:, 2].max() - atomPos[:, 2].min())
    # Already apex-down (apex near global min z, molecule has z extent)
    if z_span > z_tol and z_apex <= z_min + z_tol:
        return rho.astype(np.float32), atomPos.copy(), origin.copy(), False

    # Planar / wrong-axis: (x,y,z) → (x,z,−y)
    nx, ny, nz = rho.shape
    rho2 = np.transpose(rho, (0, 2, 1))[:, :, ::-1].copy()
    apos2 = atomPos.copy()
    apos2[:, 0] = atomPos[:, 0]
    apos2[:, 1] = atomPos[:, 2]
    apos2[:, 2] = -atomPos[:, 1]
    # out2[i,j,k] = in[i, ny-1-k, j] → origin_y'=oz, origin_z'=-(oy+(ny-1)*s)
    origin2 = np.array([origin[0], origin[2], -(origin[1] + (ny - 1) * step)], dtype=np.float64)
    if float(apos2[apex_mask, 2].mean()) > float(apos2[:, 2].mean()):
        rho2 = rho2[:, :, ::-1].copy()
        apos2[:, 2] = -apos2[:, 2]
        nzz = rho2.shape[2]
        origin2[2] = -origin2[2] - (nzz - 1) * step
    return rho2.astype(np.float32), apos2, origin2, True


def build_fdbm_grid_from_cubes(sample_cube_dir, tip_cube_dir, *, step=0.1, margin_xy=4.0,
                               z_min=-15.0, z_max=15.0, sigma_na=0.3, apex_Z=None,
                               A_pauli=None, beta_pauli=None, tip_name=None,
                               use_esp_cube=False, verbosity=0,
                               clamp_cores=True, clamp_percentile=99.0, clamp_rho_max=None,
                               use_gpu_project=True, z_symmetric=True,
                               na_kind='gaussian', rc_na=0.5):
    """Build FDBM F_total on a tall AFM grid from sample+tip Dt cubes.

    Pipeline (dipole-safe):
      1. All-electron Δρ SSOT: delta_rho_clamp_compact_na (soft-clamp → rebuild compact NA)
      2. GPU *project* (trilinear scatter) onto dest grid — preserves charge + dipole
         (not scipy sample, which breaks ∫Δρ)
      3. z-box **symmetric** about the molecular plane by default (avoids fake pz from
         uniform monopole strip on an asymmetric cell)
      4. Poisson(Δρ); tip Δρ rolled to apex; no monopole strip unless |q| is large

    Tip: apex-down only if cube is planar; then roll density peak to (0,0,0).
    na_kind/rc_na/sigma_na: forwarded to get_density_from_cube for sample+tip NA.

    Caveat (CO tip vs DFTB, 2026-07-21): all-electron cube tip Δρ has huge nuclear-cusp
    residuals vs compact/Gaussian NA even when ∫Δρ≈0; DFTB tip is valence-only (CO q≈10
    not 14) with mild Δρ. Project tip onto the **full AFM cell** (do not crop tip with a
    tight bbox — margin≲1 Å on CO fabricated q_del≈−0.7 and wrecked E_es). E_es bend of
    cube path is tip_Δρ ⊗ V_sample; see `doc/Tasks/Import_KrigingGridFF.md` session notes.
    """
    from . import AFM as afm_mod
    from spammm.utils.GridsOCL import grid_moments_centers

    d_s = get_density_from_cube(
        sample_cube_dir, sigma_na=sigma_na, rescale_na=True,
        use_esp_cube=use_esp_cube, verbosity=verbosity,
        na_kind=na_kind, rc_na=rc_na)
    d_t = get_density_from_cube(
        tip_cube_dir, sigma_na=sigma_na, rescale_na=True, verbosity=verbosity,
        na_kind=na_kind, rc_na=rc_na)

    if apex_Z is None and tip_name:
        suf = tip_name.split('_')[-1]
        apex_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}.get(suf, int(d_t['atomZ'][np.argmax(d_t['atomZ'])]))
    if apex_Z is None:
        apex_Z = int(d_t['atomZ'][np.argmax(d_t['atomZ'])])

    # ── Sample Δρ: USER SSOT clamp→compact NA (Pauli still uses full ρ_scf) ────
    if clamp_cores:
        clamp_s = delta_rho_clamp_compact_na(
            d_s['rho_scf'], d_s['origin'], d_s['step'], d_s['atomPos'], d_s['atomZ'],
            rc_na=float(rc_na), profile='r2')
        rho_diff_src = clamp_s['rho_diff']
        if verbosity >= 0:
            print(f"  [fdbm-cube] sample Δρ=clamp→compact_NA  Q_scf={clamp_s['Q_scf']:.3f} "
                  f"Q_c={clamp_s['Q_clamped']:.3f} ∫Δρ={clamp_s['Q_diff']:.3e}")
    else:
        rho_diff_src = d_s['rho_diff']
        clamp_s = None

    # ── Tip: apex-down, then same Δρ SSOT (do NOT rescale cube ρ_NA) ───────────
    tip_rho, tip_apos, tip_origin, tip_reoriented = tip_density_apex_down(
        d_t['rho_scf'], d_t['atomPos'], d_t['atomZ'], apex_Z, d_t['origin'], d_t['step'])
    if clamp_cores:
        clamp_t = delta_rho_clamp_compact_na(
            tip_rho, tip_origin, d_t['step'], tip_apos, d_t['atomZ'],
            rc_na=float(rc_na), profile='r2')
        tip_diff = clamp_t['rho_diff']
        if verbosity >= 0:
            print(f"  [fdbm-cube] tip Δρ=clamp→compact_NA  Q_scf={clamp_t['Q_scf']:.3f} "
                  f"Q_c={clamp_t['Q_clamped']:.3f} ∫Δρ={clamp_t['Q_diff']:.3e}")
    else:
        tip_diff, _, _, _ = tip_density_apex_down(
            d_t['rho_diff'], d_t['atomPos'], d_t['atomZ'], apex_Z, d_t['origin'], d_t['step'])
        clamp_t = None

    # ── Destination grid: cover sample cube AABB + margin; z symmetric ────────
    spos = d_s['atomPos']
    z_mol = float(spos[:, 2].mean())
    # XY: union of atom bbox and density cube footprint
    so, ss, sn = d_s['origin'], float(d_s['step']), d_s['rho_scf'].shape
    x0 = min(float(spos[:, 0].min()) - margin_xy, so[0] - 0.5)
    x1 = max(float(spos[:, 0].max()) + margin_xy, so[0] + (sn[0] - 1) * ss + 0.5)
    y0 = min(float(spos[:, 1].min()) - margin_xy, so[1] - 0.5)
    y1 = max(float(spos[:, 1].max()) + margin_xy, so[1] + (sn[1] - 1) * ss + 0.5)
    Lz = float(z_max - z_min)
    if Lz < 12.0:
        raise ValueError(
            f"build_fdbm_grid_from_cubes: z span {Lz:.1f} Å too small for periodic FFT "
            f"(need ≳12 Å). Got z_min={z_min}, z_max={z_max}."
        )
    if z_symmetric:
        z_half = 0.5 * Lz
        z0, z1 = z_mol - z_half, z_mol + z_half
    else:
        z0, z1 = float(z_min), float(z_max)
    nx = afm_mod._FDBMGpyFFT.round_fft_friendly(int(np.ceil((x1 - x0) / step)) + 1)
    ny = afm_mod._FDBMGpyFFT.round_fft_friendly(int(np.ceil((y1 - y0) / step)) + 1)
    nz = afm_mod._FDBMGpyFFT.round_fft_friendly(int(np.ceil((z1 - z0) / step)) + 1)
    origin = np.array([x0, y0, z0], dtype=np.float64)
    ngrid = np.array([nx, ny, nz], dtype=int)
    dV = step ** 3
    vol = float(nx * ny * nz) * dV

    def _to_dest(field, origin_src, step_src, grids=None):
        if use_gpu_project:
            if grids is None:
                from spammm.utils.GridsOCL import GridsOCL
                grids = GridsOCL()
            return grids.project_density(field, origin_src, step_src, origin, step, ngrid), grids
        return resample_field_to_grid(field, origin_src, step_src, origin, step, ngrid), None

    grids = None
    rho_scf, grids = _to_dest(d_s['rho_scf'], d_s['origin'], d_s['step'], grids)
    rho_diff, grids = _to_dest(rho_diff_src, d_s['origin'], d_s['step'], grids)
    tip_tot, grids = _to_dest(tip_rho, tip_origin, d_t['step'], grids)
    tip_del, grids = _to_dest(tip_diff, tip_origin, d_t['step'], grids)

    tip_tot, tip_del = _pad_and_roll_co_tip_pair(tip_tot, tip_del, (nx, ny, nz))

    q_scf = float(rho_scf.sum() * dV)
    q_diff = float(rho_diff.sum() * dV)
    q_tip_del = float(tip_del.sum() * dV)
    _, p_diff = grid_moments_centers(rho_diff, origin, step)
    # Only strip monopole if large; on a z-symmetric box this does not inject fake pz
    if abs(q_diff) > 1e-4:
        if verbosity >= 0:
            print(f"  [fdbm-cube] WARNING stripping sample monopole q_diff={q_diff:.3e} (z_symmetric={z_symmetric})")
        rho_diff = (rho_diff - q_diff / vol).astype(np.float32)
        q_diff = float(rho_diff.sum() * dV)
    if abs(q_tip_del) > 1e-4:
        if verbosity >= 0:
            print(f"  [fdbm-cube] WARNING stripping tip monopole q_tip_del={q_tip_del:.3e}")
        tip_del = (tip_del - q_tip_del / vol).astype(np.float32)
        q_tip_del = float(tip_del.sum() * dV)
    _, p_diff = grid_moments_centers(rho_diff, origin, step)

    if A_pauli is None or beta_pauli is None:
        pa = afm_mod.PAULI_FITTED_DEFAULTS.get('pyscf_6-31g*', {'A': 40.0, 'beta': 1.15})
        A_pauli = float(pa['A'] if A_pauli is None else A_pauli)
        beta_pauli = float(pa['beta'] if beta_pauli is None else beta_pauli)

    if verbosity >= 0:
        print(f"  [fdbm-cube] grid={nx}x{ny}x{nz} step={step} origin={origin} Lz={nz*step:.1f}Å "
              f"z_sym={z_symmetric} gpu_project={use_gpu_project}")
        print(f"  [fdbm-cube] q_scf={q_scf:.4f} q_diff={q_diff:.3e} q_tip_del={q_tip_del:.3e} "
              f"p_diff={p_diff} A={A_pauli} beta={beta_pauli} apex_Z={apex_Z} reoriented={tip_reoriented}")
        print(f"  [fdbm-cube] tip peak after roll={np.unravel_index(int(np.argmax(np.abs(tip_tot))), tip_tot.shape)}")

    os.environ.setdefault('SPAMMM_AFM_CPU_FFT', '1')
    overlap = afm_mod.compute_pauli_overlap(rho_scf, tip_tot, step, tip_rolled=True)
    E_pauli = afm_mod.scale_pauli_field(overlap, step, A_pauli, beta_pauli, return_grads=False)
    if use_esp_cube and d_s.get('V_ES') is not None and d_s.get('esp_path'):
        HARTREE_TO_EV = 27.211386
        V_ES, _ = _to_dest(np.asarray(d_s['V_ES'], dtype=np.float32) * np.float32(HARTREE_TO_EV),
                           d_s['origin'], d_s['step'], grids)
        tip_del_es = (-tip_del).astype(np.float32)
        if verbosity >= 0:
            print(f"  [fdbm-cube] V_ES from ESP.cube (Ha→eV), tip_del → −Δρ for charge convention")
    else:
        V_ES = afm_mod.fft_poisson_cpu(rho_diff, step)
        tip_del_es = tip_del
    E_es = afm_mod.compute_es_conv_field(V_ES, tip_del_es, step, tip_rolled=True, return_grads=False)
    atomTypes = d_s['atomZ'].astype(np.int32)
    E_vdw = afm_mod.compute_dispersion_grid(
        spos, atomTypes, origin, step, ngrid, C6_CO=30.0, return_grads=False, use_opencl=False)
    E_total = (E_pauli + E_es + E_vdw).astype(np.float32)

    afmulator = afm_mod.AFMulator(use_morse=False, nloc=32)
    F_total = afmulator.compute_gradient_cl(E_total, step, bAlloc=True)

    return {
        'F_total': F_total,
        'E_total': E_total,
        'E_pauli': E_pauli,
        'E_es': E_es,
        'E_vdw': E_vdw,
        'V_ES': V_ES,
        'overlap_raw': overlap,
        'rho_scf': rho_scf,
        'rho_diff': rho_diff,
        'tip_tot': tip_tot,
        'tip_del': tip_del,
        'origin': origin,
        'step': step,
        'ngrid': ngrid,
        'atomPos': spos,
        'atomZ': atomTypes,
        'A_pauli': A_pauli,
        'beta_pauli': beta_pauli,
        'q_scf': q_scf,
        'q_diff': q_diff,
        'q_tip_del': q_tip_del,
        'p_diff': p_diff,
        'tip_reoriented': tip_reoriented,
        'use_esp_cube': bool(use_esp_cube),
        'clamp_cores': bool(clamp_cores),
        'use_gpu_project': bool(use_gpu_project),
        'z_symmetric': bool(z_symmetric),
        'clamp_s': clamp_s,
        'afmulator': afmulator,
    }


def get_density_from_dftb_plus(atomPos, atomTypes, basis, slako_prefix, work_dir,
                                grid_spec=None, step=0.1, margin=4.0, z_extra=6.0, verbosity=0):
    """
    Run DFTB+ SCF for density projection and return density grids.

    Returns dict with 'rho_scf', 'rho_na', 'rho_diff', 'V_ES', 'origin', 'ngrid', 'grid_spec'.
    """
    from spammm.quantum.DFTB_utils import SK_PATHS as _SK_PATHS, WFC_HSD_PATHS as _WFC_HSD_PATHS
    from spammm.quantum.DFTB_utils import run_dftb_for_density as _run_dftb_for_density
    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'P':15,'S':16,'Br':35,'I':53}
    inv_z = {v:k for k,v in ELEM_Z.items()}
    enames = [inv_z.get(int(z), 'C') for z in atomTypes]

    if grid_spec is None:
        grid_spec, origin, ngrid, step = _make_grid_spec(atomPos, step, margin, z_extra)
    else:
        origin, ngrid, step = grid_spec['origin'], grid_spec['ngrid'], grid_spec['dA'][0]

    geo, evecs = _run_dftb_for_density(work_dir, enames, atomPos, slako_prefix)
    
    # Parse basis HSD file for density projection
    # Use waveplot_in.hsd from DFTB output if it exists (matches actual calculation)
    # Otherwise fall back to pre-defined basis file
    from spammm.quantum.DFTB.DFTBplusParser import parse_basis_hsd_ang
    waveplot_hsd = os.path.join(work_dir, 'waveplot_in.hsd')
    if os.path.exists(waveplot_hsd):
        species_list_ang = parse_basis_hsd_ang(waveplot_hsd)
    else:
        basis_hsd_path = _WFC_HSD_PATHS.get(basis)
        if basis_hsd_path is None:
            raise ValueError(f"No basis HSD file defined for basis '{basis}'. Available: {list(_WFC_HSD_PATHS.keys())}")
        species_list_ang = parse_basis_hsd_ang(basis_hsd_path)
    
    # Validate that all atoms in the molecule are present in the basis
    basis_species = set(sp['name'] for sp in species_list_ang)
    molecule_species = set(geo['species_names'])
    missing_species = molecule_species - basis_species
    if missing_species:
        raise ValueError( f"Atoms in molecule not supported by basis '{basis}': {missing_species}.  Basis contains: {sorted(basis_species)}")
    
    rho_scf, rho_na, rho_diff = _project_densities(geo, evecs, species_list_ang, grid_spec, verbosity)
    V_ES = afm.fft_poisson(rho_diff, step)
    return {'rho_scf': rho_scf, 'rho_na': rho_na, 'rho_diff': rho_diff, 'V_ES': V_ES,
            'origin': origin, 'ngrid': ngrid, 'grid_spec': grid_spec}


def get_density_from_dftb(atomPos, atomTypes, dftb_dir, basis=None,
                           grid_spec=None, step=0.15, margin=4.0, z_extra=6.0, verbosity=0):
    """
    Get density grids from pre-computed DFTB+ output files (detailed.xml + eigenvec.bin).

    Returns dict with 'rho_scf', 'rho_na', 'rho_diff', 'V_ES', 'origin', 'ngrid', 'grid_spec'.
    """
    from spammm.quantum.DFTB.DFTBplusParser import parse_detailed_xml_custom, parse_eigenvec_bin_custom, parse_basis_hsd_ang

    if grid_spec is None:
        grid_spec, origin, ngrid, step = _make_grid_spec(atomPos, step, margin, z_extra)
    else:
        origin, ngrid, step = grid_spec['origin'], grid_spec['ngrid'], grid_spec['dA'][0]

    geo   = parse_detailed_xml_custom(os.path.join(dftb_dir, 'detailed.xml'))
    evecs = parse_eigenvec_bin_custom(os.path.join(dftb_dir, 'eigenvec.bin'), geo['nstates'], geo['norb'])

    if basis is None:
        hsd = os.path.join(dftb_dir, 'waveplot_in.hsd')
        if not os.path.exists(hsd):
            raise FileNotFoundError(f"waveplot_in.hsd not found in {dftb_dir}")
        basis = parse_basis_hsd_ang(hsd)

    rho_scf, rho_na, rho_diff = _project_densities(geo, evecs, basis, grid_spec, verbosity)
    V_ES = afm.fft_poisson(rho_diff, step)
    return {'rho_scf': rho_scf, 'rho_na': rho_na, 'rho_diff': rho_diff, 'V_ES': V_ES,
            'origin': origin, 'ngrid': ngrid, 'grid_spec': grid_spec}


def get_density_from_fireball(atomPos, atomTypes, grid_spec, fdata_dir, fc_instance=None, step=0.15, margin=4.0, z_extra=6.0, verbosity=0):
    """
    Get electron density from Fireball SCF.
    
    Args:
        atomPos: (natoms, 3) positions in Angstrom
        atomTypes: (natoms,) atomic numbers
        grid_spec: dict with origin, dA, dB, dC, ngrid (optional, will auto-generate if None)
        fdata_dir: directory with Fireball basis files
        fc_instance: optional FireCore instance (will create if None)
        step: grid spacing in Angstrom (if grid_spec not provided)
        margin: margin around molecule for grid
        z_extra: extra margin in z direction
        verbosity: logging level
        
    Returns:
        dict with 'rho_scf', 'rho_na', 'rho_diff', 'V_ES', 'origin', 'ngrid', 'grid_spec'
    """
    from spammm.quantum.DFTB import Grid_dftb as ocl_grid
    
    # Auto-generate grid spec if not provided
    if grid_spec is None:
        origin, ngrid, step = afm.setup_density_grid(atomPos, step=step, margin=margin, z_extra=z_extra)
        grid_spec = {
            'origin': origin,
            'dA': [step, 0., 0.], 'dB': [0., step, 0.], 'dC': [0., 0., step],
            'ngrid': ngrid.astype(int),
        }
    else:
        origin = grid_spec['origin']
        ngrid = grid_spec['ngrid']
        step = grid_spec['dA'][0]
    
    if fc_instance is None:
        raise NotImplementedError("Fireball density provider needs FireCore instance to compute SCF and get density matrices")
    
    # Get density from FireCore and project using Grid projector
    # This would require:
    # 1. Get sparse density matrices from FireCore
    # 2. Convert to format expected by Grid projector
    # 3. Project to grid
    
    raise NotImplementedError("Fireball density provider needs implementation with density matrix extraction and projection")


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestration Functions (glue AFM.py physics with I/O and plotting)
# ═══════════════════════════════════════════════════════════════════════════════

def compose_and_relax(grads_pauli, grads_es, grads_vdw, scan_xs, scan_ys, heights,
                     origin, step, atomPos, K_LAT=0.5):
    """
    Compose force fields and run probe particle relaxation to get AFM frequency shift.
    
    This is orchestration - combines AFM.py physics functions with force interpolation.
    
    Args:
        grads_pauli: (nx, ny, nz, 3) Pauli gradients from afm.compute_pauli_field
        grads_es: (nx, ny, nz, 3) Electrostatic gradients from afm.compute_es_conv_field
        grads_vdw: (nx, ny, nz, 3) Dispersion gradients from afm.compute_dispersion_grid
        scan_xs: (nx_s,) scan x coordinates
        scan_ys: (ny_s,) scan y coordinates
        heights: (nz_s,) probe heights
        origin: (3,) grid origin
        step: grid spacing
        atomPos: (natoms, 3) atom positions (for mol_z)
        K_LAT: lateral stiffness
        
    Returns:
        df: (nx_s, ny_s, nz_s) frequency shift array
        tip_disp: dict with 'dx' and 'dy' displacement arrays (nx_s, ny_s, nz_s)
    """
    from scipy.ndimage import map_coordinates
    
    # F = -grad E
    F_total = -(grads_pauli + grads_es + grads_vdw)
    
    def force_func(positions):
        """Interpolate forces at arbitrary positions from scan-grid force field."""
        ix = (positions[:, 0] - origin[0]) / step
        iy = (positions[:, 1] - origin[1]) / step
        iz = (positions[:, 2] - origin[2]) / step
        coords = np.vstack([ix, iy, iz])
        fx = map_coordinates(F_total[..., 0], coords, order=1)
        fy = map_coordinates(F_total[..., 1], coords, order=1)
        fz = map_coordinates(F_total[..., 2], coords, order=1)
        return np.stack([fx, fy, fz], axis=-1)
    
    mol_z = atomPos[:,2].max()
    FEs_relax, tip_disp = afm.pp_relax_2d(force_func, scan_xs, scan_ys, heights, mol_z=mol_z, K_LAT=K_LAT, N_RELAX=50, step=step)
    df = afm.compute_df(FEs_relax[:,:,:,2], heights[1]-heights[0])
    return df, tip_disp


def compose_and_relax_total(F_total, scan_xs, scan_ys, heights, origin, step, atomPos, K_LAT=None,  K_RAD=20.0, bond_length=4.0,  use_gpu_relax=True, ppm_mode=False, afmulator=None, reuse_fdbm_grid=False):
    """
    Compose force field from total force field and run probe particle relaxation.

    Args:
        F_total:       (nx, ny, nz, 4) total force field (Fx, Fy, Fz, E) where F = -grad(E)
        scan_xs:       (nx_s,) scan x coordinates
        scan_ys:       (ny_s,) scan y coordinates
        heights:       (nz_s,) probe/tip-apex heights above mol_z
        origin:        (3,) grid origin
        step:          grid spacing
        atomPos:       (natoms, 3) atom positions (for mol_z)
        K_LAT:         lateral stiffness [eV/Å²]; default Hapala 0.5 N/m ≈ 0.031 eV/Å²
        K_RAD:         radial stiffness [eV/Å²]
        use_gpu_relax: True (default) = GPU relaxStrokes; False = legacy CPU scipy
        ppm_mode:      False (default) = 2D lateral-only relaxation (z fixed per slice);
                       True = spherical PPM radial bond (CO-tip, L=bond_length, Kr=K_RAD)
        afmulator:     AFMulator instance; created if None
        reuse_fdbm_grid: if True, skip setup_fdbm_grid (fast-S3 already uploaded img_FF_fdbm)

    Returns:
        df:        (nx_s, ny_s, nz_s) frequency shift array
        tip_disp:  dict with 'dx','dy' (nx_s, ny_s, nz_s) tip displacement
        FEs_relax: (nx_s, ny_s, nz_s, 4) forces at relaxed positions
    """
    if K_LAT is None:
        K_LAT = afm.K_LAT_HAPALA_EV_A2
    mol_z     = float(atomPos[:,2].max())
    nx, ny, nz_ff = F_total.shape[:3]

    if use_gpu_relax:
        if ppm_mode:
            from spammm.globals import debug_print
            debug_print(1, "  [compose_and_relax_total] GPU relaxStrokes spherical PPM "
                  f"(L={bond_length}Å, K_LAT={K_LAT:.4f} eV/Å² = {afm.stiffness_eVA2_to_Nm(K_LAT):.2f} N/m, K_RAD={K_RAD})")
            if afmulator is None:
                afmulator = afm.AFMulator(use_morse=False, nloc=32, use_fire=False)
            if not reuse_fdbm_grid:
                afmulator.setup_fdbm_grid(F_total, origin, step)
            # Smaller dt=0.1, damp=0.3 for stability with weak forces (probe far from surface)
            relax_pars_ppm = [0.1, 0.1, 0.03, 0.1]  # dt, damp, alpha, dt_fire
            FEs_relax, tip_disp = afmulator.scan_fdbm( scan_xs, scan_ys, heights, mol_z=mol_z,  K_LAT=K_LAT, K_RAD=K_RAD, bond_length=bond_length,  relax_pars=relax_pars_ppm )
            # Diagnostic: report maximum displacement for each z-height
            from spammm.globals import DEBUG_PRINT_LEVEL
            if DEBUG_PRINT_LEVEL >= 2:
                print("  [compose_and_relax_total] Tip displacement diagnostics:")
                for iz, h in enumerate(heights):
                    dx_max = np.abs(tip_disp['dx'][:,:,iz]).max()
                    dy_max = np.abs(tip_disp['dy'][:,:,iz]).max()
                    print(f"    z={h:.2f}A: max|dx|={dx_max:.4f}A, max|dy|={dy_max:.4f}A")
        else:
            print("  [compose_and_relax_total] GPU relaxStrokes2D 2D lateral-only")
            if afmulator is None:
                afmulator = afm.AFMulator(use_morse=False, nloc=32, use_fire=False)
            if not reuse_fdbm_grid:
                afmulator.setup_fdbm_grid(F_total, origin, step)
            FEs_relax, tip_disp = afmulator.scan_fdbm_2d(scan_xs, scan_ys, heights, mol_z=mol_z, K_LAT=K_LAT)
            # Diagnostic: report maximum displacement for each z-height
            from spammm.globals import DEBUG_PRINT_LEVEL
            if DEBUG_PRINT_LEVEL >= 2:
                print("  [compose_and_relax_total] Tip displacement diagnostics:")
                for iz, h in enumerate(heights):
                    dx_max = np.abs(tip_disp['dx'][:,:,iz]).max()
                    dy_max = np.abs(tip_disp['dy'][:,:,iz]).max()
                    print(f"    z={h:.2f}A: max|dx|={dx_max:.4f}A, max|dy|={dy_max:.4f}A")
    else:
        print("  [compose_and_relax_total] CPU scipy relaxation (legacy)")
        from scipy.ndimage import map_coordinates
        F_total_3 = F_total[..., :3]  # Extract (Fx,Fy,Fz) for CPU interpolation
        def force_func(positions):
            # -0.5 offset to match GPU corner convention (cell-center vs cell-corner)
            ix = (positions[:, 0] - origin[0]) / step - 0.5
            iy = (positions[:, 1] - origin[1]) / step - 0.5
            iz = (positions[:, 2] - origin[2]) / step - 0.5
            coords = np.vstack([ix, iy, iz])
            fx = map_coordinates(F_total_3[..., 0], coords, order=1)
            fy = map_coordinates(F_total_3[..., 1], coords, order=1)
            fz = map_coordinates(F_total_3[..., 2], coords, order=1)
            return np.stack([fx, fy, fz], axis=-1)
        FEs_relax, tip_disp = afm.pp_relax_2d(force_func, scan_xs, scan_ys, heights, mol_z=mol_z, K_LAT=K_LAT, N_RELAX=50, step=step)

    df = afm.compute_df(FEs_relax[:,:,:,2], heights[1]-heights[0])
    return df, tip_disp, FEs_relax


# ═══════════════════════════════════════════════════════════════════════════════
# GPU vs CPU Interpolation Debugging Functions
# ═══════════════════════════════════════════════════════════════════════════════

def compare_gpu_cpu_interpolation(grads_total, origin, step, atomPos,
                                  z_levels=[2.5, 3.0, 3.5], nxy=(80, 80),
                                  marker_point=None,
                                  output_path='/tmp/gpu_cpu_interp_comparison.png'):
    """
    Compare GPU OpenCL image sampling vs CPU scipy map_coordinates interpolation.
    
    Uses EXACT same scan box as AFM.run_scan():
    - Scan covers 90% of molecule span with 5% margins
    - Each panel has independent symmetric diverging colormap (vmin=-vmax, vcenter=0)
    - Shows CPU vs GPU for XY slices and XZ/YZ cross-sections
    
    Args:
        grads_total: (nx, ny, nz, 3) total gradient
        origin: (3,) grid origin
        step: grid spacing
        atomPos: (natoms, 3) atom positions
        z_levels: list of heights above molecule to sample for XY slices
        nxy: (nx, ny) scan grid resolution
        marker_point: (x, y) tuple to mark on XY slices
        output_path: path to save comparison plot
    """
    from scipy.ndimage import map_coordinates
    from spammm.SPM import AFM as afm
    import pyopencl as cl
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    F_total = -grads_total  # F = -grad(E)
    nx, ny, nz = F_total.shape[:3]
    mol_z = atomPos[:,2].max()
    
    # Compute scan box with 4A margin (consistent with AFM conventions)
    MARGIN = 4.0  # Angstrom margin around molecule
    mn, mx = atomPos.min(axis=0), atomPos.max(axis=0)
    x0 = mn[0] - MARGIN
    y0 = mn[1] - MARGIN
    x1 = mx[0] + MARGIN
    y1 = mx[1] + MARGIN
    dx = (x1 - x0) / max(nxy[0]-1, 1)
    dy = (y1 - y0) / max(nxy[1]-1, 1)
    
    xs = np.array([x0 + dx*ix for ix in range(nxy[0])])
    ys = np.array([y0 + dy*iy for iy in range(nxy[1])])
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    print(f"Molecule bbox: x=[{mn[0]:.2f},{mx[0]:.2f}], y=[{mn[1]:.2f},{mx[1]:.2f}], z_max={mol_z:.2f}")
    print(f"Scan box (4A margin): x=[{xs[0]:.2f},{xs[-1]:.2f}], y=[{ys[0]:.2f},{ys[-1]:.2f}]")
    print(f"Force field: {nx}x{ny}x{nz}, origin={origin}, step={step}")
    
    # Setup GPU
    afmulator = afm.AFMulator(use_morse=False, nloc=32, use_fire=False)
    F_total_4 = np.zeros((nx, ny, nz, 4), dtype=np.float32)
    F_total_4[..., :3] = F_total
    afmulator.setup_fdbm_grid(F_total_4, origin, step)
    
    # GPU sampling using the same interpFE from AFM.cl as used in relaxStrokes
    import os as _os
    kernel_dir = _os.path.join(_os.path.dirname(afm.__file__), '..', '..', 'kernels')
    afm_cl_path = _os.path.join(kernel_dir, 'AFM.cl')
    with open(afm_cl_path) as f: afm_cl_src = f.read()
    kernel_src = afm_cl_src + '''
__kernel void sampleFE(__read_only image3d_t img, __global float4* pts, __global float4* out, float4 dA, float4 dB, float4 dC){
    int gid = get_global_id(0);
    out[gid] = interpFE(pts[gid].xyz, dA, dB, dC, img);
}'''
    prg = cl.Program(afmulator.ctx, kernel_src).build()

    def sample_gpu(pts_flat):
        """Sample force field at given points using GPU interpFE (same as relaxStrokes)."""
        n_pts = pts_flat.shape[0]
        pts_buf = cl.Buffer(afmulator.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pts_flat)
        out_buf = cl.Buffer(afmulator.ctx, cl.mem_flags.WRITE_ONLY, size=n_pts * 16)
        prg.sampleFE(afmulator.queue, (n_pts,), (1,), afmulator.img_FF_fdbm, pts_buf, out_buf,
                     afmulator.fdbm_dinvA, afmulator.fdbm_dinvB, afmulator.fdbm_dinvC)
        fe_out = np.zeros(n_pts * 4, dtype=np.float32)
        cl.enqueue_copy(afmulator.queue, fe_out, out_buf)
        return fe_out.reshape(n_pts, 4)
    
    # Z grid for XZ/YZ cuts - relative to mol_z (probe positions, not tip positions)
    z_range = 8.0  # z span for cuts
    zs_cut = np.linspace(mol_z - 1.0, mol_z + z_range, 60)
    y_cut = (mn[1] + mx[1]) * 0.5  # Center y of molecule
    x_cut = (mn[0] + mx[0]) * 0.5  # Center x of molecule
    
    # Create figure: rows for z_levels, columns: XY_CPU, XY_GPU, XZ_CPU, XZ_GPU, YZ_CPU, YZ_GPU
    n_z = len(z_levels)
    fig, axes = plt.subplots(n_z, 6, figsize=(24, 4*n_z))
    if n_z == 1:
        axes = axes.reshape(1, -1)
    
    for iz, z_level in enumerate(z_levels):
        probe_z = z_level + mol_z
        iz_coord = (probe_z - origin[2]) / step
        
        # CPU XY interpolation at this height (with -0.5 offset for GPU corner convention)
        ix_coords = (XX - origin[0]) / step - 0.5
        iy_coords = (YY - origin[1]) / step - 0.5
        coords = np.array([ix_coords.ravel(), iy_coords.ravel(), np.full(ix_coords.size, iz_coord - 0.5)])
        fz_cpu_xy = map_coordinates(F_total[..., 2], coords, order=1).reshape(XX.shape)
        
        # GPU XY sampling
        pts_xy = np.zeros((XX.size, 4), dtype=np.float32)
        pts_xy[:, 0] = XX.ravel()
        pts_xy[:, 1] = YY.ravel()
        pts_xy[:, 2] = probe_z
        fe_gpu_xy = sample_gpu(pts_xy)
        fz_gpu_xy = fe_gpu_xy[:, 2].reshape(XX.shape)
        
        # XZ cut at y=y_cut (2D: x vs z)
        XX_xz, ZZ_xz = np.meshgrid(xs, zs_cut, indexing='ij')
        coords_xz = np.array([(XX_xz.ravel() - origin[0]) / step - 0.5,
                              np.full(XX_xz.size, (y_cut - origin[1]) / step - 0.5),
                              (ZZ_xz.ravel() - origin[2]) / step - 0.5])
        fz_cpu_xz = map_coordinates(F_total[..., 2], coords_xz, order=1).reshape(XX_xz.shape)
        
        pts_xz = np.zeros((XX_xz.size, 4), dtype=np.float32)
        pts_xz[:, 0] = XX_xz.ravel()
        pts_xz[:, 1] = y_cut
        pts_xz[:, 2] = ZZ_xz.ravel()
        fe_gpu_xz = sample_gpu(pts_xz)
        fz_gpu_xz = fe_gpu_xz[:, 2].reshape(XX_xz.shape)
        
        # YZ cut at x=x_cut (2D: y vs z)
        YY_yz, ZZ_yz = np.meshgrid(ys, zs_cut, indexing='ij')
        coords_yz = np.array([np.full(YY_yz.size, (x_cut - origin[0]) / step - 0.5),
                              (YY_yz.ravel() - origin[1]) / step - 0.5,
                              (ZZ_yz.ravel() - origin[2]) / step - 0.5])
        fz_cpu_yz = map_coordinates(F_total[..., 2], coords_yz, order=1).reshape(YY_yz.shape)
        
        pts_yz = np.zeros((YY_yz.size, 4), dtype=np.float32)
        pts_yz[:, 0] = x_cut
        pts_yz[:, 1] = YY_yz.ravel()
        pts_yz[:, 2] = ZZ_yz.ravel()
        fe_gpu_yz = sample_gpu(pts_yz)
        fz_gpu_yz = fe_gpu_yz[:, 2].reshape(YY_yz.shape)
        
        # Helper to plot with per-panel symmetric diverging colormap
        def plot_panel(ax, data, title, extent):
            vmax = max(np.abs(data.min()), np.abs(data.max()), 1e-6)
            im = ax.imshow(data.T, origin='lower', cmap='RdBu_r', 
                          vmin=-vmax, vmax=vmax, extent=extent, aspect='auto')
            ax.set_title(f'{title}\n±{vmax:.2f}')
            plt.colorbar(im, ax=ax, shrink=0.7)
            return vmax
        
        ext_xy = [xs[0], xs[-1], ys[0], ys[-1]]
        ext_xz = [xs[0], xs[-1], zs_cut[0], zs_cut[-1]]
        ext_yz = [ys[0], ys[-1], zs_cut[0], zs_cut[-1]]
        
        # Row iz: XY_CPU, XY_GPU, XZ_CPU, XZ_GPU, YZ_CPU, YZ_GPU
        plot_panel(axes[iz, 0], fz_cpu_xy, f'CPU XY z={z_level:.1f}A', ext_xy)
        plot_panel(axes[iz, 1], fz_gpu_xy, f'GPU XY z={z_level:.1f}A', ext_xy)
        plot_panel(axes[iz, 2], fz_cpu_xz, f'CPU XZ y={y_cut:.1f}A', ext_xz)
        plot_panel(axes[iz, 3], fz_gpu_xz, f'GPU XZ y={y_cut:.1f}A', ext_xz)
        plot_panel(axes[iz, 4], fz_cpu_yz, f'CPU YZ x={x_cut:.1f}A', ext_yz)
        plot_panel(axes[iz, 5], fz_gpu_yz, f'GPU YZ x={x_cut:.1f}A', ext_yz)
        
        axes[iz, 0].set_ylabel('y [A]')
        for col in [2, 3]:
            axes[iz, col].set_ylabel('z [A]')
        for col in [4, 5]:
            axes[iz, col].set_ylabel('z [A]')
        
        if iz == n_z - 1:
            for col in range(6):
                axes[iz, col].set_xlabel('x [A]' if col < 4 else 'y [A]')
        
        # Stats
        print(f"\nz={z_level:.1f}A: CPU [{fz_cpu_xy.min():.2f}, {fz_cpu_xy.max():.2f}], "
              f"GPU [{fz_gpu_xy.min():.2f}, {fz_gpu_xy.max():.2f}]")
        diff = fz_gpu_xy - fz_cpu_xy
        print(f"  Diff: RMS={np.sqrt(np.mean(diff**2)):.3f}, max|diff|={np.abs(diff).max():.3f}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'\nSaved comparison to {output_path}')
    
    # Additional: Separate figure for XY slices only
    fig_xy, axes_xy = plt.subplots(n_z, 3, figsize=(18, 4*n_z))
    if n_z == 1:
        axes_xy = axes_xy.reshape(1, -1)
    for iz, z_level in enumerate(z_levels):
        probe_z = z_level + mol_z
        ix_coords = (XX - origin[0]) / step - 0.5
        iy_coords = (YY - origin[1]) / step - 0.5
        iz_coord = (probe_z - origin[2]) / step - 0.5
        coords = np.array([ix_coords.ravel(), iy_coords.ravel(), np.full(ix_coords.size, iz_coord)])
        fz_cpu = map_coordinates(F_total[..., 2], coords, order=1).reshape(XX.shape)
        pts_xy = np.zeros((XX.size, 4), dtype=np.float32)
        pts_xy[:, 0] = XX.ravel(); pts_xy[:, 1] = YY.ravel(); pts_xy[:, 2] = probe_z
        fz_gpu = sample_gpu(pts_xy)[:, 2].reshape(XX.shape)
        diff = fz_gpu - fz_cpu
        for col, (data, title) in enumerate([(fz_cpu, f'CPU Fz z={z_level:.1f}Å'),
                                              (fz_gpu, f'GPU Fz z={z_level:.1f}Å'),
                                              (diff, f'Diff z={z_level:.1f}Å')]):
            ax = axes_xy[iz, col]
            vmax = max(np.abs(data.min()), np.abs(data.max()), 1e-6)
            im = ax.imshow(data.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect='auto')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, shrink=0.7)
            # Add marker if specified
            if marker_point is not None:
                ax.plot(marker_point[0], marker_point[1], 'g+', markersize=15, markeredgewidth=2)
    plt.tight_layout()
    fig_xy_path = output_path.replace('.png', '_XYonly.png')
    plt.savefig(fig_xy_path, dpi=150)
    plt.close()
    print(f'Saved XY only to {fig_xy_path}')
    
    # Additional: Separate figure for XZ/YZ center cuts only
    fig_cuts, axes_cuts = plt.subplots(2, 2, figsize=(14, 12))
    # XZ cut (with -0.5 offset)
    XX_xz, ZZ_xz = np.meshgrid(xs, zs_cut, indexing='ij')
    coords_xz = np.array([(XX_xz.ravel() - origin[0]) / step - 0.5, np.full(XX_xz.size, (y_cut - origin[1]) / step - 0.5), (ZZ_xz.ravel() - origin[2]) / step - 0.5])
    fz_cpu_xz = map_coordinates(F_total[..., 2], coords_xz, order=1).reshape(XX_xz.shape)
    pts_xz = np.zeros((XX_xz.size, 4), dtype=np.float32); pts_xz[:, 0] = XX_xz.ravel(); pts_xz[:, 1] = y_cut; pts_xz[:, 2] = ZZ_xz.ravel()
    fz_gpu_xz = sample_gpu(pts_xz)[:, 2].reshape(XX_xz.shape)
    # YZ cut (with -0.5 offset)
    YY_yz, ZZ_yz = np.meshgrid(ys, zs_cut, indexing='ij')
    coords_yz = np.array([np.full(YY_yz.size, (x_cut - origin[0]) / step - 0.5), (YY_yz.ravel() - origin[1]) / step - 0.5, (ZZ_yz.ravel() - origin[2]) / step - 0.5])
    fz_cpu_yz = map_coordinates(F_total[..., 2], coords_yz, order=1).reshape(YY_yz.shape)
    pts_yz = np.zeros((YY_yz.size, 4), dtype=np.float32); pts_yz[:, 0] = x_cut; pts_yz[:, 1] = YY_yz.ravel(); pts_yz[:, 2] = ZZ_yz.ravel()
    fz_gpu_yz = sample_gpu(pts_yz)[:, 2].reshape(YY_yz.shape)
    for row, (name, x_coords, z_coords, f_cpu, f_gpu) in enumerate([('XZ', xs, zs_cut, fz_cpu_xz, fz_gpu_xz), ('YZ', ys, zs_cut, fz_cpu_yz, fz_gpu_yz)]):
        for col, (data, title) in enumerate([(f_cpu, f'{name} CPU'), (f_gpu, f'{name} GPU')]):
            ax = axes_cuts[row, col]
            vmax = max(np.abs(data.min()), np.abs(data.max()), 1e-6)
            im = ax.imshow(data.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]], aspect='auto')
            ax.set_title(f'{title} (center cut)')
            ax.axhline(y=mol_z, color='g', linestyle='--', linewidth=0.5)
            plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    fig_cuts_path = output_path.replace('.png', '_Cuts.png')
    plt.savefig(fig_cuts_path, dpi=150)
    plt.close()
    print(f'Saved center cuts to {fig_cuts_path}')
    
    # Additional: High-res 1D profiles at center (all force components)
    x_center = (mn[0] + mx[0]) * 0.5
    y_center = (mn[1] + mx[1]) * 0.5
    z_range_1d = np.arange(mol_z - 1.0, mol_z + 8.0, 0.02)  # 0.02A step
    # CPU (with -0.5 offset)
    coords_1d = np.array([np.full(z_range_1d.size, (x_center - origin[0]) / step - 0.5), np.full(z_range_1d.size, (y_center - origin[1]) / step - 0.5), (z_range_1d - origin[2]) / step - 0.5])
    fx_cpu_1d = map_coordinates(F_total[..., 0], coords_1d, order=1)
    fy_cpu_1d = map_coordinates(F_total[..., 1], coords_1d, order=1)
    fz_cpu_1d = map_coordinates(F_total[..., 2], coords_1d, order=1)
    # GPU
    pts_1d = np.zeros((z_range_1d.size, 4), dtype=np.float32)
    pts_1d[:, 0] = x_center; pts_1d[:, 1] = y_center; pts_1d[:, 2] = z_range_1d
    fe_gpu_1d = sample_gpu(pts_1d)
    fx_gpu_1d, fy_gpu_1d, fz_gpu_1d = fe_gpu_1d[:, 0], fe_gpu_1d[:, 1], fe_gpu_1d[:, 2]
    # Plot
    fig_1d, axes_1d = plt.subplots(2, 2, figsize=(14, 10))
    for i, (comp, f_cpu, f_gpu) in enumerate([('Fx', fx_cpu_1d, fx_gpu_1d), ('Fy', fy_cpu_1d, fy_gpu_1d), ('Fz', fz_cpu_1d, fz_gpu_1d)]):
        ax = axes_1d[i // 2, i % 2]
        ax.plot(z_range_1d - mol_z, f_cpu, 'b-', linewidth=0.5, label='CPU')
        ax.plot(z_range_1d - mol_z, f_gpu, 'r-', linewidth=0.5, label='GPU')
        diff = f_gpu - f_cpu
        ax.set_title(f'{comp} 1D profile center (x={x_center:.2f}, y={y_center:.2f})  RMS={np.sqrt(np.mean(diff**2)):.4f}')
        ax.set_xlabel('z - mol_z [Å]'); ax.set_ylabel(f'{comp} [eV/Å]')
        ax.axvline(x=0, color='g', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best'); ax.grid(True, alpha=0.3)
    # 4th panel: all |F| together
    ax = axes_1d[1, 1]
    ax.plot(z_range_1d - mol_z, np.abs(fx_cpu_1d), 'b-', linewidth=0.5, label='|Fx| CPU')
    ax.plot(z_range_1d - mol_z, np.abs(fx_gpu_1d), 'r-', linewidth=0.5, label='|Fx| GPU')
    ax.plot(z_range_1d - mol_z, np.abs(fy_cpu_1d), 'b--', linewidth=0.5, label='|Fy| CPU')
    ax.plot(z_range_1d - mol_z, np.abs(fy_gpu_1d), 'r--', linewidth=0.5, label='|Fy| GPU')
    ax.plot(z_range_1d - mol_z, np.abs(fz_cpu_1d), 'b:', linewidth=0.5, label='|Fz| CPU')
    ax.plot(z_range_1d - mol_z, np.abs(fz_gpu_1d), 'r:', linewidth=0.5, label='|Fz| GPU')
    ax.set_title('|F| components at center'); ax.set_xlabel('z - mol_z [Å]'); ax.set_ylabel('|F| [eV/Å]')
    ax.axvline(x=0, color='g', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(loc='best', fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_1d_path = output_path.replace('.png', '_1D.png')
    plt.savefig(fig_1d_path, dpi=150)
    plt.close()
    print(f'Saved 1D profiles to {fig_1d_path}')
    
    return output_path


def compare_1d_at_position(grads_total, origin, step, atomPos, x_pos, y_pos, mol_z=0.0,
                          z_min=-1.0, z_max=8.0, z_step=0.02, E_total=None,
                          output_path='/tmp/gpu_cpu_1d_test.png'):
    """Test CPU vs GPU 1D profiles at specific (x,y) position using existing interpFE kernel.
    
    Args:
        grads_total: (nx, ny, nz, 3) gradient of total energy
        origin: (3,) grid origin
        step: grid spacing
        atomPos: (natoms, 3) atom positions
        x_pos, y_pos: position for 1D scan
        mol_z: molecule z reference
        z_min, z_max, z_step: z range and step
        E_total: (nx, ny, nz) total energy field (optional, for energy interpolation)
        output_path: path to save plot
    """
    import sys
    from spammm.SPM import AFM as afm
    import pyopencl as cl
    from scipy.ndimage import map_coordinates
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    F_total = -grads_total
    afmulator = afm.AFMulator(use_morse=False, nloc=32, use_fire=False)
    nx, ny, nz = F_total.shape[:3]
    F_total_4 = np.zeros((nx, ny, nz, 4), dtype=np.float32)
    F_total_4[..., :3] = F_total
    if E_total is not None:
        F_total_4[..., 3] = E_total  # Energy in 4th component
    afmulator.setup_fdbm_grid(F_total_4, origin, step)

    z_range = np.arange(mol_z + z_min, mol_z + z_max, z_step)

    # CPU sampling - interpolate from SAME F_total_4 array as GPU
    # Adjust by -0.5 to match GPU corner-based interpolation
    coords_1d = np.array([
        np.full(z_range.size, (x_pos - origin[0]) / step - 0.5),
        np.full(z_range.size, (y_pos - origin[1]) / step - 0.5),
        (z_range - origin[2]) / step - 0.5
    ])
    fx_cpu = map_coordinates(F_total_4[..., 0], coords_1d, order=1)
    fy_cpu = map_coordinates(F_total_4[..., 1], coords_1d, order=1)
    fz_cpu = map_coordinates(F_total_4[..., 2], coords_1d, order=1)
    if E_total is not None:
        E_cpu = map_coordinates(F_total_4[..., 3], coords_1d, order=1)

    # GPU sampling using existing interpFE kernel from relax.cl
    pts_1d = np.zeros((z_range.size, 4), dtype=np.float32)
    pts_1d[:, 0] = x_pos
    pts_1d[:, 1] = y_pos
    pts_1d[:, 2] = z_range
    pts_1d[:, 3] = 1.0  # w component for coordinate transform

    # Include full AFM.cl source to use interpFE function
    import os
    kernel_dir = os.path.join(os.path.dirname(afm.__file__), '..', '..', 'kernels')
    afm_cl_path = os.path.join(kernel_dir, 'AFM.cl')
    with open(afm_cl_path) as f:
        afm_cl_src = f.read()
    
    # Add sampling kernel at end of AFM.cl source
    kernel_src = afm_cl_src + '''
__kernel void sample_interpFE(__read_only image3d_t img, __global float4* pts, __global float4* out, float4 dA, float4 dB, float4 dC, int n){
    int gid = get_global_id(0);
    if (gid >= n) return;
    float3 p = pts[gid].xyz;
    out[gid] = interpFE(p, dA, dB, dC, img);
}'''
    prg = cl.Program(afmulator.ctx, kernel_src).build()
    n_pts = z_range.size
    pts_buf = cl.Buffer(afmulator.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pts_1d)
    out_buf = cl.Buffer(afmulator.ctx, cl.mem_flags.WRITE_ONLY, size=n_pts * 16)
    prg.sample_interpFE(afmulator.queue, (n_pts,), (1,), afmulator.img_FF_fdbm, pts_buf, out_buf,
                        afmulator.fdbm_dinvA, afmulator.fdbm_dinvB, afmulator.fdbm_dinvC, np.int32(n_pts))
    fe_out = np.zeros(n_pts * 4, dtype=np.float32)
    cl.enqueue_copy(afmulator.queue, fe_out, out_buf)
    fe_out = fe_out.reshape(n_pts, 4)
    fx_gpu, fy_gpu, fz_gpu = fe_out[:, 0], fe_out[:, 1], fe_out[:, 2]
    if E_total is not None:
        E_gpu = fe_out[:, 3]

    # Plot - 4 panels: Fx, Fy, Fz, E (matching float4 layout)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Fx, Fy, Fz
    for i, (comp, f_cpu, f_gpu) in enumerate([('Fx', fx_cpu, fx_gpu), ('Fy', fy_cpu, fy_gpu), ('Fz', fz_cpu, fz_gpu)]):
        ax = axes[i]
        ax.plot(z_range - mol_z, f_cpu, 'b-', linewidth=0.5, label='CPU')
        ax.plot(z_range - mol_z, f_gpu, 'r-', linewidth=0.5, label='GPU')
        diff = f_gpu - f_cpu
        rms = np.sqrt(np.mean(diff**2))
        ratio = np.abs(f_gpu).max() / (np.abs(f_cpu).max() + 1e-10)
        ax.set_title(f'{comp} at ({x_pos:.1f},{y_pos:.1f})\\nRMS={rms:.4f}, ratio={ratio:.2f}')
        ax.set_xlabel('z - mol_z [Å]')
        ax.set_ylabel(f'{comp} [eV/Å]')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Energy (4th component)
    ax = axes[3]
    if E_total is not None:
        ax.plot(z_range - mol_z, E_cpu, 'b-', linewidth=0.5, label='CPU')
        ax.plot(z_range - mol_z, E_gpu, 'r-', linewidth=0.5, label='GPU')
        diff = E_gpu - E_cpu
        rms = np.sqrt(np.mean(diff**2))
        ax.set_title(f'E at ({x_pos:.1f},{y_pos:.1f})\\nRMS={rms:.4f}')
        ax.set_xlabel('z - mol_z [Å]')
        ax.set_ylabel('E [eV]')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Energy not available\\n(pass E_total parameter)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('E (not available)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f'Saved {output_path}')
    print(f'  Fx: CPU={np.abs(fx_cpu).max():.3f}, GPU={np.abs(fx_gpu).max():.3f}, ratio={np.abs(fx_gpu).max()/np.abs(fx_cpu).max():.2f}')
    print(f'  Fy: CPU={np.abs(fy_cpu).max():.3f}, GPU={np.abs(fy_gpu).max():.3f}, ratio={np.abs(fy_gpu).max()/np.abs(fy_cpu).max():.2f}')
    print(f'  Fz: CPU={np.abs(fz_cpu).max():.3f}, GPU={np.abs(fz_gpu).max():.3f}, ratio={np.abs(fz_gpu).max()/np.abs(fz_cpu).max():.2f}')

    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Step Plotting Functions (separate from computation)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_step1_outputs(rho_grid, rho_na_grid, rho_diff, step_dir, origin, step):
    """Plot step 1 density outputs."""
    from spammm import plotUtils as pu
    z_slice = 2.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Step 1: Density Projection')
    
    pu.plot_field_slice(axes[0], rho_grid, origin, step, z_slice, cmap='magma', 
                       title='SCF Density [e/Å³]')
    pu.plot_field_slice(axes[1], rho_na_grid, origin, step, z_slice, cmap='magma',
                       title='Neutral Atom Density [e/Å³]')
    pu.plot_field_slice(axes[2], rho_diff, origin, step, z_slice, cmap='bwr', sym=True,
                       title='Delta Density [e/Å³]')
    
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'step1_rho_slices.png'), dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved step1 density plots")


def plot_step2_outputs(V_ES, step_dir, origin, step):
    """Plot step 2 electrostatics outputs."""
    from spammm import plotUtils as pu
    z_slice = 2.0
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    pu.plot_field_slice(ax, V_ES, origin, step, z_slice, cmap='bwr', sym=True,
                       title='Electrostatic Potential [eV]')
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'step2_VES_slices.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved step2 electrostatics plot")


def plot_step3_outputs(E_pauli_field, grads_pauli, step_dir, origin, step, A_pauli, beta_pauli):
    """Plot step 3 Pauli outputs."""
    from spammm import plotUtils as pu
    z_slice = 2.0
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    pu.plot_field_slice(ax, E_pauli_field, origin, step, z_slice, cmap='magma',
                       title=f'Pauli Energy [eV] (A={A_pauli:.1f}, b={beta_pauli:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'step3_Epauli_slices.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved step3 Pauli plot")


def plot_step4_outputs(E_ES_field, grads_ES, step_dir, origin, step):
    """Plot step 4 electrostatics convolution outputs."""
    from spammm import plotUtils as pu
    z_slice = 2.0
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    pu.plot_field_slice(ax, E_ES_field, origin, step, z_slice, cmap='bwr', sym=True,
                       title='ES Energy [eV]')
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'step4_EES_slices.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved step4 ES convolution plot")


def plot_step5_outputs(E_vdw, grads_vdw, step_dir, origin, step):
    """Plot step 5 dispersion outputs."""
    from spammm import plotUtils as pu
    z_slice = 2.0
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    pu.plot_field_slice(ax, E_vdw, origin, step, z_slice, cmap='magma',
                       title='Dispersion Energy [eV]')
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'step5_Evdw_slices.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved step5 dispersion plot")


def plot_step6_outputs(df, scan_xs, scan_ys, heights, step_dir):
    """Plot step 6 final AFM images."""
    save_afm_images(df, scan_xs, scan_ys, heights, step_dir, prefix='df')
    print(f"  Saved step6 AFM images")


def plot_tip_displacement(tip_disp, scan_xs, scan_ys, heights, output_dir, prefix='tip_disp'):
    """Plot tip displacement (dx, dy, total) for each height.

    For each height, creates a row of 3 images:
    - dx displacement (seismic colormap, symmetric around zero)
    - dy displacement (seismic colormap, symmetric around zero)
    - total displacement r = sqrt(dx^2 + dy^2)

    Args:
        tip_disp: dict with 'dx' and 'dy' arrays, each (nx_s, ny_s, nz_s)
        scan_xs: (nx_s,) scan x coordinates
        scan_ys: (ny_s,) scan y coordinates
        heights: (nz_s,) probe heights
        output_dir: directory for PNG output
        prefix: filename prefix
    """
    dx = tip_disp['dx']
    dy = tip_disp['dy']
    nz = len(heights)
    
    # Total displacement
    r = np.sqrt(dx**2 + dy**2)
    
    # Create figure with nz rows, 3 columns
    fig, axes = plt.subplots(nz, 3, figsize=(15, 5*nz))
    if nz == 1:
        axes = axes.reshape(1, 3)
    
    for iz in range(nz):
        h = heights[iz]
        
        # dx with seismic colormap (symmetric)
        vmax_dx = max(abs(dx[:,:,iz].min()), abs(dx[:,:,iz].max()))
        norm_dx = TwoSlopeNorm(vmin=-vmax_dx, vcenter=0, vmax=vmax_dx)
        im_dx = axes[iz, 0].imshow(dx[:,:,iz].T, origin='lower',
                                   extent=[scan_xs[0], scan_xs[-1], scan_ys[0], scan_ys[-1]],
                                   cmap='seismic', norm=norm_dx, aspect='equal')
        axes[iz, 0].set_title(f'dx at h={h:.1f} Å')
        axes[iz, 0].set_xlabel('x [Å]')
        axes[iz, 0].set_ylabel('y [Å]')
        plt.colorbar(im_dx, ax=axes[iz, 0], fraction=0.03, pad=0.02)
        
        # dy with seismic colormap (symmetric)
        vmax_dy = max(abs(dy[:,:,iz].min()), abs(dy[:,:,iz].max()))
        norm_dy = TwoSlopeNorm(vmin=-vmax_dy, vcenter=0, vmax=vmax_dy)
        im_dy = axes[iz, 1].imshow(dy[:,:,iz].T, origin='lower',
                                   extent=[scan_xs[0], scan_xs[-1], scan_ys[0], scan_ys[-1]],
                                   cmap='seismic', norm=norm_dy, aspect='equal')
        axes[iz, 1].set_title(f'dy at h={h:.1f} Å')
        axes[iz, 1].set_xlabel('x [Å]')
        axes[iz, 1].set_ylabel('y [Å]')
        plt.colorbar(im_dy, ax=axes[iz, 1], fraction=0.03, pad=0.02)
        
        # total displacement (magma colormap, non-negative)
        im_r = axes[iz, 2].imshow(r[:,:,iz].T, origin='lower',
                                 extent=[scan_xs[0], scan_xs[-1], scan_ys[0], scan_ys[-1]],
                                 cmap='magma', aspect='equal')
        axes[iz, 2].set_title(f'r = sqrt(dx²+dy²) at h={h:.1f} Å')
        axes[iz, 2].set_xlabel('x [Å]')
        axes[iz, 2].set_ylabel('y [Å]')
        plt.colorbar(im_r, ax=axes[iz, 2], fraction=0.03, pad=0.02)
    
    plt.tight_layout()
    fname = os.path.join(output_dir, f'{prefix}.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved tip displacement plot: {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# STM Computation Functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stm(projector, eigvecs, eigvals, scan_xs, scan_ys, heights,
                norb_per_atom, orb_offsets, atoms_dict,
                lumo_offsets=None, mo_indices=None, field='ldos', use_exp_basis=True,
                exp_beta=1.0, exp_r0=3.0):
    """
    Compute STM signal by projecting LUMO orbitals with exponential radial decay.

    Args:
        projector: GridProjector instance
        eigvecs: (nstates, norb_total) eigenvector matrix
        eigvals: (nstates,) eigenvalue array
        scan_xs: (nx_s,) scan x coordinates
        scan_ys: (ny_s,) scan y coordinates
        heights: (nz_s,) probe heights
        norb_per_atom: (natoms,) orbital counts
        orb_offsets: (natoms+1,) orbital offsets
        atoms_dict: atom data dict
        lumo_offsets: list of HOMO offsets (e.g., [1,2,3] for HOMO+1,+2,+3)
        use_exp_basis: use exponential decay (True) or spline basis (False)
        exp_beta: exponential decay constant (Å^-1)
        exp_r0: reference distance (Å)

    Returns:
        stm_grid: (nx_s, ny_s, nz_s) STM signal (sum of LUMO^2)
    """
    nx_s, ny_s, nz_s = len(scan_xs), len(scan_ys), len(heights)

    homo_idx = None
    if mo_indices is not None:
        mo_list = [int(i) for i in mo_indices]
    else:
        if lumo_offsets is None:
            lumo_offsets = [1, 2, 3]
        # CAVEAT: never use eigvals<0 as HOMO for DFTB (see dftb_frontier_mo_indices).
        if atoms_dict is not None and 'type' in atoms_dict:
            homo_idx, _ = dftb_frontier_mo_indices(eigvals, atomTypes=atoms_dict['type'])
        else:
            raise ValueError(
                "STM: need atoms_dict['type'] for valence HOMO; "
                "eigvals<0 is wrong for DFTB (picks near-zero virtuals). "
                "See dftb_frontier_mo_indices / doc/Reports/STM_ExtendedBasis_OrbitalCompare.md")
        mo_list = [homo_idx + int(off) for off in lumo_offsets]

    nmo = int(eigvecs.shape[0])
    bad = [int(i) for i in mo_list if (int(i) < 0 or int(i) >= nmo)]
    if len(bad) > 0:
        raise ValueError(f"STM: MO indices out of range {bad}; valid=[0,{nmo-1}]")

    if field != 'ldos' and len(mo_list) != 1:
        raise ValueError(f"STM: field='{field}' requires exactly 1 MO, got mo_list={mo_list}")

    if homo_idx is None:
        print(f"  [STM] MOs: {mo_list}")
    else:
        print(f"  [STM] HOMO index: {homo_idx}, MOs: {mo_list}")

    # Generate 2D point grid for each height
    XX, YY = np.meshgrid(scan_xs, scan_ys, indexing='ij')
    stm_grid = np.zeros((nx_s, ny_s, nz_s), dtype=np.float32)

    for iz, h in enumerate(heights):
        points = np.stack([XX.ravel(), YY.ravel(), np.full_like(XX.ravel(), h)], axis=1)
        points = points.astype(np.float32)

        # Project each selected MO
        for imo_i, imo in enumerate(mo_list):
            coeffs = eigvecs[imo].astype(np.float32)
            if iz == 0 and imo_i == 0:
                cmin = float(np.min(coeffs)); cmax = float(np.max(coeffs)); cn = float(np.linalg.norm(coeffs))
                print(f"  [STM] coeffs MO#{imo}: min={cmin:+.3e} max={cmax:+.3e} norm={cn:.6f}")
            if use_exp_basis:
                psi = projector.project_orbital_dense_points_exp(
                    points, coeffs, norb_per_atom, orb_offsets, atoms_dict,
                    beta=exp_beta, r0=exp_r0
                )
            else:
                psi = projector.project_orbital_dense_points(
                    points, coeffs, norb_per_atom, orb_offsets, atoms_dict
                )
            psi_2d = psi.reshape(nx_s, ny_s)
            if field == 'psi':
                stm_grid[:, :, iz] += psi_2d
            elif field == 'psi2':
                stm_grid[:, :, iz] += psi_2d ** 2
            else:  # 'ldos'
                stm_grid[:, :, iz] += psi_2d ** 2

    print(f"  [STM] STM grid shape: {stm_grid.shape}, range: [{stm_grid.min():.4e}, {stm_grid.max():.4e}]")
    return stm_grid


def compute_bond_resolved_stm(projector, eigvecs, eigvals, scan_xs, scan_ys, heights,
                              tip_disp, norb_per_atom, orb_offsets, atoms_dict,
                              lumo_offsets=None, mo_indices=None, field='ldos', use_exp_basis=True,
                              exp_beta=1.0, exp_r0=3.0):
    """
    Compute bond-resolved STM: STM at tip-displaced positions.

    The AFM relaxation displaces the tip laterally (dx, dy). This function
    computes the STM signal at these displaced positions, simulating the
    effect of CO tip bending on the STM image.

    Args:
        tip_disp: dict with 'dx' and 'dy' arrays (nx_s, ny_s, nz_s)
        [other args same as compute_stm]

    Returns:
        stm_grid: (nx_s, ny_s, nz_s) STM signal at displaced positions
    """
    homo_idx = None
    if mo_indices is not None:
        mo_list = [int(i) for i in mo_indices]
    else:
        if lumo_offsets is None:
            lumo_offsets = [1, 2, 3]
        # CAVEAT: never use eigvals<0 as HOMO for DFTB (see dftb_frontier_mo_indices).
        if atoms_dict is not None and 'type' in atoms_dict:
            homo_idx, _ = dftb_frontier_mo_indices(eigvals, atomTypes=atoms_dict['type'])
        else:
            raise ValueError(
                "BR-STM: need atoms_dict['type'] for valence HOMO; "
                "eigvals<0 is wrong for DFTB. See dftb_frontier_mo_indices.")
        mo_list = [homo_idx + int(off) for off in lumo_offsets]

    nmo = int(eigvecs.shape[0])
    bad = [int(i) for i in mo_list if (int(i) < 0 or int(i) >= nmo)]
    if len(bad) > 0:
        raise ValueError(f"BR-STM: MO indices out of range {bad}; valid=[0,{nmo-1}]")

    if field != 'ldos' and len(mo_list) != 1:
        raise ValueError(f"BR-STM: field='{field}' requires exactly 1 MO, got mo_list={mo_list}")

    if homo_idx is None:
        print(f"  [BR-STM] MOs: {mo_list}")
    else:
        print(f"  [BR-STM] HOMO index: {homo_idx}, MOs: {mo_list}")
    print(f"  [BR-STM] Applying tip displacement from AFM relaxation")

    nx_s = len(scan_xs)
    ny_s = len(scan_ys)
    nz_s = len(heights)

    XX, YY = np.meshgrid(scan_xs, scan_ys, indexing='ij')
    stm_grid = np.zeros((nx_s, ny_s, nz_s), dtype=np.float32)

    for iz, h in enumerate(heights):
        # Apply displacement to grid positions
        X_disp = XX + tip_disp['dx'][:, :, iz]
        Y_disp = YY + tip_disp['dy'][:, :, iz]

        points = np.stack([X_disp.ravel(), Y_disp.ravel(), np.full_like(X_disp.ravel(), h)], axis=1)
        points = points.astype(np.float32)

        # Project each selected MO at displaced positions
        for imo_i, imo in enumerate(mo_list):
            coeffs = eigvecs[imo].astype(np.float32)
            if iz == 0 and imo_i == 0:
                cmin = float(np.min(coeffs)); cmax = float(np.max(coeffs)); cn = float(np.linalg.norm(coeffs))
                print(f"  [BR-STM] coeffs MO#{imo}: min={cmin:+.3e} max={cmax:+.3e} norm={cn:.6f}")
            if use_exp_basis:
                psi = projector.project_orbital_dense_points_exp(
                    points, coeffs, norb_per_atom, orb_offsets, atoms_dict,
                    beta=exp_beta, r0=exp_r0
                )
            else:
                psi = projector.project_orbital_dense_points(
                    points, coeffs, norb_per_atom, orb_offsets, atoms_dict
                )
            psi_2d = psi.reshape(nx_s, ny_s)
            if field == 'psi':
                stm_grid[:, :, iz] += psi_2d
            elif field == 'psi2':
                stm_grid[:, :, iz] += psi_2d ** 2
            else:
                stm_grid[:, :, iz] += psi_2d ** 2

    print(f"  [BR-STM] STM grid shape: {stm_grid.shape}, range: [{stm_grid.min():.4e}, {stm_grid.max():.4e}]")
    return stm_grid


def _set_projector_species_basis(projector, atoms_dict, species_list_ang, *, rc_max=None, max_shells=None):
    """Reload/update STO radial table + atom Rcut for orbital projection.

    Always load with rc_max covering prolonged tails (default ≥6 Å) so stock→prolonged
    swaps via the same n_nodes grid; then Rcut matches per-atom orbital cutoffs.
    """
    if rc_max is None:
        rc_max = max(float(orb['cutoff']) for sp in species_list_ang for orb in sp['orbitals'])
        rc_max = max(rc_max, 6.0)
    if max_shells is None:
        max_shells = projector.basis_meta.get('max_shells', 2) if getattr(projector, 'basis_meta', None) else 2
    projector.load_basis_sto(species_list_ang, rc_max=rc_max, max_shells=max_shells)
    sp_by_nz = {sp['atomic_number']: sp for sp in species_list_ang}
    cutoffs = []
    for Z in atoms_dict['type']:
        sp = sp_by_nz[int(Z)]
        cutoffs.append(max(float(orb['cutoff']) for orb in sp['orbitals']))
    atoms_dict['Rcut'] = np.asarray(cutoffs, dtype=np.float64)
    return atoms_dict


# DFTB+/mio/3ob valence electrons per element (SCC filling). Not Z!
DFTB_VALENCE_ELEC = {'H': 1, 'C': 4, 'N': 5, 'O': 6, 'P': 5, 'S': 6, 'Br': 7, 'I': 7}


def dftb_n_valence_electrons(enames=None, atomTypes=None):
    """Total valence electrons for DFTB occupation (closed-shell → n_occ = n_elec//2)."""
    if enames is not None:
        return int(sum(DFTB_VALENCE_ELEC.get(str(e), 4) for e in enames))
    if atomTypes is not None:
        inv = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 15: 'P', 16: 'S', 35: 'Br', 53: 'I'}
        return int(sum(DFTB_VALENCE_ELEC.get(inv.get(int(z), 'C'), 4) for z in atomTypes))
    raise ValueError('dftb_n_valence_electrons: need enames or atomTypes')


def dftb_frontier_mo_indices(eigvals, n_elec=None, enames=None, atomTypes=None):
    """HOMO / LUMO from valence electron count (NOT eigvals<0).

    CAVEAT (SSOT — do not “fix” back to eigvals<0):
      DFTB eigenvalues sit near the Fermi level (~−4 eV for aromatics). Using
      ``eigvals < 0`` wrongly counts empty states between E_F and 0 as occupied and
      returns a near-gap *virtual* as “HOMO” (e.g. pentacene #56 @ −0.18 eV instead of
      #50 @ −4.79 eV). That broke STM/orbital morphology vs pySCF.
      Always use valence n_elec // 2 (DFTB_VALENCE_ELEC / detailed.out “Nr. of up electrons”).
      Report: doc/Reports/STM_ExtendedBasis_OrbitalCompare.md
    """
    eigvals = np.asarray(eigvals)
    if n_elec is None:
        n_elec = dftb_n_valence_electrons(enames=enames, atomTypes=atomTypes)
    if n_elec % 2 != 0:
        raise ValueError(f'dftb_frontier_mo_indices: odd n_elec={n_elec} (open shell not handled)')
    n_occ = n_elec // 2
    if n_occ < 1 or n_occ >= len(eigvals):
        raise ValueError(f'dftb_frontier_mo_indices: n_occ={n_occ} invalid for nMO={len(eigvals)}')
    homo = int(n_occ - 1)
    lumo = homo + 1
    return homo, lumo


def project_mo_xy_slice(projector, coeffs, norb_per_atom, orb_offsets, atoms_dict,
                        scan_xs, scan_ys, z_A, *, use_exp_basis=False):
    """Project one DFTB MO onto a constant-height xy plane (Å). Returns (nx,ny) ψ.

    DFTB STO via OpenCL ``project_orbital_dense_points`` (kernels/LCAO_grid.cl).
    For pySCF / any GTO (incl. def2-SVP double-ζ) use
    ``spammm.quantum.pySCF_utils-new.eval_mo_on_xy_slice`` — full ``numint.eval_ao``,
    no AO truncation (not the DFTB STO kernels).
    """
    XX, YY = np.meshgrid(scan_xs, scan_ys, indexing='ij')
    points = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, float(z_A))], axis=1).astype(np.float32)
    c = np.asarray(coeffs, dtype=np.float32).ravel()
    if use_exp_basis:
        psi = projector.project_orbital_dense_points_exp(
            points, c, norb_per_atom, orb_offsets, atoms_dict)
    else:
        psi = projector.project_orbital_dense_points(
            points, c, norb_per_atom, orb_offsets, atoms_dict)
    return psi.reshape(len(scan_xs), len(scan_ys))


# STM tip orbital coeffs in OpenCL [px, py, pz, s] order (mo_overlap_points_exp_sk).
# φ_tip selects which tip orbital; STM **current** is always ≥0: I ~ |⟨φ_s|H'|φ_t⟩|²
# (kernel returns t and I=t²). Orbital **phase maps** use project_mo_xy_slice (signed ψ).
STM_TIP_ORBITALS = {
    's':  np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    'pz': np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    'py': np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
}


def project_mo_stm_sk_slice(projector, mo_coeff, atoms_dict, basis_ang, enames, species_per_atom,
                            scan_xs, scan_ys, z_A, *, tip_orbital='s',
                            beta=1.0, r0=3.0, rcut=8.0, intensity=True):
    """DFTB MO-resolved STM **current** at constant z (always ≥0).

    Point tip φ_t ∈ {s, pz, py} couples to sample MO φ_s via exp+SK (mo_overlap_points_exp_sk).
    Returns (nx,ny) intensity I=t². Set ``intensity=False`` only for debugging (signed t).
    For signed orbital ψ maps use ``project_mo_xy_slice`` (field='psi').
    """
    from spammm.quantum.DFTB.DFTBplusParser import evec_to_kernel_coeffs
    tip_orbital = str(tip_orbital).lower()
    if tip_orbital not in STM_TIP_ORBITALS:
        raise ValueError(f"tip_orbital must be one of {tuple(STM_TIP_ORBITALS)}, got {tip_orbital!r}")
    natoms = len(enames)
    species_names = list(enames)
    coeffs_smp = evec_to_kernel_coeffs(
        np.asarray(mo_coeff, dtype=np.float64).ravel(), natoms,
        species_per_atom, species_names, basis_ang)
    coeffs_tip = np.tile(STM_TIP_ORBITALS[tip_orbital], (1, 1))
    XX, YY = np.meshgrid(scan_xs, scan_ys, indexing='ij')
    tip_centers = np.stack(
        [XX.ravel(), YY.ravel(), np.full(XX.size, float(z_A))], axis=1).astype(np.float32)
    tip_pos_rel = np.zeros((1, 3), dtype=np.float32)
    smp_pos = np.asarray(atoms_dict['pos'][:natoms], dtype=np.float32)
    t, I = projector.mo_overlap_points_exp_sk(
        tip_centers, tip_pos_rel, smp_pos, coeffs_tip, coeffs_smp,
        beta=float(beta), r0=float(r0), rcut=float(rcut))
    nx, ny = len(scan_xs), len(scan_ys)
    out = I if intensity else t
    return out.reshape(nx, ny)


def plot_eigspectrum_compare(E_dftb_eV, homo_d, E_pyscf_eV, homo_p, out_path, *,
                             n_near=5, title=None, mark_indices_d=None, mark_indices_p=None):
    """Side-by-side eigenvalue ladders. Energy grows **up** (unoccupied above occupied).

    Prefer ``plot_spectrum_with_orbitals`` when orbital maps are available.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 6.5), sharey=True)
    specs = [
        (axes[0], E_dftb_eV, homo_d, 'DFTB', mark_indices_d),
        (axes[1], E_pyscf_eV, homo_p, 'pySCF', mark_indices_p),
    ]
    for ax, E, homo, tag, marks in specs:
        E = np.asarray(E, dtype=np.float64)
        nmo = len(E)
        lumo = homo + 1
        if marks is None:
            lo = max(0, homo - n_near)
            hi = min(nmo - 1, lumo + n_near)
            marks = list(range(lo, hi + 1))
        for i, e in enumerate(E):
            ax.hlines(e, 0.10, 0.55, colors='0.8', lw=0.5, zorder=1)
        for i in marks:
            e = float(E[i])
            col = 'C0' if i <= homo else 'C3'
            ax.hlines(e, 0.15, 0.60, colors=col, lw=2.2, zorder=2)
            rel = i - homo
            if rel == 0:
                lab = 'HOMO'
            elif rel == 1:
                lab = 'LUMO'
            elif rel < 0:
                lab = f'H{rel:+d}'
            else:
                lab = f'L+{rel-1}'
            ax.annotate(f'{lab} #{i}\n{e:.2f} eV',
                        xy=(0.60, e), xytext=(0.72, e),
                        fontsize=6.5, color=col, va='center',
                        arrowprops=dict(arrowstyle='->', color=col, lw=0.9))
        ax.set_xlim(0, 1.35)
        ax.set_xticks([])
        ax.set_title(f'{tag}  HOMO#{homo}  gap={float(E[lumo]-E[homo]):.3f} eV', fontsize=9)
        if tag == 'DFTB':
            ax.set_ylabel('E (eV)  ↑')
        ax.axhline(float(E[homo]), color='C0', ls=':', lw=0.8, alpha=0.4)
        ax.axhline(float(E[lumo]), color='C3', ls=':', lw=0.8, alpha=0.4)
        ymin = float(E[marks[0]]) - 1.0
        ymax = float(E[marks[-1]]) + 1.0
        ax.set_ylim(ymin, ymax)  # low E bottom, high E top
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_spectrum_with_orbitals(E_dftb_eV, homo_d, maps_dftb, E_pyscf_eV, homo_p, maps_pyscf,
                                mo_rel_indices, scan_xs, scan_ys, out_path, *,
                                atom_pos=None, title=None, field='psi', layout='vertical'):
    """Shared center spectrum + maps on both sides; sloped E↔map connectors.

    ``field``:
      - ``'psi'``  — **orbital** map: signed phase (RdBu). No STM tip coupling.
      - ``'psi2'`` / ``'stm'`` / ``'ldos'`` — **STM current** (viridis, ≥0): I∝|matrix element|².

    ``layout``: ``'vertical'`` (E↑) or ``'horizontal'`` (E→, ψ/STM upright).
    """
    from matplotlib.patches import ConnectionPatch
    from matplotlib.ticker import MultipleLocator

    def _lab(k):
        if k == 0:
            return 'HOMO'
        if k == 1:
            return 'LUMO'
        if k < 0:
            return f'H{k:+d}'
        return f'L+{k-1}'

    layout = str(layout).lower().strip()
    if layout not in ('vertical', 'horizontal'):
        raise ValueError(f"layout must be 'vertical' or 'horizontal', got {layout!r}")

    rels_hi = sorted(mo_rel_indices, reverse=True)
    rels_lo = sorted(mo_rel_indices)
    n = len(rels_hi)
    E_d = np.asarray(E_dftb_eV, dtype=np.float64)
    E_p = np.asarray(E_pyscf_eV, dtype=np.float64)
    e_marked = [float(E_d[homo_d + k]) for k in rels_hi] + [float(E_p[homo_p + k]) for k in rels_hi]
    e_lo, e_hi = min(e_marked) - 0.15, max(e_marked) + 0.15
    signed_map = str(field).lower() == 'psi'
    cmap = 'RdBu_r' if signed_map else 'viridis'
    xs0, xs1 = float(scan_xs[0]), float(scan_xs[-1])
    ys0, ys1 = float(scan_ys[0]), float(scan_ys[-1])
    Lx, Ly = xs1 - xs0, ys1 - ys0
    e_ticks = np.arange(np.floor(e_lo * 2) / 2.0, np.ceil(e_hi * 2) / 2.0 + 1e-9, 0.5)

    def _strip_spines(ax):
        ax.set_frame_on(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.patch.set_visible(False)

    def _imshow_psi(ax, arr, *, rotate=False):
        a = np.asarray(arr)
        vmax = float(np.percentile(np.abs(a), 99)) or 1e-30
        vmin = -vmax if signed_map else 0.0
        if rotate:
            ax.imshow(a, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=[ys0, ys1, xs0, xs1], aspect='auto')
            if atom_pos is not None:
                ax.scatter(atom_pos[:, 1], atom_pos[:, 0], c='k', s=1.2, alpha=0.3, zorder=5)
        else:
            ax.imshow(a.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=[xs0, xs1, ys0, ys1], aspect='auto')
            if atom_pos is not None:
                ax.scatter(atom_pos[:, 0], atom_pos[:, 1], c='k', s=1.2, alpha=0.3, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        _strip_spines(ax)

    def _paint_E_scale(ax, axis='y'):
        if axis == 'y':
            ax.set_yticks(e_ticks)
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='major', length=2.0, width=0.5, direction='in')
            ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            ax.tick_params(axis='y', which='minor', length=1.0, width=0.4, direction='in')
            for t in e_ticks:
                if abs(t - round(t)) < 1e-9:
                    ax.text(0.5, t, f'{t:.0f}', fontsize=4.2, ha='center', va='center',
                            color='0.25', zorder=5, clip_on=True)
        else:
            ax.set_xticks(e_ticks)
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='major', length=2.0, width=0.5, direction='in')
            ax.xaxis.set_minor_locator(MultipleLocator(0.25))
            ax.tick_params(axis='x', which='minor', length=1.0, width=0.4, direction='in')
            for t in e_ticks:
                if abs(t - round(t)) < 1e-9:
                    ax.text(t, 0.5, f'{t:.0f}', fontsize=4.2, ha='center', va='center',
                            color='0.25', zorder=5, clip_on=True)
        _strip_spines(ax)

    if layout == 'vertical':
        # small gap (inches) so sloped connectors are visible — not flush, not huge
        gap_in = 0.10
        panel_aspect = Ly / max(Lx, 1e-12)
        row_h_in = 0.52
        panel_w_in = row_h_in / max(panel_aspect, 1e-12)
        spec_w_in = 0.30
        fig_w = 2 * panel_w_in + spec_w_in + 2 * gap_in + 0.10
        fig_h = row_h_in * n + 0.42
        fig = plt.figure(figsize=(fig_w, fig_h))
        top_m, bot_m, side_m = 0.045, 0.01, 0.012
        usable_h = 1.0 - top_m - bot_m
        row_h = usable_h / n
        lab_h = 0.10 * row_h   # room above each ψ for label
        img_h = row_h - lab_h
        gap = gap_in / fig_w
        spec_w = spec_w_in / fig_w
        panel_w = panel_w_in / fig_w
        left_x = side_m
        spec_left = left_x + panel_w + gap
        right_x = spec_left + spec_w + gap
        ax_spec = fig.add_axes([spec_left, bot_m, spec_w, usable_h])
        ax_spec.set_xlim(0.0, 1.0)
        ax_spec.set_ylim(e_lo, e_hi)
        ax_spec.set_xticks([])
        _paint_E_scale(ax_spec, 'y')
        ax_spec.text(0.22, e_hi, 'D', fontsize=5.5, ha='center', va='bottom', color='0.35')
        ax_spec.text(0.78, e_hi, 'P', fontsize=5.5, ha='center', va='bottom', color='0.35')
        ax_spec.axvline(0.5, color='0.85', ls=':', lw=0.3, zorder=0)
        for e in E_d:
            ax_spec.hlines(float(e), 0.00, 0.45, colors='0.9', lw=0.2, zorder=1)
        for e in E_p:
            ax_spec.hlines(float(e), 0.55, 1.00, colors='0.9', lw=0.2, zorder=1)

        for ir, k in enumerate(rels_hi):
            imo_d, imo_p = homo_d + k, homo_p + k
            ed, ep = float(E_d[imo_d]), float(E_p[imo_p])
            col = 'C0' if k <= 0 else 'C3'
            ax_spec.hlines(ed, 0.00, 0.42, colors=col, lw=1.35, zorder=3)
            ax_spec.hlines(ep, 0.58, 1.00, colors=col, lw=1.35, zorder=3)
            y0 = bot_m + (n - 1 - ir) * row_h
            ax_d = fig.add_axes([left_x, y0, panel_w, img_h])
            ax_p = fig.add_axes([right_x, y0, panel_w, img_h])
            _imshow_psi(ax_d, maps_dftb[imo_d])
            _imshow_psi(ax_p, maps_pyscf[imo_p])
            # labels in the lab strip above the image (not inside imshow axes)
            fig.text(left_x + 0.5 * panel_w, y0 + img_h + 0.15 * lab_h,
                     f'{_lab(k)} D#{imo_d} {ed:.2f}', fontsize=4.6, ha='center', va='bottom', color=col)
            fig.text(right_x + 0.5 * panel_w, y0 + img_h + 0.15 * lab_h,
                     f'{_lab(k)} P#{imo_p} {ep:.2f}', fontsize=4.6, ha='center', va='bottom', color=col)
            fig.add_artist(ConnectionPatch(
                xyA=(0.0, ed), coordsA=ax_spec.transData,
                xyB=(1.0, 0.5), coordsB=ax_d.transAxes,
                color=col, lw=0.7, alpha=0.9, zorder=4, arrowstyle='-'))
            fig.add_artist(ConnectionPatch(
                xyA=(1.0, ep), coordsA=ax_spec.transData,
                xyB=(0.0, 0.5), coordsB=ax_p.transAxes,
                color=col, lw=0.7, alpha=0.9, zorder=4, arrowstyle='-'))
    else:
        # DFTB = TOP half (ψ + ticks toward top); pySCF = BOTTOM half — no cross-midline links
        gap_in = 0.08
        panel_aspect = Lx / max(Ly, 1e-12)
        col_w_in = 0.68
        panel_h_in = col_w_in * panel_aspect
        spec_h_in = 0.34
        lab_in = 0.22          # label strip above DFTB / below pySCF
        title_in = 0.28        # room for fig.suptitle — no overlap with labels
        fig_w = col_w_in * n + 0.30
        fig_h = 2 * panel_h_in + spec_h_in + 2 * gap_in + 2 * lab_in + title_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        left_m, right_m = 0.02, 0.005
        usable_w = 1.0 - left_m - right_m
        col_w = usable_w / n
        img_h = panel_h_in / fig_h
        spec_h = spec_h_in / fig_h
        gap = gap_in / fig_h
        lab_h = lab_in / fig_h
        title_h = title_in / fig_h
        # top → bottom: title | lab_D | ψ_D | gap | spectrum | gap | ψ_P | lab_P
        y_lab_d = 1.0 - title_h - lab_h
        y_d = y_lab_d - img_h
        y_spec = y_d - gap - spec_h
        y_p = y_spec - gap - img_h
        y_lab_p = y_p - lab_h
        ax_spec = fig.add_axes([left_m, y_spec, usable_w, spec_h])
        ax_spec.set_ylim(0.0, 1.0)
        ax_spec.set_xlim(e_lo, e_hi)
        ax_spec.set_yticks([])
        _paint_E_scale(ax_spec, 'x')
        # TOP half of spectrum = DFTB; BOTTOM half = pySCF
        ax_spec.axhline(0.5, color='0.85', ls=':', lw=0.3, zorder=0)
        ax_spec.text(e_lo, 0.78, 'DFTB', fontsize=5.5, ha='left', va='center', color='0.35')
        ax_spec.text(e_lo, 0.22, 'pySCF', fontsize=5.5, ha='left', va='center', color='0.35')
        for e in E_d:
            ax_spec.vlines(float(e), 0.55, 1.00, colors='0.9', lw=0.2, zorder=1)
        for e in E_p:
            ax_spec.vlines(float(e), 0.00, 0.45, colors='0.9', lw=0.2, zorder=1)

        for ic, k in enumerate(rels_lo):
            imo_d, imo_p = homo_d + k, homo_p + k
            ed, ep = float(E_d[imo_d]), float(E_p[imo_p])
            col = 'C0' if k <= 0 else 'C3'
            # DFTB ticks on TOP half; pySCF on BOTTOM half
            ax_spec.vlines(ed, 0.55, 1.00, colors=col, lw=1.35, zorder=3)
            ax_spec.vlines(ep, 0.00, 0.45, colors=col, lw=1.35, zorder=3)
            x0 = left_m + ic * col_w + 0.01 * col_w
            pw = 0.98 * col_w
            ax_d = fig.add_axes([x0, y_d, pw, img_h])
            ax_p = fig.add_axes([x0, y_p, pw, img_h])
            _imshow_psi(ax_d, maps_dftb[imo_d], rotate=True)
            _imshow_psi(ax_p, maps_pyscf[imo_p], rotate=True)
            fig.text(x0 + 0.5 * pw, y_lab_d + 0.15 * lab_h,
                     f'{_lab(k)} D#{imo_d}\n{ed:.2f}', fontsize=4.6, ha='center', va='bottom', color=col)
            fig.text(x0 + 0.5 * pw, y_lab_p + 0.55 * lab_h,
                     f'{_lab(k)} P#{imo_p}\n{ep:.2f}', fontsize=4.6, ha='center', va='top', color=col)
            # connectors stay on their half: top edge of spectrum ↔ DFTB; bottom ↔ pySCF
            fig.add_artist(ConnectionPatch(
                xyA=(ed, 1.0), coordsA=ax_spec.transData,
                xyB=(0.5, 0.0), coordsB=ax_d.transAxes,
                color=col, lw=0.7, alpha=0.9, zorder=4, arrowstyle='-'))
            fig.add_artist(ConnectionPatch(
                xyA=(ep, 0.0), coordsA=ax_spec.transData,
                xyB=(0.5, 1.0), coordsB=ax_p.transAxes,
                color=col, lw=0.7, alpha=0.9, zorder=4, arrowstyle='-'))

    if title:
        fig.suptitle(title, fontsize=7.5, y=0.995)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path



def plot_frontier_orbital_gallery(maps_dftb, maps_pyscf, mo_labels, scan_xs, scan_ys, z_A,
                                  out_path, *, atom_pos=None, title=None, field='psi'):
    """Rows = MO labels, cols = DFTB | pySCF. maps_*[label] = (nx,ny).

    For energy-up spectrum↔orbital links use ``plot_spectrum_with_orbitals``.
    """
    nrows = len(mo_labels)
    fig, axes = plt.subplots(nrows, 2, figsize=(7.2, 1.55 * nrows + 0.8), squeeze=False)
    extent = [scan_xs[0], scan_xs[-1], scan_ys[0], scan_ys[-1]]
    cmap = 'RdBu_r' if field == 'psi' else 'viridis'
    for ir, lab in enumerate(mo_labels):
        for ic, (tag, mp) in enumerate((('DFTB', maps_dftb), ('pySCF', maps_pyscf))):
            ax = axes[ir, ic]
            arr = np.asarray(mp[lab])
            vmax = float(np.percentile(np.abs(arr), 99)) or 1e-30
            vmin = -vmax if field == 'psi' else 0.0
            im = ax.imshow(arr.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=extent, aspect='equal')
            if ir == 0:
                ax.set_title(tag, fontsize=10)
            if ic == 0:
                ax.set_ylabel(lab, fontsize=7)
            if atom_pos is not None:
                ax.scatter(atom_pos[:, 0], atom_pos[:, 1], c='k', s=3, alpha=0.35, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if title is None:
        title = f'Frontier orbitals ψ  z={z_A:.2f}Å (per-panel scale)'
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def compute_stm_basis_variants(atomPos, atomTypes, basis_hsd_path, work_dir,
                               scan_xs, scan_ys, heights, *,
                               projection_variants=None, zeta_override=None,
                               cutoff_extend=6.0, field='psi2', verbosity=0):
    """DFTB SCF once (stock WFC) → HOMO/LUMO STM maps for stock + prolonged projection.

    SCF DM/MOs always from stock mio/3ob. Projection radial is swapped:
      - 'stock'     : native multi-ζ WFC
      - 'prolonged' : make_slater_tail_species_list (optional SA zeta_override)

    STM uses the STO table (use_exp_basis=False) — not generic exp(β,r0).

    Returns dict with eigvals, mo indices, and maps[variant][label] = (nx,ny,nz) float32.
    """
    from spammm.quantum.DFTB.DFTBplusParser import make_slater_tail_species_list

    if projection_variants is None:
        projection_variants = ('stock', 'prolonged')

    d = get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd_path, work_dir,
        step=0.5, margin=0.5, z_extra=0.5, verbosity=verbosity, project_density=False)
    # Density unused — STM needs SCF MOs + projector only.
    projector = d['projector']
    atoms_dict = d['atoms_dict']
    basis_ang = d['basis_ang']
    eigvecs, eigvals = d['eigvecs'], d['eigvals']
    norb_per_atom, orb_offsets = d['norb_per_atom'], d['orb_offsets']
    enames = d.get('enames')
    homo, lumo = dftb_frontier_mo_indices(eigvals, enames=enames, atomTypes=atomTypes)
    mo_map = {'HOMO': homo, 'LUMO': lumo}

    prolonged = None
    if 'prolonged' in projection_variants:
        prolonged = make_slater_tail_species_list(
            basis_ang, zeta_override=zeta_override, cutoff_extend=cutoff_extend)

    maps = {}
    for variant in projection_variants:
        if variant == 'stock':
            _set_projector_species_basis(projector, atoms_dict, basis_ang, rc_max=cutoff_extend)
        elif variant == 'prolonged':
            if prolonged is None:
                raise ValueError("projection_variants contains 'prolonged' but list not built")
            _set_projector_species_basis(projector, atoms_dict, prolonged, rc_max=cutoff_extend)
        else:
            raise ValueError(f"unknown projection variant {variant!r} (use stock|prolonged)")

        maps[variant] = {}
        for lab, imo in mo_map.items():
            stm = compute_stm(
                projector, eigvecs, eigvals, scan_xs, scan_ys, heights,
                norb_per_atom, orb_offsets, atoms_dict,
                mo_indices=[imo], field=field, use_exp_basis=False)
            maps[variant][lab] = stm
            if verbosity:
                print(f"  [STM-basis] {variant}/{lab} MO#{imo} range=[{stm.min():.3e},{stm.max():.3e}]")

    return {
        'maps': maps,
        'homo': homo, 'lumo': lumo,
        'eigvals': eigvals, 'eigvecs': eigvecs,
        'E_homo': float(eigvals[homo]), 'E_lumo': float(eigvals[lumo]),
        'basis_hsd_path': basis_hsd_path,
        'zeta_override': zeta_override,
        'field': field,
        'dftb': d,
    }


def plot_stm_basis_compare_panel(maps_by_col, scan_xs, scan_ys, height, labels,
                                 col_titles, out_path, *, field='psi2', atom_pos=None,
                                 title=None, share_row_scale=False):
    """L2 gallery: rows = HOMO/LUMO (labels), columns = basis channels.

    maps_by_col: list of dict {label: (nx,ny) or (nx,ny,nz)}; 3D → slice [:,:,0].
    share_row_scale: if True, one vmax per row (cross-column intensity). Default False
        (per-column vmax) — absolute |ψ| scales differ DFTB↔pySCF; morphology needs local scale.
    """
    nrows, ncols = len(labels), len(maps_by_col)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.6 * nrows), squeeze=False)
    extent = [scan_xs[0], scan_xs[-1], scan_ys[0], scan_ys[-1]]
    cmap = 'RdBu_r' if field == 'psi' else 'viridis'
    for ir, lab in enumerate(labels):
        if share_row_scale:
            row_abs = []
            for j in range(ncols):
                arr = maps_by_col[j][lab]
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                row_abs.append(float(np.percentile(np.abs(arr), 99)))
            shared_max = max(row_abs) or 1e-30
        for ic, col in enumerate(maps_by_col):
            ax = axes[ir, ic]
            arr = col[lab]
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            vmax = shared_max if share_row_scale else (float(np.percentile(np.abs(arr), 99)) or 1e-30)
            vmin = -vmax if field == 'psi' else 0.0
            im = ax.imshow(arr.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=extent, aspect='equal')
            if ir == 0:
                ax.set_title(col_titles[ic], fontsize=9)
            if ic == 0:
                ax.set_ylabel(f'{lab}\nz={height:.1f}Å', fontsize=8)
            if atom_pos is not None:
                ax.scatter(atom_pos[:, 0], atom_pos[:, 1], c='k', s=4, alpha=0.4, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94] if title else None)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_stm(stm_grid, scan_xs, scan_ys, heights, output_dir, prefix='stm'):
    """Plot STM signal for each height.

    Args:
        stm_grid: (nx_s, ny_s, nz_s) STM signal array
        scan_xs: (nx_s,) scan x coordinates
        scan_ys: (ny_s,) scan y coordinates
        heights: (nz_s,) probe heights
        output_dir: directory for output plots
        prefix: filename prefix
    """
    nz = len(heights)
    ncols = min(7, nz)
    nrows = int(np.ceil(nz / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.8*nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    fig.suptitle(f"STM Signal (LUMO^2)", fontsize=10)

    ext = [scan_xs[0], scan_xs[-1], scan_ys[0], scan_ys[-1]]
    kw = dict(origin='lower', cmap='viridis', aspect='equal', extent=ext)

    for k in range(nz):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        im = ax.imshow(stm_grid[:, :, k].T, **kw)
        ax.set_title(f"h={heights[k]:.1f}Å", fontsize=8)
        ax.tick_params(labelsize=4)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused subplots
    for k in range(nz, nrows*ncols):
        r, c = divmod(k, ncols)
        axes[r, c].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(output_dir, f'{prefix}.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved STM plot: {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# I/O Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def save_grid_spec(grid_spec, step_dir):
    """Save grid specification to file."""
    np.save(os.path.join(step_dir, 'origin.npy'), grid_spec['origin'])
    np.save(os.path.join(step_dir, 'ngrid.npy'), grid_spec['ngrid'])
    with open(os.path.join(step_dir, 'step.txt'), 'w') as f:
        f.write(str(grid_spec['dA'][0]))  # step is same for all axes


def load_grid_spec(step_dir):
    """Load grid specification from file."""
    origin = np.load(os.path.join(step_dir, 'origin.npy'))
    ngrid = np.load(os.path.join(step_dir, 'ngrid.npy'))
    with open(os.path.join(step_dir, 'step.txt'), 'r') as f:
        step = float(f.read().strip())
    grid_spec = {
        'origin': origin,
        'dA': [step, 0., 0.], 'dB': [0., step, 0.], 'dC': [0., 0., step],
        'ngrid': ngrid.astype(int),
    }
    return grid_spec, origin, step


# ═══════════════════════════════════════════════════════════════════════════════
def run_afm_pipeline(
    rho_grid, rho_na_grid, rho_diff, V_ES,
    rho_tip_total, rho_tip_delta,
    atomPos, atomTypes,
    origin, step, ngrid,
    scan_xs, scan_ys, heights,
    output_dir,
    pauli_params={'A': None, 'beta': None},
    pauli_fit_params=None,
    fit_pauli=False,
    fit_pauli_params=None,  # Dict with zscan_dir, target_indices, z_min, z_max, basis
    vdw_params={'C6_CO': 30.0},
    relax_params={'K_LAT': 0.5},
    plot_steps=True,
    stm_params=None,  # Dict with STM parameters for Step 7
    use_gpu_gradient=True,  # Use GPU for total gradient computation
    use_gpu_relax=True,     # Use GPU relaxStrokes kernel (now with damped velocity matching CPU)
    ppm_mode=False,         # True = PPM radial bond (CO-tip, L=3A); False = linear harmonic
    afmulator=None,  # AFMulator instance for GPU gradient/relax (created if None)
    projector=None,  # GridProjector for STM (required if stm_params is set)
    norb_per_atom=None,  # Required for STM
    orb_offsets=None,  # Required for STM
    atoms_dict=None,  # Required for STM
    eigvecs=None,  # Required for STM
    eigvals=None  # Required for STM
):
    """
    High-level AFM simulation pipeline using pre-computed densities.

    This function runs steps 2-6 of the AFM simulation, assuming step 1
    (density projection) has already been done separately.
    Optionally computes Step 7: STM simulation.

    Args:
        rho_grid: (nx, ny, nz) sample SCF density
        rho_na_grid: (nx, ny, nz) neutral atom density
        rho_diff: (nx, ny, nz) delta density
        V_ES: (nx, ny, nz) electrostatic potential (optional, can compute)
        rho_tip_total: (nx, ny, nz) CO tip total density
        rho_tip_delta: (nx, ny, nz) CO tip delta density
        atomPos: (natoms, 3) atom positions
        atomTypes: (natoms,) atomic numbers
        origin: (3,) grid origin
        step: grid spacing
        ngrid: (3,) grid dimensions
        scan_xs: (nx_s,) scan x coordinates
        scan_ys: (ny_s,) scan y coordinates
        heights: (nz_s,) probe heights
        output_dir: directory for outputs
        pauli_params: dict with 'A', 'beta'
        pauli_fit_params: dict with fitted 'A', 'beta' (from external fit)
        fit_pauli: if True, fit Pauli parameters internally after computing raw overlap
        fit_pauli_params: dict with 'zscan_dir', 'target_indices', 'z_min', 'z_max', 'basis'
        vdw_params: dict with 'C6_CO'
        relax_params: dict with 'K_LAT'
        plot_steps: whether to generate plots
        stm_params: dict with STM parameters:
            - 'compute': bool (default: False)
            - 'lumo_offsets': list (default: [1,2,3])
            - 'use_exp_basis': bool (default: True)
            - 'exp_beta': float (default: 1.0)
            - 'exp_r0': float (default: 3.0)
            - 'bond_resolved': bool (default: False)
        projector: GridProjector instance (required for STM)
        norb_per_atom: (natoms,) orbital counts (required for STM)
        orb_offsets: (natoms+1,) orbital offsets (required for STM)
        atoms_dict: atom data dict (required for STM)
        eigvecs: (nstates, norb_total) eigenvectors (required for STM)
        eigvals: (nstates,) eigenvalues (required for STM)

    Returns:
        dict with 'df', 'intermediates', 'grid_spec'
    """
    os.makedirs(output_dir, exist_ok=True)
    from spammm.globals import debug_save_enabled
    
    # Step 2: Electrostatics (if V_ES not provided)
    if V_ES is None:
        print("\nStep 2: Computing electrostatics...")
        V_ES = afm.fft_poisson(rho_diff, step)
        if plot_steps:
            plot_step2_outputs(V_ES, output_dir, origin, step)
        if debug_save_enabled(2):
            np.save(os.path.join(output_dir, 'V_ES.npy'), V_ES)
    else:
        print("\nStep 2: Using provided V_ES")
        if plot_steps:
            plot_step2_outputs(V_ES, output_dir, origin, step)
    
    # Step 3a: Compute raw Pauli overlap (A=1, beta=1 — pure density convolution)
    print("\nStep 3a: Computing raw Pauli overlap (A=1, beta=1)...")
    overlap_raw = afm.compute_pauli_overlap(rho_grid, rho_tip_total, step, tip_rolled=True)
    if debug_save_enabled(2):
        np.save(os.path.join(output_dir, 'overlap_raw.npy'), overlap_raw)
    print(f"  overlap_raw: shape={overlap_raw.shape}  range=[{overlap_raw.min():.4e}, {overlap_raw.max():.4e}]")
    
    # Step 3b: Fit Pauli parameters or use provided / default values
    A_pauli = pauli_params.get('A') if pauli_params else None
    beta_pauli = pauli_params.get('beta') if pauli_params else None
    
    if fit_pauli and fit_pauli_params is not None:
        # Fit Pauli parameters internally using DFTB reference
        print("\nStep 3b: Fitting Pauli parameters using DFTB z-scan reference...")
        zscan_dir = fit_pauli_params['zscan_dir']
        target_indices = fit_pauli_params['target_indices']
        z_min = fit_pauli_params.get('z_min', 2.0)
        z_max = fit_pauli_params.get('z_max', 3.0)
        
        from spammm.quantum.DFTB import TestUtils as tu
        
        # Load DFTB reference for each target atom
        all_A, all_beta = [], []
        for idx in target_indices:
            atom_dir = os.path.join(zscan_dir, f'atom_{idx}')
            z_ref = np.load(os.path.join(atom_dir, 'zscan_z.npy'))
            e_ref_abs = np.load(os.path.join(atom_dir, 'zscan_energy_eV.npy'))
            e_ref = e_ref_abs - e_ref_abs[-1]  # Reference to far distance
            
            target_pos = atomPos[idx]
            overlap_profile = tu.extract_z_profile(overlap_raw, target_pos, origin, step, z_distances=z_ref)
            
            # Fit power law (returns tuple: A, beta, r2, e_pred)
            A_fit, beta_fit, r2_fit, _ = _fit_pauli_powerlaw(z_ref, overlap_profile, e_ref, z_min, z_max)
            all_A.append(A_fit)
            all_beta.append(beta_fit)
            
            print(f"  Atom {idx}: A={A_fit:.2f}, beta={beta_fit:.4f}, R2={r2_fit:.4f}")
        
        A_pauli = np.mean(all_A)
        beta_pauli = np.mean(all_beta)
        print(f"\nStep 3b: Fitted Pauli params: A={A_pauli:.4f}, beta={beta_pauli:.4f}")
    elif pauli_fit_params is not None:
        # Fit was done externally; use those results
        A_pauli   = pauli_fit_params['A']
        beta_pauli = pauli_fit_params['beta']
        print(f"\nStep 3b: Using externally fitted Pauli params: A={A_pauli:.4f}, beta={beta_pauli:.4f}")
    elif A_pauli is None or beta_pauli is None:
        raise ValueError(
            "Pauli parameters A and beta must be provided via pauli_params or pauli_fit_params, "
            "or set fit_pauli=True with fit_pauli_params."
        )
    else:
        print(f"\nStep 3b: Using provided Pauli params: A={A_pauli:.4f}, beta={beta_pauli:.4f}")
    
    # Step 3c: Scale overlap into energy field (energy only, no gradients)
    print(f"\nStep 3c: Scaling E_pauli = {A_pauli:.4f} * overlap^{beta_pauli:.4f}")
    E_pauli_field = afm.scale_pauli_field(overlap_raw, step, A_pauli, beta_pauli, return_grads=False)

    # Consistency diagnostics
    print(f"  overlap_raw at max: {overlap_raw.max():.4e}")
    print(f"  E_pauli_field: range=[{E_pauli_field.min():.4e}, {E_pauli_field.max():.4e}]")
    print(f"  Check: A*overlap_max^beta = {A_pauli:.4f}*{overlap_raw.max():.4e}^{beta_pauli:.4f} = {A_pauli * float(overlap_raw.max())**beta_pauli:.4e}")

    if plot_steps:
        plot_step3_outputs(E_pauli_field, None, output_dir, origin, step, A_pauli, beta_pauli)
    if debug_save_enabled(2):
        np.save(os.path.join(output_dir, 'E_Pauli_field.npy'), E_pauli_field)

    # Step 4: Electrostatic convolution (energy only, no gradients)
    print("\nStep 4: Computing electrostatic convolution...")
    E_ES_field = afm.compute_es_conv_field(V_ES, rho_tip_delta, step, tip_rolled=True, return_grads=False)
    if plot_steps:
        plot_step4_outputs(E_ES_field, None, output_dir, origin, step)
    if debug_save_enabled(2):
        np.save(os.path.join(output_dir, 'E_ES_field.npy'), E_ES_field)

    # Step 5: Dispersion (energy only, no gradients)
    print("\nStep 5: Computing dispersion...")
    E_vdw = afm.compute_dispersion_grid(
        atomPos, atomTypes, origin, step, ngrid,
        C6_CO=vdw_params['C6_CO'], return_grads=False
    )
    if plot_steps:
        plot_step5_outputs(E_vdw, None, output_dir, origin, step)
    if debug_save_enabled(2):
        np.save(os.path.join(output_dir, 'E_vdw_field.npy'), E_vdw)

    # Step 5b: Compute total energy and gradient
    print("\nStep 5b: Computing total energy field and gradient...")
    E_total = E_pauli_field + E_ES_field + E_vdw
    print(f"  E_total range: [{E_total.min():.4e}, {E_total.max():.4e}]")

    # Compute gradient of total energy (CPU or GPU)
    if use_gpu_gradient:
        print("  Using GPU for gradient computation...")
        if afmulator is None:
            # Create AFMulator instance if not provided
            afmulator = afm.AFMulator(use_morse=False, nloc=32)
        grads_cl = afmulator.compute_gradient_cl(E_total, step, bAlloc=True)
        # grads_cl is (Fx, Fy, Fz, E) where F = -grad(E)
        # This is already the force field F_total we need
        F_total = grads_cl  # (Fx, Fy, Fz, E) - full force field
    else:
        print("  Using CPU (numpy) for gradient computation...")
        # Compute gradient, then convert to force F = -grad(E)
        grads = np.stack([np.gradient(E_total, step, axis=i) for i in range(3)], axis=-1)
        # Build full array (Fx, Fy, Fz, E) where F = -grad
        F_total = np.zeros(E_total.shape + (4,), dtype=np.float32)
        F_total[..., :3] = -grads  # F = -grad(E)
        F_total[..., 3] = E_total   # E

    # Save intermediates
    if debug_save_enabled(2):
        np.save(os.path.join(output_dir, 'E_total_field.npy'), E_total)
        np.save(os.path.join(output_dir, 'F_total.npy'), F_total)

    # Step 6: Compose and relax using total force field
    print("\nStep 6: Composing force fields and running probe relaxation...")
    df, tip_disp, FEs_relax = compose_and_relax_total(
        F_total,
        scan_xs, scan_ys, heights,
        origin, step, atomPos, K_LAT=relax_params['K_LAT'],
        use_gpu_relax=use_gpu_relax, ppm_mode=ppm_mode, afmulator=afmulator
    )
    if debug_save_enabled(2):
        np.save(os.path.join(output_dir, 'df.npy'), df)
        np.save(os.path.join(output_dir, 'tip_disp_dx.npy'), tip_disp['dx'])
        np.save(os.path.join(output_dir, 'tip_disp_dy.npy'), tip_disp['dy'])
    if plot_steps:
        plot_step6_outputs(df, scan_xs, scan_ys, heights, output_dir)

    # Step 7: STM (optional)
    stm_grid = None
    stm_meta = None
    if stm_params and stm_params.get('compute', False):
        print("\nStep 7: Computing STM...")
        if projector is None or eigvecs is None or eigvals is None:
            raise ValueError("STM computation requires projector, eigvecs, and eigvals")

        lumo_offsets = stm_params.get('lumo_offsets', [1, 2, 3])
        mo_indices   = stm_params.get('mo_indices', None)
        use_exp_basis = stm_params.get('use_exp_basis', True)
        exp_beta = stm_params.get('exp_beta', 1.0)
        exp_r0 = stm_params.get('exp_r0', 3.0)
        bond_resolved = stm_params.get('bond_resolved', False)
        stm_field = stm_params.get('field', 'ldos')

        if atoms_dict is not None and 'type' in atoms_dict:
            homo, lumo = dftb_frontier_mo_indices(eigvals, atomTypes=atoms_dict['type'])
            n_occ = int(homo) + 1
        else:
            raise ValueError(
                "STM pipeline: need atoms_dict['type'] for valence HOMO; "
                "eigvals<0 is wrong for DFTB. See dftb_frontier_mo_indices.")
        if mo_indices is not None:
            mo_list = [int(i) for i in mo_indices]
            mode = 'mo_indices'
        else:
            mo_list = [int(homo) + int(off) for off in (lumo_offsets or [])] if homo is not None else []
            mode = 'lumo_offsets'
        E_homo = float(eigvals[homo]) if homo is not None else None
        E_lumo = float(eigvals[lumo]) if lumo is not None else None
        stm_meta = {
            'nmo': int(eigvecs.shape[0]),
            'norb': int(eigvecs.shape[1]),
            'nocc': n_occ,
            'homo': homo,
            'lumo': lumo,
            'E_homo': E_homo,
            'E_lumo': E_lumo,
            'mo_list': mo_list,
            'mode': mode,
            'bond_resolved': bool(bond_resolved),
            'field': str(stm_field),
            'height_min': float(heights[0]) if len(heights) > 0 else None,
            'height_max': float(heights[-1]) if len(heights) > 0 else None,
            'n_heights': int(len(heights)),
        }

        if bond_resolved:
            print(f"  Computing bond-resolved STM (displaced positions)...")
            stm_grid = compute_bond_resolved_stm(
                projector, eigvecs, eigvals, scan_xs, scan_ys, heights,
                tip_disp, norb_per_atom, orb_offsets, atoms_dict,
                lumo_offsets=lumo_offsets, mo_indices=mo_indices, field=stm_field, use_exp_basis=use_exp_basis,
                exp_beta=exp_beta, exp_r0=exp_r0
            )
        else:
            print(f"  Computing standard STM...")
            stm_grid = compute_stm(
                projector, eigvecs, eigvals, scan_xs, scan_ys, heights,
                norb_per_atom, orb_offsets, atoms_dict,
                lumo_offsets=lumo_offsets, mo_indices=mo_indices, field=stm_field, use_exp_basis=use_exp_basis,
                exp_beta=exp_beta, exp_r0=exp_r0
            )

        if debug_save_enabled(2):
            np.save(os.path.join(output_dir, 'stm_grid.npy'), stm_grid)
        if plot_steps:
            plot_stm(stm_grid, scan_xs, scan_ys, heights, output_dir, prefix='stm')

    # Return results
    grid_spec_out = {
        'origin': origin,
        'dA': [step, 0., 0.], 'dB': [0., step, 0.], 'dC': [0., 0., step],
        'ngrid': ngrid.astype(int),
    }

    result = {
        'df': df,
        'scan_xs': scan_xs,
        'scan_ys': scan_ys,
        'heights': heights,
        'intermediates': {
            'V_ES': V_ES,
            'E_pauli_field': E_pauli_field,
            'grads_pauli': None,  # Not computed in optimized mode (use F_total)
            'E_ES_field': E_ES_field,
            'grads_ES': None,  # Not computed in optimized mode (use F_total)
            'E_vdw': E_vdw,
            'grads_vdw': None,  # Not computed in optimized mode (use F_total)
            'F_total': F_total,  # Full force field (Fx,Fy,Fz,E) from GPU
            'tip_disp': tip_disp,
        },
        'grid_spec': grid_spec_out,
    }

    if stm_grid is not None:
        result['intermediates']['stm_grid'] = stm_grid
        if stm_meta is not None:
            result['intermediates']['stm_meta'] = stm_meta

    return result


def _compute_co_tip_grid(step=0.1, margin=4.0):
    """Return grid_spec for CO tip computation with O at grid center."""
    co_span = np.array([0.0, 0.0, 1.13])  # C is at z=1.13 relative to O
    ngrid = np.ceil((2 * margin + co_span) / step).astype(np.int32)
    # Round up to nearest multiple of 8 for GPU
    ngrid = ((ngrid + 7) // 8) * 8
    origin = np.array([-margin, -margin, -margin], dtype=np.float32)
    grid_spec = {
        'origin': origin,
        'dA': np.array([step, 0.0, 0.0], dtype=np.float32),
        'dB': np.array([0.0, step, 0.0], dtype=np.float32),
        'dC': np.array([0.0, 0.0, step], dtype=np.float32),
        'ngrid': ngrid,
    }
    return grid_spec, ngrid, origin


def _co_tip_cache_dir():
    """Return global CO tip cache directory."""
    return os.path.join(os.path.expanduser('~'), '.cache', 'firecore', 'co_tips')


def _co_tip_cache_key(step, margin, fdata_dir, fdata_basis, backend='dftb'):
    """Compute a deterministic cache key for CO tip parameters.

    Version suffix invalidates broken caches from manual MO outer-product DM (pre-v2).
    """
    import hashlib
    # Normalize paths for portability
    fdata_dir_abs = os.path.normpath(os.path.abspath(fdata_dir))
    fdata_basis_abs = os.path.normpath(os.path.abspath(fdata_basis))
    # v2→v3: O snapped to voxel n//2 (not half-voxel geometric center)
    key_str = f"v3:step={step:.6f}:margin={margin:.6f}:backend={backend}:fdata={fdata_dir_abs}:basis={fdata_basis_abs}"
    return hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:16]


def _get_cached_co_tip(step, margin, fdata_dir, fdata_basis, backend='dftb'):
    """Load cached CO tip if available; return (co_rho_total, co_rho_delta) or None."""
    cache_dir = _co_tip_cache_dir()
    key = _co_tip_cache_key(step, margin, fdata_dir, fdata_basis, backend)
    cache_subdir = os.path.join(cache_dir, key)
    total_path = os.path.join(cache_subdir, 'co_rho_total.npy')
    delta_path = os.path.join(cache_subdir, 'co_rho_delta.npy')
    if os.path.isfile(total_path) and os.path.isfile(delta_path):
        return np.load(total_path), np.load(delta_path)
    return None


def _save_cached_co_tip(co_rho_total, co_rho_delta, step, margin, fdata_dir, fdata_basis, backend='dftb'):
    """Save CO tip densities to global cache."""
    cache_dir = _co_tip_cache_dir()
    key = _co_tip_cache_key(step, margin, fdata_dir, fdata_basis, backend)
    cache_subdir = os.path.join(cache_dir, key)
    os.makedirs(cache_subdir, exist_ok=True)
    np.save(os.path.join(cache_subdir, 'co_rho_total.npy'), co_rho_total)
    np.save(os.path.join(cache_subdir, 'co_rho_delta.npy'), co_rho_delta)


def _call_compute_co_tip_script(out_dir, grid_spec, step, nscf, fdata_dir, fdata_basis, backend='dftb'):
    """Call compute_co_tip.py as subprocess.
    
    Args:
        backend: 'dftb' or 'firecore' (default: 'dftb')
    """
    import json, subprocess, sys
    _THIS_FILE = os.path.abspath(__file__)
    # compute_co_tip.py is in the same directory as AFM_utils.py (spammm/SPM/)
    script = os.path.join(os.path.dirname(_THIS_FILE), 'compute_co_tip.py')

    # Convert numpy arrays to lists for JSON serialization
    grid_spec_json = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in grid_spec.items()}
    grid_spec_str = json.dumps(grid_spec_json)

    cmd = [sys.executable, script, out_dir, grid_spec_str, str(step), str(nscf), fdata_dir, fdata_basis, backend]
    print(f"  Running CO tip computation (backend={backend}): {' '.join(cmd[:5])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"compute_co_tip.py failed:\n{result.stderr}\n{result.stdout}")
    print(result.stdout)
    return out_dir


def _pad_and_roll_co_tip(co_rho, target_shape):
    """Pad CO density with zeros to target grid and roll so O peak is at (0,0,0).

    FFT convolution convention: tip origin (O atom) must sit at array index (0,0,0).
    Finds the actual density peak after centering (robust to odd/even ngrid), then rolls.
    """
    tip_tot, _ = _pad_and_roll_co_tip_pair(co_rho, None, target_shape)
    return tip_tot


def _pad_and_roll_co_tip_pair(tip_tot, tip_del, target_shape):
    """Pad+roll tip_tot (and tip_del) so tip_tot peak → (0,0,0); **same** shift for both.

    Never roll tip_del on its own |Δρ| peak — that misaligns ES vs Pauli (diagnosed 2026-07-21).
    tip_del=None → only tip_tot returned as (tot, None).
    """
    nx_t, ny_t, nz_t = target_shape

    def _pad(src):
        if src is None:
            return None
        src = np.asarray(src, dtype=np.float32)
        nx_c, ny_c, nz_c = src.shape
        ox = (nx_t - nx_c) // 2
        oy = (ny_t - ny_c) // 2
        oz = (nz_t - nz_c) // 2
        padded = np.zeros(target_shape, dtype=np.float32)
        padded[ox:ox + nx_c, oy:oy + ny_c, oz:oz + nz_c] = src
        return padded

    tot = _pad(tip_tot)
    dele = _pad(tip_del)
    peak = np.unravel_index(int(np.argmax(np.abs(tot))), tot.shape)
    for ax, p in enumerate(peak):
        if p != 0:
            tot = np.roll(tot, -int(p), axis=ax)
            if dele is not None:
                dele = np.roll(dele, -int(p), axis=ax)
    return tot, dele


def get_tip_densities(tip_mode, target_shape, step, margin=4.0, sigma=0.7,
                      basis='mio-1-1', output_dir=None, co_tip_dir=None,
                      fdata_dir=None, fdata_basis=None, backend='dftb',
                      force_recompute=False, pad_mode='cpu'):
    """Prepare tip densities for FDBM Pauli/ES convolution.

    tip_mode:
      'gaussian' — isotropic Gaussian at (0,0,0) (fast; ES uses same as Pauli)
      'co'       — real CO tip density (O at (0,0,0) after roll, C along +z)

    pad_mode:
      'cpu'  — pad+roll on host to target_shape (legacy)
      'none' — return raw CO arrays (possibly smaller); caller pads on GPU (fast S3)

    Returns (rho_tip_total, rho_tip_delta) as float32.
    """
    tip_mode = tip_mode.lower()
    nx, ny, nz = target_shape

    if tip_mode == 'gaussian':
        rho_total = afm.build_gaussian_tip((nx, ny, nz), step, sigma).astype(np.float32)
        rho_delta = rho_total  # no NA subtraction for Gaussian model
        print(f"  Tip mode=gaussian  sigma={sigma}  shape={rho_total.shape}  peak@{(0,0,0)}  q≈{rho_total.sum()*step**3:.3f}")
        return rho_total, rho_delta

    if tip_mode != 'co':
        raise ValueError(f"Unknown tip_mode={tip_mode!r}; use 'gaussian' or 'co'")

    if fdata_dir is None:
        fdata_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests', 'pyFireball', 'Fdata'))
    if fdata_basis is None:
        fdata_basis = os.path.join(fdata_dir, 'basis')
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser('~'), '.cache', 'firecore', 'co_tip_work')
    os.makedirs(output_dir, exist_ok=True)

    co_rho_total_raw = co_rho_delta_raw = None
    if co_tip_dir is not None and os.path.isdir(co_tip_dir) and not force_recompute:
        print(f"  Loading precomputed CO tip from {co_tip_dir}...")
        co_rho_total_raw = np.load(os.path.join(co_tip_dir, 'co_rho_total.npy'))
        co_rho_delta_raw = np.load(os.path.join(co_tip_dir, 'co_rho_delta.npy'))
    if co_rho_total_raw is None and not force_recompute:
        cached = _get_cached_co_tip(step, margin, fdata_dir, fdata_basis, backend=backend)
        if cached is not None:
            from spammm.globals import debug_print
            debug_print(1, f"  Loading cached CO tip (step={step}, margin={margin}, backend={backend})...")
            co_rho_total_raw, co_rho_delta_raw = cached
    if co_rho_total_raw is None:
        print(f"  Computing CO tip on-the-fly (step={step}, backend={backend})...")
        co_tip_work = os.path.join(output_dir, 'co_tip_work')
        os.makedirs(co_tip_work, exist_ok=True)
        co_grid_spec, co_ngrid, co_origin = _compute_co_tip_grid(step=step, margin=margin)
        print(f"  CO grid: ngrid={co_ngrid}, origin={co_origin}")
        _call_compute_co_tip_script(co_tip_work, co_grid_spec, step, 100, fdata_dir, fdata_basis, backend=backend)
        co_rho_total_raw = np.load(os.path.join(co_tip_work, 'co_rho_total.npy'))
        co_rho_delta_raw = np.load(os.path.join(co_tip_work, 'co_rho_delta.npy'))
        _save_cached_co_tip(co_rho_total_raw, co_rho_delta_raw, step, margin, fdata_dir, fdata_basis, backend=backend)

    if pad_mode == 'none':
        from spammm.globals import debug_print
        debug_print(1, f"  Tip mode=co  raw={co_rho_total_raw.shape} (GPU pad/roll deferred)")
        return co_rho_total_raw.astype(np.float32), co_rho_delta_raw.astype(np.float32)

    rho_total = _pad_and_roll_co_tip(co_rho_total_raw, target_shape)
    rho_delta = _pad_and_roll_co_tip(co_rho_delta_raw, target_shape)
    peak = np.unravel_index(int(np.argmax(np.abs(rho_total))), rho_total.shape)
    q = float(rho_total.sum() * step**3)
    from spammm.globals import debug_print
    debug_print(1, f"  Tip mode=co  raw={co_rho_total_raw.shape} → rolled={rho_total.shape}  peak={peak}  q={q:.3f}")
    if peak != (0, 0, 0):
        print(f"  WARNING: CO tip peak not at (0,0,0) after roll: {peak}")
    return rho_total.astype(np.float32), rho_delta.astype(np.float32)


def _plot_co_tip_diagnostics(co_rho_total, co_rho_delta, output_dir, origin, step, title_suffix=""):
    """Plot diagnostic slices of padded+rolled CO tip density."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'CO Tip Diagnostics {title_suffix}')

    def _plot_slice(ax, data, axis, idx, title):
        if axis == 0:
            sl = data[idx, :, :].T
            exts = [origin[1], origin[1] + data.shape[1]*step, origin[2], origin[2] + data.shape[2]*step]
            xl, yl = 'y [A]', 'z [A]'
        elif axis == 1:
            sl = data[:, idx, :].T
            exts = [origin[0], origin[0] + data.shape[0]*step, origin[2], origin[2] + data.shape[2]*step]
            xl, yl = 'x [A]', 'z [A]'
        else:
            sl = data[:, :, idx].T
            exts = [origin[0], origin[0] + data.shape[0]*step, origin[1], origin[1] + data.shape[1]*step]
            xl, yl = 'x [A]', 'y [A]'
        im = ax.imshow(sl, origin='lower', extent=exts, cmap='magma', aspect='equal')
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        # NOTE: always aspect='equal' for spatial maps (1 Å x = 1 Å y/z). Never aspect='auto'.
        plt.colorbar(im, ax=ax, fraction=0.03)

    nx, ny, nz = co_rho_total.shape
    # Slice through origin (0,0,0) where oxygen should be after roll
    ix, iy, iz = 0, 0, 0

    _plot_slice(axes[0, 0], co_rho_total, 0, ix, f'Total YZ (ix={ix} - through origin)')
    _plot_slice(axes[0, 1], co_rho_total, 1, iy, f'Total XZ (iy={iy} - through origin)')
    _plot_slice(axes[0, 2], co_rho_total, 2, iz, f'Total XY (iz={iz} - through origin)')

    _plot_slice(axes[1, 0], co_rho_delta, 0, ix, f'Delta YZ (ix={ix} - through origin)')
    _plot_slice(axes[1, 1], co_rho_delta, 1, iy, f'Delta XZ (iy={iy} - through origin)')
    _plot_slice(axes[1, 2], co_rho_delta, 2, iz, f'Delta XY (iz={iz} - through origin)')

    fname = os.path.join(output_dir, f'co_tip_diagnostics{title_suffix.replace(" ", "_")}.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  CO tip diagnostic plot: {fname}")


def run_afm_from_xyz(
    xyz_file,
    output_dir,
    basis,
    slako_prefix='mio-1-1',
    co_tip_dir=None,
    fdata_dir=None,
    fdata_basis=None,
    work_dir=None,
    step=0.1, margin=4.0, z_extra=6.0,
    scan_range=3.0, scan_step=0.1,
    height_range=(3.0, 6.5), height_step=0.1,
    pauli_params=None,
    pauli_fit_params=None,
    fit_pauli=False,
    fit_pauli_params=None,
    vdw_params={'C6_CO': 30.0},
    relax_params={'K_LAT': 0.5},
    plot_steps=True,
    use_dense_projection=False,
    max_shells=None,
    stm_params=None,
    ppm_mode=False,
    backend='dftb'
):
    """
    Full AFM simulation pipeline from .xyz to AFM images via DFTB+ density.

    Args:
        xyz_file: path to .xyz file
        output_dir: all outputs go here
        basis: basis list from parse_basis_hsd_ang (required)
        slako_prefix: Slater-Koster prefix for DFTB+
        co_tip_dir: directory with co_rho_total.npy + co_rho_delta.npy (optional;
                    if not provided or missing, CO is computed on-the-fly)
        fdata_dir: Fireball Fdata directory (required if co_tip_dir not provided)
        fdata_basis: OpenCL basis directory (required if co_tip_dir not provided)
        work_dir: DFTB+ scratch dir (default: output_dir/dftb_work)
        step/margin/z_extra: grid parameters
        scan_range/scan_points/height_range/height_step: scan parameters
        pauli_params/vdw_params/relax_params: physics parameters
        plot_steps: save intermediate plots
        use_dense_projection: use dense matrix projection (supports d-orbitals, faster)
        max_shells: max angular momentum shells (2=sp, 3=spd); auto-detected if None
        stm_params: dict with STM parameters for optional STM computation

    Returns:
        dict with 'df', 'intermediates', 'grid_spec'
    """
    import spammm.atomicUtils as au
    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'P':15,'S':16,'Br':35,'I':53}

    os.makedirs(output_dir, exist_ok=True)
    if work_dir is None:
        work_dir = os.path.join(output_dir, 'dftb_work')

    # Load molecule
    print(f"\nLoading molecule from {xyz_file}")
    pos, _, names, _, _ = au.load_xyz(xyz_file)
    atomPos  = np.array(pos, dtype=np.float64)
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in names], dtype=np.int32)
    print(f"  {len(atomPos)} atoms")

    # Scan grid (compute points from step size)
    x_min, x_max = atomPos[:,0].min()-scan_range, atomPos[:,0].max()+scan_range
    y_min, y_max = atomPos[:,1].min()-scan_range, atomPos[:,1].max()+scan_range
    scan_points_x = int(np.ceil((x_max - x_min) / scan_step))
    scan_points_y = int(np.ceil((y_max - y_min) / scan_step))
    scan_xs = np.linspace(x_min, x_max, scan_points_x)
    scan_ys = np.linspace(y_min, y_max, scan_points_y)
    heights  = np.arange(height_range[0], height_range[1], height_step)

    # Set up Slater-Koster path
    from spammm.quantum.DFTB_utils import SK_PATHS as _SK_PATHS
    if slako_prefix == 'mio-1-1':
        slako_prefix = _SK_PATHS.get('mio-1-1', slako_prefix)
    elif slako_prefix == '3ob-3-1':
        slako_prefix = _SK_PATHS.get('3ob-3-1', slako_prefix)

    # Get densities from DFTB+ (sparse or dense method)
    if work_dir is None:
        work_dir = os.path.join(output_dir, 'dftb_work')

    if use_dense_projection:
        # Use dense matrix projection (supports d-orbitals, faster)
        print("\nUsing dense matrix projection (supports d-orbitals)")
        # Use wfc.*.hsd file from spammm/DFTB/data/ (STO basis parameters, not waveplot_in.hsd)
        # Extract basis name from slako_prefix path (e.g., '/path/to/3ob-3-1/' -> '3ob-3-1')
        basis_name = slako_prefix.rstrip('/').split('/')[-1] if '/' in slako_prefix else slako_prefix
        if not basis_name:
            basis_name = '3ob-3-1'  # Default fallback
        _ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        basis_hsd_path = os.path.join(_ROOT, 'spammm', 'DFTB', 'data', f'wfc.{basis_name}.hsd')
        if not os.path.exists(basis_hsd_path):
            raise FileNotFoundError(f"Basis file not found: {basis_hsd_path}. Make sure wfc.{basis_name}.hsd exists in spammm/DFTB/data/")
        print(f"  Using basis file: {basis_hsd_path}")
        d = get_density_from_dftb_dense(atomPos, atomTypes, basis_hsd_path, work_dir,
                                          step=step, margin=margin, z_extra=z_extra,
                                          verbosity=1 if plot_steps else 0, max_shells=max_shells)
    else:
        # Use standard sparse projection
        d = get_density_from_dftb_plus(atomPos, atomTypes, basis, slako_prefix, work_dir,
                                          step=step, margin=margin, z_extra=z_extra)

    # Plot density slices to check anisotropy
    if plot_steps:
        z_heights = [0.0, 2.0, 2.5]
        for z in z_heights:
            iz = int(np.clip(np.round((z - d['origin'][2]) / step), 0, d['rho_scf'].shape[2]-1))
            plot_xy_slice(d['rho_scf'], d['origin'], step, iz, f'SCF Density z={z}A', f'step1_rho_scf_z{z:.1f}.png', output_dir)
            plot_xy_slice(d['rho_na'], d['origin'], step, iz, f'Neutral Atom Density z={z}A', f'step1_rho_na_z{z:.1f}.png', output_dir)
            plot_xy_slice(d['rho_diff'], d['origin'], step, iz, f'Delta Density z={z}A', f'step1_rho_diff_z{z:.1f}.png', output_dir, sym=True, cmap='bwr')

    # Save grid spec for later fitting
    grid_spec_path = os.path.join(output_dir, 'grid_spec.txt')
    with open(grid_spec_path, 'w') as f:
        f.write(f"origin = {d['origin'].tolist()}\n")
        f.write(f"ngrid = {d['ngrid'].tolist()}\n")
        f.write(f"step = {step}\n")
    print(f"  Saved grid spec to {grid_spec_path}")

    # CO tip: load precomputed or compute on-the-fly
    target_shape = tuple(d['ngrid'])
    co_origin = None
    if co_tip_dir is not None and os.path.isdir(co_tip_dir):
        print(f"\nLoading precomputed CO tip from {co_tip_dir}...")
        co_rho_total_raw = np.load(os.path.join(co_tip_dir, 'co_rho_total.npy'))
        co_rho_delta_raw = np.load(os.path.join(co_tip_dir, 'co_rho_delta.npy'))
        print(f"  Raw CO tip shape: {co_rho_total_raw.shape}")
    else:
        # Check global cache first
        if fdata_dir is None or fdata_basis is None:
            _ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
            fdata_dir = fdata_dir or os.path.join(_ROOT, 'tests', 'pyFireball', 'Fdata')
            fdata_basis = fdata_basis or os.path.join(fdata_dir, 'basis')

        cached = _get_cached_co_tip(step, margin, fdata_dir, fdata_basis, backend)
        if cached is not None:
            print(f"\nLoading cached CO tip (step={step}, margin={margin})...")
            co_rho_total_raw, co_rho_delta_raw = cached
            print(f"  Raw CO tip shape: {co_rho_total_raw.shape}")
        else:
            print(f"\nComputing CO tip on-the-fly (step={step})...")
            co_tip_work = os.path.join(output_dir, 'co_tip_work')
            os.makedirs(co_tip_work, exist_ok=True)
            co_grid_spec, co_ngrid, co_origin = _compute_co_tip_grid(step=step, margin=margin)
            print(f"  CO grid: ngrid={co_ngrid}, origin={co_origin}")
            _call_compute_co_tip_script(co_tip_work, co_grid_spec, step, 100, fdata_dir, fdata_basis, backend=backend)
            co_rho_total_raw = np.load(os.path.join(co_tip_work, 'co_rho_total.npy'))
            co_rho_delta_raw = np.load(os.path.join(co_tip_work, 'co_rho_delta.npy'))
            print(f"  Raw CO tip shape: {co_rho_total_raw.shape}")
            # Save to global cache for future runs
            _save_cached_co_tip(co_rho_total_raw, co_rho_delta_raw, step, margin, fdata_dir, fdata_basis, backend)
            print(f"  Cached CO tip for future runs.")

    # Pad with zeros and roll so O atom is at index 0
    print(f"  Padding CO tip to target shape {target_shape}...")
    co_rho_total = _pad_and_roll_co_tip(co_rho_total_raw, target_shape)
    co_rho_delta = _pad_and_roll_co_tip(co_rho_delta_raw, target_shape)
    print(f"  Padded+rolled CO tip shape: {co_rho_total.shape}")

    # Diagnostic plots
    if plot_steps:
        co_diag_dir = os.path.join(output_dir, 'co_tip_diagnostics')
        os.makedirs(co_diag_dir, exist_ok=True)
        if co_origin is not None:
            _plot_co_tip_diagnostics(co_rho_total_raw, co_rho_delta_raw, co_diag_dir, co_origin, step, title_suffix="_raw")
        _plot_co_tip_diagnostics(co_rho_total, co_rho_delta, co_diag_dir, d['origin'], step, title_suffix="_padded_rolled")
        # Also save central profiles to verify symmetry
        nx, ny, nz = co_rho_total.shape
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        xs = np.arange(nx) * step + d['origin'][0]
        axes[0].plot(xs, co_rho_total[:, cy, cz], 'b-', label='x profile')
        axes[0].axvline(xs[cx], color='r', ls='--', label='center')
        axes[0].set_title('X profile through center')
        axes[0].set_xlabel('x [A]')
        axes[0].legend()
        ys = np.arange(ny) * step + d['origin'][1]
        axes[1].plot(ys, co_rho_total[cx, :, cz], 'g-', label='y profile')
        axes[1].axvline(ys[cy], color='r', ls='--', label='center')
        axes[1].set_title('Y profile through center')
        axes[1].set_xlabel('y [A]')
        axes[1].legend()
        zs = np.arange(nz) * step + d['origin'][2]
        axes[2].plot(zs, co_rho_total[cx, cy, :], 'm-', label='z profile')
        axes[2].axvline(zs[cz], color='r', ls='--', label='center')
        axes[2].set_title('Z profile through center')
        axes[2].set_xlabel('z [A]')
        axes[2].legend()
        plt.tight_layout()
        prof_path = os.path.join(co_diag_dir, 'co_tip_center_profiles.png')
        plt.savefig(prof_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  CO tip profiles: {prof_path}")

    # Prepare STM parameters for run_afm_pipeline
    stm_kwargs = {}
    if stm_params and stm_params.get('compute', False):
        # STM requires dense projection data
        if not use_dense_projection:
            raise ValueError("STM computation requires use_dense_projection=True")
        print(f"[run_afm_from_xyz] STM requested: stm_params keys={list(stm_params.keys())} use_dense_projection={use_dense_projection}")
        if d.get('projector') is None or d.get('eigvecs') is None or d.get('eigvals') is None:
            raise ValueError(f"STM requested but missing dense projection outputs: projector={d.get('projector') is not None} eigvecs={d.get('eigvecs') is not None} eigvals={d.get('eigvals') is not None}. This indicates get_density_from_dftb_plus() didn't return them.")
        stm_kwargs['stm_params'] = stm_params
        stm_kwargs['projector'] = d.get('projector')
        stm_kwargs['norb_per_atom'] = d.get('norb_per_atom')
        stm_kwargs['orb_offsets'] = d.get('orb_offsets')
        stm_kwargs['atoms_dict'] = d.get('atoms_dict')
        stm_kwargs['eigvecs'] = d.get('eigvecs')
        stm_kwargs['eigvals'] = d.get('eigvals')

    return run_afm_pipeline(
        d['rho_scf'], d['rho_na'], d['rho_diff'], d['V_ES'],
        co_rho_total, co_rho_delta,
        atomPos, atomTypes,
        d['origin'], step, d['ngrid'],
        scan_xs, scan_ys, heights,
        output_dir,
        pauli_params=pauli_params, pauli_fit_params=pauli_fit_params,
        fit_pauli=fit_pauli, fit_pauli_params=fit_pauli_params,
        vdw_params=vdw_params, relax_params=relax_params, plot_steps=plot_steps,
        ppm_mode=ppm_mode,
        **stm_kwargs
    )


def plot_diagnostic_panel(E_pauli, E_es, E_vdw, E_total, origin, step, heights, output_dir):
    """Plot diagnostic panel with 4 columns (Total, Pauli, Electrostatics, vdW) and n-rows for heights.

    Each subplot has symmetric vmin/vmax zero-centered with its own colorbar (seismic colormap).
    Shows field slices at z=0 (molecular plane) for all heights to show field structure.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import map_coordinates
    n_heights = len(heights)
    fig, axes = plt.subplots(n_heights, 4, figsize=(16, 3*n_heights))
    if n_heights == 1:
        axes = axes.reshape(1, -1)

    # Use z=0 slice (molecular plane) for all heights to show field structure
    iz_0 = int(np.clip(np.round((0.0 - origin[2]) / step), 0, E_total.shape[2]-1))
    
    for iz, z in enumerate(heights):
        # Compute actual z-index from physical z coordinate
        iz_grid = int(np.clip(np.round((z - origin[2]) / step), 0, E_total.shape[2]-1))
        for icol, (field, title) in enumerate([
            (E_total, 'Total'),
            (E_pauli, 'Pauli'),
            (E_es, 'Electrostatics'),
            (E_vdw, 'vdW'),
        ]):
            ax = axes[iz, icol]
            slice_data = field[:, :, iz_grid]
            vmax = np.max(np.abs(slice_data))
            im = ax.imshow(slice_data.T, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
            ax.set_title(f'{title} z={z:.1f}Å iz={iz_grid}')
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95, wspace=0.25, hspace=0.2)
    plt.savefig(os.path.join(output_dir, 'diagnostic_panel.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved diagnostic panel: {os.path.join(output_dir, 'diagnostic_panel.png')}")


def plot_diagnostic_slices(E_pauli, E_es, E_vdw, origin, step, heights, output_dir):
    """Plot 3x3 diagnostic: Pauli, ES, vdW with XY, XZ, YZ slices through origin.

    All slices pass through origin (0,0,0) to show field structure.
    Probe heights are marked with gray dotted lines on XZ and YZ slices.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Energy Field Slices Through Origin (0,0,0)')
    
    # Slice indices through origin
    ix = 0
    iy = 0
    iz = int(np.clip(np.round((0.0 - origin[2]) / step), 0, E_pauli.shape[2]-1))
    
    fields = [(E_pauli, 'Pauli'), (E_es, 'Electrostatics'), (E_vdw, 'vdW')]
    
    for row, (field, title) in enumerate(fields):
        # XY slice
        xy_slice = field[:, :, iz].T
        vmax = np.max(np.abs(xy_slice))
        x_min = origin[0]
        x_max = origin[0] + field.shape[0] * step
        y_min = origin[1]
        y_max = origin[1] + field.shape[1] * step
        im = axes[row, 0].imshow(xy_slice, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                                 cmap='seismic', vmin=-vmax, vmax=vmax, aspect='equal')
        axes[row, 0].set_title(f'{title} XY (iz={iz})')
        axes[row, 0].set_xlabel('x [Å]')
        axes[row, 0].set_ylabel('y [Å]')
        plt.colorbar(im, ax=axes[row, 0], fraction=0.03, pad=0.02)
        
        # XZ slice
        xz_slice = field[ix, :, :].T
        vmax = np.max(np.abs(xz_slice))
        y_min = origin[1]
        y_max = origin[1] + field.shape[1] * step
        z_min = origin[2]
        z_max = origin[2] + field.shape[2] * step
        im = axes[row, 1].imshow(xz_slice, origin='lower', extent=[y_min, y_max, z_min, z_max], 
                                 cmap='seismic', vmin=-vmax, vmax=vmax, aspect='equal')
        axes[row, 1].set_title(f'{title} XZ (ix={ix})')
        axes[row, 1].set_xlabel('y [Å]')
        axes[row, 1].set_ylabel('z [Å]')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.03, pad=0.02)
        # Mark probe heights with gray dotted lines
        for h in heights:
            axes[row, 1].axhline(y=h, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        
        # YZ slice
        yz_slice = field[:, iy, :].T
        vmax = np.max(np.abs(yz_slice))
        x_min = origin[0]
        x_max = origin[0] + field.shape[0] * step
        im = axes[row, 2].imshow(yz_slice, origin='lower', extent=[x_min, x_max, z_min, z_max], 
                                 cmap='seismic', vmin=-vmax, vmax=vmax, aspect='equal')
        axes[row, 2].set_title(f'{title} YZ (iy={iy})')
        axes[row, 2].set_xlabel('x [Å]')
        axes[row, 2].set_ylabel('z [Å]')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.03, pad=0.02)
        # Mark probe heights with gray dotted lines
        for h in heights:
            axes[row, 2].axhline(y=h, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostic_panel_slices.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved diagnostic panel slices: {os.path.join(output_dir, 'diagnostic_panel_slices.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pauli Parameter Fitting (modular, reusable, portable)
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_pauli_powerlaw(z, overlap_raw, e_ref, z_min=2.0, z_max=3.5):
    """Contact-wall Pauli fit (original / default): E_ref(z) ≈ A * overlap(z)^beta.

    Keeps only points with overlap>0 and E_ref>0 in [z_min, z_max] (typically
    ~1.5–2.0 or 2.0–3.0 Å where Pauli dominates and ES/vdW are assumed small).
    Seed: log-linear; refine: scipy curve_fit OLS on those positive points.

    For AFM-range residual fit (Kriging−ES−vdW, signed, z~[2.5,5]) use
    `_fit_pauli_powerlaw_residual` instead — do not change this function.
    """
    from scipy.optimize import curve_fit

    z = np.asarray(z, dtype=np.float64)
    overlap_raw = np.asarray(overlap_raw, dtype=np.float64)
    e_ref = np.asarray(e_ref, dtype=np.float64)
    mask = (z >= z_min) & (z <= z_max) & np.isfinite(overlap_raw) & np.isfinite(e_ref)
    if mask.sum() < 3:
        raise ValueError(f"Need >=3 points in fit range [{z_min},{z_max}]")
    o_fit = overlap_raw[mask]
    e_fit = e_ref[mask]
    pos_mask = (o_fit > 1e-15) & (e_fit > 1e-15)
    if pos_mask.sum() < 3:
        raise ValueError("Not enough positive points")
    o_pos, e_pos = o_fit[pos_mask], e_fit[pos_mask]
    log_o = np.log(o_pos)
    log_e = np.log(e_pos)
    beta_ll, lnA_ll = np.polyfit(log_o, log_e, 1)
    A_ll = np.exp(lnA_ll)
    def model(overlap, A, beta):
        return A * overlap**beta
    try:
        popt, _ = curve_fit(model, o_pos, e_pos, p0=[A_ll, beta_ll],
                            bounds=([0.0, 0.0], [1e6, 5.0]))
        A_nls, beta_nls = popt
        e_pred = model(o_pos, A_nls, beta_nls)
        ss_res = np.sum((e_pos - e_pred)**2)
        ss_tot = np.sum((e_pos - np.mean(e_pos))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception as e:
        print(f"  WARNING: Nonlinear fit failed ({e}), using log-linear")
        A_nls, beta_nls = A_ll, beta_ll
        e_pred = model(o_pos, A_nls, beta_nls)
        ss_res = np.sum((e_pos - e_pred)**2)
        ss_tot = np.sum((e_pos - np.mean(e_pos))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return A_nls, beta_nls, r2, e_pred


def _fit_pauli_powerlaw_residual(z, overlap_raw, e_ref, z_min=2.5, z_max=5.0):
    """AFM-range residual Pauli fit: E_ref(z) ≈ A * overlap(z)^beta with signed E_ref.

    Intended target: E_ref = Kriging − E_es − E_vdw on z ∈ [2.5, 5.0] Å (typical).
    Allows negative E_ref (OLS over all finite points with overlap>0). Model A·S^β
    stays ≥0, so negative residual regions pull toward weaker Pauli.

    Switch vs contact wall: use `_fit_pauli_powerlaw` for close-range E_ref>0 only.
    Returns (A, beta, R², e_pred_on_fit_mask) same as `_fit_pauli_powerlaw`.
    """
    from scipy.optimize import curve_fit

    z = np.asarray(z, dtype=np.float64)
    overlap_raw = np.asarray(overlap_raw, dtype=np.float64)
    e_ref = np.asarray(e_ref, dtype=np.float64)
    mask = ((z >= z_min) & (z <= z_max) & np.isfinite(overlap_raw) & np.isfinite(e_ref)
            & (overlap_raw > 1e-15))
    if mask.sum() < 3:
        raise ValueError(f"Need >=3 points in residual fit range [{z_min},{z_max}]")
    o_fit = overlap_raw[mask]
    e_fit = e_ref[mask]
    pos = e_fit > 1e-15
    if pos.sum() >= 3:
        beta_ll, lnA_ll = np.polyfit(np.log(o_fit[pos]), np.log(e_fit[pos]), 1)
        A_ll, beta_ll = float(np.exp(lnA_ll)), float(beta_ll)
    else:
        A_ll, beta_ll = 10.0, 0.8
    def model(overlap, A, beta):
        return A * overlap**beta
    try:
        popt, _ = curve_fit(model, o_fit, e_fit, p0=[A_ll, beta_ll],
                            bounds=([0.0, 0.0], [1e6, 5.0]))
        A_nls, beta_nls = float(popt[0]), float(popt[1])
    except Exception as e:
        print(f"  WARNING: residual nonlinear fit failed ({e}), using seed")
        A_nls, beta_nls = A_ll, beta_ll
    e_pred = model(o_fit, A_nls, beta_nls)
    ss_res = np.sum((e_fit - e_pred)**2)
    ss_tot = np.sum((e_fit - np.mean(e_fit))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return A_nls, beta_nls, r2, e_pred


def _load_fdbm_grids(fdbm_dir):
    """Load FDBM forcefield grids from directory.
    
    Loads overlap_raw (raw Pauli overlap, A=1 beta=1) for fitting,
    plus E_Pauli_field, E_ES_field, E_vdw_field for diagnostics.
    """
    paths = {
        'overlap_raw': os.path.join(fdbm_dir, 'overlap_raw.npy'),
        'pauli': os.path.join(fdbm_dir, 'E_Pauli_field.npy'),
        'es':    os.path.join(fdbm_dir, 'E_ES_field.npy'),
        'vdw':   os.path.join(fdbm_dir, 'E_vdw_field.npy'),
    }
    grids = {}
    for key, path in paths.items():
        grids[key] = np.load(path) if os.path.exists(path) else None
    return grids


def _load_dftb_zscan(zscan_dir):
    """Load DFTB z-scan reference data."""
    z_path = os.path.join(zscan_dir, 'zscan_z.npy')
    e_path = os.path.join(zscan_dir, 'zscan_energy_eV.npy')
    if not (os.path.exists(z_path) and os.path.exists(e_path)):
        return None, None
    z = np.load(z_path)
    e = np.load(e_path)
    return z, e - e[-1]  # Relative energy


def _plot_pauli_fit(z, e_ref, e_fitted, A, beta, fname, title, z_min=2.0, z_max=3.5, ref_label='Ref'):
    """Plot per-atom Pauli fit (linear + log)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    mask = (z >= z_min) & (z <= z_max)
    # Linear
    ax = axes[0]
    ax.plot(z, e_ref, 'o-', color='tab:blue', markersize=3, label=ref_label, zorder=3)
    ax.plot(z[mask], e_fitted[mask], 's--', color='tab:red', markersize=3, label=f'Fit A={A:.2f} b={beta:.3f}', zorder=2)
    ax.axvspan(z_min, z_max, alpha=0.08, color='gray', label='Fit range')
    ax.set_xlabel('z [Å]')
    ax.set_ylabel('Energy [eV]')
    ax.set_title(f'{title} (Linear)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Log
    ax = axes[1]
    pos = e_ref > 1e-12
    ax.semilogy(z[pos], e_ref[pos], 'o-', color='tab:blue', markersize=3, label=ref_label)
    ax.semilogy(z[mask], e_fitted[mask], 's--', color='tab:red', markersize=3, label='Fit')
    ax.axvspan(z_min, z_max, alpha=0.08, color='gray')
    ax.set_xlabel('z [Å]')
    ax.set_ylabel('Energy [eV]')
    ax.set_title(f'{title} (Log)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def _plot_fitting_summary(all_results, fname, basis, z_min, z_max):
    """Plot summary comparing all atoms."""
    import matplotlib.pyplot as plt
    n_atoms = len(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Panel 1: A_pauli per atom
    ax = axes[0, 0]
    idxs = [r['idx'] for r in all_results]
    As = [r['A'] for r in all_results]
    ax.bar(idxs, As, color='tab:orange')
    ax.set_xlabel('Atom index')
    ax.set_ylabel('A_pauli')
    ax.set_title(f'A_pauli per atom ({basis})')
    ax.grid(True, alpha=0.3, axis='y')
    # Panel 2: beta per atom
    ax = axes[0, 1]
    betas = [r['beta'] for r in all_results]
    ax.bar(idxs, betas, color='tab:green')
    ax.set_xlabel('Atom index')
    ax.set_ylabel('beta_pauli')
    ax.set_title(f'beta_pauli per atom ({basis})')
    ax.grid(True, alpha=0.3, axis='y')
    # Panel 3: RMSE per atom
    ax = axes[1, 0]
    rmses = [r['rmse_fit'] for r in all_results]
    ax.bar(idxs, rmses, color='tab:red')
    ax.set_xlabel('Atom index')
    ax.set_ylabel('RMSE fit [eV]')
    ax.set_title(f'RMSE(fit range) per atom ({basis})')
    ax.grid(True, alpha=0.3, axis='y')
    # Panel 4: All fitted curves overlaid
    ax = axes[1, 1]
    for r in all_results:
        z = r['z']
        e_fit = r['e_fitted']
        ax.plot(z, e_fit, '-', lw=1.0, label=f"atom {r['idx']}")
    ax.set_xlabel('z [Å]')
    ax.set_ylabel('Fitted Pauli [eV]')
    ax.set_title(f'Fitted Pauli curves ({basis})')
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axvspan(z_min, z_max, alpha=0.08, color='gray')
    plt.suptitle(f'Multi-Atom Summary: {basis}', fontsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary: {fname}")


def _run_dftb_zscan_for_atom(target_idx, mol_names, mol_pos, tip_names, sk_prefix, 
                             z_distances, out_dir, xyz_path, tip_path):
    """Run DFTB z-scan for a single target atom. Returns z_vals, e_vals."""
    import time
    from spammm.quantum.DFTB_utils import SK_PATHS as _SK_PATHS, run_dftb_sp as _run_dftb_sp
    from spammm import atomicUtils as au
    
    HAU2EV = 27.211386245988
    target_name = mol_names[target_idx]
    target_pos = mol_pos[target_idx]
    atom_dir = os.path.join(out_dir, f'atom_{target_idx}')
    os.makedirs(atom_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Target atom {target_idx}: {target_name} at [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
    print(f"Output: {atom_dir}")
    print(f"{'='*60}")

    cache_path = os.path.join(atom_dir, 'zscan_results_cache.npz')
    results = []
    if os.path.exists(cache_path):
        print(f"Loading cache: {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        cached_z = cache['z_distances']
        if np.allclose(cached_z, z_distances):
            results = cache['results'].tolist()
            print(f"  Using {len(results)} cached results")
        else:
            print("  Cache z-range mismatch, recomputing")

    if len(results) != len(z_distances):
        combined_names = list(mol_names) + list(tip_names)
        for iz, z in enumerate(z_distances):
            print(f"\n[z-scan {iz+1}/{len(z_distances)}] z = {z:.2f} Å")
            o_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + z])
            c_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + z + 1.13])
            co_pos_shifted = np.array([o_pos, c_pos])
            combined_pos = np.vstack([mol_pos, co_pos_shifted])

            work_dir = os.path.join(atom_dir, f'zscan_z{z:.2f}')
            t_start = time.time()
            try:
                energy_ha = _run_dftb_sp(work_dir, combined_names, combined_pos, sk_prefix)
                energy_ev = energy_ha * HAU2EV
                t_elapsed = time.time() - t_start
                print(f"  Energy: {energy_ha:.8f} Ha = {energy_ev:.6f} eV  ({t_elapsed:.1f}s)")
                results.append({'z': float(z), 'energy_Ha': float(energy_ha), 'energy_eV': float(energy_ev)})
            except Exception as e:
                print(f"  ERROR: {e}")
                raise

        np.savez(cache_path, results=results, z_distances=z_distances)
        print(f"\nSaved cache to {cache_path}")

    z_vals = np.array([r['z'] for r in results])
    e_vals = np.array([r['energy_eV'] for r in results])
    np.save(os.path.join(atom_dir, 'zscan_z.npy'), z_vals)
    np.save(os.path.join(atom_dir, 'zscan_energy_eV.npy'), e_vals)

    with open(os.path.join(atom_dir, 'zscan_results.txt'), 'w') as f:
        f.write("DFTB Z-Scan Results\n")
        f.write("="*70 + "\n")
        f.write(f"Target: {target_name} [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]\n")
        f.write(f"CO bond: 1.13 Å\n\n")
        f.write(f"{'z [Å]':>10}  {'E [Ha]':>16}  {'E [eV]':>16}\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"{r['z']:10.2f}  {r['energy_Ha']:16.8f}  {r['energy_eV']:16.6f}\n")

    e_rel = e_vals - e_vals[-1]
    print(f"\nAtom {target_idx} complete: {len(z_vals)} points")
    print(f"  Rel energy at contact: {e_rel[0]:.4f} eV")
    return z_vals, e_vals


def fit_pauli_parameters(xyz_file, basis='mio-1-1', target_indices=[0], 
                         fdbm_dir=None, zscan_dir=None, output_dir='fit_pauli',
                         z_min=2.0, z_max=3.0, generate_ref=False,
                         step=0.1, margin=4.0, z_extra=6.0,
                         sk_prefix=None, tip_xyz='CO.xyz',
                         scan_range=3.0, scan_step=0.1, height_range=[2.8, 3.6], height_step=0.1):
    """High-level modular function to fit Pauli parameters against DFTB reference.
    
    This function integrates the full fitting workflow:
    1. If fdbm_dir is None/missing: run FDBM pipeline with new CO tip handling
    2. If zscan_dir is None/missing and generate_ref=True: run DFTB z-scan
    3. Load FDBM grids and DFTB z-scan data
    4. For each target atom: extract profile, fit power-law, save results
    5. Generate summary plots and table
    
    Args:
        xyz_file: Path to molecule XYZ file
        basis: DFTB+ basis set (mio-1-1, 3ob-3-1, etc.)
        target_indices: List of atom indices to fit (e.g., [0, 1, 20, 21])
        fdbm_dir: Pre-computed FDBM grid directory (if None, generates on-the-fly)
        zscan_dir: Pre-computed DFTB z-scan directory (if None and generate_ref=True, generates)
        output_dir: Output directory for fitting results
        z_min, z_max: Fit range in Å (contact region)
        generate_ref: Whether to generate DFTB z-scan if missing
        step, margin, z_extra: Grid parameters for FDBM generation
        sk_prefix: DFTB+ Slater-Koster path (if None, uses default from dftb_utils)
        tip_xyz: Tip molecule XYZ file (default: CO.xyz)
        scan_range, scan_step: AFM scan parameters (for FDBM grid generation)
        height_range, height_step: AFM height parameters (for FDBM grid generation)
    
    Returns:
        dict: Fitting results with keys:
            - 'basis': basis set name
            - 'atoms': list of per-atom results (dict with A, beta, rmse, etc.)
            - 'A_mean', 'beta_mean': mean values across atoms
            - 'A_std', 'beta_std': standard deviations
    """
    import json
    import time
    from spammm import atomicUtils as au
    from spammm.quantum.DFTB import TestUtils as tu
    from spammm.quantum.DFTB_utils import SK_PATHS as _SK_PATHS
    
    A_PAULI_DEFAULT = 16.0
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load molecule
    mol_pos, _, mol_names, _, _ = au.load_xyz(xyz_file)
    mol_pos = np.array(mol_pos, dtype=np.float64)
    for idx in target_indices:
        if idx < 0 or idx >= len(mol_names):
            raise ValueError(f"Target index {idx} out of range (0-{len(mol_names)-1})")
    
    # Set up SK path
    if sk_prefix is None:
        if basis in _SK_PATHS:
            sk_prefix = _SK_PATHS[basis]
        else:
            raise ValueError(f"Basis '{basis}' not found in DFTB_utils.SK_PATHS; provide sk_prefix explicitly")
    
    # Step 1: Generate FDBM grids if needed
    if fdbm_dir is None or not os.path.isdir(fdbm_dir):
        print(f"\nGenerating FDBM grids for {basis}...")
        fdbm_dir = os.path.join(output_dir, f'fdbm_grids_{basis.replace("-", "_")}')
        os.makedirs(fdbm_dir, exist_ok=True)
        
        run_afm_from_xyz(
            xyz_file, output_dir=fdbm_dir, basis=basis,
            step=step, margin=margin, z_extra=z_extra,
            scan_range=scan_range, scan_step=scan_step,
            height_range=height_range, height_step=height_step,
            co_tip_dir=None,  # Force on-the-fly CO computation with new padding/rolling
            plot_steps=False
        )
        print(f"  FDBM grids saved to: {fdbm_dir}")
    
    # Load FDBM grids
    grids = _load_fdbm_grids(fdbm_dir)
    if grids['overlap_raw'] is None:
        raise FileNotFoundError(f"overlap_raw.npy not found in {fdbm_dir}. Run pipeline first (it saves raw overlap).")
    
    # Read grid spec
    log_path_grid = os.path.join(fdbm_dir, 'step1_density', 'log.txt')
    grid_spec_path = os.path.join(fdbm_dir, 'grid_spec.txt')
    origin, ngrid, step_grid = afm.read_grid_spec_from_log(log_path_grid)
    
    # Try grid_spec.txt (new format)
    if origin is None and os.path.exists(grid_spec_path):
        with open(grid_spec_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('origin ='):
                    origin = np.array(eval(line.split('=')[1].strip()))
                elif line.startswith('ngrid ='):
                    ngrid = np.array(eval(line.split('=')[1].strip()))
                elif line.startswith('step ='):
                    step_grid = float(line.split('=')[1].strip())
    
    if origin is None:
        # Fallback: compute from grid shape and molecule
        pauli_shape = grids['pauli'].shape
        ngrid = pauli_shape
        step_grid = step  # Use the step parameter
        # Better estimate: center grid on molecule
        mol_center = mol_pos.mean(axis=0)
        grid_size = np.array(ngrid) * step_grid
        origin = mol_center - 0.5 * grid_size
        print(f"  WARNING: Could not read grid spec, estimated from molecule center")
    print(f"  Grid: origin={origin.round(2)} ngrid={ngrid} step={step_grid}")
    
    # Step 2: Generate DFTB z-scan if needed
    zscan_dir_base = zscan_dir if zscan_dir else os.path.join(output_dir, f'zscan_{basis.replace("-", "_")}')
    
    if generate_ref:
        print(f"\nGenerating DFTB z-scan reference for {basis}...")
        tip_path = os.path.join(os.path.dirname(xyz_file), tip_xyz)
        tip_pos, _, tip_names, _, _ = au.load_xyz(tip_path)
        
        z_distances = np.arange(2.0, 10.0 + 0.15*0.5, 0.15)
        print(f"  Z-scan: {len(z_distances)} points from {z_distances.min():.2f} to {z_distances.max():.2f} Å")
        
        for target_idx in target_indices:
            _run_dftb_zscan_for_atom(
                target_idx, mol_names, mol_pos, tip_names, sk_prefix,
                z_distances, zscan_dir_base, xyz_file, tip_path
            )
        print(f"  DFTB z-scan saved to: {zscan_dir_base}")
    
    # Step 3: Fit each atom
    all_results = []
    for target_idx in target_indices:
        target_name = mol_names[target_idx]
        target_pos = mol_pos[target_idx]
        atom_out_dir = os.path.join(output_dir, f'atom_{target_idx}')
        zscan_atom_dir = os.path.join(zscan_dir_base, f'atom_{target_idx}')
        os.makedirs(atom_out_dir, exist_ok=True)
        
        print(f"\nAtom {target_idx} ({target_name}) at [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
        
        # Load DFTB z-scan
        z_ref, e_ref = _load_dftb_zscan(zscan_atom_dir)
        if z_ref is None:
            raise FileNotFoundError(f"No z-scan found in {zscan_atom_dir}. Generate with --fit_generate_ref or provide valid --fit_zscan_dir")
        
        # Extract raw overlap profile at atom XY, at absolute z = target_pos[2] + z_ref
        # z_ref are distances ABOVE the atom (e.g. 2.0..10.0 Å)
        # extract_z_profile uses: z_abs = atom_pos[2] + z_distances
        overlap_col = tu.extract_z_profile(grids['overlap_raw'], target_pos, origin, step_grid, z_distances=z_ref)
        if overlap_col is None:
            raise ValueError(f"overlap_raw extraction failed for atom {target_idx}")
        overlap_safe = np.clip(overlap_col, 1e-30, None)
        
        # Diagnostics: verify z-grid alignment
        grid_z_range = [origin[2], origin[2] + grids['overlap_raw'].shape[2] * step_grid]
        print(f"  Grid z: [{grid_z_range[0]:.2f}, {grid_z_range[1]:.2f}] Å")
        print(f"  z_ref: [{z_ref.min():.2f}, {z_ref.max():.2f}] Å (above atom at z={target_pos[2]:.2f})")
        print(f"  z_abs at contact: {target_pos[2] + z_ref.min():.2f} Å")
        print(f"  overlap at z_ref[0]={z_ref[0]:.2f}: {overlap_col[0]:.4e}")
        idx_z3 = np.argmin(np.abs(z_ref - 3.0))
        print(f"  overlap at z_ref~3.0 (={z_ref[idx_z3]:.2f}): {overlap_col[idx_z3]:.4e}")
        print(f"  e_ref at z_ref~3.0: {e_ref[idx_z3]:.4e} eV")
        
        # Fit: E_DFTB = A_fit * overlap^beta_fit
        A_fit, beta_fit, r2, e_fitted_range = _fit_pauli_powerlaw(
            z_ref, overlap_safe, e_ref, z_min=z_min, z_max=z_max
        )
        
        e_fitted_all = A_fit * overlap_safe**beta_fit
        mask_fit = (z_ref >= z_min) & (z_ref <= z_max)
        rmse_fit = np.sqrt(np.mean((e_ref[mask_fit] - e_fitted_range)**2))
        rmse_all = np.sqrt(np.mean((e_ref - e_fitted_all)**2))
        
        print(f"  Fit: A={A_fit:.4f}, beta={beta_fit:.4f}, R2={r2:.6f}, RMSE(fit)={rmse_fit:.4e} eV")
        print(f"  Consistency check at z_ref~3.0: E_fit={e_fitted_all[idx_z3]:.4e} vs E_DFTB={e_ref[idx_z3]:.4e} eV")
        
        # Save
        params = {
            'basis': basis, 'atom_idx': target_idx, 'atom_name': target_name,
            'A_pauli': float(A_fit), 'beta_pauli': float(beta_fit),
            'R2_fit': float(r2), 'RMSE_fit': float(rmse_fit), 'RMSE_all': float(rmse_all),
            'fit_z_min': z_min, 'fit_z_max': z_max,
        }
        with open(os.path.join(atom_out_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2)
        
        np.save(os.path.join(atom_out_dir, 'z_ref.npy'), z_ref)
        np.save(os.path.join(atom_out_dir, 'e_ref.npy'), e_ref)
        np.save(os.path.join(atom_out_dir, 'overlap_col.npy'), overlap_col)
        np.save(os.path.join(atom_out_dir, 'e_fitted.npy'), e_fitted_all)
        
        # Plot
        _plot_pauli_fit(
            z_ref, e_ref, e_fitted_all, A_fit, beta_fit,
            fname=os.path.join(atom_out_dir, 'fit_pauli.png'),
            title=f'{target_name}{target_idx} ({basis})',
            z_min=z_min, z_max=z_max
        )
        
        all_results.append({
            'idx': target_idx, 'name': target_name, 'pos': target_pos,
            'A': A_fit, 'beta': beta_fit, 'r2': r2,
            'rmse_fit': rmse_fit, 'rmse_all': rmse_all,
            'z': z_ref, 'e_fitted': e_fitted_all, 'e_ref': e_ref,
        })
    
    # Step 4: Summary
    if len(all_results) > 1:
        _plot_fitting_summary(all_results, os.path.join(output_dir, 'summary_all_atoms.png'), basis, z_min, z_max)
    
    # Write summary table
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("FDBM Pauli Fitting Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Basis: {basis}\n")
        f.write(f"Atoms: {[r['idx'] for r in all_results]}\n")
        f.write(f"Fit range: z=[{z_min}, {z_max}] Å\n\n")
        f.write(f"{'Atom':>6} {'Name':>4} {'A_pauli':>10} {'beta':>8} {'R2':>10} {'RMSE_fit':>10} {'RMSE_all':>10}\n")
        f.write("-"*70 + "\n")
        for r in all_results:
            f.write(f"{r['idx']:6d} {r['name']:>4} {r['A']:10.2f} {r['beta']:8.4f} {r['r2']:10.6f} {r['rmse_fit']:10.4f} {r['rmse_all']:10.4f}\n")
        
        if len(all_results) > 1:
            As = [r['A'] for r in all_results]
            betas = [r['beta'] for r in all_results]
            f.write(f"\nMean ± std:\n")
            f.write(f"  A_pauli: {np.mean(As):.2f} ± {np.std(As):.2f}\n")
            f.write(f"  beta:    {np.mean(betas):.4f} ± {np.std(betas):.4f}\n")
        f.write(f"\nTime: {time.time()-t0:.1f}s\n")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    # Return structured results
    result_dict = {
        'basis': basis,
        'atoms': all_results,
        'A_mean': np.mean([r['A'] for r in all_results]) if all_results else None,
        'beta_mean': np.mean([r['beta'] for r in all_results]) if all_results else None,
        'A_std': np.std([r['A'] for r in all_results]) if all_results else None,
        'beta_std': np.std([r['beta'] for r in all_results]) if all_results else None,
    }
    return result_dict


# =============================================================================
# pySCF-Specific Pauli Fitting
# =============================================================================

def _run_pyscf_zscan_for_atom(atom_name, atom_pos, tip_pos, tip_names, 
                              z_distances, output_dir, pyscf_method='RHF', 
                              pyscf_basis='sto-3g', pyscf_xc=None):
    """Run pySCF z-scan for isolated atom with CO tip.
    
    Computes interaction energy between isolated atom and CO tip at various heights.
    
    Args:
        atom_name: Element symbol (e.g., 'C', 'H', 'O')
        atom_pos: Atom position (3,) array in Angstrom
        tip_pos: CO tip atomic positions (N,3) array in Angstrom
        tip_names: CO tip element names (list of N strings)
        z_distances: Array of tip heights above atom (in Angstrom)
        output_dir: Directory to save results
        pyscf_method: pySCF SCF method ('RHF' or 'RKS')
        pyscf_basis: pySCF basis set (e.g., 'sto-3g', '6-31g')
        pyscf_xc: DFT XC functional for RKS (e.g., 'lda,vwn', 'pbe')
    
    Returns:
        z_array: Array of z distances (Å)
        e_array: Array of interaction energies (eV)
    """
    import pyscf
    from pyscf import gto, scf, dft
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build isolated atom pySCF molecule at origin
    atom_z = int(gto.charge(atom_name))
    mol_atom = gto.M(
        atom=f'{atom_name} 0.0 0.0 0.0',
        basis=pyscf_basis,
        charge=0,
        spin=1 if atom_z % 2 == 1 else 0,  # Handle odd-electron atoms
        unit='Ang'
    )
    
    # Run SCF for isolated atom
    if pyscf_method.upper() == 'RHF':
        mf_atom = scf.RHF(mol_atom)
    elif pyscf_method.upper() == 'RKS':
        mf_atom = dft.RKS(mol_atom)
        if pyscf_xc is not None:
            mf_atom.xc = pyscf_xc
    else:
        raise ValueError(f"Unknown method: {pyscf_method}")
    
    mf_atom.kernel()
    e_atom = mf_atom.e_tot
    
    # Tip-only SCF (CO at origin)
    tip_str = '\n'.join([f'{name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}' 
                         for name, pos in zip(tip_names, tip_pos)])
    mol_tip = gto.M(
        atom=tip_str,
        basis=pyscf_basis,
        charge=0,
        spin=0,  # CO is closed-shell
        unit='Ang'
    )
    
    if pyscf_method.upper() == 'RHF':
        mf_tip = scf.RHF(mol_tip)
    elif pyscf_method.upper() == 'RKS':
        mf_tip = dft.RKS(mol_tip)
        if pyscf_xc is not None:
            mf_tip.xc = pyscf_xc
    
    mf_tip.kernel()
    e_tip = mf_tip.e_tot
    
    # Scan tip heights
    z_array = []
    e_array = []
    
    print(f"  Running pySCF z-scan for {atom_name}: {len(z_distances)} heights")
    t0 = time.time()
    
    for i, z in enumerate(z_distances):
        # Position tip above atom at height z (oxygen at z above atom)
        # Maintain relative tip geometry
        tip_pos_shifted = tip_pos.copy()
        tip_pos_shifted[:, 2] = tip_pos_shifted[:, 2] + z  # Add z offset to maintain relative geometry
        
        tip_str_shifted = '\n'.join([f'{name} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}' 
                                     for name, pos in zip(tip_names, tip_pos_shifted)])
        
        # Combined system
        mol_combined = gto.M(
            atom=f'{atom_name} 0.0 0.0 0.0\n' + tip_str_shifted,
            basis=pyscf_basis,
            charge=0,
            spin=1 if atom_z % 2 == 1 else 0,  # Use atom spin
            unit='Ang'
        )
        
        if pyscf_method.upper() == 'RHF':
            mf_combined = scf.RHF(mol_combined)
        elif pyscf_method.upper() == 'RKS':
            mf_combined = dft.RKS(mol_combined)
            if pyscf_xc is not None:
                mf_combined.xc = pyscf_xc
        
        mf_combined.kernel()
        e_combined = mf_combined.e_tot
        
        # Interaction energy (Hartree -> eV)
        HARTREE_TO_EV = 27.2114
        e_int = (e_combined - e_atom - e_tip) * HARTREE_TO_EV
        
        z_array.append(z)
        e_array.append(e_int)
        
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(z_distances)}: z={z:.2f} Å, E_int={e_int:.4f} eV")
    
    print(f"  pySCF z-scan done in {time.time()-t0:.1f}s")
    
    # Reference energies to farthest distance (makes all positive, like DFTB)
    e_array = np.array(e_array)
    e_array = e_array - e_array[-1]
    
    # Take absolute value for Pauli repulsion (always repulsive)
    e_array = np.abs(e_array)
    
    # Save results
    np.save(os.path.join(output_dir, 'zscan_z.npy'), np.array(z_array))
    np.save(os.path.join(output_dir, 'zscan_energy_eV.npy'), e_array)
    
    return np.array(z_array), np.array(e_array)


def _load_pyscf_zscan(zscan_dir):
    """Load pySCF z-scan reference data."""
    z_path = os.path.join(zscan_dir, 'zscan_z.npy')
    e_path = os.path.join(zscan_dir, 'zscan_energy_eV.npy')
    if not (os.path.exists(z_path) and os.path.exists(e_path)):
        return None, None
    z = np.load(z_path)
    e = np.load(e_path)
    return z, e - e[-1]  # Reference to farthest distance


def fit_pauli_parameters_pyscf(xyz_file, pyscf_basis='sto-3g', pyscf_method='RHF', 
                                pyscf_xc=None, target_indices=[0], 
                                fdbm_dir=None, zscan_dir=None, output_dir='fit_pauli_pyscf',
                                z_min=2.0, z_max=3.0, generate_ref=False,
                                step=0.15, margin=4.0, z_extra=5.0,
                                tip_xyz='CO.xyz',
                                scan_range=3.0, scan_step=0.15, 
                                height_range=[2.8, 3.6], height_step=0.15):
    """Fit Pauli parameters for pySCF backend against pySCF reference.
    
    This function integrates the full fitting workflow for pySCF:
    1. If fdbm_dir is None/missing: run FDBM pipeline with pySCF backend
    2. If zscan_dir is None/missing and generate_ref=True: run pySCF z-scan
    3. Load FDBM grids (raw overlap) and pySCF z-scan data
    4. For each target atom: extract profile, fit power-law, save results
    5. Generate summary plots and table
    
    Fitting model (NO magic numbers):
        E_ref(z) = A * overlap(z)^beta
    where overlap(z) is the raw density overlap integral (A=1, beta=1)
    
    Args:
        xyz_file: Path to molecule XYZ file
        pyscf_basis: pySCF basis set (sto-3g, 6-31g, etc.)
        pyscf_method: pySCF SCF method ('RHF' or 'RKS')
        pyscf_xc: DFT XC functional for RKS (e.g., 'lda,vwn', 'pbe')
        target_indices: List of atom indices to fit (e.g., [0, 1, 20, 21])
        fdbm_dir: Pre-computed FDBM grid directory (if None, generates on-the-fly)
        zscan_dir: Pre-computed pySCF z-scan directory (if None and generate_ref=True, generates)
        output_dir: Output directory for fitting results
        z_min, z_max: Fit range in Å (contact region)
        generate_ref: Whether to generate pySCF z-scan if missing
        step, margin, z_extra: Grid parameters for FDBM generation
        tip_xyz: Tip molecule XYZ file (default: CO.xyz)
        scan_range, scan_step: AFM scan parameters (for FDBM grid generation)
        height_range, height_step: AFM height parameters (for FDBM grid generation)
    
    Returns:
        dict: Fitting results with keys:
            - 'basis': pySCF basis set name
            - 'method': pySCF SCF method
            - 'atoms': list of per-atom results (dict with A, beta, rmse, etc.)
            - 'A_mean', 'beta_mean': mean values across atoms
            - 'A_std', 'beta_std': standard deviations
    """
    import json
    import time
    from spammm import atomicUtils as au
    from spammm.quantum.DFTB import TestUtils as tu
    
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load molecule
    mol_pos, _, mol_names, _, _ = au.load_xyz(xyz_file)
    mol_pos = np.array(mol_pos, dtype=np.float64)
    for idx in target_indices:
        if idx < 0 or idx >= len(mol_names):
            raise ValueError(f"Target index {idx} out of range (0-{len(mol_names)-1})")
    
    # Load CO tip
    tip_path = os.path.join(os.path.dirname(xyz_file), tip_xyz)
    tip_pos, _, tip_names, _, _ = au.load_xyz(tip_path)
    tip_pos = np.array(tip_pos, dtype=np.float64)
    
    # Step 1: Generate FDBM grids with pySCF backend if needed
    if fdbm_dir is None or not os.path.isdir(fdbm_dir):
        print(f"\nGenerating FDBM grids with pySCF backend ({pyscf_basis})...")
        fdbm_dir = os.path.join(output_dir, f'fdbm_grids_pyscf_{pyscf_basis}')
        os.makedirs(fdbm_dir, exist_ok=True)
        
        # Use ModularPipeline with pySCF backend
        from spammm.SPM import ModularPipeline as mp_mod
        
        pipeline = mp_mod.ModularAFMPipeline(
            xyz_file=xyz_file,
            output_dir=fdbm_dir,
            basis=None,  # Not used for pySCF
            slako_prefix=None,
            step=step, margin=margin, z_extra=z_extra,
            scan_range=scan_range, scan_step=scan_step,
            height_range=height_range, height_step=height_step,
            backend='pyscf',
            pyscf_params={'method': pyscf_method, 'basis': pyscf_basis, 'xc': pyscf_xc}
        )
        
        # Run stages 1-3 (SCF, density, potentials) to get overlap_raw
        dm_dense, eigvecs, eigvals = pipeline.stage1_scf(force_recompute=True)
        rho_scf, rho_na, rho_diff = pipeline.stage2_project(dm_dense, force_recompute=True)
        V_ES, E_pauli, E_ES, E_vdw, F_total = pipeline.stage3_potentials(
            rho_scf, rho_na, rho_diff, force_recompute=True,
            pauli_params={'A': 1.0, 'beta': 1.0}  # Raw overlap (no scaling)
        )
        
        # Save raw overlap for fitting
        np.save(os.path.join(fdbm_dir, 'overlap_raw.npy'), E_pauli)
        np.save(os.path.join(fdbm_dir, 'E_Pauli_field.npy'), E_pauli)
        np.save(os.path.join(fdbm_dir, 'E_ES_field.npy'), E_ES)
        np.save(os.path.join(fdbm_dir, 'E_vdw_field.npy'), E_vdw)
        
        # Save grid spec
        np.savez(os.path.join(fdbm_dir, 'grid_spec.npz'),
                 origin=pipeline.origin, ngrid=pipeline.ngrid, step=pipeline.step)
        
        print(f"  FDBM grids saved to: {fdbm_dir}")
    
    # Load FDBM grids
    grids = _load_fdbm_grids(fdbm_dir)
    if grids['overlap_raw'] is None:
        raise FileNotFoundError(f"overlap_raw.npy not found in {fdbm_dir}")
    
    # Read grid spec
    grid_spec_path = os.path.join(fdbm_dir, 'grid_spec.npz')
    if os.path.exists(grid_spec_path):
        grid_data = np.load(grid_spec_path, allow_pickle=True)
        origin = grid_data['origin']
        ngrid = grid_data['ngrid']
        step_grid = float(grid_data['step'])
    else:
        # Fallback: estimate from grid shape
        pauli_shape = grids['overlap_raw'].shape
        ngrid = pauli_shape
        step_grid = step
        mol_center = mol_pos.mean(axis=0)
        grid_size = np.array(ngrid) * step_grid
        origin = mol_center - 0.5 * grid_size
        print(f"  WARNING: Could not read grid spec, estimated from molecule center")
    
    print(f"  Grid: origin={origin.round(2)} ngrid={ngrid} step={step_grid}")
    
    # Step 2: Generate pySCF z-scan if needed
    zscan_dir_base = zscan_dir if zscan_dir else os.path.join(output_dir, f'zscan_pyscf_{pyscf_basis}')
    
    if generate_ref:
        print(f"\nGenerating pySCF z-scan reference ({pyscf_basis})...")
        z_distances = np.arange(2.0, 30.0 + 0.15*0.5, 0.15)
        print(f"  Z-scan: {len(z_distances)} points from {z_distances.min():.2f} to {z_distances.max():.2f} Å")
        
        for target_idx in target_indices:
            target_name = mol_names[target_idx]
            atom_out_dir = os.path.join(zscan_dir_base, f'atom_{target_idx}')
            os.makedirs(atom_out_dir, exist_ok=True)
            
            _run_pyscf_zscan_for_atom(
                target_name, mol_pos[target_idx], tip_pos, tip_names,
                z_distances, atom_out_dir, pyscf_method, pyscf_basis, pyscf_xc
            )
        print(f"  pySCF z-scan saved to: {zscan_dir_base}")
    
    # Step 3: Fit each atom
    all_results = []
    for target_idx in target_indices:
        target_name = mol_names[target_idx]
        target_pos = mol_pos[target_idx]
        atom_out_dir = os.path.join(output_dir, f'atom_{target_idx}')
        zscan_atom_dir = os.path.join(zscan_dir_base, f'atom_{target_idx}')
        os.makedirs(atom_out_dir, exist_ok=True)
        
        print(f"\nAtom {target_idx} ({target_name}) at [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
        
        # Load pySCF z-scan
        z_ref, e_ref = _load_pyscf_zscan(zscan_atom_dir)
        if z_ref is None:
            raise FileNotFoundError(f"No z-scan found in {zscan_atom_dir}. Generate with generate_ref=True or provide valid zscan_dir")
        
        # Extract raw overlap profile at atom XY
        overlap_col = tu.extract_z_profile(grids['overlap_raw'], target_pos, origin, step_grid, z_distances=z_ref)
        if overlap_col is None:
            raise ValueError(f"overlap_raw extraction failed for atom {target_idx}")
        overlap_safe = np.clip(overlap_col, 1e-30, None)
        
        # Diagnostics
        grid_z_range = [origin[2], origin[2] + grids['overlap_raw'].shape[2] * step_grid]
        print(f"  Grid z: [{grid_z_range[0]:.2f}, {grid_z_range[1]:.2f}] Å")
        print(f"  z_ref: [{z_ref.min():.2f}, {z_ref.max():.2f}] Å (above atom at z={target_pos[2]:.2f})")
        print(f"  overlap at z_ref[0]={z_ref[0]:.2f}: {overlap_col[0]:.4e}")
        idx_z3 = np.argmin(np.abs(z_ref - 3.0))
        print(f"  overlap at z_ref~3.0 (={z_ref[idx_z3]:.2f}): {overlap_col[idx_z3]:.4e}")
        print(f"  e_ref at z_ref~3.0: {e_ref[idx_z3]:.4e} eV")
        
        # Fit: E_pySCF = A_fit * overlap^beta_fit (NO magic numbers)
        A_fit, beta_fit, r2, e_fitted_range = _fit_pauli_powerlaw(
            z_ref, overlap_safe, e_ref, z_min=z_min, z_max=z_max
        )
        
        e_fitted_all = A_fit * overlap_safe**beta_fit
        mask_fit = (z_ref >= z_min) & (z_ref <= z_max)
        rmse_fit = np.sqrt(np.mean((e_ref[mask_fit] - e_fitted_range)**2))
        rmse_all = np.sqrt(np.mean((e_ref - e_fitted_all)**2))
        
        print(f"  Fit: A={A_fit:.4f}, beta={beta_fit:.4f}, R2={r2:.6f}, RMSE(fit)={rmse_fit:.4e} eV")
        print(f"  Consistency check at z_ref~3.0: E_fit={e_fitted_all[idx_z3]:.4e} vs E_pySCF={e_ref[idx_z3]:.4e} eV")
        
        # Save
        params = {
            'basis': pyscf_basis, 'method': pyscf_method, 'xc': pyscf_xc,
            'atom_idx': target_idx, 'atom_name': target_name,
            'A_pauli': float(A_fit), 'beta_pauli': float(beta_fit),
            'R2_fit': float(r2), 'RMSE_fit': float(rmse_fit), 'RMSE_all': float(rmse_all),
            'fit_z_min': z_min, 'fit_z_max': z_max,
        }
        with open(os.path.join(atom_out_dir, 'params.json'), 'w') as f:
            json.dump(params, f, indent=2)
        
        np.save(os.path.join(atom_out_dir, 'z_ref.npy'), z_ref)
        np.save(os.path.join(atom_out_dir, 'e_ref.npy'), e_ref)
        np.save(os.path.join(atom_out_dir, 'overlap_col.npy'), overlap_col)
        np.save(os.path.join(atom_out_dir, 'e_fitted.npy'), e_fitted_all)
        
        # Plot
        _plot_pauli_fit(
            z_ref, e_ref, e_fitted_all, A_fit, beta_fit,
            fname=os.path.join(atom_out_dir, 'fit_pauli.png'),
            title=f'{target_name}{target_idx} (pySCF {pyscf_basis})',
            z_min=z_min, z_max=z_max,
            ref_label='pySCF Ref'
        )
        
        all_results.append({
            'idx': target_idx, 'name': target_name, 'pos': target_pos,
            'A': A_fit, 'beta': beta_fit, 'r2': r2,
            'rmse_fit': rmse_fit, 'rmse_all': rmse_all,
            'z': z_ref, 'e_fitted': e_fitted_all, 'e_ref': e_ref,
        })
    
    # Step 4: Summary
    if len(all_results) > 1:
        _plot_fitting_summary(all_results, os.path.join(output_dir, 'summary_all_atoms.png'), 
                            f'pySCF {pyscf_basis}', z_min, z_max)
    
    # Write summary table
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("pySCF Pauli Fitting Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Basis: {pyscf_basis}\n")
        f.write(f"Method: {pyscf_method}\n")
        f.write(f"XC: {pyscf_xc if pyscf_xc else 'N/A'}\n")
        f.write(f"Atoms: {[r['idx'] for r in all_results]}\n")
        f.write(f"Fit range: z=[{z_min}, {z_max}] Å\n\n")
        f.write(f"{'Atom':>6} {'Name':>4} {'A_pauli':>10} {'beta':>8} {'R2':>10} {'RMSE_fit':>10} {'RMSE_all':>10}\n")
        f.write("-"*70 + "\n")
        for r in all_results:
            f.write(f"{r['idx']:6d} {r['name']:>4} {r['A']:10.2f} {r['beta']:8.4f} {r['r2']:10.6f} {r['rmse_fit']:10.4f} {r['rmse_all']:10.4f}\n")
        
        if len(all_results) > 1:
            As = [r['A'] for r in all_results]
            betas = [r['beta'] for r in all_results]
            f.write(f"\nMean ± std:\n")
            f.write(f"  A_pauli: {np.mean(As):.2f} ± {np.std(As):.2f}\n")
            f.write(f"  beta:    {np.mean(betas):.4f} ± {np.std(betas):.4f}\n")
        f.write(f"\nTime: {time.time()-t0:.1f}s\n")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    # Return structured results
    result_dict = {
        'basis': pyscf_basis,
        'method': pyscf_method,
        'xc': pyscf_xc,
        'atoms': all_results,
        'A_mean': np.mean([r['A'] for r in all_results]) if all_results else None,
        'beta_mean': np.mean([r['beta'] for r in all_results]) if all_results else None,
        'A_std': np.std([r['A'] for r in all_results]) if all_results else None,
        'beta_std': np.std([r['beta'] for r in all_results]) if all_results else None,
    }
    return result_dict
