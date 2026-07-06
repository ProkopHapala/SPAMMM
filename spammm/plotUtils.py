"""
---
type: Module
title: plotUtils — Matplotlib Plotting Utilities
description: Pure-matplotlib 1D/2D plotting functions for scientific visualization (energies, forces, orbitals, densities, ESP, geometry). No PyQt dependency — usable from CLI scripts and GUI extensions alike.
tags: [plotting, matplotlib, visualization]
---

# plotUtils.py

Pure-matplotlib plotting utilities for SPAMMM. No Qt/GUI dependency.

## Contents

### 1D Plotting
- `plotEF` — energy/force vs coordinate
- `plot_scan_profile` — scan profile with optional min annotation
- `plot1d` / `plot1d_zip` — multi-curve 1D plots with analytical + numerical derivatives
- `plot_func` / `plot_funcs` — generic function visualization with derivatives
- `plot_compare_1d` — two-curve comparison (linear/log)
- `plot_density_comparison` / `plot_pauli_comparison` — specialized comparison wrappers

### 2D Scalar Fields
- `compute_grid_extent` — non-square grid origin/size from atomic positions (preserves aspect ratio)
- `make_2d_grid` — generate (x,y,z) grid points for 2D projection at given height
- `plot_2d_scalar` — heatmap with diverging/symmetric colormap, colorbar, atom overlay
- `overlay_atoms` — scatter atom positions with element colors + optional labels
- `plot_field_slice` / `plot_field_panel` — 2D slices of 3D fields at multiple z-heights
- `plot_cube_slice` — read and plot .cube file slices
- `plot_comparison_2d` — side-by-side orbital comparison (libwaveplot vs OpenCL)

### Molecular Geometry
- `plotAtoms` / `plotBonds` / `plotAngles` — individual element rendering
- `plotSystem` — full system plot (atoms, bonds, H-bonds, labels)
- `plotGeometry` — geometry with bonds, periodic replication, unit cell box
- `plotGeometryWithForces` — geometry + force arrows + scan path
- `plotTrj` — trajectory frame sequence to PNG

### Rendering
- `render_POVray` — POV-Ray scene export

## Usage

CLI (standalone):
```python
from spammm.plotUtils import plot_2d_scalar, compute_grid_extent, make_2d_grid
```

GUI (via re-export in `spammm/GUI/plotutils.py`):
```python
from spammm.GUI.plotutils import plot_2d_scalar, show_in_plot_window
```

## Related
- `spammm/GUI/plotutils.py` — Qt-specific wrapper (re-exports + `show_in_plot_window`)
- `spammm/GUI/AFMExtension.py` — AFM/STM plot functions using these utilities
- `spammm/GUI/QEqExtension.py` — ESP plotting using these utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import collections  as mc
from spammm import elements
#from . import utils as ut

#############################
#   Plotting 1D functions   #
#############################

def plotEF( xs, EFs, label='' ):
    plt.subplot(2,1,1); plt.plot( xs, EFs[:,0], label="E "+label ); plt.legend();plt.grid()
    plt.subplot(2,1,2); plt.plot( xs, EFs[:,1], label="F "+label ); plt.legend();plt.grid()

def plot_scan_profile(x, y, xlabel='x', ylabel='E', title=None, label=None, color='blue', marker='o', fname=None, ax=None, annotate_min=False, relative=False, dpi=150):
    x = np.asarray(x); y = np.asarray(y)
    if relative:
        y = y - np.nanmin(y)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    ax.plot(x, y, marker+'-', color=color, lw=2, ms=6, label=label)
    if label is not None:
        ax.legend()
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if annotate_min and np.any(np.isfinite(y)):
        i = np.nanargmin(y)
        ax.axvline(x[i], color=color, ls='--', alpha=0.5)
        ax.annotate(f'min\\n{x[i]:.4f}, {y[i]:.4f}', xy=(x[i], y[i]), xytext=(x[i], y[i]), color=color)
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return fig, ax

def numDeriv(x, y):
    """Numerical derivative using central difference"""
    dx = x[2:]-x[:-2]
    dy = y[2:]-y[:-2]
    x_ = x[1:-1]
    return -dy/dx, x_

# --- Modular Plotting Functions ---
def plot_with_deriv(ax1, ax2, x, y, y_deriv, label, color, linestyle='-'):
    ax1.plot(x, y,       label=label, color=color, linestyle=linestyle)
    ax2.plot(x, y_deriv, label=label, color=color, linestyle=linestyle)

def plot1d(x, ys, derivs=None, labels=None, colors=None, bNumDeriv=True, ls='-', lw=1.0, ax1=None, ax2=None, bGrid=True, bLegend=True, figsize=(6,9) ):
    """
    Plots one or more functions and their derivatives on given axes.
    
    Args:
        ax1: axis for function plot.
        ax2: axis for derivative plot.
        x: x values
        ys: list of y-value arrays (functions).
        derivs: list of analytical derivative arrays (optional).
        labels: list of legend labels.
        colors: list of plot colors.
        bNumDeriv: whether to show numerical derivative.
        linestyle: line style.
        linewidth: line width.
    """
    if ax1 is None:
        fig,(ax1,ax2) = plt.subplots(2,1, figsize=figsize)
    label=None
    c=None
    for i,y in enumerate(ys):
        if labels is not None: label = labels[i]
        if colors is not None: c = colors[i]
        ax1.plot(x, y, label=label, c=c, ls=ls, lw=lw)
        if bGrid: ax1.grid(alpha=0.2)
        if bLegend: ax1.legend()
    if derivs is not None:
        for i,dy in enumerate(derivs):
            if labels is not None: label = labels[i]
            if colors is not None: c = colors[i]
            ax2.plot(x, dy, label=f'{label} (analytical)',  c=c, ls=ls, lw=lw)
            if bNumDeriv:
                num_deriv, num_x = numDeriv(x, ys[i])
                ax2.plot(num_x, num_deriv, label=f'{label} (numerical)', c=c, ls=":", lw=2.0 )
            if bGrid: ax2.grid(alpha=0.2)
            if bLegend: ax2.legend()
    return fig, (ax1, ax2)

def plot_compare_1d(x1, y1, x2, y2, labels=["Ref", "New"], title="Comparison", xlabel="z [A]", ylabel="Value", log=True, ylim=None, xlim=None, fname=None):
    """Generic 1D comparison plot (linear or log)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x1, y1, 'b-', label=labels[0], lw=2)
    ax.plot(x2, y2, 'r--', label=labels[1], lw=2)
    if log:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim: ax.set_ylim(ylim)
    if xlim: ax.set_xlim(xlim)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
        plt.close(fig)
    return fig, ax

def plot_density_comparison(z1, rho1, z2, rho2, labels=["Fireball", "DFTB+"], title="Electron Density Comparison", fname=None):
    """Side-by-side density comparison (log scale)."""
    return plot_compare_1d(z1, rho1, z2, rho2, labels=labels, title=title, ylabel="Density [e/A^3]", log=True, ylim=[1e-6, 10], xlim=[0, 5], fname=fname)

def plot_pauli_comparison(z1, e1, z2, e2, labels=["Fireball", "DFTB+"], title="Pauli Potential Comparison", fname=None):
    """Side-by-side Pauli energy comparison (log scale)."""
    return plot_compare_1d(z1, e1, z2, e2, labels=labels, title=title, ylabel="Energy [eV]", log=True, ylim=[1e-6, 100], xlim=[0, 5], fname=fname)

def plot1d_zip(x, funcs):
    ys,dys,labels = [],[],[]
    for func in funcs:
        ys.append    (func[1][0])
        dys.append   (func[1][1])
        labels.append(func[0])
    return plot1d(x, ys, derivs=dys, labels=labels)

def plot_func(func, xs, params=None, axs=None, labels=('E','F','S'), colors=None, figsize=(6,9), **plot_kwargs):
    """Visualise a scalar function together with its derivatives.

    The *func* callable must support the signature::

        E, F, *rest = func(xs, **params)

    where *xs* is a 1-D `numpy.ndarray` and *E*, *F* (and optionally the second
    derivative *S*) are arrays of the same length.

    Parameters
    ----------
    func : callable
        Function returning at least *E* and *F*.
    xs : array-like
        Points at which to evaluate *func*.
    params : dict, optional
        Extra keyword arguments forwarded to *func*.
    axs : tuple(matplotlib.axes.Axes), optional
        Axes ``(axE, axF, axS)`` to plot on.  If *None* a new figure is
        created.
    labels : tuple(str), optional
        y-labels used for the three sub-plots.
    colors : tuple(str), optional
        Matplotlib colours.
    **plot_kwargs
        Further keyword arguments forwarded to ``Axes.plot``.

    Returns
    -------
    fig, axs : (matplotlib.figure.Figure, list(matplotlib.axes.Axes))
    """
    if params is None: params = {}
    xs  = np.asarray(xs)
    out = func(xs, **params)
    if len(out) < 2:  raise ValueError('func must return at least (E, F).')
    y,dy = out[:2]
    dyy  = out[2] if len(out) > 2 else None
    if axs is None:
        fig, (axY, axDY, axDYY) = plt.subplots(3, 1, sharex=True, figsize=figsize)
    else:
        axY, axDY, axDYY = axs
        fig = axY.figure
    if colors is None: colors = (None, None, None)
    axY                      .plot(xs, y,   color=colors[0], label=labels[0], **plot_kwargs); axY.legend()
    axDY                     .plot(xs, dy,  color=colors[1], label=labels[1], **plot_kwargs); axDY.legend()
    if dyy is not None: axDYY.plot(xs, dyy, color=colors[2], label=labels[2], **plot_kwargs); axDYY.legend()
    for ax, ylabel in zip((axY, axDY, axDYY), labels):
        ax.set_ylabel(ylabel)
        ax.grid(True)
    axDYY.set_xlabel('r')
    return fig, (axY, axDY, axDYY)


def plot_funcs(funcs, xs, bNum=False, nderivs=1, bError=False, errScale=100.0, figsize=(6,9), axs=None, titles=['Function', 'Derivative', 'Second Derivative'],   **plot_kwargs):
    """
    Plot multiple functions and their derivatives, broadcasting scalars.

    Parameters:
    -----------
    funcs : list of tuples (label, func, color, params)
        Each tuple defines a function to plot, its label, color, and parameters dict.
    xs : array-like
        Points at which to evaluate the functions.
    bNum : bool
        Whether to include numerical derivatives.
    figsize : tuple
        Figure size for the subplots.
    axs : list of matplotlib.axes.Axes, optional
        Existing axes to plot on; if None, new figure and axes are created.
    titles : list of str
        Titles for the subplots: [function, first derivative, second derivative].
    **plot_kwargs : dict
        Additional keyword arguments forwarded to Axes.plot.

    Returns:
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes
    """
    xs = np.asarray(xs)
    if axs is None:
        fig, axs = plt.subplots(1+nderivs, 1, sharex=True, figsize=figsize)
    else:
        fig = axs[0].figure
    for label, func, c, params, ref in funcs:
        out  = func(xs, **params)
        nout = len(out)
        # Primary output
        y = np.asarray(out[0])
        if y.ndim == 0 or y.shape != xs.shape: y = np.full(xs.shape, y)
        axs[0].plot(xs, y, color=c, label=label, lw=1.0, **plot_kwargs)
        if bError and ref is not None:
            axs[0].plot(xs, (y -ref[0])*errScale, color=c, label=f'{label} error*{errScale}', lw=1.0, ls='--', **plot_kwargs)
            
        # First derivative or force
        if nderivs > 0 and nout > 1:
            dy = np.asarray(out[1])
            if dy.ndim == 0 or dy.shape != xs.shape:
                dy = np.full(xs.shape, dy)
            axs[1].plot(xs, dy, color=c, label=label, lw=1.0, **plot_kwargs)
            if bNum:
                dy_num, num_x = numDeriv(xs, out[0])
                axs[1].plot(num_x, dy_num, label=f'{label} num', color=c, linestyle=':', linewidth=1.5, alpha=0.7)
        # Second derivative if present
        if nderivs > 1 and nout > 2 and out[2] is not None:
            d2 = np.asarray(out[2])
            if d2.ndim == 0 or d2.shape != xs.shape:
                d2 = np.full(xs.shape, d2)
            axs[2].plot(xs, d2, color=c, label=label, lw=1.0, **plot_kwargs)
            if bNum:
                d2_num, num_x2 = numDeriv(xs, out[1])
                axs[2].plot(num_x2, d2_num, label=f'{label} num', color=c, linestyle=':', linewidth=1.5, alpha=0.7)

    # Set titles, legends, and grid
    for idx, ax in enumerate(axs):
        ax.set_title(titles[idx])
        ax.legend()
        ax.grid(True)
    return fig, axs


#############################
#   Plotting 2D functions   #
#############################


def read_gnuplot_2d(fname):
    f = open(fname,'r')
    xs=[]
    ys=[]
    vals=[]
    nx=-1
    il=0
    for iil, l in enumerate(f):
        ws=l.split()
        if len(ws)<3:
            if(nx<0): 
                nx=int(il)
            print( iil, il )
            il=0
        else:
            xs  .append( float(ws[0]) )
            ys  .append( float(ws[1]) )
            vals.append( float(ws[2]) )
            il+=1
    xs   = np.array(xs)  .reshape(-1,nx)
    ys   = np.array(ys)  .reshape(-1,nx)
    vals = np.array(vals).reshape(-1,nx)
    return  vals, xs, ys

def read_dat( fname, ni=0, nf=1, iname=0, toRemove=None ):
    #format:            1  -96.294471702523595       -251.76919147019100       -48.443292828581697       # HHH-hhS1_NNO-hpS1 HHH-hhS1_NNO-hpS1 
    f=open( fname, 'r' )
    ints  =[]
    floats=[]
    names =[] 
    nc = ni+nf+2 # number of columns in the file
    for l in f:
        ws = l.split()
        nw = len(ws)
        if(nw<nc):
            ints_i    = [-1    ]*ni
            floats_i  = [np.nan]*nf
            name      = ws[nw-1]
        else:
            ints_i   = [ int(ws[i])   for i in range(0    ,ni           ) ]
            floats_i = [ float(ws[i]) for i in range(ni   ,ni+nf        ) ]
            name     = ws[ni+nf+1+iname]

        if toRemove is not None:
            if name in toRemove:
                continue

        ints      .append( ints_i   )
        floats    .append( floats_i )
        names     .append( name )
    return ints,floats,names

def _cube_slice_data(data, origin, spacing, plane='xy'):
    nx, ny, nz = data.shape
    if plane == 'xy':
        return data[:, :, nz//2], [origin[0], origin[0] + nx*spacing[0], origin[1], origin[1] + ny*spacing[1]], ('x (Å)', 'y (Å)'), (0, 1)
    if plane == 'xz':
        return data[:, ny//2, :], [origin[0], origin[0] + nx*spacing[0], origin[2], origin[2] + nz*spacing[2]], ('x (Å)', 'z (Å)'), (0, 2)
    if plane == 'yz':
        return data[nx//2, :, :], [origin[1], origin[1] + ny*spacing[1], origin[2], origin[2] + nz*spacing[2]], ('y (Å)', 'z (Å)'), (1, 2)
    raise ValueError(f"Invalid plane: {plane}")

def plot_cube_slice(cube_file, atoms=None, plane='xy', cmap='RdBu_r', title=None, fname=None, colorbar_label='value', dpi=150):
    from pyBall import dftb_utils as dftbu
    data, atoms_cube, origin, spacing = dftbu.read_cube_with_grid(cube_file)
    slice_data, extent, labels, axes = _cube_slice_data(data, origin, spacing, plane=plane)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(slice_data.T, origin='lower', cmap=cmap, aspect='auto', extent=extent)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    ax.set_title(title if title is not None else f'{cube_file} - {plane.upper()}')
    if atoms is not None:
        apos2d = atoms.positions[:, axes]
        for i, (x, y) in enumerate(apos2d):
            symbol = atoms.get_chemical_symbols()[i]
            ax.scatter(x, y, c='red', s=20, marker='+', zorder=10)
            ax.text(x, y, f'{symbol}{i}', ha='center', va='center', color='white', fontsize=8, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5), zorder=11)
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return fig, ax, data


#############################
#   Orbital comparison plotting   #
#############################

def plot_comparison_2d(wp_vals, ocl_vals, diff_vals, extent, title_prefix, plane_desc, 
                      method_tag, mo_indices, energies, homo, output_path, dpi=150, 
                      atom_coords=None):
    """
    Plot 2D orbital comparison: libwaveplot vs OpenCL vs difference.
    
    Args:
        wp_vals: (nstates, npoints, npoints) libwaveplot values
        ocl_vals: (nstates, npoints, npoints) OpenCL values
        diff_vals: (nstates, npoints, npoints) difference (wp - ocl)
        extent: [xmin, xmax, ymin, ymax] for imshow
        title_prefix: e.g., "H2O"
        plane_desc: e.g., "XY plane  z=0.000 Å"
        method_tag: e.g., "orb2points"
        mo_indices: list of MO indices
        energies: list of energies (eV)
        homo: HOMO index
        output_path: path to save PNG
        dpi: resolution
        atom_coords: (natoms, 3) atomic positions in Å (optional, for overlay)
    """
    import matplotlib.pyplot as plt
    nstates = len(mo_indices)
    s2 = wp_vals.shape[1:]
    ncols = 3
    fig, axes = plt.subplots(nstates, ncols, figsize=(5*ncols, 4*nstates))
    if nstates == 1: axes = axes[np.newaxis, :]
    
    # Determine which axes to plot from plane_desc
    if 'xy' in plane_desc.lower():
        x_idx, y_idx = 0, 1
    elif 'xz' in plane_desc.lower():
        x_idx, y_idx = 0, 2
    else:  # yz
        x_idx, y_idx = 1, 2
    
    for i in range(nstates):
        wp2  = wp_vals[i]
        oc2  = ocl_vals[i]
        df2  = diff_vals[i]
        clim = max(np.abs(wp2).max(), np.abs(oc2).max()) or 1e-12
        mo_idx = mo_indices[i]
        tag  = " [HOMO]" if mo_idx == homo else (" [LUMO]" if mo_idx == homo+1 else "")
        plane_title = f"{plane_desc}  [{method_tag}]"
        
        for ax, dat, ttl, cm, vl, vh in [
            (axes[i,0], wp2, f"libwaveplot MO{mo_idx}{tag}\nE={energies[i]:.2f}eV  {plane_title}", 'RdBu_r', -clim, clim),
            (axes[i,1], oc2, f"OpenCL MO{mo_idx}{tag}\n{plane_title}",                              'RdBu_r', -clim, clim),
            (axes[i,2], df2, f"diff (lib−OCL)\nRMS={np.sqrt(np.mean(diff_vals[i]**2)):.2e}", 'bwr', -clim*0.1, clim*0.1),
        ]:
            im = ax.imshow(dat, origin='lower', cmap=cm, vmin=vl, vmax=vh,
                           extent=extent, aspect='equal')
            ax.set_xlabel(extent[0] if extent[0] == extent[2] else 'x (Å)')
            ax.set_ylabel('y (Å)' if extent[0] == extent[2] else 'z (Å)')
            ax.set_title(ttl, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            # Overlay atomic positions if provided
            if atom_coords is not None:
                ax.scatter(atom_coords[:, x_idx], atom_coords[:, y_idx], 
                          c='black', marker='.', s=10, alpha=0.5, zorder=10)
    
    fig.suptitle(f"{title_prefix}: libwaveplot vs OpenCL  [{method_tag}]  {plane_desc}  (MO{mo_indices[0]}–{mo_indices[-1]})", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)


############################################
#   Ploting atoms and bonds ( Molecules )  #
############################################

def plotAtoms( apos=None, es=None, atoms=None, bNumbers=False, labels=None, sizes=100., colors='#808080', marker='o', axes=(0,1), selection=None ):
    ax1,ax2=axes
    if apos is None: apos = np.array([ a[1] for a in atoms ])  #;print(apos)
    if selection is not None: 
        apos                  = apos[selection,:]
        if es is not None: es = es  [selection]
    #print( "apos.shape ", apos.shape )
    #print( "apos ", apos )
    plt.scatter( apos[:,ax1],apos[:,ax2], marker=marker, c=colors, s=sizes, cmap='seismic', zorder=2 ); plt.axis('equal'); #plt.grid()
    bLabels = labels is not None
    if bNumbers or bLabels:
        na = len(apos)
        if not bLabels:
            labels = range(na)
        ax= plt.gca()
        for i in range(na):
            ax.annotate( str(labels[i]), (apos[i,ax1], apos[i,ax2]))

def plotBonds( lps=None, links=None, ps=None, lws=None, axes=(0,1), colors='k', labels=None, ls='solid', fnsz=10, fnclr='k', fontweight=None ):
    ax1,ax2=axes
    ax_inds=[ax1,ax2]
    if lps is None:
        ps_=ps
        if ps_.shape[1]!=2:
            ps_ = ps[:,ax_inds]
        links=np.array(links,dtype=np.int32)
        lps=np.zeros( (len(links),2,2) )
        #print( "lps.shape, ps_.shape ", lps.shape, ps_.shape )
        lps[:,0,:] = ps_[ links[:,0],: ]
        lps[:,1,:] = ps_[ links[:,1],: ]
    #lws = (kek.bondOrder-0.8)*5
    lc = mc.LineCollection(lps, linewidths=lws, colors=colors, linestyle=ls )
    ax= plt.gca()
    ax.add_collection(lc)

    if labels is not None:
        for i, s in enumerate(labels):
            p = (lps[i,0,:]+lps[i,1,:])*0.5
            ax.annotate( str(s), p, size=fnsz, color=fnclr, fontweight=fontweight )
    #ax.autoscale()
    #ax.margins(0.1)

def plotAngles( iangs, angs, ps, axes=(0,1), colors='k', labels=None, bPoly=True, alpha=0.2 ):
    ax1,ax2=axes
    ax_inds=[ax1,ax2]
    if isinstance(colors, str ):
        c = colors
        colors = [ c for i in range(len(iangs)) ]
    ax=plt.gca()
    for i,a in enumerate(iangs):
        iang=iangs[i]
        #print( ps.shape, iang, ax_inds ) 
        pp = ps[iang,:];
        pp = pp[:,ax_inds] ; #print(pp)
        t1 = plt.Polygon( pp, color=colors[i], fill=True, alpha=alpha, lw=0 )
        ax.add_patch(t1)
        p = (ps[iang[0],:] + ps[iang[1],:] + ps[iang[2],:])/3.0
        ax.annotate( "%3.0f˚" %(angs[i]*180.0/np.pi), p[ax_inds], color=colors[i] )


def plotGeometry(apos, atypes, lvs=None, bond_dist=1.8, bBondLabels=True,  replicate=(1,1,0), axes=(0,1), title=None, fname=None, figsize=(8,6),  bDrawBox=False):
    """Plot atomic geometry with bonds and bond length labels.
    
    Args:
        apos: (natoms, 3) array of atomic positions
        atypes: list of atomic types (element symbols or atomic numbers)
        lvs: (3,3) lattice vectors for periodic system (optional)
        bond_dist: maximum distance for bond detection
        bBondLabels: whether to show bond length labels
        replicate: tuple (nx, ny, nz) for periodic replication
        axes: which axes to plot (e.g., (0,1) for xy)
        title: plot title
        fname: output filename (if None, display but don't save)
        figsize: figure size
        bDrawBox: whether to draw unit cell outline
    """
    import matplotlib.pyplot as plt
    from itertools import combinations
    
    # Convert atomic numbers to symbols if needed
    ELEM_SYM = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
    if isinstance(atypes[0], (int, np.integer)):
        enames = [ELEM_SYM.get(int(aty), 'X') for aty in atypes]
    else:
        enames = atypes
    
    # Element colors and sizes
    elem_colors = {'H': 'gray', 'C': 'black', 'N': 'blue', 'O': 'red'}
    elem_sizes = {'H': 50, 'C': 100, 'N': 100, 'O': 100}
    
    # Replicate system if requested
    apos_rep = []
    enames_rep = []
    nx, ny, nz = replicate
    nx = max(nx, 1);  ny = max(ny, 1);  nz = max(nz, 1)  # at least 1 cell in each direction
    
    # Ensure apos is 2D
    if apos.ndim == 1:
        apos = apos.reshape(1, -1)
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = np.array([ix, iy, iz])
                if lvs is not None:
                    shift = shift @ lvs
                apos_rep.append(apos + shift)
                enames_rep.extend(enames)
    
    if len(apos_rep) > 0:
        apos_rep = np.vstack(apos_rep)
    else:
        apos_rep = apos
    
    # Plot atoms
    fig, ax = plt.subplots(figsize=figsize)
    
    for ename, pos in zip(enames_rep, apos_rep):
        color = elem_colors.get(ename, 'green')
        size = elem_sizes.get(ename, 50)
        ax.scatter(pos[axes[0]], pos[axes[1]], c=color, s=size, edgecolors='black', zorder=10)
        ax.text(pos[axes[0]], pos[axes[1]], ename, color='white', ha='center', va='center', 
                fontsize=8, zorder=11)
    
    # Detect bonds: ALL images within cutoff (not just best one)
    bonds = []  # list of (i, j, shift_j, dist) where shift_j is the periodic shift applied to atom j
    if lvs is not None:
        for i in range(len(apos)):
            for j in range(len(apos)):
                if j < i: continue  # avoid double-counting within same image
                for six in range(-1, 2):
                    for siy in range(-1, 2):
                        for siz in range(-1, 2):
                            if i == j and six == 0 and siy == 0 and siz == 0: continue  # skip self
                            shift = np.array([six, siy, siz]) @ lvs
                            dist = np.linalg.norm(apos[i] - (apos[j] + shift))
                            if dist < bond_dist:
                                bonds.append((i, j, shift, dist))
    else:
        for i, j in combinations(range(len(apos)), 2):
            dist = np.linalg.norm(apos[i] - apos[j])
            if dist < bond_dist:
                bonds.append((i, j, np.zeros(3), dist))
    
    # Plot bonds - replicate across all periodic cells
    label_set = set()  # track labels already drawn to avoid duplicates
    for i, j, shift, dist in bonds:
        pos_i = apos[i]
        pos_j = apos[j] + shift
        
        for cix in range(nx):
            for ciy in range(ny):
                for ciz in range(nz):
                    cell_shift = np.array([cix, ciy, ciz])
                    if lvs is not None:
                        cell_shift = cell_shift @ lvs
                    pos_i_rep = pos_i + cell_shift
                    pos_j_rep = pos_j + cell_shift
                    
                    ax.plot([pos_i_rep[axes[0]], pos_j_rep[axes[0]]], 
                            [pos_i_rep[axes[1]], pos_j_rep[axes[1]]], 
                            'k-', lw=2, zorder=5)
                    
                    if bBondLabels:
                        mid_x = (pos_i_rep[axes[0]] + pos_j_rep[axes[0]]) / 2
                        mid_y = (pos_i_rep[axes[1]] + pos_j_rep[axes[1]]) / 2
                        label_key = (round(mid_x, 2), round(mid_y, 2))
                        if label_key not in label_set:
                            label_set.add(label_key)
                            ax.text(mid_x, mid_y, f'{dist:.2f}', color='red', fontsize=8,
                                    ha='center', va='center', zorder=12)
    
    # Auto-set plot limits to include all atoms (must be done before box drawing)
    x_min = apos_rep[:, axes[0]].min()
    x_max = apos_rep[:, axes[0]].max()
    y_min = apos_rep[:, axes[1]].min()
    y_max = apos_rep[:, axes[1]].max()
    padding = 1.0
    
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_aspect('equal', adjustable='box')
    
    # Draw unit cell box AFTER limits are fixed; scalex/scaley=False prevents limit expansion
    if lvs is not None:
        # Draw box for each replicated cell
        for cix in range(nx):
            for ciy in range(ny):
                for ciz in range(nz):
                    cell_shift = np.array([cix, ciy, ciz]) @ lvs
                    corners = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,0]]) @ lvs + cell_shift
                    ax.plot(corners[:, axes[0]], corners[:, axes[1]], 'b--', lw=1.5, alpha=0.7,
                            scalex=False, scaley=False, solid_capstyle='round')
    if title:
        ax.set_title(title)
    ax.set_xlabel(f"{['x','y','z'][axes[0]]} (Å)")
    ax.set_ylabel(f"{['x','y','z'][axes[1]]} (Å)")
    ax.grid(True, alpha=0.3)
    
    if fname:
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plotGeometryWithForces(apos, atypes, lvs=None, forces=None, fixed_idx=None, highlight=None, scan_path=None, bond_dist=2.0, axes=(0,1), title=None, fname=None, figsize=(12,10), dpi=120):
    ax1, ax2 = axes
    if fixed_idx is None:
        fixed_idx = []
    if highlight is None:
        highlight = {}
    elem_colors = {'H': 'gray', 'C': 'black', 'N': 'blue', 'O': 'red'}
    elem_sizes  = {'H': 80,     'C': 120,     'N': 120,    'O': 120}
    fig, ax = plt.subplots(figsize=figsize)
    if scan_path is not None:
        i, j = scan_path
        p1 = apos[i]; p2 = apos[j]
        ax.plot([p1[ax1], p2[ax1]], [p1[ax2], p2[ax2]], 'g--', lw=3, alpha=0.5, zorder=1, label='Scan path')
    for i, (e, pos) in enumerate(zip(atypes, apos)):
        h = highlight.get(i, {})
        c  = h.get('color', elem_colors.get(e, 'green'))
        s  = h.get('size', elem_sizes.get(e, 80))
        ec = h.get('edgecolor', 'red' if i in fixed_idx else 'black')
        lw = h.get('linewidth', 3 if i in fixed_idx else 1)
        lab = h.get('label', f'{e}{i+1}')
        ax.scatter(pos[ax1], pos[ax2], c=c, s=s, edgecolors=ec, linewidths=lw, zorder=10)
        ax.text(pos[ax1], pos[ax2], lab, color='white', ha='center', va='center', fontsize=7, zorder=11)
    natoms = len(atypes)
    shifts = [np.zeros(3)]
    if lvs is not None:
        shifts = [np.array([six, siy, 0]) @ lvs for six in range(-1, 2) for siy in range(-1, 2)]
    for i in range(natoms):
        for j in range(i+1, natoms):
            for shift in shifts:
                d = np.linalg.norm(apos[i] - (apos[j] + shift))
                if d < bond_dist:
                    pi = apos[i]; pj = apos[j] + shift
                    ax.plot([pi[ax1], pj[ax1]], [pi[ax2], pj[ax2]], 'k-', lw=2, zorder=5)
                    mx, my = 0.5*(pi[ax1]+pj[ax1]), 0.5*(pi[ax2]+pj[ax2])
                    ax.text(mx, my, f'{d:.2f}', color='darkred', fontsize=7, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7), zorder=12)
    if forces is not None:
        fmax = np.max(np.linalg.norm(forces, axis=1))
        scale = 1.5 / max(fmax, 0.1)
        for pos, frc in zip(apos, forces):
            if np.linalg.norm(frc) < 0.005:
                continue
            ax.annotate('', xy=(pos[ax1] + frc[ax1]*scale, pos[ax2] + frc[ax2]*scale), xytext=(pos[ax1], pos[ax2]), arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        ax.scatter([], [], c='red', marker=r'$\rightarrow$', s=100, label=f'Force (max={fmax:.3f} eV/Å)')
    if lvs is not None:
        corners = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0]]) @ lvs
        ax.plot(corners[:, ax1], corners[:, ax2], 'b--', lw=1.5, alpha=0.6, label='Unit cell')
    ax.set_aspect('equal')
    ax.set_xlabel(f"{'xyz'[ax1]} (Å)"); ax.set_ylabel(f"{'xyz'[ax2]} (Å)")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved plot: {fname}")
    return fig, ax


def plotSystem( sys , bBonds=True, colors=None, sizes=None, extent=None, sz=50., RvdwCut=0.5, axes=(0,1), bLabels=True, labels=None, _0=1, HBs=None, bHBlabels=True, bBLabels=False,bAtoms=True  ):    
    if( bBonds ):
        if sys.bonds is None:
            bs, rbs = sys.findBonds( Rcut=3.0, RvdwCut=RvdwCut )
        rb_labs = None
        if bBLabels:
            rb_labs = [ ("%3.2f" %r) for r in rbs ]
        plotBonds( links=sys.bonds, ps=sys.apos, axes=axes )
    
    enames = [ typ.split('_')[0] for typ in sys.enames ]
    if(colors is None): colors = [ elements.ELEMENT_DICT[e][8]    for e in enames ]
    if(sizes  is None): sizes  = [ elements.ELEMENT_DICT[e][6]*sz for e in enames ]
    if((labels is None) and bLabels): labels=[ "%s%i" %(e,i+_0) for i,e in enumerate(sys.enames) ]
    #print( "labels ", labels)
    if bAtoms:
        plotAtoms( apos=sys.apos, es=sys.enames, sizes=sizes, colors=colors, marker='o', axes=axes, labels=labels )

    # H-Bonds
    if HBs is not None:
        if len(HBs)>0:
            hbs,rhbs = HBs
            rh_labs = None
            if bHBlabels:
                rh_labs = [ ("%3.2f" %r) for r in rhbs ]
            plotBonds ( ps=sys.apos, links=hbs, axes=axes, colors="g", labels=rh_labs )

    if extent is not None:
        plt.xlim(extent[0],extent[1])
        plt.ylim(extent[2],extent[3]) 

def plotTrj( trj, bBonds=True, sz=50., numbers=None, axes=(0,1), extent=None, prefix="mol_", RvdwCut=0.5, figsize=(5,5) ):
    if numbers is None: numbers=range(len(trj))
    for i,sys in enumerate(trj):
        print("plot # ", i)
        fig = plt.figure(figsize=figsize)
        
        if( bBonds ):
            if sys.bonds is None:
                sys.findBonds( Rcut=3.0, RvdwCut=RvdwCut )
                plotBonds( links=sys.bonds, ps=sys.apos, axes=(0,1) )
        
        colors = [ elements.ELEMENT_DICT[e][8]    for e in sys.enames ]
        sizes  = [ elements.ELEMENT_DICT[e][6]*sz for e in sys.enames ]
        plotAtoms( apos=sys.apos, es=sys.enames, sizes=sizes, colors=colors, marker='o', axes=axes )

        if extent is not None:
            plt.xlim(extent[0],extent[1])
            plt.ylim(extent[2],extent[3])
        
        plt.savefig( prefix+("%03i.png" %numbers[i]), bbox_inches='tight' )
        plt.close(fig)

def plot_field_slice(ax, field, origin, step, z, cmap='magma', title='', sym=False):
    """Plot 2D XY slice of 3D field at given z-height with colorbar.

    Args:
        ax: matplotlib Axes
        field: (nx, ny, nz) 3D array
        origin: (3,) grid origin
        step: grid spacing in Angstrom
        z: z-height for slice in Angstrom
        cmap: colormap name
        title: subplot title
        sym: if True, use symmetric TwoSlopeNorm around zero

    Returns:
        iz: integer z-index used
    """
    from matplotlib.colors import TwoSlopeNorm
    nx, ny, nz = field.shape
    iz = int(np.clip(np.round((z - origin[2]) / step), 0, nz-1))
    slice_xy = field[:, :, iz].T
    extent_xy = [float(origin[0]), float(origin[0]) + (nx-1)*step,
                 float(origin[1]), float(origin[1]) + (ny-1)*step]
    if sym:
        vabs = max(abs(float(slice_xy.min())), abs(float(slice_xy.max())), 1e-12)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    else:
        norm = None
    im = ax.imshow(slice_xy, origin='lower', cmap=cmap, norm=norm, extent=extent_xy, aspect='equal')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.6)
    return iz

def plot_field_panel(axes, field, origin, step, z_heights, cmap='magma', sym=False, title_prefix=''):
    """Plot 2D XY slices of 3D field at multiple z-heights across axes.

    Args:
        axes: iterable of matplotlib Axes (one per z)
        field: (nx, ny, nz) 3D array
        origin: (3,) grid origin
        step: grid spacing
        z_heights: list of z values for slices
        cmap: colormap name
        sym: use symmetric norm
        title_prefix: prefix for subplot titles
    """
    for ax, z in zip(axes, z_heights):
        plot_field_slice(ax, field, origin, step, z, cmap=cmap, sym=sym,
                        title=f'{title_prefix} z={z:.1f}A')

def render_POVray(
    sys, filename, 
    # Atom and bond parameters
    atom_scale=1.0, bond_width=0.2,
    # Camera parameters
    look_at=(0.0, 0.0, 0.0),
    camera_pos  =( 0.0, 0.0, 100.0 ),
    camera_up   =( 0.0, 1.0,   0.0 ),
    camera_right=( 1.0, 0.0,   0.0 ),
    sky=(0.0, 0.0, 1.0),
    zoom=30.0,
    orthographic=True,
    # Image parameters
    width=400, height=400,
    # Lighting parameters
    light_pos       =(10.0, 20.0, 30.0),
    light_color     =(2.5, 2.5, 2.5),
    ambient_light   =(1.0, 1.0, 1.0),
    background_color=(1.0, 1.0, 1.0),
    # Material parameters
    ambient=0.5, diffuse=0.6, specular=0.4, roughness=0.01, metallic=-1.0,  phong=10.0, phong_size=120,
    # Rendering options
    shadows=False,
    bond_clr=(0.5,0.5,0.5),
    z_color_shift=False,
    viewAxis=False,
    ):
    """Export system to POV-Ray file with customizable rendering settings
    
    Args:
        sys: System object with atoms/bonds
        filename: Output .pov file
        
        # Atom and bond parameters
        bond_scale: Bond length relative to sum of atomic radii
        atom_scale: Scale factor for atomic radii
        bond_width: Visual width of bonds
        
        # Camera parameters
        camera_pos: Camera position (x,y,z)
        look_at: Point camera looks at (x,y,z)
        sky: Up vector for camera orientation
        zoom: Camera zoom factor
        orthographic: Use orthographic projection if True, perspective if False
        
        # Image parameters
        width, height: Output image dimensions
        
        # Lighting parameters
        light_pos: Position of main light source
        light_color: RGB color of main light
        ambient_light: RGB color of ambient light
        background_color: RGB color of background
        
        # Material parameters
        ambient: Ambient light reflection
        diffuse: Diffuse light reflection
        specular: Specular highlights intensity
        roughness: Surface roughness
        metallic: Metallic finish if True
        phong, phong_size: Phong shading parameters
        
        # Rendering options
        shadows: Enable shadows if True
        z_color_shift: Enable z-dependent color shifting if True
    """

    def makeFinishString( ambient, diffuse, specular, roughness, phong, phong_size, metallic ):
        return f"""
  finish {{
    ambient    {ambient}
    diffuse    {diffuse}
    specular   {specular}
    roughness  {roughness}
    phong      {phong}
    phong_size {phong_size}
    { f"metallic {metallic}" if metallic>0 else ""}
  }}
  """

    with open(filename, 'w') as pov:
        # Write POV header with customizable parameters
        pov.write(
f'''// ***********************************************
// Camera & other global settings
// ***********************************************

#declare Zoom = {zoom};
#declare Width = {width};
#declare Height = {height};

camera{{
  {"orthographic" if orthographic else ""}
  look_at  <{look_at[0]  }    , {look_at[1]  }    , {look_at[2]} >
  location <{camera_pos[0]}   , {camera_pos[1]}   , {camera_pos[2]}>
  up       <{camera_up[0]*zoom}    , {camera_up[1]*zoom}    , {camera_up[2]*zoom} >
  right    <{camera_right[0]*zoom} , {camera_right[1]*zoom} , {camera_right[2]*zoom}>
  //sky      <{sky[0]}       , {sky[1]}        , {sky[2]} >
  sky      <{camera_up[0]}       , {camera_up[1]}        , {camera_up[2]} >
}}

background      {{ color rgb <{background_color[0]}, {background_color[1]}, {background_color[2]}> }}
light_source    {{ < {light_pos[0]}, {light_pos[1]}, {light_pos[2]}>  rgb <{light_color[0]}, {light_color[1]}, {light_color[2]}> }}
global_settings {{ ambient_light rgb< {ambient_light[0]}, {ambient_light[1]}, {ambient_light[2]}> }}

// ===== macros for common shapes

#macro myFinish()
{makeFinishString( ambient, diffuse, specular, roughness, phong, phong_size, metallic )}
#end

#macro a(X,Y,Z,RADIUS,R,G,B,T)
 sphere{{<X,Y,Z>,RADIUS
  pigment{{rgbt<R,G,B,T>}}
  myFinish()
  {"no_shadow" if not shadows else ""}
 }}
#end

#macro b(X1,Y1,Z1,RADIUS1,X2,Y2,Z2,RADIUS2,R,G,B,T)
 cone{{<X1,Y1,Z1>,RADIUS1,<X2,Y2,Z2>,RADIUS2
  pigment{{rgbt<R,G,B,T>}}
  myFinish()
  {"no_shadow" if not shadows else ""}
 }}
#end

{
f"""
 b(    0.0,    0.0,   0.0,    {bond_width*1.5},   1.0,0.0,0.0,    {bond_width},    1.0,0.0,0.0, 0.0 )  // x-axis
 b(    0.0,    0.0,   0.0,    {bond_width*1.5},   0.0,1.0,0.0,    {bond_width},    0.0,1.0,0.0, 0.0 )  // y-axis
 b(    0.0,    0.0,   0.0,    {bond_width*1.5},   0.0,0.0,1.0,    {bond_width},    0.0,0.0,1.0, 0.0 )  // z-axis
 """ if viewAxis else ""
}

''')

        #return
        # Write atoms
        pov.write('// ------ Atoms\n')
        for i in range(len(sys.apos)):
            e = sys.enames[i]  # Elements are 1-based
            e = e.split('_')[0]
            x, y, z = sys.apos[i]
            rad = elements.ELEMENT_DICT[e][6] * atom_scale
            clr = elements.getColor( e , bFloat=True)            
            pov.write('a( {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, 0.0 )\n'.format( x, y, z, rad, clr[0], clr[1], clr[2] ))

        # Write bonds
        if sys.bonds is not None:
            pov.write('\n// ------ Bonds\n')
            for bond in sys.bonds:
                i, j = bond[:2]
                # Get positions
                pos1 = sys.apos[i]
                pos2 = sys.apos[j]                
                pov.write('b( {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, 0.0 )\n'.format( 
                              pos1[0], pos1[1], pos1[2], bond_width, 
                              pos2[0], pos2[1], pos2[2], bond_width, 
                              bond_clr[0],  bond_clr[1], bond_clr[2]  ))


#############################################
#   Density comparison plotting (1D + 2D)   #
#############################################

def plot_density_z_profile(ax, z_vals, profiles, labels=None, colors=None, lws=None, lss=None,
                           ylim=(1e-3, 10), xlim=(0, 3), log=True, xlabel='z above plane (A)',
                           ylabel='rho (e/A^3)', title=None, legend=True, fontsize=8):
    """Plot 1D density profile(s) above a point on a log-scale axis.

    Args:
        ax: matplotlib Axes (created if None)
        z_vals: (n_z,) z coordinates
        profiles: list of (n_z,) arrays, or single (n_z,) array
        labels: list of labels for each profile
        colors: list of colors for each profile
        lws: list of linewidths
        lss: list of linestyle strings
        ylim, xlim: axis limits
        log: use log y-scale
        legend: show legend
        fontsize: legend font size

    Returns:
        ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    if not isinstance(profiles, (list, tuple)):
        profiles = [profiles]
    n = len(profiles)
    if labels is None: labels = [None]*n
    if colors is None: colors = [None]*n
    if lws is None: lws = [0.5]*n
    if lss is None: lss = ['-']*n
    for i, prof in enumerate(profiles):
        prof = np.maximum(np.asarray(prof), 1e-10)
        ax.plot(z_vals, prof, color=colors[i], linestyle=lss[i], linewidth=lws[i], label=labels[i])
    if log: ax.set_yscale('log')
    ax.set_ylim(*ylim); ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title, fontsize=10)
    if legend and any(l is not None for l in labels): ax.legend(fontsize=fontsize)
    return ax


def plot_density_z_fit(ax, z_vals, rho, fit_lo=0.5, fit_hi=1.5, color='k', label=None, lw=0.5):
    """Add log-linear fit line + decay constant annotation to an existing axis.

    Fits log(rho) = a*z + b in the [fit_lo, fit_hi] region and plots the fit line.

    Args:
        ax: matplotlib Axes
        z_vals: (n_z,) z coordinates
        rho: (n_z,) density values
        fit_lo, fit_hi: fit region bounds
        color: fit line color
        label: prefix for fit label (e.g. 'GPAW')
        lw: linewidth

    Returns:
        a: fitted slope (decay constant), or None if insufficient points
    """
    z = np.asarray(z_vals); r = np.maximum(np.asarray(rho), 1e-10)
    mask = (z >= fit_lo) & (z <= fit_hi) & (r > 1e-10)
    if mask.sum() < 3: return None
    a, b = np.polyfit(z[mask], np.log(r[mask]), 1)
    fit_z = np.linspace(fit_lo, fit_hi, 50)
    fit_label = f'{label} fit: {a:.2f}/A' if label else f'fit: {a:.2f}/A'
    ax.plot(fit_z, np.exp(a * fit_z + b), color + '--', linewidth=lw, alpha=0.7, label=fit_label)
    return a


def plot_density_2d_slice(ax, rho_2d, extent, atoms=None, cmap='hot_r', log=True,
                          vmin=1e-3, vmax=10, title=None, xlabel='x (A)', ylabel='y (A)',
                          colorbar=True, atom_colors=None, atom_size=8):
    """Plot 2D density slice with optional atom overlay.

    Args:
        ax: matplotlib Axes
        rho_2d: (nx, ny) 2D density slice (will be transposed for imshow)
        extent: [xmin, xmax, ymin, ymax]
        atoms: list of (sym, x, y) tuples for overlay markers, or None
        cmap: colormap name
        log: use LogNorm
        vmin, vmax: color scale bounds
        title: subplot title
        atom_colors: dict {sym: color} or None (default: cyan for non-H, lime for H)
        atom_size: marker size

    Returns:
        im: the imshow image object
    """
    from matplotlib.colors import LogNorm
    data = np.maximum(rho_2d.T, vmin)
    norm = LogNorm(vmin=vmin, vmax=vmax) if log else None
    im = ax.imshow(data, origin='lower', extent=extent, cmap=cmap, aspect='equal', norm=norm)
    if title: ax.set_title(title)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if atoms:
        if atom_colors is None: atom_colors = {}
        for sym, x, y in atoms:
            c = atom_colors.get(sym, 'cyan' if sym != 'H' else 'lime')
            ax.plot(x, y, '.', color=c, markersize=atom_size)
    if colorbar: plt.colorbar(im, ax=ax, label='rho (e/A^3)')
    return im


def plot_density_per_element(atoms, z_vals, methods, elem_order=None,
                             fit_lo=0.5, fit_hi=1.5, ylim=(1e-3, 10), xlim=(0, 3),
                             suptitle=None, figsize=None, fname=None, dpi=150):
    """Create per-element subplot grid with multiple methods overlaid.

    Args:
        atoms: list of (idx, sym, x, y, z) tuples
        z_vals: (n_z,) z coordinates
        methods: list of dicts: {'name': str, 'color': str, 'profiles': (n_atoms, n_z) array, 'ls': str, 'fit': bool}
        elem_order: list of element symbols in display order (default: O,N,C,H)
        fit_lo, fit_hi: fit region for decay constant annotation
        ylim, xlim: axis limits
        suptitle: figure title
        figsize: (width, height) per subplot
        fname: save path (if None, returns fig)
        dpi: save resolution

    Returns:
        fig
    """
    if elem_order is None: elem_order = ['O', 'N', 'C', 'H']
    seen = {}
    for i, (idx, sym, x, y, z) in enumerate(atoms):
        if sym not in seen: seen[sym] = i
    elem_list = [e for e in elem_order if e in seen]
    n = len(elem_list)
    if figsize is None: figsize = (5, 4)
    fig, axc = plt.subplots(1, n, figsize=(figsize[0]*n, figsize[1]), squeeze=False)
    if suptitle: fig.suptitle(suptitle, fontsize=14)

    for ei, sym in enumerate(elem_list):
        ai = seen[sym]
        ax = axc[0, ei]
        for m in methods:
            prof = m['profiles'][ai]
            plot_density_z_profile(ax, z_vals, prof, labels=[m['name']], colors=[m.get('color')],
                                   lws=[m.get('lw', 0.5)], lss=[m.get('ls', '-')],
                                   ylim=ylim, xlim=xlim, title=f"{sym} (atom {atoms[ai][0]})")
            if m.get('fit', False):
                plot_density_z_fit(ax, z_vals, prof, fit_lo, fit_hi, color=m.get('color'), label=m['name'])
        ax.legend(fontsize=7)

    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=dpi); plt.close(fig)
    return fig


def plot_density_methods_panel(atoms, z_vals, methods, ylim=(1e-3, 10), xlim=(0, 3),
                               suptitle=None, fname=None, dpi=150):
    """Side-by-side panels, one per method, all atoms overlaid.

    Args:
        atoms: list of (idx, sym, x, y, z) tuples
        z_vals: (n_z,) z coordinates
        methods: list of dicts: {'name': str, 'profiles': (n_atoms, n_z) array}
        ylim, xlim: axis limits
        suptitle: figure title
        fname: save path
        dpi: save resolution

    Returns:
        fig
    """
    from spammm.elements import getColor
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5), squeeze=False)
    if suptitle: fig.suptitle(suptitle, fontsize=14)

    for side, m in enumerate(methods):
        ax = axes[0, side]
        for ai, (idx, sym, x, y, z) in enumerate(atoms):
            try: color = getColor(sym, bFloat=False)
            except: color = f"C{ai}"
            ls = '-' if sym != 'H' else '--'
            ax.plot(z_vals, np.maximum(m['profiles'][ai], 1e-10), color=color, ls=ls, lw=1.0, alpha=0.8, label=f"{idx}:{sym}")
        ax.set_ylim(*ylim); ax.set_xlim(*xlim); ax.set_yscale('log')
        ax.set_title(m['name'], fontsize=12)
        ax.set_xlabel("z above plane (A)"); ax.set_ylabel("rho (e/A^3)")
        ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=dpi); plt.close(fig)
    return fig


def plot_2d_density_panel(methods_2d, suptitle=None, fname=None, dpi=150):
    """Side-by-side 2D density slices from multiple methods.

    Args:
        methods_2d: list of dicts: {'name', 'slice_2d', 'extent', 'atoms': [(sym,x,y),...]}
        suptitle: figure title
        fname: save path
        dpi: save resolution

    Returns:
        fig
    """
    n = len(methods_2d)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6), squeeze=False)
    if suptitle: fig.suptitle(suptitle, fontsize=14)

    for i, m in enumerate(methods_2d):
        ax = axes[0, i]
        plot_density_2d_slice(ax, m['slice_2d'], m['extent'], atoms=m.get('atoms'), title=m['name'])

    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=dpi); plt.close(fig)
    return fig


def plot_sa_history(history, title='SA convergence', fname=None, dpi=150):
    """Plot simulated annealing convergence history.

    Args:
        history: list of (iteration, current_obj, best_obj) tuples
        title: plot title
        fname: save path
        dpi: save resolution

    Returns:
        fig
    """
    hist = np.array(history)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(hist[:,0], hist[:,1], 'b-', lw=0.3, alpha=0.5, label='current')
    ax.plot(hist[:,0], hist[:,2], 'r-', lw=0.5, label='best')
    ax.set_xlabel('iteration'); ax.set_ylabel('objective'); ax.set_yscale('log')
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=dpi); plt.close(fig)
    return fig


def plot_density_multi_z(methods, z_heights, suptitle=None, fname=None, dpi=150,
                         cmap='hot', log=False, figsize_per_panel=(5, 4), margin=2.0):
    """Grid of 2D density slices: rows = z-heights, cols = methods.

    All panels show the same spatial extent (centered on first method's atoms, +margin).
    Each panel has its own linear color scale (vmin=0, vmax from that slice's data).
    z_heights are relative to molecular plane; each method must provide 'z0' for absolute z.

    Args:
        methods: list of dicts: {'name', 'rho_3d', 'origin', 'step', 'atoms': [(sym,x,y),...], 'z0': float}
        z_heights: list of relative z values in Angstrom (above molecular plane)
        suptitle: figure title
        fname: save path
        dpi: save resolution
        cmap: colormap
        log: use LogNorm (default False = linear)
        figsize_per_panel: (w, h) per subplot
        margin: extra Angstroms around atom bounds for the common extent

    Returns:
        fig
    """
    from matplotlib.colors import LogNorm
    nz = len(z_heights); nm = len(methods)
    fig, axes = plt.subplots(nz, nm, figsize=(figsize_per_panel[0]*nm, figsize_per_panel[1]*nz), squeeze=False)
    if suptitle: fig.suptitle(suptitle, fontsize=16)

    # Compute non-square half-widths from first method's atoms
    atoms0 = methods[0].get('atoms', [])
    if atoms0:
        xs0 = [x for _, x, _ in atoms0]; ys0 = [y for _, _, y in atoms0]
        half_x = (max(xs0)-min(xs0))/2 + margin
        half_y = (max(ys0)-min(ys0))/2 + margin
    else:
        half_x = half_y = 10.0

    # Shrink half_x/half_y so view fits within every method's grid (avoid white areas)
    for m in methods:
        step_m = np.atleast_1d(m['step'])
        if len(step_m) == 1: step_m = np.array([step_m[0]]*3)
        origin_m = m['origin']; rho_m = m['rho_3d']
        nx_m, ny_m = rho_m.shape[0], rho_m.shape[1]
        atoms_m = m.get('atoms', [])
        if atoms_m:
            xs_m = [x for _, x, _ in atoms_m]; ys_m = [y for _, _, y in atoms_m]
            mcx_m = (min(xs_m)+max(xs_m))/2; mcy_m = (min(ys_m)+max(ys_m))/2
        else:
            mcx_m = origin_m[0]+nx_m*step_m[0]/2; mcy_m = origin_m[1]+ny_m*step_m[1]/2
        grid_x0, grid_x1 = origin_m[0], origin_m[0]+(nx_m-1)*step_m[0]
        grid_y0, grid_y1 = origin_m[1], origin_m[1]+(ny_m-1)*step_m[1]
        half_x = min(half_x, mcx_m - grid_x0, grid_x1 - mcx_m)
        half_y = min(half_y, mcy_m - grid_y0, grid_y1 - mcy_m)

    for col, m in enumerate(methods):
        rho = m['rho_3d']; origin = m['origin']; step = m['step']
        step = np.atleast_1d(step)
        if len(step) == 1: step = np.array([step[0]]*3)
        nx, ny, nz_grid = rho.shape
        atoms_2d = m.get('atoms')
        z0 = m.get('z0', origin[2])
        if atoms_2d:
            xs = [x for _, x, _ in atoms_2d]; ys = [y for _, _, y in atoms_2d]
            mcx, mcy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
        else:
            mcx, mcy = origin[0]+nx*step[0]/2, origin[1]+ny*step[1]/2
        for row, z_rel in enumerate(z_heights):
            ax = axes[row, col]
            z_abs = z0 + z_rel
            iz = int(np.clip(np.round((z_abs - origin[2]) / step[2]), 0, nz_grid-1))
            slice_2d = rho[:, :, iz].T
            vmax = float(np.max(slice_2d))
            full_extent = [origin[0], origin[0]+(nx-1)*step[0], origin[1], origin[1]+(ny-1)*step[1]]
            if log:
                vmin = max(float(np.min(slice_2d)), 1e-10)
                im = ax.imshow(slice_2d, origin='lower', extent=full_extent, cmap=cmap, aspect='equal',
                              norm=LogNorm(vmin=vmin, vmax=vmax))
            else:
                im = ax.imshow(slice_2d, origin='lower', extent=full_extent, cmap=cmap, aspect='equal',
                              vmin=0.0, vmax=vmax)
            ax.set_xlim(mcx-half_x, mcx+half_x)
            ax.set_ylim(mcy-half_y, mcy+half_y)
            if atoms_2d:
                for sym, x, y in atoms_2d:
                    c = 'cyan' if sym != 'H' else 'lime'
                    ax.plot(x, y, '.', color=c, markersize=2)
            if row == 0: ax.set_title(m['name'], fontsize=12)
            if col == 0: ax.set_ylabel(f'z={z_rel:.1f} A', fontsize=11)
            ax.set_xlabel('x (A)')
            plt.colorbar(im, ax=ax, shrink=0.7, label='rho (e/A^3)')

    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=dpi, bbox_inches='tight'); plt.close(fig)
    return fig


#############################
#   2D scalar field plotting (shared with GUI)   #
#############################

ELEM_COLOR_2D = {'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'S': 'yellow', 'F': 'green', 'Cl': 'green', 'Br': 'brown', 'I': 'purple'}

def compute_grid_extent(apos, padding_factor=0.15, default_size=14.0):
    """Compute grid origin, sizes (per-axis), and center_z from atomic positions.

    Preserves the molecule's aspect ratio (non-square grid).

    Returns:
        grid_origin: (2,) array — (x_min, y_min) of grid
        size_xy: (2,) array — (width, height) of grid
        center_z: float — mean z of atoms
    """
    if len(apos) == 0:
        raise ValueError("Cannot compute grid extent: no atoms.")
    apos_2d = apos[:, :2]
    min_pos = apos_2d.min(axis=0)
    max_pos = apos_2d.max(axis=0)
    center_z = apos[:, 2].mean()
    padding = (max_pos - min_pos) * padding_factor
    grid_origin = min_pos - padding
    size_xy = max_pos - min_pos + 2 * padding
    return grid_origin, size_xy, center_z

def make_2d_grid(grid_origin, size_xy, center_z, z_height, n=200):
    """Generate 2D grid points for projection (non-square, fits molecule).

    Returns:
        points: (n_total, 3) array of (x, y, z) grid points
        extent: [xmin, xmax, ymin, ymax]
        nx, ny: grid dimensions
    """
    w, h = size_xy[0], size_xy[1]
    if w >= h:
        nx = n
        ny = max(1, int(round(n * h / w)))
    else:
        ny = n
        nx = max(1, int(round(n * w / h)))
    xs = np.linspace(grid_origin[0], grid_origin[0] + w, nx)
    ys = np.linspace(grid_origin[1], grid_origin[1] + h, ny)
    X, Y = np.meshgrid(xs, ys)  # indexing='xy': X.shape = (ny, nx)
    Z = np.full_like(X, center_z + z_height)
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    return points, extent, nx, ny

def overlay_atoms(ax, apos, enames, xs=None, ys=None, label_heavy=True):
    """Overlay atom positions on a matplotlib axes.

    Args:
        ax: matplotlib axes
        apos: (n,3) atomic positions
        enames: list of element name strings
        xs, ys: optional grid axes for bounds checking
        label_heavy: if True, annotate non-H atoms with element+index
    """
    for i, (p, e) in enumerate(zip(apos, enames)):
        x, y = p[0], p[1]
        if xs is not None and ys is not None:
            if not (xs[0] <= x <= xs[-1] and ys[0] <= y <= ys[-1]):
                continue
        c = ELEM_COLOR_2D.get(e, 'magenta')
        ax.plot(x, y, 'o', color=c, markersize=5, markeredgecolor='k', markeredgewidth=0.5, zorder=5)
        if label_heavy and e != 'H':
            ax.annotate(f"{e}{i}", (x, y), fontsize=7, ha='left', va='bottom', color='black')

def plot_2d_scalar(data_2d, extent, title, z_label='', cmap='seismic', symmetric=True,
                   apos=None, enames=None, label_heavy=True):
    """Create a matplotlib Figure with a 2D scalar field heatmap.

    Args:
        data_2d: 2D array (ny, nx) where data[i,j] corresponds to (xs[j], ys[i])
        extent: [xmin, xmax, ymin, ymax]
        title: plot title
        z_label: colorbar label
        cmap: colormap name
        symmetric: if True, vmin=-vmax (diverging colormap)
        apos: optional (n,3) atom positions for overlay
        enames: optional element names for overlay coloring
        label_heavy: if True, annotate non-H atoms

    Returns:
        matplotlib Figure
    """
    from matplotlib.figure import Figure
    data = np.asarray(data_2d, dtype=np.float64)
    if symmetric:
        vmax = max(abs(np.min(data)), abs(np.max(data)))
        if vmax < 1e-30: vmax = 1.0
        vmin = -vmax
    else:
        vmin, vmax = np.min(data), np.max(data)
        if vmax - vmin < 1e-30: vmax = vmin + 1.0
    fig = Figure(figsize=(7, 6), dpi=100)
    ax = fig.add_subplot(111)
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, aspect='equal')
    ax.set_title(title)
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    fig.colorbar(im, ax=ax, label=z_label)
    if apos is not None and enames is not None:
        xs = np.linspace(extent[0], extent[1], data.shape[1])
        ys = np.linspace(extent[2], extent[3], data.shape[0])
        overlay_atoms(ax, apos, enames, xs=xs, ys=ys, label_heavy=label_heavy)
    fig.tight_layout()
    return fig

