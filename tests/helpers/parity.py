import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float))**2)))

def max_err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))

def correlation(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1])

def dir_cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

def plot_curves(x, curves, labels, title, xlabel, ylabel='Energy [eV]', savepath=None,
                pairs=None, diff_scale=100.0, show_rmse=True):
    """General-purpose curve plotting.

    Args:
        x: 1D array for x-axis.
        curves: list of 1D arrays to plot.
        labels: list of labels for each curve.
        title, xlabel, ylabel: plot annotations.
        savepath: if given, save figure here.
        pairs: list of (model_idx, ref_idx) tuples for comparison.
            When provided, each pair gets: ref as dotted, model shifted by DC offset
            to overlap ref, and diff (model-ref)*diff_scale on twin axis.
            When None, all curves plotted as solid lines (no comparison).
        diff_scale: multiplier for difference on twin axis (default 100x).
        show_rmse: if True and pairs given, annotate RMSE/MaxErr for each pair.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    curves = [np.asarray(c, dtype=float) for c in curves]
    if pairs is None:
        # Plain plot: all curves independent, solid lines
        for i, (c, label) in enumerate(zip(curves, labels)):
            ax.plot(x, c, ls='-', lw=0.5, color=f'C{i}', label=label)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        if savepath:
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return fig
    # Comparison plot: plot non-pair curves as solid, then pairs with ref/model/diff
    paired_idxs = set()
    for mi, ri in pairs:
        paired_idxs.add(mi); paired_idxs.add(ri)
    # Plot unpaired curves first as solid lines
    ci = 0
    for i, (c, label) in enumerate(zip(curves, labels)):
        if i not in paired_idxs:
            ax.plot(x, c, ls='-', lw=0.5, color=f'C{ci}', label=label)
            ci += 1
    # Plot each pair: ref dotted, model shifted, diff on twin axis
    for pi, (mi, ri) in enumerate(pairs):
        ref = curves[ri]
        model = curves[mi]
        dc_offset = float(np.mean(model - ref))
        model_shifted = model - dc_offset
        ax.plot(x, ref, ls=':', lw=1.5, color=f'C{ci}', label=f'{labels[ri]} (ref)')
        ci += 1
        ax.plot(x, model_shifted, ls='-', lw=0.5, color=f'C{ci}', label=f'{labels[mi]} (shifted)')
        ci += 1
        # Diff on twin axis
        ax2 = ax.twinx()
        diff = (model_shifted - ref) * diff_scale
        ax2.plot(x, diff, ls='-', lw=0.5, color='C3', alpha=0.7, label=f'diff {labels[mi]}-{labels[ri]} x{diff_scale:.0f}')
        ax2.set_ylabel(f'diff x{diff_scale:.0f}', color='C3')
        ax2.tick_params(axis='y', labelcolor='C3')
        ax2.legend(loc='lower right')
        if show_rmse:
            r, m = rmse(ref, model), max_err(ref, model)
            ax.text(0.02, 0.98 - pi * 0.08, f'{labels[mi]} vs {labels[ri]}: RMSE={r:.2e} Max={m:.2e}',
                    transform=ax.transAxes, va='top', fontsize=9, family='monospace',
                    bbox=dict(facecolor='white', alpha=0.8))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig

def overlay_plot(x, curves, labels, title, xlabel, ylabel='Energy [eV]', savepath=None, show_rmse=True):
    """Backward-compatible wrapper: treats curves[0] as reference, rest as models."""
    if len(curves) <= 1:
        return plot_curves(x, curves, labels, title, xlabel, ylabel, savepath, pairs=None, show_rmse=show_rmse)
    pairs = [(i, 0) for i in range(1, len(curves))]
    return plot_curves(x, curves, labels, title, xlabel, ylabel, savepath, pairs=pairs, show_rmse=show_rmse)

def assert_parity(ref, test, rtol=1e-3, atol=1e-5, name=''):
    r, m = rmse(ref, test), max_err(ref, test)
    assert r < rtol, f'{name}: RMSE={r:.2e} > {rtol:.0e}'
    assert m < atol * 10, f'{name}: MaxErr={m:.2e} > {atol*10:.0e}'

def parity_report(ref, test, name=''):
    return f'{name}: RMSE={rmse(ref,test):.2e}  Max={max_err(ref,test):.2e}  r={correlation(ref,test):.6f}'
