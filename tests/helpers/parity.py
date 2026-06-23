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

def overlay_plot(x, curves, labels, title, xlabel, ylabel='Energy [eV]', savepath=None, show_rmse=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (c, l) in enumerate(zip(curves, labels)):
        ax.plot(x, c, '--' if i > 0 else '-', label=l)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    if show_rmse and len(curves) >= 2:
        r, m = rmse(curves[0], curves[1]), max_err(curves[0], curves[1])
        ax.text(0.02, 0.98, f'RMSE={r:.2e}\nMax={m:.2e}',
                transform=ax.transAxes, va='top', fontsize=9, family='monospace',
                bbox=dict(facecolor='white', alpha=0.8))
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig

def assert_parity(ref, test, rtol=1e-3, atol=1e-5, name=''):
    r, m = rmse(ref, test), max_err(ref, test)
    assert r < rtol, f'{name}: RMSE={r:.2e} > {rtol:.0e}'
    assert m < atol * 10, f'{name}: MaxErr={m:.2e} > {atol*10:.0e}'

def parity_report(ref, test, name=''):
    return f'{name}: RMSE={rmse(ref,test):.2e}  Max={max_err(ref,test):.2e}  r={correlation(ref,test):.6f}'
