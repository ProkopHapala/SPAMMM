import numpy as np, os
from .parity import plot_curves, rmse, correlation, max_err

def z_scan(eval_func, x0, y0, z_range):
    """eval_func(x, y, z) -> float. Returns phi(z) array."""
    return np.array([eval_func(x0, y0, z) for z in z_range])

def x_scan(eval_func, y0, z0, x_range):
    return np.array([eval_func(x, y0, z0) for x in x_range])

def compare_scans(scan_name, coord, ref, test, ref_label='ref', test_label='test', save_dir=None, rmse_tol=0.01, corr_tol=0.999):
    r, c = rmse(ref, test), correlation(ref, test)
    result = {'name': scan_name, 'rmse': r, 'correlation': c, 'max_err': max_err(ref, test),
              'pass': r < rmse_tol and c > corr_tol}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_curves(coord, [ref, test], [ref_label, test_label],
                    f'{scan_name} (RMSE={r:.2e}, r={c:.4f})',
                    'z [A]' if 'z' in scan_name else 'x [A]',
                    os.path.join(save_dir, f'{scan_name}.png'), pairs=[(1,0)])
    return result

def assert_scan(scan_name, coord, ref, test, ref_label='ref', test_label='test', save_dir=None, **kw):
    res = compare_scans(scan_name, coord, ref, test, ref_label, test_label, save_dir, **kw)
    assert res['pass'], f'{scan_name}: RMSE={res["rmse"]:.2e} corr={res["correlation"]:.4f}'
    return res
