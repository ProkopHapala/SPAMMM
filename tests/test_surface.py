import pytest, numpy as np, os, datetime
from spammm.AtomicSystem import AtomicSystem
from spammm.surfaces.Ewald2D import Ewald2D
from tests.helpers.parity import rmse, correlation, max_err, overlay_plot, assert_parity, parity_report
from tests.helpers.scan import compare_scans, assert_scan

def _load_substrate(path):
    """Load substrate XYZ, return (rx, ry, rz, q, a_vec, b_vec)."""
    mol = AtomicSystem(fname=path)
    apos = np.asarray(mol.apos, dtype=float)
    rx, ry, rz = apos[:, 0], apos[:, 1], apos[:, 2]
    q = np.asarray(mol.qs, dtype=float)
    lvec = np.asarray(mol.lvec, dtype=float)
    a_vec = lvec[0, :2]; b_vec = lvec[1, :2]
    return rx, ry, rz, q, a_vec, b_vec

def _debug_dir(name='surface'):
    d = os.path.join('debug', f'{datetime.date.today()}_{name}')
    os.makedirs(d, exist_ok=True)
    return d

# ---- Suite 3: Ewald2D Python reference tests ----

def test_ewald_neutrality_warning(substrate):
    """NaCl unit cell must be charge-neutral for Ewald."""
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    assert abs(np.sum(q)) < 1e-6, f'Unit cell not neutral: Q={np.sum(q)}'

def test_ewald_vacuum_decay(substrate):
    """Ewald vacuum potential should decay with z (far above slab)."""
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ew = Ewald2D(a_vec, b_vec, rx, ry, rz, q, n_harm=4)
    z_arr = np.linspace(5.0, 20.0, 50)
    phi = ew.phi_full_1d(0.05, 0.05, z_arr)
    assert np.all(np.isfinite(phi)), 'NaN in Ewald potential'
    # Potential should decrease in magnitude with height (dipole decays)
    assert abs(phi[0]) > abs(phi[-1]), f'Potential not decaying: phi[0]={phi[0]}, phi[-1]={phi[-1]}'

def test_ewald_symmetry(substrate):
    """Ewald potential should have NaCl 4-fold symmetry at vacuum height."""
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ew = Ewald2D(a_vec, b_vec, rx, ry, rz, q, n_harm=4)
    z = 3.0
    xv = np.linspace(0, 4, 20); yv = np.linspace(0, 4, 20)
    X, Y = np.meshgrid(xv, yv)
    phi = ew.phi_vacuum_xy(X, Y, z)
    # 4-fold rotational symmetry: phi(x,y) == phi(y,x)
    assert np.allclose(phi, phi.T, atol=1e-6), 'Ewald potential not symmetric under x<->y'

def test_ewald_brute_shape_match(substrate):
    """Ewald vs brute-force: shapes must match (constant offset is OK for 2D Ewald)."""
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ew = Ewald2D(a_vec, b_vec, rx, ry, rz, q, n_harm=4)
    z_arr = np.linspace(1.0, 5.0, 30)
    phi_ewald = ew.phi_full_1d(0.05, 0.05, z_arr)
    phi_brute = ew.phi_brute_1d(0.05, 0.05, z_arr, N_rep=20)
    # 2D Ewald has arbitrary zero — compare centered curves
    ewald_c = phi_ewald - phi_ewald.mean()
    brute_c = phi_brute - phi_brute.mean()
    r = rmse(ewald_c, brute_c)
    c = correlation(phi_ewald, phi_brute)
    assert c > 0.999, f'Ewald vs brute correlation too low: r={c:.6f}'
    assert r < 0.001, f'Ewald vs brute centered RMSE too high: {r:.2e}'

def test_brute_convergence(substrate):
    """Brute-force should converge with more shells (2D Coulomb converges slowly)."""
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ew = Ewald2D(a_vec, b_vec, rx, ry, rz, q, n_harm=4)
    z_arr = np.array([3.0])
    phi_20 = ew.phi_brute_1d(0.05, 0.05, z_arr, N_rep=20)
    phi_40 = ew.phi_brute_1d(0.05, 0.05, z_arr, N_rep=40)
    phi_60 = ew.phi_brute_1d(0.05, 0.05, z_arr, N_rep=60)
    # 2D Coulomb converges as ~1/N_rep, so relative improvement between shells
    d_20_40 = abs(phi_40[0] - phi_20[0])
    d_40_60 = abs(phi_60[0] - phi_40[0])
    assert d_40_60 < d_20_40, f'Brute not converging: d20-40={d_20_40:.4f}, d40-60={d_40_60:.4f}'

# ---- Suite 3: Ewald OpenCL parity (GPU) ----

@pytest.mark.gpu
def test_ewald_py_vs_cl(substrate):
    """Ewald2D (NumPy) vs SurfaceEwaldCL (OpenCL) on NaCl 1x1."""
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ion_data = np.column_stack([rx, ry, rz, q]).astype(np.float32)
    ew_py = Ewald2D(a_vec, b_vec, rx, ry, rz, q, n_harm=3)
    ew_cl = SurfaceEwaldCL()
    ew_cl.prepare_system(ion_data, a_vec.astype(np.float32), b_vec.astype(np.float32), n_harm=3)
    # Vacuum at z=2.0
    xv = np.linspace(0, 4, 20); yv = np.linspace(0, 4, 20)
    X, Y = np.meshgrid(xv, yv)
    phi_py = ew_py.phi_vacuum_xy(X, Y, 2.0)
    phi_cl = ew_cl.eval_vacuum(X.astype(np.float32), Y.astype(np.float32), 2.0)
    assert_parity(phi_py, phi_cl, rtol=1e-3, atol=1e-4, name='ewald_vacuum_py_vs_cl')

@pytest.mark.gpu
def test_ewald_cl_full_1d(substrate):
    """OpenCL Ewald full 1D scan vs Python reference."""
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ion_data = np.column_stack([rx, ry, rz, q]).astype(np.float32)
    ew_py = Ewald2D(a_vec, b_vec, rx, ry, rz, q, n_harm=3)
    ew_cl = SurfaceEwaldCL()
    ew_cl.prepare_system(ion_data, a_vec.astype(np.float32), b_vec.astype(np.float32), n_harm=3)
    z_arr = np.linspace(1.0, 5.0, 50)
    phi_py = ew_py.phi_full_1d(0.05, 0.05, z_arr)
    X = np.full((1, len(z_arr)), 0.05, dtype=np.float32)
    Y = np.full((1, len(z_arr)), 0.05, dtype=np.float32)
    Z = z_arr.reshape(1, -1).astype(np.float32)
    phi_cl = ew_cl.eval_full(X, Y, Z)[0, :]
    assert_parity(phi_py, phi_cl, rtol=1e-3, atol=1e-4, name='ewald_full_py_vs_cl')

# ---- Suite 3: Ewald vs GPU brute-force ----

@pytest.mark.gpu
def test_ewald_vs_brute_cl_zscan(substrate):
    """Ewald (OpenCL) vs GPU brute-force: z-scan above NaCl 1x1."""
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ion_data = np.column_stack([rx, ry, rz, q]).astype(np.float32)
    ew_cl = SurfaceEwaldCL()
    ew_cl.prepare_system(ion_data, a_vec.astype(np.float32), b_vec.astype(np.float32), n_harm=4)
    z_arr = np.linspace(1.0, 6.0, 50)
    X = np.full((1, len(z_arr)), 0.05, dtype=np.float32)
    Y = np.full((1, len(z_arr)), 0.05, dtype=np.float32)
    Z = z_arr.reshape(1, -1).astype(np.float32)
    phi_ewald = ew_cl.eval_full(X, Y, Z)[0, :]
    phi_brute = ew_cl.eval_brute(X, Y, Z, N_rep=20)[0, :]
    # 2D Ewald has arbitrary zero — compare centered
    ewald_c = phi_ewald - phi_ewald.mean()
    brute_c = phi_brute - phi_brute.mean()
    r = rmse(ewald_c, brute_c)
    c = correlation(phi_ewald, phi_brute)
    assert c > 0.999, f'Ewald vs brute CL: correlation={c:.6f}'
    assert r < 0.01, f'Ewald vs brute CL: centered RMSE={r:.2e}'

@pytest.mark.gpu
def test_ewald_vs_brute_cl_xscan(substrate):
    """Ewald (OpenCL) vs GPU brute-force: x-scan at z=3 A."""
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ion_data = np.column_stack([rx, ry, rz, q]).astype(np.float32)
    ew_cl = SurfaceEwaldCL()
    ew_cl.prepare_system(ion_data, a_vec.astype(np.float32), b_vec.astype(np.float32), n_harm=4)
    x_arr = np.linspace(0.0, 4.0, 50)
    z0 = 3.0
    X = x_arr.reshape(1, -1).astype(np.float32)
    Y = np.full((1, len(x_arr)), 0.05, dtype=np.float32)
    Z = np.full((1, len(x_arr)), z0, dtype=np.float32)
    phi_ewald = ew_cl.eval_full(X, Y, Z)[0, :]
    phi_brute = ew_cl.eval_brute(X, Y, Z, N_rep=20)[0, :]
    ewald_c = phi_ewald - phi_ewald.mean()
    brute_c = phi_brute - phi_brute.mean()
    r = rmse(ewald_c, brute_c)
    c = correlation(phi_ewald, phi_brute)
    assert c > 0.999, f'Ewald vs brute CL xscan: correlation={c:.6f}'
    assert r < 0.01, f'Ewald vs brute CL xscan: centered RMSE={r:.2e}'

# ---- Suite 3: Large system (NaCl 8x8) ----

@pytest.mark.gpu
@pytest.mark.slow
def test_ewald_vs_brute_nacl8_zscan(substrate):
    """Ewald vs GPU brute-force on NaCl 8x8 (384 atoms): z-scan."""
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_8x8_L3.xyz'))
    ion_data = np.column_stack([rx, ry, rz, q]).astype(np.float32)
    ew_cl = SurfaceEwaldCL()
    ew_cl.prepare_system(ion_data, a_vec.astype(np.float32), b_vec.astype(np.float32), n_harm=4)
    z_arr = np.linspace(1.0, 6.0, 50)
    X = np.full((1, len(z_arr)), 0.05, dtype=np.float32)
    Y = np.full((1, len(z_arr)), 0.05, dtype=np.float32)
    Z = z_arr.reshape(1, -1).astype(np.float32)
    phi_ewald = ew_cl.eval_full(X, Y, Z)[0, :]
    phi_brute = ew_cl.eval_brute(X, Y, Z, N_rep=20)[0, :]
    ewald_c = phi_ewald - phi_ewald.mean()
    brute_c = phi_brute - phi_brute.mean()
    r = rmse(ewald_c, brute_c)
    c = correlation(phi_ewald, phi_brute)
    assert c > 0.999, f'NaCl 8x8: correlation={c:.6f}'
    assert r < 0.01, f'NaCl 8x8: centered RMSE={r:.2e}'

# ---- Visual outputs ----

@pytest.mark.visual
def test_visual_ewald_brute_zscan(substrate):
    """Overlay plot: Ewald vs brute-force z-scan, 3 locations above NaCl 1x1."""
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ew = Ewald2D(a_vec, b_vec, rx, ry, rz, q, n_harm=4)
    save_dir = _debug_dir('ewald')
    z_arr = np.linspace(1.0, 6.0, 50)
    for name, x0, y0 in [('above_Na', 0.05, 0.05), ('above_Cl', 2.05, 2.05), ('midpoint', 1.0, 1.0)]:
        phi_ewald = ew.phi_full_1d(x0, y0, z_arr)
        phi_brute = ew.phi_brute_1d(x0, y0, z_arr, N_rep=20)
        overlay_plot(z_arr, [phi_ewald, phi_brute], ['Ewald', 'Brute'],
                     f'z-scan {name} (RMSE={rmse(phi_ewald,phi_brute):.2e}, r={correlation(phi_ewald,phi_brute):.4f})',
                     'z [A]', savepath=os.path.join(save_dir, f'zscan_{name}.png'))

@pytest.mark.visual
def test_visual_ewald_brute_xscan(substrate):
    """Overlay plot: Ewald vs brute-force x-scan at z=3 A."""
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ew = Ewald2D(a_vec, b_vec, rx, ry, rz, q, n_harm=4)
    save_dir = _debug_dir('ewald')
    x_arr = np.linspace(0.0, 4.0, 80)
    z0 = 3.0
    phi_ewald = np.array([ew.phi_full_1d(x, 0.05, np.array([z0]))[0] for x in x_arr])
    phi_brute = np.array([ew.phi_brute_1d(x, 0.05, np.array([z0]), N_rep=20)[0] for x in x_arr])
    overlay_plot(x_arr, [phi_ewald, phi_brute], ['Ewald', 'Brute'],
                 f'x-scan z={z0} (RMSE={rmse(phi_ewald,phi_brute):.2e}, r={correlation(phi_ewald,phi_brute):.4f})',
                 'x [A]', savepath=os.path.join(save_dir, 'xscan_z3.png'))

@pytest.mark.visual
def test_visual_ewald_2d_map(substrate):
    """2D color map of Ewald potential at z=3 A."""
    import matplotlib.pyplot as plt
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L3.xyz'))
    ew = Ewald2D(a_vec, b_vec, rx, ry, rz, q, n_harm=4)
    save_dir = _debug_dir('ewald')
    xv = np.linspace(0, 4, 100); yv = np.linspace(0, 4, 100)
    X, Y = np.meshgrid(xv, yv)
    phi = ew.phi_vacuum_xy(X, Y, 3.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(X, Y, phi, shading='auto', cmap='RdBu_r')
    ax.set_xlabel('x [A]'); ax.set_ylabel('y [A]'); ax.set_title('Ewald phi(x,y) at z=3 A')
    fig.colorbar(im, label='eV/e')
    fig.savefig(os.path.join(save_dir, 'ewald_2d_map_z3.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
