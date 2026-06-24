import pytest, numpy as np, os, datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spammm.AtomicSystem import AtomicSystem
from spammm.surfaces.Ewald2D import Ewald2D
from tests.helpers.parity import rmse, correlation, max_err, plot_curves, assert_parity, parity_report
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

def _make_finite_cluster(rx, ry, rz, q, a_vec, b_vec, K=5, evjen=True):
    """Build a symmetric Evjen finite cluster on the 2A sublattice.

    The NaCl(100) checkerboard has charges q(i,j) = ±q0*(-1)^(i+j) on a 2A grid.
    We tile -K..K (odd, 2K+1 sites per direction), centered at origin.
    Evjen boundary weights: edge=1/2, corner=1/4, to cancel multipoles.

    Expected results (float64 reference, K=20, z=2A):
        - DC offset: ~-2e-6 eV
        - RMSE after DC: ~5e-9 eV
        - Slope of residual: ~0 (machine precision)
        - Cluster: 2*(2K+1)^2 ions, x,y in [-2K, +2K] A, perfectly symmetric

    Caveats:
        - MUST tile the 2A sublattice, NOT the 4A unit cell. Tiling the 4A cell
          gives an even number of sites per direction and asymmetric spatial
          extent (e.g. [-40, +42] instead of [-40, +40]), producing a spurious
          slope in the residual error across the unit cell.
        - MUST use odd (2K+1) sites per direction for symmetry about origin.
        - np.unique(rz) sorts ascending, so iz=0 is the BOTTOM layer.
          Sign must be (-1)^(n_layers-1-iz) so top layer has Na(+q0) at origin.
          Reversing this sign gives potentials with opposite sign.
        - Evjen weights only affect boundary ions; for the symmetric checkerboard
          the cluster is already charge/dipole/quadrupole-free without Evjen,
          but Evjen improves convergence rate (DC offset shrinks faster with K).
        - The cluster is centered at (0,0) which is a Na site in the top layer.
          Scans should pass through (0,0) or other symmetry points to avoid
          finite-size slope artifacts from off-center evaluation.
    """
    a_vec = np.asarray(a_vec, dtype=float)
    b_vec = np.asarray(b_vec, dtype=float)
    # Sublattice spacing = half the unit cell vector
    dx = a_vec[0] / 2.0  # 2.0 for NaCl
    dy = b_vec[1] / 2.0
    q0 = abs(q[0])  # 0.7 for NaCl
    # Identify layers by unique z
    z_layers = np.unique(rz)
    clx, cly, clz, clq = [], [], [], []
    for iz, zv in enumerate(z_layers):
        # np.unique sorts ascending: iz=0 is bottom layer
        # Top layer (last) has Na(+q0) at origin: sign=+1
        # Bottom layer has Cl(-q0) at origin: sign=-1
        sign = (-1)**(len(z_layers) - 1 - iz)
        for i in range(-K, K+1):
            for j in range(-K, K+1):
                w = 1.0
                if evjen:
                    if abs(i) == K: w *= 0.5
                    if abs(j) == K: w *= 0.5
                qi = sign * q0 * ((-1)**(i+j)) * w
                clx.append(i * dx)
                cly.append(j * dy)
                clz.append(zv)
                clq.append(qi)
    return np.column_stack([clx, cly, clz, clq]).astype(np.float32)

def _plot_cluster(cluster, save_dir, name='cluster'):
    """Save cluster scatter plot (top/bottom layers, color=charge) and .xyz file."""
    x, y, z, q = cluster[:,0], cluster[:,1], cluster[:,2], cluster[:,3]
    z_vals = np.unique(z)
    fig, axes = plt.subplots(1, len(z_vals), figsize=(5*len(z_vals), 5))
    if len(z_vals) == 1: axes = [axes]
    for ax, zv in zip(axes, z_vals):
        mask = z == zv
        sc = ax.scatter(x[mask], y[mask], c=q[mask], cmap='RdBu', s=20, vmin=-abs(q).max(), vmax=abs(q).max())
        ax.set_aspect('equal'); ax.set_title(f'z={zv:.3f}'); ax.set_xlabel('x [A]'); ax.set_ylabel('y [A]')
        plt.colorbar(sc, ax=ax, label='q [e]')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name}.png'), dpi=100)
    plt.close(fig)
    # Save .xyz
    with open(os.path.join(save_dir, f'{name}.xyz'), 'w') as f:
        f.write(f'{len(cluster)}\n{name}\n')
        for i in range(len(cluster)):
            el = 'Na' if q[i] > 0 else 'Cl'
            f.write(f'  {el}   {x[i]:10.3f}   {y[i]:10.3f}   {z[i]:10.3f}  {q[i]:+.4f}\n')

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

# ---- Visual outputs (GPU) ----

@pytest.mark.visual
@pytest.mark.gpu
def test_visual_ewald_brute_zscan(substrate):
    """Z-scan: Ewald vs Evjen finite-cluster brute force at 3 lateral positions.

    Uses NaCl_1x1_L2_checker.xyz (8 ions, 2 layers, proper checkerboard) and
    a K=20 Evjen cluster (3362 ions, ±40A extent) on the 2A sublattice.

    Expected:
        - above_Na (0,0): potential ~+0.002 eV at z=0.5, decaying to ~0 by z=10
        - above_Cl (2,0): potential ~-0.002 eV at z=0.5 (opposite sign)
        - midpoint (1,1): potential ~0 at all z (symmetry point between ions)
        - RMSE after DC offset: ~1-3e-6 (GPU float32 noise floor)
        - Correlation r > 0.999

    Caveats:
        - A constant DC offset (~-2e-6 eV) is subtracted by plot_curves;
          this is expected from the finite cluster boundary and shrinks as 1/K^2.
        - At z > 5A the potential is < 1e-5 eV, so float32 noise (~1e-6)
          dominates the relative error. This is the GPU precision floor,
          not a physics error.
        - The 'midpoint' location (1,1) is a symmetry point where the potential
          is ~0 by symmetry; useful for checking that both methods agree on zero.
    """
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L2_checker.xyz'))
    ion_data = np.column_stack([rx, ry, rz, q]).astype(np.float32)
    ew_cl = SurfaceEwaldCL()
    ew_cl.prepare_system(ion_data, a_vec.astype(np.float32), b_vec.astype(np.float32), n_harm=4)
    cluster = _make_finite_cluster(rx, ry, rz, q, a_vec, b_vec, K=20)
    save_dir = _debug_dir('ewald')
    _plot_cluster(cluster, save_dir, 'cluster_K20')
    z_arr = np.concatenate([np.linspace(0.5, 3.0, 40), np.linspace(3.0, 10.0, 20)])
    for name, x0, y0 in [('above_Na', 0.0, 0.0), ('above_Cl', 2.0, 0.0), ('midpoint', 1.0, 1.0)]:
        X = np.full((1, len(z_arr)), x0, dtype=np.float32)
        Y = np.full((1, len(z_arr)), y0, dtype=np.float32)
        Z = z_arr.reshape(1, -1).astype(np.float32)
        phi_ewald = ew_cl.eval_full(X, Y, Z)[0, :]
        phi_brute = ew_cl.eval_cluster(X, Y, Z, cluster)[0, :]
        plot_curves(z_arr, [phi_brute, phi_ewald], ['Brute', 'Ewald'],
                    f'z-scan {name} (RMSE={rmse(phi_ewald,phi_brute):.2e}, r={correlation(phi_ewald,phi_brute):.4f})',
                    'z [A]', savepath=os.path.join(save_dir, f'zscan_{name}.png'), pairs=[(1,0)])

@pytest.mark.visual
@pytest.mark.gpu
def test_visual_ewald_brute_xscan(substrate):
    """X-scan at z=2A through cluster center (y=0, Na site).

    Scans x from -2 to +2 A at y=0, z=2A, comparing Ewald vs Evjen cluster.

    Expected:
        - Potential oscillates: +0.002 at x=0 (Na), 0 at x=±1, -0.002 at x=±2 (Cl)
        - RMSE after DC: ~1-3e-6 (float32 floor)
        - Residual diff*100 should be FLAT (no slope) — the symmetric cluster
          eliminates the spurious slope seen with asymmetric clusters.

    Caveats:
        - If the cluster were asymmetric (even sites per direction), a slope
          of ~2e-4/A would appear in the residual. This was the primary bug
          in earlier versions: tiling the 4A unit cell gave [-40,+42] extent.
        - Scanning through y=0 (Na site) is equivalent to y=2 (Cl site) by
          symmetry; both give the same residual pattern.
    """
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L2_checker.xyz'))
    ion_data = np.column_stack([rx, ry, rz, q]).astype(np.float32)
    ew_cl = SurfaceEwaldCL()
    ew_cl.prepare_system(ion_data, a_vec.astype(np.float32), b_vec.astype(np.float32), n_harm=4)
    cluster = _make_finite_cluster(rx, ry, rz, q, a_vec, b_vec, K=20)
    save_dir = _debug_dir('ewald')
    _plot_cluster(cluster, save_dir, 'cluster_K20')
    x_arr = np.linspace(-2.0, 2.0, 80, endpoint=False)
    z0 = 2.0
    X = x_arr.reshape(1, -1).astype(np.float32)
    Y = np.full((1, len(x_arr)), 0.0, dtype=np.float32)
    Z = np.full((1, len(x_arr)), z0, dtype=np.float32)
    phi_ewald = ew_cl.eval_full(X, Y, Z)[0, :]
    phi_brute = ew_cl.eval_cluster(X, Y, Z, cluster)[0, :]
    plot_curves(x_arr, [phi_brute, phi_ewald], ['Brute', 'Ewald'],
                f'x-scan z={z0} (RMSE={rmse(phi_ewald,phi_brute):.2e}, r={correlation(phi_ewald,phi_brute):.4f})',
                'x [A]', savepath=os.path.join(save_dir, 'xscan_z2.png'), pairs=[(1,0)])

@pytest.mark.visual
@pytest.mark.gpu
def test_visual_ewald_lateral_scans(substrate):
    """Lateral scans (x, y, diagonal) at z=2A through cluster center.

    Three scans at z=2A, all passing through (0,0) which is the cluster center
    and a Na site in the top layer:
        - x-scan: y=0, x from -2 to +2
        - y-scan: x=0, y from -2 to +2
        - diagonal: x=y, from -2 to +2

    Expected:
        - x and y scans should be identical by 4-fold symmetry of NaCl(100)
        - Diagonal scan has different amplitude (different path through ions)
        - RMSE after DC: ~1-3e-6 for all scans
        - All residuals should be flat (no slope)

    Caveats:
        - Scans MUST pass through the cluster center (0,0) or another symmetry
          point. Off-center scans (e.g. y=0.05) show a finite-size slope even
          with a perfectly symmetric cluster, because the residual field from
          the finite boundary varies within the unit cell.
        - The diagonal scan x=y passes through both Na and Cl sites, giving
          the richest variation; good for catching sign errors.
    """
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    rx, ry, rz, q, a_vec, b_vec = _load_substrate(substrate('NaCl_1x1_L2_checker.xyz'))
    ion_data = np.column_stack([rx, ry, rz, q]).astype(np.float32)
    ew_cl = SurfaceEwaldCL()
    ew_cl.prepare_system(ion_data, a_vec.astype(np.float32), b_vec.astype(np.float32), n_harm=4)
    cluster = _make_finite_cluster(rx, ry, rz, q, a_vec, b_vec, K=20)
    save_dir = _debug_dir('ewald')
    _plot_cluster(cluster, save_dir, 'cluster_K20')
    z0 = 2.0
    t_arr = np.linspace(-2.0, 2.0, 80, endpoint=False)
    # x-scan through cluster center (y=0, Na site)
    X = t_arr.reshape(1, -1).astype(np.float32)
    Y = np.full((1, len(t_arr)), 0.0, dtype=np.float32)
    Z = np.full((1, len(t_arr)), z0, dtype=np.float32)
    phi_ew = ew_cl.eval_full(X, Y, Z)[0, :]
    phi_br = ew_cl.eval_cluster(X, Y, Z, cluster)[0, :]
    plot_curves(t_arr, [phi_br, phi_ew], ['Brute', 'Ewald'],
                f'x-scan y=0.0 z={z0} (RMSE={rmse(phi_ew,phi_br):.2e})', 'x [A]',
                savepath=os.path.join(save_dir, 'lateral_xscan.png'), pairs=[(1,0)])
    # y-scan through cluster center (x=0, Na site)
    X = np.full((1, len(t_arr)), 0.0, dtype=np.float32)
    Y = t_arr.reshape(1, -1).astype(np.float32)
    phi_ew = ew_cl.eval_full(X, Y, Z)[0, :]
    phi_br = ew_cl.eval_cluster(X, Y, Z, cluster)[0, :]
    plot_curves(t_arr, [phi_br, phi_ew], ['Brute', 'Ewald'],
                f'y-scan x=0.0 z={z0} (RMSE={rmse(phi_ew,phi_br):.2e})', 'y [A]',
                savepath=os.path.join(save_dir, 'lateral_yscan.png'), pairs=[(1,0)])
    # diagonal scan through cluster center (x=y)
    X = t_arr.reshape(1, -1).astype(np.float32)
    Y = t_arr.reshape(1, -1).astype(np.float32)
    phi_ew = ew_cl.eval_full(X, Y, Z)[0, :]
    phi_br = ew_cl.eval_cluster(X, Y, Z, cluster)[0, :]
    plot_curves(t_arr, [phi_br, phi_ew], ['Brute', 'Ewald'],
                f'diagonal x=y z={z0} (RMSE={rmse(phi_ew,phi_br):.2e})', 'x=y [A]',
                savepath=os.path.join(save_dir, 'lateral_diagonal.png'), pairs=[(1,0)])
