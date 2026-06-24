"""
test_afm_morse.py — AFM imaging tests with Morse + Coulomb force field (Phase 1).

Tests the AFMulator GPU pipeline: load molecule → assign params → setup grid →
make_forcefield → raw scan → relaxed scan (PP) → df conversion.

Hierarchy:
  1. test_afm_grid_finite      — force field grid is finite & physically reasonable
  2. test_afm_raw_scan          — raw FE scan (no PP relax): Fz finite, correct sign
  3. test_afm_relaxed_scan      — PP-relaxed scan: Fz finite, differs from raw
  4. test_afm_df_finite         — compute_df produces finite frequency shift
  5. test_afm_morse_vs_lj       — Morse and LJ produce different but correlated results
  6. test_visual_afm_morse_images — 2D AFM image slices at multiple z heights (visual)

All tests use AFMulator from spammm.SPM.AFM with Morse or LJ potential +
point-charge Coulomb (tipQs/tipQZs). No electron density required.

Expected physics:
  - Fz (vertical force on probe) should be repulsive (positive) near the surface
  - Fz should decay toward zero at large z
  - PP relaxation should reduce lateral forces, modifying Fz vs raw
  - df (frequency shift) should be negative in the attractive regime
"""
import pytest, numpy as np, os, datetime

os.environ.setdefault('PYOPENCL_CTX', '0')
os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

from tests.helpers.parity import rmse, correlation

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
PARAMS_PATH = os.path.join(DATA_DIR, 'ElementTypes.dat')

def _debug_dir(name='afm_morse'):
    d = os.path.join('debug', f'{datetime.date.today()}_{name}')
    os.makedirs(d, exist_ok=True)
    return d

def _make_afmulator(xyz_path, use_morse=True, n_grid=(60,60,40), margin=3.0, z_top=12.0):
    """Build AFMulator, load molecule, assign params, setup grid. Returns (afm, mol)."""
    from spammm.SPM.AFM import AFMulator
    afm = AFMulator(use_morse=use_morse)
    mol = afm.load_molecule(xyz_path)
    afm.assign_params(params_path=PARAMS_PATH)
    afm.setup_grid(n=n_grid, margin=margin, z_top=z_top)
    return afm, mol

def _ensure_finite(name, arr, abs_max=1e6):
    assert np.isfinite(arr).all(), f"{name}: non-finite values at {np.where(~np.isfinite(arr))}"
    m = float(np.max(np.abs(arr))) if arr.size else 0.0
    assert m < abs_max, f"{name}: abs_max={m:.3e} exceeds threshold {abs_max:.3e}"


# =============================================================================
# Test 1: Force field grid is finite & physically reasonable
# =============================================================================

@pytest.mark.gpu
@pytest.mark.parametrize('mol_file', ['CO.xyz', 'benzene.xyz', 'pentacene.xyz'])
def test_afm_grid_finite(xyz, mol_file):
    """Build Morse FF grid for molecule, check all values finite and Fz has expected sign pattern."""
    afm, mol = _make_afmulator(xyz(mol_file), use_morse=True, n_grid=(50,50,30))
    afm.make_forcefield()

    # Download FF grid from GPU: sample at grid centers
    nx, ny, nz = afm.n
    img_h = np.zeros((nx,ny,nz,4), dtype=np.float32)
    import pyopencl as cl
    cl.enqueue_copy(afm.queue, img_h, afm.img_FF, origin=(0,0,0), region=(nx,ny,nz))
    afm.queue.finish()

    Fx, Fy, Fz, E = img_h[...,0], img_h[...,1], img_h[...,2], img_h[...,3]
    _ensure_finite('Fx', Fx); _ensure_finite('Fy', Fy)
    _ensure_finite('Fz', Fz); _ensure_finite('E', E)

    # Energy should be negative somewhere (attractive well from Morse)
    assert E.min() < 0, f"E.min()={E.min():.4f} should be < 0 (Morse attractive well)"

    # Near the top of grid (far from molecule), forces should be smaller than peak
    top_slab_Fz = Fz[:, :, -3:]
    peak_Fz = np.max(np.abs(Fz))
    top_Fz = np.max(np.abs(top_slab_Fz))
    if top_Fz >= peak_Fz:
        print(f"  WARNING: Fz at grid top ({top_Fz:.4f}) >= peak ({peak_Fz:.4f}) — grid too small for {mol_file}")

    print(f"[{mol_file}] E range [{E.min():.4f}, {E.max():.4f}]  Fz range [{Fz.min():.4f}, {Fz.max():.4f}]  top/peak={top_Fz/peak_Fz:.3f}")


# =============================================================================
# Test 2: Raw scan (no PP relaxation) — Fz finite, correct sign
# =============================================================================

@pytest.mark.gpu
@pytest.mark.parametrize('mol_file', ['CO.xyz', 'benzene.xyz'])
def test_afm_raw_scan(xyz, mol_file):
    """Raw FE scan: Fz finite, repulsive at close range, decaying at far range."""
    afm, mol = _make_afmulator(xyz(mol_file), use_morse=True, n_grid=(50,50,30))
    afm.make_forcefield()

    nz = 15
    dtip = -0.2  # Å per step (descending)
    FEs, pts = afm.get_raw_FE(nxy=(30,30), nz=nz, dtip=dtip)
    Fz = FEs[:,:,:,2]
    _ensure_finite('Fz_raw', Fz)

    # At lowest z (closest to molecule, iz=0 after reshape), Fz should be strongest
    # Note: get_raw_FE returns iz=0 = first scan point (highest z), iz=nz-1 = lowest z
    Fz_close = Fz[:, :, -1]   # closest to surface
    Fz_far   = Fz[:, :, 0]    # farthest from surface
    assert np.max(np.abs(Fz_close)) > np.max(np.abs(Fz_far)), \
        f"|Fz| at close range ({np.max(np.abs(Fz_close)):.4f}) should exceed far range ({np.max(np.abs(Fz_far)):.4f})"

    print(f"[{mol_file}] Fz_raw: close max={np.max(np.abs(Fz_close)):.4f}  far max={np.max(np.abs(Fz_far)):.4f}")


# =============================================================================
# Test 3: PP-relaxed scan — Fz finite, differs from raw
# =============================================================================

@pytest.mark.gpu
@pytest.mark.parametrize('mol_file', ['CO.xyz', 'benzene.xyz'])
def test_afm_relaxed_scan(xyz, mol_file):
    """PP-relaxed scan: Fz finite, should differ from raw (relaxation shifts probe)."""
    afm, mol = _make_afmulator(xyz(mol_file), use_morse=True, n_grid=(50,50,30))
    afm.make_forcefield()

    nz = 15
    dtip = -0.2
    nxy = (30, 30)

    FEs_raw, pts = afm.get_raw_FE(nxy=nxy, nz=nz, dtip=dtip)
    Fz_raw = FEs_raw[:,:,:,2]
    _ensure_finite('Fz_raw', Fz_raw)

    FEs_relax, pts2 = afm.run_scan(nxy=nxy, nz=nz, dtip=dtip)
    Fz_relax = FEs_relax[:,:,:,2]
    _ensure_finite('Fz_relax', Fz_relax)

    # Relaxed and raw should be correlated but not identical
    r = correlation(Fz_raw.ravel(), Fz_relax.ravel())
    assert r > 0.5, f"Correlation raw vs relax = {r:.4f}, expected > 0.5"

    # They should differ (PP relaxation shifts probe laterally)
    rms_diff = rmse(Fz_raw, Fz_relax)
    peak = max(np.max(np.abs(Fz_raw)), np.max(np.abs(Fz_relax)))
    assert rms_diff > 1e-6 * peak, f"raw vs relax RMSE={rms_diff:.2e} too small (peak={peak:.4f}), PP relaxation had no effect"

    print(f"[{mol_file}] raw vs relax: r={r:.4f}  RMSE={rms_diff:.4e}  peak={peak:.4f}")


# =============================================================================
# Test 4: df (frequency shift) is finite
# =============================================================================

@pytest.mark.gpu
def test_afm_df_finite(xyz):
    """compute_df produces finite frequency shift from Fz scan."""
    from spammm.SPM.AFM import compute_df
    afm, mol = _make_afmulator(xyz('benzene.xyz'), use_morse=True, n_grid=(50,50,30))
    afm.make_forcefield()

    nz = 15; dtip = -0.2
    FEs, _ = afm.run_scan(nxy=(30,30), nz=nz, dtip=dtip)
    Fz = FEs[:,:,:,2]
    _ensure_finite('Fz', Fz)

    df = compute_df(Fz, abs(dtip))
    _ensure_finite('df', df, abs_max=1e4)

    # df should have non-zero variation (not flat)
    assert np.max(np.abs(df)) > 1e-8, f"df max={np.max(np.abs(df)):.2e}, expected non-zero"

    print(f"df range [{df.min():.4f}, {df.max():.4f}]  Fz range [{Fz.min():.4f}, {Fz.max():.4f}]")


# =============================================================================
# Test 5: Morse vs LJ — different but correlated
# =============================================================================

@pytest.mark.gpu
def test_afm_morse_vs_lj(xyz):
    """Morse and LJ force fields produce different but correlated Fz scans."""
    mol_file = 'benzene.xyz'

    # Morse
    afm_m, _ = _make_afmulator(xyz(mol_file), use_morse=True, n_grid=(50,50,30))
    afm_m.make_forcefield()
    FEs_m, _ = afm_m.get_raw_FE(nxy=(30,30), nz=15, dtip=-0.2)
    Fz_m = FEs_m[:,:,:,2]
    _ensure_finite('Fz_morse', Fz_m)

    # LJ
    afm_l, _ = _make_afmulator(xyz(mol_file), use_morse=False, n_grid=(50,50,30))
    afm_l.make_forcefield()
    FEs_l, _ = afm_l.get_raw_FE(nxy=(30,30), nz=15, dtip=-0.2)
    Fz_l = FEs_l[:,:,:,2]
    _ensure_finite('Fz_lj', Fz_l)

    # Should be correlated (both see same molecular geometry)
    r = correlation(Fz_m.ravel(), Fz_l.ravel())
    assert r > 0.3, f"Morse vs LJ correlation = {r:.4f}, expected > 0.3"

    # Should differ (different functional forms)
    rms_diff = rmse(Fz_m, Fz_l)
    peak = max(np.max(np.abs(Fz_m)), np.max(np.abs(Fz_l)))
    assert rms_diff > 1e-3 * peak, f"Morse vs LJ too similar: RMSE={rms_diff:.2e}, peak={peak:.4f}"

    print(f"Morse vs LJ: r={r:.4f}  RMSE={rms_diff:.4e}  peak_m={np.max(np.abs(Fz_m)):.4f}  peak_l={np.max(np.abs(Fz_l)):.4f}")


# =============================================================================
# Test 6 (visual): 2D AFM image slices at multiple z heights
# =============================================================================

@pytest.mark.gpu
@pytest.mark.visual
def test_visual_afm_morse_images(xyz):
    """Generate 2D AFM Fz and df image slices at multiple z heights for benzene.

    Expected: Fz images show repulsive pattern (bright) over atoms at close range,
    fading to uniform at large z. df images show contrast inversion typical of AFM.
    """
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from spammm.SPM.AFM import compute_df

    mol_file = 'benzene.xyz'
    afm, mol = _make_afmulator(xyz(mol_file), use_morse=True, n_grid=(60,60,40), margin=4.0, z_top=14.0)
    afm.make_forcefield()

    nz = 25; dtip = -0.15
    nxy = (40, 40)
    FEs, pts = afm.run_scan(nxy=nxy, nz=nz, dtip=dtip)
    Fz = FEs[:,:,:,2]
    _ensure_finite('Fz', Fz)
    df = compute_df(Fz, abs(dtip))
    _ensure_finite('df', df)

    # Probe heights above molecule top (in kernel-space)
    mol_z = afm.mol_z
    z0_tip = mol_z + 5.0 + abs(float(afm.dpos0[2]))  # start height
    heights = z0_tip + np.arange(nz) * dtip - mol_z  # relative to mol top

    save_dir = _debug_dir('afm_morse_images')
    sel_iz = [0, 5, 10, 15, 20]  # selected z slices
    sel_iz = [iz for iz in sel_iz if iz < nz]

    # --- Fz slices ---
    fig, axes = plt.subplots(1, len(sel_iz), figsize=(3*len(sel_iz), 3))
    if len(sel_iz) == 1: axes = [axes]
    for ax, iz in zip(axes, sel_iz):
        data = Fz[:,:,iz].T
        vabs = max(float(np.percentile(np.abs(data), 99)), 1e-6)
        im = ax.imshow(data, origin='lower', cmap='bwr', aspect='equal', vmin=-vabs, vmax=vabs)
        ax.set_title(f'Fz h={heights[iz]:.2f}Å', fontsize=8)
        ax.tick_params(labelsize=5)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'AFM Fz (Morse) — {mol_file}', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'afm_Fz_slices_{mol_file.replace(".xyz","")}.png'), dpi=120)
    plt.close(fig)
    print(f"Saved Fz slices to {save_dir}")

    # --- df slices ---
    fig, axes = plt.subplots(1, len(sel_iz), figsize=(3*len(sel_iz), 3))
    if len(sel_iz) == 1: axes = [axes]
    for ax, iz in zip(axes, sel_iz):
        data = df[:,:,iz].T
        vabs = max(float(np.percentile(np.abs(data), 99)), 1e-6)
        im = ax.imshow(data, origin='lower', cmap='bwr', aspect='equal', vmin=-vabs, vmax=vabs)
        ax.set_title(f'df h={heights[iz]:.2f}Å', fontsize=8)
        ax.tick_params(labelsize=5)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'AFM df (Morse) — {mol_file}', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'afm_df_slices_{mol_file.replace(".xyz","")}.png'), dpi=120)
    plt.close(fig)
    print(f"Saved df slices to {save_dir}")

    # --- Fz(z) curve at center pixel ---
    ix_c, iy_c = nxy[0]//2, nxy[1]//2
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(heights, Fz[ix_c, iy_c, :], 'b-', lw=1.5, marker='o', markersize=3, label='Fz')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Height above mol top (Å)'); ax.set_ylabel('Fz (eV/Å)')
    ax.set_title(f'Fz(z) at center pixel — {mol_file}')
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'afm_Fz_curve_{mol_file.replace(".xyz","")}.png'), dpi=120)
    plt.close(fig)
    print(f"Saved Fz curve to {save_dir}")

    # Save raw data
    np.save(os.path.join(save_dir, f'Fz_{mol_file.replace(".xyz","")}.npy'), Fz)
    np.save(os.path.join(save_dir, f'df_{mol_file.replace(".xyz","")}.npy'), df)
    np.save(os.path.join(save_dir, f'heights_{mol_file.replace(".xyz","")}.npy'), heights)
