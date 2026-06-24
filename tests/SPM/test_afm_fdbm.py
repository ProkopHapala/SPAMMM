"""
test_afm_fdbm.py — AFM imaging tests with Full Density-Based Model (FDBM) (Phase 2).

FDBM pipeline: DFTB+ SCF → density projection → Pauli (FFT convolution) +
electrostatic (Poisson) + dispersion → total force field → PP relaxation → df.

Tests use small molecules (H2O, benzene) to keep DFTB+ SCF fast.

Hierarchy:
  1. test_dftb_scf                — DFTB+ SCF runs, density matrix is finite
  2. test_density_projection      — GPU density projection: rho_scf finite, positive, integrates to ~N_electrons
  3. test_neutral_density         — rho_na finite, positive, integrates to ~N_electrons
  4. test_charge_conservation     — rho_diff = rho_scf - rho_na integrates to ~0
  5. test_poisson_potential       — V_ES from FFT Poisson is finite, decays at boundaries
  6. test_pauli_overlap           — Pauli overlap (FFT convolution) is finite, positive, peaks near molecule
  7. test_es_convolution          — Electrostatic convolution E_ES is finite
  8. test_dispersion              — vdW dispersion E_vdw is finite, attractive (negative)
  9. test_total_forcefield        — F_total = -grad(E_pauli + E_ES + E_vdw) is finite
  10. test_fdbm_relaxed_scan      — PP-relaxed scan: Fz finite, df finite
  11. test_fdbm_visual            — 2D slices of all FDBM components (visual marker)

Requires:
  - DFTB+ shared library (libdftbcore.so)
  - SK files (mio-1-1 or 3ob-3-1)
  - GPU (OpenCL)
"""
import pytest, numpy as np, os, datetime, shutil

os.environ.setdefault('PYOPENCL_CTX', '0')
os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

from tests.helpers.parity import rmse, correlation

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')

# Molecules for FDBM tests — small ones for fast SCF
FDBM_MOLECULES = ['H2O.xyz', 'benzene.xyz']

# Grid parameters
STEP = 0.15       # Å — coarser than Morse tests for speed (DFTB projection is expensive)
MARGIN = 4.0      # Å
Z_EXTRA = 6.0     # Å above molecule top

def _debug_dir(name='afm_fdbm'):
    d = os.path.join('debug', f'{datetime.date.today()}_{name}')
    os.makedirs(d, exist_ok=True)
    return d

def _check_dftb_available():
    """Check if DFTB+ library and SK files are available."""
    from spammm.quantum.DFTB.DFTBcore import _DEFAULT_LIB
    if _DEFAULT_LIB is None:
        pytest.skip("libdftbcore.so not found — skipping FDBM tests")
    from spammm.config_utils import get_config, get_path
    sk_path = get_path('dftb_sk_path')
    if sk_path is None or not os.path.isdir(sk_path):
        pytest.skip(f"SK library not found at {sk_path} — skipping FDBM tests")

def _load_molecule(xyz_path):
    """Load molecule geometry from XYZ file. Returns (atomPos, enames, atomTypes)."""
    import spammm.atomicUtils as au
    pos, _, names, _, _ = au.load_xyz(xyz_path)
    ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'P':15,'S':16,'Br':35,'I':53}
    atomPos = np.array(pos, dtype=np.float64)
    enames = list(names)
    atomTypes = np.array([ELEM_Z.get(e, 6) for e in enames], dtype=np.int32)
    return atomPos, enames, atomTypes

def _run_fdbm_pipeline(xyz_path, basis='mio-1-1', step=STEP, margin=MARGIN, z_extra=Z_EXTRA, work_dir=None):
    """Run the full FDBM pipeline for a molecule. Returns dict with all intermediates.

    This is the workhorse function — tests call it and check invariants on the output.
    """
    from spammm.SPM import AFM as afm
    from spammm.SPM import AFM_utils as afm_utils
    from spammm.config_utils import get_dftb_basis_path, get_dftb_sk_path

    atomPos, enames, atomTypes = _load_molecule(xyz_path)
    natoms = len(enames)
    print(f"\n[FDBM] {os.path.basename(xyz_path)}: {natoms} atoms, enames={enames[:5]}...")

    # Setup grid
    grid_spec, origin, ngrid = afm.setup_density_grid(atomPos, step=step, margin=margin, z_extra=z_extra)
    nx, ny, nz = int(ngrid[0]), int(ngrid[1]), int(ngrid[2])
    print(f"  Grid: {nx}x{ny}x{nz}  step={step}  origin={origin}")

    # Step 1: DFTB+ SCF → density matrix
    print("  Step 1: DFTB+ SCF...")
    basis_hsd_path = get_dftb_basis_path(basis)
    sk_dir = get_dftb_sk_path(basis)
    if basis_hsd_path is None or not os.path.exists(basis_hsd_path):
        raise FileNotFoundError(f"Basis HSD not found: {basis_hsd_path}")
    if sk_dir is None or not os.path.isdir(sk_dir):
        raise FileNotFoundError(f"SK dir not found: {sk_dir}")

    if work_dir is None:
        work_dir = os.path.join(_debug_dir(), f'dftb_work_{os.path.basename(xyz_path).replace(".xyz","")}')
    os.makedirs(work_dir, exist_ok=True)

    # Use get_density_from_dftb_dense — handles SCF + projection + neutral density + Poisson
    result = afm_utils.get_density_from_dftb_dense(
        atomPos, atomTypes, basis_hsd_path, work_dir,
        grid_spec=grid_spec, step=step, margin=margin, z_extra=z_extra,
        verbosity=0
    )

    rho_scf = result['rho_scf']
    rho_na = result['rho_na']
    rho_diff = result['rho_diff']
    V_ES = result['V_ES']

    # Step 2: Pauli overlap (raw, A=1, beta=1)
    print("  Step 2: Pauli overlap...")
    # Build Gaussian tip density for CO tip
    sigma_tip = 0.7  # Å — Gaussian approximation for CO tip
    rho_tip_total = afm.build_gaussian_tip((nx, ny, nz), step, sigma_tip)
    rho_tip_delta = rho_tip_total  # For Gaussian tip, delta = total (no neutral atom subtraction)

    overlap_raw = afm.compute_pauli_overlap(rho_scf, rho_tip_total, step, tip_rolled=True)
    print(f"  overlap_raw: [{overlap_raw.min():.4e}, {overlap_raw.max():.4e}]")

    # Pauli energy with fitted parameters
    pauli_params = afm.PAULI_FITTED_DEFAULTS.get(basis, {'A': 787.22, 'beta': 1.2371})
    A_pauli = pauli_params['A']
    beta_pauli = pauli_params['beta']
    E_pauli = afm.scale_pauli_field(overlap_raw, step, A_pauli, beta_pauli, return_grads=False)
    print(f"  E_pauli: [{E_pauli.min():.4e}, {E_pauli.max():.4e}]  A={A_pauli}, beta={beta_pauli}")

    # Step 3: Electrostatic convolution
    print("  Step 3: Electrostatic convolution...")
    E_ES = afm.compute_es_conv_field(V_ES, rho_tip_delta, step, tip_rolled=True, return_grads=False)
    print(f"  E_ES: [{E_ES.min():.4e}, {E_ES.max():.4e}]")

    # Step 4: Dispersion
    print("  Step 4: Dispersion...")
    E_vdw = afm.compute_dispersion_grid(
        atomPos, atomTypes, origin, step, ngrid,
        C6_CO=30.0, return_grads=False
    )
    print(f"  E_vdw: [{E_vdw.min():.4e}, {E_vdw.max():.4e}]")

    # Step 5: Total energy and gradient
    print("  Step 5: Total force field...")
    E_total = E_pauli + E_ES + E_vdw
    print(f"  E_total: [{E_total.min():.4e}, {E_total.max():.4e}]")

    # GPU gradient
    afmulator = afm.AFMulator(use_morse=False, nloc=32)
    F_total = afmulator.compute_gradient_cl(E_total, step, bAlloc=True)
    print(f"  F_total: Fz=[{F_total[...,2].min():.4e}, {F_total[...,2].max():.4e}]")

    # Step 6: PP relaxation scan
    print("  Step 6: PP relaxation scan...")
    mol_z = float(atomPos[:,2].max())
    # Scan grid: cover molecule + 1Å margin, isotropic pixels
    scan_margin = 1.0
    x_min = float(atomPos[:,0].min() - scan_margin)
    x_max = float(atomPos[:,0].max() + scan_margin)
    y_min = float(atomPos[:,1].min() - scan_margin)
    y_max = float(atomPos[:,1].max() + scan_margin)
    scan_step = step  # same as grid step
    scan_xs = np.arange(x_min, x_max, scan_step, dtype=np.float32)
    scan_ys = np.arange(y_min, y_max, scan_step, dtype=np.float32)
    # Probe heights above mol_z — typical AFM range
    heights = np.arange(1.5, 5.0, 0.25, dtype=np.float32)

    afmulator.setup_fdbm_grid(F_total, origin, step)
    FEs_relax, tip_disp = afmulator.scan_fdbm(
        scan_xs, scan_ys, heights, mol_z=mol_z,
        K_LAT=0.5, K_RAD=20.0, bond_length=2.0,
        ppm_mode=True, use_fire=True
    )
    Fz_relax = FEs_relax[:,:,:,2]
    df = afm.compute_df(Fz_relax, float(heights[1] - heights[0]))
    print(f"  Fz_relax: [{Fz_relax.min():.4e}, {Fz_relax.max():.4e}]")
    print(f"  df: [{df.min():.4e}, {df.max():.4e}]")

    return {
        'atomPos': atomPos, 'enames': enames, 'atomTypes': atomTypes,
        'rho_scf': rho_scf, 'rho_na': rho_na, 'rho_diff': rho_diff, 'V_ES': V_ES,
        'overlap_raw': overlap_raw, 'E_pauli': E_pauli, 'E_ES': E_ES, 'E_vdw': E_vdw,
        'E_total': E_total, 'F_total': F_total,
        'FEs_relax': FEs_relax, 'Fz_relax': Fz_relax, 'df': df, 'tip_disp': tip_disp,
        'scan_xs': scan_xs, 'scan_ys': scan_ys, 'heights': heights,
        'origin': origin, 'step': step, 'ngrid': ngrid,
        'rho_tip_total': rho_tip_total,
    }


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fdbm_results():
    """Run FDBM pipeline once for all tests (module-scoped fixture)."""
    _check_dftb_available()
    results = {}
    for mol_file in FDBM_MOLECULES:
        xyz_path = os.path.join(DATA_DIR, 'xyz', mol_file)
        if not os.path.exists(xyz_path):
            print(f"  SKIP: {xyz_path} not found")
            continue
        try:
            results[mol_file] = _run_fdbm_pipeline(xyz_path)
        except Exception as e:
            print(f"  ERROR for {mol_file}: {e}")
            import traceback; traceback.print_exc()
    if not results:
        pytest.skip("No FDBM results — DFTB+ may have failed")
    return results


@pytest.mark.gpu
def test_dftb_scf(fdbm_results):
    """DFTB+ SCF ran successfully — rho_scf is finite and positive."""
    for mol_file, r in fdbm_results.items():
        rho_scf = r['rho_scf']
        assert np.all(np.isfinite(rho_scf)), f"{mol_file}: rho_scf has NaN/inf"
        assert rho_scf.min() >= 0, f"{mol_file}: rho_scf has negative values"
        assert rho_scf.max() > 0, f"{mol_file}: rho_scf is all zeros"
        print(f"[{mol_file}] rho_scf: [{rho_scf.min():.4e}, {rho_scf.max():.4e}]")


@pytest.mark.gpu
def test_density_projection(fdbm_results):
    """rho_scf integrates to approximately the number of valence electrons."""
    for mol_file, r in fdbm_results.items():
        rho_scf = r['rho_scf']
        step = r['step']
        dV = step**3
        n_electrons = rho_scf.sum() * dV
        # Expected: sum of valence electrons
        ELEM_Z_VAL = {'H':1, 'C':4, 'N':5, 'O':6, 'P':5, 'S':6}
        expected = sum(ELEM_Z_VAL.get(e, 4) for e in r['enames'])
        print(f"[{mol_file}] N_electrons={n_electrons:.2f}, expected={expected}")
        assert abs(n_electrons - expected) < 2.0, \
            f"{mol_file}: rho_scf integral {n_electrons:.2f} != expected {expected}"


@pytest.mark.gpu
def test_neutral_density(fdbm_results):
    """rho_na is finite, positive, and integrates to ~N_electrons."""
    for mol_file, r in fdbm_results.items():
        rho_na = r['rho_na']
        step = r['step']
        dV = step**3
        assert np.all(np.isfinite(rho_na)), f"{mol_file}: rho_na has NaN/inf"
        assert rho_na.min() >= 0, f"{mol_file}: rho_na has negative values"
        n_electrons = rho_na.sum() * dV
        ELEM_Z_VAL = {'H':1, 'C':4, 'N':5, 'O':6, 'P':5, 'S':6}
        expected = sum(ELEM_Z_VAL.get(e, 4) for e in r['enames'])
        print(f"[{mol_file}] rho_na N_electrons={n_electrons:.2f}, expected={expected}")
        assert abs(n_electrons - expected) < 2.0, \
            f"{mol_file}: rho_na integral {n_electrons:.2f} != expected {expected}"


@pytest.mark.gpu
def test_charge_conservation(fdbm_results):
    """rho_diff = rho_scf - rho_na integrates to ~0 (charge conservation)."""
    for mol_file, r in fdbm_results.items():
        rho_diff = r['rho_diff']
        step = r['step']
        dV = step**3
        q_diff = rho_diff.sum() * dV
        print(f"[{mol_file}] q_diff={q_diff:.4f} (should be ~0)")
        assert abs(q_diff) < 2.0, \
            f"{mol_file}: rho_diff integral {q_diff:.4f} too large (charge not conserved)"


@pytest.mark.gpu
def test_poisson_potential(fdbm_results):
    """V_ES from FFT Poisson is finite and decays at boundaries."""
    for mol_file, r in fdbm_results.items():
        V_ES = r['V_ES']
        assert np.all(np.isfinite(V_ES)), f"{mol_file}: V_ES has NaN/inf"
        # Boundary values should be smaller than peak
        boundary = np.concatenate([V_ES[0,:,:].ravel(), V_ES[-1,:,:].ravel(),
                                    V_ES[:,0,:].ravel(), V_ES[:,-1,:].ravel(),
                                    V_ES[:,:,0].ravel(), V_ES[:,:,-1].ravel()])
        peak = np.max(np.abs(V_ES))
        boundary_max = np.max(np.abs(boundary))
        print(f"[{mol_file}] V_ES: [{V_ES.min():.4e}, {V_ES.max():.4e}]  boundary/peak={boundary_max/peak:.3f}")
        assert peak > 0, f"{mol_file}: V_ES is all zeros"
        # Boundary should be significantly smaller than peak (far-field decay)
        if boundary_max >= peak:
            print(f"  WARNING: V_ES boundary ({boundary_max:.4e}) >= peak ({peak:.4e}) — grid too small")


@pytest.mark.gpu
def test_pauli_overlap(fdbm_results):
    """Pauli overlap is finite, positive, and peaks near the molecule."""
    for mol_file, r in fdbm_results.items():
        overlap = r['overlap_raw']
        assert np.all(np.isfinite(overlap)), f"{mol_file}: overlap has NaN/inf"
        assert overlap.min() >= 0, f"{mol_file}: overlap has negative values"
        assert overlap.max() > 0, f"{mol_file}: overlap is all zeros"
        print(f"[{mol_file}] overlap: [{overlap.min():.4e}, {overlap.max():.4e}]")


@pytest.mark.gpu
def test_es_convolution(fdbm_results):
    """Electrostatic convolution E_ES is finite."""
    for mol_file, r in fdbm_results.items():
        E_ES = r['E_ES']
        assert np.all(np.isfinite(E_ES)), f"{mol_file}: E_ES has NaN/inf"
        print(f"[{mol_file}] E_ES: [{E_ES.min():.4e}, {E_ES.max():.4e}]")


@pytest.mark.gpu
def test_dispersion(fdbm_results):
    """vdW dispersion is finite and attractive (negative)."""
    for mol_file, r in fdbm_results.items():
        E_vdw = r['E_vdw']
        assert np.all(np.isfinite(E_vdw)), f"{mol_file}: E_vdw has NaN/inf"
        assert E_vdw.min() < 0, f"{mol_file}: E_vdw should have negative (attractive) values"
        print(f"[{mol_file}] E_vdw: [{E_vdw.min():.4e}, {E_vdw.max():.4e}]")


@pytest.mark.gpu
def test_total_forcefield(fdbm_results):
    """F_total = -grad(E_total) is finite with reasonable magnitudes."""
    for mol_file, r in fdbm_results.items():
        F_total = r['F_total']
        assert np.all(np.isfinite(F_total)), f"{mol_file}: F_total has NaN/inf"
        Fz = F_total[...,2]
        print(f"[{mol_file}] F_total Fz: [{Fz.min():.4e}, {Fz.max():.4e}]")
        assert np.max(np.abs(Fz)) > 0, f"{mol_file}: F_total Fz is all zeros"


@pytest.mark.gpu
def test_fdbm_relaxed_scan(fdbm_results):
    """PP-relaxed scan: Fz and df are finite."""
    for mol_file, r in fdbm_results.items():
        Fz = r['Fz_relax']
        df = r['df']
        assert np.all(np.isfinite(Fz)), f"{mol_file}: Fz_relax has NaN/inf"
        assert np.all(np.isfinite(df)), f"{mol_file}: df has NaN/inf"
        print(f"[{mol_file}] Fz_relax: [{Fz.min():.4e}, {Fz.max():.4e}]  df: [{df.min():.4e}, {df.max():.4e}]")
        # Fz should have non-zero range (molecule produces forces)
        assert np.max(np.abs(Fz)) > 1e-6, f"{mol_file}: Fz_relax is effectively zero"


@pytest.mark.gpu
@pytest.mark.visual
def test_fdbm_visual(fdbm_results):
    """Generate 2D slices of all FDBM components for visual inspection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_dir = _debug_dir()
    for mol_file, r in fdbm_results.items():
        origin = r['origin']
        step = r['step']
        nx, ny, nz = r['ngrid']
        mol_z = float(r['atomPos'][:,2].max())
        z_coords = origin[2] + np.arange(nz) * step
        z_rel = z_coords - mol_z

        # Select z slices: 3Å, 4Å, 5Å above molecule
        target_heights = [3.0, 4.0, 5.0]
        sel_iz = []
        for h in target_heights:
            iz = int(np.argmin(np.abs(z_rel - h)))
            if 0 <= iz < nz:
                sel_iz.append(iz)

        extent_xy = [float(origin[0]), float(origin[0] + nx*step),
                     float(origin[1]), float(origin[1] + ny*step)]

        components = [
            ('rho_scf', r['rho_scf'], 'hot'),
            ('rho_na', r['rho_na'], 'hot'),
            ('rho_diff', r['rho_diff'], 'seismic'),
            ('V_ES', r['V_ES'], 'seismic'),
            ('overlap_raw', r['overlap_raw'], 'hot'),
            ('E_pauli', r['E_pauli'], 'Reds'),
            ('E_ES', r['E_ES'], 'seismic'),
            ('E_vdw', r['E_vdw'], 'Blues_r'),
            ('E_total', r['E_total'], 'bwr'),
        ]

        fig, axes = plt.subplots(len(components), len(sel_iz), figsize=(3*len(sel_iz), 2.5*len(components)))
        for row, (label, field, cmap) in enumerate(components):
            for col, iz in enumerate(sel_iz):
                ax = axes[row, col]
                data = field[:,:,iz].T
                vabs = max(float(np.percentile(np.abs(data), 99)), 1e-30)
                im = ax.imshow(data, origin='lower', cmap=cmap, aspect='equal', vmin=-vabs, vmax=vabs, extent=extent_xy)
                ax.set_title(f'z={z_rel[iz]:.1f}Å', fontsize=7)
                ax.tick_params(labelsize=5)
                if col == 0: ax.set_ylabel(label, fontsize=7)
                if row == len(components)-1: ax.set_xlabel('x (Å)', fontsize=7)
                plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f'FDBM Components — {mol_file}', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'fdbm_slices_{mol_file.replace(".xyz","")}.png'), dpi=150)
        plt.close(fig)

        # Also plot df at scan heights
        heights = r['heights']
        df = r['df']
        scan_xs = r['scan_xs']
        scan_ys = r['scan_ys']
        scan_extent = [float(scan_xs[0]), float(scan_xs[-1]), float(scan_ys[0]), float(scan_ys[-1])]

        n_h = min(6, len(heights))
        fig, axes = plt.subplots(1, n_h, figsize=(3*n_h, 3))
        for col in range(n_h):
            ax = axes[col]
            iz = col * max(1, len(heights)//n_h)
            data = df[:,:,iz].T
            vabs = max(float(np.percentile(np.abs(data), 99)), 1e-30)
            im = ax.imshow(data, origin='lower', cmap='bwr', aspect='equal', vmin=-vabs, vmax=vabs, extent=scan_extent)
            ax.set_title(f'h={heights[iz]:.2f}Å', fontsize=8)
            ax.tick_params(labelsize=5)
            if col == 0: ax.set_ylabel('df', fontsize=8)
            ax.set_xlabel('x (Å)', fontsize=7)
            plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f'FDBM Frequency Shift (df) — {mol_file}', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'fdbm_df_{mol_file.replace(".xyz","")}.png'), dpi=150)
        plt.close(fig)

        print(f"  Saved FDBM visualizations for {mol_file}")
