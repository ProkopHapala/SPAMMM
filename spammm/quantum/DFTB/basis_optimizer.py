"""basis_optimizer.py — Fit single-exponential Slater-tail basis to reference electron density.

Uses simulated annealing to optimize (amplitude N, decay zeta) per element,
matching log-density profiles in the 0.5–1.5 Å fit region above each atom.
Evaluated via **project_density_dense_points** kernel at z-profile points only,
avoiding full 3D grid projection during optimization iterations.
"""
import time, numpy as np

BOHR2ANG = 0.5291772109
B3_FACTOR = 1.0 / (BOHR2ANG**3)

# Default decay constants (1/Å) from GPAW/PySCF density profile fits
SLATER_TAIL_ZETA_DEFAULT = {'H': 2.42, 'C': 2.78, 'N': 3.00, 'O': 3.25}


def make_single_exponent_species_list(species_list_ang, params, cutoff=6.0):
    """Build species_list with single-exponential STOs from params dict.

    Args:
        species_list_ang: original species_list from convert_wfc_to_species_list_ang()
        params: {elem_name: (N, zeta)} — amplitude and decay constant per element
        cutoff: extended cutoff in Angstrom

    Returns:
        new species_list with single-exponential STOs
    """
    new_list = []
    for sp in species_list_ang:
        name = sp['name']
        N, z = params.get(name, (1.0, 2.5))
        new_orbitals = []
        for orb in sp['orbitals']:
            new_orbitals.append({
                'l': orb['l'],
                'cutoff': cutoff,
                'exponents': np.array([z]),
                'coefficients': np.array([[N]]),
            })
        new_list.append({
            'name': sp['name'],
            'atomic_number': sp['atomic_number'],
            'orbitals': new_orbitals,
            'resolution': sp['resolution'],
        })
    return new_list


def amplitude_match_params(species_list_ang, zeta_map=None, r_match=0.7):
    """Compute initial (N, zeta) params by matching original basis amplitude at r_match.

    Args:
        species_list_ang: original species_list
        zeta_map: {elem: zeta} decay constants; uses SLATER_TAIL_ZETA_DEFAULT if None
        r_match: matching distance in Angstrom

    Returns:
        {elem: [N, zeta]} dict
    """
    from .DFTBplusParser import compute_sto_radial
    zeta_map = zeta_map or SLATER_TAIL_ZETA_DEFAULT
    params = {}
    for sp in species_list_ang:
        name = sp['name']
        z = zeta_map.get(name, 2.5)
        orb0 = sp['orbitals'][0]
        orig_val = float(compute_sto_radial(np.array([r_match]), orb0['coefficients'], orb0['exponents'], orb0['l'])[0])
        slater_raw = r_match**orb0['l'] * np.exp(-z * r_match)
        N = orig_val / slater_raw if abs(slater_raw) > 1e-30 else 1.0
        params[name] = [N, z]
    return params


def build_z_profile_points(atoms, z0, z_max=3.0, dz=0.1):
    """Build (n_atoms*n_z, 3) points above each atom in z direction.

    Args:
        atoms: list of (idx, sym, x, y, z) tuples
        z0: molecular plane z-coordinate
        z_max: max height above plane
        dz: z spacing

    Returns:
        points: (n_atoms*n_z, 3) float32 array
        z_vals: (n_z,) array of relative z values
    """
    z_vals = np.arange(0, z_max + dz/2, dz)
    points = []
    for idx, sym, x, y, z in atoms:
        for zv in z_vals:
            points.append([x, y, z0 + zv])
    return np.array(points, dtype=np.float32), z_vals


def extract_z_profiles(rho_3d, atoms, origin, step, ngrid, z0, z_max=3.0, dz=0.1):
    """Extract 1D density profiles above each atom from a 3D density grid.

    Args:
        rho_3d: (nx, ny, nz) density array
        atoms: list of (idx, sym, x, y, z) tuples
        origin: (3,) grid origin in Angstrom
        step: grid spacing in Angstrom (scalar or (3,))
        ngrid: (3,) grid dimensions
        z0: molecular plane z-coordinate
        z_max, dz: profile parameters

    Returns:
        profiles: (n_atoms, n_z) array of density values
        z_vals: (n_z,) array of relative z values
    """
    step = np.atleast_1d(step)
    if len(step) == 1: step = np.array([step[0]]*3)
    z_vals = np.arange(0, z_max + dz/2, dz)
    iz0 = int(round((z0 - origin[2]) / step[2]))
    profiles = []
    for idx, sym, x, y, z in atoms:
        ix = int(round((x - origin[0]) / step[0]))
        iy = int(round((y - origin[1]) / step[1]))
        prof = []
        for zv in z_vals:
            iz = iz0 + int(round(zv / step[2]))
            if 0 <= iz < ngrid[2] and 0 <= ix < ngrid[0] and 0 <= iy < ngrid[1]:
                prof.append(max(float(rho_3d[ix, iy, iz]), 1e-10))
            else:
                prof.append(1e-10)
        profiles.append(np.array(prof))
    return np.array(profiles), z_vals


def objective_log_mse(rho_pred, rho_ref, z_vals, fit_lo=0.5, fit_hi=1.5):
    """Log-density MSE in the fit region.

    Args:
        rho_pred: (n_z,) predicted density
        rho_ref: (n_z,) reference density
        z_vals: (n_z,) z coordinates
        fit_lo, fit_hi: fit region bounds

    Returns:
        float: mean squared error of log(rho) in fit region
    """
    mask = (z_vals >= fit_lo) & (z_vals <= fit_hi)
    log_pred = np.log(np.maximum(rho_pred[mask], 1e-10))
    log_ref = np.log(np.maximum(rho_ref[mask], 1e-10))
    return np.mean((log_pred - log_ref)**2)


def optimize_basis_sa(projector, dm_dense, norb_per_atom, orb_offsets, atoms_dict,
                      points, ref_profiles, z_vals, initial_params,
                      species_list_ang, n_iter=2000, T0=0.5, T_min=0.001,
                      step_N=2.0, step_z=0.3, fit_lo=0.5, fit_hi=1.5,
                      seed=42, verbosity=1):
    """Optimize (N, zeta) per element using simulated annealing.

    Uses projector.update_basis_sto() for fast basis updates without kernel recompilation.

    Args:
        projector: initialized GridProjector with basis already loaded
        dm_dense: (norb_total, norb_total) dense density matrix
        norb_per_atom: (natoms,) orbital counts
        orb_offsets: (natoms+1,) cumulative offsets
        atoms_dict: dict with 'pos', 'Rcut', 'type'
        points: (n_points, 3) evaluation points (float32)
        ref_profiles: (n_atoms, n_z) reference density profiles
        z_vals: (n_z,) z coordinates
        initial_params: {elem: [N, zeta]} starting parameters
        species_list_ang: original species_list for make_single_exponent_species_list
        n_iter: SA iterations
        T0, T_min: initial/final temperature
        step_N, step_z: perturbation step sizes
        fit_lo, fit_hi: objective fit region
        seed: random seed
        verbosity: 0=silent, 1=progress, 2=detailed

    Returns:
        best_params: {elem: [N, zeta]} optimized parameters
        best_obj: final objective value
        history: list of (iteration, current_obj, best_obj)
    """
    rng = np.random.RandomState(seed)
    elems = list(initial_params.keys())
    n_atoms = ref_profiles.shape[0]

    def eval_with_params(params):
        species_list = make_single_exponent_species_list(species_list_ang, params)
        projector.update_basis_sto(species_list)
        rho_pts = projector.project_density_dense_points(
            points, dm_dense, norb_per_atom, orb_offsets, atoms_dict)
        n_z = len(z_vals)
        return rho_pts.reshape(n_atoms, n_z) * B3_FACTOR

    def total_obj(params):
        rho = eval_with_params(params)
        return sum(objective_log_mse(rho[i], ref_profiles[i], z_vals, fit_lo, fit_hi)
                   for i in range(n_atoms))

    current_params = {k: list(v) for k, v in initial_params.items()}
    current_obj = total_obj(current_params)
    best_params = {k: list(v) for k, v in current_params.items()}
    best_obj = current_obj
    history = [(0, current_obj, best_obj)]
    T = T0
    t0 = time.time()

    if verbosity >= 1:
        print(f"[SA] Starting: obj={current_obj:.6f}, params={best_params}")

    for it in range(n_iter):
        elem = rng.choice(elems)
        param_idx = rng.randint(2)
        trial_params = {k: list(v) for k, v in current_params.items()}
        if param_idx == 0:
            trial_params[elem][0] *= (1.0 + rng.randn() * step_N)
        else:
            trial_params[elem][1] += rng.randn() * step_z
            trial_params[elem][1] = max(0.5, trial_params[elem][1])

        try:
            trial_obj = total_obj(trial_params)
        except Exception:
            continue

        delta = trial_obj - current_obj
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_params = trial_params
            current_obj = trial_obj
            if trial_obj < best_obj:
                best_obj = trial_obj
                best_params = {k: list(v) for k, v in trial_params.items()}
                if verbosity >= 1:
                    print(f"[SA] it={it:4d} T={T:.4f} obj={trial_obj:.6f} BEST! {best_params}")

        T = max(T_min, T0 * (1.0 - it / n_iter))
        history.append((it+1, current_obj, best_obj))

        if verbosity >= 1 and it % 200 == 0:
            elapsed = time.time() - t0
            print(f"[SA] it={it:4d} T={T:.4f} cur={current_obj:.6f} best={best_obj:.6f} [{elapsed:.1f}s]")

    if verbosity >= 1:
        print(f"[SA] Done in {time.time()-t0:.1f}s, best_obj={best_obj:.6f}")
        print(f"[SA] Best params: {best_params}")

    return best_params, best_obj, history
