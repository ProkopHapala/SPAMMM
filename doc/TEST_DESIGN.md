# Test Design Document for SPAMMM

## Philosophy

Two tiers: **automatic** (machine pass/fail) and **visual** (plots/XYZ for human). Both share fixtures and helper functions. **Few modules, not many files.** Template functions + data-driven cases keep it compact.

---

## File Structure

```
tests/
  conftest.py              # fixtures: data paths, molecule loader, substrate loader
  test_topology.py          # Suite 1: bond/angle/hybridization/type assignment
  test_forcefield.py        # Suite 2: UFF/SPFF optimization, NVE conservation
  test_surface.py           # Suite 3+4+5: Ewald vs brute, GridFF, folded function
  test_afm.py               # Suite 6: AFM probe relaxation + imaging
  test_integration.py       # Suite 7: relaxed scan (molecule on substrate)
  helpers/
    __init__.py
    parity.py               # rmse, correlation, direction_cosine, overlay_plot
    geometry.py             # bond_lengths, angles, planarity, distort
    scan.py                 # 1D scan runner (z-scan, x-scan), overlay plot
```

6 test files + 3 helper modules. That's it.

---

## Fixtures (`conftest.py`)

```python
import os, pytest, numpy as np
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')

@pytest.fixture
def xyz():     return lambda n: os.path.join(DATA, 'xyz', n)
@pytest.fixture
def substrate(): return lambda n: os.path.join(DATA, 'substrates', n)
@pytest.fixture
def dat():     return lambda n: os.path.join(DATA, n)
```

---

## Helpers (`tests/helpers/`)

### `parity.py` — comparison + plotting templates

```python
import numpy as np, matplotlib.pyplot as plt, os

def rmse(a, b): return np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2))
def correlation(a, b): return np.corrcoef(a, b)[0, 1]
def max_err(a, b): return np.max(np.abs(np.asarray(a) - np.asarray(b)))
def dir_cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)

def overlay_plot(x, curves, labels, title, xlabel, savepath=None, show_rmse=True):
    """curves: list of arrays. labels: list of str. First curve = reference (solid)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (c, l) in enumerate(zip(curves, labels)):
        ax.plot(x, c, '--' if i > 0 else '-', label=l)
    ax.set_xlabel(xlabel); ax.set_ylabel('Energy [eV]'); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    if show_rmse and len(curves) >= 2:
        ax.text(0.02, 0.98, f'RMSE={rmse(curves[0], curves[1]):.2e}\nMax={max_err(curves[0], curves[1]):.2e}',
                transform=ax.transAxes, va='top', fontsize=9, family='monospace',
                bbox=dict(facecolor='white', alpha=0.8))
    if savepath: fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig

def assert_parity(ref, test, rtol=1e-3, atol=1e-5, name=''):
    """Standard parity assertion with informative message."""
    r, m = rmse(ref, test), max_err(ref, test)
    assert r < rtol, f'{name}: RMSE={r:.2e} > {rtol:.0e}'
    assert m < atol * 10, f'{name}: MaxErr={m:.2e} > {atol*10:.0e}'
```

### `geometry.py` — geometry checks + distortion

```python
import numpy as np

def bond_lengths(apos, bonds): return [np.linalg.norm(apos[i] - apos[j]) for i, j in bonds]
def bond_angle(apos, i, j, k):
    a, b = apos[i] - apos[j], apos[k] - apos[j]
    return np.degrees(np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
def planarity(apos, indices):
    p = apos[indices]; c = p.mean(axis=0); _, _, w = np.linalg.svd(p - c)
    return w[2]  # smallest singular value = out-of-plane thickness

def distort(apos, amplitude=0.2, seed=42):
    rng = np.random.default_rng(seed); return apos + rng.normal(0, amplitude, apos.shape)

def check_geometry(apos, bonds, expected_bonds, angle_tolerance=10.0):
    """Returns dict of pass/fail checks. expected_bonds: { (i,j): (r0, tol_frac) }"""
    results = {}
    bls = bond_lengths(apos, bonds)
    for (i, j), (r0, tol) in expected_bonds.items():
        r = np.linalg.norm(apos[i] - apos[j])
        results[f'bond_{i}_{j}'] = abs(r - r0) < r0 * tol
    return results
```

### `scan.py` — 1D scan runner

```python
import numpy as np, os
from .parity import overlay_plot, rmse, correlation

def z_scan(eval_func, x0, y0, z_range):
    """eval_func(x, y, z) -> float. Returns phi(z) array."""
    return np.array([eval_func(x0, y0, z) for z in z_range])

def x_scan(eval_func, y0, z0, x_range):
    return np.array([eval_func(x, y0, z0) for x in x_range])

def compare_scans(scan_name, coord, ref, test, ref_label, test_label, save_dir=None):
    """Compare two 1D scans. Returns dict with rmse, corr, pass/fail. Optionally plots."""
    r, c = rmse(ref, test), correlation(ref, test)
    result = {'name': scan_name, 'rmse': r, 'correlation': c, 'pass': r < 0.01 and c > 0.999}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        overlay_plot(coord, [ref, test], [ref_label, test_label],
                     f'{scan_name} (RMSE={r:.2e}, r={c:.4f})',
                     'z [A]' if 'z' in scan_name else 'x [A]',
                     os.path.join(save_dir, f'{scan_name}.png'))
    return result
```

---

## Test Modules

### `test_topology.py` — Suite 1

Data-driven: one parametrized test function covers all molecules.

```python
import pytest, numpy as np
from spammm.topology.FFparams import read_element_types, read_atom_types, make_REQs_from_enames

# (filename, expected_nbonds, expected_hybridization, expected_types)
CASES = [
    ('H2O.xyz',  2, {'O': 'sp3'},                 {'O': 'O_3', 'H': 'H'}),
    ('CH4.xyz',  4, {'C': 'sp3'},                 {'C': 'C_3', 'H': 'H'}),
    ('benzene.xyz', 12, {'C': 'sp2'},             {'C': 'C_R', 'H': 'H'}),
    ('HCOOH.xyz', 4, {'C': 'sp2'},                {'C': 'C_2', 'O': 'O_2'}),
    ('CO.xyz',   2, {'C': 'sp', 'O': 'sp'},       {'C': 'C_1', 'O': 'O_1'}),
]

@pytest.mark.parametrize('fname,nbonds,hyb,types', CASES)
def test_topology(xyz, dat, fname, nbonds, hyb, types):
    from spammm.AtomicSystem import AtomicSystem
    mol = AtomicSystem(fname=xyz(fname))
    assert mol.nbonds == nbonds, f'{fname}: {mol.nbonds} != {nbonds}'
    # Check hybridization and atom types via FFparams
    etypes = read_element_types(dat('ElementTypes.dat'))
    at = read_atom_types(dat('AtomTypes.dat'), etypes)
    REQs = make_REQs_from_enames(mol.enames, mol.qs, at)
    assert np.all(np.isfinite(REQs)), f'{fname}: NaN in REQs'
    # Check specific type assignments
    for i, name in enumerate(mol.enames):
        if name in types:
            assert types[name] in at, f'{name}: type {types[name]} not in AtomTypes.dat'

def test_pi_system_benzene(xyz):
    from spammm.AtomicSystem import AtomicSystem
    mol = AtomicSystem(fname=xyz('benzene.xyz'))
    # After topology build, check pi orbitals exist and are normal to plane
    # (Exact API depends on SPFF.toSPFFsp3_loc internals)
    assert mol.natoms == 12
```

### `test_forcefield.py` — Suite 2

Template function for relaxation, parametrized over force field + molecule.

```python
import pytest, numpy as np
from tests.helpers.geometry import distort, check_geometry
from tests.helpers.parity import overlay_plot, assert_parity

# (molecule, forcefield, nsteps, expected_bonds, angle_tol)
RELAX_CASES = [
    ('H2O.xyz',     'UFF', 100, {(0,1): (0.96, 0.1), (0,2): (0.96, 0.1)}, 10),
    ('CH4.xyz',     'UFF', 100, {(0,1): (1.09, 0.1), (0,2): (1.09, 0.1)}, 10),
    ('benzene.xyz', 'UFF', 200, {(0,1): (1.40, 0.1)}, 10),
    ('H2O.xyz',     'SPFF', 100, {(0,1): (0.96, 0.1), (0,2): (0.96, 0.1)}, 10),
    ('benzene.xyz', 'SPFF', 200, {(0,1): (1.40, 0.1)}, 10),
]

@pytest.mark.gpu
@pytest.mark.parametrize('mol_file,ff,nsteps,expected,atol', RELAX_CASES)
def test_relax(xyz, mol_file, ff, nsteps, expected, atol):
    from spammm.AtomicSystem import AtomicSystem
    from spammm.forcefields.MolecularDynamics import MolecularDynamics
    mol = AtomicSystem(fname=xyz(mol_file))
    mol.apos = distort(mol.apos, 0.2)
    md = MolecularDynamics()
    # ... setup ff, pack system, run relax ...
    # Check forces converged
    # Check geometry via check_geometry()
    pass  # implementation depends on exact API

@pytest.mark.gpu
def test_nve_conservation(xyz):
    """Energy conservation: methane, NVE 1000 steps, dE/E < 1e-4."""
    # Load CH4, setup UFF, run velocity Verlet with dt=0.5fs, no damping
    # Record E_total every 10 steps, check drift
    pass
```

### `test_surface.py` — Suite 3+4+5 (the big one)

This is the core module. Three test functions cover Ewald parity, Ewald vs brute, and GridFF vs Ewald.

```python
import pytest, numpy as np
from tests.helpers.parity import rmse, correlation, overlay_plot, assert_parity
from tests.helpers.scan import z_scan, x_scan, compare_scans

# ---- Suite 3: Ewald parity ----

@pytest.mark.gpu
def test_ewald_py_vs_cl(substrate, dat):
    """Ewald2D (NumPy) vs SurfaceEwaldCL (OpenCL) on NaCl 1x1."""
    from spammm.surfaces.Ewald2D import Ewald2D
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    # Load NaCl 1x1, build both, compare vacuum + full 1D scan
    # assert_parity(phi_py, phi_cl, rtol=1e-3, name='ewald_py_vs_cl')

@pytest.mark.gpu
def test_ewald_vs_brute_zscan(substrate):
    """Ewald vs GPU brute-force: z-scan above NaCl 8x8."""
    from spammm.surfaces.SurfaceEwald import SurfaceEwaldCL
    ew = SurfaceEwaldCL()
    # Load NaCl 8x8, prepare system
    z_range = np.linspace(1.0, 6.0, 50)
    # phi_ewald = ew.eval_full(...)  along z at (0.05, 0.05)
    # phi_brute = ew.eval_brute(..., N_rep=20)  same points
    # result = compare_scans('z_on_Na', z_range, phi_ewald, phi_brute,
    #                        'Ewald', 'Brute', save_dir='debug/ewald')
    # assert result['pass']

@pytest.mark.gpu
def test_ewald_vs_brute_xscan(substrate):
    """Ewald vs GPU brute-force: x-scan at z=3 A over NaCl 8x8."""
    # Same as above but x_range = np.linspace(0, 8, 50), z=3.0
    pass

def test_brute_convergence(substrate):
    """Brute-force convergence: phi(N_rep=10) vs phi(N_rep=20) < 0.001 eV."""
    pass

# ---- Suite 4: GridFF ----

@pytest.mark.gpu
def test_gridff_vs_ewald_coulomb(substrate):
    """GridFF (Coulomb channel) vs Ewald2D: z-scan + x-scan."""
    # Build GridFF with Q=1 only, sample along scans, compare with Ewald2D
    pass

@pytest.mark.gpu
def test_gridff_bspline_vs_direct(substrate):
    """B-spline interpolation at grid points == direct evaluation."""
    pass

# ---- Suite 5: Folded function ----

@pytest.mark.gpu
def test_folded_vs_brute(substrate):
    """Folded atomic function vs brute-force: z-scan + x-scan."""
    pass

# ---- Visual outputs ----

@pytest.mark.visual
@pytest.mark.gpu
def test_visual_ewald_scans(substrate):
    """Produce overlay plots: Ewald vs brute, z-scan + x-scan, 3 locations."""
    pass
```

### `test_afm.py` — Suite 6

```python
import pytest, numpy as np

@pytest.mark.gpu
def test_afm_relax_convergence(xyz, substrate):
    """AFM probe relaxation on NaCl: finite forces, no NaN, image produced."""
    from spammm.spm.AFM import AFMulator
    # Load NaCl + CO tip, run relaxation, check output
    pass

@pytest.mark.visual
@pytest.mark.gpu
def test_visual_afm_images(xyz, substrate):
    """2D AFM images at 3 z heights for NaCl and benzene."""
    pass
```

### `test_integration.py` — Suite 7

```python
import pytest, numpy as np

@pytest.mark.gpu
@pytest.mark.slow
def test_relaxed_scan_water_nacl(xyz, substrate):
    """H2O on NaCl: relaxed scan z=[2,6] A, 20 pts. Smooth curve, min at 2.5-3.5 A."""
    pass

@pytest.mark.gpu
@pytest.mark.slow
def test_relaxed_scan_benzene_nacl(xyz, substrate):
    """Benzene on NaCl: relaxed scan. Min at 3.2-3.8 A."""
    pass

@pytest.mark.visual
@pytest.mark.gpu
@pytest.mark.slow
def test_visual_relaxed_scan(xyz, substrate):
    """Plot: E(z) rigid vs relaxed for water and benzene on NaCl."""
    pass
```

---

## Kernel Organization Note

### Current state
- `gridFF.cl` — tricubic B-spline only (`fe3d_pbc_comb`, 4x4x4 stencil, C2 continuous). Requires pre-fitting.
- `AFM.cl` — `interpFE` uses OpenCL hardware trilinear via `read_imagef` (texture, C0 continuous, no fitting).

### Proposed: split into two GridFF variants

1. **`gridFF.cl`** (keep): Tricubic B-spline. Slow, accurate. For production MD / relaxed scans.
2. **`gridFF_trilinear.cl`** (new): Trilinear (E, Fx, Fy, Fz) at grid nodes. No fitting needed. Fast. For AFM / approximate sampling.

Both composable with `common.cl` + `Forces.cl`.

3. **`surface.cl`** (keep): Folded atomic function / Ewald. No grid, fast. For surface electrostatics.

### Updated composition rules
- **GridFF B-spline**: `common.cl` + `Forces.cl` + `gridFF.cl`
- **GridFF trilinear**: `common.cl` + `Forces.cl` + `gridFF_trilinear.cl`
- **Surface Ewald**: `common.cl` + `Forces.cl` + `surface.cl`
- **AFM**: `common.cl` + `Forces.cl` + `AFM.cl` (+ `gridFF_trilinear.cl` if using precomputed field)

---

## Implementation Priority

| Priority | Module | What | Why |
|----------|--------|------|-----|
| 1 | `test_topology.py` | Topology + type assignment | Foundation — wrong topology = wrong everything |
| 2 | `test_surface.py` | Ewald vs brute-force | User's key request. Existing code in `SurfaceEwald.py` + `Ewald2D.py` |
| 3 | `test_forcefield.py` | UFF/SPFF relaxation | Validates force field + MD integrator |
| 4 | `test_surface.py` (GridFF part) | GridFF vs Ewald parity | Depends on correct Ewald |
| 5 | `test_afm.py` | AFM relaxation | End-to-end, depends on GridFF |
| 6 | `test_integration.py` | Relaxed scan | Full pipeline |

---

## pytest Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
markers =
    slow: marks tests as slow
    gpu: marks tests requiring GPU
    visual: marks tests producing plots for human review
addopts = -v --tb=short
```

```bash
pytest -m "not slow and not gpu"   # fast, no GPU
pytest -m "gpu"                    # GPU tests only
pytest -m "visual"                 # visual output only
pytest                             # everything
```

---

## Artifacts

```
debug/<date>_<suite>/
  ├── zscan_ewald_vs_brute.png
  ├── xscan_ewald_vs_brute.png
  ├── relax_energy_vs_step.png
  ├── water_relaxed.xyz
  └── REPORT.md
```

Plots include RMSE, max deviation, scan parameters in captions. XYZ files include energy and force norms in comment lines.
