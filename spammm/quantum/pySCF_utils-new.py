"""
pySCF_utils-new.py — pySCF integration for FDBM density / CO z-scans (SSOT).

Purpose: Run pySCF quantum chemistry (RHF, DFT) for SPAMMM AFM/FDBM.
Prefers the local optimized fork at ``/home/prokop/git/pyscf`` (GPU OpenCL XC);
falls back to stock CPU pySCF when OpenCL/NVIDIA is unavailable.

Key functionality:
  - ensure_local_pyscf() / gpu_opencl_available() / resolve_backend()
  - make_rks() — RKS+DF with backend=gpu|smalldft|cpu
  - run_co_zscan() — **SSOT** site-correct rigid CO z-scan (DM warm-start along z)
  - make_z_grid() — non-uniform distance array (change only this for grid)
  - write_frontier_mo_cubes() / eval_mo_on_xy_slice() — HOMO/LUMO for STM refs

CLI entry (one script): ``tests/SPM/run_zscan_reference.py`` — do not fork ad-hoc scan drivers.
Benchmark cousin in the pySCF fork: ``expamples_prokop/profile_dimer_scan.py`` (dimer E(z)
path comparison); AFM CO refs go through this module + run_zscan_reference.

Open issues / caveats:
  - Tip xy = target atom xy (NOT lab origin).
  - DF ``_cderi`` must rebuild each geometry (nuclear positions); DM warm-start still helps cycles.
  - GPU AFM tols: conv_tol=1e-6, conv_tol_grad=1e-4 (below f32 XC noise).
  - Legacy ``pySCF_utils.py`` kept for parity — do not overwrite; import this file via importlib
    (hyphen in filename) or rename later with USER approval.
  - Shell for GPU: unrestricted (`all`) so NVIDIA ICD is visible — see opencl-nvidia-gpu rule.
  - MO projection is **full GTO** via ``pyscf.dft.numint.eval_ao`` + ``mo_coeff`` (all shells of
    def2-SVP etc.). No AO dropping / density truncation. OpenCL ``LCAO_grid.cl`` is DFTB STO only.
"""

import sys
import os
import time
import numpy as np

LOCAL_PYSCF = os.environ.get('SPAMMM_PYSCF_ROOT', '/home/prokop/git/pyscf')
hartree2eV = 27.211396641308
bohr2A = 0.5291772109038
HAU2EV = hartree2eV
verbosity = 0
default_conv_params = {
    'gradientmax': 0.45e-6,
    'gradientrms': 0.15e-6,
    'stepmax': 1.8e-3,
    'steprms': 1.2e-3,
}
CO_BOND = 1.13
DEFAULT_GPU_PROFILE = 'production_radial_screened_splitk'


def make_z_grid(z_max=15.0):
    """Non-uniform CO-scan z-grid — the distance array to pass into run_co_zscan.

    1.5..3.0 dz=0.1, 3.0..5.0 dz=0.2, 5.0..8.0 dz=0.5, 8.0..z_max dz=1.0
    """
    z1 = np.arange(1.5, 3.0, 0.1)
    z2 = np.arange(3.0, 5.0, 0.2)
    z3 = np.arange(5.0, 8.01, 0.5)
    z4 = np.arange(8.0, float(z_max) + 0.01, 1.0) if z_max > 8.0 else np.array([])
    return np.unique(np.round(np.concatenate([z1, z2, z3, z4]), 4))


def ensure_local_pyscf(root=None):
    """Prepend local pySCF fork to sys.path if it exists. Returns root or None."""
    root = root or LOCAL_PYSCF
    if root and os.path.isdir(os.path.join(root, 'pyscf')):
        if root not in sys.path:
            sys.path.insert(0, root)
        return root
    return None


ensure_local_pyscf()
from pyscf import gto, dft, lib  # noqa: E402

try:
    from pyscf.geomopt.berny_solver import optimize
except ImportError:
    optimize = None

try:
    from . import atomicUtils as au
except ImportError:
    from spammm import atomicUtils as au


def gpu_opencl_available():
    """True if local-fork OpenCL + an NVIDIA device are usable."""
    try:
        from pyscf.OpenCL.gpu_profiles import apply_gpu_profile  # noqa: F401
        import pyopencl as cl
        for p in cl.get_platforms():
            blob = (p.name + ' ' + p.vendor).lower()
            if 'nvidia' in blob:
                return True
            for d in p.get_devices():
                if 'nvidia' in (d.name + ' ' + d.vendor).lower():
                    return True
    except Exception:
        return False
    return False


def resolve_backend(prefer='auto'):
    """prefer: 'auto'|'gpu'|'cpu'|'smalldft'. auto → gpu if NVIDIA OpenCL else cpu."""
    prefer = (prefer or 'auto').lower()
    if prefer == 'auto':
        return 'gpu' if gpu_opencl_available() else 'cpu'
    if prefer in ('gpu', 'cpu', 'smalldft'):
        if prefer == 'gpu' and not gpu_opencl_available():
            print('[pySCF] GPU requested but OpenCL/NVIDIA unavailable → falling back to stock CPU')
            return 'cpu'
        return prefer
    raise ValueError(f'prefer_backend must be auto|gpu|cpu|smalldft, got {prefer!r}')


def unpack_mol(mol, units=bohr2A):
    apos = np.array([a[1] for a in mol._atom]) * units
    es = np.array([a[0] for a in mol._atom])
    return apos, es


def pack_mol(apos, es):
    return [(es[i], apos[i]) for i in range(len(es))]


def printlist(lst):
    for item in lst:
        print(item)


def printObj(obj):
    printlist(dir(obj))


def saveAtoms(fname, atoms, unit=bohr2A):
    apos = np.array([a[1] for a in atoms]) * unit
    es = [a[0] for a in atoms]
    au.saveXYZ(es, apos, fname)


def preparemol(fname='relaxed.xyz', conv_params=None, atoms='O 0 0 0; H 1 0 0; H 0 1 0'):
    if conv_params is None:
        conv_params = default_conv_params
    if os.path.isfile(fname):
        print("found(%s) => no need for relaxation " % fname)
        mol = gto.M(atom=fname)
    else:
        h2o = gto.M(atom=atoms)
        h2o.verbose = verbosity
        from pyscf import scf
        calc = scf.RHF(h2o)
        mol = optimize(calc, maxsteps=1000, **conv_params)
        saveAtoms(fname, mol._atom)
    return mol


def evalHf(inp, params=None):
    apos, es = inp
    from pyscf import scf
    mol = gto.M(atom=pack_mol(apos, es))
    mol.verbose = verbosity
    out = scf.UHF(mol).run()
    return out.e_tot * hartree2eV


def optHf(atoms, conv_params=None):
    if conv_params is None:
        conv_params = default_conv_params
    print(atoms)
    from pyscf import scf
    job = gto.M(atom=atoms)
    job.SCF_max_cycle = 100
    job.verbose = verbosity
    calc = scf.RHF(job)
    mol = optimize(calc, maxsteps=1000, **conv_params)
    printlist(mol)
    return mol


def _atom_str(names, pos):
    return '\n'.join(f'{e} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}' for e, p in zip(names, pos))


def make_rks(atom, basis='def2-SVP', xc='PBE', backend='gpu', profile=None,
             grid_level=None, verbose=0, df_storage='incore', n_threads=None,
             max_memory_mb=16000, conv_tol=None, conv_tol_grad=None, max_cycle=None):
    """Build RKS+DF mean-field with GPU OpenCL, smallDFT, or stock CPU.

    backend:
      'gpu'      — pyscf.OpenCL via apply_gpu_profile (NVIDIA XC; CPU DF-J ∥ XC)
      'smalldft' — pyscf.smallDFT CPU OpenMP XC
      'cpu'      — stock PySCF libxc

    GPU AFM defaults (below f32 XC noise): conv_tol=1e-6, conv_tol_grad=1e-4, max_cycle=40.
    Default GPU profile: production_radial_screened_splitk.
    """
    if profile is None:
        profile = DEFAULT_GPU_PROFILE
    if n_threads is not None:
        lib.num_threads(int(n_threads))
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=verbose)
    mol.max_memory = max_memory_mb
    mf = dft.RKS(mol, xc=xc).density_fit()
    mf.max_memory = max_memory_mb
    if mf.with_df is not None:
        mf.with_df.max_memory = max_memory_mb
    if grid_level is not None:
        mf.grids.level = grid_level

    if backend == 'gpu':
        from pyscf.OpenCL.gpu_profiles import apply_gpu_profile
        apply_gpu_profile(mf, profile, df_storage=df_storage, require_df_incore=(df_storage == 'incore'))
        mf.conv_tol = 1e-6 if conv_tol is None else conv_tol
        mf.conv_tol_grad = 1e-4 if conv_tol_grad is None else conv_tol_grad
        mf.max_cycle = 40 if max_cycle is None else max_cycle
    elif backend == 'smalldft':
        from pyscf.smallDFT import prepare_smalldft_for_scf, has_c_lib
        if not has_c_lib():
            raise RuntimeError('libsmalldft.so missing/stale — run pyscf/lib/smalldft/build.sh')
        prepare_smalldft_for_scf(mf, storage=df_storage, max_memory_mb=max_memory_mb)
        if conv_tol is not None:
            mf.conv_tol = conv_tol
        if conv_tol_grad is not None:
            mf.conv_tol_grad = conv_tol_grad
        if max_cycle is not None:
            mf.max_cycle = max_cycle
    elif backend == 'cpu':
        if df_storage:
            mf.with_df.storage = df_storage
        if conv_tol is not None:
            mf.conv_tol = conv_tol
        if conv_tol_grad is not None:
            mf.conv_tol_grad = conv_tol_grad
        if max_cycle is not None:
            mf.max_cycle = max_cycle
    else:
        raise ValueError(f'unknown backend={backend!r}; use gpu|smalldft|cpu')

    mf._spammm_backend = backend
    mf._spammm_profile = profile if backend == 'gpu' else None
    return mf


def release_scf(mf):
    if mf is None:
        return None
    for attr in ('_xc_gpu_plan', '_gpu_xc_path', '_gpu_profile_name'):
        try:
            if hasattr(mf, attr):
                delattr(mf, attr)
        except Exception:
            pass
    df = getattr(mf, 'with_df', None)
    if df is not None:
        for attr in ('_cderi', '_cderi_to_save', '_gpu_df_plan'):
            try:
                if hasattr(df, attr):
                    setattr(df, attr, None)
            except Exception:
                pass
    try:
        from pyscf.OpenCL.xc_grid import clear_xc_plan_cache
        clear_xc_plan_cache()
    except Exception:
        pass
    return None


def run_scf(mf, dm0=None):
    t0 = time.time()
    e = mf.kernel(dm0=dm0) if dm0 is not None else mf.kernel()
    dt = time.time() - t0
    dm = mf.make_rdm1()
    n_cycles = getattr(mf, 'cycles', None)
    return float(e), dm, n_cycles, dt


def run_scf_geometry(names, pos, basis='def2-SVP', xc='PBE', backend='auto', profile=None,
                     dm0=None, release=True, **kw):
    backend = resolve_backend(backend)
    if profile is None:
        profile = DEFAULT_GPU_PROFILE
    mf = make_rks(_atom_str(names, pos), basis=basis, xc=xc, backend=backend, profile=profile, **kw)
    e_ha, dm, n_cycles, dt = run_scf(mf, dm0=dm0)
    out = {
        'E_Ha': e_ha, 'E_eV': e_ha * HAU2EV, 'dm': dm, 'cycles': n_cycles, 'wall_s': dt,
        'mf': mf, 'mol': mf.mol, 'backend': backend,
        'profile': profile if backend == 'gpu' else None,
    }
    if release:
        release_scf(mf)
        out['mf'] = None
    return out


def co_tip_positions(target_xy, z_apex, bond=CO_BOND, apex='O'):
    xy = np.asarray(target_xy, dtype=np.float64)
    if apex == 'O':
        o = np.array([xy[0], xy[1], z_apex])
        c = np.array([xy[0], xy[1], z_apex + bond])
        return ['O', 'C'], np.array([o, c])
    if apex == 'C':
        c = np.array([xy[0], xy[1], z_apex])
        o = np.array([xy[0], xy[1], z_apex + bond])
        return ['C', 'O'], np.array([c, o])
    raise ValueError(f"apex must be 'O' or 'C', got {apex!r}")


def run_co_zscan(mol_names, mol_pos, target_idx, z_grid, basis='def2-SVP', xc='PBE',
                 backend='auto', profile=None, apex='O', bond=CO_BOND, warm_start=True,
                 verbosity=0, E_mol_eV=None, E_CO_eV=None, **kw):
    """Site-correct rigid CO z-scan above mol_pos[target_idx] (SSOT).

    Pass ``z_grid`` = make_z_grid(z_max) or any 1D distance array (Å).
    Tip xy = target atom xy (NOT lab origin). DM from previous z reused when warm_start.
    backend='auto' → GPU OpenCL if NVIDIA available, else stock CPU.
    Pass E_mol_eV / E_CO_eV to skip isolated refs (multi-site scans).
    Returns dict: z, E_abs_eV, E_int_eV, E_mol_eV, E_CO_eV, cycles, wall_s, backend.
    """
    backend = resolve_backend(backend)
    if profile is None:
        profile = DEFAULT_GPU_PROFILE
    mol_pos = np.asarray(mol_pos, dtype=np.float64)
    target = mol_pos[target_idx]
    xy = target[:2]
    z0 = float(target[2])

    if E_mol_eV is None:
        r_mol = run_scf_geometry(mol_names, mol_pos, basis=basis, xc=xc, backend=backend, profile=profile, **kw)
        E_mol_eV = r_mol['E_eV']
    if E_CO_eV is None:
        tip_names0, tip_pos0 = co_tip_positions((0.0, 0.0), 0.0, bond=bond, apex=apex)
        r_co = run_scf_geometry(tip_names0, tip_pos0, basis=basis, xc=xc, backend=backend, profile=profile, **kw)
        E_CO_eV = r_co['E_eV']

    z_grid = np.asarray(z_grid, dtype=np.float64)
    E_abs = np.empty(len(z_grid))
    cycles, walls = [], []
    dm_prev = None
    for i, z_rel in enumerate(z_grid):
        tip_names, tip_pos = co_tip_positions(xy, z0 + float(z_rel), bond=bond, apex=apex)
        names = list(mol_names) + tip_names
        pos = np.vstack([mol_pos, tip_pos])
        r = run_scf_geometry(names, pos, basis=basis, xc=xc, backend=backend, profile=profile,
                             dm0=dm_prev if warm_start else None, **kw)
        E_abs[i] = r['E_eV']
        cycles.append(r['cycles'])
        walls.append(r['wall_s'])
        if warm_start:
            dm_prev = r['dm']
        if verbosity:
            e_int = E_abs[i] - E_mol_eV - E_CO_eV
            print(f"  [{backend}] z={z_rel:.2f}  E_int={e_int:.4f} eV  cycles={r['cycles']}  {r['wall_s']:.2f}s")

    E_int = E_abs - E_mol_eV - E_CO_eV
    return {
        'z': z_grid, 'E_abs_eV': E_abs, 'E_int_eV': E_int, 'E_mol_eV': E_mol_eV, 'E_CO_eV': E_CO_eV,
        'cycles': cycles, 'wall_s': walls, 'target_idx': target_idx, 'apex': apex,
        'backend': backend, 'profile': profile if backend == 'gpu' else None, 'xy': xy,
    }


def write_co_zscan_xyz(path, mol_names, mol_pos, target_idx, z_grid, apex='O', bond=CO_BOND):
    mol_pos = np.asarray(mol_pos, dtype=np.float64)
    target = mol_pos[target_idx]
    xy, z0 = target[:2], float(target[2])
    with open(path, 'w') as fh:
        for z_rel in z_grid:
            tip_names, tip_pos = co_tip_positions(xy, z0 + float(z_rel), bond=bond, apex=apex)
            n = len(mol_names) + 2
            fh.write(f'{n}\n')
            fh.write(f'CO rigid scan tip@{apex}-apex above atom{target_idx} xy=({xy[0]:.3f},{xy[1]:.3f}) z={z_rel:.2f}\n')
            for e, p in zip(mol_names, mol_pos):
                fh.write(f'{e:2s}  {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')
            for e, p in zip(tip_names, tip_pos):
                fh.write(f'{e:2s}  {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n')


# ── Frontier MO cubes / STM slices ───────────────────────────────────────────

ELEM_Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'P': 15, 'S': 16, 'Br': 35, 'I': 53}


def homo_lumo_indices(mf):
    """Return (homo, lumo) 0-based MO indices from mo_occ (RKS/RHF)."""
    occ = np.asarray(mf.mo_occ)
    nocc = int(np.round(occ.sum() / 2.0)) if occ.max() > 1.5 else int(np.count_nonzero(occ > 0.5))
    # Prefer occupancy: last occupied, first virtual
    occ_idx = np.where(occ > 0.5)[0]
    if len(occ_idx) == 0:
        raise ValueError('homo_lumo_indices: no occupied MOs')
    homo = int(occ_idx[-1])
    lumo = homo + 1
    if lumo >= mf.mo_coeff.shape[1]:
        raise ValueError(f'homo_lumo_indices: no virtual above HOMO={homo}')
    return homo, lumo


def write_gaussian_cube(path, data, origin_A, step_A, atom_names, atom_pos_A, title='MO'):
    """Write Gaussian cube. data shape (nx,ny,nz); coords in Å (converted to Bohr in file)."""
    nx, ny, nz = data.shape
    origin_B = np.asarray(origin_A, dtype=np.float64) / bohr2A
    step_B = float(step_A) / bohr2A
    with open(path, 'w') as f:
        f.write(f'{title}\n')
        f.write('Generated by spammm.quantum.pySCF_utils-new\n')
        f.write(f'{len(atom_names):5d} {origin_B[0]:12.6f} {origin_B[1]:12.6f} {origin_B[2]:12.6f}\n')
        f.write(f'{nx:5d} {step_B:12.6f} {0.0:12.6f} {0.0:12.6f}\n')
        f.write(f'{ny:5d} {0.0:12.6f} {step_B:12.6f} {0.0:12.6f}\n')
        f.write(f'{nz:5d} {0.0:12.6f} {0.0:12.6f} {step_B:12.6f}\n')
        for name, pos in zip(atom_names, atom_pos_A):
            Z = ELEM_Z.get(name, 6)
            pB = np.asarray(pos, dtype=np.float64) / bohr2A
            f.write(f'{Z:5d} {float(Z):12.6f} {pB[0]:12.6f} {pB[1]:12.6f} {pB[2]:12.6f}\n')
        flat = np.asarray(data, dtype=np.float64).T.ravel()  # z-fastest
        for i, val in enumerate(flat):
            if i % 6 == 0:
                f.write('\n')
            f.write(f'{val:13.5E}')
        f.write('\n')


def eval_mo_on_grid(mf, mo_idx, origin_A, ngrid, step_A):
    """Evaluate MO on Cartesian grid (Å). Returns ψ in a.u. on (nx,ny,nz).

    Rigorous LCAO: ``ψ(r) = Σ_μ φ_μ(r) C_μi`` with **all** AOs from ``mf.mol``
    (``numint.eval_ao`` — full contracted GTO, incl. double-ζ). Not DFTB OpenCL STO.
    """
    from pyscf.dft import numint
    nx, ny, nz = [int(n) for n in ngrid]
    xs = origin_A[0] + np.arange(nx) * step_A
    ys = origin_A[1] + np.arange(ny) * step_A
    zs = origin_A[2] + np.arange(nz) * step_A
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    coords_A = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    coords_B = coords_A / bohr2A
    ni = numint.NumInt()
    ao = ni.eval_ao(mf.mol, coords_B, deriv=0)
    psi = np.dot(ao, mf.mo_coeff[:, int(mo_idx)])
    return psi.reshape(nx, ny, nz)


def eval_mo_on_xy_slice(mf, mo_idx, scan_xs, scan_ys, z_A):
    """Evaluate MO on a constant-height xy plane (Å). Returns (nx, ny) ψ (a.u.).

    Same as ``eval_mo_on_grid``: full ``eval_ao`` × ``mo_coeff`` — no AO omission.
    """
    from pyscf.dft import numint
    XX, YY = np.meshgrid(scan_xs, scan_ys, indexing='ij')
    coords_A = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, float(z_A))], axis=1)
    coords_B = coords_A / bohr2A
    ni = numint.NumInt()
    ao = ni.eval_ao(mf.mol, coords_B, deriv=0)
    psi = np.dot(ao, mf.mo_coeff[:, int(mo_idx)])
    return psi.reshape(len(scan_xs), len(scan_ys))


def eval_mo_stm_pyscf_slice(mf, mo_idx, scan_xs, scan_ys, z_A, *, tip_orbital='s', intensity=True):
    """pySCF MO-resolved STM **current** on an xy plane (Å); always ≥0 when ``intensity=True``.

    Tip φ_t selects the matrix element; result is squared (STM does not see phase):
      - ``s``  : I ∝ |ψ|²
      - ``pz`` : I ∝ |∂ψ/∂z|²
      - ``py`` : I ∝ |∂ψ/∂y|²

    Full GTO via ``numint.eval_ao``. For signed ψ use ``eval_mo_on_xy_slice``.
    """
    from pyscf.dft import numint
    tip_orbital = str(tip_orbital).lower()
    if tip_orbital not in ('s', 'pz', 'py'):
        raise ValueError(f"tip_orbital must be s|pz|py, got {tip_orbital!r}")
    XX, YY = np.meshgrid(scan_xs, scan_ys, indexing='ij')
    coords_A = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, float(z_A))], axis=1)
    coords_B = coords_A / bohr2A
    ni = numint.NumInt()
    c = mf.mo_coeff[:, int(mo_idx)]
    if tip_orbital == 's':
        ao = ni.eval_ao(mf.mol, coords_B, deriv=0)
        amp = np.dot(ao, c)
    else:
        ao_d = ni.eval_ao(mf.mol, coords_B, deriv=1)  # (4, npts, nao): val, dx, dy, dz
        deriv_i = 3 if tip_orbital == 'pz' else 2
        amp = np.einsum('pi,i->p', ao_d[deriv_i], c)
    if intensity:
        amp = amp ** 2
    return amp.reshape(len(scan_xs), len(scan_ys))


def write_frontier_mo_cubes(mf, out_dir, atom_names, atom_pos_A, *,
                            step_A=0.2, margin_A=4.0, z_extra_A=4.0,
                            labels=('HOMO', 'LUMO'), prefix='pyscf'):
    """SCF already done: write HOMO/LUMO cubes + meta under out_dir.

    Returns dict with paths, indices, energies (eV), grid metadata.
    """
    os.makedirs(out_dir, exist_ok=True)
    homo, lumo = homo_lumo_indices(mf)
    idx_map = {'HOMO': homo, 'LUMO': lumo, 'HOMO-1': homo - 1, 'LUMO+1': lumo + 1}
    pos = np.asarray(atom_pos_A, dtype=np.float64)
    lo = pos.min(axis=0) - margin_A
    hi = pos.max(axis=0) + margin_A
    lo[2] -= z_extra_A * 0.5
    hi[2] += z_extra_A
    ngrid = np.maximum(2, np.ceil((hi - lo) / step_A).astype(int) + 1)
    origin = lo.copy()
    # snap so molecule plane ~ covered
    meta = {
        'origin_A': origin, 'step_A': float(step_A), 'ngrid': ngrid,
        'basis': mf.mol.basis, 'xc': getattr(mf, 'xc', None),
        'backend': getattr(mf, '_spammm_backend', None),
        'E_tot_Ha': float(mf.e_tot),
        'homo': homo, 'lumo': lumo,
        'E_homo_eV': float(mf.mo_energy[homo]) * HAU2EV,
        'E_lumo_eV': float(mf.mo_energy[lumo]) * HAU2EV,
    }
    paths = {}
    for lab in labels:
        if lab not in idx_map:
            raise ValueError(f'unknown MO label {lab!r}')
        imo = idx_map[lab]
        if imo < 0 or imo >= mf.mo_coeff.shape[1]:
            raise ValueError(f'{lab} index {imo} out of range')
        psi = eval_mo_on_grid(mf, imo, origin, ngrid, step_A)
        path = os.path.join(out_dir, f'{prefix}_{lab}.cube')
        write_gaussian_cube(path, psi, origin, step_A, atom_names, pos,
                            title=f'{lab} MO#{imo} E={mf.mo_energy[imo]*HAU2EV:.4f}eV')
        paths[lab] = path
        np.save(os.path.join(out_dir, f'{prefix}_{lab}.npy'), psi.astype(np.float32))
    np.savez(os.path.join(out_dir, f'{prefix}_mo_meta.npz'),
             atom_pos=pos, atom_names=np.array(atom_names), **{k: meta[k] for k in meta if k != 'ngrid'},
             ngrid=ngrid)
    meta['paths'] = paths
    return meta
