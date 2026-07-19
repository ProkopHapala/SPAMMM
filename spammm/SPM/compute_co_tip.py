#!/usr/bin/env python3
"""
Standalone script to compute CO tip density via SCF + PyOpenCL projection.
Supports both FireCore and DFTB backends.
Must run in separate process because Fireball cannot reallocate for different molecules.

CO geometry: O at grid geometric center, C along +z (bond ~1.14 Å).
Caller pads+rolls so O ends at array index (0,0,0) for FFT convolution.

Usage:
    python compute_co_tip.py <output_dir> <grid_spec_json> <step> <nscf> <fdata_dir> <fdata_basis> [backend]

Outputs:
    <output_dir>/co_rho_total.npy
    <output_dir>/co_rho_delta.npy
"""
import sys, os, json, numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.realpath(os.path.join(_THIS_DIR, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

backend = 'firecore'
if len(sys.argv) >= 8:
    backend = sys.argv[7].lower()

if backend == 'firecore':
    try:
        from pyBall import FireCore as fc
        from pyBall.FireballOCL import Grid as ocl_grid
    except (OSError, ImportError) as e:
        raise RuntimeError(f"[ERROR] FireCore backend requested but libFireCore.so not found: {e}")
elif backend == 'dftb':
    from spammm.quantum import DFTB_utils as du
    from spammm.config_utils import get_dftb_basis_path
    from spammm.SPM.AFM_utils import get_density_from_dftb_dense
else:
    raise ValueError(f"Unknown backend: {backend}. Use 'firecore' or 'dftb'.")

Z_TO_ZVAL = {1: 1, 6: 4, 7: 5, 8: 6, 16: 6}
RCUT_DEFAULT = {1: 2.3, 6: 2.6, 7: 2.6, 8: 2.5}

def _onsite_occ(Z):
    if Z == 1:  return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if Z == 6:  return np.array([2.0, 2.0/3, 2.0/3, 2.0/3], dtype=np.float32)
    if Z == 7:  return np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32)
    if Z == 8:  return np.array([2.0, 4.0/3, 4.0/3, 4.0/3], dtype=np.float32)
    return np.array([float(Z_TO_ZVAL.get(int(Z), int(Z))), 0.0, 0.0, 0.0], dtype=np.float32)

def load_xyz(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    natoms = int(lines[0])
    atomTypes = []; atomPos = []
    for line in lines[2:2+natoms]:
        p = line.split()
        sym = p[0]
        z = 6 if sym == 'C' else 1 if sym == 'H' else 8 if sym == 'O' else 7
        atomTypes.append(z)
        atomPos.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(atomTypes, dtype=np.int32), np.array(atomPos, dtype=np.float64)

def main():
    if len(sys.argv) < 7:
        print("Usage: python compute_co_tip.py <out_dir> <grid_spec_json> <step> <nscf> <fdata_dir> <fdata_basis> [backend]")
        sys.exit(1)
    
    out_dir = sys.argv[1]
    grid_spec = json.loads(sys.argv[2])
    for key in ['origin', 'dA', 'dB', 'dC', 'ngrid']:
        if key in grid_spec:
            grid_spec[key] = np.array(grid_spec[key], dtype=np.float32 if key.startswith('d') else np.int32 if key=='ngrid' else np.float32)
    
    step = float(sys.argv[3])
    nscf = int(sys.argv[4])
    fdata_dir = sys.argv[5]
    fdata_basis = sys.argv[6]
    
    os.makedirs(out_dir, exist_ok=True)
    
    co_xyz = os.path.join(_ROOT, 'data', 'xyz', 'CO.xyz')
    co_types, co_pos = load_xyz(co_xyz)
    # Enforce O-first, C along +z from O (ignore file atom order if swapped)
    if int(co_types[0]) != 8:
        # swap so O is index 0
        co_types = co_types[::-1].copy()
        co_pos = co_pos[::-1].copy()
    # Reset to O at (0,0,0), C at (0,0,|bond|) along +z
    bond = float(np.linalg.norm(co_pos[1] - co_pos[0]))
    co_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond]], dtype=np.float64)

    # Place O exactly on a voxel center (index n//2), C along +z.
    # Using 0.5*(n-1)*step puts O between voxels for even n → broken XY symmetry.
    ngrid = np.asarray(grid_spec['ngrid'], dtype=np.int32)
    origin = np.asarray(grid_spec['origin'], dtype=np.float64)
    ijk = ngrid // 2
    grid_center = origin + ijk.astype(np.float64) * step
    co_pos = co_pos + grid_center
    print(f"  O voxel index={tuple(ijk.tolist())}  pos={grid_center}")
    print(f"  CO positions (O on voxel, C along +z): O={co_pos[0]}, C={co_pos[1]}  bond={bond:.4f} Å")
    print(f"  Using backend: {backend}")
    
    if backend == 'firecore':
        _fball_cwd = os.path.join(_ROOT, 'tests', 'pyFireball')
        orig_cwd = os.getcwd()
        os.chdir(_fball_cwd)
        
        fdata_local = os.path.join(_fball_cwd, 'Fdata')
        if not os.path.exists(os.path.join(fdata_local, 'info.dat')):
            if os.path.lexists(fdata_local):
                os.unlink(fdata_local)
            os.symlink(fdata_dir, fdata_local)
        
        print(f"[CO Tip Subprocess] Running Fireball SCF on CO (nscf={nscf})...")
        fc.setVerbosity(0)
        fc.preinit()
        fc.init(co_types, co_pos)
        fc.SCF(co_pos, nmax_scf=nscf)
        
        dims = fc.get_HS_dims()
        neighs = fc.get_HS_neighs(dims)
        neighs = fc.get_rho_sparse(dims, data=neighs)
        rho_sparse_co = neighs.rho
        
        rho_na_co = np.zeros_like(rho_sparse_co, dtype=np.float32)
        neigh_j = neighs.neigh_j.reshape(len(co_types), -1)
        for i in range(len(co_types)):
            slots = np.where(neigh_j[i] == (i+1))[0]
            if len(slots) == 0:
                raise RuntimeError(f"No self-neighbor for CO atom i={i}")
            iself = int(slots[0])
            occ = _onsite_occ(int(co_types[i]))
            rho_na_co[i, iself, :, :] = 0.0
            for k in range(4):
                rho_na_co[i, iself, k, k] = occ[k]
        
        projector = ocl_grid.GridProjector(fdata_dir=fdata_basis, verbosity=0)
        projector.load_basis(sorted(set(co_types.tolist())))
        atoms_dict = {
            'pos': co_pos,
            'Rcut': np.array([RCUT_DEFAULT.get(int(z), 4.5) for z in co_types]),
            'type': co_types,
        }
        
        print("  Projecting CO total density...")
        rho_total = projector.project(rho_sparse_co, neighs, atoms_dict, grid_spec, nMaxAtom=64, use_tiled=True)
        
        print("  Projecting CO neutral-atom density...")
        rho_na_grid = projector.project(rho_na_co, neighs, atoms_dict, grid_spec, nMaxAtom=64, use_tiled=True)
        
        os.chdir(orig_cwd)
    
    elif backend == 'dftb':
        # Use DFTBcore.get_dm_dense via get_density_from_dftb_dense — NOT manual outer(evecs).
        # Manual MO outer-product is wrong for non-orthogonal DFTB basis (asymmetric tip, q≠10).
        basis_name = 'mio-1-1'
        basis_hsd_path = get_dftb_basis_path(basis_name)
        if basis_hsd_path is None:
            basis_hsd_path = du.WFC_HSD_PATHS.get(basis_name)
        if basis_hsd_path is None or not os.path.exists(basis_hsd_path):
            raise RuntimeError(f"Basis HSD file not found for {basis_name}")

        work_dir = os.path.join(out_dir, 'dftb_work')
        print(f"[CO Tip Subprocess] DFTB dense DM projection (basis={basis_name})...")
        result = get_density_from_dftb_dense(
            co_pos, co_types, basis_hsd_path, work_dir,
            grid_spec=grid_spec, step=step, margin=4.0, z_extra=0.0, verbosity=0
        )
        rho_total = result['rho_scf'].astype(np.float32)
        rho_na_grid = result['rho_na'].astype(np.float32)
    
    rho_delta = (rho_total - rho_na_grid).astype(np.float32)
    
    nx, ny, nz = rho_total.shape
    dV = step**3
    q_total = float(rho_total.sum() * dV)
    q_delta = float(rho_delta.sum() * dV)
    peak = np.unravel_index(int(np.argmax(rho_total)), rho_total.shape)
    print(f"  CO rho_total: shape={rho_total.shape} range=[{rho_total.min():.4f},{rho_total.max():.4f}] q={q_total:.3f}")
    print(f"  CO rho_delta: range=[{rho_delta.min():.4f},{rho_delta.max():.4f}] q={q_delta:.3f}")
    print(f"  Peak index={peak}  grid_center_idx=({nx//2},{ny//2},{nz//2})")
    if abs(q_total - 10.0) > 1.0:
        print(f"  WARNING: CO tip electron count q={q_total:.3f} far from 10 — check DM/projection")
    
    np.save(os.path.join(out_dir, 'co_rho_total.npy'), rho_total)
    np.save(os.path.join(out_dir, 'co_rho_delta.npy'), rho_delta)
    print(f"  Saved CO densities to {out_dir}")

if __name__ == '__main__':
    main()
