"""
ModularPipeline.py — Staged AFM/STM simulation pipeline with disk caching.

Purpose: Orchestrate the full AFM/STM workflow in decoupled stages (S1-S6) with
intermediate results saved to disk. Only recomputes stages affected by parameter
changes (dirty flag system).

Key functionality:
  - Stage 1: SCF (DFTB or pySCF) → density matrix, eigenvectors, eigenvalues
  - Stage 2: Density grid projection → rho_scf, rho_na, rho_diff
  - Stage 3: FDBM potentials → Pauli, electrostatic, dispersion, total field
  - Stage 4: Probe relaxation → AFM frequency shift, tip displacements
  - Stage 5: STM projection → LDOS maps at constant height
  - Stage 6: Bond-resolved STM → STM at AFM-relaxed tip positions

  Dual basis (DFTB): prolonged/Slater-tail ρ is for Pauli ONLY; ES keeps stock
  Δρ→V_ES. Never charge-normalize prolonged ρ (∫ρ ≉ N_e by design). See
  make_slater_tail_species_list / doc/DFTB_basis_fit.md /
  doc/Tasks/Import_KrigingGridFF.md.

Role in SPAMMM: The AFM pipeline controller. Used by AFMExtension.py as the
backend for all AFM/STM simulations. Backend-agnostic: supports DFTB (GPU
projection) and pySCF (CPU evaluation).
"""

import os
import time
import numpy as np
import spammm.atomicUtils as au
from spammm.globals import debug_print
from spammm.SPM.AFM import AFMBench, afm_bench_enabled, afm_bench_no_io, afm_use_cpu_fft, afm_use_fast_s3, afm_diag_download
from spammm.SPM import AFM as afm
from spammm.SPM import AFM_utils as afm_utils
from spammm.config_utils import get_config, get_path, get_dftb_basis_path


def _bench():
    return AFMBench.get()


def _cache_write_ok():
    """Stage cache writes: off when SPAMMM_AFM_BENCH_NO_IO=1."""
    return not afm_bench_no_io()


def _bench_stage_start(title: str):
    """Start a fresh per-stage timing table (GUI default)."""
    if not afm_bench_enabled():
        return
    b = AFMBench.reset()
    b.start_run()
    print(f"\n[BENCH] ===== {title} =====", flush=True)


def _bench_stage_end(title: str):
    if afm_bench_enabled():
        _bench().report(title=title)


class ModularAFMPipeline:
    """
    Decoupled, stage-based modular pipeline for AFM and STM simulations.
    Saves intermediate results to disk allowing fast, independent stage execution.

    Supports multiple quantum chemistry backends:
    - 'dftb': DFTB+ with Slater-type orbitals (GPU-accelerated projection)
    - 'pyscf': pySCF with Gaussian-type orbitals (CPU-based evaluation)
    """
    def __init__(self, xyz_file, output_dir, basis='mio-1-1', slako_prefix='mio-1-1',
                 work_dir=None, step=0.1, margin=4.0, z_extra=6.0,
                 scan_range=3.0, scan_step=0.1, height_range=(2.8, 3.6), height_step=0.1,
                 co_tip_dir=None,
                 tip_mode='co',  # 'co' = real CO density; 'gaussian' = isotropic Gaussian tip
                 atomPos=None, enames=None,  # Optional: inject geometry directly instead of xyz_file
                 backend='dftb', pyscf_params=None):  # Backend selection and parameters
        self.xyz_file = xyz_file
        self.output_dir = output_dir
        self._injected_atomPos = atomPos   # If provided, skip loading xyz_file
        self._injected_enames = enames
        self.basis = basis
        self.slako_prefix = slako_prefix
        self.co_tip_dir = co_tip_dir
        self.tip_mode = tip_mode.lower()
        self.backend = backend.lower()
        self.pyscf_params = pyscf_params or {'method': 'RHF', 'basis': 'sto-3g', 'xc': None}
        
        self.work_dir = work_dir or os.path.join(output_dir, 'dftb_work')
        self.step = step
        self.margin = margin
        self.z_extra = z_extra
        self.scan_range = scan_range
        self.scan_step = scan_step
        self.height_range = height_range
        self.height_step = height_step
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Cache file paths
        self.cache_stage1 = os.path.join(self.output_dir, 'cache_stage1_scf.npz')
        self.cache_stage2 = os.path.join(self.output_dir, 'cache_stage2_grids.npz')
        self.cache_stage3 = os.path.join(self.output_dir, 'cache_stage3_potentials.npz')
        self.cache_stage4 = os.path.join(self.output_dir, 'cache_stage4_relax.npz')
        
        # Grid parameters
        self.origin = None
        self.ngrid = None
        self.grid_spec = None
        self.scan_xs = None
        self.scan_ys = None
        self.heights = None
        
        # DFTB structures (only used for backend='dftb')
        self.atomPos = None
        self.atomTypes = None
        self.enames = None
        self.projector = None
        self.atoms_dict = None
        self.norb_per_atom = None
        self.orb_offsets = None
        self._afmulator = None  # reused across S3 dispersion/gradient + S4 relax
        self._fdbm_grid_ready = False  # True after fast-S3 device setup_fdbm_grid_from_img

        # pySCF structures (only used for backend='pyscf')
        self._pyscf_data = None  # Cache for mol, mf, dm from pySCF
        
        # Load molecule and scan grid parameters
        self._init_geometry_and_grids()

    def _get_afmulator(self):
        """Single AFMulator for S3–S4 (avoid 3× OpenCL compile / device init)."""
        if self._afmulator is None:
            _bench().begin('AFMulator_ctor', 'INIT')
            self._afmulator = afm.AFMulator(use_morse=False, nloc=32)
            _bench().end('AFMulator_ctor')
        return self._afmulator

    def _init_geometry_and_grids(self):
        """Load molecular structure and define grid parameters."""
        ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'P':15,'S':16,'Br':35,'I':53}
        inv_z = {v:k for k,v in ELEM_Z.items()}
        
        if self._injected_atomPos is not None and self._injected_enames is not None:
            print(f"\n[ModularPipeline] Using injected geometry ({len(self._injected_atomPos)} atoms)")
            self.atomPos   = np.array(self._injected_atomPos, dtype=np.float64)
            self.enames    = list(self._injected_enames)
            self.atomTypes = np.array([ELEM_Z.get(e, 6) for e in self.enames], dtype=np.int32)
        else:
            print(f"\n[ModularPipeline] Loading molecule from {self.xyz_file}")
            pos, _, names, _, _ = au.load_xyz(self.xyz_file)
            self.atomPos  = np.array(pos, dtype=np.float64)
            self.atomTypes = np.array([ELEM_Z.get(e, 6) for e in names], dtype=np.int32)
            self.enames = [inv_z.get(int(z), 'C') for z in self.atomTypes]
        print(f"  {len(self.atomPos)} atoms loaded.")
        
        # Scan grid coordinates
        x_min = self.atomPos[:,0].min() - self.scan_range
        x_max = self.atomPos[:,0].max() + self.scan_range
        y_min = self.atomPos[:,1].min() - self.scan_range
        y_max = self.atomPos[:,1].max() + self.scan_range
        scan_points_x = int(np.ceil((x_max - x_min) / self.scan_step))
        scan_points_y = int(np.ceil((y_max - y_min) / self.scan_step))
        self.scan_xs = np.linspace(x_min, x_max, scan_points_x)
        self.scan_ys = np.linspace(y_min, y_max, scan_points_y)
        # Inclusive height ladder (arange would exclude height_range[1])
        h0, h1 = float(self.height_range[0]), float(self.height_range[1])
        dz = float(self.height_step)
        n_h = int(round((h1 - h0) / dz)) + 1
        self.heights = np.round(h0 + np.arange(max(n_h, 1), dtype=np.float64) * dz, 6)
        
        # Setup projector / backend-specific initialization
        if self.backend == 'dftb':
            self._init_dftb_backend()
        elif self.backend == 'pyscf':
            self._init_pyscf_backend()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'dftb' or 'pyscf'.")

    def _init_dftb_backend(self):
        """Initialize DFTB backend: setup GridProjector with STO basis."""
        from spammm.quantum.DFTB_utils import SK_PATHS as _SK_PATHS
        from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd, convert_wfc_to_species_list_ang
        from spammm.quantum.DFTB import Grid_dftb as dg

        _bench().begin('init.dftb_projector_setup', 'INIT')
        basis_name = self.basis
        if basis_name == 'mio-1-1':
            self.slako_prefix = _SK_PATHS.get('mio-1-1', self.slako_prefix)
        elif basis_name == '3ob-3-1':
            self.slako_prefix = _SK_PATHS.get('3ob-3-1', self.slako_prefix)

        basis_name = self.slako_prefix.rstrip('/').split('/')[-1] if '/' in self.slako_prefix else self.slako_prefix
        if not basis_name:
            basis_name = '3ob-3-1'

        # Use config system to find basis file
        basis_hsd_path = get_dftb_basis_path(basis_name)
        if basis_hsd_path is None:
            # Fallback to old hardcoded path
            _ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
            basis_hsd_path = os.path.join(_ROOT, 'spammm', 'quantum', 'DFTB', 'data', f'wfc.{basis_name}.hsd')
        
        if os.path.exists(basis_hsd_path):
            basis_data = parse_wfc_hsd(basis_hsd_path)
            basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
            self.norb_per_atom, self.orb_offsets, max_l = afm_utils.build_orbital_layout(basis_data, self.enames)
            max_shells = 3 if max_l >= 2 else 2

            coords_bohr = self.atomPos * 1.8897259886
            species_per_atom = list(range(len(self.enames)))
            dftb_data = {
                'coords_bohr': coords_bohr,
                'species_per_atom': species_per_atom,
                'species_names': self.enames
            }
            self.projector, self.atoms_dict = dg.setup_gridprojector_from_dftb(dftb_data, basis_ang, verbosity=0, max_shells=max_shells)
            debug_print(1, f"[ModularPipeline] DFTB backend initialized with {len(self.enames)} atoms")
        else:
            print(f"[ModularPipeline] WARNING: Basis file not found: {basis_hsd_path}")
        _bench().end('init.dftb_projector_setup')

    def _init_pyscf_backend(self):
        """Initialize pySCF backend: no projector setup (CPU-based evaluation)."""
        print(f"[ModularPipeline] pySCF backend initialized (method={self.pyscf_params.get('method', 'RHF')}, basis={self.pyscf_params.get('basis', 'sto-3g')})")
        # No projector needed for pySCF - density is computed directly on grid
        self.projector = None
        self.atoms_dict = None
        self.norb_per_atom = None
        self.orb_offsets = None

    def stage1_scf(self, force_recompute=False):
        """Stage 1: SCF computation (DFTB or pySCF depending on backend)."""
        if not force_recompute and os.path.exists(self.cache_stage1):
            debug_print(1, f"\n[ModularPipeline] Loading Stage 1 (SCF) from cache...")
            _bench().begin('S1.cache_load', 'IO')
            data = np.load(self.cache_stage1, allow_pickle=True)
            if self.backend == 'dftb':
                out = data['dm_dense'], data['eigvecs'], data['eigvals']
            else:  # pySCF
                self._pyscf_data = {k: data[k] for k in data.keys() if k.startswith('mol_') or k in ['dm', 'eigvecs', 'eigvals']}
                out = data.get('dm'), data['eigvecs'], data['eigvals']
            _bench().end('S1.cache_load')
            return out

        debug_print(1, f"\n[ModularPipeline] Running Stage 1 (SCF) with backend='{self.backend}'...")

        if self.backend == 'dftb':
            return self._stage1_scf_dftb()
        else:  # pySCF
            return self._stage1_scf_pyscf()

    def _stage1_scf_dftb(self):
        """DFTB backend: run DFTBcore SCF and extract density matrix."""
        from spammm.quantum.DFTB.DFTBcore import DFTBcore
        from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd
        from spammm.quantum.DFTB_utils import SK_PATHS as _SK_PATHS
        import shutil

        _bench().begin('S1.setup_inputs', 'CPU')
        basis_name = self.slako_prefix.rstrip('/').split('/')[-1] if '/' in self.slako_prefix else self.slako_prefix
        if not basis_name:
            basis_name = '3ob-3-1'
        
        # Use config system to find basis file
        basis_hsd_path = get_dftb_basis_path(basis_name)
        if basis_hsd_path is None:
            # Fallback to old hardcoded path
            _ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
            basis_hsd_path = os.path.join(_ROOT, 'spammm', 'quantum', 'DFTB', 'data', f'wfc.{basis_name}.hsd')

        sk_dir = _SK_PATHS.get(basis_name, os.path.join(os.environ.get('DFTB_SK_PATH', ''), basis_name))

        # Set up DFTBcore directory and input
        xyz_path = os.path.join(self.work_dir, 'geom.xyz')
        hsd_path = os.path.join(self.work_dir, 'dftb_in.hsd')
        au.save_xyz(xyz_path, self.enames, self.atomPos)

        basis_data = parse_wfc_hsd(basis_hsd_path)
        species = sorted(set(self.enames))
        max_am_map = {0: 's', 1: 'p', 2: 'd'}
        max_ang_lines = []
        for elem in species:
            elem_data = basis_data[elem]
            max_l = max(orb['AngularMomentum'] for orb in elem_data['orbitals'])
            max_ang_lines.append(f'    {elem} = "{max_am_map[max_l]}"')
        max_ang_str = '\n'.join(max_ang_lines)

        with open(hsd_path, 'w') as f:
            f.write(f'''Geometry = xyzFormat {{
  <<< "geom.xyz"
}}
Hamiltonian = DFTB {{
  SCC = Yes
  SCCTolerance = 1e-7
  MaxSCCIterations = 200
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{sk_dir}/"
    Separator = "-"
    Suffix = ".skf"
    LowerCaseTypeName = No
  }}
  MaxAngularMomentum = {{
{max_ang_str}
  }}
}}
''')
        for i, elem1 in enumerate(species):
            for elem2 in species[i:]:
                for sk_file in [f"{elem1}-{elem2}.skf", f"{elem2}-{elem1}.skf"]:
                    src = os.path.join(sk_dir, sk_file)
                    if os.path.exists(src):
                        shutil.copy(src, self.work_dir)
        _bench().end('S1.setup_inputs')

        old_cwd = os.getcwd()
        try:
            os.chdir(self.work_dir)
            _bench().begin('S1.DFTBcore_SCF', 'CPU')  # native lib, host-side
            dftb = DFTBcore()
            dftb.init('dftb_in.hsd')
            dftb.enable_matrix_collection(dm=True, h=False, s=False)
            dftb.run_scf()
            dm_dense = dftb.get_dm_dense()
            eigvecs, eigvals = dftb.get_eigvecs_dense()
            dftb.finalize()
            _bench().end('S1.DFTBcore_SCF')
        finally:
            os.chdir(old_cwd)

        if _cache_write_ok():
            _bench().begin('S1.cache_write', 'IO')
            np.savez_compressed(self.cache_stage1, dm_dense=dm_dense, eigvecs=eigvecs, eigvals=eigvals)
            _bench().end('S1.cache_write')
            debug_print(1, f"  Stage 1 (DFTB) complete and cached.")
        else:
            debug_print(1, f"  Stage 1 (DFTB) complete (cache write skipped).")
        return dm_dense, eigvecs, eigvals

    def _stage1_scf_pyscf(self):
        """pySCF backend: run pySCF SCF and compute density on grid (combined Stage 1+2)."""
        # For pySCF, we combine Stage 1 and Stage 2 (density is computed on grid directly)
        # This avoids caching dm_dense which is specific to STO basis representation
        result = afm_utils.get_density_from_pyscf(
            self.atomPos, self.atomTypes,
            step=self.step, margin=self.margin, z_extra=self.z_extra,
            **self.pyscf_params
        )

        # Cache the result for Stage 2 to load
        # Note: pySCF doesn't have dm_dense in the same sense as DFTB
        np.savez_compressed(self.cache_stage1,
                           dm=result['dm'],  # Density matrix in AO basis
                           eigvecs=result['eigvecs'],
                           eigvals=result['eigvals'],
                           rho_scf=result['rho_scf'],  # Pre-computed on grid
                           rho_na=result['rho_na'],
                           rho_diff=result['rho_diff'],
                           origin=result['origin'],
                           ngrid=result['ngrid'],
                           grid_spec=result['grid_spec'])

        self._pyscf_data = {'mol': result['mol'], 'mf': result['mf'], 'dm': result['dm']}

        print(f"  Stage 1 (pySCF) complete and cached.")
        # Return None for dm_dense since pySCF doesn't use dense STO projection
        return None, result['eigvecs'], result['eigvals']

    def stage2_project(self, dm_dense, force_recompute=False):
        """Stage 2: Grid density projection (SCF, Neutral Atom, and Diff).

        For DFTB backend: uses GPU projector with dm_dense.
        For pySCF backend: densities already computed in Stage 1, just loads from cache.
        """
        # For pySCF backend, densities were computed in Stage 1
        if self.backend == 'pyscf':
            if not force_recompute and os.path.exists(self.cache_stage1):
                print(f"\n[ModularPipeline] Loading Stage 2 (pySCF densities) from Stage 1 cache...")
                data = np.load(self.cache_stage1, allow_pickle=True)
                self.origin = data['origin']
                self.ngrid = data['ngrid']
                rho_scf, rho_na, rho_diff = data['rho_scf'], data['rho_na'], data['rho_diff']
                # Reconstruct grid_spec
                if 'grid_spec' in data:
                    self.grid_spec = data['grid_spec'].item() if isinstance(data['grid_spec'], np.ndarray) else data['grid_spec']
                else:
                    self.grid_spec = {
                        'origin': self.origin,
                        'dA': np.array([self.step, 0.0, 0.0], dtype=np.float32),
                        'dB': np.array([0.0, self.step, 0.0], dtype=np.float32),
                        'dC': np.array([0.0, 0.0, self.step], dtype=np.float32),
                        'ngrid': self.ngrid,
                    }
                z_profile = rho_scf.sum(axis=(0, 1))
                iz_max = int(np.argmax(z_profile))
                print(f"  [Stage2 pySCF] rho_scf: shape={rho_scf.shape} range=[{rho_scf.min():.4e},{rho_scf.max():.4e}] sum={rho_scf.sum():.4e}")
                print(f"  [Stage2 pySCF] density z-peak at iz={iz_max}, z={float(self.origin[2]) + iz_max*self.step:.3f} A")
                return rho_scf, rho_na, rho_diff
            elif force_recompute:
                # If force_recompute, we need to re-run Stage 1
                print(f"\n[ModularPipeline] force_recompute=True for pySCF, re-running Stage 1...")
                _, _, _ = self.stage1_scf(force_recompute=True)
                data = np.load(self.cache_stage1, allow_pickle=True)
                self.origin = data['origin']
                self.ngrid = data['ngrid']
                rho_scf, rho_na, rho_diff = data['rho_scf'], data['rho_na'], data['rho_diff']
                if 'grid_spec' in data:
                    self.grid_spec = data['grid_spec'].item() if isinstance(data['grid_spec'], np.ndarray) else data['grid_spec']
                else:
                    self.grid_spec = {
                        'origin': self.origin,
                        'dA': np.array([self.step, 0.0, 0.0], dtype=np.float32),
                        'dB': np.array([0.0, self.step, 0.0], dtype=np.float32),
                        'dC': np.array([0.0, 0.0, self.step], dtype=np.float32),
                        'ngrid': self.ngrid,
                    }
                return rho_scf, rho_na, rho_diff
            else:
                raise RuntimeError("pySCF Stage 1 must be run before Stage 2")

        # DFTB backend: standard GPU projection
        if not force_recompute and os.path.exists(self.cache_stage2):
            debug_print(1, f"\n[ModularPipeline] Loading Stage 2 (density grids) from cache...")
            _bench().begin('S2.cache_load', 'IO')
            data = np.load(self.cache_stage2)
            self.origin = data['origin']
            self.ngrid = data['ngrid']
            rho_scf, rho_na, rho_diff = data['rho_scf'], data['rho_na'], data['rho_diff']
            # Reconstruct grid_spec
            self.grid_spec = {
                'origin': self.origin,
                'dA': np.array([self.step, 0.0, 0.0], dtype=np.float32),
                'dB': np.array([0.0, self.step, 0.0], dtype=np.float32),
                'dC': np.array([0.0, 0.0, self.step], dtype=np.float32),
                'ngrid': self.ngrid,
            }
            _bench().end('S2.cache_load')
            return rho_scf, rho_na, rho_diff

        debug_print(1, f"\n[ModularPipeline] Projecting Stage 2 (density grids)...")
        _bench_stage_start('Stage 2 density grids')
        from spammm.quantum.DFTB.DFTBplusParser import parse_wfc_hsd, convert_wfc_to_species_list_ang
        from spammm.quantum.DFTB import Grid_dftb as dg

        _bench().begin('S2.grid_setup', 'CPU')
        # Grid parameters setup
        padding = self.margin + self.z_extra
        x_min, x_max = self.atomPos[:,0].min() - self.margin, self.atomPos[:,0].max() + self.margin
        y_min, y_max = self.atomPos[:,1].min() - self.margin, self.atomPos[:,1].max() + self.margin
        z_min, z_max = self.atomPos[:,2].min() - self.margin, self.atomPos[:,2].max() + padding

        origin = np.array([x_min, y_min, z_min], dtype=np.float32)
        ngrid = np.ceil(np.array([x_max - x_min, y_max - y_min, z_max - z_min]) / self.step).astype(np.int32)
        # Round up to clFFT-friendly size (factors 2,3,5,7) and multiple of 8 (GPU block_res)
        ngrid = np.array([afm._FDBMGpyFFT.round_fft_friendly(int(n)) for n in ngrid], dtype=np.int32)

        self.origin = origin
        self.ngrid = ngrid
        self.grid_spec = {
            'origin': origin,
            'dA': np.array([self.step, 0.0, 0.0], dtype=np.float32),
            'dB': np.array([0.0, self.step, 0.0], dtype=np.float32),
            'dC': np.array([0.0, 0.0, self.step], dtype=np.float32),
            'ngrid': ngrid,
        }

        basis_name = self.slako_prefix.rstrip('/').split('/')[-1] if '/' in self.slako_prefix else self.slako_prefix
        if not basis_name:
            basis_name = '3ob-3-1'
        
        # Use config system to find basis file
        basis_hsd_path = get_dftb_basis_path(basis_name)
        if basis_hsd_path is None:
            # Fallback to old hardcoded path
            _ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
            basis_hsd_path = os.path.join(_ROOT, 'spammm', 'quantum', 'DFTB', 'data', f'wfc.{basis_name}.hsd')

        basis_data = parse_wfc_hsd(basis_hsd_path)
        basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
        _bench().end('S2.grid_setup')

        _bench().begin('S2.project_rho_scf', 'GPU')
        rho_scf = self.projector.project_density_dense(dm_dense.astype(np.float32), self.norb_per_atom, self.orb_offsets, self.atoms_dict, self.grid_spec)
        if hasattr(self.projector, 'queue'):
            self.projector.queue.finish()
        _bench().end('S2.project_rho_scf')
        debug_print(1, f"  [Stage2] rho_scf: shape={rho_scf.shape} range=[{rho_scf.min():.4e},{rho_scf.max():.4e}] sum={rho_scf.sum():.4e}")

        coords_bohr = self.atomPos * 1.8897259886
        species_per_atom = list(range(len(self.enames)))
        geo = {
            'natoms': len(self.enames),
            'species_per_atom': species_per_atom,
            'species_names': self.enames,
            'coords_bohr': coords_bohr
        }
        _bench().begin('S2.project_rho_na', 'GPU')
        rho_na = dg.project_neutral_density(
            geo, self.projector, self.atoms_dict, self.grid_spec, basis_ang,
            norb_per_atom=self.norb_per_atom, orb_offsets=self.orb_offsets)
        if hasattr(self.projector, 'queue'):
            self.projector.queue.finish()
        _bench().end('S2.project_rho_na')
        debug_print(1, f"  [Stage2] rho_na:  shape={rho_na.shape} range=[{rho_na.min():.4e},{rho_na.max():.4e}] sum={rho_na.sum():.4e}")
        _bench().begin('S2.rho_diff', 'CPU')
        rho_diff = (rho_scf - rho_na).astype(np.float32)
        _bench().end('S2.rho_diff')

        if _cache_write_ok():
            _bench().begin('S2.cache_write', 'IO')
            np.savez(self.cache_stage2, rho_scf=rho_scf, rho_na=rho_na, rho_diff=rho_diff, origin=self.origin, ngrid=self.ngrid)
            _bench().end('S2.cache_write')
            debug_print(1, f"  Stage 2 complete and cached.")
        else:
            debug_print(1, f"  Stage 2 complete (cache write skipped).")
        _bench_stage_end('Stage 2 density grids')
        return rho_scf, rho_na, rho_diff

    def stage3_potentials(self, rho_scf, rho_na, rho_diff, force_recompute=False,
                          pauli_params=None, vdw_params={'C6_CO': 30.0}):
        """Stage 3: Poisson Electrostatic, Pauli Repulsion, Dispersion, and Total Field (F_total) computation."""
        if not force_recompute and os.path.exists(self.cache_stage3):
            debug_print(1, f"\n[ModularPipeline] Loading Stage 3 (potentials) from cache...")
            _bench().begin('S3.cache_load', 'IO')
            data = np.load(self.cache_stage3)
            out = data['V_ES'], data['E_pauli_field'], data['E_ES_field'], data['E_vdw'], data['F_total']
            _bench().end('S3.cache_load')
            return out
            
        debug_print(1, f"\n[ModularPipeline] Computing Stage 3 (FDBM potentials)...")
        _bench_stage_start('Stage 3 FDBM potentials')
        
        # Set default Pauli params based on backend
        if pauli_params is None:
            if self.backend == 'pyscf':
                from spammm.SPM import AFM as afm_mod
                pyscf_basis_key = f"pyscf_{self.pyscf_params.get('basis', 'sto-3g')}"
                if pyscf_basis_key in afm_mod.PAULI_FITTED_DEFAULTS:
                    pauli_params = afm_mod.PAULI_FITTED_DEFAULTS[pyscf_basis_key]
                    debug_print(1, f"  Using pySCF Pauli defaults for {pyscf_basis_key}: A={pauli_params['A']:.2f}, beta={pauli_params['beta']:.2f}")
                else:
                    print(f"  WARNING: No fitted Pauli params found for {pyscf_basis_key}, using default")
                    pauli_params = {'A': 39.53, 'beta': 1.1544}  # Fitted for 6-31g*
            else:
                from spammm.SPM import AFM as afm_mod
                if self.basis in afm_mod.PAULI_FITTED_DEFAULTS:
                    pauli_params = afm_mod.PAULI_FITTED_DEFAULTS[self.basis]
                    debug_print(1, f"  Using DFTB Pauli defaults for {self.basis}: A={pauli_params['A']:.2f}, beta={pauli_params['beta']:.2f}")
                else:
                    print(f"  WARNING: No fitted Pauli params found for {self.basis}, using default")
                    pauli_params = {'A': 155.33, 'beta': 1.5507}  # mio-1-1 default

        A_pauli = pauli_params.get('A', 787.22)
        beta_pauli = pauli_params.get('beta', 1.2371)
        target_shape = tuple(int(x) for x in self.ngrid)
        fdata_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests', 'pyFireball', 'Fdata'))
        fdata_basis = os.path.join(fdata_dir, 'basis')
        afmulator = self._get_afmulator()

        # ── Round-2 fast S3 (default): fused ES + GPU pad/scale; legacy via SPAMMM_AFM_FAST_S3=0 ──
        if afm_use_fast_s3():
            debug_print(1, f"  Tip mode: {self.tip_mode}  [FAST_S3]")
            _bench().begin('S3.tip_densities', 'IO')
            tip_tot, tip_del = afm_utils.get_tip_densities(
                tip_mode=self.tip_mode, target_shape=target_shape, step=self.step, margin=self.margin,
                output_dir=self.output_dir, co_tip_dir=self.co_tip_dir,
                fdata_dir=fdata_dir, fdata_basis=fdata_basis, backend=self.backend,
                pad_mode='none' if self.tip_mode == 'co' else 'cpu',
            )
            _bench().end('S3.tip_densities')
            # Host E/V for GUI plots + cache; skip only under SPAMMM_AFM_BENCH_NO_IO
            need_E = (not afm_bench_no_io()) or afm_diag_download() or _cache_write_ok()
            _bench().begin('S3.fast_fields_gpu', 'GPU')
            V_ES, E_pauli_field, E_ES_field, E_vdw, F_total = afm.stage3_fdbm_fields_fast(
                afmulator, rho_scf, rho_diff, tip_tot, tip_del,
                self.origin, self.step, self.ngrid, self.atomPos, self.atomTypes,
                A_pauli, beta_pauli, C6_CO=vdw_params['C6_CO'],
                tip_already_rolled=False, download_fields=need_E,
            )
            afmulator.queue.finish()
            _bench().end('S3.fast_fields_gpu')
            self._fdbm_grid_ready = True
            if _cache_write_ok():
                _bench().begin('S3.cache_write', 'IO')
                if V_ES is None:
                    V_ES = afm.fft_poisson(rho_diff, self.step)
                if E_pauli_field is None:
                    fft = afm._get_fdbm_fft(afmulator.ctx, afmulator.queue)
                    E_pauli_field = np.ascontiguousarray(fft._xyz_E_pauli.get())
                    E_ES_field = np.ascontiguousarray(fft._xyz_E_es.get())
                    E_vdw = afmulator.download_image_rgba_xyz(afmulator.img_disp_fast, target_shape)[..., 3]
                np.savez(self.cache_stage3, V_ES=V_ES, E_pauli_field=E_pauli_field,
                         E_ES_field=E_ES_field, E_vdw=E_vdw, F_total=F_total)
                _bench().end('S3.cache_write')
                debug_print(1, f"  Stage 3 complete and cached [FAST_S3].")
            else:
                debug_print(1, f"  Stage 3 complete [FAST_S3] (cache write skipped).")
            _bench_stage_end('Stage 3 FDBM potentials')
            return V_ES, E_pauli_field, E_ES_field, E_vdw, F_total

        # ── Legacy Stage-3 path (SPAMMM_AFM_FAST_S3=0 or CPU FFT) ──
        _fft_where = 'CPU' if afm_use_cpu_fft() else 'GPU'
        _bench().begin('S3.fft_poisson', _fft_where)
        V_ES = afm.fft_poisson(rho_diff, self.step)
        _bench().end('S3.fft_poisson')

        debug_print(1, f"  Tip mode: {self.tip_mode}  [LEGACY_S3]")
        _bench().begin('S3.tip_densities', 'CPU')
        co_rho_total, co_rho_delta = afm_utils.get_tip_densities(
            tip_mode=self.tip_mode, target_shape=target_shape, step=self.step, margin=self.margin,
            output_dir=self.output_dir, co_tip_dir=self.co_tip_dir,
            fdata_dir=fdata_dir, fdata_basis=fdata_basis, backend=self.backend,
            pad_mode='cpu',
        )
        _bench().end('S3.tip_densities')

        _bench().begin('S3.pauli_overlap_fft', _fft_where)
        overlap_raw = afm.compute_pauli_overlap(rho_scf, co_rho_total, self.step, tip_rolled=True)
        _bench().end('S3.pauli_overlap_fft')
        _bench().begin('S3.pauli_scale', 'CPU')
        E_pauli_field = afm.scale_pauli_field(overlap_raw, self.step, A_pauli, beta_pauli, return_grads=False)
        _bench().end('S3.pauli_scale')

        _bench().begin('S3.es_conv_fft', _fft_where)
        E_ES_field = afm.compute_es_conv_field(V_ES, co_rho_delta, self.step, tip_rolled=True, return_grads=False)
        _bench().end('S3.es_conv_fft')

        _bench().begin('S3.dispersion', 'GPU')
        E_vdw = afm.compute_dispersion_grid(
            self.atomPos, self.atomTypes, self.origin, self.step, self.ngrid,
            C6_CO=vdw_params['C6_CO'], return_grads=False, afmulator=afmulator
        )
        _bench().end('S3.dispersion')

        _bench().begin('S3.E_total_sum', 'CPU')
        E_total = E_pauli_field + E_ES_field + E_vdw
        _bench().end('S3.E_total_sum')

        _bench().begin('S3.compute_gradient_cl', 'GPU')
        F_total = afmulator.compute_gradient_cl(E_total, self.step, bAlloc=True)
        afmulator.queue.finish()
        _bench().end('S3.compute_gradient_cl')
        self._fdbm_grid_ready = False

        if _cache_write_ok():
            _bench().begin('S3.cache_write', 'IO')
            np.savez(self.cache_stage3, V_ES=V_ES, E_pauli_field=E_pauli_field,
                                E_ES_field=E_ES_field, E_vdw=E_vdw, F_total=F_total)
            _bench().end('S3.cache_write')
            debug_print(1, f"  Stage 3 complete and cached.")
        else:
            debug_print(1, f"  Stage 3 complete (cache write skipped).")
        _bench_stage_end('Stage 3 FDBM potentials')
        return V_ES, E_pauli_field, E_ES_field, E_vdw, F_total

    def stage4_relax(self, F_total, force_recompute=False, relax_params=None, ppm_mode=True):
        """Stage 4: Probe-particle MD relaxation (yielding AFM signal and tip displacements)."""
        if relax_params is None:
            from spammm.SPM import AFM as afm_mod
            relax_params = {'K_LAT': afm_mod.K_LAT_HAPALA_EV_A2}  # 0.5 N/m → eV/Å²
        if not force_recompute and os.path.exists(self.cache_stage4):
            debug_print(1, f"\n[ModularPipeline] Loading Stage 4 (relaxation) from cache...")
            _bench().begin('S4.cache_load', 'IO')
            data = np.load(self.cache_stage4)
            tip_disp = {'dx': data['tip_disp_dx'], 'dy': data['tip_disp_dy'], 'dz': data['tip_disp_dz']}
            out = data['df'], tip_disp, data['FEs_relax']
            _bench().end('S4.cache_load')
            return out
            
        k_ev = relax_params['K_LAT']
        K_RAD = float(relax_params.get('K_RAD', 20.0))
        bond_length = float(relax_params.get('bond_length', 3.0))  # AFM CLI SSOT (was 4.0 compose default)
        from spammm.SPM import AFM as afm_mod
        k_nm = afm_mod.stiffness_eVA2_to_Nm(k_ev)
        debug_print(1, f"\n[ModularPipeline] Running Stage 4 (probe relaxation) "
              f"K_LAT={k_ev:.4f} eV/Å² (= {k_nm:.2f} N/m) L={bond_length:.2f}Å...")
        _bench_stage_start('Stage 4 probe relaxation')
        afmulator = self._get_afmulator()
        
        _bench().begin('S4.compose_and_relax_total', 'GPU')
        df, tip_disp, FEs_relax = afm_utils.compose_and_relax_total(
            F_total,
            self.scan_xs, self.scan_ys, self.heights,
            self.origin, self.step, self.atomPos, K_LAT=relax_params['K_LAT'],
            K_RAD=K_RAD, bond_length=bond_length,
            use_gpu_relax=True, ppm_mode=ppm_mode, afmulator=afmulator,
            reuse_fdbm_grid=bool(getattr(self, '_fdbm_grid_ready', False)),
        )
        afmulator.queue.finish()
        _bench().end('S4.compose_and_relax_total')
        self._fdbm_grid_ready = False
        
        dxy = np.hypot(tip_disp['dx'], tip_disp['dy'])
        debug_print(1, f"  Stage 4 tip |dxy|_max={float(dxy.max()):.4f}Å  (soft K→large deflection / sharp PP edges)")
            
        if _cache_write_ok():
            _bench().begin('S4.cache_write', 'IO')
            np.savez(self.cache_stage4, df=df, tip_disp_dx=tip_disp['dx'],
                                tip_disp_dy=tip_disp['dy'], tip_disp_dz=tip_disp['dz'], FEs_relax=FEs_relax)
            _bench().end('S4.cache_write')
            debug_print(1, f"  Stage 4 complete and cached.")
        else:
            debug_print(1, f"  Stage 4 complete (cache write skipped).")
        _bench_stage_end('Stage 4 probe relaxation')
        return df, tip_disp, FEs_relax

    def stage5_stm(self, eigvecs, eigvals, lumo_offsets=[1, 2, 3], mo_indices=None,
                  field='ldos', use_exp_basis=True, exp_beta=1.0, exp_r0=3.0):
        """Stage 5: Standard STM projection on height slices."""
        if self.backend == 'pyscf':
            raise NotImplementedError(
                "STM imaging (Stage 5) is not yet supported with pySCF backend. "
                "The current Phase 1 implementation computes density directly on grid "
                "but doesn't export MOs for GPU projection. Use DFTB backend for STM, "
                "or implement Phase 2 (GTO GPU projection) for pySCF STM support."
            )
        print(f"\n[ModularPipeline] Running Stage 5 (Standard STM)...")
        return afm_utils.compute_stm(
            self.projector, eigvecs, eigvals, self.scan_xs, self.scan_ys, self.heights,
            self.norb_per_atom, self.orb_offsets, self.atoms_dict,
            lumo_offsets=lumo_offsets, mo_indices=mo_indices, field=field,
            use_exp_basis=use_exp_basis, exp_beta=exp_beta, exp_r0=exp_r0
        )

    def stage6_br_stm(self, eigvecs, eigvals, tip_disp, lumo_offsets=[1, 2, 3],
                      mo_indices=None, field='ldos', use_exp_basis=True, exp_beta=1.0, exp_r0=3.0):
        """Stage 6: Bond-Resolved STM (STM at AFM-relaxed tip positions)."""
        if self.backend == 'pyscf':
            raise NotImplementedError(
                "Bond-Resolved STM (Stage 6) is not yet supported with pySCF backend. "
                "The current Phase 1 implementation computes density directly on grid "
                "but doesn't export MOs for GPU projection. Use DFTB backend for STM, "
                "or implement Phase 2 (GTO GPU projection) for pySCF STM support."
            )
        print(f"\n[ModularPipeline] Running Stage 6 (Bond-Resolved STM)...")
        return afm_utils.compute_bond_resolved_stm(
            self.projector, eigvecs, eigvals, self.scan_xs, self.scan_ys, self.heights,
            tip_disp, self.norb_per_atom, self.orb_offsets, self.atoms_dict,
            lumo_offsets=lumo_offsets, mo_indices=mo_indices, field=field,
            use_exp_basis=use_exp_basis, exp_beta=exp_beta, exp_r0=exp_r0
        )

    def project_pauli_rho(self, dm_dense, projection='stock', rho_scf_stock=None):
        """Density used for Pauli/FDBM: stock STO or prolonged Slater-tail (dual-basis).

        ES must keep stock Δρ (pass ``rho_diff`` from Stage 2 stock). Prolonged ρ is
        for Pauli only — same dual-basis rule as ``run_br_stm_afm_panel``.
        """
        projection = str(projection).lower()
        if projection not in ('stock', 'prolonged'):
            raise ValueError(f"projection must be 'stock' or 'prolonged', got {projection!r}")
        if projection == 'stock':
            if rho_scf_stock is not None:
                return rho_scf_stock
            return self.projector.project_density_dense(
                dm_dense.astype(np.float32), self.norb_per_atom, self.orb_offsets,
                self.atoms_dict, self.grid_spec)
        from spammm.config_utils import get_dftb_basis_path
        from spammm.quantum.DFTB.DFTBplusParser import (
            parse_wfc_hsd, convert_wfc_to_species_list_ang, make_slater_tail_species_list,
        )
        basis_hsd = get_dftb_basis_path(self.basis if hasattr(self, 'basis') else self.slako_prefix)
        if basis_hsd is None:
            basis_name = self.slako_prefix.rstrip('/').split('/')[-1] or '3ob-3-1'
            basis_hsd = get_dftb_basis_path(basis_name)
        basis_data = parse_wfc_hsd(basis_hsd)
        basis_ang = convert_wfc_to_species_list_ang(basis_data, resolution_bohr=0.04)
        prol_ang = make_slater_tail_species_list(basis_ang)
        afm_utils._set_projector_species_basis(self.projector, self.atoms_dict, prol_ang, rc_max=6.0)
        rho = self.projector.project_density_dense(
            dm_dense.astype(np.float32), self.norb_per_atom, self.orb_offsets,
            self.atoms_dict, self.grid_spec)
        if hasattr(self.projector, 'queue'):
            self.projector.queue.finish()
        debug_print(1, f"  [dual-basis] prolonged Pauli ρ projected  shape={rho.shape}")
        return rho
