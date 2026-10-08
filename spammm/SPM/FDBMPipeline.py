"""
FDBMPipeline.py — End-to-end device-resident FDBM pipeline (single OpenCL context).

Motivation (doc/Tasks/FDBM_EndToEnd_GPU_Pipeline.md): the legacy orchestration
(``get_density_from_dftb_dense`` + ``run_fdbm_pp_from_density``) creates 4–5 OpenCL
contexts, downloads every projected density to host (×B3 factor on CPU, −ρ_NA on
CPU) and re-uploads them into Stage-3 — plus a second full DFTB SCF for the
prolonged basis. This class owns ONE context+queue and keeps ρ on the device:

    DFTBcore SCF (once, CPU/Fortran — the only non-GPU stage)
      → project (dm_scf − dm_na)·B3  directly into fft._xyz_rho_diff   (stock basis)
      → project dm_scf·B3            directly into fft._xyz_rho_scf    (prolonged basis, Pauli)
      → stage3_fdbm_fields_fast(device buffers) → img_FF_fdbm
      → scan_fdbm → ScanResult (host FEs are the output, by design)

DUAL BASIS is mandatory physics — do not "fix": prolonged ρ for Pauli only
(never normalize; A,β absorb the scale), stock Δρ for ES. No new kernels:
ρ_diff comes from ONE projection of (dm_scf − dm_na) — the projector is linear
in dm — and B3_FACTOR folds into the uploaded DM the same way.

Legacy path (``run_fdbm_pp_from_density``) is kept untouched for parity/debug.

Reuse one FDBMPipeline across molecules (dataset generation — the intended use):
different natoms/positions/DM are all per-call inputs and handled correctly.
What is cached (and keyed by):
    ctx+queue+program      — once per pipeline
    FFT plans + scratch    — per grid SHAPE (nx,ny,nz)  [origin is per-call, safe]
    tip pad/roll buffers   — per (shape, tip_mode, step, margin, basis)
    GridProjectors         — per (basis kind, hsd path)
    species/basis tables   — per hsd path
Per-molecule work that always re-runs: DFTB SCF (~0.05–0.1 s), task build,
dm/atom uploads, projection, dispersion, fields, scan — ~0.1–0.2 s/mol warm.
VRAM grows per UNIQUE grid shape — for max reuse snap molecule grids to
canonical sizes (same ngrid+step → all caches hit; different origin is fine).

Fields-only mode (`run_fields`/`fields_for_fit`): FDBM→PME/contact encode or
dataset potential labels — skips the PP-scan machinery entirely (no scan
buffers, no FEs downloads, no df/diss numpy) and can also skip the composed
force image (oracle=False) when only component potentials are wanted.
"""

import os
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from . import AFM as afm
from . import AFM_utils as afm_utils
from ..quantum.DFTB import Grid_dftb as dg


class FDBMPipeline:
    """Single-context, device-resident FDBM: SCF → projection → fields → PP scan.

    Owns: one AFMulator (ctx+queue+program), per-shape _FDBMGpyFFT, per-basis
    GridProjector, per-(step,margin) raw tip densities, per-shape pad-rolled
    tip buffers. All lazy — nothing allocated until first use.
    """

    def __init__(self, verbosity=0):
        self.verbosity = int(verbosity)
        self.afm = None            # lazy AFMulator — the ONE ctx+queue+program
        self.ffts = {}             # (nx,ny,nz)      -> _FDBMGpyFFT bound to afm ctx
        self.projectors = {}       # basis key       -> GridProjector on afm ctx
        self._species = {}         # ('stock'|'prol', hsd_path) -> species_list_ang
        self._basis_data = {}      # hsd_path -> parsed basis_data (shared by species_ang/dftb_prep)
        self._tip_raw = {}         # (mode,step,margin,basis,backend) -> (tot,del) host
        self._tip_dev = {}         # (shape, raw_key) -> (tip_tot_cl, tip_del_cl) in fft scratch
        self._rho_stock_cl = {}    # shape -> cl_array for stock rho_scf ('both' mode)
        self._ones_cl = {}         # shape -> ones cl_array (device reductions)
        self.last_prep = None      # most recent dftb_prep dict (dm reuse, STM)

    # ── lazy resource owners ────────────────────────────────────────────────

    def ensure_afm(self):
        if self.afm is None:
            self.afm = afm.AFMulator(use_morse=False, nloc=32, use_fire=True)
        return self.afm

    def ensure_fft(self, shape, step):
        """Per-shape _FDBMGpyFFT on the shared ctx — plans/scratch persist across calls."""
        a = self.ensure_afm()
        shape = tuple(int(x) for x in shape[:3])
        fft = self.ffts.get(shape)
        if fft is None:
            fft = afm._FDBMGpyFFT(ctx=a.ctx, queue=a.queue)
            fft.bind_afm_program(a.prg)
            self.ffts[shape] = fft
        fft.ensure(shape, step=step)
        return fft

    def basis_data(self, basis_hsd_path):
        """Parsed wfc.*.hsd basis_data, cached per path (shared by species_ang + dftb_prep)."""
        key = os.path.abspath(basis_hsd_path)
        bd = self._basis_data.get(key)
        if bd is None:
            from ..quantum.DFTB.DFTBplusParser import parse_wfc_hsd
            bd = parse_wfc_hsd(basis_hsd_path)
            self._basis_data[key] = bd
        return bd

    def species_ang(self, basis_hsd_path, kind='stock'):
        """Parsed angular basis ('stock') or prolonged-tail basis ('prol'), cached per HSD."""
        key = (kind, os.path.abspath(basis_hsd_path))
        sp = self._species.get(key)
        if sp is None:
            from ..quantum.DFTB.DFTBplusParser import (
                convert_wfc_to_species_list_ang, make_slater_tail_species_list)
            sp = convert_wfc_to_species_list_ang(self.basis_data(basis_hsd_path), resolution_bohr=0.04)
            if kind == 'prol':
                sp = make_slater_tail_species_list(sp)
            self._species[key] = sp
        return sp

    def ensure_projector(self, key, species_list_ang, max_shells=None):
        """Per-basis GridProjector on the shared ctx — basis table uploaded once."""
        a = self.ensure_afm()
        p = self.projectors.get(key)
        if p is None:
            p = dg.GridProjector(fdata_dir=None, ctx=a.ctx, queue=a.queue, verbosity=self.verbosity)
            p.load_basis_sto(species_list_ang, max_shells=max_shells)
            self.projectors[key] = p
        return p

    @staticmethod
    def _atoms_dict(dftb_data, species_list_ang):
        """atoms_dict for a projector — same layout as dg.setup_gridprojector_from_dftb."""
        coords_ang = dftb_data['coords_bohr'] * 0.5291772109  # Bohr -> Angstrom
        sp_by_name = {sp['name']: sp for sp in species_list_ang}
        atomic_numbers = np.array([sp_by_name[dftb_data['species_names'][si]]['atomic_number'] for si in dftb_data['species_per_atom']])
        cutoffs = np.array([sp_by_name[dftb_data['species_names'][si]]['orbitals'][0]['cutoff'] for si in dftb_data['species_per_atom']])
        return {'pos': coords_ang, 'Rcut': cutoffs, 'type': atomic_numbers}

    def _rho_stock_buf(self, shape):
        shape = tuple(int(x) for x in shape[:3])
        b = self._rho_stock_cl.get(shape)
        if b is None:
            b = cl_array.empty(self.ensure_afm().queue, shape, dtype=np.float32)
            self._rho_stock_cl[shape] = b
        return b

    def _device_sum(self, arr_cl):
        """Sum of a device float field via dot(·,1) — pyopencl Array has no .sum() here."""
        ones = self._ones_cl.get(arr_cl.shape)
        if ones is None:
            ones = cl_array.zeros(self.ensure_afm().queue, arr_cl.shape, dtype=np.float32) + 1.0
            self._ones_cl[arr_cl.shape] = ones
        return float(cl_array.dot(arr_cl, ones).get())

    # ── tip ─────────────────────────────────────────────────────────────────

    def tip_raw(self, tip_mode, step, margin, basis='3ob-3-1', backend='dftb', output_dir=None, target_shape=None):
        """Raw (unpadded) tip densities. 'co' cached (shape-independent); other modes
        are built at target_shape (e.g. gaussian) → device upload, no pad/roll."""
        if tip_mode != 'co':
            return afm_utils.get_tip_densities(tip_mode, target_shape, step, margin=margin,
                                             basis=basis, output_dir=output_dir,
                                             backend=backend, pad_mode='none')
        key = (tip_mode, float(step), float(margin), basis, backend)
        t = self._tip_raw.get(key)
        if t is None:
            t = afm_utils.get_tip_densities(tip_mode, (1, 1, 1), step, margin=margin,
                                            basis=basis, output_dir=output_dir,
                                            backend=backend, pad_mode='none')
            self._tip_raw[key] = t
        return t

    def ensure_tip(self, tip_mode, target_shape, step, margin, basis='3ob-3-1', backend='dftb', output_dir=None):
        """Device pad-rolled tip in the shape's FFT scratch. Cached per (shape, raw tip)."""
        raw_key = (tip_mode, float(step), float(margin), basis, backend)
        key = (tuple(int(x) for x in target_shape[:3]), raw_key)
        t = self._tip_dev.get(key)
        if t is not None:
            return t
        fft = self.ensure_fft(target_shape, step)
        tot_raw, del_raw = self.tip_raw(tip_mode, step, margin, basis=basis, backend=backend,
                                        output_dir=output_dir, target_shape=target_shape)
        for raw, dest in ((tot_raw, fft._xyz_tip_tot), (del_raw, fft._xyz_tip_del)):
            if tuple(raw.shape) == tuple(int(x) for x in target_shape[:3]):
                fft._xyz_host_to_cl(dest, raw)
            else:
                ox, oy, oz, rx, ry, rz = afm.pad_roll_shifts(raw, target_shape)
                fft.pad_roll_to_cl(raw, target_shape, ox, oy, oz, rx, ry, rz, dest_cl=dest)
        t = (fft._xyz_tip_tot, fft._xyz_tip_del)
        self._tip_dev[key] = t
        return t

    # ── DFTB single-point SCF (CPU/Fortran — the only non-GPU stage) ─────────

    def dftb_prep(self, atomPos, atomTypes, basis_hsd_path, work_dir, max_shells=None, run_scf=True):
        """Basis/geometry prep + one DFTBcore SCF → dm/eigvecs + all layout metadata.

        Replicates the SCF half of ``get_density_from_dftb_dense`` (files, SK copy,
        chdir, init/scf/dm/eig/finalize) so the new pipeline runs SCF exactly once
        per molecule regardless of how many basis projections follow.
        """
        from ..quantum.DFTB.DFTBcore import DFTBcore
        from spammm import atomicUtils as au

        ELEM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16,'Cl':17,'Br':35,'I':53}
        inv_z = {v:k for k,v in ELEM_Z.items()}
        enames = [inv_z[int(z)] for z in atomTypes]

        work_dir = os.path.abspath(work_dir)
        os.makedirs(work_dir, exist_ok=True)

        basis_data = self.basis_data(basis_hsd_path)
        basis_ang = self.species_ang(basis_hsd_path, 'stock')
        norb_per_atom, orb_offsets, max_l = afm_utils.build_orbital_layout(basis_data, enames)
        if max_shells is None:
            max_shells = 3 if max_l >= 2 else 2

        coords_bohr = np.asarray(atomPos, dtype=np.float64) * 1.8897259886
        species_per_atom = list(range(len(enames)))
        dftb_data = {'coords_bohr': coords_bohr, 'species_per_atom': species_per_atom, 'species_names': enames}
        geo = {'natoms': len(enames), 'species_per_atom': species_per_atom, 'species_names': enames, 'coords_bohr': coords_bohr}

        prep = {'enames': enames, 'basis_data': basis_data, 'basis_ang': basis_ang,
                'basis_hsd_path': basis_hsd_path,
                'norb_per_atom': norb_per_atom, 'orb_offsets': orb_offsets,
                'max_shells': max_shells, 'dftb_data': dftb_data, 'geo': geo,
                'dm': None, 'eigvecs': None, 'eigvals': None, 'energy': None}
        if not run_scf:
            self.last_prep = prep
            return prep

        # Write DFTBcore input (minimal HSD — same as get_density_from_dftb_dense)
        basis_name = os.path.basename(basis_hsd_path).replace('wfc.', '').replace('.hsd', '')
        from ..quantum.DFTB_utils import SK_PATHS as _SK_PATHS
        from ..config_utils import get_dftb_sk_path as _get_dftb_sk_path
        sk_dir = _SK_PATHS.get(basis_name) or _get_dftb_sk_path(basis_name) or os.path.join(os.environ.get('DFTB_SK_PATH', ''), basis_name)
        xyz_path = os.path.join(work_dir, 'geom.xyz')
        hsd_path = os.path.join(work_dir, 'dftb_in.hsd')
        au.save_xyz(xyz_path, enames, atomPos)

        species = sorted(set(enames))
        max_am_map = {0: 's', 1: 'p', 2: 'd'}
        max_ang_lines = []
        for elem in species:
            elem_data = basis_data[elem]
            max_l_e = max(orb['AngularMomentum'] for orb in elem_data['orbitals'])
            max_ang_lines.append(f'    {elem} = "{max_am_map[max_l_e]}"')
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
        # NOTE: no .skf copying — SlaterKosterFiles Prefix above points at the absolute
        # sk_dir, so DFTBcore reads the tables in place (legacy copied them into work_dir).

        old_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            dftb = DFTBcore()
            dftb.init('dftb_in.hsd')
            dftb.enable_matrix_collection(dm=True, h=False, s=False)
            prep['energy'] = dftb.run_scf()
            prep['dm'] = dftb.get_dm_dense()
            prep['eigvecs'], prep['eigvals'] = dftb.get_eigvecs_dense()
            dftb.finalize()
        finally:
            os.chdir(old_cwd)
        self.last_prep = prep
        return prep

    # ── device-resident density projection ──────────────────────────────────

    def project_densities(self, prep, grid_spec, step, dm=None, variants=('diff', 'prol'), charge_check=True):
        """Project densities straight into the shape's FFT scratch buffers (no host copies).

        variants: subset of {'diff','prol','stock'} —
            'diff'  stock ρ_diff = project(dm_scf − dm_na)   → fft._xyz_rho_diff (ES)
            'prol'  prolonged ρ_scf = project(dm_scf)         → fft._xyz_rho_scf  (Pauli)
            'stock' stock ρ_scf    = project(dm_scf)          → pipeline buffer   (Pauli, 'both' mode)
        Returns dict of device arrays {key: cl_array}.
        """
        a = self.ensure_afm()
        fft = self.ensure_fft(grid_spec['ngrid'], step)
        basis_ang = prep['basis_ang']
        norb, off = prep['norb_per_atom'], prep['orb_offsets']
        ms = prep['max_shells']
        dm32 = np.asarray(prep['dm'] if dm is None else dm, dtype=np.float32)
        out = {}

        bkey = os.path.abspath(prep['basis_hsd_path'])     # projector cache must include basis identity
        if 'diff' in variants or 'stock' in variants:
            proj_stock = self.ensure_projector(('stock', bkey), basis_ang, max_shells=ms)
            atoms_stock = self._atoms_dict(prep['dftb_data'], basis_ang)
            tdev_stock = proj_stock.build_tasks_dev(atoms_stock, grid_spec)   # tasks stay on GPU
        if 'diff' in variants:
            dm_na, _, _ = dg.build_na_dm_diagonal(prep['geo'], basis_ang, atoms_stock, norb, off)
            dm_diff = dm32 - np.asarray(dm_na, dtype=np.float32)          # B3 folded inside projector
            proj_stock.project_density_dense(dm_diff, norb, off, atoms_stock, grid_spec, out_cl=fft._xyz_rho_diff.data, tasks_dev=tdev_stock)
            out['diff'] = fft._xyz_rho_diff
        if 'stock' in variants:
            buf = self._rho_stock_buf(grid_spec['ngrid'])
            proj_stock.project_density_dense(dm32, norb, off, atoms_stock, grid_spec, out_cl=buf.data, tasks_dev=tdev_stock)
            out['stock'] = buf
        if 'prol' in variants:
            basis_prol = self.species_ang(prep['basis_hsd_path'], 'prol')
            proj_prol = self.ensure_projector(('prol', bkey), basis_prol, max_shells=ms)
            atoms_prol = self._atoms_dict(prep['dftb_data'], basis_prol)   # prolonged cutoffs → own task list
            tdev_prol = proj_prol.build_tasks_dev(atoms_prol, grid_spec)
            proj_prol.project_density_dense(dm32, norb, off, atoms_prol, grid_spec, out_cl=fft._xyz_rho_scf.data, tasks_dev=tdev_prol)
            out['prol'] = fft._xyz_rho_scf

        if charge_check and 'diff' in out:
            q_diff = self._device_sum(out['diff']) * float(step) ** 3
            print(f"  [CHARGE CHECK] q_diff={q_diff:.6f} e (should be ~0; device reduction)")
            if abs(q_diff) > 2.0:
                print("  WARNING: Large charge imbalance in rho_diff! Electrostatics may be unreliable.")
        return out

    # ── Stage-3 fields + Stage-4 scan ────────────────────────────────────────

    def fields(self, rho_scf_cl, rho_diff_cl, atomPos, atomTypes, origin, step, ngrid,
               A_pauli, beta_pauli, C6_CO=30.0, tip_mode='co', margin=4.0, basis='3ob-3-1',
               download_fields=False, download_F=False, output_dir=None):
        """Device-resident Stage-3: assumes densities + tip already on device."""
        tip_tot_cl, tip_del_cl = self.ensure_tip(tip_mode, ngrid, step, margin, basis=basis, output_dir=output_dir)
        return afm.stage3_fdbm_fields_fast(
            self.ensure_afm(), None, None, None, None,
            origin, step, ngrid, atomPos, atomTypes, float(A_pauli), float(beta_pauli),
            C6_CO=float(C6_CO), download_fields=download_fields,
            rho_scf_cl=rho_scf_cl, rho_diff_cl=rho_diff_cl,
            tip_tot_cl=tip_tot_cl, tip_del_cl=tip_del_cl, download_F=download_F)

    def fields_for_fit(self, rho_scf_cl, rho_diff_cl, atomPos, atomTypes, origin, step, ngrid,
                       A_pauli, beta_pauli, C6_CO=30.0, tip_mode='co', margin=4.0,
                       basis='3ob-3-1', backend='dftb', oracle=True):
        """FDBM potentials ONLY — for FDBM→PME/contact encode and dataset field labels.

        Skips everything downstream of the energy fields: no force/gradient image
        unless ``oracle``, no scan buffers, no FEs downloads, no df/diss numpy.

        oracle=True (default): also composes E_total+F into ``img_FF_fdbm`` so the
        field can be sampled at ARBITRARY points on the GPU via
        ``afm.sample_fdbm(queries)`` (tricubic interpFE) — the query oracle for
        PME mesh-node rastering and per-atom core fits. This allocates the two
        RGBA float images (~2×16 B/voxel) — still far cheaper than a scan, and
        required if you want GPU point queries.
        oracle=False: returns only the component device fields — no compose
        image, no img_FF_fdbm. Minimal device footprint; download components
        with ``.get()`` / ``afm.download_image_rgba_xyz(img_vdw, shape)[...,3]``.

        Returns dict: 'E_pauli'/'E_ES' cl_arrays (fft scratch — valid until the
        next fields call on this shape), 'img_vdw' image (E=channel 3),
        'shape'/'step'/'origin'. When oracle, 'sample' = afm.sample_fdbm bound fn.
        """
        a = self.ensure_afm()
        fft = self.ensure_fft(ngrid, step)
        tip_tot_cl, tip_del_cl = self.ensure_tip(tip_mode, ngrid, step, margin, basis=basis, backend=backend)
        E_pauli_cl = fft.pauli_overlap_scaled_cl(rho_scf_cl, tip_tot_cl, step,
                                                 float(A_pauli), float(beta_pauli), out_cl=fft._xyz_E_pauli)
        E_es_cl = fft.es_fused_from_rho_cl(rho_diff_cl, tip_del_cl, step, out_cl=fft._xyz_E_es)
        img_vdw = a.compute_dispersion_to_img_cl(atomPos, atomTypes, origin, step, ngrid, C6_CO=float(C6_CO))
        out = {'E_pauli': E_pauli_cl, 'E_ES': E_es_cl, 'img_vdw': img_vdw,
               'shape': tuple(int(x) for x in ngrid[:3]), 'step': float(step),
               'origin': np.asarray(origin, dtype=np.float64)[:3]}
        if oracle:
            img_F = a.compose_E_and_gradient_fast_cl(E_pauli_cl, E_es_cl, img_vdw, step, out['shape'])
            a.setup_fdbm_grid_from_img(img_F, out['shape'], out['origin'], step)
            out['img_FF'] = a.img_FF_fdbm
            out['sample'] = a.sample_fdbm          # (N,3) world pts -> (E, F) on GPU
        return out

    def run_fields(self, atomPos, atomTypes, basis_hsd, work_dir, grid_spec, origin, step, ngrid,
                   A_pauli, beta_pauli, tip_mode='co', *,
                   projection='prolonged', basis='3ob-3-1', margin=4.0, C6_CO=30.0,
                   dm_in=None, oracle=True, charge_check=True):
        """prep → project → potentials only. Mirrors ``run`` minus the scan stage —
        for FDBM→PME/contact encode and dataset field labels.

        Returns dict {variant: fields_for_fit dict} + 'bufs' (device densities)
        + 'prep' (dm/eigvecs for reuse). No ScanResult — nothing was scanned.
        """
        atomPos = np.asarray(atomPos, dtype=np.float64)
        atomTypes = np.asarray(atomTypes, dtype=np.int32)
        ngrid = tuple(int(x) for x in ngrid[:3])
        variants_needed = ('diff', 'prol') if projection == 'prolonged' else \
                          ('diff', 'stock') if projection == 'stock' else \
                          ('diff', 'stock', 'prol')
        prep = self.dftb_prep(atomPos, atomTypes, basis_hsd, work_dir)
        bufs = self.project_densities(prep, grid_spec, step, dm=dm_in,
                                      variants=variants_needed, charge_check=charge_check)
        out = {'prep': prep, 'bufs': bufs}
        for variant in ('stock', 'prolonged'):
            if variant == 'stock' and 'stock' not in bufs:
                continue
            if variant == 'prolonged' and 'prol' not in bufs:
                continue
            rho_scf_cl = bufs['stock'] if variant == 'stock' else bufs['prol']
            out[variant] = self.fields_for_fit(
                rho_scf_cl, bufs['diff'], atomPos, atomTypes, origin, step, ngrid,
                A_pauli, beta_pauli, C6_CO=C6_CO, tip_mode=tip_mode, margin=margin,
                basis=basis, oracle=oracle)
        return out

    def scan_and_post(self, tag, atomPos, atomTypes, origin, step,
                      h_min=3.7, h_max=4.7, h_step=0.1, amp=1.0, amp_align=True,
                      K_LAT_Nm=0.5, K_RAD=20.0, bond_length=3.0, scan_margin=2.0,
                      plots=None, df_cmap='gray', cmap='seismic', outdir=None,
                      osc_dir=(0., 0., 1.), base_pos=(0., 0., 0.), need_ediss=True,
                      A_pauli=None, beta_pauli=None, stage_fields=None, tip_info=None,
                      stage_height=4.2, margin=4.0):
        """Stage-4 scan + postprocess — replicates run_fdbm_pp_from_density's scan block.

        The force-field image must already be set up on the device via fields()
        (setup_fdbm_grid_from_img). Host downloads limited to the FEs/tip_disp
        products (required output) — all densities stay on the GPU.
        """
        plots = set(plots or ())
        atomPos = np.asarray(atomPos, dtype=np.float64)
        step = float(step)
        h_df, h_Fz, h_scan = afm_utils.afm_df_height_stacks(h_min, h_max, h_step, amp=amp, amp_align=amp_align, osc_dir=osc_dir)
        osc_n = np.asarray(osc_dir, dtype=np.float64)
        osc_norm = float(np.linalg.norm(osc_n))
        if not np.isfinite(osc_norm) or osc_norm <= 0.0:
            raise ValueError(f'osc_dir must be a finite non-zero vector, got {osc_dir!r}')
        osc_n = osc_n / osc_norm
        scan_xs = np.arange(float(atomPos[:, 0].min() - scan_margin),
                            float(atomPos[:, 0].max() + scan_margin), step, dtype=np.float32)
        scan_ys = np.arange(float(atomPos[:, 1].min() - scan_margin),
                            float(atomPos[:, 1].max() + scan_margin), step, dtype=np.float32)
        pad_x = int(np.ceil(abs(float(amp) * osc_n[0]) / step - 1e-12))
        pad_y = int(np.ceil(abs(float(amp) * osc_n[1]) / step - 1e-12))
        base = np.asarray(base_pos, dtype=np.float64)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError(f'base_pos must contain three finite coordinates, got {base_pos!r}')
        required_margin_xy = np.array([scan_margin + pad_x * step + abs(float(base[0])), scan_margin + pad_y * step + abs(float(base[1]))])
        if np.any(required_margin_xy > float(margin) + 1e-9):
            raise ValueError(f"lateral amplitude leaves the force-field grid: required xy margins={required_margin_xy} Å exceed density margin={margin} Å")
        scan_xs_full = (float(scan_xs[0]) + (np.arange(len(scan_xs) + 2 * pad_x, dtype=np.float64) - pad_x) * step).astype(np.float32)
        scan_ys_full = (float(scan_ys[0]) + (np.arange(len(scan_ys) + 2 * pad_y, dtype=np.float64) - pad_y) * step).astype(np.float32)
        mol_z = float(atomPos[:, 2].max())
        K_LAT = afm.stiffness_Nm_to_eVA2(float(K_LAT_Nm))
        print(f"  K_LAT={K_LAT_Nm:.3f} N/m → {K_LAT:.4f} eV/Å²  K_RAD={K_RAD}  L={bond_length} Å")
        print(f"  df h=[{float(h_df[0]):.2f},{float(h_df[-1]):.2f}]  "
              f"Fz h=[{float(h_Fz[0]):.2f},{float(h_Fz[-1]):.2f}]  "
              f"z-scan=[{float(h_scan[0]):.2f},{float(h_scan[-1]):.2f}] amp={amp} align={amp_align} "
              f"osc_n=({osc_n[0]:.3f},{osc_n[1]:.3f},{osc_n[2]:.3f})")

        a = self.ensure_afm()
        FEs_full, tip_disp_full = a.scan_fdbm(
            scan_xs_full, scan_ys_full, h_scan, mol_z=mol_z,
            K_LAT=K_LAT, K_RAD=float(K_RAD), bond_length=float(bond_length),
            ppm_mode=True, use_fire=True, osc_dir=osc_dir, base_pos=base_pos)
        a.queue.finish()
        FEs_bwd_full = None
        if need_ediss:
            FEs_bwd_full, _ = a.scan_fdbm(
                scan_xs_full, scan_ys_full, h_scan, mol_z=mol_z,
                K_LAT=K_LAT, K_RAD=float(K_RAD), bond_length=float(bond_length),
                ppm_mode=True, use_fire=True, osc_dir=osc_dir, base_pos=base_pos, reverse=True)
            a.queue.finish()

        spacing = (float(scan_xs_full[1] - scan_xs_full[0]), float(scan_ys_full[1] - scan_ys_full[0]), float(h_scan[1] - h_scan[0]))
        df_full = afm.compute_df_amp_dir(FEs_full, spacing, osc_dir=osc_n, amp=float(amp))
        idx_df = [int(np.argmin(np.abs(h_scan - h))) for h in h_df]
        idx_Fz = [int(np.argmin(np.abs(h_scan - h))) for h in h_Fz]
        sx = slice(pad_x, pad_x + len(scan_xs))
        sy = slice(pad_y, pad_y + len(scan_ys))
        FEs = FEs_full[sx, sy]
        tip_disp = {key: val[sx, sy] for key, val in tip_disp_full.items()}
        Fz = FEs[..., 2][:, :, idx_Fz]
        df = df_full[sx, sy][:, :, idx_df]
        heights = h_df
        x_ext = [float(scan_xs[0]), float(scan_xs[-1])]
        y_ext = [float(scan_ys[0]), float(scan_ys[-1])]
        if outdir is not None and 'df' in plots:
            afm_utils.plot_grid_Fz(df, heights, f'df n=({osc_n[0]:.2f},{osc_n[1]:.2f},{osc_n[2]:.2f}) {tag}', f'df_{tag}.png',
                                   x_ext=x_ext, y_ext=y_ext, save_dir=outdir, cmap=df_cmap)
        if outdir is not None and 'fz' in plots:
            afm_utils.plot_grid_Fz(Fz, h_Fz, f'Fz {tag}', f'Fz_{tag}.png',
                                   x_ext=x_ext, y_ext=y_ext, save_dir=outdir, cmap=cmap)
        if FEs_bwd_full is not None:
            E_diss_full = afm.compute_dissipation(FEs_full, FEs_bwd_full, h_scan, h_df, amp=float(amp), osc_dir=osc_n)
            E_diss = E_diss_full[sx, sy]
        else:
            E_diss = np.zeros((len(scan_xs), len(scan_ys), len(h_df)), dtype=np.float32)
        if outdir is not None and 'ediss' in plots:
            afm_utils.plot_grid_Fz(E_diss, heights, f'E_diss {tag} amp={amp:.2f}', f'Ediss_{tag}.png',
                                   x_ext=x_ext, y_ext=y_ext, save_dir=outdir, cmap='hot', symmetric=False)
        ih = len(heights) // 2
        df_lo, df_hi = df[:, :, 0], df[:, :, -1]
        df_slice_rms = float(np.sqrt(np.mean((df_hi - df_lo) ** 2)))
        df_slice_scale = max(float(np.sqrt(np.mean(df_lo ** 2))), float(np.sqrt(np.mean(df_hi ** 2))), 1e-30)
        df_slice_corr = float(np.corrcoef(df_lo.ravel(), df_hi.ravel())[0, 1]) if np.std(df_lo) > 0.0 and np.std(df_hi) > 0.0 else np.nan
        print(f"  df @h={heights[ih]:.2f} / Fz @h={float(h_Fz[ih]):.2f}: "
              f"df=[{df.min():.3e},{df.max():.3e}] Fz=[{Fz.min():.3e},{Fz.max():.3e}]")
        print(f"  df z-slice first→last: RMSΔ={df_slice_rms:.3e} relative={df_slice_rms/df_slice_scale:.3f} corr={df_slice_corr:.6f}")

        result = afm_utils.ScanResult(
            df=df, Fz=Fz, heights=heights, heights_Fz=h_Fz,
            scan_xs=scan_xs, scan_ys=scan_ys,
            amp_align=bool(amp_align and abs(float(osc_n[2])) > 1e-12),
            FEs=FEs, tip_disp=tip_disp, E_diss=E_diss,
            backend_name='fdbm', fft_path='GPU',
        )
        result.tip = tip_info
        result.stage_path = None
        result.atomPos = atomPos
        result.origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
        result.step = step
        result.A = A_pauli
        result.beta = beta_pauli
        result.tag = tag
        result.h_scan = h_scan
        result.osc_dir = osc_n.copy()
        result.df_z_slice_rms = df_slice_rms
        result.df_z_slice_relative = df_slice_rms / df_slice_scale
        result.df_z_slice_corr = df_slice_corr
        result.path = 'FDBM_GPU'
        result.stage_fields = stage_fields
        return result

    # ── top-level convenience: one molecule → dict of ScanResults ────────────

    def run(self, atomPos, atomTypes, basis_hsd, work_dir, grid_spec, origin, step, ngrid,
            A_pauli, beta_pauli, tip_mode, outdir, *,
            projection='prolonged', basis='3ob-3-1', margin=4.0,
            h_min=3.7, h_max=4.7, h_step=0.1, amp=1.0, amp_align=True,
            K_LAT_Nm=0.5, K_RAD=20.0, bond_length=3.0, scan_margin=2.0,
            plots=None, df_cmap='gray', cmap='seismic', stage_height=4.2,
            C6_CO=30.0, osc_dir=(0., 0., 1.), base_pos=(0., 0., 0.),
            need_ediss=True, dm_in=None):
        """Full device-resident FDBM: one SCF → device densities → fields+scan per variant.

        projection: 'stock' | 'prolonged' | 'both' — returns {variant: ScanResult}.
        Mirrors ``run_fdbm_pp_from_density`` semantics (same plots/extras contract).
        """
        atomPos = np.asarray(atomPos, dtype=np.float64)
        atomTypes = np.asarray(atomTypes, dtype=np.int32)
        plots = set(plots or ())
        ngrid = tuple(int(x) for x in ngrid[:3])
        os.makedirs(outdir, exist_ok=True)
        print(f"\n=== FDBM_GPU_PIPELINE  A={A_pauli:.3f} β={beta_pauli:.4f}  tip={tip_mode}  "
              f"grid={ngrid}  projection={projection} ===")

        variants_needed = ('diff', 'prol') if projection == 'prolonged' else \
                          ('diff', 'stock') if projection == 'stock' else \
                          ('diff', 'stock', 'prol')
        prep = self.dftb_prep(atomPos, atomTypes, basis_hsd, work_dir)
        bufs = self.project_densities(prep, grid_spec, step, dm=dm_in, variants=variants_needed)

        # tip diagnostics (host info only; device tip handled by ensure_tip inside fields)
        tot_raw, del_raw = self.tip_raw(tip_mode, step, margin, basis=basis, output_dir=outdir, target_shape=ngrid)
        tip_info = {'peak': None, 'q': float(tot_raw.sum() * step ** 3),
                    'dq': float(del_raw.sum() * step ** 3), 'mirrorX': None, 'mirrorY': None, 'path': None}
        if 'tip' in plots:
            tip_info = afm_utils.plot_afm_tip_debug(tot_raw, del_raw, outdir, projection, step)

        results = {}
        for variant in ('stock', 'prolonged'):
            if variant == 'stock' and 'stock' not in bufs:
                continue
            if variant == 'prolonged' and 'prol' not in bufs:
                continue
            rho_scf_cl = bufs['stock'] if variant == 'stock' else bufs['prol']
            need_stage = 'stage' in plots
            _V, E_pauli, E_ES, E_vdw, F_total = self.fields(
                rho_scf_cl, bufs['diff'], atomPos, atomTypes, origin, step, ngrid,
                A_pauli, beta_pauli, C6_CO=C6_CO, tip_mode=tip_mode, margin=margin,
                basis=basis, download_fields=need_stage, download_F=need_stage, output_dir=outdir)
            stage_fields = stage_path = None
            if need_stage:
                stage_fields = {'rho_scf': np.ascontiguousarray(rho_scf_cl.get()),
                                'E_pauli': E_pauli, 'E_ES': E_ES, 'E_vdw': E_vdw,
                                'E_total': E_pauli + E_ES + E_vdw, 'Fz': -F_total[..., 2]}
                stage_path = afm_utils.plot_afm_fdbm_stages(
                    stage_fields, origin, step, atomPos, outdir, variant, z_above=float(stage_height))
            res = self.scan_and_post(
                variant, atomPos, atomTypes, origin, step,
                h_min=h_min, h_max=h_max, h_step=h_step, amp=amp, amp_align=amp_align,
                K_LAT_Nm=K_LAT_Nm, K_RAD=K_RAD, bond_length=bond_length, scan_margin=scan_margin,
                plots=plots, df_cmap=df_cmap, cmap=cmap, outdir=outdir,
                osc_dir=osc_dir, base_pos=base_pos, need_ediss=need_ediss,
                A_pauli=A_pauli, beta_pauli=beta_pauli, stage_fields=stage_fields,
                tip_info=tip_info, stage_height=stage_height, margin=margin)
            res.stage_path = stage_path
            results[variant] = res
        return results
