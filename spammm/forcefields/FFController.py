"""
FFController.py — Pure logic orchestrator for forcefield-based relaxation and MD.

Purpose: Bridge AtomicSystem → forcefield build → GPU relaxation → positions/forces download.
No Qt dependencies — pure Python/NumPy/OpenCL. The GUI extension layer calls this.

Key functionality:
  - build_ff(AtomicSystem, ff_type='spff') — build forcefield and init MD engine
  - relax_step() — one GPU MD step (damped velocity Verlet / FIRE)
  - relax_n(nsteps) — batch relaxation, returns final energy
  - get_state() — download {positions, forces, energy} from GPU
  - set_pinned(mask) / toggle_pin(idx) / clear_pins() — constraint management
  - update_positions(apos) — push new positions to GPU (e.g. after user drag)
  - teardown() — release GPU resources

Role in SPAMMM: Called by FFExtension.py (GUI). Wraps SPFFbuilder + SPFF_cl.
"""

import numpy as np

from .SPFFbuilder import SPFF
from .SPFF_cl import SPFF_cl
from ..topology.FFparams import SPFFparams

# Mass lookup for apos.w (kernel uses it for velocity Verlet)
MASS_MAP = {'H': 1.0, 'C': 12.0, 'N': 14.0, 'O': 16.0, 'S': 32.0, 'P': 31.0,
            'F': 19.0, 'Cl': 35.0, 'Br': 80.0, 'I': 127.0, 'Si': 28.0}

# Default relaxation parameters
DEFAULT_DT = 0.01
DEFAULT_DAMP = 0.9   # 0.1 under-damps large PAHs (fmax plateaus ~12); 0.9 matches test_forcefield / serial parity
DEFAULT_FLIMIT = 100.0
DEFAULT_PIN_K = 1e6  # constraint stiffness for pinned atoms


class FFController:
    """Orchestrates forcefield building and GPU relaxation for a single molecular system.

    Lifecycle:
        1. build_ff(sys)  — build FF, init MD, upload to GPU
        2. relax_step() / relax_n(n)  — run relaxation
        3. get_state()  — download results
        4. set_pinned() / clear_pins()  — manage constraints
        5. teardown()  — cleanup (optional, GC handles it)
    """

    def __init__(self, ff_type='spff', enable_nonbond=False, debug_build_options=None):
        self.ff_type = ff_type
        self.enable_nonbond = enable_nonbond
        self.debug_build_options = debug_build_options
        self.spff = None
        self.md = None
        self.params = None
        self.natoms = 0
        self._pinned_mask = np.array([], dtype=bool)
        self._pinned_positions = np.zeros((0, 3), dtype=np.float32)
        self._built = False

    @property
    def is_built(self):
        return self._built

    def _load_params(self):
        """Load forcefield parameters from data/ directory."""
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, "../../data/")
        self.params = SPFFparams(data_path)

    def build_ff(self, sys, ff_type=None):
        """Build forcefield from AtomicSystem and initialize GPU MD engine.

        Args:
            sys: AtomicSystem with apos, enames, bonds, neighs
            ff_type: 'spff' (default) or 'uff' (future)

        Returns:
            dict with natoms, nnode, nvecs
        """
        if ff_type is not None:
            self.ff_type = ff_type

        if self.params is None:
            self._load_params()

        # Ensure sys has neighbors
        if sys.ngs is None:
            sys.neighs()
        if sys.bonds is None:
            sys.findBonds()

        # Assign atom types — prefer atom_types_spff (hybridization-aware) over generic enames
        at_spff = getattr(sys, 'atom_types_spff', None)
        if at_spff is not None:
            sys.atypes = np.array([self.params.getAtomType(t, bErr=False) for t in at_spff], dtype=np.int32)
        else:
            sys.atypes = np.array([self.params.getAtomType(e, bErr=False) for e in sys.enames], dtype=np.int32)

        if self.ff_type == 'spff':
            return self._build_spff(sys)
        elif self.ff_type == 'uff':
            # TODO: UFF path needs UFF_cl kernel integration with SPFF_cl
            raise NotImplementedError("UFF relaxation path not yet integrated with MD engine")
        else:
            raise ValueError(f"Unknown ff_type={self.ff_type!r}, expected 'spff' or 'uff'")

    def _build_spff(self, sys):
        """Build SPFF forcefield and pack to GPU."""
        # Create SPFF and build topology
        self.spff = SPFF()
        self.spff.toSPFFsp3_loc(sys, self.params.atom_types_map)

        # Set mass in apos.w for dynamics
        for ia in range(self.spff.natoms):
            e = sys.enames[ia] if ia < len(sys.enames) else 'C'
            self.spff.apos[ia, 3] = MASS_MAP.get(e, 12.0)

        self.natoms = self.spff.natoms

        # Create MD engine and upload
        self.md = SPFF_cl(enable_nonbond=self.enable_nonbond,
                                     debug_build_options=self.debug_build_options)
        self.md.realloc(self.spff, nSystems=1)
        self.md.upload_all_systems()
        self.md.setup_kernels()

        # Zero velocities
        import pyopencl as cl
        cl.enqueue_fill_buffer(self.md.queue, self.md.buffer_dict['avel'],
                               np.float32(0), 0, self.md.buffer_dict['avel'].size)
        self.md.queue.finish()

        # Reset pin state
        self._pinned_mask = np.zeros(self.natoms, dtype=bool)
        self._pinned_positions = np.zeros((self.natoms, 3), dtype=np.float32)

        self._built = True
        return {'natoms': self.natoms, 'nnode': self.spff.nnode, 'nvecs': self.spff.nvecs}

    def relax_step(self, dt=None, damp=None, Flimit=None, do_nb=None):
        """Run a single GPU relaxation step.

        Args:
            dt: timestep (default: from build or DEFAULT_DT)
            damp: damping coefficient (default: from build or DEFAULT_DAMP)
            Flimit: max force for FIRE (default: DEFAULT_FLIMIT)
            do_nb: include non-bonded interactions (default: self.enable_nonbond)
        """
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        if dt is not None or damp is not None or Flimit is not None:
            self.md.set_md_params(
                dt=dt if dt is not None else DEFAULT_DT,
                damp=damp if damp is not None else DEFAULT_DAMP,
                Flimit=Flimit if Flimit is not None else DEFAULT_FLIMIT,
            )
        do_nb = do_nb if do_nb is not None else self.enable_nonbond
        self.md.run_step_basic(do_nb=do_nb)

    def _can_use_serial(self, do_nb):
        """Check if relax_serial is applicable (single system, small enough, no non-bonded)."""
        from .SPFF_cl import SPFF_cl
        return (self.md.nSystems == 1
                and self.md.nvecs <= SPFF_cl.SERIAL_MAX_NVEC
                and self.md.nnode <= SPFF_cl.SERIAL_MAX_NNODE
                and self.md.natoms <= SPFF_cl.SERIAL_MAX_NATOM
                and self.md.nvecs <= SPFF_cl.SERIAL_WG_SIZE
                and not do_nb)

    def relax_n(self, nsteps=100, dt=None, damp=None, Flimit=None, do_nb=None):
        """Run nsteps of relaxation on GPU. Returns final energy.
        Uses relax_serial (single-kernel local-memory) when possible for ~150x speedup."""
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        _dt = dt if dt is not None else DEFAULT_DT
        _damp = damp if damp is not None else DEFAULT_DAMP
        _Flimit = Flimit if Flimit is not None else DEFAULT_FLIMIT
        _do_nb = do_nb if do_nb is not None else self.enable_nonbond
        if self._can_use_serial(_do_nb):
            self.md.relax_serial(nsteps=nsteps, dt=_dt, damp=_damp, Flimit=_Flimit)
        else:
            self.md.set_md_params(dt=_dt, damp=_damp, Flimit=_Flimit)
            self.md.relax_batch(nsteps=nsteps, do_nb=_do_nb)
        return self.md.get_total_energy()

    def get_fmax(self):
        """Return max force magnitude across all atoms (convergence metric)."""
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        forces = self.md.get_forces()[:self.natoms, :3]
        return float(np.max(np.linalg.norm(forces, axis=1)))

    def relax_until_converged(self, fmax_tol=0.05, max_steps=5000, dt=None, damp=None, Flimit=None, do_nb=None, callback=None, batch_size=None):
        """Run relaxation until max force drops below fmax_tol or max_steps reached.

        Args:
            fmax_tol: convergence threshold on max force magnitude (eV/Å)
            max_steps: hard stop to prevent infinite loop
            dt, damp, Flimit: MD parameters
            do_nb: include non-bonded
            callback: optional fn(step, E, fmax) called every batch — return False to abort
            batch_size: steps per GPU batch (default: same as max_steps, i.e. single batch)

        Returns:
            dict: {converged, nsteps, energy, fmax}
        """
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        _dt = dt if dt is not None else DEFAULT_DT
        _damp = damp if damp is not None else DEFAULT_DAMP
        _Flimit = Flimit if Flimit is not None else DEFAULT_FLIMIT
        _do_nb = do_nb if do_nb is not None else self.enable_nonbond
        self.md.set_md_params(dt=_dt, damp=_damp, Flimit=_Flimit)
        batch = batch_size if batch_size is not None else max_steps  # single batch if not specified
        use_serial = self._can_use_serial(_do_nb)
        total = 0
        E = self.md.get_total_energy()
        fmax = self.get_fmax()
        # do-while: always run at least one batch (fmax=0 before any force eval → false "converged")
        first = True
        while first or (total < max_steps and fmax > fmax_tol):
            first = False
            n = min(batch, max_steps - total)
            if use_serial:
                self.md.relax_serial(nsteps=n, dt=_dt, damp=_damp, Flimit=_Flimit)
            else:
                self.md.relax_batch(nsteps=n, do_nb=_do_nb)
            total += n
            E = self.md.get_total_energy()
            fmax = self.get_fmax()
            if callback is not None and not callback(total, E, fmax):
                break
        return {'converged': fmax <= fmax_tol, 'nsteps': total, 'energy': E, 'fmax': fmax}

    def get_state(self):
        """Download current state from GPU.

        Returns:
            dict with:
                'positions': (natoms, 3) float32
                'forces': (natoms, 4) float32 — .xyz = force, .w = energy contribution
                'energy': float — total energy
                'pinned_mask': (natoms,) bool
        """
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        pos = self.md.get_positions()
        forces = self.md.get_forces()
        energy = float(np.sum(forces[:self.natoms, 3]))
        return {
            'positions': pos,
            'forces': forces[:self.natoms],
            'energy': energy,
            'pinned_mask': self._pinned_mask.copy(),
        }

    def get_positions(self):
        """Download and return (natoms, 3) positions from GPU."""
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        return self.md.get_positions()

    def get_forces(self):
        """Download and return (natoms, 4) forces from GPU."""
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        return self.md.get_forces()[:self.natoms]

    def get_energy(self):
        """Download and return total energy from GPU."""
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        return self.md.get_total_energy()

    def update_positions(self, apos):
        """Push new positions to GPU (e.g. after user drag in GUI).

        Args:
            apos: (natoms, 3) or (natoms, 4) array of new positions
        """
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        apos = np.asarray(apos, dtype=np.float32)
        if apos.shape[0] != self.natoms:
            raise ValueError(f"apos.shape[0]={apos.shape[0]} != natoms={self.natoms}")
        # Update spff.apos (keeping .w = mass)
        self.spff.apos[:self.natoms, :3] = apos[:, :3]
        # Re-upload to GPU
        self.md.upload_all_systems()
        # Re-apply pins if any are active
        if np.any(self._pinned_mask):
            self._apply_pinned()

    def set_pinned(self, mask, positions=None):
        """Pin atoms according to a boolean mask.

        Args:
            mask: (natoms,) bool array — True = pinned
            positions: (natoms, 3) target positions. If None, use current GPU positions.
        """
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.natoms:
            raise ValueError(f"mask.shape={mask.shape} != ({self.natoms},)")
        self._pinned_mask = mask.copy()
        if positions is None:
            positions = self.get_positions()
        self._pinned_positions = positions.copy()
        self._apply_pinned()

    def toggle_pin(self, idx):
        """Toggle pin status for a single atom. Returns new pin state (bool)."""
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        if idx < 0 or idx >= self.natoms:
            raise ValueError(f"idx={idx} out of range [0, {self.natoms})")
        self._pinned_mask[idx] = ~self._pinned_mask[idx]
        if self._pinned_mask[idx]:
            # Pin: save current position
            pos = self.get_positions()
            self._pinned_positions[idx] = pos[idx]
        self._apply_pinned()
        return bool(self._pinned_mask[idx])

    def pin_selected(self, indices):
        """Pin a set of atom indices to their current positions.

        Args:
            indices: array-like of atom indices to pin
        """
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        indices = np.asarray(indices, dtype=np.int32)
        pos = self.get_positions()
        self._pinned_mask[indices] = True
        self._pinned_positions[indices] = pos[indices]
        self._apply_pinned()

    def unpin_selected(self, indices):
        """Unpin a set of atom indices.

        Args:
            indices: array-like of atom indices to unpin
        """
        if not self._built:
            raise RuntimeError("FF not built — call build_ff() first")
        indices = np.asarray(indices, dtype=np.int32)
        self._pinned_mask[indices] = False
        self._apply_pinned()

    def clear_pins(self):
        """Remove all pin constraints."""
        if not self._built:
            return
        self._pinned_mask[:] = False
        self.md.clear_pinned()

    def get_pinned_mask(self):
        """Return (natoms,) bool mask of pinned atoms."""
        return self._pinned_mask.copy()

    def get_pinned_indices(self):
        """Return array of pinned atom indices."""
        return np.where(self._pinned_mask)[0]

    def _apply_pinned(self):
        """Upload current pin state to GPU constraint buffers."""
        indices = np.where(self._pinned_mask)[0]
        if len(indices) == 0:
            self.md.clear_pinned()
        else:
            positions = self._pinned_positions[indices]
            self.md.set_pinned(indices, positions, K=DEFAULT_PIN_K)

    def teardown(self):
        """Release GPU resources. Safe to call multiple times."""
        self._built = False
        self.spff = None
        self.md = None
        self._pinned_mask = np.array([], dtype=bool)
        self._pinned_positions = np.zeros((0, 3), dtype=np.float32)

    def rebuild(self, sys):
        """Rebuild forcefield with new topology (e.g. after atom add/remove).

        Preserves pin mask for atoms that still exist.
        """
        old_mask = self._pinned_mask.copy() if self._built else None
        old_positions = self._pinned_positions.copy() if self._built else None
        self.teardown()
        self.build_ff(sys)
        if old_mask is not None and len(old_mask) == self.natoms:
            self._pinned_mask = old_mask
            self._pinned_positions = old_positions
            self._apply_pinned()
