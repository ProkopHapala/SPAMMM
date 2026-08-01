"""RigidAssemblyExtension — unified rigid-body GUI: drag, MC/GA, PME.

One panel, three modes, all sharing a single `RigidEnsemble` (pose SSOT) and a single
`RigidBodyPairFF` GPU backend. Two build sources: pre-defined molecule files or
editor-drawn fragments (AtomicGraph connected components).

  - **Build** : "From file" loads nmol copies of a pre-defined molecule (PTCDA, NTCDI, …)
                onto a grid. "From editor" splits `backend.graph` into connected components
                via `graph_to_rigid_fragments` — each disconnected fragment becomes one
                rigid body at its mass-weighted CoM. When the assembly atom count differs
                from the editor graph, the graph is rebuilt from the assembly's world atoms
                (same pattern as `FoldedRigidExtension._ensure_backend_matched`).
  - **Drag**  : pick an atom in the main scene, pull the active molecule with an anchor
                spring (reuses `RigidBodyDynamics.update_anchors` + `run_pairff`); on
                release the pose is written back to the ensemble.
  - **MC/GA** : greedy best-of-batch planar moves (reuses
                `RigidBodyPairFF.greedy_energy_step` / `eval_energy_system`); accepted
                poses are written to the ensemble.
  - **PME**   : Pauli Master Equation scan over the assembly (reuses
                `spammm.quantum.pauli_scan.scan_xy` / `scan_xV` / `scan_1d`). Sites =
                rigid-molecule CoMs from the ensemble, oriented by `R(q)` (full SO(3),
                not φ-only). Multipole frame (Q0/Qzz) per species from a per-species dict.

Design (locked, see `doc/Tasks/RigidMoleculePose_SSOT.md`):
  - `RigidEnsemble` is the single pose authority the extension reads/writes. GPU buffers
    stay per-algorithm working storage; `_mb_*` is populated from ensemble reads.
  - `AtomicGraph`/`SPAMMM_GUI` stay independent — the extension writes atom positions
    back to the graph one-way, on demand, only for display. When counts mismatch (e.g.
    loading 4×PTCDA=104 atoms into an empty editor), the graph is rebuilt from assembly
    world atoms + enames + per-fragment bonds.
  - No new physics, no new VisPy window, no new pose store. All compute is delegated to
    existing modules. This file is glue + Qt panel + edit-mode handlers.

Reuse map (no duplication):
  - MC:        `RigidBodyPairFF.greedy_energy_step` / `eval_energy_system` / `packing_energy`
  - Drag:      `EditModeHandler` base + `FRManipMode`-style anchor pattern +
               `RigidBodyDynamics.update_anchors` / `run_pairff` / `set_active_body`
  - PME:       `pauli_scan.scan_xy` / `scan_xV` / `scan_1d` / `embed_sites_pme4` +
               `PauliSolverCL`; `RigidEnsemble.get_poses()` → `spos` + `R(q)`
  - Loaders:   `spammm.forcefields.RigidBodyDynamics.load_molecule` (general loader) +
               `graph_to_rigid_fragments` (AtomicGraph → connected components → rigid bodies)
  - Display:   main `AtomScene` (no second VisPy window); graph rebuild via
               `AtomicGraph` constructor + `backend.graph = new_graph`
  - Picking:   `AtomScene._pick_id_from_mouse` / `_id_to_idx` / `_ray_from_mouse`

User guide: `user_guide/RigidAssembly_GUI.md`. L0 tests: `tests/GUI/test_rigid_assembly_extension.py`.
"""
from __future__ import annotations

import os
import numpy as np
from PyQt5 import QtWidgets, QtCore

from .ExtensionManager import UIComponents
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH
from .EditModeHandlers import EditModeHandler
from .CollapsibleSection import CollapsibleSection

from spammm.forcefields.RigidEnsemble import RigidEnsemble
from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF, _body_sites_world, _quat_to_matrix_np, load_molecule, graph_to_rigid_fragments
from spammm.surfaces.FoldedRigid import remap_fit_for_molecule

# Molecule paths for dropdown (data, not logic)
_MOL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'mol')
_XYZ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'xyz')
MOL_PATHS = {
    'NTCDI':             os.path.join(_MOL_DIR, 'NTCDI.mol2'),
    'TBTAP':             os.path.join(_MOL_DIR, 'TBTAP.mol2'),
    'PTCDA':             os.path.join(_XYZ_DIR, 'PTCDA.xyz'),
    'formic_acid':       os.path.join(_XYZ_DIR, 'HCOOH.xyz'),
    'terephthalic_acid': os.path.join(_XYZ_DIR, 'terephthalic_acid.xyz'),
    'azaindol':          os.path.join(_XYZ_DIR, 'azaindol.xyz'),
    'uracil':            os.path.join(_XYZ_DIR, 'uracil.xyz'),
    'adenine':           os.path.join(_XYZ_DIR, 'adenine.xyz'),
}
_NO_QEQ = {'NTCDI'}  # NTCDI.mol2 has good file charges


# ─── helpers ─────────────────────────────────────────────────────────────────
def _status(window, msg):
    lbl = getattr(window, 'ra_status_label', None)
    if lbl is not None:
        lbl.setText(msg)
        QtWidgets.QApplication.processEvents()


def _assembly_world_atoms(window):
    """Flat (n_atoms_total, 3) world positions of all bodies from the ensemble + packs.

    Uses each pack's `rel` (body-frame sites, real atoms only) + ensemble poses.
    Returns (apos (n,3) f32, enames list[str]) or (None, None) if not built.
    """
    ens = window.ra_ensemble
    rbd = window.ra_rbd
    if ens is None or rbd is None or rbd._mb_packs is None:
        return None, None
    pos, quat = ens.get_poses()
    worlds = []
    enames_all = []
    for j, pack in enumerate(rbd._mb_packs):
        m = pack['types'] == 0
        rel_real = pack['rel'][m, :3]
        worlds.append(_body_sites_world(rel_real, pos[j], quat[j]))
        enames_all.extend([e for i, e in enumerate(pack['enames']) if m[i]])
    return np.vstack(worlds).astype(np.float32), enames_all


def _ensure_backend_matched(window, atom_positions, enames_all):
    """If the backend graph has a different number of atoms than the assembly,
    rebuild it from the assembly's world atoms + enames (mirrors FoldedRigid pattern).

    This is needed because the assembly may have many more atoms than what's in the
    editor graph (e.g. 4×PTCDA=104 atoms loaded from file vs 0 atoms in the editor).
    Rebuilding the graph makes display + picking work for the assembly.
    """
    backend = window.backend
    if not hasattr(backend, 'graph') or backend.graph is None:
        return
    atom_list = [a for a in backend.graph.atoms.values() if a.alive]
    if len(atom_list) == len(atom_positions):
        return  # counts match — just update positions
    if enames_all is None or len(enames_all) != len(atom_positions):
        return  # can't rebuild without enames
    # Rebuild graph from assembly atoms (same pattern as FoldedRigidExtension)
    from spammm.topology.AtomicGraph import AtomicGraph
    from spammm import elements as _elements
    new_graph = AtomicGraph()
    for i, e in enumerate(enames_all):
        pos = np.array(atom_positions[i], dtype=np.float64)
        atype = _elements.ELEMENT_DICT[e][0] if e in _elements.ELEMENT_DICT else 6
        new_graph.add_atom(pos, e, atype)
    # Add bonds within each fragment (from cached bonds0, offset by cumulative atom count)
    rbd = getattr(window, 'ra_rbd', None)
    bonds0 = getattr(window, 'ra_bonds0', None)
    if bonds0 is not None:
        atom_offset = 0
        atom_list_new = list(new_graph.atoms.values())
        for fi, frag_bonds in enumerate(bonds0):
            # Count real atoms in this fragment (from the pack's types mask)
            if rbd is not None and rbd._mb_packs is not None and fi < len(rbd._mb_packs):
                n_frag = int((rbd._mb_packs[fi]['types'] == 0).sum())
            else:
                n_frag = len(atom_positions) - atom_offset  # fallback
            for b in frag_bonds:
                i1, i2 = int(b[0]) + atom_offset, int(b[1]) + atom_offset
                if 0 <= i1 < len(atom_list_new) and 0 <= i2 < len(atom_list_new):
                    new_graph.add_bond(atom_list_new[i1], atom_list_new[i2])
            atom_offset += n_frag
    backend.graph = new_graph


def _update_graph(window, atom_positions, enames_all=None):
    """One-way ensemble → AtomicGraph write for display (reuses FoldedRigid pattern)."""
    backend = window.backend
    if hasattr(backend, 'graph') and hasattr(backend.graph, 'update_positions_from_array'):
        _ensure_backend_matched(window, atom_positions, enames_all)
        atom_list = [a for a in backend.graph.atoms.values() if a.alive]
        if len(atom_list) == len(atom_positions):
            backend.graph.update_positions_from_array(atom_positions)
    if hasattr(backend, '_sync_sys'):
        backend._sync_sys()
    if hasattr(window, 'refresh_view'):
        window.refresh_view()


def _sync_display(window):
    """Write current ensemble poses → AtomicGraph → refresh scene."""
    apos, enames = _assembly_world_atoms(window)
    if apos is not None:
        _update_graph(window, apos, enames)


def _upload_poses_to_gpu(window):
    """Push ensemble poses into the RigidBodyPairFF GPU buffers (poss/qrots)."""
    ens = window.ra_ensemble
    rbd = window.ra_rbd
    if ens is None or rbd is None:
        return
    pos, quat = ens.get_poses()
    pos4 = np.zeros((len(ens), 4), dtype=np.float32)
    pos4[:, :3] = pos
    pos4[:, 3] = getattr(rbd, 'mass_trans', 1.0)
    rbd.toGPU('poss', pos4)
    rbd.toGPU('qrots', quat.astype(np.float32).copy())


# ─── setup: build ensemble + RigidBodyPairFF from selected molecules ─────────
def _on_build(window):
    """Build RigidEnsemble + RigidBodyPairFF from either a file loader or the editor graph.

    Two sources (selected via `ra_source_combo`):
      - "From file": load `nmol` copies of the selected molecule from `LOADERS`,
        place on a grid (same `grid_pos` pattern as the testplot).
      - "From editor": split `window.backend.graph` into connected components
        (independent fragments), each becomes one rigid body at its mass-weighted CoM.
        `nmol`/`spacing` are ignored — the number of bodies = number of fragments.

    Reuses `RigidBodyPairFF.from_molecules` (which calls `_prepare_molecule_pack`
    internally) and `RigidEnsemble.from_poses`.
    """
    source = window.ra_source_combo.currentText()
    z_mol = float(window.ra_z_spin.value())
    try:
        if source == 'From editor':
            # ── AtomicGraph → fragments → rigid bodies ──────────────────────
            graph = getattr(window.backend, 'graph', None)
            if graph is None:
                _status(window, 'No backend.graph available — draw molecules first')
                return
            n_atoms = len([a for a in graph.atoms.values() if a.alive])
            if n_atoms == 0:
                _status(window, 'Graph is empty — draw molecules first')
                return
            _status(window, f'Splitting graph ({n_atoms} atoms) into fragments...')
            fragments, coms = graph_to_rigid_fragments(
                graph, qeq=not window.ra_no_qeq_chk.isChecked(), planarize=True)
            if not fragments:
                _status(window, 'No fragments found (graph has no alive atoms)')
                return
            molecules = [(f[0], f[1], f[2]) for f in fragments]  # (apos_rel, enames, REQs)
            nmol = len(fragments)
            # Body positions = CoMs (already computed); z set to z_mol
            pos = coms.astype(np.float32).copy()
            pos[:, 2] = z_mol
            quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (nmol, 1))
            tids = [f'frag{i}' for i in range(nmol)]
            bonds_list = [f[3] for f in fragments]
            _status(window, f'Found {nmol} fragments (sizes: {[len(f[1]) for f in fragments]})')
        else:
            # ── From file (existing path) ────────────────────────────────────
            mol_name = window.ra_mol_combo.currentText()
            nmol = int(window.ra_nmol_spin.value())
            spacing = float(window.ra_spacing_spin.value())
            if mol_name not in MOL_PATHS:
                _status(window, f'Unknown molecule: {mol_name}')
                return
            _status(window, f'Loading {mol_name}...')
            apos, enames, REQs, bonds = load_molecule(MOL_PATHS[mol_name], qeq=(mol_name not in _NO_QEQ) and (not window.ra_no_qeq_chk.isChecked()), name=mol_name)
            molecules = [(apos, enames, REQs)] * nmol
            # Grid positions (same pattern as testplot grid_pos)
            nx = int(np.ceil(np.sqrt(nmol)))
            pos = np.zeros((nmol, 3), dtype=np.float32)
            for i in range(nmol):
                ix, iy = i % nx, i // nx
                pos[i, 0] = (ix - 0.5 * (nx - 1)) * spacing
                pos[i, 1] = (iy - 0.5 * (nx - 1)) * spacing
                pos[i, 2] = z_mol
            quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (nmol, 1))
            # Small random rotation per molecule (like testplot)
            rng = np.random.default_rng(int(window.ra_seed_spin.value()))
            for i in range(nmol):
                phi0 = (i * 0.5 * np.pi) + float(rng.uniform(-0.35, 0.35))
                quat[i] = np.array([0, 0, np.sin(0.5 * phi0), np.cos(0.5 * phi0)], dtype=np.float32)
            tids = [mol_name] * nmol
            bonds_list = [bonds] * nmol

        # Build ensemble (pose SSOT)
        window.ra_ensemble = RigidEnsemble.from_poses(tids, pos, quat)
        # Build RigidBodyPairFF from ensemble poses
        pos, quat = window.ra_ensemble.get_poses()
        _status(window, f'Building PairFF ({nmol} mols)...')
        window.ra_rbd = RigidBodyPairFF.from_molecules(
            molecules, pos, quats=quat, active_body=0,
            He=-0.1, rc=3.0, w=0.7, k_z=0.0, z_target=z_mol, Hs=1.0, beta=1.7,
        )
        # Optional FAF substrate (only for file source — fragments have no fit)
        if source != 'From editor' and not window.ra_no_faf_chk.isChecked():
            mol_name = window.ra_mol_combo.currentText()
            from spammm.surfaces.FoldedRigid import load_or_fit_faf
            fit = load_or_fit_faf((apos, enames, REQs), mol_name=mol_name)
            fit = remap_fit_for_molecule(fit, REQs)
            # Tile atom_type_ids for nmol copies (all same species)
            at_ids = np.tile(fit['atom_type_ids'], nmol).astype(np.int32)
            fit['atom_type_ids'] = at_ids
            window.ra_rbd.attach_pairff_faf(fit, z_init=float(window.ra_z_init_spin.value()), k_z=0.0, enable=True)
            window.ra_fit = fit
        else:
            window.ra_fit = None
        # Cache per-pack bonds for display
        window.ra_bonds0 = bonds_list
        window.ra_E_last = window.ra_rbd.eval_energy_system(pos, quat, k_pack=float(window.ra_kpack_spin.value()))
        window.ra_kpack_last = float(window.ra_kpack_spin.value())
        _status(window, f'Built: {window.ra_ensemble.summary()}  device={window.ra_rbd.ctx.devices[0].name}')
        _sync_display(window)
    except Exception as e:
        import traceback; traceback.print_exc()
        _status(window, f'Build FAILED: {e}')


# ─── MC/GA mode ──────────────────────────────────────────────────────────────
def _on_mc_step(window):
    """Run one greedy MC step over all molecules (round-robin moved index)."""
    ens = window.ra_ensemble
    rbd = window.ra_rbd
    if ens is None or rbd is None:
        _status(window, 'Build assembly first')
        return
    nmol = len(ens)
    n_trial = int(window.ra_ntrial_spin.value())
    dxy = float(window.ra_dxy_spin.value())
    dphi = float(window.ra_dphi_spin.value())
    k_pack = float(window.ra_kpack_spin.value())
    if window.ra_kpack_last != k_pack:
        pos0, quat0 = ens.get_poses()
        window.ra_E_last = rbd.eval_energy_system(pos0, quat0, k_pack=k_pack)
        window.ra_kpack_last = k_pack
    seed = int(window.ra_seed_spin.value()) + 1000 + int(window.ra_mc_step_count)
    moved = [int(window.ra_mc_step_count) % nmol]
    pos, quat = ens.get_poses()
    pos, quat, E0, Ebest, acc, Ebatch = rbd.greedy_energy_step(
        pos, quat, moved, n_trial=n_trial, dxy=dxy, dphi=dphi,
        seed=seed, rmin_com=0.0, rmin_atom=float(window.ra_rmin_atom_spin.value()),
        k_pack=k_pack,
    )
    if acc:
        ens.set_poses(pos, quat)
        _upload_poses_to_gpu(window)
        _sync_display(window)
        window.ra_E_last += Ebest - E0
    E = window.ra_E_last
    window.ra_mc_step_count += 1
    window.ra_E_last = float(E)
    finite = Ebatch[np.isfinite(Ebatch)]
    _status(window, f'MC step {window.ra_mc_step_count}: E={E:.5f}  acc={int(acc)}  batch_min={finite.min():.5f}')


def _on_mc_run(window):
    """Run N MC steps in a tight loop (no timer — synchronous, like testplot)."""
    ens = window.ra_ensemble
    rbd = window.ra_rbd
    if ens is None or rbd is None:
        _status(window, 'Build assembly first')
        return
    n_steps = int(window.ra_mc_nsteps_spin.value())
    _status(window, f'Running {n_steps} MC steps...')
    for _ in range(n_steps):
        _on_mc_step(window)
    _status(window, f'MC done: {window.ra_mc_step_count} steps, E_last={window.ra_E_last:.5f}')


def _on_mc_reset(window):
    """Reset MC step counter (does not reset poses)."""
    window.ra_mc_step_count = 0
    _status(window, 'MC step counter reset')


# ─── Drag mode (edit-mode handler) ───────────────────────────────────────────
def _set_anchors(window, idx, target):
    """Set anchor spring on atom `idx` of the active molecule to `target` (world)."""
    rbd = window.ra_rbd
    if rbd is None:
        return
    anchors = rbd.anchors.copy()
    anchors[:, 3] = -1.0
    if idx >= 0:
        anchors[idx, :3] = target
        anchors[idx, 3] = float(window.ra_k_spring_spin.value())
    rbd.update_anchors(anchors)


def _closest_point_on_ray(atom_pos, r0, rd):
    rd = np.asarray(rd, dtype=np.float64)
    rd2 = float(np.dot(rd, rd))
    if rd2 < 1e-12:
        return np.asarray(r0, dtype=np.float64)
    t = float(np.dot(atom_pos - r0, rd) / rd2)
    return r0 + t * rd


def _make_ramdrag_mode(window):
    class RAManipMode(EditModeHandler):
        status_msg = "RigidAssembly Drag: LMB pick+drag atom to pull active molecule; release to drop"
        lock_drag = True
        capture_move = True

        def on_activate(self):
            if getattr(window, 'ra_rbd', None) is None:
                _status(window, "RA Drag: press Build first")
            self._pin = -1

        def _pick_idx(self, event):
            atom_id, dist = window.scene._pick_id_from_mouse(event.pos, max_dist=window.pick_radius)
            if atom_id < 0:
                return -1
            return window.scene._id_to_idx_safe(atom_id)

        def on_press(self, event, p_world, ctrl):
            if event.button != 1:
                return
            idx = self._pick_idx(event)
            if idx < 0:
                return
            self._pin = idx
            atom_pos = window.scene._pos[idx].astype(np.float64)
            r0, rd = window.scene._ray_from_mouse(event.pos)
            target = _closest_point_on_ray(atom_pos, r0, rd).astype(np.float32)
            _set_anchors(window, idx, target)
            _status(window, f"Pinned atom idx={idx}; drag to pull")

        def on_move(self, p_world, r0=None, rd=None):
            if self._pin < 0:
                return
            rbd = window.ra_rbd
            if rbd is None:
                return
            atom_pos = window.scene._pos[self._pin].astype(np.float64)
            if r0 is not None and rd is not None:
                target = _closest_point_on_ray(atom_pos, r0, rd)
            else:
                target = np.array([p_world[0], p_world[1], atom_pos[2]], dtype=np.float64)
            _set_anchors(window, self._pin, target.astype(np.float32))
            # Run a few PairFF steps to let the spring pull the molecule
            n_relax = int(window.ra_drag_nrelax_spin.value())
            dt = float(window.ra_drag_dt_spin.value())
            if n_relax > 0:
                rbd.run_pairff(n_relax, dt, fire=True)
                # Sync active pose back to ensemble
                rbd.sync_active_pose_from_gpu()
                pos_h, quat_h = rbd._mb_pos, rbd._mb_quat
                window.ra_ensemble.set_poses(pos_h, quat_h)
                _sync_display(window)

        def on_release(self, event, p_world, ctrl):
            if event.button != 1:
                return
            if self._pin < 0:
                return
            _set_anchors(window, -1, np.zeros(3, dtype=np.float32))
            self._pin = -1
            window.scene._pick_active = False
            _status(window, "Drag released; pose written to ensemble")
    return RAManipMode(window)


# ─── PME mode ────────────────────────────────────────────────────────────────
# Default multipole frame per species (Q0, Qzz). Extend as more species are calibrated.
# These are body-frame quadrupole parameters for the PME gating; the ensemble provides
# CoM + R(q), this dict provides the multipole strength per species.
_DEFAULT_MULTIPOLES = {
    'PTCDA':             (0.0, 0.0),   # placeholder — calibrate from QM
    'NTCDI':             (0.0, 0.0),
    'formic_acid':       (0.0, 0.0),
    'terephthalic_acid': (0.0, 0.0),
    'TBTAP':             (0.0, 0.0),
    'azaindol':          (0.0, 0.0),
    'uracil':            (0.0, 0.0),
    'adenine':           (0.0, 0.0),
}


def _poses_to_pme_sites(window, subset=None):
    """Build (spos (n,4), rots (n,3,3)) from ensemble poses for PME.

    spos[:, :3] = molecule CoM (rigid-body center); spos[:, 3] = Esite (from spinbox).
    rots = R(q) full 3×3 (not φ-only) — reuses `_quat_to_matrix_np`.
    subset: optional list of dense body indices (PME n_sites ≤ 4). If None, uses the
    first min(n_bodies, 4) bodies.
    """
    from spammm.quantum.pauli_scan import embed_sites_pme4
    ens = window.ra_ensemble
    if ens is None:
        return None, None, 0
    n_bodies = len(ens)
    if subset is None:
        subset = list(range(min(n_bodies, 4)))
    pos, quat = ens.get_poses(subset=subset)
    Esite = float(window.ra_pme_esite_spin.value())
    spos = np.zeros((len(subset), 4), dtype=np.float64)
    spos[:, :3] = pos.astype(np.float64)
    spos[:, 3] = Esite
    # R(q) per site — single-path _quat_to_matrix_np gives R used as `rel @ R.T`;
    # PME wants the molecular frame orientation, so we use R directly (consumer convention).
    rots = np.zeros((len(subset), 3, 3), dtype=np.float32)
    for k, q in enumerate(quat):
        rots[k] = _quat_to_matrix_np(q)
    spos4, rots4, n_act = embed_sites_pme4(spos, rots)
    return spos4, rots4, n_act


def _pme_params(window):
    """Build PME params dict from the panel spinboxes (mirrors ChargeRingsExtension)."""
    return {
        'nsite': min(len(window.ra_ensemble), 4) if window.ra_ensemble is not None else 4,
        'radius': float(window.ra_pme_radius_spin.value()),
        'phiRot': float(window.ra_pme_phirot_spin.value()),
        'Esite': float(window.ra_pme_esite_spin.value()),
        'W': float(window.ra_pme_W_spin.value()),
        'Q0': float(window.ra_pme_Q0_spin.value()),
        'Qzz': float(window.ra_pme_Qzz_spin.value()),
        'VBias': float(window.ra_pme_vbias_spin.value()),
        'z_tip': float(window.ra_pme_ztip_spin.value()),
        'zV0': float(window.ra_pme_zV0_spin.value()),
        'Temp': float(window.ra_pme_temp_spin.value()),
        'GammaT': float(window.ra_pme_gammat_spin.value()),
        'decay': float(window.ra_pme_decay_spin.value()),
        'L': float(window.ra_pme_L_spin.value()),
        'npix': int(window.ra_pme_npix_spin.value()),
        'GammaS': float(window.ra_pme_gammat_spin.value()),
        'dQ': 0.02,
        'zQd': 0.0,
        'zVd': 20.0,
        'Rtip': 3.0,
        'phi0_ax': 0.0,
    }


def _get_pme_solver(window):
    """Lazy-init PauliSolverCL (reuses ChargeRingsExtension pattern)."""
    if getattr(window, 'ra_pme_solver', None) is None:
        from spammm.quantum.PauliSolverCL import PauliSolverCL
        window.ra_pme_solver = PauliSolverCL(nSingle=4, preferred_vendor='nvidia', bPrint=False)
        _status(window, f'PME OpenCL: {window.ra_pme_solver.ctx.devices[0].name}')
    return window.ra_pme_solver


def _on_pme_scan_xy(window):
    """Run PME XY scan over the assembly sites (reuses pauli_scan.scan_xy)."""
    from spammm.quantum import pauli_scan as ps
    if window.ra_ensemble is None:
        _status(window, 'Build assembly first')
        return
    try:
        spos4, rots4, n_act = _poses_to_pme_sites(window)
        if spos4 is None:
            _status(window, 'No sites for PME')
            return
        params = _pme_params(window)
        params['nsite'] = n_act
        solver = _get_pme_solver(window)
        _status(window, f'PME XY scan: {n_act} sites...')
        xy = ps.scan_xy(solver, spos4, rots4, params, return_probs=False)
        window.ra_pme_xy = xy
        _status(window, f'PME XY done: STM range=[{xy["STM"].min():.3e},{xy["STM"].max():.3e}]')
        _show_pme_xy(window, xy, spos4)
    except Exception as e:
        import traceback; traceback.print_exc()
        _status(window, f'PME XY FAILED: {e}')


def _show_pme_xy(window, xy, spos):
    """Plot PME XY result in a matplotlib popup (reuses ChargeRingsExtension pattern)."""
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(xy['STM'], extent=xy['extent'], origin='lower', cmap='seismic')
    ax.plot(spos[:xy['n_active'], 0], spos[:xy['n_active'], 1], 'c+', ms=10, mew=1.5)
    plt.colorbar(im, ax=ax, label='STM')
    ax.set_title(f"PME XY  n_active={xy['n_active']}")
    fig.tight_layout()
    fig.show()
    window._ra_plot_windows = getattr(window, '_ra_plot_windows', [])
    window._ra_plot_windows.append(fig)


# ─── build_ui ────────────────────────────────────────────────────────────────
def build_ui(window):
    """Build the Rigid Assembly extension panel."""
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    apply_tight(layout)

    # State defaults
    window.ra_ensemble = None
    window.ra_rbd = None
    window.ra_fit = None
    window.ra_bonds0 = None
    window.ra_mc_step_count = 0
    window.ra_E_last = 0.0
    window.ra_kpack_last = None
    window.ra_pme_solver = None
    window.ra_pme_xy = None
    window._ra_plot_windows = []

    # ─── Build section ────────────────────────────────────────────────────
    build_sec = CollapsibleSection("Build Assembly", collapsed=False)
    build_host = QtWidgets.QWidget()
    build_l = QtWidgets.QVBoxLayout(build_host)
    build_l.setSpacing(SPACING)

    # Source selector: "From file" (load pre-defined molecule) vs "From editor" (split graph)
    src_row = QtWidgets.QHBoxLayout()
    src_row.addWidget(QtWidgets.QLabel('Source:'))
    window.ra_source_combo = QtWidgets.QComboBox()
    window.ra_source_combo.addItems(['From file', 'From editor'])
    window.ra_source_combo.setToolTip('From file: load nmol copies of a pre-defined molecule.\n'
                                       'From editor: split the current AtomicGraph into connected '
                                       'components (independent fragments); each fragment → one rigid body.')
    src_row.addWidget(window.ra_source_combo)
    src_row.addStretch()
    build_l.addLayout(src_row)

    row0 = QtWidgets.QHBoxLayout()
    row0.addWidget(QtWidgets.QLabel('Mol:'))
    window.ra_mol_combo = QtWidgets.QComboBox()
    window.ra_mol_combo.addItems(sorted(MOL_PATHS.keys()))
    row0.addWidget(window.ra_mol_combo)
    window.ra_nmol_spin = QtWidgets.QSpinBox(); window.ra_nmol_spin.setRange(1, 64); window.ra_nmol_spin.setValue(4); window.ra_nmol_spin.setMaximumWidth(50)
    row0.addWidget(QtWidgets.QLabel('nmol:')); row0.addWidget(window.ra_nmol_spin)
    window.ra_spacing_spin = QtWidgets.QDoubleSpinBox(); window.ra_spacing_spin.setRange(2.0, 50.0); window.ra_spacing_spin.setValue(16.0); window.ra_spacing_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_spacing_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    row0.addWidget(QtWidgets.QLabel('spacing:')); row0.addWidget(window.ra_spacing_spin)
    build_l.addLayout(row0)

    row1 = QtWidgets.QHBoxLayout()
    window.ra_z_spin = QtWidgets.QDoubleSpinBox(); window.ra_z_spin.setRange(-5.0, 20.0); window.ra_z_spin.setSingleStep(0.1); window.ra_z_spin.setValue(3.0); window.ra_z_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_z_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    row1.addWidget(QtWidgets.QLabel('z_mol:')); row1.addWidget(window.ra_z_spin)
    window.ra_z_init_spin = QtWidgets.QDoubleSpinBox(); window.ra_z_init_spin.setRange(0.0, 10.0); window.ra_z_init_spin.setSingleStep(0.1); window.ra_z_init_spin.setValue(3.0); window.ra_z_init_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_z_init_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    row1.addWidget(QtWidgets.QLabel('z_init:')); row1.addWidget(window.ra_z_init_spin)
    window.ra_seed_spin = QtWidgets.QSpinBox(); window.ra_seed_spin.setRange(0, 100000); window.ra_seed_spin.setValue(3); window.ra_seed_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_seed_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    row1.addWidget(QtWidgets.QLabel('seed:')); row1.addWidget(window.ra_seed_spin)
    build_l.addLayout(row1)

    row2 = QtWidgets.QHBoxLayout()
    window.ra_no_qeq_chk = QtWidgets.QCheckBox('no QEq')
    window.ra_no_faf_chk = QtWidgets.QCheckBox('no FAF')
    row2.addWidget(window.ra_no_qeq_chk); row2.addWidget(window.ra_no_faf_chk); row2.addStretch()
    window.ra_build_btn = QtWidgets.QPushButton('Build')
    window.ra_build_btn.clicked.connect(lambda: _on_build(window))
    row2.addWidget(window.ra_build_btn)
    build_l.addLayout(row2)
    build_sec.setContent(build_host)
    layout.addWidget(build_sec)

    # ─── MC/GA section ────────────────────────────────────────────────────
    mc_sec = CollapsibleSection("MC / GA Optimization", collapsed=False)
    mc_host = QtWidgets.QWidget()
    mc_l = QtWidgets.QVBoxLayout(mc_host)
    mc_l.setSpacing(SPACING)

    mc_row1 = QtWidgets.QHBoxLayout()
    window.ra_ntrial_spin = QtWidgets.QSpinBox(); window.ra_ntrial_spin.setRange(1, 4096); window.ra_ntrial_spin.setValue(128); window.ra_ntrial_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_ntrial_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    mc_row1.addWidget(QtWidgets.QLabel('n_trial:')); mc_row1.addWidget(window.ra_ntrial_spin)
    window.ra_dxy_spin = QtWidgets.QDoubleSpinBox(); window.ra_dxy_spin.setRange(0.01, 10.0); window.ra_dxy_spin.setSingleStep(0.1); window.ra_dxy_spin.setValue(1.5); window.ra_dxy_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_dxy_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    mc_row1.addWidget(QtWidgets.QLabel('dxy:')); mc_row1.addWidget(window.ra_dxy_spin)
    window.ra_dphi_spin = QtWidgets.QDoubleSpinBox(); window.ra_dphi_spin.setRange(0.01, 3.0); window.ra_dphi_spin.setSingleStep(0.05); window.ra_dphi_spin.setValue(0.8); window.ra_dphi_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_dphi_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    mc_row1.addWidget(QtWidgets.QLabel('dphi:')); mc_row1.addWidget(window.ra_dphi_spin)
    mc_l.addLayout(mc_row1)

    mc_row2 = QtWidgets.QHBoxLayout()
    window.ra_kpack_spin = QtWidgets.QDoubleSpinBox(); window.ra_kpack_spin.setRange(0.0, 1.0); window.ra_kpack_spin.setSingleStep(0.01); window.ra_kpack_spin.setValue(0.03); window.ra_kpack_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_kpack_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    mc_row2.addWidget(QtWidgets.QLabel('k_pack:')); mc_row2.addWidget(window.ra_kpack_spin)
    window.ra_rmin_atom_spin = QtWidgets.QDoubleSpinBox(); window.ra_rmin_atom_spin.setRange(0.0, 5.0); window.ra_rmin_atom_spin.setSingleStep(0.1); window.ra_rmin_atom_spin.setValue(1.6); window.ra_rmin_atom_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_rmin_atom_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    mc_row2.addWidget(QtWidgets.QLabel('rmin_atom:')); mc_row2.addWidget(window.ra_rmin_atom_spin)
    window.ra_mc_nsteps_spin = QtWidgets.QSpinBox(); window.ra_mc_nsteps_spin.setRange(1, 100000); window.ra_mc_nsteps_spin.setValue(50); window.ra_mc_nsteps_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_mc_nsteps_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    mc_row2.addWidget(QtWidgets.QLabel('n_steps:')); mc_row2.addWidget(window.ra_mc_nsteps_spin)
    mc_l.addLayout(mc_row2)

    mc_row3 = QtWidgets.QHBoxLayout()
    window.ra_mc_step_btn = QtWidgets.QPushButton('Step')
    window.ra_mc_step_btn.clicked.connect(lambda: _on_mc_step(window))
    mc_row3.addWidget(window.ra_mc_step_btn)
    window.ra_mc_run_btn = QtWidgets.QPushButton('Run')
    window.ra_mc_run_btn.clicked.connect(lambda: _on_mc_run(window))
    mc_row3.addWidget(window.ra_mc_run_btn)
    window.ra_mc_reset_btn = QtWidgets.QPushButton('Reset')
    window.ra_mc_reset_btn.clicked.connect(lambda: _on_mc_reset(window))
    mc_row3.addWidget(window.ra_mc_reset_btn)
    mc_row3.addStretch()
    mc_l.addLayout(mc_row3)
    mc_sec.setContent(mc_host)
    layout.addWidget(mc_sec)

    # ─── Drag section ─────────────────────────────────────────────────────
    drag_sec = CollapsibleSection("Drag (anchor spring)", collapsed=True)
    drag_host = QtWidgets.QWidget()
    drag_l = QtWidgets.QVBoxLayout(drag_host)
    drag_l.setSpacing(SPACING)
    drag_row = QtWidgets.QHBoxLayout()
    window.ra_k_spring_spin = QtWidgets.QDoubleSpinBox(); window.ra_k_spring_spin.setRange(0.01, 1000.0); window.ra_k_spring_spin.setSingleStep(0.5); window.ra_k_spring_spin.setValue(20.0); window.ra_k_spring_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_k_spring_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    drag_row.addWidget(QtWidgets.QLabel('k_spring:')); drag_row.addWidget(window.ra_k_spring_spin)
    window.ra_drag_nrelax_spin = QtWidgets.QSpinBox(); window.ra_drag_nrelax_spin.setRange(0, 500); window.ra_drag_nrelax_spin.setValue(20); window.ra_drag_nrelax_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_drag_nrelax_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    drag_row.addWidget(QtWidgets.QLabel('n_relax:')); drag_row.addWidget(window.ra_drag_nrelax_spin)
    window.ra_drag_dt_spin = QtWidgets.QDoubleSpinBox(); window.ra_drag_dt_spin.setRange(0.0001, 0.5); window.ra_drag_dt_spin.setSingleStep(0.005); window.ra_drag_dt_spin.setValue(0.02); window.ra_drag_dt_spin.setMaximumWidth(SPIN_MAX_WIDTH); window.ra_drag_dt_spin.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    drag_row.addWidget(QtWidgets.QLabel('dt:')); drag_row.addWidget(window.ra_drag_dt_spin)
    drag_l.addLayout(drag_row)
    drag_hint = QtWidgets.QLabel('Activate "RA Drag" edit mode (toolbar), then LMB drag atoms in the scene.')
    drag_hint.setWordWrap(True)
    drag_l.addWidget(drag_hint)
    drag_sec.setContent(drag_host)
    layout.addWidget(drag_sec)

    # ─── PME section ──────────────────────────────────────────────────────
    pme_sec = CollapsibleSection("PME (Charge Rings)", collapsed=True)
    pme_host = QtWidgets.QWidget()
    pme_l = QtWidgets.QVBoxLayout(pme_host)
    pme_l.setSpacing(SPACING)

    def _pme_spin(key, lo, hi, step, val, decimals=3, width=60):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(decimals); s.setValue(val); s.setMaximumWidth(width)
        setattr(window, f'ra_pme_{key}_spin', s)
        return s

    pme_row1 = QtWidgets.QHBoxLayout()
    pme_row1.addWidget(QtWidgets.QLabel('Esite:')); pme_row1.addWidget(_pme_spin('esite', -1.0, 1.0, 0.01, 0.0))
    pme_row1.addWidget(QtWidgets.QLabel('W:')); pme_row1.addWidget(_pme_spin('W', 0.0, 0.5, 0.01, 0.05))
    pme_row1.addWidget(QtWidgets.QLabel('Q0:')); pme_row1.addWidget(_pme_spin('Q0', 0.0, 5.0, 0.1, 0.0, decimals=2))
    pme_row1.addWidget(QtWidgets.QLabel('Qzz:')); pme_row1.addWidget(_pme_spin('Qzz', -20.0, 20.0, 0.5, 0.0, decimals=2))
    pme_l.addLayout(pme_row1)

    pme_row2 = QtWidgets.QHBoxLayout()
    pme_row2.addWidget(QtWidgets.QLabel('VBias:')); pme_row2.addWidget(_pme_spin('vbias', 0.0, 3.0, 0.05, 1.0))
    pme_row2.addWidget(QtWidgets.QLabel('z_tip:')); pme_row2.addWidget(_pme_spin('ztip', 1.0, 15.0, 0.5, 5.0, decimals=2))
    pme_row2.addWidget(QtWidgets.QLabel('Temp:')); pme_row2.addWidget(_pme_spin('temp', 0.1, 100.0, 0.5, 1.0, decimals=2))
    pme_row2.addWidget(QtWidgets.QLabel('GammaT:')); pme_row2.addWidget(_pme_spin('gammat', 1e-4, 1.0, 0.01, 0.01, decimals=4))
    pme_l.addLayout(pme_row2)

    pme_row3 = QtWidgets.QHBoxLayout()
    pme_row3.addWidget(QtWidgets.QLabel('decay:')); pme_row3.addWidget(_pme_spin('decay', 0.05, 2.0, 0.05, 0.5))
    pme_row3.addWidget(QtWidgets.QLabel('L:')); pme_row3.addWidget(_pme_spin('L', 5.0, 40.0, 1.0, 20.0, decimals=1))
    pme_row3.addWidget(QtWidgets.QLabel('npix:')); 
    npix_s = QtWidgets.QSpinBox(); npix_s.setRange(20, 200); npix_s.setValue(80); npix_s.setMaximumWidth(SPIN_MAX_WIDTH); npix_s.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
    window.ra_pme_npix_spin = npix_s
    pme_row3.addWidget(npix_s)
    pme_l.addLayout(pme_row3)

    pme_row4 = QtWidgets.QHBoxLayout()
    pme_row4.addWidget(QtWidgets.QLabel('zV0:')); pme_row4.addWidget(_pme_spin('zV0', -5.0, 5.0, 0.1, 0.0, decimals=2))
    pme_row4.addWidget(QtWidgets.QLabel('radius:')); pme_row4.addWidget(_pme_spin('radius', 1.0, 20.0, 0.1, 5.0, decimals=2))
    pme_row4.addWidget(QtWidgets.QLabel('phiRot:')); pme_row4.addWidget(_pme_spin('phirot', -6.3, 6.3, 0.1, 0.0))
    pme_l.addLayout(pme_row4)

    pme_btn_row = QtWidgets.QHBoxLayout()
    window.ra_pme_xy_btn = QtWidgets.QPushButton('Scan XY')
    window.ra_pme_xy_btn.clicked.connect(lambda: _on_pme_scan_xy(window))
    pme_btn_row.addWidget(window.ra_pme_xy_btn)
    pme_btn_row.addStretch()
    pme_l.addLayout(pme_btn_row)
    pme_hint = QtWidgets.QLabel('Sites = rigid-molecule CoMs from the ensemble, oriented by R(q). '
                                 'PME n_sites ≤ 4; first min(n_bodies,4) bodies used.')
    pme_hint.setWordWrap(True)
    pme_l.addWidget(pme_hint)
    pme_sec.setContent(pme_host)
    layout.addWidget(pme_sec)

    # ─── Status ───────────────────────────────────────────────────────────
    window.ra_status_label = QtWidgets.QLabel('Ready')
    layout.addWidget(window.ra_status_label)
    layout.addStretch()

    # ─── Edit modes ───────────────────────────────────────────────────────
    window.register_mode_handler('ra_drag', _make_ramdrag_mode(window))

    edit_modes = [
        ('RA Drag', lambda: window.set_edit_mode('ra_drag')),
    ]
    view_modes = []
    help_text = {
        'Build': 'Select molecule + count, set spacing/z, press Build. Creates RigidEnsemble + RigidBodyPairFF.',
        'MC/GA': 'Greedy best-of-batch planar moves. Step = one molecule; Run = N steps. Accepted poses → ensemble.',
        'Drag': 'Activate "RA Drag" edit mode in the toolbar, then LMB drag atoms in the scene to pull the active molecule.',
        'PME': 'Sites = rigid-molecule CoMs (first ≤4). Scan XY runs pauli_scan.scan_xy over the assembly.',
    }
    return UIComponents(panel=panel, edit_modes=edit_modes, view_modes=view_modes, help_text=help_text)


# ─── GUI-script entry point (mirrors FoldedRigidExtension.prepare_folded_rigid) ──
def prepare_rigid_assembly(window, mol='PTCDA', nmol=4, spacing=16.0, z=3.0, run_mc=False, n_steps=50):
    """Programmatically build the assembly and optionally run MC steps.

    GUI-script equivalent of user clicks: select molecule, set count/spacing/z,
    press Build, optionally run MC.
    """
    from .gui_script_utils import expand_extension_panel, process_events
    expand_extension_panel(window, 'rigid_assembly', open=True)
    window.ra_mol_combo.setCurrentText(mol)
    window.ra_nmol_spin.setValue(nmol)
    window.ra_spacing_spin.setValue(spacing)
    window.ra_z_spin.setValue(z)
    process_events()
    _on_build(window)
    process_events()
    if run_mc:
        window.ra_mc_nsteps_spin.setValue(n_steps)
        _on_mc_run(window)
