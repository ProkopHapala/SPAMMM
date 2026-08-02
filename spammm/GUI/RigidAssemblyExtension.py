"""RigidAssemblyExtension — unified rigid-body GUI: drag, MC/GA, PME.

One panel, three modes, all sharing a single `RigidEnsemble` (pose SSOT) and a single
`RigidBodyPairFF` GPU backend. Two build sources: pre-defined molecule files or
editor-drawn fragments (AtomicGraph connected components).

  - **Build** : "From file" loads nmol copies of a pre-defined molecule (PTCDA, NTCDI, …)
                onto a grid. Supports comma-separated species (e.g. "NTCDI,uracil") for
                mixed-species assemblies (round-robin body order). "From editor" splits
                `backend.graph` into connected components via `graph_to_rigid_fragments`
                — each disconnected fragment becomes one rigid body at its mass-weighted
                CoM. FAF substrate is fitted on the first fragment for editor builds.
                When the assembly atom count or enames differ from the editor graph,
                the graph is rebuilt from the assembly's world atoms (same pattern as
                `FoldedRigidExtension._ensure_backend_matched`). **Caveat:** enames must
                match, not just count — see `doc/Caveats.md` §9.
  - **Drag**  : pick an atom in the main scene, pull its molecule with an anchor spring
                while all molecules relax concurrently; on release all poses are written
                back to the ensemble. **Shift+LMB** toggles dynamic↔static,
                **RMB** soft-deletes (rejected on last live body).
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
               `RigidBodyDynamics.update_anchors` / `run_multimol_md`
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
from spammm.GUI.LayoutPolicy import apply_tight, SPACING, ROW_SPACING, make_flow, BUTTON_MAX_WIDTH, SPIN_MAX_WIDTH, COMBO_MAX_WIDTH, AutoGridPlacer
from .EditModeHandlers import EditModeHandler
from .CollapsibleSection import CollapsibleSection

from spammm.forcefields.RigidEnsemble import RigidEnsemble
from spammm.forcefields.RigidBodyDynamics import RigidBodyPairFF, _body_sites_world, _quat_to_matrix_np
from spammm.forcefields.RigidBodyUtils import load_molecule, graph_to_rigid_fragments, greedy_energy_step, grid_pos
from spammm.surfaces.FoldedRigid import Z_SURF_TOP, remap_fit_for_molecule

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
    'benzoic_acid':      os.path.join(_XYZ_DIR, 'benzoic_acid.xyz'),
}
_NO_QEQ = {'NTCDI'}  # NTCDI.mol2 has good file charges


# ─── helpers ─────────────────────────────────────────────────────────────────
def _status(window, msg):
    lbl = getattr(window, 'ra_status_label', None)
    if lbl is not None:
        lbl.setText(msg)
        QtWidgets.QApplication.processEvents()


def _body_state_counts(window):
    """Return (n_dynamic, n_static, n_deleted) from the RBD body-state buffer."""
    rbd = window.ra_rbd
    if rbd is None or rbd._body_state_host is None:
        return 0, 0, 0
    s = rbd._body_state_host
    return int((s == 1).sum()), int((s == 0).sum()), int((s < 0).sum())


def _toggle_body_state(window, body):
    """Toggle dynamic ↔ static for the given body. Syncs GPU pose → ensemble first."""
    rbd = window.ra_rbd
    ens = window.ra_ensemble
    if rbd is None or ens is None:
        return
    # Sync latest GPU pose into ensemble before freezing (per spec §1.2)
    _sync_ensemble_from_gpu(window)
    cur = int(rbd._body_state_host[body])
    new_state = 0 if cur == 1 else 1  # toggle dynamic ↔ static
    rbd.set_body_state(body, new_state)
    # Update ensemble metadata (active = not static)
    ens._bodies[body].active = (new_state == 1)
    _update_state_overlays(window)
    n_dyn, n_stat, n_del = _body_state_counts(window)
    _status(window, f"Body {body}: {'dynamic' if new_state == 1 else 'static'}  (dynamic={n_dyn} static={n_stat} deleted={n_del})")


def _soft_delete_body(window, body):
    """Soft-delete a body (state=-1). Reject if it's the last live body."""
    rbd = window.ra_rbd
    ens = window.ra_ensemble
    if rbd is None or ens is None:
        return
    s = rbd._body_state_host.copy()
    n_live = int((s >= 0).sum())
    if n_live <= 1:
        _status(window, "Cannot delete the last live body")
        return
    s[body] = -1
    rbd.set_body_states(s)
    ens._bodies[body].alive = False
    _update_state_overlays(window)
    n_dyn, n_stat, n_del = _body_state_counts(window)
    _status(window, f"Body {body}: deleted  (dynamic={n_dyn} static={n_stat} deleted={n_del})")


def _update_state_overlays(window):
    """Refresh static outline overlay + status counts. Called after state changes."""
    # TODO: static outline overlay (Phase C visual)
    n_dyn, n_stat, n_del = _body_state_counts(window)
    lbl = getattr(window, 'ra_state_counts_label', None)
    if lbl is not None:
        lbl.setText(f"dyn={n_dyn} stat={n_stat} del={n_del}")


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
        # Counts match — but enames may differ (e.g. editor XYZ order vs assembly BFS order)
        # Only skip rebuild if enames also match
        if enames_all is not None and len(enames_all) == len(atom_positions):
            graph_enames = [str(a.ename) for a in atom_list]
            if graph_enames == list(enames_all):
                return  # counts AND enames match — just update positions
        # Enames differ but counts match — fall through to rebuild
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
        dirty = getattr(window, '_afm_dirty', None)
        if dirty is not None:
            dirty.mark_geometry_changed()


def _update_ra_substrate_overlay(window):
    """Show FAF substrate potential map under the assembly (shared VispyUtils.update_faf_map_overlay).

    Uses the same potential_to_rgba display SSOT as demo_pairff.py / RigidBodyVispy.
    Only called when FAF is enabled (window.ra_fit is not None).
    """
    fit = getattr(window, 'ra_fit', None)
    if fit is None:
        return
    from spammm.surfaces.FoldedRigid import Z_SURF_TOP
    from .VispyUtils import update_faf_map_overlay
    apos, _ = _assembly_world_atoms(window)
    if apos is None or len(apos) == 0:
        return
    pad = 6.0
    extent = (float(apos[:, 0].min() - pad), float(apos[:, 0].max() + pad),
              float(apos[:, 1].min() - pad), float(apos[:, 1].max() + pad))
    z_eval = Z_SURF_TOP + float(window.ra_z_init_spin.value())
    visible = getattr(window, 'ra_show_substrate', True)
    img = update_faf_map_overlay(window.scene, fit, z_eval, extent,
                                 image_attr='ra_substrate_map', visible=visible)
    window.ra_substrate_map = img


def _ra_frozen_mask(window):
    """Bool array: True = static (frozen) body.  Derived from RBD body_state_host."""
    rbd = window.ra_rbd
    if rbd is None or rbd._body_state_host is None:
        return np.zeros(0, dtype=bool)
    return rbd._body_state_host == 0


def _ra_probe_params(window):
    """Read probe R0/E0/Q from the RA probe controls."""
    return (float(window.ra_probe_R0_spin.value()),
            float(window.ra_probe_E0_spin.value()),
            float(window.ra_probe_q_spin.value()))


def _on_probe_preset(window, which):
    """Apply H+ or O− probe preset: fill R0/E0 from AtomTypes, set Q, update toggle buttons."""
    from spammm.forcefields.QEq import get_atom_types
    _, atom_types = get_atom_types()
    if which == 'Hp':
        ename, q = 'H', 0.4
        window.ra_probe_Hp_btn.setChecked(True)
        window.ra_probe_Om_btn.setChecked(False)
    else:
        ename, q = 'O', -0.4
        window.ra_probe_Hp_btn.setChecked(False)
        window.ra_probe_Om_btn.setChecked(True)
    at = atom_types.get(ename)
    R0 = float(at.RvdW) if at else 1.5
    E0 = float(at.EvdW) if at else 0.002
    window.ra_probe_combo.blockSignals(True)
    window.ra_probe_combo.setCurrentText(ename)
    window.ra_probe_combo.blockSignals(False)
    window.ra_probe_R0_spin.setValue(R0)
    window.ra_probe_E0_spin.setValue(E0)
    window.ra_probe_q_spin.setValue(q)
    _recompute_ra_combined_map(window)


def _recompute_ra_combined_map(window):
    """Recompute the combined PairFF(static) + FAF probe map and update the overlay.

    Uses compute_combined_probe_map (shared headless helper in RigidBodyUtils).
    Caches the result in window.ra_combined_map for e-pair/sigma-hole overlays.
    """
    rbd = window.ra_rbd
    if rbd is None:
        return
    from spammm.forcefields.RigidBodyUtils import compute_combined_probe_map
    from spammm.surfaces.FoldedRigid import Z_SURF_TOP
    from spammm.GUI.RigidBodyVispy import potential_to_rgba
    import vispy.scene as vscene
    from vispy.visuals.transforms import STTransform
    fit = getattr(window, 'ra_fit', None)
    frozen_mask = _ra_frozen_mask(window)
    probe_R0, probe_E0, probe_q = _ra_probe_params(window)
    z_probe = Z_SURF_TOP + float(window.ra_probe_z_spin.value())
    beta = 1.7
    E_sum, E_pairff, E_faf, xs, ys, extent = compute_combined_probe_map(
        rbd, fit, frozen_mask, probe_R0, probe_E0, probe_q, z_probe, beta=beta)
    # Cache for e-pair/sigma-hole overlay
    window.ra_combined_map = {'E_sum': E_sum, 'E_pairff': E_pairff, 'E_faf': E_faf,
                              'xs': xs, 'ys': ys, 'extent': extent, 'z_probe': z_probe}
    rgba = potential_to_rgba(E_sum)
    visible = getattr(window, 'ra_show_map', True)
    # Hide the FAF substrate overlay when the combined map is shown (avoids double-overlay white box)
    if visible:
        sub_img = getattr(window.scene, 'ra_substrate_map', None)
        if sub_img is not None:
            sub_img.visible = False
    parent = window.scene.view.scene if hasattr(window.scene, 'view') else window.scene
    img = getattr(window.scene, 'ra_combined_map_img', None)
    if img is None:
        img = vscene.visuals.Image(parent=parent)
        img.set_gl_state('translucent', depth_test=False)
        img.order = -1  # same z-order as FAF substrate overlay
        setattr(window.scene, 'ra_combined_map_img', img)
    img.set_data(rgba)
    xmin, xmax, ymin, ymax = extent
    dx = (float(xmax) - float(xmin)) / max(len(xs) - 1, 1)
    dy = (float(ymax) - float(ymin)) / max(len(ys) - 1, 1)
    img.transform = STTransform(translate=(float(xmin), float(ymin), -0.1), scale=(dx, dy, 1))
    img.visible = visible
    window.ra_combined_map_img = img
    _status(window, f"Map recomputed: range=[{E_sum.min():.3f},{E_sum.max():.3f}]  grid={len(xs)}x{len(ys)}")


def _upload_poses_to_gpu(window):
    """Push authoritative ensemble poses into GPU and RBD host mirrors."""
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
    rbd._mb_pos = pos.astype(np.float32).copy()
    rbd._mb_quat = quat.astype(np.float32).copy()
    rbd.reset_dynamics_state()


def _sync_ensemble_from_gpu(window):
    """Commit all GPU rigid poses to the authoritative ensemble."""
    ens = window.ra_ensemble
    rbd = window.ra_rbd
    if ens is None or rbd is None:
        return
    rbd.sync_active_pose_from_gpu()
    ens.set_poses(rbd._mb_pos, rbd._mb_quat)


def _display_index_to_body_site(window, display_idx):
    """Map dense real-atom scene index to (body, flat PairFF site), skipping dummies."""
    rbd = window.ra_rbd
    if rbd is None or rbd._mb_packs is None:
        raise RuntimeError('Build assembly before mapping scene atoms')
    i = int(display_idx)
    if i < 0:
        raise IndexError(f'display atom index {i} is negative')
    real0 = 0
    for body, pack in enumerate(rbd._mb_packs):
        real_sites = np.flatnonzero(pack['types'] == 0)
        if i < real0 + len(real_sites):
            local_site = int(real_sites[i - real0])
            return body, int(rbd.mol_offsets[body] + local_site)
        real0 += len(real_sites)
    raise IndexError(f'display atom index {i} outside {real0} real assembly atoms')


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
    faf_enabled = not window.ra_no_faf_chk.isChecked()
    z_body = float(Z_SURF_TOP + window.ra_z_init_spin.value()) if faf_enabled else z_mol
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
            # Body positions = CoMs (already computed); editor builds do not use FAF.
            pos = coms.astype(np.float32).copy()
            pos[:, 2] = z_body
            quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (nmol, 1))
            tids = [f'frag{i}' for i in range(nmol)]
            bonds_list = [f[3] for f in fragments]
            _status(window, f'Found {nmol} fragments (sizes: {[len(f[1]) for f in fragments]})')
        else:
            # ── From file (existing path, now supports comma-separated species) ──
            mol_text = window.ra_mol_combo.currentText().strip()
            mol_names = [m.strip() for m in mol_text.split(',') if m.strip()]
            if not mol_names:
                _status(window, 'No molecule name(s) specified')
                return
            for mn in mol_names:
                if mn not in MOL_PATHS:
                    _status(window, f'Unknown molecule: {mn}')
                    return
            nmol = int(window.ra_nmol_spin.value())
            spacing = float(window.ra_spacing_spin.value())
            _status(window, f'Loading {mol_text} ({nmol} copies each)...')
            from spammm.forcefields.RigidBodyUtils import build_mixed_species_assembly
            molecules, tids, bonds_list, pos, quat, species_data = build_mixed_species_assembly(
                mol_names, nmol, MOL_PATHS, _NO_QEQ, spacing, z_body,
                int(window.ra_seed_spin.value()), qeq=not window.ra_no_qeq_chk.isChecked())
            # For FAF: use first species' data (single-species case) or species_data (mixed)
            apos, enames, REQs, bonds = species_data[0]
            mol_name = mol_names[0]

        # Build ensemble (pose SSOT)
        window.ra_ensemble = RigidEnsemble.from_poses(tids, pos, quat)
        # Build RigidBodyPairFF from ensemble poses
        pos, quat = window.ra_ensemble.get_poses()
        _status(window, f'Building PairFF ({nmol} mols)...')
        window.ra_rbd = RigidBodyPairFF.from_molecules(
            molecules, pos, quats=quat, active_body=0,
            He=-0.1, rc=3.0, w=0.7, k_z=0.0, z_target=z_body, Hs=1.0, beta=1.7,
        )
        # Optional FAF substrate
        if faf_enabled:
            from spammm.surfaces.FoldedRigid import load_or_fit_faf, faf_fit_mode, FAF_MODE_TYPED, FAF_MODE_FACTOR
            if source == 'From editor':
                # Editor build: fit on first fragment's atoms
                f0 = fragments[0]
                # Derive mol_name from enames for cache filename (e.g. "benzoic_acid" from CCOOH...)
                editor_mol_name = 'editor_frag0'
                fit = load_or_fit_faf((f0[0], f0[1], f0[2]), mol_name=editor_mol_name)
                mode = faf_fit_mode(fit)
                if mode == FAF_MODE_TYPED:
                    at_ids_parts = [remap_fit_for_molecule(fit, f[2])['atom_type_ids'] for f in fragments]
                    fit['atom_type_ids'] = np.concatenate(at_ids_parts).astype(np.int32)
                else:
                    n_real_total = sum(len(f[1]) for f in fragments)
                    fit['atom_type_ids'] = np.zeros(n_real_total, dtype=np.int32)
            else:
                n_total = len(molecules)
                fit = load_or_fit_faf((apos, enames, REQs), mol_name=mol_names[0])
                mode = faf_fit_mode(fit)
                if mode == FAF_MODE_TYPED:
                    at_ids_parts = []
                    for copy_idx in range(nmol):
                        for sp_idx, (sp_apos, sp_enames, sp_REQs, _) in enumerate(species_data):
                            sp_fit = remap_fit_for_molecule(fit, sp_REQs)
                            at_ids_parts.append(sp_fit['atom_type_ids'])
                    fit['atom_type_ids'] = np.concatenate(at_ids_parts).astype(np.int32)
                else:
                    n_real_total = sum(len(sd[2]) for sd in species_data) * nmol
                    fit['atom_type_ids'] = np.zeros(n_real_total, dtype=np.int32)
            window.ra_rbd.attach_pairff_faf(fit, z_init=float(window.ra_z_init_spin.value()), k_z=0.0, enable=True)
            window.ra_fit = fit
        else:
            window.ra_fit = None
        # Cache per-pack bonds for display
        window.ra_bonds0 = bonds_list
        out = window.ra_rbd.download_outputs()
        err = max(float(np.max(np.abs(out['pos'][:, :3] - pos))), float(np.max(np.abs(out['quats'] - quat))), float(np.max(np.abs(window.ra_rbd._mb_pos - pos))), float(np.max(np.abs(window.ra_rbd._mb_quat - quat))))
        if err > 1e-5:
            raise RuntimeError(f'Rigid pose synchronization failed after build: max error {err:.3e}')
        window.ra_E_last = window.ra_rbd.eval_energy_system(pos, quat, k_pack=float(window.ra_kpack_spin.value()))
        window.ra_kpack_last = float(window.ra_kpack_spin.value())
        _status(window, f'Built: {window.ra_ensemble.summary()}  device={window.ra_rbd.ctx.devices[0].name}')
        _sync_display(window)
        # Show substrate overlay when FAF is enabled (shared VispyUtils)
        if window.ra_fit is not None:
            _update_ra_substrate_overlay(window)
    except Exception as e:
        import traceback; traceback.print_exc()
        _status(window, f'Build FAILED: {e}')


# ─── MC/GA mode ──────────────────────────────────────────────────────────────
def _on_mc_step(window, update_ui=True):
    """Run one greedy MC step over all molecules (round-robin moved index).

    update_ui=True (default, used by the button and _on_mc_run): syncs the scene and
    status on acceptance, preserving existing behavior. update_ui=False skips scene
    sync and status pump so a paced script can run several steps per visual frame;
    the caller is responsible for refreshing the final pose of the batch (call with
    update_ui=True on the last point, even if that trial is rejected — _sync_display
    reads the authoritative ensemble pose, not the rejected trial).

    Returns a summary dict {step, E0, Ebest, E, accepted, batch_min} or None if not
    built.
    """
    ens = window.ra_ensemble
    rbd = window.ra_rbd
    if ens is None or rbd is None:
        if update_ui:
            _status(window, 'Build assembly first')
        return None
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
    pos, quat, E0, Ebest, acc, Ebatch = greedy_energy_step(rbd,
        pos, quat, moved, n_trial=n_trial, dxy=dxy, dphi=dphi,
        seed=seed, rmin_com=0.0, rmin_atom=float(window.ra_rmin_atom_spin.value()),
        k_pack=k_pack,
    )
    if acc:
        ens.set_poses(pos, quat)
        _upload_poses_to_gpu(window)
        window.ra_E_last += Ebest - E0
    E = window.ra_E_last
    window.ra_mc_step_count += 1
    window.ra_E_last = float(E)
    finite = Ebatch[np.isfinite(Ebatch)]
    batch_min = float(finite.min()) if len(finite) else float('nan')
    if update_ui:
        # Always sync the current authoritative pose, even if this trial rejected,
        # so an accepted pose from earlier in the batch is still displayed.
        _sync_display(window)
        _status(window, f'MC step {window.ra_mc_step_count}: E={E:.5f}  acc={int(acc)}  batch_min={batch_min:.5f}')
    return {'step': int(window.ra_mc_step_count), 'E0': float(E0), 'Ebest': float(Ebest),
            'E': float(E), 'accepted': bool(acc), 'batch_min': batch_min}


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
    """Set one spring on flat PairFF site `idx` to `target` (world)."""
    rbd = window.ra_rbd
    if rbd is None:
        return
    anchors = rbd.anchors.copy()
    anchors[:, 3] = -1.0
    if idx >= 0:
        anchors[idx, :3] = target
        anchors[idx, 3] = float(window.ra_k_spring_spin.value())
    rbd.update_anchors(anchors)


def _update_anchor_visuals(window, atom_world_pos, anchor_target):
    """Draw red line from anchored atom to anchor target + red cross marker at target.

    Reuses the same pattern as RigidBodyVispy.anchor_line/anchor_marker.
    Creates the visuals lazily on first call, cached on window.
    """
    import vispy.scene as vscene
    if not hasattr(window, 'ra_anchor_line'):
        window.ra_anchor_line = vscene.visuals.Line(
            parent=window.scene.view.scene, color=(1, 0, 0, 0.8), width=2.0,
            antialias=True, method='gl', connect='segments')
        window.ra_anchor_line.set_gl_state('translucent', depth_test=False)
        window.ra_anchor_line.order = 10
    if not hasattr(window, 'ra_anchor_marker'):
        window.ra_anchor_marker = vscene.visuals.Markers(parent=window.scene.view.scene)
        window.ra_anchor_marker.set_gl_state('translucent', depth_test=False)
        window.ra_anchor_marker.order = 11
    if atom_world_pos is None or anchor_target is None:
        window.ra_anchor_line.visible = False
        window.ra_anchor_marker.visible = False
        return
    line_pos = np.array([atom_world_pos[:3], anchor_target[:3]], dtype=np.float32)
    window.ra_anchor_line.set_data(line_pos)
    window.ra_anchor_line.visible = True
    window.ra_anchor_marker.set_data(
        pos=np.asarray(anchor_target[:3], dtype=np.float32).reshape(1, 3),
        face_color=(1, 0, 0, 1), size=12, edge_width=0, symbol='cross')
    window.ra_anchor_marker.visible = True


def _closest_point_on_ray(atom_pos, r0, rd):
    rd = np.asarray(rd, dtype=np.float64)
    rd2 = float(np.dot(rd, rd))
    if rd2 < 1e-12:
        return np.asarray(r0, dtype=np.float64)
    t = float(np.dot(atom_pos - r0, rd) / rd2)
    return r0 + t * rd


def _make_ramdrag_mode(window):
    class RAManipMode(EditModeHandler):
        status_msg = "RA Drag: LMB drag dynamic mol; Shift+LMB toggle static; RMB delete"
        lock_drag = True
        capture_move = True

        def on_activate(self):
            if getattr(window, 'ra_rbd', None) is None:
                _status(window, "RA Drag: press Build first")
            self._pin_display = -1
            self._pin_site = -1
            self._pin_body = -1

        def _pick_idx(self, event):
            atom_id, dist = window.scene._pick_id_from_mouse(event.pos, max_dist=window.pick_radius)
            if atom_id < 0:
                return -1
            return window.scene._id_to_idx_safe(atom_id)

        def on_press(self, event, p_world, ctrl):
            if event.button != 1:
                return
            shift = getattr(window.scene, '_last_shift', False)
            idx = self._pick_idx(event)
            if idx < 0:
                return
            body, site = _display_index_to_body_site(window, idx)
            # Shift+LMB: toggle dynamic ↔ static (per spec §1.2)
            if shift:
                _toggle_body_state(window, body)
                return
            # Plain LMB on static body: do not attach anchor
            rbd = window.ra_rbd
            if rbd._body_state_host is not None and int(rbd._body_state_host[body]) != 1:
                _status(window, f"Body {body} is static — Shift+LMB to toggle, RMB to delete")
                return
            # Plain LMB on dynamic body: existing drag behavior
            window.ra_rbd.set_active_body(body)
            window.ra_rbd.reset_dynamics_state()
            self._pin_display = idx
            self._pin_site = site
            self._pin_body = body
            atom_pos = window.scene._pos[idx].astype(np.float64)
            r0, rd = window.scene._ray_from_mouse(event.pos)
            target = _closest_point_on_ray(atom_pos, r0, rd).astype(np.float32)
            _set_anchors(window, site, target)
            _update_anchor_visuals(window, atom_pos, target)
            _status(window, f"Pinned atom {idx} on molecule {body}; dragging")

        def on_rmb_atom(self, atom_id, ctrl):
            """RMB on atom → soft-delete the body (per spec §1.2)."""
            idx = window.scene._id_to_idx_safe(atom_id)
            if idx < 0:
                return
            body, _ = _display_index_to_body_site(window, idx)
            _soft_delete_body(window, body)

        def on_move(self, p_world, r0=None, rd=None):
            if self._pin_display < 0:
                return
            rbd = window.ra_rbd
            if rbd is None:
                return
            atom_pos = window.scene._pos[self._pin_display].astype(np.float64)
            if r0 is not None and rd is not None:
                target = _closest_point_on_ray(atom_pos, r0, rd)
            else:
                target = np.array([p_world[0], p_world[1], atom_pos[2]], dtype=np.float64)
            _set_anchors(window, self._pin_site, target.astype(np.float32))
            _update_anchor_visuals(window, atom_pos, target)
            # Concurrent kernel: the dragged molecule pulls while all partners and
            # their charged O sites respond to PairFF + FAF substrate forces.
            n_relax = int(window.ra_drag_nrelax_spin.value())
            dt = float(window.ra_drag_dt_spin.value())
            if n_relax > 0:
                rbd.run_multimol_md(n_relax, dt, fire=True, faf=None)
                _sync_ensemble_from_gpu(window)
                _sync_display(window)
                # After relaxation, update line endpoint to the relaxed atom position
                atom_pos = window.scene._pos[self._pin_display].astype(np.float64)
                _update_anchor_visuals(window, atom_pos, target)

        def on_release(self, event, p_world, ctrl):
            if event.button != 1:
                return
            if self._pin_display < 0:
                return
            _set_anchors(window, -1, np.zeros(3, dtype=np.float32))
            _update_anchor_visuals(window, None, None)  # hide anchor visuals
            window.ra_rbd.reset_dynamics_state()
            self._pin_display = -1
            self._pin_site = -1
            self._pin_body = -1
            window.scene._pick_active = False
            _status(window, "Drag released; all rigid poses written to ensemble")
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
        'nx': int(window.ra_pme_nx_spin.value()),
        'nV': int(window.ra_pme_nV_spin.value()),
        'Vmin': float(window.ra_pme_Vmin_spin.value()),
        'Vmax': float(window.ra_pme_Vmax_spin.value()),
        'p1_x': float(window.ra_pme_p1x_spin.value()),
        'p1_y': float(window.ra_pme_p1y_spin.value()),
        'p2_x': float(window.ra_pme_p2x_spin.value()),
        'p2_y': float(window.ra_pme_p2y_spin.value()),
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
        _show_pme_xy(window, xy, spos4, params)
    except Exception as e:
        import traceback; traceback.print_exc()
        _status(window, f'PME XY FAILED: {e}')


def _show_pme_xy(window, xy, spos, params=None):
    """Plot PME XY result: STM + dI/dV at fixed VBias, with xV cut line overlay."""
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    VBias = float(params['VBias']) if params else float(xy.get('VBias', 0.0))
    p1 = (float(params['p1_x']), float(params['p1_y'])) if params else None
    p2 = (float(params['p2_x']), float(params['p2_y'])) if params else None
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs[0].imshow(xy['STM'], extent=xy['extent'], origin='lower', cmap='inferno')
    axs[0].plot(spos[:xy['n_active'], 0], spos[:xy['n_active'], 1], 'c+', ms=10, mew=1.5)
    if p1 is not None and p2 is not None:
        axs[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'w-', lw=1.5, label='xV cut')
        axs[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'wo', ms=4)
        axs[0].legend(loc='upper right', fontsize=8)
    axs[0].set_title(f"PME XY STM  V={VBias:.2f}V  n_active={xy['n_active']}")
    axs[0].set_xlabel('x [Å]'); axs[0].set_ylabel('y [Å]')
    fig.colorbar(im0, ax=axs[0], fraction=0.046)
    if xy.get('dIdV') is not None:
        sc = max(np.nanmax(np.abs(xy['dIdV'])), 1e-30)
        im1 = axs[1].imshow(xy['dIdV'], extent=xy['extent'], origin='lower', cmap='bwr', vmin=-sc, vmax=sc)
        axs[1].plot(spos[:xy['n_active'], 0], spos[:xy['n_active'], 1], 'k+', ms=10, mew=1.5)
        if p1 is not None and p2 is not None:
            axs[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'w-', lw=1.5)
            axs[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'wo', ms=4)
        axs[1].set_title(f'dI/dV  V={VBias:.2f}V  (rings / NDR)')
        axs[1].set_xlabel('x [Å]'); axs[1].set_ylabel('y [Å]')
        fig.colorbar(im1, ax=axs[1], fraction=0.046)
    fig.tight_layout()
    fig.show()
    window._ra_plot_windows = getattr(window, '_ra_plot_windows', [])
    window._ra_plot_windows.append(fig)


def _on_pme_scan_xv(window):
    """Run PME xV line×voltage scan over the assembly sites (NDR / charge rings)."""
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
        nx = int(params.get('nx', 100))
        nV = int(params.get('nV', 80))
        Vmin = float(params.get('Vmin', 0.0))
        Vmax = float(params.get('Vmax', params['VBias']))
        _status(window, f'PME xV scan: {n_act} sites V=[{Vmin:.2f},{Vmax:.2f}]...')
        QtWidgets.QApplication.processEvents()
        xv = ps.scan_xV(solver, spos4, rots4, params, nx=nx, nV=nV, Vmin=Vmin, Vmax=Vmax, return_probs=True)
        window.ra_pme_xv = xv
        ndr_min = float(xv['dIdV'].min())
        _status(window, f'PME xV done: Imax={xv["STM"].max():.3e} NDRmin={ndr_min:.2e}')
        _show_pme_xv(window, xv, spos4, params)
    except Exception as e:
        import traceback; traceback.print_exc()
        _status(window, f'PME xV FAILED: {e}')


def _show_pme_xv(window, xv, spos, params=None):
    """Plot PME xV result: STM(x,V) + dI/dV(x,V) with NDR + VBias horizontal line."""
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    V = xv['Vbiases']
    x = xv['dist_axis']
    Vmin, Vmax = float(V[0]), float(V[-1])
    VBias = float(params['VBias']) if params else None
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axs[0].imshow(xv['STM'], aspect='auto', origin='lower', extent=[x[0], x[-1], Vmin, Vmax], cmap='inferno')
    if VBias is not None and Vmin <= VBias <= Vmax:
        axs[0].axhline(VBias, color='cyan', lw=1.5, ls='--', label=f'XY @ V={VBias:.2f}')
        axs[0].legend(loc='upper right', fontsize=8)
    axs[0].set_title(f"PME xV STM  n_active={xv['n_active']}")
    axs[0].set_xlabel('distance along cut [Å]'); axs[0].set_ylabel('V [V]')
    fig.colorbar(im0, ax=axs[0], fraction=0.046)
    sc = max(np.nanmax(np.abs(xv['dIdV'])), 1e-30)
    im1 = axs[1].imshow(xv['dIdV'], aspect='auto', origin='lower', extent=[x[0], x[-1], Vmin, Vmax], cmap='bwr', vmin=-sc, vmax=sc)
    if VBias is not None and Vmin <= VBias <= Vmax:
        axs[1].axhline(VBias, color='cyan', lw=1.5, ls='--')
    ndr_min = float(xv['dIdV'].min())
    axs[1].set_title(f'dI/dV  NDR={ndr_min<0}  min={ndr_min:.2e}')
    axs[1].set_xlabel('distance along cut [Å]'); axs[1].set_ylabel('V [V]')
    fig.colorbar(im1, ax=axs[1], fraction=0.046)
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
    g = AutoGridPlacer(cols=4)
    window.ra_source_combo = QtWidgets.QComboBox()
    window.ra_source_combo.addItems(['From file', 'From editor'])
    window.ra_source_combo.setToolTip('From file: load nmol copies of a pre-defined molecule.\n'
                                       'From editor: split the current AtomicGraph into connected '
                                       'components (independent fragments); each fragment → one rigid body.')
    g.add_pair("Source:", window.ra_source_combo)
    g.newrow()
    window.ra_mol_combo = QtWidgets.QComboBox()
    window.ra_mol_combo.setEditable(True)
    window.ra_mol_combo.addItems(sorted(MOL_PATHS.keys()))
    g.add_pair("Mol:", window.ra_mol_combo)
    window.ra_nmol_spin = QtWidgets.QSpinBox(); window.ra_nmol_spin.setRange(1, 64); window.ra_nmol_spin.setValue(4); window.ra_nmol_spin.setMaximumWidth(50)
    g.add_pair("nmol:", window.ra_nmol_spin)
    g.newrow()
    window.ra_spacing_spin = QtWidgets.QDoubleSpinBox(); window.ra_spacing_spin.setRange(2.0, 50.0); window.ra_spacing_spin.setValue(16.0); window.ra_spacing_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g.add_pair("spacing:", window.ra_spacing_spin)
    window.ra_z_spin = QtWidgets.QDoubleSpinBox(); window.ra_z_spin.setRange(-5.0, 20.0); window.ra_z_spin.setSingleStep(0.1); window.ra_z_spin.setValue(3.0); window.ra_z_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g.add_pair("z_mol:", window.ra_z_spin)
    g.newrow()
    window.ra_z_init_spin = QtWidgets.QDoubleSpinBox(); window.ra_z_init_spin.setRange(0.0, 10.0); window.ra_z_init_spin.setSingleStep(0.1); window.ra_z_init_spin.setValue(3.0); window.ra_z_init_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g.add_pair("z_init:", window.ra_z_init_spin)
    window.ra_seed_spin = QtWidgets.QSpinBox(); window.ra_seed_spin.setRange(0, 100000); window.ra_seed_spin.setValue(3); window.ra_seed_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g.add_pair("seed:", window.ra_seed_spin)
    build_l.addLayout(g.layout())

    # Checkboxes + Build button
    g_chk = AutoGridPlacer(cols=4)
    window.ra_no_qeq_chk = QtWidgets.QCheckBox('no QEq')
    g_chk.add(window.ra_no_qeq_chk)
    window.ra_no_faf_chk = QtWidgets.QCheckBox('no FAF')
    g_chk.add(window.ra_no_faf_chk)
    window.ra_build_btn = QtWidgets.QPushButton('Build')
    window.ra_build_btn.clicked.connect(lambda: _on_build(window))
    g_chk.add(window.ra_build_btn)
    build_l.addLayout(g_chk.layout())
    build_sec.setContent(build_host)
    layout.addWidget(build_sec)

    # ─── MC/GA section ────────────────────────────────────────────────────
    mc_sec = CollapsibleSection("MC / GA Optimization", collapsed=False)
    mc_host = QtWidgets.QWidget()
    mc_l = QtWidgets.QVBoxLayout(mc_host)
    mc_l.setSpacing(SPACING)

    g_mc = AutoGridPlacer(cols=4)
    window.ra_ntrial_spin = QtWidgets.QSpinBox(); window.ra_ntrial_spin.setRange(1, 4096); window.ra_ntrial_spin.setValue(128); window.ra_ntrial_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_mc.add_pair("n_trial:", window.ra_ntrial_spin)
    window.ra_dxy_spin = QtWidgets.QDoubleSpinBox(); window.ra_dxy_spin.setRange(0.01, 10.0); window.ra_dxy_spin.setSingleStep(0.1); window.ra_dxy_spin.setValue(1.5); window.ra_dxy_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_mc.add_pair("dxy:", window.ra_dxy_spin)
    g_mc.newrow()
    window.ra_dphi_spin = QtWidgets.QDoubleSpinBox(); window.ra_dphi_spin.setRange(0.01, 3.0); window.ra_dphi_spin.setSingleStep(0.05); window.ra_dphi_spin.setValue(0.8); window.ra_dphi_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_mc.add_pair("dphi:", window.ra_dphi_spin)
    window.ra_kpack_spin = QtWidgets.QDoubleSpinBox(); window.ra_kpack_spin.setRange(0.0, 1.0); window.ra_kpack_spin.setSingleStep(0.01); window.ra_kpack_spin.setValue(0.03); window.ra_kpack_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_mc.add_pair("k_pack:", window.ra_kpack_spin)
    g_mc.newrow()
    window.ra_rmin_atom_spin = QtWidgets.QDoubleSpinBox(); window.ra_rmin_atom_spin.setRange(0.0, 5.0); window.ra_rmin_atom_spin.setSingleStep(0.1); window.ra_rmin_atom_spin.setValue(1.6); window.ra_rmin_atom_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_mc.add_pair("rmin:", window.ra_rmin_atom_spin)
    window.ra_mc_nsteps_spin = QtWidgets.QSpinBox(); window.ra_mc_nsteps_spin.setRange(1, 100000); window.ra_mc_nsteps_spin.setValue(50); window.ra_mc_nsteps_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_mc.add_pair("n_steps:", window.ra_mc_nsteps_spin)
    mc_l.addLayout(g_mc.layout())

    g_mc_btn = AutoGridPlacer(cols=4)
    window.ra_mc_step_btn = QtWidgets.QPushButton('Step')
    window.ra_mc_step_btn.clicked.connect(lambda: _on_mc_step(window))
    g_mc_btn.add(window.ra_mc_step_btn)
    window.ra_mc_run_btn = QtWidgets.QPushButton('Run')
    window.ra_mc_run_btn.clicked.connect(lambda: _on_mc_run(window))
    g_mc_btn.add(window.ra_mc_run_btn)
    window.ra_mc_reset_btn = QtWidgets.QPushButton('Reset')
    window.ra_mc_reset_btn.clicked.connect(lambda: _on_mc_reset(window))
    g_mc_btn.add(window.ra_mc_reset_btn)
    mc_l.addLayout(g_mc_btn.layout())
    mc_sec.setContent(mc_host)
    layout.addWidget(mc_sec)

    # ─── Drag section ─────────────────────────────────────────────────────
    drag_sec = CollapsibleSection("Drag (anchor spring)", collapsed=True)
    drag_host = QtWidgets.QWidget()
    drag_l = QtWidgets.QVBoxLayout(drag_host)
    drag_l.setSpacing(SPACING)
    g_drag = AutoGridPlacer(cols=4)
    window.ra_k_spring_spin = QtWidgets.QDoubleSpinBox(); window.ra_k_spring_spin.setRange(0.01, 1000.0); window.ra_k_spring_spin.setSingleStep(0.5); window.ra_k_spring_spin.setValue(20.0); window.ra_k_spring_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_drag.add_pair("k_spring:", window.ra_k_spring_spin)
    window.ra_drag_nrelax_spin = QtWidgets.QSpinBox(); window.ra_drag_nrelax_spin.setRange(0, 500); window.ra_drag_nrelax_spin.setValue(20); window.ra_drag_nrelax_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_drag.add_pair("n_relax:", window.ra_drag_nrelax_spin)
    g_drag.newrow()
    window.ra_drag_dt_spin = QtWidgets.QDoubleSpinBox(); window.ra_drag_dt_spin.setRange(0.0001, 0.5); window.ra_drag_dt_spin.setSingleStep(0.005); window.ra_drag_dt_spin.setValue(0.02); window.ra_drag_dt_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_drag.add_pair("dt:", window.ra_drag_dt_spin)
    drag_l.addLayout(g_drag.layout())
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

    def _pme_spin(key, lo, hi, step, val, decimals=3, width=70):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(decimals); s.setValue(val); s.setMaximumWidth(width)
        setattr(window, f'ra_pme_{key}_spin', s)
        return s

    g_pme = AutoGridPlacer(cols=4)
    g_pme.add_pair("Esite:", _pme_spin('esite', -1.0, 1.0, 0.01, -0.09))
    g_pme.add_pair("W:", _pme_spin('W', 0.0, 0.5, 0.01, 0.05))
    g_pme.newrow()
    g_pme.add_pair("Q0:", _pme_spin('Q0', 0.0, 5.0, 0.1, 1.0, decimals=2))
    g_pme.add_pair("Qzz:", _pme_spin('Qzz', -20.0, 20.0, 0.5, 0.0, decimals=2))
    g_pme.newrow()
    g_pme.add_pair("VBias:", _pme_spin('vbias', 0.0, 3.0, 0.05, 1.2))
    g_pme.add_pair("z_tip:", _pme_spin('ztip', 1.0, 15.0, 0.5, 6.0, decimals=2))
    g_pme.newrow()
    g_pme.add_pair("Temp:", _pme_spin('temp', 0.1, 100.0, 0.5, 2.6, decimals=2))
    g_pme.add_pair("GammaT:", _pme_spin('gammat', 1e-4, 1.0, 0.01, 0.01, decimals=4))
    g_pme.newrow()
    g_pme.add_pair("decay:", _pme_spin('decay', 0.05, 2.0, 0.05, 0.3))
    g_pme.add_pair("L:", _pme_spin('L', 5.0, 40.0, 1.0, 20.0, decimals=1))
    g_pme.newrow()
    npix_s = QtWidgets.QSpinBox(); npix_s.setRange(20, 200); npix_s.setValue(100); npix_s.setMaximumWidth(SPIN_MAX_WIDTH)
    window.ra_pme_npix_spin = npix_s
    g_pme.add_pair("npix:", npix_s)
    g_pme.add_pair("zV0:", _pme_spin('zV0', -5.0, 5.0, 0.1, -0.9, decimals=2))
    g_pme.newrow()
    g_pme.add_pair("radius:", _pme_spin('radius', 1.0, 20.0, 0.1, 5.0, decimals=2))
    g_pme.add_pair("phiRot:", _pme_spin('phirot', -6.3, 6.3, 0.1, 0.0))
    # xV scan params
    g_pme.newrow()
    nx_s = QtWidgets.QSpinBox(); nx_s.setRange(20, 200); nx_s.setValue(100); nx_s.setMaximumWidth(SPIN_MAX_WIDTH)
    window.ra_pme_nx_spin = nx_s
    g_pme.add_pair("nx:", nx_s)
    nV_s = QtWidgets.QSpinBox(); nV_s.setRange(20, 200); nV_s.setValue(80); nV_s.setMaximumWidth(SPIN_MAX_WIDTH)
    window.ra_pme_nV_spin = nV_s
    g_pme.add_pair("nV:", nV_s)
    g_pme.newrow()
    g_pme.add_pair("Vmin:", _pme_spin('Vmin', 0.0, 2.0, 0.05, 0.5, decimals=3))
    g_pme.add_pair("Vmax:", _pme_spin('Vmax', 0.1, 3.0, 0.05, 1.5, decimals=3))
    g_pme.newrow()
    g_pme.add_pair("p1_x:", _pme_spin('p1x', -40.0, 40.0, 0.5, -15.0, decimals=2))
    g_pme.add_pair("p1_y:", _pme_spin('p1y', -40.0, 40.0, 0.5, 0.0, decimals=2))
    g_pme.newrow()
    g_pme.add_pair("p2_x:", _pme_spin('p2x', -40.0, 40.0, 0.5, 15.0, decimals=2))
    g_pme.add_pair("p2_y:", _pme_spin('p2y', -40.0, 40.0, 0.5, 0.0, decimals=2))
    pme_l.addLayout(g_pme.layout())

    g_pme_btn = AutoGridPlacer(cols=4)
    window.ra_pme_xy_btn = QtWidgets.QPushButton('Scan XY')
    window.ra_pme_xy_btn.clicked.connect(lambda: _on_pme_scan_xy(window))
    g_pme_btn.add(window.ra_pme_xy_btn)
    window.ra_pme_xv_btn = QtWidgets.QPushButton('Scan xV')
    window.ra_pme_xv_btn.clicked.connect(lambda: _on_pme_scan_xv(window))
    g_pme_btn.add(window.ra_pme_xv_btn)
    pme_l.addLayout(g_pme_btn.layout())
    pme_hint = QtWidgets.QLabel('Sites = rigid-molecule CoMs from the ensemble, oriented by R(q). '
                                 'PME n_sites ≤ 4; first min(n_bodies,4) bodies used.')
    pme_hint.setWordWrap(True)
    pme_l.addWidget(pme_hint)
    pme_sec.setContent(pme_host)
    layout.addWidget(pme_sec)

    # ─── Probe map controls ──────────────────────────────────────────────
    map_sec = CollapsibleSection("Probe Map", collapsed=False)
    map_host = QtWidgets.QWidget(); map_l = QtWidgets.QVBoxLayout(map_host)
    map_l.setContentsMargins(2, 2, 2, 2); map_l.setSpacing(2)
    g_map = AutoGridPlacer(cols=4)
    # Probe element combo + H+/O− presets
    window.ra_probe_combo = QtWidgets.QComboBox()
    window.ra_probe_combo.addItems(['H', 'O'])
    window.ra_probe_combo.setMaximumWidth(COMBO_MAX_WIDTH)
    g_map.add_pair("Probe:", window.ra_probe_combo)
    window.ra_probe_Hp_btn = QtWidgets.QPushButton('H+')
    window.ra_probe_Hp_btn.setCheckable(True); window.ra_probe_Hp_btn.setMaximumWidth(40)
    window.ra_probe_Hp_btn.clicked.connect(lambda: _on_probe_preset(window, 'Hp'))
    g_map.add(window.ra_probe_Hp_btn)
    window.ra_probe_Om_btn = QtWidgets.QPushButton('O−')
    window.ra_probe_Om_btn.setCheckable(True); window.ra_probe_Om_btn.setMaximumWidth(40)
    window.ra_probe_Om_btn.clicked.connect(lambda: _on_probe_preset(window, 'Om'))
    g_map.add(window.ra_probe_Om_btn)
    g_map.newrow()
    window.ra_probe_R0_spin = QtWidgets.QDoubleSpinBox(); window.ra_probe_R0_spin.setRange(0.0, 5.0); window.ra_probe_R0_spin.setSingleStep(0.01); window.ra_probe_R0_spin.setDecimals(3); window.ra_probe_R0_spin.setValue(1.443); window.ra_probe_R0_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_map.add_pair("R0:", window.ra_probe_R0_spin)
    window.ra_probe_E0_spin = QtWidgets.QDoubleSpinBox(); window.ra_probe_E0_spin.setRange(0.0, 1.0); window.ra_probe_E0_spin.setSingleStep(1e-4); window.ra_probe_E0_spin.setDecimals(5); window.ra_probe_E0_spin.setValue(0.00191); window.ra_probe_E0_spin.setMaximumWidth(56)
    g_map.add_pair("E0:", window.ra_probe_E0_spin)
    g_map.newrow()
    window.ra_probe_q_spin = QtWidgets.QDoubleSpinBox(); window.ra_probe_q_spin.setRange(-5.0, 5.0); window.ra_probe_q_spin.setSingleStep(0.05); window.ra_probe_q_spin.setDecimals(2); window.ra_probe_q_spin.setValue(0.40); window.ra_probe_q_spin.setMaximumWidth(42)
    g_map.add_pair("Q:", window.ra_probe_q_spin)
    window.ra_probe_z_spin = QtWidgets.QDoubleSpinBox(); window.ra_probe_z_spin.setRange(0.0, 10.0); window.ra_probe_z_spin.setSingleStep(0.1); window.ra_probe_z_spin.setDecimals(2); window.ra_probe_z_spin.setValue(3.0); window.ra_probe_z_spin.setMaximumWidth(SPIN_MAX_WIDTH)
    g_map.add_pair("z:", window.ra_probe_z_spin)
    g_map.newrow()
    window.ra_show_map_chk = QtWidgets.QCheckBox("Show map")
    window.ra_show_map_chk.setChecked(True)
    g_map.add(window.ra_show_map_chk)
    window.ra_recompute_map_btn = QtWidgets.QPushButton('Recompute')
    window.ra_recompute_map_btn.clicked.connect(lambda: _recompute_ra_combined_map(window))
    g_map.add(window.ra_recompute_map_btn)
    map_l.addLayout(g_map.layout())
    map_sec.setContent(map_host)
    layout.addWidget(map_sec)

    # ─── Status ───────────────────────────────────────────────────────────
    window.ra_status_label = QtWidgets.QLabel('Ready')
    layout.addWidget(window.ra_status_label)
    window.ra_state_counts_label = QtWidgets.QLabel('dyn=0 stat=0 del=0')
    layout.addWidget(window.ra_state_counts_label)
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
