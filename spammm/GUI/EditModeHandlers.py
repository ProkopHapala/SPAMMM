"""Edit mode handler classes for SPAMMM GUI.

Each edit mode is a subclass of EditModeHandler. The GUI controller
(KekuleExplorerWindow) holds a registry of mode name -> handler instance
and dispatches mouse/signal events to the active handler.

This separates interaction logic from UI layout concerns (SoC).
"""

import numpy as np

VERBOSITY_LEVEL = 2

def debug_print(level, message):
    if VERBOSITY_LEVEL >= level:
        print(message)


class EditModeHandler:
    """Base class for all edit modes. Subclasses override hooks as needed.

    Class attributes (override in subclasses):
        status_msg: str shown in status bar when mode activates
        lock_drag: if True, AtomScene suppresses atom dragging
        link_mode: if True, Ctrl+LMB drag creates bonds
        selection_mode: if True, RMB drag selects atoms
        ring_size_visible: if True, ring size spinbox is shown
    """
    status_msg = ""
    lock_drag = False
    link_mode = False
    selection_mode = False
    ring_size_visible = False

    def __init__(self, gui):
        self.gui = gui

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def backend(self): return self.gui.backend
    @property
    def scene(self): return self.gui.scene
    @property
    def pick_radius(self): return self.gui.pick_radius
    @property
    def cur_atom_type(self): return self.gui.cur_atom_type

    @property
    def b2Dview(self): return bool(getattr(self.gui, 'b2Dview', True))

    def _block_planar(self):
        """True if hex/empty should be blocked (3D mode)."""
        if self.b2Dview:
            return False
        self.gui.statusBar().showMessage("2D-only (hex/empty) — press Enter for 2D view")
        return True

    def _mouse_xy_plane(self, p_world, r0=None, rd=None):
        """Mouse hit on construction plane z=0 (XY). Used for ring side / planar ops.

        In 2D, p_world is already on z=0. In 3D, prefer ray∩z=0 so side-of-bond
        is measured in the molecular plane even when the camera is tilted.
        """
        if r0 is not None and rd is not None:
            hit = self.scene._intersect_ray_plane(r0, rd, np.zeros(3), np.array([0.0, 0.0, 1.0]))
            if hit is not None:
                return hit
        p = np.asarray(p_world, dtype=np.float64).copy()
        p[2] = 0.0
        return p

    @staticmethod
    def _dist_point_to_ray(point, r0, rd):
        dp = np.asarray(point, dtype=np.float64) - np.asarray(r0, dtype=np.float64)
        rd = np.asarray(rd, dtype=np.float64)
        q = dp - rd * np.dot(dp, rd)
        return float(np.linalg.norm(q))

    def _pick_atom_ray(self, r0, rd, radius=None):
        atom, _ = self._pick_atom_ray_dist(r0, rd, radius=radius)
        return atom

    def _pick_bond_ray(self, r0, rd, radius=None):
        bond, _ = self._pick_bond_ray_dist(r0, rd, radius=radius)
        return bond

    def _pick_atom_ray_dist(self, r0, rd, radius=None):
        radius = self.pick_radius if radius is None else radius
        best, best_d = None, float(radius)
        for atom in self.backend.graph.atoms.values():
            if not atom.alive:
                continue
            d = self._dist_point_to_ray(atom.pos, r0, rd)
            if d < best_d:
                best, best_d = atom, d
        return best, best_d if best is not None else float('inf')

    def _pick_bond_ray_dist(self, r0, rd, radius=None):
        radius = self.pick_radius if radius is None else radius
        best, best_d = None, float(radius)
        for bond in self.backend.graph.bonds.values():
            if not bond.alive or not bond.a.alive or not bond.b.alive:
                continue
            center = (bond.a.pos + bond.b.pos) * 0.5
            d = self._dist_point_to_ray(center, r0, rd)
            if d < best_d:
                best, best_d = bond, d
        return best, best_d if best is not None else float('inf')

    def _pick_ring_cog_ray_dist(self, r0, rd, radius=None):
        """Closest ring COG to mouse ray (for Ring-mode interior hover)."""
        radius = self.pick_radius if radius is None else radius
        best, best_d = None, float(radius)
        for ring in self.backend.graph.rings.values():
            if not ring.alive:
                continue
            # Allow a slightly larger hit for COG than a single atom
            lim = max(float(radius), 0.35 * float(getattr(ring, 'radius', radius) or radius))
            d = self._dist_point_to_ray(ring.cog, r0, rd)
            if d < lim and d < best_d:
                best, best_d = ring, d
        return best, best_d if best is not None else float('inf')

    def _pick_ring_target_3d(self, r0, rd):
        """Nearest of atom / bond / ring-COG along ray. Returns (kind, obj) or (None, None).

        Bond-first ordering made corner atoms unreachable (bonds sit within pick_radius
        of every vertex). Closest-wins keeps bonds sensitive without killing atoms/COG.
        """
        atom, da = self._pick_atom_ray_dist(r0, rd)
        bond, db = self._pick_bond_ray_dist(r0, rd)
        ring, dr = self._pick_ring_cog_ray_dist(r0, rd)
        best_kind, best_obj, best_d = None, None, float('inf')
        for kind, obj, d in (('atom', atom, da), ('bond', bond, db), ('ring', ring, dr)):
            if obj is not None and d < best_d:
                best_kind, best_obj, best_d = kind, obj, d
        return best_kind, best_obj

    # ── Lifecycle hook ──────────────────────────────────────────────────────

    def on_activate(self):
        """Called when this mode becomes active. Override for mode-specific setup."""
        pass

    # ── Event hooks (default: no-op) ────────────────────────────────────────

    def on_press(self, event, p_world, ctrl): pass
    def on_move(self, p_world, r0=None, rd=None): pass
    def on_release(self, event, p_world, ctrl): pass
    def on_rmb_atom(self, atom_id, ctrl): pass
    def on_link(self, from_id, to_id): pass
    def on_atom_click(self, atom_id, shift=False): pass

    # ── Shared helpers ──────────────────────────────────────────────────────

    def _push_undo(self): self.gui._push_undo()
    def _refresh(self):
        self.gui.refresh_view()
        self.gui.sig_geometry_changed.emit()

    def _add_free_atom(self, p_world):
        self.gui._add_free_atom(p_world)

    def _toggle_h_at(self, p_world):
        nearest_idx = self.gui.find_nearest_atom_index(p_world, self.pick_radius)
        if nearest_idx is not None:
            self._push_undo()
            self.backend.toggle_h_state(nearest_idx)
            self._refresh()

    def _remove_atom(self, atom_id, ctrl):
        debug_print(2, f"[RMB_REMOVE] atom_id={atom_id} ctrl={ctrl}")
        self._push_undo()
        if ctrl:
            self.backend.remove_atom_with_bridge(atom_id)
        else:
            self.backend.remove_atom_by_id(atom_id)
        self._refresh()

    def _create_bond(self, from_id, to_id):
        debug_print(2, f"[LINK_BOND] from={from_id} to={to_id}")
        self._push_undo()
        a = self.backend.graph.atoms.get(from_id)
        b = self.backend.graph.atoms.get(to_id)
        if a is not None and b is not None and a.alive and b.alive:
            self.backend.graph.add_bond(a, b)
            if self.backend.auto_h_cap:
                self.backend.adjust_h()
            self.backend._sync_sys()
        self._refresh()

    def _hover_atom(self, atom):
        self.scene.hover_atom_marker.set_data(
            pos=np.array([atom.pos], dtype=np.float32),
            symbol='disc', edge_width=3, edge_color='yellow', face_color='transparent', size=20)

    def _hover_bond(self, bond):
        self.scene.hover_bond_line.set_data(pos=np.array([bond.a.pos, bond.b.pos], dtype=np.float32))


# ── Unified mode ────────────────────────────────────────────────────────────

class UnifiedMode(EditModeHandler):
    status_msg = "Unified: LMB atom=change type/drag | Shift+LMB=cycle npi | bond=cycle order | hex=add ring | empty=add atom | RMB=delete | Ctrl+bond"
    link_mode = True

    _BOND_ORDER_NAMES = {1.0: 'single', 1.5: 'aromatic', 2.0: 'double', 3.0: 'triple'}
    _ATOM_CYCLE = {'C': 'N', 'N': 'O', 'O': 'C'}

    def resolve_target(self, p_world, r0=None, rd=None):
        """Pick highest-priority target under cursor. Returns (type_str, target)."""
        if not self.b2Dview and r0 is not None and rd is not None:
            atom = self._pick_atom_ray(r0, rd)
            if atom: return ('atom', atom)
            bond = self._pick_bond_ray(r0, rd)
            if bond: return ('bond', bond)
            return ('empty', p_world)
        atom = self.backend.pick_atom(p_world, radius=self.pick_radius)
        if atom: return ('atom', atom)
        bond = self.backend.pick_bond(p_world, radius=self.pick_radius)
        if bond: return ('bond', bond)
        if hasattr(self.backend, 'snap_to_ring'):
            q, r = self.backend.snap_to_ring(p_world[0], p_world[1])
            ring_nodes = self.backend.grid.ring_nodes(q, r)
            from spammm.topology.HexGrid import snap_to_grid
            center = np.mean([snap_to_grid(n)[:2] for n in ring_nodes], axis=0)
            dist = np.linalg.norm(center - p_world[:2])
            if dist < self.backend.a_CC * 0.5: return ('hex', (q, r))
        return ('empty', p_world)

    def on_press(self, event, p_world, ctrl):
        r0 = getattr(self.gui, '_last_r0', None)
        rd = getattr(self.gui, '_last_rd', None)
        target_type, target = self.resolve_target(p_world, r0=r0, rd=rd)
        debug_print(2, f"[UNIFIED_PRESS] target={target_type} ctrl={ctrl} pos=({p_world[0]:.2f},{p_world[1]:.2f})")
        if event.button == 1:  # LMB
            # Atom clicks are handled by scene → sig_atom_clicked → on_atom_click
            # Ctrl+LMB on atom is handled by scene → link mode → sig_link_bond
            if target_type == 'atom':
                return  # Scene handles atom clicks; nothing to do here
            elif target_type == 'bond':
                self._push_undo()
                if ctrl:
                    new_atom = self.backend.insert_atom_into_bond(target, self.cur_atom_type, push_aside=True)
                    debug_print(2, f"Inserted atom into bond {target._id}, new atom {new_atom._id}")
                else:
                    self.backend.cycle_bond_order(target)
                    debug_print(2, f"Cycled bond order on bond {target._id} to {target.order}")
                self._refresh()
            elif target_type == 'hex':
                if self._block_planar(): return
                q, r = target
                self._push_undo()
                self.backend.add_ring(q, r)
                self._refresh()
            else:  # empty
                if self._block_planar(): return
                self._add_free_atom(p_world)
        elif event.button == 2:  # RMB
            if target_type == 'atom':
                return  # Let sig_rmb_remove handle it
            elif target_type == 'bond':
                self._push_undo()
                if ctrl:
                    survivor = self.backend.collapse_bond(target, np.array([p_world[0], p_world[1]]))
                    debug_print(2, f"Collapsed bond {target._id}, survivor {survivor._id}")
                else:
                    self.backend.delete_bond(target)
                    debug_print(2, f"Deleted bond {target._id}")
                self._refresh()
            elif target_type == 'hex':
                if self._block_planar(): return
                q, r = target
                self._push_undo()
                self.backend.remove_ring(q, r)
                self._refresh()
            # RMB on empty: no action
        elif event.button == 3:  # Middle click
            if self._block_planar(): return
            self._toggle_h_at(p_world)

    def on_move(self, p_world, r0=None, rd=None):
        target_type, target = self.resolve_target(p_world, r0=r0, rd=rd)
        if target_type == 'atom':
            self._hover_atom(target)
            next_elem = self._ATOM_CYCLE.get(target.ename, 'C')
            self.gui.statusBar().showMessage(f"Atom {target.ename}→{next_elem} (LMB) | Drag: Move | Ctrl+Drag: Bond | RMB: Delete | Ctrl+RMB: Bridge")
        elif target_type == 'bond':
            self._hover_bond(target)
            cur_name = self._BOND_ORDER_NAMES.get(target.order, f'{target.order}')
            cycle = self.backend._BOND_ORDER_CYCLE
            idx = next((i for i, v in enumerate(cycle) if abs(v - target.order) < 0.01), 0)
            next_order = cycle[(idx + 1) % len(cycle)]
            next_name = self._BOND_ORDER_NAMES.get(next_order, f'{next_order}')
            self.gui.statusBar().showMessage(f"Bond {cur_name}→{next_name} (LMB) | RMB: Delete | Ctrl+LMB: Insert atom | Ctrl+RMB: Collapse")
        elif target_type == 'hex':
            if not self.b2Dview:
                self.gui.statusBar().showMessage("3D view — hex disabled (Enter → 2D)")
                return
            q, r = target
            from spammm.topology.HexGrid import snap_to_grid
            ring_nodes = self.backend.grid.ring_nodes(q, r)
            hover_pos = [[snap_to_grid(n)[0], snap_to_grid(n)[1], -0.08] for n in ring_nodes]
            if hover_pos:
                self.gui.hover_markers.set_data(pos=np.array(hover_pos, dtype=np.float32), symbol='disc', edge_width=2, edge_color='orange', face_color='transparent', size=12)
                self.gui.hover_markers.visible = True
            self.gui.statusBar().showMessage(f"Hex ring ({q},{r}) — LMB: Add | RMB: Remove (preserve shared)")
        else:
            if self.b2Dview:
                self.gui.statusBar().showMessage(f"Empty — LMB: Add {self.cur_atom_type} atom | RMB: Nothing")
            else:
                self.gui.statusBar().showMessage("3D view — atom/bond only (Enter → 2D for hex/empty)")

    def on_rmb_atom(self, atom_id, ctrl):
        self._remove_atom(atom_id, ctrl)

    def on_link(self, from_id, to_id):
        self._create_bond(from_id, to_id)

    def on_atom_click(self, atom_id, shift=False):
        debug_print(2, f"[ATOM_CLICKED] atom_id={atom_id} shift={shift}")
        self._push_undo()
        if shift:
            idx_map = getattr(self.backend, '_atom_idx_map', {})
            idx = idx_map.get(int(atom_id))
            if idx is not None:
                current_npi = self.backend.atom_npi[idx]
                new_npi = (current_npi + 1) % 3
                self.backend.set_atom_npi_by_id(atom_id, new_npi)
                if self.backend.auto_h_cap:
                    self.backend.adjust_h()
        else:
            self.backend.cycle_atom_type(atom_id)
        self._refresh()


# ── Atom mode ───────────────────────────────────────────────────────────────

class AtomMode(EditModeHandler):
    status_msg = "LMB: Add/Change type/Drag move | Shift+LMB: Cycle npi | Ctrl+LMB drag: Create bond | RMB: Delete (Ctrl: bridge) | Scroll: Zoom"
    link_mode = True

    def on_press(self, event, p_world, ctrl):
        if event.button == 1:
            if ctrl: return  # Ctrl+LMB handled by scene link mode (bond creation / new atom)
            # Atom under cursor: scene owns pick/drag/click — do not add free atom
            if self.backend.pick_atom(p_world, radius=self.pick_radius):
                return
            selected = self.scene.get_selected_ids()
            if selected: return
            if self._block_planar(): return
            self._add_free_atom(p_world)
        elif event.button == 2:
            pass  # Handled by sig_rmb_remove → on_rmb_atom
        elif event.button == 3:
            if self._block_planar(): return
            self._toggle_h_at(p_world)

    def on_move(self, p_world, r0=None, rd=None):
        hovered_atom = self.backend.pick_atom(p_world, radius=self.pick_radius)
        if hovered_atom:
            self._hover_atom(hovered_atom)
            debug_print(3, f"Hovered atom: {hovered_atom}")

    def on_rmb_atom(self, atom_id, ctrl):
        self._remove_atom(atom_id, ctrl)

    def on_link(self, from_id, to_id):
        self._create_bond(from_id, to_id)

    def on_atom_click(self, atom_id, shift=False):
        debug_print(2, f"[ATOM_CLICKED] atom_id={atom_id} shift={shift}")
        self._push_undo()
        if shift:
            idx_map = getattr(self.backend, '_atom_idx_map', {})
            idx = idx_map.get(int(atom_id))
            if idx is not None:
                current_npi = self.backend.atom_npi[idx]
                new_npi = (current_npi + 1) % 3
                self.backend.set_atom_npi_by_id(atom_id, new_npi)
                if self.backend.auto_h_cap:
                    self.backend.adjust_h()
        else:
            self.backend.set_atom_type_by_id(atom_id, self.cur_atom_type)
        self._refresh()


# ── Pi mode ─────────────────────────────────────────────────────────────────

class PiMode(EditModeHandler):
    status_msg = "LMB: Add/Toggle | RMB: Remove | Middle-Click: Toggle H | Scroll: Zoom"

    def on_activate(self):
        self.gui.set_label_mode('Pi Orbitals')

    def on_press(self, event, p_world, ctrl):
        if event.button == 1:
            # Atom under cursor: scene owns pick/drag/click — do not add free atom
            if self.backend.pick_atom(p_world, radius=self.pick_radius):
                return
            selected = self.scene.get_selected_ids()
            if selected: return
            if self._block_planar(): return
            self._add_free_atom(p_world)
        elif event.button == 2:
            pass  # Handled by sig_rmb_remove → on_rmb_atom
        elif event.button == 3:
            if self._block_planar(): return
            self._toggle_h_at(p_world)

    def on_atom_click(self, atom_id, shift=False):
        debug_print(2, f"[ATOM_CLICKED] atom_id={atom_id} (pi cycle) shift={shift}")
        idx_map = getattr(self.backend, '_atom_idx_map', {})
        idx = idx_map.get(int(atom_id))
        if idx is None: return
        self._push_undo()
        current_npi = self.backend.atom_npi[idx]
        new_npi = (current_npi + 1) % 3
        self.backend.set_atom_npi_by_id(atom_id, new_npi)
        if self.backend.auto_h_cap:
            self.backend.adjust_h()
        self.gui.refresh_view()

    def on_move(self, p_world, r0=None, rd=None):
        hovered_atom = self.backend.pick_atom(p_world, radius=self.pick_radius)
        if hovered_atom:
            self._hover_atom(hovered_atom)
            debug_print(3, f"Hovered atom: {hovered_atom}")

    def on_rmb_atom(self, atom_id, ctrl):
        self._remove_atom(atom_id, ctrl)


# ── Bond mode ───────────────────────────────────────────────────────────────

class BondMode(EditModeHandler):
    status_msg = "LMB: Insert atom (Ctrl: push aside) | RMB: Delete bond (Ctrl: collapse) | Scroll: Zoom"
    lock_drag = True

    def on_press(self, event, p_world, ctrl):
        bond = self.backend.pick_bond(p_world, radius=self.pick_radius)
        if bond is None: return
        if event.button == 1:
            self._push_undo()
            new_atom = self.backend.insert_atom_into_bond(bond, self.cur_atom_type, push_aside=ctrl)
            debug_print(2, f"Inserted atom into bond {bond._id}, new atom {new_atom._id} (push_aside={ctrl})")
            self._refresh()
        elif event.button == 2:
            self._push_undo()
            if ctrl:
                survivor = self.backend.collapse_bond(bond, np.array([p_world[0], p_world[1]]))
                debug_print(2, f"Collapsed bond {bond._id}, survivor atom {survivor._id}")
            else:
                self.backend.delete_bond(bond)
                debug_print(2, f"Deleted bond {bond._id}")
            self._refresh()

    def on_move(self, p_world, r0=None, rd=None):
        bond = self.backend.pick_bond(p_world, radius=0.5)
        if bond:
            self._hover_bond(bond)
            debug_print(3, f"Hovered bond: {bond}")


# ── Ring mode ───────────────────────────────────────────────────────────────

class RingMode(EditModeHandler):
    status_msg = "LMB: Add ring on bond/corner/hex | RMB: Delete bond/atom/ring | Numpad +/-: Ring size"
    lock_drag = True
    ring_size_visible = True

    def _pick_hex(self, p_world):
        """Return (q,r) if near a hex center, else None."""
        if not hasattr(self.backend, 'snap_to_ring'): return None
        q, r = self.backend.snap_to_ring(p_world[0], p_world[1])
        ring_nodes = self.backend.grid.ring_nodes(q, r)
        from spammm.topology.HexGrid import snap_to_grid
        center = np.mean([snap_to_grid(n)[:2] for n in ring_nodes], axis=0)
        if np.linalg.norm(center - p_world[:2]) < self.backend.a_CC * 0.5:
            return (q, r)
        return None

    def on_press(self, event, p_world, ctrl):
        r0 = getattr(self.gui, '_last_r0', None)
        rd = getattr(self.gui, '_last_rd', None)
        p_xy = self._mouse_xy_plane(p_world, r0, rd)
        self.backend.detect_geometry_rings()
        # 3D: closest among atom/bond/ring-COG. 2D: bond then atom then ring (legacy XY).
        # Hex is the only 3D-blocked path.
        kind, target = None, None
        if not self.b2Dview and r0 is not None and rd is not None:
            kind, target = self._pick_ring_target_3d(r0, rd)
        else:
            bond = self.backend.pick_bond(p_xy, radius=self.pick_radius)
            if bond is not None:
                kind, target = 'bond', bond
            else:
                atom = self.backend.pick_atom(p_xy, radius=self.pick_radius)
                if atom is not None:
                    kind, target = 'atom', atom
                else:
                    self.backend.detect_geometry_rings()
                    ring = self.backend.pick_ring(p_xy, radius=self.pick_radius)
                    if ring is not None:
                        kind, target = 'ring', ring

        if kind == 'bond':
            bond = target
            if event.button == 1:
                self._push_undo()
                n = int(self.gui.ring_size_spinbox.value())
                side = self.backend.compute_ring_side(bond, p_xy)
                new_atoms = self.backend.add_adjacent_ring(bond, n_members=n, ename=self.cur_atom_type, side=side)
                debug_print(2, f"Added {n}-ring on bond {bond._id} side={side}, {len(new_atoms)} new atoms")
                self._refresh()
            elif event.button == 2:
                self._push_undo()
                self.backend.delete_bond(bond)
                debug_print(2, f"Deleted bond {bond._id}")
                self._refresh()
            return

        if kind == 'atom':
            atom = target
            if event.button == 1:
                n = int(self.gui.ring_size_spinbox.value())
                verts = self.backend.compute_corner_ring_positions(atom, n, p_xy)
                if verts is not None:
                    self._push_undo()
                    new_atoms = self.backend.add_corner_ring(atom, n_members=n, ename=self.cur_atom_type, mouse_pos=p_xy)
                    debug_print(2, f"Added {n}-ring at corner atom {atom._id}, {len(new_atoms)} new atoms")
                    self._refresh()
                    return
            elif event.button == 2:
                self._push_undo()
                if ctrl:
                    self.backend.remove_atom_with_bridge(atom._id)
                    debug_print(2, f"Bridged atom {atom._id}")
                else:
                    self.backend.remove_atom_by_id(atom._id)
                    debug_print(2, f"Deleted atom {atom._id}")
                self._refresh()
                return

        if kind == 'ring' and event.button == 2:
            hovered_ring = target
            atom_ids = [a._id for a in hovered_ring.atoms if a.alive]
            self._push_undo()
            self.backend.remove_atoms_by_id(atom_ids)
            debug_print(2, f"Deleted ring {hovered_ring._id} ({len(atom_ids)} atoms)")
            self._refresh()
            return

        # Hex grid center → hex ring (LMB only, 2D only)
        hex_pos = self._pick_hex(p_xy)
        if hex_pos is not None and event.button == 1:
            if self._block_planar(): return
            q, r = hex_pos
            self._push_undo()
            self.backend.add_ring(q, r)
            debug_print(2, f"Added hex ring at ({q},{r})")
            self._refresh()

    def on_move(self, p_world, r0=None, rd=None):
        n = int(self.gui.ring_size_spinbox.value())
        p_xy = self._mouse_xy_plane(p_world, r0, rd)
        self.backend.detect_geometry_rings()
        self.scene.hover_ring_lines.visible = False
        self.scene.hover_ring_markers.visible = False
        self.scene.hover_ring_text.text = ''

        kind, target = None, None
        if not self.b2Dview and r0 is not None and rd is not None:
            kind, target = self._pick_ring_target_3d(r0, rd)
        else:
            bond = self.backend.pick_bond(p_xy, radius=0.5)
            if bond is not None:
                kind, target = 'bond', bond
            else:
                atom = self.backend.pick_atom(p_xy, radius=self.pick_radius)
                if atom is not None:
                    kind, target = 'atom', atom
                else:
                    ring = self.backend.pick_ring(p_xy, radius=self.pick_radius)
                    if ring is not None:
                        kind, target = 'ring', ring

        if kind == 'bond':
            bond = target
            self._hover_bond(bond)
            debug_print(3, f"Hovered bond: {bond}")
            side = self.backend.compute_ring_side(bond, p_xy)
            verts = self.backend.compute_adjacent_ring_positions(bond, n, side)
            closed = np.vstack([verts, verts[:1]]).astype(np.float32)
            self.scene.ring_preview_line.set_data(pos=closed, color=(0.2, 0.8, 0.8, 0.6))
            self.scene.ring_preview_line.visible = True
            self.gui.hover_markers.visible = False
            return

        if kind == 'atom':
            atom = target
            verts = self.backend.compute_corner_ring_positions(atom, n, p_xy)
            if verts is not None:
                self._hover_atom(atom)
                closed = np.vstack([verts, verts[:1]]).astype(np.float32)
                self.scene.ring_preview_line.set_data(pos=closed, color=(0.2, 0.8, 0.8, 0.6))
                self.scene.ring_preview_line.visible = True
                debug_print(3, f"Hovered corner atom: {atom._id} for {n}-ring")
                self.gui.hover_markers.visible = False
                return
            # Atom hit but no corner ring possible — still highlight atom
            self._hover_atom(atom)
            self.scene.ring_preview_line.visible = False
            self.gui.hover_markers.visible = False
            return

        if kind == 'ring':
            hovered_ring = target
            alive_atoms = [a for a in hovered_ring.atoms if a.alive]
            if not alive_atoms:
                return
            ring_pos = np.array([a.pos for a in alive_atoms] + [alive_atoms[0].pos], dtype=np.float32)
            self.scene.hover_ring_lines.set_data(pos=ring_pos, color=(1.0, 0.9, 0.0, 0.4))
            self.scene.hover_ring_lines.visible = True
            cog_lines = []
            for atom in hovered_ring.atoms:
                if atom.alive:
                    cog_lines.append(hovered_ring.cog)
                    cog_lines.append(atom.pos)
            cog_arr = np.array(cog_lines, dtype=np.float32) if cog_lines else np.zeros((0, 3), dtype=np.float32)
            self.scene.hover_ring_markers.set_data(pos=cog_arr)
            self.scene.hover_ring_markers.visible = True
            self.scene.hover_ring_text.pos = hovered_ring.cog
            self.scene.hover_ring_text.text = f"R{len(hovered_ring.atoms)}"
            self.gui.statusBar().showMessage(f"Ring {hovered_ring._id} ({len(hovered_ring.atoms)} atoms) — RMB: Delete whole ring")
            debug_print(3, f"Hovered ring: {hovered_ring} (n={len(hovered_ring.atoms)})")
            self.scene.ring_preview_line.visible = False
            self.gui.hover_markers.visible = False
            return

        # Hex grid preview (2D only)
        if not self.b2Dview:
            self.scene.ring_preview_line.visible = False
            self.gui.hover_markers.visible = False
            return
        hex_pos = self._pick_hex(p_xy)
        if hex_pos is not None:
            q, r = hex_pos
            from spammm.topology.HexGrid import snap_to_grid
            ring_nodes = self.backend.grid.ring_nodes(q, r)
            hover_pos = np.array([[snap_to_grid(nd)[0], snap_to_grid(nd)[1], -0.08] for nd in ring_nodes], dtype=np.float32)
            closed = np.vstack([hover_pos, hover_pos[:1]]).astype(np.float32)
            self.scene.ring_preview_line.set_data(pos=closed, color=(0.2, 0.8, 0.8, 0.6))
            self.scene.ring_preview_line.visible = True
            self.gui.hover_markers.set_data(pos=hover_pos, symbol='disc', edge_width=2, edge_color='orange', face_color='transparent', size=12)
            self.gui.hover_markers.visible = True
            debug_print(3, f"Hovered hex ({q},{r})")
            return
        self.scene.ring_preview_line.visible = False
        self.gui.hover_markers.visible = False


# ── Hex modes ───────────────────────────────────────────────────────────────

class HexMode(EditModeHandler):
    lock_drag = True

    def on_press(self, event, p_world, ctrl):
        if self._block_planar(): return
        q, r = self.backend.snap_to_ring(p_world[0], p_world[1])
        if event.button == 1:
            self._push_undo()
            self.backend.add_ring(q, r)
            self._refresh()
        elif event.button == 2:
            self._push_undo()
            self.backend.remove_ring(q, r)
            self._refresh()
        elif event.button == 3:
            self._toggle_h_at(p_world)

    def on_move(self, p_world, r0=None, rd=None):
        if not self.b2Dview:
            self.gui.statusBar().showMessage("3D view — Hex mode disabled (Enter → 2D)")
            return
        self.backend.detect_geometry_rings()
        hovered_ring = self.backend.pick_ring(p_world, radius=1.0)
        if hovered_ring:
            ring_pos = np.array([a.pos for a in hovered_ring.atoms] + [hovered_ring.atoms[0].pos], dtype=np.float32)
            self.scene.hover_ring_lines.set_data(pos=ring_pos)
            cog_lines = []
            for atom in hovered_ring.atoms:
                cog_lines.append(hovered_ring.cog)
                cog_lines.append(atom.pos)
            self.scene.hover_ring_markers.set_data(pos=np.array(cog_lines, dtype=np.float32))
            self.scene.hover_ring_text.pos = hovered_ring.cog
            self.scene.hover_ring_text.text = str(len(hovered_ring.atoms))
            debug_print(3, f"Hovered ring: {hovered_ring} (n={len(hovered_ring.atoms)})")
        if hasattr(self.backend, 'snap_to_ring'):
            from spammm.topology.HexGrid import snap_to_grid
            q, r = self.backend.snap_to_ring(p_world[0], p_world[1])
            ring_nodes = self.backend.grid.ring_nodes(q, r)
            hover_pos = [[snap_to_grid(n)[0], snap_to_grid(n)[1], -0.08] for n in ring_nodes]
            if hover_pos:
                self.gui.hover_markers.set_data(pos=np.array(hover_pos, dtype=np.float32), symbol='disc', edge_width=2, edge_color='orange', face_color='transparent', size=12)
                self.gui.hover_markers.visible = True


class Hex1Mode(HexMode):
    status_msg = "Hex1 (paint: force add/remove): LMB: Add | RMB: Remove"

    def on_activate(self):
        self.backend.hex_mode = 'Hex1'


class Hex2Mode(HexMode):
    status_msg = "Hex2 (toggle: preserve shared): LMB: Add | RMB: Remove"

    def on_activate(self):
        self.backend.hex_mode = 'Hex2'


# ── Select mode ─────────────────────────────────────────────────────────────

class SelectMode(EditModeHandler):
    status_msg = "LMB: Add to selection | RMB: Remove | δ corner: sticky move | φ corner: sticky rotate | Ctrl+C/V | Delete | LMB/RMB-drag empty: box add/remove"
    selection_mode = True
    lock_drag = True

    def on_move(self, p_world, r0=None, rd=None):
        if self.scene._xform_mode is not None:
            mode = self.scene._xform_mode
            self.gui.statusBar().showMessage(f"Sticky {mode} (δ/φ) — move mouse, LMB click to drop")
            return
        hovered_atom = self.backend.pick_atom(p_world, radius=self.pick_radius)
        if hovered_atom:
            self._hover_atom(hovered_atom)
            debug_print(3, f"Hovered atom: {hovered_atom}")

    def on_rmb_atom(self, atom_id, ctrl):
        # Select mode: RMB is remove-from-selection (handled in AtomScene), not delete
        pass
