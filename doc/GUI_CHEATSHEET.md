# SPAMMM GUI — Keyboard & Mouse Cheatsheet

## Launch

```bash
./run_gui.sh                          # empty editor
./run_gui.sh --m path/to/molecule.mol2  # load molecule
```

## Global Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+C` | Copy selected atoms (Select mode) |
| `Ctrl+V` | Paste atoms from clipboard |
| `Delete` | Delete selected atoms (Select mode) |
| `Scroll` | Zoom |

## Edit Modes (dropdown in left panel)

### Unified
All-in-one mode — context-sensitive.

| Action | Effect |
|--------|--------|
| LMB on atom | Cycle atom type (C→N→O→C) |
| LMB on bond | Cycle bond order (single→double→triple→aromatic) |
| LMB on hex center | Add hexagonal ring |
| LMB on empty space | Add free atom (current type) |
| LMB drag on atom | Move atom |
| RMB on atom | Delete atom (Ctrl: bridge neighbors) |
| Ctrl+LMB drag atom→atom | Create bond |
| Middle-click | Toggle H atom at position |

### Atom
Focused atom editing.

| Action | Effect |
|--------|--------|
| LMB on empty | Add free atom |
| LMB on atom | Set atom to current type |
| LMB drag | Move atom |
| RMB on atom | Delete (Ctrl: bridge neighbors) |
| Ctrl+LMB drag | Create bond |
| Middle-click | Toggle H |

### Bond
Bond operations.

| Action | Effect |
|--------|--------|
| LMB on bond | Insert atom into bond (Ctrl: push aside) |
| RMB on bond | Delete bond (Ctrl: collapse bond) |

### Ring
Add fused n-gon rings. **Numpad +/-** changes ring size (3–12).

| Action | Effect |
|--------|--------|
| LMB on **bond** | Add edge ring (shares 1 bond) |
| LMB on **inner corner atom** (angle < 180°) | Add corner ring (shares 2 bonds) |
| LMB on **hex grid center** | Add hexagonal ring on grid |
| RMB on **existing ring** (hover shows yellow highlight) | Delete all atoms in that ring |
| RMB on **bond** | Delete bond |
| RMB on **atom** | Delete single atom (Ctrl+RMB: bridge neighbors) |
| Numpad `+` / `=` | Increase ring size |
| Numpad `-` / `_` | Decrease ring size |

Ring placement priority (LMB): bond → corner atom → hex center.
Ring deletion priority (RMB): existing ring → bond → atom.
Outer corners (>180°) are ignored — falls through to hex/edge.
Debug View toggle shows all detected rings (yellow COG crosses + bounding circles).

### Pi
Toggle pi-orbital participation (sp2 vs sp3).

| Action | Effect |
|--------|--------|
| LMB on atom | Cycle npi (1↔-1, sp2↔sp3) |
| LMB on empty | Add free atom |
| RMB on atom | Delete |
| Middle-click | Toggle H |

### Hex1 (paint)
Force add/remove hex grid nodes.

| Action | Effect |
|--------|--------|
| LMB on hex center | Add hex ring (force) |
| RMB on hex center | Remove hex ring (force) |

### Hex2 (toggle)
Add/remove preserving shared edges.

| Action | Effect |
|--------|--------|
| LMB on hex center | Add hex ring (preserve shared) |
| RMB on hex center | Remove hex ring (preserve shared) |

### Select
Selection and copy/paste.

| Action | Effect |
|--------|--------|
| LMB | Select/deselect atom |
| RMB drag | Box select |
| `Delete` | Remove selected |
| `Ctrl+C` | Copy selected to clipboard |
| `Ctrl+V` | Paste from clipboard |
| LMB drag selected | Move selected group |

## Grid Controls (left panel)

| Control | Effect |
|---------|--------|
| Transpose button | Transpose grid + atom geometry (swap x/y) |
| Flip X button | Flip atom geometry horizontally |
| Flip Y button | Flip atom geometry vertically |
| Rotate spin | Rotate grid |
| Offset X/Y spin | Shift grid origin |
| a_CC spin | Set C-C bond length (grid spacing) |
| Reset button | Reset all grid transforms |

## Tips

- **Ring mode** combines edge, corner, and hex ring placement — no need to switch modes.
- Corner ring uses circumcircle through the two existing bonds — preserves bond lengths.
- Ring size spinbox is only visible in Ring mode.
- Debug View toggle shows pin nodes (cyan) and atom→pin lines.
- Bond Colors toggle visualizes bond orders by color.
