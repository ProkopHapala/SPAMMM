# Task: GUI Tight Layout — Maximally Efficient Widget Packing

**Status:** Spec — problem analysis + design goals (2026-08-01)
**Priority:** P2 (GUI UX — sidebar still too wide despite LayoutPolicy)
**Related:** `spammm/GUI/LayoutPolicy.py`, `spammm/GUI/BaseGUI.py`, `spammm/GUI/SPAMMM_GUI.py`, all `*Extension.py`, `AGENTS.md` §GUI layout

---

## 1. Problem

Despite adding `LayoutPolicy.py` with `MARGIN=0`, `SPACING=1`, and `Maximum` size policy on buttons, the side panel is still wider than necessary. Several root causes remain:

### 1.1 Widgets still expand despite Maximum policy
- `QSpinBox`/`QDoubleSpinBox` have a large default `sizeHint` (includes up/down arrows + text field for 5 digits). Even with `Maximum` policy, they're wider than needed.
- `QComboBox` has a default `sizeHint` based on the longest item + arrow, but Qt adds padding.
- `QLabel` with `Preferred` policy expands to fill available space — labels in `QHBoxLayout` rows push other widgets apart.

### 1.2 Row layouts use `addStretch()` which wastes space
- Many `QHBoxLayout` rows end with `addStretch()` — this pushes widgets to the left but the row itself still takes full panel width. The row's height is determined by the tallest widget, and the row's width is the panel width (not the content width).

### 1.3 CollapsibleSection content margins
- `CollapsibleSection` has `setContentsMargins(MARGIN, 0, 0, MARGIN)` on content — with `MARGIN=0` this is fine, but the section itself is a `QWidget` that expands to full panel width.

### 1.4 No width constraint on the side_content widget
- `side_content.setMaximumWidth(PANEL_MAX_WIDTH)` caps the max, but the `QScrollArea` with `setWidgetResizable(True)` resizes the content to the scroll area's viewport width, which can be up to `PANEL_MAX_WIDTH=450`.

### 1.5 Extensions create widgets directly, bypassing BaseGUI
- Many extensions use `QtWidgets.QPushButton(...)`, `QtWidgets.QDoubleSpinBox(...)` directly instead of `self.button()`, `self.spinBox()`. These don't get the `Maximum` size policy automatically.
- The current migration added `setSizePolicy(Maximum, Fixed)` manually to each, but it's fragile and incomplete — any new extension or widget will miss it.

### 1.6 QFontMetrics not respected
- Qt's default button `sizeHint` includes padding (6px margins on each side for QPushButton). This can't be changed via size policy — it requires a stylesheet or custom paint.

---

## 2. USER preferences (design principles)

1. **Maximally tight layout** — eliminate all wasted space. Every pixel should contain a control or be removed.
2. **Width-limited side panel** — the panel must not push the canvas. Target ~280-320px, hard max ~400px.
3. **As many controls as possible in limited area** — density is the goal. If a row has 3 small widgets, they should fit, not wrap to 3 rows.
4. **Widgets only as wide as their content** — a button labeled "Go" should be ~30px, not 80px. A spinbox for 0-100 should be ~40px, not 60px.
5. **Zero or near-zero margins** — no decorative padding. Function over form.
6. **Consistent across extensions** — all extensions must follow the same tight layout rules, enforced centrally (not per-widget manual fixes).
7. **No `addStretch()` in rows** — rows should be natural width, not stretched to panel width. Use `FlowLayout` or left-aligned `QHBoxLayout` without stretch.

---

## 3. Design goals

### 3.1 Central enforcement (not per-widget manual fixes)
The current approach (manually adding `setSizePolicy` to each widget) is fragile. Instead:
- **`BaseGUI` widget factories are the only way to create widgets** — extensions must use `self.button()`, `self.spinBox()`, `self.comboBox()`, `self.label()`, etc. These factories enforce the policy.
- **A post-construction sweep** — after `build_ui()`, walk the widget tree and enforce `Maximum` size policy on all buttons/labels/combos/spinboxes that don't have it.

### 3.2 Reduce widget intrinsic sizes
- **Buttons:** Use a stylesheet to reduce internal padding: `QPushButton { padding: 1px 4px; }`
- **Spinboxes:** Use a stylesheet to reduce padding + set `setFixedWidth` based on `fontMetrics().width(max_value) + arrow_width`.
- **Labels:** Set `Maximum` size policy so they don't expand.
- **Combos:** Use `setMinimumContentsLength` to control width based on actual content.

### 3.3 Row layout without stretch
- Replace `QHBoxLayout` + `addStretch()` with `FlowLayout` for rows that may wrap.
- For fixed rows (label + spinbox pairs), use `QHBoxLayout` without stretch — the row's natural width is the sum of its children.

### 3.4 Panel width enforcement
- Set `side_content.setFixedWidth(PANEL_TARGET_WIDTH)` instead of `setMaximumWidth` — the panel should be exactly the target width, not expandable.
- The `QSplitter` still allows the user to resize, but the initial/fixed width is the target.

---

## 4. Implementation plan (phased)

### Phase 1: Stylesheet-based tight padding
- Add a global stylesheet in `BaseGUI.__init__` that reduces padding on all widget types:
  ```css
  QPushButton { padding: 1px 4px; min-width: 0; }
  QSpinBox, QDoubleSpinBox { padding: 0px 2px; min-width: 0; }
  QComboBox { padding: 0px 2px; min-width: 0; }
  QLabel { padding: 0px; }
  QCheckBox { padding: 0px; spacing: 2px; }
  ```
- This is the single most impactful change — it reduces intrinsic widget sizes globally without touching each widget.

### Phase 2: Post-construction enforcement sweep
- Add `LayoutPolicy.enforce(widget)` that recursively walks a widget tree and sets `Maximum` size policy on all buttons/labels/combos/spinboxes.
- Call it after `build_ui()` for each extension and after `initUI()` for the main GUI.

### Phase 3: Eliminate addStretch() in rows
- Audit all `addStretch()` calls in extension rows — replace with `FlowLayout` where wrapping is desired, or remove stretch where the row should be natural width.

### Phase 4: Spinbox width based on content
- In `BaseGUI.spinBox()`, compute `setFixedWidth` from `fontMetrics().width(str(vmax)) + arrow + padding` instead of `setMaximumWidth(SPIN_MAX_WIDTH)`.

### Phase 5: Panel fixed width
- Change `side_content.setMaximumWidth(PANEL_MAX_WIDTH)` to `side_content.setFixedWidth(PANEL_TARGET_WIDTH)`.

---

## 5. Files to modify

| File | Change |
|------|--------|
| `spammm/GUI/BaseGUI.py` | Add global stylesheet in `__init__`, enforce policy in widget factories |
| `spammm/GUI/LayoutPolicy.py` | Add `enforce(widget)` recursive sweep, stylesheet constant |
| `spammm/GUI/SPAMMM_GUI.py` | Call `enforce()` after `initUI()`, fix panel width |
| `spammm/GUI/ExtensionManager.py` | Call `enforce()` after each `build_ui()` |
| All `*Extension.py` | Remove `addStretch()` where appropriate, use `FlowLayout` for wrapping rows |
