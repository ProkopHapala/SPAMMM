---
name: visual-debugging
description: Use when creating diagnostic plots, visualizations, or headless visual tests for debugging
trigger:
  glob:
    - "**/tests/**/*"
    - "**/*test*.py"
    - "**/*debug*.py"
    - "**/test_*.sh"
    - "**/run_*.sh"
    - "**/*benchmark*.py"
---

## Rules

1. **Reuse before reinvent.** Check existing helpers before writing plotting code:
   - `tests/helpers/geometry.py` — `plot_geometry()`, `render_graph()`, `render_before_after()`, bond/angle checks
   - `tests/helpers/parity.py` — `plot_curves()`, `overlay_plot()`, `assert_parity()`, RMSE/correlation
   - `tests/helpers/topology_test.py` — `TopologySnapshot`, `TopologyDiff` for graph before/after assertions
   - `spammm/GUI/VispyUtils.py` — `AtomScene` for interactive 3D (not for headless tests)

2. **Two-layer testing for visual/editing features:**
   - **L1 (always):** Assert topology diff via `TopologySnapshot`/`TopologyDiff`. Fast, deterministic, no display.
   - **L2 (`--visual` flag or `@pytest.mark.visual`):** Render before/after PNG with matplotlib Agg. Annotate cursor position (clicks), highlight selected atoms, color diff (green=added, red=removed).

3. **Test backend logic, not GUI widgets.** Simulate user actions via direct API calls:
   - Click atom → `graph.pick_atom(pos, radius=0.5)`
   - Press button → call backend method directly (e.g., `backend.add_ring(q, r)`)
   - Selection query → `compile_select_query(q); apply_select_query(graph, compiled)`

4. **Output organization:** Save PNGs to `tests/visual_output/{test_name}.png`. Report exact paths.

5. **Plot style for comparison curves:**
   - Reference: `ls=':'`, `lw=1.5` (dotted, thick)
   - Model: `ls='-'`, `lw=0.5` (solid, thin)
   - Subtract DC offset: `dc = mean(model - ref)` before plotting
   - Residual on twin axis: `(model_shifted - ref) * 100`, labeled `diff x100`
   - RMSE/MaxErr text box: upper-left, monospace, semi-transparent

6. **No `plt.show()` in library code.** Only in CLI/main entry points. Use `--saveFig` / `--noPlot` flags.

7. **Foreground execution.** Never hide output (`| tail`, `| head`, `&`). Full stdout visible.

## Headless Visual Testing for Molecular Editing

**Full design doc:** `doc/HowTo/VisualDEbugging.md`

Molecular editing features are visual (clicks, selections, buttons). Test them headlessly:

- **Topology snapshot/diff:** `TopologySnapshot(graph)` captures atoms/bonds/rings/neighbors by stable `_id`. `snap_before.diff(snap_after)` returns `TopologyDiff` with `added_atoms`, `removed_bonds`, etc. Use `diff.assert_counts(name, added_atoms=6, added_bonds=6)` — no hardcoding atom IDs.
- **GUI simulation table:**
  - Click atom → `graph.pick_atom(world_pos, radius=0.5)`
  - Click bond → `graph.pick_bond(world_pos, radius=0.5)`
  - Click ring → `graph.pick_ring(world_pos, radius=1.0)`
  - Button press → call backend method directly (e.g., `backend.add_ring(q, r)`)
  - Selection query → `compile_select_query(q); apply_select_query(graph, compiled)`
- **Visual annotations on before/after PNGs:**
  - Cursor crosshair at world-space click position
  - Colored halo around selected atom IDs
  - Diff coloring: green=added, red=removed, blue=new bonds, dashed red=removed bonds
  - Atom labels: `id:element` (e.g., `42:C`)
- **pytest setup:** `--visual` flag → `visual_output_dir` fixture returns `tests/visual_output/` or `None`. L1 assertions always run; L2 rendering only when fixture is not `None`.
- **Stable IDs are key:** `AtomicGraph` uses object identity (`_id`), not array indices. Atoms don't renumber after deletion. Snapshots are reliable across operations.

- **Output location policy:** All diagnostic scripts and visual test outputs must be saved under `debug/<script_name>/` (e.g., `debug/plot_fdbm_relax/`). The `<script_name>` is the generating script's name without `.py`. Subfolders are allowed for organization. **Never** write to `/tmp/` or the repository root. This keeps artifacts persistent and easy to find.
- **Structured outputs:** Group all debugging, benchmarking, and testing outputs into organized, numbered directories (e.g., `tests/003_case_name/`). Do not clutter root directories. Explicitly report their location.
- **Foreground execution:** Run tests synchronously in the foreground with full output. Never hide output or use background commands (`&`, `| tail`, `| head`, or silent redirects). Full `stdout` must be visible.