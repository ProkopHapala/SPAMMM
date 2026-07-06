# Test Design for SPAMMM

**SSOT for the test system.** Skills (`running-tests`, `visual-debugging`, `reference-data`) link here.

---

## Philosophy

Three review levels on a shared pytest infrastructure:

| Level | Name | Verdict by | Mechanism |
|-------|------|------------|-----------|
| **L0** | Automatic | Machine | `assert`, `assert_parity`, `TopologyDiff`, `ref_data/` |
| **L1** | Agentic | LLM or human reading text | `.out` + `.log` in `debug/<script>/` |
| **L2** | Visual | Human eyes | `.png` via `--visual` or `--develop` |

**Core rules:**

- Every test has **L0 asserts** — even permissive ones (`np.isfinite`, `natoms > 0`). Thinking hard about what can be asserted automatically is part of the job.
- **If you can produce a good reference, you don't need L1** — assert distance from reference instead. See skill:`reference-data`.
- Prefer **optional sidecars** (L1/L2) inside pytest functions over standalone scripts.
- **Few modules, data-driven cases** where possible. Reuse `tests/helpers/`.

---

## Decision tree (writing a new test)

```
1. Can you assert a number/invariant confidently?     → L0 assert
2. Can you assert topology counts (TopologyDiff)?     → L0 + TopologySnapshot
3. Need semantic judgment (chemistry, layout)?        → L0 (permissive) + L1 .out
4. Need human eyes on a plot?                         → L0 + L2 --visual
5. No assertable core, only exploration?              → testplot_*.py (not pytest)
```

---

## Run modes

| Mode | Command | L0 | L1 (.out/.log) | L2 (.png) | Verbosity |
|------|---------|----|----------------|-----------|-----------|
| **Routine** | `pytest -m "not slow"` | on | off | off | low |
| **Develop** | `pytest path --develop -s` | on | **on + agent must read .out** | **on** | high |

Individual flags: `--review` (L1 only), `--visual` (L2 only), `--develop` (both + verbosity).

```bash
pytest -m "not slow"                              # routine regression
pytest tests/topology/test_editing_ops.py -s      # L0 only
pytest tests/topology/test_editing_ops.py --develop -s   # L0+L1+L2
pytest tests/topology/test_editing_ops.py --review -s    # L0+L1
pytest -m "gpu and not slow"                      # GPU fast tests
pytest --update-refs tests/test_folded_relax.py   # refresh ref_data
```

### Agent contract (develop mode)

1. Run pytest **in foreground** — never `| grep`, `| tail`, `| head`, or `&`.
2. Read stdout; follow every `REVIEW: debug/...` line.
3. Read **`.out` first** (curated evaluation packet).
4. If suspicious, read **`.log`** (verbose trace).
5. Report findings; give user `.png` paths for L2.

### Output visibility (non-negotiable)

Full stdout is part of the test result. Users wait at the terminal; silent long runs get killed. Use `-s` in develop mode. Verbosity via `spammm.globals.debug_print` and `SPAMMM_VERBOSITY`.

---

## File naming

| Pattern | Collected by pytest? | Purpose |
|---------|---------------------|---------|
| `test_*.py` | Yes | L0 (+ optional L1/L2 flags) |
| `testplot_*.py` | **No** (`collect_ignore`) | Pure visual demos, run via `python tests/...` |
| `run_*.py` | No | CLI utilities (trajectories, batch jobs) |
| `helpers/` | No | Shared utilities |

Examples:
- `test_editing_ops.py` — pytest, topology editing
- `testplot_tensor_parity.py` — standalone GPU parity plots
- `run_manipulation.py` — CLI relaxed scan export

---

## Artifact layout

All artifacts under **`debug/<script_stem>/`** (script name without `.py`):

```
debug/test_editing_ops/
  test_01_add_atom_grid.out       # L1 curated (agent reads first)
  test_01_add_atom_grid.log       # L1 trace
  test_01_add_atom_grid.png       # L2 visual
  test_07_insert_into_bond.mol2   # optional, when geometry matters
```

Multi-step: `test_foo.operation1.out`, `test_foo.step2.log`, etc.

`debug/` is **gitignored** except `debug/README.md`. Artifacts are ephemeral; agents read by explicit path after run. Do not add `debug/` to `.cursorignore`.

---

## Three output channels

| Channel | Content | When |
|---------|---------|------|
| **stdout** | Progress, timing, scalars, `REVIEW:` pointers | Always |
| **`.out`** | Curated packet: intent, atom table, metrics, agent checklist | `--review` or `--develop` |
| **`.log`** | Full execution trace at high verbosity | `--review` or `--develop` |

**Test-author duty:** Before coding, ask *what would I need to see to know this is wrong?* Put that in `.out`. Put dig details in `.log`.

`.out` template:

```markdown
## Intent
Inserted N into C-C bond; expect ring preserved.

## Graph (no coords)
<AtomicGraph.format_table output>

## Metrics
natoms=7  nbonds=8  diff=TopologyDiff(...)

## Agent checklist
1. New atom is N between correct pair
2. No degree-0 atoms
```

---

## Molecular text dumps

Use **`AtomicGraph.format_table()`** — one atom per line, stable `uid`:

```
# uid  elem  hyb  npi  neighs
  42  C     sp2  1    43,44,45
  43  H     sp3  0    42
```

Flags: `pos=False` (default), `neighbors=True`, `bond_orders=False`, `charge=False`.

When geometry matters, export **`.mol2`** via existing export paths. For arrays: `debug_summarize_array` → shape, dtype, min/max, finite check.

Invariants vocabulary: skill:`numerical-parity`, skill:`forcefield-validation`.

---

## Pytest infrastructure

### `conftest.py` fixtures

| Fixture | Purpose |
|---------|---------|
| `xyz`, `substrate`, `dat` | Data path loaders |
| `update_refs` | `--update-refs` flag |
| `develop_mode` | True if `--develop` |
| `review_enabled` | True if `--review` or `--develop` |
| `visual_output_dir` | `debug/<script>/` or None |
| `review_dir` | Same dir when review on |
| `make_review` | Factory → `ReviewSession` |

### Markers (`pytest.ini`)

- `slow` — >1s; excluded from default runs
- `gpu` — requires OpenCL
- `visual` — produces L2 plots (may still assert)
- `review` — documents L1-capable tests

### Helpers (`tests/helpers/`)

| Module | Role |
|--------|------|
| `parity.py` | RMSE, correlation, `overlay_plot`, `assert_parity` |
| `geometry.py` | bond lengths, angles, planarity, distort |
| `scan.py` | z-scan, x-scan, `compare_scans` |
| `topology_test.py` | `TopologySnapshot`, `TopologyDiff`, `render_before_after` |
| `folded_rigid.py` | rigid body relax, manipulation, ref compare |
| `review.py` | `ReviewSession`, `.out`/`.log` writers |

---

## Reference data vs agent judgment

| Situation | Tool |
|-----------|------|
| Established behavior ("did we break it?") | `tests/ref_data/*.ref.json` + L0 |
| New feature, no reference yet | L1 agent reads `.out` |
| Reference good enough | Promote to `ref_data/`, drop L1 for that check |

References are git-tracked. Debug artifacts are not. See skill:`reference-data`.

---

## Topology / editing tests (reference pattern)

`tests/topology/test_editing_ops.py`:

- **L0:** `TopologySnapshot` / `TopologyDiff.assert_counts`
- **L1:** `make_review('test_foo')` → `.out`/`.log` with `--develop`
- **L2:** `render_before_after` → `.png` with `--visual`

Simulate GUI via backend API (`MoleculeEditorBackend`), not widget clicks.

---

## Visual demos (`testplot_*`)

Standalone scripts for exploration without assertable core:

```bash
python tests/testplot_tensor_parity.py
python tests/topology/testplot_kekule.py
```

Output: `debug/<script_stem>/`. No `def test_*` inside (or pytest ignores via `testplot_` prefix).

Integrate plotting into pytest when there **is** an assertable core — use `--visual` flag instead of a separate script.

---

## Plot style (L2)

- Reference curve: `ls=':'`, `lw=1.5`
- Model curve: `ls='-'`, `lw=0.5`
- Residual on twin axis: `(model - ref) * 100`
- RMSE/MaxErr text box, upper-left, monospace
- No `plt.show()` in library code

---

## Execution time

- Default: each test **< 1s** (hard limit 5s → mark `@pytest.mark.slow`)
- Default run: `pytest -m "not slow"`
- Keep inputs small; mark slow tests and document why

---

## Directory map (current)

```
tests/
  conftest.py
  test_*.py              # physics, GUI, integration
  testplot_*.py          # visual-only demos (not collected)
  run_*.py               # CLI utilities
  topology/              # editing, Kekule, ascii art
  SPM/                   # AFM / probe microscopy
  surfaces/              # GridFF utilities
  helpers/
  ref_data/              # git-tracked regression references
```

See `tests/README.md` for file-level index.
