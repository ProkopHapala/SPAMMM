
# USER

now we should think more deeply how to efficiently test molecular editing features which are to large degree visual in nature (e.g. controled/initiated in GUI by user clicking some atom, bond or pressing a buttom)

Now for rigorous testing of molecular editong features, we should make some headless tests which can be put to py-tests.

One option is to make tests of molecular topology directly (connection graph) if we do some topology operation like insert-remove chemical group is the connectivity graph really showing intendent change. But we should sill option to plot it and review by user showing the atom numbers.

Another thing is how to check GUI headlessly (GUI without GUI). We can do it by ploting the image. We use matplotlib or visipy to produce static image which we save, we rendr two screenshots, before and after, on before we indicate position of mouse cursor (if it is operation initiated by click) or selected atoms if it is operation on selection. 

can it think it through how to do it exacly in most robust and friction-less way ?

---

## GLM 5.2

> Molecular editing features are inherently visual (user clicks atoms, bonds, presses buttons in GUI).
> This section defines a **headless testing strategy** that enables rigorous pytest-based verification
> without requiring a display, while still producing visual artifacts for human review.

### Core Principle: Two-Layer Testing

Every editing test has two layers:

| Layer | What | When | Speed |
|-------|------|------|-------|
| **L1: Topology assertions** | Compare connectivity graph before/after operation | Always (CI) | Fast (<1ms) |
| **L2: Visual rendering** | Render before/after PNG with annotations for human review | `@pytest.mark.visual` or `--visual` flag | Slow (~100ms) |

L1 is deterministic and catches regressions automatically. L2 produces images for human review following the existing human-reviewed test policy (see `TEST_RESULTS.md`).

### Layer 1: Topology Snapshot + Diff

**Concept:** Capture the graph state (atoms, bonds, rings, neighbor lists) before and after an editing operation. Assert the diff matches expectations.

**Implementation:** `tests/helpers/topology_test.py`

```python
class TopologySnapshot:
    """Captures AtomicGraph state for before/after comparison."""
    def __init__(self, graph):
        self.atoms = {a._id: (a.ename, a.pos.copy(), a.subtype)
                      for a in graph.atoms.values() if a.alive}
        self.bonds = {frozenset((b.a._id, b.b._id)): b.order
                      for b in graph.bonds.values() if b.alive}
        self.rings = {r._id: [a._id for a in r.atoms if a.alive]
                      for r in graph.rings.values() if r.alive}
        self.neighbors = {a._id: [n._id for n in graph.neighbors(a)]
                          for a in graph.atoms.values() if a.alive}

    def diff(self, other):
        return TopologyDiff(
            added_atoms   = set(other.atoms) - set(self.atoms),
            removed_atoms = set(self.atoms) - set(other.atoms),
            added_bonds   = set(other.bonds) - set(self.bonds),
            removed_bonds = set(self.bonds) - set(other.bonds),
            added_rings   = set(other.rings) - set(self.rings),
            removed_rings = set(self.rings) - set(other.rings),
        )

class TopologyDiff:
    """Structured diff between two TopologySnapshots."""
    def __init__(self, added_atoms, removed_atoms, added_bonds, removed_bonds, added_rings, removed_rings):
        self.added_atoms   = added_atoms
        self.removed_atoms = removed_atoms
        self.added_bonds   = added_bonds
        self.removed_bonds = removed_bonds
        self.added_rings   = added_rings
        self.removed_rings = removed_rings

    def assert_empty(self, name=''):
        assert not self.added_atoms,   f'{name}: unexpected added atoms {self.added_atoms}'
        assert not self.removed_atoms, f'{name}: unexpected removed atoms {self.removed_atoms}'
        assert not self.added_bonds,   f'{name}: unexpected added bonds {self.added_bonds}'
        assert not self.removed_bonds, f'{name}: unexpected removed bonds {self.removed_bonds}'

    def assert_counts(self, name='', **expected):
        for key in ('added_atoms', 'removed_atoms', 'added_bonds', 'removed_bonds', 'added_rings', 'removed_rings'):
            if key in expected:
                actual = len(getattr(self, key))
                assert actual == expected[key], f'{name}: {key} expected {expected[key]}, got {actual}'
```

**Test pattern:**

```python
def test_add_ring():
    backend = KekuleBackend()
    snap_before = TopologySnapshot(backend.graph)

    backend.add_ring(0, 0)  # simulates "user clicks add ring button"

    snap_after = TopologySnapshot(backend.graph)
    diff = snap_before.diff(snap_after)

    diff.assert_counts('add_ring', added_atoms=6, added_bonds=6, added_rings=0)
    # Ring detected on next detect_geometry_rings() call, not immediately
```

### Layer 2: Visual Rendering (Headless)

**Concept:** Render the molecular graph to a static PNG using matplotlib (Agg backend, no display needed). Annotate with atom IDs, highlight selected/clicked atoms, show cursor position. Produce before/after side-by-side images for human review.

**Implementation:** Extend `tests/helpers/geometry.py` with `render_graph()` and `render_before_after()`.

```python
def render_graph(ax, graph, title='', proj='xy',
                 highlight_atoms=None,    # set of atom _ids to highlight (selection)
                 highlight_color='orange',
                 cursor_pos=None,         # (x,y) world coords — simulates mouse click
                 cursor_radius=0.5,
                 diff=None,               # TopologyDiff for coloring changes
                 show_atom_ids=True,
                 show_bond_labels=True):
    """Render AtomicGraph to matplotlib Axes with annotations.

    - Atoms: colored circles with 'id:element' labels
    - Bonds: lines with bond order / length labels
    - highlight_atoms: drawn with colored halo ring
    - cursor_pos: drawn as crosshair + circle (simulates click position)
    - diff: added atoms green, removed atoms red, added bonds blue, removed bonds dashed red
    """
    ...

def render_before_after(graph_before, graph_after, savepath,
                        title_before='Before', title_after='After',
                        highlight_atoms=None, cursor_pos=None, diff=None,
                        proj='xy'):
    """Save side-by-side before/after PNG for human review.

    Output: tests/visual_output/{test_name}.png
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    render_graph(ax1, graph_before, title_before, proj=proj,
                 highlight_atoms=highlight_atoms, cursor_pos=cursor_pos)
    render_graph(ax2, graph_after, title_after, proj=proj,
                 highlight_atoms=highlight_atoms, diff=diff)
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

### GUI Simulation Pattern

**Key insight:** We do NOT test the Qt/Vispy GUI widget itself. We test the *backend operations* that the GUI triggers. The GUI is a thin event handler calling backend methods.

| User action in GUI | Headless simulation |
|---|---|
| Click on atom at screen position (x,y) | `atom = graph.pick_atom(world_pos, radius=0.5)` |
| Click on bond | `bond = graph.pick_bond(world_pos, radius=0.5)` |
| Click on ring | `ring = graph.pick_ring(world_pos, radius=1.0)` |
| Select atoms (box/lasso) | `selection = {atom._id for atom in selected_atoms}` |
| Press "Add Ring" button | `backend.add_ring(q, r)` |
| Press "Insert Bridge" button | `insert_bridge(backend, atom_a, atom_b)` |
| Press "Collapse Bridge" button | `collapse_bridge(backend, atom_id)` |
| Type selection query | `compiled = compile_select_query(query); apply_select_query(graph, compiled)` |

**For visual rendering, we annotate what the user "did":**
- **Click operations:** draw cursor crosshair at the world-space position that was "clicked"
- **Selection operations:** draw colored halo around selected atom IDs
- **Before/after:** left panel = state before operation (with cursor/selection annotation), right panel = state after (with diff coloring)

### Complete Test Example

```python
import pytest
from tests.helpers.topology_test import TopologySnapshot, render_before_after

@pytest.mark.visual
def test_collapse_bridge(visual_output_dir):
    """Test: collapse a CH2 bridge group — remove C, reconnect two neighbors."""
    # ── Setup ──
    backend = KekuleBackend()
    backend.add_ring(0, 0)
    backend.add_ring(1, 0)
    backend.add_ring(2, 0)  # create a PAH-like structure

    # Find a bridge candidate (C with 2 heavy + 2 H neighbors)
    atom_list, *_ = backend.graph.to_arrays()
    bridge_atom = None
    for a in atom_list:
        heavy = sum(1 for n in backend.graph.neighbors(a) if n.ename != 'H')
        hyd   = sum(1 for n in backend.graph.neighbors(a) if n.ename == 'H')
        if heavy == 2 and hyd >= 2:
            bridge_atom = a
            break
    assert bridge_atom is not None, "No bridge candidate found"

    # ── Snapshot before ──
    snap_before = TopologySnapshot(backend.graph)

    # ── Simulate user operation (no GUI) ──
    # "User clicks on bridge_atom, then presses Collapse Bridge"
    collapse_bridge_at(backend, bridge_atom)

    # ── Snapshot after ──
    snap_after = TopologySnapshot(backend.graph)
    diff = snap_before.diff(snap_after)

    # ── L1: Topology assertions (always run) ──
    diff.assert_counts('collapse_bridge',
                       removed_atoms=1,    # the bridge C (and its H caps)
                       added_bonds=1,      # neighbors reconnected
                       removed_bonds=2)    # original C-neighbor bonds broken

    # ── L2: Visual rendering (only with --visual) ──
    if visual_output_dir:
        render_before_after(
            snap_before_graph, backend.graph,
            savepath=f'{visual_output_dir}/test_collapse_bridge.png',
            highlight_atoms={bridge_atom._id},  # show "clicked" atom
            cursor_pos=bridge_atom.pos[:2],     # show "click" position
            diff=diff,
            title_before='Before: PAH with bridge',
            title_after='After: bridge collapsed',
        )
```

### pytest Configuration

```ini
# pytest.ini additions
[pytest]
markers =
    gpu: requires OpenCL GPU
    visual: produces visual output for human review
    slow: long-running tests

# Conftest fixture for visual output directory
```

```python
# tests/conftest.py
@pytest.fixture
def visual_output_dir(request):
    """Returns output dir if --visual flag set, else None."""
    if request.config.getoption('--visual', default=False):
        outdir = 'tests/visual_output'
        os.makedirs(outdir, exist_ok=True)
        return outdir
    return None

def pytest_addoption(parser):
    parser.addoption('--visual', action='store_true', default=False,
                     help='Generate visual output images for human review')
```

### Output Organization

```
tests/
├── helpers/
│   ├── topology_test.py    # TopologySnapshot, TopologyDiff, render helpers
│   ├── geometry.py         # existing: plot_geometry, bond_lengths, etc.
│   └── parity.py           # existing: RMSE, correlation, plot_curves
├── visual_output/          # generated PNGs (gitignored or kept for review)
│   ├── test_add_ring.png
│   ├── test_collapse_bridge.png
│   └── ...
├── test_topology_add_ring.py
├── test_topology_bridge_ops.py
├── test_topology_selection.py
└── ...
```

### What to Test (Priority Order)

1. **`KekuleBackend.add_ring`** — add hex ring, verify 6 atoms + 6 bonds appear, H caps adjusted
2. **`KekuleBackend.remove_ring`** — remove ring, verify atoms/bonds removed, H caps re-adjusted
3. **Bridge collapse** — remove CH2, reconnect neighbors, verify bond count
4. **Bridge insert** — add CH2 between two heavy atoms, verify new atom + 2 new bonds
5. **H cap adjustment** — after topology change, verify H count matches valence
6. **Selection query** — compile query, apply to graph, verify selected atom set
7. **Bond order assignment** — after KekuleSolver, verify bond orders sum to valence
8. **Ring detection** — `detect_rings()`, verify ring count and membership
9. **Pick operations** — `pick_atom`, `pick_bond`, `pick_ring` at known positions
10. **Multi-step editing** — sequence of operations, verify final topology

### Why This Approach Is Robust & Frictionless

- **No display required:** matplotlib Agg backend renders to PNG without X server. Tests run in CI, Docker, SSH.
- **No GUI framework dependency:** tests exercise backend logic directly, not Qt/Vispy widgets. GUI bugs are separate from logic bugs.
- **Stable IDs:** `AtomicGraph` uses object identity (`_id`), not array indices. Atoms don't renumber after deletion. Snapshots are reliable.
- **Fast by default:** L1 topology assertions run in <1ms. L2 visual rendering only when `--visual` flag is passed.
- **Human review built-in:** L2 produces the exact before/after images a human reviewer needs, with annotations showing what was "clicked" and what changed.
- **Extends existing infrastructure:** uses existing `plot_geometry` from `geometry.py`, existing `@pytest.mark.visual` from `pytest.ini`, existing human-review policy from `TEST_RESULTS.md`.
- **Diff-based assertions:** `TopologyDiff` makes it trivial to assert "exactly 1 atom added, 2 bonds removed" without hardcoding atom IDs.
