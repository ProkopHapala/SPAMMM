# debug/

Ephemeral test and diagnostic artifacts. **Gitignored** (except this file).

## Layout

```
debug/<script_stem>/          # matches pytest module name without .py
  <test_func>.out             # L1 curated evaluation packet (agent reads first)
  <test_func>.log             # verbose execution trace
  <test_func>.png             # L2 visual (--visual or --develop)
  <test_func>.mol2            # optional geometry export
  *.xyz                         # testplot_assembly: per-rank structures + clash column
  *.diag                        # testplot_assembly: top clash atom table
```

## For agents and developers

- `.gitignore` keeps artifacts out of git; files remain on disk after a test run.
- Read artifacts by **explicit path** printed as `REVIEW: debug/...` in test stdout.
- Do **not** add `debug/` to `.cursorignore`.
- In IDE: set `"explorer.excludeGitIgnore": false` to browse this folder in the sidebar.

## Modes

| Mode | Command | Artifacts |
|------|---------|-----------|
| Routine | `pytest -m "not slow"` | L0 asserts only |
| Develop | `pytest path --develop -s` | L0 + `.out`/`.log` + PNG |

## Presentation gallery

Open **`debug/presentation.html`** in a browser (from this folder) for clickable PTCDA FDBM / SA-prolonged figures. Index: repo-root `FOR_PRESENTATION.md`.

See `doc/TEST_DESIGN.md`.
