# Task: pip-installable SPAMMM packaging (P1)

**Status:** investigating  
**Priority:** P1 (nc-AFM week — `doc/ARCHITECTURE_ROADMAP.md` §TOC)  
**Parent:** conference packaging, not a physics import

## Objective

Make the repo installable with:

```bash
pip install -e .
# or: pip install .
```

so collaborators / conference demos do not depend on ad-hoc `PYTHONPATH` or running only from the repo root.

## Current gap

- No `pyproject.toml` / `setup.py` (`FeatureChecklist.md`, `doc/ToDo/Features.audit.md`).
- OpenCL kernels live in top-level `kernels/` — must remain findable after install (package data or env/`spammm` resource path).
- Element types, DFTB WFC, fits under `data/` / `spammm/quantum/DFTB/data/` — same packaging concern.
- Typical today: develop from clone with cwd on `PYTHONPATH`.

## Deliverables

1. **`pyproject.toml`** (preferred over legacy `setup.py`) — name `spammm`, Python ≥3.10 (or whatever repo already assumes), deps: `numpy`, `scipy`, `pyopencl`, `matplotlib`, GUI extras optional (`PyQt5`/`vispy` as `[gui]` extra).
2. **Package data** — ship or locate `kernels/*.cl` and required `data/` files; fail loud if missing (no silent wrong paths).
3. **Entry points (optional P1)** — e.g. `spammm-gui = spammm.GUI.SPAMMM_GUI:main` if a clean `main` exists; otherwise document `python -m spammm.GUI...`.
4. **README install blurb** — clone → `pip install -e ".[gui]"` → one smoke command.
5. **Smoke check** — clean venv: `import spammm`; load one kernel via `OpenCLBase`; run one fast L0 test.

## Design notes / ask USER if ambiguous

- Keep `kernels/` at repo root vs move under `spammm/kernels/` — moving is cleaner for packaging but touches many paths; prefer **package-data from repo-root `kernels/`** first unless USER wants relocation.
- DFTB+ / pySCF remain **optional** system deps (not forced in core `install_requires`).

## Acceptance

- [ ] `pip install -e .` in a fresh venv succeeds
- [ ] `import spammm` and kernel compile path work without manual `PYTHONPATH`
- [ ] Short install section in README
- [ ] USER confirms before Done

## Out of scope

- Publishing to PyPI (can be later).
- Bundling NVIDIA drivers / OpenCL ICD.
- Cosserat / Dyson / FF-fit driver work.
