# TASKS.md — Top-Level Task Index

Active tasks for the SPAMMM project. Each task links to a detailed spec in `doc/Tasks/`.
Tasks are designed for **agentic execution**: a coordinator agent distributes tasks to sub-agents,
who implement, test, and report back.

---

## Active Tasks

| ID | Task | Priority | Status | Spec |
|----|------|----------|--------|------|
| T01 | AFM FDBM pipeline < 1s end-to-end | High | Not started | [`doc/Tasks/PerfBenchmark_FDBM.md`](Tasks/PerfBenchmark_FDBM.md) |
| T02 | UFF/SPFF relaxation speedup (GUI) | High | Not started | [`doc/Tasks/PerfBenchmark_Relaxation.md`](Tasks/PerfBenchmark_Relaxation.md) |
| T03 | Fragment/Group library + substitution | Medium | Design done | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §11 |
| T04 | Molecular browser plugin port | Medium | Design done | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §1 |
| T05 | GridFF consolidation (B-spline vs trilinear) | Medium | Design done | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §6 |
| T06 | GUI verbosity consolidation | Low | Design done | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §10 |
| T07 | SMILES builder | Low | Not started | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §9 |

---

## Agentic Task Loop Policy

### Roles

- **Coordinator**: Reads TASKS.md, assigns task to sub-agent, reviews results, updates status.
- **Sub-agent**: Implements task, runs tests, reports results.

### Workflow per task

```
1. Coordinator assigns task: "Implement T01 — see doc/Tasks/PerfBenchmark_FDBM.md"
2. Sub-agent reads spec, explores codebase, implements changes
3. Sub-agent runs tests:
   - pytest -m "not slow and not gpu" (L0 regression — must pass)
   - pytest <relevant_test> --develop -s (L1+L2 — agent reads output, reviews plots)
   - New benchmark script (if specified in task)
4. Sub-agent reports:
   - What changed (files, lines)
   - Test results (pass/fail counts, timing numbers)
   - Artifacts produced (plots, logs)
   - Any issues encountered
5. Coordinator reviews:
   - Verifies tests pass
   - Checks timing targets met
   - Updates TASKS.md status
   - Updates Features.audit.md / ToDo.agents.md if feature complete
```

### Rules

1. **One task per sub-agent** — no parallel tasks touching same files.
2. **Run tests after every change** — no untested commits.
3. **Report timing numbers** — perf tasks must include before/after measurements.
4. **Update docs when done** — mark task complete in TASKS.md, update audit docs.
5. **Never delete tests** — only add or strengthen.
6. **Fail loud** — if something breaks, report it, don't mask it.

### Task status values

- **Not started**: Spec exists, no implementation.
- **In progress**: Sub-agent working on it.
- **Implementation done**: Code written, tests pass, awaiting coordinator review.
- **Complete**: Reviewed, docs updated, merged.
- **Blocked**: Waiting on dependency or external input.

---

## Task Dependencies

```
T01 (FDBM perf) ── no deps, standalone
T02 (Relax perf) ── no deps, standalone
T03 (Fragment lib) ── depends on ring placement (done), corner detection (done)
T04 (Browser plugins) ── depends on MolecularBrowserVispy (exists, Phase 1)
T05 (GridFF) ── no deps, but complex (kernel refactoring)
T06 (Verbosity) ── no deps, mechanical replacement
T07 (SMILES) ── no deps, but low priority
```

T01 and T02 are **independent and highest priority** — can be assigned in parallel.
