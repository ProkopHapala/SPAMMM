# TASKS.md — Top-Level Task Index

Active tasks for the SPAMMM project. Each task links to a detailed spec in `doc/Tasks/`.
Tasks are designed for **agentic execution**: a coordinator agent distributes tasks to sub-agents,
who implement, test, and report back.

---

## Active Tasks

| ID | Task | Priority | Status | Spec |
|----|------|----------|--------|------|
| T01 | AFM FDBM pipeline < 1s end-to-end | High | **Implementation done** (R1+R2; benzene warm ~0.2 s; flat_1 S3+S4 ~1.4 s) | [`doc/Tasks/PerfBenchmark_FDBM.md`](Tasks/PerfBenchmark_FDBM.md) |
| T02 | UFF/SPFF/LFF relaxation speedup (GUI) | High | **In progress** — fused UFF+FAF + LFF bring-up; GUI callback / combo open | [`doc/Tasks/PerfBenchmark_Relaxation.md`](Tasks/PerfBenchmark_Relaxation.md) |
| T03 | Fragment/Group library + substitution | Medium | Design done | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §11 |
| T04 | Molecular browser plugin port | Medium | Design done | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §1 |
| T05 | GridFF consolidation (B-spline vs trilinear) | Medium | Design done | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §6 |
| T06 | GUI verbosity consolidation | Low | Design done | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §10 |
| T07 | SMILES builder | Low | Not started | [`doc/ARCHITECTURE_ROADMAP.md`](ARCHITECTURE_ROADMAP.md) §9 |

---

## T01 note — FDBM perf (2026-07-19)

**Spec:** `doc/Tasks/PerfBenchmark_FDBM.md`. Switch: `SPAMMM_AFM_FAST_S3=1` (default) / `=0` legacy.

| Milestone | Before → After | Mechanism |
|-----------|----------------|-----------|
| Benzene warm GUI-like | ~1.65 s → ~**0.18–0.55 s** | GPU tasks + gpyFFT + fused S3 |
| Flat_1 S2 `rho_na` | 5.87 s → **0.03 s** | dense NA DM |
| Flat_1 S3 cache write | ~10 s → **~0.4 s** | uncompressed `np.savez` |
| Benzene S3 fields | 0.26 s legacy → **0.07 s** fast | fused ES + GPU pad/scale |
| Flat_1 warm S3+S4 (NO_IO) | “stuck” / many s → **~1.4 s** | R1+R2 |

**Remaining (not blocking):** async cache skip for interactive GUI; optional `step=0.15` interactive (not quality default). Full write-up in PerfBenchmark_FDBM.md.

---

## T02 note — Relax perf / LFF (2026-07-19)

**Spec:** `doc/Tasks/PerfBenchmark_Relaxation.md`. Topic: `doc/Topics/ForceFields/LFF_ProjectiveRelax.md`.

| Milestone | Status | Notes |
|-----------|--------|-------|
| SPFF fused serial (flat_1) | Measured | ~0.005 s / 2000 steps vacuum |
| UFF fused + dih/inv + FAF | Implemented | PTCDA force parity vs multi-kernel; physics vs SPFF **unverified** |
| LFF projective + FAF | Implemented | PTCDA: 50×16 ≈ 0.004 s, dOCdz≈SPFF; **unverified** pending USER review |
| GUI callback / UFF+LFF combo | Open | Main remaining T02 path to “instant Relax” button |

Do **not** mark T02 Complete without USER confirmation of plots under `debug/test_relax_ptcda_faf/`.

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
