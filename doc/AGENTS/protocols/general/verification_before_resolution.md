---
description: Mandatory verification protocol before marking any bug, task, or issue as resolved. Prevents the pathological behavior of claiming fixes without proof.
---

# Verification Before Resolution Protocol

## Purpose

Prevent the critical and recurring failure pattern where an agent:
1. Makes a code change based on a hypothesis
2. Marks the issue as "fixed" / "resolved" / "done" **without running any verification**
3. The fix turns out to be wrong, useless, or even harmful

This is the **most important protocol** in SPAMMM. Violating it erodes user trust and wastes time.

---

## The Rule

**NEVER mark an issue as "fixed", "resolved", "done", or similar without ALL of:**

1. **Run a verification** — a test, script, or interactive run that demonstrates the fix works
2. **Show the result to the USER** — output, screenshot, plot, or log
3. **Wait for USER confirmation** — the USER must explicitly confirm the issue is resolved

Until all three steps are complete, the status must remain:
- `"investigating"` — if still diagnosing
- `"unverified"` — if a change was made but not yet confirmed by the USER

---

## What Counts as Verification

| Type | Valid? | Example |
|------|--------|---------|
| Automated test passing | Yes | `pytest tests/test_fdbm.py -k symmetry --develop` |
| Script with numerical output | Yes | Run a script, show df symmetry metric |
| Visual output (plot/screenshot) | Yes | Generate `.png`, USER reviews it |
| Interactive GUI run | Yes | USER runs GUI, confirms behavior |
| **Code inspection only** | **NO** | "I changed the code, it looks correct" |
| **Reasoning about the fix** | **NO** | "The logic is now correct, so it should work" |
| **Test run without showing output** | **NO** | Must show output to USER |

---

## Common Violations (Forbidden)

1. **Hypothesis-as-fix:** "I think the bug is X, so I changed Y, marking as fixed." — No verification was run.
2. **Shallow test:** Running a test that doesn't actually test the bug (e.g., import test when the bug is in numerical output).
3. **Self-confirmation:** Agent claims the fix works based on its own analysis, without USER seeing the result.
4. **Premature doc update:** Writing "FIXED" in task docs before USER has seen the result.
5. **Reverting then claiming fixed:** Reverting a change and claiming the revert fixes the issue.

---

## Workflow

```
Bug reported
    |
    v
Investigate root cause (debug prints, tests, diagnostics)
    |
    v
Form hypothesis about root cause
    |
    v
Implement fix
    |
    v
Run verification (test/script/interactive)
    |
    v
Show output to USER (REVIEW: path/to/artifact)
    |
    v
USER confirms or rejects
    |
    v
Only NOW: update status to "fixed" in docs
```

If the USER rejects: go back to investigation. Do NOT mark as fixed.

---

## Applicable Locations

This protocol applies to ALL status tracking:
- `doc/Tasks/*.md` — bug reports, task documents
- `doc/ToDo/*.md` — todo lists
- `AGENTS.md` — any status references
- `FeatureChecklist.md` — feature status
- `TEST_RESULTS.md` — test results
- Git commit messages (do not claim "fixes #X" without verification)
- Chat responses to the USER (do not say "fixed" without confirmation)
