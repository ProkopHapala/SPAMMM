---
type: Task
title: <Task title> — master orchestration
tags: [parallel-agents, task-master, <domain>]
---

# Task master: <Task title>

- **Status:** planning
- **Task prefix:** `<TaskPrefix>`
- **Grouping:** `<common-prefix | dedicated-subfolder>`
- **Coordinator:** <owner>
- **Contract version:** 1
- **Baseline:** <commit/config>

## Agent dispatch checklist — copy/paste assignments

The coordinator checks a box only after accepting the handoff; workers do not edit
this checklist.

1. [ ] **Agent_1 — <role>:** Read this master and [`<worker task file>`](<worker-relative-path>); you are Agent_1. Execute only Agent_1's worker task, write only its owned files/artifacts, and do not perform another agent's work.
2. [ ] **Agent_2 — <role>:** Read this master and [`<worker task file>`](<worker-relative-path>); you are Agent_2. Execute only Agent_2's worker task, write only its owned files/artifacts, and do not perform another agent's work.

## Aggregate objective and acceptance

<End-to-end result, invariants, regression command, human review, non-goals.>

## Master authority

This file is the SSOT for contracts, ownership, dependencies, integration, and
status. Worker files may specialize their assigned scope but cannot override it.
Contradictions or required contract changes stop affected work and return here.

## Frozen compatibility contract

| Producer → consumer | Interface/output | Shape/order/units | Validation/error semantics |
|---|---|---|---|
| Agent_1 → Agent_3 | <API/artifact> | <exact contract> | <exact checks> |

- **Common inputs/seeds:** <paths, versions, checksums>
- **Tolerances:** <metrics and thresholds>
- **Artifacts:** `<root>/agent_<N>_<role>/...`
- **Exclusive resources:** <serialized GPU/benchmark/reference schedule>

## Worker index and ownership

| Agent | Task file | Owned files | Read-only/forbidden | Depends on |
|---|---|---|---|---|
| `Agent_1` `<role>` | [`<worker task file>`](<worker-relative-path>) | <exact files> | <paths/actions> | — |
| `Agent_2` `<role>` | [`<worker task file>`](<worker-relative-path>) | <exact files> | <paths/actions> | Agent_1 gate |

One writer per file. Shared entry points, schemas, task documents, and integration
files are coordinator-owned unless listed otherwise.

## Execution waves and gates

1. **Wave 1:** <independent agents>. Gate: <required evidence>.
2. **Wave 2:** <dependent agents>. Gate: <required evidence>.
3. **Integration:** coordinator only; <order and cross-component tests>.

Parallel preparation does not waive gates. Workers must rerun against the accepted
upstream contract version before handoff.

## Global non-interference contract

- Use separate branches/worktrees when available; never merge/rebase/reset/revert
  another worker's work.
- Write only owned files and the assigned artifact directory.
- Do not alter baselines, seeds, tolerances, schemas, shared fixtures, or master status.
- Treat production code as read-only during diagnosis unless exact ownership is given.
- Stop and report overlap, dirty-file conflict, or missing authority.

## Required handoff

Each worker returns: contract version, changed-file list, exact commands, test results,
artifact/REVIEW paths, produced interfaces, consumer notes, assumptions, and unresolved
risks. Workers do not claim aggregate completion.

## Coordinator integration and acceptance

1. Validate handoffs and compatibility contracts.
2. Integrate in dependency order; resolve shared-file edits centrally.
3. Run <unit/parity/integration/end-to-end tests> from a clean aggregate state.
4. Review artifacts and present evidence to USER.
5. Update status only after required confirmation.

## Coordinator-only ledger

| Agent | State | Contract version | Handoff/evidence | Integrated commit |
|---|---|---:|---|---|
| Agent_1 | planned | 1 | — | — |
| Agent_2 | planned | 1 | — | — |
