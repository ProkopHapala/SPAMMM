---
type: Task
title: <Task title> — Agent_<NN> <role>
tags: [parallel-agents, worker-task, <domain>]
---

# Agent_<NN>: <role>

- **Master:** [`<master task file>`](<master-relative-path>)
- **Agent ID:** `Agent_<NN>`
- **Required contract version:** 1
- **Status authority:** coordinator only

Read the master completely before working. The master overrides this file. Execute
only this packet; other worker scopes are context, not optional work.

## Goal and boundary

- **Goal:** <one independently verifiable outcome>
- **In scope:** <specific work>
- **Out of scope:** <neighboring work owned elsewhere>

## Inputs and preconditions

- Frozen inputs/fixtures: <paths/checksums>
- Upstream gate: <evidence that must exist before verdict/run>
- Interfaces consumed: <exact master contract rows>

## Exclusive ownership

- **May write:** <exact files and artifact directory>
- **Read-only:** <shared/production/task files>
- **Must not:** change master contracts/tolerances/status; edit another worker's
  files; merge, rebase, reset, revert, stage, delete, or overwrite another worker's
  work.

If any required edit falls outside ownership, stop and propose it to the coordinator.

## Work and verification

1. <implementation/debug step>
2. <isolating/local verification>
3. <produce contract-compatible output>

**Commands:**

```bash
<exact reproducible command>
```

**Expected deliverables:** <files, API, metrics, artifacts, acceptance for this slice>

## Handoff to coordinator/consumer

Return:

1. Contract version and baseline used.
2. Changed files and concise rationale.
3. Exact commands plus full pass/fail results.
4. Artifact and `REVIEW:` paths.
5. Produced interface/schema and downstream usage notes.
6. Worst discrepancy, assumptions, unresolved risks, and requested coordinator edits.

Do not mark the aggregate task fixed/resolved/done.
