---
type: Task
title: <Task title> — parallel agent plan
tags: [parallel-agents, <domain>]
---

# Task: <Task title>

- **Status:** planning
- **Layout:** short / single-file
- **Coordinator:** <owner>
- **Contract version:** 1
- **Baseline:** <commit/config>

Every agent reads this whole document for context, but executes only the packet
matching its assigned Agent ID. All other packets and coordinator sections are
read-only.

## Agent dispatch checklist — copy/paste assignments

The coordinator checks a box only after accepting the handoff; workers do not edit
this checklist.

1. [ ] **Agent_1 — <role>:** Read this file; you are Agent_1. Execute only `## Agent_1 — <role>`. Write only <owned files/artifacts>. Do not edit coordinator sections, shared contracts, or another agent's work.
2. [ ] **Agent_2 — <role>:** Read this file; you are Agent_2. Execute only `## Agent_2 — <role>`. Write only <owned files/artifacts>. Do not edit coordinator sections, shared contracts, or another agent's work.

## Aggregate objective and acceptance

<One result all agents jointly produce. State the end-to-end test and USER review.>

**Out of scope:** <explicit exclusions>

## Frozen common contract

| Item | Contract |
|---|---|
| Inputs/seeds | <paths, versions, seeds> |
| API/data | <signatures, shapes, order, units, error semantics> |
| Tolerances | <metrics and thresholds> |
| Artifacts | `<root>/<agent_id>/...` |
| Shared resources | <GPU/benchmark/reference schedule> |

This section is authoritative. Agents stop and report contradictions; only the
coordinator may change it and increment `contract_version`.

## Ownership and waves

| Agent | Owns/writes | Read-only | Must not do | Gate |
|---|---|---|---|---|
| `Agent_1` `<role>` | <files/artifacts> | <shared files> | <forbidden work> | Wave 1 |
| `Agent_2` `<role>` | <files/artifacts> | <shared files> | <forbidden work> | After `<gate>` |

One writer per file. Agents do not merge, reset, revert, stage, delete, or overwrite
another agent's work. Serialize stateful/GPU/performance runs.

## Agent_1 — <role>

- **Goal:** <single independently verifiable outcome>
- **Inputs:** <frozen inputs>
- **Owned files/artifacts:** <exact paths>
- **Read-only/forbidden:** <exact boundaries>

**Steps:**

1. <step>
2. <step>

- **Deliverables and local verification:** <outputs plus exact commands>
- **Handoff:** changed files, commands, results, REVIEW paths, assumptions, open issues.

## Agent_2 — <role>

<Repeat the same packet fields. Do not replace them with a vague work-plan bullet.>

## Coordinator integration

1. Verify each gate and reject incompatible/stale handoffs.
2. Integrate in order: <Agent_1 → Agent_2 → ...>.
3. Run <cross-component/end-to-end commands>.
4. Show evidence to USER; only then update aggregate status.

## Coordinator-only ledger

| Agent | State (`planned/in_progress/ready/integrated/rejected`) | Contract version | Evidence |
|---|---|---:|---|
| Agent_1 | planned | 1 | — |
| Agent_2 | planned | 1 | — |
