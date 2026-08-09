---
name: plan-parallel-agent-work
description: Design and scaffold manual multi-agent task plans for debugging, testing, refactoring, or feature implementation. Use when work must be split among independently launched agents, when the environment has no automatic orchestration harness, or when creating copy/paste Agent_N assignment checklists, a single shared agent plan, separate worker task files, a task_master document, ownership boundaries, dependency gates, or integration contracts.
---

# Plan Parallel Agent Work

Create design-spec task documents that let independently launched agents produce
compatible results without editing each other's work. Plan only unless the user also
asks to execute the task.

## Choose the layout

| Layout | Use when | Output |
|---|---|---|
| **Short** | 2–4 agents, one phase, compact scope, shared context is useful | One task `.md`; every agent reads all packets but executes only its assigned ID |
| **Long** | Multi-phase or multi-day work, many files/interfaces, staged dependencies, or >4 agents | One authoritative master `.md` plus one task `.md` per agent |

Use the shortest layout that keeps ownership and interfaces unambiguous. Read and
adapt only the templates required by the selected layout:

- Short: `assets/short-task-template.md`
- Long: `assets/long-task-master-template.md` and
  `assets/long-agent-task-template.md`

## Require a dispatch checklist — MANDATORY, near the top

**This is mandatory.** Every shared plan/master `.md` file MUST contain a section named
`Agent dispatch checklist — copy/paste assignments` placed **near the top** of the document
(immediately after the Problem/Summary section, BEFORE any detailed analysis or packet
sections). If the checklist is missing or buried deep in the document, the plan is incomplete.

The checklist must be a numbered checkbox list using stable IDs and must cover **ALL agents
across ALL waves** (not just the first wave). **When work is split into waves, each wave
must be a separate sub-heading** so the wave structure is visually clear at a glance:

```text
## Agent dispatch checklist — copy/paste assignments

**Standard instructions for every agent (do not remove):**
- Read this document completely before starting. You are assigned the agent ID in your checkbox.
- Execute only your assigned packet in your assigned wave. Do not interfere with other agents' work.
- Write only to your owned files. Do not edit files listed as read-only or forbidden.
- When finished: (1) check your checkbox `[ ]` → `[x]`, (2) write a brief report (what you did,
  test results, artifact paths, open questions) at the bottom of this file under `## Agent reports`,
  (3) list any contract changes that downstream agents need to know.
- Do not mark the overall task as done. Only the coordinator accepts handoffs and marks waves complete.

### Wave 1 — Parallel (launch simultaneously)

1. [ ] Agent_1 — <role>: <owned files>. Do not <neighbor work>.
2. [ ] Agent_2 — <role>: <owned files>. Do not <neighbor work>.

### Wave 2 — Parallel (after Wave 1 handoff accepted)

3. [ ] Agent_1 — <role>: Depends on Wave 1 handoff. <owned files>.
4. [ ] Agent_2 — <role>: Depends on Wave 1 handoff. <owned files>.

### Wave 3 — Serial, coordinator-only (after Wave 2 accepted)

5. [ ] Coordinator — <role>: After Wave 2 accepted. <steps>.
```

The user's per-agent launch prompt then reduces to a one-liner:

```
you are Agent_1 in @[doc/Tasks/<TaskFile>.md] do your part of WAVE 1
```

The standard instructions in the document handle the rest (read, don't interfere, check box,
write report). The user does not need to repeat these instructions per agent.

If there is only one wave, a single flat list is fine (no wave sub-headings needed).

Every plan/master must also contain an `## Agent reports` section near the bottom where
agents write their completion reports:

```text
## Agent reports

<!-- Agents: write your report here after finishing your work. Format:
### Agent_N (Wave M) — <role>
- **What I did**: ...
- **Files changed**: ...
- **Test results**: ...
- **Artifacts**: ...
- **Open questions / contract changes**: ...
-->
```

Make every line self-contained enough to copy as a manual launch prompt. Include the wave
number and dependency note for downstream agents. Link the worker file in long layout. The
coordinator checks a box only after accepting the handoff; workers must not check their own
box. Use `Agent_1`, `Agent_2`, ... in the checklist, packet headings, ownership tables,
artifact paths, and dispatch prompts.

## Plan before splitting

1. Inventory existing code, tests, task documents, SSOTs, and dirty files.
2. Define the aggregate result and its end-to-end acceptance test before agent roles.
3. Identify shared interfaces: APIs, array shapes/order, units, schemas, filenames,
   coordinate frames, tolerances, seeds, baseline commit/config, and artifact layout.
4. Split at independently verifiable seams. Prefer component or evidence ownership,
   not arbitrary line ranges.
5. Mark true dependencies as serial gates. Parallel preparation is not permission to
   publish conclusions before upstream contracts pass.
6. Assign exactly one writer to every file and one owner to every shared decision.
   If two agents must edit the same file, make those edits serial or coordinator-only.

## Decomposition patterns

### Debugging/testing

Separate evidence channels so agents do not race toward competing fixes:

- contract/input agent: reproduce, freeze inputs, verify frames/layouts/invariants;
- reference agent: establish trusted reference and error budget;
- component agents: isolate independent hypotheses or subsystems;
- integration agent/coordinator: combine evidence, run end-to-end reproduction, and
  decide what is supported, ruled out, or still ambiguous.

Agents must not change production behavior while the task is diagnostic unless an
explicit worker packet authorizes an exact file and change. A test failure is
evidence, not permission for an unplanned fix.

### Feature implementation

Freeze public contracts before parallel implementation:

- coordinator/API agent: interfaces, shared types, schemas, fixtures, acceptance;
- component agents: non-overlapping modules behind the frozen interface;
- verification agent: tests/reference/invariants without reimplementing components;
- integration agent/coordinator: shared entry points, migration, end-to-end tests,
  docs, and conflict resolution.

Do not parallelize coupled components until their interface is explicit enough that
each can be tested with a stub or fixture.

## Mandatory orchestration contract

Every plan must state:

- **Authority:** master/shared contract overrides worker notes; contradictions stop
  work and return to the coordinator.
- **Frozen baseline:** commit/config, inputs, seeds, units, query/test set, expected
  output schema, tolerances, and contract version.
- **Ownership:** files each agent may write, files it may only read, and forbidden
  paths/actions. One writer per file is absolute.
- **Isolation:** separate branch/worktree when available and separate artifact/output
  directory per agent. No agent merges, rebases, resets, stages, reverts, or deletes
  another agent's work.
- **Shared resources:** serialize GPU jobs, benchmarks, reference regeneration,
  database migrations, and other exclusive or stateful resources.
- **Dependencies:** waves/gates and the evidence required to unlock downstream work.
- **Compatibility:** exact producer→consumer contract for APIs, formats, shapes,
  ordering, units, error semantics, and filenames.
- **Handoff:** changed files, exact commands, test results, artifact paths, open
  questions, assumptions, and integration notes.
- **Integration:** coordinator-owned order, conflict policy, cross-component tests,
  and final acceptance. Workers do not claim the aggregate task is done.
- **Status:** only the coordinator updates the master ledger. In SPAMMM, do not mark
  fixed/resolved/done before verification is shown and the USER confirms.

If a shared contract must change, stop affected agents, update the master, increment
`contract_version`, list invalidated outputs, and redispatch. Never let worker files
silently redefine common contracts.

## Short layout

Create one file such as:

```text
doc/Tasks/<TaskPrefix>_ParallelPlan.md
```

The file contains the common contract, ownership matrix, waves, agent packets,
integration plan, and coordinator-only ledger. Give each manual agent the same file
and an explicit ID:

```text
Read <file> completely. You are Agent_2. Execute only packet Agent_2. Treat all other
packets and coordinator sections as read-only context.
```

Each packet must include goal, inputs, owned files, read-only files, forbidden work,
steps, outputs, local verification, dependency gate, and handoff format.

## Long layout

Choose exactly one grouping convention; never mix them.

**Common filename prefix:**

```text
doc/Tasks/<TaskPrefix>_task_master.md
doc/Tasks/<TaskPrefix>_agent01_<role>.md
doc/Tasks/<TaskPrefix>_agent02_<role>.md
```

**Dedicated subfolder:**

```text
doc/Tasks/<TaskPrefix>/task_master.md
doc/Tasks/<TaskPrefix>/agent01_<role>.md
doc/Tasks/<TaskPrefix>/agent02_<role>.md
```

Use one stable `<TaskPrefix>` derived from the task name. The master is the SSOT for
contracts, ownership, dependencies, integration, and status. Each worker reads the
master plus its own file and may not edit either unless its packet explicitly assigns
that file. Worker files specialize scope but cannot override the master.

Manual dispatch prompt:

```text
Read <task_master.md> and <your_agent_file.md> completely. You are Agent_<NN>. Obey the
master contract, modify only your owned files, write only to your artifact directory,
and return the specified handoff. Do not perform another agent's work.
```

## Review checklist

- Aggregate acceptance and the integration owner are explicit.
- Every deliverable has one owner and every owned file has one writer.
- Agent scopes are independently testable and do not duplicate implementation.
- Shared interfaces include shapes/order/units/error behavior, not just names.
- Worker outputs are compatible by construction and have a consumer.
- Dependencies are gates; expensive/exclusive resources are scheduled.
- Filenames use one prefix or one task subfolder and sort together visually.
- A top-level numbered `[ ] Agent_N` dispatch checklist is easy to find and copy.
- Artifact paths cannot overwrite each other.
- Handoffs contain reproducible commands and evidence.
- No worker can alter master status, acceptance thresholds, or shared contracts.
