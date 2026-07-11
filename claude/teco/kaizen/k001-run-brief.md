# K-001 live run — teco end-to-end spin on falkor-chat M3 (slice 1)

> Prepared 2026-07-09 while low on session credits. **Resume protocol:** after the credit
> reset, open a session in the repo root (or `falkor-chat/`) and launch the `teco` subagent
> with the brief in the "Launch brief" section below, verbatim. Then observe against the
> checklist at the bottom. This file is the single source of truth for the run.

## Why this run

- `claude/teco/kaizen/plan.md` **K-001** (high): validate nested delegation end-to-end
  (teco → architect → implementer → QA) — depth, context-passing fidelity, result quality.
  Pairs with architect K-002 and coder K-002 (handoff validation).
- `falkor-chat/docs/BACKLOG.md`: M2 complete; **next milestone M3 — Workflow engine**, no
  kaizen items drafted yet → decomposition is genuinely needed, which is teco's job.

## Assignment scope (recommended: slice 1, not all of M3)

Full M3 (DESIGN §12.4): definition model in `reference` → snapshot materialization →
run/step-run executor → chat linkage, proven by one conversational flow + one
business-process flow. That's several waves of work — too much for one observed run.

**Slice 1 (this run):** teco first has the architect decompose all of M3 into falkor-chat
kaizen items (K-020+), then **executes only the first slice**: workflow **definition model
in the `reference` graph + snapshot materialization into a workspace graph** (publish path).
Executor/chat-linkage waves are follow-ups. To run full M3 instead, drop the "execute only
slice 1" constraint in the brief.

## Launch brief (paste to teco verbatim)

---

You are coordinating the start of milestone **M3 — Workflow engine** for the `falkor-chat`
component at `falkor-chat/` in this repo. This brief is your entire input; read the
referenced docs yourself.

**Context to read first:**
- `falkor-chat/AGENTS.md` (component rules; FalkorDB OpenCypher, not Neo4j)
- `falkor-chat/docs/DESIGN.md` — §12.4 (M3 roadmap: definition model in `reference`,
  snapshot materialization, run/step-run executor, chat linkage; proof = one conversational
  + one business-process flow), §3 rule 3 (edges cannot cross graphs — the reason defs are
  materialized), the §4/§6 materialize-on-publish decision (immutable versioned snapshots),
  §6.3 (coordination is workflow, not a separate primitive), §13 (open question: workflow
  guard expression language — expr lib vs minimal DSL in `Step.config` — "decided with the
  engine")
- `falkor-chat/docs/QUERIES.md` (canonical query library) and `falkor-chat/docs/BACKLOG.md`
  (conventions: kaizen item format, graph-dba-gate → tdd-engineer pattern used by K-002/
  K-007/K-008/K-013, plan docs at `falkor-chat/docs/plans/`)

**Your assignment, in order:**
1. Delegate to **architect**: a decomposition of all of M3 into sequenced falkor-chat kaizen
   items (numbered K-020 onward, matching the existing item format: owner, inputs/prereqs,
   scope, done-condition, risks/RAM, test strategy), plus an implementation plan for slice 1
   only. Plan doc at `falkor-chat/docs/plans/m3-workflow-engine.md`; hand it onward **by
   path**, never paraphrased.
2. Execute **only slice 1**: the workflow **definition model in the `reference` graph** and
   **snapshot materialization into a workspace graph on publish** (immutable, versioned).
   Expected shape per component convention: **graph-dba** gate (model + verified Cypher +
   `test_queries.sh` assertions, raising the 149 baseline with the new count enumerated) →
   **tdd-engineer** implementation (repository/service layer per the plan). The run/step-run
   executor and chat linkage are later slices — do not build them.
3. Route acceptance-level verification to **qa-engineer** if slice 1 warrants it; at minimum
   verify both suites green yourself: `cd falkor-chat && ./scripts/test_queries.sh` (baseline
   149/149 before your changes) and `cd falkor-chat/server && pytest` (baseline 156).
4. Keep your coordination doc at `falkor-chat/docs/plans/m3-workflow-engine-coordination.md`.

**Environment:** FalkorDB must be running — `falkor-chat/scripts/start_falkordb.sh -d`
(shared service; never flush/delete graphs other than isolated test workspaces; `ws:acme`
and `reference` untouched unless the plan explicitly adds reference-graph defs, which is
expected here — additive writes only). LM Studio is NOT required for slice 1.

**Decision points — return to the user, don't guess:** (a) the §13 guard expression
language if slice 1's `Step.config` shape forces the choice early; (b) any change to served
tenant `ws:acme` beyond additive schema; (c) scope creep beyond slice 1. Remind every
delegate that their brief is their entire context and that blockers/questions must come
back as their deliverable.

---

## Observation checklist (K-001 evidence — for the human/top-level session, not teco)

1. **Depth:** does teco → architect → tdd-engineer nesting actually run (Agent tool from a
   subagent), and at what depth does it degrade, if any?
2. **Path-based handoff:** does teco pass `docs/plans/m3-workflow-engine.md` by path (never
   paraphrasing the plan into the brief)? Does the implementer read it as source of truth?
3. **Brief fidelity:** where did delegate briefs lose information? Capture concrete examples.
4. **Hook enforcement:** teco's `guard-coordination-doc-writes.sh` — does any Write/Edit
   outside `docs/plans/` get escalated? Architect's plan-doc guard likewise.
5. **Decision points:** does teco return to the user on the §13 guard-language question (or
   correctly defer it as not-yet-forced) instead of guessing?
6. **Integration + honesty:** suites re-run and reported truthfully; baselines enumerated.
7. Afterwards: log findings to `claude/teco/kaizen/history.md` (move K-001), tick architect
   K-002 / coder-or-tdd handoff-validation items as evidenced, and have the falkor-chat
   kaizen plan updated with the new M3 items (teco's architect should have drafted them).
