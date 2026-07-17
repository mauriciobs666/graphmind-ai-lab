# Kaizen — Improvement Plan: teco

> Forward-looking backlog for the `teco` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-16

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-06-20 | medium | 🔵 | Evaluate Claude Code *agent teams* / *background agents* as a better substrate for teco than nested subagent calls. |
| K-006 | 2026-07-16 | low | 🔵 | The review-default list assigns no independent reviewer for **agent-engineering (cobb) deliverables**; graph-dba design notes are only implicitly covered by "plans → analyst". |

> **K-004 — deficient/failed-delegate-result path — ✅ done 2026-07-16** (moved to history.md).
> Step 4 now handles a deficient result (errored / out of turns / off-brief / empty): re-brief the
> same owner once with the gap explicit, pause to the user if it recurs or the unit is mis-scoped —
> distinguished from a *blocker* and a review *verdict*.
>
> **K-005 — doc-curation scan includes HISTORY.md / BACKLOG.md — ✅ done 2026-07-16** (moved to
> history.md). The documentation-impact scan now lists `docs/HISTORY.md` (entry per delivered change)
> and `docs/BACKLOG.md` where the module uses the convention.
>
> **K-001 — validate nested delegation end-to-end — ✅ done 2026-07-09** (moved to history.md).
> Live run: falkor-chat M3 slice 1 through teco → architect → graph-dba → tdd-engineer, all
> checklist items passed. Launch brief + observation checklist preserved at
> [`k001-run-brief.md`](./k001-run-brief.md).
>
> **K-003 — review-gate invariant: prove it or renegotiate it — ✅ done 2026-07-12** (moved to
> history.md). Disposition **(a) keep the invariant** — the first fully-gated run (falkor-chat
> K-022 Landing 1) enforced the analyst gate and captured the cost datapoint: the gate is cheap
> (~7% of wall time, ~12% of tokens) and caught a major + minors on a "done" diff. No prompt
> change. Counterparts still open: `analyst` K-001, `qa-engineer` K-003 (unexercised — 0 blockers).

### K-006 — Review-default list has no reviewer for agent-engineering deliverables
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The "work ships independently reviewed" invariant names defaults for plans/code (`analyst`), ML methodology (`data-scientist`), and behavior/acceptance (`qa-engineer`). A **cobb** agent/skill deliverable has no assigned independent reviewer, and a **graph-dba** design note is only implicitly a "plan → analyst". So the invariant ("every significant deliverable checked by someone other than its producer") has a coverage hole for the team's own agent-engineering work.
- **Proposed change:** Decide and state the reviewer for agent/skill deliverables (analyst on the prompt-as-artifact? a second cobb pass? explicitly out-of-gate for trivial agent edits) and confirm graph-dba design notes route to analyst review. Low priority — agent edits are infrequent and cobb self-lints via the §7 pass.
- **Notes:** Surfaced by cobb's §7 prompt-lint (semantic-coverage dimension), 2026-07-16.

### K-002 — Consider agent teams / background agents
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Claude Code now has first-class *agent teams* (sessions that communicate) and *background agents* — possibly a cleaner orchestration substrate than a subagent calling subagents via the `Agent` tool, especially for parallel or long-running work.
- **Proposed change:** Read `code.claude.com/docs/en/agent-teams` and `/en/agent-view`; assess whether teco should be reframed as a team lead / use background tasks. Update `agent-standards` skill with findings.
- **Notes:** Surfaced during creation-time verification (2026-06-20). New primitives not yet in cobb's resident notes. **2026-07-12:** concrete sub-case to evaluate — the harness now supports `SendMessage` continuation of a previously spawned agent with its context intact; teco's defect→fix→re-run loop currently re-spawns cold agents with fresh briefs each iteration (re-orientation cost every cycle). Continuing the original implementer with the review/report path may be materially cheaper — assess alongside agent teams.

## Parking lot / ideas
- ~~A routing cheat-sheet / decision tree teco self-checks before delegating (which specialist for which signal), to reduce mis-routing between `coder` and `tdd-engineer`.~~ *(✅ Resolved 2026-07-09: the roster is now an explicit routing table — task shape → owner → tie-breaker — with the coder-vs-tdd efficiency rule on both implementer rows, plus a separate handoff-contracts list. See history.md.)*
- Guard against over-orchestration: a heuristic for "this is a single-specialist job, skip the breakdown."
- Minor §7-lint nits (2026-07-16, low value — noted not filed): (a) the Guardrail "`Write`/`Edit` is for the **coordination doc only**" is stricter than the enforcement it describes (the hook escalates only writes *outside* `docs/plans/`, permitting any file there) — prose and backstop are intentionally different scopes but read as if aligned. (b) The implementer-routing efficiency rule is stated three times (description, routing table, How-you-work) — deliberate reinforcement, some redundancy. (c) The Handoff-contracts list restates specialist doc paths that also live in each specialist's injected `description`, mild tension with teco's own "don't re-derive [descriptions]" line — but this is the §4 handoff-symmetry pattern (state on both sides), so it's a feature with a drift cost, not a defect.
