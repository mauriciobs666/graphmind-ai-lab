# Kaizen — Improvement Plan: teco

> Forward-looking backlog for the `teco` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-25

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-006 | 2026-07-16 | low | 🔵 | The review-default list assigns no independent reviewer for **agent-engineering (cobb) deliverables**; graph-dba design notes are only implicitly covered by "plans → analyst". |
| K-007 | 2026-07-24 | medium | 🔵 | Adopt `SendMessage` continuation of the original implementer in the defect→fix→re-run loop (step 4), instead of respawning a cold agent each cycle. |

> **K-004 — deficient/failed-delegate-result path — ✅ done 2026-07-16** (moved to history.md).
> Step 4 now handles a deficient result (errored / out of turns / off-brief / empty): re-brief the
> same owner once with the gap explicit, pause to the user if it recurs or the unit is mis-scoped —
> distinguished from a *blocker* and a review *verdict*.
>
> **K-005 — doc-curation scan includes HISTORY.md / BACKLOG.md — ✅ done 2026-07-16** (moved to
> history.md). The documentation-impact scan now lists `docs/HISTORY.md` (entry per delivered change)
> and `docs/BACKLOG.md` where the module uses the convention.
>
> **K-002 — agent teams / background agents evaluation — ✅ done 2026-07-24** (moved to
> history.md). Disposition: **reject** reframing teco as an agent-teams lead — its loop is
> sequential/dependency-gated, exactly the shape the agent-teams docs say a single
> session/subagents handles better than teams. The 2026-07-12 sub-case (defect→fix→re-run
> respawning cold) isn't an agent-teams question — it's answered by `SendMessage` continuation
> of the original delegate. Spun off as **K-007**.
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

### K-007 — Continue the original implementer via SendMessage instead of respawning cold
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Step 4's deficient-result path (K-004, done 2026-07-16) re-briefs "the same owner" on a deficient result, but today that means a fresh `Agent` call each cycle — the new instance has no memory of what it built or why, so teco re-explains context every iteration. `SendMessage` can resume a previously spawned agent with its context intact, which fits this loop directly.
- **Proposed change:** Update step 4 so the defect→fix→re-run re-brief uses `SendMessage` to the original delegate rather than a fresh `Agent` call, reserving a cold re-spawn for cases where the original agent errored out entirely or exhausted its turn budget.
- **Notes:** Spun off from K-002 (agent-teams evaluation, closed 2026-07-24). `SendMessage` continuation of `Agent`-tool subagents is confirmed available per the harness's own tool description, but the two docs read (`agent-teams`, `agent-view`) describe two different continuation mechanisms (teammate mailbox messaging vs. background-session resume/respawn) for a different primitive — verify `SendMessage`'s actual behavior on a real re-brief cycle before locking the step-4 wording.

## Parking lot / ideas
- ~~A routing cheat-sheet / decision tree teco self-checks before delegating (which specialist for which signal), to reduce mis-routing between `coder` and `tdd-engineer`.~~ *(✅ Resolved 2026-07-09: the roster is now an explicit routing table — task shape → owner → tie-breaker — with the coder-vs-tdd efficiency rule on both implementer rows, plus a separate handoff-contracts list. See history.md.)*
- Guard against over-orchestration: a heuristic for "this is a single-specialist job, skip the breakdown."
- Minor §7-lint nits (2026-07-16, low value — noted not filed): (a) the Guardrail "`Write`/`Edit` is for the **coordination doc only**" is stricter than the enforcement it describes (the hook escalates only writes *outside* `docs/plans/`, permitting any file there) — prose and backstop are intentionally different scopes but read as if aligned. (b) The implementer-routing efficiency rule is stated three times (description, routing table, How-you-work) — deliberate reinforcement, some redundancy. (c) The Handoff-contracts list restates specialist doc paths that also live in each specialist's injected `description`, mild tension with teco's own "don't re-derive [descriptions]" line — but this is the §4 handoff-symmetry pattern (state on both sides), so it's a feature with a drift cost, not a defect.
