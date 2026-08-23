# `cobb` implementation-boundary recognition — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-23

## Intent
`cobb` should recognize, on its own, when a task in front of it needs actual application/product
source-code implementation — not agent/skill/hook/MCP-standards artifacts within its documented
topic-remit — and stop to involve the stakeholder rather than attempting the implementation
itself. The mechanical write guard (`guard-cobb-topic-writes.sh`) already catches this after the
fact by escalating a mismatched path to a manual "ask" — that backstop worked in the triggering
instance below. What's missing is `cobb` recognizing the boundary *itself*, before it ever reaches
for `Write`/`Edit`, so the guard is a safety net rather than the only thing standing between
`cobb` and an out-of-remit code change.

## Problem & current state
2026-08-23 — the stakeholder observed `cobb` attempt to directly edit Python source under
`cypher-mcp/` (not `cypher-mcp/README.md` — the only `cypher-mcp` path in `cobb`'s allowlist,
confirmed via `guard-cobb-topic-writes.sh` line 92: `cypher-mcp/README.md|*/cypher-mcp/README.md`
is the only `cypher-mcp` entry). The write guard escalated as designed and the stakeholder denied
it. Stakeholder: "I noticed a cobb instance doing direct python code manipulation and would like
that the coder or the tdd-engineer would have done that."

`cobb`'s own definition (`claude/cobb/cobb.md`) states no boundary against implementing product
source code at all — unlike `architect`/`analyst`, which explicitly say "Does NOT edit source
code," `cobb` carries `All tools` and a topic-bounded (not folder-bounded) remit with no equivalent
line. Today the only thing preventing `cobb` from writing application source code is the path-based
write guard catching it after `cobb` has already decided to attempt the edit — the guard has no
visibility into *why* `cobb` reached for the tool, only *where*.

## User stories
- As the stakeholder, when `cobb` runs into a task that needs actual source-code implementation, I
  want it to stop and ask me rather than attempt the edit itself — the guard denying it after the
  fact means I already had to notice and intervene.
- As the stakeholder, when `cobb` stops to ask in that situation, I want delegating the task to
  `coder` or `tdd-engineer` (via `cobb`'s own `Agent` tool) to be one of the choices it offers me —
  not something it decides on its own, and not a dead end where it just declines with no path
  forward.

## Functional requirements
- **FR-1:** `cobb` must recognize, before reaching for `Write`/`Edit`, when the next step of a task
  requires implementing actual application/product source code — as opposed to an agent/skill/hook
  definition, kaizen curation, or an MCP/agent-standards document within its documented topic-remit
  (`claude/AGENTS.md` "Hook machinery" / its own allowlist). On recognizing this, it must not
  attempt the implementation itself.
- **FR-2:** Having recognized an FR-1 case, `cobb` must stop and involve the stakeholder rather than
  silently declining or silently proceeding. **When running interactively**, this means asking the
  stakeholder directly. **When running as a delegated subagent** (isolated context, no live human
  turn), this means returning the recognized boundary plus the choice below as its deliverable
  instead of guessing — consistent with `cobb`'s own existing subagent-mode convention
  (`claude/cobb/cobb.md`: "When a decision genuinely isn't yours... return what you did establish
  plus the sharp question or approval request... don't guess").
- **FR-3:** Among the choices `cobb` presents at that pause, delegating the implementation to
  `coder` or `tdd-engineer` via `cobb`'s own `Agent` tool must be offered as one option — so the
  stakeholder can approve delegation in one step rather than having to separately invoke `coder`/
  `tdd-engineer` themselves.

## Out of scope
- **Which of `coder` vs `tdd-engineer` is the right delegate for a given task** — ordinary task-shape
  routing (bug fix/safety-net → `tdd-engineer`, plan-ready implementation → `coder`) that already
  exists elsewhere in the team's conventions; this document only requires that delegation be offered,
  not which agent it resolves to.
- **`guard-cobb-topic-writes.sh`'s allowlist/mechanism** — unaffected. It already caught the
  triggering instance correctly; this document is about `cobb` recognizing the boundary before ever
  reaching the guard, not about changing what the guard allows or denies.
- **Other agents' implementation boundaries** — this document is scoped to `cobb` specifically,
  the only agent the triggering instance and the stakeholder's request concern.

## Acceptance criteria
- **AC-1:** Given `cobb` is working a task and the next step requires editing actual application/
  product source code outside its documented topic-remit, when it reaches that step, then it stops
  before calling `Write`/`Edit` on that path rather than attempting the change.
- **AC-2:** Given `cobb` stops under AC-1 while running interactively, when it presents its options
  to the stakeholder, then "delegate to `coder`/`tdd-engineer` via `Agent`" is one of the offered
  choices.
- **AC-3:** Given `cobb` stops under AC-1 while running as a delegated subagent, when it returns its
  final deliverable, then it states the recognized boundary and the same delegation choice instead
  of guessing or silently proceeding.
- **AC-4 (regression check):** `cobb`'s existing topic-remit (agent/skill/hook definitions, kaizen
  curation, MCP/agent-standards docs like `cypher-mcp/README.md`) is unaffected — FR-1 only applies
  to work genuinely outside that remit, not to `cobb`'s normal job.

## Open questions
None currently — see Decision log for the readback pending confirmation.

## Decision log
- 2026-08-23 — Stakeholder: "on a related subject, I noticed a cobb instance doing direct python
  code manipulation and would like that the coder or the tdd-engineer would have done that" → opened
  as a new requirements thread (`tico` switching topics within the same session).
- 2026-08-23 — Asked what Python file/area cobb touched → Stakeholder: "it tried to edit the
  cypher-mcp i denied" → confirmed via `guard-cobb-topic-writes.sh` that only `cypher-mcp/README.md`
  is in cobb's allowlist, so any other `cypher-mcp` path (e.g. a `.py` source file) would escalate —
  the guard worked correctly; the gap is cobb attempting the edit in the first place.
- 2026-08-23 — Asked what should happen instead: cobb delegates on its own, or cobb stops and asks
  → Stakeholder: "2 but one the possible options should be the 1" → settles FR-2 (stop and ask) with
  FR-3 (delegation offered as a choice, not decided unilaterally by cobb).
- 2026-08-23 — Readback given (intent, FR-1..3, out of scope, generalization from "python" to any
  application/product source code) → Stakeholder: "looks right, mark it ready for design" →
  Status → **Ready for design**; next step is an `architect`/`cobb` pass over this document.
