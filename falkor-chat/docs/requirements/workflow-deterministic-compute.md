# Deterministic computation inside a workflow — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of four sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — see
`docs/requirements/workflow-business-entities.md` for the shared background). The others:
`docs/requirements/workflow-catalog-lookup.md`, `docs/requirements/workflow-business-entities.md`,
`docs/requirements/workflow-durable-profile.md`. Read this one for exact, non-AI computation
(totals, quantity math, aggregation) inside a workflow run specifically.

## Intent
_Queued — not yet interviewed._

## Problem & current state
Today's workflow step types are `agent` (an LLM turn), `human`/`wait` (park for a reply), and
`decision` (a pure branch — its guard only *compares*, never *computes*). There is no primitive
for exact, deterministic computation (e.g. summing line totals, quantity arithmetic) as part of
a run — today it would have to be routed through an LLM `agent` turn calling a tool, which is
slow, non-deterministic, and spends a step-budget slot on something that should be exact. A
general expression-evaluation guard (`expr`) already exists as a documented, unimplemented
`NotImplementedError` in the guard language — a prior, seemingly abandoned attempt at a related
idea.

## User stories
_To be captured._

## Functional requirements
_To be captured._

## Out of scope
_To be captured._

## Acceptance criteria
_To be captured._

## Open questions
- Engine-native step type (closed op-set, risks the same "one op short" trap that stalled
  `expr`) vs. a deterministic tool (zero new engine surface, but still routes through an LLM
  `agent` turn just to invoke a pure function) — flagged by the architect as the sharpest open
  fork in the whole business-entities effort; unresolved, not yet a decision.

## Decision log
2026-08-22 — Split out as one of four sibling capabilities; not yet interviewed (queued behind
whichever the stakeholder prioritizes first).
