# Workflow composition (sub-workflow orchestration) — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of six sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — see
`docs/requirements/workflow-business-entities.md` for the shared background). The others:
`docs/requirements/workflow-catalog-lookup.md`, `docs/requirements/workflow-business-entities.md`,
`docs/requirements/workflow-deterministic-compute.md`, `docs/requirements/workflow-durable-profile.md`,
`docs/requirements/workflow-nl-query-generation.md`. This document is the youngest of the six —
it was opened mid-interview, once the stakeholder confirmed a longer-term intent to compose
several dedicated workflows under one orchestrating agent, rather than only ever granting one
agent step every tool directly.

## Intent
A deliberately future-phased capability: the near-term plan (see
`docs/requirements/workflow-durable-profile.md` and sibling docs) is one orchestrating `agent`
step granted many tools (catalog lookup, cart/order, compute, profile), mirroring how
`salesperson` and `triage` already work today — no new engine primitive required for that. But
the stakeholder wants to expand into other markets/domains later by composing *dedicated
workflows* (e.g. a profile-interview workflow, a catalog workflow) under a higher-level
orchestrator, rather than growing one flat agent step's tool list indefinitely. That composition
— one workflow run starting, waiting on, and resuming from a child workflow run, with something
like a call stack and its own state machine — does not exist in falkor-chat today. This document
tracks that as a real, named future capability so the intent isn't lost, without blocking or
being required by the other five capabilities.

## Problem & current state
Today a workflow run's step types (`agent`, `human`, `wait`, `decision`) all execute within one
run; there is no step type or mechanism for one workflow run to start a *separate* workflow run
as a child, wait on its outcome, and resume with its result. Composing specialized workflows —
e.g. a top-level orchestrator delegating "collect the customer's profile" to a dedicated
profile-interview workflow, or "look up the catalog" to a dedicated catalog workflow — is not
possible; the only existing way to combine capabilities inside one run is granting a single
`agent` step many tools.

## User stories
_To be captured — this document is queued, not yet interviewed in depth._

## Functional requirements
_To be captured._

## Out of scope
_To be captured._

## Acceptance criteria
_To be captured._

## Open questions
- Whether this is genuinely required, or whether "one agent, many tools" scales far enough in
  practice that composition is never needed — an open question the stakeholder wants tracked,
  not necessarily built, in this phase.

## Decision log
2026-08-22 — Opened mid-interview (during `workflow-durable-profile.md`'s interview) once the
stakeholder confirmed both: (a) the near-term approach is one orchestrating agent step with many
tools (no new primitive needed), and (b) a longer-term intent to compose dedicated sub-workflows
exists and should be tracked now, even though it is explicitly deferred to a future phase and
does not block any of the other five capabilities. Not yet interviewed in depth.
