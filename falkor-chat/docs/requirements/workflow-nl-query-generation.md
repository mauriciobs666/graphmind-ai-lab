# Natural-language query generation over structured graph data — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of six sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — see
`docs/requirements/workflow-business-entities.md` for the shared background). The others:
`docs/requirements/workflow-catalog-lookup.md`, `docs/requirements/workflow-business-entities.md`,
`docs/requirements/workflow-deterministic-compute.md`, `docs/requirements/workflow-durable-profile.md`,
`docs/requirements/workflow-composition.md`. This document was spun off mid-interview from
`workflow-catalog-lookup.md`, not scoped out on day one; `workflow-composition.md` is younger
still, opened mid-interview of `workflow-durable-profile.md`.

## Intent
Spun off from `workflow-catalog-lookup.md` (2026-08-22): letting a workflow answer
arbitrarily-phrased natural-language questions against structured graph data turned out, on
reflection, to be its own substantial capability — closer in nature to the rigor
`docs/plans/graphrag-eval.md` already applies to retrieval quality (golden sets, calibration,
accuracy evaluation) than to a simple fixed-shape lookup tool. Not yet interviewed in depth.

## Problem & current state
falkor-chat's tools are deliberately fixed, author-defined schemas — nothing today lets an LLM
generate its own query against structured graph data from a free-form question. `salesperson`
(the comparison case that originally surfaced this whole effort) already does this with its
`cypher_qa` tool, but with weak safety (a regex keyword blocklist is the only guard against a
generated query doing something it shouldn't) and no accuracy-evaluation methodology at all.
Doing this safely and reliably enough for production workflow use is a distinct problem from
`workflow-catalog-lookup.md`'s fixed-shape lookups: it needs injection/dangerous-operation
safety, schema-awareness, and an answer-accuracy evaluation approach (likely a `data-scientist`-
designed golden set, in the spirit of `docs/plans/graphrag-eval.md`).

## User stories
_To be captured._

## Functional requirements
_To be captured._

## Out of scope
_To be captured._

## Acceptance criteria
_To be captured._

## Open questions
_To be captured._

## Decision log
2026-08-22 — Spun off from `workflow-catalog-lookup.md`'s original FR-3 (arbitrary-phrasing
support) once the stakeholder recognized it as a distinct project with its own safety and
evaluation-methodology needs, not a detail of catalog lookup. Not yet interviewed — queued.
