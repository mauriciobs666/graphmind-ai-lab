# Structured catalog/reference lookup for workflows — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of four sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — see
`docs/requirements/workflow-business-entities.md` for the shared background). The others:
`docs/requirements/workflow-business-entities.md`, `docs/requirements/workflow-deterministic-compute.md`,
`docs/requirements/workflow-durable-profile.md`. Read this one for read-only, exact/filterable
domain data (e.g. a product catalog) specifically — not for mutable state (that's the
business-entities doc).

## Intent
_Queued — not yet interviewed._

## Problem & current state
Today the only workflow tool that reaches domain-adjacent data is `graphrag_retrieve` —
embedding-based semantic search over chat-like text, with a distance-cutoff/abstention policy.
It has no way to answer an exact, filterable question ("what is the price of X", "which items
are in category Y") against structured reference data, because nothing like that exists in the
schema at all today — `reference` (the read-mostly, replicated graph that already holds
`WorkflowDef` templates) has a placeholder box for "domain reference data / ontology / catalogs"
in the topology diagram, but nothing is built there.

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
2026-08-22 — Split out as one of four sibling capabilities; not yet interviewed (queued behind
whichever the stakeholder prioritizes first).
