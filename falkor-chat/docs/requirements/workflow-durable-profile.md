# Durable user-profile data for workflows — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M<n> TBD)

This is one of five sibling capabilities scoped out of a single "business entities in
falkor-chat workflows" idea (decision log, 2026-08-22 — see
`docs/requirements/workflow-business-entities.md` for the shared background). The others:
`docs/requirements/workflow-catalog-lookup.md`, `docs/requirements/workflow-business-entities.md`,
`docs/requirements/workflow-deterministic-compute.md`, `docs/requirements/workflow-nl-query-generation.md`
(the last one added mid-interview, spun off from `workflow-catalog-lookup.md`). Read this one for
durable, cross-conversation profile data about the person a workflow is talking to (e.g. a name,
a delivery address) specifically — not for cart/order-shaped transactional state.

## Intent
_Queued — not yet interviewed._

## Problem & current state
Today, information a workflow collects about the person it's talking to (e.g. a name, an
address) has nowhere durable to live except a run's `ctx` — flat, run-scoped, and discarded once
the run ends. `identity` (the global, read-mostly user/auth graph) and `ws:{workspaceId}`'s local
`User` node (a membership *projection* of identity) are not currently write targets for this
kind of data, and profile information is conceptually neither run-scoped nor workspace-scoped —
it is a durable, potentially cross-workspace axis the current topology does not yet serve for
mutable data.

## User stories
_To be captured._

## Functional requirements
_To be captured._

## Out of scope
_To be captured._

## Acceptance criteria
_To be captured._

## Open questions
- Where profile writes land: `identity` accepting occasional writes (breaking its current
  read-mostly character) vs. per-workspace snapshot materialization (trading a single source of
  truth for possible cross-workspace staleness) — flagged by both the architect and graph-dba as
  unresolved.

## Decision log
2026-08-22 — Split out as one of four sibling capabilities; not yet interviewed (queued behind
whichever the stakeholder prioritizes first).
