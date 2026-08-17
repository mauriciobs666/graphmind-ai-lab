# Generic Cypher MCP — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M?) · **Last updated:** 2026-08-17

## Intent
*(To be filled in as the interview proceeds.)*

## Problem & current state
Today the only MCP path to FalkorDB is `cpg/mcp/`'s `mcp__cpg__query(graph, cypher)` — one
read-only tool, `graph` and `cypher` both caller-supplied. It is **mechanically graph-agnostic**
already (it runs `GRAPH.RO_QUERY` against whatever key it's given, and the tool's own README notes
it could already reach `falkor-chat`'s `ws:*`/`reference` graphs), but it is **deliberately scoped
to CPG analysis by decision, not by code**: `docs/requirements/cpg-query-access.md` records
*"Non-CPG graphs / general agent access to FalkorDB"* as explicitly **out of scope**, and pairs
that with a second out-of-scope line — *"Authentication, per-user grants, and read-only
enforcement... FalkorDB stays open on `:6379` with no auth"* — accepted specifically because the
blast radius was CPG-only (a graph nothing else depends on for live state). Widening the tool's
sanctioned use reopens that second decision, since the same open, no-auth instance also holds
`falkor-chat`'s live chat/workspace data.

Related design-time decision (`docs/BACKLOG.md` M3 section): the official `@falkordb/mcpserver`
was evaluated and rejected — it ships **7 tools including `delete_graph`**, with no tool
filtering, which is a flat violation of the "one read-only tool" shape this repo settled on. The
recorded reversal trigger was *"an upstream server that can be filtered down to one read-only
tool."*

## User stories
*(To be filled in as the interview proceeds.)*

## Functional requirements
*(To be filled in as the interview proceeds.)*

- **FR-supersession** — *(placeholder)* this feature supersedes the "Non-CPG graphs / general
  agent access to FalkorDB" out-of-scope line in `docs/requirements/cpg-query-access.md`; that
  document must be updated to point here so the two do not disagree (mirrors how that document's
  own FR-6 handled its supersession of `joern-cpg-pipeline.md` FR-9).

## Out of scope
*(To be filled in as the interview proceeds.)*

## Acceptance criteria
*(To be filled in as the interview proceeds — expected to include a doc-consistency criterion:
`cpg-query-access.md`'s out-of-scope line is updated and no reader finds the two documents
disagreeing about blast radius.)*

## Open questions
*(To be filled in as the interview proceeds.)*

## Decision log
- 2026-08-17 — Session opened. Stakeholder asked "how can I turn my cpg mcp into a generic
  cypher mcp" (Mode 2 question). `tico` grounded the answer in `cpg/mcp/README.md` and
  `docs/requirements/cpg-query-access.md`, and flagged that this reverses that document's
  recorded out-of-scope decision on non-CPG graph access. Stakeholder confirmed: open a
  requirements interview rather than answering the "how". This document tracks it.
