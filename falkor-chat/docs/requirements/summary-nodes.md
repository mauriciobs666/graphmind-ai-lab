# Summary Nodes — Feature Requirements
> Status: Interviewing · Last updated: 2026-07-12

## Intent
The stakeholder wants the graph to hold **summary nodes**: nodes that carry a condensed summary
of the other nodes connected to them, and that are themselves retrievable via RAG (in-graph vector
search). The goal is to **save time and computing resources when a high-level overview is needed**,
and to **help connect ideas** across otherwise separate parts of the graph — rather than always
reading every underlying node.

> Note (solution vs. need): "summary node" is the stakeholder's proposed shape. The underlying need
> is *cheap, fast high-level overviews + idea-linking without traversing/reading all source nodes*.
> Recorded here as the leading design candidate for the architect, not a locked requirement.

## Problem & current state
- Today a high-level overview requires reading (and often embedding/feeding to the LLM) every
  relevant underlying node — costly in latency, tokens, and RAM on the hot path.
- _(current handling to confirm — e.g. does anything summarize threads/channels today?)_

## User stories
- As an **AI agent**, I want a condensed overview of a set of connected nodes, so that I can ground a
  reply without reading every underlying message/node.
- As a **human catching up**, I want a high-level summary of a channel/thread I've been away from, so
  that I save time.
- As a **workflow / reasoning step**, I want summaries as retrieval targets, so that I can connect
  related ideas across the graph.

## Functional requirements
- **FR-1** — The graph supports a **summary node**: a node holding summary text plus an embedding,
  retrievable via the existing RAG (in-graph vector) search alongside other content.
- **FR-2** — A summary node records **what it summarizes** (links to the covered nodes/region), so a
  reader can trace a summary back to its sources.
- **FR-3** — A summary node can be created by any of these actors: the **AI agent on demand**, a
  **human manually**, an **automatic background process**, and a **workflow step**. _(All four
  desired; sequencing/priority among them TBD — see Open questions.)_
- **FR-4** — A summary is a **point-in-time snapshot** carrying the timestamp it was generated at, so
  a reader (agent or human) can tell **whether it still reflects its sources** by comparing against
  the covered nodes' own timestamps. Automatic rebuild is *not* required — refreshing is a consumer's
  choice, not mandatory machinery. _(Suggested mechanism: a snapshot date on the summary, compared to
  the latest change among covered nodes.)_
- _(more to elicit — cross-summary linking / "connect ideas")_

## Out of scope
- **Automatic / mandatory refresh** of summaries. Summaries are snapshots; keeping them current is a
  consumer's choice, not built-in machinery (FR-4).
- **Summaries-of-summaries / hierarchical rollups** — deferred (OQ1 below); this feature covers
  summaries over primary content.
- A **fixed per-thread or per-channel rollup** — coverage is free-form, chosen at creation.
- An **explicit "related summaries" linking capability** — connecting ideas is achieved via RAG
  search, not a new link type.

## Acceptance criteria
- **AC-1** — Given a summary node exists, when a RAG (vector) search matches its content, then the
  summary is returned alongside ordinary content results.
- **AC-2** — Given a summary node, when a reader inspects it, then they can see both the nodes/region
  it covers and the timestamp it was generated at.
- **AC-3** — Given covered nodes changed after a summary's snapshot timestamp, then a reader can
  detect the summary is stale by comparing timestamps (no automatic rebuild is triggered).

## Open questions
- OQ1 — Can a summary node itself be summarized later (summaries-of-summaries → a hierarchy)? Out of
  scope for now; noted as a likely future extension.
- OQ2 — Priority/sequencing among the four creation paths (agent / human / background / workflow) —
  which comes first? (Non-urgent; for backlog ordering.)
- OQ3 — Does anything summarize threads/channels today? (Confirm current-state baseline — believed
  "no" from AGENTS.md, to verify at design time.)

## Decision log
2026-07-12 — Intent → save time & compute when a high-level overview is needed, and help connect
ideas; summaries must be RAG-searchable. Serves AI agents, humans catching up, and workflows alike.
2026-07-12 — Scope of a summary → **free-form**: a summary node can attach to any node and summarize
whatever it is connected to (not a fixed per-thread/per-channel rollup), so it can be freely searched
later. Leading design candidate, not locked.
2026-07-12 — Creation paths → **all four** desired (AI agent on demand · human manual · automatic
background · workflow step). Framed as one summary-node primitive with multiple producers; sequencing
among them deferred to backlog ordering.
2026-07-12 — Freshness → summaries are **point-in-time snapshots** with a generation timestamp;
staleness is *detectable* (compare against covered nodes) but auto-refresh is out of scope.
2026-07-12 — "Connect ideas" → a **byproduct** of summaries being RAG-searchable (option 1), not a
distinct linking capability. No new link type to build for it.
