# Joern CPG Extraction Pipeline — Feature Requirements
> Status: Interviewing · Last updated: 2026-07-12

## Intent
The stakeholder wants a pipeline that runs **Joern** to extract a **Code Property Graph (CPG)** from
source code and represents that CPG **in FalkorDB**, to support **software-development work** —
specifically **code navigation and impact analysis** ("what calls this", "what breaks if I change
it"). **RAG indexing of code is explicitly not a goal for now.** The CPG is first made available to
the **agent team — the `analyst` agent** — and later to the **workflows and agent nodes** being built
in `falkor-chat`.

## Problem & current state
- Today the agents (e.g. `analyst`, which does impact analysis and RCA) reason about code by reading
  and grepping files — there is **no structured call-graph / data-flow representation** to query.
- Note: `falkor-chat` today is a hybrid human+AI chat platform over FalkorDB. This CPG capability is
  a **distinct new component** (decided 2026-07-12) — hence this doc lives at repo-root
  `docs/requirements/`. It reuses FalkorDB but is not part of the chat platform.

## User stories
- As the **`analyst` agent**, I want to query a code graph for callers/callees and dependency paths,
  so that I can do impact analysis without reading every file by hand.
- As a **falkor-chat workflow / agent node** (later), I want the same code-navigation queries, so that
  automated workflows can reason about code structure.
- _(more to elicit — the specific questions each consumer needs answered)_

## Functional requirements
- **FR-1** — A pipeline extracts a CPG from source code using **Joern** and loads it into
  **FalkorDB** as a queryable graph.
- **FR-2** — The stored CPG can answer **caller/callee** queries: who calls a given function, and
  what a given function calls.
- **FR-3** — The stored CPG can answer **transitive-impact** queries: the up/downstream reach of a
  change across call/dependency chains ("what could break if I change X").
- **FR-4** — The stored CPG can answer **data-flow** queries: how a value propagates through the code
  (Joern's data-flow edges).
- **FR-5** — The stored CPG can answer **symbol reference** queries: where a symbol is defined and all
  places it is referenced, across files.
- **FR-6** — The CPG is queryable by the **`analyst` agent** first; the same queries are later
  reachable by **falkor-chat workflows / agent nodes**.
- **FR-7** — Extraction is **on-demand / snapshot-based**: a run produces a fresh CPG for the target
  code; there is no requirement to keep it continuously in sync (auto-sync is a possible later
  extension).
- **FR-8** — The pipeline targets **this monorepo initially** (the languages present here — Python and
  JS/TS) but is designed to be **generic**, able to extract from arbitrary repositories later.

## Out of scope
- **RAG / vector indexing of code** — not a goal for now (may come later).
- **Continuous auto-sync** with a live repo — snapshots only for the first version.
- **The access mechanism** (MCP tool / skill / direct Cypher) — that's a design (HOW) decision for
  the architect, not a requirement here.

## Acceptance criteria
- **AC-1** — Given a target repo, when the pipeline is run, then a CPG for that code exists in
  FalkorDB and is queryable.
- **AC-2** — Given a function in the CPG, when the analyst asks for its callers and callees, then both
  are returned correctly.
- **AC-3** — Given a proposed change to a symbol, when the analyst asks for its transitive impact,
  then the up/downstream reach is returned across call/dependency chains.
- **AC-4** — Given a value/parameter, when a data-flow query is run, then the propagation path(s) are
  returned.
- **AC-5** — Given a symbol, when the analyst asks where it is defined and referenced, then all
  cross-file definitions and references are returned.

## Open questions
- OQ1 — How does the analyst *access* the CPG? (query tool/MCP, direct Cypher, a skill…?) — HOW, for
  the architect; noted as a consumer constraint (the analyst is a Claude Code subagent).
- OQ2 — Naming/structure of the new component (e.g. `code-graph/`) — for the architect.
- OQ3 — Which Joern language frontends are needed for the initial monorepo targets (Python, JS/TS) and
  how well they cover those languages — design-time verification.

## Decision log
2026-07-12 — Idea → run Joern to extract a CPG and represent it in FalkorDB.
2026-07-12 — Purpose → **code navigation & impact analysis** for software development; **RAG indexing
of code is out of scope for now**. Consumers → **`analyst` agent first**, then **falkor-chat
workflows / agent nodes**.
2026-07-12 — Required queries → all four: callers/callees, transitive impact, data-flow, cross-file
symbol references (FR-2..FR-5).
2026-07-12 — Scope → **this monorepo (Python + JS/TS) initially, but built generic** for arbitrary
repos later. Freshness → **snapshot / on-demand to start; auto-sync a possible later extension**.
2026-07-12 — Placement → **new top-level component** (serves the agent team *and* falkor-chat, distinct
from the chat platform). Doc relocated from `falkor-chat/docs/requirements/` to repo-root
`docs/requirements/` accordingly.
