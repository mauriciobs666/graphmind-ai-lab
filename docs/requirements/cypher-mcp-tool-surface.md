# Cypher MCP tool surface — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (M?) · **Last updated:** 2026-09-12
>
> **Revisits:** [`cpg-query-access.md`](./cpg-query-access.md) FR-2 ("a single tool taking exactly
> two parameters — no second tool, no per-recipe tools") and the build-vs-buy call in
> [`../plans/cpg-query-access.md`](../plans/cpg-query-access.md) §3.2, per that plan's own recorded
> reversal trigger.

## Intent
`docs/plans/cpg-query-access.md` §3.2 named a specific reversal trigger for its "one tool, two
parameters" design: *"if a future need arises for multi-tool graph access (schema discovery, write
paths, non-CPG graphs for other agents), revisit — at that point FR-2 no longer binds and the
official server becomes the cheaper answer."* Two of those three conditions have since landed
(write paths via M5/M7/M8; non-CPG, multi-agent graph use via M7's `kaizen_team` rollout). This is
a **principled check-in, not a response to a specific incident** — nothing has broken, and no
concrete friction has been reported by an agent to date. The stakeholder wants to deliberately
revisit whether the one-tool constraint still earns its keep now that the landscape it was decided
against has changed, rather than let the fired trigger sit unexamined.

## Problem & current state
Today `mcp__cypher__query(graph, cypher, agent=None)` is the only MCP surface onto FalkorDB: one
tool, reads unrestricted, a narrow 6-shape attributed write path scoped to `kaizen_team`'s
`:KaizenEntry`/`:Agent` labels. There is no in-tool graph discovery (`GRAPH.LIST` via `redis-cli`
is the fallback) and no schema-discovery tool. The official `@falkordb/mcpserver` (evaluated at M3,
re-affirmed at M5) exposes 7 tools including an unfiltered `delete_graph`, with no documented way
to filter or disable individual tools in Claude Code.

## Scope
**In:** a direct, honest way for a caller of `mcp__cypher__query` to discover the set of currently
loaded FalkorDB graph names, without deliberately triggering the "graph not found" error path and
without shelling out to `redis-cli GRAPH.LIST`.

**Not in:** anything that changes what is exposed, how the tool is shaped beyond this one
capability, or whether the write path/official-server questions get reopened (see Out of scope).

## User stories
- As **any consumer of `mcp__cypher__query`** (all current agent consumers — `analyst`,
  `architect`, `graph-dba`, `qa-engineer`, `cobb`, `teco`, `data-scientist`, `security-expert`,
  `devops`, `coder`, `tdd-engineer`, `frontend-engineer`, `tico`), I want to ask directly which
  graphs are loaded, so that I don't have to deliberately mistype a graph name to trigger an error
  message just to see the list, or drop out of MCP into a shell command.

## Functional requirements
- **FR-1** — A caller of `mcp__cypher__query` can discover the set of currently loaded FalkorDB
  graph names directly, without first triggering a "graph not found" error and without shelling out
  to `redis-cli GRAPH.LIST`.
- **FR-2 (revises `cpg-query-access.md` FR-2)** — The tool **remains a single tool**. This
  capability does not introduce a second tool. If it needs a new parameter, that parameter is
  **optional**, following the precedent `generic-cypher-mcp.md` already set when it added `agent`
  without violating the "one tool" shape. *(Context for the architect, not a requirement: the exact
  mechanism — a directive in the `cypher` text akin to `EXPLAIN`/`PROFILE`, an optional parameter,
  or something else — is a design decision.)*
- **FR-3** — The set of graph names this capability reveals has the **same blast radius** as what
  already leaks today through the "graph not found" error message (i.e. every graph on the
  instance, including `falkor-chat`'s live `ws:*`/`reference` graphs) — **no narrower, no broader**.
  This is deliberately an ergonomics-only change, not a new exposure decision; widening or
  narrowing that blast radius is explicitly out of scope (see below).
- **FR-4** — `docs/requirements/cpg-query-access.md`'s FR-2 is annotated to point here, the same way
  it already points to `generic-cypher-mcp.md` for the "non-CPG graphs" line — so no reader
  concludes graph discovery is impossible in-tool because of the original FR-2's literal wording.

## Out of scope
- **Schema discovery** (a `get_graph_schema`-equivalent capability) — the third condition named in
  the original reversal trigger. It has not fired (the CPG schema is still a static doc,
  `skills/joern-cpg/references/cpg-model.md`) and is not part of this delivery.
- **Any change to what is exposed / the tool's blast radius.** Still the full instance-wide graph
  list, identical to what the error path already leaks today (FR-3). Narrowing it (e.g. to just
  `cpg_*` and `kaizen_team`) or widening it further is not part of this delivery.
- **Re-running the full build-vs-buy comparison against the official `@falkordb/mcpserver`,** or any
  multi-tool redesign. This delivery is additive (one capability inside the existing one-tool
  shape), not a re-evaluation of build vs. buy — that question stays closed per `cpg-query-access.md`
  §3.2 and its M5 (`generic-cypher-mcp.md`) reaffirmation, unless a future need reopens it again.
- **`delete_graph` or any other official-server-only capability.**
- **Any change to the write path** (`authorize_write()`'s 6 recognized shapes, the `agent`
  parameter, or `kaizen_team`'s schema) — unaffected by this delivery.

## Acceptance criteria
- **AC-1** — Given a cold agent session and a live FalkorDB instance, when an agent wants to know
  which graphs are loaded, it gets the answer in **one direct tool call** — no deliberately-wrong
  graph name, no shell command.
- **AC-2** — The set of graph names returned by that call matches `redis-cli GRAPH.LIST`'s output at
  the same point in time, exactly.
- **AC-3** — `docs/requirements/cpg-query-access.md` and `cypher-mcp/README.md` are updated so no
  reader finds them disagreeing about whether direct in-tool graph discovery exists (FR-4).

## Open questions
*(none)*

## Decision log
- 2026-09-12 — Session opened. Stakeholder: revisit FR-2 in `cpg-query-access.md` — the reversal
  trigger has fired (write paths and non-CPG multi-agent graph use have both landed since the
  original decision). `tico` grounded this in `docs/plans/cpg-query-access.md` §3.2 (the exact
  trigger text), the M5/M7/M8 delivery history (`docs/HISTORY.md`), and confirmed this exact
  reconsideration has not been re-litigated since (M5's `generic-cypher-mcp.md` reopened "one tool"
  once, but only to decide widen-vs-second-tool *within* the same server, not build-vs-buy against
  the official server).
- 2026-09-12 — Trigger for this session? → **Principled check-in**: the recorded trigger fired, not
  a specific incident. No concrete agent-facing friction has surfaced to date (checked
  `graph-dba`'s and `cobb`'s kaizen histories for `GRAPH.LIST`/discovery pain — found none; both
  fall back to `redis-cli GRAPH.LIST` via their own `Bash` access without apparent friction).
- 2026-09-12 — How far should the reconsideration reach? → **Just graph discovery.** Not a full
  build-vs-buy redo: read-only-by-default, the attributed narrow write path, and the one-tool shape
  all stay exactly as they are. Schema discovery and the official-server question stay closed.
- 2026-09-12 — `tico` surfaced that the full graph list already leaks today via the "graph not
  found" error message (`cypher-mcp/README.md`'s error table cites `redis-cli GRAPH.LIST`'s full
  output). Given that, what's the actual gap? → **Just make it honest/direct** — no new exposure,
  purely an ergonomics fix: a direct way to ask, instead of deliberately mistyping a graph name.
  This settled FR-3 (same blast radius, no wider/narrower) and the Out of scope line ruling out
  reopening what's exposed.
- 2026-09-12 — Readback delivered and confirmed. Stakeholder: "Yes... flip it to Ready for design
  and hand it to architect." **Status → Ready for design.** No material assumption left
  unconfirmed; Open questions is empty.
