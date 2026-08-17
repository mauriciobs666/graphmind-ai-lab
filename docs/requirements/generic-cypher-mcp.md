# Generic Cypher MCP — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — (M?) · **Last updated:** 2026-08-17

## Intent
The stakeholder wants FalkorDB to become a **knowledge plane** shared across the whole Claude
agent team — collective memory today, project-management state and (eventually) even
design/requirements documents themselves, tomorrow — instead of each agent's memory living
siloed in its own markdown files that only `cobb` ever reads across agents. "Turn the `cpg` MCP
into a generic Cypher MCP" was the access-mechanism instinct that opened this conversation; the
real need underneath it is agents being able to **write to and query a shared graph directly**,
with enough identity/attribution that it's trustworthy to do so.

Midway through the interview the stakeholder reconsidered the storage model: FalkorDB is known to
be memory-hungry, so rather than making the graph the authoritative store for `graph-dba`'s
kaizen learnings (displacing the markdown file), **markdown stays authoritative and the graph
becomes a parallel index/GraphRAG layer over it** — playing to what a graph database is actually
good at (traversal and, eventually, semantic similarity search) instead of duplicating what git
and a text file already do well (durable, diffable, human-reviewable storage).

This document scopes the **first concrete, buildable step** toward the larger vision: proving two
things on a narrow, low-stakes slice before deciding whether/how to extend further —
1. **An index pattern**: `graph-dba`'s kaizen inbox stays a markdown file, but its content is
   mirrored into a FalkorDB graph, kept in sync automatically, and queryable by any agent.
2. **A write pattern**: the same generic MCP tool also accepts a **direct** write — one agent
   recording a fact that has no markdown file behind it at all — because that's the shape a real
   future need (cross-agent working memory) will require, even though this delivery doesn't build
   that feature itself.

Extending either pattern to the rest of the team, to task/backlog state, to falkor-chat's data, or
to documents themselves is real and named by the stakeholder as where this is ultimately going,
but is deliberately **not** what this document commits to delivering — see Out of scope.

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
tool."* This feature reopens that too: it deliberately asks for a **write** path for one narrow,
attributed case (an agent recording a fact with no markdown backing) — not a wholesale return to
an unrestricted-write server; there is no `delete_graph`-shaped capability requested here.

Separately, `claude/AGENTS.md` records the standing convention that every agent's `kaizen/inbox.md`
is an append-only markdown file, distilled periodically by `cobb` (verify → route → log in
`history.md` → clear) — the only reader across agents today is `cobb`, during that distillation
pass. This feature does **not** change how `graph-dba` writes its inbox (still markdown, still
`cobb`-distilled, still the append-only convention `claude/AGENTS.md` describes) — it adds a
second, queryable surface over the same file that any other agent can read directly.

## User stories
- As **any agent**, I want to query `graph-dba`'s captured learnings directly, so that I benefit
  from what it discovered without waiting for `cobb` to distill and promote it into a prompt or
  doc.
- As **`graph-dba`**, I want to keep recording learnings exactly the way I do today (append to my
  markdown inbox), so nothing about my own workflow has to change for this to work.
- As **`cobb`**, I want to record an ad-hoc fact straight into the shared graph — with no markdown
  file behind it — so I can prove the knowledge plane can hold things that aren't sourced from a
  file at all, which is what future cross-agent working memory will need.
- As the **stakeholder**, I want to eventually read and write the knowledge plane myself, with my
  entries distinguishable from an agent's — not required for this delivery, since I can already
  watch graph state today via FalkorDB's existing web console.
- As the **team**, we want one generic, graph-agnostic Cypher-capable MCP tool as the access
  mechanism — not scoped to CPG graphs, and capable of both reading and a restricted, attributed
  form of writing — so this pattern can extend to future knowledge without a new tool being
  invented each time.

## Functional requirements
- **FR-1** — Agents reach the knowledge plane through a **generic, graph-name-agnostic** Cypher
  MCP tool — the same two-parameter shape as today's `mcp__cpg__query(graph, cypher)`, but not
  limited to `cpg_*` graphs, and able to both read and (per FR-6) write. This **supersedes** the
  "Non-CPG graphs / general agent access to FalkorDB" out-of-scope line in
  `docs/requirements/cpg-query-access.md`, which must be updated to point here (mirrors how that
  document's own FR-6 superseded `joern-cpg-pipeline.md` FR-9).
- **FR-2** — `graph-dba`'s kaizen inbox stays a markdown file, unchanged in how `graph-dba` writes
  to it and how `cobb` distills it. Nothing about the existing convention (`claude/AGENTS.md`)
  changes for the underlying file.
- **FR-3** — The content of that markdown file is mirrored into a graph, queryable by **any agent,
  directly** — not gated behind `cobb`'s distillation pass. This is the pilot's core "collective
  memory" payoff.
- **FR-4** — The mirror is **kept in sync automatically**: a new or edited entry in `inbox.md`
  becomes queryable in the graph without a separate manual rebuild step, and an entry `cobb`
  clears from `inbox.md` during distillation is reflected (removed/updated) in the graph too — the
  index never says something the markdown file no longer says.
- **FR-5** — A mirrored entry captures what today's markdown entry does — the dated fact, its
  evidence, the context it surfaced in, and a suggested home — and is queryable via graph
  traversal at minimum. Semantic/similarity search (true GraphRAG, vector-indexed) over entries is
  a **stretch goal** for this delivery: valuable and explicitly wanted eventually, but not
  required for this delivery to be considered done (see Acceptance criteria).
- **FR-6** — The same generic tool accepts a **direct, non-markdown-backed write**: an agent
  (proven here with `cobb`) can record a fact straight into the graph with no file behind it,
  attributed to itself.
- **FR-7** — Write attribution and restriction run on **trusted, self-reported agent identity** —
  the same no-auth trust level the rest of this repo already runs at (FalkorDB itself has no auth
  on `:6379`), not a hardened/cryptographic access-control boundary. One agent's write cannot be
  accepted as another agent's (e.g. a call claiming to be `graph-dba` cannot write an entry that
  reads as authored by `cobb`), but this is enforced at the "well-behaved callers can't do this by
  accident" level, not hardened against a malicious caller.
- **FR-8** — The identification/attribution mechanism (FR-7) must leave room for a future
  human-authored entry to be distinguished from an agent's the same way — the stakeholder's own
  read/write access is a later phase (see Out of scope), but FR-7/FR-8 should not have to be
  redone to add it.
- **FR-9** — Every doc describing the standing kaizen-inbox convention (starting with
  `claude/AGENTS.md`) is updated to note that `graph-dba`'s inbox is **also** mirrored into a
  queryable graph — an addition to the documented convention, not a contradiction of it (the file
  itself, its ownership, and `cobb`'s distillation role are unchanged).

*Context for the architect (not a requirement):* the initial-build/ongoing-sync mechanism for
FR-4 (how the mirror is produced and kept current — a file watcher, a hook on `graph-dba`'s
append action, a scheduled job, something else) is a real design decision, not specified here. So
is how FR-6's direct-write path coexists with FR-4's markdown-sourced mirror inside the same
graph/tool without the two write paths colliding. Sizing/memory-footprint implications of running
FalkorDB for this purpose (the stakeholder's stated worry that motivated keeping markdown
authoritative) are also an architecture concern, not quantified by the stakeholder here.

## Out of scope
- **falkor-chat data integration** — actually linking knowledge-plane entries to chat
  threads/messages (traversable from one to the other) is the stakeholder's stated end goal, but
  is a later phase, not part of this delivery.
- **Design/requirements/plan documents becoming graph data** — the long-term ambition behind this
  whole initiative ("ultimately... the design and requirements to be abstracted into the knowledge
  graph"), explicitly deferred by the stakeholder.
- **Mirroring any other agent's kaizen inbox.** This pilot is `graph-dba` only. Extending to the
  rest of the team is a follow-on decision made after this pilot is evaluated.
- **A real cross-agent-working-memory feature.** FR-6 proves the write path exists and works with
  one toy, non-markdown-backed fact from `cobb` — it does not design or build the actual feature
  that would use it day to day.
- **Migrating any module's `docs/BACKLOG.md` / task-backlog state to the graph** — the
  "project-management" half of the original motivation. Deferred in favor of proving the index and
  write patterns first, precisely because `BACKLOG.md` is load-bearing for live, in-flight
  coordination today (`teco` reads it, other docs cite its K-IDs) and a mistake there is costlier
  than one in an append-only, raw-capture inbox.
- **The stakeholder's own direct read/write access to the knowledge plane** through the
  agent-facing mechanism — deferred to a later phase. Monitoring in the meantime needs no new
  work: FalkorDB's existing web console (`http://localhost:3000`, published by
  `falkor-chat/scripts/start_falkordb.sh`) already shows graph state.
- **Guaranteed semantic/similarity search.** Genuine vector-indexed GraphRAG retrieval is wanted
  eventually and may fall out of this delivery as a stretch goal, but structural traversal query
  is what "done" requires (FR-5).
- **Hardened/cryptographic access control.** The pilot runs at the same trusted
  self-identification level the rest of the repo runs at today — see FR-7.
- **Deleting or replacing `claude/graph-dba/kaizen/inbox.md`.** Explicitly reversed mid-interview:
  the file stays, git-tracked, authoritative, unchanged in how it's written and distilled.

## Acceptance criteria
- **AC-1** — Given the pilot is live, when an agent other than `graph-dba` queries the knowledge
  plane for `graph-dba`'s learnings via the generic MCP tool, it gets back entries with the same
  fields today's markdown format carries (date, fact, evidence, context, suggested home), via
  graph traversal, with no `cobb`-distillation step in between.
- **AC-2** — Given the 46 lines of entries in `claude/graph-dba/kaizen/inbox.md` as of this
  document's writing, when the mirror is first built, every one of them is present in the graph.
- **AC-3** — Given the mirror exists, when `graph-dba` appends a new entry to `inbox.md`, the
  entry becomes queryable in the graph without any manual rebuild step (the exact delay is an
  implementation detail for the architect to define and test against, not a stakeholder-specified
  number).
- **AC-4** — Given the mirror exists, when `cobb` distills and clears an entry from `inbox.md`,
  that entry is no longer returned by a graph query afterward — the index never contradicts the
  file it's mirroring.
- **AC-5** — `claude/graph-dba/kaizen/inbox.md` still exists, unchanged in ownership, format, and
  distillation process, after this delivery — the graph is additive, not a replacement.
- **AC-6** — No reader finds `claude/AGENTS.md` (or any other doc describing the standing kaizen
  convention) silent about or contradicting the fact that `graph-dba`'s inbox is now also mirrored
  into a graph (FR-9 executed).
- **AC-7** — Given the generic tool, when `cobb` writes one ad-hoc fact directly into the graph
  (no markdown file behind it) via the tool, the fact is stored and queryable, attributed to
  `cobb`.
- **AC-8** — Given the trusted self-identification model (FR-7), when a tool call claims to be
  `graph-dba` and attempts to write an entry attributed to `cobb` (or vice versa), the system
  rejects it or otherwise does not accept it as authored by the claimed-against agent — enforced
  at the "well-behaved callers can't do this by accident" level, not hardened against a malicious
  caller.
- **AC-9** — `docs/requirements/cpg-query-access.md`'s "Non-CPG graphs / general agent access to
  FalkorDB" out-of-scope line is updated to point at this document; no reader finds the two
  disagreeing about blast radius (FR-1's supersession, executed).

## Open questions
*(none)*

## Decision log
- 2026-08-17 — Session opened. Stakeholder asked "how can I turn my cpg mcp into a generic
  cypher mcp" (Mode 2 question). `tico` grounded the answer in `cpg/mcp/README.md` and
  `docs/requirements/cpg-query-access.md`, and flagged that this reverses that document's
  recorded out-of-scope decision on non-CPG graph access. Stakeholder confirmed: open a
  requirements interview rather than answering the "how". This document tracks it.
- 2026-08-17 — Trigger? → Not the CPG tool itself: "I want the agents to use it as collective
  memory and project management."
- 2026-08-17 — Which agent(s)? → All of them, conceptually — a shared knowledge plane, not one
  new consumer.
- 2026-08-17 — Relationship to falkor-chat? → **Actually integrated** — a fact/decision an agent
  records should be traversable from falkor-chat's own data, not just reachable via the same tool.
  Later confirmed **deferred past this delivery** (see Out of scope).
- 2026-08-17 — What "identification" needs to cover? → Attribution, human-vs-agent distinction,
  and real access control (all three selected, not just one).
- 2026-08-17 — Scope shape (full vision vs. a slice)? → "Not sure — keep talking through the
  pieces," which is how the interview proceeded (thread by thread) rather than deciding phasing
  up front.
- 2026-08-17 — What kinds of things belong in the knowledge plane? → Learnings/gotchas, task/
  backlog state, cross-agent working memory, **and** eventually design/requirements documents
  themselves (the last one explicitly "not in scope for this first version").
- 2026-08-17 — Given the blast radius (every agent's kaizen files, every module's `BACKLOG.md`,
  the whole doc-lifecycle convention), how much does v1 migrate? → **One pilot slice first**, not
  a full-team cutover and not a docs-only-going-forward approach.
- 2026-08-17 — Which pilot slice? → Talked through the trade-off (kaizen inbox = lower risk,
  proves mechanics; `BACKLOG.md` = closer to the actual PM motivation but load-bearing for live
  coordination today). Stakeholder chose **kaizen inbox first**.
- 2026-08-17 — Which agent's kaizen inbox? → **`graph-dba`** (46 lines, moderate churn, and the
  agent that owns the FalkorDB/Cypher domain itself — a natural dogfooding choice).
- 2026-08-17 — Who can read `graph-dba`'s graph-backed entries? → **Any agent, directly** — this
  is the actual point of "collective memory," not deferred.
- 2026-08-17 — Is falkor-chat integration part of this delivery's acceptance criteria? →
  **Deferred past the pilot.**
- 2026-08-17 — What does "human vs. agent distinction" concretely mean? → The stakeholder wants to
  read/write the knowledge plane directly themselves, distinguishably from an agent's entries.
  Confirmed **later phase** — monitoring today is already covered by FalkorDB's existing web
  console (`http://localhost:3000`, from `falkor-chat/scripts/start_falkordb.sh`).
- 2026-08-17 — **Reconsidered mid-interview**: FalkorDB is memory-hungry, so don't make the graph
  authoritative for `graph-dba`'s kaizen learnings — **keep the markdown file** and use the graph
  in parallel for what it's best at (graph/GraphRAG indexing). This **reverses** the earlier
  "graph becomes the source of truth" decision and the earlier "delete `inbox.md` after migration"
  decision (both struck from this document; superseded by the entries below).
- 2026-08-17 — What does "GraphRAG" mean for this pilot — semantic similarity search, or
  structured traversal? → **Both, eventually; not sure which matters more for the pilot.**
  Resolved as: structural traversal is required for "done" (FR-5), semantic/vector search is an
  explicit stretch goal, not a blocker.
- 2026-08-17 — How fresh must the mirror be relative to `inbox.md`? → **Kept in sync
  automatically** — no manual rebuild step, reflected in FR-4/AC-3/AC-4 (including that a
  `cobb`-cleared entry disappears from the graph too).
- 2026-08-17 — Does the agent-facing MCP tool need write capability, given writing is now
  out-of-band indexing? → **Yes** — a concrete, different reason: agents need to write knowledge
  that has **no markdown file behind it at all** (future cross-agent working memory), which
  out-of-band indexing of a file can't cover.
- 2026-08-17 — Should this delivery prove that write path, or just record the requirement for
  later? → **Prove it now**, with a minimal toy write (FR-6/AC-7), not a real feature.
- 2026-08-17 — Which agent exercises the write proof? → **`cobb`** — chosen specifically because
  using `graph-dba` again wouldn't demonstrate multi-agent attribution (one agent's write being
  correctly distinguished from another's), which is part of what FR-7/FR-8 need to prove.
- 2026-08-17 — How strictly must write attribution be enforced? → **Trusted self-identification**
  — the same no-auth trust level the rest of the repo runs at today, not a hardened
  access-control boundary (FR-7).
