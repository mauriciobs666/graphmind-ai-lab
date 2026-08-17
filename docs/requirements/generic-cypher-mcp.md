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

This document scopes the **first concrete, buildable step** toward that: proving the pattern —
"a shared graph is the source of truth, one agent writes, any agent reads, entries are
attributed" — on one narrow, low-stakes slice (`graph-dba`'s kaizen learnings inbox) before
deciding whether/how to extend it to the rest of the team, to task/backlog state, to falkor-chat's
data, or to documents themselves. Those larger extensions are real and named by the stakeholder as
where this is ultimately going, but are deliberately **not** what this document commits to
delivering — see Out of scope.

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
tool."* This feature reopens that too: it deliberately asks for a **write** path, which the
existing tool's read-only guarantee was built specifically to prevent.

Separately, `claude/AGENTS.md` records the standing convention that every agent's `kaizen/inbox.md`
is an append-only markdown file, distilled periodically by `cobb` (verify → route → log in
`history.md` → clear) — the only reader across agents today is `cobb`, during that distillation
pass. This feature changes that for one agent, on a pilot basis: `graph-dba`'s raw, undistilled
learnings become directly queryable by any other agent.

## User stories
- As **any agent**, I want to query `graph-dba`'s captured learnings directly, so that I benefit
  from what it discovered without waiting for `cobb` to distill and promote it into a prompt or
  doc.
- As **`graph-dba`**, I want to record a new learning the same way I do today (a dated fact with
  evidence, context, and a suggested home) but into the shared graph instead of my private
  markdown file, so it becomes immediately queryable by the rest of the team.
- As the **stakeholder**, I want to eventually read and write the knowledge plane myself, with my
  entries distinguishable from an agent's — not required for this pilot, since I can already watch
  graph state today via FalkorDB's existing web console.
- As the **team**, we want one generic, graph-agnostic Cypher-capable MCP tool as the access
  mechanism — not scoped to CPG graphs — so this pattern can extend to future knowledge without a
  new tool being invented each time.

## Functional requirements
- **FR-1** — Agents reach the knowledge plane through a **generic, graph-name-agnostic** Cypher
  MCP tool — the same two-parameter shape as today's `mcp__cpg__query(graph, cypher)`, but not
  limited to `cpg_*` graphs. This **supersedes** the "Non-CPG graphs / general agent access to
  FalkorDB" out-of-scope line in `docs/requirements/cpg-query-access.md`, which must be updated to
  point here (mirrors how that document's own FR-6 superseded `joern-cpg-pipeline.md` FR-9).
- **FR-2** — Reading `graph-dba`'s kaizen-learnings graph is open to **every agent, directly** —
  not gated behind `cobb`'s distillation pass.
- **FR-3** — Writing a kaizen-learnings entry is restricted to the entry's own agent — only
  `graph-dba` writes entries attributed to `graph-dba`. Enforcement is via **trusted,
  self-reported agent identity**: the same no-auth trust level the rest of this repo already runs
  at (FalkorDB itself has no auth on `:6379`), not a hardened/cryptographic access-control
  boundary.
- **FR-4** — Every entry records which agent authored it, queryably. The identification mechanism
  must leave room for a future human-authored entry to be distinguished from an agent's the same
  way (see Out of scope — the stakeholder's own read/write access is a later phase, but FR-4
  should not have to be redone to add it).
- **FR-5** — A kaizen-learnings entry in the graph captures what today's markdown entry does: the
  dated fact, its evidence, the context it surfaced in, and a suggested home
  (prompt/knowledge-base/project-docs/unsure) — see the format documented at the top of
  `claude/graph-dba/kaizen/inbox.md`.
- **FR-6** — All entries currently in `claude/graph-dba/kaizen/inbox.md` are migrated into the
  graph as its starting state; nothing captured today is lost in the cutover.
- **FR-7** — `claude/graph-dba/kaizen/inbox.md` is **deleted** once migration is verified — the
  graph becomes the only copy, not a mirror kept alongside a frozen file.
- **FR-8** — Every doc describing "every agent has a `kaizen/inbox.md`" as standing convention
  (starting with `claude/AGENTS.md`, and anywhere else that turns out to say the same thing) is
  updated so the documented convention and `graph-dba`'s actual behavior don't disagree.
- **FR-9** — `cobb`'s distillation workflow (`agent-maintenance` skill §5: verify → route → log in
  `history.md` → clear) continues to function for `graph-dba`'s learnings, now operating against
  the graph instead of the markdown file.

*Context for the architect (not a requirement):* whether writes travel through the **same**
universal tool (with an authorization layer keyed on the trusted self-reported identity) or a
**separate, narrower write path** — the way the CPG's own *build*/load side already stays apart
from its read side — is a real design trade-off. The stakeholder does not have a preference; flag
it as a decision to make, not something settled here.

## Out of scope
- **falkor-chat data integration** — actually linking knowledge-plane entries to chat
  threads/messages (traversable from one to the other) is the stakeholder's stated end goal, but
  is a later phase, not part of this delivery.
- **Design/requirements/plan documents becoming graph data** — the long-term ambition behind this
  whole initiative ("ultimately... the design and requirements to be abstracted into the knowledge
  graph"), explicitly deferred by the stakeholder.
- **Migrating any other agent's kaizen inbox.** This pilot is `graph-dba` only. Extending to the
  rest of the team is a follow-on decision made after this pilot is evaluated.
- **Migrating any module's `docs/BACKLOG.md` / task-backlog state to the graph** — the
  "project-management" half of the original motivation. Deferred in favor of proving the pattern
  on learnings first, precisely because `BACKLOG.md` is load-bearing for live, in-flight
  coordination today (`teco` reads it, other docs cite its K-IDs) and a mistake there is costlier
  than one in an append-only, raw-capture inbox.
- **The stakeholder's own direct read/write access to the knowledge plane** through the
  agent-facing mechanism — deferred to a later phase. Monitoring in the meantime needs no new
  work: FalkorDB's existing web console (`http://localhost:3000`, published by
  `falkor-chat/scripts/start_falkordb.sh`) already shows graph state.
- **Hardened/cryptographic access control.** The pilot runs at the same trusted
  self-identification level the rest of the repo runs at today — see FR-3.

## Acceptance criteria
- **AC-1** — Given the pilot is live, when an agent other than `graph-dba` queries the knowledge
  plane for `graph-dba`'s learnings via the generic MCP tool, it gets back entries with the same
  fields today's format carries (date, fact, evidence, context, suggested home), with no
  `cobb`-distillation step in between.
- **AC-2** — Given `graph-dba` discovers a new learning, when it records it, the entry lands in the
  graph (not appended to `inbox.md`), attributed to `graph-dba`, in the FR-5 shape.
- **AC-3** — Given the trusted self-identification model (FR-3), when a tool call claims an agent
  identity other than `graph-dba` and attempts to write an entry attributed to `graph-dba`, the
  system rejects or otherwise does not accept it as a `graph-dba` entry — enforced at the level of
  "well-behaved callers can't do this by accident," not hardened against a malicious caller.
- **AC-4** — Given the entries in `claude/graph-dba/kaizen/inbox.md` as of this document's
  writing, when migration runs, every one of them is present in the graph afterward, and
  `inbox.md` no longer exists in the repo.
- **AC-5** — No reader finds `claude/AGENTS.md` (or any other doc describing the standing kaizen
  convention) disagreeing with `graph-dba`'s actual state — FR-8 executed, not left as a
  contradiction.
- **AC-6** — `cobb` can still run its distillation workflow end to end (verify an entry → route it
  → log the promotion in `graph-dba/kaizen/history.md` → clear it from the graph) against
  `graph-dba`'s graph-backed learnings.
- **AC-7** — `docs/requirements/cpg-query-access.md`'s "Non-CPG graphs / general agent access to
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
  Later confirmed **deferred past this pilot** (see Out of scope).
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
- 2026-08-17 — Who can write them? → **Only `graph-dba` itself**, for its own entries.
- 2026-08-17 — Should writing route through the same universal tool or a separate mechanism? →
  **Not sure — flagged for the architect** as a real design trade-off, not a stakeholder
  preference. Recorded as context for the architect, not an FR or an open question blocking this
  document.
- 2026-08-17 — Is falkor-chat integration part of this pilot's acceptance criteria? →
  **Deferred past the pilot.**
- 2026-08-17 — What does "human vs. agent distinction" concretely mean? → The stakeholder wants to
  read/write the knowledge plane directly themselves, distinguishably from an agent's entries.
- 2026-08-17 — Does the pilot need the stakeholder as a write exception now? → **Later phase.**
  Monitoring today is already covered by FalkorDB's existing web console
  (`http://localhost:3000`, from `falkor-chat/scripts/start_falkordb.sh`) — no new work needed for
  that.
- 2026-08-17 — What happens to the 46 lines of existing content in `graph-dba`'s inbox today? →
  **Migrated into the graph** as its starting state, not left behind.
- 2026-08-17 — What happens to `claude/graph-dba/kaizen/inbox.md` after migration? → **Deleted** —
  the graph is the only copy, not a frozen duplicate kept alongside it.
- 2026-08-17 — How strictly must "only `graph-dba` writes `graph-dba`'s entries" be enforced? →
  **Trusted self-identification** — the same no-auth trust level the rest of the repo runs at
  today, not a hardened access-control boundary.
