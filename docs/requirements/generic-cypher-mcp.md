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

The storage model settled, after two rounds of reconsideration, on a split that maps onto
`graph-dba`'s **existing** two-tier kaizen structure (raw capture vs. permanent record):

- The **graph becomes the working-memory / raw-capture layer** — where `graph-dba` writes new
  learnings going forward, replacing `inbox.md` as the write target. This is what a graph is
  good at (queryable, no accumulation problem because entries clear once promoted) and avoids
  making FalkorDB — which the stakeholder flagged as memory-hungry — the permanent home for
  ever-growing history.
- **`history.md` stays exactly as it is today** — markdown, permanent, git-tracked. When `cobb`
  distills and promotes a raw entry, it still appends to `history.md` "like it is nowadays." Only
  the transient/raw layer moves to the graph; the durable record does not.
- **`inbox.md` is frozen** after a one-time import of its current content into the graph — kept in
  the repo as a historical snapshot, but no longer written to.

This document scopes the **first concrete, buildable step** toward the larger vision: proving this
pattern on one narrow, low-stakes slice (`graph-dba`'s kaizen working memory) before deciding
whether/how to extend it to the rest of the team, to task/backlog state, to falkor-chat's data, or
to documents themselves — all real, all named by the stakeholder as where this is ultimately
going, and all deliberately **not** committed to by this document. See Out of scope.

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
tool."* This feature reopens that too: it deliberately asks for a write path — but a narrow,
attributed one (an agent creating its own entries; a curator clearing entries it doesn't own, as
part of a recognized role), not a wholesale return to an unrestricted-write server.

Separately, `claude/AGENTS.md` records the standing convention that every agent's `kaizen/inbox.md`
is an append-only markdown file, distilled periodically by `cobb` (verify → route → log in
`history.md` → clear). This feature changes that convention **for `graph-dba` specifically**: the
raw-capture step moves from "append to `inbox.md`" to "write to the graph"; the distillation
step's destination (`history.md`) and cadence are unchanged; only where the *raw* material lives
and how it's cleared changes.

## User stories
- As **any agent**, I want to query `graph-dba`'s working-memory graph directly, so that I benefit
  from what it discovered — even before `cobb` has distilled it — without waiting for that to
  happen.
- As **`graph-dba`**, I want to record a new learning by writing it into the shared graph instead
  of appending to a markdown file, so it's immediately queryable by the rest of the team.
- As **`cobb`**, I want my existing distillation workflow to keep working against the graph
  instead of a markdown file — verify a raw entry, route it, append the promotion to `history.md`
  exactly like today, then clear the entry from the graph — so my role doesn't change, only where
  the raw material lives.
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
  limited to `cpg_*` graphs, and able to both read and write. This **supersedes** the "Non-CPG
  graphs / general agent access to FalkorDB" out-of-scope line in
  `docs/requirements/cpg-query-access.md`, which must be updated to point here (mirrors how that
  document's own FR-6 superseded `joern-cpg-pipeline.md` FR-9).
- **FR-2** — Going forward, `graph-dba` writes a new raw learning **directly into the graph**
  through the generic tool, attributed to itself — this replaces appending to `inbox.md`.
- **FR-3** — The content currently in `claude/graph-dba/kaizen/inbox.md` (as of this document's
  writing) is imported into the graph **once**, as its starting state, so nothing already captured
  is lost in the cutover.
- **FR-4** — After that one-time import, `claude/graph-dba/kaizen/inbox.md` is **kept, frozen**, as
  a historical snapshot — no longer written to, and clearly distinguishable (to a reader) from a
  live, current file.
- **FR-5** — `claude/graph-dba/kaizen/history.md` is **unchanged**: when `cobb` distills and
  promotes a raw entry, the promotion is still appended there, in the same format, at the same
  cadence, as today.
- **FR-6** — Any agent can read `graph-dba`'s working-memory graph directly — including entries
  `cobb` hasn't distilled yet — not gated behind the distillation pass. This is the delivery's
  core "collective memory" payoff.
- **FR-7** — A working-memory entry captures what today's markdown entry does: the dated fact, its
  evidence, the context it surfaced in, and a suggested home — queryable via graph traversal at
  minimum. Semantic/similarity search (true GraphRAG, vector-indexed) over entries is an explicit
  **stretch goal**: wanted eventually, not required for this delivery to be done.
- **FR-8** — Write/modify access on the graph has two distinct shapes, not one blanket
  "can write":
  - **Author** — an agent creates new entries attributed to itself only. One agent's write cannot
    be accepted as another's (e.g. a call claiming to be `graph-dba` cannot create an entry that
    reads as authored by `cobb`, or vice versa).
  - **Curator** — `cobb`, in its existing distillation role, can clear or mark-as-promoted an
    entry **authored by another agent** (starting with `graph-dba`) — a distinct, narrower
    capability than general write access, exercised specifically by running the real distillation
    workflow (FR-5/FR-9), not a standing "cobb can write anything" grant.
  Both run on **trusted, self-reported agent identity** — the same no-auth trust level the rest of
  this repo already runs at (FalkorDB itself has no auth on `:6379`) — enforced at the
  "well-behaved callers can't do this by accident" level, not hardened against a malicious caller.
- **FR-9** — `cobb`'s distillation workflow (`agent-maintenance` skill §5) continues to function
  end to end against the graph: verify a raw entry → route it → append the promotion to
  `history.md` (FR-5) → clear the entry from the graph (FR-8's curator capability).
- **FR-10** — The identification/attribution mechanism (FR-8) must leave room for a future
  human-authored entry to be distinguished the same way — the stakeholder's own read/write access
  is a later phase (see Out of scope), but FR-8 should not have to be redone to add it.
- **FR-11** — Every doc describing the standing kaizen-inbox convention (starting with
  `claude/AGENTS.md`) is updated to describe `graph-dba`'s **actual** behavior accurately: raw
  capture now targets the graph, `inbox.md` is frozen history, `history.md` is unchanged.

*Context for the architect (not a requirement):* how the **author**/**curator** distinction in
FR-8 is technically enforced (a permission model, a tool-level check, something else) is a design
decision, not specified here. So is how "frozen, clearly distinguishable" (FR-4) is signaled on
`inbox.md` — a header note, or something else. Sizing/memory-footprint implications of running
FalkorDB for this purpose (the stakeholder's stated worry that shaped the working-memory-vs-
permanent-record split) are also an architecture concern, not quantified by the stakeholder here.

## Out of scope
- **falkor-chat data integration** — actually linking knowledge-plane entries to chat
  threads/messages (traversable from one to the other) is the stakeholder's stated end goal, but
  is a later phase, not part of this delivery.
- **Design/requirements/plan documents becoming graph data** — the long-term ambition behind this
  whole initiative ("ultimately... the design and requirements to be abstracted into the knowledge
  graph"), explicitly deferred by the stakeholder.
- **Moving any other agent's kaizen inbox to the graph.** This delivery is `graph-dba` only.
  Extending to the rest of the team is a follow-on decision made after this one is evaluated.
- **A general cross-agent-working-memory feature** — this delivery builds one specific, narrow
  instance of graph-backed working memory (`graph-dba`'s kaizen capture). A broader feature where
  *any* agent records ad-hoc, non-kaizen facts for another agent to pick up is real and wanted,
  but is a separate, later effort.
- **Migrating any module's `docs/BACKLOG.md` / task-backlog state to the graph** — the
  "project-management" half of the original motivation. Deferred in favor of proving the pattern
  first, precisely because `BACKLOG.md` is load-bearing for live, in-flight coordination today
  (`teco` reads it, other docs cite its K-IDs) and a mistake there is costlier than one in an
  append-only, raw-capture inbox.
- **The stakeholder's own direct read/write access to the knowledge plane** through the
  agent-facing mechanism — deferred to a later phase. Monitoring in the meantime needs no new
  work: FalkorDB's existing web console (`http://localhost:3000`, published by
  `falkor-chat/scripts/start_falkordb.sh`) already shows graph state.
- **Guaranteed semantic/similarity search.** Genuine vector-indexed GraphRAG retrieval is wanted
  eventually and may fall out of this delivery as a stretch goal, but structural traversal query
  is what "done" requires (FR-7).
- **Hardened/cryptographic access control.** The pilot runs at the same trusted
  self-identification level the rest of the repo runs at today — see FR-8.
- **Deleting `claude/graph-dba/kaizen/inbox.md`.** It stays, git-tracked, frozen — not removed
  and not kept live/writeable.

## Acceptance criteria
- **AC-1** — Given the working-memory graph is live, when an agent other than `graph-dba` queries
  it, it gets back entries with the same fields today's markdown format carries (date, fact,
  evidence, context, suggested home), via graph traversal — including entries `cobb` has not yet
  distilled.
- **AC-2** — Given the entries in `claude/graph-dba/kaizen/inbox.md` as of this document's
  writing, when the one-time import runs, every one of them is present in the graph afterward.
- **AC-3** — After that import, `claude/graph-dba/kaizen/inbox.md` still exists in the repo,
  unchanged in content, and a reader can tell it is no longer live (FR-4).
- **AC-4** — Given `graph-dba` discovers a new learning, when it records it, the entry appears in
  the graph (not `inbox.md`), attributed to `graph-dba`, immediately queryable by another agent —
  no separate sync/rebuild step, because there is no second copy to keep in sync.
- **AC-5** — Given a raw entry in the graph, when `cobb` runs its distillation workflow, the
  promotion is appended to `claude/graph-dba/kaizen/history.md` in the existing format, and the
  entry is then cleared from the graph — proving both FR-5 (history unchanged) and FR-8's curator
  capability (a second agent modifying entries it doesn't author, in a recognized role) in one
  real workflow, not a toy.
- **AC-6** — Given the trusted self-identification model (FR-8), when a tool call claims to be
  `graph-dba` and attempts to create an entry attributed to `cobb` (or vice versa, outside the
  curator action), the system rejects it or otherwise does not accept it as authored by the
  claimed-against agent.
- **AC-7** — No reader finds `claude/AGENTS.md` (or any other doc describing the standing kaizen
  convention) silent about or contradicting `graph-dba`'s actual behavior (FR-11 executed).
- **AC-8** — `docs/requirements/cpg-query-access.md`'s "Non-CPG graphs / general agent access to
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
  **Deferred past this delivery.**
- 2026-08-17 — What does "human vs. agent distinction" concretely mean? → The stakeholder wants to
  read/write the knowledge plane directly themselves, distinguishably from an agent's entries.
  Confirmed **later phase** — monitoring today is already covered by FalkorDB's existing web
  console (`http://localhost:3000`, from `falkor-chat/scripts/start_falkordb.sh`).
- 2026-08-17 — **Reconsidered (round 1)**: FalkorDB is memory-hungry, so don't make the graph
  authoritative for `graph-dba`'s kaizen learnings — keep the markdown file and use the graph in
  parallel as an index/GraphRAG layer over it. (Superseded by round 2 below.)
- 2026-08-17 — What does "GraphRAG" mean for this delivery — semantic similarity search, or
  structured traversal? → **Both, eventually; not sure which matters more for the pilot.**
  Resolved as: structural traversal is required for "done" (FR-7), semantic/vector search is an
  explicit stretch goal, not a blocker. (This resolution survives round 2, below, unchanged.)
- 2026-08-17 — Does the agent-facing MCP tool need write capability, given writing was (at that
  point) modeled as out-of-band indexing? → **Yes** — agents need to write knowledge that has
  **no markdown file behind it at all** (the model that round 2, below, made literal for
  `graph-dba` itself).
- 2026-08-17 — **Reconsidered (round 2 — current model)**: stakeholder clarified the actual intent
  was fuller than an index: **import existing items, use the graph as working memory, and once
  knowledge is promoted/handled append it to `history.md` "like it is nowadays."** This maps onto
  `graph-dba`'s existing two-tier kaizen structure (raw capture vs. permanent record) and
  **supersedes round 1's mirror/index model**: the graph is now the write target for new raw
  entries (not a synced copy of a still-authoritative `inbox.md`), while `history.md` — the
  permanent, distilled record — stays markdown, exactly as today.
- 2026-08-17 — What happens to `inbox.md` once the graph is the working-memory write target? →
  **Kept, frozen, as a historical snapshot** — not deleted, not left live.
- 2026-08-17 — Does `cobb`'s curator-role clearing of a distilled entry (a write on `graph-dba`'s
  own data, in a recognized role) satisfy the earlier "prove multi-agent attribution" goal, or is
  a separate toy write (an unrelated fact attributed to `cobb` itself) still needed? → **The
  curator clearing is enough** — it's real, ongoing behavior (FR-9/AC-5), not a toy, so the
  earlier standalone toy-write proof was dropped as redundant.